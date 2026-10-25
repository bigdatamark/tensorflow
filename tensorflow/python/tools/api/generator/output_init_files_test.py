# [MODULE DOC]
# Test module checking that the build-rule lists of generated API init files (the v2 and v1
# lists in the build definition files next to it) exactly match the modules implied by the
# export attributes of loaded TensorFlow symbols. Contains private helpers that collect API
# modules from loaded modules, parse the generated-file list out of a build file, and expand a
# module into the init file paths it requires, plus one test class. When run as a script it
# parses a comma-separated packages argument, imports those packages to populate the loaded
# module table, trims its arguments from argv and starts the test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Tests for api_init_files.bzl and api_init_files_v1.bzl."""
import argparse
import importlib
import sys

from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.util import tf_decorator


def _traverse_packages(packages):
  # [FUNCTION DOC]
  # HELPER: setup for the init-file tests.
  # Imports every named package so their submodules (and exported symbols) are present in the
  # loaded module table. Input: iterable of package names. Raises an import error for a missing
  # package (an empty name also fails). Returns nothing.
  for package in packages:
    importlib.import_module(package)


def _get_module_from_symbol(symbol):
  # [FUNCTION DOC]
  # HELPER: name splitting utility for the module collection helper.
  # Given a dotted public API name, returns the part before the last dot, or an empty string
  # when there is no dot (root module). Pure function.
  if '.' not in symbol:
    return ''
  return '.'.join(symbol.split('.')[:-1])


def _get_modules(package, attr_name, constants_attr_name):
  # [FUNCTION DOC]
  # HELPER: gathers the expected API modules for the init-file tests.
  # Scans loaded modules whose names contain the package text; for every attribute (decorators
  # unwrapped) either reads the constants attribute (pairs of export names and constant names)
  # or the per-symbol names attribute, converts every exported dotted name to its module part,
  # and returns the set of those module names (the root is an empty string). Inputs: package
  # substring, name of the symbol attribute holding API names, name of the module attribute
  # holding constants. Side effect: reads attributes of all matching loaded modules.
  """Get list of TF API modules.

  Args:
    package: We only look at modules that contain package in the name.
    attr_name: Attribute set on TF symbols that contains API names.
    constants_attr_name: Attribute set on TF modules that contains
      API constant names.

  Returns:
    Set of TensorFlow API modules.
  """
  modules = set()
  # TODO(annarev): split up the logic in create_python_api.py so that
  #   it can be reused in this test.
  for module in list(sys.modules.values()):
    if (not module or not hasattr(module, '__name__') or
        package not in module.__name__):
      continue

    for module_contents_name in dir(module):
      attr = getattr(module, module_contents_name)
      _, attr = tf_decorator.unwrap(attr)

      # Add modules to _tf_api_constants attribute.
      if module_contents_name == constants_attr_name:
        for exports, _ in attr:
          modules.update(
              [_get_module_from_symbol(export) for export in exports])
        continue

      # Add modules for _tf_api_names attribute.
      if (hasattr(attr, '__dict__') and attr_name in attr.__dict__):
        modules.update([
            _get_module_from_symbol(export)
            for export in getattr(attr, attr_name)])
  return modules


def _get_files_set(path, start_tag, end_tag):
  # [FUNCTION DOC]
  # HELPER: build-file parser for the init-file tests.
  # KNOWN ISSUE: the docstring says it returns a list, but it returns a set of strings.
  # Reads a text file, takes the text between the start tag (plus one following character) and
  # the end tag, splits it by commas, strips whitespace and surrounding double quotes and drops
  # empty entries. Inputs: file path, start tag text, end tag text. Raises an error if the file
  # cannot be opened; missing tags are not detected (a failed find yields minus one and the
  # slice is then wrong). Returns the set of path strings.
  """Get set of file paths from the given file.

  Args:
    path: Path to file. File at `path` is expected to contain a list of paths
      where entire list starts with `start_tag` and ends with `end_tag`. List
      must be comma-separated and each path entry must be surrounded by double
      quotes.
    start_tag: String that indicates start of path list.
    end_tag: String that indicates end of path list.

  Returns:
    List of string paths.
  """
  with open(path, 'r') as f:
    contents = f.read()
    start = contents.find(start_tag) + len(start_tag) + 1
    end = contents.find(end_tag)
    contents = contents[start:end]
    file_paths = [
        file_path.strip().strip('"') for file_path in contents.split(',')]
    return set(file_path for file_path in file_paths if file_path)


def _module_to_paths(module):
  # [FUNCTION DOC]
  # HELPER: expands a module into required init file paths for the init-file tests.
  # Input: dotted module name. For every prefix of the dotted name returns an init file path with
  # dots replaced by slashes; the empty module (root) gives the top-level init file only. Note
  # the top-level init file is not added for non-root modules. Returns a list of strings.
  """Get all API __init__.py file paths for the given module.

  Args:
    module: Module to get file paths for.

  Returns:
    List of paths for the given module. For e.g. module foo.bar
    requires 'foo/__init__.py' and 'foo/bar/__init__.py'.
  """
  submodules = []
  module_segments = module.split('.')
  for i in range(len(module_segments)):
    submodules.append('.'.join(module_segments[:i+1]))
  paths = []
  for submodule in submodules:
    if not submodule:
      paths.append('__init__.py')
      continue
    paths.append('%s/__init__.py' % (submodule.replace('.', '/')))
  return paths


class OutputInitFilesTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that the file lists in the two build definition files (v2 and v1)
  # match the API modules found from export attributes. Has a shared validation helper and one
  # test per API version.
  """Test that verifies files that list paths for TensorFlow API."""

  def _validate_paths_for_modules(
      self, actual_paths, expected_paths, file_to_update_on_error):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the version tests.
    # Inputs: sets of listed paths and expected paths, and the name of the file to mention in
    # error messages. Asserts both sets are non-empty, then that no expected path is missing and
    # no listed path is redundant; failure messages list the quoted offending paths sorted, so
    # they can be pasted into the file. Raises assertion failures; returns nothing.
    """Validates that actual_paths match expected_paths.

    Args:
      actual_paths: */__init__.py file paths listed in file_to_update_on_error.
      expected_paths: */__init__.py file paths that we need to create for
        TensorFlow API.
      file_to_update_on_error: File that contains list of */__init__.py files.
        We include it in error message printed if the file list needs to be
        updated.
    """
    self.assertTrue(actual_paths)
    self.assertTrue(expected_paths)
    missing_paths = expected_paths - actual_paths
    extra_paths = actual_paths - expected_paths

    # Surround paths with quotes so that they can be copy-pasted
    # from error messages as strings.
    missing_paths = ['\'%s\'' % path for path in missing_paths]
    extra_paths = ['\'%s\'' % path for path in extra_paths]

    self.assertFalse(
        missing_paths,
        'Please add %s to %s.' % (
            ',\n'.join(sorted(missing_paths)), file_to_update_on_error))
    self.assertFalse(
        extra_paths,
        'Redundant paths, please remove %s in %s.' % (
            ',\n'.join(sorted(extra_paths)), file_to_update_on_error))

  def test_V2_init_files(self):
    # [FUNCTION DOC]
    # Scenario: collect modules from the v2 export attributes, read the generated file list from
    # the v2 build definition file between the begin and end markers, and assert that the listed
    # init files equal the files required by the modules.
    modules = _get_modules(
        'tensorflow', '_tf_api_names', '_tf_api_constants')
    file_path = resource_loader.get_path_to_datafile(
        'api_init_files.bzl')
    paths = _get_files_set(
        file_path, '# BEGIN GENERATED FILES', '# END GENERATED FILES')
    module_paths = set(
        f for module in modules for f in _module_to_paths(module))
    self._validate_paths_for_modules(
        paths, module_paths, file_to_update_on_error=file_path)

  def test_V1_init_files(self):
    # [FUNCTION DOC]
    # Scenario: same as the v2 test but using the v1 export attribute names and the v1 build
    # definition file; asserts the listed init files equal the files required by the modules.
    modules = _get_modules(
        'tensorflow', '_tf_api_names_v1', '_tf_api_constants_v1')
    file_path = resource_loader.get_path_to_datafile(
        'api_init_files_v1.bzl')
    paths = _get_files_set(
        file_path, '# BEGIN GENERATED FILES', '# END GENERATED FILES')
    module_paths = set(
        f for module in modules for f in _module_to_paths(module))
    self._validate_paths_for_modules(
        paths, module_paths, file_to_update_on_error=file_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--packages',
      type=str,
      default='',
      help='Comma separated list of packages to traverse.')
  FLAGS, unparsed = parser.parse_known_args()

  # Traverse packages that define APIs.
  _traverse_packages(FLAGS.packages.split(','))

  # Now update argv, so that unittest library does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
