# [MODULE DOC]
# KNOWN ISSUE: the command line flags are declared with a boolean conversion on the raw argument
# text, so any non-empty value (including the text false) turns the flag on; only omitting the flag
# or passing an empty value yields false. The helper that builds the ignored-type mapping also has
# an operator precedence problem (see its own doc).
# Public API backwards-compatibility test for TensorFlow. It walks the public symbols of the
# tensorflow package (top level, compat v1 and compat v2), converts every public module, class and
# function into a protocol buffer description, and compares them with golden text-format files
# stored next to the test data (one file per API object, with capital letters in object names
# encoded as a dash plus lowercase letter). Differences fail the test unless the update_goldens
# flag is set, in which case golden files are deleted, created or rewritten. Also verifies that no
# public class subclasses a generated protocol buffer message.
# Command line flags (parsed only when run as a script): update_goldens (default false),
# only_test_core_api (default true; skips keras-related and other non-core packages and
# tolerates missing TensorBoard summary symbols) and verbose_diffs (default true).
# Import-time side effects: imports tensorflow, detects whether TensorBoard is importable, and
# resolves the paths of the README and update-warning data files. The flags object and golden
# folder paths are set only by the script entry point. Not a helper module.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""TensorFlow API compatibility tests.

This test ensures all changes to the public API of TensorFlow are intended.

If this test fails, it means a change has been made to the public API. Backwards
incompatible changes are not allowed. You can run the test with
"--update_goldens" flag set to "True" to update goldens when making changes to
the public TF python API.
"""

import argparse
import os
import re
import sys

import tensorflow as tf

from google.protobuf import message
from google.protobuf import text_format

from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.tools.api.lib import api_objects_pb2
from tensorflow.tools.api.lib import python_object_to_proto_visitor
from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse

# pylint: disable=g-import-not-at-top,unused-import
_TENSORBOARD_AVAILABLE = True
try:
  import tensorboard as _tb
except ImportError:
  _TENSORBOARD_AVAILABLE = False
# pylint: enable=g-import-not-at-top,unused-import

# FLAGS defined at the bottom:
FLAGS = None
# DEFINE_boolean, update_goldens, default False:
_UPDATE_GOLDENS_HELP = """
     Update stored golden files if API is updated. WARNING: All API changes
     have to be authorized by TensorFlow leads.
"""

# DEFINE_boolean, only_test_core_api, default False:
_ONLY_TEST_CORE_API_HELP = """
    Some TF APIs are being moved outside of the tensorflow/ directory. There is
    no guarantee which versions of these APIs will be present when running this
    test. Therefore, do not error out on API changes in non-core TF code
    if this flag is set.
"""

# DEFINE_boolean, verbose_diffs, default True:
_VERBOSE_DIFFS_HELP = """
     If set to true, print line by line diffs on all libraries. If set to
     false, only print which libraries have differences.
"""

# Initialized with _InitPathConstants function below.
_API_GOLDEN_FOLDER_V1 = None
_API_GOLDEN_FOLDER_V2 = None


def _InitPathConstants():
  # [FUNCTION DOC]
  # HELPER: sets up the golden-file folder globals used by the API compatibility test.
  # Computes the location of the version 2 root golden file relative to the test data directory
  # and stores the directory containing it as the version 2 golden folder; the version 1 folder
  # is the sibling directory named v1. When the update_goldens flag is set, symbolic links are
  # resolved first so new files are created in the real source tree. Reads the module flags
  # object (must already be parsed) and sets two module-level globals. Takes no arguments,
  # returns nothing.
  global _API_GOLDEN_FOLDER_V1
  global _API_GOLDEN_FOLDER_V2
  root_golden_path_v2 = os.path.join(resource_loader.get_data_files_path(),
                                     '..', 'golden', 'v2', 'tensorflow.pbtxt')

  if FLAGS.update_goldens:
    root_golden_path_v2 = os.path.realpath(root_golden_path_v2)
  # Get API directories based on the root golden file. This way
  # we make sure to resolve symbolic links before creating new files.
  _API_GOLDEN_FOLDER_V2 = os.path.dirname(root_golden_path_v2)
  _API_GOLDEN_FOLDER_V1 = os.path.normpath(
      os.path.join(_API_GOLDEN_FOLDER_V2, '..', 'v1'))


_TEST_README_FILE = resource_loader.get_path_to_datafile('README.txt')
_UPDATE_WARNING_FILE = resource_loader.get_path_to_datafile(
    'API_UPDATE_WARNING.txt')

_NON_CORE_PACKAGES = ['keras']
_V1_APIS_FROM_KERAS = ['layers', 'nn.rnn_cell']
_V2_APIS_FROM_KERAS = ['initializers', 'losses', 'metrics', 'optimizers']


def _KeyToFilePath(key, api_version):
  # [FUNCTION DOC]
  # HELPER: maps API object keys to golden file paths for the compatibility test.
  # Converts a dotted API object name (key) into the path of its golden text-format file:
  # each uppercase letter becomes a dash followed by the lowercase letter, and the extension
  # pbtxt is appended. The folder is the version 2 golden folder when api_version is 2, otherwise
  # the version 1 folder. Keys beginning with the experimental numpy prefix are redirected to a
  # separate numpy API golden folder located several directories up. Requires the golden folder
  # globals to have been initialized. Returns the path as text; no side effects. The wildcard key
  # works too, producing a glob pattern.
  """From a given key, construct a filepath.

  Filepath will be inside golden folder for api_version.

  Args:
    key: a string used to determine the file path
    api_version: a number indicating the tensorflow API version, e.g. 1 or 2.

  Returns:
    A string of file path to the pbtxt file which describes the public API
  """

  def _ReplaceCapsWithDash(matchobj):
    # [FUNCTION DOC]
    # HELPER: regex substitution callback for the key-to-path conversion.
    # Receives a regular expression match of one uppercase letter and returns a dash followed by
    # that letter in lowercase. No side effects.
    match = matchobj.group(0)
    return '-%s' % (match.lower())

  case_insensitive_key = re.sub('([A-Z]{1})', _ReplaceCapsWithDash, key)
  api_folder = (
      _API_GOLDEN_FOLDER_V2 if api_version == 2 else _API_GOLDEN_FOLDER_V1)
  if key.startswith('tensorflow.experimental.numpy'):
    # Jumps up one more level in order to let Copybara find the
    # 'tensorflow/third_party' string to replace
    api_folder = os.path.join(
        api_folder, '..', '..', '..', '..', '../third_party',
        'py', 'numpy', 'tf_numpy_api')
    api_folder = os.path.normpath(api_folder)
  return os.path.join(api_folder, '%s.pbtxt' % case_insensitive_key)


def _FileNameToKey(filename):
  # [FUNCTION DOC]
  # HELPER: inverse of the key-to-path conversion, used when loading golden files.
  # Takes a golden file name or path, strips the directory and extension, and turns every dash
  # followed by a lowercase letter into the uppercase letter, returning the API object key as
  # text. No side effects.
  """From a given filename, construct a key we use for api objects."""

  def _ReplaceDashWithCaps(matchobj):
    # [FUNCTION DOC]
    # HELPER: regex substitution callback for the file-name-to-key conversion.
    # Receives a match of a dash plus a lowercase letter and returns just the uppercase letter.
    # No side effects.
    match = matchobj.group(0)
    return match[1].upper()

  base_filename = os.path.basename(filename)
  base_filename_without_ext = os.path.splitext(base_filename)[0]
  api_object_key = re.sub('((-[a-z]){1})', _ReplaceDashWithCaps,
                          base_filename_without_ext)
  return api_object_key


def _VerifyNoSubclassOfMessageVisitor(path, parent, unused_children):
  # [FUNCTION DOC]
  # HELPER: visitor callback passed to the public API traversal by the no-subclass tests.
  # Inputs: path (dotted name of the visited object without the tf prefix), parent (the visited
  # object) and the list of its children (unused). Does nothing unless the object is a class
  # derived from the generated protocol buffer message base class, and ignores the base class
  # itself. Raises a not-implemented error naming the object when such a class does not list the
  # message base class directly among its bases (i.e. is an indirect subclass). Returns nothing.
  """A Visitor that crashes on subclasses of generated proto classes."""
  # If the traversed object is a proto Message class
  if not (isinstance(parent, type) and issubclass(parent, message.Message)):
    return
  if parent is message.Message:
    return
  # Check that it is a direct subclass of Message.
  if message.Message not in parent.__bases__:
    raise NotImplementedError(
        'Object tf.%s is a subclass of a generated proto Message. '
        'They are not yet supported by the API tools.' % path)


def _FilterNonCoreGoldenFiles(golden_file_list):
  # [FUNCTION DOC]
  # HELPER: removes golden files of non-core packages (currently keras) from a list of file paths.
  # Input: list of golden file paths. Returns a new list without files whose base name starts
  # with the tensorflow-dot-package prefix of any non-core package. Input is not modified.
  """Filter out non-core API pbtxt files."""
  return _FilterGoldenFilesByPrefix(golden_file_list, _NON_CORE_PACKAGES)


def _FilterV1KerasRelatedGoldenFiles(golden_file_list):
  # [FUNCTION DOC]
  # HELPER: removes golden files for the version 1 APIs that come from keras (layers and the nn
  # rnn cell package) from a list of golden file paths. Returns a new filtered list; the input
  # is not modified.
  return _FilterGoldenFilesByPrefix(golden_file_list, _V1_APIS_FROM_KERAS)


def _FilterV2KerasRelatedGoldenFiles(golden_file_list):
  # [FUNCTION DOC]
  # HELPER: removes golden files for the version 2 APIs that come from keras (initializers,
  # losses, metrics and optimizers) from a list of golden file paths. Returns a new filtered
  # list; the input is not modified.
  return _FilterGoldenFilesByPrefix(golden_file_list, _V2_APIS_FROM_KERAS)


def _FilterGoldenFilesByPrefix(golden_file_list, package_prefixes):
  # [FUNCTION DOC]
  # HELPER: shared implementation of the golden file filters.
  # Inputs: list of golden file paths and a list of package names. Each name is turned into the
  # prefix tensorflow, dot, name, dot. Returns a new list that keeps only paths whose final path
  # component (after the last slash) does not start with any of those prefixes. Input is not
  # modified.
  filtered_file_list = []
  filtered_package_prefixes = ['tensorflow.%s.' % p for p in package_prefixes]
  for f in golden_file_list:
    if any(
        f.rsplit('/')[-1].startswith(pre) for pre in filtered_package_prefixes):
      continue
    filtered_file_list.append(f)
  return filtered_file_list


def _FilterGoldenProtoDict(golden_proto_dict, omit_golden_symbols_map):
  # [FUNCTION DOC]
  # HELPER: strips selected symbols from API object descriptions before comparison.
  # KNOWN ISSUE: callers pass a plain text value (not a list) for the is_instance case; the
  # membership test then works as a substring check on that text, which only works by accident.
  # Inputs: mapping from API object keys to API object protocol buffers, and a mapping from
  # keys to collections of symbol names to omit (an empty or null mapping returns the input
  # unchanged). Makes a shallow copy of the mapping and, for each key listed, replaces the entry
  # by a copy of the protocol buffer, then removes from its module or class description every
  # member and member method whose name is listed; if the collection contains the special name
  # is_instance, the recorded is-instance list is cleared as well. Objects that are neither
  # module nor class are left alone. Raises a key error if a listed key is absent from the
  # mapping. Returns the new mapping; the original protocol buffers are not mutated.
  """Filter out golden proto dict symbols that should be omitted."""
  if not omit_golden_symbols_map:
    return golden_proto_dict
  filtered_proto_dict = dict(golden_proto_dict)
  for key, symbol_list in omit_golden_symbols_map.items():
    api_object = api_objects_pb2.TFAPIObject()
    api_object.CopyFrom(filtered_proto_dict[key])
    filtered_proto_dict[key] = api_object
    module_or_class = None
    if api_object.HasField('tf_module'):
      module_or_class = api_object.tf_module
    elif api_object.HasField('tf_class'):
      module_or_class = api_object.tf_class
    if module_or_class is not None:
      if 'is_instance' in symbol_list:
        del module_or_class.is_instance[:]
      for members in (module_or_class.member, module_or_class.member_method):
        filtered_members = [m for m in members if m.name not in symbol_list]
        # Two steps because protobuf repeated fields disallow slice assignment.
        del members[:]
        members.extend(filtered_members)
  return filtered_proto_dict


def _GetTFNumpyGoldenPattern(api_version):
  # [FUNCTION DOC]
  # HELPER: builds the file glob pattern matching the experimental numpy golden files.
  # Input: API version number (1 or 2). Returns the golden path pattern for keys starting with
  # the experimental numpy prefix, joined onto the root directory containing all test resources.
  # No side effects.
  return os.path.join(resource_loader.get_root_dir_with_all_resources(),
                      _KeyToFilePath('tensorflow.experimental.numpy*',
                                     api_version))


class ApiCompatibilityTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking that the public TensorFlow API (top level, compat v1, compat v2) matches
  # the stored golden description files and that no public class subclasses a generated proto
  # message. State: the update-warning text and the test readme text, loaded in the constructor
  # and used for log messages. Behavior depends on module flags (update goldens, core-only,
  # verbose diffs). Run through the standard TensorFlow test runner.

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Forwards all positional and keyword arguments to the base test case, then
    # reads the whole contents of the API update warning file and the test readme file (located
    # under the root resource directory) and stores them as the update-warning and readme message
    # attributes. Raises whatever file reading raises if the data files are missing.
    super(ApiCompatibilityTest, self).__init__(*args, **kwargs)

    golden_update_warning_filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(), _UPDATE_WARNING_FILE)
    self._update_golden_warning = file_io.read_file_to_string(
        golden_update_warning_filename)

    test_readme_filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(), _TEST_README_FILE)
    self._test_readme_message = file_io.read_file_to_string(
        test_readme_filename)

  def _AssertProtoDictEquals(self,
                             expected_dict,
                             actual_dict,
                             verbose=False,
                             update_goldens=False,
                             additional_missing_object_message='',
                             api_version=2):
    # [FUNCTION DOC]
    # HELPER: core comparison step of the backwards-compatibility check.
    # KNOWN ISSUE: the verbose parameter is accepted but never used; the verbose per-object
    # differences are always logged on failure regardless of it.
    # Compares the golden mapping with the actual mapping (both from object key to API object
    # protocol buffer). Keys only in the golden set are reported as removed (with the optional
    # extra message), keys only in the actual set as added, and common keys are compared with the
    # proto equality assertion; a mismatch records the key as updated. If any difference exists,
    # the readme text and a count are logged as errors. With update_goldens true, the update
    # warning is logged, golden files of removed keys are deleted, and files for added or changed
    # keys are written in text format (side effects on disk). Otherwise every difference is
    # logged and the test fails. With no differences an info message is logged. Sets the maximum
    # shown difference length of the test to unlimited. Parameters: expected_dict, actual_dict,
    # verbose (unused), update_goldens (default false), additional_missing_object_message
    # (default empty), api_version (default 2).
    """Diff given dicts of protobufs and report differences a readable way.

    Args:
      expected_dict: a dict of TFAPIObject protos constructed from golden files.
      actual_dict: a dict of TFAPIObject protos constructed by reading from the
        TF package linked to the test.
      verbose: Whether to log the full diffs, or simply report which files were
        different.
      update_goldens: Whether to update goldens when there are diffs found.
      additional_missing_object_message: Message to print when a symbol is
        missing.
      api_version: TensorFlow API version to test.
    """
    diffs = []
    verbose_diffs = []
    expected_keys = set(expected_dict.keys())
    actual_keys = set(actual_dict.keys())
    only_in_expected = expected_keys - actual_keys
    only_in_actual = actual_keys - expected_keys
    all_keys = expected_keys | actual_keys

    # This will be populated below.
    updated_keys = []

    for key in all_keys:
      diff_message = ''
      verbose_diff_message = ''
      # First check if the key is not found in one or the other.
      if key in only_in_expected:
        diff_message = 'Object %s expected but not found (removed). %s' % (
            key, additional_missing_object_message)
        verbose_diff_message = diff_message
      elif key in only_in_actual:
        diff_message = 'New object %s found (added).' % key
        verbose_diff_message = diff_message
      else:
        # Do not truncate diff
        self.maxDiff = None  # pylint: disable=invalid-name
        # Now we can run an actual proto diff.
        try:
          self.assertProtoEquals(expected_dict[key], actual_dict[key])
        except AssertionError as e:
          updated_keys.append(key)
          diff_message = 'Change detected in python object: %s.' % key
          verbose_diff_message = str(e)

      # All difference cases covered above. If any difference found, add to the
      # list.
      if diff_message:
        diffs.append(diff_message)
        verbose_diffs.append(verbose_diff_message)

    # If diffs are found, handle them based on flags.
    if diffs:
      diff_count = len(diffs)
      logging.error(self._test_readme_message)
      logging.error('%d differences found between API and golden.', diff_count)

      if update_goldens:
        # Write files if requested.
        logging.warning(self._update_golden_warning)

        # If the keys are only in expected, some objects are deleted.
        # Remove files.
        for key in only_in_expected:
          filepath = _KeyToFilePath(key, api_version)
          file_io.delete_file(filepath)

        # If the files are only in actual (current library), these are new
        # modules. Write them to files. Also record all updates in files.
        for key in only_in_actual | set(updated_keys):
          filepath = _KeyToFilePath(key, api_version)
          file_io.write_string_to_file(
              filepath, text_format.MessageToString(actual_dict[key]))
      else:
        # Include the actual differences to help debugging.
        for d, verbose_d in zip(diffs, verbose_diffs):
          logging.error('    %s', d)
          logging.error('    %s', verbose_d)
        # Fail if we cannot fix the test by updating goldens.
        self.fail('%d differences found between API and golden.' % diff_count)

    else:
      logging.info('No differences found between API and golden.')

  def testNoSubclassOfMessage(self):
    # [FUNCTION DOC]
    # Traverses the public top-level tensorflow API with a visitor that raises when a public
    # class indirectly subclasses a generated proto message. Excludes the contrib module from
    # descent and skips compat v1 and v2 (checked by separate tests). Takes no inputs; fails by
    # propagating the visitor's error; returns nothing.
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    # visitor.do_not_descend_map['tf'].append('keras')
    # Skip compat.v1 and compat.v2 since they are validated in separate tests.
    visitor.private_map['tf.compat'] = ['v1', 'v2']
    traverse.traverse(tf, visitor)

  def testNoSubclassOfMessageV1(self):
    # [FUNCTION DOC]
    # Same proto-subclass check as the top-level test, but applied to the compat v1 namespace.
    # Returns immediately (test passes trivially) if compat v1 does not exist. Excludes contrib
    # and, when only_test_core_api is set, the non-core packages from descent; hides the compat
    # v1 and v2 entries. Returns nothing; fails via the visitor's error.
    if not hasattr(tf.compat, 'v1'):
      return
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    if FLAGS.only_test_core_api:
      visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
    visitor.private_map['tf.compat'] = ['v1', 'v2']
    traverse.traverse(tf.compat.v1, visitor)

  def testNoSubclassOfMessageV2(self):
    # [FUNCTION DOC]
    # Same proto-subclass check applied to the compat v2 namespace. Returns immediately if compat
    # v2 does not exist. Excludes contrib and, when only_test_core_api is set, the non-core
    # packages from descent. Returns nothing; fails via the visitor's error.
    if not hasattr(tf.compat, 'v2'):
      return
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    if FLAGS.only_test_core_api:
      visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
    visitor.private_map['tf.compat'] = ['v1', 'v2']
    traverse.traverse(tf.compat.v2, visitor)

  def _checkBackwardsCompatibility(self,
                                   root,
                                   golden_file_patterns,
                                   api_version,
                                   additional_private_map=None,
                                   omit_golden_symbols_map=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the three backwards-compatibility tests.
    # Inputs: root (the module object to traverse), golden_file_patterns (a glob pattern or list
    # of patterns for golden files), api_version (1 or 2), additional_private_map (optional
    # mapping from dotted names to hidden member names, merged into the visitor's private map)
    # and omit_golden_symbols_map (optional mapping from object keys to symbols to ignore).
    # Traverses the root with a public-API visitor that converts objects to protocol buffers,
    # hiding contrib (and enable_v2_behavior for version 2), not descending into keras, the GPU
    # options experimental member, and many numpy scalar types; with only_test_core_api also
    # skips non-core and keras-derived packages. Reads and filters the matching golden files into
    # protocol buffers, applies the omit mapping to both golden and actual sets, and delegates
    # to the proto dictionary assertion using the module flags for verbosity and updating.
    # Returns nothing; may rewrite golden files when updating; fails the test on differences.
    # Extract all API stuff.
    visitor = python_object_to_proto_visitor.PythonObjectToProtoVisitor()

    public_api_visitor = public_api.PublicAPIVisitor(visitor)
    public_api_visitor.private_map['tf'].append('contrib')
    if api_version == 2:
      public_api_visitor.private_map['tf'].append('enable_v2_behavior')

    public_api_visitor.do_not_descend_map['tf.GPUOptions'] = ['Experimental']
    # Do not descend into these numpy classes because their signatures may be
    # different between internal and OSS.
    public_api_visitor.do_not_descend_map['tf.experimental.numpy'] = [
        'bool_', 'complex_', 'complex128', 'complex64', 'float_', 'float16',
        'float32', 'float64', 'inexact', 'int_', 'int16', 'int32', 'int64',
        'int8', 'object_', 'string_', 'uint16', 'uint32', 'uint64', 'uint8',
        'unicode_', 'iinfo']
    public_api_visitor.do_not_descend_map['tf'].append('keras')
    if FLAGS.only_test_core_api:
      public_api_visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
      if api_version == 2:
        public_api_visitor.do_not_descend_map['tf'].extend(_V2_APIS_FROM_KERAS)
      else:
        public_api_visitor.do_not_descend_map['tf'].extend(['layers'])
        public_api_visitor.do_not_descend_map['tf.nn'] = ['rnn_cell']
    if additional_private_map:
      public_api_visitor.private_map.update(additional_private_map)

    traverse.traverse(root, public_api_visitor)
    proto_dict = visitor.GetProtos()

    # Read all golden files.
    golden_file_list = file_io.get_matching_files(golden_file_patterns)
    if FLAGS.only_test_core_api:
      golden_file_list = _FilterNonCoreGoldenFiles(golden_file_list)
      if api_version == 2:
        golden_file_list = _FilterV2KerasRelatedGoldenFiles(golden_file_list)
      else:
        golden_file_list = _FilterV1KerasRelatedGoldenFiles(golden_file_list)

    def _ReadFileToProto(filename):
      # [FUNCTION DOC]
      # HELPER: parses one golden file. Reads the text file with the given name and merges its
      # text-format contents into a new API object protocol buffer, which is returned. Raises a
      # parse error for malformed content or a file error if unreadable.
      """Read a filename, create a protobuf from its contents."""
      ret_val = api_objects_pb2.TFAPIObject()
      text_format.Merge(file_io.read_file_to_string(filename), ret_val)
      return ret_val

    golden_proto_dict = {
        _FileNameToKey(filename): _ReadFileToProto(filename)
        for filename in golden_file_list
    }
    golden_proto_dict = _FilterGoldenProtoDict(golden_proto_dict,
                                               omit_golden_symbols_map)
    proto_dict = _FilterGoldenProtoDict(proto_dict, omit_golden_symbols_map)

    # Diff them. Do not fail if called with update.
    # If the test is run to update goldens, only report diffs but do not fail.
    self._AssertProtoDictEquals(
        golden_proto_dict,
        proto_dict,
        verbose=FLAGS.verbose_diffs,
        update_goldens=FLAGS.update_goldens,
        api_version=api_version)

  def testAPIBackwardsCompatibility(self):
    # [FUNCTION DOC]
    # Checks the default tensorflow API against goldens. Chooses version 2 if the package reports
    # major API version 2, else version 1; builds golden patterns for all goldens plus the numpy
    # goldens; when testing core only without TensorBoard installed in version 2, omits the
    # summary symbols audio, histogram, image, scalar and text; also ignores is-instance data for
    # the function graph type. Runs the shared check on the tensorflow package, hiding compat v1
    # and v2, then asserts that version 2 has no contrib attribute (version 1 is exempt).
    # Takes no inputs; returns nothing.
    api_version = 1
    if hasattr(tf, '_major_api_version') and tf._major_api_version == 2:
      api_version = 2
    golden_file_patterns = [
        os.path.join(resource_loader.get_root_dir_with_all_resources(),
                     _KeyToFilePath('*', api_version)),
        _GetTFNumpyGoldenPattern(api_version)]
    omit_golden_symbols_map = {}
    if (api_version == 2 and FLAGS.only_test_core_api and
        not _TENSORBOARD_AVAILABLE):
      # In TF 2.0 these summary symbols are imported from TensorBoard.
      omit_golden_symbols_map['tensorflow.summary'] = [
          'audio', 'histogram', 'image', 'scalar', 'text'
      ]
    omit_golden_symbols_map.update(
        self._ignored_is_instance_types(['tensorflow.__internal__.FuncGraph'])
    )

    self._checkBackwardsCompatibility(
        tf,
        golden_file_patterns,
        api_version,
        # Skip compat.v1 and compat.v2 since they are validated
        # in separate tests.
        additional_private_map={'tf.compat': ['v1', 'v2']},
        omit_golden_symbols_map=omit_golden_symbols_map)

    # Check that V2 API does not have contrib
    self.assertTrue(api_version == 1 or not hasattr(tf, 'contrib'))

  def testAPIBackwardsCompatibilityV1(self):
    # [FUNCTION DOC]
    # Checks the compat v1 API against the version 1 goldens. Omits the low-level pywrap symbol
    # from the top-level object, ignores is-instance data for the record writer type, and omits
    # the log_if symbol of the logging module because the open-source abseil version differs.
    # Hides the pywrap symbol and compat v1/v2 during traversal. Takes no inputs; returns nothing.
    api_version = 1
    golden_file_patterns = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        _KeyToFilePath('*', api_version))
    omit_golden_symbols_map = {'tensorflow': ['pywrap_tensorflow']}
    omit_golden_symbols_map.update(
        self._ignored_is_instance_types(['tensorflow.python_io.TFRecordWriter'])
    )
    # In OSS we have a different version of ABSL.
    omit_golden_symbols_map['tensorflow.logging'] = ['log_if']
    self._checkBackwardsCompatibility(
        tf.compat.v1,
        golden_file_patterns,
        api_version,
        additional_private_map={
            'tf': ['pywrap_tensorflow'],
            'tf.compat': ['v1', 'v2'],
        },
        omit_golden_symbols_map=omit_golden_symbols_map)

  def testAPIBackwardsCompatibilityV2(self):
    # [FUNCTION DOC]
    # Checks the compat v2 API against the version 2 goldens plus the numpy goldens. When testing
    # core only without TensorBoard, omits the summary symbols audio, histogram, image, scalar and
    # text; ignores is-instance data for the function graph type; hides compat v1/v2 during
    # traversal. Takes no inputs; returns nothing.
    api_version = 2
    golden_file_patterns = [
        os.path.join(resource_loader.get_root_dir_with_all_resources(),
                     _KeyToFilePath('*', api_version)),
        _GetTFNumpyGoldenPattern(api_version)]
    omit_golden_symbols_map = {}
    if FLAGS.only_test_core_api and not _TENSORBOARD_AVAILABLE:
      # In TF 2.0 these summary symbols are imported from TensorBoard.
      omit_golden_symbols_map['tensorflow.summary'] = [
          'audio', 'histogram', 'image', 'scalar', 'text'
      ]
    omit_golden_symbols_map.update(
        self._ignored_is_instance_types(['tensorflow.__internal__.FuncGraph'])
    )

    self._checkBackwardsCompatibility(
        tf.compat.v2,
        golden_file_patterns,
        api_version,
        additional_private_map={'tf.compat': ['v1', 'v2']},
        omit_golden_symbols_map=omit_golden_symbols_map)

  def _ignored_is_instance_types(self, extra_types=None):
    # [FUNCTION DOC]
    # HELPER: builds the omit mapping that makes the comparison ignore is-instance data for
    # types whose location differs between internal builds and the open-source wheel.
    # KNOWN ISSUE: operator precedence makes the conditional apply to the whole sum, so when
    # extra_types is null or empty the result is an empty list and none of the built-in ignored
    # types (DType, Graph, Operation, and so on) are ignored; only when extra_types is given is the
    # full list used. The type list is also given as the list of keys, each mapped to a plain text
    # value rather than a list.
    # Input: optional list of extra object keys. Returns a mapping from each key to the text
    # is_instance. No side effects.
    # In case a new type is defined within a pywrap_<module_name>.so library,
    # it will end up having proper type and location in distributed OSS wheel
    # package eventually, but that conversion happens after this test is ran.
    #
    # Making this test depend on wheel itself also breaks because wheels use
    # _upb as underlying protobuf implementation while internal TF uses cpp
    # implementation (resulting in different is_instance values for protobuf
    # metadata types in golden pbtxt depending on which protobuf implementation
    # is being used during test execution). The cpp implementation is not  even
    # included anymore in protobuf oss wheels.
    #
    # We end up in a situation when we cannot make this test pass internally and
    # externally on the same set of golden expected .pbtxt inputs. It is rare
    # and minor discrepancy, so just ignore the is_instance checks for the few
    # problematic types, they are guaraneed to have proper types in final wheel
    # anyway.
    ignored_is_instance_types = [
        'tensorflow.DType',
        'tensorflow.dtypes.DType',
        'tensorflow.__internal__.SymbolicTensor',
        'tensorflow.Graph',
        'tensorflow.Operation',
        'tensorflow.io.TFRecordWriter'
    ] + extra_types if extra_types else []
    return {k: 'is_instance' for k in ignored_is_instance_types}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--update_goldens', type=bool, default=False, help=_UPDATE_GOLDENS_HELP)
  parser.add_argument(
      '--only_test_core_api',
      type=bool,
      default=True,  # only_test_core_api default value
      help=_ONLY_TEST_CORE_API_HELP)
  parser.add_argument(
      '--verbose_diffs', type=bool, default=True, help=_VERBOSE_DIFFS_HELP)
  FLAGS, unparsed = parser.parse_known_args()
  _InitPathConstants()

  # Now update argv, so that unittest library does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
