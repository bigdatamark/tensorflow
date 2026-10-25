# [MODULE DOC]
# HELPER: helper module acting as the main program of Python TPU tests built by the TPU test
# build macro; it is not a test itself.
# Defines flags (path of the wrapped user test module, base directory for temporary test files,
# repository root substring) at import time, ensures the standard TPU flags (tpu, zone, project,
# model_dir) exist, picks a random model directory per run, imports the user test module, copies
# its test classes into this module so the test runner finds them, and finally executes the
# user module's main block. When run as a script it also restores test filter and sharding
# environment variables from their saved copies if the regular ones are missing.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Wrapper for Python TPU tests.

The py_tpu_test macro will actually use this file as its main, building and
executing the user-provided test file as a py_binary instead. This lets us do
important work behind the scenes, without complicating the tests themselves.

The main responsibilities of this file are:
  - Define standard set of model flags if test did not. This allows us to
    safely set flags at the Bazel invocation level using --test_arg.
  - Pick a random directory on GCS to use for each test case, and set it as the
    default value of --model_dir. This is similar to how Bazel provides each
    test with a fresh local directory in $TEST_TMPDIR.
"""

import ast
import importlib
import os
import sys
import uuid

from tensorflow.python.platform import flags
from tensorflow.python.util import tf_inspect

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'wrapped_tpu_test_module_relative', None,
    'The Python-style relative path to the user-given test. If test is in same '
    'directory as BUILD file as is common, then "test.py" would be ".test".')
flags.DEFINE_string('test_dir_base',
                    os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'GCS path to root directory for temporary test files.')
flags.DEFINE_string(
    'bazel_repo_root', 'tensorflow/python',
    'Substring of a bazel filepath beginning the python absolute import path.')

# List of flags which all TPU tests should accept.
REQUIRED_FLAGS = ['tpu', 'zone', 'project', 'model_dir']


def maybe_define_flags():
  # [FUNCTION DOC]
  # HELPER: ensures the flags every TPU test must accept exist.
  # For each name in the required list (tpu, zone, project, model_dir) defines a string flag with
  # no default, silently ignoring flags the user test already defined. Takes no inputs,
  # returns nothing; side effect is registering global flags.
  """Defines any required flags that are missing."""
  for f in REQUIRED_FLAGS:
    try:
      flags.DEFINE_string(f, None, 'flag defined by test lib')
    except flags.DuplicateFlagError:
      pass


def set_random_test_dir():
  # [FUNCTION DOC]
  # HELPER: gives each test run its own model directory.
  # KNOWN ISSUE: if the base directory flag is unset (its default is taken from an environment
  # variable that may be missing), joining paths fails with a type error.
  # Joins the base directory flag with a fresh random hexadecimal name and sets the result as
  # the default value of the model directory flag (which must already be defined). Returns
  # nothing.
  """Pick a random GCS directory under --test_dir_base, set as --model_dir."""
  path = os.path.join(FLAGS.test_dir_base, uuid.uuid4().hex)
  FLAGS.set_default('model_dir', path)


def calculate_parent_python_path(test_filepath):
  # [FUNCTION DOC]
  # HELPER: used to resolve the relative import of the user module.
  # Input: file path of the invoked test (text). Takes the part after the last occurrence of the
  # repository root flag, prefixes the root, drops the final path component (the wrapper name)
  # and converts separators to dots. Returns the dotted import path of the containing package.
  # Raises a value error if the root substring is not in the path.
  """Returns the absolute import path for the containing directory.

  Args:
    test_filepath: The filepath which Bazel invoked
      (ex: /filesystem/path/tensorflow/tensorflow/python/tpu/tpu_test)

  Returns:
    Absolute import path of parent (ex: tensorflow.python.tpu).

  Raises:
    ValueError: if bazel_repo_root does not appear within test_filepath.
  """
  # We find the last occurrence of bazel_repo_root, and drop everything before.
  split_path = test_filepath.rsplit(FLAGS.bazel_repo_root, 1)
  if len(split_path) < 2:
    raise ValueError(
        f'Filepath "{test_filepath}" does not contain repo root "{FLAGS.bazel_repo_root}"'
    )

  path = FLAGS.bazel_repo_root + split_path[1]

  # We drop the last portion of the path, which is the name of the test wrapper.
  path = path.rsplit('/', 1)[0]

  # We convert the directory separators into dots.
  return path.replace('/', '.')


def import_user_module():
  # [FUNCTION DOC]
  # HELPER: loads the wrapped test.
  # Imports the module named by the relative-module flag, relative to the package computed from
  # the first command-line argument. Importing runs its top-level code, including flag
  # definitions. Returns the module object; import errors propagate.
  """Imports the flag-specified user test code.

  This runs all top-level statements in the user module, specifically flag
  definitions.

  Returns:
    The user test module.
  """
  return importlib.import_module(FLAGS.wrapped_tpu_test_module_relative,
                                 calculate_parent_python_path(sys.argv[0]))


def _is_test_class(obj):
  # [FUNCTION DOC]
  # HELPER: filter used when moving test classes.
  # Takes any object; returns true if it is a class and any class in its inheritance chain is
  # named TestCase (so different test libraries work), otherwise false. No side effects.
  """Check if arbitrary object is a test class (not a test object!).

  Args:
    obj: An arbitrary object from within a module.

  Returns:
    True iff obj is a test class inheriting at some point from a module
    named "TestCase". This is because we write tests using different underlying
    test libraries.
  """
  return (tf_inspect.isclass(obj)
          and 'TestCase' in (p.__name__ for p in tf_inspect.getmro(obj)))


module_variables = vars()


def move_test_classes_into_scope(wrapped_test_module):
  # [FUNCTION DOC]
  # HELPER: makes the test runner see the wrapped tests.
  # Input: the user test module. For every test class found in its namespace, adds a variable to
  # this module's namespace named with a fixed prefix plus the class name. Returns nothing;
  # mutates this module's global namespace.
  """Add all test classes defined in wrapped module to our module.

  The test runner works by inspecting the main module for TestCase classes, so
  by adding a module-level reference to the TestCase we cause it to execute the
  wrapped TestCase.

  Args:
    wrapped_test_module: The user-provided test code to run.
  """
  for name, obj in wrapped_test_module.__dict__.items():
    if _is_test_class(obj):
      module_variables['tpu_test_imported_%s' % name] = obj


def run_user_main(wrapped_test_module):
  # [FUNCTION DOC]
  # HELPER: executes the entry-point block of the wrapped test.
  # KNOWN ISSUE: the not-found error message mentions a main-name condition spelled with the
  # word main rather than the dunder form actually searched for.
  # Input: the user test module. Parses its source, finds the last top-level conditional whose
  # test equals the main-module check, and executes that block's body with the wrapper's globals
  # and the user module namespace as locals. Raises a not-implemented error if no such block
  # exists. Side effects: runs arbitrary user code (typically starting the test runner).
  """Runs the "if __name__ == '__main__'" at the bottom of a module.

  TensorFlow practice is to have a main if at the bottom of the module which
  might call an API compat function before calling test.main().

  Since this is a statement, not a function, we can't cleanly reference it, but
  we can inspect it from the user module and run it in the context of that
  module so all imports and variables are available to it.

  Args:
    wrapped_test_module: The user-provided test code to run.

  Raises:
    NotImplementedError: If main block was not found in module. This should not
      be caught, as it is likely an error on the user's part -- absltest is all
      too happy to report a successful status (and zero tests executed) if a
      user forgets to end a class with "test.main()".
  """
  tree = ast.parse(tf_inspect.getsource(wrapped_test_module))

  # Get string representation of just the condition `__name == "__main__"`.
  target = ast.dump(ast.parse('if __name__ == "__main__": pass').body[0].test)

  # `tree.body` is a list of top-level statements in the module, like imports
  # and class definitions. We search for our main block, starting from the end.
  for expr in reversed(tree.body):
    if isinstance(expr, ast.If) and ast.dump(expr.test) == target:
      break
  else:
    raise NotImplementedError(
        f'Could not find `if __name__ == "main":` block in {wrapped_test_module.__name__}.'
        )

  # expr is defined because we would have raised an error otherwise.
  new_ast = ast.Module(body=expr.body, type_ignores=[])  # pylint:disable=undefined-loop-variable
  exec(  # pylint:disable=exec-used
      compile(new_ast, '<ast>', 'exec'),
      globals(),
      wrapped_test_module.__dict__,
  )


if __name__ == '__main__':
  # Partially parse flags, since module to import is specified by flag.
  unparsed = FLAGS(sys.argv, known_only=True)
  # Test filters from bazel's --test_filter flag are passed via the
  # TESTBRIDGE_TEST_ONLY environment variable. This wrapper script is typically
  # run from within another test case handler that doesn't actually hold the
  # user test cases. So, in oder to make --test_filter work correctly, the
  # outer test case needs to hide TESTBRIDGE_TEST_ONLY and we need to re-expose
  # it here. To do this, the outer test case stores the value in
  # SAVED_TESTBRIDGE_TEST_ONLY which we can use to set TESTBRIDGE_TEST_ONLY in
  # this local context.
  # The same holds for test sharding which is enabled through the num_shards
  # attribute in the test rule. For test sharding, the two key variables are
  # TEST_SHARD_INDEX and TEST_TOTAL_SHARDS.
  saved_test_filter = os.getenv('SAVED_TESTBRIDGE_TEST_ONLY')
  existing_test_filter = os.getenv('TESTBRIDGE_TEST_ONLY')
  if saved_test_filter and not existing_test_filter:
    os.environ['TESTBRIDGE_TEST_ONLY'] = saved_test_filter
  user_module = import_user_module()
  saved_shard_index = os.getenv('SAVED_TEST_SHARD_INDEX')
  existing_shard_index = os.getenv('TEST_SHARD_INDEX')
  if saved_shard_index and not existing_shard_index:
    os.environ['TEST_SHARD_INDEX'] = saved_shard_index
  saved_total_shards = os.getenv('SAVED_TEST_TOTAL_SHARDS')
  existing_total_shards = os.getenv('TEST_TOTAL_SHARDS')
  if saved_total_shards and not existing_total_shards:
    os.environ['TEST_TOTAL_SHARDS'] = saved_total_shards
  maybe_define_flags()
  # Parse remaining flags.
  FLAGS(unparsed, known_only=True)
  set_random_test_dir()

  move_test_classes_into_scope(user_module)
  run_user_main(user_module)
