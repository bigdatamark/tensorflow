# [MODULE DOC]
# Test framework router. Re-exports everything from absl's testing module (test case class,
# main, etc.) and adds TensorFlow-specific pieces: an alias for the TensorFlow benchmark base
# class, a replacement main that routes to the benchmark runner when a benchmark filter flag is
# present, a per-process temporary directory helper, a path helper for test data under the
# runfiles directory, a stub-out helper class for monkey-patching, and a placeholder query for
# stateful sessions. Import-time state: a module-level cached temp directory path, initially
# empty. Re-binds the name main after saving the original as a separate name.

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
# ==============================================================================

"""Imports absltest as a replacement for testing.pybase.googletest."""
import atexit
import os
import sys
import tempfile

# go/tf-wildcard-import
# pylint: disable=wildcard-import,redefined-builtin
from absl import app
from absl.testing.absltest import *
# pylint: enable=wildcard-import,redefined-builtin

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


Benchmark = benchmark.TensorFlowBenchmark  # pylint: disable=invalid-name

absltest_main = main

# We keep a global variable in this module to make sure we create the temporary
# directory only once per test binary invocation.
_googletest_temp_dir = ''


# pylint: disable=invalid-name
# pylint: disable=undefined-variable
def g_main(argv):
  # [FUNCTION DOC]
  # HELPER: target passed to the application runner by the replacement main.
  # Calls the original absl test main with the given argument list. Returns nothing; the absl
  # main typically exits the process after running tests.
  """Delegate to absltest.main."""

  absltest_main(argv=argv)


# Redefine main to allow running benchmarks
def main(argv=None):  # pylint: disable=function-redefined
  # [FUNCTION DOC]
  # Replacement test main that supports benchmarks. Input: optional argument list (defaults to
  # the process arguments when benchmarks are not requested). Delegates to the benchmark runner
  # entry point, which either runs matching benchmarks (if a benchmark filter argument is present)
  # or calls the inner wrapper that runs the normal absl test main through the app runner.
  # Returns nothing (the inner wrapper's return value is discarded).
  def main_wrapper():
    # [FUNCTION DOC]
    # HELPER: the "normal" path for main. Uses the argument list given to main, or the process
    # arguments if none, and runs the app runner with the g_main function. Returns its result.
    args = argv
    if args is None:
      args = sys.argv
    return app.run(main=g_main, argv=args)

  benchmark.benchmarks_main(true_main=main_wrapper, argv=argv)


def GetTempDir():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the fallback branch strips trailing characters with a character-set strip rather
  # than removing a ".py" suffix, so names ending in characters p, y or dot lose extra letters
  # (cosmetic, only affects the directory prefix). The environment-variable branch uses the value
  # as a name prefix rather than as the parent directory.
  # Returns a per-process temporary directory path, creating it on first call and caching it in
  # module state so later calls return the same path. If the TEST_TMPDIR environment variable is
  # set, creates a unique directory whose name begins with that value; otherwise creates one in
  # the system temp directory prefixed with the name of the outermost script file. Normalizes
  # separators, and registers an exit handler that recursively deletes the directory, logging an
  # error (not raising) if deletion fails. No inputs.
  """Return a temporary directory for tests to use."""
  global _googletest_temp_dir
  if not _googletest_temp_dir:
    if os.environ.get('TEST_TMPDIR'):
      temp_dir = tempfile.mkdtemp(prefix=os.environ['TEST_TMPDIR'])
    else:
      first_frame = tf_inspect.stack()[-1][0]
      temp_dir = os.path.join(tempfile.gettempdir(),
                              os.path.basename(tf_inspect.getfile(first_frame)))
      temp_dir = tempfile.mkdtemp(prefix=temp_dir.rstrip('.py'))

    # Make sure we have the correct path separators.
    temp_dir = temp_dir.replace('/', os.sep)

    def delete_temp_dir(dirname=temp_dir):
      # [FUNCTION DOC]
      # HELPER: exit handler registered by the temp directory function. Recursively deletes the
      # given directory (default: the created temp directory); a TensorFlow operation error is
      # logged rather than raised. Returns nothing.
      try:
        file_io.delete_recursively(dirname)
      except errors.OpError as e:
        logging.error('Error removing %s: %s', dirname, e)

    atexit.register(delete_temp_dir)

    _googletest_temp_dir = temp_dir

  return _googletest_temp_dir


def test_src_dir_path(relative_path):
  # [FUNCTION DOC]
  # Builds an absolute path to a data file in the test runfiles tree: joins the test source
  # directory and test workspace environment variables, the literal tensorflow directory, and the
  # given relative path (string). Raises a key error if either environment variable is unset.
  # Returns the path string; no side effects.
  """Creates an absolute test srcdir path given a relative path.

  Args:
    relative_path: a path relative to tensorflow root.
      e.g. "contrib/session_bundle/example".

  Returns:
    An absolute path to the linked in runfiles.
  """
  return os.path.join(
      os.environ['TEST_SRCDIR'],
      os.environ['TEST_WORKSPACE'],
      'tensorflow',
      relative_path,
  )


def StatefulSessionAvailable():
  # [FUNCTION DOC]
  # Placeholder capability query for the open-source build: always returns false, meaning
  # stateful sessions are not available. No inputs or side effects.
  return False


@tf_export(v1=['test.StubOutForTesting'])
class StubOutForTesting(object):
  # [CLASS DOC]
  # Monkey-patching helper for tests. Keeps two lists: a cache of replaced attributes made via the
  # simple setter (parent, old value, name) and a list of those made via the smart setter (owner,
  # name, original). Replacements are undone by the clean-up methods, which are also called when
  # used as a context manager and (unreliably) on destruction. Exported in v1 as a test utility.
  """Support class for stubbing methods out for unit testing.

  Sample Usage:

  You want os.path.exists() to always return true during testing.

     stubs = StubOutForTesting()
     stubs.Set(os.path, 'exists', lambda x: 1)
       ...
     stubs.CleanUp()

  The above changes os.path.exists into a lambda that returns 1.  Once
  the ... part of the code finishes, the CleanUp() looks up the old
  value of os.path.exists and restores it.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes two empty lists: the cache of simple replacements and the list of
    # smart replacements. Takes no inputs.
    self.cache = []
    self.stubs = []

  def __del__(self):
    # [FUNCTION DOC]
    # Destructor: calls the clean-up method to restore every stubbed attribute. Timing of
    # destructor invocation is not guaranteed by the language runtime, so callers should clean up
    # explicitly or use the context manager form.
    """Do not rely on the destructor to undo your stubs.

    You cannot guarantee exactly when the destructor will get called without
    relying on implementation details of a Python VM that may change.
    """
    self.CleanUp()

  # __enter__ and __exit__ allow use as a context manager.
  def __enter__(self):
    # [FUNCTION DOC]
    # Context manager entry: returns the stub helper itself. No side effects.
    return self

  def __exit__(self, unused_exc_type, unused_exc_value, unused_tb):
    # [FUNCTION DOC]
    # Context manager exit: restores all stubbed attributes via the clean-up method. The exception
    # information arguments are ignored; exceptions are not suppressed (returns nothing).
    self.CleanUp()

  def CleanUp(self):
    # [FUNCTION DOC]
    # Restores every attribute replaced through either setter: first undoes the smart
    # replacements, then the simple ones. Safe to call repeatedly. Returns nothing.
    """Undoes all SmartSet() & Set() calls, restoring original definitions."""
    self.SmartUnsetAll()
    self.UnsetAll()

  def SmartSet(self, obj, attr_name, new_attr):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the attribute is replaced on obj, but for instances the
    # class is patched (as the notes say), affecting all instances. In the inheritance search the
    # attribute is looked up on obj itself for every class, so the loop only decides which class
    # ends up patched (the most derived one) and the lookup result is the same each time.
    # Replaces an attribute by name with a new value, working for modules, classes and instances
    # while keeping inheritance intact. Inputs: obj (module, class or instance, decorator wrappers
    # are unwrapped first), attr_name (string), new_attr (any value). Remembers the original
    # (re-wrapping static methods as static methods) in the smart list, then sets the new value
    # on the chosen owner. Raises an attribute error if the attribute cannot be found.
    """Replace obj.attr_name with new_attr.

    This method is smart and works at the module, class, and instance level
    while preserving proper inheritance. It will not stub out C types however
    unless that has been explicitly allowed by the type.

    This method supports the case where attr_name is a staticmethod or a
    classmethod of obj.

    Notes:
      - If obj is an instance, then it is its class that will actually be
        stubbed. Note that the method Set() does not do that: if obj is
        an instance, it (and not its class) will be stubbed.
      - The stubbing is using the builtin getattr and setattr. So, the __get__
        and __set__ will be called when stubbing (TODO: A better idea would
        probably be to manipulate obj.__dict__ instead of getattr() and
        setattr()).

    Args:
      obj: The object whose attributes we want to modify.
      attr_name: The name of the attribute to modify.
      new_attr: The new value for the attribute.

    Raises:
      AttributeError: If the attribute cannot be found.
    """
    _, obj = tf_decorator.unwrap(obj)
    if (tf_inspect.ismodule(obj) or
        (not tf_inspect.isclass(obj) and attr_name in obj.__dict__)):
      orig_obj = obj
      orig_attr = getattr(obj, attr_name)
    else:
      if not tf_inspect.isclass(obj):
        mro = list(tf_inspect.getmro(obj.__class__))
      else:
        mro = list(tf_inspect.getmro(obj))

      mro.reverse()

      orig_attr = None
      found_attr = False

      for cls in mro:
        try:
          orig_obj = cls
          orig_attr = getattr(obj, attr_name)
          found_attr = True
        except AttributeError:
          continue

      if not found_attr:
        raise AttributeError('Attribute not found.')

    # Calling getattr() on a staticmethod transforms it to a 'normal' function.
    # We need to ensure that we put it back as a staticmethod.
    old_attribute = obj.__dict__.get(attr_name)
    if old_attribute is not None and isinstance(old_attribute, staticmethod):
      orig_attr = staticmethod(orig_attr)

    self.stubs.append((orig_obj, attr_name, orig_attr))
    setattr(orig_obj, attr_name, new_attr)

  def SmartUnsetAll(self):
    # [FUNCTION DOC]
    # Undoes all smart replacements in reverse order by restoring each saved original on its
    # owner, then empties the smart list. Repeated calls are harmless. Returns nothing.
    """Reverses SmartSet() calls, restoring things to original definitions.

    This method is automatically called when the StubOutForTesting()
    object is deleted; there is no need to call it explicitly.

    It is okay to call SmartUnsetAll() repeatedly, as later calls have
    no effect if no SmartSet() calls have been made.
    """
    for args in reversed(self.stubs):
      setattr(*args)

    self.stubs = []

  def Set(self, parent, child_name, new_child):
    # [FUNCTION DOC]
    # Replaces the named attribute of parent (module or class, or an object whose own attribute is
    # patched) with a new value. Reads the old value (raising an attribute error if missing),
    # re-wraps it as a static method if it was one, records parent, old value and name in the
    # cache for later restoration, then assigns the new value. Returns nothing.
    """In parent, replace child_name's old definition with new_child.

    The parent could be a module when the child is a function at
    module scope.  Or the parent could be a class when a class' method
    is being replaced.  The named child is set to new_child, while the
    prior definition is saved away for later, when UnsetAll() is
    called.

    This method supports the case where child_name is a staticmethod or a
    classmethod of parent.

    Args:
      parent: The context in which the attribute child_name is to be changed.
      child_name: The name of the attribute to change.
      new_child: The new value of the attribute.
    """
    old_child = getattr(parent, child_name)

    old_attribute = parent.__dict__.get(child_name)
    if old_attribute is not None and isinstance(old_attribute, staticmethod):
      old_child = staticmethod(old_child)

    self.cache.append((parent, old_child, child_name))
    setattr(parent, child_name, new_child)

  def UnsetAll(self):
    # [FUNCTION DOC]
    # Undoes all simple replacements in reverse order so repeated replacements of the same
    # attribute end with the original value, then empties the cache. Repeated calls are harmless.
    # Returns nothing.
    """Reverses Set() calls, restoring things to their original definitions.

    This method is automatically called when the StubOutForTesting()
    object is deleted; there is no need to call it explicitly.

    It is okay to call UnsetAll() repeatedly, as later calls have no
    effect if no Set() calls have been made.
    """
    # Undo calls to Set() in reverse order, in case Set() was called on the
    # same arguments repeatedly (want the original call to be last one undone)
    for (parent, old_child, child_name) in reversed(self.cache):
      setattr(parent, child_name, old_child)
    self.cache = []
