# [MODULE DOC]
# Public memory leak detection utility for tests and debugging. Exposes the MemoryChecker context
# manager, which records memory snapshots and asserts no leak or no new Python objects, delegating
# the work to the internal Python memory checker; also has a private helper that finds the current
# test name. No import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Memory leak detection utility."""

from tensorflow.python.framework.python_memory_checker import _PythonMemoryChecker
from tensorflow.python.profiler import trace
from tensorflow.python.util import tf_inspect


def _get_test_name_best_effort():
  # [FUNCTION DOC]
  # HELPER: private utility for naming the running test (apparently for memory checker reports).
  # KNOWN ISSUE: it is not called anywhere in this module; the stack-walking result is unused here.
  # Walks the current call stack from the innermost frame outward and finds the first frame whose
  # function name begins with the word test. If that frame has a local variable named self, returns
  # the class name of that object joined to the function name with a dot. Any failure while reading
  # the frame is swallowed by a bare catch and the search continues outward. Returns nothing (null
  # value) if no such frame is found. Takes no arguments; no side effects.
  """If available, return the current test name. Otherwise, `None`."""
  for stack in tf_inspect.stack():
    function_name = stack[3]
    if function_name.startswith('test'):
      try:
        class_name = stack[0].f_locals['self'].__class__.__name__
        return class_name + '.' + function_name
      except:  # pylint:disable=bare-except
        pass

  return None


# TODO(kkb): Also create decorator versions for convenience.
class MemoryChecker(object):
  # [CLASS DOC]
  # Context manager used to detect Python and native memory leaks in tests or while debugging.
  # Entering creates an internal Python memory checker held as private state; the user then calls
  # the snapshot method once per iteration at the same code location, and afterwards produces a
  # report or asserts that no leak (or no new Python objects) occurred. All real work is delegated
  # to the internal checker. Most methods are wrapped with a tracing decorator, except the snapshot
  # method, to avoid contaminating the snapshots. Instances are created with no arguments.
  """Memory leak detection class.

  This is a utility class to detect Python and C++ memory leaks. It's intended
  for both testing and debugging. Basic usage:

  >>> # MemoryChecker() context manager tracks memory status inside its scope.
  >>> with MemoryChecker() as memory_checker:
  >>>   tensors = []
  >>>   for _ in range(10):
  >>>     # Simulating `tf.constant(1)` object leak every iteration.
  >>>     tensors.append(tf.constant(1))
  >>>
  >>>     # Take a memory snapshot for later analysis.
  >>>     memory_checker.record_snapshot()
  >>>
  >>> # `report()` generates a html graph file showing allocations over
  >>> # snapshots per every stack trace.
  >>> memory_checker.report()
  >>>
  >>> # This assertion will detect `tf.constant(1)` object leak.
  >>> memory_checker.assert_no_leak_if_all_possibly_except_one()

  `record_snapshot()` must be called once every iteration at the same location.
  This is because the detection algorithm relies on the assumption that if there
  is a leak, it's happening similarly on every snapshot.
  """

  @trace.trace_wrapper
  def __enter__(self):
    # [FUNCTION DOC]
    # Context-manager entry. Creates a fresh internal Python memory checker and stores it on the
    # instance (replacing any earlier one), then returns the instance itself so it can be bound by
    # the with statement. Takes no arguments.
    self._python_memory_checker = _PythonMemoryChecker()
    return self

  @trace.trace_wrapper
  def __exit__(self, exc_type, exc_value, traceback):
    # [FUNCTION DOC]
    # Context-manager exit. Receives the exception type, value and traceback (null values when the
    # block finished normally) and intentionally does nothing; it returns nothing so exceptions are
    # not suppressed. Snapshots and the checker stay available for the report and assertions.
    pass

  # We do not enable trace_wrapper on this function to avoid contaminating
  # the snapshot.
  def record_snapshot(self):
    # [FUNCTION DOC]
    # Asks the internal Python memory checker to record one memory snapshot (of Python objects and
    # native allocations). Must be called after entering the context, otherwise the internal checker
    # does not exist and an attribute error results. Takes no arguments, returns nothing. Not traced
    # on purpose so the tracing does not alter the snapshot.
    """Take a memory snapshot for later analysis.

    `record_snapshot()` must be called once every iteration at the same
    location. This is because the detection algorithm relies on the assumption
    that if there is a leak, it's happening similarly on every snapshot.

    The recommended number of `record_snapshot()` call depends on the testing
    code complexity and the allocation pattern.
    """
    self._python_memory_checker.record_snapshot()

  @trace.trace_wrapper
  def report(self):
    # [FUNCTION DOC]
    # Delegates to the internal checker to write an HTML graph of allocations over the recorded
    # snapshots, grouped by stack trace, into a temporary directory (or the directory supplied by
    # the testing infrastructure when present). Takes no arguments, returns nothing; side effect is
    # creating output files. Requires the context to have been entered.
    """Generates a html graph file showing allocations over snapshots.

    It create a temporary directory and put all the output files there.
    If this is running under Google internal testing infra, it will use the
    directory provided the infra instead.
    """
    self._python_memory_checker.report()

  @trace.trace_wrapper
  def assert_no_leak_if_all_possibly_except_one(self):
    # [FUNCTION DOC]
    # Delegates to the internal checker, which treats a series of allocations as a leak if the same
    # Python type, or the same native stack trace, appears at every snapshot except possibly one.
    # Raises an assertion error when a leak is detected; otherwise returns nothing. Takes no
    # arguments. Requires the context to have been entered and snapshots recorded.
    """Raises an exception if a leak is detected.

    This algorithm classifies a series of allocations as a leak if it's the same
    type(Python) or it happens at the same stack trace(C++) at every snapshot,
    but possibly except one snapshot.
    """

    self._python_memory_checker.assert_no_leak_if_all_possibly_except_one()

  @trace.trace_wrapper
  def assert_no_new_python_objects(self, threshold=None):
    # [FUNCTION DOC]
    # Compares the first and last recorded snapshots and counts new Python objects per type name.
    # Parameter threshold is an optional mapping from type name strings to allowed counts; types
    # whose growth stays within the allowance are tolerated; null means no allowance. Delegates to
    # the internal checker, which raises an assertion error describing new Python objects when the
    # growth exceeds the allowance. Returns nothing.
    """Raises an exception if there are new Python objects created.

    It computes the number of new Python objects per type using the first and
    the last snapshots.

    Args:
      threshold: A dictionary of [Type name string], [count] pair. It won't
        raise an exception if the new Python objects are under this threshold.
    """
    self._python_memory_checker.assert_no_new_objects(threshold=threshold)
