# [MODULE DOC]
# Tests for the native bridge that lets C++ code enter and exit a Python context manager
# (exposed through the private _py_context_manager extension). Defines a recording context
# manager class with configurable misbehavior, expected log texts, and a test case covering the
# normal flow, an exception in the body, an exception on entering, and the unsupported cases
# where exiting raises or suppresses an exception. Running the file starts the googletest runner.
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
"""Tests for tensorflow.python.framework._py_context_manager."""

from tensorflow.python.framework import _py_context_manager
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class TestContextManager(object):
  # [CLASS DOC]
  # HELPER: fixture context manager for the tests in this module.
  # A context manager that records every entry and exit into a list of log strings and can be
  # configured with a behavior name: basic (default, no special action), raise from enter,
  # raise from exit, or suppress exception. Entering yields the text "var" as the bound value.

  def __init__(self, behavior="basic"):
    # [FUNCTION DOC]
    # Creates the fixture. Input: behavior, a string selecting special behavior (default
    # "basic"); recognized values are raise from enter, raise from exit and suppress exception.
    # Initializes an empty log list and stores the behavior. Returns nothing.
    self.log = []
    self.behavior = behavior

  def __enter__(self):
    # [FUNCTION DOC]
    # Appends an entry message to the log. If the behavior is raise from enter, raises a value
    # error saying the exception happened in enter. Otherwise returns the text "var", which the
    # body receives as its argument.
    self.log.append("__enter__()")
    if self.behavior == "raise_from_enter":
      raise ValueError("exception in __enter__")
    return "var"

  def __exit__(self, ex_type, ex_value, ex_tb):
    # [FUNCTION DOC]
    # Appends an exit message to the log that includes the exception type, value and traceback
    # (all null when the body succeeded). Inputs: the exception type, value and traceback passed by
    # the caller. If the behavior is raise from exit, raises a value error. If the behavior is
    # suppress exception, returns true to request exception suppression. Otherwise returns
    # nothing, so any exception propagates.
    self.log.append("__exit__(%s, %s, %s)" % (ex_type, ex_value, ex_tb))
    if self.behavior == "raise_from_exit":
      raise ValueError("exception in __exit__")
    if self.behavior == "suppress_exception":
      return True


# Expected log when the body doesn't raise an exception.
NO_EXCEPTION_LOG = """\
__enter__()
body('var')
__exit__(None, None, None)"""

# Expected log when the body does raise an exception.  (Regular expression.)
EXCEPTION_LOG = """\
__enter__\\(\\)
body\\('var'\\)
__exit__\\(<class 'ValueError'>, Foo, <traceback object.*>\\)"""


class OpDefUtilTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class name suggests op definition utilities, but it actually tests the
  # Python context manager bridge (a naming leftover).
  # Test case exercising the native test entry point that enters a given context manager, runs
  # a body callback with the bound value and exits, using the recording fixture class.

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: a well-behaved context manager and a body that logs its argument. Action: run
    # the native entry point. Expected: the log shows entry, the body called with "var", and an
    # exit with no exception information.
    cm = TestContextManager()

    def body(var):
      # [FUNCTION DOC]
      # HELPER: body callback for testBasic. Appends a message with the received value to the
      # context manager log. Returns nothing.
      cm.log.append("body(%r)" % var)

    _py_context_manager.test_py_context_manager(cm, body)
    self.assertEqual("\n".join(cm.log), NO_EXCEPTION_LOG)

  def testBodyRaisesException(self):
    # [FUNCTION DOC]
    # Scenario: the body logs and then raises a value error with message Foo. Action: run the
    # native entry point. Expected: the same error propagates to the caller, and the log matches
    # the expected pattern showing entry, the body, and an exit that received the error type,
    # value and a traceback object.
    cm = TestContextManager()

    def body(var):
      # [FUNCTION DOC]
      # HELPER: body callback for testBodyRaisesException. Logs the received value, then raises
      # a value error with message Foo.
      cm.log.append("body(%r)" % var)
      raise ValueError("Foo")

    with self.assertRaisesRegex(ValueError, "Foo"):
      _py_context_manager.test_py_context_manager(cm, body)
    self.assertRegex("\n".join(cm.log), EXCEPTION_LOG)

  def testEnterRaisesException(self):
    # [FUNCTION DOC]
    # Scenario: a context manager whose entry raises a value error. Action: run the native entry
    # point. Expected: the error propagates, the body is never run, and the log contains only
    # the entry message (no exit is performed).
    cm = TestContextManager("raise_from_enter")

    def body(var):
      # [FUNCTION DOC]
      # HELPER: body callback for testEnterRaisesException; would log its argument but is
      # expected never to be called.
      cm.log.append("body(%r)" % var)

    with self.assertRaisesRegex(ValueError, "exception in __enter__"):
      _py_context_manager.test_py_context_manager(cm, body)
    self.assertEqual("\n".join(cm.log), "__enter__()")

  # Test behavior in unsupported case where __exit__ raises an exception.
  def testExitRaisesException(self):
    # [FUNCTION DOC]
    # Scenario (documented as an unsupported case): the context manager raises an error when
    # exiting after a successful body. Action: run the native entry point. Expected: no error
    # reaches the caller (only a warning is logged natively) and the log equals the normal
    # no-exception log.
    cm = TestContextManager("raise_from_exit")

    def body(var):
      # [FUNCTION DOC]
      # HELPER: body callback for testExitRaisesException. Logs the received value and
      # returns nothing.
      cm.log.append("body(%r)" % var)

    # Note: this does *not* raise an exception (but does log a warning):
    _py_context_manager.test_py_context_manager(cm, body)
    self.assertEqual("\n".join(cm.log), NO_EXCEPTION_LOG)

  # Test behavior in unsupported case where __exit__ suppresses exception.
  def testExitSuppressesException(self):
    # [FUNCTION DOC]
    # Scenario (documented as an unsupported case): the body raises a value error and the
    # context manager's exit asks to suppress it. Action: run the native entry point. Expected:
    # a value error stating that context managers that suppress exceptions are not supported is
    # raised, and the log matches the exception log pattern.
    cm = TestContextManager("suppress_exception")

    def body(var):
      # [FUNCTION DOC]
      # HELPER: body callback for testExitSuppressesException. Logs the received value, then
      # raises a value error with message Foo.
      cm.log.append("body(%r)" % var)
      raise ValueError("Foo")

    with self.assertRaisesRegex(
        ValueError, "tensorflow::PyContextManager::Enter does not support "
        "context managers that suppress exception"):
      _py_context_manager.test_py_context_manager(cm, body)
    self.assertRegex("\n".join(cm.log), EXCEPTION_LOG)


if __name__ == "__main__":
  googletest.main()
