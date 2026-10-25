# [MODULE DOC]
# Unit tests for the tf_should_use module (unused-result detection). Verifies that wrapping a
# constant tensor with the internal warning function logs an error mentioning "never used"
# (and the creating function and tensor name) when the wrapper is dropped, raises a runtime
# error for unused values inside a traced function when requested, stays silent when the
# value is used through an operator or attribute access, and that the public decorator and the
# explicit mark-used method behave accordingly. Error logging is intercepted with a mock.
# Also checks that no garbage is left behind by reference cycles. Runs as a test binary.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for tf_should_use."""

# pylint: disable=unused-import
import contextlib
import gc
import sys

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_should_use


@contextlib.contextmanager
def reroute_error():
  # [FUNCTION DOC]
  # HELPER: context manager used by the tests to capture error logging.
  # While active, replaces the error-logging function of the TensorFlow logging module (as seen
  # by the should-use module) with a mock and yields the mock so callers can inspect its call
  # arguments or assert it was not called. Restores the original on exit.
  """Temporarily reroute errors written to tf_logging.error into `captured`."""
  with test.mock.patch.object(tf_should_use.tf_logging, 'error') as error:
    yield error


class TfShouldUseTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the unused-result tracking: covers logging when a wrapped tensor is dropped,
  # raising during function tracing, silence when the value is used, the public decorator and
  # mark-used. Shares the error-capturing context manager as its fixture.

  def testAddShouldUseWarningWhenNotUsed(self):
    # [FUNCTION DOC]
    # Creates a named constant, wraps it inside a nested function with eager warning enabled and
    # discards the wrapper. Asserts the captured error message says the object was never used,
    # includes the tensor name when not eager, names the creating function, and that no
    # uncollectable garbage remains.
    c = constant_op.constant(0, name='blah0')
    def in_this_function():
      # [FUNCTION DOC]
      # HELPER: creates and immediately discards a tracked wrapper so that the creation stack
      # contains this function's name. Returns nothing.
      h = tf_should_use._add_should_use_warning(c, warn_in_eager=True)
      del h
    with reroute_error() as error:
      in_this_function()
    msg = '\n'.join(error.call_args[0])
    self.assertIn('Object was never used', msg)
    if not context.executing_eagerly():
      self.assertIn('blah0:0', msg)
    self.assertIn('in_this_function', msg)
    self.assertFalse(gc.garbage)

  def testAddShouldUseExceptionInEagerAndFunction(self):
    # [FUNCTION DOC]
    # Runs a function that wraps a constant with both eager warning and in-function error
    # enabled and drops it. In eager mode asserts an error is logged naming the function. Then
    # traces the function as a compiled function and asserts that calling it raises a runtime
    # error mentioning an unused object and the tensor name. No leftover garbage is expected.
    def in_this_function():
      # [FUNCTION DOC]
      # HELPER: creates a named constant, wraps it with tracking enabled in eager and function
      # modes and discards the wrapper without using it. Returns nothing.
      c = constant_op.constant(0, name='blah0')
      h = tf_should_use._add_should_use_warning(
          c, warn_in_eager=True, error_in_function=True)
      del h
    if context.executing_eagerly():
      with reroute_error() as error:
        in_this_function()
      msg = '\n'.join(error.call_args[0])
      self.assertIn('Object was never used', msg)
      self.assertIn('in_this_function', msg)
      self.assertFalse(gc.garbage)

    tf_fn_in_this_function = def_function.function(in_this_function)
    with self.assertRaisesRegex(RuntimeError,
                                r'Object was never used.*blah0:0'):
      tf_fn_in_this_function()
    self.assertFalse(gc.garbage)

  def _testAddShouldUseWarningWhenUsed(self, fn, name):
    # [FUNCTION DOC]
    # HELPER: shared body for the "used" tests. Inputs: fn (callable applied to the wrapper to
    # use it in some way) and name (name for the constant). Wraps a constant with eager warning
    # on, applies fn, drops the wrapper and asserts that no error was logged.
    c = constant_op.constant(0, name=name)
    with reroute_error() as error:
      h = tf_should_use._add_should_use_warning(c, warn_in_eager=True)
      fn(h)
      del h
    error.assert_not_called()

  def testAddShouldUseWarningWhenUsedWithAdd(self):
    # [FUNCTION DOC]
    # Uses the wrapper by adding one to it and asserts via the shared helper that no error is
    # logged; then collects garbage and asserts none is left.
    def add(h):
      # [FUNCTION DOC]
      # HELPER: usage action; adds one to the wrapper and discards the result.
      _ = h + 1
    self._testAddShouldUseWarningWhenUsed(add, name='blah_add')
    gc.collect()
    self.assertFalse(gc.garbage)

  def testAddShouldUseWarningWhenUsedWithGetShape(self):
    # [FUNCTION DOC]
    # Uses the wrapper by reading its shape attribute and asserts no error is logged; then
    # collects garbage and asserts none is left.
    def get_shape(h):
      # [FUNCTION DOC]
      # HELPER: usage action; reads the shape attribute of the wrapper and discards it.
      _ = h.shape
    self._testAddShouldUseWarningWhenUsed(get_shape, name='blah_get_name')
    gc.collect()
    self.assertFalse(gc.garbage)

  def testShouldUseResult(self):
    # [FUNCTION DOC]
    # Decorates a function that returns a named constant with the public decorator (eager
    # warning on), calls it and ignores the result. Asserts the logged message says the object
    # was never used, includes the tensor name when not eager and names the decorated function;
    # then asserts no garbage remains.
    @tf_should_use.should_use_result(warn_in_eager=True)
    def return_const(value):
      # [FUNCTION DOC]
      # HELPER: decorated sample; returns a constant tensor with the given value, named blah2.
      return constant_op.constant(value, name='blah2')
    with reroute_error() as error:
      return_const(0.0)
    msg = '\n'.join(error.call_args[0])
    self.assertIn('Object was never used', msg)
    if not context.executing_eagerly():
      self.assertIn('blah2:0', msg)
    self.assertIn('return_const', msg)
    gc.collect()
    self.assertFalse(gc.garbage)

  def testShouldUseResultWhenNotReallyUsed(self):
    # [FUNCTION DOC]
    # Calls a decorated function and ignores its result inside a test session, then creates and
    # evaluates an unrelated constant. Asserts the unrelated activity does not count as using
    # the ignored result: the never-used error is still logged with the tensor name (non-eager)
    # and function name, and no garbage remains.
    @tf_should_use.should_use_result(warn_in_eager=True)
    def return_const(value):
      # [FUNCTION DOC]
      # HELPER: decorated sample; returns a constant tensor with the given value, named blah3.
      return constant_op.constant(value, name='blah3')
    with reroute_error() as error:
      with self.cached_session():
        return_const(0.0)
        # Creating another op and executing it does not mark the
        # unused op as being "used".
        v = constant_op.constant(1.0, name='meh')
        self.evaluate(v)
    msg = '\n'.join(error.call_args[0])
    self.assertIn('Object was never used', msg)
    if not context.executing_eagerly():
      self.assertIn('blah3:0', msg)
    self.assertIn('return_const', msg)
    gc.collect()
    self.assertFalse(gc.garbage)

  # Tests that mark_used is available in the API.
  def testMarkUsed(self):
    # [FUNCTION DOC]
    # Verifies the mark-used method exists on the decorated function's result: calls the
    # decorated function in a session and invokes mark-used on the result; expects no exception.
    @tf_should_use.should_use_result(warn_in_eager=True)
    def return_const(value):
      # [FUNCTION DOC]
      # HELPER: decorated sample; returns a constant tensor with the given value, named blah3.
      return constant_op.constant(value, name='blah3')

    with self.cached_session():
      return_const(0.0).mark_used()

if __name__ == '__main__':
  test.main()
