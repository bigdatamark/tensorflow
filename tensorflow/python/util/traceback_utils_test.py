# [MODULE DOC]
# Unit tests for traceback_utils. Each test triggers a TensorFlow error (shape mismatch in an
# eager or compiled addition, division, argmax with a bad axis, variable construction without
# arguments) and counts the lines of the formatted traceback with filtering of internal frames
# enabled versus disabled, asserting the filtered trace stays under a limit and the unfiltered one
# exceeds a larger limit (limits differ slightly on newer Python versions). Eager execution is
# enabled when the file runs as a program, then the test main is invoked.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for traceback_utils."""

import sys
import traceback

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import traceback_utils


class TracebackUtilsTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking how much the traceback filtering shortens the stack traces of typical
  # TensorFlow errors. Shares one assertion helper that sets the filtering mode and measures the
  # traceback length.

  def assert_trace_line_count(self, fn, count, filtering_enabled=True):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests in this class.
    # Inputs: fn (callable expected to raise an exception), count (line-count threshold) and
    # filtering_enabled (bool, default true). Enables or disables traceback filtering for the
    # thread (side effect that persists after the call), asserts the reported state matches,
    # calls fn, formats the traceback of the caught exception and counts its lines (lines, not
    # frames, because autograph folds frames into large strings). Asserts at least one line was
    # recorded (so fn did raise). With filtering on, asserts the count is below the threshold
    # (at most the threshold on Python 3.13 and newer); with filtering off, asserts it is above.
    trace_line_count = -1
    if filtering_enabled:
      traceback_utils.enable_traceback_filtering()
    else:
      traceback_utils.disable_traceback_filtering()
    self.assertEqual(
        traceback_utils.is_traceback_filtering_enabled(), filtering_enabled)
    try:
      fn()
    except Exception as e:  # pylint: disable=broad-except
      # We must count lines rather than frames because autograph transforms
      # stack frames into a single large string
      trace = '\n'.join(traceback.format_tb(e.__traceback__))
      trace_line_count = len(trace.split('\n'))

    self.assertGreater(trace_line_count, 0)

    if filtering_enabled:
      if sys.version_info >= (3, 13):
        self.assertLessEqual(trace_line_count, count)
      else:
        self.assertLess(trace_line_count, count)
    else:
      self.assertGreater(trace_line_count, count)

  def test_eager_add(self):
    # [FUNCTION DOC]
    # Scenario: eagerly adding tensors of incompatible shapes. Asserts the traceback has fewer
    # than 15 lines with filtering and more than 25 without.

    def fn():
      # [FUNCTION DOC]
      # HELPER: creates a 2 by 3 and a 2 by 4 tensor of zeros and adds them; raises a shape error.
      x = array_ops.zeros((2, 3))
      y = array_ops.zeros((2, 4))
      _ = x + y

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

  def test_tfn_add(self):
    # [FUNCTION DOC]
    # Scenario: the same incompatible addition inside a compiled function. Asserts fewer than 10
    # traceback lines with filtering and more than 25 without.
    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: compiled function that adds a 2 by 3 and a 2 by 4 tensor of zeros; fails with a
      # shape error at tracing time.
      x = array_ops.zeros((2, 3))
      y = array_ops.zeros((2, 4))
      return x + y

    self.assert_trace_line_count(fn, count=10, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

  def test_tfn_div(self):
    # [FUNCTION DOC]
    # Scenario: calling a compiled function that divides its input by zero. Asserts fewer than
    # 15 traceback lines with filtering and more than 30 without.
    @def_function.function
    def wrapped_fn(x):
      # [FUNCTION DOC]
      # HELPER: compiled function returning its input divided by zero.
      return x / 0.

    def fn():
      # [FUNCTION DOC]
      # HELPER: calls the compiled division function with 0.5.
      wrapped_fn(0.5)

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=30, filtering_enabled=False)

  def test_eager_argmax(self):
    # [FUNCTION DOC]
    # Scenario: eager argmax over an out-of-range axis of a one-dimensional list. Asserts fewer
    # than 15 traceback lines with filtering and more than 30 without.
    def fn():
      # [FUNCTION DOC]
      # HELPER: calls argmax on a two-element list with axis 2, which raises an error.
      _ = math_ops.argmax([0, 1], axis=2)

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=30, filtering_enabled=False)

  def test_tfn_argmax(self):
    # [FUNCTION DOC]
    # Scenario: compiled function performing argmax over an out-of-range axis. Asserts fewer than
    # 15 traceback lines with filtering (16 allowed on Python 3.13 and newer) and more than 25
    # without.
    @def_function.function
    def wrapped_fn(x):
      # [FUNCTION DOC]
      # HELPER: compiled function returning argmax of its input along axis 2.
      return math_ops.argmax(x, axis=2)

    def fn():
      # [FUNCTION DOC]
      # HELPER: calls the compiled argmax function with a two-element list.
      wrapped_fn([0, 1])

    if sys.version_info >= (3, 13):
      self.assert_trace_line_count(fn, count=16, filtering_enabled=True)
    else:
      self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

  def test_variable_constructor(self):
    # [FUNCTION DOC]
    # Scenario: constructing a variable with no arguments, which raises an error. Asserts fewer
    # than 15 traceback lines with filtering and more than 30 without.
    def fn():
      # [FUNCTION DOC]
      # HELPER: constructs a variable without an initial value; raises an error.
      _ = variables.Variable()

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=30, filtering_enabled=False)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
