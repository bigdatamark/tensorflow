# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Tests for the experimental by-reference capture API of graphs built for compiled functions.
# By-reference captures read the current value of a Python variable each time the function is
# called, so mutating the variable between calls changes the result without a manual retrace.
# Covers direct capture, capture in a plain outer Python function, and (skipped) nested cases.
# ==============================================================================
"""Tests for detecting free vars in tf.function."""

import unittest

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class ByRefCaptureTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the graph method that captures a side input by reference.
  # No shared fixtures; each test defines its own captured variable and compiled function.

  @combinations.generate(
      combinations.combine(val_type=[int, constant_op.constant]))
  def test_direct_capture_mutation(self, val_type):
    # [FUNCTION DOC]
    # Parameterized over a plain integer type and a constant-tensor constructor (val_type).
    # Binds a variable to value one, defines a compiled function that captures it by reference
    # and returns it plus one, and asserts the result is two. Rebinds the variable to value two
    # and asserts the same function now returns three, showing the new value was picked up.
    x = val_type(1)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Inner compiled function: registers a by-reference capture named x of the outer variable
      # on the current default graph and returns the captured value plus one.
      graph = ops.get_default_graph()
      cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
      return cap_x + 1

    self.assertEqual(f(), 2)
    x = val_type(2)
    self.assertEqual(f(), 3)

  @unittest.skip("By ref capture API does not work for nested tf.function.")
  def test_capture_in_nested_function(self):
    # [FUNCTION DOC]
    # Skipped test (the by-reference API does not support nested compiled functions). Scenario:
    # an outer compiled function captures a constant tensor by reference and calls an inner
    # compiled function that captures it under another name and adds one hundred. Expects
    # the first assertion expects two, then after rebinding to two expects one hundred two.
    # KNOWN ISSUE: the first expected value (two) disagrees with the code, which would return
    # one hundred one (one plus one hundred); the test is skipped so this never runs.
    x = constant_op.constant(1)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Outer compiled function: captures the outer tensor by reference, defines an inner
      # compiled function, and returns the result of calling it.
      graph = ops.get_default_graph()
      # Capture the same x for the outer tf.function
      graph._experimental_capture_side_input_by_ref("x", lambda: x)

      @def_function.function
      def g():
        # [FUNCTION DOC]
        # Inner compiled function: captures the outer tensor by reference under the name xx and
        # returns it plus one hundred.
        graph = ops.get_default_graph()
        cap_x = graph._experimental_capture_side_input_by_ref("xx", lambda: x)
        return cap_x + 100

      return g()

    self.assertEqual(f(), 2)
    x = constant_op.constant(2)
    self.assertEqual(f(), 102)

  def test_capture_in_outer_function(self):
    # [FUNCTION DOC]
    # A plain Python helper (not itself compiled) captures an integer variable by reference on
    # the default graph and adds one; a compiled function calls the helper. Asserts the result
    # is two, then rebinds the variable to two and asserts the result becomes three.
    x = 1

    def g():
      # [FUNCTION DOC]
      # HELPER: plain Python function called from the compiled function in this test.
      # Captures the outer variable by reference under the name x on the current default graph
      # and returns the captured value plus one.
      graph = ops.get_default_graph()
      cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
      return cap_x + 1

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled function that simply returns the result of calling the helper.
      return g()

    self.assertEqual(f(), 2)
    x = 2
    self.assertEqual(f(), 3)

  @unittest.skip("By ref capture API does not work for nested tf.function.")
  def test_capture_in_outer_tf_function(self):
    # [FUNCTION DOC]
    # Skipped test (nested compiled functions are unsupported by the by-reference API). Scenario:
    # an inner compiled function captures an integer by reference and adds one; an outer
    # compiled function also registers a capture of it so it retraces, then calls the inner one.
    # Expects two, then after rebinding the variable to two, three.
    x = 1

    @def_function.function
    def g():
      # [FUNCTION DOC]
      # Inner compiled function: captures the outer variable by reference and returns it plus one.
      graph = ops.get_default_graph()
      cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
      return cap_x + 1

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Outer compiled function: registers the same by-reference capture so it retraces when the
      # value changes, then returns the result of the inner function.
      # Call `_experimental_capture_side_input_by_ref` so that the outer
      # tf.function will retrace when needed.
      graph = ops.get_default_graph()
      graph._experimental_capture_side_input_by_ref("x", lambda: x)
      return g()

    self.assertEqual(f(), 2)
    x = 2
    self.assertEqual(f(), 3)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
