# [MODULE DOC]
# Tests for while loops compiled by XLA: hand-rolled loops built with the XLA while-loop helper
# from function-defined bodies and conditions, loops with maximum iteration counts (including
# nested loops and gradients under control-flow v2), and map over tensors. Also defines a helper
# that detects compile-on-demand mode. When run as a script, sets the XLA flags environment
# variable to require a minimum cluster size of two before starting the tests.
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
# ==============================================================================
"""Tests for while loops in XLA."""

import os

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


class WhileTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case covering XLA-compiled while loops, loops with maximum iterations, nested loops with
  # gradients, and map over tensors. Inherits session and test-scope helpers from the XLA test base.

  def testSingletonLoopHandrolled(self):
    # [FUNCTION DOC]
    # Builds a loop with one integer state that counts up by one while below ten, runs it on the
    # XLA device starting from zero, and asserts the final value is ten.
    # Define a function for the loop body
    @function.Defun(dtypes.int32)
    def loop_body(step):
      # [FUNCTION DOC]
      # Loop body: takes the integer counter and returns it incremented by one.
      step_out = step + constant_op.constant(1, dtype=dtypes.int32)
      return step_out

    # Define a function for the loop condition
    @function.Defun(dtypes.int32)
    def loop_cond(step):
      # [FUNCTION DOC]
      # Loop condition: returns true while the integer counter is less than ten.
      return step < 10

    with self.session() as sess:
      init_index = array_ops.placeholder(dtypes.int32, [])
      with self.test_scope():
        loop_outputs = xla.while_loop([init_index], loop_cond, loop_body)

      result = sess.run(loop_outputs, {init_index: 0})
      self.assertAllClose(result, [10], rtol=1e-3)

  def testCountingLoopHandrolled(self):
    # [FUNCTION DOC]
    # Builds a two-state loop (integer counter, float sum) that adds one to the counter and 1.5 to
    # the sum while the counter is below ten. Asserts that starting from zero gives ten and fifteen,
    # and that starting at ten runs no iterations, leaving ten and zero.
    # Define a function for the loop body
    @function.Defun(dtypes.int32, dtypes.float32)
    def loop_body(step, rsum):
      # [FUNCTION DOC]
      # Loop body: returns the counter plus one and the running sum plus 1.5.
      step_out = step + constant_op.constant(1, dtype=dtypes.int32)
      sum_out = rsum + constant_op.constant(1.5, dtype=dtypes.float32)
      return step_out, sum_out

    # Define a function for the loop condition
    @function.Defun(dtypes.int32, dtypes.float32)
    def loop_cond(step, rsum):
      # [FUNCTION DOC]
      # Loop condition: ignores the sum and returns true while the counter is below ten.
      del rsum
      return step < 10

    with self.session() as sess:
      init_index = array_ops.placeholder(dtypes.int32, [])
      init_sum = array_ops.placeholder(dtypes.float32, [])
      with self.test_scope():
        loop_outputs = xla.while_loop([init_index, init_sum], loop_cond,
                                      loop_body)

      result = sess.run(loop_outputs, {init_index: 0, init_sum: 0.0})
      self.assertAllClose(result, [10, 15.0], rtol=1e-3)
      no_iters_result = sess.run(loop_outputs, {init_index: 10, init_sum: 0.0})
      self.assertAllClose(no_iters_result, [10, 0.0], rtol=1e-3)

  def testCountingLoopHandrolledC64(self):
    # [FUNCTION DOC]
    # Like the counting loop test but the running sum is complex64 and increments by a complex
    # constant each iteration. Asserts the sum is fifteen plus twenty imaginary after ten
    # iterations, and zero when the loop starts at the limit.
    # Define a function for the loop body
    @function.Defun(dtypes.int32, dtypes.complex64)
    def loop_body(step, rsum):
      # [FUNCTION DOC]
      # Loop body: returns the counter plus one and the complex sum plus a fixed complex constant.
      step_out = step + constant_op.constant(1, dtype=dtypes.int32)
      sum_out = rsum + constant_op.constant(1.5 + 2j, dtype=dtypes.complex64)
      return step_out, sum_out

    # Define a function for the loop condition
    @function.Defun(dtypes.int32, dtypes.complex64)
    def loop_cond(step, rsum):
      # [FUNCTION DOC]
      # Loop condition: ignores the sum and returns true while the counter is below ten.
      del rsum
      return step < 10

    with self.session() as sess:
      init_index = array_ops.placeholder(dtypes.int32, [])
      init_sum = array_ops.placeholder(dtypes.complex64, [])
      with self.test_scope():
        loop_outputs = xla.while_loop([init_index, init_sum], loop_cond,
                                      loop_body)

      result = sess.run(loop_outputs, {init_index: 0, init_sum: 0.0})
      self.assertAllClose(result[1], np.complex64(15 + 20j), rtol=1e-3)
      no_iters_result = sess.run(loop_outputs, {init_index: 10, init_sum: 0.0})
      self.assertAllClose(no_iters_result[1], np.complex64(0), rtol=1e-3)

  def testLoopWithConstantOutput(self):
    # [FUNCTION DOC]
    # Builds a loop whose second state starts at 42 and is replaced by the constant seven by the
    # body. Asserts the final results are ten and seven (a loop-invariant constant output).
    # Define a function for the loop body
    @function.Defun(dtypes.int32, dtypes.int32)
    def loop_body(step, x):
      # [FUNCTION DOC]
      # Loop body: discards the second input, returns the counter plus one and the constant seven.
      del x
      step_out = step + constant_op.constant(1, dtype=dtypes.int32)
      return (step_out, 7)

    # Define a function for the loop condition
    @function.Defun(dtypes.int32, dtypes.int32)
    def loop_cond(step, x):
      # [FUNCTION DOC]
      # Loop condition: ignores the second state and returns true while the counter is below ten.
      del x
      return step < 10

    with self.session() as sess:
      init_index = array_ops.placeholder(dtypes.int32, [])
      with self.test_scope():
        loop_outputs = xla.while_loop([init_index, 42], loop_cond, loop_body)

      result = sess.run(loop_outputs, {init_index: 0})
      self.assertAllClose(result, [10, 7], rtol=1e-3)

  def _testMaxItersSimple(self):
    # [FUNCTION DOC]
    # HELPER: shared body for the simple maximum-iterations tests (control-flow v2 variant).
    # Skips when compile-on-demand mode is on. Enters an XLA control-flow context, builds a loop
    # that always continues but is capped by a maximum iteration count equal to the size of a
    # placeholder, multiplies state by a constant each step, takes the gradient of the result with
    # respect to the constant, runs it with a three-element placeholder feed, prints the result and
    # exits the context. Nothing is asserted beyond not failing.
    if is_compile_on_demand():
      self.skipTest("list_ops are not supported in cpu_ondemand")
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      v = constant_op.constant(1.0)
      p = array_ops.placeholder(dtype=dtypes.int32)

      def create_while_loop():
        # [FUNCTION DOC]
        # HELPER: builds the capped while loop and returns the identity of its second output.
        # Maximum iterations come from the size of the placeholder defined in the enclosing test.
        iterations = array_ops.size(p, name="iterations")
        r = while_loop.while_loop(
            lambda *_: True,
            lambda i, x: (i + 1, v * x), (0, 1.0),
            maximum_iterations=iterations,
            name="outer")
        return array_ops.identity(r[1])

      output = create_while_loop()
      output = gradients_impl.gradients(output, v)[0]

      result = sess.run(output, feed_dict={p: [0, 0, 0]})
      print(result)
      xla_context.Exit()

  def testMaxItersSimple(self):
    # [FUNCTION DOC]
    # Test that is always skipped because the scenario fails with v1 control flow; the call to the
    # shared helper is commented out.
    self.skipTest("Fails with v1 control flow")
    # This fails with old control.
    # self._testMaxItersSimple()

  @test_util.enable_control_flow_v2
  def testMaxItersSimpleV2(self):
    # [FUNCTION DOC]
    # Runs the simple maximum-iterations scenario with control-flow v2 enabled by decorator.
    self._testMaxItersSimple()

  def _testNestedWhileLoopWithMaxItersFromOuterContext(self):
    # [FUNCTION DOC]
    # HELPER: shared body for the nested-loop tests. Skips under compile-on-demand. Inside an XLA
    # control-flow context builds an outer loop (five iterations) containing a middle loop whose
    # maximum iterations come from a tensor defined in the outer context, which itself contains an
    # inner loop and a gradient computation. Runs it with a three-element placeholder feed and exits
    # the context. Nothing is asserted beyond running without error.
    if is_compile_on_demand():
      self.skipTest("list_ops are not supported in cpu_ondemand")
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      v = constant_op.constant(1.0)
      p = array_ops.placeholder(dtype=dtypes.int32)

      def mid_body_builder(iterations):
        # [FUNCTION DOC]
        # HELPER: takes the iteration-limit tensor and returns the middle-loop body function.

        def mid_body(i, x):
          # [FUNCTION DOC]
          # Middle-loop body: runs an inner capped loop from the current value, then returns the
          # counter plus one and the gradient of (x plus inner result) with respect to the constant.
          r = while_loop.while_loop(
              lambda *_: True,
              lambda i, x: (i + 1, v * x), (0, x),
              maximum_iterations=iterations,
              name="inner")
          return (i + 1, gradients_impl.gradients(x + r[1], v)[0])

        return mid_body

      def outer_body(i, x):
        # [FUNCTION DOC]
        # Outer-loop body: computes the iteration limit from the placeholder size, runs the middle
        # loop, and returns the counter plus one and x plus the middle loop's second output.
        iterations = array_ops.size(p, name="iterations")
        return (i + 1, x + while_loop.while_loop(
            lambda *_: True,
            mid_body_builder(iterations), (0, x),
            maximum_iterations=iterations,
            name="mid")[1])

      def create_while_loop():
        # [FUNCTION DOC]
        # HELPER: builds the outer loop (limited to five iterations) and returns the identity of
        # its second output.
        r = while_loop.while_loop(
            lambda *_: True,
            outer_body, (0, 1.0),
            maximum_iterations=5,
            name="outer")
        return array_ops.identity(r[1])

      # p:placeholder
      # j = 0
      # i, x = 0, 1.
      # while j++ < 5:
      #   i1, x1 = 0, x
      #   while i1++ < len(p):
      #     i2, x2 = 0, x1
      #     while i2++ < len(p):
      #       x2 = v * x2
      #     x1 = grad(x1 + x2, v)
      #   x = x1
      # output = x
      output = create_while_loop()
      sess.run(output, feed_dict={p: [0, 0, 0]})
      xla_context.Exit()

  def testNestedWhileLoopWithMaxItersFromOuterContext(self):
    # [FUNCTION DOC]
    # Runs the nested-loop scenario with the default control-flow version.
    self._testNestedWhileLoopWithMaxItersFromOuterContext()

  @test_util.enable_control_flow_v2
  def testNestedWhileLoopWithMaxItersFromOuterContextV2(self):
    # [FUNCTION DOC]
    # Runs the nested-loop scenario with control-flow v2 enabled by decorator.
    self._testNestedWhileLoopWithMaxItersFromOuterContext()

  @test_util.enable_control_flow_v2
  def testMap(self):
    # [FUNCTION DOC]
    # With control-flow v2 and an XLA control-flow context, maps "add three then multiply by two"
    # over a constant of six integers and asserts the result equals the expected array. Skipped
    # under compile-on-demand.
    if is_compile_on_demand():
      self.skipTest("list_ops are not supported in cpu_ondemand")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      nums = [1, 2, 3, 4, 5, 6]
      elems = constant_op.constant(nums, name="data")
      r = map_fn.map_fn(lambda x: math_ops.multiply(math_ops.add(x, 3), 2),
                        elems)
      self.assertAllEqual(r, np.array([(x + 3) * 2 for x in nums]))
      xla_context.Exit()

  @test_util.enable_control_flow_v2
  def testMapBackPropFalse(self):
    # [FUNCTION DOC]
    # Same as the map test but disables back-propagation support in the map call. Asserts the same
    # expected values. Skipped under compile-on-demand.
    if is_compile_on_demand():
      self.skipTest("list_ops are not supported in cpu_ondemand")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      nums = [1, 2, 3, 4, 5, 6]
      elems = constant_op.constant(nums, name="data")
      r = map_fn.map_fn(
          lambda x: math_ops.multiply(math_ops.add(x, 3), 2),
          elems,
          back_prop=False)
      self.assertAllEqual(r, np.array([(x + 3) * 2 for x in nums]))
      xla_context.Exit()


def is_compile_on_demand():
  # [FUNCTION DOC]
  # HELPER: lets tests skip scenarios unsupported in compile-on-demand mode.
  # Returns true if the XLA flags environment variable is set and its text contains the
  # compile-on-demand flag name; otherwise false. Reads the environment only.
  return ("TF_XLA_FLAGS" in os.environ and
          "tf_xla_compile_on_demand" in os.environ["TF_XLA_FLAGS"])


if __name__ == "__main__":
  os.environ["TF_XLA_FLAGS"] = ("--tf_xla_min_cluster_size=2 " +
                                os.environ.get("TF_XLA_FLAGS", ""))
  test.main()
