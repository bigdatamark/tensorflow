# [MODULE DOC]
# Unit tests for the stack operations in the data flow ops: the resource-handle variants (create
# stack, push, pop, close) and the deprecated reference-handle variants. Tests cover push followed
# by pop, pushing with memory swapping to host, stacks used inside while loops, multiple
# independent stacks, stacks sharing a name, and closing stacks. Each scenario is written as a
# private helper that takes a flag for running on a GPU-enabled session, and a public test that
# calls the helper once with the flag off and once with it on. Some tests are restricted to
# graph-mode (version 1) execution. There are no import-time side effects other than the standard
# test entry point.
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
"""Tests for tensorflow.ops.stack_ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


class StackOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the resource-based stack operations (stack creation with a handle, push, pop,
  # close). Holds no state; each helper builds its own graph pieces in a cached session and checks
  # results with numeric comparison.

  def _testStackPushPop(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testStackPushPop.
    # Takes a flag for whether the session may use a GPU. Creates an unbounded float stack named
    # foo, pushes a one by two matrix of values, then (control-ordered after the push) pops it and
    # asserts the popped value equals the pushed one. Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, [[4.0, 5.0]])
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
      self.assertAllClose([[4.0, 5.0]], self.evaluate(c1))

  @test_util.run_deprecated_v1
  def testStackPushPop(self):
    # [FUNCTION DOC]
    # Test: runs the push-then-pop scenario once without and once with GPU allowed. Runs under the
    # deprecated version 1 mode decorator.
    self._testStackPushPop(use_gpu=False)
    self._testStackPushPop(use_gpu=True)

  def _testStackPushPopSwap(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testStackPushPopSwap.
    # Takes a GPU-allowed flag. Pushes a float vector of two thousand elements with the swap-memory
    # option enabled (letting the stack move the data out of device memory), pops it back with control
    # ordering and asserts the values are unchanged. Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      a = np.arange(2000)
      x = constant_op.constant(a, dtype=dtypes.float32)
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, x, swap_memory=True)
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
      self.assertAllClose(a, self.evaluate(c1))

  @test_util.run_deprecated_v1
  def testStackPushPopSwap(self):
    # [FUNCTION DOC]
    # Test: runs the push-pop with memory swapping scenario with GPU disallowed and then allowed,
    # in deprecated version 1 mode.
    self._testStackPushPopSwap(use_gpu=False)
    self._testStackPushPopSwap(use_gpu=True)

  def _testStackWhileSwap(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testStackWhileSwap.
    # Takes a GPU-allowed flag. A first while loop runs ten iterations, each pushing a vector of
    # two thousand ones onto one stack with memory swapping and incrementing a counter. A second
    # while loop counts the counter down, popping one vector per iteration and accumulating the sum.
    # Asserts the sum equals ten for every element. Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      n = constant_op.constant(0)
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")

      def c(x):
        # [FUNCTION DOC]
        # HELPER: loop condition for the push loop in _testStackWhileSwap.
        # Takes the loop counter and returns a boolean tensor that is true while the counter is below ten.
        return math_ops.less(x, 10)

      def b(x):
        # [FUNCTION DOC]
        # HELPER: loop body for the push loop in _testStackWhileSwap.
        # Takes the loop counter. Creates a vector of two thousand ones, pushes it onto the enclosing
        # stack with memory swapping (ordered after the counter), and returns the counter plus one,
        # ordered after the push. Side effect: adds push operations to the graph.
        with ops.control_dependencies([x]):
          a = constant_op.constant(np.ones(2000), dtype=dtypes.float32)
          v = gen_data_flow_ops.stack_push_v2(h, a, swap_memory=True)
        with ops.control_dependencies([v]):
          return math_ops.add(x, 1)

      r = while_loop.while_loop(c, b, [n])

      v = constant_op.constant(np.zeros(2000), dtype=dtypes.float32)

      def c1(x, y):
        # [FUNCTION DOC]
        # HELPER: loop condition for the pop loop in _testStackWhileSwap.
        # Takes the counter and an accumulator (the accumulator is ignored) and returns a boolean tensor
        # that is true while the counter is greater than zero.
        del y
        return math_ops.greater(x, 0)

      def b1(x, y):
        # [FUNCTION DOC]
        # HELPER: loop body for the pop loop in _testStackWhileSwap.
        # Takes the counter and the accumulator. Returns a list holding the counter minus one and the
        # accumulator plus a vector popped from the enclosing stack. Side effect: adds pop operations to
        # the graph.
        nx = math_ops.subtract(x, 1)
        ny = y + gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
        return [nx, ny]

      _, ry = while_loop.while_loop(
          c1, b1, [r, v],
          [r.get_shape(), tensor_shape.unknown_shape()])
      self.assertAllClose(np.ones(2000) * 10.0, self.evaluate(ry))

  @test_util.run_v1_only("b/120545219")
  def testStackWhileSwap(self):
    # [FUNCTION DOC]
    # Test: runs the while-loop push and pop scenario without and with GPU. Restricted to graph-mode
    # (version 1) execution, with a bug-tracker reference given as the reason.
    self._testStackWhileSwap(use_gpu=False)
    self._testStackWhileSwap(use_gpu=True)

  def _testMultiStack(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testMultiStack.
    # Takes a GPU-allowed flag. Creates two stacks with different names, pushes four onto the first
    # and five onto the second, pops each back (control-ordered) and asserts the sum of the popped
    # values is nine, showing the stacks are independent. Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_push_v2(h1, 4.0)
      with ops.control_dependencies([c1]):
        c1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
      h2 = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="bar")
      c2 = gen_data_flow_ops.stack_push_v2(h2, 5.0)
      with ops.control_dependencies([c2]):
        c2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)
      r = c1 + c2
      self.assertAllClose(9.0, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testMultiStack(self):
    # [FUNCTION DOC]
    # Test: runs the two independent stacks scenario with GPU disallowed then allowed, in deprecated
    # version 1 mode.
    self._testMultiStack(use_gpu=False)
    self._testMultiStack(use_gpu=True)

  def _testSameNameStacks(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testSameNameStacks.
    # Takes a GPU-allowed flag. Creates two resource stacks with the same name, pushes four onto the
    # first and five onto the second (ordered), then pops from each and asserts the first returns four
    # and the second five, so same-named stacks do not interfere. Returns nothing.
    """Different stacks with the same name do not interfere."""
    with self.cached_session(use_gpu=use_gpu) as sess:
      h1 = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      h2 = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")

      c1 = gen_data_flow_ops.stack_push_v2(h1, 4.0)
      with ops.control_dependencies([c1]):
        c2 = gen_data_flow_ops.stack_push_v2(h2, 5.0)
      with ops.control_dependencies([c2]):
        pop1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
        pop2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)

      out1, out2 = self.evaluate([pop1, pop2])
      self.assertAllClose(out1, 4.0)
      self.assertAllClose(out2, 5.0)

  @test_util.run_deprecated_v1
  def testSameNameStacks(self):
    # [FUNCTION DOC]
    # Test: runs the same-name stacks scenario with GPU disallowed and allowed, in deprecated version
    # 1 mode.
    self._testSameNameStacks(use_gpu=False)
    self._testSameNameStacks(use_gpu=True)

  def _testCloseStack(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testCloseStack.
    # Takes a GPU-allowed flag. Creates a stack and immediately closes it, evaluating the close
    # operation; passes if no error occurs. Returns nothing.
    with self.cached_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_close_v2(h)
      self.evaluate(c1)

  @test_util.run_deprecated_v1
  def testCloseStack(self):
    # [FUNCTION DOC]
    # Test: runs the create-and-close scenario with GPU disallowed and allowed, in deprecated version 1
    # mode.
    self._testCloseStack(use_gpu=False)
    self._testCloseStack(use_gpu=True)

  def _testPushCloseStack(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testPushCloseStack.
    # Takes a GPU-allowed flag. Creates a stack, pushes a one by two matrix, then closes it
    # (control-ordered after the push) and evaluates the close; passes if no error occurs. Returns
    # nothing.
    with self.cached_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops.stack_v2(
          -1, elem_type=dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, [[4.0, 5.0]])
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_close_v2(h)
      self.evaluate(c1)

  @test_util.run_deprecated_v1
  def testPushCloseStack(self):
    # [FUNCTION DOC]
    # Test: runs the push-then-close scenario with GPU disallowed and allowed, in deprecated version 1
    # mode.
    self._testPushCloseStack(use_gpu=False)
    self._testPushCloseStack(use_gpu=True)


class StackOpRefTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the deprecated reference-handle variant of the stack operations (stack creation
  # returning a reference handle, with push, pop and close operations that take it). Mirrors the
  # resource-based test case with the same scenarios. Holds no state; helpers build graph pieces in
  # a cached session.
  """Tests for deprecated non-resource variant of stack ops."""

  def _testStackPushPop(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testStackPushPop.
    # Takes a GPU-allowed flag. Creates a float stack named foo using the reference variant, pushes a
    # one by two matrix, pops it (control-ordered) and asserts the popped value equals the pushed one.
    # Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push(h, [[4.0, 5.0]])
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop(h, dtypes.float32)
      self.assertAllClose([[4.0, 5.0]], self.evaluate(c1))

  @test_util.run_deprecated_v1
  def testStackPushPop(self):
    # [FUNCTION DOC]
    # Test: runs the reference-variant push-then-pop scenario with GPU disallowed and allowed, in
    # deprecated version 1 mode.
    self._testStackPushPop(use_gpu=False)
    self._testStackPushPop(use_gpu=True)

  def _testStackPushPopSwap(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testStackPushPopSwap.
    # Takes a GPU-allowed flag. Pushes a float vector of two thousand elements with memory swapping
    # onto a reference stack, pops it back (control-ordered) and asserts equality with the input.
    # Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      a = np.arange(2000)
      x = constant_op.constant(a, dtype=dtypes.float32)
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push(h, x, swap_memory=True)
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop(h, dtypes.float32)
      self.assertAllClose(a, self.evaluate(c1))

  @test_util.run_deprecated_v1
  def testStackPushPopSwap(self):
    # [FUNCTION DOC]
    # Test: runs the reference-variant push-pop with swapping scenario with GPU disallowed and
    # allowed, in deprecated version 1 mode.
    self._testStackPushPopSwap(use_gpu=False)
    self._testStackPushPopSwap(use_gpu=True)

  def _testMultiStack(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testMultiStack.
    # Takes a GPU-allowed flag. Two reference stacks with different names get four and five pushed
    # and popped back (control-ordered); asserts the sum of popped values is nine. Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_push(h1, 4.0)
      with ops.control_dependencies([c1]):
        c1 = gen_data_flow_ops.stack_pop(h1, dtypes.float32)
      h2 = gen_data_flow_ops._stack(dtypes.float32, stack_name="bar")
      c2 = gen_data_flow_ops.stack_push(h2, 5.0)
      with ops.control_dependencies([c2]):
        c2 = gen_data_flow_ops.stack_pop(h2, dtypes.float32)
      r = c1 + c2
      self.assertAllClose(9.0, self.evaluate(r))

  def _testStackWhileSwap(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testStackWhileSwap.
    # Takes a GPU-allowed flag. Same scenario as the resource variant using a reference stack: ten
    # iterations of a loop push vectors of two thousand ones with memory swapping, then a second loop
    # pops them while counting down, accumulating the sum; asserts every element sums to ten. Returns
    # nothing.
    with self.cached_session(use_gpu=use_gpu):
      n = constant_op.constant(0)
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")

      def c(x):
        # [FUNCTION DOC]
        # HELPER: loop condition for the push loop in _testStackWhileSwap.
        # Takes the counter and returns a boolean tensor that is true while the counter is below ten.
        return math_ops.less(x, 10)

      def b(x):
        # [FUNCTION DOC]
        # HELPER: loop body for the push loop in _testStackWhileSwap.
        # Takes the counter, pushes a vector of two thousand ones onto the enclosing reference stack with
        # memory swapping (ordered after the counter) and returns the counter plus one, ordered after the
        # push. Side effect: adds push operations to the graph.
        with ops.control_dependencies([x]):
          a = constant_op.constant(np.ones(2000), dtype=dtypes.float32)
          v = gen_data_flow_ops.stack_push(h, a, swap_memory=True)
        with ops.control_dependencies([v]):
          return math_ops.add(x, 1)

      r = while_loop.while_loop(c, b, [n])

      v = constant_op.constant(np.zeros(2000), dtype=dtypes.float32)

      def c1(x, y):
        # [FUNCTION DOC]
        # HELPER: loop condition for the pop loop in _testStackWhileSwap.
        # Takes the counter and an unused accumulator and returns a boolean tensor that is true while the
        # counter is greater than zero.
        del y
        return math_ops.greater(x, 0)

      def b1(x, y):
        # [FUNCTION DOC]
        # HELPER: loop body for the pop loop in _testStackWhileSwap.
        # Takes the counter and accumulator; returns a list of the counter minus one and the accumulator
        # plus a vector popped from the enclosing reference stack. Side effect: adds pop operations to the
        # graph.
        nx = math_ops.subtract(x, 1)
        ny = y + gen_data_flow_ops.stack_pop(h, dtypes.float32)
        return [nx, ny]

      _, ry = while_loop.while_loop(
          c1, b1, [r, v],
          [r.get_shape(), tensor_shape.unknown_shape()])
      self.assertAllClose(np.ones(2000) * 10.0, self.evaluate(ry))

  @test_util.run_v1_only("b/120545219")
  def testStackWhileSwap(self):
    # [FUNCTION DOC]
    # Test: runs the reference-variant while-loop scenario with GPU disallowed and allowed. Restricted
    # to graph-mode (version 1) execution, with a bug-tracker reference given as the reason.
    self._testStackWhileSwap(use_gpu=False)
    self._testStackWhileSwap(use_gpu=True)

  @test_util.run_deprecated_v1
  def testMultiStack(self):
    # [FUNCTION DOC]
    # Test: runs the reference-variant two independent stacks scenario with GPU disallowed and
    # allowed, in deprecated version 1 mode.
    self._testMultiStack(use_gpu=False)
    self._testMultiStack(use_gpu=True)

  def _testSameNameStacks(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testSameNameStacks.
    # Takes a GPU-allowed flag. Creates two reference stacks with the same name and pushes to each.
    # Unlike the resource variant, it asserts the second element of each evaluated handle (the unique
    # stack identifier) differs between the two, showing same-named stacks are distinct. Returns
    # nothing.
    with self.cached_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_push(h1, 4.0)
      h2 = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c2 = gen_data_flow_ops.stack_push(h2, 5.0)
      _ = c1 + c2
      self.assertNotEqual(self.evaluate(h1)[1], self.evaluate(h2)[1])

  @test_util.run_deprecated_v1
  def testSameNameStacks(self):
    # [FUNCTION DOC]
    # Test: runs the reference-variant same-name scenario with GPU disallowed and allowed, in
    # deprecated version 1 mode.
    self._testSameNameStacks(use_gpu=False)
    self._testSameNameStacks(use_gpu=True)

  def _testCloseStack(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testCloseStack.
    # Takes a GPU-allowed flag. Creates a reference stack and closes it, evaluating the close
    # operation; passes if no error occurs. Returns nothing.
    with self.cached_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_close(h)
      self.evaluate(c1)

  @test_util.run_deprecated_v1
  def testCloseStack(self):
    # [FUNCTION DOC]
    # Test: runs the reference-variant create-and-close scenario with GPU disallowed and allowed, in
    # deprecated version 1 mode.
    self._testCloseStack(use_gpu=False)
    self._testCloseStack(use_gpu=True)

  def _testPushCloseStack(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared implementation for testPushCloseStack.
    # Takes a GPU-allowed flag. Creates a reference stack, pushes a one by two matrix and closes it
    # (control-ordered after the push); passes if no error occurs. Returns nothing.
    with self.cached_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._stack(dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push(h, [[4.0, 5.0]])
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_close(h)
      self.evaluate(c1)

  @test_util.run_deprecated_v1
  def testPushCloseStack(self):
    # [FUNCTION DOC]
    # Test: runs the reference-variant push-then-close scenario with GPU disallowed and allowed, in
    # deprecated version 1 mode.
    self._testPushCloseStack(use_gpu=False)
    self._testPushCloseStack(use_gpu=True)


if __name__ == "__main__":
  test.main()
