# [MODULE DOC]
# Tests for the stack ops (version 2 stack creation, push, pop and close) when compiled with XLA
# on the XLA test device. Each test builds a computation function and runs it via the XLA compile
# wrapper inside a session. Run as a test binary via the test main entry point.
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
"""Tests for tensorflow.ops.stack_ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.platform import test


class StackOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for stack operations under XLA compilation: push/pop round trips, memory swapping,
  # multiple and same-named stacks, and closing stacks. No instance state.

  def testStackPushPop(self):
    # [FUNCTION DOC]
    # Compiles a computation that creates a stack, pushes a fed float matrix and pops it back;
    # expects the popped value to equal the pushed one.
    with self.session(), self.test_scope():

      v = array_ops.placeholder(dtypes.float32)

      def fn():
        # [FUNCTION DOC]
        # HELPER: computation body compiled by the test. Creates a stack of capacity five,
        # pushes the placeholder value, and (after the push) pops and returns it.
        h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
        c = gen_data_flow_ops.stack_push_v2(h, v)
        with ops.control_dependencies([c]):
          c1 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
        return c1

      self.assertAllClose([[4.0, 5.0]],
                          xla.compile(fn)[0].eval({v: [[4.0, 5.0]]}))

  def testStackPushPopSwap(self):
    # [FUNCTION DOC]
    # Same as the push/pop test but pushes a 2000-element vector with the swap-memory option
    # enabled; expects the popped value to equal the pushed vector.
    with self.session(), self.test_scope():
      a = np.arange(2000)
      x = array_ops.placeholder(dtypes.float32)

      def fn():
        # [FUNCTION DOC]
        # HELPER: computation body compiled by the test. Creates a stack, pushes the placeholder
        # with memory swapping enabled, then pops and returns it.
        h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
        c = gen_data_flow_ops.stack_push_v2(h, x, swap_memory=True)
        with ops.control_dependencies([c]):
          return gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)

      self.assertAllClose(a, xla.compile(fn)[0].eval({x: a}))

  def testMultiStack(self):
    # [FUNCTION DOC]
    # Compiles a computation using two stacks with different names: one receives the fed value,
    # the other the constant 5.0; each is popped and the results are added. Expects 9.0 when
    # the fed value is 4.0.
    with self.session(), self.test_scope():
      v = array_ops.placeholder(dtypes.float32)

      def fn():
        # [FUNCTION DOC]
        # HELPER: computation body compiled by the test. Pushes/pops the placeholder on one
        # stack and 5.0 on another, returning the sum of the two popped values.
        h1 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
        c1 = gen_data_flow_ops.stack_push_v2(h1, v)
        with ops.control_dependencies([c1]):
          c1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
        h2 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="bar")
        c2 = gen_data_flow_ops.stack_push_v2(h2, 5.0)
        with ops.control_dependencies([c2]):
          c2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)
        return c1 + c2

      self.assertAllClose(9.0, xla.compile(fn)[0].eval({v: 4.0}))

  def testSameNameStacks(self):
    # [FUNCTION DOC]
    # Compiles a computation that creates two distinct stacks sharing the same name, pushes a
    # different fed value onto each and pops from each; runs with values 4.0 and 5.0 and expects
    # each stack to return its own value (no interference).
    """Different stacks with the same name do not interfere."""
    with self.session() as sess, self.test_scope():
      v1 = array_ops.placeholder(dtypes.float32)
      v2 = array_ops.placeholder(dtypes.float32)

      def fn():
        # [FUNCTION DOC]
        # HELPER: computation body compiled by the test. Creates two same-named stacks, pushes
        # one value onto each in sequence, pops both and returns the two popped values as a list.
        h1 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
        h2 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")

        c1 = gen_data_flow_ops.stack_push_v2(h1, v1)
        with ops.control_dependencies([c1]):
          c2 = gen_data_flow_ops.stack_push_v2(h2, v2)
        with ops.control_dependencies([c2]):
          pop1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
          pop2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)
        return [pop1, pop2]

      [pop1_compiled, pop2_compiled] = xla.compile(fn)
      out1, out2 = sess.run([pop1_compiled, pop2_compiled], {v1: 4.0, v2: 5.0})
      self.assertAllClose(out1, 4.0)
      self.assertAllClose(out2, 5.0)

  def testCloseStack(self):
    # [FUNCTION DOC]
    # Compiles and runs a computation that creates a stack and immediately closes it; expects it
    # to run without error.
    with self.session() as sess, self.test_scope():

      def fn():
        # [FUNCTION DOC]
        # HELPER: computation body compiled by the test. Creates a stack and closes it; returns
        # nothing.
        h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
        gen_data_flow_ops.stack_close_v2(h)

      sess.run(xla.compile(fn))

  def testPushCloseStack(self):
    # [FUNCTION DOC]
    # Compiles and runs a computation that creates a stack, pushes a fed value and then closes
    # the stack without popping; expects it to run without error.
    with self.session() as sess, self.test_scope():
      v = array_ops.placeholder(dtypes.float32)

      def fn():
        # [FUNCTION DOC]
        # HELPER: computation body compiled by the test. Creates a stack, pushes the placeholder
        # value, then (after the push) closes the stack; returns nothing.
        h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
        c = gen_data_flow_ops.stack_push_v2(h, v)
        with ops.control_dependencies([c]):
          gen_data_flow_ops.stack_close_v2(h)

      sess.run(xla.compile(fn), {v: [[4.0, 5.0]]})


if __name__ == "__main__":
  test.main()
