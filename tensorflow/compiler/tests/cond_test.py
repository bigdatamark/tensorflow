# [MODULE DOC]
# Tests for conditional and switch-case control flow (cond, switch_case) compiled by XLA, mostly
# combined with tensor arrays and compile-time-constant propagation. Many cases are skipped with
# bug references. Defines one control-flow-v2 test class; runs the standard test main as a script.
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
# ==============================================================================
"""Tests for tf.cond in XLA."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import session
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


@test_util.with_control_flow_v2
class CondTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case (run with control flow v2) for cond and switch-case ops, tensor arrays inside
  # branches, and constant propagation of branch inputs. Each test manually enters and exits an
  # XLA control-flow context.

  def testCondAndTensorArrayInDefun(self):
    # [FUNCTION DOC]
    # Inside an XLA control-flow context, a defined function creates a one-element float tensor
    # array and uses cond with a true predicate to write 5 or 10 at index 0, returning the stacked
    # array. Asserts the result is a list containing 5. Returns nothing. Known follow-up bugs are
    # noted in a comment as making the test not very useful.
    # TODO(b/132430685): Make test more useful. Also b/129396295, b/127846988
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # Nested function under test: creates a one-element float tensor array, writes 5 (true
        # branch) or 10 (false branch) at index 0 via cond on a true constant and returns the stacked
        # array.
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = cond.cond(
            constant_op.constant(True),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))

        return output.stack()

      output_t = f()
      self.assertAllEqual([5.], self.evaluate(output_t))

      xla_context.Exit()

  def testCondAndTensorArrayInDefun_constFolding(self):
    # [FUNCTION DOC]
    # Same as the previous tensor-array cond test but built in a fresh graph and session with a
    # false constant predicate, exercising constant folding. Asserts the stacked result is a list
    # containing 10. Returns nothing.
    g = ops.Graph()
    with session.Session(graph=g), g.as_default(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # Nested function under test: tensor array written with 5 or 10 depending on a constant
        # false predicate; returns the stacked array.
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = cond.cond(
            constant_op.constant(False),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))

        return output.stack()

      output_t = f()
      self.assertAllEqual([10.], self.evaluate(output_t))

      xla_context.Exit()

  def testCondAndTensorArray_xlaCompile(self):
    # [FUNCTION DOC]
    # Skipped immediately (bug reference) because it fails with an uninitialized-arguments error
    # in the XLA if-op compilation. The remaining body shows the intent: compile a function using
    # cond with a tensor array via the XLA compile API and assert the result is a list containing
    # 5. Returns nothing.
    self.skipTest("b/127846988")
    # Fails with "Uninitialized arguments" in XlaIfOp::Compile
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      def f():
        # [FUNCTION DOC]
        # Nested function: builds a tensor array and returns the stacked result of cond writing 5 or
        # 10 depending on a true constant. Intended to be compiled by XLA.
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = cond.cond(
            constant_op.constant(True),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))

        return output.stack()

      output_t, = xla.compile(f)
      self.assertAllEqual([5.], self.evaluate(output_t))

      xla_context.Exit()

  def testCondConstPropagation(self):
    # [FUNCTION DOC]
    # Verifies that inside cond, a strided slice indexed by a placeholder is treated as a
    # compile-time constant. Feeds a vector and index 1 and asserts the output is 1 (the true
    # branch indexes the vector; the false branch returns 5). Returns nothing.
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = array_ops.placeholder(dtypes.int32)

      # TODO(b/129021699): Wrapping this in a tf.function does not work.
      def if_true():
        # [FUNCTION DOC]
        # True branch: returns the element of the placeholder vector at the placeholder index,
        # which produces a strided slice needing a compile-time constant index.
        # This emits a StridedSlice op which expects the index to be a
        # compile-time const.
        return x[p]

      def if_false():
        # [FUNCTION DOC]
        # False branch: returns the constant 5.
        return 5.

      output = cond.cond(
          constant_op.constant(True), if_true, if_false)

      self.assertAllEqual(1.,
                          sess.run(output, feed_dict={
                              x: [0., 1., 2.],
                              p: 1
                          }))

      xla_context.Exit()

  def testCondConstPropagation_xlaCompile(self):
    # [FUNCTION DOC]
    # Skipped immediately (bug reference). Intended: compile via the XLA compile API a function
    # with cond whose true branch indexes a default-valued vector with a constant index and whose
    # false branch returns 5; asserts the output is 1. Returns nothing.
    self.skipTest("b/132430685")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder_with_default([0., 1., 2.], shape=[3])
      p = constant_op.constant(1)

      def f():
        # [FUNCTION DOC]
        # Nested function: returns cond over a true constant selecting the indexing branch or the
        # constant branch.
        # TODO(b/129021699): Wrapping this in a tf.function does not work.
        def if_true():
          # [FUNCTION DOC]
          # True branch: returns the vector element at the constant index.
          # This emits a StridedSlice op which expects the index to be a
          # compile-time const.
          return x[p]

        def if_false():
          # [FUNCTION DOC]
          # False branch: returns the constant 5.
          return 5.

        return cond.cond(
            constant_op.constant(True), if_true, if_false)

      output = xla.compile(f)

      self.assertAllEqual(1., self.evaluate(output))

      xla_context.Exit()

  def testCondConstPropagation_errorMsg(self):
    # [FUNCTION DOC]
    # Skipped immediately (bug reference). Intended: the slice bound in the true branch comes from
    # a random integer, which is not a compile-time constant, so running must raise an
    # invalid-argument error mentioning that it must be a compile-time constant. Returns nothing.
    self.skipTest("b/132430685")
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = random_ops.random_uniform([], minval=1, maxval=3, dtype=dtypes.int32)

      # TODO(b/129021699): Wrapping this in a tf.function does not work.
      def if_true():
        # [FUNCTION DOC]
        # True branch: returns a prefix slice of the placeholder vector whose length is a random
        # (non-constant) scalar.
        # This emits a StridedSlice op which expects the index to be a
        # compile-time const.
        return x[:p]

      def if_false():
        # [FUNCTION DOC]
        # False branch: returns a vector of 5 whose length is the random scalar.
        return array_ops.fill([p], 5.)

      output = cond.cond(
          constant_op.constant(True), if_true, if_false)

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must be a compile-time constant"):
        sess.run(
            output, feed_dict={
                x: [0., 1., 2.],
            })

      xla_context.Exit()

  def testCondConstPropagation_errorMsg_xlaCompile(self):
    # [FUNCTION DOC]
    # Like the skipped error-message test, but compiled with the XLA compile API and a random
    # boolean predicate. Running the compiled output with a fed vector is expected to raise an
    # invalid-argument error saying the value must be a compile-time constant. Returns nothing.
    # Unlike its siblings this one is not skipped.
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = random_ops.random_uniform([], minval=1, maxval=3, dtype=dtypes.int32)
      condition = math_ops.cast(
          random_ops.random_uniform([], minval=0, maxval=2, dtype=dtypes.int32),
          dtypes.bool)

      def f():
        # [FUNCTION DOC]
        # Nested function returning cond over the random predicate with the slicing or filling
        # branches.
        # TODO(b/129021699): Wrapping this in a tf.function does not work.
        def if_true():
          # [FUNCTION DOC]
          # True branch: prefix slice of the vector with a random non-constant length.
          # This emits a StridedSlice op which expects the index to be a
          # compile-time const.
          return x[:p]

        def if_false():
          # [FUNCTION DOC]
          # False branch: vector filled with 5 whose length is the random scalar.
          return array_ops.fill([p], 5.)

        return cond.cond(condition, if_true, if_false)

      output = xla.compile(f)

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must be a compile-time constant"):
        sess.run(
            output, feed_dict={
                x: [0., 1., 2.],
            })

      xla_context.Exit()

  def testSwitchCaseAndTensorArrayInDefun(self):
    # [FUNCTION DOC]
    # Skipped immediately (bug reference). Intended: inside a defined function, switch-case on
    # index 1 over three branches writing 5, 10 or 15 into a tensor array; asserts the stacked
    # result is a list containing 10. Returns nothing.
    self.skipTest("b/127846988")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # Nested function: tensor array written by the switch-case branch chosen by constant
        # index 1; returns the stacked array.
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_switch_case.switch_case(
            constant_op.constant(1), {
                0: lambda: ta.write(0, 5.),
                1: lambda: ta.write(0, 10.),
                2: lambda: ta.write(0, 15.),
            })

        return output.stack()

      output_t = f()
      self.assertAllEqual([10.], self.evaluate(output_t))

      xla_context.Exit()

  def testSwitchCaseAndTensorArray_xlaCompile(self):
    # [FUNCTION DOC]
    # Skipped immediately (bug reference). Intended: same switch-case tensor array scenario
    # compiled with the XLA compile API; asserts the result is a list containing 10. Returns
    # nothing.
    self.skipTest("b/127846988")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      def f():
        # [FUNCTION DOC]
        # Nested function: switch-case on constant 1 over three tensor-array writes; returns the
        # stacked array.
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_switch_case.switch_case(
            constant_op.constant(1), {
                0: lambda: ta.write(0, 5.),
                1: lambda: ta.write(0, 10.),
                2: lambda: ta.write(0, 15.),
            })

        return output.stack()

      output_t, = xla.compile(f)
      self.assertAllEqual([10.], self.evaluate(output_t))

      xla_context.Exit()

  def testSwitchCaseConstPropagation(self):
    # [FUNCTION DOC]
    # Skipped immediately (bug reference). Intended: switch-case with index 2 selecting a branch
    # that indexes a placeholder vector with a placeholder index; with the vector 0, 1, 7 and
    # index 2 the output must be 7. Returns nothing.
    self.skipTest("b/127846988")
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = array_ops.placeholder(dtypes.int32)

      def branch0():
        # [FUNCTION DOC]
        # Branch zero: returns the constant 5.
        return 5.

      def branch1():
        # [FUNCTION DOC]
        # Branch one: returns the constant 15.
        return 15.

      # TODO(b/129021699): Wrapping this in a tf.function does not work.
      def branch2():
        # [FUNCTION DOC]
        # Branch two: returns the placeholder vector element at the placeholder index (strided
        # slice needing a compile-time constant index).
        # This emits a StridedSlice op which expects the index to be a
        # compile-time const.
        return x[p]

      output = control_flow_switch_case.switch_case(
          constant_op.constant(2), {
              0: branch0,
              1: branch1,
              2: branch2,
          })

      self.assertAllEqual(7.,
                          sess.run(output, feed_dict={
                              x: [0., 1., 7.],
                              p: 2,
                          }))

      xla_context.Exit()

  def testCondNoInputs(self):
    # [FUNCTION DOC]
    # Regression test for a failed-precondition error about expecting one input shape when cond
    # branches take no inputs. For predicates true and false (via placeholders with default
    # values) builds cond returning 2 or 1 and asserts the value equals one plus the predicate as
    # integer. Returns nothing.
    """Verifies against `Failed precondition: Expected one input shape`."""

    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      for pred in True, False:
        cond_out = cond.cond(
            array_ops.placeholder_with_default(pred, []),
            lambda: constant_op.constant(2.),
            lambda: constant_op.constant(1.))
        self.assertEqual(int(pred) + 1., self.evaluate(cond_out))

      xla_context.Exit()


if __name__ == '__main__':
  test.main()
