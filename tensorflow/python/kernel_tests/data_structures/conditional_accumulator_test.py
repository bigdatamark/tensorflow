# [MODULE DOC]
# Unit tests for the ConditionalAccumulator op (data_flow_ops.ConditionalAccumulator), mostly in
# graph mode. Covers constructor validation, dtype support, applying and taking gradients with
# mean or sum reduction, static and dynamic shape checking, stale-gradient rejection through
# the global step, size tracking, multi-threaded apply/take, and cancellation by closing the
# session. Not a helper module; no import-time side effects beyond imports.
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

import time

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# from functools import reduce


class ConditionalAccumulatorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for ConditionalAccumulator. Each test creates one or more accumulators (named "Q"
  # with a float dtype and usually shape of a single element) in a cached session, applies
  # gradients with a local step, takes averaged or summed gradients, and checks values, counts and
  # errors. Several threaded tests are v1-only and switch the graph to thread-local state first.
  # Holds no instance state; one private method is a thread helper.

  def testConstructorWithInvalidArg(self):
    # [FUNCTION DOC]
    # Test: constructing an accumulator with an unknown reduction type string must raise a value
    # error.
    with ops.Graph().as_default():
      with self.assertRaises(ValueError):
        data_flow_ops.ConditionalAccumulator(
            dtypes_lib.float32, name="Q", reduction_type="Invalid")

  @test_util.run_deprecated_v1
  def testAccumulatorSizeEmpty(self):
    # [FUNCTION DOC]
    # Test: a freshly created accumulator reports zero accumulated gradients.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(dtypes_lib.float32, name="Q")
      self.assertEqual(q.num_accumulated().eval(), 0)

  @test_util.run_deprecated_v1
  def testAccumulatorSetGlobalStep(self):
    # [FUNCTION DOC]
    # Test: running the set-global-step op with value 1 on an accumulator succeeds without error.
    # Nothing else is asserted.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      set_global_step_op = q.set_global_step(1)
      set_global_step_op.run()

  @test_util.run_deprecated_v1
  def testAccumulatorApplyGradFloat32(self):
    # [FUNCTION DOC]
    # Test: applying one float gradient of matching shape runs successfully. Nothing is asserted
    # beyond absence of errors.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      accum_op = q.apply_grad((10.0,))
      accum_op.run()

  @test_util.run_deprecated_v1
  def testDtypes(self):
    # [FUNCTION DOC]
    # Test: for half, single and double precision floats, creates an accumulator, applies ten
    # gradients (0..9) and takes the mean, asserting it equals the arithmetic mean of the inputs.
    # The session variable obtained from the context manager is unused.
    with self.cached_session() as sess:
      dtypes = [dtypes_lib.float16, dtypes_lib.float32, dtypes_lib.float64]

      for i in range(len(dtypes)):
        dtype = dtypes[i]
        q = data_flow_ops.ConditionalAccumulator(
            dtype, shape=tensor_shape.TensorShape([1]))

        elems = np.arange(10).astype(dtype.as_numpy_dtype)
        for e in elems:
          q.apply_grad((e,)).run()

        result = self.evaluate(q.take_grad(1))

        self.assertEqual(sum(elems) / len(elems), result)

  @test_util.run_deprecated_v1
  def testAccumulatorMultipleAccumulators(self):
    # [FUNCTION DOC]
    # Test: four accumulators with the same name but different dtype/instances each get a distinct
    # gradient applied; taking one gradient from each returns the value applied to that very
    # accumulator, showing the accumulators do not share state.
    with self.cached_session():
      q_f32_0 = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      q_f32_1 = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      q_f16_0 = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float16, name="Q", shape=tensor_shape.TensorShape([1]))
      q_f16_1 = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float16, name="Q", shape=tensor_shape.TensorShape([1]))

      accums = [q_f16_0, q_f16_1, q_f32_0, q_f32_1]
      for i in range(len(accums)):
        accums[i].apply_grad((i + 10.0,)).run()

      for i in range(len(accums)):
        result = accums[i].take_grad(1).eval()
        self.assertEqual(result, i + 10.0)

  @test_util.run_deprecated_v1
  def testAccumulatorApplyAndTakeGradWithShape(self):
    # [FUNCTION DOC]
    # Test: accumulator with fixed shape 3x2; applies two matrices, takes the average and compares
    # each element with the manually computed average.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=(3, 2))
      elems = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
               [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
      elems_ave = [[(a + b) / len(elems) for a, b in zip(x, y)]
                   for x, y in zip(elems[0], elems[1])]
      accum_ops = [q.apply_grad(x) for x in elems]
      takeg_t = q.take_grad(1)

      for accum_op in accum_ops:
        accum_op.run()

      is_all_equal = True
      val = self.evaluate(takeg_t)
      for i in range(len(val)):
        for j in range(len(val[i])):
          is_all_equal &= (val[i][j] == elems_ave[i][j])
      self.assertTrue(is_all_equal)

  @test_util.run_deprecated_v1
  def testAccumulatorApplyGradWithWrongShape(self):
    # [FUNCTION DOC]
    # Test: applying gradients whose static shape differs from the accumulator's 3x2 shape must raise
    # a value error at graph-construction time (two wrong shapes are tried).
    q = data_flow_ops.ConditionalAccumulator(
        dtypes_lib.float32, name="Q", shape=(3, 2))

    with self.assertRaises(ValueError):
      q.apply_grad([[1.0, 2.0], [3.0, 4.0]])

    with self.assertRaises(ValueError):
      q.apply_grad([[1.0], [2.0], [3.0]])

  @test_util.run_deprecated_v1
  def testAccumulatorDynamicShape(self):
    # [FUNCTION DOC]
    # Test: accumulator with unknown shape fed through a placeholder; two matrices are applied via
    # feeding, and the averaged result is compared element by element with the expected mean.
    with self.cached_session() as sess:
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=None)

      x = array_ops.placeholder(dtypes_lib.float32)

      accum_op = q.apply_grad(x)

      elems = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
               [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
      elems_ave = [[(a + b) / len(elems) for a, b in zip(c, d)]
                   for c, d in zip(elems[0], elems[1])]
      takeg_t = q.take_grad(1)

      for elem in elems:
        sess.run(accum_op, feed_dict={x: elem})

      is_all_equal = True
      val = self.evaluate(takeg_t)
      for i in range(len(val)):
        for j in range(len(val[i])):
          is_all_equal &= (val[i][j] == elems_ave[i][j])
      self.assertTrue(is_all_equal)

  @test_util.run_v1_only("b/120545219")
  def testAccumulatorWrongDynamicShape(self):
    # [FUNCTION DOC]
    # Test (v1 only): with an unknown-shape accumulator, the first applied gradient fixes the shape;
    # later gradients with a different shape must raise an invalid-argument error.
    with self.cached_session() as sess:
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=None)

      x = array_ops.placeholder(dtypes_lib.float32)

      accum_op = q.apply_grad(x)

      # First successful apply_grad determines shape
      sess.run(accum_op, feed_dict={x: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(accum_op, feed_dict={x: [[1.0, 2.0], [3.0, 4.0]]})

      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(accum_op, feed_dict={x: [[1.0], [2.0], [3.0]]})

  @test_util.run_deprecated_v1
  def testAccumulatorSizeAfterApplyGrad(self):
    # [FUNCTION DOC]
    # Test: the accumulated count is 0 initially and grows to 1 and then 2 after each apply.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      accum_op = q.apply_grad((10.0,))
      self.assertEqual(q.num_accumulated().eval(), 0)
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 1)
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 2)

  @test_util.run_deprecated_v1
  def testAccumulatorSizeAfterApplyGradAndTakeGrad(self):
    # [FUNCTION DOC]
    # Test: after two applies and a take of 2 the count returns to 0; then four more applies with
    # local step 1 give count 4 and a successful take resets it to 0 (taking always resets size).
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      accum_op = q.apply_grad((10.0,))
      extract_t = q.take_grad(2)

      # Applying gradient multiple times to increase size from 0 to 2.
      self.assertEqual(q.num_accumulated().eval(), 0)
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 1)
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 2)

      # Extract will reduce size to 0
      extract_t.op.run()
      self.assertEqual(q.num_accumulated().eval(), 0)

      # Take gradients always sets the size back to 0 if successful.
      accum_op = q.apply_grad((10.0,), local_step=1)
      accum_op.run()
      accum_op.run()
      accum_op.run()
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 4)
      extract_t.op.run()
      self.assertEqual(q.num_accumulated().eval(), 0)

  @test_util.run_deprecated_v1
  def testAccumulatorTakeGradMean(self):
    # [FUNCTION DOC]
    # Test: default (mean) reduction. Applies 10 and 20 with local step 0 and takes 1 gradient,
    # expecting 15; repeats with local step 1 and the take count passed as a tensor constant,
    # again expecting 15.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      elems = [10.0, 20.0]

      accum_ops = [q.apply_grad((x,), local_step=0) for x in elems]
      takeg_t = q.take_grad(1)

      for accum_op in accum_ops:
        accum_op.run()

      val = self.evaluate(takeg_t)
      self.assertEqual(15.0, val)

      accum_ops = [q.apply_grad((x,), local_step=1) for x in elems]
      takeg_t = q.take_grad(constant_op.constant(1))

      for accum_op in accum_ops:
        accum_op.run()

      val = self.evaluate(takeg_t)
      self.assertEqual(15.0, val)

  @test_util.run_deprecated_v1
  def testAccumulatorTakeGradSum(self):
    # [FUNCTION DOC]
    # Test: accumulator with sum reduction. Same flow as the mean test but the taken value must be
    # 30 both times.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([1]),
          reduction_type="SUM")
      elems = [10.0, 20.0]

      accum_ops = [q.apply_grad((x,), local_step=0) for x in elems]
      takeg_t = q.take_grad(1)

      for accum_op in accum_ops:
        accum_op.run()

      val = self.evaluate(takeg_t)
      self.assertEqual(30.0, val)

      accum_ops = [q.apply_grad((x,), local_step=1) for x in elems]
      takeg_t = q.take_grad(constant_op.constant(1))

      for accum_op in accum_ops:
        accum_op.run()

      val = self.evaluate(takeg_t)
      self.assertEqual(30.0, val)

  @test_util.run_deprecated_v1
  def testAccumulatorTakeGradInvalidReductionType(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: duplicates the scenario of the constructor-invalid-argument test; the name says
    # take-grad but only the constructor is exercised and nothing is taken.
    # Test: constructing an accumulator with an invalid reduction type raises a value error.
    with self.assertRaises(ValueError):
      data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([1]),
          reduction_type="Invalid")

  @test_util.run_v1_only("b/120545219")
  def testAccumulatorInvalidTakeGrad(self):
    # [FUNCTION DOC]
    # Test (v1 only): applies two gradients then takes with a negative count; evaluating the take
    # must raise an invalid-argument error.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      elems = [10.0, 20.0]
      accum_ops = [q.apply_grad((x,)) for x in elems]

      takeg_t = q.take_grad(-1)

      for accum_op in accum_ops:
        accum_op.run()

      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(takeg_t)

  @test_util.run_deprecated_v1
  def testAccumulatorRepeatedTakeGradMean(self):
    # [FUNCTION DOC]
    # Test: two rounds of mean accumulation with different local steps (0 then 1) and different
    # values; each take must return the mean of the round's inputs.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))

      elems = [10.0, 20.0]
      elems_ave = sum(elems) / len(elems)
      accum_ops = [q.apply_grad((x,), local_step=0) for x in elems]
      takeg_t = q.take_grad(1)

      for accum_op in accum_ops:
        accum_op.run()

      val = self.evaluate(takeg_t)
      self.assertEqual(elems_ave, val)

      elems = [20.0, 30.0]
      elems_ave = sum(elems) / len(elems)
      accum_ops = [q.apply_grad((x,), local_step=1) for x in elems]
      takeg_t = q.take_grad(1)

      for accum_op in accum_ops:
        accum_op.run()

      val = self.evaluate(takeg_t)
      self.assertEqual(elems_ave + 0.0, val)

  @test_util.run_deprecated_v1
  def testAccumulatorRepeatedTakeGradSum(self):
    # [FUNCTION DOC]
    # Test: same as the repeated-mean test but with sum reduction; expects 30 then 50.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([1]),
          reduction_type="SUM")

      elems = [10.0, 20.0]
      elems_sum = 30.0
      accum_ops = [q.apply_grad((x,), local_step=0) for x in elems]
      takeg_t = q.take_grad(1)

      for accum_op in accum_ops:
        accum_op.run()

      val = self.evaluate(takeg_t)
      self.assertEqual(elems_sum, val)

      elems = [20.0, 30.0]
      elems_sum = 50.0
      accum_ops = [q.apply_grad((x,), local_step=1) for x in elems]
      takeg_t = q.take_grad(1)

      for accum_op in accum_ops:
        accum_op.run()

      val = self.evaluate(takeg_t)
      self.assertEqual(elems_sum, val)

  @test_util.run_deprecated_v1
  def testAccumulatorIncrementGlobalStep(self):
    # [FUNCTION DOC]
    # Test: wires an accumulator's set-global-step op to a variable incremented with each loop
    # iteration; initializes variables and runs set-step plus increment three times. Passes if no
    # error occurs; nothing is asserted.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))

      global_step = variables.Variable(0, name="global_step")
      new_global_step = math_ops.add(global_step, 1)
      inc_global_step = state_ops.assign(global_step, new_global_step)

      set_global_step_op = q.set_global_step(new_global_step)

      self.evaluate(variables.global_variables_initializer())
      for _ in range(3):
        set_global_step_op.run()
        self.evaluate(inc_global_step)

  @test_util.run_deprecated_v1
  def testAccumulatorSetGlobalStepPreventsAccumulation(self):
    # [FUNCTION DOC]
    # Test: applies gradients with local steps 1000..1004; for each global step value, sets it, runs
    # every apply and takes one gradient. Only gradients with local step at or above the global step
    # count, so the taken value must equal the mean of those local steps.
    with self.cached_session():
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))

      local_steps = range(1000, 1005)
      accum_ops = [q.apply_grad((0.0 + x,), local_step=x) for x in local_steps]

      for ls in local_steps:
        set_global_step_op = q.set_global_step(ls)
        set_global_step_op.run()

        for accum_op in accum_ops:
          accum_op.run()
        takeg_t = q.take_grad(1)

        val = self.evaluate(takeg_t)
        self.assertEqual(0.0 + sum(x for x in local_steps
                                   if x >= ls) / sum(1 for x in local_steps
                                                     if x >= ls), val)

  @test_util.run_v1_only("b/120545219")
  def testParallelApplyGrad(self):
    # [FUNCTION DOC]
    # Test (v1 only): makes the graph's device stack thread-local, then applies ten gradients from
    # ten threads and checks that the mean taken equals the mean of the inputs. The session variable
    # is unused.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      accum_ops = [q.apply_grad((x,), local_step=0) for x in elems]
      takeg_t = q.take_grad(1)

      def apply_grad(accum_op):
        # [FUNCTION DOC]
        # Nested thread target: evaluates the given apply op. Input: the op. Returns nothing.
        self.evaluate(accum_op)

      threads = [
          self.checkedThread(
              target=apply_grad, args=(o,)) for o in accum_ops
      ]

      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      val = self.evaluate(takeg_t)

      self.assertEqual(val, sum(elems) / len(elems))

  @test_util.run_v1_only("b/120545219")
  def testParallelTakeGrad(self):
    # [FUNCTION DOC]
    # Test (v1 only): ten threads each take a gradient while another thread applies ten gradients
    # one second apart; takes block until available. The multiset of results must equal the ten
    # applied values. Graph is made thread-local first.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      elems = [e for e in range(10)]
      accum_ops = [q.apply_grad((np.float32(e),), local_step=e) for e in elems]
      takeg_t = q.take_grad(1)

      def apply_grad():
        # [FUNCTION DOC]
        # Nested thread target: for each prepared apply op, sleeps one second and then evaluates it.
        # Takes no arguments, returns nothing.
        for accum_op in accum_ops:
          time.sleep(1.0)
          self.evaluate(accum_op)

      apply_grad_thread = self.checkedThread(target=apply_grad)

      results = []

      def take_grad():
        # [FUNCTION DOC]
        # Nested thread target: evaluates the take op and appends the result to the shared results list.
        results.append(self.evaluate(takeg_t))

      threads = [self.checkedThread(target=take_grad) for _ in range(10)]

      for thread in threads:
        thread.start()
      apply_grad_thread.start()

      for thread in threads:
        thread.join()
      apply_grad_thread.join()

      self.assertItemsEqual(elems, results)

  @test_util.run_v1_only("b/120545219")
  def testAccumulatorApplyAndBlockingTake(self):
    # [FUNCTION DOC]
    # Test (v1 only): a take of 3 gradients blocks in one thread while another thread applies three
    # gradients after a one second delay. The take must return the mean of the three values.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))

      elems = [10.0, 20.0, 30.0]
      elems_ave = sum(elems) / len(elems)
      accum_ops = [q.apply_grad((x,), local_step=0) for x in elems]
      takeg_t = q.take_grad(3)

      def apply_grad():
        # [FUNCTION DOC]
        # Nested thread target: sleeps one second and then evaluates every apply op in order.
        time.sleep(1.0)
        for accum_op in accum_ops:
          self.evaluate(accum_op)

      return_array = []

      def take_grad():
        # [FUNCTION DOC]
        # Nested thread target: evaluates the take op and appends the result to the shared list.
        return_array.append(self.evaluate(takeg_t))

      accum_thread = self.checkedThread(target=apply_grad)
      takeg_thread = self.checkedThread(target=take_grad)
      accum_thread.start()
      takeg_thread.start()
      accum_thread.join()
      takeg_thread.join()

      self.assertEqual([elems_ave], return_array)

  def _blocking_takeg(self, sess, takeg_op):
    # [FUNCTION DOC]
    # HELPER: thread target for testAccumulatorCancel.
    # Evaluates the given take op and asserts it raises an op error containing "was cancelled".
    # Inputs: a session (unused, evaluation uses the test's default session) and the take op.
    # Returns nothing.
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(takeg_op)

  @test_util.run_v1_only("b/120545219")
  def testAccumulatorCancel(self):
    # [FUNCTION DOC]
    # Test (v1 only): starts a thread blocked on taking a gradient from an empty accumulator, waits
    # one second, then closes the session, which must cancel the blocked op so that the helper sees
    # a "was cancelled" error; joins the thread.
    with self.cached_session() as sess:
      q = data_flow_ops.ConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      takeg_t = q.take_grad(1)

      takeg_thread = self.checkedThread(
          self._blocking_takeg, args=(sess, takeg_t))

      takeg_thread.start()

      time.sleep(1.0)

      sess.close()  # Will cancel blocked operation

      takeg_thread.join()


if __name__ == "__main__":
  test.main()
