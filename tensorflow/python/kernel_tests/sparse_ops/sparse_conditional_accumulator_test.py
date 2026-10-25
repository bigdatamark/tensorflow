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
# [MODULE DOC]
# Unit tests for the sparse conditional accumulator (data_flow_ops.SparseConditionalAccumulator)
# with indexed-slices gradients. Covers construction (node definition attributes, optional shape,
# invalid reduction type), size and global step ops, applying gradients of several float types,
# mean and sum reduction, repeated take, stale-gradient handling via local steps, multithreaded
# apply/take and cancellation, validation errors for malformed indices, values and shapes, shape
# inference of the result, and 32-bit index and shape inputs. Many tests are graph-mode only.
# No import-time side effects.
# ==============================================================================

import time

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


def _indexedslice(x, noshape=False):
  # [FUNCTION DOC]
  # HELPER: converts a dense array into an indexed-slices gradient for the accumulator tests.
  # Input: an array-like x and a noshape flag (default false). The slice indices are the
  # first-axis rows whose elements sum to a non-zero value (so a row of values that cancel to
  # zero would be dropped); values are those rows. Returns an IndexedSlices object with indices
  # as a list, the selected rows, and the dense shape of x, or no dense shape if noshape is true.
  x = np.array(x)
  dense_shape = x.shape
  ndim = len(dense_shape)
  indices = np.where(np.sum(x, tuple(range(1, ndim))))[0]
  values = x[indices]
  if noshape:
    dense_shape = None
  return indexed_slices.IndexedSlices(
      indices=indices.tolist(), values=values, dense_shape=dense_shape)


class IndexedSlicesConditionalAccumulatorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for SparseConditionalAccumulator used with indexed-slices gradients. Provides
  # assertion helpers comparing indexed slices and dense arrays; each test builds its own
  # accumulator (named "Q") and graph/session.

  def _assertEqual_indexedslices(self, expected_tensor, result):
    # [FUNCTION DOC]
    # HELPER: assertion shared by accumulator tests. Asserts the indices and values of the
    # expected and actual indexed slices are equal, and compares dense shapes only when both
    # are present. Returns nothing.
    self.assertAllEqual(expected_tensor.indices, result.indices)
    self.assertAllEqual(expected_tensor.values, result.values)
    if (result.dense_shape is not None and
        expected_tensor.dense_shape is not None):
      self.assertAllEqual(expected_tensor.dense_shape, result.dense_shape)

  def _assertEqual_nparray(self, expected_array, result, sess):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the sess parameter is accepted but never used.
    # HELPER: assertion shared by accumulator tests. Converts the expected dense array into
    # indexed slices and compares with the actual result via the indexed-slices assertion.
    # Returns nothing.
    expected_tensor = _indexedslice(expected_array)
    self._assertEqual_indexedslices(expected_tensor, result)

  def testConstructor(self):
    # [FUNCTION DOC]
    # Creates a float accumulator named Q in a new graph with default arguments and asserts the
    # accumulator handle is a tensor and its node definition has the expected attributes:
    # float dtype, unknown-rank shape, empty container and shared name, mean reduction.
    # Returns nothing.
    with ops.Graph().as_default():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q")
    self.assertTrue(isinstance(q.accumulator_ref, tensor.Tensor))
    self.assertProtoEquals(
        """
      name:'Q' op:'SparseConditionalAccumulator'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'shape' value { shape { unknown_rank: true} } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      attr { key: 'reduction_type' value {s: 'MEAN'} }
      """, q.accumulator_ref.op.node_def)

  def testConstructorWithInvalidArg(self):
    # [FUNCTION DOC]
    # Asserts that constructing an accumulator with an unsupported reduction type raises a value
    # error. Returns nothing.
    with ops.Graph().as_default():
      with self.assertRaises(ValueError):
        data_flow_ops.SparseConditionalAccumulator(
            dtypes_lib.float32, name="Q", reduction_type="Invalid")

  def testConstructorWithShape(self):
    # [FUNCTION DOC]
    # Creates an accumulator with an explicit four-dimensional shape (1, 5, 2, 8) and asserts
    # the node definition records those dimensions, with the other attributes at defaults.
    # Returns nothing.
    with ops.Graph().as_default():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([1, 5, 2, 8]))
    self.assertTrue(isinstance(q.accumulator_ref, tensor.Tensor))
    self.assertProtoEquals(
        """
      name:'Q' op:'SparseConditionalAccumulator'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'shape' value { shape { dim {size: 1 }
                                          dim {size: 5 }
                                          dim {size: 2 }
                                          dim {size: 8 }
      } } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      attr { key: 'reduction_type' value {s: 'MEAN'} }
      """, q.accumulator_ref.op.node_def)

  @test_util.run_deprecated_v1
  def testAccumulatorSizeEmpty(self):
    # [FUNCTION DOC]
    # Graph-mode test. Asserts a new accumulator reports zero accumulated gradients. Returns
    # nothing.
    with self.cached_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q")
      self.assertEqual(q.num_accumulated().eval(), 0)

  @test_util.run_deprecated_v1
  def testAccumulatorSetGlobalStep(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates an accumulator and runs the op that sets its global step to one;
    # passes if it runs without error (no value is asserted). Returns nothing.
    with self.cached_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      set_global_step_op = q.set_global_step(1)
      set_global_step_op.run()

  @test_util.run_deprecated_v1
  def testAccumulatorApplyGradFloat32(self):
    # [FUNCTION DOC]
    # Graph-mode test. Applies one float32 indexed-slices gradient (rows 0 and 2 of a 3x3
    # shape) and asserts the accumulator reports one accumulated gradient. Returns nothing.
    with self.cached_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))
      accum_op = q.apply_indexed_slices_grad(
          indexed_slices.IndexedSlices(
              indices=[0, 2],
              values=np.array([[0, 0, 1], [3, 0, 4]]).astype(np.float32)))
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 1)

  @test_util.run_deprecated_v1
  def testDtypes(self):
    # [FUNCTION DOC]
    # Graph-mode test. For float16, float32 and float64, creates an accumulator for 3x3x3
    # gradients, applies two gradients with a single non-zero element on the diagonal, takes the
    # average and asserts it equals the mean of the two. Returns nothing.
    with self.cached_session() as sess:
      dtypes = [dtypes_lib.float16, dtypes_lib.float32, dtypes_lib.float64]

      for i in range(len(dtypes)):
        dtype = dtypes[i]
        q = data_flow_ops.SparseConditionalAccumulator(
            dtype, shape=tensor_shape.TensorShape([3, 3, 3]))

        elems = np.arange(2)
        sum_elems = np.zeros([3, 3, 3]).astype(dtype.as_numpy_dtype)
        for e in elems:
          mat_to_add = np.zeros([3, 3, 3]).astype(dtype.as_numpy_dtype)
          mat_to_add[i, i, i] = e + 1
          sum_elems += mat_to_add
          t = _indexedslice(mat_to_add)
          q.apply_indexed_slices_grad(t).run()

        result = self.evaluate(q.take_indexed_slices_grad(1))

        self._assertEqual_nparray(sum_elems / len(elems), result, sess)

  @test_util.run_deprecated_v1
  def testAccumulatorMultipleAccumulators(self):
    # [FUNCTION DOC]
    # Graph-mode test. Creates four accumulators (two float16, two float32, all named Q, shape
    # 2x2), applies a distinct one-hot gradient to each and asserts that taking one gradient
    # from each returns exactly the gradient applied to that accumulator, i.e. they are
    # independent. Returns nothing.
    with self.cached_session() as sess:
      q_f32_0 = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      q_f32_1 = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      q_f16_0 = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float16, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      q_f16_1 = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float16, name="Q", shape=tensor_shape.TensorShape([2, 2]))

      accums = [q_f16_0, q_f16_1, q_f32_0, q_f32_1]

      elems = [[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0],
                                                                      [0, 1]]]

      expected_tensors = []

      for i in range(len(accums)):
        tensor_to_add = np.array(elems[i]).astype(accums[i]
                                                  .dtype.as_numpy_dtype)
        expected_tensor = _indexedslice(tensor_to_add)
        expected_tensors.append(expected_tensor)
        st = _indexedslice(tensor_to_add)
        accums[i].apply_indexed_slices_grad(st).run()

      for i in range(len(accums)):
        result = sess.run(accums[i].take_indexed_slices_grad(1))
        self._assertEqual_indexedslices(expected_tensors[i], result)

  @test_util.run_deprecated_v1
  def testAccumulatorTakeGradMean(self):
    # [FUNCTION DOC]
    # Graph-mode test. Applies two sparse gradients (one as indexed slices, one through the
    # indices/values/shape form) to an accumulator of unspecified shape, takes the result with
    # the default mean reduction and asserts indices 0, 1, 2 and values where the overlapping
    # row 0 is averaged over the two gradients while rows present in only one gradient keep
    # their value, and dense shape of unknown first dimension (-1) by 2. Returns nothing.
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=())

      grad_indexed_slices = indexed_slices.IndexedSlices(
          indices=[0, 1], values=np.array([[1, 0], [0, 2]]).astype(np.float32))
      accum_op = q.apply_indexed_slices_grad(grad_indexed_slices)
      accum_op.run()
      accum_op = q.apply_grad([0, 2],
                              np.array([[0, 1], [3, 0]]).astype(np.float32),
                              [3, 2])
      accum_op.run()

      takeg_t = q.take_indexed_slices_grad(1)
      val = self.evaluate(takeg_t)
      self.assertAllEqual([0, 1, 2], val.indices)
      self.assertAllEqual([[0.5, 0.5], [0, 2], [3, 0]], val.values)
      self.assertAllEqual([-1, 2], val.dense_shape)

  @test_util.run_deprecated_v1
  def testAccumulatorTakeGradSum(self):
    # [FUNCTION DOC]
    # Graph-mode test. Same two gradients as the mean test but with the sum reduction; asserts
    # indices 0, 1, 2, values with the overlapping row 0 summed, and dense shape (-1, 2).
    # Returns nothing.
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=(), reduction_type="SUM")

      grad_indexed_slices = indexed_slices.IndexedSlices(
          indices=[0, 1], values=np.array([[1, 0], [0, 2]]).astype(np.float32))
      accum_op = q.apply_indexed_slices_grad(grad_indexed_slices)
      accum_op.run()
      accum_op = q.apply_grad([0, 2],
                              np.array([[0, 1], [3, 0]]).astype(np.float32),
                              [3, 2])
      accum_op.run()

      takeg_t = q.take_indexed_slices_grad(1)
      val = self.evaluate(takeg_t)
      self.assertAllEqual([0, 1, 2], val.indices)
      self.assertAllEqual([[1, 1], [0, 2], [3, 0]], val.values)
      self.assertAllEqual([-1, 2], val.dense_shape)

  @test_util.run_deprecated_v1
  def testAccumulatorTakeGradInvalidReductionType(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests it tests take-gradient, but it only checks that the
    # constructor rejects an invalid reduction type, duplicating the constructor test.
    # Graph-mode test. Asserts a value error is raised when constructing an accumulator with an
    # invalid reduction type. Returns nothing.
    with self.assertRaises(ValueError):
      data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=(), reduction_type="Invalid")

  @test_util.run_deprecated_v1
  def testAccumulatorRepeatedTakeGrad(self):
    # [FUNCTION DOC]
    # Graph-mode test. Applies two gradients at local step 0, takes the mean and asserts the
    # result; then applies two gradients at local step 1 (the step advanced after the take) and
    # takes again, asserting the second mean contains only the new gradients (values 5, 5; 0, 20;
    # 30, 0). Returns nothing.
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=())

      grad_indexed_slices = indexed_slices.IndexedSlices(
          indices=[0, 1], values=np.array([[1, 0], [0, 2]]).astype(np.float32))
      accum_op = q.apply_indexed_slices_grad(grad_indexed_slices, local_step=0)
      accum_op.run()
      accum_op = q.apply_grad(
          [0, 2],
          np.array([[0, 1], [3, 0]]).astype(np.float32), [3, 2],
          local_step=0)
      accum_op.run()

      takeg_t = q.take_indexed_slices_grad(1)
      val = self.evaluate(takeg_t)
      self.assertAllEqual(val.indices, [0, 1, 2])
      self.assertAllEqual(val.values, [[0.5, 0.5], [0, 2], [3, 0]])
      self.assertAllEqual(val.dense_shape, [-1, 2])

      grad_indexed_slices = indexed_slices.IndexedSlices(
          indices=[0, 1],
          values=np.array([[10, 0], [0, 20]]).astype(np.float32))
      accum_op = q.apply_indexed_slices_grad(grad_indexed_slices, local_step=1)
      accum_op.run()
      accum_op = q.apply_grad(
          [0, 2],
          np.array([[0, 10], [30, 0]]).astype(np.float32), [3, 2],
          local_step=1)
      accum_op.run()

      takeg_t = q.take_indexed_slices_grad(1)
      val = self.evaluate(takeg_t)
      self.assertAllEqual(val.indices, [0, 1, 2])
      self.assertAllEqual(val.values, [[5, 5], [0, 20], [30, 0]])
      self.assertAllEqual(val.dense_shape, [-1, 2])

  @test_util.run_v1_only("b/120545219")
  def testParallelApplyGradMean(self):
    # [FUNCTION DOC]
    # Graph-mode-only test (skipped under v2 behavior, bug reference in the decorator). Makes
    # the default graph thread-local, then applies ten diagonal 2x2 gradients (values 10 to 100)
    # concurrently from ten threads, takes the mean and asserts it equals the average (55) on
    # the diagonal. Side effect: switches the default graph to per-thread device stacks.
    # Returns nothing.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      accum_ops = []
      for x in elems:
        x = _indexedslice(np.array([[x, 0], [0, x]]).astype(np.float32))
        accum_ops.append(q.apply_indexed_slices_grad(x, local_step=0))
      takeg_t = q.take_indexed_slices_grad(1)

      def apply_indexed_slices_grad(accum_op):
        # [FUNCTION DOC]
        # HELPER: thread target. Evaluates the given apply-gradient op in the session; returns
        # nothing. Errors propagate to the checked thread.
        self.evaluate(accum_op)

      threads = [
          self.checkedThread(
              target=apply_indexed_slices_grad, args=(o,)) for o in accum_ops
      ]

      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      val = self.evaluate(takeg_t)

      expected_val = sum(elems) / len(elems)
      self._assertEqual_nparray(
          np.array([[expected_val, 0], [0, expected_val]]).astype(np.float32),
          val, sess)

  @test_util.run_v1_only("b/120545219")
  def testParallelApplyGradSum(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Like the parallel mean test but with the sum reduction: ten threads
    # apply diagonal gradients (10 to 100) and the taken result must have 550 on the diagonal.
    # Side effect: makes the default graph thread-local. Returns nothing.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([2, 2]),
          reduction_type="SUM")
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      accum_ops = []
      for x in elems:
        x = _indexedslice(np.array([[x, 0], [0, x]]).astype(np.float32))
        accum_ops.append(q.apply_indexed_slices_grad(x, local_step=0))
      takeg_t = q.take_indexed_slices_grad(1)

      def apply_indexed_slices_grad(accum_op):
        # [FUNCTION DOC]
        # HELPER: thread target. Evaluates the given apply-gradient op in the session; returns
        # nothing. Errors propagate to the checked thread.
        self.evaluate(accum_op)

      threads = [
          self.checkedThread(target=apply_indexed_slices_grad, args=(o,))
          for o in accum_ops
      ]

      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      val = self.evaluate(takeg_t)

      expected_val = 550.0
      self._assertEqual_nparray(
          np.array([[expected_val, 0], [0, expected_val]]).astype(np.float32),
          val, sess)

  @test_util.run_v1_only("b/120545219")
  def testParallelTakeGrad(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Ten gradients (row 1 holding 1..10) are applied sequentially by one
    # thread, one per second, each at an increasing local step, while ten threads concurrently
    # try to take one gradient each. Asserts the i-th collected result equals the i-th applied
    # gradient. Side effect: makes the default graph thread-local; takes about ten seconds.
    # Returns nothing.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      elems = [e + 1 for e in range(10)]
      accum_ops = []
      for e in elems:
        v = _indexedslice(np.array([[0, 0], [e, 0]]).astype(np.float32))
        accum_ops.append(q.apply_indexed_slices_grad(v, local_step=e - 1))
      takeg_t = q.take_indexed_slices_grad(1)

      results = []

      def apply_indexed_slices_grad():
        # [FUNCTION DOC]
        # HELPER: thread target. Evaluates each apply-gradient op in order, sleeping one second
        # before each. Returns nothing.
        for accum_op in accum_ops:
          time.sleep(1.0)
          self.evaluate(accum_op)

      apply_indexed_slices_grad_thread = self.checkedThread(
          target=apply_indexed_slices_grad)

      def take_grad():
        # [FUNCTION DOC]
        # HELPER: thread target. Evaluates the take-gradient op (blocks until a gradient is
        # available) and appends the result to the shared results list. Returns nothing.
        t = self.evaluate(takeg_t)
        results.append(t)

      threads = [self.checkedThread(target=take_grad) for _ in range(10)]

      for thread in threads:
        thread.start()
      apply_indexed_slices_grad_thread.start()

      for thread in threads:
        thread.join()
      apply_indexed_slices_grad_thread.join()

      for i in range(len(accum_ops)):
        self._assertEqual_nparray(
            np.array([[0, 0], [elems[i], 0]]), results[i], sess)

  @test_util.run_v1_only("b/120545219")
  def testAccumulatorApplyAndBlockingTake(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Starts a thread applying three gradients (values 10, 20, 30 at the
    # same position) and another thread taking a gradient requiring three accumulated ones,
    # which blocks until all are applied. Asserts the taken gradient is the mean (20). Side
    # effect: makes the default graph thread-local. Returns nothing.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))

      elems = [10.0, 20.0, 30.0]
      elems_ave = sum(elems) / len(elems)
      accum_ops = []
      for x in elems:
        x = _indexedslice(np.array([[0, x], [0, 0]]).astype(np.float32))
        accum_ops.append(q.apply_indexed_slices_grad(x, local_step=0))
      takeg_t = q.take_indexed_slices_grad(3)

      results = []

      def apply_indexed_slices_grad():
        # [FUNCTION DOC]
        # HELPER: thread target. Evaluates each apply-gradient op in order. Returns nothing.
        for accum_op in accum_ops:
          self.evaluate(accum_op)

      def take_grad():
        # [FUNCTION DOC]
        # HELPER: thread target. Evaluates the blocking take-gradient op and appends its result
        # to the shared results list. Returns nothing.
        results.append(self.evaluate(takeg_t))

      accum_thread = self.checkedThread(target=apply_indexed_slices_grad)
      takeg_thread = self.checkedThread(target=take_grad)
      accum_thread.start()
      takeg_thread.start()
      accum_thread.join()
      takeg_thread.join()

      self._assertEqual_nparray([[0, elems_ave], [0, 0]], results[0], sess)

  def _blocking_takeg(self, sess, takeg_op):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the sess parameter is accepted but unused (the op is run through the test's
    # own evaluate method).
    # HELPER: thread target for the cancel test. Evaluates a take-gradient op that blocks and
    # asserts it fails with an op error saying it was cancelled. Returns nothing.
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(takeg_op)

  @test_util.run_v1_only("b/120545219")
  def testAccumulatorCancel(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Starts a thread that blocks on taking a gradient from an empty
    # accumulator, waits one second, then closes the session and asserts the blocked take fails
    # as cancelled (checked in the helper thread). Side effect: makes the default graph
    # thread-local and closes the session. Returns nothing.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([1, 2, 3]))
      takeg_t = q.take_indexed_slices_grad(1)

      takeg_thread = self.checkedThread(
          self._blocking_takeg, args=(sess, takeg_t))

      takeg_thread.start()

      time.sleep(1.0)

      sess.close()  # Will cancel blocked operation

      takeg_thread.join()

  @test_util.run_v1_only("b/120545219")
  def testNonVectorIndices(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Applying a gradient whose indices are a matrix must raise an
    # invalid-argument error saying indices should be a vector. Returns nothing.
    with self.cached_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Input indices should be vector but received shape:"):
        q.apply_grad(
            grad_indices=[[0, 1], [1, 0]],
            grad_values=np.array([1, 2]).astype(np.float32)).run()

  @test_util.run_v1_only("b/120545219")
  def testZeroDimensionValues(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Applying a gradient whose values are a scalar must raise an
    # invalid-argument error saying values cannot be zero-dimensional. Returns nothing.
    with self.cached_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Values cannot be 0-dimensional."):
        q.apply_grad(
            grad_indices=[0], grad_values=np.array(1).astype(np.float32)).run()

  @test_util.run_v1_only("b/120545219")
  def testWrongNonEmptyInputValues(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Applying two indices with a values array holding only one row must
    # raise an invalid-argument error about expecting non-empty input values. Returns nothing.
    with self.cached_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  " non-empty input values, got "):
        q.apply_grad(
            grad_indices=[0, 1],
            grad_values=np.array([[0, 1, 1]]).astype(np.float32)).run()

  @test_util.run_v1_only("b/120545219")
  def testDynamicNonVectorIndices(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Same as the non-vector indices test but with placeholders fed at run
    # time, so the error (indices should be a vector) comes from the runtime check. Returns
    # nothing.
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      x_indices = array_ops.placeholder(dtypes_lib.int64)
      x_values = array_ops.placeholder(dtypes_lib.float32)

      accum_op = q.apply_grad(grad_indices=x_indices, grad_values=x_values)

      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Input indices should be vector but received shape:"):
        sess.run(accum_op,
                 feed_dict={
                     x_indices: [[0, 1], [1, 0]],
                     x_values: np.array([1, 2]).astype(np.float32)
                 })

  @test_util.run_v1_only("b/120545219")
  def testDynamicWrongNonEmptyInputValues(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Feeds placeholders with two indices but a one-row values array and
    # asserts the runtime invalid-argument error about non-empty input values. Returns nothing.
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      x_indices = array_ops.placeholder(dtypes_lib.int64)
      x_values = array_ops.placeholder(dtypes_lib.float32)

      accum_op = q.apply_grad(grad_indices=x_indices, grad_values=x_values)

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  " non-empty input values, got "):
        sess.run(accum_op,
                 feed_dict={
                     x_indices: [0, 1],
                     x_values: np.array([[0, 1, 1]]).astype(np.float32)
                 })

  @test_util.run_v1_only("b/120545219")
  def testEmptyShapeApply(self):
    # [FUNCTION DOC]
    # Graph-mode-only test with an accumulator of scalar shape. Asserts scalar indices raise a
    # vector-indices error and scalar values raise a zero-dimensional values error (with and
    # without an explicit gradient shape), then checks the valid form (vector indices, vector
    # values) applies without error. Returns nothing.
    with self.cached_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([]))

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Input indices should be vector"):
        q.apply_grad(grad_indices=0, grad_values=[1.0], grad_shape=[]).run()

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Input indices should be vector"):
        q.apply_grad(grad_indices=0, grad_values=[1.0]).run()

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Values cannot be 0-dimensional."):
        q.apply_grad(grad_indices=[0], grad_values=1.0, grad_shape=[]).run()

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Values cannot be 0-dimensional."):
        q.apply_grad(grad_indices=[0], grad_values=1.0).run()

      # The right way to apply a scalar
      q.apply_grad(grad_indices=[0], grad_values=[1.0], grad_shape=[]).run()
      q.apply_grad(grad_indices=[0], grad_values=[1.0]).run()

  @test_util.run_v1_only("b/120545219")
  def testValidateShape(self):
    # [FUNCTION DOC]
    # Graph-mode-only test with an accumulator of partially known shape (2, 2, unknown). Asserts
    # specific invalid-argument shape-mismatch errors for: gradient shape of wrong rank or
    # dimension, index beyond limits, values of wrong rank or dimension. After a first
    # successful gradient the accumulator constrains the unknown dimension and later mismatching
    # values are rejected; after taking the gradient the constraint is cleared and a new
    # one (different last dimension) is imposed and enforced. Returns nothing.
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=[2, 2, None])

      # Provided shape has wrong rank
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected shape rank at least 3, got 2"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array([[1, 2]]).astype(np.float32),
            grad_shape=[2, 2]).run()

      # Provided shape has wrong dim
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected shape dim 1 to be 2, got 3"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32),
            grad_shape=[2, 3, 2]).run()

      # Indices exceeded accumulator's shape's limits
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: index of slice 0 exceeded limits of shape;"
          " index is 3 exceeded 2"):
        q.apply_grad(
            grad_indices=[3],
            grad_values=np.array([[[1, 2], [3, 4]]]).astype(np.float32)).run()

      # Values' rank does not match shape
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values rank at least 3, got 2"):
        q.apply_grad(
            grad_indices=[0, 1],
            grad_values=np.array([[1, 2], [3, 4]]).astype(np.float32)).run()

      # Values' dim does not match shape
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values dim 1 to be 2, got 3"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array(
                [[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)).run()

      # First successful gradient creates additional constraints
      # Shape will be additionally be constrained to [None,2,2,2] hereafter.
      q.apply_grad(
          grad_indices=[0],
          grad_values=np.array(
              [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)).run()

      # Values' rank does not match accumulated gradient
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values rank 4, got 3"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array([[[1, 2], [3, 4]]]).astype(np.float32)).run()

      # Values' dim does not match accumulated gradient
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values dim 3 to be 2, got 3"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array(
                [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(
                    np.float32)).run()

      # After take grad, constraints on accumulated gradient are removed
      self.evaluate(q.take_grad(1))

      # First successful gradient imposes new constraints.
      # Hereafter, shape will additionally constrained to [None,2,2,3]
      q.apply_grad(
          grad_indices=[0],
          grad_values=np.array(
              [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(
                  np.float32),
          local_step=1).run()

      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values dim 3 to be 3, got 2"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array(
                [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32),
            local_step=1).run()

  @test_util.run_deprecated_v1
  def testReturnShape(self):
    # [FUNCTION DOC]
    # Graph-mode test. For accumulators with partially known shapes, applies one four-dimensional
    # gradient and asserts the dense shape of the taken result: fully filled in (2, 2, 2, 2) when
    # the leading dimension was declared, and unknown leading dimension (-1) with the remaining
    # dimensions filled in when it was left unknown. Returns nothing.
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=[2, None])

      q.apply_grad(
          grad_indices=[0],
          grad_values=np.array(
              [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)).run()

      val = self.evaluate(q.take_indexed_slices_grad(1))
      self.assertAllEqual(val.dense_shape, [2, 2, 2, 2])

      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=[None, 2])

      q.apply_grad(
          grad_indices=[0],
          grad_values=np.array(
              [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(
                  np.float32)).run()

      val = self.evaluate(q.take_indexed_slices_grad(1))
      self.assertAllEqual(val.dense_shape, [-1, 2, 2, 3])

  @test_util.run_deprecated_v1
  def testApplyGradtInt32IndicesAndShape(self):
    # [FUNCTION DOC]
    # Graph-mode test. Applies two gradients with 32-bit integer indices and shape (one through
    # the indices/values/shape form, one through indexed slices), asserts two gradients are
    # counted, and that the taken result has the same indices, values and dense shape as one of
    # the (identical) gradients, i.e. the mean. Returns nothing.
    with self.cached_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))
      accum_op = q.apply_grad(
          grad_indices=constant_op.constant(
              [0, 2], dtype=dtypes_lib.int32),
          grad_values=constant_op.constant(
              [[0, 0, 1], [3, 0, 4]], dtype=dtypes_lib.float32),
          grad_shape=constant_op.constant(
              [3, 3], dtype=dtypes_lib.int32))
      accum_op.run()
      accum_op = q.apply_indexed_slices_grad(
          indexed_slices.IndexedSlices(
              indices=constant_op.constant(
                  [0, 2], dtype=dtypes_lib.int32),
              values=constant_op.constant(
                  [[0, 0, 1], [3, 0, 4]], dtype=dtypes_lib.float32),
              dense_shape=constant_op.constant(
                  [3, 3], dtype=dtypes_lib.int32)))
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 2)

      val = self.evaluate(q.take_indexed_slices_grad(1))
      self.assertAllEqual(val.indices, [0, 2])
      self.assertAllEqual(val.values, [[0, 0, 1], [3, 0, 4]])
      self.assertAllEqual(val.dense_shape, [3, 3])


if __name__ == "__main__":
  test.main()
