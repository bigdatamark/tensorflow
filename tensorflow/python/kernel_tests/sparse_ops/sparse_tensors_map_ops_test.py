# [MODULE DOC]
# Tests and a benchmark for the sparse-tensors-map ops, which store sparse tensors (or batches of
# sparse tensors) in a shared in-session map and hand back integer handles, and later take many
# handles and rebuild one batched sparse tensor with a leading minibatch dimension. Defines
# module-level aliases for the private add, add-many and take-many wrapper functions. The test
# class covers round trips, fed placeholders, rank inconsistency errors and bad handles; the
# benchmark class compares the handle-map approach with serializing and deserializing sparse
# tensors. Running the module as a script starts the TensorFlow test runner.
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
"""Tests for SparseTensorsMap."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

# pylint: disable=protected-access
add_sparse_to_tensors_map = sparse_ops._add_sparse_to_tensors_map
add_many_sparse_to_tensors_map = sparse_ops._add_many_sparse_to_tensors_map
take_many_sparse_from_tensors_map = (
    sparse_ops._take_many_sparse_from_tensors_map)

# pylint: enable=protected-access


class SparseTensorsMapTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the ops that add sparse tensors to a session-level map and take them back as a
  # batch. It holds no state; helper methods build placeholder and fixture sparse values, and all
  # tests run in graph mode on CPU.

  def _SparseTensorPlaceholder(self, dtype=None):
    # [FUNCTION DOC]
    # HELPER: creates a sparse tensor whose indices, values and dense shape are placeholders.
    # Input: dtype, the value type (nothing means 32-bit integer). Indices and shape are 64-bit
    # integers. Returns the sparse tensor of placeholders; adds placeholders to the default graph.
    if dtype is None:
      dtype = dtypes.int32
    return sparse_tensor_lib.SparseTensor(
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtype), array_ops.placeholder(dtypes.int64))

  def _SparseTensorValue_5x6(self, permutation):
    # [FUNCTION DOC]
    # HELPER: builds a sparse value of dense shape 5 by 6 with six entries (values 0, 10, 13, 14,
    # 32, 33). Input: permutation, a numpy array of positions used to reorder the entries (indices
    # and values together). Returns the sparse value with 64-bit indices and 32-bit values.
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2],
                    [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([5, 6]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_3x4(self, permutation):
    # [FUNCTION DOC]
    # HELPER: builds a sparse value of dense shape 3 by 4 with six entries (values 0, 10, 13, 14,
    # 32, 33). Input: permutation, a numpy array of positions used to reorder the entries (indices
    # and values together). Returns the sparse value with 64-bit indices and 32-bit values.
    ind = np.array([[0, 0], [1, 0], [1, 2], [1, 3], [2, 2],
                    [2, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([3, 4]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_1x1x1(self):
    # [FUNCTION DOC]
    # HELPER: builds a rank-3 sparse value with a single entry (value 0 at the origin) and dense
    # shape 3 by 4 by 5, used to provoke a rank mismatch. Takes no arguments. Despite the name,
    # the dense shape is not 1x1x1.
    ind = np.array([[0, 0, 0]]).astype(np.int64)
    val = np.array([0]).astype(np.int32)
    shape = np.array([3, 4, 5]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  @test_util.run_deprecated_v1
  def testAddTakeMany(self):
    # [FUNCTION DOC]
    # Scenario: add a 5x6 and a 3x4 sparse value to the map (same shared name), stack the two
    # scalar handles into a vector, and take them back as one batched sparse tensor.
    # Expected: handles are scalar; the combined indices have the minibatch number (0 then 1) in
    # the first column followed by the original indices, values are concatenated in order, and the
    # dense shape is 2 by 5 by 6 (the elementwise maximum of the inputs).
    with self.session(graph=ops.Graph(), use_gpu=False) as sess:
      sp_input0 = self._SparseTensorValue_5x6(np.arange(6))
      sp_input1 = self._SparseTensorValue_3x4(np.arange(6))
      handle0 = add_sparse_to_tensors_map(sp_input0, shared_name="a")
      handle1 = add_sparse_to_tensors_map(sp_input1, shared_name="a")
      self.assertEqual(handle0.get_shape(), ())
      handles_concat = array_ops_stack.stack([handle0, handle1])

      sp_out = take_many_sparse_from_tensors_map(
          sparse_map_op=handle0.op, sparse_handles=handles_concat)

      combined_indices, combined_values, combined_shape = self.evaluate(sp_out)

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], sp_input0[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], sp_input1[0])
      self.assertAllEqual(combined_values[:6], sp_input0[1])
      self.assertAllEqual(combined_values[6:], sp_input1[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testFeedAddTakeMany(self):
    # [FUNCTION DOC]
    # Scenario: add a placeholder-based sparse tensor to the map, run it twice with a 5x6 and a
    # 3x4 value fed in to get two handle numbers, convert them to a vector, and take them as one
    # batched sparse tensor.
    # Expected: same as the direct add-and-take test: minibatch numbers prepended to the indices,
    # values concatenated, and dense shape 2 by 5 by 6.
    with self.session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_3x4(np.arange(6))
      handle = add_sparse_to_tensors_map(sp_input)

      handle0_value = sess.run(handle, feed_dict={sp_input: input0_val})
      handle1_value = sess.run(handle, feed_dict={sp_input: input1_val})

      sparse_handles = ops.convert_to_tensor(
          [handle0_value, handle1_value], dtype=dtypes.int64)

      sp_roundtrip = take_many_sparse_from_tensors_map(
          sparse_map_op=handle.op, sparse_handles=sparse_handles)

      combined_indices, combined_values, combined_shape = self.evaluate(
          sp_roundtrip)

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], input0_val[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], input1_val[0])
      self.assertAllEqual(combined_values[:6], input0_val[1])
      self.assertAllEqual(combined_values[6:], input1_val[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testAddManyTakeManyRoundTrip(self):
    # [FUNCTION DOC]
    # Scenario: feed a string sparse tensor of dense shape 4 by 5 (three entries) into the
    # add-many op, which stores one sparse tensor per row, then take all the handles back.
    # Expected: four handles (one per row of the first dimension), and the round-tripped indices,
    # values and dense shape equal the originals.
    with self.session(use_gpu=False) as sess:
      # N == 4 because shape_value == [4, 5]
      indices_value = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
      values_value = np.array([b"a", b"b", b"c"])
      shape_value = np.array([4, 5], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder(dtype=dtypes.string)
      handles = add_many_sparse_to_tensors_map(sparse_tensor)
      roundtrip = take_many_sparse_from_tensors_map(
          sparse_map_op=handles.op, sparse_handles=handles)
      handles_value, roundtrip_value = sess.run(
          [handles, roundtrip],
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertEqual(handles_value.shape, (4,))
      self.assertAllEqual(roundtrip_value.indices, indices_value)
      self.assertAllEqual(roundtrip_value.values, values_value)
      self.assertAllEqual(roundtrip_value.dense_shape, shape_value)

  @test_util.run_deprecated_v1
  def testDeserializeFailsInconsistentRank(self):
    # [FUNCTION DOC]
    # Scenario: store a rank-2 and a rank-3 sparse value through a placeholder, collect both
    # handles, and take them together.
    # Expected: evaluation fails with an op error saying the rank is inconsistent across sparse
    # tensors (the message states ranks 3 and 4 after the batch dimension is added).
    with self.session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_1x1x1()
      handle = add_sparse_to_tensors_map(sp_input)

      handle0_value = sess.run(handle, feed_dict={sp_input: input0_val})
      handle1_value = sess.run(handle, feed_dict={sp_input: input1_val})

      handle_concat = ops.convert_to_tensor(
          [handle0_value, handle1_value], dtype=dtypes.int64)

      sp_roundtrip = take_many_sparse_from_tensors_map(
          sparse_map_op=handle.op, sparse_handles=handle_concat)

      with self.assertRaisesOpError(
          r"Inconsistent rank across SparseTensors: rank prior to "
          r"SparseTensor\[1\] was: 3 but rank of SparseTensor\[1\] is: 4"):
        self.evaluate(sp_roundtrip)

  @test_util.run_deprecated_v1
  def testTakeManyFailsWrongInputOp(self):
    # [FUNCTION DOC]
    # Scenario: add a sparse value to the map and evaluate its handle; take two handles, the real
    # one and the real one plus ten (never registered).
    # Expected: evaluation fails with an op error saying it was unable to find the sparse tensor
    # with handle 10 (the bad handle number as seen by the map).
    with self.session(use_gpu=False) as sess:
      input_val = self._SparseTensorValue_5x6(np.arange(6))
      handle = add_sparse_to_tensors_map(input_val)
      handle_value = self.evaluate(handle)
      bad_handle = handle_value + 10
      sp_roundtrip = take_many_sparse_from_tensors_map(
          sparse_map_op=handle.op, sparse_handles=[handle_value, bad_handle])

      with self.assertRaisesOpError(r"Unable to find SparseTensor: 10"):
        self.evaluate(sp_roundtrip)


class BenchmarkSparseTensorsMapVsSerialization(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class comparing the cost of passing a batch of sparse tensors through the
  # sparse-tensors map (add many, take many) against serializing and deserializing the same batch.
  # Holds no state.

  def benchmarkVeryLarge2DFloatSparseTensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name and the benchmark names say float, but the values are strings.
    # Builds a 64 by 10000 sparse tensor with 10000 string entries at random batch rows (seeded
    # numpy randomness), held in variables on the CPU. Builds both the map round trip and the
    # serialize-deserialize path, initializes variables, first asserts that both give identical
    # values, indices and dense shape, then runs an op benchmark of each for at least 2000
    # iterations under distinct benchmark names. Side effect: reports benchmark results.
    np.random.seed(127)
    num_elements = 10000
    batch_size = 64
    indices_batch = np.random.randint(
        batch_size, size=num_elements, dtype=np.int64)
    indices_value = np.arange(num_elements, dtype=np.int64)
    indices = np.asarray(
        sorted(zip(indices_batch, indices_value)), dtype=np.int64)
    values = ["feature_value_for_embedding_lookup"] * num_elements
    shape = np.asarray([batch_size, num_elements], dtype=np.int64)
    with session.Session(config=benchmark.benchmark_config()) as sess:
      with ops.device("/cpu:0"):
        indices = variables.Variable(indices)
        values = variables.Variable(values)
        shape = variables.Variable(shape)
        st = sparse_tensor_lib.SparseTensor(indices, values, shape)

        st_handles = add_many_sparse_to_tensors_map(st)
        st_roundtrip = take_many_sparse_from_tensors_map(
            sparse_map_op=st_handles.op, sparse_handles=st_handles)
        st_roundtrip_op = st_roundtrip.values.op

        st_serialized = sparse_ops.serialize_many_sparse(st)
        st_deserialized = sparse_ops.deserialize_many_sparse(
            st_serialized, dtype=values.dtype)
        st_deserialized_op = st_deserialized.values.op

        self.evaluate(variables.global_variables_initializer())

        st_roundtrip_values = self.evaluate(st_roundtrip)
        st_deserialized_values = self.evaluate(st_deserialized)
        np.testing.assert_equal(st_roundtrip_values.values,
                                st_deserialized_values.values)
        np.testing.assert_equal(st_roundtrip_values.indices,
                                st_deserialized_values.indices)
        np.testing.assert_equal(st_roundtrip_values.dense_shape,
                                st_deserialized_values.dense_shape)

        self.run_op_benchmark(
            sess,
            st_roundtrip_op,
            min_iters=2000,
            name="benchmark_very_large_2d_float_st_tensor_maps")
        self.run_op_benchmark(
            sess,
            st_deserialized_op,
            min_iters=2000,
            name="benchmark_very_large_2d_float_st_serialization")


if __name__ == "__main__":
  test.main()
