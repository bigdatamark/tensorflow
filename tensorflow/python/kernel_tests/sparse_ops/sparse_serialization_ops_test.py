# [MODULE DOC]
# Unit tests for serializing sparse tensors into string or variant encodings and deserializing
# them back (single, batched, nested-batched and "many" variants). A single test class uses
# shared helper methods that are run with the string encoding and, in separate tests, the
# variant encoding. Scenarios: round trips, stacking serialized items into minibatches with
# a prepended batch index, mixed dense shapes in a batch (combined shape takes the elementwise
# maximum), scalar sparse tensors, and failures for wrong requested dtype, inconsistent
# rank, unparsable serialized data and a non-sparse variant input. Most tests are graph-mode
# only (deprecated v1 decorator). No import-time side effects.
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
"""Tests for SerializeSparse."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SerializeSparseTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the sparse serialize and deserialize operations. Holds fixtures for 5x6,
  # 3x4 and rank-three single-entry sparse values with 32-bit integer values, a placeholder
  # builder, and many private helper methods that take the serialize and deserialize functions
  # (and the output encoding) as parameters so the same scenario is reused for string
  # and variant encodings and for single versus many deserialization.

  def _SparseTensorPlaceholder(self, dtype=None):
    # [FUNCTION DOC]
    # HELPER: builds graph-mode placeholder inputs for feed-based serialization tests.
    # Parameter dtype is the value dtype, defaulting to 32-bit integer when null. Returns a
    # sparse tensor with placeholder 64-bit indices, placeholder values of that dtype and a
    # placeholder 64-bit dense shape.
    if dtype is None:
      dtype = dtypes.int32
    return sparse_tensor_lib.SparseTensor(
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtype), array_ops.placeholder(dtypes.int64))

  def _SparseTensorValue_5x6(self, permutation):
    # [FUNCTION DOC]
    # HELPER: returns the 5x6 fixture as a numpy-backed sparse value with six entries and
    # 32-bit integer values 0, 10, 13, 14, 32, 33. Parameter permutation is an integer array
    # used to reorder indices and values together (identity keeps canonical order).
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2],
                    [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([5, 6]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_3x4(self, permutation):
    # [FUNCTION DOC]
    # HELPER: returns a 3x4 fixture as a numpy-backed sparse value with six entries and the same
    # 32-bit integer values as the 5x6 fixture. Parameter permutation reorders indices and
    # values together. Used as the second, differently shaped item in batch tests.
    ind = np.array([[0, 0], [1, 0], [1, 2], [1, 3], [2, 2],
                    [2, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([3, 4]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_1x1x1(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests a 1x1x1 shape, but the dense shape returned is 3 by 4 by 5.
    # HELPER: returns a rank-three sparse value with a single entry at index zero-zero-zero,
    # value 0 (32-bit integer) and dense shape 3 by 4 by 5. Used to create a rank mismatch
    # against the rank-two fixtures. Takes no arguments.
    ind = np.array([[0, 0, 0]]).astype(np.int64)
    val = np.array([0]).astype(np.int32)
    shape = np.array([3, 4, 5]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  def _testSerializeDeserializeHelper(self,
                                      serialize_fn,
                                      deserialize_fn,
                                      out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for round-trip tests. Parameters: serialize_fn and deserialize_fn are
    # the functions under test; out_type is the serialized encoding (string by default, or
    # variant). Serializes the in-order 5x6 fixture, deserializes it requesting 32-bit
    # integer values on the CPU only, evaluates, and asserts indices, values and dense shape
    # equal the original.
    with self.cached_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorValue_5x6(np.arange(6))
      serialized = serialize_fn(sp_input, out_type=out_type)
      sp_deserialized = deserialize_fn(serialized, dtype=dtypes.int32)

      indices, values, shape = self.evaluate(sp_deserialized)

      self.assertAllEqual(indices, sp_input[0])
      self.assertAllEqual(values, sp_input[1])
      self.assertAllEqual(shape, sp_input[2])

  def testSerializeDeserialize(self):
    # [FUNCTION DOC]
    # Test: round trip of a sparse tensor using the default string encoding via the shared helper.
    self._testSerializeDeserializeHelper(sparse_ops.serialize_sparse,
                                         sparse_ops.deserialize_sparse)

  def testVariantSerializeDeserialize(self):
    # [FUNCTION DOC]
    # Test: same round trip as the string version but with the variant encoding.
    self._testSerializeDeserializeHelper(sparse_ops.serialize_sparse,
                                         sparse_ops.deserialize_sparse,
                                         dtypes.variant)

  def _testSerializeDeserializeBatchHelper(self,
                                           serialize_fn,
                                           deserialize_fn,
                                           out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for batch round-trip tests. Parameters: serialize_fn, deserialize_fn
    # (the functions under test) and out_type (encoding, string by default). Serializes the
    # 5x6 fixture, stacks two copies into a vector of two serialized items, deserializes with
    # 32-bit integer values, runs a session and asserts the result has a leading batch index
    # column (0 for the first six rows, 1 for the next six), the original indices and values
    # repeated twice, and combined dense shape 2 by 5 by 6.
    with self.cached_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorValue_5x6(np.arange(6))
      serialized = serialize_fn(sp_input, out_type=out_type)
      serialized = array_ops_stack.stack([serialized, serialized])

      sp_deserialized = deserialize_fn(serialized, dtype=dtypes.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized)

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], sp_input[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], sp_input[0])
      self.assertAllEqual(combined_values[:6], sp_input[1])
      self.assertAllEqual(combined_values[6:], sp_input[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testSerializeDeserializeBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: batch round trip with string encoding using the single-item deserialize
    # function applied to a batch.
    self._testSerializeDeserializeBatchHelper(sparse_ops.serialize_sparse,
                                              sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testSerializeDeserializeManyBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: batch round trip with string encoding using the dedicated "many"
    # deserialize function.
    self._testSerializeDeserializeBatchHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: batch round trip using the variant encoding.
    self._testSerializeDeserializeBatchHelper(sparse_ops.serialize_sparse,
                                              sparse_ops.deserialize_sparse,
                                              dtypes.variant)

  def _testSerializeDeserializeBatchInconsistentShapeHelper(
      self, serialize_fn, deserialize_fn, out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for batches whose items have different dense shapes. Parameters:
    # serialize_fn, deserialize_fn, and out_type (encoding). Serializes a 5x6 and a 3x4
    # fixture, stacks them, deserializes with 32-bit integer values and asserts: batch index
    # column 0 then 1, each item's own indices and values in order, and the combined dense
    # shape is 2 by 5 by 6 (the elementwise maximum of the item shapes).
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorValue_5x6(np.arange(6))
      sp_input1 = self._SparseTensorValue_3x4(np.arange(6))
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = serialize_fn(sp_input1, out_type=out_type)
      serialized = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized, dtype=dtypes.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized)

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], sp_input0[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], sp_input1[0])
      self.assertAllEqual(combined_values[:6], sp_input0[1])
      self.assertAllEqual(combined_values[6:], sp_input1[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testSerializeDeserializeBatchInconsistentShape(self):
    # [FUNCTION DOC]
    # Graph-mode test: batch of differently shaped items, string encoding, single-item
    # deserialize function, via the shared inconsistent-shape helper.
    self._testSerializeDeserializeBatchInconsistentShapeHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeBatchInconsistentShape(self):
    # [FUNCTION DOC]
    # Graph-mode test: batch of differently shaped items using the variant encoding.
    self._testSerializeDeserializeBatchInconsistentShapeHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse,
        dtypes.variant)

  def _testSerializeDeserializeNestedBatchHelper(self,
                                                 serialize_fn,
                                                 deserialize_fn,
                                                 out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for nested (two-level) batch tests. Parameters: serialize_fn,
    # deserialize_fn and out_type. Serializes the 5x6 fixture, stacks two copies, then stacks
    # that again into a two by two grid of serialized items. After deserializing, asserts the
    # four minibatches each carry a two-column prefix of batch coordinates (0-0, 0-1, 1-0, 1-1)
    # followed by the original indices, with original values, and combined shape 2 by 2 by 5
    # by 6.
    with self.cached_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorValue_5x6(np.arange(6))
      serialized = serialize_fn(sp_input, out_type=out_type)
      serialized = array_ops_stack.stack([serialized, serialized])
      serialized = array_ops_stack.stack([serialized, serialized])

      sp_deserialized = deserialize_fn(serialized, dtype=dtypes.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized)

      # minibatch 0
      self.assertAllEqual(combined_indices[:6, :2], [[0, 0]] * 6)
      self.assertAllEqual(combined_indices[:6, 2:], sp_input[0])
      self.assertAllEqual(combined_values[:6], sp_input[1])
      # minibatch 1
      self.assertAllEqual(combined_indices[6:12, :2], [[0, 1]] * 6)
      self.assertAllEqual(combined_indices[6:12, 2:], sp_input[0])
      self.assertAllEqual(combined_values[6:12], sp_input[1])
      # minibatch 2
      self.assertAllEqual(combined_indices[12:18, :2], [[1, 0]] * 6)
      self.assertAllEqual(combined_indices[12:18, 2:], sp_input[0])
      self.assertAllEqual(combined_values[12:18], sp_input[1])
      # minibatch 3
      self.assertAllEqual(combined_indices[18:, :2], [[1, 1]] * 6)
      self.assertAllEqual(combined_indices[18:, 2:], sp_input[0])
      self.assertAllEqual(combined_values[18:], sp_input[1])

      self.assertAllEqual(combined_shape, [2, 2, 5, 6])

  @test_util.run_deprecated_v1
  def testSerializeDeserializeNestedBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: nested batch round trip with string encoding.
    self._testSerializeDeserializeNestedBatchHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeNestedBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: nested batch round trip with the variant encoding.
    self._testSerializeDeserializeNestedBatchHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse,
        dtypes.variant)

  def _testFeedSerializeDeserializeBatchHelper(self,
                                               serialize_fn,
                                               deserialize_fn,
                                               out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for feed-based batch tests. Parameters: serialize_fn,
    # deserialize_fn and out_type. Creates two placeholder sparse tensors, serializes and
    # stacks them, deserializes, then runs a session feeding the 5x6 and 3x4 fixtures. Asserts
    # the batch index column, each item's indices and values, and combined shape 2 by 5 by 6.
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_3x4(np.arange(6))
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = serialize_fn(sp_input1, out_type=out_type)
      serialized_concat = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized_concat, dtype=dtypes.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized, {sp_input0: input0_val,
                            sp_input1: input1_val})

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], input0_val[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], input1_val[0])
      self.assertAllEqual(combined_values[:6], input0_val[1])
      self.assertAllEqual(combined_values[6:], input1_val[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testFeedSerializeDeserializeBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: feed-based batch with string encoding and single-item deserialize.
    self._testFeedSerializeDeserializeBatchHelper(sparse_ops.serialize_sparse,
                                                  sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testFeedSerializeDeserializeManyBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: feed-based batch with string encoding and the "many" deserialize function.
    self._testFeedSerializeDeserializeBatchHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testFeedVariantSerializeDeserializeBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: feed-based batch with the variant encoding.
    self._testFeedSerializeDeserializeBatchHelper(sparse_ops.serialize_sparse,
                                                  sparse_ops.deserialize_sparse,
                                                  dtypes.variant)

  def _testSerializeManyShapeHelper(self,
                                    serialize_many_fn,
                                    out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for the serialize-many shape test. Parameters: serialize_many_fn
    # (splits a sparse tensor along its first dimension into one serialized item per row) and
    # out_type (encoding). Feeds a string-valued sparse tensor with dense shape 4 by 5 and three
    # entries, and asserts the serialized output array has shape 4 by 3 (one item per row,
    # three components each).
    with self.cached_session(use_gpu=False) as sess:
      # N == 4 because shape_value == [4, 5]
      indices_value = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
      values_value = np.array([b"a", b"b", b"c"])
      shape_value = np.array([4, 5], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder(dtype=dtypes.string)
      serialized = serialize_many_fn(sparse_tensor, out_type=out_type)
      serialized_value = sess.run(
          serialized,
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertEqual(serialized_value.shape, (4, 3))

  @test_util.run_deprecated_v1
  def testSerializeManyShape(self):
    # [FUNCTION DOC]
    # Graph-mode test: runs the serialize-many shape helper with the default string encoding.
    self._testSerializeManyShapeHelper(sparse_ops.serialize_many_sparse)

  def testVariantSerializeManyShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite its name this test does nothing and asserts nothing; its body is a
    # placeholder because a serialized variant cannot currently be converted to a numpy value.
    # Placeholder for the variant version of the serialize-many shape test.
    # NOTE: The following test is a no-op as it is currently not possible to
    # convert the serialized variant value to a numpy value.
    pass

  def _testSerializeManyDeserializeBatchHelper(self,
                                               serialize_many_fn,
                                               deserialize_fn,
                                               out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for serialize-many then deserialize round trips. Parameters:
    # serialize_many_fn, deserialize_fn and out_type (encoding). Feeds a string-valued sparse
    # tensor of dense shape 4 by 5 with three entries, serializes it row by row, deserializes
    # requesting string values, and asserts the recovered indices, values and dense shape equal
    # the fed ones.
    with self.cached_session(use_gpu=False) as sess:
      # N == 4 because shape_value == [4, 5]
      indices_value = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
      values_value = np.array([b"a", b"b", b"c"])
      shape_value = np.array([4, 5], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder(dtype=dtypes.string)
      serialized = serialize_many_fn(sparse_tensor, out_type=out_type)
      deserialized = deserialize_fn(serialized, dtype=dtypes.string)
      deserialized_value = sess.run(
          deserialized,
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertAllEqual(deserialized_value.indices, indices_value)
      self.assertAllEqual(deserialized_value.values, values_value)
      self.assertAllEqual(deserialized_value.dense_shape, shape_value)

  @test_util.run_deprecated_v1
  def testSerializeManyDeserializeBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: serialize-many then single-item deserialize, string encoding.
    self._testSerializeManyDeserializeBatchHelper(
        sparse_ops.serialize_many_sparse, sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testSerializeManyDeserializeManyBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: serialize-many then the "many" deserialize function, string encoding.
    self._testSerializeManyDeserializeBatchHelper(
        sparse_ops.serialize_many_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testVariantSerializeManyDeserializeBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: serialize-many then single-item deserialize with the variant encoding.
    self._testSerializeManyDeserializeBatchHelper(
        sparse_ops.serialize_many_sparse, sparse_ops.deserialize_sparse,
        dtypes.variant)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeScalar(self):
    # [FUNCTION DOC]
    # Graph-mode test: serializes a rank-zero sparse tensor (empty index row, one value 37,
    # empty dense shape) to variant, deserializes it and asserts that the indices, values and
    # dense shape come back identical to the fed ones.
    with self.session(use_gpu=False) as sess:
      indices_value = np.array([[]], dtype=np.int64)
      values_value = np.array([37], dtype=np.int32)
      shape_value = np.array([], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder()
      serialized = sparse_ops.serialize_sparse(
          sparse_tensor, out_type=dtypes.variant)
      deserialized = sparse_ops.deserialize_sparse(
          serialized, dtype=dtypes.int32)
      deserialized_value = sess.run(
          deserialized,
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertAllEqual(deserialized_value.indices, indices_value)
      self.assertAllEqual(deserialized_value.values, values_value)
      self.assertAllEqual(deserialized_value.dense_shape, shape_value)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeScalarBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test: serializes a rank-zero sparse tensor to variant, stacks two copies,
    # deserializes the batch, and asserts the result has indices 0 and 1 as a single column,
    # values 37 and 37, and dense shape of one dimension of size 2.
    with self.session(use_gpu=False) as sess:
      indices_value = np.array([[]], dtype=np.int64)
      values_value = np.array([37], dtype=np.int32)
      shape_value = np.array([], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder()
      serialized = sparse_ops.serialize_sparse(
          sparse_tensor, out_type=dtypes.variant)
      stacked = array_ops_stack.stack([serialized, serialized])
      deserialized = sparse_ops.deserialize_sparse(stacked, dtype=dtypes.int32)
      deserialized_value = sess.run(
          deserialized,
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertAllEqual(deserialized_value.indices,
                          np.array([[0], [1]], dtype=np.int64))
      self.assertAllEqual(deserialized_value.values,
                          np.array([37, 37], dtype=np.int32))
      self.assertAllEqual(deserialized_value.dense_shape,
                          np.array([2], dtype=np.int64))

  def _testDeserializeFailsWrongTypeHelper(self,
                                           serialize_fn,
                                           deserialize_fn,
                                           out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for the wrong-dtype failure tests. Parameters: serialize_fn,
    # deserialize_fn and out_type. Serializes two 32-bit integer sparse tensors, stacks them,
    # deserializes while requesting 64-bit integer values, and asserts that running raises an op
    # error saying the requested type was 64-bit integer but the item holds 32-bit values.
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_3x4(np.arange(6))
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = serialize_fn(sp_input1, out_type=out_type)
      serialized_concat = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized_concat, dtype=dtypes.int64)

      with self.assertRaisesOpError(
          r"Requested SparseTensor of type int64 but "
          r"SparseTensor\[0\].values.dtype\(\) == int32"):
        sess.run(sp_deserialized,
                 {sp_input0: input0_val,
                  sp_input1: input1_val})

  @test_util.run_deprecated_v1
  def testDeserializeFailsWrongType(self):
    # [FUNCTION DOC]
    # Graph-mode test: wrong requested dtype fails, string encoding, single-item deserialize.
    self._testDeserializeFailsWrongTypeHelper(sparse_ops.serialize_sparse,
                                              sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testDeserializeManyFailsWrongType(self):
    # [FUNCTION DOC]
    # Graph-mode test: wrong requested dtype fails, string encoding, "many" deserialize function.
    self._testDeserializeFailsWrongTypeHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testVariantDeserializeFailsWrongType(self):
    # [FUNCTION DOC]
    # Graph-mode test: wrong requested dtype fails with the variant encoding.
    self._testDeserializeFailsWrongTypeHelper(sparse_ops.serialize_sparse,
                                              sparse_ops.deserialize_sparse,
                                              dtypes.variant)

  def _testDeserializeFailsInconsistentRankHelper(self,
                                                  serialize_fn,
                                                  deserialize_fn,
                                                  out_type=dtypes.string):
    # [FUNCTION DOC]
    # HELPER: shared body for the inconsistent-rank failure tests. Parameters: serialize_fn,
    # deserialize_fn and out_type. Serializes a rank-two sparse tensor and a rank-three one,
    # stacks them, deserializes, and asserts running raises an op error saying the shape is
    # inconsistent: rank before the second item was 2 but its rank is 3.
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_1x1x1()
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = serialize_fn(sp_input1, out_type=out_type)
      serialized_concat = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized_concat, dtype=dtypes.int32)

      with self.assertRaisesOpError(
          r"Inconsistent shape across SparseTensors: rank prior to "
          r"SparseTensor\[1\] was: 2 but rank of SparseTensor\[1\] is: 3"):
        sess.run(sp_deserialized,
                 {sp_input0: input0_val,
                  sp_input1: input1_val})

  @test_util.run_deprecated_v1
  def testDeserializeFailsInconsistentRank(self):
    # [FUNCTION DOC]
    # Graph-mode test: rank mismatch fails, string encoding, single-item deserialize.
    self._testDeserializeFailsInconsistentRankHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testDeserializeManyFailsInconsistentRank(self):
    # [FUNCTION DOC]
    # Graph-mode test: rank mismatch fails, string encoding, "many" deserialize function.
    self._testDeserializeFailsInconsistentRankHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testVariantDeserializeFailsInconsistentRank(self):
    # [FUNCTION DOC]
    # Graph-mode test: rank mismatch fails with the variant encoding.
    self._testDeserializeFailsInconsistentRankHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse,
        dtypes.variant)

  def _testDeserializeFailsInvalidProtoHelper(self,
                                              serialize_fn,
                                              deserialize_fn,
                                              out_type=dtypes.string):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the out_type parameter only affects the first item; the second item is always
    # a list of plain strings, so with the variant encoding the stacking may fail differently.
    # HELPER: shared body for the invalid-serialized-data failure tests. Parameters:
    # serialize_fn, deserialize_fn and out_type. Stacks one genuinely serialized sparse tensor
    # with three arbitrary short strings, deserializes, and asserts running raises an op
    # error saying the serialized proto could not be parsed.
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = ["a", "b", "c"]
      serialized_concat = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized_concat, dtype=dtypes.int32)

      with self.assertRaisesOpError(r"Could not parse serialized proto"):
        sess.run(sp_deserialized, {sp_input0: input0_val})

  @test_util.run_deprecated_v1
  def testDeserializeFailsInvalidProto(self):
    # [FUNCTION DOC]
    # Graph-mode test: unparsable serialized data fails, single-item deserialize function.
    self._testDeserializeFailsInvalidProtoHelper(sparse_ops.serialize_sparse,
                                                 sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testDeserializeManyFailsInvalidProto(self):
    # [FUNCTION DOC]
    # Graph-mode test: unparsable serialized data fails, "many" deserialize function.
    self._testDeserializeFailsInvalidProtoHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  def testDeserializeInvalidVariant(self):
    # [FUNCTION DOC]
    # Test: creates a mutex lock object (a variant tensor that is not a serialized sparse
    # tensor) and passes it to the deserialize function inside a traced function; asserts a
    # value error saying the shape must be at least rank one is raised when the function is called.
    mu = gen_resource_variable_ops.mutex_v2()
    mu_lock = gen_resource_variable_ops.mutex_lock(mutex=mu)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced body: deserializes the captured mutex-lock variant as 32-bit integer sparse data
      # and returns the result; expected to fail during tracing or execution.
      return sparse_ops.deserialize_sparse(
          serialized_sparse=mu_lock, dtype=dtypes.int32)

    with self.assertRaisesRegex(ValueError, r"Shape must be at least rank 1"):
      f()


if __name__ == "__main__":
  test.main()
