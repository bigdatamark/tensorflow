# [MODULE DOC]
# Unit tests for the sparse tensor module: construction and conversion of SparseTensor and
# SparseTensorValue, static shape inference (including inside traced functions), consumers,
# eager helpers, conversion helper behavior, and the SparseTensorSpec type specification
# (serialization, component specs, component and tensor-list round trips, batching).
# Run as a standard test script; no import-time side effects beyond imports.
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

"""Tests for tensorflow.python.framework.sparse_tensor."""
from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


class SparseTensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the SparseTensor class itself: construction, shape, type checks, consumers,
  # replacing values and eager-only helpers.

  def testPythonConstruction(self):
    # [FUNCTION DOC]
    # Scenario: builds sparse tensors from Python lists directly, from a SparseTensorValue and
    # from another SparseTensor. Asserts int64 indices, string values, int64 dense shape, static
    # shape (4, 5), and that evaluating the tensor returns the original component contents.
    indices = [[1, 2], [2, 0], [3, 4]]
    values = [b"a", b"b", b"c"]
    shape = [4, 5]
    sp_value = sparse_tensor.SparseTensorValue(indices, values, shape)
    for sp in [
        sparse_tensor.SparseTensor(indices, values, shape),
        sparse_tensor.SparseTensor.from_value(sp_value),
        sparse_tensor.SparseTensor.from_value(
            sparse_tensor.SparseTensor(indices, values, shape))]:
      self.assertEqual(sp.indices.dtype, dtypes.int64)
      self.assertEqual(sp.values.dtype, dtypes.string)
      self.assertEqual(sp.dense_shape.dtype, dtypes.int64)
      self.assertEqual(sp.get_shape(), (4, 5))

      value = self.evaluate(sp)
      self.assertAllEqual(indices, value.indices)
      self.assertAllEqual(values, value.values)
      self.assertAllEqual(shape, value.dense_shape)
      sp_value = self.evaluate(sp)
      self.assertAllEqual(sp_value.indices, value.indices)
      self.assertAllEqual(sp_value.values, value.values)
      self.assertAllEqual(sp_value.dense_shape, value.dense_shape)

  def testShape(self):
    # [FUNCTION DOC]
    # Scenario: a sparse tensor of dense shape 3 by 4 is passed through a traced function that
    # transposes it. Asserts the traced result still has a static rank of 2.

    @def_function.function
    def test_fn(tensor):
      # [FUNCTION DOC]
      # HELPER: Traced function body for the shape test. Takes a sparse tensor, transposes it,
      # asserts its static rank is 2 and returns the transposed tensor.
      tensor = sparse_ops.sparse_transpose(tensor)
      self.assertEqual(tensor.shape.rank, 2)
      return tensor

    tensor = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1., 2], dense_shape=[3, 4])
    test_fn(tensor)

  def testIsSparse(self):
    # [FUNCTION DOC]
    # Scenario: checks is_sparse on a number, a string, a numpy scalar array (all false), and on
    # a SparseTensor and a SparseTensorValue (both true).
    self.assertFalse(sparse_tensor.is_sparse(3))
    self.assertFalse(sparse_tensor.is_sparse("foo"))
    self.assertFalse(sparse_tensor.is_sparse(np.array(3)))
    self.assertTrue(
        sparse_tensor.is_sparse(sparse_tensor.SparseTensor([[0]], [0], [1])))
    self.assertTrue(
        sparse_tensor.is_sparse(
            sparse_tensor.SparseTensorValue([[0]], [0], [1])))

  def testConsumers(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a sparse tensor is used in a sparse-dense matmul, so it has one
    # consumer (that op); then converted to dense, so it has two consumers including the
    # to-dense op. Asserts the consumer counts and membership.
    with context.graph_mode():
      sp = sparse_tensor.SparseTensor([[0, 0], [1, 2]], [1.0, 3.0], [3, 4])
      w = ops.convert_to_tensor(np.ones([4, 1], np.float32))
      out = sparse_ops.sparse_tensor_dense_matmul(sp, w)
      self.assertEqual(len(sp.consumers()), 1)
      self.assertEqual(sp.consumers()[0], out.op)

      dense = sparse_ops.sparse_tensor_to_dense(sp)
      self.assertEqual(len(sp.consumers()), 2)
      self.assertIn(dense.op, sp.consumers())
      self.assertIn(out.op, sp.consumers())

  def testWithValues(self):
    # [FUNCTION DOC]
    # Scenario: replaces the values of a sparse tensor. Asserts indices and dense shape are
    # preserved and values are updated; then asserts that supplying values of the wrong rank
    # raises an invalid-argument or value error.
    source = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1., 2], dense_shape=[3, 4])
    new_tensor = source.with_values([5.0, 1.0])
    self.assertAllEqual(new_tensor.indices, source.indices)
    self.assertAllEqual(new_tensor.values, [5.0, 1.0])
    self.assertAllEqual(new_tensor.dense_shape, source.dense_shape)

    # ensure new value's shape is checked
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      source.with_values([[5.0, 1.0]])

  @test_util.run_in_graph_and_eager_modes
  def testIsEager(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): builds a small sparse tensor and, when running eagerly,
    # asserts the eager check reports true. Nothing is asserted in graph mode.
    st = sparse_tensor.SparseTensor([[0, 0], [0, 1]], [1.0, 3.0], [2, 2])
    if context.executing_eagerly():
      self.assertTrue(st._is_eager())

  @test_util.run_in_graph_and_eager_modes
  def testNumpy(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): in graph mode the numpy conversion must raise a value
    # error; in eager mode it must return the dense 2 by 2 array with the two set entries.
    st1 = sparse_tensor.SparseTensor([[0, 0], [0, 1]], [1.0, 3.0], [2, 2])
    if not context.executing_eagerly():
      with self.assertRaises(ValueError):
        st1._numpy()
    else:
      self.assertAllEqual(st1._numpy(), [[1.0, 3.0], [0.0, 0.0]])


class ConvertToTensorOrSparseTensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for convert_to_tensor_or_sparse_tensor with dense and sparse inputs.

  def test_convert_dense(self):
    # [FUNCTION DOC]
    # Scenario: a plain list is converted; evaluating the result must equal the original list.
    value = [42, 43]
    from_value = sparse_tensor.convert_to_tensor_or_sparse_tensor(
        value)
    self.assertAllEqual(value, self.evaluate(from_value))

  def test_convert_sparse(self):
    # [FUNCTION DOC]
    # Scenario: converts both a SparseTensorValue and a SparseTensor, evaluates the results and
    # asserts indices, values and dense shape are unchanged from the original value.
    indices = [[0, 1], [1, 0]]
    values = [42, 43]
    shape = [2, 2]
    sparse_tensor_value = sparse_tensor.SparseTensorValue(
        indices, values, shape)
    st = sparse_tensor.SparseTensor.from_value(sparse_tensor_value)
    from_value = self.evaluate(
        sparse_tensor.convert_to_tensor_or_sparse_tensor(sparse_tensor_value))
    from_tensor = self.evaluate(
        sparse_tensor.convert_to_tensor_or_sparse_tensor(st))
    for convertee in [from_value, from_tensor]:
      self.assertAllEqual(sparse_tensor_value.indices, convertee.indices)
      self.assertAllEqual(sparse_tensor_value.values, convertee.values)
      self.assertAllEqual(
          sparse_tensor_value.dense_shape, convertee.dense_shape)


class SparseTensorShapeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for static shape inference of sparse tensors from constant, partially known, negative
  # and fully unknown dense shapes, including inside traced functions.

  def test_simple(self):
    # [FUNCTION DOC]
    # Scenario: constant dense shape (5, 5). Asserts the shape property is a TensorShape equal
    # to 5 by 5 and the dense_shape property is a tensor.
    indices = [[0, 2]]
    values = [1]
    dense_shape = [5, 5]
    sp = sparse_tensor.SparseTensor(indices, values, dense_shape)

    self.assertIsInstance(sp.shape, tensor_shape.TensorShape)
    self.assertIsInstance(sp.dense_shape, tensor_lib.Tensor)
    self.assertEqual(sp.shape.as_list(), [5, 5])

  def test_unknown_shape(self):
    # [FUNCTION DOC]
    # Scenario: dense shape is a traced input vector of length 2 with unknown values. Asserts the
    # static shape is two unknown dimensions, checked while tracing a concrete function.

    @def_function.function
    def my_func(dense_shape):
      # [FUNCTION DOC]
      # HELPER: Traced body. Builds a sparse tensor from fixed indices and values and the given
      # dense shape tensor, asserts its static shape has two unknown dimensions and returns it.
      indices = [[0, 2]]
      values = [1]
      sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
      self.assertEqual(sp.shape.as_list(), [None, None])
      return sp

    my_func.get_concrete_function(
        dense_shape=tensor_lib.TensorSpec(
            dtype=dtypes.int64, shape=[2,]))

  def test_partial_shape(self):
    # [FUNCTION DOC]
    # Scenario: dense shape is built from a traced scalar input (unknown) and a constant 3.
    # Asserts the static shape is unknown by 3, while tracing a concrete function.

    @def_function.function
    def my_func(x):
      # [FUNCTION DOC]
      # HELPER: Traced body. Combines the input scalar and a constant into the dense shape,
      # builds a sparse tensor, asserts its static shape has an unknown first and known
      # second dimension, and returns it.
      indices = [[0, 2]]
      values = [1]
      y = ops.convert_to_tensor(3, dtype=dtypes.int64)
      dense_shape = [x, y]
      sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
      self.assertEqual(sp.shape.as_list(), [None, 3])
      return sp

    my_func.get_concrete_function(
        x=tensor_lib.TensorSpec(dtype=dtypes.int64, shape=[]))

  def test_neg_shape(self):
    # [FUNCTION DOC]
    # Scenario: constant dense shape containing minus one for the first dimension. Asserts the
    # negative size is treated as unknown, giving static shape unknown by 5.
    indices = [[0, 2]]
    values = [1]
    dense_shape = [-1, 5]
    sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
    self.assertEqual(sp.shape.as_list(), [None, 5])

  def test_unknown_tensor_shape(self):
    # [FUNCTION DOC]
    # Scenario: dense shape is computed at run time from the shape of a traced 2-D input of
    # unknown size, cast to int64. Asserts the static shape is two unknown dimensions.

    @def_function.function
    def my_func(x):
      # [FUNCTION DOC]
      # HELPER: Traced body. Computes the shape of x as int64, builds a sparse tensor with it,
      # asserts the static shape is two unknown dimensions and returns the sparse tensor.
      indices = [[0, 0]]
      values = [1]
      dense_shape = array_ops.shape(x)
      dense_shape = math_ops.cast(dense_shape, dtypes.int64)

      sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
      self.assertEqual(sp.shape.as_list(), [None, None])
      return sp

    my_func.get_concrete_function(
        x=tensor_lib.TensorSpec(dtype=dtypes.int64, shape=[None, None]))

  def test_unknown_rank(self):
    # [FUNCTION DOC]
    # Scenario: dense shape is a traced vector of unknown length, so the rank is unknown.
    # Asserts the static shape's rank is nothing/unknown.

    @def_function.function
    def my_func(dense_shape):
      # [FUNCTION DOC]
      # HELPER: Traced body. Builds a sparse tensor with the given unknown-length dense shape,
      # asserts the static rank is unknown and returns the sparse tensor.
      indices = [[0, 0]]
      values = [1]
      sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
      self.assertEqual(sp.shape.rank, None)
      return sp

    my_func.get_concrete_function(
        dense_shape=tensor_lib.TensorSpec(dtype=dtypes.int64, shape=[None]))


@test_util.run_all_in_graph_and_eager_modes
class SparseTensorSpecTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for SparseTensorSpec, run in both graph and eager modes. Covers
  # construction, value type, serialization, component specs, component and tensor-list round
  # trips, flat specs, full-type info, batching and unbatching.

  def assertAllTensorsEqual(self, list1, list2):
    # [FUNCTION DOC]
    # HELPER: Custom assertion. Asserts the two lists have the same length and that each pair of
    # corresponding elements is equal as arrays.
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    # [FUNCTION DOC]
    # Scenario: a default spec has unknown rank and 32-bit float dtype; a spec built with a
    # shape of two unknown dimensions and string dtype reports those values.
    spec1 = sparse_tensor.SparseTensorSpec()
    self.assertEqual(spec1.shape.rank, None)
    self.assertEqual(spec1.dtype, dtypes.float32)

    spec2 = sparse_tensor.SparseTensorSpec([None, None], dtypes.string)
    self.assertEqual(spec2.shape.as_list(), [None, None])
    self.assertEqual(spec2.dtype, dtypes.string)

  def testValueType(self):
    # [FUNCTION DOC]
    # Scenario: asserts the spec's value type is the SparseTensor class.
    spec1 = sparse_tensor.SparseTensorSpec()
    self.assertEqual(spec1.value_type, sparse_tensor.SparseTensor)

  @parameterized.parameters([
      (sparse_tensor.SparseTensorSpec(),
       (tensor_shape.TensorShape(None), dtypes.float32)),
      (sparse_tensor.SparseTensorSpec(shape=[5, None, None]),
       (tensor_shape.TensorShape([5, None, None]), dtypes.float32)),
      (sparse_tensor.SparseTensorSpec(dtype=dtypes.int32),
       (tensor_shape.TensorShape(None), dtypes.int32)),
  ])  # pyformat: disable
  def testSerialize(self, st_spec, expected):
    # [FUNCTION DOC]
    # Parameterized over specs with default, partial shape and int dtype. Asserts the serialized
    # form equals the expected pair of shape and dtype, compared by text representation because
    # shape equality is unconventional.
    serialization = st_spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      (sparse_tensor.SparseTensorSpec(dtype=dtypes.string), [
          tensor_lib.TensorSpec([None, None], dtypes.int64),
          tensor_lib.TensorSpec([None], dtypes.string),
          tensor_lib.TensorSpec([None], dtypes.int64)
      ]),
      (sparse_tensor.SparseTensorSpec(shape=[5, None, None]), [
          tensor_lib.TensorSpec([None, 3], dtypes.int64),
          tensor_lib.TensorSpec([None], dtypes.float32),
          tensor_lib.TensorSpec([3], dtypes.int64)
      ]),
  ])
  def testComponentSpecs(self, st_spec, expected):
    # [FUNCTION DOC]
    # Parameterized over an unknown-rank string spec and a rank-3 spec. Asserts the component
    # specs list (indices, values, dense shape) matches the expected specs.
    self.assertEqual(st_spec._component_specs, expected)

  @parameterized.parameters([
      {
          "st_spec": sparse_tensor.SparseTensorSpec(),
          "indices": [[0, 1], [10, 8]],
          "values": [3.0, 5.0],
          "dense_shape": [100, 100]
      },
      {
          "st_spec": sparse_tensor.SparseTensorSpec([100, None, None]),
          "indices": [[0, 1, 3], [10, 8, 2]],
          "values": [3.0, 5.0],
          "dense_shape": [100, 20, 20]
      },
  ])
  def testToFromComponents(self, st_spec, indices, values, dense_shape):
    # [FUNCTION DOC]
    # Parameterized over an unknown-shape spec and a partially known rank-3 spec. Converts a
    # sparse tensor to components, asserts they equal the inputs, rebuilds a tensor and asserts
    # indices, values and dense shape round-trip.
    st = sparse_tensor.SparseTensor(indices, values, dense_shape)
    actual_components = st_spec._to_components(st)
    self.assertAllTensorsEqual(actual_components,
                               [indices, values, dense_shape])
    st_reconstructed = st_spec._from_components(actual_components)
    self.assertAllEqual(st.indices, st_reconstructed.indices)
    self.assertAllEqual(st.values, st_reconstructed.values)
    self.assertAllEqual(st.dense_shape, st_reconstructed.dense_shape)

  def testFromComponentsDynamicDenseShapeTensor(self):
    # [FUNCTION DOC]
    # Scenario: a traced function takes a sparse tensor input described by a spec with shape
    # unknown by 10 by 100. Tracing a concrete function must give the input that static shape.
    @def_function.function(input_signature=[
        sparse_tensor.SparseTensorSpec([None, 10, 100])])
    def sparse_fun(st):
      # [FUNCTION DOC]
      # HELPER: Traced body. Asserts the input's static shape matches the spec shape and returns
      # its dense shape tensor.
      self.assertEqual(st.get_shape().as_list(), [None, 10, 100])
      return st.dense_shape

    # Force tracing the TF function.
    _ = sparse_fun.get_concrete_function()

  @test_util.run_v1_only("SparseTensorValue is deprecated in v2")
  def testFromNumpyComponents(self):
    # [FUNCTION DOC]
    # Scenario (version-1 only): rebuilds from numpy arrays using a default spec. Asserts the
    # result is a SparseTensorValue holding the same indices, values and dense shape.
    indices = np.array([[0], [8]])
    values = np.array([1.0, 9.0])
    dense_shape = np.array([100])
    spec = sparse_tensor.SparseTensorSpec()
    st = spec._from_components([indices, values, dense_shape])
    self.assertIsInstance(st, sparse_tensor.SparseTensorValue)
    self.assertAllEqual(st.indices, indices)
    self.assertAllEqual(st.values, values)
    self.assertAllEqual(st.dense_shape, dense_shape)

  @parameterized.parameters([
      sparse_tensor.SparseTensorSpec(dtype=dtypes.string),
      sparse_tensor.SparseTensorSpec(shape=[5, None, None]),
  ])
  def testFlatTensorSpecs(self, st_spec):
    # [FUNCTION DOC]
    # Parameterized over a string spec and a partial-shape spec. Asserts the flat tensor specs
    # are one variant-typed spec of unknown shape.
    self.assertEqual(st_spec._flat_tensor_specs,
                     [tensor_lib.TensorSpec(None, dtypes.variant)])

  @parameterized.parameters([
      dtypes.float32,
      dtypes.string,
  ])
  def testFullTypesForFlatTensors(self, dt):
    # [FUNCTION DOC]
    # Parameterized over float and string dtypes. Asserts the full-type list computed for the
    # spec's flat tensors has one entry per flat spec and is a single unset type definition.
    st_spec = sparse_tensor.SparseTensorSpec(dtype=dt)
    full_type_list = fulltypes_for_flat_tensors(st_spec)
    expect = [full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET)]
    self.assertEqual(len(st_spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)

  @parameterized.parameters([
      {
          "st_spec": sparse_tensor.SparseTensorSpec(),
          "indices": [[0, 1], [10, 8]],
          "values": [3.0, 5.0],
          "dense_shape": [100, 100]
      },
      {
          "st_spec": sparse_tensor.SparseTensorSpec([100, None, None]),
          "indices": [[0, 1, 3], [10, 8, 2]],
          "values": [3.0, 5.0],
          "dense_shape": [100, 20, 20]
      },
  ])
  def testToFromTensorList(self, st_spec, indices, values, dense_shape):
    # [FUNCTION DOC]
    # Parameterized over two specs. Serializes a sparse tensor to a tensor list, deserializes
    # it and asserts indices, values and dense shape are preserved.
    st = sparse_tensor.SparseTensor(indices, values, dense_shape)
    tensor_list = st_spec._to_tensor_list(st)
    st_reconstructed = st_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(st.indices, st_reconstructed.indices)
    self.assertAllEqual(st.values, st_reconstructed.values)
    self.assertAllEqual(st.dense_shape, st_reconstructed.dense_shape)

  @parameterized.parameters([
      (sparse_tensor.SparseTensorSpec([2, None], dtypes.float32), 32,
       sparse_tensor.SparseTensorSpec([32, 2, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([4, None], dtypes.float32), None,
       sparse_tensor.SparseTensorSpec([None, 4, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([2], dtypes.float32), 32,
       sparse_tensor.SparseTensorSpec([32, 2], dtypes.float32)),
  ])
  def testBatch(self, spec, batch_size, expected):
    # [FUNCTION DOC]
    # Parameterized over specs and batch sizes (including an unknown size). Asserts batching
    # prepends the batch size to the spec shape.
    self.assertEqual(spec._batch(batch_size), expected)

  @parameterized.parameters([
      (sparse_tensor.SparseTensorSpec([32, None, None], dtypes.float32),
       sparse_tensor.SparseTensorSpec([None, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([None, None, None], dtypes.float32),
       sparse_tensor.SparseTensorSpec([None, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([32, 2], dtypes.float32),
       sparse_tensor.SparseTensorSpec([2], dtypes.float32)),
  ])
  def testUnbatch(self, spec, expected):
    # [FUNCTION DOC]
    # Parameterized over batched specs. Asserts unbatching drops the leading dimension.
    self.assertEqual(spec._unbatch(), expected)


if __name__ == "__main__":
  googletest.main()
