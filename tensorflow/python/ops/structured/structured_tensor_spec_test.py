# [MODULE DOC]
# Unit tests for StructuredTensor.Spec, the type specification of structured tensors.
# Defines shared tensor and ragged spec constants and one test case that checks spec
# construction, construction errors, the value type, conversion to and from components, and
# batching and unbatching through datasets. Tests run in both graph and eager modes. Running
# the file as a script launches the test runner.
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
"""Tests for StructuredTensor.Spec."""

from absl.testing import parameterized
import numpy as np


from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import googletest


# TypeSpecs consts for fields types.
T_3 = tensor_spec.TensorSpec([3])
T_1_2 = tensor_spec.TensorSpec([1, 2])
T_1_2_8 = tensor_spec.TensorSpec([1, 2, 8])
T_1_2_3_4 = tensor_spec.TensorSpec([1, 2, 3, 4])
T_2_3 = tensor_spec.TensorSpec([2, 3])
R_1_N = ragged_tensor.RaggedTensorSpec([1, None])
R_2_N = ragged_tensor.RaggedTensorSpec([2, None])
R_1_N_N = ragged_tensor.RaggedTensorSpec([1, None, None])
R_2_1_N = ragged_tensor.RaggedTensorSpec([2, 1, None])

# TensorSpecs for nrows & row_splits in the _to_components encoding.
NROWS_SPEC = tensor_spec.TensorSpec([], dtypes.int64)
PARTITION_SPEC = row_partition.RowPartitionSpec()


# pylint: disable=g-long-lambda
@test_util.run_all_in_graph_and_eager_modes
class StructuredTensorSpecTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Test case for StructuredTensor.Spec: construction and validation, value type, round trips
  # through the component encoding, and dataset batch/unbatch behavior. Runs in graph and eager
  # modes. Overrides the equality assertion to compare structured tensors field by field and
  # provides a helper for comparing mappings of tensors.

  # TODO(edloper): Add a subclass of TensorFlowTestCase that overrides
  # assertAllEqual etc to work with StructuredTensors.
  def assertAllEqual(self, a, b, msg=None):
    # [FUNCTION DOC]
    # Equality assertion extended to structured tensors. Inputs: a, b (values to compare) and msg
    # (optional message). If neither is structured, defers to the base assertion and returns its
    # result. If only one is structured it raises a value error (not supported yet). If both
    # are, asserts equal shape text, equal field names, and recursively equal field values.
    if not (isinstance(a, structured_tensor.StructuredTensor) or
            isinstance(b, structured_tensor.StructuredTensor)):
      return super(StructuredTensorSpecTest, self).assertAllEqual(a, b, msg)
    if not (isinstance(a, structured_tensor.StructuredTensor) and
            isinstance(b, structured_tensor.StructuredTensor)):
      # TODO(edloper) Add support for this once structured_factory_ops is added.
      raise ValueError('Not supported yet')

    self.assertEqual(repr(a.shape), repr(b.shape))
    self.assertEqual(set(a.field_names()), set(b.field_names()))
    for field in a.field_names():
      self.assertAllEqual(a.field_value(field), b.field_value(field))

  def assertAllTensorsEqual(self, x, y):
    # [FUNCTION DOC]
    # HELPER: Comparison helper for tests with mappings of tensors.
    # KNOWN ISSUE: it is defined but never called in this file (dead code).
    # Inputs x and y must both be dicts (otherwise an assertion error is raised). Asserts the
    # key sets are equal and that the values under each key are equal. Returns nothing.
    assert isinstance(x, dict) and isinstance(y, dict)
    self.assertEqual(set(x), set(y))
    for key in x:
      self.assertAllEqual(x[key], y[key])

  def testConstruction(self):
    # [FUNCTION DOC]
    # Builds two specs directly from a ragged shape spec and a mapping of field specs: first a
    # rank-3 spec with one field, then a rank-2 spec with tensor, ragged and nested-spec fields.
    # Asserts that the stored shape equals the requested static shape and the stored field
    # specs equal the supplied mapping.
    spec1_fields = dict(a=T_1_2_3_4)
    spec1 = StructuredTensor.Spec(
        _ragged_shape=DynamicRaggedShape.Spec(
            row_partitions=[],
            static_inner_shape=tensor_shape.TensorShape([1, 2, 3]),
            dtype=dtypes.int64),
        _fields=spec1_fields)
    self.assertEqual(spec1._shape, (1, 2, 3))
    self.assertEqual(spec1._field_specs, spec1_fields)

    spec2_fields = dict(a=T_1_2, b=T_1_2_8, c=R_1_N, d=R_1_N_N, s=spec1)
    spec2 = StructuredTensor.Spec(
        _ragged_shape=DynamicRaggedShape.Spec(
            row_partitions=[],
            static_inner_shape=tensor_shape.TensorShape([1, 2]),
            dtype=dtypes.int64),
        _fields=spec2_fields)
    self.assertEqual(spec2._shape, (1, 2))
    self.assertEqual(spec2._field_specs, spec2_fields)

  # Note that there is no error for creating a spec without known rank.
  @parameterized.parameters([
      (None,),
      ({1: tensor_spec.TensorSpec(None)},),
      ({'x': 0},),
  ])
  def testConstructionErrors(self, field_specs):
    # [FUNCTION DOC]
    # Parameterized test: constructing a spec with invalid field specs (nothing, a non-string
    # key, or a value that is not a type spec) with an otherwise valid scalar shape must raise
    # a type error.
    with self.assertRaises(TypeError):
      structured_tensor.StructuredTensor.Spec(
          _ragged_shape=DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=[],
              dtype=dtypes.int64),
          _fields=field_specs)

  def testValueType(self):
    # [FUNCTION DOC]
    # Builds a small spec and asserts its value type is the StructuredTensor class.
    spec1 = StructuredTensor.Spec(
        _ragged_shape=DynamicRaggedShape.Spec(
            row_partitions=[],
            static_inner_shape=[1, 2],
            dtype=dtypes.int64),
        _fields=dict(a=T_1_2))
    self.assertEqual(spec1.value_type, StructuredTensor)

  @parameterized.parameters([
      {
          'shape': [],
          'fields': dict(x=[[1.0, 2.0]]),
          'field_specs': dict(x=T_1_2),
      },
      {
          'shape': [2],
          'fields': dict(
              a=ragged_factory_ops.constant_value([[1.0], [2.0, 3.0]]),
              b=[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
          'field_specs': dict(a=R_2_N, b=T_2_3),
      },
  ])  # pyformat: disable
  def testToFromComponents(self, shape, fields, field_specs):
    # [FUNCTION DOC]
    # Parameterized test of the component encoding round trip. Inputs: shape, a mapping of field
    # values and the matching field specs (a scalar case with a dense field and a vector case
    # with ragged and dense fields). Builds a structured tensor, encodes it with a spec into
    # components, decodes it back, and asserts the reconstruction equals the original.
    struct = StructuredTensor.from_fields(fields, shape)
    spec = StructuredTensor.Spec(_ragged_shape=DynamicRaggedShape.Spec(
        row_partitions=[],
        static_inner_shape=shape,
        dtype=dtypes.int64), _fields=field_specs)
    actual_components = spec._to_components(struct)
    rt_reconstructed = spec._from_components(actual_components)
    self.assertAllEqual(struct, rt_reconstructed)

  def testToFromComponentsEmptyScalar(self):
    # [FUNCTION DOC]
    # Round trip through components for a scalar structured tensor with no fields; asserts the
    # reconstruction equals the original.
    struct = StructuredTensor.from_fields(fields={}, shape=[])
    spec = struct._type_spec
    components = spec._to_components(struct)
    rt_reconstructed = spec._from_components(components)
    self.assertAllEqual(struct, rt_reconstructed)

  def testToFromComponentsEmptyTensor(self):
    # [FUNCTION DOC]
    # Round trip through components for a structured tensor of shape one by two by three with
    # no fields; asserts the reconstruction equals the original.
    struct = StructuredTensor.from_fields(fields={}, shape=[1, 2, 3])
    spec = struct._type_spec
    components = spec._to_components(struct)
    rt_reconstructed = spec._from_components(components)
    self.assertAllEqual(struct, rt_reconstructed)

  @parameterized.parameters([
      {
          'unbatched': lambda: [
              StructuredTensor.from_fields({'a': 1, 'b': [5, 6]}),
              StructuredTensor.from_fields({'a': 2, 'b': [7, 8]})],
          'batch_size': 2,
          'batched': lambda: StructuredTensor.from_fields(shape=[2], fields={
              'a': [1, 2],
              'b': [[5, 6], [7, 8]]}),
      },
      {
          'unbatched': lambda: [
              StructuredTensor.from_fields(shape=[3], fields={
                  'a': [1, 2, 3],
                  'b': [[5, 6], [6, 7], [7, 8]]}),
              StructuredTensor.from_fields(shape=[3], fields={
                  'a': [2, 3, 4],
                  'b': [[2, 2], [3, 3], [4, 4]]})],
          'batch_size': 2,
          'batched': lambda: StructuredTensor.from_fields(shape=[2, 3], fields={
              'a': [[1, 2, 3], [2, 3, 4]],
              'b': [[[5, 6], [6, 7], [7, 8]],
                    [[2, 2], [3, 3], [4, 4]]]}),
      },
      {
          'unbatched': lambda: [
              StructuredTensor.from_fields(shape=[], fields={
                  'a': 1,
                  'b': StructuredTensor.from_fields({'x': [5]})}),
              StructuredTensor.from_fields(shape=[], fields={
                  'a': 2,
                  'b': StructuredTensor.from_fields({'x': [6]})})],
          'batch_size': 2,
          'batched': lambda: StructuredTensor.from_fields(shape=[2], fields={
              'a': [1, 2],
              'b': StructuredTensor.from_fields(shape=[2], fields={
                  'x': [[5], [6]]})}),
      },
      {
          'unbatched': lambda: [
              StructuredTensor.from_fields(shape=[], fields={
                  'Ragged3d': ragged_factory_ops.constant_value([[1, 2], [3]]),
                  'Ragged2d': ragged_factory_ops.constant_value([1]),
              }),
              StructuredTensor.from_fields(shape=[], fields={
                  'Ragged3d': ragged_factory_ops.constant_value([[1]]),
                  'Ragged2d': ragged_factory_ops.constant_value([2, 3]),
              })],
          'batch_size': 2,
          'batched': lambda: StructuredTensor.from_fields(shape=[2], fields={
              'Ragged3d': ragged_factory_ops.constant_value(
                  [[[1, 2], [3]], [[1]]]),
              'Ragged2d': ragged_factory_ops.constant_value([[1], [2, 3]]),
          }),
          'use_only_batched_spec': True,
      },
  ])  # pyformat: disable
  def testBatchUnbatchValues(self,
                             unbatched,
                             batch_size,
                             batched,
                             use_only_batched_spec=False):
    # [FUNCTION DOC]
    # Parameterized test of dataset batching and unbatching with structured tensors. Inputs:
    # unbatched and batched (callables producing a list of structured tensors and the equivalent
    # batched one; deferred because they create tensors), batch_size (given by every case but
    # unused here) and use_only_batched_spec (optional, default false). Builds a dataset from the
    # batched value, unbatches it and, in eager mode, re-batches and compares to the batched
    # value. Unless use_only_batched_spec is set, also builds a dataset from a generator of the
    # unbatched values using the spec inferred from the first one, batches it by two and (in
    # eager mode) compares to the batched value.
    batched = batched()  # Deferred init because it creates tensors.
    unbatched = unbatched()  # Deferred init because it creates tensors.

    def unbatch_gen():
      # [FUNCTION DOC]
      # HELPER: Generator feeding the unbatched values to the dataset-from-generator test.
      # Yields each element of the enclosing list of unbatched structured tensors in order.
      for i in unbatched:
        yield i

    ds = dataset_ops.Dataset.from_tensors(batched)
    ds2 = ds.unbatch()
    if context.executing_eagerly():
      v = list(ds2.batch(2))
      self.assertAllEqual(v[0], batched)

    if not use_only_batched_spec:
      unbatched_spec = type_spec.type_spec_from_value(unbatched[0])

      dsu = dataset_ops.Dataset.from_generator(
          unbatch_gen, output_signature=unbatched_spec)
      dsu2 = dsu.batch(2)
      if context.executing_eagerly():
        v = list(dsu2)
        self.assertAllEqual(v[0], batched)

  def _lambda_for_fields(self):
    # [FUNCTION DOC]
    # HELPER: Fixture factory for field mappings.
    # KNOWN ISSUE: it is not used anywhere in this file (dead code).
    # Returns a callable that, when invoked, builds a mapping from field names to values of
    # varied kinds: dense numpy arrays, ragged tensors of several ragged ranks and dtypes, and
    # nested structured tensors. Deferring creation avoids making tensors at definition time.
    return lambda: {
        'a':
            np.ones([1, 2, 3, 1]),
        'b':
            np.ones([1, 2, 3, 1, 5]),
        'c':
            ragged_factory_ops.constant(
                np.zeros([1, 2, 3, 1], dtype=np.uint8), dtype=dtypes.uint8),
        'd':
            ragged_factory_ops.constant(
                np.zeros([1, 2, 3, 1, 3]).tolist(), ragged_rank=1),
        'e':
            ragged_factory_ops.constant(
                np.zeros([1, 2, 3, 1, 2, 2]).tolist(), ragged_rank=2),
        'f':
            ragged_factory_ops.constant(
                np.zeros([1, 2, 3, 1, 3]), dtype=dtypes.float32),
        'g':
            StructuredTensor.from_pyval([[
                [  # pylint: disable=g-complex-comprehension
                    [{
                        'x': j,
                        'y': k
                    }] for k in range(3)
                ] for j in range(2)
            ]]),
        'h':
            StructuredTensor.from_pyval([[
                [  # pylint: disable=g-complex-comprehension
                    [[
                        {
                            'x': j,
                            'y': k,
                            'z': z
                        } for z in range(j)
                    ]] for k in range(3)
                ] for j in range(2)
            ]]),
    }


if __name__ == '__main__':
  googletest.main()
