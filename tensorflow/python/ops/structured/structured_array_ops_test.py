# [MODULE DOC]
# Unit tests for the StructuredTensor array-op dispatch implementations in
# structured_array_ops: expand_dims, size, shape, zeros_like, ones_like, rank, concat,
# random_shuffle, gather and the private helpers that build structured tensors shaped like
# other tensors. Tests run in both graph and eager modes through one test class with a custom
# equality assertion for structured tensors. Running the file as a script starts the test runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for structured_array_ops."""

from absl.testing import parameterized

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape
from tensorflow.python.ops.structured import structured_array_ops
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


# TODO(martinz):create StructuredTensorTestCase.
# pylint: disable=g-long-lambda
@test_util.run_all_in_graph_and_eager_modes
class StructuredArrayOpsTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the array ops registered for StructuredTensor (see structured_array_ops).
  # Decorated to run every test in graph and eager modes; parameterized tests use named
  # parameter sets. Overrides assertAllEqual so structured tensors can be compared with
  # python-value expectations.

  def assertAllEqual(self, a, b, msg=None):
    # [FUNCTION DOC]
    # HELPER: custom equality assertion for tests in this class.
    # Falls back to the base assertion when neither a nor b is a StructuredTensor. Otherwise a
    # non-structured side is converted from python values, both must have the same nested
    # structure, and the contained tensors are evaluated and compared pairwise. Fails the test
    # on structure mismatch or unequal values. msg is an optional failure message.
    if not (isinstance(a, structured_tensor.StructuredTensor) or
            isinstance(b, structured_tensor.StructuredTensor)):
      return super(StructuredArrayOpsTest, self).assertAllEqual(a, b, msg)

    if not isinstance(a, structured_tensor.StructuredTensor):
      a = structured_tensor.StructuredTensor.from_pyval(a)
    elif not isinstance(b, structured_tensor.StructuredTensor):
      b = structured_tensor.StructuredTensor.from_pyval(b)

    try:
      nest.assert_same_structure(a, b, expand_composites=True)
    except (TypeError, ValueError) as e:
      self.assertIsNone(e, (msg + ": " if msg else "") + str(e))
    a_tensors = [
        x for x in nest.flatten(a, expand_composites=True)
        if isinstance(x, tensor.Tensor)
    ]
    b_tensors = [
        x for x in nest.flatten(b, expand_composites=True)
        if isinstance(x, tensor.Tensor)
    ]
    self.assertLen(a_tensors, len(b_tensors))
    a_arrays, b_arrays = self.evaluate((a_tensors, b_tensors))
    for a_array, b_array in zip(a_arrays, b_arrays):
      self.assertAllEqual(a_array, b_array, msg)

  def _assertStructuredEqual(self, a, b, msg, check_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this helper is never called anywhere in this file (dead code).
    # HELPER: recursively compares two StructuredTensors: optionally the shape text, the set of
    # field names, the python type of each field value, and the values (recursing into nested
    # structured fields, else using assertAllEqual). Fails the test on any difference.
    if check_shape:
      self.assertEqual(repr(a.shape), repr(b.shape))
    self.assertEqual(set(a.field_names()), set(b.field_names()))
    for field in a.field_names():
      a_value = a.field_value(field)
      b_value = b.field_value(field)
      self.assertIs(type(a_value), type(b_value))
      if isinstance(a_value, structured_tensor.StructuredTensor):
        self._assertStructuredEqual(a_value, b_value, msg, check_shape)
      else:
        self.assertAllEqual(a_value, b_value, msg)

  @parameterized.named_parameters([
      dict(
          testcase_name="0D_0",
          st={"x": 1},
          axis=0,
          expected=[{"x": 1}]),
      dict(
          testcase_name="0D_minus_1",
          st={"x": 1},
          axis=-1,
          expected=[{"x": 1}]),
      dict(
          testcase_name="1D_0",
          st=[{"x": [1, 3]}, {"x": [2, 7, 9]}],
          axis=0,
          expected=[[{"x": [1, 3]}, {"x": [2, 7, 9]}]]),
      dict(
          testcase_name="1D_1",
          st=[{"x": [1]}, {"x": [2, 10]}],
          axis=1,
          expected=[[{"x": [1]}], [{"x": [2, 10]}]]),
      dict(
          testcase_name="2D_0",
          st=[[{"x": [1]}, {"x": [2]}], [{"x": [3, 4]}]],
          axis=0,
          expected=[[[{"x": [1]}, {"x": [2]}], [{"x": [3, 4]}]]]),
      dict(
          testcase_name="2D_1",
          st=[[{"x": 1}, {"x": 2}], [{"x": 3}]],
          axis=1,
          expected=[[[{"x": 1}, {"x": 2}]], [[{"x": 3}]]]),
      dict(
          testcase_name="2D_2",
          st=[[{"x": [1]}, {"x": [2]}], [{"x": [3, 4]}]],
          axis=2,
          expected=[[[{"x": [1]}], [{"x": [2]}]], [[{"x": [3, 4]}]]]),
      dict(
          testcase_name="3D_0",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=0,
          expected=[[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]],
                     [[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_minus_4",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=-4,  # same as zero
          expected=[[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]],
                     [[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_1",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=1,
          expected=[[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_minus_3",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=-3,  # same as 1
          expected=[[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_2",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=2,
          expected=[[[[{"x": [1]}, {"x": [2]}]], [[{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_minus_2",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=-2,  # same as 2
          expected=[[[[{"x": [1]}, {"x": [2]}]], [[{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_3",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=3,
          expected=[[[[{"x": [1]}], [{"x": [2]}]], [[{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
  ])  # pyformat: disable
  def testExpandDims(self, st, axis, expected):
    # [FUNCTION DOC]
    # Parameterized over structured tensors of rank 0 to 3, positive and negative axes. Builds a
    # StructuredTensor from python values, calls expand_dims with the axis and asserts the result
    # equals the expected nested value with a length-1 dimension inserted.
    st = StructuredTensor.from_pyval(st)
    result = array_ops.expand_dims(st, axis)
    self.assertAllEqual(result, expected)

  def testExpandDimsAxisTooBig(self):
    # [FUNCTION DOC]
    # Expanding a rank-3 structured tensor at axis 4 must raise a value error saying the axis is
    # out of bounds (valid range minus 4 up to below 4).
    st = [[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]]
    st = StructuredTensor.from_pyval(st)
    with self.assertRaisesRegex(ValueError,
                                "axis=4 out of bounds: expected -4<=axis<4"):
      array_ops.expand_dims(st, 4)

  def testExpandDimsAxisTooSmall(self):
    # [FUNCTION DOC]
    # Expanding a rank-3 structured tensor at axis minus 5 must raise a value error saying the
    # axis is out of bounds.
    st = [[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]]
    st = StructuredTensor.from_pyval(st)
    with self.assertRaisesRegex(ValueError,
                                "axis=-5 out of bounds: expected -4<=axis<4"):
      array_ops.expand_dims(st, -5)

  def testExpandDimsScalar(self):
    # [FUNCTION DOC]
    # Expands the last axis of a rank-3 ragged structured tensor with scalar fields and checks
    # the static shape text is two, unknown, unknown, one (differs from a value-built shape).
    # Note that if we expand_dims for the final dimension and there are scalar
    # fields, then the shape is (2, None, None, 1), whereas if it is constructed
    # from pyval it is (2, None, None, None).
    st = [[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]]
    st = StructuredTensor.from_pyval(st)
    result = array_ops.expand_dims(st, 3)
    expected_shape = tensor_shape.TensorShape([2, None, None, 1])
    self.assertEqual(repr(expected_shape), repr(result.shape))

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar_int32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int32,
          expected=1),
      dict(
          testcase_name="scalar_int64",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int64,
          expected=1),
      dict(
          testcase_name="list_0_int32",
          row_partitions=None,
          shape=(0),
          dtype=dtypes.int32,
          expected=0),
      dict(
          testcase_name="list_0_0_int32",
          row_partitions=None,
          shape=(0, 0),
          dtype=dtypes.int32,
          expected=0),
      dict(
          testcase_name="list_int32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int32,
          expected=7),
      dict(
          testcase_name="list_int64",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int64,
          expected=7),
      dict(
          testcase_name="matrix_int32",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.int32,
          expected=6),
      dict(
          testcase_name="tensor_int32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.int32,
          expected=6),
      dict(
          testcase_name="ragged_1_int32",
          row_partitions=[[0, 3, 4]],
          shape=(2, None),
          dtype=dtypes.int32,
          expected=4),
      dict(
          testcase_name="ragged_2_float32",
          row_partitions=[[0, 3, 4], [0, 2, 3, 5, 7]],
          shape=(2, None, None),
          dtype=dtypes.float32,
          expected=7),
  ])  # pyformat: disable
  def testSizeObject(self, row_partitions, shape, dtype, expected):
    # [FUNCTION DOC]
    # Builds a field-less StructuredTensor from a shape and optional row-splits partitions, then
    # checks both the v1 and v2 size ops with the given out_type equal the expected element
    # count (1 for scalars, outer length for vectors, innermost value count for higher ranks).
    if row_partitions is not None:
      row_partitions = [
          row_partition.RowPartition.from_row_splits(r) for r in row_partitions
      ]
    st = StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)
    # NOTE: size is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.size(st, out_type=dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.size_v2(st, out_type=dtype)
    self.assertAllEqual(actual2, expected)

  def test_shape_v2(self):
    # [FUNCTION DOC]
    # Builds a rank-2 structured tensor with a ragged field and checks the v2 shape op returns a
    # dynamic ragged shape whose static lengths are two and the row lengths one and two.
    rt = ragged_tensor.RaggedTensor.from_row_lengths(["a", "b", "c"], [1, 2])
    st = StructuredTensor.from_fields_and_rank({"r": rt}, rank=2)
    actual = array_ops.shape_v2(st, out_type=dtypes.int64)
    actual_static_lengths = actual.static_lengths()
    self.assertAllEqual([2, (1, 2)], actual_static_lengths)

  def test_shape(self):
    # [FUNCTION DOC]
    # Same scenario as the v2 shape test, checking both the v1 and v2 shape ops return static
    # lengths of two and the ragged row lengths one and two.
    rt = ragged_tensor.RaggedTensor.from_row_lengths(["a", "b", "c"], [1, 2])
    st = StructuredTensor.from_fields_and_rank({"r": rt}, rank=2)
    actual = array_ops.shape(st, out_type=dtypes.int64).static_lengths()
    actual_v2 = array_ops.shape_v2(st, out_type=dtypes.int64).static_lengths()
    expected = [2, (1, 2)]
    self.assertAllEqual(expected, actual)
    self.assertAllEqual(expected, actual_v2)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          dtype=dtypes.int32,
          expected=3),
      dict(
          testcase_name="list_empty_2",
          values=[{}, {}],
          dtype=dtypes.int32,
          expected=2),
      dict(
          testcase_name="list_empty_1",
          values=[{}],
          dtype=dtypes.int32,
          expected=1),
      dict(
          testcase_name="list_example_1",
          values=[{"x": [3]}, {"x": [4, 5]}],
          dtype=dtypes.int32,
          expected=2),
      dict(
          testcase_name="list_example_2",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=dtypes.float32,
          expected=3),
      dict(
          testcase_name="list_example_2_None",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=None,
          expected=3),
  ])  # pyformat: disable
  def testSizeAlt(self, values, dtype, expected):
    # [FUNCTION DOC]
    # Builds structured tensors from python values (empty and field-bearing, vector and ragged)
    # and checks v1 and v2 size give the expected count, including when out_type is not given.
    st = StructuredTensor.from_pyval(values)
    # NOTE: size is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.size(st, out_type=dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.size_v2(st, out_type=dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar_int32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int32,
          expected=0),
      dict(
          testcase_name="scalar_bool",
          row_partitions=None,
          shape=(),
          dtype=dtypes.bool,
          expected=False),
      dict(
          testcase_name="scalar_int64",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int64,
          expected=0),
      dict(
          testcase_name="scalar_float32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.float32,
          expected=0.0),
      dict(
          testcase_name="list_0_int32",
          row_partitions=None,
          shape=(0),
          dtype=dtypes.int32,
          expected=[]),
      dict(
          testcase_name="list_0_0_int32",
          row_partitions=None,
          shape=(0, 0),
          dtype=dtypes.int32,
          expected=[]),
      dict(
          testcase_name="list_int32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int32,
          expected=[0, 0, 0, 0, 0, 0, 0]),
      dict(
          testcase_name="list_int64",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int64,
          expected=[0, 0, 0, 0, 0, 0, 0]),
      dict(
          testcase_name="list_float32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.float32,
          expected=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
      dict(
          testcase_name="matrix_int32",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.int32,
          expected=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="matrix_float64",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.float64,
          expected=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
      dict(
          testcase_name="tensor_int32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.int32,
          expected=[[[0], [0], [0]], [[0], [0], [0]]]),
      dict(
          testcase_name="tensor_float32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.float32,
          expected=[[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]),
      dict(
          testcase_name="ragged_1_float32",
          row_partitions=[[0, 3, 4]],
          shape=(2, None),
          dtype=dtypes.float32,
          expected=[[0.0, 0.0, 0.0], [0.0]]),
      dict(
          testcase_name="ragged_2_float32",
          row_partitions=[[0, 3, 4], [0, 2, 3, 5, 7]],
          shape=(2, None, None),
          dtype=dtypes.float32,
          expected=[[[0.0, 0.0], [0.0], [0.0, 0.0]], [[0.0, 0.0]]]),
  ])  # pyformat: disable
  def testZerosLikeObject(self, row_partitions, shape, dtype, expected):
    # [FUNCTION DOC]
    # Builds a field-less structured tensor from shape and row partitions and checks the v1 and
    # v2 zeros_like ops with a dtype produce zeros of that dtype with the expected nesting.
    if row_partitions is not None:
      row_partitions = [
          row_partition.RowPartition.from_row_splits(r) for r in row_partitions
      ]
    st = StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)
    # NOTE: zeros_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.zeros_like(st, dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.zeros_like_v2(st, dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          dtype=dtypes.int32,
          expected=[[0, 0], [0]]),
      dict(
          testcase_name="list_empty_2",
          values=[{}, {}],
          dtype=dtypes.int32,
          expected=[0, 0]),
      dict(
          testcase_name="list_empty_1",
          values=[{}],
          dtype=dtypes.int32,
          expected=[0]),
      dict(
          testcase_name="list_example_1",
          values=[{"x": [3]}, {"x": [4, 5]}],
          dtype=dtypes.int32,
          expected=[0, 0]),
      dict(
          testcase_name="list_example_2",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=dtypes.float32,
          expected=[[0.0], [0.0, 0.0]]),
      dict(
          testcase_name="list_example_2_None",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=None,
          expected=[[0.0], [0.0, 0.0]]),
  ])  # pyformat: disable
  def testZerosLikeObjectAlt(self, values, dtype, expected):
    # [FUNCTION DOC]
    # Builds structured tensors from python values and checks v1 and v2 zeros_like give zeros in
    # the ragged shape of the value (default float when dtype is not given).
    st = StructuredTensor.from_pyval(values)
    # NOTE: zeros_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.zeros_like(st, dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.zeros_like_v2(st, dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar_int32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int32,
          expected=1),
      dict(
          testcase_name="scalar_bool",
          row_partitions=None,
          shape=(),
          dtype=dtypes.bool,
          expected=True),
      dict(
          testcase_name="scalar_int64",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int64,
          expected=1),
      dict(
          testcase_name="scalar_float32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.float32,
          expected=1.0),
      dict(
          testcase_name="list_0_int32",
          row_partitions=None,
          shape=(0),
          dtype=dtypes.int32,
          expected=[]),
      dict(
          testcase_name="list_0_0_int32",
          row_partitions=None,
          shape=(0, 0),
          dtype=dtypes.int32,
          expected=[]),
      dict(
          testcase_name="list_int32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int32,
          expected=[1, 1, 1, 1, 1, 1, 1]),
      dict(
          testcase_name="list_int64",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int64,
          expected=[1, 1, 1, 1, 1, 1, 1]),
      dict(
          testcase_name="list_float32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.float32,
          expected=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
      dict(
          testcase_name="matrix_int32",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.int32,
          expected=[[1, 1, 1], [1, 1, 1]]),
      dict(
          testcase_name="matrix_float64",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.float64,
          expected=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      dict(
          testcase_name="tensor_int32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.int32,
          expected=[[[1], [1], [1]], [[1], [1], [1]]]),
      dict(
          testcase_name="tensor_float32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.float32,
          expected=[[[1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0]]]),
      dict(
          testcase_name="ragged_1_float32",
          row_partitions=[[0, 3, 4]],
          shape=(2, None),
          dtype=dtypes.float32,
          expected=[[1.0, 1.0, 1.0], [1.0]]),
      dict(
          testcase_name="ragged_2_float32",
          row_partitions=[[0, 3, 4], [0, 2, 3, 5, 7]],
          shape=(2, None, None),
          dtype=dtypes.float32,
          expected=[[[1.0, 1.0], [1.0], [1.0, 1.0]], [[1.0, 1.0]]]),
  ])  # pyformat: disable
  def testOnesLikeObject(self, row_partitions, shape, dtype, expected):
    # [FUNCTION DOC]
    # Builds a field-less structured tensor from shape and row partitions and checks the v1 and
    # v2 ones_like ops with a dtype produce ones of that dtype with the expected nesting.
    if row_partitions is not None:
      row_partitions = [
          row_partition.RowPartition.from_row_splits(r) for r in row_partitions
      ]
    st = StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)
    # NOTE: ones_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.ones_like(st, dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.ones_like_v2(st, dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          dtype=dtypes.int32,
          expected=[[1, 1], [1]]),
      dict(
          testcase_name="list_empty_2",
          values=[{}, {}],
          dtype=dtypes.int32,
          expected=[1, 1]),
      dict(
          testcase_name="list_empty_1",
          values=[{}],
          dtype=dtypes.int32,
          expected=[1]),
      dict(
          testcase_name="list_example_1",
          values=[{"x": [3]}, {"x": [4, 5]}],
          dtype=dtypes.int32,
          expected=[1, 1]),
      dict(
          testcase_name="list_example_2",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=dtypes.float32,
          expected=[[1.0], [1.0, 1.0]]),
      dict(
          testcase_name="list_example_2_None",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=None,
          expected=[[1.0], [1.0, 1.0]]),
  ])  # pyformat: disable
  def testOnesLikeObjectAlt(self, values, dtype, expected):
    # [FUNCTION DOC]
    # Builds structured tensors from python values and checks v1 and v2 ones_like give ones in
    # the ragged shape of the value (default float when dtype is not given).
    st = StructuredTensor.from_pyval(values)
    # NOTE: ones_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.ones_like(st, dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.ones_like_v2(st, dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar",
          row_partitions=None,
          shape=(),
          expected=0),
      dict(
          testcase_name="list_0",
          row_partitions=None,
          shape=(0,),
          expected=1),
      dict(
          testcase_name="list_0_0",
          row_partitions=None,
          shape=(0, 0),
          expected=2),
      dict(
          testcase_name="list_7",
          row_partitions=None,
          shape=(7,),
          expected=1),
      dict(
          testcase_name="matrix",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          expected=2),
      dict(
          testcase_name="tensor",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          expected=3),
      dict(
          testcase_name="ragged_1",
          row_partitions=[[0, 3, 4]],
          shape=(2, None),
          expected=2),
      dict(
          testcase_name="ragged_2",
          row_partitions=[[0, 3, 4], [0, 2, 3, 5, 7]],
          shape=(2, None, None),
          expected=3),
  ])  # pyformat: disable
  def testRank(self, row_partitions, shape, expected):
    # [FUNCTION DOC]
    # Builds a field-less structured tensor from shape and row partitions and checks the
    # structured rank op returns the expected rank (0 for scalar up to 3).
    if row_partitions is not None:
      row_partitions = [
          row_partition.RowPartition.from_row_splits(r) for r in row_partitions
      ]
    st = StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)

    # NOTE: rank is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = structured_array_ops.rank(st)
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          expected=2),
      dict(
          testcase_name="list_empty_2",
          values=[{}, {}],
          expected=1),
      dict(
          testcase_name="list_empty_1",
          values=[{}],
          expected=1),
      dict(
          testcase_name="list_example_1",
          values=[{"x": [3]}, {"x": [4, 5]}],
          expected=1),
      dict(
          testcase_name="list_example_2",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          expected=2),
  ])  # pyformat: disable
  def testRankAlt(self, values, expected):
    # [FUNCTION DOC]
    # Builds structured tensors from python values and checks the public rank op returns the
    # expected rank.
    st = StructuredTensor.from_pyval(values)
    # NOTE: rank is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.rank(st)
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty",
          values=[[{}], [{}]],
          axis=0,
          expected=[{}, {}]),
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          axis=0,
          expected=[{}, {}, {}]),
      dict(
          testcase_name="list_with_fields",
          values=[[{"a": 4, "b": [3, 4]}], [{"a": 5, "b": [5, 6]}]],
          axis=0,
          expected=[{"a": 4, "b": [3, 4]}, {"a": 5, "b": [5, 6]}]),
      dict(
          testcase_name="list_with_submessages",
          values=[[{"a": {"foo": 3}, "b": [3, 4]}],
                  [{"a": {"foo": 4}, "b": [5, 6]}]],
          axis=0,
          expected=[{"a": {"foo": 3}, "b": [3, 4]},
                    {"a": {"foo": 4}, "b": [5, 6]}]),
      dict(
          testcase_name="list_with_empty_submessages",
          values=[[{"a": {}, "b": [3, 4]}],
                  [{"a": {}, "b": [5, 6]}]],
          axis=0,
          expected=[{"a": {}, "b": [3, 4]},
                    {"a": {}, "b": [5, 6]}]),
      dict(
          testcase_name="lists_of_lists",
          values=[[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                   [{"a": {}, "b": [7, 8, 9]}]],
                  [[{"a": {}, "b": [10]}]]],
          axis=0,
          expected=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                    [{"a": {}, "b": [7, 8, 9]}],
                    [{"a": {}, "b": [10]}]]),
      dict(
          testcase_name="lists_of_lists_axis_1",
          values=[[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                   [{"a": {}, "b": [7, 8, 9]}]],
                  [[{"a": {}, "b": []}], [{"a": {}, "b": [3]}]]],
          axis=1,
          expected=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]},
                     {"a": {}, "b": []}],
                    [{"a": {}, "b": [7, 8, 9]}, {"a": {}, "b": [3]}]]),
      dict(
          testcase_name="lists_of_lists_axis_minus_2",
          values=[[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                   [{"a": {}, "b": [7, 8, 9]}]],
                  [[{"a": {}, "b": [10]}]]],
          axis=-2,  # Same as axis=0.
          expected=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                    [{"a": {}, "b": [7, 8, 9]}],
                    [{"a": {}, "b": [10]}]]),
      dict(
          testcase_name="from_structured_tensor_util_test",
          values=[[{"x0": 0, "y": {"z": [[3, 13]]}},
                   {"x0": 1, "y": {"z": [[3], [4, 13]]}},
                   {"x0": 2, "y": {"z": [[3, 5], [4]]}}],
                  [{"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                   {"x0": 4, "y": {"z": [[3], [4]]}}]],
          axis=0,
          expected=[{"x0": 0, "y": {"z": [[3, 13]]}},
                    {"x0": 1, "y": {"z": [[3], [4, 13]]}},
                    {"x0": 2, "y": {"z": [[3, 5], [4]]}},
                    {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                    {"x0": 4, "y": {"z": [[3], [4]]}}]),
  ])  # pyformat: disable
  def testConcat(self, values, axis, expected):
    # [FUNCTION DOC]
    # Converts each python value to a structured tensor, concatenates the list along the given
    # axis (including a negative axis and nested fields) and asserts the expected combined value.
    values = [StructuredTensor.from_pyval(v) for v in values]
    actual = array_ops.concat(values, axis)
    self.assertAllEqual(actual, expected)

  def testConcatTuple(self):
    # [FUNCTION DOC]
    # Concatenates a tuple (not a list) of two structured tensors on axis zero and asserts the
    # result holds both rows.
    values = (StructuredTensor.from_pyval([{
        "a": 3
    }]), StructuredTensor.from_pyval([{
        "a": 4
    }]))
    actual = array_ops.concat(values, axis=0)
    self.assertAllEqual(actual, [{"a": 3}, {"a": 4}])

  @parameterized.named_parameters([
      dict(
          testcase_name="field_dropped",
          values=[[{"a": [2]}], [{}]],
          axis=0,
          error_type=ValueError,
          error_regex="a"),
      dict(
          testcase_name="field_added",
          values=[[{"b": [3]}], [{"b": [3], "a": [7]}]],
          axis=0,
          error_type=ValueError,
          error_regex="b"),
      dict(testcase_name="rank_submessage_change",
           values=[[{"a": [{"b": [[3]]}]}],
                   [{"a": [[{"b": [3]}]]}]],
           axis=0,
           error_type=ValueError,
           error_regex="Ranks of sub-message do not match",
          ),
      dict(testcase_name="rank_message_change",
           values=[[{"a": [3]}],
                   [[{"a": 3}]]],
           axis=0,
           error_type=ValueError,
           error_regex="Ranks of sub-message do not match",
          ),
      dict(testcase_name="concat_scalar",
           values=[{"a": [3]}, {"a": [4]}],
           axis=0,
           error_type=ValueError,
           error_regex="axis=0 out of bounds",
          ),
      dict(testcase_name="concat_axis_large",
           values=[[{"a": [3]}], [{"a": [4]}]],
           axis=1,
           error_type=ValueError,
           error_regex="axis=1 out of bounds",
          ),
      dict(testcase_name="concat_axis_large_neg",
           values=[[{"a": [3]}], [{"a": [4]}]],
           axis=-2,
           error_type=ValueError,
           error_regex="axis=-2 out of bounds",
          ),
      dict(testcase_name="concat_deep_rank_wrong",
           values=[[{"a": [3]}], [{"a": [[4]]}]],
           axis=0,
           error_type=ValueError,
           error_regex="must have rank",
          ),
  ])  # pyformat: disable
  def testConcatError(self, values, axis, error_type, error_regex):
    # [FUNCTION DOC]
    # Concatenating incompatible structured tensors (dropped or added fields, rank changes,
    # scalar input, out-of-range axis) must raise the given error type with a message matching
    # the given pattern.
    values = [StructuredTensor.from_pyval(v) for v in values]
    with self.assertRaisesRegex(error_type, error_regex):
      array_ops.concat(values, axis)

  def testConcatWithRagged(self):
    # [FUNCTION DOC]
    # Concatenating a list that mixes a structured tensor with a plain dense tensor must raise a
    # value error saying values must be a list of structured tensors.
    values = [StructuredTensor.from_pyval({}), array_ops.constant(3)]
    with self.assertRaisesRegex(ValueError,
                                "values must be a list of StructuredTensors"):
      array_ops.concat(values, 0)

  def testConcatNotAList(self):
    # [FUNCTION DOC]
    # Passing a single structured tensor (not a sequence) to the structured concat must raise a
    # value error saying values must be a list of structured tensors.
    values = StructuredTensor.from_pyval({})
    with self.assertRaisesRegex(ValueError,
                                "values must be a list of StructuredTensors"):
      structured_array_ops.concat(values, 0)

  def testConcatEmptyList(self):
    # [FUNCTION DOC]
    # Concatenating an empty list must raise a value error saying values must not be empty.
    with self.assertRaisesRegex(ValueError, "values must not be an empty list"):
      structured_array_ops.concat([], 0)

  def testExtendOpErrorNotList(self):
    # [FUNCTION DOC]
    # Calling the private extend-op helper with a lone structured tensor instead of a list must
    # raise a value error saying a list was expected.
    # Should be a list.
    values = StructuredTensor.from_pyval({})

    def leaf_op(values):
      # [FUNCTION DOC]
      # HELPER: trivial leaf op returning the first element of values.
      return values[0]

    with self.assertRaisesRegex(ValueError, "Expected a list"):
      structured_array_ops._extend_op(values, leaf_op)

  def testExtendOpErrorEmptyList(self):
    # [FUNCTION DOC]
    # Calling the private extend-op helper with an empty list must raise a value error saying the
    # list cannot be empty.

    def leaf_op(values):
      # [FUNCTION DOC]
      # HELPER: trivial leaf op returning the first element of values.
      return values[0]

    with self.assertRaisesRegex(ValueError, "List cannot be empty"):
      structured_array_ops._extend_op([], leaf_op)

  def testRandomShuffle2021(self):
    # [FUNCTION DOC]
    # Sets the global seed and shuffles a five-row nested structured tensor through the public
    # random_shuffle with op seed 2021; asserts the result equals one specific expected row order.
    # Depends on the seed-driven random generator behaviour.
    original = StructuredTensor.from_pyval([
        {"x0": 0, "y": {"z": [[3, 13]]}},
        {"x0": 1, "y": {"z": [[3], [4, 13]]}},
        {"x0": 2, "y": {"z": [[3, 5], [4]]}},
        {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
        {"x0": 4, "y": {"z": [[3], [4]]}}])  # pyformat: disable
    random_seed.set_seed(1066)
    result = random_ops.random_shuffle(original, seed=2021)
    expected = StructuredTensor.from_pyval([
        {"x0": 0, "y": {"z": [[3, 13]]}},
        {"x0": 1, "y": {"z": [[3], [4, 13]]}},
        {"x0": 4, "y": {"z": [[3], [4]]}},
        {"x0": 2, "y": {"z": [[3, 5], [4]]}},
        {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},])  # pyformat: disable
    self.assertAllEqual(result, expected)

  def testRandomShuffle2022Eager(self):
    # [FUNCTION DOC]
    # Sets the global seed and shuffles the same five-row structured tensor by calling the
    # structured random_shuffle directly with op seed 2022; asserts one specific expected order.
    # The name says Eager but the test runs in both modes.
    original = StructuredTensor.from_pyval([
        {"x0": 0, "y": {"z": [[3, 13]]}},
        {"x0": 1, "y": {"z": [[3], [4, 13]]}},
        {"x0": 2, "y": {"z": [[3, 5], [4]]}},
        {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
        {"x0": 4, "y": {"z": [[3], [4]]}}])  # pyformat: disable
    expected = StructuredTensor.from_pyval([
        {"x0": 1, "y": {"z": [[3], [4, 13]]}},
        {"x0": 0, "y": {"z": [[3, 13]]}},
        {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
        {"x0": 4, "y": {"z": [[3], [4]]}},
        {"x0": 2, "y": {"z": [[3, 5], [4]]}}])  # pyformat: disable
    random_seed.set_seed(1066)
    result = structured_array_ops.random_shuffle(original, seed=2022)
    self.assertAllEqual(result, expected)

  def testRandomShuffleScalarError(self):
    # [FUNCTION DOC]
    # Shuffling a scalar structured tensor must raise a value error mentioning scalar.
    original = StructuredTensor.from_pyval(
        {"x0": 2, "y": {"z": [[3, 5], [4]]}})  # pyformat: disable

    with self.assertRaisesRegex(ValueError, "scalar"):
      random_ops.random_shuffle(original)

  def testStructuredTensorArrayLikeNoRank(self):
    # [FUNCTION DOC]
    # Inside a traced function, builds a zeros tensor of unknown rank and asks the private
    # structured-tensor-like helper for a matching structure; expects a value error saying a
    # structured tensor cannot be built with unknown rank.
    """Test when the rank is unknown."""

    @def_function.function
    def my_fun(foo):
      # [FUNCTION DOC]
      # HELPER: traced function creating a zeros tensor of unknown rank and calling the
      # structured-tensor-like helper on it (raises for unknown rank); returns nothing.
      bar_shape = math_ops.range(foo)
      bar = array_ops.zeros(shape=bar_shape)
      structured_array_ops._structured_tensor_like(bar)

    with self.assertRaisesRegex(ValueError,
                                "Can't build StructuredTensor w/ unknown rank"):
      my_fun(array_ops.constant(3))

  def testStructuredTensorArrayRankOneKnownShape(self):
    # [FUNCTION DOC]
    # A dense zeros vector of known length four yields, via the private helper, four empty
    # structured rows.
    """Fully test structured_tensor_array_like."""
    foo = array_ops.zeros(shape=[4])
    result = structured_array_ops._structured_tensor_like(foo)
    self.assertAllEqual([{}, {}, {}, {}], result)

  # Note that we have to be careful about whether the indices are int32
  # or int64.
  def testStructuredTensorArrayRankOneUnknownShape(self):
    # [FUNCTION DOC]
    # In a traced function a zeros vector with a dynamic length of four goes through the private
    # helper; the result must equal a structured tensor built from a dynamic ragged shape of
    # length four.
    """Fully test structured_tensor_array_like."""

    @def_function.function
    def my_fun(my_shape):
      # [FUNCTION DOC]
      # HELPER: traced function making zeros of the given dynamic shape and returning the
      # structured-tensor-like result for it.
      my_zeros = array_ops.zeros(my_shape)
      return structured_array_ops._structured_tensor_like(my_zeros)

    result = my_fun(array_ops.constant(4))
    shape = DynamicRaggedShape._from_inner_shape([4], dtype=dtypes.int32)
    expected = StructuredTensor.from_shape(shape)
    self.assertAllEqual(expected, result)

  def testStructuredTensorArrayRankTwoUnknownShape(self):
    # [FUNCTION DOC]
    # In a traced function a zeros matrix with dynamic shape two by two goes through the private
    # helper; the result must be a two by two grid of empty structured values.
    """Fully test structured_tensor_array_like."""

    @def_function.function
    def my_fun(my_shape):
      # [FUNCTION DOC]
      # HELPER: traced function making zeros of the given dynamic shape and returning the
      # structured-tensor-like result for it.
      my_zeros = array_ops.zeros(my_shape)
      return structured_array_ops._structured_tensor_like(my_zeros)

    result = my_fun(array_ops.constant([2, 2]))
    self.assertAllEqual([[{}, {}], [{}, {}]], result)

  def testStructuredTensorArrayRankZero(self):
    # [FUNCTION DOC]
    # A scalar dense tensor yields, via the private helper, a single empty structured value.
    """Fully test structured_tensor_array_like."""
    foo = array_ops.zeros(shape=[])
    result = structured_array_ops._structured_tensor_like(foo)
    self.assertAllEqual({}, result)

  def testStructuredTensorLikeStructuredTensor(self):
    # [FUNCTION DOC]
    # Passing a structured tensor with fields to the private helper yields a field-less
    # structured tensor of the same shape (two empty rows).
    """Fully test structured_tensor_array_like."""
    foo = structured_tensor.StructuredTensor.from_pyval([{"a": 3}, {"a": 7}])
    result = structured_array_ops._structured_tensor_like(foo)
    self.assertAllEqual([{}, {}], result)

  def testStructuredTensorArrayLike(self):
    # [FUNCTION DOC]
    # Regression test for the ragged-input branch of the private helper: a ragged tensor built
    # from a dense matrix of zeros with row splits zero, three, five yields a rank-3 structure.
    """There was a bug in a case in a private function.

    This was difficult to reach externally, so I wrote a test
    to check it directly.
    """
    rt = ragged_tensor.RaggedTensor.from_row_splits(
        array_ops.zeros(shape=[5, 3]), [0, 3, 5])
    result = structured_array_ops._structured_tensor_like(rt)
    self.assertEqual(3, result.rank)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty",
          params=[{}, {}, {}],
          indices=[0, 2],
          axis=0,
          batch_dims=0,
          expected=[{}, {}]),
      dict(
          testcase_name="list_of_lists_empty",
          params=[[{}, {}], [{}], [{}, {}, {}]],
          indices=[2, 0],
          axis=0,
          batch_dims=0,
          expected=[[{}, {}, {}], [{}, {}]]),
      dict(
          testcase_name="list_with_fields",
          params=[{"a": 4, "b": [3, 4]}, {"a": 5, "b": [5, 6]},
                  {"a": 7, "b": [9, 10]}],
          indices=[2, 0, 0],
          axis=0,
          batch_dims=0,
          expected=[{"a": 7, "b": [9, 10]}, {"a": 4, "b": [3, 4]},
                    {"a": 4, "b": [3, 4]}]),
      dict(
          testcase_name="list_with_submessages",
          params=[{"a": {"foo": 3}, "b": [3, 4]},
                  {"a": {"foo": 4}, "b": [5, 6]},
                  {"a": {"foo": 7}, "b": [9, 10]}],
          indices=[2, 0],
          axis=0,
          batch_dims=0,
          expected=[{"a": {"foo": 7}, "b": [9, 10]},
                    {"a": {"foo": 3}, "b": [3, 4]}]),
      dict(
          testcase_name="list_with_empty_submessages",
          params=[{"a": {}, "b": [3, 4]},
                  {"a": {}, "b": [5, 6]},
                  {"a": {}, "b": [9, 10]}],
          indices=[2, 0],
          axis=0,
          batch_dims=0,
          expected=[{"a": {}, "b": [9, 10]},
                    {"a": {}, "b": [3, 4]}]),
      dict(
          testcase_name="lists_of_lists",
          params=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                  [{"a": {}, "b": [7, 8, 9]}],
                  [{"a": {}, "b": []}]],
          indices=[2, 0, 0],
          axis=0,
          batch_dims=0,
          expected=[[{"a": {}, "b": []}],
                    [{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                    [{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}]]),
      dict(
          testcase_name="lists_of_lists_axis_1",
          params=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                  [{"a": {}, "b": [7, 8, 9]}, {"a": {}, "b": [2, 8, 2]}],
                  [{"a": {}, "b": []}, {"a": {}, "b": [4]}]],
          indices=[1, 0],
          axis=1,
          batch_dims=0,
          expected=[[{"a": {}, "b": [5]}, {"a": {}, "b": [3, 4]}],
                    [{"a": {}, "b": [2, 8, 2]}, {"a": {}, "b": [7, 8, 9]}],
                    [{"a": {}, "b": [4]}, {"a": {}, "b": []}]]),
      dict(
          testcase_name="lists_of_lists_axis_minus_2",
          params=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                  [{"a": {}, "b": [7, 8, 9]}],
                  [{"a": {}, "b": []}]],
          indices=[2, 0, 0],
          axis=-2,  # same as 0
          batch_dims=0,
          expected=[[{"a": {}, "b": []}],
                    [{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                    [{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}]]),
      dict(
          testcase_name="lists_of_lists_axis_minus_1",
          params=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                  [{"a": {}, "b": [7, 8, 9]}, {"a": {}, "b": [2, 8, 2]}],
                  [{"a": {}, "b": []}, {"a": {}, "b": [4]}]],
          indices=[1, 0],
          axis=-1,  # same as 1
          batch_dims=0,
          expected=[[{"a": {}, "b": [5]}, {"a": {}, "b": [3, 4]}],
                    [{"a": {}, "b": [2, 8, 2]}, {"a": {}, "b": [7, 8, 9]}],
                    [{"a": {}, "b": [4]}, {"a": {}, "b": []}]]),
      dict(
          testcase_name="from_structured_tensor_util_test",
          params=[{"x0": 0, "y": {"z": [[3, 13]]}},
                  {"x0": 1, "y": {"z": [[3], [4, 13]]}},
                  {"x0": 2, "y": {"z": [[3, 5], [4]]}},
                  {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                  {"x0": 4, "y": {"z": [[3], [4]]}}],
          indices=[1, 0, 4, 3, 2],
          axis=0,
          batch_dims=0,
          expected=[{"x0": 1, "y": {"z": [[3], [4, 13]]}},
                    {"x0": 0, "y": {"z": [[3, 13]]}},
                    {"x0": 4, "y": {"z": [[3], [4]]}},
                    {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                    {"x0": 2, "y": {"z": [[3, 5], [4]]}}]),
      dict(
          testcase_name="scalar_index_axis_0",
          params=[{"x0": 0, "y": {"z": [[3, 13]]}},
                  {"x0": 1, "y": {"z": [[3], [4, 13]]}},
                  {"x0": 2, "y": {"z": [[3, 5], [4]]}},
                  {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                  {"x0": 4, "y": {"z": [[3], [4]]}}],
          indices=3,
          axis=0,
          batch_dims=0,
          expected={"x0": 3, "y": {"z": [[3, 7, 1], [4]]}}),
      dict(
          testcase_name="params_2D_vector_index_axis_1_batch_dims_1",
          params=[[{"x0": 0, "y": {"z": [[3, 13]]}},
                   {"x0": 1, "y": {"z": [[3], [4, 13]]}}],
                  [{"x0": 2, "y": {"z": [[3, 5], [4]]}},
                   {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                   {"x0": 4, "y": {"z": [[3], [4]]}}]],
          indices=[1, 0],
          axis=1,
          batch_dims=1,
          expected=[{"x0": 1, "y": {"z": [[3], [4, 13]]}},
                    {"x0": 2, "y": {"z": [[3, 5], [4]]}}]),
  ])  # pyformat: disable
  def testGather(self, params, indices, axis, batch_dims, expected):
    # [FUNCTION DOC]
    # Gathers rows from a structured tensor with dense indices (vector or scalar, with axis and
    # batch_dims variations, including negative axis) via the public gather and asserts the
    # expected structured value. validate_indices is passed but not meaningfully tested.
    params = StructuredTensor.from_pyval(params)
    # validate_indices isn't actually used, and we aren't testing names
    actual = array_ops.gather(
        params,
        indices,
        validate_indices=True,
        axis=axis,
        name=None,
        batch_dims=batch_dims)
    self.assertAllEqual(actual, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="params_2D_index_2D_axis_1_batch_dims_1",
          params=[[{"x0": 0, "y": {"z": [[3, 13]]}},
                   {"x0": 1, "y": {"z": [[3], [4, 13]]}}],
                  [{"x0": 2, "y": {"z": [[3, 5], [4]]}},
                   {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                   {"x0": 4, "y": {"z": [[3], [4]]}}]],
          indices=[[1, 0], [0, 2]],
          axis=1,
          batch_dims=1,
          expected=[[{"x0": 1, "y": {"z": [[3], [4, 13]]}},
                     {"x0": 0, "y": {"z": [[3, 13]]}}],
                    [{"x0": 2, "y": {"z": [[3, 5], [4]]}},
                     {"x0": 4, "y": {"z": [[3], [4]]}}]]),
      dict(
          testcase_name="params_1D_index_2D_axis_0_batch_dims_0",
          params=[{"x0": 0, "y": {"z": [[3, 13]]}}],
          indices=[[0], [0, 0]],
          axis=0,
          batch_dims=0,
          expected=[[{"x0": 0, "y": {"z": [[3, 13]]}}],
                    [{"x0": 0, "y": {"z": [[3, 13]]}},
                     {"x0": 0, "y": {"z": [[3, 13]]}}]]),
  ])  # pyformat: disable
  def testGatherRagged(self, params, indices, axis, batch_dims, expected):
    # [FUNCTION DOC]
    # Same as the dense gather test but the indices are converted to a ragged tensor first;
    # asserts the gathered structured result equals the expected nested value.
    params = StructuredTensor.from_pyval(params)
    # Shouldn't need to do this, but see cl/366396997
    indices = ragged_factory_ops.constant(indices)
    # validate_indices isn't actually used, and we aren't testing names
    actual = array_ops.gather(
        params,
        indices,
        validate_indices=True,
        axis=axis,
        name=None,
        batch_dims=batch_dims)
    self.assertAllEqual(actual, expected)

  @parameterized.named_parameters([
      dict(testcase_name="params_scalar",
           params={"a": [3]},
           indices=0,
           axis=0,
           batch_dims=0,
           error_type=ValueError,
           error_regex="axis=0 out of bounds",
          ),
      dict(testcase_name="axis_large",
           params=[{"a": [3]}],
           indices=0,
           axis=1,
           batch_dims=0,
           error_type=ValueError,
           error_regex="axis=1 out of bounds",
          ),
      dict(testcase_name="axis_large_neg",
           params=[{"a": [3]}],
           indices=0,
           axis=-2,
           batch_dims=0,
           error_type=ValueError,
           error_regex="axis=-2 out of bounds",
          ),
      dict(testcase_name="batch_large",
           params=[[{"a": [3]}]],
           indices=0,
           axis=0,
           batch_dims=1,
           error_type=ValueError,
           error_regex="batch_dims=1 out of bounds",
          ),
  ])  # pyformat: disable
  def testGatherError(self, params, indices, axis, batch_dims, error_type,
                      error_regex):
    # [FUNCTION DOC]
    # Calls the structured gather with an invalid axis or batch_dims (scalar params, axis too
    # large or too negative, batch_dims too large) and expects the given error type with a
    # message matching the pattern.
    params = StructuredTensor.from_pyval(params)
    with self.assertRaisesRegex(error_type, error_regex):
      structured_array_ops.gather(
          params,
          indices,
          validate_indices=True,
          axis=axis,
          name=None,
          batch_dims=batch_dims)


if __name__ == "__main__":
  googletest.main()
