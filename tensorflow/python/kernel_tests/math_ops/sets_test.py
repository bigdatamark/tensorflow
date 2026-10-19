# [MODULE DOC]
# Tests for the set operations module (set size, intersection, difference, union) on dense and
# sparse tensors of several integer dtypes and strings, in two and three dimensions, including
# duplicate handling, empty sets, invalid (out-of-order) indices, static shape checks, sorted
# union output and a raw-op shape error. Defines module-level helpers that convert lists to
# arrays, constants and sparse tensors, plus the set of dtypes under test. Not a helper module;
# no import-time side effects beyond constants.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for set_ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_set_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest

_DTYPES = set([
    dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8,
    dtypes.uint16, dtypes.string
])


def _values(values, dtype):
  # [FUNCTION DOC]
  # HELPER: small test utility. Converts a (nested) list of values into a NumPy array of the
  # element type matching the given TensorFlow dtype (a unicode string type for the string dtype).
  # Returns the array; no side effects.
  return np.array(
      values,
      dtype=(np.str_ if (dtype == dtypes.string) else dtype.as_numpy_dtype))


def _constant(values, dtype):
  # [FUNCTION DOC]
  # HELPER: small test utility. Builds a constant tensor of the given dtype from a (nested) list
  # of values, converting through the array helper first. Returns the tensor.
  return constant_op.constant(_values(values, dtype), dtype=dtype)


def _dense_to_sparse(dense, dtype):
  # [FUNCTION DOC]
  # HELPER: small test utility. Converts a ragged list of rows (lists of values) into a
  # two-dimensional sparse tensor of the given dtype: each cell becomes an entry at its row and
  # column position, and the dense shape is the row count by the longest row length. For the
  # string dtype cells are converted to text first. Returns the sparse tensor.
  indices = []
  values = []
  max_row_len = 0
  for row in dense:
    max_row_len = max(max_row_len, len(row))
  shape = [len(dense), max_row_len]
  row_ix = 0
  for row in dense:
    col_ix = 0
    for cell in row:
      indices.append([row_ix, col_ix])
      values.append(str(cell) if dtype == dtypes.string else cell)
      col_ix += 1
    row_ix += 1
  return sparse_tensor_lib.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(values, dtype),
      constant_op.constant(shape, dtypes.int64))


class SetOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the set operations module (size, intersection, difference, union) over dense
  # and sparse inputs of several integer dtypes and strings, in two and three dimensions. Public
  # tests loop over dtypes and call per-dtype private helpers; shared helpers build the ops,
  # check static shapes, and compare results. Combines the TensorFlow test base case with the
  # parameterized test mixin.

  @test_util.run_deprecated_v1
  def test_set_size_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the two-dimensional set-size check for every supported dtype.
    for dtype in _DTYPES:
      self._test_set_size_2d(dtype)

  def _test_set_size_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: for test_set_size_2d. For the given dtype, checks the per-row set sizes of small
    # ragged sparse inputs (including empty rows) via the size helper. Returns nothing.
    self.assertAllEqual([1], self._set_size(_dense_to_sparse([[1]], dtype)))
    self.assertAllEqual([2, 1],
                        self._set_size(_dense_to_sparse([[1, 9], [1]], dtype)))
    self.assertAllEqual(
        [3, 0], self._set_size(_dense_to_sparse([[1, 9, 2], []], dtype)))
    self.assertAllEqual(
        [0, 3], self._set_size(_dense_to_sparse([[], [1, 9, 2]], dtype)))

  def test_invalid_size(self):
    # [FUNCTION DOC]
    # Test: a sparse tensor whose indices lie far outside its declared dense shape is passed to
    # set size with index validation disabled; expects a value, invalid-argument or internal
    # error mentioning an index out of range. Returns nothing.
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError,
                                 errors_impl.InternalError),
                                "Index out of range|expected <"):
      sparse_data = sparse_tensor_lib.SparseTensor(
          constant_op.constant([[804, 7450], [48245, 2577]], dtypes.int64),
          constant_op.constant([1, 1], dtypes.int64),
          constant_op.constant([812, 390], dtypes.int64))
      self.evaluate(sets.set_size(sparse_data, validate_indices=False))

  @test_util.run_deprecated_v1
  def test_set_size_duplicates_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the duplicate-handling two-dimensional set-size check for every
    # supported dtype.
    for dtype in _DTYPES:
      self._test_set_size_duplicates_2d(dtype)

  def _test_set_size_duplicates_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: for test_set_size_duplicates_2d. Verifies that repeated values in a row count only
    # once (a row of all ones has size one; a longer row with repeats gives its distinct count),
    # plus empty and negative-valued rows. Returns nothing.
    self.assertAllEqual(
        [1], self._set_size(_dense_to_sparse([[1, 1, 1, 1, 1, 1]], dtype)))
    self.assertAllEqual([2, 7, 3, 0, 1],
                        self._set_size(
                            _dense_to_sparse([[1, 9], [
                                6, 7, 8, 8, 6, 7, 5, 3, 3, 0, 6, 6, 9, 0, 0, 0
                            ], [999, 1, -1000], [], [-1]], dtype)))

  @test_util.run_deprecated_v1
  def test_set_size_3d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the three-dimensional set-size check for every supported dtype.
    for dtype in _DTYPES:
      self._test_set_size_3d(dtype)

  def test_set_size_3d_invalid_indices(self):
    # [FUNCTION DOC]
    # Test: runs the three-dimensional set-size check with out-of-order indices for every
    # supported dtype; each is expected to fail with an out-of-order op error.
    for dtype in _DTYPES:
      self._test_set_size_3d(dtype, invalid_indices=True)

  def _test_set_size_3d(self, dtype, invalid_indices=False):
    # [FUNCTION DOC]
    # HELPER: builds a three-dimensional sparse tensor (shape three by two by three) of the dtype
    # with some empty sets. If the flag (default false) is set, the indices are deliberately out
    # of order and the size computation must raise an op error saying so; otherwise the per-set
    # sizes must equal the expected three by two table. Returns nothing.
    if invalid_indices:
      indices = constant_op.constant([
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
          [0, 0, 0], [0, 0, 2],             # 0,0
                                            # 2,0
          [2, 1, 1]                         # 2,1
      ], dtypes.int64)
    else:
      indices = constant_op.constant([
          [0, 0, 0], [0, 0, 2],             # 0,0
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
                                            # 2,0
          [2, 1, 1]                         # 2,1
      ], dtypes.int64)

    sp = sparse_tensor_lib.SparseTensor(
        indices,
        _constant([
            1, 9,     # 0,0
            3, 3,     # 0,1
            1,        # 1,0
            9, 7, 8,  # 1,1
                      # 2,0
            5         # 2,1
        ], dtype),
        constant_op.constant([3, 2, 3], dtypes.int64))

    if invalid_indices:
      with self.assertRaisesRegex(errors_impl.OpError, "out of order"):
        self._set_size(sp)
    else:
      self.assertAllEqual([
          [2,   # 0,0
           1],  # 0,1
          [1,   # 1,0
           3],  # 1,1
          [0,   # 2,0
           1]   # 2,1
      ], self._set_size(sp))

  def _set_size(self, sparse_data):
    # [FUNCTION DOC]
    # HELPER: computes set sizes for a sparse tensor with index validation both on and off,
    # asserting each result has unknown static shape and 32-bit integer dtype, evaluates both,
    # asserts the results are identical, and returns the first.
    # KNOWN ISSUE: opens a cached session bound to an unused name; the evaluation itself does
    # not use it.
    # Validate that we get the same results with or without `validate_indices`.
    ops = [
        sets.set_size(sparse_data, validate_indices=True),
        sets.set_size(sparse_data, validate_indices=False)
    ]
    for op in ops:
      self.assertEqual(None, op.get_shape().dims)
      self.assertEqual(dtypes.int32, op.dtype)
    with self.cached_session() as sess:
      results = self.evaluate(ops)
    self.assertAllEqual(results[0], results[1])
    return results[0]

  @test_util.run_deprecated_v1
  def test_set_intersection_multirow_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the multirow two-dimensional intersection check for every dtype.
    for dtype in _DTYPES:
      self._test_set_intersection_multirow_2d(dtype)

  def _test_set_intersection_multirow_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: intersects two rows of values per side (the first row shares two values, the second
    # none) for the given dtype, once with dense first operand and sparse second operand, once
    # with both sparse. Asserts result indices, values, shape and per-row counts. Returns nothing.
    a_values = [[9, 1, 5], [2, 4, 3]]
    b_values = [[1, 9], [1]]
    expected_indices = [[0, 0], [0, 1]]
    expected_values = _values([1, 9], dtype)
    expected_shape = [2, 2]
    expected_counts = [2, 0]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    intersection = self._set_intersection(a, sp_b)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        intersection,
        dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_intersection_count(a, sp_b))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    intersection = self._set_intersection(sp_a, sp_b)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        intersection,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_intersection_count(sp_a, sp_b))

  @test_util.run_deprecated_v1
  def test_dense_set_intersection_multirow_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the dense-only multirow intersection check for every dtype.
    for dtype in _DTYPES:
      self._test_dense_set_intersection_multirow_2d(dtype)

  def _test_dense_set_intersection_multirow_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: intersects two dense two-row inputs of the given dtype; asserts the result sparse
    # indices, values and shape, and per-row counts (two shared in the first row, none in the
    # second). Returns nothing.
    a_values = [[9, 1, 5], [2, 4, 3]]
    b_values = [[1, 9], [1, 5]]
    expected_indices = [[0, 0], [0, 1]]
    expected_values = _values([1, 9], dtype)
    expected_shape = [2, 2]
    expected_counts = [2, 0]

    # Dense to dense.
    a = _constant(a_values, dtype)
    b = _constant(b_values, dtype)
    intersection = self._set_intersection(a, b)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        intersection,
        dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_intersection_count(a, b))

  @test_util.run_deprecated_v1
  def test_set_intersection_duplicates_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the intersection-with-duplicates check for every dtype.
    for dtype in _DTYPES:
      self._test_set_intersection_duplicates_2d(dtype)

  def _test_set_intersection_duplicates_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: intersects a single row with a repeated value against a one-element row, in
    # dense-dense, dense-sparse and sparse-sparse combinations. Asserts the duplicate appears
    # once in the result with matching indices, shape and count. Returns nothing.
    a_values = [[1, 1, 3]]
    b_values = [[1]]
    expected_indices = [[0, 0]]
    expected_values = _values([1], dtype)
    expected_shape = [1, 1]
    expected_counts = [1]

    # Dense to dense.
    a = _constant(a_values, dtype=dtype)
    b = _constant(b_values, dtype=dtype)
    intersection = self._set_intersection(a, b)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        intersection,
        dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_intersection_count(a, b))

    # Dense to sparse.
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    intersection = self._set_intersection(a, sp_b)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        intersection,
        dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_intersection_count(a, sp_b))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    intersection = self._set_intersection(sp_a, sp_b)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        intersection,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_intersection_count(sp_a, sp_b))

  @test_util.run_deprecated_v1
  def test_set_intersection_3d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the three-dimensional intersection check for every dtype.
    for dtype in _DTYPES:
      self._test_set_intersection_3d(dtype=dtype)

  def test_set_intersection_3d_invalid_indices(self):
    # [FUNCTION DOC]
    # Test: runs the three-dimensional intersection check with out-of-order indices for every
    # dtype; each is expected to raise an out-of-order op error.
    for dtype in _DTYPES:
      self._test_set_intersection_3d(dtype=dtype, invalid_indices=True)

  def _test_set_intersection_3d(self, dtype, invalid_indices=False):
    # [FUNCTION DOC]
    # HELPER: builds two three-dimensional sparse tensors of the dtype (shapes four by two by
    # three and four by two by four) with some empty sets. With the flag (default false) the
    # first operand has out-of-order indices and intersection must raise an op error about order.
    # Otherwise it checks sparse-sparse intersection (indices, values, shape four by two by two,
    # counts), and for dtypes supported by the dense conversion also dense-sparse and dense-dense
    # intersections made by converting back to dense with distinct fill values per operand.
    # Returns nothing.
    if invalid_indices:
      indices = constant_op.constant(
          [
              [0, 1, 0],
              [0, 1, 1],  # 0,1
              [1, 0, 0],  # 1,0
              [1, 1, 0],
              [1, 1, 1],
              [1, 1, 2],  # 1,1
              [0, 0, 0],
              [0, 0, 2],  # 0,0
              # 2,0
              [2, 1, 1]  # 2,1
              # 3,*
          ],
          dtypes.int64)
    else:
      indices = constant_op.constant(
          [
              [0, 0, 0],
              [0, 0, 2],  # 0,0
              [0, 1, 0],
              [0, 1, 1],  # 0,1
              [1, 0, 0],  # 1,0
              [1, 1, 0],
              [1, 1, 1],
              [1, 1, 2],  # 1,1
              # 2,0
              [2, 1, 1]  # 2,1
              # 3,*
          ],
          dtypes.int64)
    sp_a = sparse_tensor_lib.SparseTensor(
        indices,
        _constant(
            [
                1,
                9,  # 0,0
                3,
                3,  # 0,1
                1,  # 1,0
                9,
                7,
                8,  # 1,1
                # 2,0
                5  # 2,1
                # 3,*
            ],
            dtype),
        constant_op.constant([4, 2, 3], dtypes.int64))
    sp_b = sparse_tensor_lib.SparseTensor(
        constant_op.constant(
            [
                [0, 0, 0],
                [0, 0, 3],  # 0,0
                # 0,1
                [1, 0, 0],  # 1,0
                [1, 1, 0],
                [1, 1, 1],  # 1,1
                [2, 0, 1],  # 2,0
                [2, 1, 1],  # 2,1
                [3, 0, 0],  # 3,0
                [3, 1, 0]  # 3,1
            ],
            dtypes.int64),
        _constant(
            [
                1,
                3,  # 0,0
                # 0,1
                3,  # 1,0
                7,
                8,  # 1,1
                2,  # 2,0
                5,  # 2,1
                4,  # 3,0
                4  # 3,1
            ],
            dtype),
        constant_op.constant([4, 2, 4], dtypes.int64))

    if invalid_indices:
      with self.assertRaisesRegex(errors_impl.OpError, "out of order"):
        self._set_intersection(sp_a, sp_b)
    else:
      expected_indices = [
          [0, 0, 0],  # 0,0
          # 0,1
          # 1,0
          [1, 1, 0],
          [1, 1, 1],  # 1,1
          # 2,0
          [2, 1, 0],  # 2,1
          # 3,*
      ]
      expected_values = _values(
          [
              1,  # 0,0
              # 0,1
              # 1,0
              7,
              8,  # 1,1
              # 2,0
              5,  # 2,1
              # 3,*
          ],
          dtype)
      expected_shape = [4, 2, 2]
      expected_counts = [
          [
              1,  # 0,0
              0  # 0,1
          ],
          [
              0,  # 1,0
              2  # 1,1
          ],
          [
              0,  # 2,0
              1  # 2,1
          ],
          [
              0,  # 3,0
              0  # 3,1
          ]
      ]

      # Sparse to sparse.
      intersection = self._set_intersection(sp_a, sp_b)
      self._assert_set_operation(
          expected_indices,
          expected_values,
          expected_shape,
          intersection,
          dtype=dtype)
      self.assertAllEqual(expected_counts,
                          self._set_intersection_count(sp_a, sp_b))

      # NOTE: sparse_to_dense doesn't support uint8 and uint16.
      if dtype not in [dtypes.uint8, dtypes.uint16]:
        # Dense to sparse.
        a = math_ops.cast(
            sparse_ops.sparse_to_dense(
                sp_a.indices,
                sp_a.dense_shape,
                sp_a.values,
                default_value="-1" if dtype == dtypes.string else -1),
            dtype=dtype)
        intersection = self._set_intersection(a, sp_b)
        self._assert_set_operation(
            expected_indices,
            expected_values,
            expected_shape,
            intersection,
            dtype=dtype)
        self.assertAllEqual(expected_counts,
                            self._set_intersection_count(a, sp_b))

        # Dense to dense.
        b = math_ops.cast(
            sparse_ops.sparse_to_dense(
                sp_b.indices,
                sp_b.dense_shape,
                sp_b.values,
                default_value="-2" if dtype == dtypes.string else -2),
            dtype=dtype)
        intersection = self._set_intersection(a, b)
        self._assert_set_operation(
            expected_indices,
            expected_values,
            expected_shape,
            intersection,
            dtype=dtype)
        self.assertAllEqual(expected_counts, self._set_intersection_count(a, b))

  def _assert_static_shapes(self, input_tensor, result_sparse_tensor):
    # [FUNCTION DOC]
    # HELPER: asserts the static shapes of a set-operation result. Derives the expected rank from
    # the input (the length of the dense-shape vector for sparse input, or the rank for dense),
    # then asserts result indices have shape unknown-by-rank, values are a vector of unknown
    # length, and the dense shape vector has length equal to the rank. Returns nothing.
    if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      sparse_shape_dims = input_tensor.dense_shape.get_shape().dims
      if sparse_shape_dims is None:
        expected_rank = None
      else:
        expected_rank = sparse_shape_dims[0].value
    else:
      expected_rank = input_tensor.get_shape().ndims
    self.assertAllEqual((None, expected_rank),
                        result_sparse_tensor.indices.get_shape().as_list())
    self.assertAllEqual((None,),
                        result_sparse_tensor.values.get_shape().as_list())
    self.assertAllEqual((expected_rank,),
                        result_sparse_tensor.dense_shape.get_shape().as_list())

  def _run_equivalent_set_ops(self, ops):
    # [FUNCTION DOC]
    # HELPER: takes a sequence of sparse-result ops that should be equivalent. In a cached
    # session it asserts their static indices/values shapes agree, evaluates them together with
    # their dynamic shapes, asserts static shapes are compatible with dynamic ones and with the
    # actual result shapes, and that all ops give identical indices, values and dense shapes.
    # Returns the first op's evaluated result.
    """Assert all ops return the same shapes, and return 1st result."""
    # Collect shapes and results for all ops, and assert static shapes match.
    dynamic_indices_shape_ops = []
    dynamic_values_shape_ops = []
    static_indices_shape = None
    static_values_shape = None
    with self.cached_session() as sess:
      for op in ops:
        if static_indices_shape is None:
          static_indices_shape = op.indices.get_shape()
        else:
          self.assertAllEqual(
              static_indices_shape.as_list(), op.indices.get_shape().as_list())
        if static_values_shape is None:
          static_values_shape = op.values.get_shape()
        else:
          self.assertAllEqual(
              static_values_shape.as_list(), op.values.get_shape().as_list())
        dynamic_indices_shape_ops.append(array_ops.shape(op.indices))
        dynamic_values_shape_ops.append(array_ops.shape(op.values))
      results = sess.run(
          list(ops) + dynamic_indices_shape_ops + dynamic_values_shape_ops)
      op_count = len(ops)
      op_results = results[0:op_count]
      dynamic_indices_shapes = results[op_count:2 * op_count]
      dynamic_values_shapes = results[2 * op_count:3 * op_count]

    # Assert static and dynamic tensor shapes, and result shapes, are all
    # consistent.
    static_indices_shape.assert_is_compatible_with(dynamic_indices_shapes[0])
    static_values_shape.assert_is_compatible_with(dynamic_values_shapes[0])
    self.assertAllEqual(dynamic_indices_shapes[0], op_results[0].indices.shape)
    self.assertAllEqual(dynamic_values_shapes[0], op_results[0].values.shape)

    # Assert dynamic shapes and values are the same for all ops.
    for i in range(1, len(ops)):
      self.assertAllEqual(dynamic_indices_shapes[0], dynamic_indices_shapes[i])
      self.assertAllEqual(dynamic_values_shapes[0], dynamic_values_shapes[i])
      self.assertAllEqual(op_results[0].indices, op_results[i].indices)
      self.assertAllEqual(op_results[0].values, op_results[i].values)
      self.assertAllEqual(op_results[0].dense_shape, op_results[i].dense_shape)

    return op_results[0]

  def _set_intersection(self, a, b):
    # [FUNCTION DOC]
    # HELPER: computes the set intersection of a and b (dense or sparse) four ways: validation of
    # indices on and off, each with the operands in both orders. Asserts static shapes on each,
    # asserts all four evaluate identically, and returns the evaluated sparse result.
    # Validate that we get the same results with or without `validate_indices`,
    # and with a & b swapped.
    ops = (
        sets.set_intersection(
            a, b, validate_indices=True),
        sets.set_intersection(
            a, b, validate_indices=False),
        sets.set_intersection(
            b, a, validate_indices=True),
        sets.set_intersection(
            b, a, validate_indices=False),)
    for op in ops:
      self._assert_static_shapes(a, op)
    return self._run_equivalent_set_ops(ops)

  def _set_intersection_count(self, a, b):
    # [FUNCTION DOC]
    # HELPER: evaluates and returns the per-set sizes of the intersection of a and b.
    # KNOWN ISSUE: opens a cached session bound to an unused name; evaluation does not use it.
    op = sets.set_size(sets.set_intersection(a, b))
    with self.cached_session() as sess:
      return self.evaluate(op)

  @test_util.run_deprecated_v1
  def test_set_difference_multirow_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the multirow two-dimensional difference check for every dtype.
    for dtype in _DTYPES:
      self._test_set_difference_multirow_2d(dtype)

  def _test_set_difference_multirow_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: four rows with empty and repeated sets. For the dtype, checks first-minus-second
    # and second-minus-first differences, each with dense-sparse and sparse-sparse operands,
    # asserting result indices, values, shape and per-row counts. Returns nothing.
    a_values = [[1, 1, 1], [1, 5, 9], [4, 5, 3], [5, 5, 1]]
    b_values = [[], [1, 2], [1, 2, 2], []]

    # a - b.
    expected_indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0],
                        [3, 1]]
    expected_values = _values([1, 5, 9, 3, 4, 5, 1, 5], dtype)
    expected_shape = [4, 3]
    expected_counts = [1, 2, 3, 2]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    difference = self._set_difference(a, sp_b, True)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, sp_b, True))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    difference = self._set_difference(sp_a, sp_b, True)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(sp_a, sp_b, True))

    # b - a.
    expected_indices = [[1, 0], [2, 0], [2, 1]]
    expected_values = _values([2, 1, 2], dtype)
    expected_shape = [4, 2]
    expected_counts = [0, 1, 2, 0]

    # Dense to sparse.
    difference = self._set_difference(a, sp_b, False)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, sp_b, False))

    # Sparse to sparse.
    difference = self._set_difference(sp_a, sp_b, False)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(sp_a, sp_b, False))

  @test_util.run_deprecated_v1
  def test_dense_set_difference_multirow_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the dense-only multirow difference check for every dtype.
    for dtype in _DTYPES:
      self._test_dense_set_difference_multirow_2d(dtype)

  def _test_dense_set_difference_multirow_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: two dense rows per operand. Checks both difference directions with dense operands
    # for the dtype, asserting indices, values, shape and counts. Returns nothing.
    a_values = [[1, 5, 9], [4, 5, 3]]
    b_values = [[1, 2, 6], [1, 2, 2]]

    # a - b.
    expected_indices = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
    expected_values = _values([5, 9, 3, 4, 5], dtype)
    expected_shape = [2, 3]
    expected_counts = [2, 3]

    # Dense to dense.
    a = _constant(a_values, dtype=dtype)
    b = _constant(b_values, dtype=dtype)
    difference = self._set_difference(a, b, True)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_difference_count(a, b, True))

    # b - a.
    expected_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_values = _values([2, 6, 1, 2], dtype)
    expected_shape = [2, 2]
    expected_counts = [2, 2]

    # Dense to dense.
    difference = self._set_difference(a, b, False)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, b, False))

  @test_util.run_deprecated_v1
  def test_sparse_set_difference_multirow_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the sparse-only multirow difference check for every dtype.
    for dtype in _DTYPES:
      self._test_sparse_set_difference_multirow_2d(dtype)

  def _test_sparse_set_difference_multirow_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: two ragged sparse operands (with empty rows and repeats). Checks both difference
    # directions for the dtype, asserting indices, values, shape and per-row counts. Returns
    # nothing.
    sp_a = _dense_to_sparse(
        [[], [1, 5, 9], [4, 5, 3, 3, 4, 5], [5, 1]], dtype=dtype)
    sp_b = _dense_to_sparse([[], [1, 2], [1, 2, 2], []], dtype=dtype)

    # a - b.
    expected_indices = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
    expected_values = _values([5, 9, 3, 4, 5, 1, 5], dtype)
    expected_shape = [4, 3]
    expected_counts = [0, 2, 3, 2]

    difference = self._set_difference(sp_a, sp_b, True)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(sp_a, sp_b, True))

    # b - a.
    expected_indices = [[1, 0], [2, 0], [2, 1]]
    expected_values = _values([2, 1, 2], dtype)
    expected_shape = [4, 2]
    expected_counts = [0, 1, 2, 0]

    difference = self._set_difference(sp_a, sp_b, False)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(sp_a, sp_b, False))

  @test_util.run_deprecated_v1
  def test_set_difference_duplicates_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the difference-with-duplicates check for every dtype.
    for dtype in _DTYPES:
      self._test_set_difference_duplicates_2d(dtype)

  def _test_set_difference_duplicates_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: one-row operands with repeated values. Checks both difference directions, each with
    # dense-sparse and sparse-sparse operands for the dtype, asserting that duplicates collapse
    # (indices, values, shape, counts). Returns nothing.
    # KNOWN ISSUE: in the sparse-sparse cases the count assertions are computed from the dense
    # first operand instead of the sparse one, so the count of the sparse-sparse path is never
    # actually tested.
    a_values = [[1, 1, 3]]
    b_values = [[1, 2, 2]]

    # a - b.
    expected_indices = [[0, 0]]
    expected_values = _values([3], dtype)
    expected_shape = [1, 1]
    expected_counts = [1]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    difference = self._set_difference(a, sp_b, True)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, sp_b, True))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    difference = self._set_difference(sp_a, sp_b, True)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, sp_b, True))

    # b - a.
    expected_indices = [[0, 0]]
    expected_values = _values([2], dtype)
    expected_shape = [1, 1]
    expected_counts = [1]

    # Dense to sparse.
    difference = self._set_difference(a, sp_b, False)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, sp_b, False))

    # Sparse to sparse.
    difference = self._set_difference(sp_a, sp_b, False)
    self._assert_set_operation(
        expected_indices,
        expected_values,
        expected_shape,
        difference,
        dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, sp_b, False))

  @test_util.run_deprecated_v1
  def test_sparse_set_difference_3d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the three-dimensional sparse difference check for every dtype.
    for dtype in _DTYPES:
      self._test_sparse_set_difference_3d(dtype)

  def test_sparse_set_difference_3d_invalid_indices(self):
    # [FUNCTION DOC]
    # Test: runs the three-dimensional difference check with out-of-order indices for every
    # dtype; each is expected to raise an out-of-order op error.
    for dtype in _DTYPES:
      self._test_sparse_set_difference_3d(dtype, invalid_indices=True)

  def _test_sparse_set_difference_3d(self, dtype, invalid_indices=False):
    # [FUNCTION DOC]
    # HELPER: builds two three-dimensional sparse operands (four by two by three and four by two
    # by four) of the dtype with some empty sets. With the flag (default false) the first has
    # out-of-order indices and both difference directions must raise an op error about order.
    # Otherwise it checks the first-minus-second and second-minus-first differences, asserting
    # indices, values, shape and per-set counts. Returns nothing.
    if invalid_indices:
      indices = constant_op.constant(
          [
              [0, 1, 0],
              [0, 1, 1],  # 0,1
              [1, 0, 0],  # 1,0
              [1, 1, 0],
              [1, 1, 1],
              [1, 1, 2],  # 1,1
              [0, 0, 0],
              [0, 0, 2],  # 0,0
              # 2,0
              [2, 1, 1]  # 2,1
              # 3,*
          ],
          dtypes.int64)
    else:
      indices = constant_op.constant(
          [
              [0, 0, 0],
              [0, 0, 2],  # 0,0
              [0, 1, 0],
              [0, 1, 1],  # 0,1
              [1, 0, 0],  # 1,0
              [1, 1, 0],
              [1, 1, 1],
              [1, 1, 2],  # 1,1
              # 2,0
              [2, 1, 1]  # 2,1
              # 3,*
          ],
          dtypes.int64)
    sp_a = sparse_tensor_lib.SparseTensor(
        indices,
        _constant(
            [
                1,
                9,  # 0,0
                3,
                3,  # 0,1
                1,  # 1,0
                9,
                7,
                8,  # 1,1
                # 2,0
                5  # 2,1
                # 3,*
            ],
            dtype),
        constant_op.constant([4, 2, 3], dtypes.int64))
    sp_b = sparse_tensor_lib.SparseTensor(
        constant_op.constant(
            [
                [0, 0, 0],
                [0, 0, 3],  # 0,0
                # 0,1
                [1, 0, 0],  # 1,0
                [1, 1, 0],
                [1, 1, 1],  # 1,1
                [2, 0, 1],  # 2,0
                [2, 1, 1],  # 2,1
                [3, 0, 0],  # 3,0
                [3, 1, 0]  # 3,1
            ],
            dtypes.int64),
        _constant(
            [
                1,
                3,  # 0,0
                # 0,1
                3,  # 1,0
                7,
                8,  # 1,1
                2,  # 2,0
                5,  # 2,1
                4,  # 3,0
                4  # 3,1
            ],
            dtype),
        constant_op.constant([4, 2, 4], dtypes.int64))

    if invalid_indices:
      with self.assertRaisesRegex(errors_impl.OpError, "out of order"):
        self._set_difference(sp_a, sp_b, False)
      with self.assertRaisesRegex(errors_impl.OpError, "out of order"):
        self._set_difference(sp_a, sp_b, True)
    else:
      # a-b
      expected_indices = [
          [0, 0, 0],  # 0,0
          [0, 1, 0],  # 0,1
          [1, 0, 0],  # 1,0
          [1, 1, 0],  # 1,1
          # 2,*
          # 3,*
      ]
      expected_values = _values(
          [
              9,  # 0,0
              3,  # 0,1
              1,  # 1,0
              9,  # 1,1
              # 2,*
              # 3,*
          ],
          dtype)
      expected_shape = [4, 2, 1]
      expected_counts = [
          [
              1,  # 0,0
              1  # 0,1
          ],
          [
              1,  # 1,0
              1  # 1,1
          ],
          [
              0,  # 2,0
              0  # 2,1
          ],
          [
              0,  # 3,0
              0  # 3,1
          ]
      ]

      difference = self._set_difference(sp_a, sp_b, True)
      self._assert_set_operation(
          expected_indices,
          expected_values,
          expected_shape,
          difference,
          dtype=dtype)
      self.assertAllEqual(expected_counts,
                          self._set_difference_count(sp_a, sp_b))

      # b-a
      expected_indices = [
          [0, 0, 0],  # 0,0
          # 0,1
          [1, 0, 0],  # 1,0
          # 1,1
          [2, 0, 0],  # 2,0
          # 2,1
          [3, 0, 0],  # 3,0
          [3, 1, 0]  # 3,1
      ]
      expected_values = _values(
          [
              3,  # 0,0
              # 0,1
              3,  # 1,0
              # 1,1
              2,  # 2,0
              # 2,1
              4,  # 3,0
              4,  # 3,1
          ],
          dtype)
      expected_shape = [4, 2, 1]
      expected_counts = [
          [
              1,  # 0,0
              0  # 0,1
          ],
          [
              1,  # 1,0
              0  # 1,1
          ],
          [
              1,  # 2,0
              0  # 2,1
          ],
          [
              1,  # 3,0
              1  # 3,1
          ]
      ]

      difference = self._set_difference(sp_a, sp_b, False)
      self._assert_set_operation(
          expected_indices,
          expected_values,
          expected_shape,
          difference,
          dtype=dtype)
      self.assertAllEqual(expected_counts,
                          self._set_difference_count(sp_a, sp_b, False))

  def _set_difference(self, a, b, aminusb=True):
    # [FUNCTION DOC]
    # HELPER: computes the set difference of a and b (dense or sparse). The flag (default true)
    # selects first-minus-second versus second-minus-first. Builds four ops: validation on and
    # off, and with operands swapped and the direction flipped (which must be equivalent for the
    # sparse result shapes). Asserts static shapes on each, asserts they evaluate identically, and
    # returns the evaluated sparse result.
    # Validate that we get the same results with or without `validate_indices`,
    # and with a & b swapped.
    ops = (
        sets.set_difference(
            a, b, aminusb=aminusb, validate_indices=True),
        sets.set_difference(
            a, b, aminusb=aminusb, validate_indices=False),
        sets.set_difference(
            b, a, aminusb=not aminusb, validate_indices=True),
        sets.set_difference(
            b, a, aminusb=not aminusb, validate_indices=False),)
    for op in ops:
      self._assert_static_shapes(a, op)
    return self._run_equivalent_set_ops(ops)

  def _set_difference_count(self, a, b, aminusb=True):
    # [FUNCTION DOC]
    # HELPER: evaluates and returns the per-set sizes of the difference of a and b, in the
    # direction selected by the flag (default first-minus-second).
    # KNOWN ISSUE: opens a cached session bound to an unused name; evaluation does not use it.
    op = sets.set_size(sets.set_difference(a, b, aminusb))
    with self.cached_session() as sess:
      return self.evaluate(op)

  @test_util.run_deprecated_v1
  def test_set_union_multirow_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the multirow two-dimensional union check for every dtype.
    for dtype in _DTYPES:
      self._test_set_union_multirow_2d(dtype)

  def _test_set_union_multirow_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: unions two two-row operands for the dtype, with dense-sparse and sparse-sparse
    # combinations, asserting result indices, values, shape and per-row counts. Returns nothing.
    a_values = [[9, 1, 5], [2, 4, 3]]
    b_values = [[1, 9], [1]]
    expected_indices = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]]
    expected_values = _values([1, 5, 9, 1, 2, 3, 4], dtype)
    expected_shape = [2, 4]
    expected_counts = [3, 4]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    union = self._set_union(a, sp_b)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, union, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_union_count(a, sp_b))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    union = self._set_union(sp_a, sp_b)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, union, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_union_count(sp_a, sp_b))

  @test_util.run_deprecated_v1
  def test_dense_set_union_multirow_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the dense-only multirow union check for every dtype.
    for dtype in _DTYPES:
      self._test_dense_set_union_multirow_2d(dtype)

  def _test_dense_set_union_multirow_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: unions two dense two-row operands for the dtype; asserts result indices, values,
    # shape and per-row counts. Returns nothing.
    a_values = [[9, 1, 5], [2, 4, 3]]
    b_values = [[1, 9], [1, 2]]
    expected_indices = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]]
    expected_values = _values([1, 5, 9, 1, 2, 3, 4], dtype)
    expected_shape = [2, 4]
    expected_counts = [3, 4]

    # Dense to dense.
    a = _constant(a_values, dtype=dtype)
    b = _constant(b_values, dtype=dtype)
    union = self._set_union(a, b)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, union, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_union_count(a, b))

  @test_util.run_deprecated_v1
  def test_set_union_duplicates_2d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the union-with-duplicates check for every dtype.
    for dtype in _DTYPES:
      self._test_set_union_duplicates_2d(dtype)

  def _test_set_union_duplicates_2d(self, dtype):
    # [FUNCTION DOC]
    # HELPER: unions a single row with a repeated value and a one-element row (dense-sparse and
    # sparse-sparse) for the dtype; asserts duplicates collapse (indices, values, shape) and the
    # count is two. Returns nothing.
    a_values = [[1, 1, 3]]
    b_values = [[1]]
    expected_indices = [[0, 0], [0, 1]]
    expected_values = _values([1, 3], dtype)
    expected_shape = [1, 2]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    union = self._set_union(a, sp_b)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, union, dtype=dtype)
    self.assertAllEqual([2], self._set_union_count(a, sp_b))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    union = self._set_union(sp_a, sp_b)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, union, dtype=dtype)
    self.assertAllEqual([2], self._set_union_count(sp_a, sp_b))

  @test_util.run_deprecated_v1
  def test_sparse_set_union_3d(self):
    # [FUNCTION DOC]
    # Test (graph mode): runs the three-dimensional sparse union check for every dtype.
    for dtype in _DTYPES:
      self._test_sparse_set_union_3d(dtype)

  def test_sparse_set_union_3d_invalid_indices(self):
    # [FUNCTION DOC]
    # Test: runs the three-dimensional union check with out-of-order indices for every dtype;
    # each is expected to raise an out-of-order op error.
    for dtype in _DTYPES:
      self._test_sparse_set_union_3d(dtype, invalid_indices=True)

  def _test_sparse_set_union_3d(self, dtype, invalid_indices=False):
    # [FUNCTION DOC]
    # HELPER: builds two three-dimensional sparse operands (four by two by three and four by two
    # by four) of the dtype with some empty sets. With the flag (default false) the first has
    # out-of-order indices and the union must raise an op error about order. Otherwise it checks
    # the union's indices, values, shape (four by two by three) and per-set counts. Returns
    # nothing.
    # KNOWN ISSUE: the variable holding the union result is named as if it were an intersection.
    if invalid_indices:
      indices = constant_op.constant(
          [
              [0, 1, 0],
              [0, 1, 1],  # 0,1
              [1, 0, 0],  # 1,0
              [0, 0, 0],
              [0, 0, 2],  # 0,0
              [1, 1, 0],
              [1, 1, 1],
              [1, 1, 2],  # 1,1
              # 2,0
              [2, 1, 1]  # 2,1
              # 3,*
          ],
          dtypes.int64)
    else:
      indices = constant_op.constant(
          [
              [0, 0, 0],
              [0, 0, 2],  # 0,0
              [0, 1, 0],
              [0, 1, 1],  # 0,1
              [1, 0, 0],  # 1,0
              [1, 1, 0],
              [1, 1, 1],
              [1, 1, 2],  # 1,1
              # 2,0
              [2, 1, 1]  # 2,1
              # 3,*
          ],
          dtypes.int64)
    sp_a = sparse_tensor_lib.SparseTensor(
        indices,
        _constant(
            [
                1,
                9,  # 0,0
                3,
                3,  # 0,1
                1,  # 1,0
                9,
                7,
                8,  # 1,1
                # 2,0
                5  # 2,1
                # 3,*
            ],
            dtype),
        constant_op.constant([4, 2, 3], dtypes.int64))
    sp_b = sparse_tensor_lib.SparseTensor(
        constant_op.constant(
            [
                [0, 0, 0],
                [0, 0, 3],  # 0,0
                # 0,1
                [1, 0, 0],  # 1,0
                [1, 1, 0],
                [1, 1, 1],  # 1,1
                [2, 0, 1],  # 2,0
                [2, 1, 1],  # 2,1
                [3, 0, 0],  # 3,0
                [3, 1, 0]  # 3,1
            ],
            dtypes.int64),
        _constant(
            [
                1,
                3,  # 0,0
                # 0,1
                3,  # 1,0
                7,
                8,  # 1,1
                2,  # 2,0
                5,  # 2,1
                4,  # 3,0
                4  # 3,1
            ],
            dtype),
        constant_op.constant([4, 2, 4], dtypes.int64))

    if invalid_indices:
      with self.assertRaisesRegex(errors_impl.OpError, "out of order"):
        self._set_union(sp_a, sp_b)
    else:
      expected_indices = [
          [0, 0, 0],
          [0, 0, 1],
          [0, 0, 2],  # 0,0
          [0, 1, 0],  # 0,1
          [1, 0, 0],
          [1, 0, 1],  # 1,0
          [1, 1, 0],
          [1, 1, 1],
          [1, 1, 2],  # 1,1
          [2, 0, 0],  # 2,0
          [2, 1, 0],  # 2,1
          [3, 0, 0],  # 3,0
          [3, 1, 0],  # 3,1
      ]
      expected_values = _values(
          [
              1,
              3,
              9,  # 0,0
              3,  # 0,1
              1,
              3,  # 1,0
              7,
              8,
              9,  # 1,1
              2,  # 2,0
              5,  # 2,1
              4,  # 3,0
              4,  # 3,1
          ],
          dtype)
      expected_shape = [4, 2, 3]
      expected_counts = [
          [
              3,  # 0,0
              1  # 0,1
          ],
          [
              2,  # 1,0
              3  # 1,1
          ],
          [
              1,  # 2,0
              1  # 2,1
          ],
          [
              1,  # 3,0
              1  # 3,1
          ]
      ]

      intersection = self._set_union(sp_a, sp_b)
      self._assert_set_operation(
          expected_indices,
          expected_values,
          expected_shape,
          intersection,
          dtype=dtype)
      self.assertAllEqual(expected_counts, self._set_union_count(sp_a, sp_b))

  def _set_union(self, a, b):
    # [FUNCTION DOC]
    # HELPER: computes the set union of a and b (dense or sparse) four ways: validation on and
    # off, each with operands in both orders. Asserts static shapes on each, asserts all four
    # evaluate identically, and returns the evaluated sparse result.
    # Validate that we get the same results with or without `validate_indices`,
    # and with a & b swapped.
    ops = (
        sets.set_union(
            a, b, validate_indices=True),
        sets.set_union(
            a, b, validate_indices=False),
        sets.set_union(
            b, a, validate_indices=True),
        sets.set_union(
            b, a, validate_indices=False),)
    for op in ops:
      self._assert_static_shapes(a, op)
    return self._run_equivalent_set_ops(ops)

  def _set_union_count(self, a, b):
    # [FUNCTION DOC]
    # HELPER: evaluates and returns the per-set sizes of the union of a and b.
    # KNOWN ISSUE: opens a cached session bound to an unused name; evaluation does not use it.
    op = sets.set_size(sets.set_union(a, b))
    with self.cached_session() as sess:
      return self.evaluate(op)

  def _assert_set_operation(self, expected_indices, expected_values,
                            expected_shape, sparse_tensor_value, dtype):
    # [FUNCTION DOC]
    # HELPER: shared checker for set-operation results. Inputs: expected indices, expected
    # values, expected dense shape, an evaluated sparse result and the dtype. Asserts the indices
    # equal the expected ones and the value counts match. Values are compared as sets per group of
    # entries sharing all but the last index (string values are decoded from bytes first), so the
    # order within a group is not checked. Finally asserts the dense shape. Returns nothing.
    self.assertAllEqual(expected_indices, sparse_tensor_value.indices)
    self.assertAllEqual(len(expected_indices), len(expected_values))
    self.assertAllEqual(len(expected_values), len(sparse_tensor_value.values))
    expected_set = set()
    actual_set = set()
    last_indices = None
    for indices, expected_value, actual_value in zip(
        expected_indices, expected_values, sparse_tensor_value.values):
      if dtype == dtypes.string:
        actual_value = actual_value.decode("utf-8")
      if last_indices and (last_indices[:-1] != indices[:-1]):
        self.assertEqual(
            expected_set, actual_set,
            "Expected %s, got %s, at %s." % (expected_set, actual_set, indices))
        expected_set.clear()
        actual_set.clear()
      expected_set.add(expected_value)
      actual_set.add(actual_value)
      last_indices = indices
    self.assertEqual(
        expected_set, actual_set, "Expected %s, got %s, at %s." %
        (expected_set, actual_set, last_indices))
    self.assertAllEqual(expected_shape, sparse_tensor_value.dense_shape)

  @parameterized.parameters(*_DTYPES)
  def test_set_union_output_is_sorted(self, dtype):
    # [FUNCTION DOC]
    # Parameterized test (one case per supported dtype): unions two sparse tensors with
    # unsorted, overlapping and empty rows (values kept below ten so text order matches numeric
    # order) and asserts the result has the expected dense shape, indices and values, with each
    # row's union sorted ascending. Returns nothing.
    # We don't use any numbers >= 10 so that lexicographical order agrees with
    # numeric order in this test, for the type dtype == tf.string.

    # [3 7 5 3 1]
    # [2 6 5 4]
    # []
    # [9 8]
    sp_a = sparse_tensor_lib.SparseTensor(
        indices=constant_op.constant(
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2],
             [1, 3], [3, 0], [3, 1]],
            dtype=dtypes.int64),
        values=_constant([3, 7, 5, 3, 1, 2, 6, 5, 4, 9, 8], dtype),
        dense_shape=constant_op.constant([4, 5], dtype=dtypes.int64))

    # [9 7]
    # [5 2 0]
    # [6]
    # []
    sp_b = sparse_tensor_lib.SparseTensor(
        indices=constant_op.constant(
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0]],
            dtype=dtypes.int64),
        values=_constant([9, 7, 5, 2, 0, 6], dtype),
        dense_shape=constant_op.constant([4, 3], dtype=dtypes.int64))
    # The union should be
    # [1 3 5 7 9]
    # [0 2 4 5 6]
    # [6]
    # [8 9]
    result = sets.set_union(sp_a, sp_b)
    self.assertAllEqual(result.dense_shape, [4, 5])
    self.assertAllEqual(result.indices,
                        [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1],
                         [1, 2], [1, 3], [1, 4], [2, 0], [3, 0], [3, 1]])
    self.assertAllEqual(
        result.values,
        _constant([1, 3, 5, 7, 9, 0, 2, 4, 5, 6, 6, 8, 9], dtype))

  def test_raw_ops_setsize_invalid_shape(self):
    # [FUNCTION DOC]
    # Test: calls the low-level set-size op directly with a scalar (non-vector) shape input and
    # expects an invalid-argument error stating that the shape must be a one-dimensional tensor.
    # Returns nothing.
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Shape must be a 1D tensor"):
      invalid_shape = 1
      self.evaluate(
          gen_set_ops.set_size(
              set_indices=1,
              set_values=[1, 1],
              set_shape=invalid_shape,
              validate_indices=True,
              name=""))


if __name__ == "__main__":
  googletest.main()
