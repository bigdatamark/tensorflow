# [MODULE DOC]
# KNOWN ISSUE: the module docstring says the tests are for the reorder op, but this file tests
# the sparse split op (copy-paste leftover).
# Unit tests for the sparse split operation. Small fixtures (a 4x6 and a 5x7 integer matrix, a
# 3x4x2 string tensor and an empty 4x6 tensor) are split along rows or columns, with positive and
# negative axis values, and indices, values and dense shapes of every piece are compared with
# expected results. Also covers splitting followed by concatenation, argument validation errors,
# empty inputs, a large input regression test for a GPU memory bug, and out-of-bounds indices
# (ignored on GPU, error on CPU). Running it as a script starts the TensorFlow test runner.
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
"""Tests for SparseReorder."""

import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SparseSplitOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the sparse split op. It holds no state; helper methods build the shared
  # sparse tensor fixtures, and each test splits one of them and asserts on every piece.

  def _SparseTensor_4x6(self):
    # [FUNCTION DOC]
    # HELPER: builds the 4x6 integer sparse tensor fixture (14 stored entries) used by most split
    # tests. Takes no arguments, returns the sparse tensor with 64-bit indices and values, whose
    # values encode their row and column (e.g. 35 at row 3, column 5). No side effects.
    # [0 |  |2 |  |4 |5 ]
    # [  |11|  |13|14|  ]
    # [20|  |  |23|  |25]
    # [30|  |32|33|  |35]
    ind = np.array([[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4],
                    [2, 0], [2, 3], [2, 5], [3, 0], [3, 2], [3, 3],
                    [3, 5]]).astype(np.int64)
    val = np.array(
        [0, 2, 4, 5, 11, 13, 14, 20, 23, 25, 30, 32, 33, 35]).astype(np.int64)
    shape = np.array([4, 6]).astype(np.int64)
    return sparse_tensor.SparseTensor(ind, val, shape)

  def _SparseTensor_5x7(self):
    # [FUNCTION DOC]
    # HELPER: builds the 5x7 integer sparse tensor fixture (18 stored entries) used by the uneven
    # split tests. Takes no arguments, returns the sparse tensor with 64-bit indices and values.
    # No side effects.
    # [0 |  |2 |  |4 |5 |  ]
    # [  |11|  |13|14|  |16]
    # [20|  |  |23|  |25|  ]
    # [30|  |32|33|  |35|  ]
    # [  |41|  |  |44|  |46]
    ind = np.array([[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4],
                    [1, 6], [2, 0], [2, 3], [2, 5], [3, 0], [3, 2], [3, 3],
                    [3, 5], [4, 1], [4, 4], [4, 6]]).astype(np.int64)
    val = np.array(
        [0, 2, 4, 5, 11, 13, 14, 16, 20, 23, 25, 30, 32, 33, 35, 41, 44,
         46]).astype(np.int64)
    shape = np.array([5, 7]).astype(np.int64)
    return sparse_tensor.SparseTensor(ind, val, shape)

  def _SparseTensorValue_3x4x2(self):
    # [FUNCTION DOC]
    # HELPER: builds the raw 3x4x2 string sparse value fixture (10 entries such as a0, a1, b0).
    # Takes no arguments, returns a sparse tensor value (plain numpy indices, values and shape).
    # No side effects.
    #  slice(:,:, 0)
    #  ['a0'|    |'b0'|    ]
    #  [    |'c0'|    |'d0']
    #  [    |    |'e0'|    ]
    #  slice(:,:, 1)
    #  ['a1'|    |'b1'|    ]
    #  [    |'c1'|    |'d1']
    #  [    |    |'e1'|    ]
    ind = np.array([[0, 0, 0], [0, 0, 1], [0, 2, 0], [0, 2, 1], [1, 1, 0],
                    [1, 1, 1], [1, 3, 0], [1, 3, 1], [2, 2, 0],
                    [2, 2, 1]]).astype(np.int64)
    val = np.array(['a0', 'a1', 'b0', 'b1', 'c0', 'c1', 'd0', 'd1', 'e0', 'e1'])
    shape = np.array([3, 4, 2]).astype(np.int64)
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def _SparseTensor_3x4x2(self):
    # [FUNCTION DOC]
    # HELPER: converts the raw 3x4x2 string sparse value fixture into a sparse tensor.
    # Takes no arguments, returns that sparse tensor. No side effects.
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x4x2(
    ))

  def _SparseTensor_4x6_empty(self, val_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: builds a sparse tensor of dense shape 4 by 6 with no stored entries.
    # Input: val_dtype, numpy element type of the empty values array (default 64-bit integer).
    # Returns the empty sparse tensor. No side effects.
    ind = np.empty(shape=(0, 2), dtype=np.int64)
    val = np.array([]).astype(val_dtype)
    shape = np.array([4, 6]).astype(np.int64)
    return sparse_tensor.SparseTensor(ind, val, shape)

  def testSplitMatrixRows(self):
    # [FUNCTION DOC]
    # Scenario: split the 4x6 matrix into two pieces along the row axis, given both as axis 0 and
    # as the negative axis minus two, and evaluate.
    # Expected: two pieces of dense shape 2 by 6 whose indices (re-based per piece) and values
    # match the first two and last two rows.
    for axis in (0, -2):
      sp_tensors = self.evaluate(
          sparse_ops.sparse_split(
              sp_input=self._SparseTensor_4x6(), num_split=2, axis=axis))
      self.assertAllEqual(len(sp_tensors), 2)
      self.assertAllEqual(
          sp_tensors[0].indices,
          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4]])
      self.assertAllEqual(sp_tensors[0].values, [0, 2, 4, 5, 11, 13, 14])
      self.assertAllEqual(sp_tensors[0].dense_shape, [2, 6])
      self.assertAllEqual(
          sp_tensors[1].indices,
          [[0, 0], [0, 3], [0, 5], [1, 0], [1, 2], [1, 3], [1, 5]])
      self.assertAllEqual(sp_tensors[1].values, [20, 23, 25, 30, 32, 33, 35])
      self.assertAllEqual(sp_tensors[1].dense_shape, [2, 6])

  def testSplitMatrixUnevenCols(self):
    # [FUNCTION DOC]
    # Scenario: split the 5x7 matrix along the column axis (1 and negative one) into 3 pieces and
    # into 4 pieces; 7 is not divisible by either count. Only the 3-piece result is explicitly
    # evaluated; the 4-piece result is checked directly (works in eager execution).
    # Expected: the earlier pieces get the extra columns (widths 3, 2, 2 and 2, 2, 2, 1) with
    # re-based indices, matching values and dense shapes, and the expected number of pieces.
    for axis in (1, -1):
      sp_tensors_3 = self.evaluate(
          sparse_ops.sparse_split(
              sp_input=self._SparseTensor_5x7(), num_split=3, axis=axis))
      self.assertAllEqual(len(sp_tensors_3), 3)
      self.assertAllEqual(
          sp_tensors_3[0].indices,
          [[0, 0], [0, 2], [1, 1], [2, 0], [3, 0], [3, 2], [4, 1]])
      self.assertAllEqual(sp_tensors_3[0].values, [0, 2, 11, 20, 30, 32, 41])
      self.assertAllEqual(sp_tensors_3[0].dense_shape, [5, 3])
      self.assertAllEqual(sp_tensors_3[1].indices,
                          [[0, 1], [1, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensors_3[1].values, [4, 13, 14, 23, 33, 44])
      self.assertAllEqual(sp_tensors_3[1].dense_shape, [5, 2])
      self.assertAllEqual(sp_tensors_3[2].indices,
                          [[0, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensors_3[2].values, [5, 16, 25, 35, 46])
      self.assertAllEqual(sp_tensors_3[2].dense_shape, [5, 2])
      sp_tensors_4 = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_5x7(), num_split=4, axis=axis)
      self.assertAllEqual(len(sp_tensors_4), 4)
      self.assertAllEqual(sp_tensors_4[0].indices,
                          [[0, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensors_4[0].values, [0, 11, 20, 30, 41])
      self.assertAllEqual(sp_tensors_4[0].dense_shape, [5, 2])
      self.assertAllEqual(sp_tensors_4[1].indices,
                          [[0, 0], [1, 1], [2, 1], [3, 0], [3, 1]])
      self.assertAllEqual(sp_tensors_4[1].values, [2, 13, 23, 32, 33])
      self.assertAllEqual(sp_tensors_4[1].dense_shape, [5, 2])
      self.assertAllEqual(sp_tensors_4[2].indices,
                          [[0, 0], [0, 1], [1, 0], [2, 1], [3, 1], [4, 0]])
      self.assertAllEqual(sp_tensors_4[2].values, [4, 5, 14, 25, 35, 44])
      self.assertAllEqual(sp_tensors_4[2].dense_shape, [5, 2])
      self.assertAllEqual(sp_tensors_4[3].indices, [[1, 0], [4, 0]])
      self.assertAllEqual(sp_tensors_4[3].values, [16, 46])
      self.assertAllEqual(sp_tensors_4[3].dense_shape, [5, 1])

  def testSplitMatrixUnevenRows(self):
    # [FUNCTION DOC]
    # Scenario: split the 5x7 matrix along the row axis (0 and negative two) into 2 pieces and
    # into 3 pieces. The 3-piece result is not passed through explicit evaluation.
    # Expected: the earlier pieces get the extra rows (heights 3 and 2, then 2, 2 and 1), with
    # re-based indices and matching values and dense shapes; the piece counts are asserted.
    for axis in (0, -2):
      sp_tensors_2 = self.evaluate(
          sparse_ops.sparse_split(
              sp_input=self._SparseTensor_5x7(), num_split=2, axis=axis))
      self.assertAllEqual(sp_tensors_2[0].indices,
                          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3],
                           [1, 4], [1, 6], [2, 0], [2, 3], [2, 5]])
      self.assertAllEqual(sp_tensors_2[0].values,
                          [0, 2, 4, 5, 11, 13, 14, 16, 20, 23, 25])
      self.assertAllEqual(sp_tensors_2[0].dense_shape, [3, 7])
      self.assertAllEqual(
          sp_tensors_2[1].indices,
          [[0, 0], [0, 2], [0, 3], [0, 5], [1, 1], [1, 4], [1, 6]])
      self.assertAllEqual(sp_tensors_2[1].values, [30, 32, 33, 35, 41, 44, 46])
      self.assertAllEqual(sp_tensors_2[1].dense_shape, [2, 7])
      self.assertAllEqual(len(sp_tensors_2), 2)
      sp_tensors_3 = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_5x7(), num_split=3, axis=axis)
      self.assertAllEqual(len(sp_tensors_3), 3)
      self.assertAllEqual(
          sp_tensors_3[0].indices,
          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4], [1, 6]])
      self.assertAllEqual(sp_tensors_3[0].values, [0, 2, 4, 5, 11, 13, 14, 16])
      self.assertAllEqual(sp_tensors_3[0].dense_shape, [2, 7])

      self.assertAllEqual(sp_tensors_3[1].values, [20, 23, 25, 30, 32, 33, 35])
      self.assertAllEqual(sp_tensors_3[1].dense_shape, [2, 7])
      self.assertAllEqual(sp_tensors_3[2].indices, [[0, 1], [0, 4], [0, 6]])
      self.assertAllEqual(sp_tensors_3[2].values, [41, 44, 46])
      self.assertAllEqual(sp_tensors_3[2].dense_shape, [1, 7])

  def testSplitAllRows(self):
    # [FUNCTION DOC]
    # Scenario: split the 4x6 matrix into 4 pieces along the row axis (0 and negative two).
    # Expected: four pieces of dense shape 1 by 6, each holding one row's entries with a zero row
    # index; indices, values and shapes are asserted.
    for axis in (0, -2):
      sp_tensors = self.evaluate(
          sparse_ops.sparse_split(
              sp_input=self._SparseTensor_4x6(), num_split=4, axis=axis))
      self.assertAllEqual(len(sp_tensors), 4)
      self.assertAllEqual(sp_tensors[0].indices,
                          [[0, 0], [0, 2], [0, 4], [0, 5]])
      self.assertAllEqual(sp_tensors[0].values, [0, 2, 4, 5])
      self.assertAllEqual(sp_tensors[0].dense_shape, [1, 6])
      self.assertAllEqual(sp_tensors[1].indices, [[0, 1], [0, 3], [0, 4]])
      self.assertAllEqual(sp_tensors[1].values, [11, 13, 14])
      self.assertAllEqual(sp_tensors[1].dense_shape, [1, 6])
      self.assertAllEqual(sp_tensors[2].indices, [[0, 0], [0, 3], [0, 5]])
      self.assertAllEqual(sp_tensors[2].values, [20, 23, 25])
      self.assertAllEqual(sp_tensors[2].dense_shape, [1, 6])
      self.assertAllEqual(sp_tensors[3].indices,
                          [[0, 0], [0, 2], [0, 3], [0, 5]])
      self.assertAllEqual(sp_tensors[3].values, [30, 32, 33, 35])
      self.assertAllEqual(sp_tensors[3].dense_shape, [1, 6])

  def testSplitColumns(self):
    # [FUNCTION DOC]
    # Scenario: split the 4x6 matrix into 3 pieces along the column axis (1 and negative one).
    # Expected: three pieces of dense shape 4 by 2 with column indices re-based per piece and the
    # corresponding values.
    for axis in (1, -1):
      sparse_tensors = self.evaluate(
          sparse_ops.sparse_split(
              sp_input=self._SparseTensor_4x6(), num_split=3, axis=axis))
      self.assertAllEqual(len(sparse_tensors), 3)
      self.assertAllEqual(sparse_tensors[0].indices,
                          [[0, 0], [1, 1], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensors[0].values, [0, 11, 20, 30])
      self.assertAllEqual(sparse_tensors[0].dense_shape, [4, 2])
      self.assertAllEqual(sparse_tensors[1].indices,
                          [[0, 0], [1, 1], [2, 1], [3, 0], [3, 1]])
      self.assertAllEqual(sparse_tensors[1].values, [2, 13, 23, 32, 33])
      self.assertAllEqual(sparse_tensors[1].dense_shape, [4, 2])
      self.assertAllEqual(sparse_tensors[2].indices,
                          [[0, 0], [0, 1], [1, 0], [2, 1], [3, 1]])
      self.assertAllEqual(sparse_tensors[2].values, [4, 5, 14, 25, 35])
      self.assertAllEqual(sparse_tensors[2].dense_shape, [4, 2])

  def testSplitAllColumns(self):
    # [FUNCTION DOC]
    # Scenario: split the 4x6 matrix into 6 pieces along the column axis (1 and negative one).
    # Expected: six pieces of dense shape 4 by 1, each holding the entries of one column with a
    # zero column index; indices, values and shapes are asserted.
    for axis in (1, -1):
      sparse_tensors = self.evaluate(
          sparse_ops.sparse_split(
              sp_input=self._SparseTensor_4x6(), num_split=6, axis=axis))
      self.assertAllEqual(len(sparse_tensors), 6)
      self.assertAllEqual(sparse_tensors[0].indices, [[0, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensors[0].values, [0, 20, 30])
      self.assertAllEqual(sparse_tensors[0].dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensors[1].indices, [[1, 0]])
      self.assertAllEqual(sparse_tensors[1].values, [11])
      self.assertAllEqual(sparse_tensors[1].dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensors[2].indices, [[0, 0], [3, 0]])
      self.assertAllEqual(sparse_tensors[2].values, [2, 32])
      self.assertAllEqual(sparse_tensors[2].dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensors[3].indices, [[1, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensors[3].dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensors[3].values, [13, 23, 33])
      self.assertAllEqual(sparse_tensors[4].indices, [[0, 0], [1, 0]])
      self.assertAllEqual(sparse_tensors[4].values, [4, 14])
      self.assertAllEqual(sparse_tensors[4].dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensors[5].indices, [[0, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensors[5].values, [5, 25, 35])
      self.assertAllEqual(sparse_tensors[5].dense_shape, [4, 1])

  def testSliceConcat(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says slice, but the test splits; it also only compares indices of the
    # re-concatenated tensor, not the values or dense shape.
    # Scenario: for the 3x4x2 string fixture given both as a raw value and as a tensor, split into
    # 2 pieces along axis 1 and negative two, concatenate the pieces along dimension 1, evaluate.
    # Expected: the concatenated indices equal the indices of the original fixture.
    for sp_input in (self._SparseTensorValue_3x4x2(),
                     self._SparseTensor_3x4x2()):
      for axis in (1, -2):
        sparse_tensors = sparse_ops.sparse_split(
            sp_input=sp_input, num_split=2, axis=axis)
        concat_tensor = self.evaluate(
            sparse_ops.sparse_concat(1, sparse_tensors))
        expected_output = self._SparseTensor_3x4x2()
        self.assertAllEqual(concat_tensor.indices, expected_output.indices)

  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Scenario: split the 4x6 matrix with axis values minus three and two, both outside the valid
    # range for a two-dimensional tensor.
    # Expected: an invalid-argument error whose message says the axis should be in the range from
    # minus two (inclusive) to two (exclusive).
    for axis in (-3, 2):
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                   r'axis should be in range \[-2, 2\)'):
        self.evaluate(
            sparse_ops.sparse_split(
                sp_input=self._SparseTensor_4x6(), num_split=3, axis=axis))

  def testArgumentErrors(self):
    # [FUNCTION DOC]
    # Scenario: call the split function with positional arguments, with no arguments, and with
    # the input, split count or axis missing.
    # Expected: value errors saying keyword arguments are required, and that the input, the split
    # count and the axis, respectively, are required.
    with self.assertRaisesRegex(ValueError, 'Keyword arguments are required'):
      sparse_ops.sparse_split(3, 2, 1)
    with self.assertRaisesRegex(ValueError, 'sp_input is required'):
      sparse_ops.sparse_split()
    with self.assertRaisesRegex(ValueError, 'num_split is required'):
      sparse_ops.sparse_split(sp_input=1)
    with self.assertRaisesRegex(ValueError, 'axis is required'):
      sparse_ops.sparse_split(num_split=2, sp_input=1)

  def testSplitEmpty(self):
    # [FUNCTION DOC]
    # Scenario: split an empty 4x6 sparse tensor into 2 pieces along rows and along columns.
    # Expected: every piece has no indices and no values, with dense shape 2 by 6 for the row
    # split and 4 by 3 for the column split.
    sp_empty = self._SparseTensor_4x6_empty()
    sparse_splits0 = sparse_ops.sparse_split(
        sp_input=sp_empty, num_split=2, axis=0)
    sparse_splits1 = sparse_ops.sparse_split(
        sp_input=sp_empty, num_split=2, axis=1)
    empty_inds = np.empty(shape=(0, 2), dtype=np.int64)
    self.assertAllEqual(sparse_splits0[0].indices, empty_inds)
    self.assertAllEqual(sparse_splits0[0].values, [])
    self.assertAllEqual(sparse_splits0[0].dense_shape, [2, 6])
    self.assertAllEqual(sparse_splits0[1].indices, empty_inds)
    self.assertAllEqual(sparse_splits0[1].values, [])
    self.assertAllEqual(sparse_splits0[1].dense_shape, [2, 6])
    self.assertAllEqual(sparse_splits1[0].indices, empty_inds)
    self.assertAllEqual(sparse_splits1[0].values, [])
    self.assertAllEqual(sparse_splits1[0].dense_shape, [4, 3])
    self.assertAllEqual(sparse_splits1[1].indices, empty_inds)
    self.assertAllEqual(sparse_splits1[1].values, [])
    self.assertAllEqual(sparse_splits1[1].dense_shape, [4, 3])

  def testInvalidArgumentError(self):
    # [FUNCTION DOC]
    # Scenario: regression test for a reported GitHub issue (53660); split with an axis given as a
    # two-element list instead of a scalar.
    # Expected: an invalid-argument error saying the axis should be a scalar.
    # Test case for GitHub issue 53660.
    axis = [1, 2]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                 r'axis should be a scalar'):
      self.evaluate(
          sparse_ops.sparse_split(
              sp_input=self._SparseTensor_4x6(), num_split=3, axis=axis))

  def testBig(self):
    # [FUNCTION DOC]
    # Scenario: regression test for a GPU kernel memory allocation bug missed by small inputs.
    # For 250, 2500 and 25000 rows, build a sparse float tensor with one zero-valued entry per row
    # (column zero) and dense shape rows by 3, split it in two along rows, and evaluate.
    # Expected: the first piece equals the first half of the indices and values, the second piece
    # has the remaining indices shifted down by half the row count, and both have half the rows.
    # This test was added after discovering a memory allocation bug in the GPU
    # kernel that the existing tests did not catch due to being too small.
    for n in [250, 2500, 25000]:
      indices = np.zeros([n, 2], dtype=np.int64)
      indices[:, 0] = np.arange(n)
      values = np.zeros([n], dtype=np.float32)
      dense_shape = np.array([n, 3], dtype=np.int64)
      sp_input = sparse_tensor.SparseTensor(indices, values, dense_shape)
      sp_tensors = self.evaluate(
          sparse_ops.sparse_split(sp_input=sp_input, num_split=2, axis=0))
      self.assertAllEqual(sp_tensors[0].indices, indices[:n // 2])
      self.assertAllEqual(sp_tensors[1].indices, indices[n // 2:] - [n // 2, 0])
      self.assertAllEqual(sp_tensors[0].values, values[:n // 2])
      self.assertAllEqual(sp_tensors[1].values, values[n // 2:])
      self.assertAllEqual(sp_tensors[0].dense_shape, [n // 2, 3])
      self.assertAllEqual(sp_tensors[1].dense_shape, [n // 2, 3])

  def testSparseIndexOutOfBounds(self):
    # [FUNCTION DOC]
    # Scenario: call the raw split op on a 3x5 tensor whose indices include an out-of-range
    # column and a negative row, splitting in two along dimension 1.
    # Expected: on GPU the invalid entries are silently ignored and evaluation succeeds; on CPU a
    # value error or invalid-argument error mentioning out of bounds is raised.
    if test_util.is_gpu_available():
      # On GPU, out-of-bounds indices are simply ignored.
      self.evaluate(
          gen_sparse_ops.sparse_split(
              split_dim=1,
              indices=[[0, 0], [1, 10], [-1, 2]],
              values=[1.0, 2.0, 3.0],
              shape=[3, 5],
              num_split=2,
          )
      )
    else:
      # On CPU, out-of-bounds index raises error.
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), 'out of bounds'
      ):
        self.evaluate(
            gen_sparse_ops.sparse_split(
                split_dim=1,
                indices=[[0, 0], [1, 10], [-1, 2]],
                values=[1.0, 2.0, 3.0],
                shape=[3, 5],
                num_split=2,
            )
        )


if __name__ == '__main__':
  test.main()
