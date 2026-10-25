# [MODULE DOC]
# KNOWN ISSUE: the module docstring says the tests are for the reorder op, but this file actually
# tests the sparse slice op and its gradient (a copy-paste leftover from the reorder tests).
# Unit tests for the sparse slice operation and its gradient. They build small sparse tensors
# (a 4x6 and a 5x7 integer matrix, a 3x4x2 string tensor, and an empty 4x6 tensor), slice them by
# rows, columns, uneven partitions and empty windows, and compare the resulting indices, values
# and dense shape with expected values. Also covers numeric gradient checks, explicit slice
# gradient values, and error handling for negative sizes, huge sizes and inconsistent inputs.
# Importing the module registers the sparse gradient functions; running it as a script starts the
# TensorFlow test runner.
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
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SparseSliceOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the sparse slice op, its gradient and its error handling. It holds no
  # state; helper methods build the shared sparse tensor fixtures (4x6, 5x7, 3x4x2, empty 4x6)
  # on demand, and the test methods slice them and assert on indices, values and dense shapes.

  def _SparseTensor_4x6(self, val_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: builds the 4x6 sparse tensor fixture shared by the slice tests.
    # Input: val_dtype, the numpy element type of the values (default 64-bit integer).
    # Returns a sparse tensor with 14 stored entries in a 4 row by 6 column dense shape; the
    # value at each position encodes its coordinates (row times ten plus column pattern, e.g.
    # 11 at row 1 column 1). Indices are 64-bit. No side effects.
    # [0 |  |2 |  |4 |5 ]
    # [  |11|  |13|14|  ]
    # [20|  |  |23|  |25]
    # [30|  |32|33|  |35]
    ind = np.array([[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1,
                                                                     4], [2, 0],
                    [2, 3], [2, 5], [3, 0], [3, 2], [3, 3], [3, 5]]).astype(
                        np.int64)
    val = np.array([0, 2, 4, 5, 11, 13, 14, 20, 23, 25, 30, 32, 33, 35]).astype(
        val_dtype)
    shape = np.array([4, 6]).astype(np.int64)
    return sparse_tensor.SparseTensor(ind, val, shape)

  def _SparseTensor_5x7(self):
    # [FUNCTION DOC]
    # HELPER: builds the 5x7 integer sparse tensor fixture used by the uneven slice tests.
    # Takes no arguments. Returns a sparse tensor with 18 stored 64-bit integer entries and
    # dense shape 5 rows by 7 columns (values encode their coordinates, e.g. 46 at row 4,
    # column 6). No side effects.
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
    # HELPER: builds the raw (non-graph) 3x4x2 string sparse value fixture.
    # Takes no arguments. Returns a sparse tensor value (plain indices, values, dense shape
    # triple of numpy arrays) with 10 string entries such as a0, a1, b0 spread over a dense
    # shape of 3 by 4 by 2. No side effects.
    #  slice(:,:, 0)
    #  ['a0'|    |'b0'|    ]
    #  [    |'c0'|    |'d0']
    #  [    |    |'e0'|    ]
    #  slice(:,:, 1)
    #  ['a1'|    |'b1'|    ]
    #  [    |'c1'|    |'d1']
    #  [    |    |'e1'|    ]
    ind = np.array([[0, 0, 0], [0, 0, 1], [0, 2, 0], [0, 2, 1], [1, 1, 0],
                    [1, 1, 1], [1, 3, 0], [1, 3, 1], [2, 2, 0], [2, 2,
                                                                 1]]).astype(
                                                                     np.int64)
    val = np.array(['a0', 'a1', 'b0', 'b1', 'c0', 'c1', 'd0', 'd1', 'e0', 'e1'])
    shape = np.array([3, 4, 2]).astype(np.int64)
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def _SparseTensor_3x4x2(self):
    # [FUNCTION DOC]
    # HELPER: wraps the 3x4x2 string sparse value fixture into a sparse tensor.
    # Takes no arguments; returns the sparse tensor created from the raw value fixture. Note that
    # no test in this file calls it. No side effects.
    return sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_3x4x2())

  def _SparseTensor_4x6_empty(self, val_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: builds a sparse tensor with dense shape 4 by 6 and no stored entries.
    # Input: val_dtype, numpy element type of the (empty) values array, default 64-bit integer.
    # Returns the empty sparse tensor (zero rows of indices). No side effects.
    ind = np.empty(shape=(0, 2), dtype=np.int64)
    val = np.array([]).astype(val_dtype)
    shape = np.array([4, 6]).astype(np.int64)
    return sparse_tensor.SparseTensor(ind, val, shape)

  @test_util.run_deprecated_v1
  def testSliceMatrixRows(self):
    # [FUNCTION DOC]
    # Scenario: slice the 4x6 matrix into two row blocks (rows 0-1 with a window of 2 rows, and
    # rows 2-3 with a window larger than what remains). Runs in a graph-mode session.
    # Expected: each result keeps only the entries inside its window, re-based to the window
    # origin, with dense shape 2 by 6 for both; indices, values and shape are asserted exactly.
    with self.session():
      sp_input = self._SparseTensor_4x6()
      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [2, 6])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [2, 0], [3, 7])
      self.assertAllEqual(
          sp_tensor0.indices,
          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4]])
      self.assertAllEqual(sp_tensor0.values, [0, 2, 4, 5, 11, 13, 14])
      self.assertAllEqual(sp_tensor0.dense_shape, [2, 6])
      self.assertAllEqual(
          sp_tensor1.indices,
          [[0, 0], [0, 3], [0, 5], [1, 0], [1, 2], [1, 3], [1, 5]])
      self.assertAllEqual(sp_tensor1.values, [20, 23, 25, 30, 32, 33, 35])
      self.assertAllEqual(sp_tensor1.dense_shape, [2, 6])

  @test_util.run_deprecated_v1
  def testSliceMatrixUnevenCols(self):
    # [FUNCTION DOC]
    # Scenario: partition the 5x7 matrix by columns using windows that do not divide 7 evenly
    # (widths 3, 2, 2, and then widths of 2 starting at columns 0, 2, 4, 6). Graph-mode session.
    # Expected: indices are re-based to each window, values match the entries in range, and the
    # last window is truncated (dense shape 5 by 1 when only one column remains).
    with self.session():
      sp_input = self._SparseTensor_5x7()
      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [5, 3])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [0, 3], [5, 2])
      sp_tensor2 = sparse_ops.sparse_slice(sp_input, [0, 5], [5, 2])

      self.assertAllEqual(
          sp_tensor0.indices,
          [[0, 0], [0, 2], [1, 1], [2, 0], [3, 0], [3, 2], [4, 1]])
      self.assertAllEqual(sp_tensor0.values, [0, 2, 11, 20, 30, 32, 41])
      self.assertAllEqual(sp_tensor0.dense_shape, [5, 3])
      self.assertAllEqual(sp_tensor1.indices,
                          [[0, 1], [1, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensor1.values, [4, 13, 14, 23, 33, 44])
      self.assertAllEqual(sp_tensor1.dense_shape, [5, 2])
      self.assertAllEqual(sp_tensor2.indices,
                          [[0, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensor2.values, [5, 16, 25, 35, 46])
      self.assertAllEqual(sp_tensor2.dense_shape, [5, 2])

      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [5, 2])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [0, 2], [5, 2])
      sp_tensor2 = sparse_ops.sparse_slice(sp_input, [0, 4], [5, 2])
      sp_tensor3 = sparse_ops.sparse_slice(sp_input, [0, 6], [5, 2])
      self.assertAllEqual(sp_tensor0.indices,
                          [[0, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensor0.values, [0, 11, 20, 30, 41])
      self.assertAllEqual(sp_tensor0.dense_shape, [5, 2])
      self.assertAllEqual(sp_tensor1.indices,
                          [[0, 0], [1, 1], [2, 1], [3, 0], [3, 1]])
      self.assertAllEqual(sp_tensor1.values, [2, 13, 23, 32, 33])
      self.assertAllEqual(sp_tensor1.dense_shape, [5, 2])
      self.assertAllEqual(sp_tensor2.indices,
                          [[0, 0], [0, 1], [1, 0], [2, 1], [3, 1], [4, 0]])
      self.assertAllEqual(sp_tensor2.values, [4, 5, 14, 25, 35, 44])
      self.assertAllEqual(sp_tensor2.dense_shape, [5, 2])
      self.assertAllEqual(sp_tensor3.indices, [[1, 0], [4, 0]])
      self.assertAllEqual(sp_tensor3.values, [16, 46])
      self.assertAllEqual(sp_tensor3.dense_shape, [5, 1])

  @test_util.run_deprecated_v1
  def testSliceMatrixUnevenRows(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method ends with a redundant bare return statement; it has no effect.
    # Scenario: partition the 5x7 matrix by rows with windows of 3 rows, then 2 rows starting at
    # rows 0, 2 and 4. Graph-mode session.
    # Expected: each slice holds the entries of its rows with row indices re-based, and the last
    # slice is truncated (dense shape 2 by 7, or 1 by 7). Not every slice has its indices asserted.
    with self.session():
      sp_input = self._SparseTensor_5x7()
      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [3, 7])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [3, 0], [3, 7])
      self.assertAllEqual(sp_tensor0.indices,
                          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3],
                           [1, 4], [1, 6], [2, 0], [2, 3], [2, 5]])
      self.assertAllEqual(sp_tensor0.values,
                          [0, 2, 4, 5, 11, 13, 14, 16, 20, 23, 25])
      self.assertAllEqual(sp_tensor0.dense_shape, [3, 7])
      self.assertAllEqual(
          sp_tensor1.indices,
          [[0, 0], [0, 2], [0, 3], [0, 5], [1, 1], [1, 4], [1, 6]])
      self.assertAllEqual(sp_tensor1.values, [30, 32, 33, 35, 41, 44, 46])
      self.assertAllEqual(sp_tensor1.dense_shape, [2, 7])

      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [2, 7])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [2, 0], [2, 7])
      sp_tensor2 = sparse_ops.sparse_slice(sp_input, [4, 0], [2, 7])
      self.assertAllEqual(
          sp_tensor0.indices,
          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4], [1, 6]])
      self.assertAllEqual(sp_tensor0.values, [0, 2, 4, 5, 11, 13, 14, 16])
      self.assertAllEqual(sp_tensor0.dense_shape, [2, 7])

      self.assertAllEqual(sp_tensor1.values, [20, 23, 25, 30, 32, 33, 35])
      self.assertAllEqual(sp_tensor1.dense_shape, [2, 7])
      self.assertAllEqual(sp_tensor2.indices, [[0, 1], [0, 4], [0, 6]])
      self.assertAllEqual(sp_tensor2.values, [41, 44, 46])
      self.assertAllEqual(sp_tensor2.dense_shape, [1, 7])
    return

  @test_util.run_deprecated_v1
  def testSliceAllRows(self):
    # [FUNCTION DOC]
    # Scenario: slice the 4x6 matrix one row at a time (windows of one row; the last two use
    # oversized windows of width 7 or height 2 past the end). Graph-mode session.
    # Expected: each slice has dense shape 1 by 6 and contains exactly the stored entries of that
    # row, with the row index re-based to zero; indices, values and shape are asserted.
    with self.session():
      sp_input = self._SparseTensor_4x6()
      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [1, 6])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [1, 0], [1, 6])
      sp_tensor2 = sparse_ops.sparse_slice(sp_input, [2, 0], [1, 7])
      sp_tensor3 = sparse_ops.sparse_slice(sp_input, [3, 0], [2, 7])
      self.assertAllEqual(sp_tensor0.indices, [[0, 0], [0, 2], [0, 4], [0, 5]])
      self.assertAllEqual(sp_tensor0.values, [0, 2, 4, 5])
      self.assertAllEqual(sp_tensor0.dense_shape, [1, 6])
      self.assertAllEqual(sp_tensor1.indices, [[0, 1], [0, 3], [0, 4]])
      self.assertAllEqual(sp_tensor1.values, [11, 13, 14])
      self.assertAllEqual(sp_tensor1.dense_shape, [1, 6])
      self.assertAllEqual(sp_tensor2.indices, [[0, 0], [0, 3], [0, 5]])
      self.assertAllEqual(sp_tensor2.values, [20, 23, 25])
      self.assertAllEqual(sp_tensor2.dense_shape, [1, 6])
      self.assertAllEqual(sp_tensor3.indices, [[0, 0], [0, 2], [0, 3], [0, 5]])
      self.assertAllEqual(sp_tensor3.values, [30, 32, 33, 35])
      self.assertAllEqual(sp_tensor3.dense_shape, [1, 6])

  @test_util.run_deprecated_v1
  def testSliceColumns(self):
    # [FUNCTION DOC]
    # Scenario: slice the 4x6 matrix into three column blocks starting at columns 0, 2 and 4,
    # with widths 2, 2 and 3 (the last window runs past the end, and row sizes may exceed 4).
    # Graph-mode session.
    # Expected: results keep all four rows (dense shape 4 by 2), with column indices re-based and
    # the entries of each column block as values.
    with self.session():
      sp_input = self._SparseTensor_4x6()
      sparse_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [4, 2])
      sparse_tensor1 = sparse_ops.sparse_slice(sp_input, [0, 2], [5, 2])
      sparse_tensor2 = sparse_ops.sparse_slice(sp_input, [0, 4], [5, 3])

      self.assertAllEqual(sparse_tensor0.indices,
                          [[0, 0], [1, 1], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor0.values, [0, 11, 20, 30])
      self.assertAllEqual(sparse_tensor0.dense_shape, [4, 2])
      self.assertAllEqual(sparse_tensor1.indices,
                          [[0, 0], [1, 1], [2, 1], [3, 0], [3, 1]])
      self.assertAllEqual(sparse_tensor1.values, [2, 13, 23, 32, 33])
      self.assertAllEqual(sparse_tensor1.dense_shape, [4, 2])
      self.assertAllEqual(sparse_tensor2.indices,
                          [[0, 0], [0, 1], [1, 0], [2, 1], [3, 1]])
      self.assertAllEqual(sparse_tensor2.values, [4, 5, 14, 25, 35])
      self.assertAllEqual(sparse_tensor2.dense_shape, [4, 2])

  @test_util.run_deprecated_v1
  def testSliceAllColumns(self):
    # [FUNCTION DOC]
    # Scenario: slice the 4x6 matrix one column at a time, six single-column windows (some
    # declared with oversized row extents or a width of 3 at the last column). Graph-mode session.
    # Expected: every slice has dense shape 4 by 1 and holds the entries of its column, with
    # indices of the form row, zero; indices, values and dense shape are asserted for all six.
    with self.session():
      sp_input = self._SparseTensor_4x6()
      sparse_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [4, 1])
      sparse_tensor1 = sparse_ops.sparse_slice(sp_input, [0, 1], [4, 1])
      sparse_tensor2 = sparse_ops.sparse_slice(sp_input, [0, 2], [4, 1])
      sparse_tensor3 = sparse_ops.sparse_slice(sp_input, [0, 3], [4, 1])
      sparse_tensor4 = sparse_ops.sparse_slice(sp_input, [0, 4], [5, 1])
      sparse_tensor5 = sparse_ops.sparse_slice(sp_input, [0, 5], [6, 3])
      self.assertAllEqual(sparse_tensor0.indices, [[0, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor0.values, [0, 20, 30])
      self.assertAllEqual(sparse_tensor0.dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensor1.indices, [[1, 0]])
      self.assertAllEqual(sparse_tensor1.values, [11])
      self.assertAllEqual(sparse_tensor1.dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensor2.indices, [[0, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor2.values, [2, 32])
      self.assertAllEqual(sparse_tensor2.dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensor3.indices, [[1, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor3.dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensor3.values, [13, 23, 33])
      self.assertAllEqual(sparse_tensor4.indices, [[0, 0], [1, 0]])
      self.assertAllEqual(sparse_tensor4.values, [4, 14])
      self.assertAllEqual(sparse_tensor4.dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensor5.indices, [[0, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor5.values, [5, 25, 35])
      self.assertAllEqual(sparse_tensor5.dense_shape, [4, 1])

  def testSliceEmpty(self):
    # [FUNCTION DOC]
    # Scenario: runs on GPU if available. Slices an empty sparse tensor, slices a non-empty tensor
    # with a zero-sized window, and slices a window (rows 2-3, column 1) that contains no entries.
    # Expected: in every case the result has zero index rows and no values, and the dense shape
    # equals the requested size clipped to the input (4 by 1, 0 by 0, and 2 by 1 respectively).
    with test_util.use_gpu():
      sp_empty = self._SparseTensor_4x6_empty()
      sp_input = self._SparseTensor_4x6()
      sparse_tensor0 = sparse_ops.sparse_slice(sp_empty, [0, 0], [4, 1])
      sparse_tensor1 = sparse_ops.sparse_slice(sp_input, [1, 1], [0, 0])
      sparse_tensor2 = sparse_ops.sparse_slice(sp_input, [2, 1], [2, 1])
      empty_inds = np.empty(shape=(0, 2), dtype=np.int64)
      self.assertAllEqual(sparse_tensor0.indices, empty_inds)
      self.assertAllEqual(sparse_tensor0.values, [])
      self.assertAllEqual(sparse_tensor0.dense_shape, [4, 1])
      self.assertAllEqual(sparse_tensor1.indices, empty_inds)
      self.assertAllEqual(sparse_tensor1.values, [])
      self.assertAllEqual(sparse_tensor1.dense_shape, [0, 0])
      self.assertAllEqual(sparse_tensor2.indices, empty_inds)
      self.assertAllEqual(sparse_tensor2.values, [])
      self.assertAllEqual(sparse_tensor2.dense_shape, [2, 1])

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Scenario: build the 4x6 tensor with float values and take three slices (column windows
    # starting at 0, 2 and 4). For each, count the stored entries of input and output and run the
    # numeric gradient checker on the values of the input with respect to the output values.
    # Expected: the maximum theoretical versus numeric gradient error is below 0.001.
    sp_input = self._SparseTensor_4x6(val_dtype=np.float32)
    start_and_size = [([0, 0], [4, 2]),
                      ([0, 2], [5, 2]),
                      ([0, 4], [5, 3])]

    with self.session():
      for start, size in start_and_size:
        sp_output = sparse_ops.sparse_slice(sp_input, start, size)
        nnz_in = len(self.evaluate(sp_input.values))
        nnz_out = len(self.evaluate(sp_output.values))

        err = gradient_checker.compute_gradient_error(
            [sp_input.values], [(nnz_in,)], sp_output.values, (nnz_out,))
        self.assertLess(err, 1e-3)

  def testGradientsExplicit(self):
    # [FUNCTION DOC]
    # Scenario: for several slice windows of the 4x6 tensor (first column, second column, a 3x1
    # window at row 1 column 3, an empty window, and a slice of an empty input), feed the sliced
    # values themselves as the incoming gradient to the slice gradient op together with the input
    # indices, the window start and the output indices.
    # Expected: the gradient is a flat list with one entry per stored input value, equal to the
    # input value where the entry lies inside the window and zero elsewhere; empty for empty input.
    sp_input = self._SparseTensor_4x6()
    start, size = [0, 0], [4, 1]
    sp_output = sparse_ops.sparse_slice(sp_input, start, size)
    input_grad_vals = sparse_ops.sparse_slice_grad(sp_output.values,
                                                   sp_input.indices, start,
                                                   sp_output.indices)
    # pyformat: disable
    self.assertAllEqual(input_grad_vals, [0, 0, 0, 0,
                                          0, 0, 0,
                                          20, 0, 0,
                                          30, 0, 0, 0])
    # pyformat: enable

    start, size = [0, 1], [4, 1]
    sp_output = sparse_ops.sparse_slice(sp_input, start, size)
    input_grad_vals = sparse_ops.sparse_slice_grad(sp_output.values,
                                                   sp_input.indices, start,
                                                   sp_output.indices)
    # pyformat: disable
    self.assertAllEqual(input_grad_vals, [0, 0, 0, 0,
                                          11, 0, 0,
                                          0, 0, 0,
                                          0, 0, 0, 0])
    # pyformat: enable

    start, size = [1, 3], [3, 1]
    sp_output = sparse_ops.sparse_slice(sp_input, start, size)
    input_grad_vals = sparse_ops.sparse_slice_grad(sp_output.values,
                                                   sp_input.indices, start,
                                                   sp_output.indices)
    # pyformat: disable
    self.assertAllEqual(input_grad_vals, [0, 0, 0, 0,
                                          0, 13, 0,
                                          0, 23, 0,
                                          0, 0, 33, 0])
    # pyformat: enable

    # Test empty slice of non-empty input.
    start, size = [2, 1], [2, 1]
    sp_output = sparse_ops.sparse_slice(sp_input, start, size)
    input_grad_vals = sparse_ops.sparse_slice_grad(sp_output.values,
                                                   sp_input.indices, start,
                                                   sp_output.indices)
    # pyformat: disable
    self.assertAllEqual(input_grad_vals, [0, 0, 0, 0,
                                          0, 0, 0,
                                          0, 0, 0,
                                          0, 0, 0, 0])
    # pyformat: enable

    sp_input = self._SparseTensor_4x6_empty()
    start, size = [0, 0], [4, 1]
    sp_output = sparse_ops.sparse_slice(sp_input, start, size)
    input_grad_vals = sparse_ops.sparse_slice_grad(sp_output.values,
                                                   sp_input.indices, start,
                                                   sp_output.indices)
    self.assertAllEqual(input_grad_vals, [])

  def testNegativeSize(self):
    # [FUNCTION DOC]
    # Scenario: call the raw slice op on CPU with a single entry, a start of ten by ten and a
    # negative size in the first dimension, then evaluate.
    # Expected: an invalid-argument error is raised.
    with self.session(use_gpu=False):
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_slice(
            indices=[[0, 0]],
            values=[0],
            shape=[1, 1],
            start=[10, 10],
            size=[-100, 100])
        self.evaluate(res)

  def testLargeSize(self):
    # [FUNCTION DOC]
    # Scenario: call the raw slice op on CPU with a start near two to the 62 (and minus one in the
    # other dimension) and sizes of two to the 62, which could overflow when start plus size is
    # computed, then evaluate.
    # Expected: the op completes without crashing or raising; no output is asserted.
    with self.session(use_gpu=False):
      # Confirm potential integer overflow due to size is handled by op.
      res = sparse_ops.gen_sparse_ops.sparse_slice(
          indices=[[0, 0]],
          values=[0],
          shape=[1, 1],
          start=[2**62, -1],
          size=[2**62, 2**62])
      self.evaluate(res)

  def testInvalidSparseInput(self):
    # [FUNCTION DOC]
    # Scenario: call the raw slice op with one three-dimensional index, three values and a
    # two-dimensional dense shape (inconsistent sizes).
    # Expected: a value error or invalid-argument error whose message says the number of elements
    # does not match.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        'Number of elements .* do not match',
    ):
      self.evaluate(
          gen_sparse_ops.sparse_slice(
              indices=[[0, 0, 0]],
              values=[0, 1, 2],
              shape=[3, 3],
              start=[0, 0],
              size=[1, 1],
          )
      )


if __name__ == '__main__':
  test.main()
