# [MODULE DOC]
# Unit tests for sparse_ops.sparse_concat. Covers concatenation of one to three sparse tensors
# along various (including negative) dimensions, tensors with no non-zero entries, string
# values, rank and shape mismatches (static and runtime errors), the expand-non-concat-dimension
# option, static shape inference with unknown input shapes, and a regression case for the
# static output shape. No import-time side effects beyond the test runner entry point.
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
"""Tests for SparseConcat."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SparseConcatTest(test.TestCase):
  # [CLASS DOC]
  # Test case for sparse_ops.sparse_concat. Supplies a family of fixture builders producing small
  # sparse tensors (3x3, 3x5, 3x2, 2x3, 2x3x4, empty, string-valued, unknown-shape) that the
  # tests concatenate and check.

  def _SparseTensor_UnknownShape(self,
                                 ind_shape=None,
                                 val_shape=None,
                                 shape_shape=None):
    # [FUNCTION DOC]
    # HELPER: fixture. Builds a sparse tensor from three placeholders (64-bit integer indices,
    # float values, 64-bit integer dense shape) whose static shapes are given by the optional
    # arguments (each defaults to fully unknown). Graph-mode only because of placeholders.
    return sparse_tensor.SparseTensor(
        array_ops.placeholder(
            dtypes.int64, shape=ind_shape),
        array_ops.placeholder(
            dtypes.float32, shape=val_shape),
        array_ops.placeholder(
            dtypes.int64, shape=shape_shape))

  def _SparseTensorValue_3x3(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a NumPy-backed SparseTensorValue of dense shape 3x3 with values
    # 1, 2, 3, 4 at positions (0,2), (1,0), (2,0), (2,2).
    # [    1]
    # [2    ]
    # [3   4]
    ind = np.array([[0, 2], [1, 0], [2, 0], [2, 2]])
    val = np.array([1, 2, 3, 4])
    shape = np.array([3, 3])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.float32), np.array(shape, np.int64))

  def _SparseTensor_3x3(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns the 3x3 sparse value fixture converted to a SparseTensor.
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x3())

  def _SparseTensorValue_3x5(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a NumPy-backed SparseTensorValue of dense shape 3x5 with values
    # 1, 2, 1, 0 at (1,1), (2,0), (2,3), (2,4); note one stored value is an explicit zero.
    # [         ]
    # [  1      ]
    # [2     1 0]
    ind = np.array([[1, 1], [2, 0], [2, 3], [2, 4]])
    val = np.array([1, 2, 1, 0])
    shape = np.array([3, 5])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.float32), np.array(shape, np.int64))

  def _SparseTensor_3x5(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns the 3x5 sparse value fixture converted to a SparseTensor.
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x5())

  def _SparseTensor_3x2(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a constant-backed SparseTensor of dense shape 3x2 with values 1
    # and 2 at (1,0) and (2,0).
    # [   ]
    # [1  ]
    # [2  ]
    ind = np.array([[1, 0], [2, 0]])
    val = np.array([1, 2])
    shape = np.array([3, 2])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x3(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a constant-backed SparseTensor of dense shape 2x3 with values
    # 1, 1, 2 at (0,1), (1,0), (1,2).
    # [  1  ]
    # [1   2]
    ind = np.array([[0, 1], [1, 0], [1, 2]])
    val = np.array([1, 1, 2])
    shape = np.array([2, 3])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x3x4(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a constant-backed rank-3 SparseTensor of dense shape 2x3x4 with
    # seven entries whose values encode their coordinates (for example 122 at (1,2,2)).
    ind = np.array([
        [0, 0, 1],
        [0, 1, 0], [0, 1, 2],
        [1, 0, 3],
        [1, 1, 1], [1, 1, 3],
        [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 111, 113, 122])
    shape = np.array([2, 3, 4])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_NoNonZeros(self, dense_shape):
    # [FUNCTION DOC]
    # HELPER: fixture. Takes a dense shape (sequence of integers) and returns a constant-backed
    # float SparseTensor of that shape with zero stored entries (empty indices and values).
    ind = np.empty(shape=(0, len(dense_shape)))
    val = np.array([])
    shape = np.array(dense_shape)
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_String3x3(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a 3x3 SparseTensor with string values a, b, c, d at (0,2), (1,0),
    # (2,0), (2,2).
    # [    a]
    # [b    ]
    # [c   d]
    ind = np.array([[0, 2], [1, 0], [2, 0], [2, 2]])
    val = np.array(["a", "b", "c", "d"])
    shape = np.array([3, 3])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.string),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_String3x5(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a 3x5 SparseTensor with string values e, f, g, h at (1,1), (2,0),
    # (2,3), (2,4).
    # [         ]
    # [  e      ]
    # [f     g h]
    ind = np.array([[1, 1], [2, 0], [2, 3], [2, 4]])
    val = np.array(["e", "f", "g", "h"])
    shape = np.array([3, 5])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.string),
        constant_op.constant(shape, dtypes.int64))

  def testConcat1(self):
    # [FUNCTION DOC]
    # Concatenates a single sparse input (as value and as tensor) with concat dimensions far out
    # of range and in range; the single-input case is short-circuited so the dimension is
    # ignored. Asserts the output equals the input with static shapes 4x2, 4 and 2. Returns
    # nothing.
    with self.session() as sess:
      # concat(A):
      # [    1]
      # [2    ]
      # [3   4]
      for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
        # Note that we ignore concat_dim in this case since we short-circuit the
        # single-input case in python.
        for concat_dim in (-2000, 1, 2000):
          sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a])

          self.assertEqual(sp_concat.indices.get_shape(), [4, 2])
          self.assertEqual(sp_concat.values.get_shape(), [4])
          self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

          concat_out = self.evaluate(sp_concat)

          self.assertAllEqual(concat_out.indices,
                              [[0, 2], [1, 0], [2, 0], [2, 2]])
          self.assertAllEqual(concat_out.values, [1, 2, 3, 4])
          self.assertAllEqual(concat_out.dense_shape, [3, 3])

  def testConcat2(self):
    # [FUNCTION DOC]
    # Concatenates 3x3 and 3x5 inputs along the last axis (given as -1 and 1), for both value
    # and tensor forms. Asserts eight entries with column indices of the second input shifted by
    # three, merged values and dense shape 3x8. Returns nothing.
    with self.session() as sess:
      # concat(A, B):
      # [    1          ]
      # [2       1      ]
      # [3   4 2     1 0]
      for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
        for sp_b in (self._SparseTensorValue_3x5(), self._SparseTensor_3x5()):
          for concat_dim in (-1, 1):
            sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b])

            self.assertEqual(sp_concat.indices.get_shape(), [8, 2])
            self.assertEqual(sp_concat.values.get_shape(), [8])
            self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

            concat_out = self.evaluate(sp_concat)

            self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [1, 4],
                                                     [2, 0], [2, 2], [2, 3],
                                                     [2, 6], [2, 7]])
            self.assertAllEqual(concat_out.values, [1, 2, 1, 3, 4, 2, 1, 0])
            self.assertAllEqual(concat_out.dense_shape, [3, 8])

  def testConcatDim0(self):
    # [FUNCTION DOC]
    # Concatenates 3x3 and 2x3 inputs along the first axis (given as -2 and 0). Asserts seven
    # entries with row indices of the second input shifted by three and dense shape 5x3.
    # Returns nothing.
    with self.session() as sess:
      # concat(A, D):
      # [    1]
      # [2    ]
      # [3   4]
      # [  1  ]
      # [1   2]
      sp_a = self._SparseTensor_3x3()
      sp_d = self._SparseTensor_2x3()

      for concat_dim in (-2, 0):
        sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_d])

        self.assertEqual(sp_concat.indices.get_shape(), [7, 2])
        self.assertEqual(sp_concat.values.get_shape(), [7])
        self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

        concat_out = self.evaluate(sp_concat)

        self.assertAllEqual(
            concat_out.indices,
            [[0, 2], [1, 0], [2, 0], [2, 2], [3, 1], [4, 0], [4, 2]])
        self.assertAllEqual(concat_out.values, np.array([1, 2, 3, 4, 1, 1, 2]))
        self.assertAllEqual(concat_out.dense_shape, np.array([5, 3]))

  def testConcat3(self):
    # [FUNCTION DOC]
    # Concatenates 3x3, 3x5 and 3x2 inputs along the last axis (-1 and 1). Asserts ten entries in
    # row-major order with proper column offsets, merged values and dense shape 3x10. Returns
    # nothing.
    with self.session() as sess:
      # concat(A, B, C):
      # [    1              ]
      # [2       1       1  ]
      # [3   4 2     1 0 2  ]
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x5()
      sp_c = self._SparseTensor_3x2()

      for concat_dim in (-1, 1):
        sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b, sp_c])

        self.assertEqual(sp_concat.indices.get_shape(), [10, 2])
        self.assertEqual(sp_concat.values.get_shape(), [10])
        self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

        concat_out = self.evaluate(sp_concat)

        self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [1, 4], [1, 8],
                                                 [2, 0], [2, 2], [2, 3], [2, 6],
                                                 [2, 7], [2, 8]])
        self.assertAllEqual(concat_out.values, [1, 2, 1, 1, 3, 4, 2, 1, 0, 2])
        self.assertAllEqual(concat_out.dense_shape, [3, 10])

  def testConcatNoNonZeros(self):
    # [FUNCTION DOC]
    # Concatenates three empty rank-3 sparse tensors (second dimension 3, 7, 5) along axis 1 and
    # asserts the result has no entries and dense shape 2x15x4. Returns nothing.
    sp_a = self._SparseTensor_NoNonZeros((2, 3, 4))
    sp_b = self._SparseTensor_NoNonZeros((2, 7, 4))
    sp_c = self._SparseTensor_NoNonZeros((2, 5, 4))

    with self.session() as sess:
      concat_dim = 1
      sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b, sp_c])

      self.assertEqual(sp_concat.indices.get_shape(), [0, 3])
      self.assertEqual(sp_concat.values.get_shape(), [0])
      self.assertEqual(sp_concat.dense_shape.get_shape(), [3])

      concat_out = self.evaluate(sp_concat)

      self.assertEqual(concat_out.indices.shape, (0, 3))
      self.assertEqual(concat_out.values.shape, (0,))
      self.assertAllEqual(concat_out.dense_shape, [2, 15, 4])

  def testConcatSomeNoNonZeros(self):
    # [FUNCTION DOC]
    # Concatenates an empty tensor, a non-empty 2x3x4 tensor and another empty tensor along axis
    # 1. Asserts the output keeps only the non-empty tensor's entries, with the second index
    # offset by the first input's extent, and dense shape 2x15x4. Returns nothing.
    sp_a = self._SparseTensor_NoNonZeros((2, 7, 4))
    sp_b = self._SparseTensor_2x3x4()
    sp_c = self._SparseTensor_NoNonZeros((2, 5, 4))
    output_nnz = sp_b.indices.get_shape()[0]

    with self.session() as sess:
      concat_dim = 1
      sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b, sp_c])

      self.assertEqual(sp_concat.indices.get_shape(), [output_nnz, 3])
      self.assertEqual(sp_concat.values.get_shape(), [output_nnz])
      self.assertEqual(sp_concat.dense_shape.get_shape(), [3])

      concat_out = self.evaluate(sp_concat)

      self.assertAllEqual(concat_out.indices,
                          sp_b.indices + [0, sp_a.dense_shape[1], 0])
      self.assertAllEqual(concat_out.values, sp_b.values)
      self.assertAllEqual(concat_out.dense_shape, [2, 15, 4])

  def testConcatNonNumeric(self):
    # [FUNCTION DOC]
    # CPU-only session. Concatenates 3x3 and 3x5 string-valued sparse tensors along the last
    # axis and asserts indices, byte-string values and dense shape 3x8. Returns nothing.
    with self.session(use_gpu=False) as sess:
      # concat(A, B):
      # [    a          ]
      # [b       e      ]
      # [c   d f     g h]
      sp_a = self._SparseTensor_String3x3()
      sp_b = self._SparseTensor_String3x5()

      for concat_dim in (-1, 1):
        sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b])

        self.assertEqual(sp_concat.indices.get_shape(), [8, 2])
        self.assertEqual(sp_concat.values.get_shape(), [8])
        self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

        concat_out = self.evaluate(sp_concat)

        self.assertAllEqual(
            concat_out.indices,
            [[0, 2], [1, 0], [1, 4], [2, 0], [2, 2], [2, 3], [2, 6], [2, 7]])
        self.assertAllEqual(concat_out.values,
                            [b"a", b"b", b"e", b"c", b"d", b"f", b"g", b"h"])
        self.assertAllEqual(concat_out.dense_shape, [3, 8])

  @test_util.run_deprecated_v1
  def testMismatchedRank(self):
    # [FUNCTION DOC]
    # Deprecated-v1 (graph) test. Concatenating a rank-2 and a rank-3 tensor must raise a value
    # error at graph construction time (shape inference) for concat dimensions -1 and 1.
    # Returns nothing.
    with self.session():
      sp_a = self._SparseTensor_3x3()
      sp_e = self._SparseTensor_2x3x4()

      # Rank mismatches can be caught at shape-inference time
      for concat_dim in (-1, 1):
        with self.assertRaises(ValueError):
          sparse_ops.sparse_concat(concat_dim, [sp_a, sp_e])

  @test_util.run_deprecated_v1
  def testMismatchedRankExpandNonconcatDim(self):
    # [FUNCTION DOC]
    # Deprecated-v1 (graph) test. Same as the rank mismatch test but with the option to expand
    # non-concat dimensions enabled; a value error is still expected at construction time.
    # Returns nothing.
    with self.session():
      sp_a = self._SparseTensor_3x3()
      sp_e = self._SparseTensor_2x3x4()

      # Rank mismatches should be caught at shape-inference time, even for
      # expand_nonconcat_dim=True.
      for concat_dim in (-1, 1):
        with self.assertRaises(ValueError):
          sparse_ops.sparse_concat(
              concat_dim, [sp_a, sp_e], expand_nonconcat_dim=True)

  @test_util.run_deprecated_v1
  def testMismatchedShapes(self):
    # [FUNCTION DOC]
    # Deprecated-v1 (graph) test. Concatenates four tensors whose non-concat dimensions differ
    # (3x3, 3x5, 3x2, 2x3) along the last axis; graph construction succeeds but running it must
    # raise an op error saying input shapes must match. Returns nothing.
    with self.session() as sess:
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x5()
      sp_c = self._SparseTensor_3x2()
      sp_d = self._SparseTensor_2x3()
      for concat_dim in (-1, 1):
        sp_concat = sparse_ops.sparse_concat(concat_dim,
                                             [sp_a, sp_b, sp_c, sp_d])

        # Shape mismatches can only be caught when the op is run
        with self.assertRaisesOpError("Input shapes must match"):
          self.evaluate(sp_concat)

  def testMismatchedShapesExpandNonconcatDim(self):
    # [FUNCTION DOC]
    # Concatenates the same four mismatched-shape tensors with expansion of non-concat
    # dimensions enabled, along axis 0 (-2 and 0) and axis 1 (-1 and 1). Asserts the exact
    # merged indices and values and dense shapes 11x5 and 3x13 respectively, i.e. smaller
    # tensors are treated as embedded in the larger extent. Returns nothing.
    with self.session() as sess:
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x5()
      sp_c = self._SparseTensor_3x2()
      sp_d = self._SparseTensor_2x3()
      for concat_dim0 in (-2, 0):
        for concat_dim1 in (-1, 1):
          sp_concat_dim0 = sparse_ops.sparse_concat(
              concat_dim0, [sp_a, sp_b, sp_c, sp_d], expand_nonconcat_dim=True)
          sp_concat_dim1 = sparse_ops.sparse_concat(
              concat_dim1, [sp_a, sp_b, sp_c, sp_d], expand_nonconcat_dim=True)

          sp_concat_dim0_out = self.evaluate(sp_concat_dim0)
          sp_concat_dim1_out = self.evaluate(sp_concat_dim1)

          self.assertAllEqual(sp_concat_dim0_out.indices,
                              [[0, 2], [1, 0], [2, 0], [2, 2], [4, 1], [5, 0],
                               [5, 3], [5, 4], [7, 0], [8, 0], [9, 1], [10, 0],
                               [10, 2]])
          self.assertAllEqual(sp_concat_dim0_out.values,
                              [1, 2, 3, 4, 1, 2, 1, 0, 1, 2, 1, 1, 2])
          self.assertAllEqual(sp_concat_dim0_out.dense_shape, [11, 5])

          self.assertAllEqual(sp_concat_dim1_out.indices,
                              [[0, 2], [0, 11], [1, 0], [1, 4], [1, 8], [1, 10],
                               [1, 12], [2, 0], [2, 2], [2, 3], [2, 6], [2, 7],
                               [2, 8]])
          self.assertAllEqual(sp_concat_dim1_out.values,
                              [1, 1, 2, 1, 1, 1, 2, 3, 4, 2, 1, 0, 2])
          self.assertAllEqual(sp_concat_dim1_out.dense_shape, [3, 13])

  @test_util.run_deprecated_v1
  def testShapeInferenceUnknownShapes(self):
    # [FUNCTION DOC]
    # Deprecated-v1 (graph) test. Concatenates four placeholder-based sparse tensors with
    # differently unknown or partially known static shapes along axis 0 and asserts the inferred
    # static shapes: indices unknown by 3, values unknown, dense shape of length 3. Returns
    # nothing.
    with self.session():
      sp_inputs = [
          self._SparseTensor_UnknownShape(),
          self._SparseTensor_UnknownShape(val_shape=[3]),
          self._SparseTensor_UnknownShape(ind_shape=[1, 3]),
          self._SparseTensor_UnknownShape(shape_shape=[3])
      ]

      for concat_dim in (-2, 0):
        sp_concat = sparse_ops.sparse_concat(concat_dim, sp_inputs)

        self.assertEqual(sp_concat.indices.get_shape().as_list(), [None, 3])
        self.assertEqual(sp_concat.values.get_shape().as_list(), [None])
        self.assertEqual(sp_concat.dense_shape.get_shape(), [3])

  def testConcatShape(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue (GitHub 21964). Concatenates two 2x2 sparse tensors
    # along the last axis and asserts the static dense shape of the result is 2x4. Returns
    # nothing.
    # Test case for GitHub 21964.
    x = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[2, 2])
    y = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[2, 2])
    z = sparse_ops.sparse_concat(-1, [x, y])
    self.assertEqual(z.get_shape().as_list(), [2, 4])


if __name__ == "__main__":
  test.main()
