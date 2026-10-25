# [MODULE DOC]
# Unit tests for the Python-level sparse operations in the sparse ops module. Contains a
# module-level random sparsifying helper and many small test classes, one per feature:
# converting sparse tensors to boolean indicator tensors, merging id and value tensors
# (including multi-dimensional vocabularies), retaining entries by mask, resetting or
# tightening the dense shape, setting static shape in eager and traced code, filling empty
# rows (including gradient behavior and invalid indices), adding a variable-backed sparse
# tensor, reductions (sum and max, dense and sparse results), element-wise math with dense
# tensors, softmax, element-wise minimum and maximum, transpose, and sparse placeholders.
# Several tests are graph-mode only (deprecated v1 decorator) and several force CPU or GPU
# placement. Two reduce tests are disabled by a naming trick. No import-time side effects
# other than importing the sparse gradient registrations.
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
"""Tests for Python ops defined in sparse_ops."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  # [FUNCTION DOC]
  # HELPER: converts a dense numpy array into a sparse tensor for the tests in this module.
  # Parameters: x is a numeric array; thresh is the cutoff below which entries are treated as
  # zero (default 0.5); index_dtype is the integer type of the indices (default 64-bit).
  # KNOWN ISSUE: it modifies the caller's array in place by zeroing all entries below the
  # threshold, so callers that reuse the array afterwards see the thresholded data.
  # Returns a pair: a sparse tensor holding the non-zero entries (indices as a matrix of
  # coordinates, values, and the array's shape as the dense shape) and the count of stored values.
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)


class SparseToIndicatorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for converting a sparse tensor of integer ids into a boolean indicator tensor whose
  # last dimension is a given size. Fixtures: a 5x6 and a 2x3x4 sparse tensor of constants.

  def _SparseTensor_5x6(self, dtype):
    # [FUNCTION DOC]
    # HELPER: builds the 5x6 sparse tensor from constants, with six entries and values 0, 10,
    # 13, 14, 32, 33. Parameter dtype is the value dtype; indices and shape are 64-bit integers.
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x3x4(self, dtype):
    # [FUNCTION DOC]
    # HELPER: builds a 2x3x4 sparse tensor from constants with eight entries, values 1, 10, 12,
    # 103, 150, 149, 150, 122. Two entries in the position group of row one, column one share
    # the value 150 (so the indicator collapses duplicates). Parameter dtype is the value dtype.
    # Includes two entries with the form [1, 1, x] : 150.
    ind = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 2], [1, 0, 3], [1, 1, 0],
                    [1, 1, 1], [1, 1, 2], [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 150, 149, 150, 122])
    shape = np.array([2, 3, 4])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, dtypes.int64))

  def testInt32(self):
    # [FUNCTION DOC]
    # Test (CPU): converts the 32-bit integer 5x6 tensor to an indicator with last dimension 50
    # and asserts it equals a 5 by 50 boolean array with true exactly at the six value positions.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_5x6(dtypes.int32)
      output = sparse_ops.sparse_to_indicator(sp_input, 50)

      expected_output = np.zeros((5, 50), dtype=np.bool_)
      expected_trues = ((0, 0), (1, 10), (1, 13), (1, 14), (3, 32), (3, 33))
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)

  def testInt64(self):
    # [FUNCTION DOC]
    # Test (CPU): same as the 32-bit test but with 64-bit integer values.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_5x6(dtypes.int64)
      output = sparse_ops.sparse_to_indicator(sp_input, 50)

      expected_output = np.zeros((5, 50), dtype=np.bool_)
      expected_trues = [(0, 0), (1, 10), (1, 13), (1, 14), (3, 32), (3, 33)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)

  def testHigherRank(self):
    # [FUNCTION DOC]
    # Test (CPU): converts the rank-three 2x3x4 tensor to an indicator with last dimension 200
    # and asserts the result is a 2 by 3 by 200 boolean array with true at the seven distinct
    # (row, column, value) positions (duplicate 150 appears once).
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x3x4(dtypes.int64)
      output = sparse_ops.sparse_to_indicator(sp_input, 200)

      expected_output = np.zeros((2, 3, 200), dtype=np.bool_)
      expected_trues = [(0, 0, 1), (0, 1, 10), (0, 1, 12), (1, 0, 103),
                        (1, 1, 149), (1, 1, 150), (1, 2, 122)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)


class SparseMergeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for merging a sparse tensor of ids and a parallel sparse tensor of values into one
  # sparse tensor whose last dimension is a vocabulary of a given size. The fixtures
  # are intentionally unsorted so the optional already-sorted flag changes the output order.

  def _SparseTensorValue_3x50(self, indices_dtype, values_dtype):
    # [FUNCTION DOC]
    # HELPER: builds two numpy-backed sparse values sharing the same coordinates (a 3 by 3
    # dense shape, six unsorted entries): an id tensor (ids 0, 13, 10, 33, 32, 14) of
    # indices_dtype and a value tensor (-3, 4, 1, 9, 5, 1) of values_dtype. Returns the pair
    # (ids, values). The ordering is deliberately not canonical.
    # NOTE: This input is intentionally not sorted to validate the
    # already_sorted flag below.
    ind = np.array([[0, 0], [1, 0], [1, 2], [2, 0], [2, 1], [1, 1]])
    # NB: these are not sorted
    indices = np.array([0, 13, 10, 33, 32, 14])
    values = np.array([-3, 4, 1, 9, 5, 1])
    shape = np.array([3, 3])
    indices = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(indices, indices_dtype), np.array(shape, np.int64))
    values = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(values, values_dtype), np.array(shape, np.int64))
    return indices, values

  def _SparseTensor_3x50(self, indices_dtype, values_dtype):
    # [FUNCTION DOC]
    # HELPER: same data as the value-building fixture but each returned element is wrapped as a
    # real sparse tensor object. Parameters are the id dtype and value dtype. Returns (ids, values).
    indices, values = self._SparseTensorValue_3x50(indices_dtype, values_dtype)
    return (sparse_tensor.SparseTensor.from_value(indices),
            sparse_tensor.SparseTensor.from_value(values))

  def _AssertResultsSorted(self, output, vocab_size):
    # [FUNCTION DOC]
    # HELPER: asserts a merge result is in canonical order: indices (row, id) sorted, values
    # -3, 1, 4, 1, 5, 9, and dense shape of 3 followed by vocab_size (an integer).
    self.assertAllEqual(output.indices,
                        [[0, 0], [1, 10], [1, 13], [1, 14], [2, 32], [2, 33]])
    self.assertAllEqual(output.values, [-3, 1, 4, 1, 5, 9])
    self.assertAllEqual(output.dense_shape, [3, vocab_size])

  def _AssertResultsNotSorted(self, output, vocab_size):
    # [FUNCTION DOC]
    # HELPER: asserts a merge result kept the input (non-canonical) order, as happens when the
    # already-sorted flag is set: indices (0,0), (1,13), (1,10), (2,33), (2,32), (1,14), values
    # -3, 4, 1, 9, 5, 1, and dense shape of 3 followed by vocab_size.
    self.assertAllEqual(output.indices,
                        [[0, 0], [1, 13], [1, 10], [2, 33], [2, 32], [1, 14]])
    self.assertAllEqual(output.values, [-3, 4, 1, 9, 5, 1])
    self.assertAllEqual(output.dense_shape, [3, vocab_size])

  def testInt32AndFloat32(self):
    # [FUNCTION DOC]
    # Test (CPU): merges 32-bit integer ids with 32-bit float values, trying both raw values
    # and sparse tensor objects for each input (four combinations), vocabulary size 50, and
    # asserts every result is canonically sorted.
    vocab_size = 50
    indices_v, values_v = self._SparseTensorValue_3x50(np.int32, np.float32)
    with test_util.force_cpu():
      for indices in (indices_v,
                      sparse_tensor.SparseTensor.from_value(indices_v)):
        for values in (values_v,
                       sparse_tensor.SparseTensor.from_value(values_v)):
          sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

          output = self.evaluate(sp_output)
          self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat32(self):
    # [FUNCTION DOC]
    # Test (CPU): merges 64-bit integer ids with 32-bit float values, vocabulary size 50, and
    # asserts the output is canonically sorted.
    vocab_size = 50
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float32)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat64(self):
    # [FUNCTION DOC]
    # Test (CPU): merges 64-bit integer ids with 64-bit float values, vocabulary size 50, and
    # asserts the output is canonically sorted.
    vocab_size = 50
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt32AndFloat32NonCanonicalOrder(self):
    # [FUNCTION DOC]
    # Test (CPU): merges 32-bit integer ids with 32-bit float values with the already-sorted
    # flag set, so no sorting happens; asserts the output keeps the input's unsorted order.
    vocab_size = 50
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int32, np.float32)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size, already_sorted=True)

      output = self.evaluate(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)

  def testInt64AndFloat32NonCanonicalOrder(self):
    # [FUNCTION DOC]
    # Test (CPU): same as the 32-bit non-canonical test but with 64-bit integer ids.
    vocab_size = 50
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float32)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size, already_sorted=True)

      output = self.evaluate(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)

  def testInt64AndFloat64NonCanonicalOrder(self):
    # [FUNCTION DOC]
    # Test (CPU): 64-bit ids and 64-bit float values with the already-sorted flag set, where the
    # vocabulary size is passed as a 64-bit integer constant tensor instead of a plain integer;
    # asserts the output keeps the input's unsorted order.
    vocab_size = 50
    vocab_size_tensor = constant_op.constant(vocab_size, dtypes.int64)
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size_tensor, already_sorted=True)

      output = self.evaluate(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)

  def testShouldSetLastDimensionInDynamicShape(self):
    # [FUNCTION DOC]
    # Test (explicit graph): builds id and value sparse tensors whose dense shape is only
    # dynamically known (a default-valued placeholder), merges them with vocabulary size 5,
    # and asserts the static shape of the merged tensor reports 5 for its last dimension.
    with ops.Graph().as_default():
      shape = constant_op.constant([2, 2], dtype=dtypes.int64)
      dynamic_shape = array_ops.placeholder_with_default(shape, shape=[2])
      ids = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1]],
          values=[1, 3],
          dense_shape=dynamic_shape)
      values = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1]],
          values=[0.4, 0.7],
          dense_shape=dynamic_shape)
      merged = sparse_ops.sparse_merge(
          sp_ids=ids, sp_values=values, vocab_size=5)
      self.assertEqual(5, merged.get_shape()[1])


class SparseMergeHighDimTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for merging a list of two id tensors (two vocabulary dimensions) with one value
  # tensor into a rank-three sparse tensor, where the vocabulary size is a list of two integers.

  def _SparseTensor_3x50(self, indices_dtype, values_dtype):
    # [FUNCTION DOC]
    # HELPER: builds the inputs for the multi-dimensional merge. Returns a pair: a list of two
    # sparse id tensors (first ids 0, 13, 10, 33, 32, 14; second ids 12, 4, 0, 0, 1, 30), both
    # of indices_dtype, and one sparse value tensor (-3, 4, 1, 9, 5, 1) of values_dtype, all with
    # the same unsorted coordinates and a 3 by 3 dense shape.
    # NOTE: This input is intentionally not sorted to validate the
    # already_sorted flag below.
    ind = np.array([[0, 0], [1, 0], [1, 2], [2, 0], [2, 1], [1, 1]])
    # NB: these are not sorted
    indices0 = np.array([0, 13, 10, 33, 32, 14])
    indices1 = np.array([12, 4, 0, 0, 1, 30])
    values = np.array([-3, 4, 1, 9, 5, 1])
    shape = np.array([3, 3])
    indices0 = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(indices0, indices_dtype), np.array(shape, np.int64))
    indices1 = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(indices1, indices_dtype), np.array(shape, np.int64))
    values = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(values, values_dtype), np.array(shape, np.int64))
    return ([sparse_tensor.SparseTensor.from_value(indices0),
             sparse_tensor.SparseTensor.from_value(indices1)],
            sparse_tensor.SparseTensor.from_value(values))

  def _AssertResultsSorted(self, output, vocab_size):
    # [FUNCTION DOC]
    # HELPER: asserts the multi-dimensional merge result is canonically sorted with
    # three-column indices, values -3, 1, 4, 1, 5, 9, and dense shape equal to 3 followed by the
    # entries of vocab_size (a list of two integers).
    self.assertAllEqual(
        output.indices,
        [[0, 0, 12], [1, 10, 0], [1, 13, 4], [1, 14, 30], [2, 32, 1],
         [2, 33, 0]])
    self.assertAllEqual(output.values, [-3, 1, 4, 1, 5, 9])
    self.assertAllEqual(output.dense_shape, [3] + vocab_size)

  def testInt64AndFloat32(self):
    # [FUNCTION DOC]
    # Test (CPU): merges two 64-bit id tensors and 32-bit float values with vocabulary sizes 50
    # and 31 and asserts the sorted rank-three result.
    vocab_size = [50, 31]
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float32)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat64(self):
    # [FUNCTION DOC]
    # Test (CPU): merges two 64-bit id tensors and 64-bit float values with vocabulary sizes 50
    # and 31 and asserts the sorted rank-three result.
    vocab_size = [50, 31]
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat64Shape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name suggesting a shape check, this test is an exact duplicate
    # of the preceding 64-bit float test; it never inspects static shape separately.
    # Test (CPU): merges two 64-bit id tensors and 64-bit float values with vocabulary sizes 50
    # and 31 and asserts the sorted rank-three result including the dense shape.
    vocab_size = [50, 31]
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)


class SparseRetainTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for keeping only the sparse entries selected by a boolean mask. Fixture: a 5x6
  # sparse tensor with six entries, available both as a raw value and as a tensor object.

  def _SparseTensorValue_5x6(self):
    # [FUNCTION DOC]
    # HELPER: returns the numpy-backed 5x6 sparse value with six entries (values 0, 10, 13, 14,
    # 32, 33 as 32-bit integers, 64-bit indices and dense shape). No inputs.
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.int32), np.array(shape, np.int64))

  def _SparseTensor_5x6(self):
    # [FUNCTION DOC]
    # HELPER: returns the 5x6 fixture converted to a sparse tensor object. No inputs.
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_5x6())

  def testBasic(self):
    # [FUNCTION DOC]
    # Test (CPU): for both the raw value and the tensor form, keeps entries selected by a mask
    # (positions one, four, five) and asserts the retained indices, values 0, 14, 32 and the
    # unchanged dense shape 5 by 6.
    with test_util.force_cpu():
      for sp_input in (self._SparseTensorValue_5x6(), self._SparseTensor_5x6()):
        to_retain = np.array([1, 0, 0, 1, 1, 0], dtype=np.bool_)
        sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

        output = self.evaluate(sp_output)

        self.assertAllEqual(output.indices, [[0, 0], [1, 4], [3, 2]])
        self.assertAllEqual(output.values, [0, 14, 32])
        self.assertAllEqual(output.dense_shape, [5, 6])

  def testRetainNone(self):
    # [FUNCTION DOC]
    # Test (CPU): retains nothing (all-false mask) and asserts the result has an empty indices
    # matrix with two columns, empty values and unchanged dense shape 5 by 6.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_5x6()
      to_retain = np.zeros((6,), dtype=np.bool_)
      sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices, np.array([]).reshape((0, 2)))
      self.assertAllEqual(output.values, [])
      self.assertAllEqual(output.dense_shape, [5, 6])

  def testMismatchedRetainShape(self):
    # [FUNCTION DOC]
    # Test (CPU): supplies a mask of length five for a tensor with six stored values and
    # asserts a value error is raised.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_5x6()
      to_retain = np.array([1, 0, 0, 1, 0], dtype=np.bool_)
      with self.assertRaises(ValueError):
        sparse_ops.sparse_retain(sp_input, to_retain)


class SparseResetShapeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for resetting the dense shape of a sparse tensor to a larger explicit shape or to
  # the tightest bounding box of its entries, including error handling for wrong rank or
  # too-small dimensions (statically and at run time). Class attributes hold a rank-three
  # 2x5x6 fixture: six indices, six 32-bit values and the shape.

  _IND_2_5_6 = np.array(
      [[0, 0, 0], [0, 1, 0], [0, 1, 3], [1, 1, 4], [1, 3, 2], [1, 3, 3]],
      dtype=np.int64)
  _VAL_2_5_6 = np.array([0, 10, 13, 14, 32, 33], dtype=np.int32)
  _SHP_2_5_6 = np.array([2, 5, 6], dtype=np.int64)

  def _SparseTensor_2x5x6(self):
    # [FUNCTION DOC]
    # HELPER: returns the 2x5x6 fixture as a sparse tensor built from constants (64-bit indices,
    # 32-bit values, 64-bit shape). No inputs.
    return sparse_tensor.SparseTensor(
        constant_op.constant(self._IND_2_5_6, dtypes.int64),
        constant_op.constant(self._VAL_2_5_6, dtypes.int32),
        constant_op.constant(self._SHP_2_5_6, dtypes.int64))

  def _SparseTensor_2x5x6_Empty(self):
    # [FUNCTION DOC]
    # HELPER: returns a sparse tensor with the 2x5x6 dense shape but zero stored entries (empty
    # indices with three columns, empty 32-bit values). No inputs.
    return sparse_tensor.SparseTensor(
        constant_op.constant(
            np.empty(shape=[0, 3], dtype=np.int64), dtypes.int64),
        constant_op.constant(np.empty(shape=[0], dtype=np.int32), dtypes.int32),
        constant_op.constant(self._SHP_2_5_6, dtypes.int64))

  def _SparseTensorValue_2x5x6(self):
    # [FUNCTION DOC]
    # HELPER: returns the 2x5x6 fixture as a numpy-backed sparse value (used for feeding or for
    # passing raw values). No inputs.
    return sparse_tensor.SparseTensorValue(self._IND_2_5_6, self._VAL_2_5_6,
                                           self._SHP_2_5_6)

  def testStaticShapeInfoPreservedWhenNewShapeIsProvidedAndStatic(self):
    # [FUNCTION DOC]
    # Test: resets the shape of the 2x5x6 tensor to a constant 3 by 6 by 7 and asserts the
    # output's static shape equals that new shape.
    sp_input = self._SparseTensor_2x5x6()
    new_shape = np.array([3, 6, 7], dtype=np.int64)
    sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)
    self.assertAllEqual([3, 6, 7], sp_output.get_shape())

  def testBasic(self):
    # [FUNCTION DOC]
    # Test (CPU): resets the 2x5x6 tensor's shape to 3 by 6 by 7 and asserts indices and values
    # are unchanged while the dense shape is the new one.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x5x6()
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  def testInputUnavailableInGraphConstructionOk(self):
    # [FUNCTION DOC]
    # Test (CPU): passes the raw numpy-backed value (not a tensor) as the input, resets the
    # shape to 3 by 6 by 7, and asserts the same unchanged indices and values with new shape.
    with test_util.force_cpu():
      sp_input = self._SparseTensorValue_2x5x6()
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  @test_util.run_deprecated_v1
  def testFeedInputUnavailableInGraphConstructionOk(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): uses a sparse placeholder as the input, resets the
    # shape to 3 by 6 by 7, feeds the 2x5x6 value, and asserts unchanged indices and values with
    # the new dense shape.
    with self.session(use_gpu=False) as sess:
      sp_input = array_ops.sparse_placeholder(dtype=dtypes.int32)
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = sess.run(sp_output,
                        feed_dict={sp_input: self._SparseTensorValue_2x5x6()})

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  def testTightBoundingBox(self):
    # [FUNCTION DOC]
    # Test (CPU): resets the shape without giving a new shape, which shrinks it to the tight
    # bounding box of the entries; asserts indices and values are unchanged and the dense
    # shape becomes 2 by 4 by 5.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x5x6()
      sp_output = sparse_ops.sparse_reset_shape(sp_input)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [2, 4, 5])

  def testTightBoundingBoxEmpty(self):
    # [FUNCTION DOC]
    # Test (CPU): tight-bounding-box reset of a tensor with no entries; asserts indices keep
    # shape zero by three, values keep shape zero, and the dense shape becomes all zeros.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x5x6_Empty()
      sp_output = sparse_ops.sparse_reset_shape(sp_input)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices.shape, [0, 3])
      self.assertAllEqual(output.values.shape, [0])
      self.assertAllEqual(output.dense_shape, [0, 0, 0])

  def testInvalidRank(self):
    # [FUNCTION DOC]
    # Test (CPU): supplies a new shape of rank two for a rank-three tensor and asserts a value
    # error is raised at graph construction.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x5x6()
      new_shape = np.array([3, 7], dtype=np.int64)

      with self.assertRaises(ValueError):
        sparse_ops.sparse_reset_shape(sp_input, new_shape)

  @test_util.run_deprecated_v1
  def testInvalidRankNewShapeUnavailableInGraphConstruction(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): the new shape is a placeholder, so rank is checked at
    # run time; feeding a rank-two shape for a rank-three tensor must raise an op error whose
    # message says an element-wise equality did not hold.
    with self.session(use_gpu=False) as sess:
      new_shape = array_ops.placeholder(dtype=dtypes.int64)
      sp_input = self._SparseTensor_2x5x6()
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x == y did not hold element-wise"):
        sess.run(out, feed_dict={new_shape: np.array([3, 7], dtype=np.int64)})

  def testInvalidDimensionSizeStatic(self):
    # [FUNCTION DOC]
    # Test: a constant new shape 3 by 7 by 5 is smaller than the data in the last dimension
    # (size 6 needed); asserts a value error whose message says the new shape should have
    # dimension sizes at least as large as the old ones.
    sp_input = self._SparseTensor_2x5x6()
    new_shape = np.array([3, 7, 5], dtype=np.int64)

    with self.assertRaisesRegex(ValueError, "should have dimension sizes"):
      sparse_ops.sparse_reset_shape(sp_input, new_shape)

  @test_util.run_deprecated_v1
  def testInvalidDimensionSizeDynamic(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): the new shape is a 32-bit placeholder fed with 3 by
    # 7 by 5, too small in the last dimension; asserts running raises an op error saying an
    # element-wise less-or-equal check did not hold.
    with self.session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_2x5x6()
      new_shape = array_ops.placeholder(dtype=dtypes.int32)
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x <= y did not hold element-wise"):
        sess.run(out, feed_dict={new_shape: [3, 7, 5]})

  @test_util.run_deprecated_v1
  def testInvalidDimensionSizeInputUnavailableInGraphConstruction(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): the input is a sparse placeholder, the constant new
    # shape 3 by 7 by 5 is too small for the fed data; asserts running raises an op error saying
    # an element-wise less-or-equal check did not hold.
    sp_input = array_ops.sparse_placeholder(dtype=dtypes.int32)
    with self.session(use_gpu=False) as sess:
      new_shape = np.array([3, 7, 5], dtype=np.int64)
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x <= y did not hold element-wise"):
        sess.run(out, feed_dict={sp_input: self._SparseTensorValue_2x5x6()})


class SparseSetShapeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the static shape setter on sparse tensors: compatible partial shapes are
  # accepted and merged, incompatible ones raise errors, in eager mode and inside a traced function.

  def testSetShapeEagerValidates(self):
    # [FUNCTION DOC]
    # Test: builds a 5x6 sparse tensor from constants, checks its static shape, then applies a
    # series of compatible static shapes (unknown, partial, full) without error, and asserts
    # that setting a wrong rank, or a wrong known dimension (three, seven, or both), raises
    # a value error each time.
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    sp = sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int64),
        constant_op.constant(shape, dtypes.int64))

    self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

    sp.set_shape(tensor_shape.TensorShape(None))
    sp.set_shape(tensor_shape.TensorShape([None, None]))
    sp.set_shape(tensor_shape.TensorShape([5, None]))
    sp.set_shape(tensor_shape.TensorShape([None, 6]))
    sp.set_shape(tensor_shape.TensorShape([5, 6]))

    with self.assertRaises(ValueError):
      sp.set_shape([None, None, None])

    with self.assertRaises(ValueError):
      sp.set_shape([3, None])

    with self.assertRaises(ValueError):
      sp.set_shape([None, 7])

    with self.assertRaises(ValueError):
      sp.set_shape([3, 6])

  def testSetShapeFunctionMerges(self):
    # [FUNCTION DOC]
    # Test: traces a function whose dense shape input has unknown rank and checks, during
    # tracing, that setting progressively more specific static shapes merges information (the
    # shape grows from unknown, to rank two, to first dimension 5, to fully 5 by 6 and stays
    # there) and that incompatible shapes raise value errors. The test then gets a concrete
    # function from a shape spec with unknown shape and 64-bit integer type.

    @def_function.function
    def dynamic_shape_sparse(dense_shape):
      # [FUNCTION DOC]
      # Traced body: builds a sparse tensor from fixed constants and the given dense-shape
      # argument, then performs the static shape merging and error assertions described in
      # the enclosing test. Returns nothing; the checks run while tracing.
      ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
      val = np.array([0, 10, 13, 14, 32, 33])
      sp = sparse_tensor.SparseTensor(
          constant_op.constant(ind, dtypes.int64),
          constant_op.constant(val, dtypes.int64),
          dense_shape)

      sp.set_shape(tensor_shape.TensorShape(None))
      self.assertEqual(sp.shape, tensor_shape.TensorShape(None))

      sp.set_shape(tensor_shape.TensorShape([None, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([None, None]))

      sp.set_shape(tensor_shape.TensorShape([5, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, None]))

      sp.set_shape(tensor_shape.TensorShape([None, 6]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      sp.set_shape(tensor_shape.TensorShape([None, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      sp.set_shape(tensor_shape.TensorShape([5, 6]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      with self.assertRaises(ValueError):
        sp.set_shape([None, None, None])

      with self.assertRaises(ValueError):
        sp.set_shape([3, None])

      with self.assertRaises(ValueError):
        sp.set_shape([None, 7])

      with self.assertRaises(ValueError):
        sp.set_shape([3, 6])

    dense_shape_spec = tensor_spec.TensorSpec(None, dtypes.int64)
    _ = dynamic_shape_sparse.get_concrete_function(dense_shape_spec)


class SparseFillEmptyRowsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for inserting a default value into every empty row of a sparse matrix, which also
  # returns a boolean indicator of which rows were empty. Covers number and string values,
  # gradients, unordered input, no-empty-rows, zero-entry and zero-row inputs, invalid
  # indices, and the gradient op's handling of empty or invalid reverse index maps. Many tests
  # run on GPU when available.

  def _SparseTensorValue_5x6(self, dtype=np.int32):
    # [FUNCTION DOC]
    # HELPER: returns the numpy-backed 5x6 sparse value with six entries (rows 0, 1, 1, 1, 3, 3,
    # leaving rows 2 and 4 empty). Parameter dtype is the numpy type of the values (default
    # 32-bit integer); indices and shape are 64-bit integers.
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64), np.array(val, dtype), np.array(
            shape, np.int64))

  def _SparseTensor_5x6(self):
    # [FUNCTION DOC]
    # HELPER: returns the default 5x6 fixture (32-bit integer values) as a sparse tensor object.
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_5x6())

  def _SparseTensor_String5x6(self):
    # [FUNCTION DOC]
    # HELPER: returns a 5x6 sparse tensor with the same coordinates as the numeric fixture but
    # string values "a" through "f", built from constants.
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array(["a", "b", "c", "d", "e", "f"])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.string),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x6(self):
    # [FUNCTION DOC]
    # HELPER: returns a 2x6 sparse tensor (32-bit values 0, 10, 13, 14 in rows zero and one) that
    # has no empty rows.
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4]])
    val = np.array([0, 10, 13, 14])
    shape = np.array([2, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int32),
        constant_op.constant(shape, dtypes.int64))

  def testFillNumber(self):
    # [FUNCTION DOC]
    # Test (GPU when available): for both raw value and tensor forms of the 5x6 fixture, fills
    # empty rows with the default -1 and asserts the output has new entries at rows two and
    # four, values extended with -1, unchanged dense shape, and an indicator marking rows two
    # and four as empty.
    with test_util.use_gpu():
      for sp_input in (self._SparseTensorValue_5x6(), self._SparseTensor_5x6()):
        sp_output, empty_row_indicator = (
            sparse_ops.sparse_fill_empty_rows(sp_input, -1))

        output, empty_row_indicator_out = self.evaluate(
            [sp_output, empty_row_indicator])

        self.assertAllEqual(
            output.indices,
            [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
        self.assertAllEqual(output.values, [0, 10, 13, 14, -1, 32, 33, -1])
        self.assertAllEqual(output.dense_shape, [5, 6])
        self.assertAllEqual(empty_row_indicator_out,
                            np.array([0, 0, 1, 0, 1]).astype(np.bool_))

  def testSparseFillEmptyRowsGradEmpty(self):
    # [FUNCTION DOC]
    # Test (GPU when available): runs the gradient of fill-empty-rows with an empty reverse index
    # map and empty incoming gradient values; asserts the value gradient is empty.
    with test_util.use_gpu():
      grad, _ = self.evaluate(
          sparse_ops.sparse_fill_empty_rows_grad(
              reverse_index_map=[], grad_values=[]))
      self.assertAllEqual(grad, [])

  def testSparseFillEmptyRowsGradInvalidReverseIndexMap(self):
    # [FUNCTION DOC]
    # Test (GPU when available): runs the fill-empty-rows gradient with an invalid reverse index
    # map (entries negative one and three) and no gradient values. Behavior depends on device:
    # if a GPU is available, invalid indices are ignored and the result is two zeros; otherwise
    # (CPU) it must raise a value error or invalid-argument error saying elements in the
    # reverse index must be in range.
    # On CPU, invalid indices raise assertion.  On GPU, invalid indices
    # are simply ignored, for performance reasons.
    with test_util.use_gpu():
      if test_util.is_gpu_available():
        grad, _ = self.evaluate(
            sparse_ops.sparse_fill_empty_rows_grad(
                reverse_index_map=[-1, 3], grad_values=[]
            )
        )
        self.assertAllEqual(grad, [0., 0.])
      else:
        with self.assertRaisesRegex(
            (ValueError, errors.InvalidArgumentError),
            "Elements in reverse index must be in .*",
        ):
          self.evaluate(
              sparse_ops.sparse_fill_empty_rows_grad(
                  reverse_index_map=[-1, 3], grad_values=[]
              )
          )

  @test_util.run_deprecated_v1
  def testFillFloat(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): fills empty rows of a 64-bit float 5x6 tensor with
    # a default constant -1.0 and asserts indices, values, shape and empty-row indicator as in
    # the integer case. Then uses the numerical gradient checker to confirm the gradient
    # error with respect to the input values and to the default value is positive but below 1e-8.
    with self.session():
      values = constant_op.constant(
          [0.0, 10.0, 13.0, 14.0, 32.0, 33.0], dtype=dtypes.float64)
      default_value = constant_op.constant(-1.0, dtype=dtypes.float64)
      sp_input = sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]]),
          values=values,
          dense_shape=np.array([5, 6]))
      sp_output, empty_row_indicator = (sparse_ops.sparse_fill_empty_rows(
          sp_input, default_value))
      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 0], [1, 0], [1, 3], [1, 4],
                                           [2, 0], [3, 2], [3, 3], [4, 0]])
      self.assertAllClose(output.values, [0, 10, 13, 14, -1, 32, 33, -1])
      self.assertAllEqual(output.dense_shape, [5, 6])
      self.assertAllEqual(empty_row_indicator_out,
                          np.array([0, 0, 1, 0, 1]).astype(np.bool_))

      values_grad_err = gradient_checker.compute_gradient_error(
          values, values.shape.as_list(), sp_output.values, [8], delta=1e-8)
      self.assertGreater(values_grad_err, 0)
      self.assertLess(values_grad_err, 1e-8)

      default_value_grad_err = gradient_checker.compute_gradient_error(
          default_value,
          default_value.shape.as_list(),
          sp_output.values, [8],
          delta=1e-8)
      self.assertGreater(default_value_grad_err, 0)
      self.assertLess(default_value_grad_err, 1e-8)

  def testFillString(self):
    # [FUNCTION DOC]
    # Test (CPU): fills empty rows of the string-valued 5x6 tensor with the empty string and
    # asserts the new indices, values with empty strings at rows two and four, unchanged
    # shape and the empty-row indicator.
    with test_util.force_cpu():
      sp_input = self._SparseTensor_String5x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, ""))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(
          output.indices,
          [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
      self.assertAllEqual(output.values,
                          [b"a", b"b", b"c", b"d", b"", b"e", b"f", b""])
      self.assertAllEqual(output.dense_shape, [5, 6])
      self.assertAllEqual(empty_row_indicator_out,
                          np.array([0, 0, 1, 0, 1]).astype(np.bool_))

  def testNoEmptyRows(self):
    # [FUNCTION DOC]
    # Test (GPU when available): when no row is empty the output equals the input and the
    # empty-row indicator is all false.
    with test_util.use_gpu():
      sp_input = self._SparseTensor_2x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 0], [1, 0], [1, 3], [1, 4]])
      self.assertAllEqual(output.values, [0, 10, 13, 14])
      self.assertAllEqual(output.dense_shape, [2, 6])
      self.assertAllEqual(empty_row_indicator_out, np.zeros(2).astype(np.bool_))

  def testNoEmptyRowsAndUnordered(self):
    # [FUNCTION DOC]
    # Test (GPU when available): input with no empty rows but entries out of order (row one
    # before row zero); asserts the output is sorted by row and the indicator is all false.
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.array([[1, 2], [1, 3], [0, 1], [0, 3]]),
          values=np.array([1, 3, 2, 4]),
          dense_shape=np.array([2, 5]))
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 1], [0, 3], [1, 2], [1, 3]])
      self.assertAllEqual(output.values, [2, 4, 1, 3])
      self.assertAllEqual(output.dense_shape, [2, 5])
      self.assertAllEqual(empty_row_indicator_out, np.zeros(2).astype(np.bool_))

  def testUnordered(self):
    # [FUNCTION DOC]
    # Test (GPU when available): input on three rows with row one empty and row two entries
    # out of column order; asserts rows are grouped (row two entries keep their input order after
    # the filled row), the default -1 is inserted for row one, and the indicator is
    # false, true, false.
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.array([[2, 3], [2, 2], [0, 1], [0, 3]]),
          values=np.array([1, 3, 2, 4]),
          dense_shape=np.array([3, 5]))
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices,
                          [[0, 1], [0, 3], [1, 0], [2, 3], [2, 2]])
      self.assertAllEqual(output.values, [2, 4, -1, 1, 3])
      self.assertAllEqual(output.dense_shape, [3, 5])
      self.assertAllEqual(empty_row_indicator_out, [False, True, False])

  def testEmptyIndicesTensor(self):
    # [FUNCTION DOC]
    # Test (GPU when available): a 2x5 tensor with no entries; asserts both rows are filled
    # with the default -1 at column zero and the indicator is all true.
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.ones([0, 2]),
          values=np.ones([0]),
          dense_shape=np.array([2, 5]))
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 0], [1, 0]])
      self.assertAllEqual(output.values, [-1, -1])
      self.assertAllEqual(output.dense_shape, [2, 5])
      self.assertAllEqual(empty_row_indicator_out, np.ones(2).astype(np.bool_))

  def testEmptyOutput(self):
    # [FUNCTION DOC]
    # Test (GPU when available): a tensor with zero rows (dense shape 0 by 3) and no entries;
    # asserts the output has no entries, keeps the shape, and the indicator is empty.
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.ones([0, 2]),
          values=np.ones([0]),
          dense_shape=np.array([0, 3]))
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, np.ones([0, 2]))
      self.assertAllEqual(output.values, np.ones([0]))
      self.assertAllEqual(output.dense_shape, [0, 3])
      self.assertAllEqual(empty_row_indicator_out, [])

  def testInvalidIndices(self):
    # [FUNCTION DOC]
    # Test (GPU when available): input has row indices (99) beyond the declared 2 rows; asserts
    # evaluation raises an invalid-argument error saying the offending index is invalid.
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.array([[1, 2], [1, 3], [99, 1], [99, 3]]),
          values=np.array([1, 3, 2, 4]),
          dense_shape=np.array([2, 5]))

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  r"indices\(2, 0\) is invalid"):
        self.evaluate(sparse_ops.sparse_fill_empty_rows(sp_input, -1))


class SparseAddTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for adding two sparse tensors; currently only the case where values come from a
  # variable.

  def testValuesInVariable(self):
    # [FUNCTION DOC]
    # Test: builds a one-element sparse tensor whose values are a non-trainable 32-bit float
    # variable, adds it to itself, initializes variables on CPU, and asserts the summed
    # value is 2.
    indices = constant_op.constant([[0]], dtype=dtypes.int64)
    values = variables.Variable([1], trainable=False, dtype=dtypes.float32)
    shape = constant_op.constant([1], dtype=dtypes.int64)

    sp_input = sparse_tensor.SparseTensor(indices, values, shape)
    sp_output = sparse_ops.sparse_add(sp_input, sp_input)

    with test_util.force_cpu():
      self.evaluate(variables.global_variables_initializer())
      output = self.evaluate(sp_output)
      self.assertAllEqual(output.values, [2])


class SparseReduceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for sparse reductions (sum and max) with optional kept dimensions, producing dense
  # or sparse results, plus error handling, gradients and integer overflow. Class attributes
  # define a small 2x3 fixture with three stored ones. Two of the randomized comparison tests
  # are disabled by renaming.

  # [[1, ?, 2]
  #  [?, 3, ?]]
  # where ? is implicitly-zero.
  ind = np.array([[0, 0], [0, 2], [1, 1]]).astype(np.int64)
  vals = np.array([1, 1, 1]).astype(np.int32)
  dense_shape = np.array([2, 3]).astype(np.int64)

  def _compare(self, sp_t, reduction_axes, ndims, keep_dims, do_sum):
    # [FUNCTION DOC]
    # HELPER: compares sparse reductions to a numpy reference. Parameters: sp_t is the sparse
    # tensor; reduction_axes is null (reduce all), one integer, or a list of integers (may be
    # negative); ndims is the tensor's rank (to normalize negative axes); keep_dims says
    # whether reduced dimensions are kept with size one; do_sum chooses sum (true) or max
    # (false). Densifies the input, computes the numpy answer axis by axis, then computes the
    # dense-output and sparse-output reductions in a session and asserts both are close to
    # the numpy answer. Returns nothing.
    densified = self.evaluate(sparse_ops.sparse_tensor_to_dense(sp_t))

    np_ans = densified
    if reduction_axes is None:
      if do_sum:
        np_ans = np.sum(np_ans, keepdims=keep_dims)
      else:
        np_ans = np.max(np_ans, keepdims=keep_dims)
    else:
      if not isinstance(reduction_axes, list):  # Single scalar.
        reduction_axes = [reduction_axes]
      reduction_axes = np.array(reduction_axes).astype(np.int32)
      # Handles negative axes.
      reduction_axes = (reduction_axes + ndims) % ndims
      # Loop below depends on sorted.
      reduction_axes.sort()
      for ra in reduction_axes.ravel()[::-1]:
        if do_sum:
          np_ans = np.sum(np_ans, axis=ra, keepdims=keep_dims)
        else:
          np_ans = np.max(np_ans, axis=ra, keepdims=keep_dims)

    with self.cached_session():
      if do_sum:
        tf_dense_ans = sparse_ops.sparse_reduce_sum(sp_t, reduction_axes,
                                                    keep_dims)
      else:
        tf_dense_ans = sparse_ops.sparse_reduce_max(sp_t, reduction_axes,
                                                    keep_dims)
      out_dense = self.evaluate(tf_dense_ans)

      if do_sum:
        tf_sparse_ans = sparse_ops.sparse_reduce_sum_sparse(sp_t,
                                                            reduction_axes,
                                                            keep_dims)
      else:
        tf_sparse_ans = sparse_ops.sparse_reduce_max_sparse(sp_t,
                                                            reduction_axes,
                                                            keep_dims)
      # Convert to dense for comparison purposes.
      out_sparse = sparse_ops.sparse_tensor_to_dense(tf_sparse_ans)

    self.assertAllClose(np_ans, out_dense)
    self.assertAllClose(np_ans, out_sparse)

  def _compare_all(self, sp_t, reduction_axes, ndims):
    # [FUNCTION DOC]
    # HELPER: runs the single-case comparison helper for all four combinations of keep-dimensions
    # (false, true) and sum or max, for the given sparse tensor, reduction axes and rank.
    self._compare(sp_t, reduction_axes, ndims, False, False)
    self._compare(sp_t, reduction_axes, ndims, False, True)
    self._compare(sp_t, reduction_axes, ndims, True, False)
    self._compare(sp_t, reduction_axes, ndims, True, True)

  # (TODO:b/133851381): Re-enable this test.
  def disabledtestSimpleAndRandomInputs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this test never runs because its name does not start with the test prefix
    # (disabled via a tracking note); it may be stale.
    # Intended test (CPU): skips on one buggy numpy version; compares reductions for the 2x3
    # fixture over many axis choices (all, single, lists, negatives), then for random sparsified
    # arrays of several shapes using a fixed seed, reducing over all axes and random subsets of axes.
    if np.__version__ == "1.13.0":
      self.skipTest("numpy 1.13.0 bug")

    sp_t = sparse_tensor.SparseTensor(self.ind, self.vals, self.dense_shape)

    with test_util.force_cpu():
      self._compare_all(sp_t, None, ndims=2)
      self._compare_all(sp_t, 0, ndims=2)
      self._compare_all(sp_t, [1], ndims=2)
      self._compare_all(sp_t, [0, 1], ndims=2)
      self._compare_all(sp_t, [1, 0], ndims=2)
      self._compare_all(sp_t, [-1], ndims=2)
      self._compare_all(sp_t, [1, -2], ndims=2)

    np.random.seed(1618)
    test_dims = [(1618, 1, 11, 7, 1), (1,), (1, 1, 1)]
    with test_util.force_cpu():
      for dims in test_dims:
        sp_t, unused_nnz = _sparsify(np.random.randn(*dims))
        # reduce all using None
        self._compare_all(sp_t, None, ndims=len(dims))
        # reduce random axes from 1D to N-D
        for d in range(1, len(dims) + 1):
          axes = np.random.choice(len(dims), size=d, replace=False).tolist()
          self._compare_all(sp_t, axes, ndims=len(dims))

  def testInvalidAxes(self):
    # [FUNCTION DOC]
    # Test (CPU): reduce-sum and reduce-max over the 2x3 fixture with out-of-range axes (negative
    # three and two) must raise an op error stating the invalid reduction dimension.
    sp_t = sparse_tensor.SparseTensor(self.ind, self.vals, self.dense_shape)
    with test_util.force_cpu():
      with self.assertRaisesOpError("Invalid reduction dimension -3"):
        self.evaluate(sparse_ops.sparse_reduce_sum(sp_t, -3))
      with self.assertRaisesOpError("Invalid reduction dimension 2"):
        self.evaluate(sparse_ops.sparse_reduce_sum(sp_t, 2))
      with self.assertRaisesOpError("Invalid reduction dimension -3"):
        self.evaluate(sparse_ops.sparse_reduce_max(sp_t, -3))
      with self.assertRaisesOpError("Invalid reduction dimension 2"):
        self.evaluate(sparse_ops.sparse_reduce_max(sp_t, 2))

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): with a fixed seed, for two random shapes and random
    # axis subsets (and the last axis via a negative index), checks the numerical gradient
    # error of reduce-sum with respect to the sparse values is below 1e-3.
    np.random.seed(8161)
    test_dims = [(11, 1, 5, 7, 1), (2, 2)]
    with self.session(use_gpu=False):
      for dims in test_dims:
        sp_t, nnz = _sparsify(np.random.randn(*dims))
        # reduce random axes from 1D to N-D
        for d in range(1, len(dims) + 1):
          axes = np.random.choice(len(dims), size=d, replace=False).tolist()
          reduced = sparse_ops.sparse_reduce_sum(sp_t, axes)

          err = gradient_checker.compute_gradient_error(
              sp_t.values, (nnz,), reduced,
              self.evaluate(reduced).shape)
          self.assertLess(err, 1e-3)

        # Tests for negative axes.
        reduced = sparse_ops.sparse_reduce_sum(sp_t, -1)
        err = gradient_checker.compute_gradient_error(
            sp_t.values, (nnz,), reduced,
            self.evaluate(reduced).shape)
        self.assertLess(err, 1e-3)

  def _testSparseReduceShape(self, sp_t, reduction_axes, ndims, keep_dims,
                             do_sum):
    # [FUNCTION DOC]
    # HELPER: checks the statically inferred output shape of a sparse reduction. Parameters
    # are as in the value comparison helper (sparse tensor, axes, rank, keep dimensions flag, sum
    # versus max flag). Computes the numpy reduction of the densified input and asserts its
    # shape equals the static shape of the sparse reduction's output. Returns nothing.
    densified = self.evaluate(sparse_ops.sparse_tensor_to_dense(sp_t))

    np_op = np.sum
    tf_op = sparse_ops.sparse_reduce_sum
    if not do_sum:
      np_op = np.max
      tf_op = sparse_ops.sparse_reduce_max

    np_ans = densified
    if reduction_axes is None:
      np_ans = np_op(np_ans, keepdims=keep_dims)
    else:
      if not isinstance(reduction_axes, list):  # Single scalar.
        reduction_axes = [reduction_axes]
      reduction_axes = np.array(reduction_axes).astype(np.int32)
      # Handles negative axes.
      reduction_axes = (reduction_axes + ndims) % ndims
      # Loop below depends on sorted.
      reduction_axes.sort()
      for ra in reduction_axes.ravel()[::-1]:
        np_ans = np_op(np_ans, axis=ra, keepdims=keep_dims)

    tf_ans = tf_op(sp_t, reduction_axes, keep_dims)
    self.assertAllEqual(np_ans.shape, tf_ans.get_shape().as_list())

  # (TODO:b/133851381): Re-enable this test
  def disabledtestSparseReduceSumOrMaxShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this test never runs because its name lacks the test prefix (disabled
    # pending a tracked bug).
    # Intended test (CPU): for sum and max, kept and dropped dimensions, checks the static
    # output shape of the reduction for seven axis choices on the 2x3 fixture.
    sp_t = sparse_tensor.SparseTensor(self.ind, self.vals, self.dense_shape)

    with test_util.force_cpu():
      for do_sum in [True, False]:
        for keep_dims in [True, False]:
          self._testSparseReduceShape(sp_t, None, 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, 0, 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [1], 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [0, 1], 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [1, 0], 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [-1], 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [1, -2], 2, keep_dims, do_sum)

  def testIntegerOverflow(self):
    # [FUNCTION DOC]
    # Test (CPU): calls the raw reduce-max, reduce-max-sparse and reduce-sum kernels with a dense
    # shape of 2 to the 32 by 2 to the 31 (element count overflows 64-bit integers) and asserts
    # each evaluation raises an invalid-argument error.
    with self.cached_session(use_gpu=False):
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_reduce_max(
            input_indices=[[1, 2], [3, 4]],
            input_shape=[2**32, 2**31],
            input_values=[1, 3],
            reduction_axes=[0],
            keep_dims=False,
            name=None)

        self.evaluate(res)
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_reduce_max_sparse(
            input_indices=[[1, 2], [3, 4]],
            input_shape=[2**32, 2**31],
            input_values=[1, 3],
            reduction_axes=[0],
            keep_dims=False,
            name=None)

        self.evaluate(res)
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_reduce_sum(
            input_indices=[[1, 2], [3, 4]],
            input_shape=[2**32, 2**31],
            input_values=[1, 3],
            reduction_axes=[0],
            keep_dims=False,
            name=None)

        self.evaluate(res)


class SparseMathOpsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for element-wise arithmetic between a sparse tensor and a dense tensor (multiply,
  # divide, add with broadcasting of the dense side), shape validation and gradients.

  def _check(self, result_tensor, result_np, input_sp_t):
    # [FUNCTION DOC]
    # HELPER: asserts a sparse arithmetic result is consistent with expectations. Parameters:
    # result_tensor is the computed sparse tensor, result_np is the expected dense array,
    # input_sp_t is the original sparse tensor. Asserts both are sparse tensors, that the result
    # keeps the input's indices and dense shape, and that the densified result is close
    # to the expected array (according to dtype tolerances).
    self.assertTrue(isinstance(result_tensor, sparse_tensor.SparseTensor))
    self.assertTrue(isinstance(input_sp_t, sparse_tensor.SparseTensor))
    self.assertAllCloseAccordingToType(input_sp_t.indices,
                                       result_tensor.indices)
    self.assertAllCloseAccordingToType(input_sp_t.dense_shape,
                                       result_tensor.dense_shape)

    res_densified = sparse_ops.sparse_to_dense(
        result_tensor.indices, result_tensor.dense_shape, result_tensor.values)
    self.assertAllCloseAccordingToType(result_np, res_densified)

  @test_util.run_deprecated_v1
  def testCwiseShapeValidation(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only), regression for a reported issue: multiplies a dense
    # 3 by 4 by 1 tensor by a sparse tensor of incompatible dense shape 1 by 1 by 4 by 2 and
    # asserts evaluation raises an invalid-argument error saying it broadcasts dense to
    # sparse only and got incompatible shapes.
    # Test case for GitHub 24072.
    with test_util.force_cpu():
      a = array_ops.ones([3, 4, 1], dtype=dtypes.int32)
      b = sparse_tensor.SparseTensor([[0, 0, 1, 0], [0, 0, 3, 0]], [10, 20],
                                     [1, 1, 4, 2])
      c = a * b
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "broadcasts dense to sparse only; got incompatible shapes"):
        self.evaluate(c)

  def testCwiseDivAndMul(self):
    # [FUNCTION DOC]
    # Test (CPU): with a fixed seed, for four dtypes and four sparse and dense shape pairs
    # (dense side broadcast), builds random sparse and dense operands (values at least 1), and
    # checks sparse divided by dense, sparse times dense and dense times sparse against dense
    # references. For integer dtypes also asserts true division yields 64-bit float values.
    np.random.seed(1618)
    sp_shapes = [(10, 10, 10), (5, 5), (1618,), (3, 3, 7)]
    dense_shapes = [(10, 10, 1), (5, 5), (1,), (1, 7)]

    with test_util.force_cpu():
      for dtype in [np.float32, np.float64, np.int32, np.int64]:
        for sp_shape, dense_shape in zip(sp_shapes, dense_shapes):
          sp_vals_np = np.random.rand(*sp_shape).astype(dtype) + 1
          dense_vals_np = np.random.rand(*dense_shape).astype(dtype) + 1
          sp_t, unused_nnz = _sparsify(sp_vals_np, thresh=1.5)
          sp_t_densified = sparse_ops.sparse_tensor_to_dense(sp_t)
          dense_t = constant_op.constant(dense_vals_np)

          self._check(sp_t / dense_t, sp_t_densified / dense_vals_np, sp_t)
          # Check commutative.
          self._check(sp_t * dense_t, sp_t_densified * dense_vals_np, sp_t)
          self._check(dense_t * sp_t, sp_t_densified * dense_vals_np, sp_t)

          if dtype in [np.int32, np.int64]:
            res = sp_t / dense_t  # should invoke "__truediv__"
            self.assertEqual(res.values.dtype, np.float64)

  def testCwiseAdd(self):
    # [FUNCTION DOC]
    # Test (CPU): adds a 2x2 identity-pattern sparse tensor and a dense all-ones tensor
    # (once same shape, once a broadcast shape of one element) via the sparse-dense add op, and
    # asserts the result keeps the sparse pattern with each entry equal to two.
    with test_util.force_cpu():
      # Identity(2) + AllOnes(2,2).  Should be equal to 2 * Identity(2).
      indices = [[0, 0], [1, 1]]
      vals = [1, 1]
      shape = (2, 2)

      sp_t = sparse_tensor.SparseTensor(indices, vals, shape)
      dense_t = array_ops.ones(shape, dtype=dtypes.int32)
      self._check(
          sparse_ops.sparse_dense_cwise_add(sp_t, dense_t),
          np.identity(2) * 2, sp_t)

      # Variant of above, but broadcasts the dense side.
      dense_t = array_ops.ones([1], dtype=dtypes.int32)
      self._check(
          sparse_ops.sparse_dense_cwise_add(sp_t, dense_t),
          np.identity(2) * 2, sp_t)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): for two float dtypes and four shape pairs with a
    # fixed seed, checks numerical gradient errors of sparse times dense (both operands, below
    # 1e-4) and sparse divided by dense (sparse values below 1e-4, dense operand below 2e-4).
    np.random.seed(1618)
    sp_shapes = [(10, 10, 10), (5, 5), (1618,), (3, 3, 7)]
    dense_shapes = [(10, 10, 1), (5, 5), (1,), (1, 7)]

    with self.session(use_gpu=False):
      for dtype in [np.float32, np.float64]:
        for sp_shape, dense_shape in zip(sp_shapes, dense_shapes):
          sp_vals_np = np.random.rand(*sp_shape).astype(dtype) + 1
          dense_vals_np = np.random.rand(*dense_shape).astype(dtype) + 1
          sp_t, nnz = _sparsify(sp_vals_np, thresh=1.5)
          dense_t = constant_op.constant(dense_vals_np)

          cmul = sp_t * dense_t
          err = gradient_checker.compute_gradient_error([sp_t.values, dense_t],
                                                        [(nnz,), dense_shape],
                                                        cmul.values, (nnz,))
          self.assertLess(err, 1e-4)

          cdiv = sp_t / dense_t
          err = gradient_checker.compute_gradient_error(sp_t.values, (nnz,),
                                                        cdiv.values, (nnz,))
          self.assertLess(err, 1e-4)
          err = gradient_checker.compute_gradient_error(
              dense_t,
              dense_shape,
              cdiv.values, (nnz,),
              x_init_value=dense_vals_np)
          self.assertLess(err, 2e-4)


class SparseSoftmaxTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the sparse softmax, which normalizes over the last dimension considering only
  # stored entries: equivalence with dense softmax, higher ranks, gradients, and overflow and
  # negative-shape error handling of the raw kernel.

  @test_util.run_deprecated_v1
  def testEquivalentToDensified(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): for three float dtypes, makes a random n by m matrix
    # (random sizes from a fixed seed), wraps it as a batch of one with no masking, applies
    # sparse softmax and asserts the values match dense softmax of the same data (within
    # dtype tolerance).
    np.random.seed(1618)
    n, m = np.random.choice(20, size=2)

    for dtype in [np.float16, np.float32, np.float64]:
      sp_vals_np = np.random.rand(n, m).astype(dtype)

      batched_sp_t, unused_nnz1 = _sparsify(
          sp_vals_np.reshape((1, n, m)), thresh=0.)  # No masking.

      with test_util.force_cpu():
        densified = constant_op.constant(sp_vals_np)

        sp_result = self.evaluate(
            sparse_ops.sparse_softmax(batched_sp_t)).values.reshape((n, m))
        dense_result = nn_ops.softmax(densified)

        self.assertAllCloseAccordingToType(dense_result, sp_result)

  def testHigherRanks(self):
    # [FUNCTION DOC]
    # Test (CPU): for two shapes with the same data (rank three and rank four), sparsifies a
    # fixed array, applies sparse softmax, and asserts the values are 1, 1, 1, 0.5, 0.5
    # (per-row normalization over stored entries only), indices are unchanged and the dense
    # shape is preserved. The comments inside describe the expected batches.
    # For the first shape:
    # First batch:
    # [?   e.]
    # [1.  ? ]
    # Second batch:
    # [e   ? ]
    # [e   e ]
    #
    # The softmax results should be:
    # [?   1.]     [1    ?]
    # [1.  ? ] and [.5  .5]
    # where ? means implicitly zero.
    #
    # The second shape: same input data, but with a higher-rank shape.
    shapes = [[2, 2, 2], [2, 1, 2, 2]]
    for shape in shapes:
      values = np.asarray(
          [0., np.e, 1., 0., np.e, 0., np.e, np.e]).reshape(shape)
      sp_t, unused_nnz = _sparsify(values, thresh=1e-2)
      expected_values = [1., 1., 1., .5, .5]

      with test_util.force_cpu():
        result = sparse_ops.sparse_softmax(sp_t)

        self.assertAllEqual(expected_values, result.values)
        self.assertAllEqual(sp_t.indices, result.indices)
        self.assertAllEqual(shape, result.dense_shape)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): for 32-bit and 64-bit floats, applies sparse softmax
    # to a random sparsified 2 by 5 by 10 tensor and asserts the numerical gradient error with
    # respect to the values is below 1e-4.
    x_shape = [2, 5, 10]
    with self.cached_session(use_gpu=False):
      for dtype in [np.float32, np.float64]:
        x_np = np.random.randn(*x_shape).astype(dtype)
        x_tf, nnz = _sparsify(x_np)
        y_tf = sparse_ops.sparse_softmax(x_tf)
        err = gradient_checker.compute_gradient_error(x_tf.values, (nnz,),
                                                      y_tf.values, (nnz,))
        self.assertLess(err, 1e-4)

  def testIntegerOverflow(self):
    # [FUNCTION DOC]
    # Test (CPU): calls the raw sparse softmax kernel with dense shape 2 to the 32 by 2 to the
    # 31 and asserts an invalid-argument error is raised.
    with self.cached_session(use_gpu=False):
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_softmax(
            sp_indices=[[1, 1]],
            sp_values=[2.0],
            sp_shape=[2**32, 2**31],
            name=None)

        self.evaluate(res)

  def testReshapeNegativeShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name mentions reshape, but nothing is reshaped; this appears copied from
    # the reshape tests. It actually exercises sparse softmax with a negative dense shape.
    # Test (CPU): calls the raw sparse softmax kernel with dense shape containing negative one
    # and 1, and asserts an invalid-argument error is raised.
    with self.cached_session(use_gpu=False):
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_softmax(
            sp_indices=[[1, 1]], sp_values=[2.0], sp_shape=[-1, 1], name=None)

        self.evaluate(res)


class SparseMinimumMaximumTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for element-wise minimum and maximum of two sparse tensors, including basic cases,
  # invalid raw inputs, random comparison with dense results, and shape mismatch errors.

  def _assertSparseTensorValueEqual(self, a, b):
    # [FUNCTION DOC]
    # HELPER: asserts two sparse values (or tensors) a and b have equal indices, values and
    # dense shape. Returns nothing.
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def testBasic(self):
    # [FUNCTION DOC]
    # Test (CPU): one-dimensional shape 7. First, zero versus one at the same index: maximum
    # equals the one tensor and minimum equals the zero tensor. Then zeros at different indices:
    # both maximum and minimum have two entries (zero at index zero and one) as the union.
    with test_util.force_cpu():
      # 1-D, values at index 0.
      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [7])
      max_tf = sparse_ops.sparse_maximum(sp_zero, sp_one)
      min_tf = sparse_ops.sparse_minimum(sp_zero, sp_one)
      self._assertSparseTensorValueEqual(sp_one, max_tf)
      self._assertSparseTensorValueEqual(sp_zero, min_tf)

      # Values at different indices.
      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
      sp_zero_2 = sparse_tensor.SparseTensor([[1]], [0], [7])
      expected = sparse_tensor.SparseTensor([[0], [1]], [0, 0], [7])
      max_tf = sparse_ops.sparse_maximum(sp_zero, sp_zero_2)
      min_tf = sparse_ops.sparse_minimum(sp_zero, sp_zero_2)
      self._assertSparseTensorValueEqual(expected, max_tf)
      self._assertSparseTensorValueEqual(expected, min_tf)

  def testInvalidSparseInputs(self):
    # [FUNCTION DOC]
    # Test (CPU): calls the raw sparse-sparse maximum kernel with an index matrix of rank one
    # but a shape of different rank for the second operand; asserts a value error or
    # invalid-argument error saying index rank and shape rank do not match.
    with test_util.force_cpu():
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError),
          ".*Index rank .* and shape rank .* do not match.*",
      ):
        self.evaluate(
            gen_sparse_ops.sparse_sparse_maximum(
                [[1]], [0], [2], [[]], [1], [2]
            )
        )

  @test_util.run_deprecated_v1
  def testRandom(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): with a fixed seed, for three shapes and five numeric
    # dtypes, sparsifies two random arrays (keeping values above negative 0.5), computes sparse
    # maximum and minimum, densifies the results, and asserts they equal the numpy element-wise
    # maximum and minimum of the densified inputs.
    np.random.seed(1618)
    shapes = [(13,), (6, 8), (1, 7, 1)]
    for shape in shapes:
      for dtype in [np.int32, np.int64, np.float16, np.float32, np.float64]:
        a_np = np.random.randn(*shape).astype(dtype)
        b_np = np.random.randn(*shape).astype(dtype)
        sp_a, unused_a_nnz = _sparsify(a_np, thresh=-.5)
        sp_b, unused_b_nnz = _sparsify(b_np, thresh=-.5)

        with self.cached_session(use_gpu=False):
          maximum_tf = sparse_ops.sparse_maximum(sp_a, sp_b)
          maximum_tf_densified = sparse_ops.sparse_tensor_to_dense(
              maximum_tf).eval()
          minimum_tf = sparse_ops.sparse_minimum(sp_a, sp_b)
          minimum_tf_densified = sparse_ops.sparse_tensor_to_dense(
              minimum_tf).eval()

          a_densified = sparse_ops.sparse_tensor_to_dense(sp_a).eval()
          b_densified = sparse_ops.sparse_tensor_to_dense(sp_b).eval()

        self.assertAllEqual(
            np.maximum(a_densified, b_densified), maximum_tf_densified)
        self.assertAllEqual(
            np.minimum(a_densified, b_densified), minimum_tf_densified)

  def testMismatchedShapes(self):
    # [FUNCTION DOC]
    # Test (CPU): sparse maximum of operands with different ranks must raise an op error saying
    # the operands do not have the same ranks; operands of the same rank but different
    # dense shapes (1 versus 2) must raise an op error saying the shapes do not match.
    with test_util.force_cpu():
      sp_zero = sparse_tensor.SparseTensor([[0, 0]], [0], [1, 1])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [2])
      with self.assertRaisesOpError("Operands do not have the same ranks"):
        self.evaluate(sparse_ops.sparse_maximum(sp_zero, sp_one))

      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [1])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [2])
      with self.assertRaisesOpError("Operands' shapes do not match"):
        self.evaluate(sparse_ops.sparse_maximum(sp_zero, sp_one))


class SparseTransposeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for transposing sparse tensors with an arbitrary permutation of dimensions,
  # compared against dense transposition.

  def testTranspose(self):
    # [FUNCTION DOC]
    # Test (CPU): skips on one buggy numpy version. With a fixed seed, for random shapes of rank
    # one through five and four dtypes, sparsifies a random array, transposes it with a random
    # permutation, and asserts the static shape and the densified values equal those of a dense
    # transpose.
    if np.__version__ == "1.13.0":
      self.skipTest("numpy 1.13.0 bug")

    with test_util.force_cpu():
      np.random.seed(1618)
      shapes = [np.random.randint(1, 10, size=rank) for rank in range(1, 6)]
      for shape in shapes:
        for dtype in [np.int32, np.int64, np.float32, np.float64]:
          dn_input = np.random.randn(*shape).astype(dtype)
          rank = self.evaluate(array_ops.rank(dn_input))
          perm = np.random.choice(rank, rank, False)
          sp_input, unused_a_nnz = _sparsify(dn_input)
          sp_trans = sparse_ops.sparse_transpose(sp_input, perm=perm)
          dn_trans = sparse_ops.sparse_tensor_to_dense(sp_trans)
          expected_trans = array_ops.transpose(dn_input, perm=perm)
          self.assertAllEqual(expected_trans.shape, sp_trans.get_shape())
          self.assertAllEqual(dn_trans, expected_trans)


class SparsePlaceholderTest(test.TestCase):
  # [CLASS DOC]
  # Graph-mode tests for the sparse placeholder constructor and the static shape information
  # it assigns to the dense shape and the indices matrix.

  @test_util.run_deprecated_v1
  def testPlaceholder(self):
    # [FUNCTION DOC]
    # Graph-mode test: a 32-bit float sparse placeholder with fully known shape 10 by 47 reports
    # that static shape, and its indices have shape unknown by 2.
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=(10, 47))
    self.assertAllEqual([10, 47], foo.get_shape())
    self.assertAllEqual([None, 2], foo.indices.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testPartialShapePlaceholder(self):
    # [FUNCTION DOC]
    # Graph-mode test: a sparse placeholder with partially known shape (unknown by 47) reports
    # that partial shape, and its indices have shape unknown by 2 (rank known from the shape).
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=(None, 47))
    self.assertAllEqual([None, 47], foo.get_shape().as_list())
    self.assertAllEqual([None, 2], foo.indices.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testNoShapePlaceholder(self):
    # [FUNCTION DOC]
    # Graph-mode test: a sparse placeholder with no shape information reports an unknown
    # shape, and its indices have shape unknown by unknown.
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=None)
    self.assertAllEqual(None, foo.get_shape())
    self.assertAllEqual([None, None], foo.indices.get_shape().as_list())


if __name__ == "__main__":
  googletest.main()
