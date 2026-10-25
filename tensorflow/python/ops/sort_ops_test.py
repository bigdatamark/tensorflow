# [MODULE DOC]
# Unit tests for the sort wrapper module (sort_ops): sort and argsort on tensors. Covers random
# arrays of low and high rank, integer and unsigned integer extremes, negative axes, descending
# order, stable argsort, scalar and out-of-bounds axis errors, constant transposition for
# statically known rank, and static shape propagation of argsort. Compares against numpy results.
# No import-time side effects other than imports; running the file as a script starts the test main.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the sort wrapper."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.platform import test


class SortTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering sort_ops.sort and sort_ops.argsort. Holds no state; provides helpers that
  # build random numpy arrays and compare sort results with numpy's sort (flipped for descending).

  def random_array(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: builds random input data for the sort tests.
    # Inputs: shape (list of dimension sizes) and dtype (numpy type). For integer types returns
    # random integers spanning the full range of the type (note the upper bound is exclusive);
    # otherwise returns uniform random values in the unit interval cast to dtype. Returns a numpy
    # array. Uses the global numpy random state; raises nothing itself.
    if np.issubdtype(dtype, np.integer):
      imin = np.iinfo(dtype).min
      imax = np.iinfo(dtype).max
      return np.random.randint(imin, imax, shape, dtype)
    else:
      return np.random.random(shape).astype(dtype)

  def _test_sort(self, values, axis, direction):
    # [FUNCTION DOC]
    # HELPER: shared assertion used by the sort tests.
    # Computes the expected result with numpy sorting along the axis (reversed along that axis
    # when direction is descending) and asserts that sort_ops.sort on a constant tensor built from
    # values, with the same axis and direction, is equal. Returns nothing; fails the test on mismatch.
    expected = np.sort(values, axis=axis)
    if direction == 'DESCENDING':
      expected = np.flip(expected, axis=axis)
    self.assertAllEqual(
        expected,
        sort_ops.sort(
            constant_op.constant(values), axis=axis, direction=direction))

  def testRandom_lowDimensionality(self):
    # [FUNCTION DOC]
    # Test: runs the low-dimensionality random sort check with non-negative axes, float32 data and
    # ascending order. Expects sort results to match numpy.
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=np.float32, direction='ASCENDING')

  def testRandom_lowDimensionality_negative(self):
    # [FUNCTION DOC]
    # Test: same as the low-dimensionality random check but the sort axis is converted to a
    # negative index counted from the end. float32 data, ascending. Expects numpy-equal results.
    self._testRandom_lowDimensionality(
        negative_axis=True, dtype=np.float32, direction='ASCENDING')

  def _testRandom_lowDimensionality(self, negative_axis, dtype, direction):
    # [FUNCTION DOC]
    # HELPER: shared body for the low-rank random sort tests.
    # Inputs: negative_axis (boolean, whether to turn the chosen axis into a negative index),
    # dtype (numpy type), direction (ascending or descending string). Seeds numpy with a fixed
    # value, then 20 times builds a random array of rank 1 or 2 with dimension sizes 0 to 19,
    # picks a random axis, and checks sort against numpy inside a cached session. Returns nothing.
    np.random.seed(42)
    for _ in range(20):
      rank = np.random.randint(1, 3)
      shape = [np.random.randint(0, 20) for _ in range(rank)]
      arr = self.random_array(shape, dtype)
      sort_axis = np.random.choice(rank)
      if negative_axis:
        sort_axis = -1 - sort_axis
      with self.cached_session():
        self._test_sort(arr, sort_axis, direction)

  def testRandom_highDimensionality(self):
    # [FUNCTION DOC]
    # Test: runs the high-rank random sort check using float32 data. Expects numpy-equal results.
    self._testRandom_highDimensionality(np.float32)

  def _testRandom_highDimensionality(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body for the high-rank random sort test.
    # Input: dtype (numpy type). Seeds numpy with a fixed value, then 20 times builds a random
    # array of rank 5 to 14 with small dimensions (1 to 3), picks a random axis and checks an
    # ascending sort against numpy in a cached session. Returns nothing.
    np.random.seed(100)
    for _ in range(20):
      rank = np.random.randint(5, 15)
      shape = [np.random.randint(1, 4) for _ in range(rank)]
      arr = self.random_array(shape, dtype)
      sort_axis = np.random.choice(rank)
      with self.cached_session():
        self._test_sort(arr, sort_axis, 'ASCENDING')

  def testIntArray(self):
    # [FUNCTION DOC]
    # Test: sorts signed 64-bit integer data. Runs the low-rank random check in both ascending and
    # descending order, then (unless address or undefined-behavior sanitizers are enabled, which
    # is a skipped case tracked by a bug reference) sorts ten evenly spaced values spanning the
    # full int64 range in both directions. Expects numpy-equal results.
    dtype = np.int64
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=dtype, direction='ASCENDING')
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=dtype, direction='DESCENDING')

    # TODO(b/190410105) re-enable test once proper sort kernel is added.
    if not test_util.is_asan_enabled() and not test_util.is_ubsan_enabled():
      edges = np.linspace(
          np.iinfo(dtype).min, np.iinfo(dtype).max, 10, dtype=dtype)
      self._test_sort(edges, 0, 'ASCENDING')
      self._test_sort(edges, 0, 'DESCENDING')

  def testUIntArray(self):
    # [FUNCTION DOC]
    # Test: sorts unsigned 64-bit integer data. Runs the low-rank random check in both directions,
    # then sorts ten evenly spaced values spanning the full uint64 range in both directions
    # (not guarded by the sanitizer check used in the signed test). Expects numpy-equal results.
    dtype = np.uint64
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=dtype, direction='ASCENDING')
    self._testRandom_lowDimensionality(
        negative_axis=False, dtype=dtype, direction='DESCENDING')
    edges = np.linspace(
        np.iinfo(dtype).min, np.iinfo(dtype).max, 10, dtype=dtype)
    self._test_sort(edges, 0, 'ASCENDING')
    self._test_sort(edges, 0, 'DESCENDING')

  def testScalar(self):
    # [FUNCTION DOC]
    # Test: sorting a scalar whose static shape is unknown (built via a zeros tensor with a random
    # length-1 shape input, giving an empty shape at run time). Expects either a value error or an
    # invalid-argument error whose message mentions being out of bounds when evaluated.
    # Create an empty scalar where the static shape is unknown.
    zeros_length_1 = array_ops.zeros(
        random_ops.random_uniform([1], minval=0, maxval=1, dtype=dtypes.int32),
        dtype=dtypes.int32)
    scalar = array_ops.zeros(zeros_length_1)
    with self.cached_session():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  'out of bounds'):
        self.evaluate(sort_ops.sort(scalar))

  def testNegativeOutOfBounds_staticShape(self):
    # [FUNCTION DOC]
    # Test: sorting a length-3 constant vector along axis -4 (beyond the rank). Expects a value
    # error or invalid-argument error whose message mentions a slice index being out of bounds.
    arr = constant_op.constant([3, 4, 5])
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                'slice index .* out of bounds'):
      self.evaluate(sort_ops.sort(arr, axis=-4))

  def testDescending(self):
    # [FUNCTION DOC]
    # Test: descending sort along axis 0 of a random 10x5x5 array. Expects the result to equal
    # numpy's ascending sort reversed along the first axis.
    arr = np.random.random((10, 5, 5))
    with self.cached_session():
      self.assertAllEqual(
          np.sort(arr, axis=0)[::-1],
          sort_ops.sort(
              constant_op.constant(arr), axis=0, direction='DESCENDING'))

  def testSort_staticallyKnownRank_constantTransposition(self):
    # [FUNCTION DOC]
    # Test (graph mode): builds a random tensor whose rank is statically 5 but whose dimension
    # sizes are unknown, sorts it along axis 1, then looks up the graph tensor named for the
    # transposition permutation. Expects its value to be statically computable as a constant and
    # equal to the permutation that swaps axis 1 with the last axis (0, 4, 2, 3, 1).
    with ops.Graph().as_default():
      # The transposition array should be a constant if the rank of "values" is
      # statically known.
      tensor = random_ops.random_uniform(
          # Rank is statically known to be 5, but the dimension lengths are not
          # known.
          random_ops.random_uniform(
              shape=(5,), minval=0, maxval=10, dtype=dtypes.int32))
      sort_ops.sort(tensor, axis=1)
      transposition = (
          ops.get_default_graph().get_tensor_by_name('sort/transposition:0'))
      self.assertIsNot(tensor_util.constant_value(transposition), None)
      self.assertAllEqual(
          # Swaps "1" and "4" to put "1" at the end.
          tensor_util.constant_value(transposition),
          [0, 4, 2, 3, 1])

  def testArgsort_1d(self):
    # [FUNCTION DOC]
    # Test: argsort of a random vector of 42 values. Gathering the input with the returned indices
    # must equal numpy's sorted array.
    arr = np.random.random(42)
    with self.cached_session():
      self.assertAllEqual(
          np.sort(arr), array_ops.gather(arr, sort_ops.argsort(arr)))

  def testArgsortStable(self):
    # [FUNCTION DOC]
    # Test: stable argsort of a vector with duplicate values. Expects ties to keep original order,
    # giving the fixed index lists for ascending and for descending direction.
    arr = constant_op.constant([1, 5, 2, 2, 3], dtype=dtypes.int32)
    ascending = [0, 2, 3, 4, 1]
    descending = [1, 4, 2, 3, 0]
    with self.cached_session():
      self.assertAllEqual(
          sort_ops.argsort(arr, direction='ASCENDING', stable=True), ascending)
      self.assertAllEqual(
          sort_ops.argsort(arr, direction='DESCENDING', stable=True),
          descending)

  def testArgsort(self):
    # [FUNCTION DOC]
    # Test: argsort of a random 4-D array along each of its four axes. Expects results equal to
    # numpy's argsort along the same axis.
    arr = np.random.random((5, 6, 7, 8))
    for axis in range(4):
      with self.cached_session():
        self.assertAllEqual(
            np.argsort(arr, axis=axis), sort_ops.argsort(arr, axis=axis))

  def testArgsortTensorShape(self):
    # [FUNCTION DOC]
    # Test (graph mode): argsort of a float placeholder with partially unknown shape, for each
    # axis. Expects the static shape of the result to equal the placeholder's static shape.
    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(dtypes.float32, shape=[1, None, 5])
      for axis in range(3):
        with self.cached_session():
          self.assertAllEqual(
              placeholder.shape.as_list(),
              sort_ops.argsort(placeholder, axis=axis).shape.as_list())


if __name__ == '__main__':
  test.main()
