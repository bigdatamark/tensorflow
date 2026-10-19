# [MODULE DOC]
# Test module for the unique and unique-with-counts array operations (v1 and axis-aware v2 kernels).
# It defines two test cases, one per operation, covering integer, string, boolean and float inputs,
# explicit output index types, axis selection, shape inference, empty input and ordering by first
# appearance. It exposes no public API and has no import-time side effects beyond imports; when run
# as a script it starts the TensorFlow test runner.
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
"""Tests for tensorflow.kernels.unique_op."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


class UniqueTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the unique operation (and its axis-aware v2 variant), which returns the distinct
  # values of a tensor in order of first appearance together with an index mapping each input
  # element to its position in the output. Has no shared fixtures beyond the base test case.

  def testInt32(self):
    # [FUNCTION DOC]
    # Scenario: 7000 random integers in a small range are passed to unique. Asserts that the index
    # output has one entry per input, the number of distinct values matches the reference count,
    # and every input element equals the distinct value its index points to. Returns nothing.
    x = np.random.randint(2, high=10, size=7000)
    y, idx = array_ops.unique(x)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testInt32OutIdxInt64(self):
    # [FUNCTION DOC]
    # Same scenario as the basic integer test but requests 64-bit integer indices via the output
    # index type argument. Asserts index length, distinct-value count and that indexing the distinct
    # values reconstructs the input. Returns nothing.
    x = np.random.randint(2, high=10, size=7000)
    y, idx = array_ops.unique(x, out_idx=dtypes.int64)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testString(self):
    # [FUNCTION DOC]
    # Scenario: a list of 7000 single-character strings (letters from a random code range) goes
    # through unique. Asserts index length, distinct count, and that each original string equals the
    # ASCII-decoded distinct value selected by its index. Returns nothing.
    indx = np.random.randint(65, high=122, size=7000)
    x = [chr(i) for i in indx]
    y, idx = array_ops.unique(x)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))

  def testInt32Axis(self):
    # [FUNCTION DOC]
    # Scenario: a 3x3 integer matrix with duplicate rows and columns is passed to the v2 unique op
    # once with axis zero and once with axis one, for both 32-bit and 64-bit axis dtypes (each in a
    # sub-test). Asserts the output rank is two, the unique slices (rows [1,0,0],[2,0,0] along
    # axis 0; columns along axis 1) and the index vectors (0,0,1 and 0,1,1). Returns nothing.
    for dtype in [np.int32, np.int64]:
      with self.subTest(dtype=dtype):
        x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]])
        y0, idx0 = gen_array_ops.unique_v2(x, axis=np.array([0], dtype))
        self.assertEqual(y0.shape.rank, 2)
        tf_y0, tf_idx0 = self.evaluate([y0, idx0])
        y1, idx1 = gen_array_ops.unique_v2(x, axis=np.array([1], dtype))
        self.assertEqual(y1.shape.rank, 2)
        tf_y1, tf_idx1 = self.evaluate([y1, idx1])
        self.assertAllEqual(tf_y0, np.array([[1, 0, 0], [2, 0, 0]]))
        self.assertAllEqual(tf_idx0, np.array([0, 0, 1]))
        self.assertAllEqual(tf_y1, np.array([[1, 0], [1, 0], [2, 0]]))
        self.assertAllEqual(tf_idx1, np.array([0, 1, 1]))

  def testInt32V2(self):
    # [FUNCTION DOC]
    # Scenario: random integers passed to the v2 unique op with an empty axis list, meaning the
    # whole flattened input is deduplicated. Asserts index length, distinct count and that the
    # distinct values reconstruct the input. Returns nothing.
    # This test is only temporary, once V2 is used
    # by default, the axis will be wrapped to allow `axis=None`.
    x = np.random.randint(2, high=10, size=7000)
    y, idx = gen_array_ops.unique_v2(x, axis=np.array([], np.int32))
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testBool(self):
    # [FUNCTION DOC]
    # Scenario: 7000 random boolean values passed to unique. Asserts index length, distinct count
    # and that the distinct values indexed by the output indices reproduce the input. Returns
    # nothing.
    x = np.random.choice([True, False], size=7000)
    y, idx = array_ops.unique(x)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testBoolV2(self):
    # [FUNCTION DOC]
    # Scenario: random booleans passed to the v2 unique op with an empty axis list. Asserts index
    # length, distinct count and reconstruction of the input from distinct values and indices.
    # Returns nothing.
    x = np.random.choice([True, False], size=7000)
    y, idx = gen_array_ops.unique_v2(x, axis=np.array([], np.int32))
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  @test_util.run_deprecated_v1
  def testShapeInferenceV2(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only, deprecated-v1 decorator): checks static shape inference of the v2
    # unique op on a 3x2x1 input. Positive and negative axes yield an index shape equal to that
    # axis' size; an empty axis on a vector yields the vector length. Asserts value errors for a
    # non-vector axis, an empty axis on a non-vector input, an axis list longer than one, and
    # out-of-range axes (3 and -4). With unknown-shape input or a placeholder axis the index shape
    # is fully unknown (a single unknown dimension). Returns nothing.
    """Test shape inference."""
    x = np.arange(6).reshape(3, 2, 1)
    _, idx = gen_array_ops.unique_v2(x, axis=[0])
    self.assertEqual(idx.shape.as_list(), [3])
    _, idx = gen_array_ops.unique_v2(x, axis=[1])
    self.assertEqual(idx.shape.as_list(), [2])
    _, idx = gen_array_ops.unique_v2(x, axis=[2])
    self.assertEqual(idx.shape.as_list(), [1])
    _, idx = gen_array_ops.unique_v2(x, axis=[-1])
    self.assertEqual(idx.shape.as_list(), [1])
    _, idx = gen_array_ops.unique_v2(x, axis=[-2])
    self.assertEqual(idx.shape.as_list(), [2])
    _, idx = gen_array_ops.unique_v2(x, axis=[-3])
    self.assertEqual(idx.shape.as_list(), [3])
    _, idx = gen_array_ops.unique_v2([0, 1, 2], axis=[])
    self.assertEqual(idx.shape.as_list(), [3])

    with self.assertRaisesRegex(ValueError, "axis expects a 1D vector"):
      gen_array_ops.unique_v2(x, axis=[[0]])

    with self.assertRaisesRegex(ValueError, "x expects a 1D vector"):
      gen_array_ops.unique_v2(x, axis=[])

    with self.assertRaisesRegex(
        ValueError, "axis does not support input tensors larger than"):
      gen_array_ops.unique_v2(x, axis=[1, 2])

    with self.assertRaisesRegex(ValueError,
                                 r"axis expects to be in the range \[-3, 3\)"):
      gen_array_ops.unique_v2(x, axis=[3])

    with self.assertRaisesRegex(ValueError,
                                 r"axis expects to be in the range \[-3, 3\)"):
      gen_array_ops.unique_v2(x, axis=[-4])

    x_t = array_ops.placeholder(dtypes.int32, shape=None)
    _, idx = gen_array_ops.unique_v2(x_t, axis=[0])
    self.assertEqual(idx.shape.as_list(), [None])

    axis_t = array_ops.placeholder(dtypes.int32, shape=None)
    _, idx = gen_array_ops.unique_v2(x, axis=axis_t)
    self.assertEqual(idx.shape.as_list(), [None])

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: unique applied to an empty integer array. Asserts the index output length and the
    # distinct-value count are both zero (matching the reference). Returns nothing.
    x = np.random.randint(2, size=0)
    y, idx = array_ops.unique(x)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))

  def testOrderedByAppearance(self):
    # [FUNCTION DOC]
    # Scenario: a fixed 20-element integer vector is deduplicated. Asserts the distinct values
    # come out in order of first appearance (not sorted) and that the index vector matches the
    # expected positions exactly. Returns nothing.
    x = np.array([3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6])
    true_y = np.array([3, 5, 4, 1, 9, 8, 6, 7, 2])
    true_idx = np.array(
        [0, 1, 0, 2, 3, 2, 4, 5, 6, 0, 1, 7, 5, 5, 2, 6, 2, 8, 1, 6])
    y, idx = array_ops.unique(x)
    tf_y, tf_idx = self.evaluate([y, idx])
    self.assertAllEqual(tf_y, true_y)
    self.assertAllEqual(tf_idx, true_idx)


class UniqueWithCountsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the unique-with-counts operation (and its axis-aware v2 variant), which returns the
  # distinct values in order of first appearance, an index mapping each input element to its
  # distinct value, and the number of occurrences of each distinct value. No shared fixtures.

  def testInt32(self):
    # [FUNCTION DOC]
    # Scenario: 7000 random integers through unique-with-counts. Asserts index length, distinct
    # count, reconstruction of the input from values and indices, and that each reported count
    # equals the number of occurrences of that value. Returns nothing.
    x = np.random.randint(2, high=10, size=7000)
    y, idx, count = array_ops.unique_with_counts(x)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testInt32OutIdxInt64(self):
    # [FUNCTION DOC]
    # Scenario: unique-with-counts on random integers with 64-bit index output requested. Asserts
    # index length, distinct count, input reconstruction and per-value counts. Returns nothing.
    x = np.random.randint(2, high=10, size=7000)
    y, idx, count = array_ops.unique_with_counts(x, out_idx=dtypes.int64)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testString(self):
    # [FUNCTION DOC]
    # Scenario: 7000 random single-character strings through unique-with-counts. Asserts index
    # length, distinct count, reconstruction of the input from ASCII-decoded distinct values, and
    # (in a sub-test per distinct value) that the reported count equals the number of matching
    # input strings. Returns nothing.
    indx = np.random.randint(65, high=122, size=7000)
    x = [chr(i) for i in indx]

    y, idx, count = array_ops.unique_with_counts(x)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))
    for value, count in zip(tf_y, tf_count):
      with self.subTest(value=value, count=count):
        v = [1 if x[i] == value.decode('ascii') else 0 for i in range(7000)]
        self.assertEqual(count, sum(v))

  def testInt32Axis(self):
    # [FUNCTION DOC]
    # Scenario: a 3x3 integer matrix with duplicate rows/columns goes through the v2
    # unique-with-counts op along axis zero and axis one, for 32-bit and 64-bit axis dtypes (one
    # sub-test each). Asserts rank two outputs, the unique slices, the index vectors and the
    # counts (2,1 along axis 0; 1,2 along axis 1). Returns nothing.
    for dtype in [np.int32, np.int64]:
      with self.subTest(dtype=dtype):
        x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]])
        y0, idx0, count0 = gen_array_ops.unique_with_counts_v2(
            x, axis=np.array([0], dtype))
        self.assertEqual(y0.shape.rank, 2)
        tf_y0, tf_idx0, tf_count0 = self.evaluate([y0, idx0, count0])
        y1, idx1, count1 = gen_array_ops.unique_with_counts_v2(
            x, axis=np.array([1], dtype))
        self.assertEqual(y1.shape.rank, 2)
        tf_y1, tf_idx1, tf_count1 = self.evaluate([y1, idx1, count1])
        self.assertAllEqual(tf_y0, np.array([[1, 0, 0], [2, 0, 0]]))
        self.assertAllEqual(tf_idx0, np.array([0, 0, 1]))
        self.assertAllEqual(tf_count0, np.array([2, 1]))
        self.assertAllEqual(tf_y1, np.array([[1, 0], [1, 0], [2, 0]]))
        self.assertAllEqual(tf_idx1, np.array([0, 1, 1]))
        self.assertAllEqual(tf_count1, np.array([1, 2]))

  def testInt32V2(self):
    # [FUNCTION DOC]
    # Scenario: random integers through the v2 unique-with-counts op with an empty axis list
    # (flattened input). Asserts index length, distinct count, input reconstruction and per-value
    # counts. Returns nothing.
    # This test is only temporary, once V2 is used
    # by default, the axis will be wrapped to allow `axis=None`.
    x = np.random.randint(2, high=10, size=7000)
    y, idx, count = gen_array_ops.unique_with_counts_v2(
        x, axis=np.array([], np.int32))
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testBool(self):
    # [FUNCTION DOC]
    # Scenario: 7000 random booleans through unique-with-counts. Asserts index length, distinct
    # count, input reconstruction and that counts match the occurrences of each value. Returns
    # nothing.
    x = np.random.choice([True, False], size=7000)
    y, idx, count = array_ops.unique_with_counts(x)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testBoolV2(self):
    # [FUNCTION DOC]
    # Scenario: random booleans through the v2 unique-with-counts op with an empty axis list.
    # Asserts index length, distinct count, input reconstruction and per-value counts. Returns
    # nothing.
    x = np.random.choice([True, False], size=7000)
    y, idx, count = gen_array_ops.unique_with_counts_v2(
        x, axis=np.array([], np.int32))
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testFloat(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertion comparing the distinct-value count against the numpy reference is
    # commented out because newer numpy treats NaNs as equal, so that check is not performed.
    # Scenario: a float list containing two NaNs goes through the v2 unique-with-counts op with an
    # empty axis list. Asserts index length; each NaN input maps to a NaN distinct value, other
    # values map to equal values, and every NaN has count one (each NaN is its own key) while other
    # counts match the occurrences. Returns nothing.
    # NOTE(mrry): The behavior when a key is NaN is inherited from
    # `std::unordered_map<float, ...>`: each NaN becomes a unique key in the
    # map.
    x = [0.0, 1.0, np.nan, np.nan]
    y, idx, count = gen_array_ops.unique_with_counts_v2(
        x, axis=np.array([], np.int32))
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    # TODO(b/202197513): numpy>=1.20.0 makes np.unique() treat np.nan as equal
    # so len(np.unique(x)) == 3. So this no longer can be tested this way.
    # self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      if np.isnan(x[i]):
        self.assertTrue(np.isnan(tf_y[tf_idx[i]]))
      else:
        self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      if np.isnan(value):
        self.assertEqual(count, 1)
      else:
        self.assertEqual(count, np.sum(x == value))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: unique-with-counts on an empty integer array. Asserts that the index, distinct
    # value and count outputs all have shape of a single zero-length dimension. Returns nothing.
    x = np.random.randint(2, size=0)
    y, idx, count = array_ops.unique_with_counts(x)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(tf_idx.shape, (0,))
    self.assertEqual(tf_y.shape, (0,))
    self.assertEqual(tf_count.shape, (0,))

  def testOrderedByAppearance(self):
    # [FUNCTION DOC]
    # Scenario: a fixed 20-element integer vector through unique-with-counts. Asserts distinct
    # values in first-appearance order, the exact index vector and the exact count vector. Returns
    # nothing.
    x = np.array([3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6])
    true_y = np.array([3, 5, 4, 1, 9, 8, 6, 7, 2])
    true_idx = np.array(
        [0, 1, 0, 2, 3, 2, 4, 5, 6, 0, 1, 7, 5, 5, 2, 6, 2, 8, 1, 6])
    true_count = np.array([3, 3, 4, 1, 1, 3, 3, 1, 1])
    y, idx, count = array_ops.unique_with_counts(x)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])
    self.assertAllEqual(tf_y, true_y)
    self.assertAllEqual(tf_idx, true_idx)
    self.assertAllEqual(tf_count, true_count)


if __name__ == '__main__':
  test.main()
