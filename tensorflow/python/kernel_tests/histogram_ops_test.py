# [MODULE DOC]
# Tests for the histogram ops module: the fixed-width binning op (assigns each value a bin index)
# and the fixed-width histogram op (counts values per bin). Covers empty, 1-D and 2-D inputs,
# output dtypes, invalid ranges or bin counts, shape inference, and overflow/large-range cases.
# Defines two test classes and runs the TensorFlow test main when executed as a script.
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
"""Tests for tensorflow.ops.histogram_ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import histogram_ops
from tensorflow.python.platform import test


class BinValuesFixedWidth(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the fixed-width binning op, which maps each value to an integer bin index (values
  # below the range go to the first bin, values above to the last). Some tests are parameterized
  # over floating-point dtypes. No shared fixtures.

  def test_empty_input_gives_all_zero_counts(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "all zero counts" but the op returns bin indices, not counts; the
    # test actually checks that an empty input yields an empty result.
    # Bins five-bin binning of an empty list over the range zero to five and asserts the output
    # dtype is 32-bit integer and the result is empty. Returns nothing.
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = []
    expected_bins = []
    with self.cached_session():
      bins = histogram_ops.histogram_fixed_width_bins(
          values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, bins.dtype)
      self.assertAllClose(expected_bins, self.evaluate(bins))

  @parameterized.parameters(
      np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
  )
  def test_1d_values_int32_output(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over 32-bit float, 64-bit float and bfloat16 numpy types (dtype argument).
    # Bins a 1-D array of six values, including out-of-range ones, into five bins over zero to
    # five; asserts the output dtype is 32-bit integer and indices are as expected (out-of-range
    # values clamp to the first and last bins). Returns nothing.
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = np.array([0.0, 5.0]).astype(dtype)
    values = np.array([-1.0, 0.0, 1.5, 2.0, 5.0, 15]).astype(dtype)
    expected_bins = [0, 0, 1, 2, 4, 4]
    with self.cached_session():
      bins = histogram_ops.histogram_fixed_width_bins(
          values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, bins.dtype)
      self.assertAllClose(expected_bins, self.evaluate(bins))

  def test_2d_values(self):
    # [FUNCTION DOC]
    # Bins a 2-by-3 tensor into five bins over zero to five and asserts the output keeps the
    # 2-D shape, has 32-bit integer dtype, and holds the expected bin indices. Returns nothing.
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = constant_op.constant(
        [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]], shape=(2, 3))
    expected_bins = [[0, 0, 1], [2, 4, 4]]
    with self.cached_session():
      bins = histogram_ops.histogram_fixed_width_bins(
          values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, bins.dtype)
      self.assertAllClose(expected_bins, self.evaluate(bins))

  def test_negative_nbins(self):
    # [FUNCTION DOC]
    # Asserts that requesting a negative number of bins raises an invalid-argument or value error
    # whose message says the count must be greater than zero. Returns nothing.
    value_range = [0.0, 5.0]
    values = []
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "must > 0"):
      with self.session():
        bins = histogram_ops.histogram_fixed_width_bins(
            values, value_range, nbins=-1)
        self.evaluate(bins)


class HistogramFixedWidthTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the fixed-width histogram op, which counts how many values fall in each of a given
  # number of equal-width bins over a value range. Covers argument validation, dtypes, shapes,
  # static shape inference and numerically extreme ranges.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so the base test case setup is skipped, and the
    # random generator created here is never used by any test in this class.
    # Creates a seeded (zero) numpy random state and stores it as an attribute. Returns nothing.
    self.rng = np.random.RandomState(0)

  def test_with_invalid_value_range(self):
    # [FUNCTION DOC]
    # Asserts errors for bad value_range arguments: a scalar (not rank one) and a three-element
    # vector (not two elements) each raise an invalid-argument or value error with the matching
    # message. Returns nothing.
    values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Shape must be rank 1 but is rank 0|should be a vector"):
      self.evaluate(histogram_ops.histogram_fixed_width(values, 1.0))
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Dimension must be 2 but is 3|should be a vector of 2 elements"):
      self.evaluate(
          histogram_ops.histogram_fixed_width(values, [1.0, 2.0, 3.0]))

  def test_with_invalid_nbins(self):
    # [FUNCTION DOC]
    # Asserts errors for bad nbins arguments: a vector instead of a scalar, and a negative number,
    # each raise an invalid-argument or value error with the matching message. Returns nothing.
    values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Shape must be rank 0 but is rank 1|should be a scalar"):
      self.evaluate(
          histogram_ops.histogram_fixed_width(values, [1.0, 5.0], nbins=[1, 2]))
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Requires nbins > 0|should be a positive number"):
      self.evaluate(
          histogram_ops.histogram_fixed_width(values, [1.0, 5.0], nbins=-5))

  def test_empty_input_gives_all_zero_counts(self):
    # [FUNCTION DOC]
    # Builds a five-bin histogram of an empty list and asserts the dtype is 32-bit integer and all
    # five counts are zero. Returns nothing.
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = []
    expected_bin_counts = [0, 0, 0, 0, 0]
    hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
    self.assertEqual(dtypes.int32, hist.dtype)
    self.assertAllClose(expected_bin_counts, self.evaluate(hist))

  def test_1d_values_int64_output(self):
    # [FUNCTION DOC]
    # Builds a five-bin histogram of six values requesting 64-bit integer output; asserts the
    # dtype is 64-bit integer and the counts are as expected. Returns nothing.
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    expected_bin_counts = [2, 1, 1, 0, 2]
    hist = histogram_ops.histogram_fixed_width(
        values, value_range, nbins=5, dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, hist.dtype)
    self.assertAllClose(expected_bin_counts, self.evaluate(hist))

  def test_1d_float64_values(self):
    # [FUNCTION DOC]
    # Uses 64-bit float numpy inputs for values and range; asserts the default output dtype is
    # still 32-bit integer and the counts are as expected. Returns nothing.
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = np.float64([0.0, 5.0])
    values = np.float64([-1.0, 0.0, 1.5, 2.0, 5.0, 15])
    expected_bin_counts = [2, 1, 1, 0, 2]
    hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
    self.assertEqual(dtypes.int32, hist.dtype)
    self.assertAllClose(expected_bin_counts, self.evaluate(hist))

  def test_2d_values(self):
    # [FUNCTION DOC]
    # Builds a five-bin histogram of a nested 2-by-3 list; asserts the values are flattened into a
    # single histogram with 32-bit integer dtype and the expected counts. Returns nothing.
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]]
    expected_bin_counts = [2, 1, 1, 0, 2]
    hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
    self.assertEqual(dtypes.int32, hist.dtype)
    self.assertAllClose(expected_bin_counts, self.evaluate(hist))

  @test_util.run_deprecated_v1
  def test_shape_inference(self):
    # [FUNCTION DOC]
    # Graph-mode only (deprecated v1 decorator). Checks static shapes: with a constant bin count
    # the histogram shape is a vector of five; with a placeholder bin count it is rank one with
    # unknown length, and feeding five yields the expected counts. Dtype is 32-bit integer.
    value_range = [0.0, 5.0]
    values = [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]]
    expected_bin_counts = [2, 1, 1, 0, 2]
    placeholder = array_ops.placeholder(dtypes.int32)
    with self.session():
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
      self.assertAllEqual(hist.shape.as_list(), (5,))
      self.assertEqual(dtypes.int32, hist.dtype)
      self.assertAllClose(expected_bin_counts, self.evaluate(hist))

      hist = histogram_ops.histogram_fixed_width(
          values, value_range, nbins=placeholder)
      self.assertEqual(hist.shape.ndims, 1)
      self.assertIs(hist.shape.dims[0].value, None)
      self.assertEqual(dtypes.int32, hist.dtype)
      self.assertAllClose(expected_bin_counts, hist.eval({placeholder: 5}))

  def test_single_bin(self):
    # [FUNCTION DOC]
    # With a single bin and float values near the 32-bit float maximum, asserts both values are
    # counted in that bin (no overflow in range arithmetic). Returns nothing.
    hist = histogram_ops.histogram_fixed_width(
        values=constant_op.constant([3e+38, 100], dtype=dtypes.float32),
        value_range=constant_op.constant([-1e+38, 3e+38]),
        nbins=1)
    self.assertAllEqual(hist, [2])

  def test_range_overflow(self):
    # [FUNCTION DOC]
    # With a range whose width exceeds the 32-bit float maximum and two bins, asserts the two
    # values land in separate bins (one each), i.e. the width computation does not overflow.
    hist = histogram_ops.histogram_fixed_width(
        values=constant_op.constant([3e+38, 100], dtype=dtypes.float32),
        value_range=constant_op.constant([-1e+38, 3e+38]),
        nbins=2)
    self.assertAllEqual(hist, [1, 1])

  def test_large_range(self):
    # [FUNCTION DOC]
    # Uses 32-bit integer values and range spanning the full integer range with two bins; asserts
    # the minimum and maximum values fall into separate bins (one each). Returns nothing.
    hist = histogram_ops.histogram_fixed_width(
        values=constant_op.constant(
            [-(2**31), 2**31 - 1], dtype=dtypes.int32
        ),
        value_range=constant_op.constant(
            [-(2**31), 2**31 - 1], dtype=dtypes.int32
        ),
        nbins=2,
    )
    self.assertAllEqual(hist, [1, 1])


if __name__ == '__main__':
  test.main()
