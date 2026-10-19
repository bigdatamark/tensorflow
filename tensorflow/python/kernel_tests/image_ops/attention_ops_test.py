# [MODULE DOC]
# Unit tests for the glimpse extraction image operations (the public extract_glimpse in its
# original and v2 forms and the raw generated op kept for compatibility with old saved graphs).
# Defines a single test case class that checks output shapes, the values picked out of the input
# for centered, off-center, out-of-bounds and fully non-overlapping glimpses, the zero-noise
# padding behavior, empty inputs and invalid negative sizes. No import-time side effects other
# than the usual test main entry point when run as a script.
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
"""Tests for image.extract_glimpse()."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


class ExtractGlimpseTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the glimpse extraction ops. It has one shared verification helper that
  # builds row-index and column-index images, extracts glimpses from both, and compares them
  # against expected row and column numbers; the individual tests choose image size, glimpse
  # size and offsets, plus separate tests for zero noise, non-normalized coordinates, empty
  # input and invalid input.

  def _VerifyValues(self, tensor_in_sizes, glimpse_sizes, offsets,
                    expected_rows, expected_cols):
    # [FUNCTION DOC]
    # HELPER: shared verification routine for the glimpse extraction tests in this class.
    # Inputs: tensor_in_sizes is a pair of input rows and columns; glimpse_sizes is a pair of
    # glimpse rows and columns; offsets is a pair of row and column offsets of the glimpse
    # center (relative, centered, normalized coordinates); expected_rows and expected_cols are
    # lists of expected row numbers / column numbers per glimpse position, where a null entry
    # means the position lies outside the image and may hold any random value in range.
    # Builds one image whose values are the row number and one whose values are the column
    # number, extracts a glimpse from each (swapping row/column order to the height/width
    # convention), evaluates both, and asserts the glimpse shape equals the requested size and
    # every entry equals the expectation (or lies between zero and the larger image dimension
    # for out-of-bounds entries). Returns nothing; fails through assertions.
    """Verifies the output values of the glimpse extraction kernel.

    Args:
      tensor_in_sizes: Input tensor dimensions in [input_rows, input_cols].
      glimpse_sizes: Dimensions of the glimpse in [glimpse_rows, glimpse_cols].
      offsets: Relative location of the center of the glimpse in the input
        image expressed as [row_offset, col_offset].
      expected_rows: A list containing the expected row numbers (None for
         out of bound entries that are expected to be replaced by uniform
         random entries in [0,1) ).
      expected_cols: Same as expected_rows, but for column numbers.
    """

    rows = tensor_in_sizes[0]
    cols = tensor_in_sizes[1]
    # Row Tensor with entries by row.
    # [[ 1 1 1 ... ]
    #  [ 2 2 2 ... ]
    #  [ 3 3 3 ... ]
    #  [ ...
    # ]
    t_rows = array_ops.tile(
        [[1.0 * r] for r in range(1, rows + 1)], [1, cols], name='tile_rows')

    # Shuffle to switch to a convention of (batch_size, height, width, depth).
    t_rows_4d = array_ops.transpose(
        array_ops.expand_dims(array_ops.expand_dims(t_rows, 0), 3),
        [0, 2, 1, 3])

    # Column Tensor with entries by column.
    # [[ 1 2 3 4 ... ]
    #  [ 1 2 3 4 ... ]
    #  [ 1 2 3 4 ... ]
    #  [ ...         ]
    # ]
    t_cols = array_ops.tile(
        [[1.0 * r for r in range(1, cols + 1)]], [rows, 1], name='tile_cols')

    # Shuffle to switch to a convention of (batch_size, height, width, depth).
    t_cols_4d = array_ops.transpose(
        array_ops.expand_dims(array_ops.expand_dims(t_cols, 0), 3),
        [0, 2, 1, 3])

    # extract_glimpses from Row and Column Tensor, respectively.
    # Switch order for glimpse_sizes and offsets to switch from (row, col)
    # convention to tensorflows (height, width) convention.
    t1 = constant_op.constant([glimpse_sizes[1], glimpse_sizes[0]], shape=[2])
    t2 = constant_op.constant([offsets[1], offsets[0]], shape=[1, 2])
    glimpse_rows = (array_ops.transpose(
        image_ops.extract_glimpse(t_rows_4d, t1, t2), [0, 2, 1, 3]))
    glimpse_cols = (array_ops.transpose(
        image_ops.extract_glimpse(t_cols_4d, t1, t2), [0, 2, 1, 3]))

    # Evaluate the TensorFlow Graph.
    with self.cached_session() as sess:
      value_rows, value_cols = self.evaluate([glimpse_rows, glimpse_cols])

    # Check dimensions of returned glimpse.
    self.assertEqual(value_rows.shape[1], glimpse_sizes[0])
    self.assertEqual(value_rows.shape[2], glimpse_sizes[1])
    self.assertEqual(value_cols.shape[1], glimpse_sizes[0])
    self.assertEqual(value_cols.shape[2], glimpse_sizes[1])

    # Check entries.
    min_random_val = 0
    max_random_val = max(rows, cols)
    for i in range(glimpse_sizes[0]):
      for j in range(glimpse_sizes[1]):
        if expected_rows[i] is None or expected_cols[j] is None:
          self.assertGreaterEqual(value_rows[0][i][j][0], min_random_val)
          self.assertLessEqual(value_rows[0][i][j][0], max_random_val)
          self.assertGreaterEqual(value_cols[0][i][j][0], min_random_val)
          self.assertLessEqual(value_cols[0][i][j][0], max_random_val)
        else:
          self.assertEqual(value_rows[0][i][j][0], expected_rows[i])
          self.assertEqual(value_cols[0][i][j][0], expected_cols[j])

  def testCenterGlimpse(self):
    # [FUNCTION DOC]
    # Test: a small glimpse taken at the exact image center (zero offsets) of a 41 by 61 image
    # returns the middle rows and columns; asserted through the shared verification helper.
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[3, 5],
        offsets=[0.0, 0.0],
        expected_rows=[20, 21, 22],
        expected_cols=[29, 30, 31, 32, 33])

  def testEmptyTensor(self):
    # [FUNCTION DOC]
    # Test: extracting a one by one glimpse from an image batch with zero batch entries and zero
    # depth, with an empty offsets array, must succeed and return an empty float32 result of
    # shape zero, one, one, zero.
    empty_image = np.zeros((0, 4, 3, 0))
    offsets = np.zeros((0, 2))
    with self.cached_session():
      result = image_ops.extract_glimpse(empty_image, [1, 1], offsets)
      self.assertAllEqual(
          np.zeros((0, 1, 1, 0), dtype=np.float32), self.evaluate(result))

  def testLargeCenterGlimpse(self):
    # [FUNCTION DOC]
    # Test: a centered glimpse exactly as large as the image returns every row (1 to 41) and
    # every column (1 to 61) in order, checked with the shared verification helper.
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[41, 61],
        offsets=[0.0, 0.0],
        expected_rows=list(range(1, 42)),
        expected_cols=list(range(1, 62)))

  def testTooLargeCenterGlimpse(self):
    # [FUNCTION DOC]
    # Test: a centered glimpse two larger than the image in each dimension returns the whole
    # image padded by one out-of-bounds (random-valued) entry on each side; the helper accepts
    # any in-range value for those positions.
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[43, 63],
        offsets=[0.0, 0.0],
        expected_rows=[None] + list(range(1, 42)) + [None],
        expected_cols=[None] + list(range(1, 62)) + [None])

  def testGlimpseFullOverlap(self):
    # [FUNCTION DOC]
    # Test: a small glimpse at a positive off-center offset lies completely inside the image and
    # returns the expected shifted rows (22 to 24) and columns (38 to 42).
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[3, 5],
        offsets=[0.1, 0.3],
        expected_rows=[22, 23, 24],
        expected_cols=[38, 39, 40, 41, 42])

  def testGlimpseFullOverlap2(self):
    # [FUNCTION DOC]
    # Test: a glimpse with negative offsets near the top-left that still lies completely inside
    # the image returns rows 1 to 11 and columns 8 to 10.
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[11, 3],
        offsets=[-0.7, -0.7],
        expected_rows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        expected_cols=[8, 9, 10])

  def testGlimpseBeforeLeftMargin(self):
    # [FUNCTION DOC]
    # Test: a glimpse whose column offset places it just against the left edge (without
    # extending past it) returns rows 1 to 11 and columns 1 to 5.
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[11, 5],
        offsets=[-0.7, -0.9],
        expected_rows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        expected_cols=[1, 2, 3, 4, 5])

  def testGlimpseLowerRightCorner(self):
    # [FUNCTION DOC]
    # Test: a glimpse centered on the lower-right corner (offsets one, one) returns the last
    # rows and columns of the image followed by out-of-bounds entries, which only need to lie in
    # the permitted random range.
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[7, 5],
        offsets=[1.0, 1.0],
        expected_rows=[38, 39, 40, 41, None, None, None],
        expected_cols=[59, 60, 61, None, None])

  def testGlimpseNoOverlap(self):
    # [FUNCTION DOC]
    # Test: a glimpse positioned entirely outside a 20 by 30 image (offsets beyond the border)
    # has no overlap, so every entry is an out-of-bounds value within the permitted range.
    self._VerifyValues(
        tensor_in_sizes=[20, 30],
        glimpse_sizes=[3, 3],
        offsets=[-2.0, 2.0],
        expected_rows=[None, None, None],
        expected_cols=[None, None, None])

  def testGlimpseOnLeftMargin(self):
    # [FUNCTION DOC]
    # Test: a glimpse centered on the left border (column offset of minus one) is half outside
    # the image: the first three columns are out-of-bounds entries and the rest are columns 1 to
    # 4; rows 1 to 11 are returned in full.
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[11, 7],
        offsets=[-0.7, -1.0],
        expected_rows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        expected_cols=[None, None, None, 1, 2, 3, 4])

  def testGlimpseUpperMargin(self):
    # [FUNCTION DOC]
    # Test: a glimpse centered on the upper border (offset minus one in the first coordinate)
    # has three out-of-bounds leading entries followed by rows 1 to 4, with columns 56 to 60.
    self._VerifyValues(
        tensor_in_sizes=[41, 61],
        glimpse_sizes=[7, 5],
        offsets=[-1, 0.9],
        expected_rows=[None, None, None, 1, 2, 3, 4],
        expected_cols=[56, 57, 58, 59, 60])

  def testGlimpseNoiseZeroV1Compatible(self):
    # [FUNCTION DOC]
    # Test: backward compatibility of the raw generated glimpse op (not the public wrapper) with
    # zero noise on a five by five ramp image. A three by three glimpse far outside the image
    # must be all zeros, and a seven by seven glimpse at the origin without centering returns
    # the image surrounded by a zero border (the old, shifted placement semantics). Compared with
    # exact array equality after evaluating in a test session.
    # Note: The old versions of extract_glimpse was incorrect in implementation.
    # This test is for compatibility so that graph save in old versions behave
    # the same. Notice the API uses gen_image_ops.extract_glimpse() on purpose.
    #
    # Image:
    # [  0.   1.   2.   3.   4.]
    # [  5.   6.   7.   8.   9.]
    # [ 10.  11.  12.  13.  14.]
    # [ 15.  16.  17.  18.  19.]
    # [ 20.  21.  22.  23.  24.]
    img = constant_op.constant(
        np.arange(25).reshape((1, 5, 5, 1)), dtype=dtypes.float32)
    with self.test_session():
      # Result 1:
      # [ 0.  0.  0.]
      # [ 0.  0.  0.]
      # [ 0.  0.  0.]
      result1 = gen_image_ops.extract_glimpse(
          img, [3, 3], [[-2, 2]],
          centered=False,
          normalized=False,
          noise='zero',
          uniform_noise=False)
      self.assertAllEqual(
          np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
          self.evaluate(result1)[0, :, :, 0])

      # Result 2:
      # [  0.   0.   0.   0.   0.   0.   0.]
      # [  0.   0.   1.   2.   3.   4.   0.]
      # [  0.   5.   6.   7.   8.   9.   0.]
      # [  0.  10.  11.  12.  13.  14.   0.]
      # [  0.  15.  16.  17.  18.  19.   0.]
      # [  0.  20.  21.  22.  23.  24.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      result2 = gen_image_ops.extract_glimpse(
          img, [7, 7], [[0, 0]],
          normalized=False,
          noise='zero',
          uniform_noise=False)
      self.assertAllEqual(
          np.asarray([[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 4, 0],
                      [0, 5, 6, 7, 8, 9, 0], [0, 10, 11, 12, 13, 14, 0],
                      [0, 15, 16, 17, 18, 19, 0], [0, 20, 21, 22, 23, 24, 0],
                      [0, 0, 0, 0, 0, 0, 0]]),
          self.evaluate(result2)[0, :, :, 0])


  def testGlimpseNoiseZero(self):
    # [FUNCTION DOC]
    # Test: the v2 glimpse op with zero noise on a five by five ramp image, not centered and not
    # normalized. A glimpse located completely outside returns all zeros; a seven by seven
    # glimpse at the origin returns the lower-right image quadrant in the top-left corner and
    # zeros elsewhere (the corrected placement semantics, unlike the legacy op).
    # Image:
    # [  0.   1.   2.   3.   4.]
    # [  5.   6.   7.   8.   9.]
    # [ 10.  11.  12.  13.  14.]
    # [ 15.  16.  17.  18.  19.]
    # [ 20.  21.  22.  23.  24.]
    img = constant_op.constant(
        np.arange(25).reshape((1, 5, 5, 1)), dtype=dtypes.float32)
    with self.test_session():
      # Result 1:
      # [ 0.  0.  0.]
      # [ 0.  0.  0.]
      # [ 0.  0.  0.]
      result1 = image_ops.extract_glimpse_v2(
          img, [3, 3], [[-2, -2]],
          centered=False,
          normalized=False,
          noise='zero')
      self.assertAllEqual(
          np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
          self.evaluate(result1)[0, :, :, 0])

      # Result 2:
      # [ 12.  13.  14.   0.   0.   0.   0.]
      # [ 17.  18.  19.   0.   0.   0.   0.]
      # [ 22.  23.  24.   0.   0.   0.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      # [  0.   0.   0.   0.   0.   0.   0.]
      result2 = image_ops.extract_glimpse_v2(
          img, [7, 7], [[0, 0]], normalized=False, noise='zero')
      self.assertAllEqual(
          np.asarray([[12, 13, 14, 0, 0, 0, 0], [17, 18, 19, 0, 0, 0, 0],
                      [22, 23, 24, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]),
          self.evaluate(result2)[0, :, :, 0])

  def testGlimpseNonNormalizedNonCentered(self):
    # [FUNCTION DOC]
    # Test: with both centering and normalization turned off, offsets are absolute pixel
    # positions of the glimpse's top-left corner. Three by three glimpses at offsets (0, 0) and
    # (1, 0) on a five by five ramp image must equal the corresponding image sub-blocks.
    img = constant_op.constant(
        np.arange(25).reshape((1, 5, 5, 1)), dtype=dtypes.float32)
    with self.test_session():
      result1 = image_ops.extract_glimpse_v2(
          img, [3, 3], [[0, 0]], centered=False, normalized=False)
      result2 = image_ops.extract_glimpse_v2(
          img, [3, 3], [[1, 0]], centered=False, normalized=False)
      self.assertAllEqual(
          np.asarray([[0, 1, 2], [5, 6, 7], [10, 11, 12]]),
          self.evaluate(result1)[0, :, :, 0])
      self.assertAllEqual(
          np.asarray([[5, 6, 7], [10, 11, 12], [15, 16, 17]]),
          self.evaluate(result2)[0, :, :, 0])

  def testGlimpseNegativeInput(self):
    # [FUNCTION DOC]
    # Test: requesting a glimpse with a negative size component must fail. Either an
    # invalid-argument error or a value error is accepted (raised at graph construction or at
    # evaluation time).
    img = np.arange(9).reshape([1, 3, 3, 1])
    with self.test_session():
      with self.assertRaises((errors.InvalidArgumentError, ValueError)):
        result = image_ops.extract_glimpse_v2(
            img,
            size=[1023, -63],
            offsets=[1023, 63],
            centered=False,
            normalized=False)
        self.evaluate(result)


if __name__ == '__main__':
  test.main()
