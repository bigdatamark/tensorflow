# [MODULE DOC]
# Tests for the data-format dimension-map and vector-permute operators on the XLA test device.
# Two test classes: one maps dimension indices between layout strings (4D, 5D and custom letter
# layouts, including negative indices), the other permutes vectors and 2D arrays of sizes
# (batch/height/width/channel orderings). No import-time side effects; run via the test main.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the DataFormatVecPermute operator."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class XlaDataFormatDimMapTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the data-format dimension-map operator on the XLA test device.

  def _test(self, input_data, src_format, dst_format, expected):
    # [FUNCTION DOC]
    # HELPER: runs the dimension-map operator once per integer type (32-bit and 64-bit) on the
    # given input (scalar or nested list of dimension indices, negatives allowed), mapping from
    # source layout string to destination layout string, and asserts the result equals expected.
    # Returns nothing.
    for dtype in {np.int32, np.int64}:
      x = np.array(input_data, dtype=dtype)
      with self.session() as session:
        with self.test_scope():
          placeholder = array_ops.placeholder(dtypes.as_dtype(x.dtype), x.shape)
          param = {placeholder: x}
          output = nn_ops.data_format_dim_map(
              placeholder, src_format=src_format, dst_format=dst_format)
        result = session.run(output, param)
      self.assertAllEqual(result, expected)

  def test(self):
    # [FUNCTION DOC]
    # Runs many dimension-map cases: scalars, vectors and 2D inputs (including negative indices)
    # from NHWC to NCHW, NHWC to HWNC and WHCN, an arbitrary four-letter layout pair, and the 5D
    # layouts NDHWC to NCDHW, DHWNC and WHDCN. Each case is checked against hand-computed output.
    # The same two cases ([1, -3, -2]) are repeated once.
    self._test(0, "NHWC", "NCHW", 0)
    self._test(1, "NHWC", "NCHW", 2)
    self._test(2, "NHWC", "NCHW", 3)
    self._test(3, "NHWC", "NCHW", 1)
    self._test(-1, "NHWC", "NCHW", 1)
    self._test(-2, "NHWC", "NCHW", 3)
    self._test(-3, "NHWC", "NCHW", 2)
    self._test(-4, "NHWC", "NCHW", 0)
    self._test([1, 3], "NHWC", "NCHW", [2, 1])
    self._test([1, 3, -2], "NHWC", "NCHW", [2, 1, 3])
    self._test([1, -3, -2], "NHWC", "NCHW", [2, 2, 3])
    self._test([[1, -3], [1, -1]], "NHWC", "NCHW", [[2, 2], [2, 1]])

    self._test([1, -3, -2], "NHWC", "NCHW", [2, 2, 3])
    self._test([-4, -3, -2, -1, 0, 1, 2, 3], "NHWC", "HWNC",
               [2, 0, 1, 3, 2, 0, 1, 3])
    self._test([-4, -3, -2, -1, 0, 1, 2, 3], "NHWC", "WHCN",
               [3, 1, 0, 2, 3, 1, 0, 2])
    self._test([-4, -3, -2, -1, 0, 1, 2, 3], "qwer", "rewq",
               [3, 2, 1, 0, 3, 2, 1, 0])

    self._test(0, "NDHWC", "NCDHW", 0)
    self._test(1, "NDHWC", "NCDHW", 2)
    self._test(2, "NDHWC", "NCDHW", 3)
    self._test(3, "NDHWC", "NCDHW", 4)
    self._test(4, "NDHWC", "NCDHW", 1)
    self._test([1, 4], "NDHWC", "NCDHW", [2, 1])
    self._test([1, 4, -2], "NDHWC", "NCDHW", [2, 1, 4])
    self._test([1, -3, -2], "NDHWC", "NCDHW", [2, 3, 4])
    self._test([[1, -4], [1, -1]], "NDHWC", "NCDHW", [[2, 2], [2, 1]])

    self._test([1, -3, -2], "NDHWC", "NCDHW", [2, 3, 4])
    self._test([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], "NDHWC", "DHWNC",
               [3, 0, 1, 2, 4, 3, 0, 1, 2, 4])
    self._test([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], "NDHWC", "WHDCN",
               [4, 2, 1, 0, 3, 4, 2, 1, 0, 3])


class XlaPermuteOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the data-format vector-permute operator on the XLA test device, using size
  # vectors (length 4 or 2) and 4-by-2 padding-like arrays across several layout pairs.

  def _runPermuteAndCompare(self, x, src_format, dst_format, expected):
    # [FUNCTION DOC]
    # HELPER: runs the vector-permute operator on the array x (placeholder typed from x) with the
    # given source and destination layout strings and asserts the result equals expected.
    # Returns nothing.
    with self.session() as session:
      with self.test_scope():
        placeholder = array_ops.placeholder(dtypes.as_dtype(x.dtype), x.shape)
        param = {placeholder: x}
        output = nn_ops.data_format_vec_permute(
            placeholder, src_format=src_format, dst_format=dst_format)
      result = session.run(output, param)
    self.assertAllEqual(result, expected)

  def testNHWCToNCHW(self):
    # [FUNCTION DOC]
    # Permutes a length-4 integer vector from NHWC to NCHW order for both integer widths.
    for dtype in {np.int32, np.int64}:
      x = np.array([7, 4, 9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "NCHW", [7, 3, 4, 9])

  def testNHWCToNCHW_Size2(self):
    # [FUNCTION DOC]
    # Permutes a length-2 (spatial only) vector from NHWC to NCHW; expects it unchanged.
    for dtype in {np.int32, np.int64}:
      x = np.array([4, 9], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "NCHW", [4, 9])

  def testNCHWToNHWC(self):
    # [FUNCTION DOC]
    # Permutes a length-4 vector from NCHW to NHWC order for both integer widths.
    for dtype in {np.int32, np.int64}:
      x = np.array([7, 4, 9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "NCHW", "NHWC", [7, 9, 3, 4])

  def testNCHWToNHWC_Size2(self):
    # [FUNCTION DOC]
    # Permutes a length-2 vector from NCHW to NHWC; expects it unchanged.
    for dtype in {np.int32, np.int64}:
      x = np.array([9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "NCHW", "NHWC", [9, 3])

  def testNHWCToHWNC(self):
    # [FUNCTION DOC]
    # Permutes a length-4 vector from NHWC to HWNC order for both integer widths.
    for dtype in {np.int32, np.int64}:
      x = np.array([7, 4, 9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "HWNC", [4, 9, 7, 3])

  def testHWNCToNHWC(self):
    # [FUNCTION DOC]
    # Permutes a length-4 vector from HWNC to NHWC order for both integer widths.
    for dtype in {np.int32, np.int64}:
      x = np.array([7, 4, 9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "HWNC", "NHWC", [9, 7, 4, 3])

  def testNHWCToNCHW2D(self):
    # [FUNCTION DOC]
    # Permutes the rows of a 4-by-2 array from NHWC to NCHW order, for both integer widths.
    for dtype in {np.int32, np.int64}:
      x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "NCHW",
                                 [[7, 4], [5, 1], [9, 3], [4, 5]])

  def testNHWCToHWNC2D(self):
    # [FUNCTION DOC]
    # Permutes the rows of a 4-by-2 array from NHWC to HWNC order, for both integer widths.
    for dtype in {np.int32, np.int64}:
      x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "HWNC",
                                 [[9, 3], [4, 5], [7, 4], [5, 1]])

  def testHWNCToNHWC2D(self):
    # [FUNCTION DOC]
    # Permutes the rows of a 4-by-2 array from HWNC to NHWC order, for both integer widths.
    for dtype in {np.int32, np.int64}:
      x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
      self._runPermuteAndCompare(x, "HWNC", "NHWC",
                                 [[4, 5], [7, 4], [9, 3], [5, 1]])

  def testNCHWToNHWC2D(self):
    # [FUNCTION DOC]
    # Permutes the rows of a 4-by-2 array from NCHW to NHWC order, for both integer widths.
    for dtype in {np.int32, np.int64}:
      x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
      self._runPermuteAndCompare(x, "NCHW", "NHWC",
                                 [[7, 4], [4, 5], [5, 1], [9, 3]])


if __name__ == "__main__":
  test.main()
