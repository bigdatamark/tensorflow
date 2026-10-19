# [MODULE DOC]
# Tests for slice and strided-slice array ops under the XLA test harness: static and dynamic
# begin offsets, negative sizes and strides, degenerate empty slices, and shrink-axis masks.
# Two test classes; every test runs over all numeric types supported by the device. No
# import-time side effects beyond imports; as a script it starts the googletest runner.
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
"""Tests for slicing."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class SliceTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the basic slice op with a begin offset and size, covering one-dimensional,
  # empty, three-dimensional, dynamic-begin and negative-size cases. No state of its own.

  def test1D(self):
    # [FUNCTION DOC]
    # Slices a ten-element vector starting at two with size four and expects two, three, four,
    # five, for every supported numeric type. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[10])
        with self.test_scope():
          o = array_ops.slice(i, [2], [4])
        params = {
            i: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([2, 3, 4, 5], result)

  def testZeroSlice(self):
    # [FUNCTION DOC]
    # Slices a two-element vector with begin zero and size zero and expects an empty result,
    # for every supported numeric type. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2])
        with self.test_scope():
          o = array_ops.slice(i, [0], [0])
        params = {
            i: [0, 1],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([], result)

  def test3D(self):
    # [FUNCTION DOC]
    # Slices a three-by-three-by-ten tensor with begin one, two, two and size one, one, four and
    # expects a single row holding six, five, four, three, for every supported numeric type.
    # Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 3, 10])
        with self.test_scope():
          o = array_ops.slice(i, [1, 2, 2], [1, 1, 4])
        params = {
            i: [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                 [5, 3, 1, 7, 9, 2, 4, 6, 8, 0]],
                [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [8, 7, 6, 5, 4, 3, 2, 1, 8, 7]],
                [[7, 5, 7, 5, 7, 5, 7, 5, 7, 5],
                 [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                 [9, 8, 7, 9, 8, 7, 9, 8, 7, 9]]]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[6, 5, 4, 3]]], result)

  def test3DWithDynamicBegin(self):
    # [FUNCTION DOC]
    # Same slice as the static three-dimensional test but with the begin offset fed at run time
    # through a placeholder (one, two, two); expects the same single row six, five, four, three.
    # Returns nothing.
    """Tests a slice where the start offset is not known at compile time."""
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 3, 10])
        begin = array_ops.placeholder(dtypes.int32, shape=[3])
        with self.test_scope():
          o = array_ops.slice(i, begin, [1, 1, 4])
        params = {
            i: [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                 [5, 3, 1, 7, 9, 2, 4, 6, 8, 0]],
                [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [8, 7, 6, 5, 4, 3, 2, 1, 8, 7]],
                [[7, 5, 7, 5, 7, 5, 7, 5, 7, 5],
                 [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                 [9, 8, 7, 9, 8, 7, 9, 8, 7, 9]]],
            begin: [1, 2, 2]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[6, 5, 4, 3]]], result)

  def test3DWithDynamicBeginAndNegativeSize(self):
    # [FUNCTION DOC]
    # Slices with a run-time begin of one, one, two and size one, minus one, four, where minus
    # one means all remaining elements along that axis; expects two rows, all ones and then six,
    # five, four, three. Returns nothing.
    """Tests a slice where `begin` is fed dynamically and `size` contains -1."""
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 3, 10])
        begin = array_ops.placeholder(dtypes.int32, shape=[3])
        with self.test_scope():
          o = array_ops.slice(i, begin, [1, -1, 4])
        params = {
            i: [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                 [5, 3, 1, 7, 9, 2, 4, 6, 8, 0]],
                [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [8, 7, 6, 5, 4, 3, 2, 1, 8, 7]],
                [[7, 5, 7, 5, 7, 5, 7, 5, 7, 5],
                 [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                 [9, 8, 7, 9, 8, 7, 9, 8, 7, 9]]],
            begin: [1, 1, 2]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[1, 1, 1, 1], [6, 5, 4, 3]]], result)


class StridedSliceTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the strided slice op: positive and negative strides, dynamic begin and end,
  # degenerate (empty) results and shrink-axis masks. No state of its own.

  def test1D(self):
    # [FUNCTION DOC]
    # Strided slice of a ten-element vector from two to six with stride two; expects two and
    # four, for every supported numeric type. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[10])
        with self.test_scope():
          o = array_ops.strided_slice(i, [2], [6], [2])
        params = {
            i: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([2, 4], result)

  def test1DDynamic(self):
    # [FUNCTION DOC]
    # Strided slice of a ten-element vector where begin is fed at run time (zero) and end is
    # computed on the device as begin plus one with stride one; expects the single element zero.
    # Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[10])
        begin = array_ops.placeholder(dtypes.int32, shape=[1])
        with self.test_scope():
          end = math_ops.add(begin, [1])
          o = array_ops.strided_slice(i, begin, end, [1])
        params = {
            i: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            begin: [0]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([0], result)

  def test1DNegativeStride(self):
    # [FUNCTION DOC]
    # Strided slice of a ten-element vector from six down to (excluding) two with stride minus
    # two; expects six and four. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[10])
        with self.test_scope():
          o = array_ops.strided_slice(i, [6], [2], [-2])
        params = {
            i: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([6, 4], result)

  def test2DDegenerate(self):
    # [FUNCTION DOC]
    # Strided slice of a two-by-three matrix with begin minus one, zero and end zero, three
    # (default stride); the start is past the end so the result is empty with shape zero by three.
    # Asserts the result shape. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 3])
        with self.test_scope():
          o = array_ops.strided_slice(i, [-1, 0], [0, 3])
        params = {
            i: [[0, 1, 2],
                [3, 4, 5]]
        }
        result = o.eval(feed_dict=params)

        self.assertEqual(tensor_shape.TensorShape((0, 3)), result.shape)

  def test2DDegenerateNegativeStride(self):
    # [FUNCTION DOC]
    # Strided slice of a two-by-three matrix with begin zero, zero, end minus one, three and
    # strides minus one, one; the range along the first axis is empty, so the result shape is
    # zero by three. Asserts the result shape. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 3])
        with self.test_scope():
          o = array_ops.strided_slice(i, [0, 0], [-1, 3], [-1, 1])
        params = {
            i: [[0, 1, 2],
                [3, 4, 5]]
        }
        result = o.eval(feed_dict=params)

        self.assertEqual(tensor_shape.TensorShape((0, 3)), result.shape)

  def test2DFullSlice(self):
    # [FUNCTION DOC]
    # Strided slice of a two-by-four matrix with a run-time begin of one, one and end computed
    # as begin plus one in each axis, strides one; expects the single element five.
    # Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        with self.test_scope():
          i = array_ops.placeholder(dtype, shape=[2, 4])
          begin = array_ops.placeholder(dtypes.int32, shape=[2])
          end = math_ops.add(begin, [1, 1])
          o = array_ops.strided_slice(i, begin, end, [1, 1])
        params = {
            i: [[0, 1, 2, 3], [4, 5, 6, 7]],
            begin: [1, 1]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[5]], result)

  def test3D(self):
    # [FUNCTION DOC]
    # Strided slice of a three-by-three-by-ten tensor with begin zero, two, two, end two, three,
    # six and strides one, one, two; expects two rows each holding one pair, namely one and nine,
    # then six and four. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 3, 10])
        with self.test_scope():
          o = array_ops.strided_slice(i, [0, 2, 2], [2, 3, 6], [1, 1, 2])
        params = {
            i: [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                 [5, 3, 1, 7, 9, 2, 4, 6, 8, 0]],
                [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [8, 7, 6, 5, 4, 3, 2, 1, 8, 7]],
                [[7, 5, 7, 5, 7, 5, 7, 5, 7, 5],
                 [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                 [9, 8, 7, 9, 8, 7, 9, 8, 7, 9]]]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[1, 9]], [[6, 4]]], result)

  def test3DNegativeStride(self):
    # [FUNCTION DOC]
    # Strided slice of a three-by-four-by-ten tensor walking backwards on every axis (begin two,
    # two, six; end zero, zero, two; strides minus one, minus one, minus two); expects a
    # two-by-two-by-two result of nine, eight, one, one and two, four, five, seven.
    # Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 4, 10])
        with self.test_scope():
          o = array_ops.strided_slice(i, [2, 2, 6], [0, 0, 2], [-1, -1, -2])
        params = {
            i: [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                 [5, 3, 1, 7, 9, 2, 4, 6, 8, 0],
                 [4, 5, 2, 4, 3, 7, 6, 8, 9, 4]],
                [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 [4, 3, 4, 5, 7, 6, 5, 3, 4, 5],
                 [8, 7, 6, 5, 4, 3, 2, 1, 8, 7],
                 [7, 1, 7, 1, 8, 1, 8, 1, 3, 1]],
                [[7, 5, 7, 5, 7, 5, 7, 5, 7, 5],
                 [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                 [9, 8, 7, 9, 8, 7, 9, 8, 7, 9],
                 [9, 9, 5, 5, 6, 6, 3, 3, 6, 6]]]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[9, 8],
                              [1, 1]],
                             [[2, 4],
                              [5, 7]]], result)

  # Test shrink_axis_mask. This `strided_slice` call is equivalent to `i[1,:]`.
  def testShrinkAxisMask(self):
    # [FUNCTION DOC]
    # Strided slice of a two-by-three matrix with shrink-axis mask one, which drops the first
    # axis after taking index one; expects the vector three, four, five. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 3])
        with self.test_scope():
          o = array_ops.strided_slice(i, [1, 0], [10, 3], shrink_axis_mask=1)
        params = {
            i: [[0, 1, 2], [3, 4, 5]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([3, 4, 5], result)

  # Test shrink_axis_mask with the range for the second dimension implicit.
  # This `strided_slice` call is equivalent to `i[1]`.
  def testShrinkAxisMaskImplicitRange(self):
    # [FUNCTION DOC]
    # Same as the shrink-axis test but only the first axis is specified (begin one, end ten), so
    # the second axis range is implicit; expects the vector three, four, five. Returns nothing.
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 3])
        with self.test_scope():
          o = array_ops.strided_slice(i, [1], [10], shrink_axis_mask=1)
        params = {
            i: [[0, 1, 2], [3, 4, 5]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([3, 4, 5], result)

if __name__ == "__main__":
  googletest.main()
