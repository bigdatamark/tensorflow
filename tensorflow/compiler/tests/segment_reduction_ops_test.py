# [MODULE DOC]
# Test module for segment reduction ops (sum, product, minimum, maximum; sorted-style V2 variants
# with an explicit segment count, and unsorted variants) under the XLA test harness. One test
# class feeds small arrays through the device ops and compares against hand-computed results.
# No import-time side effects beyond imports; as a script it starts the googletest runner.
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
"""Test cases for segment reduction ops."""

import functools

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class SegmentReductionOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the segment reduction ops. Provides small wrapper helpers that run each op
  # on the test device and many tests covering default, fewer-than-needed and more-than-needed
  # segment counts, negative indices (ignored by unsorted ops), multi-dimensional data and
  # indices, and shape errors. Holds no state of its own.
  """Test cases for segment reduction ops."""

  def _findDevice(self, device_name):
    # [FUNCTION DOC]
    # HELPER: Device-detection utility for the tests.
    # Input: device_name is a device type string such as TPU. Lists local devices and returns
    # true if any has that device type, otherwise false.
    devices = device_lib.list_local_devices()
    for d in devices:
      if d.device_type == device_name:
        return True
    return False

  def _segmentReduction(self, op, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Core runner behind the per-op wrappers.
    # Inputs: op is a callable taking data, segment ids and segment count; data is a numpy array;
    # indices is a numpy integer array, or a plain integer meaning a scalar segment id;
    # num_segments is the output segment count. Creates placeholders (scalar 32-bit integer for
    # a plain integer index, otherwise matching the array), builds op in the device scope within
    # a session and runs it with data and indices fed. Returns the resulting numpy array.
    with self.session() as sess, self.test_scope():
      d = array_ops.placeholder(data.dtype, shape=data.shape)
      if isinstance(indices, int):
        i = array_ops.placeholder(np.int32, shape=[])
      else:
        i = array_ops.placeholder(indices.dtype, shape=indices.shape)
      return sess.run(op(d, i, num_segments), {d: data, i: indices})

  def _unsortedSegmentSum(self, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Runs the unsorted segment sum on the device via the core runner and returns the
    # numpy result. Inputs: data array, segment ids, segment count.
    return self._segmentReduction(math_ops.unsorted_segment_sum, data, indices,
                                  num_segments)

  def _segmentSumV2(self, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Runs the segment sum (version two, explicit segment count) on the device via the
    # core runner and returns the numpy result. Inputs: data array, segment ids, segment count.
    return self._segmentReduction(math_ops.segment_sum_v2, data, indices,
                                  num_segments)

  def _segmentProdV2(self, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Runs the segment product (version two, explicit segment count) on the device via
    # the core runner and returns the numpy result. Inputs: data array, segment ids, count.
    return self._segmentReduction(math_ops.segment_prod_v2, data, indices,
                                  num_segments)

  def _segmentMinV2(self, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Runs the segment minimum (version two, explicit segment count) on the device via
    # the core runner and returns the numpy result. Inputs: data array, segment ids, count.
    return self._segmentReduction(math_ops.segment_min_v2, data, indices,
                                  num_segments)

  def _segmentMaxV2(self, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Runs the segment maximum (version two, explicit segment count) on the device via
    # the core runner and returns the numpy result. Inputs: data array, segment ids, count.
    return self._segmentReduction(math_ops.segment_max_v2, data, indices,
                                  num_segments)

  def _unsortedSegmentProd(self, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Runs the unsorted segment product on the device via the core runner and returns
    # the numpy result. Inputs: data array, segment ids, segment count.
    return self._segmentReduction(math_ops.unsorted_segment_prod, data, indices,
                                  num_segments)

  def _unsortedSegmentMin(self, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Runs the unsorted segment minimum on the device via the core runner and returns
    # the numpy result. Inputs: data array, segment ids, segment count.
    return self._segmentReduction(math_ops.unsorted_segment_min, data, indices,
                                  num_segments)

  def _unsortedSegmentMax(self, data, indices, num_segments):
    # [FUNCTION DOC]
    # HELPER: Runs the unsorted segment maximum on the device via the core runner and returns
    # the numpy result. Inputs: data array, segment ids, segment count.
    return self._segmentReduction(math_ops.unsorted_segment_max, data, indices,
                                  num_segments)

  def testSegmentSum(self):
    # [FUNCTION DOC]
    # For every numeric type, sums six values zero through five by segment ids zero, zero, two,
    # three, three, three with four segments; expects one, zero, two, twelve (empty segment gives
    # zero). Returns nothing.
    for dtype in self.numeric_types:
      self.assertAllClose(
          np.array([1, 0, 2, 12], dtype=dtype),
          self._segmentSumV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 4))

  def testSegmentProd(self):
    # [FUNCTION DOC]
    # For every numeric type, multiplies the same six values by the same segment ids with four
    # segments; expects zero, one, two, sixty (empty segment gives the identity one).
    # Returns nothing.
    for dtype in self.numeric_types:
      self.assertAllClose(
          np.array([0, 1, 2, 60], dtype=dtype),
          self._segmentProdV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 4))

  def testSegmentProdNumSegmentsLess(self):
    # [FUNCTION DOC]
    # Same product scenario with a segment count of three, fewer than the ids need; the id
    # three entries are dropped, so expects zero, one, two. Returns nothing.
    for dtype in self.numeric_types:
      self.assertAllClose(
          np.array([0, 1, 2], dtype=dtype),
          self._segmentProdV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 3))

  def testSegmentProdNumSegmentsMore(self):
    # [FUNCTION DOC]
    # Same product scenario with a segment count of five; expects zero, one, two, sixty, one
    # (the extra empty segment holds the identity). Returns nothing.
    for dtype in self.numeric_types:
      self.assertAllClose(
          np.array([0, 1, 2, 60, 1], dtype=dtype),
          self._segmentProdV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 5))

  def testSegmentMin(self):
    # [FUNCTION DOC]
    # For integer and float types, takes segment minimums of the six values with four segments.
    # The empty segment is expected to hold the largest value of the type (positive infinity
    # for 64-bit floats when a TPU device is present). Expects zero, that largest value, two,
    # three. Returns nothing.
    for dtype in self.int_types | self.float_types:
      maxval = dtypes.as_dtype(dtype).max
      if dtype == np.float64 and self._findDevice("TPU"):
        maxval = np.inf
      self.assertAllClose(
          np.array([0, maxval, 2, 3], dtype=dtype),
          self._segmentMinV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 4))

  def testSegmentMinNumSegmentsLess(self):
    # [FUNCTION DOC]
    # Same minimum scenario with three segments; the id three entries are dropped. Expects zero,
    # the largest value of the type for the empty segment (infinity for 64-bit floats on TPU),
    # and two. Returns nothing.
    for dtype in self.int_types | self.float_types:
      maxval = dtypes.as_dtype(dtype).max
      if dtype == np.float64 and self._findDevice("TPU"):
        maxval = np.inf
      self.assertAllClose(
          np.array([0, maxval, 2], dtype=dtype),
          self._segmentMinV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 3))

  def testSegmentMinNumSegmentsMore(self):
    # [FUNCTION DOC]
    # Same minimum scenario with five segments; expects zero, largest value, two, three, and the
    # largest value again for the extra empty segment (infinity for 64-bit floats on TPU).
    # Returns nothing.
    for dtype in self.int_types | self.float_types:
      maxval = dtypes.as_dtype(dtype).max
      if dtype == np.float64 and self._findDevice("TPU"):
        maxval = np.inf
      self.assertAllClose(
          np.array([0, maxval, 2, 3, maxval], dtype=dtype),
          self._segmentMinV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 5))

  def testSegmentMax(self):
    # [FUNCTION DOC]
    # For integer and float types, takes segment maximums of the six values with four segments.
    # The empty segment is expected to hold the smallest value of the type (negative infinity
    # for 64-bit floats when a TPU device is present). Expects one, that smallest value, two,
    # five. Returns nothing.
    for dtype in self.int_types | self.float_types:
      minval = dtypes.as_dtype(dtype).min
      if dtype == np.float64 and self._findDevice("TPU"):
        minval = -np.inf
      self.assertAllClose(
          np.array([1, minval, 2, 5], dtype=dtype),
          self._segmentMaxV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 4))

  def testSegmentMaxNumSegmentsLess(self):
    # [FUNCTION DOC]
    # Same maximum scenario with three segments; the id three entries are dropped. Expects one,
    # the smallest value of the type for the empty segment (negative infinity for 64-bit floats
    # on TPU), and two. Returns nothing.
    for dtype in self.int_types | self.float_types:
      minval = dtypes.as_dtype(dtype).min
      if dtype == np.float64 and self._findDevice("TPU"):
        minval = -np.inf
      self.assertAllClose(
          np.array([1, minval, 2], dtype=dtype),
          self._segmentMaxV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 3))

  def testSegmentMaxNumSegmentsMore(self):
    # [FUNCTION DOC]
    # Same maximum scenario with five segments; expects one, smallest value, two, five, and the
    # smallest value again for the extra empty segment (negative infinity for 64-bit floats on
    # TPU). Returns nothing.
    for dtype in self.int_types | self.float_types:
      minval = dtypes.as_dtype(dtype).min
      if dtype == np.float64 and self._findDevice("TPU"):
        minval = -np.inf
      self.assertAllClose(
          np.array([1, minval, 2, 5, minval], dtype=dtype),
          self._segmentMaxV2(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([0, 0, 2, 3, 3, 3], dtype=np.int32), 5))

  def testUnsortedSegmentSum0DIndices1DData(self):
    # [FUNCTION DOC]
    # For every numeric type, uses a scalar segment id of two with a six-element vector and four
    # segments; expects a four-by-six matrix of zeros except row two, which equals the data.
    # Returns nothing.
    for dtype in self.numeric_types:
      self.assertAllClose(
          np.array(
              [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5],
               [0, 0, 0, 0, 0, 0]],
              dtype=dtype),
          self._unsortedSegmentSum(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype), 2, 4))

  def testUnsortedSegmentSum1DIndices1DData(self):
    # [FUNCTION DOC]
    # For every numeric type, sums six values by unordered ids three, zero, two, one, three,
    # three into four segments; expects one, three, two, nine. Returns nothing.
    for dtype in self.numeric_types:
      self.assertAllClose(
          np.array([1, 3, 2, 9], dtype=dtype),
          self._unsortedSegmentSum(
              np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
              np.array([3, 0, 2, 1, 3, 3], dtype=np.int32), 4))

  def testUnsortedSegmentSum1DIndices1DDataNegativeIndices(self):
    # [FUNCTION DOC]
    # For every numeric type, sums seven values with ids that include negative entries into four
    # segments; entries with negative ids are ignored. Expects six, three, zero, six.
    # Returns nothing.
    for dtype in self.numeric_types:
      self.assertAllClose(
          np.array([6, 3, 0, 6], dtype=dtype),
          self._unsortedSegmentSum(
              np.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype),
              np.array([3, -1, 0, 1, 0, -1, 3], dtype=np.int32), 4))

  def testUnsortedSegmentSum1DIndices2DDataDisjoint(self):
    # [FUNCTION DOC]
    # For every numeric type, sums the rows of a five-by-four matrix by distinct ids eight, one,
    # zero, three, seven into ten segments; expects each row placed at its id with all other
    # segment rows zero. Returns nothing.
    for dtype in self.numeric_types:
      data = np.array(
          [[0, 1, 2, 3], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43],
           [50, 51, 52, 53]],
          dtype=dtype)
      indices = np.array([8, 1, 0, 3, 7], dtype=np.int32)
      num_segments = 10
      y = self._unsortedSegmentSum(data, indices, num_segments)
      self.assertAllClose(
          np.array(
              [[30, 31, 32, 33], [20, 21, 22, 23], [0, 0, 0, 0],
               [40, 41, 42, 43], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
               [50, 51, 52, 53], [0, 1, 2, 3], [0, 0, 0, 0]],
              dtype=dtype), y)

  def testUnsortedSegmentSum1DIndices2DDataNonDisjoint(self):
    # [FUNCTION DOC]
    # For every numeric type, sums the rows of a five-by-four matrix by ids zero, one, two, zero,
    # one into four segments; rows sharing an id are added, and the unused segment is zero.
    # Returns nothing.
    for dtype in self.numeric_types:
      data = np.array(
          [[0, 1, 2, 3], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43],
           [50, 51, 52, 53]],
          dtype=dtype)
      indices = np.array([0, 1, 2, 0, 1], dtype=np.int32)
      num_segments = 4
      y = self._unsortedSegmentSum(data, indices, num_segments)
      self.assertAllClose(
          np.array(
              [[40, 42, 44, 46], [70, 72, 74, 76], [30, 31, 32, 33],
               [0, 0, 0, 0]],
              dtype=dtype), y)

  def testUnsortedSegmentSum2DIndices3DData(self):
    # [FUNCTION DOC]
    # For every numeric type, uses a four-by-two-by-three data tensor with a four-by-two id
    # matrix into eight segments; each data row of length three goes to the segment named by its
    # id and rows sharing an id are added. Unused segments are zero. Returns nothing.
    for dtype in self.numeric_types:
      data = np.array(
          [[[0, 1, 2], [10, 11, 12]], [[100, 101, 102], [110, 111, 112]], [[
              80, 81, 82
          ], [123, 124, 125]], [[103, 104, 105], [106, 107, 108]]],
          dtype=dtype)
      indices = np.array([[3, 5], [3, 1], [5, 0], [6, 2]], dtype=np.int32)
      num_segments = 8
      y = self._unsortedSegmentSum(data, indices, num_segments)
      self.assertAllClose(
          np.array(
              [[123, 124, 125], [110, 111, 112], [106, 107, 108], [
                  100, 102, 104
              ], [0, 0, 0.], [90, 92, 94], [103, 104, 105], [0, 0, 0]],
              dtype=dtype), y)

  def testUnsortedSegmentSum1DIndices3DData(self):
    # [FUNCTION DOC]
    # For every numeric type, uses a four-by-two-by-three data tensor with ids three, zero, two,
    # five into six segments; each two-by-three slice is placed at its segment and unused
    # segments are zero. Returns nothing.
    for dtype in self.numeric_types:
      data = np.array(
          [[[0, 1, 2], [10, 11, 12]], [[100, 101, 102], [110, 111, 112]], [[
              120, 121, 122
          ], [123, 124, 125]], [[103, 104, 105], [106, 107, 108]]],
          dtype=dtype)
      indices = np.array([3, 0, 2, 5], dtype=np.int32)
      num_segments = 6
      y = self._unsortedSegmentSum(data, indices, num_segments)
      self.assertAllClose(
          np.array(
              [[[100, 101, 102.], [110, 111, 112]], [[0, 0, 0], [0, 0, 0]],
               [[120, 121, 122], [123, 124, 125]], [[0, 1, 2.], [10, 11, 12]],
               [[0, 0, 0], [0, 0, 0]], [[103, 104, 105], [106, 107, 108]]],
              dtype=dtype), y)

  def testUnsortedSegmentSumShapeError(self):
    # [FUNCTION DOC]
    # For every numeric type, passes data of shape four, eight, seven with ids of shape three,
    # two (not a prefix of the data shape) and expects a value error to be raised when the op
    # is built. Returns nothing.
    for dtype in self.numeric_types:
      data = np.ones((4, 8, 7), dtype=dtype)
      indices = np.ones((3, 2), dtype=np.int32)
      num_segments = 4
      self.assertRaises(
          ValueError,
          functools.partial(self._segmentReduction,
                            math_ops.unsorted_segment_sum, data, indices,
                            num_segments))

  def testUnsortedSegmentOps1DIndices1DDataNegativeIndices(self):
    # [FUNCTION DOC]
    # Checks unsorted segment product, minimum and maximum with seven values and ids that include
    # negative entries (ignored) into four segments. Product for all numeric types expects eight,
    # three, one, zero; minimum for integer and float types expects two, three, the type's
    # largest value (for the empty segment), zero; maximum expects four, three, the type's
    # smallest value, six. Returns nothing.
    """Tests for min, max, and prod ops.

    These share most of their implementation with sum, so we only test basic
    functionality.
    """
    for dtype in self.numeric_types:
      self.assertAllClose(
          np.array([8, 3, 1, 0], dtype=dtype),
          self._unsortedSegmentProd(
              np.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype),
              np.array([3, -1, 0, 1, 0, -1, 3], dtype=np.int32), 4))

    for dtype in self.int_types | self.float_types:
      minval = dtypes.as_dtype(dtype).min
      maxval = dtypes.as_dtype(dtype).max

      self.assertAllClose(
          np.array([2, 3, maxval, 0], dtype=dtype),
          self._unsortedSegmentMin(
              np.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype),
              np.array([3, -1, 0, 1, 0, -1, 3], dtype=np.int32), 4))
      self.assertAllClose(
          np.array([4, 3, minval, 6], dtype=dtype),
          self._unsortedSegmentMax(
              np.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype),
              np.array([3, -1, 0, 1, 0, -1, 3], dtype=np.int32), 4))


if __name__ == "__main__":
  googletest.main()
