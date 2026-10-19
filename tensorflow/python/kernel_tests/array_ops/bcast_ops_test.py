# [MODULE DOC]
# Unit tests for the broadcast shape op (computing the broadcast of two shape vectors) and the
# broadcast gradient args op (computing which axes of each operand must be reduced in the
# gradient). Covers ordinary shapes, shapes containing zero-sized dimensions, and both 32-bit and
# 64-bit integer shape inputs. Defines one test case class; no import-time side effects.
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
"""Tests for tensorflow.kernels.bcast_ops."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.gen_array_ops import broadcast_args
from tensorflow.python.ops.gen_array_ops import broadcast_gradient_args
from tensorflow.python.platform import test


class BcastOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the shape broadcasting op and the broadcast gradient args op. Two small helper
  # methods evaluate the ops; the tests assert against hand-computed shapes and reduction axes.

  def _GetBroadcastShape(self, xs, ys):
    # [FUNCTION DOC]
    # HELPER: evaluates the broadcast shape op on two shape vectors xs and ys and returns the
    # resulting broadcast shape as a numpy array. Raises an error if shapes are incompatible.
    return self.evaluate(broadcast_args(xs, ys))

  def _GetGradientArgs(self, xs, ys):
    # [FUNCTION DOC]
    # HELPER: evaluates the broadcast gradient args op on two shape vectors and returns the pair
    # of numpy arrays listing the axes to reduce for the first and second operand respectively.
    return self.evaluate(broadcast_gradient_args(xs, ys))

  def testBasic(self):
    # [FUNCTION DOC]
    # Checks the broadcast shape of ten pairs of non-empty shapes (scalar-like, trailing-aligned,
    # and mixed ones-and-sizes, in both argument orders), all expected to equal 2x3x5.
    # Returns nothing.
    r = self._GetBroadcastShape([2, 3, 5], [1])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([1], [2, 3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([2, 3, 5], [5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([5], [2, 3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([2, 3, 5], [3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([3, 5], [2, 3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([2, 3, 5], [3, 1])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([3, 1], [2, 3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([2, 1, 5], [3, 1])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([3, 1], [2, 1, 5])
    self.assertAllEqual(r, [2, 3, 5])

  def testBasicGradient(self):
    # [FUNCTION DOC]
    # For the same ten shape pairs as the basic test, asserts the reduction axes returned for
    # each operand: broadcast (expanded or missing) axes are listed, others yield an empty list.
    # Returns nothing.
    r0, r1 = self._GetGradientArgs([2, 3, 5], [1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 2])

    r0, r1 = self._GetGradientArgs([1], [2, 3, 5])
    self.assertAllEqual(r0, [0, 1, 2])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1])

    r0, r1 = self._GetGradientArgs([5], [2, 3, 5])
    self.assertAllEqual(r0, [0, 1])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [3, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0])

    r0, r1 = self._GetGradientArgs([3, 5], [2, 3, 5])
    self.assertAllEqual(r0, [0])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [3, 1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 2])

    r0, r1 = self._GetGradientArgs([3, 1], [2, 3, 5])
    self.assertAllEqual(r0, [0, 2])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 1, 5], [3, 1])
    self.assertAllEqual(r0, [1])
    self.assertAllEqual(r1, [0, 2])

    r0, r1 = self._GetGradientArgs([3, 1], [2, 1, 5])
    self.assertAllEqual(r0, [0, 2])
    self.assertAllEqual(r1, [1])

  def testZeroDims(self):
    # [FUNCTION DOC]
    # Checks the broadcast shape for pairs where some dimensions are zero (including against a
    # dimension of one), in both argument orders; all expected to equal 2x0x3x0x5. Returns nothing.
    r = self._GetBroadcastShape([2, 0, 3, 0, 5], [3, 0, 5])
    self.assertAllEqual(r, [2, 0, 3, 0, 5])

    r = self._GetBroadcastShape([3, 0, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r, [2, 0, 3, 0, 5])

    r = self._GetBroadcastShape([2, 0, 3, 0, 5], [3, 1, 5])
    self.assertAllEqual(r, [2, 0, 3, 0, 5])

    r = self._GetBroadcastShape([3, 1, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r, [2, 0, 3, 0, 5])

  def testZeroDimsGradient(self):
    # [FUNCTION DOC]
    # Asserts the reduction axes returned by the gradient args op for shape pairs that contain
    # zero-sized dimensions, in both argument orders. Returns nothing.
    r0, r1 = self._GetGradientArgs([2, 0, 3, 0, 5], [3, 0, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1])

    r0, r1 = self._GetGradientArgs([3, 0, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r0, [0, 1])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 0, 3, 0, 5], [3, 1, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 3])

    r0, r1 = self._GetGradientArgs([3, 1, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r0, [0, 1, 3])
    self.assertAllEqual(r1, [])

  def testDataTypes(self):
    # [FUNCTION DOC]
    # For both 32-bit and 64-bit integer shape constants, asserts the broadcast shape and the
    # gradient reduction axes for the shapes 2x3x5 and 1. Returns nothing.
    for dtype in [dtypes.int32, dtypes.int64]:
      r = self._GetBroadcastShape(
          constant_op.constant([2, 3, 5], dtype=dtype),
          constant_op.constant([1], dtype=dtype))
      self.assertAllEqual(r, [2, 3, 5])

      r0, r1 = self._GetGradientArgs(
          constant_op.constant([2, 3, 5], dtype=dtype),
          constant_op.constant([1], dtype=dtype))
      self.assertAllEqual(r0, [])
      self.assertAllEqual(r1, [0, 1, 2])


if __name__ == "__main__":
  test.main()
