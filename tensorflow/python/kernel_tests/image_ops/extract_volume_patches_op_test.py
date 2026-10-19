# [MODULE DOC]
# Functional tests for the extract-volume-patches op, the three-dimensional counterpart of image
# patch extraction. Checks hand-computed outputs for several kernel and stride combinations with
# valid and same padding over four floating point types. Dilation rates are not supported by the
# op and are not tested. No import-time side effects besides the test main entry point when run
# as a script.
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
"""Functional tests for ExtractVolumePatches op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ExtractVolumePatches(test.TestCase):
  # [CLASS DOC]
  # Test case for the extract-volume-patches op. A shared helper compares op output with the
  # expected patches across float types; each test provides a small integer volume and the
  # expected patches for one kernel, stride and padding setup.
  """Functional tests for ExtractVolumePatches op."""

  def _VerifyValues(self, image, ksizes, strides, padding, patches):
    # [FUNCTION DOC]
    # HELPER: shared input-output checker for the volume patch extraction tests.
    # Inputs: image is a numeric numpy array of shape batch, planes, rows, columns, depth;
    # ksizes and strides are three-element lists (planes, rows, columns) padded with ones for
    # the batch and depth dimensions; padding is the mode name ("VALID" or "SAME"); patches is the
    # expected numpy array output. For half, single, double precision and bfloat16 it casts the
    # image, runs the op and asserts the result is close to the expected patches cast to the same
    # type. Returns nothing.
    """Tests input-output pairs for the ExtractVolumePatches op.

    Args:
      image: Input tensor with shape:
             [batch, in_planes, in_rows, in_cols, depth].
      ksizes: Patch size specified as: [ksize_planes, ksize_rows, ksize_cols].
      strides: Output strides, specified as:
               [stride_planes, stride_rows, stride_cols].
      padding: Padding type.
      patches: Expected output.

    Note:
      rates are not supported as of now.
    """
    ksizes = [1] + ksizes + [1]
    strides = [1] + strides + [1]

    for dtype in [
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ]:
      out_tensor = array_ops.extract_volume_patches(
          constant_op.constant(image.astype(dtype)),
          ksizes=ksizes,
          strides=strides,
          padding=padding,
          name="im2col_3d")
      self.assertAllClose(patches.astype(dtype), self.evaluate(out_tensor))

  # pylint: disable=bad-whitespace
  def testKsize1x1x1Stride1x1x1(self):
    # [FUNCTION DOC]
    # Test: with a one by one by one kernel and unit strides, for both padding modes, the output
    # equals the input (shape 2, 3, 4, 5, 6).
    """Verifies that for 1x1x1 kernel the output equals the input."""
    image = np.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6]) + 1
    patches = image
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1, 1],
          strides=[1, 1, 1],
          padding=padding,
          patches=patches)

  def testKsize1x1x1Stride2x3x4(self):
    # [FUNCTION DOC]
    # Test: with a one by one by one kernel and strides two, three and four over planes, rows and
    # columns, for both padding modes, the output equals the input subsampled with those steps.
    """Test for 1x1x1 kernel and strides."""
    image = np.arange(6 * 2 * 4 * 5 * 3).reshape([6, 2, 4, 5, 3]) + 1
    patches = image[:, ::2, ::3, ::4, :]
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1, 1],
          strides=[2, 3, 4],
          padding=padding,
          patches=patches)

  def testKsize1x1x2Stride2x2x3(self):
    # [FUNCTION DOC]
    # Test: with a one by one by two kernel and strides two, two and three on a small ramp volume,
    # for both padding modes, the output holds the expected pairs of neighbouring columns at the
    # strided positions (a hand-written expected array of shape 1, 2, 2, 2, 2).
    """Test for 1x1x2 kernel and strides."""
    image = np.arange(45).reshape([1, 3, 3, 5, 1]) + 1
    patches = np.array([[[[[ 1,  2],
                           [ 4,  5]],
                          [[11, 12],
                           [14, 15]]],
                         [[[31, 32],
                           [34, 35]],
                          [[41, 42],
                           [44, 45]]]]])
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1, 2],
          strides=[2, 2, 3],
          padding=padding,
          patches=patches)

  def testKsize2x2x2Stride1x1x1Valid(self):
    # [FUNCTION DOC]
    # Test: a two by two by two kernel with valid padding on a two by two by two single-channel
    # volume yields a single patch containing all eight values in order.
    """Test for 2x2x2 kernel with VALID padding."""
    image = np.arange(8).reshape([1, 2, 2, 2, 1]) + 1
    patches = np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8]]]]])
    self._VerifyValues(
        image,
        ksizes=[2, 2, 2],
        strides=[1, 1, 1],
        padding="VALID",
        patches=patches)

  def testKsize2x2x2Stride1x1x1Same(self):
    # [FUNCTION DOC]
    # Test: the same volume and kernel with same padding yields eight patches; windows that extend
    # beyond the volume are padded with zeros at the far edges.
    """Test for 2x2x2 kernel with SAME padding."""
    image = np.arange(8).reshape([1, 2, 2, 2, 1]) + 1
    patches = np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8],
                           [2, 0, 4, 0, 6, 0, 8, 0]],
                          [[3, 4, 0, 0, 7, 8, 0, 0],
                           [4, 0, 0, 0, 8, 0, 0, 0]]],
                         [[[5, 6, 7, 8, 0, 0, 0, 0],
                           [6, 0, 8, 0, 0, 0, 0, 0]],
                          [[7, 8, 0, 0, 0, 0, 0, 0],
                           [8, 0, 0, 0, 0, 0, 0, 0]]]]])
    self._VerifyValues(
        image,
        ksizes=[2, 2, 2],
        strides=[1, 1, 1],
        padding="SAME",
        patches=patches)

if __name__ == "__main__":
  test.main()
