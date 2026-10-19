# [MODULE DOC]
# Tests for the extract-image-patches op on XLA devices. Defines one test class whose cases cover
# kernel size, stride, dilation rate and padding variants. No import-time side effects.
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
"""Functional tests for ExtractImagePatches op."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ExtractImagePatches(xla_test.XLATestCase):
  # [CLASS DOC]
  # Functional tests of image patch extraction in the XLA test scope; a shared verification helper
  # checks each input-output pair.
  """Functional tests for ExtractImagePatches op."""

  def _VerifyValues(self, image, ksizes, strides, rates, padding, patches):
    # [FUNCTION DOC]
    # HELPER: shared by all tests to check patch extraction results.
    # Takes an image (batch, rows, cols, depth), kernel sizes, strides and rates (each two values:
    # rows and columns), a padding mode and the expected patches. Pads each size list with ones for
    # the batch and depth dimensions, extracts patches on the test device in a session, and asserts
    # the result is close to the expected patches.
    """Tests input-output pairs for the ExtractImagePatches op.

    Args:
      image: Input tensor with shape: [batch, in_rows, in_cols, depth].
      ksizes: Patch size specified as: [ksize_rows, ksize_cols].
      strides: Output strides, specified as [stride_rows, stride_cols].
      rates: Atrous rates, specified as [rate_rows, rate_cols].
      padding: Padding type.
      patches: Expected output.
    """
    ksizes = [1] + ksizes + [1]
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]

    with self.session():
      image_placeholder = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        out_tensor = array_ops.extract_image_patches(
            image_placeholder,
            ksizes=ksizes,
            strides=strides,
            rates=rates,
            padding=padding,
            name="im2col")
      feed_dict = {image_placeholder: image}
      self.assertAllClose(patches, out_tensor.eval(feed_dict=feed_dict))

  def testKsize1x1Stride1x1Rate1x1(self):
    # [FUNCTION DOC]
    # 1x1 kernel, stride 1, rate 1 on a 2x3x4x5 image: the output equals the input, for both valid
    # and same padding.
    """Verifies that for 1x1 kernel the output equals the input."""
    # [2, 3, 4, 5]
    image = np.reshape(range(120), [2, 3, 4, 5])
    # [2, 3, 4, 5]
    patches = np.reshape(range(120), [2, 3, 4, 5])
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1],
          strides=[1, 1],
          rates=[1, 1],
          padding=padding,
          patches=patches)

  def testKsize1x1Stride2x3Rate1x1(self):
    # [FUNCTION DOC]
    # 1x1 kernel with strides 2 and 3: output equals the input subsampled every 2 rows and 3
    # columns, for both paddings.
    """Test for 1x1 kernel and strides."""
    # [2, 4, 5, 3]
    image = np.reshape(range(120), [2, 4, 5, 3])
    # [2, 2, 2, 3]
    patches = image[:, ::2, ::3, :]
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1],
          strides=[2, 3],
          rates=[1, 1],
          padding=padding,
          patches=patches)

  def testKsize2x2Stride1x1Rate1x1Valid(self):
    # [FUNCTION DOC]
    # 2x2 kernel with valid padding on a 2x2 image: one patch holding all four values.
    """Test for 2x2 kernel with VALID padding."""
    # [1, 2, 2, 1]
    image = [[[[1], [2]], [[3], [4]]]]
    # [1, 1, 1, 4]
    patches = [[[[1, 2, 3, 4]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        patches=patches)

  def testKsize2x2Stride1x1Rate1x1Same(self):
    # [FUNCTION DOC]
    # 2x2 kernel with same padding on a 2x2 image: four patches, positions beyond the border zero.
    """Test for 2x2 kernel with SAME padding."""
    # [1, 2, 2, 1]
    image = [[[[1], [2]], [[3], [4]]]]
    # [1, 2, 2, 4]
    patches = [[[[1, 2, 3, 4], [2, 0, 4, 0]], [[3, 4, 0, 0], [4, 0, 0, 0]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        patches=patches)

  def testKsize2x2Stride1x1Rate2x2Valid(self):
    # [FUNCTION DOC]
    # 2x2 kernel with dilation rate 2 and valid padding on a 4x4 image: 2x2 patches of values taken
    # two apart.
    """Test for 2x2 kernel with 2x2 dilation."""
    # [1, 2, 2, 1]
    image = np.arange(16).reshape(1, 4, 4, 1).astype(np.float32)
    # [1, 2, 2, 4]
    patches = [[[[0, 2, 8, 10], [1, 3, 9, 11]],
                [[4, 6, 12, 14], [5, 7, 13, 15]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[2, 2],
        padding="VALID",
        patches=patches)

  def testKsize2x2Stride1x1Rate1x1ValidDepth2(self):
    # [FUNCTION DOC]
    # 2x2 kernel with valid padding on a 2x2 image of depth 2: one patch of eight values with the
    # depth values interleaved per position.
    """Test for 2x2 kernel with VALID padding."""
    # [1, 2, 2, 2]
    image = [[[[1, 5], [2, 6]], [[3, 7], [4, 8]]]]
    # [1, 1, 1, 8]
    patches = [[[[1, 5, 2, 6, 3, 7, 4, 8]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        patches=patches)


if __name__ == "__main__":
  test.main()
