# [MODULE DOC]
# Tests for the legacy (v1) image summary op with float and uint8 images of several channel
# depths, including NaN handling and decoding of the encoded PNG. No import-time side effects.

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
"""Tests for summary V1 image op."""

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import image_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.summary import summary


class SummaryV1ImageOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy image summary op on float and unsigned 8-bit images with 1, 3 and 4 channels,
  # checking decoded pixels and serialized summary metadata.

  def _AsSummary(self, s):
    # [FUNCTION DOC]
    # HELPER: parses a serialized summary string (bytes) and returns the summary message.
    summ = summary_pb2.Summary()
    summ.ParseFromString(s)
    return summ

  def _CheckProto(self, image_summ, shape):
    # [FUNCTION DOC]
    # HELPER: checks the non-image parts of a summary message.
    # Clears the encoded image data from each value, then asserts the message equals the expected text
    # for the first three images: tags img/image/0..2 with height, width and colorspace from shape.
    """Verify that the non-image parts of the image_summ proto match shape."""
    # Only the first 3 images are returned.
    for v in image_summ.value:
      v.image.ClearField("encoded_image_string")
    expected = "\n".join("""
        value {
          tag: "img/image/%d"
          image { height: %d width: %d colorspace: %d }
        }""" % ((i,) + shape[1:]) for i in range(3))
    self.assertProtoEquals(expected, image_summ)

  @test_util.run_deprecated_v1
  def testImageSummary(self):
    # [FUNCTION DOC]
    # Graph-mode (deprecated v1) test. For depths 1, 3, 4 and both positive-only and signed random
    # float images with one NaN pixel, summarizes the batch, decodes the first PNG and asserts the NaN
    # pixel became the bad color and the rest matches the rescaled values within tolerance. Also
    # checks the summary is scalar and the proto metadata.
    for depth in (1, 3, 4):
      for positive in False, True:
        with self.session(graph=ops.Graph()) as sess:
          shape = (4, 5, 7) + (depth,)
          bad_color = [255, 0, 0, 255][:depth]
          # Build a mostly random image with one nan
          const = np.random.randn(*shape).astype(np.float32)
          const[0, 1, 2] = 0  # Make the nan entry not the max
          if positive:
            const = 1 + np.maximum(const, 0)
            scale = 255 / const.reshape(4, -1).max(axis=1)
            offset = 0
          else:
            scale = 127 / np.abs(const.reshape(4, -1)).max(axis=1)
            offset = 128
          adjusted = np.floor(scale[:, None, None, None] * const + offset)
          const[0, 1, 2, depth // 2] = np.nan

          # Summarize
          summ = summary.image("img", const)
          value = self.evaluate(summ)
          self.assertEqual([], summ.get_shape())
          image_summ = self._AsSummary(value)

          # Decode the first image and check consistency
          image = image_ops.decode_png(image_summ.value[0]
                                       .image.encoded_image_string).eval()
          self.assertAllEqual(image[1, 2], bad_color)
          image[1, 2] = adjusted[0, 1, 2]
          self.assertAllClose(image, adjusted[0], rtol=2e-5, atol=2e-5)

          # Check the rest of the proto
          self._CheckProto(image_summ, shape)

  @test_util.run_deprecated_v1
  def testImageSummaryUint8(self):
    # [FUNCTION DOC]
    # Graph-mode test. For depths 1, 3, 4 summarizes a random uint8 image batch and asserts the dtype,
    # scalar summary, exact pixel equality of the decoded first image and the proto metadata.
    np.random.seed(7)
    for depth in (1, 3, 4):
      with self.session(graph=ops.Graph()) as sess:
        shape = (4, 5, 7) + (depth,)

        # Build a random uint8 image
        images = np.random.randint(256, size=shape).astype(np.uint8)
        tf_images = ops.convert_to_tensor(images)
        self.assertEqual(tf_images.dtype, dtypes.uint8)

        # Summarize
        summ = summary.image("img", tf_images)
        value = self.evaluate(summ)
        self.assertEqual([], summ.get_shape())
        image_summ = self._AsSummary(value)

        # Decode the first image and check consistency.
        # Since we're uint8, everything should be exact.
        image = image_ops.decode_png(image_summ.value[0]
                                     .image.encoded_image_string).eval()
        self.assertAllEqual(image, images[0])

        # Check the rest of the proto
        self._CheckProto(image_summ, shape)


if __name__ == "__main__":
  test.main()
