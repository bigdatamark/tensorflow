# [MODULE DOC]
# Unit tests for the generic image decoding op, which auto-detects BMP, GIF, JPEG and PNG data.
# The tests read real sample images from the core library test data directories (paths are
# relative to the repository root, so tests must run from there) and compare the generic
# decoder with the format-specific decoders, check shapes and channel handling, and check error
# behavior for bad channel counts and non-image bytes. Defines a module-level path prefix for
# the test data. Importing the neural network gradient module is for side effects only.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for decode_image."""

import os.path

import numpy as np

from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

prefix_path = "tensorflow/core/lib"


class DecodeImageOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the generic decode_image op. Each test reads one sample image file of a given
  # format, decodes it through the generic op and through the format-specific op, and compares.

  def testBmp(self):
    # [FUNCTION DOC]
    # Test: reads a real BMP sample file, decodes it with the generic and the BMP-specific
    # decoder, asserts the file is 4194 bytes long and that both decoders give identical output.
    # Read a real bmp and verify shape
    path = os.path.join(prefix_path, "bmp", "testdata", "lena.bmp")
    with self.session():
      bmp0 = io_ops.read_file(path)
      image0 = image_ops.decode_image(bmp0)
      image1 = image_ops.decode_bmp(bmp0)
      bmp0, image0, image1 = self.evaluate([bmp0, image0, image1])
      self.assertEqual(len(bmp0), 4194)
      self.assertAllEqual(image0, image1)

  def testGif(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the bad-channel-count assertion sits inside the per-frame loop (indented under
    # it), so the same failure check is repeated for every one of the twelve frames; it was most
    # likely meant to run once after the loop.
    # Test: decodes a real animated GIF with the generic and GIF-specific decoders, asserts a
    # four-dimensional result of twelve frames, forty rows, twenty columns, three channels, that
    # both decoders agree, and that each frame equals a synthetic expected frame (a white stripe
    # moving across the image, wrapping from vertical to horizontal strips). Also asserts that
    # requesting a single channel raises an invalid-argument error.
    # Read some real GIFs
    path = os.path.join(prefix_path, "gif", "testdata", "scan.gif")
    width = 20
    height = 40
    stride = 5
    shape = (12, height, width, 3)

    with self.session():
      gif0 = io_ops.read_file(path)
      image0 = image_ops.decode_image(gif0)
      image1 = image_ops.decode_gif(gif0)
      gif0, image0, image1 = self.evaluate([gif0, image0, image1])

      self.assertEqual(image0.shape, shape)
      self.assertAllEqual(image0, image1)

      for frame_idx, frame in enumerate(image0):
        gt = np.zeros(shape[1:], dtype=np.uint8)
        start = frame_idx * stride
        end = (frame_idx + 1) * stride
        if end <= width:
          gt[:, start:end, :] = 255
        else:
          start -= width
          end -= width
          gt[start:end, :, :] = 255

        self.assertAllClose(frame, gt)

        with self.assertRaises(errors_impl.InvalidArgumentError):
          bad_channels = image_ops.decode_image(gif0, channels=1)
          self.evaluate(bad_channels)

  def testJpeg(self):
    # [FUNCTION DOC]
    # Test: decodes a real JPEG sample with the generic and JPEG-specific decoders; asserts the
    # file is 3771 bytes, the image shape is 256 by 128 by 3 and both decoders agree. Also
    # asserts that asking for four channels raises an invalid-argument error.
    # Read a real jpeg and verify shape
    path = os.path.join(prefix_path, "jpeg", "testdata", "jpeg_merge_test1.jpg")
    with self.session():
      jpeg0 = io_ops.read_file(path)
      image0 = image_ops.decode_image(jpeg0)
      image1 = image_ops.decode_jpeg(jpeg0)
      jpeg0, image0, image1 = self.evaluate([jpeg0, image0, image1])
      self.assertEqual(len(jpeg0), 3771)
      self.assertEqual(image0.shape, (256, 128, 3))
      self.assertAllEqual(image0, image1)

      with self.assertRaises(errors_impl.InvalidArgumentError):
        bad_channels = image_ops.decode_image(jpeg0, channels=4)
        self.evaluate(bad_channels)

  def testPng(self):
    # [FUNCTION DOC]
    # Test: decodes a real grayscale PNG with requested channel counts 0 (keep source), 1, 3 and
    # 4 using the generic and PNG-specific decoders; asserts the shape is 26 by 51 with the
    # requested channels (or the source's one channel when zero was requested) and that both
    # decoders agree.
    # Read some real PNGs, converting to different channel numbers
    inputs = [(1, "lena_gray.png")]
    for channels_in, filename in inputs:
      for channels in 0, 1, 3, 4:
        with self.cached_session() as sess:
          path = os.path.join(prefix_path, "png", "testdata", filename)
          png0 = io_ops.read_file(path)
          image0 = image_ops.decode_image(png0, channels=channels)
          image1 = image_ops.decode_png(png0, channels=channels)
          png0, image0, image1 = self.evaluate([png0, image0, image1])
          self.assertEqual(image0.shape, (26, 51, channels or channels_in))
          self.assertAllEqual(image0, image1)

  @test_util.run_deprecated_v1
  def testInvalidBytes(self):
    # [FUNCTION DOC]
    # Test (graph mode only, via the deprecated-v1 decorator): decoding a byte string that is not
    # any supported image format must raise an invalid-argument error when evaluated.
    image_bytes = b"ThisIsNotAnImage!"
    decode = image_ops.decode_image(image_bytes)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(decode)


if __name__ == "__main__":
  test.main()
