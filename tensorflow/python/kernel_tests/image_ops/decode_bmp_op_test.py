# [MODULE DOC]
# Unit tests for the BMP decoding image operation. Contains one test case class that feeds
# hand-built byte strings of tiny two by two BMP files (24-bit colour and 8-bit grayscale) to the
# decoder and compares the decoded pixel arrays with the known expected pixel values. No
# import-time side effects besides the test main entry point when run as a script.
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
"""Tests for DecodeBmpOp."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


class DecodeBmpOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the BMP decode op. Each test constructs the raw bytes of a minimal BMP file
  # inline (no external test data files), decodes it in a cached session and checks the
  # resulting pixel array.

  def testex1(self):
    # [FUNCTION DOC]
    # Test: decodes a 24-bit two by two BMP (bytes taken from a Wikipedia example, bottom-up
    # row order with row padding) as a byte string tensor, squeezes the batch-like dimensions,
    # and asserts the result equals the expected red, green, blue, white pixel array.
    img_bytes = [[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [255, 255, 255]]]
    # Encoded BMP bytes from Wikipedia
    # BMP header bytes: https://en.wikipedia.org/wiki/List_of_file_signatures
    encoded_bytes = [
        0x42, 0x4d,
        0x46, 0, 0, 0,
        0, 0,
        0, 0,
        0x36, 0, 0, 0,
        0x28, 0, 0, 0,
        0x2, 0, 0, 0,
        0x2, 0, 0, 0,
        0x1, 0,
        0x18, 0,
        0, 0, 0, 0,
        0x10, 0, 0, 0,
        0x13, 0xb, 0, 0,
        0x13, 0xb, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0xff,
        0xff, 0xff, 0xff,
        0, 0,
        0xff, 0, 0,
        0, 0xff, 0,
        0, 0,
    ]

    byte_string = bytes(bytearray(encoded_bytes))
    img_in = constant_op.constant(byte_string, dtype=dtypes.string)
    decode = array_ops.squeeze(image_ops.decode_bmp(img_in))

    with self.cached_session():
      decoded = self.evaluate(decode)
      self.assertAllEqual(decoded, img_bytes)

  def testGrayscale(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the in-code TODO states the decode op does not validate the input magic bytes,
    # so this test does not cover header validation (the file signature bytes are not checked).
    # Test: decodes an 8-bit grayscale two by two BMP built from a byte list and asserts the
    # output is a two by two by one array with columns alternating 255 and 0 on both rows.
    img_bytes = [[[255], [0]], [[255], [0]]]
    # BMP header bytes: https://en.wikipedia.org/wiki/List_of_file_signatures
    encoded_bytes = [
        0x42,
        0x4d,
        0x3d,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0x36,
        0,
        0,
        0,
        0x28,
        0,
        0,
        0,
        0x2,
        0,
        0,
        0,
        0x2,
        0,
        0,
        0,
        0x1,
        0,
        0x8,
        0,
        0,
        0,
        0,
        0,
        0x10,
        0,
        0,
        0,
        0x13,
        0xb,
        0,
        0,
        0x13,
        0xb,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0xff,
        0,
        0,
        0,
        0xff,
        0,
        0,
        0,
    ]

    byte_string = bytes(bytearray(encoded_bytes))
    img_in = constant_op.constant(byte_string, dtype=dtypes.string)
    # TODO(b/159600494): Currently, `decode_bmp` op does not validate input
    # magic bytes.
    decode = image_ops.decode_bmp(img_in)

    with self.cached_session():
      decoded = self.evaluate(decode)
      self.assertAllEqual(decoded, img_bytes)


if __name__ == "__main__":
  test.main()
