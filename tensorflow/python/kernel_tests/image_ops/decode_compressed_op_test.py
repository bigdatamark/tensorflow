# [MODULE DOC]
# Unit tests for the decode-compressed parsing op (decompression of string tensors using zlib,
# gzip, zstandard or no compression), including its combination with the raw-bytes decoding op.
# Uses the standard compression libraries (and the third-party zstandard package, which must be
# installed) to produce the compressed inputs. The module header text mentions the raw decode op,
# but the file's main subject is decompression. No import-time side effects other than the test
# main entry point when run as a script.
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
"""Tests for DecodeRaw op from parsing_ops."""

import gzip
import io
import zlib

import zstandard as zstd

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class DecodeCompressedOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the decode-compressed op. Tests loop over the compression types zlib, gzip,
  # zstandard and none; a shared helper compresses test input with the matching library.

  def _compress(self, bytes_in, compression_type):
    # [FUNCTION DOC]
    # HELPER: produces compressed test inputs for the decompression tests in this class.
    # Inputs: bytes_in is the raw byte string; compression_type is "ZLIB", "ZSTD", an empty/null
    # value meaning no compression, or any other value which is treated as gzip.
    # Returns the compressed byte string (the unchanged input for no compression). No side
    # effects; gzip output is built in an in-memory buffer.
    if not compression_type:
      return bytes_in
    elif compression_type == "ZLIB":
      return zlib.compress(bytes_in)
    elif compression_type == "ZSTD":
      return zstd.compress(bytes_in)
    else:
      out = io.BytesIO()
      with gzip.GzipFile(fileobj=out, mode="wb") as f:
        f.write(bytes_in)
      return out.getvalue()

  def testDecompressShapeInference(self):
    # [FUNCTION DOC]
    # Test: in graph mode, for each compression type, decompressing a string placeholder of
    # shape two must give a result whose statically known shape is also two (the op preserves
    # input shape during shape inference). No values are evaluated.
    with ops.Graph().as_default():
      for compression_type in ["ZLIB", "GZIP", "ZSTD", ""]:
        with self.cached_session():
          in_bytes = array_ops.placeholder(dtypes.string, shape=[2])
          decompressed = parsing_ops.decode_compressed(
              in_bytes, compression_type=compression_type)
          self.assertEqual([2], decompressed.get_shape().as_list())

  def testDecompress(self):
    # [FUNCTION DOC]
    # Test: for each compression type, compresses two short strings, decompresses the pair with
    # the decode-compressed op and asserts the original two strings are recovered in order.
    for compression_type in ["ZLIB", "GZIP", "ZSTD", ""]:
      with self.cached_session():

        def decode(in_bytes, compression_type=compression_type):
          # [FUNCTION DOC]
          # HELPER: local wrapper that applies the decode-compressed op with the compression type
          # bound from the enclosing loop iteration. Input: in_bytes, a list or tensor of
          # compressed strings. Returns the tensor of decompressed strings.
          return parsing_ops.decode_compressed(
              in_bytes, compression_type=compression_type)

        in_val = [self._compress(b"AaAA", compression_type),
                  self._compress(b"bBbb", compression_type)]
        result = self.evaluate(decode(in_val))
        self.assertAllEqual([b"AaAA", b"bBbb"], result)

  def testDecompressWithRaw(self):
    # [FUNCTION DOC]
    # Test: for each compression type, compresses a four-byte string, decompresses it and then
    # decodes the bytes as 16-bit integers; asserts the result is one row of two little-endian
    # integers built from the character codes of the bytes.
    for compression_type in ["ZLIB", "GZIP", "ZSTD", ""]:
      with self.cached_session():

        def decode(in_bytes, compression_type=compression_type):
          # [FUNCTION DOC]
          # HELPER: local wrapper that decompresses a list of compressed strings with the bound
          # compression type, then reinterprets the resulting bytes as 16-bit integers.
          # Input: in_bytes, list or tensor of compressed strings. Returns an integer tensor.
          decompressed = parsing_ops.decode_compressed(in_bytes,
                                                       compression_type)
          return parsing_ops.decode_raw(decompressed, out_type=dtypes.int16)

        result = self.evaluate(
            decode([self._compress(b"AaBC", compression_type)]))

        self.assertAllEqual(
            [[ord("A") + ord("a") * 256, ord("B") + ord("C") * 256]], result)


if __name__ == "__main__":
  test.main()
