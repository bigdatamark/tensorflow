# -*- coding: utf-8 -*-
# [MODULE DOC]
# Tests for the ragged unicode decode and split operations (unicode_decode, unicode_split and
# their with-offsets variants) plus the raw generated decode op. Covers scalars, vectors,
# ragged nests, sparse and padded conversion, error modes (replace, ignore, strict),
# control-character replacement, several input encodings and error conditions. Contains small
# module-level helpers that compute expected values in plain Python. Runs every test class in
# both graph and eager modes. No import-time side effects besides decorators.
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
"""Tests for unicode_decode and unicode_decode_with_splits."""

from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test


def _nested_encode(x, encoding):
  # [FUNCTION DOC]
  # HELPER: builds test inputs for the unicode decode/split tests.
  # Recursively walks a nested list of text strings and encodes each string to bytes with the
  # given encoding name. Returns a nested list with the same structure holding byte strings.
  """Encode each string in a nested list with `encoding`."""
  if isinstance(x, list):
    return [_nested_encode(v, encoding) for v in x]
  else:
    return x.encode(encoding)


def _nested_codepoints(x):
  # [FUNCTION DOC]
  # HELPER: computes expected codepoints for the unicode decode tests.
  # Recursively maps each string in a nested list to a list of its integer codepoints by
  # encoding as big-endian 32-bit and combining each group of four bytes. Handles byte values
  # that appear as characters. Returns a nested list of integer lists.
  """Replace each string in a nested list with a list of its codepoints."""
  # Works for Python 2 and 3, and for both UCS2 and UCS4 builds
  if isinstance(x, list):
    return [_nested_codepoints(v) for v in x]
  else:
    b = list(x.encode("utf-32-be"))
    if any(isinstance(c, str) for c in b):
      b = [ord(c) for c in b]
    return [(b0 << 24) + (b1 << 16) + (b2 << 8) + b3
            for b0, b1, b2, b3 in zip(b[::4], b[1::4], b[2::4], b[3::4])]


def _nested_offsets(x, encoding):
  # [FUNCTION DOC]
  # HELPER: computes expected byte offsets for the with-offsets tests.
  # Recursively maps each string in a nested list to the list of starting byte offsets of its
  # characters when encoded with the given encoding (cumulative sums of per-character encoded
  # lengths, starting at zero). An empty string gives an empty list.
  """Replace each string in a nested list with a list of start offsets."""
  if isinstance(x, list):
    return [_nested_offsets(v, encoding) for v in x]
  else:
    if not x:
      return []
    encoded_x = x.encode("utf-32-be")
    encoded_chars = [encoded_x[i:i + 4] for i in range(0, len(encoded_x), 4)]
    char_lens = [
        len(c.decode("utf-32-be").encode(encoding)) for c in encoded_chars
    ]
    return [0] + np.cumsum(char_lens).tolist()[:-1]


def _nested_splitchars(x, encoding):
  # [FUNCTION DOC]
  # HELPER: computes expected character substrings for the unicode split tests.
  # Recursively maps each string in a nested list to a list of single-character byte strings
  # encoded with the given encoding. Has separate code paths for Python 2 style byte strings
  # and Python 3. Returns a nested list of byte-string lists.
  """Replace each string in a nested list with a list of char substrings."""
  if isinstance(x, list):
    return [_nested_splitchars(v, encoding) for v in x]
  else:
    b = x.encode("utf-32-be")
    chars = zip(b[::4], b[1::4], b[2::4], b[3::4])
    if str is bytes:
      return [b"".join(c).decode("utf-32-be").encode(encoding) for c in chars]
    else:
      return [bytes(c).decode("utf-32-be").encode(encoding) for c in chars]


def _make_sparse_tensor(indices, values, dense_shape, dtype=np.int32):
  # [FUNCTION DOC]
  # HELPER: builds expected sparse results for the sparse-output tests.
  # Takes lists for indices, values and dense shape plus an optional value dtype (default
  # 32-bit integer) and returns a sparse tensor value with 64-bit indices and shape.
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64), np.array(values, dtype),
      np.array(dense_shape, np.int64))


@test_util.run_all_in_graph_and_eager_modes
class UnicodeDecodeTest(test_util.TensorFlowTestCase,
                        parameterized.TestCase):
  # [CLASS DOC]
  # Tests for ragged_string_ops.unicode_decode and unicode_decode_with_offsets, and the raw
  # generated decode op. Run in both graph and eager modes. Verifies codepoints and byte
  # offsets for scalar, vector and ragged inputs, sparse and padded conversions, error
  # handling modes and multiple encodings.

  def testScalarDecode(self):
    # [FUNCTION DOC]
    # Decodes a scalar UTF-8 string of four Chinese characters. Asserts the result is the
    # flat list of their codepoints.
    text = constant_op.constant(u"仅今年前".encode("utf-8"))
    chars = ragged_string_ops.unicode_decode(text, "utf-8")
    self.assertAllEqual(chars, [ord(c) for c in u"仅今年前"])

  def testScalarDecodeWithOffset(self):
    # [FUNCTION DOC]
    # Decodes the same scalar string with offsets. Asserts the codepoints and that the byte
    # start offsets are zero, three, six and nine (three bytes per character).
    text = constant_op.constant(u"仅今年前".encode("utf-8"))
    chars, starts = ragged_string_ops.unicode_decode_with_offsets(text, "utf-8")
    self.assertAllEqual(chars, [ord(c) for c in u"仅今年前"])
    self.assertAllEqual(starts, [0, 3, 6, 9])

  def testVectorDecode(self):
    # [FUNCTION DOC]
    # Decodes a vector of two strings (Chinese text and ASCII text). Asserts a ragged result
    # with the codepoints of each string as one row.
    text = constant_op.constant([u"仅今年前".encode("utf-8"), b"hello"])
    chars = ragged_string_ops.unicode_decode(text, "utf-8")
    expected_chars = [[ord(c) for c in u"仅今年前"],
                      [ord(c) for c in u"hello"]]
    self.assertAllEqual(chars, expected_chars)

  def testVectorDecodeWithOffset(self):
    # [FUNCTION DOC]
    # Decodes the two-string vector with offsets. Asserts the per-row codepoints and the
    # per-row byte start offsets (multi-byte characters step by three, ASCII by one).
    text = constant_op.constant([u"仅今年前".encode("utf-8"), b"hello"])
    chars, starts = ragged_string_ops.unicode_decode_with_offsets(text, "utf-8")
    expected_chars = [[ord(c) for c in u"仅今年前"],
                      [ord(c) for c in u"hello"]]
    self.assertAllEqual(chars, expected_chars)
    self.assertAllEqual(starts, [[0, 3, 6, 9], [0, 1, 2, 3, 4]])

  @parameterized.parameters([
      {"texts": u"仅今年前"},
      {"texts": [u"G\xf6\xf6dnight", u"\U0001f60a"]},
      {"texts": ["Hello", "world", "", u"👍"]},
      {"texts": [["Hi", "there"], ["", u"\U0001f60a"]], "ragged_rank": 0},
      {"texts": [["Hi", "there", ""], [u"😊"]], "ragged_rank": 1},
      {"texts": [[[u"😊", u"🤠🧐"], []], [[u"🤓👻🤖"]]], "ragged_rank": 2},
      {"texts": [[[u"😊"], [u"🤠🧐"]], [[u"🤓👻🤖"]]], "ragged_rank": 1},
      {"texts": [[[u"😊"], [u"🤠🧐"]], [[u"🤓"], [u"👻"]]], "ragged_rank": 0},
      {"texts": []}
  ])  # pyformat: disable
  def testBasicDecode(self, texts, ragged_rank=None):
    # [FUNCTION DOC]
    # Parameterized over nested text inputs (scalar, vector, ragged of various ranks, empty).
    # texts is the nested text; ragged_rank optionally fixes the ragged rank of the input.
    # Encodes to a ragged constant, decodes as UTF-8 and asserts equality with the codepoints
    # computed by the helper.
    input_tensor = ragged_factory_ops.constant_value(
        _nested_encode(texts, "UTF-8"), ragged_rank=ragged_rank, dtype=bytes)
    result = ragged_string_ops.unicode_decode(input_tensor, "UTF-8")
    expected = _nested_codepoints(texts)
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      {"texts": u"仅今年前"},
      {"texts": [u"G\xf6\xf6dnight", u"\U0001f60a"]},
      {"texts": ["Hello", "world", "", u"👍"]},
      {"texts": [["Hi", "there"], ["", u"\U0001f60a"]], "ragged_rank": 0},
      {"texts": [["Hi", "there", ""], [u"😊"]], "ragged_rank": 1},
      {"texts": [[[u"😊", u"🤠🧐"], []], [[u"🤓👻🤖"]]], "ragged_rank": 2},
      {"texts": []}
  ])  # pyformat: disable
  def testBasicDecodeWithOffsets(self, texts, ragged_rank=None):
    # [FUNCTION DOC]
    # Same inputs as the basic decode test but uses the with-offsets variant. Asserts the first
    # output equals the expected codepoints and the second equals the expected start offsets.
    input_tensor = ragged_factory_ops.constant_value(
        _nested_encode(texts, "UTF-8"), ragged_rank=ragged_rank, dtype=bytes)
    result = ragged_string_ops.unicode_decode_with_offsets(
        input_tensor, "UTF-8")
    expected_codepoints = _nested_codepoints(texts)
    expected_offsets = _nested_offsets(texts, "UTF-8")
    self.assertAllEqual(expected_codepoints, result[0])
    self.assertAllEqual(expected_offsets, result[1])

  def testDocstringExamples(self):
    # [FUNCTION DOC]
    # Reproduces the documented decode example with two strings (one with accented letters, one
    # emoji). Asserts the codepoints from both the plain and with-offsets variants and the
    # expected byte offsets.
    texts = [s.encode("utf8") for s in [u"G\xf6\xf6dnight", u"\U0001f60a"]]
    codepoints1 = ragged_string_ops.unicode_decode(texts, "UTF-8")
    codepoints2, offsets = ragged_string_ops.unicode_decode_with_offsets(
        texts, "UTF-8")
    self.assertAllEqual(
        codepoints1, [[71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]])
    self.assertAllEqual(
        codepoints2, [[71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]])
    self.assertAllEqual(offsets, [[0, 1, 3, 5, 6, 7, 8, 9, 10], [0]])

  @parameterized.parameters([
      dict(
          texts=["Hello", "world", "", u"👍"],
          expected=_make_sparse_tensor(
              indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1],
                       [1, 2], [1, 3], [1, 4], [3, 0]],
              values=[72, 101, 108, 108, 111, 119, 111, 114, 108, 100, 128077],
              dense_shape=[4, 5])),
      dict(
          texts=[["Hi", "there"], ["", u"\U0001f60a"]],
          expected=_make_sparse_tensor(
              indices=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2],
                       [0, 1, 3], [0, 1, 4], [1, 1, 0]],
              values=[72, 105, 116, 104, 101, 114, 101, 128522],
              dense_shape=[2, 2, 5])),
      dict(
          texts=[],
          expected=_make_sparse_tensor(np.zeros([0, 2], np.int64), [], [0, 0])),
  ])
  def testDecodeWithSparseOutput(self, texts, expected):
    # [FUNCTION DOC]
    # Decodes a byte-string array and converts the ragged result to a sparse tensor. texts is
    # the nested text, expected the sparse tensor value. Asserts the result is a sparse tensor
    # whose indices, values and dense shape match.
    input_tensor = np.array(_nested_encode(texts, "UTF-8"), dtype=bytes)
    result = ragged_string_ops.unicode_decode(input_tensor, "UTF-8").to_sparse()
    self.assertIsInstance(result, sparse_tensor.SparseTensor)
    self.assertAllEqual(expected.indices, result.indices)
    self.assertAllEqual(expected.values, result.values)
    self.assertAllEqual(expected.dense_shape, result.dense_shape)

  @parameterized.parameters([
      dict(
          texts=["Hello", "world", "", u"👍"],
          expected=[[72, 101, 108, 108, 111], [119, 111, 114, 108, 100],
                    [-1, -1, -1, -1, -1], [0x1F44D, -1, -1, -1, -1]]),
      dict(
          texts=[["Hi", "there"], ["", u"\U0001f60a"]],
          expected=[[[72, 105, -1, -1, -1], [116, 104, 101, 114, 101]],
                    [[-1, -1, -1, -1, -1], [128522, -1, -1, -1, -1]]],
          ragged_rank=0),
      dict(
          texts=[["Hi", "there", ""], [u"😊"]],
          expected=[[[72, 105, -1, -1, -1],
                     [116, 104, 101, 114, 101],
                     [-1, -1, -1, -1, -1]],
                    [[128522, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1]]]),
      dict(
          texts=[[[u"😊", u"🤠🧐"], []], [[u"🤓👻🤖"]]],
          expected=[
              [[[128522, -1, -1], [129312, 129488, -1]],
               [[-1, -1, -1], [-1, -1, -1]]],
              [[[129299, 128123, 129302], [-1, -1, -1]],
               [[-1, -1, -1], [-1, -1, -1]]]]),
      dict(texts=[], expected=np.zeros([0, 0], np.int64)),
  ])  # pyformat: disable
  def testDecodeWithPaddedOutput(self, texts, expected, ragged_rank=None):
    # [FUNCTION DOC]
    # Decodes ragged text input and converts the result to a dense tensor padded with minus
    # one. Asserts it equals the expected padded nested list of codepoints.
    input_tensor = ragged_factory_ops.constant_value(
        _nested_encode(texts, "UTF-8"), ragged_rank=ragged_rank, dtype=bytes)
    result = ragged_string_ops.unicode_decode(
        input_tensor, "UTF-8").to_tensor(default_value=-1)
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="replace",
          expected=[[0xFFFD],
                    [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [ord('='), ord('='), 0xFFFD, ord('='), ord('=')],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')]]),
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="replace",
          replacement_char=0,
          expected=[[0], [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [ord('='), ord('='), 0, ord('='), ord('=')],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')]]),
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="ignore",
          expected=[[], [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [ord('='), ord('='), ord('='), ord('=')],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')]]),
      dict(
          input=[b"\x00", b"hello", b"==\x01==", b"world",
                 # C1 control characters are not replaced.
                 u"\x80\x9f".encode()],
          input_encoding="UTF-8",
          replace_control_characters=True,
          expected=[[0xFFFD],
                    [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [61, 61, 65533, 61, 61],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')],
                    [0x80, 0x9F]]),
      dict(
          input=[b"\x00", b"hello", b"==\x01==", b"world",
                 u"\u0080\u009f".encode()],
          input_encoding="UTF-8",
          replace_control_characters=True,
          replacement_char=0,
          expected=[[0], [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [ord('='), ord('='), 0, ord('='), ord('=')],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')],
                    [0x80, 0x9F]]),
      dict(
          input=[b"\xed\xa0\xbd",  # single high surrogate
                 b"\xed\xb8\x80",  # single low surrogate
                 b"\xed\xa0\xbd\xed\xb8\x80"],  # surrogate pair
          input_encoding="UTF-8",
          expected=[[0xFFFD, 0xFFFD, 0xFFFD],
                    [0xFFFD, 0xFFFD, 0xFFFD],
                    [0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD]]),
      dict(
          input=["\ufdd0".encode(),  # noncharacter
                 "\ufffe\uffff".encode(),  # last two in BMP
                 "\U0010ffff".encode(),  # last in plane 16 = last in Unicode
                 b"\xc0\x80",  # overlong U+0000 encoding
                 b"\xf4\x90\x80\x80"],  # U+110000, beyond Unicode
          input_encoding="UTF-8",
          expected=[[0xFDD0],
                    [0xFFFE, 0xFFFF],
                    [0x10FFFF],
                    [0xFFFD] * 2,
                    [0xFFFD] * 4]),
  ])  # pyformat: disable
  def testErrorModes(self, expected=None, **args):
    # [FUNCTION DOC]
    # Passes keyword arguments straight to the decode op to exercise invalid-byte handling
    # (replace with default or custom replacement character, ignore), control character
    # replacement, surrogates, noncharacters and overlong encodings. Asserts the expected
    # codepoints.
    result = ragged_string_ops.unicode_decode(**args)
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="replace",
          expected=[[0xFFFD],
                    [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [ord('='), ord('='), 0xFFFD, ord('='), ord('=')],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')]],
          expected_offsets=[[0], [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="replace",
          replacement_char=0,
          expected=[[0], [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [ord('='), ord('='), 0, ord('='), ord('=')],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')]],
          expected_offsets=[[0], [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="ignore",
          expected=[[], [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [ord('='), ord('='), ord('='), ord('=')],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')]],
          expected_offsets=[[], [0, 1, 2, 3, 4],
                            [0, 1, 3, 4], [0, 1, 2, 3, 4]]),
      dict(
          input=[b"\x00", b"hello", b"==\x01==", b"world"],
          input_encoding="UTF-8",
          replace_control_characters=True,
          expected=[[0xFFFD],
                    [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [ord('='), ord('='), 0xFFFD, ord('='), ord('=')],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')]],
          expected_offsets=[[0], [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
      dict(
          input=[b"\x00", b"hello", b"==\x01==", b"world"],
          input_encoding="UTF-8",
          replace_control_characters=True,
          replacement_char=0,
          expected=[[0], [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')],
                    [0x3D, 0x3D, 0, 0x3D, 0x3D],
                    [ord('w'), ord('o'), ord('r'), ord('l'), ord('d')]],
          expected_offsets=[[0], [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
      dict(
          input=[b"\xD8\x01"],
          input_encoding="UTF-8",
          replacement_char=0x41,
          expected=[[0x41, 1]],
          expected_offsets=[[0, 1]]),
  ])  # pyformat: disable
  def testErrorModesWithOffsets(self,
                                expected=None,
                                expected_offsets=None,
                                **args):
    # [FUNCTION DOC]
    # Like the error modes test but for the with-offsets decode. Keyword arguments go to the
    # op. Asserts the expected codepoints and the expected byte offsets (ignored invalid bytes
    # leave gaps in the offsets).
    result = ragged_string_ops.unicode_decode_with_offsets(**args)
    self.assertAllEqual(result[0], expected)
    self.assertAllEqual(result[1], expected_offsets)

  @parameterized.parameters(
      ("UTF-8", [u"こんにちは", u"你好", u"Hello"]),
      ("UTF-16-BE", [u"こんにちは", u"你好", u"Hello"]),
      ("UTF-32-BE", [u"こんにちは", u"你好", u"Hello"]),
      ("US-ASCII", [u"Hello", "world"]),
      ("ISO-8859-1", [u"ÀÈÓ", "AEO"]),
      ("SHIFT-JIS", [u"Hello", u"こんにちは"]),
  )
  def testDecodeWithDifferentEncodings(self, encoding, texts):
    # [FUNCTION DOC]
    # Encodes sample texts with the given encoding name (UTF variants, ASCII, Latin-1, Shift
    # JIS), decodes with the same encoding and asserts the codepoints match.
    expected = _nested_codepoints(texts)
    input_tensor = constant_op.constant(_nested_encode(texts, encoding))
    result = ragged_string_ops.unicode_decode(input_tensor, encoding)
    self.assertAllEqual(expected, result)

  @parameterized.parameters(
      ("UTF-8", [u"こんにちは", u"你好", u"Hello"]),
      ("UTF-16-BE", [u"こんにちは", u"你好", u"Hello"]),
      ("UTF-32-BE", [u"こんにちは", u"你好", u"Hello"]),
      ("US-ASCII", [u"Hello", "world"]),
      ("ISO-8859-1", [u"ÀÈÓ", "AEO"]),
      ("SHIFT-JIS", [u"Hello", u"こんにちは"]),
  )
  def testDecodeWithOffsetsWithDifferentEncodings(self, encoding, texts):
    # [FUNCTION DOC]
    # Same as the different-encodings decode test but with the with-offsets variant; asserts
    # both codepoints and byte start offsets for the given encoding.
    expected_codepoints = _nested_codepoints(texts)
    expected_offsets = _nested_offsets(texts, encoding)
    input_tensor = constant_op.constant(_nested_encode(texts, encoding))
    result = ragged_string_ops.unicode_decode_with_offsets(
        input_tensor, encoding)
    self.assertAllEqual(expected_codepoints, result[0])
    self.assertAllEqual(expected_offsets, result[1])

  @parameterized.parameters([
      dict(input=[b"\xFEED"],
           errors="strict",
           input_encoding="UTF-8",
           exception=errors.InvalidArgumentError,
           message="Invalid formatting on input string"),
      dict(input="x",
           input_encoding="UTF-8",
           replacement_char=11141111,
           exception=errors.InvalidArgumentError,
           message="replacement_char out of unicode codepoint range"),
      dict(input="x",
           input_encoding="UTF-8",
           errors="oranguatan",
           exception=(ValueError, errors.InvalidArgumentError)),
  ])  # pyformat: disable
  def testExceptions(self, exception=None, message=None, **args):
    # [FUNCTION DOC]
    # Calls decode with invalid input (strict mode on malformed bytes, out-of-range replacement
    # character, unknown error mode). Asserts the expected exception type and message pattern.
    with self.assertRaisesRegex(exception, message):
      self.evaluate(ragged_string_ops.unicode_decode(**args))

  def testUnknownRankError(self):
    # [FUNCTION DOC]
    # Skipped in eager mode. In graph mode feeds a placeholder of unknown rank to decode and
    # asserts a value error saying the input rank must be statically known.
    if context.executing_eagerly():
      return
    s = array_ops.placeholder(dtypes.string)
    message = "Rank of `input` must be statically known."
    with self.assertRaisesRegex(ValueError, message):
      self.evaluate(ragged_string_ops.unicode_decode(s, input_encoding="UTF-8"))

  @parameterized.parameters([
      dict(
          doc="Single string",
          input=_nested_encode([u"仅今年前"], "utf-8"),
          input_encoding="UTF-8",
          expected_char_values=_nested_codepoints(u"仅今年前"),
          expected_row_splits=[0, 4],
          expected_char_to_byte_starts=[0, 3, 6, 9]),
      dict(
          doc="Multiple strings",
          input=_nested_encode([u"仅今年前", u"你好"], "utf-8"),
          input_encoding="UTF-8",
          expected_char_values=_nested_codepoints(u"仅今年前你好"),
          expected_row_splits=[0, 4, 6],
          expected_char_to_byte_starts=[0, 3, 6, 9, 0, 3]),
      dict(
          doc="errors=replace",
          input=b"=\xFE=",
          input_encoding="UTF-8",
          errors="replace",
          expected_char_values=[0x3D, 0xFFFD, 0x3D],
          expected_row_splits=[0, 3],
          expected_char_to_byte_starts=[0, 1, 2]),
      dict(
          doc="errors=ignore",
          input=b"=\xFE=",
          input_encoding="UTF-8",
          errors="ignore",
          expected_char_values=[61, 61],
          expected_row_splits=[0, 2],
          expected_char_to_byte_starts=[0, 2]),
  ])
  def testDecodeGenOp(self,
                      doc,
                      expected_row_splits=None,
                      expected_char_values=None,
                      expected_char_to_byte_starts=None,
                      **args):
    # [FUNCTION DOC]
    # Calls the raw generated decode-with-offsets op with keyword arguments from each case.
    # doc is a label only. Asserts the row splits, character values and character-to-byte
    # start offsets of the result.
    """Test for the c++ interface (gen_string_ops.unicode_decode)."""
    result = gen_string_ops.unicode_decode_with_offsets(**args)
    self.assertAllEqual(expected_row_splits, result.row_splits)
    self.assertAllEqual(expected_char_values, result.char_values)
    self.assertAllEqual(expected_char_to_byte_starts,
                        result.char_to_byte_starts)


@test_util.run_all_in_graph_and_eager_modes
class UnicodeSplitTest(test_util.TensorFlowTestCase,
                       parameterized.TestCase):
  # [CLASS DOC]
  # Tests for ragged_string_ops.unicode_split and unicode_split_with_offsets, run in both graph
  # and eager modes. Mirrors the decode tests but expects per-character byte strings instead
  # of codepoints.

  def testScalarSplit(self):
    # [FUNCTION DOC]
    # Splits a scalar UTF-8 string of four Chinese characters. Asserts the result is the list
    # of each character encoded as bytes.
    text = constant_op.constant(u"仅今年前".encode("UTF-8"))
    chars = ragged_string_ops.unicode_split(text, "UTF-8")
    self.assertAllEqual(chars, [c.encode("UTF-8") for c in u"仅今年前"])

  def testScalarSplitWithOffset(self):
    # [FUNCTION DOC]
    # Splits the scalar string with offsets. Asserts the character byte strings and start
    # offsets zero, three, six and nine.
    text = constant_op.constant(u"仅今年前".encode("UTF-8"))
    chars, starts = ragged_string_ops.unicode_split_with_offsets(text, "UTF-8")
    self.assertAllEqual(chars, [c.encode("UTF-8") for c in u"仅今年前"])
    self.assertAllEqual(starts, [0, 3, 6, 9])

  def testVectorSplit(self):
    # [FUNCTION DOC]
    # Splits a vector of two strings. Asserts a ragged result of per-character byte strings
    # for each row.
    text = constant_op.constant([u"仅今年前".encode("UTF-8"), b"hello"])
    chars = ragged_string_ops.unicode_split(text, "UTF-8")
    expected_chars = [[c.encode("UTF-8") for c in u"仅今年前"],
                      [c.encode("UTF-8") for c in u"hello"]]
    self.assertAllEqual(chars, expected_chars)

  def testVectorSplitWithOffset(self):
    # [FUNCTION DOC]
    # Splits the two-string vector with offsets. Asserts the per-row character byte strings
    # and byte start offsets.
    text = constant_op.constant([u"仅今年前".encode("UTF-8"), b"hello"])
    chars, starts = ragged_string_ops.unicode_split_with_offsets(text, "UTF-8")
    expected_chars = [[c.encode("UTF-8") for c in u"仅今年前"],
                      [c.encode("UTF-8") for c in u"hello"]]
    self.assertAllEqual(chars, expected_chars)
    self.assertAllEqual(starts, [[0, 3, 6, 9], [0, 1, 2, 3, 4]])

  @parameterized.parameters([
      {"texts": u"仅今年前"},
      {"texts": [u"G\xf6\xf6dnight", u"\U0001f60a"]},
      {"texts": ["Hello", "world", "", u"👍"]},
      {"texts": [["Hi", "there"], ["", u"\U0001f60a"]], "ragged_rank": 0},
      {"texts": [["Hi", "there", ""], [u"😊"]], "ragged_rank": 1},
      {"texts": [[[u"😊", u"🤠🧐"], []], [[u"🤓👻🤖"]]], "ragged_rank": 2},
      {"texts": []}
  ])  # pyformat: disable
  def testBasicSplit(self, texts, ragged_rank=None):
    # [FUNCTION DOC]
    # Parameterized over nested text inputs with optional ragged rank. Encodes, splits as UTF-8
    # and asserts equality with the expected nested lists of character byte strings.
    input_tensor = ragged_factory_ops.constant_value(
        _nested_encode(texts, "UTF-8"), ragged_rank=ragged_rank, dtype=bytes)
    result = ragged_string_ops.unicode_split(input_tensor, "UTF-8")
    expected = _nested_splitchars(texts, "UTF-8")
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      {"texts": u"仅今年前"},
      {"texts": [u"G\xf6\xf6dnight", u"\U0001f60a"]},
      {"texts": ["Hello", "world", "", u"👍"]},
      {"texts": [["Hi", "there"], ["", u"\U0001f60a"]], "ragged_rank": 0},
      {"texts": [["Hi", "there", ""], [u"😊"]], "ragged_rank": 1},
      {"texts": [[[u"😊", u"🤠🧐"], []], [[u"🤓👻🤖"]]], "ragged_rank": 2},
      {"texts": []}
  ])  # pyformat: disable
  def testBasicSplitWithOffsets(self, texts, ragged_rank=None):
    # [FUNCTION DOC]
    # Same inputs as the basic split test with the with-offsets variant. Asserts both the
    # character byte strings and the byte start offsets.
    input_tensor = ragged_factory_ops.constant_value(
        _nested_encode(texts, "UTF-8"), ragged_rank=ragged_rank, dtype=bytes)
    result = ragged_string_ops.unicode_split_with_offsets(input_tensor, "UTF-8")
    expected_codepoints = _nested_splitchars(texts, "UTF-8")
    expected_offsets = _nested_offsets(texts, "UTF-8")
    self.assertAllEqual(expected_codepoints, result[0])
    self.assertAllEqual(expected_offsets, result[1])

  def testDocstringExamples(self):
    # [FUNCTION DOC]
    # Reproduces the documented split example with two strings. Asserts the character byte
    # strings from both the plain and with-offsets variants and the expected byte offsets.
    texts = [s.encode("utf8") for s in [u"G\xf6\xf6dnight", u"\U0001f60a"]]
    codepoints1 = ragged_string_ops.unicode_split(texts, "UTF-8")
    codepoints2, offsets = ragged_string_ops.unicode_split_with_offsets(
        texts, "UTF-8")
    self.assertAllEqual(
        codepoints1,
        [[b"G", b"\xc3\xb6", b"\xc3\xb6", b"d", b"n", b"i", b"g", b"h", b"t"],
         [b"\xf0\x9f\x98\x8a"]])
    self.assertAllEqual(
        codepoints2,
        [[b"G", b"\xc3\xb6", b"\xc3\xb6", b"d", b"n", b"i", b"g", b"h", b"t"],
         [b"\xf0\x9f\x98\x8a"]])
    self.assertAllEqual(offsets, [[0, 1, 3, 5, 6, 7, 8, 9, 10], [0]])

  @parameterized.parameters([
      dict(
          texts=["Hello", "world", "", u"👍"],
          expected=_make_sparse_tensor(
              indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1],
                       [1, 2], [1, 3], [1, 4], [3, 0]],
              values=[b"H", b"e", b"l", b"l", b"o",
                      b"w", b"o", b"r", b"l", b"d", b"\xf0\x9f\x91\x8d"],
              dense_shape=[4, 5],
              dtype=bytes)),
      dict(
          texts=[["Hi", "there"], ["", u"\U0001f60a"]],
          expected=_make_sparse_tensor(
              indices=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2],
                       [0, 1, 3], [0, 1, 4], [1, 1, 0]],
              values=[b"H", b"i", b"t", b"h", b"e", b"r", b"e",
                      b"\xf0\x9f\x98\x8a"],
              dense_shape=[2, 2, 5],
              dtype=bytes)),
      dict(
          texts=[],
          expected=_make_sparse_tensor(
              np.zeros([0, 2], np.int64), [], [0, 0], dtype=bytes)),
  ])  # pyformat: disable
  def testSplitWithSparseOutput(self, texts, expected):
    # [FUNCTION DOC]
    # Splits a byte-string array and converts the ragged result to a sparse tensor. Asserts a
    # sparse tensor whose indices, values and dense shape equal the expected ones.
    input_tensor = np.array(_nested_encode(texts, "UTF-8"), dtype=bytes)
    result = ragged_string_ops.unicode_split(input_tensor, "UTF-8").to_sparse()
    self.assertIsInstance(result, sparse_tensor.SparseTensor)
    self.assertAllEqual(expected.indices, result.indices)
    self.assertAllEqual(expected.values, result.values)
    self.assertAllEqual(expected.dense_shape, result.dense_shape)

  @parameterized.parameters([
      dict(
          texts=["Hello", "world", "", u"👍"],
          expected=[[b"H", b"e", b"l", b"l", b"o"],
                    [b"w", b"o", b"r", b"l", b"d"],
                    ["", "", "", "", ""],
                    [b"\xf0\x9f\x91\x8d", "", "", "", ""]]),
      dict(
          texts=[["Hi", "there"], ["", u"\U0001f60a"]],
          expected=[[[b"H", b"i", "", "", ""],
                     [b"t", b"h", b"e", b"r", b"e"]],
                    [["", "", "", "", ""],
                     [b"\xf0\x9f\x98\x8a", "", "", "", ""]]],
          ragged_rank=0),
      dict(
          texts=[["Hi", "there", ""], [u"😊"]],
          expected=[[[b"H", b"i", "", "", ""],
                     [b"t", b"h", b"e", b"r", b"e"],
                     ["", "", "", "", ""]],
                    [[b"\xf0\x9f\x98\x8a", "", "", "", ""],
                     ["", "", "", "", ""],
                     ["", "", "", "", ""]]]),
      dict(
          texts=[[[u"😊", u"🤠🧐"], []], [[u"🤓👻🤖"]]],
          expected=[[[[b"\xf0\x9f\x98\x8a", "", ""],
                      [b"\xf0\x9f\xa4\xa0", b"\xf0\x9f\xa7\x90", ""]],
                     [["", "", ""],
                      ["", "", ""]]],
                    [[[b"\xf0\x9f\xa4\x93", b"\xf0\x9f\x91\xbb",
                       b"\xf0\x9f\xa4\x96"],
                      ["", "", ""]],
                     [["", "", ""],
                      ["", "", ""]]]]),
      dict(texts=[], expected=np.zeros([0, 0], np.int64)),
  ])  # pyformat: disable
  def testSplitWithPaddedOutput(self, texts, expected, ragged_rank=None):
    # [FUNCTION DOC]
    # Splits ragged text and converts to a dense tensor padded with the empty string. Asserts
    # equality with the expected byte-string array.
    input_tensor = ragged_factory_ops.constant_value(
        _nested_encode(texts, "UTF-8"), ragged_rank=ragged_rank, dtype=bytes)
    result = ragged_string_ops.unicode_split(
        input_tensor, "UTF-8").to_tensor(default_value="")
    self.assertAllEqual(np.array(expected, dtype=bytes), result)

  @parameterized.parameters([
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="replace",
          expected=[[b"\xef\xbf\xbd"],
                    [b"h", b"e", b"l", b"l", b"o"],
                    [b"=", b"=", b"\xef\xbf\xbd", b"=", b"="],
                    [b"w", b"o", b"r", b"l", b"d"]]),
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="replace",
          replacement_char=0,
          expected=[[b"\x00"],
                    [b"h", b"e", b"l", b"l", b"o"],
                    [b"=", b"=", b"\x00", b"=", b"="],
                    [b"w", b"o", b"r", b"l", b"d"]]),
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="ignore",
          expected=[[],
                    [b"h", b"e", b"l", b"l", b"o"],
                    [b"=", b"=", b"=", b"="],
                    [b"w", b"o", b"r", b"l", b"d"]]),
  ])  # pyformat: disable
  def testErrorModes(self, expected=None, **args):
    # [FUNCTION DOC]
    # Passes keyword arguments to the split op to exercise invalid-byte handling (replace with
    # default or custom character, ignore). Asserts the expected character byte strings.
    result = ragged_string_ops.unicode_split(**args)
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="replace",
          expected=[[b"\xef\xbf\xbd"],
                    [b"h", b"e", b"l", b"l", b"o"],
                    [b"=", b"=", b"\xef\xbf\xbd", b"=", b"="],
                    [b"w", b"o", b"r", b"l", b"d"]],
          expected_offsets=[[0], [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="replace",
          replacement_char=0,
          expected=[[b"\x00"],
                    [b"h", b"e", b"l", b"l", b"o"],
                    [b"=", b"=", b"\x00", b"=", b"="],
                    [b"w", b"o", b"r", b"l", b"d"]],
          expected_offsets=[[0], [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
      dict(
          input=[b"\xFE", b"hello", b"==\xFF==", b"world"],
          input_encoding="UTF-8",
          errors="ignore",
          expected=[[],
                    [b"h", b"e", b"l", b"l", b"o"],
                    [b"=", b"=", b"=", b"="],
                    [b"w", b"o", b"r", b"l", b"d"]],
          expected_offsets=[[], [0, 1, 2, 3, 4],
                            [0, 1, 3, 4], [0, 1, 2, 3, 4]]),
  ])  # pyformat: disable
  def testErrorModesWithOffsets(self,
                                expected=None,
                                expected_offsets=None,
                                **args):
    # [FUNCTION DOC]
    # Error-mode check for the with-offsets split. Keyword arguments go to the op. Asserts the
    # expected character byte strings and byte offsets.
    result = ragged_string_ops.unicode_split_with_offsets(**args)
    self.assertAllEqual(expected, result[0])
    self.assertAllEqual(expected_offsets, result[1])

  @parameterized.parameters(
      ("UTF-8", [u"こんにちは", u"你好", u"Hello"]),
      ("UTF-16-BE", [u"こんにちは", u"你好", u"Hello"]),
      ("UTF-32-BE", [u"こんにちは", u"你好", u"Hello"]),
  )
  def testSplitWithDifferentEncodings(self, encoding, texts):
    # [FUNCTION DOC]
    # Encodes sample texts in a UTF variant, splits with the same encoding and asserts the
    # character byte strings match the expected ones.
    expected = _nested_splitchars(texts, encoding)
    input_tensor = constant_op.constant(_nested_encode(texts, encoding))
    result = ragged_string_ops.unicode_split(input_tensor, encoding)
    self.assertAllEqual(expected, result)

  @parameterized.parameters(
      ("UTF-8", [u"こんにちは", u"你好", u"Hello"]),
      ("UTF-16-BE", [u"こんにちは", u"你好", u"Hello"]),
      ("UTF-32-BE", [u"こんにちは", u"你好", u"Hello"]),
  )
  def testSplitWithOffsetsWithDifferentEncodings(self, encoding, texts):
    # [FUNCTION DOC]
    # With-offsets variant of the different-encodings split test; asserts both character byte
    # strings and byte start offsets.
    expected_codepoints = _nested_splitchars(texts, encoding)
    expected_offsets = _nested_offsets(texts, encoding)
    input_tensor = constant_op.constant(_nested_encode(texts, encoding))
    result = ragged_string_ops.unicode_split_with_offsets(
        input_tensor, encoding)
    self.assertAllEqual(expected_codepoints, result[0])
    self.assertAllEqual(expected_offsets, result[1])

  @parameterized.parameters([
      dict(input=[b"\xFEED"],
           errors="strict",
           input_encoding="UTF-8",
           exception=errors.InvalidArgumentError,
           message="Invalid formatting on input string"),
      dict(input="x",
           input_encoding="UTF-8",
           replacement_char=11141111,
           exception=errors.InvalidArgumentError,
           message="replacement_char out of unicode codepoint range"),
      dict(input="x",
           input_encoding="UTF-8",
           errors="oranguatan",
           exception=(ValueError, errors.InvalidArgumentError)),
  ])  # pyformat: disable
  def testExceptions(self, exception=None, message=None, **args):
    # [FUNCTION DOC]
    # Calls split with invalid input (strict mode on malformed bytes, out-of-range replacement
    # character, unknown error mode). Asserts the expected exception type and message pattern.
    with self.assertRaisesRegex(exception, message):
      self.evaluate(ragged_string_ops.unicode_split(**args))

  def testUnknownRankError(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite living in the split test class, it calls the decode op, not the
    # split op, so it duplicates the decode class test and never checks split's rank handling.
    # Skipped in eager mode. In graph mode feeds a placeholder of unknown rank and asserts a
    # value error saying the input rank must be statically known.
    if context.executing_eagerly():
      return
    s = array_ops.placeholder(dtypes.string)
    message = "Rank of `input` must be statically known."
    with self.assertRaisesRegex(ValueError, message):
      self.evaluate(ragged_string_ops.unicode_decode(s, input_encoding="UTF-8"))


if __name__ == "__main__":
  test.main()
