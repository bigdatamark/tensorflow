# [MODULE DOC]
# Tests for string_ops.unicode_transcode, which converts byte strings between character
# encodings. Covers UTF-8/16/32 conversions, invalid input handling (replace, ignore, strict),
# replacement characters, control-character replacement, byte order marks, CJK encodings,
# invalid encoding or policy names and input buffer forwarding. Test-only module with no
# import-time side effects.
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
"""Tests for unicode_transcode op."""

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


# Note: for now only tests for algorithmic converters since no file-based
# converters can be loaded. TODO(gbillock): add ability to include at least
# the ucmcore converters from the conversion data sets.
class UnicodeTranscodeOpTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the transcode op. Per the comment above it, only algorithmic
  # converters are tested because file-based converters cannot be loaded here.

  def test_transcode_utf8_simple(self):
    # [FUNCTION DOC]
    # Transcodes ASCII-only strings to UTF-8 from UTF-8, Latin-1 and US-ASCII. Asserts the
    # output equals the input in each case.
    strings = [[b"a", b"abc"], [b"ABC", b"DEF"]]

    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, strings)

      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="ISO-8859-1",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, strings)

      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="US-ASCII",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, strings)

  def test_transcode_utf16_to_utf8(self):
    # [FUNCTION DOC]
    # Transcodes big-endian UTF-16 data (including a surrogate pair) to UTF-8 using the plain
    # UTF-16 encoding name. Asserts the output matches the Python-decoded and re-encoded value.
    strings = [b"\x00a\x00b\x20\xAC", b"\xD8\x01\xDC\x37"]  # U+10437
    expected = [s.decode("UTF-16-BE").encode("UTF-8") for s in strings]

    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-16",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, expected)

  def test_transcode_bad_utf8(self):
    # [FUNCTION DOC]
    # Transcodes a string with a null byte and an invalid byte in replace mode with a space as
    # the replacement. With control-character replacement on, asserts both become spaces; with
    # it off, asserts the null byte is kept and only the invalid byte is replaced.
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=True)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"  ")

      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"\x00 ")

  def test_transcode_bad_utf8_with_some_good(self):
    # [FUNCTION DOC]
    # A string with valid text around one invalid byte, replace mode with a space. Asserts only
    # the invalid byte is replaced.
    bad_string = b"abc\xffabcdefg"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"abc abcdefg")

  def test_transcode_bad_utf8_with_defaults(self):
    # [FUNCTION DOC]
    # Transcodes a null byte plus an invalid byte using default options. Asserts the null byte
    # is kept and the invalid byte becomes the standard Unicode replacement character.
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string, input_encoding="UTF-8", output_encoding="UTF-8")
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"\x00\xef\xbf\xbd")

  def test_transcode_bad_utf8_with_space_replacement(self):
    # [FUNCTION DOC]
    # Same invalid input with a custom space replacement character and default error policy.
    # Asserts the null byte is kept and the invalid byte becomes a space.
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string, input_encoding="UTF-8", output_encoding="UTF-8",
          replacement_char=ord(" "))
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"\x00 ")

  @test_util.run_deprecated_v1
  def test_transcode_bad_utf8_with_strict_errors(self):
    # [FUNCTION DOC]
    # Strict policy on an invalid byte after a null byte. Asserts evaluating raises an op error
    # mentioning invalid formatting of the input string. Graph mode version one.
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="strict")
      with self.assertRaisesOpError(
          "Invalid formatting on input string"):
        self.evaluate(outputs)

  @test_util.run_deprecated_v1
  def test_transcode_bad_utf8_start_with_strict_errors(self):
    # [FUNCTION DOC]
    # Strict policy when the invalid byte is the first byte of the string. Asserts the same
    # invalid-formatting op error. Graph mode version one.
    bad_string = b"\xffabcd"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="strict")
      with self.assertRaisesOpError(
          "Invalid formatting on input string"):
        self.evaluate(outputs)

  def test_transcode_bad_utf8_with_elision_of_malformatting(self):
    # [FUNCTION DOC]
    # Ignore policy on a null byte plus invalid byte. Asserts the invalid byte is dropped and
    # the null byte kept.
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="ignore")
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"\x00")

  def test_transcode_bad_utf8_with_elision_including_control_chars(self):
    # [FUNCTION DOC]
    # Ignore policy with control-character replacement enabled. Asserts both the invalid byte
    # and the null control character are removed, leaving an empty string.
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="ignore",
          replace_control_characters=True)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"")

  def test_transcode_bad_utf8_termination_with_defaults(self):
    # [FUNCTION DOC]
    # A string ending in a truncated multi-byte sequence with default options. Asserts the
    # trailing fragment becomes one replacement character.
    bad_string = b"a\xf0"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string, input_encoding="UTF-8", output_encoding="UTF-8")
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"a\xef\xbf\xbd")   # 0xFFFD

  def test_transcode_utf8_with_replacement_char(self):
    # [FUNCTION DOC]
    # Input already containing the Unicode replacement character, transcoded in strict mode and
    # in replace mode with a different replacement. Asserts the character is passed through
    # unchanged both times since it is valid input.
    strings = [b"a\xef\xbf\xbd"]
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings, input_encoding="UTF-8", output_encoding="UTF-8",
          errors="strict")
      values = self.evaluate(outputs)
      self.assertAllEqual(values, [b"a\xef\xbf\xbd"])

      outputs = string_ops.unicode_transcode(
          strings, input_encoding="UTF-8", output_encoding="UTF-8",
          errors="replace", replacement_char=ord("?"))
      values = self.evaluate(outputs)
      self.assertAllEqual(values, [b"a\xef\xbf\xbd"])

  def test_transcode_utf8_to_utf16(self):
    # [FUNCTION DOC]
    # Transcodes UTF-8 strings (including one outside the basic plane) to big-endian UTF-16.
    # Asserts the output equals the Python-computed encoding.
    strings = [b"ab\xe2\x82\xac", b"\xf0\x90\x90\xb7"]  # U+10437
    expected = [s.decode("UTF-8").encode("UTF-16-BE") for s in strings]

    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-8",
          output_encoding="UTF-16-BE",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, expected)

  def test_transcode_utf32_to_utf8(self):
    # [FUNCTION DOC]
    # Transcodes UTF-32 data to UTF-8 using the plain UTF-32 name. Asserts the output equals the
    # Python-computed encoding.
    strings = [
        b"\x00\x00\x00a\x00\x00\x00b\x00\x00\x20\xAC", b"\x00\x01\x04\x37"
    ]  # U+10437
    expected = [s.decode("UTF-32-BE").encode("UTF-8") for s in strings]
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-32",
          output_encoding="UTF-8",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, expected)

  def test_transcode_utf8_to_utf32(self):
    # [FUNCTION DOC]
    # Transcodes UTF-8 strings to big-endian UTF-32. Asserts the output equals the
    # Python-computed encoding.
    strings = [b"ab\xe2\x82\xac", b"\xf0\x90\x90\xb7"]
    expected = [s.decode("UTF-8").encode("UTF-32-BE") for s in strings]
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-8",
          output_encoding="UTF-32-BE",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, expected)

  # Documentation in ICU suggests that getNextUChar may produce a different
  # error code if the input sequence contains particular non-coding sequences.
  # This test checks that condition.
  def test_transcode_ascii_with_shift_chars(self):
    # [FUNCTION DOC]
    # Transcodes ASCII strings consisting of shift-out and shift-in control bytes to UTF-8.
    # Asserts they pass through unchanged, guarding against error-code differences in the
    # underlying ICU reader.
    strings = [b"\x0e\x0e", b"\x0f\x0f"]
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="US-ASCII",
          output_encoding="UTF-8",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = self.evaluate(outputs)
      self.assertAllEqual(values, strings)

  def test_transcode_utf8_with_bom(self):
    # [FUNCTION DOC]
    # A UTF-8 string starting with a byte order mark transcoded to UTF-8 and to big-endian
    # UTF-16. Asserts the mark is preserved in the UTF-8 output and encoded in the UTF-16 one.
    bom_string = b"\xef\xbb\xbfabcdefg"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-8", output_encoding="UTF-8")
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"\xef\xbb\xbfabcdefg")  # BOM preserved

      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-8", output_encoding="UTF-16-BE")
      values = self.evaluate(outputs)
      utf16expected = bom_string.decode("UTF-8").encode("UTF-16-BE")
      self.assertAllEqual(values, utf16expected)

  def test_transcode_utf16_le_be_with_bom(self):
    # [FUNCTION DOC]
    # Byte-order-marked UTF-16 data decoded with explicit big-endian and little-endian names.
    # Asserts the mark is preserved when the order matches and the text is mangled when the
    # declared order is wrong.
    bom_string = b"\xfe\xff\x00\x61"  # Big-endian BOM with 'a' encoded
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-16-BE", output_encoding="UTF-8")
      values = self.evaluate(outputs)
      # BOM is preserved in output
      self.assertAllEqual(values, b"\xef\xbb\xbfa")

      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-16-LE", output_encoding="UTF-8")
      values = self.evaluate(outputs)
      # mangled BOM and value from (incorrect) LE encoding
      self.assertAllEqual(values, b"\xef\xbf\xbe\xe6\x84\x80")

      bom_string = b"\xff\xfe\x61\x00"  # Little-endian BOM with 'a' encoded
      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-16-LE", output_encoding="UTF-8")
      values = self.evaluate(outputs)
      self.assertAllEqual(values, b"\xef\xbb\xbfa")

  @parameterized.parameters(
      # BOM is stripped if it is used to decide the byte order of the input.
      (b"\xfe\xff\x00*", "UTF-16", b"*"),
      (b"\xff\xfe*\x00", "UTF-16", b"*"),
      # BOM is *not* stripped if it is not used to decide the byte order of
      # the input.
      (b"\xef\xbb\xbf*", "UTF-8", b"\xef\xbb\xbf*"),
      (b"\xfe\xff\x00*", "UTF-16-BE", b"\xef\xbb\xbf*"),
      (b"\xff\xfe*\x00", "UTF-16-LE", b"\xef\xbb\xbf*"),
      # If the encoding is UTF-16, and no BOM is present, then UTF-16-BE
      # is assumed.
      (b"\x00*", "UTF-16", b"*"),
      # BOM is never stripped from any position other than the beginning of
      # the string, for any encoding.
      (b"<\xef\xbb\xbf>", "UTF-8", b"<\xef\xbb\xbf>"),
      (b"\x00<\xfe\xff\x00>", "UTF-16", b"<\xef\xbb\xbf>"),
      (b"\x00<\xfe\xff\x00>", "UTF-16-BE", b"<\xef\xbb\xbf>"),
      (b"<\x00\xff\xfe>\x00", "UTF-16-LE", b"<\xef\xbb\xbf>"),
      (b"\xfe\xff\x00<\xfe\xff\x00>", "UTF-16", b"<\xef\xbb\xbf>"),
      (b"\xff\xfe<\x00\xff\xfe>\x00", "UTF-16", b"<\xef\xbb\xbf>"),
  )
  @test_util.run_deprecated_v1
  def test_bom_handling(self, string, input_encoding, expected):
    # [FUNCTION DOC]
    # Parameterized by input bytes, input encoding name and expected UTF-8 output. Verifies the
    # mark is stripped only when it decides the byte order of plain UTF-16, defaults to big
    # endian without a mark, and is never stripped elsewhere. Graph mode version one.
    with self.test_session():
      output = string_ops.unicode_transcode(
          string, input_encoding=input_encoding, output_encoding="UTF-8")
      self.assertAllEqual(output, expected)

  @test_util.run_deprecated_v1
  def test_invalid_encoding_causes_errors(self):
    # [FUNCTION DOC]
    # An unknown input encoding name must produce an op error at run time about failing to
    # create the converter; an unknown output encoding name must raise a value error at graph
    # construction because it is checked against allowed values. Graph mode version one.
    strings = [[b"a", b"abc"], [b"ABC", b"DEF"]]

    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="invalid",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      with self.assertRaisesOpError(
          "Could not create converter for input encoding: invalid"):
        self.evaluate(outputs)

    with self.assertRaisesRegex(ValueError, "Op passed string 'invalid'"):
      with self.cached_session() as sess:
        outputs = string_ops.unicode_transcode(
            strings,
            input_encoding="UTF-8",
            output_encoding="invalid",
            errors="replace",
            replacement_char=ord(" "),
            replace_control_characters=False)
        self.evaluate(outputs)

  @test_util.run_deprecated_v1
  def test_invalid_error_policy_causes_errors(self):
    # [FUNCTION DOC]
    # Passes an unknown error policy. Asserts a value error listing the allowed policies
    # (strict, replace, ignore). Graph mode version one.
    strings = [[b"a", b"abc"], [b"ABC", b"DEF"]]

    with self.assertRaisesRegex(
        ValueError, "'invalid' not in: \"strict\", \"replace\", \"ignore\"."):
      with self.cached_session() as sess:
        outputs = string_ops.unicode_transcode(
            strings,
            input_encoding="UTF-8",
            output_encoding="UTF-8",
            errors="invalid",
            replacement_char=ord(" "),
            replace_control_characters=False)
        self.evaluate(outputs)

  def test_forwarding(self):
    # [FUNCTION DOC]
    # Feeds the transcode op with the output of a substring op that only it consumes, which
    # exercises the optimized path that reuses the input buffer. Asserts the expected
    # five-byte prefixes.
    with self.cached_session():
      # Generate an input that is uniquely consumed by the transcode op.
      # This exercises code paths which are optimized for this case
      # (e.g., using forwarding).
      inp = string_ops.substr(
          constant_op.constant([b"AbCdEfG", b"HiJkLmN"], dtypes.string),
          pos=0,
          len=5)
      transcoded = string_ops.unicode_transcode(
          inp, input_encoding="UTF-8", output_encoding="UTF-8")

      self.assertAllEqual([b"AbCdE", b"HiJkL"], transcoded)

  @test_util.run_deprecated_v1
  def test_cjk_encodings(self):
    # [FUNCTION DOC]
    # Transcodes Japanese (Shift JIS), simplified Chinese (GB18030), traditional Chinese (Big5)
    # and Korean (EUC-KR) samples to UTF-8 in one session run. Asserts each equals the
    # Python-decoded and re-encoded expectation.
    strings_ja = [
        b"\x5c\x5c",  # Yen sign
        b"\x8f\x70",  # kanji character "waza"
        b"\x83\x4f"
    ]  # katakana character "gu"
    strings_zh_cn = [b"\xca\xf5"]  # simplified "shu4"
    strings_zh_tw = [b"\xb3\x4e"]  # traditional "shu4"
    strings_ko = [b"\xc7\xd1\xb9\xce"]  # hangul "hanmin"

    expected_ja = [s.decode("shift_jis").encode("UTF-8") for s in strings_ja]
    expected_zh_cn = [
        s.decode("gb18030").encode("UTF-8") for s in strings_zh_cn
    ]
    expected_zh_tw = [s.decode("big5").encode("UTF-8") for s in strings_zh_tw]
    expected_ko = [s.decode("euc_kr").encode("UTF-8") for s in strings_ko]

    with self.cached_session() as sess:
      outputs_ja = string_ops.unicode_transcode(
          strings_ja,
          input_encoding="shift_jis",
          output_encoding="UTF-8",
          replacement_char=ord(" "),
          replace_control_characters=False)

      outputs_zh_cn = string_ops.unicode_transcode(
          strings_zh_cn,
          input_encoding="gb18030",
          output_encoding="UTF-8",
          replacement_char=ord(" "),
          replace_control_characters=False)

      outputs_zh_tw = string_ops.unicode_transcode(
          strings_zh_tw,
          input_encoding="big5",
          output_encoding="UTF-8",
          replacement_char=ord(" "),
          replace_control_characters=False)

      outputs_ko = string_ops.unicode_transcode(
          strings_ko,
          input_encoding="euc_kr",
          output_encoding="UTF-8",
          replacement_char=ord(" "),
          replace_control_characters=False)

      result_ja, result_zh_cn, result_zh_tw, result_ko = sess.run(
          [outputs_ja, outputs_zh_cn, outputs_zh_tw, outputs_ko])

      self.assertAllEqual(result_ja, expected_ja)
      self.assertAllEqual(result_zh_cn, expected_zh_cn)
      self.assertAllEqual(result_zh_tw, expected_zh_tw)
      self.assertAllEqual(result_ko, expected_ko)


if __name__ == "__main__":
  test.main()
