# [MODULE DOC]
# Tests for ragged_string_ops.unicode_encode, which turns integer codepoint tensors (dense or
# ragged, rank one to four) into encoded byte strings in UTF-8, UTF-16-BE and UTF-32-BE.
# Covers strict, ignore and replace error handling, custom replacement characters, required
# parameters, scalar rejection and unknown-rank rejection. Test-only module with no import-time
# side effects.
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
"""Tests for UnicodeEncode op from ragged_string_ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl as errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import test


class UnicodeEncodeOpTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the unicode encode op, run across three output encodings.
  # Overrides the equality assertion so that tensors are first evaluated in a session and
  # converted to plain nested lists before comparison. Many tests are restricted to v1 only.

  def assertAllEqual(self, rt, expected):
    # [FUNCTION DOC]
    # HELPER: custom comparison used by all tests in this class.
    # KNOWN ISSUE: overrides the base-class assertion with a different meaning; it evaluates the
    # first argument in a session (so it only works for graph tensors or ragged tensors), turns
    # numpy arrays or ragged values into nested lists, and uses plain equality, not the numeric
    # tolerance semantics of the base method.
    # Inputs: rt is the tensor or ragged tensor to evaluate, expected the nested list or bytes
    # value. Returns nothing; fails the test when values differ.
    with self.cached_session() as sess:
      value = sess.run(rt)
      if isinstance(value, np.ndarray):
        value = value.tolist()
      elif isinstance(value, ragged_tensor_value.RaggedTensorValue):
        value = value.to_list()
      self.assertEqual(value, expected)

  def testScalar(self):
    # [FUNCTION DOC]
    # Passes a plain integer and then a scalar tensor to the encode op. Asserts a value error is
    # raised both times since input must have at least rank one.
    with self.cached_session():
      with self.assertRaises(ValueError):
        ragged_string_ops.unicode_encode(72, "UTF-8")
    with self.cached_session():
      with self.assertRaises(ValueError):
        ragged_string_ops.unicode_encode(constant_op.constant(72), "UTF-8")

  def testRequireParams(self):
    # [FUNCTION DOC]
    # Calls the encode op with no arguments, with only the input, and with only the encoding.
    # Asserts a type error is raised each time because both are required.
    with self.cached_session():
      with self.assertRaises(TypeError):
        ragged_string_ops.unicode_encode()  # pylint: disable=no-value-for-parameter
    with self.cached_session():
      with self.assertRaises(TypeError):
        ragged_string_ops.unicode_encode(72)  # pylint: disable=no-value-for-parameter
    with self.cached_session():
      with self.assertRaises(TypeError):
        ragged_string_ops.unicode_encode(encoding="UTF-8")  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsValid(self, encoding):
    # [FUNCTION DOC]
    # Encodes a vector including control characters, an emoji, noncharacters and the largest
    # valid codepoint in strict mode. Asserts no error and the exact expected bytes, since
    # noncharacters are valid.
    # Noncharacters are not errors.
    # See https://www.unicode.org/versions/corrigendum9.html.
    test_value = np.array([ord("H"), ord("e"), ord("o"), 0, ord("\b"), 0x1F600,
                           0xFDD0,  # noncharacter
                           0xFFFE, 0xFFFF,  # last two in BMP
                           0x10FFFF],  # largest valid
                          np.int32)
    expected_value = (
        u"Heo\0\b\U0001f600\ufdd0\ufffe\uffff\U0010ffff"
    ).encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "strict")
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsNegative(self, encoding):
    # [FUNCTION DOC]
    # Encodes a negative codepoint in strict mode. Asserts running it raises an invalid-argument
    # error.
    test_value = np.array([-1], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsTooLarge(self, encoding):
    # [FUNCTION DOC]
    # Encodes a codepoint just beyond the Unicode range in strict mode. Asserts an
    # invalid-argument error.
    test_value = np.array([0x110000], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  def testStrictErrorsSurrogatePair(self, encoding):
    # [FUNCTION DOC]
    # Encodes a high and low surrogate pair as separate codepoints in strict mode. Asserts an
    # invalid-argument error because surrogates are invalid codepoints.
    test_value = np.array([0xD83D, 0xDE00], np.int32)
    with self.cached_session() as session:
      with self.assertRaises(errors.InvalidArgumentError):
        session.run(
            ragged_string_ops.unicode_encode(test_value, encoding, "strict"))

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testIgnoreErrors(self, encoding):
    # [FUNCTION DOC]
    # Encodes a vector mixing valid codepoints, noncharacters and invalid values (too large,
    # negative, surrogates) in ignore mode. Asserts the invalid codepoints are dropped and the
    # rest encoded. Only runs under graph mode version one.
    # Noncharacters are not errors.
    # See https://www.unicode.org/versions/corrigendum9.html.
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o'),
                           0, ord('\b'), 0x1F600,  # valid
                           0xD83D, 0xDE00,  # invalid, surrogate code points
                           0xFDD0,  # noncharacter
                           0xFFFE, 0xFFFF,  # last two in BMP
                           0x10FFFF,  # last in plane 16 = last in Unicode
                           0x110000],  # invalid, beyond Unicode
                          np.int32)
    expected_value = (
        u"Heo\0\b\U0001F600\ufdd0\ufffe\uffff\U0010ffff"
    ).encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "ignore")
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testIgnoreErrorsSurrogate(self, encoding):
    # [FUNCTION DOC]
    # Encodes only a surrogate pair of codepoints in ignore mode. Asserts the output is the
    # empty byte string. Graph mode version one only.
    test_value = np.array([0xD83D, 0xDE00], np.int32)
    expected_value = b""
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "ignore")
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testReplaceErrors(self, encoding):
    # [FUNCTION DOC]
    # Replace-mode scenarios: invalid codepoints become the default replacement character; a
    # custom replacement character is used when given; replace is the default mode; an unpaired
    # surrogate is replaced with a custom character; a noncharacter is left alone; and a
    # replacement character outside the valid range raises an invalid-argument error on
    # evaluation. Graph mode version one only.
    # Noncharacters are not errors.
    # See https://www.unicode.org/versions/corrigendum9.html.
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o'),
                           0, ord('\b'), 0x1F600,  # valid
                           0xD83D, 0xDE00,  # invalid, surrogate code points
                           0xFDD0,  # noncharacter
                           0xFFFE, 0xFFFF,  # last two in BMP
                           0x10FFFF,  # last in plane 16 = last in Unicode
                           0x110000],  # invalid, beyond Unicode
                          np.int32)
    expected_value = (u"He\ufffd\ufffdo\0\b\U0001F600\ufffd\ufffd"
                      u"\ufdd0\ufffe\uffff\U0010ffff\ufffd").encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace")
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Test custom replacement character
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o')],
                          np.int32)
    expected_value = u"Heooo".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", ord('o'))
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Verify "replace" is default
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o')],
                          np.int32)
    expected_value = u"He\U0000fffd\U0000fffdo".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Verify non-default replacement with an unpaired surrogate.
    test_value = np.array([0xD801], np.int32)
    expected_value = "\u0041".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", 0x41)
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Test with a noncharacter code point.  These are not errors.
    # See https://www.unicode.org/versions/corrigendum9.html.
    test_value = np.array([0x1FFFF], np.int32)
    expected_value = u"\U0001ffff".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", 0x41)
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Replacement_char must be within range
    test_value = np.array([ord('H'), ord('e'), 0x7FFFFFFF, -1, ord('o')],
                          np.int32)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding,
                                                         "replace", 0x110000)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(unicode_encode_op)

  # -- regular Tensor tests -- #

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testVector(self, encoding):
    # [FUNCTION DOC]
    # Encodes rank-one codepoint arrays (ASCII, mixed with multi-byte and emoji, and
    # single-character cases). Asserts a single encoded byte string for each. Graph mode
    # version one only.
    test_value = np.array([ord('H'), ord('e'), ord('l'), ord('l'), ord('o')],
                          np.int32)
    expected_value = u"Hello".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

    test_value = np.array([ord('H'), ord('e'), 0xC3, 0xC3, 0x1F604], np.int32)
    expected_value = u"He\xc3\xc3\U0001f604".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

    # Single character string
    test_value = np.array([ord('H')], np.int32)
    expected_value = u"H".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

    test_value = np.array([0x1F604], np.int32)
    expected_value = u"\U0001f604".encode(encoding)
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testMatrix(self, encoding):
    # [FUNCTION DOC]
    # Encodes a two-row codepoint matrix. Asserts one encoded byte string per row. Graph mode
    # version one only.
    test_value = np.array(
        [[72, 0x1F604, 108, 108, 111], [87, 0x1F604, 114, 108, 100]], np.int32)
    expected_value = [
        u"H\U0001f604llo".encode(encoding), u"W\U0001f604rld".encode(encoding)
    ]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test3DimMatrix(self, encoding):
    # [FUNCTION DOC]
    # Encodes a dense rank-three codepoint tensor. Asserts a 3x2 nested list of encoded strings.
    # Graph mode version one only.
    test_value = constant_op.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]],
         [[102, 105, 120, 101, 100], [119, 111, 114, 100, 115]],
         [[72, 121, 112, 101, 114], [99, 117, 98, 101, 46]]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World".encode(encoding)],
                      [u"fixed".encode(encoding), u"words".encode(encoding)],
                      [u"Hyper".encode(encoding), u"cube.".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test4DimMatrix(self, encoding):
    # [FUNCTION DOC]
    # Encodes a dense rank-four codepoint tensor. Asserts a nested list of encoded strings with
    # the innermost dimension removed. Graph mode version one only.
    test_value = constant_op.constant(
        [[[[72, 101, 108, 108, 111]], [[87, 111, 114, 108, 100]]],
         [[[102, 105, 120, 101, 100]], [[119, 111, 114, 100, 115]]],
         [[[72, 121, 112, 101, 114]], [[99, 117, 98, 101, 46]]]], np.int32)
    expected_value = [[[u"Hello".encode(encoding)],
                       [u"World".encode(encoding)]],
                      [[u"fixed".encode(encoding)],
                       [u"words".encode(encoding)]],
                      [[u"Hyper".encode(encoding)],
                       [u"cube.".encode(encoding)]]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  # -- Ragged Tensor tests -- #

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testRaggedMatrix(self, encoding):
    # [FUNCTION DOC]
    # Encodes a ragged codepoint matrix with rows of different length. Asserts one encoded
    # string per row. Graph mode version one only.
    test_value = ragged_factory_ops.constant(
        [[ord('H'), 0xC3, ord('l'), ord('l'), ord('o')],
         [ord('W'), 0x1F604, ord('r'), ord('l'), ord('d'), ord('.')]], np.int32)
    expected_value = [
        u"H\xc3llo".encode(encoding), u"W\U0001f604rld.".encode(encoding)
    ]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test3DimMatrixWithRagged2ndDim(self, encoding):
    # [FUNCTION DOC]
    # Encodes a rank-three ragged tensor whose second dimension varies in length. Asserts the
    # nested list of encoded strings keeps the ragged structure. Graph mode version one only.
    test_value = ragged_factory_ops.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]],
         [[102, 105, 120, 101, 100]],
         [[72, 121, 112, 101, 114], [119, 111, 114, 100, 115],
          [99, 117, 98, 101, 46]]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World".encode(encoding)],
                      [u"fixed".encode(encoding)],
                      [
                          u"Hyper".encode(encoding), u"words".encode(encoding),
                          u"cube.".encode(encoding)
                      ]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test3DimMatrixWithRagged3rdDim(self, encoding):
    # [FUNCTION DOC]
    # Encodes a rank-three ragged tensor whose innermost lists differ in length, including an
    # empty one. Asserts the empty list yields the empty string. Graph mode version one only.
    test_value = ragged_factory_ops.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100, 46]],
         [[68, 111, 110, 39, 116], [119, 195, 114, 114, 121, 44, 32, 98, 101]],
         [[0x1F604], []]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World.".encode(encoding)],
                      [
                          u"Don't".encode(encoding),
                          u"w\xc3rry, be".encode(encoding)
                      ], [u"\U0001f604".encode(encoding), u"".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test3DimMatrixWithRagged2ndAnd3rdDim(self, encoding):
    # [FUNCTION DOC]
    # Encodes a rank-three tensor ragged in both inner dimensions, including an empty middle
    # row. Asserts the empty row stays an empty list in the output. Graph mode version one only.
    test_value = ragged_factory_ops.constant(
        [[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100, 46]], [],
         [[0x1F604]]], np.int32)
    expected_value = [[u"Hello".encode(encoding), u"World.".encode(encoding)],
                      [], [u"\U0001f604".encode(encoding)]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def test4DimRaggedMatrix(self, encoding):
    # [FUNCTION DOC]
    # Encodes a rank-four ragged tensor including an empty innermost list. Asserts the nested
    # encoded strings, with the empty string for the empty list. Graph mode version one only.
    test_value = ragged_factory_ops.constant(
        [[[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]]],
         [[[]], [[72, 121, 112, 101]]]], np.int32)
    expected_value = [[[u"Hello".encode(encoding), u"World".encode(encoding)]],
                      [[u"".encode(encoding)], [u"Hype".encode(encoding)]]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  @parameterized.parameters("UTF-8", "UTF-16-BE", "UTF-32-BE")
  @test_util.run_v1_only("b/120545219")
  def testRaggedMatrixWithMultiDimensionInnerValues(self, encoding):
    # [FUNCTION DOC]
    # Builds a ragged tensor from nested row splits whose flat values are themselves
    # multi-dimensional and encodes it. Asserts the nested encoded strings follow the splits.
    # Graph mode version one only.
    test_flat_values = constant_op.constant([[[72, 101, 108, 108, 111],
                                              [87, 111, 114, 108, 100]],
                                             [[102, 105, 120, 101, 100],
                                              [119, 111, 114, 100, 115]],
                                             [[72, 121, 112, 101, 114],
                                              [99, 117, 98, 101, 46]]])
    test_row_splits = [
        constant_op.constant([0, 2, 3], dtype=np.int64),
        constant_op.constant([0, 1, 1, 3], dtype=np.int64)
    ]
    test_value = ragged_tensor.RaggedTensor.from_nested_row_splits(
        test_flat_values, test_row_splits)
    expected_value = [[[[u"Hello".encode(encoding), u"World".encode(encoding)]],
                       []],
                      [[[u"fixed".encode(encoding), u"words".encode(encoding)],
                        [u"Hyper".encode(encoding),
                         u"cube.".encode(encoding)]]]]
    unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
    self.assertAllEqual(unicode_encode_op, expected_value)

  def testUnknownInputRankError(self):
    # [FUNCTION DOC]
    # Wraps the encode op in a compiled function whose input signature has unknown rank and
    # calls it with a list of codepoints. Asserts a value error stating that the rank must be
    # statically known.
    # Use a tf.function that erases shape information.
    @def_function.function(input_signature=[tensor_spec.TensorSpec(None)])
    def f(v):
      # [FUNCTION DOC]
      # HELPER: inner function for the unknown-rank test.
      # Takes a codepoint tensor of unknown rank and returns it encoded as UTF-8 strings; raises
      # a value error at trace time because the rank is unknown.
      return ragged_string_ops.unicode_encode(v, "UTF-8")

    with self.assertRaisesRegex(
        ValueError, "Rank of input_tensor must be statically known."):
      f([72, 101, 108, 108, 111])


if __name__ == "__main__":
  test.main()
