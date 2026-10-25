# [MODULE DOC]
# Kernel tests for the string substring operation (string_ops.substr). Covers scalar, vector,
# matrix and broadcast inputs, byte and UTF-8 character units, positive and negative positions,
# int32 and int64 index types, and the error paths (out of range, bad shapes, wrong dtypes,
# invalid unit, unsupported rank). Test-only module with no import-time side effects.
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
"""Tests for Substr op from string_ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class SubstrOpTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case covering string_ops.substr across index dtypes and the byte and
  # UTF-8 character units. Each test builds string tensors, evaluates the op in a cached
  # session and compares against hand-computed expected substrings or expected errors.

  @parameterized.parameters(
      (np.int32, 1, "BYTE"),
      (np.int64, 1, "BYTE"),
      (np.int32, -4, "BYTE"),
      (np.int64, -4, "BYTE"),
      (np.int32, 1, "UTF8_CHAR"),
      (np.int64, 1, "UTF8_CHAR"),
      (np.int32, -4, "UTF8_CHAR"),
      (np.int64, -4, "UTF8_CHAR"),
  )
  def testScalarString(self, dtype, pos, unit):
    # [FUNCTION DOC]
    # Takes a scalar string (ASCII for byte unit, multi-byte text for character unit) and
    # extracts three units at a positive or negative position given as a scalar of the
    # parameterized integer dtype. Asserts the result equals the expected substring.
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"He\xc3\xc3\U0001f604".encode("utf-8"),
    }[unit]
    expected_value = {
        "BYTE": b"ell",
        "UTF8_CHAR": u"e\xc3\xc3".encode("utf-8"),
    }[unit]
    position = np.array(pos, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  def testScalarString_EdgeCases(self, dtype, unit):
    # [FUNCTION DOC]
    # Edge cases on scalar strings: an empty string yields empty output; taking the whole
    # string with position zero or minus five returns it unchanged; a length larger than
    # what remains after a negative position is truncated to the available suffix.
    # Empty string
    test_string = {
        "BYTE": b"",
        "UTF8_CHAR": u"".encode("utf-8"),
    }[unit]
    expected_value = b""
    position = np.array(0, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

    # Full string
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"H\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    position = np.array(0, dtype)
    length = np.array(5, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, test_string)

    # Full string (Negative)
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"H\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    position = np.array(-5, dtype)
    length = np.array(5, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, test_string)

    # Length is larger in magnitude than a negative position
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"H\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    expected_string = {
        "BYTE": b"ello",
        "UTF8_CHAR": u"\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    position = np.array(-4, dtype)
    length = np.array(5, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_string)

  @parameterized.parameters(
      (np.int32, 1, "BYTE"),
      (np.int64, 1, "BYTE"),
      (np.int32, -4, "BYTE"),
      (np.int64, -4, "BYTE"),
      (np.int32, 1, "UTF8_CHAR"),
      (np.int64, 1, "UTF8_CHAR"),
      (np.int32, -4, "UTF8_CHAR"),
      (np.int64, -4, "UTF8_CHAR"),
  )
  def testVectorStrings(self, dtype, pos, unit):
    # [FUNCTION DOC]
    # Applies a scalar position and length to a vector of two strings, for byte and
    # character units. Asserts each element is cut to the expected three-unit substring.
    test_string = {
        "BYTE": [b"Hello", b"World"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"H\xc3llo",
                                                  u"W\U0001f604rld"]],
    }[unit]
    expected_value = {
        "BYTE": [b"ell", b"orl"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"\xc3ll", u"\U0001f604rl"]],
    }[unit]
    position = np.array(pos, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  def testMatrixStrings(self, dtype, unit):
    # [FUNCTION DOC]
    # Applies scalar position and length to a matrix of strings, first with a positive
    # position and then with a negative one. Asserts the elementwise expected substrings
    # for both byte and character units.
    test_string = {
        "BYTE": [[b"ten", b"eleven", b"twelve"],
                 [b"thirteen", b"fourteen", b"fifteen"],
                 [b"sixteen", b"seventeen", b"eighteen"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227n",
                                                   u"\xc6\u053c\u025bv\u025bn",
                                                   u"tw\u0c1dlv\u025b"]],
                      [x.encode("utf-8") for x in [u"He\xc3\xc3o",
                                                   u"W\U0001f604rld",
                                                   u"d\xfcd\xea"]]],
    }[unit]
    position = np.array(1, dtype)
    length = np.array(4, dtype)
    expected_value = {
        "BYTE": [[b"en", b"leve", b"welv"], [b"hirt", b"ourt", b"ifte"],
                 [b"ixte", b"even", b"ight"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d227n",
                                                   u"\u053c\u025bv\u025b",
                                                   u"w\u0c1dlv"]],
                      [x.encode("utf-8") for x in [u"e\xc3\xc3o",
                                                   u"\U0001f604rld",
                                                   u"\xfcd\xea"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

    position = np.array(-3, dtype)
    length = np.array(2, dtype)
    expected_value = {
        "BYTE": [[b"te", b"ve", b"lv"], [b"ee", b"ee", b"ee"],
                 [b"ee", b"ee", b"ee"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227",
                                                   u"v\u025b", u"lv"]],
                      [x.encode("utf-8") for x in [u"\xc3\xc3", u"rl",
                                                   u"\xfcd"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  def testElementWisePosLen(self, dtype, unit):
    # [FUNCTION DOC]
    # Uses a 3x3 matrix of strings with position and length matrices of the same shape,
    # mixing positive and negative positions. Asserts each element is cut with its own
    # position and length.
    test_string = {
        "BYTE": [[b"ten", b"eleven", b"twelve"],
                 [b"thirteen", b"fourteen", b"fifteen"],
                 [b"sixteen", b"seventeen", b"eighteen"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227n",
                                                   u"\xc6\u053c\u025bv\u025bn",
                                                   u"tw\u0c1dlv\u025b"]],
                      [x.encode("utf-8") for x in [u"He\xc3\xc3o",
                                                   u"W\U0001f604rld",
                                                   u"d\xfcd\xea"]],
                      [x.encode("utf-8") for x in [u"sixt\xea\xean",
                                                   u"se\U00010299enteen",
                                                   u"ei\U0001e920h\x86een"]]],
    }[unit]
    position = np.array([[1, -4, 3], [1, 2, -4], [-5, 2, 3]], dtype)
    length = np.array([[2, 2, 4], [4, 3, 2], [5, 5, 5]], dtype)
    expected_value = {
        "BYTE": [[b"en", b"ev", b"lve"], [b"hirt", b"urt", b"te"],
                 [b"xteen", b"vente", b"hteen"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d227n",
                                                   u"\u025bv",
                                                   u"lv\u025b"]],
                      [x.encode("utf-8") for x in [u"e\xc3\xc3o",
                                                   u"rld",
                                                   u"d\xfc"]],
                      [x.encode("utf-8") for x in [u"xt\xea\xean",
                                                   u"\U00010299ente",
                                                   u"h\x86een"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  def testBroadcast(self, dtype, unit):
    # [FUNCTION DOC]
    # Checks broadcasting in three ways: vector position/length onto a matrix of strings,
    # a vector of strings onto matrix position/length, and a scalar string against vector
    # position/length. Asserts the broadcast elementwise substrings for each case.
    # Broadcast pos/len onto input string
    test_string = {
        "BYTE": [[b"ten", b"eleven", b"twelve"],
                 [b"thirteen", b"fourteen", b"fifteen"],
                 [b"sixteen", b"seventeen", b"eighteen"],
                 [b"nineteen", b"twenty", b"twentyone"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227n",
                                                   u"\xc6\u053c\u025bv\u025bn",
                                                   u"tw\u0c1dlv\u025b"]],
                      [x.encode("utf-8") for x in [u"th\xcdrt\xea\xean",
                                                   u"f\U0001f604urt\xea\xean",
                                                   u"f\xcd\ua09ctee\ua0e4"]],
                      [x.encode("utf-8") for x in [u"s\xcdxt\xea\xean",
                                                   u"se\U00010299enteen",
                                                   u"ei\U0001e920h\x86een"]],
                      [x.encode("utf-8") for x in [u"nineteen",
                                                   u"twenty",
                                                   u"twentyone"]]],
    }[unit]
    position = np.array([1, -4, 3], dtype)
    length = np.array([1, 2, 3], dtype)
    expected_value = {
        "BYTE": [[b"e", b"ev", b"lve"], [b"h", b"te", b"tee"],
                 [b"i", b"te", b"hte"], [b"i", b"en", b"nty"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d227",
                                                   u"\u025bv", u"lv\u025b"]],
                      [x.encode("utf-8") for x in [u"h", u"t\xea", u"tee"]],
                      [x.encode("utf-8") for x in [u"\xcd", u"te", u"h\x86e"]],
                      [x.encode("utf-8") for x in [u"i", u"en", u"nty"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

    # Broadcast input string onto pos/len
    test_string = {
        "BYTE": [b"thirteen", b"fourteen", b"fifteen"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"th\xcdrt\xea\xean",
                                                  u"f\U0001f604urt\xea\xean",
                                                  u"f\xcd\ua09ctee\ua0e4"]],
    }[unit]
    position = np.array([[1, -2, 3], [-3, 2, 1], [5, 5, -5]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    expected_value = {
        "BYTE": [[b"hir", b"en", b"t"], [b"e", b"ur", b"ift"],
                 [b"ee", b"ee", b"ft"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"h\xcdr", u"\xean", u"t"]],
                      [x.encode("utf-8") for x in [u"\xea", u"ur",
                                                   u"\xcd\ua09ct"]],
                      [x.encode("utf-8") for x in [u"\xea\xea", u"\xea\xea",
                                                   u"\ua09ct"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

    # Test 1D broadcast
    test_string = {
        "BYTE": b"thirteen",
        "UTF8_CHAR": u"th\xcdrt\xea\xean".encode("utf-8"),
    }[unit]
    position = np.array([1, -4, 7], dtype)
    length = np.array([3, 2, 1], dtype)
    expected_value = {
        "BYTE": [b"hir", b"te", b"n"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"h\xcdr", u"t\xea", u"n"]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testBadBroadcast(self, dtype, unit):
    # [FUNCTION DOC]
    # Passes position and length vectors of length four against a 3x3 string matrix, which
    # cannot broadcast. Asserts a value error is raised at graph construction time.
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array([1, 2, -3, 4], dtype)
    length = np.array([1, 2, 3, 4], dtype)
    with self.assertRaises(ValueError):
      string_ops.substr(test_string, position, length, unit=unit)

  @parameterized.parameters(
      (np.int32, 6, "BYTE"),
      (np.int64, 6, "BYTE"),
      (np.int32, -6, "BYTE"),
      (np.int64, -6, "BYTE"),
      (np.int32, 6, "UTF8_CHAR"),
      (np.int64, 6, "UTF8_CHAR"),
      (np.int32, -6, "UTF8_CHAR"),
      (np.int64, -6, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testOutOfRangeError_Scalar(self, dtype, pos, unit):
    # [FUNCTION DOC]
    # A scalar string of five units with position six or minus six is out of range.
    # Asserts evaluating the op raises an invalid-argument error.
    # Scalar/Scalar
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"H\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    position = np.array(pos, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

  @parameterized.parameters(
      (np.int32, 4, "BYTE"),
      (np.int64, 4, "BYTE"),
      (np.int32, -4, "BYTE"),
      (np.int64, -4, "BYTE"),
      (np.int32, 4, "UTF8_CHAR"),
      (np.int64, 4, "UTF8_CHAR"),
      (np.int32, -4, "UTF8_CHAR"),
      (np.int64, -4, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testOutOfRangeError_VectorScalar(self, dtype, pos, unit):
    # [FUNCTION DOC]
    # A vector of strings in which one is shorter than the scalar position (positive or
    # negative magnitude four). Asserts evaluation raises an invalid-argument error.
    # Vector/Scalar
    test_string = {
        "BYTE": [b"good", b"good", b"bad", b"good"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"g\xc3\xc3d", u"b\xc3d",
                                                  u"g\xc3\xc3d"]],
    }[unit]
    position = np.array(pos, dtype)
    length = np.array(1, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testOutOfRangeError_MatrixMatrix(self, dtype, unit):
    # [FUNCTION DOC]
    # A matrix of strings with a matrix of positions where one position exceeds its string,
    # first with positive then with negative positions. Asserts an invalid-argument error
    # is raised when evaluating in each case.
    # Matrix/Matrix
    test_string = {
        "BYTE": [[b"good", b"good", b"good"], [b"good", b"good", b"bad"],
                 [b"good", b"good", b"good"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"g\xc3\xc3d"]],
                      [x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"b\xc3d"]],
                      [x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"g\xc3\xc3d"]]],
    }[unit]
    position = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 3]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

    # Matrix/Matrix (with negative)
    position = np.array([[1, 2, -3], [1, 2, -4], [1, 2, -3]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testOutOfRangeError_Broadcast(self, dtype, unit):
    # [FUNCTION DOC]
    # Broadcasts a vector of positions and lengths over a matrix of strings where one string
    # is too short, first with positive then negative positions. Asserts an invalid-argument
    # error is raised on evaluation.
    # Broadcast
    test_string = {
        "BYTE": [[b"good", b"good", b"good"], [b"good", b"good", b"bad"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"g\xc3\xc3d"]],
                      [x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"b\xc3d"]]],
    }[unit]
    position = np.array([1, 2, 4], dtype)
    length = np.array([1, 2, 3], dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

    # Broadcast (with negative)
    position = np.array([-1, -2, -4], dtype)
    length = np.array([1, 2, 3], dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testMismatchPosLenShapes(self, dtype, unit):
    # [FUNCTION DOC]
    # Passes position and length tensors whose rank or dimensionality differ from each other
    # (matrix vs vector, then 3x3 vs 1x3). Asserts a value error is raised in both cases.
    test_string = {
        "BYTE": [[b"ten", b"eleven", b"twelve"],
                 [b"thirteen", b"fourteen", b"fifteen"],
                 [b"sixteen", b"seventeen", b"eighteen"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227n",
                                                   u"\xc6\u053c\u025bv\u025bn",
                                                   u"tw\u0c1dlv\u025b"]],
                      [x.encode("utf-8") for x in [u"th\xcdrt\xea\xean",
                                                   u"f\U0001f604urt\xea\xean",
                                                   u"f\xcd\ua09ctee\ua0e4"]],
                      [x.encode("utf-8") for x in [u"s\xcdxt\xea\xean",
                                                   u"se\U00010299enteen",
                                                   u"ei\U0001e920h\x86een"]]],
    }[unit]
    position = np.array([[1, 2, 3]], dtype)
    length = np.array([2, 3, 4], dtype)
    # Should fail: position/length have different rank
    with self.assertRaises(ValueError):
      string_ops.substr(test_string, position, length)

    position = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype)
    length = np.array([[2, 3, 4]], dtype)
    # Should fail: position/length have different dimensionality
    with self.assertRaises(ValueError):
      string_ops.substr(test_string, position, length)

  @test_util.run_deprecated_v1
  def testWrongDtype(self):
    # [FUNCTION DOC]
    # Passes a floating-point position, then a floating-point length. Asserts a type error
    # is raised each time since only integer indices are accepted.
    with self.cached_session():
      with self.assertRaises(TypeError):
        string_ops.substr(b"test", 3.0, 1)
      with self.assertRaises(TypeError):
        string_ops.substr(b"test", 3, 1.0)

  @test_util.run_deprecated_v1
  def testInvalidUnit(self):
    # [FUNCTION DOC]
    # Passes an unsupported unit name. Asserts a value error is raised.
    with self.cached_session():
      with self.assertRaises(ValueError):
        string_ops.substr(b"test", 3, 1, unit="UTF8")

  def testInvalidPos(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue: a scalar string with a two-element position
    # vector and scalar length (positions one and minus one, then one and two). Asserts the
    # call or its evaluation raises a value error or an invalid-argument error.
    # Test case for GitHub issue 46900.
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      x = string_ops.substr(b"abc", len=1, pos=[1, -1])
      self.evaluate(x)

    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      x = string_ops.substr(b"abc", len=1, pos=[1, 2])
      self.evaluate(x)

  @parameterized.parameters(
      (np.int32, "BYTE", 1),
      (np.int32, "BYTE", 2),
      (np.int64, "BYTE", 1),
      (np.int64, "BYTE", 2),
      (np.int32, "UTF8_CHAR", 1),
      (np.int32, "UTF8_CHAR", 2),
      (np.int64, "UTF8_CHAR", 1),
      (np.int64, "UTF8_CHAR", 2),
  )
  def testSingleString(self, dtype, unit, rank):
    # [FUNCTION DOC]
    # A single-element string array (optionally given an extra leading dimension for rank two)
    # with vector positions and lengths. Asserts the output holds the expected three
    # substrings and has the same rank as the input.

    test_string = {
        "BYTE": [b"abcdefghijklmnopqrstuvwxyz"],
        "UTF8_CHAR": [
            u"\U0001d229\U0001d227n\U0001d229\U0001d227n\U0001d229\U0001d227n"
            .encode("utf-8")
        ],
    }[unit]
    position = np.array([1, 2, 3], dtype)
    length = np.array([1, 2, 1], dtype)
    expected_value = {
        "BYTE": [b"b", b"cd", b"d"],
        "UTF8_CHAR": [
            x.decode("utf-8") for x in
            [b"\xf0\x9d\x88\xa7", b"n\xf0\x9d\x88\xa9", b"\xf0\x9d\x88\xa9"]
        ],
    }[unit]

    test_string_tensor = np.array(test_string)
    expected_string_tensor = np.array(expected_value)
    if rank == 2:
      test_string_tensor = np.expand_dims(test_string_tensor, axis=0)
      expected_string_tensor = np.expand_dims(expected_string_tensor, axis=0)

    substr_op = string_ops.substr(
        test_string_tensor, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_string_tensor)
      self.assertEqual(substr.ndim, rank)

  @parameterized.parameters(
      (np.int32, "BYTE", 3),
      (np.int32, "BYTE", 10),
      (np.int64, "BYTE", 3),
      (np.int64, "BYTE", 10),
      (np.int32, "UTF8_CHAR", 3),
      (np.int32, "UTF8_CHAR", 10),
      (np.int64, "UTF8_CHAR", 3),
      (np.int64, "UTF8_CHAR", 10),
  )
  def testSingleStringHighRankFails(self, dtype, unit, rank):
    # [FUNCTION DOC]
    # Expands a single string array to rank three or ten and calls substr. Asserts an
    # unimplemented error is raised, since the op supports only up to rank two.

    test_string = {
        "BYTE": [b"abcdefghijklmnopqrstuvwxyz"],
        "UTF8_CHAR": [
            u"\U0001d229\U0001d227n\U0001d229\U0001d227n\U0001d229\U0001d227n"
            .encode("utf-8")
        ],
    }[unit]
    position = np.array([1, 2, 3], dtype)
    length = np.array([1, 2, 1], dtype)

    test_string_tensor = np.array(test_string)
    for _ in range(rank - 1):
      test_string_tensor = np.expand_dims(test_string_tensor, axis=0)

    with self.assertRaises(errors_impl.UnimplementedError):
      # substr is only supported up to rank 2
      substr_op = string_ops.substr(
          test_string_tensor, position, length, unit=unit)
      self.evaluate(substr_op)


if __name__ == "__main__":
  test.main()
