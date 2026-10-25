# [MODULE DOC]
# Tests for the as_string operation in string_ops, which converts numeric, boolean, complex and
# string tensors to string tensors. Covers floating point types (half, bfloat16, float32,
# float64), signed and unsigned integers of several widths, booleans, complex types and
# strings, together with the formatting options (shortest, scientific, width, fill, precision)
# and the errors raised for invalid option combinations. Run through the standard test main
# entry point. No import-time side effects. Not a helper module.
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
"""Tests for as_string_op."""
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class AsStringOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the as_string operation. Each test method converts tensors of one family of
  # element types to strings and compares the output with Python percent-style formatting
  # of the same values. No shared fixtures beyond the base test case helpers.

  def testFloat(self):
    # [FUNCTION DOC]
    # For half, bfloat16, float32 and float64 tensors holding ordinary values plus positive
    # infinity, not-a-number and negative infinity, converts with as_string using combinations of
    # shortest, scientific, width, fill and precision options and asserts the decoded output
    # equals Python general, exponent or fixed formatting of the same numbers.
    # Also asserts that selecting both scientific and shortest, and a fill string longer than one
    # character, each raise an op error with the matching message. Returns nothing.
    float_inputs_ = [
        0, 1, -1, 0.5, 0.25, 0.125, float("INF"), float("NAN"), float("-INF")
    ]

    for dtype in (dtypes.half, dtypes.bfloat16, dtypes.float32, dtypes.float64):
      inputs = ops.convert_to_tensor(float_inputs_, dtype=dtype)
      s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]
      result = string_ops.as_string(inputs, shortest=True)
      self.assertAllEqual(s(result), ["%g" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, scientific=True)
      self.assertAllEqual(s(result), ["%e" % x for x in float_inputs_])

      result = string_ops.as_string(inputs)
      self.assertAllEqual(s(result), ["%f" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, width=3)
      self.assertAllEqual(s(result), ["%3f" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, width=3, fill="0")
      self.assertAllEqual(s(result), ["%03f" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, width=3, fill="0", shortest=True)
      self.assertAllEqual(s(result), ["%03g" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, precision=10, width=3)
      self.assertAllEqual(s(result), ["%03.10f" % x for x in float_inputs_])

      result = string_ops.as_string(
          inputs, precision=10, width=3, fill="0", shortest=True
      )
      self.assertAllEqual(s(result), ["%03.10g" % x for x in float_inputs_])

    with self.assertRaisesOpError("Cannot select both"):
      self.evaluate(
          string_ops.as_string(inputs, scientific=True, shortest=True)
      )

    with self.assertRaisesOpError("Fill string must be one or fewer"):
      self.evaluate(string_ops.as_string(inputs, fill="ab"))

  def testInt(self):
    # [FUNCTION DOC]
    # Converts signed (8, 32, 64 bit) and unsigned (8, 32, 64 bit) integer tensors, restricted to
    # values that fit in eight bits, with default, width and width-plus-zero-fill options and
    # asserts the output matches Python decimal formatting. Then asserts that the scientific
    # option, the shortest option, and a precision option are each rejected for integers with an
    # op error carrying the expected message. Returns nothing.
    # Cannot use values outside -128..127 for test, because we're also
    # testing int8
    int_inputs = [0, -1, 1, -128, 127, -101, 101, -0]
    int_dtypes = [dtypes.int8, dtypes.int32, dtypes.int64]
    uint_inputs = [0, 1, 127, 255, 101]
    uint_dtypes = [dtypes.uint8, dtypes.uint32, dtypes.uint64]
    s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]

    for dtypes_, inputs_ in [
        (int_dtypes, int_inputs),
        (uint_dtypes, uint_inputs),
    ]:
      for dtype in dtypes_:
        inputs = ops.convert_to_tensor(inputs_, dtype=dtype)
        result = string_ops.as_string(inputs)
        self.assertAllEqual(s(result), ["%d" % x for x in inputs_])

        result = string_ops.as_string(inputs, width=3)
        self.assertAllEqual(s(result), ["%3d" % x for x in inputs_])

        result = string_ops.as_string(inputs, width=3, fill="0")
        self.assertAllEqual(s(result), ["%03d" % x for x in inputs_])

      with self.assertRaisesOpError("scientific and shortest"):
        self.evaluate(string_ops.as_string(inputs, scientific=True))

      with self.assertRaisesOpError("scientific and shortest"):
        self.evaluate(string_ops.as_string(inputs, shortest=True))

      with self.assertRaisesOpError("precision not supported"):
        self.evaluate(string_ops.as_string(inputs, precision=0))

  def testLargeInt(self):
    # [FUNCTION DOC]
    # Converts Python lists holding the minimum and maximum 32-bit and 64-bit signed integers
    # and asserts the output strings equal their decimal representation. Returns nothing.
    # Cannot use values outside -128..127 for test, because we're also
    # testing int8
    s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]
    inputs = [np.iinfo(np.int32).min, np.iinfo(np.int32).max]
    result = string_ops.as_string(inputs)
    self.assertAllEqual(s(result), ["%d" % x for x in inputs])

    inputs = [np.iinfo(np.int64).min, np.iinfo(np.int64).max]
    result = string_ops.as_string(inputs)
    self.assertAllEqual(s(result), ["%d" % x for x in inputs])

  def testHalfInt(self):
    # [FUNCTION DOC]
    # Converts 16-bit signed and unsigned integer tensors holding each type's minimum and maximum
    # and asserts the output strings equal their decimal representation. Returns nothing.
    s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]
    for dtype, np_dtype in [
        (dtypes.int16, np.int16),
        (dtypes.uint16, np.uint16),
    ]:
      inputs = [np.iinfo(np_dtype).min, np.iinfo(np_dtype).max]
      result = string_ops.as_string(ops.convert_to_tensor(inputs, dtype=dtype))
      self.assertAllEqual(s(result), ["%d" % x for x in inputs])

  def testBool(self):
    # [FUNCTION DOC]
    # Converts a list of a false and a true value and asserts the output is the lowercase words
    # false and true. Returns nothing.
    bool_inputs_ = [False, True]
    s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]
    result = string_ops.as_string(bool_inputs_)
    self.assertAllEqual(s(result), ["false", "true"])

  def testComplex(self):
    # [FUNCTION DOC]
    # Builds complex numbers from ordinary, infinite and not-a-number parts and converts them as
    # complex64 and complex128 tensors with the shortest, scientific, width, fill and precision
    # options. Asserts each output equals a parenthesised pair of real and imaginary parts
    # formatted in Python style, after normalising negative not-a-number text to plain not-a-number.
    # Finally asserts that selecting both scientific and shortest raises an op error.
    # Returns nothing.
    inputs = [
        0,
        1,
        -1,
        0.5,
        0.25,
        0.125,
        complex("INF"),
        complex("NAN"),
        complex("-INF"),
    ]
    complex_inputs_ = [(x + (x + 1) * 1j) for x in inputs]

    for dtype in (dtypes.complex64, dtypes.complex128):
      inputs = ops.convert_to_tensor(complex_inputs_, dtype=dtype)

      def clean_nans(s_l):
        # [FUNCTION DOC]
        # HELPER: normalises output of the complex conversion test before comparison.
        # Evaluates the given string tensor, decodes every element from ascii bytes and replaces
        # the text for negative not-a-number with plain not-a-number. Returns a list of Python
        # strings.
        return [
            s.decode("ascii").replace("-nan", "nan") for s in self.evaluate(s_l)
        ]

      result = string_ops.as_string(inputs, shortest=True)
      self.assertAllEqual(
          clean_nans(result),
          ["(%g,%g)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs, scientific=True)
      self.assertAllEqual(
          clean_nans(result),
          ["(%e,%e)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs)
      self.assertAllEqual(
          clean_nans(result),
          ["(%f,%f)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs, width=3)
      self.assertAllEqual(
          clean_nans(result),
          ["(%03f,%03f)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs, width=3, fill="0", shortest=True)
      self.assertAllEqual(
          clean_nans(result),
          ["(%03g,%03g)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs, precision=10, width=3)
      self.assertAllEqual(
          clean_nans(result),
          ["(%03.10f,%03.10f)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(
          inputs, precision=10, width=3, fill="0", shortest=True
      )
      self.assertAllEqual(
          clean_nans(result),
          ["(%03.10g,%03.10g)" % (x.real, x.imag) for x in complex_inputs_],
      )

    with self.assertRaisesOpError("Cannot select both"):
      self.evaluate(
          string_ops.as_string(inputs, scientific=True, shortest=True)
      )

  def testString(self):
    # [FUNCTION DOC]
    # Asserts that converting a string returns it unchanged, and that requesting width twenty
    # left-pads it with spaces to that width. Returns nothing.
    self.assertAllEqual(string_ops.as_string("hello, world!"), "hello, world!")
    widened_string = self.evaluate(
        string_ops.as_string("hello, world!", width=20)
    )
    self.assertAllEqual(widened_string, "       hello, world!")


if __name__ == "__main__":
  test.main()
