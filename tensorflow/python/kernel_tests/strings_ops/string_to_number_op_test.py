# [MODULE DOC]
# Tests for the string-to-number parsing operation (parsing_ops.string_to_number) for float32,
# float64, int32 and int64 output types. A module constant holds the common error message prefix
# used to match conversion failures. One private helper runs a list of good and bad inputs
# through a string placeholder. Run via the standard test main entry point. No import-time side
# effects beyond defining the constant.
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
"""Tests for StringToNumber op from parsing_ops."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test

_ERROR_MESSAGE = "StringToNumberOp could not correctly convert string: "


class StringToNumberOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for string-to-number conversion. A private helper drives each type-specific test
  # with lists of valid input and expected number pairs and invalid input and expected error pairs.

  def _test(self, tf_type, good_pairs, bad_pairs):
    # [FUNCTION DOC]
    # HELPER: shared driver for the type-specific conversion tests.
    # Inputs: the target numeric type, a list of pairs of string and expected number, and a list
    # of pairs of string and expected error message text. Builds a conversion graph on a string
    # placeholder, then for each good pair feeds the string and asserts the result is close to the
    # number (not-a-number compares as expected by the close-comparison helper); for each bad pair
    # asserts evaluation raises an op error matching the message. Graph mode only (uses
    # evaluation with feeds). Returns nothing.
    with self.cached_session():
      # Build a small testing graph.
      input_string = array_ops.placeholder(dtypes.string)
      output = parsing_ops.string_to_number(
          input_string, out_type=tf_type)

      # Check all the good input/output pairs.
      for instr, outnum in good_pairs:
        result, = output.eval(feed_dict={input_string: [instr]})
        self.assertAllClose([outnum], [result])

      # Check that the bad inputs produce the right errors.
      for instr, outstr in bad_pairs:
        with self.assertRaisesOpError(outstr):
          output.eval(feed_dict={input_string: [instr]})

  @test_util.run_deprecated_v1
  def testToFloat(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the explanatory comments inside the table of cases are attached to the wrong
    # entries (the less-than-minimum comment sits before the not-a-number entry instead of the
    # large negative value).
    # Converts to 32-bit float: integers, decimals, a hexadecimal value, leading spaces, the
    # largest finite value, values just beyond it (which become positive or negative infinity),
    # not-a-number and infinity, and expects a conversion error for trailing garbage text.
    # Graph mode only. Returns nothing.
    self._test(dtypes.float32,
               [("0", 0), ("3", 3), ("-1", -1),
                ("1.12", 1.12), ("0xF", 15), ("   -10.5", -10.5),
                ("3.40282e+38", 3.40282e+38),
                # Greater than max value of float.
                ("3.40283e+38", float("INF")),
                ("-3.40283e+38", float("-INF")),
                # Less than min value of float.
                ("NAN", float("NAN")),
                ("INF", float("INF"))],
               [("10foobar", _ERROR_MESSAGE + "10foobar")])

  @test_util.run_deprecated_v1
  def testToDouble(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: comments in the case table say greater than the maximum or less than the minimum
    # of a float, but for 64-bit floats these values are simply representable and convert exactly.
    # Converts to 64-bit float the same table as the 32-bit test, with the near-limit values
    # now expected to convert to themselves instead of infinity, and expects a conversion error
    # for trailing garbage. Graph mode only. Returns nothing.
    self._test(dtypes.float64,
               [("0", 0), ("3", 3), ("-1", -1),
                ("1.12", 1.12), ("0xF", 15), ("   -10.5", -10.5),
                ("3.40282e+38", 3.40282e+38),
                # Greater than max value of float.
                ("3.40283e+38", 3.40283e+38),
                # Less than min value of float.
                ("-3.40283e+38", -3.40283e+38),
                ("NAN", float("NAN")),
                ("INF", float("INF"))],
               [("10foobar", _ERROR_MESSAGE + "10foobar")])

  @test_util.run_deprecated_v1
  def testToInt32(self):
    # [FUNCTION DOC]
    # Converts to 32-bit integer: zero, positives, negatives, leading spaces and the exact minimum
    # and maximum. Expects conversion errors for values one beyond either limit, a decimal
    # number and trailing garbage. Graph mode only. Returns nothing.
    self._test(dtypes.int32,
               [("0", 0), ("3", 3), ("-1", -1),
                ("    -10", -10),
                ("-2147483648", -2147483648),
                ("2147483647", 2147483647)],
               [   # Less than min value of int32.
                   ("-2147483649", _ERROR_MESSAGE + "-2147483649"),
                   # Greater than max value of int32.
                   ("2147483648", _ERROR_MESSAGE + "2147483648"),
                   ("2.9", _ERROR_MESSAGE + "2.9"),
                   ("10foobar", _ERROR_MESSAGE + "10foobar")])

  @test_util.run_deprecated_v1
  def testToInt64(self):
    # [FUNCTION DOC]
    # Converts to 64-bit integer the same values as the 32-bit test plus values just outside the
    # 32-bit range, which are now valid. Expects conversion errors for a decimal number and for
    # trailing garbage. Graph mode only. Returns nothing.
    self._test(dtypes.int64,
               [("0", 0), ("3", 3), ("-1", -1),
                ("    -10", -10),
                ("-2147483648", -2147483648),
                ("2147483647", 2147483647),
                ("-2147483649", -2147483649),  # Less than min value of int32.
                ("2147483648", 2147483648)],  # Greater than max value of int32.
               [("2.9", _ERROR_MESSAGE + "2.9"),
                ("10foobar", _ERROR_MESSAGE + "10foobar")])


if __name__ == "__main__":
  test.main()
