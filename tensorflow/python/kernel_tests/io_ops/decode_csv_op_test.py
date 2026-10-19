# [MODULE DOC]
# Unit tests for the CSV decoding operation exposed through the parsing ops module.
# Contains a single test class that feeds CSV record strings and per-column default values to the
# decoder and checks either the decoded column values or the expected error messages. Covers
# quoting, delimiters, missing-value markers, defaults, column selection and invalid input.
# No import-time side effects other than normal imports; running the file launches the test main.
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
"""Tests for DecodeCSV op from parsing_ops."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class DecodeCSVOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the CSV decoding op: simple values, quoting rules, alternative delimiters,
  # NA markers, defaults, required fields, column selection and malformed-input errors.
  # Each test builds an argument mapping and delegates to the shared checking helper.

  def _test(self, args, expected_out=None, expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: shared checker used by every test in this class.
    # Inputs: args is a mapping of keyword arguments forwarded to the CSV decoder; expected_out is
    # an optional list of expected columns; expected_err_re is an optional error-message pattern.
    # Without an expected error, decodes and evaluates the result and compares each column to the
    # expected one (approximate comparison for float columns, exact otherwise).
    # With an expected error, asserts that decoding and evaluating raises an exception whose
    # message matches the pattern. Returns nothing.
    if expected_err_re is None:
      decode = parsing_ops.decode_csv(**args)
      out = self.evaluate(decode)

      for i, field in enumerate(out):
        if field.dtype == np.float32 or field.dtype == np.float64:
          self.assertAllClose(field, expected_out[i])
        else:
          self.assertAllEqual(field, expected_out[i])
    else:
      with self.assertRaisesWithPredicateMatch(Exception, expected_err_re):
        decode = parsing_ops.decode_csv(**args)
        self.evaluate(decode)

  def testSimple(self):
    # [FUNCTION DOC]
    # Decodes three single-field records (one quoted) with an integer default; expects one integer
    # column with values one, two, three.
    args = {
        "records": ["1", "2", '"3"'],
        "record_defaults": [[1]],
    }

    expected_out = [[1, 2, 3]]

    self._test(args, expected_out)

  def testSimpleWithScalarDefaults(self):
    # [FUNCTION DOC]
    # Decodes two-field records when the defaults are given as plain scalars rather than lists;
    # expects two integer columns.
    args = {
        "records": ["1,4", "2,5", "3,6"],
        "record_defaults": [1, 2],
    }

    expected_out = [[1, 2, 3], [4, 5, 6]]

    self._test(args, expected_out)

  def testSimpleWith2DDefaults(self):
    # [FUNCTION DOC]
    # Passes a rank-2 default value, which is invalid. Expects an invalid-argument error mentioning
    # rank at most one when running eagerly, or a value error about the shape rank in graph mode.
    args = {
        "records": ["1", "2", "3"],
        "record_defaults": [[[0]]],
    }

    if context.executing_eagerly():
      err_spec = errors.InvalidArgumentError, (
          "Each record default should be at "
          "most rank 1")
    else:
      err_spec = ValueError, "Shape must be at most rank 1 but is rank 2"
    with self.assertRaisesWithPredicateMatch(*err_spec):
      self._test(args)

  def testSimpleNoQuoteDelimiter(self):
    # [FUNCTION DOC]
    # With quote handling disabled and a string default, a quoted record is returned verbatim
    # including its quote characters.
    args = {
        "records": ["1", "2", '"3"'],
        "record_defaults": [[""]],
        "use_quote_delim": False,
    }

    expected_out = [[b"1", b"2", b'"3"']]

    self._test(args, expected_out)

  def testScalar(self):
    # [FUNCTION DOC]
    # Decodes a single scalar record string with two fields, the second being an empty quoted
    # field; expects the first value and the default for the second, as scalar outputs.
    args = {"records": '1,""', "record_defaults": [[3], [4]]}

    expected_out = [1, 4]

    self._test(args, expected_out)

  def test2D(self):
    # [FUNCTION DOC]
    # Decodes a two-dimensional batch of records; expects the output to keep the input shape, with
    # the empty quoted field replaced by the default value.
    args = {"records": [["1", "2"], ['""', "4"]], "record_defaults": [[5]]}
    expected_out = [[[1, 2], [5, 4]]]

    self._test(args, expected_out)

  def test2DNoQuoteDelimiter(self):
    # [FUNCTION DOC]
    # Two-dimensional record batch with quote handling disabled; expects raw strings, including
    # lone quote characters, to be returned unchanged with the input shape preserved.
    args = {
        "records": [["1", "2"], ['""', '"']],
        "record_defaults": [[""]],
        "use_quote_delim": False
    }
    expected_out = [[[b"1", b"2"], [b'""', b'"']]]

    self._test(args, expected_out)

  def testDouble(self):
    # [FUNCTION DOC]
    # Uses an empty double-precision array as default (field required) and decodes values at the
    # extremes of the double range; expects a double column with matching values.
    args = {
        "records": ["1.0", "-1.79e+308", '"1.79e+308"'],
        "record_defaults": [np.array([], dtype=np.double)],
    }

    expected_out = [[1.0, -1.79e+308, 1.79e+308]]

    self._test(args, expected_out)

  def testInt64(self):
    # [FUNCTION DOC]
    # Uses an empty 64-bit integer default and decodes a value above the 32-bit range; expects it
    # to be parsed correctly into a 64-bit integer column.
    args = {
        "records": ["1", "2", '"2147483648"'],
        "record_defaults": [np.array([], dtype=np.int64)],
    }

    expected_out = [[1, 2, 2147483648]]

    self._test(args, expected_out)

  def testComplexString(self):
    # [FUNCTION DOC]
    # Decodes string fields containing embedded commas, newlines, escaped doubled quotes and
    # surrounding spaces; expects quotes removed and escapes resolved, unquoted spaces kept.
    args = {
        "records": ['"1.0"', '"ab , c"', '"a\nbc"', '"ab""c"', " abc "],
        "record_defaults": [["1"]]
    }

    expected_out = [[b"1.0", b"ab , c", b"a\nbc", b'ab"c', b" abc "]]

    self._test(args, expected_out)

  def testMultiRecords(self):
    # [FUNCTION DOC]
    # Decodes several records with three fields of mixed types (float, integer, string); expects
    # three columns of the respective types.
    args = {
        "records": ["1.0,4,aa", "0.2,5,bb", "3,6,cc"],
        "record_defaults": [[1.0], [1], ["aa"]]
    }

    expected_out = [[1.0, 0.2, 3], [4, 5, 6], [b"aa", b"bb", b"cc"]]

    self._test(args, expected_out)

  def testNA(self):
    # [FUNCTION DOC]
    # Sets a missing-value marker string; fields equal to it are replaced by the column defaults
    # (including an empty string default for the string column).
    args = {
        "records": ["2.0,NA,aa", "NA,5,bb", "3,6,NA"],
        "record_defaults": [[0.0], [0], [""]],
        "na_value": "NA"
    }

    expected_out = [[2.0, 0.0, 3], [0, 5, 6], [b"aa", b"bb", b""]]

    self._test(args, expected_out)

  def testWithDefaults(self):
    # [FUNCTION DOC]
    # Records with empty fields; expects each empty field to be replaced by its column default.
    args = {
        "records": [",1,", "0.2,3,bcd", "3.0,,"],
        "record_defaults": [[1.0], [0], ["a"]]
    }

    expected_out = [[1.0, 0.2, 3.0], [1, 3, 0], [b"a", b"bcd", b"a"]]

    self._test(args, expected_out)

  def testWithDefaultsAndNoQuoteDelimiter(self):
    # [FUNCTION DOC]
    # Combines default substitution with quote handling disabled; a lone quote character in the
    # last field is returned literally as the string value.
    args = {
        "records": [",1,", "0.2,3,bcd", '3.0,,"'],
        "record_defaults": [[1.0], [0], ["a"]],
        "use_quote_delim": False,
    }

    expected_out = [[1.0, 0.2, 3.0], [1, 3, 0], [b"a", b"bcd", b"\""]]

    self._test(args, expected_out)

  def testWithTabDelim(self):
    # [FUNCTION DOC]
    # Uses a tab as the field delimiter; expects fields split on tabs with an empty trailing field
    # replaced by its default.
    args = {
        "records": ["1\t1", "0.2\t3", "3.0\t"],
        "record_defaults": [[1.0], [0]],
        "field_delim": "\t"
    }

    expected_out = [[1.0, 0.2, 3.0], [1, 3, 0]]

    self._test(args, expected_out)

  def testWithoutDefaultsError(self):
    # [FUNCTION DOC]
    # A required column (empty default) has an empty field in one record; expects an error saying
    # the field is required but missing in that record.
    args = {
        "records": [",1", "0.2,3", "3.0,"],
        "record_defaults": [[1.0], np.array([], dtype=np.int32)]
    }

    self._test(
        args, expected_err_re="Field 1 is required but missing in record 2!")

  def testWrongFieldIntError(self):
    # [FUNCTION DOC]
    # An integer column contains a non-numeric value; expects an error naming the field, record
    # and the invalid 32-bit integer text.
    args = {
        "records": [",1", "0.2,234a", "3.0,2"],
        "record_defaults": [[1.0], np.array([], dtype=np.int32)]
    }

    self._test(
        args, expected_err_re="Field 1 in record 1 is not a valid int32: 234a")

  def testOutOfRangeError(self):
    # [FUNCTION DOC]
    # An integer value too large for 32 bits; expects the same invalid-integer error as for
    # malformed text.
    args = {
        "records": ["1", "9999999999999999999999999", "3"],
        "record_defaults": [[1]]
    }

    self._test(
        args, expected_err_re="Field 0 in record 1 is not a valid int32: ")

  def testWrongFieldFloatError(self):
    # [FUNCTION DOC]
    # A float column contains text with trailing garbage; expects an error naming the field and
    # record as not a valid float.
    args = {
        "records": [",1", "0.2,2", "3.0adf,3"],
        "record_defaults": [[1.0], np.array([], dtype=np.int32)]
    }

    self._test(
        args, expected_err_re="Field 0 in record 2 is not a valid float: ")

  def testWrongFieldStringError(self):
    # [FUNCTION DOC]
    # A string record has quote characters inside an unquoted field; expects an error that
    # unquoted fields cannot contain quotes or line breaks.
    args = {"records": ['"1,a,"', "0.22", 'a"bc'], "record_defaults": [["a"]]}

    self._test(
        args, expected_err_re="Unquoted fields cannot have quotes/CRLFs inside")

  def testWrongDefaults(self):
    # [FUNCTION DOC]
    # Only one default is given but records have two fields; expects an error about the expected
    # versus actual field count in the first record.
    args = {"records": [",1", "0.2,2", "3.0adf,3"], "record_defaults": [[1.0]]}

    self._test(args, expected_err_re="Expect 1 fields but have 2 in record 0")

  def testShortQuotedString(self):
    # [FUNCTION DOC]
    # A record consisting of a single opening quote; expects an error that a quoted field must end
    # with a closing quote.
    args = {
        "records": ["\""],
        "record_defaults": [["default"]],
    }

    self._test(
        args, expected_err_re="Quoted field has to end with quote followed.*")

  def testSelectCols(self):
    # [FUNCTION DOC]
    # Selects the first two of three columns; expects only those columns, with empty fields
    # replaced by defaults.
    args = {
        "records": [",,", "4,5,6"],
        "record_defaults": [[1], [2]],
        "select_cols": [0, 1]
    }
    expected_out = [[1, 4], [2, 5]]
    self._test(args, expected_out)

  def testSelectColsInclLast(self):
    # [FUNCTION DOC]
    # Selects all columns including the last one, an edge case for the parser; expects all three
    # columns decoded correctly.
    # The last col is a edge-casey; add test for that
    args = {
        "records": [",,", "4,5,6"],
        "record_defaults": [[0], [1], [2]],
        "select_cols": [0, 1, 2]
    }
    expected_out = [[0, 4], [1, 5], [2, 6]]
    self._test(args, expected_out)

  def testWrongSelectColsInclLast(self):
    # [FUNCTION DOC]
    # Selects an index past the final column of a short record; expects a field-count mismatch
    # error for the first record.
    # The last col is a edge-casey; add test for that
    args = {
        "records": [",,", "4,5,6"],
        "record_defaults": [[0], [1], [2]],
        "select_cols": [0, 1, 3]
    }
    self._test(args, expected_err_re="Expect 3 fields but have 2 in record 0")

  def testWrongSelectColsLen(self):
    # [FUNCTION DOC]
    # The number of selected columns differs from the number of defaults; expects a value error
    # saying the lengths do not match.
    args = {
        "records": ["1,2,3", "4,5,6"],
        "record_defaults": [[0], [0], [0]],
        "select_cols": [0]
    }
    with self.assertRaisesWithPredicateMatch(
        ValueError, "Length of select_cols and record_defaults do not match."):
      self._test(args)

  def testWrongSelectColsSorting(self):
    # [FUNCTION DOC]
    # Selected column indices given in descending order; expects a value error that they must be
    # strictly increasing.
    args = {
        "records": ["1,2,3"],
        "record_defaults": [[0], [1]],
        "select_cols": [1, 0]
    }
    with self.assertRaisesWithPredicateMatch(
        ValueError, "select_cols is not strictly increasing."):
      self._test(args)

  def testWrongSelectColsIndicesNegative(self):
    # [FUNCTION DOC]
    # A negative selected column index; expects a value error about negative values.
    args = {
        "records": ["1,2,3"],
        "record_defaults": [[0], [1]],
        "select_cols": [-1, 0]  # -1 is not a valid index
    }
    with self.assertRaisesWithPredicateMatch(
        ValueError, "select_cols contains negative values."):
      self._test(args)

  def testWrongSelectColsIndicesTooHigh(self):
    # [FUNCTION DOC]
    # A selected index beyond the available columns; only one column parses, so expects a
    # field-count mismatch error for the first record.
    args = {
        "records": ["1,2,3"],
        "record_defaults": [[0], [1]],
        "select_cols": [0, 3]  # 3 is not a valid index
    }
    # Only successfully parses one of the columns
    self._test(args, expected_err_re="Expect 2 fields but have 1 in record 0")

  def testNumpyAttribute(self):
    # [FUNCTION DOC]
    # Passes a numpy array of zeros as the defaults and a constant tensor record. Eagerly, expects
    # it to be accepted and decode to five values; in graph mode, expects an error that a list was
    # expected for the defaults.
    args = {
        "record_defaults": np.zeros(5),
        "records": constant_op.constant("1,2,3,4,5"),
    }
    if context.executing_eagerly():
      self._test(args, expected_out=[1, 2, 3, 4, 5])
    else:
      self._test(args, expected_err_re="Expected list for 'record_defaults'")


if __name__ == "__main__":
  test.main()
