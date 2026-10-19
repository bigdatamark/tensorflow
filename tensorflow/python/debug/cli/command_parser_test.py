# [MODULE DOC]
# Unit tests for the tfdbg command_parser module: command splitting, output redirect extraction,
# tensor name and slicing parsing, index and range parsing, readable size and time parsing,
# interval parsing, and the Interval containment check. One test class per parser function group.
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
"""Tests for TensorFlow Debugger command parser."""
import sys

from tensorflow.python.debug.cli import command_parser
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class ParseCommandTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for command_parser.parse_command (splitting commands into arguments).

  def testParseNoBracketsOrQuotes(self):
    # [FUNCTION DOC]
    # Empty, single-word and multi-word commands split on any whitespace (spaces and tabs) into
    # the expected argument lists.
    command = ""
    self.assertEqual([], command_parser.parse_command(command))

    command = "a"
    self.assertEqual(["a"], command_parser.parse_command(command))

    command = "foo bar baz qux"
    self.assertEqual(["foo", "bar", "baz", "qux"],
                     command_parser.parse_command(command))

    command = "foo bar\tbaz\t qux"
    self.assertEqual(["foo", "bar", "baz", "qux"],
                     command_parser.parse_command(command))

  def testParseLeadingTrailingWhitespaces(self):
    # [FUNCTION DOC]
    # Leading and trailing spaces and newlines are ignored when splitting.
    command = "  foo bar baz qux   "
    self.assertEqual(["foo", "bar", "baz", "qux"],
                     command_parser.parse_command(command))

    command = "\nfoo bar baz qux\n"
    self.assertEqual(["foo", "bar", "baz", "qux"],
                     command_parser.parse_command(command))

  def testParseCommandsWithBrackets(self):
    # [FUNCTION DOC]
    # Whitespace inside square brackets does not split an argument, including when the bracket is
    # glued to a name or stands alone and when extra spaces surround arguments.
    command = "pt foo[1, 2, :]"
    self.assertEqual(["pt", "foo[1, 2, :]"],
                     command_parser.parse_command(command))
    command = "pt  foo[1, 2, :]   -a"
    self.assertEqual(["pt", "foo[1, 2, :]", "-a"],
                     command_parser.parse_command(command))

    command = "inject_value foo [1, 2,:] 0"
    self.assertEqual(["inject_value", "foo", "[1, 2,:]", "0"],
                     command_parser.parse_command(command))

  def testParseCommandWithTwoArgsContainingBrackets(self):
    # [FUNCTION DOC]
    # Two bracketed arguments in one command are each kept intact, including an empty bracket
    # pair.
    command = "pt foo[1, :] bar[:, 2]"
    self.assertEqual(["pt", "foo[1, :]", "bar[:, 2]"],
                     command_parser.parse_command(command))

    command = "pt foo[] bar[:, 2]"
    self.assertEqual(["pt", "foo[]", "bar[:, 2]"],
                     command_parser.parse_command(command))

  def testParseCommandWithUnmatchedBracket(self):
    # [FUNCTION DOC]
    # With an unclosed bracket the result must differ from the result for a properly closed one
    # (the unmatched bracket is not treated as grouping).
    command = "pt  foo[1, 2, :"
    self.assertNotEqual(["pt", "foo[1, 2, :]"],
                        command_parser.parse_command(command))

  def testParseCommandsWithQuotes(self):
    # [FUNCTION DOC]
    # Quoted text (double or single quotes) stays one argument even with spaces and brackets, and
    # the surrounding pair of quotes is stripped, including for a quoted first word.
    command = "inject_value foo \"np.zeros([100, 500])\""
    self.assertEqual(["inject_value", "foo", "np.zeros([100, 500])"],
                     command_parser.parse_command(command))
    # The pair of double quotes should have been stripped.

    command = "inject_value foo 'np.zeros([100, 500])'"
    self.assertEqual(["inject_value", "foo", "np.zeros([100, 500])"],
                     command_parser.parse_command(command))
    # The pair of single quotes should have been stripped.

    command = "\"command prefix with spaces\" arg1"
    self.assertEqual(["command prefix with spaces", "arg1"],
                     command_parser.parse_command(command))

  def testParseCommandWithTwoArgsContainingQuotes(self):
    # [FUNCTION DOC]
    # Two quoted arguments are parsed with quotes stripped; an empty quoted argument becomes an
    # empty string.
    command = "foo \"bar\" \"qux\""
    self.assertEqual(["foo", "bar", "qux"],
                     command_parser.parse_command(command))

    command = "foo \"\" \"qux\""
    self.assertEqual(["foo", "", "qux"],
                     command_parser.parse_command(command))


class ExtractOutputFilePathTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for command_parser.extract_output_file_path (output redirect detection).

  def testNoOutputFilePathIsReflected(self):
    # [FUNCTION DOC]
    # Arguments without a redirect are returned unchanged with no output path.
    args, output_path = command_parser.extract_output_file_path(["pt", "a:0"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertIsNone(output_path)

  def testHasOutputFilePathInOneArgsIsReflected(self):
    # [FUNCTION DOC]
    # A final argument consisting of the redirect sign glued to a path yields that path and is
    # removed from the arguments.
    args, output_path = command_parser.extract_output_file_path(
        ["pt", "a:0", ">/tmp/foo.txt"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertEqual(output_path, "/tmp/foo.txt")

  def testHasOutputFilePathInTwoArgsIsReflected(self):
    # [FUNCTION DOC]
    # A separate redirect sign argument followed by a path argument yields the path and both are
    # removed from the arguments.
    args, output_path = command_parser.extract_output_file_path(
        ["pt", "a:0", ">", "/tmp/foo.txt"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertEqual(output_path, "/tmp/foo.txt")

  def testHasGreaterThanSignButNoFileNameCausesSyntaxError(self):
    # [FUNCTION DOC]
    # A trailing redirect sign with no path must raise a syntax error saying the redirect file
    # path is empty.
    with self.assertRaisesRegex(SyntaxError, "Redirect file path is empty"):
      command_parser.extract_output_file_path(
          ["pt", "a:0", ">"])

  def testOutputPathMergedWithLastArgIsHandledCorrectly(self):
    # [FUNCTION DOC]
    # A redirect sign and path glued onto the last real argument are split off into the output
    # path, leaving the argument itself.
    args, output_path = command_parser.extract_output_file_path(
        ["pt", "a:0>/tmp/foo.txt"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertEqual(output_path, "/tmp/foo.txt")

  def testOutputPathInLastArgGreaterThanInSecondLastIsHandledCorrectly(self):
    # [FUNCTION DOC]
    # When the redirect sign is glued to the end of the second-to-last argument and the path is
    # the last argument, the sign is stripped from that argument and the path is returned.
    args, output_path = command_parser.extract_output_file_path(
        ["pt", "a:0>", "/tmp/foo.txt"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertEqual(output_path, "/tmp/foo.txt")

  def testFlagWithEqualGreaterThanShouldIgnoreIntervalFlags(self):
    # [FUNCTION DOC]
    # Greater-than interval values given to option flags (glued with an equals sign, or as a
    # separate argument after a flag, with time, memory, negative and exponent numbers) must not
    # be mistaken for a redirect: arguments are unchanged and no output path is returned.
    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--execution_time=>100ms"])
    self.assertEqual(["lp", "--execution_time=>100ms"], args)
    self.assertIsNone(output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--execution_time", ">1.2s"])
    self.assertEqual(["lp", "--execution_time", ">1.2s"], args)
    self.assertIsNone(output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "-e", ">1200"])
    self.assertEqual(["lp", "-e", ">1200"], args)
    self.assertIsNone(output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--foo_value", ">-.2MB"])
    self.assertEqual(["lp", "--foo_value", ">-.2MB"], args)
    self.assertIsNone(output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--bar_value", ">-42e3GB"])
    self.assertEqual(["lp", "--bar_value", ">-42e3GB"], args)
    self.assertIsNone(output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--execution_time", ">=100ms"])
    self.assertEqual(["lp", "--execution_time", ">=100ms"], args)
    self.assertIsNone(output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--execution_time=>=100ms"])
    self.assertEqual(["lp", "--execution_time=>=100ms"], args)
    self.assertIsNone(output_path)

  def testFlagWithEqualGreaterThanShouldRecognizeFilePaths(self):
    # [FUNCTION DOC]
    # A trailing argument starting with the sign whose remainder is not a valid number-led
    # interval (or which follows no flag) is a file path: it is returned and removed.
    args, output_path = command_parser.extract_output_file_path(
        ["lp", ">1.2s"])
    self.assertEqual(["lp"], args)
    self.assertEqual("1.2s", output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--execution_time", ">x.yms"])
    self.assertEqual(["lp", "--execution_time"], args)
    self.assertEqual("x.yms", output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--memory", ">a.1kB"])
    self.assertEqual(["lp", "--memory"], args)
    self.assertEqual("a.1kB", output_path)

    args, output_path = command_parser.extract_output_file_path(
        ["lp", "--memory", ">e002MB"])
    self.assertEqual(["lp", "--memory"], args)
    self.assertEqual("e002MB", output_path)

  def testOneArgumentIsHandledCorrectly(self):
    # [FUNCTION DOC]
    # A single argument with no redirect is returned unchanged with no output path.
    args, output_path = command_parser.extract_output_file_path(["lt"])
    self.assertEqual(["lt"], args)
    self.assertIsNone(output_path)

  def testEmptyArgumentIsHandledCorrectly(self):
    # [FUNCTION DOC]
    # An empty argument list returns an empty list and no output path.
    args, output_path = command_parser.extract_output_file_path([])
    self.assertEqual([], args)
    self.assertIsNone(output_path)


class ParseTensorNameTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for command_parser.parse_tensor_name_with_slicing.

  def testParseTensorNameWithoutSlicing(self):
    # [FUNCTION DOC]
    # A plain tensor name yields the name and an empty slicing string.
    (tensor_name,
     tensor_slicing) = command_parser.parse_tensor_name_with_slicing(
         "hidden/weights/Variable:0")

    self.assertEqual("hidden/weights/Variable:0", tensor_name)
    self.assertEqual("", tensor_slicing)

  def testParseTensorNameWithSlicing(self):
    # [FUNCTION DOC]
    # A name followed by a bracketed slicing yields the name and the bracketed slicing separately.
    (tensor_name,
     tensor_slicing) = command_parser.parse_tensor_name_with_slicing(
         "hidden/weights/Variable:0[:, 1]")

    self.assertEqual("hidden/weights/Variable:0", tensor_name)
    self.assertEqual("[:, 1]", tensor_slicing)


class ValidateSlicingStringTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for command_parser.validate_slicing_string.

  def testValidateValidSlicingStrings(self):
    # [FUNCTION DOC]
    # Bracketed strings of digits, commas, spaces and colons are valid.
    self.assertTrue(command_parser.validate_slicing_string("[1]"))
    self.assertTrue(command_parser.validate_slicing_string("[2,3]"))
    self.assertTrue(command_parser.validate_slicing_string("[4, 5, 6]"))
    self.assertTrue(command_parser.validate_slicing_string("[7,:, :]"))

  def testValidateInvalidSlicingStrings(self):
    # [FUNCTION DOC]
    # Empty, unbalanced, or code-like strings (function calls, identifiers) are invalid.
    self.assertFalse(command_parser.validate_slicing_string(""))
    self.assertFalse(command_parser.validate_slicing_string("[1,"))
    self.assertFalse(command_parser.validate_slicing_string("2,3]"))
    self.assertFalse(command_parser.validate_slicing_string("[4, foo()]"))
    self.assertFalse(command_parser.validate_slicing_string("[5, bar]"))


class ParseIndicesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for command_parser.parse_indices.

  def testParseValidIndicesStringsWithBrackets(self):
    # [FUNCTION DOC]
    # Bracketed index lists (with surrounding spaces and negative numbers) parse to integer lists.
    self.assertEqual([0], command_parser.parse_indices("[0]"))
    self.assertEqual([0], command_parser.parse_indices(" [0] "))
    self.assertEqual([-1, 2], command_parser.parse_indices("[-1, 2]"))
    self.assertEqual([3, 4, -5],
                     command_parser.parse_indices("[3,4,-5]"))

  def testParseValidIndicesStringsWithoutBrackets(self):
    # [FUNCTION DOC]
    # Index lists without brackets parse the same way.
    self.assertEqual([0], command_parser.parse_indices("0"))
    self.assertEqual([0], command_parser.parse_indices(" 0 "))
    self.assertEqual([-1, 2], command_parser.parse_indices("-1, 2"))
    self.assertEqual([3, 4, -5], command_parser.parse_indices("3,4,-5"))

  def testParseInvalidIndicesStringsWithoutBrackets(self):
    # [FUNCTION DOC]
    # A non-numeric element, an unbalanced closing bracket, and a trailing comma each raise a
    # value error from integer conversion naming the offending text.
    with self.assertRaisesRegex(
        ValueError, r"invalid literal for int\(\) with base 10: 'a'"):
      self.assertEqual([0], command_parser.parse_indices("0,a"))

    with self.assertRaisesRegex(
        ValueError, r"invalid literal for int\(\) with base 10: '2\]'"):
      self.assertEqual([0], command_parser.parse_indices("1, 2]"))

    with self.assertRaisesRegex(
        ValueError, r"invalid literal for int\(\) with base 10: ''"):
      self.assertEqual([0], command_parser.parse_indices("3, 4,"))


class ParseRangesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for command_parser.parse_ranges. The class constant stands for infinity: the largest
  # finite float.

  INF_VALUE = sys.float_info.max

  def testParseEmptyRangeString(self):
    # [FUNCTION DOC]
    # Empty or whitespace-only input yields an empty list.
    self.assertEqual([], command_parser.parse_ranges(""))
    self.assertEqual([], command_parser.parse_ranges("  "))

  def testParseSingleRange(self):
    # [FUNCTION DOC]
    # A single range, including ones with infinite bounds, parses into a list holding one pair
    # (infinity mapped to the largest float).
    self.assertAllClose([[-0.1, 0.2]],
                        command_parser.parse_ranges("[-0.1, 0.2]"))
    self.assertAllClose([[-0.1, self.INF_VALUE]],
                        command_parser.parse_ranges("[-0.1, inf]"))
    self.assertAllClose([[-self.INF_VALUE, self.INF_VALUE]],
                        command_parser.parse_ranges("[-inf, inf]"))

  def testParseSingleListOfRanges(self):
    # [FUNCTION DOC]
    # A list of ranges parses into a list of pairs, including infinite bounds.
    self.assertAllClose([[-0.1, 0.2], [10.0, 12.0]],
                        command_parser.parse_ranges("[[-0.1, 0.2], [10,  12]]"))
    self.assertAllClose(
        [[-self.INF_VALUE, -1.0], [1.0, self.INF_VALUE]],
        command_parser.parse_ranges("[[-inf, -1.0],[1.0, inf]]"))

  def testParseInvalidRangeString(self):
    # [FUNCTION DOC]
    # Malformed text raises a syntax error; a range with the wrong number of elements raises a
    # value error; a complex-number first or second element raises a value error naming the
    # offending position.
    with self.assertRaises(SyntaxError):
      command_parser.parse_ranges("[[1,2]")

    with self.assertRaisesRegex(ValueError,
                                "Incorrect number of elements in range"):
      command_parser.parse_ranges("[1,2,3]")

    with self.assertRaisesRegex(ValueError,
                                "Incorrect number of elements in range"):
      command_parser.parse_ranges("[inf]")

    with self.assertRaisesRegex(ValueError,
                                "Incorrect type in the 1st element of range"):
      command_parser.parse_ranges("[1j, 1]")

    with self.assertRaisesRegex(ValueError,
                                "Incorrect type in the 2nd element of range"):
      command_parser.parse_ranges("[1, 1j]")


class ParseReadableSizeStrTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for command_parser.parse_readable_size_str.

  def testParseNoUnitWorks(self):
    # [FUNCTION DOC]
    # Bare numbers (with surrounding whitespace) are taken as byte counts.
    self.assertEqual(0, command_parser.parse_readable_size_str("0"))
    self.assertEqual(1024, command_parser.parse_readable_size_str("1024 "))
    self.assertEqual(2000, command_parser.parse_readable_size_str(" 2000 "))

  def testParseKiloBytesWorks(self):
    # [FUNCTION DOC]
    # Values with a k or kB unit (also fractional, with optional space) are scaled by 1024.
    self.assertEqual(0, command_parser.parse_readable_size_str("0kB"))
    self.assertEqual(1024**2, command_parser.parse_readable_size_str("1024 kB"))
    self.assertEqual(1024**2 * 2,
                     command_parser.parse_readable_size_str("2048k"))
    self.assertEqual(1024**2 * 2,
                     command_parser.parse_readable_size_str("2048kB"))
    self.assertEqual(1024 / 4, command_parser.parse_readable_size_str("0.25k"))

  def testParseMegaBytesWorks(self):
    # [FUNCTION DOC]
    # Values with an M or MB unit are scaled by 1024 squared.
    self.assertEqual(0, command_parser.parse_readable_size_str("0MB"))
    self.assertEqual(1024**3, command_parser.parse_readable_size_str("1024 MB"))
    self.assertEqual(1024**3 * 2,
                     command_parser.parse_readable_size_str("2048M"))
    self.assertEqual(1024**3 * 2,
                     command_parser.parse_readable_size_str("2048MB"))
    self.assertEqual(1024**2 / 4,
                     command_parser.parse_readable_size_str("0.25M"))

  def testParseGigaBytesWorks(self):
    # [FUNCTION DOC]
    # Values with a G or GB unit are scaled by 1024 cubed.
    self.assertEqual(0, command_parser.parse_readable_size_str("0GB"))
    self.assertEqual(1024**4, command_parser.parse_readable_size_str("1024 GB"))
    self.assertEqual(1024**4 * 2,
                     command_parser.parse_readable_size_str("2048G"))
    self.assertEqual(1024**4 * 2,
                     command_parser.parse_readable_size_str("2048GB"))
    self.assertEqual(1024**3 / 4,
                     command_parser.parse_readable_size_str("0.25G"))

  def testParseUnsupportedUnitRaisesException(self):
    # [FUNCTION DOC]
    # Unknown units raise a value error with the "failed to parse byte size" message.
    with self.assertRaisesRegex(
        ValueError, "Failed to parsed human-readable byte size str: \"0foo\""):
      command_parser.parse_readable_size_str("0foo")

    with self.assertRaisesRegex(
        ValueError, "Failed to parsed human-readable byte size str: \"2E\""):
      command_parser.parse_readable_size_str("2EB")


class ParseReadableTimeStrTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for command_parser.parse_readable_time_str.

  def testParseNoUnitWorks(self):
    # [FUNCTION DOC]
    # Bare numbers (with surrounding whitespace) are taken as microseconds.
    self.assertEqual(0, command_parser.parse_readable_time_str("0"))
    self.assertEqual(100, command_parser.parse_readable_time_str("100 "))
    self.assertEqual(25, command_parser.parse_readable_time_str(" 25 "))

  def testParseSeconds(self):
    # [FUNCTION DOC]
    # Values with an s suffix are converted to microseconds (times one million).
    self.assertEqual(1e6, command_parser.parse_readable_time_str("1 s"))
    self.assertEqual(2e6, command_parser.parse_readable_time_str("2s"))

  def testParseMicros(self):
    # [FUNCTION DOC]
    # Values with a us suffix are returned as-is in microseconds.
    self.assertEqual(2, command_parser.parse_readable_time_str("2us"))

  def testParseMillis(self):
    # [FUNCTION DOC]
    # Values with an ms suffix are converted to microseconds (times one thousand).
    self.assertEqual(2e3, command_parser.parse_readable_time_str("2ms"))

  def testParseUnsupportedUnitRaisesException(self):
    # [FUNCTION DOC]
    # Unknown suffixes raise a value error from float conversion; a negative value raises a
    # value error saying time must be positive.
    with self.assertRaisesRegex(ValueError, r".*float.*2us.*"):
      command_parser.parse_readable_time_str("2uss")

    with self.assertRaisesRegex(ValueError, r".*float.*2m.*"):
      command_parser.parse_readable_time_str("2m")

    with self.assertRaisesRegex(
        ValueError, r"Invalid time -1. Time value must be positive."):
      command_parser.parse_readable_time_str("-1s")


class ParseInterval(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the time and memory interval parsers and for Interval.contains.

  def testParseTimeInterval(self):
    # [FUNCTION DOC]
    # Bracket and parenthesis forms and the four one-sided operators parse to Interval objects
    # with microsecond bounds, correct inclusion flags, zero start and infinite end defaults.
    self.assertEqual(
        command_parser.Interval(10, True, 1e3, True),
        command_parser.parse_time_interval("[10us, 1ms]"))
    self.assertEqual(
        command_parser.Interval(10, False, 1e3, False),
        command_parser.parse_time_interval("(10us, 1ms)"))
    self.assertEqual(
        command_parser.Interval(10, False, 1e3, True),
        command_parser.parse_time_interval("(10us, 1ms]"))
    self.assertEqual(
        command_parser.Interval(10, True, 1e3, False),
        command_parser.parse_time_interval("[10us, 1ms)"))
    self.assertEqual(
        command_parser.Interval(0, False, 1e3, True),
        command_parser.parse_time_interval("<=1ms"))
    self.assertEqual(
        command_parser.Interval(1e3, True, float("inf"), False),
        command_parser.parse_time_interval(">=1ms"))
    self.assertEqual(
        command_parser.Interval(0, False, 1e3, False),
        command_parser.parse_time_interval("<1ms"))
    self.assertEqual(
        command_parser.Interval(1e3, False, float("inf"), False),
        command_parser.parse_time_interval(">1ms"))

  def testParseTimeGreaterLessThanWithInvalidValueStrings(self):
    # [FUNCTION DOC]
    # One-sided operators followed by a value that does not start with a number raise a value
    # error naming the operator.
    with self.assertRaisesRegex(ValueError, "Invalid value string after >= "):
      command_parser.parse_time_interval(">=wms")
    with self.assertRaisesRegex(ValueError, "Invalid value string after > "):
      command_parser.parse_time_interval(">Yms")
    with self.assertRaisesRegex(ValueError, "Invalid value string after <= "):
      command_parser.parse_time_interval("<= _ms")
    with self.assertRaisesRegex(ValueError, "Invalid value string after < "):
      command_parser.parse_time_interval("<-ms")

  def testParseTimeIntervalsWithInvalidValueStrings(self):
    # [FUNCTION DOC]
    # Two-value intervals whose first or second item does not start with a number raise a value
    # error naming which item is invalid.
    with self.assertRaisesRegex(ValueError, "Invalid first item in interval:"):
      command_parser.parse_time_interval("[wms, 10ms]")
    with self.assertRaisesRegex(ValueError, "Invalid second item in interval:"):
      command_parser.parse_time_interval("[ 0ms, _ms]")
    with self.assertRaisesRegex(ValueError, "Invalid first item in interval:"):
      command_parser.parse_time_interval("(xms, _ms]")
    with self.assertRaisesRegex(ValueError, "Invalid first item in interval:"):
      command_parser.parse_time_interval("((3ms, _ms)")

  def testInvalidTimeIntervalRaisesException(self):
    # [FUNCTION DOC]
    # Missing closing bracket, wrong number of values, and a start later than the end each raise
    # a value error with the corresponding message.
    with self.assertRaisesRegex(
        ValueError, r"Invalid interval format: \[10us, 1ms. Valid formats are: "
        r"\[min, max\], \(min, max\), <max, >min"):
      command_parser.parse_time_interval("[10us, 1ms")
    with self.assertRaisesRegex(
        ValueError,
        r"Incorrect interval format: \[10us, 1ms, 2ms\]. Interval should "
        r"specify two values: \[min, max\] or \(min, max\)"):
      command_parser.parse_time_interval("[10us, 1ms, 2ms]")
    with self.assertRaisesRegex(
        ValueError,
        r"Invalid interval \[1s, 1ms\]. Start must be before end of interval."):
      command_parser.parse_time_interval("[1s, 1ms]")

  def testParseMemoryInterval(self):
    # [FUNCTION DOC]
    # Same forms as the time case, parsed as memory sizes into byte bounds with inclusion flags,
    # zero start and infinite end defaults.
    self.assertEqual(
        command_parser.Interval(1024, True, 2048, True),
        command_parser.parse_memory_interval("[1k, 2k]"))
    self.assertEqual(
        command_parser.Interval(1024, False, 2048, False),
        command_parser.parse_memory_interval("(1kB, 2kB)"))
    self.assertEqual(
        command_parser.Interval(1024, False, 2048, True),
        command_parser.parse_memory_interval("(1k, 2k]"))
    self.assertEqual(
        command_parser.Interval(1024, True, 2048, False),
        command_parser.parse_memory_interval("[1k, 2k)"))
    self.assertEqual(
        command_parser.Interval(0, False, 2048, True),
        command_parser.parse_memory_interval("<=2k"))
    self.assertEqual(
        command_parser.Interval(11, True, float("inf"), False),
        command_parser.parse_memory_interval(">=11"))
    self.assertEqual(
        command_parser.Interval(0, False, 2048, False),
        command_parser.parse_memory_interval("<2k"))
    self.assertEqual(
        command_parser.Interval(11, False, float("inf"), False),
        command_parser.parse_memory_interval(">11"))

  def testParseMemoryIntervalsWithInvalidValueStrings(self):
    # [FUNCTION DOC]
    # Invalid one-sided values with memory-style units raise a value error naming the operator.
    # Note the test actually calls the time interval parser, not the memory one.
    with self.assertRaisesRegex(ValueError, "Invalid value string after >= "):
      command_parser.parse_time_interval(">=wM")
    with self.assertRaisesRegex(ValueError, "Invalid value string after > "):
      command_parser.parse_time_interval(">YM")
    with self.assertRaisesRegex(ValueError, "Invalid value string after <= "):
      command_parser.parse_time_interval("<= _MB")
    with self.assertRaisesRegex(ValueError, "Invalid value string after < "):
      command_parser.parse_time_interval("<-MB")

  def testInvalidMemoryIntervalRaisesException(self):
    # [FUNCTION DOC]
    # A memory interval with start greater than end raises a value error with the expected text.
    with self.assertRaisesRegex(
        ValueError,
        r"Invalid interval \[5k, 3k\]. Start of interval must be less than or "
        "equal to end of interval."):
      command_parser.parse_memory_interval("[5k, 3k]")

  def testIntervalContains(self):
    # [FUNCTION DOC]
    # Builds an interval from 1 to 10 and toggles each bound's inclusion flag, checking that
    # containment of the boundary values follows the flags while interior values are always in.
    interval = command_parser.Interval(
        start=1, start_included=True, end=10, end_included=True)
    self.assertTrue(interval.contains(1))
    self.assertTrue(interval.contains(10))
    self.assertTrue(interval.contains(5))

    interval.start_included = False
    self.assertFalse(interval.contains(1))
    self.assertTrue(interval.contains(10))

    interval.end_included = False
    self.assertFalse(interval.contains(1))
    self.assertFalse(interval.contains(10))

    interval.start_included = True
    self.assertTrue(interval.contains(1))
    self.assertFalse(interval.contains(10))


if __name__ == "__main__":
  googletest.main()
