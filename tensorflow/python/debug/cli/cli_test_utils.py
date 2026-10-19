# [MODULE DOC]
# HELPER: shared assertion utilities for tfdbg command-line interface tests.
# Offers a whitespace-insensitive comparison of two lists of text lines and a numeric closeness
# check of an array against its textual array representation. Used by other tfdbg CLI test
# modules; no import-time side effects other than compiling a regular expression.
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
"""Testing utilities for tfdbg command-line interface."""
import re

import numpy as np


def assert_lines_equal_ignoring_whitespace(test, expected_lines, actual_lines):
  # [FUNCTION DOC]
  # HELPER: test assertion shared by tfdbg CLI tests.
  # Asserts through the given test case that the two sequences of strings have the same length
  # and that, line by line, they are equal once all whitespace is removed. Fails the test with a
  # line-count mismatch message otherwise. Returns nothing.
  """Assert equality in lines, ignoring all whitespace.

  Args:
    test: An instance of unittest.TestCase or its subtypes (e.g.,
      TensorFlowTestCase).
    expected_lines: Expected lines as an iterable of strings.
    actual_lines: Actual lines as an iterable of strings.
  """
  test.assertEqual(
      len(expected_lines), len(actual_lines),
      "Mismatch in the number of lines: %d vs %d" % (
          len(expected_lines), len(actual_lines)))
  for expected_line, actual_line in zip(expected_lines, actual_lines):
    test.assertEqual("".join(expected_line.split()),
                     "".join(actual_line.split()))


# Regular expression for separators between values in a string representation
# of an ndarray, exclusing whitespace.
_ARRAY_VALUE_SEPARATOR_REGEX = re.compile(r"(array|\(|\[|\]|\)|\||,)")


def assert_array_lines_close(test, expected_array, array_lines):
  # [FUNCTION DOC]
  # HELPER: test assertion shared by tfdbg CLI tests.
  # Parses the numbers out of lines showing an array's text form (separators such as the word
  # "array", parentheses, brackets, bars and commas are replaced by spaces), and asserts via the
  # test case that these values are close to the flattened expected array. The shape implied by
  # the lines is ignored. A non-numeric token raises a value error. Returns nothing.
  """Assert that the array value represented by lines is close to expected.

  Note that the shape of the array represented by the `array_lines` is ignored.

  Args:
    test: An instance of TensorFlowTestCase.
    expected_array: Expected value of the array.
    array_lines: A list of strings representing the array.
      E.g., "array([[ 1.0, 2.0 ], [ 3.0, 4.0 ]])"
      Assumes that values are separated by commas, parentheses, brackets, "|"
      characters and whitespace.
  """
  elements = []
  for line in array_lines:
    line = re.sub(_ARRAY_VALUE_SEPARATOR_REGEX, " ", line)
    elements.extend(float(s) for s in line.split())
  test.assertAllClose(np.array(expected_array).flatten(), elements)
