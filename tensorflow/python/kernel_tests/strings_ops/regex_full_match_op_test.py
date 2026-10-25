# [MODULE DOC]
# Tests for the regular-expression full-match string operation. One parameterized test class
# runs the same scenarios against both the dynamic-pattern and static-pattern generated ops;
# a second class checks that the public string_ops wrapper delegates to the static variant when
# the pattern is a Python string and to the dynamic variant when it is a tensor. Run via the
# standard test main entry point. No import-time side effects. Not a helper module.
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
"""Tests for RegexFullMatch op from string_ops."""

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


@parameterized.parameters(
    (gen_string_ops.regex_full_match),
    (gen_string_ops.static_regex_full_match))
class RegexFullMatchOpVariantsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case: the class decorator instantiates every test once for the dynamic
  # full-match op and once for the static full-match op, passing the op as the argument named op.
  # Covers matching, multi-dimensional input, empty patterns and invalid patterns.

  @test_util.run_deprecated_v1
  def testRegexFullMatch(self, op):
    # [FUNCTION DOC]
    # Applies the given op to two strings with the pattern requiring start and end with the
    # letter a, and asserts the result is true for the first string and false for the second.
    # Graph mode only. Returns nothing.
    values = ["abaaba", "abcdabcde"]
    with self.cached_session():
      input_tensor = constant_op.constant(values, dtypes.string)
      matched = op(input_tensor, "a.*a").eval()
      self.assertAllEqual([True, False], matched)

  @test_util.run_deprecated_v1
  def testRegexFullMatchTwoDims(self, op):
    # [FUNCTION DOC]
    # Same pattern as the basic test but on a two-by-two matrix of strings; asserts the output
    # keeps the shape with true, false in each row. Graph mode only. Returns nothing.
    values = [["abaaba", "abcdabcde"], ["acdcba", "ebcda"]]
    with self.cached_session():
      input_tensor = constant_op.constant(values, dtypes.string)
      matched = op(input_tensor, "a.*a").eval()
      self.assertAllEqual([[True, False], [True, False]], matched)

  @test_util.run_deprecated_v1
  def testEmptyMatch(self, op):
    # [FUNCTION DOC]
    # Uses an empty pattern on two non-empty strings and asserts neither fully matches (both
    # results false). Graph mode only. Returns nothing.
    values = ["abc", "1"]
    with self.cached_session():
      input_tensor = constant_op.constant(values, dtypes.string)
      matched = op(input_tensor, "").eval()
      self.assertAllEqual([False, False], matched)

  @test_util.run_deprecated_v1
  def testInvalidPattern(self, op):
    # [FUNCTION DOC]
    # Uses a malformed pattern (an unterminated character class) and asserts evaluating the op
    # raises an op error whose message mentions an invalid pattern. Returns nothing.
    values = ["abc", "1"]
    with self.cached_session():
      input_tensor = constant_op.constant(values, dtypes.string)
      invalid_pattern = "A["
      matched = op(input_tensor, invalid_pattern)
      with self.assertRaisesOpError("Invalid pattern"):
        self.evaluate(matched)


class RegexFullMatchOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying which underlying graph op the public regex_full_match wrapper creates,
  # judged by the name of the resulting tensor.

  @test_util.run_deprecated_v1
  def testRegexFullMatchDelegation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the failure message of the second assertion reports the name of the first op
    # instead of the name of the tensor being checked, so a failure would print a misleading name.
    # Calls the wrapper with a string pattern and asserts the op name does not begin with the
    # dynamic op name, then with a tensor pattern and asserts it does. Graph mode only.
    # Returns nothing.
    with self.cached_session():
      input_tensor = constant_op.constant("foo", dtypes.string)
      pattern = "[a-z]"
      op = string_ops.regex_full_match(input_tensor, pattern)
      self.assertFalse(op.name.startswith("RegexFullMatch"), op.name)

      pattern_tensor = constant_op.constant("[a-z]*", dtypes.string)
      op_tensor = string_ops.regex_full_match(input_tensor, pattern_tensor)
      self.assertTrue(op_tensor.name.startswith("RegexFullMatch"), op.name)

  @test_util.run_deprecated_v1
  def testStaticRegexFullMatchDelegation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the failure message of the second assertion reports the name of the first op
    # instead of the name of the tensor being checked, so a failure would print a misleading name.
    # Calls the wrapper with a plain string pattern and asserts the op name begins with the static
    # op name, then with a tensor pattern and asserts it begins with the dynamic op name.
    # Graph mode only. Returns nothing.
    with self.cached_session():
      input_tensor = constant_op.constant("foo", dtypes.string)
      pattern = "[a-z]*"
      op = string_ops.regex_full_match(input_tensor, pattern)
      self.assertTrue(op.name.startswith("StaticRegexFullMatch"), op.name)

      pattern_tensor = constant_op.constant("[a-z]*", dtypes.string)
      op_vec = string_ops.regex_full_match(input_tensor, pattern_tensor)
      self.assertTrue(op_vec.name.startswith("RegexFullMatch"), op.name)


if __name__ == "__main__":
  test.main()
