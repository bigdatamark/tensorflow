# [MODULE DOC]
# Tests for the regular-expression replace string operation. A parameterized class runs scenarios
# against both the dynamic-pattern and static-pattern generated ops; a second class checks
# that the public string_ops wrapper picks the static op when pattern and rewrite are Python
# strings and the dynamic op otherwise. Also defines two tiny module-level helpers used as
# parameters. Run via the standard test main entry point. No import-time side effects.
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
"""Tests for RegexReplace op from string_ops."""

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


@parameterized.parameters(
    (gen_string_ops.regex_replace),
    (gen_string_ops.static_regex_replace))
class RegexReplaceOpVariantsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case: the class decorator runs each test once with the dynamic replace op
  # and once with the static replace op, passed in as the argument named op.

  @test_util.run_deprecated_v1
  def testForwarding(self, op):
    # [FUNCTION DOC]
    # Builds an input by taking five-character substrings of two strings (so the input buffer is
    # consumed only by the regex op, exercising the buffer-forwarding optimisation), replaces
    # every lowercase letter with a dot, and asserts the output is the expected dotted strings.
    # Graph mode only. Returns nothing.
    with self.cached_session():
      # Generate an input that is uniquely consumed by the regex op.
      # This exercises code paths which are optimized for this case
      # (e.g., using forwarding).
      inp = string_ops.substr(
          constant_op.constant(["AbCdEfG",
                                "HiJkLmN"], dtypes.string),
          pos=0,
          len=5)
      stripped = op(inp, "\\p{Ll}", ".")
      self.assertAllEqual([b"A.C.E", b"H.J.L"], stripped)

  @test_util.run_deprecated_v1
  def testRemovePrefix(self, op):
    # [FUNCTION DOC]
    # Removes a leading a-colon or b-colon prefix from a vector of strings with global replacement
    # turned off, and asserts only anchored prefixes are removed (a string where the prefix is
    # not at the start is unchanged). Graph mode only. Returns nothing.
    values = ["a:foo", "a:bar", "a:foo", "b:baz", "b:qux", "ca:b"]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = op(input_vector, "^(a:|b:)", "", replace_global=False)
      self.assertAllEqual([b"foo", b"bar", b"foo", b"baz", b"qux", b"ca:b"],
                          stripped)

  @test_util.run_deprecated_v1
  def testRegexReplace(self, op):
    # [FUNCTION DOC]
    # Wraps each greedy match of a-to-a in parentheses using a back-reference to the whole match,
    # on a string with an embedded newline and one without, and asserts the expected output.
    # Graph mode only. Returns nothing.
    values = ["aba\naba", "abcdabcde"]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = op(input_vector, "a.*a", "(\\0)")
      self.assertAllEqual([b"(aba)\n(aba)", b"(abcda)bcde"], stripped)

  @test_util.run_deprecated_v1
  def testEmptyMatch(self, op):
    # [FUNCTION DOC]
    # Replaces the empty pattern with the letter x and asserts x is inserted at every position
    # including both ends of each string. Graph mode only. Returns nothing.
    values = ["abc", "1"]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = op(input_vector, "", "x")
      self.assertAllEqual([b"xaxbxcx", b"x1x"], stripped)

  @test_util.run_deprecated_v1
  def testInvalidPattern(self, op):
    # [FUNCTION DOC]
    # Uses a malformed pattern (unterminated character class) and asserts evaluation raises an op
    # error mentioning an invalid pattern. Returns nothing.
    values = ["abc", "1"]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      invalid_pattern = "A["
      replace = op(input_vector, invalid_pattern, "x")
      with self.assertRaisesOpError("Invalid pattern"):
        self.evaluate(replace)

  @test_util.run_deprecated_v1
  def testGlobal(self, op):
    # [FUNCTION DOC]
    # Replaces every occurrence of ab with abc using the global flag set positionally, on strings
    # including an empty one, and asserts all occurrences are replaced and the empty string stays
    # empty. Graph mode only. Returns nothing.
    values = ["ababababab", "abcabcabc", ""]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = op(input_vector, "ab", "abc", True)
      self.assertAllEqual([b"abcabcabcabcabc", b"abccabccabcc", b""], stripped)


def as_string(s):
  # [FUNCTION DOC]
  # HELPER: parameter converter for the delegation test that leaves the value as a plain string.
  # Takes any value and returns it unchanged.
  return s


def as_tensor(s):
  # [FUNCTION DOC]
  # HELPER: parameter converter for the delegation test that turns a Python string into a
  # constant string tensor. Takes a string and returns the tensor.
  return constant_op.constant(s, dtypes.string)


class RegexReplaceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case verifying which underlying replace op the public regex_replace wrapper creates,
  # judged by the name of the resulting tensor.

  @parameterized.parameters(
      (as_string, as_tensor),
      (as_tensor, as_string),
      (as_tensor, as_tensor))
  @test_util.run_deprecated_v1
  def testRegexReplaceDelegation(self, pattern_fn, rewrite_fn):
    # [FUNCTION DOC]
    # Parameterized over three combinations in which at least one of pattern and rewrite is a
    # tensor (converter functions are the arguments). Calls the wrapper on a constant string and
    # asserts the resulting op name starts with the dynamic replace op name. Graph mode only.
    # Returns nothing.
    with self.cached_session():
      input_vector = constant_op.constant("foo", dtypes.string)
      pattern = pattern_fn("[a-z]")
      replace = rewrite_fn(".")
      op = string_ops.regex_replace(input_vector, pattern, replace)
      self.assertTrue(op.name.startswith("RegexReplace"))

  @test_util.run_deprecated_v1
  def testStaticRegexReplaceDelegation(self):
    # [FUNCTION DOC]
    # Calls the wrapper with plain string pattern and rewrite and asserts the resulting op name
    # starts with the static replace op name. Graph mode only. Returns nothing.
    with self.cached_session():
      input_vector = constant_op.constant("foo", dtypes.string)
      pattern = "[a-z]"
      replace = "."
      op = string_ops.regex_replace(input_vector, pattern, replace)
      self.assertTrue(op.name.startswith("StaticRegexReplace"))

if __name__ == "__main__":
  test.main()
