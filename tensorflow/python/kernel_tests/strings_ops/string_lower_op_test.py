# -*- coding: utf-8 -*-
# [MODULE DOC]
# Tests for the string lower-casing operation (string_ops.string_lower) on vectors, matrices and
# non-ASCII text with an explicit UTF-8 encoding. Run via the standard test main entry point.
# No import-time side effects. Not a helper module.
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
"""Tests for string_lower_op."""

from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringLowerOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the string lower-casing operation; each test evaluates it in a cached session.
  """Test cases for tf.strings.lower."""

  def test_string_lower(self):
    # [FUNCTION DOC]
    # Lower-cases a list of two mixed-case ASCII strings and asserts the lowercase byte strings.
    # Returns nothing.
    strings = ["Pigs on The Wing", "aNimals"]

    with self.cached_session():
      output = string_ops.string_lower(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [b"pigs on the wing", b"animals"])

  def test_string_lower_2d(self):
    # [FUNCTION DOC]
    # Lower-cases a two-by-two matrix of strings containing mixed case, spaces, newlines, tabs and
    # carriage returns, and asserts case changes while whitespace is preserved exactly.
    # Returns nothing.
    strings = [["pigS on THE wIng", "aniMals"], [" hello ", "\n\tWorld! \r \n"]]

    with self.cached_session():
      output = string_ops.string_lower(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [[b"pigs on the wing", b"animals"],
                                   [b" hello ", b"\n\tworld! \r \n"]])

  def test_string_upper_unicode(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name says upper-case but the test lower-cases text; it was likely
    # copied from the upper-case test file and never renamed.
    # Lower-cases an upper-case string with accented letters using the UTF-8 encoding option and
    # asserts the output bytes are the lowercase accented text. Returns nothing.
    strings = [["ÓÓSSCHLOË"]]
    with self.cached_session():
      output = string_ops.string_lower(strings, encoding="utf-8")
      output = self.evaluate(output)
      # output: "óósschloë"
      self.assertAllEqual(output, [[b"\xc3\xb3\xc3\xb3sschlo\xc3\xab"]])


if __name__ == "__main__":
  test.main()
