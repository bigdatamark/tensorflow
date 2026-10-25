# -*- coding: utf-8 -*-
# [MODULE DOC]
# Tests for the string upper-casing operation (string_ops.string_upper) on vectors, matrices and
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
"""Tests for string_upper_op."""

from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringUpperOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the string upper-casing operation; each test evaluates it in a cached session.
  """Test cases for tf.strings.upper."""

  def test_string_upper(self):
    # [FUNCTION DOC]
    # Upper-cases a list of two mixed-case ASCII strings and asserts the uppercase byte strings.
    # Returns nothing.
    strings = ["Pigs on The Wing", "aNimals"]

    with self.cached_session():
      output = string_ops.string_upper(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [b"PIGS ON THE WING", b"ANIMALS"])

  def test_string_upper_2d(self):
    # [FUNCTION DOC]
    # Upper-cases a two-by-two matrix of strings containing mixed case and whitespace characters,
    # asserting letters change case and whitespace is preserved exactly. Returns nothing.
    strings = [["pigS on THE wIng", "aniMals"], [" hello ", "\n\tWorld! \r \n"]]

    with self.cached_session():
      output = string_ops.string_upper(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [[b"PIGS ON THE WING", b"ANIMALS"],
                                   [b" HELLO ", b"\n\tWORLD! \r \n"]])

  def test_string_upper_unicode(self):
    # [FUNCTION DOC]
    # Upper-cases a lowercase string with accented letters using the UTF-8 encoding option and
    # asserts the output bytes are the uppercase accented text. Returns nothing.
    strings = [["óósschloë"]]
    with self.cached_session():
      output = string_ops.string_upper(strings, encoding="utf-8")
      output = self.evaluate(output)
      # output: "ÓÓSSCHLOË"
      self.assertAllEqual(output, [[b"\xc3\x93\xc3\x93SSCHLO\xc3\x8b"]])


if __name__ == "__main__":
  test.main()
