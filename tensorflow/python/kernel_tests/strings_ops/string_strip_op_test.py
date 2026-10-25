# [MODULE DOC]
# Tests for the string strip operation (string_ops.string_strip), which removes leading and
# trailing whitespace from each string. Covers vectors, matrices and empty or all-whitespace
# strings. Run via the standard test main entry point. No import-time side effects.
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
"""Tests for string_strip_op."""

from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringStripOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the string strip operation; each test evaluates it in a cached session.
  """ Test cases for tf.strings.strip."""

  def test_string_strip(self):
    # [FUNCTION DOC]
    # Strips a list of two strings that have no surrounding whitespace and asserts they are
    # returned unchanged. Returns nothing.
    strings = ["pigs on the wing", "animals"]

    with self.cached_session() as sess:
      output = string_ops.string_strip(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [b"pigs on the wing", b"animals"])

  def test_string_strip_2d(self):
    # [FUNCTION DOC]
    # Strips a two-by-two matrix where some strings have surrounding spaces, tabs, newlines or
    # carriage returns, and asserts the outer whitespace is removed while the shape is kept.
    # Returns nothing.
    strings = [["pigs on the wing", "animals"],
               [" hello ", "\n\tworld \r \n"]]

    with self.cached_session() as sess:
      output = string_ops.string_strip(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [[b"pigs on the wing", b"animals"],
                                   [b"hello", b"world"]])

  def test_string_strip_with_empty_strings(self):
    # [FUNCTION DOC]
    # Strips a list containing an empty string and a whitespace-only string and asserts both
    # become empty while other entries lose their surrounding spaces. Returns nothing.
    strings = [" hello ", "", "world ", " \t \r \n "]

    with self.cached_session() as sess:
      output = string_ops.string_strip(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [b"hello", b"", b"world", b""])


if __name__ == "__main__":
  test.main()
