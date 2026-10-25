# [MODULE DOC]
# Unit tests for the compatibility conversion helpers (bytes and text conversion) in the util
# compat module, covering valid encoding names and invalid encoding names. Runs the TensorFlow
# test main when executed directly. No import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Compat tests."""

from tensorflow.python.platform import test
from tensorflow.python.util import compat


class CompatTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the bytes and text conversion functions of the compat module, focused on how
  # they treat encoding names.

  def testCompatValidEncoding(self):
    # [FUNCTION DOC]
    # Scenario: convert a text string to bytes with an alternate spelling of the utf-8 encoding
    # name, and convert bytes to text with the standard name. Expects the bytes result to equal
    # the plain ascii bytes and the text result to equal the original word.
    self.assertEqual(compat.as_bytes("hello", "utf8"), b"hello")
    self.assertEqual(compat.as_text(b"hello", "utf-8"), "hello")

  def testCompatInvalidEncoding(self):
    # [FUNCTION DOC]
    # Scenario: call the bytes conversion and the text conversion with an encoding name that
    # does not exist. Expects each call to raise a lookup error.
    with self.assertRaises(LookupError):
      compat.as_bytes("hello", "invalid")

    with self.assertRaises(LookupError):
      compat.as_text(b"hello", "invalid")


if __name__ == "__main__":
  test.main()
