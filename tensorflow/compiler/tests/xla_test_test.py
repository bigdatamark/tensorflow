# [MODULE DOC]
# Unit tests for the manifest parsing helper of the XLA test fixture module (xla_test).
# Only the function that turns the text of a "disabled tests" manifest into a combined
# regular expression is exercised. No XLA device or session is needed. Defines no flags and has
# no import-time side effects; running the file as a program starts the standard test runner.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the XLATestCase test fixture base class."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import test


class XlaTestCaseTestCase(test.TestCase):
  # [CLASS DOC]
  # Test case covering parse_disabled_manifest from the xla_test module. It has no shared
  # fixtures; each test passes a small manifest string and checks the returned regular
  # expression text (the first element of the returned pair).

  def testManifestEmptyLineDoesNotCatchAll(self):
    # [FUNCTION DOC]
    # Scenario: a manifest consisting of a blank line, one test name and a trailing newline.
    # Action: parse it. Expected: the resulting regex text is exactly that one test name, i.e.
    # blank lines do not add an empty alternative (which would match every test name).
    manifest = """
testCaseOne
"""
    disabled_regex, _ = xla_test.parse_disabled_manifest(manifest)
    self.assertEqual(disabled_regex, "testCaseOne")

  def testManifestWholeLineCommentDoesNotCatchAll(self):
    # [FUNCTION DOC]
    # Scenario: a manifest whose first line is a whole-line comment followed by two test names.
    # Action: parse it. Expected: the regex text is the two names joined by an alternation
    # separator, so the comment line contributes nothing (no empty alternative).
    manifest = """# I am a comment
testCaseOne
testCaseTwo
"""
    disabled_regex, _ = xla_test.parse_disabled_manifest(manifest)
    self.assertEqual(disabled_regex, "testCaseOne|testCaseTwo")


if __name__ == "__main__":
  test.main()
