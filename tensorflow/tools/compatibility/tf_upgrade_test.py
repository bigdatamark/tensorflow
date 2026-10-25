# [MODULE DOC]
# Unit tests for the pre-1.0 to 1.0 TensorFlow upgrader (tf_upgrade). Runs small source text
# snippets through the AST code upgrader configured with TFAPIChangeSpec and checks the rewritten
# text, report and error list. Contains two test classes: one for in-memory conversions (renames,
# reorders, keyword renames, warnings, parse errors) and one for in-place file conversion.
# Run as a main program it starts the standard TensorFlow test runner.
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
"""Tests for tf upgrader."""

import io
import os
import tempfile

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import tf_upgrade


class TestUpgrade(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering the 1.0 upgrader on in-memory source text. Shared fixture is the
  # _upgrade helper which runs the upgrader with the 1.0 change spec on a text snippet.
  """Test various APIs that have been changed in 1.0.

  We also test whether a converted file is executable. test_file_v0_11.py
  aims to exhaustively test that API changes are convertible and actually
  work when run with current TensorFlow.
  """

  def _upgrade(self, old_file_text):
    # [FUNCTION DOC]
    # HELPER: Shared runner for the upgrade tests in this class.
    # Takes the old source text, wraps it in an in-memory input stream, runs the AST code upgrader
    # (built from the 1.0 change spec) writing to an in-memory output stream, and returns a tuple
    # of the processed count, the report text, the list of errors and the upgraded text.
    # No files are touched; parse failures are reported in the report rather than raised.
    in_file = io.StringIO(old_file_text)
    out_file = io.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade.TFAPIChangeSpec())
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def testParseError(self):
    # [FUNCTION DOC]
    # Upgrades syntactically invalid source (a dangling binary operator) and asserts the report
    # contains a "Failed to parse" message.
    _, report, unused_errors, unused_new_text = self._upgrade(
        "import tensorflow as tf\na + \n")
    self.assertNotEqual(report.find("Failed to parse"), -1)

  def testReport(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertion applies a truthiness check to the result of a text search, which
    # returns minus one (truthy) when not found and zero (falsy) when found at the start, so it
    # can pass when the message is missing; the check is effectively weak.
    # Upgrades a call to the removed multiply function and sanity-checks that the report
    # mentions the rename to the new multiply function.
    text = "tf.mul(a, b)\n"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    # This is not a complete test, but it is a sanity test that a report
    # is generating information.
    self.assertTrue(report.find("Renamed function `tf.mul` to `tf.multiply`"))

  def testRename(self):
    # [FUNCTION DOC]
    # Upgrades nested calls to the old multiply and subtract functions and asserts the output
    # text uses the new multiply and subtract names for both calls.
    text = "tf.mul(a, tf.sub(b, c))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.multiply(a, tf.subtract(b, c))\n")

  def testRenamePack(self):
    # [FUNCTION DOC]
    # Upgrades calls to the old pack and unpack functions in two separate runs and asserts they
    # become calls to stack and unstack respectively.
    text = "tf.pack(a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.stack(a)\n")
    text = "tf.unpack(a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.unstack(a)\n")

  def testReorder(self):
    # [FUNCTION DOC]
    # Upgrades positional calls to concat and split and asserts the arguments become keyword
    # arguments in the new order (axis and values; axis, number or sizes of splits, and value).
    text = "tf.concat(a, b)\ntf.split(a, b, c)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(axis=a, values=b)\n"
                     "tf.split(axis=a, num_or_size_splits=b, value=c)\n")

  def testConcatReorderWithKeywordArgs(self):
    # [FUNCTION DOC]
    # Upgrades concat calls that use the old concat dimension keyword, values keyword in either
    # order, or a mix of positional and keyword arguments. Asserts the old keyword becomes axis
    # while argument order is preserved and positional arguments are converted to keywords.
    text = "tf.concat(concat_dim=a, values=b)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(axis=a, values=b)\n")
    text = "tf.concat(values=b, concat_dim=a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(values=b, axis=a)\n")
    text = "tf.concat(a, values=b)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(axis=a, values=b)\n")

  def testConcatReorderNested(self):
    # [FUNCTION DOC]
    # Upgrades a concat call whose second argument is itself a concat call and asserts both the
    # outer and inner calls are converted to keyword arguments.
    text = "tf.concat(a, tf.concat(c, d))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text, "tf.concat(axis=a, values=tf.concat(axis=c, values=d))\n")

  def testInitializers(self):
    # [FUNCTION DOC]
    # Upgrades source that references the zeros and ones initializers both as bare names and as
    # calls. Asserts bare references are turned into calls while existing calls (including one
    # with a space before the parentheses) are left unchanged.
    text = ("tf.zeros_initializer;tf.zeros_initializer ()\n"
            "tf.ones_initializer;tf.ones_initializer ()\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text, "tf.zeros_initializer();tf.zeros_initializer ()\n"
                  "tf.ones_initializer();tf.ones_initializer ()\n")

  def testKeyword(self):
    # [FUNCTION DOC]
    # Upgrades a reduce-any call using the old reduction indices keyword and asserts the keyword
    # is renamed to axis with the value unchanged.
    text = "tf.reduce_any(a, reduction_indices=[1, 2])\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.reduce_any(a, axis=[1, 2])\n")

  def testComplexExpression(self):
    # [FUNCTION DOC]
    # Smoke test: upgrades an expression with a parenthesized sum, subscript and method call
    # and only checks that the upgrader does not raise. No output is asserted.
    text = "(foo + bar)[a].word()"
    _ = self._upgrade(text)

  def testReverse(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the first assertion compares the upgraded text with itself, so it always
    # passes and verifies nothing; probably meant to compare against expected output.
    # Upgrades a call to reverse and asserts the first reported error says reverse requires a
    # manual check.
    text = "tf.reverse(a, b)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, new_text)
    self.assertIn("tf.reverse requires manual check", errors[0])

  def testListComprehension(self):
    # [FUNCTION DOC]
    # Upgrades concat calls whose second argument is a list comprehension, with varied
    # whitespace, tabs and newlines inside and around the arguments. Asserts the positional
    # arguments become keywords while the original whitespace and comprehension are preserved.
    def _test(input, output):  # pylint: disable=redefined-builtin
      # [FUNCTION DOC]
      # HELPER: Local checker for testListComprehension.
      # Upgrades the input text and asserts the upgraded text equals the expected output text.
      # The report and error values are ignored. Returns nothing.
      _, unused_report, errors, new_text = self._upgrade(input)
      self.assertEqual(new_text, output)
    _test("tf.concat(0,  \t[x for x in y])\n",
          "tf.concat(axis=0,  \tvalues=[x for x in y])\n")
    _test("tf.concat(0,[x for x in y])\n",
          "tf.concat(axis=0,values=[x for x in y])\n")
    _test("tf.concat(0,[\nx for x in y])\n",
          "tf.concat(axis=0,values=[\nx for x in y])\n")
    _test("tf.concat(0,[\n \tx for x in y])\n",
          "tf.concat(axis=0,values=[\n \tx for x in y])\n")

  # TODO(aselle): Explicitly not testing command line interface and process_tree
  # for now, since this is a one off utility.


class TestUpgradeFiles(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering upgrading real files on disk with the 1.0 change spec.

  def testInplace(self):
    # [FUNCTION DOC]
    # Writes a temporary file containing an old multiply call, upgrades it in place (same input
    # and output path), then reads it back and asserts it contains the new multiply call. Deletes
    # the temporary file at the end (not removed if an assertion fails).
    """Check to make sure we don't have a file system race."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "tf.mul(a, b)\n"
    upgraded = "tf.multiply(a, b)\n"
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade.TFAPIChangeSpec())
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)


if __name__ == "__main__":
  test_lib.main()
