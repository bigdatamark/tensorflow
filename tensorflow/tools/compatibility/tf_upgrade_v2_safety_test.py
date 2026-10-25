# [MODULE DOC]
# KNOWN ISSUE: the end of the file contains stray duplicated test method definitions after the
# main-guard call (see their own notes).
# Unit tests for the safety-mode API change spec of the TensorFlow 1.x to 2.0 upgrade tool. The
# module defines one test class that runs the generic AST code upgrader with the safety-mode
# spec on small source snippets held in memory and checks the rewritten text and the report. It
# exercises import rewriting to the compat v1 namespace, exclusions (contrib, google, compat) and
# the Addons migration warnings. No import-time side effects beyond the test imports; running the
# file as a script starts the TensorFlow test runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf 2.0 upgrader in safety mode."""
import io

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import tf_upgrade_v2_safety


class TfUpgradeV2SafetyTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering the safety-mode change spec (which redirects TensorFlow imports to the compat
  # v1 namespace and warns about contrib/Addons symbols). Shared fixture: the private upgrade helper
  # that runs the code upgrader on in-memory text. Derived from the TensorFlow test case base class.

  def _upgrade(self, old_file_text):
    # [FUNCTION DOC]
    # HELPER: shared fixture for the tests in this class; runs the safety-mode upgrader on a snippet.
    # Input: old_file_text, the source text to upgrade (string). Builds in-memory input and output
    # streams, creates the code upgrader with the safety-mode change spec, and processes the stream
    # under fake file names. Returns a tuple of the number of processed files, the report text,
    # the list of error messages, and the resulting upgraded text. No files are touched.
    in_file = io.StringIO(old_file_text)
    out_file = io.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2_safety.TFAPIChangeSpec())
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def testContribWarning(self):
    # [FUNCTION DOC]
    # Scenario: a call to a contrib function in the source. Upgrades it and asserts the report
    # contains the warning that contrib will not be distributed. Returns nothing.
    text = "tf.contrib.foo()"
    _, report, _, _ = self._upgrade(text)
    expected_info = "tf.contrib will not be distributed"
    self.assertIn(expected_info, report)

  def testTensorFlowImport(self):
    # [FUNCTION DOC]
    # Scenario: plain import forms of the tensorflow package. Asserts that an aliased import,
    # an import combined with another module, a bare import (which gets an alias of the package
    # name), a submodule import and an aliased submodule import are all rewritten to the compat v1
    # namespace with the rest of the line preserved. Returns nothing.
    text = "import tensorflow as tf"
    expected_text = ("import tensorflow.compat.v1 as tf")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import tensorflow as tf, other_import as y"
    expected_text = ("import tensorflow.compat.v1 as tf, other_import as y")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import tensorflow"
    expected_text = ("import tensorflow.compat.v1 as tensorflow")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import tensorflow.foo"
    expected_text = "import tensorflow.compat.v1.foo"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import tensorflow.foo as bar"
    expected_text = "import tensorflow.compat.v1.foo as bar"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testTensorFlowGoogleImport(self):
    # [FUNCTION DOC]
    # Scenario: imports from the internal google subpackage. Asserts that imports of that
    # subpackage are left unchanged, and that already-versioned google compat imports (v1 and v2)
    # also stay unchanged. Returns nothing.
    text = "import tensorflow.google as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "import tensorflow.google"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "import tensorflow.google.compat.v1 as tf"
    expected_text = "import tensorflow.google.compat.v1 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import tensorflow.google.compat.v2 as tf"
    expected_text = "import tensorflow.google.compat.v2 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testTensorFlowImportInIndent(self):
    # [FUNCTION DOC]
    # Scenario: an import nested inside an indented try block with a trailing comment. Asserts the
    # import is rewritten to the compat v1 namespace while indentation, comment and surrounding
    # lines are preserved exactly. Returns nothing.
    text = """
try:
  import tensorflow as tf  # import line

  tf.ones([4, 5])
except AttributeError:
  pass
"""

    expected_text = """
try:
  import tensorflow.compat.v1 as tf  # import line

  tf.ones([4, 5])
except AttributeError:
  pass
"""
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testTensorFlowFromImport(self):
    # [FUNCTION DOC]
    # Scenario: from-import forms. Asserts that importing a name from the package, from a
    # submodule, and a wildcard import from the package are rewritten to read from the compat v1
    # namespace. Returns nothing.
    text = "from tensorflow import foo"
    expected_text = "from tensorflow.compat.v1 import foo"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "from tensorflow.foo import bar"
    expected_text = "from tensorflow.compat.v1.foo import bar"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "from tensorflow import *"
    expected_text = "from tensorflow.compat.v1 import *"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testTensorFlowImportAlreadyHasCompat(self):
    # [FUNCTION DOC]
    # Scenario: imports that already name a compat namespace (v1, v2, or a from-import of compat).
    # Asserts the text is left unchanged, so the upgrader does not double-rewrite. Returns nothing.
    text = "import tensorflow.compat.v1 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "import tensorflow.compat.v2 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "from tensorflow.compat import v2 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

  def testTensorFlowGoogleFromImport(self):
    # [FUNCTION DOC]
    # Scenario: from-imports of v1 and v2 from the google compat namespace. Asserts both are left
    # unchanged by the upgrader. Returns nothing.
    text = "from tensorflow.google.compat import v1 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "from tensorflow.google.compat import v2 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

  def testTensorFlowDontChangeContrib(self):
    # [FUNCTION DOC]
    # Scenario: contrib imports are not rewritten in safety mode. Upgrades two snippets, an import
    # of the contrib module with an alias and a from-import of contrib from the top-level package,
    # and asserts the output text is identical to the input in both cases. Returns nothing.
    text = "import tensorflow.contrib as foo"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "from tensorflow import contrib"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

  def test_contrib_to_addons_move(self):
    # [FUNCTION DOC]
    # Scenario: contrib symbols that moved to TensorFlow Addons produce a migration warning. For a
    # small mapping of four contrib layer symbols (poincare normalize, maxout, group norm, instance
    # norm) to their Addons names, upgrades a call to each symbol and asserts the Addons name
    # appears in the report text. Returns nothing; assertion failure if a name is missing.
    small_mapping = {
        "tf.contrib.layers.poincare_normalize":
            "tfa.layers.PoincareNormalize",
        "tf.contrib.layers.maxout":
            "tfa.layers.Maxout",
        "tf.contrib.layers.group_norm":
            "tfa.layers.GroupNormalization",
        "tf.contrib.layers.instance_norm":
            "tfa.layers.InstanceNormalization",
    }
    for symbol, replacement in small_mapping.items():
      text = "{}('stuff', *args, **kwargs)".format(symbol)
      _, report, _, _ = self._upgrade(text)
      self.assertIn(replacement, report)

if __name__ == "__main__":
  test_lib.main()
  def testTensorFlowDontChangeContrib(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this is a stray duplicate of the test method of the same name defined in the
    # test class; it sits after the main-guard call at module level inside the guard block, so it is
    # only defined (never run as a test) after the test runner has already finished, and it takes a
    # self parameter with no class around it.
    # Intended scenario: contrib imports must be left unchanged by the safety-mode upgrader. It
    # upgrades "import tensorflow.contrib as foo" and "from tensorflow import contrib" and asserts
    # the output text equals the input. Returns nothing; raises an assertion error on mismatch.
    text = "import tensorflow.contrib as foo"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "from tensorflow import contrib"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

  def test_contrib_to_addons_move(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: stray duplicate of the test method of the same name in the test class, placed
    # after the main-guard call at module level; it is never run as a test and has no class or
    # self available.
    # Intended scenario: for a small mapping of contrib symbols to Addons replacements, upgrade a
    # call to each contrib symbol and assert the replacement name appears in the report. Returns
    # nothing; raises an assertion error when a replacement is missing from the report.
    small_mapping = {
        "tf.contrib.layers.poincare_normalize":
            "tfa.layers.PoincareNormalize",
        "tf.contrib.layers.maxout":
            "tfa.layers.Maxout",
        "tf.contrib.layers.group_norm":
            "tfa.layers.GroupNormalization",
        "tf.contrib.layers.instance_norm":
            "tfa.layers.InstanceNormalization",
    }
    for symbol, replacement in small_mapping.items():
      text = "{}('stuff', *args, **kwargs)".format(symbol)
      _, report, _, _ = self._upgrade(text)
      self.assertIn(replacement, report)

if __name__ == "__main__":
  test_lib.main()
