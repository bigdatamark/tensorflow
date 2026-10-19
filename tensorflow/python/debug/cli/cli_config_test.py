# [MODULE DOC]
# Unit tests for the tfdbg command-line configuration object (cli_config.CLIConfig).
# Covers creating the config file on first use, defaults, forward compatibility when a stored
# field is missing, setting values with type casting, persistence across instances, summary
# rendering with highlighting, and change callbacks. Uses a temporary directory for the file.
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
"""Tests for cli_config."""
import json
import os
import tempfile

from tensorflow.python.debug.cli import cli_config
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest


class CLIConfigTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the CLIConfig class. Each test uses a fresh temporary directory holding the
  # configuration file path, created in setUp and removed in tearDown.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates a temporary directory and records the path of a config file inside it, asserts
    # that the file does not yet exist, then calls the parent setUp. Returns nothing.
    self._tmp_dir = tempfile.mkdtemp()
    self._tmp_config_path = os.path.join(self._tmp_dir, ".tfdbg_config")
    self.assertFalse(gfile.Exists(self._tmp_config_path))
    super(CLIConfigTest, self).setUp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Recursively deletes the temporary directory created in setUp, then calls the parent
    # tearDown. Returns nothing.
    file_io.delete_recursively(self._tmp_dir)
    super(CLIConfigTest, self).tearDown()

  def testConstructCLIConfigWithoutFile(self):
    # [FUNCTION DOC]
    # Constructs a config when no file exists. Expects defaults (recursion depth 20, mouse mode
    # on), a key error when reading an unknown property, and the config file to be created.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    self.assertEqual(20, config.get("graph_recursion_depth"))
    self.assertEqual(True, config.get("mouse_mode"))
    with self.assertRaises(KeyError):
      config.get("property_that_should_not_exist")
    self.assertTrue(gfile.Exists(self._tmp_config_path))

  def testCLIConfigForwardCompatibilityTest(self):
    # [FUNCTION DOC]
    # Creates a config, removes the recursion-depth field from the stored JSON file, and
    # reloads it. Expects the missing field to fall back to its default value of 20.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    with open(self._tmp_config_path, "rt") as f:
      config_json = json.load(f)
    # Remove a field to simulate forward compatibility test.
    del config_json["graph_recursion_depth"]
    with open(self._tmp_config_path, "wt") as f:
      json.dump(config_json, f)

    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    self.assertEqual(20, config.get("graph_recursion_depth"))

  def testModifyConfigValue(self):
    # [FUNCTION DOC]
    # Sets an integer and a boolean property with values of the right type and expects get to
    # return the new values.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    config.set("graph_recursion_depth", 9)
    config.set("mouse_mode", False)
    self.assertEqual(9, config.get("graph_recursion_depth"))
    self.assertEqual(False, config.get("mouse_mode"))

  def testModifyConfigValueWithTypeCasting(self):
    # [FUNCTION DOC]
    # Sets properties using strings ("18" and "false") and expects them to be cast to the
    # property's native type (integer 18 and boolean false).
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    config.set("graph_recursion_depth", "18")
    config.set("mouse_mode", "false")
    self.assertEqual(18, config.get("graph_recursion_depth"))
    self.assertEqual(False, config.get("mouse_mode"))

  def testModifyConfigValueWithTypeCastingFailure(self):
    # [FUNCTION DOC]
    # Sets the boolean mouse-mode property to an unparseable string and expects a value error.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    with self.assertRaises(ValueError):
      config.set("mouse_mode", "maybe")

  def testLoadFromModifiedConfigFile(self):
    # [FUNCTION DOC]
    # Modifies two properties on one config, then creates a second config from the same file
    # path and expects it to see the persisted modified values.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    config.set("graph_recursion_depth", 9)
    config.set("mouse_mode", False)
    config2 = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    self.assertEqual(9, config2.get("graph_recursion_depth"))
    self.assertEqual(False, config2.get("mouse_mode"))

  def testSummarizeFromConfig(self):
    # [FUNCTION DOC]
    # Asks the config for its summary and expects the output lines to be a title, a blank line,
    # and one indented line per property showing its current value.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    output = config.summarize()
    self.assertEqual(
        ["Command-line configuration:",
         "",
         "  graph_recursion_depth: %d" % config.get("graph_recursion_depth"),
         "  mouse_mode: %s" % config.get("mouse_mode")], output.lines)

  def testSummarizeFromConfigWithHighlight(self):
    # [FUNCTION DOC]
    # Summarizes with the mouse-mode property highlighted. Expects the same text lines as the
    # plain summary, plus font attributes on the fourth line: the property name underlined and
    # bold, and the value bold.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    output = config.summarize(highlight="mouse_mode")
    self.assertEqual(
        ["Command-line configuration:",
         "",
         "  graph_recursion_depth: %d" % config.get("graph_recursion_depth"),
         "  mouse_mode: %s" % config.get("mouse_mode")], output.lines)
    self.assertEqual((2, 12, ["underline", "bold"]),
                     output.font_attr_segs[3][0])
    self.assertEqual((14, 18, "bold"), output.font_attr_segs[3][1])

  def testSetCallback(self):
    # [FUNCTION DOC]
    # Registers a callback on the recursion-depth property that copies the new value into a
    # test mapping, then changes the property and expects the callback to have observed it.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)

    test_value = {"graph_recursion_depth": -1}
    def callback(config):
      # [FUNCTION DOC]
      # Test callback: receives the config and stores its current recursion depth into the
      # enclosing test mapping. Returns nothing.
      test_value["graph_recursion_depth"] = config.get("graph_recursion_depth")
    config.set_callback("graph_recursion_depth", callback)

    config.set("graph_recursion_depth", config.get("graph_recursion_depth") - 1)
    self.assertEqual(test_value["graph_recursion_depth"],
                     config.get("graph_recursion_depth"))

  def testSetCallbackInvalidPropertyName(self):
    # [FUNCTION DOC]
    # Registering a callback for a nonexistent property name must raise a key error.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)

    with self.assertRaises(KeyError):
      config.set_callback("nonexistent_property_name", print)

  def testSetCallbackNotCallable(self):
    # [FUNCTION DOC]
    # Registering a non-callable value (an integer) as a callback must raise a type error.
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)

    with self.assertRaises(TypeError):
      config.set_callback("graph_recursion_depth", 1)


if __name__ == "__main__":
  googletest.main()
