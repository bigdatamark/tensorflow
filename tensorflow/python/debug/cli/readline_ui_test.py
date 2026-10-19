# [MODULE DOC]
# Unit tests for the readline-based debugger UI. Defines a mock UI subclass that feeds scripted
# commands and records screen outputs instead of using the terminal, and a test class covering
# creation through the UI factory, running with immediate exit, empty, initial, valid and invalid
# commands, the exit callback, output redirection to a file and configuration commands.
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
"""Tests of the readline-based CLI."""
import os
import argparse
import tempfile

from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import readline_ui
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest


class MockReadlineUI(readline_ui.ReadlineUI):
  # [CLASS DOC]
  # HELPER: test double for the readline UI. Takes commands from a preset list instead of the
  # keyboard (tracking a counter of how many were consumed) and appends each displayed output to a
  # list kept in an observers mapping under the screen outputs key. Uses a temporary config file.
  """Test subclass of ReadlineUI that bypasses terminal manipulations."""

  def __init__(self, on_ui_exit=None, command_sequence=None):
    # [FUNCTION DOC]
    # Constructor. on_ui_exit is an optional exit callback; command_sequence is the list of
    # command strings to feed in order. Creates a temporary config file (its descriptor is not
    # closed), initializes the parent with a configuration object for it, and sets the command list,
    # a zero counter and the observers mapping with an empty list of screen outputs.
    _, config_file_path = tempfile.mkstemp()  # safe to ignore fd
    readline_ui.ReadlineUI.__init__(
        self,
        on_ui_exit=on_ui_exit,
        config=cli_config.CLIConfig(config_file_path=config_file_path))

    self._command_sequence = command_sequence
    self._command_counter = 0

    self.observers = {"screen_outputs": []}

  def _get_user_command(self):
    # [FUNCTION DOC]
    # HELPER: replaces keyboard input. Returns the next scripted command and advances the counter.
    # Raises an index error if the script is exhausted (tests end their lists with an exit).
    command = self._command_sequence[self._command_counter]
    self._command_counter += 1
    return command

  def _display_output(self, screen_output):
    # [FUNCTION DOC]
    # HELPER: replaces printing. Appends the given output object to the recorded screen outputs.
    self.observers["screen_outputs"].append(screen_output)


class CursesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the name suggests curses UI tests, but these tests exercise the readline UI and
  # the UI factory.
  # Tests for the readline UI using the mock UI. Fixture: a temporary directory and config file
  # path, plus a babble command handler producing a configurable number of "bar" lines.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Creates a temporary directory, derives a config file path inside it, asserts that
    # the file does not exist yet, then runs the parent setup.
    self._tmp_dir = tempfile.mkdtemp()
    self._tmp_config_path = os.path.join(self._tmp_dir, ".tfdbg_config")
    self.assertFalse(gfile.Exists(self._tmp_config_path))
    super(CursesTest, self).setUp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Cleanup. Recursively deletes the temporary directory, then runs the parent cleanup.
    file_io.delete_recursively(self._tmp_dir)
    super(CursesTest, self).tearDown()

  def _babble(self, args, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: sample command handler registered by tests. args is a list of argument strings
    # supporting an optional count flag (default 60); screen_info is ignored. Returns a rich text
    # lines object holding that many lines of the word bar. Bad arguments make the parser exit.
    ap = argparse.ArgumentParser(
        description="Do babble.", usage=argparse.SUPPRESS)
    ap.add_argument(
        "-n",
        "--num_times",
        dest="num_times",
        type=int,
        default=60,
        help="How many times to babble")

    parsed = ap.parse_args(args)

    lines = ["bar"] * parsed.num_times
    return debugger_cli_common.RichTextLines(lines)

  def testUIFactoryCreatesReadlineUI(self):
    # [FUNCTION DOC]
    # Asks the UI factory for the readline type with a temp config; expects a readline UI instance.
    ui = ui_factory.get_ui(
        "readline",
        config=cli_config.CLIConfig(config_file_path=self._tmp_config_path))
    self.assertIsInstance(ui, readline_ui.ReadlineUI)

  def testUIFactoryRaisesExceptionOnInvalidUIType(self):
    # [FUNCTION DOC]
    # Asks the factory for an unknown UI type; expects a value error naming the invalid type.
    with self.assertRaisesRegex(ValueError, "Invalid ui_type: 'foobar'"):
      ui_factory.get_ui(
          "foobar",
          config=cli_config.CLIConfig(config_file_path=self._tmp_config_path))

  def testUIFactoryRaisesExceptionOnInvalidUITypeGivenAvailable(self):
    # [FUNCTION DOC]
    # Asks for the readline type while restricting available types to curses only; expects a value
    # error saying readline is invalid.
    with self.assertRaisesRegex(ValueError, "Invalid ui_type: 'readline'"):
      ui_factory.get_ui(
          "readline",
          available_ui_types=["curses"],
          config=cli_config.CLIConfig(config_file_path=self._tmp_config_path))

  def testRunUIExitImmediately(self):
    # [FUNCTION DOC]
    # Runs the mock UI with only an exit command; expects it to return and produce no screen output.
    """Make sure that the UI can exit properly after launch."""

    ui = MockReadlineUI(command_sequence=["exit"])
    ui.run_ui()

    # No screen output should have happened.
    self.assertEqual(0, len(ui.observers["screen_outputs"]))

  def testRunUIEmptyCommand(self):
    # [FUNCTION DOC]
    # Feeds an empty command then exit; expects exactly one screen output (the invalid-prefix
    # error for the empty command).
    """Issue an empty command then exit."""

    ui = MockReadlineUI(command_sequence=["", "exit"])
    ui.run_ui()
    self.assertEqual(1, len(ui.observers["screen_outputs"]))

  def testRunUIWithInitCmd(self):
    # [FUNCTION DOC]
    # Registers the babble handler and runs with it as the initial command; expects one output of
    # sixty "bar" lines before the scripted exit.
    """Run UI with an initial command specified."""

    ui = MockReadlineUI(command_sequence=["exit"])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui(init_command="babble")

    screen_outputs = ui.observers["screen_outputs"]
    self.assertEqual(1, len(screen_outputs))
    self.assertEqual(["bar"] * 60, screen_outputs[0].lines)

  def testRunUIWithValidUsersCommands(self):
    # [FUNCTION DOC]
    # Scripts two babble commands (3 and 6 lines) then exit; expects two outputs with those line
    # counts.
    """Run UI with an initial command specified."""

    ui = MockReadlineUI(command_sequence=["babble -n 3", "babble -n 6", "exit"])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    screen_outputs = ui.observers["screen_outputs"]
    self.assertEqual(2, len(screen_outputs))
    self.assertEqual(["bar"] * 3, screen_outputs[0].lines)
    self.assertEqual(["bar"] * 6, screen_outputs[1].lines)

  def testRunUIWithInvalidUsersCommands(self):
    # [FUNCTION DOC]
    # Scripts a valid babble command, an unregistered command, then exit; expects two outputs: the
    # three bar lines and an error line about the invalid command prefix.
    """Run UI with an initial command specified."""

    ui = MockReadlineUI(command_sequence=["babble -n 3", "wobble", "exit"])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    screen_outputs = ui.observers["screen_outputs"]
    self.assertEqual(2, len(screen_outputs))
    self.assertEqual(["bar"] * 3, screen_outputs[0].lines)
    self.assertEqual(["ERROR: Invalid command prefix \"wobble\""],
                     screen_outputs[1].lines)

  def testRunUIWithOnUIExitCallback(self):
    # [FUNCTION DOC]
    # Runs the UI with an exit callback and an exit command; expects the callback not called
    # before the run, called after it, and no screen output.
    observer = {"callback_invoked": False}

    def callback_for_test():
      # [FUNCTION DOC]
      # Test callback: records that it was invoked in the shared observer mapping.
      observer["callback_invoked"] = True

    ui = MockReadlineUI(on_ui_exit=callback_for_test, command_sequence=["exit"])

    self.assertFalse(observer["callback_invoked"])
    ui.run_ui()

    self.assertEqual(0, len(ui.observers["screen_outputs"]))
    self.assertTrue(observer["callback_invoked"])

  def testIncompleteRedirectWorks(self):
    # [FUNCTION DOC]
    # Runs a babble command with output redirection to a temporary file. Expects the screen output
    # of two bar lines and the file to contain the two lines with trailing newlines.
    _, output_path = tempfile.mkstemp()  # safe to ignore fd

    ui = MockReadlineUI(
        command_sequence=["babble -n 2 > %s" % output_path, "exit"])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    screen_outputs = ui.observers["screen_outputs"]
    self.assertEqual(1, len(screen_outputs))
    self.assertEqual(["bar"] * 2, screen_outputs[0].lines)

    with gfile.Open(output_path, "r") as f:
      self.assertEqual("bar\nbar\n", f.read())

  def testConfigSetAndShow(self):
    # [FUNCTION DOC]
    # Runs a config set command (recursion depth five) then config show; expects the second output
    # to begin with the configuration heading, a blank line and the setting with value five.
    """Run UI with an initial command specified."""

    ui = MockReadlineUI(command_sequence=[
        "config set graph_recursion_depth 5", "config show", "exit"])
    ui.run_ui()
    outputs = ui.observers["screen_outputs"]
    self.assertEqual(
        ["Command-line configuration:",
         "",
         "  graph_recursion_depth: 5"], outputs[1].lines[:3])


if __name__ == "__main__":
  googletest.main()
