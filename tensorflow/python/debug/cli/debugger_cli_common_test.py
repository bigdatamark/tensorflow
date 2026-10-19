# [MODULE DOC]
# Unit tests for debugger_cli_common, the building blocks of the TensorFlow debugger CLI.
# Covers CommandLineExit, RichTextLines (construction, extend, append, prepend, slice, file
# output), RichLine, regex_find, wrap_rich_text_lines, CommandHandlerRegistry (registration,
# dispatch, help), TabCompletionRegistry, CommandHistory (including file persistence),
# MenuItem/Menu and get_tensorflow_version_lines. Run through the TensorFlow test main entry.
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
"""Tests for Building Blocks of the TensorFlow Debugger CLI."""
import os
import stat
import tempfile

import numpy as np

from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest


class CommandLineExitTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the CommandLineExit exception class. No shared fixtures.

  def testConstructionWithoutToken(self):
    # [FUNCTION DOC]
    # Scenario: build the exception without a token. Expects it to be an exception instance.
    exit_exc = debugger_cli_common.CommandLineExit()

    self.assertTrue(isinstance(exit_exc, Exception))

  def testConstructionWithToken(self):
    # [FUNCTION DOC]
    # Scenario: build the exception with a mapping as exit token. Expects an exception
    # instance whose exit token equals the given mapping.
    exit_exc = debugger_cli_common.CommandLineExit(exit_token={"foo": "bar"})

    self.assertTrue(isinstance(exit_exc, Exception))
    self.assertEqual({"foo": "bar"}, exit_exc.exit_token)


class RichTextLinesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for RichTextLines and RichLine: construction, merging, append/prepend and file
  # output. No shared fixtures.

  def testRichTextLinesConstructorComplete(self):
    # [FUNCTION DOC]
    # Scenario: construct with two lines, segments and annotations for both rows. Expects two
    # lines, two segment entries with one segment each, two annotations and a line count of two.
    # Test RichTextLines constructor.
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]},
        annotations={0: "longer wavelength",
                     1: "shorter wavelength"})

    self.assertEqual(2, len(screen_output.lines))
    self.assertEqual(2, len(screen_output.font_attr_segs))
    self.assertEqual(1, len(screen_output.font_attr_segs[0]))
    self.assertEqual(1, len(screen_output.font_attr_segs[1]))
    self.assertEqual(2, len(screen_output.annotations))

    self.assertEqual(2, screen_output.num_lines())

  def testRichTextLinesConstructorWithInvalidType(self):
    # [FUNCTION DOC]
    # Scenario: construct with an integer instead of lines. Expects a value error whose message
    # mentions an unexpected type in lines.
    with self.assertRaisesRegex(ValueError, "Unexpected type in lines"):
      debugger_cli_common.RichTextLines(123)

  def testRichTextLinesConstructorWithString(self):
    # [FUNCTION DOC]
    # Scenario: construct with a single string plus segments and annotations for row zero.
    # Expects one line, one segment entry with one segment, and one annotation.
    # Test constructing a RichTextLines object with a string, instead of a list
    # of strings.
    screen_output = debugger_cli_common.RichTextLines(
        "Roses are red",
        font_attr_segs={0: [(0, 5, "red")]},
        annotations={0: "longer wavelength"})

    self.assertEqual(1, len(screen_output.lines))
    self.assertEqual(1, len(screen_output.font_attr_segs))
    self.assertEqual(1, len(screen_output.font_attr_segs[0]))
    self.assertEqual(1, len(screen_output.annotations))

  def testRichLinesAppendRichLine(self):
    # [FUNCTION DOC]
    # Scenario: start with one attributed line, then append a rich line made by concatenating a
    # plain rich line with an attributed one. Expects two lines and a segment list of length one
    # for each row.
    rtl = debugger_cli_common.RichTextLines(
        "Roses are red",
        font_attr_segs={0: [(0, 5, "red")]})
    rtl.append_rich_line(debugger_cli_common.RichLine("Violets are ") +
                         debugger_cli_common.RichLine("blue", "blue"))
    self.assertEqual(2, len(rtl.lines))
    self.assertEqual(2, len(rtl.font_attr_segs))
    self.assertEqual(1, len(rtl.font_attr_segs[0]))
    self.assertEqual(1, len(rtl.font_attr_segs[1]))

  def testRichLineLenMethodWorks(self):
    # [FUNCTION DOC]
    # Scenario: take the length of rich lines built with no text, empty text, one character and
    # text with an attribute. Expects lengths 0, 0, 1 and 6 (text length only).
    self.assertEqual(0, len(debugger_cli_common.RichLine()))
    self.assertEqual(0, len(debugger_cli_common.RichLine("")))
    self.assertEqual(1, len(debugger_cli_common.RichLine("x")))
    self.assertEqual(6, len(debugger_cli_common.RichLine("x y z ", "blue")))

  def testRichTextLinesConstructorIncomplete(self):
    # [FUNCTION DOC]
    # Scenario: construct with lines and segments but no annotations. Expects two lines, the
    # given segments, and an empty annotation mapping.
    # Test RichTextLines constructor, with incomplete keyword arguments.
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]})

    self.assertEqual(2, len(screen_output.lines))
    self.assertEqual(2, len(screen_output.font_attr_segs))
    self.assertEqual(1, len(screen_output.font_attr_segs[0]))
    self.assertEqual(1, len(screen_output.font_attr_segs[1]))
    self.assertEqual({}, screen_output.annotations)

  def testModifyRichTextLinesObject(self):
    # [FUNCTION DOC]
    # Scenario: append a string directly to the exposed lines list. Expects the object to
    # report three lines afterwards, showing the list is mutable in place.
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"])

    self.assertEqual(2, len(screen_output.lines))

    screen_output.lines.append("Sugar is sweet")
    self.assertEqual(3, len(screen_output.lines))

  def testMergeRichTextLines(self):
    # [FUNCTION DOC]
    # Scenario: extend one two-line object with another that also has a non-row annotation key.
    # Expects four lines, the second object's segments and annotations shifted by two rows, and
    # the non-row key kept. (The segment mapping is asserted twice; the repeat adds nothing.)
    screen_output_1 = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]},
        annotations={0: "longer wavelength",
                     1: "shorter wavelength"})
    screen_output_2 = debugger_cli_common.RichTextLines(
        ["Lilies are white", "Sunflowers are yellow"],
        font_attr_segs={0: [(0, 6, "white")],
                        1: [(0, 7, "yellow")]},
        annotations={
            "metadata": "foo",
            0: "full spectrum",
            1: "medium wavelength"
        })

    screen_output_1.extend(screen_output_2)

    self.assertEqual(4, screen_output_1.num_lines())
    self.assertEqual([
        "Roses are red", "Violets are blue", "Lilies are white",
        "Sunflowers are yellow"
    ], screen_output_1.lines)
    self.assertEqual({
        0: [(0, 5, "red")],
        1: [(0, 7, "blue")],
        2: [(0, 6, "white")],
        3: [(0, 7, "yellow")]
    }, screen_output_1.font_attr_segs)
    self.assertEqual({
        0: [(0, 5, "red")],
        1: [(0, 7, "blue")],
        2: [(0, 6, "white")],
        3: [(0, 7, "yellow")]
    }, screen_output_1.font_attr_segs)
    self.assertEqual({
        "metadata": "foo",
        0: "longer wavelength",
        1: "shorter wavelength",
        2: "full spectrum",
        3: "medium wavelength"
    }, screen_output_1.annotations)

  def testMergeRichTextLinesEmptyOther(self):
    # [FUNCTION DOC]
    # Scenario: extend a populated object with an empty one. Expects lines, segments and
    # annotations to stay unchanged.
    screen_output_1 = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]},
        annotations={0: "longer wavelength",
                     1: "shorter wavelength"})
    screen_output_2 = debugger_cli_common.RichTextLines([])

    screen_output_1.extend(screen_output_2)

    self.assertEqual(2, screen_output_1.num_lines())
    self.assertEqual(["Roses are red", "Violets are blue"],
                     screen_output_1.lines)
    self.assertEqual({
        0: [(0, 5, "red")],
        1: [(0, 7, "blue")],
    }, screen_output_1.font_attr_segs)
    self.assertEqual({
        0: [(0, 5, "red")],
        1: [(0, 7, "blue")],
    }, screen_output_1.font_attr_segs)
    self.assertEqual({
        0: "longer wavelength",
        1: "shorter wavelength",
    }, screen_output_1.annotations)

  def testMergeRichTextLinesEmptySelf(self):
    # [FUNCTION DOC]
    # Scenario: extend an empty object with a populated one. Expects the result to match the
    # populated object's lines, segments and annotations with no index shift.
    screen_output_1 = debugger_cli_common.RichTextLines([])
    screen_output_2 = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]},
        annotations={0: "longer wavelength",
                     1: "shorter wavelength"})

    screen_output_1.extend(screen_output_2)

    self.assertEqual(2, screen_output_1.num_lines())
    self.assertEqual(["Roses are red", "Violets are blue"],
                     screen_output_1.lines)
    self.assertEqual({
        0: [(0, 5, "red")],
        1: [(0, 7, "blue")],
    }, screen_output_1.font_attr_segs)
    self.assertEqual({
        0: [(0, 5, "red")],
        1: [(0, 7, "blue")],
    }, screen_output_1.font_attr_segs)
    self.assertEqual({
        0: "longer wavelength",
        1: "shorter wavelength",
    }, screen_output_1.annotations)

  def testAppendALineWithAttributeSegmentsWorks(self):
    # [FUNCTION DOC]
    # Scenario: append a line with its own attribute segments to a one-line object. Expects two
    # lines and the new segments recorded under row one.
    screen_output_1 = debugger_cli_common.RichTextLines(
        ["Roses are red"],
        font_attr_segs={0: [(0, 5, "red")]},
        annotations={0: "longer wavelength"})

    screen_output_1.append("Violets are blue", [(0, 7, "blue")])

    self.assertEqual(["Roses are red", "Violets are blue"],
                     screen_output_1.lines)
    self.assertEqual({
        0: [(0, 5, "red")],
        1: [(0, 7, "blue")],
    }, screen_output_1.font_attr_segs)

  def testPrependALineWithAttributeSegmentsWorks(self):
    # [FUNCTION DOC]
    # Scenario: prepend a line with segments to a one-line object. Expects the new line first
    # and the segments re-indexed so the old row's segments move to row one.
    screen_output_1 = debugger_cli_common.RichTextLines(
        ["Roses are red"],
        font_attr_segs={0: [(0, 5, "red")]},
        annotations={0: "longer wavelength"})

    screen_output_1.prepend("Violets are blue", font_attr_segs=[(0, 7, "blue")])

    self.assertEqual(["Violets are blue", "Roses are red"],
                     screen_output_1.lines)
    self.assertEqual({
        0: [(0, 7, "blue")],
        1: [(0, 5, "red")],
    }, screen_output_1.font_attr_segs)

  def testWriteToFileSucceeds(self):
    # [FUNCTION DOC]
    # Scenario: write a two-line object to a fresh temporary file. Expects the file to contain
    # the two lines each ending with a newline. Side effect: the temporary file is removed at
    # the end.
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]})

    fd, file_path = tempfile.mkstemp()
    os.close(fd)  # file opened exclusively, so we need to close this
    # a better fix would be to make the API take a fd
    screen_output.write_to_file(file_path)

    with gfile.Open(file_path, "r") as f:
      self.assertEqual("Roses are red\nViolets are blue\n", f.read())

    # Clean up.
    gfile.Remove(file_path)

  def testAttemptToWriteToADirectoryFails(self):
    # [FUNCTION DOC]
    # Scenario: write to the root directory path. Expects an exception to be raised.
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]})

    with self.assertRaises(Exception):
      screen_output.write_to_file("/")

  def testAttemptToWriteToFileInNonexistentDirectoryFails(self):
    # [FUNCTION DOC]
    # Scenario: write to a file inside a subdirectory that does not exist (under a new temporary
    # directory). Expects an exception to be raised.
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]})

    file_path = os.path.join(tempfile.mkdtemp(), "foo", "bar.txt")
    with self.assertRaises(Exception):
      screen_output.write_to_file(file_path)


class CommandHandlerRegistryTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for CommandHandlerRegistry: registration validation, dispatch, aliases, error
  # wrapping, exit handling and help output. Provides several small handler methods as fixtures.

  def setUp(self):
    # [FUNCTION DOC]
    # Stores the message used by the exception-raising handler fixture.
    self._intentional_error_msg = "Intentionally raised exception"

  def _noop_handler(self, argv, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: Handler fixture used by the registry tests.
    # Ignores its inputs and returns a one-line RichTextLines saying done.
    # A handler that does nothing other than returning "Done."
    return debugger_cli_common.RichTextLines(["Done."])

  def _handler_raising_exception(self, argv, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: Handler fixture used by the registry tests.
    # Always raises a runtime error carrying the intentional error message.
    # A handler that intentionally raises an exception.
    raise RuntimeError(self._intentional_error_msg)

  def _handler_returning_wrong_type(self, argv, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: Handler fixture used by the registry tests.
    # Returns a plain string instead of a RichTextLines, which is an invalid return type.
    # A handler that returns a wrong type, instead of the correct type
    # (RichTextLines).
    return "Hello"

  def _echo_screen_cols(self, argv, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: Handler fixture used by the registry tests.
    # Returns a one-line RichTextLines reporting the column count from the screen info mapping.
    # Fails if screen info is not provided.
    # A handler that uses screen_info.
    return debugger_cli_common.RichTextLines(
        ["cols = %d" % screen_info["cols"]])

  def _exiting_handler(self, argv, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: Handler fixture used by the registry tests.
    # Raises a command-line-exit exception whose token is the first argument, or null if no
    # arguments are given.
    """A handler that exits with an exit token."""

    if argv:
      exit_token = argv[0]
    else:
      exit_token = None

    raise debugger_cli_common.CommandLineExit(exit_token=exit_token)

  def testRegisterEmptyCommandPrefix(self):
    # [FUNCTION DOC]
    # Scenario: register a handler under an empty prefix. Expects a value error about an empty
    # command prefix.
    registry = debugger_cli_common.CommandHandlerRegistry()

    # Attempt to register an empty-string as a command prefix should trigger
    # an exception.
    with self.assertRaisesRegex(ValueError, "Empty command prefix"):
      registry.register_command_handler("", self._noop_handler, "")

  def testRegisterAndInvokeHandler(self):
    # [FUNCTION DOC]
    # Scenario: register a no-op handler and dispatch it. Expects registration queries to be true
    # only for the registered prefix, the output to be the done line, a value error for an
    # unregistered prefix, and a value error for an empty prefix.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("noop", self._noop_handler, "")

    self.assertTrue(registry.is_registered("noop"))
    self.assertFalse(registry.is_registered("beep"))

    cmd_output = registry.dispatch_command("noop", [])
    self.assertEqual(["Done."], cmd_output.lines)

    # Attempt to invoke an unregistered command prefix should trigger an
    # exception.
    with self.assertRaisesRegex(ValueError, "No handler is registered"):
      registry.dispatch_command("beep", [])

    # Empty command prefix should trigger an exception.
    with self.assertRaisesRegex(ValueError, "Prefix is empty"):
      registry.dispatch_command("", [])

  def testExitingHandler(self):
    # [FUNCTION DOC]
    # Scenario: register a handler that raises the exit exception and dispatch it with one
    # argument. Expects the exception to propagate out of dispatch carrying that argument as its
    # token.
    """Test that exit exception is correctly raised."""

    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("exit", self._exiting_handler, "")

    self.assertTrue(registry.is_registered("exit"))

    exit_token = None
    try:
      registry.dispatch_command("exit", ["foo"])
    except debugger_cli_common.CommandLineExit as e:
      exit_token = e.exit_token

    self.assertEqual("foo", exit_token)

  def testInvokeHandlerWithScreenInfo(self):
    # [FUNCTION DOC]
    # Scenario: dispatch a handler with a screen info mapping holding a column count of 100.
    # Expects the handler to receive it and the output to report the column count.
    registry = debugger_cli_common.CommandHandlerRegistry()

    # Register and invoke a command handler that uses screen_info.
    registry.register_command_handler("cols", self._echo_screen_cols, "")

    cmd_output = registry.dispatch_command(
        "cols", [], screen_info={"cols": 100})
    self.assertEqual(["cols = 100"], cmd_output.lines)

  def testRegisterAndInvokeHandlerWithAliases(self):
    # [FUNCTION DOC]
    # Scenario: register a handler with two aliases. Expects the prefix and both aliases to be
    # reported registered and dispatching through either alias to give the done output.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop", self._noop_handler, "", prefix_aliases=["n", "NOOP"])

    # is_registered() should work for full prefix and aliases.
    self.assertTrue(registry.is_registered("noop"))
    self.assertTrue(registry.is_registered("n"))
    self.assertTrue(registry.is_registered("NOOP"))

    cmd_output = registry.dispatch_command("n", [])
    self.assertEqual(["Done."], cmd_output.lines)

    cmd_output = registry.dispatch_command("NOOP", [])
    self.assertEqual(["Done."], cmd_output.lines)

  def testHandlerWithWrongReturnType(self):
    # [FUNCTION DOC]
    # Scenario: dispatch a handler that returns a string. Expects a value error saying the
    # return value is not null or a RichTextLines.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("wrong_return",
                                      self._handler_returning_wrong_type, "")

    # If the command handler fails to return a RichTextLines instance, an error
    # should be triggered.
    with self.assertRaisesRegex(
        ValueError,
        "Return value from command handler.*is not None or a RichTextLines "
        "instance"):
      registry.dispatch_command("wrong_return", [])

  def testRegisterDuplicateHandlers(self):
    # [FUNCTION DOC]
    # Scenario: register the same prefix twice. Expects a value error on the second attempt and
    # the original handler to still work.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("noop", self._noop_handler, "")

    # Registering the same command prefix more than once should trigger an
    # exception.
    with self.assertRaisesRegex(
        ValueError, "A handler is already registered for command prefix"):
      registry.register_command_handler("noop", self._noop_handler, "")

    cmd_output = registry.dispatch_command("noop", [])
    self.assertEqual(["Done."], cmd_output.lines)

  def testRegisterDuplicateAliases(self):
    # [FUNCTION DOC]
    # Scenario: register a second handler whose alias clashes with an existing alias, then one
    # whose alias clashes with an existing prefix. Expects a value error both times and the
    # second handler to remain unregistered.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop", self._noop_handler, "", prefix_aliases=["n"])

    # Clash with existing alias.
    with self.assertRaisesRegex(ValueError,
                                "clashes with existing prefixes or aliases"):
      registry.register_command_handler(
          "cols", self._echo_screen_cols, "", prefix_aliases=["n"])

    # The name clash should have prevent the handler from being registered.
    self.assertFalse(registry.is_registered("cols"))

    # Aliases can also clash with command prefixes.
    with self.assertRaisesRegex(ValueError,
                                "clashes with existing prefixes or aliases"):
      registry.register_command_handler(
          "cols", self._echo_screen_cols, "", prefix_aliases=["noop"])

    self.assertFalse(registry.is_registered("cols"))

  def testDispatchHandlerRaisingException(self):
    # [FUNCTION DOC]
    # Scenario: dispatch a handler that raises a runtime error. Expects dispatch not to raise
    # but to return text with more than two lines (including a traceback), whose first line
    # starts with the error-during-handling message and whose second line ends with the
    # intentional error message.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("raise_exception",
                                      self._handler_raising_exception, "")

    # The registry should catch and wrap exceptions that occur during command
    # handling.
    cmd_output = registry.dispatch_command("raise_exception", [])
    # The error output contains a stack trace.
    # So the line count should be >= 2.
    self.assertGreater(len(cmd_output.lines), 2)
    self.assertTrue(cmd_output.lines[0].startswith(
        "Error occurred during handling of command"))
    self.assertTrue(cmd_output.lines[1].endswith(self._intentional_error_msg))

  def testRegisterNonCallableHandler(self):
    # [FUNCTION DOC]
    # Scenario: register a non-callable handler (an integer). Expects a value error saying the
    # handler is not callable.
    registry = debugger_cli_common.CommandHandlerRegistry()

    # Attempt to register a non-callable handler should fail.
    with self.assertRaisesRegex(ValueError, "handler is not callable"):
      registry.register_command_handler("non_callable", 1, "")

  def testRegisterHandlerWithInvalidHelpInfoType(self):
    # [FUNCTION DOC]
    # Scenario: register a handler with a list as help info. Expects a value error saying help
    # info is not a string.
    registry = debugger_cli_common.CommandHandlerRegistry()

    with self.assertRaisesRegex(ValueError, "help_info is not a str"):
      registry.register_command_handler("noop", self._noop_handler, ["foo"])

  def testGetHelpFull(self):
    # [FUNCTION DOC]
    # Scenario: register two commands with aliases and fetch the full help. Expects commands in
    # alphabetical order (cols, the built-in help, then noop), each with an aliases line, a blank
    # line, indented help text and two trailing blank lines; checks specific line positions.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop",
        self._noop_handler,
        "No operation.\nI.e., do nothing.",
        prefix_aliases=["n", "NOOP"])
    registry.register_command_handler(
        "cols",
        self._echo_screen_cols,
        "Show screen width in number of columns.",
        prefix_aliases=["c"])

    help_lines = registry.get_help().lines

    # The help info should list commands in alphabetically sorted order,
    # regardless of order in which the commands are registered.
    self.assertEqual("cols", help_lines[0])
    self.assertTrue(help_lines[1].endswith("Aliases: c"))
    self.assertFalse(help_lines[2])
    self.assertTrue(help_lines[3].endswith(
        "Show screen width in number of columns."))

    self.assertFalse(help_lines[4])
    self.assertFalse(help_lines[5])

    # The default help command should appear in the help output.
    self.assertEqual("help", help_lines[6])

    self.assertEqual("noop", help_lines[12])
    self.assertTrue(help_lines[13].endswith("Aliases: n, NOOP"))
    self.assertFalse(help_lines[14])
    self.assertTrue(help_lines[15].endswith("No operation."))
    self.assertTrue(help_lines[16].endswith("I.e., do nothing."))

  def testGetHelpSingleCommand(self):
    # [FUNCTION DOC]
    # Scenario: fetch help for one command by its prefix and by its alias, and for an unknown
    # command. Expects the command name, aliases line, blank line and help text for the known
    # command both ways, and an invalid-prefix message for the unknown one.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop",
        self._noop_handler,
        "No operation.\nI.e., do nothing.",
        prefix_aliases=["n", "NOOP"])
    registry.register_command_handler(
        "cols",
        self._echo_screen_cols,
        "Show screen width in number of columns.",
        prefix_aliases=["c"])

    # Get help info for one of the two commands, using full prefix.
    help_lines = registry.get_help("cols").lines

    self.assertTrue(help_lines[0].endswith("cols"))
    self.assertTrue(help_lines[1].endswith("Aliases: c"))
    self.assertFalse(help_lines[2])
    self.assertTrue(help_lines[3].endswith(
        "Show screen width in number of columns."))

    # Get help info for one of the two commands, using alias.
    help_lines = registry.get_help("c").lines

    self.assertTrue(help_lines[0].endswith("cols"))
    self.assertTrue(help_lines[1].endswith("Aliases: c"))
    self.assertFalse(help_lines[2])
    self.assertTrue(help_lines[3].endswith(
        "Show screen width in number of columns."))

    # Get help info for a nonexistent command.
    help_lines = registry.get_help("foo").lines

    self.assertEqual("Invalid command prefix: \"foo\"", help_lines[0])

  def testHelpCommandWithoutIntro(self):
    # [FUNCTION DOC]
    # Scenario: dispatch the help command with no intro set: for all commands, for one existing
    # command and for an unknown one. Expects exact line lists, including the built-in help and
    # version commands in the full listing.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop",
        self._noop_handler,
        "No operation.\nI.e., do nothing.",
        prefix_aliases=["n", "NOOP"])
    registry.register_command_handler(
        "cols",
        self._echo_screen_cols,
        "Show screen width in number of columns.",
        prefix_aliases=["c"])

    # Get help for all commands.
    output = registry.dispatch_command("help", [])
    self.assertEqual(["cols", "  Aliases: c", "",
                      "  Show screen width in number of columns.", "", "",
                      "help", "  Aliases: h", "", "  Print this help message.",
                      "", "", "noop", "  Aliases: n, NOOP", "",
                      "  No operation.", "  I.e., do nothing.", "", "",
                      "version", "  Aliases: ver", "",
                      "  Print the versions of TensorFlow and its key "
                      "dependencies.", "", ""],
                     output.lines)

    # Get help for one specific command prefix.
    output = registry.dispatch_command("help", ["noop"])
    self.assertEqual(["noop", "  Aliases: n, NOOP", "", "  No operation.",
                      "  I.e., do nothing."], output.lines)

    # Get help for a nonexistent command prefix.
    output = registry.dispatch_command("help", ["foo"])
    self.assertEqual(["Invalid command prefix: \"foo\""], output.lines)

  def testHelpCommandWithIntro(self):
    # [FUNCTION DOC]
    # Scenario: set a help intro and dispatch the help command. Expects the intro lines first,
    # followed by the help, noop and version command entries.
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop",
        self._noop_handler,
        "No operation.\nI.e., do nothing.",
        prefix_aliases=["n", "NOOP"])

    help_intro = debugger_cli_common.RichTextLines(
        ["Introductory comments.", ""])
    registry.set_help_intro(help_intro)

    output = registry.dispatch_command("help", [])
    self.assertEqual(help_intro.lines + [
        "help", "  Aliases: h", "", "  Print this help message.", "", "",
        "noop", "  Aliases: n, NOOP", "", "  No operation.",
        "  I.e., do nothing.", "", "",
        "version", "  Aliases: ver", "",
        "  Print the versions of TensorFlow and its key dependencies.", "", ""
    ], output.lines)


class RegexFindTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for regex_find. Fixture: a two-line RichTextLines with no attributes.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the two-line original output used by every test in this class.
    self._orig_screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"])

  def testRegexFindWithoutExistingFontAttrSegs(self):
    # [FUNCTION DOC]
    # Scenario: search for a word that occurs on both lines. Expects one highlight segment per
    # line at the match positions and the match-lines annotation listing both rows.
    new_screen_output = debugger_cli_common.regex_find(self._orig_screen_output,
                                                       "are", "yellow")

    self.assertEqual(2, len(new_screen_output.font_attr_segs))
    self.assertEqual([(6, 9, "yellow")], new_screen_output.font_attr_segs[0])
    self.assertEqual([(8, 11, "yellow")], new_screen_output.font_attr_segs[1])

    # Check field in annotations carrying a list of matching line indices.
    self.assertEqual([0, 1], new_screen_output.annotations[
        debugger_cli_common.REGEX_MATCH_LINES_KEY])

  def testRegexFindWithExistingFontAttrSegs(self):
    # [FUNCTION DOC]
    # Scenario: give row zero an existing segment, then search. Expects the new match segment
    # merged with the existing one in order of start position, a new segment on row one, and the
    # match-lines annotation listing both rows.
    # Add a font attribute segment first.
    self._orig_screen_output.font_attr_segs[0] = [(9, 12, "red")]
    self.assertEqual(1, len(self._orig_screen_output.font_attr_segs))

    new_screen_output = debugger_cli_common.regex_find(self._orig_screen_output,
                                                       "are", "yellow")
    self.assertEqual(2, len(new_screen_output.font_attr_segs))

    self.assertEqual([(6, 9, "yellow"), (9, 12, "red")],
                     new_screen_output.font_attr_segs[0])

    self.assertEqual([0, 1], new_screen_output.annotations[
        debugger_cli_common.REGEX_MATCH_LINES_KEY])

  def testRegexFindWithNoMatches(self):
    # [FUNCTION DOC]
    # Scenario: search for text that does not occur. Expects no segments and an empty match-lines
    # annotation list.
    new_screen_output = debugger_cli_common.regex_find(self._orig_screen_output,
                                                       "infrared", "yellow")

    self.assertEqual({}, new_screen_output.font_attr_segs)
    self.assertEqual([], new_screen_output.annotations[
        debugger_cli_common.REGEX_MATCH_LINES_KEY])

  def testInvalidRegex(self):
    # [FUNCTION DOC]
    # Scenario: search with an unterminated bracket pattern. Expects a value error about an
    # invalid regular expression.
    with self.assertRaisesRegex(ValueError, "Invalid regular expression"):
      debugger_cli_common.regex_find(self._orig_screen_output, "[", "yellow")

  def testRegexFindOnPrependedLinesWorks(self):
    # [FUNCTION DOC]
    # Scenario: build objects via prepend (once with a one-element list line, once with a list
    # line carrying segments), then search. Expects the highlight at the right place on row zero,
    # alongside the prepended segment in the second case.
    rich_lines = debugger_cli_common.RichTextLines(["Violets are blue"])
    rich_lines.prepend(["Roses are red"])
    searched_rich_lines = debugger_cli_common.regex_find(
        rich_lines, "red", "bold")
    self.assertEqual(
        {0: [(10, 13, "bold")]}, searched_rich_lines.font_attr_segs)

    rich_lines = debugger_cli_common.RichTextLines(["Violets are blue"])
    rich_lines.prepend(["A poem"], font_attr_segs=[(0, 1, "underline")])
    searched_rich_lines = debugger_cli_common.regex_find(
        rich_lines, "poem", "italic")
    self.assertEqual(
        {0: [(0, 1, "underline"), (2, 6, "italic")]},
        searched_rich_lines.font_attr_segs)


class WrapScreenOutputTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for wrap_rich_text_lines. Fixture: a three-line object with multiple segments and
  # annotations on rows one and two.

  def setUp(self):
    # [FUNCTION DOC]
    # Builds the three-line original output with segments and annotations used by the tests.
    self._orig_screen_output = debugger_cli_common.RichTextLines(
        ["Folk song:", "Roses are red", "Violets are blue"],
        font_attr_segs={1: [(0, 5, "red"), (6, 9, "gray"), (10, 12, "red"),
                            (12, 13, "crimson")],
                        2: [(0, 7, "blue"), (8, 11, "gray"), (12, 14, "blue"),
                            (14, 16, "indigo")]},
        annotations={1: "longer wavelength",
                     2: "shorter wavelength"})

  def testNoActualWrapping(self):
    # [FUNCTION DOC]
    # Scenario: wrap with a column limit of 100. Expects lines, segments and annotations
    # unchanged and line indices zero, one, two.
    # Large column limit should lead to no actual wrapping.
    out, new_line_indices = debugger_cli_common.wrap_rich_text_lines(
        self._orig_screen_output, 100)

    self.assertEqual(self._orig_screen_output.lines, out.lines)
    self.assertEqual(self._orig_screen_output.font_attr_segs,
                     out.font_attr_segs)
    self.assertEqual(self._orig_screen_output.annotations, out.annotations)
    self.assertEqual(new_line_indices, [0, 1, 2])

  def testWrappingWithAttrCutoff(self):
    # [FUNCTION DOC]
    # Scenario: wrap with 11 columns, then add a non-row annotation to the result. Expects five
    # lines, segments split and clipped at wrap boundaries, row annotations moved to the first
    # wrapped row of each source line, the non-row key present, and line indices zero, one, three.
    out, new_line_indices = debugger_cli_common.wrap_rich_text_lines(
        self._orig_screen_output, 11)

    # Add non-row-index field to out.
    out.annotations["metadata"] = "foo"

    # Check wrapped text.
    self.assertEqual(5, len(out.lines))
    self.assertEqual("Folk song:", out.lines[0])
    self.assertEqual("Roses are r", out.lines[1])
    self.assertEqual("ed", out.lines[2])
    self.assertEqual("Violets are", out.lines[3])
    self.assertEqual(" blue", out.lines[4])

    # Check wrapped font_attr_segs.
    self.assertFalse(0 in out.font_attr_segs)
    self.assertEqual([(0, 5, "red"), (6, 9, "gray"), (10, 11, "red")],
                     out.font_attr_segs[1])
    self.assertEqual([(0, 1, "red"), (1, 2, "crimson")], out.font_attr_segs[2])
    self.assertEqual([(0, 7, "blue"), (8, 11, "gray")], out.font_attr_segs[3])
    self.assertEqual([(1, 3, "blue"), (3, 5, "indigo")], out.font_attr_segs[4])

    # Check annotations.
    self.assertFalse(0 in out.annotations)
    self.assertEqual("longer wavelength", out.annotations[1])
    self.assertFalse(2 in out.annotations)
    self.assertEqual("shorter wavelength", out.annotations[3])
    self.assertFalse(4 in out.annotations)

    # Check that the non-row-index field is present in output.
    self.assertEqual("foo", out.annotations["metadata"])

    self.assertEqual(new_line_indices, [0, 1, 3])

  def testWrappingWithMultipleAttrCutoff(self):
    # [FUNCTION DOC]
    # Scenario: replace the fixture with one having a single long segment per line and wrap at 5
    # columns. Expects nine lines, the long segments split across several wrapped rows, annotations
    # on the first wrapped row of their lines, and line indices zero, two, five.
    self._orig_screen_output = debugger_cli_common.RichTextLines(
        ["Folk song:", "Roses are red", "Violets are blue"],
        font_attr_segs={1: [(0, 12, "red")],
                        2: [(1, 16, "blue")]},
        annotations={1: "longer wavelength",
                     2: "shorter wavelength"})

    out, new_line_indices = debugger_cli_common.wrap_rich_text_lines(
        self._orig_screen_output, 5)

    # Check wrapped text.
    self.assertEqual(9, len(out.lines))
    self.assertEqual("Folk ", out.lines[0])
    self.assertEqual("song:", out.lines[1])
    self.assertEqual("Roses", out.lines[2])
    self.assertEqual(" are ", out.lines[3])
    self.assertEqual("red", out.lines[4])
    self.assertEqual("Viole", out.lines[5])
    self.assertEqual("ts ar", out.lines[6])
    self.assertEqual("e blu", out.lines[7])
    self.assertEqual("e", out.lines[8])

    # Check wrapped font_attr_segs.
    self.assertFalse(0 in out.font_attr_segs)
    self.assertFalse(1 in out.font_attr_segs)
    self.assertEqual([(0, 5, "red")], out.font_attr_segs[2])
    self.assertEqual([(0, 5, "red")], out.font_attr_segs[3])
    self.assertEqual([(0, 2, "red")], out.font_attr_segs[4])
    self.assertEqual([(1, 5, "blue")], out.font_attr_segs[5])
    self.assertEqual([(0, 5, "blue")], out.font_attr_segs[6])
    self.assertEqual([(0, 5, "blue")], out.font_attr_segs[7])
    self.assertEqual([(0, 1, "blue")], out.font_attr_segs[8])

    # Check annotations.
    self.assertFalse(0 in out.annotations)
    self.assertFalse(1 in out.annotations)
    self.assertEqual("longer wavelength", out.annotations[2])
    self.assertFalse(3 in out.annotations)
    self.assertFalse(4 in out.annotations)
    self.assertEqual("shorter wavelength", out.annotations[5])
    self.assertFalse(6 in out.annotations)
    self.assertFalse(7 in out.annotations)
    self.assertFalse(8 in out.annotations)

    self.assertEqual(new_line_indices, [0, 2, 5])

  def testWrappingInvalidArguments(self):
    # [FUNCTION DOC]
    # Scenario: wrap with a string instead of a RichTextLines, then with a string column count.
    # Expects a value error each time, with messages about invalid input type and invalid cols type.
    with self.assertRaisesRegex(ValueError,
                                "Invalid type of input screen_output"):
      debugger_cli_common.wrap_rich_text_lines("foo", 12)

    with self.assertRaisesRegex(ValueError, "Invalid type of input cols"):
      debugger_cli_common.wrap_rich_text_lines(
          debugger_cli_common.RichTextLines(["foo", "bar"]), "12")

  def testWrappingEmptyInput(self):
    # [FUNCTION DOC]
    # Scenario: wrap an empty object at 10 columns. Expects empty lines and an empty index list.
    out, new_line_indices = debugger_cli_common.wrap_rich_text_lines(
        debugger_cli_common.RichTextLines([]), 10)

    self.assertEqual([], out.lines)
    self.assertEqual([], new_line_indices)


class SliceRichTextLinesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for RichTextLines.slice. Fixture: a two-line object with segments and annotations,
  # including a non-row annotation key.

  def setUp(self):
    # [FUNCTION DOC]
    # Builds the two-line original object with segments and annotations used by the tests.
    self._original = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]},
        annotations={
            0: "longer wavelength",
            1: "shorter wavelength",
            "foo_metadata": "bar"
        })

  def testSliceBeginning(self):
    # [FUNCTION DOC]
    # Scenario: slice the first line. Expects that line, its segments, its annotation and the
    # non-row metadata annotation preserved, and a line count of one.
    sliced = self._original.slice(0, 1)

    self.assertEqual(["Roses are red"], sliced.lines)
    self.assertEqual({0: [(0, 5, "red")]}, sliced.font_attr_segs)

    # Non-line-number metadata should be preserved.
    self.assertEqual({
        0: "longer wavelength",
        "foo_metadata": "bar"
    }, sliced.annotations)

    self.assertEqual(1, sliced.num_lines())

  def testSliceEnd(self):
    # [FUNCTION DOC]
    # Scenario: slice the last line. Expects the line, with segment and annotation indices
    # re-based from one to zero, the metadata annotation preserved, and a line count of one.
    sliced = self._original.slice(1, 2)

    self.assertEqual(["Violets are blue"], sliced.lines)

    # The line index should have changed from 1 to 0.
    self.assertEqual({0: [(0, 7, "blue")]}, sliced.font_attr_segs)
    self.assertEqual({
        0: "shorter wavelength",
        "foo_metadata": "bar"
    }, sliced.annotations)

    self.assertEqual(1, sliced.num_lines())

  def testAttemptSliceWithNegativeIndex(self):
    # [FUNCTION DOC]
    # Scenario: slice with a negative end index. Expects a value error about a negative index.
    with self.assertRaisesRegex(ValueError, "Encountered negative index"):
      self._original.slice(0, -1)


class TabCompletionRegistryTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for TabCompletionRegistry. Fixture: a registry with one context of two words
  # (print_tensor and pt) holding unsorted tensor-style names, and one context (node_info)
  # holding unsorted node names.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the registry and registers the two contexts, deliberately with unsorted items to
    # check that completions come back sorted.
    self._tc_reg = debugger_cli_common.TabCompletionRegistry()

    # Register the items in an unsorted order deliberately, to test the sorted
    # output from get_completions().
    self._tc_reg.register_tab_comp_context(
        ["print_tensor", "pt"],
        ["node_b:1", "node_b:2", "node_a:1", "node_a:2"])
    self._tc_reg.register_tab_comp_context(["node_info"],
                                           ["node_c", "node_b", "node_a"])

  def testTabCompletion(self):
    # [FUNCTION DOC]
    # Scenario: query completions with various prefixes, through both words of a context, and
    # for an unknown word. Expects sorted matching items with their common prefix, an empty list
    # with empty prefix when nothing matches, and a pair of nulls for an unregistered word.
    # The returned completions should have sorted order.
    self.assertEqual(
        (["node_a:1", "node_a:2", "node_b:1", "node_b:2"], "node_"),
        self._tc_reg.get_completions("print_tensor", "node_"))

    self.assertEqual((["node_a:1", "node_a:2", "node_b:1", "node_b:2"],
                      "node_"), self._tc_reg.get_completions("pt", ""))

    self.assertEqual((["node_a:1", "node_a:2"], "node_a:"),
                     self._tc_reg.get_completions("print_tensor", "node_a"))

    self.assertEqual((["node_a:1"], "node_a:1"),
                     self._tc_reg.get_completions("pt", "node_a:1"))

    self.assertEqual(([], ""),
                     self._tc_reg.get_completions("print_tensor", "node_a:3"))

    self.assertEqual((None, None), self._tc_reg.get_completions("foo", "node_"))

  def testExtendCompletionItems(self):
    # [FUNCTION DOC]
    # Scenario: extend the items of one context word. Expects the new items merged in sorted
    # order for that word and for the other word of the same context, and the other context
    # unaffected.
    self.assertEqual(
        (["node_a:1", "node_a:2", "node_b:1", "node_b:2"], "node_"),
        self._tc_reg.get_completions("print_tensor", "node_"))
    self.assertEqual((["node_a", "node_b", "node_c"], "node_"),
                     self._tc_reg.get_completions("node_info", "node_"))

    self._tc_reg.extend_comp_items("print_tensor", ["node_A:1", "node_A:2"])

    self.assertEqual((["node_A:1", "node_A:2", "node_a:1", "node_a:2",
                       "node_b:1", "node_b:2"], "node_"),
                     self._tc_reg.get_completions("print_tensor", "node_"))

    # Extending the completions for one of the context's context words should
    # have taken effect on other context words of the same context as well.
    self.assertEqual((["node_A:1", "node_A:2", "node_a:1", "node_a:2",
                       "node_b:1", "node_b:2"], "node_"),
                     self._tc_reg.get_completions("pt", "node_"))
    self.assertEqual((["node_a", "node_b", "node_c"], "node_"),
                     self._tc_reg.get_completions("node_info", "node_"))

  def testExtendCompletionItemsNonexistentContext(self):
    # [FUNCTION DOC]
    # Scenario: extend an unregistered context word. Expects a key error saying the word has not
    # been registered.
    with self.assertRaisesRegex(KeyError,
                                "Context word \"foo\" has not been registered"):
      self._tc_reg.extend_comp_items("foo", ["node_A:1", "node_A:2"])

  def testRemoveCompletionItems(self):
    # [FUNCTION DOC]
    # Scenario: remove two items via one word of a context. Expects the remaining items for the
    # context, and the other context unaffected.
    self.assertEqual(
        (["node_a:1", "node_a:2", "node_b:1", "node_b:2"], "node_"),
        self._tc_reg.get_completions("print_tensor", "node_"))
    self.assertEqual((["node_a", "node_b", "node_c"], "node_"),
                     self._tc_reg.get_completions("node_info", "node_"))

    self._tc_reg.remove_comp_items("pt", ["node_a:1", "node_a:2"])

    self.assertEqual((["node_b:1", "node_b:2"], "node_b:"),
                     self._tc_reg.get_completions("print_tensor", "node_"))
    self.assertEqual((["node_a", "node_b", "node_c"], "node_"),
                     self._tc_reg.get_completions("node_info", "node_"))

  def testRemoveCompletionItemsNonexistentContext(self):
    # [FUNCTION DOC]
    # Scenario: remove items from an unregistered context word. Expects a key error saying the
    # word has not been registered.
    with self.assertRaisesRegex(KeyError,
                                "Context word \"foo\" has not been registered"):
      self._tc_reg.remove_comp_items("foo", ["node_a:1", "node_a:2"])

  def testDeregisterContext(self):
    # [FUNCTION DOC]
    # Scenario: deregister one word of a two-word context. Expects its completions to become a
    # pair of nulls while the other word of the context still returns its items.
    self.assertEqual(
        (["node_a:1", "node_a:2", "node_b:1", "node_b:2"], "node_"),
        self._tc_reg.get_completions("print_tensor", "node_"))
    self.assertEqual((["node_a", "node_b", "node_c"], "node_"),
                     self._tc_reg.get_completions("node_info", "node_"))

    self._tc_reg.deregister_context(["print_tensor"])

    self.assertEqual((None, None),
                     self._tc_reg.get_completions("print_tensor", "node_"))

    # The alternative context word should be unaffected.
    self.assertEqual(
        (["node_a:1", "node_a:2", "node_b:1", "node_b:2"], "node_"),
        self._tc_reg.get_completions("pt", "node_"))

  def testDeregisterNonexistentContext(self):
    # [FUNCTION DOC]
    # Scenario: deregister a word twice. Expects the second attempt to raise a key error saying
    # the context word is unregistered.
    self.assertEqual(
        (["node_a:1", "node_a:2", "node_b:1", "node_b:2"], "node_"),
        self._tc_reg.get_completions("print_tensor", "node_"))
    self.assertEqual((["node_a", "node_b", "node_c"], "node_"),
                     self._tc_reg.get_completions("node_info", "node_"))

    self._tc_reg.deregister_context(["print_tensor"])

    with self.assertRaisesRegex(
        KeyError,
        "Cannot deregister unregistered context word \"print_tensor\""):
      self._tc_reg.deregister_context(["print_tensor"])


class CommandHistoryTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for CommandHistory. Fixture: a history with limit three backed by a temporary file.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates a temporary history file (keeping its descriptor) and a history object with limit
    # three that uses it.
    self._fd, self._history_file_path = tempfile.mkstemp()
    self._cmd_hist = debugger_cli_common.CommandHistory(
        limit=3, history_file_path=self._history_file_path)

  def tearDown(self):
    # [FUNCTION DOC]
    # If the temporary history file still exists, closes its descriptor and deletes it.
    if os.path.isfile(self._history_file_path):
      os.close(self._fd)
      os.remove(self._history_file_path)

  def _restoreFileReadWritePermissions(self, file_path):
    # [FUNCTION DOC]
    # HELPER: Permission-reset utility for the history tests.
    # Changes the file at file_path so everyone can read and write it again.
    os.chmod(file_path,
             (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR |
              stat.S_IWGRP | stat.S_IWOTH))

  def testLookUpMostRecent(self):
    # [FUNCTION DOC]
    # Scenario: add commands step by step, including going over the limit of three. After each
    # step expects the most-recent lookups for various counts to return the newest commands in
    # chronological order, with older ones dropped once the limit is exceeded.
    self.assertEqual([], self._cmd_hist.most_recent_n(3))

    self._cmd_hist.add_command("list_tensors")
    self._cmd_hist.add_command("node_info node_a")

    self.assertEqual(["node_info node_a"], self._cmd_hist.most_recent_n(1))
    self.assertEqual(["list_tensors", "node_info node_a"],
                     self._cmd_hist.most_recent_n(2))
    self.assertEqual(["list_tensors", "node_info node_a"],
                     self._cmd_hist.most_recent_n(3))

    self._cmd_hist.add_command("node_info node_b")

    self.assertEqual(["node_info node_b"], self._cmd_hist.most_recent_n(1))
    self.assertEqual(["node_info node_a", "node_info node_b"],
                     self._cmd_hist.most_recent_n(2))
    self.assertEqual(["list_tensors", "node_info node_a", "node_info node_b"],
                     self._cmd_hist.most_recent_n(3))
    self.assertEqual(["list_tensors", "node_info node_a", "node_info node_b"],
                     self._cmd_hist.most_recent_n(4))

    # Go over the limit.
    self._cmd_hist.add_command("node_info node_a")

    self.assertEqual(["node_info node_a"], self._cmd_hist.most_recent_n(1))
    self.assertEqual(["node_info node_b", "node_info node_a"],
                     self._cmd_hist.most_recent_n(2))
    self.assertEqual(
        ["node_info node_a", "node_info node_b", "node_info node_a"],
        self._cmd_hist.most_recent_n(3))
    self.assertEqual(
        ["node_info node_a", "node_info node_b", "node_info node_a"],
        self._cmd_hist.most_recent_n(4))

  def testLookUpPrefix(self):
    # [FUNCTION DOC]
    # Scenario: add three commands, two sharing a prefix, then look up by prefix. Expects the
    # matching commands in order, a count limit honored, and an empty list for an unused prefix.
    self._cmd_hist.add_command("node_info node_b")
    self._cmd_hist.add_command("list_tensors")
    self._cmd_hist.add_command("node_info node_a")

    self.assertEqual(["node_info node_b", "node_info node_a"],
                     self._cmd_hist.lookup_prefix("node_info", 10))

    self.assertEqual(["node_info node_a"], self._cmd_hist.lookup_prefix(
        "node_info", 1))

    self.assertEqual([], self._cmd_hist.lookup_prefix("print_tensor", 10))

  def testAddNonStrCommand(self):
    # [FUNCTION DOC]
    # Scenario: add a list instead of a string as a command. Expects a type error about a
    # non-string entry.
    with self.assertRaisesRegex(
        TypeError, "Attempt to enter non-str entry to command history"):
      self._cmd_hist.add_command(["print_tensor node_a:0"])

  def testRepeatingCommandsDoNotGetLoggedRepeatedly(self):
    # [FUNCTION DOC]
    # Scenario: add the same command twice in a row. Expects it to appear only once in history.
    self._cmd_hist.add_command("help")
    self._cmd_hist.add_command("help")

    self.assertEqual(["help"], self._cmd_hist.most_recent_n(2))

  def testLoadingCommandHistoryFileObeysLimit(self):
    # [FUNCTION DOC]
    # Scenario: add four commands with limit three, then create a second history from the same
    # file. Expects it to load only the newest three, and the file to be rewritten with just
    # those three lines.
    self._cmd_hist.add_command("help 1")
    self._cmd_hist.add_command("help 2")
    self._cmd_hist.add_command("help 3")
    self._cmd_hist.add_command("help 4")

    cmd_hist_2 = debugger_cli_common.CommandHistory(
        limit=3, history_file_path=self._history_file_path)
    self.assertEqual(["help 2", "help 3", "help 4"],
                     cmd_hist_2.most_recent_n(3))

    with open(self._history_file_path, "rt") as f:
      self.assertEqual(
          ["help 2\n", "help 3\n", "help 4\n"], f.readlines())

  def testCommandHistoryHandlesReadingIOErrorGraciously(self):
    # [FUNCTION DOC]
    # Scenario: write a command to the history file, make the file unreadable by everyone, then
    # construct a history on it. Expects construction not to raise. Restores permissions after.
    # (When run as a superuser the permission change may not block reading, so the test then
    # passes trivially.)
    with open(self._history_file_path, "wt") as f:
      f.write("help\n")

    # Change file to not readable by anyone.
    os.chmod(self._history_file_path, 0)

    # The creation of a CommandHistory object should not error out.
    debugger_cli_common.CommandHistory(
        limit=3, history_file_path=self._history_file_path)

    self._restoreFileReadWritePermissions(self._history_file_path)


class MenuNodeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for MenuItem construction and enable/disable state. No shared fixtures.

  def testCommandTypeConstructorSucceeds(self):
    # [FUNCTION DOC]
    # Scenario: construct a menu item with a caption and content. Expects both to be readable
    # back unchanged.
    menu_node = debugger_cli_common.MenuItem("water flower", "water_flower")

    self.assertEqual("water flower", menu_node.caption)
    self.assertEqual("water_flower", menu_node.content)

  def testDisableWorks(self):
    # [FUNCTION DOC]
    # Scenario: disable and re-enable a new item. Expects it enabled initially, disabled after
    # disabling, and enabled again afterwards.
    menu_node = debugger_cli_common.MenuItem("water flower", "water_flower")
    self.assertTrue(menu_node.is_enabled())

    menu_node.disable()
    self.assertFalse(menu_node.is_enabled())
    menu_node.enable()
    self.assertTrue(menu_node.is_enabled())

  def testConstructAsDisabledWorks(self):
    # [FUNCTION DOC]
    # Scenario: construct an item with enabled set to false, then enable it. Expects disabled at
    # first and enabled afterwards.
    menu_node = debugger_cli_common.MenuItem(
        "water flower", "water_flower", enabled=False)
    self.assertFalse(menu_node.is_enabled())

    menu_node.enable()
    self.assertTrue(menu_node.is_enabled())


class MenuTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for Menu: item management and single-line formatting. Fixture: a menu holding two
  # items (water flower, measure wavelength).

  def setUp(self):
    # [FUNCTION DOC]
    # Creates an empty menu (asserting zero items), appends two items and asserts the count is
    # two. Keeps the items as attributes for later comparisons.
    self.menu = debugger_cli_common.Menu()
    self.assertEqual(0, self.menu.num_items())

    self.node1 = debugger_cli_common.MenuItem("water flower", "water_flower")
    self.node2 = debugger_cli_common.MenuItem(
        "measure wavelength", "measure_wavelength")
    self.menu.append(self.node1)
    self.menu.append(self.node2)
    self.assertEqual(2, self.menu.num_items())

  def testFormatAsSingleLineWithStrItemAttrsWorks(self):
    # [FUNCTION DOC]
    # Scenario: format with a prefix, a divider and a single attribute string for enabled items.
    # Expects one line of captions each followed by the divider, a segment per item whose
    # attribute list holds the item and the attribute, and no annotations.
    output = self.menu.format_as_single_line(
        prefix="Menu: ", divider=", ", enabled_item_attrs="underline")
    self.assertEqual(["Menu: water flower, measure wavelength, "], output.lines)
    self.assertEqual((6, 18, [self.node1, "underline"]),
                     output.font_attr_segs[0][0])
    self.assertEqual((20, 38, [self.node2, "underline"]),
                     output.font_attr_segs[0][1])
    self.assertEqual({}, output.annotations)

  def testFormatAsSingleLineWithListItemAttrsWorks(self):
    # [FUNCTION DOC]
    # Scenario: same as the single attribute test but with a list of two attributes. Expects each
    # segment's attribute list to hold the item followed by both attributes.
    output = self.menu.format_as_single_line(
        prefix="Menu: ", divider=", ", enabled_item_attrs=["underline", "bold"])
    self.assertEqual(["Menu: water flower, measure wavelength, "], output.lines)
    self.assertEqual((6, 18, [self.node1, "underline", "bold"]),
                     output.font_attr_segs[0][0])
    self.assertEqual((20, 38, [self.node2, "underline", "bold"]),
                     output.font_attr_segs[0][1])
    self.assertEqual({}, output.annotations)

  def testFormatAsSingleLineWithNoneItemAttrsWorks(self):
    # [FUNCTION DOC]
    # Scenario: format with no item attributes. Expects segments whose attribute lists contain
    # only the item objects, at the correct caption positions.
    output = self.menu.format_as_single_line(prefix="Menu: ", divider=", ")
    self.assertEqual(["Menu: water flower, measure wavelength, "], output.lines)
    self.assertEqual((6, 18, [self.node1]), output.font_attr_segs[0][0])
    self.assertEqual((20, 38, [self.node2]), output.font_attr_segs[0][1])
    self.assertEqual({}, output.annotations)

  def testInsertNode(self):
    # [FUNCTION DOC]
    # Scenario: insert a new item at position one. Expects captions in the new order and the
    # formatted line to include the inserted caption in that position.
    self.assertEqual(["water flower", "measure wavelength"],
                     self.menu.captions())

    node2 = debugger_cli_common.MenuItem("write poem", "write_poem")
    self.menu.insert(1, node2)
    self.assertEqual(["water flower", "write poem", "measure wavelength"],
                     self.menu.captions())

    output = self.menu.format_as_single_line(prefix="Menu: ", divider=", ")
    self.assertEqual(["Menu: water flower, write poem, measure wavelength, "],
                     output.lines)

  def testFormatAsSingleLineWithDisabledNode(self):
    # [FUNCTION DOC]
    # Scenario: append a disabled item and format with a disabled-item attribute. Expects enabled
    # items to get segments with just the item, and the disabled item a segment holding only the
    # disabled attribute, at the right positions. (The local item variable shadows nothing on
    # self; the assertions use the fixture's own items for the enabled ones.)
    node2 = debugger_cli_common.MenuItem(
        "write poem", "write_poem", enabled=False)
    self.menu.append(node2)

    output = self.menu.format_as_single_line(
        prefix="Menu: ", divider=", ", disabled_item_attrs="bold")
    self.assertEqual(["Menu: water flower, measure wavelength, write poem, "],
                     output.lines)
    self.assertEqual((6, 18, [self.node1]), output.font_attr_segs[0][0])
    self.assertEqual((20, 38, [self.node2]), output.font_attr_segs[0][1])
    self.assertEqual((40, 50, ["bold"]), output.font_attr_segs[0][2])


class GetTensorFlowVersionLinesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for get_tensorflow_version_lines. No shared fixtures.

  def testGetVersionWithoutDependencies(self):
    # [FUNCTION DOC]
    # Scenario: request version lines without dependencies. Expects two lines, the first being
    # the TensorFlow version line.
    out = debugger_cli_common.get_tensorflow_version_lines()
    self.assertEqual(2, len(out.lines))
    self.assertEqual("TensorFlow version: %s" % pywrap_tf_session.__version__,
                     out.lines[0])

  def testGetVersionWithDependencies(self):
    # [FUNCTION DOC]
    # Scenario: request version lines with dependencies. Expects the TensorFlow version line and
    # an indented numpy version line among the lines.
    out = debugger_cli_common.get_tensorflow_version_lines(True)
    self.assertIn("TensorFlow version: %s" % pywrap_tf_session.__version__,
                  out.lines)
    self.assertIn("  numpy: %s" % np.__version__, out.lines)


if __name__ == "__main__":
  googletest.main()
