# [MODULE DOC]
# Line-based (readline) command-line interface of the TensorFlow debugger. Defines ReadlineUI, a
# subclass of the common base UI that reads commands from standard input with emacs-style editing
# and tab completion, dispatches them to registered handlers and prints the resulting text lines
# to standard output. Importing the module loads the system readline library; no other import-time
# side effects. Created normally via the UI factory.
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
"""Readline-Based Command-Line Interface of TensorFlow Debugger (tfdbg)."""
import readline

from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common


class ReadlineUI(base_ui.BaseUI):
  # [CLASS DOC]
  # Plain-terminal debugger UI built on the base UI class. State is inherited (command handler
  # registry, tab completion registry, config, exit callback) plus an input function reading from
  # the keyboard. Overrides running the UI, command dispatching and output display; each command
  # line is parsed, dispatched and its output lines printed. Instances are normally made by the
  # UI factory.
  """Readline-based Command-line UI."""

  def __init__(self, on_ui_exit=None, config=None):
    # [FUNCTION DOC]
    # Constructor. on_ui_exit is an optional callback invoked when the UI loop ends; config is an
    # optional command-line configuration object. Initializes the base class with both and then
    # sets up readline input handling (see the input initializer). Returns nothing.
    base_ui.BaseUI.__init__(self, on_ui_exit=on_ui_exit, config=config)
    self._init_input()

  def _init_input(self):
    # [FUNCTION DOC]
    # HELPER: one-time terminal input setup for the constructor.
    # Configures the global readline library: emacs editing mode, only the newline as word
    # delimiter (so the completer receives the whole line), this object's completion method as the
    # completer, and the tab key bound to completion. Stores the built-in line input function as
    # the input source. Side effects: changes process-wide readline settings. Returns nothing.
    readline.parse_and_bind("set editing-mode emacs")

    # Disable default readline delimiter in order to receive the full text
    # (not just the last word) in the completer.
    readline.set_completer_delims("\n")
    readline.set_completer(self._readline_complete)
    readline.parse_and_bind("tab: complete")

    self._input = input

  def _readline_complete(self, text, state):
    # [FUNCTION DOC]
    # HELPER: completion callback registered with readline.
    # text is the whole current input line; state is the zero-based index of the candidate wanted.
    # Splits the line into command context, prefix to complete and the preceding words, asks the
    # tab completion registry for candidates, prepends the preceding words to each and returns the
    # candidate at position state. Raises an index error when state is past the last candidate
    # (readline treats that as the end of the list; note it does not return nothing as the
    # readline protocol usually expects).
    context, prefix, except_last_word = self._analyze_tab_complete_input(text)
    candidates, _ = self._tab_completion_registry.get_completions(context,
                                                                  prefix)
    candidates = [(except_last_word + candidate) for candidate in candidates]
    return candidates[state]

  def run_ui(self,
             init_command=None,
             title=None,
             title_color=None,
             enable_mouse_on_start=True):
    # [FUNCTION DOC]
    # KNOWN ISSUE: title_color and enable_mouse_on_start are accepted for interface compatibility
    # but ignored here; the title is printed even if it is nothing.
    # Runs the interactive loop. init_command is an optional command string executed first; title
    # is printed once at the start. Then loops reading and dispatching user commands until a
    # command yields an exit token, calls the exit callback if one was given, and returns the exit
    # token. Side effects: standard output, reading standard input, command handlers' effects.
    """Run the CLI: See the doc of base_ui.BaseUI.run_ui for more details."""

    print(title)

    if init_command is not None:
      self._dispatch_command(init_command)

    exit_token = self._ui_loop()

    if self._on_ui_exit:
      self._on_ui_exit()

    return exit_token

  def _ui_loop(self):
    # [FUNCTION DOC]
    # HELPER: main loop for run_ui. Repeatedly reads a user command and dispatches it until a
    # dispatch returns a non-null exit token, which is then returned. Does not return otherwise.
    while True:
      command = self._get_user_command()

      exit_token = self._dispatch_command(command)
      if exit_token is not None:
        return exit_token

  def _get_user_command(self):
    # [FUNCTION DOC]
    # HELPER: input step for the UI loop (overridden by test doubles). Prints a blank line, shows
    # the prompt, reads a line and returns it with surrounding whitespace removed. May raise an
    # end-of-input error from the input function.
    print("")
    return self._input(self.CLI_PROMPT).strip()

  def _dispatch_command(self, command):
    # [FUNCTION DOC]
    # HELPER: executes one command line for the UI loop and run_ui.
    # command is the text typed by the user. A command in the configured exit list returns the
    # explicit-user-exit token. Otherwise the line is parsed into prefix, arguments and optional
    # output file path; a syntax error is printed and nothing is returned. A registered prefix is
    # dispatched to its handler without screen info (a command-line-exit exception from the handler
    # returns its exit token); an unregistered prefix yields an error message output. The output
    # is displayed and, if a file path was given, written there, printing a success or failure
    # message (any write error is swallowed). Returns nothing for normal commands, so the loop
    # continues.
    """Dispatch user command.

    Args:
      command: (str) Command to dispatch.

    Returns:
      An exit token object. None value means that the UI loop should not exit.
      A non-None value means the UI loop should exit.
    """

    if command in self.CLI_EXIT_COMMANDS:
      # Explicit user command-triggered exit: EXPLICIT_USER_EXIT as the exit
      # token.
      return debugger_cli_common.EXPLICIT_USER_EXIT

    try:
      prefix, args, output_file_path = self._parse_command(command)
    except SyntaxError as e:
      print(str(e))
      return

    if self._command_handler_registry.is_registered(prefix):
      try:
        screen_output = self._command_handler_registry.dispatch_command(
            prefix, args, screen_info=None)
      except debugger_cli_common.CommandLineExit as e:
        return e.exit_token
    else:
      screen_output = debugger_cli_common.RichTextLines([
          self.ERROR_MESSAGE_PREFIX + "Invalid command prefix \"%s\"" % prefix
      ])

    self._display_output(screen_output)
    if output_file_path:
      try:
        screen_output.write_to_file(output_file_path)
        print("Wrote output to %s" % output_file_path)
      except Exception:  # pylint: disable=broad-except
        print("Failed to write output to %s" % output_file_path)

  def _display_output(self, screen_output):
    # [FUNCTION DOC]
    # HELPER: output step for command dispatch (overridden by test doubles). Prints each text
    # line of the given rich text lines object to standard output; formatting attributes are
    # ignored. Returns nothing.
    for line in screen_output.lines:
      print(line)
