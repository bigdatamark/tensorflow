# [MODULE DOC]
# Defines the base class of the tfdbg command-line user interface. It wires together a command
# handler registry, a tab-completion registry and a user configuration object, and provides the
# built-in "config" command plus command and tab-input parsing helpers. Concrete interfaces
# (for example curses-based ones) subclass it and implement the UI loop. No import-time side
# effects beyond imports.
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
"""Base Class of TensorFlow Debugger (tfdbg) Command-Line Interface."""
import argparse

from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common


class BaseUI(object):
  # [CLASS DOC]
  # Base class for tfdbg user interfaces. Holds a command handler registry, a tab-completion
  # registry, a configuration object and an optional exit callback, and defines class constants
  # for the prompt, exit command words and message prefixes. At construction it registers the
  # exit/help words for top-level tab completion and the "config" command (alias "cfg").
  # Subclasses must override the run method.
  """Base class of tfdbg user interface."""

  CLI_PROMPT = "tfdbg> "
  CLI_EXIT_COMMANDS = ["exit", "quit"]
  ERROR_MESSAGE_PREFIX = "ERROR: "
  INFO_MESSAGE_PREFIX = "INFO: "

  def __init__(self, on_ui_exit=None, config=None):
    # [FUNCTION DOC]
    # Constructor. on_ui_exit is an optional callback to call when the UI exits (stored only).
    # config is an optional configuration object; if missing, a new one is created, which
    # reads and rewrites the user's config file. Creates the command handler and tab-completion
    # registries, registers the top-level completion items (exit words, help command and its
    # aliases), builds an argument parser with "set" (property name and value) and "show"
    # sub-commands, and registers the "config" command with alias "cfg". Returns nothing.
    """Constructor of the base class.

    Args:
      on_ui_exit: (`Callable`) the callback to be called when the UI exits.
      config: An instance of `cli_config.CLIConfig()` carrying user-facing
        configurations.
    """

    self._on_ui_exit = on_ui_exit

    self._command_handler_registry = (
        debugger_cli_common.CommandHandlerRegistry())

    self._tab_completion_registry = debugger_cli_common.TabCompletionRegistry()

    # Create top-level tab-completion context and register the exit and help
    # commands.
    self._tab_completion_registry.register_tab_comp_context(
        [""], self.CLI_EXIT_COMMANDS +
        [debugger_cli_common.CommandHandlerRegistry.HELP_COMMAND] +
        debugger_cli_common.CommandHandlerRegistry.HELP_COMMAND_ALIASES)

    self._config = config or cli_config.CLIConfig()
    self._config_argparser = argparse.ArgumentParser(
        description="config command", usage=argparse.SUPPRESS)
    subparsers = self._config_argparser.add_subparsers()
    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("property_name", type=str)
    set_parser.add_argument("property_value", type=str)
    set_parser = subparsers.add_parser("show")
    self.register_command_handler(
        "config",
        self._config_command_handler,
        self._config_argparser.format_help(),
        prefix_aliases=["cfg"])

  def set_help_intro(self, help_intro):
    # [FUNCTION DOC]
    # Sets the introductory rich-text lines shown at the start of the help command output, by
    # forwarding help_intro to the command registry. Returns nothing.
    """Set an introductory message to the help output of the command registry.

    Args:
      help_intro: (RichTextLines) Rich text lines appended to the beginning of
        the output of the command "help", as introductory information.
    """

    self._command_handler_registry.set_help_intro(help_intro=help_intro)

  def register_command_handler(self,
                               prefix,
                               handler,
                               help_info,
                               prefix_aliases=None):
    # [FUNCTION DOC]
    # Registers a command with the registry (prefix string, handler callback, help text, and an
    # optional list of alias prefixes) and also adds the prefix and its aliases to the
    # top-level tab-completion items. Returns nothing; errors come from the registry.
    """A wrapper around CommandHandlerRegistry.register_command_handler().

    In addition to calling the wrapped register_command_handler() method, this
    method also registers the top-level tab-completion context based on the
    command prefixes and their aliases.

    See the doc string of the wrapped method for more details on the args.

    Args:
      prefix: (str) command prefix.
      handler: (callable) command handler.
      help_info: (str) help information.
      prefix_aliases: (list of str) aliases of the command prefix.
    """

    self._command_handler_registry.register_command_handler(
        prefix, handler, help_info, prefix_aliases=prefix_aliases)

    self._tab_completion_registry.extend_comp_items("", [prefix])
    if prefix_aliases:
      self._tab_completion_registry.extend_comp_items("", prefix_aliases)

  def register_tab_comp_context(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: thin pass-through to the tab-completion registry.
    # Forwards all positional and keyword arguments unchanged to the registry method that
    # registers a tab-completion context. Returns nothing.
    """Wrapper around TabCompletionRegistry.register_tab_comp_context()."""

    self._tab_completion_registry.register_tab_comp_context(*args, **kwargs)

  def run_ui(self,
             init_command=None,
             title=None,
             title_color=None,
             enable_mouse_on_start=True):
    # [FUNCTION DOC]
    # Abstract hook for running the UI loop. Parameters (optional start-up command string,
    # title string, title color name, and whether to enable the mouse at start) are ignored here.
    # Always raises a not-implemented error; subclasses override it and return an exit token
    # of any type (possibly nothing).
    """Run the UI until user- or command- triggered exit.

    Args:
      init_command: (str) Optional command to run on CLI start up.
      title: (str) Optional title to display in the CLI.
      title_color: (str) Optional color of the title, e.g., "yellow".
      enable_mouse_on_start: (bool) Whether the mouse mode is to be enabled on
        start-up.

    Returns:
      An exit token of arbitrary type. Can be None.
    """

    raise NotImplementedError("run_ui() is not implemented in BaseUI")

  def _parse_command(self, command):
    # [FUNCTION DOC]
    # HELPER: command-line parsing for UI subclasses.
    # Strips the command string; if empty returns an empty prefix, an empty argument list and
    # no output path. Otherwise splits it into items, extracts an optional output-file
    # redirection path, and returns the first item as prefix, the remaining items as the
    # argument list, and the output file path (or nothing).
    """Parse a command string into prefix and arguments.

    Args:
      command: (str) Command string to be parsed.

    Returns:
      prefix: (str) The command prefix.
      args: (list of str) The command arguments (i.e., not including the
        prefix).
      output_file_path: (str or None) The path to save the screen output
        to (if any).
    """
    command = command.strip()
    if not command:
      return "", [], None

    command_items = command_parser.parse_command(command)
    command_items, output_file_path = command_parser.extract_output_file_path(
        command_items)

    return command_items[0], command_items[1:], output_file_path

  def _analyze_tab_complete_input(self, text):
    # [FUNCTION DOC]
    # HELPER: input splitting for tab completion in UI subclasses.
    # Takes the raw input text, drops leading whitespace, and splits on single spaces. Returns
    # three strings: the completion context (empty for empty or single-word input, otherwise
    # the first word), the prefix to complete (the last word, empty for empty input), and the
    # text before the last word including its trailing space (empty for zero or one word).
    """Analyze raw input to tab-completer.

    Args:
      text: (str) the full, raw input text to be tab-completed.

    Returns:
      context: (str) the context str. For example,
        If text == "print_tensor softmax", returns "print_tensor".
        If text == "print", returns "".
        If text == "", returns "".
      prefix: (str) the prefix to be tab-completed, from the last word.
        For example, if text == "print_tensor softmax", returns "softmax".
        If text == "print", returns "print".
        If text == "", returns "".
      except_last_word: (str) the input text, except the last word.
        For example, if text == "print_tensor softmax", returns "print_tensor".
        If text == "print_tensor -a softmax", returns "print_tensor -a".
        If text == "print", returns "".
        If text == "", returns "".
    """
    text = text.lstrip()
    if not text:
      # Empty (top-level) context.
      context = ""
      prefix = ""
      except_last_word = ""
    else:
      items = text.split(" ")
      if len(items) == 1:
        # Single word: top-level context.
        context = ""
        prefix = items[0]
        except_last_word = ""
      else:
        # Multiple words.
        context = items[0]
        prefix = items[-1]
        except_last_word = " ".join(items[:-1]) + " "

    return context, prefix, except_last_word

  @property
  def config(self):
    # [FUNCTION DOC]
    # Read-only property returning the configuration object held by this UI.
    """Obtain the CLIConfig of this `BaseUI` instance."""
    return self._config

  def _config_command_handler(self, args, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: handler registered for the "config" command.
    # args is the list of argument strings; screen_info is ignored. Parses the arguments with
    # the config argument parser. If a property name and value were given (the "set" form), it
    # updates the configuration and returns a summary with that property highlighted;
    # otherwise (the "show" form) it returns a plain summary. Invalid property names or values
    # raise errors from the configuration object; bad syntax makes the parser exit.
    """Command handler for the "config" command."""
    del screen_info  # Currently unused.

    parsed = self._config_argparser.parse_args(args)
    if hasattr(parsed, "property_name") and hasattr(parsed, "property_value"):
      # set.
      self._config.set(parsed.property_name, parsed.property_value)
      return self._config.summarize(highlight=parsed.property_name)
    else:
      # show.
      return self._config.summarize()
