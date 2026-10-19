# [MODULE DOC]
# Defines the debugger wrapper session that drives the local command-line interface of tfdbg.
# Exposes one class, which wraps a session so that each run first shows a run-start screen (where
# the user chooses to debug, run through, profile, or run until a tensor filter passes) and, for
# debug or profile runs, an analyzer screen at run end over the dumped tensors or profile data.
# Uses the framework module's base wrapper, the CLI utilities for UI creation, analysis and
# profiling, and the debug data loader. Also defines a module constant giving the prefix of
# temporary dump directories. No import-time side effects beyond imports.
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
"""Debugger Wrapper Session Consisting of a Local Curses-based CLI."""
import argparse
import os
import sys
import tempfile

from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import profile_analyzer_cli
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.lib.io import file_io


_DUMP_ROOT_PREFIX = "tfdbg_"


# TODO(donglin) Remove use_random_config_path after b/137652456 is fixed.
class LocalCLIDebugWrapperSession(framework.BaseDebugWrapperSession):
  # [CLASS DOC]
  # Debug wrapper session whose run-start and run-end callbacks launch an interactive local CLI.
  # State: dump root directory (temporary directory by default), registered tensor filters (by
  # name; the infinity-or-NaN filter is preregistered), argument parsers for the run, run-info and
  # print-feed commands, the active tensor filter and its excluded node names and saved response,
  # remaining run-through count, skip-debug flag, the pending run-start response (absent means the
  # user must be asked), whether the CLI is at run start, the UI type, and an optional CLI
  # configuration. Dumps are deleted after each run.
  """Concrete subclass of BaseDebugWrapperSession implementing a local CLI.

  This class has all the methods that a `session.Session` object has, in order
  to support debugging with minimal code changes. Invoking its `run()` method
  will launch the command-line interface (CLI) of tfdbg.
  """

  def __init__(self,
               sess,
               dump_root=None,
               ui_type="readline",
               thread_name_filter=None,
               config_file_path=False):
    # [FUNCTION DOC]
    # Initializes the base wrapper, decides the dump root, builds argument parsers, and sets the
    # initial state (default tensor filter registered, no active filter, run-through count one).
    # Inputs: sess (session to wrap), dump_root (optional path; user home is expanded; must be
    # missing or an empty directory; if absent a fresh temporary directory with the module prefix
    # is created now), ui_type (string, default readline), thread_name_filter (optional regular
    # expression), config_file_path (optional path; the default of false means no configuration
    # object is created, otherwise a CLI configuration is loaded from it).
    # Raises a value error if dump_root is a file or a non-empty directory. Returns nothing.
    # Side effect: may create a temporary directory.
    """Constructor of LocalCLIDebugWrapperSession.

    Args:
      sess: The TensorFlow `Session` object being wrapped.
      dump_root: (`str`) optional path to the dump root directory. Must be a
        directory that does not exist or an empty directory. If the directory
        does not exist, it will be created by the debugger core during debug
        `run()` calls and removed afterwards. If `None`, the debug dumps will
        be at tfdbg_<random_string> under the system temp directory.
      ui_type: (`str`) requested UI type. Currently supported:
        (readline)
      thread_name_filter: Regular-expression white list for thread name. See
        the doc of `BaseDebugWrapperSession` for details.
      config_file_path: Optional override to the default configuration file
        path, which is at `${HOME}/.tfdbg_config`.

    Raises:
      ValueError: If dump_root is an existing and non-empty directory or if
        dump_root is a file.
    """
    framework.BaseDebugWrapperSession.__init__(
        self, sess, thread_name_filter=thread_name_filter)

    if not dump_root:
      self._dump_root = tempfile.mkdtemp(prefix=_DUMP_ROOT_PREFIX)
    else:
      dump_root = os.path.expanduser(dump_root)
      if os.path.isfile(dump_root):
        raise ValueError("dump_root path points to a file: %s" % dump_root)
      elif os.path.isdir(dump_root) and os.listdir(dump_root):
        raise ValueError("dump_root path points to a non-empty directory: %s" %
                         dump_root)

      self._dump_root = dump_root

    self._initialize_argparsers()

    # Registered tensor filters.
    self._tensor_filters = {}
    # Register frequently-used filter(s).
    self.add_tensor_filter("has_inf_or_nan", debug_data.has_inf_or_nan)

    # Below are the state variables of this wrapper object.
    # _active_tensor_filter: what (if any) tensor filter is in effect. If such
    #   a filter is in effect, this object will call run() method of the
    #   underlying TensorFlow Session object until the filter passes. This is
    #   activated by the "-f" flag of the "run" command.
    # _run_through_times: keeps track of how many times the wrapper needs to
    #   run through without stopping at the run-end CLI. It is activated by the
    #   "-t" option of the "run" command.
    # _skip_debug: keeps track of whether the current run should be executed
    #   without debugging. It is activated by the "-n" option of the "run"
    #   command.
    #
    # _run_start_response: keeps track what OnRunStartResponse the wrapper
    #   should return at the next run-start callback. If this information is
    #   unavailable (i.e., is None), the run-start CLI will be launched to ask
    #   the user. This is the case, e.g., right before the first run starts.
    self._active_tensor_filter = None
    self._active_filter_exclude_node_names = None
    self._active_tensor_filter_run_start_response = None
    self._run_through_times = 1
    self._skip_debug = False
    self._run_start_response = None
    self._is_run_start = True
    self._ui_type = ui_type
    self._config = None
    if config_file_path:
      self._config = cli_config.CLIConfig(config_file_path=config_file_path)

  def _is_disk_usage_reset_each_run(self):
    # [FUNCTION DOC]
    # HELPER: override consulted by the base wrapper when decorating run options.
    # Always returns true because the dumped tensors are deleted after every run in this wrapper.
    # The dumped tensors are all cleaned up after every Session.run
    # in a command-line wrapper.
    return True

  def _initialize_argparsers(self):
    # [FUNCTION DOC]
    # HELPER: called from the constructor.
    # Builds the mapping of command name to argument parser: the run command (options for number
    # of runs to proceed, no-debug, run until a tensor filter passes, node-name exclusion for the
    # filter, regular-expression filters for node names, op types and tensor dtypes, and
    # profiling), the run-info command (no options), and the print-feed command (the standard
    # print-tensor parser). Stores it on the instance and returns nothing.
    self._argparsers = {}
    ap = argparse.ArgumentParser(
        description="Run through, with or without debug tensor watching.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-t",
        "--times",
        dest="times",
        type=int,
        default=1,
        help="How many Session.run() calls to proceed with.")
    ap.add_argument(
        "-n",
        "--no_debug",
        dest="no_debug",
        action="store_true",
        help="Run through without debug tensor watching.")
    ap.add_argument(
        "-f",
        "--till_filter_pass",
        dest="till_filter_pass",
        type=str,
        default="",
        help="Run until a tensor in the graph passes the specified filter.")
    ap.add_argument(
        "-fenn",
        "--filter_exclude_node_names",
        dest="filter_exclude_node_names",
        type=str,
        default="",
        help="When applying the tensor filter, exclude node with names "
        "matching the regular expression. Applicable only if --tensor_filter "
        "or -f is used.")
    ap.add_argument(
        "--node_name_filter",
        dest="node_name_filter",
        type=str,
        default="",
        help="Regular-expression filter for node names to be watched in the "
        "run, e.g., loss, reshape.*")
    ap.add_argument(
        "--op_type_filter",
        dest="op_type_filter",
        type=str,
        default="",
        help="Regular-expression filter for op type to be watched in the run, "
        "e.g., (MatMul|Add), Variable.*")
    ap.add_argument(
        "--tensor_dtype_filter",
        dest="tensor_dtype_filter",
        type=str,
        default="",
        help="Regular-expression filter for tensor dtype to be watched in the "
        "run, e.g., (float32|float64), int.*")
    ap.add_argument(
        "-p",
        "--profile",
        dest="profile",
        action="store_true",
        help="Run and profile TensorFlow graph execution.")
    self._argparsers["run"] = ap

    ap = argparse.ArgumentParser(
        description="Display information about this Session.run() call.",
        usage=argparse.SUPPRESS)
    self._argparsers["run_info"] = ap

    self._argparsers["print_feed"] = command_parser.get_print_tensor_argparser(
        "Print the value of a feed in feed_dict.")

  def add_tensor_filter(self, filter_name, tensor_filter):
    # [FUNCTION DOC]
    # Registers (or replaces) a tensor filter under a name so that it can be used by the run
    # command's filter option and by the analyzer. Inputs: filter_name (string), tensor_filter
    # (callable taking a dump datum and tensor value and returning a boolean). Returns nothing.
    """Add a tensor filter.

    Args:
      filter_name: (`str`) name of the filter.
      tensor_filter: (`callable`) the filter callable. See the doc string of
        `DebugDumpDir.find()` for more details about its signature.
    """

    self._tensor_filters[filter_name] = tensor_filter

  def on_session_init(self, request):
    # [FUNCTION DOC]
    # Session-init callback. Ignores the request and always returns a response telling the
    # framework to proceed.
    """Overrides on-session-init callback.

    Args:
      request: An instance of `OnSessionInitRequest`.

    Returns:
      An instance of `OnSessionInitResponse`.
    """

    return framework.OnSessionInitResponse(
        framework.OnSessionInitAction.PROCEED)

  def on_run_start(self, request):
    # [FUNCTION DOC]
    # Run-start callback deciding how the upcoming run is performed. Input: a run-start request
    # (run call count, fetches, feed dictionary, whether a callable runner is used). Marks the CLI
    # as at run start and updates run-history state. Then: if a tensor filter is active, returns
    # the saved response so runs repeat until the filter passes; exits the process if the user
    # previously asked to exit; for later runs (not first, debugging not skipped) returns a
    # non-debug run response while run-through runs remain, or, when the count reaches zero, the
    # saved response or a debug run response with the dump URL; otherwise (first run, or no saved
    # response) prepares and launches the run-start CLI, stores the user's response, remembers it
    # for the active filter if any, and decrements the run-through count if above one.
    # Returns the run-start response. Side effects: may show the UI or exit the process.
    """Overrides on-run-start callback.

    Args:
      request: An instance of `OnRunStartRequest`.

    Returns:
      An instance of `OnRunStartResponse`.
    """
    self._is_run_start = True
    self._update_run_calls_state(
        request.run_call_count, request.fetches, request.feed_dict,
        is_callable_runner=request.is_callable_runner)

    if self._active_tensor_filter:
      # If we are running until a filter passes, we just need to keep running
      # with the previous `OnRunStartResponse`.
      return self._active_tensor_filter_run_start_response

    self._exit_if_requested_by_user()

    if self._run_call_count > 1 and not self._skip_debug:
      if self._run_through_times > 0:
        # Just run through without debugging.
        return framework.OnRunStartResponse(
            framework.OnRunStartAction.NON_DEBUG_RUN, [])
      elif self._run_through_times == 0:
        # It is the run at which the run-end CLI will be launched: activate
        # debugging.
        return (self._run_start_response or
                framework.OnRunStartResponse(
                    framework.OnRunStartAction.DEBUG_RUN,
                    self._get_run_debug_urls()))

    if self._run_start_response is None:
      self._prep_cli_for_run_start()

      self._run_start_response = self._launch_cli()
      if self._active_tensor_filter:
        self._active_tensor_filter_run_start_response = self._run_start_response
      if self._run_through_times > 1:
        self._run_through_times -= 1

    self._exit_if_requested_by_user()
    return self._run_start_response

  def _exit_if_requested_by_user(self):
    # [FUNCTION DOC]
    # HELPER: used by the run-start callback.
    # If the stored run-start response is the explicit user exit marker, prints a note to the
    # standard error stream and terminates the process with status one. Otherwise does nothing.
    if self._run_start_response == debugger_cli_common.EXPLICIT_USER_EXIT:
      # Explicit user "exit" command leads to sys.exit(1).
      print(
          "Note: user exited from debugger CLI: Calling sys.exit(1).",
          file=sys.stderr)
      sys.exit(1)

  def _prep_cli_for_run_start(self):
    # [FUNCTION DOC]
    # HELPER: used by the run-start callback.
    # Creates the UI of the configured type and sets its help intro: on the first run the tfdbg
    # logo and TensorFlow version lines, then an upcoming-run heading and the run information.
    # Sets the screen title (run-start plus the run description), the initial command (run info)
    # and the title color. Does not launch the UI. Returns nothing.
    """Prepare (but not launch) the CLI for run-start."""
    self._run_cli = ui_factory.get_ui(self._ui_type, config=self._config)

    help_intro = debugger_cli_common.RichTextLines([])
    if self._run_call_count == 1:
      # Show logo at the onset of the first run.
      help_intro.extend(cli_shared.get_tfdbg_logo())
      help_intro.extend(debugger_cli_common.get_tensorflow_version_lines())
    help_intro.extend(debugger_cli_common.RichTextLines("Upcoming run:"))
    help_intro.extend(self._run_info)

    self._run_cli.set_help_intro(help_intro)

    # Create initial screen output detailing the run.
    self._title = "run-start: " + self._run_description
    self._init_command = "run_info"
    self._title_color = "blue_on_white"

  def on_run_end(self, request):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring names session-init request/response types; the actual types are
    # the run-end request and response.
    # Run-end callback. Marks the CLI as not at run start. For a debug run: picks partition graphs
    # from the run metadata or the request's client graph; re-raises the run's TensorFlow error if
    # one occurred and the dump directory is missing; loads the dump and attaches the Python
    # graph; if a tensor filter is active, deletes the dump and returns when no dumped tensor
    # passes it, else clears the active filter and notes it; prepares and launches the analyzer
    # CLI, stores the user's response as the next run-start response, and deletes the dump. For a
    # profile run: prepares and launches the profiler CLI and stores the response. For other runs:
    # clears the stored response. Always returns an empty run-end response.
    """Overrides on-run-end callback.

    Actions taken:
      1) Load the debug dump.
      2) Bring up the Analyzer CLI.

    Args:
      request: An instance of OnSessionInitRequest.

    Returns:
      An instance of OnSessionInitResponse.
    """

    self._is_run_start = False
    if request.performed_action == framework.OnRunStartAction.DEBUG_RUN:
      partition_graphs = None
      if request.run_metadata and request.run_metadata.partition_graphs:
        partition_graphs = request.run_metadata.partition_graphs
      elif request.client_graph_def:
        partition_graphs = [request.client_graph_def]

      if request.tf_error and not os.path.isdir(self._dump_root):
        # It is possible that the dump root may not exist due to errors that
        # have occurred prior to graph execution (e.g., invalid device
        # assignments), in which case we will just raise the exception as the
        # unwrapped Session does.
        raise request.tf_error

      debug_dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=partition_graphs)
      debug_dump.set_python_graph(self._sess.graph)

      passed_filter = None
      passed_filter_exclude_node_names = None
      if self._active_tensor_filter:
        if not debug_dump.find(
            self._tensor_filters[self._active_tensor_filter], first_n=1,
            exclude_node_names=self._active_filter_exclude_node_names):
          # No dumped tensor passes the filter in this run. Clean up the dump
          # directory and move on.
          self._remove_dump_root()
          return framework.OnRunEndResponse()
        else:
          # Some dumped tensor(s) from this run passed the filter.
          passed_filter = self._active_tensor_filter
          passed_filter_exclude_node_names = (
              self._active_filter_exclude_node_names)
          self._active_tensor_filter = None
          self._active_filter_exclude_node_names = None

      self._prep_debug_cli_for_run_end(
          debug_dump, request.tf_error, passed_filter,
          passed_filter_exclude_node_names)

      self._run_start_response = self._launch_cli()

      # Clean up the dump generated by this run.
      self._remove_dump_root()
    elif request.performed_action == framework.OnRunStartAction.PROFILE_RUN:
      self._prep_profile_cli_for_run_end(self._sess.graph, request.run_metadata)
      self._run_start_response = self._launch_cli()
    else:
      # No debug information to show following a non-debug run() call.
      self._run_start_response = None

    # Return placeholder response that currently holds no additional
    # information.
    return framework.OnRunEndResponse()

  def _remove_dump_root(self):
    # [FUNCTION DOC]
    # HELPER: cleanup used after runs and as the UI exit callback.
    # Recursively deletes the dump root directory if it exists. Returns nothing.
    if os.path.isdir(self._dump_root):
      file_io.delete_recursively(self._dump_root)

  def _prep_debug_cli_for_run_end(self,
                                  debug_dump,
                                  tf_error,
                                  passed_filter,
                                  passed_filter_exclude_node_names):
    # [FUNCTION DOC]
    # HELPER: used by the run-end callback.
    # Chooses the initial command and title color: after an error, the help command with a red
    # title and an error intro; otherwise the list-tensors command with a black title, and, if a
    # filter just passed, the same command restricted to that filter (plus the excluded node name
    # expression if any) with a red title. Creates the analyzer UI over the dump with the tensor
    # filters, removing the dump when the UI exits, registers tab completion for tensor names
    # and node names, and sets the title (run-end plus run description) and optional help intro.
    # Inputs: the dump directory object, the error (or nothing), the passed filter name (or
    # nothing) and its excluded node-name expression (or nothing). Does not launch. Returns nothing.
    """Prepare (but not launch) CLI for run-end, with debug dump from the run.

    Args:
      debug_dump: (debug_data.DebugDumpDir) The debug dump directory from this
        run.
      tf_error: (None or OpError) OpError that happened during the run() call
        (if any).
      passed_filter: (None or str) Name of the tensor filter that just passed
        and caused the preparation of this run-end CLI (if any).
      passed_filter_exclude_node_names: (None or str) Regular expression used
        with the tensor filter to exclude ops with names matching the regular
        expression.
    """

    if tf_error:
      help_intro = cli_shared.get_error_intro(tf_error)

      self._init_command = "help"
      self._title_color = "red_on_white"
    else:
      help_intro = None
      self._init_command = "lt"

      self._title_color = "black_on_white"
      if passed_filter is not None:
        # Some dumped tensor(s) from this run passed the filter.
        self._init_command = "lt -f %s" % passed_filter
        if passed_filter_exclude_node_names:
          self._init_command += (" --filter_exclude_node_names %s" %
                                 passed_filter_exclude_node_names)
        self._title_color = "red_on_white"

    self._run_cli = analyzer_cli.create_analyzer_ui(
        debug_dump,
        self._tensor_filters,
        ui_type=self._ui_type,
        on_ui_exit=self._remove_dump_root,
        config=self._config)

    # Get names of all dumped tensors.
    dumped_tensor_names = []
    for datum in debug_dump.dumped_tensor_data:
      dumped_tensor_names.append("%s:%d" %
                                 (datum.node_name, datum.output_slot))

    # Tab completions for command "print_tensors".
    self._run_cli.register_tab_comp_context(["print_tensor", "pt"],
                                            dumped_tensor_names)

    # Tab completion for commands "node_info", "list_inputs" and
    # "list_outputs". The list comprehension is used below because nodes()
    # output can be unicodes and they need to be converted to strs.
    self._run_cli.register_tab_comp_context(
        ["node_info", "ni", "list_inputs", "li", "list_outputs", "lo"],
        [str(node_name) for node_name in debug_dump.nodes()])
    # TODO(cais): Reduce API surface area for aliases vis-a-vis tab
    #    completion contexts and registered command handlers.

    self._title = "run-end: " + self._run_description

    if help_intro:
      self._run_cli.set_help_intro(help_intro)

  def _prep_profile_cli_for_run_end(self, py_graph, run_metadata):
    # [FUNCTION DOC]
    # HELPER: used by the run-end callback for profile runs.
    # Sets the initial command to the profile listing command, creates the profiler UI from the
    # Python graph and run metadata (reusing the current UI's configuration), and sets the title
    # to a run-end profiler-mode title with the run description. Does not launch. Returns nothing.
    self._init_command = "lp"
    self._run_cli = profile_analyzer_cli.create_profiler_ui(
        py_graph, run_metadata, ui_type=self._ui_type,
        config=self._run_cli.config)
    self._title = "run-end (profiler mode): " + self._run_description

  def _launch_cli(self):
    # [FUNCTION DOC]
    # HELPER: shows the prepared UI.
    # Registers the run, run-info and print-feed command handlers on the current UI, runs it with
    # the stored initial command, title and title color, and returns the response (exit token)
    # produced when the user leaves the UI. Blocks until the user exits.
    """Launch the interactive command-line interface.

    Returns:
      The OnRunStartResponse specified by the user using the "run" command.
    """

    self._register_this_run_info(self._run_cli)
    response = self._run_cli.run_ui(
        init_command=self._init_command,
        title=self._title,
        title_color=self._title_color)

    return response

  def _run_info_handler(self, args, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: command handler for the run-info command.
    # Inputs: args (unused command arguments), screen_info (optional, unused). Builds rich text
    # lines: on the first run the logo and version lines, followed by the run information. When
    # not at run start and the main menu is present, inserts a list-tensors item at the front of
    # the menu if missing. Returns the rich text lines.
    output = debugger_cli_common.RichTextLines([])

    if self._run_call_count == 1:
      output.extend(cli_shared.get_tfdbg_logo())
      output.extend(debugger_cli_common.get_tensorflow_version_lines())
    output.extend(self._run_info)

    if (not self._is_run_start and
        debugger_cli_common.MAIN_MENU_KEY in output.annotations):
      menu = output.annotations[debugger_cli_common.MAIN_MENU_KEY]
      if "list_tensors" not in menu.captions():
        menu.insert(
            0, debugger_cli_common.MenuItem("list_tensors", "list_tensors"))

    return output

  def _print_feed_handler(self, args, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: command handler for the print-feed command.
    # Inputs: args (command-line argument tokens: a tensor name with optional slicing plus
    # printing options such as print-all, ranges and numeric summary), screen_info (optional
    # screen details used to derive array print options). Looks up the feed whose graph element
    # name matches the requested name in the current run's feed dictionary and returns its
    # formatted value, labelled as a feed. Returns an error text if the feed dictionary is empty
    # or has no matching key.
    np_printoptions = cli_shared.numpy_printoptions_from_screen_info(
        screen_info)

    if not self._feed_dict:
      return cli_shared.error(
          "The feed_dict of the current run is None or empty.")

    parsed = self._argparsers["print_feed"].parse_args(args)
    tensor_name, tensor_slicing = (
        command_parser.parse_tensor_name_with_slicing(parsed.tensor_name))

    feed_key = None
    feed_value = None
    for key in self._feed_dict:
      key_name = common.get_graph_element_name(key)
      if key_name == tensor_name:
        feed_key = key_name
        feed_value = self._feed_dict[key]
        break

    if feed_key is None:
      return cli_shared.error(
          "The feed_dict of the current run does not contain the key %s" %
          tensor_name)
    else:
      return cli_shared.format_tensor(
          feed_value,
          feed_key + " (feed)",
          np_printoptions,
          print_all=parsed.print_all,
          tensor_slicing=tensor_slicing,
          highlight_options=cli_shared.parse_ranges_highlight(parsed.ranges),
          include_numeric_summary=parsed.numeric_summary)

  def _run_handler(self, args, screen_info=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error text mentions a flag spelled feon, but the actual option is spelled
    # fenn. Also, the local action is reassigned to a debug run in filter mode after the response
    # was already built, so that reassignment has no effect on the response.
    # Handler for the run command at run start. Input: args (tokens parsed with the run parser;
    # screen_info unused). Empty node-name, op-type and dtype filters become nothing. Raises a
    # value error if the filter-exclusion option is used without the filter option. With the
    # profile option, exits the CLI with a profile-run response. Otherwise stores skip-debug and
    # run-through times, builds a non-debug response (when times exceeds one or no-debug is set) or
    # a debug response with the dump URL, carrying the regular-expression allowlists. With the
    # filter option: if the named filter is registered, activates it with its exclusion
    # expression and saved response; otherwise returns an error text. Finally always exits the
    # CLI by raising a command-line exit carrying the response.
    """Command handler for "run" command during on-run-start."""

    del screen_info  # Currently unused.

    parsed = self._argparsers["run"].parse_args(args)
    parsed.node_name_filter = parsed.node_name_filter or None
    parsed.op_type_filter = parsed.op_type_filter or None
    parsed.tensor_dtype_filter = parsed.tensor_dtype_filter or None

    if parsed.filter_exclude_node_names and not parsed.till_filter_pass:
      raise ValueError(
          "The --filter_exclude_node_names (or -feon) flag is valid only if "
          "the --till_filter_pass (or -f) flag is used.")

    if parsed.profile:
      raise debugger_cli_common.CommandLineExit(
          exit_token=framework.OnRunStartResponse(
              framework.OnRunStartAction.PROFILE_RUN, []))

    self._skip_debug = parsed.no_debug
    self._run_through_times = parsed.times

    if parsed.times > 1 or parsed.no_debug:
      # If requested -t times > 1, the very next run will be a non-debug run.
      action = framework.OnRunStartAction.NON_DEBUG_RUN
      debug_urls = []
    else:
      action = framework.OnRunStartAction.DEBUG_RUN
      debug_urls = self._get_run_debug_urls()
    run_start_response = framework.OnRunStartResponse(
        action,
        debug_urls,
        node_name_regex_allowlist=parsed.node_name_filter,
        op_type_regex_allowlist=parsed.op_type_filter,
        tensor_dtype_regex_allowlist=parsed.tensor_dtype_filter)

    if parsed.till_filter_pass:
      # For the run-till-filter-pass (run -f) mode, use the DEBUG_RUN
      # option to access the intermediate tensors, and set the corresponding
      # state flag of the class itself to True.
      if parsed.till_filter_pass in self._tensor_filters:
        action = framework.OnRunStartAction.DEBUG_RUN
        self._active_tensor_filter = parsed.till_filter_pass
        self._active_filter_exclude_node_names = (
            parsed.filter_exclude_node_names)
        self._active_tensor_filter_run_start_response = run_start_response
      else:
        # Handle invalid filter name.
        return debugger_cli_common.RichTextLines(
            ["ERROR: tensor filter \"%s\" does not exist." %
             parsed.till_filter_pass])

    # Raise CommandLineExit exception to cause the CLI to exit.
    raise debugger_cli_common.CommandLineExit(exit_token=run_start_response)

  def _register_this_run_info(self, curses_cli):
    # [FUNCTION DOC]
    # HELPER: called by the CLI launcher.
    # Registers on the given UI the run command (alias r), run-info (alias ri) and print-feed
    # (alias pf) handlers with their parser help text. Registers tab completion of tensor filter
    # names for the run command when filters exist, and of feed names for print-feed when the
    # feed dictionary is a mapping. Returns nothing.
    curses_cli.register_command_handler(
        "run",
        self._run_handler,
        self._argparsers["run"].format_help(),
        prefix_aliases=["r"])
    curses_cli.register_command_handler(
        "run_info",
        self._run_info_handler,
        self._argparsers["run_info"].format_help(),
        prefix_aliases=["ri"])
    curses_cli.register_command_handler(
        "print_feed",
        self._print_feed_handler,
        self._argparsers["print_feed"].format_help(),
        prefix_aliases=["pf"])

    if self._tensor_filters:
      # Register tab completion for the filter names.
      curses_cli.register_tab_comp_context(["run", "r"],
                                           list(self._tensor_filters.keys()))
    if self._feed_dict and hasattr(self._feed_dict, "keys"):
      # Register tab completion for feed_dict keys.
      feed_keys = [common.get_graph_element_name(key)
                   for key in self._feed_dict.keys()]
      curses_cli.register_tab_comp_context(["print_feed", "pf"], feed_keys)

  def _get_run_debug_urls(self):
    # [FUNCTION DOC]
    # HELPER: supplies debug URLs for debug runs.
    # Returns a one-element list holding the file URL of the dump root directory.
    """Get the debug_urls value for the current run() call.

    Returns:
      debug_urls: (list of str) Debug URLs for the current run() call.
        Currently, the list consists of only one URL that is a file:// URL.
    """

    return ["file://" + self._dump_root]

  def _update_run_calls_state(self,
                              run_call_count,
                              fetches,
                              feed_dict,
                              is_callable_runner=False):
    # [FUNCTION DOC]
    # HELPER: called at each run start.
    # Records the run call count and feed dictionary, computes a short description of the run and
    # the run-info text (which also lists the registered tensor filters), and decrements the
    # remaining run-through count. Inputs: run_call_count (integer), fetches, feed_dict (mapping
    # or nothing), is_callable_runner (boolean, default false). Returns nothing.
    """Update the internal state with regard to run() call history.

    Args:
      run_call_count: (int) Number of run() calls that have occurred.
      fetches: a node/tensor or a list of node/tensor that are the fetches of
        the run() call. This is the same as the fetches argument to the run()
        call.
      feed_dict: None of a dict. This is the feed_dict argument to the run()
        call.
      is_callable_runner: (bool) whether a runner returned by
        Session.make_callable is being run.
    """

    self._run_call_count = run_call_count
    self._feed_dict = feed_dict
    self._run_description = cli_shared.get_run_short_description(
        run_call_count,
        fetches,
        feed_dict,
        is_callable_runner=is_callable_runner)
    self._run_through_times -= 1

    self._run_info = cli_shared.get_run_start_intro(
        run_call_count,
        fetches,
        feed_dict,
        self._tensor_filters,
        is_callable_runner=is_callable_runner)
