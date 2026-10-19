# [MODULE DOC]
# Session run hooks that expose the tfdbg debugger to monitored sessions. Provides a hook that
# drives the local command-line debugger, a hook that dumps debug data to the filesystem, a hook
# that streams debug data to gRPC debug servers, and a TensorBoard-oriented variant of the gRPC
# hook. Each hook lazily builds the matching debug wrapper session from the wrapper modules on
# its first run and translates session run hook callbacks into the wrapper's start/end-of-run
# machinery and run options. No import-time side effects other than imports.
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
"""tfdbg CLI as SessionRunHook."""

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook


class LocalCLIDebugHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session run hook that runs the local command-line debugger around each monitored-session run.
  # State: UI type, dump root, thread filter and config path settings, a lazily created local CLI
  # wrapper session, tensor filters registered before that wrapper exists, and the action chosen
  # for the current run. Created with settings and passed to a monitored session as a hook.
  """Command-line-interface debugger hook.

  Can be used as a hook for `tf.compat.v1.train.MonitoredSession`.
  """

  def __init__(self,
               ui_type="readline",
               dump_root=None,
               thread_name_filter=None,
               config_file_path=None):
    # [FUNCTION DOC]
    # Stores the settings only; the wrapper session is created later on the first run.
    # Inputs: ui_type (string, default readline), dump_root (optional directory path, must not
    # exist or be empty), thread_name_filter (optional regular expression for threads),
    # config_file_path (optional override of the debugger configuration file path).
    # Initializes an empty wrapper reference and an empty mapping of pending tensor filters.
    # Returns nothing.
    """Create a local debugger command-line interface (CLI) hook.

    Args:
      ui_type: (`str`) requested user-interface type. Currently supported:
        (readline).
      dump_root: (`str`) optional path to the dump root directory. Must be a
        directory that does not exist or an empty directory. If the directory
        does not exist, it will be created by the debugger core during debug
        `run()` calls and removed afterwards.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      config_file_path: Optional override to the default configuration file
        path, which is at `${HOME}/.tfdbg_config`.
    """

    self._ui_type = ui_type
    self._dump_root = dump_root
    self._thread_name_filter = thread_name_filter
    self._session_wrapper = None
    self._pending_tensor_filters = {}
    self._config_file_path = config_file_path

  def add_tensor_filter(self, filter_name, tensor_filter):
    # [FUNCTION DOC]
    # Registers a named tensor filter. If the wrapper session already exists the filter is added
    # to it directly; otherwise it is remembered in a pending mapping (by name) and registered
    # when the wrapper is created on the first run. Inputs: filter_name (string) and
    # tensor_filter (callable). Returns nothing.
    """Add a tensor filter.

    See doc of `LocalCLIDebugWrapperSession.add_tensor_filter()` for details.
    Override default behavior to accommodate the possibility of this method
    being
    called prior to the initialization of the underlying
    `LocalCLIDebugWrapperSession` object.

    Args:
      filter_name: See doc of `LocalCLIDebugWrapperSession.add_tensor_filter()`
        for details.
      tensor_filter: See doc of
        `LocalCLIDebugWrapperSession.add_tensor_filter()` for details.
    """

    if self._session_wrapper:
      self._session_wrapper.add_tensor_filter(filter_name, tensor_filter)
    else:
      self._pending_tensor_filters[filter_name] = tensor_filter

  def begin(self):
    # [FUNCTION DOC]
    # Session run hook callback invoked when the session is created; intentionally does nothing.
    pass

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # Hook callback invoked before each session run. On the first call creates the local CLI
    # wrapper session around the run context's session using the stored settings and registers
    # all pending tensor filters. Then increments the wrapper's run counter, builds a start-of-run
    # request from the original fetches and feeds, and asks the wrapper how to proceed; the chosen
    # action is remembered for the matching after-run call.
    # Returns run arguments carrying fresh run options: for a debug action, the options are
    # decorated with debug URLs, debug ops, regular-expression allowlists and the tolerance flag
    # from the response; for a profile action they are decorated for profiling; otherwise they are
    # left plain. Side effects: may create the wrapper (and, via it, the dump directory or UI).
    if not self._session_wrapper:
      self._session_wrapper = local_cli_wrapper.LocalCLIDebugWrapperSession(
          run_context.session,
          ui_type=self._ui_type,
          dump_root=self._dump_root,
          thread_name_filter=self._thread_name_filter,
          config_file_path=self._config_file_path)

      # Actually register tensor filters registered prior to the construction
      # of the underlying LocalCLIDebugWrapperSession object.
      for filter_name in self._pending_tensor_filters:
        self._session_wrapper.add_tensor_filter(
            filter_name, self._pending_tensor_filters[filter_name])

    # Increment run call counter.
    self._session_wrapper.increment_run_call_count()

    # Adapt run_context to an instance of OnRunStartRequest for invoking
    # superclass on_run_start().
    on_run_start_request = framework.OnRunStartRequest(
        run_context.original_args.fetches, run_context.original_args.feed_dict,
        None, None, self._session_wrapper.run_call_count)

    on_run_start_response = self._session_wrapper.on_run_start(
        on_run_start_request)
    self._performed_action = on_run_start_response.action

    run_args = session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=config_pb2.RunOptions())
    if self._performed_action == framework.OnRunStartAction.DEBUG_RUN:
      # pylint: disable=protected-access
      self._session_wrapper._decorate_run_options_for_debug(
          run_args.options,
          on_run_start_response.debug_urls,
          debug_ops=on_run_start_response.debug_ops,
          node_name_regex_allowlist=(
              on_run_start_response.node_name_regex_allowlist),
          op_type_regex_allowlist=(
              on_run_start_response.op_type_regex_allowlist),
          tensor_dtype_regex_allowlist=(
              on_run_start_response.tensor_dtype_regex_allowlist),
          tolerate_debug_op_creation_failures=(
              on_run_start_response.tolerate_debug_op_creation_failures))
      # pylint: enable=protected-access
    elif self._performed_action == framework.OnRunStartAction.PROFILE_RUN:
      # pylint: disable=protected-access
      self._session_wrapper._decorate_run_options_for_profile(run_args.options)
      # pylint: enable=protected-access

    return run_args

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Hook callback invoked after each run. Builds an end-of-run request from the action performed
    # in the preceding before-run call and the run metadata, and passes it to the wrapper's
    # end-of-run handler (which may show the debugger UI). The run context is unused. Returns
    # nothing.
    # Adapt run_context and run_values to OnRunEndRequest and invoke superclass
    # on_run_end()
    on_run_end_request = framework.OnRunEndRequest(self._performed_action,
                                                   run_values.run_metadata)
    self._session_wrapper.on_run_end(on_run_end_request)


class DumpingDebugHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session run hook that makes every monitored-session run dump debug data to the filesystem
  # through a lazily created dumping wrapper session. State: session root directory, optional
  # watch callback, thread filter, and the wrapper session (initially absent).
  """A debugger hook that dumps debug data to filesystem.

  Can be used as a hook for `tf.compat.v1.train.MonitoredSession`.
  """

  def __init__(self,
               session_root,
               watch_fn=None,
               thread_name_filter=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it creates a CLI hook, but this constructs the dumping hook.
    # Stores settings only. Inputs: session_root (directory under which dumps are written),
    # watch_fn (optional callback choosing watch options per run), thread_name_filter (optional
    # regular expression for threads). The wrapper session starts out absent. Returns nothing.
    """Create a local debugger command-line interface (CLI) hook.

    Args:
      session_root: See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
      watch_fn: See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
    """

    self._session_root = session_root
    self._watch_fn = watch_fn
    self._thread_name_filter = thread_name_filter
    self._session_wrapper = None

  def begin(self):
    # [FUNCTION DOC]
    # Session run hook callback invoked when the session is created; intentionally does nothing.
    pass

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # Hook callback invoked before each run. On the first call creates the dumping wrapper session
    # around the run context's session (and flags that disk byte usage should be reset). Increments
    # the wrapper's run counter, asks it for the debug URLs and watch options for the original
    # fetches and feeds, then marks the graph for watching in fresh run options using those debug
    # URLs, debug ops, regular-expression allowlists, tolerance flag and the reset flag.
    # Returns run arguments carrying those run options.
    reset_disk_byte_usage = False
    if not self._session_wrapper:
      self._session_wrapper = dumping_wrapper.DumpingDebugWrapperSession(
          run_context.session,
          self._session_root,
          watch_fn=self._watch_fn,
          thread_name_filter=self._thread_name_filter)
      reset_disk_byte_usage = True

    self._session_wrapper.increment_run_call_count()

    # pylint: disable=protected-access
    debug_urls, watch_options = self._session_wrapper._prepare_run_watch_config(
        run_context.original_args.fetches, run_context.original_args.feed_dict)
    # pylint: enable=protected-access
    run_options = config_pb2.RunOptions()
    debug_utils.watch_graph(
        run_options,
        run_context.session.graph,
        debug_urls=debug_urls,
        debug_ops=watch_options.debug_ops,
        node_name_regex_allowlist=watch_options.node_name_regex_allowlist,
        op_type_regex_allowlist=watch_options.op_type_regex_allowlist,
        tensor_dtype_regex_allowlist=watch_options.tensor_dtype_regex_allowlist,
        tolerate_debug_op_creation_failures=(
            watch_options.tolerate_debug_op_creation_failures),
        reset_disk_byte_usage=reset_disk_byte_usage)

    run_args = session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=run_options)
    return run_args

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # Hook callback invoked after each run; intentionally does nothing.
    pass


class GrpcDebugHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session run hook that streams debug events to gRPC debug servers through a lazily created gRPC
  # wrapper session. State: list of server addresses (a single address is wrapped into a list),
  # watch callback, thread filter, and the wrapper session (initially absent).
  """A hook that streams debugger-related events to any grpc_debug_server.

  For example, the debugger data server is a grpc_debug_server. The debugger
  data server writes debugger-related events it receives via GRPC to logdir.
  This enables debugging features in Tensorboard such as health pills.

  When the arguments of debug_utils.watch_graph changes, strongly consider
  changing arguments here too so that features are available to tflearn users.

  Can be used as a hook for `tf.compat.v1.train.MonitoredSession`.
  """

  def __init__(self,
               grpc_debug_server_addresses,
               watch_fn=None,
               thread_name_filter=None):
    # [FUNCTION DOC]
    # Stores settings. Inputs: grpc_debug_server_addresses (a list of address strings, or a single
    # string which is wrapped into a list), watch_fn (callback returning watch options per run;
    # needed in practice because it is called on every run), thread_name_filter (optional regular
    # expression). Returns nothing.
    """Constructs a GrpcDebugHook.

    Args:
      grpc_debug_server_addresses: (`list` of `str`) A list of the gRPC debug
        server addresses, in the format of <host:port>, with or without the
        "grpc://" prefix. For example: ["localhost:7000", "192.168.0.2:8000"]
      watch_fn: A function that allows for customizing which ops to watch at
        which specific steps. See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__` for details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
    """
    self._grpc_debug_wrapper_session = None
    self._thread_name_filter = thread_name_filter
    self._grpc_debug_server_addresses = (
        grpc_debug_server_addresses
        if isinstance(grpc_debug_server_addresses, list) else
        [grpc_debug_server_addresses])

    self._watch_fn = watch_fn

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the watch callback is invoked unconditionally, so a hook built without one
    # (the default) fails with a not-callable error on the first run.
    # Hook callback before each run. Lazily creates the gRPC wrapper session, calls the watch
    # callback with the original fetches and feeds to get watch options, and marks the graph for
    # watching in fresh run options with the wrapper's debug URLs, debug ops, regular-expression
    # allowlists and tolerance flag. Returns run arguments carrying those run options.
    """Called right before a session is run.

    Args:
      run_context: A session_run_hook.SessionRunContext. Encapsulates
        information on the run.

    Returns:
      A session_run_hook.SessionRunArgs object.
    """

    if not self._grpc_debug_wrapper_session:
      self._grpc_debug_wrapper_session = grpc_wrapper.GrpcDebugWrapperSession(
          run_context.session,
          self._grpc_debug_server_addresses,
          watch_fn=self._watch_fn,
          thread_name_filter=self._thread_name_filter)

    fetches = run_context.original_args.fetches
    feed_dict = run_context.original_args.feed_dict
    watch_options = self._watch_fn(fetches, feed_dict)
    run_options = config_pb2.RunOptions()
    debug_utils.watch_graph(
        run_options,
        run_context.session.graph,
        debug_urls=self._grpc_debug_wrapper_session.prepare_run_debug_urls(
            fetches, feed_dict),
        debug_ops=watch_options.debug_ops,
        node_name_regex_allowlist=watch_options.node_name_regex_allowlist,
        op_type_regex_allowlist=watch_options.op_type_regex_allowlist,
        tensor_dtype_regex_allowlist=watch_options.tensor_dtype_regex_allowlist,
        tolerate_debug_op_creation_failures=(
            watch_options.tolerate_debug_op_creation_failures))

    return session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=run_options)


class TensorBoardDebugHook(GrpcDebugHook):
  # [CLASS DOC]
  # gRPC debug hook preconfigured for the TensorBoard debugger plugin: uses a fixed watch callback
  # that watches all tensors with gated gRPC identity debug ops, and optionally publishes graph
  # tracebacks and source code before each run. State: raw server addresses, a send-traceback
  # flag, and the last graph version sent (initially minus one).
  """A tfdbg hook that can be used with TensorBoard Debugger Plugin.

  This hook is the same as `GrpcDebugHook`, except that it uses a predefined
    `watch_fn` that
    1) uses `DebugIdentity` debug ops with the `gated_grpc` attribute set to
        `True`, to allow the interactive enabling and disabling of tensor
       breakpoints.
    2) watches all tensors in the graph.
  This saves the need for the user to define a `watch_fn`.
  """

  def __init__(self,
               grpc_debug_server_addresses,
               thread_name_filter=None,
               send_traceback_and_source_code=True):
    # [FUNCTION DOC]
    # Initializes the parent gRPC hook with the fixed gated watch callback, then stores the raw
    # addresses (overwriting the parent's list form), the send flag and the sent graph version
    # (minus one). Inputs: grpc_debug_server_addresses (string or list of strings),
    # thread_name_filter (optional), send_traceback_and_source_code (boolean, default true).
    # Side effect: installs the process-wide interrupt handler. Returns nothing.
    """Constructor of TensorBoardDebugHook.

    Args:
      grpc_debug_server_addresses: gRPC address(es) of debug server(s), as a
        `str` or a `list` of `str`s. E.g., "localhost:2333",
        "grpc://localhost:2333", ["192.168.0.7:2333", "192.168.0.8:2333"].
      thread_name_filter: Optional filter for thread names.
      send_traceback_and_source_code: Whether traceback of graph elements and
        the source code are to be sent to the debug server(s).
    """

    def _gated_grpc_watch_fn(fetches, feeds):
      # [FUNCTION DOC]
      # HELPER: watch callback for the TensorBoard hook constructor.
      # Ignores its inputs and returns watch options requesting one identity debug op with the
      # gated gRPC attribute enabled.
      del fetches, feeds  # Unused.
      return framework.WatchOptions(
          debug_ops=["DebugIdentity(gated_grpc=true)"])

    super(TensorBoardDebugHook, self).__init__(
        grpc_debug_server_addresses,
        watch_fn=_gated_grpc_watch_fn,
        thread_name_filter=thread_name_filter)

    self._grpc_debug_server_addresses = grpc_debug_server_addresses
    self._send_traceback_and_source_code = send_traceback_and_source_code
    self._sent_graph_version = -1
    grpc_wrapper.register_signal_handler()

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # If enabled, publishes the session graph's traceback and source code to the servers when the
    # graph version is newer than the last sent (updating the stored version), then delegates to
    # the parent hook's before-run and returns its run arguments. Side effect: network sends.
    if self._send_traceback_and_source_code:
      self._sent_graph_version = grpc_wrapper.publish_traceback(
          self._grpc_debug_server_addresses, run_context.session.graph,
          run_context.original_args.feed_dict,
          run_context.original_args.fetches, self._sent_graph_version)
    return super(TensorBoardDebugHook, self).before_run(run_context)
