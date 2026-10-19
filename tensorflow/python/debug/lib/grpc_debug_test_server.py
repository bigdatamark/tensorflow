# [MODULE DOC]
# Test-support gRPC debug server. Provides a stream handler and a servicer built on the base
# debug server that either dump received debug data (core metadata, graph definitions, tensor
# values) to the file system, using the same layout as file-based debug dumps, or keep it in
# memory; it also records tracebacks and source files and offers query methods for them. Also
# offers a function that starts such a server on a separate thread with an unused port, and a
# helper that polls the server until a debugged session run reaches it. Helper module for
# debugger tests; no import-time side effects.
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
"""GRPC debug server for testing."""
import collections
import errno
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time

import portpicker

from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.util import compat


def _get_dump_file_path(dump_root, device_name, debug_node_name):
  # [FUNCTION DOC]
  # HELPER: computes where the test server writes a tensor dump file.
  # Joins the dump root with the device path derived from the device name. If the debug node name
  # contains slashes, its directory part becomes subdirectories and the file name is the base
  # name; colons in the file name are replaced by underscores. A suffix with the current time in
  # microseconds is appended to the file name. Returns the full path string; creates nothing.
  """Get the file path of the dump file for a debug node.

  Args:
    dump_root: (str) Root dump directory.
    device_name: (str) Name of the device that the debug node resides on.
    debug_node_name: (str) Name of the debug node, e.g.,
      cross_entropy/Log:0:DebugIdentity.

  Returns:
    (str) Full path of the dump file.
  """

  dump_root = os.path.join(
      dump_root, debug_data.device_name_to_device_path(device_name))
  if "/" in debug_node_name:
    dump_dir = os.path.join(dump_root, os.path.dirname(debug_node_name))
    dump_file_name = re.sub(":", "_", os.path.basename(debug_node_name))
  else:
    dump_dir = dump_root
    dump_file_name = re.sub(":", "_", debug_node_name)

  now_microsec = int(round(time.time() * 1000 * 1000))
  dump_file_name += "_%d" % now_microsec

  return os.path.join(dump_dir, dump_file_name)


class EventListenerTestStreamHandler(
    grpc_debug_server.EventListenerBaseStreamHandler):
  # [CLASS DOC]
  # HELPER: test stream handler used by the test servicer.
  # Stream handler that either writes received events to a dump directory on disk or, when no
  # dump directory is given, stores them in the owning servicer's in-memory lists. Holds the dump
  # directory, a reference to the servicer, the grpc path from core metadata, and caches of graph
  # definitions (with device names and wall times) received before the grpc path is known.
  """Implementation of EventListenerBaseStreamHandler that dumps to file."""

  def __init__(self, dump_dir, event_listener_servicer):
    # [FUNCTION DOC]
    # Calls the base constructor, stores the dump directory (may be empty or nothing for in-memory
    # mode) and the servicer, creates the dump directory if needed, and initializes the grpc path
    # to nothing and empty caches for graph definitions, device names and wall times.
    super(EventListenerTestStreamHandler, self).__init__()
    self._dump_dir = dump_dir
    self._event_listener_servicer = event_listener_servicer
    if self._dump_dir:
      self._try_makedirs(self._dump_dir)

    self._grpc_path = None
    self._cached_graph_defs = []
    self._cached_graph_def_device_names = []
    self._cached_graph_def_wall_times = []

  def on_core_metadata_event(self, event):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the stored grpc path is never assigned (it stays nothing), so on every core
    # metadata event the grpc path is appended again to the dump directory and cached graph
    # definitions are rewritten; later graph definitions also keep being cached rather than
    # written immediately.
    # Asks the servicer to toggle its configured watches, then parses the event's JSON message.
    # If the grpc path is not yet set, takes the path from the metadata (dropping one leading
    # slash), extends the dump directory with it and writes all cached graph definitions. Then,
    # if dumping, writes the event to a core metadata file; otherwise appends the raw JSON string
    # to the servicer's list of core metadata strings. Returns nothing.
    self._event_listener_servicer.toggle_watch()

    core_metadata = json.loads(event.log_message.message)

    if not self._grpc_path:
      grpc_path = core_metadata["grpc_path"]
      if grpc_path:
        if grpc_path.startswith("/"):
          grpc_path = grpc_path[1:]
      if self._dump_dir:
        self._dump_dir = os.path.join(self._dump_dir, grpc_path)

        # Write cached graph defs to filesystem.
        for graph_def, device_name, wall_time in zip(
            self._cached_graph_defs,
            self._cached_graph_def_device_names,
            self._cached_graph_def_wall_times):
          self._write_graph_def(graph_def, device_name, wall_time)

    if self._dump_dir:
      self._write_core_metadata_event(event)
    else:
      self._event_listener_servicer.core_metadata_json_strings.append(
          event.log_message.message)

  def on_graph_def(self, graph_def, device_name, wall_time):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring calls this the tensor value-carrying Event callback; it actually
    # handles graph definitions.
    # When dumping: writes the graph definition to disk if the grpc path is known, otherwise
    # caches it with its device name and wall time. When not dumping: appends the graph to the
    # servicer's list of partition graph definitions. Returns nothing.
    """Implementation of the tensor value-carrying Event proto callback.

    Args:
      graph_def: A GraphDef object.
      device_name: Name of the device on which the graph was created.
      wall_time: An epoch timestamp (in microseconds) for the graph.
    """
    if self._dump_dir:
      if self._grpc_path:
        self._write_graph_def(graph_def, device_name, wall_time)
      else:
        self._cached_graph_defs.append(graph_def)
        self._cached_graph_def_device_names.append(device_name)
        self._cached_graph_def_wall_times.append(wall_time)
    else:
      self._event_listener_servicer.partition_graph_defs.append(graph_def)

  def on_value_event(self, event):
    # [FUNCTION DOC]
    # Handles a tensor-value event. When dumping, writes the event to a file under the dump
    # directory. Otherwise loads the tensor from the event and appends it to the servicer's
    # in-memory mapping keyed by the debug node name; then parses the node name into node name,
    # output slot and debug op and, if it is one of the servicer's active breakpoints, returns an
    # empty reply message. Returns nothing in all other cases.
    """Implementation of the tensor value-carrying Event proto callback.

    Writes the Event proto to the file system for testing. The path written to
    follows the same pattern as the file:// debug URLs of tfdbg, i.e., the
    name scope of the op becomes the directory structure under the dump root
    directory.

    Args:
      event: The Event proto carrying a tensor value.

    Returns:
      If the debug node belongs to the set of currently activated breakpoints,
      a `EventReply` proto will be returned.
    """
    if self._dump_dir:
      self._write_value_event(event)
    else:
      value = event.summary.value[0]
      tensor_value = debug_data.load_tensor_from_event(event)
      self._event_listener_servicer.debug_tensor_values[value.node_name].append(
          tensor_value)

      items = event.summary.value[0].node_name.split(":")
      node_name = items[0]
      output_slot = int(items[1])
      debug_op = items[2]
      if ((node_name, output_slot, debug_op) in
          self._event_listener_servicer.breakpoints):
        return debug_service_pb2.EventReply()

  def _try_makedirs(self, dir_path):
    # [FUNCTION DOC]
    # HELPER: creates a directory tree if it is not already a directory, tolerating a race where
    # another party creates it first (an already-exists error is ignored; other OS errors are
    # re-raised).
    if not os.path.isdir(dir_path):
      try:
        os.makedirs(dir_path)
      except OSError as error:
        if error.errno != errno.EEXIST:
          raise

  def _write_core_metadata_event(self, event):
    # [FUNCTION DOC]
    # HELPER: writes the serialized core metadata event as a binary file in the dump directory,
    # named with the metadata prefix, the core metadata tag and the event wall time. Creates the
    # directory first.
    core_metadata_path = os.path.join(
        self._dump_dir,
        debug_data.METADATA_FILE_PREFIX + debug_data.CORE_METADATA_TAG +
        "_%d" % event.wall_time)
    self._try_makedirs(self._dump_dir)
    with open(core_metadata_path, "wb") as f:
      f.write(event.SerializeToString())

  def _write_graph_def(self, graph_def, device_name, wall_time):
    # [FUNCTION DOC]
    # HELPER: serializes the graph definition, computes an integer hash of the bytes, wraps it in
    # an Event with the given wall time and writes it as a binary file under the dump directory,
    # in the device's subdirectory, with a name made from the metadata prefix, graph tag, hash tag,
    # hash and wall time. Creates directories as needed. Returns nothing.
    encoded_graph_def = graph_def.SerializeToString()
    graph_hash = int(hashlib.sha1(encoded_graph_def).hexdigest(), 16)
    event = event_pb2.Event(graph_def=encoded_graph_def, wall_time=wall_time)
    graph_file_path = os.path.join(
        self._dump_dir,
        debug_data.device_name_to_device_path(device_name),
        debug_data.METADATA_FILE_PREFIX + debug_data.GRAPH_FILE_TAG +
        debug_data.HASH_TAG + "%d_%d" % (graph_hash, wall_time))
    self._try_makedirs(os.path.dirname(graph_file_path))
    with open(graph_file_path, "wb") as f:
      f.write(event.SerializeToString())

  def _write_value_event(self, event):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if parsing the plugin content fails, the error message refers to the content
    # variable which is not yet assigned at that point, so an unbound-variable error would be
    # raised instead of the intended value error.
    # HELPER: writes a tensor-value event to disk. Requires plugin data in the value metadata
    # (raises a value error if missing), parses its JSON content to find the device name (value
    # error if unparsable), computes the dump path and writes the serialized event there,
    # creating directories. Returns nothing.
    value = event.summary.value[0]

    # Obtain the device name from the metadata.
    summary_metadata = event.summary.value[0].metadata
    if not summary_metadata.plugin_data:
      raise ValueError("The value lacks plugin data.")
    try:
      content = json.loads(compat.as_text(summary_metadata.plugin_data.content))
    except ValueError as err:
      raise ValueError("Could not parse content into JSON: %r, %r" % (content,
                                                                      err))
    device_name = content["device"]

    dump_full_path = _get_dump_file_path(
        self._dump_dir, device_name, value.node_name)
    self._try_makedirs(os.path.dirname(dump_full_path))
    with open(dump_full_path, "wb") as f:
      f.write(event.SerializeToString())


class EventListenerTestServicer(grpc_debug_server.EventListenerBaseServicer):
  # [CLASS DOC]
  # Test servicer built on the base servicer. Uses the test stream handler to dump or store debug
  # data, optionally toggles configured watches on each core metadata event, and records
  # traceback and source-file messages (call types, call keys, origin stacks, graph tracebacks,
  # graph versions, source files). Public in-memory attributes: core metadata JSON strings,
  # partition graph definitions and a mapping from debug node names to lists of tensor values.
  # Provides query methods and a clear method for tests.
  """An implementation of EventListenerBaseServicer for testing."""

  def __init__(self, server_port, dump_dir, toggle_watch_on_core_metadata=None):
    # [FUNCTION DOC]
    # Initializes the in-memory stores and toggle-watch state, then calls the base constructor
    # with the port and a stream handler factory bound to the dump directory and this servicer.
    # server_port is an integer port; dump_dir is a directory path (empty or nothing means keep
    # data in memory); toggle_watch_on_core_metadata is an optional list of watch tuples (node
    # name, output slot, debug op) toggled on core metadata events. Also creates empty lists for
    # tracebacks and source files.
    """Constructor of EventListenerTestServicer.

    Args:
      server_port: (int) The server port number.
      dump_dir: (str) The root directory to which the data files will be
        dumped. If empty or None, the received debug data will not be dumped
        to the file system: they will be stored in memory instead.
      toggle_watch_on_core_metadata: A list of
        (node_name, output_slot, debug_op) tuples to toggle the
        watchpoint status during the on_core_metadata calls (optional).
    """
    self.core_metadata_json_strings = []
    self.partition_graph_defs = []
    self.debug_tensor_values = collections.defaultdict(list)
    self._initialize_toggle_watch_state(toggle_watch_on_core_metadata)

    grpc_debug_server.EventListenerBaseServicer.__init__(
        self, server_port,
        functools.partial(EventListenerTestStreamHandler, dump_dir, self))

    # Members for storing the graph ops traceback and source files.
    self._call_types = []
    self._call_keys = []
    self._origin_stacks = []
    self._origin_id_to_strings = []
    self._graph_tracebacks = []
    self._graph_versions = []
    self._source_files = []

  def _initialize_toggle_watch_state(self, toggle_watches):
    # [FUNCTION DOC]
    # HELPER: stores the list of watches to toggle (possibly nothing) and builds a mapping from
    # each watch tuple to a state flag initially false (not watched).
    self._toggle_watches = toggle_watches
    self._toggle_watch_state = {}
    if self._toggle_watches:
      for watch_key in self._toggle_watches:
        self._toggle_watch_state[watch_key] = False

  def toggle_watch(self):
    # [FUNCTION DOC]
    # For each configured watch: if currently on, queues an unwatch request, otherwise queues a
    # watch request, then flips the stored flag. Called on each core metadata event. Returns
    # nothing.
    for watch_key in self._toggle_watch_state:
      node_name, output_slot, debug_op = watch_key
      if self._toggle_watch_state[watch_key]:
        self.request_unwatch(node_name, output_slot, debug_op)
      else:
        self.request_watch(node_name, output_slot, debug_op)
      self._toggle_watch_state[watch_key] = (
          not self._toggle_watch_state[watch_key])

  def clear_data(self):
    # [FUNCTION DOC]
    # Resets all recorded data to empty: core metadata strings, partition graph definitions, tensor
    # values, call types, call keys, origin stacks, origin string tables, graph tracebacks, graph
    # versions and source files. Does not reset toggle-watch state. Returns nothing.
    self.core_metadata_json_strings = []
    self.partition_graph_defs = []
    self.debug_tensor_values = collections.defaultdict(list)
    self._call_types = []
    self._call_keys = []
    self._origin_stacks = []
    self._origin_id_to_strings = []
    self._graph_tracebacks = []
    self._graph_versions = []
    self._source_files = []

  def SendTracebacks(self, request, context):
    # [FUNCTION DOC]
    # Records the call type, call key, origin stack, origin id-to-string table, graph traceback
    # and graph version of the request in the corresponding lists, and returns an empty reply.
    # The context argument is unused.
    self._call_types.append(request.call_type)
    self._call_keys.append(request.call_key)
    self._origin_stacks.append(request.origin_stack)
    self._origin_id_to_strings.append(request.origin_id_to_string)
    self._graph_tracebacks.append(request.graph_traceback)
    self._graph_versions.append(request.graph_version)
    return debug_service_pb2.EventReply()

  def SendSourceFiles(self, request, context):
    # [FUNCTION DOC]
    # Appends the received source-files message to the list of stored source files and returns an
    # empty reply. The context argument is unused.
    self._source_files.append(request)
    return debug_service_pb2.EventReply()

  def query_op_traceback(self, op_name):
    # [FUNCTION DOC]
    # Searches the received graph tracebacks for a log entry with the given op name and returns its
    # traceback as a list of tuples (file name, line number, function name). Raises a value error
    # if no entry matches.
    """Query the traceback of an op.

    Args:
      op_name: Name of the op to query.

    Returns:
      The traceback of the op, as a list of 3-tuples:
        (filename, lineno, function_name)

    Raises:
      ValueError: If the op cannot be found in the tracebacks received by the
        server so far.
    """
    for op_log_proto in self._graph_tracebacks:
      for log_entry in op_log_proto.log_entries:
        if log_entry.name == op_name:
          return self._code_def_to_traceback(log_entry.code_def,
                                             op_log_proto.id_to_string)
    raise ValueError(
        "Op '%s' does not exist in the tracebacks received by the debug "
        "server." % op_name)

  def query_origin_stack(self):
    # [FUNCTION DOC]
    # Returns a list with one traceback per received traceback request, each a list of tuples
    # (file name, line number, function name) decoded from the stored origin stacks.
    """Query the stack of the origin of the execution call.

    Returns:
      A `list` of all tracebacks. Each item corresponds to an execution call,
        i.e., a `SendTracebacks` request. Each item is a `list` of 3-tuples:
        (filename, lineno, function_name).
    """
    ret = []
    for stack, id_to_string in zip(
        self._origin_stacks, self._origin_id_to_strings):
      ret.append(self._code_def_to_traceback(stack, id_to_string))
    return ret

  def query_call_types(self):
    # [FUNCTION DOC]
    # Returns the live list of call types recorded from traceback requests.
    return self._call_types

  def query_call_keys(self):
    # [FUNCTION DOC]
    # Returns the live list of call keys recorded from traceback requests.
    return self._call_keys

  def query_graph_versions(self):
    # [FUNCTION DOC]
    # Returns the live list of graph versions recorded from traceback requests.
    return self._graph_versions

  def query_source_file_line(self, file_path, lineno):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final error is built with the message template and the path as two separate
    # arguments instead of formatting, so the path is not substituted into the message.
    # Returns the text of the given one-based line number from the received source file whose
    # path equals file_path. Raises a value error if no source file has been received or the path
    # is not found.
    """Query the content of a given line in a source file.

    Args:
      file_path: Path to the source file.
      lineno: Line number as an `int`.

    Returns:
      Content of the line as a string.

    Raises:
      ValueError: If no source file is found at the given file_path.
    """
    if not self._source_files:
      raise ValueError(
          "This debug server has not received any source file contents yet.")
    for source_files in self._source_files:
      for source_file_proto in source_files.source_files:
        if source_file_proto.file_path == file_path:
          return source_file_proto.lines[lineno - 1]
    raise ValueError(
        "Source file at path %s has not been received by the debug server",
        file_path)

  def _code_def_to_traceback(self, code_def, id_to_string):
    # [FUNCTION DOC]
    # HELPER: converts a code definition message into a list of tuples (file name, line number,
    # function name), resolving file and function ids through the id-to-string table.
    return [(id_to_string[trace.file_id],
             trace.lineno,
             id_to_string[trace.function_id]) for trace in code_def.traces]


def start_server_on_separate_thread(dump_to_filesystem=True,
                                    server_start_delay_sec=0.0,
                                    poll_server=False,
                                    blocking=True,
                                    toggle_watch_on_core_metadata=None):
  # [FUNCTION DOC]
  # Starts a test servicer on a thread. Picks an unused port, creates a temporary dump directory
  # if dump_to_filesystem is true (otherwise data is kept in memory), builds the servicer (with
  # the optional toggle watches) and starts a thread that waits server_start_delay_sec seconds
  # then runs the server (blocking mode per the blocking flag). If poll_server is true, polls
  # until the server receives data (up to 50 attempts, 0.2 seconds apart, small GPU memory
  # fraction), raising a value error on failure, then clears the recorded data. Returns the port,
  # the grpc URL, the dump directory (or nothing), the thread and the servicer. Side effects:
  # creates a temp directory and a non-joined thread.
  """Create a test gRPC debug server and run on a separate thread.

  Args:
    dump_to_filesystem: (bool) whether the debug server will dump debug data
      to the filesystem.
    server_start_delay_sec: (float) amount of time (in sec) to delay the server
      start up for.
    poll_server: (bool) whether the server will be polled till success on
      startup.
    blocking: (bool) whether the server should be started in a blocking mode.
    toggle_watch_on_core_metadata: A list of
        (node_name, output_slot, debug_op) tuples to toggle the
        watchpoint status during the on_core_metadata calls (optional).

  Returns:
    server_port: (int) Port on which the server runs.
    debug_server_url: (str) grpc:// URL to the server.
    server_dump_dir: (str) The debug server's dump directory.
    server_thread: The server Thread object.
    server: The `EventListenerTestServicer` object.

  Raises:
    ValueError: If polling the server process for ready state is not successful
      within maximum polling count.
  """
  server_port = portpicker.pick_unused_port()
  debug_server_url = "grpc://localhost:%d" % server_port

  server_dump_dir = tempfile.mkdtemp() if dump_to_filesystem else None
  server = EventListenerTestServicer(
      server_port=server_port,
      dump_dir=server_dump_dir,
      toggle_watch_on_core_metadata=toggle_watch_on_core_metadata)

  def delay_then_run_server():
    # [FUNCTION DOC]
    # HELPER: thread target. Sleeps for the configured start delay, then runs the server with the
    # configured blocking mode. Returns nothing.
    time.sleep(server_start_delay_sec)
    server.run_server(blocking=blocking)

  server_thread = threading.Thread(target=delay_then_run_server)
  server_thread.start()

  if poll_server:
    if not _poll_server_till_success(
        50,
        0.2,
        debug_server_url,
        server_dump_dir,
        server,
        gpu_memory_fraction=0.1):
      raise ValueError(
          "Failed to start test gRPC debug server at port %d" % server_port)
    server.clear_data()
  return server_port, debug_server_url, server_dump_dir, server_thread, server


def _poll_server_till_success(max_attempts,
                              sleep_per_poll_sec,
                              debug_server_url,
                              dump_dir,
                              server,
                              gpu_memory_fraction=1.0):
  # [FUNCTION DOC]
  # HELPER: used by the server start function to confirm the server is reachable.
  # Opens a session limited to the given GPU memory fraction (default all). On each attempt, up
  # to max_attempts: clears server data, builds a small constant and variable, runs the variable
  # initializer with a debug watch pointing at the debug server URL (ignoring failed-precondition
  # errors), then checks success: with a dump directory, success means it exists and the dump
  # holds data (then deletes it); without one, success means the server holds tensor values.
  # On failure sleeps sleep_per_poll_sec seconds. Prints progress. Returns true on success and
  # false after all attempts fail.
  """Poll server until success or exceeding max polling count.

  Args:
    max_attempts: (int) How many times to poll at maximum
    sleep_per_poll_sec: (float) How many seconds to sleep for after each
      unsuccessful poll.
    debug_server_url: (str) gRPC URL to the debug server.
    dump_dir: (str) Dump directory to look for files in. If None, will directly
      check data from the server object.
    server: The server object.
    gpu_memory_fraction: (float) Fraction of GPU memory to be
      allocated for the Session used in server polling.

  Returns:
    (bool) Whether the polling succeeded within max_polls attempts.
  """
  poll_count = 0

  config = config_pb2.ConfigProto(gpu_options=config_pb2.GPUOptions(
      per_process_gpu_memory_fraction=gpu_memory_fraction))
  with session.Session(config=config) as sess:
    for poll_count in range(max_attempts):
      server.clear_data()
      print("Polling: poll_count = %d" % poll_count)

      x_init_name = "x_init_%d" % poll_count
      x_init = constant_op.constant([42.0], shape=[1], name=x_init_name)
      x = variables.Variable(x_init, name=x_init_name)

      run_options = config_pb2.RunOptions()
      debug_utils.add_debug_tensor_watch(
          run_options, x_init_name, 0, debug_urls=[debug_server_url])
      try:
        sess.run(x.initializer, options=run_options)
      except errors.FailedPreconditionError:
        pass

      if dump_dir:
        if os.path.isdir(
            dump_dir) and debug_data.DebugDumpDir(dump_dir).size > 0:
          file_io.delete_recursively(dump_dir)
          print("Poll succeeded.")
          return True
        else:
          print("Poll failed. Sleeping for %f s" % sleep_per_poll_sec)
          time.sleep(sleep_per_poll_sec)
      else:
        if server.debug_tensor_values:
          print("Poll succeeded.")
          return True
        else:
          print("Poll failed. Sleeping for %f s" % sleep_per_poll_sec)
          time.sleep(sleep_per_poll_sec)

    return False
