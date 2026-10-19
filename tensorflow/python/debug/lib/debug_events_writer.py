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
# [MODULE DOC]
# Thin Python wrapper over the native debug-events writer used by tfdbg v2. Defines the default
# circular buffer size (in number of events) and a writer class that wraps each kind of tfdbg
# proto (source file, stack frame, graph op creation, debugged graph, execution, graph execution
# trace) in a debug event stamped with the current time and hands it to the native writer for the
# dump root directory. Imports the native extension; no other import-time side effects.
"""Writer class for `DebugEvent` protos in tfdbg v2."""

import time

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer

# Default size of each circular buffer (unit: number of DebugEvent protos).
DEFAULT_CIRCULAR_BUFFER_SIZE = 1000


class DebugEventsWriter(object):
  # [CLASS DOC]
  # Writer of tfdbg v2 debug events for one dump root directory. Holds the dump root and run id;
  # all real work (file sets, buffering, flushing) is done by the native writer keyed by the dump
  # root, so several instances with the same dump root share the same underlying files. Each
  # write method wraps its proto in a debug event, stamps the wall time if unset and forwards it.
  """A writer for TF debugging events. Used by tfdbg v2."""

  def __init__(self,
               dump_root,
               tfdbg_run_id,
               circular_buffer_size=DEFAULT_CIRCULAR_BUFFER_SIZE):
    # [FUNCTION DOC]
    # Constructor. Takes the dump root directory (required, non-empty; created by the native layer
    # if missing), the debugger run id, and the optional circular buffer size for the execution
    # and graph-execution-trace files (default is the module constant; zero or negative disables
    # circular buffering). Raises a value error if the dump root is empty or null. Side effect:
    # initializes the native writer for that dump root.
    """Construct a DebugEventsWriter object.

    NOTE: Given the same `dump_root`, all objects from this constructor
      will point to the same underlying set of writers. In other words, they
      will write to the same set of debug events files in the `dump_root`
      folder.

    Args:
      dump_root: The root directory for dumping debug data. If `dump_root` does
        not exist as a directory, it will be created.
      tfdbg_run_id: Debugger Run ID.
      circular_buffer_size: Size of the circular buffer for each of the two
        execution-related debug events files: with the following suffixes: -
          .execution - .graph_execution_traces If <= 0, the circular-buffer
          behavior will be abolished in the constructed object.
    """
    if not dump_root:
      raise ValueError("Empty or None dump root")
    self._dump_root = dump_root
    self._tfdbg_run_id = tfdbg_run_id
    _pywrap_debug_events_writer.Init(self._dump_root, self._tfdbg_run_id,
                                     circular_buffer_size)

  def WriteSourceFile(self, source_file):
    # [FUNCTION DOC]
    # Takes a source-file proto, wraps it in a debug event, stamps the current time if unset and
    # passes it to the native writer for the source-files file. Returns nothing.
    """Write a SourceFile proto with the writer.

    Args:
      source_file: A SourceFile proto, describing the content of a source file
        involved in the execution of the debugged TensorFlow program.
    """
    # TODO(cais): Explore performance optimization that avoids memcpy.
    debug_event = debug_event_pb2.DebugEvent(source_file=source_file)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteSourceFile(self._dump_root, debug_event)

  def WriteStackFrameWithId(self, stack_frame_with_id):
    # [FUNCTION DOC]
    # Takes a stack-frame-with-id proto, wraps it in a timestamped debug event and passes it to
    # the native writer for the stack-frames file. Returns nothing.
    """Write a StackFrameWithId proto with the writer.

    Args:
      stack_frame_with_id: A StackFrameWithId proto, describing the content a
        stack frame involved in the execution of the debugged TensorFlow
        program.
    """
    debug_event = debug_event_pb2.DebugEvent(
        stack_frame_with_id=stack_frame_with_id)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteStackFrameWithId(self._dump_root,
                                                      debug_event)

  def WriteGraphOpCreation(self, graph_op_creation):
    # [FUNCTION DOC]
    # Takes a graph-op-creation proto, wraps it in a timestamped debug event and passes it to the
    # native writer for the graphs file. Returns nothing.
    """Write a GraphOpCreation proto with the writer.

    Args:
      graph_op_creation: A GraphOpCreation proto, describing the details of the
        creation of an op inside a TensorFlow Graph.
    """
    debug_event = debug_event_pb2.DebugEvent(
        graph_op_creation=graph_op_creation)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteGraphOpCreation(self._dump_root,
                                                     debug_event)

  def WriteDebuggedGraph(self, debugged_graph):
    # [FUNCTION DOC]
    # Takes a debugged-graph proto (a graph whose construction finished), wraps it in a
    # timestamped debug event and passes it to the native writer for the graphs file. Returns
    # nothing.
    """Write a DebuggedGraph proto with the writer.

    Args:
      debugged_graph: A DebuggedGraph proto, describing the details of a
        TensorFlow Graph that has completed its construction.
    """
    debug_event = debug_event_pb2.DebugEvent(debugged_graph=debugged_graph)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteDebuggedGraph(self._dump_root, debug_event)

  def WriteExecution(self, execution):
    # [FUNCTION DOC]
    # Takes an execution proto (an op or graph execution), wraps it in a timestamped debug event
    # and passes it to the native writer for the execution file (circular buffer, if enabled).
    # Returns nothing.
    """Write a Execution proto with the writer.

    Args:
      execution: An Execution proto, describing a TensorFlow op or graph
        execution event.
    """
    debug_event = debug_event_pb2.DebugEvent(execution=execution)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteExecution(self._dump_root, debug_event)

  def WriteGraphExecutionTrace(self, graph_execution_trace):
    # [FUNCTION DOC]
    # Takes a graph-execution-trace proto (value of an intermediate tensor), wraps it in a
    # timestamped debug event and passes it to the native writer for the graph-execution-traces
    # file (circular buffer, if enabled). Returns nothing.
    """Write a GraphExecutionTrace proto with the writer.

    Args:
      graph_execution_trace: A GraphExecutionTrace proto, concerning the value
        of an intermediate tensor or a list of intermediate tensors that are
        computed during the graph's execution.
    """
    debug_event = debug_event_pb2.DebugEvent(
        graph_execution_trace=graph_execution_trace)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteGraphExecutionTrace(
        self._dump_root, debug_event)

  def RegisterDeviceAndGetId(self, device_name):
    # [FUNCTION DOC]
    # Registers a device name with the native writer for this dump root (recording it in the
    # graphs file if new) and returns the integer device id assigned to it. No docstring exists.
    return _pywrap_debug_events_writer.RegisterDeviceAndGetId(
        self._dump_root, device_name)

  def FlushNonExecutionFiles(self):
    # [FUNCTION DOC]
    # Flushes the metadata, source-files, stack-frames and graphs files for this dump root to
    # disk. Returns nothing.
    """Flush the non-execution debug event files."""
    _pywrap_debug_events_writer.FlushNonExecutionFiles(self._dump_root)

  def FlushExecutionFiles(self):
    # [FUNCTION DOC]
    # Writes the circular buffers' current content to the execution and graph-execution-traces
    # files for this dump root and clears the buffers. Returns nothing.
    """Flush the execution debug event files.

    Causes the current content of the cyclic buffers to be written to
    the .execution and .graph_execution_traces debug events files.
    Also clears those cyclic buffers.
    """
    _pywrap_debug_events_writer.FlushExecutionFiles(self._dump_root)

  def Close(self):
    # [FUNCTION DOC]
    # Closes the native writer (all file sets) for this dump root. Returns nothing.
    """Close the writer."""
    _pywrap_debug_events_writer.Close(self._dump_root)

  @property
  def dump_root(self):
    # [FUNCTION DOC]
    # Returns the dump root directory this writer was created with.
    return self._dump_root

  def _EnsureTimestampAdded(self, debug_event):
    # [FUNCTION DOC]
    # HELPER: used by every write method before handing an event to the native writer.
    # Takes a debug event proto and, if its wall time is zero, sets it to the current time in
    # seconds (mutating the argument). Returns nothing.
    if debug_event.wall_time == 0:
      debug_event.wall_time = time.time()
