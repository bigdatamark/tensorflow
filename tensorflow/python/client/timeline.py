# [MODULE DOC]
# Builds Chrome Trace Format timelines from TensorFlow step statistics. Exposes Timeline (turns a
# step stats record plus an optional graph into a trace), two result tuple types
# (AllocationMaximum, StepStatsAnalysis) and the internal trace formatter and tensor tracker
# helpers. No import-time side effects besides imports.
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
"""Timeline visualization for TensorFlow using Chrome Trace Format."""

import collections
import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from tensorflow.core.framework import step_stats_pb2
# The timeline target is usually imported as part of BUILD target
# "platform_test", which includes also includes the "platform"
# dependency.  This is why the logging import here is okay.
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging


class AllocationMaximum(
    collections.namedtuple(
        'AllocationMaximum', ('timestamp', 'num_bytes', 'tensors')
    )
):
  # [CLASS DOC]
  # Immutable record (named tuple) with fields timestamp, num_bytes and tensors describing the
  # peak memory use of one allocator: when it happened, total bytes, and the set of tensor names
  # live then. Created by Timeline when computing memory counters.
  """Stores the maximum allocation for a given allocator within the timelne.

  Parameters:
    timestamp: `tensorflow::Env::NowMicros()` when this maximum was reached.
    num_bytes: the total memory used at this time.
    tensors: the set of tensors allocated at this time.
  """


class StepStatsAnalysis(
    collections.namedtuple(
        'StepStatsAnalysis', ('chrome_trace', 'allocator_maximums')
    )
):
  # [CLASS DOC]
  # Immutable record (named tuple) returned by step stats analysis: chrome_trace holds the trace
  # formatter object with all events, allocator_maximums maps allocator names to
  # AllocationMaximum records.
  """Stores the step stats analysis output.

  Parameters:
    chrome_trace: A dict containing the chrome trace analysis.
    allocator_maximums: A dict mapping allocator names to AllocationMaximum.
  """


class _ChromeTraceFormatter(object):
  # [CLASS DOC]
  # HELPER: internal event collector used by Timeline to produce Chrome Trace JSON.
  # Keeps two lists, metadata events (process and thread names) and regular events, and offers
  # emit methods that append events of each kind, plus a method to serialize everything.
  """A helper class for generating traces in Chrome Trace Format."""

  def __init__(self, show_memory: bool = False) -> None:
    # [FUNCTION DOC]
    # Initializes the formatter with empty event and metadata lists and stores the optional
    # show_memory flag (boolean, default false; stored but not otherwise used). Returns nothing.
    """Constructs a new Chrome Trace formatter."""
    self._show_memory = show_memory
    self._events = []
    self._metadata = []

  def _create_event(
      self,
      ph: str,
      category: str,
      name: str,
      pid: int,
      tid: int,
      timestamp: int,
  ) -> Dict[str, Union[str, int]]:
    # [FUNCTION DOC]
    # HELPER: builds the common part of every event for the emit methods.
    # Takes event type code, category, name, process id, thread id and timestamp and returns a new
    # mapping with keys for type, category, name, process id, thread id and timestamp. No side
    # effects.
    """Creates a new Chrome Trace event.

    For details of the file format, see:
    https://github.com/catapult-project/catapult/blob/master/tracing/README.md

    Args:
      ph:  The type of event - usually a single character.
      category: The event category as a string.
      name:  The event name as a string.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      timestamp:  The timestamp of this event as a long integer.

    Returns:
      A JSON compatible event object.
    """
    event = {}
    event['ph'] = ph
    event['cat'] = category
    event['name'] = name
    event['pid'] = pid
    event['tid'] = tid
    event['ts'] = timestamp
    return event

  def emit_pid(self, name: str, pid: int) -> None:
    # [FUNCTION DOC]
    # Appends a process-name metadata event for the given process id and display name to the
    # metadata list. Returns nothing.
    """Adds a process metadata event to the trace.

    Args:
      name:  The process name as a string.
      pid:  Identifier of the process as an integer.
    """
    event = {}
    event['name'] = 'process_name'
    event['ph'] = 'M'
    event['pid'] = pid
    event['args'] = {'name': name}
    self._metadata.append(event)

  def emit_tid(self, name, pid, tid):
    # [FUNCTION DOC]
    # Appends a thread-name metadata event (display name, process id, thread id) to the metadata
    # list. Returns nothing.
    """Adds a thread metadata event to the trace.

    Args:
      name:  The thread name as a string.
      pid:  Identifier of the process as an integer.
      tid:  Identifier of the thread as an integer.
    """
    event = {}
    event['name'] = 'thread_name'
    event['ph'] = 'M'
    event['pid'] = pid
    event['tid'] = tid
    event['args'] = {'name': name}
    self._metadata.append(event)

  def emit_region(
      self,
      timestamp: int,
      duration: int,
      pid: int,
      tid: int,
      category: str,
      name: str,
      args: Dict[str, Any],
  ) -> None:
    # [FUNCTION DOC]
    # Appends a complete-duration event ("X" type) with start timestamp, duration, process and
    # thread ids, category, name and a JSON-compatible arguments mapping. Returns nothing.
    """Adds a region event to the trace.

    Args:
      timestamp:  The start timestamp of this region as a long integer.
      duration:  The duration of this region as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      category: The event category as a string.
      name:  The event name as a string.
      args:  A JSON-compatible dictionary of event arguments.
    """
    event = self._create_event('X', category, name, pid, tid, timestamp)
    event['dur'] = duration
    event['args'] = args
    self._events.append(event)

  def emit_obj_create(
      self,
      category: str,
      name: str,
      timestamp: int,
      pid: int,
      tid: int,
      object_id: int,
  ) -> None:
    # [FUNCTION DOC]
    # Appends an object-creation event ("N" type) carrying the object id. Returns nothing.
    """Adds an object creation event to the trace.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      object_id: Identifier of the object as an integer.
    """
    event = self._create_event('N', category, name, pid, tid, timestamp)
    event['id'] = object_id
    self._events.append(event)

  def emit_obj_delete(
      self,
      category: str,
      name: str,
      timestamp: int,
      pid: int,
      tid: int,
      object_id: int,
  ) -> None:
    # [FUNCTION DOC]
    # Appends an object-deletion event ("D" type) carrying the object id. Returns nothing.
    """Adds an object deletion event to the trace.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      object_id: Identifier of the object as an integer.
    """
    event = self._create_event('D', category, name, pid, tid, timestamp)
    event['id'] = object_id
    self._events.append(event)

  def emit_obj_snapshot(
      self,
      category: str,
      name: str,
      timestamp: int,
      pid: int,
      tid: int,
      object_id: int,
      snapshot: Dict[str, Any],
  ) -> None:
    # [FUNCTION DOC]
    # Appends an object-snapshot event ("O" type) with the object id and the given JSON-compatible
    # snapshot stored under the arguments. Returns nothing.
    """Adds an object snapshot event to the trace.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      object_id: Identifier of the object as an integer.
      snapshot:  A JSON-compatible representation of the object.
    """
    event = self._create_event('O', category, name, pid, tid, timestamp)
    event['id'] = object_id
    event['args'] = {'snapshot': snapshot}
    self._events.append(event)

  def emit_flow_start(
      self, name: str, timestamp: int, pid: int, tid: int, flow_id: int
  ) -> None:
    # [FUNCTION DOC]
    # Appends a flow-start event ("s" type, category DataFlow) with the flow id so a viewer can
    # draw an arrow to the matching flow end. Returns nothing.
    """Adds a flow start event to the trace.

    When matched with a flow end event (with the same 'flow_id') this will
    cause the trace viewer to draw an arrow between the start and end events.

    Args:
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      flow_id: Identifier of the flow as an integer.
    """
    event = self._create_event('s', 'DataFlow', name, pid, tid, timestamp)
    event['id'] = flow_id
    self._events.append(event)

  def emit_flow_end(
      self, name: str, timestamp: int, pid: int, tid: int, flow_id: int
  ) -> None:
    # [FUNCTION DOC]
    # Appends a flow-end event ("t" type, category DataFlow) with the flow id matching an earlier
    # flow start. Returns nothing.
    """Adds a flow end event to the trace.

    When matched with a flow start event (with the same 'flow_id') this will
    cause the trace viewer to draw an arrow between the start and end events.

    Args:
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      flow_id: Identifier of the flow as an integer.
    """
    event = self._create_event('t', 'DataFlow', name, pid, tid, timestamp)
    event['id'] = flow_id
    self._events.append(event)

  def emit_counter(
      self,
      category: str,
      name: str,
      pid: int,
      timestamp: int,
      counter: str,
      value: int,
  ) -> None:
    # [FUNCTION DOC]
    # Appends a counter event ("C" type, thread id 0) whose arguments hold one counter name mapped
    # to an integer value. Returns nothing.
    """Emits a record for a single counter.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      pid:  Identifier of the process generating this event as an integer.
      timestamp:  The timestamp of this event as a long integer.
      counter: Name of the counter as a string.
      value:  Value of the counter as an integer.
    """
    event = self._create_event('C', category, name, pid, 0, timestamp)
    event['args'] = {counter: value}
    self._events.append(event)

  def emit_counters(self, category, name, pid, timestamp, counters):
    # [FUNCTION DOC]
    # Appends a counter event ("C" type, thread id 0) whose arguments are a copy of the given
    # mapping of counter names to values. Returns nothing.
    """Emits a counter record for the dictionary 'counters'.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      pid:  Identifier of the process generating this event as an integer.
      timestamp:  The timestamp of this event as a long integer.
      counters: Dictionary of counter values.
    """
    event = self._create_event('C', category, name, pid, 0, timestamp)
    event['args'] = counters.copy()
    self._events.append(event)

  def format_to_string(self, pretty: bool = False) -> str:
    # [FUNCTION DOC]
    # Serializes metadata events followed by regular events into a JSON string under a trace-events
    # key. If pretty (default false) is true the output is indented with four spaces, otherwise it
    # is compact. Returns the string.
    """Formats the chrome trace to a string.

    Args:
      pretty: (Optional.)  If True, produce human-readable JSON output.

    Returns:
      A JSON-formatted string in Chrome Trace format.
    """
    trace = {}
    trace['traceEvents'] = self._metadata + self._events
    if pretty:
      return json.dumps(trace, indent=4, separators=(',', ': '))
    else:
      return json.dumps(trace, separators=(',', ':'))


class _TensorTracker(object):
  # [CLASS DOC]
  # HELPER: internal record used by Timeline to track one tensor's name, trace object id, creating
  # process id, allocator, size and the lists of reference and unreference timestamps. Exposes
  # read-only properties and methods to add references. Not thread safe.
  """An internal class to track the lifetime of a Tensor."""

  def __init__(
      self,
      name: str,
      object_id: int,
      timestamp: int,
      pid: int,
      allocator: str,
      num_bytes: int,
  ) -> None:
    # [FUNCTION DOC]
    # Stores the tensor name, trace object id, creation timestamp, process id, allocator name and
    # byte count, and starts with empty reference and unreference timestamp lists. Returns nothing
    # (the docstring's mention of returning a tracker object is just the constructed instance).
    """Creates an object to track tensor references.

    This class is not thread safe and is intended only for internal use by
    the 'Timeline' class in this file.

    Args:
      name:  The name of the Tensor as a string.
      object_id:  Chrome Trace object identifier assigned for this Tensor.
      timestamp:  The creation timestamp of this event as a long integer.
      pid:  Process identifier of the associated device, as an integer.
      allocator:  Name of the allocator used to create the Tensor.
      num_bytes:  Number of bytes allocated (long integer).

    Returns:
      A 'TensorTracker' object.
    """
    self._name = name
    self._pid = pid
    self._object_id = object_id
    self._create_time = timestamp
    self._allocator = allocator
    self._num_bytes = num_bytes
    self._ref_times = []
    self._unref_times = []

  @property
  def name(self) -> str:
    # [FUNCTION DOC]
    # Read-only property: the tensor name.
    """Name of this tensor."""
    return self._name

  @property
  def pid(self) -> int:
    # [FUNCTION DOC]
    # Read-only property: process id of the tensors lane that holds this tensor.
    """ID of the process which created this tensor (an integer)."""
    return self._pid

  @property
  def create_time(self) -> int:
    # [FUNCTION DOC]
    # Read-only property: creation timestamp.
    """Timestamp when this tensor was created (long integer)."""
    return self._create_time

  @property
  def object_id(self) -> int:
    # [FUNCTION DOC]
    # Read-only property: trace object identifier.
    """Returns the object identifier of this tensor (integer)."""
    return self._object_id

  @property
  def num_bytes(self) -> int:
    # [FUNCTION DOC]
    # Read-only property: requested size in bytes.
    """Size of this tensor in bytes (long integer)."""
    return self._num_bytes

  @property
  def allocator(self) -> str:
    # [FUNCTION DOC]
    # Read-only property: allocator name.
    """Name of the allocator used to create this tensor (string)."""
    return self._allocator

  @property
  def last_unref(self) -> int:
    # [FUNCTION DOC]
    # Read-only property: the largest recorded unreference timestamp. Raises a value error if no
    # unreference was ever recorded.
    """Last unreference timestamp of this tensor (long integer)."""
    return max(self._unref_times)

  def add_ref(self, timestamp: int) -> None:
    # [FUNCTION DOC]
    # Records a reference timestamp (integer) in the reference list. Returns nothing.
    """Adds a reference to this tensor with the specified timestamp.

    Args:
      timestamp:  Timestamp of object reference as an integer.
    """
    self._ref_times.append(timestamp)

  def add_unref(self, timestamp: int) -> None:
    # [FUNCTION DOC]
    # Records an unreference timestamp (integer) in the unreference list. Returns nothing.
    """Adds an unref to this tensor with the specified timestamp.

    Args:
      timestamp:  Timestamp of object unreference as an integer.
    """
    self._unref_times.append(timestamp)


class Timeline(object):
  # [CLASS DOC]
  # Public entry point: converts a step-stats record (per-device node execution statistics) into
  # a Chrome Trace Format timeline. Holds the original and possibly rewritten step stats, an
  # optional graph, a trace formatter, process id and flow id counters, per-device process id
  # maps, tensor trackers and allocator maximums. Typical use: construct it, then call
  # analyze_step_stats or generate_chrome_trace_format. Not thread safe.
  """A class for visualizing execution timelines of TensorFlow steps."""

  def __init__(
      self, step_stats: step_stats_pb2.StepStats, graph: Optional[Any] = None
  ) -> None:
    # [FUNCTION DOC]
    # Stores the step stats proto (required) and optional graph (default none) and initializes all
    # bookkeeping: an empty trace formatter, counters at zero and empty maps for device and tensor
    # process ids, tensors, flow starts, allocation times and allocator maximums. Returns nothing.
    """Constructs a new Timeline.

    A 'Timeline' is used for visualizing the execution of a TensorFlow
    computation.  It shows the timings and concurrency of execution at
    the granularity of TensorFlow Ops.
    This class is not thread safe.

    Args:
      step_stats: The 'step_stats_pb2.StepStats' proto recording execution
        times.
      graph: (Optional) The 'Graph' that was executed.
    """

    self._origin_step_stats = step_stats
    self._step_stats = None
    self._graph = graph
    self._chrome_trace = _ChromeTraceFormatter()
    self._next_pid = 0
    self._device_pids = {}  # device name -> pid for compute activity.
    self._tensor_pids = {}  # device name -> pid for tensors.
    self._tensors = {}  # tensor_name -> TensorTracker
    self._next_flow_id = 0
    self._flow_starts = {}  # tensor_name -> (timestamp, pid, tid)
    self._alloc_times = {}  # tensor_name -> ( time, allocator, size )
    self._allocator_maximums = {}  # allocator name => maximum bytes long

  def _alloc_pid(self) -> int:
    # [FUNCTION DOC]
    # HELPER: hands out fresh fake process ids for trace lanes. Returns the next id and increments
    # the counter.
    """Allocate a process Id."""
    pid = self._next_pid
    self._next_pid += 1
    return pid

  def _alloc_flow_id(self) -> int:
    # [FUNCTION DOC]
    # HELPER: hands out fresh flow ids for dataflow arrows. Returns the next id and increments the
    # counter.
    """Allocate a flow Id."""
    flow_id = self._next_flow_id
    self._next_flow_id += 1
    return flow_id

  def _parse_op_label(
      self, label: str
  ) -> Tuple[str, str, List[str]]:
    # [FUNCTION DOC]
    # HELPER: parses an op timeline label of the form "name = op(arg, arg, ...)". Returns a triple
    # of node name, op type and list of input names (split on comma-space; empty list when there
    # are none). If the label does not match, returns unknown, unknown and an empty list.
    """Parses the fields in a node timeline label."""
    # Expects labels of the form: name = op(arg, arg, ...).
    match = re.match(r'(.*) = (.*)\((.*)\)', label)
    if match is None:
      return 'unknown', 'unknown', []
    nn, op, inputs = match.groups()
    if not inputs:
      inputs = []
    else:
      inputs = inputs.split(', ')
    return nn, op, inputs

  def _parse_kernel_label(self, label, node_name):
    # [FUNCTION DOC]
    # HELPER: extracts node name and op for GPU kernel records. If the label contains an annotation
    # between a double-at marker and a hash sign, that text replaces node_name. The node name is
    # then split on colon; returns a pair of name and op, with op being unknown if absent.
    # KNOWN ISSUE: the docstring is copied from the op label parser and does not describe this
    # kernel-label behavior.
    """Parses the fields in a node timeline label."""
    # Expects labels of the form: retval (arg) detail @@annotation
    start = label.find('@@')
    end = label.find('#')
    if start >= 0 and end >= 0 and start + 2 < end:
      node_name = label[start + 2 : end]
    # Node names should always have the form 'name:op'.
    fields = node_name.split(':') + ['unknown']
    name, op = fields[:2]
    return name, op

  def _assign_lanes(self) -> None:
    # [FUNCTION DOC]
    # HELPER: for each device in the (working) step stats, greedily assigns each node execution to
    # the first lane whose previous activity ended before this node starts, opening a new lane
    # when none fits, and stores the lane index as the node's thread id. Mutates the step stats.
    # Returns nothing.
    """Assigns non-overlapping lanes for the activities on each device."""
    for device_stats in self._step_stats.dev_stats:
      # TODO(pbar): Genuine thread IDs in step_stats_pb2.NodeExecStats
      # might be helpful.
      lanes = [0]
      for ns in device_stats.node_stats:
        l = -1
        for i, lts in enumerate(lanes):
          if ns.all_start_micros > lts:
            l = i
            lanes[l] = ns.all_start_micros + ns.all_end_rel_micros
            break
        if l < 0:
          l = len(lanes)
          lanes.append(ns.all_start_micros + ns.all_end_rel_micros)
        ns.thread_id = l

  def _emit_op(
      self, nodestats: step_stats_pb2.NodeExecStats, pid: int, is_gputrace: bool
  ) -> None:
    # [FUNCTION DOC]
    # HELPER: emits one region event for an op execution. Op type and inputs come from the kernel
    # label for GPU-trace devices, are fixed as RecvTensor for that node, or are parsed from the
    # op label otherwise. Arguments hold name, op, numbered inputs and (on ROCm builds) the kernel
    # name. Adds an event to the trace formatter; returns nothing.
    """Generates a Chrome Trace event to show Op execution.

    Args:
      nodestats: The 'step_stats_pb2.NodeExecStats' proto recording op
        execution.
      pid: The pid assigned for the device where this op ran.
      is_gputrace: If True then this op came from the GPUTracer.
    """
    node_name = nodestats.node_name
    start = nodestats.all_start_micros
    duration = nodestats.all_end_rel_micros
    tid = nodestats.thread_id
    inputs = []
    if is_gputrace:
      node_name, op = self._parse_kernel_label(
          nodestats.timeline_label, node_name
      )
    elif node_name == 'RecvTensor':
      # RPC tracing does not use the standard timeline_label format.
      op = 'RecvTensor'
    else:
      _, op, inputs = self._parse_op_label(nodestats.timeline_label)
    args = {'name': node_name, 'op': op}
    if build_info.build_info['is_rocm_build']:
      args['kernel'] = nodestats.timeline_label.split('@@')[0]
    for i, iname in enumerate(inputs):
      args['input%d' % i] = iname
    self._chrome_trace.emit_region(start, duration, pid, tid, 'Op', op, args)

  def _emit_tensor_snapshot(
      self,
      tensor: _TensorTracker,
      timestamp: int,
      pid: int,
      tid: int,
      value: step_stats_pb2.NodeOutput,
  ) -> None:
    # [FUNCTION DOC]
    # HELPER: emits an object-snapshot event for a tensor, whose payload is the text of the
    # output's tensor description with double quotes removed. Returns nothing.
    """Generate Chrome Trace snapshot event for a computed Tensor.

    Args:
      tensor: A 'TensorTracker' object.
      timestamp:  The timestamp of this snapshot as a long integer.
      pid: The pid assigned for showing the device where this op ran.
      tid: The tid of the thread computing the tensor snapshot.
      value: A JSON-compliant snapshot of the object.
    """
    desc = str(value.tensor_description).replace('"', '')
    snapshot = {'tensor_description': desc}
    self._chrome_trace.emit_obj_snapshot(
        'Tensor', tensor.name, timestamp, pid, tid, tensor.object_id, snapshot
    )

  def _produce_tensor(
      self,
      name: str,
      timestamp: int,
      tensors_pid: int,
      allocator: str,
      num_bytes: int,
  ) -> _TensorTracker:
    # [FUNCTION DOC]
    # HELPER: creates a tracker for a tensor using the current tensor count as its object id,
    # registers it by name (replacing any prior one) and returns it.
    """Creates a new tensor tracker."""
    object_id = len(self._tensors)
    tensor = _TensorTracker(
        name, object_id, timestamp, tensors_pid, allocator, num_bytes
    )
    self._tensors[name] = tensor
    return tensor

  def _is_gputrace_device(self, device_name: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: true when the device name contains a stream or memcpy marker, meaning its stats come
    # from the GPU tracer.
    """Returns true if this device is part of the GPUTracer logging."""
    return '/stream:' in device_name or '/memcpy' in device_name

  def _allocate_pids(self) -> None:
    # [FUNCTION DOC]
    # HELPER: allocates a process id for an Allocators lane and, per device, one for compute and
    # one for tensors; records them in the device maps and emits process-name metadata for each.
    # Returns nothing.
    """Allocate fake process ids for each device in the step_stats_pb2.StepStats."""
    self._allocators_pid = self._alloc_pid()
    self._chrome_trace.emit_pid('Allocators', self._allocators_pid)

    # Add processes in the Chrome trace to show compute and data activity.
    for dev_stats in self._step_stats.dev_stats:
      device_pid = self._alloc_pid()
      self._device_pids[dev_stats.device] = device_pid
      tensors_pid = self._alloc_pid()
      self._tensor_pids[dev_stats.device] = tensors_pid
      self._chrome_trace.emit_pid(dev_stats.device + ' Compute', device_pid)
      self._chrome_trace.emit_pid(dev_stats.device + ' Tensors', tensors_pid)

  def _analyze_tensors(self, show_memory: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: for every node output creates a tensor tracker (named by node name, with an index
    # suffix for non-first outputs), records its creation and end times, and remembers where the
    # dataflow starts. If show_memory is true, also emits object creation and snapshot events.
    # Returns nothing.
    """Analyze tensor references to track dataflow."""
    for dev_stats in self._step_stats.dev_stats:
      device_pid = self._device_pids[dev_stats.device]
      tensors_pid = self._tensor_pids[dev_stats.device]
      for node_stats in dev_stats.node_stats:
        tid = node_stats.thread_id
        node_name = node_stats.node_name
        start_time = node_stats.all_start_micros
        end_time = node_stats.all_start_micros + node_stats.all_end_rel_micros
        for index, output in enumerate(node_stats.output):
          if index:
            output_name = '%s:%d' % (node_name, index)
          else:
            output_name = node_name

          allocation = output.tensor_description.allocation_description
          num_bytes = allocation.requested_bytes
          allocator_name = allocation.allocator_name
          tensor = self._produce_tensor(
              output_name, start_time, tensors_pid, allocator_name, num_bytes
          )
          tensor.add_ref(start_time)
          tensor.add_unref(end_time)
          self._flow_starts[output_name] = (end_time, device_pid, tid)

          if show_memory:
            self._chrome_trace.emit_obj_create(
                'Tensor',
                output_name,
                start_time,
                tensors_pid,
                tid,
                tensor.object_id,
            )
            self._emit_tensor_snapshot(
                tensor, end_time - 1, tensors_pid, tid, output
            )

  def _show_compute(self, show_dataflow: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: emits an op region event for every node. For non-GPU, non-RecvTensor nodes it also
    # parses inputs, adds reference and unreference times to the matching tensor trackers (names
    # of inserted send/receive nodes are mapped back to the original node) and, if show_dataflow
    # is true, emits flow start/end pairs when producer and consumer differ in process or thread.
    # Missing tensors are logged at verbose level. Returns nothing.
    """Visualize the computation activity."""
    for dev_stats in self._step_stats.dev_stats:
      device_name = dev_stats.device
      device_pid = self._device_pids[device_name]
      is_gputrace = self._is_gputrace_device(device_name)

      for node_stats in dev_stats.node_stats:
        tid = node_stats.thread_id
        start_time = node_stats.all_start_micros
        end_time = node_stats.all_start_micros + node_stats.all_end_rel_micros
        self._emit_op(node_stats, device_pid, is_gputrace)

        if is_gputrace or node_stats.node_name == 'RecvTensor':
          continue

        _, _, inputs = self._parse_op_label(node_stats.timeline_label)
        for input_name in inputs:
          if input_name not in self._tensors:
            # This can happen when partitioning has inserted a Send/Recv.
            # We remove the numeric suffix so that the dataflow appears to
            # come from the original node.  Ideally, the StepStats would
            # contain logging for the Send and Recv nodes.
            index = input_name.rfind('/_')
            if index > 0:
              input_name = input_name[:index]

          if input_name in self._tensors:
            tensor = self._tensors[input_name]
            tensor.add_ref(start_time)
            tensor.add_unref(end_time - 1)

            if show_dataflow:
              # We use a different flow ID for every graph edge.
              create_time, create_pid, create_tid = self._flow_starts[
                  input_name
              ]
              # Don't add flows when producer and consumer ops are on the same
              # pid/tid since the horizontal arrows clutter the visualization.
              if create_pid != device_pid or create_tid != tid:
                flow_id = self._alloc_flow_id()
                self._chrome_trace.emit_flow_start(
                    input_name, create_time, create_pid, create_tid, flow_id
                )
                self._chrome_trace.emit_flow_end(
                    input_name, start_time, device_pid, tid, flow_id
                )
          else:
            logging.vlog(
                1, "Can't find tensor %s - removed by CSE?", input_name
            )

  def _show_memory_counters(self) -> None:
    # [FUNCTION DOC]
    # HELPER: emits a deletion event per tensor and builds per-allocator lists of allocation and
    # free events. Walks them in time order keeping a running byte total and live tensor set,
    # emits a memory counter per step, tracks the peak as an AllocationMaximum per allocator and
    # stores the result as the allocator maximums. Returns nothing.
    """Produce a counter series for each memory allocator."""
    # Iterate over all tensor trackers to build a list of allocations and
    # frees for each allocator. Then sort the lists and emit a cumulative
    # counter series for each allocator.
    allocations = {}
    for name in self._tensors:
      tensor = self._tensors[name]
      self._chrome_trace.emit_obj_delete(
          'Tensor', name, tensor.last_unref, tensor.pid, 0, tensor.object_id
      )
      allocator = tensor.allocator
      if allocator not in allocations:
        allocations[allocator] = []
      num_bytes = tensor.num_bytes
      allocations[allocator].append((tensor.create_time, num_bytes, name))
      allocations[allocator].append((tensor.last_unref, -num_bytes, name))

    alloc_maxes = {}

    # Generate a counter series showing total allocations for each allocator.
    for allocator in allocations:
      alloc_list = allocations[allocator]
      alloc_list.sort()
      total_bytes = 0
      alloc_tensor_set = set()
      alloc_maxes[allocator] = AllocationMaximum(
          timestamp=0, num_bytes=0, tensors=set()
      )
      for time, num_bytes, name in sorted(
          alloc_list, key=lambda allocation: allocation[0]
      ):
        total_bytes += num_bytes
        if num_bytes < 0:
          alloc_tensor_set.discard(name)
        else:
          alloc_tensor_set.add(name)

        if total_bytes > alloc_maxes[allocator].num_bytes:
          alloc_maxes[allocator] = AllocationMaximum(
              timestamp=time,
              num_bytes=total_bytes,
              tensors=copy.deepcopy(alloc_tensor_set),
          )

        self._chrome_trace.emit_counter(
            'Memory',
            allocator,
            self._allocators_pid,
            time,
            allocator,
            total_bytes,
        )
    self._allocator_maximums = alloc_maxes

  def _preprocess_op_time(self, op_time: str) -> None:
    # [FUNCTION DOC]
    # HELPER: chooses the working step stats. For schedule it uses the original as is. Otherwise
    # it deep-copies it, finds each op's earliest GPU kernel start and latest kernel end, and
    # rewrites op times: gpu mode uses kernel start through the later end, all mode keeps the
    # scheduling start. Returns nothing.
    """Update the start and end time of ops in step stats.

    Args:
      op_time: How the execution time of op is shown in timeline. Possible
        values are "schedule", "gpu" and "all".  "schedule" will show op from
        the time it is scheduled to the end of the scheduling. Notice by the end
        of its scheduling its async kernels may not start yet. It is shown using
        the default value from step_stats. "gpu" will show op with the execution
        time of its kernels on GPU. "all" will show op from the start of its
        scheduling to the end of its last kernel.
    """
    if op_time == 'schedule':
      self._step_stats = self._origin_step_stats
      return
    self._step_stats = copy.deepcopy(self._origin_step_stats)
    # Separate job task and gpu tracer stream
    stream_all_stats = []
    job_stats = []
    for stats in self._step_stats.dev_stats:
      if '/stream:all' in stats.device:
        stream_all_stats.append(stats)
      elif '/job' in stats.device:
        job_stats.append(stats)

    # Record the start time of the first kernel and the end time of
    # the last gpu kernel for all ops.
    op_gpu_start = {}
    op_gpu_end = {}
    for stats in stream_all_stats:
      for kernel in stats.node_stats:
        name, _ = self._parse_kernel_label(
            kernel.timeline_label, kernel.node_name
        )
        start = kernel.all_start_micros
        end = kernel.all_start_micros + kernel.all_end_rel_micros
        if name in op_gpu_start:
          op_gpu_start[name] = min(op_gpu_start[name], start)
          op_gpu_end[name] = max(op_gpu_end[name], end)
        else:
          op_gpu_start[name] = start
          op_gpu_end[name] = end

    # Update the start and end time of each op according to the op_time
    for stats in job_stats:
      for op in stats.node_stats:
        if op.node_name in op_gpu_start:
          end = max(
              op_gpu_end[op.node_name],
              op.all_start_micros + op.all_end_rel_micros,
          )
          if op_time == 'gpu':
            op.all_start_micros = op_gpu_start[op.node_name]
          op.all_end_rel_micros = end - op.all_start_micros

  def analyze_step_stats(
      self,
      show_dataflow: bool = True,
      show_memory: bool = True,
      op_time: str = 'schedule',
  ) -> StepStatsAnalysis:
    # [FUNCTION DOC]
    # Runs the full analysis: preprocess op times, allocate process ids, assign lanes, analyze
    # tensors, show compute and (if show_memory) memory counters. Inputs: show_dataflow (default
    # true), show_memory (default true) and op_time (schedule, gpu or all; default schedule).
    # Returns a StepStatsAnalysis with the trace formatter and allocator maximums. Mutates
    # internal state, so it is meant to be called once per Timeline.
    """Analyze the step stats and format it into Chrome Trace Format.

    Args:
      show_dataflow: (Optional.) If True, add flow events to the trace
        connecting producers and consumers of tensors.
      show_memory: (Optional.) If True, add object snapshot events to the trace
        showing the sizes and lifetimes of tensors.
      op_time: (Optional.) How the execution time of op is shown in timeline.
        Possible values are "schedule", "gpu" and "all". "schedule" will show op
        from the time it is scheduled to the end of the scheduling. Notice by
        the end of its scheduling its async kernels may not start yet. It is
        shown using the default value from step_stats. "gpu" will show op with
        the execution time of its kernels on GPU. "all" will show op from the
        start of its scheduling to the end of its last kernel.

    Returns:
      A 'StepStatsAnalysis' object.
    """
    self._preprocess_op_time(op_time)
    self._allocate_pids()
    self._assign_lanes()
    self._analyze_tensors(show_memory)
    self._show_compute(show_dataflow)
    if show_memory:
      self._show_memory_counters()
    return StepStatsAnalysis(
        chrome_trace=self._chrome_trace,
        allocator_maximums=self._allocator_maximums,
    )

  def generate_chrome_trace_format(
      self,
      show_dataflow: bool = True,
      show_memory: bool = False,
      op_time: str = 'schedule',
  ) -> str:
    # [FUNCTION DOC]
    # Runs the step stats analysis with the given options (show_dataflow default true, show_memory
    # default false, op_time default schedule) and returns the trace as a pretty-printed JSON
    # string in Chrome Trace Format.
    # pyformat: disable
    """Produces a trace in Chrome Trace Format.

    Args:
      show_dataflow: (Optional.) If True, add flow events to the trace
        connecting producers and consumers of tensors.
      show_memory: (Optional.) If True, add object snapshot events to the trace
        showing the sizes and lifetimes of tensors.
      op_time: (Optional.) How the execution time of op is shown in timeline.
        Possible values are "schedule", "gpu" and "all".
        "schedule" will show op from the time it is scheduled to the end of
          the scheduling.
          Notice by the end of its scheduling its async kernels may not start
          yet. It is shown using the default value from step_stats.
        "gpu" will show op with the execution time of its kernels on GPU.
        "all" will show op from the start of its scheduling to the end of
          its last kernel.
    Returns:
      A JSON formatted string in Chrome Trace format.
    """
    # pyformat: enable
    step_stats_analysis = self.analyze_step_stats(
        show_dataflow=show_dataflow, show_memory=show_memory, op_time=op_time
    )

    return step_stats_analysis.chrome_trace.format_to_string(pretty=True)
