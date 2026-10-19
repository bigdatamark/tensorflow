# [MODULE DOC]
# Implements the dumping op callback behind the tfdbg v2 API for enabling and disabling dump of
# debug information. Exposes the public enable and disable functions (exported under the
# debugging experimental namespace) and a helper to detect function op types. The internal callback
# class traces graph op creation and eager/graph execution, writing debug events (graphs, op
# creations, executions, stack frames, source files) via the debug-events writer. State is kept in
# a thread-local. Import-time side effect: registers the gradient of the DebugIdentityV2 op.
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
"""Dumping op callbacks: Enables dump-based features in tfdbg v2."""

import atexit
import os
import re
import socket
import threading
import uuid

from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export

_state = threading.local()
DEFAULT_TENSOR_DEBUG_MODE = "NO_TENSOR"

# pylint:disable=protected-access
_FUNCTION_PREFIXES = (
    compat.as_bytes(function_lib._FORWARD_PREFIX),
    compat.as_bytes(function_lib._BACKWARD_PREFIX),
    compat.as_bytes(function_lib._INFERENCE_PREFIX))
# pylint:enable=protected-access


def is_op_type_function(op_type):
  # [FUNCTION DOC]
  # Returns true if the op type (string or bytes) starts with one of the function name prefixes
  # (forward, backward, inference) used for eagerly executed function graphs. Pure check.
  return compat.as_bytes(op_type).startswith(_FUNCTION_PREFIXES)


@ops.RegisterGradient("DebugIdentityV2")
def _debug_identity_v2_grad(op, dy):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the DebugIdentityV2 op. Ignores the op and passes the
  # incoming gradient through unchanged. Registration happens at import time.
  """Gradient function for the DebugIdentityV2 op."""
  del op  # Unused
  return dy


def _get_tfdbg_run_id():
  # [FUNCTION DOC]
  # HELPER: returns a short run ID: the first eight characters of a random identifier.
  return str(uuid.uuid4())[:8]


def _get_id():
  # [FUNCTION DOC]
  # HELPER: returns a new random unique ID string (full random identifier). No arguments.
  """Get a short unique ID."""
  return str(uuid.uuid4())


def _concrete_tensor_to_proto(tensor):
  # [FUNCTION DOC]
  # HELPER: converts an eager tensor to a tensor protocol buffer by taking its numpy value.
  return tensor_util.make_tensor_proto(tensor.numpy())


class _DumpingCallback(object):
  # [CLASS DOC]
  # HELPER: internal implementation behind the public enable/disable functions.
  # Holds the state of one thread's dumping callback: configuration (dump root, debug mode, buffer
  # size, op regex, dtype filter), ID maps for graphs and stack frames, source file list, tensor
  # aliases (v1 graph mode) and a lazily created debug-events writer. Its callback method is
  # registered as an op callback; the function callback as a concrete-function callback. Thread-safety
  # is provided by several locks.
  """An object holding the states surrounding the dumping callback."""

  def __init__(self,
               dump_root,
               tensor_debug_mode,
               circular_buffer_size,
               op_regex,
               tensor_dtypes):
    # [FUNCTION DOC]
    # Constructor. Stores the dump root, debug mode, circular buffer size, op regular expression and
    # tensor dtype filter, generates a short run ID, records the host name, and initializes empty
    # state: source file list, stack-frame and context ID maps, function-to-graph map, op-type cache,
    # symbolic tensor counter, tensor alias map, placeholder debug-tensor map, locks, and no writer.
    self._dump_root = dump_root
    self._tfdbg_run_id = _get_tfdbg_run_id()
    self._tensor_debug_mode = tensor_debug_mode
    self._circular_buffer_size = circular_buffer_size
    self._op_regex = op_regex
    self._tensor_dtypes = tensor_dtypes

    self._hostname = socket.gethostname()
    # A list of source-file paths.
    self._source_file_paths = []
    # A map from stack frame (FileLineCol) to unique ID.
    self._stack_frame_to_id = dict()
    # Mapping op context to unique ID.
    self._context_to_id = dict()
    self._function_to_graph_id = dict()
    self._op_type_to_context_id = dict()
    # Keeps track of counter for symbolic tensors output by in-graph ops.
    # It is used to make unique names for debugger-generated tensors.
    self._symbolic_tensor_counter = 0
    # A map from the names of debugger-generated Identity and DebugIdentityV2
    # tensors to the names of the original insrumented graph tensors. This is
    # applicable to v1 graph mode only.
    self._tensor_aliases = dict()
    self._source_file_paths_lock = threading.Lock()
    self._stack_frame_to_id_lock = threading.Lock()
    self._context_lock = threading.Lock()
    self._symbolic_tensor_counter_lock = threading.Lock()
    # A dict mapping Placeholder tensors to their instrumenting debug tensors.
    # Used only under V1 graph mode, where we can't rely on auto control
    # dependency to execute the debug tensors and hence need to attach the debug
    # tensors as control dependencies of the ops that consume the Placeholder.
    self._placeholder_to_debug_tensor = (
        object_identity.ObjectIdentityDictionary())
    self._writer = None

  def function_callback(self, function):
    # [FUNCTION DOC]
    # Callback invoked when a concrete function is created. Takes the function, determines the ID of
    # its graph (possibly writing a debugged-graph event) and, under the context lock, records the
    # mapping from the function to the graph ID. The function is held strongly (possible memory
    # growth until the callback is disabled). Returns nothing.
    """A callback to be called on creation of ConcreteFunctions."""
    graph_id = self._get_context_id(function.graph)
    with self._context_lock:
      # NOTE(cais): We currently store the function (ConcreteFunction)
      # as keys of this dict, because weakrefs to them sometimes become
      # unreferenceable by the time the op callback is called. This approach
      # may cause memory leaks due to the holding of the functions. If that's
      # the case, calling `tf.debugging.disable_dump_debug_info()` should
      # cause GC of this object and this dict.
      self._function_to_graph_id[function] = graph_id

  @property
  def dump_root(self):
    # [FUNCTION DOC]
    # Read-only getter: the directory path where debug events are written.
    return self._dump_root

  @dump_root.setter
  def dump_root(self, dump_root):
    # [FUNCTION DOC]
    # Setter for the dump root directory. If the new value differs, stores it and discards the cached
    # writer so a new one is created lazily for the new directory. Returns nothing.
    if self._dump_root != dump_root:
      self._dump_root = dump_root
      self._writer = None

  @property
  def tfdbg_run_id(self):
    # [FUNCTION DOC]
    # Read-only property: the short random run ID generated for this callback.
    return self._tfdbg_run_id

  @property
  def tensor_debug_mode(self):
    # [FUNCTION DOC]
    # Read-only property: the configured tensor debug mode (enum value).
    return self._tensor_debug_mode

  @property
  def circular_buffer_size(self):
    # [FUNCTION DOC]
    # Read-only property: the configured circular buffer size for execution events.
    return self._circular_buffer_size

  def get_writer(self):
    # [FUNCTION DOC]
    # Returns the debug-events writer for the current dump root, lazily creating it (with the run ID
    # and circular buffer size) on first use or after the dump root changed. Takes no arguments.
    """Get the debug events writer for the currently configured dump root."""
    if not self._writer:
      self._writer = debug_events_writer.DebugEventsWriter(
          self._dump_root,
          self._tfdbg_run_id,
          circular_buffer_size=self._circular_buffer_size)
    return self._writer

  def _get_context_id(self, context):
    # [FUNCTION DOC]
    # HELPER: assigns IDs to graphs/contexts.
    # Takes a hashable context (such as a graph). Returns its existing unique ID, or, for a new one,
    # creates an ID under the lock using double-checked locking, and writes a debugged-graph event
    # (graph ID, name if any, outer context ID) through the writer. Mutates the context-to-ID map.
    """Get a unique ID for an op-construction context (e.g., a graph).

    If the graph has been encountered before, reuse the same unique ID.
    When encountering a new context (graph), this methods writes a DebugEvent
    proto with the debugged_graph field to the proper DebugEvent file.

    Args:
      context: A context to get the unique ID for. Must be hashable. E.g., a
        Graph object.

    Returns:
      A unique ID for the context.
    """
    # Use the double-checked lock pattern to optimize the common case.
    if context in self._context_to_id:  # 1st check, without lock.
      return self._context_to_id[context]
    graph_is_new = False
    with self._context_lock:
      if context not in self._context_to_id:  # 2nd check, with lock.
        graph_is_new = True
        context_id = _get_id()
        self._context_to_id[context] = context_id
    if graph_is_new:
      self.get_writer().WriteDebuggedGraph(debug_event_pb2.DebuggedGraph(
          graph_id=context_id,
          graph_name=getattr(context, "name", None),
          outer_context_id=self._get_outer_context_id(context)))
    return self._context_to_id[context]

  def _get_outer_context_id(self, graph):
    # [FUNCTION DOC]
    # HELPER: used when writing debugged-graph events.
    # KNOWN ISSUE: the docstring says it returns the outer context name, but it returns the unique ID
    # of the outer context.
    # Takes a graph; if it has a non-empty outer graph returns that graph's context ID (which may write
    # a new debugged-graph event), otherwise returns nothing/null.
    """Get the ID of the immediate outer context of the input graph.

    Args:
      graph: The graph (context) in question.

    Returns:
      If an outer context exists, the immediate outer context name as a string.
      If such as outer context does not exist (i.e., `graph` is itself
      outermost), `None`.
    """
    if hasattr(graph, "outer_graph") and graph.outer_graph:
      return self._get_context_id(graph.outer_graph)
    else:
      return None

  def _write_source_file_content(self, file_path):
    # [FUNCTION DOC]
    # HELPER: records source files for stack frames.
    # Takes a file path. If the file was already written returns its index. Otherwise (under a lock)
    # loads its lines when it is an uncompiled Python source (logging a warning on read errors),
    # writes a source-file event with host name and lines (possibly none), appends the path to the
    # list and returns its integer index.
    """Send the content of a source file via debug-events writer.

    Args:
      file_path: Path to the source file.

    Returns:
      An int index for the file.
    """
    if file_path in self._source_file_paths:
      return self._source_file_paths.index(file_path)
    with self._source_file_paths_lock:
      if file_path not in self._source_file_paths:
        lines = None
        if source_utils.is_extension_uncompiled_python_source(file_path):
          try:
            lines, _ = source_utils.load_source(file_path)
          except IOError as e:
            logging.warn(
                "Failed to read source code from path: %s. Reason: %s",
                file_path, e)
        writer = self.get_writer()
        writer.WriteSourceFile(debug_event_pb2.SourceFile(
            file_path=file_path, host_name=self._hostname, lines=lines))
        self._source_file_paths.append(file_path)
      return self._source_file_paths.index(file_path)

  def _process_stack_frames(self):
    # [FUNCTION DOC]
    # HELPER: captures the current Python stack for event records.
    # KNOWN ISSUE: the docstring says it returns a list of stack frame IDs, but it actually returns a
    # code location protocol buffer holding the host name and the list of IDs.
    # Takes no arguments. Extracts the stack, and for each frame not yet seen assigns a new ID, writes
    # the source file content and a stack-frame event through the writer (under a lock). Reuses IDs for
    # known frames. Returns the code location message.
    """Process stack frames.

    Send the content of source-files, on a best-effort basis.

    Returns:
      A list of stack frame IDs.
    """
    stack_frames = tf_stack.extract_stack()
    stack_frame_ids = []
    writer = None
    for file_path, lineno, func, _ in stack_frames:
      abs_path = os.path.abspath(file_path)
      if (abs_path, lineno, func) in self._stack_frame_to_id:
        stack_frame_ids.append(
            self._stack_frame_to_id[(abs_path, lineno, func)])
        continue
      with self._stack_frame_to_id_lock:
        if (abs_path, lineno, func) not in self._stack_frame_to_id:
          stack_frame_id = _get_id()
          self._stack_frame_to_id[(abs_path, lineno, func)] = stack_frame_id
          file_index = self._write_source_file_content(abs_path)
          file_line_col = graph_debug_info_pb2.GraphDebugInfo.FileLineCol(
              file_index=file_index, line=lineno, func=func)
          stack_frame_with_id = debug_event_pb2.StackFrameWithId(
              id=stack_frame_id, file_line_col=file_line_col)
          writer = self.get_writer()
          writer.WriteStackFrameWithId(stack_frame_with_id)
        stack_frame_ids.append(
            self._stack_frame_to_id[(abs_path, lineno, func)])

    code_location = debug_event_pb2.CodeLocation(
        host_name=self._hostname, stack_frame_ids=stack_frame_ids)
    return code_location

  def _process_v1_graph_mode_tensor(self,
                                    op_type,
                                    tensor,
                                    debug_tensor,
                                    tensor_debug_mode):
    # [FUNCTION DOC]
    # HELPER: v1 graph mode glue for the symbolic tensor instrumentation.
    # Inputs: op type, original tensor, its debug tensor, debug mode. For placeholder ops it records
    # the debug tensor for later control dependencies and returns the original tensor. Otherwise, in
    # full-tensor mode for non-constant ops it records an alias and returns the debug tensor itself;
    # else it creates an identity tensor with a control dependency on the debug op, records an alias
    # and returns the identity. Mutates the alias and placeholder maps.
    """For V1 graph mode, determine what tensor to output from callback.

    Args:
      op_type: Type of the op that outputs the original symbolic tensor.
      tensor: The original output symbolic tensor.
      debug_tensor: The debugger-instrumented tensor.
      tensor_debug_mode: Debug mode used, a tfdbg TensorDebugMode enum.

    Returns:
      A symbolic tensor to be returned by the dumping op_callback.
    """
    # Placeholders need special treatment under V1 graph mode. The
    # callback can't simply override the Placeholder tensor to a debug tensor,
    # as that would cause the Placeholder op to lack a value.
    if op_type in ("Placeholder", "PlaceholderWithDefault"):
      self._placeholder_to_debug_tensor[tensor] = debug_tensor
      return tensor
    else:
      # TODO(cais): Evaluate performance optimization options. For the
      # `NO_TENSOR` debug mode, an alternative is to add `debug_tensor` as a
      # control dependency of `tensor.op` without an additional identity op.
      if (tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR and
          op_type != "Const"):
        # NOTE(b/153716279): Under v1 graph mode, overriding the output tensor
        # of Const ops can lead to downstream errors related to shapes. We opt
        # to use an identity op to avoid this issue at the cost of slightly
        # larger graph size.
        self._tensor_aliases[debug_tensor.name] = tensor.name
        return debug_tensor
      else:
        with self._symbolic_tensor_counter_lock:
          identity_name = "tfdbg_identity_%d" % self._symbolic_tensor_counter
        identity = array_ops.identity(tensor, name=identity_name)
        identity.op._add_control_input(  # pylint: disable=protected-access
            debug_tensor.op)
        self._tensor_aliases[identity.name] = tensor.name
        return identity

  def _instrument_symbolic_tensors(self,
                                   tensors,
                                   op_type,
                                   op_name,
                                   tfdbg_context_id,
                                   tensor_ids):
    # [FUNCTION DOC]
    # HELPER: instruments graph-mode tensors for the op callback.
    # Inputs: tuple of output tensors, op type, op name, graph context ID, and tensor IDs (one per
    # output). For each tensor, according to the debug mode, adds a debug-identity op (fed by an empty
    # constant, a numeric summary, or the tensor itself) unless the tensor is filtered out by the
    # dump filter or has an unsupported dtype. In v1 graph mode it returns a list of replacement
    # tensors (original ones when not instrumented); otherwise returns nothing since automatic
    # control dependencies suffice. Raises not-implemented for unsupported modes.
    """Add debugging instrumentation for symbolic (i.e., non-eager) tensors.

    The detailed fashion in which the tensors are instrumented is determined
    by the tensor_debug_mode configured for the currently enabled dumping
    callback.

    Args:
      tensors: A tuple of Tensors to instrument. It is assumed that their
        ordering corresponds to the ordering of output tensors of an original
        op. Output slot indices (0-based) will be generated based on the
        ordering.
      op_type: Type name of the op that emits the Tensors (e.g., "MatMul").
      op_name: Name of the op that emits the Tensors (e.g., "dense_1/MatMul").
      tfdbg_context_id: A unique ID for the context that the op belongs to
        (e.g., a graph).
      tensor_ids: A list of unique ID numbers for the tensors, for tfdbg's
        internal use.

    Returns:
      Non-eager Tensors that override the `tensors` as the output of the op
      that originally generated `tensors`. In some cases (e.g., non-V1 graph
      mode), this may be `None`, as the instrumentation can simply rely on
      automatic control dependencies (see `auto_control_deps.py`) instead of
      tensor overriding.
    """
    tensor_debug_mode = self._tensor_debug_mode
    debug_urls = ["file://%s" % self._dump_root]
    is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
    instrumented_tensors = [] if is_v1_graph_mode else None
    for output_slot, tensor in enumerate(tensors):
      with self._symbolic_tensor_counter_lock:
        debug_identity_name = ("DebugIdentityV2_%d" %
                               self._symbolic_tensor_counter)
      debug_identity_op_kwargs = {
          "tfdbg_context_id": tfdbg_context_id,
          "op_name": op_name,
          "output_slot": output_slot,
          "tensor_debug_mode": self._tensor_debug_mode,
          "debug_urls": debug_urls,
          "name": debug_identity_name,
          "circular_buffer_size": self._circular_buffer_size,
          "tfdbg_run_id": self._tfdbg_run_id,
      }
      if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
        if (not self._should_dump_tensor(op_type, tensor.dtype) or
            not tensor.dtype.is_numpy_compatible):
          if is_v1_graph_mode:
            instrumented_tensors.append(tensor)
          continue
        if is_v1_graph_mode and not tensor.dtype.is_numpy_compatible:
          # Avoid instrumenting Placeholder under is_v1_graph_mode. Doing that
          # would cause runtime complaint about Placeholders not being fed.
          instrumented_tensors.append(tensor)
          continue
        # Except in V1 graph mode + control flow, debug_identity_v2 triggers
        # auto control dependency because it's a stateful op.
        debug_tensor = gen_debug_ops.debug_identity_v2(
            # Use an empty (shape=[0]) float32 tensor for the NO_TENSOR mode
            # as a low-overhead placeholder, since no actual tensor value is
            # traced.
            constant_op.constant([], dtype=dtypes.float32),
            **debug_identity_op_kwargs)
        if is_v1_graph_mode:
          instrumented_tensors.append(self._process_v1_graph_mode_tensor(
              op_type, tensor, debug_tensor, tensor_debug_mode))
      elif tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH,
                                 debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
                                 debug_event_pb2.TensorDebugMode.FULL_HEALTH,
                                 debug_event_pb2.TensorDebugMode.SHAPE):
        dtype = tensor.dtype
        dtype_is_dumpable = (
            tensor_debug_mode in (
                debug_event_pb2.TensorDebugMode.CURT_HEALTH,
                debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
                debug_event_pb2.TensorDebugMode.FULL_HEALTH) and
            dtype.is_floating or
            tensor_debug_mode == debug_event_pb2.TensorDebugMode.SHAPE and
            (dtype.is_floating or dtype.is_integer or dtype.is_bool))
        if (not self._should_dump_tensor(op_type, tensor.dtype) or
            not dtype_is_dumpable):
          if is_v1_graph_mode:
            instrumented_tensors.append(tensor)
          continue
        debug_tensor = gen_debug_ops.debug_identity_v2(
            gen_debug_ops.debug_numeric_summary_v2(
                tensor,
                tensor_id=tensor_ids[output_slot],
                tensor_debug_mode=self._tensor_debug_mode,
                output_dtype=dtypes.float64), **debug_identity_op_kwargs)
        if is_v1_graph_mode:
          instrumented_tensors.append(self._process_v1_graph_mode_tensor(
              op_type, tensor, debug_tensor, tensor_debug_mode))
      elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
        if (not self._should_dump_tensor(op_type, tensor.dtype) or
            not tensor.dtype.is_numpy_compatible):
          # Instrumenting DT_VARIANT and DT_RESOURCE type tensors under
          # V1 graph mode is known to have issues. TODO(cais): Investigate.
          if is_v1_graph_mode:
            instrumented_tensors.append(tensor)
          continue
        debug_tensor = gen_debug_ops.debug_identity_v2(
            tensor, **debug_identity_op_kwargs)
        if is_v1_graph_mode:
          instrumented_tensors.append(self._process_v1_graph_mode_tensor(
              op_type, tensor, debug_tensor, tensor_debug_mode))
      else:
        raise NotImplementedError(
            "Symbolic tensor instrumentation is not implemented for debug mode "
            "%s" % self._tensor_debug_mode)
    return instrumented_tensors

  def _dump_eager_tensors(self,
                          tensors,
                          op_type,
                          input_tensor_ids,
                          output_tensor_device_ids,
                          graph_id=None):
    # [FUNCTION DOC]
    # HELPER: builds the execution record for eagerly executed ops.
    # Inputs: eager output tensors, op type string, input tensor IDs, device IDs of the outputs (same
    # length as tensors, asserted), and optional graph ID. Returns an execution protocol buffer with
    # the code location. Depending on the debug mode, it also attaches tensor protos: nothing for
    # no-tensor mode; health summaries for floating outputs (empty placeholder proto otherwise);
    # shape summaries for float, integer and bool outputs; the full value for full-tensor mode.
    # Tensors filtered out by the dump filter or non-numpy-compatible are skipped. Raises
    # not-implemented for unsupported modes.
    """Dump the value of eager tensors.

    The destination of the dumping is determined by the dump_root of the
    currently enabled dumping callback. The tensors may be transformed prior to
    dumping (e.g., reduced as summary statistics such as minimum, maximum and
    arithmetic  mean). The details of this transformation (if any) depends on
    the tensor_debug_mode of the currently enabled dumping callback.

    Args:
      tensors: The EagerTensors whose values are to be dumped, with or without
        value transform.
      op_type: Type of the op that generates the tensors, as a string.
      input_tensor_ids: IDs of the input EagerTensors to the op.
      output_tensor_device_ids: Debugged-generated IDs for the devices on which
        the output tensors are allocated, as a `list` of `int`s. Must match
        `tensors` in length.
      graph_id: ID of the executed graph, applicable only to eager execution of
        a FuncGraph.

    Returns:
      A tfdbg Execution protocol buffer.
    """
    tensor_debug_mode = self._tensor_debug_mode
    output_tensor_ids = [
        t._id for t in tensors]  # pylint:disable=protected-access
    assert len(tensors) == len(output_tensor_device_ids)
    if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
      return debug_event_pb2.Execution(
          op_type=op_type,
          graph_id=graph_id,
          num_outputs=len(tensors),
          input_tensor_ids=input_tensor_ids,
          output_tensor_ids=output_tensor_ids,
          output_tensor_device_ids=output_tensor_device_ids,
          tensor_debug_mode=tensor_debug_mode,
          code_location=self._process_stack_frames())
    elif tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH,
                               debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
                               debug_event_pb2.TensorDebugMode.FULL_HEALTH,
                               debug_event_pb2.TensorDebugMode.SHAPE,
                               debug_event_pb2.TensorDebugMode.FULL_TENSOR):
      execution_proto = debug_event_pb2.Execution(
          op_type=op_type,
          num_outputs=len(tensors),
          graph_id=graph_id,
          input_tensor_ids=input_tensor_ids,
          output_tensor_ids=output_tensor_ids,
          output_tensor_device_ids=output_tensor_device_ids,
          tensor_debug_mode=tensor_debug_mode,
          code_location=self._process_stack_frames())
      for tensor in tensors:
        if (self._should_dump_tensor(op_type, tensor.dtype) and
            tensor.dtype.is_numpy_compatible):
          if tensor_debug_mode in (
              debug_event_pb2.TensorDebugMode.CURT_HEALTH,
              debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
              debug_event_pb2.TensorDebugMode.FULL_HEALTH):
            if tensor.dtype.is_floating:
              tensor_proto = _concrete_tensor_to_proto(
                  gen_debug_ops.debug_numeric_summary_v2(
                      tensor,
                      tensor_debug_mode=tensor_debug_mode,
                      output_dtype=dtypes.float64))
            else:
              # A placeholder for non-floating-type output tensors.
              tensor_proto = tensor_pb2.TensorProto()
          elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.SHAPE:
            if (tensor.dtype.is_floating or tensor.dtype.is_integer or
                tensor.dtype.is_bool):
              tensor_proto = _concrete_tensor_to_proto(
                  gen_debug_ops.debug_numeric_summary_v2(
                      tensor,
                      tensor_debug_mode=tensor_debug_mode,
                      output_dtype=dtypes.float64))
            else:
              # A placeholder for non-floating-type output tensors.
              tensor_proto = tensor_pb2.TensorProto()
          elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
            tensor_proto = _concrete_tensor_to_proto(tensor)
          if tensor_proto:
            execution_proto.tensor_protos.append(tensor_proto)
      return execution_proto
    else:
      raise NotImplementedError(
          "Tensor instrumentation is not implemented for debug mode %s yet " %
          self._tensor_debug_mode)

  def callback(self,
               op_type,
               inputs,
               attrs,
               outputs,
               op_name=None,
               graph=None):
    # [FUNCTION DOC]
    # The op callback registered with the framework. Inputs: op type, input tensors, attributes
    # (ignored), output tensors, optional op name, optional graph (set when the op is being built in
    # a graph, nothing when executed eagerly).
    # Graph case: determines the context ID, allocates output tensor IDs, derives the op name from
    # the first output for constant/placeholder ops, in v1 graph mode adds control inputs from
    # placeholder debug tensors, writes a graph-op-creation event with the code location, and (unless
    # the op type is in the skip list) returns the instrumented replacement tensors (may be nothing).
    # Eager case: returns nothing for the numeric-summary op and skip-listed ops; otherwise finds the
    # function graph ID if any, registers output devices and writes an execution event produced by
    # the eager dumping routine. Side effects are writes through the debug-events writer.
    """Op callback for tracing (dumping) a TF program's execution."""
    del attrs  # Unused

    writer = self.get_writer()
    if graph:
      is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
      context_id = self._get_context_id(graph)  # Innermost context ID.
      output_tensor_ids = self._get_symbolic_tensor_ids(len(outputs))
      if op_type in ("Const", "Placeholder", "PlaceholderWithDefault"):
        # In some cases, the op name of a Const or Placeholder op in a graph
        # can be duplicate (e.g., `None` or "resource").
        # When this happens, we use the output tensor name to infer
        # the non-duplicated tensor name.
        op_name = outputs[0].name.split(":")[0]
      if is_v1_graph_mode:
        for input_tensor in inputs:
          if input_tensor in self._placeholder_to_debug_tensor and outputs:
            outputs[0].op._add_control_input(  # pylint: disable=protected-access
                self._placeholder_to_debug_tensor[input_tensor].op)
      graph_op_creation = debug_event_pb2.GraphOpCreation(
          op_type=op_type,
          op_name=op_name,
          graph_name=graph.name if hasattr(graph, "name") else None,
          graph_id=context_id,
          input_names=[
              self._lookup_tensor_name(input_tensor) for input_tensor in inputs
          ],
          num_outputs=len(outputs),
          output_tensor_ids=output_tensor_ids,
          code_location=self._process_stack_frames())
      writer.WriteGraphOpCreation(graph_op_creation)
      if outputs and compat.as_bytes(
          op_type) not in op_callbacks_common.OP_CALLBACK_SKIP_OPS:
        return self._instrument_symbolic_tensors(
            outputs, op_type, op_name, context_id, output_tensor_ids)
    else:
      op_type_bytes = compat.as_bytes(op_type)
      if op_type_bytes == b"DebugNumericSummaryV2":
        # TODO(b/140334369): Remove this special casing logic once op_callback.
        # automatically prevents infinite recursion in eager mode.
        return None
      if op_type_bytes in op_callbacks_common.OP_CALLBACK_SKIP_OPS:
        return None
      context_id = self._func_graph_id_from_func_name(op_type)
      input_ids = [t._id for t in inputs]  # pylint:disable=protected-access
      output_tensor_device_ids = [writer.RegisterDeviceAndGetId(output.device)
                                  for output in outputs] if outputs else []
      writer.WriteExecution(self._dump_eager_tensors(
          outputs, op_type, input_ids, output_tensor_device_ids,
          graph_id=context_id))

  def _lookup_tensor_name(self, tensor):
    # [FUNCTION DOC]
    # HELPER: resolves debugger-generated tensor names for graph op creation records.
    # Takes a graph tensor and returns the original instrumented tensor's name if the tensor is a
    # debugger-created alias, otherwise the tensor's own name. Read-only.
    """Look up the name of a graph tensor.

    This method maps the name of a debugger-generated Identity or
    DebugIdentityV2 tensor to the name of the original instrumented tensor,
    if `tensor` is such a debugger-created tensor.
    Otherwise, it returns the name of `tensor` as is.

    Args:
      tensor: The graph tensor to look up the name for.

    Returns:
      Name of the original instrumented tensor as known to the debugger.
    """
    return self._tensor_aliases.get(tensor.name, tensor.name)

  def _func_graph_id_from_func_name(self, op_type):
    # [FUNCTION DOC]
    # HELPER: maps an eagerly executed function op type to its graph ID.
    # Takes an op type string or bytes. If it has a function-name prefix (forward, backward or
    # inference), looks up the cached ID, else scans the recorded functions for one whose name equals
    # the op type, caching and returning its graph ID. Returns nothing/null when the op type is not a
    # function or no match exists.
    """Attempt to get the ID of a FuncGraph based on an op type name.

    Also caches the ID for faster access later.

    Args:
      op_type: Op type string, which may be the name of a function.

    Returns:
      If the op_type name does not fit the pattern of a function name (e.g.,
      one that starts with "__inference_"), `None` is returned immediately.
      Else, if the FuncGraph is found, ID of the underlying FuncGraph is
      returned as a string.
      Else, `None` is returned.
    """
    op_type = compat.as_bytes(op_type)
    if is_op_type_function(op_type):
      # op_type for eagerly-executed FuncGraphs have the prefixed and suffixed
      # form such as "__inference_my_function_13579", wherein the middle part
      # "my_function" is the name of the Python function from which the
      # FuncGraph is compiled. Due to the suffix, the op_type is unique for
      # - duplicate Python function names
      # - multiple compilation of the same Python function
      if op_type in self._op_type_to_context_id:
        return self._op_type_to_context_id[op_type]
      with self._context_lock:
        for function in self._function_to_graph_id:
          if function.name == op_type:
            graph_id = self._function_to_graph_id[function]
            self._op_type_to_context_id[op_type] = graph_id
            return graph_id
      return None
    else:
      return None

  def _get_symbolic_tensor_ids(self, num_tensors):
    # [FUNCTION DOC]
    # HELPER: allocates tensor IDs for graph-op output tensors.
    # Takes the number of tensors; under the counter lock increments the symbolic tensor counter once
    # per tensor and returns the list of new counter values (empty list when the number is zero).
    # Mutates the counter.
    tensor_ids = []
    if num_tensors:
      with self._symbolic_tensor_counter_lock:
        for _ in range(num_tensors):
          self._symbolic_tensor_counter += 1
          tensor_ids.append(self._symbolic_tensor_counter)
    return tensor_ids

  def _should_dump_tensor(self, op_type, dtype):
    # [FUNCTION DOC]
    # HELPER: filter used when instrumenting or dumping tensors.
    # Decides whether a tensor of the given op type string and dtype should be dumped. Starts true;
    # if an op regular expression is configured the op type must match it at the start; if tensor
    # dtypes are configured, a list/tuple requires the dtype to equal one entry, otherwise the
    # callable is invoked with the dtype. Returns a truthy/falsy value (may be a match object or
    # nothing rather than a strict boolean). No side effects.
    """Determine if the given tensor's value will be dumped.

    The determination is made given the configurations such as `op_regex`,
    `tensor_dtypes`.

    Args:
      op_type: Name of the op's type, as a string (e.g., "MatMul").
      dtype: The dtype of the tensor, as a `dtypes.DType` object.

    Returns:
      A bool indicating whether the tensor's value will be dumped.
    """
    should_dump = True
    if self._op_regex:
      should_dump = (should_dump and
                     re.match(self._op_regex, op_type))
    if self._tensor_dtypes:
      if isinstance(self._tensor_dtypes, (list, tuple)):
        should_dump = (should_dump and
                       any(dtype == dtype_item for dtype_item
                           in self._tensor_dtypes))
      else:  # A callable that takes a DType argument and return a boolean.
        should_dump = should_dump and self._tensor_dtypes(dtype)
    return should_dump


@tf_export("debugging.experimental.enable_dump_debug_info")
def enable_dump_debug_info(dump_root,
                           tensor_debug_mode=DEFAULT_TENSOR_DEBUG_MODE,
                           circular_buffer_size=1000,
                           op_regex=None,
                           tensor_dtypes=None):
  # [FUNCTION DOC]
  # Public API that enables dumping of tfdbg v2 debug information for the current thread.
  # Inputs: dump_root (directory path string), tensor_debug_mode (string name of a debug mode;
  # default no-tensor), circular_buffer_size (integer, default 1000; non-positive disables the
  # circular behavior), op_regex (optional regular expression matched against op types),
  # tensor_dtypes (optional list/tuple of dtypes or dtype names, or a callable taking a dtype and
  # returning a boolean).
  # Raises a value error for an unknown debug mode name, for tensor_dtypes of the wrong kind, or when
  # a callback already exists in this thread with a different circular buffer size or debug mode;
  # raises not-implemented for modes outside the supported set.
  # Side effects: creates the thread-local callback if absent, registers it as an op callback and as a
  # concrete-function callback, updates its dump root if different, logs, and registers the disable
  # function to run at interpreter exit (once per call).
  # Returns the debug-events writer used by the callback.
  """Enable dumping debugging information from a TensorFlow program.

  The debugging information is dumped to a directory on the file system
  specified as `dump_root`.

  The dumped debugging information can be ingested by debugger UIs.

  The files in the dump directory contain the following information:
    - TensorFlow Function construction (e.g., compilation of Python functions
      decorated with @tf.function), the op types, names (if available), context,
      the input and output tensors, and the associated stack traces.
    - Execution of TensorFlow operations (ops) and Functions and their stack
      traces, op types, names (if available) and contexts. In addition,
      depending on the value of the `tensor_debug_mode` argument (see Args
      section below), the value(s) of the output tensors or more concise
      summaries of the tensor values will be dumped.
    - A snapshot of Python source files involved in the execution of the
      TensorFlow program.

  Once enabled, the dumping can be disabled with the corresponding
  `disable_dump_debug_info()` method under the same Python namespace.
  Calling this method more than once with the same `dump_root` is idempotent.
  Calling this method more than once with different `tensor_debug_mode`s
  leads to a `ValueError`.
  Calling this method more than once with different `circular_buffer_size`s
  leads to a `ValueError`.
  Calling this method with a different `dump_root` abolishes the
  previously-enabled `dump_root`.

  Usage example:

  ```py
  tf.debugging.experimental.enable_dump_debug_info('/tmp/my-tfdbg-dumps')

  # Code to build, train and run your TensorFlow model...
  ```

  NOTE: If your code is running on TPUs, be sure to call
  `tf.config.set_soft_device_placement(True)` before calling
  `tf.debugging.experimental.enable_dump_debug_info()` as this API uses
  automatic outside compilation on TPUs. For example:

  ```py
  tf.config.set_soft_device_placement(True)
  tf.debugging.experimental.enable_dump_debug_info(
      logdir, tensor_debug_mode="FULL_HEALTH")

  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  strategy = tf.distribute.TPUStrategy(resolver)
  with strategy.scope():
    # ...
  ```

  Args:
    dump_root: The directory path where the dumping information will be written.
    tensor_debug_mode: Debug mode for tensor values, as a string.
      The currently supported options are:
      - "NO_TENSOR": (Default) Only traces the output tensors of all executed
        ops (including those executed eagerly at the Python level or as a part
        of a TensorFlow graph) and functions, while not extracting any
        information from the values of the tensors.
      - "CURT_HEALTH": For each floating-dtype tensor (e.g., tensors of dtypes
        such as `float32`, `float64` and `bfloat16`), extracts a binary bit
        indicating whether it contains any -infinity, +infinity or NaN.
      - "CONCISE_HEALTH": For each floating-dtype tensor, extract total
        element count, and counts of -infinity, +infinity and NaN elements.
      - "FULL_HEALTH": For each floating-dtype tensor, extracts the dtype,
        rank (number of dimensions), total element count, and counts of
        -infinity, +infinity and NaN elements.
      - "SHAPE": For each tensor (regardless of dtype), extracts its dtype,
        rank, total element count and shape.
    circular_buffer_size: Size of the circular buffers for execution events.
      These circular buffers are designed to reduce the overhead of debugging
      dumping. They hold the most recent debug events concerning eager execution
      of ops and `tf.function`s and traces of tensor values computed inside
      `tf.function`s. They are written to the file system only when the proper
      flushing method is called (see description of return values below).
      Expected to be an integer. If <= 0, the circular-buffer behavior will be
      disabled, i.e., the execution debug events will be written to the file
      writers in the same way as non-execution events such as op creations and
      source-file snapshots.
    op_regex: Dump data from only the tensors from op types that matches to the
      regular expression (through Python's `re.match()`).
      "Op type" refers to the names of the TensorFlow operations (e.g.,
      "MatMul", "LogSoftmax"), which may repeat in a TensorFlow
      function. It does *not* refer to the names of nodes (e.g.,
      "dense/MatMul", "dense_1/MatMul_1") which are unique within a function.
      - Example 1: Dump tensor data from only MatMul and Relu ops
        `op_regex="^(MatMul|Relu)$"`.
      - Example 2: Dump tensors from all ops *except* Relu:
        `op_regex="(?!^Relu$)"`.
      This filter operates in a logical AND relation with `tensor_dtypes`.
    tensor_dtypes: Dump data from only the tensors of which the specified
      dtypes. This optional argument can be in any of the following format:
      - a list or tuple of `DType` objects or strings that can be converted
        to `DType` objects via `tf.as_dtype()`. Examples:
        - `tensor_dtype=[tf.float32, tf.float64]`,
        - `tensor_dtype=["float32", "float64"]`,
        - `tensor_dtypes=(tf.int32, tf.bool)`,
        - `tensor_dtypes=("int32", "bool")`
      - a callable that takes a single `DType` argument and returns a Python
        `boolean` indicating whether the dtype is to be included in the data
        dumping. Examples:
        - `tensor_dtype=lambda dtype: dtype.is_integer`.
      This filter operates in a logical AND relation with `op_regex`.
  Returns:
    A DebugEventsWriter instance used by the dumping callback. The caller
    may use its flushing methods, including `FlushNonExecutionFiles()` and
    `FlushExecutionFiles()`.
  """
  # TODO(cais): Revise the "UIs (currently under construction)" part of the doc
  # string above.
  # TODO(cais): Add Python code example to the doc string above.
  global _state

  tensor_debug_mode_keys = debug_event_pb2.TensorDebugMode.keys()
  if tensor_debug_mode not in tensor_debug_mode_keys:
    raise ValueError(
        "Invalid value in tensor_debug_mode ('%s'). Valid options are: %s" %
        (tensor_debug_mode, tensor_debug_mode_keys))

  tensor_debug_mode = debug_event_pb2.TensorDebugMode.Value(tensor_debug_mode)
  if tensor_debug_mode not in (debug_event_pb2.TensorDebugMode.NO_TENSOR,
                               debug_event_pb2.TensorDebugMode.CURT_HEALTH,
                               debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
                               debug_event_pb2.TensorDebugMode.FULL_HEALTH,
                               debug_event_pb2.TensorDebugMode.SHAPE,
                               debug_event_pb2.TensorDebugMode.FULL_TENSOR):
    raise NotImplementedError(
        "tfdbg dumping: support for tensor debug mode %s is not "
        "implemented yet" %
        debug_event_pb2.TensorDebugMode.Name(tensor_debug_mode))

  # Validate the types of tensor_dtypes.
  if tensor_dtypes is not None:
    if (not isinstance(tensor_dtypes, (list, tuple)) and
        not callable(tensor_dtypes)):
      raise ValueError(
          "If specified, tensor_dtypes is expected to be a list, a tuple, or "
          "a callable that takes a DType argument and returns a boolean, "
          "but received %s" % (tensor_dtypes,))
    if isinstance(tensor_dtypes, (list, tuple)):
      tensor_dtypes = [
          dtypes.as_dtype(dtype_item) for dtype_item in tensor_dtypes]

  if hasattr(_state, "dumping_callback"):
    if _state.dumping_callback.circular_buffer_size != circular_buffer_size:
      raise ValueError(
          "There is already a dumping callback configured with a different "
          "circular-buffer size (%d). Therefore the newly request "
          "circular-buffer size (%d) will not be honored." %
          (_state.dumping_callback.circular_buffer_size, circular_buffer_size))
    if _state.dumping_callback.tensor_debug_mode != tensor_debug_mode:
      raise ValueError(
          "There is already a dumping callback configured for dump root "
          "%s with a different "
          "tensor-debug mode (%s). Therefore the newly request "
          "tensor-debug mode (%s) size will not be honored." %
          (_state.dumping_callback.dump_root,
           tensor_debug_mode_keys[_state.dumping_callback.tensor_debug_mode],
           tensor_debug_mode_keys[tensor_debug_mode]))
  else:
    _state.dumping_callback = _DumpingCallback(dump_root,
                                               tensor_debug_mode,
                                               circular_buffer_size,
                                               op_regex,
                                               tensor_dtypes)
    op_callbacks.add_op_callback(_state.dumping_callback.callback)
    function_lib.CONCRETE_FUNCTION_CALLBACKS.append(
        _state.dumping_callback.function_callback)

  if _state.dumping_callback.dump_root != dump_root:
    _state.dumping_callback.dump_root = dump_root

  logging.info(
      "Enabled dumping callback in thread %s "
      "(dump root: %s, tensor debug mode: %s)",
      threading.current_thread().name,
      _state.dumping_callback.dump_root,
      debug_event_pb2.TensorDebugMode.Name(tensor_debug_mode))

  atexit.register(disable_dump_debug_info)
  return _state.dumping_callback.get_writer()


@tf_export("debugging.experimental.disable_dump_debug_info")
def disable_dump_debug_info():
  # [FUNCTION DOC]
  # Disables the dumping callback of the current thread, if one is enabled. Takes no arguments and
  # returns nothing. If a callback exists: closes the debug-events writer for its dump root and run
  # ID, removes the op callback and the concrete-function callback (if still registered), and
  # deletes the thread-local callback state. Otherwise it is a no-op. Safe to call repeatedly.
  # Also registered with atexit by the enable function.
  """Disable the currently-enabled debugging dumping.

  If the `enable_dump_debug_info()` method under the same Python namespace
  has been invoked before, calling this method disables it. If no call to
  `enable_dump_debug_info()` has been made, calling this method is a no-op.
  Calling this method more than once is idempotent.
  """
  if hasattr(_state, "dumping_callback"):
    dump_root = _state.dumping_callback.dump_root
    tfdbg_run_id = _state.dumping_callback.tfdbg_run_id
    debug_events_writer.DebugEventsWriter(dump_root, tfdbg_run_id).Close()
    op_callbacks.remove_op_callback(_state.dumping_callback.callback)
    if (
        _state.dumping_callback.function_callback
        in function_lib.CONCRETE_FUNCTION_CALLBACKS
    ):
      function_lib.CONCRETE_FUNCTION_CALLBACKS.remove(
          _state.dumping_callback.function_callback
      )
    delattr(_state, "dumping_callback")
    logging.info("Disabled dumping callback in thread %s (dump root: %s)",
                 threading.current_thread().name, dump_root)
