# [MODULE DOC]
# Implements the unified eager/graph numerics-checking mechanism behind the public debugging
# enable and disable check-numerics APIs. It registers a per-thread op callback that, for every
# floating-point op output, either eagerly checks the value for infinity/NaN or inserts a
# check-numerics op into graphs, producing a detailed error message (dtype, shape, inputs, and the
# op-creation stack trace in graph mode). Exposes the allowlists of ignored and safe ops, the
# message formatter, the callback wrapper class and the enable/disable functions. Import-time side
# effects: creates a thread-local state object and a monitoring counter that tracks how often the
# callback is created; the public functions are exported to the public API.
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
"""Eager-graph unified check numerics callback."""

import collections
import threading

import numpy as np

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export


# Many ops have benign NaN outputs, and running them with check_numerics
# on will create unwanted errors
# TODO(b/142497024): Replace this allowlist with function decorators in the ops
IGNORE_OP_OUTPUTS = (
    # For FusedBatchNorm, if the input tensor is empty then batch_mean and
    # batch_variance will be NaN. reserve_space holds intermediate values
    # derived from batch_mean and batch_variance used for gradient calculation
    (b"FusedBatchNorm", 1),  # batch_mean
    (b"FusedBatchNorm", 2),  # batch_variance
    (b"FusedBatchNorm", 3),  # reserve_space_1
    (b"FusedBatchNorm", 4),  # reserve_space_2

    # Same as above
    (b"FusedBatchNormV2", 1),  # batch_mean
    (b"FusedBatchNormV2", 2),  # batch_variance
    (b"FusedBatchNormV2", 3),  # reserve_space_1
    (b"FusedBatchNormV2", 4),  # reserve_space_2

    # Same as above, but reserve_space_3 holds additional intermediate values
    (b"FusedBatchNormV3", 1),  # batch_mean
    (b"FusedBatchNormV3", 2),  # batch_variance
    (b"FusedBatchNormV3", 3),  # reserve_space_1
    (b"FusedBatchNormV3", 4),  # reserve_space_2
    (b"FusedBatchNormV3", 5),  # reserve_space_3
)

# Some frequently used ops are generally safe and we can skip them to reduce
# overhead. NOTE: This list is compiled by observing operations called by
# models in practice and is not a comprehensive list of safe operations.
SAFE_OPS = (
    b"Concat",
    b"ConcatV2",
    b"ExpandDims",
    b"Fill",
    b"Gather",
    b"Maximum",
    b"Minimum",
    b"Reshape",
    b"Slice",
    b"Squeeze",
    b"Stack",
    b"StridedSlice",
    b"StridedSliceGrad",
    b"TensorListConcatV2",
    b"TensorListGather",
    b"TensorListGetItem",
    b"TensorListPopBack",
    b"TensorListStack",
    b"Transpose",
    b"Unpack",
)

_state = threading.local()

_check_numerics_callback_create_counter = monitoring.Counter(
    "/tensorflow/api/python/debugging/check_numerics_callback_create_counter",
    "Counter for number of times the check_numerics op callback is created.")


def limit_string_length(string, max_len=50):
  # [FUNCTION DOC]
  # HELPER: Path shortening utility for the check-numerics error message formatter.
  # Returns the input string unchanged if max_len is nothing or the string is no longer than
  # max_len; otherwise returns an ellipsis marker followed by the last max_len characters.
  """Limit the length of input string.

  Args:
    string: Input string.
    max_len: (int or None) If int, the length limit. If None, no limit.

  Returns:
    Possibly length-limited string.
  """
  if max_len is None or len(string) <= max_len:
    return string
  else:
    return "..." + string[len(string) - max_len:]


# A dictionary that supports looking up the original input tensor names.
_CHECK_NUMERICS_INPUT_LOOKUP = collections.defaultdict(dict)


def _maybe_lookup_original_input_tensor(graph, tensor):
  # [FUNCTION DOC]
  # HELPER: Used by the error message formatter to show pre-instrumentation input tensors.
  # Given a graph (may be nothing) and a tensor, returns the original tensor recorded in the module
  # level lookup table for that graph under the tensor's name (the table maps a checked tensor name
  # to the original output). If the graph is missing or has no entry, returns the given tensor.
  if (graph and
      graph in _CHECK_NUMERICS_INPUT_LOOKUP and
      tensor.name in _CHECK_NUMERICS_INPUT_LOOKUP[graph]):
    return _CHECK_NUMERICS_INPUT_LOOKUP[graph][tensor.name]
  else:
    return tensor


def get_check_numerics_error_message(slot,
                                     num_outputs,
                                     op_type,
                                     tensor,
                                     inputs,
                                     graph=None,
                                     traceback=None,
                                     stack_height_limit=30,
                                     path_length_limit=50):
  # [FUNCTION DOC]
  # Builds the multi-line error text reported when an op output contains infinity or NaN.
  # Inputs: slot (index of the offending output), num_outputs (total outputs of the op), op_type
  # (op type name), tensor (the offending tensor), inputs (list of the op's input tensors), graph
  # (optional; nothing means eager execution), traceback (optional list of frames of file path, line
  # number, function name, source line, graph mode only), stack_height_limit (default 30; nothing
  # means unlimited) and path_length_limit (default 50; nothing means unlimited).
  # Output: the message string. It states graph vs eager, the output slot, op type, dtype and shape.
  # In eager mode it counts and lists negative infinity, positive infinity and NaN elements. It lists
  # inputs (mapped back to original tensors when known), the graph name when present, and in graph
  # mode a trimmed stack trace marking non-TensorFlow frames as user code, noting omitted frames.
  # Raises nothing itself; no side effects.
  """Create a meaningful and user-friendly error message about offending tensor.

  The error message reveals the following info about the op that outputs
  NaN/Infinity: dtype, shape (to the extent known at graph-construction time),
  input tensors, stack trace for op creation (if is graph mode).

  Args:
    slot: (int) slot index of the tensor output.
    num_outputs: (int) total number of outputs of the op.
    op_type: (str) Type of the that generates `tensor`.
    tensor: (Tensor) the offending tensor, i.e., the tensor that contains
      Infinities or NaNs.
    inputs: (array of Tensor) inputs to the op that generates `tensor`.
    graph: (tf.Graph) the graph object that `tensor` belongs to. Available only
      under graph mode.
    traceback: (list of trace frames) the stack trace of the op's creation.
      Available only under graph model.
    stack_height_limit: (int or None) If int, limit to the height of the stack
      trace printed in the error message. If None, no limit to the height.
    path_length_limit: (int or None) Length limit for file paths included in the
      formatted stack trace.

  Returns:
    (str) A formatted error message.
  """
  eager_vs_graph_qualifier = "graph" if graph else "eagerly-executing"
  message = "\n"
  message += (
      "\n!!! Detected Infinity or NaN in output %d of "
      "%s op \"%s\" (# of outputs: %d) !!!\n" %
      (slot, eager_vs_graph_qualifier, op_type, num_outputs))

  message += "  dtype: %s\n" % tensor.dtype
  message += "  shape: %s\n" % (tensor.shape,)

  if not graph:
    # This is an eager tensor. We can get its numpy value and count
    # NaNs and Infs.
    is_inf = np.isinf(tensor)

    num_neg_inf = np.sum(np.logical_and(np.less(tensor, 0.), is_inf))
    num_pos_inf = np.sum(np.logical_and(np.greater(tensor, 0.), is_inf))
    num_nan = np.sum(np.isnan(tensor))
    if num_neg_inf > 0:
      message += "  # of -Inf elements: %s\n" % num_neg_inf
    if num_pos_inf > 0:
      message += "  # of +Inf elements: %s\n" % num_pos_inf
    if num_nan:
      message += "  # of +NaN elements: %s\n" % num_nan

  if len(inputs) > 1:
    message += "\n  Input tensors (%d):\n" % len(inputs)
    for slot, input_tensor in enumerate(inputs):
      message += "         %d: %s\n" % (
          slot, _maybe_lookup_original_input_tensor(graph, input_tensor))
  elif len(inputs) == 1:
    message += "\n  Input tensor: %s\n" % (
        _maybe_lookup_original_input_tensor(graph, inputs[0]))
  if graph and hasattr(graph, "name") and graph.name:
    message += "  Graph name: \"%s\"\n" % graph.name

  # Format the stack trace for the op's creation. We omit files that
  # belong to tensorflow itself.
  if graph and traceback:
    message += (
        "\n  Stack trace of op's creation (\"->\": inferred user code):\n")
    if stack_height_limit is not None and len(traceback) > stack_height_limit:
      num_omitted_frames = len(traceback) - stack_height_limit
      message += "    + ... (Omitted %d frames)\n" % num_omitted_frames
    for filepath, lineno, function_name, source_line in traceback[
        -stack_height_limit:]:
      user_code_indicator = "    "
      if not source_utils.guess_is_tensorflow_py_library(filepath):
        user_code_indicator = " -> "

      message += "    + %s (L%d) %s\n" % (
          limit_string_length(filepath, path_length_limit), lineno,
          function_name)
      if source_line is not None:
        message += "%s|   %s\n" % (user_code_indicator, source_line)
  message += "\n"
  return message


def _debug_summary(x):
  # [FUNCTION DOC]
  # HELPER: Used by the graph-mode callback in non-V1 mode to check numerics asynchronously.
  # Returns a debug numeric summary op output for tensor x that reduces it to three slots
  # flagging infinity and NaN presence; that summary is later fed to the check-numerics op.
  return gen_debug_ops.debug_numeric_summary_v2(
      x,
      tensor_debug_mode=(
          debug_event_pb2.TensorDebugMode.REDUCE_INF_NAN_THREE_SLOTS))


class CheckNumericsCallback(object):
  # [CLASS DOC]
  # Holds the configuration and state of the numerics-checking op callback for one thread.
  # State: the stack height limit and path length limit used in error messages, plus an identity
  # keyed mapping from placeholder tensors to their debug tensors (used only in V1 graph mode).
  # Created lazily by the enable function and stored in thread-local state; its callback method is
  # registered as a global op callback.
  """Wrapper for the numerics-checking callback for thread locality."""

  def __init__(self, stack_height_limit, path_length_limit):
    # [FUNCTION DOC]
    # Stores the stack height limit and path length limit (ints or nothing for unlimited) and
    # creates an empty identity-keyed mapping of placeholder tensors to debug tensors.
    self._stack_height_limit = stack_height_limit
    self._path_length_limit = path_length_limit
    # A dict mapping Placeholder tensors to their instrumenting debug tensors.
    # Used only under V1 graph mode, where we can't rely on auto control
    # dependency to execute the debug tensors and hence need to attach the debug
    # tensors as control dependencies of the ops that consume the Placeholder.
    self._placeholder_to_debug_tensor = (
        object_identity.ObjectIdentityDictionary())

  def callback(self,
               op_type,
               inputs,
               attrs,
               outputs,
               op_name=None,
               graph=None):
    # [FUNCTION DOC]
    # Op callback invoked for each created or executed op. Inputs: op_type (op type name), inputs
    # (input tensors), attrs and op_name (both ignored), outputs (output tensors), graph (the graph
    # being built, or nothing when executing eagerly).
    # Returns nothing (no change) for ops in the shared skip list or the safe-op list, and in eager
    # mode also for the check-numerics op itself (to avoid recursion). In graph mode it returns a
    # list of output tensors in which every floating-point output not on the ignore list is
    # instrumented with a check-numerics op carrying a detailed error message (in non-V1 mode the
    # op checks an asynchronous debug summary instead of the raw output). It records a mapping from
    # the checked tensor name back to the original output, and in V1 graph mode adds control
    # inputs so that debug tensors of consumed placeholders run. In eager mode it directly runs the
    # check on each floating-point output, which raises an invalid-argument error on bad values;
    # the eager path returns nothing.
    """Eager-function unified callback for checking numerics."""
    del attrs, op_name  # Unused
    op_type_bytes = compat.as_bytes(op_type)
    is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
    if (op_type_bytes in op_callbacks_common.OP_CALLBACK_SKIP_OPS or
        op_type_bytes in SAFE_OPS):
      return None
    if graph:
      # Under graph mode. Insert check_numerics op.
      instrumented_outputs = []
      if is_v1_graph_mode:
        for input_tensor in inputs:
          if input_tensor in self._placeholder_to_debug_tensor and outputs:
            outputs[0].op._add_control_input(  # pylint: disable=protected-access
                self._placeholder_to_debug_tensor[input_tensor].op)
      for slot, output in enumerate(outputs):
        if (output.dtype.is_floating and
            (op_type_bytes, slot) not in IGNORE_OP_OUTPUTS):
          checked_output = array_ops.check_numerics_v2(
              # TF v2 has automatic control dependencies added to stateful async
              # ops, which allows us to run check_numerics asynchronously.
              # In the above case we use debug_summary to reduce all output
              # tensors asynchronously from the op being checked and then
              # process the tensor summary with check_numerics.
              output if is_v1_graph_mode else _debug_summary(output),
              get_check_numerics_error_message(
                  slot,
                  len(outputs),
                  op_type,
                  output,
                  inputs,
                  graph=graph,
                  traceback=output.op.traceback,
                  stack_height_limit=self._stack_height_limit,
                  path_length_limit=self._path_length_limit))
          _CHECK_NUMERICS_INPUT_LOOKUP[graph][checked_output.name] = output
          instrumented_outputs.append(self._get_output_tensor(
              op_type_bytes, output, checked_output, is_v1_graph_mode))
        else:
          instrumented_outputs.append(output)
      return instrumented_outputs
    else:
      if op_type_bytes == b"CheckNumericsV2":
        # TODO(b/140334369): Remove this special casing logic once op_callback.
        # automatically prevents infinite recursion in eager mode.
        return None
      # Under eager mode. Eagerly execute check_numerics op.
      for slot, output in enumerate(outputs):
        if (output.dtype.is_floating and
            (op_type_bytes, slot) not in IGNORE_OP_OUTPUTS):
          array_ops.check_numerics_v2(
              output,
              get_check_numerics_error_message(
                  slot, len(outputs), op_type, output, inputs,
                  stack_height_limit=self._stack_height_limit,
                  path_length_limit=self._path_length_limit))

  def _get_output_tensor(self,
                         op_type,
                         tensor,
                         checked_tensor,
                         is_v1_graph_mode):
    # [FUNCTION DOC]
    # HELPER: Chooses the tensor the graph-mode callback substitutes for an op output.
    # Inputs: op_type (bytes), the original symbolic tensor, the checked tensor, and whether V1
    # graph mode is active. In V1 mode a placeholder output is returned unchanged but its checked
    # tensor is remembered (side effect on the placeholder mapping); other ops return the checked
    # tensor. In non-V1 mode the original tensor is returned, relying on automatic control
    # dependencies to run the check.
    """Determine what tensor to output from callback.

    Args:
      op_type: Type of the op that outputs the original symbolic tensor, as
        `bytes`.
      tensor: The original output symbolic tensor.
      checked_tensor: The debugger-instrumented, numerics-checking tensor.
      is_v1_graph_mode: Whether the debugged proggram is running under V1 graph
        mode.

    Returns:
      A symbolic tensor to be returned by the dumping op_callback.
    """
    if is_v1_graph_mode:
      # Placeholders need special treatment under V1 graph mode. The
      # callback can't simply override the Placeholder tensor to the debug
      # tensor, as that would cause the Placeholder op to lack a value.
      # The debug tensor is remembered and will be attached as control
      # inputs to ops that consumer the Placeholders later.
      if op_type == b"Placeholder":
        self._placeholder_to_debug_tensor[tensor] = checked_tensor
        return tensor
      else:
        return checked_tensor
    else:
      # Under non-v1 graph mode, rely on auto control dependency to run the
      # checked tensor.
      return tensor


@tf_export("debugging.enable_check_numerics")
def enable_check_numerics(stack_height_limit=30,
                          path_length_limit=50):
  # [FUNCTION DOC]
  # KNOWN ISSUE: If a callback already exists in the calling thread, the new limit arguments are
  # silently ignored (the old limits stay) and the existing callback is simply re-added.
  # Turns on numerics checking in the calling thread. Inputs: stack_height_limit (default 30) and
  # path_length_limit (default 50), used when formatting error messages for graph ops. Creates a
  # thread-local callback object if none exists, registers its callback as an op callback, logs an
  # informational message and increments the creation monitoring counter (on every call). Returns
  # nothing.
  r"""Enable tensor numerics checking in an eager/graph unified fashion.

  The numerics checking mechanism will cause any TensorFlow eager execution or
  graph execution to error out as soon as an op's output tensor contains
  infinity or NaN.

  This method is idempotent. Calling it multiple times has the same effect
  as calling it once.

  This method takes effect only on the thread in which it is called.

  When a op's float-type output tensor contains any Infinity or NaN, an
  `tf.errors.InvalidArgumentError` will be thrown, with an error message that
  reveals the following information:
    - The type of the op that generated the tensor with bad numerics.
    - Data type (dtype) of the tensor.
    - Shape of the tensor (to the extent known at the time of eager execution
      or graph construction).
    - Name of the containing graph (if available).
    - (Graph mode only): The stack trace of the intra-graph op's creation,
      with a stack-height limit and a path-length limit for visual clarity.
      The stack frames that belong to the user's code (as opposed to
      tensorflow's internal code) are highlighted with a text arrow ("->").
    - (Eager mode only): How many of the offending tensor's elements are
      `Infinity` and `NaN`, respectively.

  Once enabled, the check-numerics mechanism can be disabled by using
  `tf.debugging.disable_check_numerics()`.

  Example usage:

  1. Catching infinity during the execution of a `tf.function` graph:

     ```py
     import tensorflow as tf

     tf.debugging.enable_check_numerics()

     @tf.function
     def square_log_x_plus_1(x):
       v = tf.math.log(x + 1)
       return tf.math.square(v)

     x = -1.0

     # When the following line runs, a function graph will be compiled
     # from the Python function `square_log_x_plus_1()`. Due to the
     # `enable_check_numerics()` call above, the graph will contain
     # numerics checking ops that will run during the function graph's
     # execution. The function call generates an -infinity when the Log
     # (logarithm) op operates on the output tensor of the Add op.
     # The program errors out at this line, printing an error message.
     y = square_log_x_plus_1(x)
     z = -y
    ```

  2. Catching NaN during eager execution:

     ```py
     import numpy as np
     import tensorflow as tf

     tf.debugging.enable_check_numerics()

     x = np.array([[0.0, -1.0], [4.0, 3.0]])

     # The following line executes the Sqrt op eagerly. Due to the negative
     # element in the input array, a NaN is generated. Due to the
     # `enable_check_numerics()` call above, the program errors immediately
     # at this line, printing an error message.
     y = tf.math.sqrt(x)
     z = tf.matmul(y, y)
     ```

  NOTE: If your code is running on TPUs, be sure to call
  `tf.config.set_soft_device_placement(True)` before calling
  `tf.debugging.enable_check_numerics()` as this API uses automatic outside
  compilation on TPUs. For example:

  ```py
  tf.config.set_soft_device_placement(True)
  tf.debugging.enable_check_numerics()

  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  strategy = tf.distribute.TPUStrategy(resolver)
  with strategy.scope():
    # ...
  ```

  Args:
    stack_height_limit: Limit to the height of the printed stack trace.
      Applicable only to ops in `tf.function`s (graphs).
    path_length_limit: Limit to the file path included in the printed stack
      trace. Applicable only to ops in `tf.function`s (graphs).
  """
  if not hasattr(_state, "check_numerics_callback"):
    _state.check_numerics_callback = CheckNumericsCallback(
        stack_height_limit, path_length_limit)
  op_callbacks.add_op_callback(_state.check_numerics_callback.callback)

  logging.info(
      "Enabled check-numerics callback in thread %s",
      threading.current_thread().name)
  _check_numerics_callback_create_counter.get_cell().increase_by(1)


@tf_export("debugging.disable_check_numerics")
def disable_check_numerics():
  # [FUNCTION DOC]
  # Turns off numerics checking in the calling thread. Takes no inputs and returns nothing. If no
  # callback is stored for the thread it returns immediately. Otherwise it removes the callback from
  # the registered op callbacks, deletes the thread-local callback object and logs a message; a
  # missing-registration error is tolerated silently.
  """Disable the eager/graph unified numerics checking mechanism.

  This method can be used after a call to `tf.debugging.enable_check_numerics()`
  to disable the numerics-checking mechanism that catches infinity and NaN
  values output by ops executed eagerly or in tf.function-compiled graphs.

  This method is idempotent. Calling it multiple times has the same effect
  as calling it once.

  This method takes effect only on the thread in which it is called.
  """
  if not hasattr(_state, "check_numerics_callback"):
    return
  try:
    op_callbacks.remove_op_callback(_state.check_numerics_callback.callback)
    delattr(_state, "check_numerics_callback")
    logging.info(
        "Disabled check-numerics callback in thread %s",
        threading.current_thread().name)
  except KeyError:
    # Tolerate disabling the check numerics callback without
    # enable_check_numerics() being called first.
    pass
