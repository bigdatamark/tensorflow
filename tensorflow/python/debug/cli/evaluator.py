# [MODULE DOC]
# Expression evaluator for the TensorFlow debugger (tfdbg) command-line interface.
# Lets a user write an arbitrary numerical expression in which tensors recorded in a debug dump are
# referenced by backtick-quoted debug tensor names (node name, output slot, optional debug op,
# optional device prefix and optional execution-index suffix). The expression is rewritten to
# refer to the tensor values loaded from the dump and then evaluated as ordinary code, with numpy
# available under its usual short name.
# Exposes the ExpressionEvaluator class plus a private parser for debug tensor names.
# Import-time side effects: only compiles a few regular expressions and defines the default debug
# op name (the identity debug op). No registrations or flags.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Library for arbitrary expression evaluation based on a debugger data dump."""
import re

import numpy as np  # pylint: disable=unused-import

from tensorflow.python.debug.lib import debug_data

_DUMP_TENSOR_PATTERN = re.compile(r"`.*?`")
_DEVICE_NAME_PREFIX_PATTERN = re.compile(
    r"/job:(\w)+/replica:(\d)+/task:(\d)+/(\w)+:(\d)+:")
_EXEC_INDEX_SUFFIX_PATTERN = re.compile(r"\[(\d)*\]$")

_DEFAULT_DEBUG_OP = "DebugIdentity"


def _parse_debug_tensor_name(debug_tensor_name):
  # [FUNCTION DOC]
  # HELPER: splits debug tensor names found in expressions for ExpressionEvaluator.evaluate.
  # Parses a debug tensor name string into its parts. An optional device-name prefix (job, replica,
  # task and device type/index followed by a colon) is stripped first. The remainder must have
  # exactly two or three colon-separated items: node name, output slot, and optionally a debug op.
  # An optional bracketed non-negative integer at the end of the last item is the execution index
  # (default zero) and is removed before the slot or debug op is read.
  # Returns a five-item tuple: device name (or nothing if no prefix), node name, output slot as an
  # integer, debug op name (the identity debug op when omitted), and execution index as an integer.
  # Raises a value error if the item count is not two or three, and the integer conversion raises
  # a value error if the slot is not a number or the bracket content is empty/non-numeric.
  # No side effects.
  # pylint: disable=line-too-long
  """Parse a debug tensor name in a to-be-evaluated expression.

  Args:
    debug_tensor_name: name of the debug tensor, with or without
      device name as a prefix, with or without debug op, with or
      without '[<exec_index>]' as a suffix.
      E.g., without device name prefix, without debug op suffix:
        "hidden_0/MatMul:0"
      E.g., with device name prefix:
        "/job:worker/replica:0/task:1/gpu:0:hidden_0/MatMul:0"
      E.g., with debug op suffix:
        "hidden_0/MatMul:0:DebugNumericSummary"
      E.g., with device name prefix and debug op suffix:
        "/job:worker/replica:0/task:1/gpu:0:hidden_0/MatMul:0:DebugNumericSummary"
      E.g., with device name prefix, debug op and an exec index:
        "/job:worker/replica:0/task:1/gpu:0:hidden_0/MatMul:0:DebugNumericSummary[1]"

  Returns:
    device_name: If device name prefix exists, the device name; otherwise,
      `None`.
    node_name: Name of the node.
    output_slot: Output slot index as an `int`.
    debug_op: If the debug op suffix exists, the debug op name; otherwise,
      `None`.
    exec_index: Execution index (applicable to cases in which a debug tensor
      is computed multiple times in a `tf.Session.run` call, e.g., due to
      `tf.while_loop`). If the exec_index suffix does not exist, this value
      defaults to `0`.

  Raises:
    ValueError: If the input `debug_tensor_name` is malformed.
  """
  # pylint: enable=line-too-long
  device_prefix_match = re.match(_DEVICE_NAME_PREFIX_PATTERN, debug_tensor_name)
  if device_prefix_match:
    device_name = debug_tensor_name[
        device_prefix_match.start() : device_prefix_match.end() - 1]
    debug_tensor_name = debug_tensor_name[device_prefix_match.end():]
  else:
    device_name = None

  split_items = debug_tensor_name.split(":")
  if len(split_items) not in (2, 3):
    raise ValueError(
        "The debug tensor name in the to-be-evaluated expression is malformed: "
        "'%s'" % debug_tensor_name)
    # TODO(cais): Provide examples of good debug tensor names in the error
    # message.

  exec_index_match = re.search(_EXEC_INDEX_SUFFIX_PATTERN, split_items[-1])
  if exec_index_match:
    exec_index = int(split_items[-1][
        exec_index_match.start() + 1 : exec_index_match.end() - 1])
    split_items[-1] = split_items[-1][:exec_index_match.start()]
  else:
    exec_index = 0

  if len(split_items) == 2:
    node_name = split_items[0]
    output_slot = int(split_items[1])
    debug_op = _DEFAULT_DEBUG_OP
  else:
    split_items = debug_tensor_name.split(":")
    node_name = split_items[0]
    output_slot = int(split_items[1])
    debug_op = split_items[2]

  return device_name, node_name, output_slot, debug_op, exec_index


class ExpressionEvaluator(object):
  # [CLASS DOC]
  # Evaluates user-supplied expressions in which backtick-quoted debug tensor names are replaced
  # by tensor values read from a debug dump object. Holds the dump (used to look up tensors) and a
  # cache mapping each tensor name string already fetched to its value, so repeated references
  # do not re-read the dump. Created with a dump and used through the evaluate method.
  """Evaluates Python expressions using debug tensor values from a dump."""

  def __init__(self, dump):
    # [FUNCTION DOC]
    # Constructor. Takes dump, a debug dump directory object that provides tensor lookup by node
    # name, output slot, debug op and device name. Stores it and creates an empty cache of tensor
    # values keyed by the tensor name text. Returns nothing; raises nothing.
    """Constructor of ExpressionEvaluator.

    Args:
      dump: an instance of `DebugDumpDir`.
    """
    self._dump = dump
    self._cached_tensor_values = {}

  def evaluate(self, expression):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it parses an expression, but it actually rewrites the text
    # and then executes it with the built-in dynamic evaluator, returning the computed value.
    # The raised error message also always names the identity debug op even if another debug op
    # was requested.
    # Input: expression, a string of code in which each backtick-quoted span names a debug tensor.
    # Matches are processed from last to first so that earlier positions stay valid. Each name is
    # stripped of whitespace and parsed; its value is fetched from the dump (taking the entry at
    # the parsed execution index) unless already cached, then the quoted span is replaced by a
    # reference to the cached value. Finally the rewritten text is evaluated, with access to
    # numpy and the evaluator itself, and the result returned (any type).
    # Raises a value error if the tensor is absent from the dump, or if a name is malformed.
    # Errors from the evaluated code itself (e.g. syntax error for an unmatched backtick)
    # propagate. Side effect: fills the tensor value cache. Executes arbitrary user code.
    """Parse an expression.

    Args:
      expression: the expression to be parsed.

    Returns:
      The result of the evaluation.

    Raises:
      ValueError: If the value of one or more of the debug tensors in the
        expression are not available.
    """
    dump_tensors_iter = re.finditer(_DUMP_TENSOR_PATTERN, expression)
    rewritten_expression = expression
    for match in reversed(list(dump_tensors_iter)):
      tensor_name = match.group(0)[1:-1].strip()
      device_name, node_name, output_slot, debug_op, exec_index = (
          _parse_debug_tensor_name(tensor_name))
      if tensor_name not in self._cached_tensor_values:
        try:
          value = self._dump.get_tensors(
              node_name, output_slot, debug_op,
              device_name=device_name)[exec_index]
        except debug_data.WatchKeyDoesNotExistInDebugDumpDirError:
          raise ValueError(
              "Eval failed due to the value of %s:%d:DebugIdentity being "
              "unavailable" % (node_name, output_slot))
        self._cached_tensor_values[tensor_name] = value
      rewritten_expression = (
          rewritten_expression[:match.start(0)] +
          "self._cached_tensor_values['" + tensor_name + "']" +
          rewritten_expression[match.end(0):])

    return eval(rewritten_expression)  # pylint: disable=eval-used
