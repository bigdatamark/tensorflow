# [MODULE DOC]
# HELPER: internal helper module providing graph-only op builders.
# Holds graph-only variants of a few op functions (currently only a placeholder builder) that are
# kept separate from the array ops module to avoid a cyclic dependency. Internal use only; no
# import-time side effects.
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
"""Graph-only versions of a few op functions, for internal use only."""

# Must be separate from array_ops to avoid a cyclic dependency.

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


def graph_placeholder(dtype, shape, name=None):
  # [FUNCTION DOC]
  # HELPER: internal graph-mode placeholder builder.
  # Creates a placeholder operation directly in the default graph. Inputs: dtype (a data type,
  # reduced to its base type), shape (a shape object, or a list/tuple converted to one), name
  # (optional operation name). Returns the output tensor of the new operation, or, if operation
  # callbacks are active, the first replacement output returned by the callbacks. Side effects:
  # adds an operation to the default graph and invokes registered op callbacks. Must be called
  # when a graph is the default context.
  """Graph-only version of tf.compat.v1.placeholder(), for internal use only."""
  dtype = dtype.base_dtype
  dtype_value = attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
  if isinstance(shape, (list, tuple)):
    shape = tensor_shape.TensorShape(shape)
  shape = attr_value_pb2.AttrValue(shape=shape.as_proto())
  g = ops.get_default_graph()
  attrs = {"dtype": dtype_value, "shape": shape}
  op = g._create_op_internal(  # pylint: disable=protected-access
      "Placeholder", [], [dtype], input_types=[],
      attrs=attrs, name=name)
  result, = op.outputs
  if op_callbacks.should_invoke_op_callbacks():
    # TODO(b/147670703): Once the special-op creation code paths
    # are unified. Remove this `if` block.
    callback_outputs = op_callbacks.invoke_op_callbacks(
        "Placeholder", tuple(), attrs, tuple(op.outputs),
        op_name=name, graph=g)
    if callback_outputs is not None:
      result, = callback_outputs
  return result
