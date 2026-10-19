# [MODULE DOC]
# HELPER: helper module for converting the byte order of serialized tensor data.
# Provides a list of data types whose raw content can be byte swapped (mirroring the C++ tensor
# bundle byte swap logic) and functions that swap the endianness of constant tensors stored in
# protocol buffer graph definitions, either inside function library bodies or in top-level graph
# nodes. Import-time side effect: only builds the module-level list of swappable types.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Utilities for byte swapping the tensor content."""

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes

# Based on tensor_bundle/byte_swap.cc
byte_swappable = [
    dtypes.float16,
    dtypes.float32,
    dtypes.float64,
    dtypes.bfloat16,
    dtypes.complex64,
    dtypes.complex128,
    dtypes.uint16,
    dtypes.uint32,
    dtypes.uint64,
    dtypes.int16,
    dtypes.int32,
    dtypes.int64,
    dtypes.qint16,
    dtypes.quint16,
    dtypes.qint32,
]


def byte_swap_tensor_content(tensor, from_endiness, to_endiness):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring is just "Byte swaps."; the parameter names are misspelled
  # (endiness) and nothing validates the endianness strings (invalid names raise a value error
  # from the integer conversion only when content exists).
  # Swaps the byte order of the raw content of a tensor protocol buffer in place. Inputs: the
  # tensor proto, and source and target byte order names ("big" or "little"). Does nothing unless
  # the data type is in the swappable list and the raw content is non-empty. The element size is
  # computed as content length divided by the product of the nonzero dimension sizes; content is
  # split into elements of that size, each reinterpreted from the source order and re-encoded in
  # the target order, and the joined result replaces the content. Returns nothing. May fail with
  # a division by zero error if content exists but all dimensions are unknown or empty.
  """Byte swaps.

  Args:
    tensor: Target tensor to change endiness.
    from_endiness: The original endianness format. "big" or "little"
    to_endiness: The target endianness format. "big" or "little"
  """
  if tensor.dtype in byte_swappable:
    tshape = tensor.tensor_shape.dim
    tensor_bytes = tensor.tensor_content
    if tensor_bytes:
      tensor_size = 1
      for sz in tshape:
        if sz.size != 0:
          tensor_size *= sz.size
      chunksize = len(tensor_bytes) // tensor_size
      # Split tensor_data into chunks for byte swapping.
      to_swap = [
          tensor_bytes[i : i + chunksize]
          for i in range(0, len(tensor_bytes), chunksize)
      ]
      # Swap and replace tensor_content.
      tensor.tensor_content = b"".join(
          [
              int.from_bytes(byteswap, from_endiness).to_bytes(
                  chunksize, to_endiness
              )
              for byteswap in to_swap
          ]
      )


def swap_tensor_content_in_graph_function(
    graph_def, from_endiness, to_endiness
):
  # [FUNCTION DOC]
  # Swaps endianness of constant tensors inside the function library of a graph definition.
  # Inputs: a graph definition or meta graph definition, and the source and target byte order
  # names ("big" or "little"). For a meta graph it uses the embedded graph's library; any other
  # input type is silently ignored (returns without effect). Only constant nodes in library
  # functions are modified, in place; top-level graph nodes are untouched (see the node variant).
  # Returns nothing.
  """Fix endiness of tensor contents.

  Args:
    graph_def: Target graph_def to change endiness.
    from_endiness: The original endianness format. "big" or "little"
    to_endiness: The target endianness format. "big" or "little"
  """
  if isinstance(graph_def, meta_graph_pb2.MetaGraphDef):
    functions = graph_def.graph_def.library.function
  elif isinstance(graph_def, graph_pb2.GraphDef):
    functions = graph_def.library.function
  else:
    return
  for function in functions:
    node_def = function.node_def
    for node in node_def:
      if node.op == "Const":
        tensor = node.attr["value"].tensor
        byte_swap_tensor_content(tensor, from_endiness, to_endiness)


def swap_tensor_content_in_graph_node(graph_def, from_endiness, to_endiness):
  # [FUNCTION DOC]
  # Swaps endianness of the value of every constant node in the top-level node list of a graph
  # definition (function library bodies are not visited). Inputs: the graph definition and the
  # source and target byte order names ("big" or "little"). Mutates the tensors in place; returns
  # nothing. No docstring exists for this function.
  for node in graph_def.node:
    if node.op == "Const":
      tensor = node.attr["value"].tensor
      byte_swap_tensor_content(tensor, from_endiness, to_endiness)
