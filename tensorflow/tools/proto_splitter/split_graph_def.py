# [MODULE DOC]
# Splitter for GraphDef protos. Provides a set of composable splitters that break an oversized
# GraphDef into several chunks, each below the maximum proto size from the constants module.
# Nodes in the node list and functions in the function library are split into separate chunks;
# individual large nodes are split further (constant tensor contents are moved to their own chunk
# as raw bytes, other large messages become their own chunk). Exposes GraphDefSplitter (the
# entry point), several size-based splitters, and a helper to extract constant values.
# No import-time side effects beyond defining module constants and size-check callbacks.
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
"""GraphDef splitter."""

from collections.abc import Sequence
import itertools
from typing import Optional, Type

from google.protobuf import message
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter import split
from tensorflow.tools.proto_splitter import util

_CONST_OP = "Const"


class GraphDefSplitter(split.ComposableSplitter):
  # [CLASS DOC]
  # Top-level splitter for a GraphDef proto. It holds the GraphDef (modified in place) and, when
  # asked to build chunks, delegates to a repeated-message splitter for the node list and another
  # for the function library, deciding which to run first based on which part is larger.
  # Created with the proto and the usual composable-splitter keyword arguments.
  """Implements proto splitter for GraphDef.

  This Splitter will modify the passed in proto in place.
  """

  def build_chunks(self):
    # [FUNCTION DOC]
    # Splits the held GraphDef into chunks when its serialized size reaches the maximum size.
    # Takes no arguments; returns nothing. Raises a type error if the proto is not a GraphDef.
    # If the proto is already under the limit, does nothing. Otherwise creates splitters for the
    # node list and for the library functions, runs the larger part first (and the other only if
    # the total is still too big), then, if the proto remains oversized, moves the whole library
    # into its own chunk at index 1 (index 0 is the base chunk) and clears it from the proto.
    # Side effects: mutates the proto in place and registers chunks with the parent splitter.
    """Splits a GraphDef proto into smaller chunks."""
    proto = self._proto
    if not isinstance(proto, graph_pb2.GraphDef):
      raise TypeError("Can only split GraphDef type protos.")

    proto_size = proto.ByteSize()
    if proto_size < constants.max_size():
      return

    # Split `GraphDef.node`
    node_splitter = RepeatedMessageSplitter(
        proto,
        "node",
        [ConstantNodeDefSplitter, LargeMessageSplitter],
        parent_splitter=self,
        fields_in_parent=[],
    )

    # Split `GraphDef.library.function`
    function_splitter = RepeatedMessageSplitter(
        proto.library,
        ["function"],
        [FunctionDefSplitter],
        parent_splitter=self,
        fields_in_parent=["library"],
    )

    library_size = proto.library.ByteSize()
    approx_node_size = proto_size - library_size

    if library_size > approx_node_size:
      library_size -= function_splitter.build_chunks()
      if library_size + approx_node_size > constants.max_size():
        approx_node_size -= node_splitter.build_chunks()

    else:
      approx_node_size -= node_splitter.build_chunks()
      if library_size + approx_node_size > constants.max_size():
        library_size -= function_splitter.build_chunks()

    if proto.ByteSize() > constants.max_size():
      # Since there are chunks with the "library" field tag, insert this
      # chunk before the other chunks at index 1 (index 0 is reserved for the
      # base chunk).
      self.add_chunk(proto.library, ["library"], 1)
      proto.ClearField("library")

_KEEP_TENSOR_PROTO_FIELDS = ("dtype", "tensor_shape", "version_number")


def chunk_constant_value(node: node_def_pb2.NodeDef, size: int):
  # [FUNCTION DOC]
  # Removes the tensor data from a Const node and returns it as bytes.
  # Inputs: node, a NodeDef; size, its serialized size (used only in the error message).
  # For a Const node, takes the raw tensor content (or, if absent, converts the typed value
  # fields to a byte string), clears the tensor proto while preserving its dtype, shape and
  # version number, and returns the bytes. Mutates the node in place.
  # Raises a value error describing the node (name, op, inputs, device, attribute sizes) if the
  # node is not a Const, since only constants can be split further.
  """Extracts and clears the constant value from a NodeDef.

  Args:
    node: NodeDef with const value to extract.
    size: Size of NodeDef (for error reporting).

  Returns:
    Bytes representation of the Constant tensor content.
  """
  if node.op == _CONST_OP:
    tensor_proto = node.attr["value"].tensor
    if tensor_proto.tensor_content:
      b = tensor_proto.tensor_content
    else:
      # The raw tensor value could be stored in one of the "xxx_val" attributes.
      # Extract it here, and convert to bytes.
      b = tensor_util.MakeNdarray(tensor_proto).tobytes()

    # Keep the TensorProto's dtype, tensor_shape, and version_number fields,
    # but clear the raw tensor content / "xxx_val" attributes.
    kept_attributes = {
        key: getattr(tensor_proto, key) for key in _KEEP_TENSOR_PROTO_FIELDS
    }
    tensor_proto.Clear()
    for field, val in kept_attributes.items():
      if isinstance(val, message.Message):
        getattr(tensor_proto, field).MergeFrom(val)
      else:
        setattr(tensor_proto, field, val)

    return b
  else:
    attributes_and_sizes = ", ".join(
        [
            f"{key}: {util.format_bytes(val.ByteSize())}"
            for key, val in node.attr.items()
        ]
    )
    raise ValueError(
        "Unable to split GraphDef because at least one of the nodes "
        "individually exceeds the max size of "
        f"{util.format_bytes(constants.max_size())}. "
        "Currently only Const nodes can be further split."
        "\nNode info:"
        f"\n\tsize: {util.format_bytes(size)}"
        f"\n\tname: {node.name}"
        f"\n\top: {node.op}"
        f"\n\tinputs: {node.input}"
        f"\n\top: {node.op}"
        f"\n\tdevice: {node.device}"
        f"\n\tattr (and sizes): {attributes_and_sizes}"
    )


def _split_repeated_field(
    proto: message.Message,
    new_proto: message.Message,
    fields: util.FieldTypes,
    start_index: int,
    end_index: Optional[int] = None,
) -> None:
  # [FUNCTION DOC]
  # HELPER: used by the repeated-message splitter to fill the chunk protos.
  # Copies a slice of a repeated field from proto into the same field of new_proto (merging).
  # Inputs: proto, source message; new_proto, destination message; fields, the field path;
  # start_index, first element to copy; end_index, exclusive end (nothing means to the end).
  # Returns nothing; mutates new_proto only.
  """Generic function for copying a repeated field from one proto to another."""
  util.get_field(new_proto, fields)[0].MergeFrom(
      util.get_field(proto, fields)[0][start_index:end_index]
  )


_ABOVE_MAX_SIZE = lambda x: x > constants.max_size()
_GREEDY_SPLIT = lambda x: x > constants.max_size() // 3
_ALWAYS_SPLIT = lambda x: True


class SplitBasedOnSize(split.ComposableSplitter):
  # [CLASS DOC]
  # HELPER: base class for the per-message splitters used by the repeated-message splitter.
  # KNOWN ISSUE: the slots list names "fn", which is never assigned anywhere; only proto_size is
  # used.
  # Stores the already-computed serialized size of the proto. Its own chunk building does
  # nothing; subclasses override it.
  """A Splitter that's based on the size of the input proto."""

  __slots__ = ("fn", "proto_size")

  def __init__(self, proto, proto_size, **kwargs):
    # [FUNCTION DOC]
    # Stores proto_size (serialized size of the proto in bytes) and forwards the proto and any
    # keyword arguments (parent splitter, fields in parent) to the base splitter constructor.
    """Initializer."""
    self.proto_size = proto_size
    super().__init__(proto, **kwargs)

  def build_chunks(self) -> int:
    # [FUNCTION DOC]
    # Default implementation: creates no chunks and returns zero (number of bytes removed).
    """Splits the proto, and returns the size of the chunks created."""
    return 0


class RepeatedMessageSplitter(split.ComposableSplitter):
  # [CLASS DOC]
  # Splits a repeated message field of a proto (for example the nodes of a GraphDef) into several
  # chunks. Holds the field path and the list of per-element splitter classes to apply to each
  # element before deciding where to cut the list.
  """Splits a repeated message field on a proto."""

  __slots__ = ("repeated_field", "message_splitters")

  def __init__(
      self,
      proto,
      repeated_field: util.FieldTypes,
      message_splitters: Sequence[Type[SplitBasedOnSize]],
      **kwargs,
  ):
    # [FUNCTION DOC]
    # Inputs: proto, the message containing the field; repeated_field, a field name or a list of
    # field names/indices locating the repeated field (a single name is wrapped into a list);
    # message_splitters, sequence of splitter classes tried in order on each element; extra
    # keyword arguments go to the base constructor. Stores the field path and splitter classes.
    """Initializer."""
    super().__init__(proto, **kwargs)
    if not isinstance(repeated_field, list):
      repeated_field = [repeated_field]
    self.repeated_field = repeated_field
    self.message_splitters = message_splitters

  def build_chunks(self) -> int:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the validity check raises only when the field is not repeated AND is a
    # message type; a non-repeated scalar field passes the check despite the error message.
    # Walks every element of the repeated field. Each element is first offered to the configured
    # element splitters, which may move parts of it into chunks; the returned sizes are
    # subtracted from the element size. Elements are accumulated, and when the running size
    # would reach the maximum size a new chunk (an empty message of the same type as the proto)
    # is started at that index. Afterwards the elements from the first cut onward are copied into
    # the new chunk messages and deleted from the original field.
    # Returns the total number of bytes moved out of the original proto. Raises a value error for
    # the invalid field kind noted above. Mutates the proto and registers chunks with the parent.
    """Splits the proto, and returns the size of the chunks created."""
    proto = self._proto

    total_size_diff = 0

    field, field_desc = util.get_field(proto, self.repeated_field)
    if not util.is_repeated(field_desc) and field_desc.message_type:
      raise ValueError(
          "RepeatedMessageSplitter can only be used on repeated fields. "
          f"Got proto={type(proto)}, field='{field_desc.name}'"
      )

    # List of indices at which to split the repeated field. For example, [3, 5]
    # means that the field list is split into: [:3], [3:5], [5:]
    repeated_msg_split = []
    # Should be the same length as the list above. Contains new protos to hold
    # the elements that are split from the original proto.
    # From the [3, 5] example above, the messages in this list contain nodes
    # [3:5] and [5:]
    repeated_msg_graphs = []
    # Track the total size of the current node split.
    total_size = 0

    # Linearly iterate through all nodes. It may be possible to optimize this
    # further by making best guesses as to where to split the nodes, since
    # most nodes (aside from constants) are relatively small.
    for n, ele in enumerate(field):
      size = ele.ByteSize()

      for splitter_cls in self.message_splitters:
        splitter = splitter_cls(
            ele,
            size,
            parent_splitter=self,
            fields_in_parent=self.repeated_field + [n],
        )
        size_diff = splitter.build_chunks()
        total_size_diff += size_diff
        size -= size_diff

      # Create a new GraphDef chunk if the current list of nodes is too large.
      if total_size + size >= constants.max_size():
        new_msg = type(self._proto)()
        repeated_msg_split.append(n)
        repeated_msg_graphs.append(new_msg)
        self.add_chunk(new_msg, [])

        if len(repeated_msg_split) >= 1:
          total_size_diff += total_size

        total_size = 0

      total_size += size

    if repeated_msg_split:
      # Finish writing repeated chunks.
      start = repeated_msg_split[0]
      for end, msg in zip(
          itertools.chain.from_iterable([repeated_msg_split[1:], [None]]),
          repeated_msg_graphs,
      ):
        _split_repeated_field(proto, msg, self.repeated_field, start, end)
        start = end
      del field[repeated_msg_split[0] :]

    return total_size_diff


class ConstantNodeDefSplitter(SplitBasedOnSize):
  # [CLASS DOC]
  # Element splitter that moves the tensor content of an oversized Const node into its own
  # chunk of raw bytes.
  """Extracts constant value from a `Const` NodeDef."""

  def build_chunks(self) -> int:
    # [FUNCTION DOC]
    # If the node size exceeds the maximum, extracts the constant bytes (raising a value error for
    # non-Const nodes), adds them as a chunk located at the tensor content field, and returns the
    # byte count. Otherwise does nothing and returns zero.
    """Splits a NodeDef proto, and returns the size of the chunks created."""
    if _ABOVE_MAX_SIZE(self.proto_size):
      constant_bytes = chunk_constant_value(self._proto, self.proto_size)
      self.add_chunk(
          constant_bytes,
          ["attr", "value", "tensor", "tensor_content"],
      )
      return len(constant_bytes)
    return 0


class LargeMessageSplitter(SplitBasedOnSize):
  # [CLASS DOC]
  # Moves an entire message into a separate chunk when a size predicate says it is big enough.
  # The predicate defaults to "larger than a third of the maximum size".
  """Splits a message into a separaet chunk if its over a certain size."""

  __slots__ = ("size_check",)

  def __init__(self, proto, proto_size, size_check=_GREEDY_SPLIT, **kwargs):
    # [FUNCTION DOC]
    # Inputs: proto and proto_size as in the base class; size_check, optional callback taking a
    # size and returning true when the message should be split off (default: greater than a third
    # of the maximum size); extra keyword arguments go to the base constructor.
    """Initializer."""

    self.size_check = size_check
    super().__init__(proto, proto_size, **kwargs)

  def build_chunks(self) -> int:
    # [FUNCTION DOC]
    # If the size check accepts the stored size, copies the whole proto into a new message of the
    # same type, clears the original proto, adds the copy as a chunk, and returns the original
    # size. Otherwise returns zero and changes nothing.
    """Creates a chunk for the entire proto and returns the original size."""
    if self.size_check(self.proto_size):
      new_proto = type(self._proto)()
      new_proto.MergeFrom(self._proto)
      self._proto.Clear()
      self.add_chunk(new_proto, [])
      return self.proto_size
    return 0


class FunctionDefSplitter(SplitBasedOnSize):
  # [CLASS DOC]
  # Element splitter for FunctionDef messages in the function library: moves a medium-sized
  # function wholesale into a chunk, or splits the nodes of a too-large function.
  """Splits the FunctionDef message type."""

  def build_chunks(self) -> int:
    # [FUNCTION DOC]
    # If the function is over a third of the max size but not above the max, moves the whole
    # function to a chunk. If it is above the max size, splits its node definitions with a
    # repeated-message splitter (constant splitting, then whole-message splitting).
    # Returns the total bytes moved out; mutates the proto and registers chunks.
    """Splits the proto, and returns the size of the chunks created."""
    size_diff = 0

    # First check if the entire FunctionDef can be split into a separate chunk.
    # We do this before the `RepeatedMessageSplitter`, which is costly because
    # it iterates through every `node_def`.
    if _GREEDY_SPLIT(self.proto_size) and not _ABOVE_MAX_SIZE(self.proto_size):
      size_diff += LargeMessageSplitter(
          self._proto,
          self.proto_size,
          parent_splitter=self,
          fields_in_parent=[],
      ).build_chunks()

    if _ABOVE_MAX_SIZE(self.proto_size):
      # Split FunctionDefLibrary.function.node_def
      size_diff += RepeatedMessageSplitter(
          self._proto,
          "node_def",
          [ConstantNodeDefSplitter, LargeMessageSplitter],
          parent_splitter=self,
          fields_in_parent=[],
      ).build_chunks()
    return size_diff
