# [MODULE DOC]
# HELPER: command-line generator script for test data (not library code). Builds a tree of
# string nodes and writes it, in text form, plus two chunked versions (depth-first and
# breadth-first chunk ordering) into the directory given by the path flag, for the C++ merger
# tests. Also writes the chunked-message layout as text for each ordering. Defines the path flag
# at import time; running as a script starts the application main function.
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

"""Generates test data for Merger.

Constructs depth- and breadth-first tree-like chunked protos test data for
Merger::Read and Merger::Merge.
"""

from collections.abc import Sequence
import os
from typing import Optional, Union

from absl import app
from absl import flags
from absl import logging

from google.protobuf import message
from tensorflow.python.lib.io import file_io
from tensorflow.tools.proto_splitter import chunk_pb2
from tensorflow.tools.proto_splitter import split
from tensorflow.tools.proto_splitter import util
from tensorflow.tools.proto_splitter.testdata import test_message_pb2

SPLITTER_TESTDATA_PATH = flags.DEFINE_string(
    "path", None, help="Path to testdata directory.")

_CHILD_NODES_FIELD_TAG = (
    test_message_pb2.StringNode.DESCRIPTOR.fields_by_name[
        "child_nodes"
    ].number
)


class StringNodeSplitter(split.ComposableSplitter):
  # [CLASS DOC]
  # HELPER: base class for the tree test-data splitters. Overrides chunk registration so that
  # every chunk is stored in the root splitter's chunk list while the tree-shaped chunked
  # message is filled in at each level. Subclasses choose the chunk ordering.
  """Splits a StringNode proto with N strings into a tree with depth N."""

  def __init__(self, proto: test_message_pb2.StringNode,
               chunked_message: Optional[chunk_pb2.ChunkedMessage] = None,
               **kwargs):
    # [FUNCTION DOC]
    # Inputs: proto, the string-node message to split; chunked_message, optional chunked-message
    # layout to fill for this node (used only if the base class did not already create one);
    # extra keyword arguments (parent splitter, fields in parent) go to the base constructor.
    super().__init__(proto, **kwargs)
    self._chunked_message = self._chunked_message or chunked_message

  def add_chunk(
      self, chunk: Union[message.Message, bytes], field_tags: util.FieldTypes
  ) -> None:
    # [FUNCTION DOC]
    # Overrides the base chunk registration. Inputs: chunk, a message or bytes; field_tags, the
    # path to the chunk's location in this node's proto. Appends a chunked field (with the field
    # tags generated from the path) to the layout, sets its chunk index to the current total
    # chunk count of the root, and stores the chunk in the root. Asserts (assertion error) that
    # a chunked message exists. Returns nothing.
    """Adds a new chunk and updates the ChunkedMessage proto."""
    assert self._chunked_message is not None
    field = self._chunked_message.chunked_fields.add(
        field_tag=util.get_field_tag(self._proto, field_tags)
    )
    field.message.chunk_index = self.total_chunks_len()
    self.add_root_chunk(chunk)

  def total_chunks_len(self) -> int:
    # [FUNCTION DOC]
    # HELPER: used when assigning chunk indices. Returns the number of chunks held by the root
    # splitter, asking the parent chain recursively when this splitter has a parent.
    """Returns length of chunks stored in root splitter."""
    if self._parent_splitter is not None:
      return self._parent_splitter.total_chunks_len()
    return len(self._chunks)

  def add_root_chunk(self, chunk: Union[message.Message, bytes]) -> None:
    # [FUNCTION DOC]
    # HELPER: used by chunk registration. Input: chunk, a message or bytes. Appends it to this
    # splitter's chunk list if it is the root, otherwise forwards it up to the parent. Asserts
    # that the root has a chunk list. Returns nothing.
    """Adds chunk to root splitter chunks."""
    if self._parent_splitter is None:
      assert self._chunks is not None
      self._chunks.append(chunk)
    else:
      self._parent_splitter.add_root_chunk(chunk)


class DFStringNodeSplitter(StringNodeSplitter):
  # [CLASS DOC]
  # HELPER: test-data splitter that orders chunks depth-first: each child is added, then its own
  # descendants are processed before the next sibling.
  """Depth-first string node splitter."""

  def build_chunks(self) -> Sequence[Union[message.Message, bytes]]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a sequence of chunks, but returns nothing (null) when
    # the node has no children; callers recursing on children ignore the value anyway.
    # Raises a type error if the proto is not a string node. For each child: adds it as a chunk,
    # then recursively splits it with a child splitter sharing the corresponding chunked-message
    # entry. Afterwards clears the child list. At the root only, appends the chunked message to
    # the chunk list, writes its text form to a text file for the depth-first tree in the path
    # directory (side effect), and returns the chunk list.
    if not isinstance(self._proto, test_message_pb2.StringNode):
      raise TypeError("Can only split TreeString type protos")

    if not self._proto.child_nodes:
      return
    for i, node in enumerate(self._proto.child_nodes):
      self.add_chunk(node, [_CHILD_NODES_FIELD_TAG, i])
      DFStringNodeSplitter(
          proto=node,
          parent_splitter=self,
          fields_in_parent=[_CHILD_NODES_FIELD_TAG],
          chunked_message=self._chunked_message.chunked_fields[i].message
      ).build_chunks()

    self._proto.ClearField("child_nodes")
    if self._parent_splitter is None:
      self._chunks.append(self._chunked_message)
      file_io.write_string_to_file(
          os.path.join(SPLITTER_TESTDATA_PATH.value, "df-split-tree.pbtxt"),
          str(self._chunked_message))

    return self._chunks


class BFStringNodeSplitter(StringNodeSplitter):
  # [CLASS DOC]
  # HELPER: test-data splitter that orders chunks breadth-first: all children of a node are
  # added before any child is split further.
  """Breadth-first string node splitter."""

  def build_chunks(self) -> Sequence[Union[message.Message, bytes]]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a sequence of chunks, but returns nothing (null) when
    # the node has no children.
    # Raises a type error if the proto is not a string node. Adds every child as a chunk first,
    # then recursively splits each child with a child splitter bound to its chunked-message
    # entry, then clears the child list. At the root only, appends the chunked message to the
    # chunk list, writes its text form to a text file for the breadth-first tree in the path
    # directory (side effect), and returns the chunk list.
    if not isinstance(self._proto, test_message_pb2.StringNode):
      raise TypeError("Can only split TreeString type protos")

    if not self._proto.child_nodes:
      return
    for i, node in enumerate(self._proto.child_nodes):
      self.add_chunk(node, [_CHILD_NODES_FIELD_TAG, i])
    for i, node in enumerate(self._proto.child_nodes):
      BFStringNodeSplitter(
          proto=node,
          parent_splitter=self,
          fields_in_parent=[_CHILD_NODES_FIELD_TAG],
          chunked_message=self._chunked_message.chunked_fields[i].message
      ).build_chunks()

    self._proto.ClearField("child_nodes")
    if self._parent_splitter is None:
      self._chunks.append(self._chunked_message)
      file_io.write_string_to_file(
          os.path.join(SPLITTER_TESTDATA_PATH.value, "bf-split-tree.pbtxt"),
          str(self._chunked_message))

    return self._chunks


def main(argv: Sequence[str]) -> None:
  # [FUNCTION DOC]
  # Entry point. Input: argv, command-line arguments (extra arguments raise a usage error; a
  # missing path flag also raises a usage error). Builds a four-level string tree where each node
  # at level k has k+1 children, logs it, writes its text form to a text file in the path
  # directory, then writes depth-first and breadth-first chunked versions (each from a fresh
  # copy of the tree) to files in the same directory. Returns nothing.
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if SPLITTER_TESTDATA_PATH.value is None:
    raise app.UsageError("'path' flag not specified.")

  levels = 4
  def make_string_tree(
      string_tree: test_message_pb2.StringNode, level: int = 0, label: str = "0"
  ) -> test_message_pb2.StringNode:
    # [FUNCTION DOC]
    # HELPER: nested tree builder for the entry point. Inputs: string_tree, node to fill; level,
    # current depth (default 0); label, text for the node value (default "0"). Sets the node's
    # value to the label and, until the last level, adds level plus one children recursively with
    # labels extended by the next level and child index. Returns the same node (mutated).
    string_tree.val = label
    if level >= levels-1:
      return string_tree
    for i in range(level+1):
      make_string_tree(string_tree.child_nodes.add(),
                       level+1, label+str(level+1)+str(i))
    return string_tree

  def copy_string_tree(string_tree: test_message_pb2.StringNode):
    # [FUNCTION DOC]
    # HELPER: nested copy helper for the entry point; returns a new, independent string node
    # equal to the given tree (needed because splitting mutates its input).
    new_tree = test_message_pb2.StringNode()
    new_tree.CopyFrom(string_tree)
    return new_tree

  string_tree = make_string_tree(test_message_pb2.StringNode())
  logging.info("StringNode tree generated:\n%s", string_tree)
  file_io.write_string_to_file(
      os.path.join(SPLITTER_TESTDATA_PATH.value, "split-tree.pbtxt"),
      str(string_tree))

  # depth-first chunk ordering
  DFStringNodeSplitter(copy_string_tree(string_tree)).write(
      os.path.join(SPLITTER_TESTDATA_PATH.value, "df-split-tree"))

  # breadth-first
  BFStringNodeSplitter(copy_string_tree(string_tree)).write(
      os.path.join(SPLITTER_TESTDATA_PATH.value, "bf-split-tree"))


if __name__ == "__main__":
  app.run(main)
