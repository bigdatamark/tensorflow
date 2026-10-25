# [MODULE DOC]
# Unit tests for the GraphDef splitter. Builds synthetic GraphDefs with constant nodes of chosen
# sizes (also inside library functions), lowers the global maximum chunk size for debugging, runs
# the splitter, and checks the number, types, sizes and contents of the chunks and the resulting
# chunked-message metadata. Runs under the TensorFlow test runner when executed as a script.
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
#
# ==============================================================================
"""Tests for GraphDef splitter."""

import itertools

from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import test
from tensorflow.tools.proto_splitter import chunk_pb2
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter import split_graph_def
from tensorflow.tools.proto_splitter import util
from tensorflow.tools.proto_splitter.python import test_util


class GraphDefSplitterTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the GraphDef splitter: no split, large constants, many small nodes, large
  # nodes, and splitting of library functions. Shares helper methods for building and copying
  # graphs and for checking chunk sizes and field tags.

  def _make_graph_def_with_constant_nodes(
      self, node_sizes, dtype=None, **function_node_sizes
  ):
    # [FUNCTION DOC]
    # HELPER: builds test graphs for the tests in this class.
    # Thin wrapper over the shared test utility. Inputs: node_sizes, list of approximate byte
    # sizes for top-level constant nodes; dtype, optional element type; keyword arguments mapping
    # function names to lists of node sizes for library functions. Returns the new GraphDef.
    return test_util.make_graph_def_with_constant_nodes(
        node_sizes, dtype, **function_node_sizes
    )

  def _copy_graph(self, graph_def):
    # [FUNCTION DOC]
    # HELPER: gives tests an unmodified copy, since the splitter mutates its input.
    # Returns a new GraphDef holding a copy of the given graph_def.
    """Create a copy of GraphDef."""
    graph_def_copy = graph_pb2.GraphDef()
    graph_def_copy.CopyFrom(graph_def)
    return graph_def_copy

  def _assert_chunk_sizes(self, chunks, max_size):
    # [FUNCTION DOC]
    # HELPER: assertion shared by the tests.
    # For every chunk that is a protobuf message (raw byte chunks are skipped), asserts its
    # serialized size is at most max_size. Returns nothing; fails the test otherwise.
    """Asserts that all chunk proto sizes are <= max_size."""
    for chunk in chunks:
      if isinstance(chunk, message.Message):
        self.assertLessEqual(chunk.ByteSize(), max_size)

  def _assert_field_tags(self, expected_fields, actual_fields):
    # [FUNCTION DOC]
    # HELPER: assertion shared by the tests.
    # Asserts both lists of field tags have the same length and that each pair is equal as
    # protos. Returns nothing; fails the test on mismatch.
    self.assertLen(actual_fields, len(expected_fields))
    for expected, actual in zip(expected_fields, actual_fields):
      self.assertProtoEquals(expected, actual)

  def testSplitNoChunks(self):
    # [FUNCTION DOC]
    # Scenario: five small nodes totalling below the debug max size of 500. Splitting the graph
    # must produce exactly one chunk, equal to the original GraphDef.
    sizes = [50, 100, 50, 50, 100]
    max_size = 500
    constants.debug_set_max_size(max_size)

    graph_def = self._make_graph_def_with_constant_nodes(sizes)
    s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
    chunks, _ = s.split()
    self.assertLen(chunks, 1)
    self.assertProtoEquals(graph_def, chunks[0])

  def testLargeConstant(self):
    # [FUNCTION DOC]
    # Scenario: max size 500 with two constant nodes (indices 2 and 4) of 1000 bytes. Expects
    # three chunks, all message chunks within the limit; the tensor contents of nodes 4 and 2 are
    # chunks 2 and 1; the chunked message has two fields pointing at chunks 1 and 2 with field
    # tags for the tensor content of the respective nodes.
    sizes = [50, 50, 1000, 50, 1000]
    max_size = 500
    constants.debug_set_max_size(max_size)

    #    Expected Chunks (Max Size = 500)
    #    -----------------------------
    #       Chunk #: Contents
    #    -----------------------------
    #       0: GraphDef
    #    -----------------------------
    #       1: GraphDef.nodes[2].attr["value"].tensor.tensor_content
    #    -----------------------------
    #       2: GraphDef.nodes[4].attr["value"].tensor.tensor_content
    #    -----------------------------

    graph_def = self._make_graph_def_with_constant_nodes(sizes)
    s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
    chunks, chunked_message = s.split()
    self.assertLen(chunks, 3)
    self._assert_chunk_sizes(chunks, max_size)

    self.assertEqual(
        graph_def.node[4].attr["value"].tensor.tensor_content, chunks[2]
    )
    self.assertEqual(
        graph_def.node[2].attr["value"].tensor.tensor_content, chunks[1]
    )

    # Check the ChunkedMessage proto.
    self.assertLen(chunked_message.chunked_fields, 2)
    self.assertEqual(1, chunked_message.chunked_fields[0].message.chunk_index)
    self.assertEqual(2, chunked_message.chunked_fields[1].message.chunk_index)
    self._assert_field_tags(
        util.get_field_tag(
            graph_def, ["node", 2, "attr", "value", "tensor", "tensor_content"]
        ),
        chunked_message.chunked_fields[0].field_tag,
    )
    self._assert_field_tags(
        util.get_field_tag(
            graph_def, ["node", 4, "attr", "value", "tensor", "tensor_content"]
        ),
        chunked_message.chunked_fields[1].field_tag,
    )

  def testLotsOfNodes(self):
    # [FUNCTION DOC]
    # Scenario: fifteen nodes of about 95 bytes with max size 500. Expects three GraphDef
    # chunks whose concatenated nodes equal the original nodes in order, all within the limit,
    # and a chunked message with two fields referring to chunks 1 and 2 with empty field tags.
    # The actual sizes in the generated graph has a slight deviation, but are
    # between [90, 100] (tested in testMakeGraphDef with atol=5).
    sizes = [95] * 15
    max_size = 500
    constants.debug_set_max_size(500)

    #    Expected Chunks (Max Size = 500)
    #    -----------------------------
    #       Chunk #: Contents
    #    -----------------------------
    #       0: GraphDef  # (nodes [0:5])
    #    -----------------------------
    #       1: GraphDef  # (nodes [5:10])
    #    -----------------------------
    #       2: GraphDef  # (nodes [10:15])
    #    -----------------------------
    graph_def = self._make_graph_def_with_constant_nodes(sizes)
    s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
    chunks, chunked_message = s.split()
    self.assertLen(chunks, 3)
    self._assert_chunk_sizes(chunks, max_size)
    for node, chunk in zip(
        graph_def.node,
        itertools.chain(chunks[0].node, chunks[1].node, chunks[2].node),
    ):
      self.assertProtoEquals(node, chunk)

    # Check the ChunkedMessage proto.
    self.assertLen(chunked_message.chunked_fields, 2)
    self.assertEqual(1, chunked_message.chunked_fields[0].message.chunk_index)
    self.assertEqual(2, chunked_message.chunked_fields[1].message.chunk_index)
    self.assertEmpty(chunked_message.chunked_fields[0].field_tag)
    self.assertEmpty(chunked_message.chunked_fields[1].field_tag)

  def testLargeNodes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the in-test comment says six chunks are created, but the assertion (and the
    # listed contents) expects five.
    # Scenario: max size 200, six nodes where those over a third of the limit (indices 1, 2, 3
    # and 5) should be greedily moved into their own chunks. Expects five chunks within the
    # limit, the large nodes equal to chunks 1 to 4, small nodes remaining in the parent, empty
    # placeholders where large nodes were, and four chunked fields with matching field tags.
    # Large nodes are greedily split from the original proto if they are
    # larger than max_size / 3.
    sizes = [50, 95, 95, 95, 50, 95]
    max_size = 200
    constants.debug_set_max_size(max_size)

    # This should create 6 chunks:
    #   [parent GraphDef, node[1], node[2], node[3], node[5], ChunkedMessage]
    graph_def = self._make_graph_def_with_constant_nodes(sizes)
    s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
    chunks, chunked_message = s.split()
    self.assertLen(chunks, 5)
    self._assert_chunk_sizes(chunks, max_size)

    # Check large nodes are chunked away.
    self.assertProtoEquals(graph_def.node[1], chunks[1])
    self.assertProtoEquals(graph_def.node[2], chunks[2])
    self.assertProtoEquals(graph_def.node[3], chunks[3])
    self.assertProtoEquals(graph_def.node[5], chunks[4])

    # Check that the parent GraphDef still contains small nodes.
    self.assertProtoEquals(graph_def.node[0], chunks[0].node[0])
    self.assertProtoEquals(graph_def.node[4], chunks[0].node[4])

    # Check that the parent GraphDef contains empty nodes where the large nodes
    # were originally.
    self.assertEqual(0, chunks[0].node[1].ByteSize())
    self.assertEqual(0, chunks[0].node[2].ByteSize())
    self.assertEqual(0, chunks[0].node[3].ByteSize())
    self.assertEqual(0, chunks[0].node[5].ByteSize())

    # Check the ChunkedMessage proto.
    self.assertLen(chunked_message.chunked_fields, 4)
    self._assert_field_tags(
        util.get_field_tag(graph_def, ["node", 1]),
        chunked_message.chunked_fields[0].field_tag,
    )
    self._assert_field_tags(
        util.get_field_tag(graph_def, ["node", 2]),
        chunked_message.chunked_fields[1].field_tag,
    )
    self._assert_field_tags(
        util.get_field_tag(graph_def, ["node", 3]),
        chunked_message.chunked_fields[2].field_tag,
    )
    self._assert_field_tags(
        util.get_field_tag(graph_def, ["node", 5]),
        chunked_message.chunked_fields[3].field_tag,
    )

  def testFunctionLotsOfNodes(self):
    # [FUNCTION DOC]
    # Scenario: no top-level nodes, one library function with five 50-byte nodes, max size 200.
    # Expects two chunks (a GraphDef then a FunctionDef) within the limit, the function nodes
    # split across them in order, and a chunked message whose single field points at the
    # library function entry and chunk index 1.
    sizes = []
    fn1 = [50, 50, 50, 50, 50]
    max_size = 200
    constants.debug_set_max_size(max_size)

    graph_def = self._make_graph_def_with_constant_nodes(sizes, fn=fn1)
    s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
    chunks, chunked_message = s.split()

    self.assertLen(chunks, 2)
    self.assertIsInstance(chunks[0], graph_pb2.GraphDef)
    self.assertIsInstance(chunks[1], function_pb2.FunctionDef)
    self._assert_chunk_sizes(chunks, max_size)

    for node, chunk in zip(
        graph_def.library.function[0].node_def,
        itertools.chain(
            chunks[0].library.function[0].node_def, chunks[1].node_def
        ),
    ):
      self.assertProtoEquals(node, chunk)

    expected_message = chunk_pb2.ChunkedMessage()
    text_format.Parse(
        """
        chunk_index: 0
        chunked_fields {
            field_tag {
                field: 2
            }
            field_tag {
                field: 1
            }
            field_tag {
                index: 0
            }
            message {
                chunk_index: 1
            }
        }""",
        expected_message,
    )
    self.assertProtoEquals(expected_message, chunked_message)

  def testFunctionLargeNodes(self):
    # [FUNCTION DOC]
    # Scenario: one library function with nodes of 500, 500, 50 and 500 bytes, max size 200.
    # Expects four chunks within the limit: a GraphDef followed by the tensor contents of the
    # three large constants, in node order.
    sizes = []
    fn1 = [500, 500, 50, 500]
    max_size = 200
    constants.debug_set_max_size(max_size)

    graph_def = self._make_graph_def_with_constant_nodes(sizes, fn=fn1)
    s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
    chunks, _ = s.split()

    self.assertLen(chunks, 4)
    self._assert_chunk_sizes(chunks, max_size)
    self.assertIsInstance(chunks[0], graph_pb2.GraphDef)

    def get_const_value(index):
      # [FUNCTION DOC]
      # HELPER: reads expected data for the enclosing test. Given an index, returns the tensor
      # content bytes of that node definition of the first library function in the original
      # graph.
      node_def = graph_def.library.function[0].node_def[index]
      return node_def.attr["value"].tensor.tensor_content

    expected_values = [
        get_const_value(0),
        get_const_value(1),
        get_const_value(3),
    ]
    for expected, chunk in zip(expected_values, chunks[1:]):
      self.assertEqual(expected, chunk)

  def testChunkGraphDefAndFunctions(self):
    # [FUNCTION DOC]
    # Scenario: six 50-byte top-level nodes plus four library functions (one with three nodes,
    # three with one), max size 200. Expects four chunks, all within the limit (two GraphDef
    # node slices, the large function, and the function library).
    sizes = [50, 50, 50, 50, 50, 50]
    fn1 = [50, 50, 50]
    fn2 = [50]
    fn3 = [50]
    fn4 = [50]
    max_size = 200
    constants.debug_set_max_size(max_size)

    graph_def = self._make_graph_def_with_constant_nodes(
        sizes, fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4
    )
    s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
    chunks, _ = s.split()

    # Expected chunks:
    # GraphDef.nodes[:3], GraphDef.nodes[3:], fn1, FunctionDefLibrary
    self.assertLen(chunks, 4)
    self._assert_chunk_sizes(chunks, max_size)


if __name__ == "__main__":
  test.main()
