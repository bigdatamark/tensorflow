# [MODULE DOC]
# Unit tests for the tfdbg utility functions that add debug tensor watches to run options. A
# single test class builds a small graph (two variables, a constant, a matrix product and an
# addition) in a session and checks the watch entries produced by adding single watches and by
# watching the whole graph with allowlists and denylists of node name, op type and dtype.
# Only runs under the graph-mode (version 1) session API.
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
"""Tests for TensorFlow Debugger (tfdbg) Utilities."""
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import googletest


@test_util.run_v1_only("Requires tf.Session")
class DebugUtilsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the debug watch utilities. Shared fixtures: a class-level session and graph with
  # nodes for two variables (a, b), a constant (c), a matrix product (p1) and a sum (s), plus the
  # expected number of watches when everything is watched; and a fresh run-options message per
  # test.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level setup: creates a session, builds the shared graph (two initialized-from-constant
    # variables, a constant containing a not-a-number value, their matrix product and a sum),
    # stores the graph, and computes the expected watch count (four nodes per variable, one
    # constant, one add, one matmul, plus one wildcard watch).
    cls._sess = session.Session()
    with cls._sess:
      cls._a_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      cls._b_init_val = np.array([[2.0], [-1.0]])
      cls._c_val = np.array([[-4.0], [np.nan]])

      cls._a_init = constant_op.constant(
          cls._a_init_val, shape=[2, 2], name="a1_init")
      cls._b_init = constant_op.constant(
          cls._b_init_val, shape=[2, 1], name="b_init")

      cls._a = variable_v1.VariableV1(cls._a_init, name="a1")
      cls._b = variable_v1.VariableV1(cls._b_init, name="b")
      cls._c = constant_op.constant(cls._c_val, shape=[2, 1], name="c")

      # Matrix product of a and b.
      cls._p = math_ops.matmul(cls._a, cls._b, name="p1")

      # Sum of two vectors.
      cls._s = math_ops.add(cls._p, cls._c, name="s")

    cls._graph = cls._sess.graph

    # These are all the expected nodes in the graph:
    #   - Two variables (a, b), each with four nodes (Variable, init, Assign,
    #     read).
    #   - One constant (c).
    #   - One add operation and one matmul operation.
    #   - One wildcard node name ("*") that covers nodes created internally
    #     by TensorFlow itself (e.g., Grappler).
    cls._expected_num_nodes = 4 * 2 + 1 + 1 + 1 + 1

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test setup: creates a fresh empty run-options message stored on the test instance.
    self._run_options = config_pb2.RunOptions()

  def _verify_watches(self, watch_opts, expected_output_slot,
                      expected_debug_ops, expected_debug_urls):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for the graph-watching tests.
    # Inputs: the repeated field of watch entries, the expected output slot (integer), expected
    # debug ops (list of strings) and expected debug URLs (list of strings). For each watch it
    # asserts the debug ops and URLs match, and that the output slot matches, except for the
    # wildcard node name, which must have slot minus one. Returns the list of node names in the
    # watches, in order.
    """Verify a list of debug tensor watches.

    This requires all watches in the watch list have exactly the same
    output_slot, debug_ops and debug_urls.

    Args:
      watch_opts: Repeated protobuf field of DebugTensorWatch.
      expected_output_slot: Expected output slot index, as an integer.
      expected_debug_ops: Expected debug ops, as a list of strings.
      expected_debug_urls: Expected debug URLs, as a list of strings.

    Returns:
      List of node names from the list of debug tensor watches.
    """
    node_names = []
    for watch in watch_opts:
      node_names.append(watch.node_name)

      if watch.node_name == "*":
        self.assertEqual(-1, watch.output_slot)
        self.assertEqual(expected_debug_ops, watch.debug_ops)
        self.assertEqual(expected_debug_urls, watch.debug_urls)
      else:
        self.assertEqual(expected_output_slot, watch.output_slot)
        self.assertEqual(expected_debug_ops, watch.debug_ops)
        self.assertEqual(expected_debug_urls, watch.debug_urls)

    return node_names

  def testAddDebugTensorWatches_defaultDebugOp(self):
    # [FUNCTION DOC]
    # Adds two single-tensor watches with different nodes, slots and URLs but no debug op given.
    # Asserts two watch entries exist with the right node names and slots, that the debug op
    # defaults to the identity debug op, and that each has its own URL.
    debug_utils.add_debug_tensor_watch(
        self._run_options, "foo/node_a", 1, debug_urls="file:///tmp/tfdbg_1")
    debug_utils.add_debug_tensor_watch(
        self._run_options, "foo/node_b", 0, debug_urls="file:///tmp/tfdbg_2")

    debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
    self.assertEqual(2, len(debug_watch_opts))

    watch_0 = debug_watch_opts[0]
    watch_1 = debug_watch_opts[1]

    self.assertEqual("foo/node_a", watch_0.node_name)
    self.assertEqual(1, watch_0.output_slot)
    self.assertEqual("foo/node_b", watch_1.node_name)
    self.assertEqual(0, watch_1.output_slot)
    # Verify default debug op name.
    self.assertEqual(["DebugIdentity"], watch_0.debug_ops)
    self.assertEqual(["DebugIdentity"], watch_1.debug_ops)

    # Verify debug URLs.
    self.assertEqual(["file:///tmp/tfdbg_1"], watch_0.debug_urls)
    self.assertEqual(["file:///tmp/tfdbg_2"], watch_1.debug_urls)

  def testAddDebugTensorWatches_explicitDebugOp(self):
    # [FUNCTION DOC]
    # Adds one watch with an explicit single debug op (nan count). Asserts one entry with the
    # given node and slot, debug ops equal to a list holding that op, and the single URL.
    debug_utils.add_debug_tensor_watch(
        self._run_options,
        "foo/node_a",
        0,
        debug_ops="DebugNanCount",
        debug_urls="file:///tmp/tfdbg_1")

    debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
    self.assertEqual(1, len(debug_watch_opts))

    watch_0 = debug_watch_opts[0]

    self.assertEqual("foo/node_a", watch_0.node_name)
    self.assertEqual(0, watch_0.output_slot)

    # Verify default debug op name.
    self.assertEqual(["DebugNanCount"], watch_0.debug_ops)

    # Verify debug URLs.
    self.assertEqual(["file:///tmp/tfdbg_1"], watch_0.debug_urls)

  def testAddDebugTensorWatches_multipleDebugOps(self):
    # [FUNCTION DOC]
    # Adds one watch with a list of two debug ops. Asserts a single entry whose debug ops equal
    # the given list in order, with the expected node, slot and URL.
    debug_utils.add_debug_tensor_watch(
        self._run_options,
        "foo/node_a",
        0,
        debug_ops=["DebugNanCount", "DebugIdentity"],
        debug_urls="file:///tmp/tfdbg_1")

    debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
    self.assertEqual(1, len(debug_watch_opts))

    watch_0 = debug_watch_opts[0]

    self.assertEqual("foo/node_a", watch_0.node_name)
    self.assertEqual(0, watch_0.output_slot)

    # Verify default debug op name.
    self.assertEqual(["DebugNanCount", "DebugIdentity"], watch_0.debug_ops)

    # Verify debug URLs.
    self.assertEqual(["file:///tmp/tfdbg_1"], watch_0.debug_urls)

  def testAddDebugTensorWatches_multipleURLs(self):
    # [FUNCTION DOC]
    # Adds one watch with a list of two URLs. Asserts a single entry with the expected node, slot
    # and debug op, and with both URLs preserved in order.
    debug_utils.add_debug_tensor_watch(
        self._run_options,
        "foo/node_a",
        0,
        debug_ops="DebugNanCount",
        debug_urls=["file:///tmp/tfdbg_1", "file:///tmp/tfdbg_2"])

    debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
    self.assertEqual(1, len(debug_watch_opts))

    watch_0 = debug_watch_opts[0]

    self.assertEqual("foo/node_a", watch_0.node_name)
    self.assertEqual(0, watch_0.output_slot)

    # Verify default debug op name.
    self.assertEqual(["DebugNanCount"], watch_0.debug_ops)

    # Verify debug URLs.
    self.assertEqual(["file:///tmp/tfdbg_1", "file:///tmp/tfdbg_2"],
                     watch_0.debug_urls)

  def testWatchGraph_allNodes(self):
    # [FUNCTION DOC]
    # Watches the whole graph with two debug ops and one URL and no filters. Asserts the watch
    # count equals the expected total, all watches carry the expected ops and URL, every node of
    # the fixture graph appears among the watched names, and the wildcard watch is present.
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_ops=["DebugIdentity", "DebugNanCount"],
        debug_urls="file:///tmp/tfdbg_1")

    debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
    self.assertEqual(self._expected_num_nodes, len(debug_watch_opts))

    # Verify that each of the nodes in the graph with output tensors in the
    # graph have debug tensor watch.
    node_names = self._verify_watches(debug_watch_opts, 0,
                                      ["DebugIdentity", "DebugNanCount"],
                                      ["file:///tmp/tfdbg_1"])

    # Verify the node names.
    self.assertIn("a1_init", node_names)
    self.assertIn("a1", node_names)
    self.assertIn("a1/Assign", node_names)
    self.assertIn("a1/read", node_names)

    self.assertIn("b_init", node_names)
    self.assertIn("b", node_names)
    self.assertIn("b/Assign", node_names)
    self.assertIn("b/read", node_names)

    self.assertIn("c", node_names)
    self.assertIn("p1", node_names)
    self.assertIn("s", node_names)

    # Assert that the wildcard node name has been created.
    self.assertIn("*", node_names)

  def testWatchGraph_nodeNameAllowlist(self):
    # [FUNCTION DOC]
    # Watches the graph with a node-name allowlist pattern selecting the first variable's nodes
    # and the matmul. Asserts exactly those node names are watched (no wildcard).
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_allowlist="(a1$|a1_init$|a1/.*|p1$)")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertEqual(
        sorted(["a1_init", "a1", "a1/Assign", "a1/read", "p1"]),
        sorted(node_names))

  def testWatchGraph_opTypeAllowlist(self):
    # [FUNCTION DOC]
    # Watches the graph with an op-type allowlist matching variables and matrix multiplication.
    # Asserts only the two variable nodes and the matmul node are watched.
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        op_type_regex_allowlist="(Variable|MatMul)")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertEqual(sorted(["a1", "b", "p1"]), sorted(node_names))

  def testWatchGraph_nodeNameAndOpTypeAllowlists(self):
    # [FUNCTION DOC]
    # Combines a node-name allowlist and an op-type allowlist (matmul only). Asserts the filters
    # act as a logical AND, leaving only the matmul node.
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_allowlist="([a-z]+1$)",
        op_type_regex_allowlist="(MatMul)")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertEqual(["p1"], node_names)

  def testWatchGraph_tensorDTypeAllowlist(self):
    # [FUNCTION DOC]
    # Watches the graph with a dtype allowlist matching reference-typed tensors. Asserts only the
    # variable and variable-assign nodes are watched (order-insensitive).
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        tensor_dtype_regex_allowlist=".*_ref")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertItemsEqual(["a1", "a1/Assign", "b", "b/Assign"], node_names)

  def testWatchGraph_nodeNameAndTensorDTypeAllowlists(self):
    # [FUNCTION DOC]
    # Combines a node-name allowlist (names starting with a) and a reference-dtype allowlist.
    # Asserts only the first variable and its assign node are watched.
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_allowlist="^a.*",
        tensor_dtype_regex_allowlist=".*_ref")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertItemsEqual(["a1", "a1/Assign"], node_names)

  def testWatchGraph_nodeNameDenylist(self):
    # [FUNCTION DOC]
    # Watches the graph using a node-name denylist covering the first variable's nodes and the
    # matmul. Asserts the watched names are exactly the remaining nodes (second variable's four
    # nodes, the constant and the sum).
    debug_utils.watch_graph_with_denylists(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_denylist="(a1$|a1_init$|a1/.*|p1$)")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertEqual(
        sorted(["b_init", "b", "b/Assign", "b/read", "c", "s"]),
        sorted(node_names))

  def testWatchGraph_opTypeDenylist(self):
    # [FUNCTION DOC]
    # Watches the graph using an op-type denylist covering variables, identity, assign and
    # constants. Asserts only the matmul and the sum nodes remain.
    debug_utils.watch_graph_with_denylists(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        op_type_regex_denylist="(Variable|Identity|Assign|Const)")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertEqual(sorted(["p1", "s"]), sorted(node_names))

  def testWatchGraph_nodeNameAndOpTypeDenylists(self):
    # [FUNCTION DOC]
    # Combines a node-name denylist (the matmul) and an op-type denylist. Asserts the filters act
    # as a logical OR of exclusions, leaving only the sum node.
    debug_utils.watch_graph_with_denylists(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_denylist="p1$",
        op_type_regex_denylist="(Variable|Identity|Assign|Const)")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertEqual(["s"], node_names)

  def testWatchGraph_tensorDTypeDenylists(self):
    # [FUNCTION DOC]
    # Watches the graph using a dtype denylist matching reference-typed tensors. Asserts the
    # variable and assign nodes are absent from the watches while the sum node is present.
    debug_utils.watch_graph_with_denylists(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        tensor_dtype_regex_denylist=".*_ref")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertNotIn("a1", node_names)
    self.assertNotIn("a1/Assign", node_names)
    self.assertNotIn("b", node_names)
    self.assertNotIn("b/Assign", node_names)
    self.assertIn("s", node_names)

  def testWatchGraph_nodeNameAndTensorDTypeDenylists(self):
    # [FUNCTION DOC]
    # Combines a node-name denylist (the sum node) and a reference-dtype denylist. Asserts the
    # variable nodes, their assign nodes and the sum node are all absent from the watches.
    debug_utils.watch_graph_with_denylists(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_denylist="^s$",
        tensor_dtype_regex_denylist=".*_ref")

    node_names = self._verify_watches(
        self._run_options.debug_options.debug_tensor_watch_opts, 0,
        ["DebugIdentity"], ["file:///tmp/tfdbg_1"])
    self.assertNotIn("a1", node_names)
    self.assertNotIn("a1/Assign", node_names)
    self.assertNotIn("b", node_names)
    self.assertNotIn("b/Assign", node_names)
    self.assertNotIn("s", node_names)


if __name__ == "__main__":
  googletest.main()
