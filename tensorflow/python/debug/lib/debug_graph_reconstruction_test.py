# [MODULE DOC]
# Tests that tfdbg can reconstruct the original (non-debugger-decorated) partition graphs from the
# partition graphs of a debug-instrumented run: simple graphs, control edges, conditionals, while
# loops and gradient/training graphs. Test module only; compares graph definitions after removing
# send/receive style nodes that differ between instrumented and plain runs.
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
"""Tests for the reconstruction of non-debugger-decorated GraphDefs."""
import tempfile

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class ReconstructNonDebugGraphTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for graph reconstruction. Each test builds a small graph in a session, then uses the
  # shared comparison helper to run it plain and with debug watches dumping to a temporary
  # directory, reconstructing the original graph from the instrumented one. Holds a deny-list of
  # op types ignored in comparisons, and per-test dump directory and debug URL.

  _OP_TYPE_DENYLIST = ("_Send", "_Recv", "_HostSend", "_HostRecv", "_Retval")

  def _no_rewrite_session_config(self):
    # [FUNCTION DOC]
    # HELPER: builds a session configuration with dependency optimization and host pinning turned
    # off and the minimum graph node count for rewrites set to -1, so the executed graph stays
    # close to the one built. Returns the configuration object; no arguments.
    rewriter_config = rewriter_config_pb2.RewriterConfig(
        dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
        pin_to_host_optimization=rewriter_config_pb2.RewriterConfig.OFF,
        min_graph_nodes=-1)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    return config_pb2.ConfigProto(graph_options=graph_options)

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: runs the parent setup, creates a temporary dump directory and a file-style debug URL
    # pointing at it (stored on the instance), and resets the default graph.
    super(ReconstructNonDebugGraphTest, self).setUp()
    self._dump_dir = tempfile.mkdtemp()
    self._debug_url = "file://" + self._dump_dir
    ops.reset_default_graph()

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: recursively deletes the temporary dump directory, then runs the parent
    # teardown.
    file_io.delete_recursively(self._dump_dir)
    super(ReconstructNonDebugGraphTest, self).tearDown()

  def _graphDefWithoutDenylistedNodes(self, graph_def):
    # [FUNCTION DOC]
    # HELPER: normalizes a graph definition for comparison. Copies every node except those whose op
    # type is on the deny-list (send/receive/return ops). For while-loop Enter nodes, sets the
    # parallel iterations attribute to 1 (as the debugger does); for Switch and Identity nodes,
    # clears their inputs (they may be fed by send/receive nodes). Returns the new graph definition;
    # the input is not modified.
    output_graph_def = graph_pb2.GraphDef()
    for node in graph_def.node:
      if node.op not in self._OP_TYPE_DENYLIST:
        new_node = output_graph_def.node.add()
        new_node.CopyFrom(node)

        if new_node.op == "Enter":
          # The debugger sets parallel_iterations attribute of while-loop Enter
          # nodes to 1 for debugging.
          for attr_key in new_node.attr:
            if attr_key == "parallel_iterations":
              new_node.attr[attr_key].i = 1
        elif new_node.op == "Switch" or new_node.op == "Identity":
          # We don't check the inputs to Switch or Identity ops as their inputs
          # may be Send/Recv nodes.
          del new_node.input[:]

    return output_graph_def

  def _compareOriginalAndReconstructedGraphDefs(self,
                                                sess,
                                                fetches,
                                                feed_dict=None,
                                                expected_output=None):
    # [FUNCTION DOC]
    # HELPER: core assertion shared by all tests. Parameters: sess (session), fetches (what to
    # run), feed_dict (optional feeds), expected_output (optional value to compare outputs with).
    # Runs once plain and once with every node watched and dumped, collecting partition graphs
    # each time (asserting outputs when expected_output is given). Builds a dump directory object
    # with validation and gets its reconstructed non-debug partition graphs. Asserts the counts
    # match and, per partition, that the reconstruction equals the plain graph after normalizing,
    # both through the dump object and through the standalone reconstruction function.
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(fetches, feed_dict=feed_dict, options=run_options,
                      run_metadata=run_metadata)
    if expected_output is not None:
      self.assertAllClose(expected_output, output)
    non_debug_graph_defs = run_metadata.partition_graphs

    debug_utils.watch_graph(
        run_options, sess.graph, debug_urls=self._debug_url)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(fetches, feed_dict=feed_dict, options=run_options,
                      run_metadata=run_metadata)
    if expected_output is not None:
      self.assertAllClose(expected_output, output)

    dump = debug_data.DebugDumpDir(
        self._dump_dir, partition_graphs=run_metadata.partition_graphs,
        validate=True)
    reconstructed = dump.reconstructed_non_debug_partition_graphs()

    self.assertEqual(len(non_debug_graph_defs), len(reconstructed))
    for i, non_debug_graph_def in enumerate(non_debug_graph_defs):
      device_name = debug_graphs._infer_device_name(non_debug_graph_def)
      test_util.assert_equal_graph_def(
          self._graphDefWithoutDenylistedNodes(reconstructed[device_name]),
          self._graphDefWithoutDenylistedNodes(non_debug_graph_def))

      # Test debug_graphs.reconstruct_non_debug_graph_def.
      reconstructed_again = (
          debug_graphs.reconstruct_non_debug_graph_def(
              run_metadata.partition_graphs[i]))
      test_util.assert_equal_graph_def(
          self._graphDefWithoutDenylistedNodes(reconstructed_again),
          self._graphDefWithoutDenylistedNodes(non_debug_graph_def))

  def testReconstructSimpleGraph(self):
    # [FUNCTION DOC]
    # Scenario: add of two initialized variables. Asserts reconstruction matches and output is 42.
    with session.Session() as sess:
      u = variables.Variable([12.0], name="u")
      v = variables.Variable([30.0], name="v")
      w = math_ops.add(u, v, name="w")
      self.evaluate(u.initializer)
      self.evaluate(v.initializer)

      self._compareOriginalAndReconstructedGraphDefs(
          sess, w, expected_output=[42.0])

  def testReconstructGraphWithControlEdge(self):
    # [FUNCTION DOC]
    # Scenario: operations chained with control dependencies on a variable. Asserts the
    # reconstruction matches and the final value is 400.
    with session.Session() as sess:
      a = variables.Variable(10.0, name="a")
      with ops.control_dependencies([a]):
        b = math_ops.add(a, a, name="b")
      with ops.control_dependencies([a, b]):
        c = math_ops.multiply(b, b, name="c")
      self.evaluate(a.initializer)

      self._compareOriginalAndReconstructedGraphDefs(
          sess, c, expected_output=400.0)

  def testReconstructGraphWithCond(self):
    # [FUNCTION DOC]
    # Scenario: a conditional choosing between two additions, run in a session without graph
    # rewrites. Asserts reconstruction matches and the result is 21.
    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = variables.Variable(10.0, name="x")
      y = variables.Variable(20.0, name="y")
      cond = tf_cond.cond(
          x > y, lambda: math_ops.add(x, 1), lambda: math_ops.add(y, 1))
      self.evaluate(x.initializer)
      self.evaluate(y.initializer)

      self._compareOriginalAndReconstructedGraphDefs(
          sess, cond, expected_output=21.0)

  def testReconstructGraphWithWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario: a while loop that adds 2 to 10 while below 16, in a no-rewrite session. Asserts
    # the reconstruction matches the plain graph (no output check).
    with session.Session(config=self._no_rewrite_session_config()) as sess:
      loop_body = lambda i: math_ops.add(i, 2)
      loop_cond = lambda i: math_ops.less(i, 16)
      i = constant_op.constant(10, name="i")
      loop = while_loop.while_loop(loop_cond, loop_body, [i])

      self._compareOriginalAndReconstructedGraphDefs(sess, loop)

  def testReconstructGraphWithGradients(self):
    # [FUNCTION DOC]
    # Scenario: a toy loss minimized by a gradient descent optimizer (training op), no-rewrite
    # session. Asserts the reconstruction of the training graph matches (no output check).
    with session.Session(config=self._no_rewrite_session_config()) as sess:
      u = variables.Variable(12.0, name="u")
      v = variables.Variable(30.0, name="v")
      x = constant_op.constant(1.1, name="x")
      toy_loss = x * (u - v)
      train_op = gradient_descent.GradientDescentOptimizer(
          learning_rate=0.1).minimize(toy_loss, name="train_op")
      self.evaluate(u.initializer)
      self.evaluate(v.initializer)

      self._compareOriginalAndReconstructedGraphDefs(sess, train_op)


if __name__ == "__main__":
  test.main()
