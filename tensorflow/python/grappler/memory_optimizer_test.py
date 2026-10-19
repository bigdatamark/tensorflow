# [MODULE DOC]
# Unit tests for the Grappler memory optimizer as reached through the Python graph optimization
# wrapper: swap-to-host annotations and recomputation rewrites (heuristics, name-scoped targets
# and recompute hints). Contains two test classes; runs via the standard test main entry point.
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
"""Tests for the swig wrapper tf_optimizer."""

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import training as train


class MemoryOptimizerSwapTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Grappler memory optimizer's swapping behavior through the Python graph
  # optimization entry point. Each test builds a tiny variable graph, runs the optimizer with a
  # hand-built rewriter configuration and inspects the resulting graph definition. No shared
  # fixtures beyond the default graph of the test framework.
  """Tests the Grappler memory optimizer."""

  @test_util.run_deprecated_v1
  def testNoSwapping(self):
    # [FUNCTION DOC]
    # Scenario: two variables and two add-n nodes are placed in the train-op collection, with no
    # swap annotations. The optimizer runs with model pruning, constant folding and dependency
    # optimization disabled and memory optimization set to manual. Asserts that the optimized
    # graph has the same number of nodes and the same set of node names as the original.
    """Make sure the graph is preserved when there is nothing to swap."""
    a = variable_v1.VariableV1(10, name='a')
    b = variable_v1.VariableV1(20, name='b')
    c = math_ops.add_n([a, b], name='c')
    d = math_ops.add_n([b, c], name='d')
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())
    graph_size = len(mg.graph_def.node)
    nodes = [node.name for node in mg.graph_def.node]

    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            disable_model_pruning=True,
            constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
            dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL))
    graph = tf_optimizer.OptimizeGraph(config, mg)

    self.assertEqual(len(graph.node), graph_size)
    self.assertItemsEqual([node.name for node in graph.node], nodes)

  @test_util.run_v1_only('b/120545219')
  def testSimpleSwap(self):
    # [FUNCTION DOC]
    # Scenario: on the first GPU device, builds the same small variable graph and marks the final
    # add node with a swap-to-host attribute. Runs the optimizer in manual memory mode with one
    # meta-optimizer iteration. Asserts the graph grows by exactly two nodes (a swap-out and a
    # swap-in node for the annotated input), that those named nodes exist, and that the swap-in,
    # swap-out and annotated node inputs are rewired as expected. Only runs under graph mode v1.
    """Check that the swap annotations are followed."""
    with ops.device('/gpu:0'):
      a = variable_v1.VariableV1(10, name='a')
      b = variable_v1.VariableV1(20, name='b')
      c = math_ops.add_n([a, b], name='c')
      d = math_ops.add_n([b, c], name='d')
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(d)

      d.op._set_attr('_swap_to_host', attr_value_pb2.AttrValue(i=0))

      mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())
      graph_size = len(mg.graph_def.node)

      config = config_pb2.ConfigProto()
      config.graph_options.rewrite_options.CopyFrom(
          rewriter_config_pb2.RewriterConfig(
              disable_model_pruning=True,
              meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE,
              constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
              memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL,
              min_graph_nodes=-1))
      graph = tf_optimizer.OptimizeGraph(config, mg)

      self.assertEqual(len(graph.node), graph_size + 2)
      self.assertTrue(
          set(node.name for node in graph.node) > set(
              ['a', 'b', 'c', 'd', 'swap_in_d_0', 'swap_out_d_0']))
      for node in graph.node:
        if node.name == 'swap_in_d_0':
          self.assertEqual('swap_out_d_0', node.input[0])
          self.assertEqual('^b/read', node.input[1])
        elif node.name == 'swap_out_d_0':
          self.assertEqual('b/read', node.input[0])
        elif node.name == 'd':
          self.assertEqual('swap_in_d_0', node.input[0])
          self.assertEqual('c', node.input[1])


class MemoryOptimizerRecomputeTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Python-facing side of the Grappler recomputation rewrites. It builds
  # layered convolution graphs trained with an Adam optimizer, runs the optimizer with various
  # memory-optimization settings and checks for recomputed nodes (names containing a recomputed
  # prefix). Shared fixtures are the helper methods that build the graphs and session configs.
  """Tests the Python interface to recomputation rewrites.

  See core/grappler/optimizers/memory_optimizer_test.cc for functional tests.
  """

  def _GetMetaGraph(self, batch_size=14, image_dim=12, optimizer_scope_name=''):
    # [FUNCTION DOC]
    # HELPER: graph-building fixture for the recomputation tests in this class.
    # Builds, in a fresh graph with a fixed random seed, ten layers of convolution, scaling by two
    # and ReLU over a variable activation, then a mean loss and an Adam training op created inside
    # a name scope. Inputs: batch_size (integer, default 14), image_dim (integer height and width,
    # default 12), optimizer_scope_name (string name scope for the optimizer, default empty).
    # Returns a four-element tuple: the exported meta graph, and the names of the variable
    # initializer op, the training op and the loss tensor.
    """A simple layered graph with conv, an intermediate op, and a ReLU."""
    graph = ops.Graph()
    with graph.as_default():
      random_seed.set_random_seed(1)
      current_activation = variable_scope.get_variable(
          name='start', shape=[batch_size, image_dim, image_dim, 5])
      conv_filter = variable_scope.get_variable(
          name='filter', shape=[5, 5, 5, 5])
      for layer_number in range(10):
        with variable_scope.variable_scope('layer_{}'.format(layer_number)):
          after_conv = nn.conv2d(current_activation, conv_filter, [1, 1, 1, 1],
                                 'SAME')
          current_activation = 2. * after_conv
          current_activation = nn.relu(current_activation)
      loss = math_ops.reduce_mean(current_activation)
      with ops.name_scope(optimizer_scope_name):
        optimizer = train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(loss)
      init_op = variables.global_variables_initializer()
      metagraph = train.export_meta_graph()
    return (metagraph, init_op.name, train_op.name, loss.name)

  def testRewritingDefaultGradientNames(self):
    # [FUNCTION DOC]
    # Scenario: builds the default meta graph and optimizes it with the recomputation-heuristics
    # memory mode (other optimizers off, no minimum node count). Asserts that the rewritten graph
    # has more nodes than the original, that the original has no recomputed nodes, and that the
    # rewritten graph has exactly twenty recomputed nodes (two per layer).
    """Tests that rewriting occurs with default gradient names."""
    (original_metagraph, _, _, _) = self._GetMetaGraph()
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            disable_model_pruning=True,
            constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
            dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
            arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            min_graph_nodes=-1,
            memory_optimization=(
                rewriter_config_pb2.RewriterConfig.RECOMPUTATION_HEURISTICS)))
    rewritten_graph_def = tf_optimizer.OptimizeGraph(config, original_metagraph)
    self.assertGreater(
        len(rewritten_graph_def.node),
        len(original_metagraph.graph_def.node))
    self.assertEqual(
        0,
        len([node for node in original_metagraph.graph_def.node
             if 'Recomputed/' in node.name]))
    self.assertEqual(
        20,  # Two per layer
        len([node for node in rewritten_graph_def.node
             if 'Recomputed/' in node.name]))

  def testRewritingNameScopedGradientNames(self):
    # [FUNCTION DOC]
    # Scenario: builds the meta graph with the optimizer inside a name scope called optimizer, so
    # gradient nodes live under a nested scope. Optimizes with recomputation heuristics and a
    # target node name scope of the gradients prefix, which must also match the sub-scope.
    # Asserts the graph grows and that exactly twenty recomputed nodes appear (none originally).
    """Tests that rewriting occurs with non-standard gradient names."""
    (original_metagraph, _, _, _) = self._GetMetaGraph(
        optimizer_scope_name='optimizer')
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            disable_model_pruning=True,
            constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
            dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
            arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            min_graph_nodes=-1,
            memory_optimization=rewriter_config_pb2.RewriterConfig
            .RECOMPUTATION_HEURISTICS,
            # Checks that name scope "gradients/" also match sub-scope.
            memory_optimizer_target_node_name_scope='gradients/'))
    rewritten_graph_def = tf_optimizer.OptimizeGraph(config, original_metagraph)
    self.assertGreater(
        len(rewritten_graph_def.node),
        len(original_metagraph.graph_def.node))
    self.assertEqual(
        0,
        len([node for node in original_metagraph.graph_def.node
             if 'Recomputed/' in node.name]))
    self.assertEqual(
        20,  # Two per layer
        len([node for node in rewritten_graph_def.node
             if 'Recomputed/' in node.name]))

  def testRewritingNameScopedGradientNamesScope(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is copied from the previous test and claims rewriting occurs, but
    # this test actually asserts that NO rewriting occurs because the target scope matches nothing.
    # Scenario: optimizer placed in a name scope of foo/bar, but the memory optimizer target node
    # name scope is set to a non-matching prefix. Asserts the node count is unchanged and that
    # neither the original nor the rewritten graph contains recomputed nodes.
    """Tests that rewriting occurs with non-standard gradient names."""
    (original_metagraph, _, _,
     _) = self._GetMetaGraph(optimizer_scope_name='foo/bar')
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            disable_model_pruning=True,
            constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
            dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
            arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
            memory_optimization=rewriter_config_pb2.RewriterConfig
            .RECOMPUTATION_HEURISTICS,
            # This should not match anything.
            memory_optimizer_target_node_name_scope='r/gradients/'))
    rewritten_graph_def = tf_optimizer.OptimizeGraph(config, original_metagraph)
    self.assertEqual(
        len(rewritten_graph_def.node), len(original_metagraph.graph_def.node))
    self.assertEqual(0,
                     len([
                         node for node in original_metagraph.graph_def.node
                         if 'Recomputed/' in node.name
                     ]))
    self.assertEqual(0,
                     len([
                         node for node in rewritten_graph_def.node
                         if 'Recomputed/' in node.name
                     ]))

  def _GetMemoryOptimizerSessionConfig(self):
    # [FUNCTION DOC]
    # HELPER: supplies a session configuration for the recomputation tests.
    # Takes no inputs and returns a session configuration message whose graph options enable the
    # default heuristics memory optimization with model pruning disabled. No side effects.
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True,
        memory_optimization=rewriter_config_pb2.RewriterConfig.HEURISTICS)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_options)
    return config_pb2.ConfigProto(graph_options=graph_options)

  def _RunMetaGraphWithConfig(
      self, config, metagraph, init_op_name, train_op_name, loss_op_name):
    # [FUNCTION DOC]
    # HELPER: executes a meta graph under a given session configuration for the tests here.
    # Inputs: config (session configuration message), metagraph (meta graph to import), and the
    # string names of the initializer op, training op and loss tensor. Imports the meta graph into
    # a fresh graph, opens a session with the configuration, runs the initializer, runs the
    # training op twice and returns the evaluated loss value (a number).
    graph = ops.Graph()
    with graph.as_default():
      train.import_meta_graph(metagraph)
      init_op = graph.get_operation_by_name(init_op_name)
      train_op = graph.get_operation_by_name(train_op_name)
      loss_op = graph.get_tensor_by_name(loss_op_name)
      with session.Session(config=config, graph=graph) as sess:
        self.evaluate(init_op)
        self.evaluate(train_op)
        self.evaluate(train_op)
        return self.evaluate(loss_op)

  def testRecomputationRewritingNoErrors(self):
    # [FUNCTION DOC]
    # Scenario: runs the default meta graph twice through the helper (initialize, train twice,
    # evaluate loss): once with a default configuration and once with the heuristics memory
    # optimizer enabled. Asserts the two loss values are close within a relative tolerance of one
    # percent, i.e. rewriting does not significantly change numerical output.
    """Tests that graph output is not significantly different with rewriting."""
    (original_metagraph, init_op_name, train_op_name, loss_op_name
    ) = self._GetMetaGraph()
    original_loss = self._RunMetaGraphWithConfig(
        config=config_pb2.ConfigProto(),
        metagraph=original_metagraph,
        init_op_name=init_op_name,
        train_op_name=train_op_name,
        loss_op_name=loss_op_name)
    memory_optimized_loss = self._RunMetaGraphWithConfig(
        config=self._GetMemoryOptimizerSessionConfig(),
        metagraph=original_metagraph,
        init_op_name=init_op_name,
        train_op_name=train_op_name,
        loss_op_name=loss_op_name)
    self.assertAllClose(original_loss, memory_optimized_loss, rtol=1e-2)

  def _annotated_graph(self):
    # [FUNCTION DOC]
    # HELPER: graph-building fixture for the recompute-hint tests.
    # Builds a small three-layer convolution graph (fixed random seed) in a new graph and sets a
    # recompute-hint attribute on the scaled, the added and the ReLU output ops (the attribute
    # value is irrelevant, only its presence matters). Adds a mean loss, an Adam training op and
    # a variable initializer. Returns a three-element tuple: the graph, the init op and train op.
    graph = ops.Graph()
    with graph.as_default():
      random_seed.set_random_seed(2)
      current_activation = variable_scope.get_variable(
          name='start', shape=[1, 2, 2, 5])
      conv_filter = variable_scope.get_variable(
          name='filter', shape=[5, 5, 5, 5])
      for layer_number in range(3):
        with variable_scope.variable_scope('layer_{}'.format(layer_number)):
          after_conv = nn.conv2d(current_activation, conv_filter, [1, 1, 1, 1],
                                 'SAME')
          current_activation = 2. * after_conv
          current_activation.op._set_attr(
              '_recompute_hint',
              # The value of the attribute does not matter; just that the key
              # exists in the op's attributes.
              attr_value_pb2.AttrValue(i=1))
          current_activation += 5.
          current_activation.op._set_attr(
              '_recompute_hint', attr_value_pb2.AttrValue(i=0))
          current_activation = nn.relu(current_activation)
          current_activation.op._set_attr(
              '_recompute_hint', attr_value_pb2.AttrValue(i=1))
      loss = math_ops.reduce_mean(current_activation)
      optimizer = train.AdamOptimizer(0.001)
      train_op = optimizer.minimize(loss)
      init_op = variables.global_variables_initializer()
    return graph, init_op, train_op

  def testHintNoMetaGraph(self):
    # [FUNCTION DOC]
    # Scenario: builds the annotated graph and runs it in a session whose configuration sets
    # memory optimization to manual, which honors recompute hints. Runs the initializer and one
    # training step. It asserts nothing explicitly; it only checks that execution raises no error.
    # It does not verify a rewrite happened (see the next test).
    # Closer to expected usage, but does not check that a re-write actually
    # happens; see testHintDoesRewrite.
    graph, init_op, train_op = self._annotated_graph()
    with graph.as_default():
      manual_memory_config = rewriter_config_pb2.RewriterConfig(
          memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL)
      graph_options = config_pb2.GraphOptions(
          rewrite_options=manual_memory_config)
      session_config = config_pb2.ConfigProto(graph_options=graph_options)
      with session.Session(config=session_config) as sess:
        self.evaluate(init_op)
        self.evaluate(train_op)

  @test_util.run_v1_only('b/120545219')
  def testHintDoesRewrite(self):
    # [FUNCTION DOC]
    # Scenario: exports a meta graph from the annotated graph and checks it has no recomputed
    # nodes. Optimizes it with manual memory optimization and no minimum node count. Asserts the
    # rewritten graph contains exactly nine recomputed nodes. Only runs under graph mode v1.
    graph = self._annotated_graph()[0]
    with graph.as_default():
      metagraph = train.export_meta_graph()
    self.assertEqual(
        0,
        len([node for node in metagraph.graph_def.node
             if 'Recomputed/' in node.name]))
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            min_graph_nodes=-1,
            memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL))
    rewritten_graph_def = tf_optimizer.OptimizeGraph(config, metagraph)
    self.assertEqual(
        9,
        len([
            node for node in rewritten_graph_def.node
            if 'Recomputed/' in node.name
        ]))

if __name__ == '__main__':
  test.main()
