# [MODULE DOC]
# KNOWN ISSUE: the module docstring says it tests tpu_function helpers, but the tests cover the
# TPU replicate context, TPU rewriting of layers, graph pruning, collective ops and einsum.
# Unit tests for the TPU graph-building code: the replicate context marking ops, name collisions
# in captured values, rewriting a loop that uses an infeed queue with layer regularizers,
# pruning unconnected ops from the compiled cluster, argument validation of the all-to-all op,
# and the einsum handling of the rewrite. Includes small module-level helpers that build and
# search graphs. Runs through the standard test main entry point when executed directly.
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
# =============================================================================

"""Tests for tpu_function helpers."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops


class TPUContextTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the TPU replicate context: marking of ops created inside it and handling of
  # value-name collisions between graphs. No shared fixtures.

  def testIsInContext(self):
    # [FUNCTION DOC]
    # Scenario: in a new graph, one identity op is made outside and one inside a manually entered
    # replicate context. Asserts the utility that checks for an XLA context reports false for
    # the outside op and true for the inside op.
    """Test that control_flow_util can check that we're in a TPU context."""
    with ops.Graph().as_default():
      z1 = array_ops.identity(1)
      pivot = control_flow_ops.no_op()
      context = tpu_replication.TPUReplicateContext(
          b"context", 1, pivot=pivot)
      context.Enter()
      z2 = array_ops.identity(1)
      context.Exit()
      self.assertFalse(control_flow_util.IsInXLAContext(z1.op))
      self.assertTrue(control_flow_util.IsInXLAContext(z2.op))

  def testHandlesNameCollision(self):
    # [FUNCTION DOC]
    # Scenario: an outer graph holds a tensor named "a"; inside a traced function a replicate
    # context is entered, that outer tensor is captured, and two new tensors also named "a" are
    # created and added. Asserts the names are as expected and that building the concrete
    # function succeeds (a past bug cached captured values by name and caused a shape mismatch).
    """Test AddValue handles name collisions for ops from different graphs."""
    with ops.Graph().as_default():
      z = array_ops.zeros([2, 3], name="a")
      assert z.name == "a:0", "Expected: a:0, Found: %s" % z.name

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: traced function for the name-collision test. Enters a replicate context,
        # captures the outer tensor, creates two same-named tensors, adds them, exits the
        # context and returns the sum.
        pivot = control_flow_ops.no_op()
        context = tpu_replication.TPUReplicateContext(
            b"context", 1, pivot=pivot)
        context.Enter()
        array_ops.identity(z)  # Capture z.
        z1 = array_ops.zeros([3, 2], name="a")
        assert z1.name == "a:0", "Expected: a:0, Found: %s" % z1.name
        z2 = array_ops.zeros([3, 2], name="a")
        # Prior to fixing b/166794533 this would fail with a shape mismatch
        # because context.AddValue would have cached `z` by its name which
        # collides with z1's name.
        result = z1 + z2
        context.Exit()
        return result

      f.get_concrete_function()


class TPULayerRewriteTest(test.TestCase):
  # [CLASS DOC]
  # Tests that TPU rewriting works with layers; currently one test about regularizers.

  def testUsingInfeedQueueWithRegularizer(self):
    # [FUNCTION DOC]
    # Scenario: builds a convolution layer whose kernel regularizer depends on a tensor fed through
    # an infeed queue, inside a repeat loop, and rewrites the loop for TPU. Asserts only that no
    # error is raised.
    """Test that Layer regularizers can reference data created in loops."""

    with ops.Graph().as_default():

      def make_regularizer(scale):
        # [FUNCTION DOC]
        # HELPER: takes a scale tensor and returns a regularizer function closing over it.
        def regularizer(inputs):
          # [FUNCTION DOC]
          # HELPER: regularizer returning the scale times the sum of squares of the inputs.
          return scale * math_ops.reduce_sum(math_ops.square(inputs))
        return regularizer

      def training_step(inputs, scale):
        # [FUNCTION DOC]
        # HELPER: loop body. Applies a channels-first convolution with sixteen filters and the
        # scale-based regularizer to the inputs and returns the operation of the mean squared
        # output as loss. The scale argument is only used by the regularizer.
        outputs = convolutional.conv2d(
            inputs,
            filters=16,
            kernel_size=(3, 3),
            data_format="channels_first",
            kernel_regularizer=make_regularizer(scale))
        loss = math_ops.reduce_mean(math_ops.square(outputs))
        return loss.op

      inputs = array_ops.zeros(shape=(128, 32, 32, 16))
      scale = array_ops.ones(shape=())
      infeed = tpu_feed.InfeedQueue(
          tuple_types=[dtypes.float32, dtypes.float32],
          tuple_shapes=[inputs.shape, scale.shape])

      def loop():
        # [FUNCTION DOC]
        # HELPER: computation passed to TPU rewrite; repeats the training step five times with
        # the infeed queue supplying its inputs. Returns the loop result.
        return training_loop.repeat(5, training_step, infeed_queue=infeed)

      # This should not throw an error.
      tpu.rewrite(loop)


class TPUGraphPruneTest(test.TestCase):
  # [CLASS DOC]
  # Tests for pruning ops that are not connected to the TPU cluster.

  def test_prune_unconnected_ops(self):
    # [FUNCTION DOC]
    # Scenario: builds a graph with placeholders a and b, a constant, variables x and y and two
    # adds; tags every node as TPU-replicated and rewires inputs so a and y become unconnected
    # to the rest. Re-imports the graph and prunes unconnected ops. Asserts ops a and x keep the
    # replicate attribute while b and y lose it, with an error naming the missing attribute.
    with ops.Graph().as_default():
      a = array_ops.placeholder(dtype=dtypes.float32, name="a")
      b = array_ops.placeholder(dtype=dtypes.float32, name="b")
      constant_op.constant(1.0, name="constant")
      x = variable_scope.get_variable(
          name="x",
          dtype=dtypes.float32,
          shape=[],
          use_resource=True,
          initializer=init_ops.constant_initializer(2.0))
      y = variable_scope.get_variable(
          name="y",
          dtype=dtypes.float32,
          shape=[],
          use_resource=True,
          initializer=init_ops.constant_initializer(3.0))
      math_ops.add(a, b)
      math_ops.add(x, y)
      graph_def = ops.get_default_graph().as_graph_def()

      for node in graph_def.node:
        # Attach a TPU_REPLICATE_ATTR to each node.
        node.attr[tpu_replication._TPU_REPLICATE_ATTR].s = b"0"
        # Rewire placeholder "a" and variable "y" leaving them unconnected.
        for (input_index, node_input) in enumerate(node.input):
          if node_input == "b":
            node.input[input_index] = "constant"
          if node_input == "y":
            node.input[input_index] = "x"

    with ops.Graph().as_default() as graph:
      # Reimport the graph and prune unconnected ops.
      importer.import_graph_def(graph_def)
      tpu.prune_unconnected_ops_from_xla(ops.get_default_graph())

      # Verify that ops "a" and "x" still have TPU_REPLICATE_ATTR.
      a = graph.get_operation_by_name("import/a").get_attr(
          tpu_replication._TPU_REPLICATE_ATTR)
      self.assertEqual(b"0", a)
      x = graph.get_operation_by_name("import/x").get_attr(
          tpu_replication._TPU_REPLICATE_ATTR)
      self.assertEqual(b"0", x)
      # Verify that ops "b" and "y" have TPU_REPLICATE_ATTR removed.
      with self.assertRaisesRegex(
          ValueError,
          "Operation \'import/b\' has no attr named \'_tpu_replicate\'"):
        graph.get_operation_by_name("import/b").get_attr(
            tpu_replication._TPU_REPLICATE_ATTR)
      with self.assertRaisesRegex(
          ValueError,
          "Operation \'import/y\' has no attr named \'_tpu_replicate\'"):
        graph.get_operation_by_name("import/y").get_attr(
            tpu_replication._TPU_REPLICATE_ATTR)


class TPUOpsTest(test.TestCase):
  # [CLASS DOC]
  # Tests argument validation of the TPU all-to-all op wrapper.

  def test_all_to_all_zero_split_count(self):
    # [FUNCTION DOC]
    # Scenario: all-to-all with split count zero. Asserts a value error saying the split count
    # must be at least one.
    with self.assertRaisesRegex(
        ValueError, "split_count 0 must at least be one"):
      tpu_ops.all_to_all(
          x=[0.0, 0.1652, 0.6543],
          group_assignment=[1, -1],
          concat_dimension=0,
          split_dimension=0,
          split_count=0)

  def test_all_to_all_group_assignment_wrong_shape(self):
    # [FUNCTION DOC]
    # Scenario: group assignment given as a rank-one list. Asserts a value error saying it must
    # have rank two.
    with self.assertRaisesRegex(
        ValueError, "group_assignment must have rank 2"):
      tpu_ops.all_to_all(
          x=[0.0, 0.1652, 0.6543],
          group_assignment=[1, -1],
          concat_dimension=0,
          split_dimension=0,
          split_count=2)

  def test_all_to_all_split_count_not_equal_to_group_assignment_shape(self):
    # [FUNCTION DOC]
    # Scenario: split count one with a group assignment whose second dimension is two. Asserts a
    # value error saying the split count must equal that dimension size.
    with self.assertRaisesRegex(
        ValueError, "split_count 1 must equal the size of the second dimension "
        "of group_assignment 2"):
      tpu_ops.all_to_all(
          x=[0.0, 0.1652, 0.6543],
          group_assignment=[[0, 1], [2, 3]],
          concat_dimension=0,
          split_dimension=0,
          split_count=1)

  def test_all_to_all_split_count_not_divide_input_shape(self):
    # [FUNCTION DOC]
    # Scenario: input of three rows split along dimension zero with split count two. Asserts a
    # value error saying the input dimension is not divisible by the split count.
    with self.assertRaisesRegex(
        ValueError, "input dimension 3 not divisible by split_count 2"):
      tpu_ops.all_to_all(
          x=[[0.0], [0.1652], [0.6543]],
          group_assignment=[[0, 1], [2, 3]],
          concat_dimension=1,
          split_dimension=0,
          split_count=2)


def do_einsum():
  # [FUNCTION DOC]
  # HELPER: builds two float placeholders (shapes two by three by four and two by four by five)
  # in the current graph and returns their batched matrix product via the einsum op.
  a = array_ops.placeholder(dtype=dtypes.float32, name="a", shape=[2, 3, 4])
  b = array_ops.placeholder(dtype=dtypes.float32, name="b", shape=[2, 4, 5])
  return special_math_ops.einsum("abc,acd->abd", a, b)


def find_einsum(g):
  # [FUNCTION DOC]
  # HELPER: takes a graph and returns true if any node is a plain Einsum op, else false.
  graph_def = g.as_graph_def()
  for node in graph_def.node:
    if node.op == "Einsum":
      return True
  return False


def find_xla_einsum(g):
  # [FUNCTION DOC]
  # HELPER: takes a graph and returns true if any node is an XLA einsum op, else false.
  graph_def = g.as_graph_def()
  for node in graph_def.node:
    if node.op == "XlaEinsum":
      return True
  return False


class TPUXlaEinsumTest(test.TestCase):
  # [CLASS DOC]
  # Tests which einsum op ends up in graphs with and without TPU rewriting.

  def test_tpu_rewrite_uses_xla_einsum(self):
    # [FUNCTION DOC]
    # Scenario: the einsum helper is TPU-rewritten in a new graph. Asserts the graph contains
    # an Einsum or XLA einsum node.
    with ops.Graph().as_default() as g:
      tpu.rewrite(do_einsum)
      self.assertTrue(find_einsum(g) or find_xla_einsum(g))

  def test_default_does_not_use_xla_einsum(self):
    # [FUNCTION DOC]
    # Scenario: the einsum helper is run without TPU rewriting. Asserts no XLA einsum node appears.
    with ops.Graph().as_default() as g:
      do_einsum()
      self.assertFalse(find_xla_einsum(g))


if __name__ == "__main__":
  test.main()
