# [MODULE DOC]
# Unit tests for the meta graph utilities module: exporting and importing MetaGraphDef messages
# (whole graphs and name-scoped subgraphs), stripped op lists (including nested, recursive and
# partitioned-call functions), default-valued attribute stripping, collection handling (shared
# variable objects, metric variables, selective restore), unbound inputs, queues, while loops,
# device clearing, and partitioned variables across scopes. Most tests build graphs explicitly, use
# temporary directories under the test temp dir, and run under the graph-mode test decorators.
# Contains small shared helper function for test directories.
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
# =============================================================================
"""Tests for tensorflow.python.framework.meta_graph.py."""

import math
import os.path
import random
import shutil

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import queue_runner_impl


# pylint: disable=invalid-name
def _TestDir(test_name):
  # [FUNCTION DOC]
  # HELPER: test utility giving each test its own scratch directory. Input: a test name. Creates
  # (after deleting any previous one) a directory with that name under the test temp directory and
  # returns its path. Side effects: removes and recreates files on disk.
  test_dir = os.path.join(test.get_temp_dir(), test_name)
  if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
  gfile.MakeDirs(test_dir)
  return test_dir


# pylint: enable=invalid-name


class SimpleMetaGraphTest(test.TestCase):
  # [CLASS DOC]
  # Tests for basic meta graph export and import, op list stripping, default attribute stripping
  # and collection handling of variables. No shared fixtures.

  @test_util.run_deprecated_v1
  def testNoVariables(self):
    # [FUNCTION DOC]
    # Test: builds a graph with a placeholder plus constant add and two collections holding the
    # input and output tensors, runs it, exports a meta graph to a file and checks meta info
    # (version strings present, no variables). Imports the file into a fresh graph, re-exports and
    # asserts the protos are equal, then asserts the collections give tensors that compute the same
    # result as the original.
    test_dir = _TestDir("no_variables")
    filename = os.path.join(test_dir, "metafile")

    input_feed_value = -10  # Arbitrary input value for feed_dict.

    orig_graph = ops.Graph()
    with self.session(graph=orig_graph) as sess:
      # Create a minimal graph with zero variables.
      input_tensor = array_ops.placeholder(
          dtypes.float32, shape=[], name="input")
      offset = constant_op.constant(42, dtype=dtypes.float32, name="offset")
      output_tensor = math_ops.add(input_tensor, offset, name="add_offset")

      # Add input and output tensors to graph collections.
      ops.add_to_collection("input_tensor", input_tensor)
      ops.add_to_collection("output_tensor", output_tensor)

      output_value = sess.run(output_tensor, {input_tensor: input_feed_value})
      self.assertEqual(output_value, 32)

      # Generates MetaGraphDef.
      meta_graph_def, var_list = meta_graph.export_scoped_meta_graph(
          filename=filename,
          graph_def=ops.get_default_graph().as_graph_def(add_shapes=True),
          collection_list=["input_tensor", "output_tensor"],
          saver_def=None)
      self.assertTrue(meta_graph_def.HasField("meta_info_def"))
      self.assertNotEqual(meta_graph_def.meta_info_def.tensorflow_version, "")
      self.assertNotEqual(meta_graph_def.meta_info_def.tensorflow_git_version,
                          "")
      self.assertEqual({}, var_list)

    # Create a clean graph and import the MetaGraphDef nodes.
    new_graph = ops.Graph()
    with self.session(graph=new_graph) as sess:
      # Import the previously export meta graph.
      meta_graph.import_scoped_meta_graph(filename)

      # Re-exports the current graph state for comparison to the original.
      new_meta_graph_def, _ = meta_graph.export_scoped_meta_graph(filename +
                                                                  "_new")
      test_util.assert_meta_graph_protos_equal(self, meta_graph_def,
                                               new_meta_graph_def)

      # Ensures that we can still get a reference to our graph collections.
      new_input_tensor = ops.get_collection("input_tensor")[0]
      new_output_tensor = ops.get_collection("output_tensor")[0]
      # Verifies that the new graph computes the same result as the original.
      new_output_value = sess.run(new_output_tensor,
                                  {new_input_tensor: input_feed_value})
      self.assertEqual(new_output_value, output_value)

  @test_util.run_deprecated_v1
  def testStrippedOpListNestedFunctions(self):
    # [FUNCTION DOC]
    # Test: defines two nested functions (one calling the other, wrapping square). Before any call
    # the stripped op list must be empty; after calling the outer one on a constant, it must list
    # exactly the constant and square ops (library functions excluded).
    with self.cached_session():
      # Square two levels deep
      @function.Defun(dtypes.int32)
      def f0(x):
        # [FUNCTION DOC]
        # Inner test function: returns the element-wise square of its integer input.
        return math_ops.square(x)

      @function.Defun(dtypes.int32)
      def f1(x):
        # [FUNCTION DOC]
        # Outer test function: calls the inner square function on its input and returns the result.
        return f0(x)

      # At this point we've defined two functions but haven't called them, so
      # there should be no used ops.
      op_list = meta_graph.stripped_op_list_for_graph(ops.get_default_graph()
                                                      .as_graph_def())
      self.assertEqual(len(op_list.op), 0)

      # If we call the function on a constant, there should be two ops
      _ = f1(constant_op.constant(7))
      op_list = meta_graph.stripped_op_list_for_graph(ops.get_default_graph()
                                                      .as_graph_def())
      self.assertEqual(["Const", "Square"], [op.name for op in op_list.op])

  def testStrippedOpListRecursiveFunctions(self):
    # [FUNCTION DOC]
    # Test: manually builds a graph definition whose library has function A calling B, and B
    # calling a constant op and A (a cycle), with the graph using A. Asserts that computing the
    # stripped op list terminates and returns only the constant op.
    # The function module doesn't support recursive functions, so we build a
    # recursive function situation by ourselves: A calls B calls A and Const.
    graph = graph_pb2.GraphDef()
    a = graph.library.function.add()
    b = graph.library.function.add()
    a.signature.name = "A"
    b.signature.name = "B"
    a.node_def.add().op = "B"
    b.node_def.add().op = "Const"
    b.node_def.add().op = "A"

    # Use A in the graph
    graph.node.add().op = "A"

    # The stripped op list should contain just Const.
    op_list = meta_graph.stripped_op_list_for_graph(graph)
    self.assertEqual(["Const"], [op.name for op in op_list.op])

  def testStrippedOpListPartitionedCalls(self):
    # [FUNCTION DOC]
    # Test: manually builds a graph definition where the graph calls function A through a
    # partitioned-call node, A calls B through a stateful partitioned-call node, and B has a
    # constant and a node using A. Asserts that the stripped op list contains exactly the constant,
    # partitioned-call and stateful partitioned-call ops (order ignored).
    # Function A calls B via StatefulPartitionedCall.
    graph = graph_pb2.GraphDef()
    a = graph.library.function.add()
    b = graph.library.function.add()
    a.signature.name = "A"
    b.signature.name = "B"
    node_in_a = a.node_def.add()
    node_in_a.op = "StatefulPartitionedCall"
    node_in_a.attr["f"].func.name = "B"
    b.node_def.add().op = "Const"
    b.node_def.add().op = "A"

    # Use A in the graph via PartitionedCall.
    node = graph.node.add()
    node.op = "PartitionedCall"
    node.attr["f"].func.name = "A"

    op_list = meta_graph.stripped_op_list_for_graph(graph)
    self.assertSameElements(
        ["Const", "PartitionedCall", "StatefulPartitionedCall"],
        [op.name for op in op_list.op])

  @test_util.run_deprecated_v1
  def testDefaultAttrStripping(self):
    # [FUNCTION DOC]
    # Test: with a complex-number op on float32 inputs (type attributes at their defaults), exports
    # with default stripping on and asserts those attributes are removed and the stripped flag is
    # set; exports with it off and asserts they remain and the flag is false. Then uses float64
    # inputs (non-default attributes) and asserts the attributes are kept with float64 and
    # complex128 values while the flag is set.
    """Verifies that default attributes are stripped from a graph def."""

    # Complex Op has 2 attributes with defaults:
    #   o "T"    : float32.
    #   o "Tout" : complex64.

    # When inputs to the Complex Op are float32 instances, "T" maps to float32
    # and "Tout" maps to complex64. Since these attr values map to their
    # defaults, they must be stripped unless stripping of default attrs is
    # disabled.
    with self.cached_session():
      real_num = constant_op.constant(1.0, dtype=dtypes.float32, name="real")
      imag_num = constant_op.constant(2.0, dtype=dtypes.float32, name="imag")
      math_ops.complex(real_num, imag_num, name="complex")

      # strip_default_attrs is enabled.
      meta_graph_def, _ = meta_graph.export_scoped_meta_graph(
          graph_def=ops.get_default_graph().as_graph_def(),
          strip_default_attrs=True)
      node_def = test_util.get_node_def_from_graph("complex",
                                                   meta_graph_def.graph_def)
      self.assertNotIn("T", node_def.attr)
      self.assertNotIn("Tout", node_def.attr)
      self.assertTrue(meta_graph_def.meta_info_def.stripped_default_attrs)

      # strip_default_attrs is disabled.
      meta_graph_def, _ = meta_graph.export_scoped_meta_graph(
          graph_def=ops.get_default_graph().as_graph_def(),
          strip_default_attrs=False)
      node_def = test_util.get_node_def_from_graph("complex",
                                                   meta_graph_def.graph_def)
      self.assertIn("T", node_def.attr)
      self.assertIn("Tout", node_def.attr)
      self.assertFalse(meta_graph_def.meta_info_def.stripped_default_attrs)

    # When inputs to the Complex Op are float64 instances, "T" maps to float64
    # and "Tout" maps to complex128. Since these attr values don't map to their
    # defaults, they must not be stripped.
    with self.session(graph=ops.Graph()):
      real_num = constant_op.constant(1.0, dtype=dtypes.float64, name="real")
      imag_num = constant_op.constant(2.0, dtype=dtypes.float64, name="imag")
      math_ops.complex(real_num, imag_num, name="complex")
      meta_graph_def, _ = meta_graph.export_scoped_meta_graph(
          graph_def=ops.get_default_graph().as_graph_def(),
          strip_default_attrs=True)
      node_def = test_util.get_node_def_from_graph("complex",
                                                   meta_graph_def.graph_def)
      self.assertEqual(node_def.attr["T"].type, dtypes.float64)
      self.assertEqual(node_def.attr["Tout"].type, dtypes.complex128)
      self.assertTrue(meta_graph_def.meta_info_def.stripped_default_attrs)

  @test_util.run_deprecated_v1
  def testDefaultAttrStrippingNestedFunctions(self):
    # [FUNCTION DOC]
    # Test: calls a function that calls another function containing a complex-number op, exports
    # with default stripping, finds the inner op node in the function library and asserts its
    # default-valued type attributes were removed and the stripped flag is set.
    """Verifies that default attributes are stripped from function node defs."""
    with self.cached_session():

      @function.Defun(dtypes.float32, dtypes.float32)
      def f0(i, j):
        # [FUNCTION DOC]
        # Inner test function: builds a complex number from two float inputs, naming the op for
        # lookup later.
        return math_ops.complex(i, j, name="double_nested_complex")

      @function.Defun(dtypes.float32, dtypes.float32)
      def f1(i, j):
        # [FUNCTION DOC]
        # Outer test function: calls the inner function with its two inputs and returns the result.
        return f0(i, j)

      _ = f1(constant_op.constant(1.0), constant_op.constant(2.0))
      meta_graph_def, _ = meta_graph.export_scoped_meta_graph(
          graph_def=ops.get_default_graph().as_graph_def(),
          strip_default_attrs=True)

      double_nested_complex_node_def = None
      for function_def in meta_graph_def.graph_def.library.function:
        for node_def in function_def.node_def:
          if node_def.name.startswith("double_nested_complex"):
            double_nested_complex_node_def = node_def
            break
        if double_nested_complex_node_def:
          break

      self.assertIsNotNone(double_nested_complex_node_def)
      self.assertNotIn("T", double_nested_complex_node_def.attr)
      self.assertNotIn("Tout", double_nested_complex_node_def.attr)
      self.assertTrue(meta_graph_def.meta_info_def.stripped_default_attrs)

  def testDefaultAttrStrippingUnregisteredOps(self):
    # [FUNCTION DOC]
    # Test: builds a graph definition with a single node of an unregistered op carrying an integer
    # attribute, creates a meta graph with default stripping on, and asserts that the attribute is
    # preserved (unregistered ops are not stripped) while the stripped flag is still set.
    """Verifies that nodes with un-registered ops are not stripped."""
    graph_def = graph_pb2.GraphDef()
    node = graph_def.node.add()
    node.name = "node_with_unreg_op"
    node.op = "unreg_op"
    node.attr["attr_1"].i = 1

    meta_info_def = meta_graph_pb2.MetaGraphDef.MetaInfoDef()
    meta_info_def.stripped_op_list.op.add()

    with self.cached_session():
      meta_graph_def = meta_graph.create_meta_graph_def(
          meta_info_def=meta_info_def, graph_def=graph_def,
          strip_default_attrs=True)
      node_def = test_util.get_node_def_from_graph("node_with_unreg_op",
                                                   meta_graph_def.graph_def)
      self.assertEqual(node_def.attr["attr_1"].i, 1)
      self.assertTrue(meta_graph_def.meta_info_def.stripped_default_attrs)

  @test_util.run_deprecated_v1
  def testVariableObjectsAreSharedAmongCollections(self):
    # [FUNCTION DOC]
    # Test: creates a variable so it is in both global and trainable collections (same object),
    # exports and imports into a new graph, and asserts each collection has one entry and that the
    # same variable object is shared by both collections after import.
    with ops.Graph().as_default() as graph1:
      v = variables.Variable(3.0)
      # A single instance of Variable is shared among the collections:
      global_vars = graph1.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      trainable_vars = graph1.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(global_vars), 1)
      self.assertEqual(len(trainable_vars), 1)
      self.assertIs(global_vars[0], trainable_vars[0])
      self.assertIs(v, global_vars[0])

    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(graph=graph1)
    del graph1  # To avoid accidental references in code involving graph2.

    with ops.Graph().as_default() as graph2:
      meta_graph.import_scoped_meta_graph(orig_meta_graph)
      global_vars = graph2.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      trainable_vars = graph2.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(global_vars), 1)
      self.assertEqual(len(trainable_vars), 1)
      # A single instance of Variable is shared among the collections:
      self.assertIs(global_vars[0], trainable_vars[0])

  @test_util.run_deprecated_v1
  def testMetricVariablesCollectionLoadsBytesList(self):
    # [FUNCTION DOC]
    # Test: exports a graph with one variable, copies its serialized global variables collection
    # into the metric variables collection of the message, imports it and asserts the metric
    # variables collection holds one variable object with the same name, dtype and shape as the
    # original.
    with ops.Graph().as_default() as graph1:
      v1 = variables.Variable(
          [1, 2, 3], shape=[3], dtype=dtypes.float64, name="v")

    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(graph=graph1)

    # Copy bytes list from global variables collection to metric variables.
    orig_meta_graph.collection_def[ops.GraphKeys.METRIC_VARIABLES].CopyFrom(
        orig_meta_graph.collection_def["variables"])

    with ops.Graph().as_default() as graph2:
      meta_graph.import_scoped_meta_graph(orig_meta_graph)
      var_list = graph2.get_collection(ops.GraphKeys.METRIC_VARIABLES)
      self.assertEqual(len(var_list), 1)
      v2 = var_list[0]
      self.assertIsInstance(v2, variables.Variable)
      self.assertEqual(v1.name, v2.name)
      self.assertEqual(v1.dtype, v2.dtype)
      self.assertEqual(v1.shape, v2.shape)


class ScopedMetaGraphTest(test.TestCase):
  # [CLASS DOC]
  # Tests for exporting and importing name-scoped subgraphs (layers, queues, nested scopes, while
  # loops, device clearing, selective collection restore) with helper methods shared between tests.

  def _testScopedExport(self, test_dir, exported_filenames):
    # [FUNCTION DOC]
    # HELPER: shared by the scoped export and import test. Builds a small three-layer network
    # (hidden1 with a conditional bias and colocation, hidden2 with a while loop, softmax_linear
    # with logits collection), then exports each layer scope to the given filenames in the given
    # directory (the first with the default unbound inputs collection, the rest without). Asserts
    # the exported variable names for the first scope. Returns the list of three exported messages.
    graph = ops.Graph()
    with graph.as_default():
      # Creates an inference graph.
      # Hidden 1
      colocate_constraint = constant_op.constant(1.2, name="constraint")
      images = constant_op.constant(
          1.2, dtypes.float32, shape=[100, 28], name="images")
      with ops.name_scope("hidden1"):
        with graph.colocate_with(colocate_constraint.op):
          weights1 = variables.Variable(
              random_ops.truncated_normal(
                  [28, 128], stddev=1.0 / math.sqrt(float(28))),
              name="weights")
        # The use of cond.cond here is purely for adding test
        # coverage the save and restore of control flow context (which doesn't
        # make any sense here from a machine learning perspective).  The typical
        # biases is a simple Variable without the conditions.
        biases1 = variables.Variable(
            cond.cond(
                math_ops.less(random.random(), 0.5),
                lambda: array_ops.ones([128]), lambda: array_ops.zeros([128])),
            name="biases")
        hidden1 = nn_ops.relu(math_ops.matmul(images, weights1) + biases1)

      # Hidden 2
      with ops.name_scope("hidden2"):
        weights2 = variables.Variable(
            random_ops.truncated_normal(
                [128, 32], stddev=1.0 / math.sqrt(float(128))),
            name="weights")

        # The use of while_loop.while_loop here is purely for adding test
        # coverage the save and restore of control flow context (which doesn't
        # make any sense here from a machine learning perspective).  The typical
        # biases is a simple Variable without the conditions.
        def loop_cond(it, _):
          # [FUNCTION DOC]
          # While loop condition for the test network: true while the counter is less than two.
          return it < 2

        def loop_body(it, biases2):
          # [FUNCTION DOC]
          # While loop body for the test network: adds a constant vector of 0.1 values (length 32)
          # to the running bias value and increments the counter; returns both as a pair.
          biases2 += constant_op.constant(0.1, shape=[32])
          return it + 1, biases2

        _, biases2 = while_loop.while_loop(loop_cond, loop_body, [
            constant_op.constant(0),
            variables.Variable(array_ops.zeros([32]), name="biases")
        ])
        hidden2 = nn_ops.relu(math_ops.matmul(hidden1, weights2) + biases2)
      # Linear
      with ops.name_scope("softmax_linear"):
        weights3 = variables.Variable(
            random_ops.truncated_normal(
                [32, 10], stddev=1.0 / math.sqrt(float(32))),
            name="weights")
        biases3 = variables.Variable(array_ops.zeros([10]), name="biases")
        logits = math_ops.matmul(hidden2, weights3) + biases3
        ops.add_to_collection("logits", logits)

      # Exports each sub-graph.
      # Exports the first one with unbound_inputs_col_name set to default.
      orig_meta_graph1, var_list = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filenames[0]),
          graph=ops.get_default_graph(),
          export_scope="hidden1")
      self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
      var_names = [v.name for _, v in var_list.items()]
      self.assertEqual(["hidden1/biases:0", "hidden1/weights:0"],
                       sorted(var_names))

      # Exports the rest with no unbound_inputs_col_name.
      orig_meta_graph2, _ = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filenames[1]),
          graph=ops.get_default_graph(),
          export_scope="hidden2",
          unbound_inputs_col_name=None)
      orig_meta_graph3, _ = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filenames[2]),
          graph=ops.get_default_graph(),
          export_scope="softmax_linear",
          unbound_inputs_col_name=None)

    return [orig_meta_graph1, orig_meta_graph2, orig_meta_graph3]

  def _testScopedImport(self, test_dir, exported_filenames):
    # [FUNCTION DOC]
    # HELPER: shared by the scoped export and import test. Creates a graph with an images input,
    # asserts importing the first exported file fails with an unbound inputs error both without an
    # input map and with an input map using a wrong name, then imports the three exported scopes
    # into new scopes wiring outputs of one layer to the next through the unbound input map, and
    # asserts the variable names. Re-exports each new scope and returns the three messages.
    graph = ops.Graph()
    # Create all the missing inputs.
    with graph.as_default():
      new_image = constant_op.constant(
          1.2, dtypes.float32, shape=[100, 28], name="images")

    with self.assertRaisesRegex(ValueError, "Graph contains unbound inputs"):
      meta_graph.import_scoped_meta_graph(
          os.path.join(test_dir, exported_filenames[0]),
          graph=graph,
          import_scope="new_hidden1")

    with self.assertRaisesRegex(ValueError, "Graph contains unbound inputs"):
      meta_graph.import_scoped_meta_graph(
          os.path.join(test_dir, exported_filenames[0]),
          graph=graph,
          input_map={"image:0": new_image},
          import_scope="new_hidden1")

    # Verifies we can import the original "hidden1" into "new_hidden1".
    var_list = meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filenames[0]),
        graph=graph,
        input_map={"$unbound_inputs_images": new_image},
        import_scope="new_hidden1")

    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
    new_var_names = [v.name for _, v in var_list.items()]
    self.assertEqual(["new_hidden1/biases:0", "new_hidden1/weights:0"],
                     sorted(new_var_names))

    # Verifies we can import the original "hidden2" into "new_hidden2".
    hidden1 = array_ops.identity(
        graph.as_graph_element("new_hidden1/Relu:0"), name="hidden1/Relu")
    var_list = meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filenames[1]),
        graph=graph,
        input_map={"$unbound_inputs_hidden1/Relu": hidden1},
        import_scope="new_hidden2",
        unbound_inputs_col_name=None)

    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
    new_var_names = [v.name for _, v in var_list.items()]
    self.assertEqual(["new_hidden2/biases:0", "new_hidden2/weights:0"],
                     sorted(new_var_names))

    # Verifies we can import the original "softmax_linear" into
    # "new_softmax_linear".
    hidden2 = array_ops.identity(
        graph.as_graph_element("new_hidden2/Relu:0"), name="hidden2/Relu")
    var_list = meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filenames[2]),
        graph=graph,
        input_map={"$unbound_inputs_hidden2/Relu": hidden2},
        import_scope="new_softmax_linear",
        unbound_inputs_col_name=None)
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
    new_var_names = [v.name for _, v in var_list.items()]
    self.assertEqual(
        ["new_softmax_linear/biases:0", "new_softmax_linear/weights:0"],
        sorted(new_var_names))

    # Exports the scoped meta graphs again.
    new_meta_graph1, var_list = meta_graph.export_scoped_meta_graph(
        graph=graph, export_scope="new_hidden1")
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))

    new_meta_graph2, var_list = meta_graph.export_scoped_meta_graph(
        graph=graph, export_scope="new_hidden2", unbound_inputs_col_name=None)
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))

    new_meta_graph3, var_list = meta_graph.export_scoped_meta_graph(
        graph=graph,
        export_scope="new_softmax_linear",
        unbound_inputs_col_name=None)
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))

    return [new_meta_graph1, new_meta_graph2, new_meta_graph3]

  # Verifies that we can export the subgraph under each layer and import
  # them into new layers in a new graph.
  @test_util.run_deprecated_v1
  def testScopedExportAndImport(self):
    # [FUNCTION DOC]
    # Test: exports three layer scopes to files, imports them under new scopes, then asserts each
    # original exported message equals the re-exported one after deleting the unbound inputs
    # collection (whose strings differ slightly between the two paths).
    test_dir = _TestDir("scoped_export_import")
    filenames = [
        "exported_hidden1.pbtxt", "exported_hidden2.pbtxt",
        "exported_softmax_linear.pbtxt"
    ]
    orig_meta_graphs = self._testScopedExport(test_dir, filenames)
    new_meta_graphs = self._testScopedImport(test_dir, filenames)
    for a, b in zip(orig_meta_graphs, new_meta_graphs):
      # The unbound input strings are slightly different with the C API enabled
      # ("images" vs "images:0") due to the original import_graph_def code
      # vs. ImportGraphDef in C++.
      # TODO(skyewm): update the pbtxts once _USE_C_API is removed.
      del a.collection_def["unbound_inputs"]
      del b.collection_def["unbound_inputs"]
      test_util.assert_meta_graph_protos_equal(self, a, b)

  def testWhileLoopGradients(self):
    # [FUNCTION DOC]
    # Test: builds a while loop on a variable under an export scope, exports it, computes the
    # gradient value, imports the message under another scope in a new graph, re-exports and
    # asserts the messages are equal, then builds and runs gradients on the imported tensors and
    # asserts the gradient value matches the original.
    # Create a simple while loop.
    with ops.Graph().as_default():
      with ops.name_scope("export"):
        var = variables.Variable(0.)
        var_name = var.name
        _, output = while_loop.while_loop(
            lambda i, x: i < 5,
            lambda i, x: (i + 1, x + math_ops.cast(i, dtypes.float32)),
            [0, var])
        output_name = output.name

      # Generate a MetaGraphDef containing the while loop with an export scope.
      meta_graph_def, _ = meta_graph.export_scoped_meta_graph(
          export_scope="export")

      # Build and run the gradients of the while loop. We use this below to
      # verify that the gradients are correct with the imported MetaGraphDef.
      init_op = variables.global_variables_initializer()
      grad = gradients_impl.gradients([output], [var])
      with session.Session() as sess:
        self.evaluate(init_op)
        expected_grad_value = self.evaluate(grad)

    # Restore the MetaGraphDef into a new Graph with an import scope.
    with ops.Graph().as_default():
      meta_graph.import_scoped_meta_graph(meta_graph_def, import_scope="import")

      # Re-export and make sure we get the same MetaGraphDef.
      new_meta_graph_def, _ = meta_graph.export_scoped_meta_graph(
          export_scope="import")
      test_util.assert_meta_graph_protos_equal(
          self, meta_graph_def, new_meta_graph_def)

      # Make sure we can still build gradients and get the same result.

      def new_name(tensor_name):
        # [FUNCTION DOC]
        # HELPER: maps a tensor name from the export scope to its name in the import scope by
        # replacing the leading export prefix with the import prefix. Returns the new name.
        base_tensor_name = tensor_name.replace("export/", "")
        return "import/" + base_tensor_name

      var = ops.get_default_graph().get_tensor_by_name(new_name(var_name))
      output = ops.get_default_graph().get_tensor_by_name(new_name(output_name))
      grad = gradients_impl.gradients([output], [var])

      init_op = variables.global_variables_initializer()

      with session.Session() as sess:
        self.evaluate(init_op)
        actual_grad_value = self.evaluate(grad)
        self.assertEqual(expected_grad_value, actual_grad_value)

  @test_util.run_v1_only("b/120545219")
  def testImportWhileLoopInWhileLoop(self):
    # [FUNCTION DOC]
    # Test: exports a graph containing a while loop, then imports it repeatedly inside the body of
    # another while loop in a new graph, initializes variables and runs the result. Asserts nothing
    # explicitly; passes if import and execution do not raise. Runs only in graph-mode (v1).
    # Create a simple while loop.
    with ops.Graph().as_default():
      var = variables.Variable(0.0)
      _, output = while_loop.while_loop(lambda i, x: i < 5,
                                              lambda i, x: (i + 1, x * 2.0),
                                              [0, var])
      output_name = output.name

      # Generate a MetaGraphDef containing the while loop with an export scope.
      meta_graph_def, _ = meta_graph.export_scoped_meta_graph()

    # Restore the MetaGraphDef in a while loop in a new graph.
    with ops.Graph().as_default():

      def body(i, _):
        # [FUNCTION DOC]
        # Loop body for the outer loop: imports the meta graph into the current graph, fetches the
        # imported output tensor by name and returns the incremented counter with that tensor.
        meta_graph.import_scoped_meta_graph(meta_graph_def)
        return i + 1, ops.get_default_graph().get_tensor_by_name(output_name)

      _, x = while_loop.while_loop(lambda i, x: i < 2, body, [0, 0.0], name="")
      with session.Session() as sess:
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(x)

  @test_util.run_deprecated_v1
  def testScopedImportUnderNameScope(self):
    # [FUNCTION DOC]
    # Test: exports a graph with one trainable variable, imports it with an import scope inside an
    # existing name scope, and asserts one variable is returned named with both scopes prefixed.
    graph = ops.Graph()
    with graph.as_default():
      variables.Variable(initial_value=1.0, trainable=True, name="myvar")
    meta_graph_def, _ = meta_graph.export_scoped_meta_graph(graph=graph)

    graph = ops.Graph()
    with graph.as_default():
      with ops.name_scope("foo"):
        imported_variables = meta_graph.import_scoped_meta_graph(
            meta_graph_def, import_scope="bar")
        self.assertEqual(len(imported_variables), 1)
        self.assertEqual(list(imported_variables.values())[0].name,
                         "foo/bar/myvar:0")

  @test_util.run_deprecated_v1
  def testScopedImportUnderNameScopeNoVarScope(self):
    # [FUNCTION DOC]
    # Test: same as the previous scoped import test, but with no import scope given; asserts the
    # imported variable gets only the surrounding name scope prefix.
    graph = ops.Graph()
    with graph.as_default():
      variables.Variable(initial_value=1.0, trainable=True, name="myvar")
    meta_graph_def, _ = meta_graph.export_scoped_meta_graph(graph=graph)

    graph = ops.Graph()
    with graph.as_default():
      with ops.name_scope("foo"):
        imported_variables = meta_graph.import_scoped_meta_graph(
            meta_graph_def)
        self.assertEqual(len(imported_variables), 1)
        self.assertEqual(list(imported_variables.values())[0].name,
                         "foo/myvar:0")

  def testImportsUsingSameScopeName(self):
    # [FUNCTION DOC]
    # Test: imports the same meta graph twice with the same import scope; asserts that each import
    # returns one variable keyed by its unscoped name, and that the second import's scope is made
    # unique with a numeric suffix.
    with ops.Graph().as_default():
      variables.Variable(0, name="v")
      meta_graph_def, _ = meta_graph.export_scoped_meta_graph()
    with ops.Graph().as_default():
      for suffix in ["", "_1"]:
        imported_variables = meta_graph.import_scoped_meta_graph(
            meta_graph_def, import_scope="s")
        self.assertEqual(len(imported_variables), 1)
        self.assertEqual(list(imported_variables.keys())[0], "v:0")
        self.assertEqual(list(imported_variables.values())[0].name,
                         "s" + suffix + "/v:0")

  @test_util.run_deprecated_v1
  def testScopedImportWithSelectedCollections(self):
    # [FUNCTION DOC]
    # Test: exports a graph with a variable (populating global and trainable collections) to a
    # file, then imports it four times with different restore predicates. Asserts the included
    # collections are populated and the omitted ones are empty in the new graph, covering both,
    # each, and neither.
    meta_graph_filename = os.path.join(
        _TestDir("selected_collections_import"), "meta_graph.pb")

    graph = ops.Graph()
    # Add a variable to populate two collections. The functionality tested is
    # not specific to variables, but using variables in the test is convenient.
    with graph.as_default():
      variables.Variable(initial_value=1.0, trainable=True)
    self.assertTrue(
        all(
            graph.get_collection(key)
            for key in
            [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.TRAINABLE_VARIABLES]
        ))
    meta_graph.export_scoped_meta_graph(
        filename=meta_graph_filename, graph=graph)

    def _test_import(include_collection_keys, omit_collection_keys):
      # [FUNCTION DOC]
      # HELPER: runs one import scenario. Inputs: collection keys expected to be restored and keys
      # expected to be omitted (must be disjoint, asserted). Imports the exported file into a new
      # graph with a predicate allowing only the included keys, then asserts included collections
      # are non-empty and omitted ones are empty under the import scope.
      assert set(include_collection_keys).isdisjoint(omit_collection_keys)
      newgraph = ops.Graph()
      import_scope = "some_scope_name"

      def _restore_collections_predicate(collection_key):
        # [FUNCTION DOC]
        # HELPER: predicate passed to import. Returns true when the key is in the included keys and
        # not in the omitted keys of the enclosing scenario.
        return (collection_key in include_collection_keys and
                collection_key not in omit_collection_keys)

      meta_graph.import_scoped_meta_graph(
          meta_graph_filename,
          graph=newgraph,
          import_scope=import_scope,
          restore_collections_predicate=_restore_collections_predicate)
      collection_values = [
          newgraph.get_collection(name=key, scope=import_scope)
          for key in include_collection_keys
      ]
      self.assertTrue(all(collection_values))
      collection_values = [
          newgraph.get_collection(name=key, scope=import_scope)
          for key in omit_collection_keys
      ]
      self.assertFalse(any(collection_values))

    _test_import(
        include_collection_keys=[
            ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.TRAINABLE_VARIABLES
        ],
        omit_collection_keys=[])
    _test_import(
        include_collection_keys=[ops.GraphKeys.GLOBAL_VARIABLES],
        omit_collection_keys=[ops.GraphKeys.TRAINABLE_VARIABLES])
    _test_import(
        include_collection_keys=[ops.GraphKeys.TRAINABLE_VARIABLES],
        omit_collection_keys=[ops.GraphKeys.GLOBAL_VARIABLES])
    _test_import(
        include_collection_keys=[],
        omit_collection_keys=[
            ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.TRAINABLE_VARIABLES
        ])

  def _testScopedExportWithQueue(self, test_dir, exported_filename):
    # [FUNCTION DOC]
    # HELPER: shared by the queue scope test. Builds a graph with a first-in-first-out queue,
    # enqueue, close and dequeue ops under a scope plus a registered queue runner, exports that
    # scope to the given file in the directory and returns the exported message.
    graph = ops.Graph()
    with graph.as_default():
      with ops.name_scope("queue1"):
        input_queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
        enqueue = input_queue.enqueue((9876), name="enqueue")
        close = input_queue.close(name="close")
        qr = queue_runner_impl.QueueRunner(input_queue, [enqueue], close)
        queue_runner_impl.add_queue_runner(qr)
        input_queue.dequeue(name="dequeue")

      orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filename),
          graph=ops.get_default_graph(),
          export_scope="queue1")

    return orig_meta_graph

  def _testScopedImportWithQueue(self, test_dir, exported_filename,
                                 new_exported_filename):
    # [FUNCTION DOC]
    # HELPER: shared by the queue scope test. Imports the previously exported file into a new
    # graph under a new scope, checks the dequeue tensor and close op exist there, re-exports the
    # new scope to a second file and returns that message.
    graph = ops.Graph()
    meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filename),
        graph=graph,
        import_scope="new_queue1")
    graph.as_graph_element("new_queue1/dequeue:0")
    graph.as_graph_element("new_queue1/close")
    with graph.as_default():
      new_meta_graph, _ = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, new_exported_filename),
          graph=graph,
          export_scope="new_queue1")

    return new_meta_graph

  # Verifies that we can export the subgraph containing a FIFOQueue under
  # "queue1" and import it into "new_queue1" in a new graph.
  @test_util.run_deprecated_v1
  def testScopedWithQueue(self):
    # [FUNCTION DOC]
    # Test: exports a queue subgraph, imports it into a new scope in a new graph, re-exports and
    # asserts the original and re-exported messages are equal.
    test_dir = _TestDir("scoped_with_queue")
    orig_meta_graph = self._testScopedExportWithQueue(test_dir,
                                                      "exported_queue1.pbtxt")
    new_meta_graph = self._testScopedImportWithQueue(
        test_dir, "exported_queue1.pbtxt", "exported_new_queue1.pbtxt")
    test_util.assert_meta_graph_protos_equal(self, orig_meta_graph,
                                             new_meta_graph)

  # Verifies that we can export a subgraph in a nested name scope containing a
  # "hidden1/hidden2" and import it into "new_hidden1/new_hidden2" in a new
  # graph.
  def doTestExportNestedNames(self, use_resource=False):
    # [FUNCTION DOC]
    # HELPER: shared by the nested names tests. KNOWN ISSUE: the end of this helper computes lists
    # of assign node names and expected colocation values but never uses or asserts on them, so
    # colocation preservation is not actually tested.
    # Input: a flag selecting resource variables for biases. Builds a graph with variables and a
    # relu under a three-level scope, exports the two outer levels, asserts variable names and node
    # names stripped of the scope, imports into a nested new scope in another graph and asserts the
    # imported variable names.
    graph1 = ops.Graph()
    with graph1.as_default():
      with ops.name_scope("hidden1/hidden2/hidden3"):
        images = constant_op.constant(
            1.0, dtypes.float32, shape=[3, 2], name="images")
        if use_resource:
          weights1 = variables.Variable(
              [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="weights")
          biases1 = resource_variable_ops.ResourceVariable(
              [0.1] * 3, name="biases")
        else:
          biases1 = variables.Variable([0.1] * 3, name="biases")
          weights1 = variables.Variable(
              [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="weights")
        nn_ops.relu(math_ops.matmul(images, weights1) + biases1, name="relu")

    orig_meta_graph, var_list = meta_graph.export_scoped_meta_graph(
        export_scope="hidden1/hidden2", graph=graph1)
    var_names = [v.name for _, v in var_list.items()]
    self.assertEqual(["hidden3/biases:0", "hidden3/weights:0"],
                     sorted(var_list.keys()))
    self.assertEqual([
        "hidden1/hidden2/hidden3/biases:0", "hidden1/hidden2/hidden3/weights:0"
    ], sorted(var_names))
    for node in orig_meta_graph.graph_def.node:
      self.assertTrue(node.name.startswith("hidden3"))

    graph2 = ops.Graph()
    new_var_list = meta_graph.import_scoped_meta_graph(
        orig_meta_graph, import_scope="new_hidden1/new_hidden2", graph=graph2)
    self.assertEqual(["hidden3/biases:0", "hidden3/weights:0"],
                     sorted(new_var_list.keys()))
    new_var_names = [v.name for _, v in new_var_list.items()]
    self.assertEqual([
        "new_hidden1/new_hidden2/hidden3/biases:0",
        "new_hidden1/new_hidden2/hidden3/weights:0"
    ], sorted(new_var_names))

    nodes = [
        "new_hidden1/new_hidden2/hidden3/biases/Assign",
        "new_hidden1/new_hidden2/hidden3/weights/Assign"
    ]
    expected = [
        b"loc:@new_hidden1/new_hidden2/hidden3/biases",
        b"loc:@new_hidden1/new_hidden2/hidden3/weights"
    ]

  @test_util.run_deprecated_v1
  def testExportNestedNames(self):
    # [FUNCTION DOC]
    # Test: runs the nested names scenario using standard (non-resource) variables.
    self.doTestExportNestedNames(use_resource=False)

  @test_util.run_deprecated_v1
  def testExportNestedNamesResource(self):
    # [FUNCTION DOC]
    # Test: runs the nested names scenario using resource variables.
    self.doTestExportNestedNames(use_resource=True)

  @test_util.run_deprecated_v1
  def testPotentialCycle(self):
    # [FUNCTION DOC]
    # Test: builds a graph where a node inside a scope depends on an outside node that is reused by
    # name, exports the scope and asserts that importing without an input map raises an unbound
    # inputs error, while importing with an input map for the unbound matmul input succeeds.
    graph1 = ops.Graph()
    with graph1.as_default():
      a = constant_op.constant(1.0, shape=[2, 2])
      b = constant_op.constant(2.0, shape=[2, 2])
      matmul = math_ops.matmul(a, b)
      with ops.name_scope("hidden1"):
        c = nn_ops.relu(matmul)
        d = constant_op.constant(3.0, shape=[2, 2])
        matmul = math_ops.matmul(c, d)

    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
        export_scope="hidden1", graph=graph1)

    graph2 = ops.Graph()
    with graph2.as_default():
      with self.assertRaisesRegex(ValueError, "Graph contains unbound inputs"):
        meta_graph.import_scoped_meta_graph(
            orig_meta_graph, import_scope="new_hidden1")

      meta_graph.import_scoped_meta_graph(
          orig_meta_graph,
          import_scope="new_hidden1",
          input_map={
              "$unbound_inputs_MatMul": constant_op.constant(
                  4.0, shape=[2, 2])
          })

  @test_util.run_deprecated_v1
  def testClearDevices(self):
    # [FUNCTION DOC]
    # Test: builds a graph with ops placed on different devices and asserts their devices. Then
    # checks three ways of clearing devices: clearing on export from a graph, clearing on export
    # from a graph definition, and clearing on import; in each case the imported ops have an empty
    # device.
    graph1 = ops.Graph()
    with graph1.as_default():
      with ops.device("/device:CPU:0"):
        a = variables.Variable(
            constant_op.constant(
                1.0, shape=[2, 2]), name="a")
      with ops.device("/job:ps/replica:0/task:0/device:GPU:0"):
        b = variables.Variable(
            constant_op.constant(
                2.0, shape=[2, 2]), name="b")
      with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
        math_ops.matmul(a, b, name="matmul")

    self.assertEqual("/device:CPU:0", str(graph1.as_graph_element("a").device))
    self.assertEqual("/job:ps/replica:0/task:0/device:GPU:0",
                     str(graph1.as_graph_element("b").device))
    self.assertEqual("/job:localhost/replica:0/task:0/device:CPU:0",
                     str(graph1.as_graph_element("matmul").device))

    # Verifies that devices are cleared on export.
    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
        graph=graph1, clear_devices=True)

    graph2 = ops.Graph()
    with graph2.as_default():
      meta_graph.import_scoped_meta_graph(orig_meta_graph, clear_devices=False)

    self.assertEqual("", str(graph2.as_graph_element("a").device))
    self.assertEqual("", str(graph2.as_graph_element("b").device))
    self.assertEqual("", str(graph2.as_graph_element("matmul").device))

    # Verifies that devices are cleared on export when passing in graph_def.
    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
        graph_def=graph1.as_graph_def(), clear_devices=True)

    graph2 = ops.Graph()
    with graph2.as_default():
      meta_graph.import_scoped_meta_graph(orig_meta_graph, clear_devices=False)

    self.assertEqual("", str(graph2.as_graph_element("a").device))
    self.assertEqual("", str(graph2.as_graph_element("b").device))
    self.assertEqual("", str(graph2.as_graph_element("matmul").device))

    # Verifies that devices are cleared on import.
    orig_meta_graph, _ = meta_graph.export_scoped_meta_graph(
        graph=graph1, clear_devices=False)

    graph2 = ops.Graph()
    with graph2.as_default():
      meta_graph.import_scoped_meta_graph(orig_meta_graph, clear_devices=True)

    self.assertEqual("", str(graph2.as_graph_element("a").device))
    self.assertEqual("", str(graph2.as_graph_element("b").device))
    self.assertEqual("", str(graph2.as_graph_element("matmul").device))


class MetaGraphWithVariableScopeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for meta graph handling of collections of variables created by metric functions.

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Test: fills a queue, computes a streaming mean metric (creating local variables), exports to
    # a file, imports it into a new graph and initializes local variables. Then imports a stored
    # older test data meta graph in which the local variables collection is a node list, asserts
    # two entries exist and that creating the initializer raises an attribute error because those
    # entries lack an initializer.

    def _enqueue_vector(sess, queue, values, shape=None):
      # [FUNCTION DOC]
      # HELPER: enqueues one vector. Inputs: a session, a queue, a list of values and an optional
      # shape (default one row with as many columns as values). Converts the values to a constant
      # with the queue's first dtype and runs the enqueue op. Returns nothing.
      if not shape:
        shape = (1, len(values))
      dtype = queue.dtypes[0]
      sess.run(
          queue.enqueue(constant_op.constant(
              values, dtype=dtype, shape=shape)))

    meta_graph_filename = os.path.join(
        _TestDir("metrics_export"), "meta_graph.pb")

    graph = ops.Graph()
    with self.session(graph=graph) as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      _, update_op = metrics.mean(values)

      initializer = variables.local_variables_initializer()
      self.evaluate(initializer)
      self.evaluate(update_op)

    meta_graph.export_scoped_meta_graph(
        filename=meta_graph_filename, graph=graph)

    # Verifies that importing a meta_graph with LOCAL_VARIABLES collection
    # works correctly.
    graph = ops.Graph()
    with self.session(graph=graph) as sess:
      meta_graph.import_scoped_meta_graph(meta_graph_filename)
      initializer = variables.local_variables_initializer()
      self.evaluate(initializer)

    # Verifies that importing an old meta_graph where "local_variables"
    # collection is of node_list type works, but cannot build initializer
    # with the collection.
    graph = ops.Graph()
    with self.session(graph=graph) as sess:
      meta_graph.import_scoped_meta_graph(
          test.test_src_dir_path(
              "python/framework/testdata/metrics_export_meta_graph.pb"))
      self.assertEqual(len(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES)),
                       2)
      with self.assertRaisesRegex(
          AttributeError, "has no attribute 'initializer'"
      ):
        initializer = variables.local_variables_initializer()


class ExportImportAcrossScopesTest(test.TestCase):
  # [CLASS DOC]
  # Tests that exporting a scope and importing under another scope gives the same graph as
  # building it directly there, for partitioned variables.

  @test_util.run_deprecated_v1
  def testPartitionedVariables(self):
    # [FUNCTION DOC]
    # Test: runs the across-scopes check with a graph function creating a partitioned variable and
    # a variable in several collections, once with reference variables and once with resource
    # variables.

    def make_graph_with_partitioned_variables(use_resource):
      # [FUNCTION DOC]
      # HELPER: graph builder passed to the across-scopes check. Creates a variable partitioned in
      # three along the first axis and a scalar variable in several named collections, using
      # resource variables if the flag is set. Returns nothing.
      variable_scope.get_variable(
          name="weights",
          partitioner=partitioned_variables.fixed_size_partitioner(3, axis=0),
          initializer=random_ops.truncated_normal([100, 10]),
          use_resource=use_resource)
      # The next variable illustrates the necessity of restoring collections
      # in a deterministic fashion when using ResourceVariables.
      variable_scope.get_variable(
          name="another",
          shape=[],
          collections=["a", "b", "z", "f", "e", "d", "g"],
          use_resource=use_resource)

    self._testExportImportAcrossScopes(
        make_graph_with_partitioned_variables, use_resource=False)
    self._testExportImportAcrossScopes(
        make_graph_with_partitioned_variables, use_resource=True)

  def _testExportImportAcrossScopes(self, graph_fn, use_resource):
    # [FUNCTION DOC]
    # HELPER: shared by the partitioned variable test. Inputs: a graph building function and a
    # resource variable flag. Builds the graph under one scope, exports it with part of the scope
    # dropped, imports under a new scope, builds the expected graph directly under the final scope,
    # exports both and (for resource variables) clears shared name and debug name attributes that
    # are not updated by scoping. Asserts the two messages are equal.
    """Tests export and importing a graph across scopes.

    Args:
      graph_fn: A closure that creates a graph on the current scope.
      use_resource: A bool indicating whether or not to use ResourceVariables.
    """
    with ops.Graph().as_default() as original_graph:
      with variable_scope.variable_scope("dropA/dropB/keepA"):
        graph_fn(use_resource=use_resource)
    exported_meta_graph_def = meta_graph.export_scoped_meta_graph(
        graph=original_graph,
        export_scope="dropA/dropB")[0]

    with ops.Graph().as_default() as imported_graph:
      meta_graph.import_scoped_meta_graph(
          exported_meta_graph_def,
          import_scope="importA")

    with ops.Graph().as_default() as expected_graph:
      with variable_scope.variable_scope("importA/keepA"):
        graph_fn(use_resource=use_resource)

    result = meta_graph.export_scoped_meta_graph(graph=imported_graph)[0]
    expected = meta_graph.export_scoped_meta_graph(graph=expected_graph)[0]

    if use_resource:
      # Clear all shared_name and debug_name attributes before comparing, since
      # they are orthogonal to scopes and are not updated on export/import.
      for meta_graph_def in [result, expected]:
        for node in meta_graph_def.graph_def.node:
          for attr_to_remove in ["shared_name", "debug_name"]:
            attr_value = node.attr.get(attr_to_remove, None)
            if attr_value and attr_value.HasField("s"):
              if attr_value.s:
                node.attr[attr_to_remove].s = b""

    test_util.assert_meta_graph_protos_equal(self, expected, result)


if __name__ == "__main__":
  test.main()
