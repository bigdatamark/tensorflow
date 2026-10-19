# [MODULE DOC]
# Unit tests for the convert_to_constants module, which freezes variables into constants.
# Three test classes cover (1) converting concrete functions with the TF2 entry point, including
# saved models, control flow (if, while, case, recurrent networks) and reference variables,
# (2) the same scenarios through the TF1 session-based concrete function converter, and
# (3) the session-graph converter on raw graph definitions, checking that variables become
# constants, deny-listed variables stay, and resulting graphs contain expected nodes.
# Also defines a graph merging helper class and two small graph inspection helpers.
# Running the file executes the standard test main; no import-time side effects beyond imports.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for convert_to_constants.py."""

import os
import re

import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops import while_v2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import compat
from tensorflow.python.util import nest


class _GraphMerger(object):
  # [CLASS DOC]
  # HELPER: test utility for comparing graph definitions in the session-graph tests.
  # Namespace of static methods that merge protocol buffer messages (nodes, function definitions,
  # graph definitions) so a test can check that an expected subgraph is contained in a larger graph.
  # Holds no state and is never instantiated.
  """GraphDef merging methods for testing purposes."""

  @staticmethod
  def merge_any(x1, x2, empty_fn):
    # [FUNCTION DOC]
    # HELPER: building block of the graph merging utilities.
    # Static method. Creates an empty message with the factory empty_fn, copies x1 into it, then
    # merges x2 on top (x2 values override scalars, repeated fields are appended). Returns the new
    # message; inputs are not modified.
    """Merges two values using the message's CopyFrom/MergeFrom methods."""
    merged = empty_fn()
    merged.CopyFrom(x1)
    merged.MergeFrom(x2)
    return merged

  @staticmethod
  def merge_nodes(node1, node2):
    # [FUNCTION DOC]
    # HELPER: building block of the graph merging utilities.
    # Static method merging two node definitions. Uses the generic merge, then rewrites the input list
    # so it is the inputs of node1 followed by the inputs of node2 that are not already present
    # (avoids the duplicate inputs the plain merge would produce). Returns the merged node.
    """Merges two NodeDef messages."""
    merged = _GraphMerger.merge_any(node1, node2, node_def_pb2.NodeDef)
    merged_inputs = node1.input[:]
    merged_inputs.extend([i for i in node2.input[:] if i not in merged_inputs])
    merged.input[:] = merged_inputs
    return merged

  @staticmethod
  def merge_lists(repeated1, repeated2, empty_fn, key_fn, merge_fn):
    # [FUNCTION DOC]
    # HELPER: building block of the graph merging utilities.
    # Static method treating two repeated fields as maps keyed by key_fn. For every key present in
    # either list, merges the entries (using an empty message for a missing side) with merge_fn.
    # Returns a list of merged entries sorted by key, so the result is order independent.
    """Merges two lists representing maps."""
    merged = {}
    xs1 = {key_fn(x): x for x in repeated1}
    xs2 = {key_fn(x): x for x in repeated2}
    for name in set().union(xs1.keys(), xs2.keys()):
      x1 = empty_fn() if name not in xs1 else xs1[name]
      x2 = empty_fn() if name not in xs2 else xs2[name]
      merged[name] = merge_fn(x1, x2)
    return sorted(merged.values(), key=key_fn)

  @staticmethod
  def merge_node_lists(repeated_nodes1, repeated_nodes2):
    # [FUNCTION DOC]
    # HELPER: building block of the graph merging utilities.
    # Static method merging two repeated node fields by node name using the node merge function.
    # Returns the sorted list of merged nodes.
    """Merges two repeated node fields."""
    return _GraphMerger.merge_lists(repeated_nodes1, repeated_nodes2,
                                    node_def_pb2.NodeDef, lambda n: n.name,
                                    _GraphMerger.merge_nodes)

  @staticmethod
  def merge_functions(fn1, fn2):
    # [FUNCTION DOC]
    # HELPER: building block of the graph merging utilities.
    # Static method merging two function definitions. Generic-merges them, then replaces the
    # signature input arguments, output arguments and body nodes with name-keyed merges of the
    # two inputs (so duplicates are combined and ordering is deterministic). Returns the new function.
    """Merges two FunctionDefs."""
    merged = _GraphMerger.merge_any(fn1, fn2, function_pb2.FunctionDef)

    del merged.signature.input_arg[:]
    merged.signature.input_arg.extend(
        _GraphMerger.merge_lists(
            fn1.signature.input_arg[:], fn2.signature.input_arg[:],
            op_def_pb2.OpDef.ArgDef, lambda a: a.name,
            lambda x, y: _GraphMerger.merge_any(x, y, op_def_pb2.OpDef.ArgDef)))

    del merged.signature.output_arg[:]
    merged.signature.output_arg.extend(
        _GraphMerger.merge_lists(
            fn1.signature.output_arg[:], fn2.signature.output_arg[:],
            op_def_pb2.OpDef.ArgDef, lambda a: a.name,
            lambda x, y: _GraphMerger.merge_any(x, y, op_def_pb2.OpDef.ArgDef)))

    del merged.node_def[:]
    merged.node_def.extend(
        _GraphMerger.merge_node_lists(fn1.node_def[:], fn2.node_def[:]))

    return merged

  @staticmethod
  def merge_graphs(graph1, graph2):
    # [FUNCTION DOC]
    # HELPER: used by the graph containment assertion in the session-graph tests.
    # Static method building a new graph definition whose nodes are the name-keyed merge of the nodes
    # of both graphs and whose function library is the merge of both libraries keyed by function
    # name. Other graph fields are not carried over. Returns the merged graph; inputs are unchanged.
    """Merges two GraphDef messages."""
    merged = graph_pb2.GraphDef()
    merged.node.extend(
        _GraphMerger.merge_node_lists(graph1.node[:], graph2.node[:]))

    merged.library.function.extend(
        _GraphMerger.merge_lists(graph1.library.function,
                                 graph2.library.function,
                                 function_pb2.FunctionDef,
                                 lambda f: f.signature.name,
                                 _GraphMerger.merge_functions))

    return merged


def has_stateful_partitioned_call_op(graph_def):
  # [FUNCTION DOC]
  # HELPER: small shared assertion helper for the tests in this file.
  # Returns true if any node in the given graph definition is a stateful partitioned call
  # (a call to a function that is not inlined), otherwise false.
  """Determines if a StatefulPartitionedCall op exists in the graph."""
  for node in graph_def.node:
    if node.op == "StatefulPartitionedCall":
      return True
  return False


def get_num_variables(graph_def):
  # [FUNCTION DOC]
  # HELPER: small shared assertion helper for the tests in this file.
  # Returns how many nodes in the graph definition read a resource variable.
  """Returns the number of ReadVariableOp in the graph."""
  return sum(node.op == "ReadVariableOp" for node in graph_def.node)


class VariablesToConstantsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the TF2 conversion of concrete functions (variables to constants), covering basic
  # models, saved models, reference variables from TF1 saved models and control flow ops.
  # Shared fixtures: a freeze helper and a helper verifying the converted function has no
  # variables, matches the original outputs and shapes, and survives a save and load round trip.

  def _freezeModel(self, func):
    # [FUNCTION DOC]
    # HELPER: fixture used by the control flow tests of this class.
    # Wraps func in a trackable root object, gets its concrete function without arguments and
    # converts its variables to constants with control flow lowering disabled. Returns the root
    # and the frozen function. Input func is a traced function with a fixed input signature.
    """Freezes the function.

    Args:
      func: Function.

    Returns:
      root: AutoTrackable object with original ConcreteFunction.
      output_func: frozen ConcreteFunction.
    """
    root = autotrackable.AutoTrackable()
    root.f = func
    input_func = root.f.get_concrete_function()

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func, lower_control_flow=False)
    return root, output_func

  def _testConvertedFunction(self, obj, func, converted_concrete_func,
                             input_data):
    # [FUNCTION DOC]
    # HELPER: shared verification used by most tests in this class.
    # Checks the converted graph has no variable reads and no stateful partitioned calls; checks
    # that calling the converted function with the input data gives the same values as the original
    # (almost equal); checks each input shape is retained; saves the converted function as a
    # signature in a temporary directory, loads it back and checks the loaded signature gives the
    # same values. Parameters: obj is unused, func the original function, converted_concrete_func
    # the frozen one, input_data a mapping from input names to tensors. Writes files under the
    # test temp directory.
    # Ensure the converted graph has no variables and no function calls.
    constant_graph_def = converted_concrete_func.graph.as_graph_def()
    self.assertEqual(0, get_num_variables(constant_graph_def))
    self.assertFalse(has_stateful_partitioned_call_op(constant_graph_def))

    # Check that the converted ConcreteFunction produces the same result as the
    # original Function.
    expected_value = nest.flatten(func(**input_data))
    actual_value = nest.flatten(converted_concrete_func(**input_data))

    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual.numpy())

    # Ensure the shape is retained.
    for tensor in converted_concrete_func.inputs:
      actual_shape = input_data[tensor.name.split(":")[0]].shape
      self.assertEqual(tensor.shape, actual_shape)

    # Save the converted ConcreteFunction as a signature.
    save_dir = os.path.join(self.get_temp_dir(), "frozen_saved_model")
    root = autotrackable.AutoTrackable()
    root.f = converted_concrete_func
    save(root, save_dir, {"mykey": converted_concrete_func})

    # Load it back and make sure it works.
    loaded_obj = load(save_dir)
    actual_value = nest.flatten(loaded_obj.signatures["mykey"](**input_data))
    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual.numpy())

  @test_util.run_v2_only
  def testConstSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: a model with only constants is saved and loaded as a saved model, then its
    # serving signature is converted. Expects no variables before conversion, a function library
    # present, and the converted function to match the original and round trip. TF2 only.
    """Test a basic model with constants while saving/loading the SavedModel."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data["x"])

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save(root, save_dir, to_save)
    saved_model = load(save_dir)
    input_func = saved_model.signatures["serving_default"]

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(0, get_num_variables(variable_graph_def))
    self.assertTrue(variable_graph_def.library.function)

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testVariableModel(self):
    # [FUNCTION DOC]
    # Scenario: a traced function multiplies the input by two variables. Expects two variable reads
    # before conversion and, after conversion, identical results and a successful save and load.
    # TF2 only.
    """Test a basic model with Variables."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data["x"])

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, get_num_variables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testScalarModel(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is a copy of the variable model test; the real scenario uses a
    # scalar (rank zero) input rather than a one element vector.
    # Same checks as the variable model test but with a scalar input, verifying the scalar shape is
    # retained after conversion. TF2 only.
    """Test a basic model with Variables."""
    input_data = {"x": constant_op.constant(1., shape=[])}
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data["x"])

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, get_num_variables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testVariableSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: a model with two variables is saved and loaded; the loaded serving signature calls
    # its function through a stateful partitioned call. Converts it and expects equal results,
    # no variables and no partitioned calls after conversion. TF2 only.
    """Test a basic model with Variables with saving/loading the SavedModel."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    to_save = root.f.get_concrete_function(input_data["x"])

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save(root, save_dir, to_save)
    saved_model = load(save_dir)
    input_func = saved_model.signatures["serving_default"]

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertTrue(has_stateful_partitioned_call_op(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testMultiFunctionModel(self):
    # [FUNCTION DOC]
    # Scenario: a model has two traced methods that each lazily create one variable; only the add
    # method is traced and converted. Expects exactly one variable read before conversion and a
    # correct converted function. TF2 only.
    """Test a basic model with multiple tf.functions."""

    class BasicModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # Local trackable model with two methods, add and sub, that each lazily create their own
      # variable on first trace (the attributes start unset). Used by the multi function test.

      def __init__(self):
        # [FUNCTION DOC]
        # Initializes both lazily created variable attributes to unset. Returns nothing.
        self.y = None
        self.z = None

      @def_function.function
      def add(self, x):
        # [FUNCTION DOC]
        # Traced method. On first trace creates the variable y with initial value two, then returns
        # the input plus that variable. Side effect: sets the y attribute.
        if self.y is None:
          self.y = variables.Variable(2.)
        return x + self.y

      @def_function.function
      def sub(self, x):
        # [FUNCTION DOC]
        # Traced method. On first trace creates the variable z with initial value three, then returns
        # the input minus that variable. Side effect: sets the z attribute.
        if self.z is None:
          self.z = variables.Variable(3.)
        return x - self.z

    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = BasicModel()
    input_func = root.add.get_concrete_function(input_data["x"])

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(1, get_num_variables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.add, output_func, input_data)

  def _singleMetaGraphSavedModel(self):
    # [FUNCTION DOC]
    # HELPER: fixture for the reference variable import test.
    # Builds a TF1 graph with a placeholder, two reference variables (one unused) and a local
    # resource variable, runs initializers in a session, and exports it with the simple save API,
    # using the local variable initializer as legacy init op. Returns the directory path of the
    # saved model, created under the test temp directory.
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=[1, 1], dtype=dtypes.float32, name="start")
      distractor = ref_variable.RefVariable(-1., name="distractor")
      v = ref_variable.RefVariable(3., name="v")
      local_variable = variable_v1.VariableV1(
          1.,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          trainable=False,
          use_resource=True)
      output = array_ops.identity(start * v * local_variable, name="output")
      with session_lib.Session() as session:
        session.run([v.initializer, distractor.initializer,
                     local_variable.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": start},
            outputs={"output": output},
            legacy_init_op=local_variable.initializer)
    return path

  @test_util.run_v2_only
  def testRefVariableImport(self):
    # [FUNCTION DOC]
    # Scenario: a TF1 saved model with reference variables is loaded and its serving signature is
    # converted. Expects the converted function to match the original and to round trip. TF2 only.
    """Test a model with 1.X ReferenceVariables."""
    input_data = {"start": constant_op.constant(1., shape=[1, 1])}

    saved = self._singleMetaGraphSavedModel()
    imported = load(saved)
    fn = imported.signatures["serving_default"]

    output_func = convert_to_constants.convert_variables_to_constants_v2(fn)
    root = autotrackable.AutoTrackable()
    self._testConvertedFunction(root, fn, output_func, input_data)

  @test_util.run_v2_only
  def testIf(self):
    # [FUNCTION DOC]
    # Scenario: a function with a conditional (if) op whose branches use a variable (matrix multiply
    # in one branch, add in the other). Freezes it and expects equal results. TF2 only.
    """Test a model with the If op."""
    input_data = {
        "x": constant_op.constant([1., 2.], shape=[1, 2]),
        "b": constant_op.constant(True)
    }

    weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=dtypes.float32)

    def true_fn(x):
      # [FUNCTION DOC]
      # True branch of the conditional: returns the matrix product of the input and the weights variable.
      return math_ops.matmul(x, weights)

    def false_fn(x):
      # [FUNCTION DOC]
      # False branch of the conditional: returns the input plus the weights variable.
      return math_ops.add(x, weights)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[1, 2], dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)
    ])
    def model(x, b):
      # [FUNCTION DOC]
      # Traced model with a float matrix input and a boolean predicate; returns the result of a
      # conditional choosing between the true and false branch functions.
      return cond.cond(
          b, true_fn=lambda: true_fn(x), false_fn=lambda: false_fn(x))

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testStatelessIf(self):
    # [FUNCTION DOC]
    # Scenario: a stateless conditional (version 2 cond) over a constant tensor, branches return
    # the constant or the constant plus two. Freezes it and expects equal results. TF2 only.
    """Test a model with the StatelessIf op."""
    input_data = {"b": constant_op.constant(True)}

    x = constant_op.constant([1., 2.], shape=[1, 2], name="x")

    def true_fn():
      # [FUNCTION DOC]
      # True branch: returns the captured constant tensor unchanged.
      return x

    def false_fn():
      # [FUNCTION DOC]
      # False branch: returns the captured constant tensor plus two.
      return x + 2

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)])
    def model(b):
      # [FUNCTION DOC]
      # Traced model with a boolean scalar input; returns the stateless conditional selecting between
      # the true and false branch functions.
      return cond_v2.cond_v2(b, true_fn, false_fn)

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testStaticRnn(self):
    # [FUNCTION DOC]
    # Scenario: a static recurrent network with an LSTM cell over a random input of three steps,
    # whose graph contains conditional ops. Freezes it and expects equal results. TF2 only.
    """Test a StaticRnn containing If ops."""
    input_data = {
        "x":
            constant_op.constant(
                np.array(np.random.random_sample((3, 10)), dtype=np.float32))
    }

    cell = rnn_cell_impl.LSTMCell(10)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[3, 10], dtype=dtypes.float32)
    ])
    def model(x):
      # [FUNCTION DOC]
      # Traced model for the enclosing test: takes the float input tensor declared in the input
      # signature and returns the output of the recurrent network or loop built in the test.
      seq = array_ops.split(x, 3, 0)
      return rnn.static_rnn(
          cell, seq, dtype=dtypes.float32, sequence_length=[1])

    root, output_func = self._freezeModel(model)

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testWhile(self):
    # [FUNCTION DOC]
    # Scenario: a while loop that keeps adding a weights variable to the input until the sum reaches
    # one hundred. Freezes it and expects equal results. TF2 only.
    """Test a While loop."""
    input_data = {"x": constant_op.constant([1., 2., 3., 4.], shape=[2, 2])}

    weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=dtypes.float32)

    def condition(x):
      # [FUNCTION DOC]
      # Loop condition: true while the sum of all elements is below one hundred.
      return math_ops.reduce_sum(x) < 100

    def body(x):
      # [FUNCTION DOC]
      # Loop body: returns the input plus the weights variable.
      return math_ops.add(x, weights)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[2, 2], dtype=dtypes.float32)
    ])
    def model(x):
      # [FUNCTION DOC]
      # Traced model for the enclosing test: takes the float input tensor declared in the input
      # signature and returns the output of the recurrent network or loop built in the test.
      return while_loop.while_loop(condition, body, [x])

    root, output_func = self._freezeModel(model)

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testStatelessWhile(self):
    # [FUNCTION DOC]
    # Scenario: a stateless while loop that squares a scalar until it reaches four or more.
    # Freezes the traced model and expects equal results. TF2 only.
    """Test a StatelessWhile loop."""
    input_data = {"x": constant_op.constant(2.)}

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32)
    ])
    def model(x):
      # [FUNCTION DOC]
      # Traced model for the enclosing test: takes the float input tensor declared in the input
      # signature and returns the output of the recurrent network or loop built in the test.
      return while_v2.while_loop(
          lambda v: v < 4.,
          lambda v: v * v, [x],
          return_same_structure=False,
          name="while_1")  # x**2

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testDynamicRnn(self):
    # [FUNCTION DOC]
    # Scenario: a dynamic recurrent network with an LSTM cell, whose graph contains while loops.
    # Freezes it and expects equal results. TF2 only.
    """Test a DynamicRnn containing While loops."""
    input_data = {
        "x":
            constant_op.constant(
                np.array(
                    np.random.random_sample((3, 10, 10)), dtype=np.float32))
    }

    cell = rnn_cell_impl.LSTMCell(10)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[3, 10, 10], dtype=dtypes.float32)
    ])
    def model(x):
      # [FUNCTION DOC]
      # Traced model for the enclosing test: takes the float input tensor declared in the input
      # signature and returns the output of the recurrent network or loop built in the test.
      return rnn.dynamic_rnn(cell, x, dtype=dtypes.float32)

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  @test_util.disable_tfrt("b/180451239")
  def testSwitchCase(self):
    # [FUNCTION DOC]
    # Scenario: a switch-case over a random integer index choosing among three branches that use
    # three different variables. Freezes it and expects equal results. TF2 only; disabled for the
    # TFRT runtime.
    """Test a switch_case statement."""
    input_data = {
        "i": constant_op.constant(np.random.randint(0, 3, dtype=np.int32)),
        "x": constant_op.constant(
            np.asarray(np.random.random_sample((10, 3)), dtype=np.float32)),
    }

    w0 = variables.Variable(np.random.random_sample((3, 4)), dtype=np.float32)
    w1 = variables.Variable(np.random.random_sample((3, 4)), dtype=np.float32)
    w2 = variables.Variable(np.random.random_sample((4,)), dtype=np.float32)

    def branch0(x):
      # [FUNCTION DOC]
      # Branch zero: matrix product of the input and the first weight variable.
      return math_ops.matmul(x, w0)

    def branch1(x):
      # [FUNCTION DOC]
      # Branch one: matrix product of the input and the second weight variable.
      return math_ops.matmul(x, w1)

    def branch2(x):
      # [FUNCTION DOC]
      # Branch two: pads the input with one extra column, then adds the third weight variable.
      x = array_ops.pad(x, [[0, 0], [0, 1]])
      return x + w2

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=[10, 3], dtype=dtypes.float32),
    ])
    def model(i, x):
      # [FUNCTION DOC]
      # Traced model with an integer branch index and a float matrix; returns the switch-case result
      # over the three branch functions.
      return control_flow_switch_case.switch_case(
          i, [lambda: branch0(x), lambda: branch1(x), lambda: branch2(x)])

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)


class ConvertVariablesToConstantsV2SessionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the TF1 session-mode conversion of concrete functions (the function-in-v1 converter).
  # Each test builds its model inside a fresh default graph and a session, and checks results by
  # running tensors in that session. Mirrors the scenarios of the TF2 test class.

  def _freezeModel(self, func):
    # [FUNCTION DOC]
    # HELPER: fixture used by the control flow tests of this class.
    # Wraps func in a trackable root, gets its concrete function and converts variables to constants
    # with the TF1 converter, control flow lowering disabled. Returns the root and frozen function.
    """Freezes the function.

    Args:
      func: Function.

    Returns:
      root: AutoTrackable object with original ConcreteFunction.
      output_func: frozen ConcreteFunction.
    """
    root = autotrackable.AutoTrackable()
    root.f = func
    input_func = root.f.get_concrete_function()

    output_func = convert_to_constants.convert_var_to_const_function_in_v1(
        input_func, lower_control_flow=False)
    return root, output_func

  def _testConvertedFunction(self, sess, obj, func, converted_concrete_func,
                             input_data):
    # [FUNCTION DOC]
    # HELPER: shared verification used by most tests in this class.
    # Same checks as the TF2 class helper (no variable reads or partitioned calls, equal values,
    # shapes retained, save and load round trip) but values are obtained by running tensors in the
    # given session. Parameters: sess session, obj unused, func original function, converted
    # function, input_data mapping from input names to tensors.
    # Ensure the converted graph has no variables and no function calls.
    constant_graph_def = converted_concrete_func.graph.as_graph_def()
    self.assertEqual(0, get_num_variables(constant_graph_def))
    self.assertFalse(has_stateful_partitioned_call_op(constant_graph_def))

    # Check that the converted ConcreteFunction produces the same result as the
    # original Function.
    expected_value = nest.flatten(func(**input_data))
    actual_value = nest.flatten(converted_concrete_func(**input_data))

    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(sess.run(expected), sess.run(actual))

    # Ensure the shape is retained.
    for tensor in converted_concrete_func.inputs:
      actual_shape = input_data[tensor.name.split(":")[0]].shape
      self.assertEqual(tensor.shape, actual_shape)

    # Save the converted ConcreteFunction as a signature.
    save_dir = os.path.join(self.get_temp_dir(), "frozen_saved_model")
    root = autotrackable.AutoTrackable()
    root.f = converted_concrete_func
    save(root, save_dir, {"mykey": converted_concrete_func})

    # Load it back and make sure it works.
    loaded_obj = load(save_dir)
    actual_value = nest.flatten(loaded_obj.signatures["mykey"](**input_data))
    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(sess.run(expected), sess.run(actual))

  def testRaiseErrorInEagerMode(self):
    # [FUNCTION DOC]
    # Scenario: calling the TF1 converter while eager execution is active. Expects a runtime error
    # saying the conversion must be carried out in a session.
    """Test the raised exception in Eager mode."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data["x"])

    with self.assertRaisesRegex(RuntimeError,
                                "must be carried out in a Session"):
      convert_to_constants.convert_var_to_const_function_in_v1(
          input_func)

  def testConvertVariables(self):
    # [FUNCTION DOC]
    # Scenario: two variables multiplied with the input inside a graph and session. Converts and
    # expects two variable reads before, equal results after, and a successful save and load.
    """Test a basic model with Variables."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = autotrackable.AutoTrackable()
        root.v1 = variables.Variable(3.)
        root.v2 = variables.Variable(2.)
        root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
        input_func = root.f.get_concrete_function(input_data["x"])

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertEqual(2, get_num_variables(variable_graph_def))

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)

        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testConvertVariablesWithAssignments(self):
    # [FUNCTION DOC]
    # Scenario: same as the basic variable test but three assignment ops are registered in the
    # variable assignment collection before conversion. Expects the conversion to succeed and the
    # result to still match the original function.
    """Test a basic model with Variables and assignment ops."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = autotrackable.AutoTrackable()
        root.v1 = variables.Variable(3.)
        root.v2 = variables.Variable(2.)
        root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
        input_func = root.f.get_concrete_function(input_data["x"])

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertEqual(2, get_num_variables(variable_graph_def))

        assign_op_1 = root.v1.assign(1.5)
        assign_op_2 = root.v2.assign(3.0)
        assign_op_3 = root.v1.assign(4.0)
        ops.get_default_graph().add_to_collection(
            convert_to_constants.VAR_ASSIGN_COLLECTION, assign_op_1)
        ops.get_default_graph().add_to_collection(
            convert_to_constants.VAR_ASSIGN_COLLECTION, assign_op_2)
        ops.get_default_graph().add_to_collection(
            convert_to_constants.VAR_ASSIGN_COLLECTION, assign_op_3)

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testConstSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: a constants-only model saved and loaded, then its serving signature converted in a
    # session. Expects no variables, a function library, and matching round trip results.
    """Test a basic model with constants while saving/loading the SavedModel."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = autotrackable.AutoTrackable()
        root.f = def_function.function(lambda x: 2. * x)
        to_save = root.f.get_concrete_function(input_data["x"])

        save_dir = os.path.join(self.get_temp_dir(), "saved_model")
        save(root, save_dir, to_save)
        saved_model = load(save_dir)
        input_func = saved_model.signatures["serving_default"]

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertEqual(0, get_num_variables(variable_graph_def))
        self.assertTrue(variable_graph_def.library.function)

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testVariableSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: a two-variable model initialized in the session, saved and loaded; the loaded graph
    # has a stateful partitioned call. Converts and expects matching results and a clean graph.
    """Test a basic model with Variables with saving/loading the SavedModel."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = autotrackable.AutoTrackable()
        root.v1 = variables.Variable(3.)
        root.v2 = variables.Variable(2.)
        root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
        to_save = root.f.get_concrete_function(input_data["x"])
        sess.run(variables.global_variables_initializer())

        save_dir = os.path.join(self.get_temp_dir(), "saved_model")
        save(root, save_dir, to_save)
        saved_model = load(save_dir)
        input_func = saved_model.signatures["serving_default"]

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertTrue(has_stateful_partitioned_call_op(variable_graph_def))

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testMultiFunctionModel(self):
    # [FUNCTION DOC]
    # Scenario: a model with two lazily-variable traced methods; the add method is converted in a
    # session. Expects one variable read before and correct results after.
    """Test a basic model with multiple tf.functions."""

    class BasicModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # Local trackable model with two methods, add and sub, that each lazily create their own
      # variable on first trace. Used by the multi function test.

      def __init__(self):
        # [FUNCTION DOC]
        # Initializes both lazily created variable attributes to unset. Returns nothing.
        self.y = None
        self.z = None

      @def_function.function
      def add(self, x):
        # [FUNCTION DOC]
        # Traced method. On first trace creates the variable y with initial value two, then returns
        # the input plus that variable. Side effect: sets the y attribute.
        if self.y is None:
          self.y = variables.Variable(2.)
        return x + self.y

      @def_function.function
      def sub(self, x):
        # [FUNCTION DOC]
        # Traced method. On first trace creates the variable z with initial value three, then returns
        # the input minus that variable. Side effect: sets the z attribute.
        if self.z is None:
          self.z = variables.Variable(3.)
        return x - self.z

    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = BasicModel()
        input_func = root.add.get_concrete_function(input_data["x"])

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertEqual(1, get_num_variables(variable_graph_def))

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)
        self._testConvertedFunction(sess, root, root.add, output_func,
                                    input_data)

  def testIf(self):
    # [FUNCTION DOC]
    # Scenario: conditional op with a variable in both branches, inside a graph and session;
    # frozen with the TF1 converter and compared with the original.
    """Test a model with the If op."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {
            "x": constant_op.constant([1., 2.], shape=[1, 2]),
            "b": constant_op.constant(True)
        }

        weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]],
                                     dtype=dtypes.float32)

        def true_fn(x):
          # [FUNCTION DOC]
          # True branch: returns the matrix product of the input and the weights variable.
          return math_ops.matmul(x, weights)

        def false_fn(x):
          # [FUNCTION DOC]
          # False branch: returns the input plus the weights variable.
          return math_ops.add(x, weights)

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[1, 2], dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)
        ])
        def model(x, b):
          # [FUNCTION DOC]
          # Traced model with a float matrix input and a boolean predicate; returns the conditional
          # selecting between the branch functions.
          return cond.cond(
              b, true_fn=lambda: true_fn(x), false_fn=lambda: false_fn(x))

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testStatelessIf(self):
    # [FUNCTION DOC]
    # Scenario: stateless conditional over a constant tensor, frozen in a session and compared.
    """Test a model with the StatelessIf op."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"b": constant_op.constant(True)}

        x = constant_op.constant([1., 2.], shape=[1, 2], name="x")

        def true_fn():
          # [FUNCTION DOC]
          # True branch: returns the captured constant tensor unchanged.
          return x

        def false_fn():
          # [FUNCTION DOC]
          # False branch: returns the captured constant tensor plus two.
          return x + 2

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)
        ])
        def model(b):
          # [FUNCTION DOC]
          # Traced model with a boolean scalar input; returns the stateless conditional result.
          return cond_v2.cond_v2(b, true_fn, false_fn)

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testStaticRnn(self):
    # [FUNCTION DOC]
    # Scenario: static recurrent network with an LSTM cell containing conditional ops, frozen in a
    # session and compared with the original.
    """Test a StaticRnn containing If ops."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {
            "x":
                constant_op.constant(
                    np.array(
                        np.random.random_sample((3, 10)), dtype=np.float32))
        }

        cell = rnn_cell_impl.LSTMCell(10)

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[3, 10], dtype=dtypes.float32)
        ])
        def model(x):
          # [FUNCTION DOC]
          # Traced model for the enclosing test: takes the float input tensor declared in the input
          # signature and returns the output of the recurrent network or loop built in the test.
          seq = array_ops.split(x, 3, 0)
          return rnn.static_rnn(
              cell, seq, dtype=dtypes.float32, sequence_length=[1])

        root, output_func = self._freezeModel(model)

        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testWhile(self):
    # [FUNCTION DOC]
    # Scenario: while loop adding a weights variable until the sum reaches one hundred, frozen in a
    # session and compared with the original.
    """Test a While loop."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant([1., 2., 3., 4.], shape=[2, 2])}

        weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]],
                                     dtype=dtypes.float32)

        def condition(x):
          # [FUNCTION DOC]
          # Loop condition: true while the sum of all elements is below one hundred.
          return math_ops.reduce_sum(x) < 100

        def body(x):
          # [FUNCTION DOC]
          # Loop body: returns the input plus the weights variable.
          return math_ops.add(x, weights)

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[2, 2], dtype=dtypes.float32)
        ])
        def model(x):
          # [FUNCTION DOC]
          # Traced model for the enclosing test: takes the float input tensor declared in the input
          # signature and returns the output of the recurrent network or loop built in the test.
          return while_loop.while_loop(condition, body, [x])

        root, output_func = self._freezeModel(model)

        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testStatelessWhile(self):
    # [FUNCTION DOC]
    # Scenario: stateless while loop squaring a scalar until it reaches four or more, frozen in a
    # session and compared.
    """Test a StatelessWhile loop."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(2.)}

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32)
        ])
        def model(x):
          # [FUNCTION DOC]
          # Traced model for the enclosing test: takes the float input tensor declared in the input
          # signature and returns the output of the recurrent network or loop built in the test.
          return while_v2.while_loop(
              lambda v: v < 4.,
              lambda v: v * v, [x],
              return_same_structure=False,
              name="while_1")  # x**2

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testDynamicRnn(self):
    # [FUNCTION DOC]
    # Scenario: dynamic recurrent network with an LSTM cell (while loops), frozen in a session and
    # compared with the original.
    """Test a DynamicRnn containing While loops."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {
            "x":
                constant_op.constant(
                    np.array(
                        np.random.random_sample((3, 10, 10)), dtype=np.float32))
        }

        cell = rnn_cell_impl.LSTMCell(10)

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[3, 10, 10], dtype=dtypes.float32)
        ])
        def model(x):
          # [FUNCTION DOC]
          # Traced model for the enclosing test: takes the float input tensor declared in the input
          # signature and returns the output of the recurrent network or loop built in the test.
          return rnn.dynamic_rnn(cell, x, dtype=dtypes.float32)

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  @test_util.disable_tfrt("b/180451239")
  def testSwitchCase(self):
    # [FUNCTION DOC]
    # Scenario: switch-case over three branches using three variables, frozen in a session and
    # compared with the original. Disabled for the TFRT runtime.
    """Test a switch_case statement."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {
            "i":
                constant_op.constant(np.random.randint(0, 3, dtype=np.int32)),
            "x":
                constant_op.constant(
                    np.asarray(
                        np.random.random_sample((10, 3)), dtype=np.float32)),
        }

        w0 = variables.Variable(
            np.random.random_sample((3, 4)), dtype=np.float32)
        w1 = variables.Variable(
            np.random.random_sample((3, 4)), dtype=np.float32)
        w2 = variables.Variable(np.random.random_sample((4,)), dtype=np.float32)

        def branch0(x):
          # [FUNCTION DOC]
          # Branch zero: matrix product of the input and the first weight variable.
          return math_ops.matmul(x, w0)

        def branch1(x):
          # [FUNCTION DOC]
          # Branch one: matrix product of the input and the second weight variable.
          return math_ops.matmul(x, w1)

        def branch2(x):
          # [FUNCTION DOC]
          # Branch two: pads the input with one extra column, then adds the third weight variable.
          x = array_ops.pad(x, [[0, 0], [0, 1]])
          return x + w2

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
            tensor_spec.TensorSpec(shape=[10, 3], dtype=dtypes.float32),
        ])
        def model(i, x):
          # [FUNCTION DOC]
          # Traced model with an integer branch index and a float matrix; returns the switch-case result.
          return control_flow_switch_case.switch_case(
              i, [lambda: branch0(x), lambda: branch1(x), lambda: branch2(x)])

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)


class ConvertVariablesToConstantsSessionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for converting variables to constants directly on graph definitions taken from a session
  # (the session-graph converter): reference and resource variables, functions with and without
  # inlining, switch ops, deny-listed variables, identity chains and v1/v2 case constructs.
  # Helpers assert that a graph contains an expected subgraph and that no variable ops remain.

  def _assertGraphContains(self, graph, subgraph):
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the conversion tests that compare graph text.
    # Asserts that the expected subgraph (graph definition or text format) is contained in graph:
    # both are normalized (generated case function names renumbered), the graph is merged with an
    # empty graph to canonicalize ordering, then merged with the subgraph and compared for equality.
    # Fails the test if merging adds anything.
    """Asserts that the given subgraph is contained within the given graph."""

    def normalize_uids(msg):
      # [FUNCTION DOC]
      # HELPER: renames generated case branch function names so they have deterministic sequential
      # suffixes per prefix. Input msg is a graph definition or its text; returns text. Pure.
      """Replace auto-id function names with something consistent."""
      # These functions have non-deterministic names, the non-determinism coming
      # from having an ops.uid() suffix in their names. We're replacing these
      # with new sequential IDs starting from 0 for each prefix, which is
      # is sufficient for tests.
      if isinstance(msg, graph_pb2.GraphDef):
        msg = text_format.MessageToString(msg)
      name_prefixes = ["case_cond_true.*", "case_cond_false.*"]
      name_regex = r"\b(" + "|".join(name_prefixes) + r")_([0-9]+)\b"
      names = {}
      for (name, index) in re.findall(name_regex, msg):
        names.setdefault(name, set()).add(int(index))
      for name, indices in names.items():
        for new_index, old_index in enumerate(sorted(list(indices))):
          msg = re.sub(r"\b" + name + "_" + str(old_index) + r"\b",
                       name + "_" + str(new_index), msg)
      return msg

    norm_graph = text_format.Parse(normalize_uids(graph), graph_pb2.GraphDef())
    norm_subgraph = text_format.Parse(
        normalize_uids(subgraph), graph_pb2.GraphDef())

    # Graph S is contained in C if and only if merge(C,S) == C.
    # We merge the input graph with an empty graph to normalize repeated fields:
    # assertProtoEquals is sensitive to ordering.
    norm_graph = _GraphMerger.merge_graphs(norm_graph, graph_pb2.GraphDef())
    merged_graph = _GraphMerger.merge_graphs(norm_graph, norm_subgraph)
    self.assertProtoEquals(norm_graph, merged_graph)

  def _ensure_no_variables_in_graph(self, graph_def):
    # [FUNCTION DOC]
    # HELPER: assertion shared by the tests. Fails if any node is a variable, variable handle or
    # variable read op. Returns nothing.
    """Ensures there are no variables in the graph."""
    for node in graph_def.node:
      self.assertNotIn(
          node.op, ["Variable", "VariableV2", "VarHandleOp", "ReadVariableOp"])

  def _test_variable_to_const_conversion(self, use_resource):
    # [FUNCTION DOC]
    # HELPER: shared body of the reference and resource variable tests.
    # Builds a graph with one used and one unused variable feeding a multiply, converts it with the
    # session converter, checks no variable ops remain, imports the result into a new graph and
    # checks it has four nodes and still evaluates to two. Parameter use_resource selects resource
    # versus reference variables.
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=use_resource):
        variable_node = variable_scope.get_variable(
            "variable_node", initializer=1.0)
        variable_scope.get_variable("unused_variable_node", initializer=1.0)
        output_node = math_ops.multiply(variable_node, 2.0, name="output_node")
        with session_lib.Session() as sess:
          self.evaluate(variable_node.initializer)
          output = self.evaluate(output_node)
          self.assertNear(2.0, output, 0.00001)
          variable_graph_def = sess.graph.as_graph_def()
          constant_graph_def = (
              convert_to_constants
              .convert_variables_to_constants_from_session_graph(
                  session=sess,
                  graph_def=variable_graph_def,
                  output_node_names=["output_node"]))

          self._ensure_no_variables_in_graph(constant_graph_def)

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with ops.Graph().as_default():
      _ = importer.import_graph_def(constant_graph_def, name="")
      self.assertEqual(4, len(constant_graph_def.node))
      self._ensure_no_variables_in_graph(constant_graph_def)
      with session_lib.Session() as sess:
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = self.evaluate(output_node)
        self.assertNear(2.0, output, 0.00001)

  def test_resource_variable_can_be_written_after_denylisting(self):
    # [FUNCTION DOC]
    # Scenario: a resource variable is assigned the sum of itself and a second variable before being
    # read. Converts with the first variable deny-listed and the initializer kept as an output.
    # Expects the deny-listed variable to remain a handle op, the other to become constant, and
    # repeated evaluation of the imported graph to give three then five (the assignment persists).
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        variable_node = variable_scope.get_variable(
            "variable_node", initializer=1.0)
        another_variable = variable_scope.get_variable(
            "unused_variable_node", initializer=2.0)
        with ops.control_dependencies(
            [variable_node.assign(another_variable + variable_node)]):
          output_node = array_ops.identity(variable_node, name="output_node")
        initializer_name = variable_node.initializer.name
        with session_lib.Session() as sess:
          self.evaluate(variable_node.initializer)
          self.evaluate(another_variable.initializer)
          output = self.evaluate(output_node)
          self.assertNear(3.0, output, 0.00001)
          variable_graph_def = sess.graph.as_graph_def()

          # Test variable name black list. This should result in the variable
          # not being a const.  Furthermore, the paths that read from and assign
          # to the denylisted variable should continue to be valid.
          constant_graph_def_with_denylist = (
              convert_to_constants
              .convert_variables_to_constants_from_session_graph(
                  session=sess,
                  graph_def=variable_graph_def,
                  output_node_names=["output_node", initializer_name],
                  variable_names_denylist=set(["variable_node"])))

          variable_node = None
          for node in constant_graph_def_with_denylist.node:
            if node.name == "variable_node":
              variable_node = node
          self.assertIsNotNone(variable_node)
          self.assertEqual(variable_node.op, "VarHandleOp")

    # Now we make sure another_variable is now a constant, but the original
    # variable is not, and that the graph can be executed and update the
    # variable can be updated with each execution.
    with ops.Graph().as_default():
      _ = importer.import_graph_def(constant_graph_def_with_denylist, name="")
      with session_lib.Session() as sess:
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        self.evaluate(sess.graph.get_operation_by_name(initializer_name))
        output = self.evaluate(output_node)
        self.assertNear(3.0, output, 0.00001)
        output = self.evaluate(output_node)
        self.assertNear(5.0, output, 0.00001)

  def _inline_functions(self, graph_def, arrays):
    # [FUNCTION DOC]
    # HELPER: for the function tests; runs the graph optimizer with only function inlining enabled.
    # Exports a meta graph from graph_def, registers the names in arrays as the fetch collection,
    # and returns the optimized graph definition.
    meta_graph = export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for name in arrays:
      fetch_collection.node_list.value.append(name)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

    # Initialize RewriterConfig with everything disabled except function
    # inlining.
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.optimizers.append("function")
    return tf_optimizer.OptimizeGraph(config, meta_graph)

  def _test_convert_variables_with_functions(self, inline_functions):
    # [FUNCTION DOC]
    # HELPER: shared body of the function tests. Builds a graph calling a defined function on a
    # variable, optionally inlines functions with the optimizer first (per inline_functions), then
    # converts with the session converter and checks no variable ops remain.
    """Freezes a graph with functions."""

    @function.Defun(dtypes.float32)
    def plus_one(x):
      # [FUNCTION DOC]
      # Defined graph function that returns its float input plus one.
      return x + 1.0

    with ops.Graph().as_default():
      variable_node = variables.Variable(1.0, name="variable_node")
      _ = variables.Variable(1.0, name="unused_variable_node")
      defun_node = plus_one(variable_node)
      _ = math_ops.multiply(defun_node, 2.0, name="output_node")

      with session_lib.Session() as sess:
        self.evaluate(variables.variables_initializer([variable_node]))
        variable_graph_def = sess.graph.as_graph_def()

        if inline_functions:
          # Run Grappler to create the VarOpHandle --> Placeholder -->
          # ResourceVariable pattern.
          variable_graph_def = self._inline_functions(
              variable_graph_def, ["variable_node", "output_node"])

        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                session=sess,
                graph_def=variable_graph_def,
                output_node_names=["output_node"]))

    self._ensure_no_variables_in_graph(constant_graph_def)

  def testReferenceVariables(self):
    # [FUNCTION DOC]
    # Scenario: freezes a graph with reference variables using the shared conversion helper.
    """Freezes a graph with reference variables."""
    self._test_variable_to_const_conversion(use_resource=False)

  def testResourceVariables(self):
    # [FUNCTION DOC]
    # Scenario: freezes a graph with resource variables using the shared conversion helper.
    """Freezes a graph with resource variables."""
    self._test_variable_to_const_conversion(use_resource=True)

  def testWithFunctions(self):
    # [FUNCTION DOC]
    # Scenario: freezes a graph that calls a defined function, without inlining.
    """Freezes a graph with functions."""
    self._test_convert_variables_with_functions(inline_functions=False)

  def testWithInlinedFunctions(self):
    # [FUNCTION DOC]
    # Scenario: freezes a graph with a defined function after the optimizer has inlined it.
    """Freezes a graph with functions that have been inlined using Grappler."""
    self._test_convert_variables_with_functions(inline_functions=True)

  def testGraphWithSwitch(self):
    # [FUNCTION DOC]
    # Scenario: a graph using a version 1 case over resource variables, producing a switch with the
    # resource data type. Converts it and expects no variable ops remain.
    """Freezes a graph which contains a Switch with type RESOURCE_DT."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        x = variable_scope.get_variable("var_x", initializer=1.0)
        y = variable_scope.get_variable("var_y", initializer=2.0)
        f1 = lambda: variable_scope.get_variable("var_f1", initializer=17.0)
        f2 = lambda: variable_scope.get_variable("var_f2", initializer=23.0)
        cond_node = control_flow_case.case([(gen_math_ops.less(x, y), f1)],
                                           default=f2)
        _ = math_ops.multiply(cond_node, 2.0, name="output_node")

        with session_lib.Session() as sess:
          sess.run(variables.global_variables_initializer())
          variable_graph_def = sess.graph.as_graph_def()

          constant_graph_def = (
              convert_to_constants
              .convert_variables_to_constants_from_session_graph(
                  session=sess,
                  graph_def=variable_graph_def,
                  output_node_names=["output_node"]))

    self._ensure_no_variables_in_graph(constant_graph_def)

  def testConvertSingleVariable(self):
    # [FUNCTION DOC]
    # Scenario: one reference variable, converted keeping only its read output. Expects the
    # variable node to become a float constant of value one and the read node to be an identity.
    """Tests that a single variable is properly converted to a constant."""

    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=False):
        _ = variable_scope.get_variable("x", initializer=1.0)
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["x/read"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "x/read" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }""")

  def testConvertSingleResourceVariable(self):
    # [FUNCTION DOC]
    # Scenario: one resource variable converted. Expects the variable to become a constant and the
    # read to become an identity, compared as an expected subgraph.
    """Tests that a resource variable is properly converted to a constant."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        _ = variable_scope.get_variable("x", initializer=1.0)
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["x/Read/ReadVariableOp"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "x/Read/ReadVariableOp" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }""")

  def testConvertOneVariableOfTwo(self):
    # [FUNCTION DOC]
    # Scenario: two reference variables multiplied; the second is deny-listed. Expects the first
    # converted to a constant and the second kept as a variable, checked against an expected subgraph.
    """Tests that one variable can be kept unconverted."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=False):
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=1.0)
        _ = math_ops.multiply(x, y, name="out")
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess,
                variable_graph_def, ["out"],
                variable_names_denylist=["y"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "x/read" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "y" op: "VariableV2"
              attr { key: "dtype" value { type: DT_FLOAT } }
            }
            node {
              name: "y/read" op: "Identity" input: "y"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "out" op: "Mul" input: "x/read" input: "y/read"
              attr {key: "T" value {type: DT_FLOAT}}
            }""")

  def testConvertOneResourceVariableOfTwo(self):
    # [FUNCTION DOC]
    # Scenario: same as the previous test but with resource variables; the second is deny-listed
    # and stays a variable handle with a read, while the first becomes constant.
    """Tests that one variable can be kept unconverted."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=1.0)
        _ = math_ops.multiply(x, y, name="out")
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess,
                variable_graph_def, ["out"],
                variable_names_denylist=["y"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "y" op: "VarHandleOp"
              attr { key: "dtype" value { type: DT_FLOAT } }
            }
            node {
              name: "out/ReadVariableOp" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "out/ReadVariableOp_1" op: "ReadVariableOp" input: "y"
              attr { key: "dtype" value { type: DT_FLOAT } }
            }
            node {
              name: "out" op: "Mul"
              input: "out/ReadVariableOp" input: "out/ReadVariableOp_1"
              attr {key: "T" value {type: DT_FLOAT}}
            }""")

  def testConvertIdentityChain(self):
    # [FUNCTION DOC]
    # Scenario: a resource variable followed by a chain of two identity ops. Expects the variable
    # to become a constant and the identities to be preserved in order.
    """Tests that a chain of Identity ops is converted properly."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        x = variable_scope.get_variable("x", initializer=1.0)
        y = array_ops.identity(x, name="y")
        _ = array_ops.identity(y, name="z")
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["z"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "y/ReadVariableOp" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "y" op: "Identity" input: "y/ReadVariableOp"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "z" op: "Identity" input: "y"
              attr { key: "T" value { type: DT_FLOAT } }
            }""")

  def testConvertCase(self):
    # [FUNCTION DOC]
    # Scenario: a version 1 style case with reference variables (control flow v2 disabled).
    # Expects both variables converted to constants and the cond/switch/merge structure preserved.
    """Tests that a v1 case() construction converts properly."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=False):
        control_flow_v2_toggles.disable_control_flow_v2()
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=2.0)
        _ = control_flow_case.case([(gen_math_ops.less(x, y), lambda: x)],
                                   default=lambda: y)
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["case/cond/Merge"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "y" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 2 }}}
            }
            node {name: "x/read" op: "Identity" input: "x"}
            node {name: "y/read" op: "Identity" input: "y"}
            node {name: "Less" op: "Less" input: "x/read" input: "y/read"}
            node {name: "case/cond/pred_id" op: "Identity" input: "Less"}
            node {
              name: "case/cond/Switch_1" op: "Switch"
              input: "case/cond/pred_id" input: "x/read"
            }
            node {
              name: "case/cond/Switch_2" op: "Switch"
              input: "case/cond/pred_id" input: "y/read"
            }
            node {
              name: "case/cond/Merge" op: "Merge"
              input: "case/cond/Switch_2" input: "case/cond/Switch_1:1"
              attr {key: "T" value {type: DT_FLOAT}}
            }""")

  def testConvertV2Case(self):
    # [FUNCTION DOC]
    # Scenario: a version 2 case (control flow v2 enabled while building) with reference variables.
    # Expects variables converted to constants, including inside the generated branch functions,
    # matching an expected graph with its function library. Disables control flow v2 afterwards.
    """Tests that a v2 case() converts properly."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=False):
        control_flow_v2_toggles.enable_control_flow_v2()
        a = variable_scope.get_variable("a", initializer=2.0)
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=2.0)
        _ = control_flow_case.case([(gen_math_ops.less(x, y), lambda: a)],
                                   default=lambda: y)
        control_flow_v2_toggles.disable_control_flow_v2()
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["case/cond"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "y" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 2 }}}
            }
            node {name: "x/read" op: "Identity" input: "x"}
            node {name: "y/read" op: "Identity" input: "y"}
            node {name: "Less" op: "Less" input: "x/read" input: "y/read"}
            node {
              name: "case/cond" op: "StatelessIf"
              input: "Less" input: "a/read" input: "y/read"
              attr {key: "Tcond" value {type: DT_BOOL}}
              attr {key: "Tin" value {list {type: DT_FLOAT type: DT_FLOAT}}}
              attr {key: "Tout" value {list {type: DT_FLOAT}}}
            }
            library {
              function {
                signature {
                  name: "case_cond_false_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "y_read_0" type: DT_FLOAT}
                  output_arg {name: "y_read" type: DT_FLOAT}
                }
              }
              function {
                signature {
                  name: "case_cond_true_frozen_0"
                  input_arg {name: "a_read_0" type: DT_FLOAT}
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  output_arg {name: "a_read" type: DT_FLOAT}
                }
              }
            }""")

  def testConvertV2ResourceCase(self):
    # [FUNCTION DOC]
    # Scenario: a version 2 case with resource variables. Expects the variables converted to
    # constants, reads rewritten to identities inside the branch functions, matching the expected graph.
    """Tests that a v2 case() with resource variables converts properly."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        control_flow_v2_toggles.enable_control_flow_v2()
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=2.0)
        _ = control_flow_case.case([(gen_math_ops.less(x, y), lambda: x)],
                                   default=lambda: y)
        control_flow_v2_toggles.disable_control_flow_v2()
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["case/cond"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {name: "x" op: "Const"}
            node {name: "y" op: "Const"}
            node {
              name: "case/cond" op: "If" input: "Less" input: "x" input: "y"
              attr {key: "Tcond" value {type: DT_BOOL}}
              attr {key: "Tin" value {list {type: DT_FLOAT type: DT_FLOAT}}}
              attr {key: "Tout" value {list {type: DT_FLOAT}}}
            }
            library {
              function {
                signature {
                  name: "case_cond_false_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "readvariableop_y" type: DT_FLOAT}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                }
              }
              function {
                signature {
                  name: "case_cond_true_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "readvariableop_x" type: DT_FLOAT}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                }
              }
            }""")

  def testConvertV2UnconvertedResourceNestedCase(self):
    # [FUNCTION DOC]
    # Scenario: nested version 2 case with three resource variables where y is deny-listed.
    # Expects x and z to become constants while y stays a variable handle, with the handle passed as
    # a resource argument through the nested functions and read inside them; compared with an
    # expected graph and function library.
    """Tests unconverted variable propagation through nested functions."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        control_flow_v2_toggles.enable_control_flow_v2()
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=2.0)
        z = variable_scope.get_variable("z", initializer=3.0)
        # pylint: disable=g-long-lambda
        _ = control_flow_case.case(
            [(gen_math_ops.less(x, y), lambda: x)],
            default=lambda: control_flow_case.case(
                [(gen_math_ops.less(z, y), lambda: z)], default=lambda: y))
        # pylint: enable=g-long-lambda
        control_flow_v2_toggles.disable_control_flow_v2()
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess,
                variable_graph_def, ["case/cond"],
                variable_names_denylist=["y"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {name: "x" op: "Const"}
            node {name: "y" op: "VarHandleOp"}
            node {name: "z" op: "Const"}

            node {name: "Less/ReadVariableOp" op: "Identity" input: "x"}
            node {name: "Less/ReadVariableOp_1" op: "ReadVariableOp" input: "y"}

            node {
              name: "case/cond" op: "If"
              input: "x" input: "z" input: "y"
              attr {
                key: "Tin"
                value {list
                  {type: DT_FLOAT type: DT_FLOAT type: DT_RESOURCE}}}
              attr {
                key: "_read_only_resource_inputs"
                value {list {i: 1 i: 2 i: 3}}}
              attr {key: "then_branch"
                    value {func {name: "case_cond_true_frozen_0"}}}
              attr {key: "else_branch"
                    value {func {name: "case_cond_false_frozen_0"}}}
              attr {key: "output_shapes" value {list {shape {}}}}
            }
            library {
              function {
                signature {
                  name: "case_cond_true_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "placeholder_1" type: DT_RESOURCE}
                  input_arg {name: "readvariableop_x" type: DT_FLOAT}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                  is_stateful: true
                }

                node_def {name: "ReadVariableOp" op: "Identity"
                  input: "readvariableop_x"}}

              function {
                signature {
                  name: "case_cond_false_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "less_readvariableop_1_y" type: DT_RESOURCE}
                  input_arg {name: "less_readvariableop_z" type: DT_FLOAT}
                  output_arg {name: "case_cond_identity" type: DT_FLOAT}
                  is_stateful: true
                }

                node_def {name: "Less/ReadVariableOp_1" op: "ReadVariableOp"
                  input: "less_readvariableop_1_y"}

                node_def {name: "Less/ReadVariableOp" op: "Identity"
                  input: "less_readvariableop_z"}

                node_def {name: "case/cond" op: "If"
                  input: "less_readvariableop_z"
                  input: "less_readvariableop_1_y"
                  attr {
                    key: "Tin"
                    value {list {type: DT_FLOAT type: DT_RESOURCE}}}
                  attr {key: "then_branch"
                        value {func {name: "case_cond_true_frozen_1"}}}
                  attr {key: "else_branch"
                        value {func {name: "case_cond_false_frozen_1"}}}
                  attr {
                    key: "_read_only_resource_inputs"
                    value {list {i: 1 i: 2}}}}}

              function {
                signature {
                  name: "case_cond_false_frozen_1"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "readvariableop_y" type: DT_RESOURCE}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                  is_stateful: true
                }

                node_def {name: "ReadVariableOp" op: "ReadVariableOp"
                  input: "readvariableop_y"}}

              function {
                signature {
                  name: "case_cond_true_frozen_1"
                  input_arg {name: "placeholder" type: DT_RESOURCE}
                  input_arg {name: "readvariableop_z" type: DT_FLOAT}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                  is_stateful: true
                }

                node_def {name: "ReadVariableOp" op: "Identity"
                  input: "readvariableop_z"}}}""")

  def _addNoinlineAttributeToFunction(self, saved_model_dir, func_name):
    # [FUNCTION DOC]
    # HELPER: fixture for the function inlining disabled test.
    # Reads the saved model in saved_model_dir, marks every library function whose name (after the
    # inference prefix) starts with func_name as not inlinable, deletes the old saved model file and
    # writes the modified model back to the same location. Side effects: file removal and writing.
    # Returns nothing.
    saved_model_proto = loader_impl.parse_saved_model(saved_model_dir)
    new_saved_model = saved_model_pb2.SavedModel()
    new_saved_model.CopyFrom(saved_model_proto)
    new_meta_graph_def = new_saved_model.meta_graphs[0]
    prefix_len = len("__inference_")
    for func_def in new_meta_graph_def.graph_def.library.function:
      func_name_without_prefix = func_def.signature.name[prefix_len:]
      if func_name_without_prefix.startswith(func_name):
        func_def.attr["_noinline"].CopyFrom(attr_value_pb2.AttrValue(b=True))
    old_saved_model_file = os.path.join(saved_model_dir,
                                        constants.SAVED_MODEL_FILENAME_PB)
    if os.path.exists(old_saved_model_file):
      os.remove(old_saved_model_file)
    path = os.path.join(
        compat.as_bytes(saved_model_dir),
        compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    file_io.write_string_to_file(
        path, new_saved_model.SerializeToString(deterministic=True))

  @test_util.run_v2_only
  def testVariableModelWithFunctionAndFunctionInliningDisabled(self):
    # [FUNCTION DOC]
    # Scenario: a TF1 graph model with a traced method using two variables is saved, the method is
    # marked not inlinable in the saved model file, then loaded and converted with the TF2 converter.
    # Expects the converted graph to contain no variable ops. TF2 only.
    """Test a model with Variables and disable function inlining."""

    class BasicModel:
      # [CLASS DOC]
      # Plain local model with one eagerly created variable and one lazily created variable, and a
      # traced add method, plus a run method that calls it. Used by the inlining disabled test.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets the first variable attribute to unset and creates the second variable with value two.
        self.v1 = None
        self.v2 = variables.Variable(2.)

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[1], dtype=dtypes.float32)
      ])
      def add_all(self, x):
        # [FUNCTION DOC]
        # Traced method for a one element float input. On first trace creates the first variable with
        # value three; returns the input plus both variables. Side effect: sets that attribute.
        if self.v1 is None:
          self.v1 = variables.Variable(3.)
        return x + self.v1 + self.v2

      def run(self, x):
        # [FUNCTION DOC]
        # Calls the traced add method on the input and returns its result.
        y = self.add_all(x)
        return y

    save_dir = os.path.join(self.get_temp_dir(), "frozen_saved_model")
    with ops.Graph().as_default():
      model = BasicModel()
      a = array_ops.placeholder(dtypes.float32, shape=[1])
      b = model.run(a)
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        simple_save.simple_save(sess, save_dir, {"myinput": a}, {"myoutput": b})

    # Add _noinline to the SavedModel.
    self._addNoinlineAttributeToFunction(
        saved_model_dir=save_dir, func_name="add_all")

    saved_model = load(save_dir)
    func = saved_model.signatures["serving_default"]
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
    constant_graph_def = frozen_func.graph.as_graph_def()
    self._ensure_no_variables_in_graph(constant_graph_def)


if __name__ == "__main__":
  test.main()
