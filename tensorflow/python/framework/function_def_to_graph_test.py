# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Unit tests for the function definition to graph conversion utilities: function_def_to_graph,
# function_def_to_graph_def and the recursive library copier. Covers inputs/outputs, shapes,
# resource handles, library inclusion, control dependencies and argument attributes.
"""Tests for tensorflow.python.framework.function_def_to_graph."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import op_def_library
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class FunctionDefToGraphTest(test.TestCase):
  # [CLASS DOC]
  # Tests for converting a function definition into a function graph: inputs and outputs, shape
  # handling, resource handle inputs, and library function inclusion/copying. Shares a fixture
  # that builds a function definition computing sums of squares and cubes.

  def _build_function_def(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder shared by the function graph conversion tests.
    # In a fresh graph creates float placeholders x and y, outputs sum_squares and sum_cubes (sums
    # of powers), converts the ops to a function definition with those inputs and outputs, and
    # renames its signature. Takes nothing; returns the function definition.
    with ops.Graph().as_default() as g:
      # Inputs
      x = array_ops.placeholder(dtypes.float32, name="x")
      y = array_ops.placeholder(dtypes.float32, name="y")

      # Outputs
      sum_squares = math_ops.add_n(
          [math_ops.pow(x, 2), math_ops.pow(y, 2)], name="sum_squares")
      sum_cubes = math_ops.add_n(
          [math_ops.pow(x, 3), math_ops.pow(y, 3)], name="sum_cubes")
    fdef = graph_to_function_def.graph_to_function_def(
        g,
        g.get_operations(),
        [x, y],  # Inputs
        [sum_squares, sum_cubes])  # Outputs.
    fdef.signature.name = "_whats_in_a_name"
    return fdef

  @test_util.run_deprecated_v1
  def testInputsAndOutputs(self):
    # [FUNCTION DOC]
    # Test: the converted function graph has the right name, inputs and outputs. Runs in legacy
    # graph mode. Setup: shared definition computing sums of squares and cubes of x and y.
    # Action: converts, then runs the graph in a session feeding x as 2 and y as 3.
    # Expects the graph name to match the definition, inputs to echo 2 and 3, and outputs 13 and 35.
    fdef = self._build_function_def()
    g = function_def_to_graph.function_def_to_graph(fdef)
    self.assertEqual(g.name, "_whats_in_a_name")
    with self.session(graph=g) as sess:
      inputs = sess.run(g.inputs, feed_dict={"x:0": 2, "y:0": 3})
      self.assertSequenceEqual(inputs, [2.0, 3.0])
      outputs = sess.run(g.outputs, feed_dict={"x:0": 2, "y:0": 3})
      self.assertSequenceEqual(outputs, [13.0, 35.0])

  def testShapes(self):
    # [FUNCTION DOC]
    # Test: input shape handling in function graph conversion. Three cases: no shapes given (all
    # inputs and outputs unknown); both inputs given length-five vector shapes (propagated to
    # outputs); first unknown and second a five-by-seven shape (outputs take the known shape).
    # Also expects a value error when the shapes list length differs from the input count.
    fdef = self._build_function_def()

    g = function_def_to_graph.function_def_to_graph(fdef)
    self.assertIsNone(g.inputs[0].shape.dims)  # Unknown dims.
    self.assertIsNone(g.inputs[1].shape.dims)  # Unknown dims.
    self.assertIsNone(g.outputs[0].shape.dims)  # Unknown dims.
    self.assertIsNone(g.outputs[1].shape.dims)  # Unknown dims.

    g = function_def_to_graph.function_def_to_graph(
        fdef,
        input_shapes=[
            tensor_shape.TensorShape([5]),
            tensor_shape.TensorShape([5])
        ])
    self.assertSequenceEqual(g.inputs[0].shape.dims, [5])
    self.assertSequenceEqual(g.inputs[1].shape.dims, [5])
    self.assertSequenceEqual(g.outputs[0].shape.dims, [5])
    self.assertSequenceEqual(g.outputs[1].shape.dims, [5])

    g = function_def_to_graph.function_def_to_graph(
        fdef, input_shapes=[None, tensor_shape.TensorShape([5, 7])])
    self.assertIsNone(g.inputs[0].shape.dims)
    self.assertSequenceEqual(g.inputs[1].shape.dims, [5, 7])
    self.assertSequenceEqual(g.outputs[0].shape.dims, [5, 7])
    self.assertSequenceEqual(g.outputs[1].shape.dims, [5, 7])

    # Should raise a ValueError if the length of input_shapes does not match
    # the number of input args in FunctionDef.signature.input_arg.
    with self.assertRaises(ValueError):
      g = function_def_to_graph.function_def_to_graph(
          fdef, input_shapes=[tensor_shape.TensorShape([5, 7])])

  def testResourceHandleInputShapes(self):
    # [FUNCTION DOC]
    # Test: shape handling for resource handle inputs during conversion. Setup: in a graph, a
    # variable and a traced function with a fixed-rank int input capturing the variable; the
    # library function definition is taken with shapes recorded. Action: converts it to a function
    # graph. Expects the first input shape to be unknown-batch by 2 by 2 and the captured resource
    # handle input to have a scalar shape.
    # Test that shape inference and validation with resource handles works as
    # expected.

    # Create a graph to generate the input and handle shape attributes in the
    # FunctionDef.
    with ops.Graph().as_default() as g:
      v = variables.Variable(array_ops.ones((2, 3), dtype=dtypes.float32))

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec((None, 2, 2), dtypes.int32)])
      def lookup(inp):
        # [FUNCTION DOC]
        # Nested traced function with an int tensor input of unknown batch size: returns a mapping
        # with a gather result from the captured variable (needs non-scalar shape) and the
        # variable's resource handle (output shape must be scalar).
        return {
            # gather_nd expects a nonscalar shape for `v`, otherwise raises
            # error when doing shape inference.
            "shape inference": array_ops.gather_nd(v, inp),
            # Triggers output shape validation. Expected shape must be [].
            "handle": v.handle}

      lookup.get_concrete_function().add_to_graph()
      fdef = g.as_graph_def(add_shapes=True).library.function[0]

    fg = function_def_to_graph.function_def_to_graph(fdef)
    self.assertSequenceEqual(fg.inputs[0].shape.as_list(), [None, 2, 2])
    self.assertSequenceEqual(fg.inputs[1].shape.as_list(), [])

  def testIncludeLibraryFunctions(self):
    # [FUNCTION DOC]
    # Test: converting with library inclusion keeps called functions in the resulting graph.
    # Setup: traced functions g and f (f calls g); g is added to the default graph.
    # Action: converts f's definition with library functions included.
    # Expects the output graph definition library to contain exactly one function, named as g.
    @def_function.function
    def g(x):
      # [FUNCTION DOC]
      # Nested traced function: adds one to its input; the callee registered in the library.
      return x + 1

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # Nested traced function that calls the function g on its input and returns the result.
      return g(x)

    cfg = g.get_concrete_function(1.0)
    cfg.add_to_graph()
    gname = cfg.function_def.signature.name
    function_def = f.get_concrete_function(1.0).function_def
    func_graph = function_def_to_graph.function_def_to_graph(
        function_def, include_library_functions=True)
    graph_def = func_graph.as_graph_def()
    self.assertLen(graph_def.library.function, 1)
    self.assertEqual(graph_def.library.function[0].signature.name, gname)

  def testCopyFunctionDefToGraphDefRecursively(self):
    # [FUNCTION DOC]
    # Test: the recursive library copier copies a function and all nested callees exactly once.
    # Setup: a chain of traced functions outer, middle, inner, plus an unrelated target function
    # whose graph definition starts with an empty library; outer is added to the default graph.
    # Action: copies outer into the target graph definition with an empty copied-names set.
    # Expects the copied-names set and the target library's function names both to equal the set
    # of outer plus all function names in outer's own library.
    @def_function.function
    def inner(x):
      # [FUNCTION DOC]
      # Nested traced function: adds one to its input; innermost of the call chain.
      return x + 1

    @def_function.function
    def middle(x):
      # [FUNCTION DOC]
      # Nested traced function: calls the inner function and adds one; middle of the chain.
      return inner(x) + 1

    @def_function.function
    def outer(x):
      # [FUNCTION DOC]
      # Nested traced function: calls the middle function and adds one; outermost of the chain.
      return middle(x) + 1

    @def_function.function
    def target_func(x):
      # [FUNCTION DOC]
      # Nested traced identity function whose graph definition receives the copied functions.
      return x

    target_graph_def = target_func.get_concrete_function(1).graph.as_graph_def()

    self.assertEmpty(target_graph_def.library.function)

    concrete_outer = outer.get_concrete_function(1)
    default_graph = ops.get_default_graph()
    # Copy FunctionDefs in `outer` to the default graph.
    concrete_outer.add_to_graph(default_graph)
    outer_function_name = concrete_outer.function_def.signature.name
    copied_functions = set()
    # Copy all FunctionDefs from `outer` to `target_func`.
    function_def_to_graph.copy_function_def_to_graph_def_recursively(
        outer_function_name, target_graph_def, copied_functions, default_graph
    )

    outer_graph_def = concrete_outer.graph.as_graph_def()
    nested_function_names = {
        f.signature.name for f in outer_graph_def.library.function
    }
    expected_function_names = {outer_function_name} | nested_function_names

    self.assertEqual(copied_functions, expected_function_names)

    target_function_names = {
        f.signature.name for f in target_graph_def.library.function
    }

    self.assertEqual(target_function_names, expected_function_names)


class FunctionDefToGraphDefTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the proto-level conversion of a function definition to a graph definition, covering
  # tensor renaming, shapes, control dependencies and argument attributes. Shares a fixture that
  # builds a function definition with multi-output ops and a list-output op.

  def _build_function_def(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder shared by the graph-definition conversion tests.
    # Builds, in a fresh graph, placeholders x (float), y and z (int), two Foo1 ops and a list
    # output op, then converts them to a function definition with inputs x, y, z and outputs x,
    # d_2, e_2 and the first list output. Asserts the definition has three nodes with the expected
    # op types and inputs. Returns the function definition.
    with ops.Graph().as_default() as g:
      # Inputs:    x    y    z
      #            |\   |   /
      #            | \  |  /
      #            |  foo_1     list_output
      #            |   / \       /       \
      #            | d_1 e_1  a:1        a:0
      #            |  \   |   /           |
      #            |   \  |  /            |
      #            |    foo_2             |
      #            |     / \              |
      # Outputs:   x   d_2 e_2           a:0

      x = array_ops.placeholder(dtypes.float32, name="x")
      y = array_ops.placeholder(dtypes.int32, name="y")
      z = array_ops.placeholder(dtypes.int32, name="z")

      d_1, e_1 = op_def_library.apply_op("Foo1", name="foo_1", a=x, b=y, c=z)

      list_output0, list_output1 = test_ops.list_output(
          T=[dtypes.int32, dtypes.int32], name="list_output")

      d_2, e_2 = test_ops.foo1(a=d_1, b=e_1, c=list_output1, name="foo_2")

    fdef = graph_to_function_def.graph_to_function_def(
        g,
        g.get_operations(),
        [x, y, z],  # Inputs
        [x, d_2, e_2, list_output0])  # Outputs.

    # Assert that the FunctionDef was correctly built.
    assert len(fdef.node_def) == 3  # 2 Foo1 nodes and 1 ListOutput node.
    assert fdef.node_def[0].op == "Foo1"
    assert fdef.node_def[0].input == ["x", "y", "z"]
    assert fdef.node_def[1].op == "ListOutput"
    assert not fdef.node_def[1].input
    assert fdef.node_def[2].op == "Foo1"
    assert fdef.node_def[2].input == [
        "foo_1:d:0", "foo_1:e:0", "list_output:a:1"
    ]
    return fdef

  def testTensorNames(self):
    # [FUNCTION DOC]
    # Test: conversion renames node inputs and returns the correct nested-to-flat name mapping.
    # Setup: the shared function definition with two Foo1 nodes and a list-output node.
    # Expects inputs of the first and second Foo1 body nodes to use flat names (node:index) and the
    # mapping to equal the full expected dictionary, including control names for every input and
    # node and both list output entries.
    fdef = self._build_function_def()
    g, tensor_name_map = function_def_to_graph.function_def_to_graph_def(fdef)

    # Verify that inputs of body nodes are correctly renamed.
    # foo_1
    self.assertSequenceEqual(g.node[3].input, ["x:0", "y:0", "z:0"])
    # foo_2
    self.assertSequenceEqual(g.node[5].input,
                             ["foo_1:0", "foo_1:1", "list_output:1"])

    # Verify that the `tensor_name_map` has the correct mapping.
    self.assertDictEqual(
        tensor_name_map, {
            "x": "x:0",
            "^x": "^x",
            "y": "y:0",
            "^y": "^y",
            "z": "z:0",
            "^z": "^z",
            "foo_1:d:0": "foo_1:0",
            "foo_1:e:0": "foo_1:1",
            "^foo_1": "^foo_1",
            "list_output:a:0": "list_output:0",
            "list_output:a:1": "list_output:1",
            "^list_output": "^list_output",
            "foo_2:d:0": "foo_2:0",
            "foo_2:e:0": "foo_2:1",
            "^foo_2": "^foo_2",
        })

  def testShapes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second shape check on the first placeholder node repeats the earlier
    # unknown-rank assertion for node zero instead of checking node one, so the unknown-rank flag
    # of the second placeholder is never verified.
    # Test: explicit input shapes are written to placeholder nodes of the graph definition.
    # Setup: three inputs given a scalar shape, a vector shape of five, and nothing. Expects the
    # first two placeholders to carry matching shape attributes with known rank, and the third to
    # carry no shape attribute.
    fdef = self._build_function_def()
    g, _ = function_def_to_graph.function_def_to_graph_def(
        fdef,
        input_shapes=[
            tensor_shape.TensorShape([]),
            tensor_shape.TensorShape([5]), None
        ])
    self.assertEqual("shape" in g.node[0].attr, True)
    self.assertSequenceEqual(
        tensor_shape.TensorShape(g.node[0].attr["shape"].shape).as_list(), [])
    self.assertEqual(g.node[0].attr["shape"].shape.unknown_rank, False)
    self.assertEqual("shape" in g.node[1].attr, True)
    self.assertSequenceEqual(
        tensor_shape.TensorShape(g.node[1].attr["shape"].shape).as_list(), [5])
    self.assertEqual(g.node[0].attr["shape"].shape.unknown_rank, False)
    self.assertFalse("shape" in g.node[2].attr)

  def testControlDependencies(self):
    # [FUNCTION DOC]
    # Test: control dependencies of body ops survive the function definition to graph conversion.
    # Setup: a variable and a traced function whose constant y depends on x, the input and an
    # assign op. Action: builds a function graph from the concrete function's definition.
    # Expects op y to have exactly three control inputs named assign, inp and x, in that order.

    v = variables.Variable(1)

    @def_function.function
    def fn(inp):
      # [FUNCTION DOC]
      # Nested traced function: assigns 3 to the outer variable without reading it, creates
      # constant x, then creates constant y under control dependencies on x, the input and the
      # assign op. Takes a tensor inp; returns the python float 4.0 (converted to a constant).
      assign = v.assign(3, name="assign", read_value=False)
      x = constant_op.constant(2.0, name="x")
      # TODO(b/79881896): Test external control dependency once that's
      # supported.
      with ops.control_dependencies([x, inp, assign]):
        constant_op.constant(3.0, name="y")
      return 4.0

    inp = constant_op.constant(1.0)
    fdef = fn.get_concrete_function(inp).function_def
    func_graph = function_def_to_graph.function_def_to_graph(fdef)

    op = func_graph.get_operation_by_name("y")
    self.assertEqual(len(op.control_inputs), 3)
    self.assertEqual(op.control_inputs[0].name, "assign")
    self.assertEqual(op.control_inputs[1].name, "inp")
    self.assertEqual(op.control_inputs[2].name, "x")

  def testAttributesForArgDef(self):
    # [FUNCTION DOC]
    # Test: internal (underscore-prefixed) argument attributes on a function definition are copied
    # to the placeholder nodes. Setup: traces an identity function, sets a custom string attribute
    # on the first argument's attribute map. Action: converts the definition to a graph definition.
    # Expects exactly one placeholder node whose test attribute equals the value that was set.

    @def_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Nested identity function traced into a function definition for the test; takes a tensor x
      # and returns it unchanged.
      return x

    inp = constant_op.constant(1.0)
    fdef = fn.get_concrete_function(inp).function_def
    fdef.arg_attr[0].attr["_test_attr"].s = "value".encode("ascii")
    graph_def = function_def_to_graph.function_def_to_graph_def(fdef)
    placeholders = [
        ndef for ndef in graph_def[0].node if ndef.op == "Placeholder"
    ]
    self.assertEqual(1, len(placeholders))
    self.assertEqual(placeholders[0].attr["_test_attr"].s,
                     "value".encode("ascii"))


if __name__ == "__main__":
  test.main()
