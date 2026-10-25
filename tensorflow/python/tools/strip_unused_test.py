# [MODULE DOC]
# Tests for strip_unused_lib: write small graphs to the test temporary directory, strip them
# with the file based function, reload the result and check node counts, op types and outputs.
# Covers a single input node and multiple inputs with per-input dtypes.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests the node stripping tool."""

import os

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.tools import strip_unused_lib


class StripUnusedTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for graph stripping. Each test writes a text GraphDef to the temporary directory,
  # runs strip_unused_from_files to a binary output, then imports and runs the result.

  def testStripUnused(self):
    # [FUNCTION DOC]
    # Builds a graph: constant, subtract (the wanted input), multiply by two (output), and an
    # extra later add; checks the output value, writes the graph. Asserts that stripping with a
    # missing input name raises a key error and with a tensor-style name raises a value error.
    # Then strips with the subtract node as input and asserts the result has three nodes, no
    # add or subtract ops, a shape attribute on the input, and that feeding the input gives the
    # doubled value.
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"

    # We'll create an input graph that has a single constant containing 1.0,
    # and that then multiplies it by 2.
    with ops.Graph().as_default():
      constant_node = constant_op.constant(1.0, name="constant_node")
      wanted_input_node = math_ops.subtract(constant_node,
                                            3.0,
                                            name="wanted_input_node")
      output_node = math_ops.multiply(
          wanted_input_node, 2.0, name="output_node")
      math_ops.add(output_node, 2.0, name="later_node")
      sess = session.Session()
      output = self.evaluate(output_node)
      self.assertNear(-4.0, output, 0.00001)
      graph_io.write_graph(sess.graph, self.get_temp_dir(), input_graph_name)

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
    input_binary = False
    output_binary = True
    output_node_names = "output_node"
    output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)

    def strip(input_node_names):
      # [FUNCTION DOC]
      # HELPER: Nested wrapper that runs the file based stripping on the test's graph paths.
      # Takes the comma separated input node names; uses the enclosing input and output paths,
      # format flags, output node names and float dtype. Returns the library result (nothing on
      # success); errors propagate to the caller.
      strip_unused_lib.strip_unused_from_files(input_graph_path, input_binary,
                                               output_graph_path, output_binary,
                                               input_node_names,
                                               output_node_names,
                                               dtypes.float32.as_datatype_enum)

    with self.assertRaises(KeyError):
      strip("does_not_exist")

    with self.assertRaises(ValueError):
      strip("wanted_input_node:0")

    input_node_names = "wanted_input_node"
    strip(input_node_names)

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with ops.Graph().as_default():
      output_graph_def = graph_pb2.GraphDef()
      with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = importer.import_graph_def(output_graph_def, name="")

      self.assertEqual(3, len(output_graph_def.node))
      for node in output_graph_def.node:
        self.assertNotEqual("Add", node.op)
        self.assertNotEqual("Sub", node.op)
        if node.name == input_node_names:
          self.assertTrue("shape" in node.attr)

      with session.Session() as sess:
        input_node = sess.graph.get_tensor_by_name("wanted_input_node:0")
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = sess.run(output_node, feed_dict={input_node: [10.0]})
        self.assertNear(20.0, output, 0.00001)

  def testStripUnusedMultipleInputs(self):
    # [FUNCTION DOC]
    # Builds a graph where two subtract nodes feed a multiply, plus an extra add. Strips with both
    # subtract nodes as inputs, using a list of two float dtype enums. Asserts the result has
    # three nodes with no add or subtract ops, and that feeding values ten and minus five to the
    # inputs yields minus fifty at the output.
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"

    # We'll create an input graph that multiplies two input nodes.
    with ops.Graph().as_default():
      constant_node1 = constant_op.constant(1.0, name="constant_node1")
      constant_node2 = constant_op.constant(2.0, name="constant_node2")
      input_node1 = math_ops.subtract(constant_node1, 3.0, name="input_node1")
      input_node2 = math_ops.subtract(constant_node2, 5.0, name="input_node2")
      output_node = math_ops.multiply(
          input_node1, input_node2, name="output_node")
      math_ops.add(output_node, 2.0, name="later_node")
      sess = session.Session()
      output = self.evaluate(output_node)
      self.assertNear(6.0, output, 0.00001)
      graph_io.write_graph(sess.graph, self.get_temp_dir(), input_graph_name)

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
    input_binary = False
    input_node_names = "input_node1,input_node2"
    input_node_types = [
        dtypes.float32.as_datatype_enum, dtypes.float32.as_datatype_enum
    ]
    output_binary = True
    output_node_names = "output_node"
    output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)

    strip_unused_lib.strip_unused_from_files(input_graph_path, input_binary,
                                             output_graph_path, output_binary,
                                             input_node_names,
                                             output_node_names,
                                             input_node_types)

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with ops.Graph().as_default():
      output_graph_def = graph_pb2.GraphDef()
      with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = importer.import_graph_def(output_graph_def, name="")

      self.assertEqual(3, len(output_graph_def.node))
      for node in output_graph_def.node:
        self.assertNotEqual("Add", node.op)
        self.assertNotEqual("Sub", node.op)
        if node.name == input_node_names:
          self.assertTrue("shape" in node.attr)

      with session.Session() as sess:
        input_node1 = sess.graph.get_tensor_by_name("input_node1:0")
        input_node2 = sess.graph.get_tensor_by_name("input_node2:0")
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = sess.run(output_node,
                          feed_dict={input_node1: [10.0],
                                     input_node2: [-5.0]})
        self.assertNear(-50.0, output, 0.00001)


if __name__ == "__main__":
  test.main()
