# [MODULE DOC]
# Tests for the MLIR conversion APIs: importing graph definitions (including invalid input and
# explicit input/output signatures) and concrete functions (debug locations, calls, control
# outputs) and checking the produced MLIR text with pattern matches.
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
# =============================================================================
"""Tests for python.compiler.mlir."""

from tensorflow.python.compiler.mlir import mlir
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.pywrap_mlir import import_graphdef


class MLIRGraphDefImportTest(test.TestCase):
  # [CLASS DOC]
  # Tests for converting graph definitions to MLIR.

  def testImport(self):
    # [FUNCTION DOC]
    # Converts an empty graph definition and asserts the output has a main function.
    """Tests the basic flow of `tf.mlir.experimental.convert_graph_def`."""
    mlir_module = mlir.convert_graph_def('')
    # An empty graph should contain at least an empty main function.
    self.assertIn('func @main', mlir_module)

  def testInvalidPbtxt(self):
    # [FUNCTION DOC]
    # Asserts converting invalid proto text raises an invalid-argument error about parsing.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Could not parse input proto'):
      mlir.convert_graph_def('some invalid proto')

  def testGraphDefToTf(self):
    # [FUNCTION DOC]
    # Builds a graph of an add function over two 10 by 10 float inputs, imports it with explicit
    # input names, types, shapes and output name, and asserts the main function signature and the
    # input/output attributes. Repeats with scalar shapes, then asserts invalid-argument errors
    # for mismatched input name/type counts and for incompatible shapes.
    """Tests the basic flow of `tf.mlir.experimental.convert_graph_def`

        with tf-standard-pipeline converting all the way to the TF dialect.
    """

    tensor_shape = (10, 10)

    @def_function.function(
        input_signature=(
            tensor_spec.TensorSpec(shape=tensor_shape, dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=tensor_shape, dtype=dtypes.float32),
        ))
    def add_func(lhs, rhs):
      # [FUNCTION DOC]
      # Traced function returning the elementwise sum of its two inputs.
      return math_ops.add(lhs, rhs)

    tf_graph_def = add_func.get_concrete_function().graph.as_graph_def()

    mlir_tf = import_graphdef(
        tf_graph_def,
        "tf-standard-pipeline",
        False,
        input_names=["lhs", "rhs"],
        input_data_types=["DT_FLOAT", "DT_FLOAT"],
        input_data_shapes=["10,10", "10,10"],
        output_names=["Add"])
    # Check whether the mlir-function signature has the mentioned
    # inputs and outputs.
    self.assertRegex(
        mlir_tf,
        r"func @main\(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>")
    self.assertRegex(mlir_tf, r'inputs = "lhs,rhs"')
    self.assertRegex(mlir_tf, r'outputs = "Add"')

    # Same check with scalar input (empty input shape).
    mlir_tf = import_graphdef(
        tf_graph_def,
        "tf-standard-pipeline",
        False,
        input_names=["lhs", "rhs"],
        input_data_types=["DT_FLOAT", "DT_FLOAT"],
        input_data_shapes=["", ""],
        output_names=["Add"])
    self.assertRegex(mlir_tf,
                     r"func @main\(%arg0: tensor<f32>, %arg1: tensor<f32>")

    # Test invalid test cases where no. of input names is invalid/wrong.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Length of input node array and data type doesn't match"):

      import_graphdef(
          tf_graph_def,
          "tf-standard-pipeline",
          False,
          input_names=["lhs"],
          input_data_types=["DT_FLOAT", "DT_FLOAT"],
          input_data_shapes=["10,10", "10,10"],
          output_names=["Add"])

    # Test invalid test cases where the input shapes argument is wrong.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Dimensions must be equal"):

      import_graphdef(
          tf_graph_def,
          "tf-standard-pipeline",
          False,
          input_names=["lhs", "rhs"],
          input_data_types=["DT_FLOAT", "DT_FLOAT"],
          input_data_shapes=["10,11", "10,10"],
          output_names=["Add"])


class MLIRConcreteFunctionImportTest(test.TestCase):
  # [CLASS DOC]
  # Tests for converting concrete functions to MLIR (v2 only).

  @test_util.run_v2_only
  def testImport(self):
    # [FUNCTION DOC]
    # Converts a traced squaring function with debug info and asserts the module has a function
    # with the name and a location pointing at this test file.

    @def_function.function
    def sqr(i):
      # [FUNCTION DOC]
      # Traced function returning its input squared.
      return i * i

    concrete_function = sqr.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32))
    mlir_module = mlir.convert_function(concrete_function, show_debug_info=True)
    self.assertRegex(mlir_module, r'func @.*sqr.*\(')
    self.assertRegex(mlir_module, r'loc\(".*mlir_test.py":.*:1\)')

  @test_util.run_v2_only
  def testImportWithCall(self):
    # [FUNCTION DOC]
    # Converts a function that calls another traced function and asserts both a public caller and
    # a private callee function appear in the module.

    @def_function.function
    def callee(i):
      # [FUNCTION DOC]
      # Traced identity function, called by the caller.
      return i

    @def_function.function
    def caller(i):
      # [FUNCTION DOC]
      # Traced function that returns the result of calling the callee.
      return callee(i)

    concrete_function = caller.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32))
    mlir_module = mlir.convert_function(concrete_function)
    self.assertRegex(mlir_module, r'func @.*caller.*\(')
    self.assertRegex(mlir_module, r'func private @.*callee.*\(')

  @test_util.run_v2_only
  def testImportWithControlRet(self):
    # [FUNCTION DOC]
    # Converts a function that only prints a message, with an empty pass pipeline, and asserts the
    # print op and a fetch of a control value appear in the module.

    @def_function.function
    def logging():
      # [FUNCTION DOC]
      # Traced function that prints a fixed message and returns nothing.
      logging_ops.print_v2('some message')

    concrete_function = logging.get_concrete_function()
    mlir_module = mlir.convert_function(concrete_function, pass_pipeline='')
    self.assertRegex(mlir_module, r'tf\.PrintV2')
    self.assertRegex(mlir_module, r'tf_executor.fetch.*: !tf_executor.control')


if __name__ == '__main__':
  test.main()
