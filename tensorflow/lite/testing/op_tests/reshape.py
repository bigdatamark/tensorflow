# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the reshape op with the target shape given either as a constant or as a tensor input.
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
"""Test configs for reshape."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_reshape_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Defines parameter groups for tensors of 420 elements (float32 and int32 with constant or fed shape,
  # a scalar-output case, a fully-quantized constant-shape case, and a zero-sized input case), with
  # target shapes that may contain a minus one wildcard. Calls make_zip_of_tests with the nested builders.
  """Make a set of tests to do reshape."""

  # All shapes below are suitable for tensors with 420 elements.
  test_parameters = [
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape": [[3, 4, 5, 7], [4, 105], [21, 5, 2, 2], [420]],
          "output_shape": [[15, 28], [420], [1, -1, 5, 7], [-1]],
          "constant_shape": [True, False],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[1]],
          "output_shape": [[]],
          "constant_shape": [True, False],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[3, 4, 5, 7], [4, 105], [21, 5, 2, 2], [420]],
          "output_shape": [[15, 28], [420], [1, -1, 5, 7], [-1]],
          "constant_shape": [True],
          "fully_quantize": [True],
      },
      {
          # Zero in input shape.
          "dtype": [tf.float32],
          "input_shape": [[1, 4, 0]],
          "output_shape": [[2, -1], [2, 0, -1]],
          "constant_shape": [True, False],
          "fully_quantize": [False],
      }
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests (the docstring says reshape tests).
    # Creates an input placeholder of the requested dtype and shape. If the shape is constant it is used
    # directly; otherwise an int32 placeholder named output_shape is created and added to the inputs.
    # Applies the reshape op. Returns the list of input tensors and the list of outputs.
    """Build the graph for reshape tests."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])

    # Get shape as either a placeholder or constants.
    if parameters["constant_shape"]:
      output_shape = parameters["output_shape"]
      input_tensors = [input_tensor]
    else:
      # The shape of the shape tensor.
      shape_tensor_shape = [len(parameters["output_shape"])]
      output_shape = tf.compat.v1.placeholder(
          dtype=tf.int32, name="output_shape", shape=shape_tensor_shape)
      input_tensors = [input_tensor, output_shape]
    out = tf.reshape(input_tensor, shape=output_shape)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Creates random data of the requested dtype and shape in the range minus 1 to 1. When the target
    # shape is not constant it is appended as an array input. Runs the graph and returns the list of
    # input values and the list of computed outputs.
    """Build inputs for reshape op."""

    values = [
        create_tensor_data(
            parameters["dtype"],
            parameters["input_shape"],
            min_value=-1,
            max_value=1)
    ]
    if not parameters["constant_shape"]:
      values.append(np.array(parameters["output_shape"]))

    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
