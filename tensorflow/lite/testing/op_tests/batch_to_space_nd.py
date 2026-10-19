# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of the batch-to-space (N-d) op.
# Defines one registered test-maker that varies dtype, input shape, block shape, crops, whether
# block shape and crops are constants or fed tensors, and quantization modes, then hands the
# graphs to the shared zip-test generator. Importing registers the maker in the global registry.
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
"""Test configs for batch_to_space_nd."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_batch_to_space_nd_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Builds parameter sets for 4-D inputs
  # (several block shapes and crops, int/float dtypes), a single-batch no-op case, a 3-D case, and
  # fully-quantized variants (optionally 16x8). If the options request running with the flex
  # delegate, it also appends a 6-D case with three spatial dimensions. Finally calls the zip-test
  # generator with the local builders. Returns nothing.
  """Make a set of tests to do batch_to_space_nd."""

  test_parameters = [
      {
          "dtype": [tf.float32, tf.int64, tf.int32],
          "input_shape": [[12, 3, 3, 1]],
          "block_shape": [[1, 4], [2, 2], [3, 4]],
          "crops": [[[0, 0], [0, 0]], [[1, 1], [1, 1]]],
          "constant_block_shape": [True, False],
          "constant_crops": [True, False],
          "dynamic_range_quantize": [False],
      },
      # Single batch (no-op)
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 3, 3, 1]],
          "block_shape": [[1, 1]],
          "crops": [[[0, 0], [0, 0]], [[1, 1], [1, 1]]],
          "constant_block_shape": [True],
          "constant_crops": [True],
          "dynamic_range_quantize": [True, False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 3, 3, 1]],
          "block_shape": [[1, 1]],
          "crops": [[[0, 0], [0, 0]], [[1, 1], [1, 1]]],
          "constant_block_shape": [True],
          "constant_crops": [True],
          "fully_quantize": [True],
          "quant_16x8": [False, True],
      },
      # 3D use case.
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 3, 3]],
          "block_shape": [[1]],
          "crops": [[[0, 0]], [[1, 1]]],
          "constant_block_shape": [True],
          "constant_crops": [True],
          "dynamic_range_quantize": [True, False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 3, 3]],
          "block_shape": [[1]],
          "crops": [[[0, 0]], [[1, 1]]],
          "constant_block_shape": [True],
          "constant_crops": [True],
          "fully_quantize": [True],
          "quant_16x8": [False, True],
      },
  ]

  if options.run_with_flex:
    # Non-4D use case: 1 batch dimension, 3 spatial dimensions, 2 others.
    test_parameters = test_parameters + [{
        "dtype": [tf.float32],
        "input_shape": [[8, 2, 2, 2, 1, 1]],
        "block_shape": [[2, 2, 2]],
        "crops": [[[0, 0], [0, 0], [0, 0]]],
        "constant_block_shape": [True, False],
        "constant_crops": [True, False],
        "dynamic_range_quantize": [False],
    }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the batch-to-space zip tests.
    # Creates the data placeholder, then takes block shape and crops either as constants from the
    # parameters or as extra int32 placeholders (named shape and crops) when the corresponding
    # constant flag is false. Applies batch-to-space. Returns the list of all input placeholders
    # (data first, then block shape and crops if non-constant) and the list with the output tensor.
    """Build a batch_to_space graph given `parameters`."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    input_tensors = [input_tensor]

    # Get block_shape either as a const or as a placeholder (tensor).
    if parameters["constant_block_shape"]:
      block_shape = parameters["block_shape"]
    else:
      shape = [len(parameters["block_shape"])]
      block_shape = tf.compat.v1.placeholder(
          dtype=tf.int32, name="shape", shape=shape)
      input_tensors.append(block_shape)

    # Get crops either as a const or as a placeholder (tensor).
    if parameters["constant_crops"]:
      crops = parameters["crops"]
    else:
      shape = [len(parameters["crops"]), 2]
      crops = tf.compat.v1.placeholder(
          dtype=tf.int32, name="crops", shape=shape)
      input_tensors.append(crops)

    out = tf.batch_to_space(input_tensor, block_shape, crops)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the batch-to-space zip tests.
    # Generates random data in the range minus one to one for the main input, appends arrays of the
    # block shape and crops when those are fed tensors rather than constants, runs the session, and
    # returns the list of input values together with the computed expected outputs.
    values = [
        create_tensor_data(
            parameters["dtype"],
            parameters["input_shape"],
            min_value=-1.0,
            max_value=1.0,
        )
    ]
    if not parameters["constant_block_shape"]:
      values.append(np.array(parameters["block_shape"]))
    if not parameters["constant_crops"]:
      values.append(np.array(parameters["crops"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
