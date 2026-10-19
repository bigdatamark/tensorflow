# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering the gather op (select slices along
# an axis by index), with constant or runtime parameters and indices, string data, batch
# dimensions, and a fully-quantized group. One registered test-maker. Importing the module
# registers it in the global registry of zip_test_utils.
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
"""Test configs for gather."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_gather_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines four parameter groups (numeric
  # types, string type, float with constant indices and full quantization, and a batch-dims
  # group), and calls the zip generator expecting no reference-framework failures. Returns
  # nothing.
  """Make a set of tests to do gather."""

  test_parameters = [
      {
          "params_dtype": [tf.float32, tf.int32, tf.int64, tf.int16],
          "params_shape": [[1, 2, 20]],
          "indices_dtype": [tf.int32, tf.int64],
          "indices_shape": [[3], [5]],
          "axis": [-1, 0, 1],
          "batch_dims": [0],
          "constant_params": [False, True],
      },
      {
          "params_dtype": [tf.string],
          "params_shape": [[8]],
          "indices_dtype": [tf.int32],
          "indices_shape": [[3], [3, 2]],
          "axis": [0],
          "batch_dims": [0],
          "constant_params": [False, True],
      },
      {
          "params_dtype": [tf.float32],
          "params_shape": [[1, 2, 20]],
          "indices_dtype": [tf.int32, tf.int64],
          "indices_shape": [[3], [5]],
          "axis": [-1, 0, 1],
          "batch_dims": [0],
          "constant_params": [False],
          # Fix the indice values to prevent representative dataset generator
          # from generating invalid values.
          "constant_indices": [True],
          "fully_quantize": [True],
          "input_range": [(-10, 10)],
      },
      {
          # Test with batch_dims.
          "params_dtype": [tf.float32, tf.int32],
          "params_shape": [[2, 2, 3, 5]],
          "indices_dtype": [tf.int32],
          "indices_shape": [[2, 2, 2]],
          "axis": [0, 2],
          "batch_dims": [1, 2],
          "constant_params": [False, True],
      }
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: indices are bounded by the size of the first dimension of the parameters, not
    # of the gathered axis, so they can be out of range for axes other than zero; the axis is also
    # clamped to the parameter rank rather than rank minus one.
    # Builds the params as random constant data or a named placeholder, and the indices as random
    # constant data (when the constant-indices flag is present and set) or a named placeholder.
    # Applies gather with the clamped axis and the batch dims. Returns a pair: list of the
    # placeholder inputs actually created, and list with the output.
    """Build the gather op testing graph."""
    inputs = []

    if parameters["constant_params"]:
      params = create_tensor_data(parameters["params_dtype"],
                                  parameters["params_shape"])
    else:
      params = tf.compat.v1.placeholder(
          dtype=parameters["params_dtype"],
          name="params",
          shape=parameters["params_shape"])
      inputs.append(params)

    if parameters.get("constant_indices", False):
      indices = create_tensor_data(
          parameters["indices_dtype"],
          parameters["indices_shape"],
          min_value=0,
          max_value=parameters["params_shape"][0] - 1)
    else:
      indices = tf.compat.v1.placeholder(
          dtype=parameters["indices_dtype"],
          name="indices",
          shape=parameters["indices_shape"])
      inputs.append(indices)

    axis = min(len(parameters["params_shape"]), parameters["axis"])
    out = tf.gather(
        params, indices, axis=axis, batch_dims=parameters["batch_dims"])
    return inputs, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates data only for the inputs that are placeholders: params (range from the optional
    # input-range parameter, default minus ten to ten) and indices (zero up to the first
    # dimension size minus one). Evaluates the graph and returns a pair: list of input values and
    # list of expected outputs.
    input_values = []
    min_value, max_value = parameters.get("input_range", (-10, 10))
    if not parameters["constant_params"]:
      params = create_tensor_data(
          parameters["params_dtype"],
          parameters["params_shape"],
          min_value=min_value,
          max_value=max_value)
      input_values.append(params)
    if not parameters.get("constant_indices", False):
      indices = create_tensor_data(
          parameters["indices_dtype"],
          parameters["indices_shape"],
          min_value=0,
          max_value=parameters["params_shape"][0] - 1)
      input_values.append(indices)
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=0)
