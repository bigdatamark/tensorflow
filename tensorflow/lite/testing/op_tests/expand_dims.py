# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering the insert-a-dimension (expand
# dims) op, with the axis given either as a constant or as a runtime input. One registered
# test-maker. Importing the module registers it in the global registry of zip_test_utils.
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
"""Test configs for expand_dims."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_expand_dims_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines two parameter groups (float32 or
  # int32 with constant or placeholder axis; float32 fully quantized with constant axis) over two
  # shapes and axes from minus three to two, and calls the zip generator. Returns nothing.
  """Make a set of tests to do expand_dims."""

  test_parameters = [{
      "input_type": [tf.float32, tf.int32],
      "input_shape": [[5, 4], [1, 5, 4]],
      "axis_value": [0, 1, 2, -1, -2, -3],
      "constant_axis": [True, False],
      "fully_quantize": [False],
  }, {
      "input_type": [tf.float32],
      "input_shape": [[5, 4], [1, 5, 4]],
      "axis_value": [0, 1, 2, -1, -2, -3],
      "constant_axis": [True],
      "fully_quantize": [True],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it builds the where op graph; it actually builds an
    # expand-dims graph.
    # Creates the data placeholder, and an int32 one-element axis that is either a constant with
    # the parameterized value or a second named placeholder. Inserts a dimension at that axis.
    # Returns a pair: list of placeholder inputs (one or two) and list with the output.
    """Build the where op testing graph."""
    inputs = []
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_type"],
        name="input",
        shape=parameters["input_shape"])
    inputs.append(input_value)

    if parameters["constant_axis"]:
      axis_value = tf.constant(
          parameters["axis_value"], dtype=tf.int32, shape=[1])
    else:
      axis_value = tf.compat.v1.placeholder(
          dtype=tf.int32, name="axis", shape=[1])
      inputs.append(axis_value)

    out = tf.expand_dims(input_value, axis=axis_value)
    return inputs, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random data between minus one and one for the data input; when the axis is not
    # constant, also supplies the axis value as a one-element int32 array. Evaluates the graph and
    # returns a pair: list of input values and list of expected outputs.
    """Builds the inputs for expand_dims."""
    input_values = []
    input_values.append(
        create_tensor_data(
            parameters["input_type"],
            parameters["input_shape"],
            min_value=-1,
            max_value=1))
    if not parameters["constant_axis"]:
      input_values.append(np.array([parameters["axis_value"]], dtype=np.int32))
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
