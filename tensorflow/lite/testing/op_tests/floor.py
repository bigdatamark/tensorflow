# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering the elementwise floor op on float
# inputs of several ranks. One registered test-maker. Importing the module registers it in the
# global registry of zip_test_utils.
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
"""Test configs for floor."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_floor_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines a grid of float32 inputs with
  # shapes from scalar to 4-D, and calls the zip generator. Returns nothing.
  """Make a set of tests to do floor."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates a named placeholder of the parameterized dtype and shape and applies floor. Returns
    # a pair: list with the input tensor and list with the output tensor.
    """Build the floor op testing graph."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape"])
    out = tf.floor(input_value)
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random data, evaluates the graph in the session, and returns a pair: list with the
    # input value and list of expected outputs.
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(outputs, feed_dict={inputs[0]: input_value})

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
