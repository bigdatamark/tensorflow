# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of the two-argument arctangent (atan2)
# op. Defines one registered test-maker that builds graphs with two same-shaped inputs (y and x)
# in float32 or float64 and passes them to the shared zip-test generator. Importing registers the
# maker in the global registry.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for atan2."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_atan2_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Declares one parameter set (float32
  # and float64 dtypes; scalar and several ranked shapes) and calls the zip-test generator with the
  # local graph and input builders. Returns nothing; the generator writes the test artifacts.
  """Make a set of tests to do atan2."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.float64],
      "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]]
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the atan2 zip tests.
    # Creates two placeholders named y and x with the parameterized dtype and shape and applies the
    # atan2 op. Returns the list of both placeholders and the list with the single output tensor.
    """Build the atan2 op testing graph."""
    input_value1 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="y",
        shape=parameters["input_shape"])
    input_value2 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="x",
        shape=parameters["input_shape"])
    out = tf.math.atan2(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the atan2 zip tests.
    # Generates two random tensors of the parameterized dtype and shape, feeds them to the two
    # placeholders, runs the session for the expected outputs, and returns the list of the two input
    # values with the output values.
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape"])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape"])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
