# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering the older global-normalization
# batch norm op (with the scale-after-normalization option), whose result is added to a runtime
# input. One registered test-maker. Importing the module registers it in the global registry of
# zip_test_utils.
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
"""Test configs for global_batch_norm."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_global_batch_norm_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines a grid over two input shapes,
  # two epsilon values and the scale-after flag (float32 only), and calls the zip generator.
  # Returns nothing.
  """Make a set of tests to do batch_norm_with_global_normalization."""

  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[1, 1, 6, 2], [3, 4, 5, 4]],
      "epsilon": [0.1, 0.0001],
      "scale_after": [True, False],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Takes the channel count from the last input dimension and creates random constant scale,
    # offset, mean, variance vectors and a random constant tensor x of the full input shape.
    # Applies global-normalization batch norm to x (a constant, so the normalization is folded
    # into a constant). Creates a placeholder and adds the normalized constant to it. Returns a
    # pair: list with the placeholder and list with the sum.
    """Build the global batch norm testing graph."""
    input_shape = parameters["input_shape"]
    scale_shape = input_shape[3]

    scale = create_tensor_data(parameters["dtype"], scale_shape)
    offset = create_tensor_data(parameters["dtype"], scale_shape)
    mean = create_tensor_data(parameters["dtype"], scale_shape)
    variance = create_tensor_data(parameters["dtype"], scale_shape)

    x = create_tensor_data(parameters["dtype"], parameters["input_shape"])
    x_norm = tf.nn.batch_norm_with_global_normalization(
        x, mean, variance, scale, offset, parameters["epsilon"],
        parameters["scale_after"])

    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.add(input_tensor, x_norm)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random data for the single placeholder, evaluates the graph, and returns a pair:
    # list with the input value and list of expected outputs.
    input_value = create_tensor_data(parameters["dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
