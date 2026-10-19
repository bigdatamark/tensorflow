# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the hyperbolic tangent
# activation. Defines and registers one generator function that creates parameterized graphs and
# passes them to the shared zip-test machinery. Import-time side effect: registers the generator
# in the global registry of test-making functions.
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
"""Test configs for tanh."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_tanh_tests(options):
  # [FUNCTION DOC]
  # Generates the tanh test set. The options argument is the test-run configuration passed to
  # the zip-test builder. Parameters sweep float32 input shapes from scalar up to six
  # dimensions, a fully-quantize flag, and a single input value range from -4 to 10. Returns
  # nothing; output is produced through the builder.
  """Make a set of tests to do tanh."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                      [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
      "fully_quantize": [True, False],
      "input_range": [(-4, 10)]
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: a float32 placeholder named input of the
    # given shape followed by the hyperbolic tangent. Returns a pair: list with the input
    # placeholder and list with the output.
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.nn.tanh(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random float32 data of the parameter shape
    # drawn within the parameter input range (minimum and maximum) is fed into the graph in the
    # given session. Returns a pair: list with the input array and the evaluated outputs.
    min_value, max_value = parameters["input_range"]
    input_values = create_tensor_data(np.float32, parameters["input_shape"],
                                      min_value, max_value)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
