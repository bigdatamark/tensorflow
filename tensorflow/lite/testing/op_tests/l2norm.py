# [MODULE DOC]
# Test configs for the L2 normalization op. Registers make_l2norm_tests.
# Import-time side effect: registration of the generator.
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
"""Test configs for l2norm."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_l2norm_tests(options):
  # [FUNCTION DOC]
  # Generator for l2norm tests. Input: options, the test options object. Parameter groups: float
  # inputs with several shapes, axes (single, list, negative) and epsilons (none, 1e-12, 1e-3);
  # two fully quantized groups, one of which is a separate group so the normalization over the
  # last axis of a matrix fuses into the dedicated TFLite op. Calls the zip builder expecting nine
  # TensorFlow-side failures (unsupported combinations). Returns nothing.
  """Make a set of tests to do l2norm."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[5, 7], [1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3]],
      "dim": [0, 1, 2, 3, [2, 3], -2],
      "epsilon": [None, 1e-12, 1e-3],
      "fully_quantize": [False],
  }, {
      "input_shape": [[1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3]],
      "dim": [3],
      "epsilon": [None, 1e-12],
      "fully_quantize": [True],
  }, {  # use another group of test so the dim is set to fuse to tfl.l2norm
      "input_shape": [[5, 7]],
      "dim": [1],
      "epsilon": [None, 1e-12],
      "fully_quantize": [True],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph: float32 placeholder followed by L2 normalization over the chosen axis.
    # Epsilon is passed only if it is set (truthy); otherwise TensorFlow default is used.
    # Returns input placeholder list and output list.
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    if parameters["epsilon"]:
      out = tf.nn.l2_normalize(
          input_tensor, parameters["dim"], epsilon=parameters["epsilon"])
    else:
      out = tf.nn.l2_normalize(input_tensor, parameters["dim"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Creates random float32 data in -1 to 1, runs the session for the expected outputs.
    # Returns the list of input values and the computed outputs.
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-1, max_value=1)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=9)
