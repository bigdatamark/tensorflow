# [MODULE DOC]
# Test configs for local response normalization. Registers make_local_response_norm_tests.
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
"""Test configs for local_response_norm."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_local_response_norm_tests(options):
  # [FUNCTION DOC]
  # Generator for local response normalization tests. Input: options, test options object.
  # Parameter combinations of shapes, depth radius, bias, alpha and beta, each possibly null
  # (meaning TensorFlow defaults are not overridden by a real value). Calls the zip builder.
  # Returns nothing.
  """Make a set of tests to do local_response_norm."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3]],
      "depth_radius": [None, 0, 1, 3, 5],
      "bias": [None, 0.3, -0.1],
      "alpha": [None, 2, -3],
      "beta": [None, 0.25, 2],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph: float32 placeholder followed by local response normalization with the
    # chosen parameters. Returns input placeholder list and output list.
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.nn.local_response_normalization(
        input_tensor,
        depth_radius=parameters["depth_radius"],
        bias=parameters["bias"],
        alpha=parameters["alpha"],
        beta=parameters["beta"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Creates random float32 data between -4 and 10, runs the session for expected outputs.
    # Returns input values list and computed outputs.
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-4, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
