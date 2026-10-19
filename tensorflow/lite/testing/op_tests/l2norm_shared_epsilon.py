# [MODULE DOC]
# Regression test configs for a bug (b/122651451) where two L2 normalizations share the same
# epsilon constant. Registers make_l2norm_shared_epsilon_tests. Import-time side effect:
# registration of the generator.
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
"""Test configs for l2norm_shared_epsilon."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_l2norm_shared_epsilon_tests(options):
  # [FUNCTION DOC]
  # Generator for the shared-epsilon regression test. Input: options, test options object.
  # Single parameter set (shape of five by seven, axis one, epsilon 1e-8). Calls the zip builder.
  # Returns nothing.
  """Regression test for a bug (b/122651451)."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[5, 7]],
      "dim": [1],
      "epsilon": [1e-8],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph: one float32 placeholder, one constant epsilon tensor, two L2 normalizations
    # that both use that same epsilon tensor, and their sum as output. Returns input placeholder
    # list and output list.
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    epsilon = tf.constant(parameters["epsilon"])
    out1 = tf.nn.l2_normalize(input_tensor, parameters["dim"], epsilon=epsilon)
    out2 = tf.nn.l2_normalize(input_tensor, parameters["dim"], epsilon=epsilon)
    out = out1 + out2
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Creates random float32 data in -4 to 10, runs the session for expected output.
    # Returns list of input values and computed outputs.
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-4, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
