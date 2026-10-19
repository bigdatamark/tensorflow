# [MODULE DOC]
# Test configs for the is_finite op. Registers one generator, make_is_finite_tests.
# Import-time side effect: registration of the generator.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for is_finite."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_is_finite_tests(options):
  # [FUNCTION DOC]
  # Generator for is_finite tests. Input: options, the test options object. Uses float32 inputs
  # of a vector shape and a four-dimensional shape and calls the zip builder. Returns nothing.
  """Make a set of tests to do is_finite."""

  test_parameters = [
      {
          "input_shape": [[100], [3, 15, 14, 3]],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph: a float32 placeholder followed by the is-finite op.
    # Input: parameters mapping. Returns the list of input placeholders and the list of outputs.
    """Build the graph for the test case."""

    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.math.is_finite(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Creates random float32 data between -10 and 10 and overwrites five random positions with
    # positive infinity, negative infinity, not-a-number, the largest and the smallest float32.
    # Runs the session for expected outputs. Returns list of inputs and computed outputs.
    # Note: positions may coincide, so fewer than five distinct special values can end up present.
    """Build the inputs for the test case."""
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-10, max_value=10)

    # Inject NaN and Inf value.
    def random_index(shape):
      # [FUNCTION DOC]
      # HELPER: picks special-value positions inside build_inputs.
      # Input: shape, a sequence of dimension sizes. Returns a tuple with one random index per
      # dimension, each between zero and that dimension size minus one.
      result = []
      for dim in shape:
        result.append(np.random.randint(low=0, high=dim))
      return tuple(result)

    input_values[random_index(input_values.shape)] = np.inf
    input_values[random_index(input_values.shape)] = -np.inf
    input_values[random_index(input_values.shape)] = np.nan
    input_values[random_index(input_values.shape)] = tf.float32.max
    input_values[random_index(input_values.shape)] = tf.float32.min

    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
