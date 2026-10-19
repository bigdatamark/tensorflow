# [MODULE DOC]
# Test configuration for the TFLite AddN op (sum of several tensors). Defines and registers the
# test builder make_add_n_tests used by the example generator. Registration happens at import time.
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
"""Test configs for add_n."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_add_n_tests(options):
  # [FUNCTION DOC]
  # Test builder for the AddN op, registered with the example generator. Takes the generation
  # options and returns nothing; writes test cases via the zip helper. Parameter sets: float32
  # and int32 with 2 to 5 inputs for shapes of rank 4, 1 and 0, plus a dynamic range quantized
  # float32 scalar case.
  """Make a set of tests for AddN op."""

  test_parameters = [
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape": [[2, 5, 3, 1]],
          "num_inputs": [2, 3, 4, 5],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape": [[5]],
          "num_inputs": [2, 3, 4, 5],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape": [[]],
          "num_inputs": [2, 3, 4, 5],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[]],
          "num_inputs": [2, 3, 4, 5],
          "dynamic_range_quantize": [True],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph callback for make_add_n_tests. Creates the parameterized number of
    # placeholders named input_i with the given dtype and shape, sums them with the AddN op, and
    # returns the list of inputs and a list holding the output.
    """Builds the graph given the current parameters."""
    input_tensors = []
    for i in range(parameters["num_inputs"]):
      input_tensors.append(
          tf.compat.v1.placeholder(
              dtype=parameters["dtype"],
              name="input_{}".format(i),
              shape=parameters["input_shape"]))
    out = tf.add_n(input_tensors)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input callback for make_add_n_tests. Creates random data for each input with the
    # parameter dtype and shape, runs the session on it, and returns the list of input values and
    # the computed outputs.
    """Builds operand inputs for op."""
    input_data = []
    for _ in range(parameters["num_inputs"]):
      input_data.append(
          create_tensor_data(parameters["dtype"], parameters["input_shape"]))
    return input_data, sess.run(
        outputs, feed_dict={i: d for i, d in zip(inputs, input_data)})

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
