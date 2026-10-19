# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of graphs containing control
# dependencies (an assert op that a convolution depends on). Defines a module constant for the
# convolution filter input depth and one registered test-maker. Importing registers the maker in
# the global registry.
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
"""Test configs for control_dep."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

TEST_INPUT_DEPTH = 3


@register_make_test_function()
def make_control_dep_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Declares one parameter set with four
  # float input shapes; only the last has a channel count matching the filter depth, so the other
  # three are expected TensorFlow failures (three passed to the zip-test generator). Returns
  # nothing.
  """Make a set of tests that use control dependencies."""

  test_parameters = [{
      "input_shape": [[], [1, 1, 1, 1], [1, 15, 14, 1], [3, 15, 14, 3]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the control-dependency zip tests.
    # Creates a float32 placeholder named input, a zero filter of three by three by the module depth
    # by eight, and an assert that the input is at least the input minus one (always true). Under a
    # control dependency on that assert, applies a stride-one same-padded 2-D convolution. Returns the
    # list with the placeholder and the list with the output tensor.
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    filter_value = tf.zeros((3, 3, TEST_INPUT_DEPTH, 8), tf.float32)
    assert_op = tf.compat.v1.assert_greater_equal(input_tensor,
                                                  input_tensor - 1)
    with tf.control_dependencies([assert_op]):
      out = tf.nn.conv2d(
          input=input_tensor,
          filters=filter_value,
          strides=(1, 1, 1, 1),
          padding="SAME")
      return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the control-dependency zip tests.
    # Generates random float32 data of the parameterized shape, runs the session with it, and returns
    # the list with the data and the expected outputs.
    input_values = create_tensor_data(tf.float32, parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=3)
