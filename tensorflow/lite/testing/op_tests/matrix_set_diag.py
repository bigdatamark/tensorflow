# [MODULE DOC]
# Test configs for the matrix set_diag op. Registers make_matrix_set_diag_tests.
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
"""Test configs for matrix_set_diag."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_matrix_set_diag_tests(options):
  # [FUNCTION DOC]
  # Generator for set_diag tests. Input: options, test options object. Parameters: pairs of matrix
  # shape and diagonal shape with int32, float32 or uint8 data. Calls the zip builder.
  # Returns nothing.
  """Make a set of tests for tf.linalg.set_diag op."""

  test_parameters = [
      {
          "input_diag_shapes": [([3, 3], [3]), ([2, 3], [2]), ([2, 4,
                                                                4], [2, 4]),
                                ([3, 4, 5, 6], [3, 4, 5])],
          "input_dtype": [tf.int32, tf.float32, tf.uint8],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph: placeholders for the matrix and the new diagonal, and the set_diag op.
    # Returns the two placeholders and the output list.
    input_shape = parameters["input_diag_shapes"][0]
    diag_shape = parameters["input_diag_shapes"][1]
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"], name="input", shape=input_shape)
    diag_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"], name="diagonal", shape=diag_shape)
    outs = tf.linalg.set_diag(input_tensor, diag_tensor)
    return [input_tensor, diag_tensor], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Creates random data for the matrix and diagonal and runs the session for expected outputs.
    # Returns the two input values and the computed outputs.
    input_shape = parameters["input_diag_shapes"][0]
    diag_shape = parameters["input_diag_shapes"][1]
    input_values = create_tensor_data(parameters["input_dtype"], input_shape)
    diag_values = create_tensor_data(parameters["input_dtype"], diag_shape)
    return [input_values, diag_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values, diag_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
