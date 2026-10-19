# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering the elementwise exponential. One
# registered test-maker covers float inputs over several ranks, with a wide input range and a
# narrow range with full (8-bit and 16x8) quantization. Importing the module registers it in the
# global registry of zip_test_utils.
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
"""Test configs for exp."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_exp_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines two parameter groups (range
  # minus 100 to 9 without quantization; range minus 2 to 2 with full quantization, 8-bit or
  # 16x8) over scalar to 4-D float shapes, and calls the zip generator. Returns nothing.
  """Make a set of tests to do exp."""

  test_parameters = [
      {
          "input_dtype": [tf.float32],
          "input_shape": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3]],
          "input_range": [(-100, 9)],
      },
      {
          "input_dtype": [tf.float32],
          "input_shape": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3]],
          "input_range": [(-2, 2)],
          "fully_quantize": [True],
          "quant_16x8": [False, True],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates a named placeholder of the parameterized dtype and shape and applies exponential.
    # Returns a pair: list with the input tensor and list with the output tensor.
    """Build the exp op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])

    out = tf.exp(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Unpacks the input range from the parameters, generates random data in that range, evaluates
    # the graph, and returns a pair: list with the input value and list of expected outputs.
    min_value, max_value = parameters["input_range"]
    values = [
        create_tensor_data(
            parameters["input_dtype"],
            parameters["input_shape"],
            min_value=min_value,
            max_value=max_value,
        )
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
