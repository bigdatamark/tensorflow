# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering depth-to-space rearrangement. It
# defines one registered test-maker with integer dtypes (never fully quantized) and float32
# (with and without full quantization), block sizes two and four. Importing the module registers
# the test-maker in the global registry of zip_test_utils.
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
"""Test configs for depth_to_space."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_depth_to_space_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines two parameter groups (integer
  # dtypes without quantization; float32 with quantization on or off) over a fixed four-
  # dimensional shape and two block sizes, and calls the zip generator. Returns nothing.
  """Make a set of tests to do depth_to_space."""

  test_parameters = [{
      "dtype": [tf.int32, tf.uint8, tf.int64],
      "input_shape": [[2, 3, 4, 16]],
      "block_size": [2, 4],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[2, 3, 4, 16]],
      "block_size": [2, 4],
      "fully_quantize": [True, False],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates a named placeholder with the parameterized dtype and shape and applies the
    # depth-to-space op with the parameterized block size. Returns a pair: list of input tensors
    # and list of output tensors.
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.compat.v1.depth_to_space(
        input_tensor, block_size=parameters["block_size"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random input data: default range normally, or restricted to minus one to one when
    # full quantization is requested. Runs the graph for reference outputs and returns a pair:
    # list with the input values and list of expected outputs.
    if not parameters["fully_quantize"]:
      input_values = create_tensor_data(parameters["dtype"],
                                        parameters["input_shape"])
    else:
      input_values = create_tensor_data(
          parameters["dtype"],
          parameters["input_shape"],
          min_value=-1,
          max_value=1)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
