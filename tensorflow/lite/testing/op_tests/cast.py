# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of the cast op. Defines one registered
# test-maker listing many (input dtype, output dtype) conversions among integer, unsigned, float16,
# bfloat16, float32 and float64 types over scalar and ranked shapes. Importing registers the maker
# in the global registry.
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
"""Test configs for cast."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_cast_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Declares one parameter set per
  # input/output dtype pair (20 pairs), each over five shapes, and calls the zip-test generator
  # with the local builders. Returns nothing.
  """Generate examples for cast."""
  test_parameters = [
      {
          "input_dtype": [tf.float32],
          "output_dtype": [tf.int16],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int16],
          "output_dtype": [tf.float32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int32],
          "output_dtype": [tf.float32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int8],
          "output_dtype": [tf.float32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.float32],
          "output_dtype": [tf.int8],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.uint16],
          "output_dtype": [tf.float32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.uint32],
          "output_dtype": [tf.int32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.uint8],
          "output_dtype": [tf.int8],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int8],
          "output_dtype": [tf.uint8],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.uint16],
          "output_dtype": [tf.int16],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int16],
          "output_dtype": [tf.uint16],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int32],
          "output_dtype": [tf.float64],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.float64],
          "output_dtype": [tf.int32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.float32],
          "output_dtype": [tf.float64],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.float64],
          "output_dtype": [tf.float32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int64],
          "output_dtype": [tf.float64],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.float64],
          "output_dtype": [tf.int64],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.float32],
          "output_dtype": [tf.float16],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.float16],
          "output_dtype": [tf.float32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.bfloat16],
          "output_dtype": [tf.float32],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.float32],
          "output_dtype": [tf.bfloat16],
          "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the cast zip tests.
    # Creates a placeholder named input with the input dtype and shape and casts it to the output
    # dtype. Returns the list with the placeholder and the list with the output tensor.
    """Build the cast testing graph."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.cast(input_value, parameters["output_dtype"])
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the cast zip tests.
    # Generates random data of the input dtype and shape, runs the session with it, and returns the
    # list of input values and the expected outputs.
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
