# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of the bitcast op. Defines one
# registered test-maker covering several input/output dtype pairs (integers, unsigned integers,
# float32) where element widths differ, so shapes gain or lose a trailing dimension. Importing
# registers the maker in the global registry.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for bitcast."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_bitcast_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Declares parameter sets of input and
  # output dtype pairs with shapes; when converting from a narrower to a wider type the last
  # dimension is 2 so it is consumed. Calls the zip-test generator with the local builders.
  # Returns nothing.
  """Generate examples for bitcast."""
  test_parameters = [
      {
          "input_dtype": [tf.int32],
          "output_dtype": [tf.uint32],
          "input_shape": [[], [1], [1, 2], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.uint32],
          "output_dtype": [tf.int32],
          "input_shape": [[], [1], [1, 2], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.uint32],
          "output_dtype": [tf.int16],
          "input_shape": [[], [1], [1, 2], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int16],
          "output_dtype": [tf.uint32],
          "input_shape": [[2], [1, 2], [1, 2, 2], [3, 4, 5, 6, 2]],
      },
      {
          "input_dtype": [tf.int32],
          "output_dtype": [tf.int16],
          "input_shape": [[], [1], [1, 2], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.uint16],
          "output_dtype": [tf.uint32],
          "input_shape": [[2], [1, 2], [1, 2, 2], [3, 4, 5, 6, 2]],
      },
      {
          "input_dtype": [tf.float32],
          "output_dtype": [tf.int16],
          "input_shape": [[], [1], [1, 2], [3, 4, 5, 6]],
      },
      {
          "input_dtype": [tf.int16],
          "output_dtype": [tf.float32],
          "input_shape": [[2], [1, 2], [1, 2, 2], [3, 4, 5, 6, 2]],
      }
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the bitcast zip tests.
    # Creates a placeholder named input with the parameterized dtype and shape and reinterprets it
    # as the output dtype. Returns the list with the placeholder and the list with the output.
    """Build the bitcast testing graph."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"],
    )
    out = tf.bitcast(input_value, parameters["output_dtype"])
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the bitcast zip tests.
    # Generates random data of the input dtype and shape, runs the session with it fed to the
    # placeholder, and returns the list of input values and the expected outputs.
    input_value = create_tensor_data(
        parameters["input_dtype"], parameters["input_shape"]
    )
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value]))
    )

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
