# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of the broadcast-to op. Defines one
# registered test-maker (under the explicit name make_broadcast_to_tests) covering varied input and
# output shapes, up to eight dimensions, and a zero-sized output case. Importing registers the
# maker in the global registry.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for broadcast_to."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function("make_broadcast_to_tests")
def make_broadcast_to_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Declares parameter sets (float32 and
  # int32) crossing input shapes with output shapes (many combinations are invalid broadcasts and
  # are counted as expected TensorFlow failures, 16 in total), plus eight-dimensional identity,
  # expanding and zero-sized output cases. Calls the zip-test generator. Returns nothing.
  """Make a set of tests to do broadcast_to."""

  # Chose a set of parameters
  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 2], [2, 3, 4], [1], [2, 5, 2, 3, 4]],
      "output_shape": [[3, 1, 2], [5, 2, 3, 4], [10, 10],
                       [1, 2, 1, 2, 5, 2, 3, 4]],
  }, {
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[3, 2, 3, 4, 5, 6, 7, 8]],
      "output_shape": [[3, 2, 3, 4, 5, 6, 7, 8]],
  }, {
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 3, 1, 2, 1, 4, 1, 1]],
      "output_shape": [[2, 3, 1, 2, 2, 4, 1, 1]],
  }, {
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[2, 1, 1, 2, 1, 4, 1, 1]],
      "output_shape": [[2, 3, 2, 2, 2, 4, 1, 1]],
  }, {
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[3, 4, 1]],
      "output_shape": [[3, 4, 0]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the broadcast-to zip tests.
    # KNOWN ISSUE: the docstring says "cond tests" (copy-paste); the graph is a broadcast-to op.
    # Creates a placeholder named input with the parameterized dtype and shape and broadcasts it to
    # the constant output shape. Returns the list with the placeholder and the list with the output.
    """Build the graph for cond tests."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])

    out = tf.broadcast_to(input_tensor, shape=parameters["output_shape"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the broadcast-to zip tests.
    # Generates random data of the input dtype and shape, runs the session with it, and returns the
    # list of input values and the expected outputs.
    input_values = [
        create_tensor_data(parameters["input_dtype"], parameters["input_shape"])
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=16)
