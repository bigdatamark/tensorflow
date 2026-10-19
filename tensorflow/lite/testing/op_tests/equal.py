# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering elementwise equality comparison
# with broadcasting. One registered test-maker covers float, integer and string types plus a
# fully-quantized float group. Importing the module registers it in the global registry of
# zip_test_utils.
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
"""Test configs for equal."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_equal_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines two parameter groups (several
  # dtypes with pairs of shapes, some needing broadcasting; and float32 fully quantized) and
  # calls the zip generator, declaring that six combinations are expected to fail in the
  # reference framework. Returns nothing.
  """Make a set of tests to do equal."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int16, tf.int32, tf.int64, tf.string],
      "input_shape_pair": [([], []), ([1, 1, 1, 3], [1, 1, 1, 3]),
                           ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                           ([5, 5], [1]), ([10], [2, 4, 10])],
      "fully_quantize": [False],
  }, {
      "input_dtype": [tf.float32],
      "input_shape_pair": [([1, 1, 1, 3], [1, 1, 1, 3]), ([2, 3, 3], [2, 3])],
      "fully_quantize": [True],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates two named placeholders of the parameterized dtype with the two shapes of the pair
    # and compares them for equality. Returns a pair: list of both input tensors and list with
    # the boolean output.
    """Build the equal op testing graph."""
    input_value1 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape_pair"][0])
    input_value2 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_pair"][1])
    out = tf.equal(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random data for both operands with their shapes, evaluates the graph, and returns
    # a pair: list of the two input values and list of expected outputs.
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=6)
