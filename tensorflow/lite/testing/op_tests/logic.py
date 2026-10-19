# [MODULE DOC]
# Test configs for the logical binary operators (logical or, and, xor). A private factory builds
# a generator for any given two-input boolean TensorFlow op, and three registered generators
# (make_logical_or_tests, make_logical_and_tests, make_logical_xor_tests) wrap it. Import-time
# side effect: registration of those three generators.
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
"""Test configs for logic operators."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def _make_logical_tests(op):
  # [FUNCTION DOC]
  # HELPER: factory used by the three registered logical-op generators in this module.
  # Input: op, a two-input boolean TensorFlow op. Returns a generator function (taking options and
  # an expected-failure count) that produces the test zip for that op. Creates no tests itself.
  """Make a set of tests to do logical operations."""

  def logical(options, expected_tf_failures=0):
    # [FUNCTION DOC]
    # Inner generator returned by the factory. Inputs: options (test options object) and
    # expected_tf_failures (number of TensorFlow-side failures tolerated, default zero).
    # Parameter set: boolean inputs with scalar, same-shape and broadcasting shape pairs. Calls the
    # zip builder. Returns nothing.
    """Generate examples."""
    test_parameters = [{
        "input_shape_pair": [([], []), ([1, 1, 1, 3], [1, 1, 1, 3]),
                             ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                             ([5, 5], [1]), ([10], [2, 4, 10])],
    }]

    def build_graph(parameters):
      # [FUNCTION DOC]
      # Builds the graph: two boolean placeholders and the captured logical op applied to them.
      # Returns the two placeholders and the output list.
      """Build the logical testing graph."""
      input_value1 = tf.compat.v1.placeholder(
          dtype=tf.bool, name="input1", shape=parameters["input_shape_pair"][0])
      input_value2 = tf.compat.v1.placeholder(
          dtype=tf.bool, name="input2", shape=parameters["input_shape_pair"][1])
      out = op(input_value1, input_value2)
      return [input_value1, input_value2], [out]

    def build_inputs(parameters, sess, inputs, outputs):
      # [FUNCTION DOC]
      # Creates random boolean data for both inputs, runs the session for expected outputs.
      # Returns the two input values and the computed outputs.
      input_value1 = create_tensor_data(tf.bool,
                                        parameters["input_shape_pair"][0])
      input_value2 = create_tensor_data(tf.bool,
                                        parameters["input_shape_pair"][1])
      return [input_value1, input_value2], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

    make_zip_of_tests(
        options,
        test_parameters,
        build_graph,
        build_inputs,
        expected_tf_failures=expected_tf_failures)

  return logical


@register_make_test_function()
def make_logical_or_tests(options):
  # [FUNCTION DOC]
  # Registered generator for logical_or tests. Input: options. Runs the factory-built generator
  # expecting one TensorFlow-side failure. Returns nothing.
  """Make a set of tests to do logical_or."""
  return _make_logical_tests(tf.logical_or)(options, expected_tf_failures=1)


@register_make_test_function()
def make_logical_and_tests(options):
  # [FUNCTION DOC]
  # Registered generator for logical_and tests. Input: options. Runs the factory-built generator
  # expecting one TensorFlow-side failure. Returns nothing.
  """Make a set of tests to do logical_and."""
  return _make_logical_tests(tf.logical_and)(options, expected_tf_failures=1)


@register_make_test_function()
def make_logical_xor_tests(options):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says logical_not is tested as well, but only logical xor is used.
  # Registered generator for logical_xor tests. Input: options. Runs the factory-built generator
  # expecting one TensorFlow-side failure. Returns nothing.
  """Make a set of tests to do logical_xor, test logical_not as well."""
  return _make_logical_tests(tf.math.logical_xor)(
      options, expected_tf_failures=1)
