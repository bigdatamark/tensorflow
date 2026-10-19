# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering batched matrix
# multiplication that the converter is expected to unroll into fully connected operations.
# Defines and registers one generator function that creates parameterized graphs and passes them
# to the shared zip-test machinery. Import-time side effect: registers the generator in the
# global registry of test-making functions.
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
"""Test configs for unroll_batch_matmul."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_unroll_batch_matmul_tests(options):
  # [FUNCTION DOC]
  # KNOWN ISSUE: an in-code comment says the broadcasting shape cases are not supported, yet
  # those cases are appended to the tested shape list anyway.
  # Generates the batched matmul test set. The options argument is the test-run configuration
  # passed to the zip-test builder. Each shape entry is a list of: first operand shape, second
  # operand shape, transpose-first flag, transpose-second flag. Entries cover 3-D and 4-D
  # equal-batch cases with all transpose combinations, plus a few broadcasting cases (different
  # batch ranks or size-one batch dimensions). Type is float32. Returns nothing.
  """Make a set of tests to test unroll_batch_matmul."""

  # The test cases below requires broadcasting support (BatchMatMulV2 semantic),
  # whis isn't supported as of this change.
  broadcast_shape_params = [
      # Simple broadcast.
      [(1, 2, 3), (3, 5), False, False],
      # Empty batch broadcast.
      [(2, 5, 3), (3, 7), False, False],
      # Single batch with non-empty batch broadcast.
      [(1, 5, 3), (4, 3, 7), False, False],
      # Broadcast both operands
      [(3, 1, 5, 3), (1, 4, 3, 7), False, False],
  ]

  test_parameters = [{
      "dtype": [tf.float32],
      "shape": [[(2, 2, 3),
                 (2, 3, 2), False, False], [(2, 2, 3), (2, 3, 2), True, True],
                [(2, 2, 3),
                 (2, 2, 3), False, True], [(2, 2, 3), (2, 2, 3), True, False],
                [(4, 2, 2, 3), (4, 2, 3, 2), False, False],
                [(4, 2, 2, 3), (4, 2, 3, 2), True, True],
                [(4, 2, 2, 3), (4, 2, 2, 3), False, True],
                [(4, 2, 2, 3),
                 (4, 2, 2, 3), True, False]] + broadcast_shape_params,
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination by delegating to an inner function and
    # returning its result: a pair of the list of two input placeholders and the list with the
    # matrix multiplication output.
    """Build the batch_matmul op testing graph."""

    def _build_graph():
      # [FUNCTION DOC]
      # HELPER: inner graph constructor for the batch matmul graph builder.
      # Creates two placeholders of the parameter type with the two operand shapes, and applies a
      # matrix multiplication with the transpose flags from the shape entry. Returns a pair: list
      # with both placeholders and list with the output.
      """Build the graph."""
      input_tensor1 = tf.compat.v1.placeholder(
          dtype=parameters["dtype"], shape=parameters["shape"][0])
      input_tensor2 = tf.compat.v1.placeholder(
          dtype=parameters["dtype"], shape=parameters["shape"][1])
      # Should be unrolled and replaced with fully_connected ops in the end.
      out = tf.matmul(
          input_tensor1,
          input_tensor2,
          transpose_a=parameters["shape"][2],
          transpose_b=parameters["shape"][3])
      return [input_tensor1, input_tensor2], [out]

    return _build_graph()

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random data for each operand using the operand
    # shapes and the parameter type, fed into the graph in the given session. Returns a pair:
    # list of the two input arrays and the evaluated outputs.
    input_value1 = create_tensor_data(
        parameters["dtype"], shape=parameters["shape"][0])
    input_value2 = create_tensor_data(
        parameters["dtype"], shape=parameters["shape"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
