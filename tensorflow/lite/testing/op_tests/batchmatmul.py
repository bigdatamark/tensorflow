# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of batch matrix multiplication.
# Defines one registered test-maker (under the explicit name make_batchmatmul_tests) covering
# batch and broadcast shapes with unknown dimensions, optional adjoint of either operand, and
# optional full quantization. Importing registers the maker in the global registry.
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
"""Test configs for batchmatmul."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function("make_batchmatmul_tests")
def make_batchmatmul_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Each shapes entry holds four shapes:
  # the two placeholder shapes (possibly with unknown dimensions) and the two concrete shapes used to
  # generate data. Sets the options flag that disables unfolding of batch matmul (a side effect on
  # the passed options object), then calls the zip-test generator expecting zero TensorFlow
  # failures. Returns nothing.
  """Make a set of tests to do basic batch matrix multiply."""

  test_parameters = [
      {
          "dtype": [tf.float32],
          "shapes": [((3, 4, 7), (7, 9), (3, 4, 7), (7, 9)),
                     ((None, 4, 5), (None, 5, 6), (3, 4, 5), (3, 5, 6)),
                     ((None, 1, 3, 4), (None, 4, 2), (2, 1, 3, 4), (5, 4, 2)),
                     ((None, None, None, 3, 4), (None, None, None, 4, 3),
                      (2, 2, 2, 3, 4), (2, 2, 2, 4, 3))],
          "adjoint_b": [False, True],
          "adjoint_a": [False, True],
          "rhs_constant": [False],
          "fully_quantize": [False, True],
      },
  ]

  def swap_last_two_dims(*args):
    # [FUNCTION DOC]
    # HELPER: shape utility for the batch matmul tests, used to model adjoint operands.
    # Takes the dimensions of a shape as separate positional arguments and returns a tuple of the
    # same dimensions with the final two exchanged. Needs at least two dimensions.
    """Return a tuple with the last two dimensions swapped."""
    return args[:-2] + (args[-1],) + (args[-2],)

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the batch matmul zip tests.
    # Uses the first shape for the left placeholder (last two dims swapped when adjoint_a is set).
    # If rhs_constant is true, the right operand is a constant of random data in minus one to one
    # using the fourth shape (swapped when adjoint_b), and only the left placeholder is an input;
    # otherwise the right operand is a placeholder using the second shape (swapped for adjoint_b).
    # Applies matmul with the adjoint flags. Returns the list of input placeholders and the list
    # with the output tensor.
    """Build a simple graph with BatchMatMul."""
    placeholder0_shape = parameters["shapes"][0]
    adj_a = parameters["adjoint_a"]
    adj_b = parameters["adjoint_b"]
    rhs_constant = parameters["rhs_constant"]
    if adj_a:
      placeholder0_shape = swap_last_two_dims(*placeholder0_shape)
    input0_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], shape=placeholder0_shape)
    if rhs_constant:
      if adj_b:
        constant1_shape = swap_last_two_dims(*parameters["shapes"][3])
      else:
        constant1_shape = parameters["shapes"][3]
      data = create_tensor_data(
          parameters["dtype"], constant1_shape, min_value=-1.0, max_value=1.0)
      input1_constant = tf.constant(
          data, shape=constant1_shape, dtype=parameters["dtype"])
      out = tf.matmul(
          input0_tensor, input1_constant, adjoint_a=adj_a, adjoint_b=adj_b)
      return [input0_tensor], [out]
    else:
      if adj_b:
        placeholder1_shape = swap_last_two_dims(*parameters["shapes"][1])
      else:
        placeholder1_shape = parameters["shapes"][1]
      input1_tensor = tf.compat.v1.placeholder(
          dtype=parameters["dtype"], shape=placeholder1_shape)
      out = tf.matmul(
          input0_tensor, input1_tensor, adjoint_a=adj_a, adjoint_b=adj_b)
      return [input0_tensor, input1_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the batch matmul zip tests.
    # Generates random data in minus one to one for the left operand using the third (concrete)
    # shape, swapped for adjoint_a. When the right side is non-constant, also generates data using
    # the fourth shape (swapped for adjoint_b). Runs the session and returns the list of input
    # values and the expected outputs. The docstring's mention of assigning variables and freezing
    # the graph is not reflected here; no variables are assigned.
    """Feed inputs, assign variables, and freeze graph."""
    input0_shape = parameters["shapes"][2]
    adj_a = parameters["adjoint_a"]
    adj_b = parameters["adjoint_b"]
    rhs_constant = parameters["rhs_constant"]
    if adj_a:
      input0_shape = swap_last_two_dims(*input0_shape)
    input0_value = create_tensor_data(
        parameters["dtype"], input0_shape, min_value=-1.0, max_value=1.0)
    if rhs_constant:
      output_values = sess.run(
          outputs, feed_dict=dict(zip(inputs, [input0_value])))
      return [input0_value], output_values
    else:
      input1_shape = parameters["shapes"][
          3] if not adj_b else swap_last_two_dims(*parameters["shapes"][3])
      input1_value = create_tensor_data(
          parameters["dtype"], input1_shape, min_value=-1.0, max_value=1.0)
      output_values = sess.run(
          outputs, feed_dict=dict(zip(inputs, [input0_value, input1_value])))
      return [input0_value, input1_value], output_values

  options.disable_batchmatmul_unfold = True
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=0)
