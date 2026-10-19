# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering fully-connected layers with 4-bit
# weights and dynamically quantized (hybrid) activations. One registered test-maker; the weights
# are constructed to be exactly representable in 4 bits and the inputs exactly representable in
# 8 bits, so results can be compared with the unquantized reference. Importing the module
# registers it in the global registry of zip_test_utils.
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
"""Test configs for fully_connected_4bit."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_fully_connected_4bit_hybrid_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object and defines three shape pairs (small,
  # one that uses the optimized kernel, one that does not), all with dynamic range quantization.
  # Side effect: sets the experimental low-bit quantization-aware-training option on the options
  # object. Calls the zip generator expecting no reference-framework failures. Returns nothing.
  """Make a set of tests to do fully_connected."""

  test_parameters = [
      # Simple 3x3 test.
      {
          "shape1": [[3, 3]],
          "shape2": [[3, 3]],
          "dynamic_range_quantize": [True],
      },
      # Use optimized kernel.
      {
          "shape1": [[40, 42]],
          "shape2": [[42, 40]],
          "dynamic_range_quantize": [True],
      },
      # No optimization.
      {
          "shape1": [[1, 40]],
          "shape2": [[40, 3]],
          "dynamic_range_quantize": [True],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates a float32 placeholder for the first operand. Builds the weights from random values
    # scaled and rounded to integers in the range minus seven to seven, stored as a float
    # constant, then passed through fake quantization with 4 bits and a narrow range of minus
    # seven to seven. Multiplies the input by the quantized weights. Uses unseeded global random
    # numbers. Returns a pair: list with the input tensor and list with the output.
    """Build a matmul graph given `parameters`."""
    input_tensor1 = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input1", shape=parameters["shape1"]
    )
    # Create a float filter with no quantization loss.
    float_data = np.random.uniform(-1, 1, parameters["shape2"])
    scale = np.abs(float_data).max() / 7.0
    int_data = np.round(float_data / scale)
    input_tensor2 = tf.constant(int_data, dtype=tf.float32)
    quantized = tf.quantization.fake_quant_with_min_max_vars(
        input_tensor2, min=-7, max=7, num_bits=4, narrow_range=True
    )
    out = tf.matmul(input_tensor1, quantized)
    return [input_tensor1], [out]

  def create_input_data(parameters):
    # [FUNCTION DOC]
    # HELPER: produces input data for the 4-bit hybrid test.
    # Generates random numbers in zero to one with the first operand shape, then divides each row
    # by its own maximum over 127 and rounds, giving integer values within the 8-bit range per
    # row so dynamic activation quantization loses nothing. Returns the resulting array.
    # Rows that are all zeros would divide by zero (practically impossible with random data).
    """Create a float input with no quantization loss."""
    float_data = np.random.random(parameters["shape1"]).astype(np.float32)
    scale = np.abs(float_data).max(axis=1, keepdims=True) / 127.0
    return np.round(float_data / scale)

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Uses the specialized input creator rather than the generic one, evaluates the graph in the
    # session, and returns a pair: list with the input values and list of expected outputs.
    # pylint: disable=g-doc-return-or-yield, g-doc-args
    """Build list of input values.

    Use the specialized method, as dynamic range quantization will cause
    differing outputs from TF, which does not quantize inputs.
    """
    values = [create_input_data(parameters)]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  options.experimental_low_bit_qat = True
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=0,
  )
