# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering simple unary elementwise float ops:
# sin, log, sqrt, reciprocal square root, and square. A private factory builds a test-maker for a
# given unary op, and five registered test-makers use it. Importing the module registers those
# five test-makers in the global registry of zip_test_utils.
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
"""Test configs for elementwise ops."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def _make_elementwise_tests(
    op,
    allow_fully_quantize=False,
    min_value=-100,
    max_value=100,
):
  # [FUNCTION DOC]
  # HELPER: factory shared by the sin, log, sqrt, rsqrt and square test-makers.
  # Takes the unary op (a callable from one tensor to one tensor), a flag saying whether the
  # fully-quantized parameter groups are kept (default no), and the minimum and maximum values
  # (defaults minus 100 and 100) of the generated input data. Returns a new function that takes
  # the test options object and generates the test archive; nothing else is done at factory time.
  """Make a set of tests to do element-wise operations."""

  def f(options):
    # [FUNCTION DOC]
    # HELPER: the generator function returned by the factory.
    # KNOWN ISSUE: each parameter group lists an input range, but the builders ignore it and use
    # the factory arguments instead.
    # Takes the test options object. Defines two parameter groups (float32 not quantized; float32
    # fully quantized with 8-bit or 16x8 activations) over scalar through 4-D shapes. When full
    # quantization is not allowed, the quantized group is filtered out. Calls the zip generator
    # with nested builders. Returns nothing.
    """Actual function that generates examples."""
    test_parameters = [
        {
            "input_dtype": [tf.float32],
            "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
            "fully_quantize": [False],
            "quant_16x8": [False],
            "input_range": [[min_value, max_value]],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
            "fully_quantize": [True],
            "quant_16x8": [False, True],
            "input_range": [[min_value, max_value]],
        },
    ]

    if not allow_fully_quantize:
      test_parameters = [
          test_parameter for test_parameter in test_parameters
          if True not in test_parameter["fully_quantize"]
      ]

    def build_graph(parameters):
      # [FUNCTION DOC]
      # Creates a named float placeholder of the parameterized dtype and shape and applies the
      # unary op under test. Returns a pair: list with the input tensor and list with the output.
      """Build the unary op testing graph."""
      input_value = tf.compat.v1.placeholder(
          dtype=parameters["input_dtype"],
          name="input1",
          shape=parameters["input_shape"])
      out = op(input_value)
      return [input_value], [out]

    def build_inputs(parameters, sess, inputs, outputs):
      # [FUNCTION DOC]
      # Generates random data within the factory's minimum and maximum, evaluates the graph in the
      # session, and returns a pair: list with the input value and list of expected outputs.
      input_value = create_tensor_data(parameters["input_dtype"],
                                       parameters["input_shape"],
                                       min_value=min_value,
                                       max_value=max_value)
      return [input_value], sess.run(
          outputs, feed_dict={inputs[0]: input_value})

    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)

  return f


@register_make_test_function()
def make_sin_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker for sine using default settings (no full quantization, range minus 100
  # to 100). Takes the options object; generates tests; returns nothing.
  """Make a set of tests to do sin."""
  return _make_elementwise_tests(tf.sin)(options)


@register_make_test_function()
def make_log_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker for natural log. Allows fully-quantized cases and restricts inputs to
  # the positive range 0.1 to 10. Takes the options object; generates tests; returns nothing.
  """Make a set of tests to do log."""
  return _make_elementwise_tests(
      tf.math.log,
      allow_fully_quantize=True,
      min_value=0.1,
      max_value=10,
  )(options)


@register_make_test_function()
def make_sqrt_tests(options):
  # [FUNCTION DOC]
  # KNOWN ISSUE: uses the default input range minus 100 to 100, so negative inputs produce
  # not-a-number results for square root.
  # Registered test-maker for square root without quantized cases. Takes the options object;
  # generates tests; returns nothing.
  """Make a set of tests to do sqrt."""
  return _make_elementwise_tests(tf.sqrt)(options)


@register_make_test_function()
def make_rsqrt_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker for reciprocal square root. Allows fully-quantized cases and restricts
  # inputs to 0.1 to 1. Takes the options object; generates tests; returns nothing.
  """Make a set of tests to do 1/sqrt."""
  return _make_elementwise_tests(
      tf.math.rsqrt,
      allow_fully_quantize=True,
      min_value=0.1,
      max_value=1,
  )(options)


@register_make_test_function()
def make_square_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker for square using default settings. Takes the options object; generates
  # tests; returns nothing.
  """Make a set of tests to do square."""
  return _make_elementwise_tests(tf.square)(options)
