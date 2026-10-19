# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of 2-D convolution. Defines one
# registered test-maker varying input and filter shapes, strides, dilations, padding (named or
# explicit), constant versus fed filter, channel multiplier, and quantization modes (full, 16x8,
# dynamic range). Importing registers the maker in the global registry.
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
"""Test configs for conv."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_conv_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Declares parameter sets: a broad
  # float sweep, a fully-quantized 16x8 set, a fully-quantized sweep with constant filter, and two
  # dynamic-range-quantized sets (one using explicit padding amounts). Only channels-last layout is
  # used. Invalid combinations are counted as 60 expected TensorFlow failures passed to the zip-test
  # generator. Returns nothing.
  """Make a set of tests to do convolution."""

  test_parameters = [
      {
          "input_shape": [[1, 3, 4, 3], [4, 6, 6, 1]],
          "filter_shape": [[1, 1], [2, 3], [3, 3]],
          "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
          "dilations": [[1, 1, 1, 1], [1, 3, 2, 1], [1, 2, 2, 1]],
          "padding": ["SAME", "VALID"],
          "data_format": ["NHWC"],  # TODO(aselle): NCHW  would be good
          "constant_filter": [True, False],
          "channel_multiplier": [1, 2],
          "fully_quantize": [False],
          "quant_16x8": [False],
          "dynamic_range_quantize": [False]
      },
      {
          "input_shape": [[1, 3, 4, 3]],
          "filter_shape": [[1, 1], [2, 3]],
          "strides": [[1, 1, 1, 1]],
          "dilations": [[1, 1, 1, 1]],
          "padding": ["SAME"],
          "data_format": ["NHWC"],
          "constant_filter": [True],
          "channel_multiplier": [1, 2],
          "fully_quantize": [True],
          "quant_16x8": [True],
          "dynamic_range_quantize": [False],
      },
      # TODO(b/134702301): The fully_quantize param is just ignored by the MLIR
      # testing path now, resulting in duplicate tests. Either ignore these
      # tests or handle it properly in the mlir_convert() function.
      {
          "input_shape": [[1, 3, 4, 3], [4, 6, 6, 1]],
          "filter_shape": [[1, 1], [2, 3], [3, 3]],
          "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
          "dilations": [[1, 1, 1, 1], [1, 3, 2, 1], [1, 2, 2, 1]],
          "padding": ["SAME", "VALID"],
          "data_format": ["NHWC"],  # TODO(aselle): NCHW  would be good
          "constant_filter": [True],
          "channel_multiplier": [1, 2],
          "fully_quantize": [True],
          "quant_16x8": [False],
          "dynamic_range_quantize": [False]
      },
      {
          "input_shape": [[1, 3, 4, 3]],
          "filter_shape": [[1, 1]],
          "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
          "dilations": [[1, 1, 1, 1]],
          "padding": ["SAME", "VALID"],
          "data_format": ["NHWC"],
          "constant_filter": [True],
          "channel_multiplier": [2],
          "fully_quantize": [False],
          "quant_16x8": [False],
          "dynamic_range_quantize": [True]
      },
      {
          "input_shape": [[1, 3, 4, 3]],
          "filter_shape": [[1, 1]],
          "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
          "dilations": [[1, 1, 1, 1]],
          "padding": [[[0, 0], [1, 1], [1, 1], [0, 0]]],
          "data_format": ["NHWC"],
          "constant_filter": [True],
          "channel_multiplier": [2],
          "fully_quantize": [False],
          "quant_16x8": [False],
          "dynamic_range_quantize": [True]
      },
  ]

  def get_tensor_shapes(parameters):
    # [FUNCTION DOC]
    # HELPER: shape utility for the conv tests.
    # Returns a two-element list: the input shape, and the filter shape formed from the two spatial
    # filter sizes followed by the input channel count (fourth input dimension) and the channel
    # multiplier.
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_shape"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the conv zip tests.
    # Creates a float32 placeholder named input. The filter is a constant of random data in minus ten
    # to ten when constant_filter is set, otherwise a second placeholder named filter. Applies 2-D
    # convolution with the parameterized strides, dilations, padding and data format. Returns the list
    # of placeholders (input, plus filter if non-constant) and the list with the output tensor.
    """Build a conv graph given `parameters`."""
    input_shape, filter_shape = get_tensor_shapes(parameters)
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)

    # Get filter input either as a placeholder or constants. Also get a list of
    # the input tensors that are represented as placeholders.
    if parameters["constant_filter"]:
      filter_input = create_tensor_data(
          np.float32, filter_shape, min_value=-10, max_value=10)
      input_tensors = [input_tensor]
    else:
      filter_input = tf.compat.v1.placeholder(
          dtype=tf.float32, name="filter", shape=filter_shape)
      input_tensors = [input_tensor, filter_input]

    out = tf.nn.conv2d(
        input=input_tensor,
        filters=filter_input,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the conv zip tests.
    # Generates random input data in minus one to one and, only when the filter is a fed tensor,
    # random filter data in the default range. Runs the session and returns the list of values and
    # the expected outputs.
    # Build list of input values either containing 1 tensor (input) or 2 tensors
    # (input, filter) based on whether filter is constant or variable input.
    input_shape, filter_shape = get_tensor_shapes(parameters)
    values = [
        create_tensor_data(np.float32, input_shape, min_value=-1, max_value=1)
    ]
    if not parameters["constant_filter"]:
      values.append(create_tensor_data(np.float32, filter_shape))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=60)
