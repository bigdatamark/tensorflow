# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering dilated 2-D convolution expressed
# through the generic convolution API, which TensorFlow lowers to a space-to-batch, convolution,
# batch-to-space sequence; the converter must recognize it as a dilated convolution. One
# registered test-maker. Importing the module registers it in the global registry of
# zip_test_utils.
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
"""Test configs for identifying dilated conv."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_identify_dilated_conv_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines a grid over input shapes,
  # filter sizes, strides, dilations, padding, constant-filter flag and channel multiplier, and
  # calls the zip generator declaring that 168 combinations are expected to fail in the reference
  # framework. Returns nothing.
  """Make a set of tests to do dilated convolution."""

  test_parameters = [
      {
          "input_shape": [[1, 3, 4, 3], [4, 6, 6, 1]],
          "filter_shape": [[1, 1], [2, 3], [3, 3]],
          "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
          "dilations": [[1, 3, 2, 1], [1, 2, 2, 1], [1, 2, 1, 1]],
          "padding": ["VALID", "SAME"],
          "data_format": ["NHWC"],
          "constant_filter": [True, False],
          "channel_multiplier": [1, 2],
      },
  ]

  def get_tensor_shapes(parameters):
    # [FUNCTION DOC]
    # HELPER: computes input and filter shapes shared by the graph and input builders.
    # Takes the parameter mapping; returns a list of two shapes: the input shape, and the filter
    # shape (filter height and width, then input channel count, then channel multiplier).
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_shape"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates a float32 placeholder for the input. The filter is either random constant data in
    # the range minus ten to ten or a second named placeholder. Applies the generic convolution
    # with the parameterized strides, dilations, padding and data format. Returns a pair: list of
    # placeholder inputs (one or two) and list with the output.
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

    # Use `tf.nn.convolution` here since it will create the `batch_to_space` and
    # the `space_to_batch` ops respectively.
    out = tf.nn.convolution(
        input=input_tensor,
        filters=filter_input,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random input data between minus one and one, plus filter data (default range)
    # when the filter is a placeholder. Evaluates the graph and returns a pair: list of input
    # values and list of expected outputs.
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
      expected_tf_failures=168)
