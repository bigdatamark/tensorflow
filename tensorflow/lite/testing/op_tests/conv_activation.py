# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of 2-D convolution followed by an
# activation. A factory builds a test-maker for any activation callable; registered wrappers cover
# relu6, relu and relu1 (the latter defined locally as clamp to minus one to one). Importing
# registers the three wrappers in the global registry.
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
"""Test configs for conv with activations."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def make_conv_activation_tests(activation_op):
  # [FUNCTION DOC]
  # HELPER: factory shared by the registered conv-activation wrappers.
  # Takes an activation callable (one tensor in, one tensor out) and returns a one-argument function
  # (taking generator options) that produces the convolution-plus-activation zip tests. No side
  # effects until the returned function is called.
  """Make a set of tests to do convolution with activation."""

  def f(options):
    # [FUNCTION DOC]
    # HELPER: the test maker returned by the factory; it closes over the activation callable.
    # Takes the generator options object. Declares parameter sets: float sweep with constant or fed
    # filter, fully-quantized (optionally 16x8) sweep with constant filter, and dynamic-range
    # quantized sweep. Calls the zip-test generator expecting 48 TensorFlow failures. Returns
    # nothing.
    """Actual function that generates examples."""
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
            "dynamic_range_quantize": [False],
        },
        # TODO(b/134702301): The fully_quantize param is just ignored by the
        # MLIR testing path now, resulting in duplicate tests. Either ignore
        # these tests or handle it properly in the mlir_convert() function.
        {
            "input_shape": [[1, 3, 4, 3], [4, 6, 6, 1]],
            "filter_shape": [[1, 1], [2, 3]],
            "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
            "dilations": [[1, 1, 1, 1], [1, 3, 2, 1]],
            "padding": ["SAME", "VALID"],
            "data_format": ["NHWC"],  # TODO(aselle): NCHW  would be good
            "constant_filter": [True],
            "channel_multiplier": [1, 2],
            "fully_quantize": [True],
            "quant_16x8": [False, True],
            "dynamic_range_quantize": [False],
        },
        {
            "input_shape": [[1, 3, 4, 3]],
            "filter_shape": [[1, 1], [2, 3], [3, 3]],
            "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
            "dilations": [[1, 1, 1, 1]],
            "padding": ["SAME", "VALID"],
            "data_format": ["NHWC"],
            "constant_filter": [True],
            "channel_multiplier": [1, 2],
            "fully_quantize": [False],
            "quant_16x8": [False],
            "dynamic_range_quantize": [True],
        },
    ]

    def get_tensor_shapes(parameters):
      # [FUNCTION DOC]
      # HELPER: shape utility for the conv-activation tests.
      # Returns a two-element list: the input shape, and the filter shape made of the two spatial
      # filter sizes, the input channel count (fourth input dimension) and the channel multiplier.
      input_shape = parameters["input_shape"]
      filter_size = parameters["filter_shape"]
      filter_shape = filter_size + [
          input_shape[3], parameters["channel_multiplier"]
      ]
      return [input_shape, filter_shape]

    def build_graph(parameters):
      # [FUNCTION DOC]
      # HELPER: graph builder callback for the conv-activation zip tests.
      # Creates a float32 placeholder input; the filter is a constant of random data in minus ten to
      # ten, or a second placeholder when constant_filter is false. Applies 2-D convolution with the
      # parameterized strides, dilations, padding and data format, then the activation callable.
      # Returns the list of placeholders and the list with the output tensor.
      """Build a conv graph given `parameters`."""
      input_shape, filter_shape = get_tensor_shapes(parameters)
      input_tensor = tf.compat.v1.placeholder(
          dtype=tf.float32, name="input", shape=input_shape)

      # Get filter input either as a placeholder or constants. Also get a list
      # of the input tensors that are represented as placeholders.
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
      out = activation_op(out)
      return input_tensors, [out]

    def build_inputs(parameters, sess, inputs, outputs):
      # [FUNCTION DOC]
      # HELPER: input-feeding callback for the conv-activation zip tests.
      # Generates random input data in minus one to one and, if the filter is fed, random filter
      # data; runs the session and returns the list of values and the expected outputs.
      """Build inputs for conv with activation."""

      input_shape, filter_shape = get_tensor_shapes(parameters)
      values = [
          create_tensor_data(
              np.float32, input_shape, min_value=-1, max_value=1)
      ]
      if not parameters["constant_filter"]:
        values.append(create_tensor_data(np.float32, filter_shape))
      return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

    make_zip_of_tests(
        options,
        test_parameters,
        build_graph,
        build_inputs,
        expected_tf_failures=48)

  return f


@register_make_test_function()
def make_conv_relu6_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for convolution followed by relu6. Takes the generator options object,
  # builds the generic conv-activation maker with the relu6 op and invokes it. Returns whatever that
  # call returns (nothing).
  """Make a set of tests to do conv_relu6."""
  return make_conv_activation_tests(tf.nn.relu6)(options)


@register_make_test_function()
def make_conv_relu_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for convolution followed by relu. Takes the generator options object,
  # builds the generic conv-activation maker with the relu op and invokes it. Returns whatever that
  # call returns (nothing).
  """Make a set of tests to do conv_relu."""
  return make_conv_activation_tests(tf.nn.relu)(options)


def relu1(input_tensor):
  # [FUNCTION DOC]
  # HELPER: relu1 activation for the conv-activation tests (not a registered maker).
  # Takes a tensor and returns it clamped elementwise to the range minus one to one, built as
  # minimum of one and maximum of the input and minus one; the reverse nesting is noted in a
  # comment as unsupported by the converter.
  # Note that the following is not supported:
  #   out = tf.maximum(-1.0, tf.minimum(input_tensor, 1.0))
  out = tf.minimum(1.0, tf.maximum(input_tensor, -1.0))
  return out


@register_make_test_function()
def make_conv_relu1_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for convolution followed by relu1 (clamp to minus one to one). Takes the
  # generator options object, builds the generic conv-activation maker with the local relu1 function
  # and invokes it. Returns whatever that call returns (nothing).
  """Make a set of tests to do conv_relu1."""
  return make_conv_activation_tests(relu1)(options)
