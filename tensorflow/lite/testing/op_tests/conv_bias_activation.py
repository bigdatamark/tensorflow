# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of two consecutive convolutions, each
# followed by a bias add and an activation, using channels-first layout so converter layout
# transformations are exercised. A factory builds the maker for any activation; one registered
# wrapper covers relu6. Importing registers that wrapper in the global registry.
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
"""Test configs for conv followed with bias Add and activations."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def make_conv_bias_activation_tests(activation_op):
  # [FUNCTION DOC]
  # HELPER: factory shared by the registered conv-bias-activation wrappers.
  # Takes an activation callable and returns the inner test-maker function (taking generator
  # options) which generates the two-convolution tests. No side effects until the returned function
  # is called.
  """Make a set of tests to do convolution with activation and bias.

  This test will create multiple consecutive convolutions with NCHW layout to
  make sure that the tranformations to NHWC works as expected. Note this
  doesn't check any performance so manual checking of the generated model is
  advised.

  Args:
    activation_op: The activation op to be used in the test.

  Returns:
    The function that creates the test.
  """

  def create_test(options):
    # [FUNCTION DOC]
    # HELPER: the test maker returned by the factory; it closes over the activation callable.
    # Takes the generator options object. Declares one parameter set with channels-first layout,
    # two first-conv filter sizes, a second-conv filter shape, and channel multipliers one and two.
    # Calls the zip-test generator expecting two TensorFlow failures. Returns nothing.
    """Actual function that generates examples."""
    test_parameters = [
        {
            "input_shape": [[1, 3, 4, 3]],
            "filter_shape": [[2, 3], [3, 3]],
            "filter_2_shape": [[2, 1, 1, 3]],
            "strides": [[1, 1, 1, 1]],
            "dilations": [[1, 1, 1, 1]],
            "data_format": ["NCHW"],
            "channel_multiplier": [1, 2],
            "fully_quantize": [False],
            "dynamic_range_quantize": [False],
        },
    ]

    def get_tensor_shapes(parameters):
      # [FUNCTION DOC]
      # HELPER: shape utility for the conv-bias-activation tests.
      # Returns a two-element list: the input shape, and the first filter shape made of the two
      # spatial filter sizes, the input channel count (fourth input dimension) and the channel
      # multiplier.
      input_shape = parameters["input_shape"]
      filter_size = parameters["filter_shape"]
      filter_shape = filter_size + [
          input_shape[3], parameters["channel_multiplier"]
      ]
      return [input_shape, filter_shape]

    # TF CPU doesn't support cases with NCHW. Instead
    # use XLA which doesn't have the same restrictions.
    @tf.function(jit_compile=True)
    def add_conv(input_tensor, filter_input, parameters):
      # [FUNCTION DOC]
      # HELPER: convolution step for channels-first graphs (compiled with XLA because the CPU
      # kernel does not support that layout).
      # Takes the input tensor, the filter tensor and the parameter mapping; applies 2-D convolution
      # with valid padding and the parameterized strides, dilations and data format. Returns the
      # output tensor.
      out = tf.nn.conv2d(
          input=input_tensor,
          filters=filter_input,
          strides=parameters["strides"],
          dilations=parameters["dilations"],
          padding="VALID",
          data_format=parameters["data_format"])
      return out

    def add_bias_add(data_input, filter_shape):
      # [FUNCTION DOC]
      # HELPER: bias step for the conv-bias-activation tests.
      # Creates a constant random float bias whose length is the last entry of the given filter shape
      # and adds it to the data using channels-last bias semantics. Returns the output tensor.
      bias_input = create_tensor_data(np.float32, (filter_shape[-1],))
      out = tf.nn.bias_add(data_input, bias_input, data_format="NHWC")
      return out

    def build_graph(parameters):
      # [FUNCTION DOC]
      # HELPER: graph builder callback for the conv-bias-activation zip tests.
      # Creates a float32 placeholder input and runs two stacked stages of convolution (constant
      # random filter in minus ten to ten; second stage uses the second filter shape), bias add and
      # activation; the convolution uses the XLA-compiled helper for channels-first layout and plain
      # convolution otherwise. Both bias adds size the bias from the first filter shape. Returns the
      # list with the placeholder and the list with the output tensor.
      """Build a conv graph given `parameters`."""
      input_shape, filter_shape = get_tensor_shapes(parameters)
      input_tensor = tf.compat.v1.placeholder(
          dtype=tf.float32, name="input", shape=input_shape)

      filter_input = create_tensor_data(
          np.float32, filter_shape, min_value=-10, max_value=10)
      input_tensors = [input_tensor]

      if parameters["data_format"] == "NCHW":
        out = add_conv(input_tensor, filter_input, parameters)
      else:
        out = tf.nn.conv2d(
            input=input_tensor,
            filters=filter_input,
            strides=parameters["strides"],
            dilations=parameters["dilations"],
            padding="VALID",
            data_format=parameters["data_format"])
      out = add_bias_add(out, filter_shape)
      out = activation_op(out)

      # Add another conv + bias_add + activation.

      # Create constant filter for the second conv2d.
      filter_input_2 = create_tensor_data(
          np.float32, parameters["filter_2_shape"], min_value=-10, max_value=10)
      if parameters["data_format"] == "NCHW":
        out = add_conv(out, filter_input_2, parameters)
      else:
        out = tf.nn.conv2d(
            input=out,
            filters=filter_input_2,
            strides=parameters["strides"],
            dilations=parameters["dilations"],
            padding="VALID",
            data_format=parameters["data_format"])
      out = add_bias_add(out, filter_shape)
      out = activation_op(out)
      return input_tensors, [out]

    def build_inputs(parameters, sess, inputs, outputs):
      # [FUNCTION DOC]
      # HELPER: input-feeding callback for the conv-bias-activation zip tests.
      # Generates random input data in minus one to one for the input shape, runs the session, and
      # returns the list with the data and the expected outputs.
      """Build inputs for conv with activation."""

      input_shape, _ = get_tensor_shapes(parameters)
      values = [
          create_tensor_data(
              np.float32, input_shape, min_value=-1, max_value=1)
      ]
      return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

    make_zip_of_tests(
        options,
        test_parameters,
        build_graph,
        build_inputs,
        expected_tf_failures=2)

  return create_test


@register_make_test_function()
def make_conv_bias_relu6_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for convolution with bias followed by relu6. Takes the generator options
  # object, builds the generic conv-bias-activation maker with the relu6 op and invokes it. Returns
  # whatever that call returns (nothing).
  """Make a set of tests to do conv_bias_relu6."""
  return make_conv_bias_activation_tests(tf.nn.relu6)(options)
