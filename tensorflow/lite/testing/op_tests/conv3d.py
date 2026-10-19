# [MODULE DOC]
# KNOWN ISSUE: the module docstring says "exp" (copy-paste); this module generates 3-D convolution
# tests, not exp tests.
# Test-generation module for the TFLite converter zip tests of 3-D convolution. Defines one
# registered test-maker over two input shapes, two filter shapes, several strides and both
# paddings with fed (placeholder) input and filter. Importing registers the maker in the global
# registry.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for exp."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_conv3d_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Declares one float32 parameter set
  # crossing input shapes, filter shapes, five stride tuples, unit dilation, and same/valid padding,
  # then calls the zip-test generator with default settings. Returns nothing.
  """Make a set of tests to do conv3d."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[2, 3, 4, 5, 3], [2, 5, 6, 8, 3]],
      "filter_shape": [[2, 2, 2, 3, 2], [1, 2, 2, 3, 2]],
      "strides": [(1, 1, 1, 1, 1), (1, 1, 1, 2, 1), (1, 1, 2, 2, 1),
                  (1, 2, 1, 2, 1), (1, 2, 2, 2, 1)],
      "dilations": [(1, 1, 1, 1, 1)],
      "padding": ["SAME", "VALID"],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the conv3d zip tests.
    # KNOWN ISSUE: the docstring says "exp op" (copy-paste); the graph holds a 3-D convolution.
    # Creates placeholders named input and filter with the parameterized dtype and shapes and
    # applies 3-D convolution with the given strides, dilations and padding. Returns the list of both
    # placeholders and the list with the output tensor.
    """Build the exp op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    filter_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="filter",
        shape=parameters["filter_shape"])

    out = tf.nn.conv3d(
        input_tensor,
        filter_tensor,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"])
    return [input_tensor, filter_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the conv3d zip tests.
    # Generates random input data in the range minus one hundred to nine and random filter data in
    # minus three to three, runs the session, and returns the list of both values and the expected
    # outputs.
    values = [
        create_tensor_data(
            parameters["input_dtype"],
            parameters["input_shape"],
            min_value=-100,
            max_value=9),
        create_tensor_data(
            parameters["input_dtype"],
            parameters["filter_shape"],
            min_value=-3,
            max_value=3)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
