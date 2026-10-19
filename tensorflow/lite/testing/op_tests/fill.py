# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering the fill op (create a tensor of a
# given shape filled with one scalar value). Two registered test-makers: one with runtime dims and
# runtime values of several dtypes, one with a constant half-precision value. Importing the module
# registers both in the global registry of zip_test_utils.
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
"""Test configs for fill."""
import tensorflow.compat.v2 as tf
from tensorflow.lite.testing.zip_test_utils import create_scalar_data
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_fill_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines a grid over dims dtype (int32
  # or int64), dims shape and value dtype (int32, int64, float32, bool, string), and calls the zip
  # generator, declaring that twenty combinations are expected to fail in the reference framework.
  # Returns nothing.
  """Make a set of tests to do fill."""

  test_parameters = [{
      "dims_dtype": [tf.int32, tf.int64],
      "dims_shape": [[], [1], [3], [3, 3]],
      "value_dtype": [tf.int32, tf.int64, tf.float32, tf.bool, tf.string],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates a dims placeholder and a scalar value placeholder and fills a tensor of the given
    # dims with the value. Returns a pair: list of both input tensors and list with the output.
    """Build the fill op testing graph."""
    input1 = tf.compat.v1.placeholder(
        dtype=parameters["dims_dtype"],
        name="dims",
        shape=parameters["dims_shape"])
    input2 = tf.compat.v1.placeholder(
        dtype=parameters["value_dtype"], name="value", shape=[])
    out = tf.fill(input1, input2)
    return [input1, input2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random dims data with minimum value one (to avoid zero-sized dimensions) and a
    # random scalar value, evaluates the graph, and returns a pair: list of input values (dims,
    # value) and list of expected outputs.
    input1 = create_tensor_data(parameters["dims_dtype"],
                                parameters["dims_shape"], 1)
    input2 = create_scalar_data(parameters["value_dtype"])
    return [input1, input2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input1, input2])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=20)


@register_make_test_function()
def make_fill_16_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker for fill with a constant half-precision value. Takes the test options
  # object, defines a grid over dims dtype and dims shape, and calls the zip generator expecting
  # no reference-framework failures. Returns nothing.
  """Make a set of tests to do fill with fp16."""

  test_parameters = [{
      "dims_dtype": [tf.int32, tf.int64],
      "dims_shape": [[], [1], [3], [3, 3]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates a dims placeholder and fills a tensor of those dims with a constant half-precision
    # one. Returns a pair: list with the dims tensor and list with the output.
    """Build the fill op testing graph."""
    input1 = tf.compat.v1.placeholder(
        dtype=parameters["dims_dtype"],
        name="dims",
        shape=parameters["dims_shape"])
    const_fp16 = tf.constant(1.0, dtype=tf.float16)
    out = tf.fill(input1, const_fp16)
    return [input1], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random dims data with minimum value one, evaluates the graph, and returns a pair:
    # list with the dims value and list of expected outputs.
    input1 = create_tensor_data(parameters["dims_dtype"],
                                parameters["dims_shape"], 1)
    return [input1], sess.run(outputs, feed_dict=dict(zip(inputs, [input1])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=0)
