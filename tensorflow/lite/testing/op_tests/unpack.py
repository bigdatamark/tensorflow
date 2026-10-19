# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the unpack (unstack)
# operation, which splits a tensor along one axis into slices. Defines and registers one
# generator function that creates parameterized graphs and passes them to the shared zip-test
# machinery. Import-time side effect: registers the generator in the global registry of
# test-making functions.
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
"""Test configs for unpack."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_unpack_tests(options):
  # [FUNCTION DOC]
  # Generates the unpack test set. The options argument is the test-run configuration passed to
  # the zip-test builder. Parameters sweep input shape (3-D, 2-D, 4-D), axis (0 to 3) and type
  # (int32, bool, float32). Axes beyond the input rank are clamped by an inner helper. Returns
  # nothing; output is produced through the builder.
  """Make a set of tests to do unpack."""

  test_parameters = [{
      "base_shape": [[3, 4, 3], [3, 4], [5, 6, 7, 8]],
      "axis": [0, 1, 2, 3],
      "dtype": [tf.int32, tf.bool, tf.float32],
  }]

  def get_valid_axis(parameters):
    # [FUNCTION DOC]
    # HELPER: keeps the axis valid for the graph builder of the unpack tests.
    # Takes the parameter combination and returns the axis, reduced step by step until it is at
    # most the last dimension index of the input shape (so too-large axes become the last axis).
    # Does not modify the parameters.
    """Return a tweaked version of 'axis'."""
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    while axis > len(shape) - 1:
      axis -= 1
    return axis

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: a placeholder named input of the given
    # type and shape, unstacked along the clamped axis. Returns a pair: list with the input
    # placeholder and a list holding only the first slice (the other slices are not outputs).
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name=("input"),
        shape=parameters["base_shape"])
    outs = tf.unstack(input_tensor, axis=get_valid_axis(parameters))
    return [input_tensor], [outs[0]]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random data of the parameter type and shape is
    # fed into the graph in the given session. Returns a pair: list with the input array and the
    # evaluated outputs.
    input_value = create_tensor_data(
        parameters["dtype"], shape=parameters["base_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
