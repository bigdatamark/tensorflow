# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the reverse op (reverse_v2) along a single axis for float32 and boolean tensors.
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
"""Test configs for reverse_v2."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_reverse_v2_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Defines one parameter group of dtype (float32, bool), base shape (2-D to 4-D) and a requested axis
  # from minus 2 to 3, then calls make_zip_of_tests with the nested builders.
  """Make a set of tests to do reverse_v2."""

  test_parameters = [{
      "dtype": [tf.float32, tf.bool],
      "base_shape": [[3, 4, 3], [3, 4], [5, 6, 7, 8]],
      "axis": [-2, -1, 0, 1, 2, 3],
  }]

  def get_valid_axis(parameters):
    # [FUNCTION DOC]
    # HELPER: adjusts the requested axis so it is usable for the chosen shape.
    # Input: parameters mapping. Copies the shape and, while the axis exceeds the last dimension index,
    # decrements it. Returns the adjusted integer axis. Negative axes are returned unchanged.
    """Return a tweaked version of 'axis'."""
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    while axis > len(shape) - 1:
      axis -= 1
    return axis

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests.
    # Creates a placeholder named input of the requested dtype and base shape and reverses it along the
    # single axis returned by the axis adjusting helper. Returns the list of input placeholders and the
    # list of outputs.
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name=("input"),
        shape=parameters["base_shape"])
    outs = tf.reverse(input_tensor, axis=[get_valid_axis(parameters)])
    return [input_tensor], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Creates random data of the requested dtype and base shape, runs the graph in the session and
    # returns the list of input values and the list of computed outputs.
    input_value = create_tensor_data(
        parameters["dtype"], shape=parameters["base_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
