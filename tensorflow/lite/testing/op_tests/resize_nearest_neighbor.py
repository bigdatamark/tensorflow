# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the nearest-neighbor resize op with a constant target size.
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
"""Test configs for resize_nearest_neighbor."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_resize_nearest_neighbor_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Defines parameter groups of dtype, input shape (4-D), target size (height and width), the
  # align-corners flag, the half-pixel-centers flag and the fully-quantize flag. Groups cover float32 with and without full quantization for align-corners and half-pixel-centers variants (never combined).
  # Calls make_zip_of_tests with the nested builders.
  """Make a set of tests to do resize_nearest_neighbor."""

  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[1, 3, 4, 3], [1, 10, 2, 1]],
      "size": [[1, 1], [4, 3], [2, 2], [5, 6]],
      "align_corners": [False],
      "half_pixel_centers": [False],
      "fully_quantize": [True, False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 16, 24, 3], [1, 12, 18, 3]],
      "size": [[8, 12], [12, 18]],
      "align_corners": [True],
      "half_pixel_centers": [False],
      "fully_quantize": [True, False]
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 16, 24, 3], [1, 12, 18, 3]],
      "size": [[8, 12], [12, 18]],
      "align_corners": [False],
      "half_pixel_centers": [True],
      "fully_quantize": [True, False]
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests for the resize nearest_neighbor tests.
    # Creates an input placeholder of the requested dtype and shape and applies the v1 nearest-neighbor resize op with the
    # constant size, align-corners and half-pixel-centers options. Returns the list of input
    # placeholders and the list of outputs.
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.compat.v1.image.resize_nearest_neighbor(
        input_tensor,
        size=parameters["size"],
        align_corners=parameters["align_corners"],
        half_pixel_centers=parameters["half_pixel_centers"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Creates random data of the requested dtype and shape in the range minus 1 to 1, runs the graph in
    # the session and returns the list of input values and the list of computed outputs.
    input_values = create_tensor_data(
        parameters["dtype"],
        parameters["input_shape"],
        min_value=-1,
        max_value=1)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
