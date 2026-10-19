# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the tensor split
# operation with an integer number of splits. Defines and registers one generator function that
# creates parameterized graphs and passes them to the shared zip-test machinery. Import-time side
# effect: registers the generator in the global registry of test-making functions.
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
"""Test configs for split."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_split_tests(options):
  # [FUNCTION DOC]
  # Generates the split test set. The options argument is the test-run configuration passed to
  # the zip-test builder. Parameters sweep input shape (1-D to 4-D), number of splits (1 to 5),
  # axis (positive and negative) and a fully-quantize flag. Combinations where the split count
  # does not divide the axis size make TensorFlow itself fail; the builder is told to expect
  # 224 such failures. Returns nothing; output is produced through the builder.
  """Make a set of tests to do tf.split."""

  test_parameters = [{
      "input_shape": [[1, 3, 4, 6], [2, 4, 1], [6, 4], [8]],
      "num_or_size_splits": [1, 2, 3, 4, 5],
      "axis": [0, 1, 2, 3, -4, -3, -2, -1],
      "fully_quantize": [True, False],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination (mapping from names to values): a float32
    # placeholder named input with the given shape, split into the requested number of pieces
    # along the given axis. Returns a pair: list with the input placeholder and the list of all
    # split output tensors.
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.split(input_tensor, parameters["num_or_size_splits"],
                   parameters["axis"])
    return [input_tensor], out

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs. Creates one random float32 array of the
    # parameter input shape with values between -1 and 1, evaluates the graph in the given
    # session by feeding it to the input tensors, and returns a pair: list of input arrays and
    # the evaluated outputs.
    values = [
        create_tensor_data(
            np.float32, parameters["input_shape"], min_value=-1, max_value=1)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=224)
