# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the split operation
# with explicit per-piece sizes (the "split V" variant). Defines and registers one generator
# function that creates parameterized graphs and passes them to the shared zip-test machinery.
# Import-time side effect: registers the generator in the global registry of test-making
# functions.
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
"""Test configs for splitv."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_splitv_tests(options):
  # [FUNCTION DOC]
  # Generates the size-based split test set. The options argument is the test-run configuration
  # passed to the zip-test builder. Parameters sweep input shape, a list of piece sizes (some
  # with a -1 entry meaning "the remainder") and axis (positive and negative). Invalid
  # combinations are expected to fail in TensorFlow; the builder is told to expect 158 failures.
  # Returns nothing; output is produced through the builder.
  """Make a set of tests to do tf.split_v."""

  test_parameters = [{
      "input_shape": [[1, 3, 4, 6], [2, 4, 1], [6, 4], [8]],
      "size_splits": [[2, 2], [1, 3], [4, 2], [5, 3], [-1, 1], [-1, 2], [-1,
                                                                         4]],
      "axis": [0, 1, 2, 3, -4, -3, -2, -1],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: a float32 placeholder named input is split
    # by the given list of sizes along the given axis. Returns a pair: list with the input
    # placeholder and a list holding only the first split output (the other pieces are not
    # exposed as graph outputs).
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.split(input_tensor, parameters["size_splits"], parameters["axis"])
    return [input_tensor], [out[0]]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: one random float32 array of the parameter
    # input shape is fed into the graph in the given session. Returns a pair: list of input
    # arrays and the evaluated outputs.
    values = [create_tensor_data(np.float32, parameters["input_shape"])]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=158)
