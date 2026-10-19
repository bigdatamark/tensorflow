# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the transpose
# operation (permuting tensor dimensions). Defines and registers one generator function that
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
"""Test configs for transpose."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_transpose_tests(options):
  # [FUNCTION DOC]
  # Generates the transpose test set. The options argument is the test-run configuration passed
  # to the zip-test builder. Parameter groups cover 3-D, 4-D, 5-D and 2-D (negative permutation
  # entries) float inputs, int32/int64/float32 types for the 3-D case, permutations given either
  # as a constant or as a runtime input, and fully-quantized variants (constant permutation
  # only, except one group that tries both). Returns nothing.
  """Make a set of tests to do transpose."""

  # TODO(nupurgarg): Add test for uint8.
  test_parameters = [{
      "dtype": [tf.int32, tf.int64, tf.float32],
      "input_shape": [[2, 2, 3]],
      "perm": [[0, 1, 2], [0, 2, 1]],
      "constant_perm": [True, False],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4]],
      "perm": [[0, 1, 2, 3], [3, 0, 1, 2]],
      "constant_perm": [True, False],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4, 5]],
      "perm": [[4, 3, 2, 1, 0]],
      "constant_perm": [True, False],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[2, 2, 3]],
      "perm": [[0, 1, 2], [0, 2, 1]],
      "constant_perm": [True],
      "fully_quantize": [True],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4]],
      "perm": [[0, 1, 2, 3], [3, 0, 1, 2]],
      "constant_perm": [True],
      "fully_quantize": [True],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4, 5]],
      "perm": [[0, 1, 2, 3, 4], [3, 4, 0, 1, 2]],
      "constant_perm": [True],
      "fully_quantize": [True, False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[2, 2]],
      "perm": [[-2, -1]],
      "constant_perm": [True, False],
      "fully_quantize": [False],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: a placeholder named input. If the
    # permutation is constant it is used directly; otherwise a 32-bit integer placeholder named
    # perm with length equal to the permutation size is added as a second graph input. Applies
    # transpose with that permutation. Returns a pair: list of graph inputs and list with the
    # output.
    """Build a transpose graph given `parameters`."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])

    if parameters["constant_perm"]:
      perm = parameters["perm"]
      input_tensors = [input_tensor]
    else:
      shape = len(parameters["perm"])
      perm = tf.compat.v1.placeholder(dtype=tf.int32, name="perm", shape=shape)
      input_tensors = [input_tensor, perm]

    out = tf.transpose(a=input_tensor, perm=perm)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random data (range -1 to 1) of the parameter
    # type and shape; when the permutation is not constant, the permutation as an array is
    # appended as a second value. Evaluates the graph in the given session. Returns a pair: list
    # of input arrays and the evaluated outputs.
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"],
                           min_value=-1, max_value=1)
    ]
    if not parameters["constant_perm"]:
      values.append(np.array(parameters["perm"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs)
