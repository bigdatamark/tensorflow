# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering a squeeze followed by
# a transpose, which exercises fusion or ordering of the two operations. Defines and registers
# one generator function that creates parameterized graphs and passes them to the shared
# zip-test machinery. Import-time side effect: registers the generator in the global registry.
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
"""Test configs for squeeze_transpose."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_squeeze_transpose_tests(options):
  # [FUNCTION DOC]
  # Generates the squeeze-then-transpose test set. The options argument is the test-run
  # configuration passed to the zip-test builder. Parameters sweep element type (int32, float32,
  # int64) and the squeeze axis (last dimension given as negative or positive) over a fixed
  # four-dimensional input whose first and last dimensions are one. No TensorFlow failures are
  # expected. Returns nothing; output is produced through the builder.
  """Make a set of tests to do squeeze followed by transpose."""

  test_parameters = [{
      "dtype": [tf.int32, tf.float32, tf.int64],
      "input_shape": [[1, 4, 10, 1]],
      "axis": [[-1], [3]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: a placeholder named input, squeezed over
    # the given axis, then transposed with the permutation swapping the two remaining
    # dimensions. Returns a pair: list with the input placeholder and list with the final output.
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.squeeze(input_tensor, axis=parameters["axis"])
    out = tf.transpose(a=out, perm=[1, 2])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random data of the parameter element type and
    # shape is fed into the graph in the given session. Returns a pair: list with the input array
    # and the evaluated outputs.
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=0)
