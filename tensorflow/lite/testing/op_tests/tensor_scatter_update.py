# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the tensor scatter
# update operation (overwriting scalar entries at given positions of a tensor). Defines and
# registers one generator function that creates parameterized graphs and passes them to the
# shared zip-test machinery. Import-time side effect: registers the generator in the global
# registry of test-making functions.
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
"""Test configs for TensorScatterUpdate."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_tensor_scatter_update_tests(options):
  # [FUNCTION DOC]
  # Generates the tensor scatter update test set. The options argument is the test-run
  # configuration passed to the zip-test builder. Parameters sweep input type (float32, int32,
  # int64, bool), input shape (a vector of 14 or a 2 by 4 by 7 tensor) and the number of updates
  # (1, 3 or 5). Returns nothing; output is produced through the builder.
  """Make a set of tests to do tensor_scatter_update."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64, tf.bool],
      "input_shape": [[14], [2, 4, 7]],
      "updates_count": [1, 3, 5],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: three placeholders, the input tensor
    # (given type and shape), a 32-bit integer indices matrix with one row per update and one
    # column per input dimension, and a vector of scalar replacement values (one per row).
    # Applies the scatter-update operation. Returns a pair: list of the three placeholders in
    # that order and list with the output.
    """Build the tensor_scatter_update op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    # The indices will be a list of "input_shape".
    indices_tensor = tf.compat.v1.placeholder(
        dtype=tf.int32,
        name="indices",
        shape=([parameters["updates_count"],
                len(parameters["input_shape"])]))
    # The updates will be a list of scalar, shaped of "updates_count".
    updates_tensors = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="updates",
        shape=[parameters["updates_count"]])

    out = tf.tensor_scatter_nd_update(input_tensor, indices_tensor,
                                      updates_tensors)
    return [input_tensor, indices_tensor, updates_tensors], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs. Draws random distinct index tuples (one random
    # in-range coordinate per input dimension) until the requested count is reached. Creates
    # random input data, the indices as a 32-bit integer array, and random update values in the
    # range -3 to 3 (of the input type, including boolean). Evaluates the graph in the given
    # session. Returns a pair: list of the three arrays (input, indices, updates) and the
    # evaluated outputs. Uses the global random generator.
    indices = set()
    while len(indices) < parameters["updates_count"]:
      loc = []
      for d in parameters["input_shape"]:
        loc.append(np.random.randint(0, d))
      indices.add(tuple(loc))

    values = [
        create_tensor_data(parameters["input_dtype"],
                           parameters["input_shape"]),
        np.array(list(indices), dtype=np.int32),
        create_tensor_data(
            parameters["input_dtype"],
            parameters["updates_count"],
            min_value=-3,
            max_value=3)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
