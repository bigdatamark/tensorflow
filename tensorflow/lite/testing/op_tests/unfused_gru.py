# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering an unfused (built from
# primitive operations) bidirectional gated recurrent unit network. Defines and registers one
# generator function that creates parameterized graphs and passes them to the shared zip-test
# machinery using a frozen graph. Import-time side effect: registers the generator in the global
# registry of test-making functions.
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
"""Test configs for unfused_gru."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_unfused_gru_tests(options):
  # [FUNCTION DOC]
  # Generates the unfused GRU test set. The options argument is the test-run configuration
  # passed to the zip-test builder. Parameters sweep the number of units (2 or 5), batch size (1
  # or 2) and a fixed time length of 3 steps. The builder is asked to use a frozen graph.
  # Returns nothing; output is produced through the builder.
  """Make a set of tests for unfused gru op."""

  test_parameters = [{
      "units": [2, 5],
      "batch_size": [1, 2],
      "time": [3],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: one float32 placeholder per time step
    # (shaped batch size by units), forward and backward GRU cells with the given number of
    # units, and a static bidirectional RNN over the placeholders. Returns a pair: the list of
    # per-step input placeholders and the list of per-step outputs (forward and backward
    # concatenated).
    """Build the graph for unfused_gru."""
    inputs = [
        tf.compat.v1.placeholder(
            tf.float32, [parameters["batch_size"], parameters["units"]])
        for _ in range(parameters["time"])
    ]
    cell_fw = tf.compat.v1.nn.rnn_cell.GRUCell(parameters["units"])
    cell_bw = tf.compat.v1.nn.rnn_cell.GRUCell(parameters["units"])
    outputs, _, _ = tf.compat.v1.nn.static_bidirectional_rnn(
        cell_fw, cell_bw, inputs, dtype=tf.float32)

    return inputs, outputs

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: one random float32 array per time step (batch
    # size by units). Initializes all global variables in the given session (side effect: the
    # randomly initialized cell weights are fixed in the session), then evaluates the graph with
    # the arrays fed in. Returns a pair: list of input arrays and the evaluated outputs.
    """Build the inputs for unfused_gru."""
    input_values = [
        create_tensor_data(tf.float32,
                           [parameters["batch_size"], parameters["units"]])
        for _ in range(parameters["time"])
    ]
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      use_frozen_graph=True)
