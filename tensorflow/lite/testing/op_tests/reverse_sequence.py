# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the reverse-sequence op with constant sequence lengths.
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
"""Test configs for reverse_sequence."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_reverse_sequence_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Defines three parameter groups of dtype (float32, int32, int64), input shape (2-D to 5-D),
  # per-batch sequence lengths (including zero lengths), sequence axis and batch axis, then calls
  # make_zip_of_tests with the nested builders.
  """Make a set of tests to do reverse_sequence."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape": [[8, 4, 5, 5, 6], [4, 4, 3, 5]],
      "seq_lengths": [[2, 2, 2, 2], [2, 1, 1, 0]],
      "seq_axis": [0, 3],
      "batch_axis": [1]
  }, {
      "input_dtype": [tf.float32],
      "input_shape": [[2, 4, 5, 5, 6]],
      "seq_lengths": [[2, 1]],
      "seq_axis": [2],
      "batch_axis": [0]
  }, {
      "input_dtype": [tf.float32],
      "input_shape": [[4, 2]],
      "seq_lengths": [[3, 1]],
      "seq_axis": [0],
      "batch_axis": [1]
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests.
    # Creates a placeholder named input of the requested dtype and shape and applies the reverse-sequence
    # op with the constant sequence lengths and the batch and sequence axes. Returns the list of input
    # placeholders and the list of outputs.
    """Build the graph for reverse_sequence tests."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    outs = tf.reverse_sequence(
        input=input_value,
        seq_lengths=parameters["seq_lengths"],
        batch_axis=parameters["batch_axis"],
        seq_axis=parameters["seq_axis"])
    return [input_value], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Creates random data of the requested dtype and shape, runs the graph in the session and returns
    # the list of input values and the list of computed outputs.
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
