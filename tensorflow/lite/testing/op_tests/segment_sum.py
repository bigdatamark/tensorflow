# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the segment_sum op with constant segment ids.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for segment_sum."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_segment_sum_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Defines one parameter group of data shape (1-D to 3-D), data dtype (float32, int32) and segment id
  # lists of length 4. Calls make_zip_of_tests with no expected TensorFlow failures.
  """Make a set of tests to do segment_sum."""

  test_parameters = [
      {
          "data_shape": [[4, 4], [4], [4, 3, 2]],
          "data_dtype": [tf.float32, tf.int32],
          "segment_ids": [[0, 0, 1, 1], [0, 1, 2, 2], [0, 1, 2, 3],
                          [0, 0, 0, 0]],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests.
    # Creates a placeholder named data and a constant int32 segment ids tensor from the parameters and
    # applies segment_sum. Returns the list holding the data placeholder and the list of outputs.
    """Build the segment_sum op testing graph."""
    data = tf.compat.v1.placeholder(
        dtype=parameters["data_dtype"],
        name="data",
        shape=parameters["data_shape"])
    segment_ids = tf.constant(parameters["segment_ids"], dtype=tf.int32)
    out = tf.math.segment_sum(data, segment_ids)
    return [data], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Creates random data of the requested dtype and shape, runs the graph and returns the list of
    # input values and the list of computed outputs.
    data = create_tensor_data(parameters["data_dtype"],
                              parameters["data_shape"])
    return [data], sess.run(outputs, feed_dict=dict(zip(inputs, [data])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=0)
