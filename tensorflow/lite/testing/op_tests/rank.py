# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the rank op on float32 and int32 tensors of many shapes, including scalars and
# empty tensors.
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
"""Test configs for rank."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_rank_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object from the driver. Returns nothing.
  # Defines one parameter group of input dtype (float32, int32) and input shape (scalar, empty, and
  # several 1-D to 4-D shapes) and calls make_zip_of_tests with the nested builders.
  """Make a set of tests to do rank."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[], [0], [1, 1, 1, 3], [2, 3, 4, 5], [5, 5], [10]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback passed to make_zip_of_tests for the rank tests.
    # Creates a placeholder named input with the requested dtype and unspecified shape and applies the
    # rank op to it. Returns the list of input placeholders and the list of outputs.
    """Build the rank op testing graph."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"], name="input")
    out = tf.rank(input_value)
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback passed to make_zip_of_tests for the rank tests.
    # Creates random data of the requested dtype and shape, runs the graph in the session and returns
    # the list of input values and the list of computed outputs.
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
