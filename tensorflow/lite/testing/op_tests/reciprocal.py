# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the reciprocal op on float32, int32 and int64 tensors.
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
"""Test configs for reciprocal."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function("make_reciprocal_tests")
def make_reciprocal_tests(options):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the decorator is given an explicit name equal to the function name, which is redundant
  # but harmless.
  # Registered under the explicit name make_reciprocal_tests for the zip-test driver.
  # Input: options, the test-run options object. Returns nothing.
  # Defines one parameter group of input dtype (float32, int32, int64) and input shape (three shapes).
  # Calls make_zip_of_tests with the nested builders and declares that 6 generated cases are expected
  # to fail in TensorFlow itself (those are tolerated rather than counted as errors).
  """Make a set of tests to do reciprocal."""

  # Chose a set of parameters
  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape": [[1, 2], [1, 2, 3, 4], [10]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback passed to make_zip_of_tests for the reciprocal tests.
    # KNOWN ISSUE: the docstring says it builds the graph for cond tests, which is a copy-paste leftover;
    # it actually builds a reciprocal graph.
    # Creates a placeholder named input with the requested dtype and shape and applies the reciprocal op.
    # Returns the list of input placeholders and the list of outputs.
    """Build the graph for cond tests."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])

    out = tf.math.reciprocal(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback passed to make_zip_of_tests for the reciprocal tests.
    # Creates one random tensor of the requested dtype and shape, runs the graph in the session and
    # returns the list of input values and the list of computed outputs.
    input_values = [
        create_tensor_data(parameters["input_dtype"], parameters["input_shape"])
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=6)
