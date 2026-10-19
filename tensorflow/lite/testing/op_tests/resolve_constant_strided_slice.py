# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one is a regression test: a constant tensor is sliced with the strided-slice syntax and added
# to a placeholder, to check the converter resolves a constant strided slice correctly.
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
"""Test configs for resolve_constant_strided_slice."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_resolve_constant_strided_slice_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # KNOWN ISSUE: the docstring says the tests show that strided_slice yields incorrect results; the
  # code simply builds a regression graph and does not assert incorrectness itself.
  # Input: options, the test-run options object. Returns nothing. Uses a single dummy parameter group
  # (an unused iteration counter) so exactly one case is generated, then calls make_zip_of_tests.
  """Make a set of tests to show strided_slice yields incorrect results."""

  test_parameters = [{
      "unused_iteration_counter": [1],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests. The parameters are ignored.
    # Creates a float32 placeholder of shape 4 by 2 and a constant 4 by 4 float32 tensor holding the
    # numbers 0 to 15. The output is the placeholder plus the first two columns of the constant (taken
    # with a strided slice). Returns the list with the placeholder and the list with the output.
    """Build the strided_slice op testing graph."""
    del parameters
    input_values = tf.compat.v1.placeholder(dtype=tf.float32, shape=[4, 2])
    data = tf.constant(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        tf.float32)
    return [input_values], [input_values + data[:, :2]]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests. The parameters are ignored.
    # Feeds a 4 by 2 float32 array of zeros to the first input, runs the graph and returns the list
    # of input values and the list of computed outputs.
    del parameters
    input_values = np.zeros([4, 2], dtype=np.float32)
    return [input_values], sess.run(
        outputs, feed_dict={inputs[0]: input_values})

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
