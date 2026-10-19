# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the bitwise right-shift op on integer tensors, including broadcasting pairs.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for right_shift operator."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_right_shift_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Defines one parameter group of integer dtype (uint8, int8, uint16, int16, uint32, int32) and a pair
  # of input shapes (scalars, equal shapes, broadcastable shapes, and one incompatible pair that is
  # intended to fail). Calls make_zip_of_tests declaring that 6 generated cases (the incompatible
  # pair for each dtype) are expected to fail in TensorFlow.
  """Generate examples for right_shift."""
  test_parameters = [
      {
          "input_dtype": [
              tf.uint8,
              tf.int8,
              tf.uint16,
              tf.int16,
              tf.uint32,
              tf.int32,
          ],
          "input_shape_pair": [
              ([], []),
              ([2, 3, 4], [2, 3, 4]),
              ([1, 1, 1, 3], [1, 1, 1, 3]),
              ([5, 5], [1]),
              ([10], [2, 4, 10]),
              ([2, 3, 3], [2, 3]),  # this test case is intended to fail
          ],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests.
    # Creates two placeholders named input1 and input2 with the shared dtype and the two shapes of the
    # pair, and applies the bitwise right shift to them. Returns the list of both placeholders and the
    # list of outputs.
    """Build the right_shift testing graph."""
    input_value1 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape_pair"][0],
    )
    input_value2 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_pair"][1],
    )
    out = tf.bitwise.right_shift(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Creates random integer data of the dtype for each of the two shapes, runs the graph and returns
    # the list of both input values and the list of computed outputs.
    input_value1 = create_tensor_data(
        parameters["input_dtype"], parameters["input_shape_pair"][0]
    )
    input_value2 = create_tensor_data(
        parameters["input_dtype"], parameters["input_shape_pair"][1]
    )
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2]))
    )

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=6,
  )
