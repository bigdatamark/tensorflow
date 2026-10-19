# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the shape op. A reshape to a fed shape is placed before it so it is not constant folded.
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
"""Test configs for shape."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_shape_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Defines one parameter group of input dtype (float32, int32), input shape of 1 by 4, a new shape
  # with the same element count, and the output type (int32 or int64). Calls make_zip_of_tests.
  """Make a set of tests to do shape."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 4]],
      "new_shape": [[1, 4], [4, 1], [2, 2]],
      "out_type": [tf.int32, tf.int64],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests.
    # Creates an input placeholder and an int32 placeholder named new_shape, reshapes the input to it,
    # and takes the shape of the result with the requested output type. The reshape keeps the shape op
    # from being optimized away during conversion. Returns the list of both placeholders and the list
    # of outputs.
    """Build the shape op testing graph."""
    # Note that we intentionally leave out the shape from the input placeholder
    # to prevent the Shape operation from being optimized out during conversion.
    # TODO(haoliang): Test shape op directly after we have better support for
    # dynamic input. Currently we need to introduce a Reshape op to prevent
    # shape being constant-folded.
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        shape=parameters["input_shape"],
        name="input")
    shape_of_new_shape = [len(parameters["new_shape"])]
    new_shape = tf.compat.v1.placeholder(
        dtype=tf.int32, shape=shape_of_new_shape, name="new_shape")
    reshaped = tf.reshape(input_value, shape=new_shape)
    out = tf.shape(input=reshaped, out_type=parameters["out_type"])
    return [input_value, new_shape], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Creates random input data and an array from the new shape parameter, runs the graph and returns
    # the list of both input values and the list of computed outputs.
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    new_shape = np.array(parameters["new_shape"])
    return [input_value, new_shape], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value, new_shape])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
