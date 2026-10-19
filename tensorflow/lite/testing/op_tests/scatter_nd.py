# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the scatter_nd op with indices, updates and output shape all fed as inputs.
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
"""Test configs for scatter_nd."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_scatter_nd_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Defines two parameter groups: 1-D scatter (4 indices into a length-8 output) and 2-D indices
  # scattering slices into a 2x3x5 output. Each lists dtype and shape of indices, updates and output
  # shape tensor, plus the concrete indices and shape values. Update dtypes are int32, int64, float32
  # and bool. Calls make_zip_of_tests with the nested builders.
  """Make a set of tests to do scatter_nd."""

  test_parameters = [{
      "indices_dtype": [tf.int32],
      "indices_shape": [[4, 1]],
      "indices_value": [[[4], [3], [1], [7]]],
      "updates_dtype": [tf.int32, tf.int64, tf.float32, tf.bool],
      "updates_shape": [[4]],
      "shape_dtype": [tf.int32],
      "shape_shape": [[1]],
      "shape_value": [[8]]
  }, {
      "indices_dtype": [tf.int32],
      "indices_shape": [[4, 2]],
      "indices_value": [[[0, 0], [1, 0], [0, 2], [1, 2]]],
      "updates_dtype": [tf.int32, tf.int64, tf.float32, tf.bool],
      "updates_shape": [[4, 5]],
      "shape_dtype": [tf.int32],
      "shape_shape": [[3]],
      "shape_value": [[2, 3, 5]]
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests.
    # Creates three placeholders named indices, updates and shape with the dtypes and shapes from the
    # parameters and applies the scatter_nd op. Returns the list of the three placeholders and the list
    # of outputs.
    """Build the scatter_nd op testing graph."""
    indices = tf.compat.v1.placeholder(
        dtype=parameters["indices_dtype"],
        name="indices",
        shape=parameters["indices_shape"])
    updates = tf.compat.v1.placeholder(
        dtype=parameters["updates_dtype"],
        name="updates",
        shape=parameters["updates_shape"])
    shape = tf.compat.v1.placeholder(
        dtype=parameters["shape_dtype"],
        name="shape",
        shape=parameters["shape_shape"])
    out = tf.scatter_nd(indices, updates, shape)
    return [indices, updates, shape], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Builds the indices and the output shape arrays from the fixed values in the parameters and random
    # updates data of the requested dtype and shape. Runs the graph and returns the list of the three
    # input values and the list of computed outputs.
    indices = np.array(parameters["indices_value"])
    updates = create_tensor_data(parameters["updates_dtype"],
                                 parameters["updates_shape"])
    shape = np.array(parameters["shape_value"])
    return [indices, updates, shape], sess.run(
        outputs, feed_dict=dict(zip(inputs, [indices, updates, shape])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
