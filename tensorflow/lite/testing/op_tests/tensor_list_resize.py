# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the tensor list
# resize operation. Defines and registers one generator function that creates parameterized
# graphs and passes them, with default extra conversion options, to the shared zip-test
# machinery. Import-time side effect: registers the generator in the global registry of
# test-making functions.
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
"""Test configs for tensor_list_resize."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.ops import list_ops


@register_make_test_function()
def make_tensor_list_resize_tests(options):
  # [FUNCTION DOC]
  # Generates the tensor-list resize test set. The options argument is the test-run
  # configuration passed to the zip-test builder. Parameters sweep element type (float32,
  # int32), number of elements (4 to 6), element shape (scalar, vector of 5, 3 by 3) and new
  # size (1, 3, 5, 7, i.e. both shrinking and growing). A default extra-conversion-options
  # object is passed to the builder. Returns nothing.
  """Make a set of tests to do TensorListResize."""

  test_parameters = [
      {
          "element_dtype": [tf.float32, tf.int32],
          "num_elements": [4, 5, 6],
          "element_shape": [[], [5], [3, 3]],
          "new_size": [1, 3, 5, 7],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: a placeholder holding all elements stacked
    # along a leading dimension is converted into a tensor list, resized to the new size, and
    # stacked back into one tensor. Returns a pair: list with the data placeholder and list with
    # the stacked output.
    """Build the TensorListResize op testing graph."""
    data = tf.compat.v1.placeholder(
        dtype=parameters["element_dtype"],
        shape=[parameters["num_elements"]] + parameters["element_shape"])
    tensor_list = list_ops.tensor_list_from_tensor(data,
                                                   parameters["element_shape"])
    tensor_list = list_ops.tensor_list_resize(tensor_list,
                                              parameters["new_size"])
    out = list_ops.tensor_list_stack(
        tensor_list, element_dtype=parameters["element_dtype"])
    return [data], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random data with the element count as leading
    # dimension followed by the element shape is fed into the graph in the given session.
    # Returns a pair: list with the data array and the evaluated outputs.
    data = create_tensor_data(parameters["element_dtype"],
                              [parameters["num_elements"]] +
                              parameters["element_shape"])
    return [data], sess.run(outputs, feed_dict=dict(zip(inputs, [data])))

  extra_convert_options = ExtraConvertOptions()
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_convert_options)
