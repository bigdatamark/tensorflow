# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering tensor lists whose
# element shape is not known when the list is created (filled inside a while loop). Defines and
# registers one generator function that creates parameterized graphs and passes them to the
# shared zip-test machinery. Import-time side effect: registers the generator in the global
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
"""Test configs for tensor_list_dynamic_shape."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.ops import list_ops


@register_make_test_function()
def make_tensor_list_dynamic_shape_tests(options):
  # [FUNCTION DOC]
  # Generates the dynamic-shape tensor-list test set. The options argument is the test-run
  # configuration passed to the zip-test builder. Parameters sweep element type (float32,
  # int32), number of elements (4 to 6) and element shape (scalar, vector of 5, or 3 by 3).
  # Returns nothing; output is produced through the builder.
  """Make a set of tests for tensorlists with dynamic shape."""

  test_parameters = [
      {
          "element_dtype": [tf.float32, tf.int32],
          "num_elements": [4, 5, 6],
          "element_shape": [[], [5], [3, 3]],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring names the set-item op, but the graph is a while-loop that fills
    # a reserved tensor list (unknown element shape) with set-item calls and then stacks it.
    # Builds the graph for one parameter combination: a placeholder item of the element shape;
    # a reserved tensor list with the given number of elements and unspecified element shape; a
    # loop that, for each index below the element count, stores item doubled plus one at that
    # index; finally the list is stacked into one tensor. Returns a pair: list with the item
    # placeholder and list with the stacked output.
    """Build the TensorListSetItem op testing graph."""
    item = tf.compat.v1.placeholder(
        dtype=parameters["element_dtype"], shape=parameters["element_shape"])
    tensor_list = list_ops.tensor_list_reserve(
        element_shape=None,
        num_elements=parameters["num_elements"],
        element_dtype=parameters["element_dtype"])

    init_state = (0, tensor_list)
    condition = lambda i, _: i < parameters["num_elements"]

    def loop_body(i, tensor_list):
      # [FUNCTION DOC]
      # HELPER: loop body for the while loop of the dynamic-shape tensor-list graph.
      # Takes the loop counter and the current tensor list. Computes the captured item doubled
      # plus the constant one (in the element type), stores it into the list at the counter
      # position, and returns the incremented counter and the updated list.
      new_item = tf.add(
          tf.add(item, item),
          tf.constant(value=1, dtype=parameters["element_dtype"]))
      new_list = list_ops.tensor_list_set_item(tensor_list, i, new_item)
      return i + 1, new_list

    _, tensor_list = tf.while_loop(
        cond=condition, body=loop_body, loop_vars=init_state)
    out = list_ops.tensor_list_stack(
        tensor_list,
        num_elements=parameters["num_elements"],
        element_dtype=parameters["element_dtype"])
    return [item], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random data of the element type and element
    # shape is fed into the graph in the given session. Returns a pair: list with the item array
    # and the evaluated outputs.
    item = create_tensor_data(parameters["element_dtype"],
                              parameters["element_shape"])
    return [item], sess.run(outputs, feed_dict=dict(zip(inputs, [item])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
