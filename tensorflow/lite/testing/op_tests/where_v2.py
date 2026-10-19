# [MODULE DOC]
# Test-configuration module for the "where_v2" op of the TFLite converter test suite. It defines one
# registered generator that builds conversion tests, covering condition-only (index) form and the
# three-argument select form with broadcasting. Registration happens at import time via a decorator.
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
"""Test configs for where_v2."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_where_v2_tests(options):
  # [FUNCTION DOC]
  # Registered test generator for where_v2. Input: options, the test-generation options object.
  # Builds parameter sets over condition dtypes (float, bool, ints), condition shapes, value dtypes
  # (null means condition-only where) and broadcasting value shapes, including a 5D case with an
  # unknown dimension. Returns nothing; asks the zip builder to produce the tests with two expected
  # TensorFlow failures. Defines nested helpers for graph building, shape fixing and input building.
  """Make a set of tests to do where_v2."""

  test_parameters = [
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[1, 2, 3, 4]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([1, 2, 3, 4], [1, 1, 1, 1]),],
      },
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[2], [1]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([2, 1, 2, 1], [2, 1, 2, 1]),],
      },
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[1, 4, 2]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([1, 3, 4, 2], [1, 3, 4, 2]),],
      },
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[1, 2]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([1, 2, 2], [1, 2, 2]),],
      },
      {
          "condition_dtype": [tf.bool],
          "input_condition_shape": [[1, 1]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([1, 1, 2, 2], [1, 1, 2, 2]),],
      },
      {
          "condition_dtype": [tf.bool],
          "input_condition_shape": [[4]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([4, 4], [4, 4]),],
      },
      {
          "condition_dtype": [tf.bool],
          "input_condition_shape": [[2]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([2, 3], [2, 3]),],
      },
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[1, 2], None],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 2, 2], [1, 2]),],
      },
      # Requires kernel supporting broadcasting for 5D case.
      {
          "condition_dtype": [tf.float32],
          "input_condition_shape": [[1, 1, 1, 1, 1]],
          "input_dtype": [tf.float32],
          "input_shape_set": [([], [1, None, 1, 2, 512])],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the where_v2 test generator.
    # KNOWN ISSUE: it mutates the shared parameters mapping, forcing the condition dtype to bool when
    # a non-bool condition is combined with explicit x and y values.
    # Creates a condition placeholder, optionally x and y placeholders (only when the value dtype is
    # given), and applies the v2 where op. Returns inputs (including null entries for absent x and y)
    # and a list with the single output.
    """Build the where op testing graph."""
    # To actually use where op, x, y params to where_v2 needs to be None.
    # This is needed when type is not bool, so we actually use where op.
    if parameters["condition_dtype"] != tf.bool and parameters[
        "input_dtype"] is not None:
      parameters["condition_dtype"] = tf.bool
    input_condition = tf.compat.v1.placeholder(
        dtype=parameters["condition_dtype"],
        name="input_condition",
        shape=parameters["input_condition_shape"])
    input_value1 = None
    input_value2 = None
    if parameters["input_dtype"] is not None:
      input_value1 = tf.compat.v1.placeholder(
          dtype=parameters["input_dtype"],
          name="input_x",
          shape=parameters["input_shape_set"][0])
      input_value2 = tf.compat.v1.placeholder(
          dtype=parameters["input_dtype"],
          name="input_y",
          shape=parameters["input_shape_set"][1])
    out = tf.compat.v2.where(input_condition, input_value1, input_value2)
    return [input_condition, input_value1, input_value2], [out]

  def build_input_shape(input_shape):
    # [FUNCTION DOC]
    # HELPER: turns a shape with unknown dimensions into a concrete one.
    # Input: list of dimensions, some null. Returns a list where each null dimension becomes one.
    return [1 if v is None else v for v in input_shape]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-data callback for the where_v2 test generator.
    # Creates random data for the condition and, when a value dtype is given, for x and y with
    # concrete shapes. Runs the graph in the session, feeding all three inputs or only the condition,
    # and returns the list of input values and the reference outputs.
    input_condition = create_tensor_data(parameters["condition_dtype"],
                                         parameters["input_condition_shape"])
    input_value1 = None
    input_value2 = None
    if parameters["input_dtype"] is not None:
      input_value1 = create_tensor_data(
          parameters["input_dtype"],
          build_input_shape(parameters["input_shape_set"][0]))
      input_value2 = create_tensor_data(
          parameters["input_dtype"],
          build_input_shape(parameters["input_shape_set"][1]))
      return [input_condition, input_value1, input_value2], sess.run(
          outputs,
          feed_dict=dict(
              zip(inputs, [input_condition, input_value1, input_value2])))
    else:
      return [input_condition, input_value1, input_value2], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_condition])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=2)
