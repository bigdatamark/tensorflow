# [MODULE DOC]
# Test-configuration module for the "where" op of the TFLite converter test suite. It defines one
# registered generator that builds a zip of conversion tests comparing TensorFlow and TFLite results.
# Registration with the shared test-function registry happens at import time via a decorator.
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
"""Test configs for where."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_where_tests(options):
  # [FUNCTION DOC]
  # Registered test generator for where (select between two tensors by a less-than condition).
  # Input: options, the test-generation options object forwarded to the zip builder.
  # Builds a list of parameter sets: float32/int32 dtypes, equal shapes, scalars, broadcasting up to
  # 8 dimensions with optional unknown (dynamic) dimensions, v1 vs v2 where, and fully-quantized
  # variants. Returns nothing; as a side effect it asks the zip builder to generate the test archive
  # with four expected TensorFlow failures. Contains nested helpers for graph and input building.
  """Make a set of tests to do where."""

  test_parameters = [
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 2, 3, 4], [1, 2, 3, 4]),],
          "use_where_v2": [False, True],
          "fully_quantize": [False],
      },
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([], []),],
          "use_where_v2": [],
          "fully_quantize": [False],
      },
      {
          "input_dtype": [tf.float32],
          "input_shape_set": [
              ([1, 2, 3, 4], [1, 2, 3, 4]),
              ([], []),
          ],
          "use_where_v2": [False, True],
          "fully_quantize": [True],
      },
      # High dimension broadcasting support in MLIR converter.
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([8, 7, 6, 5, 4, 3, 2, 1], [4, 3, 2, 1]),
                              ([8, 7, 6, 5, 4, 3, 2, 1], [None, 3, 2, 1]),
                              ([8, 7, 6, 5, None, 3, 2, 1], [None, 3, 2, 1])],
          "use_where_v2": [True],
          "fully_quantize": [False],
          "dynamic_size_value": [4, 1],
      },
      {
          "input_dtype": [tf.float32],
          "input_shape_set": [([8, 7, 6, 5, 4, 3, 2, 1], [4, 3, 2, 1])],
          "use_where_v2": [True],
          "fully_quantize": [True],
          "dynamic_size_value": [4],
      },
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([], []), ([1], []), ([], [1])],
          "use_where_v2": [False, True],
          "fully_quantize": [False],
      },
  ]

  def populate_dynamic_shape(parameters, input_shape):
    # [FUNCTION DOC]
    # HELPER: replaces unknown dimensions in a shape for the where test generator.
    # Inputs: parameters (mapping with the dynamic size value) and input_shape (list of dimensions,
    # some may be null). Returns a new list where each null dimension is replaced by the dynamic size value.
    return [
        parameters["dynamic_size_value"] if x is None else x
        for x in input_shape
    ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the where test generator.
    # Creates two placeholders (names input2 and input3) of the given dtype and shapes, computes an
    # elementwise less-than condition, and applies either the v2 or v1 where op according to the
    # use_where_v2 parameter. Returns the list of input placeholders and the list with the one output.
    """Build the where op testing graph."""
    input_value1 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_set"][0])
    input_value2 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input3",
        shape=parameters["input_shape_set"][1])
    less = tf.less(input_value1, input_value2)
    where = tf.compat.v2.where if parameters[
        "use_where_v2"] else tf.compat.v1.where
    out = where(less, input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-data callback for the where test generator.
    # Resolves dynamic dimensions, creates two random tensors in the range minus one to one, runs the
    # graph in the session, and returns the input values plus the computed reference outputs.
    input_shape_1 = populate_dynamic_shape(parameters,
                                           parameters["input_shape_set"][0])
    input_shape_2 = populate_dynamic_shape(parameters,
                                           parameters["input_shape_set"][1])

    input_value1 = create_tensor_data(
        parameters["input_dtype"], input_shape_1, min_value=-1, max_value=1)
    input_value2 = create_tensor_data(
        parameters["input_dtype"], input_shape_2, min_value=-1, max_value=1)
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=4)
