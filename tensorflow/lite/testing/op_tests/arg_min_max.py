# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of the arg-max and arg-min ops.
# It defines one registered test-maker function that builds many small TensorFlow graphs
# (varying dtype, shape, output index type, axis choice and quantization flags) and hands them to
# the shared zip-test machinery. Importing it registers the maker in the global test registry.
# Not a library for general use; it is consumed by the TFLite op test generator.
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
"""Test configs for arg_min_max."""
import random

import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_arg_min_max_tests(options):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring mentions only arg_max, but the generated tests cover both arg-max
  # and arg-min (selected by the is_arg_max parameter).
  # Registered test maker. Takes the generator options object. Declares parameter sets: float32/int32
  # inputs of assorted shapes with a random non-last axis (arg-max only), float32/int32 inputs with
  # the last axis for both arg-max and arg-min, and boolean inputs (arg-max, random axis); output
  # index type is 32- or 64-bit integer. Then calls the zip-test generator with the graph builder
  # and input builder, expecting 8 failures in TensorFlow itself. Returns nothing; output files are
  # written by the generator.
  """Make a set of tests to do arg_max."""

  test_parameters = [
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape": [[], [1, 1, 1, 3], [2, 3, 4, 5], [2, 3, 3], [5, 5],
                          [10]],
          "output_type": [tf.int32, tf.int64],
          "is_arg_max": [True],
          "is_last_axis": [False],
          "dynamic_range_quantize": [False, True],
      },
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                          [2, 10], [3, 4, 50], [2, 3, 5, 100]],
          "output_type": [tf.int32, tf.int64],
          "is_arg_max": [False, True],
          "is_last_axis": [True],
          "dynamic_range_quantize": [False, True],
      },
      {
          "input_dtype": [tf.bool],
          "input_shape": [[1, 1, 1, 3], [2, 3, 4, 5], [2, 3, 3], [5, 5], [10]],
          "output_type": [tf.int32, tf.int64],
          "is_arg_max": [True],
          "is_last_axis": [False],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the arg min/max zip tests.
    # KNOWN ISSUE: the docstring says "topk" but the graph contains an arg-max or arg-min op.
    # Input is one parameter set (mapping of names to values). Creates a placeholder named input with
    # the given dtype and shape. Axis is the last one if is_last_axis is set, otherwise a random axis
    # between 0 and the highest dimension (0 for scalars; uses the random module, so nondeterministic).
    # Applies arg-max or arg-min with the requested output type. Returns the list of input
    # placeholders and the list of output tensors.
    """Build the topk op testing graph."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    if not parameters["is_last_axis"]:
      axis = random.randint(0, max(len(parameters["input_shape"]) - 1, 0))
    else:
      axis = -1
    if parameters["is_arg_max"]:
      out = tf.math.argmax(
          input=input_value, axis=axis, output_type=parameters["output_type"])
    else:
      out = tf.math.argmin(
          input=input_value, axis=axis, output_type=parameters["output_type"])
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the arg min/max zip tests.
    # Generates random tensor data of the parameterized dtype and shape, runs the session on the
    # output tensors with that data fed to the input placeholder, and returns the list of input
    # values plus the computed expected outputs.
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=8)
