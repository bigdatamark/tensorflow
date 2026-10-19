# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of the complex absolute-value op.
# Defines one registered test-maker covering complex64 (float32 result) and complex128 (float64
# result) inputs over scalar and ranked shapes. Importing registers the maker in the global
# registry.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for complex abs."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_complex_abs_tests(options):
  # [FUNCTION DOC]
  # Registered test maker taking the generator options object. Declares two parameter sets pairing
  # each complex dtype with its matching real output type, each over five shapes, and calls the
  # zip-test generator with default settings. Returns nothing.
  """Make a set of tests to do complex abs."""

  # Chose a set of parameters
  test_parameters = [{
      "dtype": [tf.complex64],
      "input_shape": [[], [1], [2, 3], [1, 3, 4, 3], [2, 2, 3, 4, 5, 6]],
      "Tout": [tf.float32]
  }, {
      "dtype": [tf.complex128],
      "input_shape": [[], [1], [2, 3], [1, 3, 4, 3], [2, 2, 3, 4, 5, 6]],
      "Tout": [tf.float64]
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the complex-abs zip tests.
    # Creates a placeholder named input with the complex dtype and shape and applies the raw
    # complex-abs op with the requested real output type. Returns the list with the placeholder and
    # the list with the output tensor.
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.raw_ops.ComplexAbs(x=input_tensor, Tout=parameters["Tout"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the complex-abs zip tests.
    # Generates random complex data with parts in the range minus ten to ten for the given shape,
    # runs the session with it, and returns the list with the data and the expected outputs.
    input_values = create_tensor_data(
        parameters["dtype"].as_numpy_dtype,
        parameters["input_shape"],
        min_value=-10,
        max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
