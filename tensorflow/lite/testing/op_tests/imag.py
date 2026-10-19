# [MODULE DOC]
# Test configs for the imag op (imaginary part of a complex tensor). Defines and registers
# one generator, make_imag_tests, which feeds the shared test-zip builder. Import-time side
# effect: registration of the generator.
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
"""Test configs for imag op."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_imag_tests(options):
  # [FUNCTION DOC]
  # Generator for imag op tests; registered in the make-test-function registry.
  # Input: options, the test options object. Parameter set: complex64 and complex128 dtypes
  # crossed with shapes from scalar up to six dimensions. Calls the zip builder with the graph and
  # input builders. Returns nothing.
  """Make a set of tests to do imag op."""

  # Chose a set of parameters
  test_parameters = [{
      "dtype": [tf.complex64, tf.complex128],
      "input_shape": [[], [1], [2, 3], [1, 3, 4, 3], [2, 2, 3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph: one complex placeholder of the chosen dtype and shape, followed by the
    # imaginary-part op. Input: parameters mapping. Returns the list of input placeholders and the
    # list of output tensors.
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.math.imag(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Creates random complex-typed data in the range -10 to 10 matching the dtype and shape,
    # runs the session to get expected outputs. Inputs: parameters, sess, inputs, outputs.
    # Returns the list of input values and the computed outputs.
    input_values = create_tensor_data(
        parameters["dtype"].as_numpy_dtype,
        parameters["input_shape"],
        min_value=-10,
        max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
