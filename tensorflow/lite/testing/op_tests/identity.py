# [MODULE DOC]
# Test configs for the identity family of ops: identity, identity_n (one and two inputs)
# and snapshot. The module defines one registered generator function (make_identity_tests) that
# is discovered through the test-function registry and used to build the TFLite op test zip.
# Import-time side effect: registers the generator under its function name.
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
"""Test configs for identity."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.ops import array_ops


@register_make_test_function()
def make_identity_tests(options):
  # [FUNCTION DOC]
  # Generator for the identity op tests; registered via the make-test-function registry.
  # Input: options, the test options object forwarded to the zip builder (output paths, flags).
  # Builds a single parameter set covering scalar, vector and matrix float32 inputs for four op
  # variants: identity, identity_n, snapshot and identity_n with two inputs. Defines the graph and
  # input builders and calls the zip-of-tests builder. Returns nothing; writes test files as a
  # side effect of the zip builder.
  """Make a set of tests to do identity."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[], [1], [3, 3]],
      "op_to_use": [
          "identity", "identity_n", "snapshot", "identity_n_with_2_inputs"
      ],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the TensorFlow graph for one parameter combination.
    # Input: parameters, a mapping with the input shape and the op variant to use.
    # Creates one float32 placeholder (two for the two-input variant), multiplies each by two (a
    # workaround so scalar shapes convert), then applies identity, snapshot or identity_n.
    # Returns a pair: the list of input placeholders and the list of output tensors.
    # An unknown variant name would leave the output list undefined and fail with a name error.
    """Make a set of tests to do identity."""

    input_tensors = []
    input_count = (2 if parameters["op_to_use"] == "identity_n_with_2_inputs"
                   else 1)
    input_tensors = [
        tf.compat.v1.placeholder(
            dtype=tf.float32, name="input", shape=parameters["input_shape"])
        for _ in range(input_count)
    ]

    # We add the Multiply before Identity just as a walk-around to make the test
    # pass when input_shape is scalar.
    # During graph transformation, converter will replace the Identity op with
    # Reshape when input has shape. However, currently converter can't
    # distinguish between missing shape and scalar shape. As a result, when
    # input has scalar shape, this conversion still fails.
    inputs_doubled = [input_tensor * 2.0 for input_tensor in input_tensors]
    if parameters["op_to_use"] == "identity":
      identity_outputs = [tf.identity(inputs_doubled[0])]
    elif parameters["op_to_use"] == "snapshot":
      identity_outputs = [array_ops.snapshot(inputs_doubled[0])]
    elif parameters["op_to_use"] in ("identity_n", "identity_n_with_2_inputs"):
      identity_outputs = tf.identity_n(inputs_doubled)
    return input_tensors, identity_outputs

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Builds concrete input values and the reference outputs for one parameter combination.
    # Inputs: parameters (mapping), sess (TensorFlow session), inputs (placeholders from the graph
    # builder), outputs (output tensors). Creates random float32 data in the range -4 to 10 for each
    # placeholder, runs the session to get expected outputs. Returns a pair: the list of input
    # values and the list of computed output values.
    input_values = [
        create_tensor_data(
            np.float32, parameters["input_shape"], min_value=-4, max_value=10)
        for _ in range(len(inputs))
    ]

    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
