# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of conditional execution (cond). Defines
# one registered test-maker, run with control-flow v2 enabled, that selects between two inputs by
# a boolean predicate, for float and string dtypes. Importing registers the maker in the global
# registry.
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
"""Test configs for cond."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util


@register_make_test_function("make_cond_tests")
@test_util.enable_control_flow_v2
def make_cond_tests(options):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says relu1 (copy-paste); the tests exercise a conditional between
  # two inputs, not relu1.
  # Registered test maker (decorated to enable control-flow v2) taking the generator options object.
  # Declares one parameter set crossing float32 and string dtypes with predicate false and true (the
  # string case also guards handling of branch subgraphs with dynamically allocated tensors) and
  # calls the zip-test generator. Returns nothing.
  """Make a set of tests to do relu1."""

  # Chose a set of parameters
  test_parameters = [{
      # Note: The `tf.string` test case also serves as a regression test to
      # ensure that branch subgraph with dynamically allocated inputs/outputs
      # are handled correctly.
      "dtype": [tf.float32, tf.string],
      "pred": [False, True],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the cond zip tests.
    # Creates two one-element placeholders of the parameterized dtype and a one-element boolean
    # placeholder reshaped to a scalar (workaround: the converter cannot handle scalar inputs). Builds
    # a conditional whose true branch returns the first input and false branch the second. Returns the
    # list of three placeholders (first, second, predicate) and the list with the output tensor.
    """Build the graph for cond tests."""
    input1 = tf.compat.v1.placeholder(dtype=parameters["dtype"], shape=(1,))
    input2 = tf.compat.v1.placeholder(dtype=parameters["dtype"], shape=(1,))
    # MLIR TFLite converter can't handle scalar inputs. This is a workaround
    # to input (1,) tensors and then reshape to scalar.
    # TODO(b/129003347): Remove the workaround after scalar inputs are
    # supported.
    pred = tf.compat.v1.placeholder(dtype=tf.bool, shape=(1,))
    pred_scalar = tf.reshape(pred, ())

    out = tf.cond(
        pred=pred_scalar, true_fn=lambda: input1, false_fn=lambda: input2)
    return [input1, input2, pred], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the cond zip tests.
    # Generates random one-element data for the two branch inputs and a one-element boolean array
    # holding the parameterized predicate, runs the session, and returns the list of the three
    # values with the expected outputs.
    input_values = [
        create_tensor_data(parameters["dtype"], (1,)),
        create_tensor_data(parameters["dtype"], (1,)),
        np.array([parameters["pred"]], dtype=np.bool_),
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
