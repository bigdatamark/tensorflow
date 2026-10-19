# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering embedding lookup. One registered
# test-maker uses float parameters and int32 ids. Importing the module registers it in the global
# registry of zip_test_utils.
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
"""Test configs for embedding_lookup."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_embedding_lookup_tests(options):
  # [FUNCTION DOC]
  # KNOWN ISSUE: docstring says the tests do gather, but the op under test is embedding lookup.
  # Registered test-maker. Takes the test options object, defines a grid over table shape (one or
  # two dimensions) and ids shape, and calls the zip generator. Returns nothing.
  """Make a set of tests to do gather."""

  test_parameters = [
      {
          "params_dtype": [tf.float32],
          "params_shape": [[10], [10, 10]],
          "ids_dtype": [tf.int32],
          "ids_shape": [[3], [5]],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: docstring says gather; the op built is embedding lookup.
    # Creates two named placeholders (params table and ids) and looks the ids up in the table.
    # Returns a pair: list of the two input tensors and list with the output.
    """Build the gather op testing graph."""
    params = tf.compat.v1.placeholder(
        dtype=parameters["params_dtype"],
        name="params",
        shape=parameters["params_shape"])
    ids = tf.compat.v1.placeholder(
        dtype=parameters["ids_dtype"],
        name="ids",
        shape=parameters["ids_shape"])
    out = tf.nn.embedding_lookup(params=params, ids=ids)
    return [params, ids], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random table data and random integer ids between zero and the table size minus
    # one (so ids are valid), evaluates the graph, and returns a pair: list of input values
    # (params, ids) and list of expected outputs.
    params = create_tensor_data(parameters["params_dtype"],
                                parameters["params_shape"])
    ids = create_tensor_data(parameters["ids_dtype"], parameters["ids_shape"],
                             0, parameters["params_shape"][0] - 1)
    return [params, ids], sess.run(
        outputs, feed_dict=dict(zip(inputs, [params, ids])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
