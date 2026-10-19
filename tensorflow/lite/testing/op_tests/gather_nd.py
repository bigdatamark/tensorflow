# [MODULE DOC]
# Test-generation module for the TFLite op test suite covering gather-nd (gather slices using
# multi-dimensional indices) over many dtypes, index types and index shapes, including empty
# inputs. One registered test-maker. Importing the module registers it in the global registry of
# zip_test_utils.
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
"""Test configs for gather_nd."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_gather_nd_tests(options):
  # [FUNCTION DOC]
  # Registered test-maker. Takes the test options object, defines four parameter groups (params
  # shapes with one, two or three dimensions, various index shapes, and an empty-tensor case) and
  # calls the zip generator. Returns nothing.
  """Make a set of tests to do gather_nd."""

  test_parameters = [
      {
          "params_dtype": [
              tf.float32,
              tf.int16,
              tf.int32,
              tf.int64,
              tf.string,
              tf.bool,
          ],
          "params_shape": [[5, 1]],
          "indices_dtype": [tf.int16, tf.int32, tf.int64],
          "indices_shape": [[1, 1]],
      },
      {
          "params_dtype": [
              tf.float32,
              tf.int16,
              tf.int32,
              tf.int64,
              tf.string,
              tf.bool,
          ],
          "params_shape": [[5, 5]],
          "indices_dtype": [tf.int16, tf.int32, tf.int64],
          "indices_shape": [[2, 1], [2, 2]],
      },
      {
          "params_dtype": [
              tf.float32,
              tf.int16,
              tf.int32,
              tf.int64,
              tf.string,
              tf.bool,
          ],
          "params_shape": [[5, 5, 10]],
          "indices_dtype": [tf.int16, tf.int32, tf.int64],
          "indices_shape": [[3, 1], [2, 2], [2, 3], [2, 1, 3]],
      },
      {
          "params_dtype": [tf.float32, tf.string],
          "params_shape": [[1, 0]],
          "indices_dtype": [tf.int64],
          "indices_shape": [[0, 2]],
      },
  ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Creates named placeholders for params and indices with the parameterized dtypes and shapes
    # and applies gather-nd. Returns a pair: list of both input tensors and list with the output.
    """Build the gather_nd op testing graph."""
    params = tf.compat.v1.placeholder(
        dtype=parameters["params_dtype"],
        name="params",
        shape=parameters["params_shape"])
    indices = tf.compat.v1.placeholder(
        dtype=parameters["indices_dtype"],
        name="indices",
        shape=parameters["indices_shape"])
    out = tf.gather_nd(params, indices)
    return [params, indices], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates random params and random indices between zero and the first params dimension
    # minus one (used as the bound for every index component), evaluates the graph, and returns a
    # pair: list of input values (params, indices) and list of expected outputs.
    params = create_tensor_data(parameters["params_dtype"],
                                parameters["params_shape"])
    indices = create_tensor_data(parameters["indices_dtype"],
                                 parameters["indices_shape"], 0,
                                 parameters["params_shape"][0] - 1)
    return [params, indices], sess.run(
        outputs, feed_dict=dict(zip(inputs, [params, indices])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
