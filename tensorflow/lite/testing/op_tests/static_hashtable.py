# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering static hash tables
# (key/value lookup tables built from constant tensors). Defines and registers one generator
# function that creates parameterized graphs and passes them, together with default extra
# conversion options, to the shared zip-test machinery. Import-time side effect: registers the
# generator in the global registry of test-making functions under an explicit name.
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
"""Test configs for static hashtable."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function("make_static_hashtable_tests")
def make_static_hashtable_tests(options):
  # [FUNCTION DOC]
  # Generates the static hash table test set. The options argument is the test-run configuration
  # passed to the zip-test builder. Parameters choose a table definition (key type, value type,
  # keys, values, default value): either string keys to 64-bit integer values, or 64-bit integer
  # keys to string values; and an input shape (scalar, or vectors of length 1, 3 or 10). A
  # default-constructed extra-conversion-options object is passed to the builder. Returns nothing.
  """Make a set of tests to use static hashtable."""

  # Chose a set of parameters
  test_parameters = [{
      "table": [(tf.string, tf.int64, ["1", "2", "3"], [4, 5, 6], -1),
                (tf.int64, tf.string, [1, 2, 3], ["4", "5", "6"], "-1")],
      "input_shape": [[], [3], [1], [10]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the looked-up keys are the constant key tensor, not the input placeholder, so
    # the output does not depend on the graph input; the input only exists as a graph input.
    # Builds the graph for one parameter combination: constant key and value tensors feed a
    # key-value initializer and a static hash table with the given default. Under a dependency on
    # the table initializer, creates a placeholder named input (key type, given shape) and looks
    # up the constant keys. Returns a pair: list with the input placeholder and list with the
    # lookup output.
    """Build the graph for static hashtable tests."""
    (key_dtype, value_dtype, keys, values, default_value) = parameters["table"]

    key_tensor = tf.constant(keys, dtype=key_dtype)
    value_tensor = tf.constant(values, dtype=value_dtype)

    initializer = tf.lookup.KeyValueTensorInitializer(key_tensor, value_tensor)
    table = tf.lookup.StaticHashTable(initializer, default_value)

    with tf.control_dependencies(
        [tf.compat.v1.initializers.tables_initializer()]):
      input_value = tf.compat.v1.placeholder(
          dtype=key_dtype, name="input", shape=parameters["input_shape"])
      out = table.lookup(key_tensor)
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random data of the table's key type and the
    # parameter input shape is fed into the graph in the given session. Returns a pair: list with
    # the input array and the evaluated outputs.
    (key_dtype, _, _, _, _) = parameters["table"]
    input_values = [create_tensor_data(key_dtype, parameters["input_shape"])]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  extra_convert_options = ExtraConvertOptions()
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_convert_options)
