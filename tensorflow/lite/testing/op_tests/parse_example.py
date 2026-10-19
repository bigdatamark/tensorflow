# [MODULE DOC]
# Test configs for parse_example (decoding serialized example records). Contains a helper that
# creates serialized example data and the registered generator make_parse_example_tests.
# Import-time side effect: registration of the generator under an explicit name.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for parse example."""
import string

import numpy as np
import tensorflow as tf

from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def create_example_data(feature_dtype, feature_shape):
  # [FUNCTION DOC]
  # HELPER: creates serialized example input data for the parse_example tests.
  # Inputs: feature_dtype (TensorFlow dtype: float types, integer types or string), feature_shape
  # (list of sizes; only the first size is used for strings). Builds one example with a feature
  # named x holding random floats, random integers between -100 and 100, or the same random ten
  # letter byte string repeated. Returns an array holding one serialized example.
  """Create structured example data."""
  features = {}
  if feature_dtype in (tf.float32, tf.float16, tf.float64):
    data = np.random.rand(*feature_shape)
    features["x"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(data)))
  elif feature_dtype in (tf.int32, tf.uint8, tf.int64, tf.int16):
    data = np.random.randint(-100, 100, size=feature_shape)
    features["x"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(data)))
  elif feature_dtype == tf.string:
    letters = list(string.ascii_uppercase)
    data = "".join(np.random.choice(letters, size=10)).encode("utf-8")
    features["x"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[data]*feature_shape[0]))
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return np.array([example.SerializeToString()])


@register_make_test_function("make_parse_example_tests")
def make_parse_example_tests(options):
  # [FUNCTION DOC]
  # Generator for parse_example tests. Input: options, test options object. Parameters: feature
  # dtype (string, float32, int64), dense or sparse feature, feature length. Enables custom ops in
  # the extra conversion options and calls the zip builder. Returns nothing.
  """Make a set of tests to use parse_example."""

  # Chose a set of parameters
  test_parameters = [{
      "feature_dtype": [tf.string, tf.float32, tf.int64],
      "is_dense": [True, False],
      "feature_shape": [[1], [2], [16]],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph: a string placeholder of one serialized example, parsed with either a fixed
    # length feature (with a default value) or a variable length feature. Output is the feature, or
    # the values of the sparse result. Returns the input placeholder list and output list.
    """Build the graph for parse_example tests."""
    feature_dtype = parameters["feature_dtype"]
    feature_shape = parameters["feature_shape"]
    is_dense = parameters["is_dense"]
    input_value = tf.compat.v1.placeholder(
        dtype=tf.string, name="input", shape=[1])
    if is_dense:
      feature_default_value = np.zeros(shape=feature_shape)
      if feature_dtype == tf.string:
        feature_default_value = np.array(["missing"] * feature_shape[0])
      features = {
          "x":
              tf.io.FixedLenFeature(
                  shape=feature_shape,
                  dtype=feature_dtype,
                  default_value=feature_default_value)
      }
    else:  # Sparse
      features = {"x": tf.io.VarLenFeature(dtype=feature_dtype)}
    out = tf.io.parse_example(serialized=input_value, features=features)
    output_tensor = out["x"]
    if not is_dense:
      output_tensor = out["x"].values
    return [input_value], [output_tensor]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Creates serialized example data and runs the session for expected outputs. Returns the list
    # of input values and computed outputs.
    feature_dtype = parameters["feature_dtype"]
    feature_shape = parameters["feature_shape"]
    input_values = [create_example_data(feature_dtype, feature_shape)]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  extra_convert_options = ExtraConvertOptions()
  extra_convert_options.allow_custom_ops = True
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_convert_options)
