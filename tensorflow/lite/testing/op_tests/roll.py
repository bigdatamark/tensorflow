# [MODULE DOC]
# Part of the TFLite op test suite: each module in this directory defines one registered generator that
# produces a zip of converted-model test cases for a single TensorFlow op. Importing the module only
# registers that generator in zip_test_utils; nothing runs until the driver invokes it. Not a helper.
# This one covers the roll op. It holds a module-level list of parameter groups (scalar, 1-D, multiple
# and duplicate axes) shared by two generators: one with constant shift and axis, and one that feeds
# shift and axis as int64 tensor inputs and also adds a dynamic-dimension case.
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
"""Test configs for roll."""
import numpy as np
import tensorflow as tf

from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

test_parameters = [
    # Scalar axis.
    {
        "input_dtype": [tf.float32, tf.int32],
        "input_shape": [[2, 4, 5], [3, 8, 4]],
        "shift": [1, -3, 5],
        "axis": [0, 1, 2],
    },
    # 1-D axis.
    {
        "input_dtype": [tf.float32, tf.int32],
        "input_shape": [[2, 4, 5], [3, 8, 4]],
        "shift": [[1], [-3], [5]],
        "axis": [[0], [1], [2]],
    },
    # Multiple axis.
    {
        "input_dtype": [tf.float32, tf.int32],
        "input_shape": [[2, 4, 5], [3, 8, 4]],
        "shift": [[1, 3, 2], [3, -6, 5], [-5, 7, 8]],
        "axis": [[0, 1, 2]],
    },
    # Duplicate axis.
    {
        "input_dtype": [tf.float32],
        "input_shape": [[2, 4, 5], [3, 8, 4]],
        "shift": [[1, 3, -2]],
        "axis": [[0, 1, 1]],
    },
]


@register_make_test_function()
def make_roll_with_constant_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Uses the module-level parameter groups (float32 and int32; 3-D shapes; shifts and axes as scalars,
  # lists, multiple or duplicated axes) with shift and axis baked in as constants, and calls
  # make_zip_of_tests with the nested builders.
  """Make a set of tests to do roll with constant shift and axis."""

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests.
    # Creates a placeholder named input of the requested dtype and shape and rolls it by the constant
    # shift along the constant axis from the parameters. Returns the list of input placeholders and
    # the list of outputs.
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    outs = tf.roll(
        input_value, shift=parameters["shift"], axis=parameters["axis"])
    return [input_value], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests.
    # Creates random data of the requested dtype and shape, runs the graph and returns the list of
    # input values and the list of computed outputs.
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)


@register_make_test_function()
def make_roll_tests(options):
  # [FUNCTION DOC]
  # Registered by the make-test-function decorator under its function name so the TFLite zip-test driver can find it.
  # Input: options, the test-run options object. Returns nothing.
  # Extends the module-level parameter groups with an extra group that has an unknown (null) first
  # dimension, scalar shifts and axes. Shift and axis are fed as int64 inputs. Creates an extra
  # converter options object with custom ops allowed (side effect only on that object) and passes it
  # to make_zip_of_tests along with the nested builders.
  """Make a set of tests to do roll."""

  ext_test_parameters = test_parameters + [
      # Scalar axis.
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape": [[None, 8, 4]],
          "shift": [-3, 5],
          "axis": [1, 2],
      }
  ]

  def set_dynamic_shape(shape):
    # [FUNCTION DOC]
    # HELPER: turns a shape that may contain unknown (null) dimensions into a concrete one.
    # Input: shape, a list of integers or null values. Returns a new list in which every null is
    # replaced by 4.
    return [4 if x is None else x for x in shape]

  def get_shape(param):
    # [FUNCTION DOC]
    # HELPER: computes the placeholder shape for a shift or axis value.
    # Input: param, a scalar or a list. Returns an empty shape for a scalar, otherwise a one-element
    # shape holding the list length.
    if np.isscalar(param):
      return []
    return [len(param)]

  def get_value(param, dtype):
    # [FUNCTION DOC]
    # HELPER: converts a shift or axis parameter to the numpy value fed to the graph.
    # Inputs: param, a scalar or a list; dtype, the numpy dtype. Returns a numpy scalar of that dtype
    # for a scalar, or a numpy array cast to that dtype for a list.
    if np.isscalar(param):
      return np.dtype(dtype).type(param)
    return np.array(param).astype(dtype)

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for make_zip_of_tests (the one in the tensor-input generator).
    # Creates placeholders named input (requested dtype and shape, which may have an unknown dimension),
    # shift and axis (both int64, shaped via the shape helper), and rolls the input. Returns the list of
    # the three placeholders and the list of outputs.
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    shift_tensor = tf.compat.v1.placeholder(
        dtype=tf.int64, name="shift", shape=get_shape(parameters["shift"]))
    axis_tensor = tf.compat.v1.placeholder(
        dtype=tf.int64, name="axis", shape=get_shape(parameters["axis"]))
    outs = tf.roll(input_tensor, shift_tensor, axis_tensor)
    return [input_tensor, shift_tensor, axis_tensor], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input builder callback for make_zip_of_tests (tensor-input generator).
    # Creates random input data using the concrete shape (unknown dimension replaced by 4), converts
    # shift and axis to int64 values, runs the graph and returns the list of the three input values and
    # the list of computed outputs.
    input_value = create_tensor_data(
        parameters["input_dtype"], set_dynamic_shape(parameters["input_shape"]))
    shift_value = get_value(parameters["shift"], np.int64)
    axis_value = get_value(parameters["axis"], np.int64)
    return [input_value, shift_value, axis_value], sess.run(
        outputs,
        feed_dict=dict(zip(inputs, [input_value, shift_value, axis_value])))

  extra_convert_options = ExtraConvertOptions()
  extra_convert_options.allow_custom_ops = True
  make_zip_of_tests(options, ext_test_parameters, build_graph, build_inputs,
                    extra_convert_options)
