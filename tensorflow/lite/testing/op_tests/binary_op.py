# [MODULE DOC]
# Test-generation module for the TFLite converter zip tests of binary elementwise ops (add, div,
# sub, mul, pow, floor div, floor mod, squared difference). A shared generic maker builds graphs
# for any two-input operator across broadcast shapes, dtypes, optional relu activation and
# quantization modes; thin registered wrappers supply the operator and extra parameter sets.
# Importing registers the wrappers in the global registry.
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
"""Test configs for binary_op."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def make_binary_op_tests(options,
                         binary_operator,
                         allow_fully_quantize=False,
                         expected_tf_failures=0,
                         test_parameters=None):
  # [FUNCTION DOC]
  # HELPER: shared generic test maker used by the registered binary-op wrappers in this module.
  # Inputs: options (generator options), binary_operator (callable taking two tensors and returning
  # one), allow_fully_quantize (default false; if false, all parameter sets requesting full
  # quantization are dropped), expected_tf_failures (default zero), and test_parameters (optional
  # list of extra parameter sets, prepended to the built-in ones; nothing means empty).
  # Built-in sets cover equal shapes, broadcasting in both directions, scalars, with/without relu
  # activation, full quantization and dynamic-range quantization. A float64 set is added only when
  # running with flex; a high-dimension broadcast set with unknown dimensions is added unless the
  # options skip high dimension inputs. Calls the zip-test generator. Returns nothing.
  """Make a set of tests to do binary ops with and without broadcast."""

  if test_parameters is None:
    test_parameters = []

  test_parameters = test_parameters + [
      # Avoid creating all combinations to keep the test size small.
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape_1": [[1, 3, 4, 3]],
          "input_shape_2": [[1, 3, 4, 3]],
          "activation": [True],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[5]],
          "input_shape_2": [[5]],
          "activation": [False, True],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32, tf.int32, tf.int64],
          "input_shape_1": [[1, 3, 4, 3]],
          "input_shape_2": [[3]],
          "activation": [True, False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape_1": [[3]],
          "input_shape_2": [[1, 3, 4, 3]],
          "activation": [True, False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[]],
          "input_shape_2": [[]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[1, 3, 4, 3]],
          "input_shape_2": [[1, 3, 4, 3]],
          "activation": [False],
          "fully_quantize": [True],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[5]],
          "input_shape_2": [[5]],
          "activation": [False],
          "fully_quantize": [True],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[1, 3, 4, 3]],
          "input_shape_2": [[3]],
          "activation": [False],
          "fully_quantize": [True],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[3]],
          "input_shape_2": [[1, 3, 4, 3]],
          "activation": [False],
          "fully_quantize": [True],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[]],
          "input_shape_2": [[]],
          "activation": [False],
          "fully_quantize": [True],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[1, 3, 4, 3]],
          "input_shape_2": [[1, 3, 4, 3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [True],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[5]],
          "input_shape_2": [[5]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [True],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[1, 3, 4, 3]],
          "input_shape_2": [[3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [True],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[3]],
          "input_shape_2": [[1, 3, 4, 3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [True],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[]],
          "input_shape_2": [[]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [True],
      },
  ]

  # float64 types are supported via flex only.
  if options.run_with_flex:
    test_parameters = test_parameters + [
        {
            "dtype": [tf.float64],
            "input_shape_1": [[7]],
            "input_shape_2": [[7]],
            "activation": [False],
            "fully_quantize": [False],
            "dynamic_range_quantize": [False],
        },
    ]

  if not options.skip_high_dimension_inputs:
    test_parameters = test_parameters + [
        # High dimension broadcasting support in MLIR converter.
        # Note(b/204360746): XNNPack delegate don't support high dimension.
        {
            "dtype": [tf.float32],
            "input_shape_1": [[8, 7, 6, 5, 4, 3, 2, 1],
                              [8, 7, 6, 5, None, 3, 2, 1], [2, None]],
            "input_shape_2": [[4, 3, 2, 1], [None, 3, 2, 1]],
            "activation": [False],
            "fully_quantize": [False],
            "dynamic_range_quantize": [False],
            "dynamic_size_value": [4, 1],
        }
    ]

  # test_parameters include fully_quantize option only when
  # allow_fully_quantize is True.
  if not allow_fully_quantize:
    test_parameters = [
        test_parameter for test_parameter in test_parameters
        if True not in test_parameter["fully_quantize"]
    ]

  def populate_dynamic_shape(parameters, input_shape):
    # [FUNCTION DOC]
    # HELPER: shape utility for the binary-op tests.
    # Returns a new list copy of the shape in which each unknown (null) dimension is replaced by the
    # dynamic size value taken from the parameters. Known dimensions are kept. The parameter set must
    # contain the dynamic size value whenever an unknown dimension is present.
    return [
        parameters["dynamic_size_value"] if x is None else x
        for x in input_shape
    ]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # HELPER: graph builder callback for the binary-op zip tests.
    # Creates placeholders input1 and input2 with the parameterized dtype and shapes (shapes may
    # contain unknown dimensions), applies the binary operator, and applies relu when the activation
    # flag is set and the dtype is not a 32- or 64-bit integer. Returns the list of both
    # placeholders and the list with the output tensor.
    """Builds the graph given the current parameters."""
    input1 = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input1",
        shape=parameters["input_shape_1"])
    input2 = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input2",
        shape=parameters["input_shape_2"])
    out = binary_operator(input1, input2)
    if parameters["activation"] and (parameters["dtype"] != tf.int32 and
                                     parameters["dtype"] != tf.int64):
      out = tf.nn.relu(out)
    return [input1, input2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input-feeding callback for the binary-op zip tests.
    # Resolves unknown dimensions to concrete sizes, generates random data for both operands (range
    # minus one to one when full quantization is allowed, default range otherwise), runs the session
    # and returns the list of the two input values together with the expected outputs.
    """Builds operand inputs for op."""
    input_shape_1 = populate_dynamic_shape(parameters,
                                           parameters["input_shape_1"])
    input_shape_2 = populate_dynamic_shape(parameters,
                                           parameters["input_shape_2"])
    if allow_fully_quantize:
      input1 = create_tensor_data(
          parameters["dtype"], input_shape_1, min_value=-1, max_value=1)
      input2 = create_tensor_data(
          parameters["dtype"], input_shape_2, min_value=-1, max_value=1)
    else:
      input1 = create_tensor_data(parameters["dtype"], input_shape_1)
      input2 = create_tensor_data(parameters["dtype"], input_shape_2)
    return [input1, input2], sess.run(
        outputs, feed_dict={
            inputs[0]: input1,
            inputs[1]: input2
        })

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=expected_tf_failures)


def make_binary_op_tests_func(binary_operator):
  # [FUNCTION DOC]
  # HELPER: factory for test makers. Takes a binary operator callable and returns a one-argument
  # function (taking options) that runs the generic binary-op maker with default settings for that
  # operator. Not registered itself; no side effects until the returned function is called.
  """Return a function that does a test on a binary operator."""
  return lambda options: make_binary_op_tests(options, binary_operator)


@register_make_test_function()
def make_add_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for addition. Supplies extra parameter sets with unsigned 32-bit and
  # signed 16-bit integer dtypes (mixed broadcast shapes) and runs the generic binary-op maker with
  # the add operator and full quantization allowed. Returns nothing.
  """Make zip tests for add op with uint32 and int16 case."""
  test_parameters = [
      {
          "dtype": [tf.uint32],
          "input_shape_1": [[1, 3, 3, 3], [1], [3, 3]],
          "input_shape_2": [[3], [1]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.int16],
          "input_shape_1": [[1, 3, 3, 3], [1], [3, 3]],
          "input_shape_2": [[3], [1]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
  ]
  make_binary_op_tests(
      options,
      tf.add,
      allow_fully_quantize=True,
      test_parameters=test_parameters)


@register_make_test_function()
def make_div_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for division. Supplies an extra float32 five-dimensional case broadcast
  # against a vector, with and without dynamic-range quantization, and runs the generic binary-op
  # maker with the v1 compat division operator (full quantization not allowed). Returns nothing.
  """Make zip tests for div op with 5D case."""
  test_parameters = [
      {
          "dtype": [tf.float32],
          "input_shape_1": [[1, 3, 3, 3, 3]],
          "input_shape_2": [[3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False, True],
      },
  ]
  make_binary_op_tests(
      options, tf.compat.v1.div, test_parameters=test_parameters)


@register_make_test_function()
def make_sub_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for subtraction. Supplies an extra float32 five-dimensional case
  # broadcast against a vector (with and without dynamic-range quantization) and runs the generic
  # binary-op maker with the subtract operator and full quantization allowed. Returns nothing.
  """Make zip tests for sub op with additional cases."""
  test_parameters = [
      {
          "dtype": [tf.float32],
          "input_shape_1": [[1, 3, 3, 3, 3]],
          "input_shape_2": [[3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False, True],
      },
  ]
  make_binary_op_tests(
      options,
      tf.subtract,
      allow_fully_quantize=True,
      test_parameters=test_parameters)


@register_make_test_function()
def make_mul_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for multiplication. Supplies extra parameter sets for complex64
  # (five-dimensional against a vector), signed 16-bit and unsigned 32-bit integers, then runs the
  # generic binary-op maker with the multiply operator and full quantization allowed. Returns
  # nothing.
  """Make zip tests for mul op with additional complex cases."""
  test_parameters = [
      {
          "dtype": [tf.complex64],
          "input_shape_1": [[1, 3, 3, 3, 3]],
          "input_shape_2": [[3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.int16],
          "input_shape_1": [[1, 3, 3, 3]],
          "input_shape_2": [[3], [1, 3, 3, 3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.uint32],
          "input_shape_1": [[1, 3, 3, 3]],
          "input_shape_2": [[3], [1, 3, 3, 3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
  ]
  make_binary_op_tests(
      options,
      tf.multiply,
      allow_fully_quantize=True,
      test_parameters=test_parameters)


@register_make_test_function()
def make_pow_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for power. Runs the generic binary-op maker with the power operator and
  # no extra parameter sets, expecting 7 failures in TensorFlow itself. Returns nothing.
  make_binary_op_tests(options, tf.pow, expected_tf_failures=7)


@register_make_test_function()
def make_floor_div_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for floor division. Supplies an extra case with signed 8-bit and 16-bit
  # integers (five-dimensional against a vector) and runs the generic binary-op maker with the
  # floor-division operator. Returns nothing.
  """Make zip tests for floor_div op with int16 case."""
  test_parameters = [
      {
          "dtype": [tf.int8, tf.int16],
          "input_shape_1": [[1, 3, 3, 3, 3]],
          "input_shape_2": [[3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
  ]
  make_binary_op_tests(
      options, tf.math.floordiv, test_parameters=test_parameters
  )


@register_make_test_function()
def make_floor_mod_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for floor modulo. Supplies an extra case with signed 8-bit and 16-bit
  # integers (five-dimensional against a vector) and runs the generic binary-op maker with the
  # floor-modulo operator. Returns nothing.
  """Make zip tests for floor_mod op with int16 case."""
  test_parameters = [
      {
          "dtype": [tf.int8, tf.int16],
          "input_shape_1": [[1, 3, 3, 3, 3]],
          "input_shape_2": [[3]],
          "activation": [False],
          "fully_quantize": [False],
          "dynamic_range_quantize": [False],
      },
  ]
  make_binary_op_tests(
      options, tf.math.floormod, test_parameters=test_parameters
  )


@register_make_test_function()
def make_squared_difference_tests(options):
  # [FUNCTION DOC]
  # Registered test maker for squared difference. Runs the generic binary-op maker with the
  # squared-difference operator, no extra parameter sets, and full quantization allowed. Returns
  # nothing.
  make_binary_op_tests(
      options, tf.math.squared_difference, allow_fully_quantize=True)
