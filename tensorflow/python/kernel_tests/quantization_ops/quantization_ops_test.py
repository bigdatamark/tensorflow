# [MODULE DOC]
# Tests for the quantization related ops (fake quantization with min/max variables, per-channel and
# gradient variants, quantized bias add, instance norm, pooling, requantize, add, relu, relu6, down
# and shrink range, and quantize-and-dequantize V3). Most tests only verify that malformed range
# arguments (non-scalar min or max, wrong ranks or sizes) are rejected with clear errors, in both
# graph and eager modes; the last class also checks a valid quantize-and-dequantize result.
# Running the file as a script starts the test runner.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.quantize ops."""
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class FakeQuantWithMinMaxVarsOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named FakeQuantWithMinMaxVarsOpTest covering one quantization related op. It only
  # checks that invalid arguments are rejected with a clear error. It has no shared fixtures; its
  # single test method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the fake quantization op with per-tensor min and max where one of them is a rank 2
    # tensor instead of a scalar; both orders are tried. Expects a value error or invalid-argument
    # error saying the argument must be rank 0.
    inputs = constant_op.constant(
        value=[[1.0], [2.0], [4.0]], dtype=dtypes.float32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars(
              inputs=inputs, min=0.0, max=[[1.0], [2.0], [4.0]]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars(
              inputs=inputs, min=[[1.0], [2.0], [4.0]], max=1.0))


class FakeQuantWithMinMaxVarsPerChannelOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named FakeQuantWithMinMaxVarsPerChannelOpTest covering one quantization related op. It
  # only checks that invalid arguments are rejected with a clear error. It has no shared fixtures;
  # its single test method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the per-channel fake quantization op with bad min and max: a rank 2 min, a min and
    # max of different lengths, a rank 2 max, and mismatched lengths in the other direction. Expects
    # value or invalid-argument errors saying the argument must be rank 1 or the sizes/dimensions
    # are incorrect.
    inputs = constant_op.constant(
        value=[[1.0], [2.0], [4.0]], dtype=dtypes.float32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 1"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel(
              inputs=inputs, min=[[0.0]], max=[1.0]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Dimensions must be equal|incorrect size"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel(
              inputs=inputs, min=[0.0, 0.1], max=[1.0]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 1"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel(
              inputs=inputs, min=[1.0], max=[[1.0]]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Dimensions must be equal|incorrect size"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel(
              inputs=inputs, min=[0.0], max=[1.0, 1.1]))


class FakeQuantWithMinMaxVarsGradientOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named FakeQuantWithMinMaxVarsGradientOpTest covering one quantization related op. It
  # only checks that invalid arguments are rejected with a clear error. It has no shared fixtures;
  # its single test method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the fake quantization gradient op with float gradients and inputs but a non-scalar
    # max (or min and max). Expects a value or invalid-argument error about equal rank or rank 0.
    gradients = constant_op.constant(
        value=[[1.0], [2.0], [4.0]], dtype=dtypes.float32)
    inputs = constant_op.constant(
        value=[[1.0], [2.0], [4.0]], dtype=dtypes.float32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be equal rank|must be rank 0"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_gradient(
              gradients=gradients,
              inputs=inputs,
              min=0.0,
              max=[[1.0], [2.0], [4.0]]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_gradient(
              gradients=gradients,
              inputs=inputs,
              min=[[1.0], [2.0], [4.0]],
              max=[[1.0], [2.0], [4.0]]))


class FakeQuantWithMinMaxVarsPerChannelGradientOpTest(
    test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named FakeQuantWithMinMaxVarsPerChannelGradientOpTest covering one quantization
  # related op. It only checks that invalid arguments are rejected with a clear error. It has no
  # shared fixtures; its single test method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the per-channel fake quantization gradient op with rank 2 min or max, or min and
    # max of different lengths. Expects value or invalid-argument errors about equal ranks, rank 1,
    # or dimension 0 mismatch / incorrect size.
    gradients = constant_op.constant(
        value=[[1.0], [2.0], [4.0]], dtype=dtypes.float32)
    inputs = constant_op.constant(
        value=[[1.0], [2.0], [4.0]], dtype=dtypes.float32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Shapes must be equal rank|must be rank 1"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel_gradient(
              gradients=gradients, inputs=inputs, min=[[0.0]], max=[1.0]))

    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Dimension 0 in both shapes must be equal|incorrect size"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel_gradient(
              gradients=gradients, inputs=inputs, min=[0.0, 0.1], max=[1.0]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Shapes must be equal rank|must be rank 1"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel_gradient(
              gradients=gradients, inputs=inputs, min=[1.0], max=[[1.0]]))

    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Dimension 0 in both shapes must be equal|incorrect size"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel_gradient(
              gradients=gradients, inputs=inputs, min=[0.0], max=[1.0, 1.1]))


class QuantizedBiasedAddTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named QuantizedBiasedAddTest covering one quantization related op. It only checks that
  # invalid arguments are rejected with a clear error. It has no shared fixtures; its single test
  # method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the quantized bias add op with 8-bit quantized input and bias, giving each of the
    # four range arguments (input min, input max, bias min, bias max) in turn as an empty list
    # instead of a scalar. Expects a value or invalid-argument error saying must be rank 0.
    inputs = constant_op.constant(
        np.int8(0), shape=[3, 3, 3, 3], dtype=dtypes.qint8)
    bias = constant_op.constant(np.int8(0), shape=[3], dtype=dtypes.qint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_bias_add(
              input=inputs,
              bias=bias,
              min_input=[],
              max_input=1.0,
              min_bias=0.0,
              max_bias=1.0,
              out_type=dtypes.qint32))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_bias_add(
              input=inputs,
              bias=bias,
              min_input=0.0,
              max_input=[],
              min_bias=0.0,
              max_bias=1.0,
              out_type=dtypes.qint32))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_bias_add(
              input=inputs,
              bias=bias,
              min_input=0.0,
              max_input=1.0,
              min_bias=[],
              max_bias=1.0,
              out_type=dtypes.qint32))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_bias_add(
              input=inputs,
              bias=bias,
              min_input=0.0,
              max_input=1.0,
              min_bias=0.0,
              max_bias=[],
              out_type=dtypes.qint32))


class QuantizedInstanceNormOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named QuantizedInstanceNormOpTest covering one quantization related op. It only checks
  # that invalid arguments are rejected with a clear error. It has no shared fixtures; its single
  # test method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the quantized instance norm op with a quantized 4-D input and a non-scalar x max,
    # then a non-scalar x min. Expects value or invalid-argument errors saying must be rank 0.
    inputs = constant_op.constant(
        np.uint8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.quantized_instance_norm(
              x=inputs, x_min=0.0, x_max=[[1.0], [2.0], [4.0]]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.quantized_instance_norm(
              x=inputs, x_min=[[1.0], [2.0], [4.0]], x_max=1.0))


class QuantizedAvgPoolingOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named QuantizedAvgPoolingOpTest covering one quantization related op. It only checks
  # that invalid arguments are rejected with a clear error. It has no shared fixtures; its single
  # test method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the quantized average pooling op with unit window and strides and same padding,
    # giving the input min and then the input max as an empty list. Expects invalid-argument or
    # value errors saying the range must be rank 0.
    inputs = constant_op.constant(
        np.uint8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "must be.* rank 0"):
      self.evaluate(
          nn_ops.quantized_avg_pool(
              input=inputs,
              min_input=[],
              max_input=1.0,
              ksize=ksize,
              strides=strides,
              padding=padding))

    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "must be.* rank 0"):
      self.evaluate(
          nn_ops.quantized_avg_pool(
              input=inputs,
              min_input=0.0,
              max_input=[],
              ksize=ksize,
              strides=strides,
              padding=padding))


class QuantizedMaxPoolingOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named QuantizedMaxPoolingOpTest covering one quantization related op. It only checks
  # that invalid arguments are rejected with a clear error. It has no shared fixtures; its single
  # test method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the quantized max pooling op with unit window and strides and same padding, giving
    # the input min and then the input max as an empty list. Expects invalid-argument or value
    # errors saying the range must be rank 0.
    inputs = constant_op.constant(
        np.uint8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "must be.* rank 0"):
      self.evaluate(
          nn_ops.quantized_max_pool(
              input=inputs,
              min_input=[],
              max_input=1.0,
              ksize=ksize,
              strides=strides,
              padding=padding))

    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "must be.* rank 0"):
      self.evaluate(
          nn_ops.quantized_max_pool(
              input=inputs,
              min_input=0.0,
              max_input=[],
              ksize=ksize,
              strides=strides,
              padding=padding))


class RequantizeOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named RequantizeOpTest covering one quantization related op. It only checks that
  # invalid arguments are rejected with a clear error. It has no shared fixtures; its single test
  # method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the requantize op on a 32-bit quantized input, giving each of input min, input
    # max, requested output min and requested output max in turn as an empty list. Expects value or
    # invalid-argument errors saying must be rank 0.
    inputs = constant_op.constant(
        np.int32(0), shape=[3, 3, 3, 3], dtype=dtypes.qint32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.requantize(
              input=inputs,
              input_min=[],
              input_max=1.0,
              requested_output_min=0.0,
              requested_output_max=1.0,
              out_type=dtypes.qint8))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.requantize(
              input=inputs,
              input_min=0.0,
              input_max=[],
              requested_output_min=0.0,
              requested_output_max=1.0,
              out_type=dtypes.qint8))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.requantize(
              input=inputs,
              input_min=0.0,
              input_max=1.0,
              requested_output_min=[],
              requested_output_max=1.0,
              out_type=dtypes.qint8))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.requantize(
              input=inputs,
              input_min=0.0,
              input_max=1.0,
              requested_output_min=0.0,
              requested_output_max=[],
              out_type=dtypes.qint8))


class QuantizedAddOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named QuantizedAddOpTest covering one quantization related op. It only checks that
  # invalid arguments are rejected with a clear error. It has no shared fixtures; its single test
  # method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the quantized add op with quantized x and a smaller y and gives the x min as an
    # empty list. Expects a value or invalid-argument error saying must be rank 0.
    x = constant_op.constant(
        np.int8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)
    y = constant_op.constant(np.int8(0), shape=[3], dtype=dtypes.quint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.quantized_add(
              x=x,
              y=y,
              min_x=[],
              max_x=1.0,
              min_y=0.0,
              max_y=1.0,
              Toutput=dtypes.qint32))


class QuantizedReluOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named QuantizedReluOpTest covering one quantization related op. It only checks that
  # invalid arguments are rejected with a clear error. It has no shared fixtures; its single test
  # method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the quantized relu op with an empty list as the feature minimum. Expects a value
    # or invalid-argument error saying must be rank 0.
    inputs = constant_op.constant(
        np.int8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_relu(
              features=inputs,
              min_features=[],
              max_features=127.0,
              out_type=dtypes.quint8))


class QuantizedRelu6OpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named QuantizedRelu6OpTest covering one quantization related op. It only checks that
  # invalid arguments are rejected with a clear error. It has no shared fixtures; its single test
  # method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the quantized relu6 op with an empty list as the feature minimum. Expects a value
    # or invalid-argument error saying must be rank 0.
    inputs = constant_op.constant(
        np.int8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_relu6(
              features=inputs,
              min_features=[],
              max_features=127.0,
              out_type=dtypes.quint8))


class QuantizeDownAndShrinkRangeOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case named QuantizeDownAndShrinkRangeOpTest covering one quantization related op. It only
  # checks that invalid arguments are rejected with a clear error. It has no shared fixtures; its
  # single test method runs in both graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Test: calls the quantize down and shrink range op on a 32-bit quantized input with an empty
    # list as the input minimum. Expects a value or invalid-argument error saying must be rank 0.
    inputs = constant_op.constant(
        np.int32(0), shape=[3, 3, 3, 3], dtype=dtypes.qint32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.quantize_down_and_shrink_range(
              input=inputs, input_min=[], input_max=4.0,
              out_type=dtypes.quint8))


class QuantizeAndDequantizeV3OpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the quantize-and-dequantize V3 op: one valid-result test and two error tests for
  # a bad num_bits tensor and mismatched min/max shapes when an axis is given.

  @test_util.run_in_graph_and_eager_modes
  def test_valid(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the input value is built with a trailing comma so it is a one-element tuple of a
    # tensor, not a tensor; the assertion compensates by indexing element zero.
    # Test: inside a fresh graph switched to eager mode, quantizes six floats with a range of -127 to
    # 127 and 8 bits, signed, with the range not given. Asserts the result is within 0.05 of the
    # input element-wise.
    with ops.Graph().as_default(), context.eager_mode():
      input_value = constant_op.constant([-0.8, -0.5, 0, 0.3, 0.8, -2.0],
                                         shape=(6,),
                                         dtype=dtypes.float32),
      input_min = constant_op.constant(-127, shape=(), dtype=dtypes.float32)
      input_max = constant_op.constant(127, shape=(), dtype=dtypes.float32)
      num_bits = constant_op.constant(8, shape=(), dtype=dtypes.int32)

      quantized = array_ops.quantize_and_dequantize_v3(
          input_value,
          input_min,
          input_max,
          num_bits,
          signed_input=True,
          range_given=False)
      self.assertSequenceAlmostEqual(
          input_value[0].numpy(), quantized.numpy()[0], delta=0.05)

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_num_bits(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the input value is a one-element tuple (trailing comma) rather than a tensor.
    # KNOWN ISSUE: the failure branch for an unexpected exception formats the message using a name
    # attribute of the exception instance, which exception instances do not have, so that branch
    # would itself raise an attribute error instead of failing with the intended message.
    # Test: passes an empty num_bits tensor of shape zero. Accepts either an invalid-argument error
    # (message about num_bits needing to be a scalar) or a value error (shape must be rank 0), and
    # fails if another exception or no exception occurs.
    input_value = constant_op.constant([-0.8, -0.5, 0, 0.3, 0.8, -2.0],
                                       shape=(6,),
                                       dtype=dtypes.float32),
    input_min = constant_op.constant(-127, shape=(), dtype=dtypes.float32)
    input_max = constant_op.constant(127, shape=(), dtype=dtypes.float32)
    # Tensor with invalid shape and invalid number of elements.
    num_bits = constant_op.constant([], shape=(0,), dtype=dtypes.int32)

    # Test that running the op raises error. It raises different errors
    # depending on whether the shape inference is run first or the op's
    # Compute() is run first.
    try:
      array_ops.quantize_and_dequantize_v3(
          input_value, input_min, input_max, num_bits, signed_input=True)
    except Exception as ex:  # pylint: disable=broad-except
      if isinstance(ex, errors.InvalidArgumentError):
        self.assertRegex(str(ex), "The `num_bits` tensor should be a scalar.")
      elif isinstance(ex, ValueError):
        self.assertRegex(str(ex), "Shape must be rank 0")
      else:
        self.fail(
            "Raised exception other than expected: %s. "
            "Expected exceptions are errors.InvalidArgumentError or ValueError"
            % ex.__name__
        )
    else:
      self.fail(
          "Did not raise an exception where it is expected to raise either "
          "a ValueError or errors.InvalidArgumentError.")

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_input_min_max_with_axis_specified(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the input value is a one-element tuple (trailing comma) rather than a tensor.
    # Test: passes a scalar min and a rank 1 max with axis zero and range given. Expects an
    # invalid-argument or value error saying the shape must be rank 1.
    input_value = (
        constant_op.constant([1.8], shape=(1,), dtype=dtypes.float32),
    )
    input_min = constant_op.constant(1.0, shape=(), dtype=dtypes.float32)
    input_max = constant_op.constant([2.0], shape=(1,), dtype=dtypes.float32)
    num_bits = 8

    # Test that running the op raises error. It raises different errors
    # depending on whether the shape inference is run first or the op's
    # Compute() is run first.
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Shape must be rank 1"):
      array_ops.quantize_and_dequantize_v3(
          input_value,
          input_min,
          input_max,
          num_bits=num_bits,
          axis=0,
          range_given=True,
      )


if __name__ == "__main__":
  googletest.main()
