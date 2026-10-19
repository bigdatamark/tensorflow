# [MODULE DOC]
# HELPER: shared test base for the BiasAdd tests; it is imported by the concrete test modules
# (the regular bias test and the deterministic-op test) and is not run on its own.
# Defines BiasAddTestBase, a test case checking nn_ops.bias_add forward results against numpy for
# integer and floating types in channels-last and channels-first layouts (CPU and GPU), input
# validation errors, and gradients (input, bias and gradient-of-gradient) in both graph and eager
# modes. Importing it also registers the nn gradient functions as a side effect.
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
"""Functional tests for BiasAdd."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class BiasAddTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: reusable base class for BiasAdd tests, meant to be subclassed by concrete test files.
  # The class decorator makes every test run in both graph and eager modes. Contains numpy
  # reference helpers (reference bias addition, layout conversions between channels-last and
  # channels-first), forward-test helpers, gradient-checking helpers and the actual test methods.
  # It holds no instance state.

  def _npBias(self, inputs, bias):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of bias addition.
    # Inputs: a numpy array and a one-dimensional bias whose length equals the last dimension of
    # the input. Returns the input with the bias broadcast along the last dimension and added.
    # Raises an assertion error if the bias is not one-dimensional or its length mismatches.
    assert len(bias.shape) == 1
    assert inputs.shape[-1] == bias.shape[0]
    return inputs + bias.reshape(([1] *
                                  (len(inputs.shape) - 1)) + [bias.shape[0]])

  def testNpBias(self):
    # [FUNCTION DOC]
    # Scenario: sanity check of the numpy reference helper on a two by three integer matrix with
    # a three-element bias; asserts the result equals the hand-computed sums.
    self.assertAllClose(
        np.array([[11, 22, 33], [41, 52, 63]]),
        self._npBias(
            np.array([[10, 20, 30], [40, 50, 60]]), np.array([1, 2, 3])))

  def _testBias(self, np_inputs, np_bias, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: forward check in the default channels-last layout.
    # Inputs: numpy input and bias arrays, and a flag choosing GPU or CPU placement (default CPU).
    # Computes the reference with numpy, evaluates the bias_add op in a cached session, and asserts
    # the two agree within dtype-dependent tolerances. Returns nothing.
    np_val = self._npBias(np_inputs, np_bias)
    with self.cached_session(use_gpu=use_gpu):
      tf_val = self.evaluate(nn_ops.bias_add(np_inputs, np_bias))
    self.assertAllCloseAccordingToType(np_val, tf_val)

  def _AtLeast3d(self, np_value):
    # [FUNCTION DOC]
    # HELPER: layout utility. Returns the array reshaped by prepending size-one dimensions so it
    # has at least three dimensions; arrays that already have three or more are returned as is.
    # fill the input value to at least 3-dimension
    if np_value.ndim < 3:
      return np.reshape(np_value, (1,) * (3 - np_value.ndim) + np_value.shape)
    return np_value

  def _NHWCToNCHW(self, np_value):
    # [FUNCTION DOC]
    # HELPER: layout utility. Pads the array to at least three dimensions, then moves the last
    # (channel) dimension to position one, keeping the batch first. Returns the transposed array.
    # fill the input value to at least 3-dimension
    np_value = self._AtLeast3d(np_value)
    # move the last dimension to second
    np_dim = list(range(np_value.ndim))
    np_dim_new = list(np_dim[0:1]) + list(np_dim[-1:]) + list(np_dim[1:-1])
    return np.transpose(np_value, np_dim_new)

  def _NCHWToNHWC(self, np_value):
    # [FUNCTION DOC]
    # HELPER: layout utility, inverse of the channels-last to channels-first conversion. Moves the
    # second (channel) dimension to the end. Raises an assertion error for arrays with fewer than
    # three dimensions. Returns the transposed array.
    assert len(np_value.shape) >= 3
    np_dim = list(range(np_value.ndim))
    # move the second dimension to the last
    np_dim_new = list(np_dim[0:1]) + list(np_dim[2:]) + list(np_dim[1:2])
    return np.transpose(np_value, np_dim_new)

  def _testBiasNCHW(self, np_inputs, np_bias, use_gpu):
    # [FUNCTION DOC]
    # HELPER: forward check in the channels-first layout.
    # Inputs: channels-last numpy input, bias, and a GPU flag. Computes the numpy reference,
    # converts the input to channels-first, runs bias_add with the channels-first format, converts
    # the output back, and asserts closeness to the reference (padded to at least three
    # dimensions) within dtype-dependent tolerances. Returns nothing.
    np_val = self._npBias(np_inputs, np_bias)
    np_inputs = self._NHWCToNCHW(np_inputs)
    with self.cached_session(use_gpu=use_gpu):
      tf_val = self.evaluate(
          nn_ops.bias_add(np_inputs, np_bias, data_format="NCHW"))
    tf_val = self._NCHWToNHWC(tf_val)
    self.assertAllCloseAccordingToType(self._AtLeast3d(np_val), tf_val)

  def _testAll(self, np_inputs, np_bias):
    # [FUNCTION DOC]
    # HELPER: runs both forward checks (channels-last and channels-first) on CPU, and also on GPU
    # when the input dtype is float16, float32, float64 or int32. Inputs: numpy input and bias.
    self._testBias(np_inputs, np_bias, use_gpu=False)
    self._testBiasNCHW(np_inputs, np_bias, use_gpu=False)
    if np_inputs.dtype in [np.float16, np.float32, np.float64, np.int32]:
      self._testBias(np_inputs, np_bias, use_gpu=True)
      self._testBiasNCHW(np_inputs, np_bias, use_gpu=True)

  def _expectedException(self):
    # [FUNCTION DOC]
    # HELPER: returns the error class expected for invalid bias_add arguments: the invalid-argument
    # runtime error when executing eagerly, otherwise the value error raised at graph build time.
    if context.executing_eagerly():
      return errors_impl.InvalidArgumentError
    else:
      return ValueError

  def testInputDims(self):
    # [FUNCTION DOC]
    # Scenario: bias_add on a one-dimensional input list; asserts the expected error (the input
    # must have at least two dimensions).
    with self.assertRaises(self._expectedException()):
      nn_ops.bias_add([1, 2], [1])

  def testBiasVec(self):
    # [FUNCTION DOC]
    # Scenario: bias_add where the bias is two-dimensional rather than a vector; asserts the
    # expected error.
    with self.assertRaises(self._expectedException()):
      nn_ops.bias_add(
          array_ops.reshape([1, 2], shape=[1, 2]),
          array_ops.reshape([1, 2], shape=[1, 2]))

  def testBiasInputsMatch(self):
    # [FUNCTION DOC]
    # Scenario: bias_add where the bias length (one) differs from the input's last dimension
    # (two); asserts the expected error.
    with self.assertRaises(self._expectedException()):
      nn_ops.bias_add(
          array_ops.reshape([1, 2], shape=[1, 2]),
          array_ops.reshape([1], shape=[1]))

  def testIntTypes(self):
    # [FUNCTION DOC]
    # Scenario: a small two by three matrix and a three-element bias cast to int8, int16, int32 and
    # int64, each run through all forward checks.
    for t in [np.int8, np.int16, np.int32, np.int64]:
      self._testAll(
          np.array([[10, 20, 30], [40, 50, 60]]).astype(t),
          np.array([1, 2, 3]).astype(t))

  def testFloatTypes(self):
    # [FUNCTION DOC]
    # Scenario: random three-dimensional input (four by three by three) and a three-element bias in
    # float16, float32, float64 and bfloat16, each run through all forward checks.
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testAll(
          np.random.rand(4, 3, 3).astype(t),
          np.random.rand(3).astype(t))

  def test4DFloatTypes(self):
    # [FUNCTION DOC]
    # Scenario: random four-dimensional inputs (small, a large batch of 2048, and a large channel
    # count of 2048) with matching biases, for float16, float32, float64 and bfloat16, each run
    # through all forward checks.
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testAll(
          np.random.rand(4, 3, 2, 3).astype(t),
          np.random.rand(3).astype(t))
      self._testAll(
          np.random.rand(2048, 4, 4, 4).astype(t),
          np.random.rand(4).astype(t))
      self._testAll(
          np.random.rand(4, 4, 4, 2048).astype(t),
          np.random.rand(2048).astype(t))

  def test5DFloatTypes(self):
    # [FUNCTION DOC]
    # Scenario: a random five-dimensional input with a four-element bias for float16, float32,
    # float64 and bfloat16, each run through all forward checks.
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testAll(
          np.random.rand(4, 3, 2, 3, 4).astype(t),
          np.random.rand(4).astype(t))

  def _random_tensor(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: returns a constant tensor of the given shape and dtype filled with uniform random
    # values in the range from minus one to one. Used as the upstream gradient in gradient tests.
    return constant_op.constant(2 * np.random.rand(*shape) - 1, dtype=dtype)

  def _computeGradient(self, np_input, bias, dtype, data_format):
    # [FUNCTION DOC]
    # HELPER: computes analytical and numerical Jacobians for bias_add.
    # Inputs: numpy input, numpy bias, the TensorFlow dtype for the constants, and the data format
    # string. Returns a pair of triples: analytical (input, bias, gradient-of-bias-gradient) and
    # numerical Jacobians in the same order. In eager mode the input and bias gradients are
    # checked separately (work-around for TF issue 33660) and the gradient of the bias gradient
    # is checked via a tape with a random upstream gradient. In graph mode a single gradient check
    # covers input and bias, and the second-order check uses the gradient of the L2 loss (output
    # cast to float32 first for bfloat16, since L2 loss is unsupported there on CPU).
    input_shape = output_shape = np_input.shape
    bias_shape = bias.shape
    input_tensor = constant_op.constant(
        np_input, shape=input_shape, dtype=dtype)
    bias_tensor = constant_op.constant(bias, shape=bias_shape, dtype=dtype)

    if context.executing_eagerly():

      def bias_add(input_tensor, bias_tensor):
        # [FUNCTION DOC]
        # HELPER: eager-mode wrapper calling the bias_add op with the enclosing data format.
        # Inputs: input and bias tensors. Returns the output tensor.
        return nn_ops.bias_add(
            input_tensor, bias_tensor, data_format=data_format)

      # The following is a work-around for TF issue 33660. Instead of
      # calculating the analytical and numerical gradients for both
      # inputs in a single call to compute_gradient, compute_gradient
      # is called for each input separately.
      def bias_add_1(input_tensor):
        # [FUNCTION DOC]
        # HELPER: single-argument function of the input with the bias held fixed; lets the
        # gradient checker differentiate with respect to the input alone.
        return bias_add(input_tensor, bias_tensor)

      def bias_add_2(bias_tensor):
        # [FUNCTION DOC]
        # HELPER: single-argument function of the bias with the input held fixed; lets the
        # gradient checker differentiate with respect to the bias alone.
        return bias_add(input_tensor, bias_tensor)

      input_jacob_a, input_jacob_n = gradient_checker_v2.compute_gradient(
          bias_add_1, [input_tensor])
      bias_jacob_a, bias_jacob_n = gradient_checker_v2.compute_gradient(
          bias_add_2, [bias_tensor])

      # Test gradient of BiasAddGrad
      def bias_add_grad_function(upstream_gradients):
        # [FUNCTION DOC]
        # HELPER: function whose output is the bias gradient, used to test the gradient of the
        # bias-add gradient. Input: an upstream gradient tensor shaped like the output. Records
        # bias_add on a gradient tape with the bias watched, multiplies the output by the upstream
        # gradient, and returns the tape's gradient with respect to the bias.
        with backprop.GradientTape() as tape:
          tape.watch(bias_tensor)
          bias_add_output = bias_add(input_tensor, bias_tensor)
          gradient_injector_output = bias_add_output * upstream_gradients
          return tape.gradient(gradient_injector_output, bias_tensor)

      upstream_tensor = self._random_tensor(output_shape, dtype)
      grad_jacob_a, grad_jacob_n = gradient_checker_v2.compute_gradient(
          bias_add_grad_function, [upstream_tensor])
    else:
      output_tensor = nn_ops.bias_add(
          input_tensor, bias_tensor, data_format=data_format)
      jacobians = gradient_checker.compute_gradient([input_tensor, bias_tensor],
                                                    [input_shape, bias_shape],
                                                    output_tensor, output_shape)
      (input_jacob_a, input_jacob_n), (bias_jacob_a, bias_jacob_n) = jacobians
      # Test gradient of BiasAddGrad
      if dtype == dtypes.bfloat16:
        # L2Loss is not supported for bfloat16 on CPU.
        output_tensor = math_ops.cast(output_tensor, dtype=dtypes.float32)
      bias_add_grad = gradients_impl.gradients(
          nn_ops.l2_loss(output_tensor), bias_tensor)[0]
      grad_jacob_a, grad_jacob_n = gradient_checker.compute_gradient(
          output_tensor, output_shape, bias_add_grad, bias_shape)

    return ((input_jacob_a, bias_jacob_a, grad_jacob_a),
            (input_jacob_n, bias_jacob_n, grad_jacob_n))

  def _testGradient(self, np_input, bias, dtype, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared gradient assertion for bias_add.
    # Inputs: channels-last numpy input, bias, dtype, data format (channels-first inputs are
    # converted first) and a GPU flag. Computes analytical and numerical Jacobians; for float16 and
    # bfloat16 the numerical ones come from a float32 run, since low-precision numerical gradients
    # are too imprecise (so only the gradient, not the op, is verified). Asserts all three Jacobian
    # pairs agree with a threshold of 1e-10 for float64, 5e-2 for inputs of 512 or more elements,
    # otherwise 5e-3. Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      if data_format == "NCHW":
        np_input = self._NHWCToNCHW(np_input)
      jacob_a, jacob_n = self._computeGradient(np_input, bias, dtype,
                                               data_format)
      input_jacob_a, bias_jacob_a, grad_jacob_a = jacob_a
      input_jacob_n, bias_jacob_n, grad_jacob_n = jacob_n

      if dtype in [np.float16, dtypes.bfloat16.as_numpy_dtype]:
        # Compare fp16/bf16 analytical gradients to fp32 numerical gradients,
        # since fp16/bf16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker, but pragmatic, check (in particular, it does not test
        # the op itself, only its gradient).
        _, jacob_n = self._computeGradient(np_input, bias, np.float32,
                                           data_format)
        input_jacob_n, bias_jacob_n, grad_jacob_n = jacob_n

      if dtype == dtypes.float64:
        threshold = 1e-10
      elif np_input.size >= 512:
        # The 5e-3 threshold seems to have been marginal in these cases, and
        # small changes in the test were pushing it over the limit.
        threshold = 5e-2
      else:
        threshold = 5e-3
      self.assertAllClose(input_jacob_a, input_jacob_n, threshold, threshold)
      self.assertAllClose(bias_jacob_a, bias_jacob_n, threshold, threshold)
      self.assertAllClose(grad_jacob_a, grad_jacob_n, threshold, threshold)

  def testGradientTensor2D(self):
    # [FUNCTION DOC]
    # Scenario: gradient check on a three by two input and two-element bias, channels-last, CPU and
    # GPU, for float16, float32, float64 and bfloat16.
    for (data_format, use_gpu) in ("NHWC", False), ("NHWC", True):
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.bfloat16):
        np_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            dtype=dtype.as_numpy_dtype).reshape(3, 2)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  def testGradientTensor3D(self):
    # [FUNCTION DOC]
    # Scenario: gradient check on a one by three by two input and two-element bias, in both
    # channels-last and channels-first formats, CPU and GPU, for four floating dtypes.
    for (data_format, use_gpu) in [("NHWC", False), ("NHWC", True),
                                   ("NCHW", False), ("NCHW", True)]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.bfloat16):
        # pylint: disable=too-many-function-args
        np_input = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dtype=dtype.as_numpy_dtype).reshape(1, 3, 2)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  def testGradientTensor4D(self):
    # [FUNCTION DOC]
    # Scenario: gradient check, channels-last on CPU only, for four floating dtypes, using three
    # four-dimensional inputs: a small one, a 512-element one with channel size two and one with
    # channel size 64 and a random 64-element bias. The inputs are created as float32 arrays.
    for (data_format, use_gpu) in [("NHWC", False)]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.bfloat16):
        np_input = np.arange(
            1.0, 49.0,
            dtype=dtype.as_numpy_dtype).reshape([2, 3, 4, 2]).astype(np.float32)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)
        np_input = np.arange(
            1.0, 513.0,
            dtype=dtype.as_numpy_dtype).reshape([64, 2, 2,
                                                 2]).astype(np.float32)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)
        np_input = np.arange(
            1.0, 513.0,
            dtype=dtype.as_numpy_dtype).reshape([2, 2, 2,
                                                 64]).astype(np.float32)
        self._testGradient(np_input,
                           np.random.rand(64).astype(dtype.as_numpy_dtype),
                           dtype, data_format, use_gpu)

  def testGradientTensor5D(self):
    # [FUNCTION DOC]
    # Scenario: gradient check on a five-dimensional input with a two-element bias, in both data
    # formats, CPU and GPU, for four floating dtypes.
    for (data_format, use_gpu) in [("NHWC", False), ("NHWC", True),
                                   ("NCHW", False), ("NCHW", True)]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.bfloat16):
        np_input = np.arange(
            1.0, 49.0,
            dtype=dtype.as_numpy_dtype).reshape([1, 2, 3, 4,
                                                 2]).astype(np.float32)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  def test1x1Image(self):
    # [FUNCTION DOC]
    # Scenario: gradient check on a four by one by one by 32 float32 input with a random 32-element
    # bias, on CPU, for both channels-last and channels-first formats.
    for (data_format, use_gpu) in [("NHWC", False), ("NCHW", False)]:
      np_input = np.arange(1.0, 129.0).reshape([4, 1, 1, 32]).astype(np.float32)
      self._testGradient(np_input,
                         np.random.rand(32).astype(np.float32), dtypes.float32,
                         data_format, use_gpu)

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: forward checks on empty inputs (a zero-size dimension in several positions, for
    # two and three dimensions) with random bias of matching length. Seeds numpy with seven.
    np.random.seed(7)
    for shape in (0, 0), (2, 0), (0, 2), (4, 3, 0), (4, 0, 3), (0, 4, 3):
      self._testAll(np.random.randn(*shape), np.random.randn(shape[-1]))

  def testEmptyGradient(self):
    # [FUNCTION DOC]
    # Scenario: float64 gradient checks on empty two-dimensional inputs (channels-last, CPU and
    # GPU) and empty three-dimensional inputs (both data formats, CPU and GPU), with random bias
    # of matching length. Unseeded random numbers.
    for (data_format, use_gpu) in ("NHWC", False), ("NHWC", True):
      for shape in (0, 0), (2, 0), (0, 2):
        self._testGradient(
            np.random.randn(*shape), np.random.randn(shape[-1]), dtypes.float64,
            data_format, use_gpu)

    for (data_format, use_gpu) in [("NHWC", False), ("NHWC", True),
                                   ("NCHW", False), ("NCHW", True)]:
      for shape in (4, 3, 0), (4, 0, 3), (0, 4, 3):
        self._testGradient(
            np.random.randn(*shape), np.random.randn(shape[-1]), dtypes.float64,
            data_format, use_gpu)
