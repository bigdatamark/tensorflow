# [MODULE DOC]
# Test module for the rectifier family of activation ops in the nn_ops module: relu, relu6,
# leaky_relu, elu, selu and crelu, plus their gradients. It defines one test case class per
# activation. Each compares the TensorFlow result with a NumPy reference implementation across
# data types (and CPU versus GPU placement), checks NaN and empty-input handling, and verifies
# first and second order gradients numerically with the v2 gradient checker.
# Importing it imports the nn gradient registrations for their side effect; running it as a
# script starts the standard TensorFlow test runner. It is a test module, not a helper.
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
"""Tests for Relu and ReluGrad."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def _elu_grad_grad(activation):
  # [FUNCTION DOC]
  # HELPER: reference value used by the ELU second-derivative test.
  # Computes the analytic second derivative of the exponential linear unit at a scalar input.
  # Input: activation, a plain number. Returns the exponential of the input when it is negative,
  # otherwise zero (the second derivative of the linear branch). No side effects, no exceptions.
  if activation < 0:
    return np.exp(activation)
  return 0


class ReluTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the plain relu op: values against a NumPy reference for several dtypes on
  # CPU and GPU, quantized int8 shape restrictions, NaN propagation, empty tensors, and first
  # and second order gradients. It holds no state; helper methods build reference outputs.

  def _npRelu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: NumPy reference implementation for the relu tests.
    # Input: np_features, a NumPy array. Returns an array of the same shape holding the
    # element-wise maximum of the input and zero. No side effects.
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    # [FUNCTION DOC]
    # Sanity-checks the NumPy reference: feeds a small 2x5 array with mixed signs into the
    # reference relu and asserts the result equals the hand-written expected array with negatives
    # replaced by zero.
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testRelu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for the relu value tests.
    # Input: np_features, a NumPy array. Runs the TensorFlow relu op on it, computes the NumPy
    # reference, and asserts the two are numerically close and have the same shape. Returns
    # nothing; fails the test on mismatch.
    np_relu = self._npRelu(np_features)
    tf_relu = nn_ops.relu(np_features)
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, tf_relu)

  def testNumbersCPU(self):
    # [FUNCTION DOC]
    # For each of int32, int64, float16, float32, float64 and bfloat16, builds a 2x5 array of
    # mixed-sign values, pins execution to the CPU device even if a GPU kernel exists, and checks
    # relu against the reference through the shared helper.
    for t in [
        np.int32, np.int64, np.float16, np.float32, np.float64,
        dtypes.bfloat16.as_numpy_dtype
    ]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    # [FUNCTION DOC]
    # Same value check as the CPU test but without device pinning, for float16, float32, float64
    # and bfloat16. Skips the test when no GPU is available.
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testReluInt8x4GoodShape(self):
    # [FUNCTION DOC]
    # Quantized int8 relu with a valid element count (8, a multiple of four). Requires a CUDA GPU
    # and is skipped otherwise. Builds a constant of quantized 8-bit integers, applies relu, and
    # asserts the values match the NumPy reference and the shapes match.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest("No GPU available")
    inputs = np.array([[-50, 7, 23, 0], [-1, -5, 6, 11]])
    np_relu = self._npRelu(inputs)
    tf_relu = nn_ops.relu(constant_op.constant(inputs, dtypes.qint8))
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, tf_relu)

  @test_util.disable_xla("b/123338077")  # Passes with XLA
  def testReluInt8x4BadShape(self):
    # [FUNCTION DOC]
    # Quantized int8 relu with an element count that is not a multiple of four. Requires a CUDA
    # GPU, otherwise skipped; disabled under XLA. Evaluates relu on a 9-element and then a
    # 17-element tensor and asserts each raises an invalid-argument error whose message says the
    # tensor size must be a multiple of four and reports the actual size.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest("No GPU available")
    inputs = constant_op.constant(
        np.array([[-50, 7, 23], [0, 1, -5], [6, -2, 11]]), dtypes.qint8)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Tensor size must be a multiple of 4 for Relu<qint8>. Got 9"):
      self.evaluate(nn_ops.relu(inputs))

    inputs = constant_op.constant(
        np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -8, 7, -6, 5, -4, 3, -2, 1]),
        dtypes.qint8)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Tensor size must be a multiple of 4 for Relu<qint8>. Got 17"):
      self.evaluate(nn_ops.relu(inputs))

  def testNoElement(self):
    # [FUNCTION DOC]
    # Checks relu on an empty 2x0 float32 array: output must equal the (empty) reference and have
    # the same shape.
    self._testRelu(np.array([[], []], dtype=np.float32))

  @test_util.disable_xla("b/157978028: Does not yet pass with XLA")
  def testNaNPropagation(self):
    # [FUNCTION DOC]
    # For float16, float32 and float64, feeds inputs containing not-a-number entries and asserts
    # relu output matches the reference (NaNs stay NaN). Disabled under XLA.
    for t in [np.float16, np.float32, np.float64]:
      self._testRelu(np.array([-1, np.nan, 1, np.nan]).astype(t))

  # The gradient test for ReLU is a bit tricky as the derivative is not well
  # defined at around zero and we want to avoid that in terms of input values.
  def testGradientFloat32(self):
    # [FUNCTION DOC]
    # Numerically checks the relu gradient for float32 input in column-major layout, avoiding
    # values near zero, using a finite-difference step of 1/1024. Asserts the maximum error
    # between theoretical and numeric Jacobians is below 1e-6.
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(
          nn_ops.relu, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-6)

  # The gradient test for ReLU is a bit tricky as the derivative is not well
  # defined at around zero and we want to avoid that in terms of input values.
  def testGradientFloat16(self):
    # [FUNCTION DOC]
    # Same gradient check as the float32 variant but for float16 input, with the checker's
    # default step size. Asserts the maximum Jacobian error is below 1e-6.
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float16,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu, [x]))
    self.assertLess(err, 1e-6)

  def testGradientFloat64(self):
    # [FUNCTION DOC]
    # Gradient check of relu for float64 input with step 1/1024; asserts the maximum Jacobian
    # error is below 1e-15.
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(
          nn_ops.relu, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-15)

  def testGradGradFloat32(self):
    # [FUNCTION DOC]
    # Second order gradient check for float32: defines a local function returning the first
    # derivative of relu computed with a gradient tape, then numerically differentiates that
    # function and asserts the maximum error is below 1e-4.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: first-derivative function whose Jacobian the grad-grad test checks.
        # Input: x, a float32 tensor (asserted). Watches it on a gradient tape, applies relu and
        # returns the gradient of the output with respect to x.
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    # [FUNCTION DOC]
    # Second order gradient check for float64, built like the float32 version: numerically
    # differentiates the tape-computed first derivative of relu with step 1/1024 and asserts the
    # maximum error is below 1e-10.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: first-derivative function for the float64 grad-grad check.
        # Input: x, a float64 tensor (asserted). Returns the tape gradient of relu(x) with
        # respect to x.
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-10)

  def testGradientScalar(self):
    # [FUNCTION DOC]
    # End-to-end training check: a variable starting at 100 with loss equal to relu(x) squared is
    # minimized by one gradient descent step with learning rate 0.25. Initializes variables,
    # runs the step and asserts the variable becomes 50 (100 minus 0.25 times gradient 200).
    x = variables.Variable(100.)

    def loss():
      # [FUNCTION DOC]
      # HELPER: loss closure for the optimizer; returns the square of relu applied to the
      # captured variable x. Takes no arguments.
      return nn_ops.relu(x)**2

    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.25)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(optimizer.minimize(loss))
    self.assertAllClose(x.read_value(), 50.0)

  def testGradientNoElement(self):
    # [FUNCTION DOC]
    # Gradient of relu on an empty 2x0 float32 input: computes the Jacobian of a tape-based
    # gradient function and asserts the first Jacobian entry equals an empty array of shape 0x0.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: returns the tape gradient of relu(x) with respect to x for the empty-input
        # gradient test. Input: x, a float tensor.
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray([[], []], dtype=np.float32)
      z = list(gradient_checker_v2.compute_gradient(f, [x]))[0][0]
      self.assertAllEqual(z, np.reshape(x, (0, 0)))


class Relu6Test(test.TestCase):
  # [CLASS DOC]
  # Test case for relu6 (relu clipped at six): value checks against a NumPy reference for several
  # dtypes on CPU and GPU, NaN propagation and numerical gradient checks. Holds no state.

  def _npRelu6(self, np_features):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for relu6. Input: np_features, a NumPy array. Returns an array of
    # the same shape with each element clamped to the range zero to six. No side effects.
    sixes = np.copy(np_features)
    sixes.fill(6.0)
    return np.minimum(
        np.maximum(np_features, np.zeros(np_features.shape)), sixes)

  def testNpRelu6(self):
    # [FUNCTION DOC]
    # Sanity-checks the relu6 reference on a small 2x5 array containing negatives, values at and
    # above six, and asserts the clamped expected array.
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 6.0], [0.1, 0.0, 6.0, 0.0, 0.9]]),
        self._npRelu6(
            np.array([[-0.9, 0.7, -0.5, 0.3, 6.0], [0.1, -0.3, 6.5, -0.7,
                                                    0.9]])))

  def _testRelu6(self, np_features):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine. Input: np_features, a NumPy array. Applies the
    # TensorFlow relu6 op and asserts it is close to the reference with equal shape. Returns
    # nothing.
    np_relu6 = self._npRelu6(np_features)
    tf_relu6 = nn_ops.relu6(np_features)
    self.assertAllClose(np_relu6, tf_relu6)
    self.assertShapeEqual(np_relu6, tf_relu6)

  def testNumbersCPU(self):
    # [FUNCTION DOC]
    # Checks relu6 on mixed-sign 2x5 data for int32, int64, float16, float32, float64 and
    # bfloat16 with execution pinned to the CPU device.
    for t in [
        np.int32, np.int64, np.float16, np.float32, np.float64,
        dtypes.bfloat16.as_numpy_dtype
    ]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu6(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    # [FUNCTION DOC]
    # Checks relu6 for float16, float32, float64 and bfloat16 without device pinning. Skipped
    # when no GPU is available.
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ]:
      self._testRelu6(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  @test_util.disable_xla("b/157978028: Does not yet pass with XLA")
  def testNaNPropagation(self):
    # [FUNCTION DOC]
    # Feeds float16, float32 and float64 inputs with not-a-number entries (and a value above six)
    # to relu6 and asserts agreement with the reference. Disabled under XLA.
    for t in [np.float16, np.float32, np.float64]:
      self._testRelu6(np.array([-1, np.nan, 1, 7, np.nan]).astype(t))

  # The gradient test for ReLU6 is a bit tricky as the derivative is
  # not well defined at around zero and six and we want to avoid that
  # in terms of input values.
  def testGradientFloat32(self):
    # [FUNCTION DOC]
    # Numerical gradient check of relu6 for float32 input chosen away from the kinks at zero and
    # six. Asserts the maximum Jacobian error is below 1e-4.
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    # [FUNCTION DOC]
    # Numerical gradient check of relu6 for float64 input away from the kinks; asserts the
    # maximum Jacobian error is below 1e-10.
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    self.assertLess(err, 1e-10)


class LeakyReluTest(test.TestCase):
  # [CLASS DOC]
  # Test case for leaky_relu: reference comparison across dtypes and devices with a configurable
  # negative slope, NaN propagation, first and second order gradients, a training step, and
  # behavior for out-of-range slope values. Holds no state.

  def _npLeakyRelu(self, np_features, alpha=0.1):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for leaky relu.
    # KNOWN ISSUE: the element-wise maximum formulation equals the true leaky relu only for
    # slopes between zero and one; for other slopes it differs from the op's definition.
    # Inputs: np_features, a NumPy array; alpha, the negative slope (default 0.1). Returns the
    # element-wise maximum of the input and alpha times the input.
    return np.maximum(np_features, alpha * np_features)

  def testNpLeakyRelu(self):
    # [FUNCTION DOC]
    # Sanity-checks the leaky relu reference with slope 0.1 on a small mixed-sign 2x5 array
    # against a hand-written expected array.
    self.assertAllClose(
        np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                  [0.1, -0.03, 0.5, -0.07, 0.9]]),
        self._npLeakyRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]]),
            alpha=0.1))

  def _testLeakyRelu(self, np_features, alpha):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine. Inputs: np_features, a NumPy array; alpha, the slope.
    # Runs the TensorFlow leaky relu op, then asserts closeness to the reference using
    # dtype-dependent tolerances and equal shapes. Returns nothing.
    np_leaky_relu = self._npLeakyRelu(np_features, alpha)
    tf_leaky_relu = nn_ops.leaky_relu(np_features, alpha)
    self.assertAllCloseAccordingToType(np_leaky_relu, tf_leaky_relu)
    self.assertShapeEqual(np_leaky_relu, tf_leaky_relu)

  def testNumbersCPU(self):
    # [FUNCTION DOC]
    # Checks leaky relu with slope 0.2 on mixed-sign data for int32, int64, float16, float32,
    # float64 and bfloat16 with execution pinned to the CPU device.
    for t in [
        np.int32, np.int64, np.float16, np.float32, np.float64,
        dtypes.bfloat16.as_numpy_dtype
    ]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testLeakyRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            alpha=0.2)

  def testNumbersGPU(self):
    # [FUNCTION DOC]
    # Checks leaky relu with slope 0.1 for float16, float32, float64 and bfloat16 without device
    # pinning. Skipped when no GPU is available.
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ]:
      self._testLeakyRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          alpha=0.1)

  def testNaNPropagation(self):
    # [FUNCTION DOC]
    # Feeds float16, float32 and float64 inputs containing not-a-number entries with slope 0.2
    # and asserts agreement with the reference.
    for t in [np.float16, np.float32, np.float64]:
      self._testLeakyRelu(np.array([-1, np.nan, 1, np.nan]).astype(t),
                          alpha=0.2)

  # The gradient test for Leaky ReLU is a bit tricky as the derivative is not
  # well defined at around zero and we want to avoid that in terms of input
  # values.
  def testGradientFloat32(self):
    # [FUNCTION DOC]
    # Numerical gradient check of leaky relu (default slope) for float32 input away from zero;
    # asserts the maximum Jacobian error is below 1e-4.
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    # [FUNCTION DOC]
    # Numerical gradient check of leaky relu for float64 input away from zero; asserts the
    # maximum Jacobian error is below 1e-10.
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x]))
    self.assertLess(err, 1e-10)

  def testGradGradFloat32(self):
    # [FUNCTION DOC]
    # Second order gradient check of leaky relu for float32: numerically differentiates a local
    # function that returns the tape gradient of the op, and asserts the maximum error is below
    # 1e-4.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: first-derivative function for the grad-grad check. Input: x, a float32 tensor
        # (asserted). Returns the tape gradient of leaky relu(x) with respect to x.
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.leaky_relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    # [FUNCTION DOC]
    # Second order gradient check of leaky relu for float64; same method as the float32 variant
    # with a tolerance of 1e-10.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: first-derivative function for the float64 grad-grad check. Input: x, a float64
        # tensor (asserted). Returns the tape gradient of leaky relu(x) with respect to x.
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.leaky_relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-10)

  def testGradientScalar(self):
    # [FUNCTION DOC]
    # Training check: a variable starting at -100 with loss equal to the square of leaky relu
    # with slope 0.05 is minimized by one gradient descent step with learning rate 0.2.
    # Asserts the variable becomes -99.9.
    x = variables.Variable(-100.)

    def loss():
      # [FUNCTION DOC]
      # HELPER: loss closure; returns the square of leaky relu (slope 0.05) of the captured
      # variable x. Takes no arguments.
      return nn_ops.leaky_relu(x, 0.05)**2

    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.2)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(optimizer.minimize(loss))
    self.assertAllClose(x.read_value(), -99.9)

  def testUnexpectedAlphaValue(self):
    # [FUNCTION DOC]
    # Checks leaky relu with slopes outside the usual range: with slope 10 negative inputs are
    # multiplied by ten, and with slope -10 they are multiplied by minus ten (so they become
    # positive). Asserts both against hand-written expected arrays.
    self.assertAllClose(
        np.array([[-9.0, 0.7, -5.0, 0.3, -0.1], [0.1, -3.0, 0.5, -27.0, 0.9]]),
        nn_ops.leaky_relu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.01],
                      [0.1, -0.3, 0.5, -2.7, 0.9]]),
            alpha=10))
    self.assertAllClose(
        np.array([[9.0, 0.7, 5.0, 0.3, 0.1], [0.1, 3.0, 0.5, 27.0, 0.9]]),
        nn_ops.leaky_relu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.01],
                      [0.1, -0.3, 0.5, -2.7, 0.9]]),
            alpha=-10))


class EluTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the exponential linear unit (elu): reference comparison across dtypes and
  # devices, NaN propagation, first order gradients, and second order gradients computed both
  # analytically and numerically. Holds no state.

  def _npElu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for elu. Input: np_features, a NumPy array. Returns an array where
    # negative entries become exp(x) minus one and non-negative entries are unchanged.
    return np.where(np_features < 0, np.exp(np_features) - 1, np_features)

  def testNpElu(self):
    # [FUNCTION DOC]
    # Sanity-checks the elu reference on a small mixed-sign 2x5 array against hand-computed
    # expected values.
    self.assertAllClose(
        np.array([[-0.59343034025, 0.7, -0.39346934028, 0.3, -0.09516258196],
                  [0.1, -0.25918177931, 0.5, -0.5034146962, 0.9]]),
        self._npElu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testElu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine. Input: np_features, a NumPy array. Applies the
    # TensorFlow elu op and asserts closeness (with dtype-dependent tolerance) to the reference
    # and equal shape. Returns nothing.
    np_elu = self._npElu(np_features)
    tf_elu = nn_ops.elu(np_features)
    self.assertAllCloseAccordingToType(np_elu, tf_elu)
    self.assertShapeEqual(np_elu, tf_elu)

  def testNumbersCPU(self):
    # [FUNCTION DOC]
    # Checks elu on mixed-sign data for float16, float32, float64 and bfloat16 with execution
    # pinned to the CPU device (integer types are not tested for elu).
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testElu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    # [FUNCTION DOC]
    # Checks elu for float16, float32, float64 and bfloat16 without device pinning. Skipped when
    # no GPU is available.
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ]:
      self._testElu(np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNaNPropagation(self):
    # [FUNCTION DOC]
    # Feeds float16, float32 and float64 inputs containing not-a-number entries to elu and
    # asserts agreement with the reference.
    for t in [np.float16, np.float32, np.float64]:
      self._testElu(np.array([-1, np.nan, 1, np.nan]).astype(t))

  def testGradientFloat32(self):
    # [FUNCTION DOC]
    # Numerical gradient check of elu for float32 input in column-major layout; asserts the
    # maximum Jacobian error is below 1e-4.
    with self.cached_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = np.asarray(x_val, dtype=np.float32, order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.elu, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    # [FUNCTION DOC]
    # Numerical gradient check of elu for float64 input; asserts the maximum Jacobian error is
    # below 1e-6.
    with self.cached_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = np.asarray(x_val, dtype=np.float64, order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.elu, [x]))
    self.assertLess(err, 1e-6)

  def testGradGrad(self):
    # [FUNCTION DOC]
    # Compares the second derivative of elu computed with a persistent gradient tape at the
    # scalar points -1, -0.5, 0.5 and 1 against the analytic helper value, asserting an absolute
    # error below 1e-4 at each point.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: computes the second derivative of elu at x. Input: x, a float tensor. Uses a
        # persistent tape to take the gradient of the op, then the gradient of that gradient,
        # and returns the latter.
        with backprop.GradientTape(persistent=True) as tape:
          tape.watch(x)
          y = nn_ops.elu(x)
          dy = tape.gradient(y, x)
        return tape.gradient(dy, x)

      for x in [-1., -0.5, 0.5, 1.]:
        got = self.evaluate(f(constant_op.constant(x)))
        want = _elu_grad_grad(x)
        err = np.abs(got - want)
        self.assertLess(err, 1e-4)

  def testGradGradFloat32(self):
    # [FUNCTION DOC]
    # Numerical second order gradient check of elu for float32: differentiates a local function
    # returning the tape gradient of the op; asserts the maximum error is below 1e-4.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: first-derivative function for the grad-grad check. Input: x, a float32 tensor
        # (asserted). Returns the tape gradient of elu(x) with respect to x.
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.elu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    # [FUNCTION DOC]
    # Numerical second order gradient check of elu for float64; same method as the float32
    # variant with a tolerance of 1e-6.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: first-derivative function for the float64 grad-grad check. Input: x, a float64
        # tensor (asserted). Returns the tape gradient of elu(x) with respect to x.
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.elu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-6)


class SeluTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the scaled exponential linear unit (selu): reference comparison across dtypes,
  # first order gradients and second order gradients checked numerically. Holds no state.

  def _npSelu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for selu. Input: np_features, a NumPy array. Returns an array where
    # negative entries become the scaled alpha times exp(x) minus one, and non-negative entries
    # are multiplied by the scale constant (about 1.0507).
    scale = 1.0507009873554804934193349852946
    scale_alpha = 1.7580993408473768599402175208123
    return np.where(np_features < 0, scale_alpha * (np.exp(np_features) - 1),
                    scale * np_features)

  def testNpSelu(self):
    # [FUNCTION DOC]
    # Sanity-checks the selu reference on a small mixed-sign 2x5 array against hand-computed
    # expected values.
    self.assertAllClose(
        np.array([[-1.0433095, 0.73549069, -0.6917582, 0.3152103, -0.16730527],
                  [0.1050701, -0.45566732, 0.5253505, -0.88505305, 0.9456309]]),
        self._npSelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testSelu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine. Input: np_features, a NumPy array. Applies the
    # TensorFlow selu op and asserts closeness (dtype-dependent tolerance) to the reference and
    # equal shape. Returns nothing.
    np_selu = self._npSelu(np_features)
    tf_selu = nn_ops.selu(np_features)
    self.assertAllCloseAccordingToType(np_selu, tf_selu)
    self.assertShapeEqual(np_selu, tf_selu)

  def testNumbers(self):
    # [FUNCTION DOC]
    # For float16, float32, float64 and bfloat16, checks selu on mixed-sign data twice: once with
    # default placement and once with execution pinned to the CPU device.
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testSelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))
      # Force executed on CPU in case GPU kernels are available.
      with ops.device("/device:CPU:0"):
        self._testSelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testGradientFloat32(self):
    # [FUNCTION DOC]
    # Numerical gradient check of selu for float32 input with step 1/1024; asserts the maximum
    # Jacobian error is below 1e-4.
    with self.cached_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = np.asarray(x_val, dtype=np.float32, order="F")
      err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(
          nn_ops.selu, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    # [FUNCTION DOC]
    # Numerical gradient check of selu for float64 input with default step; asserts the maximum
    # Jacobian error is below 1e-6.
    with self.cached_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = np.asarray(x_val, dtype=np.float64, order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.selu, [x]))
    self.assertLess(err, 1e-6)

  def testGradGradFloat32(self):
    # [FUNCTION DOC]
    # Numerical second order gradient check of selu for float32 with step 1/1024: differentiates
    # a local function returning the tape gradient of the op; asserts error below 1e-4.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: first-derivative function for the grad-grad check. Input: x, a float32 tensor
        # (asserted). Returns the tape gradient of selu(x) with respect to x.
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.selu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    # [FUNCTION DOC]
    # Numerical second order gradient check of selu for float64 with default step; asserts the
    # maximum error is below 1e-6.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: first-derivative function for the float64 grad-grad check. Input: x, a float64
        # tensor (asserted). Returns the tape gradient of selu(x) with respect to x.
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.selu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-6)


class CreluTest(test.TestCase):
  # [CLASS DOC]
  # Test case for crelu (concatenated relu, which concatenates relu of the input and relu of its
  # negation along an axis): static output shape, values across dtypes and devices, and explicit
  # axis selection. Holds no state.

  def testCreluShape(self):
    # [FUNCTION DOC]
    # Applies crelu to a random normal tensor of shape 50 by 5 by 7 by 10 and asserts the static
    # output shape doubles the last dimension to 50 by 5 by 7 by 20.
    f = random_ops.random_normal([50, 5, 7, 10])
    t = nn_ops.crelu(f)
    self.assertEqual([50, 5, 7, 20], t.get_shape())

  def _testCrelu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine. Input: np_features, a NumPy array. Builds the reference
    # by concatenating relu of the input and relu of its negation along the last axis, applies
    # the TensorFlow crelu with its default axis, and asserts close values and equal shapes.
    np_relu = np.maximum(np_features, np.zeros_like(np_features))
    np_neg_relu = np.maximum(-np_features, np.zeros_like(np_features))
    np_crelu = np.concatenate((np_relu, np_neg_relu),
                              len(np_features.shape) - 1)

    tf_crelu = nn_ops.crelu(np_features)

    self.assertAllClose(np_crelu, tf_crelu)
    self.assertShapeEqual(np_crelu, tf_crelu)

  def testNumbersCPU(self):
    # [FUNCTION DOC]
    # Checks crelu on mixed-sign data for int32, int64, float16, float32, float64 and bfloat16
    # with execution pinned to the CPU device.
    for t in [
        np.int32, np.int64, np.float16, np.float32, np.float64,
        dtypes.bfloat16.as_numpy_dtype
    ]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testCrelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    # [FUNCTION DOC]
    # Checks crelu for float16, float32, float64 and bfloat16 without device pinning. Skipped
    # when no GPU is available.
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ]:
      self._testCrelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersWithAxis0(self):
    # [FUNCTION DOC]
    # Applies crelu along axis zero to a 2x5 integer array and asserts the result is the 4x5
    # array with the relu rows stacked above the relu-of-negation rows (in reversed row order).
    tf_crelu = nn_ops.crelu(
        np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]), axis=0)
    np_crelu = np.array([[0, 7, 0, 3, 0], [1, 0, 5, 0, 9], [9, 0, 5, 0, 1],
                         [0, 3, 0, 7, 0]])
    self.assertAllEqual(np_crelu, tf_crelu)

  def testNumbersWithAxis1(self):
    # [FUNCTION DOC]
    # Applies crelu along axis one to a 2x5 integer array and asserts the result is the 2x10
    # array with each row's relu values followed by the relu of its negated values.
    tf_crelu = nn_ops.crelu(
        np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]), axis=1)
    np_crelu = np.array([[0, 7, 0, 3, 0, 9, 0, 5, 0, 1],
                         [1, 0, 5, 0, 9, 0, 3, 0, 7, 0]])
    self.assertAllEqual(np_crelu, tf_crelu)


if __name__ == "__main__":
  test.main()
