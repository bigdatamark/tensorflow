# [MODULE DOC]
# Unit tests for the gradient functions registered in math_grad.py. The tests cover the helper
# that infers broadcast reduction axes, and the gradients of squared difference, abs (real and
# complex), reduce min/max/prod, reduce euclidean norm, segment min/max/prod, floor mod,
# div_no_nan, mul_no_nan, xlogy, xlog1py, xdivy, pow and nextafter. Most tests build a small
# tensor, compute a numeric-versus-analytic gradient error with the gradient checker (v1 or v2)
# or compare tape/graph gradients with known values, and assert the error is small or the values
# match. Many tests are decorated to run only in legacy graph mode, others in both graph and
# eager modes. There are no import-time side effects other than defining the test classes; the
# module runs the TensorFlow test main when executed as a script.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Python ops defined in math_grad.py."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_grad
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class InferGradientReductionAxes(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the private helper in math_grad that infers, for a binary
  # broadcasting op, which axes of each input gradient must be summed. Holds no state.

  @parameterized.parameters(
      (None, None, None, None),
      (None, [], None, None),
      ([], [], [], []),
      ([], [None], [0], []),
      ([None], [None], None, None),
      ([None, 1], [None], [1], [0]),
      ([None, 1], [1, None], [1], [0]),
      ([None, 1], [2, None], None, None),
      ([2], [1], [], [0]),
      ([2], [2], [], []),
      ([3, 1, 5, 1, 7], [2, 1, 4, 7], [1, 3], [0, 2]),
  )
  def testShapes(self, x_shape, y_shape, expected_x_axes, expected_y_axes):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: two possibly-unknown shapes (a nothing value means unknown
    # rank, a nothing dimension means unknown size) and the expected reduction axes for each
    # (nothing means they cannot be determined statically). It calls the helper on the shapes in
    # both orders and asserts that results are symmetric (swapping the arguments swaps the
    # outputs) and that they equal the expected axes. Returns nothing; fails on assertion error.
    x_axes1, y_axes1 = math_grad._InferGradientReductionAxes(
        tensor_shape.TensorShape(x_shape), tensor_shape.TensorShape(y_shape))
    y_axes2, x_axes2 = math_grad._InferGradientReductionAxes(
        tensor_shape.TensorShape(y_shape), tensor_shape.TensorShape(x_shape))
    self.assertEqual(x_axes1, x_axes2)
    self.assertEqual(y_axes1, y_axes2)
    self.assertEqual(expected_x_axes, x_axes1)
    self.assertEqual(expected_y_axes, y_axes1)


class SquaredDifferenceOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests the gradient of the squared difference op with broadcasting between operands of
  # different rank. Stateless; uses a cached session in graph mode.

  def _testGrad(self, left_shape, right_shape):
    # [FUNCTION DOC]
    # HELPER: shared checker for the squared difference gradient test.
    # Inputs: shapes (lists of ints) of the left and right operands. Picks the longer shape as
    # the output shape, creates random normal constants of each shape, builds the squared
    # difference, and uses the numeric gradient checker to compute the error of the gradient
    # with respect to each operand. Asserts both errors are below a tiny tolerance. Returns
    # nothing. Side effects: draws from the global numpy random state, adds ops to the graph.
    # KNOWN ISSUE: the output shape is chosen by comparing ranks only, so for equal-rank shapes
    # that differ in size the right shape is used even if it is not the broadcast result.

    if len(left_shape) > len(right_shape):
      output_shape = left_shape
    else:
      output_shape = right_shape
    l = np.random.randn(*left_shape)
    r = np.random.randn(*right_shape)

    with self.cached_session():
      left_tensor = constant_op.constant(l, shape=left_shape)
      right_tensor = constant_op.constant(r, shape=right_shape)
      output = math_ops.squared_difference(left_tensor, right_tensor)
      left_err = gradient_checker.compute_gradient_error(
          left_tensor, left_shape, output, output_shape, x_init_value=l)
      right_err = gradient_checker.compute_gradient_error(
          right_tensor, right_shape, output, output_shape, x_init_value=r)
    self.assertLess(left_err, 1e-10)
    self.assertLess(right_err, 1e-10)

  @test_util.run_deprecated_v1
  def testGrad(self):
    # [FUNCTION DOC]
    # Runs the squared difference gradient check for two broadcasting cases: a rank-4 left
    # operand against a rank-2 right operand, and a rank-2 left operand against a rank-3 right
    # operand. Asserts the numeric and analytic gradients agree. Legacy graph mode only.
    self._testGrad([1, 2, 3, 2], [3, 2])
    self._testGrad([2, 4], [3, 2, 4])


class AbsOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests the gradient of the absolute value op for real and complex inputs, using random
  # values biased away from zero where abs is not differentiable. Stateless.

  def _biasedRandN(self, shape, bias=0.1, sigma=1.0):
    # [FUNCTION DOC]
    # HELPER: generates random test input for the abs gradient tests.
    # Inputs: shape (list of ints), bias (minimum distance from zero, default 0.1) and sigma
    # (standard deviation of the normal samples, default 1.0). Returns a numpy array of normal
    # samples, each pushed away from zero by the bias in the direction of its sign. Draws from
    # the global numpy random state.
    """Returns samples from a normal distribution shifted `bias` away from 0."""
    value = np.random.randn(*shape) * sigma
    return value + np.sign(value) * bias

  def _testGrad(self, shape, dtype=None, max_error=None, bias=None, sigma=None):
    # [FUNCTION DOC]
    # HELPER: shared checker for the abs gradient tests.
    # Inputs: shape of the input, dtype (real or complex), the maximum allowed gradient error,
    # and the bias and sigma used for random generation. Seeds the numpy random generator, builds
    # either a complex tensor from two independent biased real parts or a real tensor, applies
    # abs, and uses the numeric gradient checker to measure the gradient error. Asserts the error
    # is below max_error. Returns nothing.
    # KNOWN ISSUE: sigma defaults to nothing and is not passed on in the real-valued branch, and
    # bias defaults to nothing, so calling it without explicit bias would fail inside the
    # random generator; the defaults are unusable and callers always pass the values.
    np.random.seed(7)
    if dtype in (dtypes.complex64, dtypes.complex128):
      value = math_ops.complex(
          self._biasedRandN(
              shape, bias=bias, sigma=sigma),
          self._biasedRandN(
              shape, bias=bias, sigma=sigma))
    else:
      value = ops.convert_to_tensor(
          self._biasedRandN(
              shape, bias=bias), dtype=dtype)

    with self.cached_session():
      output = math_ops.abs(value)
      error = gradient_checker.compute_gradient_error(
          value, shape, output, output.get_shape().as_list())
    self.assertLess(error, max_error)

  @test_util.run_deprecated_v1
  def testComplexAbs(self):
    # [FUNCTION DOC]
    # Checks the abs gradient for float32 and complex64 inputs of shape three by three, first
    # with values biased away from zero and a tight error bound, then with values clustered near
    # zero (no bias, small spread) and a very loose bound to ensure the gradient stays stable
    # near the pole. Legacy graph mode only.
    # Bias random test values away from zero to avoid numeric instabilities.
    self._testGrad(
        [3, 3], dtype=dtypes.float32, max_error=2e-5, bias=0.1, sigma=1.0)
    self._testGrad(
        [3, 3], dtype=dtypes.complex64, max_error=2e-5, bias=0.1, sigma=1.0)

    # Ensure stability near the pole at zero.
    self._testGrad(
        [3, 3], dtype=dtypes.float32, max_error=100.0, bias=0.0, sigma=0.1)
    self._testGrad(
        [3, 3], dtype=dtypes.complex64, max_error=100.0, bias=0.0, sigma=0.1)


class MinOrMaxGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of reduce min and reduce max when the extreme value is tied (the same
  # tensor concatenated with itself). Stateless; graph mode.

  @test_util.run_deprecated_v1
  def testMinGradient(self):
    # [FUNCTION DOC]
    # Takes a one-element float tensor, concatenates it with itself, reduces by minimum, and
    # checks the numeric versus analytic gradient error is below 1e-4, verifying that ties split
    # the gradient correctly. Legacy graph mode only.
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    outputs = math_ops.reduce_min(array_ops.concat([inputs, inputs], 0))
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], outputs, [])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testMaxGradient(self):
    # [FUNCTION DOC]
    # Same scenario as the minimum test but with reduce by maximum over the doubled one-element
    # tensor: asserts the gradient error is below 1e-4. Legacy graph mode only.
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    outputs = math_ops.reduce_max(array_ops.concat([inputs, inputs], 0))
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], outputs, [])
      self.assertLess(error, 1e-4)


class MaximumOrMinimumGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of the elementwise maximum and minimum ops against a scalar bound.
  # Stateless; graph mode.

  @test_util.run_deprecated_v1
  def testMaximumGradient(self):
    # [FUNCTION DOC]
    # Applies elementwise maximum of a four-element float vector with the scalar three and
    # asserts the gradient error is below 1e-4. Legacy graph mode only.
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32)
    outputs = math_ops.maximum(inputs, 3.0)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [4], outputs, [4])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testMinimumGradient(self):
    # [FUNCTION DOC]
    # Applies elementwise minimum of a four-element float vector with the scalar two and
    # asserts the gradient error is below 1e-4. Legacy graph mode only.
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32)
    outputs = math_ops.minimum(inputs, 2.0)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [4], outputs, [4])
      self.assertLess(error, 1e-4)


class ProdGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests the gradient of reduce product for real and complex matrices, with default and
  # negative reduction axes. Stateless; graph mode.

  @test_util.run_deprecated_v1
  def testProdGradient(self):
    # [FUNCTION DOC]
    # Reduces a two by two float matrix to its full product and asserts the gradient error from
    # the numeric checker is below 1e-4. Legacy graph mode only.
    inputs = constant_op.constant([[1., 2.], [3., 4.]],
                                  dtype=dtypes.float32)
    outputs = math_ops.reduce_prod(inputs)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testProdGradientForNegativeAxis(self):
    # [FUNCTION DOC]
    # Reduces a two by two float matrix by product along the last axis, given as a negative
    # index, and asserts the gradient error is below 1e-4. Legacy graph mode only.
    inputs = constant_op.constant([[1., 2.], [3., 4.]],
                                  dtype=dtypes.float32)
    outputs = math_ops.reduce_prod(inputs, -1)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testProdGradientComplex(self):
    # [FUNCTION DOC]
    # For both complex64 and complex128, reduces a two by two complex matrix (including a
    # purely imaginary entry) to its full product and asserts the gradient error is below 1e-4.
    # Legacy graph mode only.
    for dtype in dtypes.complex64, dtypes.complex128:
      inputs = constant_op.constant([[1 + 3j, 2 - 1j], [3j, 4]],
                                    dtype=dtype)
      outputs = math_ops.reduce_prod(inputs)
      with self.cached_session():
        error = gradient_checker.compute_gradient_error(
            inputs, inputs.get_shape().as_list(),
            outputs, outputs.get_shape().as_list())
        self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testProdGradientForNegativeAxisComplex(self):
    # [FUNCTION DOC]
    # Complex counterpart of the negative-axis test: for complex64 and complex128 matrices,
    # reduces by product along the last axis and asserts the gradient error is below 1e-4.
    # Legacy graph mode only.
    for dtype in dtypes.complex64, dtypes.complex128:
      inputs = constant_op.constant([[1 + 3j, 2 - 1j], [3j, 4]],
                                    dtype=dtype)
      outputs = math_ops.reduce_prod(inputs, -1)
      with self.cached_session():
        error = gradient_checker.compute_gradient_error(
            inputs, inputs.get_shape().as_list(),
            outputs, outputs.get_shape().as_list())
        self.assertLess(error, 1e-4)


@test_util.run_all_in_graph_and_eager_modes
class EuclideanNormGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests the gradient of the reduce euclidean norm op in many shapes, axes and dtypes (float32
  # and float64), using the v2 gradient checker or a gradient tape. Every test method is run in
  # both graph and eager modes through the class decorator. Stateless.

  def testBasic(self):
    # [FUNCTION DOC]
    # For float32 and float64, computes the numeric versus analytic Jacobian of the norm of a
    # single-element vector and asserts the maximum error is below 1e-3.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([3], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testNegative(self):
    # [FUNCTION DOC]
    # Same as the basic test but with a negative single element, checking the gradient sign is
    # handled; asserts maximum Jacobian error below 1e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([-3], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testKeepdims(self):
    # [FUNCTION DOC]
    # Intended to cover the keep-dimensions option of the norm; computes the Jacobian error of
    # the norm of a single-element vector for float32 and float64 and asserts it is below 1e-3.
    # KNOWN ISSUE: despite the name, the keep-dimensions option is never passed, so this test is
    # an exact duplicate of the basic test and does not exercise that option.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([3], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testGradientChain(self):
    # [FUNCTION DOC]
    # Checks the gradient when the norm output is further multiplied by the constant five (a
    # chained gradient with a non-unit upstream gradient). Asserts maximum Jacobian error below
    # 1e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([3], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x) * 5, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testTwoElements(self):
    # [FUNCTION DOC]
    # Checks the norm gradient of a two-element vector with entries of mixed sign (three and
    # minus four); asserts maximum Jacobian error below 1e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([3, -4], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testNegativeZero(self):
    # [FUNCTION DOC]
    # Verifies signed-zero handling: for the vector one and negative zero, records the norm
    # under a gradient tape and asserts the gradient equals one and negative zero, and that the
    # reciprocal of the gradient matches the reciprocal of the expected answer (so the sign of
    # the zero is preserved). Runs for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([1.0, -0.0], dtype=dtype)

      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = math_ops.reduce_euclidean_norm(x)

      dx = tape.gradient(y, x)
      dx_answer = constant_op.constant([1.0, -0.0], dtype=dtype)
      self.assertAllClose(dx, dx_answer)
      self.assertAllClose(1.0 / dx, 1.0 / dx_answer)

  def testZeros(self):
    # [FUNCTION DOC]
    # Verifies the gradient at the all-zero vector (zero and negative zero): the tape gradient
    # of the norm is expected to be not-a-number in both entries, since the norm is not
    # differentiable there. Runs for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([0.0, -0.0], dtype=dtype)

      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = math_ops.reduce_euclidean_norm(x)

      dx = tape.gradient(y, x)
      dx_answer = constant_op.constant(
          [float("NaN"), float("NaN")], dtype=dtype)
      self.assertAllClose(dx, dx_answer)

  def test2D_1(self):
    # [FUNCTION DOC]
    # Jacobian check of the norm over all axes of a two by two matrix; asserts maximum error
    # below 1e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[-3, 5], [7, 11]], dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 1e-3)

  def test2D_2(self):
    # [FUNCTION DOC]
    # Jacobian check of the norm reduced along axis zero of a two by two matrix; asserts maximum
    # error below 1e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[-3, 5], [7, 11]], dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 0), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 1e-3)

  def test2D_3(self):
    # [FUNCTION DOC]
    # Jacobian check of the norm reduced along axis one of a two by two matrix; asserts maximum
    # error below 1e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[-3, 5], [7, 11]], dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 1), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 1e-3)

  def test2D_4(self):
    # [FUNCTION DOC]
    # Jacobian check of the norm reduced along axis one of a two by one column matrix; asserts
    # maximum error below 1e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[3], [4]], dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 1), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 1e-3)

  def test3D_1(self):
    # [FUNCTION DOC]
    # Jacobian check of the norm over all axes of a two by two by two tensor; asserts maximum
    # error below 2e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[[-3, 5], [7, 11]], [[13, 17], [19, 23]]],
                               dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 2e-3)

  def test3D_2(self):
    # [FUNCTION DOC]
    # Jacobian check of the norm reduced along axis zero of a two by two by two tensor; asserts
    # maximum error below 2e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[[-3, 5], [7, 11]], [[13, 17], [19, 23]]],
                               dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 0), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 2e-3)

  def test3D_3(self):
    # [FUNCTION DOC]
    # Jacobian check of the norm reduced along axis one of a two by two by two tensor; asserts
    # maximum error below 3e-3 (a looser bound than the neighbouring tests) for float32 and
    # float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[[-3, 5], [7, 11]], [[13, 17], [19, 23]]],
                               dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 1), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 3e-3)

  def test3D_4(self):
    # [FUNCTION DOC]
    # Jacobian check of the norm reduced along axis two of a two by two by two tensor; asserts
    # maximum error below 2e-3 for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[[-3, 5], [7, 11]], [[13, 17], [19, 23]]],
                               dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 2), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 2e-3)


class SegmentMinOrMaxGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of segment min and segment max, with distinct values and with ties.
  # Stateless; graph mode.

  @test_util.run_deprecated_v1
  def testSegmentMinGradient(self):
    # [FUNCTION DOC]
    # Takes data of three floats with segment ids zero, zero, one, applies segment min, and
    # asserts the gradient error (input size three, output size two) is below 1e-4. Legacy
    # graph mode only.
    data = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 1], dtype=dtypes.int64)
    segment_min = math_ops.segment_min(data, segment_ids)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(data, [3], segment_min,
                                                      [2])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testSegmentMaxGradient(self):
    # [FUNCTION DOC]
    # Same data and segmentation as the min test but with segment max; asserts the gradient
    # error is below 1e-4. Legacy graph mode only.
    data = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 1], dtype=dtypes.int64)
    segment_max = math_ops.segment_max(data, segment_ids)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(data, [3], segment_max,
                                                      [2])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testSegmentMinGradientWithTies(self):
    # [FUNCTION DOC]
    # Duplicates a one-element tensor into a two-element data vector in a single segment so the
    # minimum is tied, applies segment min, and asserts the gradient error with respect to the
    # original input is below 1e-4. Legacy graph mode only.
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    data = array_ops.concat([inputs, inputs], 0)
    segment_ids = constant_op.constant([0, 0], dtype=dtypes.int64)
    segment_min = math_ops.segment_min(data, segment_ids)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], segment_min,
                                                      [1])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testSegmentMaxGradientWithTies(self):
    # [FUNCTION DOC]
    # Tied-values scenario for segment max: a duplicated one-element tensor in a single
    # segment; asserts the gradient error with respect to the original input is below 1e-4.
    # Legacy graph mode only.
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    data = array_ops.concat([inputs, inputs], 0)
    segment_ids = constant_op.constant([0, 0], dtype=dtypes.int64)
    segment_max = math_ops.segment_max(data, segment_ids)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], segment_max,
                                                      [1])
      self.assertLess(error, 1e-4)


@test_util.run_all_in_graph_and_eager_modes
class SegmentProdGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests the gradient of segment product for several segmentations (no overlap, shared
  # segments, zeros in the data, and an empty segment). Runs in graph and eager modes via the
  # class decorator. Stateless.

  def _run_gradient_check(self, data, segment_ids):
    # [FUNCTION DOC]
    # HELPER: shared checker for the segment product gradient tests.
    # Inputs: data tensor (float matrix) and segment ids tensor (sorted integer vector). Builds
    # a function that applies segment product with the given ids, computes the numeric versus
    # analytic Jacobian with the v2 checker, and asserts the maximum error is below 2e-4.
    # Returns nothing.

    def _segment_prod(x):
      # [FUNCTION DOC]
      # HELPER: wraps segment product with the captured segment ids so the gradient checker can
      # call it with the data as its only argument. Input: data tensor. Returns the segment
      # product tensor.
      return math_ops.segment_prod(x, segment_ids)

    err = gradient_checker_v2.max_error(
        *gradient_checker_v2.compute_gradient(_segment_prod, [data]))
    self.assertLess(err, 2e-4)

  def testSegmentProdGradientWithoutOverlap(self):
    # [FUNCTION DOC]
    # Three rows each in their own segment (ids zero, one, two): checks the segment product
    # gradient is accurate using the shared checker.
    data = constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]],
                                dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 1, 2], dtype=dtypes.int64)
    self._run_gradient_check(data, segment_ids)

  def testSegmentProdGradientWithoutZeros(self):
    # [FUNCTION DOC]
    # Three rows with the first two sharing a segment (ids zero, zero, one) and no zero entries
    # in the data: checks the segment product gradient with the shared checker.
    data = constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]],
                                dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 1], dtype=dtypes.int64)
    self._run_gradient_check(data, segment_ids)

  def testSegmentProdGradientWithZeros(self):
    # [FUNCTION DOC]
    # Data containing several zero entries, including columns where more than one row in a
    # segment is zero (ids zero, zero, one): checks the segment product gradient handles zeros
    # without dividing by them, using the shared checker.
    data = constant_op.constant([[0, 2, 3, 4], [0, 0, 2, 0], [5, 0, 7, 0]],
                                dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 1], dtype=dtypes.int64)
    self._run_gradient_check(data, segment_ids)

  def testSegmentProdGradientWithEmptySegment(self):
    # [FUNCTION DOC]
    # Segment ids zero, zero, two leave segment one empty; checks the segment product gradient
    # is still accurate with the shared checker.
    data = constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]],
                                dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 2], dtype=dtypes.int64)
    self._run_gradient_check(data, segment_ids)


class FloorModGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests the gradient of the floor modulus op. Stateless; graph mode.

  @test_util.run_deprecated_v1
  def testFloorModGradient(self):
    # [FUNCTION DOC]
    # Computes floor mod of 131 by 17 (chosen away from the discontinuity where the quotient is
    # an integer) and asserts the gradient error with respect to the dividend is below 1e-4.
    # Legacy graph mode only.
    # Making sure the input is not near the discontinuity point where
    # x/y == floor(x/y)
    ns = constant_op.constant([17.], dtype=dtypes.float32)
    inputs = constant_op.constant([131.], dtype=dtypes.float32)
    floor_mod = math_ops.floormod(inputs, ns)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1],
                                                      floor_mod, [1])
      self.assertLess(error, 1e-4)


class DivNoNanGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of the safe division op that returns zero when the denominator is zero.
  # Stateless; graph mode.

  @test_util.run_deprecated_v1
  def testBasicGradient(self):
    # [FUNCTION DOC]
    # Divides the integers from minus three to two (as floats) by one plus their absolute value
    # and asserts the numeric gradient error is below 1e-4. Legacy graph mode only.
    inputs = constant_op.constant(np.arange(-3, 3),
                                  dtype=dtypes.float32)
    outputs = math_ops.div_no_nan(inputs, 1 + math_ops.abs(inputs))
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.get_shape().as_list(), outputs,
          outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithDenominatorIsZero(self):
    # [FUNCTION DOC]
    # Divides a vector by an all-zero denominator and computes graph gradients with respect to
    # both operands; asserts both gradients are all zeros (no infinities or not-a-numbers).
    # Legacy graph mode only.
    x = constant_op.constant(np.arange(-3, 3),
                             dtype=dtypes.float32)
    y = array_ops.zeros_like(x,
                             dtype=dtypes.float32)
    outputs = math_ops.div_no_nan(x, y)
    with self.cached_session():
      dx, dy = gradients.gradients(outputs, [x, y])
      self.assertAllClose(dx, np.zeros(x.shape.as_list()))
      self.assertAllClose(dy, np.zeros(y.shape.as_list()))


class MulNoNanGradientTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of the safe multiplication op that returns zero when the right operand is
  # zero, even for infinite or undefined left values. Stateless; graph mode.

  @test_util.run_deprecated_v1
  def testBasicGradient(self):
    # [FUNCTION DOC]
    # Multiplies the integers from minus three to two (as floats) by one plus their absolute
    # value and asserts the numeric gradient error is below 1e-4. Legacy graph mode only.
    inputs = constant_op.constant(np.arange(-3, 3), dtype=dtypes.float32)
    outputs = math_ops.mul_no_nan(inputs, 1 + math_ops.abs(inputs))
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.get_shape().as_list(), outputs,
          outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithRhsIsZero(self):
    # [FUNCTION DOC]
    # Multiplies a vector containing zero, one, not-a-number and both infinities by an all-zero
    # right operand and computes graph gradients; asserts the gradient with respect to the left
    # operand is all zeros and the gradient with respect to the right equals the left values.
    # Legacy graph mode only.
    x_vals = [0, 1.0, np.nan, np.inf, -np.inf]
    x = constant_op.constant(x_vals, dtype=dtypes.float32)
    y = array_ops.zeros_like(x, dtype=dtypes.float32)
    outputs = math_ops.mul_no_nan(x, y)
    with self.cached_session():
      dx, dy = gradients.gradients(outputs, [x, y])
      self.assertAllClose(dx, np.zeros(x.shape.as_list()))
      self.assertAllClose(dy, x_vals)


class XlogyTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of xlogy (x times log y, defined as zero when x is zero) at ordinary and
  # edge values for half, single and double precision floats. Stateless; graph mode.

  def _xlogy_gradients(self, x, y):
    # [FUNCTION DOC]
    # HELPER: computes the xlogy gradients for the tests.
    # Inputs: scalar tensors x and y. Builds the xlogy op twice, takes the graph gradient with
    # respect to x and to y, evaluates both in the current session, and returns the pair of
    # numeric values (x gradient, y gradient).
    xlogy_xgrad = self.evaluate(gradients.gradients(math_ops.xlogy(x, y), x)[0])
    xlogy_ygrad = self.evaluate(gradients.gradients(math_ops.xlogy(x, y), y)[0])
    return xlogy_xgrad, xlogy_ygrad

  @test_util.run_deprecated_v1
  def testNonZeroValuesGrad(self):
    # [FUNCTION DOC]
    # With x of 0.1 and y of 3.1 for each float dtype, asserts the gradient with respect to x
    # equals log of y and the gradient with respect to y equals x divided by y. Legacy graph
    # mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xlogy_xgrad, xlogy_ygrad = self._xlogy_gradients(x, y)
      xlogy_expected_xgrad = self.evaluate(math_ops.log(y))
      xlogy_expected_ygrad = self.evaluate(x / y)
      self.assertAllClose(xlogy_expected_xgrad, xlogy_xgrad)
      self.assertAllClose(xlogy_expected_ygrad, xlogy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXGrad(self):
    # [FUNCTION DOC]
    # With x equal to zero and y of 3.1, asserts both gradients are zero for each float dtype.
    # Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xlogy_xgrad, xlogy_ygrad = self._xlogy_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xlogy_xgrad)
      self.assertAllClose(zero, xlogy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroYGrad(self):
    # [FUNCTION DOC]
    # With x of 0.1 and y equal to zero, asserts the gradient with respect to x is negative
    # infinity and the gradient with respect to y is positive infinity, for each float dtype.
    # Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(0., dtype=dtype)
      xlogy_xgrad, xlogy_ygrad = self._xlogy_gradients(x, y)
      self.assertAllClose(-np.inf, xlogy_xgrad)
      self.assertAllClose(np.inf, xlogy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXYGrad(self):
    # [FUNCTION DOC]
    # With both x and y equal to zero, asserts both gradients are zero for each float dtype.
    # Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(0., dtype=dtype)
      xlogy_xgrad, xlogy_ygrad = self._xlogy_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xlogy_xgrad)
      self.assertAllClose(zero, xlogy_ygrad)


class Xlog1pyTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of xlog1py (x times log of one plus y, zero when x is zero) at ordinary and
  # edge values for half, single and double precision floats. Stateless; graph mode.

  def _xlog1py_gradients(self, x, y):
    # [FUNCTION DOC]
    # HELPER: computes the xlog1py gradients for the tests.
    # Inputs: scalar tensors x and y. Builds the op twice, takes graph gradients with respect
    # to x and y, evaluates both, and returns the pair (x gradient, y gradient).
    xlog1py_xgrad = self.evaluate(
        gradients.gradients(math_ops.xlog1py(x, y), x)[0])
    xlog1py_ygrad = self.evaluate(
        gradients.gradients(math_ops.xlog1py(x, y), y)[0])
    return xlog1py_xgrad, xlog1py_ygrad

  @test_util.run_deprecated_v1
  def testNonZeroValuesGrad(self):
    # [FUNCTION DOC]
    # With x of 0.1 and y of 3.1, asserts the x gradient equals log of one plus y and the y
    # gradient equals x divided by one plus y, for each float dtype. Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xlog1py_xgrad, xlog1py_ygrad = self._xlog1py_gradients(x, y)
      xlog1py_expected_xgrad = self.evaluate(math_ops.log1p(y))
      xlog1py_expected_ygrad = self.evaluate(x / (1. + y))
      self.assertAllClose(xlog1py_expected_xgrad, xlog1py_xgrad)
      self.assertAllClose(xlog1py_expected_ygrad, xlog1py_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXGrad(self):
    # [FUNCTION DOC]
    # With x equal to zero and y of 3.1, asserts both gradients are zero for each float dtype.
    # Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xlog1py_xgrad, xlog1py_ygrad = self._xlog1py_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xlog1py_xgrad)
      self.assertAllClose(zero, xlog1py_ygrad)

  @test_util.run_deprecated_v1
  def testNegOneYGrad(self):
    # [FUNCTION DOC]
    # With x of 0.1 and y equal to minus one (where one plus y is zero), asserts the x gradient
    # is negative infinity and the y gradient is positive infinity for each float dtype. Legacy
    # graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(-1., dtype=dtype)
      xlog1py_xgrad, xlog1py_ygrad = self._xlog1py_gradients(x, y)
      self.assertAllClose(-np.inf, xlog1py_xgrad)
      self.assertAllClose(np.inf, xlog1py_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXNegOneYGrad(self):
    # [FUNCTION DOC]
    # With x equal to zero and y equal to minus one, asserts both gradients are zero for each
    # float dtype. Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(-1., dtype=dtype)
      xlog1py_xgrad, xlog1py_ygrad = self._xlog1py_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xlog1py_xgrad)
      self.assertAllClose(zero, xlog1py_ygrad)


class XdivyTest(test.TestCase):
  # [CLASS DOC]
  # Tests gradients of xdivy (x divided by y, zero when x is zero) at ordinary and edge values
  # for half, single and double precision floats. Stateless; graph mode.

  def _xdivy_gradients(self, x, y):
    # [FUNCTION DOC]
    # HELPER: computes the xdivy gradients for the tests.
    # Inputs: scalar tensors x and y. Builds the op twice, takes graph gradients with respect
    # to x and y, evaluates both, and returns the pair (x gradient, y gradient).
    xdivy_xgrad = self.evaluate(gradients.gradients(math_ops.xdivy(x, y), x)[0])
    xdivy_ygrad = self.evaluate(gradients.gradients(math_ops.xdivy(x, y), y)[0])
    return xdivy_xgrad, xdivy_ygrad

  @test_util.run_deprecated_v1
  def testNonZeroValuesGrad(self):
    # [FUNCTION DOC]
    # With x of 0.1 and y of 3.1, asserts the x gradient equals the reciprocal of y and the y
    # gradient equals minus x divided by y squared, for each float dtype. Legacy graph mode
    # only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xdivy_xgrad, xdivy_ygrad = self._xdivy_gradients(x, y)
      xdivy_expected_xgrad = self.evaluate(1 / y)
      xdivy_expected_ygrad = self.evaluate(-x / y**2)
      self.assertAllClose(xdivy_expected_xgrad, xdivy_xgrad)
      self.assertAllClose(xdivy_expected_ygrad, xdivy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXGrad(self):
    # [FUNCTION DOC]
    # With x equal to zero and y of 3.1, asserts both gradients are zero for each float dtype.
    # Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xdivy_xgrad, xdivy_ygrad = self._xdivy_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xdivy_xgrad)
      self.assertAllClose(zero, xdivy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroYGrad(self):
    # [FUNCTION DOC]
    # With x of 0.1 and y equal to zero, asserts the x gradient is positive infinity and the y
    # gradient is negative infinity for each float dtype. Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(0., dtype=dtype)
      xdivy_xgrad, xdivy_ygrad = self._xdivy_gradients(x, y)
      self.assertAllClose(np.inf, xdivy_xgrad)
      self.assertAllClose(-np.inf, xdivy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXYGrad(self):
    # [FUNCTION DOC]
    # With both x and y equal to zero, asserts both gradients are zero for each float dtype.
    # Legacy graph mode only.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(0., dtype=dtype)
      xdivy_xgrad, xdivy_ygrad = self._xdivy_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xdivy_xgrad)
      self.assertAllClose(zero, xdivy_ygrad)


@test_util.run_all_in_graph_and_eager_modes
class PowGradTest(test.TestCase):
  # [CLASS DOC]
  # Tests the gradient of the power op when the base contains zero, via graph gradients and via
  # a gradient tape. Runs in graph and eager modes through the class decorator. Stateless.

  def test_zero_grad_tf_gradients(self):
    # [FUNCTION DOC]
    # Skipped when executing eagerly (graph gradients are unsupported there). Otherwise squares
    # the vector minus one, zero, one using graph gradients and asserts the gradient is minus
    # two, zero, two, i.e. no not-a-number appears at the zero base.
    if context.executing_eagerly():
      self.skipTest("tf.gradients not supported in eager.")

    x = constant_op.constant([-1., 0., 1.])
    g = self.evaluate(gradients.gradients(math_ops.pow(x, 2), x)[0])
    self.assertAllClose([-2., 0., 2.], g)

  def test_zero_grad_tape(self):
    # [FUNCTION DOC]
    # Same scenario using a gradient tape that watches the input: squares the vector minus one,
    # zero, one and asserts the gradient is minus two, zero, two. Works in both modes.
    x = constant_op.constant([-1, 0., 1.])
    with backprop.GradientTape() as tape:
      tape.watch(x)
      g = tape.gradient(math_ops.pow(x, 2), x)
    g = self.evaluate(g)
    self.assertAllClose([-2., 0., 2.], g)


@test_util.run_all_in_graph_and_eager_modes
class NextAfterTest(test.TestCase):
  # [CLASS DOC]
  # Tests the gradient of the nextafter op (next representable value after the first argument
  # toward the second), including dynamic shapes, broadcasting and the numeric gradient
  # checker. Runs in graph and eager modes via the class decorator. Stateless.

  def _nextafter_gradient(self, x1, x2):
    # [FUNCTION DOC]
    # HELPER: computes nextafter gradients for the tests.
    # Inputs: two tensors x1 and x2. Records nextafter of them under a gradient tape that
    # watches both, and returns the list of gradients with respect to x1 and x2.
    # KNOWN ISSUE: the gradient call is made inside the tape context block (via the return
    # statement) rather than after it; it works but is unusual and records extra tape activity.
    with backprop.GradientTape() as tape:
      tape.watch(x1)
      tape.watch(x2)
      y = math_ops.nextafter(x1, x2)
      return tape.gradient(y, [x1, x2])

  def testBasic(self):
    # [FUNCTION DOC]
    # For float32 and float64 scalars (0.1 and 3.1), asserts the gradient with respect to the
    # first argument is one and with respect to the second is zero.
    for dtype in [dtypes.float32, dtypes.float64]:
      x1 = constant_op.constant(0.1, dtype=dtype)
      x2 = constant_op.constant(3.1, dtype=dtype)
      dx1, dx2 = self._nextafter_gradient(x1, x2)
      expected_dx1 = constant_op.constant(1, dtype=dtype)
      expected_dx2 = constant_op.constant(0, dtype=dtype)
      self.assertAllClose(expected_dx1, dx1)
      self.assertAllClose(expected_dx2, dx2)

  def testDynamicShapes(self):
    # [FUNCTION DOC]
    # Same scenario as the basic test but the inputs are placeholders with defaults and unknown
    # shape, so the gradient must be built without static shape information. Asserts the
    # gradients are one and zero for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      default_x1 = constant_op.constant(0.1, dtype=dtype)
      default_x2 = constant_op.constant(3.1, dtype=dtype)
      x1 = array_ops.placeholder_with_default(default_x1, shape=None)
      x2 = array_ops.placeholder_with_default(default_x2, shape=None)
      dx1, dx2 = self._nextafter_gradient(x1, x2)
      expected_dx1 = constant_op.constant(1, dtype=dtype)
      expected_dx2 = constant_op.constant(0, dtype=dtype)
      self.assertAllClose(expected_dx1, dx1)
      self.assertAllClose(expected_dx2, dx2)

  def testWithGradientChecker(self):
    # [FUNCTION DOC]
    # Uses the v2 numeric gradient checker on nextafter of a five-element vector toward a
    # constant vector of twos, differentiating with respect to the first argument only. Asserts
    # maximum error below 1e-3 for float32 and float64, inside a cached session.
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        x1 = np.array([-1, 0, 1, 2, 3], dtype=dtype.as_numpy_dtype)
        x2 = np.array([2, 2, 2, 2, 2], dtype=dtype.as_numpy_dtype)
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: math_ops.nextafter(x, x2), [x1]))  # pylint: disable=cell-var-from-loop
        self.assertLess(err, 1e-3)

  def testBroadcastingWithGradientChecker(self):
    # [FUNCTION DOC]
    # Like the checker test but the second argument is a single-element array that broadcasts
    # against the five-element first argument. Asserts maximum error below 1e-3 for float32 and
    # float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        x1 = np.array([-1, 0, 1, 2, 3], dtype=dtype.as_numpy_dtype)
        x2 = np.array([2], dtype=dtype.as_numpy_dtype)
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: math_ops.nextafter(x, x2), [x1]))  # pylint: disable=cell-var-from-loop
        self.assertLess(err, 1e-3)


if __name__ == "__main__":
  test.main()
