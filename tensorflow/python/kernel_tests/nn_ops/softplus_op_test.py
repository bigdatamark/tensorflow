# [MODULE DOC]
# Unit tests for the softplus activation (nn_ops.softplus) and its first, second and third order
# gradients. Covers numeric agreement with a NumPy log-add-exp reference across float dtypes
# including values near the log of machine epsilon, numeric gradient checks, and rejection of
# integer inputs. Importing nn_grad registers the gradient functions. Running as a script starts
# the TensorFlow test runner.
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
"""Tests for Softplus and SoftplusGrad."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SoftplusTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the softplus op: forward values against a NumPy reference (including
  # extreme-magnitude inputs near the log of machine epsilon), numeric accuracy of first, second
  # and third derivatives, and dtype validation. Stateless; each test builds its own session.

  def _npSoftplus(self, np_features):
    # [FUNCTION DOC]
    # HELPER: NumPy reference implementation used by the softplus value tests.
    # Takes an array-like of features, converts it to an array, and returns the log of the sum of
    # exponentials of zero (in the input's dtype) and the input, which is the softplus value.
    # No side effects.
    np_features = np.asarray(np_features)
    zero = np.asarray(0).astype(np_features.dtype)
    return np.logaddexp(zero, np_features)

  def _testSoftplus(self, np_features, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: shared checker for the forward-value test.
    # Inputs: an array-like of features and a flag asking for a GPU-enabled session (default
    # false). Computes the reference, evaluates softplus in a cached session, and asserts values
    # are close with dtype-dependent tolerances (looser for half and bfloat16), that all results
    # are strictly positive, and that the op's static shape matches the reference. Returns nothing;
    # raises on assertion failure.
    np_softplus = self._npSoftplus(np_features)
    with self.cached_session(use_gpu=use_gpu):
      softplus = nn_ops.softplus(np_features)
      tf_softplus = self.evaluate(softplus)
    self.assertAllCloseAccordingToType(
        np_softplus, tf_softplus, half_rtol=5e-3, half_atol=5e-3,
        bfloat16_rtol=5e-2, bfloat16_atol=5e-2
    )
    self.assertTrue(np.all(tf_softplus > 0))
    self.assertShapeEqual(np_softplus, softplus)

  def testNumbers(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the GPU-requested call with boundary values, the fifth list element adds ten
    # and then subtracts log_eps (instead of just adding ten) and the list has nine entries rather
    # than the ten used in the CPU call (the entry for minus log_eps is missing), so the two calls
    # do not test the same inputs; this looks like a typo.
    # Scenario: for float16, float32, float64 and bfloat16, checks a fixed 2x5 integer array on
    # CPU and GPU requests, then checks lists of values around plus and minus the log of machine
    # epsilon (for bfloat16 epsilon is computed from an assumed 8-bit precision since no finfo
    # exists), offset by one and ten, to exercise the numerically stable branches of softplus.
    for t in [
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ]:
      self._testSoftplus(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      self._testSoftplus(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=True)
      if t == dtypes.bfloat16.as_numpy_dtype:
        # bfloat16 dtype doesn't have finfo.
        # Calculate epsilon using machine_epsilon = base ^ (-(precision - 1))
        log_eps = np.log(2 ** (-(8 - 1)))
      else:
        log_eps = np.log(np.finfo(t).eps)
      one = t(1)
      ten = t(10)
      self._testSoftplus(
          [
              log_eps, log_eps - one, log_eps + one, log_eps - ten,
              log_eps + ten, -log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=False)
      self._testSoftplus(
          [
              log_eps, log_eps - one, log_eps + one, log_eps - ten,
              log_eps + ten - log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=True)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: softplus on a 2x5 float constant in the range -0.9..0.9; computes the maximum error
    # between analytic and numeric Jacobians, prints it, and asserts it is below 1e-4.
    # Graph-mode only (run as deprecated v1).
    with self.cached_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.softplus(x, name="softplus")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("softplus (float) gradient err = ", err)
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradGrad(self):
    # [FUNCTION DOC]
    # Scenario: builds the first derivative of softplus with respect to the input, then checks the
    # numeric versus analytic Jacobian of that derivative (second-order gradient) on a 2x5 input.
    # Prints the error and asserts it is below 5e-5. Graph-mode only (deprecated v1).
    with self.cached_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.softplus(x, name="softplus")
      (grad,) = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], grad, [2, 5], x_init_value=x_init)
    print("softplus (float) gradient of gradient err = ", err)
    self.assertLess(err, 5e-5)

  @test_util.run_deprecated_v1
  def testGradGradGrad(self):
    # [FUNCTION DOC]
    # Scenario: builds first and second derivatives of softplus, then checks numeric versus
    # analytic Jacobian of the second derivative (third-order gradient) on a 2x5 input. Prints the
    # error and asserts it is below 5e-5. Graph-mode only (deprecated v1).
    with self.cached_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.softplus(x, name="softplus")
      (grad,) = gradients_impl.gradients(y, x)
      (grad_grad,) = gradients_impl.gradients(grad, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], grad_grad, [2, 5], x_init_value=x_init)
    print("softplus (float) third-order gradient err = ", err)
    self.assertLess(err, 5e-5)

  @test_util.run_deprecated_v1
  def testNoInts(self):
    # [FUNCTION DOC]
    # Scenario: softplus applied to an int32 scalar constant. Expects a type error saying the
    # features argument has data type int32 not in the list of allowed values. Graph-mode only
    # (deprecated v1).
    with self.cached_session():
      with self.assertRaisesRegex(
          TypeError,
          "'features' has DataType int32 not in list of allowed values"):
        nn_ops.softplus(constant_op.constant(42)).eval()


if __name__ == "__main__":
  test.main()
