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
# [MODULE DOC]
# Test module for the local response normalization (LRN) op and its gradient.
# Defines one test case class that compares the op against a pure numpy reference
# implementation on random shapes and parameters, checks gradients numerically,
# and checks that the gradient op rejects mismatched shapes.
# No import-time side effects other than importing the nn gradient registrations.
# Running the file as a script starts the standard TensorFlow test runner.
"""Tests for local response normalization."""

import copy

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class LRNOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the local response normalization op: forward values versus a numpy
  # reference, numerical gradient accuracy, gradient at zero input, and shape validation of
  # the gradient op. Has no shared fixtures; each test builds its own graph and session.

  def _LRN(self, input_image, lrn_depth_radius=5, bias=1.0, alpha=1.0,
           beta=0.5):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation used by the forward-value test.
    # Computes the expected normalized output for a four dimensional (batch, rows, cols,
    # depth) array. Each element is divided by the power beta of (bias plus alpha times the
    # sum of squares of the values within lrn_depth_radius channels on either side, clipped
    # to the valid depth range). Inputs: input_image (numpy array), lrn_depth_radius,
    # bias, alpha, beta (numbers with defaults 5, 1.0, 1.0, 0.5). Returns a new array of
    # the same shape; the input is deep copied and not modified. Raises nothing itself.
    """Compute expected result."""
    output = copy.deepcopy(input_image)
    batch_size = input_image.shape[0]
    rows = input_image.shape[1]
    cols = input_image.shape[2]
    depth = input_image.shape[3]
    for b in range(batch_size):
      for r in range(rows):
        for c in range(cols):
          for d in range(depth):
            begin = max(0, d - lrn_depth_radius)
            end = min(depth, d + lrn_depth_radius + 1)
            patch = input_image[b, r, c, begin:end]
            output[b, r, c, d] /= (
                np.power(bias + alpha * np.sum(patch * patch), beta))
    return output

  def _RunAndVerify(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body of the forward-value test, run for each data type.
    # Picks a random 4-D shape (depth at least 2), random depth radius within the range
    # supported by cuDNN, random bias, alpha and beta, builds a placeholder and the LRN op,
    # feeds random float data, and evaluates it in a cached session. Compares against the
    # numpy reference, prints the maximum absolute error, and asserts it is below 1e-4 for
    # float32 or 1e-2 otherwise, and that the static shape matches. Input: dtype of the
    # placeholder. Returns nothing; fails via assertions. Uses numpy global randomness.
    with self.cached_session():
      # random shape
      shape = np.random.randint(1, 16, size=4)
      # Make depth at least 2 to make it meaningful
      shape[3] += 1
      p = array_ops.placeholder(dtype, shape=shape)
      # random depth_radius, bias, alpha, beta. cuDNN requires depth_radius to
      # be in [1, 7].
      lrn_depth_radius = np.random.randint(1, min(8, shape[3]))

      bias = 1.0 + np.random.rand()
      alpha = 2.0 * np.random.rand()
      # cuDNN requires beta >= 0.01.
      beta = 0.01 + 2.0 * np.random.rand()
      lrn_t = nn.local_response_normalization(
          p,
          name="lrn",
          depth_radius=lrn_depth_radius,
          bias=bias,
          alpha=alpha,
          beta=beta)
      params = {p: np.random.rand(*shape).astype("f")}
      result = lrn_t.eval(feed_dict=params)
    expected = self._LRN(
        params[p],
        lrn_depth_radius=lrn_depth_radius,
        bias=bias,
        alpha=alpha,
        beta=beta)
    err = np.amax(np.abs(result - expected))
    print("LRN error for bias ", bias, "alpha ", alpha, " beta ", beta, " is ",
          err)
    if dtype == dtypes.float32:
      self.assertTrue(err < 1e-4)
    else:
      self.assertTrue(err < 1e-2)
    self.assertShapeEqual(expected, lrn_t)

  @test_util.run_deprecated_v1
  def testCompute(self):
    # [FUNCTION DOC]
    # Runs the forward-value check twice with float32, and also with float16 when no GPU is
    # available (half precision LRN is not supported on GPU). Expects computed results to
    # match the numpy reference within tolerance. Graph mode only (deprecated v1 style).
    for _ in range(2):
      self._RunAndVerify(dtypes.float32)
      # Enable when LRN supports tf.float16 on GPU.
      if not test.is_gpu_available():
        self._RunAndVerify(dtypes.float16)

  @test_util.run_deprecated_v1
  def testGradientsZeroInput(self):
    # [FUNCTION DOC]
    # Scenario: LRN with depth radius 2, bias 1, alpha 0, beta 1 on an all-zero 4x4x4x4
    # float32 input. Computes the gradient of the output with respect to the input and
    # expects it to be all ones with the same shape as the input (identity behavior since
    # alpha is zero and bias is one). Graph mode only.
    with self.session():
      shape = [4, 4, 4, 4]
      p = array_ops.placeholder(dtypes.float32, shape=shape)
      inp_array = np.zeros(shape).astype("f")
      lrn_op = nn.local_response_normalization(p, 2, 1.0, 0.0, 1.0, name="lrn")
      grad = gradients_impl.gradients([lrn_op], [p])[0]
      params = {p: inp_array}
      r = grad.eval(feed_dict=params)
    expected = np.ones(shape).astype("f")
    self.assertAllClose(r, expected)
    self.assertShapeEqual(expected, grad)

  @test_util.run_in_graph_and_eager_modes
  def testIncompatibleInputAndOutputImageShapes(self):
    # [FUNCTION DOC]
    # Scenario: calls the LRN gradient op with 4-D input gradients and input image but an
    # output image of rank 6. Expects either a value error (graph construction) or an
    # invalid-argument error (execution) to be raised. Runs in both graph and eager modes.
    depth_radius = 1
    bias = 1.59018219
    alpha = 0.117728651
    beta = 0.404427052
    input_grads = random_ops.random_uniform(
        shape=[4, 4, 4, 4],
        minval=-10000,
        maxval=10000,
        dtype=dtypes.float32,
        seed=-2033)
    input_image = random_ops.random_uniform(
        shape=[4, 4, 4, 4],
        minval=-10000,
        maxval=10000,
        dtype=dtypes.float32,
        seed=-2033)
    invalid_output_image = random_ops.random_uniform(
        shape=[4, 4, 4, 4, 4, 4],
        minval=-10000,
        maxval=10000,
        dtype=dtypes.float32,
        seed=-2033)
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      self.evaluate(
          nn.lrn_grad(
              input_grads=input_grads,
              input_image=input_image,
              output_image=invalid_output_image,
              depth_radius=depth_radius,
              bias=bias,
              alpha=alpha,
              beta=beta))

  def _RunAndVerifyGradients(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body of the numerical gradient test, run for each data type.
    # Chooses a small random 4-D shape (depth at least 2) and random radius, bias, alpha
    # and beta, builds a constant random input of the given dtype (float32 or float16), applies
    # LRN, and uses the gradient checker to compute the error between analytic and numeric
    # Jacobians. Prints the error and asserts it is below 1e-4 for float32, otherwise below
    # 1.0. Returns nothing; uses numpy global randomness.
    with self.cached_session():
      # random shape
      shape = np.random.randint(1, 5, size=4)
      # Make depth at least 2 to make it meaningful
      shape[3] += 1
      # random depth_radius, bias, alpha, beta. cuDNN requires depth_radius to
      # be in [1, 7].
      lrn_depth_radius = np.random.randint(1, min(8, shape[3]))
      bias = 1.0 + np.random.rand()
      alpha = 1.0 * np.random.rand()
      # cuDNN requires beta >= 0.01.
      beta = 0.01 + 1.0 * np.random.rand()
      if dtype == dtypes.float32:
        inp_array = np.random.rand(*shape).astype(np.float32)
      else:
        inp_array = np.random.rand(*shape).astype(np.float16)

      inp = constant_op.constant(
          list(inp_array.ravel(order="C")), shape=shape, dtype=dtype)
      lrn_op = nn.local_response_normalization(
          inp,
          name="lrn",
          depth_radius=lrn_depth_radius,
          bias=bias,
          alpha=alpha,
          beta=beta)
      err = gradient_checker.compute_gradient_error(inp, shape, lrn_op, shape)
    print("LRN Gradient error for bias ", bias, "alpha ", alpha, " beta ", beta,
          " is ", err)
    if dtype == dtypes.float32:
      self.assertLess(err, 1e-4)
    else:
      self.assertLess(err, 1.0)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Runs the numerical gradient check twice for float32, and for float16 as well when no
    # GPU is available. Expects gradient error to stay within the dtype-specific tolerance.
    for _ in range(2):
      self._RunAndVerifyGradients(dtypes.float32)
      # Enable when LRN supports tf.float16 on GPU.
      if not test.is_gpu_available():
        self._RunAndVerifyGradients(dtypes.float16)


if __name__ == "__main__":
  test.main()
