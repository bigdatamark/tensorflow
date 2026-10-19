# [MODULE DOC]
# Unit tests for the fused cumulative log-sum-exp op. Each test compares the fused op against a
# naive reference (cumulative sum of exponentials, or a mapped reduce-logsumexp over prefixes),
# across dtypes, axes, reverse and exclusive modes, CPU and GPU requests, and gradients. No
# import-time side effects; run as a script it starts the TensorFlow test runner.
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
"""Functional tests for cumulative_logsumexp op."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CumulativeLogsumexpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the cumulative log-sum-exp op. Holds a class attribute listing the floating
  # point dtypes exercised (float32, float64, float16, bfloat16) and a set of helper methods that
  # compute and compare naive and fused results.
  valid_dtypes = [
      dtypes.float32,
      dtypes.float64,
      dtypes.float16,
      dtypes.bfloat16,
  ]

  def _computeLogSumExp(self, x, **kwargs):
    # [FUNCTION DOC]
    # HELPER: builds the naive and fused results compared by the cumulative log-sum-exp tests.
    # Takes a tensor x and keyword arguments (axis, reverse, exclusive) forwarded to both ops.
    # Returns a pair: the cumulative sum of the exponential of x, and the exponential of the
    # fused cumulative log-sum-exp of x. Both are unevaluated tensors; no side effects.
    result_naive = math_ops.cumsum(math_ops.exp(x), **kwargs)
    result_fused = math_ops.exp(math_ops.cumulative_logsumexp(x, **kwargs))
    return result_naive, result_fused

  def _testLogSumExp(self, x, dtype=dtypes.float32, use_gpu=False, **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for the forward-value tests.
    # Inputs: x (array-like values), dtype (tensor dtype, default float32), use_gpu (whether the
    # session may use a GPU, default false), and keyword arguments forwarded to the ops (axis,
    # reverse, exclusive). Converts x to a tensor of the dtype, evaluates the naive and fused
    # results in a cached session, and asserts they are close; tolerance is 2e-2 for float16 and
    # bfloat16, otherwise 1e-6. Returns nothing; fails by assertion.
    with self.cached_session(use_gpu=use_gpu):
      x = ops.convert_to_tensor(x, dtype=dtype)

      result_naive, result_fused = self.evaluate(
          self._computeLogSumExp(x, **kwargs))

    tol = 2e-2 if dtype in [dtypes.float16, dtypes.bfloat16] else 1e-6
    self.assertAllClose(result_naive, result_fused, rtol=tol, atol=tol)

  def _testLogSumExpAllArgs(self, x, axis=0, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: runs the forward-value check over every supported dtype and every combination of
    # reverse and exclusive flags. Inputs: x (array-like values), axis (default 0), use_gpu
    # (default false). Returns nothing; fails by assertion from the per-case helper.
    for dtype in self.valid_dtypes:
      for reverse in (True, False):
        for exclusive in (True, False):
          self._testLogSumExp(
              x, dtype=dtype, use_gpu=use_gpu,
              reverse=reverse, exclusive=exclusive,
              axis=axis)

  def testMinusInfinity(self):
    # [FUNCTION DOC]
    # Test that the fused op matches the naive result when the input contains negative infinity
    # (the log of zero) entries mixed with log of one. Runs all dtype/reverse/exclusive
    # combinations on both a CPU-only and a GPU-allowed session. Returns nothing.
    x = np.log([0., 0., 1., 1., 1., 1., 0., 0.])
    self._testLogSumExpAllArgs(x, use_gpu=False)
    self._testLogSumExpAllArgs(x, use_gpu=True)

  def test1D(self):
    # [FUNCTION DOC]
    # Test the fused op on a one-dimensional input of ten evenly spaced values between -0.5 and
    # 0.4, over all dtypes and reverse/exclusive flags, with and without GPU. Returns nothing.
    x = np.arange(10) / 10.0 - 0.5
    self._testLogSumExpAllArgs(x, use_gpu=False)
    self._testLogSumExpAllArgs(x, use_gpu=True)

  def test2D(self):
    # [FUNCTION DOC]
    # Test the fused op on a 2 by 10 input along each axis value of negative two, negative one,
    # zero and one, over all dtypes and reverse/exclusive flags, with and without GPU.
    # Returns nothing.
    x = np.reshape(np.arange(20) / 20.0 - 0.5, (2, 10))

    for axis in (-2, -1, 0, 1):
      self._testLogSumExpAllArgs(x, axis=axis, use_gpu=False)
      self._testLogSumExpAllArgs(x, axis=axis, use_gpu=True)

  def _testGradient(self, x, use_gpu=False, **kwargs):
    # [FUNCTION DOC]
    # HELPER: gradient comparison used by the gradient test. Inputs: x (array-like values,
    # converted to float64), use_gpu (default false), keyword arguments forwarded to the ops
    # (reverse, exclusive). Computes the theoretical gradient of the naive formulation and of the
    # fused op with the version-2 gradient checker, and asserts they are close. Returns nothing.
    # Note the nested lambdas are small closures over the keyword arguments and are not documented.
    with self.cached_session(use_gpu=use_gpu):
      x = ops.convert_to_tensor(x, dtype=dtypes.float64)

      grad_naive_theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda y: math_ops.cumsum(math_ops.exp(y), **kwargs), [x])
      grad_fused_theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda y: math_ops.exp(math_ops.cumulative_logsumexp(y, **kwargs)),
          [x])

      self.assertAllClose(grad_fused_theoretical, grad_naive_theoretical)

  def testGradient(self):
    # [FUNCTION DOC]
    # Test that the gradient of the fused op equals that of the naive formulation, for all
    # combinations of reverse and exclusive on a ten-element input, with CPU-only and GPU-allowed
    # sessions. Returns nothing.
    for reverse in (True, False):
      for exclusive in (True, False):
        x = np.arange(10) / 10.0 - 0.5

        self._testGradient(x, use_gpu=False,
                           reverse=reverse, exclusive=exclusive)
        self._testGradient(x, use_gpu=True,
                           reverse=reverse, exclusive=exclusive)

  def _logSumExpMap(self, x):
    # [FUNCTION DOC]
    # HELPER: overflow-safe reference implementation for the large-value test. Takes a
    # one-dimensional tensor x and maps over indices, taking the reduce-logsumexp of each prefix
    # of x. Returns a tensor of the same dtype as x holding the prefix results (unevaluated).
    return map_fn.map_fn(
        lambda i: math_ops.reduce_logsumexp(x[:i + 1]),
        math_ops.range(array_ops.shape(x)[0]),
        dtype=x.dtype)

  def test1DLarge(self):
    # [FUNCTION DOC]
    # Test that the fused op stays correct for large inputs (multiples of twenty up to 380) where
    # a naive exponential would overflow. Compares the fused result against the prefix
    # reduce-logsumexp reference in float32, on CPU-only and GPU-allowed sessions. Returns nothing.
    # This test ensures that the operation is correct even when the naive
    # implementation would overflow.
    x_np = np.arange(20) * 20.0

    for use_gpu in (True, False):
      with self.cached_session(use_gpu=use_gpu):
        x_tf = ops.convert_to_tensor(x_np, dtype=dtypes.float32)

        result_fused = self.evaluate(math_ops.cumulative_logsumexp(x_tf))
        result_map = self.evaluate(self._logSumExpMap(x_tf))

      self.assertAllClose(result_fused, result_map)


if __name__ == '__main__':
  test.main()
