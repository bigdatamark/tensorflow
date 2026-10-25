# [MODULE DOC]
# Unit tests for miscellaneous neural-network ops in the nn module: zero fraction, softmax and
# log softmax, log Poisson loss, L2 loss and L2 normalization, the family of dropout variants,
# sampled logits / sampled softmax / NCE losses, the many batch/layer normalization and moment
# helpers, convolution-related helpers, relu variants and other activation helpers. Tests compare
# against NumPy reference computations or numeric gradients and run in graph and/or eager mode.
# Not a helper module; no import-time side effects other than defining a table of dropout
# functions used for parameterized tests and importing gradient registrations.
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
"""Tests for miscellaneous functionality in tensorflow.ops.nn."""

import functools
import math
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test as test_lib


class ZeroFractionTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the zero-fraction function (fraction of elements equal to zero): small random
  # tensor against NumPy, empty input, very large int8 tensors (more than 2**27 elements) and a
  # placeholder of unknown size.

  def _ZeroFraction(self, x):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for the zero fraction. Takes a NumPy array with a non-empty shape
    # (asserted) and returns one minus the ratio of nonzero elements to total elements.
    assert x.shape
    total_elements = np.prod(x.shape)
    nonzeros = np.count_nonzero(x.flatten())
    return 1.0 - nonzeros / total_elements

  @test_util.run_deprecated_v1
  def testZeroFraction(self):
    # [FUNCTION DOC]
    # Builds a random 5x17 zero/one float tensor, computes the zero fraction with the library and
    # with the NumPy reference and asserts they agree within 1e-8.
    x_shape = [5, 17]
    x_np = np.random.randint(0, 2, size=x_shape).astype(np.float32)
    y_np = self._ZeroFraction(x_np)

    x_tf = constant_op.constant(x_np)
    x_tf.set_shape(x_shape)
    y_tf = nn_impl.zero_fraction(x_tf)
    y_tf_np = self.evaluate(y_tf)

    eps = 1e-8
    self.assertAllClose(y_tf_np, y_np, eps)

  @test_util.run_deprecated_v1
  def testZeroFractionEmpty(self):
    # [FUNCTION DOC]
    # Computes the zero fraction of an empty array and asserts the result is not-a-number.
    x = np.zeros(0)
    y = self.evaluate(nn_impl.zero_fraction(x))
    self.assertTrue(np.isnan(y))

  @test_util.run_deprecated_v1
  def testZeroFraction2_27Zeros(self):
    # [FUNCTION DOC]
    # Regression test for large inputs: an int8 tensor of all zeros with about 1% more than 2**27
    # elements must give a zero fraction of exactly 1.0 (checks no int32 overflow in counting).
    sparsity = nn_impl.zero_fraction(
        array_ops.zeros([int(2**27 * 1.01)], dtype=dtypes.int8))
    self.assertAllClose(1.0, self.evaluate(sparsity))

  @test_util.run_deprecated_v1
  def testZeroFraction2_27Ones(self):
    # [FUNCTION DOC]
    # Same large-input regression test with an int8 tensor of all ones; expects a zero fraction
    # of 0.0.
    sparsity = nn_impl.zero_fraction(
        array_ops.ones([int(2**27 * 1.01)], dtype=dtypes.int8))
    self.assertAllClose(0.0, self.evaluate(sparsity))

  @test_util.run_deprecated_v1
  def testUnknownSize(self):
    # [FUNCTION DOC]
    # Uses a float placeholder with unknown shape, runs the zero fraction in a session feeding a
    # 2x2 value with one zero, and asserts the result is 0.25.
    value = array_ops.placeholder(dtype=dtypes.float32)
    sparsity = nn_impl.zero_fraction(value)
    with self.cached_session() as sess:
      self.assertAllClose(0.25,
                          sess.run(sparsity, {value: [[0., 1.], [0.3, 2.]]}))


@test_util.run_all_in_graph_and_eager_modes
class SoftmaxTest(test_lib.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for softmax (version 2 API and the plain version): values against a NumPy reference,
  # negative/positive axes, bfloat16 input, and gradients. Runs in both graph and eager modes.

  def _softmax(self, x):
    # [FUNCTION DOC]
    # HELPER: NumPy reference softmax over the last axis of a 2-D array (asserted); subtracts the
    # row maximum for stability and returns the array unchanged when there are zero columns.
    assert len(x.shape) == 2
    if x.shape[1] == 0:
      return x
    m = x.max(1)[:, np.newaxis]
    u = np.exp(x - m)
    z = u.sum(1)[:, np.newaxis]
    return u / z

  def testSoftmax(self):
    # [FUNCTION DOC]
    # Compares softmax_v2 on a random 5x10 float32 input, with default axis and with axis one,
    # against the NumPy reference within 1e-3.
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._softmax(x_np)
    x_tf = constant_op.constant(x_np)
    y_tf = nn_ops.softmax_v2(x_tf)
    y_tf_last_dim = nn_ops.softmax_v2(x_tf, 1)
    y_tf_np = self.evaluate(y_tf)
    y_tf_last_dim_np = self.evaluate(y_tf_last_dim)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)
    self.assertAllClose(y_tf_last_dim_np, y_np, eps)

  def testSoftmaxAxes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variable named as a greater-than-rank axis case actually uses axis zero,
    # the same as the positive-axis case, so the second assertion compares two identical calls.
    # Checks that softmax_v2 on a 3x4 array gives the same result for axis minus two and axis zero.
    arr = np.linspace(0., 1, 12).reshape(3, 4)
    x_neg_axis = nn_ops.softmax_v2(arr, axis=-2)
    y_pos_axis = nn_ops.softmax_v2(arr, axis=0)
    z_gt_axis = nn_ops.softmax_v2(arr, axis=0)
    x_neg_axis_tf = self.evaluate(x_neg_axis)
    y_pos_axis_tf = self.evaluate(y_pos_axis)
    z_gt_axis_tf = self.evaluate(z_gt_axis)
    eps = 1e-3
    self.assertAllClose(x_neg_axis_tf, y_pos_axis_tf, eps)
    self.assertAllClose(y_pos_axis_tf, z_gt_axis_tf, eps)

  def testSoftmaxExtendType(self):
    # [FUNCTION DOC]
    # Runs softmax on a random float32 tensor and on its bfloat16 cast and asserts the bfloat16
    # result is close to the bfloat16 cast of the float32 result, with tolerance 1e-3 times the
    # number of columns.
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)

    x_f32_tf = constant_op.constant(x_np)
    x_bf16_tf = math_ops.cast(x_f32_tf, dtypes.bfloat16)
    y_f32_tf = self.evaluate(nn_ops.softmax(x_f32_tf))
    y_bf16_tf = self.evaluate(nn_ops.softmax(x_bf16_tf))
    expected = math_ops.cast(y_f32_tf, dtypes.bfloat16)
    tol = x_shape[1] * 1e-3
    self.assertAllClose(y_bf16_tf, expected, rtol=tol, atol=tol)

  @parameterized.parameters(((5, 10),), ((2, 3, 4),))
  def testGradient(self, x_shape):
    # [FUNCTION DOC]
    # Parameterized over shapes 5x10 and 2x3x4: compares theoretical and numerical Jacobians of
    # softmax_v2 on random float64 input and asserts they are close.
    x_np = np.random.randn(*x_shape).astype(np.float64)
    x_tf = constant_op.constant(x_np)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        nn_ops.softmax_v2, [x_tf])
    self.assertAllClose(theoretical, numerical)


@test_util.run_all_in_graph_and_eager_modes
class LogPoissonLossTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the log Poisson loss with and without the Stirling approximation term, comparing
  # with a NumPy reference and checking the gradient. Runs in graph and eager modes.

  def _log_poisson_loss(self, x, z, compute_full_loss=False):
    # [FUNCTION DOC]
    # HELPER: NumPy reference of the log Poisson loss. Inputs: x (log rate), z (targets), and
    # compute_full_loss (default false). Returns exp(x) minus z times x elementwise; when the
    # full loss is requested, adds the Stirling approximation term only where targets exceed one.
    lpl = np.exp(x) - z * x
    if compute_full_loss:
      stirling_approx = z * np.log(z) - z + 0.5 * np.log(2. * np.pi * z)
      lpl += np.ma.masked_array(stirling_approx, mask=(z <= 1)).filled(0.)
    return lpl

  def testLogPoissonLoss(self):
    # [FUNCTION DOC]
    # On random 5x10 float32 log-rates and integer targets zero to four, compares the library loss
    # with and without the full (Stirling) term against the NumPy reference within 1e-3.
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    z_np = np.random.randint(0, 5, size=x_shape).astype(np.float32)
    y_np = self._log_poisson_loss(x_np, z_np, compute_full_loss=False)
    y_np_stirling = self._log_poisson_loss(x_np, z_np, compute_full_loss=True)
    y_tf = nn_impl.log_poisson_loss(z_np, x_np, compute_full_loss=False)
    y_tf_stirling = nn_impl.log_poisson_loss(z_np, x_np, compute_full_loss=True)
    y_tf_np = self.evaluate(y_tf)
    y_tf_np_stirling = self.evaluate(y_tf_stirling)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)
    self.assertAllClose(y_tf_np_stirling, y_np_stirling, eps)

  def testGradient(self):
    # [FUNCTION DOC]
    # Checks theoretical against numerical gradient of the log Poisson loss with respect to the
    # log-rate on random float64 input (full loss not tested, per the TODO) and asserts closeness.
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float64)
    z_np = np.random.randint(0, 5, size=x_shape).astype(np.float64)
    with self.cached_session():
      x_tf = constant_op.constant(x_np)
      # TODO(b/241834841): Test with `compute_full_loss` set as True
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_impl.log_poisson_loss, [z_np, x_tf])
      self.assertAllClose(theoretical, numerical)


@test_util.run_all_in_graph_and_eager_modes
class LogSoftmaxTest(test_lib.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for log softmax (version 2 API): values against a NumPy reference, axis handling and
  # gradients. Runs in graph and eager modes.

  def _log_softmax(self, x):
    # [FUNCTION DOC]
    # HELPER: NumPy reference log softmax over the last axis of a 2-D array (asserted): subtract
    # row max, then subtract the log of the sum of exponentials. Returns the resulting array.
    assert len(x.shape) == 2
    m = x.max(1)[:, np.newaxis]
    u = x - m
    return u - np.log(np.sum(np.exp(u), 1, keepdims=True))

  def testLogSoftmax(self):
    # [FUNCTION DOC]
    # Compares log_softmax_v2 on a random 5x10 float32 input with the NumPy reference within 1e-3.
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._log_softmax(x_np)
    x_tf = constant_op.constant(x_np)
    y_tf = nn_ops.log_softmax_v2(x_tf)
    y_tf_np = self.evaluate(y_tf)
    eps = 1e-3
    self.assertAllClose(y_tf_np, y_np, eps)

  def testLogSoftmaxAxes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variable named as a greater-than-rank axis case actually uses axis zero,
    # identical to the positive-axis case, so the second assertion compares two identical calls.
    # Checks log_softmax_v2 on a 3x4 array gives the same result for axis minus two and axis zero.
    arr = np.linspace(0., 1, 12).reshape(3, 4)
    x_neg_axis = nn_ops.log_softmax_v2(arr, axis=-2)
    y_pos_axis = nn_ops.log_softmax_v2(arr, axis=0)
    z_gt_axis = nn_ops.log_softmax_v2(arr, axis=0)
    x_neg_axis_tf = self.evaluate(x_neg_axis)
    y_pos_axis_tf = self.evaluate(y_pos_axis)
    z_gt_axis_tf = self.evaluate(z_gt_axis)
    eps = 1e-3
    self.assertAllClose(x_neg_axis_tf, y_pos_axis_tf, eps)
    self.assertAllClose(y_pos_axis_tf, z_gt_axis_tf, eps)

  @parameterized.parameters(((5, 10),), ((2, 3, 4),))
  def testGradient(self, x_shape):
    # [FUNCTION DOC]
    # Parameterized over shapes 5x10 and 2x3x4: compares theoretical and numerical Jacobians of
    # log_softmax_v2 on random float64 input and asserts closeness.
    x_np = np.random.randn(*x_shape).astype(np.float64)
    with self.cached_session():
      x_tf = constant_op.constant(x_np)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_ops.log_softmax_v2, [x_tf])
      self.assertAllClose(theoretical, numerical)


@test_util.run_all_in_graph_and_eager_modes
class L2LossTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the L2 loss op (half the sum of squares): known value and gradient check. Runs in
  # graph and eager modes.

  def testL2Loss(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: operator precedence makes the loop iterate over nothing on machines without a
    # CUDA GPU (the conditional applies to the whole list sum, yielding an empty list), and on
    # CUDA machines it iterates over the three types; so on CPU-only runs nothing is asserted.
    # Intended behavior: for float32, float64 and (with CUDA) bfloat16 inputs, a 2x2 tensor with
    # values one, zero, three, two must give an L2 loss of 7.0.
    for dtype in [dtypes.float32, dtypes.float64] + \
                 [dtypes.bfloat16] if test_util.is_gpu_available(
                                          cuda_only=True) else []:
      x = constant_op.constant([1.0, 0.0, 3.0, 2.0],
                               shape=[2, 2],
                               name="x",
                               dtype=dtype)
      l2loss = nn_ops.l2_loss(x)
      value = self.evaluate(l2loss)
      self.assertAllClose(7.0, value)

  def testGradient(self):
    # [FUNCTION DOC]
    # With a fixed random seed, builds a random 20x7x3 float64 tensor and asserts the theoretical
    # and numerical gradients of the L2 loss are close.
    x_shape = [20, 7, 3]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    with self.cached_session():
      x = constant_op.constant(x_val, name="x")
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_ops.l2_loss, [x])
      self.assertAllClose(theoretical, numerical)


@test_util.run_all_in_graph_and_eager_modes
class L2NormalizeTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for L2 normalization along one or several dimensions, for real and complex inputs, and
  # its gradient. Uses a NumPy reference helper. Runs in graph and eager modes.

  def _l2Normalize(self, x, dim):
    # [FUNCTION DOC]
    # HELPER: NumPy reference L2 normalization. Inputs: x array, dim an integer axis or a list of
    # axes. Computes the vector norm over the axis/axes, restores the reduced dimensions and
    # returns x divided by the norm.
    if isinstance(dim, list):
      norm = np.linalg.norm(x, axis=tuple(dim))
      for d in dim:
        norm = np.expand_dims(norm, d)
      return x / norm
    else:
      norm = np.apply_along_axis(np.linalg.norm, dim, x)
      return x / np.expand_dims(norm, dim)

  def testL2Normalize(self):
    # [FUNCTION DOC]
    # For each axis of a seeded random 20x7x3 float32 tensor, asserts that l2_normalize matches
    # the NumPy reference.
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float32)
    for dim in range(len(x_shape)):
      y_np = self._l2Normalize(x_np, dim)
      x_tf = constant_op.constant(x_np, name="x")
      y_tf = nn_impl.l2_normalize(x_tf, dim)
      self.assertAllClose(y_np, self.evaluate(y_tf))

  def testL2NormalizeDimArray(self):
    # [FUNCTION DOC]
    # Asserts l2_normalize over a list of two axes (one and two) of a seeded random float32
    # tensor matches the NumPy reference.
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float32)
    dim = [1, 2]
    y_np = self._l2Normalize(x_np, dim)
    x_tf = constant_op.constant(x_np, name="x")
    y_tf = nn_impl.l2_normalize(x_tf, dim)
    self.assertAllClose(y_np, self.evaluate(y_tf))

  def testL2NormalizeGradient(self):
    # [FUNCTION DOC]
    # Asserts theoretical and numerical gradients of l2_normalize (default axis) agree on a seeded
    # random 20x7x3 float64 tensor; other axes are not covered (open TODO).
    x_shape = [20, 7, 3]
    np.random.seed(1)
    x_np = np.random.random_sample(x_shape).astype(np.float64)
    with self.cached_session():
      x_tf = constant_op.constant(x_np, name="x")
      # TODO(b/241834841): Test l2_normalize with `axis` set to other dims
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_impl.l2_normalize, [x_tf])
      self.assertAllClose(theoretical, numerical)

  def testL2NormalizeComplex(self):
    # [FUNCTION DOC]
    # For complex64 and complex128 seeded random tensors of shape 20x7x3 and each axis, asserts
    # l2_normalize matches the NumPy reference.
    x_shape = [20, 7, 3]
    for dtype in [np.complex64, np.complex128]:
      np.random.seed(1)
      x_np = (
          np.random.random_sample(x_shape).astype(dtype) +
          np.random.random_sample(x_shape).astype(dtype) * 1j)
      for dim in range(len(x_shape)):
        y_np = self._l2Normalize(x_np, dim)
        x_tf = constant_op.constant(x_np, name="x")
        y_tf = nn_impl.l2_normalize(x_tf, dim)
        self.assertAllClose(y_np, self.evaluate(y_tf))


DROPOUT_FNS = [
    ("stateful_v1", nn_ops.dropout),
    ("stateful_v2", nn_ops.dropout_v2),
    ("stateless", functools.partial(nn_ops.stateless_dropout, seed=(1, 2))),
    ("stateless_philox", functools.partial(
        nn_ops.stateless_dropout, seed=(1, 2), rng_alg="philox")),
    ("generator", functools.partial(  # pylint: disable=g-long-lambda
        nn_ops.general_dropout, uniform_sampler=lambda shape, dtype: (  # pylint: disable=g-long-lambda
            stateful_random_ops.Generator.from_seed(1).uniform(
                shape=shape, dtype=dtype)))),
    ]


class DropoutTest(test_lib.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests over all dropout implementations in the dropout function table (stateful
  # v1/v2, stateless, stateless philox, generator-based): statistics of kept elements, shaped
  # (broadcast) noise, rate/keep-probability validation, unknown shapes, large rates, variable
  # inputs, zero rate and integer-input rejection.

  @parameterized.named_parameters(
      ("_%s_%s_%s" % (case_name, use_noise_shape, keep_prob), case_name,  # pylint: disable=g-complex-comprehension
       dropout_fn, use_noise_shape, keep_prob)
      for keep_prob in [0.1, 0.5, 0.8]
      for use_noise_shape in ["no", "concrete", "partial"]
      for case_name, dropout_fn in DROPOUT_FNS)
  def testDropout(self, case_name, dropout_fn, use_noise_shape, keep_prob):
    # [FUNCTION DOC]
    # Applies dropout ten times to an all-ones float32 tensor (without noise shape, with a
    # concrete noise shape, or with a partially unknown one) and asserts: static shape preserved;
    # outputs contain only zero and one over keep probability; total kept count is within 15
    # percent of expectation. Skipped for the generator case outside eager mode.
    # Runs dropout with 0-1 tensor 10 times, sum the number of ones and validate
    # that it is producing approximately the right number of ones over a large
    # number of samples, based on the keep probability.
    if "generator" in case_name and not context.executing_eagerly():
      self.skipTest("tf.random.Generator can only be used in TF2.")
    if use_noise_shape == "no":
      x_dim = 70
      y_dim = 30
    else:
      x_dim = 70 * 30
      y_dim = 3
    num_iter = 10
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    if use_noise_shape == "no":
      noise_shape = None
    elif use_noise_shape == "concrete":
      noise_shape = [x_dim, 1]
    else:
      noise_shape = [None, 1]
    dropout = dropout_fn(t, rate=(1 - keep_prob), noise_shape=noise_shape)
    final_count = 0
    self.assertEqual([x_dim, y_dim], dropout.get_shape())
    for _ in range(0, num_iter):
      value = self.evaluate(dropout)
      final_count += np.count_nonzero(value)
      # Verifies that there are only two values: 0 and 1/keep_prob.
      sorted_value = np.unique(np.sort(value))
      self.assertEqual(0, sorted_value[0])
      self.assertAllClose(1 / keep_prob, sorted_value[1])

    # Check that we are in the 15% error range
    expected_count = x_dim * y_dim * keep_prob * num_iter
    rel_error = math.fabs(final_count - expected_count) / expected_count
    self.assertLess(rel_error, 0.15)

  @parameterized.named_parameters(
      ("_%s_%s" % (case_name, keep_prob), case_name, dropout_fn, keep_prob)  # pylint: disable=g-complex-comprehension
      for keep_prob in [0.1, 0.5, 0.8]
      for case_name, dropout_fn in DROPOUT_FNS)
  def testShapedDropoutCorrelation(self, case_name, dropout_fn, keep_prob):
    # [FUNCTION DOC]
    # With a noise shape of one value per row on a 40x30 ones tensor, asserts the static shape is
    # preserved and, over ten runs, each row is entirely kept or entirely dropped. Skipped for
    # the generator case outside eager mode.
    # Runs a shaped dropout and tests that the correlations are correct.
    if "generator" in case_name and not context.executing_eagerly():
      self.skipTest("tf.random.Generator can only be used in TF2.")
    x_dim = 40
    y_dim = 30
    num_iter = 10
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    dropout = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim, 1])
    self.assertEqual([x_dim, y_dim], dropout.get_shape())
    for _ in range(0, num_iter):
      value = self.evaluate(dropout)
      # Verifies that each row has only one type of activation.
      for i in range(x_dim):
        sorted_value = np.unique(np.sort(value[i, :]))
        self.assertEqual(sorted_value.size, 1)

  @parameterized.named_parameters(
      ("_%s_%s_%s" % (case_name, keep_prob, use_keep_prob), case_name,  # pylint: disable=g-complex-comprehension
       dropout_fn, keep_prob, use_keep_prob)
      for use_keep_prob in [False, True]
      for keep_prob in [0.1, 0.5, 0.8]
      for case_name, dropout_fn in DROPOUT_FNS)
  def testDropoutPlaceholderRateAndKeepProb(self, case_name, dropout_fn,
                                            keep_prob, use_keep_prob):
    # [FUNCTION DOC]
    # Same statistical check as the basic dropout test, but passing either the rate or (only for
    # the stateful v1 implementation) the keep probability as keyword argument. Asserts shape,
    # only zero and one over keep probability values, and a kept count within 15 percent of the
    # expectation. Skips keep-probability cases for other implementations and the generator case
    # outside eager mode.
    # Runs dropout with 0-1 tensor 10 times, sum the number of ones and validate
    # that it is producing approximately the right number of ones over a large
    # number of samples, based on the keep probability.
    if use_keep_prob and case_name != "stateful_v1":
      self.skipTest("Only V1 `dropout` has the `keep_prob` argument.")
    if "generator" in case_name and not context.executing_eagerly():
      self.skipTest("tf.random.Generator can only be used in TF2.")
    x_dim = 70
    y_dim = 30
    num_iter = 10
    t = constant_op.constant(
        1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    final_count = 0
    for _ in range(0, num_iter):
      if use_keep_prob:
        dropout = dropout_fn(t, keep_prob=keep_prob)
      else:
        dropout = dropout_fn(t, rate=1 - keep_prob)
      self.assertEqual([x_dim, y_dim], dropout.get_shape())
      value = self.evaluate(dropout)
      final_count += np.count_nonzero(value)
      # Verifies that there are only two values: 0 and 1/keep_prob.
      sorted_value = np.unique(np.sort(value))
      self.assertEqual(0, sorted_value[0])
      self.assertAllClose(1 / keep_prob, sorted_value[1])
    # Check that we are in the 15% error range
    expected_count = x_dim * y_dim * keep_prob * num_iter
    rel_error = math.fabs(final_count - expected_count) / expected_count
    self.assertLess(rel_error, 0.15)

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  @test_util.run_deprecated_v1
  def testShapedDropoutUnknownShape(self, dropout_fn):
    # [FUNCTION DOC]
    # Graph mode: passes an int32 placeholder of unknown shape as noise shape and asserts the
    # static output shape still equals the input's static shape.
    x_dim = 40
    y_dim = 30
    keep_prob = 0.5
    x = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    dropout_x = dropout_fn(
        x,
        rate=(1 - keep_prob),
        noise_shape=array_ops.placeholder(dtypes.int32))
    self.assertEqual(x.get_shape(), dropout_x.get_shape())

  @parameterized.named_parameters(
      ("_%s_%s" % (case_name, use_keep_prob), case_name, dropout_fn,  # pylint: disable=g-complex-comprehension
       use_keep_prob)
      for use_keep_prob in [False, True]
      for case_name, dropout_fn in DROPOUT_FNS)
  @test_util.run_deprecated_v1
  def testInvalidRateAndKeepProb(self, case_name, dropout_fn, use_keep_prob):
    # [FUNCTION DOC]
    # Graph mode: asserts a value error for out-of-range rates (negative, above one), a list value,
    # a float64 placeholder, and a placeholder of shape two. Uses the keep-probability keyword only
    # for the stateful v1 implementation; other keep-probability cases are skipped.
    if use_keep_prob and case_name != "stateful_v1":
      self.skipTest("Only V1 `dropout` has the `keep_prob` argument.")
    if use_keep_prob:
      fn = lambda x, y: dropout_fn(x, keep_prob=y)
    else:
      fn = lambda x, y: dropout_fn(x, rate=y)
    x_dim = 40
    y_dim = 30
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      fn(t, -1.0)
    with self.assertRaises(ValueError):
      fn(t, 1.1)
    with self.assertRaises(ValueError):
      fn(t, [0.0, 1.0])
    with self.assertRaises(ValueError):
      fn(t, array_ops.placeholder(dtypes.float64))
    with self.assertRaises(ValueError):
      fn(t, array_ops.placeholder(dtypes.float32, shape=[2]))

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  def testLargeRate(self, dropout_fn):
    # [FUNCTION DOC]
    # Smoke test: each dropout implementation accepts a rate of 0.9 on a 40x30 tensor without
    # raising; the result is discarded.
    x_dim = 40
    y_dim = 30
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    _ = dropout_fn(t, rate=0.9)

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  def testVariableRef(self, dropout_fn):
    # [FUNCTION DOC]
    # Smoke test: dropout accepts a variable (created by variable-scope get-variable) as input
    # and does not raise; the result is discarded.
    x = variable_scope.get_variable("x", shape=[10, 10], dtype=dtypes.float32)
    _ = dropout_fn(x, rate=0.1)

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  @test_util.run_deprecated_v1
  def testShapedDropoutShapeError(self, dropout_fn):
    # [FUNCTION DOC]
    # Graph mode: asserts a value error for noise shapes incompatible with a 40x30 input (too
    # large dimension, wrong rank, wrong size), and that broadcast-compatible noise shapes are
    # accepted without error.
    # Runs shaped dropout and verifies an error is thrown on misshapen noise.
    x_dim = 40
    y_dim = 30
    keep_prob = 0.5
    t = constant_op.constant(1.0, shape=[x_dim, y_dim], dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      _ = dropout_fn(
          t, rate=(1 - keep_prob), noise_shape=[x_dim, y_dim + 10])
    with self.assertRaises(ValueError):
      _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim, y_dim, 5])
    with self.assertRaises(ValueError):
      _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim + 3])
    with self.assertRaises(ValueError):
      _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim])
    # test that broadcasting proceeds
    _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[y_dim])
    _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[1, y_dim])
    _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[x_dim, 1])
    _ = dropout_fn(t, rate=(1 - keep_prob), noise_shape=[1, 1])

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  def testNoDropout(self, dropout_fn):
    # [FUNCTION DOC]
    # With rate zero, dropout of a zero vector of length five must return an output equal to
    # the input, for every implementation.
    x = array_ops.zeros((5,))
    y = dropout_fn(x, rate=0)
    self.assertAllEqual(x, y)

  @parameterized.named_parameters(
      ("_%s" % case_name, dropout_fn)
      for case_name, dropout_fn in DROPOUT_FNS)
  def testDropoutWithIntegerInputs(self, dropout_fn):
    # [FUNCTION DOC]
    # Asserts every dropout implementation raises a value error when given an integer tensor.
    x = constant_op.constant([1, 1, 1, 1, 1])
    with self.assertRaises(ValueError):
      _ = dropout_fn(x, rate=0.5)


@test_util.run_all_without_tensor_float_32(
    "Tests _compute_sampled_logits and related functions, which call matmul")
class ComputeSampledLogitsTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the internal sampled-logits computation used by sampled softmax and NCE losses:
  # output shapes, values with and without removal of accidental hits and with log-probability
  # correction, and sharded (partitioned) weights. Holds a comparison tolerance set in setUp.
  # Includes helpers that generate random embeddings and expected outputs and shard weights.

  def setUp(self):
    # [FUNCTION DOC]
    # Sets the comparison tolerance attribute to 1e-3 used by the tests in this class.
    self._eps = 1e-3

  def _GenerateTestData(self, num_classes, dim, batch_size, num_true, labels,
                        sampled, subtract_log_q):
    # [FUNCTION DOC]
    # HELPER: generates random inputs and the expected outputs of the sampled-logits computation.
    # Inputs: num_classes, dim, batch_size, num_true, labels (flat list of target classes),
    # sampled (list of sampled class ids), subtract_log_q (whether expected logits subtract the
    # log of the expected counts, all set to 0.5). Returns a tuple: random float32 weights and
    # biases, hidden activations, the sampled-values tuple (ids, true expected counts, sampled
    # expected counts), the expected logits (true logits then sampled logits) and the expected
    # labels (one over num_true for true columns, zero for sampled columns).
    """Randomly generates input/output data for a single test case.

    This function returns numpy constants for use in a test case.

    Args:
      num_classes: An int. The number of embedding classes in the test case.
      dim: An int. The dimension of the embedding.
      batch_size: An int. The batch size.
      num_true: An int. The number of target classes per training example.
      labels: A list of batch_size * num_true ints. The target classes.
      sampled: A list of indices in [0, num_classes).
      subtract_log_q: A bool corresponding to the parameter in
        _compute_sampled_logits().

    Returns:
      weights: Embedding weights to use as test input. It is a numpy array
          of shape [num_classes, dim]
      biases: Embedding biases to use as test input. It is a numpy array
          of shape [num_classes].
      hidden_acts: Forward activations of the network to use as test input.
          It is a numpy array of shape [batch_size, dim].
      sampled_vals: A tuple based on `sampled` to use as test input in the
          format returned by a *_candidate_sampler function.
      exp_logits: The output logits expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
      exp_labels: The output labels expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
    """
    weights = np.random.randn(num_classes, dim).astype(np.float32)
    biases = np.random.randn(num_classes).astype(np.float32)
    hidden_acts = np.random.randn(batch_size, dim).astype(np.float32)

    true_exp = np.full([batch_size, 1], fill_value=0.5, dtype=np.float32)
    sampled_exp = np.full([len(sampled)], fill_value=0.5, dtype=np.float32)
    sampled_vals = (sampled, true_exp, sampled_exp)

    sampled_w, sampled_b = weights[sampled], biases[sampled]
    true_w, true_b = weights[labels], biases[labels]

    true_logits = np.sum(
        hidden_acts.reshape((batch_size, 1, dim)) * true_w.reshape(
            (batch_size, num_true, dim)),
        axis=2)
    true_b = true_b.reshape((batch_size, num_true))
    true_logits += true_b
    sampled_logits = np.dot(hidden_acts, sampled_w.T) + sampled_b

    if subtract_log_q:
      true_logits -= np.log(true_exp)
      sampled_logits -= np.log(sampled_exp[np.newaxis, :])

    exp_logits = np.concatenate([true_logits, sampled_logits], axis=1)
    exp_labels = np.hstack(
        (np.ones_like(true_logits) / num_true, np.zeros_like(sampled_logits)))

    return weights, biases, hidden_acts, sampled_vals, exp_logits, exp_labels

  def _ShardTestEmbeddings(self, weights, biases, num_shards):
    # [FUNCTION DOC]
    # HELPER: splits weights and biases into partitions. Inputs: weights and biases arrays and the
    # number of shards. Creates partitioned variables with a fixed-size partitioner in a fresh
    # graph, initializes them in a session and returns a list of evaluated weight shards and a
    # list of evaluated bias shards.
    """Shards the weights and biases returned by _GenerateTestData.

    Args:
      weights: The weights returned by _GenerateTestData.
      biases: The biases returned by _GenerateTestData.
      num_shards: The number of shards to create.

    Returns:
      sharded_weights: A list of size `num_shards` containing all the weights.
      sharded_biases: A list of size `num_shards` containing all the biases.
    """
    with ops.Graph().as_default() as g:
      sharded_weights = variable_scope.get_variable(
          "w",
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards),
          initializer=constant_op.constant(weights))
      sharded_biases = variable_scope.get_variable(
          "b",
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards),
          initializer=constant_op.constant(biases))
      with self.session(graph=g) as sess:
        self.evaluate(variables.global_variables_initializer())
        return self.evaluate([list(sharded_weights), list(sharded_biases)])

  def testShapes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertions call the equality check with the tolerance as a third argument,
    # which is only used as a failure message, not as a tolerance.
    # For one to four true labels per example, builds random data and calls the sampled-logits
    # computation without log-q correction or hit removal, and asserts the logits and labels
    # shapes match the expected ones.
    np.random.seed(0)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=[1, 0, 2, 3],
           subtract_log_q=False)
      logits_tensor, labels_tensor = _compute_sampled_logits(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=False,
          remove_accidental_hits=False,
          partition_strategy="div",
          name="sampled_logits_basic_num_true_%d" % num_true)
      got_logits, got_labels = self.evaluate([logits_tensor, labels_tensor])
      self.assertEqual(exp_logits.shape, got_logits.shape, self._eps)
      self.assertEqual(exp_labels.shape, got_labels.shape, self._eps)

  def testBasic(self):
    # [FUNCTION DOC]
    # For one to four true labels per example, computes sampled logits without log-q correction or
    # accidental-hit removal and asserts logits and labels match the generated expectations
    # within the tolerance.
    """Without accidental hit removal or subtract_log_q."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=[1, 0, 2, 3],
           subtract_log_q=False)
      logits_tensor, labels_tensor = _compute_sampled_logits(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=False,
          remove_accidental_hits=False,
          partition_strategy="div",
          name="sampled_logits_basic_num_true_%d" % num_true)
      got_logits, got_labels = self.evaluate([logits_tensor, labels_tensor])
      self.assertAllClose(exp_logits, got_logits, self._eps)
      self.assertAllClose(exp_labels, got_labels, self._eps)

  def testAccidentalHitRemoval(self):
    # [FUNCTION DOC]
    # For one to four true labels per example, computes sampled logits with accidental-hit removal
    # (and no log-q correction) and asserts that for every sampled class that coincides with a
    # true label of the example, the exponentiated logit is near zero (within the tolerance).
    """With accidental hit removal, no subtract_log_q."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    sampled = [1, 0, 2, 3]

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, _,
       _) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=sampled,
           subtract_log_q=False)
      logits_tensor, _ = _compute_sampled_logits(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=len(sampled),
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=False,
          remove_accidental_hits=True,
          partition_strategy="div",
          name="sampled_logits_accidental_hit_removal_num_true_%d" % num_true)
      # Test that the exponentiated logits of accidental hits are near 0.
      # First we need to find the hits in this random test run:
      labels_reshape = labels.reshape((batch_size, num_true))
      got_logits = self.evaluate(logits_tensor)
      for row in range(batch_size):
        row_labels = labels_reshape[row, :]
        for col in range(len(sampled)):
          if sampled[col] in row_labels:
            # We need to add the num_true_test offset into logits_*
            self.assertNear(
                np.exp(got_logits[row, col + num_true]), 0., self._eps)

  def testSubtractLogQ(self):
    # [FUNCTION DOC]
    # For one to four true labels per example, computes sampled logits with log-q correction and
    # without hit removal; asserts logits and labels match the generated expectations.
    """With subtract_log_q, no accidental hit removal."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=[1, 0, 2, 3],
           subtract_log_q=True)
      logits_tensor, labels_tensor = _compute_sampled_logits(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=True,
          remove_accidental_hits=False,
          partition_strategy="div",
          name="sampled_logits_subtract_log_q_num_true_%d" % num_true)
      got_logits, got_labels = self.evaluate([logits_tensor, labels_tensor])
      self.assertAllClose(exp_logits, got_logits, self._eps)
      self.assertAllClose(exp_labels, got_labels, self._eps)

  def testSharded(self):
    # [FUNCTION DOC]
    # Same as the basic test but weights and biases are supplied as three shards (lists of
    # constants produced by the sharding helper); asserts the logits and labels match the
    # expectations within the tolerance.
    """With sharded weights and sharded biases."""
    np.random.seed(0)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(
          low=0, high=num_classes, size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(
           num_classes=num_classes,
           dim=10,
           batch_size=batch_size,
           num_true=num_true,
           labels=labels,
           sampled=[1, 0, 2, 3],
           subtract_log_q=False)
      weight_shards, bias_shards = self._ShardTestEmbeddings(
          weights, biases, num_shards=3)
      logits_tensor, labels_tensor = _compute_sampled_logits(
          weights=[constant_op.constant(shard) for shard in weight_shards],
          biases=[constant_op.constant(shard) for shard in bias_shards],
          labels=constant_op.constant(
              labels, dtype=dtypes.int64, shape=(batch_size, num_true)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_vals,
          subtract_log_q=False,
          remove_accidental_hits=False,
          partition_strategy="div",
          name="sampled_logits_sharded_num_true_%d" % num_true)
      got_logits, got_labels = self.evaluate([logits_tensor, labels_tensor])
      self.assertAllClose(exp_logits, got_logits, self._eps)
      self.assertAllClose(exp_labels, got_labels, self._eps)

  def testNCELoss(self):
    # [FUNCTION DOC]
    # Generates data with log-q correction for one true label, computes the expected noise
    # contrastive loss as the row sums of a NumPy sigmoid cross entropy over expected logits and
    # labels, and asserts the version-2 NCE loss matches within 1e-4, both with unsharded and
    # with three-way sharded weights and biases.
    # A simple test to verify the numerics.

    def _SigmoidCrossEntropyWithLogits(logits, targets):
      # [FUNCTION DOC]
      # HELPER: NumPy reference sigmoid cross entropy. Takes two same-shaped arrays (asserted);
      # clips the sigmoid prediction to epsilon 0.0001 and returns the elementwise loss array.
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      pred = 1. / (1. + np.exp(-logits))
      eps = 0.0001
      pred = np.minimum(np.maximum(pred, eps), 1 - eps)
      return -targets * np.log(pred) - (1. - targets) * np.log(1. - pred)

    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals, exp_logits,
     exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=10,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=[1, 0, 2, 3],
         subtract_log_q=True)
    exp_nce_loss = np.sum(
        _SigmoidCrossEntropyWithLogits(exp_logits, exp_labels), 1)

    got_nce_loss = nn_impl.nce_loss_v2(
        weights=constant_op.constant(weights),
        biases=constant_op.constant(biases),
        labels=constant_op.constant(labels, shape=(batch_size, 1)),
        inputs=constant_op.constant(hidden_acts),
        num_sampled=4,
        num_classes=num_classes,
        num_true=1,
        sampled_values=sampled_vals)

    self.assertAllClose(exp_nce_loss, self.evaluate(got_nce_loss), 1e-4)

    # Test with sharded weights and sharded biases.
    weight_shards, bias_shards = self._ShardTestEmbeddings(
        weights, biases, num_shards=3)
    got_nce_loss = nn_impl.nce_loss_v2(
        weights=[constant_op.constant(shard) for shard in weight_shards],
        biases=[constant_op.constant(shard) for shard in bias_shards],
        labels=constant_op.constant(labels, shape=(batch_size, 1)),
        inputs=constant_op.constant(hidden_acts),
        num_sampled=4,
        num_classes=num_classes,
        num_true=1,
        sampled_values=sampled_vals)

    self.assertAllClose(exp_nce_loss, self.evaluate(got_nce_loss), 1e-4)

  def testSampledSoftmaxLoss(self):
    # [FUNCTION DOC]
    # Generates data with log-q correction for one true label, computes the expected loss with a
    # NumPy softmax cross entropy over the expected logits and labels, and asserts the version-2
    # sampled softmax loss (no accidental-hit removal) matches within 1e-4 for unsharded and
    # three-way sharded weights and biases.
    # A simple test to verify the numerics.

    def _SoftmaxCrossEntropyWithLogits(logits, targets):
      # [FUNCTION DOC]
      # HELPER: NumPy reference softmax cross entropy. Takes two same-shaped arrays (asserted);
      # uses a max-shifted softmax, adds a tiny constant before the log and returns the per-row
      # loss vector.
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      stable_exp_logits = np.exp(logits -
                                 np.amax(logits, axis=1, keepdims=True))
      pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
      return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals, exp_logits,
     exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=10,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=[1, 0, 2, 3],
         subtract_log_q=True)
    exp_sampled_softmax_loss = _SoftmaxCrossEntropyWithLogits(
        exp_logits, exp_labels)

    got_sampled_softmax_loss = nn_impl.sampled_softmax_loss_v2(
        weights=constant_op.constant(weights),
        biases=constant_op.constant(biases),
        labels=constant_op.constant(labels, shape=(batch_size, 1)),
        inputs=constant_op.constant(hidden_acts),
        num_sampled=4,
        num_classes=num_classes,
        num_true=1,
        sampled_values=sampled_vals,
        remove_accidental_hits=False)

    self.assertAllClose(exp_sampled_softmax_loss,
                        self.evaluate(got_sampled_softmax_loss), 1e-4)

    # Test with sharded weights and sharded biases.
    weight_shards, bias_shards = self._ShardTestEmbeddings(
        weights, biases, num_shards=3)
    got_sampled_softmax_loss = nn_impl.sampled_softmax_loss_v2(
        weights=[constant_op.constant(shard) for shard in weight_shards],
        biases=[constant_op.constant(shard) for shard in bias_shards],
        labels=constant_op.constant(labels, shape=(batch_size, 1)),
        inputs=constant_op.constant(hidden_acts),
        num_sampled=4,
        num_classes=num_classes,
        num_true=1,
        sampled_values=sampled_vals,
        remove_accidental_hits=False)

    self.assertAllClose(exp_sampled_softmax_loss,
                        self.evaluate(got_sampled_softmax_loss), 1e-4)

  def testSampledSoftmaxLossBf16(self):
    # [FUNCTION DOC]
    # Like the sampled softmax test but all inputs (weights, biases, labels, activations and
    # expected counts) are bfloat16; the result is cast to float32 and compared with the float
    # NumPy reference using a loose tolerance of 0.1.
    # A simple test to verify the numerics for bfloat16.
    def _SoftmaxCrossEntropyWithLogits(logits, targets):
      # [FUNCTION DOC]
      # HELPER: NumPy reference softmax cross entropy. Takes two same-shaped arrays (asserted);
      # uses a max-shifted softmax, adds a tiny constant before the log and returns the per-row
      # loss vector.
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      stable_exp_logits = np.exp(logits -
                                 np.amax(logits, axis=1, keepdims=True))
      pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
      return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

    np.random.seed(0)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    sampled = [1, 0, 2, 3]
    (weights, biases, hidden_acts, _, exp_logits,
     exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=10,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=sampled,
         subtract_log_q=True)
    exp_sampled_softmax_loss = _SoftmaxCrossEntropyWithLogits(
        exp_logits, exp_labels)

    true_exp_bf16 = np.full([batch_size, 1],
                            fill_value=0.5,
                            dtype=dtypes.bfloat16.as_numpy_dtype)
    sampled_exp_bf16 = np.full([len(sampled)],
                               fill_value=0.5,
                               dtype=dtypes.bfloat16.as_numpy_dtype)
    sampled_vals_bf16 = (sampled, true_exp_bf16, sampled_exp_bf16)

    got_sampled_softmax_loss = math_ops.cast(
        nn_impl.sampled_softmax_loss_v2(
            weights=constant_op.constant(weights, dtype=dtypes.bfloat16),
            biases=constant_op.constant(biases, dtype=dtypes.bfloat16),
            labels=constant_op.constant(
                labels, shape=(batch_size, 1), dtype=dtypes.bfloat16),
            inputs=constant_op.constant(hidden_acts, dtype=dtypes.bfloat16),
            num_sampled=4,
            num_classes=num_classes,
            num_true=1,
            sampled_values=sampled_vals_bf16,
            remove_accidental_hits=False), dtypes.float32)

    self.assertAllClose(exp_sampled_softmax_loss,
                        self.evaluate(got_sampled_softmax_loss), 1e-1)


class CReluTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test for the concatenated ReLU activation, which concatenates the positive part and the
  # negated negative part along the last axis.

  def test(self):
    # [FUNCTION DOC]
    # On a seeded random 3x4 float32 array, asserts crelu equals the concatenation along axis one
    # of the positive part and the negated negative part, within 1e-4.
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.concatenate([x * (x > 0), -x * (x < 0)], axis=1)

    z = self.evaluate(nn_ops.crelu(constant_op.constant(x)))
    self.assertAllClose(y, z, 1e-4)


class ReluTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the ReLU activation: values on random data and propagation of not-a-number inputs.

  def test(self):
    # [FUNCTION DOC]
    # On a seeded random 3x4 float32 array, asserts relu equals the elementwise maximum with zero.
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.maximum(x, 0.0)

    z = self.evaluate(nn_ops.relu(constant_op.constant(x)))
    self.assertAllEqual(y, z)

  @test_util.disable_xla(
      "This test relies on undefined behavior that XLA does not replicate")
  @test_util.run_deprecated_v1
  def testNaNs(self):
    # [FUNCTION DOC]
    # Graph mode, disabled under XLA (relies on undefined behavior there). For input lengths 0 to
    # 17 of all not-a-number values, evaluates relu on CPU only and asserts every output is
    # not-a-number. Known limitation: GPU is excluded because of an open bug.
    # Test that relu(nan) = nan for various sizes.
    for i in range(18):
      x = np.zeros(i) + np.nan
      # TODO(b/178335491): This is broken on GPU today.
      with self.cached_session(use_gpu=False):
        z = nn_ops.relu(constant_op.constant(x)).eval()
        self.assertTrue(np.isnan(z).all())


class LeakyReluTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the leaky ReLU activation: positive-range pass-through, values across numeric types
  # with default slope, and default/explicit op naming.

  def testRange(self):
    # [FUNCTION DOC]
    # Applies leaky ReLU to random uniform positive 3x4x4x3 float32 data and asserts the shape is
    # preserved, outputs lie between zero and one and equal the inputs.
    batch_size = 3
    height, width = 4, 4
    np.random.seed(1)  # Make it reproducible.
    inputs = np.random.uniform(size=(batch_size, height, width,
                                     3)).astype(np.float32)
    inputs = constant_op.constant(inputs)

    outputs = nn_ops.leaky_relu(inputs)
    self.assertEqual(inputs.shape, outputs.shape)

    inputs, outputs = self.evaluate([inputs, outputs])

    self.assertGreaterEqual(outputs.min(), 0.0)
    self.assertLessEqual(outputs.max(), 1.0)
    self.assertAllClose(inputs, outputs)

  @test_util.run_deprecated_v1
  def testValues(self):
    # [FUNCTION DOC]
    # Graph mode. For int32, int64, float16, float32 and float64 inputs minus two to two, asserts
    # the default-slope (0.2) leaky ReLU gives minus 0.4, minus 0.2, 0, 1, 2 (looser tolerance for
    # float16).
    for dtype in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      np_values = np.array([-2, -1, 0, 1, 2], dtype=dtype)
      outputs = nn_ops.leaky_relu(constant_op.constant(np_values))

      outputs = self.evaluate(outputs)

      tol = 2e-3 if dtype == np.float16 else 1e-6
      self.assertAllClose(
          outputs, [-0.4, -0.2, 0.0, 1.0, 2.0], rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testName(self):
    # [FUNCTION DOC]
    # Graph mode. Asserts the leaky ReLU output tensor is named as given when a name is passed and
    # has the default name for the op otherwise.
    np_values = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    outputs_with_name_set = nn_ops.leaky_relu(
        constant_op.constant(np_values), name="test_relu_op")
    self.assertEqual(outputs_with_name_set.name, "test_relu_op:0")
    outputs_without_name_set = nn_ops.leaky_relu(
        constant_op.constant(np_values))
    self.assertEqual(outputs_without_name_set.name, "LeakyRelu:0")


class GeluTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test for the GELU activation, exact and tanh-approximate forms, against NumPy/SciPy
  # references including negative arguments.

  def test(self):
    # [FUNCTION DOC]
    # Evaluates gelu on 1000 float32 points from minus twelve to five, exact form compared with
    # the product of x and the normal CDF (relative tolerance 2e-5, no absolute tolerance), and
    # the approximate form compared with the tanh formula using default tolerances. Requires SciPy.

    def gelu(x, approximate=False):
      # [FUNCTION DOC]
      # HELPER: reference GELU. Input x array and a flag approximate (default false). With the
      # flag, returns the tanh-based approximation; otherwise x times the standard normal CDF
      # (imports SciPy lazily). Returns an array.
      if approximate:
        return 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
      else:
        from scipy.stats import norm  # pylint: disable=g-import-not-at-top
        return x * norm.cdf(x)

    # Make sure we test for negative arguments where GeLU is difficult to
    # evaluate accurately.
    x = np.linspace(-12, 5, 1000).astype(np.float32)
    y = gelu(x)
    z = self.evaluate(nn_ops.gelu(x))
    self.assertAllClose(y, z, atol=0, rtol=2e-5)

    y = gelu(x, True)
    z = self.evaluate(nn_ops.gelu(x, True))
    self.assertAllClose(y, z)


@test_util.run_all_in_graph_and_eager_modes
class SwishTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the swish activation (x times sigmoid of beta times x): values with default and
  # custom beta, and gradient checks. Runs in graph and eager modes.

  def testValues(self):
    # [FUNCTION DOC]
    # Asserts swish on a 2x100 float32 range from minus seven to seven equals x times sigmoid of x.
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    tf_values = constant_op.constant(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values)
    expected_tf_outputs = tf_values * math_ops.sigmoid(tf_values)

    actual_outputs, expected_outputs = self.evaluate(
        [actual_tf_outputs, expected_tf_outputs])

    self.assertAllClose(actual_outputs, expected_outputs)

  def testValuesWithBeta(self):
    # [FUNCTION DOC]
    # Asserts swish with beta 0.5 equals x times sigmoid of half x on the same range.
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    tf_values = constant_op.constant(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values, beta=0.5)
    expected_tf_outputs = tf_values * math_ops.sigmoid(0.5 * tf_values)

    actual_outputs, expected_outputs = self.evaluate(
        [actual_tf_outputs, expected_tf_outputs])

    self.assertAllClose(actual_outputs, expected_outputs)

  def testGradients(self):
    # [FUNCTION DOC]
    # Asserts theoretical and numerical gradients of default swish agree on random 5x3x4 input
    # scaled by five.
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = constant_op.constant(input_values)
    with self.cached_session():
      def f(x):  # pylint: disable=invalid-name
        # [FUNCTION DOC]
        # HELPER: wraps swish with default beta so the gradient checker can call it.
        # Takes a tensor and returns the swish activation of it.
        return nn_impl.swish(x)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [x_tf])
      self.assertAllClose(theoretical, numerical)

  def testGradientsWithBeta(self):
    # [FUNCTION DOC]
    # Asserts theoretical and numerical gradients of swish with beta 0.5 agree on random 5x3x4
    # input scaled by five.
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = constant_op.constant(input_values)
    with self.cached_session():
      def f(x):  # pylint: disable=invalid-name
        # [FUNCTION DOC]
        # HELPER: wraps swish with beta 0.5 so the gradient checker can call it.
        # Takes a tensor and returns the swish activation of it.
        return nn_impl.swish(x, beta=0.5)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [x_tf])
      self.assertAllClose(theoretical, numerical)


class MomentsTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the moments (mean and variance) op across 2-D and 4-D inputs and several reduction
  # axes. All cases delegate to one shared helper that checks values, absence of not-a-number
  # results, and optionally gradients.

  def doOutputTest(self,
                   input_shape,
                   moments_axes,
                   tol=1e-4,
                   check_gradients=False):
    # [FUNCTION DOC]
    # HELPER: shared body for the moments tests. Inputs: input_shape, moments_axes (axes to
    # reduce), tol (relative and absolute tolerance, default 1e-4), check_gradients (default
    # false). For means 0, 1, 1000, spreads 1.0, 0.1 and keep-dims true/false, builds random
    # float32 data in a new graph, computes moments with the version-2 API, optionally asserts
    # gradient errors of mean and variance below 1e-3, then asserts no not-a-number and closeness
    # to NumPy mean and variance.
    for mu in [0.0, 1.0, 1e3]:
      for sigma in [1.0, 0.1]:
        for keep_dims in [True, False]:
          input_values = np.random.rand(*input_shape) * sigma + mu
          expected_mean = np.mean(
              input_values, axis=moments_axes, keepdims=keep_dims)
          expected_var = np.var(
              input_values, axis=moments_axes, keepdims=keep_dims)
          with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
              inputs = constant_op.constant(
                  input_values, shape=input_shape, dtype=dtypes.float32)
              mean, variance = nn_impl.moments_v2(
                  inputs, moments_axes, keepdims=keep_dims)

              if check_gradients:
                err = gradient_checker.compute_gradient_error(
                    inputs, input_shape, mean, mean.shape.as_list())
                self.assertLess(err, 1e-3)
                err = gradient_checker.compute_gradient_error(
                    inputs, input_shape, variance, variance.shape.as_list())
                self.assertLess(err, 1e-3)

              # Evaluate.
              [mean, variance] = self.evaluate([mean, variance])
              # Make sure that there are no NaNs
              self.assertFalse(np.isnan(mean).any())
              self.assertFalse(np.isnan(variance).any())
              self.assertAllClose(mean, expected_mean, rtol=tol, atol=tol)
              self.assertAllClose(variance, expected_var, rtol=tol, atol=tol)

  def testOutputAndGradient2DInput0(self):
    # [FUNCTION DOC]
    # Moments of a 10x10 input over axis zero, with gradient checking.
    self.doOutputTest((10, 10), (0,), check_gradients=True)

  def testOutputAndGradient2DInput01(self):
    # [FUNCTION DOC]
    # Moments of a 10x10 input over axes zero and one, with gradient checking.
    self.doOutputTest((10, 10), (0, 1), check_gradients=True)

  def testOutput2DInput0(self):
    # [FUNCTION DOC]
    # Moments of a 10x300 input over axis zero (values only).
    self.doOutputTest((10, 300), (0,))

  def testOutput2DInput1(self):
    # [FUNCTION DOC]
    # Moments of a 10x300 input over axis one (values only).
    self.doOutputTest((10, 300), (1,))

  def testOutput2DInput01(self):
    # [FUNCTION DOC]
    # Moments of a 10x300 input over axes zero and one (values only).
    self.doOutputTest((10, 300), (0, 1))

  def testOutput4DInput0(self):
    # [FUNCTION DOC]
    # Moments of a 10x10x10x30 input over axis zero (values only).
    self.doOutputTest((10, 10, 10, 30), (0,))

  def testOutput4DInput1(self):
    # [FUNCTION DOC]
    # Moments of a 10x10x10x30 input over axis one (values only).
    self.doOutputTest((10, 10, 10, 30), (1,))

  def testOutput4DInput3(self):
    # [FUNCTION DOC]
    # Moments of a 10x10x10x30 input over axis three (values only).
    self.doOutputTest((10, 10, 10, 30), (3,))

  def testOutput4DInput012(self):
    # [FUNCTION DOC]
    # Moments of a 10x10x10x30 input over axes zero, one and two (values only).
    self.doOutputTest((10, 10, 10, 30), (0, 1, 2))

  def testOutput4DInput123(self):
    # [FUNCTION DOC]
    # Moments of a 10x10x10x30 input over axes one, two and three (values only).
    self.doOutputTest((10, 10, 10, 30), (1, 2, 3))


class DataFormatDimMapTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the op mapping dimension indices between data layouts (for example NHWC to NCHW),
  # including negative indices, nested inputs and different layout permutations.

  def _test(self, x_val, y_val_expected):
    # [FUNCTION DOC]
    # HELPER: maps the given dimension index value(s) with the default layout mapping and asserts
    # the result equals the expected value(s). Inputs: x_val (integer or nested list of integers),
    # y_val_expected (expected mapped values).
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x)

    y_val = self.evaluate(y)
    self.assertAllEqual(y_val, y_val_expected)

  def test(self):
    # [FUNCTION DOC]
    # Checks the default NHWC to NCHW mapping for scalars (including negative indices), lists
    # and a nested list via the shared helper.
    self._test(0, 0)
    self._test(1, 2)
    self._test(2, 3)
    self._test(3, 1)
    self._test(-1, 1)
    self._test(-2, 3)
    self._test(-3, 2)
    self._test(-4, 0)
    self._test([1, 3], [2, 1])
    self._test([1, 3, -2], [2, 1, 3])
    self._test([1, -3, -2], [2, 2, 3])
    self._test([[1, -3], [1, -1]], [[2, 2], [2, 1]])

  def testNHWCtoNCHW(self):
    # [FUNCTION DOC]
    # With explicit source layout NHWC and destination NCHW, on the GPU-preferring device, asserts
    # indices one, minus three, minus two map to two, two, three.
    x_val = [1, -3, -2]
    y_val_expected = [2, 2, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="NCHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNHWCtoHWNC(self):
    # [FUNCTION DOC]
    # Maps indices minus four through three from layout NHWC to HWNC on the GPU-preferring device
    # and asserts the expected permuted indices.
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [2, 0, 1, 3, 2, 0, 1, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNHWCtoWHCN(self):
    # [FUNCTION DOC]
    # Maps indices minus four through three from layout NHWC to WHCN on the GPU-preferring device
    # and asserts the expected permuted indices.
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [3, 1, 0, 2, 3, 1, 0, 2]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNDHWCtoNCDHW(self):
    # [FUNCTION DOC]
    # Maps indices one, minus four, minus three, minus two from the five-dimensional layout NDHWC
    # to NCDHW on the GPU-preferring device and asserts the result is two, two, three, four.
    x_val = [1, -4, -3, -2]
    y_val_expected = [2, 2, 3, 4]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNDHWCtoDHWNC(self):
    # [FUNCTION DOC]
    # Maps indices minus five through four from layout NDHWC to DHWNC on the GPU-preferring device
    # and asserts the expected permuted indices.
    x_val = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    y_val_expected = [3, 0, 1, 2, 4, 3, 0, 1, 2, 4]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="DHWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testDNHWCtoWHDCN(self):
    # [FUNCTION DOC]
    # Maps indices minus five through four from layout NDHWC to WHDCN on the GPU-preferring device
    # and asserts the expected permuted indices.
    x_val = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    y_val_expected = [4, 2, 1, 0, 3, 4, 2, 1, 0, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="WHDCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testArbitraryASCII(self):
    # [FUNCTION DOC]
    # Uses arbitrary four-letter format strings as source and destination layouts and asserts the
    # mapping of indices minus four through three follows the letter permutation.
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [3, 2, 1, 0, 3, 2, 1, 0]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="qwer", dst_format="rewq")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testInvalidLength(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error saying the source format must be of length four or five
    # when eight-character format strings are given (disabled under XLA, which rethrows a
    # different error).
    x = [-4, -3, -2, -1, 0, 1, 2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Source format must be of length 4 or 5"):
      op = nn_ops.data_format_dim_map(
          x, src_format="12345678", dst_format="87654321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testDuplicateSrc(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error (formats must determine a permutation) when the source
    # format has a repeated letter.
    x = [-4, -3, -2, -1, 0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_dim_map(x, src_format="1233", dst_format="4321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testDuplicateDst(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error (formats must determine a permutation) when the
    # destination format has a repeated letter.
    x = [-4, -3, -2, -1, 0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_dim_map(x, src_format="1234", dst_format="3321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testExtraSpecifiers(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error (formats must determine a permutation) when the
    # destination format contains a letter absent from the source format.
    x = [-4, -3, -2, -1, 0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_dim_map(x, src_format="1234", dst_format="5321")
      with test_util.use_gpu():
        self.evaluate(op)


class DataFormatVectorPermuteTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for the op that permutes a vector (or a matrix of row pairs) of per-dimension values
  # from one data layout to another: four- and five-dimensional layouts, size-2 and size-3
  # spatial-only vectors, 2-D inputs, unsupported-by-XLA layouts, and format validation errors.
  # Each test evaluates on the GPU-preferring device and compares with a fixed expected result.

  def testNHWCToNCHW(self):
    # [FUNCTION DOC]
    # Default permutation (NHWC to NCHW) of a four-element vector; expects the batch first, then
    # channel, then height and width.
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 3, 4, 9])

  def testNHWCToNCHW_Size2(self):
    # [FUNCTION DOC]
    # Default permutation of a two-element (spatial only) vector; expects it to be unchanged.
    x_val = [4, 9]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [4, 9])

  def testNDHWCtoNCDHW(self):
    # [FUNCTION DOC]
    # Permutes a five-element vector from NDHWC to NCDHW and expects the channel entry moved
    # second.
    x_val = [7, 4, 9, 3, 5]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 5, 4, 9, 3])

  def testNDHWCtoNCDHW_Size3(self):
    # [FUNCTION DOC]
    # Permutes a three-element (spatial only) vector from NDHWC to NCDHW; expects it unchanged.
    x_val = [4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [4, 9, 3])

  @test_util.disable_xla("unsupported data format")
  def testNHWCToWHCN(self):
    # [FUNCTION DOC]
    # Permutes a four-element vector from NHWC to WHCN and expects the reversed-order result
    # (disabled under XLA: unsupported format).
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 4, 3, 7])

  @test_util.disable_xla("unsupported data format")
  def testNHWCToWHCN_Size2(self):
    # [FUNCTION DOC]
    # Permutes the two spatial entries from NHWC to WHCN, which swaps them (disabled under XLA).
    x_val = [4, 9]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 4])

  @test_util.disable_xla("unsupported data format")
  def testNDHWCToWHDCN(self):
    # [FUNCTION DOC]
    # Permutes a five-element vector from NDHWC to WHDCN and expects the reversed-order result
    # (disabled under XLA: unsupported format).
    x_val = [7, 4, 9, 3, 5]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="WHDCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [3, 9, 4, 5, 7])

  @test_util.disable_xla("unsupported data format")
  def testNDHWCToWHDCN_Size3(self):
    # [FUNCTION DOC]
    # Permutes three spatial entries from NDHWC to WHDCN, which reverses them (disabled under XLA).
    x_val = [4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="WHDCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [3, 9, 4])

  def testNCHWToNHWC(self):
    # [FUNCTION DOC]
    # Permutes a four-element vector from NCHW to NHWC and expects the channel entry moved last.
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NCHW", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 9, 3, 4])

  def testNCHWToNHWC_Size2(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test name suggests NCHW to NHWC, but no formats are passed, so it actually
    # exercises the default NHWC to NCHW permutation of two spatial entries.
    # Expects the two-element vector to be unchanged.
    x_val = [9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 3])

  def testNCDHWToNDHWC(self):
    # [FUNCTION DOC]
    # Permutes a five-element vector from NCDHW to NDHWC and expects the channel entry moved last.
    x_val = [7, 4, 9, 3, 5]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NCDHW", dst_format="NDHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 9, 3, 5, 4])

  def testNCDHWToNDHWC_Size3(self):
    # [FUNCTION DOC]
    # Permutes three spatial entries from NCDHW to NDHWC; expects them unchanged.
    x_val = [9, 3, 5]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NCDHW", dst_format="NDHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 3, 5])

  def testNHWCToHWNC(self):
    # [FUNCTION DOC]
    # Permutes a four-element vector from NHWC to HWNC and checks the expected order.
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [4, 9, 7, 3])

  def testHWNCToNHWC(self):
    # [FUNCTION DOC]
    # Permutes a four-element vector from HWNC to NHWC and checks the expected order.
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="HWNC", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 7, 4, 3])

  def testNHWCToNCHW2D(self):
    # [FUNCTION DOC]
    # Default permutation of a 4x2 matrix: whole rows are reordered from NHWC to NCHW order.
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [5, 1], [9, 3], [4, 5]])

  def testNHWCToHWNC2D(self):
    # [FUNCTION DOC]
    # Row permutation of a 4x2 matrix from NHWC to HWNC and checks the expected rows.
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[9, 3], [4, 5], [7, 4], [5, 1]])

  def testHWNCToNHWC2D(self):
    # [FUNCTION DOC]
    # Row permutation of a 4x2 matrix from HWNC to NHWC and checks the expected rows.
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="HWNC", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[4, 5], [7, 4], [9, 3], [5, 1]])

  def testNCHWToNHWC2D(self):
    # [FUNCTION DOC]
    # Row permutation of a 4x2 matrix from NCHW to NHWC and checks the expected rows.
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NCHW", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [4, 5], [5, 1], [9, 3]])

  def testNDHWCToNCDHW2D(self):
    # [FUNCTION DOC]
    # Row permutation of a 5x2 matrix from NDHWC to NCDHW and checks the expected rows.
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1], [8, 2]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [8, 2], [9, 3], [4, 5], [5, 1]])

  @test_util.disable_xla("unsupported data format")
  def testNDHWCToDHWNC2D(self):
    # [FUNCTION DOC]
    # Row permutation of a 5x2 matrix from NDHWC to DHWNC and checks the expected rows (disabled
    # under XLA: unsupported format).
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1], [8, 2]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NDHWC", dst_format="DHWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[9, 3], [4, 5], [5, 1], [7, 4], [8, 2]])

  @test_util.disable_xla("unsupported data format")
  def testDHWNCToNDHWC2D(self):
    # [FUNCTION DOC]
    # Row permutation of a 5x2 matrix from DHWNC to NDHWC and checks the expected rows (disabled
    # under XLA: unsupported format).
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1], [8, 2]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="DHWNC", dst_format="NDHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[5, 1], [7, 4], [9, 3], [4, 5], [8, 2]])

  def testNCDHWToNDHWC2D(self):
    # [FUNCTION DOC]
    # Row permutation of a 5x2 matrix from NCDHW to NDHWC and checks the expected rows.
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1], [8, 2]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(
        x, src_format="NCDHW", dst_format="NDHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [4, 5], [5, 1], [8, 2], [9, 3]])

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testInvalidLength(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error (source format must be of length four or five) when
    # eight-character formats are used for the vector permutation (disabled under XLA).
    x = [0, 1, 2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Source format must be of length 4 or 5"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="12345678", dst_format="87654321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testDuplicateSrc(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error (formats must determine a permutation) when the source
    # format repeats a letter (disabled under XLA).
    x = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="1233", dst_format="4321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testDuplicateDst(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error (formats must determine a permutation) when the
    # destination format repeats a letter (disabled under XLA).
    x = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="1234", dst_format="3321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testExtraSpecifiers(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error (formats must determine a permutation) when the
    # destination format has a letter missing from the source format (disabled under XLA).
    x = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Destination and source format must determine a permutation"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="1234", dst_format="5321")
      with test_util.use_gpu():
        self.evaluate(op)

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def test2DNoWH(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error saying the format must contain H and W when a 2-D input is
    # permuted using formats without those letters (disabled under XLA).
    x = [[0, 1], [2, 3]]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Format specifier must contain H and W for 2D case"):
      op = nn_ops.data_format_vec_permute(
          x, src_format="1234", dst_format="4321")
      with test_util.use_gpu():
        self.evaluate(op)


@test_util.run_all_in_graph_and_eager_modes
class AvgPoolTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests that the generic version-2 average pooling gives the same result as the dimension
  # specific ops (1-D, 2-D, 3-D) for tensor and NumPy inputs, plus a golden-value 1-D case.
  # Runs in graph and eager modes.

  def test1DTensor(self):
    # [FUNCTION DOC]
    # Applies generic and 1-D average pooling (window and stride 2, SAME padding) to a ones tensor
    # of shape 3x6x5 and asserts identical results.
    x = array_ops.ones([3, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool1d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DNumpy(self):
    # [FUNCTION DOC]
    # Same as the 1-D tensor test but with a NumPy ones array (float32 on ROCm builds, float64
    # otherwise); asserts generic and 1-D average pooling agree.
    # explicitly use float32 for ROCm, as MIOpen does not yet support float64
    # np.ones defaults to using float64 when dtype is not explicitly specified
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.ones([3, 6, 5], dtype=dtype)
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool1d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DNumpyWithGolden(self):
    # [FUNCTION DOC]
    # Average-pools a 2x3x1 NumPy array with window two, stride one and SAME padding using the
    # 1-D op and asserts the exact expected averages (edge windows are averaged over valid
    # elements only).
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.array([[[3], [6], [5]], [[1], [0], [1]]], dtype=dtype)
    ksize = 2
    strides = 1
    y = nn_ops.avg_pool1d(x, ksize, strides, "SAME")
    expected_y = np.array([[[4.5], [5.5], [5.0]], [[0.5], [0.5], [1.0]]],
                          dtype=dtype)
    self.assertAllEqual(self.evaluate(y), expected_y)

  def test2DTensor(self):
    # [FUNCTION DOC]
    # Generic versus 2-D average pooling on a ones tensor of shape 3x6x6x5; asserts equal results.
    x = array_ops.ones([3, 6, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test2DNumpy(self):
    # [FUNCTION DOC]
    # Generic versus 2-D average pooling on a NumPy ones array of shape 3x6x6x5 (float32 on ROCm
    # builds, float64 otherwise); asserts equal results.
    # explicitly use float32 for ROCm, as MIOpen does not yet support float64
    # np.ones defaults to using float64 when dtype is not explicitly specified
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.ones([3, 6, 6, 5], dtype=dtype)
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DTensor(self):
    # [FUNCTION DOC]
    # Generic versus 3-D average pooling on a ones tensor of shape 3x7x6x6x5; asserts equal results.
    x = array_ops.ones([3, 7, 6, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool3d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DNumpy(self):
    # [FUNCTION DOC]
    # Generic versus 3-D average pooling on a float32 NumPy ones array of shape 3x7x6x6x5;
    # asserts equal results.
    x = np.ones([3, 7, 6, 6, 5], dtype=np.float32)
    ksize = 2
    strides = 2

    y1 = nn_ops.avg_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.avg_pool3d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))


@test_util.run_all_in_graph_and_eager_modes
class MaxPoolTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests that the generic version-2 max pooling gives the same result as the dimension specific
  # ops (1-D, 2-D, 3-D) for tensor and NumPy inputs, a golden-value 1-D case, and error cases for
  # bad input rank and bad window sizes. Runs in graph and eager modes.

  def test1DTensor(self):
    # [FUNCTION DOC]
    # Generic versus 1-D max pooling (window and stride 2, SAME padding) on a ones tensor of
    # shape 3x6x5; asserts identical results.
    x = array_ops.ones([3, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool1d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DNumpy(self):
    # [FUNCTION DOC]
    # Same as the 1-D tensor test with a NumPy ones array (float32 on ROCm builds, float64
    # otherwise); asserts generic and 1-D max pooling agree.
    # explicitly use float32 for ROCm, as MIOpen does not yet support float64
    # np.ones defaults to using float64 when dtype is not explicitly specified
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.ones([3, 6, 5], dtype=dtype)
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool1d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DNumpyWithGolden(self):
    # [FUNCTION DOC]
    # Max-pools a 2x3x1 NumPy array with window two, stride one and SAME padding using the 1-D op
    # and asserts the exact expected maxima.
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.array([[[3], [6], [5]], [[1], [0], [1]]], dtype=dtype)
    ksize = 2
    strides = 1
    y = nn_ops.max_pool1d(x, ksize, strides, "SAME")
    expected_y = np.array([[[6], [6], [5]], [[1], [1], [1]]], dtype=dtype)
    self.assertAllEqual(self.evaluate(y), expected_y)

  def test2DTensor(self):
    # [FUNCTION DOC]
    # Generic versus 2-D max pooling on a ones tensor of shape 3x6x6x5; asserts equal results.
    x = array_ops.ones([3, 6, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test2DNumpy(self):
    # [FUNCTION DOC]
    # Generic versus 2-D max pooling on a NumPy ones array of shape 3x6x6x5 (float32 on ROCm
    # builds, float64 otherwise); asserts equal results.
    # explicitly use float32 for ROCm, as MIOpen does not yet support float64
    # np.ones defaults to using float64 when dtype is not explicitly specified
    dtype = np.float32 if test_lib.is_built_with_rocm() else np.float64
    x = np.ones([3, 6, 6, 5], dtype=dtype)
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DTensor(self):
    # [FUNCTION DOC]
    # Generic versus 3-D max pooling on a ones tensor of shape 3x7x6x6x5; asserts equal results.
    x = array_ops.ones([3, 7, 6, 6, 5])
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool3d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DNumpy(self):
    # [FUNCTION DOC]
    # Generic versus 3-D max pooling on a float32 NumPy ones array of shape 3x7x6x6x5; asserts
    # equal results.
    x = np.ones([3, 7, 6, 6, 5], dtype=np.float32)
    ksize = 2
    strides = 2

    y1 = nn_ops.max_pool_v2(x, ksize, strides, "SAME")
    y2 = nn_ops.max_pool3d(x, ksize, strides, "SAME")

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def testIncorrectSizeInputSmall(self):
    # [FUNCTION DOC]
    # Asserts max pooling a rank-2 input raises a value error saying the input rank must be 3, 4
    # or 5 and was 2.
    x = array_ops.ones([3, 4])
    with self.assertRaisesRegex(
        ValueError,
        "`input.shape.rank` must be 3, 4 or 5.*of rank 2."):
      nn_ops.max_pool_v2(x, 2, 2, "SAME")

  def testIncorrectSizeInput(self):
    # [FUNCTION DOC]
    # Asserts max pooling a rank-6 input raises a value error saying the input rank must be 3, 4
    # or 5 and was 6.
    x = array_ops.ones([3, 4, 1, 2, 1, 2])
    with self.assertRaisesRegex(
        ValueError,
        "`input.shape.rank` must be 3, 4 or 5.*of rank 6."):
      nn_ops.max_pool_v2(x, 2, 2, "SAME")

  @test_util.disable_xla("XLA catches the error and rethrows as different one")
  def testIncoorectKSize(self):
    # [FUNCTION DOC]
    # Two invalid window-size cases (disabled under XLA). First, negative window sizes must raise
    # an invalid-argument error that sizes must be positive. Second, a window size larger than the
    # platform maximum integer must raise an overflow error in eager mode and a value error in
    # graph mode. (The test name has a typo in its spelling.)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "Sliding window ksize must be positive."
    ):
      op = nn_ops.max_pool_v2(
          array_ops.ones([3, 4, 4, 5]), [1, -1, -1, 1], 2, "SAME"
      )
      with test_util.use_gpu():
        self.evaluate(op)

    ksize = sys.maxsize + 100  # Set to a value larger than sys.maxsize
    with self.assertRaises(
        OverflowError if context.executing_eagerly() else ValueError
    ):
      op = nn_ops.max_pool_v2(
          array_ops.ones([3, 4, 4, 5]), ksize=ksize, strides=2, padding="SAME"
      )
      with test_util.use_gpu():
        self.evaluate(op)


@test_util.run_all_in_graph_and_eager_modes
class ConvolutionTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test for the generic convolution op with inputs of unknown rank. Runs in graph and eager
  # modes.

  def testUnknownSize(self):
    # [FUNCTION DOC]
    # Traces a compiled function that applies convolution with a fixed ones kernel of shape
    # 3x6x6x5 to an input whose spec has fully unknown shape, and asserts that building the
    # concrete function succeeds without error (no values are evaluated).
    x = tensor_spec.TensorSpec(None, dtypes.float32, name="x")
    k = np.ones([3, 6, 6, 5], dtype=np.float32)

    @def_function.function
    def F(value):
      # [FUNCTION DOC]
      # HELPER: traced body for the unknown-shape test. Takes an input tensor and returns its
      # SAME-padded convolution with the enclosing kernel.
      return nn_ops.convolution(value, k, "SAME")

    F.get_concrete_function(x)


class ConvTransposeTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests that the generic transposed convolution matches the 1-D, 2-D and 3-D specific ops,
  # given the output shape as a list or as a tensor, plus error cases for bad output shapes.

  def test1D(self):
    # [FUNCTION DOC]
    # Compares generic and 1-D transposed convolution (output shape given as a list, stride 2) on
    # ones inputs; asserts equal results.
    t = array_ops.ones([2, 4, 3])
    v = array_ops.ones([2, 5, 3])
    strides = 2

    y1 = nn_ops.conv1d_transpose(t, v, [2, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, [2, 8, 5], strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test1DTensor(self):
    # [FUNCTION DOC]
    # Same as the 1-D test but the generic op receives the output shape as a constant tensor.
    t = array_ops.ones([2, 4, 3])
    v = array_ops.ones([2, 5, 3])
    strides = 2

    y1 = nn_ops.conv1d_transpose(t, v, [2, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, constant_op.constant([2, 8, 5]), strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test2D(self):
    # [FUNCTION DOC]
    # Compares generic and 2-D (version 2) transposed convolution with list output shape and
    # stride 2 on ones inputs; asserts equal results.
    t = array_ops.ones([2, 4, 4, 3])
    v = array_ops.ones([2, 2, 5, 3])
    strides = 2

    y1 = nn_ops.conv2d_transpose_v2(t, v, [2, 8, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, [2, 8, 8, 5], strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test2DTensor(self):
    # [FUNCTION DOC]
    # Same as the 2-D test but the generic op receives the output shape as a constant tensor.
    t = array_ops.ones([2, 4, 4, 3])
    v = array_ops.ones([2, 2, 5, 3])
    strides = 2

    y1 = nn_ops.conv2d_transpose_v2(t, v, [2, 8, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, constant_op.constant([2, 8, 8, 5]),
                               strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3D(self):
    # [FUNCTION DOC]
    # Compares generic and 3-D (version 2) transposed convolution with list output shape and
    # stride 2 on ones inputs; asserts equal results.
    t = array_ops.ones([2, 4, 4, 4, 3])
    v = array_ops.ones([2, 2, 2, 5, 3])
    strides = 2

    y1 = nn_ops.conv3d_transpose_v2(t, v, [2, 8, 8, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, [2, 8, 8, 8, 5], strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def test3DTensor(self):
    # [FUNCTION DOC]
    # Same as the 3-D test but the generic op receives the output shape as a constant tensor.
    t = array_ops.ones([2, 4, 4, 4, 3])
    v = array_ops.ones([2, 2, 2, 5, 3])
    strides = 2

    y1 = nn_ops.conv3d_transpose_v2(t, v, [2, 8, 8, 8, 5], strides)
    y2 = nn_ops.conv_transpose(t, v, constant_op.constant([2, 8, 8, 8, 5]),
                               strides)

    self.assertAllEqual(self.evaluate(y1), self.evaluate(y2))

  def testIncorrectSizeInputSmall(self):
    # [FUNCTION DOC]
    # Asserts a value error saying the output shape must have length 3, 4 or 5 when an output
    # shape of length two is given (inputs are placeholders of nothing/null).
    with self.assertRaisesRegex(
        ValueError,
        "`output_shape` must be of length 3, 4 or 5.*of length 2."):
      nn_ops.conv_transpose(None, 2, [2, 3], "SAME")

  def testIncorrectSizeInput(self):
    # [FUNCTION DOC]
    # Asserts a value error saying the output shape must have length 3, 4 or 5 when an output
    # shape of length six is given.
    with self.assertRaisesRegex(
        ValueError,
        "`output_shape` must be of length 3, 4 or 5.* of length 6."):
      nn_ops.conv_transpose(None, 2, [2, 3, 4, 2, 5, 1], "SAME")

  def testTensorsNoShape(self):
    # [FUNCTION DOC]
    # Asserts a value error saying the output shape must be a tensor or sized collection when the
    # output shape is null.
    with self.assertRaisesRegex(
        ValueError, "`output_shape` must be a tensor or sized collection"):
      nn_ops.conv_transpose(None, None, None, None)


class RaggedEmbeddingTest(test_lib.TestCase):
  # [CLASS DOC]
  # Tests for embedding lookup with ragged id tensors: ragged outputs for one and several ragged
  # dimensions, argument validation errors, and the maximum-norm option.

  def testRaggedTensor(self):
    # [FUNCTION DOC]
    # Looks up a ragged id tensor (rank 1 ragged) in a 4x3 weight table and asserts the result is
    # the ragged tensor of the corresponding weight rows.
    weights = constant_op.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    ragged_ids = ragged_factory_ops.constant([[1, 2, 3], [0], [1, 2]],
                                             ragged_rank=1)

    embedded_ragged = nn.embedding_lookup(weights, ragged_ids)
    expected_output = ragged_factory_ops.constant(
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[0, 0, 0]], [[1, 1, 1], [2, 2, 2]]
        ],
        ragged_rank=1)

    self.assertAllEqual(expected_output, embedded_ragged)

  def testMultipleRaggedDimTensor(self):
    # [FUNCTION DOC]
    # Looks up ragged ids with two ragged dimensions (including an empty row) in a 7x2 weight
    # table and asserts the nested ragged result of weight rows.
    weights = constant_op.constant([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                                    [5, 5], [6, 6]])
    ragged_ids = ragged_factory_ops.constant(
        [[[[3, 4], [0, 6]], []], [[[2, 1], [1, 0]], [[2, 5], [2, 3]]], [[[1, 0]]
                                                                       ]],
        ragged_rank=2)

    embedded_ragged = nn.embedding_lookup(weights, ragged_ids)
    expected_output = ragged_factory_ops.constant(
        [[[[[3, 3], [4, 4]], [[0, 0], [6, 6]]], []],
         [[[[2, 2], [1, 1]], [[1, 1], [0, 0]]],
          [[[2, 2], [5, 5]], [[2, 2], [3, 3]]]], [[[[1, 1], [0, 0]]]]],
        ragged_rank=2)

    self.assertAllEqual(expected_output, embedded_ragged)

  def testMissingWeights(self):
    # [FUNCTION DOC]
    # Asserts a value error saying the params must be specified when the weights are null.
    ragged_ids = ragged_factory_ops.constant([[1, 2, 3], [0], [1, 2]])

    with self.assertRaisesRegex(ValueError, "params must be specified.*"):
      nn.embedding_lookup(None, ragged_ids)

  def testEmptyWeights(self):
    # [FUNCTION DOC]
    # Asserts a value error saying the params must not be empty when an empty list of weights is
    # given.
    ragged_ids = ragged_factory_ops.constant([[1, 2, 3], [0], [1, 2]])

    with self.assertRaisesRegex(ValueError, "params should not be empty.*"):
      nn.embedding_lookup([], ragged_ids)

  def testInvalidIndicesType(self):
    # [FUNCTION DOC]
    # Asserts a value error about the value type of the inputs when the ragged ids are floating
    # point.
    weights = constant_op.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    ragged_ids = ragged_factory_ops.constant([[1., 2., 3.], [1., 2.]])

    with self.assertRaisesRegex(
        ValueError, "The values contained by the inputs have type*"):
      nn.embedding_lookup(weights, ragged_ids)

  def testMaxNormForEmbeddings(self):
    # [FUNCTION DOC]
    # Looks up ragged ids in a float 4x4 table with maximum norms 1, 2 and 5 and asserts each
    # result equals the table rows rescaled so no row norm exceeds the limit.
    weights = constant_op.constant(
        [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        dtype=dtypes.float32)
    ragged_ids = ragged_factory_ops.constant([[1, 2, 3], [0], [1, 2]],
                                             ragged_rank=1)

    actual_embeddings = [
        nn.embedding_lookup(weights, ragged_ids, max_norm=max_norm)
        for max_norm in [1, 2, 5]
    ]

    expected_embeddings = (
        # max_norm = 1
        [[[.5, .5, .5, .5], [.5, .5, .5, .5], [.5, .5, .5, .5]], [[0, 0, 0, 0]],
         [[.5, .5, .5, .5], [.5, .5, .5, .5]]],
        # max_norm = 2
        [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[0, 0, 0, 0]],
         [[1, 1, 1, 1], [1, 1, 1, 1]]],
        # max_norm = 5
        [[[1, 1, 1, 1], [2, 2, 2, 2], [2.5, 2.5, 2.5, 2.5]], [[0, 0, 0, 0]],
         [[1, 1, 1, 1], [2, 2, 2, 2]]],
    )

    for expected, actual in zip(expected_embeddings, actual_embeddings):
      self.assertAllClose(
          ragged_factory_ops.constant(expected, dtype=float, ragged_rank=1),
          actual)


class IsotonicTest(parameterized.TestCase, test_lib.TestCase):
  # [CLASS DOC]
  # Tests for isotonic regression (monotone fit via pooling adjacent violators): increasing and
  # decreasing fits, a non-default axis, gradients (graph and v2 variants) and output type
  # promotion. The op returns the fitted values and an integer segment id per element.

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_and_decreasing(self):
    # [FUNCTION DOC]
    # On two already increasing rows, an increasing fit returns the input with each element in
    # its own segment, while a decreasing fit averages each row into a single segment. Repeats
    # with the negated input to check the mirrored behavior. Runs in graph and eager modes.
    x = constant_op.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                             dtype=dtypes.float64)
    y, segments = nn_ops.isotonic_regression(x, decreasing=False)
    self.assertAllClose(y, x)
    self.assertAllClose(segments, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

    y, segments = nn_ops.isotonic_regression(x, decreasing=True)
    self.assertAllClose(
        y,
        [
            [2, 2, 2, 2, 2],  # Average of the inputs.
            [7, 7, 7, 7, 7]
        ])
    self.assertAllClose(segments, array_ops.zeros((2, 5)))

    # pylint: disable=invalid-unary-operand-type
    y, segments = nn_ops.isotonic_regression(-x, decreasing=True)
    self.assertAllClose(segments, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

    self.assertAllClose(y, -x)
    y, segments = nn_ops.isotonic_regression(-x, decreasing=False)
    self.assertAllClose(
        -y,
        [
            [2, 2, 2, 2, 2],  # Average of the inputs.
            [7, 7, 7, 7, 7]
        ])
    self.assertAllClose(segments, array_ops.zeros((2, 5)))

  @test_util.run_in_graph_and_eager_modes
  def test_different_axis(self):
    # [FUNCTION DOC]
    # Fits a decreasing isotonic regression along axis zero of a 2x5 matrix and asserts values
    # are either unchanged or averaged per column, and segment ids mark which columns pooled.
    x = constant_op.constant([[0, 6, 2, 8, 4], [5, 1, 7, 3, 9]],
                             dtype=dtypes.float64)
    y, segments = nn_ops.isotonic_regression(x, decreasing=True, axis=0)
    self.assertAllClose(
        y,
        [
            [2.5, 6, 4.5, 8, 6.5],  # Either identity or average.
            [2.5, 1, 4.5, 3, 6.5]
        ])
    self.assertAllClose(segments, [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0]])

  @test_util.run_v2_only
  def testGradientV2(self, dtype=np.float64, batch_size=30, dimensions=50):
    # [FUNCTION DOC]
    # Version-2-only test. Inputs with defaults: dtype float64, batch size 30, dimensions 50.
    # Compares theoretical and numerical gradients of a compiled isotonic regression of seeded
    # random data (finite difference step 1e-5) and asserts they are close.

    @def_function.function
    def ComputeIsotonicFn(x):
      # [FUNCTION DOC]
      # HELPER: wraps isotonic regression for the gradient checker. Takes a tensor, returns only
      # the fitted values (segments carry no gradient).
      y, _ = nn_ops.isotonic_regression(x)  # No gradient wrt segments.
      return y

    np.random.seed(0)
    x_init = np.random.randn(batch_size, dimensions).astype(dtype)
    grad_theoretical, grad_numerical = gradient_checker_v2.compute_gradient(
        ComputeIsotonicFn, [x_init], delta=1e-5)
    self.assertAllClose(grad_theoretical, grad_numerical)

  @test_util.run_v1_only("compute_gradient_error is v1 only")
  def testGradientV1(self, dtype=np.float64, batch_size=30, dimensions=50):
    # [FUNCTION DOC]
    # Version-1-only test with the same defaults as the v2 gradient test. Feeds a placeholder,
    # computes the gradient error of the isotonic output with the graph gradient checker using
    # seeded random initial data and asserts the maximum error is close to zero.
    np.random.seed(0)
    x_init = np.random.randn(batch_size, dimensions).astype(dtype)
    with self.cached_session():
      x = array_ops.placeholder(dtype, (batch_size, dimensions))
      y, _ = nn_ops.isotonic_regression(x)  # Segments have no gradient.
      max_error = gradient_checker.compute_gradient_error(
          x, (batch_size, dimensions), y, (batch_size, dimensions), x_init)
    self.assertAllClose(max_error, 0.)

  @parameterized.parameters([[dtypes.half, dtypes.half],
                             [dtypes.bfloat16, dtypes.bfloat16],
                             [dtypes.float32, dtypes.float32],
                             [dtypes.float64, dtypes.float64],
                             [dtypes.int32, dtypes.float64],
                             [dtypes.int16, dtypes.float32]])
  def testTypePromotion(self, dtype_in, expected_dtype_out):
    # [FUNCTION DOC]
    # Parameterized over input and expected output types: half, bfloat16, float32 and float64
    # stay the same, int32 promotes to float64 and int16 to float32. Asserts the fitted values
    # have the expected type and the segment ids are int32.
    x = constant_op.constant([[0, 6, 2, 8, 4], [5, 1, 7, 3, 9]], dtype=dtype_in)
    y, segments = nn_ops.isotonic_regression(x)
    self.assertEqual(y.dtype, expected_dtype_out)
    self.assertEqual(segments.dtype, dtypes.int32)


if __name__ == "__main__":
  test_lib.main()
