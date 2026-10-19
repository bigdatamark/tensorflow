# [MODULE DOC]
# Accuracy tests for special math functions (log1p, Hurwitz zeta, polygamma, regularized
# incomplete gamma igamma and igammac, and the gamma sample gradient) when compiled with XLA.
# Results are compared with scipy special functions over random inputs of float32 and float64,
# with per-device (notably TPU) tolerance adjustments and skips.
# Import-time side effects: defines a boolean command-line flag "vary_seed" (default false) that,
# when set, seeds numpy with unpredictable entropy in each test's setUp. Defines jit-compiled
# wrapper functions for the ops under test. When run as a program, sets the environment variable
# for XLA flags to turn off fast math on CPU before starting the test main.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for special math operations."""

import os

from absl import flags
from absl.testing import parameterized

import numpy as np
import scipy.special as sps

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

flags.DEFINE_bool('vary_seed', False,
                  ('Whether to vary the PRNG seed unpredictably.  '
                   'With --runs_per_test=N, produces N iid runs.'))

NUM_SAMPLES = int(1e3)


@def_function.function(jit_compile=True)
def _igamma(a, x):
  # [FUNCTION DOC]
  # HELPER: XLA-jit-compiled wrapper around the regularized lower incomplete gamma op, so the
  # tests exercise the XLA lowering. Takes shape parameter a and argument x; returns the result.
  return math_ops.igamma(a, x)


@def_function.function(jit_compile=True)
def _igammac(a, x):
  # [FUNCTION DOC]
  # HELPER: XLA-jit-compiled wrapper around the regularized upper incomplete gamma op. Takes
  # shape parameter a and argument x; returns the result.
  return math_ops.igammac(a, x)


@def_function.function(jit_compile=True)
def _polygamma(n, x):
  # [FUNCTION DOC]
  # HELPER: XLA-jit-compiled wrapper around the polygamma op. Takes order n and argument x;
  # returns the n-th derivative of the digamma function at x.
  return math_ops.polygamma(n, x)


@def_function.function(jit_compile=True)
def _zeta(a, q):
  # [FUNCTION DOC]
  # HELPER: XLA-jit-compiled wrapper around the Hurwitz zeta op. Takes exponent a and offset q;
  # returns the result.
  return math_ops.zeta(a, q)


# This is df/da / df/dx, where f = igamma.
def implicit_reparameterization_grad(a, x):
  # [FUNCTION DOC]
  # HELPER: reference computation of the gamma sample gradient with respect to the shape
  # parameter via implicit reparameterization (the partial derivative of igamma with respect to a
  # divided by its derivative with respect to x). Takes shape a and sample x tensors. Computes the
  # density of x as the exponential of a log-density, and returns the negated igamma gradient with
  # respect to a divided by that density. May produce NaN or infinity when numerator and
  # denominator both underflow to zero.
  log_prob = math_ops.xlogy(a - 1., x) - math_ops.lgamma(a) - x
  prob = math_ops.exp(log_prob)
  return -gen_math_ops.igamma_grad_a(a, x) / prob


@def_function.function(jit_compile=True)
def _log1p(x):
  # [FUNCTION DOC]
  # HELPER: XLA-jit-compiled wrapper around the log of one plus x op; returns the result.
  return math_ops.log1p(x)


class Log1pTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case checking the XLA log1p accuracy against numpy over ranges of magnitudes (from tiny
  # to order one, positive and negative values) for float32 and float64, with TPU tolerance
  # adjustments. Inputs are random exponentials of uniformly drawn values.

  def setUp(self):
    # [FUNCTION DOC]
    # If the vary_seed flag is set, seeds numpy's random generator from 64 bytes of operating
    # system entropy (reduced modulo a 32-bit bound) so each run uses different data. Then calls
    # the parent setUp.
    if flags.FLAGS.vary_seed:
      entropy = os.urandom(64)
      answer = int.from_bytes(entropy, 'big')
      np.random.seed(answer % (2**32 - 1))
    super(Log1pTest, self).setUp()

  def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # HELPER: returns the (relative, absolute) tolerance pair to use. On non-TPU devices returns
    # the given values unchanged. On TPU returns fixed looser values: 4e-4 and 0 for float32,
    # otherwise 1e-10 and 0.
    if self.device not in ['TPU']:
      return rtol, atol

    if dtype == np.float32:
      return 4e-4, 0.
    return 1e-10, 0.

  def _test_range(self, low, high, dtype, rtol, atol, is_negative=False):
    # [FUNCTION DOC]
    # HELPER: shared check for the log1p tests. Draws 1000 samples as the exponential of values
    # uniform in [low, high), cast to dtype, negated when is_negative is true (default false).
    # Computes the expected values with numpy log1p, runs the jit-compiled log1p inside the XLA
    # test scope, and asserts closeness using the TPU-adjusted tolerances.
    # Test values near zero.
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    x = np.exp(np.random.uniform(
        low=low, high=high, size=[NUM_SAMPLES])).astype(dtype)
    if is_negative:
      x = -x
    expected_values = np.log1p(x)
    with self.session() as sess:
      with self.test_scope():
        actual = _log1p(x)
      actual = sess.run(actual)
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-7, 0.),
                            (np.float64, 1e-15, 0.))
  def testSmallX(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Checks log1p for tiny magnitudes (exponent between -40 and -20), positive and negative.
    self._test_range(-40., -20., dtype, rtol, atol, is_negative=False)
    self._test_range(-40., -20., dtype, rtol, atol, is_negative=True)

  @parameterized.parameters((np.float32, 2e-7, 0.),
                            (np.float64, 1e-15, 0.))
  def testGreaterThanNegativeTwentyExponent(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Checks log1p for exponents between -20 and -10, positive and negative.
    self._test_range(-20., -10., dtype, rtol, atol, is_negative=False)
    self._test_range(-20., -10., dtype, rtol, atol, is_negative=True)

  @parameterized.parameters((np.float32, 2e-7, 0.),
                            (np.float64, 1e-15, 0.))
  def testGreaterThanNegativeTenExponent(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Checks log1p for exponents between -10 and -5, positive and negative.
    self._test_range(-10., -5., dtype, rtol, atol, is_negative=False)
    self._test_range(-10., -5., dtype, rtol, atol, is_negative=True)

  @parameterized.parameters((np.float32, 2e-7, 0.),
                            (np.float64, 1e-15, 0.))
  def testGreaterThanNegativeFiveExponent(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Checks log1p for exponents between -5 and -1, positive and negative.
    self._test_range(-5., -1., dtype, rtol, atol, is_negative=False)
    self._test_range(-5., -1., dtype, rtol, atol, is_negative=True)

  @parameterized.parameters((np.float32, 4e-7, 0.),
                            (np.float64, 3e-14, 0.))
  def testXGreaterThanOneTenth(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Checks log1p for exponents between -1 and 0 (magnitudes above roughly one third),
    # positive and negative.
    self._test_range(-1., 0., dtype, rtol, atol, is_negative=False)
    self._test_range(-1., 0., dtype, rtol, atol, is_negative=True)

  @parameterized.parameters((np.float32, 2e-7, 0.),
                            (np.float64, 2e-15, 0.))
  def testXGreaterThanOne(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Checks log1p for exponents between 0 and 3 (magnitudes at least one), positive values only.
    self._test_range(0., 3., dtype, rtol, atol, is_negative=False)


class ZetaTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the Hurwitz zeta op under XLA: special-case behavior at poles and invalid
  # arguments, and accuracy against scipy over small, medium and large argument ranges, with TPU
  # tolerance adjustments and float64 skips.

  def setUp(self):
    # [FUNCTION DOC]
    # If the vary_seed flag is set, seeds numpy's random generator from operating system entropy
    # so each run uses different data. Then calls the parent setUp.
    if flags.FLAGS.vary_seed:
      entropy = os.urandom(64)
      answer = int.from_bytes(entropy, 'big')
      np.random.seed(answer % (2**32 - 1))
    super(ZetaTest, self).setUp()

  def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # HELPER: returns the (relative, absolute) tolerance pair to use. Unchanged off TPU. On TPU
    # returns 2e-2 and 1e-7 for float32, otherwise 2e-4 and 1e-20.
    if self.device not in ['TPU']:
      return rtol, atol

    if dtype == np.float32:
      return 2e-2, 1e-7
    return 2e-4, 1e-20

  def testBadValues(self):
    # [FUNCTION DOC]
    # Checks special cases of zeta on random positive offsets: exponent exactly 1 gives infinity;
    # exponent below 1 gives NaN; non-positive offsets with non-integer exponent give NaN; poles
    # at non-positive integer offsets give infinity for even integer exponents and NaN for odd
    # integer exponents; non-positive offsets with non-integer exponent give NaN.
    q = np.random.uniform(low=0.3, high=20., size=[10])
    with self.session() as sess:
      with self.test_scope():
        y = _zeta(np.float64(1.), q)
      actual = sess.run(y)
    # When x == 1, this is the Harmonic series.
    self.assertTrue(np.all(np.isinf(actual)))

    with self.session() as sess:
      with self.test_scope():
        y = _zeta(np.float64(0.1), q)
      actual = sess.run(y)
    # When x < 1, this is undefined.
    self.assertTrue(np.all(np.isnan(actual)))

    with self.session() as sess:
      with self.test_scope():
        y = _zeta([1.1, 1.2, 2.1, 2.2, 3.1], [-2.0, -1.1, -1.0, -0.5, -0.1])
      actual = sess.run(y)
    # For q <= 0, x must be an integer.
    self.assertTrue(np.all(np.isnan(actual)))

    with self.session() as sess:
      with self.test_scope():
        y = _zeta([2.0, 4.0, 6.0], [0.0, -1.0, -2.0])
      actual = sess.run(y)
    # For integer q <= 0, zeta has poles with a defined limit of +inf where x is
    # an even integer.
    self.assertTrue(np.all(np.isinf(actual)))

    with self.session() as sess:
      with self.test_scope():
        y = _zeta([3.0, 5.0, 7.0], [0.0, -1.0, -2.0])
      actual = sess.run(y)
    # For non-positive integer q, zeta has poles with an undefined limit where x
    # is an odd integer.
    self.assertTrue(np.all(np.isnan(actual)))

    with self.session() as sess:
      with self.test_scope():
        y = _zeta([1.1, 2.2, 3.3], [-1.1, -1.0, 0.0])
      actual = sess.run(y)
    # For non-positive q, zeta is not defined if x is not an integer.
    self.assertTrue(np.all(np.isnan(actual)))

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testLargeXSmallQ(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares zeta with scipy for exponents in [100, 200) and offsets in [0.3, 1). Skipped for
    # float64 on devices other than XLA CPU and GPU because of numerical instability on TPU.
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    if self.device not in ['XLA_GPU', 'XLA_CPU'] and dtype == np.float64:
      # TODO(b/165739664): Figure out why on TPU F64 Zeta sometimes returns
      # infs.
      self.skipTest(
          'Skipping test because some F64 operations are numerically '
          'unstable on TPU.')

    x = np.random.uniform(low=100., high=200., size=[NUM_SAMPLES]).astype(dtype)
    q = np.random.uniform(low=0.3, high=1., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.zeta(x, q)
    with self.session() as sess:
      with self.test_scope():
        y = _zeta(x, q)
      actual = sess.run(y)

    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testSmallValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares zeta with scipy for exponents in [1.1, 10) and tiny positive offsets up to one.
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.
    x = np.random.uniform(low=1.1, high=10., size=[NUM_SAMPLES]).astype(dtype)
    q = np.random.uniform(
        low=np.finfo(dtype).tiny, high=1., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.zeta(x, q)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_zeta(x, q))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testMediumValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares zeta with scipy for exponents in [1.1, 100) and offsets in [1, 10).
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    x = np.random.uniform(low=1.1, high=100., size=[NUM_SAMPLES]).astype(dtype)
    q = np.random.uniform(low=1., high=1e1, size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.zeta(x, q)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_zeta(x, q))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 2e-2, 1e-5), (np.float64, 1e-4, 1e-30))
  def testLargeValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares zeta with scipy for exponents in [100, 1000) and offsets in [1, 10). Uses the given
    # tolerances directly (no TPU adjustment).
    x = np.random.uniform(
        low=100., high=int(1e3), size=[NUM_SAMPLES]).astype(dtype)
    q = np.random.uniform(
        low=1., high=int(1e1), size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.zeta(x, q)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_zeta(x, q))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)


class PolygammaTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the polygamma op under XLA: invalid-order behavior, agreement with digamma for
  # order zero, and accuracy against scipy for small and medium orders, with TPU tolerance
  # adjustments.

  def setUp(self):
    # [FUNCTION DOC]
    # If the vary_seed flag is set, seeds numpy's random generator from operating system entropy
    # so each run uses different data. Then calls the parent setUp.
    if flags.FLAGS.vary_seed:
      entropy = os.urandom(64)
      answer = int.from_bytes(entropy, 'big')
      np.random.seed(answer % (2**32 - 1))
    super(PolygammaTest, self).setUp()

  def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # HELPER: returns the (relative, absolute) tolerance pair to use. Unchanged off TPU. On TPU
    # returns 2e-2 and 1e-7 for float32, otherwise 2e-4 and 1e-20.
    if self.device not in ['TPU']:
      return rtol, atol

    if dtype == np.float32:
      return 2e-2, 1e-7
    return 2e-4, 1e-20

  def testBadValues(self):
    # [FUNCTION DOC]
    # Checks polygamma on random positive arguments returns NaN for a negative order and for a
    # non-integer order.
    x = np.random.uniform(low=0.3, high=20., size=[10])
    with self.session() as sess:
      with self.test_scope():
        y = _polygamma(np.float64(-1.), x)
      actual = sess.run(y)
    # Not defined for negative numbers.
    self.assertTrue(np.all(np.isnan(actual)))

    with self.session() as sess:
      with self.test_scope():
        y = _polygamma(np.float64(0.1), x)
      actual = sess.run(y)
    # Not defined for non-integers.
    self.assertTrue(np.all(np.isnan(actual)))

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testRecoverDigamma(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Order zero polygamma on values in [0.1, 50) should equal scipy digamma. Skipped for float64
    # on devices other than XLA CPU and GPU.
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    if self.device not in ['XLA_GPU', 'XLA_CPU'] and dtype == np.float64:
      self.skipTest(
          'Skipping test because some F64 operations are '
          'numerically unstable on TPU.'
      )

    x = np.random.uniform(low=0.1, high=50., size=[NUM_SAMPLES]).astype(dtype)
    expected_values = sps.digamma(x)
    with self.session() as sess:
      with self.test_scope():
        y = _polygamma(dtype(0.), x)
      actual = sess.run(y)

    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testSmallN(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares polygamma with scipy for orders 1 to 4 and tiny positive arguments up to one.
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.
    n = np.random.randint(low=1, high=5, size=[NUM_SAMPLES]).astype(dtype)
    x = np.random.uniform(
        low=np.finfo(dtype).tiny, high=1., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.polygamma(n, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_polygamma(n, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testMediumLargeN(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares polygamma with scipy for orders 5 to 9 and arguments in [1, 10).
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    n = np.random.randint(low=5, high=10, size=[NUM_SAMPLES]).astype(dtype)
    x = np.random.uniform(low=1., high=1e1, size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.polygamma(n, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_polygamma(n, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)


class IgammaTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the regularized lower incomplete gamma op under XLA: accuracy against scipy
  # over small, medium and large ranges, numerical gradient checks with respect to the shape
  # parameter, and agreement of the gamma sample gradient op with an implicit reparameterization
  # reference. Includes TPU tolerance adjustments and float64 skips.

  def setUp(self):
    # [FUNCTION DOC]
    # If the vary_seed flag is set, seeds numpy's random generator from operating system entropy
    # so each run uses different data. Then calls the parent setUp.
    if flags.FLAGS.vary_seed:
      entropy = os.urandom(64)
      answer = int.from_bytes(entropy, 'big')
      np.random.seed(answer % (2**32 - 1))
    super(IgammaTest, self).setUp()

  # Skip Float64 test on TPU due to missing ops.
  def maybe_skip_test(self, dtype):
    # [FUNCTION DOC]
    # HELPER: skips the current test when dtype is float64 and the device is not XLA CPU or GPU
    # (float64 ops are missing on TPU). Otherwise does nothing.
    if self.device not in ['XLA_GPU', 'XLA_CPU'] and dtype == np.float64:
      self.skipTest(
          'Skipping test because some F64 operations not supported on TPU.')

  def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # HELPER: returns the (relative, absolute) tolerance pair to use. Unchanged off TPU. On TPU
    # returns 2e-2 and 1e-7 for float32, otherwise 2e-4 and 1e-20.
    if self.device not in ['TPU']:
      return rtol, atol

    if dtype == np.float32:
      return 2e-2, 1e-7
    return 2e-4, 1e-20

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testLargeXSmallA(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares igamma with scipy for large x in [100, 200) and small a in [0.3, 1).
    self.maybe_skip_test(dtype)
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.
    x = np.random.uniform(low=100., high=200., size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(low=0.3, high=1., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammainc(a, x)
    with self.session() as sess:
      with self.test_scope():
        y = _igamma(a, x)
      actual = sess.run(y)
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testSmallValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares igamma with scipy for both x and a tiny positive values up to one.
    self.maybe_skip_test(dtype)
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.
    x = np.random.uniform(
        low=np.finfo(dtype).tiny, high=1., size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(
        low=np.finfo(dtype).tiny, high=1., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammainc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_igamma(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testMediumValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares igamma with scipy for x and a both in [1, 100). (A comment in the code says "near
    # zero", which does not match the ranges used.)
    self.maybe_skip_test(dtype)
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.
    x = np.random.uniform(low=1., high=100., size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(low=1., high=100., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammainc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_igamma(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 2e-2, 1e-5), (np.float64, 1e-4, 1e-30))
  def testLargeValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares igamma with scipy for x and a both in [100, 10000). Skipped on TPU as numerically
    # unstable. Does not call the float64 skip helper or adjust tolerances. (A comment in the code
    # says "near zero", which does not match the ranges used.)
    if self.device == 'TPU':
      # TODO(b/154908275): Remove this once fixed for large a, x.
      self.skipTest('Skipping test since numerically unstable on TPU.')
    # Test values near zero.
    x = np.random.uniform(
        low=100., high=int(1e4), size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(
        low=100., high=int(1e4), size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammainc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_igamma(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  # We don't check small values because the numerical gradients become quite
  # large.
  @parameterized.parameters((np.float32, 0.09), (np.float64, 1e-7))
  def testGradMediumValues(self, dtype, tolerance):
    # [FUNCTION DOC]
    # Builds constants x and a (1000 values each in [1, 100)) and numerically checks the gradient
    # of igamma with respect to a (x fixed) using a finite-difference step of 1e-3. Asserts the
    # maximum error between theoretical and numerical Jacobians is at most the tolerance.
    # Skipped for float64 where unsupported. Small values are not checked as gradients are large.
    self.maybe_skip_test(dtype)
    with self.session():
      with self.test_scope():
        x = constant_op.constant(
            np.random.uniform(low=1., high=100.,
                              size=[NUM_SAMPLES]).astype(dtype))
        a = constant_op.constant(
            np.random.uniform(low=1., high=100.,
                              size=[NUM_SAMPLES]).astype(dtype))

        f = lambda b: _igamma(b, x)
        max_error = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(f, x=[a], delta=1e-3))
    self.assertLessEqual(max_error, tolerance)

  @parameterized.parameters((np.float32, 0.5), (np.float64, 1e-7))
  def testGradLargeValues(self, dtype, tolerance):
    # [FUNCTION DOC]
    # Same as the medium-value gradient test but with x and a in [100, 10000) and a
    # finite-difference step of 1e-2; asserts the maximum gradient error is within the tolerance.
    self.maybe_skip_test(dtype)
    with self.session():
      with self.test_scope():
        x = constant_op.constant(
            np.random.uniform(low=100., high=int(1e4),
                              size=[NUM_SAMPLES]).astype(dtype))
        a = constant_op.constant(
            np.random.uniform(low=100., high=int(1e4),
                              size=[NUM_SAMPLES]).astype(dtype))

        f = lambda b: _igamma(b, x)
        max_error = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(f, x=[a], delta=1e-2))
    self.assertLessEqual(max_error, tolerance)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testRandomGammaGradSmallValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # With tiny positive x and a (up to one), compares the random-gamma gradient op with the
    # implicit reparameterization reference. Entries where the reference is NaN or infinite
    # are dropped from both arrays before the closeness assertion.
    self.maybe_skip_test(dtype)
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.

    with self.session() as sess:
      with self.test_scope():
        x = constant_op.constant(
            np.random.uniform(
                low=np.finfo(dtype).tiny, high=1.,
                size=[NUM_SAMPLES]).astype(dtype))
        a = constant_op.constant(
            np.random.uniform(
                low=np.finfo(dtype).tiny, high=1.,
                size=[NUM_SAMPLES]).astype(dtype))
        gamma_sample_grad = gen_random_ops.random_gamma_grad(a, x)
        actual_grad = implicit_reparameterization_grad(a, x)
        gamma_sample_grad, actual_grad = sess.run(
            [gamma_sample_grad, actual_grad])
        # We do this because the ratio computed in
        # implicit_reparameterization_grad can very easily result in a NaN due
        # to the computed numerator and denominator zeroing out.
        gamma_sample_grad = gamma_sample_grad[
            ~np.logical_or(np.isnan(actual_grad), np.isinf(actual_grad))]
        actual_grad = actual_grad[
            ~np.logical_or(np.isnan(actual_grad), np.isinf(actual_grad))]
    self.assertAllClose(actual_grad, gamma_sample_grad, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testRandomGammaGradMediumValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Same comparison as the small-value gamma gradient test but with x and a in [1, 10).
    self.maybe_skip_test(dtype)
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)

    with self.session() as sess:
      with self.test_scope():
        x = constant_op.constant(
            np.random.uniform(low=1., high=10.,
                              size=[NUM_SAMPLES]).astype(dtype))
        a = constant_op.constant(
            np.random.uniform(low=1., high=10.,
                              size=[NUM_SAMPLES]).astype(dtype))
        gamma_sample_grad = gen_random_ops.random_gamma_grad(a, x)
        actual_grad = implicit_reparameterization_grad(a, x)
        gamma_sample_grad, actual_grad = sess.run(
            [gamma_sample_grad, actual_grad])
        # We do this because the ratio computed in
        # implicit_reparameterization_grad can very easily result in a NaN due
        # to the computed numerator and denominator zeroing out.
        gamma_sample_grad = gamma_sample_grad[
            ~np.logical_or(np.isnan(actual_grad), np.isinf(actual_grad))]
        actual_grad = actual_grad[
            ~np.logical_or(np.isnan(actual_grad), np.isinf(actual_grad))]
    self.assertAllClose(actual_grad, gamma_sample_grad, atol=atol, rtol=rtol)


class IgammacTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the regularized upper incomplete gamma op under XLA: accuracy against scipy
  # over small, medium and large ranges, with TPU tolerance adjustments and float64 skips.

  def setUp(self):
    # [FUNCTION DOC]
    # If the vary_seed flag is set, seeds numpy's random generator from operating system entropy
    # so each run uses different data. Then calls the parent setUp.
    if flags.FLAGS.vary_seed:
      entropy = os.urandom(64)
      answer = int.from_bytes(entropy, 'big')
      np.random.seed(answer % (2**32 - 1))
    super(IgammacTest, self).setUp()

  # Skip Float64 test on TPU due to missing ops.
  def maybe_skip_test(self, dtype):
    # [FUNCTION DOC]
    # HELPER: skips the current test when dtype is float64 and the device is not XLA CPU or GPU
    # (float64 ops are missing on TPU). Otherwise does nothing.
    if self.device not in ['XLA_GPU', 'XLA_CPU'] and dtype == np.float64:
      # TODO(b/154908275): Remove this once fixed for large a, x.
      self.skipTest(
          'Skipping test because some F64 operations not supported on TPU.')

  def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # HELPER: returns the (relative, absolute) tolerance pair to use. Unchanged off TPU. On TPU
    # returns 2e-2 and 1e-7 for float32, otherwise 2e-4 and 1e-20.
    if self.device not in ['TPU']:
      return rtol, atol

    if dtype == np.float32:
      return 2e-2, 1e-7
    return 2e-4, 1e-20

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testLargeXSmallA(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares igammac with scipy for large x in [100, 200) and small a in [0.3, 1).
    self.maybe_skip_test(dtype)
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.
    x = np.random.uniform(low=100., high=200., size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(low=0.3, high=1., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammaincc(a, x)
    with self.session() as sess:
      with self.test_scope():
        y = _igammac(a, x)
      actual = sess.run(y)
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testSmallValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares igammac with scipy for both x and a tiny positive values up to one.
    self.maybe_skip_test(dtype)
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.
    x = np.random.uniform(
        low=np.finfo(dtype).tiny, high=1., size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(
        low=np.finfo(dtype).tiny, high=1., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammaincc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_igammac(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testMediumValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares igammac with scipy for x and a both in [1, 100). (A comment in the code says "near
    # zero", which does not match the ranges used.)
    self.maybe_skip_test(dtype)
    rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
    # Test values near zero.
    x = np.random.uniform(low=1., high=100., size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(low=1., high=100., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammaincc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_igammac(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 2e-2, 1e-5), (np.float64, 1e-4, 1e-30))
  def testLargeValues(self, dtype, rtol, atol):
    # [FUNCTION DOC]
    # Compares igammac with scipy for x and a both in [100, 10000). Skipped on TPU as numerically
    # unstable. Does not call the float64 skip helper or adjust tolerances. (A comment in the code
    # says "near zero", which does not match the ranges used.)
    if self.device == 'TPU':
      self.skipTest('Skipping test since numerically unstable on TPU.')
    # Test values near zero.
    x = np.random.uniform(
        low=100., high=int(1e4), size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(
        low=100., high=int(1e4), size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammaincc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(_igammac(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'
  test.main()
