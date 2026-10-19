# [MODULE DOC]
# Unit tests for the Gamma distribution, the softplus-parameterized GammaWithSoftplusConcentrationRate
# variant and the registered Gamma-to-Gamma KL divergence. Covers shapes, log-density and density
# (including broadcasting), CDF, mean, mode (with and without allowing NaN statistics), variance,
# standard deviation, entropy, sampling statistics, reparameterized gradients and argument
# validation. Compares with scipy when available. Import-time side effect: tries to import scipy
# special functions and statistics, logging a warning for any that are missing.
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

import importlib

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import gamma as gamma_lib
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader used to import scipy modules for the comparison tests.
  # Takes a module name string, imports it dynamically and returns the module object, or the null
  # value if the import fails (after logging a warning through the TensorFlow logger).
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")
stats = try_import("scipy.stats")


@test_util.run_all_in_graph_and_eager_modes
class GammaTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Gamma distribution family and its KL divergence, run in both graph and eager
  # modes. Provides two private helpers: a Kolmogorov-Smirnov goodness-of-fit check and a
  # numerical integral check of sampled densities. Optional scipy is used for reference values.

  def testGammaShape(self):
    # [FUNCTION DOC]
    # Builds a Gamma with a batch of five concentrations and a scalar rate. Asserts the batch
    # shape tensor and static batch shape are five and that the event shape (tensor and static)
    # is empty.
    alpha = constant_op.constant([3.0] * 5)
    beta = constant_op.constant(11.0)
    gamma = gamma_lib.Gamma(concentration=alpha, rate=beta)

    self.assertEqual(self.evaluate(gamma.batch_shape_tensor()), (5,))
    self.assertEqual(gamma.batch_shape, tensor_shape.TensorShape([5]))
    self.assertAllEqual(self.evaluate(gamma.event_shape_tensor()), [])
    self.assertEqual(gamma.event_shape, tensor_shape.TensorShape([]))

  def testGammaLogPDF(self):
    # [FUNCTION DOC]
    # Evaluates log density and density of a Gamma (concentration 2, rate 3, batch of six) at six
    # points. Asserts static shapes of six and, if scipy exists, agreement with scipy values.
    batch_size = 6
    alpha = constant_op.constant([2.0] * batch_size)
    beta = constant_op.constant([3.0] * batch_size)
    alpha_v = 2.0
    beta_v = 3.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    gamma = gamma_lib.Gamma(concentration=alpha, rate=beta)
    log_pdf = gamma.log_prob(x)
    self.assertEqual(log_pdf.get_shape(), (6,))
    pdf = gamma.prob(x)
    self.assertEqual(pdf.get_shape(), (6,))
    if not stats:
      return
    expected_log_pdf = stats.gamma.logpdf(x, alpha_v, scale=1 / beta_v)
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def testGammaLogPDFBoundary(self):
    # [FUNCTION DOC]
    # With concentration one (an exponential distribution) and several rates, asserts the log
    # density at zero is finite and equals the log of the rate.
    # When concentration = 1, we have an exponential distribution. Check that at
    # 0 we have finite log prob.
    rate = np.array([0.1, 0.5, 1., 2., 5., 10.], dtype=np.float32)
    gamma = gamma_lib.Gamma(concentration=1., rate=rate)
    log_pdf = gamma.log_prob(0.)
    self.assertAllClose(np.log(rate), self.evaluate(log_pdf))

  def testGammaLogPDFMultidimensional(self):
    # [FUNCTION DOC]
    # Evaluates log density and density for a Gamma with a six-by-two batch of parameters at a
    # column of six points (broadcast across the last axis). Asserts output shape six by two and,
    # if scipy exists, agreement with scipy.
    batch_size = 6
    alpha = constant_op.constant([[2.0, 4.0]] * batch_size)
    beta = constant_op.constant([[3.0, 4.0]] * batch_size)
    alpha_v = np.array([2.0, 4.0])
    beta_v = np.array([3.0, 4.0])
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    gamma = gamma_lib.Gamma(concentration=alpha, rate=beta)
    log_pdf = gamma.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.get_shape(), (6, 2))
    pdf = gamma.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.get_shape(), (6, 2))
    if not stats:
      return
    expected_log_pdf = stats.gamma.logpdf(x, alpha_v, scale=1 / beta_v)
    self.assertAllClose(log_pdf_values, expected_log_pdf)
    self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testGammaLogPDFMultidimensionalBroadcasting(self):
    # [FUNCTION DOC]
    # Same as the multidimensional log density test but the rate is a scalar that must broadcast
    # against a six-by-two concentration. Asserts output shape six by two and, if scipy exists,
    # agreement with scipy.
    batch_size = 6
    alpha = constant_op.constant([[2.0, 4.0]] * batch_size)
    beta = constant_op.constant(3.0)
    alpha_v = np.array([2.0, 4.0])
    beta_v = 3.0
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    gamma = gamma_lib.Gamma(concentration=alpha, rate=beta)
    log_pdf = gamma.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.get_shape(), (6, 2))
    pdf = gamma.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.get_shape(), (6, 2))

    if not stats:
      return
    expected_log_pdf = stats.gamma.logpdf(x, alpha_v, scale=1 / beta_v)
    self.assertAllClose(log_pdf_values, expected_log_pdf)
    self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testGammaCDF(self):
    # [FUNCTION DOC]
    # Evaluates the CDF of a Gamma (concentration 2, rate 3, batch of six) at six points. Asserts
    # static shape of six and, if scipy exists, agreement with the scipy CDF.
    batch_size = 6
    alpha = constant_op.constant([2.0] * batch_size)
    beta = constant_op.constant([3.0] * batch_size)
    alpha_v = 2.0
    beta_v = 3.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    gamma = gamma_lib.Gamma(concentration=alpha, rate=beta)
    cdf = gamma.cdf(x)
    self.assertEqual(cdf.get_shape(), (6,))
    if not stats:
      return
    expected_cdf = stats.gamma.cdf(x, alpha_v, scale=1 / beta_v)
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testGammaMean(self):
    # [FUNCTION DOC]
    # Builds a Gamma with three parameter pairs, asserts the mean has static shape three and, if
    # scipy exists, matches the scipy mean.
    alpha_v = np.array([1.0, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    gamma = gamma_lib.Gamma(concentration=alpha_v, rate=beta_v)
    self.assertEqual(gamma.mean().get_shape(), (3,))
    if not stats:
      return
    expected_means = stats.gamma.mean(alpha_v, scale=1 / beta_v)
    self.assertAllClose(self.evaluate(gamma.mean()), expected_means)

  def testGammaModeAllowNanStatsIsFalseWorksWhenAllBatchMembersAreDefined(self):
    # [FUNCTION DOC]
    # Builds a Gamma where every concentration exceeds one, so the mode is defined everywhere,
    # and asserts the mode has shape three and equals (concentration minus one) divided by rate.
    alpha_v = np.array([5.5, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    gamma = gamma_lib.Gamma(concentration=alpha_v, rate=beta_v)
    expected_modes = (alpha_v - 1) / beta_v
    self.assertEqual(gamma.mode().get_shape(), (3,))
    self.assertAllClose(self.evaluate(gamma.mode()), expected_modes)

  def testGammaModeAllowNanStatsFalseRaisesForUndefinedBatchMembers(self):
    # [FUNCTION DOC]
    # Builds a Gamma with allow-NaN-statistics disabled where the first concentration is below
    # one (mode undefined). Asserts that evaluating the mode raises an op error mentioning the
    # failed less-than assertion.
    # Mode will not be defined for the first entry.
    alpha_v = np.array([0.5, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    gamma = gamma_lib.Gamma(
        concentration=alpha_v, rate=beta_v, allow_nan_stats=False)
    with self.assertRaisesOpError("x < y"):
      self.evaluate(gamma.mode())

  def testGammaModeAllowNanStatsIsTrueReturnsNaNforUndefinedBatchMembers(self):
    # [FUNCTION DOC]
    # Builds a Gamma with allow-NaN-statistics enabled where the first concentration is below one.
    # Asserts the mode has shape three, is not-a-number for the first member and equals
    # (concentration minus one) divided by rate for the others.
    # Mode will not be defined for the first entry.
    alpha_v = np.array([0.5, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    gamma = gamma_lib.Gamma(
        concentration=alpha_v, rate=beta_v, allow_nan_stats=True)
    expected_modes = (alpha_v - 1) / beta_v
    expected_modes[0] = np.nan
    self.assertEqual(gamma.mode().get_shape(), (3,))
    self.assertAllClose(self.evaluate(gamma.mode()), expected_modes)

  def testGammaVariance(self):
    # [FUNCTION DOC]
    # Asserts the variance of a three-member Gamma has static shape three and, if scipy exists,
    # matches the scipy variance.
    alpha_v = np.array([1.0, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    gamma = gamma_lib.Gamma(concentration=alpha_v, rate=beta_v)
    self.assertEqual(gamma.variance().get_shape(), (3,))
    if not stats:
      return
    expected_variances = stats.gamma.var(alpha_v, scale=1 / beta_v)
    self.assertAllClose(self.evaluate(gamma.variance()), expected_variances)

  def testGammaStd(self):
    # [FUNCTION DOC]
    # Asserts the standard deviation of a three-member Gamma has static shape three and, if scipy
    # exists, matches the scipy standard deviation.
    alpha_v = np.array([1.0, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    gamma = gamma_lib.Gamma(concentration=alpha_v, rate=beta_v)
    self.assertEqual(gamma.stddev().get_shape(), (3,))
    if not stats:
      return
    expected_stddev = stats.gamma.std(alpha_v, scale=1. / beta_v)
    self.assertAllClose(self.evaluate(gamma.stddev()), expected_stddev)

  def testGammaEntropy(self):
    # [FUNCTION DOC]
    # Asserts the entropy of a three-member Gamma has static shape three and, if scipy exists,
    # matches the scipy entropy.
    alpha_v = np.array([1.0, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    gamma = gamma_lib.Gamma(concentration=alpha_v, rate=beta_v)
    self.assertEqual(gamma.entropy().get_shape(), (3,))
    if not stats:
      return
    expected_entropy = stats.gamma.entropy(alpha_v, scale=1 / beta_v)
    self.assertAllClose(self.evaluate(gamma.entropy()), expected_entropy)

  def testGammaSampleSmallAlpha(self):
    # [FUNCTION DOC]
    # Draws 100000 seeded samples from a Gamma with a very small concentration (0.05) and rate 1.
    # Asserts static and evaluated shapes, passes the Kolmogorov-Smirnov helper check and, if
    # scipy exists, that sample mean and variance are close to the theoretical values.
    alpha_v = 0.05
    beta_v = 1.0
    alpha = constant_op.constant(alpha_v)
    beta = constant_op.constant(beta_v)
    n = 100000
    gamma = gamma_lib.Gamma(concentration=alpha, rate=beta)
    samples = gamma.sample(n, seed=137)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.get_shape(), (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertTrue(self._kstest(alpha_v, beta_v, sample_values))
    if not stats:
      return
    self.assertAllClose(
        sample_values.mean(),
        stats.gamma.mean(alpha_v, scale=1 / beta_v),
        atol=.01)
    self.assertAllClose(
        sample_values.var(),
        stats.gamma.var(alpha_v, scale=1 / beta_v),
        atol=.15)

  def testGammaSample(self):
    # [FUNCTION DOC]
    # Draws 100000 seeded samples from a Gamma with concentration 4 and rate 3. Asserts shapes,
    # the Kolmogorov-Smirnov helper check and, if scipy exists, sample mean and variance close to
    # theoretical values.
    alpha_v = 4.0
    beta_v = 3.0
    alpha = constant_op.constant(alpha_v)
    beta = constant_op.constant(beta_v)
    n = 100000
    gamma = gamma_lib.Gamma(concentration=alpha, rate=beta)
    samples = gamma.sample(n, seed=137)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.get_shape(), (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertTrue(self._kstest(alpha_v, beta_v, sample_values))
    if not stats:
      return
    self.assertAllClose(
        sample_values.mean(),
        stats.gamma.mean(alpha_v, scale=1 / beta_v),
        atol=.01)
    self.assertAllClose(
        sample_values.var(),
        stats.gamma.var(alpha_v, scale=1 / beta_v),
        atol=.15)

  def testGammaFullyReparameterized(self):
    # [FUNCTION DOC]
    # Samples a Gamma inside a gradient tape watching concentration and rate, then asserts the
    # gradients of the samples with respect to both parameters exist (are not null).
    alpha = constant_op.constant(4.0)
    beta = constant_op.constant(3.0)
    with backprop.GradientTape() as tape:
      tape.watch(alpha)
      tape.watch(beta)
      gamma = gamma_lib.Gamma(concentration=alpha, rate=beta)
      samples = gamma.sample(100)
    grad_alpha, grad_beta = tape.gradient(samples, [alpha, beta])
    self.assertIsNotNone(grad_alpha)
    self.assertIsNotNone(grad_beta)

  def testGammaSampleMultiDimensional(self):
    # [FUNCTION DOC]
    # Samples 10000 draws from a Gamma whose parameters broadcast to a ten-by-one hundred batch.
    # Asserts static and evaluated shapes and, if scipy exists, that per-batch sample means and
    # variances are within tolerance of theory, and that fewer than three percent of the batch
    # cells fail the Kolmogorov-Smirnov helper check.
    alpha_v = np.array([np.arange(1, 101, dtype=np.float32)])  # 1 x 100
    beta_v = np.array([np.arange(1, 11, dtype=np.float32)]).T  # 10 x 1
    gamma = gamma_lib.Gamma(concentration=alpha_v, rate=beta_v)
    n = 10000
    samples = gamma.sample(n, seed=137)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.get_shape(), (n, 10, 100))
    self.assertEqual(sample_values.shape, (n, 10, 100))
    zeros = np.zeros_like(alpha_v + beta_v)  # 10 x 100
    alpha_bc = alpha_v + zeros
    beta_bc = beta_v + zeros
    if not stats:
      return
    self.assertAllClose(
        sample_values.mean(axis=0),
        stats.gamma.mean(alpha_bc, scale=1 / beta_bc),
        atol=0.,
        rtol=.05)
    self.assertAllClose(
        sample_values.var(axis=0),
        stats.gamma.var(alpha_bc, scale=1 / beta_bc),
        atol=10.0,
        rtol=0.)
    fails = 0
    trials = 0
    for ai, a in enumerate(np.reshape(alpha_v, [-1])):
      for bi, b in enumerate(np.reshape(beta_v, [-1])):
        s = sample_values[:, bi, ai]
        trials += 1
        fails += 0 if self._kstest(a, b, s) else 1
    self.assertLess(fails, trials * 0.03)

  def _kstest(self, alpha, beta, samples):
    # [FUNCTION DOC]
    # HELPER: goodness-of-fit check used by the Gamma sampling tests.
    # Takes a concentration, a rate and an array of samples. Returns true if the
    # Kolmogorov-Smirnov statistic against the Gamma CDF is below 0.02. If scipy is not installed
    # it returns true so the test passes vacuously.
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    if not stats:
      return True  # If we can't test, return that the test passes.
    ks, _ = stats.kstest(samples, stats.gamma(alpha, scale=1 / beta).cdf)
    # Return True when the test passes.
    return ks < 0.02

  def testGammaPdfOfSampleMultiDims(self):
    # [FUNCTION DOC]
    # Samples 50000 draws from a Gamma with batch shape two by two and evaluates their density.
    # Asserts the shapes, that the density integrates to about one over each batch cell (via the
    # integral helper) and, if scipy exists, that sample mean and variance match theory.
    gamma = gamma_lib.Gamma(concentration=[7., 11.], rate=[[5.], [6.]])
    num = 50000
    samples = gamma.sample(num, seed=137)
    pdfs = gamma.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual(samples.get_shape(), (num, 2, 2))
    self.assertEqual(pdfs.get_shape(), (num, 2, 2))
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)
    if not stats:
      return
    self.assertAllClose(
        stats.gamma.mean([[7., 11.], [7., 11.]],
                         scale=1 / np.array([[5., 5.], [6., 6.]])),
        sample_vals.mean(axis=0),
        atol=.1)
    self.assertAllClose(
        stats.gamma.var([[7., 11.], [7., 11.]],
                        scale=1 / np.array([[5., 5.], [6., 6.]])),
        sample_vals.var(axis=0),
        atol=.1)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1e-3):
    # [FUNCTION DOC]
    # HELPER: numerical-integration assertion used by the density-of-samples test.
    # Takes sampled values and their densities (equal-length sequences) and an absolute tolerance
    # (default 0.001). Sorts the pairs by sample value, integrates with the trapezoid rule and
    # asserts the total is within the tolerance of one. Fails the test otherwise; returns nothing.
    s_p = zip(sample_vals, pdf_vals)
    prev = (0, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testGammaNonPositiveInitializationParamsRaises(self):
    # [FUNCTION DOC]
    # With argument validation on, builds a Gamma with a zero concentration and then one with a
    # zero rate. Each time asserts that evaluating the mean raises an op error about the
    # positivity check.
    alpha_v = constant_op.constant(0.0, name="alpha")
    beta_v = constant_op.constant(1.0, name="beta")
    with self.assertRaisesOpError("x > 0"):
      gamma = gamma_lib.Gamma(
          concentration=alpha_v, rate=beta_v, validate_args=True)
      self.evaluate(gamma.mean())
    alpha_v = constant_op.constant(1.0, name="alpha")
    beta_v = constant_op.constant(0.0, name="beta")
    with self.assertRaisesOpError("x > 0"):
      gamma = gamma_lib.Gamma(
          concentration=alpha_v, rate=beta_v, validate_args=True)
      self.evaluate(gamma.mean())

  def testGammaWithSoftplusConcentrationRate(self):
    # [FUNCTION DOC]
    # Builds the softplus variant from raw (possibly negative) parameters and asserts the stored
    # concentration and rate equal the softplus of the raw inputs.
    alpha_v = constant_op.constant([0.0, -2.1], name="alpha")
    beta_v = constant_op.constant([1.0, -3.6], name="beta")
    gamma = gamma_lib.GammaWithSoftplusConcentrationRate(
        concentration=alpha_v, rate=beta_v)
    self.assertAllEqual(
        self.evaluate(nn_ops.softplus(alpha_v)),
        self.evaluate(gamma.concentration))
    self.assertAllEqual(
        self.evaluate(nn_ops.softplus(beta_v)), self.evaluate(gamma.rate))

  def testGammaGammaKL(self):
    # [FUNCTION DOC]
    # Computes the registered KL divergence between two Gammas and a Monte Carlo estimate from
    # 10000 seeded samples. Asserts the KL shape equals the batch shape and, if scipy special
    # functions exist, that the analytic KL matches the closed form tightly and the sample
    # estimate loosely.
    alpha0 = np.array([3.])
    beta0 = np.array([1., 2., 3., 1.5, 2.5, 3.5])

    alpha1 = np.array([0.4])
    beta1 = np.array([0.5, 1., 1.5, 2., 2.5, 3.])

    # Build graph.
    g0 = gamma_lib.Gamma(concentration=alpha0, rate=beta0)
    g1 = gamma_lib.Gamma(concentration=alpha1, rate=beta1)
    x = g0.sample(int(1e4), seed=0)
    kl_sample = math_ops.reduce_mean(g0.log_prob(x) - g1.log_prob(x), 0)
    kl_actual = kullback_leibler.kl_divergence(g0, g1)

    # Execute graph.
    [kl_sample_, kl_actual_] = self.evaluate([kl_sample, kl_actual])

    self.assertEqual(beta0.shape, kl_actual.get_shape())

    if not special:
      return
    kl_expected = ((alpha0 - alpha1) * special.digamma(alpha0)
                   + special.gammaln(alpha1)
                   - special.gammaln(alpha0)
                   + alpha1 * np.log(beta0)
                   - alpha1 * np.log(beta1)
                   + alpha0 * (beta1 / beta0 - 1.))

    self.assertAllClose(kl_expected, kl_actual_, atol=0., rtol=1e-6)
    self.assertAllClose(kl_sample_, kl_actual_, atol=0., rtol=1e-1)


if __name__ == "__main__":
  test.main()
