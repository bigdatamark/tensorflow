# [MODULE DOC]
# KNOWN ISSUE: the module docstring says "Tests for initializers", but this file tests the
# Exponential distribution, not initializers.
# Unit tests for the Exponential distribution and its softplus-rate variant from the distributions
# ops package. Checks log-density, density, CDF, log survival function, mean, variance, entropy,
# sampling statistics, reparameterized gradients and the softplus rate transform, comparing with
# scipy statistics when scipy is installed. Has no import-time side effects beyond an optional
# scipy import (a warning is logged if it is missing). Runs via the test main entry point.
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
"""Tests for initializers."""

import importlib

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import exponential as exponential_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader used to import scipy for the comparison tests below.
  # Takes a module name string, imports it dynamically and returns the module object. If the
  # import fails, logs a warning through the TensorFlow logger and returns the null value instead
  # of raising. Side effect: a warning log message on failure.
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


stats = try_import("scipy.stats")


@test_util.run_all_in_graph_and_eager_modes
class ExponentialTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Exponential distribution and ExponentialWithSoftplusRate. Decorated to
  # run every test in both graph and eager modes. No shared fixtures; each test builds its own
  # distribution and compares against scipy when it is available.

  def testExponentialLogPDF(self):
    # [FUNCTION DOC]
    # Builds an Exponential with rate 2 for a batch of six and evaluates log density and density
    # at six points. Asserts both outputs have static shape of six and, if scipy exists, that
    # values match the scipy exponential log pdf and its exponential.
    batch_size = 6
    lam = constant_op.constant([2.0] * batch_size)
    lam_v = 2.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    exponential = exponential_lib.Exponential(rate=lam)

    log_pdf = exponential.log_prob(x)
    self.assertEqual(log_pdf.get_shape(), (6,))

    pdf = exponential.prob(x)
    self.assertEqual(pdf.get_shape(), (6,))

    if not stats:
      return
    expected_log_pdf = stats.expon.logpdf(x, scale=1 / lam_v)
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def testExponentialLogPDFBoundary(self):
    # [FUNCTION DOC]
    # Evaluates the log density at zero for several rates and asserts it is finite and equals the
    # natural log of each rate.
    # Check that Log PDF is finite at 0.
    rate = np.array([0.1, 0.5, 1., 2., 5., 10.], dtype=np.float32)
    exponential = exponential_lib.Exponential(rate=rate)
    log_pdf = exponential.log_prob(0.)
    self.assertAllClose(np.log(rate), self.evaluate(log_pdf))

  def testExponentialCDF(self):
    # [FUNCTION DOC]
    # Evaluates the CDF of an Exponential with rate 2 at six points. Asserts static shape of six
    # and, if scipy exists, closeness to the scipy exponential CDF.
    batch_size = 6
    lam = constant_op.constant([2.0] * batch_size)
    lam_v = 2.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    exponential = exponential_lib.Exponential(rate=lam)

    cdf = exponential.cdf(x)
    self.assertEqual(cdf.get_shape(), (6,))

    if not stats:
      return
    expected_cdf = stats.expon.cdf(x, scale=1 / lam_v)
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testExponentialLogSurvival(self):
    # [FUNCTION DOC]
    # Evaluates the log survival function for batch of seven at seven points. Asserts static shape
    # of seven and, if scipy exists, closeness to the scipy exponential log survival function.
    batch_size = 7
    lam = constant_op.constant([2.0] * batch_size)
    lam_v = 2.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0, 10.0], dtype=np.float32)

    exponential = exponential_lib.Exponential(rate=lam)

    log_survival = exponential.log_survival_function(x)
    self.assertEqual(log_survival.get_shape(), (7,))

    if not stats:
      return
    expected_log_survival = stats.expon.logsf(x, scale=1 / lam_v)
    self.assertAllClose(self.evaluate(log_survival), expected_log_survival)

  def testExponentialMean(self):
    # [FUNCTION DOC]
    # Builds Exponentials with three rates, asserts the mean has static shape of three and, if
    # scipy exists, matches the scipy mean (one over the rate).
    lam_v = np.array([1.0, 4.0, 2.5])
    exponential = exponential_lib.Exponential(rate=lam_v)
    self.assertEqual(exponential.mean().get_shape(), (3,))
    if not stats:
      return
    expected_mean = stats.expon.mean(scale=1 / lam_v)
    self.assertAllClose(self.evaluate(exponential.mean()), expected_mean)

  def testExponentialVariance(self):
    # [FUNCTION DOC]
    # Builds Exponentials with three rates, asserts the variance has static shape of three and, if
    # scipy exists, matches the scipy variance.
    lam_v = np.array([1.0, 4.0, 2.5])
    exponential = exponential_lib.Exponential(rate=lam_v)
    self.assertEqual(exponential.variance().get_shape(), (3,))
    if not stats:
      return
    expected_variance = stats.expon.var(scale=1 / lam_v)
    self.assertAllClose(
        self.evaluate(exponential.variance()), expected_variance)

  def testExponentialEntropy(self):
    # [FUNCTION DOC]
    # Builds Exponentials with three rates, asserts the entropy has static shape of three and, if
    # scipy exists, matches the scipy entropy.
    lam_v = np.array([1.0, 4.0, 2.5])
    exponential = exponential_lib.Exponential(rate=lam_v)
    self.assertEqual(exponential.entropy().get_shape(), (3,))
    if not stats:
      return
    expected_entropy = stats.expon.entropy(scale=1 / lam_v)
    self.assertAllClose(self.evaluate(exponential.entropy()), expected_entropy)

  def testExponentialSample(self):
    # [FUNCTION DOC]
    # Draws 100000 samples (tensor-valued count, fixed seed) from Exponentials with rates 3 and 4.
    # Asserts the sample shape, that no sample is negative and, if scipy exists, that a
    # Kolmogorov-Smirnov statistic against the true CDF is below 0.01 for each batch member.
    lam = constant_op.constant([3.0, 4.0])
    lam_v = [3.0, 4.0]
    n = constant_op.constant(100000)
    exponential = exponential_lib.Exponential(rate=lam)

    samples = exponential.sample(n, seed=137)
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 2))
    self.assertFalse(np.any(sample_values < 0.0))
    if not stats:
      return
    for i in range(2):
      self.assertLess(
          stats.kstest(sample_values[:, i],
                       stats.expon(scale=1.0 / lam_v[i]).cdf)[0], 0.01)

  def testExponentialSampleMultiDimensional(self):
    # [FUNCTION DOC]
    # Draws 100000 samples from an Exponential with a two-by-two batch of rates. Asserts the static
    # sample shape, non-negativity and, if scipy exists, Kolmogorov-Smirnov statistics below 0.01
    # for each batch element.
    batch_size = 2
    lam_v = [3.0, 22.0]
    lam = constant_op.constant([lam_v] * batch_size)

    exponential = exponential_lib.Exponential(rate=lam)

    n = 100000
    samples = exponential.sample(n, seed=138)
    self.assertEqual(samples.get_shape(), (n, batch_size, 2))

    sample_values = self.evaluate(samples)

    self.assertFalse(np.any(sample_values < 0.0))
    if not stats:
      return
    for i in range(2):
      self.assertLess(
          stats.kstest(sample_values[:, 0, i],
                       stats.expon(scale=1.0 / lam_v[i]).cdf)[0], 0.01)
      self.assertLess(
          stats.kstest(sample_values[:, 1, i],
                       stats.expon(scale=1.0 / lam_v[i]).cdf)[0], 0.01)

  def testFullyReparameterized(self):
    # [FUNCTION DOC]
    # Samples inside a gradient tape that watches the rate tensor and asserts the gradient of the
    # samples with respect to the rate exists (is not null), showing reparameterized sampling.
    lam = constant_op.constant([0.1, 1.0])
    with backprop.GradientTape() as tape:
      tape.watch(lam)
      exponential = exponential_lib.Exponential(rate=lam)
      samples = exponential.sample(100)
    grad_lam = tape.gradient(samples, lam)
    self.assertIsNotNone(grad_lam)

  def testExponentialWithSoftplusRate(self):
    # [FUNCTION DOC]
    # Builds ExponentialWithSoftplusRate from negative raw rates and asserts the resulting rate
    # equals the softplus of the raw values.
    lam = [-2.2, -3.4]
    exponential = exponential_lib.ExponentialWithSoftplusRate(rate=lam)
    self.assertAllClose(
        self.evaluate(nn_ops.softplus(lam)), self.evaluate(exponential.rate))


if __name__ == "__main__":
  test.main()
