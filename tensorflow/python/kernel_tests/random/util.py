# [MODULE DOC]
# HELPER: shared statistical test utilities for the random-ops kernel tests.
# Provides functions to z-test sample moments against analytic moments, a Pearson chi-squared
# statistic on uniform samples, a standard normal CDF, an Anderson-Darling statistic, and a
# checker for the mean, median and variance of truncated normal samples.
# Depends on numpy and the special-math helpers of the distributions package. No import-time
# side effects.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for testing random variables."""

import math

import numpy as np

from tensorflow.python.ops.distributions import special_math


def test_moment_matching(
    samples,
    number_moments,
    dist,
    stride=0):
  # [FUNCTION DOC]
  # HELPER: statistical check used by random-op tests to validate sample distributions.
  # Inputs: samples (numpy array of draws; two-dimensional arrays are flattened before
  # striding), number_moments (how many moments, starting at the first, to check), dist (a
  # distribution object offering a moment method) and stride (spacing between samples used
  # for the i-th moment; zero means all samples, larger values spread the moments out).
  # For each moment computes the sample mean of the i-th powers, the analytic moment, and the
  # variance of the estimator (plus a small allowance for floating-point error, floored at the
  # smallest normal number). Returns a list with one absolute z-score (array) per moment.
  # Raises an assertion error if any total variance is not positive.
  """Return z-test scores for sample moments to match analytic moments.

  Given `samples`, check that the first sample `number_moments` match
  the given  `dist` moments by doing a z-test.

  Args:
    samples: Samples from target distribution.
    number_moments: Python `int` describing how many sample moments to check.
    dist: SciPy distribution object that provides analytic moments.
    stride: Distance between samples to check for statistical properties.
      A stride of 0 means to use all samples, while other strides test for
      spatial correlation.
  Returns:
    Array of z_test scores.
  """

  sample_moments = []
  expected_moments = []
  variance_sample_moments = []
  for i in range(1, number_moments + 1):
    if len(samples.shape) == 2:
      strided_range = samples.flat[::(i - 1) * stride + 1]
    else:
      strided_range = samples[::(i - 1) * stride + 1, ...]
    sample_moments.append(np.mean(strided_range**i, axis=0))
    expected_moments.append(dist.moment(i))
    variance_sample_moments.append(
        (dist.moment(2 * i) - dist.moment(i) ** 2) / len(strided_range))

  z_test_scores = []
  for i in range(1, number_moments + 1):
    # Assume every operation has a small numerical error.
    # It takes i multiplications to calculate one i-th moment.
    total_variance = (
        variance_sample_moments[i - 1] +
        i * np.finfo(samples.dtype).eps)
    tiny = np.finfo(samples.dtype).tiny
    assert np.all(total_variance > 0)
    total_variance = np.where(total_variance < tiny, tiny, total_variance)
    # z_test is approximately a unit normal distribution.
    z_test_scores.append(abs(
        (sample_moments[i - 1] - expected_moments[i - 1]) / np.sqrt(
            total_variance)))
  return z_test_scores


def chi_squared(x, bins):
  # [FUNCTION DOC]
  # HELPER: statistic for uniformity tests of random ops.
  # Inputs: x, array of samples assumed to lie in the unit interval (values outside are ignored
  # by the histogram), and bins, the number of equal-width bins. Returns the Pearson chi-squared
  # statistic against a uniform expectation of equal counts per bin.
  """Pearson's Chi-squared test."""
  x = np.ravel(x)
  n = len(x)
  histogram, _ = np.histogram(x, bins=bins, range=(0, 1))
  expected = n / float(bins)
  return np.sum(np.square(histogram - expected) / expected)


def normal_cdf(x):
  # [FUNCTION DOC]
  # HELPER: used by the Anderson-Darling statistic.
  # Input: x, scalar or array. Returns the standard normal cumulative probability for each
  # element, computed through the error function applied elementwise.
  """Cumulative distribution function for a standard normal distribution."""
  return 0.5 + 0.5 * np.vectorize(math.erf)(x / math.sqrt(2))


def anderson_darling(x):
  # [FUNCTION DOC]
  # HELPER: goodness-of-fit statistic used by normal-sampling tests.
  # Input: x, array of samples (flattened and sorted internally). Returns the Anderson-Darling
  # statistic comparing the samples to a standard normal distribution; larger means a worse
  # fit. Samples that make the CDF reach exactly zero or one produce infinite values.
  """Anderson-Darling test for a standard normal distribution."""
  x = np.sort(np.ravel(x))
  n = len(x)
  i = np.linspace(1, n, n)
  z = np.sum((2 * i - 1) * np.log(normal_cdf(x)) +
             (2 * (n - i) + 1) * np.log(1 - normal_cdf(x)))
  return -n - z / n


def test_truncated_normal(assert_equal,
                          assert_all_close,
                          n,
                          y,
                          means=None,
                          stddevs=None,
                          minvals=None,
                          maxvals=None,
                          mean_atol=5e-4,
                          median_atol=8e-4,
                          variance_rtol=1e-3):
  # [FUNCTION DOC]
  # HELPER: shared check for truncated-normal sampling tests.
  # Inputs: assert_equal and assert_all_close (assertion callbacks from the calling test),
  # n (expected number of samples, all of which must lie inside the bounds), y (numpy array of
  # samples), optional means, stddevs, minvals and maxvals (default mean 0, standard deviation
  # 1, bounds minus two and plus two), and tolerances mean_atol, median_atol (absolute) and
  # variance_rtol (relative). Computes the analytic mean, median and variance of the truncated
  # normal and asserts the sample statistics match within tolerance. Returns nothing; failures
  # are reported by the callbacks.
  """Tests truncated normal distribution's statistics."""
  def _normal_cdf(x):
    # [FUNCTION DOC]
    # HELPER: local scalar standard normal CDF via the complementary error function.
    return .5 * math.erfc(-x / math.sqrt(2))

  def normal_pdf(x):
    # [FUNCTION DOC]
    # HELPER: local scalar standard normal density.
    return math.exp(-(x**2) / 2.) / math.sqrt(2 * math.pi)

  def probit(x):
    # [FUNCTION DOC]
    # HELPER: local inverse standard normal CDF, delegating to the special-math helper.
    return special_math.ndtri(x)

  a = -2.
  b = 2.
  mu = 0.
  sigma = 1.

  if minvals is not None:
    a = minvals

  if maxvals is not None:
    b = maxvals

  if means is not None:
    mu = means

  if stddevs is not None:
    sigma = stddevs

  alpha = (a - mu) / sigma
  beta = (b - mu) / sigma
  z = _normal_cdf(beta) - _normal_cdf(alpha)

  assert_equal((y >= a).sum(), n)
  assert_equal((y <= b).sum(), n)

  # For more information on these calculations, see:
  # Burkardt, John. "The Truncated Normal Distribution".
  # Department of Scientific Computing website. Florida State University.
  expected_mean = mu + (normal_pdf(alpha) - normal_pdf(beta)) / z * sigma
  y = y.astype(float)
  actual_mean = np.mean(y)
  assert_all_close(actual_mean, expected_mean, atol=mean_atol)

  expected_median = mu + probit(
      (_normal_cdf(alpha) + _normal_cdf(beta)) / 2.) * sigma
  actual_median = np.median(y)
  assert_all_close(actual_median, expected_median, atol=median_atol)

  expected_variance = sigma**2 * (1 + (
      (alpha * normal_pdf(alpha) - beta * normal_pdf(beta)) / z) - (
          (normal_pdf(alpha) - normal_pdf(beta)) / z)**2)
  actual_variance = np.var(y)
  assert_all_close(
      actual_variance,
      expected_variance,
      rtol=variance_rtol)
