# [MODULE DOC]
# XLA compiler tests for random-number generating ops (uniform, normal, truncated normal,
# parameterized truncated normal, shuffle). Checks that generators are not constant, that
# samples fall inside the requested ranges, and (via a large-sample statistical check) that
# sample mean, median and variance agree with the theoretical truncated-normal values. Tests
# use large sample counts. No import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for random-number generation ops in the XLA JIT compiler."""

import math

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import special_math
from tensorflow.python.platform import googletest


class RandomOpsTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized XLA test case for random-number generating operators. Shared helpers choose the
  # dtypes to test, check generator non-constancy, and check truncated-normal range and
  # statistics. Tests draw from the XLA device's random generators, so results vary run to run.
  """Test cases for random-number generating operators."""

  def _random_types(self):
    # [FUNCTION DOC]
    # HELPER: picks the dtypes to test. Returns the set of numeric types supported by the test
    # device, excluding complex types and the 8-bit and 64-bit integer types.
    return set(self.numeric_types) - set(
        self.complex_types) - {np.uint64, np.int64, np.uint8, np.int8}

  def _testRngIsNotConstant(self, rng, dtype):
    # [FUNCTION DOC]
    # HELPER: builds a random tensor on the XLA device by calling the generator callback rng with
    # the dtype, evaluates it three times, and asserts that at least one pair of evaluations
    # differs exactly (a deterministic generator would produce identical outputs). Returns
    # nothing.
    # Tests that 'rng' does not always return the same value.
    with self.session():
      with self.test_scope():
        x = rng(dtype)

      # The random-number generator, if working correctly, should produce the
      # same output multiple times with low probability.
      y = self.evaluate(x)
      z = self.evaluate(x)
      w = self.evaluate(x)

      # We use exact equality here. If the random-number generator is producing
      # deterministic output, all three outputs will be bitwise identical.
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or (not np.array_equal(y, w)))

  def testRandomUniformIsNotConstant(self):
    # [FUNCTION DOC]
    # For every supported random dtype, checks that a two-element uniform generator (upper bound
    # the dtype's maximum) does not return identical values on repeated evaluations.

    def rng(dtype):
      # [FUNCTION DOC]
      # HELPER: generator callback; returns a 2-element uniform random tensor of the given dtype
      # (converted to a framework dtype) with maximum value the dtype's largest value.
      dtype = dtypes.as_dtype(dtype)
      return random_ops.random_uniform(shape=[2], dtype=dtype, maxval=dtype.max)

    for dtype in self._random_types():
      self._testRngIsNotConstant(rng, dtype)

  def testRandomNormalIsNotConstant(self):
    # [FUNCTION DOC]
    # For every supported random dtype that is also a float type, checks that a two-element
    # normal generator does not return identical values on repeated evaluations.

    def rng(dtype):
      # [FUNCTION DOC]
      # HELPER: generator callback; returns a 2-element standard-normal random tensor of dtype.
      return random_ops.random_normal(shape=[2], dtype=dtype)

    for dtype in self._random_types() & self.float_types:
      self._testRngIsNotConstant(rng, dtype)

  @parameterized.parameters({
      'mean': 1.4,
      'stddev': 1.2
  }, {
      'mean': 2.3,
      'stddev': 2.0
  })
  def testRandomNormal(self, mean, stddev):
    # [FUNCTION DOC]
    # Parameterized over (mean 1.4, stddev 1.2) and (mean 2.3, stddev 2.0). For each supported
    # float dtype, draws one million normal samples on the XLA device and applies the truncated
    # normal range-and-statistics check using the full dtype range as bounds (so effectively
    # checks the sample mean, median and variance of the normal distribution).
    num_elts = 1000000
    for dtype in self._random_types() & self.float_types:
      with self.session():
        with self.test_scope():
          normal = random_ops.random_normal([num_elts],
                                            dtype=dtype,
                                            mean=mean,
                                            stddev=stddev)
          self._checkTruncatedNormalIsInRange(
              normal,
              a=normal.dtype.min,
              b=normal.dtype.max,
              mu=mean,
              sigma=stddev,
              count=num_elts,
              stat_test=True)

  def testRandomUniformIsInRange(self):
    # [FUNCTION DOC]
    # For each supported random dtype (skipping 16-bit float types on XLA CPU/GPU devices because
    # of a known bug), draws 1000 uniform samples in [-2, 33) and asserts all are at least -2 and
    # below 33.
    for dtype in self._random_types():
      # TODO (b/112272078): enable bfloat16 for CPU and GPU when the bug is
      # fixed.
      if (self.device in ['XLA_GPU', 'XLA_CPU'
                         ]) and (dtype in [dtypes.bfloat16, dtypes.half]):
        continue
      with self.session():
        with self.test_scope():
          x = random_ops.random_uniform(
              shape=[1000], dtype=dtype, minval=-2, maxval=33)
        y = self.evaluate(x)
        msg = str(y) + str(dtype)
        self.assertEqual((y >= -2).sum(), 1000, msg)
        self.assertEqual((y < 33).sum(), 1000, msg)

  def testTruncatedNormalIsNotConstant(self):
    # [FUNCTION DOC]
    # For the 32- and 64-bit float dtypes supported, checks that a two-element truncated normal
    # generator does not return identical values on repeated evaluations.

    def rng(dtype):
      # [FUNCTION DOC]
      # HELPER: generator callback; returns a 2-element truncated-normal random tensor of dtype.
      return random_ops.truncated_normal(shape=[2], dtype=dtype)

    # TODO(b/34339814): make this test work with 16 bit float types.
    for dtype in self._random_types() & {np.float32, np.float64}:
      self._testRngIsNotConstant(rng, dtype)

  def _checkTruncatedNormalIsInRange(self, x, a, b, mu, sigma, count,
                                     stat_test):
    # [FUNCTION DOC]
    # HELPER: validates samples from a (truncated) normal distribution. Evaluates tensor x and
    # asserts every value lies within lower bound a and upper bound b (count values each side).
    # If stat_test is false returns there (statistics are unreliable in low-probability regions).
    # Otherwise compares the sample mean, median and variance with the theoretical values of the
    # normal distribution with location mu and scale sigma truncated to [a, b], with tolerance
    # 0.1 for bfloat16 and 0.02 otherwise. Parameters: x tensor, a, b, mu, sigma, count (expected
    # number of samples), stat_test flag. Returns nothing.

    def normal_cdf(x):
      # [FUNCTION DOC]
      # HELPER: standard normal cumulative distribution function of a number, via the
      # complementary error function. Returns a float.
      return .5 * math.erfc(-x / math.sqrt(2))

    def normal_pdf(x):
      # [FUNCTION DOC]
      # HELPER: standard normal probability density of a number. Returns a float.
      return math.exp(-(x**2) / 2.) / math.sqrt(2 * math.pi)

    def probit(x):
      # [FUNCTION DOC]
      # HELPER: inverse of the standard normal cumulative distribution; evaluates the library
      # inverse-normal op on x and returns the numeric result.
      return self.evaluate(special_math.ndtri(x))

    y = self.evaluate(x)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    z = normal_cdf(beta) - normal_cdf(alpha)

    self.assertEqual((y >= a).sum(), count)
    self.assertEqual((y <= b).sum(), count)

    # Skip statistical test for low probability regions.
    if not stat_test:
      return

    # For more information on these calculations, see:
    # Burkardt, John. "The Truncated Normal Distribution".
    # Department of Scientific Computing website. Florida State University.
    expected_mean = mu + (normal_pdf(alpha) - normal_pdf(beta)) / z * sigma
    actual_mean = np.mean(y, dtype=np.float64)
    if x.dtype == dtypes.bfloat16:
      atol = rtol = 1e-1
    else:
      atol = rtol = 2e-2
    self.assertAllClose(actual_mean, expected_mean, atol=atol, rtol=rtol)

    expected_median = mu + probit(
        (normal_cdf(alpha) + normal_cdf(beta)) / 2.) * sigma
    actual_median = np.median(y)
    self.assertAllClose(actual_median, expected_median, atol=atol, rtol=rtol)

    expected_variance = sigma**2 * (1 + (
        (alpha * normal_pdf(alpha) - beta * normal_pdf(beta)) / z) - (
            (normal_pdf(alpha) - normal_pdf(beta)) / z)**2)
    actual_variance = np.var(y, dtype=np.float64)
    self.assertAllClose(
        actual_variance, expected_variance, atol=atol, rtol=rtol)

  def testTruncatedNormalIsInRange(self):
    # [FUNCTION DOC]
    # For 32/64-bit float dtypes, draws ten million truncated-normal samples (defaults: mean 0,
    # stddev 1, truncated at plus or minus 2) and runs the range-and-statistics check.
    count = 10000000
    # TODO(b/34339814): make this test work with 16 bit float types.
    for dtype in self._random_types() & {np.float32, np.float64}:
      with self.session():
        with self.test_scope():
          x = random_ops.truncated_normal(shape=[count], dtype=dtype)
        self._checkTruncatedNormalIsInRange(
            x, a=-2, b=2, mu=0, sigma=1, count=count, stat_test=True)

  def _implParameterizedTruncatedNormalIsInRange(self, a, b, mu, sigma, count,
                                                 stat_test):
    # [FUNCTION DOC]
    # HELPER: for each 32/64-bit float dtype, draws count samples from the parameterized
    # truncated normal op with mean mu, stddev sigma and bounds a and b (bounds may be infinite),
    # then runs the range check, with the statistical check only if stat_test is true.
    # TODO(b/34339814): make this test work with 16 bit float types.
    for dtype in self._random_types() & {np.float32, np.float64}:
      with self.session():
        with self.test_scope():
          x = random_ops.parameterized_truncated_normal(
              shape=[count],
              dtype=dtype,
              means=mu,
              stddevs=sigma,
              minvals=a,
              maxvals=b)
        self._checkTruncatedNormalIsInRange(
            x, a=a, b=b, mu=mu, sigma=sigma, count=count, stat_test=stat_test)

  def testParameterizedTruncatedNormalBroadcasting(self):
    # [FUNCTION DOC]
    # Draws a 1 by ten-million tensor from the parameterized truncated normal with scalar mean 0
    # and stddev 1 and one-element bound lists (-1 and 1), exercising broadcasting of the
    # parameters, then runs the range and statistics check for 32/64-bit float dtypes.
    for dtype in self._random_types() & {np.float32, np.float64}:
      with self.session():
        with self.test_scope():
          a = -1.
          b = 1.
          mu = 0.
          sigma = 1.
          count = 10000000
          x = random_ops.parameterized_truncated_normal(
              shape=[1, count],
              dtype=dtype,
              means=mu,
              stddevs=sigma,
              minvals=[a],
              maxvals=[b])
        self._checkTruncatedNormalIsInRange(
            x, a=a, b=b, mu=mu, sigma=sigma, count=count, stat_test=True)

  def testParameterizedTruncatedNormalBatched(self):
    # [FUNCTION DOC]
    # Draws a 2 by ten-million tensor from the parameterized truncated normal with per-row means
    # 0 and 1, stddev 0.1 and wide bounds (-100, 100); checks each row separately against its
    # own mean. Only the 32-bit float dtype is tested (64-bit is a known open bug).
    # TODO(b/112289993): Make this test work with dtype np.float64.
    for dtype in self._random_types() & {np.float32}:
      with self.session():
        with self.test_scope():
          count = 10000000
          a = -100.
          b = 100.
          mu0 = 0.
          mu1 = 1.
          sigma = .1
          x = random_ops.parameterized_truncated_normal(
              shape=[2, count],
              dtype=dtype,
              means=[mu0, mu1],
              stddevs=sigma,
              minvals=[a],
              maxvals=[b])
        self._checkTruncatedNormalIsInRange(
            x[0], a=a, b=b, mu=mu0, sigma=sigma, count=count, stat_test=True)
        self._checkTruncatedNormalIsInRange(
            x[1], a=a, b=b, mu=mu1, sigma=sigma, count=count, stat_test=True)

  def testParameterizedTruncatedNormalIsInRangeCenter(self):
    # [FUNCTION DOC]
    # Parameterized truncated normal with bounds -10 to 20, mean 5, stddev 5 (central region);
    # ten million samples with the statistical check on.
    count = 10000000
    self._implParameterizedTruncatedNormalIsInRange(
        a=-10, b=20, mu=5, sigma=5, count=count, stat_test=True)

  def testParameterizedTruncatedNormalIsInRangeLeft(self):
    # [FUNCTION DOC]
    # Truncation in the far left tail of a standard normal: bounds (-10, -4) and then
    # (negative infinity, -4); ten million samples each, range check only.
    count = 10000000
    # the region is on the left side of the parent normal distribution
    self._implParameterizedTruncatedNormalIsInRange(
        a=-10, b=-4, mu=0, sigma=1, count=count, stat_test=False)
    self._implParameterizedTruncatedNormalIsInRange(
        a=-np.inf, b=-4, mu=0, sigma=1, count=count, stat_test=False)

  def testParameterizedTruncatedNormalIsInRangeRight(self):
    # [FUNCTION DOC]
    # Truncation in the far right tail of a standard normal: bounds (4, 10) and then
    # (4, positive infinity); ten million samples each, range check only.
    count = 10000000
    # the region is on the right side of the parent normal distribution
    self._implParameterizedTruncatedNormalIsInRange(
        a=4, b=10, mu=0, sigma=1, count=count, stat_test=False)
    self._implParameterizedTruncatedNormalIsInRange(
        a=4, b=np.inf, mu=0, sigma=1, count=count, stat_test=False)

  def testShuffle1d(self):
    # [FUNCTION DOC]
    # Shuffles the integer range 0 to 65535 on the XLA device and asserts the set of values is
    # unchanged (order is not checked).
    with self.session():
      with self.test_scope():
        x = math_ops.range(1 << 16)
        shuffle = random_ops.random_shuffle(x)
      result = self.evaluate(shuffle)
      expected = range(1 << 16)
      # Compare sets to avoid randomness behavior changes but make sure still
      # have all the values.
      self.assertAllEqual(set(result), set(expected))

  def testShuffle2d(self):
    # [FUNCTION DOC]
    # Shuffles rows of a 20x20 diagonal matrix on the XLA device; asserts the flattened result has
    # the same length and the same set of values as the original (rows move, contents stay).
    with self.session():
      with self.test_scope():
        x = array_ops.diag(math_ops.range(20))
        shuffle = random_ops.random_shuffle(x)
      result = self.evaluate(shuffle)
      expected = np.diag(range(20)).flatten()
      # Compare sets to avoid randomness behavior changes but make sure still
      # have all the values.
      self.assertAllEqual(len(result.flatten()), len(expected))
      self.assertAllEqual(set(result.flatten()), set(expected))

  def testRandomShuffleInputRank0(self):
    # [FUNCTION DOC]
    # Shuffling a scalar (rank 0) value on the XLA device must run without error; the result is
    # not inspected.
    with self.session():
      with self.test_scope():
        shuffle = random_ops.random_shuffle(value=1e20)
      self.evaluate(shuffle)


if __name__ == '__main__':
  googletest.main()
