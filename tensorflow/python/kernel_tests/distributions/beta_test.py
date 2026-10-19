# [MODULE DOC]
# Unit tests for the Beta distribution (and its softplus-concentration variant) in the
# distributions package of TensorFlow ops. Covers static and dynamic shapes with broadcasting,
# the concentration properties, argument validation of the density, density values and
# broadcasting, mean, variance, mode (including NaN handling), entropy, sampling (statistics,
# multidimensional, seeded repeatability, full reparameterization), cdf and log-cdf, the softplus
# variant, and the registered Beta-to-Beta KL divergence.
# Defines a helper for optional scipy imports and one test class; scipy-based comparisons are
# skipped when scipy is missing. Import-time side effect: imports scipy.special and scipy.stats
# if available, logging a warning for each that is missing. Running as a script runs the tests.
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
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import beta as beta_lib
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader used to obtain scipy.special and scipy.stats for the tests.
  # Imports the module named by the single string argument. On an import error, logs a warning with
  # the module name and error text and returns nothing (null); otherwise returns the module.
  # No other side effects.
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")
stats = try_import("scipy.stats")


@test_util.run_all_in_graph_and_eager_modes
class BetaTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Beta distribution, decorated so every test method runs in both graph and
  # eager modes. No shared fixtures. Tests build Beta distributions from numpy or Python
  # concentration parameters, evaluate results with the test evaluation helper, and compare against
  # hand-computed expectations or scipy.stats reference values when scipy is available.

  def testSimpleShapes(self):
    # [FUNCTION DOC]
    # Scenario: two random vectors of length 3 as concentration1 and concentration0.
    # Expected: the event shape is empty and the batch shape is three, both as a dynamic tensor and
    # as a static shape.
    a = np.random.rand(3)
    b = np.random.rand(3)
    dist = beta_lib.Beta(a, b)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tensor_shape.TensorShape([]), dist.event_shape)
    self.assertEqual(tensor_shape.TensorShape([3]), dist.batch_shape)

  def testComplexShapes(self):
    # [FUNCTION DOC]
    # Scenario: both concentrations are random arrays of shape 3x2x2.
    # Expected: the event shape is empty and the batch shape is 3x2x2, statically and dynamically.
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(3, 2, 2)
    dist = beta_lib.Beta(a, b)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tensor_shape.TensorShape([]), dist.event_shape)
    self.assertEqual(tensor_shape.TensorShape([3, 2, 2]), dist.batch_shape)

  def testComplexShapesBroadcast(self):
    # [FUNCTION DOC]
    # Scenario: concentration1 has shape 3x2x2 and concentration0 has shape 2x2.
    # Expected: the parameters broadcast, so the batch shape is 3x2x2 and the event shape is empty.
    a = np.random.rand(3, 2, 2)
    b = np.random.rand(2, 2)
    dist = beta_lib.Beta(a, b)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tensor_shape.TensorShape([]), dist.event_shape)
    self.assertEqual(tensor_shape.TensorShape([3, 2, 2]), dist.batch_shape)

  def testAlphaProperty(self):
    # [FUNCTION DOC]
    # Scenario: concentration1 given as a 1x3 nested list.
    # Expected: the concentration1 property has static shape 1x3 and evaluates to the supplied values.
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = beta_lib.Beta(a, b)
    self.assertEqual([1, 3], dist.concentration1.get_shape())
    self.assertAllClose(a, self.evaluate(dist.concentration1))

  def testBetaProperty(self):
    # [FUNCTION DOC]
    # Scenario: concentration0 given as a 1x3 nested list.
    # Expected: the concentration0 property has static shape 1x3 and evaluates to the supplied values.
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = beta_lib.Beta(a, b)
    self.assertEqual([1, 3], dist.concentration0.get_shape())
    self.assertAllClose(b, self.evaluate(dist.concentration0))

  def testPdfXProper(self):
    # [FUNCTION DOC]
    # Scenario: validate_args is on; the density is evaluated at valid points inside the open unit
    # interval and then at invalid ones.
    # Expected: valid points evaluate without error; points that are zero or negative raise an op
    # error saying the sample must be positive; points equal to or above one raise an op error saying
    # the sample must be less than one.
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    dist = beta_lib.Beta(a, b, validate_args=True)
    self.evaluate(dist.prob([.1, .3, .6]))
    self.evaluate(dist.prob([.2, .3, .5]))
    # Either condition can trigger.
    with self.assertRaisesOpError("sample must be positive"):
      self.evaluate(dist.prob([-1., 0.1, 0.5]))
    with self.assertRaisesOpError("sample must be positive"):
      self.evaluate(dist.prob([0., 0.1, 0.5]))
    with self.assertRaisesOpError("sample must be less than `1`"):
      self.evaluate(dist.prob([.1, .2, 1.2]))
    with self.assertRaisesOpError("sample must be less than `1`"):
      self.evaluate(dist.prob([.1, .2, 1.0]))

  def testPdfTwoBatches(self):
    # [FUNCTION DOC]
    # Scenario: two batch members with parameters (1,1) and (2,2) evaluated at 0.5.
    # Expected: densities 1 and 1.5 within a small tolerance, with static result shape of two.
    a = [1., 2]
    b = [1., 2]
    x = [.5, .5]
    dist = beta_lib.Beta(a, b)
    pdf = dist.prob(x)
    self.assertAllClose([1., 3. / 2], self.evaluate(pdf), rtol=1e-5, atol=1e-5)
    self.assertEqual((2,), pdf.get_shape())

  def testPdfTwoBatchesNontrivialX(self):
    # [FUNCTION DOC]
    # Scenario: batch parameters (1,1) and (2,2) evaluated at 0.3 and 0.7 respectively.
    # Expected: densities 1 and 63/50 within a small tolerance, with static result shape of two.
    a = [1., 2]
    b = [1., 2]
    x = [.3, .7]
    dist = beta_lib.Beta(a, b)
    pdf = dist.prob(x)
    self.assertAllClose([1, 63. / 50], self.evaluate(pdf), rtol=1e-5, atol=1e-5)
    self.assertEqual((2,), pdf.get_shape())

  def testPdfUniformZeroBatch(self):
    # [FUNCTION DOC]
    # Scenario: scalar parameters 1 and 1, which make the Beta uniform, and five points in the unit
    # interval.
    # Expected: all five densities equal 1 and the static result shape is five.
    # This is equivalent to a uniform distribution
    a = 1.
    b = 1.
    x = np.array([.1, .2, .3, .5, .8], dtype=np.float32)
    dist = beta_lib.Beta(a, b)
    pdf = dist.prob(x)
    self.assertAllClose([1.] * 5, self.evaluate(pdf))
    self.assertEqual((5,), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenSameRank(self):
    # [FUNCTION DOC]
    # Scenario: parameters with shape 1x2 and event values with shape 2x2 (same rank).
    # Expected: the parameters broadcast over the leading dimension; result values match the
    # hand-computed densities and the static shape is 2x2.
    a = [[1., 2]]
    b = [[1., 2]]
    x = [[.5, .5], [.3, .7]]
    dist = beta_lib.Beta(a, b)
    pdf = dist.prob(x)
    self.assertAllClose([[1., 3. / 2], [1., 63. / 50]],
                        self.evaluate(pdf),
                        rtol=1e-5,
                        atol=1e-5)
    self.assertEqual((2, 2), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenLowerRank(self):
    # [FUNCTION DOC]
    # Scenario: parameters with shape 2 and events with shape 2x2 (lower-rank parameters).
    # Expected: parameters broadcast against the events; values match hand-computed densities and
    # the static shape is 2x2.
    a = [1., 2]
    b = [1., 2]
    x = [[.5, .5], [.2, .8]]
    pdf = beta_lib.Beta(a, b).prob(x)
    self.assertAllClose([[1., 3. / 2], [1., 24. / 25]],
                        self.evaluate(pdf),
                        rtol=1e-5,
                        atol=1e-5)
    self.assertEqual((2, 2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    # [FUNCTION DOC]
    # Scenario: parameters with shape 2x2 and a single event row of shape 1x2.
    # Expected: the event broadcasts over the batch rows; densities match expectations and the static
    # shape is 2x2.
    a = [[1., 2], [2., 3]]
    b = [[1., 2], [2., 3]]
    x = [[.5, .5]]
    pdf = beta_lib.Beta(a, b).prob(x)
    self.assertAllClose([[1., 3. / 2], [3. / 2, 15. / 8]],
                        self.evaluate(pdf),
                        rtol=1e-5,
                        atol=1e-5)
    self.assertEqual((2, 2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    # [FUNCTION DOC]
    # Scenario: parameters with shape 2x2 and an event vector of shape 2.
    # Expected: the event broadcasts over the batch; densities match expectations and static shape
    # is 2x2.
    a = [[1., 2], [2., 3]]
    b = [[1., 2], [2., 3]]
    x = [.5, .5]
    pdf = beta_lib.Beta(a, b).prob(x)
    self.assertAllClose([[1., 3. / 2], [3. / 2, 15. / 8]],
                        self.evaluate(pdf),
                        rtol=1e-5,
                        atol=1e-5)
    self.assertEqual((2, 2), pdf.get_shape())

  def testLogPdfOnBoundaryIsFiniteWhenAlphaIsOne(self):
    # [FUNCTION DOC]
    # Scenario: concentration1 equal to one with a variety of concentration0 values, evaluated at the
    # boundary point zero.
    # Expected: all densities are finite (no infinity or not-a-number), since the density at zero is
    # well defined when the first concentration is one.
    b = [[0.01, 0.1, 1., 2], [5., 10., 2., 3]]
    pdf = self.evaluate(beta_lib.Beta(1., b).prob(0.))
    self.assertAllEqual(np.ones_like(pdf, dtype=np.bool_), np.isfinite(pdf))

  def testBetaMean(self):
    # [FUNCTION DOC]
    # Scenario: concentrations of length three.
    # Expected: the mean has static shape three; if scipy is available it matches the scipy mean.
    # Returns early without the value check when scipy is missing.
    a = [1., 2, 3]
    b = [2., 4, 1.2]
    dist = beta_lib.Beta(a, b)
    self.assertEqual(dist.mean().get_shape(), (3,))
    if not stats:
      return
    expected_mean = stats.beta.mean(a, b)
    self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

  def testBetaVariance(self):
    # [FUNCTION DOC]
    # Scenario: concentrations of length three.
    # Expected: the variance has static shape three; if scipy is available it matches the scipy
    # variance. Returns early without the value check when scipy is missing.
    a = [1., 2, 3]
    b = [2., 4, 1.2]
    dist = beta_lib.Beta(a, b)
    self.assertEqual(dist.variance().get_shape(), (3,))
    if not stats:
      return
    expected_variance = stats.beta.var(a, b)
    self.assertAllClose(expected_variance, self.evaluate(dist.variance()))

  def testBetaMode(self):
    # [FUNCTION DOC]
    # Scenario: concentrations both above one for all batch members (first alpha 1.1).
    # Expected: the mode has static shape three and equals (a minus 1) divided by (a plus b minus 2).
    a = np.array([1.1, 2, 3])
    b = np.array([2., 4, 1.2])
    expected_mode = (a - 1) / (a + b - 2)
    dist = beta_lib.Beta(a, b)
    self.assertEqual(dist.mode().get_shape(), (3,))
    self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

  def testBetaModeInvalid(self):
    # [FUNCTION DOC]
    # Scenario: allow_nan_stats is off, and the mode is requested for parameters where a concentration
    # is not above one (first alpha equal to one in the first case, first beta equal to one in the second).
    # Expected: each evaluation raises an op error from a less-than condition check, because the mode
    # is undefined for such parameters.
    a = np.array([1., 2, 3])
    b = np.array([2., 4, 1.2])
    dist = beta_lib.Beta(a, b, allow_nan_stats=False)
    with self.assertRaisesOpError("Condition x < y.*"):
      self.evaluate(dist.mode())

    a = np.array([2., 2, 3])
    b = np.array([1., 4, 1.2])
    dist = beta_lib.Beta(a, b, allow_nan_stats=False)
    with self.assertRaisesOpError("Condition x < y.*"):
      self.evaluate(dist.mode())

  def testBetaModeEnableAllowNanStats(self):
    # [FUNCTION DOC]
    # Scenario: the same invalid parameter sets as the invalid-mode test but with allow_nan_stats on.
    # Expected: no error; the mode has static shape three and equals the formula for the valid entries
    # with the first entry replaced by not-a-number.
    a = np.array([1., 2, 3])
    b = np.array([2., 4, 1.2])
    dist = beta_lib.Beta(a, b, allow_nan_stats=True)

    expected_mode = (a - 1) / (a + b - 2)
    expected_mode[0] = np.nan
    self.assertEqual((3,), dist.mode().get_shape())
    self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

    a = np.array([2., 2, 3])
    b = np.array([1., 4, 1.2])
    dist = beta_lib.Beta(a, b, allow_nan_stats=True)

    expected_mode = (a - 1) / (a + b - 2)
    expected_mode[0] = np.nan
    self.assertEqual((3,), dist.mode().get_shape())
    self.assertAllClose(expected_mode, self.evaluate(dist.mode()))

  def testBetaEntropy(self):
    # [FUNCTION DOC]
    # Scenario: concentrations of length three.
    # Expected: the entropy has static shape three; if scipy is available it matches the scipy entropy.
    # Returns early without the value check when scipy is missing.
    a = [1., 2, 3]
    b = [2., 4, 1.2]
    dist = beta_lib.Beta(a, b)
    self.assertEqual(dist.entropy().get_shape(), (3,))
    if not stats:
      return
    expected_entropy = stats.beta.entropy(a, b)
    self.assertAllClose(expected_entropy, self.evaluate(dist.entropy()))

  def testBetaSample(self):
    # [FUNCTION DOC]
    # Scenario: Beta with parameters 1 and 2; 100000 samples requested using a constant tensor count.
    # Expected: the sample array has shape 100000, no sample is negative; if scipy is available a
    # Kolmogorov-Smirnov statistic against the Beta cdf is under 0.01, the sample mean is within 0.01
    # of the true mean, and the sample covariance is within 0.1 of the true variance.
    a = 1.
    b = 2.
    beta = beta_lib.Beta(a, b)
    n = constant_op.constant(100000)
    samples = beta.sample(n)
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000,))
    self.assertFalse(np.any(sample_values < 0.0))
    if not stats:
      return
    self.assertLess(
        stats.kstest(
            # Beta is a univariate distribution.
            sample_values,
            stats.beta(a=1., b=2.).cdf)[0],
        0.01)
    # The standard error of the sample mean is 1 / (sqrt(18 * n))
    self.assertAllClose(
        sample_values.mean(axis=0), stats.beta.mean(a, b), atol=1e-2)
    self.assertAllClose(
        np.cov(sample_values, rowvar=0), stats.beta.var(a, b), atol=1e-1)

  def testBetaFullyReparameterized(self):
    # [FUNCTION DOC]
    # Scenario: both concentrations are constant tensors watched on a gradient tape while 100 samples
    # are drawn.
    # Expected: the gradients of the samples with respect to both concentrations are not null, showing
    # that Beta sampling is fully reparameterized.
    a = constant_op.constant(1.0)
    b = constant_op.constant(2.0)
    with backprop.GradientTape() as tape:
      tape.watch(a)
      tape.watch(b)
      beta = beta_lib.Beta(a, b)
      samples = beta.sample(100)
    grad_a, grad_b = tape.gradient(samples, [a, b])
    self.assertIsNotNone(grad_a)
    self.assertIsNotNone(grad_b)

  # Test that sampling with the same seed twice gives the same results.
  def testBetaSampleMultipleTimes(self):
    # [FUNCTION DOC]
    # Scenario: the global random seed is set to the same value before creating each of two Beta
    # distributions (differently named) and sampling 100 values with the same op seed.
    # Expected: both sample arrays are equal, i.e. sampling is repeatable under identical seeds.
    # Side effect: sets the global TensorFlow random seed twice.
    a_val = 1.
    b_val = 2.
    n_val = 100

    random_seed.set_random_seed(654321)
    beta1 = beta_lib.Beta(
        concentration1=a_val, concentration0=b_val, name="beta1")
    samples1 = self.evaluate(beta1.sample(n_val, seed=123456))

    random_seed.set_random_seed(654321)
    beta2 = beta_lib.Beta(
        concentration1=a_val, concentration0=b_val, name="beta2")
    samples2 = self.evaluate(beta2.sample(n_val, seed=123456))

    self.assertAllClose(samples1, samples2)

  def testBetaSampleMultidimensional(self):
    # [FUNCTION DOC]
    # Scenario: random float32 concentrations of shape 3x2x2 and 100000 samples.
    # Expected: sample shape is 100000x3x2x2, no sample is negative; if scipy is available the mean of
    # one slice of the batch is within 0.1 of the scipy mean.
    a = np.random.rand(3, 2, 2).astype(np.float32)
    b = np.random.rand(3, 2, 2).astype(np.float32)
    beta = beta_lib.Beta(a, b)
    n = constant_op.constant(100000)
    samples = beta.sample(n)
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 3, 2, 2))
    self.assertFalse(np.any(sample_values < 0.0))
    if not stats:
      return
    self.assertAllClose(
        sample_values[:, 1, :].mean(axis=0),
        stats.beta.mean(a, b)[1, :],
        atol=1e-1)

  def testBetaCdf(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the early return when scipy is missing is inside the loop over dtypes, so it also
    # skips the second dtype; the 0 <= x and 1 >= x assertions are trivially true statements about the
    # random inputs rather than checks of the cdf.
    # Scenario: random parameters and points of shape 30x40x50 for float32 and float64.
    # Action: evaluates the cdf of the Beta at the random points.
    # Expected: if scipy is available, the result matches the scipy cdf within loose tolerances.
    shape = (30, 40, 50)
    for dt in (np.float32, np.float64):
      a = 10. * np.random.random(shape).astype(dt)
      b = 10. * np.random.random(shape).astype(dt)
      x = np.random.random(shape).astype(dt)
      actual = self.evaluate(beta_lib.Beta(a, b).cdf(x))
      self.assertAllEqual(np.ones(shape, dtype=np.bool_), 0. <= x)
      self.assertAllEqual(np.ones(shape, dtype=np.bool_), 1. >= x)
      if not stats:
        return
      self.assertAllClose(stats.beta.cdf(x, a, b), actual, rtol=9e-3, atol=5e-6)

  def testBetaLogCdf(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the early return when scipy is missing is inside the loop over dtypes, so it also
    # skips the second dtype; the 0 <= x and 1 >= x assertions are trivially true statements about the
    # random inputs rather than checks of the log-cdf.
    # Scenario: random parameters and points of shape 30x40x50 for float32 and float64.
    # Action: evaluates the exponential of the log-cdf.
    # Expected: if scipy is available, the result matches the scipy cdf within loose tolerances.
    shape = (30, 40, 50)
    for dt in (np.float32, np.float64):
      a = 10. * np.random.random(shape).astype(dt)
      b = 10. * np.random.random(shape).astype(dt)
      x = np.random.random(shape).astype(dt)
      actual = self.evaluate(math_ops.exp(beta_lib.Beta(a, b).log_cdf(x)))
      self.assertAllEqual(np.ones(shape, dtype=np.bool_), 0. <= x)
      self.assertAllEqual(np.ones(shape, dtype=np.bool_), 1. >= x)
      if not stats:
        return
      self.assertAllClose(stats.beta.cdf(x, a, b), actual, rtol=3e-3, atol=2e-5)

  def testBetaWithSoftplusConcentration(self):
    # [FUNCTION DOC]
    # Scenario: negative raw values minus 4.2 and minus 9.1 passed to the softplus-concentration
    # variant.
    # Expected: its concentration1 and concentration0 equal the softplus of the respective inputs.
    a, b = -4.2, -9.1
    dist = beta_lib.BetaWithSoftplusConcentration(a, b)
    self.assertAllClose(
        self.evaluate(nn_ops.softplus(a)), self.evaluate(dist.concentration1))
    self.assertAllClose(
        self.evaluate(nn_ops.softplus(b)), self.evaluate(dist.concentration0))

  def testBetaBetaKL(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the early return when scipy is missing sits inside the loop over shapes and after
    # the distributions are built, so the KL comparisons are skipped in that case; the final
    # KL(d1||d1) check is also unreachable then.
    # Scenario: for each of two batch shapes, random parameters for two Beta distributions, each also
    # built in the softplus-concentration form using the inverse softplus of the parameters.
    # Action: computes the registered KL divergence for all four pairings.
    # Expected: each result has the batch shape and matches the closed-form KL from scipy special
    # functions; KL of a distribution with itself is zero.
    for shape in [(10,), (4, 5)]:
      a1 = 6.0 * np.random.random(size=shape) + 1e-4
      b1 = 6.0 * np.random.random(size=shape) + 1e-4
      a2 = 6.0 * np.random.random(size=shape) + 1e-4
      b2 = 6.0 * np.random.random(size=shape) + 1e-4
      # Take inverse softplus of values to test BetaWithSoftplusConcentration
      a1_sp = np.log(np.exp(a1) - 1.0)
      b1_sp = np.log(np.exp(b1) - 1.0)
      a2_sp = np.log(np.exp(a2) - 1.0)
      b2_sp = np.log(np.exp(b2) - 1.0)

      d1 = beta_lib.Beta(concentration1=a1, concentration0=b1)
      d2 = beta_lib.Beta(concentration1=a2, concentration0=b2)
      d1_sp = beta_lib.BetaWithSoftplusConcentration(concentration1=a1_sp,
                                                     concentration0=b1_sp)
      d2_sp = beta_lib.BetaWithSoftplusConcentration(concentration1=a2_sp,
                                                     concentration0=b2_sp)

      if not special:
        return
      kl_expected = (special.betaln(a2, b2) - special.betaln(a1, b1) +
                     (a1 - a2) * special.digamma(a1) +
                     (b1 - b2) * special.digamma(b1) +
                     (a2 - a1 + b2 - b1) * special.digamma(a1 + b1))

      for dist1 in [d1, d1_sp]:
        for dist2 in [d2, d2_sp]:
          kl = kullback_leibler.kl_divergence(dist1, dist2)
          kl_val = self.evaluate(kl)
          self.assertEqual(kl.get_shape(), shape)
          self.assertAllClose(kl_val, kl_expected)

      # Make sure KL(d1||d1) is 0
      kl_same = self.evaluate(kullback_leibler.kl_divergence(d1, d1))
      self.assertAllClose(kl_same, np.zeros_like(kl_expected))


if __name__ == "__main__":
  test.main()
