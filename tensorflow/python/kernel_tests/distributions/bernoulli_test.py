# [MODULE DOC]
# Unit tests for the Bernoulli distribution in the distributions package of TensorFlow ops.
# Covers construction from probabilities or logits, argument validation, static and dynamic
# shapes, dtypes, the probability mass function and its broadcasting, boundary conditions,
# entropy, sampling (statistics, seeding, non-reparameterization), mean, variance and standard
# deviation, and the registered Bernoulli-to-Bernoulli KL divergence.
# Defines small module-level helpers (optional scipy import, random distribution factory and a
# numpy entropy reference) and a single test class; running the file as a script runs the tests.
# Import-time side effect: tries to import scipy.special and logs a warning if it is missing.
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
"""Tests for the Bernoulli distribution."""

import importlib

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader used to get scipy.special for the tests in this module.
  # Imports the module whose name is given and returns it. If the import fails with an import
  # error, logs a warning naming the module and the error text and returns nothing (null), so
  # callers can skip scipy-based checks. Takes one argument: the dotted module name as a string.
  # No other exceptions are handled; no state is modified besides logging.
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")


def make_bernoulli(batch_shape, dtype=dtypes.int32):
  # [FUNCTION DOC]
  # HELPER: factory producing random Bernoulli distributions for the shape and dtype tests.
  # Takes batch_shape (any sequence of dimension sizes) and an optional dtype for samples
  # (default 32-bit integer). Draws uniform random probabilities of that shape with numpy,
  # converts them to a 32-bit float constant tensor and returns a Bernoulli distribution built
  # from those probabilities with the requested sample dtype. Uses global numpy randomness.
  p = np.random.uniform(size=list(batch_shape))
  p = constant_op.constant(p, dtype=dtypes.float32)
  return bernoulli.Bernoulli(probs=p, dtype=dtype)


def entropy(p):
  # [FUNCTION DOC]
  # HELPER: numpy reference implementation of Bernoulli entropy used as the expected value.
  # Takes a probability p (scalar or array) and returns the elementwise entropy in nats, computed
  # as minus the sum of each outcome probability times its log. Pure; no side effects. Gives
  # not-a-number or infinite results at p equal to 0 or 1 (log of zero), so only use interior values.
  q = 1. - p
  return -q * np.log(q) - p * np.log(p)


class BernoulliTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Bernoulli distribution class. Each test builds distributions from
  # probabilities or logits (often with numpy-computed expectations and optionally scipy.special
  # for logit/expit conversions) and checks properties, shapes, dtypes, pmf values, entropy,
  # sampling and KL divergence. Many tests run in both graph and eager mode; the placeholder-based
  # ones are restricted to graph mode (deprecated v1 behavior). No shared fixtures or setUp.

  @test_util.run_in_graph_and_eager_modes
  def testP(self):
    # [FUNCTION DOC]
    # Scenario: a Bernoulli distribution is constructed from probabilities 0.2 and 0.4.
    # Expected: evaluating the probs property returns approximately the same two values.
    # Runs in graph and eager modes.
    p = [0.2, 0.4]
    dist = bernoulli.Bernoulli(probs=p)
    self.assertAllClose(p, self.evaluate(dist.probs))

  @test_util.run_in_graph_and_eager_modes
  def testLogits(self):
    # [FUNCTION DOC]
    # Scenario: a distribution is built from logits of minus 42 and plus 42; then another from
    # probabilities 0.01, 0.99 and 0.42.
    # Expected: the logits property returns the supplied logits; if scipy is available the probs
    # equal the logistic sigmoid of the logits and the logits of the second distribution equal the
    # logit function of the probabilities. Returns early (skipping the scipy checks) when scipy.special
    # could not be imported.
    logits = [-42., 42.]
    dist = bernoulli.Bernoulli(logits=logits)
    self.assertAllClose(logits, self.evaluate(dist.logits))

    if not special:
      return

    self.assertAllClose(special.expit(logits), self.evaluate(dist.probs))

    p = [0.01, 0.99, 0.42]
    dist = bernoulli.Bernoulli(probs=p)
    self.assertAllClose(special.logit(p), self.evaluate(dist.logits))

  @test_util.run_in_graph_and_eager_modes
  def testInvalidP(self):
    # [FUNCTION DOC]
    # Scenario: validate_args is enabled and probabilities outside the unit interval are passed.
    # Expected: values above 1 (1.01 and 2) raise an op error mentioning that probs has components
    # greater than 1; negative values (-0.01 and -3) raise an op error from the non-negativity
    # condition; the boundary and valid values 0, 0.5 and 1 (without validation) are returned
    # unchanged. Errors are raised at evaluation time, either at construction or when evaluating probs.
    invalid_ps = [1.01, 2.]
    for p in invalid_ps:
      with self.assertRaisesOpError("probs has components greater than 1"):
        dist = bernoulli.Bernoulli(probs=p, validate_args=True)
        self.evaluate(dist.probs)

    invalid_ps = [-0.01, -3.]
    for p in invalid_ps:
      with self.assertRaisesOpError("Condition x >= 0"):
        dist = bernoulli.Bernoulli(probs=p, validate_args=True)
        self.evaluate(dist.probs)

    valid_ps = [0.0, 0.5, 1.0]
    for p in valid_ps:
      dist = bernoulli.Bernoulli(probs=p)
      self.assertEqual(p, self.evaluate(dist.probs))  # Should not fail

  @test_util.run_in_graph_and_eager_modes
  def testShapes(self):
    # [FUNCTION DOC]
    # Scenario: distributions are created for batch shapes of scalar, one element and 2x3x4.
    # Expected: the static and dynamic batch shapes equal the requested shape, and the static and
    # dynamic event shapes are both empty (Bernoulli is a scalar-event distribution).
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_bernoulli(batch_shape)
      self.assertAllEqual(batch_shape, dist.batch_shape.as_list())
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], dist.event_shape.as_list())
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))

  @test_util.run_in_graph_and_eager_modes
  def testDtype(self):
    # [FUNCTION DOC]
    # Scenario: checks dtype propagation for a 32-bit integer-sample distribution and then a 64-bit one.
    # Expected: dist dtype is the requested integer type, and sample and mode have that dtype;
    # mean, variance, stddev, entropy, prob and log_prob (even with a float event such as 0.5)
    # have the dtype of the probabilities (float32). The 64-bit variant checks dtype, sample and mode only.
    dist = make_bernoulli([])
    self.assertEqual(dist.dtype, dtypes.int32)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.probs.dtype, dist.mean().dtype)
    self.assertEqual(dist.probs.dtype, dist.variance().dtype)
    self.assertEqual(dist.probs.dtype, dist.stddev().dtype)
    self.assertEqual(dist.probs.dtype, dist.entropy().dtype)
    self.assertEqual(dist.probs.dtype, dist.prob(0).dtype)
    self.assertEqual(dist.probs.dtype, dist.prob(0.5).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_prob(0).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_prob(0.5).dtype)

    dist64 = make_bernoulli([], dtypes.int64)
    self.assertEqual(dist64.dtype, dtypes.int64)
    self.assertEqual(dist64.dtype, dist64.sample(5).dtype)
    self.assertEqual(dist64.dtype, dist64.mode().dtype)

  @test_util.run_in_graph_and_eager_modes
  def _testPmf(self, **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared body for the pmf tests, invoked by testPmfWithP with probability or logit kwargs.
    # Not collected as a test itself because its name starts with an underscore; it carries the
    # graph-and-eager decorator so it runs in both modes when called. Takes keyword arguments forwarded to the Bernoulli constructor and
    # builds the distribution (expected to describe a 2x2 batch with success probabilities 0.2, 0.4,
    # 0.3, 0.6). For several event inputs of different ranks (scalar, vector, matrix) it compares
    # prob against the hard-coded expected pmf tables, and log_prob against their natural logs,
    # checking broadcasting of events against the batch. Returns nothing.
    dist = bernoulli.Bernoulli(**kwargs)
    # pylint: disable=bad-continuation
    xs = [
        0,
        [1],
        [1, 0],
        [[1, 0]],
        [[1, 0], [1, 1]],
    ]
    expected_pmfs = [
        [[0.8, 0.6], [0.7, 0.4]],
        [[0.2, 0.4], [0.3, 0.6]],
        [[0.2, 0.6], [0.3, 0.4]],
        [[0.2, 0.6], [0.3, 0.4]],
        [[0.2, 0.6], [0.3, 0.6]],
    ]
    # pylint: enable=bad-continuation

    for x, expected_pmf in zip(xs, expected_pmfs):
      self.assertAllClose(self.evaluate(dist.prob(x)), expected_pmf)
      self.assertAllClose(self.evaluate(dist.log_prob(x)), np.log(expected_pmf))

  @test_util.run_deprecated_v1
  def testPmfCorrectBroadcastDynamicShape(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): probabilities come from a float placeholder with unknown shape, and
    # events of rank 1 and rank 2 are used.
    # Expected: feeding probabilities 0.2, 0.3, 0.4 yields pmf 0.2, 0.7, 0.4 for the vector event
    # 1,0,1 and the same values wrapped in an extra leading dimension for the nested event.
    with self.cached_session():
      p = array_ops.placeholder(dtype=dtypes.float32)
      dist = bernoulli.Bernoulli(probs=p)
      event1 = [1, 0, 1]
      event2 = [[1, 0, 1]]
      self.assertAllClose(
          dist.prob(event1).eval({
              p: [0.2, 0.3, 0.4]
          }), [0.2, 0.7, 0.4])
      self.assertAllClose(
          dist.prob(event2).eval({
              p: [0.2, 0.3, 0.4]
          }), [[0.2, 0.7, 0.4]])

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def testPmfInvalid(self):
    # [FUNCTION DOC]
    # Scenario: validate_args is on with probabilities 0.1, 0.2, 0.7; events contain an invalid
    # negative value or a value above 1.
    # Expected: evaluating prob raises an op error saying the values must be non-negative for the
    # negative event, and an error saying elements cannot exceed 1 for the event containing 2.
    # Deprecated-v1 mode (graph) is forced by the second decorator.
    p = [0.1, 0.2, 0.7]
    dist = bernoulli.Bernoulli(probs=p, validate_args=True)
    with self.assertRaisesOpError("must be non-negative."):
      self.evaluate(dist.prob([1, 1, -1]))
    with self.assertRaisesOpError("Elements cannot exceed 1."):
      self.evaluate(dist.prob([2, 0, 1]))

  @test_util.run_in_graph_and_eager_modes
  def testPmfWithP(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 matrix of probabilities, then the same expressed as logits through the scipy logit
    # function.
    # Expected: the shared pmf checker passes for both parameterizations. The logit variant is skipped
    # (early return) if scipy is unavailable.
    p = [[0.2, 0.4], [0.3, 0.6]]
    self._testPmf(probs=p)
    if not special:
      return
    self._testPmf(logits=special.logit(p))

  @test_util.run_in_graph_and_eager_modes
  def testPmfWithFloatArgReturnsXEntropy(self):
    # [FUNCTION DOC]
    # Scenario: probabilities of shape 4x1 and non-binary float events 0, 0.1 and 0.8 with
    # validation turned off.
    # Expected: log_prob returns the cross-entropy style expression: event times log p plus
    # (1 minus event) times log (1 minus p), computed in float32, broadcast to 4x3.
    p = [[0.2], [0.4], [0.3], [0.6]]
    samps = [0, 0.1, 0.8]
    self.assertAllClose(
        np.float32(samps) * np.log(np.float32(p)) +
        (1 - np.float32(samps)) * np.log(1 - np.float32(p)),
        self.evaluate(
            bernoulli.Bernoulli(probs=p, validate_args=False).log_prob(samps)))

  @test_util.run_deprecated_v1
  def testBroadcasting(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a placeholder of unknown shape supplies probability, and scalar or vector
    # events are used, or a vector of probabilities with a scalar event.
    # Expected: log_prob values equal log 0.5 (for the scalar case) or a vector of log 0.5 (for the
    # broadcast cases), confirming that scalar probability broadcasts over vector events and vice versa.
    with self.cached_session():
      p = array_ops.placeholder(dtypes.float32)
      dist = bernoulli.Bernoulli(probs=p)
      self.assertAllClose(np.log(0.5), dist.log_prob(1).eval({p: 0.5}))
      self.assertAllClose(
          np.log([0.5, 0.5, 0.5]), dist.log_prob([1, 1, 1]).eval({
              p: 0.5
          }))
      self.assertAllClose(
          np.log([0.5, 0.5, 0.5]), dist.log_prob(1).eval({
              p: [0.5, 0.5, 0.5]
          }))

  @test_util.run_deprecated_v1
  def testPmfShapes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a few shape assertions compare with the integer one in parentheses, which is just
    # the integer 1 and not a one-element tuple; it passes only because shapes compare equal to a
    # dimension sized list, so the check is weaker than it reads.
    # Scenario (graph mode): a placeholder of shape unknown-by-1 is fed 2x1 probabilities and the
    # rank of the log_prob result is checked; then a scalar probability 0.5 is used with events of rank
    # 0, 1 and 2; then a 2x1 batch of probabilities is used with a scalar event.
    # Expected: result ranks and static shapes follow broadcasting rules: rank 2 for the placeholder
    # case, empty, one-element and 2x1 shapes for the scalar-probability cases, and 2x1 for the batch.
    with self.cached_session():
      p = array_ops.placeholder(dtypes.float32, shape=[None, 1])
      dist = bernoulli.Bernoulli(probs=p)
      self.assertEqual(2, len(dist.log_prob(1).eval({p: [[0.5], [0.5]]}).shape))

      dist = bernoulli.Bernoulli(probs=0.5)
      self.assertEqual(2, len(self.evaluate(dist.log_prob([[1], [1]])).shape))

      dist = bernoulli.Bernoulli(probs=0.5)
      self.assertEqual((), dist.log_prob(1).get_shape())
      self.assertEqual((1), dist.log_prob([1]).get_shape())
      self.assertEqual((2, 1), dist.log_prob([[1], [1]]).get_shape())

      dist = bernoulli.Bernoulli(probs=[[0.5], [0.5]])
      self.assertEqual((2, 1), dist.log_prob(1).get_shape())

  @test_util.run_in_graph_and_eager_modes
  def testBoundaryConditions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test asserts not-a-number for log_prob of event 1 when the probability is 1,
    # although the mathematically correct value is zero; it records an implementation quirk (the
    # cross-entropy form evaluates zero times log of zero), so the expectation looks wrong.
    # Scenario: a Bernoulli with probability exactly 1.
    # Expected: log_prob of event 0 evaluates to not-a-number, and log_prob of event 1 also evaluates
    # to not-a-number.
    dist = bernoulli.Bernoulli(probs=1.0)
    self.assertAllClose(np.nan, self.evaluate(dist.log_prob(0)))
    self.assertAllClose([np.nan], [self.evaluate(dist.log_prob(1))])

  @test_util.run_in_graph_and_eager_modes
  def testEntropyNoBatch(self):
    # [FUNCTION DOC]
    # Scenario: a scalar probability of 0.2.
    # Expected: the evaluated entropy equals the numpy reference entropy for 0.2.
    p = 0.2
    dist = bernoulli.Bernoulli(probs=p)
    self.assertAllClose(self.evaluate(dist.entropy()), entropy(p))

  @test_util.run_in_graph_and_eager_modes
  def testEntropyWithBatch(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 matrix of probabilities with argument validation off.
    # Expected: entropy has the same 2x2 layout and each element equals the numpy reference entropy of
    # the corresponding probability.
    p = [[0.1, 0.7], [0.2, 0.6]]
    dist = bernoulli.Bernoulli(probs=p, validate_args=False)
    self.assertAllClose(
        self.evaluate(dist.entropy()),
        [[entropy(0.1), entropy(0.7)], [entropy(0.2),
                                        entropy(0.6)]])

  @test_util.run_in_graph_and_eager_modes
  def testSampleN(self):
    # [FUNCTION DOC]
    # Scenario: 100000 samples are drawn from a distribution with probabilities 0.2 and 0.6, then a
    # distribution from the log of two probabilities is sampled once with a seed.
    # Expected: samples are 32-bit integers with values only 0 and 1 (both appear), the per-column
    # sample means are within 0.01 of the probabilities, and the seeded single sample has shape 1x2.
    # The last part guards against a mismatched-type crash (bug reference in a comment).
    p = [0.2, 0.6]
    dist = bernoulli.Bernoulli(probs=p)
    n = 100000
    samples = dist.sample(n)
    samples.set_shape([n, 2])
    self.assertEqual(samples.dtype, dtypes.int32)
    sample_values = self.evaluate(samples)
    self.assertTrue(np.all(sample_values >= 0))
    self.assertTrue(np.all(sample_values <= 1))
    # Note that the standard error for the sample mean is ~ sqrt(p * (1 - p) /
    # n). This means that the tolerance is very sensitive to the value of p
    # as well as n.
    self.assertAllClose(p, np.mean(sample_values, axis=0), atol=1e-2)
    self.assertEqual(set([0, 1]), set(sample_values.flatten()))
    # In this test we're just interested in verifying there isn't a crash
    # owing to mismatched types. b/30940152
    dist = bernoulli.Bernoulli(np.log([.2, .4]))
    self.assertAllEqual((1, 2), dist.sample(1, seed=42).get_shape().as_list())

  @test_util.run_in_graph_and_eager_modes
  def testNotReparameterized(self):
    # [FUNCTION DOC]
    # Scenario: probabilities are watched on a gradient tape while 100 samples are drawn.
    # Expected: the gradient of the samples with respect to the probabilities is nothing/null,
    # because Bernoulli sampling is not reparameterized.
    p = constant_op.constant([0.2, 0.6])
    with backprop.GradientTape() as tape:
      tape.watch(p)
      dist = bernoulli.Bernoulli(probs=p)
      samples = dist.sample(100)
    grad_p = tape.gradient(samples, p)
    self.assertIsNone(grad_p)

  @test_util.run_deprecated_v1
  def testSampleActsLikeSampleN(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): the same seed is used for two sample calls, first with a Python count and
    # then with a placeholder count fed with 1000.
    # Expected: two draws with identical seed and count are equal, both when the count is static and
    # when it is dynamic (fed through a session), showing deterministic seeding.
    with self.cached_session() as sess:
      p = [0.2, 0.6]
      dist = bernoulli.Bernoulli(probs=p)
      n = 1000
      seed = 42
      self.assertAllEqual(
          self.evaluate(dist.sample(n, seed)),
          self.evaluate(dist.sample(n, seed)))
      n = array_ops.placeholder(dtypes.int32)
      sample1, sample2 = sess.run([dist.sample(n, seed), dist.sample(n, seed)],
                                  feed_dict={n: 1000})
      self.assertAllEqual(sample1, sample2)

  @test_util.run_in_graph_and_eager_modes
  def testMean(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 float32 probability matrix.
    # Expected: the mean of the distribution equals the probability matrix exactly.
    p = np.array([[0.2, 0.7], [0.5, 0.4]], dtype=np.float32)
    dist = bernoulli.Bernoulli(probs=p)
    self.assertAllEqual(self.evaluate(dist.mean()), p)

  @test_util.run_in_graph_and_eager_modes
  def testVarianceAndStd(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 matrix of probabilities (0.2, 0.7, 0.5, 0.4).
    # Expected: variance equals p times (1 minus p) elementwise, and the standard deviation equals
    # its square root, both as float32 arrays with the same 2x2 layout.
    var = lambda p: p * (1. - p)
    p = [[0.2, 0.7], [0.5, 0.4]]
    dist = bernoulli.Bernoulli(probs=p)
    self.assertAllClose(
        self.evaluate(dist.variance()),
        np.array([[var(0.2), var(0.7)], [var(0.5), var(0.4)]],
                 dtype=np.float32))
    self.assertAllClose(
        self.evaluate(dist.stddev()),
        np.array([[np.sqrt(var(0.2)), np.sqrt(var(0.7))],
                  [np.sqrt(var(0.5)), np.sqrt(var(0.4))]],
                 dtype=np.float32))

  @test_util.run_in_graph_and_eager_modes
  def testBernoulliBernoulliKL(self):
    # [FUNCTION DOC]
    # Scenario: two batches of six Bernoulli distributions with probability 0.5 and 0.4 respectively.
    # Action: calls the registered KL divergence function between the distributions.
    # Expected: the result has shape of the batch (six) and equals p log(p/q) plus (1-p) log((1-p)/(1-q))
    # computed with numpy.
    batch_size = 6
    a_p = np.array([0.5] * batch_size, dtype=np.float32)
    b_p = np.array([0.4] * batch_size, dtype=np.float32)

    a = bernoulli.Bernoulli(probs=a_p)
    b = bernoulli.Bernoulli(probs=b_p)

    kl = kullback_leibler.kl_divergence(a, b)
    kl_val = self.evaluate(kl)

    kl_expected = (a_p * np.log(a_p / b_p) + (1. - a_p) * np.log(
        (1. - a_p) / (1. - b_p)))

    self.assertEqual(kl.get_shape(), (batch_size,))
    self.assertAllClose(kl_val, kl_expected)


if __name__ == "__main__":
  test.main()
