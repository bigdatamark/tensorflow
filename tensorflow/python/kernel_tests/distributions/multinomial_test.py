# [MODULE DOC]
# Unit tests for the Multinomial distribution in the distributions ops package. Covers shapes,
# parameter accessors (total count, probabilities, logits), probability mass evaluation including
# underflow, count validation and broadcasting, mean, covariance (single, batched and
# multidimensional), agreement of sampled moments with analytic ones, and the fact that sampling is
# not reparameterized. Many tests are restricted to graph mode (v1 only) and use explicit session
# evaluation. No import-time side effects.
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
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import multinomial
from tensorflow.python.platform import test


class MultinomialTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Multinomial distribution. Shared fixture: a seeded numpy random state stored
  # on the instance by the set-up method, used to generate random logits in the sampling tests.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: overrides the base test set-up without calling the parent implementation, so
    # base-class per-test initialization (such as random seed resetting) is skipped.
    # Creates a numpy random state seeded with 42 and stores it on the instance as the shared
    # random generator for tests. Takes no inputs and returns nothing.
    self._rng = np.random.RandomState(42)

  @test_util.run_v1_only("b/120545219")
  def testSimpleShapes(self):
    # [FUNCTION DOC]
    # Builds a Multinomial with a single probability vector of length three. Asserts the event
    # shape tensor is three, the batch shape tensor is empty, and static event and batch shapes
    # are (3) and scalar. Graph mode only.
    with self.cached_session():
      p = [.1, .3, .6]
      dist = multinomial.Multinomial(total_count=1., probs=p)
      self.assertEqual(3, dist.event_shape_tensor().eval())
      self.assertAllEqual([], dist.batch_shape_tensor())
      self.assertEqual(tensor_shape.TensorShape([3]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([]), dist.batch_shape)

  @test_util.run_v1_only("b/120545219")
  def testComplexShapes(self):
    # [FUNCTION DOC]
    # Builds a Multinomial with a three-by-two batch of total counts and probabilities of shape
    # three by two by two. Asserts the event shape is two and the batch shape is three by two, both
    # as tensors and as static shapes. Graph mode only.
    with self.cached_session():
      p = 0.5 * np.ones([3, 2, 2], dtype=np.float32)
      n = [[3., 2], [4, 5], [6, 7]]
      dist = multinomial.Multinomial(total_count=n, probs=p)
      self.assertEqual(2, dist.event_shape_tensor().eval())
      self.assertAllEqual([3, 2], dist.batch_shape_tensor())
      self.assertEqual(tensor_shape.TensorShape([2]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([3, 2]), dist.batch_shape)

  @test_util.run_v1_only("b/120545219")
  def testN(self):
    # [FUNCTION DOC]
    # Builds a Multinomial with a column of total counts and a two-row probability matrix. Asserts
    # the total count keeps static shape two by one and equals the given counts. Graph mode only.
    p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
    n = [[3.], [4]]
    with self.cached_session():
      dist = multinomial.Multinomial(total_count=n, probs=p)
      self.assertEqual((2, 1), dist.total_count.get_shape())
      self.assertAllClose(n, dist.total_count)

  @test_util.run_v1_only("b/120545219")
  def testP(self):
    # [FUNCTION DOC]
    # Builds a Multinomial from a one-row probability matrix. Asserts probs and logits both have
    # static shape one by three and probs equals the input. Graph mode only.
    p = [[0.1, 0.2, 0.7]]
    with self.cached_session():
      dist = multinomial.Multinomial(total_count=3., probs=p)
      self.assertEqual((1, 3), dist.probs.get_shape())
      self.assertEqual((1, 3), dist.logits.get_shape())
      self.assertAllClose(p, dist.probs)

  @test_util.run_v1_only("b/120545219")
  def testLogits(self):
    # [FUNCTION DOC]
    # Builds a Multinomial from logits equal to log of probabilities shifted by a constant (minus
    # fifty). Asserts probs and logits have static shape one by three, probs recover the original
    # probabilities (softmax is shift invariant) and logits equal the input. Graph mode only.
    p = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    logits = np.log(p) - 50.
    with self.cached_session():
      multinom = multinomial.Multinomial(total_count=3., logits=logits)
      self.assertEqual((1, 3), multinom.probs.get_shape())
      self.assertEqual((1, 3), multinom.logits.get_shape())
      self.assertAllClose(p, multinom.probs)
      self.assertAllClose(logits, multinom.logits)

  @test_util.run_v1_only("b/120545219")
  def testPmfUnderflow(self):
    # [FUNCTION DOC]
    # Builds a Multinomial with logits of minus 200 and 0, total count one, and evaluates the log
    # probability of the first category. Asserts it is about minus 200 (no numerical underflow to
    # negative infinity) within tight relative tolerance. Graph mode only.
    logits = np.array([[-200, 0]], dtype=np.float32)
    with self.cached_session():
      dist = multinomial.Multinomial(total_count=1., logits=logits)
      lp = dist.log_prob([1., 0.]).eval()[0]
      self.assertAllClose(-200, lp, atol=0, rtol=1e-6)

  @test_util.run_v1_only("b/120545219")
  def testPmfandCountsAgree(self):
    # [FUNCTION DOC]
    # With validation on and total count five, asserts valid count vectors summing to five
    # evaluate without error, a vector with a negative entry raises an op error saying counts must
    # be non-negative, and a vector not summing to the total count raises an op error about the sum.
    # Graph mode only.
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.cached_session():
      dist = multinomial.Multinomial(total_count=n, probs=p, validate_args=True)
      dist.prob([2., 3, 0]).eval()
      dist.prob([3., 0, 2]).eval()
      with self.assertRaisesOpError("must be non-negative"):
        dist.prob([-1., 4, 2]).eval()
      with self.assertRaisesOpError("counts must sum to `self.total_count`"):
        dist.prob([3., 3, 0]).eval()

  @test_util.run_v1_only("b/120545219")
  def testPmfNonIntegerCounts(self):
    # [FUNCTION DOC]
    # With validation on, asserts integer counts summing to the total evaluate fine, counts with
    # the wrong sum raise an op error, and fractional counts fed through a placeholder raise an op
    # error saying counts cannot contain fractional components. With validation off, asserts that
    # both integer and fractional count vectors evaluate without error. Graph mode only.
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.cached_session():
      # No errors with integer n.
      multinom = multinomial.Multinomial(
          total_count=n, probs=p, validate_args=True)
      multinom.prob([2., 1, 2]).eval()
      multinom.prob([3., 0, 2]).eval()
      # Counts don't sum to n.
      with self.assertRaisesOpError("counts must sum to `self.total_count`"):
        multinom.prob([2., 3, 2]).eval()
      # Counts are non-integers.
      x = array_ops.placeholder(dtypes.float32)
      with self.assertRaisesOpError(
          "cannot contain fractional components."):
        multinom.prob(x).eval(feed_dict={x: [1.0, 2.5, 1.5]})

      multinom = multinomial.Multinomial(
          total_count=n, probs=p, validate_args=False)
      multinom.prob([1., 2., 2.]).eval()
      # Non-integer arguments work.
      multinom.prob([1.0, 2.5, 1.5]).eval()

  def testPmfBothZeroBatches(self):
    # [FUNCTION DOC]
    # Evaluates the probability mass of a count vector for a scalar-batch Multinomial with two
    # equal probabilities and total count one. Asserts the result is 0.5 and has scalar shape.
    with self.cached_session():
      # Both zero-batches.  No broadcast
      p = [0.5, 0.5]
      counts = [1., 0]
      pmf = multinomial.Multinomial(total_count=1., probs=p).prob(counts)
      self.assertAllClose(0.5, self.evaluate(pmf))
      self.assertEqual((), pmf.get_shape())

  def testPmfBothZeroBatchesNontrivialN(self):
    # [FUNCTION DOC]
    # Evaluates the probability mass of counts (3, 2) for total count five and probabilities
    # (0.1, 0.9). Asserts the value equals 81 over 10000 (binomial coefficient ten) and has scalar
    # shape.
    with self.cached_session():
      # Both zero-batches.  No broadcast
      p = [0.1, 0.9]
      counts = [3., 2]
      dist = multinomial.Multinomial(total_count=5., probs=p)
      pmf = dist.prob(counts)
      # 5 choose 3 = 5 choose 2 = 10. 10 * (.9)^2 * (.1)^3 = 81/10000.
      self.assertAllClose(81. / 10000, self.evaluate(pmf))
      self.assertEqual((), pmf.get_shape())

  def testPmfPStretchedInBroadcastWhenSameRank(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shape assertion compares against a parenthesized integer two, not a
    # one-element tuple; it only passes because shape comparison coerces an integer to a
    # one-dimensional shape.
    # Evaluates the mass of two count vectors against one probability row (same rank, broadcast
    # over the batch). Asserts values (0.1, 0.9) and static shape of size two.
    with self.cached_session():
      p = [[0.1, 0.9]]
      counts = [[1., 0], [0, 1]]
      pmf = multinomial.Multinomial(total_count=1., probs=p).prob(counts)
      self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
      self.assertEqual((2), pmf.get_shape())

  def testPmfPStretchedInBroadcastWhenLowerRank(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shape assertion compares against a parenthesized integer two, not a
    # one-element tuple; it relies on integer-to-shape coercion in shape comparison.
    # Evaluates the mass of two count vectors against a single probability vector of lower rank.
    # Asserts values (0.1, 0.9) and static shape of size two.
    with self.cached_session():
      p = [0.1, 0.9]
      counts = [[1., 0], [0, 1]]
      pmf = multinomial.Multinomial(total_count=1., probs=p).prob(counts)
      self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
      self.assertEqual((2), pmf.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shape assertion compares against a parenthesized integer two, not a
    # one-element tuple; it relies on integer-to-shape coercion in shape comparison.
    # Evaluates the mass of one count row against a two-row probability matrix (counts broadcast
    # over the batch, same rank). Asserts values (0.1, 0.7) and static shape of size two. Graph
    # mode only.
    with self.cached_session():
      p = [[0.1, 0.9], [0.7, 0.3]]
      counts = [[1., 0]]
      pmf = multinomial.Multinomial(total_count=1., probs=p).prob(counts)
      self.assertAllClose(pmf, [0.1, 0.7])
      self.assertEqual((2), pmf.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shape assertion compares against a parenthesized integer two, not a
    # one-element tuple; it relies on integer-to-shape coercion in shape comparison.
    # Evaluates the mass of a single count vector (lower rank) against a two-row probability
    # matrix. Asserts values (0.1, 0.7) and static shape of size two. Graph mode only.
    with self.cached_session():
      p = [[0.1, 0.9], [0.7, 0.3]]
      counts = [1., 0]
      pmf = multinomial.Multinomial(total_count=1., probs=p).prob(counts)
      self.assertAllClose(pmf, [0.1, 0.7])
      self.assertEqual(pmf.get_shape(), (2))

  def testPmfShapeCountsStretchedN(self):
    # [FUNCTION DOC]
    # Evaluates mass of one count vector against a two-by-two batch of total counts and a
    # two-by-two-by-two probability array. Asserts evaluation succeeds and static shape is two by
    # two.
    with self.cached_session():
      # [2, 2, 2]
      p = [[[0.1, 0.9], [0.1, 0.9]], [[0.7, 0.3], [0.7, 0.3]]]
      # [2, 2]
      n = [[3., 3], [3, 3]]
      # [2]
      counts = [2., 1]
      pmf = multinomial.Multinomial(total_count=n, probs=p).prob(counts)
      self.evaluate(pmf)
      self.assertEqual(pmf.get_shape(), (2, 2))

  def testPmfShapeCountsPStretchedN(self):
    # [FUNCTION DOC]
    # Evaluates mass of one count vector for a single probability vector and a four-by-three batch
    # of total counts. Asserts evaluation succeeds and static shape is four by three.
    with self.cached_session():
      p = [0.1, 0.9]
      counts = [3., 2]
      n = np.full([4, 3], 5., dtype=np.float32)
      pmf = multinomial.Multinomial(total_count=n, probs=p).prob(counts)
      self.evaluate(pmf)
      self.assertEqual((4, 3), pmf.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testMultinomialMean(self):
    # [FUNCTION DOC]
    # Asserts the mean of a Multinomial with total count five has static shape three and equals
    # five times the probabilities. Graph mode only.
    with self.cached_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      dist = multinomial.Multinomial(total_count=n, probs=p)
      expected_means = 5 * np.array(p, dtype=np.float32)
      self.assertEqual((3,), dist.mean().get_shape())
      self.assertAllClose(expected_means, dist.mean())

  @test_util.run_v1_only("b/120545219")
  def testMultinomialCovariance(self):
    # [FUNCTION DOC]
    # Asserts the covariance of a three-category Multinomial with total count five has static
    # shape three by three and equals the hand-computed matrix. Graph mode only.
    with self.cached_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      dist = multinomial.Multinomial(total_count=n, probs=p)
      expected_covariances = [[9. / 20, -1 / 10, -7 / 20],
                              [-1 / 10, 4 / 5, -7 / 10],
                              [-7 / 20, -7 / 10, 21 / 20]]
      self.assertEqual((3, 3), dist.covariance().get_shape())
      self.assertAllClose(expected_covariances, dist.covariance())

  @test_util.run_v1_only("b/120545219")
  def testMultinomialCovarianceBatch(self):
    # [FUNCTION DOC]
    # Builds a Multinomial with batch shape four by two (parameters of shape two, and four by one
    # by two) and two categories. Asserts covariance has static shape four by two by two by two and
    # equals the hand-computed matrix repeated over the batch. Graph mode only.
    with self.cached_session():
      # Shape [2]
      n = [5.] * 2
      # Shape [4, 1, 2]
      p = [[[0.1, 0.9]], [[0.1, 0.9]]] * 2
      dist = multinomial.Multinomial(total_count=n, probs=p)
      # Shape [2, 2]
      inner_var = [[9. / 20, -9 / 20], [-9 / 20, 9 / 20]]
      # Shape [4, 2, 2, 2]
      expected_covariances = [[inner_var, inner_var]] * 4
      self.assertEqual((4, 2, 2, 2), dist.covariance().get_shape())
      self.assertAllClose(expected_covariances, dist.covariance())

  def testCovarianceMultidimensional(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the global unseeded numpy random generator (not the seeded fixture), so
    # inputs differ per run; only static shapes are asserted, so this does not affect the outcome.
    # Builds two Multinomials with random Dirichlet probabilities and random integer total counts
    # (batch three by five with four categories, and batch six by three with three categories) and
    # asserts the static covariance shapes are the batch shape followed by categories twice.
    # Shape [3, 5, 4]
    p = np.random.dirichlet([.25, .25, .25, .25], [3, 5]).astype(np.float32)
    # Shape [6, 3, 3]
    p2 = np.random.dirichlet([.3, .3, .4], [6, 3]).astype(np.float32)

    ns = np.random.randint(low=1, high=11, size=[3, 5]).astype(np.float32)
    ns2 = np.random.randint(low=1, high=11, size=[6, 1]).astype(np.float32)

    with self.cached_session():
      dist = multinomial.Multinomial(ns, p)
      dist2 = multinomial.Multinomial(ns2, p2)

      covariance = dist.covariance()
      covariance2 = dist2.covariance()
      self.assertEqual((3, 5, 4, 4), covariance.get_shape())
      self.assertEqual((6, 3, 3, 3), covariance2.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testCovarianceFromSampling(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment inside says the test uses a Dirichlet multinomial built via
    # broadcast between alpha and n, but the code actually builds a plain Multinomial from
    # probabilities and counts (stale copied comment).
    # Builds a Multinomial with batch shape three by two and three categories, draws one million
    # seeded samples, computes sample mean, covariance, variance and standard deviation, and
    # asserts each is within 0.01 absolute and relative tolerance of the analytic mean, covariance,
    # variance and standard deviation. Graph mode only; uses an explicit session.
    # We will test mean, cov, var, stddev on a DirichletMultinomial constructed
    # via broadcast between alpha, n.
    theta = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    theta /= np.sum(theta, 1)[..., array_ops.newaxis]
    n = np.array([[10., 9.], [8., 7.], [6., 5.]], dtype=np.float32)
    with self.cached_session() as sess:
      # batch_shape=[3, 2], event_shape=[3]
      dist = multinomial.Multinomial(n, theta)
      x = dist.sample(int(1000e3), seed=1)
      sample_mean = math_ops.reduce_mean(x, 0)
      x_centered = x - sample_mean[array_ops.newaxis, ...]
      sample_cov = math_ops.reduce_mean(math_ops.matmul(
          x_centered[..., array_ops.newaxis],
          x_centered[..., array_ops.newaxis, :]), 0)
      sample_var = array_ops.matrix_diag_part(sample_cov)
      sample_stddev = math_ops.sqrt(sample_var)
      [
          sample_mean_,
          sample_cov_,
          sample_var_,
          sample_stddev_,
          analytic_mean,
          analytic_cov,
          analytic_var,
          analytic_stddev,
      ] = sess.run([
          sample_mean,
          sample_cov,
          sample_var,
          sample_stddev,
          dist.mean(),
          dist.covariance(),
          dist.variance(),
          dist.stddev(),
      ])
      self.assertAllClose(sample_mean_, analytic_mean, atol=0.01, rtol=0.01)
      self.assertAllClose(sample_cov_, analytic_cov, atol=0.01, rtol=0.01)
      self.assertAllClose(sample_var_, analytic_var, atol=0.01, rtol=0.01)
      self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.01, rtol=0.01)

  @test_util.run_v1_only("b/120545219")
  def testSampleUnbiasedNonScalarBatch(self):
    # [FUNCTION DOC]
    # Builds a Multinomial with batch shape four by three, two categories, random logits from the
    # seeded fixture and per-column total counts. Draws 30000 seeded samples and asserts static
    # shapes of the sample mean and covariance, the sample mean within ten percent and the sample
    # covariance within twenty percent of the analytic values. Graph mode only.
    with self.cached_session() as sess:
      dist = multinomial.Multinomial(
          total_count=[7., 6., 5.],
          logits=math_ops.log(2. * self._rng.rand(4, 3, 2).astype(np.float32)))
      n = int(3e4)
      x = dist.sample(n, seed=0)
      sample_mean = math_ops.reduce_mean(x, 0)
      # Cyclically rotate event dims left.
      x_centered = array_ops.transpose(x - sample_mean, [1, 2, 3, 0])
      sample_covariance = math_ops.matmul(
          x_centered, x_centered, adjoint_b=True) / n
      [
          sample_mean_,
          sample_covariance_,
          actual_mean_,
          actual_covariance_,
      ] = sess.run([
          sample_mean,
          sample_covariance,
          dist.mean(),
          dist.covariance(),
      ])
      self.assertAllEqual([4, 3, 2], sample_mean.get_shape())
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.10)
      self.assertAllEqual([4, 3, 2, 2], sample_covariance.get_shape())
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  @test_util.run_v1_only("b/120545219")
  def testSampleUnbiasedScalarBatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: inline comment says the centered samples are already transposed to n by two, but
    # there are four categories so the shape is n by four (stale comment).
    # Builds a scalar-batch Multinomial with four categories and total count five from random
    # logits. Draws 5000 seeded samples and asserts static shapes of the sample mean (four) and
    # covariance (four by four), the mean within ten percent and covariance within twenty percent
    # of the analytic values. Graph mode only.
    with self.cached_session() as sess:
      dist = multinomial.Multinomial(
          total_count=5.,
          logits=math_ops.log(2. * self._rng.rand(4).astype(np.float32)))
      n = int(5e3)
      x = dist.sample(n, seed=0)
      sample_mean = math_ops.reduce_mean(x, 0)
      x_centered = x - sample_mean  # Already transposed to [n, 2].
      sample_covariance = math_ops.matmul(
          x_centered, x_centered, adjoint_a=True) / n
      [
          sample_mean_,
          sample_covariance_,
          actual_mean_,
          actual_covariance_,
      ] = sess.run([
          sample_mean,
          sample_covariance,
          dist.mean(),
          dist.covariance(),
      ])
      self.assertAllEqual([4], sample_mean.get_shape())
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.10)
      self.assertAllEqual([4, 4], sample_covariance.get_shape())
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testNotReparameterized(self):
    # [FUNCTION DOC]
    # Samples a Multinomial inside a gradient tape watching total count and probabilities, and
    # asserts both gradients of the samples are the null value (sampling is not reparameterized).
    total_count = constant_op.constant(5.0)
    p = constant_op.constant([0.2, 0.6])
    with backprop.GradientTape() as tape:
      tape.watch(total_count)
      tape.watch(p)
      dist = multinomial.Multinomial(
          total_count=total_count,
          probs=p)
      samples = dist.sample(100)
    grad_total_count, grad_p = tape.gradient(samples, [total_count, p])
    self.assertIsNone(grad_total_count)
    self.assertIsNone(grad_p)


if __name__ == "__main__":
  test.main()
