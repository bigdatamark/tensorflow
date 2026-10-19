# [MODULE DOC]
# Unit tests for the DirichletMultinomial distribution in the distributions package of
# TensorFlow ops. Covers static and dynamic shapes with batched total counts and concentrations,
# the total_count and concentration properties, validation of counts (non-negative, integer valued,
# summing to the total count), probability mass values and broadcasting, the identity that one
# vote has the mean as probability, mean, covariance (closed form, broadcasting, multidimensional,
# and comparison with sample statistics), behavior with large and small total concentration,
# disabling all checks with validate_args off, unbiasedness of sampling, and that sampling is not
# reparameterized. The distribution module is bound to a short alias at import time; no other
# import-time side effects. Running the file as a script runs the tests.
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
from tensorflow.python.ops.distributions import dirichlet_multinomial
from tensorflow.python.platform import test


ds = dirichlet_multinomial


class DirichletMultinomialTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the DirichletMultinomial distribution. Holds a seeded numpy random state created
  # in setUp, used by the sampling tests. Most tests run in graph mode (deprecated v1 decorator) and
  # use a cached session; others use the default evaluation helper. The alias for the distribution
  # module is a module-level name.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class per-test initialization (such as
    # graph reset and random seed handling) is skipped.
    # Per-test fixture: creates a numpy random state seeded with 42 and stores it on the instance as a
    # private attribute for deterministic random concentrations in the sampling tests. Takes no
    # arguments, returns nothing.
    self._rng = np.random.RandomState(42)

  @test_util.run_deprecated_v1
  def testSimpleShapes(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): total count 1 and a random concentration vector of length three.
    # Expected: the event shape tensor evaluates to three, the batch shape is empty, and the static
    # event shape is three and the static batch shape is empty.
    with self.cached_session():
      alpha = np.random.rand(3)
      dist = ds.DirichletMultinomial(1., alpha)
      self.assertEqual(3, dist.event_shape_tensor().eval())
      self.assertAllEqual([], dist.batch_shape_tensor())
      self.assertEqual(tensor_shape.TensorShape([3]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([]), dist.batch_shape)

  @test_util.run_deprecated_v1
  def testComplexShapes(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): concentration of shape 3x2x2 and total count of shape 3x2.
    # Expected: event shape is two and batch shape is 3x2, both as tensors and as static shapes.
    with self.cached_session():
      alpha = np.random.rand(3, 2, 2)
      n = [[3., 2], [4, 5], [6, 7]]
      dist = ds.DirichletMultinomial(n, alpha)
      self.assertEqual(2, dist.event_shape_tensor().eval())
      self.assertAllEqual([3, 2], dist.batch_shape_tensor())
      self.assertEqual(tensor_shape.TensorShape([2]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([3, 2]), dist.batch_shape)

  @test_util.run_deprecated_v1
  def testNproperty(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): total count 5 with shape 1x1 and concentration 1x3.
    # Expected: the total_count property has static shape 1x1 and evaluates to the supplied value.
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.cached_session():
      dist = ds.DirichletMultinomial(n, alpha)
      self.assertEqual([1, 1], dist.total_count.get_shape())
      self.assertAllClose(n, dist.total_count)

  @test_util.run_deprecated_v1
  def testAlphaProperty(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): total count 1 and concentration of shape 1x3.
    # Expected: the concentration property has static shape 1x3 and evaluates to the supplied values.
    alpha = [[1., 2, 3]]
    with self.cached_session():
      dist = ds.DirichletMultinomial(1, alpha)
      self.assertEqual([1, 3], dist.concentration.get_shape())
      self.assertAllClose(alpha, dist.concentration)

  @test_util.run_deprecated_v1
  def testPmfNandCountsAgree(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): total count 5, concentration 1,2,3, validation on.
    # Expected: counts that sum to five evaluate without error; a negative count raises an op error
    # saying counts must be non-negative; counts that do not sum to the total raise an op error saying
    # the last dimension must sum to the total count.
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.cached_session():
      dist = ds.DirichletMultinomial(n, alpha, validate_args=True)
      dist.prob([2., 3, 0]).eval()
      dist.prob([3., 0, 2]).eval()
      with self.assertRaisesOpError("must be non-negative"):
        dist.prob([-1., 4, 2]).eval()
      with self.assertRaisesOpError(
          "last-dimension must sum to `self.total_count`"):
        dist.prob([3., 3, 0]).eval()

  @test_util.run_deprecated_v1
  def testPmfNonIntegerCounts(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): validation on, total count 5, concentration 1,2,3.
    # Expected: integer-valued counts (including floating values with no fraction) are accepted; fed
    # fractional counts raise an op error saying counts cannot contain fractional components. With
    # validation off, both integer counts that do not satisfy the sum and fractional counts evaluate
    # without error.
    alpha = [[1., 2, 3]]
    n = [[5.]]
    with self.cached_session():
      dist = ds.DirichletMultinomial(n, alpha, validate_args=True)
      dist.prob([2., 3, 0]).eval()
      dist.prob([3., 0, 2]).eval()
      dist.prob([3.0, 0, 2.0]).eval()
      # Both equality and integer checking fail.
      placeholder = array_ops.placeholder(dtypes.float32)
      with self.assertRaisesOpError(
          "cannot contain fractional components"):
        dist.prob(placeholder).eval(feed_dict={placeholder: [1.0, 2.5, 1.5]})
      dist = ds.DirichletMultinomial(n, alpha, validate_args=False)
      dist.prob([1., 2., 3.]).eval()
      # Non-integer arguments work.
      dist.prob([1.0, 2.5, 1.5]).eval()

  def testPmfBothZeroBatches(self):
    # [FUNCTION DOC]
    # Scenario: total count 1, concentration of two classes, and one vote in class zero.
    # Expected: the probability is 1/3 (the mean for the class) and the static shape is empty.
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      # Both zero-batches.  No broadcast
      alpha = [1., 2]
      counts = [1., 0]
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1 / 3., self.evaluate(pmf))
      self.assertEqual((), pmf.get_shape())

  def testPmfBothZeroBatchesNontrivialN(self):
    # [FUNCTION DOC]
    # Scenario: total count 5, concentration 1 and 2, counts 3 and 2.
    # Expected: probability 1/7 with empty static shape.
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      # Both zero-batches.  No broadcast
      alpha = [1., 2]
      counts = [3., 2]
      dist = ds.DirichletMultinomial(5., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1 / 7., self.evaluate(pmf))
      self.assertEqual((), pmf.get_shape())

  def testPmfBothZeroBatchesMultidimensionalN(self):
    # [FUNCTION DOC]
    # Scenario: a 4x3 array of total counts all equal to 5 with a single concentration vector and counts.
    # Expected: the probability is 1/7 everywhere and the static shape is 4x3.
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [1., 2]
      counts = [3., 2]
      n = np.full([4, 3], 5., dtype=np.float32)
      dist = ds.DirichletMultinomial(n, alpha)
      pmf = dist.prob(counts)
      self.assertAllClose([[1 / 7., 1 / 7., 1 / 7.]] * 4, self.evaluate(pmf))
      self.assertEqual((4, 3), pmf.get_shape())

  def testPmfAlphaStretchedInBroadcastWhenSameRank(self):
    # [FUNCTION DOC]
    # Scenario: concentration of shape 1x2 and counts of shape 2x2 with total count of one.
    # Expected: probabilities 1/3 and 2/3 with static shape two (concentration broadcast over rows).
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [[1., 2]]
      counts = [[1., 0], [0., 1]]
      dist = ds.DirichletMultinomial([1.], alpha)
      pmf = dist.prob(counts)
      self.assertAllClose([1 / 3., 2 / 3.], self.evaluate(pmf))
      self.assertAllEqual([2], pmf.get_shape())

  def testPmfAlphaStretchedInBroadcastWhenLowerRank(self):
    # [FUNCTION DOC]
    # Scenario: concentration of rank 1 and counts of shape 2x2.
    # Expected: probabilities 1/3 and 2/3 with static shape two.
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [1., 2]
      counts = [[1., 0], [0., 1]]
      pmf = ds.DirichletMultinomial(1., alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 3.], self.evaluate(pmf))
      self.assertAllEqual([2], pmf.get_shape())

  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    # [FUNCTION DOC]
    # Scenario: concentration 2x2, total counts of two ones, counts of shape 1x2.
    # Expected: probabilities 1/3 and 2/5 with static shape two (counts broadcast over rows).
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [[1., 2], [2., 3]]
      counts = [[1., 0]]
      pmf = ds.DirichletMultinomial([1., 1.], alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 5.], self.evaluate(pmf))
      self.assertAllEqual([2], pmf.get_shape())

  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    # [FUNCTION DOC]
    # Scenario: concentration 2x2 and a rank-1 counts vector.
    # Expected: probabilities 1/3 and 2/5 with static shape two.
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.cached_session():
      alpha = [[1., 2], [2., 3]]
      counts = [1., 0]
      pmf = ds.DirichletMultinomial(1., alpha).prob(counts)
      self.assertAllClose([1 / 3., 2 / 5.], self.evaluate(pmf))
      self.assertAllEqual([2], pmf.get_shape())

  @test_util.run_deprecated_v1
  def testPmfForOneVoteIsTheMeanWithOneRecordInput(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): total count one with concentration 1,2,3, and for each class a one-hot
    # count vector.
    # Expected: the probability of each one-hot vector equals the corresponding element of the mean,
    # the mean has shape three and the probability is a scalar.
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [1., 2, 3]
    with self.cached_session():
      for class_num in range(3):
        counts = np.zeros([3], dtype=np.float32)
        counts[class_num] = 1
        dist = ds.DirichletMultinomial(1., alpha)
        mean = dist.mean().eval()
        pmf = dist.prob(counts).eval()

        self.assertAllClose(mean[class_num], pmf)
        self.assertAllEqual([3], mean.shape)
        self.assertAllEqual([], pmf.shape)

  @test_util.run_deprecated_v1
  def testMeanDoubleTwoVotes(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): distributions with total counts one and two and the same concentration.
    # Expected: the mean for total count two is twice the mean for total count one for every class, and
    # the mean has shape three. The one-hot count arrays built inside the loop are unused.
    # The probabilities of two votes falling into class k for
    # DirichletMultinomial(2, alpha) is twice as much as the probability of one
    # vote falling into class k for DirichletMultinomial(1, alpha)
    alpha = [1., 2, 3]
    with self.cached_session():
      for class_num in range(3):
        counts_one = np.zeros([3], dtype=np.float32)
        counts_one[class_num] = 1.
        counts_two = np.zeros([3], dtype=np.float32)
        counts_two[class_num] = 2

        dist1 = ds.DirichletMultinomial(1., alpha)
        dist2 = ds.DirichletMultinomial(2., alpha)

        mean1 = dist1.mean().eval()
        mean2 = dist2.mean().eval()

        self.assertAllClose(mean2[class_num], 2 * mean1[class_num])
        self.assertAllEqual([3], mean1.shape)

  @test_util.run_deprecated_v1
  def testCovarianceFromSampling(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a batch of two concentration vectors with total count five; 250000 seeded
    # samples.
    # Action: computes sample mean, covariance, variance and standard deviation by hand, and the
    # analytic ones from the distribution, in one session run.
    # Expected: sample and analytic mean, covariance, variance and stddev agree within absolute
    # tolerances of 0.02 to 0.05.
    # We will test mean, cov, var, stddev on a DirichletMultinomial constructed
    # via broadcast between alpha, n.
    alpha = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    # Ideally we'd be able to test broadcasting but, the multinomial sampler
    # doesn't support different total counts.
    n = np.float32(5)
    with self.cached_session() as sess:
      # batch_shape=[2], event_shape=[3]
      dist = ds.DirichletMultinomial(n, alpha)
      x = dist.sample(int(250e3), seed=1)
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
      self.assertAllClose(sample_mean_, analytic_mean, atol=0.04, rtol=0.)
      self.assertAllClose(sample_cov_, analytic_cov, atol=0.05, rtol=0.)
      self.assertAllClose(sample_var_, analytic_var, atol=0.05, rtol=0.)
      self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.02, rtol=0.)

  @test_util.run_without_tensor_float_32(
      "Tests DirichletMultinomial.covariance, which calls matmul")
  def testCovariance(self):
    # [FUNCTION DOC]
    # Scenario: concentration (1,2) and several total counts, with the test restricted so that
    # TensorFloat-32 is not used because covariance relies on matrix multiplication.
    # Expected: for each count the covariance has static shape 2x2 and equals n times (n plus the
    # concentration sum) over (1 plus the concentration sum) times the shared matrix of normalized
    # variance and covariance entries computed by hand.
    # Shape [2]
    alpha = [1., 2]
    ns = [2., 3., 4., 5.]
    alpha_0 = np.sum(alpha)

    # Diagonal entries are of the form:
    # Var(X_i) = n * alpha_i / alpha_sum * (1 - alpha_i / alpha_sum) *
    # (alpha_sum + n) / (alpha_sum + 1)
    variance_entry = lambda a, a_sum: a / a_sum * (1 - a / a_sum)
    # Off diagonal entries are of the form:
    # Cov(X_i, X_j) = -n * alpha_i * alpha_j / (alpha_sum ** 2) *
    # (alpha_sum + n) / (alpha_sum + 1)
    covariance_entry = lambda a, b, a_sum: -a * b / a_sum**2
    # Shape [2, 2].
    shared_matrix = np.array([[
        variance_entry(alpha[0], alpha_0),
        covariance_entry(alpha[0], alpha[1], alpha_0)
    ], [
        covariance_entry(alpha[1], alpha[0], alpha_0),
        variance_entry(alpha[1], alpha_0)
    ]])

    with self.cached_session():
      for n in ns:
        # n is shape [] and alpha is shape [2].
        dist = ds.DirichletMultinomial(n, alpha)
        covariance = dist.covariance()
        expected_covariance = n * (n + alpha_0) / (1 + alpha_0) * shared_matrix

        self.assertEqual([2, 2], covariance.get_shape())
        self.assertAllClose(expected_covariance, self.evaluate(covariance))

  def testCovarianceNAlphaBroadcast(self):
    # [FUNCTION DOC]
    # Scenario: concentration of shape 4x3 and total counts of shape 4x1.
    # Expected: the covariance has static shape 4x3x3 and equals the shared normalized matrix scaled
    # per batch member by the factor involving the count and concentration sum.
    alpha_v = [1., 2, 3]
    alpha_0 = 6.

    # Shape [4, 3]
    alpha = np.array(4 * [alpha_v], dtype=np.float32)
    # Shape [4, 1]
    ns = np.array([[2.], [3.], [4.], [5.]], dtype=np.float32)

    variance_entry = lambda a, a_sum: a / a_sum * (1 - a / a_sum)
    covariance_entry = lambda a, b, a_sum: -a * b / a_sum**2
    # Shape [4, 3, 3]
    shared_matrix = np.array(
        4 * [[[
            variance_entry(alpha_v[0], alpha_0),
            covariance_entry(alpha_v[0], alpha_v[1], alpha_0),
            covariance_entry(alpha_v[0], alpha_v[2], alpha_0)
        ], [
            covariance_entry(alpha_v[1], alpha_v[0], alpha_0),
            variance_entry(alpha_v[1], alpha_0),
            covariance_entry(alpha_v[1], alpha_v[2], alpha_0)
        ], [
            covariance_entry(alpha_v[2], alpha_v[0], alpha_0),
            covariance_entry(alpha_v[2], alpha_v[1], alpha_0),
            variance_entry(alpha_v[2], alpha_0)
        ]]],
        dtype=np.float32)

    with self.cached_session():
      # ns is shape [4, 1], and alpha is shape [4, 3].
      dist = ds.DirichletMultinomial(ns, alpha)
      covariance = dist.covariance()
      expected_covariance = shared_matrix * (
          ns * (ns + alpha_0) / (1 + alpha_0))[..., array_ops.newaxis]

      self.assertEqual([4, 3, 3], covariance.get_shape())
      self.assertAllClose(expected_covariance, self.evaluate(covariance))

  def testCovarianceMultidimensional(self):
    # [FUNCTION DOC]
    # Scenario: two distributions with batched concentrations and total counts of shapes 3x5x4 with
    # 3x5x1 and 6x3x3 with 6x1x1.
    # Expected: the static covariance shapes are 3x5x4x4 and 6x3x3x3. Values are not checked.
    alpha = np.random.rand(3, 5, 4).astype(np.float32)
    alpha2 = np.random.rand(6, 3, 3).astype(np.float32)

    ns = np.random.randint(low=1, high=11, size=[3, 5, 1]).astype(np.float32)
    ns2 = np.random.randint(low=1, high=11, size=[6, 1, 1]).astype(np.float32)

    with self.cached_session():
      dist = ds.DirichletMultinomial(ns, alpha)
      dist2 = ds.DirichletMultinomial(ns2, alpha2)

      covariance = dist.covariance()
      covariance2 = dist2.covariance()
      self.assertEqual([3, 5, 4, 4], covariance.get_shape())
      self.assertEqual([6, 3, 3, 3], covariance2.get_shape())

  def testZeroCountsResultsInPmfEqualToOne(self):
    # [FUNCTION DOC]
    # Scenario: total count zero and zero counts.
    # Expected: the probability is exactly one with empty static shape, since there is one way to select nothing.
    # There is only one way for zero items to be selected, and this happens with
    # probability 1.
    alpha = [5, 0.5]
    counts = [0., 0]
    with self.cached_session():
      dist = ds.DirichletMultinomial(0., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(1.0, self.evaluate(pmf))
      self.assertEqual((), pmf.get_shape())

  def testLargeTauGivesPreciseProbabilities(self):
    # [FUNCTION DOC]
    # Scenario: concentration equals a large total strength 100 times mean probabilities 0.1, 0.1, 0.8,
    # with one, two and three draws.
    # Expected: the probabilities approach those of independent draws: 0.8 for one draw, 0.64 for two
    # draws of the third class, and 3 times 0.1 times 0.8 squared for three draws. Static shape is empty.
    # If tau is large, we are doing coin flips with probability mu.
    mu = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    tau = np.array([100.], dtype=np.float32)
    alpha = tau * mu

    # One (three sided) coin flip.  Prob[coin 3] = 0.8.
    # Note that since it was one flip, value of tau didn't matter.
    counts = [0., 0, 1]
    with self.cached_session():
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.8, self.evaluate(pmf), atol=1e-4)
      self.assertEqual((), pmf.get_shape())

    # Two (three sided) coin flips.  Prob[coin 3] = 0.8.
    counts = [0., 0, 2]
    with self.cached_session():
      dist = ds.DirichletMultinomial(2., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.8**2, self.evaluate(pmf), atol=1e-2)
      self.assertEqual((), pmf.get_shape())

    # Three (three sided) coin flips.
    counts = [1., 0, 2]
    with self.cached_session():
      dist = ds.DirichletMultinomial(3., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(3 * 0.1 * 0.8 * 0.8, self.evaluate(pmf), atol=1e-2)
      self.assertEqual((), pmf.get_shape())

  def testSmallTauPrefersCorrelatedResults(self):
    # [FUNCTION DOC]
    # Scenario: concentration equals a small total strength 0.1 times means of one half each.
    # Expected: a single draw has probability 0.5; with two draws the probability of both in the same
    # class is more than five times the probability of splitting between the classes.
    # If tau is small, then correlation between draws is large, so draws that
    # are both of the same class are more likely.
    mu = np.array([0.5, 0.5], dtype=np.float32)
    tau = np.array([0.1], dtype=np.float32)
    alpha = tau * mu

    # If there is only one draw, it is still a coin flip, even with small tau.
    counts = [1., 0]
    with self.cached_session():
      dist = ds.DirichletMultinomial(1., alpha)
      pmf = dist.prob(counts)
      self.assertAllClose(0.5, self.evaluate(pmf))
      self.assertEqual((), pmf.get_shape())

    # If there are two draws, it is much more likely that they are the same.
    counts_same = [2., 0]
    counts_different = [1, 1.]
    with self.cached_session():
      dist = ds.DirichletMultinomial(2., alpha)
      pmf_same = dist.prob(counts_same)
      pmf_different = dist.prob(counts_different)
      self.assertLess(5 * self.evaluate(pmf_different), self.evaluate(pmf_same))
      self.assertEqual((), pmf_same.get_shape())

  @test_util.run_deprecated_v1
  def testNonStrictTurnsOffAllChecks(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): deliberately invalid inputs (negative concentration, negative counts, a
    # negative non-integer total count) with validation off.
    # Expected: evaluating the probability does not raise an error.
    # Make totally invalid input.
    with self.cached_session():
      alpha = [[-1., 2]]  # alpha should be positive.
      counts = [[1., 0], [0., -1]]  # counts should be non-negative.
      n = [-5.3]  # n should be a non negative integer equal to counts.sum.
      dist = ds.DirichletMultinomial(n, alpha, validate_args=False)
      dist.prob(counts).eval()  # Should not raise.

  @test_util.run_deprecated_v1
  def testSampleUnbiasedNonScalarBatch(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): total count 5 and a random concentration batch of shape 4x3x2; 3000 seeded
    # samples.
    # Action: computes the sample mean and sample covariance (after rotating the event axis) and
    # compares with the distribution mean and covariance.
    # Expected: static shapes are 4x3x2 for the mean and 4x3x2x2 for the covariance, and values agree
    # within 20 percent relative tolerance.
    with self.cached_session() as sess:
      dist = ds.DirichletMultinomial(
          total_count=5.,
          concentration=1. + 2. * self._rng.rand(4, 3, 2).astype(np.float32))
      n = int(3e3)
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
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.20)
      self.assertAllEqual([4, 3, 2, 2], sample_covariance.get_shape())
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  @test_util.run_deprecated_v1
  def testSampleUnbiasedScalarBatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment says the samples are already transposed to shape n by 2, but the
    # concentration has four classes so the samples actually have shape n by 4.
    # Scenario (graph mode): total count 5 and a random concentration vector of length four; 5000
    # seeded samples.
    # Action: computes the sample mean and sample covariance and compares with the distribution values.
    # Expected: static shapes are 4 for the mean and 4x4 for the covariance, with values within 20
    # percent relative tolerance.
    with self.cached_session() as sess:
      dist = ds.DirichletMultinomial(
          total_count=5.,
          concentration=1. + 2. * self._rng.rand(4).astype(np.float32))
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
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.20)
      self.assertAllEqual([4, 4], sample_covariance.get_shape())
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testNotReparameterized(self):
    # [FUNCTION DOC]
    # Scenario: total count and concentration constants are watched on a gradient tape while 100
    # samples are drawn.
    # Expected: the gradients of the samples with respect to both the total count and the concentration
    # are nothing/null.
    total_count = constant_op.constant(5.0)
    concentration = constant_op.constant([0.1, 0.1, 0.1])
    with backprop.GradientTape() as tape:
      tape.watch(total_count)
      tape.watch(concentration)
      dist = ds.DirichletMultinomial(
          total_count=total_count,
          concentration=concentration)
      samples = dist.sample(100)
    grad_total_count, grad_concentration = tape.gradient(
        samples, [total_count, concentration])
    self.assertIsNone(grad_total_count)
    self.assertIsNone(grad_concentration)


if __name__ == "__main__":
  test.main()
