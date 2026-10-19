# [MODULE DOC]
# Test module for categorical (multinomial) sampling ops on the XLA test device, both stateful and
# stateless. Checks output range, non-constant output, statistical correctness via chi-squared,
# seed determinism and empty-sample cases. No import-time side effects.
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
"""Tests for multinomial generation ops in the XLA JIT compiler."""

import collections

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


# TODO(srvasude): Merge this with
# third_party/tensorflow/python/kernel_tests/random/multinomial_op_test.py.
class CategoricalTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for multinomial (categorical) sampling ops, stateful and stateless. Uses the
  # integer type sets from the XLA test base class. Holds no state of its own.
  """Test cases for random-number generating operators."""

  def output_dtypes(self):
    # [FUNCTION DOC]
    # HELPER: returns the set of output integer types to test: those supported by the test
    # device that are also 32-bit or 64-bit integers.
    return set(self.int_types).intersection([np.int32, np.int64])

  def _chi2(self, expected, actual):
    # [FUNCTION DOC]
    # HELPER: statistics helper for the sampling-correctness test.
    # Computes the sum over all entries of squared difference divided by the expected value,
    # using array conversion of both inputs (expected must be nonzero). Returns a number.
    """Returns Chi2 GOF statistic."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = actual - expected
    chi2 = np.sum(diff * diff / expected)
    return chi2

  def _do_sampling(self, logits, num_samples):
    # [FUNCTION DOC]
    # HELPER: sampling helper for the correctness test.
    # Sets the global random seed to a fixed value, draws the requested number of samples per
    # batch row from the multinomial op on the test device with 32-bit integer output, then
    # asserts all drawn labels are within zero to number-of-classes. Returns a list (per batch
    # row) of the observed frequencies of each class. Side effect: sets the global random seed.
    """Categorical samples from given input.

    Args:
      logits: Numpy ndarray of shape [batch_size, num_classes].
      num_samples: Int; number of samples to draw.

    Returns:
      Frequencies from sampled classes; shape [batch_size, num_classes].
    """
    with self.session(), self.test_scope():
      random_seed.set_random_seed(1618)
      op = random_ops.multinomial(logits, num_samples,
                                  output_dtype=dtypes.int32)
      d = self.evaluate(op)

    batch_size, num_classes = logits.shape
    freqs_mat = []
    for i in range(batch_size):
      cnts = dict(collections.Counter(d[i, :]))

      # Requires drawn class labels be in range.
      self.assertLess(max(cnts.keys()), num_classes)
      self.assertGreaterEqual(min(cnts.keys()), 0)

      freqs = [(cnts[k] * 1. / num_samples if k in cnts else 0)
               for k in range(num_classes)]
      freqs_mat.append(freqs)

    return freqs_mat

  def _testRngIsNotConstant(self, rng, dtype, output_dtype):
    # [FUNCTION DOC]
    # HELPER: shared check for the non-constant test.
    # Builds the random output on the test device by calling the supplied generator function with
    # an input dtype and output dtype, evaluates it three times, and asserts that at least two of
    # the three results differ. Returns nothing.
    # Tests that 'rng' does not always return the same value.
    with self.session():
      with self.test_scope():
        x = rng(dtype, output_dtype)

      # The random-number generator, if working correctly, should produce the
      # same output multiple times with low probability.
      y = self.evaluate(x)
      z = self.evaluate(x)
      w = self.evaluate(x)

      # We use exact equality here. If the random-number generator is producing
      # deterministic output, all three outputs will be bitwise identical.
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or
                      (not np.array_equal(y, w)))

  def testCategoricalIsNotConstant(self):
    # [FUNCTION DOC]
    # For each supported output integer type, checks that sampling ten draws from a uniform
    # three-class distribution gives varying results across evaluations. Returns nothing.
    def rng(dtype, output_dtype):
      # [FUNCTION DOC]
      # HELPER: generator passed to the non-constant check. Draws ten multinomial samples from
      # equal logits over three classes in the given input dtype and output dtype.
      return random_ops.multinomial(np.array([[1., 1., 1.]], dtype=dtype), 10,
                                    output_dtype=output_dtype)

    dtype = np.float32
    for output_dtype in self.output_dtypes():
      self._testRngIsNotConstant(rng, dtype, output_dtype)

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm, skip_message="Test fails on ROCm.")
  def testCategoricalIsInRange(self):
    # [FUNCTION DOC]
    # For each float dtype and supported output integer type, draws 1000 samples from equal
    # logits over 20 classes and asserts every sample is at least zero and below 20. Disabled
    # on ROCm builds where it fails. Returns nothing.
    for dtype in self.float_types:
      for output_dtype in self.output_dtypes():
        with self.session():
          with self.test_scope():
            x = random_ops.multinomial(
                array_ops.ones(shape=[1, 20], dtype=dtype), 1000,
                output_dtype=output_dtype)
          y = self.evaluate(x)
          self.assertTrue((y >= 0).sum() == 1000)
          self.assertTrue((y < 20).sum() == 1000)

  def testSamplingCorrectness(self):
    # [FUNCTION DOC]
    # Statistical test. Seeds numpy, then for several probability tables (two fixed ones, a random
    # Dirichlet vector and a batch of three random vectors) samples 40000 draws from the logits
    # (log probabilities) and asserts the chi-squared statistic between the true probabilities and
    # observed frequencies is below 1e-3. Side effect: sets the numpy random seed. Returns nothing.
    np.random.seed(1618)  # Make it reproducible.
    num_samples = 40000

    rand_probs = np.random.dirichlet([1., 1., 2., 3.])
    rand_probs2 = np.random.dirichlet([1., 4., 5.], size=3)  # batched
    for probs in [[.5, .5], [.85, .05, .1], rand_probs, rand_probs2]:
      probs = np.asarray(probs)
      if len(probs.shape) == 1:
        probs = probs.reshape(1, probs.size)  # singleton batch

      logits = np.log(probs).astype(np.float32)
      freqs = self._do_sampling(logits, num_samples)

      # the test here is similar to
      # python/kernel_tests/random/multinomial_op_test.py
      # Note that df >= 1 in all these cases. Choosing a cutoff of 1e-3
      # corresponds to an alpha value of 2.5% for df = 1, and smaller for larger
      # df.
      chi2 = self._chi2(probs, freqs)
      self.assertLess(chi2, 1e-3)

  def testStatelessMultinomialIsInRange(self):
    # [FUNCTION DOC]
    # Same range check as the stateful test but with the stateless multinomial op, a fed
    # two-element integer seed, 20 classes, 1000 samples, for float32 and bfloat16 (when
    # supported) and each output integer type. Asserts all samples lie in zero to 19. Returns nothing.
    for dtype in self.float_types.intersection(
        [dtypes.float32, dtypes.bfloat16]):
      for output_dtype in self.output_dtypes():
        with self.session() as sess:
          with self.test_scope():
            seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
            x = stateless_random_ops.stateless_multinomial(
                array_ops.ones(shape=[1, 20], dtype=dtype),
                1000,
                seed_t,
                output_dtype=output_dtype)
          y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
          self.assertTrue((y >= 0).sum() == 1000)
          self.assertTrue((y < 20).sum() == 1000)

  def testDeterminismMultinomial(self):
    # [FUNCTION DOC]
    # Checks that stateless multinomial output depends only on the seed. For two logit tables,
    # runs the op for a list of 25 seed pairs repeated three times and asserts that for every
    # pair of runs, the seeds are equal exactly when the outputs are all equal (probabilistic).
    # Returns nothing.
    # Stateless values should be equal iff the seeds are equal (roughly)
    num_samples = 10
    with self.session(), self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      seeds = [(x, y) for x in range(5) for y in range(5)] * 3
      for logits in ([[0.1, 0.25, 0.5, 0.15]], [[0.5, 0.5], [0.8, 0.2],
                                                [0.25, 0.75]]):
        pure = stateless_random_ops.stateless_multinomial(
            logits, num_samples, seed=seed_t)
        values = [(seed, pure.eval(feed_dict={seed_t: seed})) for seed in seeds]
        for s0, v0 in values:
          for s1, v1 in values:
            self.assertEqual(s0 == s1, np.all(v0 == v1))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Draws zero samples from a 42-row, 40-class zero logit table and asserts the output shape
    # is 42 by 0. Returns nothing.
    with self.session():
      with self.test_scope():
        x = random_ops.multinomial(
            array_ops.zeros([42, 40]), 0, output_dtype=dtypes.int32)
        y = self.evaluate(x)
        self.assertEqual(y.shape, (42, 0))

  def testEmptyStateless(self):
    # [FUNCTION DOC]
    # Stateless variant of the empty test: zero samples with a fed seed from a 42 by 40 zero
    # logit table, asserting the output shape is 42 by 0. Returns nothing.
    with self.session() as sess:
      with self.test_scope():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless_random_ops.stateless_multinomial(
            array_ops.zeros([42, 40]),
            0,
            seed=seed_t,
            output_dtype=dtypes.int32)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
        self.assertEqual(y.shape, (42, 0))



if __name__ == '__main__':
  googletest.main()
