# [MODULE DOC]
# Tests and a benchmark for the multinomial sampling op. Defines a reference sampler composed of
# existing ops (Gumbel noise plus argmax) and an alias for the native op, then tests: tiny-entropy
# distributions, independence between ops and between steps, same-seed reproducibility, bfloat16
# support, large and extreme logits, statistical correctness via a chi-square-like distance between
# composed and native frequencies, and empty-batch / empty-sample / empty-class edge cases. Also has
# a benchmark timing the native op against the composed one. Running as a script starts the tests.
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
"""Tests for Multinomial."""

import collections
import timeit

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


def composed_sampler(logits, num_samples):
  # [FUNCTION DOC]
  # HELPER: reference implementation for comparing against the native multinomial op.
  # Takes a logits tensor of shape batch by classes (with a fully known static shape) and the number
  # of samples to draw. Draws uniform noise of shape batch by classes by samples, turns it into
  # Gumbel noise, adds it to the logits expanded along a last axis, and returns the argmax over the
  # class axis: a tensor of shape batch by samples of class ids. Adds random ops to the graph.
  # [batch size, num classes, num samples]
  unif = random_ops.random_uniform(logits.get_shape().concatenate(
      tensor_shape.TensorShape([num_samples])))
  noise = -math_ops.log(-math_ops.log(unif))
  # [batch size, num classes, 1]
  logits = array_ops.expand_dims(logits, -1)

  # [batch size, num samples]
  return math_ops.argmax(logits + noise, axis=1)


native_sampler = random_ops.multinomial


class MultinomialTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the multinomial sampling op, covering edge cases, independence, seeding and
  # statistical correctness, plus helper methods for building ops, normalizing and measuring
  # frequency differences. Many tests use the GPU when available.

  @test_util.run_in_graph_and_eager_modes
  def testSmallEntropy(self):
    # [FUNCTION DOC]
    # Test (graph and eager): with a fixed global seed and logits strongly favouring one class per row
    # (minus ten versus ten), draws a thousand samples in both 32 and 64-bit output types. Asserts
    # each row consists entirely of the favoured class.
    random_seed.set_random_seed(1618)
    for output_dtype in [np.int32, np.int64]:
      with test_util.device(use_gpu=True):
        # A logit value of -10 corresponds to a probability of ~5e-5.
        logits = constant_op.constant([[-10., 10., -10.], [-10., -10., 10.]])
        num_samples = 1000
        samples = self.evaluate(random_ops.multinomial(
            logits, num_samples, output_dtype=output_dtype))
        self.assertAllEqual([[1] * num_samples, [2] * num_samples], samples)

  @test_util.run_deprecated_v1
  def testOneOpMultipleStepsIndependent(self):
    # [FUNCTION DOC]
    # Test (graph mode only): evaluates one sampling op twice and asserts the two results are not
    # identical, i.e. consecutive runs of the same op are independent.
    with test_util.use_gpu():
      sample_op1, _ = self._make_ops(10)
      # Consecutive runs shouldn't yield identical output.
      sample1a = self.evaluate(sample_op1)
      sample1b = self.evaluate(sample_op1)
      self.assertFalse(np.equal(sample1a, sample1b).all())

  def testEagerOneOpMultipleStepsIndependent(self):
    # [FUNCTION DOC]
    # Test (eager mode): evaluates two sampling ops built from the same distribution with no seed
    # and asserts their outputs are not identical.
    with context.eager_mode(), test_util.device(use_gpu=True):
      sample1, sample2 = self._make_ops(10)
      # Consecutive runs shouldn't yield identical output.
      self.assertFalse(np.equal(sample1.numpy(), sample2.numpy()).all())

  @test_util.run_deprecated_v1
  def testBfloat16(self):
    # [FUNCTION DOC]
    # Test (graph mode only): builds a sampling op on a bfloat16 distribution and evaluates it; the
    # test passes if no error is raised. No value is asserted.
    with test_util.use_gpu():
      sample_op1, _ = self._make_ops(10, dtype=dtypes.bfloat16)
      self.evaluate(sample_op1)

  def testEagerBfloat16(self):
    # [FUNCTION DOC]
    # Test (eager mode): builds and thereby runs a sampling op with a bfloat16 distribution; passes
    # if no error is raised. No value is asserted.
    with context.eager_mode(), test_util.device(use_gpu=True):
      self._make_ops(10, dtype=dtypes.bfloat16)

  def testTwoOpsIndependent(self):
    # [FUNCTION DOC]
    # Test: builds two unseeded ops with thirty-two samples each, evaluates them together, and
    # asserts they differ. Failure by chance is about one in two to the power of thirty-two.
    with test_util.use_gpu():
      sample_op1, sample_op2 = self._make_ops(32)
      sample1, sample2 = self.evaluate([sample_op1, sample_op2])
      # We expect sample1 and sample2 to be independent.
      # 1 in 2^32 chance of this assertion failing.
      self.assertFalse(np.equal(sample1, sample2).all())

  @test_util.run_deprecated_v1
  def testTwoOpsSameSeedDrawSameSequences(self):
    # [FUNCTION DOC]
    # Test (graph mode only): builds two ops with the same explicit op seed and asserts that both
    # draw identical sequences of a thousand samples.
    with test_util.use_gpu():
      sample_op1, sample_op2 = self._make_ops(1000, seed=1)
      sample1, sample2 = self.evaluate([sample_op1, sample_op2])
      self.assertAllEqual(sample1, sample2)

  def testLargeLogits(self):
    # [FUNCTION DOC]
    # Test: samples ten values from logits of plus and then minus one thousand over five classes
    # (to check numerical stability) and asserts all class ids are within zero to four.
    for neg in [True, False]:
      with test_util.use_gpu():
        logits = np.array([[1000.] * 5])
        if neg:
          logits *= -1
        samples = self.evaluate(random_ops.multinomial(logits, 10))
      # Sampled classes should be in-range.
      self.assertTrue((samples >= 0).all())
      self.assertTrue((samples < 5).all())

  def testSamplingCorrectness(self):
    # [FUNCTION DOC]
    # Test: seeds numpy for reproducibility and, for several distributions (two simple ones, a random
    # vector of ten classes, and a random batch of three by five), draws 21000 samples with both the
    # composed and the native samplers. Computes each sampler distance from the true probabilities
    # and from each other with the chi-square-like helper, and asserts every value is below 0.001.
    np.random.seed(1618)  # Make it reproducible.
    num_samples = 21000

    rand_probs = self._normalize(np.random.random_sample((10,)))
    rand_probs2 = self._normalize(np.random.random_sample((3, 5)))  # batched
    for probs in [[.5, .5], [.85, .05, .1], rand_probs, rand_probs2]:
      probs = np.asarray(probs)
      if len(probs.shape) == 1:
        probs = probs.reshape(1, probs.size)  # singleton batch

      logits = np.log(probs).astype(np.float32)
      composed_freqs = self._do_sampling(logits, num_samples, composed_sampler)
      native_freqs = self._do_sampling(logits, num_samples, native_sampler)

      # the test here is similar to core/lib/random/distribution_sampler_test.cc
      composed_chi2 = self._chi2(probs, composed_freqs)
      native_chi2 = self._chi2(probs, native_freqs)
      composed_native_chi2 = self._chi2(composed_freqs, native_freqs)

      def check(chi2s):
        # [FUNCTION DOC]
        # HELPER: assertion helper for testSamplingCorrectness. Takes a sequence of distance values and
        # asserts each is less than 0.001. Returns nothing.
        for chi2 in chi2s:
          self.assertLess(chi2, 1e-3)

      check(composed_chi2)
      check(native_chi2)
      check(composed_native_chi2)

  def _make_ops(self, num_samples, seed=None, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: builds the ops used by the independence tests.
    # Takes the number of samples, an optional op seed (default none) and an optional float dtype
    # (default 32-bit float). Builds a one-row distribution of four classes, takes its log, and
    # creates two separate multinomial ops from it. Returns a pair of the two sampling results.
    prob_dist = constant_op.constant([[0.15, 0.5, 0.3, 0.05]], dtype=dtype)
    logits = math_ops.log(prob_dist)
    # Two independent sets of samples from the same distribution
    sample_op1 = random_ops.multinomial(logits, num_samples, seed)
    sample_op2 = random_ops.multinomial(logits, num_samples, seed)
    return (sample_op1, sample_op2)

  def _normalize(self, vec):
    # [FUNCTION DOC]
    # HELPER: probability normalizer. Takes a numpy array that is either one-dimensional or a batch
    # (two-dimensional). Returns it divided by its sum: per row for a batch (rows keep their shape),
    # otherwise over the whole vector.
    batched = (len(vec.shape) == 2)
    return vec / vec.sum(axis=1, keepdims=True) if batched else vec / vec.sum()

  def _do_sampling(self, logits, num_samples, sampler):
    # [FUNCTION DOC]
    # HELPER: runs a sampler and measures empirical class frequencies.
    # Takes logits (numpy array batch by classes), the number of samples and a sampler function
    # (logits tensor and sample count to class-id tensor). Sets the global seed, evaluates the
    # sampler, asserts every drawn id is within zero to number of classes minus one, and returns a
    # list per batch row of the fraction of draws that hit each class.
    """Samples using the supplied sampler and inputs.

    Args:
      logits: Numpy ndarray of shape [batch_size, num_classes].
      num_samples: Int; number of samples to draw.
      sampler: A sampler function that takes (1) a [batch_size, num_classes]
        Tensor, (2) num_samples and returns a [batch_size, num_samples] Tensor.

    Returns:
      Frequencies from sampled classes; shape [batch_size, num_classes].
    """
    with test_util.use_gpu():
      random_seed.set_random_seed(1618)
      op = sampler(constant_op.constant(logits), num_samples)
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

  def _chi2(self, expected, actual):
    # [FUNCTION DOC]
    # HELPER: distance measure. Takes expected and actual frequency arrays of equal shape, and
    # returns the sum over the first axis of squared differences divided by the expected value
    # (a chi-square-like statistic, not scaled by sample count), as an array per class.
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = actual - expected
    chi2 = np.sum(diff * diff / expected, axis=0)
    return chi2

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: samples from zero logits of five classes for batch sizes zero and three and sample
    # counts zero and seven; asserts the output shape is batch by samples in every combination.
    classes = 5
    with test_util.use_gpu():
      for batch in 0, 3:
        for samples in 0, 7:
          x = self.evaluate(
              random_ops.multinomial(
                  array_ops.zeros([batch, classes]), samples))
          self.assertEqual(x.shape, (batch, samples))

  @test_util.run_deprecated_v1
  def testEmptyClasses(self):
    # [FUNCTION DOC]
    # Test (graph mode only): builds a multinomial op with zero classes and asserts that evaluating
    # it raises an op error saying the number of classes should be positive.
    with test_util.use_gpu():
      x = random_ops.multinomial(array_ops.zeros([5, 0]), 7)
      with self.assertRaisesOpError("num_classes should be positive"):
        self.evaluate(x)

  def testNegativeMinLogits(self):
    # [FUNCTION DOC]
    # Test: with a fixed seed, uses 1024 classes where 1023 have the smallest 32-bit float and the
    # last has zero; draws a thousand samples and asserts all choose the last class, so minimum
    # logits do not overflow.
    random_seed.set_random_seed(78844)
    with test_util.use_gpu():
      logits = constant_op.constant([[np.finfo(np.float32).min] * 1023 + [0]])
      num_samples = 1000
      samples = self.evaluate(random_ops.multinomial(logits, num_samples))
      self.assertAllEqual([[1023] * num_samples], samples)


# Benchmarking code
def native_op_vs_composed_ops(batch_size, num_classes, num_samples, num_iters):
  # [FUNCTION DOC]
  # HELPER: timing function for the benchmark.
  # Takes batch size, number of classes, number of samples and number of timing iterations. Seeds
  # numpy, makes random normal logits, builds a session with optimizations off, builds grouped
  # native and composed sampling ops, and times repeated runs of each. Returns a pair of total
  # seconds for the native op and the composed op.
  np.random.seed(1618)  # Make it reproducible.
  shape = [batch_size, num_classes]
  logits_np = np.random.randn(*shape).astype(np.float32)

  # No CSE/CF.
  optimizer_options = config_pb2.OptimizerOptions(
      opt_level=config_pb2.OptimizerOptions.L0)
  config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(
      optimizer_options=optimizer_options))

  with session.Session(config=config) as sess:
    logits = constant_op.constant(logits_np, shape=shape)
    native_op = control_flow_ops.group(native_sampler(logits, num_samples))
    composed_op = control_flow_ops.group(composed_sampler(logits, num_samples))

    native_dt = timeit.timeit(lambda: sess.run(native_op), number=num_iters)
    composed_dt = timeit.timeit(lambda: sess.run(composed_op), number=num_iters)
    return native_dt, composed_dt


class MultinomialBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class comparing the native multinomial op against the composed implementation.

  def benchmarkNativeOpVsComposedOps(self):
    # [FUNCTION DOC]
    # Benchmark: for batch sizes 32 and 128, class counts 10000 and 100000 and sample counts 1, 4 and
    # 32, times both implementations for fifty iterations, prints a tab-separated table row with
    # the speedup, and reports two benchmark entries (native and composed) with wall time. Takes no
    # inputs and returns nothing; side effects are printing and benchmark reporting.
    num_iters = 50
    print("Composition of existing ops vs. Native Multinomial op [%d iters]" %
          num_iters)
    print("BatchSize\tNumClasses\tNumSamples\tsec(composed)\tsec(native)\t"
          "speedup")

    for batch_size in [32, 128]:
      for num_classes in [10000, 100000]:
        for num_samples in [1, 4, 32]:
          n_dt, c_dt = native_op_vs_composed_ops(batch_size, num_classes,
                                                 num_samples, num_iters)
          print("%d\t%d\t%d\t%.3f\t%.3f\t%.2f" % (batch_size, num_classes,
                                                  num_samples, c_dt, n_dt,
                                                  c_dt / n_dt))

          self.report_benchmark(
              name="native_batch%d_classes%d_s%d" %
              (batch_size, num_classes, num_samples),
              iters=num_iters,
              wall_time=n_dt)
          self.report_benchmark(
              name="composed_batch%d_classes%d_s%d" %
              (batch_size, num_classes, num_samples),
              iters=num_iters,
              wall_time=c_dt)


if __name__ == "__main__":
  test.main()
