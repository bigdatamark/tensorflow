# [MODULE DOC]
# Kernel tests for the binomial random sampling ops: the stateful generator method and the
# stateless random binomial function. Covers statistical moment matching against scipy,
# seed reproducibility, stateless determinism, output shape broadcasting, corner-case
# parameters (invalid or extreme probabilities) and int64 shape tensors.
# Defines the supported dtype tuple used by the tests and a single test class; running the file
# as a script starts the TensorFlow test runner. No import-time side effects beyond imports.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.stateful_random_ops.binomial."""
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

# All supported dtypes for binomial().
_SUPPORTED_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64,
                     dtypes.int32, dtypes.int64)


class RandomBinomialTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering binomial sampling (stateful generator and stateless variants).
  # Holds no instance state; provides a sampler-factory helper used by the moment and seed tests.
  # Statistical tests use scipy when it is installed and silently return otherwise.
  """This is a large test due to the moments computation taking some time."""

  def _Sampler(
      self, num, counts, probs, dtype, gen=None, sample_shape=None, seed=None):
    # [FUNCTION DOC]
    # HELPER: builds sampling callables for the moment and seed tests in this class.
    # Inputs: num (base sample count; the default sample shape is ten times it), counts and probs
    # (binomial parameters, scalars or arrays), dtype (output dtype), gen (optional existing
    # generator to reuse), sample_shape (optional explicit shape overriding the default), seed
    # (used to create a fresh generator only when gen is not given).
    # Returns a zero-argument callable that draws binomial samples from the generator.
    # No side effects until the callable is invoked, which advances the generator state.
    def func():
      # [FUNCTION DOC]
      # HELPER: inner closure returned by the sampler factory.
      # Takes no arguments. Picks the sample shape (explicit or ten times num), uses the supplied
      # generator or creates one from the seed, and returns a tensor of binomial samples with the
      # captured counts, probs and dtype. Advances the generator state.
      shape = [10 * num] if sample_shape is None else sample_shape
      generator = gen if gen is not None else (
          stateful_random_ops.Generator.from_seed(seed))
      return generator.binomial(
          shape=shape, counts=counts, probs=probs, dtype=dtype)

    return func

  @test_util.run_v2_only
  def testMoments(self):
    # [FUNCTION DOC]
    # Checks that sampled binomial moments (up to six) match scipy's binomial distribution.
    # Runs only in v2 mode. Uses one seeded generator, loops over all supported dtypes, strides,
    # counts and probabilities, draws 500000 samples as float64 and asserts all z-scores are
    # below 6. Returns early with a logged warning if scipy cannot be imported.
    try:
      from scipy import stats  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      tf_logging.warn("Cannot test moments: %s", e)
      return
    # The moments test is a z-value test.  This is the largest z-value
    # we want to tolerate. Since the z-test approximates a unit normal
    # distribution, it should almost definitely never exceed 6.
    z_limit = 6.0
    gen = stateful_random_ops.Generator.from_seed(seed=12345)
    for dt in _SUPPORTED_DTYPES:
      # Test when n * p > 10, and n * p < 10
      for stride in 0, 4, 10:
        for counts in (1., 10., 22., 50.):
          for prob in (0.1, 0.5, 0.8):
            sampler = self._Sampler(int(5e4), counts, prob, dt, gen=gen)
            z_scores = util.test_moment_matching(
                # Use float64 samples.
                self.evaluate(sampler()).astype(np.float64),
                number_moments=6,
                dist=stats.binom(counts, prob),
                stride=stride,
            )
            self.assertAllLess(z_scores, z_limit)

  @test_util.run_v2_only
  def testSeed(self):
    # [FUNCTION DOC]
    # Verifies that two samplers created with the same seed produce identical samples.
    # Runs only in v2 mode; for float16, float32 and float64 draws 10000 samples from each of two
    # fresh generators with equal seeds and asserts the results are exactly equal.
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sx = self._Sampler(1000, counts=10., probs=0.4, dtype=dt, seed=345)
      sy = self._Sampler(1000, counts=10., probs=0.4, dtype=dt, seed=345)
      self.assertAllEqual(self.evaluate(sx()), self.evaluate(sy()))

  def testStateless(self):
    # [FUNCTION DOC]
    # Verifies determinism of the stateless binomial op for float16, float32 and float64.
    # Builds two op instances with the same seed pair, evaluates each twice, and asserts that all
    # four results are identical (repeat evaluation and equal seeds give equal output).
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sx = stateless_random_ops.stateless_random_binomial(
          shape=[1000], seed=[12, 34], counts=10., probs=0.4, output_dtype=dt)
      sy = stateless_random_ops.stateless_random_binomial(
          shape=[1000], seed=[12, 34], counts=10., probs=0.4, output_dtype=dt)
      sx0, sx1 = self.evaluate(sx), self.evaluate(sx)
      sy0, sy1 = self.evaluate(sy), self.evaluate(sy)
      self.assertAllEqual(sx0, sx1)
      self.assertAllEqual(sx0, sy0)
      self.assertAllEqual(sy0, sy1)

  def testZeroShape(self):
    # [FUNCTION DOC]
    # Checks that requesting a zero-length output shape with empty counts and probs works and
    # yields a tensor whose static shape is a single dimension of size zero.
    rnd = stateful_random_ops.Generator.from_seed(12345).binomial([0], [], [])
    self.assertEqual([0], rnd.shape.as_list())

  def testShape(self):
    # [FUNCTION DOC]
    # Checks the static output shape of the generator's binomial method under broadcasting.
    # Cases: scalar parameters with vector and empty shapes, matching vector parameters,
    # scalar-with-vector mixes in both directions, and multi-dimensional parameters that broadcast
    # against the requested shape. Asserts each result's static shape equals the requested shape.
    rng = stateful_random_ops.Generator.from_seed(12345)
    # Scalar parameters.
    rnd = rng.binomial(shape=[10], counts=np.float32(2.), probs=np.float32(0.5))
    self.assertEqual([10], rnd.shape.as_list())
    rnd = rng.binomial(shape=[], counts=np.float32(2.), probs=np.float32(0.5))
    self.assertEqual([], rnd.shape.as_list())

    # Vector parameters.
    rnd = rng.binomial(
        shape=[10],
        counts=array_ops.ones([10], dtype=np.float32),
        probs=0.3 * array_ops.ones([10], dtype=np.float32))
    self.assertEqual([10], rnd.shape.as_list())
    rnd = rng.binomial(
        shape=[5, 2],
        counts=array_ops.ones([2], dtype=np.float32),
        probs=0.4 * array_ops.ones([2], dtype=np.float32))
    self.assertEqual([5, 2], rnd.shape.as_list())

    # Scalar counts, vector probs.
    rnd = rng.binomial(
        shape=[10],
        counts=np.float32(5.),
        probs=0.8 * array_ops.ones([10], dtype=np.float32))
    self.assertEqual([10], rnd.shape.as_list())

    # Vector counts, scalar probs.
    rnd = rng.binomial(
        shape=[10],
        counts=array_ops.ones([10], dtype=np.float32),
        probs=np.float32(0.9))
    self.assertEqual([10], rnd.shape.as_list())

    # Tensor parameters
    rnd = rng.binomial(
        shape=[10, 2, 3],
        counts=array_ops.ones([2, 1], dtype=np.float32),
        probs=0.9 * array_ops.ones([1, 3], dtype=np.float32))
    self.assertEqual([10, 2, 3], rnd.shape.as_list())

    # Tensor parameters
    rnd = rng.binomial(
        shape=[10, 2, 3, 5],
        counts=array_ops.ones([2, 1, 5], dtype=np.float32),
        probs=0.9 * array_ops.ones([1, 3, 1], dtype=np.float32))
    self.assertEqual([10, 2, 3, 5], rnd.shape.as_list())

  @test_util.run_v2_only
  def testCornerCases(self):
    # [FUNCTION DOC]
    # Checks binomial sampling at degenerate parameters (v2 only). Probability zero gives zero,
    # probability one gives the count, a not-a-number probability gives not-a-number when the count
    # is positive, and zero counts with out-of-range or not-a-number probabilities give zero.
    # Asserts the six samples equal the expected array exactly.
    rng = stateful_random_ops.Generator.from_seed(12345)
    counts = np.array([5, 5, 5, 0, 0, 0], dtype=np.float32)
    probs = np.array([0, 1, float("nan"), -10, 10, float("nan")],
                     dtype=np.float32)
    expected = np.array([0, 5, float("nan"), 0, 0, 0], dtype=np.float32)
    result = rng.binomial(
        shape=[6], counts=counts, probs=probs, dtype=np.float32)
    self.assertAllEqual(expected, self.evaluate(result))

  @test_util.run_v2_only
  def testMomentsForTensorInputs(self):
    # [FUNCTION DOC]
    # Moment-matching test with tensor-valued (broadcasting) counts and probabilities (v2 only).
    # For each dtype and stride, draws a large sample with random integer counts and uniform
    # probabilities, compares up to six moments against a scipy-based wrapper, and asserts all
    # z-scores are below 6. Returns early with a warning if scipy is unavailable.
    try:
      from scipy import stats  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      tf_logging.warn("Cannot test moments: %s", e)
      return
    # The moments test is a z-value test.  This is the largest z-value
    # we want to tolerate. Since the z-test approximates a unit normal
    # distribution, it should almost definitely never exceed 6.
    z_limit = 6.0

    class ScipyBinomialWrapper(object):
      # [CLASS DOC]
      # HELPER: adapter letting the moment-matching utility query scipy binomial moments for
      # broadcast arrays of parameters. Stores the counts and probs arrays; exposes a moment method.
      """Wrapper for stats.binom to support broadcasting."""

      def __init__(self, counts, probs):
        # [FUNCTION DOC]
        # HELPER: constructor storing the counts and probs arrays (any broadcast-compatible
        # shapes) for later moment computation. Returns nothing.
        self.counts = counts
        self.probs = probs

      def moment(self, i):
        # [FUNCTION DOC]
        # HELPER: computes the i-th raw moment of the binomial distribution for every element of
        # the broadcast of counts and probs. Input i is the moment order. Returns a float64 array
        # with the broadcast shape. Evaluates scipy's distribution one element at a time.
        counts, probs = np.broadcast_arrays(self.counts, self.probs)
        broadcast_shape = counts.shape

        counts = np.reshape(counts, (-1,))
        probs = np.reshape(probs, (-1,))
        counts_and_probs = np.stack([counts, probs], axis=-1)
        moments = np.fromiter(
            (stats.binom(cp[0], cp[1]).moment(i) for cp in counts_and_probs),
            dtype=np.float64)
        return np.reshape(moments, broadcast_shape)

    gen = stateful_random_ops.Generator.from_seed(seed=23455)
    for dt in _SUPPORTED_DTYPES:
      # Test when n * p > 10, and n * p < 10
      for stride in 0, 4, 10:
        counts = np.float64(np.random.randint(low=1, high=20, size=(2, 1, 4)))
        probs = np.random.uniform(size=(1, 3, 4))

        sampler = self._Sampler(
            int(5e4),
            counts,
            probs,
            dt,
            gen=gen,
            sample_shape=[10 * int(5e4), 2, 3, 4])
        # Use float64 samples.
        samples = self.evaluate(sampler()).astype(np.float64)
        z_scores = util.test_moment_matching(
            samples,
            number_moments=6,
            dist=ScipyBinomialWrapper(counts, probs),
            stride=stride,
        )
        self.assertAllLess(z_scores, z_limit)

  def testStatelessDtypeInt64(self):
    # [FUNCTION DOC]
    # Checks the stateless binomial op accepts a shape given as an int64 tensor. Requests a
    # single float16 sample with fixed seed, counts ten and probability 0.4, and asserts the
    # result equals the value five (a regression-style exact-value check).
    srb = stateless_random_ops.stateless_random_binomial(
        shape=[constant_op.constant(1, dtype=dtypes.int64)],
        seed=[12, 34],
        counts=10.0,
        probs=0.4,
        output_dtype=dtypes.float16)
    out = self.evaluate(srb)
    self.assertEqual(out, [5.0])


if __name__ == "__main__":
  test.main()
