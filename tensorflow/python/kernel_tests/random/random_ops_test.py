# [MODULE DOC]
# Kernel tests for the basic random ops: normal, truncated normal and uniform sampling (floating
# point and integer), plus static shape inference and deterministic-mode behavior. Covers
# distinctness of repeated draws, CPU versus GPU agreement, seed reproducibility, op and graph
# seed handling in a single session, no merging of identical random ops, value ranges,
# distribution checks, error cases (invalid minval/maxval, missing seed when determinism is
# enabled) and a very large shape. Defines a float-dtype helper function and several test classes.
# Running as a script starts the TensorFlow test runner; no import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.random_ops."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def get_float_types():
  # [FUNCTION DOC]
  # HELPER: supplies the list of floating-point dtypes the tests iterate over.
  # Takes no inputs. Returns a list with float16, float32 and float64, plus bfloat16 only when a
  # CUDA GPU with compute capability 8.0 or higher is available. Queries the GPU configuration.
  float_types = [dtypes.float16, dtypes.float32, dtypes.float64]
  if test_util.is_gpu_available(
      cuda_only=True, min_cuda_compute_capability=(8, 0)):
    float_types += [dtypes.bfloat16]
  return float_types

class RandomOpTestCommon(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the normal and uniform test classes. Provides a common check
  # that repeated evaluation of a random op in one session gives varying results. Contains no
  # tests of its own.

  # Checks that executing the same rng_func multiple times rarely produces the
  # same result.
  def _testSingleSessionNotConstant(self,
                                    rng_func,
                                    num,
                                    dtype,
                                    min_or_mean,
                                    max_or_stddev,
                                    use_gpu,
                                    op_seed=None,
                                    graph_seed=None):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the single-session non-constant tests.
    # Inputs: rng_func (a random-op function taking a shape, two distribution parameters, a dtype
    # and a seed), num (number of values), dtype, min_or_mean and max_or_stddev (the two
    # distribution parameters), use_gpu (whether the session may use the GPU), op_seed (optional
    # per-op seed) and graph_seed (optional graph-level seed set before building the op).
    # Builds the op in a fresh graph, evaluates it three times and asserts that at least one pair
    # of results is not exactly equal. Side effect: may set the graph-level random seed.
    # KNOWN ISSUE: the session handle is bound but never used.
    with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
      if graph_seed is not None:
        random_seed.set_random_seed(graph_seed)
      x = rng_func([num], min_or_mean, max_or_stddev, dtype=dtype, seed=op_seed)

      y = self.evaluate(x)
      z = self.evaluate(x)
      w = self.evaluate(x)

      # We use exact equality here. If the random-number generator is producing
      # the same output, all three outputs will be bitwise identical.
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or (not np.array_equal(y, w)))


class RandomNormalTest(RandomOpTestCommon):
  # [CLASS DOC]
  # Tests for the random normal op: distinctness of repeated draws, CPU/GPU agreement, seed
  # reproducibility, no merging of identical ops, and non-constant output across repeated
  # evaluations with no seed, an op seed or a graph seed. Uses the sampler helper to draw
  # ten rows of samples at a time.

  def _Sampler(self, num, mu, sigma, dtype, use_gpu, seed=None):
    # [FUNCTION DOC]
    # HELPER: builds sampling callables for the normal tests.
    # Inputs: num (samples per row), mu (mean), sigma (standard deviation), dtype, use_gpu
    # (whether the session may use the GPU) and seed (optional op seed). Returns a zero-argument
    # callable that builds a fresh graph and session, evaluates the normal op ten times and
    # returns a ten-by-num numpy array.

    def func():
      # [FUNCTION DOC]
      # HELPER: inner closure of the sampler factory; takes no inputs. In a new graph and
      # session, builds one normal op, evaluates it ten times into the rows of an array and
      # returns the array.
      # KNOWN ISSUE: the session handle is bound but never used.
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_normal(
            [num], mean=mu, stddev=sigma, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in range(10):
          ret[i, :] = self.evaluate(rng)
      return ret

    return func

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    # [FUNCTION DOC]
    # Checks that two successive draws of 1000 normal samples differ almost everywhere, for every
    # float dtype on the GPU-capable session. Asserts fewer than ten equal entries; prints the
    # arrays and the count when the limit is hit (output side effect).
    for dt in get_float_types():
      sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True)
      x = sampler()
      y = sampler()
      # Number of different samples.
      count = (x == y).sum()
      if count >= 10:
        print("x = ", x)
        print("y = ", y)
        print("count = ", count)
      self.assertTrue(count < 10)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  @test_util.run_deprecated_v1
  def testCPUGPUMatch(self):
    # [FUNCTION DOC]
    # Checks CPU and GPU normal sampling with the same seed give nearly the same million values
    # (graph mode). Tolerance is 1e-6, looser for float16 (1e-3) and bfloat16 (0.1).
    for dt in get_float_types():
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(
            1000000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      rtol = atol = 1e-6
      if dt == dtypes.float16:
        rtol = atol = 1e-3
      elif dt == dtypes.bfloat16:
        rtol = atol = 1e-1
      self.assertAllClose(results[False], results[True], rtol=rtol, atol=atol)

  @test_util.run_deprecated_v1
  def testSeed(self):
    # [FUNCTION DOC]
    # Checks that two samplers with the same op seed give identical normal samples for each float
    # dtype (graph mode).
    for dt in get_float_types():
      sx = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True, seed=345)
      sy = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True, seed=345)
      self.assertAllEqual(sx(), sy())

  @test_util.run_deprecated_v1
  def testNoCSE(self):
    # [FUNCTION DOC]
    # Checks two identical unseeded normal ops in one graph are not merged: on CPU and GPU
    # sessions, asserts the norm of their difference exceeds 0.1 (graph mode).
    for use_gpu in [False, True]:
      with self.session(use_gpu=use_gpu):
        shape = [2, 3, 4]
        rnd1 = random_ops.random_normal(shape, 0.0, 1.0, dtypes.float32)
        rnd2 = random_ops.random_normal(shape, 0.0, 1.0, dtypes.float32)
        diff = rnd2 - rnd1
        self.assertTrue(np.linalg.norm(diff.eval()) > 0.1)

  @test_util.run_deprecated_v1
  def testSingleSessionNotConstant(self):
    # [FUNCTION DOC]
    # With no seeds, repeated evaluation of the normal op (mean zero, stddev one, 100 values)
    # must not always return the same values; runs for CPU and GPU and every float dtype.
    for use_gpu in [False, True]:
      for dt in get_float_types():
        self._testSingleSessionNotConstant(
            random_ops.random_normal, 100, dt, 0.0, 1.0, use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testSingleSessionOpSeedNotConstant(self):
    # [FUNCTION DOC]
    # Same as the unseeded single-session test but with a fixed per-op seed: repeated evaluations
    # of the single op must still produce varying values (the seed fixes the stream, not the value).
    for use_gpu in [False, True]:
      for dt in get_float_types():
        self._testSingleSessionNotConstant(
            random_ops.random_normal,
            100,
            dt,
            0.0,
            1.0,
            use_gpu=use_gpu,
            op_seed=1345)

  @test_util.run_deprecated_v1
  def testSingleSessionGraphSeedNotConstant(self):
    # [FUNCTION DOC]
    # Same as the unseeded single-session test but with a graph-level seed set before building the
    # op: repeated evaluations must still produce varying values.
    for use_gpu in [False, True]:
      for dt in get_float_types():
        self._testSingleSessionNotConstant(
            random_ops.random_normal,
            100,
            dt,
            0.0,
            1.0,
            use_gpu=use_gpu,
            graph_seed=965)


@test_util.with_eager_op_as_function
class TruncatedNormalTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the truncated normal op (run with eager ops executed as functions): distinctness,
  # CPU/GPU agreement, seeding, effective standard deviation, recovery after an error under
  # determinism, a huge static shape, no merging of identical ops and eager global seeding.

  def _Sampler(self, num, mu, sigma, dtype, use_gpu, seed=None):
    # [FUNCTION DOC]
    # HELPER: builds sampling callables for the truncated normal tests.
    # Inputs: num (samples per row), mu (mean), sigma (requested standard deviation), dtype,
    # use_gpu and seed (optional op seed). Returns a zero-argument callable that evaluates the op
    # ten times in a fresh graph and session and returns a ten-by-num numpy array.

    def func():
      # [FUNCTION DOC]
      # HELPER: inner closure of the sampler factory; takes no inputs. Builds one truncated normal
      # op in a new graph and session, evaluates it ten times into the rows of an array and
      # returns it.
      # KNOWN ISSUE: the session handle is bound but never used.
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.truncated_normal(
            [num], mean=mu, stddev=sigma, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in range(10):
          ret[i, :] = self.evaluate(rng)
      return ret

    return func

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    # [FUNCTION DOC]
    # Checks two successive draws of 1000 truncated normal samples differ almost everywhere.
    # KNOWN ISSUE: the whole check only runs when no GPU is available (the kernel is noted as
    # unsupported on GPU), so on GPU machines the test silently does nothing.
    # Uses CPU sessions for each float dtype and asserts fewer than ten equal entries; prints
    # diagnostics when the limit is hit.
    # NOTE: TruncatedNormal on GPU is not supported.
    if not test.is_gpu_available():
      for dt in get_float_types():
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=False)
        x = sampler()
        y = sampler()
        # Number of different samples.
        count = (x == y).sum()
        if count >= 10:
          print("x = ", x)
          print("y = ", y)
          print("count = ", count)
        self.assertTrue(count < 10)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  @test_util.run_deprecated_v1
  def testCPUGPUMatch(self):
    # [FUNCTION DOC]
    # Checks CPU and GPU truncated normal sampling with one seed agree on a million values per
    # float dtype (graph mode); the large count exercises multiple GPU rounds. Returns without
    # checking anything when no GPU is available. Tolerances as in the normal test.
    # Skip the test if there is no GPU.
    if not test.is_gpu_available():
      return

    for dt in get_float_types():
      results = {}
      for use_gpu in [False, True]:
        # We need a particular larger number of samples to test multiple rounds
        # on GPU
        sampler = self._Sampler(
            1000000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      atol = rtol = 1e-6
      if dt == dtypes.float16:
        atol = rtol = 1e-3
      if dt == dtypes.bfloat16:
        atol = rtol = 1e-1
      self.assertAllClose(results[False], results[True], rtol=rtol, atol=atol)

  @test_util.run_deprecated_v1
  def testSeed(self):
    # [FUNCTION DOC]
    # Checks two samplers with the same op seed give identical truncated normal samples for each
    # float dtype (graph mode, GPU allowed).
    for dt in get_float_types():
      sx = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True, seed=345)
      sy = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=True, seed=345)
      self.assertAllEqual(sx(), sy())

  # The effective standard deviation of truncated normal is 85% of the
  # requested one.
  def testStdDev(self):
    # [FUNCTION DOC]
    # Checks the empirical standard deviation of 1,000,000 truncated normal samples with requested
    # stddev three is about 85 percent of the request: asserts the ratio is within 0.04 of 0.85,
    # for every float dtype. Prints the measured values.
    for dt in get_float_types():
      stddev = 3.0
      sampler = self._Sampler(100000, 0.0, stddev, dt, use_gpu=True)
      x = sampler()
      print("std(x)", np.std(x), abs(np.std(x) / stddev - 0.85))
      self.assertLess(abs(np.std(x) / stddev - 0.85), 0.04)

  def testSuccessAfterError(self):
    # [FUNCTION DOC]
    # Checks the truncated normal kernel still works after it has raised an error. Enables
    # global op determinism, asserts the raw op without a seed raises an invalid-argument error
    # mentioning that a seed must be specified, disables determinism, then reruns the standard
    # deviation test.
    # KNOWN ISSUE: determinism is disabled by a plain call after the assertion, not in a cleanup
    # block, so a failure of the assertion would leave global determinism enabled for later tests.
    # Side effect: toggles the process-wide determinism setting.
    # Force an error on the TruncatedNormal kernel.
    config.enable_op_determinism()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "When determinism is enabled, random ops must have a seed specified"):
      self.evaluate(gen_random_ops.truncated_normal((1,), dtypes.float32))
    config.disable_op_determinism()

    # Ensure the StdDev of the TruncatedNormal works as intended.
    self.testStdDev()

  @test_util.run_deprecated_v1
  def testLargeShape(self):
    # [FUNCTION DOC]
    # Checks static shape handling for a shape beyond 32-bit element counts. Creates a variable of
    # 2^33 by 1 zeros (graph mode, never initialized or run), builds a truncated normal op from its
    # shape and asserts the inferred static shape is that large shape.
    with self.session():
      v = variables.Variable(
          array_ops.zeros(dtype=dtypes.float32, shape=[2**33, 1]))
      n = random_ops.truncated_normal(v.shape)
      self.assertEqual([8589934592, 1], n.shape.as_list())

  @test_util.run_deprecated_v1
  def testNoCSE(self):
    # [FUNCTION DOC]
    # Checks two identical unseeded truncated normal ops in one graph are not merged: asserts the
    # norm of their difference exceeds 0.1 (graph mode).
    with self.session():
      shape = [2, 3, 4]
      rnd1 = random_ops.truncated_normal(shape, 0.0, 1.0, dtypes.float32)
      rnd2 = random_ops.truncated_normal(shape, 0.0, 1.0, dtypes.float32)
      diff = rnd2 - rnd1
      self.assertTrue(np.linalg.norm(diff.eval()) > 0.1)

  def testEagerSeed(self):
    # [FUNCTION DOC]
    # Checks that setting the same global seed in eager mode twice reproduces the same draw.
    # Forces context creation, sets the global seed to 42, draws a scalar, resets the seed, draws
    # again and asserts equality. Side effect: changes the eager context's global seed.
    # KNOWN ISSUE: despite living in the truncated normal class, it samples from the plain normal op.
    with context.eager_mode():
      # Ensure a context has been created
      random_ops.random_normal([])
      # Set the same seed twice and check that the values match
      context.set_global_seed(42)
      rnd1 = random_ops.random_normal([])
      context.set_global_seed(42)
      rnd2 = random_ops.random_normal([])
      self.assertAllEqual(rnd1, rnd2)


@test_util.with_eager_op_as_function
@test_util.for_all_test_methods(test_util.disable_xla,
                                "This never passed on XLA")
class RandomUniformTest(RandomOpTestCommon):
  # [CLASS DOC]
  # Tests for the random uniform op over float and integer dtypes (XLA is disabled for every
  # test method, and eager ops run as functions): value range, distinctness, invalid bounds,
  # uniformity of integers, degenerate bounds, CPU/GPU agreement, seeds (including huge seed
  # values), no merging of identical ops, and non-constant output in a single session.

  def _Sampler(self, num, minv, maxv, dtype, use_gpu, seed=None):
    # [FUNCTION DOC]
    # HELPER: builds sampling callables for the uniform tests.
    # Inputs: num (samples per row), minv and maxv (inclusive lower and exclusive upper bound),
    # dtype, use_gpu and seed (optional op seed). Returns a zero-argument callable that evaluates
    # the uniform op ten times in a fresh graph and session and returns a ten-by-num array.

    def func():
      # [FUNCTION DOC]
      # HELPER: inner closure of the sampler factory; takes no inputs. Builds one uniform op in a
      # new graph and session, evaluates it ten times into array rows and returns the array.
      # Errors from the op (such as equal bounds with a nonzero count) propagate to the caller.
      # KNOWN ISSUE: the session handle is bound but never used.
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_uniform(
            [num], minval=minv, maxval=maxv, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in range(10):
          ret[i, :] = self.evaluate(rng)
      return ret

    return func

  def testRange(self):
    # [FUNCTION DOC]
    # Checks samples from bounds minus two (inclusive) to eight (exclusive) stay within range,
    # for every float dtype plus int32 and int64: minimum at least -2 and maximum below 8.
    for dt in get_float_types() + [dtypes.int32, dtypes.int64]:
      sampler = self._Sampler(1000, minv=-2, maxv=8, dtype=dt, use_gpu=True)
      x = sampler()
      self.assertTrue(-2 <= np.min(x))
      self.assertTrue(np.max(x) < 8)

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    # [FUNCTION DOC]
    # Checks two successive draws of 1000 uniform values differ almost everywhere, for float and
    # integer dtypes (floats on zero to one, ints up to 2^30). Allowed equal entries: 10 by
    # default, 50 for float16, 90 for bfloat16. Prints diagnostics when the limit is hit.
    for dt in get_float_types() + [dtypes.int32, dtypes.int64]:
      maxv = 1.0 if dt.is_floating else 1 << 30
      sampler = self._Sampler(1000, minv=0, maxv=maxv, dtype=dt, use_gpu=True)
      x = sampler()
      y = sampler()
      count = (x == y).sum()
      count_limit = 10
      if dt == dtypes.float16:
        count_limit = 50
      elif dt == dtypes.bfloat16:
        count_limit = 90
      if count >= count_limit:
        print("x = ", x)
        print("y = ", y)
        print("count = ", count)
      self.assertTrue(count < count_limit)

  @test_util.run_deprecated_v1
  def testUniformIntsWithInvalidShape(self):
    # [FUNCTION DOC]
    # Error case (graph mode): for int32 and int64, passing a non-scalar minval or maxval must
    # raise a value error whose message says the bound must be a scalar.
    for dtype in dtypes.int32, dtypes.int64:
      with self.assertRaisesRegex(
          ValueError, "minval must be a scalar; got a tensor of shape"):
        random_ops.random_uniform(
            [1000], minval=[1, 2], maxval=3, dtype=dtype)
      with self.assertRaisesRegex(
          ValueError, "maxval must be a scalar; got a tensor of shape"):
        random_ops.random_uniform(
            [1000], minval=1, maxval=[2, 3], dtype=dtype)

  # Check that uniform ints actually follow a uniform distribution.
  @test_util.run_deprecated_v1
  def testUniformInts(self):
    # [FUNCTION DOC]
    # Statistical test that integer samples are uniform. Draws 100000 seeded ints in the range
    # minus two to fifteen (exclusive) for int32 and int64, histograms them into one bin per
    # value, asserts the shapes and total count, and that every bin deviates from the expected
    # count by less than five binomial standard deviations (graph mode).
    minv = -2
    maxv = 15
    n = 100000
    p = 1 / (maxv - minv)
    # The counts should follow an (n, p) binomial distribution.
    mean = p * n
    std = np.sqrt(n * p * (1 - p))
    for dt in dtypes.int32, dtypes.int64:
      # Use a fixed seed here to make the test deterministic.
      # Without the fixed seed, the 5 * std bound will (very rarely) fail.
      sampler = self._Sampler(
          n // 10, minv=minv, maxv=maxv, dtype=dt, use_gpu=True, seed=17)
      x = sampler().ravel()
      self.assertEqual(x.shape, (n,))
      counts, _ = np.histogram(x, bins=maxv - minv)
      self.assertEqual(counts.shape, (maxv - minv,))
      self.assertEqual(counts.sum(), n)
      error = np.abs(counts - mean)
      self.assertLess(error.max(), 5 * std)

  # Check that minval = maxval is fine iff we're producing no numbers
  def testUniformIntsDegenerate(self):
    # [FUNCTION DOC]
    # For int32 and int64 with equal bounds: requesting zero values per row is fine and gives a
    # ten-by-zero array, while requesting one value raises an op error saying the minimum must be
    # less than the maximum.
    for dt in dtypes.int32, dtypes.int64:
      def sample(n):
        # [FUNCTION DOC]
        # HELPER: draws ten rows of n uniform integers with both bounds equal to zero using the
        # current dtype from the enclosing loop. Input n is the number of values per row; returns
        # the numpy array of samples; raises the op error when n is positive.
        return self._Sampler(n, minv=0, maxv=0, dtype=dt, use_gpu=True)()
      self.assertEqual(sample(0).shape, (10, 0))
      with self.assertRaisesOpError('Need minval < maxval, got 0 >= 0'):
        sample(1)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  @test_util.run_deprecated_v1
  def testCPUGPUMatch(self):
    # [FUNCTION DOC]
    # Checks CPU and GPU uniform sampling with one seed give exactly equal results (no tolerance)
    # for float and integer dtypes, a million values each (graph mode). Floats use zero to one,
    # integers zero to seventeen.
    for dt in get_float_types() + [dtypes.int32, dtypes.int64]:
      maxv = 1.0 if dt.is_floating else 17
      results = {}
      for use_gpu in False, True:
        sampler = self._Sampler(
            1000000, minv=0, maxv=maxv, dtype=dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      self.assertAllEqual(results[False], results[True])

  @test_util.run_deprecated_v1
  def testSeed(self):
    # [FUNCTION DOC]
    # Checks that two samplers with equal op seeds give identical samples for float and integer
    # dtypes, including seeds far larger than 64 bits in magnitude (positive and negative),
    # which must be accepted (graph mode).
    for dt in get_float_types() + [dtypes.int32, dtypes.int64]:
      for seed in [345, 2**100, -2**100]:
        sx = self._Sampler(1000, 0, 17, dtype=dt, use_gpu=True, seed=seed)
        sy = self._Sampler(1000, 0, 17, dtype=dt, use_gpu=True, seed=seed)
        self.assertAllEqual(sx(), sy())

  @test_util.run_deprecated_v1
  def testNoCSE(self):
    # [FUNCTION DOC]
    # Checks two identical unseeded uniform ops in one graph are not merged: for float16, float32
    # and int32 with range zero to seventeen, asserts the norm of their difference exceeds 0.1.
    shape = [2, 3, 4]
    for dtype in dtypes.float16, dtypes.float32, dtypes.int32:
      with self.session():
        rnd1 = random_ops.random_uniform(shape, 0, 17, dtype=dtype)
        rnd2 = random_ops.random_uniform(shape, 0, 17, dtype=dtype)
        diff = (rnd2 - rnd1).eval()
        self.assertTrue(np.linalg.norm(diff) > 0.1)

  @test_util.run_deprecated_v1
  def testSingleSessionNotConstant(self):
    # [FUNCTION DOC]
    # With no seeds, repeated evaluation of the uniform op (range zero to seventeen, 100 values)
    # must not always return the same values; runs for CPU and GPU, all float dtypes and int32/int64.
    for use_gpu in [False, True]:
      for dt in get_float_types() + [dtypes.int32, dtypes.int64]:
        self._testSingleSessionNotConstant(
            random_ops.random_uniform, 100, dt, 0, 17, use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testSingleSessionOpSeedNotConstant(self):
    # [FUNCTION DOC]
    # Same as the unseeded single-session uniform test (range ten to twenty) but with a fixed op
    # seed: repeated evaluations must still vary.
    for use_gpu in [False, True]:
      for dt in get_float_types() + [dtypes.int32, dtypes.int64]:
        self._testSingleSessionNotConstant(
            random_ops.random_uniform,
            100,
            dt,
            10,
            20,
            use_gpu=use_gpu,
            op_seed=1345)

  @test_util.run_deprecated_v1
  def testSingleSessionGraphSeedNotConstant(self):
    # [FUNCTION DOC]
    # Same as the unseeded single-session uniform test (range twenty to two hundred) but with a
    # graph-level seed: repeated evaluations must still vary.
    for use_gpu in [False, True]:
      for dt in get_float_types() + [dtypes.int32, dtypes.int64]:
        self._testSingleSessionNotConstant(
            random_ops.random_uniform,
            100,
            dt,
            20,
            200,
            use_gpu=use_gpu,
            graph_seed=965)


class RandomShapeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for static shape inference of the truncated normal, normal and uniform ops in graph
  # mode, with fully known, partially known (known rank) and unknown shape arguments.

  @test_util.run_deprecated_v1
  def testTruncatedNormal(self):
    # [FUNCTION DOC]
    # Static shape inference for the truncated normal op: a literal shape gives that shape, a
    # placeholder of known length three gives three unknown dimensions, and a placeholder of
    # unknown length gives unknown rank.
    # Fully known shape.
    rnd1 = random_ops.truncated_normal([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    # Partially known shape.
    rnd2 = random_ops.truncated_normal(
        array_ops.placeholder(dtypes.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    # Unknown shape.
    rnd3 = random_ops.truncated_normal(array_ops.placeholder(dtypes.int32))
    self.assertIs(None, rnd3.get_shape().ndims)

  @test_util.run_deprecated_v1
  def testRandomNormal(self):
    # [FUNCTION DOC]
    # Static shape inference for the normal op: literal shape known, length-three placeholder
    # gives three unknown dimensions, unknown-length placeholder gives unknown rank.
    # Fully known shape.
    rnd1 = random_ops.random_normal([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    # Partially known shape.
    rnd2 = random_ops.random_normal(
        array_ops.placeholder(dtypes.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    # Unknown shape.
    rnd3 = random_ops.random_normal(array_ops.placeholder(dtypes.int32))
    self.assertIs(None, rnd3.get_shape().ndims)

  @test_util.run_deprecated_v1
  def testRandomUniform(self):
    # [FUNCTION DOC]
    # Static shape inference for the uniform op: literal shape known, length-three placeholder
    # gives three unknown dimensions, unknown-length placeholder gives unknown rank.
    # Fully known shape.
    rnd1 = random_ops.random_uniform([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    # Partially known shape.
    rnd2 = random_ops.random_uniform(
        array_ops.placeholder(dtypes.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    # Unknown shape.
    rnd3 = random_ops.random_uniform(array_ops.placeholder(dtypes.int32))
    self.assertIs(None, rnd3.get_shape().ndims)


class DeterministicOpsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for random ops while op determinism is enabled globally: unseeded use must fail and
  # seeded use must succeed. The fixture enables determinism for each test and disables it again.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture before each test: runs the base setUp, clears the global random seed, and enables
    # process-wide op determinism (a global side effect undone in tearDown).
    super().setUp()
    random_seed.set_random_seed(None)
    config.enable_op_determinism()

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture after each test: runs the base tearDown and disables process-wide op determinism.
    super().tearDown()
    config.disable_op_determinism()

  def testDeterministicOpsErrors(self):
    # [FUNCTION DOC]
    # With determinism on and no seed, the normal, truncated normal and uniform ops must each
    # raise a runtime error saying a seed is required, and the raw standard normal op must raise an
    # invalid-argument error saying a seed must be specified when evaluated.
    with self.assertRaisesRegex(
        RuntimeError,
        "Random ops require a seed to be set when determinism is enabled."):
      random_ops.random_normal((1,))
    with self.assertRaisesRegex(
        RuntimeError,
        "Random ops require a seed to be set when determinism is enabled."):
      random_ops.truncated_normal((1,))
    with self.assertRaisesRegex(
        RuntimeError,
        "Random ops require a seed to be set when determinism is enabled."):
      random_ops.random_uniform((1,))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "When determinism is enabled, random ops must have a seed specified"):
      self.evaluate(gen_random_ops.random_standard_normal((1,), dtypes.float32))

  def testErrorNotThrownWithSeed(self):
    # [FUNCTION DOC]
    # With determinism on, supplying a seed avoids the errors: an op seed, then a global seed,
    # then the raw standard normal op with either seed argument set. Passes if nothing raises.
    # KNOWN ISSUE: no explicit assertions are made; success means only that no exception occurs.
    random_ops.random_normal((1,), seed=0)
    random_seed.set_random_seed(0)
    random_ops.random_normal((1,))
    self.evaluate(gen_random_ops.random_standard_normal((1,), dtypes.float32,
                                                        seed=1))
    self.evaluate(gen_random_ops.random_standard_normal((1,), dtypes.float32,
                                                        seed2=1))


if __name__ == "__main__":
  test.main()
