# [MODULE DOC]
# Kernel tests for the random gamma sampling op in the random ops module. Covers numpy dtype
# acceptance, empty parameter tensors, statistical moment matching against scipy, avoidance of
# exact zeros for small shape parameters, distinct successive draws, CPU/GPU agreement, seed
# reproducibility, no common-subexpression merging, static shape inference, positivity, and the
# size-overflow error. Defines one test class; running as a script starts the test runner.
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
"""Tests for tensorflow.ops.random_ops.random_gamma."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class RandomGammaTest(test.TestCase):
  # [CLASS DOC]
  # Test case for random gamma sampling. Each test starts with fixed global numpy and TensorFlow
  # seeds (set in setUp). Provides sampler and moment/zero-density helper methods; statistical
  # tests are skipped via early return when scipy is not installed.
  """This is a medium test due to the moments computation taking some time."""

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture run before each test: seeds numpy's global generator and TensorFlow's global
    # random seed with the same fixed value (137) for reproducibility. Takes no inputs, returns
    # nothing. Does not call the base-class setUp.
    # KNOWN ISSUE: the parent setUp is not invoked, so base-class per-test initialisation is skipped.
    np.random.seed(137)
    random_seed.set_random_seed(137)

  def _Sampler(self, num, alpha, beta, dtype, use_gpu=True, seed=None):
    # [FUNCTION DOC]
    # HELPER: builds sampling callables for the moment, distinctness, CPU/GPU and seed tests.
    # Inputs: num (samples per draw), alpha (gamma shape), beta (inverse scale / rate), dtype
    # (output dtype), use_gpu (whether the session may use the GPU, default true), seed (optional
    # op seed). Returns a zero-argument callable that builds a fresh graph and session, evaluates
    # the gamma op ten times and returns a ten-by-num numpy array of samples.

    def func():
      # [FUNCTION DOC]
      # HELPER: inner closure returned by the sampler factory. Takes no inputs. Creates a new
      # graph and session, builds one gamma sampling op, evaluates it ten times (filling the rows
      # of a result array) and returns the array. Side effect: opens and closes a session.
      # KNOWN ISSUE: the session variable is bound but never used (evaluation goes through the
      # test case's evaluate helper), so the use_gpu setting matters only via the session context.
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_gamma(
            [num], alpha, beta=beta, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in range(10):
          ret[i, :] = self.evaluate(rng)
      return ret

    return func

  def testNpDtypes(self):
    # [FUNCTION DOC]
    # Smoke test: random gamma accepts numpy arrays for alpha and beta and a numpy float32 dtype.
    # Evaluates the op once; passes if no error is raised. Nothing is asserted on the values.
    self.evaluate(random_ops.random_gamma(
        [5], alpha=np.ones([2, 1, 3]), beta=np.ones([3]), dtype=np.float32))

  def testEmptySamplingNoError(self):
    # [FUNCTION DOC]
    # Smoke test: sampling with an alpha tensor containing a zero-sized dimension must not raise.
    # Evaluates the op once; passes if no error is raised.
    self.evaluate(random_ops.random_gamma(
        [5], alpha=np.ones([2, 0, 3]), beta=np.ones([3]), dtype=dtypes.float32))

  @test_util.run_deprecated_v1
  def testMomentsFloat32(self):
    # [FUNCTION DOC]
    # Runs the shared moment-matching check for float32 samples (graph mode, deprecated v1).
    self._testMoments(dtypes.float32)

  @test_util.run_deprecated_v1
  def testMomentsFloat64(self):
    # [FUNCTION DOC]
    # Runs the shared moment-matching check for float64 samples (graph mode, deprecated v1).
    self._testMoments(dtypes.float64)

  def _testMoments(self, dt):
    # [FUNCTION DOC]
    # HELPER: shared body for the float32 and float64 moment tests.
    # Input dt is the sample dtype. For several strides, shape parameters (with an extra very
    # small one for float64) and scales of 9 and 17, draws 20000 seeded samples per row and
    # compares up to min(6, scale // 2) moments with scipy's gamma distribution, asserting all
    # z-scores are under 6. Returns early with a logged warning if scipy is unavailable.
    try:
      from scipy import stats  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      tf_logging.warn("Cannot test moments: %s" % e)
      return

    # The moments test is a z-value test.  This is the largest z-value
    # we want to tolerate. Since the z-test approximates a unit normal
    # distribution, it should almost definitely never exceed 6.
    z_limit = 6.0

    for stride in 0, 1, 4, 17:
      alphas = [0.2, 1.0, 3.0]
      if dt == dtypes.float64:
        alphas = [0.01] + alphas
      for alpha in alphas:
        for scale in 9, 17:
          # Gamma moments only defined for values less than the scale param.
          max_moment = min(6, scale // 2)
          sampler = self._Sampler(20000, alpha, 1 / scale, dt, seed=12345)
          z_scores = util.test_moment_matching(
              sampler(),
              max_moment,
              stats.gamma(alpha, scale=scale),
              stride=stride,
          )
          self.assertAllLess(z_scores, z_limit)

  def _testZeroDensity(self, alpha):
    # [FUNCTION DOC]
    # HELPER: shared body for the small-shape non-zero tests.
    # Input alpha is the gamma shape value. For float16, float32 and float64, draws seeded
    # samples, counts non-positive values, and compares to the number allowed by the
    # distribution's probability mass below the dtype's smallest normal value (with slack).
    # Asserts the list of failing dtypes is empty. Returns early with a warning if scipy is missing.
    """Zero isn't in the support of the gamma distribution.

    But quantized floating point math has its limits.
    TODO(bjp): Implement log-gamma sampler for small-shape distributions.

    Args:
      alpha: float shape value to test
    """
    try:
      from scipy import stats  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      tf_logging.warn("Cannot test zero density proportions: %s" % e)
      return
    allowable_zeros = {
        dtypes.float16: stats.gamma(alpha).cdf(np.finfo(np.float16).tiny),
        dtypes.float32: stats.gamma(alpha).cdf(np.finfo(np.float32).tiny),
        dtypes.float64: stats.gamma(alpha).cdf(np.finfo(np.float64).tiny)
    }
    failures = []
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sampler = self._Sampler(10000, alpha, 1.0, dt, seed=12345)
      x = sampler()
      allowable = allowable_zeros[dt] * x.size
      allowable = allowable * 2 if allowable < 10 else allowable * 1.05
      if np.sum(x <= 0) > allowable:
        failures += [dt]
    self.assertEqual([], failures)

  def testNonZeroSmallShape(self):
    # [FUNCTION DOC]
    # Runs the zero-density check with a very small shape parameter (0.01).
    self._testZeroDensity(0.01)

  def testNonZeroSmallishShape(self):
    # [FUNCTION DOC]
    # Runs the zero-density check with a smallish shape parameter (0.35).
    self._testZeroDensity(0.35)

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    # [FUNCTION DOC]
    # Checks that two unseeded draws of 1000 samples differ almost everywhere. For each float
    # dtype, takes two draws from the same sampler and asserts the number of equal entries is
    # below 20 for float16 and below 10 otherwise.
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sampler = self._Sampler(1000, 2.0, 1.0, dt)
      x = sampler()
      y = sampler()
      # Number of different samples.
      count = (x == y).sum()
      count_limit = 20 if dt == dtypes.float16 else 10
      self.assertLess(count, count_limit)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  @test_util.run_deprecated_v1
  def testCPUGPUMatch(self):
    # [FUNCTION DOC]
    # Checks that CPU and GPU kernels give nearly identical results for the same seed (graph
    # mode). For each float dtype, samples with the GPU disabled and enabled and asserts closeness,
    # with looser tolerances for float16.
    # KNOWN ISSUE: uses a shape parameter of zero, which is outside the gamma distribution's valid
    # range, so it does not exercise typical sampling paths.
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      if dt == dtypes.float16:
        self.assertAllClose(results[False], results[True], rtol=1e-3, atol=1e-3)
      else:
        self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)

  def testSeed(self):
    # [FUNCTION DOC]
    # Checks that two samplers with the same op seed return identical samples for each float
    # dtype.
    # KNOWN ISSUE: uses a shape parameter of zero, outside the gamma distribution's valid range,
    # so the equality may hold trivially rather than testing real sampling.
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sx = self._Sampler(1000, 0.0, 1.0, dt, seed=345)
      sy = self._Sampler(1000, 0.0, 1.0, dt, seed=345)
      self.assertAllEqual(sx(), sy())

  @test_util.run_deprecated_v1
  def testNoCSE(self):
    # [FUNCTION DOC]
    # Checks that two identical unseeded gamma ops in one graph are not merged into one. For each
    # float dtype, builds two ops of 24 samples, subtracts them and asserts the difference has
    # norm above 0.1 (graph mode, deprecated v1).
    # KNOWN ISSUE: the docstring expands CSE as constant subexpression eliminator; the standard
    # meaning is common subexpression elimination.
    """CSE = constant subexpression eliminator.

    SetIsStateful() should prevent two identical random ops from getting
    merged.
    """
    for dtype in dtypes.float16, dtypes.float32, dtypes.float64:
      with self.cached_session():
        rnd1 = random_ops.random_gamma([24], 2.0, dtype=dtype)
        rnd2 = random_ops.random_gamma([24], 2.0, dtype=dtype)
        diff = rnd2 - rnd1
        self.assertGreater(np.linalg.norm(diff.eval()), 0.1)

  @test_util.run_deprecated_v1
  def testShape(self):
    # [FUNCTION DOC]
    # Checks static shape inference of the gamma op in graph mode. Fully known cases: sample
    # shape followed by the broadcast shape of alpha and beta. Partially known: unknown leading
    # dims from a shape placeholder of known length. Unknown: unknown rank when the sample shape
    # or alpha has unknown rank. Asserts the inferred shape lists or rank.
    # Fully known shape.
    rnd = random_ops.random_gamma([150], 2.0)
    self.assertEqual([150], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma([150], 2.0, beta=[3.0, 4.0])
    self.assertEqual([150, 2], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma([150], array_ops.ones([1, 2, 3]))
    self.assertEqual([150, 1, 2, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma([20, 30], array_ops.ones([1, 2, 3]))
    self.assertEqual([20, 30, 1, 2, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma(
        [123], array_ops.placeholder(
            dtypes.float32, shape=(2,)))
    self.assertEqual([123, 2], rnd.get_shape().as_list())
    # Partially known shape.
    rnd = random_ops.random_gamma(
        array_ops.placeholder(
            dtypes.int32, shape=(1,)), array_ops.ones([7, 3]))
    self.assertEqual([None, 7, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma(
        array_ops.placeholder(
            dtypes.int32, shape=(3,)), array_ops.ones([9, 6]))
    self.assertEqual([None, None, None, 9, 6], rnd.get_shape().as_list())
    # Unknown shape.
    rnd = random_ops.random_gamma(
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.float32))
    self.assertIs(None, rnd.get_shape().ndims)
    rnd = random_ops.random_gamma([50], array_ops.placeholder(dtypes.float32))
    self.assertIs(None, rnd.get_shape().ndims)

  @test_util.run_deprecated_v1
  def testPositive(self):
    # [FUNCTION DOC]
    # Checks that sampling with a tiny shape parameter (0.001) never yields a value at or below
    # zero. For float16, float32 and float64 draws 10000 seeded samples and asserts the count
    # of non-positive entries is zero (graph mode, deprecated v1).
    n = int(10e3)
    for dt in [dtypes.float16, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        x = random_ops.random_gamma(shape=[n], alpha=0.001, dtype=dt, seed=0)
        self.assertEqual(0, math_ops.reduce_sum(math_ops.cast(
            math_ops.less_equal(x, 0.), dtype=dtypes.int64)).eval())

  def testSizeTooLarge(self):
    # [FUNCTION DOC]
    # Checks that an enormous requested output size is rejected with an overflow error. Only
    # runs the check in eager mode (graph mode errors are caught elsewhere by the graph optimizer);
    # expects a value error or invalid-argument error whose message mentions overflow.
    # Grappler asserts on size overflow, so this error is only caught when
    # running eagerly.
    if context.executing_eagerly():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "overflow"):
        rate = constant_op.constant(1.0, shape=(4, 4, 4, 4, 4))
        self.evaluate(
            random_ops.random_gamma(
                shape=[46902, 51188, 34063, 59195], alpha=rate))

if __name__ == "__main__":
  test.main()
