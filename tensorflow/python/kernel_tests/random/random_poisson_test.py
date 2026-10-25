# [MODULE DOC]
# Kernel tests for the random Poisson sampling op. Covers statistical moment matching against
# scipy, CPU/GPU agreement, seed reproducibility, no merging of identical ops, empty and
# inferred static shapes, all pairings of rate and output dtypes, an infinite rate, and the
# size-overflow error. Defines the tuple of supported dtypes and one test class; running as a
# script starts the TensorFlow test runner. No import-time side effects.
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
"""Tests for tensorflow.ops.random_ops.random_poisson."""
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

# All supported dtypes for random_poisson().
_SUPPORTED_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64,
                     dtypes.int32, dtypes.int64)


class RandomPoissonTest(test.TestCase):
  # [CLASS DOC]
  # Test case for Poisson sampling. Provides a sampler-factory helper that draws ten rows of
  # samples; statistical tests return early when scipy is unavailable. No shared instance state.
  """This is a large test due to the moments computation taking some time."""

  def _Sampler(self, num, lam, dtype, use_gpu, seed=None):
    # [FUNCTION DOC]
    # HELPER: builds sampling callables for the Poisson tests.
    # Inputs: num (samples per row), lam (Poisson rate), dtype (output dtype), use_gpu (whether
    # the session may use the GPU) and seed (optional op seed). Returns a zero-argument callable
    # that evaluates the op ten times in a fresh graph and session and returns a ten-by-num array.

    def func():
      # [FUNCTION DOC]
      # HELPER: inner closure of the sampler factory; takes no inputs. Builds one Poisson op in a
      # new graph and session, evaluates it ten times into array rows and returns the array.
      # KNOWN ISSUE: the session handle is bound but never used.
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_poisson(lam, [num], dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in range(10):
          ret[i, :] = self.evaluate(rng)
      return ret

    return func

  def testMoments(self):
    # [FUNCTION DOC]
    # Moment-matching test against scipy's Poisson distribution. For every supported dtype,
    # several strides and rates 3 and 20 (below and above 10), draws 10000 seeded samples on CPU,
    # compares five moments and asserts all z-scores are below 6. Returns early with a logged
    # warning if scipy cannot be imported.
    try:
      from scipy import stats  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      tf_logging.warn("Cannot test moments: %s", e)
      return

    # The moments test is a z-value test.  This is the largest z-value
    # we want to tolerate. Since the z-test approximates a unit normal
    # distribution, it should almost definitely never exceed 6.
    z_limit = 6.0
    for dt in _SUPPORTED_DTYPES:
      # Test when lam < 10 and when lam >= 10
      for stride in 0, 4, 10:
        for lam in (3., 20):
          max_moment = 5
          sampler = self._Sampler(10000, lam, dt, use_gpu=False, seed=12345)
          z_scores = util.test_moment_matching(
              sampler(),
              max_moment,
              stats.poisson(lam),
              stride=stride,
          )
          self.assertAllLess(z_scores, z_limit)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  @test_util.run_deprecated_v1
  def testCPUGPUMatch(self):
    # [FUNCTION DOC]
    # Checks CPU and GPU kernels give nearly identical samples for the same seed (rate one, 1000
    # values) for every supported dtype (graph mode); tolerance 1e-3 for float16, 1e-6 otherwise.
    for dt in _SUPPORTED_DTYPES:
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(1000, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      if dt == dtypes.float16:
        self.assertAllClose(results[False], results[True], rtol=1e-3, atol=1e-3)
      else:
        self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)

  @test_util.run_deprecated_v1
  def testSeed(self):
    # [FUNCTION DOC]
    # Checks two samplers with the same op seed return identical samples for the three float
    # dtypes (graph mode).
    # KNOWN ISSUE: integer output dtypes, which are supported elsewhere in this file, are not covered.
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sx = self._Sampler(1000, 1.0, dt, use_gpu=True, seed=345)
      sy = self._Sampler(1000, 1.0, dt, use_gpu=True, seed=345)
      self.assertAllEqual(sx(), sy())

  @test_util.run_deprecated_v1
  def testNoCSE(self):
    # [FUNCTION DOC]
    # Checks two identical unseeded Poisson ops (rate two, 24 values) are not merged into one:
    # for each float dtype, asserts the norm of their difference is at least one (graph mode).
    # KNOWN ISSUE: the docstring expands CSE as constant subexpression eliminator; the standard
    # meaning is common subexpression elimination.
    """CSE = constant subexpression eliminator.

    SetIsStateful() should prevent two identical random ops from getting
    merged.
    """
    for dtype in dtypes.float16, dtypes.float32, dtypes.float64:
      with self.cached_session():
        rnd1 = random_ops.random_poisson(2.0, [24], dtype=dtype)
        rnd2 = random_ops.random_poisson(2.0, [24], dtype=dtype)
        diff = rnd2 - rnd1
        # Since these are all positive integers, the norm will
        # be at least 1 if they are different.
        self.assertGreaterEqual(np.linalg.norm(diff.eval()), 1)

  def testZeroShape(self):
    # [FUNCTION DOC]
    # Checks that an empty rate and empty sample shape give a result whose static shape is a
    # single dimension of size zero, and that evaluating it yields an empty float32 array.
    with self.cached_session():
      rnd = random_ops.random_poisson([], [], seed=12345)
      self.assertEqual([0], rnd.get_shape().as_list())
      self.assertAllClose(np.array([], dtype=np.float32), self.evaluate(rnd))

  @test_util.run_deprecated_v1
  def testShape(self):
    # [FUNCTION DOC]
    # Static shape inference in graph mode: fully known (sample shape followed by the rate's
    # shape), partially known (placeholder sample shape of known length gives unknown leading
    # dimensions) and unknown (unknown-rank rate or sample shape gives unknown rank).
    # Fully known shape
    rnd = random_ops.random_poisson(2.0, [150], seed=12345)
    self.assertEqual([150], rnd.get_shape().as_list())
    rnd = random_ops.random_poisson(
        lam=array_ops.ones([1, 2, 3]),
        shape=[150],
        seed=12345)
    self.assertEqual([150, 1, 2, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_poisson(
        lam=array_ops.ones([1, 2, 3]),
        shape=[20, 30],
        seed=12345)
    self.assertEqual([20, 30, 1, 2, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_poisson(
        lam=array_ops.placeholder(dtypes.float32, shape=(2,)),
        shape=[12],
        seed=12345)
    self.assertEqual([12, 2], rnd.get_shape().as_list())
    # Partially known shape.
    rnd = random_ops.random_poisson(
        lam=array_ops.ones([7, 3]),
        shape=array_ops.placeholder(dtypes.int32, shape=(1,)),
        seed=12345)
    self.assertEqual([None, 7, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_poisson(
        lam=array_ops.ones([9, 6]),
        shape=array_ops.placeholder(dtypes.int32, shape=(3,)),
        seed=12345)
    self.assertEqual([None, None, None, 9, 6], rnd.get_shape().as_list())
    # Unknown shape.
    rnd = random_ops.random_poisson(
        lam=array_ops.placeholder(dtypes.float32),
        shape=array_ops.placeholder(dtypes.int32),
        seed=12345)
    self.assertIs(None, rnd.get_shape().ndims)
    rnd = random_ops.random_poisson(
        lam=array_ops.placeholder(dtypes.float32),
        shape=[50],
        seed=12345)
    self.assertIs(None, rnd.get_shape().ndims)

  @test_util.run_deprecated_v1
  def testDTypeCombinationsV2(self):
    # [FUNCTION DOC]
    # Smoke test over every pairing of supported rate dtype and output dtype: samples ten values
    # with rate one in a cached session (graph mode). Passes if no pairing raises an error.
    """Tests random_poisson_v2() for all supported dtype combinations."""
    with self.cached_session():
      for lam_dt in _SUPPORTED_DTYPES:
        for out_dt in _SUPPORTED_DTYPES:
          random_ops.random_poisson(
              constant_op.constant([1], dtype=lam_dt), [10],
              dtype=out_dt).eval()

  @test_util.run_deprecated_v1
  def testInfRate(self):
    # [FUNCTION DOC]
    # Checks that an infinite rate yields infinite samples: two samples with rate infinity must
    # both equal infinity (graph mode).
    sample = random_ops.random_poisson(shape=[2], lam=np.inf)
    self.assertAllEqual([np.inf, np.inf], self.evaluate(sample))

  def testSizeTooLarge(self):
    # [FUNCTION DOC]
    # Error case: requesting an enormous output (a huge sample shape times a rate tensor) must
    # raise a value error or invalid-argument error whose message mentions overflow.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "overflow"):
      rate = constant_op.constant(1.0, shape=(4, 4, 4, 4, 4))
      self.evaluate(
          random_ops.random_poisson(
              shape=[46902, 51188, 34063, 59195], lam=rate))


if __name__ == "__main__":
  test.main()
