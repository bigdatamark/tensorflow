# [MODULE DOC]
# Test module for the stateless random ops (uniform, normal, truncated normal, multinomial,
# gamma, poisson, shuffle, split, fold_in). It builds parameterized "cases" pairing each
# stateless op with its stateful counterpart, and checks: stateless results match the stateful
# ones after seed scrambling, CPU and GPU agree, old and new kernels (forward-compatibility
# horizon) agree, explicit Philox algorithm equals the default, results are deterministic
# (equal iff seeds equal), and invalid minval/maxval raise errors.
# Module-level helpers pick a device, invert the Philox bijection and generate cases.
# Import-time side effects: none. When run as a script it disables soft device placement,
# enables XLA devices and starts the test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for stateless random ops."""

import functools

from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.platform import test


# Note that in theory each test will reset the eager context and may choose to
# hide some devices, so we shouldn't cache this transient info. Tests in this
# file don't make those config changes, so caching is fine. It provides a good
# speed-up.
_cached_device = None


def get_device():
  # [FUNCTION DOC]
  # HELPER: lets the tests run all ops on the best available device.
  # Returns the first logical device found, searching types in the order XLA GPU, GPU, XLA CPU,
  # CPU. The result is cached in a module-level global after the first call.
  # Raises a value error listing the available devices if none of those types exist.
  global _cached_device
  if _cached_device is not None:
    return _cached_device
  # Precedence from high to low
  for device_type in ('XLA_GPU', 'GPU', 'XLA_CPU', 'CPU'):
    devices = config.list_logical_devices(device_type)
    if devices:
      _cached_device = devices[0]
      return _cached_device
  raise ValueError('Cannot find any suitable device. Available devices: %s' %
                   config.list_logical_devices())


BEFORE_EXPIRE = (2020, 10, 24)
AFTER_EXPIRE = (2020, 10, 26)


def invert_philox(key, value):
  # [FUNCTION DOC]
  # HELPER: computes seeds so stateful and stateless ops can be compared in tests.
  # Runs the ten Philox rounds backwards. Inputs: key, a pair of 32-bit integers, and value,
  # a sequence of four 32-bit counter words. Returns an array of four unsigned 64-bit integers
  # holding the pre-image of value under the Philox permutation with that key.
  # No side effects.
  """Invert the Philox bijection."""
  key = np.array(key, dtype=np.uint64)
  value = np.array(value, dtype=np.uint64)
  step = np.array([0x9E3779B9, 0xBB67AE85], dtype=np.uint64)
  for n in range(10)[::-1]:
    key0, key1 = key + n * step
    mask = np.uint64(0xffffffff)

    v0_mul = np.uint64(0x991a7cdb)
    v2_mul = np.uint64(0x6d7cae67)
    v0 = value[3] * v0_mul & mask
    v2 = value[1] * v2_mul & mask

    hi0_mul = np.uint64(0xD2511F53)
    hi1_mul = np.uint64(0xCD9E8D57)

    shift_amount = np.uint64(32)
    hi0 = v0 * hi0_mul >> shift_amount
    hi1 = v2 * hi1_mul >> shift_amount
    v1 = hi1 ^ value[0] ^ key0
    v3 = hi0 ^ value[2] ^ key1
    value = v0, v1, v2, v3
  return np.array(value, dtype=np.uint64)


SEEDS = ((7, 17), (11, 5), (2, 3))
SEED_TYPES = [dtypes.int32, dtypes.int64]


def float_cases(shape_dtypes=(None,)):
  # [FUNCTION DOC]
  # HELPER: generates parameterized test cases for floating-point random ops.
  # Input: shape_dtypes, a tuple of dtypes used to build the shape tensor (the null value
  # means the shape is passed as a plain tuple); default is just the null value.
  # Yields tuples of (name string, stateless callable, stateful callable) for every combination
  # of float16/bfloat16/float32/float64, shape (scalar, vector, matrix) and the uniform, normal
  # and truncated-normal ops with and without extra parameters. Each callable takes a seed.
  cases = (
      # Uniform distribution, with and without range
      ('uniform', stateless.stateless_random_uniform, random_ops.random_uniform,
       {}),
      ('uniform2', stateless.stateless_random_uniform,
       random_ops.random_uniform, dict(minval=2.2, maxval=7.1)),
      # Normal distribution, with and without mean+stddev
      ('normal', stateless.stateless_random_normal, random_ops.random_normal,
       {}),
      ('normal2', stateless.stateless_random_normal, random_ops.random_normal,
       dict(mean=2, stddev=3)),
      # Truncated normal distribution, with and without mean+stddev
      ('trnorm', stateless.stateless_truncated_normal,
       random_ops.truncated_normal, {}),
      ('trnorm2', stateless.stateless_truncated_normal,
       random_ops.truncated_normal, dict(mean=3, stddev=4)),
  )
  # Explicitly passing in params because capturing cell variable from loop is
  # problematic in Python
  def wrap(op: ops.Operation, dtype, shape, shape_dtype, seed, **kwargs):
    # [FUNCTION DOC]
    # HELPER: binds arguments of a random op for float_cases.
    # Calls op with the given seed, shape (converted to a constant tensor of shape_dtype unless
    # that is the null value), dtype and extra keyword arguments. bfloat16 is replaced by
    # float32 when running on a GPU lacking compute capability 8.0. Returns the op's output.
    device_type = get_device().device_type
    # Some dtypes are not supported on some devices
    if (dtype == dtypes.bfloat16 and device_type == 'GPU' and
        not test_util.is_gpu_available(
            cuda_only=True, min_cuda_compute_capability=(8, 0))):
      dtype = dtypes.float32
    shape_ = (constant_op.constant(shape, dtype=shape_dtype)
              if shape_dtype is not None else shape)
    return op(seed=seed, shape=shape_, dtype=dtype, **kwargs)

  def _name(a):
    # [FUNCTION DOC]
    # HELPER: builds readable case names in float_cases.
    # Returns the name attribute of a (dtype-like) object if it has one, otherwise returns the
    # input unchanged.
    if hasattr(a, 'name'):
      return a.name
    else:
      return a

  for dtype in dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64:
    for shape_dtype in shape_dtypes:
      for shape in (), (3,), (2, 5):
        for name, stateless_op, stateful_op, kwargs in cases:
          yield (('%s_%s_%s_%s' %
                  (name, _name(dtype), shape, _name(shape_dtype))).replace(
                      ' ', ''),
                 functools.partial(wrap, stateless_op, dtype, shape,
                                   shape_dtype, **kwargs),
                 functools.partial(wrap, stateful_op, dtype, shape, shape_dtype,
                                   **kwargs))


def int_cases(shape_dtypes=(None,), minval_maxval=None):
  # [FUNCTION DOC]
  # HELPER: generates parameterized test cases for integer uniform random ops.
  # Inputs: shape_dtypes, tuple of dtypes for the shape tensor (null value means plain tuple);
  # minval_maxval, optional list of (minval, maxval) pairs, defaulting to a single pair (2, 11111).
  # Yields tuples of (name, stateless uniform callable, stateful uniform callable) for each
  # range, shape dtype, shape and integer dtype (int32, int64); callables take a seed.

  def wrap(op: ops.Operation, minval, maxval, shape, shape_dtype, dtype,
           seed, **kwargs):
    # [FUNCTION DOC]
    # HELPER: binds arguments of a uniform op for int_cases.
    # Calls op with seed, shape (made into a constant tensor when shape_dtype is not the null
    # value), minval, maxval, dtype and extra keyword arguments; returns the op's output.
    shape_ = (constant_op.constant(shape, dtype=shape_dtype)
              if shape_dtype is not None else shape)
    return op(
        seed=seed, shape=shape_, minval=minval, maxval=maxval, dtype=dtype,
        **kwargs)

  if minval_maxval is None:
    minval_maxval = ((2, 11111),)
  for minval, maxval in minval_maxval:
    for shape_dtype in shape_dtypes:
      for shape in (), (3,), (2, 5):
        for dtype in dtypes.int32, dtypes.int64:
          yield ('uniform_%s_%s' % (minval, maxval),
                 functools.partial(wrap, stateless.stateless_random_uniform,
                                   minval, maxval, shape, shape_dtype, dtype),
                 functools.partial(wrap, random_ops.random_uniform, minval,
                                   maxval, shape, shape_dtype, dtype))


def multinomial_cases():
  # [FUNCTION DOC]
  # HELPER: generates parameterized test cases for multinomial sampling.
  # Takes no inputs. Yields tuples of (name, stateless multinomial callable, stateful multinomial
  # callable) for logits dtypes float16/bfloat16/float32/float64, output dtypes int32/int64 and
  # two small logits matrices; ten samples are drawn. Callables take a seed.
  num_samples = 10
  def wrap(op: ops.Operation, logits, logits_dtype, output_dtype, seed):
    # [FUNCTION DOC]
    # HELPER: binds arguments of a multinomial op for multinomial_cases.
    # Converts logits to a constant of logits_dtype (bfloat16 becomes float32 on GPUs lacking
    # compute capability 8.0) and calls op with the seed, ten samples and output_dtype.
    # Returns the sampled indices.
    device_type = get_device().device_type
    # Some dtypes are not supported on some devices
    if (logits_dtype == dtypes.bfloat16 and device_type == 'GPU' and
        not test_util.is_gpu_available(
            cuda_only=True, min_cuda_compute_capability=(8, 0))):
      logits_dtype = dtypes.float32
    return op(seed=seed,
              logits=constant_op.constant(logits, dtype=logits_dtype),
              num_samples=num_samples, output_dtype=output_dtype)
  for logits_dtype in (
      np.float16,
      dtypes.bfloat16.as_numpy_dtype,
      np.float32,
      np.float64,
  ):
    for output_dtype in dtypes.int32, dtypes.int64:
      for logits in ([[0.1, 0.25, 0.5, 0.15]], [[0.5, 0.5], [0.8, 0.2],
                                                [0.25, 0.75]]):
        yield ('multinomial',
               functools.partial(wrap, stateless.stateless_multinomial, logits,
                                 logits_dtype, output_dtype),
               functools.partial(wrap, random_ops.multinomial, logits,
                                 logits_dtype, output_dtype))


def gamma_cases():
  # [FUNCTION DOC]
  # HELPER: generates parameterized test cases for gamma sampling.
  # Takes no inputs. Yields tuples of (name, stateless gamma callable, stateful gamma callable)
  # for dtypes float16/float32/float64 and two alpha matrices; the sample shape is ten draws
  # per alpha element. Callables take a seed.
  def wrap(op: ops.Operation, alpha, dtype, shape, seed):
    # [FUNCTION DOC]
    # HELPER: binds arguments of a gamma op for gamma_cases.
    # Calls op with the seed, shape, alpha converted to a constant of the given dtype, and the
    # dtype. Returns the sampled tensor.
    return op(seed=seed, shape=shape,
              alpha=constant_op.constant(alpha, dtype=dtype), dtype=dtype)
  for dtype in np.float16, np.float32, np.float64:
    for alpha in ([[.5, 1., 2.]], [[0.5, 0.5], [0.8, 0.2], [0.25, 0.75]]):
      yield ('gamma',
             functools.partial(wrap, stateless.stateless_random_gamma, alpha,
                               dtype, (10,) + tuple(np.shape(alpha))),
             functools.partial(wrap, random_ops.random_gamma, alpha, dtype,
                               (10,)))


def poisson_cases():
  # [FUNCTION DOC]
  # HELPER: generates parameterized test cases for poisson sampling.
  # Takes no inputs. Yields tuples of (name, stateless poisson callable, stateful poisson
  # callable) for every pairing of rate dtype and output dtype among float16, float32, float64,
  # int32, int64 and two rate matrices. Callables take a seed.
  def wrap(op: ops.Operation, lam, lam_dtype, out_dtype, shape, seed):
    # [FUNCTION DOC]
    # HELPER: binds arguments of a poisson op for poisson_cases.
    # Calls op with the seed, shape, rate lam converted to a constant of lam_dtype, and the
    # output dtype out_dtype. Returns the sampled tensor.
    return op(seed=seed, shape=shape,
              lam=constant_op.constant(lam_dtype(lam), dtype=lam_dtype),
              dtype=out_dtype)
  for lam_dtype in np.float16, np.float32, np.float64, np.int32, np.int64:
    for out_dtype in np.float16, np.float32, np.float64, np.int32, np.int64:
      for lam in ([[5.5, 1., 2.]], [[7.5, 10.5], [3.8, 8.2], [1.25, 9.75]]):
        yield ('poisson',
               functools.partial(wrap, stateless.stateless_random_poisson, lam,
                                 lam_dtype, out_dtype,
                                 (10,) + tuple(np.shape(lam))),
               functools.partial(wrap, random_ops.random_poisson, lam,
                                 lam_dtype, out_dtype, (10,)))


def shuffle_cases():
  # [FUNCTION DOC]
  # HELPER: generates parameterized test cases for random shuffling.
  # Takes no inputs. Yields tuples of (name, stateless shuffle callable, stateful shuffle
  # callable) for dtypes int32/int64/float32/float64 and several shapes including empty and
  # size-one corner cases; the value shuffled is a range of numbers reshaped to the shape.
  # Callables take a seed.
  for dtype in np.int32, np.int64, np.float32, np.float64:
    # [], [0, ...] and [1, ...] are important corner cases
    for shape in ([], [0], [1], [100], [0, 0], [1, 0], [0, 1], [1, 2], [5, 3],
                  [7, 5, 3, 2]):
      value = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
      yield ('shuffle',
             functools.partial(stateless.stateless_shuffle, value),
             functools.partial(random_ops.random_shuffle, value))


@test_util.with_eager_op_as_function
class StatelessOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case covering the stateless random ops. Shared helper methods implement
  # the checks (match with stateful ops, CPU/GPU agreement, old/new kernel agreement, explicit
  # algorithm, determinism); the public test methods feed them cases from the module-level case
  # generators and seeds. Wrapped to run eager ops as functions.

  def _test_match(self, case, seed):
    # [FUNCTION DOC]
    # HELPER: shared check that a stateless op equals its stateful counterpart.
    # Inputs: case (name, stateless op, stateful op) and seed (pair of integers). Converts the
    # seed into the equivalent scrambled stateless seed by inverting Philox, sets the global
    # random seed to the first element, runs the stateful op with the second element and the
    # stateless op with the scrambled seed on the chosen device, and asserts equal results.
    # Side effect: sets the global random seed.
    # Stateless ops should be the same as stateful ops on the first call
    # after seed scrambling.
    key = 0x3ec8f720, 0x02461e29
    preseed = invert_philox(key, (seed[0], 0, seed[1], 0)).astype(np.uint64)
    preseed = preseed[::2] | preseed[1::2] << 32
    with ops.device(get_device().name):
      _, stateless_op, stateful_op = case
      random_seed.set_random_seed(seed[0])
      stateful = stateful_op(seed=seed[1])
      pure = stateless_op(seed=preseed)
      self.assertAllEqual(stateful, pure)

  def _test_match_stateless_cpu_gpu(self, case, seed):
    # [FUNCTION DOC]
    # HELPER: shared check that a stateless op gives the same values on CPU and the test device.
    # Inputs: case (name, stateless op, stateful op) and seed. Runs the stateless op under the
    # CPU device and under the selected device with the same seed and asserts the results are
    # close.
    # Stateless ops should produce the same result on CPUs and GPUs.
    _, stateless_op, _ = case

    with ops.device('CPU'):
      result_cpu = stateless_op(seed=seed)

    with ops.device(get_device().name):
      result_gpu = stateless_op(seed=seed)
      self.assertAllClose(result_cpu, result_gpu)

  def _test_old_and_new_stateless_match(self, case, seed):
    # [FUNCTION DOC]
    # HELPER: shared check that the new stateless kernels match the old ones.
    # Inputs: case and seed. Runs the stateless op once with the forward-compatibility horizon
    # set before the expiry date and once after it, and asserts the results are close.
    """Tests that the new stateless ops match the old stateless ones."""
    with ops.device(get_device().name):
      _, stateless_op, _ = case
      with compat.forward_compatibility_horizon(*BEFORE_EXPIRE):
        old = stateless_op(seed=seed)
      with compat.forward_compatibility_horizon(*AFTER_EXPIRE):
        new = stateless_op(seed=seed)
      self.assertAllClose(old, new)

  def _test_explicit_alg(self, case, seed):
    # [FUNCTION DOC]
    # HELPER: shared check that passing the Philox algorithm explicitly equals the default.
    # Inputs: case and seed. Runs the stateless op with and without the algorithm argument on
    # the selected device and asserts the results are close.
    """Tests that alg=philox and alg=None are the same (on CPU/GPU)."""
    with ops.device(get_device().name):
      _, stateless_op, _ = case
      implicit_alg = stateless_op(seed=seed)
      # All device types allowed in this test will result in Philox
      explicit_alg = stateless_op(seed=seed, alg='philox')
      self.assertAllClose(implicit_alg, explicit_alg)

  def _test_determinism(self, case, seed_type):
    # [FUNCTION DOC]
    # HELPER: shared check that stateless output depends only on the seed.
    # Inputs: case and seed_type (dtype of the seed tensor). Evaluates the stateless op for 25
    # seed pairs repeated three times (eagerly, or with a placeholder in graph mode) and, for
    # every pair of results, asserts equality iff the seeds are equal. For bfloat16 only the
    # equal-seed direction is asserted because low precision can make different seeds collide.
    # Stateless values should be equal iff the seeds are equal (roughly)
    seeds = [(x, y) for x in range(5) for y in range(5)] * 3  # pylint: disable=g-complex-comprehension
    with self.test_session(), ops.device(get_device().name):
      _, stateless_op, _ = case
      if context.executing_eagerly():
        values = [
            (seed, stateless_op(seed=constant_op.constant(seed, seed_type)))
            for seed in seeds]
      else:
        # Have this branch because the above branch is too slow in graph
        # mode
        seed_t = array_ops.placeholder(seed_type, shape=[2])
        pure = stateless_op(seed=seed_t)
        values = [
            (seed, pure.eval(feed_dict={seed_t: seed})) for seed in seeds
        ]
      for s0, v0 in values:
        for s1, v1 in values:
          if dtypes.as_dtype(v0.dtype) != dtypes.bfloat16:
            self.assertEqual(s0 == s1, np.all(v0 == v1))
          elif s0 == s1:
            # Skip the s0 != s1 case because v0 and v1 can be either equal or
            # unequal in that case due to bfloat16's low precision
            self.assertAllEqual(v0, v1)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(float_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testMatchFloat(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over float cases and seeds. Skipped on XLA devices (no int64 seed support).
    # Otherwise asserts the stateless op equals the stateful op after seed scrambling.
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Skip on XLA because XLA kernels do not support int64 '
                    'seeds needed by this test.')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(int_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testMatchInt(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over integer uniform cases and seeds. Skipped on XLA devices (no int64 seed
    # support). Otherwise asserts the stateless op equals the stateful op after seed scrambling.
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Skip on XLA because XLA kernels do not support int64 '
                    'seeds needed by this test.')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(multinomial_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testMatchMultinomial(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over multinomial cases and seeds. Skipped on XLA devices (no kernel).
    # Otherwise asserts the stateless multinomial equals the stateful one after seed scrambling.
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking XLA kernel')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(gamma_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testMatchGamma(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over gamma cases and seeds. Skipped on GPU and XLA devices (no kernel).
    # Otherwise asserts the stateless gamma equals the stateful gamma after seed scrambling.
    if get_device().device_type == 'GPU':
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking GPU kernel')
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking XLA kernel')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(gamma_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testStatelessGammaCpuGpuMatch(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over gamma cases and seeds. Skipped unless the selected device is a GPU.
    # Asserts the stateless gamma sampler gives close results on CPU and GPU for the same seed.
    if get_device().device_type != 'GPU':
      # This test compares the numbers produced by the CPU and GPU kernel for
      # stateless_random_gamma.
      self.skipTest('This test requires GPU')
    self._test_match_stateless_cpu_gpu(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(poisson_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testMatchPoisson(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over poisson cases and seeds. Skipped on GPU and XLA devices (no kernel).
    # Otherwise asserts the stateless poisson equals the stateful poisson after seed scrambling.
    if get_device().device_type == 'GPU':
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking GPU kernel')
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking XLA kernel')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension,undefined-variable
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(shuffle_cases()))
  def testMatchShuffle(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over shuffle cases and seeds. Skipped on GPU and XLA devices (no kernel).
    # Otherwise asserts the stateless shuffle equals the stateful shuffle after seed scrambling.
    if get_device().device_type == 'GPU':
      self.skipTest('Lacking GPU kernel')
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      self.skipTest('Lacking XLA kernel')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(float_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testOldAndNewStatelessMatchFloat(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over float cases and seeds. Asserts the new stateless float ops produce
    # results close to the old implementation (before/after the compatibility horizon).
    self._test_old_and_new_stateless_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(
          int_cases(minval_maxval=((2, 11111), (None, None)))))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testOldAndNewStatelessMatchInt(self, case, seed):
    # [FUNCTION DOC]
    # Parameterized over integer uniform cases (including ones with no minval/maxval) and seeds.
    # Asserts the new stateless integer ops are close to the old implementation.
    self._test_old_and_new_stateless_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s' % (case[0], case_id), case)
      for case_id, case in enumerate(float_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testExplicitAlgFloat(self, case):
    # [FUNCTION DOC]
    # Parameterized over float cases. Uses a fixed seed and asserts that explicitly requesting
    # the Philox algorithm gives the same values as the default.
    seed = (7, 17)
    self._test_explicit_alg(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s' % (case[0], case_id), case)
      for case_id, case in enumerate(
          int_cases(minval_maxval=((2, 11111), (None, None)))))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testExplicitAlgInt(self, case):
    # [FUNCTION DOC]
    # Parameterized over integer uniform cases. Uses a fixed seed and asserts that explicitly
    # requesting the Philox algorithm gives the same values as the default.
    seed = (7, 17)
    self._test_explicit_alg(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], seed_type.name, case_id), case, seed_type)  # pylint: disable=g-complex-comprehension
      for seed_type in SEED_TYPES
      for case_id, case in enumerate(
          float_cases(shape_dtypes=(dtypes.int32, dtypes.int64))))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testDeterminismFloat(self, case, seed_type):
    # [FUNCTION DOC]
    # Parameterized over float cases (with tensor shapes) and seed dtypes int32/int64. Skips int64
    # seeds on XLA devices. Asserts outputs are equal exactly when seeds are equal.
    if seed_type == dtypes.int64 and get_device().device_type in ('XLA_GPU',
                                                                  'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest(
          'Skip on XLA because XLA kernels do not support int64 seeds.')
    self._test_determinism(case, seed_type)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], seed_type.name, case_id), case, seed_type)  # pylint: disable=g-complex-comprehension
      for seed_type in SEED_TYPES
      for case_id, case in enumerate(
          int_cases(shape_dtypes=(dtypes.int32, dtypes.int64))))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testDeterminismInt(self, case, seed_type):
    # [FUNCTION DOC]
    # Parameterized over integer uniform cases (with tensor shapes) and seed dtypes int32/int64.
    # Skips int64 seeds on XLA devices. Asserts outputs are equal exactly when seeds are equal.
    if seed_type == dtypes.int64 and get_device().device_type in ('XLA_GPU',
                                                                  'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest(
          'Skip on XLA because XLA kernels do not support int64 seeds.')
    self._test_determinism(case, seed_type)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], seed_type.name, case_id), case, seed_type)  # pylint: disable=g-complex-comprehension
      for seed_type in SEED_TYPES
      for case_id, case in enumerate(multinomial_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testDeterminismMultinomial(self, case, seed_type):
    # [FUNCTION DOC]
    # Parameterized over multinomial cases and seed dtypes. Skipped on XLA devices. Asserts
    # outputs are equal exactly when seeds are equal.
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking XLA kernel')
    self._test_determinism(case, seed_type)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], seed_type.name, case_id), case, seed_type)  # pylint: disable=g-complex-comprehension
      for seed_type in SEED_TYPES
      for case_id, case in enumerate(gamma_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testDeterminismGamma(self, case, seed_type):
    # [FUNCTION DOC]
    # Parameterized over gamma cases and seed dtypes. Skipped on XLA devices. Asserts outputs
    # are equal exactly when seeds are equal.
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking XLA kernel')
    self._test_determinism(case, seed_type)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], seed_type.name, case_id), case, seed_type)  # pylint: disable=g-complex-comprehension
      for seed_type in SEED_TYPES
      for case_id, case in enumerate(poisson_cases()))
  @test_util.disable_tfrt('tensorflow::DirectSession::Run crashes. b/156187396')
  def testDeterminismPoisson(self, case, seed_type):
    # [FUNCTION DOC]
    # Parameterized over poisson cases and seed dtypes. Skipped on GPU and XLA devices. Asserts
    # outputs are equal exactly when seeds are equal.
    if get_device().device_type == 'GPU':
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking GPU kernel')
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking XLA kernel')
    self._test_determinism(case, seed_type)

  @test_util.run_v2_only
  def testGetKeyCounterAlg(self):
    # [FUNCTION DOC]
    # Runs only in v2 mode. Calls the raw op that derives a key and counter from a two-element
    # seed and asserts key has shape of one element and counter two elements; then calls the op
    # returning the default algorithm and asserts it is a scalar.
    seed = [1, 2]
    key, counter = gen_stateless_random_ops_v2.stateless_random_get_key_counter(
        seed)
    self.assertAllEqual(key.shape, [1])
    self.assertAllEqual(counter.shape, [2])
    alg = gen_stateless_random_ops_v2.stateless_random_get_alg()
    self.assertAllEqual(alg.shape, [])

  def assertDTypeEqual(self, a, b):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the split and fold_in tests.
    # Converts both arguments (dtype objects or names) to dtypes and asserts they are equal.
    self.assertEqual(dtypes.as_dtype(a), dtypes.as_dtype(b))

  def assertNoEqualPair(self, ls):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the split and fold_in tests.
    # Takes a list of tensors and asserts that no two distinct elements are entirely equal.
    for i in range(len(ls)):
      for j in range(i + 1, len(ls)):
        self.assertFalse(math_ops.reduce_all(ls[i] == ls[j]))

  @parameterized.parameters(['int32', 'int64'])
  @test_util.run_v2_only
  def testSplit(self, dtype):
    # [FUNCTION DOC]
    # Runs in v2 mode for seed dtypes int32 and int64. Splits one seed into three new seeds and
    # asserts the result has shape three by two, keeps the input dtype, and that the original
    # and the three new seeds are all pairwise different.
    """Test for `split`."""
    seed = constant_op.constant([1, 2], dtype=dtype)
    new_seed = stateless.split(seed, 3)
    self.assertEqual(new_seed.shape, [3, 2])
    self.assertDTypeEqual(new_seed.dtype, dtype)
    self.assertNoEqualPair([seed] + array_ops_stack.unstack(new_seed))

  @parameterized.parameters(['int32', 'int64'])
  @test_util.run_v2_only
  def testFoldIn(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the original seed is always created as int32 regardless of the dtype
    # parameter; only the folded-in data uses dtype (the output is expected to take that dtype).
    # Runs in v2 mode. Folds the values 3 and then 4 into a seed in sequence, asserting each
    # result has shape of two elements and the parameter dtype, and that the original seed
    # (cast to dtype) and both derived seeds are pairwise different.
    """Test for `fold_in`."""
    orig_seed = constant_op.constant([1, 2], dtype='int32')
    seed = stateless.fold_in(orig_seed, constant_op.constant(3, dtype=dtype))
    new_seeds = []
    new_seeds.append(seed)
    seed = stateless.fold_in(seed, constant_op.constant(4, dtype=dtype))
    new_seeds.append(seed)
    for s in new_seeds:
      self.assertEqual(s.shape, [2])
      self.assertDTypeEqual(s.dtype, dtype)
    self.assertNoEqualPair([math_ops.cast(orig_seed, dtype)] + new_seeds)

  @test_util.run_v2_only
  def testErrors(self):
    # [FUNCTION DOC]
    # Runs in v2 mode. Asserts that inside a compiled function, a non-scalar minval or a
    # non-scalar maxval passed to stateless uniform raises a value error whose message says the
    # bound must be a scalar.
    """Tests that proper errors are raised.
    """
    shape = [2, 3]
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        'minval must be a scalar; got a tensor of shape '):
      @def_function.function
      def f():
        # [FUNCTION DOC]
        # Nested function traced as a graph: calls stateless uniform with a vector-valued minval,
        # which should raise a value error at trace time. Returns nothing.
        stateless.stateless_random_uniform(
            shape=shape, seed=[1, 2], minval=array_ops.zeros(shape, 'int32'),
            maxval=100, dtype='int32')
      f()
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        'maxval must be a scalar; got a tensor of shape '):
      @def_function.function
      def f2():
        # [FUNCTION DOC]
        # Nested function traced as a graph: calls stateless uniform with a vector-valued maxval,
        # which should raise a value error at trace time. Returns nothing.
        stateless.stateless_random_uniform(
            shape=shape, seed=[1, 2], minval=0,
            maxval=array_ops.ones(shape, 'int32') * 100,
            dtype='int32')
      f2()


if __name__ == '__main__':
  config.set_soft_device_placement(False)
  context.context().enable_xla_devices()
  test.main()
