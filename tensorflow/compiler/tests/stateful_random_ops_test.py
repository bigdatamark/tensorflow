# [MODULE DOC]
# Tests for stateful random-number generator ops (the generator class with Philox, Threefry and
# automatic algorithm selection) running on an XLA device. Covers basic use, use inside a traced
# function, agreement with known reference outputs, equality between XLA and CPU kernels
# (including counter overflow), non-constant output, range and finiteness checks, statistical
# distribution tests (chi-squared, Anderson-Darling, truncated normal) and error reporting.
# Import-time side effects: reads the flags object into a module-level name and defines module
# constants for algorithms, integer types and float types. When run as a program, enables eager
# execution and disables soft device placement before starting the test main.
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
"""Tests for stateful random-number generation ops."""

import itertools
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import device_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util as \
random_test_util
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import stateful_random_ops as \
random
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test


FLAGS = flags.FLAGS


def xla_device():
  # [FUNCTION DOC]
  # HELPER: finds the XLA device to run tests on. Lists local devices and returns the first
  # device description found in this preference order: TPU, XLA GPU, XLA CPU. Raises a value
  # error listing the available devices if none of these exists.
  devices = device_lib.list_local_devices()
  def find_type(device_type):
    # [FUNCTION DOC]
    # HELPER: returns the first local device whose type equals device_type, or nothing if none.
    for d in devices:
      if d.device_type == device_type:
        return d
    return None
  d = find_type("TPU") or find_type("XLA_GPU") or find_type("XLA_CPU")
  if d is None:
    raise ValueError(
        "Can't find any XLA device. Available devices:\n%s" % devices)
  return d


def xla_device_name():
  # [FUNCTION DOC]
  # HELPER: returns the full name string of the XLA device chosen by the device-finding helper,
  # for use in device placement scopes. Raises if no XLA device exists.
  return str(xla_device().name)


ALGS = [
    random_ops_util.Algorithm.PHILOX.value,
    random_ops_util.Algorithm.THREEFRY.value,
    random_ops_util.Algorithm.AUTO_SELECT.value,
]
INTS = [dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64]
FLOATS = [dtypes.bfloat16, dtypes.float32, dtypes.float64]


class StatefulRandomOpsTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for stateful random-number generator operators on the XLA device. Most tests are
  # parameterized over algorithm and dtype and create generators from fixed seeds inside the XLA
  # device scope, so results are deterministic. No instance state.
  """Test cases for stateful random-number generator operators."""

  @parameterized.parameters(ALGS)
  def testSimple(self, alg):
    # [FUNCTION DOC]
    # Smoke test parameterized over algorithm: creates a generator from seed zero on the XLA
    # device and draws normal, bounded uniform (unsigned 32-bit) and full-range integer values.
    # Asserts nothing beyond that no error occurs.
    """A simple test."""
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=0, alg=alg)
      gen.normal(shape=(3,))
      gen.uniform(shape=(3,), minval=0, maxval=10, dtype=dtypes.uint32)
      gen.uniform_full_int(shape=(3,))

  @parameterized.parameters(ALGS)
  def testDefun(self, alg):
    # [FUNCTION DOC]
    # Parameterized over algorithm: creates a generator on the XLA device and calls it from inside
    # a traced function that draws normal, bounded uniform and full-range integer values, then
    # runs that function. Asserts nothing beyond that no error occurs.
    """Test for defun."""
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=0, alg=alg)
      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: traced function body; draws three kinds of random values from the enclosing
        # generator and returns them as a tuple of three tensors.
        x = gen.normal(shape=(3,))
        y = gen.uniform(shape=(3,), minval=0, maxval=10, dtype=dtypes.uint32)
        z = gen.uniform_full_int(shape=(3,))
        return (x, y, z)
      f()

  def _compareToKnownOutputs(self, g, counter, key, expect):
    # [FUNCTION DOC]
    # HELPER: checks a generator against known reference outputs.
    # Inputs: g (the generator), counter and key (lists of 32-bit words), expect (list of expected
    # 32-bit outputs). Packs pairs of 32-bit words (low word first) into 64-bit values, sets the
    # generator state to the counter values followed by the key values, then draws as many 32-bit
    # values as the counter has words and asserts they equal expect. Resets the state and draws
    # half as many 64-bit values, asserting they equal the packed expected words. Mutates the
    # generator state. Returns nothing.
    """Compares against known outputs for specific counter and key inputs."""
    def uint32s_to_uint64(a, b):
      # [FUNCTION DOC]
      # HELPER: combines a low 32-bit word a and high 32-bit word b into one 64-bit integer.
      return b << 32 | a

    def uint32s_to_uint64s(ls):
      # [FUNCTION DOC]
      # HELPER: converts a list of 32-bit words (even length, low word first in each pair) into a
      # list of half as many 64-bit integers.
      return [uint32s_to_uint64(ls[2 * i], ls[2 * i + 1])
              for i in range(len(ls) // 2)]

    ctr_len = len(counter)
    counter = uint32s_to_uint64s(counter)
    key = uint32s_to_uint64s(key)
    state = counter + key
    g.reset(state)
    got = g.uniform_full_int(shape=(ctr_len,), dtype=dtypes.uint32)
    self.assertAllEqual(expect, got)
    g.reset(state)
    got = g.uniform_full_int(shape=(ctr_len // 2,), dtype=dtypes.uint64)
    self.assertAllEqual(uint32s_to_uint64s(expect), got)

  def testThreefry2x32(self):
    # [FUNCTION DOC]
    # Creates a Threefry generator on the XLA device and checks three published known-answer
    # vectors (all-zero, all-ones and the digits-of-pi counter/key pairs) via the known-output
    # helper.
    """Tests ThreeFry2x32 conforms to known results.
    """
    # Based on
    # https://github.com/google/jax/blob/8565a3486adf16beb388b2364c9cd930d7a0d92d/tests/random_test.py#L65-L85
    # which is in turn based on
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32

    with ops.device(xla_device_name()):
      g = random.Generator.from_seed(seed=0, alg=random.RNG_ALG_THREEFRY)
      self._compareToKnownOutputs(
          g,
          [0x00000000, 0x00000000], [0x00000000, 0x00000000],
          [0x6b200159, 0x99ba4efe])
      self._compareToKnownOutputs(
          g,
          [0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff],
          [0x1cb996fc, 0xbb002be7])
      self._compareToKnownOutputs(
          g,
          [0x243f6a88, 0x85a308d3], [0x13198a2e, 0x03707344],
          [0xc4923a9c, 0x483df7a0])

  def testPhilox4x32(self):
    # [FUNCTION DOC]
    # Creates a Philox generator on the XLA device and checks three published known-answer
    # vectors (four-word counters with two-word keys) via the known-output helper.
    """Tests Philox4x32 conforms to known results.
    """
    # Based on
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_philox.cpp#L50-L52

    with ops.device(xla_device_name()):
      g = random.Generator.from_seed(seed=0, alg=random.RNG_ALG_PHILOX)
      self._compareToKnownOutputs(
          g,
          [0x00000000, 0x00000000, 0x00000000, 0x00000000],
          [0x00000000, 0x00000000],
          [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8])
      self._compareToKnownOutputs(
          g,
          [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
          [0xffffffff, 0xffffffff],
          [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd])
      self._compareToKnownOutputs(
          g,
          [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
          [0xa4093822, 0x299f31d0],
          [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1])

  @parameterized.parameters(INTS)
  def testXLAEqualsCPU(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over integer dtype. Creates Philox generators with the same seed on CPU and on
    # the XLA device; five times draws a 315x49 full-range integer array from each and skips the
    # state forward by 100. Asserts the values and the generator states are identical.
    """Tests that XLA and CPU kernels generate the same integers."""
    seed = 1234
    shape = [315, 49]
    with ops.device("/device:CPU:0"):
      cpu_gen = random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
    with ops.device(xla_device_name()):
      xla_gen = random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
    # Repeat multiple times to make sure that the state after
    # number-generation are the same between CPU and XLA.
    for _ in range(5):
      with ops.device("/device:CPU:0"):
        # Test both number-generation and skip
        cpu = cpu_gen.uniform_full_int(shape=shape, dtype=dtype)
        cpu_gen.skip(100)
      with ops.device(xla_device_name()):
        xla = xla_gen.uniform_full_int(shape=shape, dtype=dtype)
        xla_gen.skip(100)
      self.assertAllEqual(cpu, xla)
      self.assertAllEqual(cpu_gen.state, xla_gen.state)

  def testXLAEqualsCPUAroundCounterOverflow(self):
    # [FUNCTION DOC]
    # Like the CPU/XLA equality test but fixed to 64-bit unsigned integers and a seed just below
    # the 64-bit limit, so the counter carries past 64 bits of the 128-bit Philox counter. Asserts
    # the values and generator states match between CPU and XLA (the final value comparison after
    # the loop repeats the last in-loop check).
    """Tests XLA and CPU kernels generate the same integers in overflow case.

       Specifically this tests the case where the counter is incremented past
       what can fit within 64 bits of the 128 bit Philox counter.
    """
    dtype = dtypes.uint64
    seed = 2**64 - 10
    shape = [315, 49]
    with ops.device("/device:CPU:0"):
      cpu_gen = random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
    with ops.device(xla_device_name()):
      xla_gen = random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
    # Repeat multiple times to make sure that the state after
    # number-generation are the same between CPU and XLA.
    for _ in range(5):
      with ops.device("/device:CPU:0"):
        # Test both number-generation and skip
        cpu = cpu_gen.uniform_full_int(shape=shape, dtype=dtype)
        cpu_gen.skip(100)
      with ops.device(xla_device_name()):
        xla = xla_gen.uniform_full_int(shape=shape, dtype=dtype)
        xla_gen.skip(100)
      self.assertAllEqual(cpu, xla)
      self.assertAllEqual(cpu_gen.state, xla_gen.state)
    self.assertAllEqual(cpu, xla)

  def _testRngIsNotConstant(self, rng, dtype):
    # [FUNCTION DOC]
    # HELPER: calls the given generator function twice with the dtype, converts both results to
    # numpy arrays and asserts they are not equal (a working generator repeats with very low
    # probability).
    # Tests that 'rng' does not always return the same value.
    # The random-number generator, if working correctly, should produce the
    # same output multiple times with low probability.
    x = rng(dtype).numpy()
    y = rng(dtype).numpy()
    self.assertFalse(np.array_equal(x, y))

  def check_dtype(self, dtype):
    # [FUNCTION DOC]
    # HELPER: skips the current test when the selected XLA device is a TPU and dtype is float64,
    # which TPUs do not support. Otherwise does nothing.
    device = xla_device()
    if device.device_type == "TPU" and dtype == dtypes.float64:
      self.skipTest("TPU doesn't support float64.")

  @parameterized.parameters(list(itertools.product(ALGS, INTS + FLOATS)))
  def testUniformIsNotConstant(self, alg, dtype):
    # [FUNCTION DOC]
    # Parameterized over algorithm and integer or float dtype. Checks that two uniform draws
    # (shape of two elements, upper bound the dtype's maximum) from one generator differ.
    self.check_dtype(dtype)
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=1234, alg=alg)
      def rng(dtype):
        # [FUNCTION DOC]
        # HELPER: draws a uniform tensor of two elements of the given dtype with the upper bound
        # set to the dtype's maximum value, from the enclosing generator.
        maxval = dtype.max
        return gen.uniform(shape=[2], dtype=dtype, maxval=maxval)

      self._testRngIsNotConstant(rng, dtype)

  @parameterized.parameters(list(itertools.product(ALGS, FLOATS)))
  def testNormalIsNotConstant(self, alg, dtype):
    # [FUNCTION DOC]
    # Parameterized over algorithm and float dtype. Checks that two normal draws (two elements
    # each) from one generator differ.
    self.check_dtype(dtype)
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=1234, alg=alg)
      def rng(dtype):
        # [FUNCTION DOC]
        # HELPER: draws a normal tensor of two elements of the given dtype from the enclosing
        # generator.
        return gen.normal(shape=[2], dtype=dtype)

      self._testRngIsNotConstant(rng, dtype)

  @parameterized.parameters(list(itertools.product(ALGS, INTS + FLOATS)))
  def testUniformIsInRange(self, alg, dtype):
    # [FUNCTION DOC]
    # Parameterized over algorithm and dtype. Draws 1000 uniform values with bounds 2 and 33 and
    # asserts all are at least the lower bound and at most the upper bound.
    self.check_dtype(dtype)
    minval = 2
    maxval = 33
    size = 1000
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=1234, alg=alg)
      x = gen.uniform(
          shape=[size], dtype=dtype, minval=minval, maxval=maxval).numpy()
      self.assertTrue(np.all(x >= minval))
      self.assertTrue(np.all(x <= maxval))

  @parameterized.parameters(list(itertools.product(ALGS, FLOATS)))
  def testNormalIsFinite(self, alg, dtype):
    # [FUNCTION DOC]
    # Parameterized over algorithm and float dtype. Draws 10000 normal values and asserts every
    # one is finite.
    self.check_dtype(dtype)
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=1234, alg=alg)
      x = gen.normal(shape=[10000], dtype=dtype).numpy()
      self.assertTrue(np.all(np.isfinite(x)))

  @parameterized.parameters(list(itertools.product(
      ALGS, INTS + FLOATS, (12, 13, 123, 4321))))
  def testDistributionOfUniform(self, alg, dtype, seed):
    # [FUNCTION DOC]
    # Parameterized over algorithm, dtype and seed. Skips a list of known-failing combinations
    # (certain Threefry/auto-select, TPU and megacore cases with 64-bit or 32-bit integer types).
    # Otherwise draws 1000 uniform values (upper bound 1 for floats, 100 for integers,
    # normalized to the unit interval), bins them into 10 equal bins and asserts the chi-squared
    # statistic is below 16.92 (the 5 percent critical value for 9 degrees of freedom).
    """Use Pearson's Chi-squared test to test for uniformity."""
    self.check_dtype(dtype)
    three_fry = random_ops_util.Algorithm.THREEFRY.value
    auto_select = random_ops_util.Algorithm.AUTO_SELECT.value
    is_tpu = xla_device().device_type == "TPU"
    is_megacore = "megacore" in os.environ.get("TEST_TARGET", "").lower()
    # TODO(b/244649364): Investigate why these combinations fail.
    if ((alg, dtype, seed) in [(three_fry, dtypes.int64, 123),
                               (three_fry, dtypes.uint64, 123)] or
        is_tpu and
        (alg, dtype, seed) in [(auto_select, dtypes.int64, 123),
                               (auto_select, dtypes.uint64, 123)] or
        is_megacore and
        (alg, dtype, seed) in [(auto_select, dtypes.int32, 123),
                               (auto_select, dtypes.uint32, 123),
                               (auto_select, dtypes.int32, 12),
                               (auto_select, dtypes.uint32, 12)]):
      self.skipTest(
          "This (device, alg, dtype, seed) combination fails (b/244649364).")
    with ops.device(xla_device_name()):
      n = 1000
      gen = random.Generator.from_seed(seed=seed, alg=alg)
      maxval = 1
      if dtype.is_integer:
        maxval = 100
      t = gen.uniform(shape=[n], maxval=maxval, dtype=dtype)
      x = t.numpy().astype(float)
      if maxval > 1:
        # Normalize y to range [0, 1).
        x = x / maxval
      # Tests that the values are distributed amongst 10 bins with equal
      # probability. 16.92 is the Chi^2 value for 9 degrees of freedom with
      # p=0.05. This test is probabilistic and would be flaky if the random
      # seed were not fixed.
      val = random_test_util.chi_squared(x, 10)
      self.assertLess(val, 16.92)

  @parameterized.parameters(list(itertools.product(ALGS, FLOATS)))
  def testDistributionOfNormal(self, alg, dtype):
    # [FUNCTION DOC]
    # Parameterized over algorithm and float dtype. Draws 1000 normal values with a fixed seed and
    # asserts the Anderson-Darling statistic is below 2.492 (the 5 percent critical value for a
    # known mean and variance).
    """Use Anderson-Darling test to test distribution appears normal."""
    self.check_dtype(dtype)
    with ops.device(xla_device_name()):
      n = 1000
      gen = random.Generator.from_seed(seed=1234, alg=alg)
      x = gen.normal(shape=[n], dtype=dtype).numpy()
      # The constant 2.492 is the 5% critical value for the Anderson-Darling
      # test where the mean and variance are known. This test is probabilistic
      # so to avoid flakiness the seed is fixed.
      self.assertLess(
          random_test_util.anderson_darling(x.astype(float)), 2.492)

  @parameterized.parameters(list(itertools.product(ALGS, FLOATS)))
  def testTruncatedNormal(self, alg, dtype):
    # [FUNCTION DOC]
    # Parameterized over algorithm and float dtype. Draws 100000 truncated-normal values with a
    # fixed seed and passes them to the shared truncated-normal checker with tolerances for mean
    # (2e-3), median (4e-3) and variance (1e-2 for bfloat16, otherwise 5e-3).
    self.check_dtype(dtype)
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=123, alg=alg)
      n = 100000
      y = gen.truncated_normal(shape=[n], dtype=dtype).numpy()
      random_test_util.test_truncated_normal(
          self.assertEqual, self.assertAllClose, n, y,
          mean_atol=2e-3, median_atol=4e-3,
          variance_rtol=1e-2 if dtype == dtypes.bfloat16 else 5e-3)

  @test_util.disable_mlir_bridge(
      "b/180412086: MLIR bridge gives wrong error messages.")
  def testErrors(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge (wrong error messages). Calls the raw stateful normal and
    # read-and-skip ops with invalid inputs and asserts the error type and message: algorithm not
    # a scalar; algorithm of wrong type; unsupported algorithm id (twice); state variable of the
    # wrong dtype; state with more than one dimension; state too small for Threefry and for
    # Philox.
    """Tests that proper errors are raised.
    """
    shape = [2, 3]
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=1234, alg=random.RNG_ALG_THREEFRY)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          r"algorithm.* must be of shape \[\], not"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            gen.state.handle, [0, 0], shape)
      with self.assertRaisesWithPredicateMatch(
          TypeError, "EagerTensor of dtype int64"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            gen.state.handle, 1.1, shape)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "Unsupported algorithm id"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            gen.state.handle, 123, shape)
      with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError,
                                               "Unsupported algorithm id"):
        gen_stateful_random_ops.rng_read_and_skip(
            gen.state.handle, alg=123, delta=10)
      var = variables.Variable([0, 0], dtype=dtypes.uint32)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "Trying to read variable .* Expected int64 got"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_THREEFRY, shape)
      var = variables.Variable([[0]], dtype=dtypes.int64)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "RNG state must have one and only one dimension, not"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_THREEFRY, shape)
      var = variables.Variable([0], dtype=dtypes.int64)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "The size of the state must be at least"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_THREEFRY, shape)
      var = variables.Variable([0, 0], dtype=dtypes.int64)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "The size of the state must be at least"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_PHILOX, shape)


if __name__ == "__main__":
  ops.enable_eager_execution()
  config.set_soft_device_placement(False)
  test.main()
