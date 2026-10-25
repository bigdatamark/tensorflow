# [MODULE DOC]
# Unit tests for the experimental NumPy-compatible random functions in the np_random module
# (seed, randn, standard_normal, uniform, poisson, random, rand, randint). Most tests compare the
# output shape and dtype of the TensorFlow implementation with the real NumPy function (values are
# random, so they are not compared); one test checks statistical properties and seeding of randn.
# Import-time side effects: none besides imports; when run as a script it enables eager execution
# and numpy methods on tensors, then runs the test main.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf numpy random number methods."""
# pylint: disable=g-direct-tensorflow-import

from absl.testing import parameterized
import numpy as onp

from tensorflow.python.framework import ops
# Needed for ndarray.reshape.
from tensorflow.python.ops.numpy_ops import np_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops.numpy_ops import np_random
from tensorflow.python.platform import test


class SeedTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the seed function of the numpy random module.

  def test(self):
    # [FUNCTION DOC]
    # Checks that seed accepts a Python integer and a numpy-style integer scalar without error,
    # and that passing a tuple of integers raises a ValueError.
    np_random.seed(1)
    np_random.seed(np_dtypes.int32(1))
    with self.assertRaises(ValueError):
      np_random.seed((1, 3))


class RandomTestBase(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the per-function random tests below.
  # Base test case providing a comparison helper. Subclasses set the attributes np_func (the
  # TensorFlow implementation) and onp_func (the reference NumPy function) in setUp.

  def _test(self, *args, **kw_args):
    # [FUNCTION DOC]
    # HELPER: comparison routine used by the random test subclasses.
    # KNOWN ISSUE: the global allow-float64 setting is restored only if all assertions pass; a
    # failing assertion leaves it changed. Also each call wraps self.onp_func again, so repeated
    # calls on one instance nest the wrappers.
    # Calls both implementations with the same arguments and asserts that output shape and dtype
    # match. Keyword options consumed (not forwarded): onp_dtype (optional dtype the NumPy result
    # is cast to before comparison) and allow_float64 (default true; temporarily sets the global
    # numpy-dtype option). The NumPy result is converted to an array so scalars compare as arrays.
    # Returns nothing; side effect: sets and restores the global float64 option.
    onp_dtype = kw_args.pop('onp_dtype', None)
    allow_float64 = kw_args.pop('allow_float64', True)
    old_allow_float64 = np_dtypes.is_allow_float64()
    np_dtypes.set_allow_float64(allow_float64)
    old_func = getattr(self, 'onp_func', None)
    # TODO(agarwal): Note that onp can return a scalar type while np returns
    # ndarrays. Currently np does not support scalar types.
    self.onp_func = lambda *args, **kwargs: onp.asarray(  # pylint: disable=g-long-lambda
        old_func(*args, **kwargs))
    np_out = self.np_func(*args, **kw_args)
    onp_out = onp.asarray(self.onp_func(*args, **kw_args))
    if onp_dtype is not None:
      onp_out = onp_out.astype(onp_dtype)
    self.assertEqual(np_out.shape, onp_out.shape)
    self.assertEqual(np_out.dtype, onp_out.dtype)
    np_dtypes.set_allow_float64(old_allow_float64)


class RandNTest(RandomTestBase):
  # [CLASS DOC]
  # Tests the randn function against NumPy for shape and dtype.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: selects randn as the implementation under test and the NumPy randn as the
    # reference, then runs the parent setUp.
    self.np_func = np_random.randn
    self.onp_func = onp.random.randn
    super(RandNTest, self).setUp()

  @parameterized.parameters((), (2), (2, 3))
  def test_float64(self, *dims):
    # [FUNCTION DOC]
    # Parameterized over no dimensions, one dimension and two dimensions. With float64 allowed,
    # asserts randn output shape and dtype equal NumPy's.
    self._test(*dims)

  @parameterized.parameters((), (2), ((2,)), (2, 3))
  def test_float32(self, *dims):
    # [FUNCTION DOC]
    # Same dimension cases, but with float64 disabled; asserts the output is single precision,
    # matching NumPy output cast to float32, in shape and dtype.
    self._test(*dims, allow_float64=False, onp_dtype=np_dtypes.float32)


class StandardNormalTest(RandomTestBase):
  # [CLASS DOC]
  # Tests standard_normal against NumPy for shape and dtype.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: sets standard_normal and the NumPy equivalent as the compared functions.
    self.np_func = np_random.standard_normal
    self.onp_func = onp.random.standard_normal
    super(StandardNormalTest, self).setUp()

  @parameterized.parameters((None,), ((),), ((1,),), ((1, 2),))
  def test(self, size):
    # [FUNCTION DOC]
    # Parameterized over size values (nothing, empty, one-dimensional, two-dimensional); asserts
    # shape and dtype match NumPy.
    self._test(size)


class UniformTest(RandomTestBase):
  # [CLASS DOC]
  # Tests uniform against NumPy for shape and dtype, including broadcasting of the bounds.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: sets uniform and the NumPy equivalent as the compared functions.
    self.np_func = np_random.uniform
    self.onp_func = onp.random.uniform
    super(UniformTest, self).setUp()

  @parameterized.parameters(
      ((), (), None),
      (1, (), None),
      ((), 1, None),
      (1, 1, None),
      ((1, 2), (2, 1), None),
      ((1, 2, 1), (2, 1, 1), (2, 2, 2)),
      ((), (), (2, 2, 2)),
  )
  def test_broadcast(self, low_shape, high_shape, size):
    # [FUNCTION DOC]
    # Builds a zeros array of low_shape and a ones array of high_shape (both double precision) as
    # bounds and calls uniform with the given size (maybe nothing); asserts shape and dtype
    # match NumPy, exercising broadcasting of the bounds.
    low = np_array_ops.zeros(low_shape).astype(np_dtypes.float64)
    high = np_array_ops.ones(high_shape).astype(np_dtypes.float64)
    self._test(low=low, high=high, size=size)

  def test_float32(self):
    # [FUNCTION DOC]
    # With float64 disabled, asserts uniform with scalar bounds and a 1x2 size yields single
    # precision output with the same shape and dtype as NumPy cast to float32.
    self._test(0, 1, (1, 2), allow_float64=False, onp_dtype=np_dtypes.float32)

  def test_dtype_cast(self):
    # [FUNCTION DOC]
    # Passes small integer scalars (8-bit signed and unsigned) as bounds and asserts the output
    # still matches NumPy's shape and dtype, i.e. bounds are cast to the float dtype.
    self._test(np_dtypes.int8(0), np_dtypes.uint8(1), (1, 2))


class PoissonTest(RandomTestBase):
  # [CLASS DOC]
  # Tests poisson against NumPy for shape and dtype.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: defines a wrapper that calls poisson and casts the result to 64-bit integer, uses it
    # as the implementation under test, and sets the NumPy poisson as the reference.
    def np_wrapper(lam, size):
      # [FUNCTION DOC]
      # HELPER: adapter for the Poisson test. Calls poisson with rate lam and shape size and
      # returns the result cast to 64-bit integer.
      return np_random.poisson(lam, size).astype(np_dtypes.int64)
    self.np_func = np_wrapper
    self.onp_func = onp.random.poisson
    super(PoissonTest, self).setUp()

  @parameterized.parameters(
      (1.0, None),
      (1.0, 1),
      (2.0, (3, 3)),
  )
  def test(self, lam, size):
    # [FUNCTION DOC]
    # Parameterized over rate and size (nothing, integer, two-dimensional). Asserts shape and
    # 64-bit integer dtype equal NumPy's result cast to 64-bit integer.
    self._test(lam, size, onp_dtype=np_dtypes.int64)


class RandomTest(RandomTestBase):
  # [CLASS DOC]
  # Tests random (uniform samples in zero to one) against NumPy for shape and dtype.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: sets random and the NumPy equivalent as the compared functions.
    self.np_func = np_random.random
    self.onp_func = onp.random.random
    super(RandomTest, self).setUp()

  @parameterized.parameters((None,), ((),), ((1,),), ((1, 2),))
  def test(self, size):
    # [FUNCTION DOC]
    # Parameterized over size values; asserts output shape and dtype match NumPy.
    self._test(size)


class RandTest(RandomTestBase):
  # [CLASS DOC]
  # Tests rand (variadic-dimension uniform samples) against NumPy for shape and dtype.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: sets rand and the NumPy equivalent as the compared functions.
    self.np_func = np_random.rand
    self.onp_func = onp.random.rand
    super(RandTest, self).setUp()

  @parameterized.parameters((), (1,), (1, 2))
  def test(self, *size):
    # [FUNCTION DOC]
    # Parameterized over zero, one and two dimension arguments; asserts shape and dtype match
    # NumPy.
    self._test(*size)


class RandIntTest(RandomTestBase):
  # [CLASS DOC]
  # Tests randint against NumPy for shape and dtype across integer dtype specifications.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: sets randint and the NumPy equivalent as the compared functions.
    self.np_func = np_random.randint
    self.onp_func = onp.random.randint
    super(RandIntTest, self).setUp()

  @parameterized.parameters(
      (0, 1, None, 'l'),
      (0, 1, None, np_dtypes.int64),
      (0, 1, 2, np_dtypes.int32),
      (0, 1, (), np_dtypes.int32),
      (0, 1, (2), np_dtypes.int64),
      (0, 1, (2, 2), 'l'),
  )
  def test(self, low, high, size, dtype):
    # [FUNCTION DOC]
    # Parameterized over bounds zero to one, sizes (nothing, integer, empty, tuples) and dtype
    # specs (a character code, 32-bit and 64-bit integer); asserts shape and dtype match NumPy.
    self._test(low, high, size=size, dtype=dtype)


class RandNDistriutionTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class name has a typo (Distriution).
  # Statistical tests of randn: output shape, dtype, mean/standard deviation and seed behaviour.

  def assertNotAllClose(self, a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: negated closeness assertion for the distribution test.
    # Asserts that a and b are NOT all close (extra keyword arguments go to the closeness check).
    # Returns if the closeness check fails; raises AssertionError reporting the element count if
    # all elements are close.
    try:
      self.assertAllClose(a, b, **kwargs)
    except AssertionError:
      return
    raise AssertionError(
        'The two values are close at all %d elements' % np_array_ops.size(a)
    )

  def testDistribution(self):
    # [FUNCTION DOC]
    # Runs the nested check over several shapes, including empty and zero-sized ones. Asserts per
    # shape: shape and dtype of randn outputs, mean near zero and standard deviation near one over
    # many samples, different outputs for different seeds, identical outputs for the same seed.

    def run_test(*args):
      # [FUNCTION DOC]
      # HELPER: shape-parameterized body of the distribution test. Input: variadic dimensions.
      # Seeds the generator, draws 1000 randn samples, checks shape and default float dtype, and
      # (for non-empty shapes) checks mean and standard deviation within tolerance 0.1, outputs
      # differ under a different seed and are identical under the original seed.
      num_samples = 1000
      tol = 0.1  # High tolerance to keep the # of samples low else the test
      # takes a long time to run.
      np_random.seed(10)
      outputs = [np_random.randn(*args) for _ in range(num_samples)]

      # Test output shape.
      for output in outputs:
        self.assertEqual(output.shape, tuple(args))
        default_dtype = (
            np_dtypes.float64
            if np_dtypes.is_allow_float64()
            else np_dtypes.float32
        )
        self.assertEqual(output.dtype.as_numpy_dtype, default_dtype)

      if np_array_ops.prod(args):  # Don't bother with empty arrays.
        outputs = [output.tolist() for output in outputs]

        # Test that the properties of normal distribution are satisfied.
        mean = np_array_ops.mean(outputs, axis=0)
        stddev = np_array_ops.std(outputs, axis=0)
        self.assertAllClose(mean, np_array_ops.zeros(args), atol=tol)
        self.assertAllClose(stddev, np_array_ops.ones(args), atol=tol)

        # Test that outputs are different with different seeds.
        np_random.seed(20)
        diff_seed_outputs = [
            np_random.randn(*args).tolist() for _ in range(num_samples)
        ]
        self.assertNotAllClose(outputs, diff_seed_outputs)

        # Test that outputs are the same with the same seed.
        np_random.seed(10)
        same_seed_outputs = [
            np_random.randn(*args).tolist() for _ in range(num_samples)
        ]
        self.assertAllClose(outputs, same_seed_outputs)

    run_test()
    run_test(0)
    run_test(1)
    run_test(5)
    run_test(2, 3)
    run_test(0, 2, 3)
    run_test(2, 0, 3)
    run_test(2, 3, 0)
    run_test(2, 3, 5)


if __name__ == '__main__':
  ops.enable_eager_execution()
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
