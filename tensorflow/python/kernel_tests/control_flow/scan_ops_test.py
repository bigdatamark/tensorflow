# [MODULE DOC]
# Functional tests for the cumulative scan ops cumsum and cumprod in the math ops module. A pair
# of module-level numpy helpers build reference results (reversal along an axis and an option
# adapter that emulates the exclusive and reverse flags on top of numpy's plain cumulative sum or
# product). Two test classes, one per op, compare TensorFlow results with the numpy reference for
# many dtypes, ranks, axes and flag combinations, check error messages for invalid axes, and check
# analytic gradients against numeric ones. No flags or registrations at import time; tests are run
# through the standard test main at the bottom.
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
"""Functional tests for scan ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def numpy_reverse(x, axis):
  # [FUNCTION DOC]
  # HELPER: numpy reference helper for the scan tests. Takes x, a numpy array, and axis, an integer
  # that may be negative (counted from the end). Returns a view of x with elements reversed along that
  # axis; other axes untouched. No side effects and no exceptions of its own beyond numpy indexing.
  length = len(x.shape)
  if axis < 0:
    axis = length + axis

  ix = [
      slice(None, None, -1) if i == axis else slice(None) for i in range(length)
  ]
  return x[tuple(ix)]


def handle_options(func, x, axis, exclusive, reverse):
  # [FUNCTION DOC]
  # HELPER: numpy reference adapter for the scan tests. Takes func (numpy's cumulative sum or product
  # function), x (numpy array), axis (integer, negative allowed), exclusive (boolean) and reverse
  # (boolean). Optionally reverses x along axis, then either computes the plain scan, or, for
  # exclusive mode, prepends an identity slice (zeros for the sum, ones for the product) and scans all
  # but the last slice, concatenating along axis. If reverse is set, the result is reversed back.
  # Returns the numpy array. Raises a value error with the message "Unknown scan function" if
  # exclusive is requested with any function other than the cumulative sum or product.
  """Adds tf options to numpy scan ops."""
  length = len(x.shape)
  if axis < 0:
    axis = length + axis

  if reverse:
    x = numpy_reverse(x, axis)

  if exclusive:
    ix_head = [slice(0, 1) if i == axis else slice(None) for i in range(length)]
    ix_init = [
        slice(0, -1) if i == axis else slice(None) for i in range(length)
    ]
    if func == np.cumsum:
      init = np.zeros_like(x[tuple(ix_head)])
    elif func == np.cumprod:
      init = np.ones_like(x[tuple(ix_head)])
    else:
      raise ValueError("Unknown scan function.")
    x = np.concatenate([init, func(x[tuple(ix_init)], axis)], axis=axis)
  else:
    x = func(x, axis=axis)

  if reverse:
    x = numpy_reverse(x, axis)
  return x


class CumsumTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the cumulative sum op. Holds a class-level list of dtypes to exercise (int32, int64,
  # float16, float32, float64, complex64, complex128, bfloat16). Compares against the numpy reference
  # via the helper, for all exclusive and reverse combinations, plus invalid axis errors and gradients.

  valid_dtypes = [
      np.int32,
      np.int64,
      np.float16,
      np.float32,
      np.float64,
      np.complex64,
      np.complex128,
      dtypes.bfloat16.as_numpy_dtype,
  ]

  def _compare(self, x, axis, exclusive, reverse):
    # [FUNCTION DOC]
    # HELPER: compares the cumulative sum for one configuration. Takes x (numpy array), axis (integer),
    # exclusive and reverse (booleans). Computes the numpy reference with the option adapter, evaluates
    # the TensorFlow cumulative sum in a cached session, and asserts the two are close (default
    # tolerances). Needs graph-mode evaluation. Returns nothing.
    np_out = handle_options(np.cumsum, x, axis, exclusive, reverse)
    with self.cached_session():
      tf_out = math_ops.cumsum(x, axis, exclusive, reverse).eval()

    self.assertAllClose(np_out, tf_out)

  def _compareAll(self, x, axis):
    # [FUNCTION DOC]
    # HELPER: runs the comparison for every combination of exclusive and reverse (four) for the given x
    # and axis.
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)

  @test_util.run_deprecated_v1
  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: empty (length zero) vector of every valid dtype; scans along axes minus one and zero in
    # all flag combinations. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.zeros([0]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def testAxisType(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test never asserts anything; it only checks that the op accepts both int32 and
    # int64 axis tensors without raising.
    # Scenario: for each valid dtype, a one-to-five vector; for each axis dtype int64 and int32, a scalar
    # axis constant zero is built and cumulative sum evaluated in a cached session. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis_dtype in [dtypes.int64, dtypes.int32]:
        with self.cached_session():
          axis = constant_op.constant(0, axis_dtype)
          tf_out = math_ops.cumsum(x, axis).eval()

  @test_util.run_deprecated_v1
  def testNaN(self):
    # [FUNCTION DOC]
    # Scenario: NaN propagation. For float16, float32, float64 and bfloat16, a vector of one to five
    # gets a NaN placed at each position in turn; all flag combinations on axes minus one and zero are
    # compared against numpy. Graph-mode only.
    for dtype in (
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ):
      for nan_idx in range(0, 5):
        x = np.arange(1, 6).reshape([5]).astype(dtype)
        x[nan_idx] = np.nan
        for axis in (-1, 0):
          self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test1D(self):
    # [FUNCTION DOC]
    # Scenario: one-dimensional vector of one to five for every valid dtype, axes minus one and zero, all
    # flag combinations. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test2D(self):
    # [FUNCTION DOC]
    # Scenario: 2 by 5 matrix of zero to nine for every valid dtype over axes minus two to one, all flag
    # combinations. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(0, 10).reshape([2, 5]).astype(dtype)
      for axis in (-2, -1, 0, 1):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test3D(self):
    # [FUNCTION DOC]
    # Scenario: 2 by 2 by 5 array of zero to nineteen for every valid dtype over axes minus three to two,
    # all flag combinations. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(0, 20).reshape([2, 2, 5]).astype(dtype)
      for axis in (-3, -2, -1, 0, 1, 2):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test6D(self):
    # [FUNCTION DOC]
    # Scenario: six-dimensional array (2,2,3,3,2,2) of one to 144 for every valid dtype over axes minus
    # six, minus three, zero and three. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
      for axis in range(-6, 6, 3):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123860949")  # The computation is constant folded
  def testLarge(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the version check compares the numpy version string directly with a version-object
    # using a greater-or-equal comparison, which mixes string and object types and may not behave as a
    # real version comparison.
    # Scenario: large input, one million ones divided by 1024, scanned on axis zero in all flag
    # combinations for every valid dtype. Skips float16 on numpy 2.0 or newer and always skips bfloat16
    # (numpy bug 27709). Graph-mode only; XLA disabled since the computation is constant folded
    # (b/123860949).
    for dtype in self.valid_dtypes:
      if np.__version__ >= np.lib.NumpyVersion("2.0.0") and dtype == np.float16:
        continue
      if dtype == dtypes.bfloat16.as_numpy_dtype:
        # https://github.com/numpy/numpy/issues/27709, which might be fixed
        # in some numpy version after 2.1.3.
        continue

      x = np.ones([1000000], dtype=dtype) / 1024
      self._compareAll(x, 0)

  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Scenario: invalid axis errors. Builds a 2 by 5 float32 tensor; in a session expects an
    # invalid-argument error whose message says the scan axis must be in the range minus two to two for
    # axis minus three and for axis two, and a message that the axis must be a scalar when given a
    # one-element list. Runs in the default mode.
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    input_tensor = ops.convert_to_tensor(x)
    with self.session():
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumsum(input_tensor, -3).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumsum(input_tensor, 2).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "axis must be a scalar" in str(e)):
        math_ops.cumsum(input_tensor, [0]).eval()

  def _compareGradient(self, shape, axis, exclusive, reverse):
    # [FUNCTION DOC]
    # HELPER: gradient check for one configuration. Takes shape (list of ints), axis, exclusive and
    # reverse. Builds a float64 array of zero to forty-nine reshaped to shape, converts to a tensor,
    # applies the cumulative sum, computes the theoretical and numeric Jacobians with delta one, and
    # asserts they match within tight relative and absolute tolerance of 1e-8. Returns nothing.
    x = np.arange(0, 50).reshape(shape).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      result = math_ops.cumsum(t, axis, exclusive, reverse)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, shape, result, shape, x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: gradient of a 50-element vector over axes minus one and zero, non-exclusive, forward.
    # Graph-mode only.
    for axis in (-1, 0):
      self._compareGradient([50], axis, False, False)

  @test_util.run_deprecated_v1
  def testGradientReverse(self):
    # [FUNCTION DOC]
    # Scenario: gradient of a 50-element vector over axes minus one and zero, non-exclusive, reversed.
    # Graph-mode only.
    for axis in (-1, 0):
      self._compareGradient([50], axis, False, True)

  @test_util.run_deprecated_v1
  def testGradientExclusive(self):
    # [FUNCTION DOC]
    # Scenario: gradient of a 50-element vector over axes minus one and zero, exclusive, forward.
    # Graph-mode only.
    for axis in (-1, 0):
      self._compareGradient([50], axis, True, False)

  @test_util.run_deprecated_v1
  def testGradientExclusiveReverse(self):
    # [FUNCTION DOC]
    # Scenario: gradient of a 50-element vector over axes minus one and zero, exclusive and reversed.
    # Graph-mode only.
    for axis in (-1, 0):
      self._compareGradient([50], axis, True, True)

  @test_util.run_deprecated_v1
  def testGradient2D(self):
    # [FUNCTION DOC]
    # Scenario: gradient of a 5 by 10 matrix over axes minus one, zero and one for all exclusive and
    # reverse combinations. Graph-mode only.
    for axis in (-1, 0, 1):
      for exclusive in [True, False]:
        for reverse in [True, False]:
          self._compareGradient([5, 10], axis, exclusive, reverse)


class CumprodTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the cumulative product op. Mirrors the cumulative sum tests with the same dtype list
  # (including bfloat16), compares to numpy via the helper, and checks invalid-axis errors and
  # gradients, using smaller gradient inputs because products grow quickly.

  valid_dtypes = [
      np.int32,
      np.int64,
      np.float16,
      np.float32,
      np.float64,
      np.complex64,
      np.complex128,
      dtypes.bfloat16.as_numpy_dtype,
  ]

  def _compare(self, x, axis, exclusive, reverse):
    # [FUNCTION DOC]
    # HELPER: compares the cumulative product for one configuration. Takes x (numpy array), axis,
    # exclusive and reverse. Computes the numpy reference and the TensorFlow result in a cached
    # session, with tolerance 1e-6 (relaxed to 1e-2 for bfloat16 inputs), and asserts closeness.
    # Returns nothing.
    np_out = handle_options(np.cumprod, x, axis, exclusive, reverse)
    with self.cached_session():
      tf_out = math_ops.cumprod(x, axis, exclusive, reverse).eval()

    atol = rtol = 1e-6
    if x.dtype == dtypes.bfloat16.as_numpy_dtype:
      atol = rtol = 1e-2
    self.assertAllClose(np_out, tf_out, atol=atol, rtol=rtol)

  def _compareAll(self, x, axis):
    # [FUNCTION DOC]
    # HELPER: runs the comparison for every combination of exclusive and reverse for the given x and axis.
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)

  @test_util.run_deprecated_v1
  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: empty vector of every valid dtype over axes minus one and zero, all flag combinations.
    # Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.zeros([0]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def testAxisType(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no assertion; it only checks that int32 and int64 axis tensors are accepted without
    # raising, and the output is discarded.
    # Scenario: for each dtype, a one-to-five vector with scalar axis constant zero of each axis dtype,
    # evaluated with the cumulative product. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis_dtype in [dtypes.int64, dtypes.int32]:
        with self.cached_session():
          axis = constant_op.constant(0, axis_dtype)
          tf_out = math_ops.cumprod(x, axis).eval()

  @test_util.run_deprecated_v1
  def testNaN(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the cumulative sum NaN test, bfloat16 is omitted from the dtype list.
    # Scenario: NaN propagation for float16, float32, float64; NaN placed at each position of a one to
    # five vector, axes minus one and zero, all flag combinations. Graph-mode only.
    for dtype in (np.float16, np.float32, np.float64):
      for nan_idx in range(0, 5):
        x = np.arange(1, 6).reshape([5]).astype(dtype)
        x[nan_idx] = np.nan
        for axis in (-1, 0):
          self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test1D(self):
    # [FUNCTION DOC]
    # Scenario: one-dimensional vector of one to five for all valid dtypes, axes minus one and zero, all
    # flag combinations. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test2D(self):
    # [FUNCTION DOC]
    # Scenario: 2 by 5 matrix of one to ten for all valid dtypes, axes minus two to one, all flag
    # combinations. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 11).reshape([2, 5]).astype(dtype)
      for axis in (-2, -1, 0, 1):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test3D(self):
    # [FUNCTION DOC]
    # Scenario: 2 by 2 by 5 array of one to twenty for all valid dtypes, axes minus three to two. Graph-mode
    # only.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 21).reshape([2, 2, 5]).astype(dtype)
      for axis in (-3, -2, -1, 0, 1, 2):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test6D(self):
    # [FUNCTION DOC]
    # Scenario: six-dimensional array (2,2,3,3,2,2) of one to 144 for all valid dtypes over axes minus
    # six, minus three, zero and three. Graph-mode only.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
      for axis in range(-6, 6, 3):
        self._compareAll(x, axis)

  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Scenario: invalid axis errors for the cumulative product, as for the sum: axis minus three and two
    # give invalid-argument errors mentioning the allowed range, and a one-element list axis gives an
    # error that the axis must be a scalar.
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    input_tensor = ops.convert_to_tensor(x)
    with self.session():
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumprod(input_tensor, -3).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumprod(input_tensor, 2).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "axis must be a scalar" in str(e)):
        math_ops.cumprod(input_tensor, [0]).eval()

  def _compareGradient(self, shape, axis, exclusive, reverse):
    # [FUNCTION DOC]
    # HELPER: gradient check for one configuration. Takes shape, axis, exclusive and reverse. Builds a
    # float64 array of one to eight reshaped to shape, applies the cumulative product, compares
    # theoretical and numeric Jacobians with delta one and tolerance 1e-8. Returns nothing.
    x = np.arange(1, 9).reshape(shape).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      result = math_ops.cumprod(t, axis, exclusive, reverse)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, shape, result, shape, x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: gradient for an 8-element vector over axes minus one and zero, non-exclusive, forward.
    # Graph-mode only.
    for axis in (-1, 0):
      self._compareGradient([8], axis, False, False)

  @test_util.run_deprecated_v1
  def testGradientReverse(self):
    # [FUNCTION DOC]
    # Scenario: gradient for an 8-element vector, axes minus one and zero, non-exclusive, reversed.
    # Graph-mode only.
    for axis in (-1, 0):
      self._compareGradient([8], axis, False, True)

  @test_util.run_deprecated_v1
  def testGradientExclusive(self):
    # [FUNCTION DOC]
    # Scenario: gradient for an 8-element vector, axes minus one and zero, exclusive, forward.
    # Graph-mode only.
    for axis in (-1, 0):
      self._compareGradient([8], axis, True, False)

  @test_util.run_deprecated_v1
  def testGradientExclusiveReverse(self):
    # [FUNCTION DOC]
    # Scenario: gradient for an 8-element vector, axes minus one and zero, exclusive and reversed.
    # Graph-mode only.
    for axis in (-1, 0):
      self._compareGradient([8], axis, True, True)

  @test_util.run_deprecated_v1
  def testGradient2D(self):
    # [FUNCTION DOC]
    # Scenario: gradient for a 2 by 4 matrix over axes minus two to one and all flag combinations.
    # Graph-mode only.
    for axis in (-2, -1, 0, 1):
      for exclusive in [True, False]:
        for reverse in [True, False]:
          self._compareGradient([2, 4], axis, exclusive, reverse)


if __name__ == "__main__":
  test.main()
