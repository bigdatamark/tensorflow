# [MODULE DOC]
# Functional tests for the cumulative scan ops (cumulative sum, cumulative log-sum-exp and
# cumulative product) under the XLA test harness. Module-level helpers build numpy reference
# results honoring the exclusive and reverse options; three test classes compare device results
# with them across ranks, axes and types, and check axis validation errors.
# No import-time side effects beyond imports; as a script it starts the test runner.
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

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def numpy_reverse(x, axis):
  # [FUNCTION DOC]
  # HELPER: Numpy reference utility for the scan tests.
  # Inputs: x is a numpy array; axis is an integer axis, possibly negative (counted from the
  # end). Returns a view of x with the order of elements along that axis flipped. Mutates nothing.
  length = len(x.shape)
  if axis < 0:
    axis = length + axis

  ix = tuple(
      slice(None, None, -1) if i == axis else slice(None) for i in range(length)
  )
  return x[ix]


def handle_options(func, init_fn, x, axis, exclusive, reverse):
  # [FUNCTION DOC]
  # HELPER: Builds the numpy reference result for the scan tests, adding the exclusive and
  # reverse options that numpy scans lack.
  # Inputs: func is a numpy accumulate function taking an array and an axis; init_fn maps an
  # array slice to the identity-valued first element used in exclusive mode (zeros, ones or
  # negative infinity); x is the numpy input; axis is an integer, possibly negative; exclusive
  # and reverse are booleans. If reverse, flips x along the axis first. If exclusive, the output
  # starts with the identity element followed by the scan of all but the last element along the
  # axis; otherwise it is the plain scan. If reverse, the result is flipped back.
  # Returns the numpy result array. Mutates nothing.
  """Adds tf options to numpy scan ops."""
  length = len(x.shape)
  if axis < 0:
    axis = length + axis

  if reverse:
    x = numpy_reverse(x, axis)

  if exclusive:
    ix_head = tuple(slice(0, 1) if i == axis else slice(None)
                    for i in range(length))
    ix_init = tuple(
        slice(0, -1) if i == axis else slice(None) for i in range(length)
    )
    init = init_fn(x[ix_head])
    x = np.concatenate([init, func(x[ix_init], axis=axis)], axis=axis)
  else:
    x = func(x, axis=axis)

  if reverse:
    x = numpy_reverse(x, axis)
  return x


class CumsumTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the cumulative sum op. Compares device results against a numpy reference
  # for empty, one-, two-, three- and six-dimensional inputs, every axis, and all combinations of
  # the exclusive and reverse options; also checks axis dtypes, bfloat16 input and axis errors.
  # State: a class attribute listing the value types tested (32-bit float, 32- and 64-bit ints).

  valid_dtypes = [np.float32, np.int32, np.int64]

  def axis_dtypes(self):
    # [FUNCTION DOC]
    # Returns the set of integer types (32-bit and 64-bit) that are both supported by the
    # current device and allowed for the axis argument.
    return set(self.int_types).intersection([np.int32, np.int64])

  def _compare(self, x, axis, exclusive, reverse):
    # [FUNCTION DOC]
    # HELPER: Shared comparison for the cumulative sum tests.
    # Inputs: x numpy array, axis integer, exclusive and reverse booleans. Computes the numpy
    # reference with zero as identity, runs the cumulative sum op on the device inside a session
    # with a placeholder fed with x, and asserts the two results are close. Returns nothing.
    np_out = handle_options(np.cumsum, np.zeros_like, x, axis, exclusive,
                            reverse)
    with self.session(), self.test_scope():
      p = array_ops.placeholder(x.dtype)
      tf_out = math_ops.cumsum(p, axis, exclusive, reverse).eval(
          feed_dict={p: x})

    self.assertAllClose(np_out, tf_out)

  def _compareAll(self, x, axis):
    # [FUNCTION DOC]
    # HELPER: Runs the comparison for x and axis over all four combinations of the exclusive and
    # reverse flags. Returns nothing.
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)

  def testEmpty(self):
    # [FUNCTION DOC]
    # For each tested type, scans a zero-length vector along axes minus one and zero with all
    # option combinations and expects agreement with the reference. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.zeros([0]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  def testAxisType(self):
    # [FUNCTION DOC]
    # Checks that the axis argument may be a constant of each supported integer type. For each
    # value type and axis type, runs the scan on a five-element vector with axis zero; only
    # asserts that evaluation succeeds. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis_dtype in self.axis_dtypes():
        with self.session(), self.test_scope():
          p = array_ops.placeholder(x.dtype)
          axis = constant_op.constant(0, axis_dtype)
          math_ops.cumsum(p, axis).eval(feed_dict={p: x})

  def test1D(self):
    # [FUNCTION DOC]
    # Compares scans of a five-element vector against the reference for axes minus one and zero,
    # all option combinations, all tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  def test2D(self):
    # [FUNCTION DOC]
    # Compares scans of a two-by-five matrix against the reference for axes minus two through
    # one, all option combinations, all tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(0, 10).reshape([2, 5]).astype(dtype)
      for axis in (-2, -1, 0, 1):
        self._compareAll(x, axis)

  def test3D(self):
    # [FUNCTION DOC]
    # Compares scans of a two-by-two-by-five tensor against the reference for axes minus three
    # through two, all option combinations, all tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(0, 20).reshape([2, 2, 5]).astype(dtype)
      for axis in (-3, -2, -1, 0, 1, 2):
        self._compareAll(x, axis)

  def test6D(self):
    # [FUNCTION DOC]
    # Compares scans of a six-dimensional tensor (shape two, two, three, three, two, two) against
    # the reference for axes minus six, minus three, zero and three, all option combinations and
    # tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
      for axis in range(-6, 6, 3):
        self._compareAll(x, axis)

  def testMixedPrecision(self):
    # [FUNCTION DOC]
    # Runs an exclusive cumulative sum over a bfloat16 vector of one through four on axis minus
    # one and expects the values zero, one, three, six. Returns nothing.
    with self.session(), self.test_scope():
      y = math_ops.cumsum(
          constant_op.constant([1., 2., 3., 4.], dtypes.bfloat16),
          -1,
          exclusive=True).eval()
    self.assertAllEqual(y, [0., 1., 3., 6.])

  @test_util.disable_mlir_bridge("Error handling")
  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge because error handling differs. On a two-by-five float matrix,
    # expects an invalid-argument error saying the scan axis must be in the range minus two
    # (inclusive) to two (exclusive) for axis minus three and for axis two, and an
    # invalid-argument error saying the axis must be a scalar for a one-element list axis.
    # Returns nothing.
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    with self.session(), self.test_scope():
      input_tensor = ops.convert_to_tensor(x)
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


class CumulativeLogSumExpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the cumulative log-sum-exp op. Same structure as the cumulative sum tests
  # but the reference is the numpy log-add-exp accumulation with negative infinity as the
  # identity for the exclusive option, and a relative tolerance is used.
  # State: a class attribute listing the value types tested (32- and 64-bit floats).

  valid_dtypes = [np.float32, np.float64]

  def axis_dtypes(self):
    # [FUNCTION DOC]
    # Returns the set of integer types (32-bit and 64-bit) that are both supported by the
    # current device and allowed for the axis argument.
    return set(self.int_types).intersection([np.int32, np.int64])

  def _compare(self, x, axis, exclusive, reverse):
    # [FUNCTION DOC]
    # HELPER: Shared comparison for the cumulative log-sum-exp tests.
    # Inputs: x numpy array, axis integer, exclusive and reverse booleans. Computes the numpy
    # reference using log-add-exp accumulation with negative infinity as the identity, runs the
    # op on the device in a session with a placeholder fed with x, and asserts closeness with
    # relative tolerance four times ten to the minus five. Returns nothing.

    def neginf_like(x):
      # [FUNCTION DOC]
      # HELPER: Supplies the identity element for exclusive log-sum-exp scans.
      # Takes a numpy array and returns an array of the same shape filled with negative infinity.
      return -np.inf * np.ones_like(x)

    np_out = handle_options(np.logaddexp.accumulate, neginf_like, x, axis,
                            exclusive, reverse)
    with self.session(), self.test_scope():
      p = array_ops.placeholder(x.dtype)
      tf_out = math_ops.cumulative_logsumexp(p, axis, exclusive,
                                             reverse).eval(feed_dict={p: x})

    self.assertAllClose(np_out, tf_out, rtol=4e-5)

  def _compareAll(self, x, axis):
    # [FUNCTION DOC]
    # HELPER: Runs the comparison for x and axis over all four combinations of the exclusive and
    # reverse flags. Returns nothing.
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)

  def testEmpty(self):
    # [FUNCTION DOC]
    # For each tested type, scans a zero-length vector along axes minus one and zero with all
    # option combinations and expects agreement with the reference. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.zeros([0]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  def testAxisType(self):
    # [FUNCTION DOC]
    # Checks that the axis argument may be a constant of each supported integer type. For each
    # value type and axis type, runs the scan on a five-element vector with axis zero; only
    # asserts that evaluation succeeds. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis_dtype in self.axis_dtypes():
        with self.session(), self.test_scope():
          p = array_ops.placeholder(x.dtype)
          axis = constant_op.constant(0, axis_dtype)
          math_ops.cumulative_logsumexp(p, axis).eval(feed_dict={p: x})

  def test1D(self):
    # [FUNCTION DOC]
    # Compares scans of a five-element vector against the reference for axes minus one and zero,
    # all option combinations, all tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  def test2D(self):
    # [FUNCTION DOC]
    # Compares scans of a two-by-five matrix against the reference for axes minus two through
    # one, all option combinations, all tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(0, 10).reshape([2, 5]).astype(dtype)
      for axis in (-2, -1, 0, 1):
        self._compareAll(x, axis)

  def test3D(self):
    # [FUNCTION DOC]
    # Compares scans of a two-by-two-by-five tensor against the reference for axes minus three
    # through two, all option combinations, all tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(0, 20).reshape([2, 2, 5]).astype(dtype)
      for axis in (-3, -2, -1, 0, 1, 2):
        self._compareAll(x, axis)

  def test6D(self):
    # [FUNCTION DOC]
    # Compares scans of a six-dimensional tensor (shape two, two, three, three, two, two) against
    # the reference for axes minus six, minus three, zero and three, all option combinations and
    # tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
      for axis in range(-6, 6, 3):
        self._compareAll(x, axis)

  @test_util.disable_mlir_bridge("Error handling")
  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge because error handling differs. On a two-by-five float matrix,
    # expects an invalid-argument error saying the scan axis must be in the range minus two
    # (inclusive) to two (exclusive) for axis minus three and for axis two, and an
    # invalid-argument error saying the axis must be a scalar for a one-element list axis.
    # Returns nothing.
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    with self.session(), self.test_scope():
      input_tensor = ops.convert_to_tensor(x)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumulative_logsumexp(input_tensor, -3).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumulative_logsumexp(input_tensor, 2).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "axis must be a scalar" in str(e)):
        math_ops.cumulative_logsumexp(input_tensor, [0]).eval()


class CumprodTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the cumulative product op. Same structure as the cumulative sum tests with
  # one as the identity for the exclusive option; inputs start at one to keep products nonzero.
  # State: a class attribute listing the value types tested (32-bit float and 32-bit integer).

  valid_dtypes = [np.float32, np.int32]

  def axis_dtypes(self):
    # [FUNCTION DOC]
    # Returns the set of integer types (32-bit and 64-bit) that are both supported by the
    # current device and allowed for the axis argument.
    return set(self.int_types).intersection([np.int32, np.int64])

  def _compare(self, x, axis, exclusive, reverse):
    # [FUNCTION DOC]
    # HELPER: Shared comparison for the cumulative product tests.
    # Inputs: x numpy array, axis integer, exclusive and reverse booleans. Computes the numpy
    # reference with one as identity, runs the cumulative product op on the device in a session
    # with a placeholder fed with x, and asserts the two results are close. Returns nothing.
    np_out = handle_options(np.cumprod, np.ones_like, x, axis, exclusive,
                            reverse)
    with self.session(), self.test_scope():
      p = array_ops.placeholder(x.dtype)
      prod = math_ops.cumprod(p, axis, exclusive, reverse)
      tf_out = prod.eval(feed_dict={p: x})

    self.assertAllClose(np_out, tf_out)

  def _compareAll(self, x, axis):
    # [FUNCTION DOC]
    # HELPER: Runs the comparison for x and axis over all four combinations of the exclusive and
    # reverse flags. Returns nothing.
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)

  def testEmpty(self):
    # [FUNCTION DOC]
    # For each tested type, scans a zero-length vector along axes minus one and zero with all
    # option combinations and expects agreement with the reference. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.zeros([0]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  def testAxisType(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the sibling tests, the op is applied directly to the numpy array x
    # instead of the placeholder, so the placeholder is created and fed but never used by the op.
    # Checks that the axis argument may be a constant of each supported integer type. For each
    # value type and axis type, runs the product scan on a five-element vector with axis zero;
    # only asserts that evaluation succeeds. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis_dtype in self.axis_dtypes():
        with self.session(), self.test_scope():
          p = array_ops.placeholder(x.dtype)
          axis = constant_op.constant(0, axis_dtype)
          math_ops.cumprod(x, axis).eval(feed_dict={p: x})

  def test1D(self):
    # [FUNCTION DOC]
    # Compares scans of a five-element vector against the reference for axes minus one and zero,
    # all option combinations, all tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  def test2D(self):
    # [FUNCTION DOC]
    # Compares scans of a two-by-five matrix holding one through ten against the reference for
    # axes minus two through one, all option combinations, all tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 11).reshape([2, 5]).astype(dtype)
      for axis in (-2, -1, 0, 1):
        self._compareAll(x, axis)

  def test3D(self):
    # [FUNCTION DOC]
    # Compares scans of a two-by-two-by-five tensor holding one through twenty against the
    # reference for axes minus three through two, all option combinations, all tested types.
    # Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 21).reshape([2, 2, 5]).astype(dtype)
      for axis in (-3, -2, -1, 0, 1, 2):
        self._compareAll(x, axis)

  def test6D(self):
    # [FUNCTION DOC]
    # Compares scans of a six-dimensional tensor (shape two, two, three, three, two, two) against
    # the reference for axes minus six, minus three, zero and three, all option combinations and
    # tested types. Returns nothing.
    for dtype in self.valid_dtypes:
      x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
      for axis in range(-6, 6, 3):
        self._compareAll(x, axis)

  @test_util.disable_mlir_bridge("Error handling")
  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge because error handling differs. On a two-by-five float matrix,
    # expects an invalid-argument error saying the scan axis must be in the range minus two
    # (inclusive) to two (exclusive) for axis minus three and for axis two, and an
    # invalid-argument error saying the axis must be a scalar for a one-element list axis.
    # Returns nothing.
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    with self.session(), self.test_scope():
      input_tensor = ops.convert_to_tensor(x)
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


if __name__ == "__main__":
  test.main()
