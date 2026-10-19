# [MODULE DOC]
# XLA compiler tests for reduction ops (sum, product, min, max, mean, all, any) over real,
# complex and boolean data with 32-bit and 64-bit axis index types, including empty inputs,
# negative and invalid axes, duplicate axes, and precision of 16-bit float sums. Results are
# compared with numpy. No import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for reduction operators."""

import functools
import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


@parameterized.named_parameters(('32_bit_index', dtypes.int32),
                                ('64_bit_index', dtypes.int64))
class ReduceOpsTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for reduction ops under XLA. The decorator above makes every test run twice, once
  # with a 32-bit and once with a 64-bit axis index type, passed to each test as its index
  # dtype argument. Class attributes hold shared inputs: REAL_DATA and COMPLEX_DATA (including
  # empty arrays), their NONEMPTY variants, BOOL_DATA and ONES (a 34000 by 2 array of ones).
  def _testReduction(self,
                     tf_reduce_fn,
                     np_reduce_fn,
                     dtype,
                     test_inputs,
                     index_dtype,
                     rtol=1e-4,
                     atol=1e-4):
    # [FUNCTION DOC]
    # HELPER: shared reduction check. For each input array, builds the reduction on the XLA
    # device with the axis fed as a placeholder of the given index dtype; asserts the result
    # equals the numpy reduction along axis 0 (axis list 0), axis 1 (axis list 1), and axis 1
    # again for axis list -1, within rtol/atol (defaults 1e-4). Unless the MLIR bridge is
    # enabled (its error text differs), also asserts that axes -33 and 2 raise an invalid
    # argument error mentioning an invalid reduction dimension. Parameters: tf_reduce_fn (takes
    # data and axis), np_reduce_fn (numpy reference taking an axis), dtype, test_inputs (list of
    # arrays), index_dtype. Returns nothing.
    """Tests that the output of 'tf_reduce_fn' matches numpy's output."""

    for test_input in test_inputs:
      with self.session() as sess:
        with self.test_scope():
          a = array_ops.placeholder(dtype)
          index = array_ops.placeholder(index_dtype)
          out = tf_reduce_fn(a, index)
        result = sess.run(out, {a: test_input, index: [0]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=0), rtol=rtol, atol=atol)

        result = sess.run(out, {a: test_input, index: [1]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=1), rtol=rtol, atol=atol)

        result = sess.run(out, {a: test_input, index: [-1]})
        self.assertAllClose(
            result, np_reduce_fn(test_input, axis=1), rtol=rtol, atol=atol)

        # MLIR bridge doesn't return the same error so it can't be matched
        # directly.
        if not test_util.is_mlir_bridge_enabled():
          with self.assertRaisesWithPredicateMatch(
              errors_impl.InvalidArgumentError, 'Invalid reduction dim'):
            sess.run(out, {a: test_input, index: [-33]})

          with self.assertRaisesWithPredicateMatch(
              errors_impl.InvalidArgumentError, 'Invalid reduction dim'):
            sess.run(out, {a: test_input, index: [2]})

  REAL_DATA = [
      np.zeros(shape=(2, 0)),
      np.zeros(shape=(0, 30)),
      np.arange(1, 7).reshape(2, 3),
      np.arange(-10, -4).reshape(2, 3),
      np.arange(-4, 2).reshape(2, 3),
  ]
  COMPLEX_DATA = [
      np.zeros(shape=(2, 0)).astype(np.complex64),
      np.zeros(shape=(0, 30)).astype(np.complex64),
      np.arange(1, 13, dtype=np.float32).view(np.complex64).reshape(2, 3),
      np.arange(-14, -2, dtype=np.float32).view(np.complex64).reshape(2, 3),
      np.arange(-4, 8, dtype=np.float32).view(np.complex64).reshape(2, 3),
  ]
  NONEMPTY_REAL_DATA = [x for x in REAL_DATA if np.size(x) > 0]
  NONEMPTY_COMPLEX_DATA = [x for x in COMPLEX_DATA if np.size(x) > 0]
  BOOL_DATA = [
      np.array([], dtype=np.bool_).reshape(2, 0),
      np.array([], dtype=np.bool_).reshape(0, 3),
      np.array([[False, True, False], [True, True, False]]),
  ]
  ONES = [np.ones([34000, 2])]

  def testReduceSumF32(self, index_dtype):
    # [FUNCTION DOC]
    # Reduce-sum check on single-precision real data (including empty arrays).
    self._testReduction(math_ops.reduce_sum, np.sum, np.float32, self.REAL_DATA,
                        index_dtype)

  def testReduceSumC64(self, index_dtype):
    # [FUNCTION DOC]
    # Reduce-sum check on complex64 data (including empty arrays).
    self._testReduction(math_ops.reduce_sum, np.sum, np.complex64,
                        self.COMPLEX_DATA, index_dtype)

  def testReduceProdF32(self, index_dtype):
    # [FUNCTION DOC]
    # Reduce-product check on single-precision real data (including empty arrays).
    self._testReduction(math_ops.reduce_prod, np.prod, np.float32,
                        self.REAL_DATA, index_dtype)

  def testReduceProdC64(self, index_dtype):
    # [FUNCTION DOC]
    # Reduce-product check on complex64 data (including empty arrays).
    self._testReduction(math_ops.reduce_prod, np.prod, np.complex64,
                        self.COMPLEX_DATA, index_dtype)

  def testReduceMin(self, index_dtype):
    # [FUNCTION DOC]
    # Reduce-min check for float32, int32 and int64 (those supported by the device) on real data,
    # using a reference that yields positive infinity (or the integer maximum) for empty
    # reductions.

    def reference_min(dtype, inp, axis):
      # [FUNCTION DOC]
      # HELPER: numpy reference for minimum. Takes the dtype, input array and axis; for an empty
      # reduced axis returns an array of the remaining shape filled with positive infinity (float
      # types) or the largest integer of that dtype; otherwise returns the numpy minimum.
      """Wrapper around np.amin that returns +infinity for an empty input."""
      if inp.shape[axis] == 0:
        if np.issubdtype(dtype, np.floating):
          return np.full(inp.shape[0:axis] + inp.shape[axis + 1:], float('inf'))
        return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                       np.iinfo(dtype).max)
      return np.amin(inp, axis)

    for dtype in set(self.all_types).intersection(
        [np.float32, np.int32, np.int64]):
      self._testReduction(math_ops.reduce_min,
                          functools.partial(reference_min, dtype), dtype,
                          self.REAL_DATA, index_dtype)

  def testReduceMax(self, index_dtype):
    # [FUNCTION DOC]
    # Reduce-max check for float32, int32 and int64 (those supported by the device) on real data,
    # using a reference that yields negative infinity (or the integer minimum) for empty
    # reductions.

    def reference_max(dtype, inp, axis):
      # [FUNCTION DOC]
      # HELPER: numpy reference for maximum. Takes the dtype, input array and axis; for an empty
      # reduced axis returns an array of the remaining shape filled with negative infinity (float
      # types) or the smallest integer of that dtype; otherwise returns the numpy maximum.
      """Wrapper around np.amax that returns -infinity for an empty input."""
      if inp.shape[axis] == 0:
        if np.issubdtype(dtype, np.floating):
          return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                         float('-inf'))
        return np.full(inp.shape[0:axis] + inp.shape[axis + 1:],
                       np.iinfo(dtype).min)
      return np.amax(inp, axis)

    for dtype in set(self.all_types).intersection(
        [np.float32, np.int32, np.int64]):
      self._testReduction(math_ops.reduce_max,
                          functools.partial(reference_max, dtype), dtype,
                          self.REAL_DATA, index_dtype)

  def testReduceMeanF32(self, index_dtype):
    # [FUNCTION DOC]
    # Reduce-mean check on non-empty single-precision data (empty inputs excluded because XLA
    # returns 0 rather than NaN for them).
    # TODO(phawkins): mean on XLA currently returns 0 instead of NaN when
    # reducing across zero inputs.
    self._testReduction(math_ops.reduce_mean, np.mean, np.float32,
                        self.NONEMPTY_REAL_DATA, index_dtype)

  def testReduceMeanF16(self, index_dtype):
    # [FUNCTION DOC]
    # If float16 is supported by the device, checks reduce-mean over a 34000 by 2 array of ones
    # (large enough to expose accumulation precision problems); otherwise does nothing.
    if np.float16 in self.all_types:
      self._testReduction(math_ops.reduce_mean, np.mean, np.float16, self.ONES,
                          index_dtype)

  def testReduceMeanC64(self, index_dtype):
    # [FUNCTION DOC]
    # Reduce-mean check on non-empty complex64 data.
    self._testReduction(math_ops.reduce_mean, np.mean, np.complex64,
                        self.NONEMPTY_COMPLEX_DATA, index_dtype)

  def testReduceAll(self, index_dtype):
    # [FUNCTION DOC]
    # Logical-and reduction check on boolean data (including empty arrays).
    self._testReduction(math_ops.reduce_all, np.all, np.bool_, self.BOOL_DATA,
                        index_dtype)

  def testReduceAny(self, index_dtype):
    # [FUNCTION DOC]
    # Logical-or reduction check on boolean data (including empty arrays).
    self._testReduction(math_ops.reduce_any, np.any, np.bool_, self.BOOL_DATA,
                        index_dtype)

  @test_util.disable_mlir_bridge('Error messages differ')
  def testReduceSumWithDuplicateAxes(self, index_dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the index_dtype parameter is accepted but ignored; the axis placeholder is
    # always 32-bit integer, so both parameterized variants run the same check.
    # Reduces a float vector over axes listed as zero twice and asserts an invalid-argument error
    # saying the axes contain a duplicate dimension. Skipped under the MLIR bridge (decorator)
    # because its error messages differ.
    with self.session() as sess:
      with self.test_scope():
        a = array_ops.placeholder(np.float32)
        index = array_ops.placeholder(np.int32)
        out = math_ops.reduce_sum(a, index)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          'Axes contains duplicate dimension'):
        sess.run(out, {a: [10, 20, 30], index: [0, 0]})


class ReduceOpPrecisionTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case checking that reduce-sum for 16-bit float types (half and bfloat16) does not lose
  # too much precision when intermediate sums would overflow if accumulated naively.

  def _testReduceSum(self,
                     expected_result,
                     dtype,
                     test_inputs,
                     rtol=1e-3,
                     atol=1e-4):
    # [FUNCTION DOC]
    # HELPER: for each input in test_inputs (converted to the given dtype), reduces along axis 0
    # on the XLA device and asserts the result, compared as float32, is close to expected_result
    # within rtol (default 1e-3) and atol (default 1e-4). Returns nothing.
    """Tests reduce sum on a list of input arrays.

    For each array in test_inputs, check that performing reduce sum on the array
    produces a value that is close to the expected result.

    Args:
      expected_result: the expected result.
      dtype: the data type of the reduce sum operation.
      test_inputs: a list of input arrays for the reduce sum operation.
      rtol: the relative error.
      atol: the absolute error.
    """

    for test_input in test_inputs:
      with self.session() as sess:
        with self.test_scope():
          a = array_ops.placeholder(dtype)
          index = array_ops.placeholder(dtypes.int32)
          out = math_ops.reduce_sum(a, index)
        result = sess.run(out, {
            a: np.array(test_input, dtype=dtype),
            index: [0]
        })
        # Compare the results using float32 type.
        self.assertAllClose(
            np.float32(result),
            np.float32(expected_result),
            rtol=rtol,
            atol=atol)

  def testReduceSumF16(self):
    # [FUNCTION DOC]
    # Skips if float16 is unsupported. Otherwise sums every ordering of the three values (max,
    # max, minus max) of float16 and expects the result to equal float16 max, which requires
    # that intermediate sums do not overflow or lose precision.
    """Tests the reduce sum of float16 doesn't lose too much precision."""

    if np.float16 not in self.all_types:
      return

    f16_max = np.finfo(np.float16).max
    self._testReduceSum(
        f16_max, np.float16,
        itertools.permutations([f16_max, f16_max, f16_max * (-1.0)], 3))

  def testReduceSumBF16(self):
    # [FUNCTION DOC]
    # Skips if bfloat16 is unsupported. Otherwise sums every ordering of (bfloat16 max, a value of
    # half the smaller of bfloat16 max and the headroom to float32 max, minus bfloat16 max) and
    # expects the middle value, again checking that accumulation does not overflow.
    """Tests the reduce sum of bfloat16 doesn't lose too much precision."""

    if dtypes.bfloat16.as_numpy_dtype not in self.all_types:
      return

    bf16_max = np.float32(dtypes.bfloat16.max)
    f32_max = dtypes.float32.max
    value = min(bf16_max, f32_max - bf16_max) / 2
    self._testReduceSum(
        dtypes.bfloat16.as_numpy_dtype(value), dtypes.bfloat16.as_numpy_dtype,
        itertools.permutations([bf16_max, value, bf16_max * (-1.0)], 3))


if __name__ == '__main__':
  googletest.main()
