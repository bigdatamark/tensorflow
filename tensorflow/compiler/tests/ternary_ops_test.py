# [MODULE DOC]
# Test module for ternary (three-input) operators on the XLA test device: linspace, range, select
# (two versions, with broadcasting), slice, clip by value and the regularized incomplete beta
# function (compared against SciPy). Contains one test class with a shared helper that feeds three
# placeholders and compares to an expected value. No import-time side effects; running as a script
# starts the test runner.
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
"""Test cases for ternary operators."""

from absl.testing import parameterized
import numpy as np
import scipy.special as sps

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class TernaryOpsTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for ternary operators on the XLA test device. Inherits XLA fixtures (session, test
  # scope, sets of supported numeric types) and parameterized test support. Holds no state; tests
  # call the shared helper with an operator, three input arrays and the expected result.

  def _testTernary(self, op, a, b, c, expected, rtol=1e-3, atol=1e-6):
    # [FUNCTION DOC]
    # HELPER: shared check for all ternary operator tests.
    # Inputs: op (function of three tensors), a, b, c (numpy arrays or scalars of any dtype),
    # expected (array to compare against), and optional rtol (default 0.001) and atol (default
    # 0.000001) tolerances. Creates a placeholder for each input matching its dtype and shape,
    # applies op in the XLA test scope, runs it feeding the arrays and asserts the result is close
    # to expected. Returns the computed result.
    with self.session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name="b")
        pc = array_ops.placeholder(dtypes.as_dtype(c.dtype), c.shape, name="c")
        output = op(pa, pb, pc)
      result = session.run(output, {pa: a, pb: b, pc: c})
      self.assertAllClose(result, expected, rtol=rtol, atol=atol)
      return result

  @parameterized.parameters(
      {'start': 1, 'end': 2, 'num': 1},
      {'start': 1, 'end': 4, 'num': 3},
      {'start': 0, 'end': 41, 'num': 42})
  @test_util.disable_mlir_bridge(
      'TODO(b/156174708): Dynamic result types not supported')
  def testLinspace(self, start, end, num):
    # [FUNCTION DOC]
    # Parameterized test (skipped under the MLIR bridge, dynamic result types unsupported) of the
    # linspace op. Inputs: start, end (numbers) and num (count of points). Compares with NumPy's
    # linspace in 32-bit float and additionally asserts the first element equals start and the last
    # equals end exactly, as the op specification requires.
    expected = np.linspace(start, end, num, dtype=np.float32)
    result = self._testTernary(
        math_ops.linspace,
        np.float32(start),
        np.float32(end),
        np.int32(num),
        expected)
    # According to linspace spec, start has to be the first element and end has
    # to be last element.
    self.assertEqual(result[-1], expected[-1])
    self.assertEqual(result[0], expected[0])

  def testRange(self):
    # [FUNCTION DOC]
    # Test: the range op on 32-bit integers: 1 to 2 step 1 gives [1]; 1 to 7 step 2 gives [1, 3, 5].
    self._testTernary(
        math_ops.range,
        np.int32(1),
        np.int32(2),
        np.int32(1),
        expected=np.array([1], dtype=np.int32))
    self._testTernary(
        math_ops.range,
        np.int32(1),
        np.int32(7),
        np.int32(2),
        expected=np.array([1, 3, 5], dtype=np.int32))

  def testSelect(self):
    # [FUNCTION DOC]
    # Test: the first-version where op, for every numeric type. Cases: scalar condition false with
    # scalar branches; scalar condition true with vectors; scalar false with matrices; vector
    # condition choosing elementwise between vectors; and vector condition selecting whole rows of
    # matrices. Asserts each result equals the expected selection.
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.where,
          np.array(False),
          np.array(2, dtype=dtype),
          np.array(7, dtype=dtype),
          expected=np.array(7, dtype=dtype))

      self._testTernary(
          array_ops.where,
          np.array(True),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([1, 2, 3, 4], dtype=dtype))

      self._testTernary(
          array_ops.where,
          np.array(False),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype))

      self._testTernary(
          array_ops.where,
          np.array([0, 1, 1, 0], dtype=np.bool_),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([5, 2, 3, 8], dtype=dtype))

      self._testTernary(
          array_ops.where,
          np.array([0, 1, 0], dtype=np.bool_),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [3, 4], [11, 12]], dtype=dtype))

  def testSelectV2(self):
    # [FUNCTION DOC]
    # Test: the second-version where op (with full broadcasting), for every numeric type. Same
    # basic cases as the first version, plus broadcasting of the condition across rows, broadcasting
    # the then-branch to the else-branch shape, the else-branch to the then-branch shape, and
    # scalar or vector branches broadcast to the condition's shape. Asserts expected selections.
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.where_v2,
          np.array(False),
          np.array(2, dtype=dtype),
          np.array(7, dtype=dtype),
          expected=np.array(7, dtype=dtype))

      self._testTernary(
          array_ops.where_v2,
          np.array(True),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([1, 2, 3, 4], dtype=dtype))

      self._testTernary(
          array_ops.where_v2,
          np.array(False),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype))

      self._testTernary(
          array_ops.where_v2,
          np.array([0, 1, 1, 0], dtype=np.bool_),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([5, 2, 3, 8], dtype=dtype))

      # Broadcast the condition
      self._testTernary(
          array_ops.where_v2,
          np.array([0, 1], dtype=np.bool_),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 2], [9, 4], [11, 6]], dtype=dtype))

      # Broadcast the then branch to the else
      self._testTernary(
          array_ops.where_v2,
          np.array([[0, 1], [1, 0], [1, 1]], dtype=np.bool_),
          np.array([[1, 2]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 2], [1, 10], [1, 2]], dtype=dtype))

      # Broadcast the else branch to the then
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [0, 0]], dtype=np.bool_),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[7, 2], [1, 10], [1, 2]], dtype=dtype))

      # Broadcast the then/else branches to the condition
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [1, 1]], dtype=np.bool_),
          np.array(7, dtype=dtype),
          np.array(8, dtype=dtype),
          expected=np.array([[7, 8], [8, 7], [7, 7]], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [0, 0]], dtype=np.bool_),
          np.array(7, dtype=dtype),
          np.array([8, 9], dtype=dtype),
          expected=np.array([[7, 9], [8, 7], [8, 9]], dtype=dtype))

  def testSlice(self):
    # [FUNCTION DOC]
    # Test: the slice op for every numeric type: slicing zero columns from a 3 by 0 matrix (begin 1,0
    # size 2,0) gives a 2 by 0 matrix; slicing a 3 by 3 matrix at begin 0,1 with size 2,1 gives the
    # column [[2], [5]].
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.slice,
          np.array([[], [], []], dtype=dtype),
          np.array([1, 0], dtype=np.int32),
          np.array([2, 0], dtype=np.int32),
          expected=np.array([[], []], dtype=dtype),
      )

      self._testTernary(
          array_ops.slice,
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype),
          np.array([0, 1], dtype=np.int32),
          np.array([2, 1], dtype=np.int32),
          expected=np.array([[2], [5]], dtype=dtype),
      )

  def testClipByValue(self):
    # [FUNCTION DOC]
    # Test: the internal clip-by-value op for numeric types excluding complex and unsigned
    # integer types. Clips the vector [-2, 10, 6] with array/scalar bound combinations (array lower
    # with scalar upper, scalar lower with array upper, and two arrays) and compares with the
    # elementwise minimum of the upper bound and maximum of the lower bound.
    for dtype in (
        self.numeric_types - self.complex_types - self.unsigned_int_types
    ):
      test_cases = [
          (np.array([2, 4, 5], dtype=dtype), dtype(7)),  #
          (dtype(1), np.array([2, 4, 5], dtype=dtype)),  #
          (
              np.array([-2, 7, 7], dtype=dtype),
              np.array([-2, 9, 8], dtype=dtype),
          ),
      ]
      x = np.array([-2, 10, 6], dtype=dtype)
      for lower, upper in test_cases:
        self._testTernary(
            gen_math_ops._clip_by_value,
            x,
            lower,
            upper,
            expected=np.minimum(np.maximum(x, lower), upper))

  def testBetaincSanity(self):
    # [FUNCTION DOC]
    # Test: sanity check of the regularized incomplete beta function for float32 and float64 (the
    # only supported types), on four fixed triples including boundary values of x (0) and b (1).
    # Compares with SciPy using relative tolerance 5e-5 and absolute tolerance 6e-5.
    # This operation is only supported for float32 and float64.
    for dtype in self.numeric_types & {np.float32, np.float64}:
      # Sanity check a few identities:
      # - betainc(a, b, 0) == 0
      # - betainc(a, b, 1) == 1
      # - betainc(a, 1, x) == x ** a
      # Compare against the implementation in SciPy.
      a = np.array([.3, .4, .2, .2], dtype=dtype)
      b = np.array([1., 1., .4, .4], dtype=dtype)
      x = np.array([.3, .4, .0, .1], dtype=dtype)
      expected = sps.betainc(a, b, x)
      self._testTernary(
          math_ops.betainc, a, b, x, expected, rtol=5e-5, atol=6e-5
      )

  @parameterized.parameters(
      {
          'sigma': 1e15,
          'rtol': 1e-6,
          'atol': 1e-4
      },
      {
          'sigma': 30,
          'rtol': 1e-6,
          'atol': 2e-3
      },
      {
          'sigma': 1e-8,
          'rtol': 5e-4,
          'atol': 3e-4
      },
      {
          'sigma': 1e-16,
          'rtol': 1e-6,
          'atol': 2e-4
      },
  )
  def testBetainc(self, sigma, rtol, atol):
    # [FUNCTION DOC]
    # Parameterized test of the regularized incomplete beta function for float32 and float64.
    # Inputs: sigma (scale of the random shape parameters), rtol and atol (comparison tolerances).
    # Draws 10 by 10 random arrays: a and b as absolute values of normal samples scaled by sigma,
    # and x uniform in the unit interval. Compares with SciPy within the tolerances. The random
    # draws use the global NumPy generator without a fixed seed.
    # This operation is only supported for float32 and float64.
    for dtype in self.numeric_types & {np.float32, np.float64}:
      # Randomly generate a, b, x in the numerical domain of betainc.
      # Compare against the implementation in SciPy.
      a = np.abs(np.random.randn(10, 10) * sigma).astype(dtype)  # in (0, infty)
      b = np.abs(np.random.randn(10, 10) * sigma).astype(dtype)  # in (0, infty)
      x = np.random.rand(10, 10).astype(dtype)  # in (0, 1)
      expected = sps.betainc(a, b, x, dtype=dtype)
      self._testTernary(
          math_ops.betainc, a, b, x, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
  googletest.main()
