# [MODULE DOC]
# Unit tests for the mathematical functions of the experimental NumPy API in np_math_ops (binary
# and unary ops, dot/matmul/vdot, argsort, argmax/argmin, isclose, average, clip, ptp, linspace,
# logspace, geomspace, tensor numpy-method availability, isinf variants). Results are compared
# with real NumPy for value, dtype and shape using a shared match helper, over many input array
# kinds (Python lists, numpy arrays, tensors, framework arrays of several float types). When run
# as a script it enables tensor equality, eager execution, legacy dtype conversion mode and numpy
# methods on tensors, then runs the test main.
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
"""Tests for tf numpy mathematical methods."""

import itertools
from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


class MathTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the math functions in np_math_ops. Shared fixtures (set in setUp): a list
  # of transforms that turn a Python value into different array kinds, and a list of numeric
  # types used for type-promotion checks. Provides comparison helpers for unary and binary ops.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: runs the parent setUp, then defines array_transforms (identity, tensor conversion,
    # numpy array in default/float32/float64, framework array in default/float32/float64) and
    # types (32/64-bit integers and floats).
    super(MathTest, self).setUp()
    self.array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64),
    ]
    self.types = [np.int32, np.int64, np.float32, np.float64]

  def _testBinaryOp(self,
                    math_fun,
                    np_fun,
                    name,
                    operands=None,
                    extra_operands=None,
                    check_promotion=True,
                    check_promotion_result_type=True):
    # [FUNCTION DOC]
    # HELPER: shared driver for the binary-operation tests.
    # Compares a framework binary function with its NumPy counterpart. Inputs: math_fun and
    # np_fun (the two functions), name (used in failure messages), operands (optional list of
    # operand pairs; default is scalar/vector/matrix combinations), extra_operands (optional
    # additional pairs always tested), check_promotion (if false only equal-typed pairs are used
    # in the promotion phase), check_promotion_result_type (if false result dtype is not
    # compared in the promotion phase). For each pair it applies every array transform to both
    # operands and matches results, then tries all pairs of numeric types. Returns nothing;
    # asserts via the match helper.

    def run_test(a, b):
      # [FUNCTION DOC]
      # HELPER: tests one operand pair. First over every array transform applied to both
      # operands, then over all combinations of the numeric types (type promotion), matching the
      # framework result to the NumPy result.
      for fn in self.array_transforms:
        arg1 = fn(a)
        arg2 = fn(b)
        self.match(
            math_fun(arg1, arg2),
            np_fun(arg1, arg2),
            msg='{}({}, {})'.format(name, arg1, arg2))
      # Tests type promotion
      for type_a in self.types:
        for type_b in self.types:
          if not check_promotion and type_a != type_b:
            continue
          arg1 = np_array_ops.array(a, dtype=type_a)
          arg2 = np_array_ops.array(b, dtype=type_b)
          self.match(
              math_fun(arg1, arg2),
              np_fun(arg1, arg2),
              msg='{}({}, {})'.format(name, arg1, arg2),
              check_dtype=check_promotion_result_type)

    if operands is None:
      operands = [(5, 2), (5, [2, 3]), (5, [[2, 3], [6, 7]]), ([1, 2, 3], 7),
                  ([1, 2, 3], [5, 6, 7])]
    for operand1, operand2 in operands:
      run_test(operand1, operand2)
    if extra_operands is not None:
      for operand1, operand2 in extra_operands:
        run_test(operand1, operand2)

  def testDot(self):
    # [FUNCTION DOC]
    # Checks dot against NumPy on the default operand pairs plus a vector times matrix and a
    # three-dimensional by three-dimensional case. Returns the helper result (nothing).
    extra_operands = [([1, 2], [[5, 6, 7], [8, 9, 10]]),
                      (np.arange(2 * 3 * 5).reshape([2, 3, 5]).tolist(),
                       np.arange(5 * 7 * 11).reshape([7, 5, 11]).tolist())]
    return self._testBinaryOp(
        np_math_ops.dot, np.dot, 'dot', extra_operands=extra_operands)

  def testMinimum(self):
    # [FUNCTION DOC]
    # Checks elementwise minimum against NumPy; result dtype is not compared under type promotion
    # because NumPy's promoted result type is unusual.
    # The numpy version has strange result type when promotion happens,
    # so set check_promotion_result_type to False.
    return self._testBinaryOp(
        np_math_ops.minimum,
        np.minimum,
        'minimum',
        check_promotion_result_type=False)

  def testMaximum(self):
    # [FUNCTION DOC]
    # Checks elementwise maximum against NumPy; result dtype is not compared under type promotion
    # because NumPy's promoted result type is unusual.
    # The numpy version has strange result type when promotion happens,
    # so set check_promotion_result_type to False.
    return self._testBinaryOp(
        np_math_ops.maximum,
        np.maximum,
        'maximum',
        check_promotion_result_type=False)

  def testMatmul(self):
    # [FUNCTION DOC]
    # Checks matrix multiplication of a 1x2 and a 2x3 integer matrix against NumPy across array
    # kinds and numeric types.
    operands = [([[1, 2]], [[3, 4, 5], [6, 7, 8]])]
    return self._testBinaryOp(
        np_math_ops.matmul, np.matmul, 'matmul', operands=operands)

  def testMatmulError(self):
    # [FUNCTION DOC]
    # Asserts matmul raises a ValueError when either operand is a scalar (zero-dimensional) and
    # the other is a 2x3 matrix.
    with self.assertRaisesRegex(ValueError, r''):
      np_math_ops.matmul(
          np_array_ops.ones([], np.int32), np_array_ops.ones([2, 3], np.int32))
    with self.assertRaisesRegex(ValueError, r''):
      np_math_ops.matmul(
          np_array_ops.ones([2, 3], np.int32), np_array_ops.ones([], np.int32))

  def testVDot(self):
    # [FUNCTION DOC]
    # Checks vdot (flattened dot product) against NumPy for two equal-shape matrices and for a
    # matrix versus a flat vector.
    operands = [([[1, 2], [3, 4]], [[3, 4], [6, 7]]),
                ([[1, 2], [3, 4]], [3, 4, 6, 7])]
    return self._testBinaryOp(
        np_math_ops.vdot, np.vdot, 'vdot', operands=operands)

  def testLcm(self):
    # [FUNCTION DOC]
    # Computes the least common multiple of two 8-bit integer scalars (6 and 22) with the
    # framework function and NumPy, and asserts they are equal.
    a = np_array_ops.array(6, dtype=np.int8)
    b = np_array_ops.array(22, dtype=np.int8)
    res_tf = np_math_ops.lcm(a, b)
    res_np = np.lcm(np.array(a), np.array(b))
    self.assertEqual(res_tf, res_np)

  def _testUnaryOp(self, math_fun, np_fun, name):
    # [FUNCTION DOC]
    # HELPER: shared driver for unary-operation tests. Inputs: math_fun (framework function),
    # np_fun (NumPy reference), name (for failure messages). Applies it to a scalar, a vector and
    # a matrix (with negative values), each through every array transform, and matches results
    # with NumPy. Returns nothing.

    def run_test(a):
      # [FUNCTION DOC]
      # HELPER: applies both functions to value a under every array transform and matches them.
      for fn in self.array_transforms:
        arg1 = fn(a)
        self.match(
            math_fun(arg1), np_fun(arg1), msg='{}({})'.format(name, arg1))

    run_test(5)
    run_test([2, 3])
    run_test([[2, -3], [-6, 7]])

  def testLog(self):
    # [FUNCTION DOC]
    # Checks natural log against NumPy using the unary-op helper.
    self._testUnaryOp(np_math_ops.log, np.log, 'log')

  def testExp(self):
    # [FUNCTION DOC]
    # Checks exponential against NumPy using the unary-op helper.
    self._testUnaryOp(np_math_ops.exp, np.exp, 'exp')

  def testTanh(self):
    # [FUNCTION DOC]
    # Checks hyperbolic tangent against NumPy using the unary-op helper.
    self._testUnaryOp(np_math_ops.tanh, np.tanh, 'tanh')

  def testSqrt(self):
    # [FUNCTION DOC]
    # Checks square root against NumPy using the unary-op helper (inputs include negatives, so
    # not-a-number results are compared too).
    self._testUnaryOp(np_math_ops.sqrt, np.sqrt, 'sqrt')

  def match(self, actual, expected, msg='', check_dtype=True):
    # [FUNCTION DOC]
    # HELPER: shared comparison used by all tests in this class.
    # Asserts actual is a framework ndarray, optionally (check_dtype, default true) that its dtype
    # equals expected's dtype, that shapes are equal, and that values are close (relative
    # tolerance one millionth) after converting both to lists. msg is appended to failure
    # messages. Raises assertion errors on mismatch; returns nothing.
    self.assertIsInstance(actual, np_arrays.ndarray)
    if check_dtype:
      self.assertEqual(
          actual.dtype, expected.dtype,
          'Dtype mismatch.\nActual: {}\nExpected: {}\n{}'.format(
              actual.dtype.as_numpy_dtype, expected.dtype, msg))
    self.assertEqual(
        actual.shape, expected.shape,
        'Shape mismatch.\nActual: {}\nExpected: {}\n{}'.format(
            actual.shape, expected.shape, msg))
    np.testing.assert_allclose(actual.tolist(), expected.tolist(), rtol=1e-6)

  def testArgsort(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the nested method defined at the end (complex dtype error test) is indented
    # inside this method, so it is only a local function that is never called; the complex dtype
    # check is dead code and never runs.
    # Checks argsort against NumPy via the unary-op helper, then checks stability: argsort of an
    # all-zero array with the stable kind returns the identity order.
    self._testUnaryOp(np_math_ops.argsort, np.argsort, 'argsort')

    # Test stability
    r = np.arange(100)
    a = np.zeros(100)
    np.testing.assert_equal(np_math_ops.argsort(a, kind='stable'), r)

    def testArgsortRaisesErrorForComplexDtypes(self):
      # [FUNCTION DOC]
      # KNOWN ISSUE: defined nested inside testArgsort and never invoked, so it is never run by
      # the test runner.
      # Intended test: argsort of complex64 and complex128 arrays must raise a TypeError whose
      # message says argsort does not support complex dtypes.
      """Test that argsort raises TypeError for complex64 and complex128."""
      complex64_array = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
      with self.assertRaisesRegex(
          TypeError, 'argsort does not support complex64/complex128 dtypes'
      ):
        np_math_ops.argsort(complex64_array)

      complex128_array = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
      with self.assertRaisesRegex(
          TypeError, 'argsort does not support complex64/complex128 dtypes'
      ):
        np_math_ops.argsort(complex128_array)

  def testArgMaxArgMin(self):
    # [FUNCTION DOC]
    # Checks argmax and argmin against NumPy for scalars up to three-dimensional nested lists,
    # under every array transform, both with no axis and with every valid axis (negative and
    # positive; scalars are treated as one-dimensional).
    data = [
        0,
        5,
        [1],
        [1, 2, 3],
        [[1, 2, 3]],
        [[4, 6], [7, 8]],
        [[[4, 6], [9, 10]], [[7, 8], [12, 34]]],
    ]
    for fn, d in itertools.product(self.array_transforms, data):
      arr = fn(d)
      self.match(np_math_ops.argmax(arr), np.argmax(arr))
      self.match(np_math_ops.argmin(arr), np.argmin(arr))
      if hasattr(arr, 'shape'):
        ndims = len(arr.shape)
      else:
        ndims = np_array_ops.array(arr, copy=False).ndim
      if ndims == 0:
        # Numpy flattens the scalar ndarray and treats it as a 1-d array of
        # size 1.
        ndims = 1
      for axis in range(-ndims, ndims):
        self.match(
            np_math_ops.argmax(arr, axis=axis), np.argmax(arr, axis=axis))
        self.match(
            np_math_ops.argmin(arr, axis=axis), np.argmin(arr, axis=axis))

  @parameterized.parameters([False, True])
  def testIsCloseEqualNan(self, equal_nan):
    # [FUNCTION DOC]
    # Parameterized over the equal-nan flag (false, true). Compares isclose of two float vectors
    # containing not-a-number values with NumPy's result for the same flag.
    a = np.asarray([1, 1, np.nan, 1, np.nan], np.float32)
    b = np.asarray([1, 2, 1, np.nan, np.nan], np.float32)
    self.match(
        np_math_ops.isclose(a, b, equal_nan=equal_nan),
        np.isclose(a, b, equal_nan=equal_nan))

  def testAverageWrongShape(self):
    # [FUNCTION DOC]
    # Asserts average raises an invalid-argument error when the weights shape is incompatible
    # with the data (mismatched full shape, mismatched shape with an axis, scalar weights with an
    # axis, wrong-length weights with an axis).
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), weights=np.ones([2, 4]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([2, 4]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([5]))

  def testClip(self):
    # [FUNCTION DOC]
    # Checks clip against NumPy for scalars, lists, floats, missing lower or upper bounds and
    # array-valued bounds. Dtype is not compared because NumPy's output type differs between
    # scalar and list inputs.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: clip comparison for one input.
      # KNOWN ISSUE: the loop reassigns arr to the transformed value each iteration, so
      # transforms are applied cumulatively to the previous result rather than the original input. Applies each array transform and matches clip with
      # NumPy using the extra arguments; the check_dtype keyword (default true) is removed from
      # the keyword arguments and controls dtype comparison.
      check_dtype = kwargs.pop('check_dtype', True)
      for fn in self.array_transforms:
        arr = fn(arr)
        self.match(
            np_math_ops.clip(arr, *args, **kwargs),
            np.clip(arr, *args, **kwargs),
            check_dtype=check_dtype)

    # NumPy exhibits weird typing behavior when a/a_min/a_max are scalars v/s
    # lists, e.g.,
    #
    # np.clip(np.array(0, dtype=np.int32), -5, 5).dtype == np.int64
    # np.clip(np.array([0], dtype=np.int32), -5, 5).dtype == np.int32
    # np.clip(np.array([0], dtype=np.int32), [-5], [5]).dtype == np.int64
    #
    # So we skip matching type. In tf-numpy the type of the output array is
    # always the same as the input array.
    run_test(0, -1, 5, check_dtype=False)
    run_test(-1, -1, 5, check_dtype=False)
    run_test(5, -1, 5, check_dtype=False)
    run_test(-10, -1, 5, check_dtype=False)
    run_test(10, -1, 5, check_dtype=False)
    run_test(10, None, 5, check_dtype=False)
    run_test(10, -1, None, check_dtype=False)
    run_test([0, 20, -5, 4], -1, 5, check_dtype=False)
    run_test([0, 20, -5, 4], None, 5, check_dtype=False)
    run_test([0, 20, -5, 4], -1, None, check_dtype=False)
    run_test([0.5, 20.2, -5.7, 4.4], -1.5, 5.1, check_dtype=False)

    run_test([0, 20, -5, 4], [-5, 0, -5, 0], [0, 5, 0, 5], check_dtype=False)
    run_test([[1, 2, 3], [4, 5, 6]], [2, 0, 2], 5, check_dtype=False)
    run_test([[1, 2, 3], [4, 5, 6]], 0, [5, 3, 1], check_dtype=False)

  def testPtp(self):
    # [FUNCTION DOC]
    # Checks ptp (peak-to-peak range) against NumPy for integer and float vectors and a matrix
    # along each positive and negative axis.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: for each array transform, applies it to arr and matches ptp with NumPy's, passing
      # the extra arguments (such as axis) to both.
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_math_ops.ptp(arg, *args, **kwargs), np.ptp(arg, *args, **kwargs))

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)

  def testLinSpace(self):
    # [FUNCTION DOC]
    # Checks linspace against NumPy for ascending and descending ranges, with default count, a
    # count of ten and without the endpoint; start and stop are each passed through every array
    # transform (all pairs). Uses a local copy of the transform list.
    array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64)
    ]

    def run_test(start, stop, **kwargs):
      # [FUNCTION DOC]
      # HELPER: for all pairs of transforms applied to start and stop, matches linspace with
      # NumPy's, forwarding the extra keyword arguments (count, endpoint).
      for fn1 in array_transforms:
        for fn2 in array_transforms:
          arg1 = fn1(start)
          arg2 = fn2(stop)
          self.match(
              np_math_ops.linspace(arg1, arg2, **kwargs),
              np.linspace(arg1, arg2, **kwargs),
              msg='linspace({}, {})'.format(arg1, arg2))

    run_test(0, 1)
    run_test(0, 1, num=10)
    run_test(0, 1, endpoint=False)
    run_test(0, -1)
    run_test(0, -1, num=10)
    run_test(0, -1, endpoint=False)

  def testLogSpace(self):
    # [FUNCTION DOC]
    # Checks logspace against NumPy for positive and negative exponent ranges, with default
    # count, count ten, no endpoint, and base two; start and stop go through all pairs of array
    # transforms (local copy of the transform list).
    array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64)
    ]

    def run_test(start, stop, **kwargs):
      # [FUNCTION DOC]
      # HELPER: for all pairs of transforms applied to start and stop, matches logspace with
      # NumPy's, forwarding the extra keyword arguments (count, endpoint, base).
      for fn1 in array_transforms:
        for fn2 in array_transforms:
          arg1 = fn1(start)
          arg2 = fn2(stop)
          self.match(
              np_math_ops.logspace(arg1, arg2, **kwargs),
              np.logspace(arg1, arg2, **kwargs),
              msg='logspace({}, {})'.format(arg1, arg2))

    run_test(0, 5)
    run_test(0, 5, num=10)
    run_test(0, 5, endpoint=False)
    run_test(0, 5, base=2.0)
    run_test(0, -5)
    run_test(0, -5, num=10)
    run_test(0, -5, endpoint=False)
    run_test(0, -5, base=2.0)

  def testGeomSpace(self):
    # [FUNCTION DOC]
    # Checks geomspace against NumPy for a positive range (1 to 1000) and a negative range, with
    # five points, with and without the endpoint.

    def run_test(start, stop, **kwargs):
      # [FUNCTION DOC]
      # HELPER: matches geomspace with NumPy's for the given start and stop (no array transforms),
      # forwarding the extra keyword arguments.
      arg1 = start
      arg2 = stop
      self.match(
          np_math_ops.geomspace(arg1, arg2, **kwargs),
          np.geomspace(arg1, arg2, **kwargs),
          msg='geomspace({}, {})'.format(arg1, arg2))

    run_test(1, 1000, num=5)
    run_test(1, 1000, num=5, endpoint=False)
    run_test(-1, -1000, num=5)
    run_test(-1, -1000, num=5, endpoint=False)

  @parameterized.parameters([
      'T', 'ndim', 'size', 'data', '__pos__', '__round__', 'tolist', 'flatten',
      'transpose', 'reshape', 'ravel', 'clip', 'astype', 'max', 'mean', 'min'])
  def testNumpyMethodsOnTensor(self, np_method):
    # [FUNCTION DOC]
    # Parameterized over names of numpy-style attributes and methods (transpose property, ndim,
    # size, data, flatten, reshape, clip, astype, max, mean, min, etc.). Asserts a tensor made
    # from a small list has an attribute with each name.
    a = ops.convert_to_tensor([1, 2])
    self.assertTrue(hasattr(a, np_method))

  def testFlatten(self):
    # [FUNCTION DOC]
    # Asserts tensor flatten matches NumPy flatten for row-major and column-major orders, and that
    # the "any" and "keep" orders both give the row-major result; an invalid order raises a
    # ValueError.
    a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    a2 = ops.convert_to_tensor(a1)
    self.assertAllEqual(a1.flatten('C'), a2.flatten('C'))
    self.assertAllEqual(a1.flatten('F'), a2.flatten('F'))
    self.assertAllEqual(a1.flatten('C'), a2.flatten('A'))
    self.assertAllEqual(a1.flatten('C'), a2.flatten('K'))
    with self.assertRaises(ValueError):
      a2.flatten('invalid')

  def testIsInf(self):
    # [FUNCTION DOC]
    # Asserts that the extreme 32-bit integer values (minimum and maximum) are reported as not
    # infinite, not positive infinity and not negative infinity by isinf, isposinf and isneginf.
    x1 = ops.convert_to_tensor(-2147483648)
    x2 = ops.convert_to_tensor(2147483647)
    self.assertFalse(np_math_ops.isinf(x1))
    self.assertFalse(np_math_ops.isinf(x2))
    self.assertFalse(np_math_ops.isposinf(x1))
    self.assertFalse(np_math_ops.isposinf(x2))
    self.assertFalse(np_math_ops.isneginf(x1))
    self.assertFalse(np_math_ops.isneginf(x2))

if __name__ == '__main__':
  tensor.enable_tensor_equality()
  ops.enable_eager_execution()
  ops.set_dtype_conversion_mode('legacy')
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
