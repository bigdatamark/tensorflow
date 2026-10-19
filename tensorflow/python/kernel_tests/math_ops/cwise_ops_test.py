# [MODULE DOC]
# Test module for coefficient-wise (elementwise) math ops other than the unary ones: comparison,
# logical, select (where and where_v2, including batch select), minimum and maximum, operator
# overloads on tensors, finiteness checks, rounding, complex-number ops and polynomial
# evaluation. Compares TensorFlow results with NumPy on CPU and GPU, and checks gradients where
# relevant. Defines module-level operator lambdas used by the overload tests. KNOWN ISSUE: it
# also contains two unused helper functions copied from the unary test module. No import-time
# side effects; running as a script starts the TensorFlow test runner.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for coefficient-wise operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
_POW = lambda x, y: x**y
_TRUEDIV = lambda x, y: x / y
_FLOORDIV = lambda x, y: x // y
_MOD = lambda x, y: x % y

_LT = lambda x, y: x < y
_LE = lambda x, y: x <= y
_GT = lambda x, y: x > y
_GE = lambda x, y: x >= y

_AND = lambda x, y: x & y
_OR = lambda x, y: x | y
_XOR = lambda x, y: x ^ y
_INV = lambda x: ~x


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Dead code. Nothing in this file calls it (it was copied from the unary test).
  # HELPER: Builds a sparse tensor from a dense array. Inputs: array x, threshold (default 0.5)
  # below which entries are zeroed, and the integer dtype of the indices (default 64-bit).
  # Side effect: modifies x in place. Returns a pair: the sparse tensor of nonzero entries and
  # the one-dimensional array of those values.
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), x_values


def _default_tolerance(dtype):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Dead code. Nothing in this file calls it (it was copied from the unary test),
  # and unlike that copy it does not handle bfloat16.
  # HELPER: Picks a comparison tolerance by NumPy dtype: 5e-3 for float16, 1e-3 for float32 and
  # complex64, 1e-5 for float64 and complex128, and nothing (null value) for any other dtype.
  """Returns a sensible default tolerance for comparing results of a given type.

  Args:
    dtype: A datatype.
  """
  if dtype == np.float16:
    return 5e-3
  elif dtype in (np.float32, np.complex64):
    return 1e-3
  elif dtype in (np.float64, np.complex128):
    return 1e-5
  else:
    return None  # Fail fast for unexpected types


class ComparisonOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for comparison ops (less, less-or-equal, greater, greater-or-equal, equal,
  # not-equal): scalars, tensors, broadcasting across many shape pairs, complex equality, and
  # incompatible-shape errors. Shared helpers compare TensorFlow results with NumPy.

  def _compareScalar(self, func, x, y, dtype):
    # [FUNCTION DOC]
    # HELPER: Applies comparison op func to single-element tensors holding x and y cast to the
    # NumPy dtype (GPU if available) and returns the single boolean result.
    with test_util.use_gpu():
      out = func(
          ops.convert_to_tensor(np.array([x]).astype(dtype)),
          ops.convert_to_tensor(np.array([y]).astype(dtype)))
      ret = self.evaluate(out)
    return ret[0]

  def testScalarCompareScalar(self):
    # [FUNCTION DOC]
    # Test: all six comparison ops on single-element tensors of values minus one, zero and one
    # for float16/32/64 and int32/64, and equality and inequality only for complex64/128 using
    # complex values. Each result is checked against Python's own comparison as sub-tests.
    dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
    data = [-1, 0, 1]
    for t in dtypes:
      for x in data:
        for y in data:
          with self.subTest(t=t, x=x, y=y):
            self.assertEqual(self._compareScalar(math_ops.less, x, y, t), x < y)
            self.assertEqual(
                self._compareScalar(math_ops.less_equal, x, y, t), x <= y)
            self.assertEqual(
                self._compareScalar(math_ops.greater, x, y, t), x > y)
            self.assertEqual(
                self._compareScalar(math_ops.greater_equal, x, y, t), x >= y)
            self.assertEqual(
                self._compareScalar(math_ops.equal, x, y, t), x == y)
            self.assertEqual(
                self._compareScalar(math_ops.not_equal, x, y, t), x != y)
    data = [-1, 0, 1, -1j, 1j, 1 + 1j, 1 - 1j]
    for t in [np.complex64, np.complex128]:
      for x in data:
        for y in data:
          with self.subTest(t=t, x=x, y=y):
            self.assertEqual(
                self._compareScalar(math_ops.equal, x, y, t), x == y)
            self.assertEqual(
                self._compareScalar(math_ops.not_equal, x, y, t), x != y)

  def _compare(self, x, y, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: Applies a TensorFlow comparison op to x and y (GPU if available) and asserts the
    # result equals the NumPy function applied to the same inputs.
    np_ans = np_func(x, y)
    with test_util.use_gpu():
      out = tf_func(ops.convert_to_tensor(x), ops.convert_to_tensor(y))
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)

  def testTensorCompareTensor(self):
    # [FUNCTION DOC]
    # Test: all six comparison ops on array pairs (shape one by three by two) for real dtypes,
    # and equality and inequality only for complex dtypes (which have no ordering), on GPU if
    # available, compared with NumPy.
    x = np.linspace(-15, 15, 6).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    y = np.linspace(20, -10, 6).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        self._compare(xt, yt, np.less, math_ops.less)
        self._compare(xt, yt, np.less_equal, math_ops.less_equal)
        self._compare(xt, yt, np.greater, math_ops.greater)
        self._compare(xt, yt, np.greater_equal, math_ops.greater_equal)
        self._compare(xt, yt, np.equal, math_ops.equal)
        self._compare(xt, yt, np.not_equal, math_ops.not_equal)
    # Complex types do not support ordering but do support equality tests.
    for t in [np.complex64, np.complex128]:
      with self.subTest(t=t):
        xt = x.astype(t)
        xt -= 1j * xt
        yt = y.astype(t)
        yt -= 1j * yt
        self._compare(xt, yt, np.equal, math_ops.equal)
        self._compare(xt, yt, np.not_equal, math_ops.not_equal)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: Builds evenly spaced arrays with shapes xs and ys of the given dtype (complex
    # values get a negated imaginary part copy) and compares the TensorFlow op with the NumPy
    # function in both operand orders.
    x = np.linspace(-15, 15, np.prod(xs)).astype(dtype).reshape(xs)
    y = np.linspace(20, -10, np.prod(ys)).astype(dtype).reshape(ys)
    if dtype in (np.complex64, np.complex128):
      x -= 1j * x
      y -= 1j * y
    self._compare(x, y, np_func, tf_func)
    self._compare(y, x, np_func, tf_func)

  def _testBCastByFunc(self, np_func, tf_func, include_complex=False):
    # [FUNCTION DOC]
    # HELPER: Runs a comparison op over eleven broadcast shape pairs (including zero-size
    # dimensions) for float16/32/64 and int32/64, plus complex64/128 when include_complex is true
    # (default false), as sub-tests. Inputs: NumPy reference function and TensorFlow op.
    shapes = [
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
    ]
    dtypes = [
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
    ]
    if include_complex:
      dtypes.extend([np.complex64, np.complex128])

    for (xs, ys) in shapes:
      for dtype in dtypes:
        with self.subTest(xs=xs, ys=ys, dtype=dtype):
          self._compareBCast(xs, ys, dtype, np_func, tf_func)

  def testBCastLess(self):
    # [FUNCTION DOC]
    # Test: less-than with broadcasting across shape pairs (real dtypes only).
    self._testBCastByFunc(np.less, math_ops.less)

  def testBCastLessEqual(self):
    # [FUNCTION DOC]
    # Test: less-or-equal with broadcasting across shape pairs (real dtypes only).
    self._testBCastByFunc(np.less_equal, math_ops.less_equal)

  def testBCastGreater(self):
    # [FUNCTION DOC]
    # Test: greater-than with broadcasting across shape pairs (real dtypes only).
    self._testBCastByFunc(np.greater, math_ops.greater)

  def testBCastGreaterEqual(self):
    # [FUNCTION DOC]
    # Test: greater-or-equal with broadcasting across shape pairs (real dtypes only).
    self._testBCastByFunc(np.greater_equal, math_ops.greater_equal)

  def testBCastEqual(self):
    # [FUNCTION DOC]
    # Test: equal with broadcasting across shape pairs, including complex dtypes.
    self._testBCastByFunc(np.equal, math_ops.equal, include_complex=True)

  def testBCastNotEqual(self):
    # [FUNCTION DOC]
    # Test: not-equal with broadcasting across shape pairs, including complex dtypes.
    self._testBCastByFunc(
        np.not_equal, math_ops.not_equal, include_complex=True)

  def testShapeMismatch(self):
    # [FUNCTION DOC]
    # Test: comparison ops (less, less-or-equal, greater, greater-or-equal, equal, not-equal) on
    # two-by-five versus five-by-two arrays for several dtypes. Expects an incompatible-shapes
    # error (value error or invalid-argument error) for each op and dtype.
    dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
    funcs = [
        math_ops.less, math_ops.less_equal, math_ops.greater,
        math_ops.greater_equal, math_ops.equal, math_ops.not_equal
    ]
    x = np.arange(0, 10).reshape([2, 5])
    y = np.arange(0, 10).reshape([5, 2])
    for t in dtypes:
      for f in funcs:
        with self.subTest(t=t, f=f):
          with self.assertRaisesIncompatibleShapesError(
              (ValueError, errors.InvalidArgumentError)):
            f(x.astype(t), y.astype(t))


class LogicalOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for logical ops (and, or, xor, not) on boolean tensors: scalars, tensors,
  # broadcasting, shape mismatches, and the rule that tensors cannot be used as Python booleans.

  def _compareBinary(self, x, y, np_func, tf_func, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: Applies a binary logical TensorFlow op to x and y on the chosen device (default CPU)
    # and asserts the result dtype is boolean and that values and shape match the NumPy
    # reference function.
    np_ans = np_func(x, y)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_val = self.evaluate(out)
    self.assertEqual(out.dtype, dtypes_lib.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def _not(self, x, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: Computes logical not of boolean array x on the chosen device (default CPU) and
    # asserts result dtype is boolean and that values and shape match NumPy.
    np_ans = np.logical_not(x)
    with test_util.device(use_gpu=use_gpu):
      out = math_ops.logical_not(ops.convert_to_tensor(x))
      tf_val = self.evaluate(out)
    self.assertEqual(out.dtype, dtypes_lib.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def testScalar(self):
    # [FUNCTION DOC]
    # Test: logical not on each single-element boolean array, and logical and, or and xor over
    # every pair, on GPU and CPU, compared against NumPy.
    data = [np.array([True]), np.array([False])]
    for use_gpu in [True, False]:
      for x in data:
        with self.subTest(use_gpu=use_gpu, x=x):
          self._not(x, use_gpu)
      for x in data:
        for y in data:
          with self.subTest(use_gpu=use_gpu, x=x, y=y):
            self._compareBinary(x, y, np.logical_and, math_ops.logical_and,
                                use_gpu)
            self._compareBinary(x, y, np.logical_or, math_ops.logical_or,
                                use_gpu)
            self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor,
                                use_gpu)

  def testTensor(self):
    # [FUNCTION DOC]
    # Test: logical not, and, or and xor on random boolean arrays of shape one by three by two,
    # on CPU and GPU, compared against NumPy.
    x = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    y = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    for use_gpu in [True, False]:
      with self.subTest(use_gpu=use_gpu):
        self._not(x, use_gpu)
        self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
        self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
        self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)

  def testBCast(self):
    # [FUNCTION DOC]
    # Test: logical and, or and xor with broadcasting over eleven shape pairs (including shapes
    # with zero-size dimensions) using random boolean data, on CPU and GPU, as sub-tests.
    shapes = [
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
    ]
    for (xs, ys) in shapes:
      x = np.random.randint(0, 2, np.prod(xs)).astype(np.bool_).reshape(xs)
      y = np.random.randint(0, 2, np.prod(ys)).astype(np.bool_).reshape(ys)
      for use_gpu in [True, False]:
        with self.subTest(xs=xs, ys=ys, use_gpu=use_gpu):
          self._compareBinary(x, y, np.logical_and, math_ops.logical_and,
                              use_gpu)
          self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
          self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor,
                              use_gpu)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    # [FUNCTION DOC]
    # Test: logical and, or and xor on boolean arrays of incompatible shapes (one by three by two
    # and three by two by one). Expects a value error whose message mentions that dimensions
    # must match. Graph mode only.
    x = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    y = np.random.randint(0, 2, 6).astype(np.bool_).reshape(3, 2, 1)  # pylint: disable=too-many-function-args
    for f in [math_ops.logical_and, math_ops.logical_or, math_ops.logical_xor]:
      with self.subTest(f=f):
        with self.assertRaisesWithPredicateMatch(
            ValueError, lambda e: "Dimensions must" in str(e)):
          f(x, y)

  @test_util.run_deprecated_v1
  def testUsingAsPythonValueFails(self):
    # [FUNCTION DOC]
    # Test: a graph tensor cannot be used as a Python boolean. Expects a type error when a
    # boolean tensor is used in a condition, when a tensor comparison result is, and for a
    # chained comparison (since the first comparison result is evaluated as a boolean). Graph
    # mode only.
    # Ensure that we raise an error when the user attempts to treat a
    # `Tensor` as a Python `bool`.
    b = constant_op.constant(False)
    with self.assertRaises(TypeError):
      if b:
        pass

    x = constant_op.constant(3)
    y = constant_op.constant(4)
    with self.assertRaises(TypeError):
      if x > y:
        pass

    z = constant_op.constant(7)

    # The chained comparison should fail because Python computes `x <
    # y` and short-circuits the comparison with `z` if it is `False`.
    with self.assertRaises(TypeError):
      _ = x < y < z


class SelectOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the select ops (original where and the broadcasting v2 variant): scalar and
  # tensor conditions, broadcasting, gradients, shape mismatches, empty tensors and
  # not-a-number handling. Most tests take the op under test as a parameter of a private helper
  # and are run once per op.

  def _compare(self, fn, c, x, y, use_gpu):
    # [FUNCTION DOC]
    # HELPER: Applies select function fn to condition c and values x and y on the chosen device
    # and asserts equality and equal shape with NumPy's where.
    np_ans = np.where(c, x, y)
    with test_util.device(use_gpu=use_gpu):
      out = fn(c, x, y)
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self,
                        fn,
                        c,
                        x,
                        y,
                        numeric_gradient_type=None,
                        x_init_value=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: When x and y have different shapes, the initial value for x is overwritten with
    # y broadcast to x's shape (not x itself), even if the caller supplied an initial value;
    # this looks like it should use x. Also the Jacobian is sized from the condition shape.
    # HELPER: Gradient check of select function fn with respect to first value input x.
    # Inputs: condition c, values x and y, optional dtype to recompute the numeric gradient at
    # higher precision (default none), and optional initial value for x (default x). Compares
    # analytic and numeric Jacobians with tolerance 1e-3 for float16/float32 and 1e-5 for float64.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = fn(c, inx, iny)
      s = list(np.shape(c))
      if x_init_value is None:
        x_init_value = x
      if x.shape != y.shape:
        x_init_value = np.broadcast_to(y, x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x_init_value)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = fn(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, outf, s, x_init_value=xf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, fn, c, x, y, numeric_gradient_type=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The gradient is taken with respect to y, but the initial value supplied to the
    # checker is x (not y), and the step is the unusually large 1.0; this makes the check look
    # inconsistent with the sibling x-gradient helper, although it passes because select just
    # routes values.
    # HELPER: Gradient check of select function fn with respect to second value input y.
    # Inputs: condition c, values x and y, optional dtype to recompute the numeric gradient at
    # higher precision (default none). Compares analytic and numeric Jacobians with tolerance
    # 1e-3 for float16/float32 and 1e-5 for float64.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = fn(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=x, delta=1.0)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = fn(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, s, outf, s, x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testScalar(self, fn):
    # [FUNCTION DOC]
    # HELPER: Checks select function fn with a scalar true condition and random same-shaped
    # values across float, integer and complex dtypes: CPU for all, GPU for floating dtypes.
    c = True
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        self._compare(fn, c, xt, yt, use_gpu=False)
        if t in [np.float16, np.float32, np.float64]:
          self._compare(fn, c, xt, yt, use_gpu=True)

  def testScalar(self):
    # [FUNCTION DOC]
    # Test: runs the scalar-condition helper for both the original and v2 select ops.
    self._testScalar(array_ops.where)
    self._testScalar(array_ops.where_v2)

  def _testScalarBroadcast(self, fn, c, x, y):
    # [FUNCTION DOC]
    # HELPER: For select function fn, condition c and values x and y, compares fn against NumPy
    # for each dtype in float16/32/64, int32/64 and complex64/128 on CPU, and on GPU for the
    # floating dtypes only, as sub-tests.
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        self._compare(fn, c, xt, yt, use_gpu=False)
        if t in [np.float16, np.float32, np.float64]:
          self._compare(fn, c, xt, yt, use_gpu=True)

  def testScalarBroadcast(self):
    # [FUNCTION DOC]
    # Test: v2 select with a scalar true condition and a value broadcast against another across
    # seven shape pairs, in both operand orders, via the broadcast helper. Only the v2 op
    # supports broadcasting.
    c = True
    # where_v2 only
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(3, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)

  def _testBasic(self, fn):
    # [FUNCTION DOC]
    # HELPER: Checks select function fn with a random boolean condition and random same-shaped
    # float values cast to float16/32/64, int32/64 and complex64/128, on CPU for all dtypes and
    # on GPU for floating dtypes, against NumPy's where.
    c = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        self._compare(fn, c, xt, yt, use_gpu=False)
        if t in [np.float16, np.float32, np.float64]:
          self._compare(fn, c, xt, yt, use_gpu=True)

  def testBasic(self):
    # [FUNCTION DOC]
    # Test: runs the basic select helper for both the original and v2 select ops.
    self._testBasic(array_ops.where)
    self._testBasic(array_ops.where_v2)

  def _testBasicBroadcast(self, fn, c, x, y):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Identical in body to the scalar-broadcast helper (same loop over dtypes and
    # same comparisons); the two differ only in how callers build the condition.
    # HELPER: For select function fn, condition c and values x and y, compares fn against NumPy
    # for each dtype in float16/32/64, int32/64 and complex64/128 on CPU, and on GPU for the
    # floating dtypes only, as sub-tests.
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        self._compare(fn, c, xt, yt, use_gpu=False)
        if t in [np.float16, np.float32, np.float64]:
          self._compare(fn, c, xt, yt, use_gpu=True)

  def testBasicBroadcast(self):
    # [FUNCTION DOC]
    # Test: v2 select with broadcasting of the condition (four condition shapes) and of the
    # values (seven shape pairs, both operand orders), via the broadcast helper. Only the v2 op
    # supports broadcasting.
    c0 = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    c1 = np.random.randint(0, 2, 2).astype(np.bool_).reshape(1, 1, 2)  # pylint: disable=too-many-function-args
    c2 = np.random.randint(0, 2, 3).astype(np.bool_).reshape(1, 3, 1)  # pylint: disable=too-many-function-args
    c3 = np.random.randint(0, 2, 1).astype(np.bool_).reshape(1, 1, 1)  # pylint: disable=too-many-function-args
    for c in [c0, c1, c2, c3]:
      # where_v2 only
      with self.subTest(c=c):
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1, 1) * 100
        self._testBasicBroadcast(array_ops.where_v2, c, x, y)
        self._testBasicBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 3, 1) * 100
        self._testBasicBroadcast(array_ops.where_v2, c, x, y)
        self._testBasicBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1, 2) * 100
        self._testBasicBroadcast(array_ops.where_v2, c, x, y)
        self._testBasicBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1) * 100
        self._testBasicBroadcast(array_ops.where_v2, c, x, y)
        self._testBasicBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1) * 100
        self._testBasicBroadcast(array_ops.where_v2, c, x, y)
        self._testBasicBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 2) * 100
        self._testBasicBroadcast(array_ops.where_v2, c, x, y)
        self._testBasicBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(3, 2) * 100
        self._testBasicBroadcast(array_ops.where_v2, c, x, y)
        self._testBasicBroadcast(array_ops.where_v2, c, y, x)

  def _testGradients(self, fn):
    # [FUNCTION DOC]
    # HELPER: Checks gradients of select function fn on random same-shaped x and y with a random
    # boolean condition for float16, float32 and float64, for both value inputs. For float16
    # the numerical gradient is computed in float64 because float16 numerical gradients are too
    # imprecise (a weaker check of the gradient only).
    c = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float16, np.float32, np.float64]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        if t == np.float16:
          # Compare fp16 theoretical gradients to fp32 numerical gradients,
          # since fp16 numerical gradients are too imprecise unless great
          # care is taken with choosing the inputs and the delta. This is
          # a weaker check (in particular, it does not test the op itself,
          # only its gradient), but it's much better than nothing.
          self._compareGradientX(fn, c, xt, yt, np.float64)
          self._compareGradientY(fn, c, xt, yt, np.float64)
        else:
          self._compareGradientX(fn, c, xt, yt)
          self._compareGradientY(fn, c, xt, yt)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Test: runs the gradient helper for both the original and v2 select ops. Graph mode only.
    self._testGradients(array_ops.where)
    self._testGradients(array_ops.where_v2)

  @test_util.run_deprecated_v1
  def testGradientsBroadcast(self):
    # [FUNCTION DOC]
    # Test: v2 select gradient with respect to the first value input when the second value
    # broadcasts (seven differently shaped second operands) for float32 and float64. Only the
    # x-gradient is checked, not the y-gradient. Graph mode only.
    c = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    for t in [np.float32, np.float64]:
      # where_v2 only
      with self.subTest(t=t):
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1, 1) * 100
        self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 3, 1) * 100
        self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1, 2) * 100
        self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1) * 100
        self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1) * 100
        self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 2) * 100
        self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(3, 2) * 100
        self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))

  def _testShapeMismatch(self, fn):
    # [FUNCTION DOC]
    # HELPER: For select function fn, supplies a condition of shape one by three by two with
    # values of incompatible shape (one by three by two and two by five by three) across float,
    # integer and complex dtypes, and expects a value error for each.
    c = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)  # pylint: disable=too-many-function-args
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(2, 5, 3) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        with self.assertRaises(ValueError):
          fn(c, xt, yt)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    # [FUNCTION DOC]
    # Test: runs the shape-mismatch helper for both the original and v2 select ops. Graph mode
    # only.
    self._testShapeMismatch(array_ops.where)
    self._testShapeMismatch(array_ops.where_v2)

  def _testEmptyTensor(self, fn):
    # [FUNCTION DOC]
    # HELPER: Applies select function fn to an empty boolean condition and empty float32 values
    # of shape one by three by zero in a cached session, and asserts the evaluated result equals
    # an empty float32 array of that shape.
    c = np.random.randint(0, 3, 0).astype(np.bool_).reshape(1, 3, 0)  # pylint: disable=too-many-function-args
    x = np.random.rand(1, 3, 0) * 100
    y = np.random.rand(1, 3, 0) * 100
    z_expected = np.zeros((1, 3, 0), dtype=np.float32)
    with self.cached_session():
      xt = x.astype(np.float32)
      yt = y.astype(np.float32)
      z = fn(c, xt, yt).eval()
      self.assertAllEqual(z_expected, z)

  @test_util.run_deprecated_v1
  def testEmptyTensor(self):
    # [FUNCTION DOC]
    # Test: select with zero-size inputs for both the original and v2 select ops. Graph mode only.
    self._testEmptyTensor(array_ops.where)
    self._testEmptyTensor(array_ops.where_v2)

  def _testNan(self, fn):
    # [FUNCTION DOC]
    # HELPER: For select function fn, tries a true or false scalar condition with both branch
    # values chosen from a finite number or not-a-number, and asserts the result is
    # not-a-number exactly when the selected branch value is.
    with self.cached_session():
      for c in False, True:
        for a in 7.0, np.nan:
          for b in 5.0, np.nan:
            with self.subTest(c=c, a=a, b=b):
              x = fn(c, a, b).eval()
              y = a if c else b
              self.assertEqual(np.isnan(x), np.isnan(y))

  @test_util.run_deprecated_v1
  def testNan(self):
    # [FUNCTION DOC]
    # Test: not-a-number values do not leak through select. Runs the nan helper for both the
    # original and v2 select ops. Graph mode only.
    """Verify that nans don't propagate where they shouldn't."""
    self._testNan(array_ops.where)
    self._testNan(array_ops.where_v2)


class BatchSelectOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the select op when the condition is a vector and the values are tensors of rank
  # two or more (rows are selected whole): values, gradients and shape-mismatch errors.
  """Test broadcasting of Select when 'c' is a vec and 't' &'e' are rank2+."""

  def _compare(self, c, x, y, use_gpu):
    # [FUNCTION DOC]
    # HELPER: Runs the select op with a vector condition c over higher-rank x and y on the chosen
    # device, and asserts equality and equal shape with a NumPy reference that picks whole rows
    # from x or y by the condition.
    np_ans = np.dstack(
        [x_i if c_i else y_i for c_i, x_i, y_i in zip(c, x, y)]).transpose(
            [2, 0, 1])
    with test_util.device(use_gpu=use_gpu):
      out = array_ops.where(c, x, y)
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, c, x, y, numeric_gradient_type=None):
    # [FUNCTION DOC]
    # HELPER: Gradient check of batch select with respect to the first value input x. Inputs:
    # condition c, values x and y, and optional dtype for recomputing the numeric gradient at
    # higher precision (default none). Compares analytic and numeric Jacobians with tolerance
    # 1e-3 for float16 and float32 and 1e-5 for float64.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = array_ops.where(c, inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = array_ops.where(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, outf, s, x_init_value=xf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, c, x, y, numeric_gradient_type=None):
    # [FUNCTION DOC]
    # HELPER: Gradient check of batch select with respect to the second value input y. Inputs:
    # condition c, values x and y, and optional dtype for recomputing the numeric gradient at
    # higher precision (default none). Compares analytic and numeric Jacobians with tolerance
    # 1e-3 for float16 and float32 and 1e-5 for float64.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = array_ops.where(c, inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=y)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = array_ops.where(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, s, outf, s, x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def testBasic(self):
    # [FUNCTION DOC]
    # Test: batch select where a length-16 boolean vector selects whole rows from random
    # 16-by-2-by-8 value tensors, for float, integer and complex dtypes; CPU for all types and
    # GPU for floating types only.
    c = np.random.randint(0, 2, 16).astype(np.bool_)
    x = np.random.rand(16, 2, 8) * 100
    y = np.random.rand(16, 2, 8) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        self._compare(c, xt, yt, use_gpu=False)
        if t in [np.float16, np.float32, np.float64]:
          self._compare(c, xt, yt, use_gpu=True)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Test: gradients of batch select (vector condition, higher-rank values) for float16, float32
    # and float64 with respect to both value inputs. For float16, theoretical gradients are
    # compared with float64 numerical gradients because float16 numerical ones are too imprecise
    # (a weaker check of the gradient only). Graph mode only.
    c = np.random.randint(0, 2, 16).astype(np.bool_)
    x = np.random.rand(16, 2, 8) * 100
    y = np.random.rand(16, 2, 8) * 100
    for t in [np.float16, np.float32, np.float64]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        if t == np.float16:
          # Compare fp16 theoretical gradients to fp32 numerical gradients,
          # since fp16 numerical gradients are too imprecise unless great
          # care is taken with choosing the inputs and the delta. This is
          # a weaker check (in particular, it does not test the op itself,
          # only its gradient), but it's much better than nothing.
          self._compareGradientX(c, xt, yt, np.float64)
          self._compareGradientY(c, xt, yt, np.float64)
        else:
          self._compareGradientX(c, xt, yt)
          self._compareGradientY(c, xt, yt)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    # [FUNCTION DOC]
    # Test: batch select with a condition vector of length 8 against value tensors with 16 rows.
    # Expects a value error for every dtype. Graph mode only.
    c = np.random.randint(0, 2, 8).astype(np.bool_)
    x = np.random.rand(16, 3, 2) * 100
    y = np.random.rand(16, 3, 2) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      with self.subTest(t=t):
        xt = x.astype(t)
        yt = y.astype(t)
        with self.assertRaises(ValueError):
          array_ops.where(c, xt, yt)


@test_util.with_eager_op_as_function
class MinMaxOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for elementwise minimum and maximum ops (values across dtypes, broadcasting,
  # not-a-number propagation, gradients). Decorated so ops run as functions in eager mode.

  def _compare(self, x, y, use_gpu):
    # [FUNCTION DOC]
    # HELPER: Evaluates elementwise minimum and maximum of x and y on the chosen device in one
    # call and asserts both equal NumPy's minimum and maximum exactly.
    np_min, np_max = np.minimum(x, y), np.maximum(x, y)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      omin, omax = math_ops.minimum(inx, iny), math_ops.maximum(inx, iny)
      tf_min, tf_max = self.evaluate([omin, omax])
    self.assertAllEqual(np_min, tf_min)
    self.assertAllEqual(np_max, tf_max)

  def testBasic(self):
    # [FUNCTION DOC]
    # Test: minimum and maximum on random same-shaped arrays across float16/32/64 and signed
    # and unsigned 8, 16, 32 and 64-bit integer dtypes, on CPU and GPU, against NumPy.
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(1, 3, 2) * 100.
    for t in [np.float16, np.float32, np.float64, np.int8, np.uint8, np.int16,
              np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
      with self.subTest(t=t):
        self._compare(x.astype(t), y.astype(t), use_gpu=False)
        self._compare(x.astype(t), y.astype(t), use_gpu=True)

  # When eager_op_as_function mode is enabled xla auto-clustering kicks in.
  # By default xla enables fast min_max computations which do not propagate NaN.
  # TODO(b/205140614): remove decorators once TF and XLA behaviour are the same.
  @test_util.set_xla_env_flag(flag="--xla_cpu_enable_fast_min_max=false")
  @test_util.set_xla_env_flag(flag="--xla_gpu_enable_fast_min_max=false")
  def testNaNPropagation(self):
    # [FUNCTION DOC]
    # Test: minimum and maximum propagate not-a-number when either or both inputs are. Uses all
    # four combinations of finite and not-a-number entries for float16, float32 and float64 on
    # CPU and GPU. The decorators disable the XLA fast min/max mode (which drops NaN) for CPU and
    # GPU through an environment flag while the test runs.
    x = np.array([1., np.nan, 1., np.nan], dtype=np.float64)
    y = np.array([1., 1., np.nan, np.nan], dtype=np.float64)
    for t in [np.float16, np.float32, np.float64]:
      with self.subTest(t=t):
        self._compare(x.astype(t), y.astype(t), use_gpu=False)
        self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testDifferentShapes(self):
    # [FUNCTION DOC]
    # Test: minimum and maximum where the second operand (length two) broadcasts against a
    # one-by-three-by-two first operand, for float and 32/64-bit integer dtypes, on CPU and GPU.
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(2) * 100.  # should broadcast
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      with self.subTest(t=t):
        self._compare(x.astype(t), y.astype(t), use_gpu=False)
        self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testScalar(self):
    # [FUNCTION DOC]
    # Test: minimum and maximum of a random array and a Python-scalar-valued operand
    # (broadcast), for float32 and int32 only (64-bit types skipped because TensorFlow converts
    # Python scalars to 32 bit), on CPU and GPU.
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(1).item() * 100.  # should broadcast
    # dropped np.float64, int64 because TF automatically converts to 32 bit
    for t in [np.float32, np.int32]:
      with self.subTest(t=t):
        self._compare(x.astype(t), t(y), use_gpu=False)
        self._compare(x.astype(t), t(y), use_gpu=True)

  def _compareGradientX(self, func, x, y):
    # [FUNCTION DOC]
    # HELPER: Gradient check of binary function func with respect to its first input x.
    # Compares analytic and numeric Jacobians, with tolerance 1e-3 for float16 and float32 and
    # 1e-5 for float64 (other dtypes not checked).
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, func, x, y):
    # [FUNCTION DOC]
    # HELPER: Gradient check of binary function func with respect to its second input y.
    # Compares analytic and numeric Jacobians, with tolerance 1e-3 for float16 and float32 and
    # 1e-5 for float64 (other dtypes not checked).
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=y)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Test: gradients of the minimum and maximum ops with respect to each input. Picks random
    # x and y = x plus or minus one (so inputs never tie) and checks both gradients for both
    # ops. Graph mode only.
    x = np.random.rand(1, 3, 2) * 100.
    # ensure x != y
    y = x + (np.random.randint(2, size=x.shape) - .5) * 2  # -1 or +1
    self._compareGradientX(math_ops.maximum, x, y)
    self._compareGradientY(math_ops.maximum, x, y)
    self._compareGradientX(math_ops.minimum, x, y)
    self._compareGradientY(math_ops.minimum, x, y)


class MathOpsOverloadTest(test.TestCase):
  # [CLASS DOC]
  # Test case for Python operator overloads on tensors (arithmetic, comparison, logical and
  # invert), exercised with a tensor on one side and a Python literal on the other, evaluated
  # on CPU and compared against NumPy.

  def _computeTensorAndLiteral(self, x, y, dtype, func):
    # [FUNCTION DOC]
    # HELPER: Converts x to a tensor of the given dtype on CPU, applies binary function func with
    # the literal y on the right (exercising forward operators), evaluates and returns the
    # result as a NumPy value.
    with test_util.force_cpu():
      inx = ops.convert_to_tensor(x, dtype=dtype)
      z = func(inx, y)  # Should use __add__, __sub__, etc.
      return self.evaluate(z)

  def _computeLiteralAndTensor(self, x, y, dtype, func):
    # [FUNCTION DOC]
    # HELPER: Converts y to a tensor of the given dtype on CPU, applies binary function func with
    # the literal x on the left (exercising reflected operators), evaluates and returns the
    # result as a NumPy value.
    with test_util.force_cpu():
      iny = ops.convert_to_tensor(y, dtype=dtype)
      z = func(x, iny)  # Should use __radd__, __rsub__, etc.
      return self.evaluate(z)

  def _compareBinary(self, x, y, dtype, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: Checks a binary operator overload both ways. Computes the NumPy reference for
    # literals x and y (cast to the dtype's NumPy type, or float32 for bfloat16, except for true
    # division where the float result is kept), then asserts the tensor-literal and
    # literal-tensor evaluations are close, with tolerance 1e-2 for bfloat16 and float16 and
    # 1e-6 otherwise.
    # astype and assertAllClose do not properly handle bfloat16 values
    np_ans = np_func(x, y)
    if np_func != np.true_divide:
      # for true_divide the result is a float, event for integer args.
      np_ans = np_ans.astype(np.float32 if dtype == dtypes_lib.bfloat16
                             else dtype.as_numpy_dtype)
    rtol = 1e-2 if dtype in (dtypes_lib.bfloat16, dtypes_lib.float16) else 1e-6
    self.assertAllClose(np_ans,
                        self._computeTensorAndLiteral(x, y, dtype, tf_func),
                        rtol=rtol)
    self.assertAllClose(np_ans,
                        self._computeLiteralAndTensor(x, y, dtype, tf_func),
                        rtol=rtol)

  def _compareUnary(self, x, dtype, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: Applies a unary TensorFlow operator to x converted to the given TensorFlow dtype,
    # forced on CPU, and asserts closeness with the NumPy function result cast to that dtype's
    # NumPy type.
    np_ans = np_func(x).astype(dtype.as_numpy_dtype)
    with test_util.force_cpu():
      self.assertAllClose(
          np_ans, self.evaluate(tf_func(ops.convert_to_tensor(x, dtype=dtype))))

  def testOverload(self):
    # [FUNCTION DOC]
    # Test: arithmetic operator overloads (add, subtract, multiply, power, true division, floor
    # division, modulo) between a tensor and a Python integer literal in both operand orders
    # (literals ten and three) across float, bfloat16, unsigned, signed and complex dtypes.
    # Skips floor division and modulo for complex dtypes and power for unsigned dtypes.
    dtypes = [
        dtypes_lib.float16,
        dtypes_lib.float32,
        dtypes_lib.float64,
        dtypes_lib.bfloat16,
        dtypes_lib.uint8,
        dtypes_lib.uint16,
        dtypes_lib.uint32,
        dtypes_lib.uint64,
        dtypes_lib.int8,
        dtypes_lib.int16,
        dtypes_lib.int32,
        dtypes_lib.int64,
        dtypes_lib.complex64,
        dtypes_lib.complex128,
    ]
    funcs = [
        (np.add, _ADD),
        (np.subtract, _SUB),
        (np.multiply, _MUL),
        (np.power, _POW),
        (np.true_divide, _TRUEDIV),
        (np.floor_divide, _FLOORDIV),
        (np.mod, _MOD),
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        with self.subTest(dtype=dtype, np_func=np_func, tf_func=tf_func):
          if dtype in (dtypes_lib.complex64,
                       dtypes_lib.complex128) and tf_func in (_FLOORDIV, _MOD):
            continue  # floordiv makes no sense for complex
          if dtype in (dtypes_lib.uint8, dtypes_lib.uint16, dtypes_lib.uint32,
                       dtypes_lib.uint64) and tf_func == _POW:
            continue  # power not supported for unsigned types
          self._compareBinary(10, 3, dtype, np_func, tf_func)

  def testOverloadComparisons(self):
    # [FUNCTION DOC]
    # Test: comparison and logical operators on tensors vs Python literals. For float, unsigned
    # and signed integer dtypes checks less, less-or-equal, greater and greater-or-equal with
    # literals ten and five. For booleans checks and, or, xor, equal and not-equal over all four
    # scalar combinations and a list case, and unary invert (logical not) on scalars and a list.
    # Each uses both operand orders via the helpers; results compared with NumPy.
    dtypes = [
        dtypes_lib.float16,
        dtypes_lib.float32,
        dtypes_lib.float64,
        dtypes_lib.uint8,
        dtypes_lib.uint16,
        dtypes_lib.uint32,
        dtypes_lib.uint64,
        dtypes_lib.int8,
        dtypes_lib.int16,
        dtypes_lib.int32,
        dtypes_lib.int64,
    ]
    funcs = [
        (np.less, _LT),
        (np.less_equal, _LE),
        (np.greater, _GT),
        (np.greater_equal, _GE),
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        with self.subTest(dtype=dtype, np_func=np_func, tf_func=tf_func):
          self._compareBinary(10, 5, dtype, np_func, tf_func)
    logical_funcs = [(np.logical_and, _AND), (np.logical_or, _OR),
                     (np.logical_xor, _XOR), (np.equal, math_ops.equal),
                     (np.not_equal, math_ops.not_equal)]
    for np_func, tf_func in logical_funcs:
      with self.subTest(np_func=np_func, tf_func=tf_func):
        self._compareBinary(True, False, dtypes_lib.bool, np_func, tf_func)
        self._compareBinary(True, True, dtypes_lib.bool, np_func, tf_func)
        self._compareBinary(False, False, dtypes_lib.bool, np_func, tf_func)
        self._compareBinary(False, True, dtypes_lib.bool, np_func, tf_func)
        self._compareBinary([True, True, False, False],
                            [True, False, True, False], dtypes_lib.bool,
                            np_func, tf_func)
    self._compareUnary(True, dtypes_lib.bool, np.logical_not, _INV)
    self._compareUnary(False, dtypes_lib.bool, np.logical_not, _INV)
    self._compareUnary([True, False], dtypes_lib.bool, np.logical_not, _INV)


class IsFiniteInfNanTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the is-finite, is-infinite and is-not-a-number ops over float16, float32,
  # float64 and bfloat16 special values, plus a (misplaced) square-root edge-case test.

  def _compare(self, x, use_gpu):
    # [FUNCTION DOC]
    # HELPER: Evaluates the is-finite, is-infinite and is-not-a-number ops on array x on the
    # chosen device and asserts the three boolean results and their shapes match NumPy's.
    # bfloat16 input is first cast to float32 for the NumPy side to avoid implicit float16 casts.
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      ofinite, oinf, onan = math_ops.is_finite(inx), math_ops.is_inf(
          inx), math_ops.is_nan(inx)
      tf_finite, tf_inf, tf_nan = self.evaluate([ofinite, oinf, onan])
    if x.dtype == dtypes_lib.bfloat16.as_numpy_dtype:
      # Numpy will implicitly convert bfloat16 value to float16, so we cast to
      # float32 to avoid this.
      x = x.astype(np.float32)
    np_finite, np_inf, np_nan = np.isfinite(x), np.isinf(x), np.isnan(x)
    self.assertAllEqual(np_inf, tf_inf)
    self.assertAllEqual(np_nan, tf_nan)
    self.assertAllEqual(np_finite, tf_finite)
    self.assertShapeEqual(np_inf, oinf)
    self.assertShapeEqual(np_nan, onan)
    self.assertShapeEqual(np_finite, ofinite)

  def _testDtype(self, dtype):
    # [FUNCTION DOC]
    # HELPER: Builds a test vector for the given NumPy dtype containing zero, plus and minus
    # one, the dtype's resolution (both signs), minimum and maximum, both infinities and a
    # not-a-number, then runs the comparison on CPU and GPU. For bfloat16, where machine limits
    # are unavailable, hardcoded approximate values are used instead.
    if dtype != dtypes_lib.bfloat16.as_numpy_dtype:
      fi = np.finfo(dtype)
      data = np.array([
          0, -1, 1, fi.resolution, -fi.resolution, fi.min, fi.max, -np.inf,
          np.inf, np.nan
      ]).astype(dtype)
    else:
      # np.finfo does not support bfloat16
      data = np.array([
          0, -1, 1, 0.01, -0.01, -3.3895e+38, 3.3895e+38, -np.inf, np.inf,
          np.nan
      ]).astype(dtype)
    self._compare(data, use_gpu=False)
    self._compare(data, use_gpu=True)

  def testHalf(self):
    # [FUNCTION DOC]
    # Test: finite, infinite and not-a-number classification for float16 data.
    self._testDtype(np.float16)

  def testFloat(self):
    # [FUNCTION DOC]
    # Test: finite, infinite and not-a-number classification for float32 data.
    self._testDtype(np.float32)

  def testDouble(self):
    # [FUNCTION DOC]
    # Test: finite, infinite and not-a-number classification for float64 data.
    self._testDtype(np.float64)

  def testBfloat16(self):
    # [FUNCTION DOC]
    # Test: finite, infinite and not-a-number classification for bfloat16 data.
    self._testDtype(dtypes_lib.bfloat16.as_numpy_dtype)

  def testSqrt(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: This test checks the square-root op but lives in the finite/infinite/NaN test
    # class; it does not exercise the finiteness ops directly (only the not-a-number op on
    # negative-input results).
    # Test: square root of constant vectors of several sizes (including vector-width boundaries)
    # and values (very negative, negative, zero, tiny, positive, large) for float16, float32 and
    # float64, on GPU if available. For negative inputs the result must be not-a-number exactly
    # where NumPy's is; otherwise values must be close according to dtype. Large values are
    # capped at one hundredth of the max because the vectorized float32 algorithm is inaccurate
    # for very large arguments.
    for dtype in [np.float16, np.float32, np.float64]:
      fi = np.finfo(dtype)
      for size in [1, 3, 4, 7, 8, 63, 64, 65]:
        # For float32 Eigen uses Carmack's fast vectorized sqrt algorithm.
        # It is not accurate for very large arguments, so we test for
        # fi.max/100 instead of fi.max here.
        for value in [fi.min, -2, -1, 0, fi.tiny, 1, 2, 1000, fi.max / 100]:
          with self.subTest(dtype=dtype, size=size, value=value):
            x = np.full((size,), value, dtype=dtype)
            np_y = np.sqrt(x)
            np_nan = np.isnan(np_y)
            with test_util.use_gpu():
              tf_y = math_ops.sqrt(x)
              tf_nan = math_ops.is_nan(tf_y)
              if value < 0:
                self.assertAllEqual(np_nan, self.evaluate(tf_nan))
              else:
                self.assertAllCloseAccordingToType(np_y, self.evaluate(tf_y))


class RoundingTest(test.TestCase):
  # [CLASS DOC]
  # Test case for rounding ops (floor, ceiling and round-to-nearest-even) against NumPy across
  # floating dtypes, with shared comparison helpers.

  def _compare_values(self, x, y=None):
    # [FUNCTION DOC]
    # HELPER: Checks the round-to-nearest-even op. Takes input x and an optional expected result
    # y (default: NumPy's round-to-even of x). Evaluates the op and asserts equality and equal
    # shape with the expected values.
    y = np.rint(x) if y is None else np.asarray(y)

    tf_rint = math_ops.rint(x)
    np_rint = self.evaluate(tf_rint)

    self.assertAllEqual(y, np_rint)
    self.assertShapeEqual(y, tf_rint)

  def _compare(self, x):
    # [FUNCTION DOC]
    # HELPER: Computes floor and ceiling of array x in one evaluation and asserts equality and
    # equal shapes with NumPy's results.
    np_floor, np_ceil = np.floor(x), np.ceil(x)

    inx = ops.convert_to_tensor(x)
    ofloor, oceil = math_ops.floor(inx), math_ops.ceil(inx)
    tf_floor, tf_ceil = self.evaluate([ofloor, oceil])

    self.assertAllEqual(np_floor, tf_floor)
    self.assertAllEqual(np_ceil, tf_ceil)
    self.assertShapeEqual(np_floor, ofloor)
    self.assertShapeEqual(np_ceil, oceil)

  def _testDtype(self, dtype):
    # [FUNCTION DOC]
    # HELPER: Runs rounding checks for one dtype. Builds quarter-step values and checks floor and
    # ceiling. For float16 and bfloat16 stops there since round-to-even is unsupported for them.
    # Otherwise also checks round-to-even on the data, on a near-half pair (0.5 rounds to 0 and
    # slightly above rounds to 1), and on a NumPy documentation example including negative zero.
    data = (np.arange(-3, 3) / 4.).reshape(1, 3, 2).astype(dtype)
    self._compare(data)
    # TODO(reedwm): rint op is not supported for float16 and bfloat16
    if dtype in (np.float16, dtypes_lib.bfloat16.as_numpy_dtype):
      return
    self._compare_values(data)
    x = [0.5, 0.5000001]
    y = [0.0, 1.0]
    self._compare_values(x, y=y)

    # numpy example
    x = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]
    y = [-2., -2., -0., 0., 2., 2., 2.]
    self._compare_values(x, y=y)

  def testTypes(self):
    # [FUNCTION DOC]
    # Test: floor, ceiling and round-to-even ops across float16, float32, float64 and bfloat16,
    # delegating to the dtype helper per type as sub-tests.
    for dtype in [np.float16, np.float32, np.float64,
                  dtypes_lib.bfloat16.as_numpy_dtype]:
      with self.subTest(dtype=dtype):
        self._testDtype(dtype)


class ComplexMakeRealImagTest(test.TestCase):
  # [CLASS DOC]
  # Test case for complex-number ops: construction from real and imaginary parts, extraction of
  # real and imaginary parts, phase angle, conjugation, and gradients through these ops.

  def _compareMake(self, real, imag, use_gpu):
    # [FUNCTION DOC]
    # HELPER: Builds a complex tensor from real and imaginary parts (arrays or scalars) on the
    # chosen device and asserts equality and equal shape with the NumPy-built complex values.
    np_ans = real + (1j) * imag

    with test_util.device(use_gpu=use_gpu):
      real = ops.convert_to_tensor(real)
      imag = ops.convert_to_tensor(imag)
      tf_ans = math_ops.complex(real, imag)
      out = self.evaluate(tf_ans)

    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def testMake(self):
    # [FUNCTION DOC]
    # Test: building complex numbers from float32 parts on CPU and GPU in three forms: array
    # real with array imaginary, array real with scalar imaginary, and scalar real with array
    # imaginary. Compared against NumPy.
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    for use_gpu in [False, True]:
      with self.subTest(use_gpu=use_gpu):
        self._compareMake(real, imag, use_gpu)
        self._compareMake(real, 12.0, use_gpu)
        self._compareMake(23.0, imag, use_gpu)

  def testRealImagNumericType(self):
    # [FUNCTION DOC]
    # Test: real and imaginary part ops accept plain Python numbers (a real float, an imaginary
    # number and a complex number) on both devices and return the same values as NumPy.
    for use_gpu in [True, False]:
      for value in [1., 1j, 1. + 1j]:
        with self.subTest(use_gpu=use_gpu, value=value):
          np_real, np_imag = np.real(value), np.imag(value)
          with test_util.device(use_gpu=use_gpu):
            tf_real = math_ops.real(value)
            tf_imag = math_ops.imag(value)
            self.assertAllEqual(np_real, self.evaluate(tf_real))
            self.assertAllEqual(np_imag, self.evaluate(tf_imag))

  def _compareRealImag(self, cplx, use_gpu):
    # [FUNCTION DOC]
    # HELPER: On the chosen device, extracts real and imaginary parts of cplx and asserts they
    # equal NumPy's. Also applies the real and imaginary ops to the real part, asserting the
    # real of a real is itself and the imaginary of a real is all zeros.
    np_real, np_imag = np.real(cplx), np.imag(cplx)
    np_zeros = np_real * 0

    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(cplx)
      tf_real = math_ops.real(inx)
      tf_imag = math_ops.imag(inx)
      tf_real_real = math_ops.real(tf_real)
      tf_imag_real = math_ops.imag(tf_real)
      self.assertAllEqual(np_real, self.evaluate(tf_real))
      self.assertAllEqual(np_imag, self.evaluate(tf_imag))
      self.assertAllEqual(np_real, self.evaluate(tf_real_real))
      self.assertAllEqual(np_zeros, self.evaluate(tf_imag_real))

  def testRealImag64(self):
    # [FUNCTION DOC]
    # Test: real and imaginary parts of a complex64 array (from float32 parts) on CPU and GPU.
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + 1j * imag
    self._compareRealImag(cplx, use_gpu=False)
    self._compareRealImag(cplx, use_gpu=True)

  def testRealImag128(self):
    # [FUNCTION DOC]
    # Test: real and imaginary parts of a complex128 array (from float64 parts) on CPU and GPU.
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float64)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float64)
    cplx = real + 1j * imag
    self._compareRealImag(cplx, use_gpu=False)
    self._compareRealImag(cplx, use_gpu=True)

  def _compareAngle(self, cplx, use_gpu):
    # [FUNCTION DOC]
    # HELPER: Computes the phase angle of cplx (complex or real array) on the chosen device and
    # asserts closeness and equal shape with NumPy's angle.
    np_angle = np.angle(cplx)

    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(cplx)
      tf_angle = math_ops.angle(inx)
      tf_angle_val = self.evaluate(tf_angle)

    self.assertAllClose(np_angle, tf_angle_val)
    self.assertShapeEqual(np_angle, tf_angle)

  def testAngle(self):
    # [FUNCTION DOC]
    # Test: phase angle of random complex values (float32 magnitude and phases spread over a
    # full turn) plus the four axis points, and of a real float64 array, on CPU and GPU against
    # NumPy.
    mag = np.random.rand(10).astype(np.float32)
    angle = (2 * np.pi * np.arange(10) / 10.).astype(np.float32)
    cplx = mag * np.exp(1j * angle)
    cplx = np.append(cplx, [1., 1.j, -1., -1.j])
    self._compareAngle(cplx, use_gpu=False)
    self._compareAngle(cplx, use_gpu=True)
    real = (np.arange(-2, 2) / 2.).astype(np.float64)
    self._compareAngle(real, use_gpu=False)
    self._compareAngle(real, use_gpu=True)

  def testAngle64(self):
    # [FUNCTION DOC]
    # Test: phase angle of random complex128 values (float64 magnitudes, phases spread over a
    # tenth of a turn) plus the four axis points, and of a real float64 array, on CPU and GPU
    # against NumPy.
    mag = np.random.rand(10).astype(np.float64)
    angle = (2 * np.pi * np.arange(10) / 100.).astype(np.float64)
    cplx = mag * np.exp(1j * angle)
    cplx = np.append(cplx, [1., 1.j, -1., -1.j])
    self._compareAngle(cplx, use_gpu=False)
    self._compareAngle(cplx, use_gpu=True)
    real = (np.arange(-2, 2) / 2.).astype(np.float64)
    self._compareAngle(real, use_gpu=False)
    self._compareAngle(real, use_gpu=True)

  @test_util.run_deprecated_v1
  def testRealReal(self):
    # [FUNCTION DOC]
    # Test: the real-part op on real-typed placeholders (int32, int64, float32, float64)
    # returns the same tensor object unchanged. Graph mode only.
    for dtype in (dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.float32,
                  dtypes_lib.float64):
      with self.subTest(dtype=dtype):
        x = array_ops.placeholder(dtype)
        y = math_ops.real(x)
        self.assertEqual(x, y)

  def _compareConj(self, cplx, use_gpu):
    # [FUNCTION DOC]
    # HELPER: Computes the conjugate of complex array cplx on the chosen device (use_gpu true
    # selects GPU) and asserts equality and equal shape with NumPy's conjugate.
    np_ans = np.conj(cplx)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(cplx)
      tf_conj = math_ops.conj(inx)
      tf_ans = self.evaluate(tf_conj)
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, tf_conj)

  def testConj64(self):
    # [FUNCTION DOC]
    # Test: conjugation of a float32-based complex64 array on CPU and GPU against NumPy.
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + 1j * imag
    self._compareConj(cplx, use_gpu=False)
    self._compareConj(cplx, use_gpu=True)

  def testConj128(self):
    # [FUNCTION DOC]
    # Test: conjugation of a float64-based complex128 array on CPU and GPU against NumPy.
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float64)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float64)
    cplx = real + 1j * imag
    self._compareConj(cplx, use_gpu=False)
    self._compareConj(cplx, use_gpu=True)

  @test_util.run_deprecated_v1
  def testConjReal(self):
    # [FUNCTION DOC]
    # Test: conjugation of a real-typed placeholder (int32, int64, float16, float32, float64)
    # returns the same tensor object unchanged. Graph mode only.
    for dtype in (dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.float16,
                  dtypes_lib.float32, dtypes_lib.float64):
      with self.subTest(dtype=dtype):
        x = array_ops.placeholder(dtype)
        y = math_ops.conj(x)
        self.assertEqual(x, y)

  @test_util.run_deprecated_v1
  def testConjString(self):
    # [FUNCTION DOC]
    # Test: conjugation of a string placeholder must fail with a type error saying a numeric or
    # variant tensor was expected. Graph mode only.
    x = array_ops.placeholder(dtypes_lib.string)
    with self.assertRaisesRegex(TypeError,
                                r"Expected numeric or variant tensor"):
      math_ops.conj(x)

  def _compareGradient(self, x):
    # [FUNCTION DOC]
    # HELPER: Gradient check through complex construction and conjugation. x is a float matrix
    # whose first column is real and second imaginary. Combines them into complex numbers,
    # conjugates, and takes the loss as the sum of squared real parts plus squared imaginary
    # parts. Asserts analytic and numeric Jacobians agree within 1e-3 (step 1e-3).
    # x[:, 0] is real, x[:, 1] is imag.  We combine real and imag into
    # complex numbers. Then, we extract real and imag parts and
    # computes the squared sum. This is obviously the same as sum(real
    # * real) + sum(imag * imag). We just want to make sure the
    # gradient function is checked.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      real, imag = array_ops.split(value=inx, num_or_size_splits=2, axis=1)
      real, imag = array_ops.reshape(real, [-1]), array_ops.reshape(imag, [-1])
      cplx = math_ops.complex(real, imag)
      cplx = math_ops.conj(cplx)
      loss = math_ops.reduce_sum(math_ops.square(
          math_ops.real(cplx))) + math_ops.reduce_sum(
              math_ops.square(math_ops.imag(cplx)))
      epsilon = 1e-3
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, list(x.shape), loss, list(loss.shape), x_init_value=x,
          delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  def _compareBroadcastGradient(self, x):
    # [FUNCTION DOC]
    # HELPER: Gradient check of complex construction with a scalar zero as one part. For input
    # x, tries the complex op with (x, zero) and (zero, x), sums the magnitudes, and asserts
    # analytic and numeric gradients with respect to x agree within 1e-3, as sub-tests.
    x_ = ops.convert_to_tensor(x)
    epsilon = 1e-3
    with self.cached_session():
      for args in [(x_, 0.), (0., x_)]:
        with self.subTest(args=args):
          z = math_ops.reduce_sum(math_ops.abs(math_ops.complex(*args)))
          jacob_t, jacob_n = gradient_checker.compute_gradient(
              x_, list(x.shape), z, [1], x_init_value=x, delta=epsilon)
          self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Test: gradients of complex construction, conjugation, real and imaginary extraction on
    # a five-by-two float32 matrix (complex64) and a float64 matrix (complex128). The broadcast
    # gradient check is run only for the float32 data. Graph mode only.
    # complex64
    data = np.arange(1, 2, 0.10).reshape([5, 2]).astype(np.float32)
    self._compareGradient(data)
    self._compareBroadcastGradient(data)
    # complex128
    data = np.arange(1, 2, 0.10).reshape([5, 2]).astype(np.float64)
    self._compareGradient(data)

  def _compareMulGradient(self, data):
    # [FUNCTION DOC]
    # HELPER: Gradient check for complex multiplication. data is a float matrix with four
    # columns: real and imaginary parts of x, then of y. Builds complex vectors x and y,
    # multiplies them, defines the loss as the sum of real plus imaginary parts of the product,
    # and asserts analytic and numeric Jacobians agree within 0.005 (step 0.005).
    # data is a float matrix of shape [n, 4].  data[:, 0], data[:, 1],
    # data[:, 2], data[:, 3] are real parts of x, imaginary parts of
    # x, real parts of y and imaginary parts of y.
    with self.cached_session():
      inp = ops.convert_to_tensor(data)
      xr, xi, yr, yi = array_ops.split(value=inp, num_or_size_splits=4, axis=1)

      def vec(x):  # Reshape to a vector
        # [FUNCTION DOC]
        # HELPER: Flattens tensor x into a one-dimensional vector tensor.
        return array_ops.reshape(x, [-1])

      xr, xi, yr, yi = vec(xr), vec(xi), vec(yr), vec(yi)

      def cplx(r, i):  # Combine to a complex vector
        # [FUNCTION DOC]
        # HELPER: Combines real part r and imaginary part i vectors into one complex vector
        # tensor.
        return math_ops.complex(r, i)

      x, y = cplx(xr, xi), cplx(yr, yi)
      # z is x times y in complex plane.
      z = x * y
      # Defines the loss function as the sum of all coefficients of z.
      loss = math_ops.reduce_sum(math_ops.real(z) + math_ops.imag(z))
      epsilon = 0.005
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inp, list(data.shape), loss, [1], x_init_value=data, delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  @test_util.run_deprecated_v1
  def testMulGradient(self):
    # [FUNCTION DOC]
    # Test: gradient of complex multiplication. Builds a float32 matrix with two rows by four
    # columns (real and imaginary parts of two numbers) and checks the multiplication gradient
    # through the helper. Runs in graph mode only.
    data = np.arange(1, 2, 0.125).reshape([2, 4]).astype(np.float32)
    self._compareMulGradient(data)


class PolyvalTest(test.TestCase):
  # [CLASS DOC]
  # Test case for polynomial evaluation (math_ops polyval) compared against NumPy: various
  # dtypes and degrees, broadcasting, empty coefficients, and invalid coefficient types.

  def _runtest(self, dtype, degree):
    # [FUNCTION DOC]
    # HELPER: Shared check for polynomial evaluation. Takes a NumPy dtype and a degree; builds a
    # random two-by-two input and degree-plus-one random coefficient matrices, evaluates the
    # polynomial with TensorFlow in a cached session, and asserts closeness to NumPy's result.
    x = np.random.rand(2, 2).astype(dtype)
    coeffs = [np.random.rand(2, 2).astype(dtype) for _ in range(degree + 1)]
    np_val = np.polyval(coeffs, x)
    with self.cached_session():
      tf_val = math_ops.polyval(coeffs, x)
      self.assertAllClose(np_val, self.evaluate(tf_val))

  def testSimple(self):
    # [FUNCTION DOC]
    # Test: polynomial evaluation for int32, float32, float64, complex64 and complex128 with
    # degrees zero through four on random two-by-two inputs and coefficients; results must be
    # close to NumPy's polynomial evaluation (sub-test per dtype and degree).
    for dtype in [
        np.int32, np.float32, np.float64, np.complex64, np.complex128
    ]:
      for degree in range(5):
        with self.subTest(dtype=dtype, degree=degree):
          self._runtest(dtype, degree)

  def testBroadcast(self):
    # [FUNCTION DOC]
    # Test: polynomial evaluation where the input and each of four coefficients have different
    # but broadcast-compatible shapes (all pairs from four shapes). Results must be close to
    # NumPy's polynomial evaluation, run as sub-tests per shape pair.
    dtype = np.float32
    degree = 3
    shapes = [(1,), (2, 1), (1, 2), (2, 2)]
    for x_shape in shapes:
      for coeff_shape in shapes:
        with self.subTest(x_shape=x_shape, coeff_shape=coeff_shape):
          x = np.random.rand(*x_shape).astype(dtype)
          coeffs = [
              np.random.rand(*coeff_shape).astype(dtype)
              for _ in range(degree + 1)
          ]
          np_val = np.polyval(coeffs, x)
          with self.cached_session():
            tf_val = math_ops.polyval(coeffs, x)
            self.assertAllClose(np_val, self.evaluate(tf_val))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: polynomial evaluation with an empty coefficient list on a random float32 matrix.
    # The TensorFlow result must be close to NumPy's result (all zeros).
    x = np.random.rand(2, 2).astype(np.float32)
    coeffs = []
    np_val = np.polyval(coeffs, x)
    with self.cached_session():
      tf_val = math_ops.polyval(coeffs, x)
      self.assertAllClose(np_val, self.evaluate(tf_val))

  def test_coeffs_raise(self):
    # [FUNCTION DOC]
    # Test: polynomial evaluation rejects coefficients that are not a list (an empty mapping is
    # passed). Expects a value error stating that the coefficients argument must be a list.
    x = np.random.rand(2, 2).astype(np.float32)
    coeffs = {}
    with self.assertRaisesRegex(ValueError, "Argument coeffs must be list"):
      math_ops.polyval(coeffs, x)


class RealTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the real-part op on complex and real inputs, and its error on non-numeric
  # inputs. Uses one shared helper that compares the op output with expected values.

  def _run_test(self, input_values, expected_values):
    # [FUNCTION DOC]
    # HELPER: Applies the real-part op to input_values (list or array) and asserts the result
    # equals expected_values exactly. Returns nothing; propagates errors from the op.
    res = math_ops.real(input_values)
    self.assertAllEqual(res, expected_values)

  def test_real(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Each case carries a dtype, but the inputs are plain lists, so the dtype is
    # only used as a label for the sub-test and is never applied to the input values.
    # Test: the real-part op on complex lists (returns the real components) and on real lists
    # (returns the values unchanged). Each case is run through the helper under a sub-test.
    test_cases = [
        # Complex tensor
        (np.complex64, [-2.25 + 4.75j, 3.25 + 5.75j], [-2.25, 3.25]),
        (np.complex128, [-2.25 + 4.75j, 3.25 + 5.75j], [-2.25, 3.25]),
        # Real tensor
        (np.float32, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        (np.float64, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ]

    for dtype, input_values, expected_values in test_cases:
      with self.subTest(dtype=dtype):
        self._run_test(input_values, expected_values)

  def test_real_raises_error_for_non_numeric_tensor(self):
    # [FUNCTION DOC]
    # Test: the real-part op rejects non-numeric input. Passes an array of two strings through
    # the helper and expects a type error whose message says the input must be a numeric tensor.
    x = np.array(["Hello", "World"])
    with self.assertRaisesRegex(TypeError, "input must be a numeric tensor"):
      self._run_test(x, None)


if __name__ == "__main__":
  test.main()
