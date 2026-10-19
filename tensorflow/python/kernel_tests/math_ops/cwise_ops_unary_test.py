# [MODULE DOC]
# Test module for unary coefficient-wise (elementwise) math ops. Compares TensorFlow results with
# NumPy/SciPy/math references for many dtypes, in dense and sparse form on CPU and GPU, including
# gradient and second-order gradient checks. Defines two private module constants for negation
# and absolute value operators. No import-time side effects beyond imports; running as a script
# starts the TensorFlow test runner.
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
"""Functional tests for unary coefficient-wise operations."""

import math

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

_NEG = lambda x: -x
_ABS = abs


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  # [FUNCTION DOC]
  # HELPER: Builds a sparse tensor from a dense array for the sparse op tests.
  # Inputs: array x, threshold (default 0.5) below which entries are zeroed, and the integer
  # dtype for indices (default 64-bit). Side effect: modifies x in place by setting entries below
  # the threshold to zero. Returns a pair: the sparse tensor of the nonzero entries, and the
  # one-dimensional array of those nonzero values.
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), x_values


def _default_tolerance(dtype):
  # [FUNCTION DOC]
  # HELPER: Picks comparison tolerance for gradient checks by dtype. Takes a NumPy dtype and
  # returns 5e-3 for bfloat16 and float16, 1e-3 for float32 and complex64, 1e-5 for float64 and
  # complex128, and nothing (null value) for any other dtype so that callers fail fast.
  """Returns a sensible default tolerance for comparing results of a given type.

  Args:
    dtype: A datatype.
  """
  if dtype == dtypes_lib.bfloat16.as_numpy_dtype:
    return 5e-3
  if dtype == np.float16:
    return 5e-3
  elif dtype in (np.float32, np.complex64):
    return 1e-3
  elif dtype in (np.float64, np.complex128):
    return 1e-5
  else:
    return None  # Fail fast for unexpected types


class UnaryOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for unary elementwise math ops (abs, negation, exp, log, trigonometric, hyperbolic,
  # sigmoid, sign, erf, lgamma and others) across integer, floating and complex dtypes, dense and
  # sparse, CPU and GPU. Shared comparison helpers evaluate each op against a NumPy reference
  # and optionally check gradients. Also covers second-order gradients. No instance state.

  def _compareCpu(self, x, np_func, tf_func, grad_rtol=None, grad_atol=None):
    # [FUNCTION DOC]
    # HELPER: Dense CPU comparison with gradient check. Inputs: array x, NumPy reference
    # function, TensorFlow op, and optional gradient relative and absolute tolerances (default:
    # per-dtype defaults). Evaluates the op on CPU, checks shape and values against the
    # reference (looser tolerance for half and bfloat16). Returns early, skipping gradients, for
    # complex sign and for round. Otherwise compares analytic and numeric Jacobians, using a
    # float64 recomputation for half/bfloat16 and finite-difference steps per dtype for float32,
    # complex64, float64 and complex128. Integer dtypes get no gradient check.
    if grad_rtol is None:
      grad_rtol = _default_tolerance(x.dtype)
    if grad_atol is None:
      grad_atol = _default_tolerance(x.dtype)
    np_ans = np_func(x)
    with self.cached_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      y = tf_func(inx)
      tf_cpu = self.evaluate(y)
      self.assertShapeEqual(np_ans, y)
      if x.dtype == np.float16:
        self.assertAllClose(np_ans, tf_cpu, rtol=1e-3, atol=1e-3)
      elif x.dtype == dtypes_lib.bfloat16.as_numpy_dtype:
        self.assertAllClose(np_ans, tf_cpu, rtol=1e-2, atol=1e-2)
      else:
        self.assertAllClose(np_ans, tf_cpu)

      if x.dtype in (np.complex64, np.complex128) and tf_func == math_ops.sign:
        return  # Return early

      if tf_func == math_ops.round:
        return  # Return early

      if x.dtype in (np.float16, dtypes_lib.bfloat16.as_numpy_dtype):
        s = list(np.shape(x))
        jacob_t, _ = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x)
        xf = x.astype(np.float64)
        inxf = ops.convert_to_tensor(xf)
        yf = tf_func(inxf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, yf, s, x_init_value=xf, delta=1e-2)
        jacob_n = jacob_n.astype(x.dtype)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
      elif x.dtype in (np.float32, np.complex64):
        s = list(np.shape(x))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x, delta=1e-3)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
      elif x.dtype in (np.float64, np.complex128):
        s = list(np.shape(x))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x, delta=1e-5)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)

  def _check(self, result_tensor, result_np, input_sp_t, tol):
    # [FUNCTION DOC]
    # HELPER: Assertion routine for sparse results. Asserts both the result and the input are
    # sparse tensors, that indices and dense shape are unchanged by the op, and that the result
    # values are close to the NumPy values (default closeness if tol is none, otherwise tol as
    # both relative and absolute tolerance). Returns nothing.
    self.assertTrue(isinstance(result_tensor, sparse_tensor.SparseTensor))
    self.assertTrue(isinstance(input_sp_t, sparse_tensor.SparseTensor))
    self.assertAllEqual(input_sp_t.indices, result_tensor.indices)
    self.assertAllEqual(input_sp_t.dense_shape, result_tensor.dense_shape)
    if tol is None:
      self.assertAllClose(result_np, result_tensor.values)
    else:
      self.assertAllClose(result_np, result_tensor.values, rtol=tol, atol=tol)

  def _compareSparseCpu(self, x, np_func, tf_func, tol):
    # [FUNCTION DOC]
    # HELPER: Sparse-tensor comparison forced onto the CPU. Converts x to a sparse tensor by
    # zeroing small entries (modifying x in place), applies the op, and checks indices, dense
    # shape and values against the NumPy reference applied to the stored values, with tolerance
    # tol (none means default).
    x_sp, x_sp_vals = _sparsify(x)
    res_np = np_func(x_sp_vals)
    with test_util.force_cpu():
      self._check(tf_func(x_sp), res_np, x_sp, tol)

  def _compareGpu(self, x, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: Dense comparison on the GPU device (default device if no GPU). Applies the NumPy
    # reference to x, evaluates the TensorFlow op on the converted tensor, and asserts the two
    # are close according to dtype with an absolute tolerance of 2e-6. No gradient check is done.
    np_ans = np_func(x)
    with test_util.use_gpu():
      result = tf_func(ops.convert_to_tensor(x))
      tf_gpu = self.evaluate(result)
      # Slightly increase the tolerance for float64 computations. This is
      # desired for specifically lgamma but shouldn't be of concern for other
      # functions.
      self.assertAllCloseAccordingToType(np_ans, tf_gpu, atol=2e-6)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareSparseGpu(self, x, np_func, tf_func, tol):
    # [FUNCTION DOC]
    # HELPER: Sparse-tensor comparison on the GPU device (when one is available; otherwise the
    # default device is used). Converts x to a sparse tensor by zeroing small entries (modifying
    # x in place), applies the op, and checks indices, dense shape and values against the NumPy
    # reference applied to the stored values, with tolerance tol (none means default).
    x_sp, x_sp_vals = _sparsify(x)
    res_np = np_func(x_sp_vals)
    with test_util.use_gpu():
      self._check(tf_func(x_sp), res_np, x_sp, tol)

  def _compareBoth(self, x, np_func, tf_func, grad_tol=None):
    # [FUNCTION DOC]
    # HELPER: Runs the dense comparison on CPU (with gradient check) and then on GPU (value
    # check only). Inputs: array x, NumPy reference function, TensorFlow op, and optional
    # gradient tolerance applied to both relative and absolute tolerance (default none, meaning
    # per-dtype defaults). Returns nothing; fails the test on mismatch.
    self._compareCpu(x, np_func, tf_func, grad_rtol=grad_tol,
                     grad_atol=grad_tol)
    self._compareGpu(x, np_func, tf_func)

  def _compareBothSparse(self, x, np_func, tf_func, tol=None):
    # [FUNCTION DOC]
    # HELPER: Runs the sparse comparison on CPU and then on GPU. Inputs: dense array x, NumPy
    # reference function, TensorFlow op, and optional tolerance (default none, meaning the
    # default closeness). Returns nothing; fails the test on mismatch.
    self._compareSparseCpu(x, np_func, tf_func, tol)
    self._compareSparseGpu(x, np_func, tf_func, tol)

  def _inv(self, x):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for the reciprocal op. Returns one divided by x elementwise.
    return 1.0 / x

  def _rsqrt(self, x):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for reciprocal square root. Returns the reciprocal of the
    # elementwise square root of x.
    return self._inv(np.sqrt(x))

  def _sigmoid(self, x):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for the logistic function. Returns one over one plus the
    # exponential of minus x, elementwise.
    return 1.0 / (1.0 + np.exp(-x))

  def _log_sigmoid(self, x):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for the log-sigmoid op. Returns the natural log of the logistic
    # function of x elementwise (not numerically stable for very negative inputs).
    return np.log(self._sigmoid(x))

  def _replace_domain_error_with_inf(self, fn):
    # [FUNCTION DOC]
    # HELPER: Adapts a scalar math function (such as the log-gamma function) to match the
    # TensorFlow convention of returning infinity at poles. Takes a callable fn and returns a
    # wrapper that converts domain-error value errors into infinity (see the wrapper).

    def func(x):
      # [FUNCTION DOC]
      # HELPER: The wrapper returned by the enclosing helper. Calls the wrapped scalar function
      # on x; if it raises a value error whose message mentions a domain error, returns an
      # array of positive infinity shaped like x; any other value error is re-raised.
      try:
        return fn(x)
      except ValueError as e:
        if "domain error" in str(e):
          return np.inf * np.ones_like(x)
        else:
          raise e

    return func

  @test_util.run_deprecated_v1
  def testFloatBasic(self):
    # [FUNCTION DOC]
    # Test: unary ops on a small float32 array and derived arrays (greater than one, nonzero,
    # positive, and within minus one to one so domain-restricted ops are valid). Dense
    # comparisons on CPU and GPU for abs, negation (named op and Python operator), reciprocal,
    # square, sqrt, rsqrt, exp, expm1, log, log1p, hyperbolic, trigonometric and inverse
    # functions, sigmoid, log-sigmoid, sign, lgamma (domain errors mapped to infinity), erf and
    # erfc, plus Bessel functions when SciPy imports (else a warning is logged). Also sparse
    # comparisons for abs, negation, square, sqrt (tolerance 1e-3), tanh, sign and erf.
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    w = x - x.min() + 1.02  # all greater than 1
    y = (x + .5).astype(np.float32)  # no zero
    z = (x + 15.5).astype(np.float32)  # all positive
    k = np.arange(-0.90, 0.90, 0.25).astype(np.float32)  # between -1 and 1

    self._compareBoth(x, np.abs, math_ops.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, math_ops.reciprocal)
    self._compareBoth(x, np.square, math_ops.square)
    self._compareBoth(z, np.sqrt, math_ops.sqrt)
    self._compareBoth(z, self._rsqrt, math_ops.rsqrt)
    self._compareBoth(x, np.exp, math_ops.exp)
    self._compareBoth(x, np.expm1, math_ops.expm1)
    self._compareBoth(z, np.log, math_ops.log)
    self._compareBoth(z, np.log1p, math_ops.log1p)
    self._compareBoth(x, np.sinh, math_ops.sinh)
    self._compareBoth(x, np.cosh, math_ops.cosh)
    self._compareBoth(x, np.tanh, math_ops.tanh)
    self._compareBoth(x, np.arcsinh, math_ops.asinh)
    self._compareBoth(w, np.arccosh, math_ops.acosh)
    self._compareBoth(k, np.arctanh, math_ops.atanh)
    self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareBoth(x, self._log_sigmoid, math_ops.log_sigmoid)
    self._compareBoth(y, np.sign, math_ops.sign)
    self._compareBoth(x, np.sin, math_ops.sin)
    self._compareBoth(x, np.cos, math_ops.cos)
    self._compareBoth(k, np.arcsin, math_ops.asin)
    self._compareBoth(k, np.arccos, math_ops.acos)
    self._compareBoth(x, np.arctan, math_ops.atan)
    self._compareBoth(x, np.tan, math_ops.tan)
    self._compareBoth(
        y, np.vectorize(self._replace_domain_error_with_inf(math.lgamma)),
        math_ops.lgamma)
    self._compareBoth(x, np.vectorize(math.erf), math_ops.erf)
    self._compareBoth(x, np.vectorize(math.erfc), math_ops.erfc)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      self._compareBoth(x, special.i0e, special_math_ops.bessel_i0e)
      self._compareBoth(x, special.i1e, special_math_ops.bessel_i1e)
    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

    self._compareBothSparse(x, np.abs, math_ops.abs)
    self._compareBothSparse(x, np.negative, math_ops.negative)
    self._compareBothSparse(x, np.square, math_ops.square)
    self._compareBothSparse(z, np.sqrt, math_ops.sqrt, tol=1e-3)
    self._compareBothSparse(x, np.tanh, math_ops.tanh)
    self._compareBothSparse(y, np.sign, math_ops.sign)
    self._compareBothSparse(x, np.vectorize(math.erf), math_ops.erf)

  @test_util.run_deprecated_v1
  def testFloatTanhEdge(self):
    # [FUNCTION DOC]
    # Test: hyperbolic tangent on large-magnitude float32 inputs (six values starting at 40 and
    # six starting at minus 40), where the result saturates near one and minus one. Compared
    # with NumPy on both CPU and GPU.
    x = np.arange(40, 40 + 6).reshape(6).astype(np.float32)
    self._compareBoth(x, np.tanh, math_ops.tanh)
    x = np.arange(-40, -40 + 6).reshape(6).astype(np.float32)
    self._compareBoth(x, np.tanh, math_ops.tanh)

  @test_util.run_deprecated_v1
  def testFloatEmpty(self):
    # [FUNCTION DOC]
    # Test: unary ops on an empty float32 tensor of shape two by zero by five, dense on CPU and
    # GPU and sparse. Because the vectorized reference cannot run on empty data, lgamma, erf and
    # erfc are compared against an arbitrary NumPy function (sign), which is fine since there are
    # no elements. Bessel functions are included when SciPy is available; otherwise a warning is
    # logged. Verifies the ops handle zero-size inputs and return matching empty outputs.
    x = np.empty((2, 0, 5), dtype=np.float32)
    self._compareBoth(x, np.abs, math_ops.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(x, self._inv, math_ops.reciprocal)
    self._compareBoth(x, np.square, math_ops.square)
    self._compareBoth(x, np.sqrt, math_ops.sqrt)
    self._compareBoth(x, self._rsqrt, math_ops.rsqrt)
    self._compareBoth(x, np.exp, math_ops.exp)
    self._compareBoth(x, np.expm1, math_ops.expm1)
    self._compareBoth(x, np.log, math_ops.log)
    self._compareBoth(x, np.log1p, math_ops.log1p)
    self._compareBoth(x, np.sinh, math_ops.sinh)
    self._compareBoth(x, np.arcsinh, math_ops.asinh)
    self._compareBoth(x, np.cosh, math_ops.cosh)
    self._compareBoth(x, np.arccosh, math_ops.acosh)
    self._compareBoth(x, np.tanh, math_ops.tanh)
    self._compareBoth(x, np.arctanh, math_ops.atanh)
    self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareBoth(x, np.sign, math_ops.sign)
    self._compareBoth(x, np.sin, math_ops.sin)
    self._compareBoth(x, np.cos, math_ops.cos)
    # Can't use vectorize below, so just use some arbitrary function
    self._compareBoth(x, np.sign, math_ops.lgamma)
    self._compareBoth(x, np.sign, math_ops.erf)
    self._compareBoth(x, np.sign, math_ops.erfc)
    self._compareBoth(x, np.tan, math_ops.tan)
    self._compareBoth(x, np.arcsin, math_ops.asin)
    self._compareBoth(x, np.arccos, math_ops.acos)
    self._compareBoth(x, np.arctan, math_ops.atan)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      self._compareBoth(x, special.i0e, special_math_ops.bessel_i0e)
      self._compareBoth(x, special.i1e, special_math_ops.bessel_i1e)
    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

    self._compareBothSparse(x, np.abs, math_ops.abs)
    self._compareBothSparse(x, np.negative, math_ops.negative)
    self._compareBothSparse(x, np.square, math_ops.square)
    self._compareBothSparse(x, np.sqrt, math_ops.sqrt, tol=1e-3)
    self._compareBothSparse(x, np.tanh, math_ops.tanh)
    self._compareBothSparse(x, np.sign, math_ops.sign)
    self._compareBothSparse(x, np.sign, math_ops.erf)

  @test_util.run_deprecated_v1
  def testDoubleBasic(self):
    # [FUNCTION DOC]
    # Test: unary ops on a small float64 array and derived arrays (greater than one, nonzero,
    # positive, and within minus one to one). Dense comparisons on CPU and GPU for abs, negation,
    # reciprocal, square, sqrt, rsqrt, exp, expm1, log, log1p, hyperbolic, trigonometric and
    # inverse functions, sigmoid, sign, lgamma, erf and erfc, plus Bessel functions when SciPy is
    # importable (otherwise a warning is logged). Also sparse comparisons for abs, negation,
    # square, sqrt, tanh, sign and erf.
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64)
    w = x - x.min() + 1.02  # all greater than 1
    y = (x + .5).astype(np.float64)  # no zero
    z = (x + 15.5).astype(np.float64)  # all positive
    k = np.arange(-0.90, 0.90,
                  0.35).reshape(1, 3, 2).astype(np.float64)  # between -1 and 1
    self._compareBoth(x, np.abs, math_ops.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, math_ops.reciprocal)
    self._compareBoth(x, np.square, math_ops.square)
    self._compareBoth(z, np.sqrt, math_ops.sqrt)
    self._compareBoth(z, self._rsqrt, math_ops.rsqrt)
    self._compareBoth(x, np.exp, math_ops.exp)
    self._compareBoth(x, np.expm1, math_ops.expm1)
    self._compareBoth(z, np.log, math_ops.log)
    self._compareBoth(z, np.log1p, math_ops.log1p)
    self._compareBoth(x, np.sinh, math_ops.sinh)
    self._compareBoth(x, np.cosh, math_ops.cosh)
    self._compareBoth(x, np.tanh, math_ops.tanh)
    self._compareBoth(x, np.arcsinh, math_ops.asinh)
    self._compareBoth(w, np.arccosh, math_ops.acosh)
    self._compareBoth(k, np.arctanh, math_ops.atanh)
    self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareBoth(y, np.sign, math_ops.sign)
    self._compareBoth(x, np.sin, math_ops.sin)
    self._compareBoth(x, np.cos, math_ops.cos)
    self._compareBoth(
        y, np.vectorize(self._replace_domain_error_with_inf(math.lgamma)),
        math_ops.lgamma)
    self._compareBoth(x, np.vectorize(math.erf), math_ops.erf)
    self._compareBoth(x, np.vectorize(math.erfc), math_ops.erfc)
    self._compareBoth(x, np.arctan, math_ops.atan)
    self._compareBoth(k, np.arcsin, math_ops.asin)
    self._compareBoth(k, np.arccos, math_ops.acos)
    self._compareBoth(k, np.tan, math_ops.tan)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      self._compareBoth(x, special.i0e, special_math_ops.bessel_i0e)
      self._compareBoth(x, special.i1e, special_math_ops.bessel_i1e)
    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

    self._compareBothSparse(x, np.abs, math_ops.abs)
    self._compareBothSparse(x, np.negative, math_ops.negative)
    self._compareBothSparse(x, np.square, math_ops.square)
    self._compareBothSparse(z, np.sqrt, math_ops.sqrt, tol=1e-3)
    self._compareBothSparse(x, np.tanh, math_ops.tanh)
    self._compareBothSparse(y, np.sign, math_ops.sign)
    self._compareBothSparse(x, np.vectorize(math.erf), math_ops.erf)

  @test_util.run_deprecated_v1
  def testHalfBasic(self):
    # [FUNCTION DOC]
    # Test: unary ops on a small float16 array and derived arrays (greater than one, nonzero,
    # positive, and within minus one to one). Dense comparisons on CPU and GPU for abs, negation,
    # reciprocal, square, sqrt, rsqrt, exp, expm1, log, log1p, hyperbolic, trigonometric and
    # inverse functions, sigmoid, sign, lgamma (domain errors mapped to infinity), erf and erfc;
    # acosh uses a looser gradient tolerance of 1e-3. Also sparse comparisons for abs, negation,
    # square, sqrt, tanh, sign and erf.
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float16)
    w = x - x.min() + 1.1  # all greater than 1
    y = (x + .5).astype(np.float16)  # no zero
    z = (x + 15.5).astype(np.float16)  # all positive
    k = np.arange(-0.90, 0.90, 0.05).astype(np.float16)  # between -1 and 1
    self._compareBoth(x, np.abs, math_ops.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, math_ops.reciprocal)
    self._compareBoth(x, np.square, math_ops.square)
    self._compareBoth(z, np.sqrt, math_ops.sqrt)
    self._compareBoth(z, self._rsqrt, math_ops.rsqrt)
    self._compareBoth(x, np.exp, math_ops.exp)
    self._compareBoth(x, np.expm1, math_ops.expm1)
    self._compareBoth(z, np.log, math_ops.log)
    self._compareBoth(z, np.log1p, math_ops.log1p)
    self._compareBoth(x, np.sinh, math_ops.sinh)
    self._compareBoth(x, np.cosh, math_ops.cosh)
    self._compareBoth(x, np.tanh, math_ops.tanh)
    self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareBoth(y, np.sign, math_ops.sign)
    self._compareBoth(x, np.sin, math_ops.sin)
    self._compareBoth(x, np.cos, math_ops.cos)
    self._compareBoth(x, np.tan, math_ops.tan)
    self._compareBoth(k, np.arcsin, math_ops.asin)
    self._compareBoth(k, np.arccos, math_ops.acos)
    self._compareBoth(x, np.arctan, math_ops.atan)
    self._compareBoth(x, np.arcsinh, math_ops.asinh)
    # The derivative of acosh close to 1 is very large, and needs a high
    # tolerance for small precision.
    self._compareBoth(w, np.arccosh, math_ops.acosh, grad_tol=1e-3)
    self._compareBoth(k, np.arctanh, math_ops.atanh)
    self._compareBoth(
        y, np.vectorize(self._replace_domain_error_with_inf(math.lgamma)),
        math_ops.lgamma)
    self._compareBoth(x, np.vectorize(math.erf), math_ops.erf)
    self._compareBoth(x, np.vectorize(math.erfc), math_ops.erfc)
    self._compareBothSparse(x, np.abs, math_ops.abs)
    self._compareBothSparse(x, np.negative, math_ops.negative)
    self._compareBothSparse(x, np.square, math_ops.square)
    self._compareBothSparse(z, np.sqrt, math_ops.sqrt, tol=1e-3)
    self._compareBothSparse(x, np.tanh, math_ops.tanh)
    self._compareBothSparse(y, np.sign, math_ops.sign)
    self._compareBothSparse(x, np.vectorize(math.erf), math_ops.erf, tol=1e-3)

  @test_util.run_deprecated_v1
  def testBFloat16Basic(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Square on bfloat16 is compared twice (once CPU-only and again on both
    # devices); the second check subsumes the first.
    # Test: unary ops on a small bfloat16 array and derived arrays (greater than one, nonzero,
    # positive, and within minus one to one). Reference results are computed in float32 and cast
    # back. Checks abs, negation, reciprocal, sign, trigonometric, hyperbolic and inverse
    # functions, erf and erfc on both devices; round, exp, expm1, log, log1p, rsqrt and square on
    # CPU only. atanh uses a looser gradient tolerance of 1e-2.

    def compute_f32(np_func):
      # [FUNCTION DOC]
      # KNOWN ISSUE: The docstring calls this a decorator, but it is only used as an ordinary
      # function wrapper at call sites; it is never applied with decorator syntax.
      # HELPER: Builds reference functions for the bfloat16 test. Takes a NumPy function and
      # returns a new function that evaluates it in 32-bit float precision and casts back to the
      # input dtype, since NumPy lacks accurate bfloat16 math.
      """Decorator to compute Numpy function with float32 math."""

      def f(x):
        # [FUNCTION DOC]
        # HELPER: The wrapper produced by the enclosing helper. Casts x to 32-bit float, applies
        # the captured NumPy function, and casts the result back to the dtype of x. Returns the
        # converted array.
        y = np_func(x.astype(np.float32))
        return y.astype(x.dtype)

      return f

    bfloat16 = dtypes_lib.bfloat16.as_numpy_dtype
    x = np.arange(-6, 6,
                  2).reshape(1, 3, 2).astype(bfloat16)
    w = x - x.min() + 1.1  # all greater than 1
    y = (x + .5).astype(bfloat16)  # no zero
    z = (x + 15.5).astype(bfloat16)  # all positive
    k = np.arange(-0.90, 0.90, 0.05).astype(bfloat16)  # between -1 and 1
    self._compareBoth(x, np.abs, math_ops.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, compute_f32(self._inv), math_ops.reciprocal)
    self._compareCpu(x, np.round, math_ops.round)
    self._compareCpu(x, np.exp, math_ops.exp)
    self._compareCpu(x, np.expm1, math_ops.expm1)
    self._compareCpu(z, compute_f32(np.log), math_ops.log)
    self._compareCpu(z, compute_f32(np.log1p), math_ops.log1p)
    self._compareBoth(y, np.sign, math_ops.sign)
    self._compareCpu(z, self._rsqrt, math_ops.rsqrt)
    self._compareCpu(x, np.square, math_ops.square)
    self._compareBoth(x, compute_f32(np.sin), math_ops.sin)
    self._compareBoth(x, compute_f32(np.cos), math_ops.cos)
    self._compareBoth(x, compute_f32(np.tan), math_ops.tan)
    self._compareBoth(x, compute_f32(np.sinh), math_ops.sinh)
    self._compareBoth(x, compute_f32(np.cosh), math_ops.cosh)
    self._compareBoth(x, compute_f32(np.tanh), math_ops.tanh)
    self._compareBoth(k, compute_f32(np.arcsin), math_ops.asin)
    self._compareBoth(k, compute_f32(np.arccos), math_ops.acos)
    self._compareBoth(x, compute_f32(np.arctan), math_ops.atan)
    self._compareBoth(x, compute_f32(np.arcsinh), math_ops.asinh)
    self._compareBoth(w, compute_f32(np.arccosh), math_ops.acosh)
    self._compareBoth(k, compute_f32(np.arctanh), math_ops.atanh,
                      grad_tol=1e-2)
    self._compareBoth(x, compute_f32(np.vectorize(math.erf)), math_ops.erf)
    self._compareBoth(x, compute_f32(np.vectorize(math.erfc)), math_ops.erfc)
    self._compareBoth(x, compute_f32(np.square), math_ops.square)

  def testInt8Basic(self):
    # [FUNCTION DOC]
    # Test: abs (CPU only), negation and sign (CPU and GPU) on a small signed 8-bit integer
    # array, using both the named ops and the Python operators, compared against NumPy.
    x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int8)
    self._compareCpu(x, np.abs, math_ops.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(x, np.sign, math_ops.sign)

  def testUInt8Basic(self):
    # [FUNCTION DOC]
    # Test: square op on a small unsigned 8-bit integer array, compared against NumPy on both
    # CPU and GPU.
    x = np.arange(6).reshape(1, 3, 2).astype(np.uint8)
    self._compareBoth(x, np.square, math_ops.square)

  def testInt16Basic(self):
    # [FUNCTION DOC]
    # Test: abs (CPU only), negation and sign (CPU and GPU) on a small signed 16-bit integer
    # array, using both the named ops and the Python operators, compared against NumPy.
    x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int16)
    self._compareCpu(x, np.abs, math_ops.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(x, np.sign, math_ops.sign)

  def testUInt16Basic(self):
    # [FUNCTION DOC]
    # Test: square op on a small unsigned 16-bit integer array, compared against NumPy on both
    # CPU and GPU.
    x = np.arange(6).reshape(1, 3, 2).astype(np.uint16)
    self._compareBoth(x, np.square, math_ops.square)

  def testInt32Basic(self):
    # [FUNCTION DOC]
    # Test: unary ops on a small signed 32-bit integer array. abs on CPU only; negation and
    # square on both CPU and GPU; sign on CPU only; and sparse abs, negation, square and sign
    # on CPU and GPU. Results are compared against NumPy.
    x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int32)
    self._compareCpu(x, np.abs, math_ops.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(x, np.square, math_ops.square)
    self._compareCpu(x, np.sign, math_ops.sign)

    self._compareBothSparse(x, np.abs, math_ops.abs)
    self._compareBothSparse(x, np.negative, math_ops.negative)
    self._compareBothSparse(x, np.square, math_ops.square)
    self._compareBothSparse(x, np.sign, math_ops.sign)

  def testUInt32Basic(self):
    # [FUNCTION DOC]
    # Test: square op on a small unsigned 32-bit integer array, compared against NumPy on both
    # CPU and GPU.
    x = np.arange(6).reshape(1, 3, 2).astype(np.uint32)
    self._compareBoth(x, np.square, math_ops.square)

  def testInt64Basic(self):
    # [FUNCTION DOC]
    # Test: abs, negation (both via ops and via Python operators) and sign on a small signed
    # 64-bit integer array with large values, dense on CPU only; plus sparse abs, negation and
    # sign on CPU and GPU. Results are compared against NumPy.
    x = np.arange(-6 << 40, 6 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
    self._compareCpu(x, np.abs, math_ops.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, math_ops.negative)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(x, np.sign, math_ops.sign)

    self._compareBothSparse(x, np.abs, math_ops.abs)
    self._compareBothSparse(x, np.negative, math_ops.negative)
    self._compareBothSparse(x, np.sign, math_ops.sign)

  def testInt64Square(self):
    # [FUNCTION DOC]
    # Test: square op on 64-bit integers of moderate magnitude (to avoid overflow), dense on CPU
    # and sparse on CPU and GPU, compared against NumPy.
    x = np.arange(-6 << 20, 6 << 20, 2 << 20).reshape(1, 3, 2).astype(np.int64)
    self._compareCpu(x, np.square, math_ops.square)
    self._compareBothSparse(x, np.square, math_ops.square)

  def testUInt64Basic(self):
    # [FUNCTION DOC]
    # Test: square op on a small unsigned 64-bit integer array, compared against NumPy on both
    # CPU and GPU.
    x = np.arange(6).reshape(1, 3, 2).astype(np.uint64)
    self._compareBoth(x, np.square, math_ops.square)

  @test_util.run_deprecated_v1
  def testComplex64Basic(self):
    # [FUNCTION DOC]
    # Test: unary ops on a small complex64 array (and a shifted copy with no zeros). Dense CPU
    # and GPU comparisons for abs, negation, reciprocal and exp; CPU-only comparisons (with
    # gradients) for square, sqrt, rsqrt, expm1, log, log1p, hyperbolic functions, asin, atan,
    # sigmoid, sin and cos; asinh and acosh with a loosened gradient relative tolerance of 1e-2
    # because the library has only about six digits of precision; atanh; sparse comparisons for
    # abs, negation, square, sqrt and tanh; and sign against a reference dividing by magnitude.
    x = (1 + 1j) * np.arange(-3, 3).reshape(1, 3, 2).astype(np.complex64)
    y = x + (0.5 + 0.5j)  # no zeros
    self._compareBoth(x, np.abs, math_ops.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, math_ops.reciprocal)
    self._compareCpu(x, np.square, math_ops.square)
    self._compareCpu(y, np.sqrt, math_ops.sqrt)
    self._compareCpu(y, self._rsqrt, math_ops.rsqrt)
    self._compareBoth(x, np.exp, math_ops.exp)
    self._compareCpu(x, np.expm1, math_ops.expm1)
    self._compareCpu(y, np.log, math_ops.log)
    self._compareCpu(y, np.log1p, math_ops.log1p)
    self._compareCpu(x, np.sinh, math_ops.sinh)
    self._compareCpu(x, np.cosh, math_ops.cosh)
    self._compareCpu(x, np.tanh, math_ops.tanh)
    self._compareCpu(x, np.arcsin, math_ops.asin)
    self._compareCpu(x, np.arctan, math_ops.atan)

    # Complex64 versions of asinh() and acosh() in libstdc++ only have 6 digits
    # of precision.
    # Small gradient values + low precision --> High relative error
    self._compareCpu(y, np.arcsinh, math_ops.asinh, grad_rtol=1e-2)
    self._compareCpu(y, np.arccosh, math_ops.acosh, grad_rtol=1e-2)

    self._compareCpu(y, np.arctanh, math_ops.atanh)
    self._compareCpu(x, self._sigmoid, math_ops.sigmoid)
    self._compareCpu(x, np.sin, math_ops.sin)
    self._compareCpu(x, np.cos, math_ops.cos)

    self._compareBothSparse(x, np.abs, math_ops.abs)
    self._compareBothSparse(x, np.negative, math_ops.negative)
    self._compareBothSparse(x, np.square, math_ops.square)
    self._compareBothSparse(x, np.sqrt, math_ops.sqrt, 1e-3)
    self._compareBothSparse(x, np.tanh, math_ops.tanh)

    # Numpy uses an incorrect definition of sign; use the right one instead.
    def complex_sign(x):
      # [FUNCTION DOC]
      # HELPER: Reference implementation of the complex sign (unit-magnitude direction).
      # Takes a complex array x and returns x divided by its elementwise magnitude (undefined,
      # not-a-number, for zero entries, which the test inputs avoid).
      return x / np.abs(x)

    self._compareBoth(y, complex_sign, math_ops.sign)
    self._compareBothSparse(y, complex_sign, math_ops.sign)

  @test_util.run_deprecated_v1
  def testComplex128Basic(self):
    # [FUNCTION DOC]
    # Test: unary ops on a small complex128 array (and a shifted copy with no zeros). Dense CPU
    # and GPU comparisons for abs, negation, reciprocal and exp; CPU-only comparisons (including
    # gradients) for square, sqrt, rsqrt, expm1, log, log1p, hyperbolic and inverse hyperbolic
    # functions, sigmoid, sin, cos, asin and atan; sparse comparisons for abs, negation, square,
    # sqrt (tolerance 1e-3) and tanh; and the sign op against a reference that divides by the
    # magnitude, dense and sparse. Runs graph-mode only (deprecated v1 style).
    x = (1 + 1j) * np.arange(-3, 3).reshape(1, 3, 2).astype(np.complex128)
    y = x + (0.5 + 0.5j)  # no zeros
    self._compareBoth(x, np.abs, math_ops.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, math_ops.negative)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, math_ops.reciprocal)
    self._compareCpu(x, np.square, math_ops.square)
    self._compareCpu(y, np.sqrt, math_ops.sqrt)
    self._compareCpu(y, self._rsqrt, math_ops.rsqrt)
    self._compareBoth(x, np.exp, math_ops.exp)
    self._compareCpu(x, np.expm1, math_ops.expm1)
    self._compareCpu(y, np.log, math_ops.log)
    self._compareCpu(y, np.log1p, math_ops.log1p)
    self._compareCpu(x, np.sinh, math_ops.sinh)
    self._compareCpu(x, np.cosh, math_ops.cosh)
    self._compareCpu(x, np.tanh, math_ops.tanh)
    self._compareCpu(y, np.arcsinh, math_ops.asinh)
    self._compareCpu(y, np.arccosh, math_ops.acosh)
    self._compareCpu(y, np.arctanh, math_ops.atanh)
    self._compareCpu(x, self._sigmoid, math_ops.sigmoid)
    self._compareCpu(x, np.sin, math_ops.sin)
    self._compareCpu(x, np.cos, math_ops.cos)
    self._compareCpu(x, np.arcsin, math_ops.asin)
    self._compareCpu(x, np.arctan, math_ops.atan)

    self._compareBothSparse(x, np.abs, math_ops.abs)
    self._compareBothSparse(x, np.negative, math_ops.negative)
    self._compareBothSparse(x, np.square, math_ops.square)
    self._compareBothSparse(x, np.sqrt, math_ops.sqrt, 1e-3)
    self._compareBothSparse(x, np.tanh, math_ops.tanh)

    # Numpy uses an incorrect definition of sign; use the right one instead.
    def complex_sign(x):
      # [FUNCTION DOC]
      # HELPER: Reference implementation of the complex sign (unit-magnitude direction).
      # Takes a complex array x and returns x divided by its elementwise magnitude (undefined,
      # not-a-number, for zero entries, which the test inputs avoid).
      return x / np.abs(x)

    self._compareBoth(y, complex_sign, math_ops.sign)
    self._compareBothSparse(y, complex_sign, math_ops.sign)

  @test_util.run_deprecated_v1
  def testGradGrad(self):
    # [FUNCTION DOC]
    # Test: gradients of the generated backward ops (reciprocal, reciprocal square root, sigmoid,
    # square root and hyperbolic tangent gradient ops), i.e. second-order derivative checks. The
    # global NumPy random seed is set to 7. For each op, with a valid input range, and each of
    # float32, float64, complex64 and complex128 (with a per-dtype tolerance), two random
    # length-5 constant tensors are fed to the op, and the analytic and numeric Jacobians with
    # respect to both inputs are compared within the tolerance. Side effect: reseeds NumPy.
    np.random.seed(7)
    shape = (5,)
    dtype_tols = [(np.float32, 5e-4), (np.float64, 1e-6), (np.complex64, 5e-4),
                  (np.complex128, 1e-6)]
    op_range = [
        (gen_math_ops.reciprocal_grad, [-2, 2]),
        (gen_math_ops.rsqrt_grad, [0.1, 3]),
        (gen_math_ops.sigmoid_grad, [-2, 2]),
        (gen_math_ops.sqrt_grad, [0.1, 3]),
        (gen_math_ops.tanh_grad, [-2, 2]),
    ]

    def rand(dtype, real_range):
      # [FUNCTION DOC]
      # HELPER: Random input generator for the gradient-of-gradient test.
      # Takes a numeric dtype and a two-element range; returns a random array of the enclosing
      # shape with values uniform in that range, cast to the dtype. For complex dtypes it also
      # adds an imaginary part uniform between -2 and 2. Uses the global NumPy random state.
      x = np.random.uniform(
          real_range[0], real_range[1], size=shape[0]).astype(dtype)
      if dtype in (np.complex64, np.complex128):
        x += 1j * np.random.uniform(-2, 2, size=shape[0]).astype(dtype)
      return x

    for op, real_range in op_range:
      with self.cached_session():
        for dtype, tol in dtype_tols:
          x = constant_op.constant(rand(dtype, real_range))
          y = constant_op.constant(rand(dtype, real_range))
          z = op(x, y)
          grads = gradient_checker.compute_gradient(
              [x, y], [shape, shape],
              z,
              shape,
              x_init_value=[rand(dtype, real_range),
                            rand(dtype, real_range)])
          if isinstance(grads, tuple):
            grads = [grads]
          for analytical, numerical in grads:
            self.assertAllClose(analytical, numerical, rtol=tol, atol=tol)

  @test_util.run_in_graph_and_eager_modes
  def testComplexAbsGradGrad(self):
    # [FUNCTION DOC]
    # Test: second-order gradient of the complex magnitude op. Defines a function of a scalar
    # (magnitude of a complex number built from the cosine of the input and a constant imaginary
    # part one) and a function returning its tape gradient. Numerically checks the gradient of
    # that gradient at input 2.0 using the v2 gradient checker, asserting the maximum error
    # between analytic and numeric Jacobians is below 1e-3. Runs in graph and eager modes.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: Function under test for the complex-abs second-derivative check. Takes a real
      # scalar tensor x, builds a complex number whose real part is the cosine of x and whose
      # imaginary part is the constant one, and returns the magnitude of that complex number.
      real = math_ops.cos(x)
      imag = ops.convert_to_tensor(1.)
      return math_ops.abs(math_ops.complex(real, imag))

    def g(x):
      # [FUNCTION DOC]
      # HELPER: Builds the first derivative of the wrapped function for the gradient-of-gradient
      # check. Takes a tensor x, records the wrapped function f of x on a gradient tape and
      # returns the gradient of its output with respect to x.
      with backprop.GradientTape() as t:
        t.watch(x)
        y = f(x)
      return t.gradient(y, x)

    err = gradient_checker_v2.max_error(
        *gradient_checker_v2.compute_gradient(g, [ops.convert_to_tensor(2.0)]))
    self.assertLess(err, 1e-3)


if __name__ == "__main__":
  test.main()
