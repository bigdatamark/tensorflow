# [MODULE DOC]
# Test and benchmark module for the matrix determinant and log-determinant ops. Compares the
# TensorFlow results with NumPy for 2x2 and 5x5 matrices, multidimensional batches and the real
# and complex single/double types, plus edge cases: empty matrices and batches, overflow to
# infinity, infinite and not-a-number entries, invalid shapes and concurrent execution. A
# benchmark class times the determinant op on CPU and (when available) GPU for various sizes.
# Running the file as a script starts the TensorFlow test runner. No import-time side effects.
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
"""Tests for tensorflow.ops.tf.MatrixDeterminant."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class DeterminantOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the matrix determinant op and the log-determinant op (which returns a sign and the
  # logarithm of the absolute value). Shared helpers compute the reference answer with NumPy (or
  # use a caller-supplied expected value) and compare it with the TensorFlow result, evaluating on
  # the GPU when one is available.

  def _compareDeterminantBase(self, matrix_x, tf_ans, expected=None):
    # [FUNCTION DOC]
    # HELPER: compares a determinant result with the reference.
    # Inputs: matrix_x is the input NumPy array (matrix or batch); tf_ans the determinant tensor;
    # expected an optional reference value overriding the NumPy computation (default none).
    # The reference is ones (of the input type) for zero by zero matrices, the supplied expected
    # value if given, else NumPy's determinant. Evaluates the tensor, asserts its static shape
    # equals the reference shape and values are close with an absolute tolerance of five times
    # ten to the minus five. Returns nothing.
    out = self.evaluate(tf_ans)
    shape = matrix_x.shape
    if shape[-1] == 0 and shape[-2] == 0:
      np_ans = np.ones(shape[:-2]).astype(matrix_x.dtype)
    elif expected is None:
      np_ans = np.array(np.linalg.det(matrix_x)).astype(matrix_x.dtype)
    else:
      np_ans = expected
    self.assertShapeEqual(np_ans, tf_ans)
    self.assertAllClose(np_ans, out, atol=5e-5)

  def _compareLogDeterminantBase(self, matrix_x, tf_ans, expected=None):
    # [FUNCTION DOC]
    # HELPER: compares a log-determinant result with the reference.
    # KNOWN ISSUE: the empty-matrix check here uses "either dimension is zero" whereas the plain
    # determinant helper requires both to be zero, so the two helpers treat zero by N inputs
    # differently.
    # Inputs: matrix_x is the input NumPy array; tf_ans a pair (sign tensor, log of absolute
    # determinant tensor); expected an optional reference determinant overriding NumPy's slogdet
    # (default none), from which the sign and log absolute value are derived. For empty matrices
    # the reference is sign one and log zero. Asserts the static shape of the log tensor, then
    # compares sign times exponential of the log value (so infinities and NaNs compare as whole
    # determinants) with an absolute tolerance of five times ten to the minus five. Returns
    # nothing.
    sign_tf, abs_log_det_tf = tf_ans
    shape = matrix_x.shape
    if shape[-1] == 0 or shape[-2] == 0:
      np_sign, np_ans = (1.0, np.zeros(shape[:-2]).astype(matrix_x.dtype))
    elif expected is None:
      np_sign, np_ans = np.linalg.slogdet(matrix_x)
      np_ans = np_ans.astype(matrix_x.dtype)
    else:
      np_ans = np.log(np.abs(expected))
      np_sign = np.sign(expected)

    self.assertShapeEqual(np_ans, abs_log_det_tf)
    sign_tf_val = self.evaluate(sign_tf)
    abs_log_det_tf_val = self.evaluate(abs_log_det_tf)
    self.assertAllClose(
        sign_tf_val * np.exp(abs_log_det_tf_val),
        np_sign * np.exp(np_ans),
        atol=5e-5)

  def _compareDeterminant(self, matrix_x, expected=None):
    # [FUNCTION DOC]
    # HELPER: entry point used by all determinant tests.
    # Inputs: matrix_x is the input array (matrix or batch); expected an optional reference
    # determinant (default none, meaning NumPy computes it). Inside a GPU-preferring device
    # scope, runs both the determinant op and the raw log-determinant op on the input and checks
    # each against the reference using the two base comparison helpers. Returns nothing.
    with test_util.use_gpu():
      self._compareDeterminantBase(
          matrix_x, linalg_ops.matrix_determinant(matrix_x), expected
      )
      self._compareLogDeterminantBase(
          matrix_x, gen_linalg_ops.log_matrix_determinant(matrix_x), expected
      )

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: single precision real inputs: a regular and an all-zero two by two matrix, a
    # five by five matrix (which forces the LU decomposition path), and a random batch with
    # shape three by four by five of two by two matrices (unseeded NumPy randomness). Each is
    # compared against NumPy through the comparison helper.
    # 2x2 matrices
    self._compareDeterminant(np.array([[2., 3.], [3., 4.]]).astype(np.float32))
    self._compareDeterminant(np.array([[0., 0.], [0., 0.]]).astype(np.float32))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [
            2., 5., 8., 3., 8.
        ], [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.float32))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(np.random.rand(3, 4, 5, 2, 2).astype(np.float32))

  def testBasicDouble(self):
    # [FUNCTION DOC]
    # Scenario: same as the single precision basic test but with double precision inputs.
    # 2x2 matrices
    self._compareDeterminant(np.array([[2., 3.], [3., 4.]]).astype(np.float64))
    self._compareDeterminant(np.array([[0., 0.], [0., 0.]]).astype(np.float64))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [
            2., 5., 8., 3., 8.
        ], [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.float64))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(np.random.rand(3, 4, 5, 2, 2).astype(np.float64))

  def testBasicComplex64(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the random batch is built from real random numbers cast to the complex type,
    # so its imaginary parts are all zero and the batch does not exercise complex arithmetic.
    # Scenario: complex64 inputs: real-valued and zero two by two matrices, one with genuinely
    # complex entries, a five by five matrix, and the random (actually real) batch. Each is
    # compared against NumPy through the comparison helper.
    # 2x2 matrices
    self._compareDeterminant(
        np.array([[2., 3.], [3., 4.]]).astype(np.complex64))
    self._compareDeterminant(
        np.array([[0., 0.], [0., 0.]]).astype(np.complex64))
    self._compareDeterminant(
        np.array([[1. + 1.j, 1. - 1.j], [-1. + 1.j, -1. - 1.j]]).astype(
            np.complex64))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [
            2., 5., 8., 3., 8.
        ], [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.complex64))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(np.random.rand(3, 4, 5, 2, 2).astype(np.complex64))

  def testBasicComplex128(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: as in the complex64 test, the random batch has zero imaginary parts because
    # it is a real random array cast to a complex type.
    # Scenario: same as the complex64 test but with complex128 inputs.
    # 2x2 matrices
    self._compareDeterminant(
        np.array([[2., 3.], [3., 4.]]).astype(np.complex128))
    self._compareDeterminant(
        np.array([[0., 0.], [0., 0.]]).astype(np.complex128))
    self._compareDeterminant(
        np.array([[1. + 1.j, 1. - 1.j], [-1. + 1.j, -1. - 1.j]]).astype(
            np.complex128))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [
            2., 5., 8., 3., 8.
        ], [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.complex128))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(
        np.random.rand(3, 4, 5, 2, 2).astype(np.complex128))

  def testInfiniteDeterminant(self):
    # [FUNCTION DOC]
    # Scenario: a diagonal two by two double precision matrix whose entries are the largest
    # representable value, so the true determinant overflows to infinity. Checks the ops agree
    # with NumPy's overflowed answer through the comparison helper.
    max_double = np.finfo("d").max
    huge_matrix = np.array([[max_double, 0.0], [0.0, max_double]])
    self._compareDeterminant(huge_matrix)

  @test_util.run_v1_only("b/120545219")
  def testNonSquareMatrix(self):
    # [FUNCTION DOC]
    # Scenario (graph mode v1 only, skipped otherwise): passes a two by three matrix to the
    # determinant op and asserts a value error is raised at graph construction.
    # When the determinant of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      linalg_ops.matrix_determinant(
          np.array([[1., 2., 3.], [3., 5., 4.]]).astype(np.float32))

  @test_util.run_v1_only("b/120545219")
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment says the input must be two dimensional, but the op accepts any
    # rank of two or more (batches); the test only shows a rank one vector is rejected.
    # Scenario (graph mode v1 only): passes a one dimensional tensor and asserts a value error.
    # The input to the determinant should be a 2-dimensional tensor.
    tensor1 = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_ops.matrix_determinant(tensor1)

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: an input with zero two by two matrices and an input of two zero by zero
    # matrices. Checks via the comparison helper (the latter's determinant is one for each).
    self._compareDeterminant(np.empty([0, 2, 2]))
    self._compareDeterminant(np.empty([2, 0, 0]))

  @test_util.run_v1_only("b/120545219")
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Scenario (graph mode v1 only): two identical random matrices generated from the same seed
    # have their determinants computed in a single run, allowing concurrent execution. Asserts
    # the two scalar results are exactly equal.
    with self.session():
      matrix1 = random_ops.random_normal([5, 5], seed=42)
      matrix2 = random_ops.random_normal([5, 5], seed=42)
      det1 = linalg_ops.matrix_determinant(matrix1)
      det2 = linalg_ops.matrix_determinant(matrix2)
      det1_val, det2_val = self.evaluate([det1, det2])
      self.assertEqual(det1_val, det2_val)

  def testInfAndNans(self):
    # [FUNCTION DOC]
    # Scenario: single precision matrices containing infinities or not-a-number values, with the
    # expected determinant supplied explicitly: two by two cases (positive infinity, not a number
    # for infinite cancellation and for a NaN entry) and five by five cases (negative infinity
    # for one infinite entry, not a number for a NaN entry). Checked via the comparison helper.
    # 2x2 matrices
    np_inf = np.float64(np.inf)
    np_nan = np.float64(np.nan)
    self._compareDeterminant(
        np.array([[np.inf, 1], [1, 1]]).astype(np.float32), expected=np_inf
    )
    self._compareDeterminant(
        np.array([[np.inf, np.inf], [1, 1]]).astype(np.float32), expected=np_nan
    )
    self._compareDeterminant(
        np.array([[np.inf, -np.inf], [1, 1]]).astype(np.float32),
        expected=np_nan,
    )
    self._compareDeterminant(
        np.array([[np.nan, 1], [1, 1]]).astype(np.float32), expected=np_nan
    )

    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([
            [np.inf, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 9.0, 2.0, 0.0],
            [2.0, 5.0, 8.0, 3.0, 8.0],
            [1.0, 6.0, 7.0, 4.0, 7.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]).astype(np.float32),
        expected=-np_inf,
    )
    self._compareDeterminant(
        np.array([
            [np.nan, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 9.0, 2.0, 0.0],
            [2.0, 5.0, 8.0, 3.0, 8.0],
            [1.0, 6.0, 7.0, 4.0, 7.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]).astype(np.float32),
        expected=np_nan,
    )


class MatrixDeterminantBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark for the matrix determinant op. The class attribute lists the matrix and batch
  # shapes to time; each is run on CPU and, when a GPU is available, on GPU.

  shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1000, 1000),
      (1024, 1024),
      (2048, 2048),
      (513, 4, 4),
      (513, 16, 16),
      (513, 256, 256),
  ]

  def _GenerateMatrix(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds the benchmark input.
    # Input: shape, a tuple whose last two entries are equal (square matrices) with optional
    # leading batch dimensions. Raises an assertion error if the matrices are not square.
    # Returns a graph variable holding a single precision array: identity plus a constant matrix
    # of one over twice the size, tiled across the batch. Creates a variable in the current graph.
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.float32) / (
        2.0 * n) + np.diag(np.ones(n).astype(np.float32))
    return variables.Variable(np.tile(matrix, batch_shape + (1, 1)))

  def benchmarkMatrixDeterminantOp(self):
    # [FUNCTION DOC]
    # Benchmark: for every configured shape, builds a graph on CPU with the generated variable
    # and the determinant op, initializes variables and times it in a session for at least
    # twenty five iterations under a name with device and shape. Repeats on the first GPU when
    # one is available. Reports through the benchmark framework; returns nothing.
    for shape in self.shapes:
      with ops.Graph().as_default(), session.Session(
          config=benchmark.benchmark_config()) as sess, ops.device("/cpu:0"):
        matrix = self._GenerateMatrix(shape)
        d = linalg_ops.matrix_determinant(matrix)
        self.evaluate(variables.global_variables_initializer())
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(
                d,),
            min_iters=25,
            name="matrix_determinant_cpu_{shape}".format(shape=shape))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), session.Session(
            config=benchmark.benchmark_config()) as sess, ops.device("/gpu:0"):
          matrix = self._GenerateMatrix(shape)
          d = linalg_ops.matrix_determinant(matrix)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(
                  d,),
              min_iters=25,
              name="matrix_determinant_gpu_{shape}".format(shape=shape))


if __name__ == "__main__":
  test.main()
