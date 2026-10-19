# [MODULE DOC]
# Tests and a benchmark for the matrix exponential function in the linear algebra implementation
# module. Compares results against a slow Taylor-series reference implemented with numpy, for real
# and complex types, batches, empty inputs, error cases (non-square, too few dimensions), infinite
# input, dynamic shapes and repeated execution. At script run time it also generates many extra
# test methods on the test class (random matrices and matrices scaled to chosen L1 norms) before
# starting the test runner. The benchmark times the op over a range of sizes on CPU and GPU.
#
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
"""Tests for tensorflow.ops.linalg.linalg_impl.matrix_exponential."""

import itertools

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def np_expm(x):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: numpy reference implementation of the matrix exponential for the tests.
  # Input: x, a square numpy matrix (any float or complex dtype). Returns a matrix of the same
  # shape and dtype computed by summing the first 40 terms of the Taylor series (power of x
  # divided by factorial). Accurate only for matrices of modest norm; no side effects.
  """Slow but accurate Taylor series matrix exponential."""
  y = np.zeros(x.shape, dtype=x.dtype)
  xn = np.eye(x.shape[0], dtype=x.dtype)
  for n in range(40):
    if n > 0:
      xn /= float(n)
    y += xn
    xn = np.dot(xn, x)
  return y


@test_util.run_all_without_tensor_float_32("Avoid TF32-based matmuls.")
class ExponentialOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the matrix exponential, with TF32 matmuls avoided for the whole class. Contains
  # verification helpers and tests; more test methods are attached dynamically when the module
  # is run as a script.

  def _verifyExponential(self, x, np_type):
    # [FUNCTION DOC]
    # HELPER: checks the TensorFlow matrix exponential against the numpy reference. Inputs: x
    # (numpy array of matrices, possibly batched or empty) and np_type (dtype to cast to).
    # Computes the exponential on GPU if available, builds the reference per batch element (or an
    # empty array for empty input) and asserts closeness with 1e-3 relative and absolute
    # tolerance. Returns nothing.
    inp = x.astype(np_type)
    with test_util.use_gpu():
      tf_ans = linalg_impl.matrix_exponential(inp)
      if x.size == 0:
        np_ans = np.empty(x.shape, dtype=np_type)
      else:
        if x.ndim > 2:
          np_ans = np.zeros(inp.shape, dtype=np_type)
          for i in itertools.product(*[range(x) for x in inp.shape[:-2]]):
            np_ans[i] = np_expm(inp[i])
        else:
          np_ans = np_expm(inp)
      out = self.evaluate(tf_ans)
      self.assertAllClose(np_ans, out, rtol=1e-3, atol=1e-3)

  def _verifyExponentialReal(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the exponential verification on x for 32-bit and 64-bit real dtypes. Note that
    # casting complex input to a real dtype would discard the imaginary part.
    for np_type in [np.float32, np.float64]:
      self._verifyExponential(x, np_type)

  def _verifyExponentialComplex(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the exponential verification on x for the two complex dtypes (64-bit and
    # 128-bit complex).
    for np_type in [np.complex64, np.complex128]:
      self._verifyExponential(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    # [FUNCTION DOC]
    # HELPER: builds a four-dimensional batch from two matrices. Stacks them into a pair and tiles
    # the pair to batch shape 2 by 6 (outer dimension 2, second dimension 3 times 2). Returns the
    # numpy array.
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  def testNonsymmetricReal(self):
    # [FUNCTION DOC]
    # Test: exponentials of two nonsymmetric real 2x2 matrices and of a batch built from them
    # match the numpy reference for float32 and float64.
    # 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    self._verifyExponentialReal(matrix1)
    self._verifyExponentialReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyExponentialReal(self._makeBatch(matrix1, matrix2))

  @test_util.run_deprecated_v1
  def testNonsymmetricComplex(self):
    # [FUNCTION DOC]
    # Test: complex versions (real part plus the same as imaginary part) of the nonsymmetric 2x2
    # matrices, and a batch of them, match the numpy reference for both complex dtypes.
    # Runs in deprecated graph mode.
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyExponentialComplex(matrix1)
    self._verifyExponentialComplex(matrix2)
    # Complex batch
    self._verifyExponentialComplex(self._makeBatch(matrix1, matrix2))

  def testSymmetricPositiveDefiniteReal(self):
    # [FUNCTION DOC]
    # Test: exponentials of two symmetric positive definite real 2x2 matrices and a batch of them
    # match the numpy reference for float32 and float64.
    # 2x2 matrices
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._verifyExponentialReal(matrix1)
    self._verifyExponentialReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyExponentialReal(self._makeBatch(matrix1, matrix2))

  def testSymmetricPositiveDefiniteComplex(self):
    # [FUNCTION DOC]
    # Test: complex versions of two symmetric positive definite 2x2 matrices, and a batch of
    # them, match the numpy reference for both complex dtypes.
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyExponentialComplex(matrix1)
    self._verifyExponentialComplex(matrix2)
    # Complex batch
    self._verifyExponentialComplex(self._makeBatch(matrix1, matrix2))

  @test_util.run_deprecated_v1
  def testNonSquareMatrix(self):
    # [FUNCTION DOC]
    # Test: computing the exponential of a 2x3 matrix raises a value error. Deprecated graph mode.
    # When the exponential of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      linalg_impl.matrix_exponential(np.array([[1., 2., 3.], [3., 4., 5.]]))

  @test_util.run_deprecated_v1
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variable is named tensor3 but holds a one-dimensional tensor.
    # Test: a one-dimensional tensor input to the exponential raises a value error.
    # Deprecated graph mode.
    # The input to the exponential should be at least a 2-dimensional tensor.
    tensor3 = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_impl.matrix_exponential(tensor3)

  def testInfinite(self):
    # [FUNCTION DOC]
    # Test: a matrix containing infinity yields a result in which every entry is not-a-number,
    # and the op terminates rather than looping forever (regression test for a bug report).
    # Check that the op does not loop forever on infinite inputs. (b/158433036)
    in_tensor = [[np.inf, 1.], [1., 1.]]
    result = self.evaluate(linalg_impl.matrix_exponential(in_tensor))
    self.assertTrue(np.all(np.isnan(result)))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: empty inputs (zero-size batch of 2x2, and batch of two 0x0 matrices) produce results
    # matching the empty reference for real dtypes.
    self._verifyExponentialReal(np.empty([0, 2, 2]))
    self._verifyExponentialReal(np.empty([2, 0, 0]))

  @test_util.run_deprecated_v1
  def testDynamic(self):
    # [FUNCTION DOC]
    # Test: the exponential accepts a placeholder of unknown shape; runs it with a fed 2x2 matrix
    # and checks only that execution succeeds (no numeric assertion). Deprecated graph mode.
    with self.session() as sess:
      inp = array_ops.placeholder(ops.dtypes.float32)
      expm = linalg_impl.matrix_exponential(inp)
      matrix = np.array([[1., 2.], [3., 4.]])
      sess.run(expm, feed_dict={inp: matrix})

  @test_util.run_deprecated_v1
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Test: two random 5x5 matrices generated with the same seed have their exponentials computed
    # in one evaluation; asserts both results are identical. Deprecated graph mode.
    with self.session():
      matrix1 = random_ops.random_normal([5, 5], seed=42)
      matrix2 = random_ops.random_normal([5, 5], seed=42)
      expm1 = linalg_impl.matrix_exponential(matrix1)
      expm2 = linalg_impl.matrix_exponential(matrix2)
      expm = self.evaluate([expm1, expm2])
      self.assertAllEqual(expm[0], expm[1])


class MatrixExponentialBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark for the matrix exponential. The class attribute shapes lists the square matrix
  # shapes (with optional batch dimension) to time.

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
    # HELPER: builds benchmark input. Input: shape tuple with equal last two entries and optional
    # leading batch entries. Returns a variable holding float32 matrices equal to identity plus
    # a small constant (one over twice the size), tiled over the batch. Raises an assertion error
    # for non-square shapes.
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.float32) / (2.0 * n) + np.diag(
        np.ones(n).astype(np.float32))
    return variables.Variable(np.tile(matrix, batch_shape + (1, 1)))

  def benchmarkMatrixExponentialOp(self):
    # [FUNCTION DOC]
    # Benchmark: for each shape, in a fresh graph and session on CPU, builds the matrix variable,
    # initializes variables, and times the exponential for at least 25 iterations; repeats on the
    # GPU when one is available. Side effects: creates graphs and sessions and emits reports.
    for shape in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix = self._GenerateMatrix(shape)
        expm = linalg_impl.matrix_exponential(matrix)
        self.evaluate(variables.global_variables_initializer())
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(expm),
            min_iters=25,
            name="matrix_exponential_cpu_{shape}".format(shape=shape))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/gpu:0"):
          matrix = self._GenerateMatrix(shape)
          expm = linalg_impl.matrix_exponential(matrix)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(expm),
              min_iters=25,
              name="matrix_exponential_gpu_{shape}".format(shape=shape))


def _TestRandomSmall(dtype, batch_dims, size):
  # [FUNCTION DOC]
  # HELPER: factory producing a test method for the exponential test class.
  # KNOWN ISSUE: the generated method is registered under names starting with testL1Norms even
  # though it tests random matrices, and it verifies through the real-only helper, so complex
  # dtypes are cast to real types (imaginary part dropped).
  # Inputs: dtype (numpy element type used to draw the matrices), batch_dims (tuple of leading
  # batch sizes) and size (matrix dimension). Returns a function taking the test instance.

  def Test(self):
    # [FUNCTION DOC]
    # HELPER: generated test body. Seeds the global numpy random state with 42, draws a uniform
    # random matrix batch in the range minus one to one with the chosen shape and dtype, and
    # verifies the exponential for real dtypes against the reference.
    np.random.seed(42)
    shape = batch_dims + (size, size)
    matrix = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
    self._verifyExponentialReal(matrix)

  return Test


def _TestL1Norms(dtype, shape, scale):
  # [FUNCTION DOC]
  # HELPER: factory producing a test method that checks the exponential on a matrix scaled to a
  # given L1 norm (to exercise different algorithm branches). Inputs: dtype (numpy element type),
  # shape (tuple of the full array shape), scale (target L1 norm multiplier). Returns a function
  # taking the test instance.

  def Test(self):
    # [FUNCTION DOC]
    # HELPER: generated test body.
    # KNOWN ISSUE: for complex dtypes the real-only verification drops the imaginary part, and
    # for batched shapes the norm is taken over the whole batch, not per matrix.
    # Seeds the global numpy random state with 42, draws a uniform
    # random array of the shape, divides it by its maximum column absolute sum (L1 norm) so the
    # norm is one, multiplies by the scale and verifies the exponential for real dtypes.
    np.random.seed(42)
    matrix = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype)
    l1_norm = np.max(np.sum(np.abs(matrix), axis=matrix.ndim - 2))
    matrix /= l1_norm
    self._verifyExponentialReal(scale * matrix)

  return Test


if __name__ == "__main__":
  for dtype_ in [np.float32, np.float64, np.complex64, np.complex128]:
    for batch_ in [(), (2,), (2, 2)]:
      for size_ in [4, 7]:
        name = "%s_%d_%d" % (dtype_.__name__, len(batch_), size_)
        setattr(ExponentialOpTest, "testL1Norms_" + name,
                _TestRandomSmall(dtype_, batch_, size_))

  for shape_ in [(3, 3), (2, 3, 3)]:
    for dtype_ in [np.float32, np.complex64]:
      for scale_ in [0.1, 1.5, 5.0, 20.0]:
        name = "%s_%d_%d" % (dtype_.__name__, len(shape_), int(scale_ * 10))
        setattr(ExponentialOpTest, "testL1Norms_" + name,
                _TestL1Norms(dtype_, shape_, scale_))
    for dtype_ in [np.float64, np.complex128]:
      for scale_ in [0.01, 0.2, 0.5, 1.5, 6.0, 25.0]:
        name = "%s_%d_%d" % (dtype_.__name__, len(shape_), int(scale_ * 100))
        setattr(ExponentialOpTest, "testL1Norms_" + name,
                _TestL1Norms(dtype_, shape_, scale_))
  test.main()
