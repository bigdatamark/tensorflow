# [MODULE DOC]
# Test and benchmark module for the Cholesky decomposition op. Contains correctness tests
# (single and batched matrices, real and complex types, empty inputs, invalid shapes,
# non-positive-definite input yielding NaNs, concurrent execution), gradient tests comparing
# analytic and finite difference gradients for small and multi-block matrix sizes, and a
# benchmark class that times the op and alternative composite gradient implementations on CPU
# and (when available) GPU. The module-level gradient functions are helpers for the benchmark
# only. Running the file as a script starts the TensorFlow test runner. No import-time side
# effects.
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
"""Tests for tensorflow.ops.tf.Cholesky."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


# Different gradient implementations for benchmark purposes
def _GradWithInverseL(l, l_inverse, grad):
  # [FUNCTION DOC]
  # HELPER: shared gradient formula for the composite gradient implementations in the benchmark.
  # Inputs: l is a lower triangular Cholesky factor (possibly batched); l_inverse its inverse;
  # grad the incoming gradient with respect to the factor. Computes the gradient with respect to
  # the original matrix: forms the product of the conjugate transpose of l with grad, halves its
  # diagonal, keeps only the lower triangle, sandwiches it between the inverse and its conjugate
  # transpose, symmetrizes by adding the conjugate transpose, and halves. Returns that gradient
  # tensor. Builds graph operations only.
  middle = math_ops.matmul(l, grad, adjoint_a=True)
  middle = array_ops.matrix_set_diag(middle,
                                     0.5 * array_ops.matrix_diag_part(middle))
  middle = array_ops.matrix_band_part(middle, -1, 0)
  grad_a = math_ops.matmul(
      math_ops.matmul(l_inverse, middle, adjoint_a=True), l_inverse)
  grad_a += math_ops.conj(array_ops.matrix_transpose(grad_a))
  return grad_a * 0.5


def TriAngSolveCompositeGrad(l, grad):
  # [FUNCTION DOC]
  # HELPER: benchmark-only alternative Cholesky gradient using two triangular solves and no
  # explicit inverse.
  # Inputs: l is the lower triangular factor (possibly batched); grad is the incoming gradient.
  # Forms the lower-triangular, diagonal-halved product of the factor's conjugate transpose and
  # the gradient, solves a triangular system against the factor, conjugate-transposes, solves
  # again, then symmetrizes (adds the adjoint) and halves. Returns the gradient with respect to
  # the original matrix. Builds graph operations only.
  # Gradient is l^{-H} @ ((l^{H} @ grad) * (tril(ones)-1/2*eye)) @ l^{-1}

  # Compute ((l^{H} @ grad) * (tril(ones)-1/2*eye)) = middle
  middle = math_ops.matmul(l, grad, adjoint_a=True)
  middle = array_ops.matrix_set_diag(middle,
                                     0.5 * array_ops.matrix_diag_part(middle))
  middle = array_ops.matrix_band_part(middle, -1, 0)

  # Compute l^{-H} @ middle = z
  l_inverse_middle = linalg_ops.matrix_triangular_solve(l, middle, adjoint=True)

  # We need to compute z @ l^{-1}. With matrix_triangular_solve we
  # actually compute l^{-H} @ z^{H} = grad. Since we later add grad^{H}
  # we can ommit the conjugate transpose here.
  z_h = math_ops.conj(array_ops.matrix_transpose(l_inverse_middle))
  grad_a = linalg_ops.matrix_triangular_solve(l, z_h, adjoint=True)
  grad_a += linalg.adjoint(grad_a)
  return grad_a * 0.5


def MatrixInverseCompositeGrad(l, grad):
  # [FUNCTION DOC]
  # HELPER: benchmark-only alternative Cholesky gradient that inverts the factor with the general
  # matrix inverse op. Inputs: l the lower triangular factor, grad the incoming gradient.
  # Returns the gradient computed by the shared inverse-based formula.
  l_inverse = linalg_ops.matrix_inverse(l)
  return _GradWithInverseL(l, l_inverse, grad)


def TriAngInvCompositeGrad(l, grad):
  # [FUNCTION DOC]
  # HELPER: benchmark-only alternative Cholesky gradient that inverts the factor by solving a
  # triangular system against an identity matrix (built with the factor's size, batch shape and
  # dtype). Inputs: l the lower triangular factor, grad the incoming gradient. Returns the
  # gradient computed by the shared inverse-based formula.
  num_rows = array_ops.shape(l)[-1]
  batch_shape = array_ops.shape(l)[:-2]
  l_inverse = linalg_ops.matrix_triangular_solve(l,
                                                 linalg_ops.eye(
                                                     num_rows,
                                                     batch_shape=batch_shape,
                                                     dtype=l.dtype))
  return _GradWithInverseL(l, l_inverse, grad)


class CholeskyOpTest(test.TestCase):
  # [CLASS DOC]
  # Correctness tests for the Cholesky op on small hand-built and random positive definite
  # matrices, batches, complex types, empty inputs and invalid inputs. Shared verification
  # helpers check the reconstruction and the triangular structure of the result.

  def _verifyCholeskyBase(self, x, chol, verification):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the Cholesky tests.
    # Inputs: x is the original input array; chol the Cholesky result tensor; verification the
    # tensor product of the factor with its conjugate transpose. Evaluates both tensors, asserts
    # the product is close to x, the factor's static shape equals x's, and, if the matrices are
    # non-empty, that every matrix in the batch is lower triangular with a strictly positive
    # diagonal. Returns nothing.
    chol_np, verification_np = self.evaluate([chol, verification])
    self.assertAllClose(x, verification_np)
    self.assertShapeEqual(x, chol)
    # Check that the cholesky is lower triangular, and has positive diagonal
    # elements.
    if chol_np.shape[-1] > 0:
      chol_reshaped = np.reshape(chol_np, (-1, chol_np.shape[-2],
                                           chol_np.shape[-1]))
      for chol_matrix in chol_reshaped:
        self.assertAllClose(chol_matrix, np.tril(chol_matrix))
        self.assertTrue((np.diag(chol_matrix) > 0.0).all())

  def _verifyCholesky(self, x):
    # [FUNCTION DOC]
    # HELPER: end to end check used by most Cholesky tests.
    # Input: x, a positive definite matrix or batch (array). Computes its Cholesky factor,
    # multiplies the factor by its own conjugate transpose (with reduced-precision matmul modes
    # disabled) and delegates to the base verification helper, which asserts reconstruction and
    # triangular structure. Returns nothing.
    # Verify that LL^T == x.
    chol = linalg_ops.cholesky(x)
    verification = test_util.matmul_without_tf32(chol, chol, adjoint_b=True)
    self._verifyCholeskyBase(x, chol, verification)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: a fixed three by three symmetric positive definite matrix in single and double
    # precision, and a Hermitian complex matrix built from it (imaginary off-diagonal parts) in
    # both complex precisions. Each case runs as a sub-test through the verification helper,
    # asserting reconstruction and lower triangular positive diagonal factors.
    data = np.array([[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]])
    for dtype in (np.float32, np.float64):
      with self.subTest(dtype=dtype):
        self._verifyCholesky(data.astype(dtype))
    for dtype in (np.complex64, np.complex128):
      with self.subTest(dtype=dtype):
        complex_data = np.tril(1j * data, -1).astype(dtype)
        complex_data += np.triu(-1j * data, 1).astype(dtype)
        complex_data += data
        self._verifyCholesky(complex_data)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testBatch(self):
    # [FUNCTION DOC]
    # Scenario: batches of matrices. Verifies a batch of one diagonal matrix, a stacked pair of
    # them, a stacked pair of three by three matrices, ten random real positive definite matrices
    # (each made by multiplying a random matrix's transpose with itself) and ten random complex
    # positive definite matrices (conjugate transpose product). Uses unseeded NumPy randomness.
    # Each is checked through the verification helper (reconstruction and triangular structure).
    simple_array = np.array([[[1., 0.], [0., 5.]]])  # shape (1, 2, 2)
    self._verifyCholesky(simple_array)
    self._verifyCholesky(np.vstack((simple_array, simple_array)))
    odd_sized_array = np.array([[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]])
    self._verifyCholesky(np.vstack((odd_sized_array, odd_sized_array)))

    # Generate random positive-definite matrices.
    matrices = np.random.rand(10, 5, 5)
    for i in range(10):
      with self.subTest(i=i):
        matrices[i] = np.dot(matrices[i].T, matrices[i])
    self._verifyCholesky(matrices)

    # Generate random complex valued positive-definite matrices.
    matrices = np.random.rand(10, 5, 5) + 1j * np.random.rand(10, 5, 5)
    for i in range(10):
      with self.subTest(i=i):
        matrices[i] = np.dot(matrices[i].T.conj(), matrices[i])
    self._verifyCholesky(matrices)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNonSquareMatrix(self):
    # [FUNCTION DOC]
    # Scenario: a two by three matrix and a batch of two such matrices are passed to the op.
    # Asserts a value error or an invalid-argument error is raised (the type depends on whether
    # the check happens in shape inference or at run time).
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      linalg_ops.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]]))
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      linalg_ops.cholesky(
          np.array([[[1., 2., 3.], [3., 4., 5.]], [[1., 2., 3.], [3., 4., 5.]]
                   ]))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variable is named as if it were rank three but holds a rank one vector,
    # and the same call is asserted twice in two identical blocks, so the second block adds no
    # coverage.
    # Scenario: passes a one dimensional tensor to the op and asserts a value error or an
    # invalid-argument error is raised because a matrix (rank two or more) is required.
    tensor3 = constant_op.constant([1., 2.])
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      linalg_ops.cholesky(tensor3)
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      linalg_ops.cholesky(tensor3)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNotInvertibleCpu(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says CPU only, but the decorator allows running on GPU as well.
    # Scenario: a symmetric matrix that is not positive definite is factored. Selects the lower
    # triangular elements of the result with a mask and asserts they are all not-a-number values
    # (the op signals failure with NaNs instead of raising).
    # Non-invertible inputs result in lower-triangular NaNs.
    x = constant_op.constant([[1., -1., 0.], [-1., 1., -1.], [0., -1., 1.]])
    chol = linalg_ops.cholesky(x)
    # Extract the lower-triangular elements.
    lower_mask = array_ops.matrix_band_part(
        constant_op.constant(True, shape=x.shape), -1, 0)
    chol_lower = array_ops.boolean_mask(chol, lower_mask)
    # Assert all NaN.
    all_nan = self.evaluate(
        math_ops.reduce_all(math_ops.reduce_all(math_ops.is_nan(chol_lower))))
    self.assertTrue(all_nan)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: inputs with zero batch entries and with zero by zero matrices. Verified through
    # the helper (shapes preserved, nothing to check for contents).
    self._verifyCholesky(np.empty([0, 2, 2]))
    self._verifyCholesky(np.empty([2, 0, 0]))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Scenario: two independent Cholesky ops on matrices generated from the same fixed seed (so
    # they are identical) are evaluated in the same run, allowing the runtime to execute them
    # concurrently. Asserts both results are close to each other.
    seed = [42, 24]
    matrix_shape = [5, 5]
    matrix1 = stateless_random_ops.stateless_random_normal(matrix_shape, seed)
    matrix2 = stateless_random_ops.stateless_random_normal(matrix_shape, seed)
    matrix1 = math_ops.matmul(matrix1, matrix1, adjoint_a=True)
    matrix2 = math_ops.matmul(matrix2, matrix2, adjoint_a=True)
    c1 = linalg_ops.cholesky(matrix1)
    c2 = linalg_ops.cholesky(matrix2)
    c1_val, c2_val = self.evaluate([c1, c2])
    self.assertAllClose(c1_val, c2_val)


class CholeskyGradTest(test.TestCase):
  # [CLASS DOC]
  # Gradient tests for the Cholesky op: compares theoretical and numerical (finite difference)
  # Jacobians for real and complex inputs, batched and unbatched, on small matrices and on
  # matrices just above one and two multiples of the op's blocked back-propagation size (a class
  # constant of sixteen).
  _backprop_block_size = 16

  def getShapes(self, shapeList):
    # [FUNCTION DOC]
    # HELPER: builds test shapes. Input: shapeList, a list of row counts. Returns a lazy
    # sequence of pairs (rows, columns) where columns is one and a fifth times the rows, rounded
    # down, so the random input has more columns than rows. No side effects.
    return ((elem, int(np.floor(1.2 * elem))) for elem in shapeList)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testSmallMatrices(self):
    # [FUNCTION DOC]
    # Scenario: seeds NumPy's generator with zero and checks the gradient on real single and
    # double precision inputs with one, two and ten rows (and a fifth more columns), both
    # batched and unbatched, via the finite difference runner. Asserts theory and numeric
    # gradients agree within tolerance.
    np.random.seed(0)
    shapes = self.getShapes([1, 2, 10])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float32, dtypes_lib.float64))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testSmallMatricesComplex(self):
    # [FUNCTION DOC]
    # Scenario: same as the small real test but for complex64 and complex128 inputs.
    # Seeds NumPy's generator with zero; asserts gradient agreement through the runner.
    np.random.seed(0)
    shapes = self.getShapes([1, 2, 10])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex64, dtypes_lib.complex128))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testOneBlockMatrices(self):
    # [FUNCTION DOC]
    # Scenario: a matrix with one more row than the blocked back-propagation size, real single and
    # double precision, using a scalar-reduced output to keep the test fast. Seeds NumPy's
    # generator with zero; asserts gradient agreement.
    np.random.seed(0)
    shapes = self.getShapes([self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes,
        dtypes=(dtypes_lib.float32, dtypes_lib.float64),
        scalar_test=True)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testTwoBlockMatrixFloat(self):
    # [FUNCTION DOC]
    # Scenario: a matrix just over two blocks in size, single precision real, scalar-reduced
    # output. Seeds NumPy's generator with zero; asserts gradient agreement.
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float32,), scalar_test=True)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testTwoBlockMatrixDouble(self):
    # [FUNCTION DOC]
    # Scenario: as the two-block float test but in double precision real.
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float64,), scalar_test=True)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testTwoBlockMatrixComplexFloat(self):
    # [FUNCTION DOC]
    # Scenario: as the two-block float test but with single precision complex (complex64) input.
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex64,), scalar_test=True)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testTwoBlockMatrixComplexDouble(self):
    # [FUNCTION DOC]
    # Scenario: as the two-block float test but with double precision complex (complex128) input.
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex128,), scalar_test=True)

  def _runOneTest(self, shape, dtype, batch, scalar_test):
    # [FUNCTION DOC]
    # HELPER: single gradient check used by the finite difference runner.
    # Inputs: shape is a pair (rows, columns) of the random input; dtype a TensorFlow data type
    # (real or complex); batch a boolean choosing to tile the matrix into a batch of two;
    # scalar_test a boolean reducing the output to its mean to speed up the check. Chooses a
    # tolerance by precision (tight for double, slightly looser for complex double, loose for
    # others) and a finite difference step of the cube root of the machine epsilon. Builds a
    # function that forms the Hermitian matrix x times its conjugate transpose divided by the row
    # count, optionally tiles it, takes the Cholesky factor and optionally reduces to a mean, then
    # compares theoretical and numerical gradients with that tolerance. Uses unseeded-per-call
    # NumPy randomness (the callers seed it). Returns nothing; fails on mismatch.
    if dtype == dtypes_lib.float64:
      tol = 1e-5
    elif dtype == dtypes_lib.complex128:
      tol = 5e-5
    else:
      tol = 5e-3
    epsilon = np.finfo(dtype.as_numpy_dtype).eps
    delta = epsilon**(1.0 / 3.0)

    def RandomInput():
      # [FUNCTION DOC]
      # HELPER: generates the random test input for the gradient check. Takes no arguments; uses
      # the enclosing shape and dtype. Returns a NumPy array of standard normal values in that
      # type, with an added independent imaginary part when the type is complex.
      a = np.random.randn(shape[0], shape[1]).astype(dtype.as_numpy_dtype)
      if dtype.is_complex:
        a += 1j * np.random.randn(shape[0], shape[1]).astype(
            dtype.as_numpy_dtype)
      return a

    def Compute(x):
      # [FUNCTION DOC]
      # HELPER: function whose gradient is checked. Input: x, a random matrix tensor. Forms the
      # Hermitian matrix x times its conjugate transpose divided by the row count, tiles it to a
      # batch of two when batching was requested, applies the Cholesky op, and, for scalar
      # tests, reduces the factor to its mean. Returns the resulting tensor.
      # Turn the random matrix x into a Hermitian matrix by
      # computing the quadratic form x * x^H.
      a = test_util.matmul_without_tf32(
          x, math_ops.conj(array_ops.matrix_transpose(x))) / shape[0]
      if batch:
        a = array_ops.tile(array_ops.expand_dims(a, 0), [2, 1, 1])
      # Finally take the cholesky decomposition of the Hermitian matrix.
      c = linalg_ops.cholesky(a)
      if scalar_test:
        # Reduce to a single scalar output to speed up test.
        c = math_ops.reduce_mean(c)
      return c

    theoretical, numerical = gradient_checker_v2.compute_gradient(
        Compute, [RandomInput()], delta=delta)
    self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  def runFiniteDifferences(self,
                           shapes,
                           dtypes=(dtypes_lib.float32, dtypes_lib.float64,
                                   dtypes_lib.complex64, dtypes_lib.complex128),
                           scalar_test=False):
    # [FUNCTION DOC]
    # HELPER: runs the gradient check over a grid of cases.
    # Inputs: shapes is an iterable of (rows, columns) pairs; dtypes a tuple of TensorFlow types
    # (default float32, float64, complex64 and complex128); scalar_test a boolean (default false)
    # that reduces outputs to a scalar. For every shape, type and both batched and unbatched
    # variants it runs a single gradient check. Returns nothing.
    for shape_ in shapes:
      for dtype_ in dtypes:
        for batch_ in False, True:
          self._runOneTest(shape_, dtype_, batch_, scalar_test)


class CholeskyBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmarks for the Cholesky op and for alternative composite implementations of its gradient.
  # The class attribute lists the matrix and batch shapes to measure. Benchmarks run on CPU and,
  # if a GPU is available, on GPU, and report through the benchmark framework.

  shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1000, 1000),
      (1024, 1024),
      (2048, 2048),
      (513, 2, 2),
      (513, 8, 8),
      (513, 256, 256),
      (4, 513, 2, 2),
  ]

  def _GenerateMatrix(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds a well-conditioned positive definite input for the benchmarks.
    # Input: shape, a tuple whose last two entries are equal (square matrices) and whose leading
    # entries are batch dimensions. Raises an assertion error if the matrices are not square.
    # Returns a single precision array: the identity plus a constant matrix of one over twice the
    # size, tiled across the batch dimensions.
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.float32) / (
        2.0 * n) + np.diag(np.ones(n).astype(np.float32))
    return np.tile(matrix, batch_shape + (1, 1))

  def benchmarkCholeskyOp(self):
    # [FUNCTION DOC]
    # Benchmark: for each configured shape, builds a graph with a variable holding the generated
    # matrix and the Cholesky op pinned to CPU, initializes variables and times it in a session
    # with at least twenty five iterations, named by device and shape. If a GPU is available,
    # repeats on the first GPU. Side effects: creates graphs and sessions and reports benchmark
    # results. Returns nothing.
    for shape in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix = variables.Variable(self._GenerateMatrix(shape))
        l = linalg_ops.cholesky(matrix)
        self.evaluate(variables.global_variables_initializer())
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(
                l,),
            min_iters=25,
            name="cholesky_cpu_{shape}".format(shape=shape))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/device:GPU:0"):
          matrix = variables.Variable(self._GenerateMatrix(shape))
          l = linalg_ops.cholesky(matrix)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(
                  l,),
              min_iters=25,
              name="cholesky_gpu_{shape}".format(shape=shape))

  def benchmarkGradVariants(self):
    # [FUNCTION DOC]
    # Benchmark: compares three composite gradient implementations (general matrix inverse,
    # triangular solve against the identity, and two triangular solves) over all configured
    # shapes, on the first GPU when one is available and always on CPU. Reports timings named
    # with the implementation, device and shape. Returns nothing.

    def _BenchmarkGrad(grad_fn, name, device):
      # [FUNCTION DOC]
      # HELPER: times one gradient implementation on one device.
      # Inputs: grad_fn is a function taking the factor and the incoming gradient and returning
      # the gradient tensor; name is a label for the report; device is a device name string.
      # For every shape, builds a graph on that device with a variable holding a NumPy Cholesky
      # factor of the generated matrix and a variable of random gradient values, applies the
      # function, initializes variables and runs the benchmark with at least twenty five
      # iterations. Returns nothing; reports results.
      for shape in self.shapes:
        matrix = self._GenerateMatrix(shape)
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device(device):
          l = variables.Variable(np.linalg.cholesky(matrix))
          grad_matrix = variables.Variable(
              np.random.randn(*matrix.shape).astype(np.float32))
          grad = grad_fn(l, grad_matrix)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(
                  grad,),
              min_iters=25,
              name="{name}_{dev}_{shape}".format(
                  name=name, dev=grad.device, shape=shape))

    if test.is_gpu_available(True):
      _BenchmarkGrad(MatrixInverseCompositeGrad, "composite_matrix_inverse",
                     "/device:GPU:0")
      _BenchmarkGrad(TriAngInvCompositeGrad, "composite_tri_ang_inverse",
                     "/device:GPU:0")
      _BenchmarkGrad(TriAngSolveCompositeGrad, "composite_triangular_solve",
                     "/device:GPU:0")

    _BenchmarkGrad(MatrixInverseCompositeGrad, "composite_matrix_inverse",
                   "/cpu:0")
    _BenchmarkGrad(TriAngInvCompositeGrad, "composite_tri_ang_inverse",
                   "/cpu:0")
    _BenchmarkGrad(TriAngSolveCompositeGrad, "composite_triangular_solve",
                   "/cpu:0")


if __name__ == "__main__":
  test.main()
