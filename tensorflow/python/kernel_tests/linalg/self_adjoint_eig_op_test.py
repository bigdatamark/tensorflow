# [MODULE DOC]
# Unit tests for the self-adjoint (Hermitian) eigendecomposition ops linalg_ops.self_adjoint_eig
# and self_adjoint_eigvals. Contains a test-registration helper, a class with static tests (rank
# validation, concurrency, and a denormal-sensitive regression matrix read from a test data
# file), helpers to sort eigenpairs and align eigenvector phases, a factory generating value
# tests and a factory generating gradient tests for many dtype, shape and compute-vectors
# combinations. The script entry point registers the generated tests and starts the test runner.
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
"""Tests for tensorflow.ops.linalg_ops.self_adjoint_eig."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


def _AddTest(test_class, op_name, testcase_name, fn):
  # [FUNCTION DOC]
  # HELPER: registers a dynamically generated test method on a test class. Inputs: the test class,
  # an operation name, a test case name and the test function. The attribute name joins the word
  # test, the operation name and the case name. Raises a runtime error if a method with that name
  # already exists; otherwise sets the attribute on the class (mutating it) and returns nothing.
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)


@test_util.run_all_without_tensor_float_32
class SelfAdjointEigTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the self-adjoint eigendecomposition ops, run with TF32 disabled. Holds static
  # tests; value tests are attached dynamically by the script entry point.

  @test_util.run_deprecated_v1
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # Scenario: the eigendecomposition op is applied to a scalar and to a rank-one vector.
    # Expects a value error each time because rank at least two is required. Runs under TF1
    # behavior.
    # The input to self_adjoint_eig should be a tensor of
    # at least rank 2.
    scalar = constant_op.constant(1.)
    with self.assertRaises(ValueError):
      linalg_ops.self_adjoint_eig(scalar)
    vector = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_ops.self_adjoint_eig(vector)

  @test_util.run_deprecated_v1
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Scenario: pairs of identically seeded 5x5 random matrices are decomposed, once with
    # eigenvectors (eigenvalues and vectors) and once eigenvalues only. All results are evaluated
    # together. Expects identical results within each pair, with only approximate equality between
    # the eigenvalues with and without vectors since the algorithms differ. Runs under TF1
    # behavior.
    all_ops = []
    with self.session():
      for compute_v_ in True, False:
        matrix1 = random_ops.random_normal([5, 5], seed=42)
        matrix2 = random_ops.random_normal([5, 5], seed=42)
        if compute_v_:
          e1, v1 = linalg_ops.self_adjoint_eig(matrix1)
          e2, v2 = linalg_ops.self_adjoint_eig(matrix2)
          all_ops += [e1, v1, e2, v2]
        else:
          e1 = linalg_ops.self_adjoint_eigvals(matrix1)
          e2 = linalg_ops.self_adjoint_eigvals(matrix2)
          all_ops += [e1, e2]
      val = self.evaluate(all_ops)
      self.assertAllEqual(val[0], val[2])
      # The algorithm is slightly different for compute_v being True and False,
      # so require approximate equality only here.
      self.assertAllClose(val[2], val[4])
      self.assertAllEqual(val[4], val[5])
      self.assertAllEqual(val[1], val[3])

  def testMatrixThatFailsWhenFlushingDenormsToZero(self):
    # [FUNCTION DOC]
    # Scenario: loads a 32x32 float matrix from a text data file in the test source directory,
    # known to produce wrong results if denormal floats are flushed to zero. Expects the
    # eigenvalue count to be 32, the eigenvector matrix to be orthogonal (absolute tolerance
    # 2e-3) and the matrix to be reconstructed from its eigenpairs. Reads a file; fails if the
    # file is missing or has the wrong shape.
    # Test a 32x32 matrix which is known to fail if denorm floats are flushed to
    # zero.
    matrix = np.genfromtxt(
        test.test_src_dir_path(
            "python/kernel_tests/linalg/testdata/"
            "self_adjoint_eig_fail_if_denorms_flushed.txt")).astype(np.float32)
    self.assertEqual(matrix.shape, (32, 32))
    matrix_tensor = constant_op.constant(matrix)
    with self.session():
      (e, v) = self.evaluate(linalg_ops.self_adjoint_eig(matrix_tensor))
      self.assertEqual(e.size, 32)
      self.assertAllClose(
          np.matmul(v, v.transpose()), np.eye(32, dtype=np.float32), atol=2e-3)
      self.assertAllClose(matrix,
                          np.matmul(np.matmul(v, np.diag(e)), v.transpose()))


def SortEigenDecomposition(e, v):
  # [FUNCTION DOC]
  # HELPER: orders an eigendecomposition for comparison. Inputs: eigenvalue array e and
  # eigenvector array v (columns are vectors). If v has fewer than two dimensions, returns both
  # unchanged. Otherwise sorts eigenvalues ascending along the last axis and reorders the last
  # axis of both arrays by that permutation. Returns the pair; no side effects.
  if v.ndim < 2:
    return e, v
  else:
    perm = np.argsort(e, -1)
    return np.take(e, perm, -1), np.take(v, perm, -1)


def EquilibrateEigenVectorPhases(x, y):
  # [FUNCTION DOC]
  # HELPER: aligns complex phases so eigenvector sets can be compared. Computes, per column, the
  # inner product of x with y, normalizes it to unit magnitude and multiplies x by it. Inputs:
  # two numpy arrays of eigenvectors in columns that differ only by a phase factor. Returns the
  # rotated version of x; does not modify the inputs except through the in-place normalization
  # of the temporary phase array.
  """Equilibrate the phase of the Eigenvectors in the columns of `x` and `y`.

  Eigenvectors are only unique up to an arbitrary phase. This function rotates x
  such that it matches y. Precondition: The columns of x and y differ by a
  multiplicative complex phase factor only.

  Args:
    x: `np.ndarray` with Eigenvectors
    y: `np.ndarray` with Eigenvectors

  Returns:
    `np.ndarray` containing an equilibrated version of x.
  """
  phases = np.sum(np.conj(x) * y, -2, keepdims=True)
  phases /= np.abs(phases)
  return phases * x


def _GetSelfAdjointEigTest(dtype_, shape_, compute_v_):
  # [FUNCTION DOC]
  # HELPER: factory producing one value test for the eigendecomposition. Inputs: TensorFlow
  # dtype, full shape (batch dimensions then a square matrix size), and whether to compute
  # eigenvectors as well as eigenvalues. Returns the test function closing over them.

  def CompareEigenVectors(self, x, y, tol):
    # [FUNCTION DOC]
    # HELPER: nested check that two eigenvector arrays match after phase alignment. Rotates x to
    # match y and asserts close within the absolute tolerance tol. Returns nothing.
    x = EquilibrateEigenVectorPhases(x, y)
    self.assertAllClose(x, y, atol=tol)

  def CompareEigenDecompositions(self, x_e, x_v, y_e, y_v, tol):
    # [FUNCTION DOC]
    # HELPER: nested check that two batched eigendecompositions agree. Flattens the batch
    # dimensions, and for each matrix sorts both decompositions by eigenvalue, asserts the
    # eigenvalues are close (absolute and relative tolerance tol) and compares eigenvectors up to
    # phase. Inputs: the test case, eigenvalues and eigenvectors of both decompositions, and tol.
    num_batches = int(np.prod(x_e.shape[:-1]))
    n = x_e.shape[-1]
    x_e = np.reshape(x_e, [num_batches] + [n])
    x_v = np.reshape(x_v, [num_batches] + [n, n])
    y_e = np.reshape(y_e, [num_batches] + [n])
    y_v = np.reshape(y_v, [num_batches] + [n, n])
    for i in range(num_batches):
      x_ei, x_vi = SortEigenDecomposition(x_e[i, :], x_v[i, :, :])
      y_ei, y_vi = SortEigenDecomposition(y_e[i, :], y_v[i, :, :])
      self.assertAllClose(x_ei, y_ei, atol=tol, rtol=tol)
      CompareEigenVectors(self, x_vi, y_vi, tol)

  def Test(self):
    # [FUNCTION DOC]
    # Generated value test body. Seeds numpy, builds a random square matrix of the captured dtype
    # (with random imaginary part if complex), makes it Hermitian by adding its conjugate
    # transpose, and tiles it across the batch shape. Computes the numpy reference
    # eigendecomposition. With vectors: runs the op, asserts the product of vectors, diagonal
    # eigenvalues and conjugate-transposed vectors reconstructs the input, and compares to the
    # reference. Without vectors: asserts the sorted eigenvalues match the reference. Tolerance is
    # looser for single precision.
    np.random.seed(1)
    n = shape_[-1]
    batch_shape = shape_[:-2]
    np_dtype = dtype_.as_numpy_dtype
    a = np.random.uniform(
        low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
    if dtype_.is_complex:
      a += 1j * np.random.uniform(
          low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
    a += np.conj(a.T)
    a = np.tile(a, batch_shape + (1, 1))
    if dtype_ in (dtypes_lib.float32, dtypes_lib.complex64):
      atol = 1e-4
    else:
      atol = 1e-12
    np_e, np_v = np.linalg.eigh(a)
    with self.session():
      if compute_v_:
        tf_e, tf_v = linalg_ops.self_adjoint_eig(constant_op.constant(a))

        # Check that V*diag(E)*V^T is close to A.
        a_ev = test_util.matmul_without_tf32(
            test_util.matmul_without_tf32(tf_v, array_ops.matrix_diag(tf_e)),
            tf_v,
            adjoint_b=True)
        self.assertAllClose(self.evaluate(a_ev), a, atol=atol)

        # Compare to numpy.linalg.eigh.
        CompareEigenDecompositions(self, np_e, np_v, self.evaluate(tf_e),
                                   self.evaluate(tf_v), atol)
      else:
        tf_e = linalg_ops.self_adjoint_eigvals(constant_op.constant(a))
        self.assertAllClose(
            np.sort(np_e, -1), np.sort(self.evaluate(tf_e), -1), atol=atol)

  return Test


class SelfAdjointEigGradTest(test.TestCase):
  # [CLASS DOC]
  # Empty test case for the eigendecomposition gradient; its tests are attached dynamically by
  # the script entry point.
  pass  # Filled in below


def _GetSelfAdjointEigGradTest(dtype_, shape_, compute_v_):
  # [FUNCTION DOC]
  # HELPER: factory producing one gradient test. Inputs: TensorFlow dtype, full shape, and whether
  # eigenvectors are computed. Returns the test function closing over them.

  def Test(self):
    # [FUNCTION DOC]
    # Generated gradient test body. Seeds numpy, picks a finite-difference step from the dtype
    # epsilon and a tolerance (loose for single precision). Builds functions of the input: with
    # vectors, one returning eigenvalues and one returning eigenvectors normalized so the first
    # row has zero phase (sign for real); without vectors, the eigenvalues-only op. For each,
    # asserts the theoretical and numerical gradients are close on a random Hermitian input.
    np.random.seed(1)
    n = shape_[-1]
    batch_shape = shape_[:-2]
    np_dtype = dtype_.as_numpy_dtype

    def RandomInput():
      # [FUNCTION DOC]
      # HELPER: nested generator of a random Hermitian matrix of the captured dtype, made by adding
      # the conjugate transpose of a uniform random matrix (with random imaginary part if
      # complex), tiled across the batch shape. Draws from numpy's global random state. Returns it.
      a = np.random.uniform(
          low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
      if dtype_.is_complex:
        a += 1j * np.random.uniform(
            low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
      a += np.conj(a.T)
      a = np.tile(a, batch_shape + (1, 1))
      return a

    # Optimal stepsize for central difference is O(epsilon^{1/3}).
    epsilon = np.finfo(np_dtype).eps
    delta = 0.1 * epsilon**(1.0 / 3.0)
    # Tolerance obtained by perturbing inputs and delta.
    # Changing the step size delta above yields max errors of approx 0.016 for
    # float32.
    _ = RandomInput()
    if dtype_ in (dtypes_lib.float32, dtypes_lib.complex64):
      tol = 2e-2
    else:
      tol = 1e-7
    with self.session():

      def Compute(x):
        # [FUNCTION DOC]
        # HELPER: nested function computing the eigendecomposition of x and normalizing the
        # eigenvector phase so the first row of each vector has zero phase (multiplying by the sign
        # for real dtypes, by the unit complex phase for complex). Returns the pair of eigenvalues
        # and normalized eigenvectors.
        e, v = linalg_ops.self_adjoint_eig(x)
        # (complex) Eigenvectors are only unique up to an arbitrary phase
        # We normalize the vectors such that the first component has phase 0.
        top_rows = v[..., 0:1, :]
        if dtype_.is_complex:
          angle = -math_ops.angle(top_rows)
          phase = math_ops.complex(math_ops.cos(angle), math_ops.sin(angle))
        else:
          phase = math_ops.sign(top_rows)
        v *= phase
        return e, v

      if compute_v_:
        funcs = [lambda x: Compute(x)[0], lambda x: Compute(x)[1]]
      else:
        funcs = [linalg_ops.self_adjoint_eigvals]

      for f in funcs:
        theoretical, numerical = gradient_checker_v2.compute_gradient(
            f,
            [RandomInput()],
            delta=delta)
        self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


if __name__ == "__main__":
  dtypes_to_test = [
      dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.complex64,
      dtypes_lib.complex128
  ]
  for compute_v in True, False:
    for dtype in dtypes_to_test:
      for size in 1, 2, 5, 10:
        for batch_dims in [(), (3,)] + [(3, 2)] * (max(size, size) < 10):
          shape = batch_dims + (size, size)
          name = "%s_%s_%s" % (dtype.name, "_".join(map(str, shape)), compute_v)
          _AddTest(SelfAdjointEigTest, "SelfAdjointEig", name,
                   _GetSelfAdjointEigTest(dtype, shape, compute_v))
          _AddTest(SelfAdjointEigGradTest, "SelfAdjointEigGrad", name,
                   _GetSelfAdjointEigGradTest(dtype, shape, compute_v))
  test.main()
