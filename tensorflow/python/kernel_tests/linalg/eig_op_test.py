# [MODULE DOC]
# Test module for the general (non-symmetric) eigendecomposition ops in linalg_ops: the eig op
# (eigenvalues and eigenvectors) and the eigvals op (eigenvalues only), plus the gradients of
# both. Hand-written tests cover invalid ranks, concurrent execution, a known hard 32x32
# self-adjoint case and mismatched output types. Most tests are generated programmatically:
# factory functions return test functions for every combination of element type, matrix size and
# batch shape, and the main block attaches them to the test classes before starting the runner.
# KNOWN ISSUE: the generated tests are only registered when the file is run as a script, so
# importing the module (for example from another test runner) yields only the hand-written tests.
# The module also defines small NumPy helpers that sort eigenvalues and align eigenvector phases.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.linalg_ops.eig."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.platform import test


def _AddTest(test_class, op_name, testcase_name, fn):
  # [FUNCTION DOC]
  # HELPER: registers a generated test function on a test class.
  # Inputs: test_class is the class to extend; op_name and testcase_name are strings combined
  # (with a test prefix and underscores) into the method name; fn is the test function taking the
  # test instance. Raises a runtime error if a method with that name already exists. Side effect:
  # adds the method to the class. Returns nothing.
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)


class EigTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the eig and eigvals ops. Contains a few hand-written tests; the comprehensive
  # correctness tests (per type, size and batch shape) are added to this class at run time by the
  # main block through the test registration helper.

  @test_util.run_deprecated_v1
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment refers to the self-adjoint eigendecomposition, but the code
    # under test is the general eig op.
    # Scenario (graph mode v1): passes a scalar and then a one dimensional tensor to eig and
    # asserts a value error each time, since at least rank two is required.
    # The input to self_adjoint_eig should be a tensor of
    # at least rank 2.
    scalar = constant_op.constant(1.)
    with self.assertRaises(ValueError):
      linalg_ops.eig(scalar)
    vector = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_ops.eig(vector)

  @test_util.run_deprecated_v1
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Scenario (graph mode v1): for both the with-vectors and the eigenvalues-only variants,
    # two identical random five by five matrices (same seed) are decomposed and all results are
    # evaluated together so ops may run concurrently. Asserts the two eigenvalue sets from the
    # eig op are exactly equal, are only approximately equal to the eigenvalues-only results
    # (different algorithm), the two eigenvalues-only results are exactly equal, and the two
    # eigenvector results are exactly equal.
    all_ops = []
    with self.session():
      for compute_v_ in True, False:
        matrix1 = random_ops.random_normal([5, 5], seed=42)
        matrix2 = random_ops.random_normal([5, 5], seed=42)
        if compute_v_:
          e1, v1 = linalg_ops.eig(matrix1)
          e2, v2 = linalg_ops.eig(matrix2)
          all_ops += [e1, v1, e2, v2]
        else:
          e1 = linalg_ops.eigvals(matrix1)
          e2 = linalg_ops.eigvals(matrix2)
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
    # KNOWN ISSUE: this test lives in the general eig test file but exercises the self-adjoint
    # eigendecomposition op, not eig.
    # Scenario: loads a known 32 by 32 symmetric matrix from a text data file under the source
    # tree (fails if the file is missing) that breaks if tiny floating point values are flushed
    # to zero, and computes its self-adjoint eigendecomposition. Asserts there are 32
    # eigenvalues, the eigenvectors are orthonormal (tolerance two thousandths) and the
    # eigenvectors and eigenvalues reconstruct the original matrix.
    # Test a 32x32 matrix which is known to fail if denorm floats are flushed to
    # zero.
    matrix = np.genfromtxt(
        test.test_src_dir_path(
            "python/kernel_tests/linalg/testdata/"
            "self_adjoint_eig_fail_if_denorms_flushed.txt")).astype(np.float32)
    self.assertEqual(matrix.shape, (32, 32))
    matrix_tensor = constant_op.constant(matrix)
    with self.session() as _:
      (e, v) = self.evaluate(linalg_ops.self_adjoint_eig(matrix_tensor))
      self.assertEqual(e.size, 32)
      self.assertAllClose(
          np.matmul(v, v.transpose()), np.eye(32, dtype=np.float32), atol=2e-3)
      self.assertAllClose(matrix,
                          np.matmul(np.matmul(v, np.diag(e)), v.transpose()))

  def testMismatchedDtypes(self):
    # [FUNCTION DOC]
    # Scenario: calls the raw eig op with a single precision real input but requests a
    # complex128 output type (the matching output would be complex64). Asserts a value error or
    # invalid-argument error whose message mentions an invalid output type.
    tensor = constant_op.constant([[0, 1], [2, 3]], dtype=dtypes_lib.float32)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Invalid output dtype"):
      self.evaluate(
          gen_linalg_ops.eig(
              input=tensor,
              Tout=dtypes_lib.complex128,  # Expected dtype: complex64.
              compute_v=True))


def SortEigenValues(e):
  # [FUNCTION DOC]
  # HELPER: gives eigenvalues a canonical order so results can be compared.
  # Input: e, a NumPy array of (possibly complex) eigenvalues, last axis being the eigenvalue
  # axis. Returns the array sorted along the last axis by the sum of real and imaginary parts.
  perm = np.argsort(e.real + e.imag, -1)
  return np.take(e, perm, -1)


def SortEigenDecomposition(e, v):
  # [FUNCTION DOC]
  # HELPER: canonical ordering of an eigendecomposition for comparison.
  # Inputs: e is a NumPy array of eigenvalues; v the matching eigenvector matrix (eigenvectors
  # in columns). If v has fewer than two dimensions both are returned unchanged. Otherwise
  # returns the eigenvalues and the eigenvector columns reordered by the sum of real and
  # imaginary parts of the eigenvalues.
  if v.ndim < 2:
    return e, v
  perm = np.argsort(e.real + e.imag, -1)
  return np.take(e, perm, -1), np.take(v, perm, -1)


def EquilibrateEigenVectorPhases(x, y):
  # [FUNCTION DOC]
  # HELPER: removes the arbitrary phase ambiguity between two eigenvector sets.
  # Inputs: x and y are NumPy arrays of eigenvectors (in columns) assumed to differ only by a
  # unit complex factor per column. Computes each column's phase from the inner product of x and
  # y, normalizes it to unit magnitude (dividing in place) and returns x rotated by it so it can
  # be compared with y. Division by zero would give invalid values if a column is orthogonal.
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


def _GetEigTest(dtype_, shape_, compute_v_):
  # [FUNCTION DOC]
  # HELPER: factory producing one generated eig correctness test.
  # Inputs: dtype_ is a TensorFlow element type; shape_ a tuple of optional batch dimensions
  # followed by the square matrix size; compute_v_ a boolean selecting the eig op (with
  # eigenvectors) versus the eigenvalues-only op. Returns a test function (taking the test
  # instance) that is meant to be attached to the test class.

  def CompareEigenVectors(self, x, y, tol):
    # [FUNCTION DOC]
    # HELPER: asserts two eigenvector matrices agree up to phase. Inputs: the test instance,
    # arrays x and y and an absolute tolerance. Rotates x onto y with the phase helper and
    # asserts closeness within the tolerance.
    x = EquilibrateEigenVectorPhases(x, y)
    self.assertAllClose(x, y, atol=tol)

  def CompareEigenDecompositions(self, x_e, x_v, y_e, y_v, tol):
    # [FUNCTION DOC]
    # HELPER: asserts two (batched) eigendecompositions agree. Inputs: the test instance;
    # eigenvalues and eigenvectors of the first and second decomposition (arrays with matching
    # batch dimensions); an absolute and relative tolerance for values. Flattens the batch
    # dimensions, sorts each decomposition canonically, then asserts eigenvalues are close and
    # eigenvectors are close up to phase for every batch entry.
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
    # The generated test body (a closure over the element type, shape and mode). Seeds NumPy
    # with one, builds one random matrix with entries in minus one to one (with random imaginary
    # parts for complex types), tiled across the batch dimensions, and computes NumPy's
    # eigendecomposition. In the vectors mode it runs the eig op, asserts the eigenvectors times
    # the diagonal of eigenvalues times the inverse of the eigenvectors reconstructs the input,
    # and compares eigenvalues and eigenvectors with NumPy's. In the eigenvalues-only mode it
    # asserts the sorted eigenvalues match NumPy's. The tolerance is looser for single precision
    # types than for double precision.
    np.random.seed(1)
    n = shape_[-1]
    batch_shape = shape_[:-2]
    np_dtype = dtype_.as_numpy_dtype

    def RandomInput():
      # [FUNCTION DOC]
      # HELPER: builds the random test matrix. No arguments (uses the enclosing type, size and
      # batch shape). Returns an array of uniform random entries in minus one to one, plus an
      # independent random imaginary part for complex types, tiled (copied identically) across
      # the batch dimensions. Consumes NumPy random numbers.
      # Most matrices are diagonalizable
      a = np.random.uniform(
          low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
      if dtype_.is_complex:
        a += 1j * np.random.uniform(
            low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
      a = np.tile(a, batch_shape + (1, 1))
      return a

    if dtype_ in (dtypes_lib.float32, dtypes_lib.complex64):
      atol = 1e-4
    else:
      atol = 1e-12

    a = RandomInput()
    np_e, np_v = np.linalg.eig(a)
    with self.session():
      if compute_v_:
        tf_e, tf_v = linalg_ops.eig(constant_op.constant(a))

        # Check that V*diag(E)*V^(-1) is close to A.
        a_ev = math_ops.matmul(
            math_ops.matmul(tf_v, array_ops.matrix_diag(tf_e)),
            linalg_ops.matrix_inverse(tf_v))
        self.assertAllClose(self.evaluate(a_ev), a, atol=atol)

        # Compare to numpy.linalg.eig.
        CompareEigenDecompositions(self, np_e, np_v, self.evaluate(tf_e),
                                   self.evaluate(tf_v), atol)
      else:
        tf_e = linalg_ops.eigvals(constant_op.constant(a))
        self.assertAllClose(
            SortEigenValues(np_e),
            SortEigenValues(self.evaluate(tf_e)),
            atol=atol)

  return Test


class EigGradTest(test.TestCase):
  # [CLASS DOC]
  # Gradient tests for the eig and eigvals ops. The class body is empty: gradient tests for each
  # complex element type, size and batch shape are attached at run time by the main block.
  pass  # Filled in below


def _GetEigGradTest(dtype_, shape_, compute_v_):
  # [FUNCTION DOC]
  # HELPER: factory producing one generated gradient test for the eig or eigvals op.
  # Inputs: dtype_ is a TensorFlow element type; shape_ a tuple of optional batch dimensions
  # followed by the square matrix size; compute_v_ a boolean choosing whether eigenvalues and
  # eigenvectors (true) or only eigenvalues (false) are differentiated. Returns a test function
  # taking the test instance, meant to be attached to the gradient test class.

  def Test(self):
    # [FUNCTION DOC]
    # The generated gradient test body (a closure over type, shape and mode). Seeds NumPy with
    # one, picks a central-difference step from the machine epsilon and a tolerance (loose for
    # single precision types, tight for double), and builds one random matrix tiled across the
    # batch. For the vectors mode it checks the gradient of the sorted eigenvalues and of the
    # phase-normalized eigenvectors separately; otherwise the gradient of the eigenvalue op. For
    # each function it compares theoretical and numerical gradients within the tolerance.
    np.random.seed(1)
    n = shape_[-1]
    batch_shape = shape_[:-2]
    np_dtype = dtype_.as_numpy_dtype

    def RandomInput():
      # [FUNCTION DOC]
      # HELPER: builds the random test matrix. No arguments (uses the enclosing type, size and
      # batch shape). Returns uniform random entries in minus one to one, with an independent
      # random imaginary part for complex types, tiled identically across the batch dimensions.
      # Consumes NumPy random numbers.
      # Most matrices are diagonalizable
      a = np.random.uniform(
          low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
      if dtype_.is_complex:
        a += 1j * np.random.uniform(
            low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
      a = np.tile(a, batch_shape + (1, 1))
      return a

    # Optimal stepsize for central difference is O(epsilon^{1/3}).
    epsilon = np.finfo(np_dtype).eps
    delta = 0.1 * epsilon**(1.0 / 3.0)
    # tolerance obtained by looking at actual differences using
    # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
    # after discarding one random input sample
    _ = RandomInput()
    if dtype_ in (dtypes_lib.float32, dtypes_lib.complex64):
      tol = 1e-2
    else:
      tol = 1e-7
    with self.session():

      def Compute(x):
        # [FUNCTION DOC]
        # HELPER: deterministic version of the eig op for gradient checking. Input: x, a matrix
        # tensor. Runs the eig op, sorts the eigenvalues by the sum of real and imaginary parts
        # and reorders the eigenvectors the same way, then rotates each eigenvector so its first
        # component has zero phase (eigenvectors are unique only up to a phase). Returns the
        # pair of sorted eigenvalues and normalized eigenvectors.
        e, v = linalg_ops.eig(x)

        # We sort eigenvalues by e.real+e.imag to have consistent
        # order between runs
        b_dims = len(e.shape) - 1
        idx = sort_ops.argsort(math_ops.real(e) + math_ops.imag(e), axis=-1)
        e = array_ops.gather(e, idx, batch_dims=b_dims)
        v = array_ops.gather(v, idx, batch_dims=b_dims)

        # (complex) Eigenvectors are only unique up to an arbitrary phase
        # We normalize the vectors such that the first component has phase 0.
        top_rows = v[..., 0:1, :]
        angle = -math_ops.angle(top_rows)
        phase = math_ops.complex(math_ops.cos(angle), math_ops.sin(angle))
        v *= phase
        return e, v

      if compute_v_:
        funcs = [lambda x: Compute(x)[0], lambda x: Compute(x)[1]]
      else:
        funcs = [linalg_ops.eigvals]

      for f in funcs:
        theoretical, numerical = gradient_checker_v2.compute_gradient(
            f, [RandomInput()], delta=delta)
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
          _AddTest(EigTest, "Eig", name, _GetEigTest(dtype, shape, compute_v))

          if dtype not in [dtypes_lib.float32, dtypes_lib.float64]:
            _AddTest(EigGradTest, "EigGrad", name,
                     _GetEigGradTest(dtype, shape, compute_v))
  test.main()
