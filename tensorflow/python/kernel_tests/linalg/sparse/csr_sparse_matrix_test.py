# [MODULE DOC]
# Tests for the CSR sparse matrix wrapper class and its matmul function: construction from a
# sparse tensor or a dense tensor (with or without explicit indices), conjugate, transpose and
# hermitian transpose, and sparse/dense matrix multiplication in all operand combinations.
# Most tests silently pass when no GPU is available. No import-time side effects.
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
"""CSR sparse matrix tests."""

import itertools

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.platform import test


class CSRSparseMatrixTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the CSR sparse matrix class: constructors, round trips to dense and sparse
  # tensors, conjugation and transposition. Caches GPU availability as a class attribute.

  @classmethod
  def setUpClass(cls):  # pylint: disable=g-missing-super-call
    # [FUNCTION DOC]
    # Class-level fixture: records on the class whether a GPU is available (does not call the
    # parent setup). Returns nothing.
    cls._gpu_available = test_util.is_gpu_available()

  @test_util.run_in_graph_and_eager_modes
  def testConstructorFromSparseTensor(self):
    # [FUNCTION DOC]
    # Test (graph and eager): skipped (passes) without a GPU. Builds a 5x6 sparse tensor with four
    # entries, casts it to float32, constructs a CSR sparse matrix from it, and asserts the
    # matrix shape equals the dense shape. Converts back to a sparse tensor and asserts indices,
    # values and dense shape match the originals.
    if not self._gpu_available:
      return

    a_indices = np.array([[0, 0], [2, 3], [2, 4], [3, 0]])
    a_values = [1.0, 5.0, -1.0, -2.0]
    a_dense_shape = [5, 6]

    a_st = sparse_tensor.SparseTensor(a_indices, a_values, a_dense_shape)
    a_st = math_ops.cast(a_st, dtypes.float32)
    a_sm = sparse_csr_matrix_ops.CSRSparseMatrix(a_st)
    self.assertEqual(a_sm.shape, a_dense_shape)

    a_st_rt = a_sm.to_sparse_tensor()
    a_st_rt = self.evaluate(a_st_rt)

    self.assertAllEqual(a_indices, a_st_rt.indices)
    self.assertAllClose(a_values, a_st_rt.values)
    self.assertAllEqual(a_dense_shape, a_st_rt.dense_shape)

  @test_util.run_in_graph_and_eager_modes
  def testConstructorFromDenseTensorNoIndices(self):
    # [FUNCTION DOC]
    # Test (graph and eager): skipped without a GPU. Builds a random 5x7x13 float32 batch with
    # non-positive entries zeroed, constructs a CSR matrix from the dense array without giving
    # indices, and asserts the shape. Evaluates the number of stored entries per batch and the
    # dense round trip, asserting the counts equal per-batch nonzero counts and the dense result
    # matches the input.
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [5, 7, 13]
    a_mats = sparsify(np.random.randn(*dense_shape)).astype(np.float32)

    a_sm = sparse_csr_matrix_ops.CSRSparseMatrix(a_mats)
    self.assertEqual(a_sm.shape, a_mats.shape)

    a_sm_rt = a_sm.to_dense()
    a_sm_nnz = a_sm.nnz()
    a_sm_nnz, a_sm_rt = self.evaluate([a_sm_nnz, a_sm_rt])

    # Count number of nonzero entries for each batch using bincount.
    nz = np.bincount(a_mats.nonzero()[0], minlength=a_mats.shape[0])
    self.assertAllEqual(nz, a_sm_nnz)
    self.assertAllClose(a_mats, a_sm_rt)

  @test_util.run_in_graph_and_eager_modes
  def testConstructorFromDenseTensorWithIndices(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: an inline comment in the body mentions counting nonzeros per batch with a
    # bincount, but no such counting happens here; the comment was copied from the no-indices test.
    # Test (graph and eager): skipped without a GPU. Builds a fully random dense 5x7x13 float32
    # batch and constructs a CSR matrix keeping only two explicitly given index locations.
    # Asserts the matrix shape, then converts to a sparse tensor and asserts its indices equal
    # the given ones, its dense shape matches, and its values equal the dense entries at those
    # indices.
    if not self._gpu_available:
      return

    dense_shape = [5, 7, 13]
    a_mats = np.random.randn(*dense_shape).astype(np.float32)
    indices = np.array([[0, 0, 0],
                        [1, 0, 0]], dtype=np.int64)

    a_sm = sparse_csr_matrix_ops.CSRSparseMatrix(a_mats, indices=indices)
    self.assertEqual(a_sm.shape, a_mats.shape)

    a_sm_st = a_sm.to_sparse_tensor()
    a_sm_st = self.evaluate(a_sm_st)

    # Count number of nonzero entries for each batch using bincount.
    self.assertAllEqual(indices, a_sm_st.indices)
    self.assertAllEqual(dense_shape, a_sm.shape)
    self.assertAllEqual(dense_shape, a_sm_st.dense_shape)
    self.assertAllClose([a_mats[tuple(x)] for x in indices], a_sm_st.values)

  @test_util.run_in_graph_and_eager_modes
  def testConj(self):
    # [FUNCTION DOC]
    # Test (graph and eager): skipped without a GPU. Builds a random complex64 5x7x13 batch
    # (entries with non-positive real part zeroed), wraps it as a CSR matrix, calls conjugate,
    # asserts the result is again a CSR sparse matrix, and that its dense form equals the
    # numpy conjugate of the input.
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m.real > 0)
    dense_shape = [5, 7, 13]
    a_mats = sparsify(
        (np.random.randn(*dense_shape) + 1.j * np.random.randn(*dense_shape))
        .astype(np.complex64))
    a_sm = sparse_csr_matrix_ops.CSRSparseMatrix(a_mats)
    a_sm_conj = a_sm.conj()
    self.assertIsInstance(a_sm_conj, sparse_csr_matrix_ops.CSRSparseMatrix)
    a_sm_conj_dense = a_sm_conj.to_dense()
    a_sm_conj_dense = self.evaluate(a_sm_conj_dense)
    self.assertAllClose(a_mats.conj(), a_sm_conj_dense)

  @test_util.run_in_graph_and_eager_modes
  def testTranspose(self):
    # [FUNCTION DOC]
    # Test (graph and eager): skipped without a GPU. For both conjugate=false and true, builds a
    # random complex64 5x7x13 sparse batch, computes the expected result by swapping the last
    # two axes (and conjugating when requested), then applies the matrix's transpose or
    # hermitian transpose method. Asserts the result is a CSR sparse matrix whose dense form
    # matches the expectation.
    if not self._gpu_available:
      return

    for conjugate in False, True:
      sparsify = lambda m: m * (m > 0)
      dense_shape = [5, 7, 13]
      a_mats = sparsify((np.random.randn(*dense_shape) +
                         1.j * np.random.randn(*dense_shape))).astype(
                             np.complex64)
      expected = np.transpose(a_mats, (0, 2, 1))
      if conjugate:
        expected = np.conj(expected)
      a_sm = sparse_csr_matrix_ops.CSRSparseMatrix(a_mats)
      if conjugate:
        a_sm_t = a_sm.hermitian_transpose()
      else:
        a_sm_t = a_sm.transpose()
      self.assertIsInstance(a_sm_t, sparse_csr_matrix_ops.CSRSparseMatrix)
      a_sm_t_dense = a_sm_t.to_dense()
      a_sm_t_dense = self.evaluate(a_sm_t_dense)
      self.assertAllClose(expected, a_sm_t_dense)


class SparseMatrixMatmulTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the CSR sparse matrix matmul function, comparing sparse-sparse, sparse-dense and
  # dense-sparse products against a dense reference matmul for all valid transpose/adjoint flag
  # combinations. Caches GPU availability as a class attribute.

  @classmethod
  def setUpClass(cls):  # pylint: disable=g-missing-super-call
    # [FUNCTION DOC]
    # Class-level fixture: records on the class whether a GPU is available (does not call the
    # parent setup). Returns nothing.
    cls._gpu_available = test_util.is_gpu_available()

  def _testSparseSparse(self, transpose_a, transpose_b, adjoint_a, adjoint_b):
    # [FUNCTION DOC]
    # HELPER: shared body for the sparse-times-sparse matmul test.
    # Inputs: four booleans giving transpose/adjoint flags for the left and right operands.
    # Skips (returns) without a GPU. For float32 and complex64, builds random sparse batched
    # operands (shapes chosen according to flags), wraps both as CSR matrices, computes a dense
    # reference product without TF32, computes the CSR matmul, asserts the result is a CSR
    # matrix and that its dense form is close to the reference. Returns nothing.
    if not self._gpu_available:
      return
    sparsify = lambda m: m * (m > 0)
    dense_shape_a = [5, 13, 7] if transpose_a or adjoint_a else [5, 7, 13]
    dense_shape_b = [5, 15, 13] if transpose_b or adjoint_b else [5, 13, 15]
    dtypes_to_test = [np.float32, np.complex64]
    for dtype in dtypes_to_test:
      a_mats = sparsify((np.random.randn(*dense_shape_a) +
                         1.j * np.random.randn(*dense_shape_a))).astype(dtype)
      b_mats = sparsify((np.random.randn(*dense_shape_b) +
                         1.j * np.random.randn(*dense_shape_b))).astype(dtype)
      a_sm = sparse_csr_matrix_ops.CSRSparseMatrix(a_mats)
      b_sm = sparse_csr_matrix_ops.CSRSparseMatrix(b_mats)
      c_dense = test_util.matmul_without_tf32(
          a_mats,
          b_mats,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
      c_sm = sparse_csr_matrix_ops.matmul(
          a_sm,
          b_sm,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
      self.assertIsInstance(c_sm, sparse_csr_matrix_ops.CSRSparseMatrix)
      c_sm_dense = c_sm.to_dense()
      c_dense, c_sm_dense = self.evaluate([c_dense, c_sm_dense])
      self.assertAllClose(c_dense, c_sm_dense)

  @test_util.run_in_graph_and_eager_modes
  def testSparseSparse(self):
    # [FUNCTION DOC]
    # Test (graph and eager): iterates over all combinations of the four transpose/adjoint flags,
    # skipping combinations that set both transpose and adjoint on the same operand, and runs
    # the sparse-sparse matmul check for each.
    for (t_a, t_b, adj_a, adj_b) in itertools.product(*(([False, True],) * 4)):
      if (t_a and adj_a) or (t_b and adj_b):
        continue
      self._testSparseSparse(t_a, t_b, adj_a, adj_b)

  def _testSparseDense(self, transpose_a, transpose_b, adjoint_a, adjoint_b):
    # [FUNCTION DOC]
    # HELPER: shared body for the sparse-times-dense matmul test.
    # Inputs: four booleans giving transpose/adjoint flags. Skips without a GPU. For float32 and
    # complex64, builds a random sparse left operand wrapped as a CSR matrix and a fully dense
    # random right operand, computes a dense reference product without TF32, computes the
    # matmul of the CSR matrix with the dense array, evaluates both, and asserts closeness.
    # Returns nothing.
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape_a = [5, 13, 7] if transpose_a or adjoint_a else [5, 7, 13]
    dense_shape_b = [5, 15, 13] if transpose_b or adjoint_b else [5, 13, 15]
    dtypes_to_test = [np.float32, np.complex64]
    for dtype in dtypes_to_test:
      a_mats = sparsify((np.random.randn(*dense_shape_a) +
                         1.j * np.random.randn(*dense_shape_a))).astype(dtype)
      b_mats = (np.random.randn(*dense_shape_b) +
                1.j * np.random.randn(*dense_shape_b)).astype(dtype)
      a_sm = sparse_csr_matrix_ops.CSRSparseMatrix(a_mats)
      c_dense = test_util.matmul_without_tf32(
          a_mats,
          b_mats,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
      c_sm_dense = sparse_csr_matrix_ops.matmul(
          a_sm,
          b_mats,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
      c_dense, c_sm_dense = self.evaluate([c_dense, c_sm_dense])
      self.assertAllClose(c_dense, c_sm_dense)

  @test_util.run_in_graph_and_eager_modes
  def testSparseDense(self):
    # [FUNCTION DOC]
    # Test (graph and eager): runs the sparse-times-dense matmul check for every valid
    # combination of transpose/adjoint flags (never both on the same operand).
    for (t_a, t_b, adj_a, adj_b) in itertools.product(*(([False, True],) * 4)):
      if (t_a and adj_a) or (t_b and adj_b):
        continue
      self._testSparseDense(t_a, t_b, adj_a, adj_b)

  def _testDenseSparse(self, transpose_a, transpose_b, adjoint_a, adjoint_b):
    # [FUNCTION DOC]
    # HELPER: shared body for the dense-times-sparse matmul test.
    # Inputs: four booleans giving transpose/adjoint flags. Skips without a GPU. For float32 and
    # complex64, builds a fully dense random left operand and a sparse right operand wrapped as
    # a CSR matrix, computes a dense reference product without TF32, computes the matmul of the
    # dense array with the CSR matrix, evaluates both, and asserts closeness. Returns nothing.
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape_a = [5, 13, 7] if transpose_a or adjoint_a else [5, 7, 13]
    dense_shape_b = [5, 15, 13] if transpose_b or adjoint_b else [5, 13, 15]
    dtypes_to_test = [np.float32, np.complex64]
    for dtype in dtypes_to_test:
      a_mats = (np.random.randn(*dense_shape_a) +
                1.j * np.random.randn(*dense_shape_a)).astype(dtype)
      b_mats = sparsify((np.random.randn(*dense_shape_b) +
                         1.j * np.random.randn(*dense_shape_b))).astype(dtype)
      b_sm = sparse_csr_matrix_ops.CSRSparseMatrix(b_mats)
      c_dense = test_util.matmul_without_tf32(
          a_mats,
          b_mats,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
      c_sm_dense = sparse_csr_matrix_ops.matmul(
          a_mats,
          b_sm,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
      c_dense, c_sm_dense = self.evaluate([c_dense, c_sm_dense])
      self.assertAllClose(c_dense, c_sm_dense)

  @test_util.run_in_graph_and_eager_modes
  def testDenseSparse(self):
    # [FUNCTION DOC]
    # Test (graph and eager): runs the dense-times-sparse matmul check for every valid
    # combination of transpose/adjoint flags (never both on the same operand).
    for (t_a, t_b, adj_a, adj_b) in itertools.product(*(([False, True],) * 4)):
      if (t_a and adj_a) or (t_b and adj_b):
        continue
      self._testDenseSparse(t_a, t_b, adj_a, adj_b)


if __name__ == "__main__":
  test.main()
