# [MODULE DOC]
# Tests for the linear algebra ops exposed by the linalg ops and linalg packages: Cholesky
# solve, log determinant, sign and log determinant, adjoint, identity-matrix construction (eye),
# matrix rank, pseudo-inverse, LU reconstruct / inverse / solve, and the symmetric tridiagonal
# eigensolver. Tests compare against numpy or scipy references in graph and eager modes, with
# static and dynamic shapes. Module-level helpers build random positive definite matrices and
# tensors with hidden shape or value. No import-time side effects beyond defining test classes;
# running the file as a program starts the test runner.
#
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
"""Tests for tensorflow.python.ops.linalg_ops."""

import itertools

from absl.testing import parameterized
import numpy as np
import scipy.linalg

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.platform import test


def _RandomPDMatrix(n, rng, dtype=np.float64):
  # [FUNCTION DOC]
  # HELPER: shared test utility that builds a random n by n positive definite matrix.
  # KNOWN ISSUE: for complex dtypes the product of the conjugate with the plain transpose is
  # generally not Hermitian positive definite (it would need the conjugate transpose), so the
  # result is only guaranteed positive definite for real dtypes.
  # Inputs: n is the matrix size, rng is a numpy random state used for normal samples, dtype is
  # the numpy element type (default double); for complex types the imaginary part is also drawn
  # from the generator. Returns the product of the conjugated random matrix and its transpose.
  # Side effect: advances the random state.
  """Random positive definite matrix."""
  temp = rng.randn(n, n).astype(dtype)
  if dtype in [np.complex64, np.complex128]:
    temp.imag = rng.randn(n, n)
  return np.conj(temp).dot(temp.T)


class CholeskySolveTest(test.TestCase):
  # [CLASS DOC]
  # Tests the Cholesky-based linear solve against random positive definite matrices. Fixture: a
  # numpy random state seeded with zero, created in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: stores a deterministic random state (seed zero) on the test instance.
    # Takes no inputs and returns nothing.
    self.rng = np.random.RandomState(0)

  @test_util.run_deprecated_v1
  def test_works_with_five_different_random_pos_def_matrices(self):
    # [FUNCTION DOC]
    # Test scenario: for matrix sizes one to five and single and double precision, builds a batch
    # of two random positive definite matrices, factors it with Cholesky, then for one and two
    # right-hand-side columns solves with the Cholesky factor. Asserts that multiplying the
    # original matrices by the solution reproduces the right-hand side within the per-dtype
    # tolerance. Uses sub-tests per parameter combination and graph mode.
    for n in range(1, 6):
      for np_type, atol in [(np.float32, 0.05), (np.float64, 1e-5)]:
        with self.session():
          # Create 2 x n x n matrix
          array = np.array(
              [_RandomPDMatrix(n, self.rng),
               _RandomPDMatrix(n, self.rng)]).astype(np_type)
          chol = linalg_ops.cholesky(array)
          for k in range(1, 3):
            with self.subTest(n=n, np_type=np_type, atol=atol, k=k):
              rhs = self.rng.randn(2, n, k).astype(np_type)
              x = linalg_ops.cholesky_solve(chol, rhs)
              self.assertAllClose(rhs, math_ops.matmul(array, x), atol=atol)


class LogdetTest(test.TestCase):
  # [CLASS DOC]
  # Tests the log determinant op for positive definite real and complex matrices, comparing with
  # numpy. Fixture: a numpy random state seeded with 42 created in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: stores a deterministic random state (seed 42) on the test instance.
    # Takes no inputs and returns nothing.
    self.rng = np.random.RandomState(42)

  @test_util.run_deprecated_v1
  def test_works_with_five_different_random_pos_def_matrices(self):
    # [FUNCTION DOC]
    # Test scenario: for sizes one to five and four dtypes (single/double real and complex),
    # builds a random positive definite matrix, computes the reference log absolute determinant
    # with numpy and the log determinant with the library, and asserts they agree within the
    # dtype tolerance. Runs in graph mode with sub-tests.
    for n in range(1, 6):
      for np_dtype, atol in [(np.float32, 0.05), (np.float64, 1e-5),
                             (np.complex64, 0.05), (np.complex128, 1e-5)]:
        with self.subTest(n=n, np_dtype=np_dtype, atol=atol):
          matrix = _RandomPDMatrix(n, self.rng, np_dtype)
          _, logdet_np = np.linalg.slogdet(matrix)
          with self.session():
            # Create 2 x n x n matrix
            # matrix = np.array(
            #     [_RandomPDMatrix(n, self.rng, np_dtype),
            #      _RandomPDMatrix(n, self.rng, np_dtype)]).astype(np_dtype)
            logdet_tf = linalg.logdet(matrix)
            self.assertAllClose(logdet_np, self.evaluate(logdet_tf), atol=atol)

  def test_works_with_underflow_case(self):
    # [FUNCTION DOC]
    # Test scenario: uses a twenty by twenty scaled identity (entries 1e-6) whose plain
    # determinant would underflow in low precision, for four dtypes. Asserts that the library
    # log determinant matches numpy's value within the dtype tolerance.
    for np_dtype, atol in [(np.float32, 0.05), (np.float64, 1e-5),
                           (np.complex64, 0.05), (np.complex128, 1e-5)]:
      with self.subTest(np_dtype=np_dtype, atol=atol):
        matrix = (np.eye(20) * 1e-6).astype(np_dtype)
        _, logdet_np = np.linalg.slogdet(matrix)
        with self.session():
          logdet_tf = linalg.logdet(matrix)
          self.assertAllClose(logdet_np, self.evaluate(logdet_tf), atol=atol)


class SlogdetTest(test.TestCase):
  # [CLASS DOC]
  # Tests the sign-and-log-absolute-determinant op against numpy for random positive definite
  # and near-underflow matrices. Fixture: a numpy random state seeded with 42 created in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: stores a deterministic random state (seed 42) on the test instance.
    # Takes no inputs and returns nothing.
    self.rng = np.random.RandomState(42)

  @test_util.run_deprecated_v1
  def test_works_with_five_different_random_pos_def_matrices(self):
    # [FUNCTION DOC]
    # Test scenario: for sizes one to five and four dtypes, builds a random positive definite
    # matrix and asserts that both the sign and the log absolute determinant from the library
    # match numpy's slogdet within the dtype tolerance. Graph mode with sub-tests.
    for n in range(1, 6):
      for np_dtype, atol in [(np.float32, 0.05), (np.float64, 1e-5),
                             (np.complex64, 0.05), (np.complex128, 1e-5)]:
        with self.subTest(n=n, np_dtype=np_dtype, atol=atol):
          matrix = _RandomPDMatrix(n, self.rng, np_dtype)
          sign_np, log_abs_det_np = np.linalg.slogdet(matrix)
          with self.session():
            sign_tf, log_abs_det_tf = linalg.slogdet(matrix)
            self.assertAllClose(
                log_abs_det_np, self.evaluate(log_abs_det_tf), atol=atol)
            self.assertAllClose(sign_np, self.evaluate(sign_tf), atol=atol)

  def test_works_with_underflow_case(self):
    # [FUNCTION DOC]
    # Test scenario: a twenty by twenty identity scaled by 1e-6, for four dtypes. Asserts that
    # the library sign and log absolute determinant match numpy within the dtype tolerance,
    # showing no underflow problem.
    for np_dtype, atol in [(np.float32, 0.05), (np.float64, 1e-5),
                           (np.complex64, 0.05), (np.complex128, 1e-5)]:
      with self.subTest(np_dtype=np_dtype, atol=atol):
        matrix = (np.eye(20) * 1e-6).astype(np_dtype)
        sign_np, log_abs_det_np = np.linalg.slogdet(matrix)
        with self.session():
          sign_tf, log_abs_det_tf = linalg.slogdet(matrix)
          self.assertAllClose(
              log_abs_det_np, self.evaluate(log_abs_det_tf), atol=atol)
          self.assertAllClose(sign_np, self.evaluate(sign_tf), atol=atol)


class AdjointTest(test.TestCase):
  # [CLASS DOC]
  # Tests the matrix adjoint (conjugate transpose) op against numpy. No shared fixtures.

  def test_compare_to_numpy(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtype list names float64 twice and never float32, so one case is
    # redundant and single precision real is untested.
    # Test scenario: takes a two by three complex matrix cast to each of several dtypes and
    # computes its adjoint. Asserts the static shape is three by two and the values equal the
    # numpy conjugate transpose.
    for dtype in np.float64, np.float64, np.complex64, np.complex128:
      with self.subTest(dtype=dtype):
        matrix_np = np.array([[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j,
                                                         6 + 6j]]).astype(dtype)
        expected_transposed = np.conj(matrix_np.T)
        with self.session():
          matrix = ops.convert_to_tensor(matrix_np)
          transposed = linalg.adjoint(matrix)
          self.assertEqual((3, 2), transposed.get_shape())
          self.assertAllEqual(expected_transposed, self.evaluate(transposed))


class EyeTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Tests the identity-matrix constructor: static shape inference with static and dynamic
  # row/column counts and batch shapes, and value correctness against numpy for many dtypes.
  # Uses parameterized test generation.

  def testShapeInferenceNoBatch(self):
    # [FUNCTION DOC]
    # Test scenario: builds identity matrices without a batch shape, square and rectangular.
    # Asserts the inferred static shapes are two by two and two by three respectively.
    self.assertEqual((2, 2), linalg_ops.eye(num_rows=2).shape)
    self.assertEqual((2, 3), linalg_ops.eye(num_rows=2, num_columns=3).shape)

  def testShapeInferenceStaticBatch(self):
    # [FUNCTION DOC]
    # Test scenario: builds identity matrices with a static batch shape of two by three, once
    # square and once rectangular. Asserts the inferred shapes are batch dimensions followed by
    # the matrix dimensions.
    batch_shape = (2, 3)
    self.assertEqual(
        (2, 3, 2, 2),
        linalg_ops.eye(num_rows=2, batch_shape=batch_shape).shape)
    self.assertEqual(
        (2, 3, 2, 3),
        linalg_ops.eye(
            num_rows=2, num_columns=3, batch_shape=batch_shape).shape)

  @parameterized.named_parameters(
      ("DynamicRow",
       lambda: array_ops.placeholder_with_default(2, shape=None),
       lambda: None),
      ("DynamicRowStaticColumn",
       lambda: array_ops.placeholder_with_default(2, shape=None),
       lambda: 3),
      ("StaticRowDynamicColumn",
       lambda: 2,
       lambda: array_ops.placeholder_with_default(3, shape=None)),
      ("DynamicRowDynamicColumn",
       lambda: array_ops.placeholder_with_default(2, shape=None),
       lambda: array_ops.placeholder_with_default(3, shape=None)))
  def testShapeInferenceStaticBatchWith(self, num_rows_fn, num_columns_fn):
    # [FUNCTION DOC]
    # Test scenario (parameterized by named cases): num_rows_fn and num_columns_fn are callbacks
    # producing the row count and column count, each either a plain number, nothing, or a
    # placeholder tensor with unknown shape. Builds an identity with a static batch of two by
    # three. Asserts the rank is four and the batch dimensions are statically two and three;
    # the row and column dimensions are asserted only when the corresponding input is a static
    # number.
    num_rows = num_rows_fn()
    num_columns = num_columns_fn()
    batch_shape = (2, 3)
    identity_matrix = linalg_ops.eye(
        num_rows=num_rows,
        num_columns=num_columns,
        batch_shape=batch_shape)
    self.assertEqual(4, identity_matrix.shape.ndims)
    self.assertEqual((2, 3), identity_matrix.shape[:2])
    if num_rows is not None and not isinstance(num_rows, tensor.Tensor):
      self.assertEqual(2, identity_matrix.shape[-2])

    if num_columns is not None and not isinstance(num_columns, tensor.Tensor):
      self.assertEqual(3, identity_matrix.shape[-1])

  @parameterized.parameters(
      itertools.product(
          # num_rows
          [0, 1, 2, 5],
          # num_columns
          [None, 0, 1, 2, 5],
          # batch_shape
          [None, [], [2], [2, 3]],
          # dtype
          [
              dtypes.int32,
              dtypes.int64,
              dtypes.float32,
              dtypes.float64,
              dtypes.complex64,
              dtypes.complex128
          ])
      )
  def test_eye_no_placeholder(self, num_rows, num_columns, batch_shape, dtype):
    # [FUNCTION DOC]
    # Test scenario (parameterized over rows, optional columns, optional batch shape and six
    # dtypes): builds the expected identity with numpy, tiled over the batch shape if given,
    # evaluates the library identity with the same arguments and asserts exact equality.
    eye_np = np.eye(num_rows, M=num_columns, dtype=dtype.as_numpy_dtype)
    if batch_shape is not None:
      eye_np = np.tile(eye_np, batch_shape + [1, 1])
    eye_tf = self.evaluate(linalg_ops.eye(
        num_rows,
        num_columns=num_columns,
        batch_shape=batch_shape,
        dtype=dtype))
    self.assertAllEqual(eye_np, eye_tf)

  @parameterized.parameters(
      itertools.product(
          # num_rows
          [0, 1, 2, 5],
          # num_columns
          [0, 1, 2, 5],
          # batch_shape
          [[], [2], [2, 3]],
          # dtype
          [
              dtypes.int32,
              dtypes.int64,
              dtypes.float32,
              dtypes.float64,
              dtypes.complex64,
              dtypes.complex128
          ])
      )
  @test_util.run_deprecated_v1
  def test_eye_with_placeholder(
      self, num_rows, num_columns, batch_shape, dtype):
    # [FUNCTION DOC]
    # Test scenario (parameterized over rows, columns, batch shape and six dtypes, graph mode):
    # creates integer placeholders for the row count, column count and batch shape, builds the
    # identity from them, and runs it in a session feeding the concrete values. Asserts the
    # result equals the numpy identity tiled over the batch shape.
    eye_np = np.eye(num_rows, M=num_columns, dtype=dtype.as_numpy_dtype)
    eye_np = np.tile(eye_np, batch_shape + [1, 1])
    num_rows_placeholder = array_ops.placeholder(
        dtypes.int32, name="num_rows")
    num_columns_placeholder = array_ops.placeholder(
        dtypes.int32, name="num_columns")
    batch_shape_placeholder = array_ops.placeholder(
        dtypes.int32, name="batch_shape")
    eye = linalg_ops.eye(
        num_rows_placeholder,
        num_columns=num_columns_placeholder,
        batch_shape=batch_shape_placeholder,
        dtype=dtype)
    with self.session() as sess:
      eye_tf = sess.run(
          eye,
          feed_dict={
              num_rows_placeholder: num_rows,
              num_columns_placeholder: num_columns,
              batch_shape_placeholder: batch_shape
          })
    self.assertAllEqual(eye_np, eye_tf)


class _MatrixRankTest(object):
  # [CLASS DOC]
  # HELPER: mixin holding the matrix rank test methods; concrete subclasses supply the numpy
  # dtype and whether static shapes are used (use_static_shape), and are combined with the test
  # case base class.

  def test_batch_default_tolerance(self):
    # [FUNCTION DOC]
    # Test scenario: a batch of three three-by-three integer matrices with known ranks two, one
    # and three, fed through a placeholder with static or unknown shape. Asserts the computed
    # ranks with default tolerance are two, one and three.
    x_ = np.array(
        [
            [
                [2, 3, -2],  # = row2+row3
                [-1, 1, -2],
                [3, 2, 0]
            ],
            [
                [0, 2, 0],  # = 2*row2
                [0, 1, 0],
                [0, 3, 0]
            ],  # = 3*row2
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ],
        self.dtype)
    x = array_ops.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    self.assertAllEqual([2, 1, 3], self.evaluate(linalg.matrix_rank(x)))

  def test_custom_tolerance_broadcasts(self):
    # [FUNCTION DOC]
    # Test scenario: builds a matrix with singular values 0.1, 0.2 and 0.3 (orthogonal matrix
    # from a QR factorization of random data, scaled and solved) and requests ranks with a
    # column of four tolerances that broadcast against the batch. Asserts the ranks are
    # three, two, one and zero for tolerances just above and below each singular value.
    q = linalg.qr(random_ops.random_uniform([3, 3], dtype=self.dtype))[0]
    e = constant_op.constant([0.1, 0.2, 0.3], dtype=self.dtype)
    a = linalg.solve(q, linalg.transpose(a=e * q), adjoint=True)
    self.assertAllEqual([3, 2, 1, 0],
                        self.evaluate(
                            linalg.matrix_rank(
                                a, tol=[[0.09], [0.19], [0.29], [0.31]])))

  def test_nonsquare(self):
    # [FUNCTION DOC]
    # Test scenario: a batch of two three-by-four integer matrices with known ranks two and one,
    # fed through a placeholder with static or unknown shape. Asserts the computed ranks are two
    # and one.
    x_ = np.array(
        [
            [
                [2, 3, -2, 2],  # = row2+row3
                [-1, 1, -2, 4],
                [3, 2, 0, -2]
            ],
            [
                [0, 2, 0, 6],  # = 2*row2
                [0, 1, 0, 3],
                [0, 3, 0, 9]
            ]
        ],  # = 3*row2
        self.dtype)
    x = array_ops.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    self.assertAllEqual([2, 1], self.evaluate(linalg.matrix_rank(x)))


@test_util.run_all_in_graph_and_eager_modes
class MatrixRankStatic32Test(test.TestCase, _MatrixRankTest):
  # [CLASS DOC]
  # Runs the matrix rank tests with single precision and statically known shapes, in both graph
  # and eager modes.
  dtype = np.float32
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class MatrixRankDynamic64Test(test.TestCase, _MatrixRankTest):
  # [CLASS DOC]
  # Runs the matrix rank tests with double precision and unknown static shapes, in both graph
  # and eager modes.
  dtype = np.float64
  use_static_shape = False


class _PinvTest(object):
  # [CLASS DOC]
  # HELPER: mixin with pseudo-inverse tests. Concrete subclasses define the numpy dtype, whether
  # shapes are static and whether the default cutoff ratio is used (otherwise a custom per-matrix
  # cutoff is passed).

  def expected_pinv(self, a, rcond):
    # [FUNCTION DOC]
    # HELPER: reference implementation for the pseudo-inverse tests. Inputs: a is a numpy array
    # of matrices (optionally batched), rcond is the cutoff ratio (nothing for the default, a
    # scalar, or a per-batch-element array). For rank below three it simply calls numpy.
    # Otherwise it computes the default cutoff (ten times the larger dimension times machine
    # epsilon) when none is given, and loops over batch indices calling numpy per matrix,
    # using the scalar cutoff or the per-index entry. Returns an array of the transposed
    # matrix shape with the same dtype.
    """Calls `np.linalg.pinv` but corrects its broken batch semantics."""
    if a.ndim < 3:
      return np.linalg.pinv(a, rcond)
    if rcond is None:
      rcond = 10. * max(a.shape[-2], a.shape[-1]) * np.finfo(a.dtype).eps
    s = np.concatenate([a.shape[:-2], [a.shape[-1], a.shape[-2]]])
    a_pinv = np.zeros(s, dtype=a.dtype)
    for i in np.ndindex(a.shape[:(a.ndim - 2)]):
      a_pinv[i] = np.linalg.pinv(
          a[i], rcond=rcond if isinstance(rcond.tolist(), float) else rcond[i])
    return a_pinv

  def test_symmetric(self):
    # [FUNCTION DOC]
    # Test scenario: a batch of two symmetric three-by-three matrices (one shifted by one), with
    # either the default cutoff or a custom per-matrix cutoff that forces the smallest component
    # to zero. Computes the library pseudo-inverse with argument validation and asserts it is
    # close to the numpy reference; with static shapes also asserts the static shape matches.
    a_ = self.dtype([[1., .4, .5], [.4, .2, .25], [.5, .25, .35]])
    a_ = np.stack([a_ + 1., a_], axis=0)  # Batch of matrices.
    a = array_ops.placeholder_with_default(
        a_, shape=a_.shape if self.use_static_shape else None)
    if self.use_default_rcond:
      rcond = None
    else:
      rcond = self.dtype([0., 0.01])  # Smallest 1 component is forced to zero.
    expected_a_pinv_ = self.expected_pinv(a_, rcond)
    a_pinv = linalg.pinv(a, rcond, validate_args=True)
    a_pinv_ = self.evaluate(a_pinv)
    self.assertAllClose(expected_a_pinv_, a_pinv_, atol=2e-5, rtol=2e-5)
    if not self.use_static_shape:
      return
    self.assertAllEqual(expected_a_pinv_.shape, a_pinv.shape)

  def test_nonsquare(self):
    # [FUNCTION DOC]
    # Test scenario: a batch of two three-by-four matrices, with the default cutoff or a custom
    # per-matrix cutoff that zeroes the two smallest components. Computes the library
    # pseudo-inverse with argument validation, asserts closeness to the numpy reference with
    # slightly looser tolerances, and with static shapes asserts the shape matches.
    a_ = self.dtype([[1., .4, .5, 1.], [.4, .2, .25, 2.], [.5, .25, .35, 3.]])
    a_ = np.stack([a_ + 0.5, a_], axis=0)  # Batch of matrices.
    a = array_ops.placeholder_with_default(
        a_, shape=a_.shape if self.use_static_shape else None)
    if self.use_default_rcond:
      rcond = None
    else:
      # Smallest 2 components are forced to zero.
      rcond = self.dtype([0., 0.25])
    expected_a_pinv_ = self.expected_pinv(a_, rcond)
    a_pinv = linalg.pinv(a, rcond, validate_args=True)
    a_pinv_ = self.evaluate(a_pinv)
    self.assertAllClose(expected_a_pinv_, a_pinv_, atol=1e-5, rtol=1e-4)
    if not self.use_static_shape:
      return
    self.assertAllEqual(expected_a_pinv_.shape, a_pinv.shape)


@test_util.run_all_in_graph_and_eager_modes
class PinvTestDynamic32DefaultRcond(test.TestCase, _PinvTest):
  # [CLASS DOC]
  # Pseudo-inverse tests with single precision, dynamic shapes and the default cutoff ratio.
  dtype = np.float32
  use_static_shape = False
  use_default_rcond = True


@test_util.run_all_in_graph_and_eager_modes
class PinvTestStatic64DefaultRcond(test.TestCase, _PinvTest):
  # [CLASS DOC]
  # Pseudo-inverse tests with double precision, static shapes and the default cutoff ratio.
  dtype = np.float64
  use_static_shape = True
  use_default_rcond = True


@test_util.run_all_in_graph_and_eager_modes
class PinvTestDynamic32CustomtRcond(test.TestCase, _PinvTest):
  # [CLASS DOC]
  # Pseudo-inverse tests with single precision, dynamic shapes and a custom cutoff ratio.
  # The class name contains a typo ("Customt").
  dtype = np.float32
  use_static_shape = False
  use_default_rcond = False


@test_util.run_all_in_graph_and_eager_modes
class PinvTestStatic64CustomRcond(test.TestCase, _PinvTest):
  # [CLASS DOC]
  # Pseudo-inverse tests with double precision, static shapes and a custom cutoff ratio.
  dtype = np.float64
  use_static_shape = True
  use_default_rcond = False


def make_tensor_hiding_attributes(value, hide_shape, hide_value=True):
  # [FUNCTION DOC]
  # HELPER: small shared test utility that converts a value to a tensor while optionally hiding
  # static information.
  # KNOWN ISSUE: nothing in this file calls this function, so it is unused here.
  # Inputs: value is any array-like, hide_shape says whether the static
  # shape should be unknown, hide_value (default true) says whether the value should be hidden
  # behind a placeholder with default. If hide_value is false a plain tensor is returned.
  # Otherwise returns a placeholder-with-default holding the value, with unknown shape if
  # hide_shape is set, else the value's own shape (or unknown if it has none).
  if not hide_value:
    return ops.convert_to_tensor(value)

  shape = None if hide_shape else getattr(value, "shape", None)
  return array_ops.placeholder_with_default(value, shape=shape)


class _LUReconstruct(object):
  # [CLASS DOC]
  # HELPER: mixin with tests for reconstructing a matrix from its LU factorization; subclasses
  # choose static or dynamic shapes through use_static_shape (default single precision, static).
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    # [FUNCTION DOC]
    # Test scenario: factors a single two-by-two matrix with LU, reconstructs it from the
    # factors with argument validation, and asserts the static shape (when static) and values
    # equal the original within a relative tolerance.
    x_ = np.array([[3, 4], [1, 2]], dtype=self.dtype)
    x = array_ops.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = linalg.lu_reconstruct(*linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(x_, y_, atol=0., rtol=1e-3)

  def test_batch(self):
    # [FUNCTION DOC]
    # Test scenario: same as the non-batch case but with a batch of two two-by-two matrices.
    # Asserts shape (when static) and values of the reconstruction match the input.
    x_ = np.array([
        [[3, 4], [1, 2]],
        [[7, 8], [3, 4]],
    ], dtype=self.dtype)
    x = array_ops.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = linalg.lu_reconstruct(*linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(x_, y_, atol=0., rtol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class LUReconstructStatic(test.TestCase, _LUReconstruct):
  # [CLASS DOC]
  # LU reconstruction tests with statically known shapes, in graph and eager modes.
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class LUReconstructDynamic(test.TestCase, _LUReconstruct):
  # [CLASS DOC]
  # LU reconstruction tests with unknown static shapes, in graph and eager modes.
  use_static_shape = False


class _LUMatrixInverse(object):
  # [CLASS DOC]
  # HELPER: mixin with tests for computing a matrix inverse from its LU factorization; subclasses
  # choose static or dynamic shapes through use_static_shape (default single precision, static).
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    # [FUNCTION DOC]
    # Test scenario: factors a single two-by-two matrix with LU, computes the inverse from the
    # factors with argument validation, and asserts the shape (when static) and that the values
    # match numpy's inverse within a relative tolerance.
    x_ = np.array([[1, 2], [3, 4]], dtype=self.dtype)
    x = array_ops.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = linalg.lu_matrix_inverse(*linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(np.linalg.inv(x_), y_, atol=0., rtol=1e-3)

  def test_batch(self):
    # [FUNCTION DOC]
    # Test scenario: same as the non-batch case with a batch of three two-by-two matrices.
    # Asserts shape (when static) and values match numpy's batched inverse.
    x_ = np.array([
        [[1, 2], [3, 4]],
        [[7, 8], [3, 4]],
        [[0.25, 0.5], [0.75, -2.]],
    ],
                  dtype=self.dtype)
    x = array_ops.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = linalg.lu_matrix_inverse(*linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(np.linalg.inv(x_), y_, atol=0., rtol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class LUMatrixInverseStatic(test.TestCase, _LUMatrixInverse):
  # [CLASS DOC]
  # LU-based matrix inverse tests with statically known shapes, in graph and eager modes.
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class LUMatrixInverseDynamic(test.TestCase, _LUMatrixInverse):
  # [CLASS DOC]
  # LU-based matrix inverse tests with unknown static shapes, in graph and eager modes.
  use_static_shape = False


class _LUSolve(object):
  # [CLASS DOC]
  # HELPER: mixin with tests for solving linear systems from an LU factorization; subclasses
  # choose static or dynamic shapes through use_static_shape (default single precision, static).
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    # [FUNCTION DOC]
    # Test scenario: factors a two-by-two matrix with LU and solves against a column of ones.
    # Asserts the permutation is swap of the two rows, the shape (when static) matches and the
    # solution matches numpy's solve within a relative tolerance.
    x_ = np.array([[1, 2], [3, 4]], dtype=self.dtype)
    x = array_ops.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    rhs_ = np.array([[1, 1]], dtype=self.dtype).T
    rhs = array_ops.placeholder_with_default(
        rhs_, shape=rhs_.shape if self.use_static_shape else None)

    lower_upper, perm = linalg.lu(x)
    y = linalg.lu_solve(lower_upper, perm, rhs, validate_args=True)
    y_, perm_ = self.evaluate([y, perm])

    self.assertAllEqual([1, 0], perm_)
    expected_ = np.linalg.solve(x_, rhs_)
    if self.use_static_shape:
      self.assertAllEqual(expected_.shape, y.shape)
    self.assertAllClose(expected_, y_, atol=0., rtol=1e-3)

  def test_batch_broadcast(self):
    # [FUNCTION DOC]
    # Test scenario: factors a batch of three two-by-two matrices and solves against a single
    # right-hand-side column that broadcasts over the batch. Asserts the expected per-matrix
    # permutations, the shape (when static) and that the solution matches numpy's broadcast
    # solve.
    x_ = np.array([
        [[1, 2], [3, 4]],
        [[7, 8], [3, 4]],
        [[0.25, 0.5], [0.75, -2.]],
    ],
                  dtype=self.dtype)
    x = array_ops.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    rhs_ = np.array([[1, 1]], dtype=self.dtype).T
    rhs = array_ops.placeholder_with_default(
        rhs_, shape=rhs_.shape if self.use_static_shape else None)

    lower_upper, perm = linalg.lu(x)
    y = linalg.lu_solve(lower_upper, perm, rhs, validate_args=True)
    y_, perm_ = self.evaluate([y, perm])

    self.assertAllEqual([[1, 0], [0, 1], [1, 0]], perm_)
    expected_ = np.linalg.solve(x_, rhs_[np.newaxis])
    if self.use_static_shape:
      self.assertAllEqual(expected_.shape, y.shape)
    self.assertAllClose(expected_, y_, atol=0., rtol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class LUSolveStatic(test.TestCase, _LUSolve):
  # [CLASS DOC]
  # LU solve tests with statically known shapes, in graph and eager modes.
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class LUSolveDynamic(test.TestCase, _LUSolve):
  # [CLASS DOC]
  # LU solve tests with unknown static shapes, in graph and eager modes.
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class EighTridiagonalTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the symmetric (Hermitian) tridiagonal eigensolver for real and complex dtypes against
  # scipy or numpy references, covering small, Toeplitz, random, selected-range and eigenvector
  # cases. Provides shared residual, orthogonality and comparison helpers; runs in graph and
  # eager modes with parameterized dtypes.

  def check_residual(self, matrix, eigvals, eigvectors, atol):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Inputs: the full matrix, computed eigenvalues, eigenvectors and an
    # absolute tolerance. Asserts that the matrix times the eigenvectors is close to the
    # eigenvectors times the diagonal matrix of eigenvalues. Returns nothing.
    # Test that A*eigvectors is close to eigvectors*diag(eigvals).
    l = math_ops.cast(linalg.diag(eigvals), dtype=eigvectors.dtype)
    av = math_ops.matmul(matrix, eigvectors)
    vl = math_ops.matmul(eigvectors, l)
    self.assertAllClose(av, vl, atol=atol)

  def check_orthogonality(self, eigvectors, tol):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Inputs: eigenvector matrix and a tolerance. Computes the adjoint
    # of the eigenvectors times themselves minus the identity and asserts every entry's absolute
    # value is below the tolerance (eigenvectors are orthonormal). Returns nothing.
    # Test that eigenvectors are orthogonal.
    k = array_ops.shape(eigvectors)[1]
    vtv = math_ops.matmul(
        eigvectors, eigvectors, adjoint_a=True) - linalg.eye(
            k, dtype=eigvectors.dtype)
    self.assertAllLess(math_ops.abs(vtv), tol)

  def run_test(self, alpha, beta, eigvals_only=True):
    # [FUNCTION DOC]
    # HELPER: shared driver for the tridiagonal eigensolver tests. Inputs: alpha is the main
    # diagonal, beta the off-diagonal (length one less), eigvals_only (default true) selects
    # whether eigenvectors are also computed. Builds the dense Hermitian matrix, computes
    # reference eigenvalues (numpy for complex input, scipy tridiagonal solver otherwise), runs
    # the library solver, and asserts eigenvalues match within a tolerance scaled by size,
    # epsilon and the largest eigenvalue. If eigenvectors are requested it also checks
    # orthogonality and the residual. Returns nothing.
    n = alpha.shape[0]
    matrix = np.diag(alpha) + np.diag(beta, 1) + np.diag(np.conj(beta), -1)
    # scipy.linalg.eigh_tridiagonal doesn't support complex inputs, so for
    # this we call the slower numpy.linalg.eigh.
    if np.issubdtype(alpha.dtype, np.complexfloating):
      eigvals_expected, _ = np.linalg.eigh(matrix)
    else:
      eigvals_expected = scipy.linalg.eigh_tridiagonal(
          alpha, beta, eigvals_only=True)
    eigvals = linalg.eigh_tridiagonal(alpha, beta, eigvals_only=eigvals_only)
    if not eigvals_only:
      eigvals, eigvectors = eigvals

    eps = np.finfo(alpha.dtype).eps
    atol = n * eps * np.amax(np.abs(eigvals_expected))
    self.assertAllClose(eigvals_expected, eigvals, atol=atol)
    if not eigvals_only:
      self.check_orthogonality(eigvectors, 2 * np.sqrt(n) * eps)
      self.check_residual(matrix, eigvals, eigvectors, atol)

  @parameterized.parameters((np.float32), (np.float64), (np.complex64),
                            (np.complex128))
  def test_small(self, dtype):
    # [FUNCTION DOC]
    # Test scenario (parameterized by dtype): for sizes one to three, uses all-ones diagonals and
    # off-diagonals (made complex by adding an imaginary part for complex dtypes) and runs the
    # shared eigenvalue comparison.
    for n in [1, 2, 3]:
      alpha = np.ones([n], dtype=dtype)
      beta = np.ones([n - 1], dtype=dtype)
      if np.issubdtype(alpha.dtype, np.complexfloating):
        beta += 1j * beta
      self.run_test(alpha, beta)

  @parameterized.parameters((np.float32), (np.float64), (np.complex64),
                            (np.complex128))
  def test_toeplitz(self, dtype):
    # [FUNCTION DOC]
    # Test scenario (parameterized by dtype): size eight Toeplitz matrices with constant diagonal
    # and off-diagonal pairs including extreme magnitudes (1e10 and 1e-10); complex dtypes get an
    # added imaginary part. Runs the shared eigenvalue comparison for each pair.
    n = 8
    for a, b in [[2, -1], [1, 0], [0, 1], [-1e10, 1e10], [-1e-10, 1e-10]]:
      alpha = a * np.ones([n], dtype=dtype)
      beta = b * np.ones([n - 1], dtype=dtype)
      if np.issubdtype(alpha.dtype, np.complexfloating):
        beta += 1j * beta
      self.run_test(alpha, beta)

  @parameterized.parameters((np.float32), (np.float64), (np.complex64),
                            (np.complex128))
  def test_random_uniform(self, dtype):
    # [FUNCTION DOC]
    # Test scenario (parameterized by dtype): sizes eight and fifty with uniformly random
    # diagonal and off-diagonal values (plus a random imaginary part for complex dtypes). Runs
    # the shared eigenvalue comparison. Uses unseeded global numpy randomness.
    for n in [8, 50]:
      alpha = np.random.uniform(size=(n,)).astype(dtype)
      beta = np.random.uniform(size=(n - 1,)).astype(dtype)
      if np.issubdtype(beta.dtype, np.complexfloating):
        beta += 1j * np.random.uniform(size=(n - 1,)).astype(dtype)
      self.run_test(alpha, beta)

  @parameterized.parameters((np.float32), (np.float64), (np.complex64),
                            (np.complex128))
  def test_select(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the loops stop at n minus two for the last index, so the largest eigenvalue
    # index is never selected.
    # Test scenario (parameterized by dtype): a size-four random tridiagonal matrix. Computes all
    # eigenvalues, then for each valid index range asks the solver for only that index range and
    # for the equivalent value range, asserting both equal the corresponding slice of the full
    # result within a tolerance based on epsilon. Uses unseeded global numpy randomness.
    n = 4
    alpha = np.random.uniform(size=(n,)).astype(dtype)
    beta = np.random.uniform(size=(n - 1,)).astype(dtype)
    eigvals_all = linalg.eigh_tridiagonal(alpha, beta, select="a")

    eps = np.finfo(alpha.dtype).eps
    atol = 2 * n * eps
    for first in range(n - 1):
      for last in range(first + 1, n - 1):
        # Check that we get the expected eigenvalues by selecting by
        # index range.
        eigvals_index = linalg.eigh_tridiagonal(
            alpha, beta, select="i", select_range=(first, last))
        self.assertAllClose(
            eigvals_all[first:(last + 1)], eigvals_index, atol=atol)

        # Check that we get the expected eigenvalues by selecting by
        # value range.
        eigvals_value = linalg.eigh_tridiagonal(
            alpha,
            beta,
            select="v",
            select_range=(eigvals_all[first], eigvals_all[last]))
        self.assertAllClose(
            eigvals_all[first:(last + 1)], eigvals_value, atol=atol)

  @parameterized.parameters((np.float32), (np.float64), (np.complex64),
                            (np.complex128))
  def test_extreme_eigenvalues_test(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test never calls the solver or asserts anything, so it passes trivially.
    # Test scenario (parameterized by dtype): intended to exercise the solver on matrices whose
    # entries are huge or tiny. It builds diagonal and off-diagonal arrays from the largest and
    # smallest representable magnitudes for each pair.
    huge = 0.33 * np.finfo(dtype).max
    tiny = 3 * np.finfo(dtype).tiny
    for (a, b) in [(tiny, tiny), (huge, np.sqrt(huge))]:
      alpha = np.array([-a, -np.sqrt(a), np.sqrt(a), a]).astype(dtype)

      beta = b * np.ones([3], dtype=dtype)
      if np.issubdtype(alpha.dtype, np.complexfloating):
        beta += 1j * beta

  @parameterized.parameters((np.float32), (np.float64), (np.complex64),
                            (np.complex128))
  def test_eigenvectors(self, dtype):
    # [FUNCTION DOC]
    # Test scenario (parameterized by dtype): returns immediately (skips) when a CUDA GPU is
    # available or XLA is enabled, since those lack the needed tridiagonal solver. Otherwise runs
    # the shared comparison with eigenvectors requested for a random size-eight matrix and for a
    # fully degenerate matrix (constant diagonal, tiny off-diagonal) to check an orthogonal basis
    # is produced. Uses unseeded global numpy randomness.
    if test.is_gpu_available(cuda_only=True) or test_util.is_xla_enabled():
      # cuda and XLA do not yet expose the stabilized tridiagonal solver
      # needed for inverse iteration.
      return
    n = 8
    alpha = np.random.uniform(size=(n,)).astype(dtype)
    beta = np.random.uniform(size=(n - 1,)).astype(dtype)
    if np.issubdtype(beta.dtype, np.complexfloating):
      beta += 1j * np.random.uniform(size=(n - 1,)).astype(dtype)
    self.run_test(alpha, beta, eigvals_only=False)

    # Test that we can correctly generate an orthogonal basis for
    # a fully degenerate matrix.
    eps = np.finfo(dtype).eps
    alpha = np.ones(n).astype(dtype)
    beta = 0.01 * np.sqrt(eps) * np.ones((n - 1)).astype(dtype)
    self.run_test(alpha, beta, eigvals_only=False)


if __name__ == "__main__":
  test.main()
