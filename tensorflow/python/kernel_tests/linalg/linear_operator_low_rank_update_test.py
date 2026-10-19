# [MODULE DOC]
# Test module for the low-rank-update linear operator in the linalg package (an operator of the
# form base operator plus U times a diagonal D times V-adjoint, where D and V are optional).
# Provides a mixin base class holding the shared operator factory (random positive diagonal
# base operator, random correlated-column perturbation factors, matching dense reference matrix,
# and a check of whether the operator chose the Cholesky path) and tape/variable tests, five
# concrete test classes that combine the mixin with the square or non-square derived-class test
# base for different configurations (with or without diagonal, positive or not, with or without
# a separate V), and a class testing that the operator's shape is the broadcast of its
# arguments, with validation errors for incompatible shapes. When run as a script it registers
# the generated base-class tests for the five derived classes and starts the test runner.
# Creates a module-level seeded random generator.
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

import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)


class BaseLinearOperatorLowRankUpdatetest(object):
  # [CLASS DOC]
  # HELPER: shared mixin for the low-rank-update operator test classes in this module; it is not
  # a test case by itself (plain object base) and must be combined with one of the derived-class
  # test bases.
  # Holds the common operator factory and two operator tests. Subclasses configure the variant
  # through three class attributes: whether a diagonal update D is used, whether D is positive,
  # and whether a separate V is used (otherwise V equals U). The attributes default to nothing
  # and must be set by the subclasses.
  """Base test for this type of operator."""

  # Subclasses should set these attributes to either True or False.

  # If True, A = L + UDV^H
  # If False, A = L + UV^H or A = L + UU^H, depending on _use_v.
  _use_diag_update = None

  # If True, diag is > 0, which means D is symmetric positive definite.
  _is_diag_update_positive = None

  # If True, A = L + UDV^H
  # If False, A = L + UDU^H or A = L + UU^H, depending on _use_diag_update
  _use_v = None

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Hook for the base test classes: returns the list of operator shape descriptions to test,
    # all square: an empty 0 by 0 matrix, a 1 by 1, a batched 1 by 3 by 3, a batched 3 by 4 by 4
    # and a 2 by 1 by 4 by 4. A larger 10 by 10 case was removed because of test timeouts.
    shape_info = linear_operator_test_util.OperatorShapesInfo
    # Previously we had a (2, 10, 10) shape at the end.  We did this to test the
    # inversion and determinant lemmas on not-tiny matrices, since these are
    # known to have stability issues.  This resulted in test timeouts, so this
    # shape has been removed, but rest assured, the tests did pass.
    return [
        shape_info((0, 0)),
        shape_info((1, 1)),
        shape_info((1, 3, 3)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 4, 4))]

  def _gen_positive_diag(self, dtype, diag_shape):
    # [FUNCTION DOC]
    # HELPER: generates random diagonals for the operator factory of this mixin.
    # Draws uniform random values between 1e-4 and 1 with the given shape, so all entries are
    # strictly positive. For complex dtypes the values are drawn as 32-bit floats and cast to the
    # complex type (imaginary part zero). Inputs: dtype (a data type object) and diag_shape (list
    # of dimensions). Returns a tensor of that dtype and shape.
    if dtype.is_complex:
      diag = linear_operator_test_util.random_uniform(
          diag_shape, minval=1e-4, maxval=1., dtype=dtypes.float32)
      return math_ops.cast(diag, dtype=dtype)

    return linear_operator_test_util.random_uniform(
        diag_shape, minval=1e-4, maxval=1., dtype=dtype)

  def operator_and_matrix(self, shape_info, dtype, use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory hook used by the base test classes. Inputs: shape_info (its shape is the full
    # batch-plus-matrix shape), dtype, use_placeholder (feed all tensor arguments through
    # placeholders of unknown shape) and ensure_self_adjoint_and_pd (force a positive diagonal
    # update). Builds a positive-definite self-adjoint diagonal base operator L with random
    # positive diagonal (condition numbers up to about 1e4), random correlated-column factors U
    # and V with rank about half the matrix size plus one, and a diagonal update D that is
    # positive when configured or requested and otherwise small random normal values. Creates the
    # low-rank-update operator using V and D only when the class attributes say so, then
    # computes the matching dense matrix (L + U D V-adjoint, or the variant without V and/or D).
    # Also asserts that the operator's internal Cholesky-use flag matches the expectation: true
    # only for L + UU-adjoint, or L + U D U-adjoint with positive D. Returns (operator, dense
    # matrix). Raises an assertion failure if the Cholesky expectation is violated.
    # Recall A = L + UDV^H
    shape = list(shape_info.shape)
    diag_shape = shape[:-1]
    k = shape[-2] // 2 + 1
    u_perturbation_shape = shape[:-1] + [k]
    diag_update_shape = shape[:-2] + [k]

    # base_operator L will be a symmetric positive definite diagonal linear
    # operator, with condition number as high as 1e4.
    base_diag = self._gen_positive_diag(dtype, diag_shape)
    lin_op_base_diag = base_diag

    # U
    u = linear_operator_test_util.random_normal_correlated_columns(
        u_perturbation_shape, dtype=dtype)
    lin_op_u = u

    # V
    v = linear_operator_test_util.random_normal_correlated_columns(
        u_perturbation_shape, dtype=dtype)
    lin_op_v = v

    # D
    if self._is_diag_update_positive or ensure_self_adjoint_and_pd:
      diag_update = self._gen_positive_diag(dtype, diag_update_shape)
    else:
      diag_update = linear_operator_test_util.random_normal(
          diag_update_shape, stddev=1e-4, dtype=dtype)
    lin_op_diag_update = diag_update

    if use_placeholder:
      lin_op_base_diag = array_ops.placeholder_with_default(
          base_diag, shape=None)
      lin_op_u = array_ops.placeholder_with_default(u, shape=None)
      lin_op_v = array_ops.placeholder_with_default(v, shape=None)
      lin_op_diag_update = array_ops.placeholder_with_default(
          diag_update, shape=None)

    base_operator = linalg.LinearOperatorDiag(
        lin_op_base_diag,
        is_positive_definite=True,
        is_self_adjoint=True)

    operator = linalg.LinearOperatorLowRankUpdate(
        base_operator,
        lin_op_u,
        v=lin_op_v if self._use_v else None,
        diag_update=lin_op_diag_update if self._use_diag_update else None,
        is_diag_update_positive=self._is_diag_update_positive)

    # The matrix representing L
    base_diag_mat = array_ops.matrix_diag(base_diag)

    # The matrix representing D
    diag_update_mat = array_ops.matrix_diag(diag_update)

    # Set up mat as some variant of A = L + UDV^H
    if self._use_v and self._use_diag_update:
      # In this case, we have L + UDV^H and it isn't symmetric.
      expect_use_cholesky = False
      matrix = base_diag_mat + math_ops.matmul(
          u, math_ops.matmul(diag_update_mat, v, adjoint_b=True))
    elif self._use_v:
      # In this case, we have L + UDV^H and it isn't symmetric.
      expect_use_cholesky = False
      matrix = base_diag_mat + math_ops.matmul(u, v, adjoint_b=True)
    elif self._use_diag_update:
      # In this case, we have L + UDU^H, which is PD if D > 0, since L > 0.
      expect_use_cholesky = self._is_diag_update_positive
      matrix = base_diag_mat + math_ops.matmul(
          u, math_ops.matmul(diag_update_mat, u, adjoint_b=True))
    else:
      # In this case, we have L + UU^H, which is PD since L > 0.
      expect_use_cholesky = True
      matrix = base_diag_mat + math_ops.matmul(u, u, adjoint_b=True)

    if expect_use_cholesky:
      self.assertTrue(operator._use_cholesky)
    else:
      self.assertFalse(operator._use_cholesky)

    return operator, matrix

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test: builds a low-rank-update operator whose diagonal base operator, U and (when the
    # configuration uses them) V and the diagonal update are all trainable variables, then runs
    # the shared gradient-tape safety check on it.
    base_operator = linalg.LinearOperatorDiag(
        variables_module.Variable([1.], name="diag"),
        is_positive_definite=True,
        is_self_adjoint=True)

    operator = linalg.LinearOperatorLowRankUpdate(
        base_operator,
        u=variables_module.Variable([[2.]], name="u"),
        v=variables_module.Variable([[1.25]], name="v")
        if self._use_v else None,
        diag_update=variables_module.Variable([1.25], name="diag_update")
        if self._use_diag_update else None,
        is_diag_update_positive=self._is_diag_update_positive)
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Test: builds the same variable-backed operator as the tape-safety test, initializes all the
    # operator's variables in a session and runs the shared check that converting the operator's
    # variables to tensors yields an equivalent operator.
    base_operator = linalg.LinearOperatorDiag(
        variables_module.Variable([1.], name="diag"),
        is_positive_definite=True,
        is_self_adjoint=True)

    operator = linalg.LinearOperatorLowRankUpdate(
        base_operator,
        u=variables_module.Variable([[2.]], name="u"),
        v=variables_module.Variable([[1.25]], name="v")
        if self._use_v else None,
        diag_update=variables_module.Variable([1.25], name="diag_update")
        if self._use_diag_update else None,
        is_diag_update_positive=self._is_diag_update_positive)
    with self.cached_session() as sess:
      sess.run([x.initializer for x in operator.variables])
      self.check_convert_variables_to_tensors(operator)


class LinearOperatorLowRankUpdatetestWithDiagUseCholesky(
    BaseLinearOperatorLowRankUpdatetest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the configuration with a positive diagonal update and no separate V (matrix is
  # base plus U D U-adjoint, which is positive definite), so the operator is expected to use the
  # Cholesky path. Combines the shared mixin with the square derived-class test base and sets
  # tighter tolerances (condition numbers up to about 1e4). Does not run in an explicit mode
  # decorator; TensorFloat-32 is disabled per test and restored afterwards.
  """A = L + UDU^H, D > 0, L > 0 ==> A > 0 and we can use a Cholesky."""

  _use_diag_update = True
  _is_diag_update_positive = True
  _use_v = False

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: saves the current TensorFloat-32 setting on the instance, disables it
    # globally, and sets comparison tolerances (absolute and relative: 1e-5 for float32, 1e-10
    # for float64, relative 1e-4 for complex64) suited to condition numbers up to about 1e4.
    # Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Decrease tolerance since we are testing with condition numbers as high as
    # 1e4.
    self._atol[dtypes.float32] = 1e-5
    self._rtol[dtypes.float32] = 1e-5
    self._atol[dtypes.float64] = 1e-10
    self._rtol[dtypes.float64] = 1e-10
    self._rtol[dtypes.complex64] = 1e-4


class LinearOperatorLowRankUpdatetestWithDiagCannotUseCholesky(
    BaseLinearOperatorLowRankUpdatetest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the configuration with a diagonal update of unspecified sign and no separate V
  # (base plus U D U-adjoint with D not known positive), so the operator must not use the
  # Cholesky path; the Cholesky and eigenvalue tests are skipped. Combines the shared mixin with
  # the square derived-class test base and uses looser tolerances. TensorFloat-32 is disabled per
  # test and restored afterwards.
  """A = L + UDU^H, D !> 0, L > 0 ==> A !> 0 and we cannot use a Cholesky."""

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the names of generated tests to skip, here the
    # Cholesky and symmetric eigenvalue tests, which do not apply because the matrix is not
    # known to be positive definite.
    return ["cholesky", "eigvalsh"]

  _use_diag_update = True
  _is_diag_update_positive = False
  _use_v = False

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: saves the current TensorFloat-32 setting on the instance, disables it
    # globally, and sets looser comparison tolerances than the Cholesky variant (1e-4 for
    # float32, 1e-9 for float64, relative 2e-4 for complex64) because this class does not use
    # the more accurate Cholesky path. Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Decrease tolerance since we are testing with condition numbers as high as
    # 1e4.  This class does not use Cholesky, and thus needs even looser
    # tolerance.
    self._atol[dtypes.float32] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._atol[dtypes.float64] = 1e-9
    self._rtol[dtypes.float64] = 1e-9
    self._rtol[dtypes.complex64] = 2e-4


class LinearOperatorLowRankUpdatetestNoDiagUseCholesky(
    BaseLinearOperatorLowRankUpdatetest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the configuration with neither a diagonal update nor a separate V (base plus
  # U U-adjoint, always positive definite), so the operator is expected to use the Cholesky path.
  # Combines the shared mixin with the square derived-class test base and uses tight tolerances.
  # TensorFloat-32 is disabled per test and restored afterwards.
  """A = L + UU^H, L > 0 ==> A > 0 and we can use a Cholesky."""

  _use_diag_update = False
  _is_diag_update_positive = None
  _use_v = False

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: saves the current TensorFloat-32 setting on the instance, disables it
    # globally, and sets comparison tolerances (1e-5 for float32, 1e-10 for float64, relative
    # 1e-4 for complex64) suited to condition numbers up to about 1e4. Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Decrease tolerance since we are testing with condition numbers as high as
    # 1e4.
    self._atol[dtypes.float32] = 1e-5
    self._rtol[dtypes.float32] = 1e-5
    self._atol[dtypes.float64] = 1e-10
    self._rtol[dtypes.float64] = 1e-10
    self._rtol[dtypes.complex64] = 1e-4


class LinearOperatorLowRankUpdatetestNoDiagCannotUseCholesky(
    BaseLinearOperatorLowRankUpdatetest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the configuration with a separate V and no diagonal update (base plus U
  # V-adjoint, not symmetric), so the operator must not use the Cholesky path; the Cholesky and
  # eigenvalue tests are skipped. Combines the shared mixin with the square derived-class test
  # base and uses looser tolerances. TensorFloat-32 is disabled per test and restored afterwards.
  """A = L + UV^H, L > 0 ==> A is not symmetric and we cannot use a Cholesky."""

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the names of generated tests to skip, here the
    # Cholesky and symmetric eigenvalue tests, which do not apply because the matrix is not
    # symmetric.
    return ["cholesky", "eigvalsh"]

  _use_diag_update = False
  _is_diag_update_positive = None
  _use_v = True

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: saves the current TensorFloat-32 setting on the instance, disables it
    # globally, and sets looser comparison tolerances (1e-4 for float32, 1e-9 for float64,
    # absolute 1e-5 and relative 2e-4 for complex64) because this class does not use Cholesky.
    # Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Decrease tolerance since we are testing with condition numbers as high as
    # 1e4.  This class does not use Cholesky, and thus needs even looser
    # tolerance.
    self._atol[dtypes.float32] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._atol[dtypes.float64] = 1e-9
    self._rtol[dtypes.float64] = 1e-9
    self._atol[dtypes.complex64] = 1e-5
    self._rtol[dtypes.complex64] = 2e-4


class LinearOperatorLowRankUpdatetestWithDiagNotSquare(
    BaseLinearOperatorLowRankUpdatetest,
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring is copied from the Cholesky class and is wrong here. The class
  # sets a separate V and a diagonal update, so the matrix is base plus U D V-adjoint, which is
  # not symmetric, and no Cholesky is used. Also, the name says non-square, but the shapes come
  # from the mixin (listed first in the bases), which are all square.
  # Test case for the full configuration (separate V and a positive diagonal update), combined
  # with the non-square derived-class test base. Only disables TensorFloat-32 per test (restored
  # afterwards); keeps default tolerances.
  """A = L + UDU^H, D > 0, L > 0 ==> A > 0 and we can use a Cholesky."""

  _use_diag_update = True
  _is_diag_update_positive = True
  _use_v = True

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: saves the current TensorFloat-32 setting on the instance and disables
    # TensorFloat-32 globally. Does not change tolerances. Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)


@test_util.run_all_without_tensor_float_32(
    "Linear op calls matmul which uses TensorFloat-32.")
class LinearOperatorLowRankUpdateBroadcastsShape(test.TestCase):
  # [CLASS DOC]
  # Test case checking that the low-rank-update operator's shape is the broadcast of the batch
  # shapes of its base operator, U, V and diagonal update, both statically and dynamically, and
  # that incompatible batch shapes or inner dimensions make the constructor raise a value error.
  # All tests run with TensorFloat-32 turned off because the operator uses matmul.
  """Test that the operator's shape is the broadcast of arguments."""

  def test_static_shape_broadcasts_up_from_operator_to_other_args(self):
    # [FUNCTION DOC]
    # Test: a 3-row identity base operator (no batch) with U of shape (2, 3, 2) and a diagonal
    # update of shape (2, 2), all ones. The operator's static shape and the shape of its dense
    # form must both be (2, 3, 3): the batch dimension of the other arguments is broadcast up.
    # Note the diagonal is passed as the third positional argument.
    base_operator = linalg.LinearOperatorIdentity(num_rows=3)
    u = array_ops.ones(shape=[2, 3, 2])
    diag = array_ops.ones(shape=[2, 2])

    operator = linalg.LinearOperatorLowRankUpdate(base_operator, u, diag)

    # domain_dimension is 3
    self.assertAllEqual([2, 3, 3], operator.shape)
    self.assertAllEqual([2, 3, 3], self.evaluate(operator.to_dense()).shape)

  @test_util.run_deprecated_v1
  def test_dynamic_shape_broadcasts_up_from_operator_to_other_args(self):
    # [FUNCTION DOC]
    # Test (graph mode only, legacy v1 behavior): the base identity operator's row count and the
    # shapes of U, V and the diagonal update are placeholders fed at run time: row count 3, U of
    # shape (1, 1, 2, 3, 2), V of shape (1, 2, 1, 3, 2) and diagonal of shape (2, 1, 1, 2). The
    # operator's shape tensor and the shape of its dense form must both evaluate to the
    # broadcast shape (2, 2, 2, 3, 3).
    num_rows_ph = array_ops.placeholder(dtypes.int32)
    base_operator = linalg.LinearOperatorIdentity(num_rows=num_rows_ph)

    u_shape_ph = array_ops.placeholder(dtypes.int32)
    u = array_ops.ones(shape=u_shape_ph)

    v_shape_ph = array_ops.placeholder(dtypes.int32)
    v = array_ops.ones(shape=v_shape_ph)

    diag_shape_ph = array_ops.placeholder(dtypes.int32)
    diag_update = array_ops.ones(shape=diag_shape_ph)

    operator = linalg.LinearOperatorLowRankUpdate(base_operator,
                                                  u=u,
                                                  diag_update=diag_update,
                                                  v=v)

    feed_dict = {
        num_rows_ph: 3,
        u_shape_ph: [1, 1, 2, 3, 2],  # batch_shape = [1, 1, 2]
        v_shape_ph: [1, 2, 1, 3, 2],  # batch_shape = [1, 2, 1]
        diag_shape_ph: [2, 1, 1, 2]  # batch_shape = [2, 1, 1]
    }

    with self.cached_session():
      shape_tensor = operator.shape_tensor().eval(feed_dict=feed_dict)
      self.assertAllEqual([2, 2, 2, 3, 3], shape_tensor)
      dense = operator.to_dense().eval(feed_dict=feed_dict)
      self.assertAllEqual([2, 2, 2, 3, 3], dense.shape)

  def test_u_and_v_incompatible_batch_shape_raises(self):
    # [FUNCTION DOC]
    # Test: U with batch size 5 and V with batch size 4 cannot be broadcast together, so the
    # constructor raises a value error mentioning incompatible shapes.
    base_operator = linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float64)
    u = rng.rand(5, 3, 2)
    v = rng.rand(4, 3, 2)
    with self.assertRaisesRegex(ValueError, "Incompatible shapes"):
      linalg.LinearOperatorLowRankUpdate(base_operator, u=u, v=v)

  def test_u_and_base_operator_incompatible_batch_shape_raises(self):
    # [FUNCTION DOC]
    # Test: a base operator with batch shape (4) and U with batch size 5 cannot be broadcast, so
    # the constructor raises a value error mentioning incompatible shapes.
    base_operator = linalg.LinearOperatorIdentity(
        num_rows=3, batch_shape=[4], dtype=np.float64)
    u = rng.rand(5, 3, 2)
    with self.assertRaisesRegex(ValueError, "Incompatible shapes"):
      linalg.LinearOperatorLowRankUpdate(base_operator, u=u)

  def test_u_and_base_operator_incompatible_domain_dimension(self):
    # [FUNCTION DOC]
    # Test: a 3-row base operator with U having 4 rows has mismatched dimensions, so the
    # constructor raises a value error saying the dimensions are not compatible.
    base_operator = linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float64)
    u = rng.rand(5, 4, 2)
    with self.assertRaisesRegex(ValueError, "not compatible"):
      linalg.LinearOperatorLowRankUpdate(base_operator, u=u)

  def test_u_and_diag_incompatible_low_rank_raises(self):
    # [FUNCTION DOC]
    # Test: U has rank (last dimension) 2 but the diagonal update has 4 entries, so the
    # constructor raises a value error saying the dimensions are not compatible.
    base_operator = linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float64)
    u = rng.rand(5, 3, 2)
    diag = rng.rand(5, 4)  # Last dimension should be 2
    with self.assertRaisesRegex(ValueError, "not compatible"):
      linalg.LinearOperatorLowRankUpdate(base_operator, u=u, diag_update=diag)

  def test_diag_incompatible_batch_shape_raises(self):
    # [FUNCTION DOC]
    # Test: U has batch size 5 and the diagonal update has batch size 4, which cannot be
    # broadcast, so the constructor raises a value error mentioning incompatible shapes.
    base_operator = linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float64)
    u = rng.rand(5, 3, 2)
    diag = rng.rand(4, 2)  # First dimension should be 5
    with self.assertRaisesRegex(ValueError, "Incompatible shapes"):
      linalg.LinearOperatorLowRankUpdate(base_operator, u=u, diag_update=diag)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(
      LinearOperatorLowRankUpdatetestWithDiagUseCholesky)
  linear_operator_test_util.add_tests(
      LinearOperatorLowRankUpdatetestWithDiagCannotUseCholesky)
  linear_operator_test_util.add_tests(
      LinearOperatorLowRankUpdatetestNoDiagUseCholesky)
  linear_operator_test_util.add_tests(
      LinearOperatorLowRankUpdatetestNoDiagCannotUseCholesky)
  linear_operator_test_util.add_tests(
      LinearOperatorLowRankUpdatetestWithDiagNotSquare)
  test.main()
