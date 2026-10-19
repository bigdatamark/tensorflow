# [MODULE DOC]
# Tests for the linear operator addition module: the tiered add-operators entry point and the
# individual adders (scaled identity, diagonal, lower-triangular, full matrix). Covers result
# types, dense values, hint propagation, naming, error cases and tier ordering. Defines a
# module-level seeded random generator and a sentinel adder that fails if used. No other
# import-time side effects.
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

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_addition
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)

add_operators = linear_operator_addition.add_operators


# pylint: disable=unused-argument
class _BadAdder(linear_operator_addition._Adder):
  # [CLASS DOC]
  # HELPER: sentinel adder used by the tier-order tests. Any attempt to query or use it fails
  # with an assertion error, proving that a tier containing it was never reached.
  """Adder that will fail if used."""

  def can_add(self, op1, op2):
    # [FUNCTION DOC]
    # HELPER: always raises an assertion error saying can_add was called; inputs are ignored.
    # Used to detect that a tier was consulted when it should not have been.
    raise AssertionError("BadAdder.can_add called!")

  def _add(self, op1, op2, operator_name, hints):
    # [FUNCTION DOC]
    # HELPER: always raises an assertion error saying this line should not be reached; all
    # inputs (two operators, the result name and the hints) are ignored. Never returns.
    raise AssertionError("This line should not be reached")


# pylint: enable=unused-argument


class LinearOperatorAdditionCorrectnessTest(test.TestCase):
  # [CLASS DOC]
  # Tests the operator-addition entry point with the default adder tiers: single operators,
  # empty or invalid input, combining diagonal, lower-triangular and full-matrix operators into
  # one result, naming and hint propagation, and shape-compatibility errors. No shared fixtures
  # beyond the module-level seeded random generator.
  """Tests correctness of addition with combinations of a few Adders.

  Tests here are done with the _DEFAULT_ADDITION_TIERS, which means
  add_operators should reduce all operators resulting in one single operator.

  This shows that we are able to correctly combine adders using the tiered
  system.  All Adders should be tested separately, and there is no need to test
  every Adder within this class.
  """

  def test_one_operator_is_returned_unchanged(self):
    # [FUNCTION DOC]
    # Test scenario: adds a list holding a single diagonal operator. Asserts the result list has
    # length one and its element is the very same operator object.
    op_a = linalg.LinearOperatorDiag([1., 1.])
    op_sum = add_operators([op_a])
    self.assertEqual(1, len(op_sum))
    self.assertIs(op_sum[0], op_a)

  def test_at_least_one_operators_required(self):
    # [FUNCTION DOC]
    # Test scenario: adds an empty list. Asserts a value error whose message says the input must
    # contain at least one operator.
    with self.assertRaisesRegex(ValueError, "must contain at least one"):
      add_operators([])

  def test_attempting_to_add_numbers_raises(self):
    # [FUNCTION DOC]
    # Test scenario: adds a list of plain numbers. Asserts a type error whose message says only
    # linear operators are allowed.
    with self.assertRaisesRegex(TypeError, "contain only LinearOperator"):
      add_operators([1, 2])

  @test_util.run_deprecated_v1
  def test_two_diag_operators(self):
    # [FUNCTION DOC]
    # Test scenario: adds two positive-definite diagonal operators (values one and two). Asserts a
    # single diagonal operator results, dense form has threes on the diagonal, it is positive
    # definite, self-adjoint and non-singular, and has the generated name built from the
    # operand names. Graph mode.
    op_a = linalg.LinearOperatorDiag(
        [1., 1.], is_positive_definite=True, name="A")
    op_b = linalg.LinearOperatorDiag(
        [2., 2.], is_positive_definite=True, name="B")
    with self.cached_session():
      op_sum = add_operators([op_a, op_b])
      self.assertEqual(1, len(op_sum))
      op = op_sum[0]
      self.assertIsInstance(op, linalg_lib.LinearOperatorDiag)
      self.assertAllClose([[3., 0.], [0., 3.]], op.to_dense())
      # Adding positive definite operators produces positive def.
      self.assertTrue(op.is_positive_definite)
      # Real diagonal ==> self-adjoint.
      self.assertTrue(op.is_self_adjoint)
      # Positive definite ==> non-singular
      self.assertTrue(op.is_non_singular)
      # Enforce particular name for this simple case
      self.assertEqual("Add/B__A/", op.name)

  @test_util.run_deprecated_v1
  def test_three_diag_operators(self):
    # [FUNCTION DOC]
    # Test scenario: adds three positive-definite diagonal operators (one, two, three). Asserts
    # one diagonal operator results with sixes on the diagonal, positive definite, self-adjoint
    # and non-singular. Graph mode.
    op1 = linalg.LinearOperatorDiag(
        [1., 1.], is_positive_definite=True, name="op1")
    op2 = linalg.LinearOperatorDiag(
        [2., 2.], is_positive_definite=True, name="op2")
    op3 = linalg.LinearOperatorDiag(
        [3., 3.], is_positive_definite=True, name="op3")
    with self.cached_session():
      op_sum = add_operators([op1, op2, op3])
      self.assertEqual(1, len(op_sum))
      op = op_sum[0]
      self.assertTrue(isinstance(op, linalg_lib.LinearOperatorDiag))
      self.assertAllClose([[6., 0.], [0., 6.]], op.to_dense())
      # Adding positive definite operators produces positive def.
      self.assertTrue(op.is_positive_definite)
      # Real diagonal ==> self-adjoint.
      self.assertTrue(op.is_self_adjoint)
      # Positive definite ==> non-singular
      self.assertTrue(op.is_non_singular)

  @test_util.run_deprecated_v1
  def test_diag_tril_diag(self):
    # [FUNCTION DOC]
    # Test scenario: adds a diagonal, a lower-triangular (hinted self-adjoint and non-singular)
    # and another diagonal operator. Asserts the result is one lower-triangular operator with
    # sixes on the diagonal, self-adjoint, and with the non-singular hint left unknown (nothing)
    # because hints for sums are not inferred from the parts. Graph mode.
    op1 = linalg.LinearOperatorDiag(
        [1., 1.], is_non_singular=True, name="diag_a")
    op2 = linalg.LinearOperatorLowerTriangular(
        [[2., 0.], [0., 2.]],
        is_self_adjoint=True,
        is_non_singular=True,
        name="tril")
    op3 = linalg.LinearOperatorDiag(
        [3., 3.], is_non_singular=True, name="diag_b")
    with self.cached_session():
      op_sum = add_operators([op1, op2, op3])
      self.assertEqual(1, len(op_sum))
      op = op_sum[0]
      self.assertIsInstance(op, linalg_lib.LinearOperatorLowerTriangular)
      self.assertAllClose([[6., 0.], [0., 6.]], op.to_dense())

      # The diag operators will be self-adjoint (because real and diagonal).
      # The TriL operator has the self-adjoint hint set.
      self.assertTrue(op.is_self_adjoint)

      # Even though op1/2/3 are non-singular, this does not imply op is.
      # Since no custom hint was provided, we default to None (unknown).
      self.assertEqual(None, op.is_non_singular)

  @test_util.run_deprecated_v1
  def test_matrix_diag_tril_diag_uses_custom_name(self):
    # [FUNCTION DOC]
    # Test scenario: adds a full-matrix, a diagonal, a lower-triangular and another diagonal
    # operator while passing a custom result name. Asserts one full-matrix operator results with
    # the expected summed dense entries and the custom name. Graph mode.
    op0 = linalg.LinearOperatorFullMatrix(
        [[-1., -1.], [-1., -1.]], name="matrix")
    op1 = linalg.LinearOperatorDiag([1., 1.], name="diag_a")
    op2 = linalg.LinearOperatorLowerTriangular(
        [[2., 0.], [1.5, 2.]], name="tril")
    op3 = linalg.LinearOperatorDiag([3., 3.], name="diag_b")
    with self.cached_session():
      op_sum = add_operators([op0, op1, op2, op3], operator_name="my_operator")
      self.assertEqual(1, len(op_sum))
      op = op_sum[0]
      self.assertIsInstance(op, linalg_lib.LinearOperatorFullMatrix)
      self.assertAllClose([[5., -1.], [0.5, 5.]], op.to_dense())
      self.assertEqual("my_operator", op.name)

  def test_incompatible_domain_dimensions_raises(self):
    # [FUNCTION DOC]
    # Test scenario: adds a two-by-three full matrix and a diagonal operator of different size
    # (domain dimension four). Asserts a value error saying the operators must share the same
    # domain dimension.
    op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3))
    op2 = linalg.LinearOperatorDiag(rng.rand(2, 4))
    with self.assertRaisesRegex(ValueError, "must.*same `domain_dimension`"):
      add_operators([op1, op2])

  def test_incompatible_range_dimensions_raises(self):
    # [FUNCTION DOC]
    # Test scenario: adds a two-by-three full matrix and a three-by-three diagonal operator
    # (range dimension differs). Asserts a value error saying the operators must share the same
    # range dimension.
    op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3))
    op2 = linalg.LinearOperatorDiag(rng.rand(3, 3))
    with self.assertRaisesRegex(ValueError, "must.*same `range_dimension`"):
      add_operators([op1, op2])

  def test_non_broadcastable_batch_shape_raises(self):
    # [FUNCTION DOC]
    # Test scenario: adds operators with batch shapes two and four, which cannot broadcast.
    # Asserts a value error reporting incompatible shapes.
    op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3))
    op2 = linalg.LinearOperatorDiag(rng.rand(4, 3, 3))
    with self.assertRaisesRegex(ValueError, "Incompatible shapes"):
      add_operators([op1, op2])


class LinearOperatorOrderOfAdditionTest(test.TestCase):
  # [CLASS DOC]
  # Tests that adders are applied tier by tier in the supplied order, using a sentinel adder
  # that fails if its tier is ever consulted. No shared fixtures.
  """Test that the order of addition is done as specified by tiers."""

  def test_tier_0_additions_done_in_tier_0(self):
    # [FUNCTION DOC]
    # Test scenario: three diagonal operators with tiers of a diagonal-returning adder followed
    # by the failing sentinel. Asserts a single diagonal result and no assertion error, so the
    # second tier is never reached.
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([1.])
    diag3 = linalg.LinearOperatorDiag([1.])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnDiag()],
        [_BadAdder()],
    ]
    # Should not raise since all were added in tier 0, and tier 1 (with the
    # _BadAdder) was never reached.
    op_sum = add_operators([diag1, diag2, diag3], addition_tiers=addition_tiers)
    self.assertEqual(1, len(op_sum))
    self.assertIsInstance(op_sum[0], linalg.LinearOperatorDiag)

  def test_tier_1_additions_done_by_tier_1(self):
    # [FUNCTION DOC]
    # Test scenario: two diagonals and a lower-triangular operator with tiers diagonal adder,
    # lower-triangular adder, sentinel. Asserts a single lower-triangular result and that the
    # sentinel tier is never reached.
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([1.])
    tril = linalg.LinearOperatorLowerTriangular([[1.]])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnDiag()],
        [linear_operator_addition._AddAndReturnTriL()],
        [_BadAdder()],
    ]
    # Should not raise since all were added by tier 1, and the
    # _BadAdder) was never reached.
    op_sum = add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)
    self.assertEqual(1, len(op_sum))
    self.assertIsInstance(op_sum[0], linalg.LinearOperatorLowerTriangular)

  def test_tier_1_additions_done_by_tier_1_with_order_flipped(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says tier one does the work, but the code (and its comments) show
    # tier zero handles everything in this ordering.
    # Test scenario: same operators with the lower-triangular adder tier first, then diagonal,
    # then sentinel. The first tier converts everything to lower-triangular. Asserts a single
    # lower-triangular result and that no later tier raises.
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([1.])
    tril = linalg.LinearOperatorLowerTriangular([[1.]])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnTriL()],
        [linear_operator_addition._AddAndReturnDiag()],
        [_BadAdder()],
    ]
    # Tier 0 could convert to TriL, and this converted everything to TriL,
    # including the Diags.
    # Tier 1 was never used.
    # Tier 2 was never used (therefore, _BadAdder didn't raise).
    op_sum = add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)
    self.assertEqual(1, len(op_sum))
    self.assertIsInstance(op_sum[0], linalg.LinearOperatorLowerTriangular)

  @test_util.run_deprecated_v1
  def test_cannot_add_everything_so_return_more_than_one_operator(self):
    # [FUNCTION DOC]
    # Test scenario: two diagonals and a lower-triangular operator with only a diagonal-returning
    # tier. Asserts two operators come back: a diagonal holding the sum of the two diagonals
    # (three) and the untouched lower-triangular operator (five). Graph mode.
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([2.])
    tril5 = linalg.LinearOperatorLowerTriangular([[5.]])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnDiag()],
    ]
    # Tier 0 (the only tier) can only convert to Diag, so it combines the two
    # diags, but the TriL is unchanged.
    # Result should contain two operators, one Diag, one TriL.
    op_sum = add_operators([diag1, diag2, tril5], addition_tiers=addition_tiers)
    self.assertEqual(2, len(op_sum))
    found_diag = False
    found_tril = False
    with self.cached_session():
      for op in op_sum:
        if isinstance(op, linalg.LinearOperatorDiag):
          found_diag = True
          self.assertAllClose([[3.]], op.to_dense())
        if isinstance(op, linalg.LinearOperatorLowerTriangular):
          found_tril = True
          self.assertAllClose([[5.]], op.to_dense())
      self.assertTrue(found_diag and found_tril)

  def test_intermediate_tier_is_not_skipped(self):
    # [FUNCTION DOC]
    # Test scenario: tiers are diagonal adder, failing sentinel, lower-triangular adder, with a
    # lower-triangular operand that tier zero cannot absorb. Asserts an assertion error from the
    # sentinel's can-add check, showing the middle tier is consulted before the third.
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([1.])
    tril = linalg.LinearOperatorLowerTriangular([[1.]])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnDiag()],
        [_BadAdder()],
        [linear_operator_addition._AddAndReturnTriL()],
    ]
    # tril cannot be added in tier 0, and the intermediate tier 1 with the
    # BadAdder will catch it and raise.
    with self.assertRaisesRegex(AssertionError, "BadAdder.can_add called"):
      add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)


class AddAndReturnScaledIdentityTest(test.TestCase):
  # [CLASS DOC]
  # Tests the adder that sums identity and scaled-identity operators into a scaled identity.
  # Fixture: one adder instance stored in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: creates the scaled-identity adder and stores it on the test instance.
    self._adder = linear_operator_addition._AddAndReturnScaledIdentity()

  @test_util.run_deprecated_v1
  def test_identity_plus_identity(self):
    # [FUNCTION DOC]
    # Test scenario: adds an identity and a batched identity (same size) with positive-definite
    # and non-singular hints. Asserts the adder accepts the pair, returns a scaled identity whose
    # dense form is twice the batched identity, carries the hints and the given name. Graph mode.
    id1 = linalg.LinearOperatorIdentity(num_rows=2)
    id2 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(id1, id2))
    operator = self._adder.add(id1, id2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)

    with self.cached_session():
      self.assertAllClose(2 * linalg_ops.eye(num_rows=2, batch_shape=[3]),
                          operator.to_dense())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)

  @test_util.run_deprecated_v1
  def test_identity_plus_scaled_identity(self):
    # [FUNCTION DOC]
    # Test scenario: adds a batched identity and a scaled identity with multiplier 2.2.
    # Asserts the adder accepts the pair and returns a scaled identity equal to 3.2 times the
    # batched identity, with the positive-definite and non-singular hints and given name.
    id1 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
    id2 = linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=2.2)
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(id1, id2))
    operator = self._adder.add(id1, id2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)

    with self.cached_session():
      self.assertAllClose(3.2 * linalg_ops.eye(num_rows=2, batch_shape=[3]),
                          operator.to_dense())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)

  @test_util.run_deprecated_v1
  def test_scaled_identity_plus_scaled_identity(self):
    # [FUNCTION DOC]
    # Test scenario: adds a scaled identity with three multipliers of 2.2 and a scaled identity
    # with multiplier minus one. Asserts the adder accepts the pair and returns a scaled identity
    # equal to 1.2 times the batched identity, with the hints and given name.
    id1 = linalg.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=[2.2, 2.2, 2.2])
    id2 = linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=-1.0)
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(id1, id2))
    operator = self._adder.add(id1, id2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)

    with self.cached_session():
      self.assertAllClose(1.2 * linalg_ops.eye(num_rows=2, batch_shape=[3]),
                          operator.to_dense())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)


class AddAndReturnDiagTest(test.TestCase):
  # [CLASS DOC]
  # Tests the adder that sums identity-like and diagonal operators into a diagonal operator.
  # Fixture: one adder instance stored in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: creates the diagonal-returning adder and stores it on the test instance.
    self._adder = linear_operator_addition._AddAndReturnDiag()

  @test_util.run_deprecated_v1
  def test_identity_plus_identity_returns_diag(self):
    # [FUNCTION DOC]
    # Test scenario: adds an identity and a batched identity with hints. Asserts the adder
    # accepts them and returns a diagonal operator whose dense form is twice the batched
    # identity, with the hints and given name. Graph mode.
    id1 = linalg.LinearOperatorIdentity(num_rows=2)
    id2 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(id1, id2))
    operator = self._adder.add(id1, id2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorDiag)

    with self.cached_session():
      self.assertAllClose(2 * linalg_ops.eye(num_rows=2, batch_shape=[3]),
                          operator.to_dense())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)

  @test_util.run_deprecated_v1
  def test_diag_plus_diag(self):
    # [FUNCTION DOC]
    # Test scenario: adds a batched random diagonal (shape two by three by four) and an unbatched
    # random diagonal of length four, using the shared seeded generator. Asserts the adder
    # accepts them and returns a diagonal operator equal to the broadcast sum, with the hints and
    # given name. Graph mode.
    diag1 = rng.rand(2, 3, 4)
    diag2 = rng.rand(4)
    op1 = linalg.LinearOperatorDiag(diag1)
    op2 = linalg.LinearOperatorDiag(diag2)
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(op1, op2))
    operator = self._adder.add(op1, op2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorDiag)

    with self.cached_session():
      self.assertAllClose(
          linalg.LinearOperatorDiag(diag1 + diag2).to_dense(),
          operator.to_dense())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)


class AddAndReturnTriLTest(test.TestCase):
  # [CLASS DOC]
  # Tests the adder that sums diagonal and lower-triangular operators into a lower-triangular
  # operator. Fixture: one adder instance stored in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: creates the lower-triangular-returning adder and stores it on the test
    # instance.
    self._adder = linear_operator_addition._AddAndReturnTriL()

  @test_util.run_deprecated_v1
  def test_diag_plus_tril(self):
    # [FUNCTION DOC]
    # Test scenario: adds a diagonal operator and a lower-triangular operator. Asserts the adder
    # can add diagonal with diagonal and diagonal with lower-triangular, returns a
    # lower-triangular operator with the expected dense sum, the hints and the given name.
    # Graph mode.
    diag = linalg.LinearOperatorDiag([1., 2.])
    tril = linalg.LinearOperatorLowerTriangular([[10., 0.], [30., 0.]])
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(diag, diag))
    self.assertTrue(self._adder.can_add(diag, tril))
    operator = self._adder.add(diag, tril, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorLowerTriangular)

    with self.cached_session():
      self.assertAllClose([[11., 0.], [30., 2.]], operator.to_dense())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)


class AddAndReturnMatrixTest(test.TestCase):
  # [CLASS DOC]
  # Tests the fallback adder that sums arbitrary operators into a full-matrix operator.
  # Fixture: one adder instance stored in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: creates the full-matrix-returning adder and stores it on the test instance.
    self._adder = linear_operator_addition._AddAndReturnMatrix()

  @test_util.run_deprecated_v1
  def test_diag_plus_diag(self):
    # [FUNCTION DOC]
    # Test scenario: adds two diagonal operators with hints that deny positive definiteness and
    # non-singularity (the sum has a zero entry). Asserts the adder accepts the pair, returns a
    # full-matrix operator with the expected dense sum, the false hints and the given name.
    # Graph mode.
    diag1 = linalg.LinearOperatorDiag([1., 2.])
    diag2 = linalg.LinearOperatorDiag([-1., 3.])
    hints = linear_operator_addition._Hints(
        is_positive_definite=False, is_non_singular=False)

    self.assertTrue(self._adder.can_add(diag1, diag2))
    operator = self._adder.add(diag1, diag2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorFullMatrix)

    with self.cached_session():
      self.assertAllClose([[0., 0.], [0., 5.]], operator.to_dense())
    self.assertFalse(operator.is_positive_definite)
    self.assertFalse(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)


if __name__ == "__main__":
  test.main()
