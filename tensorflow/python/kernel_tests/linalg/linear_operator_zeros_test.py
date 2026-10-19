# [MODULE DOC]
# Tests for the all-zeros linear operator (LinearOperatorZeros). Defines a square-operator test
# class and a non-square-operator test class built on the generic linear-operator test harness,
# plus extra tests of the operator's assertions, argument validation (static and dynamic shape
# paths), shape-compatibility errors, default property flags, matmul result type and rejection of
# reference-typed shape arguments. Creates a module-level seeded random generator (seed 2016).
# When run as a script it registers the harness-generated tests and starts the test runner.
#
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

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test


rng = np.random.RandomState(2016)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorZerosTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test class for the square all-zeros operator, run in graph and eager modes. Inherits the
  # generic square-operator tests, skips those meaningless for a zero (singular) operator, and
  # adds tests for assertions, argument validation and flags.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static hook for the harness: returns the list of generic test names to skip because they
    # need an invertible operator (cholesky, cond, inverse, log absolute determinant, solve and
    # solve with broadcast). No inputs, no side effects.
    return [
        "cholesky",
        "cond",
        "inverse",
        "log_abs_det",
        "solve",
        "solve_with_broadcast"
    ]

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Static hook for the harness: returns shape descriptions for square operators: 1x1, batch of
    # one 3x3, batch of three 4x4 and batch (2,1) of 4x4. No inputs, no side effects.
    shapes_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shapes_info((1, 1)),
        shapes_info((1, 3, 3)),
        shapes_info((3, 4, 4)),
        shapes_info((2, 1, 4, 4))]

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Static hook for the harness: returns the optional generic tests to enable; here only the
    # matmul-with-same-operator-type test. No inputs, no side effects.
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Harness hook: builds a zeros operator and the equivalent dense zero matrix for a square
    # shape. Inputs: build_info (shape description), dtype, use_placeholder and
    # ensure_self_adjoint_and_pd (both ignored). Returns a pair of operator and zero matrix.
    # Raises an assertion error if the last two dimensions of the shape differ.
    del ensure_self_adjoint_and_pd
    del use_placeholder
    shape = list(build_info.shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    operator = linalg_lib.LinearOperatorZeros(
        num_rows, batch_shape=batch_shape, dtype=dtype)
    matrix = array_ops.zeros(shape=shape, dtype=dtype)

    return operator, matrix

  def test_assert_positive_definite(self):
    # [FUNCTION DOC]
    # Test: asking a 2x2 zeros operator to assert positive definiteness raises an op error
    # mentioning "non-positive definite".
    operator = linalg_lib.LinearOperatorZeros(num_rows=2)
    with self.assertRaisesOpError("non-positive definite"):
      operator.assert_positive_definite()

  def test_assert_non_singular(self):
    # [FUNCTION DOC]
    # Test: asserting non-singularity of a 2x2 zeros operator raises an op error mentioning
    # "non-invertible" (construction happens inside the expected-error scope too).
    with self.assertRaisesOpError("non-invertible"):
      operator = linalg_lib.LinearOperatorZeros(num_rows=2)
      operator.assert_non_singular()

  def test_assert_self_adjoint(self):
    # [FUNCTION DOC]
    # Test: the self-adjointness assertion of a 2x2 zeros operator evaluates without failing.
    with self.cached_session():
      operator = linalg_lib.LinearOperatorZeros(num_rows=2)
      self.evaluate(operator.assert_self_adjoint())  # Should not fail

  def test_non_scalar_num_rows_raises_static(self):
    # [FUNCTION DOC]
    # Test: a list (non-scalar) given as number of rows or number of columns makes the
    # constructor raise a value error saying it must be a 0-D tensor.
    with self.assertRaisesRegex(ValueError, "must be a 0-D Tensor"):
      linalg_lib.LinearOperatorZeros(num_rows=[2])
    with self.assertRaisesRegex(ValueError, "must be a 0-D Tensor"):
      linalg_lib.LinearOperatorZeros(num_rows=2, num_columns=[2])

  def test_non_integer_num_rows_raises_static(self):
    # [FUNCTION DOC]
    # Test: a floating-point number of rows or columns makes the constructor raise a type error
    # saying it must be integer.
    with self.assertRaisesRegex(TypeError, "must be integer"):
      linalg_lib.LinearOperatorZeros(num_rows=2.)
    with self.assertRaisesRegex(TypeError, "must be integer"):
      linalg_lib.LinearOperatorZeros(num_rows=2, num_columns=2.)

  def test_negative_num_rows_raises_static(self):
    # [FUNCTION DOC]
    # Test: a negative number of rows or columns makes the constructor raise a value error saying
    # it must be non-negative.
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorZeros(num_rows=-2)
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorZeros(num_rows=2, num_columns=-2)

  def test_non_1d_batch_shape_raises_static(self):
    # [FUNCTION DOC]
    # Test: a scalar batch shape makes the constructor raise a value error saying it must be 1-D.
    with self.assertRaisesRegex(ValueError, "must be a 1-D"):
      linalg_lib.LinearOperatorZeros(num_rows=2, batch_shape=2)

  def test_non_integer_batch_shape_raises_static(self):
    # [FUNCTION DOC]
    # Test: a batch shape with a floating-point entry makes the constructor raise a type error
    # saying it must be integer.
    with self.assertRaisesRegex(TypeError, "must be integer"):
      linalg_lib.LinearOperatorZeros(num_rows=2, batch_shape=[2.])

  def test_negative_batch_shape_raises_static(self):
    # [FUNCTION DOC]
    # Test: a batch shape with a negative entry makes the constructor raise a value error saying
    # it must be non-negative.
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorZeros(num_rows=2, batch_shape=[-2])

  def test_non_scalar_num_rows_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: number of rows supplied as a placeholder holding a one-element list, with proper-shape
    # assertions enabled; materializing the dense matrix fails with an error saying it must be a
    # 0-D tensor.
    with self.cached_session():
      num_rows = array_ops.placeholder_with_default([2], shape=None)
      with self.assertRaisesError("must be a 0-D Tensor"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_negative_num_rows_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: number of rows supplied as a placeholder holding negative two, with proper-shape
    # assertions enabled; materializing the dense matrix fails with an error saying it must be
    # non-negative.
    with self.cached_session():
      n = array_ops.placeholder_with_default(-2, shape=None)
      with self.assertRaisesError("must be non-negative"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows=n, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_non_1d_batch_shape_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: batch shape supplied as a scalar placeholder with proper-shape assertions enabled;
    # materializing the dense matrix fails with an error saying it must be 1-D.
    with self.cached_session():
      batch_shape = array_ops.placeholder_with_default(2, shape=None)
      with self.assertRaisesError("must be a 1-D"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_negative_batch_shape_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: batch shape supplied as a placeholder containing a negative entry with proper-shape
    # assertions enabled; materializing the dense matrix fails with a non-negative error.
    with self.cached_session():
      batch_shape = array_ops.placeholder_with_default([-2], shape=None)
      with self.assertRaisesError("must be non-negative"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_wrong_matrix_dimensions_raises_static(self):
    # [FUNCTION DOC]
    # Test: multiplying a 2x2 zeros operator by a static 3x3 array raises a value error about
    # incompatible dimensions.
    operator = linalg_lib.LinearOperatorZeros(num_rows=2)
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegex(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  def test_wrong_matrix_dimensions_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: with a placeholder row count of 2 (float64, proper-shape assertions on) and a
    # placeholder 3x3 right-hand side, evaluating the product fails with an incompatible
    # dimensions error.
    num_rows = array_ops.placeholder_with_default(2, shape=None)
    x = array_ops.placeholder_with_default(rng.rand(3, 3), shape=None)

    with self.cached_session():
      with self.assertRaisesError("Dimensions.*not.compatible"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows, assert_proper_shapes=True, dtype=dtypes.float64)
        self.evaluate(operator.matmul(x))

  def test_is_x_flags(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment says the flags default to all true, but the test asserts
    # that positive-definite and non-singular are false and only self-adjoint is true.
    # Test: default hints of a zeros operator: not positive definite, not non-singular, and
    # self-adjoint.
    # The is_x flags are by default all True.
    operator = linalg_lib.LinearOperatorZeros(num_rows=2)
    self.assertFalse(operator.is_positive_definite)
    self.assertFalse(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)

  def test_zeros_matmul(self):
    # [FUNCTION DOC]
    # Test: composing a zeros operator with an identity operator, in either order, yields a
    # zeros operator instance (not a generic composition).
    operator1 = linalg_lib.LinearOperatorIdentity(num_rows=2)
    operator2 = linalg_lib.LinearOperatorZeros(num_rows=2)
    self.assertTrue(isinstance(
        operator1.matmul(operator2),
        linalg_lib.LinearOperatorZeros))

    self.assertTrue(isinstance(
        operator2.matmul(operator1),
        linalg_lib.LinearOperatorZeros))

  def test_ref_type_shape_args_raises(self):
    # [FUNCTION DOC]
    # Test: passing a variable as number of rows, number of columns or batch shape makes the
    # constructor raise a type error saying the argument cannot be a reference type.
    with self.assertRaisesRegex(TypeError, "num_rows.cannot.be.reference"):
      linalg_lib.LinearOperatorZeros(num_rows=variables_module.Variable(2))

    with self.assertRaisesRegex(TypeError, "num_columns.cannot.be.reference"):
      linalg_lib.LinearOperatorZeros(
          num_rows=2, num_columns=variables_module.Variable(3))

    with self.assertRaisesRegex(TypeError, "batch_shape.cannot.be.reference"):
      linalg_lib.LinearOperatorZeros(
          num_rows=2, batch_shape=variables_module.Variable([2]))


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorZerosNotSquareTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test class for the non-square all-zeros operator, run in graph and eager modes; inherits the
  # generic non-square operator tests from the harness.

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Harness hook: builds a non-square zeros operator (explicit rows and columns, flagged
    # non-square and not self-adjoint) and the equivalent dense zero matrix. Inputs: build_info,
    # dtype, and two ignored arguments (use_placeholder, ensure_self_adjoint_and_pd). Returns a
    # pair of operator and zero matrix.
    del use_placeholder
    del ensure_self_adjoint_and_pd
    shape = list(build_info.shape)

    batch_shape = shape[:-2]
    num_rows = shape[-2]
    num_columns = shape[-1]

    operator = linalg_lib.LinearOperatorZeros(
        num_rows, num_columns, is_square=False, is_self_adjoint=False,
        batch_shape=batch_shape, dtype=dtype)
    matrix = array_ops.zeros(shape=shape, dtype=dtype)

    return operator, matrix


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorZerosTest)
  linear_operator_test_util.add_tests(LinearOperatorZerosNotSquareTest)
  test.main()
