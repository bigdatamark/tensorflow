# [MODULE DOC]
# Test module for the identity-style linear operators in the linalg package: the identity operator
# and the scaled identity operator. Defines two test classes derived from the shared square
# linear-operator test base class (which supplies the generic matmul, solve, determinant, trace,
# to-dense, etc. checks) plus a number of operator-specific tests for argument validation, flag
# handling, broadcasting and type-preserving operations. When run as a script it registers the
# generated base-class tests for both classes and then starts the test runner. Creates a
# module-level seeded random number generator used to build test inputs.
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
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test


rng = np.random.RandomState(2016)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorIdentityTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the identity linear operator. Inherits the generic square-operator checks from
  # the shared base class and supplies the operator/matrix factory, the dtypes and the optional
  # tests to run. Adds identity-specific tests: assertions, argument validation (static and
  # dynamic), broadcasting against batch matrices, flag defaults and type-preserving operations.
  # Runs every test in both graph and eager modes. Fixture: TensorFloat-32 is turned off for
  # each test and restored afterwards.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting to the value
    # saved in setUp. Takes no inputs, returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: remembers whether TensorFloat-32 execution is currently enabled
    # (stored on the instance) and then disables it globally so numerical comparisons are exact
    # enough. Side effect: changes a process-wide config setting. Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  @staticmethod
  def dtypes_to_test():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the list of data types to test, namely 32-bit and
    # 64-bit real floats and 64-bit and 128-bit complex numbers. Half precision is left out
    # (see the to-do) because the linear solve does not support 16-bit floats.
    # TODO(langmore) Test tf.float16 once tf.linalg.solve works in
    # 16bit.
    return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the names of optional generic tests to enable, here
    # the matmul and solve tests that pass an operator of the same type as the right-hand side.
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory hook used by the base class. Inputs: build_info (object whose shape field gives the
    # full batch-plus-matrix shape, which must be square), dtype of the operator, use_placeholder
    # (unused here since the identity takes no tensor arguments) and
    # ensure_self_adjoint_and_pd (ignored and deleted because the identity is always Hermitian
    # positive definite). Builds an identity operator with the requested number of rows, batch
    # shape and dtype, and the equivalent dense identity matrix. Returns the pair (operator,
    # dense matrix). Raises an assertion error if the last two dimensions differ.
    # Identity matrix is already Hermitian Positive Definite.
    del ensure_self_adjoint_and_pd

    shape = list(build_info.shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    operator = linalg_lib.LinearOperatorIdentity(
        num_rows, batch_shape=batch_shape, dtype=dtype)
    mat = linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=dtype)

    return operator, mat

  def test_assert_positive_definite(self):
    # [FUNCTION DOC]
    # Test: builds a 2-row identity operator and evaluates its positive-definiteness assertion
    # op; expects it to succeed without raising.
    with self.cached_session():
      operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
      self.evaluate(operator.assert_positive_definite())  # Should not fail

  def test_assert_non_singular(self):
    # [FUNCTION DOC]
    # Test: builds a 2-row identity operator and evaluates its non-singularity assertion op;
    # expects it to succeed without raising.
    with self.cached_session():
      operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
      self.evaluate(operator.assert_non_singular())  # Should not fail

  def test_assert_self_adjoint(self):
    # [FUNCTION DOC]
    # Test: builds a 2-row identity operator and evaluates its self-adjointness assertion op;
    # expects it to succeed without raising.
    with self.cached_session():
      operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
      self.evaluate(operator.assert_self_adjoint())  # Should not fail

  def test_float16_matmul(self):
    # [FUNCTION DOC]
    # Test: half-precision matmul, which the base class cannot cover because the linear solve
    # lacks float16 support. Multiplies a 2-row float16 identity operator with a random 2 by 3
    # float16 array and asserts the result is close to the input unchanged.
    # float16 cannot be tested by base test class because tf.linalg.solve does
    # not work with float16.
    with self.cached_session():
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows=2, dtype=dtypes.float16)
      x = rng.randn(2, 3).astype(np.float16)
      y = operator.matmul(x)
      self.assertAllClose(x, self.evaluate(y))

  def test_non_scalar_num_rows_raises_static(self):
    # [FUNCTION DOC]
    # Test: constructing the operator with a list (non-scalar) for the number of rows raises a
    # value error mentioning that it must be a 0-D tensor, detected statically at construction.
    with self.assertRaisesRegex(ValueError, "must be a 0-D Tensor"):
      linalg_lib.LinearOperatorIdentity(num_rows=[2])

  def test_non_integer_num_rows_raises_static(self):
    # [FUNCTION DOC]
    # Test: a floating-point number of rows makes the constructor raise a type error saying the
    # value must be an integer.
    with self.assertRaisesRegex(TypeError, "must be integer"):
      linalg_lib.LinearOperatorIdentity(num_rows=2.)

  def test_negative_num_rows_raises_static(self):
    # [FUNCTION DOC]
    # Test: a negative number of rows makes the constructor raise a value error saying the value
    # must be non-negative.
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorIdentity(num_rows=-2)

  def test_non_1d_batch_shape_raises_static(self):
    # [FUNCTION DOC]
    # Test: a scalar batch shape (instead of a 1-D sequence) makes the constructor raise a value
    # error saying it must be 1-D.
    with self.assertRaisesRegex(ValueError, "must be a 1-D"):
      linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=2)

  def test_non_integer_batch_shape_raises_static(self):
    # [FUNCTION DOC]
    # Test: a batch shape containing a float makes the constructor raise a type error saying the
    # entries must be integers.
    with self.assertRaisesRegex(TypeError, "must be integer"):
      linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=[2.])

  def test_negative_batch_shape_raises_static(self):
    # [FUNCTION DOC]
    # Test: a batch shape containing a negative entry makes the constructor raise a value error
    # saying it must be non-negative.
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=[-2])

  def test_non_scalar_num_rows_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: the number of rows is a placeholder whose value (a one-element list) is only known at
    # run time. With proper-shape assertions enabled, building the operator and densifying it
    # must raise an error stating the value must be a 0-D tensor.
    with self.cached_session():
      num_rows = array_ops.placeholder_with_default([2], shape=None)

      with self.assertRaisesError("must be a 0-D Tensor"):
        operator = linalg_lib.LinearOperatorIdentity(
            num_rows, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_negative_num_rows_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: the number of rows is a placeholder holding a negative value known only at run time.
    # With proper-shape assertions enabled, densifying the operator must raise an error saying
    # the value must be non-negative.
    with self.cached_session():
      num_rows = array_ops.placeholder_with_default(-2, shape=None)
      with self.assertRaisesError("must be non-negative"):
        operator = linalg_lib.LinearOperatorIdentity(
            num_rows, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_non_1d_batch_shape_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: the batch shape is a placeholder holding a scalar known only at run time. With
    # proper-shape assertions enabled, densifying the operator must raise an error saying the
    # batch shape must be 1-D.
    with self.cached_session():
      batch_shape = array_ops.placeholder_with_default(2, shape=None)
      with self.assertRaisesError("must be a 1-D"):
        operator = linalg_lib.LinearOperatorIdentity(
            num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_negative_batch_shape_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: the batch shape is a placeholder with a negative entry known only at run time. With
    # proper-shape assertions enabled, densifying the operator must raise an error saying the
    # batch shape must be non-negative.
    with self.cached_session():
      batch_shape = array_ops.placeholder_with_default([-2], shape=None)
      with self.assertRaisesError("must be non-negative"):
        operator = linalg_lib.LinearOperatorIdentity(
            num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_wrong_matrix_dimensions_raises_static(self):
    # [FUNCTION DOC]
    # Test: multiplying a 2-row identity operator with a 3 by 3 array (statically known
    # incompatible inner dimension) raises a value error about dimensions not being compatible.
    operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegex(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  def test_wrong_matrix_dimensions_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: same mismatch as the static variant but with the number of rows and the right-hand
    # side supplied as placeholders with unknown static shape. With proper-shape assertions
    # enabled, evaluating the matmul must raise an error about incompatible dimensions.
    num_rows = array_ops.placeholder_with_default(2, shape=None)
    x = array_ops.placeholder_with_default(
        rng.rand(3, 3).astype(np.float32), shape=None)

    with self.cached_session():
      with self.assertRaisesError("Dimensions.*not.compatible"):
        operator = linalg_lib.LinearOperatorIdentity(
            num_rows, assert_proper_shapes=True)
        self.evaluate(operator.matmul(x))

  def test_default_batch_shape_broadcasts_with_everything_static(self):
    # [FUNCTION DOC]
    # Test: an identity operator with the default (empty) batch shape applied to a random
    # statically shaped 4-D batch (1, 2, 3, 4) must broadcast with it; the product must have the
    # same static shape as the input and equal the input values. Not covered by the base class
    # because batch matmul does not broadcast.
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session() as sess:
      x = random_ops.random_normal(shape=(1, 2, 3, 4))
      operator = linalg_lib.LinearOperatorIdentity(num_rows=3, dtype=x.dtype)

      operator_matmul = operator.matmul(x)
      expected = x

      self.assertAllEqual(operator_matmul.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

  def test_default_batch_shape_broadcasts_with_everything_dynamic(self):
    # [FUNCTION DOC]
    # Test: like the static variant, but the 4-D input (1, 2, 3, 4) is a placeholder with unknown
    # static shape. The default-batch-shape identity operator times the input must equal the
    # input numerically.
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      x = array_ops.placeholder_with_default(rng.randn(1, 2, 3, 4), shape=None)
      operator = linalg_lib.LinearOperatorIdentity(num_rows=3, dtype=x.dtype)

      operator_matmul = operator.matmul(x)
      expected = x

      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

  def test_broadcast_matmul_static_shapes(self):
    # [FUNCTION DOC]
    # Test: identity operator with batch shape (2, 1) and 3 rows multiplied with a random
    # (1, 2, 3, 4) input; the result must broadcast to shape (2, 2, 3, 4), equal the input
    # broadcast against zeros of that shape, and have a matching static shape.
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session() as sess:
      # Given this x and LinearOperatorIdentity shape of (2, 1, 3, 3), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      x = random_ops.random_normal(shape=(1, 2, 3, 4))
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows=3, batch_shape=(2, 1), dtype=x.dtype)

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)

      # Expected result of matmul and solve.
      expected = x + zeros

      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

  def test_broadcast_matmul_dynamic_shapes(self):
    # [FUNCTION DOC]
    # Test: dynamic-shape counterpart of the static broadcast test. The input, the row count and
    # the batch shape (2, 1) are all placeholders; the float64 identity operator times the
    # (1, 2, 3, 4) input must numerically equal the input broadcast against zeros of shape
    # (2, 2, 3, 4).
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      # Given this x and LinearOperatorIdentity shape of (2, 1, 3, 3), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      x = array_ops.placeholder_with_default(rng.rand(1, 2, 3, 4), shape=None)
      num_rows = array_ops.placeholder_with_default(3, shape=None)
      batch_shape = array_ops.placeholder_with_default((2, 1), shape=None)

      operator = linalg_lib.LinearOperatorIdentity(
          num_rows, batch_shape=batch_shape, dtype=dtypes.float64)

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)

      # Expected result of matmul and solve.
      expected = x + zeros

      operator_matmul = operator.matmul(x)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

  def test_is_x_flags(self):
    # [FUNCTION DOC]
    # Test: the property flags (positive definite, non-singular, self-adjoint) of a default
    # identity operator are all true. Explicitly passing "unknown" for the non-singular flag must
    # raise a value error stating the identity is always non-singular.
    # The is_x flags are by default all True.
    operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)

    # Any of them False raises because the identity is always self-adjoint etc..
    with self.assertRaisesRegex(ValueError, "is always non-singular"):
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows=2,
          is_non_singular=None,
      )

  def test_identity_adjoint_type(self):
    # [FUNCTION DOC]
    # Test: the adjoint of an identity operator is itself an identity operator instance.
    operator = linalg_lib.LinearOperatorIdentity(
        num_rows=2, is_non_singular=True)
    self.assertIsInstance(
        operator.adjoint(), linalg_lib.LinearOperatorIdentity)

  def test_identity_cholesky_type(self):
    # [FUNCTION DOC]
    # Test: the Cholesky factor of an identity operator flagged positive definite and
    # self-adjoint is again an identity operator instance.
    operator = linalg_lib.LinearOperatorIdentity(
        num_rows=2,
        is_positive_definite=True,
        is_self_adjoint=True,
    )
    self.assertIsInstance(
        operator.cholesky(), linalg_lib.LinearOperatorIdentity)

  def test_identity_inverse_type(self):
    # [FUNCTION DOC]
    # Test: the inverse of an identity operator flagged non-singular is again an identity
    # operator instance.
    operator = linalg_lib.LinearOperatorIdentity(
        num_rows=2, is_non_singular=True)
    self.assertIsInstance(
        operator.inverse(), linalg_lib.LinearOperatorIdentity)

  def test_ref_type_shape_args_raises(self):
    # [FUNCTION DOC]
    # Test: passing a mutable variable (reference type) as the number of rows, or as the batch
    # shape, makes the constructor raise a type error whose message names that argument and
    # mentions a reference.
    with self.assertRaisesRegex(TypeError, "num_rows.*reference"):
      linalg_lib.LinearOperatorIdentity(num_rows=variables_module.Variable(2))

    with self.assertRaisesRegex(TypeError, "batch_shape.*reference"):
      linalg_lib.LinearOperatorIdentity(
          num_rows=2, batch_shape=variables_module.Variable([3]))


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorScaledIdentityTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the scaled identity linear operator (identity times a possibly batched scalar
  # multiplier). Inherits generic square-operator checks from the shared base class, supplies the
  # operator/matrix factory, dtypes and optional tests, and adds specific tests for assertions,
  # shape validation, broadcasting of matmul and solve, flags, closure of operations (results
  # staying identity-like), gradient tape safety and variable conversion. Runs in graph and
  # eager modes; TensorFloat-32 is disabled for each test and restored afterwards.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: records the current TensorFloat-32 setting on the instance and
    # disables TensorFloat-32 globally for accurate numerics. Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  @staticmethod
  def dtypes_to_test():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the data types to test (float32, float64, complex64,
    # complex128). Half precision is excluded because the linear solve does not support it.
    # TODO(langmore) Test tf.float16 once tf.linalg.solve works in
    # 16bit.
    return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the names of optional generic tests to run (matmul
    # and solve with an operand of the same operator type).
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory hook used by the base class. Inputs: build_info (its shape gives batch shape plus a
    # square matrix shape), dtype, use_placeholder (when true the multiplier is fed through a
    # placeholder with unknown shape) and ensure_self_adjoint_and_pd (when true the multiplier is
    # made real-valued non-negative and the operator is flagged self-adjoint and positive
    # definite). Draws a random multiplier per batch member with magnitude between 1 and 2 (well
    # conditioned), builds a scaled identity operator and the equivalent dense matrix. Returns
    # (operator, dense matrix). Raises an assertion error if the matrix is not square.

    shape = list(build_info.shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    # Uniform values that are at least length 1 from the origin.  Allows the
    # operator to be well conditioned.
    # Shape batch_shape
    multiplier = linear_operator_test_util.random_sign_uniform(
        shape=batch_shape, minval=1., maxval=2., dtype=dtype)

    if ensure_self_adjoint_and_pd:
      # Abs on complex64 will result in a float32, so we cast back up.
      multiplier = math_ops.cast(math_ops.abs(multiplier), dtype=dtype)

    # Nothing to feed since LinearOperatorScaledIdentity takes no Tensor args.
    lin_op_multiplier = multiplier

    if use_placeholder:
      lin_op_multiplier = array_ops.placeholder_with_default(
          multiplier, shape=None)

    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows,
        lin_op_multiplier,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)

    multiplier_matrix = array_ops.expand_dims(
        array_ops.expand_dims(multiplier, -1), -1)
    matrix = multiplier_matrix * linalg_ops.eye(
        num_rows, batch_shape=batch_shape, dtype=dtype)

    return operator, matrix

  def test_assert_positive_definite_does_not_raise_when_positive(self):
    # [FUNCTION DOC]
    # Test: a scaled identity with a positive multiplier passes its positive-definiteness
    # assertion op without raising.
    with self.cached_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=1.)
      self.evaluate(operator.assert_positive_definite())  # Should not fail

  def test_assert_positive_definite_raises_when_negative(self):
    # [FUNCTION DOC]
    # Test: a scaled identity with a negative multiplier fails its positive-definiteness
    # assertion op with an op error mentioning "not positive definite".
    with self.cached_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=-1.)
      with self.assertRaisesOpError("not positive definite"):
        self.evaluate(operator.assert_positive_definite())

  def test_assert_non_singular_does_not_raise_when_non_singular(self):
    # [FUNCTION DOC]
    # Test: a scaled identity whose batched multipliers are all non-zero passes its
    # non-singularity assertion op without raising.
    with self.cached_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=[1., 2., 3.])
      self.evaluate(operator.assert_non_singular())  # Should not fail

  def test_assert_non_singular_raises_when_singular(self):
    # [FUNCTION DOC]
    # Test: a batched multiplier containing a zero makes the non-singularity assertion op fail
    # with an op error mentioning "was singular".
    with self.cached_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=[1., 2., 0.])
      with self.assertRaisesOpError("was singular"):
        self.evaluate(operator.assert_non_singular())

  def test_assert_self_adjoint_does_not_raise_when_self_adjoint(self):
    # [FUNCTION DOC]
    # Test: a complex multiplier with zero imaginary part is self-adjoint, so the self-adjointness
    # assertion op passes without raising.
    with self.cached_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=[1. + 0J])
      self.evaluate(operator.assert_self_adjoint())  # Should not fail

  def test_assert_self_adjoint_raises_when_not_self_adjoint(self):
    # [FUNCTION DOC]
    # Test: a complex multiplier with a non-zero imaginary part is not self-adjoint, so the
    # assertion op fails with an op error mentioning "not self-adjoint".
    with self.cached_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=[1. + 1J])
      with self.assertRaisesOpError("not self-adjoint"):
        self.evaluate(operator.assert_self_adjoint())

  def test_float16_matmul(self):
    # [FUNCTION DOC]
    # Test: half-precision matmul that the base class cannot cover (the linear solve lacks
    # float16 support). Builds a scaled identity with a random float16 multiplier of batch size
    # 3 and multiplies a random 2 by 3 float16 array; asserts the result equals the input scaled
    # by the multiplier (broadcast over the matrix dimensions).
    # float16 cannot be tested by base test class because tf.linalg.solve does
    # not work with float16.
    with self.cached_session():
      multiplier = rng.rand(3).astype(np.float16)
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=multiplier)
      x = rng.randn(2, 3).astype(np.float16)
      y = operator.matmul(x)
      self.assertAllClose(multiplier[..., None, None] * x, self.evaluate(y))

  def test_non_scalar_num_rows_raises_static(self):
    # [FUNCTION DOC]
    # Test: a list passed as the number of rows makes the constructor raise a value error saying
    # it must be a 0-D tensor. Other row-count validation tests live in the identity test case.
    # Many "test_...num_rows" tests are performed in LinearOperatorIdentity.
    with self.assertRaisesRegex(ValueError, "must be a 0-D Tensor"):
      linalg_lib.LinearOperatorScaledIdentity(
          num_rows=[2], multiplier=123.)

  def test_wrong_matrix_dimensions_raises_static(self):
    # [FUNCTION DOC]
    # Test: multiplying a 2-row scaled identity with a 3 by 3 array raises a value error about
    # dimensions not being compatible (detected from static shapes).
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=2.2)
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegex(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  def test_wrong_matrix_dimensions_raises_dynamic(self):
    # [FUNCTION DOC]
    # Test: dynamic-shape version of the dimension mismatch check. The row count and the
    # right-hand side are placeholders; with proper-shape assertions enabled, evaluating the
    # matmul must raise an error about incompatible dimensions.
    num_rows = array_ops.placeholder_with_default(2, shape=None)
    x = array_ops.placeholder_with_default(
        rng.rand(3, 3).astype(np.float32), shape=None)

    with self.cached_session():
      with self.assertRaisesError("Dimensions.*not.compatible"):
        operator = linalg_lib.LinearOperatorScaledIdentity(
            num_rows,
            multiplier=[1., 2],
            assert_proper_shapes=True)
        self.evaluate(operator.matmul(x))

  def test_broadcast_matmul_and_solve(self):
    # [FUNCTION DOC]
    # Test: a scaled identity with 3 rows and batch multiplier of shape (2, 1) filled with 2.2,
    # applied to a random (1, 2, 3, 4) input. Matmul must equal the input times 2.2 broadcast to
    # shape (2, 2, 3, 4) and solve must equal the input divided by 2.2 broadcast likewise; static
    # shapes of both results must match the expected shape.
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session() as sess:
      # Given this x and LinearOperatorScaledIdentity shape of (2, 1, 3, 3), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      x = random_ops.random_normal(shape=(1, 2, 3, 4))

      # operator is 2.2 * identity (with a batch shape).
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=3, multiplier=2.2 * array_ops.ones((2, 1)))

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)

      # Test matmul
      expected = x * 2.2 + zeros
      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

      # Test solve
      expected = x / 2.2 + zeros
      operator_solve = operator.solve(x)
      self.assertAllEqual(operator_solve.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_solve, expected]))

  def test_broadcast_matmul_and_solve_scalar_scale_multiplier(self):
    # [FUNCTION DOC]
    # Test: a scaled identity with 3 rows and a plain scalar multiplier 2.2 (no batch shape)
    # applied to a random (1, 2, 3, 4) input. Matmul must equal the input times 2.2 and solve
    # must equal the input divided by 2.2, each with the same static shape as the input.
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session() as sess:
      # Given this x and LinearOperatorScaledIdentity shape of (3, 3), the
      # broadcast shape of operator and 'x' is (1, 2, 3, 4), which is the same
      # shape as x.
      x = random_ops.random_normal(shape=(1, 2, 3, 4))

      # operator is 2.2 * identity (with a batch shape).
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=3, multiplier=2.2)

      # Test matmul
      expected = x * 2.2
      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

      # Test solve
      expected = x / 2.2
      operator_solve = operator.solve(x)
      self.assertAllEqual(operator_solve.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_solve, expected]))

  def test_is_x_flags(self):
    # [FUNCTION DOC]
    # Test: with an explicit not-positive-definite flag and an explicit non-singular flag on a
    # real multiplier, the operator reports not positive definite, non-singular, and (set
    # automatically because the multiplier is real) self-adjoint.
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=1.,
        is_positive_definite=False, is_non_singular=True)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)  # Auto-set due to real multiplier

  def test_identity_adjoint_type(self):
    # [FUNCTION DOC]
    # Test: the adjoint of a scaled identity operator is itself a scaled identity operator
    # instance.
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=1., is_non_singular=True)
    self.assertIsInstance(
        operator.adjoint(), linalg_lib.LinearOperatorScaledIdentity)

  def test_identity_matmul(self):
    # [FUNCTION DOC]
    # Test: closure of matmul among identity operators. Identity times identity stays an identity
    # operator; scaled identity times scaled identity stays a scaled identity; mixing an identity
    # with a scaled identity (either order) yields a scaled identity whose multiplier is the
    # scaled one (3). Note the identity-times-identity check is repeated twice verbatim.
    operator1 = linalg_lib.LinearOperatorIdentity(num_rows=2)
    operator2 = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=3.)
    self.assertIsInstance(
        operator1.matmul(operator1),
        linalg_lib.LinearOperatorIdentity)

    self.assertIsInstance(
        operator1.matmul(operator1),
        linalg_lib.LinearOperatorIdentity)

    self.assertIsInstance(
        operator2.matmul(operator2),
        linalg_lib.LinearOperatorScaledIdentity)

    operator_matmul = operator1.matmul(operator2)
    self.assertIsInstance(
        operator_matmul,
        linalg_lib.LinearOperatorScaledIdentity)
    self.assertAllClose(3., self.evaluate(operator_matmul.multiplier))

    operator_matmul = operator2.matmul(operator1)
    self.assertIsInstance(
        operator_matmul,
        linalg_lib.LinearOperatorScaledIdentity)
    self.assertAllClose(3., self.evaluate(operator_matmul.multiplier))

  def test_identity_solve(self):
    # [FUNCTION DOC]
    # Test: closure of solve among identity operators. Identity solved against identity stays an
    # identity; scaled against scaled stays scaled. Identity solved against a scaled identity
    # (multiplier 3) yields a scaled identity with multiplier 3; scaled (3) solved against
    # identity yields a scaled identity with multiplier one third.
    operator1 = linalg_lib.LinearOperatorIdentity(num_rows=2)
    operator2 = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=3.)
    self.assertIsInstance(
        operator1.solve(operator1),
        linalg_lib.LinearOperatorIdentity)

    self.assertIsInstance(
        operator2.solve(operator2),
        linalg_lib.LinearOperatorScaledIdentity)

    operator_solve = operator1.solve(operator2)
    self.assertIsInstance(
        operator_solve,
        linalg_lib.LinearOperatorScaledIdentity)
    self.assertAllClose(3., self.evaluate(operator_solve.multiplier))

    operator_solve = operator2.solve(operator1)
    self.assertIsInstance(
        operator_solve,
        linalg_lib.LinearOperatorScaledIdentity)
    self.assertAllClose(1. / 3., self.evaluate(operator_solve.multiplier))

  def test_scaled_identity_cholesky_type(self):
    # [FUNCTION DOC]
    # Test: the Cholesky factor of a scaled identity flagged positive definite and self-adjoint
    # is again a scaled identity operator instance.
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2,
        multiplier=3.,
        is_positive_definite=True,
        is_self_adjoint=True,
    )
    self.assertIsInstance(
        operator.cholesky(),
        linalg_lib.LinearOperatorScaledIdentity)

  def test_scaled_identity_inverse_type(self):
    # [FUNCTION DOC]
    # Test: the inverse of a scaled identity flagged non-singular is again a scaled identity
    # operator instance.
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2,
        multiplier=3.,
        is_non_singular=True,
    )
    self.assertIsInstance(
        operator.inverse(),
        linalg_lib.LinearOperatorScaledIdentity)

  def test_ref_type_shape_args_raises(self):
    # [FUNCTION DOC]
    # Test: passing a mutable variable (reference type) as the number of rows makes the
    # constructor raise a type error naming the num_rows argument and mentioning a reference.
    with self.assertRaisesRegex(TypeError, "num_rows.*reference"):
      linalg_lib.LinearOperatorScaledIdentity(
          num_rows=variables_module.Variable(2), multiplier=1.23)

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test: builds a scaled identity whose multiplier is a trainable variable and runs the shared
    # gradient-tape safety check on it, verifying operator methods are differentiable with
    # respect to the variable.
    multiplier = variables_module.Variable(1.23)
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=multiplier)
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Test: builds a scaled identity whose multiplier is a variable, initializes the variable in
    # a session, then runs the shared check that converting the operator's variables to tensors
    # produces an equivalent operator.
    multiplier = variables_module.Variable(1.23)
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=multiplier)
    with self.cached_session() as sess:
      sess.run([multiplier.initializer])
      self.check_convert_variables_to_tensors(operator)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorIdentityTest)
  linear_operator_test_util.add_tests(LinearOperatorScaledIdentityTest)
  test.main()
