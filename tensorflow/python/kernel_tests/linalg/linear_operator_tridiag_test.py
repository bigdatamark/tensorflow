# [MODULE DOC]
# Tests for the tridiagonal linear operator (LinearOperatorTridiag). Defines a shared base mixin that
# builds random tridiagonal operators plus their dense matrices, and three concrete test classes,
# one per supported diagonals format (compact, sequence, matrix), derived from the generic square
# linear-operator test harness. Each class runs the harness tests in graph and eager modes with
# TF32 disabled and adds gradient-tape-safety checks. When run as a main program (and not under
# XLA) it registers the generated harness tests for each class and then starts the test runner.
# No import-time side effects beyond imports.
#
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

from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test


class _LinearOperatorTriDiagBase(object):
  # [CLASS DOC]
  # HELPER: shared mixin for the three tridiagonal-operator test classes in this file.
  # Stateless mixin that supplies the operator factory and the list of shapes used by the generic
  # linear-operator test harness. Concrete test classes combine it with the square derived-class
  # test harness and pick the diagonals format.

  def build_operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False,
      diagonals_format='sequence'):
    # [FUNCTION DOC]
    # HELPER: builds the operator under test and its dense reference matrix for the test harness.
    # Inputs: build_info (shape description; its shape gives batch and matrix dimensions), dtype
    # (element type), use_placeholder (if true, inputs are wrapped as placeholders with default
    # values and unknown static shape), ensure_self_adjoint_and_pd (default false; if true the
    # diagonal is made real-valued positive and the super-diagonal is the conjugate of the
    # sub-diagonal shifted by one so the matrix is self-adjoint and positive definite, and the
    # operator is flagged accordingly), diagonals_format (one of sequence, compact or matrix;
    # default sequence).
    # Behavior: draws random diagonals (main in 4..6 magnitude, off-diagonals in 1..2), stacks
    # them into a dense banded matrix, then passes the diagonals in the requested format to the
    # tridiagonal operator constructor.
    # Returns a pair: the operator and the dense matrix. Raises no explicit errors; an unknown
    # format would leave the diagonals variable unset and fail with an unbound-name error.
    shape = list(build_info.shape)

    # Ensure that diagonal has large enough values. If we generate a
    # self adjoint PD matrix, then the diagonal will be dominant guaranteeing
    # positive definitess.
    diag = linear_operator_test_util.random_sign_uniform(
        shape[:-1], minval=4., maxval=6., dtype=dtype)
    # We'll truncate these depending on the format
    subdiag = linear_operator_test_util.random_sign_uniform(
        shape[:-1], minval=1., maxval=2., dtype=dtype)
    if ensure_self_adjoint_and_pd:
      # Abs on complex64 will result in a float32, so we cast back up.
      diag = math_ops.cast(math_ops.abs(diag), dtype=dtype)
      # The first element of subdiag is ignored. We'll add a dummy element
      # to superdiag to pad it.
      superdiag = math_ops.conj(subdiag)
      superdiag = manip_ops.roll(superdiag, shift=-1, axis=-1)
    else:
      superdiag = linear_operator_test_util.random_sign_uniform(
          shape[:-1], minval=1., maxval=2., dtype=dtype)

    matrix_diagonals = array_ops_stack.stack(
        [superdiag, diag, subdiag], axis=-2)
    matrix = gen_array_ops.matrix_diag_v3(
        matrix_diagonals,
        k=(-1, 1),
        num_rows=-1,
        num_cols=-1,
        align='LEFT_RIGHT',
        padding_value=0.)

    if diagonals_format == 'sequence':
      diagonals = [superdiag, diag, subdiag]
    elif diagonals_format == 'compact':
      diagonals = array_ops_stack.stack([superdiag, diag, subdiag], axis=-2)
    elif diagonals_format == 'matrix':
      diagonals = matrix

    lin_op_diagonals = diagonals

    if use_placeholder:
      if diagonals_format == 'sequence':
        lin_op_diagonals = [array_ops.placeholder_with_default(
            d, shape=None) for d in lin_op_diagonals]
      else:
        lin_op_diagonals = array_ops.placeholder_with_default(
            lin_op_diagonals, shape=None)

    operator = linalg_lib.LinearOperatorTridiag(
        diagonals=lin_op_diagonals,
        diagonals_format=diagonals_format,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)
    return operator, matrix

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Static method returning the list of operator shape descriptions the harness iterates over:
    # a single 3x3 matrix and batched cases with batch shapes (1), (3) and (2, 1) with matrix
    # sizes 6, 4 and 3. Takes no inputs, no side effects.
    shape_info = linear_operator_test_util.OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shape_info((3, 3)),
        shape_info((1, 6, 6)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 3, 3))
    ]


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagCompactTest(
    _LinearOperatorTriDiagBase,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test class for the tridiagonal operator built from a single compact tensor whose second-to-last
  # dimension holds the super-, main and sub-diagonals. Inherits the generic square-operator
  # tests; runs in graph and eager modes with TF32 turned off per test (saved in an attribute).
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Restores the TF32 execution setting that setUp saved. No inputs, returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Remembers whether TF32 execution is currently enabled, then globally disables it so numeric
    # comparisons are exact enough. Side effect: changes global TF32 config until tearDown.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Harness hook: delegates to the shared builder with the compact diagonals format. Inputs are
    # the shape info, dtype, placeholder flag and optional self-adjoint/positive-definite flag
    # (default false). Returns the operator and its dense matrix.
    return self.build_operator_and_matrix(
        build_info, dtype, use_placeholder,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        diagonals_format='compact')

  @test_util.disable_xla('Current implementation does not yet support pivoting')
  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test: builds a compact-format tridiagonal operator from a trainable variable (3x3 values)
    # and asserts, via the harness check, that operator methods work under a gradient tape and
    # yield gradients with respect to the variable. Disabled under XLA.
    diag = variables_module.Variable([[3., 6., 2.], [2., 4., 2.], [5., 1., 2.]])
    operator = linalg_lib.LinearOperatorTridiag(
        diag, diagonals_format='compact')
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Test: creates the same compact operator from a variable, initializes the variable in a
    # cached session, then runs the harness check that converting the operator's variables to
    # tensors works and yields an equivalent operator.
    diag = variables_module.Variable([[3., 6., 2.], [2., 4., 2.], [5., 1., 2.]])
    operator = linalg_lib.LinearOperatorTridiag(
        diag, diagonals_format='compact')
    with self.cached_session() as sess:
      sess.run([diag.initializer])
      self.check_convert_variables_to_tensors(operator)


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagSequenceTest(
    _LinearOperatorTriDiagBase,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test class for the tridiagonal operator built from a sequence of three separate tensors
  # (super-, main, sub-diagonal). Inherits the generic square-operator tests; runs in graph and
  # eager modes with TF32 turned off per test (saved in an attribute).
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Restores the TF32 execution setting that setUp saved. No inputs, returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Remembers whether TF32 execution is currently enabled, then globally disables it.
    # Side effect: changes global TF32 config until tearDown.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Harness hook: delegates to the shared builder with the sequence diagonals format. Inputs are
    # the shape info, dtype, placeholder flag and optional self-adjoint/positive-definite flag
    # (default false). Returns the operator and its dense matrix.
    return self.build_operator_and_matrix(
        build_info, dtype, use_placeholder,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        diagonals_format='sequence')

  @test_util.disable_xla('Current implementation does not yet support pivoting')
  def test_tape_safe(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second operator (one variable among plain lists) is constructed but never
    # passed to any check, so that half of the test asserts nothing; the comment mentions testing
    # the middle variable "below" but nothing follows.
    # Test: builds a sequence-format operator from three variables and runs the harness tape-safety
    # check while skipping the diagonal-part and trace options (they only depend on the middle
    # variable). Then only constructs a second operator with just the middle diagonal a variable.
    diagonals = [
        variables_module.Variable([3., 6., 2.]),
        variables_module.Variable([2., 4., 2.]),
        variables_module.Variable([5., 1., 2.])]
    operator = linalg_lib.LinearOperatorTridiag(
        diagonals, diagonals_format='sequence')
    # Skip the diagonal part and trace since this only dependent on the
    # middle variable. We test this below.
    self.check_tape_safe(operator, skip_options=['diag_part', 'trace'])

    diagonals = [
        [3., 6., 2.],
        variables_module.Variable([2., 4., 2.]),
        [5., 1., 2.]
    ]
    operator = linalg_lib.LinearOperatorTridiag(
        diagonals, diagonals_format='sequence')


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagMatrixTest(
    _LinearOperatorTriDiagBase,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test class for the tridiagonal operator built from a full dense (banded) matrix. Inherits the
  # generic square-operator tests; runs in graph and eager modes with TF32 turned off per test.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Restores the TF32 execution setting that setUp saved. No inputs, returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Remembers whether TF32 execution is currently enabled, then globally disables it.
    # Side effect: changes global TF32 config until tearDown.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Harness hook: delegates to the shared builder with the matrix diagonals format. Inputs are
    # the shape info, dtype, placeholder flag and optional self-adjoint/positive-definite flag
    # (default false). Returns the operator and its dense matrix.
    return self.build_operator_and_matrix(
        build_info, dtype, use_placeholder,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        diagonals_format='matrix')

  @test_util.disable_xla('Current implementation does not yet support pivoting')
  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test: builds a matrix-format tridiagonal operator from a trainable 3x3 variable and runs the
    # harness check that operator methods are safe under a gradient tape. Disabled under XLA.
    matrix = variables_module.Variable([[3., 2., 0.], [1., 6., 4.], [0., 2, 2]])
    operator = linalg_lib.LinearOperatorTridiag(
        matrix, diagonals_format='matrix')
    self.check_tape_safe(operator)


if __name__ == '__main__':
  if not test_util.is_xla_enabled():
    linear_operator_test_util.add_tests(LinearOperatorTriDiagCompactTest)
    linear_operator_test_util.add_tests(LinearOperatorTriDiagSequenceTest)
    linear_operator_test_util.add_tests(LinearOperatorTriDiagMatrixTest)
  test.main()
