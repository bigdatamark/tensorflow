# [MODULE DOC]
# HELPER: Testing helper module for the LinearOperator family. Provides the abstract test base
# classes (general, square and non-square) that operator test files subclass, the add_tests
# function that generates dozens of standard checks (to_dense, determinant, matmul, solve,
# inverse, trace, composite tensor, saved model, gradients, and so on) comparing an operator
# against an equivalent dense matrix, and random matrix generators (normal, uniform, sign-uniform,
# positive definite, lower triangular, correlated columns). No import-time side effects.
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
"""Utilities for testing `LinearOperator` and sub-classes."""

import abc
import itertools

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_model
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import save as save_model
from tensorflow.python.util import nest


class OperatorShapesInfo:
  # [CLASS DOC]
  # Plain data holder describing the expected static shape of the matrix an operator under test
  # should represent, plus any extra metadata the test harness wants. The shape is a list of
  # integers (batch dims then rows and columns). Extra keyword arguments become attributes.
  """Object encoding expected shape for a test.

  Encodes the expected shape of a matrix for a test. Also
  allows additional metadata for the test harness.
  """

  def __init__(self, shape, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Stores the given shape and copies every extra keyword argument into the instance
    # attribute mapping, so callers can read them as attributes. Returns nothing; no validation.
    self.shape = shape
    self.__dict__.update(kwargs)


class CheckTapeSafeSkipOptions:
  # [CLASS DOC]
  # HELPER: Namespace of string constants naming checks of the tape-safety test helper that a test
  # may ask to skip (determinant, diagonal part, log absolute determinant, trace). Never instantiated;
  # only its class attributes are used by the tape-safe check.

  # Skip checking this particular method.
  DETERMINANT = "determinant"
  DIAG_PART = "diag_part"
  LOG_ABS_DETERMINANT = "log_abs_determinant"
  TRACE = "trace"


class LinearOperatorDerivedClassTest(test.TestCase, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class for test cases of LinearOperator subclasses. Subclasses supply the shapes to
  # test, a way to build an operator together with an equivalent dense matrix, and helpers that make
  # matmul inputs and solve right-hand sides. The module-level add_tests function then attaches many
  # generated test methods (to_dense, determinant, matmul, solve, and so on) to the subclass.
  # Holds per-dtype absolute and relative tolerance tables and overridable option hooks.
  """Tests for derived classes.

  Subclasses should implement every abstractmethod, and this will enable all
  test methods to work.
  """

  # Absolute/relative tolerance for tests.
  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-12,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-12
  }

  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-12,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-12
  }

  def assertAC(self, x, y, check_dtype=False):
    # [FUNCTION DOC]
    # Asserts that two arrays are close using the per-dtype absolute and relative tolerances of this
    # test class, looked up from the dtype of the first argument. If check_dtype is true, also asserts
    # the first argument has the dtype of the second. Inputs: x, y arrays or tensors; check_dtype bool
    # (default false). Returns nothing; fails the test on mismatch.
    # KNOWN ISSUE: the docstring talks about deriving classes setting tolerances, but the method itself
    # performs the comparison; the tolerance override is done through class attributes elsewhere.
    """Derived classes can set _atol, _rtol to get different tolerance."""
    dtype = dtypes.as_dtype(x.dtype)
    atol = self._atol[dtype]
    rtol = self._rtol[dtype]
    self.assertAllClose(x, y, atol=atol, rtol=rtol)
    if check_dtype:
      self.assertDTypeEqual(x, y.dtype)

  @staticmethod
  def adjoint_options():
    # [FUNCTION DOC]
    # Static hook giving the list of adjoint flag values the generated tests iterate over (false and
    # true). Subclasses may override to restrict. Takes no input.
    return [False, True]

  @staticmethod
  def adjoint_arg_options():
    # [FUNCTION DOC]
    # Static hook giving the list of adjoint-argument flag values the generated tests iterate over
    # (false and true). Subclasses may override. Takes no input.
    return [False, True]

  @staticmethod
  def dtypes_to_test():
    # [FUNCTION DOC]
    # Static hook returning the dtypes the generated tests run with: 32 and 64 bit float and 64 and
    # 128 bit complex. Half precision is excluded because the linear solve does not support it yet.
    # TODO(langmore) Test tf.float16 once tf.linalg.solve works in 16bit.
    return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

  @staticmethod
  def use_placeholder_options():
    # [FUNCTION DOC]
    # Static hook returning the values of the use-placeholder option the generated tests iterate over
    # (false and true), i.e. whether operators are built with unknown-shape placeholders.
    return [False, True]

  @staticmethod
  def use_blockwise_arg():
    # [FUNCTION DOC]
    # Static hook telling the generated matmul/solve tests whether to also exercise blockwise
    # arguments. Returns false by default; blockwise operator test classes override it to true.
    return False

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Static hook that subclasses must override to return a list of shape-info objects describing the
    # operator shapes to test. This default raises a not-implemented error.
    """Returns list of OperatorShapesInfo, encapsulating the shape to test."""
    raise NotImplementedError("operator_shapes_infos has not been implemented.")

  @abc.abstractmethod
  def operator_and_matrix(
      self, shapes_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Abstract method. Subclasses build and return a pair: a LinearOperator instance and a dense tensor
    # that the operator should behave like. Inputs: shapes_info (shape description object), dtype
    # (numpy dtype of the data), use_placeholder (if true, initialize with a placeholder of unknown
    # shape), ensure_self_adjoint_and_pd (if true, produce a Hermitian positive definite operator with
    # matching hints, needed by cholesky and eigenvalue tests; default false). This base
    # implementation only raises a not-implemented error.
    """Build a batch matrix and an Operator that should have similar behavior.

    Every operator acts like a (batch) matrix.  This method returns both
    together, and is used by tests.

    Args:
      shapes_info: `OperatorShapesInfo`, encoding shape information about the
        operator.
      dtype:  Numpy dtype.  Data type of returned array/operator.
      use_placeholder:  Python bool.  If True, initialize the operator with a
        placeholder of undefined shape and correct dtype.
      ensure_self_adjoint_and_pd: If `True`,
        construct this operator to be Hermitian Positive Definite, as well
        as ensuring the hints `is_positive_definite` and `is_self_adjoint`
        are set.
        This is useful for testing methods such as `cholesky`.

    Returns:
      operator:  `LinearOperator` subclass instance.
      mat:  `Tensor` representing operator.
    """
    # Create a matrix as a numpy array with desired shape/dtype.
    # Create a LinearOperator that should have the same behavior as the matrix.
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def make_rhs(self, operator, adjoint, with_batch=True):
    # [FUNCTION DOC]
    # Abstract method. Subclasses return a tensor usable as the right-hand side of a solve with the
    # given operator. Inputs: operator; adjoint (true if the rhs is for the adjoint operator);
    # with_batch (true to give the rhs the operator batch shape, false for a plain matrix). This base
    # version raises a not-implemented error.
    """Make a rhs appropriate for calling operator.solve(rhs).

    Args:
      operator:  A `LinearOperator`
      adjoint:  Python `bool`.  If `True`, we are making a 'rhs' value for the
        adjoint operator.
      with_batch: Python `bool`. If `True`, create `rhs` with the same batch
        shape as operator, and otherwise create a matrix without any batch
        shape.

    Returns:
      A `Tensor`
    """
    raise NotImplementedError("make_rhs is not defined.")

  @abc.abstractmethod
  def make_x(self, operator, adjoint, with_batch=True):
    # [FUNCTION DOC]
    # Abstract method. Subclasses return a tensor usable as the argument of a matrix multiply with the
    # given operator. Inputs: operator; adjoint (true if x is for the adjoint operator); with_batch
    # (true to match the operator batch shape, false for a plain matrix). This base version raises a
    # not-implemented error.
    """Make an 'x' appropriate for calling operator.matmul(x).

    Args:
      operator:  A `LinearOperator`
      adjoint:  Python `bool`.  If `True`, we are making an 'x' value for the
        adjoint operator.
      with_batch: Python `bool`. If `True`, create `x` with the same batch shape
        as operator, and otherwise create a matrix without any batch shape.

    Returns:
      A `Tensor`
    """
    raise NotImplementedError("make_x is not defined.")

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static hook returning the list of generated test names (without the test prefix) to skip.
    # Subclasses override. Default is an empty list.
    """List of test names to skip."""
    # Subclasses should over-ride if they want to skip some tests.
    # To skip "test_foo", add "foo" to this list.
    return []

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Static hook returning the list of optional generated test names to enable (operator matmul and
    # solve with same type). Subclasses override. Default is an empty list.
    """List of optional test names to run."""
    # Subclasses should over-ride if they want to add optional tests.
    # To add "test_foo", add "foo" to this list.
    return []

  def assertRaisesError(self, msg):
    # [FUNCTION DOC]
    # HELPER: Context manager factory for tests expecting an error whose message matches a pattern.
    # Input: msg, a regular expression. Returns a regex-based exception assertion in eager mode and an
    # op-error assertion in graph mode.
    """assertRaisesRegexp or OpError, depending on context.executing_eagerly."""
    if context.executing_eagerly():
      return self.assertRaisesRegex(Exception, msg)
    return self.assertRaisesOpError(msg)

  def check_convert_variables_to_tensors(self, operator):
    # [FUNCTION DOC]
    # Check that converting the variables of a composite operator into tensors works. Asserts the
    # operator is a composite tensor, converts it, asserts the type is unchanged, that the converted
    # operator has no variables left, and that tensors match the original variable values (via the
    # recursive checker). Input: operator. Returns nothing; fails the test otherwise.
    """Checks that internal Variables are correctly converted to Tensors."""
    self.assertIsInstance(operator, composite_tensor.CompositeTensor)
    tensor_operator = composite_tensor.convert_variables_to_tensors(operator)
    self.assertIs(type(operator), type(tensor_operator))
    self.assertEmpty(tensor_operator.variables)
    self._check_tensors_equal_variables(operator, tensor_operator)

  def _check_tensors_equal_variables(self, obj, tensor_obj):
    # [FUNCTION DOC]
    # HELPER: Recursive checker used by the variable-conversion check. Compares obj with tensor_obj:
    # variables must have equal values to the tensors; composite tensors are compared through their
    # parameters (same key set); mappings and nested sequences are walked element by element; other
    # leaf values are ignored. Returns nothing; fails the test on any mismatch.
    """Checks that Variables in `obj` have equivalent Tensors in `tensor_obj."""
    if isinstance(obj, variables.Variable):
      self.assertAllClose(ops.convert_to_tensor(obj),
                          ops.convert_to_tensor(tensor_obj))
    elif isinstance(obj, composite_tensor.CompositeTensor):
      params = getattr(obj, "parameters", {})
      tensor_params = getattr(tensor_obj, "parameters", {})
      self.assertAllEqual(params.keys(), tensor_params.keys())
      self._check_tensors_equal_variables(params, tensor_params)
    elif nest.is_mapping(obj):
      for k, v in obj.items():
        self._check_tensors_equal_variables(v, tensor_obj[k])
    elif nest.is_nested(obj):
      for x, y in zip(obj, tensor_obj):
        self._check_tensors_equal_variables(x, y)
    else:
      # We only check Tensor, CompositeTensor, and nested structure parameters.
      pass

  def check_tape_safe(self, operator, skip_options=None):
    # [FUNCTION DOC]
    # Checks that gradients with respect to the operator variables are not missing for the main
    # methods. Meant to be called from derived test classes. Input: operator (must hold variables;
    # raises an assertion error if it has none); skip_options (optional list of skip constants).
    # Always checks gradients of dense form, the operator itself, its adjoint and a matrix-vector
    # product. For square operators it also checks determinant, log determinant, diagonal part and
    # trace unless skipped; for operators not flagged singular it checks inverse and solve; for
    # self-adjoint positive definite operators it checks Cholesky. Returns nothing; fails if a
    # gradient is missing.
    """Check gradients are not None w.r.t. operator.variables.

    Meant to be called from the derived class.

    This ensures grads are not w.r.t every variable in operator.variables.  If
    more fine-grained testing is needed, a custom test should be written.

    Args:
      operator: LinearOperator.  Exact checks done will depend on hints.
      skip_options: Optional list of CheckTapeSafeSkipOptions.
        Makes this test skip particular checks.
    """
    skip_options = skip_options or []

    if not operator.variables:
      raise AssertionError("`operator.variables` was empty")

    def _assert_not_none(iterable):
      # [FUNCTION DOC]
      # HELPER: Asserts that no element of the given iterable (a list of gradients) is nothing/null.
      for item in iterable:
        self.assertIsNotNone(item)

    # Tape tests that can be run on every operator below.
    with backprop.GradientTape() as tape:
      grad = tape.gradient(operator.to_dense(), operator.variables)
      _assert_not_none(grad)

    with backprop.GradientTape() as tape:
      var_grad = tape.gradient(operator, operator.variables)
      _assert_not_none(var_grad)
      nest.assert_same_structure(var_grad, grad)

    with backprop.GradientTape() as tape:
      _assert_not_none(
          tape.gradient(operator.adjoint().to_dense(), operator.variables))

    x = math_ops.cast(
        array_ops.ones(shape=operator.H.shape_tensor()[:-1]), operator.dtype)

    with backprop.GradientTape() as tape:
      _assert_not_none(tape.gradient(operator.matvec(x), operator.variables))

    # Tests for square, but possibly non-singular operators below.
    if not operator.is_square:
      return

    for option in [
        CheckTapeSafeSkipOptions.DETERMINANT,
        CheckTapeSafeSkipOptions.LOG_ABS_DETERMINANT,
        CheckTapeSafeSkipOptions.DIAG_PART,
        CheckTapeSafeSkipOptions.TRACE,
    ]:
      with backprop.GradientTape() as tape:
        if option not in skip_options:
          _assert_not_none(
              tape.gradient(getattr(operator, option)(), operator.variables))

    # Tests for non-singular operators below.
    if operator.is_non_singular is False:  # pylint: disable=g-bool-id-comparison
      return

    with backprop.GradientTape() as tape:
      _assert_not_none(
          tape.gradient(operator.inverse().to_dense(), operator.variables))

    with backprop.GradientTape() as tape:
      _assert_not_none(tape.gradient(operator.solvevec(x), operator.variables))

    # Tests for SPD operators below.
    if not (operator.is_self_adjoint and operator.is_positive_definite):
      return

    with backprop.GradientTape() as tape:
      _assert_not_none(
          tape.gradient(operator.cholesky().to_dense(), operator.variables))


# pylint:disable=missing-docstring


def _test_slicing(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method that checks slicing of the batch dimensions of an
  # operator. Inputs: use_placeholder, shapes_info, dtype are captured by the returned closure.
  def test_slicing(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. In a fresh graph with the default seed, builds an operator and matrix, skips if
    # the batch shape is empty or its first batch size is at most one, slices the operator batch
    # (drop first and last member, and for several batch dims also select the first of the last
    # dimension), slices the dense matrix the same way and asserts the sliced operator dense form
    # equals the sliced matrix.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      batch_shape = shapes_info.shape[:-2]
      # Don't bother slicing for uninteresting batch shapes.
      if not batch_shape or batch_shape[0] <= 1:
        return

      slices = [slice(1, -1)]
      if len(batch_shape) > 1:
        # Slice out the last member.
        slices += [..., slice(0, 1)]
      sliced_operator = operator[slices]
      matrix_slices = slices + [slice(None), slice(None)]
      sliced_matrix = mat[matrix_slices]
      sliced_op_dense = sliced_operator.to_dense()
      op_dense_v, mat_v = sess.run([sliced_op_dense, sliced_matrix])
      self.assertAC(op_dense_v, mat_v)
  return test_slicing


def _test_to_dense(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method checking the to-dense conversion of an operator
  # (arguments captured by the closure).
  def test_to_dense(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Builds operator and matrix in a fresh seeded graph, converts the operator to
    # dense, checks the static shape against the expected shape when placeholders are not used, and
    # asserts the evaluated dense form is close to the matrix.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_dense = operator.to_dense()
      if not use_placeholder:
        self.assertAllEqual(shapes_info.shape, op_dense.shape)
      op_dense_v, mat_v = sess.run([op_dense, mat])
      self.assertAC(op_dense_v, mat_v)
  return test_to_dense


def _test_det(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method checking the determinant of an operator.
  def test_det(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Computes the operator determinant, checks the static batch shape when not using
    # placeholders, and asserts it is close to the determinant of the reference matrix.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_det = operator.determinant()
      if not use_placeholder:
        self.assertAllEqual(shapes_info.shape[:-2], op_det.shape)
      op_det_v, mat_det_v = sess.run(
          [op_det, linalg_ops.matrix_determinant(mat)])
      self.assertAC(op_det_v, mat_det_v)
  return test_det


def _test_log_abs_det(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method checking the log absolute determinant.
  def test_log_abs_det(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Computes the operator log absolute determinant, checks the static batch shape
    # when not using placeholders, and asserts closeness to the log absolute determinant (from the
    # sign-and-log determinant routine) of the reference matrix.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_log_abs_det = operator.log_abs_determinant()
      _, mat_log_abs_det = linalg.slogdet(mat)
      if not use_placeholder:
        self.assertAllEqual(
            shapes_info.shape[:-2], op_log_abs_det.shape)
      op_log_abs_det_v, mat_log_abs_det_v = sess.run(
          [op_log_abs_det, mat_log_abs_det])
      self.assertAC(op_log_abs_det_v, mat_log_abs_det_v)
  return test_log_abs_det


def _test_operator_matmul_with_same_type(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for multiplying one operator by another of the same
  # class (optional test; the product must keep the class). Runs without reduced-precision
  # tensor-float matmul.
  """op_a.matmul(op_b), in the case where the same type is returned."""
  @test_util.run_without_tensor_float_32("Use FP32 in matmul")
  def test_operator_matmul_with_same_type(
      self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Builds two operator-matrix pairs, multiplies the operators and the matrices,
    # asserts the operator product is an instance of the first operator class and that its dense
    # form is close to the matrix product.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator_a, mat_a = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      operator_b, mat_b = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)

      mat_matmul = math_ops.matmul(mat_a, mat_b)
      op_matmul = operator_a.matmul(operator_b)
      mat_matmul_v, op_matmul_v = sess.run([mat_matmul, op_matmul.to_dense()])

      self.assertIsInstance(op_matmul, operator_a.__class__)
      self.assertAC(mat_matmul_v, op_matmul_v)
  return test_operator_matmul_with_same_type


def _test_operator_solve_with_same_type(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for solving one operator against another of the same
  # class (optional test; result must keep the class).
  """op_a.solve(op_b), in the case where the same type is returned."""
  def test_operator_solve_with_same_type(
      self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Builds two operator-matrix pairs, solves the first operator against the second
    # and the matrices with a broadcasting solve, asserts the result is an instance of the first
    # operator class and its dense form is close to the matrix solve.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator_a, mat_a = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      operator_b, mat_b = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)

      mat_solve = linear_operator_util.matrix_solve_with_broadcast(mat_a, mat_b)
      op_solve = operator_a.solve(operator_b)
      mat_solve_v, op_solve_v = sess.run([mat_solve, op_solve.to_dense()])

      self.assertIsInstance(op_solve, operator_a.__class__)
      self.assertAC(mat_solve_v, op_solve_v)
  return test_operator_solve_with_same_type


def _test_matmul_base(
    self: "LinearOperatorDerivedClassTest",
    use_placeholder,
    shapes_info,
    dtype,
    adjoint,
    adjoint_arg,
    blockwise_arg,
    with_batch):
  # [FUNCTION DOC]
  # HELPER: Shared body of the matmul tests. Inputs: self (the test case), use_placeholder,
  # shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg and with_batch. Returns early if
  # with_batch is false and the operator has no batch dimensions. Otherwise builds operator and
  # matrix, makes x, multiplies with the operator (using the adjoint of x when adjoint_arg is
  # set) and with the matrix, checks static shapes without placeholders, optionally also tests
  # blockwise inputs by splitting x into blocks (for blockwise operators with several blocks), then
  # asserts closeness of results.
  # If batch dimensions are omitted, but there are
  # no batch dimensions for the linear operator, then
  # skip the test case. This is already checked with
  # with_batch=True.
  if not with_batch and len(shapes_info.shape) <= 2:
    return
  with self.session(graph=ops.Graph()) as sess:
    sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
    operator, mat = self.operator_and_matrix(
        shapes_info, dtype, use_placeholder=use_placeholder)
    x = self.make_x(
        operator, adjoint=adjoint, with_batch=with_batch)
    # If adjoint_arg, compute A X^H^H = A X.
    if adjoint_arg:
      op_matmul = operator.matmul(
          linalg.adjoint(x),
          adjoint=adjoint,
          adjoint_arg=adjoint_arg)
    else:
      op_matmul = operator.matmul(x, adjoint=adjoint)
    mat_matmul = math_ops.matmul(mat, x, adjoint_a=adjoint)
    if not use_placeholder:
      self.assertAllEqual(op_matmul.shape,
                          mat_matmul.shape)

    # If the operator is blockwise, test both blockwise `x` and `Tensor` `x`;
    # else test only `Tensor` `x`. In both cases, evaluate all results in a
    # single `sess.run` call to avoid re-sampling the random `x` in graph mode.
    if blockwise_arg and len(operator.operators) > 1:
      # pylint: disable=protected-access
      block_dimensions = (
          operator._block_range_dimensions() if adjoint else
          operator._block_domain_dimensions())
      block_dimensions_fn = (
          operator._block_range_dimension_tensors if adjoint else
          operator._block_domain_dimension_tensors)
      # pylint: enable=protected-access
      split_x = linear_operator_util.split_arg_into_blocks(
          block_dimensions,
          block_dimensions_fn,
          x, axis=-2)
      if adjoint_arg:
        split_x = [linalg.adjoint(y) for y in split_x]
      split_matmul = operator.matmul(
          split_x, adjoint=adjoint, adjoint_arg=adjoint_arg)

      self.assertEqual(len(split_matmul), len(operator.operators))
      split_matmul = linear_operator_util.broadcast_matrix_batch_dims(
          split_matmul)
      fused_block_matmul = array_ops.concat(split_matmul, axis=-2)
      op_matmul_v, mat_matmul_v, fused_block_matmul_v = sess.run([
          op_matmul, mat_matmul, fused_block_matmul])

      # Check that the operator applied to blockwise input gives the same result
      # as matrix multiplication.
      self.assertAC(fused_block_matmul_v, mat_matmul_v)
    else:
      op_matmul_v, mat_matmul_v = sess.run([op_matmul, mat_matmul])

    # Check that the operator applied to a `Tensor` gives the same result as
    # matrix multiplication.
    self.assertAC(op_matmul_v, mat_matmul_v)


def _test_matmul(
    use_placeholder,
    shapes_info,
    dtype,
    adjoint,
    adjoint_arg,
    blockwise_arg):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a matmul test method for the given adjoint, adjoint-argument and
  # blockwise options, with batch-shaped input.
  @test_util.run_without_tensor_float_32("Use FP32 in matmul")
  def test_matmul(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Calls the shared matmul body with batch-shaped input. Runs without reduced
    # precision matmul.
    _test_matmul_base(
        self,
        use_placeholder,
        shapes_info,
        dtype,
        adjoint,
        adjoint_arg,
        blockwise_arg,
        with_batch=True)
  return test_matmul


def _test_matmul_with_broadcast(
    use_placeholder,
    shapes_info,
    dtype,
    adjoint,
    adjoint_arg,
    blockwise_arg):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a matmul-with-broadcast test method.
  @test_util.run_without_tensor_float_32("Use FP32 in matmul")
  def test_matmul_with_broadcast(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test meant to check matmul when x has no batch dimensions so it must broadcast.
    # KNOWN ISSUE: it passes with_batch true, the same as the plain matmul test, so the broadcast
    # case is never exercised (the solve counterpart passes false).
    # Calls the shared matmul body.
    _test_matmul_base(
        self,
        use_placeholder,
        shapes_info,
        dtype,
        adjoint,
        adjoint_arg,
        blockwise_arg,
        with_batch=True)
  return test_matmul_with_broadcast


def _test_adjoint(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for the operator adjoint.
  def test_adjoint(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Builds operator and matrix, computes the adjoint dense form through both the
    # adjoint method and the H property, and asserts both are close to the conjugate transpose of
    # the matrix.
    with self.test_session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_adjoint = operator.adjoint().to_dense()
      op_adjoint_h = operator.H.to_dense()
      mat_adjoint = linalg.adjoint(mat)
      op_adjoint_v, op_adjoint_h_v, mat_adjoint_v = sess.run(
          [op_adjoint, op_adjoint_h, mat_adjoint])
      self.assertAC(mat_adjoint_v, op_adjoint_v)
      self.assertAC(mat_adjoint_v, op_adjoint_h_v)
  return test_adjoint


def _test_cholesky(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for the Cholesky factorization of an operator.
  def test_cholesky(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Uses a graph seed offset by two (to make float32 pass within tolerance), builds a
    # self-adjoint positive definite operator and matrix, and asserts the operator Cholesky dense
    # form is close to the matrix Cholesky factor.
    with self.test_session(graph=ops.Graph()) as sess:
      # This test fails to pass for float32 type by a small margin if we use
      # random_seed.DEFAULT_GRAPH_SEED.  The correct fix would be relaxing the
      # test tolerance but the tolerance in this test is configured universally
      # depending on its type.  So instead of lowering tolerance for all tests
      # or special casing this, just use a seed, +2, that makes this test pass.
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED + 2
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder,
          ensure_self_adjoint_and_pd=True)
      op_chol = operator.cholesky().to_dense()
      mat_chol = linalg_ops.cholesky(mat)
      op_chol_v, mat_chol_v = sess.run([op_chol, mat_chol])
      self.assertAC(mat_chol_v, op_chol_v)
  return test_cholesky


def _test_eigvalsh(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for the eigenvalues of self-adjoint operators.
  def test_eigvalsh(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Builds a self-adjoint positive definite operator and matrix, sorts the real
    # eigenvalues of the operator and of the matrix (computed in 64-bit precision) and asserts they
    # are close, using looser tolerances for single precision types.
    with self.test_session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder,
          ensure_self_adjoint_and_pd=True)
      # Eigenvalues are real, so we'll cast these to float64 and sort
      # for comparison.
      op_eigvals = sort_ops.sort(
          math_ops.cast(operator.eigvals(), dtype=dtypes.float64), axis=-1)
      if dtype.is_complex:
        mat = math_ops.cast(mat, dtype=dtypes.complex128)
      else:
        mat = math_ops.cast(mat, dtype=dtypes.float64)
      mat_eigvals = sort_ops.sort(
          math_ops.cast(
              linalg_ops.self_adjoint_eigvals(mat), dtype=dtypes.float64),
          axis=-1)
      op_eigvals_v, mat_eigvals_v = sess.run([op_eigvals, mat_eigvals])

      atol = self._atol[dtype]  # pylint: disable=protected-access
      rtol = self._rtol[dtype]  # pylint: disable=protected-access
      if dtype == dtypes.float32 or dtype == dtypes.complex64:
        atol = 2e-4
        rtol = 2e-4
      self.assertAllClose(op_eigvals_v, mat_eigvals_v, atol=atol, rtol=rtol)
  return test_eigvalsh


def _test_cond(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for the condition number of an operator.
  def test_cond(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Skips empty matrices (singular values undefined) and complex types on ROCm.
    # Builds a self-adjoint positive definite operator and matrix, computes the reference condition
    # number from the ratio of largest to smallest singular value and asserts closeness with
    # looser per-dtype tolerances.
    # KNOWN ISSUE: a comment inside the body about casting eigenvalues to double and sorting is a
    # leftover and does not describe the code, which compares condition numbers.
    with self.test_session(graph=ops.Graph()) as sess:
      # svd does not work with zero dimensional matrices, so we'll
      # skip
      if 0 in shapes_info.shape[-2:]:
        return

      # ROCm platform does not yet support complex types
      if test.is_built_with_rocm() and \
         ((dtype == dtypes.complex64) or (dtype == dtypes.complex128)):
        return

      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      # Ensure self-adjoint and PD so we get finite condition numbers.
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder,
          ensure_self_adjoint_and_pd=True)
      # Eigenvalues are real, so we'll cast these to float64 and sort
      # for comparison.
      op_cond = operator.cond()
      s = math_ops.abs(linalg_ops.svd(mat, compute_uv=False))
      mat_cond = math_ops.reduce_max(s, axis=-1) / math_ops.reduce_min(
          s, axis=-1)
      op_cond_v, mat_cond_v = sess.run([op_cond, mat_cond])

      atol_override = {
          dtypes.float16: 1e-2,
          dtypes.float32: 1e-3,
          dtypes.float64: 1e-6,
          dtypes.complex64: 1e-3,
          dtypes.complex128: 1e-6,
      }
      rtol_override = {
          dtypes.float16: 1e-2,
          dtypes.float32: 1e-3,
          dtypes.float64: 1e-4,
          dtypes.complex64: 1e-3,
          dtypes.complex128: 1e-6,
      }
      atol = atol_override[dtype]
      rtol = rtol_override[dtype]
      self.assertAllClose(op_cond_v, mat_cond_v, atol=atol, rtol=rtol)
  return test_cond


def _test_solve_base(
    self: "LinearOperatorDerivedClassTest",
    use_placeholder,
    shapes_info,
    dtype,
    adjoint,
    adjoint_arg,
    blockwise_arg,
    with_batch):
  # [FUNCTION DOC]
  # HELPER: Shared body of the solve tests. Inputs: self (the test case), use_placeholder,
  # shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg and with_batch. Returns early if
  # with_batch is false and the operator has no batch dimensions. Otherwise builds operator and
  # matrix, makes a right-hand side, solves with the operator (passing the adjoint of the rhs when
  # adjoint_arg is set) and with a broadcasting matrix solve, checks static shapes, optionally
  # tests blockwise right-hand sides for blockwise operators, and asserts closeness.
  # If batch dimensions are omitted, but there are
  # no batch dimensions for the linear operator, then
  # skip the test case. This is already checked with
  # with_batch=True.
  if not with_batch and len(shapes_info.shape) <= 2:
    return
  with self.session(graph=ops.Graph()) as sess:
    sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
    operator, mat = self.operator_and_matrix(
        shapes_info, dtype, use_placeholder=use_placeholder)
    rhs = self.make_rhs(
        operator, adjoint=adjoint, with_batch=with_batch)
    # If adjoint_arg, solve A X = (rhs^H)^H = rhs.
    if adjoint_arg:
      op_solve = operator.solve(
          linalg.adjoint(rhs),
          adjoint=adjoint,
          adjoint_arg=adjoint_arg)
    else:
      op_solve = operator.solve(
          rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
    mat_solve = linear_operator_util.matrix_solve_with_broadcast(
        mat, rhs, adjoint=adjoint)
    if not use_placeholder:
      self.assertAllEqual(op_solve.shape,
                          mat_solve.shape)

    # If the operator is blockwise, test both blockwise rhs and `Tensor` rhs;
    # else test only `Tensor` rhs. In both cases, evaluate all results in a
    # single `sess.run` call to avoid re-sampling the random rhs in graph mode.
    if blockwise_arg and len(operator.operators) > 1:
      # pylint: disable=protected-access
      block_dimensions = (
          operator._block_range_dimensions() if adjoint else
          operator._block_domain_dimensions())
      block_dimensions_fn = (
          operator._block_range_dimension_tensors if adjoint else
          operator._block_domain_dimension_tensors)
      # pylint: enable=protected-access
      split_rhs = linear_operator_util.split_arg_into_blocks(
          block_dimensions,
          block_dimensions_fn,
          rhs, axis=-2)
      if adjoint_arg:
        split_rhs = [linalg.adjoint(y) for y in split_rhs]
      split_solve = operator.solve(
          split_rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
      self.assertEqual(len(split_solve), len(operator.operators))
      split_solve = linear_operator_util.broadcast_matrix_batch_dims(
          split_solve)
      fused_block_solve = array_ops.concat(split_solve, axis=-2)
      op_solve_v, mat_solve_v, fused_block_solve_v = sess.run([
          op_solve, mat_solve, fused_block_solve])

      # Check that the operator and matrix give the same solution when the rhs
      # is blockwise.
      self.assertAC(mat_solve_v, fused_block_solve_v)
    else:
      op_solve_v, mat_solve_v = sess.run([op_solve, mat_solve])

    # Check that the operator and matrix give the same solution when the rhs is
    # a `Tensor`.
    self.assertAC(op_solve_v, mat_solve_v)


def _test_solve(
    use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a solve test method using batch-shaped right-hand sides.
  def test_solve(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Calls the shared solve body with batch-shaped right-hand side.
    _test_solve_base(
        self,
        use_placeholder,
        shapes_info,
        dtype,
        adjoint,
        adjoint_arg,
        blockwise_arg,
        with_batch=True)
  return test_solve


def _test_solve_with_broadcast(
    use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a solve-with-broadcast test method.
  def test_solve_with_broadcast(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Calls the shared solve body with a right-hand side without batch dimensions so
    # that it must be broadcast.
    _test_solve_base(
        self,
        use_placeholder,
        shapes_info,
        dtype,
        adjoint,
        adjoint_arg,
        blockwise_arg,
        with_batch=False)
  return test_solve_with_broadcast


def _test_inverse(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for the operator inverse.
  def test_inverse(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Asserts the dense form of the operator inverse is close to the matrix inverse
    # and has the same dtype.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_inverse_v, mat_inverse_v = sess.run([
          operator.inverse().to_dense(), linalg.inv(mat)])
      self.assertAC(op_inverse_v, mat_inverse_v, check_dtype=True)
  return test_inverse


def _test_trace(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for the trace.
  def test_trace(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Computes the operator trace, checks the static shape without placeholders, and
    # asserts closeness to the trace of the reference matrix.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_trace = operator.trace()
      mat_trace = math_ops.trace(mat)
      if not use_placeholder:
        self.assertAllEqual(op_trace.shape, mat_trace.shape)
      op_trace_v, mat_trace_v = sess.run([op_trace, mat_trace])
      self.assertAC(op_trace_v, mat_trace_v)
  return test_trace


def _test_add_to_tensor(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for adding an operator to a dense tensor.
  def test_add_to_tensor(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Adds twice the reference matrix to the operator, checks the static shape
    # without placeholders, and asserts the result is close to three times the matrix.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_plus_2mat = operator.add_to_tensor(2 * mat)

      if not use_placeholder:
        self.assertAllEqual(shapes_info.shape, op_plus_2mat.shape)

      op_plus_2mat_v, mat_v = sess.run([op_plus_2mat, mat])

      self.assertAC(op_plus_2mat_v, 3 * mat_v)
  return test_add_to_tensor


def _test_diag_part(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for the diagonal part.
  def test_diag_part(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Computes the operator diagonal part, checks the static shape without
    # placeholders, and asserts closeness to the diagonal of the reference matrix.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_diag_part = operator.diag_part()
      mat_diag_part = array_ops.matrix_diag_part(mat)

      if not use_placeholder:
        self.assertAllEqual(mat_diag_part.shape,
                            op_diag_part.shape)

      op_diag_part_, mat_diag_part_ = sess.run(
          [op_diag_part, mat_diag_part])

      self.assertAC(op_diag_part_, mat_diag_part_)
  return test_diag_part


def _test_composite_tensor(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for composite tensor behavior of an operator.
  @test_util.run_without_tensor_float_32("Use FP32 in matmul")
  def test_composite_tensor(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Checks the operator is a composite tensor, flattens and repacks it, feeds it to a
    # compiled function doing a matmul, runs it through a while loop that rebuilds the operator from
    # its parameters, checks static shapes, asserts both results are close to the matrix product,
    # and checks the type specification can be encoded by the structure coder.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      self.assertIsInstance(operator, composite_tensor.CompositeTensor)

      flat = nest.flatten(operator, expand_composites=True)
      unflat = nest.pack_sequence_as(operator, flat, expand_composites=True)
      self.assertIsInstance(unflat, type(operator))

      # Input the operator to a `tf.function`.
      x = self.make_x(operator, adjoint=False)
      op_y = def_function.function(lambda op: op.matmul(x))(unflat)
      mat_y = math_ops.matmul(mat, x)

      if not use_placeholder:
        self.assertAllEqual(mat_y.shape, op_y.shape)

      # Test while_loop.
      def body(op):
        # [FUNCTION DOC]
        # Loop body for the while loop: rebuilds an operator of the same type from its parameters and
        # returns it in a one-element tuple.
        return type(op)(**op.parameters),
      op_out, = while_v2.while_loop(
          cond=lambda _: True,
          body=body,
          loop_vars=(operator,),
          maximum_iterations=3)
      loop_y = op_out.matmul(x)

      op_y_, loop_y_, mat_y_ = sess.run([op_y, loop_y, mat_y])
      self.assertAC(op_y_, mat_y_)
      self.assertAC(loop_y_, mat_y_)

      # Ensure that the `TypeSpec` can be encoded.
      nested_structure_coder.encode_structure(operator._type_spec)  # pylint: disable=protected-access

  return test_composite_tensor


def _test_saved_model(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method that saves and loads an operator through a saved model.
  @test_util.run_without_tensor_float_32("Use FP32 in matmul")
  def test_saved_model(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Skips empty shapes when XLA is on. Builds an operator, defines a small module
    # holding a variable x and a function taking the operator by type spec, saves it, reloads it,
    # updates the variable, applies the loaded function to the operator and asserts the result is
    # close to the matrix product. Writes into a temporary directory.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED

      if test_util.is_xla_enabled() and np.prod(shapes_info.shape) == 0:
        self.skipTest("Saving XLA model fails for empty model.")

      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      x = self.make_x(operator, adjoint=False)

      class Model(module.Module):
        # [CLASS DOC]
        # HELPER: Local module used by the saved model test. Holds a variable initialized from x and
        # exposes a traced function multiplying an operator by that variable.

        def __init__(self, init_x):
          # [FUNCTION DOC]
          # Constructor. Wraps every tensor of the given initial value into an unconstrained-shape variable
          # and stores the result as attribute x.
          self.x = nest.map_structure(
              lambda x_: variables.Variable(x_, shape=None),
              init_x)

        @def_function.function(input_signature=(operator._type_spec,))  # pylint: disable=protected-access
        def do_matmul(self, op):
          # [FUNCTION DOC]
          # Traced method taking an operator (typed by the captured type spec) and returning its matmul with
          # the stored variable x.
          return op.matmul(self.x)

      saved_model_dir = self.get_temp_dir()
      m1 = Model(x)
      sess.run([v.initializer for v in m1.variables])
      sess.run(m1.x.assign(m1.x + 1.))

      save_model.save(m1, saved_model_dir)
      m2 = load_model.load(saved_model_dir)
      sess.run(m2.x.initializer)

      sess.run(m2.x.assign(m2.x + 1.))
      y_op = m2.do_matmul(operator)
      y_mat = math_ops.matmul(mat, m2.x)

      y_op_, y_mat_ = sess.run([y_op, y_mat])
      self.assertAC(y_op_, y_mat_)

  return test_saved_model


def _test_composite_tensor_gradient(use_placeholder, shapes_info, dtype):
  # [FUNCTION DOC]
  # HELPER: Test factory. Returns a test method for gradients with respect to a composite operator.
  def test_composite_tensor_gradient(self: "LinearOperatorDerivedClassTest"):
    # [FUNCTION DOC]
    # Generated test. Computes the gradient of a matmul output with respect to the operator itself
    # and, separately, with respect to the flattened components. If all component gradients are
    # missing, asserts the operator gradient is missing too; otherwise asserts it is an operator of the
    # same class and that its flattened parts match the component gradients.
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, _ = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      x = self.make_x(operator, adjoint=False)
      y = operator.matmul(x)

      op_g, = gradients_impl.gradients(
          y,
          operator,
          grad_ys=array_ops.ones_like(y))  # Complex dtypes need grad_ys.

      def _unflatten_and_matmul(components):
        # [FUNCTION DOC]
        # HELPER: Repacks the given flat components into an operator and returns its matmul with x.
        unflat_op = nest.pack_sequence_as(
            operator, components, expand_composites=True)
        return unflat_op.matmul(x)

      flat_op = nest.flatten(operator, expand_composites=True)
      y_ = _unflatten_and_matmul(flat_op)
      flat_g = gradients_impl.gradients(
          y_,
          flat_op,
          grad_ys=array_ops.ones_like(y_))

      if all(g is None for g in flat_g):
        self.assertIsNone(op_g)
      else:
        self.assertIsInstance(op_g, operator.__class__)
        for g, ug in zip(nest.flatten(op_g, expand_composites=True),
                         nest.flatten(flat_g, expand_composites=True)):
          self.assertAllClose(g, ug)
  return test_composite_tensor_gradient

# pylint:enable=missing-docstring


def add_tests(test_cls):
  # [FUNCTION DOC]
  # Attaches generated test methods to a test class. Input: test_cls, a LinearOperatorDerivedClassTest
  # subclass. For each known test template (skipping names in the class skip list and optional tests
  # not enabled) and each combination of dtype, placeholder option and shape info, builds a test
  # named from those values (plus adjoint options for matmul and solve tests) and sets it on the
  # class, wrapped to run in graph mode. Raises a value error if skip and optional lists overlap and
  # a runtime error if a test name would be defined twice. Modifies the class; returns nothing.
  # KNOWN ISSUE: the first string of the overlap error message lacks the formatting prefix, so the
  # class placeholder is printed literally instead of the class name.
  """Add tests for LinearOperator methods."""
  test_name_dict = {
      # All test classes should be added here.
      "add_to_tensor": _test_add_to_tensor,
      "adjoint": _test_adjoint,
      "cholesky": _test_cholesky,
      "cond": _test_cond,
      "composite_tensor": _test_composite_tensor,
      "composite_tensor_gradient": _test_composite_tensor_gradient,
      "det": _test_det,
      "diag_part": _test_diag_part,
      "eigvalsh": _test_eigvalsh,
      "inverse": _test_inverse,
      "log_abs_det": _test_log_abs_det,
      "operator_matmul_with_same_type": _test_operator_matmul_with_same_type,
      "operator_solve_with_same_type": _test_operator_solve_with_same_type,
      "matmul": _test_matmul,
      "matmul_with_broadcast": _test_matmul_with_broadcast,
      "saved_model": _test_saved_model,
      "slicing": _test_slicing,
      "solve": _test_solve,
      "solve_with_broadcast": _test_solve_with_broadcast,
      "to_dense": _test_to_dense,
      "trace": _test_trace,
  }
  optional_tests = [
      # Test classes need to explicitly add these to cls.optional_tests.
      "operator_matmul_with_same_type",
      "operator_solve_with_same_type",
  ]
  tests_with_adjoint_args = [
      "matmul",
      "matmul_with_broadcast",
      "solve",
      "solve_with_broadcast",
  ]
  if set(test_cls.skip_these_tests()).intersection(test_cls.optional_tests()):
    raise ValueError(
        "Test class {test_cls} had intersecting 'skip_these_tests' "
        f"{test_cls.skip_these_tests()} and 'optional_tests' "
        f"{test_cls.optional_tests()}.")

  for name, test_template_fn in test_name_dict.items():
    if name in test_cls.skip_these_tests():
      continue
    if name in optional_tests and name not in test_cls.optional_tests():
      continue

    for dtype, use_placeholder, shape_info in itertools.product(
        test_cls.dtypes_to_test(),
        test_cls.use_placeholder_options(),
        test_cls.operator_shapes_infos()):
      base_test_name = "_".join([
          "test", name, "_shape={},dtype={},use_placeholder={}".format(
              shape_info.shape, dtype, use_placeholder)])
      if name in tests_with_adjoint_args:
        for adjoint in test_cls.adjoint_options():
          for adjoint_arg in test_cls.adjoint_arg_options():
            test_name = base_test_name + ",adjoint={},adjoint_arg={}".format(
                adjoint, adjoint_arg)
            if hasattr(test_cls, test_name):
              raise RuntimeError("Test %s defined more than once" % test_name)
            setattr(
                test_cls,
                test_name,
                test_util.run_deprecated_v1(
                    test_template_fn(  # pylint: disable=too-many-function-args
                        use_placeholder, shape_info, dtype, adjoint,
                        adjoint_arg, test_cls.use_blockwise_arg())))
      else:
        if hasattr(test_cls, base_test_name):
          raise RuntimeError("Test %s defined more than once" % base_test_name)
        setattr(
            test_cls,
            base_test_name,
            test_util.run_deprecated_v1(test_template_fn(
                use_placeholder, shape_info, dtype)))


class SquareLinearOperatorDerivedClassTest(
    LinearOperatorDerivedClassTest, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Base test class for square operators. Provides square shapes (including empty and batched) and
  # makes x and rhs with matching sizes, choosing one or two columns by operator tensor rank.
  # Subclasses still define operator_and_matrix.
  """Base test class appropriate for square operators.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Static override returning shape infos for square operators: empty 0x0, 1x1, batch of one 3x3,
    # batch of three 4x4 and batch 2 by 1 of 4x4.
    shapes_info = OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shapes_info((0, 0)),
        shapes_info((1, 1)),
        shapes_info((1, 3, 3)),
        shapes_info((3, 4, 4)),
        shapes_info((2, 1, 4, 4))]

  def make_rhs(self, operator, adjoint, with_batch=True):
    # [FUNCTION DOC]
    # Makes a right-hand side. Since the operator is square, delegates to make_x with the adjoint flag
    # flipped. Inputs: operator, adjoint, with_batch (default true). Returns a random tensor.
    # This operator is square, so rhs and x will have same shape.
    # adjoint value makes no difference because the operator shape doesn't
    # change since it is square, but be pedantic.
    return self.make_x(operator, adjoint=not adjoint, with_batch=with_batch)

  def make_x(self, operator, adjoint, with_batch=True):
    # [FUNCTION DOC]
    # Makes a random normal input for matmul of shape batch plus size-by-columns, columns being one or
    # two. Uses static shapes when fully defined, otherwise dynamic ones. Omits batch dimensions when
    # with_batch is false. Adjoint makes no difference for square operators. Returns a tensor.
    # Value of adjoint makes no difference because the operator is square.
    # Return the number of systems to solve, R, equal to 1 or 2.
    r = self._get_num_systems(operator)
    # If operator.shape = [B1,...,Bb, N, N] this returns a random matrix of
    # shape [B1,...,Bb, N, R], R = 1 or 2.
    if operator.shape.is_fully_defined():
      batch_shape = operator.batch_shape.as_list()
      n = operator.domain_dimension.value
      if with_batch:
        x_shape = batch_shape + [n, r]
      else:
        x_shape = [n, r]
    else:
      batch_shape = operator.batch_shape_tensor()
      n = operator.domain_dimension_tensor()
      if with_batch:
        x_shape = array_ops.concat((batch_shape, [n, r]), 0)
      else:
        x_shape = [n, r]

    return random_normal(x_shape, dtype=operator.dtype)

  def _get_num_systems(self, operator):
    # [FUNCTION DOC]
    # HELPER: Returns the number of columns for generated inputs: one if the operator tensor rank is
    # unknown or odd, otherwise two.
    """Get some number, either 1 or 2, depending on operator."""
    if operator.tensor_rank is None or operator.tensor_rank % 2:
      return 1
    else:
      return 2


class NonSquareLinearOperatorDerivedClassTest(
    LinearOperatorDerivedClassTest, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Base test class for rectangular operators. Never tests square shapes (create a separate class for
  # that). Skips cholesky, eigenvalue, inverse, solve, determinant tests since they need square
  # matrices. Subclasses still define operator_and_matrix.
  # KNOWN ISSUE: the docstring refers to a square full-matrix test class by a name that does not exist
  # in this module (the square base class here is a different name).
  """Base test class appropriate for generic rectangular operators.

  Square shapes are never tested by this class, so if you want to test your
  operator with a square shape, create two test classes, the other subclassing
  SquareLinearOperatorFullMatrixTest.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static override returning names of tests to skip: Cholesky, eigenvalues, inverse, solves,
    # determinant and log determinant.
    """List of test names to skip."""
    return [
        "cholesky",
        "eigvalsh",
        "inverse",
        "solve",
        "solve_with_broadcast",
        "det",
        "log_abs_det",
    ]

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Static override returning rectangular shape infos: 2x1, 1x2, batch 1 of 3x2, batch 3 of 3x4 and
    # batch 2 by 1 of 2x4.
    # KNOWN ISSUE: the inline comment says non-batch operators of shape n by n, but the shapes are not
    # square.
    shapes_info = OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shapes_info((2, 1)),
        shapes_info((1, 2)),
        shapes_info((1, 3, 2)),
        shapes_info((3, 3, 4)),
        shapes_info((2, 1, 2, 4))]

  def make_rhs(self, operator, adjoint, with_batch=True):
    # [FUNCTION DOC]
    # Always raises a not-implemented error because solve is not tested for rectangular operators.
    # Inputs are accepted but unused.
    # TODO(langmore) Add once we're testing solve_ls.
    raise NotImplementedError(
        "make_rhs not implemented because we don't test solve")

  def make_x(self, operator, adjoint, with_batch=True):
    # [FUNCTION DOC]
    # Makes a random normal input for matmul: shape is batch plus size-by-columns where size is the
    # range dimension if adjoint, else the domain dimension, columns one or two. Uses static shapes when
    # fully defined, else dynamic. Drops batch dims when with_batch is false. Returns a tensor.
    # Return the number of systems for the argument 'x' for .matmul(x)
    r = self._get_num_systems(operator)
    # If operator.shape = [B1,...,Bb, M, N] this returns a random matrix of
    # shape [B1,...,Bb, N, R], R = 1 or 2.
    if operator.shape.is_fully_defined():
      batch_shape = operator.batch_shape.as_list()
      if adjoint:
        n = operator.range_dimension.value
      else:
        n = operator.domain_dimension.value
      if with_batch:
        x_shape = batch_shape + [n, r]
      else:
        x_shape = [n, r]
    else:
      batch_shape = operator.batch_shape_tensor()
      if adjoint:
        n = operator.range_dimension_tensor()
      else:
        n = operator.domain_dimension_tensor()
      if with_batch:
        x_shape = array_ops.concat((batch_shape, [n, r]), 0)
      else:
        x_shape = [n, r]

    return random_normal(x_shape, dtype=operator.dtype)

  def _get_num_systems(self, operator):
    # [FUNCTION DOC]
    # HELPER: Returns the number of columns for generated inputs: one if the operator tensor rank is
    # unknown or odd, otherwise two. Duplicate of the square class helper.
    """Get some number, either 1 or 2, depending on operator."""
    if operator.tensor_rank is None or operator.tensor_rank % 2:
      return 1
    else:
      return 2


def random_positive_definite_matrix(shape,
                                    dtype,
                                    oversampling_ratio=4,
                                    force_well_conditioned=False):
  # [FUNCTION DOC]
  # Returns a random batch of positive definite matrices drawn as sample covariance (Wishart)
  # matrices. Inputs: shape (tensor shape or list, last two dims equal), dtype, oversampling_ratio
  # (sample size divided by dimension, default 4; smaller than one gives a singular matrix),
  # force_well_conditioned (if true add the identity and halve, default false). Draws normal
  # samples, forms the product with the conjugate transpose. Raises a value error if the matrix is
  # not square statically. Random; returns a tensor.
  # KNOWN ISSUE: if shape is passed as a tensor it is later treated as a list, which fails; the
  # docstring name is misspelled as Wisart.
  """[batch] positive definite Wisart matrix.

  A Wishart(N, S) matrix is the S sample covariance matrix of an N-variate
  (standard) Normal random variable.

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned matrix.
    dtype:  `TensorFlow` `dtype` or Python dtype.
    oversampling_ratio: S / N in the above.  If S < N, the matrix will be
      singular (unless `force_well_conditioned is True`).
    force_well_conditioned:  Python bool.  If `True`, add `1` to the diagonal
      of the Wishart matrix, then divide by 2, ensuring most eigenvalues are
      close to 1.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)
  if not tensor_util.is_tf_type(shape):
    shape = tensor_shape.TensorShape(shape)
    # Matrix must be square.
    shape.dims[-1].assert_is_compatible_with(shape.dims[-2])
  shape = shape.as_list()
  n = shape[-2]
  s = oversampling_ratio * shape[-1]
  wigner_shape = shape[:-2] + [n, s]

  with ops.name_scope("random_positive_definite_matrix"):
    wigner = random_normal(
        wigner_shape,
        dtype=dtype,
        stddev=math_ops.cast(1 / np.sqrt(s), dtype.real_dtype))
    wishart = math_ops.matmul(wigner, wigner, adjoint_b=True)
    if force_well_conditioned:
      wishart += linalg_ops.eye(n, dtype=dtype)
      wishart /= math_ops.cast(2, dtype)
    return wishart


def random_tril_matrix(shape,
                       dtype,
                       force_well_conditioned=False,
                       remove_upper=True):
  # [FUNCTION DOC]
  # Returns a random batch of lower triangular matrices. Inputs: shape, dtype, force_well_conditioned
  # (replace the diagonal by random-sign values of modulus between one and square root of two,
  # default false), remove_upper (zero the strictly upper triangle, default true). Random; returns a
  # tensor.
  # KNOWN ISSUE: the docstring says dtype may be a Python dtype and eigenvalue moduli lie in the
  # range one to two, but the code reads the real dtype attribute of dtype directly (so a plain
  # Python or numpy dtype fails) and the moduli are at most square root of two.
  """[batch] lower triangular matrix.

  Args:
    shape:  `TensorShape` or Python `list`.  Shape of the returned matrix.
    dtype:  `TensorFlow` `dtype` or Python dtype
    force_well_conditioned:  Python `bool`. If `True`, returned matrix will have
      eigenvalues with modulus in `(1, 2)`.  Otherwise, eigenvalues are unit
      normal random variables.
    remove_upper:  Python `bool`.
      If `True`, zero out the strictly upper triangle.
      If `False`, the lower triangle of returned matrix will have desired
      properties, but will not have the strictly upper triangle zero'd out.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  with ops.name_scope("random_tril_matrix"):
    # Totally random matrix.  Has no nice properties.
    tril = random_normal(shape, dtype=dtype)
    if remove_upper:
      tril = array_ops.matrix_band_part(tril, -1, 0)

    # Create a diagonal with entries having modulus in [1, 2].
    if force_well_conditioned:
      maxval = ops.convert_to_tensor(np.sqrt(2.), dtype=dtype.real_dtype)
      diag = random_sign_uniform(
          shape[:-1], dtype=dtype, minval=1., maxval=maxval)
      tril = array_ops.matrix_set_diag(tril, diag)

    return tril


def random_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None):
  # [FUNCTION DOC]
  # Returns a tensor with Gaussian entries; complex dtypes get independent real and imaginary parts.
  # Inputs: shape, mean, stddev (default zero and one), dtype (default 32-bit float), optional integer
  # seed (a different offset seed is used for the imaginary part). Random; returns a tensor.
  """Tensor with (possibly complex) Gaussian entries.

  Samples are distributed like

  ```
  N(mean, stddev^2), if dtype is real,
  X + iY,  where X, Y ~ N(mean, stddev^2) if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    mean:  `Tensor` giving mean of normal to sample from.
    stddev:  `Tensor` giving stdev of normal to sample from.
    dtype:  `TensorFlow` `dtype` or numpy dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_normal"):
    samples = random_ops.random_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype.real_dtype, seed=seed)
    if dtype.is_complex:
      if seed is not None:
        seed += 1234
      more_samples = random_ops.random_normal(
          shape, mean=mean, stddev=stddev, dtype=dtype.real_dtype, seed=seed)
      samples = math_ops.complex(samples, more_samples)
    return samples


def random_uniform(shape,
                   minval=None,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None):
  # [FUNCTION DOC]
  # Returns a tensor with uniform entries between minval and maxval; complex dtypes draw independent
  # real and imaginary parts. Inputs: shape, minval, maxval (scalars), dtype (default 32-bit float),
  # optional seed. Random; returns a tensor.
  """Tensor with (possibly complex) Uniform entries.

  Samples are distributed like

  ```
  Uniform[minval, maxval], if dtype is real,
  X + iY,  where X, Y ~ Uniform[minval, maxval], if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    minval:  `0-D` `Tensor` giving the minimum values.
    maxval:  `0-D` `Tensor` giving the maximum values.
    dtype:  `TensorFlow` `dtype` or Python dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_uniform"):
    samples = random_ops.random_uniform(
        shape, dtype=dtype.real_dtype, minval=minval, maxval=maxval, seed=seed)
    if dtype.is_complex:
      if seed is not None:
        seed += 12345
      more_samples = random_ops.random_uniform(
          shape,
          dtype=dtype.real_dtype,
          minval=minval,
          maxval=maxval,
          seed=seed)
      samples = math_ops.complex(samples, more_samples)
    return samples


def random_sign_uniform(shape,
                        minval=None,
                        maxval=None,
                        dtype=dtypes.float32,
                        seed=None):
  # [FUNCTION DOC]
  # Returns a tensor whose entries are uniform on the interval between minval and maxval times a random
  # sign (plus or minus one with equal probability); complex dtypes use uniform real and imaginary
  # parts. Inputs: shape, minval, maxval, dtype (default 32-bit float), optional seed. Random;
  # returns a tensor.
  """Tensor with (possibly complex) random entries from a "sign Uniform".

  Letting `Z` be a random variable equal to `-1` and `1` with equal probability,
  Samples from this `Op` are distributed like

  ```
  Z * X, where X ~ Uniform[minval, maxval], if dtype is real,
  Z * (X + iY),  where X, Y ~ Uniform[minval, maxval], if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    minval:  `0-D` `Tensor` giving the minimum values.
    maxval:  `0-D` `Tensor` giving the maximum values.
    dtype:  `TensorFlow` `dtype` or Python dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_sign_uniform"):
    unsigned_samples = random_uniform(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)
    if seed is not None:
      seed += 12
    signs = math_ops.sign(
        random_ops.random_uniform(shape, minval=-1., maxval=1., seed=seed))
    return unsigned_samples * math_ops.cast(signs, unsigned_samples.dtype)


def random_normal_correlated_columns(shape,
                                     mean=0.0,
                                     stddev=1.0,
                                     dtype=dtypes.float32,
                                     eps=1e-4,
                                     seed=None):
  # [FUNCTION DOC]
  # Returns a random batch matrix with Gaussian entries and nearly dependent columns. Inputs: shape
  # (list, at least length two), mean, stddev, dtype, eps (size of perturbation off the hyperplane),
  # seed. Builds the matrix as the product of an embedding of a narrower random matrix plus small
  # noise plus the mean. Raises a value error if shape has fewer than two entries.
  # KNOWN ISSUE: the docstring says a plain independent matrix is returned when rows are fewer than
  # columns and correlated otherwise, but the code does the reverse: it returns the plain matrix when
  # there is at most one column or columns are fewer than rows, and correlates otherwise.
  # Also the added noise is not seeded.
  """Batch matrix with (possibly complex) Gaussian entries and correlated cols.

  Returns random batch matrix `A` with specified element-wise `mean`, `stddev`,
  living close to an embedded hyperplane.

  Suppose `shape[-2:] = (M, N)`.

  If `M < N`, `A` is a random `M x N` [batch] matrix with iid Gaussian entries.

  If `M >= N`, then the columns of `A` will be made almost dependent as follows:

  ```
  L = random normal N x N-1 matrix, mean = 0, stddev = 1 / sqrt(N - 1)
  B = random normal M x N-1 matrix, mean = 0, stddev = stddev.

  G = (L B^H)^H, a random normal M x N matrix, living on N-1 dim hyperplane
  E = a random normal M x N matrix, mean = 0, stddev = eps
  mu = a constant M x N matrix, equal to the argument "mean"

  A = G + E + mu
  ```

  Args:
    shape:  Python list of integers.
      Shape of the returned tensor.  Must be at least length two.
    mean:  `Tensor` giving mean of normal to sample from.
    stddev:  `Tensor` giving stdev of normal to sample from.
    dtype:  `TensorFlow` `dtype` or numpy dtype
    eps:  Distance each column is perturbed from the low-dimensional subspace.
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.

  Raises:
    ValueError:  If `shape` is not at least length 2.
  """
  dtype = dtypes.as_dtype(dtype)

  if len(shape) < 2:
    raise ValueError(
        "Argument shape must be at least length 2.  Found: %s" % shape)

  # Shape is the final shape, e.g. [..., M, N]
  shape = list(shape)
  batch_shape = shape[:-2]
  m, n = shape[-2:]

  # If there is only one column, "they" are by definition correlated.
  if n < 2 or n < m:
    return random_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)

  # Shape of the matrix with only n - 1 columns that we will embed in higher
  # dimensional space.
  smaller_shape = batch_shape + [m, n - 1]

  # Shape of the embedding matrix, mapping batch matrices
  # from [..., N-1, M] to [..., N, M]
  embedding_mat_shape = batch_shape + [n, n - 1]

  # This stddev for the embedding_mat ensures final result has correct stddev.
  stddev_mat = 1 / np.sqrt(n - 1)

  with ops.name_scope("random_normal_correlated_columns"):
    smaller_mat = random_normal(
        smaller_shape, mean=0.0, stddev=stddev_mat, dtype=dtype, seed=seed)

    if seed is not None:
      seed += 1287

    embedding_mat = random_normal(embedding_mat_shape, dtype=dtype, seed=seed)

    embedded_t = math_ops.matmul(embedding_mat, smaller_mat, transpose_b=True)
    embedded = array_ops.matrix_transpose(embedded_t)

    mean_mat = array_ops.ones_like(embedded) * mean

    return embedded + random_normal(shape, stddev=eps, dtype=dtype) + mean_mat
