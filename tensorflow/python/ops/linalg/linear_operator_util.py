# [MODULE DOC]
# HELPER: internal utility module shared by the LinearOperator classes in this package.
# Collects small helpers: conversion of non-reference inputs to tensors and detection of
# reference (variable or module) objects, dtype helpers, assertion builders (nonzero modulus,
# zero imaginary part, compatible matrix dimensions, batch-matrix rank), broadcasting versions of
# solve and batch-dimension alignment (with a memory-saving reshape trick), hint-combining
# helpers for operator meta-classes (adjoint pairs, A A-adjoint form, merging provided and
# operator hints), and helpers for blockwise operators (detecting blockwise input and splitting
# arguments into blocks). No import-time side effects.
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
"""Internal utilities for `LinearOperator` classes."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest


################################################################################
# To make more friendly for TF2.
################################################################################


def convert_nonref_to_tensor(value, dtype=None, dtype_hint=None, name=None):
  # [FUNCTION DOC]
  # HELPER: used by operator constructors to store parameters without copying variables.
  # KNOWN ISSUE: the docstring says a value error is raised for a mismatched dtype; for reference
  # objects (variables and modules) a mismatch raises a type error instead, and the docstring
  # example comparisons on identity are only informal.
  # Returns nothing-value if value is null. If value has reference semantics (variable or module
  # with dtype and shape) it is returned unchanged, after checking that an explicitly given dtype
  # matches its base dtype (raising a type error otherwise). Any other value is converted to a
  # tensor using the optional dtype, soft preference dtype_hint and op name. Conversion errors
  # come from the standard conversion machinery. No side effects.
  """Converts the given `value` to a `Tensor` if input is nonreference type.

  This function converts Python objects of various types to `Tensor` objects
  except if the input has nonreference semantics. Reference semantics are
  characterized by `is_ref` and is any object which is a
  `tf.Variable` or instance of `tf.Module`. This function accepts any input
  which `tf.convert_to_tensor` would also.

  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    dtype_hint: Optional element type for the returned tensor,
      used when dtype is None. In some cases, a caller may not have a
      dtype in mind when converting to a tensor, so dtype_hint
      can be used as a soft preference.  If the conversion to
      `dtype_hint` is not possible, this argument has no effect.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    tensor: A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.


  #### Examples:

  ```python

  x = tf.Variable(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  # ==> True

  x = tf.constant(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  # ==> True

  x = np.array(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  # ==> False
  tf.is_tensor(y)
  # ==> True

  x = tfp.util.DeferredTensor(13.37, lambda x: x)
  y = convert_nonref_to_tensor(x)
  x is y
  # ==> True
  tf.is_tensor(y)
  # ==> False
  tf.equal(y, 13.37)
  # ==> True
  ```

  """
  # We explicitly do not use a tf.name_scope to avoid graph clutter.
  if value is None:
    return None
  if is_ref(value):
    if dtype is None:
      return value
    dtype_base = base_dtype(dtype)
    value_dtype_base = base_dtype(value.dtype)
    if dtype_base != value_dtype_base:
      raise TypeError(
          f"Argument `value` must be of dtype `{dtype_name(dtype_base)}` "
          f"Received: `{dtype_name(value_dtype_base)}`.")
    return value
  return tensor_conversion.convert_to_tensor_v2_with_dispatch(
      value, dtype=dtype, dtype_hint=dtype_hint, name=name
  )


def base_dtype(dtype):
  # [FUNCTION DOC]
  # HELPER: dtype normalisation.
  # Converts the argument to a data type object and returns its non-reference base type (or the
  # type itself if it has no base type notion).
  """Returns a non-reference `dtype` based on this `dtype`."""
  dtype = dtypes.as_dtype(dtype)
  if hasattr(dtype, "base_dtype"):
    return dtype.base_dtype
  return dtype


def dtype_name(dtype):
  # [FUNCTION DOC]
  # HELPER: used to format dtype names in error messages.
  # Converts the argument to a data type object and returns its name string, falling back to a
  # class name or the string form when no name attribute exists.
  """Returns the string name for this `dtype`."""
  dtype = dtypes.as_dtype(dtype)
  if hasattr(dtype, "name"):
    return dtype.name
  if hasattr(dtype, "__name__"):
    return dtype.__name__
  return str(dtype)


def check_dtype(arg, dtype):
  # [FUNCTION DOC]
  # HELPER: dtype validation for operator arguments.
  # Raises a type error if the base dtype of arg (a tensor-like with a dtype) differs from the
  # expected dtype; otherwise returns nothing.
  """Check that arg.dtype == self.dtype."""
  if arg.dtype.base_dtype != dtype:
    raise TypeError(
        f"Expected argument to have dtype {dtype}. Found: {arg.dtype} in "
        f"tensor {arg}.")


def is_ref(x):
  # [FUNCTION DOC]
  # HELPER: reference-semantics test used when converting operator parameters.
  # KNOWN ISSUE: the docstring's return description says the result indicates the input has
  # nonreference semantics; the code returns true exactly when x HAS reference semantics.
  # Returns true if x is a variable, or a module that exposes both dtype and shape attributes;
  # false otherwise. No side effects.
  """Evaluates if the object has reference semantics.

  An object is deemed "reference" if it is a `tf.Variable` instance or is
  derived from a `tf.Module` with `dtype` and `shape` properties.

  Args:
    x: Any object.

  Returns:
    is_ref: Python `bool` indicating input is has nonreference semantics, i.e.,
      is a `tf.Variable` or a `tf.Module` with `dtype` and `shape` properties.
  """
  return (
      # Note: we check that tf.Variable is a class because we might be using a
      # different backend other than TF.
      isinstance(x, variables_module.Variable) or
      (isinstance(x, module.Module) and hasattr(x, "dtype") and
       hasattr(x, "shape")))


def assert_not_ref_type(x, arg_name):
  # [FUNCTION DOC]
  # HELPER: argument validation. Raises a type error naming arg_name if x has reference semantics
  # (a variable or suitable module); otherwise returns nothing.
  if is_ref(x):
    raise TypeError(
        f"Argument {arg_name} cannot be reference type. Found: {type(x)}.")


################################################################################
# Asserts.
################################################################################


def assert_no_entries_with_modulus_zero(
    x, message=None, name="assert_no_entries_with_modulus_zero"):
  # [FUNCTION DOC]
  # HELPER: assertion builder used for non-singularity checks.
  # Converts x (numeric tensor, real, integer or complex) to a tensor and returns an assertion op
  # that fails at run time (with the optional message prepended) unless every entry has strictly
  # positive absolute value. The name argument scopes the created ops.
  """Returns `Op` that asserts Tensor `x` has no entries with modulus zero.

  Args:
    x:  Numeric `Tensor`, real, integer, or complex.
    message:  A string message to prepend to failure message.
    name:  A name to give this `Op`.

  Returns:
    An `Op` that asserts `x` has no entries with modulus zero.
  """
  with ops.name_scope(name, values=[x]):
    x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
    dtype = x.dtype.base_dtype
    should_be_nonzero = math_ops.abs(x)
    zero = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        0, dtype=dtype.real_dtype
    )
    return check_ops.assert_less(zero, should_be_nonzero, message=message)


def assert_zero_imag_part(x, message=None, name="assert_zero_imag_part"):
  # [FUNCTION DOC]
  # HELPER: assertion builder used for self-adjointness checks.
  # KNOWN ISSUE: the docstring's Returns text says the op asserts no entries with modulus zero,
  # which is wrong; the op asserts the imaginary part is zero.
  # Converts x to a tensor. If its dtype is floating, returns a no-op (nothing to check).
  # Otherwise returns an assertion op that every imaginary part equals zero, failing at run time
  # with the optional message. The name argument scopes the created ops. Note that integer dtypes
  # reach the imaginary-part path as well.
  """Returns `Op` that asserts Tensor `x` has no non-zero imaginary parts.

  Args:
    x:  Numeric `Tensor`, real, integer, or complex.
    message:  A string message to prepend to failure message.
    name:  A name to give this `Op`.

  Returns:
    An `Op` that asserts `x` has no entries with modulus zero.
  """
  with ops.name_scope(name, values=[x]):
    x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
    dtype = x.dtype.base_dtype

    if dtype.is_floating:
      return control_flow_ops.no_op()

    zero = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        0, dtype=dtype.real_dtype
    )
    return check_ops.assert_equal(zero, math_ops.imag(x), message=message)


def assert_compatible_matrix_dimensions(operator, x):
  # [FUNCTION DOC]
  # HELPER: run-time shape check used by operator matmul and solve.
  # Returns an assertion op that fails at run time if the second-to-last dimension of tensor x
  # differs from the operator's domain dimension (as a tensor). Static checks are left to the base
  # class. Does not modify its inputs.
  """Assert that an argument to solve/matmul has proper domain dimension.

  If `operator.shape[-2:] = [M, N]`, and `x.shape[-2:] = [Q, R]`, then
  `operator.matmul(x)` is defined only if `N = Q`.  This `Op` returns an
  `Assert` that "fires" if this is not the case.  Static checks are already
  done by the base class `LinearOperator`.

  Args:
    operator:  `LinearOperator`.
    x:  `Tensor`.

  Returns:
    `Assert` `Op`.
  """
  # Static checks are done in the base class.  Only tensor asserts here.
  assert_same_dd = check_ops.assert_equal(
      array_ops.shape(x)[-2],
      operator.domain_dimension_tensor(),
      # This error message made to look similar to error raised by static check
      # in the base class.
      message=("Dimensions are not compatible.  "
               "shape[-2] of argument to be the same as this operator"))

  return assert_same_dd


def assert_is_batch_matrix(tensor):
  # [FUNCTION DOC]
  # HELPER: static rank validation. Raises a value error if the tensor's static rank is known and
  # below two; returns nothing (unknown rank passes silently).
  """Static assert that `tensor` has rank `2` or higher."""
  sh = tensor.shape
  if sh.ndims is not None and sh.ndims < 2:
    raise ValueError(
        f"Expected [batch] matrix to have at least two dimensions. Found: "
        f"{tensor}.")


def shape_tensor(shape, name=None):
  # [FUNCTION DOC]
  # HELPER: shape argument conversion.
  # Converts a shape (list, tuple or tensor-like) to a tensor, forcing 32-bit integer type when the
  # shape is an empty list or tuple (so an empty shape does not become a float). The optional name
  # names the op. Returns the tensor.
  """Convert Tensor using default type, unless empty list or tuple."""
  # Works just like random_ops._ShapeTensor.
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int32
  else:
    dtype = None
  return tensor_conversion.convert_to_tensor_v2_with_dispatch(
      shape, dtype=dtype, name=name
  )


################################################################################
# Broadcasting versions of common linear algebra functions.
# TODO(b/77519145) Do this more efficiently in some special cases.
################################################################################


def broadcast_matrix_batch_dims(batch_matrices, name=None):
  # [FUNCTION DOC]
  # HELPER: aligns batch dimensions before batched matrix operations.
  # Takes an iterable of tensors (each of rank two or more), converts them, and returns a list
  # where every tensor's leading batch dimensions are broadcast to the common batch shape while
  # the last two (matrix) dimensions are left alone. Uses static shapes when fully known,
  # otherwise dynamic broadcasting, which copies data. A list with fewer than two entries is
  # returned after conversion. Raises a value error if a tensor has static rank below two, and an
  # error if the argument is not a proper iterable. name optionally scopes the ops.
  """Broadcast leading dimensions of zero or more [batch] matrices.

  Example broadcasting one batch dim of two simple matrices.

  ```python
  x = [[1, 2],
       [3, 4]]  # Shape [2, 2], no batch dims

  y = [[[1]]]   # Shape [1, 1, 1], 1 batch dim of shape [1]

  x_bc, y_bc = broadcast_matrix_batch_dims([x, y])

  x_bc
  ==> [[[1, 2],
        [3, 4]]]  # Shape [1, 2, 2], 1 batch dim of shape [1].

  y_bc
  ==> same as y
  ```

  Example broadcasting many batch dims

  ```python
  x = tf.random.normal(shape=(2, 3, 1, 4, 4))
  y = tf.random.normal(shape=(1, 3, 2, 5, 5))
  x_bc, y_bc = broadcast_matrix_batch_dims([x, y])

  x_bc.shape
  ==> (2, 3, 2, 4, 4)

  y_bc.shape
  ==> (2, 3, 2, 5, 5)
  ```

  Args:
    batch_matrices:  Iterable of `Tensor`s, each having two or more dimensions.
    name:  A string name to prepend to created ops.

  Returns:
    bcast_matrices: List of `Tensor`s, with `bcast_matrices[i]` containing
      the values from `batch_matrices[i]`, with possibly broadcast batch dims.

  Raises:
    ValueError:  If any input `Tensor` is statically determined to have less
      than two dimensions.
  """
  with ops.name_scope(
      name or "broadcast_matrix_batch_dims", values=batch_matrices):
    check_ops.assert_proper_iterable(batch_matrices)
    batch_matrices = list(batch_matrices)

    for i, mat in enumerate(batch_matrices):
      batch_matrices[i] = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          mat
      )
      assert_is_batch_matrix(batch_matrices[i])

    if len(batch_matrices) < 2:
      return batch_matrices

    # Try static broadcasting.
    # bcast_batch_shape is the broadcast batch shape of ALL matrices.
    # E.g. if batch_matrices = [x, y], with
    # x.shape =    [2, j, k]  (batch shape =    [2])
    # y.shape = [3, 1, l, m]  (batch shape = [3, 1])
    # ==> bcast_batch_shape = [3, 2]
    bcast_batch_shape = batch_matrices[0].shape[:-2]
    for mat in batch_matrices[1:]:
      bcast_batch_shape = array_ops.broadcast_static_shape(
          bcast_batch_shape,
          mat.shape[:-2])
    if bcast_batch_shape.is_fully_defined():
      for i, mat in enumerate(batch_matrices):
        if mat.shape[:-2] != bcast_batch_shape:
          bcast_shape = array_ops.concat(
              [bcast_batch_shape.as_list(), array_ops.shape(mat)[-2:]], axis=0)
          batch_matrices[i] = array_ops.broadcast_to(mat, bcast_shape)
      return batch_matrices

    # Since static didn't work, do dynamic, which always copies data.
    bcast_batch_shape = array_ops.shape(batch_matrices[0])[:-2]
    for mat in batch_matrices[1:]:
      bcast_batch_shape = array_ops.broadcast_dynamic_shape(
          bcast_batch_shape,
          array_ops.shape(mat)[:-2])
    for i, mat in enumerate(batch_matrices):
      batch_matrices[i] = array_ops.broadcast_to(
          mat,
          array_ops.concat(
              [bcast_batch_shape, array_ops.shape(mat)[-2:]], axis=0))

    return batch_matrices


def matrix_solve_with_broadcast(matrix, rhs, adjoint=False, name=None):
  # [FUNCTION DOC]
  # HELPER: batch-broadcasting wrapper around the matrix solve op, used by dense-based operators.
  # Converts matrix and rhs (rhs cast to the matrix dtype), optionally reshapes extra leading dims
  # of rhs to save memory, broadcasts batch dimensions, solves (using the adjoint of matrix when
  # adjoint is true and it was not already applied by the reshape), and maps the solution back to
  # the expected shape. Returns the solution tensor. name optionally scopes the ops.
  """Solve systems of linear equations."""
  with ops.name_scope(name, "MatrixSolveWithBroadcast", [matrix, rhs]):
    matrix = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        matrix, name="matrix"
    )
    rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        rhs, name="rhs", dtype=matrix.dtype
    )

    # If either matrix/rhs has extra dims, we can reshape to get rid of them.
    matrix, rhs, reshape_inv, still_need_to_transpose = _reshape_for_efficiency(
        matrix, rhs, adjoint_a=adjoint)

    # This will broadcast by brute force if we still need to.
    matrix, rhs = broadcast_matrix_batch_dims([matrix, rhs])

    solution = linalg_ops.matrix_solve(
        matrix, rhs, adjoint=adjoint and still_need_to_transpose)

    return reshape_inv(solution)


def _reshape_for_efficiency(a,
                            b,
                            transpose_a=False,
                            transpose_b=False,
                            adjoint_a=False,
                            adjoint_b=False):
  # [FUNCTION DOC]
  # HELPER: memory optimisation for matmul/solve with broadcasting.
  # KNOWN ISSUE: in the final transpose step for b, the code tests the transpose_a flag instead of
  # transpose_b, so a requested transpose of b is applied only when transpose_a is set (and is
  # wrongly applied when only transpose_a is set).
  # When b has more leading dimensions than a and those extra dimensions are worth flipping,
  # returns a with b reshaped so the extra dims are folded into the trailing columns, plus a
  # function that undoes this reshaping on a result, and a flag saying whether the caller still
  # needs to transpose or adjoint a (and b). In the no-reshape cases (unknown rank, a rank at
  # least b rank, small extra dims) returns the inputs, an identity function and a true flag.
  # If it flips, requested transposes and adjoints are applied here and the flag is false.
  # Returns a tuple of (a, b, inverse function, flag). Pure with respect to its inputs.
  """Maybe reshape a, b, and return an inverse map.  For matmul/solve."""
  def identity(x):
    # [FUNCTION DOC]
    # HELPER: no-op inverse mapping used when no reshaping was done; returns its argument as is.
    return x

  # At this point, we have not taken transpose/adjoint of a/b.
  still_need_to_transpose = True

  if a.shape.ndims is None or b.shape.ndims is None:
    return a, b, identity, still_need_to_transpose

  # This could be handled in the future, but seems less common.
  if a.shape.ndims >= b.shape.ndims:
    return a, b, identity, still_need_to_transpose

  # From now on, we might modify b, but will not modify a.

  # Suppose:
  #   a.shape =     C + [m, n], b.shape =
  #   b.shape = S + C + [n, r]
  b_extra_ndims = b.shape.ndims - a.shape.ndims

  # b_extra_sh = S, b_main_sh = C + [n, r]
  b_extra_sh = array_ops.shape(b)[:b_extra_ndims]
  b_main_sh = array_ops.shape(b)[b_extra_ndims:]

  # No reason to flip unless the extra dims of b are big enough.  Why?
  # Assume adjoint/transpose = False.  Then...
  # By not flipping, we have to replicate a to shape
  #   b_extra_sh + a.shape,
  # which could use extra memory.  But in all cases, the final output has shape
  #   b_extra_sh + a.shape[:-1] + [b.shape[-1]]
  # So we only end up creating a larger object if the end dim of b is smaller
  # than the end dim of a.  This often happens, e.g. if b was a vector that was
  # expanded to a matrix (by appending a singleton).

  # Since adjoint/transpose may not be False, we must make adjustments here.
  # The dim of b that holds the multiple equations.
  a_domain_sz_ = a.shape[-2 if adjoint_a or transpose_a else -1]
  b_eq_sz_ = b.shape[-2 if adjoint_b or transpose_b else -1]
  b_extra_sz_ = (
      np.prod(b.shape[:b_extra_ndims].as_list())
      if b.shape[:b_extra_ndims].is_fully_defined() else None)
  if (a_domain_sz_ is not None and b_eq_sz_ is not None and
      b_extra_sz_ is not None):
    if b_extra_sz_ < 2 or a_domain_sz_ <= b_eq_sz_:
      return a, b, identity, still_need_to_transpose

  # At this point, we're flipping for sure!
  # Any transposes/adjoints will happen here explicitly, rather than in calling
  # code.  Why?  To avoid having to write separate complex code for each case.
  if adjoint_a:
    a = array_ops.matrix_transpose(a, conjugate=True)
  elif transpose_a:
    a = array_ops.matrix_transpose(a, conjugate=False)
  if adjoint_b:
    b = array_ops.matrix_transpose(b, conjugate=True)
  elif transpose_a:
    b = array_ops.matrix_transpose(b, conjugate=False)
  still_need_to_transpose = False

  # Recompute shapes, since the transpose/adjoint may have changed them.
  b_extra_sh = array_ops.shape(b)[:b_extra_ndims]
  b_main_sh = array_ops.shape(b)[b_extra_ndims:]

  # Permutation to put the extra dims at the end.
  perm = (
      np.concatenate(
          (np.arange(b_extra_ndims, b.shape.ndims),
           np.arange(0, b_extra_ndims)), 0))
  b_extra_on_end = array_ops.transpose(b, perm=perm)

  # Now squash this end into one long dim.
  b_squashed_end = array_ops.reshape(
      b_extra_on_end, array_ops.concat((b_main_sh[:-1], [-1]), 0))

  def reshape_inv(y):
    # [FUNCTION DOC]
    # HELPER: inverse of the reshaping done in the enclosing function.
    # Takes a result tensor y whose last dimension holds the folded extra dims, unfolds them to
    # the original extra dimensions at the end, then transposes them back to the front using the
    # inverse of the earlier permutation. Returns the reshaped tensor.
    # Expand the extra dims hanging off the end, "b_extra_sh".
    # Note we use y_sh[:-1] + [b_main_sh[-1]] rather than b_main_sh, because y
    # Could have different batch dims than a and b, because of broadcasting.
    y_extra_shape = array_ops.concat(
        (array_ops.shape(y)[:-1], [b_main_sh[-1]], b_extra_sh), 0)
    y_extra_on_end = array_ops.reshape(y, y_extra_shape)
    inverse_perm = np.argsort(perm)
    return array_ops.transpose(y_extra_on_end, perm=inverse_perm)

  return a, b_squashed_end, reshape_inv, still_need_to_transpose


################################################################################
# Helpers for hints.
################################################################################


def is_adjoint_pair(x, y):
  # [FUNCTION DOC]
  # HELPER: hint inference for composed operators.
  # Returns true if the two operators are known to be adjoints by identity: the same object that
  # is self-adjoint, or one is the adjoint wrapper of the other. Returns false when the same
  # object is explicitly marked not self-adjoint. Entries are never compared.
  """True iff x and y are adjoints of each other (by id, not entries)."""
  if x is y:  # Note that if x is y then all of their hints are the same!
    if x.is_self_adjoint is False:  # pylint:disable=g-bool-id-comparison
      return False
    if x.is_self_adjoint:
      return True
  # Use the fact that if x = LinearOperatorAdjoint(y), then x.H is y.
  return x.H is y or y.H is x


def is_aat_form(operators):
  # [FUNCTION DOC]
  # HELPER: hint inference for composition operators (detects A times adjoint of A structure).
  # Takes an iterable of operators. Raises a value error if it is empty. Returns false for an odd
  # count; otherwise returns true if every operator i and its mirror from the end form an adjoint
  # pair.
  """Returns True if operators is of the form A @ A.H, possibly recursively."""
  operators = list(operators)
  if not operators:
    raise ValueError("AAT form is undefined for empty operators")

  if len(operators) % 2:
    return False

  # Check for forms like (A1 @ A2) @ (A2.H @ A1.H)
  return all(
      is_adjoint_pair(operators[i], operators[-1 - i])
      for i in range(len(operators) // 2))


def use_operator_or_provided_hint_unless_contradicting(
    operator, hint_attr_name, provided_hint_value, message):
  # [FUNCTION DOC]
  # HELPER: hint merging for wrapper operators such as adjoint and inverse.
  # Reads the named hint attribute of the operator (true, false or unknown) and combines it with
  # the user-provided hint (true, false or null). Raises a value error with the given message when
  # one is true and the other is false. Otherwise returns true if either is true, false if either
  # is false, and null when both are unknown.
  """Get combined hint in the case where operator.hint should equal hint.

  Args:
    operator:  LinearOperator that a meta-operator was initialized with.
    hint_attr_name:  String name for the attribute.
    provided_hint_value:  Bool or None. Value passed by user in initialization.
    message:  Error message to print if hints contradict.

  Returns:
    True, False, or None.

  Raises:
    ValueError: If hints contradict.
  """
  op_hint = getattr(operator, hint_attr_name)
  # pylint: disable=g-bool-id-comparison
  if op_hint is False and provided_hint_value:
    raise ValueError(message)
  if op_hint and provided_hint_value is False:
    raise ValueError(message)
  if op_hint or provided_hint_value:
    return True
  if op_hint is False or provided_hint_value is False:
    return False
  # pylint: enable=g-bool-id-comparison
  return None


################################################################################
# Utilities for blockwise operators.
################################################################################


def arg_is_blockwise(block_dimensions, arg, arg_split_dim):
  # [FUNCTION DOC]
  # HELPER: input-structure detection for blockwise operators.
  # KNOWN ISSUE: a comment near the end ("input is not blockwise") is dangling and the nesting
  # of the conditions is hard to follow, but the logic below returns as described here.
  # Decides whether arg should be treated as a list of per-block tensors. Returns false for
  # anything that is not a list or tuple of the same length as block_dimensions. A flat list of
  # non-nested items is blockwise (true). For nested items, compares the sizes along arg_split_dim
  # against the known block dimensions: all-unknown blocks are blockwise if sizes differ (false
  # for a single element, and a value error if all equal because the structure is ambiguous);
  # matching known dimensions mean blockwise; equal sizes at least the sum of known dimensions
  # mean not blockwise; otherwise raises a value error about mismatched dimensions.
  """Detect if input should be interpreted as a list of blocks."""
  # Tuples and lists of length equal to the number of operators may be
  # blockwise.
  if (isinstance(arg, (tuple, list)) and len(arg) == len(block_dimensions)):
    # If the elements of the iterable are not nested, interpret the input as
    # blockwise.
    if not any(nest.is_nested(x) for x in arg):
      return True
    else:
      arg_dims = [
          tensor_conversion.convert_to_tensor_v2_with_dispatch(x).shape[
              arg_split_dim
          ]
          for x in arg
      ]
      self_dims = [dim.value for dim in block_dimensions]

      # If none of the operator dimensions are known, interpret the input as
      # blockwise if its matching dimensions are unequal.
      if all(self_d is None for self_d in self_dims):

        # A nested tuple/list with a single outermost element is not blockwise
        if len(arg_dims) == 1:
          return False
        elif any(dim != arg_dims[0] for dim in arg_dims):
          return True
        else:
          raise ValueError(
              "Parsing of the input structure is ambiguous. Please input "
              "a blockwise iterable of `Tensor`s or a single `Tensor`.")

      # If input dimensions equal the respective (known) blockwise operator
      # dimensions, then the input is blockwise.
      if all(self_d == arg_d or self_d is None
             for self_d, arg_d in zip(self_dims, arg_dims)):
        return True

      # If input dimensions equals are all equal, and are greater than or equal
      # to the sum of the known operator dimensions, interpret the input as
      # blockwise.
      # input is not blockwise.
      self_dim = sum(self_d for self_d in self_dims if self_d is not None)
      if all(s == arg_dims[0] for s in arg_dims) and arg_dims[0] >= self_dim:
        return False

      # If none of these conditions is met, the input shape is mismatched.
      raise ValueError("Input dimension does not match operator dimension.")
  else:
    return False


def split_arg_into_blocks(block_dims, block_dims_fn, arg, axis=-1):
  # [FUNCTION DOC]
  # HELPER: argument splitting for blockwise operators.
  # Splits tensor arg along axis (default last) into pieces sized by the block dimensions. block_dims
  # is an iterable of static dimensions; if any value is unknown, block_dims_fn (a callback that
  # returns the dynamic sizes) is called instead. Returns the list of split tensors.
  """Split `x` into blocks matching `operators`'s `domain_dimension`.

  Specifically, if we have a blockwise lower-triangular matrix, with block
  sizes along the diagonal `[M_j, M_j] j = 0,1,2..J`,  this method splits `arg`
  on `axis` into `J` tensors, whose shape at `axis` is `M_j`.

  Args:
    block_dims: Iterable of `TensorShapes`.
    block_dims_fn: Callable returning an iterable of `Tensor`s.
    arg: `Tensor`. `arg` is split into `J` tensors.
    axis: Python `Integer` representing the axis to split `arg` on.

  Returns:
    A list of `Tensor`s.
  """
  block_sizes = [dim.value for dim in block_dims]
  if any(d is None for d in block_sizes):
    block_sizes = block_dims_fn()
  return array_ops.split(arg, block_sizes, axis=axis)
