# [MODULE DOC]
# HELPER: internal helper module used by the math operator definitions to install Tensor operators.
# Provides the machinery that installs binary arithmetic operators (such as add or multiply) on the
# tensor class and on the sparse tensor class. It exposes a type-promotion helper
# (maybe_promote_tensors), a private helper that guesses a numpy-compatible type for a value, and
# override_binary_operator_helper, which wraps an operator function into forward, reflected and
# sparse wrappers and registers them on a class. Promotion follows numpy rules when numpy style
# promotion is enabled in the ops module, is skipped when automatic dtype conversion is enabled,
# and otherwise may force later operands to the dtype of the first one. There are no import-time
# side effects beyond imports; the registration happens only when the helper function is called.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Binary operator override class for Tensor overrides."""
import numbers
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.util import nest
from tensorflow.python.util import traceback_utils


def _maybe_get_dtype(x):
  # [FUNCTION DOC]
  # HELPER: used by maybe_promote_tensors to compute the numpy result type of its operands.
  # Maps one operand to something numpy's result-type computation understands. Real Python
  # numbers are returned unchanged, tensors yield their numpy dtype, framework dtype objects yield
  # their numpy equivalent, and tensor shapes are treated as 32-bit integers. Any other value
  # (including numpy arrays, deliberately, because numpy would otherwise inspect their values) is
  # returned as is. Input: a single operand of any kind. Output: a number, numpy type or the
  # operand itself. Raises a value error when given a list or tuple since no dtype can be chosen.
  """Returns a numpy type if available from x. Skips if x is numpy.ndarray."""
  # Don't put np.ndarray in this list, because np.result_type looks at the
  # value (not just dtype) of np.ndarray to decide the result type.
  if isinstance(x, numbers.Real):
    return x
  if isinstance(x, tensor_lib.Tensor):
    return x.dtype.as_numpy_dtype
  if isinstance(x, dtypes.DType):
    return x.as_numpy_dtype
  if isinstance(x, tensor_shape.TensorShape):
    return np.int32
  if isinstance(x, (list, tuple)):
    raise ValueError(f"Cannot determine dtype.  Got sequence {x}.")
  return x


def maybe_promote_tensors(*tensors, force_same_dtype=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a list of tensors, but the early-exit paths
  # (auto dtype conversion enabled, no inputs, or promotion disabled without forcing) return the
  # original tuple of arguments unchanged; only the other paths return a new list.
  # Brings the operands of a binary operator to a common dtype. Inputs: any number of tensors or
  # tensor-like values, and the optional flag force_same_dtype (default false). If automatic
  # dtype conversion is enabled, or no operands were given, the operands are returned untouched.
  # If numpy style promotion is disabled, the operands are returned untouched unless the flag is
  # set, in which case every operand after the first is converted to the first operand's base
  # dtype (which can silently truncate values). If numpy style promotion is enabled, the numpy
  # result type of all flattened operands is computed and each operand is cast (tensors) or
  # converted (other values) to that type. Returns the resulting operands. Side effects: may
  # create cast or conversion operations in the current graph or eagerly.
  """Promotes tensors if numpy style promotion is enabled.

  This function promotes `tensors` according to numpy promotion rules
  if numpy style promotion is enabled.  Otherwise, if
  `force_same_dtype` is `True`, it force-casts `tensors[1:]` to
  `tensor[0]`'s dtype. Note that this force-cast can be problematic.
  For example, when some `tensors[1:]` elements can be silently
  downcasted.

  Args:
    *tensors: the list of tensors to promote.
    force_same_dtype: bool (optional, default to `False`). When numpy
      style promotion is disabled and `force_same_dtype` is `True`,
      this function will force-casts `tensors[1:]` to `tensor[0]`'s
      dtype (which could be problematic).

  Returns:
    The promoted list of tensors.
  """
  if ops.is_auto_dtype_conversion_enabled():
    return tensors
  if not tensors:
    return tensors
  if not ops.is_numpy_style_type_promotion():
    if not force_same_dtype:
      return tensors
    promoted_tensors = []
    promoted_tensors.append(tensors[0])
    dtype = tensors[0].dtype.base_dtype
    for tensor in tensors[1:]:
      promoted_tensors.append(
          ops.convert_to_tensor(tensor, dtype, name="x"))
    return promoted_tensors
  result_type = np_dtypes._result_type(  # pylint: disable=protected-access
      *[_maybe_get_dtype(x) for x in nest.flatten(tensors)])
  def _promote_or_cast(x):
    # [FUNCTION DOC]
    # HELPER: inner helper of maybe_promote_tensors.
    # Converts one operand to the already computed numpy result type: existing tensors are cast
    # with a cast operation, anything else is converted to a new tensor of that type. Returns
    # the converted tensor.
    if isinstance(x, tensor_lib.Tensor):
      x = gen_math_ops.cast(x, result_type)
    else:
      x = ops.convert_to_tensor(x, result_type)
    return x
  return [_promote_or_cast(x) for x in tensors]


# pylint: disable=protected-access
def override_binary_operator_helper(
    func, op_name, clazz_object=tensor_lib.Tensor):
  # [FUNCTION DOC]
  # HELPER: glue used by the math operator modules to install operator overloads on tensor classes.
  # Builds three wrappers around the operator function func: a forward wrapper (left-hand
  # operand is the tensor), a reflected wrapper (right-hand operand is the tensor) and a sparse
  # wrapper. All wrappers open a name scope named after op_name and have their traceback
  # filtered. The forward wrapper promotes operands without forcing a common dtype, calls func,
  # and on a type or value error tries the right-hand operand's reflected method, re-raising the
  # original error if that is missing, returns the not-implemented marker, or fails. The reflected
  # wrapper promotes with forced same dtype. The sparse wrapper converts the dense operand to the
  # sparse tensor's dtype and builds a new sparse object from the same indices, new values and the
  # same dense shape. Inputs: func (the operator callable), op_name (name used for the dunder
  # methods and name scope), clazz_object (the Tensor class by default, or the sparse tensor
  # class). Documentation of func is copied to the wrappers. For the Tensor class both the
  # forward and reflected operators are registered; otherwise only the forward sparse one is.
  # Returns nothing. Side effect: modifies clazz_object by overriding its operator methods.
  """Register operators with different tensor and scalar versions.

  If `clazz_object` is `SparseTensor`, assumes `func` takes `(sp_indices,
  sp_values, sp_shape, dense)` and outputs `(new_sp_values)`.

  Args:
    func: the operator
    op_name: name of the operator being overridden
    clazz_object: class to override for.  Either `Tensor` or `SparseTensor`.
  """

  @traceback_utils.filter_traceback
  def binary_op_wrapper(x, y):
    # [FUNCTION DOC]
    # HELPER: forward operator wrapper created by override_binary_operator_helper.
    # Applies the wrapped operator to left operand x and right operand y after dtype promotion,
    # inside a name scope. If that raises a type or value error and y's type defines the
    # reflected method, that method is tried with x; its result is returned unless it yields the
    # not-implemented marker or raises, in which case the original error is re-raised. Without a
    # reflected method the original error propagates. Returns the operator result.
    with ops.name_scope(None, op_name, [x, y]) as name:
      try:
        # force_same_dtype=False to preserve existing TF behavior
        # TODO(b/178860388): Figure out why binary_op_wrapper and
        #   r_binary_op_wrapper use different force_same_dtype values.
        x, y = maybe_promote_tensors(x, y)
        return func(x, y, name=name)
      except (TypeError, ValueError) as e:
        # Even if dispatching the op failed, the RHS may be a tensor aware
        # object that can implement the operator with knowledge of itself
        # and the tensor.
        # If the RHS is not tensor aware we still want to raise the
        # original error from the LHS, because it may be more
        # informative.
        if hasattr(type(y), "__r%s__" % op_name):
          try:
            r_op = getattr(y, "__r%s__" % op_name)
            out = r_op(x)
            if out is NotImplemented:
              raise
            return out
          except (TypeError, ValueError):
            raise e
        else:
          raise

  @traceback_utils.filter_traceback
  def binary_op_wrapper_sparse(sp_x, y):
    # [FUNCTION DOC]
    # HELPER: sparse operator wrapper created by override_binary_operator_helper.
    # Converts the dense operand y to a tensor of the sparse operand's base dtype, applies the
    # wrapped operator to the sparse indices, values, dense shape and y, and returns a new sparse
    # object (of the class being overridden) with the same indices and dense shape but the new
    # values. Raises whatever conversion or the operator raises.
    with ops.name_scope(None, op_name, [sp_x, y]) as name:
      y = ops.convert_to_tensor(y, dtype=sp_x.dtype.base_dtype, name="y")
      # use the passed-in SparseTensor class to avoid having to import
      # SparseTensor, which would cause a cyclic dep with math_ops
      return clazz_object(
          sp_x.indices,
          func(sp_x.indices, sp_x.values, sp_x.dense_shape, y, name=name),
          sp_x.dense_shape)

  @traceback_utils.filter_traceback
  def r_binary_op_wrapper(y, x):
    # [FUNCTION DOC]
    # HELPER: reflected operator wrapper created by override_binary_operator_helper.
    # Called when the tensor is the right operand (x) and y is the left operand. Promotes both
    # with forced same dtype (the left operand's dtype wins when numpy promotion is off), then
    # applies the wrapped operator with x first and y second, inside a name scope. Returns the
    # operator result; has no fallback to other reflected methods.
    with ops.name_scope(None, op_name, [x, y]) as name:
      # TODO(b/178860388): Figure out why binary_op_wrapper and
      #   r_binary_op_wrapper use different force_same_dtype values.
      y, x = maybe_promote_tensors(y, x, force_same_dtype=True)
      return func(x, y, name=name)

  # Propagate func.__doc__ to the wrappers
  try:
    doc = func.__doc__
  except AttributeError:
    doc = None
  binary_op_wrapper.__doc__ = doc
  r_binary_op_wrapper.__doc__ = doc
  binary_op_wrapper_sparse.__doc__ = doc

  if clazz_object is tensor_lib.Tensor:
    clazz_object._override_operator("__%s__" % op_name, binary_op_wrapper)
    del binary_op_wrapper
    clazz_object._override_operator("__r%s__" % op_name, r_binary_op_wrapper)
    del r_binary_op_wrapper
  else:
    clazz_object._override_operator("__%s__" % op_name,
                                    binary_op_wrapper_sparse)
    del binary_op_wrapper_sparse
