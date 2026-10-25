# [MODULE DOC]
# Sparse "inplace" row update helpers for tensors. Exposes alias_inplace_update/add/sub (which
# really alias and overwrite their input through the underlying inplace kernels), the safer
# inplace_update/add/sub (which first deep-copy the input so the original is untouched), empty_like
# (uninitialized tensor with the shape and dtype of another) and the alias empty for the raw empty
# op. A private helper normalizes arguments for all the alias functions. The update, add and sub
# functions are marked deprecated in favor of the tensor scatter update, add and sub ops.
# Import-time side effects: none beyond defining functions and the empty alias.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Inplace operations.
"""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation


def _inplace_helper(x, i, v, op):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the alias_inplace_update/add/sub functions.
  # Converts x to a tensor and v to a tensor of x's dtype, then applies the given inplace kernel
  # op. If i is nothing, both tensors are flattened to one row, the op is applied at row zero and
  # the result is reshaped back to x's shape. Otherwise i is cast to a 32-bit integer; a scalar
  # index is turned into a one-element vector and v gets a new leading axis; a vector index is
  # passed straight through. Returns the resulting tensor (aliasing x for the real kernels).
  # Raises whatever the conversion or kernel raises for incompatible shapes or dtypes.
  """Applies an inplace op on (x, i, v).

  op is one of gen_array_ops.alias_inplace_update,
  gen_array_ops.alias_inplace_add, or gen_array_ops.alias_inplace_sub.

  If i is None, x and v must be the same shape. Computes
    x op v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    x[i, :] op v;
  Otherwise, x and v must have the same rank. Computes
    x[i, :] op v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.
    op: alias_inplace_update, alias_inplace_add, or alias_inplace_sub.

  Returns:
    Returns x.

  """
  x = ops.convert_to_tensor(x)
  v = ops.convert_to_tensor(v, x.dtype)
  if i is None:
    # Full tensor.
    return array_ops.reshape(
        op(array_ops.reshape(x, [1, -1]), [0], array_ops.reshape(v, [1, -1])),
        array_ops.shape(x))
  i = math_ops.cast(i, dtypes.int32)
  if i.get_shape().ndims == 0:
    # Single 0-dim update.
    return op(x, array_ops.reshape(i, [1]), array_ops.expand_dims(v, 0))
  return op(x, i, v)


@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_update, which offers the same functionality '
     'with well-defined read-write semantics.'))
def alias_inplace_update(x, i, v):
  # [FUNCTION DOC]
  # Overwrites rows of x selected by index i with the values v, aliasing and mutating x's buffer.
  # Inputs: x tensor to update; i nothing (whole tensor), a scalar row index or a vector of row
  # indices; v tensor of replacement values (converted to x's dtype). Returns the updated tensor
  # (an alias of x). Delegates to the shared helper with the inplace-update kernel. Emits a
  # deprecation warning when called. Shape mismatches raise errors from the kernel.
  """Applies an inplace update on input x at index i with value v. Aliases x.

  If i is None, x and v must be the same shape. Computes
    x = v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    x[i, :] = v;
  Otherwise, x and v must have the same rank. Computes
    x[i, :] = v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.

  Returns:
    Returns x.

  """
  return _inplace_helper(x, i, v, gen_array_ops.inplace_update)


@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_add, which offers the same functionality '
     'with well-defined read-write semantics.'))
def alias_inplace_add(x, i, v):
  # [FUNCTION DOC]
  # Adds v to the rows of x selected by index i, aliasing and mutating x's buffer.
  # Inputs: x tensor to update; i nothing (whole tensor), a scalar row index or a vector of row
  # indices; v tensor of values to add (converted to x's dtype). Returns the updated tensor (an
  # alias of x). Delegates to the shared helper with the inplace-add kernel. Emits a deprecation
  # warning when called. Shape mismatches raise errors from the kernel.
  """Applies an inplace add on input x at index i with value v. Aliases x.

  If i is None, x and v must be the same shape. Computes
    x += v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    x[i, :] += v;
  Otherwise, x and v must have the same rank. Computes
    x[i, :] += v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.

  Returns:
    Returns x.

  """
  return _inplace_helper(x, i, v, gen_array_ops.inplace_add)


@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_sub, which offers the same functionality '
     'with well-defined read-write semantics.'))
def alias_inplace_sub(x, i, v):
  # [FUNCTION DOC]
  # Subtracts v from the rows of x selected by index i, aliasing and mutating x's buffer.
  # Inputs: x tensor to update; i nothing (whole tensor), a scalar row index or a vector of row
  # indices; v tensor of values to subtract (converted to x's dtype). Returns the updated tensor
  # (an alias of x). Delegates to the shared helper with the inplace-sub kernel. Emits a
  # deprecation warning when called. Shape mismatches raise errors from the kernel.
  """Applies an inplace sub on input x at index i with value v. Aliases x.

  If i is None, x and v must be the same shape. Computes
    x -= v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    x[i, :] -= v;
  Otherwise, x and v must have the same rank. Computes
    x[i, :] -= v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.

  Returns:
    Returns x.

  """
  return _inplace_helper(x, i, v, gen_array_ops.inplace_sub)


def empty_like(x, init=None):
  # [FUNCTION DOC]
  # Creates a new tensor with the same shape and dtype as x without copying its data.
  # Inputs: x anything convertible to a tensor; init optional flag, when true the new tensor is
  # filled with the dtype's default (zero) value, otherwise its contents are arbitrary
  # (uninitialized). Returns the new tensor, never an alias of x. No side effects on x.
  """Returns a non-initialized tensor with the same shape and dtype as x.

  Args:
    x: A Tensor.
    init: Initialize the returned tensor with the default value of
      x.dtype(), if True. Otherwise, do not initialize. Defaults to
      None.

  Returns:
    A tensor y, whose dtype and shape are the same as those of x.
    y is guaranteed not to be an alias of x. Upon return, y may contain
    arbitrary data.

  """
  x = ops.convert_to_tensor(x)
  return gen_array_ops.empty(array_ops.shape(x), x.dtype, init=init)


@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_update, which offers the same functionality '
     'with well-defined read-write semantics.'))
def inplace_update(x, i, v):
  # [FUNCTION DOC]
  # Copy-then-update variant: deep-copies x and overwrites the rows selected by i with v in the
  # copy. Inputs: x tensor; i nothing, scalar row index or vector of row indices; v replacement
  # values. Returns the updated copy, which does not alias x, so x is left unchanged. Emits a
  # deprecation warning when called.
  """Applies an inplace update on input x at index i with value v.

  Note that this function is not actually inplace - it allocates
  a copy of x.  The utility is not avoiding memory copies but rather
  specifying a sparse update.

  If i is None, x and v must be the same shape. Computes
    y = x; y = v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    y = x; y[i, :] = v;
  Otherwise, x and v must have the same rank. Computes
    y = x; y[i, :] = v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.

  Returns:
    Returns y, which is guaranteed not to be an alias of x.

  """
  return alias_inplace_update(gen_array_ops.deep_copy(x), i, v)


@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_add, which offers the same functionality '
     'with well-defined read-write semantics.'))
def inplace_add(x, i, v):
  # [FUNCTION DOC]
  # Copy-then-add variant: deep-copies x and adds v to the rows selected by i in the copy.
  # Inputs: x tensor; i nothing, scalar row index or vector of row indices; v values to add.
  # Returns the updated copy, which does not alias x, so x is left unchanged. Emits a deprecation
  # warning when called.
  """Applies an inplace add on input x at index i with value v.

  Note that this function is not actually inplace - it allocates
  a copy of x.  The utility is not avoiding memory copies but rather
  specifying a sparse update.

  If i is None, x and v must be the same shape. Computes
    y = x; y += v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    y = x; y[i, :] += v;
  Otherwise, x and v must have the same rank. Computes
    y = x; y[i, :] += v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.

  Returns:
    Returns y, which is guaranteed not to be an alias of x.

  """
  return alias_inplace_add(gen_array_ops.deep_copy(x), i, v)


@deprecation.deprecated(
    None,
    ('Prefer tf.tensor_scatter_nd_sub, which offers the same functionality '
     'with well-defined read-write semantics.'))
def inplace_sub(x, i, v):
  # [FUNCTION DOC]
  # Copy-then-subtract variant: deep-copies x and subtracts v from the rows selected by i in the
  # copy. Inputs: x tensor; i nothing, scalar row index or vector of row indices; v values to
  # subtract. Returns the updated copy, which does not alias x, so x is left unchanged. Emits a
  # deprecation warning when called.
  """Applies an inplace sub on input x at index i with value v.

  Note that this function is not actually inplace - it allocates
  a copy of x.  The utility is not avoiding memory copies but rather
  specifying a sparse update.

  If i is None, x and v must be the same shape. Computes
    y = x; y -= v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    y = x; y[i, :] -= v;
  Otherwise, x and v must have the same rank. Computes
    y = x; y[i, :] -= v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.

  Returns:
    Returns y, which is guaranteed not to be an alias of x.

  """
  return alias_inplace_sub(gen_array_ops.deep_copy(x), i, v)

empty = gen_array_ops.empty
