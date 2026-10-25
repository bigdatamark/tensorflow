# [MODULE DOC]
# Core array functions of the TensorFlow NumPy API (the experimental numpy namespace): array
# creation (empty, zeros, ones, eye, full, array, asarray, arange, diag, tri...), reductions
# (sum, prod, mean, max, min, var, std, all, any), shape manipulation (reshape, ravel, squeeze,
# transpose, swapaxes, moveaxis, stack/split families, flip, roll, rot90), selection (take,
# where, select, compress, nonzero, take_along_axis, pad) and the machinery for NumPy-style
# indexing and index updates on the ndarray type. Each public function is a thin adaptation of
# the NumPy function onto TensorFlow ops, is exported under the experimental numpy namespace
# and gets its documentation derived from NumPy by the np_doc decorators of np_utils.
# Import-time side effects: defines the newaxis constant, installs the rounding hook, the
# numpy-style getitem and the four index-update methods (update, add, min, max) onto the
# ndarray class from np_arrays, and shadows the builtins all, any, sum, max, min and round
# inside this module (builtins are reached through the builtins module).
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Common array methods."""
# pylint: disable=g-direct-tensorflow-import

import builtins
import enum
import functools
import math
import numbers

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export


newaxis = np.newaxis
tf_export.tf_export('experimental.numpy.newaxis', v1=[]).export_constant(
    __name__, 'newaxis'
)


@tf_export.tf_export('experimental.numpy.empty', v1=[])
@np_utils.np_doc('empty')
def empty(shape, dtype=float):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # Creates an array of the given shape and dtype. Unlike NumPy, the contents are not
  # uninitialized: this simply delegates to zeros, so every element is zero.
  # Inputs: shape (list/tuple of dimensions or a scalar), dtype (python, NumPy or TF type;
  # defaults to the python float type). Returns the new array. Raises whatever zeros raises.
  return zeros(shape, dtype)


@tf_export.tf_export('experimental.numpy.empty_like', v1=[])
@np_utils.np_doc('empty_like')
def empty_like(a, dtype=None):
  # [FUNCTION DOC]
  # Returns an array with the same shape as the array-like input a. Unlike NumPy the values are
  # not uninitialized; this delegates to zeros_like so all elements are zero. The optional dtype
  # overrides the result type, otherwise the type of a is used. No side effects.
  return zeros_like(a, dtype)


@tf_export.tf_export('experimental.numpy.zeros', v1=[])
@np_utils.np_doc('zeros')
def zeros(shape, dtype=float):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # Creates an array of zeros with the given shape. The dtype (python, NumPy or TF type) is
  # resolved with NumPy-style type rules; if a false value such as nothing is passed, the
  # library default float type is used instead. Returns a tensor/ndarray; no side effects.
  dtype = (
      np_utils.result_type(dtype) if dtype else np_dtypes.default_float_type()
  )
  return array_ops.zeros(shape, dtype=dtype)


@tf_export.tf_export('experimental.numpy.zeros_like', v1=[])
@np_utils.np_doc('zeros_like')
def zeros_like(a, dtype=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Returns zeros with the same shape as the array-like a. The result dtype is the requested
  # dtype if given, otherwise the type inferred from a; it is converted to a TF dtype before
  # calling the underlying zeros-like op (a workaround for a known bug reference). No side effects.
  dtype = np_utils.result_type_unary(a, dtype)

  dtype = dtypes.as_dtype(dtype)  # Work around b/149877262
  return array_ops.zeros_like(a, dtype)


@tf_export.tf_export('experimental.numpy.ones', v1=[])
@np_utils.np_doc('ones')
def ones(shape, dtype=float):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # Creates an array of ones with the given shape. If dtype is truthy it is resolved with
  # NumPy-style type rules; if it is a false value (nothing), it is passed through unresolved so
  # the underlying TF op chooses its own default type (float32), unlike zeros which uses the
  # library default float type. Returns the new tensor; no side effects.
  if dtype:
    dtype = np_utils.result_type(dtype)
  return array_ops.ones(shape, dtype=dtype)


@tf_export.tf_export('experimental.numpy.ones_like', v1=[])
@np_utils.np_doc('ones_like')
def ones_like(a, dtype=None):
  # [FUNCTION DOC]
  # Returns an array of ones with the same shape as the array-like a. The result dtype is the
  # requested dtype if given, otherwise the type inferred from a. No side effects.
  dtype = np_utils.result_type_unary(a, dtype)
  return array_ops.ones_like(a, dtype)


@tf_export.tf_export('experimental.numpy.eye', v1=[])
@np_utils.np_doc('eye')
def eye(N, M=None, k=0, dtype=float):  # pylint: disable=invalid-name,missing-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: the columns argument M is tested for falsiness, so an explicit M of zero is
  # silently replaced by N instead of producing an array with zero columns as NumPy does.
  # Builds an N by M matrix with ones on the k-th diagonal (positive k above the main diagonal,
  # negative below) and zeros elsewhere. M defaults to N, k defaults to 0, dtype defaults to the
  # python float type. N, M and k are converted to integers. If the diagonal lies completely
  # outside the matrix a zero matrix is returned. For k equal to zero the TF identity op is
  # used; otherwise the exact diagonal length is computed and a matrix-diagonal op is applied.
  # Returns the matrix; no side effects.
  if dtype:
    dtype = np_utils.result_type(dtype)
  if not M:
    M = N
  # Making sure N, M and k are `int`
  N = int(N)
  M = int(M)
  k = int(k)
  if k >= M or -k >= N:
    # tf.linalg.diag will raise an error in this case
    return zeros([N, M], dtype=dtype)
  if k == 0:
    return linalg_ops.eye(N, M, dtype=dtype)
  # We need the precise length, otherwise tf.linalg.diag will raise an error
  diag_len = builtins.min(N, M)
  if k > 0:
    if N >= M:
      diag_len -= k
    elif N + k > M:
      diag_len = M - k
  elif k <= 0:
    if M >= N:
      diag_len += k
    elif M - k > N:
      diag_len = N + k
  diagonal_ = array_ops.ones([diag_len], dtype=dtype)
  return array_ops.matrix_diag(diagonal=diagonal_, num_rows=N, num_cols=M, k=k)


@tf_export.tf_export('experimental.numpy.identity', v1=[])
@np_utils.np_doc('identity')
def identity(n, dtype=float):
  # [FUNCTION DOC]
  # Returns the n by n identity matrix of the given dtype (default python float) by delegating
  # to eye with equal row and column counts. No side effects.
  return eye(N=n, M=n, dtype=dtype)


@tf_export.tf_export('experimental.numpy.full', v1=[])
@np_utils.np_doc('full')
def full(shape, fill_value, dtype=None):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # Creates an array of the given shape filled with fill_value. The shape is converted to an
  # integer (int32 by default) array of at least one dimension, the fill value is converted to an
  # array of the optional dtype (inferred from the value when nothing), and the value is then
  # broadcast to the shape. Also used by broadcast_to. Returns the new tensor.
  if not isinstance(shape, np_arrays.ndarray):
    shape = asarray(np_arrays.convert_to_tensor(shape, dtype_hint=np.int32))
  shape = atleast_1d(shape)
  fill_value = asarray(fill_value, dtype=dtype)
  return array_ops.broadcast_to(fill_value, shape)


# Using doc only here since np full_like signature doesn't seem to have the
# shape argument (even though it exists in the documentation online).
@tf_export.tf_export('experimental.numpy.full_like', v1=[])
@np_utils.np_doc_only('full_like')
def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):  # pylint: disable=missing-docstring,redefined-outer-name
  # [FUNCTION DOC]
  # Returns an array shaped like the array-like a, filled with fill_value converted to dtype
  # (defaulting to the type inferred from a). The order, subok and shape arguments exist only for
  # signature compatibility: order must stay the standard value, subok must stay true and shape
  # must be empty. Raises a value error if any of them is changed. No side effects.
  """order, subok and shape arguments mustn't be changed."""
  if order != 'K':
    raise ValueError('Non-standard orders are not supported.')
  if not subok:
    raise ValueError('subok being False is not supported.')
  if shape:
    raise ValueError('Overriding the shape is not supported.')

  a = asarray(a)
  dtype = dtype or np_utils.result_type(a)
  fill_value = asarray(fill_value, dtype=dtype)
  return array_ops.broadcast_to(fill_value, array_ops.shape(a))


def _array_internal(val, dtype=None, copy=True, ndmin=0):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # HELPER: shared implementation behind array, asarray and the dtype-promotion helpers.
  # Converts val to a tensor. Non-tensor inputs get their dtype resolved with NumPy rules and are
  # converted then cast (so incompatible combinations such as a float with an integer dtype are
  # allowed); tensor inputs are cast only if dtype is given. If copy is true an identity op makes
  # a new tensor. If ndmin is greater than zero, dimensions of size one are prepended (decided
  # at run time via a conditional) until the rank is at least ndmin. Raises a value error if ndmin
  # exceeds 32. Returns the resulting tensor; does not mutate val.
  """Main implementation of np.array()."""
  result_t = val

  if not isinstance(result_t, tensor_lib.Tensor):
    dtype = np_utils.result_type_unary(result_t, dtype)
    # We can't call `convert_to_tensor(result_t, dtype=dtype)` here because
    # convert_to_tensor doesn't allow incompatible arguments such as (5.5, int)
    # while np.array allows them. We need to convert-then-cast.

    # EagerTensor conversion complains about "mixed types" when converting
    # tensors with no dtype information. This is because it infers types based
    # on one selected item in the list. So e.g. when converting [2., 2j]
    # to a tensor, it will select float32 as the inferred type and not be able
    # to convert the list to a float 32 tensor.
    # Since we have some information about the final dtype we care about, we
    # supply that information so that convert_to_tensor will do best-effort
    # conversion to that dtype first.
    result_t = np_arrays.convert_to_tensor(result_t, dtype_hint=dtype)
    result_t = math_ops.cast(result_t, dtype=dtype)
  elif dtype:
    result_t = math_ops.cast(result_t, dtype)

  if copy:
    result_t = array_ops.identity(result_t)

  max_ndmin = 32
  if ndmin > max_ndmin:
    raise ValueError(
        f'ndmin bigger than allowable number of dimensions: {max_ndmin}.'
    )

  if ndmin == 0:
    return result_t

  ndims = array_ops.rank(result_t)

  def true_fn():
    # [FUNCTION DOC]
    # HELPER: branch of the ndmin conditional in _array_internal.
    # Takes no arguments. Reshapes the enclosing result tensor by prepending ones to its shape so
    # that its rank becomes ndmin, and returns the reshaped tensor.
    old_shape = array_ops.shape(result_t)
    new_shape = array_ops.concat(
        [array_ops.ones(ndmin - ndims, dtypes.int32), old_shape], axis=0
    )
    return array_ops.reshape(result_t, new_shape)

  result_t = np_utils.cond(
      np_utils.greater(ndmin, ndims), true_fn, lambda: result_t
  )
  return result_t


# TODO(wangpeng): investigate whether we can make `copy` default to False.
# pylint: disable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-doc-return-or-yield,g-doc-args
@tf_export.tf_export('experimental.numpy.array', v1=[])
@np_utils.np_doc_only('array')
def array(val, dtype=None, copy=True, ndmin=0):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a copy is made only if val is on a different device, but the
  # code always inserts an identity op whenever copy is true (the default). The docstring also
  # begins mid-sentence with a blank line, which is a formatting leftover.
  # Creates an array from an array-like val (list, scalar, NumPy array, ndarray or tensor).
  # dtype is optional and resolved with NumPy rules; copy (default true) requests a new tensor;
  # ndmin (default 0, at most 32) is the minimum rank, padded with leading dimensions of size
  # one. Returns the new tensor. Raises a value error when ndmin is too large.
  """Since Tensors are immutable, a copy is made only if val is placed on a

  different device than the current one. Even if `copy` is False, a new Tensor
  may need to be built to satisfy `dtype` and `ndim`. This is used only if `val`
  is an ndarray or a Tensor.
  """  # pylint:disable=g-docstring-missing-newline
  if dtype:
    dtype = np_utils.result_type(dtype)
  return _array_internal(val, dtype, copy, ndmin)


# pylint: enable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-doc-return-or-yield,g-doc-args


@tf_export.tf_export('experimental.numpy.asarray', v1=[])
@np_utils.np_doc('asarray')
def asarray(a, dtype=None):
  # [FUNCTION DOC]
  # Converts a to an array without copying when possible. If a is already an ndarray and either
  # no dtype is requested or it matches, a is returned unchanged; otherwise array is called with
  # copy disabled (still converting dtype as needed). dtype is optional and resolved with
  # NumPy rules. Returns the array; no side effects.
  if dtype:
    dtype = np_utils.result_type(dtype)
  if isinstance(a, np_arrays.ndarray) and (
      not dtype or dtype == a.dtype.as_numpy_dtype
  ):
    return a
  return array(a, dtype, copy=False)


@tf_export.tf_export('experimental.numpy.asanyarray', v1=[])
@np_utils.np_doc('asanyarray')
def asanyarray(a, dtype=None):
  # [FUNCTION DOC]
  # Alias of asarray, since subclasses of arrays are not distinguished. Converts a to an array
  # of the optional dtype and returns it.
  return asarray(a, dtype)


@tf_export.tf_export('experimental.numpy.ascontiguousarray', v1=[])
@np_utils.np_doc('ascontiguousarray')
def ascontiguousarray(a, dtype=None):
  # [FUNCTION DOC]
  # Returns a as an array of the optional dtype with at least one dimension (scalars become
  # one-element vectors). Storage is always contiguous C order, so this just calls array with a
  # minimum rank of one (note that array copies by default).
  return array(a, dtype, ndmin=1)


# Numerical ranges.
@tf_export.tf_export('experimental.numpy.arange', v1=[])
@np_utils.np_doc('arange')
def arange(start, stop=None, step=1, dtype=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: a to-do in the body notes bugs when start or stop is a float and dtype is an
  # integer type (the cast of the range may give results different from NumPy).
  # Returns evenly spaced values from start up to (excluding) stop with the given step. When stop
  # is nothing, start acts as the stop and counting begins at zero. The dtype defaults to the
  # promoted type of the numeric arguments. Raises a value error if step is zero. Returns an
  # empty array when the direction of step cannot reach stop; otherwise a cast TF range.
  """Returns `step`-separated values in the range [start, stop).

  Args:
    start: Start of the interval. Included in the range.
    stop: End of the interval. If not specified, `start` is treated as 0 and
      `start` value is used as `stop`. If specified, it is not included in the
      range if `step` is integer. When `step` is floating point, it may or may
      not be included.
    step: The difference between 2 consecutive values in the output range. It is
      recommended to use `linspace` instead of using non-integer values for
      `step`.
    dtype: Optional. Type of the resulting ndarray. Could be a python type, a
      NumPy type or a TensorFlow `DType`. If not provided, the largest type of
      `start`, `stop`, `step` is used.

  Raises:
    ValueError: If step is zero.
  """
  if not step:
    raise ValueError('step must be non-zero.')
  if dtype:
    dtype = np_utils.result_type(dtype)
  else:
    if stop is None:
      dtype = np_utils.result_type(start, step)
    else:
      dtype = np_utils.result_type(start, step, stop)
  if step > 0 and (
      (stop is not None and start > stop) or (stop is None and start < 0)
  ):
    return array([], dtype=dtype)
  if step < 0 and (
      (stop is not None and start < stop) or (stop is None and start > 0)
  ):
    return array([], dtype=dtype)
  # TODO(srbs): There are some bugs when start or stop is float type and dtype
  # is integer type.
  return math_ops.cast(
      math_ops.range(start, limit=stop, delta=step), dtype=dtype
  )


# Building matrices.
@tf_export.tf_export('experimental.numpy.diag', v1=[])
@np_utils.np_doc('diag')
def diag(v, k=0):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Builds or extracts a diagonal. For a one-dimensional v it returns a square matrix with v on
  # the k-th diagonal (an all-zero square matrix of side abs(k) if v is empty); for a
  # two-dimensional v it returns the k-th diagonal as a vector (empty when k is out of range).
  # k defaults to 0. The choice is made at run time with a conditional on the rank. Raises a value
  # error when the static rank exceeds two and asserts (run-time error) that the rank is 1 or 2.
  """Raises an error if input is not 1- or 2-d."""
  v = asarray(v)
  v_rank = array_ops.rank(v)

  v.shape.with_rank_at_most(2)

  # TODO(nareshmodi): Consider a np_utils.Assert version that will fail during
  # tracing time if the shape is known.
  control_flow_assert.Assert(
      np_utils.logical_or(math_ops.equal(v_rank, 1), math_ops.equal(v_rank, 2)),
      [v_rank],
  )

  def _diag(v, k):
    # [FUNCTION DOC]
    # HELPER: vector case of diag.
    # Given a vector v and offset k, returns a square matrix with v placed on the k-th diagonal,
    # or an all-zero abs(k) by abs(k) matrix when v is empty.
    return np_utils.cond(
        math_ops.equal(array_ops.size(v), 0),
        lambda: array_ops.zeros([abs(k), abs(k)], dtype=v.dtype),
        lambda: array_ops.matrix_diag(v, k=k),
    )

  def _diag_part(v, k):
    # [FUNCTION DOC]
    # HELPER: matrix case of diag.
    # Given a matrix v and offset k, returns the k-th diagonal. If k is outside the matrix bounds
    # the input is swapped for an empty matrix and offset zero (so shape inference of the
    # diagonal-part op does not fail), yielding an empty result.
    v_shape = array_ops.shape(v)
    v, k = np_utils.cond(
        np_utils.logical_or(
            np_utils.less_equal(k, -1 * np_utils.getitem(v_shape, 0)),
            np_utils.greater_equal(k, np_utils.getitem(v_shape, 1)),
        ),
        lambda: (array_ops.zeros([0, 0], dtype=v.dtype), 0),
        lambda: (v, k),
    )
    result = array_ops.matrix_diag_part(v, k=k)
    return result

  result = np_utils.cond(
      math_ops.equal(v_rank, 1), lambda: _diag(v, k), lambda: _diag_part(v, k)
  )
  return result


@tf_export.tf_export('experimental.numpy.diagonal', v1=[])
@np_utils.np_doc('diagonal')
def diagonal(a, offset=0, axis1=0, axis2=1):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Returns the diagonal of a taken over the plane spanned by axis1 and axis2 (defaults 0 and 1)
  # at the given offset (default 0). Fast path when the rank is static, the offset is zero and the
  # axes are the last two. Otherwise the axes are moved to the end and the diagonal part is taken;
  # an out-of-range offset produces an array whose last dimension is zero. Returns the new array.
  a = asarray(a)

  maybe_rank = a.shape.rank
  if (
      maybe_rank is not None
      and offset == 0
      and (axis1 == maybe_rank - 2 or axis1 == -2)
      and (axis2 == maybe_rank - 1 or axis2 == -1)
  ):
    return array_ops.matrix_diag_part(a)

  a = moveaxis(a, (axis1, axis2), (-2, -1))

  a_shape = array_ops.shape(a)

  def _zeros():  # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # HELPER: out-of-range branch of diagonal.
    # Takes no arguments; returns a pair of an all-zero array (same leading dimensions as the
    # input, last dimension zero) and the offset zero, so the diagonal op sees a valid offset.
    return (
        array_ops.zeros(
            array_ops.concat([a_shape[:-1], [0]], 0), dtype=a.dtype
        ),
        0,
    )

  # All zeros since diag_part doesn't handle all possible k (aka offset).
  # Written this way since cond will run shape inference on both branches,
  # and diag_part shape inference will fail when offset is out of bounds.
  a, offset = np_utils.cond(
      np_utils.logical_or(
          np_utils.less_equal(offset, -1 * np_utils.getitem(a_shape, -2)),
          np_utils.greater_equal(offset, np_utils.getitem(a_shape, -1)),
      ),
      _zeros,
      lambda: (a, offset),
  )

  a = array_ops.matrix_diag_part(a, k=offset)
  return a


@tf_export.tf_export('experimental.numpy.diagflat', v1=[])
@np_utils.np_doc('diagflat')
def diagflat(v, k=0):
  # [FUNCTION DOC]
  # Flattens v to one dimension and returns a square matrix with the flattened values on the
  # k-th diagonal (default 0), by delegating to diag. No side effects.
  v = asarray(v)
  return diag(array_ops.reshape(v, [-1]), k)


def _promote_dtype(*arrays):
  # [FUNCTION DOC]
  # HELPER: dtype unification for functions taking several array inputs (where, select, stack...).
  # Computes the common NumPy result type of all inputs and returns a list of arrays converted to
  # it (without copying; arrays already of that type are returned as is).
  dtype = np_utils.result_type(*arrays)

  def _fast_asarray(a):
    # [FUNCTION DOC]
    # HELPER: per-element conversion for _promote_dtype.
    # Returns a unchanged if it is an ndarray of the common dtype, otherwise converts it to that
    # dtype without copying.
    if isinstance(a, np_arrays.ndarray) and dtype == a.dtype.as_numpy_dtype:
      return a
    return _array_internal(a, dtype=dtype, copy=False)

  return [_fast_asarray(a) for a in arrays]


def _promote_dtype_binary(t1, t2):
  # [FUNCTION DOC]
  # HELPER: two-argument version of _promote_dtype, used by binary ops and index updates.
  # Computes the binary result type of t1 and t2 and returns the pair converted to it (no copy;
  # inputs already ndarrays of that type are returned as they are).
  dtype = np_utils._result_type_binary(t1, t2)  # pylint: disable=protected-access
  if not (
      isinstance(t1, np_arrays.ndarray) and dtype == t1.dtype.as_numpy_dtype
  ):
    t1 = _array_internal(t1, dtype=dtype, copy=False)
  if not (
      isinstance(t2, np_arrays.ndarray) and dtype == t2.dtype.as_numpy_dtype
  ):
    t2 = _array_internal(t2, dtype=dtype, copy=False)
  return t1, t2


@tf_export.tf_export('experimental.numpy.all', v1=[])
@np_utils.np_doc('all')
def all(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Logical AND reduction. Converts a to a boolean array and reduces over axis (nothing means all
  # axes; integer, sequence or tensor of axes) optionally keeping reduced dimensions (keepdims).
  # Shadows the Python builtin inside this module. Returns a boolean array.
  a = asarray(a, dtype=bool)
  return math_ops.reduce_all(input_tensor=a, axis=axis, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.any', v1=[])
@np_utils.np_doc('any')
def any(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Logical OR reduction. Converts a to a boolean array and reduces over axis (nothing means all
  # axes) optionally keeping reduced dimensions (keepdims). Shadows the Python builtin inside
  # this module. Returns a boolean array.
  a = asarray(a, dtype=bool)
  return math_ops.reduce_any(input_tensor=a, axis=axis, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.compress', v1=[])
@np_utils.np_doc('compress')
def compress(condition, a, axis=None):  # pylint: disable=redefined-outer-name,missing-function-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: an out-of-range axis is checked with a plain assertion, so it raises an
  # assertion error (and is skipped under optimized Python) rather than a proper axis error.
  # Selects slices of a along axis where the one-dimensional boolean condition is true. A scalar a
  # is treated as a one-element vector; axis nothing flattens a and uses axis 0; negative axes
  # count from the end. A condition shorter than the axis is padded with false. Raises a value
  # error if condition is not one-dimensional. Returns the selected elements.
  condition = asarray(condition, dtype=bool)
  a = asarray(a)

  if condition.ndim != 1:
    raise ValueError('condition must be a 1-d array.')
  # `np.compress` treats scalars as 1-d arrays.
  if a.ndim == 0:
    a = ravel(a)

  if axis is None:
    a = ravel(a)
    axis = 0

  if axis < 0:
    axis += a.ndim

  assert axis >= 0 and axis < a.ndim

  # `tf.boolean_mask` requires the first dimensions of array and condition to
  # match. `np.compress` pads condition with False when it is shorter.
  condition_t = condition
  a_t = a
  if condition.shape[0] < a.shape[axis]:
    padding = array_ops.fill([a.shape[axis] - condition.shape[0]], False)
    condition_t = array_ops.concat([condition_t, padding], axis=0)
  return array_ops.boolean_mask(tensor=a_t, mask=condition_t, axis=axis)


@tf_export.tf_export('experimental.numpy.copy', v1=[])
@np_utils.np_doc('copy')
def copy(a):
  # [FUNCTION DOC]
  # Returns a new array holding the same values as a, always making a copy (via array with copy
  # enabled). Input is array-like; the original is not modified.
  return array(a, copy=True)


def _maybe_promote_to_int(a):
  # [FUNCTION DOC]
  # HELPER: shared by cumsum and cumprod.
  # If the array a has an integer dtype narrower than the default integer type, it is converted
  # to that wider promoted type (NumPy accumulation behavior); otherwise returned unchanged.
  if dtypes.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    a_numpy_dtype = a.dtype.as_numpy_dtype
    output_type = np.promote_types(a_numpy_dtype, int)
    if output_type != a_numpy_dtype:
      a = asarray(a, dtype=output_type)

  return a


@tf_export.tf_export('experimental.numpy.cumprod', v1=[])
@np_utils.np_doc('cumprod')
def cumprod(a, axis=None, dtype=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Cumulative product of a along axis. If axis is nothing the input is flattened first; negative
  # axes count from the end. dtype optionally sets the type of the result; when omitted, small
  # integer types are promoted to the default integer. Returns the cumulative product array.
  a = asarray(a, dtype=dtype)

  if dtype is None:
    a = _maybe_promote_to_int(a)

  # If axis is None, the input is flattened.
  if axis is None:
    a = ravel(a)
    axis = 0
  elif axis < 0:
    axis += array_ops.rank(a)
  return math_ops.cumprod(a, axis)


@tf_export.tf_export('experimental.numpy.cumsum', v1=[])
@np_utils.np_doc('cumsum')
def cumsum(a, axis=None, dtype=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Cumulative sum of a along axis. If axis is nothing the input is flattened first; negative
  # axes count from the end. dtype optionally sets the type of the result; when omitted, small
  # integer types are promoted to the default integer. Returns the cumulative sum array.
  a = asarray(a, dtype=dtype)

  if dtype is None:
    a = _maybe_promote_to_int(a)

  # If axis is None, the input is flattened.
  if axis is None:
    a = ravel(a)
    axis = 0
  elif axis < 0:
    axis += array_ops.rank(a)
  return math_ops.cumsum(a, axis)


@tf_export.tf_export('experimental.numpy.imag', v1=[])
@np_utils.np_doc('imag')
def imag(val):
  # [FUNCTION DOC]
  # KNOWN ISSUE: NumPy returns a scalar for scalar input, but this always returns an array (a
  # to-do in the body acknowledges it).
  # Returns the imaginary part of the array-like val as an array (zeros for real input).
  val = asarray(val)
  # TODO(srbs): np.imag returns a scalar if `val` is a scalar, whereas we always
  # return an ndarray.
  return math_ops.imag(val)


_TO_INT_ = 0
_TO_FLOAT = 1


def _reduce(
    tf_fn,
    a,
    axis=None,
    dtype=None,
    keepdims=None,
    promote_int=_TO_INT_,
    tf_bool_fn=None,
    preserve_bool=False,
):
  # [FUNCTION DOC]
  # HELPER: common engine behind sum, prod, mean, amax, amin, var and std.
  # Applies the TF reduction function tf_fn to array a over axis (nothing means all) with
  # keepdims (nothing means false). dtype optionally forces the working type. For boolean work
  # (requested bool dtype, or bool input with preserve_bool) the alternative tf_bool_fn is used if
  # given. Without dtype, integer/bool inputs are promoted according to promote_int: the
  # integer-promotion marker widens to the default (or 32-bit under automatic dtype conversion)
  # integer width, the float marker casts to the default float, nothing leaves it unchanged.
  # Tensor axes of types other than int32/int64 are cast to int64. Returns the reduced array.
  """A general reduction function.

  Args:
    tf_fn: the TF reduction function.
    a: the array to be reduced.
    axis: (optional) the axis along which to do the reduction. If None, all
      dimensions are reduced.
    dtype: (optional) the dtype of the result.
    keepdims: (optional) whether to keep the reduced dimension(s).
    promote_int: how to promote integer and bool inputs. There are three
      choices. (1) `_TO_INT_` always promotes them to np.int_ or np.uint; (2)
      `_TO_FLOAT` always promotes them to a float type (determined by
      dtypes.default_float_type); (3) None: don't promote.
    tf_bool_fn: (optional) the TF reduction function for bool inputs. It will
      only be used if `dtype` is explicitly set to `np.bool_` or if `a`'s dtype
      is `np.bool_` and `preserve_bool` is True.
    preserve_bool: a flag to control whether to use `tf_bool_fn` if `a`'s dtype
      is `np.bool_` (some reductions such as np.sum convert bools to integers,
      while others such as np.max preserve bools.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = np_utils.result_type(dtype)
  if keepdims is None:
    keepdims = False
  a = asarray(a, dtype=dtype)
  if (
      dtype == np.bool_ or preserve_bool and a.dtype == np.bool_
  ) and tf_bool_fn is not None:
    return tf_bool_fn(input_tensor=a, axis=axis, keepdims=keepdims)
  if dtype is None:
    dtype = a.dtype.as_numpy_dtype
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
      if promote_int == _TO_INT_:
        # If a is an integer/bool type and whose bit width is less than np.int_,
        # numpy up-casts it to np.int_ based on the documentation at
        # https://numpy.org/doc/1.18/reference/generated/numpy.sum.html
        if dtype == np.bool_:
          is_signed = True
          width = 8  # We can use any number here that is less than 64
        else:
          is_signed = np.issubdtype(dtype, np.signedinteger)
          width = np.iinfo(dtype).bits
        # Numpy int_ and uint are defined as 'long' and 'unsigned long', so
        # should have the same bit width.
        if ops.is_auto_dtype_conversion_enabled():
          # We default to 32 bits when using auto dtype conversion semantics.
          if width < np.iinfo(np.int32).bits:
            if is_signed:
              dtype = np.int32
            else:
              dtype = np.uint32
        else:
          if width < np.iinfo(np.int_).bits:
            if is_signed:
              dtype = np.int_
            else:
              dtype = np.uint
        a = math_ops.cast(a, dtype)
      elif promote_int == _TO_FLOAT:
        # Use a default float type.
        a = math_ops.cast(a, np_utils.result_type(float))

  if isinstance(axis, tensor_lib.Tensor) and axis.dtype not in (
      dtypes.int32,
      dtypes.int64,
  ):
    axis = math_ops.cast(axis, dtypes.int64)

  return tf_fn(input_tensor=a, axis=axis, keepdims=keepdims)


# TODO (DarrenZhang01): Add `axis` support to the `size` API.
@tf_export.tf_export('experimental.numpy.size', v1=[])
@np_utils.np_doc('size')
def size(x, axis=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: the axis argument exists for NumPy compatibility but is not supported; passing
  # anything other than nothing raises an error (see the to-do above the function).
  # Returns the total number of elements of x. Python/NumPy scalar numbers give 1; for arrays
  # with a fully known static shape the product of the dimensions is returned as an integer,
  # otherwise a dynamic size tensor. Raises not-implemented if axis is given.
  if axis is not None:
    raise NotImplementedError(
        'axis argument is not supported in the current `np.size` implementation'
    )
  if isinstance(x, (int, float, np.int32, np.int64, np.float32, np.float64)):
    return 1
  x = asarray(x)
  if x.shape.is_fully_defined():
    return np.prod(x.shape.as_list(), dtype=int)
  else:
    return array_ops.size_v2(x)


@tf_export.tf_export('experimental.numpy.sum', v1=[])
@np_utils.np_doc('sum')
def sum(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Sum of array elements over axis (nothing means all axes). dtype optionally sets the result
  # type; small integer and boolean inputs are promoted to the default integer, and a requested
  # boolean dtype uses logical OR. keepdims keeps reduced dimensions. Shadows the builtin sum in
  # this module. Returns the reduced array.
  return _reduce(
      math_ops.reduce_sum,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      tf_bool_fn=math_ops.reduce_any,
  )


@tf_export.tf_export('experimental.numpy.prod', v1=[])
@np_utils.np_doc('prod')
def prod(a, axis=None, dtype=None, keepdims=None):
  # [FUNCTION DOC]
  # Product of array elements over axis (nothing means all axes). dtype optionally sets the
  # result type; small integer and boolean inputs are promoted to the default integer, and a
  # boolean dtype uses logical AND. keepdims keeps reduced dimensions. Returns the reduced array.
  return _reduce(
      math_ops.reduce_prod,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      tf_bool_fn=math_ops.reduce_all,
  )


@tf_export.tf_export('experimental.numpy.mean', v1=[])
@np_utils.np_doc('mean', unsupported_params=['out'])
def mean(a, axis=None, dtype=None, out=None, keepdims=None):
  # [FUNCTION DOC]
  # Arithmetic mean over axis (nothing means all axes). Integer and boolean inputs are promoted
  # to the default float type unless dtype is given. keepdims keeps reduced dimensions. The out
  # argument is unsupported: a non-null value raises a value error. Returns the mean array.
  if out is not None:
    raise ValueError('Setting out is not supported.')
  return _reduce(
      math_ops.reduce_mean,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      promote_int=_TO_FLOAT,
  )


@tf_export.tf_export('experimental.numpy.amax', v1=[])
@np_utils.np_doc('amax', unsupported_params=['out'])
def amax(a, axis=None, out=None, keepdims=None):
  # [FUNCTION DOC]
  # Maximum over axis (nothing means all axes), keeping the input dtype; boolean input stays
  # boolean (computed with logical OR). keepdims keeps reduced dimensions. The out argument is
  # unsupported and raises a value error when given. Returns the maximum array.
  if out is not None:
    raise ValueError('Setting out is not supported.')
  return _reduce(
      math_ops.reduce_max,
      a,
      axis=axis,
      dtype=None,
      keepdims=keepdims,
      promote_int=None,
      tf_bool_fn=math_ops.reduce_any,
      preserve_bool=True,
  )


@tf_export.tf_export('experimental.numpy.amin', v1=[])
@np_utils.np_doc('amin', unsupported_params=['out'])
def amin(a, axis=None, out=None, keepdims=None):
  # [FUNCTION DOC]
  # Minimum over axis (nothing means all axes), keeping the input dtype; boolean input stays
  # boolean (computed with logical AND). keepdims keeps reduced dimensions. The out argument is
  # unsupported and raises a value error when given. Returns the minimum array.
  if out is not None:
    raise ValueError('Setting out is not supported.')
  return _reduce(
      math_ops.reduce_min,
      a,
      axis=axis,
      dtype=None,
      keepdims=keepdims,
      promote_int=None,
      tf_bool_fn=math_ops.reduce_all,
      preserve_bool=True,
  )


@tf_export.tf_export('experimental.numpy.var', v1=[])
@np_utils.np_doc('var')
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: when ddof is non-zero the inner reducer casts its result to dtype even when
  # dtype is nothing, which is likely to fail; it also compares a possibly multi-valued axis with
  # zero, so only a single integer axis (or nothing) works for non-zero ddof.
  # Variance over axis (nothing means all axes). dtype optionally sets the result type; integers
  # are promoted to float. ddof is the delta degrees of freedom subtracted from the count
  # (default 0, handled by a manual computation since the TF variance op lacks it). out is
  # unsupported (value error). keepdims keeps reduced dimensions. Returns the variance array.
  if dtype:
    working_dtype = np_utils.result_type(a, dtype)
  else:
    working_dtype = None
  if out is not None:
    raise ValueError('Setting out is not supported.')
  if ddof != 0:
    # TF reduce_variance doesn't support ddof, so calculate it using raw ops.
    def reduce_fn(input_tensor, axis, keepdims):
      # [FUNCTION DOC]
      # HELPER: variance with a delta-degrees-of-freedom correction, used by var when ddof is
      # non-zero. Takes the tensor, the axis (nothing for all) and keepdims; computes the mean,
      # the sum of squared deviations (using the modulus for complex input), divides by the
      # element count minus ddof, and casts to the enclosing dtype. Returns the result tensor.
      means = math_ops.reduce_mean(input_tensor, axis=axis, keepdims=True)
      centered = input_tensor - means
      if input_tensor.dtype in (dtypes.complex64, dtypes.complex128):
        centered = math_ops.cast(
            math_ops.real(centered * math_ops.conj(centered)),
            input_tensor.dtype,
        )
      else:
        centered = math_ops.square(centered)
      squared_deviations = math_ops.reduce_sum(
          centered, axis=axis, keepdims=keepdims
      )

      if axis is None:
        n = array_ops.size(input_tensor)
      else:
        if axis < 0:
          axis += array_ops.rank(input_tensor)
        n = math_ops.reduce_prod(
            array_ops.gather(array_ops.shape(input_tensor), axis)
        )
      n = math_ops.cast(n - ddof, input_tensor.dtype)

      return math_ops.cast(math_ops.divide(squared_deviations, n), dtype)

  else:
    reduce_fn = math_ops.reduce_variance

  result = _reduce(
      reduce_fn,
      a,
      axis=axis,
      dtype=working_dtype,
      keepdims=keepdims,
      promote_int=_TO_FLOAT,
  )
  if dtype:
    result = math_ops.cast(result, dtype)
  return result


@tf_export.tf_export('experimental.numpy.std', v1=[])
@np_utils.np_doc('std')
def std(a, axis=None, keepdims=None):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: unlike NumPy (and unlike var in this module) the signature has no dtype, out or
  # ddof parameters, so those cannot be passed.
  # Standard deviation over axis (nothing means all axes); integers are promoted to the default
  # float. keepdims keeps reduced dimensions. Returns the standard deviation array.
  return _reduce(
      math_ops.reduce_std,
      a,
      axis=axis,
      dtype=None,
      keepdims=keepdims,
      promote_int=_TO_FLOAT,
  )


@tf_export.tf_export('experimental.numpy.ravel', v1=[])
@np_utils.np_doc('ravel')
def ravel(a):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Flattens the array-like a into a one-dimensional array in row-major order. Returns the
  # reshaped array; the input is not modified.
  a = asarray(a)
  return array_ops.reshape(a, [-1])


@tf_export.tf_export('experimental.numpy.real', v1=[])
@np_utils.np_doc('real')
def real(val):
  # [FUNCTION DOC]
  # KNOWN ISSUE: NumPy returns a scalar for scalar input, but this always returns an array (a
  # to-do in the body acknowledges it).
  # Returns the real part of the array-like val as an array.
  val = asarray(val)
  # TODO(srbs): np.real returns a scalar if val is a scalar, whereas we always
  # return an ndarray.
  return math_ops.real(val)


@tf_export.tf_export('experimental.numpy.repeat', v1=[])
@np_utils.np_doc('repeat')
def repeat(a, repeats, axis=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Repeats elements of a. repeats is an integer or sequence/array of per-element counts; axis
  # (nothing means flatten) selects the dimension. When the input shape is fully known the output
  # static shape is computed on a best-effort basis and set on the result, so shape information
  # survives. Returns the repeated array; the input is not modified.
  a = asarray(a)
  original_shape = a._shape_as_list()  # pylint: disable=protected-access
  # Best effort recovery of the shape.
  known_shape = original_shape is not None and None not in original_shape
  if known_shape:
    if not original_shape:
      original_shape = (repeats,)
    else:
      repeats_np = np.ravel(np.array(repeats))
      if repeats_np.size == 1:
        repeats_np = repeats_np.item()
        if axis is None:
          original_shape = (repeats_np * np.prod(original_shape),)
        else:
          original_shape[axis] = repeats_np * original_shape[axis]
      else:
        if axis is None:
          original_shape = (repeats_np.sum(),)
        else:
          original_shape[axis] = repeats_np.sum()

  repeats = asarray(repeats)
  result = array_ops.repeat(a, repeats, axis)
  if known_shape:
    result.set_shape(original_shape)

  return result


@tf_export.tf_export('experimental.numpy.around', v1=[])
@np_utils.np_doc('around')
def around(a, decimals=0):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Rounds a to the given number of decimals (default 0; may be negative) by scaling by a power
  # of ten, rounding to nearest-even, and unscaling. Integer inputs are processed in the default
  # float type and cast back to the original dtype. Also installed as the round hook of the
  # ndarray class. Returns an array of the input dtype.
  a = asarray(a)
  dtype = a.dtype.as_numpy_dtype
  factor = math.pow(10, decimals)
  if np.issubdtype(dtype, np.inexact):
    factor = math_ops.cast(factor, dtype)
  else:
    # Use float as the working dtype when a.dtype is exact (e.g. integer),
    # because `decimals` can be negative.
    float_dtype = np_utils.result_type(float)
    a = a.astype(float_dtype)
    factor = math_ops.cast(factor, float_dtype)
  a = math_ops.multiply(a, factor)
  a = math_ops.round(a)
  a = math_ops.divide(a, factor)
  return a.astype(dtype)


setattr(np_arrays.ndarray, '__round__', around)


@tf_export.tf_export('experimental.numpy.reshape', v1=[])
@np_utils.np_doc('reshape')
def reshape(a, newshape, order='C'):
  # [FUNCTION DOC]
  # Gives a new shape to the array-like a. newshape is an integer or a sequence of dimensions
  # (one may be minus one, to be inferred). order is 'C' (row-major, default) or 'F'
  # (column-major, implemented by transposing, reshaping with the reversed shape, and
  # transposing back). Raises a value error for any other order. Returns the reshaped array.
  """order argument can only b 'C' or 'F'."""
  if order not in {'C', 'F'}:
    raise ValueError('Unsupported order argument {}'.format(order))

  a = asarray(a)
  if isinstance(newshape, int):
    newshape = [newshape]

  if order == 'F':
    r = array_ops.transpose(
        array_ops.reshape(array_ops.transpose(a), newshape[::-1])
    )
  else:
    r = array_ops.reshape(a, newshape)

  return r


def _reshape_method_wrapper(a, *newshape, **kwargs):
  # [FUNCTION DOC]
  # HELPER: adapts reshape to the method-call style of the ndarray class (dimensions passed
  # as separate positional arguments or as one sequence).
  # a is the array; newshape collects the dimensions; the only accepted keyword is order
  # (default 'C'). A single non-integer positional argument is treated as the whole shape.
  # Raises a value error if any other keyword is supplied. Returns the reshaped array.
  order = kwargs.pop('order', 'C')
  if kwargs:
    raise ValueError('Unsupported arguments: {}'.format(kwargs.keys()))

  if len(newshape) == 1 and not isinstance(newshape[0], int):
    newshape = newshape[0]

  return reshape(a, newshape, order=order)


@tf_export.tf_export('experimental.numpy.expand_dims', v1=[])
@np_utils.np_doc('expand_dims')
def expand_dims(a, axis):
  # [FUNCTION DOC]
  # Inserts a new dimension of size one into the shape of array-like a at position axis
  # (negative counts from the end). Returns the new array.
  a = asarray(a)
  return array_ops.expand_dims(a, axis=axis)


@tf_export.tf_export('experimental.numpy.squeeze', v1=[])
@np_utils.np_doc('squeeze')
def squeeze(a, axis=None):
  # [FUNCTION DOC]
  # Removes dimensions of size one from the array-like a. axis (nothing for all such dimensions)
  # may be an integer or sequence selecting which ones to remove; the TF op raises an invalid
  # argument error if a selected dimension is not of size one. Returns the new array.
  a = asarray(a)
  return array_ops.squeeze(a, axis)


@tf_export.tf_export('experimental.numpy.flatten', v1=[])
@np_utils.np_doc('flatten', link=np_utils.NoLink())
def flatten(a, order='C'):
  # [FUNCTION DOC]
  # Returns a one-dimensional copy-like flattening of a. order 'C', 'A' or 'K' flatten in
  # row-major order; 'F' flattens in column-major order (via transpose). Any other order raises
  # a value error. Default order is 'C'.
  a = asarray(a)
  if order == 'C' or order == 'A' or order == 'K':
    # Row major.
    return array_ops.reshape(a, [-1])
  elif order == 'F':
    # Column major
    return array_ops.reshape(array_ops.transpose(a), [-1])
  else:
    raise ValueError(
        'order can only be C, A, K (all row major) or F (column major).'
    )


@tf_export.tf_export('experimental.numpy.transpose', v1=[])
@np_utils.np_doc('transpose')
def transpose(a, axes=None):
  # [FUNCTION DOC]
  # Permutes the dimensions of a. axes is an optional permutation of dimension indices; nothing
  # reverses the order of the dimensions. Returns the transposed array.
  a = asarray(a)
  if axes is not None:
    axes = asarray(axes)
  return array_ops.transpose(a=a, perm=axes)


@tf_export.tf_export('experimental.numpy.swapaxes', v1=[])
@np_utils.np_doc('swapaxes')
def swapaxes(a, axis1, axis2):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Interchanges two axes of a. axis1 and axis2 are integers or scalar tensors, negative values
  # count from the end. When the rank and both axes are static, a static permutation is built
  # (needed for XLA compile-time constants); otherwise the permutation is computed dynamically.
  # Returns the transposed array.
  a = asarray(a)

  def adjust_axes(axes, rank):
    # [FUNCTION DOC]
    # HELPER: normalizes negative axes for swapaxes.
    # Applies a per-axis fix-up over a (possibly nested) structure of axes using the given rank,
    # and returns a structure of the same form with non-negative axes.
    def f(x):
      # [FUNCTION DOC]
      # HELPER: fix-up for one axis. A negative python integer gets the rank added; for a tensor
      # axis a conditional select adds the enclosing array's rank (a_rank) when negative.
      # Returns the adjusted axis.
      if isinstance(x, int):
        if x < 0:
          x = x + rank
      else:
        x = array_ops.where_v2(x < 0, np_utils.add(x, a_rank), x)
      return x

    return nest.map_structure(f, axes)

  if (
      a.shape.rank is not None
      and isinstance(axis1, int)
      and isinstance(axis2, int)
  ):
    # This branch makes sure `perm` is statically known, to avoid a
    # not-compile-time-constant XLA error.
    a_rank = a.shape.rank
    axis1, axis2 = adjust_axes((axis1, axis2), a_rank)
    perm = list(range(a_rank))
    perm[axis1] = axis2
    perm[axis2] = axis1
  else:
    a_rank = array_ops.rank(a)
    axis1, axis2 = adjust_axes((axis1, axis2), a_rank)
    perm = math_ops.range(a_rank)
    perm = array_ops.tensor_scatter_update(
        perm, [[axis1], [axis2]], [axis2, axis1]
    )
  a = array_ops.transpose(a, perm)
  return a


@tf_export.tf_export('experimental.numpy.moveaxis', v1=[])
@np_utils.np_doc('moveaxis')
def moveaxis(a, source, destination):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised when source or destination is out of
  # range, but the code never validates that; it only raises when the two sequences differ in
  # length. When both are empty or zero-valued (falsy) it returns the input without converting it
  # to an array.
  # Moves the axes in source to the positions in destination (integers or sequences; negative
  # values count from the end), keeping the remaining axes in order. Uses a static permutation
  # when the rank is known, otherwise builds it dynamically with scatter operations. Returns the
  # transposed array.
  """Raises ValueError if source, destination not in (-ndim(a), ndim(a))."""
  if not source and not destination:
    return a

  a = asarray(a)

  if isinstance(source, int):
    source = (source,)
  if isinstance(destination, int):
    destination = (destination,)
  if len(source) != len(destination):
    raise ValueError('The lengths of source and destination must equal')

  a_rank = np_utils._maybe_static(array_ops.rank(a))  # pylint: disable=protected-access

  def _correct_axis(axis, rank):
    # [FUNCTION DOC]
    # HELPER: for moveaxis. Returns axis plus rank if axis is negative, else axis unchanged.
    if axis < 0:
      return axis + rank
    return axis

  source = tuple(_correct_axis(axis, a_rank) for axis in source)
  destination = tuple(_correct_axis(axis, a_rank) for axis in destination)

  if a.shape.rank is not None:
    perm = [i for i in range(a_rank) if i not in source]
    for dest, src in sorted(zip(destination, source)):
      assert dest <= len(perm)
      perm.insert(dest, src)
  else:
    r = math_ops.range(a_rank)

    def _remove_indices(a, b):
      # [FUNCTION DOC]
      # HELPER: for the dynamic-rank path of moveaxis.
      # Given a one-dimensional tensor a and a sequence of indices b, returns a with the elements
      # at positions b removed (b is sorted first and the pieces between them concatenated).
      """Remove indices (`b`) from `a`."""
      items = array_ops_stack.unstack(
          sort_ops.sort(array_ops_stack.stack(b)), num=len(b)
      )

      i = 0
      result = []

      for item in items:
        result.append(a[i:item])
        i = item + 1

      result.append(a[i:])

      return array_ops.concat(result, 0)

    minus_sources = _remove_indices(r, source)
    minus_dest = _remove_indices(r, destination)

    perm = array_ops.scatter_nd(
        array_ops.expand_dims(minus_dest, 1), minus_sources, [a_rank]
    )
    perm = array_ops.tensor_scatter_update(
        perm, array_ops.expand_dims(destination, 1), source
    )
  a = array_ops.transpose(a, perm)

  return a


@tf_export.tf_export('experimental.numpy.pad', v1=[])
@np_utils.np_doc('pad')
def pad(array, pad_width, mode, **kwargs):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # Pads an array. array is array-like, pad_width gives before/after counts per dimension
  # (converted to int32), mode must be 'constant', 'reflect' or 'symmetric' (otherwise a value
  # error is raised). The only keyword used is constant_values (default 0) for constant mode;
  # other keywords are silently ignored. Returns the padded array.
  """Only supports modes 'constant', 'reflect' and 'symmetric' currently."""
  constant_values = kwargs.get('constant_values', 0)
  if not (mode == 'constant' or mode == 'reflect' or mode == 'symmetric'):
    raise ValueError('Unsupported padding mode: ' + mode)
  mode = mode.upper()
  array = asarray(array)
  pad_width = asarray(pad_width, dtype=dtypes.int32)
  return array_ops.pad(
      tensor=array,
      paddings=pad_width,
      mode=mode,
      constant_values=constant_values,
  )


@tf_export.tf_export('experimental.numpy.take', v1=[])
@np_utils.np_doc('take')
def take(a, indices, axis=None, out=None, mode='clip'):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the default mode is 'clip' whereas NumPy defaults to raising on bad indices, and
  # the 'raise' mode is accepted by validation but then rejected with a value error.
  # Takes elements of a at indices along axis (nothing flattens a first). mode 'clip' clamps
  # indices to the valid range, 'wrap' wraps them around. out must be nothing (value error
  # otherwise); any mode other than raise/clip/wrap is a value error. Returns the gathered array.
  """out argument is not supported, and default mode is clip."""
  if out is not None:
    raise ValueError('out argument is not supported in take.')

  if mode not in {'raise', 'clip', 'wrap'}:
    raise ValueError("Invalid mode '{}' for take".format(mode))

  a = asarray(a)
  indices = asarray(indices)

  if axis is None:
    a = array_ops.reshape(a, [-1])
    axis = 0

  axis_size = array_ops.shape(a, out_type=indices.dtype)[axis]
  if mode == 'clip':
    indices = clip_ops.clip_by_value(indices, 0, axis_size - 1)
  elif mode == 'wrap':
    indices = math_ops.floormod(indices, axis_size)
  else:
    raise ValueError("The 'raise' mode to take is not supported.")

  return array_ops.gather(a, indices, axis=axis)


@tf_export.tf_export('experimental.numpy.where', v1=[])
@np_utils.np_doc_only('where')
def where(condition, x=None, y=None):
  # [FUNCTION DOC]
  # Chooses elements by a boolean condition. With only condition, returns the indices of true
  # elements (same as nonzero). With both x and y, promotes them to a common dtype and selects
  # from x where true and from y elsewhere (broadcasting). Raises a value error if exactly one of
  # x and y is given.
  """Raises ValueError if exactly one of x or y is not None."""
  condition = asarray(condition, dtype=np.bool_)
  if x is None and y is None:
    return nonzero(condition)
  elif x is not None and y is not None:
    x, y = _promote_dtype(x, y)
    return array_ops.where_v2(condition, x, y)
  raise ValueError('Both x and y must be ndarrays, or both must be None.')


@tf_export.tf_export('experimental.numpy.select', v1=[])
@np_utils.np_doc('select')
def select(condlist, choicelist, default=0):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Returns an array drawn from choicelist according to condlist: for each element the first
  # condition that is true picks the corresponding choice, else default (default 0). Values are
  # promoted to a common dtype. Raises a value error if the lists differ in length or are empty.
  if len(condlist) != len(choicelist):
    msg = 'condlist must have length equal to choicelist ({} vs {})'
    raise ValueError(msg.format(len(condlist), len(choicelist)))
  if not condlist:
    raise ValueError('condlist must be non-empty')
  choices = _promote_dtype(default, *choicelist)
  choicelist = choices[1:]
  output = choices[0]
  # The traversal is in reverse order so we can return the first value in
  # choicelist where condlist is True.
  for cond, choice in zip(condlist[::-1], choicelist[::-1]):
    output = where(cond, choice, output)
  return output


@tf_export.tf_export('experimental.numpy.shape', v1=[])
@np_utils.np_doc(
    'shape',
    link=np_utils.Link(
        'https://numpy.org/doc/1.18/reference/generated/numpy.shape.html'
    ),
)
def shape(a):
  # [FUNCTION DOC]
  # Returns the shape of the array-like a (converted to an array first), as a tensor shape
  # object. No side effects.
  a = asarray(a)
  return a.shape


@tf_export.tf_export('experimental.numpy.ndim', v1=[])
@np_utils.np_doc('ndim', link=np_utils.NoLink())
def ndim(a):
  # [FUNCTION DOC]
  # Returns the number of dimensions (rank) of the array-like a; may be nothing/unknown for
  # tensors of unknown rank in graph mode. Also used by isscalar.
  a = asarray(a)
  return a.ndim


@tf_export.tf_export('experimental.numpy.isscalar', v1=[])
@np_utils.np_doc('isscalar')
def isscalar(num):
  # [FUNCTION DOC]
  # Returns whether the array-like num has zero dimensions (true for scalar numbers and rank-zero
  # arrays).
  return ndim(num) == 0


def _boundaries_to_sizes(a, boundaries, axis):
  # [FUNCTION DOC]
  # HELPER: converts NumPy-style split boundaries into TF-style split sizes for split.
  # a is the array (its static shape along axis must be known), boundaries is a sequence of
  # increasing split positions, axis is the dimension. Sizes are clipped to the available length,
  # and a final remainder size is appended. Raises a value error if axis is out of bounds or a
  # boundary is smaller than its predecessor. Returns a list of sizes.
  """Converting boundaries of splits to sizes of splits.

  Args:
    a: the array to be split.
    boundaries: the boundaries, as in np.split.
    axis: the axis along which to split.

  Returns:
    A list of sizes of the splits, as in tf.split.
  """
  if axis >= len(a.shape):
    raise ValueError('axis %s is out of bound for shape %s' % (axis, a.shape))
  total_size = a.shape[axis]
  sizes = []
  sizes_sum = 0
  prev = 0
  for i, b in enumerate(boundaries):
    size = b - prev
    if size < 0:
      raise ValueError(
          'The %s-th boundary %s is smaller than the previous boundary %s'
          % (i, b, prev)
      )
    size = builtins.min(size, builtins.max(0, total_size - sizes_sum))
    sizes.append(size)
    sizes_sum += size
    prev = b
  sizes.append(builtins.max(0, total_size - sizes_sum))
  return sizes


@tf_export.tf_export('experimental.numpy.split', v1=[])
@np_utils.np_doc('split')
def split(ary, indices_or_sections, axis=0):
  # [FUNCTION DOC]
  # Splits ary into sub-arrays along axis (default 0). An integer indices_or_sections gives the
  # number of equal parts; a sequence gives boundary positions (converted to sizes). Returns a
  # list of arrays. Errors come from the size conversion or the TF split op.
  ary = asarray(ary)
  if not isinstance(indices_or_sections, int):
    indices_or_sections = _boundaries_to_sizes(ary, indices_or_sections, axis)
  return array_ops.split(ary, indices_or_sections, axis=axis)


def _split_on_axis(np_fun_name, axis):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # HELPER: factory for vsplit, hsplit and dsplit.
  # Takes the NumPy function name (used for documentation) and the axis to split on, and returns
  # a function of an array and indices-or-sections that splits along that axis.
  @np_utils.np_doc(np_fun_name)
  def f(ary, indices_or_sections):
    # [FUNCTION DOC]
    # HELPER: the generated splitting function. For axis 1 (horizontal split) a one-dimensional
    # input is split along axis 0 instead. If an integer section count is given and the static
    # dimension is not evenly divisible, raises a value error. Returns the list of pieces.
    # for 1-D array, hsplit becomes vsplit
    new_axis = np_utils.cond(
        math_ops.equal(axis, 1),
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            math_ops.equal(array_ops.rank(ary), 1), lambda: 0, lambda: axis
        ),
        lambda: axis,
    )
    if isinstance(indices_or_sections, int):
      ary_shape = ary.shape[new_axis]
      if ary_shape is not None and ary_shape % indices_or_sections:
        raise ValueError('array split does not result in an equal division')
    return split(ary, indices_or_sections, axis=new_axis)

  return f


vsplit = tf_export.tf_export('experimental.numpy.vsplit', v1=[])(
    _split_on_axis('vsplit', axis=0)
)
hsplit = tf_export.tf_export('experimental.numpy.hsplit', v1=[])(
    _split_on_axis('hsplit', axis=1)
)
dsplit = tf_export.tf_export('experimental.numpy.dsplit', v1=[])(
    _split_on_axis('dsplit', axis=2)
)


@tf_export.tf_export('experimental.numpy.broadcast_to', v1=[])
@np_utils.np_doc('broadcast_to')
def broadcast_to(array, shape):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # Broadcasts array to the given shape by delegating to full with the array as the fill value.
  # Returns the broadcast array; incompatible shapes raise the underlying TF error.
  return full(shape, array)


@tf_export.tf_export('experimental.numpy.stack', v1=[])
@np_utils.np_doc('stack')
def stack(arrays, axis=0):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: the unwrapping list comprehension picks the same element in both branches, so it
  # is a no-op left over from an earlier ndarray wrapper design.
  # Joins arrays along a new axis (default 0). If arrays is itself a single array/tensor, it is
  # treated as already stacked along axis 0 and axes are swapped to place the stacking axis.
  # Otherwise elements are promoted to a common dtype and stacked. Returns the stacked array.
  if isinstance(arrays, (np_arrays.ndarray, tensor_lib.Tensor)):
    arrays = asarray(arrays)
    if axis == 0:
      return arrays
    else:
      return swapaxes(arrays, 0, axis)
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return asarray(array_ops_stack.stack(unwrapped_arrays, axis))


@tf_export.tf_export('experimental.numpy.hstack', v1=[])
@np_utils.np_doc('hstack')
def hstack(tup):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the unwrapping list comprehension picks the same element in both branches, so it
  # is a no-op.
  # Stacks arrays horizontally: each input gets at least one dimension, dtypes are promoted to
  # a common type, and the arrays are concatenated along axis 0 when the first has rank one,
  # otherwise along axis 1 (the choice is made at run time). Returns the combined array.
  arrays = [atleast_1d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  rank = array_ops.rank(unwrapped_arrays[0])
  return np_utils.cond(
      math_ops.equal(rank, 1),
      lambda: array_ops.concat(unwrapped_arrays, axis=0),
      lambda: array_ops.concat(unwrapped_arrays, axis=1),
  )


@tf_export.tf_export('experimental.numpy.vstack', v1=[])
@np_utils.np_doc('vstack')
def vstack(tup):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the unwrapping list comprehension picks the same element in both branches, so it
  # is a no-op.
  # Stacks arrays vertically: each input gets at least two dimensions, dtypes are promoted to a
  # common type, and they are concatenated along axis 0. Returns the combined array.
  arrays = [atleast_2d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return array_ops.concat(unwrapped_arrays, axis=0)


@tf_export.tf_export('experimental.numpy.dstack', v1=[])
@np_utils.np_doc('dstack')
def dstack(tup):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the unwrapping list comprehension picks the same element in both branches, so it
  # is a no-op.
  # Stacks arrays depth-wise: each input gets at least three dimensions, dtypes are promoted to
  # a common type, and they are concatenated along axis 2. Returns the combined array.
  arrays = [atleast_3d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return array_ops.concat(unwrapped_arrays, axis=2)


def _pad_left_to(n, old_shape):
  # [FUNCTION DOC]
  # HELPER: shape-computation callback for atleast_1d and atleast_2d.
  # Given a target rank n and an old shape (sequence of dimensions), returns an int32 array with
  # ones prepended so the length is at least n (nothing added if already long enough).
  old_shape = asarray(old_shape, dtype=np.int32)
  new_shape = array_ops.pad(
      old_shape,
      [[math_ops.maximum(n - array_ops.size(old_shape), 0), 0]],
      constant_values=1,
  )
  return asarray(new_shape)


def _atleast_nd(n, new_shape, *arys):
  # [FUNCTION DOC]
  # HELPER: shared implementation of atleast_1d, atleast_2d and atleast_3d.
  # n is the minimal rank, new_shape a callback mapping (n, old shape) to the new shape, arys the
  # array-like inputs. Each input of lower rank is reshaped (decided at run time); others are
  # kept. Returns a single array when one input was given, otherwise a list of arrays.
  """Reshape arrays to be at least `n`-dimensional.

  Args:
    n: The minimal rank.
    new_shape: a function that takes `n` and the old shape and returns the
      desired new shape.
    *arys: ndarray(s) to be reshaped.

  Returns:
    The reshaped array(s).
  """

  def f(x):
    # [FUNCTION DOC]
    # HELPER: per-input conversion for _atleast_nd. Converts x to an array and, when its rank is
    # below n, reshapes it to the shape from the callback; returns the resulting array.
    # pylint: disable=g-long-lambda
    x = asarray(x)
    return asarray(
        np_utils.cond(
            np_utils.greater(n, array_ops.rank(x)),
            lambda: reshape(x, new_shape(n, array_ops.shape(x))),
            lambda: x,
        )
    )

  arys = list(map(f, arys))
  if len(arys) == 1:
    return arys[0]
  else:
    return arys


@tf_export.tf_export('experimental.numpy.atleast_1d', v1=[])
@np_utils.np_doc('atleast_1d')
def atleast_1d(*arys):
  # [FUNCTION DOC]
  # Converts each input to an array with at least one dimension (scalars become one-element
  # vectors). Returns one array for one input, otherwise a list.
  return _atleast_nd(1, _pad_left_to, *arys)


@tf_export.tf_export('experimental.numpy.atleast_2d', v1=[])
@np_utils.np_doc('atleast_2d')
def atleast_2d(*arys):
  # [FUNCTION DOC]
  # Converts each input to an array with at least two dimensions (ones are prepended to the
  # shape). Returns one array for one input, otherwise a list.
  return _atleast_nd(2, _pad_left_to, *arys)


@tf_export.tf_export('experimental.numpy.atleast_3d', v1=[])
@np_utils.np_doc('atleast_3d')
def atleast_3d(*arys):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Converts each input to an array with at least three dimensions following NumPy placement:
  # a scalar becomes shape 1,1,1; a vector of length N becomes 1,N,1; a matrix gets a trailing
  # dimension of one. Returns one array for one input, otherwise a list.
  def new_shape(_, old_shape):
    # [FUNCTION DOC]
    # HELPER: shape callback for atleast_3d. The first argument (target rank) is ignored.
    # From the old shape it returns 1,1,1 for rank zero, pads one on both sides for rank one,
    # and appends a one otherwise.
    # pylint: disable=g-long-lambda
    ndim_ = array_ops.size(old_shape)
    return np_utils.cond(
        math_ops.equal(ndim_, 0),
        lambda: constant_op.constant([1, 1, 1], dtype=dtypes.int32),
        lambda: np_utils.cond(
            math_ops.equal(ndim_, 1),
            lambda: array_ops.pad(old_shape, [[1, 1]], constant_values=1),
            lambda: array_ops.pad(old_shape, [[0, 1]], constant_values=1),
        ),
    )

  return _atleast_nd(3, new_shape, *arys)


@tf_export.tf_export('experimental.numpy.nonzero', v1=[])
@np_utils.np_doc('nonzero')
def nonzero(a):
  # [FUNCTION DOC]
  # Returns the indices of nonzero elements of a, as a list with one index vector per dimension.
  # Scalars are treated as one-element vectors. Raises a value error if the rank of a is unknown
  # since the number of output arrays cannot be decided.
  a = atleast_1d(a)
  if a.shape.rank is None:
    raise ValueError(
        "The rank of `a` is unknown, so we can't decide how many "
        'arrays to return.'
    )
  return array_ops_stack.unstack(
      array_ops.where_v2(math_ops.cast(a, dtypes.bool)), a.shape.rank, axis=1
  )


@tf_export.tf_export('experimental.numpy.diag_indices', v1=[])
@np_utils.np_doc('diag_indices')
def diag_indices(n, ndim=2):  # pylint: disable=missing-docstring,redefined-outer-name
  # [FUNCTION DOC]
  # Returns the index arrays for the main diagonal of an array with ndim (default 2) dimensions
  # each of size n: a tuple of ndim identical range vectors from 0 to n minus one. Raises a value
  # error if n or ndim is negative.
  if n < 0:
    raise ValueError(
        'n argument to diag_indices must be nonnegative, got {}'.format(n)
    )
  if ndim < 0:
    raise ValueError(
        'ndim argument to diag_indices must be nonnegative, got {}'.format(ndim)
    )

  return (math_ops.range(n),) * ndim


@tf_export.tf_export('experimental.numpy.tri', v1=[])
@np_utils.np_doc('tri')
def tri(N, M=None, k=0, dtype=None):  # pylint: disable=invalid-name,missing-docstring
  # [FUNCTION DOC]
  # Builds an N by M matrix (M defaults to N) with ones at and below the k-th diagonal and zeros
  # elsewhere. dtype defaults to the default float type. Implemented with a band-part op (lower
  # triangle by inverting an upper band for negative k). Returns the matrix.
  M = M if M is not None else N
  if dtype is not None:
    dtype = np_utils.result_type(dtype)
  else:
    # Use a default float type.
    dtype = np_utils.result_type(float)

  if k < 0:
    lower = -k - 1
    if lower > N:
      r = array_ops.zeros([N, M], dtype)
    else:
      # Keep as tf bool, since we create an upper triangular matrix and invert
      # it.
      o = array_ops.ones([N, M], dtype=dtypes.bool)
      r = math_ops.cast(
          math_ops.logical_not(array_ops.matrix_band_part(o, lower, -1)), dtype
      )
  else:
    o = array_ops.ones([N, M], dtype)
    if k > M:
      r = o
    else:
      r = array_ops.matrix_band_part(o, -1, k)
  return r


@tf_export.tf_export('experimental.numpy.tril', v1=[])
@np_utils.np_doc('tril')
def tril(m, k=0):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Returns the lower triangle of m (elements on and below the k-th diagonal, default 0), with the
  # others set to zero, applied to the last two dimensions. Raises a value error if the rank is
  # unknown, below two, or if the last two dimensions are not statically known.
  m = asarray(m)
  if m.shape.ndims is None:
    raise ValueError('Argument to tril should have known rank')
  m_shape = m.shape.as_list()

  if len(m_shape) < 2:
    raise ValueError('Argument to tril must have rank at least 2')

  if m_shape[-1] is None or m_shape[-2] is None:
    raise ValueError(
        'Currently, the last two dimensions of the input array '
        'need to be known.'
    )

  z = constant_op.constant(0, m.dtype)

  mask = tri(*m_shape[-2:], k=k, dtype=bool)
  return array_ops.where_v2(
      array_ops.broadcast_to(mask, array_ops.shape(m)), m, z
  )


@tf_export.tf_export('experimental.numpy.triu', v1=[])
@np_utils.np_doc('triu')
def triu(m, k=0):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Returns the upper triangle of m (elements on and above the k-th diagonal, default 0), with the
  # others set to zero, applied to the last two dimensions. Raises a value error if the rank is
  # unknown, below two, or if the last two dimensions are not statically known.
  m = asarray(m)
  if m.shape.ndims is None:
    raise ValueError('Argument to triu should have known rank')
  m_shape = m.shape.as_list()

  if len(m_shape) < 2:
    raise ValueError('Argument to triu must have rank at least 2')

  if m_shape[-1] is None or m_shape[-2] is None:
    raise ValueError(
        'Currently, the last two dimensions of the input array '
        'need to be known.'
    )

  z = constant_op.constant(0, m.dtype)

  mask = tri(*m_shape[-2:], k=k - 1, dtype=bool)
  return array_ops.where_v2(
      array_ops.broadcast_to(mask, array_ops.shape(m)), z, m
  )


@tf_export.tf_export('experimental.numpy.flip', v1=[])
@np_utils.np_doc('flip')
def flip(m, axis=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Reverses the order of elements of m along axis (an integer, negative counts from the end);
  # nothing reverses all axes. Returns the flipped array.
  m = asarray(m)

  if axis is None:
    return array_ops.reverse(m, math_ops.range(array_ops.rank(m)))

  axis = np_utils._canonicalize_axis(axis, array_ops.rank(m))  # pylint: disable=protected-access

  return array_ops.reverse(m, [axis])


@tf_export.tf_export('experimental.numpy.flipud', v1=[])
@np_utils.np_doc('flipud')
def flipud(m):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Flips m up-down (reverses along axis 0). Returns the flipped array.
  return flip(m, 0)


@tf_export.tf_export('experimental.numpy.fliplr', v1=[])
@np_utils.np_doc('fliplr')
def fliplr(m):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Flips m left-right (reverses along axis 1). Returns the flipped array.
  return flip(m, 1)


@tf_export.tf_export('experimental.numpy.roll', v1=[])
@np_utils.np_doc('roll')
def roll(a, shift, axis=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Circularly shifts elements of a by shift positions (integer or sequence) along axis. With
  # axis nothing the array is flattened, rolled, and restored to its original shape. Returns the
  # rolled array.
  a = asarray(a)

  if axis is not None:
    return manip_ops.roll(a, shift, axis)

  # If axis is None, the roll happens as a 1-d tensor.
  original_shape = array_ops.shape(a)
  a = manip_ops.roll(array_ops.reshape(a, [-1]), shift, 0)
  return array_ops.reshape(a, original_shape)


@tf_export.tf_export('experimental.numpy.rot90', v1=[])
@np_utils.np_doc('rot90')
def rot90(m, k=1, axes=(0, 1)):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: m is not converted with the array conversion first (unlike most functions here),
  # so a plain list input is passed straight to the rank and flip operations.
  # Rotates m by 90 degrees k times (default 1, taken modulo 4) in the plane given by the two
  # axes (default the first two). Zero rotations return m itself; two flip both axes; one or
  # three combine a flip with a transpose. Returns the rotated array.
  m_rank = array_ops.rank(m)
  ax1, ax2 = np_utils._canonicalize_axes(axes, m_rank)  # pylint: disable=protected-access

  k = k % 4
  if k == 0:
    return m
  elif k == 2:
    return flip(flip(m, ax1), ax2)
  else:
    perm = math_ops.range(m_rank)
    perm = array_ops.tensor_scatter_update(perm, [[ax1], [ax2]], [ax2, ax1])

    if k == 1:
      return transpose(flip(m, ax2), perm)
    else:
      return flip(transpose(m, perm), ax2)


@tf_export.tf_export('experimental.numpy.vander', v1=[])
@np_utils.np_doc('vander')
def vander(x, N=None, increasing=False):  # pylint: disable=missing-docstring,invalid-name
  # [FUNCTION DOC]
  # Builds a Vandermonde matrix from the one-dimensional x: columns are powers of x, N columns
  # (default the length of x), in decreasing power order unless increasing is true. Raises a
  # value error for static negative N or non-1-d x; with dynamic values it adds run-time
  # assertions instead. Returns the matrix.
  x = asarray(x)

  x_shape = array_ops.shape(x)
  if N is None:
    N = x_shape[0]

  N_temp = np_utils.get_static_value(N)  # pylint: disable=invalid-name
  if N_temp is not None:
    N = N_temp
    if N < 0:
      raise ValueError('N must be nonnegative')
  else:
    control_flow_assert.Assert(N >= 0, [N])

  rank = array_ops.rank(x)
  rank_temp = np_utils.get_static_value(rank)
  if rank_temp is not None:
    rank = rank_temp
    if rank != 1:
      raise ValueError('x must be a one-dimensional array')
  else:
    control_flow_assert.Assert(math_ops.equal(rank, 1), [rank])

  if increasing:
    start = 0
    limit = N
    delta = 1
  else:
    start = N - 1
    limit = -1
    delta = -1

  x = array_ops.expand_dims(x, -1)
  return math_ops.pow(
      x, math_ops.cast(math_ops.range(start, limit, delta), dtype=x.dtype)
  )


@tf_export.tf_export('experimental.numpy.ix_', v1=[])
@np_utils.np_doc('ix_')
def ix_(*args):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Builds an open mesh from several one-dimensional sequences, for indexing the cross product.
  # Each argument must be integer or boolean (boolean converted to the indices of true values);
  # output i is reshaped to have size -1 along axis i and one elsewhere. Raises a value error for
  # non-1-d arguments or other dtypes (run-time assertion when rank is dynamic). Returns a list.
  n = len(args)
  output = []
  for i, a in enumerate(args):
    a = asarray(a)
    a_rank = array_ops.rank(a)
    a_rank_temp = np_utils.get_static_value(a_rank)
    if a_rank_temp is not None:
      a_rank = a_rank_temp
      if a_rank != 1:
        raise ValueError(
            'Arguments must be 1-d, got arg {} of rank {}'.format(i, a_rank)
        )
    else:
      control_flow_assert.Assert(math_ops.equal(a_rank, 1), [a_rank])

    new_shape = [1] * n
    new_shape[i] = -1
    dtype = a.dtype
    if dtype == dtypes.bool:
      output.append(array_ops.reshape(nonzero(a)[0], new_shape))
    elif dtype.is_integer:
      output.append(array_ops.reshape(a, new_shape))
    else:
      raise ValueError(
          'Only integer and bool dtypes are supported, got {}'.format(dtype)
      )

  return output


@tf_export.tf_export('experimental.numpy.broadcast_arrays', v1=[])
@np_utils.np_doc('broadcast_arrays')
def broadcast_arrays(*args, **kwargs):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Broadcasts several array-likes against each other to a common shape. The only keyword
  # accepted is subok, which must be false; true or any other keyword raises a value error.
  # Returns the list of broadcast arrays.
  subok = kwargs.pop('subok', False)
  if subok:
    raise ValueError('subok=True is not supported.')
  if kwargs:
    raise ValueError('Received unsupported arguments {}'.format(kwargs.keys()))

  args = [asarray(arg) for arg in args]
  return np_utils.tf_broadcast(*args)


@tf_export.tf_export('experimental.numpy.sign', v1=[])
@np_utils.np_doc_only('sign')
def sign(x, out=None, where=None, **kwargs):  # pylint: disable=missing-docstring,redefined-outer-name
  # [FUNCTION DOC]
  # KNOWN ISSUE: out and where are checked by truthiness, so an array-valued argument may be
  # ambiguous or rejected differently than a simple null test, and an explicit false value is
  # silently accepted as if unset.
  # Elementwise sign of x. Unsupported out, where or extra keywords raise a value error. With
  # NumPy 2 or later the TF sign is returned directly (complex sign is x over its modulus there);
  # for older NumPy the complex sign is the sign of the real part cast back to complex.
  if out:
    raise ValueError('tf.numpy doesnt support setting out.')
  if where:
    raise ValueError('tf.numpy doesnt support setting where.')
  if kwargs:
    raise ValueError('tf.numpy doesnt support setting {}'.format(kwargs.keys()))

  x = asarray(x)

  # Numpy 2.x and later uses the same definition of sign.
  if np.lib.NumpyVersion(np.__version__) >= '2.0.0.dev0':
    return math_ops.sign(x)

  dtype = x.dtype.as_numpy_dtype
  if np.issubdtype(dtype, np.complexfloating):
    result = math_ops.cast(math_ops.sign(math_ops.real(x)), dtype)
  else:
    result = math_ops.sign(x)

  return result


# Note that np.take_along_axis may not be present in some supported versions of
# numpy.
@tf_export.tf_export('experimental.numpy.take_along_axis', v1=[])
@np_utils.np_doc('take_along_axis')
def take_along_axis(arr, indices, axis):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Takes values from arr along axis using the index array indices (same rank), broadcasting the
  # two against each other except along axis. axis nothing flattens arr first; negative axis
  # counts from the end; negative indices wrap. Implemented by moving axis last, flattening to a
  # batched gather and moving back, then restoring the static shape. Returns the gathered array.
  arr = asarray(arr)
  indices = asarray(indices)

  if axis is None:
    return take_along_axis(arr.ravel(), indices, 0)

  rank = array_ops.rank(arr)
  axis = axis + rank if axis < 0 else axis

  # Broadcast shapes to match, ensure that the axis of interest is not
  # broadcast.
  arr_shape_original = array_ops.shape(arr, out_type=indices.dtype)
  indices_shape_original = array_ops.shape(indices, out_type=indices.dtype)
  arr_shape = array_ops.tensor_scatter_update(arr_shape_original, [[axis]], [1])
  indices_shape = array_ops.tensor_scatter_update(
      indices_shape_original, [[axis]], [1]
  )
  broadcasted_shape = array_ops.broadcast_dynamic_shape(
      arr_shape, indices_shape
  )
  arr_shape = array_ops.tensor_scatter_update(
      broadcasted_shape, [[axis]], [arr_shape_original[axis]]
  )
  indices_shape = array_ops.tensor_scatter_update(
      broadcasted_shape, [[axis]], [indices_shape_original[axis]]
  )
  arr = array_ops.broadcast_to(arr, arr_shape)
  indices = array_ops.broadcast_to(indices, indices_shape)

  # Save indices shape so we can restore it later.
  possible_result_shape = indices.shape

  # Correct indices since gather doesn't correctly handle negative indices.
  indices = array_ops.where_v2(indices < 0, indices + arr_shape[axis], indices)

  swapaxes_ = lambda t: swapaxes(t, axis, -1)

  dont_move_axis_to_end = math_ops.equal(axis, np_utils.subtract(rank, 1))
  arr = np_utils.cond(
      dont_move_axis_to_end, lambda: arr, lambda: swapaxes_(arr)
  )
  indices = np_utils.cond(
      dont_move_axis_to_end, lambda: indices, lambda: swapaxes_(indices)
  )

  arr_shape = array_ops.shape(arr)
  arr = array_ops.reshape(arr, [-1, arr_shape[-1]])

  indices_shape = array_ops.shape(indices)
  indices = array_ops.reshape(indices, [-1, indices_shape[-1]])

  result = array_ops.gather(arr, indices, batch_dims=1)
  result = array_ops.reshape(result, indices_shape)
  result = np_utils.cond(
      dont_move_axis_to_end, lambda: result, lambda: swapaxes_(result)
  )
  result.set_shape(possible_result_shape)

  return result


# pylint: disable=redefined-builtin,undefined-variable
@tf_export.tf_export('experimental.numpy.max', v1=[])
@np_utils.np_doc('max', link=np_utils.AliasOf('amax'))
def max(a, axis=None, keepdims=None):
  # [FUNCTION DOC]
  # Alias of amax: maximum over axis (nothing means all) with optional keepdims. Shadows the
  # builtin max in this module. Returns the maximum array.
  return amax(a, axis=axis, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.min', v1=[])
@np_utils.np_doc('min', link=np_utils.AliasOf('amin'))
def min(a, axis=None, keepdims=None):
  # [FUNCTION DOC]
  # Alias of amin: minimum over axis (nothing means all) with optional keepdims. Shadows the
  # builtin min in this module. Returns the minimum array.
  return amin(a, axis=axis, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.round', v1=[])
@np_utils.np_doc('round', link=np_utils.AliasOf('around'))
def round(a, decimals=0):
  # [FUNCTION DOC]
  # Alias of around: rounds a to the given number of decimals (default 0). Shadows the builtin
  # round in this module. Returns the rounded array.
  return around(a, decimals=decimals)


# pylint: enable=redefined-builtin,undefined-variable


_SLICE_ERROR = (
    'only integers, slices (`:`), ellipsis (`...`), '
    'numpy.newaxis (`None`) and integer or boolean arrays are valid indices'
)


def _as_index(idx, need_scalar=True):
  # [FUNCTION DOC]
  # HELPER: index parsing for _slice_helper.
  # KNOWN ISSUE: the docstring lists only IndexError, but a boolean index whose rank is not one
  # raises not-implemented instead.
  # Parses idx as an index. Python integers and tensor dimensions pass through as scalars.
  # Otherwise idx is converted to an array; boolean arrays (rank one only) become the indices of
  # true elements; integer arrays are cast to int32 or int64. If need_scalar (default true) a
  # non-scalar array is rejected. Raises an index error for non-integer dtypes, non-scalar when
  # a scalar is needed, or unsupported integer types. Returns a pair of the parsed index and a
  # flag that is true when the index is known to have rank zero.
  """Helper function to parse idx as an index.

  Args:
    idx: index
    need_scalar: If idx needs to be a scalar value.

  Returns:
    A pair, (indx, bool). First one is the parsed index and can be a tensor,
    or scalar integer / Dimension. Second one is True if rank is known to be 0.

  Raises:
    IndexError: For incorrect indices.
  """
  if isinstance(idx, (numbers.Integral, tensor_shape.Dimension)):
    return idx, True
  data = asarray(idx)
  if data.dtype == dtypes.bool:
    if data.shape.ndims != 1:
      # TODO(agarwal): handle higher rank boolean masks.
      raise NotImplementedError('Need rank 1 for bool index %s' % idx)
    data = array_ops.where_v2(data)
    data = array_ops.reshape(data, [-1])
  if need_scalar and data.shape.rank not in (None, 0):
    raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))
  np_dtype = data.dtype.as_numpy_dtype
  if not np.issubdtype(np_dtype, np.integer):
    raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))
  if data.dtype not in (dtypes.int64, dtypes.int32):
    # TF slicing can only handle int32/int64. So we need to cast.
    promoted_dtype = np.promote_types(np.int32, np_dtype)
    if promoted_dtype == np.int32:
      data = math_ops.cast(data, dtypes.int32)
    elif promoted_dtype == np.int64:
      data = math_ops.cast(data, dtypes.int64)
    else:
      raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))
  return data, data.shape.rank == 0


class _UpdateMethod(enum.Enum):
  # [CLASS DOC]
  # HELPER: enumeration of how indexed updates combine new values with existing ones, used by
  # _slice_helper and the index-update methods installed on the ndarray class. Members are
  # update (replace), add, min (elementwise minimum) and max (elementwise maximum), with values
  # 0 to 3. No state beyond the members.
  UPDATE = 0
  ADD = 1
  MIN = 2
  MAX = 3


def _slice_helper(tensor, slice_spec, update_method=None, updates=None):
  # [FUNCTION DOC]
  # HELPER: core of NumPy-style indexing, used by the getitem hook and the index-update methods.
  # Reads (update_method nothing) or updates (update_method one of the _UpdateMethod members,
  # with updates of the same dtype as tensor) the elements of tensor selected by slice_spec, a
  # sequence of slices, ellipsis, new-axis markers, integers and integer/boolean index arrays.
  # Basic indices become a strided slice (or a strided slice update); advanced (array) indices
  # are applied afterwards with gather, or with a scatter-based read-modify-write for updates;
  # contiguous advanced indices are optimized with a gather along an axis. Negative advanced
  # indices are wrapped. Raises not-implemented if the rank of advanced indices is unknown for an
  # update, plus errors from index parsing. Returns the selected values or the updated tensor.
  # The input tensor is never mutated; updates produce a new tensor.
  """Helper function for __getitem__ and _with_index_update_helper.

  This function collects the indices in `slice_spec` into two buckets, which we
  can call "idx1" and "idx2" here. idx1 is intended for `strided_slice`, idx2
  `gather`.  They also correspond to "basic indices" and "advanced indices" in
  numpy.  This function supports both reading and writing at the indices. The
  reading path can be summarized as `gather(stride_slice(tensor, idx1),
  idx2)`. The writing path can be summarized as `strided_slice_update(tensor,
  idx1, scatter(strided_slice(tensor, idx1), idx2, updates))`.  (`gather` here
  means `tf.gather` or `tf.gather_nd`; `scatter` here means
  `tf.tensor_scatter_update`.)  The writing path is inefficient because it needs
  to first read out a portion (probably much larger than `updates`) of `tensor`
  using `strided_slice`, update it, and then write the portion back. An
  alternative approach is to only use `scatter`, which amounts to using the
  indexing mechanism of gather/scatter to implement
  strided_slice/strided_slice_update. This is feasible for XLA Gather/Scatter
  because they support spans (e.g. `2:5`) in indices (as begin/end pairs), but
  not TF gather/scatter because they don't support spans (except those that
  cover entire dimensions, i.e. `:`).  If we materialize spans into individual
  indices, the size of the index tensor would explode.  (Note that XLA
  Gather/Scatter have a similar problem for stride > 1 because they don't
  support strides.  Indices such as `1:2:8` will need to be materialized into
  individual indices such as [1, 3, 5, 7].)

  Args:
    tensor: the tensor to be read from or write into.
    slice_spec: the indices.
    update_method: (optional) a member of `_UpdateMethod`, indicating how to
      update the values (replacement, add, etc.). `None` indicates just reading.
    updates: (optional) the new values to write into `tensor`. It must have the
      same dtype as `tensor`.

  Returns:
    The result of reading (if `update_method` is `None`) or the updated `tensor`
    after writing.
  """
  begin, end, strides = [], [], []
  new_axis_mask, shrink_axis_mask = 0, 0
  begin_mask, end_mask = 0, 0
  ellipsis_mask = 0
  advanced_indices = []
  shrink_indices = []
  for index, s in enumerate(slice_spec):
    if isinstance(s, slice):
      if s.start is not None:
        begin.append(_as_index(s.start)[0])
      else:
        begin.append(0)
        begin_mask |= 1 << index
      if s.stop is not None:
        end.append(_as_index(s.stop)[0])
      else:
        end.append(0)
        end_mask |= 1 << index
      if s.step is not None:
        strides.append(_as_index(s.step)[0])
      else:
        strides.append(1)
    elif s is Ellipsis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      ellipsis_mask |= 1 << index
    elif s is array_ops.newaxis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      new_axis_mask |= 1 << index
    else:
      s, is_scalar = _as_index(s, False)
      if is_scalar:
        begin.append(s)
        end.append(s + 1)
        strides.append(1)
        shrink_axis_mask |= 1 << index
        shrink_indices.append(index)
      else:
        begin.append(0)
        end.append(0)
        strides.append(1)
        begin_mask |= 1 << index
        end_mask |= 1 << index
        advanced_indices.append((index, s, ellipsis_mask != 0))

  # stack possibly involves no tensors, so we must use op_scope correct graph.
  with ops.name_scope(
      None,
      'strided_slice',
      [tensor] + begin + end + strides,
      skip_on_eager=False,
  ) as name:
    if begin:
      packed_begin, packed_end, packed_strides = (
          array_ops_stack.stack(begin),
          array_ops_stack.stack(end),
          array_ops_stack.stack(strides),
      )
      if (
          packed_begin.dtype == dtypes.int64
          or packed_end.dtype == dtypes.int64
          or packed_strides.dtype == dtypes.int64
      ):
        if packed_begin.dtype != dtypes.int64:
          packed_begin = math_ops.cast(packed_begin, dtypes.int64)
        if packed_end.dtype != dtypes.int64:
          packed_end = math_ops.cast(packed_end, dtypes.int64)
        if packed_strides.dtype != dtypes.int64:
          packed_strides = math_ops.cast(packed_strides, dtypes.int64)
    else:
      var_empty = constant_op.constant([], dtype=dtypes.int32)
      packed_begin = packed_end = packed_strides = var_empty
    if update_method == _UpdateMethod.UPDATE and not advanced_indices:
      return array_ops.tensor_strided_slice_update(
          tensor,
          packed_begin,
          packed_end,
          packed_strides,
          updates,
          begin_mask=begin_mask,
          end_mask=end_mask,
          shrink_axis_mask=shrink_axis_mask,
          new_axis_mask=new_axis_mask,
          ellipsis_mask=ellipsis_mask,
          name=name,
      )
    else:
      # TODO(b/164251540): Find a better way to support update that does not
      #   involve one read + two writes.
      if updates is not None:
        original_tensor = tensor
      # TODO(agarwal): set_shape on tensor to set rank.
      tensor = array_ops.strided_slice(
          tensor,
          packed_begin,
          packed_end,
          packed_strides,
          begin_mask=begin_mask,
          end_mask=end_mask,
          shrink_axis_mask=shrink_axis_mask,
          new_axis_mask=new_axis_mask,
          ellipsis_mask=ellipsis_mask,
          name=name,
      )
    if not advanced_indices:
      if update_method is None:
        return tensor
      assert update_method != _UpdateMethod.UPDATE
      # TF lacks TensorStridedSliceAdd and alike, so we need to do
      # read+add+update.
      if update_method == _UpdateMethod.ADD:
        update_op = math_ops.add
      elif update_method == _UpdateMethod.MIN:
        update_op = math_ops.minimum
      elif update_method == _UpdateMethod.MAX:
        update_op = math_ops.maximum
      return array_ops.tensor_strided_slice_update(
          original_tensor,
          packed_begin,
          packed_end,
          packed_strides,
          update_op(tensor, updates),
          begin_mask=begin_mask,
          end_mask=end_mask,
          shrink_axis_mask=shrink_axis_mask,
          new_axis_mask=new_axis_mask,
          ellipsis_mask=ellipsis_mask,
          name=name + '_2',
      )
    advanced_indices_map = {}
    for index, data, had_ellipsis in advanced_indices:
      if had_ellipsis:
        num_shrink = len([x for x in shrink_indices if x > index])
        dim = index - len(slice_spec) + num_shrink
      else:
        num_shrink = len([x for x in shrink_indices if x < index])
        dim = index - num_shrink
      advanced_indices_map[dim] = data
    dims = sorted(advanced_indices_map.keys())
    dims_contiguous = True
    if len(dims) > 1:
      if dims[0] < 0 and dims[-1] >= 0:  # not all same sign
        dims_contiguous = False
      else:
        for i in range(len(dims) - 1):
          if dims[i] + 1 != dims[i + 1]:
            dims_contiguous = False
            break
    indices = [advanced_indices_map[x] for x in dims]
    indices = _promote_dtype(*indices)
    indices = np_utils.tf_broadcast(*indices)
    stacked_indices = array_ops_stack.stack(indices, axis=-1)
    # Skip the contiguous-dims optimization for update because there is no
    # tf.*scatter* op that supports the `axis` argument.
    if not dims_contiguous or updates is not None:
      if range(len(dims)) != dims:
        tensor = moveaxis(tensor, dims, range(len(dims)))
      tensor_shape_prefix = array_ops.shape(
          tensor, out_type=stacked_indices.dtype
      )[: len(dims)]
      stacked_indices = array_ops.where_v2(
          stacked_indices < 0,
          stacked_indices + tensor_shape_prefix,
          stacked_indices,
      )
      if updates is None:
        return array_ops.gather_nd(tensor, stacked_indices)
      else:
        # We only need to move-axis `updates` in the contiguous case becausce
        # only in this case the result dimensions of advanced indexing are in
        # the middle of `updates`. In the non-contiguous case, those dimensions
        # are always at the front.
        if dims_contiguous:
          # TODO(wangpeng): Support unknown rank (e.g. by partially flattening
          #   `updates`)
          if stacked_indices.shape.rank is None:
            raise NotImplementedError(
                'Rank of the advanced indices must currently be known'
            )
          batch_size = stacked_indices.shape.rank - 1
          batch_start = dims[0]
          if batch_start < 0:
            batch_start += len(dims) - batch_size

          def range_(start, length):
            # [FUNCTION DOC]
            # HELPER: returns the integer range of the given length beginning at start.
            return range(start, start + length)

          updates = moveaxis(
              updates, range_(batch_start, batch_size), range(batch_size)
          )
        if update_method == _UpdateMethod.UPDATE:
          update_op = array_ops.tensor_scatter_update
        elif update_method == _UpdateMethod.ADD:
          update_op = array_ops.tensor_scatter_add
        elif update_method == _UpdateMethod.MIN:
          update_op = array_ops.tensor_scatter_min
        elif update_method == _UpdateMethod.MAX:
          update_op = array_ops.tensor_scatter_max
        tensor = update_op(tensor, stacked_indices, updates)
        if range(len(dims)) != dims:
          tensor = moveaxis(tensor, range(len(dims)), dims)
        return array_ops.tensor_strided_slice_update(
            original_tensor,
            packed_begin,
            packed_end,
            packed_strides,
            tensor,
            begin_mask=begin_mask,
            end_mask=end_mask,
            shrink_axis_mask=shrink_axis_mask,
            new_axis_mask=new_axis_mask,
            ellipsis_mask=ellipsis_mask,
            name=name + '_2',
        )
    # Note that gather_nd does not support gathering from inside the array.
    # To avoid shuffling data back and forth, we transform the indices and
    # do a gather instead.
    rank = np_utils._maybe_static(array_ops.rank(tensor))  # pylint: disable=protected-access
    dims = [(x + rank if x < 0 else x) for x in dims]
    shape_tensor = array_ops.shape(tensor)
    dim_sizes = array_ops.gather(shape_tensor, dims)
    if len(dims) == 1:
      stacked_indices = indices[0]
    stacked_indices = math_ops.cast(stacked_indices, dtypes.int32)
    stacked_indices = array_ops.where_v2(
        stacked_indices < 0, stacked_indices + dim_sizes, stacked_indices
    )
    axis = dims[0]
    if len(dims) > 1:
      index_scaling = math_ops.cumprod(dim_sizes, reverse=True, exclusive=True)

      def _tensordot(a, b):
        # [FUNCTION DOC]
        # HELPER: stand-in for a dot product over the last axis, used to flatten multi-dimension
        # advanced indices into one linear index. Broadcasts b to the shape of a, multiplies and
        # sums over the last axis; returns the summed tensor.
        # TODO(b/168657656): This function should be replaced by
        # tensordot(axis=1) once MatMul has int32 XLA kernel.
        b = array_ops.broadcast_to(b, array_ops.shape(a))
        return math_ops.reduce_sum(a * b, axis=-1)

      stacked_indices = _tensordot(stacked_indices, index_scaling)
      flat_shape = array_ops.concat(
          [shape_tensor[:axis], [-1], shape_tensor[axis + len(dims) :]], axis=0
      )
      tensor = array_ops.reshape(tensor, flat_shape)

    return array_ops.gather(tensor, stacked_indices, axis=axis)


def _as_spec_tuple(slice_spec):
  # [FUNCTION DOC]
  # HELPER: normalizes an index expression for _slice_helper.
  # A list or tuple containing nothing, ellipsis, slices, nested sequences or non-scalar arrays
  # is treated as a multi-dimensional index and returned as a tuple of its elements; anything
  # else (including a list of plain scalars, treated as one advanced index) is wrapped in a
  # one-element tuple.
  """Convert slice_spec to tuple."""
  if isinstance(slice_spec, (list, tuple)) and not isinstance(
      slice_spec, np.ndarray
  ):
    is_index = True
    for s in slice_spec:
      if s is None or s is Ellipsis or isinstance(s, (list, tuple, slice)):
        is_index = False
        break
      elif isinstance(s, (np_arrays.ndarray, np.ndarray)) and s.ndim != 0:
        is_index = False
        break
    if not is_index:
      return tuple(slice_spec)
  return (slice_spec,)


def _getitem(self, slice_spec):
  # [FUNCTION DOC]
  # Implementation of NumPy-style indexing on the ndarray class (installed there at import time).
  # A boolean scalar, boolean tensor or boolean array index selects elements with a boolean mask;
  # any other index is normalized to a tuple and handled by _slice_helper. self is the array and
  # slice_spec the index expression. Returns the selected values.
  """Implementation of ndarray.__getitem__."""
  if (
      isinstance(slice_spec, bool)
      or (
          isinstance(slice_spec, core_tf_types.Tensor)
          and slice_spec.dtype == dtypes.bool
      )
      or (
          isinstance(slice_spec, (np.ndarray, np_arrays.ndarray))
          and slice_spec.dtype == np.bool_
      )
  ):
    return array_ops.boolean_mask(tensor=self, mask=slice_spec)

  if not isinstance(slice_spec, tuple):
    slice_spec = _as_spec_tuple(slice_spec)

  result_t = _slice_helper(self, slice_spec)
  return result_t


def _with_index_update_helper(update_method, a, slice_spec, updates):
  # [FUNCTION DOC]
  # HELPER: shared implementation of the functional index-update methods installed on ndarray
  # (update, add, min, max); each is this function with the first argument pre-bound.
  # update_method is an _UpdateMethod member, a the array, slice_spec the index expression
  # (boolean masks are converted to nonzero indices) and updates the new values. a and updates
  # are promoted to a common dtype, the update is performed, and the result is cast back to the
  # original dtype of a. Returns a new array; a is not mutated.
  """Implementation of ndarray._with_index_*."""
  if (
      isinstance(slice_spec, bool)
      or (
          isinstance(slice_spec, core_tf_types.Tensor)
          and slice_spec.dtype == dtypes.bool
      )
      or (
          isinstance(slice_spec, (np.ndarray, np_arrays.ndarray))
          and slice_spec.dtype == np.bool_
      )
  ):
    slice_spec = nonzero(slice_spec)

  if not isinstance(slice_spec, tuple):
    slice_spec = _as_spec_tuple(slice_spec)

  a_dtype = a.dtype
  a, updates = _promote_dtype_binary(a, updates)
  result_t = _slice_helper(a, slice_spec, update_method, updates)
  return result_t.astype(a_dtype)


setattr(np_arrays.ndarray, '_numpy_style_getitem', _getitem)
setattr(
    np_arrays.ndarray,
    '_with_index_update',
    functools.partial(_with_index_update_helper, _UpdateMethod.UPDATE),
)
setattr(
    np_arrays.ndarray,
    '_with_index_add',
    functools.partial(_with_index_update_helper, _UpdateMethod.ADD),
)
setattr(
    np_arrays.ndarray,
    '_with_index_min',
    functools.partial(_with_index_update_helper, _UpdateMethod.MIN),
)
setattr(
    np_arrays.ndarray,
    '_with_index_max',
    functools.partial(_with_index_update_helper, _UpdateMethod.MAX),
)
