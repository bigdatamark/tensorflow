# [MODULE DOC]
# Mathematical functions of the experimental NumPy-compatible API implemented on TensorFlow ops:
# binary arithmetic (add, subtract, multiply, divide, mod, power, ...), linear algebra style
# products (dot, matmul, tensordot, inner, cross, vdot, kron, outer), unary elementwise math,
# comparison and logical functions, reductions and sorting helpers, rounding, number-theory
# helpers (gcd, lcm), sequence generators (linspace, logspace, geomspace), statistics helpers and
# the hook that installs numpy-style methods and operator overloads on tensors. Most functions are
# decorated to export them in the experimental numpy namespace and to borrow NumPy documentation.
# Inputs are converted to framework arrays and (for binary ops) promoted to a common dtype using
# the numpy-style promotion rules in np_array_ops. Import-time side effects: defines the exported
# constants pi, e and inf, registers all exports, and attaches the matmul function to the ndarray
# class as a private attribute used by tensor matrix-multiplication.
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
"""Mathematical operations."""
# pylint: disable=g-direct-tensorflow-import

import numbers
import sys

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export


pi = np.pi
tf_export.tf_export('experimental.numpy.pi', v1=[]).export_constant(
    __name__, 'pi'
)
e = np.e
tf_export.tf_export('experimental.numpy.e', v1=[]).export_constant(
    __name__, 'e'
)
inf = np.inf
tf_export.tf_export('experimental.numpy.inf', v1=[]).export_constant(
    __name__, 'inf'
)


@tf_export.tf_export('experimental.numpy.dot', v1=[])
@np_utils.np_doc_only('dot')
def dot(a, b):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Dot product of two array-likes with NumPy semantics. Both inputs are promoted to a common
  # dtype. If either is a scalar the result is plain multiplication; if b is one-dimensional it
  # contracts the last axis of a with b; otherwise it contracts the last axis of a with the
  # second-to-last axis of b. Returns an array. Shape mismatches surface as errors from the
  # underlying tensordot.
  def f(a, b):  # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # HELPER: tensor-level implementation of dot, with rank-based branching via conditional
    # helpers so it works for dynamic ranks. Inputs are already-promoted arrays; returns a tensor.
    return np_utils.cond(
        np_utils.logical_or(
            math_ops.equal(array_ops.rank(a), 0),
            math_ops.equal(array_ops.rank(b), 0),
        ),
        lambda: a * b,
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            math_ops.equal(array_ops.rank(b), 1),
            lambda: math_ops.tensordot(a, b, axes=[[-1], [-1]]),
            lambda: math_ops.tensordot(a, b, axes=[[-1], [-2]]),
        ),
    )

  return _bin_op(f, a, b)


# TODO(wangpeng): Make element-wise ops `ufunc`s
def _bin_op(tf_fun, a, b, promote=True):
  # [FUNCTION DOC]
  # HELPER: common wrapper used by almost all binary functions in this module.
  # Inputs: tf_fun, a function taking two tensors; a and b, array-likes; promote (default true)
  # chooses whether both operands are promoted to a common numpy-style dtype, otherwise they are
  # just converted to arrays without promotion. Returns the result of tf_fun on the prepared
  # operands.
  if promote:
    a, b = np_array_ops._promote_dtype_binary(a, b)  # pylint: disable=protected-access
  else:
    a = np_array_ops.array(a)
    b = np_array_ops.array(b)
  return tf_fun(a, b)


@tf_export.tf_export('experimental.numpy.add', v1=[])
@np_utils.np_doc('add')
def add(x1, x2):
  # [FUNCTION DOC]
  # Elementwise addition with numpy-style dtype promotion and broadcasting. Inputs x1, x2 are
  # array-likes. Boolean inputs use logical or, as NumPy does. Returns an array.
  def add_or_or(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level addition for add. For boolean operands returns logical or (asserting
    # both are boolean); otherwise ordinary addition.
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_or(x1, x2)
    return math_ops.add(x1, x2)

  return _bin_op(add_or_or, x1, x2)


@tf_export.tf_export('experimental.numpy.subtract', v1=[])
@np_utils.np_doc('subtract')
def subtract(x1, x2):
  # [FUNCTION DOC]
  # Elementwise subtraction x1 minus x2 with dtype promotion and broadcasting. Returns an array.
  return _bin_op(math_ops.subtract, x1, x2)


@tf_export.tf_export('experimental.numpy.multiply', v1=[])
@np_utils.np_doc('multiply')
def multiply(x1, x2):
  # [FUNCTION DOC]
  # Elementwise multiplication with dtype promotion and broadcasting; boolean operands use logical
  # and. Returns an array.
  def mul_or_and(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level multiplication for multiply. Logical and for booleans (asserting both
    # are boolean), ordinary multiplication otherwise.
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_and(x1, x2)
    return math_ops.multiply(x1, x2)

  return _bin_op(mul_or_and, x1, x2)


@tf_export.tf_export('experimental.numpy.true_divide', v1=[])
@np_utils.np_doc('true_divide')
def true_divide(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Elementwise true (floating point) division x1 divided by x2 with promotion and broadcasting.
  # Boolean operands are first cast to the configured float type. When float64 is disallowed in
  # the numpy dtype configuration, integer-by-integer division is done in single precision.
  # Returns an array of a floating type.
  def _avoid_float64(x1, x2):
    # [FUNCTION DOC]
    # HELPER: for true_divide. If both operands have the same 32- or 64-bit integer dtype, casts
    # both to float32 so the division does not yield float64; otherwise returns them unchanged.
    # Returns the pair of operands.
    if x1.dtype == x2.dtype and x1.dtype in (dtypes.int32, dtypes.int64):
      x1 = math_ops.cast(x1, dtype=dtypes.float32)
      x2 = math_ops.cast(x2, dtype=dtypes.float32)
    return x1, x2

  def f(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of true_divide. Casts booleans to the configured float type,
    # avoids float64 when it is disabled, then applies true division. Returns a tensor.
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      float_ = np_utils.result_type(float)
      x1 = math_ops.cast(x1, float_)
      x2 = math_ops.cast(x2, float_)
    if not np_dtypes.is_allow_float64():
      # math_ops.truediv in Python3 produces float64 when both inputs are int32
      # or int64. We want to avoid that when is_allow_float64() is False.
      x1, x2 = _avoid_float64(x1, x2)
    return math_ops.truediv(x1, x2)

  return _bin_op(f, x1, x2)


@tf_export.tf_export('experimental.numpy.divide', v1=[])
@np_utils.np_doc('divide')
def divide(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Alias of true_divide: elementwise floating point division of x1 by x2.
  return true_divide(x1, x2)


@tf_export.tf_export('experimental.numpy.floor_divide', v1=[])
@np_utils.np_doc('floor_divide')
def floor_divide(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Elementwise floor division (quotient rounded toward negative infinity) with promotion and
  # broadcasting. Boolean operands are first cast to 8-bit integers. Returns an array.
  def f(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of floor_divide; casts booleans to 8-bit integers and applies
    # floor division. Returns a tensor.
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    return math_ops.floordiv(x1, x2)

  return _bin_op(f, x1, x2)


@tf_export.tf_export('experimental.numpy.mod', v1=[])
@np_utils.np_doc('mod')
def mod(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Elementwise remainder of x1 divided by x2 (sign follows the divisor) with promotion and
  # broadcasting. Booleans are cast to 8-bit integers first. Returns an array.
  def f(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of mod; casts booleans to 8-bit integers and applies modulo.
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    return math_ops.mod(x1, x2)

  return _bin_op(f, x1, x2)


@tf_export.tf_export('experimental.numpy.remainder', v1=[])
@np_utils.np_doc('remainder')
def remainder(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Alias of mod: elementwise remainder of x1 divided by x2.
  return mod(x1, x2)


@tf_export.tf_export('experimental.numpy.divmod', v1=[])
@np_utils.np_doc('divmod')
def divmod(x1, x2):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Elementwise floor quotient and remainder. Returns a pair: floor_divide result and mod result.
  # Note it computes both separately (operands are converted and promoted twice).
  return floor_divide(x1, x2), mod(x1, x2)


@tf_export.tf_export('experimental.numpy.maximum', v1=[])
@np_utils.np_doc('maximum')
def maximum(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Elementwise maximum with promotion and broadcasting; boolean operands use logical or.
  # Fast path: when x2 is the real number zero (not a boolean) and x1 is a non-boolean framework
  # array, it returns the rectified-linear activation of x1 directly. Returns an array.
  # Fast path for when maximum is used as relu.
  if (
      isinstance(x2, numbers.Real)
      and not isinstance(x2, bool)
      and x2 == 0
      and isinstance(x1, np_arrays.ndarray)
      and x1.dtype != dtypes.bool
  ):
    return nn_ops.relu(np_array_ops.asarray(x1))

  def max_or_or(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of maximum; logical or for booleans, elementwise maximum otherwise.
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_or(x1, x2)
    return math_ops.maximum(x1, x2)

  return _bin_op(max_or_or, x1, x2)


@tf_export.tf_export('experimental.numpy.minimum', v1=[])
@np_utils.np_doc('minimum')
def minimum(x1, x2):
  # [FUNCTION DOC]
  # Elementwise minimum with promotion and broadcasting; boolean operands use logical and.
  def min_or_and(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of minimum; logical and for booleans, elementwise minimum
    # otherwise.
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_and(x1, x2)
    return math_ops.minimum(x1, x2)

  return _bin_op(min_or_and, x1, x2)


@tf_export.tf_export('experimental.numpy.clip', v1=[])
@np_utils.np_doc('clip')
def clip(a, a_min, a_max):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: the error message says "Not more than one of a_min and a_max may be None", but
  # the check actually raises only when BOTH are None (one may be None), so the wording is wrong.
  # Limits the values of a to the interval from a_min to a_max. Inputs: a, array-like; a_min and
  # a_max, array-likes or nothing (at most one may be nothing). If one bound is missing it uses
  # minimum or maximum with the other bound. Otherwise the three are promoted to a common dtype,
  # broadcast and clipped. Raises ValueError if both bounds are nothing. Returns an array.
  if a_min is None and a_max is None:
    raise ValueError('Not more than one of `a_min` and `a_max` may be `None`.')
  if a_min is None:
    return minimum(a, a_max)
  elif a_max is None:
    return maximum(a, a_min)
  else:
    a, a_min, a_max = np_array_ops._promote_dtype(a, a_min, a_max)  # pylint: disable=protected-access
    return clip_ops.clip_by_value(*np_utils.tf_broadcast(a, a_min, a_max))


@tf_export.tf_export('experimental.numpy.matmul', v1=[])
@np_utils.np_doc('matmul')
def matmul(x1, x2):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Matrix product with NumPy semantics, after dtype promotion. Rank-two by rank-two uses the
  # direct matrix multiply op; a one-dimensional second operand or first operand is handled by
  # tensor contraction; other cases (including batches) use the general matmul op. Invalid-argument
  # errors from the framework are re-raised as ValueError with the original traceback (e.g. for
  # scalar operands). Returns an array. It is also attached to the ndarray class as a private
  # attribute (see below).
  def f(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of matmul, choosing the contraction strategy by rank and
    # converting framework invalid-argument errors into ValueError.
    try:
      if x1._rank() == 2 and x2._rank() == 2:  # pylint: disable=protected-access
        # Fast path for known ranks.
        return gen_math_ops.mat_mul(x1, x2)
      return np_utils.cond(
          math_ops.equal(np_utils.tf_rank(x2), 1),
          lambda: math_ops.tensordot(x1, x2, axes=1),
          lambda: np_utils.cond(  # pylint: disable=g-long-lambda
              math_ops.equal(np_utils.tf_rank(x1), 1),
              lambda: math_ops.tensordot(  # pylint: disable=g-long-lambda
                  x1, x2, axes=[[0], [-2]]
              ),
              lambda: math_ops.matmul(x1, x2),
          ),
      )
    except errors.InvalidArgumentError as err:
      raise ValueError(str(err)).with_traceback(sys.exc_info()[2])

  return _bin_op(f, x1, x2)


# Exported so it can be called from Tensor.__matmul__. NumPy's matmul handles
# batched matmul as well, so simply including promotion in TF's current
# __matmul__ implementation was not sufficient.
setattr(np_arrays.ndarray, '_matmul', matmul)


@tf_export.tf_export('experimental.numpy.tensordot', v1=[])
@np_utils.np_doc('tensordot')
def tensordot(a, b, axes=2):
  # [FUNCTION DOC]
  # Tensor contraction of a and b over the given axes (default: last two axes of a with first two
  # of b, as an integer count). Inputs a and b are array-likes, promoted to a common dtype.
  # Returns an array.
  return _bin_op(lambda a, b: math_ops.tensordot(a, b, axes=axes), a, b)


@tf_export.tf_export('experimental.numpy.inner', v1=[])
@np_utils.np_doc_only('inner')
def inner(a, b):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Inner product over the last axes of a and b (plain multiplication when either is a scalar).
  # Inputs are array-likes promoted to a common dtype. Returns an array.
  def f(a, b):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of inner; scalar multiplication or contraction of last axes.
    return np_utils.cond(
        np_utils.logical_or(
            math_ops.equal(array_ops.rank(a), 0),
            math_ops.equal(array_ops.rank(b), 0),
        ),
        lambda: a * b,
        lambda: math_ops.tensordot(a, b, axes=[[-1], [-1]]),
    )

  return _bin_op(f, a, b)


@tf_export.tf_export('experimental.numpy.cross', v1=[])
@np_utils.np_doc('cross')
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Cross product of vectors of length two or three. Inputs: a, b array-likes; axisa and axisb
  # (default last axis) say which axis of each holds the vectors; axisc (default last) is where
  # the result vector axis is placed; axis, if given, overrides all three. Two-element vectors are
  # zero-padded to three; if both inputs are two-element the result is the single scalar z
  # component. Operands are promoted and broadcast. Returns an array.
  def f(a, b):  # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # HELPER: tensor-level body of cross. Normalizes negative axes, moves the vector axes to the
    # end, pads length-2 vectors, computes the cross product on broadcast operands and moves the
    # result axis to axisc. Returns a tensor.
    # We can't assign to captured variable `axisa`, so make a new variable
    if axis is None:
      axis_a = axisa
      axis_b = axisb
      axis_c = axisc
    else:
      axis_a = axis
      axis_b = axis
      axis_c = axis
    if axis_a < 0:
      axis_a = np_utils.add(axis_a, array_ops.rank(a))
    if axis_b < 0:
      axis_b = np_utils.add(axis_b, array_ops.rank(b))

    def maybe_move_axis_to_last(a, axis):
      # [FUNCTION DOC]
      # HELPER: for cross. Returns a unchanged if axis is already the last axis, otherwise a with
      # that axis moved to the end (conditionally evaluated so it works with dynamic ranks).
      def move_axis_to_last(a, axis):
        # [FUNCTION DOC]
        # HELPER: for cross. Transposes a so that the given axis becomes the last axis, keeping
        # the order of the others. Returns a tensor.
        return array_ops.transpose(
            a,
            array_ops.concat(
                [
                    math_ops.range(axis),
                    math_ops.range(axis + 1, array_ops.rank(a)),
                    [axis],
                ],
                axis=0,
            ),
        )

      return np_utils.cond(
          axis == np_utils.subtract(array_ops.rank(a), 1),
          lambda: a,
          lambda: move_axis_to_last(a, axis),
      )

    a = maybe_move_axis_to_last(a, axis_a)
    b = maybe_move_axis_to_last(b, axis_b)
    a_dim = np_utils.getitem(array_ops.shape(a), -1)
    b_dim = np_utils.getitem(array_ops.shape(b), -1)

    def maybe_pad_0(a, size_of_last_dim):
      # [FUNCTION DOC]
      # HELPER: for cross. If the last dimension size equals two, pads a with one zero at the end
      # of the last axis; otherwise returns a unchanged.
      def pad_0(a):
        # [FUNCTION DOC]
        # HELPER: for cross. Appends a single zero along the last axis of a. Returns a tensor.
        return array_ops.pad(
            a,
            array_ops.concat(
                [
                    array_ops.zeros([array_ops.rank(a) - 1, 2], dtypes.int32),
                    constant_op.constant([[0, 1]], dtypes.int32),
                ],
                axis=0,
            ),
        )

      return np_utils.cond(
          math_ops.equal(size_of_last_dim, 2), lambda: pad_0(a), lambda: a
      )

    a = maybe_pad_0(a, a_dim)
    b = maybe_pad_0(b, b_dim)
    c = math_ops.cross(*np_utils.tf_broadcast(a, b))
    if axis_c < 0:
      axis_c = np_utils.add(axis_c, array_ops.rank(c))

    def move_last_to_axis(a, axis):
      # [FUNCTION DOC]
      # HELPER: for cross. Transposes a so that its last axis moves to position axis. Returns a
      # tensor.
      r = array_ops.rank(a)
      return array_ops.transpose(
          a,
          array_ops.concat(
              [math_ops.range(axis), [r - 1], math_ops.range(axis, r - 1)],
              axis=0,
          ),
      )

    c = np_utils.cond(
        (a_dim == 2) & (b_dim == 2),
        lambda: c[..., 2],
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            axis_c == np_utils.subtract(array_ops.rank(c), 1),
            lambda: c,
            lambda: move_last_to_axis(c, axis_c),
        ),
    )
    return c

  return _bin_op(f, a, b)


@tf_export.tf_export('experimental.numpy.vdot', v1=[])
@np_utils.np_doc_only('vdot')
def vdot(a, b):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Dot product of the flattened inputs. Inputs a and b are array-likes promoted to a common
  # dtype and reshaped to one dimension; if complex, the first is conjugated first. Returns an
  # array (scalar).
  a, b = np_array_ops._promote_dtype(a, b)  # pylint: disable=protected-access
  a = np_array_ops.reshape(a, [-1])
  b = np_array_ops.reshape(b, [-1])
  if a.dtype == np_dtypes.complex128 or a.dtype == np_dtypes.complex64:
    a = conj(a)
  return dot(a, b)


@tf_export.tf_export('experimental.numpy.power', v1=[])
@np_utils.np_doc('power')
def power(x1, x2):
  # [FUNCTION DOC]
  # Elementwise x1 raised to the power x2 with promotion and broadcasting. Returns an array.
  return _bin_op(math_ops.pow, x1, x2)


@tf_export.tf_export('experimental.numpy.float_power', v1=[])
@np_utils.np_doc('float_power')
def float_power(x1, x2):
  # [FUNCTION DOC]
  # KNOWN ISSUE: NumPy's float_power always computes in 64-bit floating point, but this simply
  # calls power, so integer inputs are not converted to floating point (dtype follows ordinary
  # promotion).
  # Elementwise power; delegates to power.
  return power(x1, x2)


@tf_export.tf_export('experimental.numpy.arctan2', v1=[])
@np_utils.np_doc('arctan2')
def arctan2(x1, x2):
  # [FUNCTION DOC]
  # Elementwise angle of the point (x2, x1) in radians, using the quadrant-aware arctangent of
  # x1 over x2, with promotion and broadcasting. Returns an array.
  return _bin_op(math_ops.atan2, x1, x2)


@tf_export.tf_export('experimental.numpy.nextafter', v1=[])
@np_utils.np_doc('nextafter')
def nextafter(x1, x2):
  # [FUNCTION DOC]
  # Elementwise next representable floating point value after x1 in the direction of x2, with
  # promotion and broadcasting. Returns an array.
  return _bin_op(math_ops.nextafter, x1, x2)


@tf_export.tf_export('experimental.numpy.heaviside', v1=[])
@np_utils.np_doc('heaviside')
def heaviside(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Step function: zero where x1 is negative, one where positive, and x2 where x1 equals zero.
  # Inputs array-likes (promoted, broadcast). If the result dtype is not floating it is converted
  # to the configured float type. Returns an array.
  def f(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of heaviside using nested conditional selection with constants of
    # the dtype of x2. Returns a tensor.
    return array_ops.where_v2(
        x1 < 0,
        constant_op.constant(0, dtype=x2.dtype),
        array_ops.where_v2(x1 > 0, constant_op.constant(1, dtype=x2.dtype), x2),
    )

  y = _bin_op(f, x1, x2)
  if not np.issubdtype(y.dtype.as_numpy_dtype, np.inexact):
    y = y.astype(np_utils.result_type(float))
  return y


@tf_export.tf_export('experimental.numpy.hypot', v1=[])
@np_utils.np_doc('hypot')
def hypot(x1, x2):
  # [FUNCTION DOC]
  # Elementwise Euclidean norm of the pair: square root of the sum of squares of x1 and x2.
  # Note: computed directly, so very large or small inputs may overflow or underflow.
  return sqrt(square(x1) + square(x2))


@tf_export.tf_export('experimental.numpy.kron', v1=[])
@np_utils.np_doc('kron')
def kron(a, b):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Kronecker product of a and b. Inputs are array-likes promoted to a common dtype; the one with
  # lower rank gets leading dimensions of size one. Each is reshaped by interleaving ones so that
  # multiplication broadcasts into all block combinations, then the product is reshaped to the
  # elementwise product of the two shapes. Returns an array.
  # pylint: disable=protected-access,g-complex-comprehension
  a, b = np_array_ops._promote_dtype(a, b)
  t_a = np_utils.cond(
      a.shape.rank < b.shape.rank,
      lambda: np_array_ops.reshape(  # pylint: disable=g-long-lambda
          a, np_array_ops._pad_left_to(b.shape.rank, a.shape)
      ),
      lambda: a,
  )
  t_b = np_utils.cond(
      b.shape.rank < a.shape.rank,
      lambda: np_array_ops.reshape(  # pylint: disable=g-long-lambda
          b, np_array_ops._pad_left_to(a.shape.rank, b.shape)
      ),
      lambda: b,
  )

  def _make_shape(shape, prepend):
    # [FUNCTION DOC]
    # HELPER: for kron. Given a shape vector, builds an interleaved shape where each dimension is
    # paired with a one: ones come first in each pair if prepend is true, otherwise last.
    # Returns a flat shape vector of twice the length.
    ones = array_ops.ones_like(shape)
    if prepend:
      shapes = [ones, shape]
    else:
      shapes = [shape, ones]
    return array_ops.reshape(array_ops_stack.stack(shapes, axis=1), [-1])

  a_shape = array_ops.shape(t_a)
  b_shape = array_ops.shape(t_b)
  a_reshaped = np_array_ops.reshape(t_a, _make_shape(a_shape, False))
  b_reshaped = np_array_ops.reshape(t_b, _make_shape(b_shape, True))
  out_shape = a_shape * b_shape
  return np_array_ops.reshape(a_reshaped * b_reshaped, out_shape)


@tf_export.tf_export('experimental.numpy.outer', v1=[])
@np_utils.np_doc('outer')
def outer(a, b):
  # [FUNCTION DOC]
  # Outer product of two arrays, each flattened first. Inputs promoted to a common dtype.
  # Returns a two-dimensional array whose entry (i, j) is a[i] times b[j].
  def f(a, b):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of outer: column of flattened a times row of flattened b.
    return array_ops.reshape(a, [-1, 1]) * array_ops.reshape(b, [-1])

  return _bin_op(f, a, b)


# This can also be implemented via tf.reduce_logsumexp
@tf_export.tf_export('experimental.numpy.logaddexp', v1=[])
@np_utils.np_doc('logaddexp')
def logaddexp(x1, x2):
  # [FUNCTION DOC]
  # Numerically stable logarithm of the sum of exponentials of x1 and x2: maximum plus log1p of
  # the exponential of minus the absolute difference. Where the difference is not a number (equal
  # infinities) the sum x1 plus x2 is used. Returns an array.
  amax = maximum(x1, x2)
  delta = x1 - x2
  return np_array_ops.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp(-abs(delta))),
  )


@tf_export.tf_export('experimental.numpy.logaddexp2', v1=[])
@np_utils.np_doc('logaddexp2')
def logaddexp2(x1, x2):
  # [FUNCTION DOC]
  # Base-two version of logaddexp: logarithm base two of the sum of two to the power x1 and two
  # to the power x2, computed stably as maximum plus log1p of two to the minus absolute difference
  # divided by the natural log of two. Not-a-number difference falls back to x1 plus x2.
  amax = maximum(x1, x2)
  delta = x1 - x2
  return np_array_ops.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp2(-abs(delta))) / np.log(2),
  )


@tf_export.tf_export('experimental.numpy.polyval', v1=[])
@np_utils.np_doc('polyval')
def polyval(p, x):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Evaluates a polynomial with coefficients p (highest power first) at points x, using
  # promoted dtypes. A zero-dimensional p is treated as a one-element list. A constant polynomial
  # is broadcast to the shape of x as NumPy does. Returns an array.
  def f(p, x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of polyval; unstacks coefficients into a list, evaluates with the
    # framework polynomial routine, and broadcasts a zero-order result to the shape of x.
    if p.shape.rank == 0:
      p = array_ops.reshape(p, [1])
    p = array_ops_stack.unstack(p)
    # TODO(wangpeng): Make tf version take a tensor for p instead of a list.
    y = math_ops.polyval(p, x)
    # If the polynomial is 0-order, numpy requires the result to be broadcast to
    # `x`'s shape.
    if len(p) == 1:
      y = array_ops.broadcast_to(y, x.shape)
    return y

  return _bin_op(f, p, x)


@tf_export.tf_export('experimental.numpy.isclose', v1=[])
@np_utils.np_doc('isclose')
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Elementwise test of whether a and b are close: absolute difference at most atol plus rtol
  # times absolute b (so it is not symmetric). Inputs: a, b array-likes; rtol relative tolerance
  # (default 1e-5); atol absolute tolerance (default 1e-8); equal_nan (default false) treats
  # not-a-number values in the same position as equal. For non-floating dtypes it is plain
  # equality. Returns a boolean array.
  def f(a, b):  # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # HELPER: tensor-level body of isclose using the tolerances from the enclosing call.
    dtype = a.dtype
    if np.issubdtype(dtype.as_numpy_dtype, np.inexact):
      rtol_ = ops.convert_to_tensor(rtol, dtype.real_dtype)
      atol_ = ops.convert_to_tensor(atol, dtype.real_dtype)
      result = math_ops.abs(a - b) <= atol_ + rtol_ * math_ops.abs(b)
      if equal_nan:
        result = result | (math_ops.is_nan(a) & math_ops.is_nan(b))
      return result
    else:
      return a == b

  return _bin_op(f, a, b)


@tf_export.tf_export('experimental.numpy.allclose', v1=[])
@np_utils.np_doc('allclose')
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  # [FUNCTION DOC]
  # Returns a boolean scalar array that is true if every element of isclose(a, b) with the given
  # tolerances (rtol default 1e-5, atol default 1e-8, equal_nan default false) is true.
  return np_array_ops.all(
      isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
  )


def _tf_gcd(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # HELPER: tensor-level greatest common divisor used by gcd and lcm.
  # Inputs: x1, x2 integer tensors (raises ValueError if either is not an integer type).
  # Broadcasts them to a common shape and runs a loop of Euclid's algorithm on absolute values
  # until all second elements are zero. Returns a tensor of non-negative gcd values.
  def _gcd_cond_fn(_, x2):
    # [FUNCTION DOC]
    # HELPER: loop condition for the gcd loop: true while any element of x2 is non-zero.
    return math_ops.reduce_any(x2 != 0)

  def _gcd_body_fn(x1, x2):
    # [FUNCTION DOC]
    # HELPER: one Euclid step of the gcd loop, elementwise: replaces (x1, x2) by (x2, x1 mod x2)
    # where x2 is non-zero (zeros in x2 are replaced by ones in the divisor to avoid errors and
    # the value is unused), then orders the pair so the larger element is first. Returns the new
    # pair.
    # math_ops.mod will raise an error when any element of x2 is 0. To avoid
    # that, we change those zeros to ones. Their values don't matter because
    # they won't be used.
    x2_safe = array_ops.where_v2(x2 != 0, x2, constant_op.constant(1, x2.dtype))
    x1, x2 = (
        array_ops.where_v2(x2 != 0, x2, x1),
        array_ops.where_v2(
            x2 != 0,
            math_ops.mod(x1, x2_safe),
            constant_op.constant(0, x2.dtype),
        ),
    )
    return (
        array_ops.where_v2(x1 < x2, x2, x1),
        array_ops.where_v2(x1 < x2, x1, x2),
    )

  if not np.issubdtype(
      x1.dtype.as_numpy_dtype, np.integer
  ) or not np.issubdtype(x2.dtype.as_numpy_dtype, np.integer):
    raise ValueError('Arguments to gcd must be integers.')
  shape = array_ops.broadcast_dynamic_shape(
      array_ops.shape(x1), array_ops.shape(x2)
  )
  x1 = array_ops.broadcast_to(x1, shape)
  x2 = array_ops.broadcast_to(x2, shape)
  value, _ = while_loop.while_loop(
      _gcd_cond_fn, _gcd_body_fn, (math_ops.abs(x1), math_ops.abs(x2))
  )
  return value


# Note that np.gcd may not be present in some supported versions of numpy.
@tf_export.tf_export('experimental.numpy.gcd', v1=[])
@np_utils.np_doc('gcd')
def gcd(x1, x2):
  # [FUNCTION DOC]
  # Elementwise greatest common divisor of integer arrays x1 and x2 (promoted, broadcast).
  # Raises ValueError for non-integer inputs. Returns an array.
  return _bin_op(_tf_gcd, x1, x2)


# Note that np.lcm may not be present in some supported versions of numpy.
@tf_export.tf_export('experimental.numpy.lcm', v1=[])
@np_utils.np_doc('lcm')
def lcm(x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Elementwise least common multiple of integer arrays x1 and x2 (promoted, broadcast); zero
  # where the gcd is zero (both inputs zero). Raises ValueError for non-integer inputs (from the
  # gcd helper). Returns an array of non-negative values.
  def f(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of lcm: absolute x1 times (absolute x2 divided by the gcd), with
    # a safe divisor substituted where the gcd is zero and zero returned there.
    d = _tf_gcd(x1, x2)
    # Same as the `x2_safe` trick above
    d_safe = array_ops.where_v2(
        math_ops.equal(d, 0), constant_op.constant(1, d.dtype), d
    )
    x1 = math_ops.abs(x1)
    x2 = math_ops.abs(x2)
    return array_ops.where_v2(
        math_ops.equal(d, 0),
        constant_op.constant(0, d.dtype),
        x1 * (x2 // d_safe),
    )

  return _bin_op(f, x1, x2)


def _bitwise_binary_op(tf_fn, x1, x2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # HELPER: shared implementation of bitwise_and, bitwise_or and bitwise_xor.
  # Inputs: tf_fn, a bitwise tensor function of two arguments; x1, x2 array-likes (promoted).
  # Boolean operands are computed as 8-bit integers and the result cast back to boolean.
  # Returns an array.
  def f(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body: casts booleans to 8-bit integers, applies tf_fn, and casts the
    # result back to boolean if the inputs were boolean.
    is_bool = x1.dtype == dtypes.bool
    if is_bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    r = tf_fn(x1, x2)
    if is_bool:
      r = math_ops.cast(r, dtypes.bool)
    return r

  return _bin_op(f, x1, x2)


@tf_export.tf_export('experimental.numpy.bitwise_and', v1=[])
@np_utils.np_doc('bitwise_and')
def bitwise_and(x1, x2):
  # [FUNCTION DOC]
  # Elementwise bitwise and of two integer or boolean arrays (promoted, broadcast).
  return _bitwise_binary_op(bitwise_ops.bitwise_and, x1, x2)


@tf_export.tf_export('experimental.numpy.bitwise_or', v1=[])
@np_utils.np_doc('bitwise_or')
def bitwise_or(x1, x2):
  # [FUNCTION DOC]
  # Elementwise bitwise or of two integer or boolean arrays (promoted, broadcast).
  return _bitwise_binary_op(bitwise_ops.bitwise_or, x1, x2)


@tf_export.tf_export('experimental.numpy.bitwise_xor', v1=[])
@np_utils.np_doc('bitwise_xor')
def bitwise_xor(x1, x2):
  # [FUNCTION DOC]
  # Elementwise bitwise exclusive or of two integer or boolean arrays (promoted, broadcast).
  return _bitwise_binary_op(bitwise_ops.bitwise_xor, x1, x2)


@tf_export.tf_export('experimental.numpy.bitwise_not', v1=[])
@np_utils.np_doc('bitwise_not', link=np_utils.AliasOf('invert'))
def bitwise_not(x):
  # [FUNCTION DOC]
  # Elementwise bitwise inversion of an integer array, or logical negation for booleans. Input x
  # is an array-like. Returns an array of the same shape.
  def f(x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of bitwise_not; logical not for booleans, bit inversion otherwise.
    if x.dtype == dtypes.bool:
      return math_ops.logical_not(x)
    return bitwise_ops.invert(x)

  return _scalar(f, x)


def _scalar(tf_fn, x, promote_to_float=False):
  # [FUNCTION DOC]
  # HELPER: shared wrapper for all unary elementwise functions in this module.
  # KNOWN ISSUE: the docstring's output-dtype description is loose; the float default applies only
  # when promote_to_float is true and the input is not already floating or complex.
  # Converts x to a framework array, optionally casts non-inexact dtypes to the configured float
  # type (promote_to_float, default false), and applies tf_fn. Returns the result of tf_fn.
  """Computes the tf_fn(x) for each element in `x`.

  Args:
    tf_fn: function that takes a single Tensor argument.
    x: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `ops.convert_to_tensor`.
    promote_to_float: whether to cast the argument to a float dtype if it is not
      already.

  Returns:
    An ndarray with the same shape as `x`. The default output dtype is
    determined by `np_utils.result_type(float)`, unless x is an ndarray with a
    floating point type, in which case the output type is same as x.dtype.
  """
  x = np_array_ops.asarray(x)
  if promote_to_float and not np.issubdtype(x.dtype.as_numpy_dtype, np.inexact):
    x = x.astype(np_utils.result_type(float))
  return tf_fn(x)


@tf_export.tf_export('experimental.numpy.log', v1=[])
@np_utils.np_doc('log')
def log(x):
  # [FUNCTION DOC]
  # Elementwise natural logarithm; integer inputs are converted to the configured float type.
  return _scalar(math_ops.log, x, True)


@tf_export.tf_export('experimental.numpy.exp', v1=[])
@np_utils.np_doc('exp')
def exp(x):
  # [FUNCTION DOC]
  # Elementwise exponential; integer inputs are converted to the configured float type.
  return _scalar(math_ops.exp, x, True)


@tf_export.tf_export('experimental.numpy.sqrt', v1=[])
@np_utils.np_doc('sqrt')
def sqrt(x):
  # [FUNCTION DOC]
  # Elementwise square root (not-a-number for negative reals); integers become the float type.
  return _scalar(math_ops.sqrt, x, True)


@tf_export.tf_export('experimental.numpy.abs', v1=[])
@np_utils.np_doc('abs', link=np_utils.AliasOf('absolute'))
def abs(x):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Elementwise absolute value (magnitude for complex). Shadows the Python builtin inside this
  # module. Dtype is preserved.
  return _scalar(math_ops.abs, x)


@tf_export.tf_export('experimental.numpy.absolute', v1=[])
@np_utils.np_doc('absolute')
def absolute(x):
  # [FUNCTION DOC]
  # Alias of abs: elementwise absolute value.
  return abs(x)


@tf_export.tf_export('experimental.numpy.fabs', v1=[])
@np_utils.np_doc('fabs')
def fabs(x):
  # [FUNCTION DOC]
  # KNOWN ISSUE: NumPy's fabs always returns floats, but this delegates to abs and keeps the
  # input dtype (integers stay integers).
  # Elementwise absolute value.
  return abs(x)


@tf_export.tf_export('experimental.numpy.ceil', v1=[])
@np_utils.np_doc('ceil')
def ceil(x):
  # [FUNCTION DOC]
  # Elementwise smallest integer value not less than x, as floats (integers are first converted
  # to the configured float type).
  return _scalar(math_ops.ceil, x, True)


@tf_export.tf_export('experimental.numpy.floor', v1=[])
@np_utils.np_doc('floor')
def floor(x):
  # [FUNCTION DOC]
  # Elementwise largest integer value not greater than x, as floats (integers are first
  # converted to the configured float type).
  return _scalar(math_ops.floor, x, True)


@tf_export.tf_export('experimental.numpy.conj', v1=[])
@np_utils.np_doc('conj')
def conj(x):
  # [FUNCTION DOC]
  # Elementwise complex conjugate (identity for real inputs).
  return _scalar(math_ops.conj, x)


@tf_export.tf_export('experimental.numpy.negative', v1=[])
@np_utils.np_doc('negative')
def negative(x):
  # [FUNCTION DOC]
  # Elementwise numerical negation; dtype is preserved.
  return _scalar(math_ops.negative, x)


@tf_export.tf_export('experimental.numpy.reciprocal', v1=[])
@np_utils.np_doc('reciprocal')
def reciprocal(x):
  # [FUNCTION DOC]
  # Elementwise one divided by x; dtype is preserved, so integer inputs use integer division.
  return _scalar(math_ops.reciprocal, x)


@tf_export.tf_export('experimental.numpy.signbit', v1=[])
@np_utils.np_doc('signbit')
def signbit(x):
  # [FUNCTION DOC]
  # KNOWN ISSUE: implemented as a less-than-zero test, so negative zero and negative
  # not-a-number report false, unlike NumPy's sign-bit check.
  # Elementwise test of whether x is negative. Booleans give all false. Returns a boolean array.
  def f(x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of signbit: false-filled result for booleans, else x less than 0.
    if x.dtype == dtypes.bool:
      return array_ops.fill(array_ops.shape(x), False)
    return x < 0

  return _scalar(f, x)


@tf_export.tf_export('experimental.numpy.sin', v1=[])
@np_utils.np_doc('sin')
def sin(x):
  # [FUNCTION DOC]
  # Elementwise sine (radians); integers are converted to the configured float type.
  return _scalar(math_ops.sin, x, True)


@tf_export.tf_export('experimental.numpy.cos', v1=[])
@np_utils.np_doc('cos')
def cos(x):
  # [FUNCTION DOC]
  # Elementwise cosine (radians); integers are converted to the configured float type.
  return _scalar(math_ops.cos, x, True)


@tf_export.tf_export('experimental.numpy.tan', v1=[])
@np_utils.np_doc('tan')
def tan(x):
  # [FUNCTION DOC]
  # Elementwise tangent (radians); integers are converted to the configured float type.
  return _scalar(math_ops.tan, x, True)


@tf_export.tf_export('experimental.numpy.sinh', v1=[])
@np_utils.np_doc('sinh')
def sinh(x):
  # [FUNCTION DOC]
  # Elementwise hyperbolic sine; integers are converted to the configured float type.
  return _scalar(math_ops.sinh, x, True)


@tf_export.tf_export('experimental.numpy.cosh', v1=[])
@np_utils.np_doc('cosh')
def cosh(x):
  # [FUNCTION DOC]
  # Elementwise hyperbolic cosine; integers are converted to the configured float type.
  return _scalar(math_ops.cosh, x, True)


@tf_export.tf_export('experimental.numpy.tanh', v1=[])
@np_utils.np_doc('tanh')
def tanh(x):
  # [FUNCTION DOC]
  # Elementwise hyperbolic tangent; integers are converted to the configured float type.
  return _scalar(math_ops.tanh, x, True)


@tf_export.tf_export('experimental.numpy.arcsin', v1=[])
@np_utils.np_doc('arcsin')
def arcsin(x):
  # [FUNCTION DOC]
  # Elementwise inverse sine in radians; integers are converted to the configured float type.
  return _scalar(math_ops.asin, x, True)


@tf_export.tf_export('experimental.numpy.arccos', v1=[])
@np_utils.np_doc('arccos')
def arccos(x):
  # [FUNCTION DOC]
  # Elementwise inverse cosine in radians; integers are converted to the configured float type.
  return _scalar(math_ops.acos, x, True)


@tf_export.tf_export('experimental.numpy.arctan', v1=[])
@np_utils.np_doc('arctan')
def arctan(x):
  # [FUNCTION DOC]
  # Elementwise inverse tangent in radians; integers are converted to the configured float type.
  return _scalar(math_ops.atan, x, True)


@tf_export.tf_export('experimental.numpy.arcsinh', v1=[])
@np_utils.np_doc('arcsinh')
def arcsinh(x):
  # [FUNCTION DOC]
  # Elementwise inverse hyperbolic sine; integers are converted to the configured float type.
  return _scalar(math_ops.asinh, x, True)


@tf_export.tf_export('experimental.numpy.arccosh', v1=[])
@np_utils.np_doc('arccosh')
def arccosh(x):
  # [FUNCTION DOC]
  # Elementwise inverse hyperbolic cosine; integers are converted to the configured float type.
  return _scalar(math_ops.acosh, x, True)


@tf_export.tf_export('experimental.numpy.arctanh', v1=[])
@np_utils.np_doc('arctanh')
def arctanh(x):
  # [FUNCTION DOC]
  # Elementwise inverse hyperbolic tangent; integers are converted to the configured float type.
  return _scalar(math_ops.atanh, x, True)


@tf_export.tf_export('experimental.numpy.deg2rad', v1=[])
@np_utils.np_doc('deg2rad')
def deg2rad(x):
  # [FUNCTION DOC]
  # Converts angles from degrees to radians elementwise (multiplies by pi over 180). Integers
  # are converted to the configured float type first.
  def f(x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of deg2rad: multiplies by pi divided by 180.
    return x * (np.pi / 180.0)

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.rad2deg', v1=[])
@np_utils.np_doc('rad2deg')
def rad2deg(x):
  # [FUNCTION DOC]
  # KNOWN ISSUE: unlike its siblings it does not convert the input to an array first, so a plain
  # Python list fails (list times float), and an integer array input keeps its own handling.
  # Converts angles from radians to degrees by multiplying by 180 over pi. Returns the product.
  return x * (180.0 / np.pi)


_tf_float_types = [
    dtypes.bfloat16,
    dtypes.float16,
    dtypes.float32,
    dtypes.float64,
]


@tf_export.tf_export('experimental.numpy.angle', v1=[])
@np_utils.np_doc('angle')
def angle(z, deg=False):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Argument (phase angle) of complex numbers. Inputs: z array-like; deg (default false) returns
  # degrees instead of radians. For real floating inputs it returns pi where negative and zero
  # elsewhere (a workaround for a known framework issue); complex inputs use the framework angle
  # op; integers are first converted to float. Returns an array.
  def f(x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of angle: pi-or-zero selection for float dtypes, else the
    # framework phase-angle op.
    if x.dtype in _tf_float_types:
      # Workaround for b/147515503
      return array_ops.where_v2(x < 0, np.pi, 0)
    else:
      return math_ops.angle(x)

  y = _scalar(f, z, True)
  if deg:
    y = rad2deg(y)
  return y


@tf_export.tf_export('experimental.numpy.cbrt', v1=[])
@np_utils.np_doc('cbrt')
def cbrt(x):
  # [FUNCTION DOC]
  # Elementwise real cube root (negative inputs give negative roots); integers are converted to
  # the configured float type.
  def f(x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of cbrt: raises the absolute value to the power one third and
    # restores the sign.
    # __pow__ can't handle negative base, so we use `abs` here.
    rt = math_ops.abs(x) ** (1.0 / 3)
    return array_ops.where_v2(x < 0, -rt, rt)

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.conjugate', v1=[])
@np_utils.np_doc('conjugate', link=np_utils.AliasOf('conj'))
def conjugate(x):
  # [FUNCTION DOC]
  # Alias behavior of conj: elementwise complex conjugate.
  return _scalar(math_ops.conj, x)


@tf_export.tf_export('experimental.numpy.exp2', v1=[])
@np_utils.np_doc('exp2')
def exp2(x):
  # [FUNCTION DOC]
  # Elementwise two raised to the power x; integers are converted to the configured float type.
  def f(x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of exp2: two to the power x.
    return 2**x

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.expm1', v1=[])
@np_utils.np_doc('expm1')
def expm1(x):
  # [FUNCTION DOC]
  # Elementwise exp(x) minus one, accurate for small x; integers become the float type.
  return _scalar(math_ops.expm1, x, True)


@tf_export.tf_export('experimental.numpy.fix', v1=[])
@np_utils.np_doc('fix')
def fix(x):
  # [FUNCTION DOC]
  # Rounds elementwise toward zero (ceiling for negatives, floor otherwise); integers become the
  # configured float type.
  def f(x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of fix: selects ceiling where negative, floor elsewhere.
    return array_ops.where_v2(x < 0, math_ops.ceil(x), math_ops.floor(x))

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.iscomplex', v1=[])
@np_utils.np_doc('iscomplex')
def iscomplex(x):
  # [FUNCTION DOC]
  # Elementwise test of whether the imaginary part is non-zero. Returns a boolean array.
  return np_array_ops.imag(x) != 0


@tf_export.tf_export('experimental.numpy.isreal', v1=[])
@np_utils.np_doc('isreal')
def isreal(x):
  # [FUNCTION DOC]
  # Elementwise test of whether the imaginary part is zero. Returns a boolean array.
  return np_array_ops.imag(x) == 0


@tf_export.tf_export('experimental.numpy.iscomplexobj', v1=[])
@np_utils.np_doc('iscomplexobj')
def iscomplexobj(x):
  # [FUNCTION DOC]
  # Returns a Python boolean: true if the dtype of x (after conversion to an array) is a complex
  # floating type, regardless of the values.
  x = np_array_ops.array(x)
  return np.issubdtype(x.dtype.as_numpy_dtype, np.complexfloating)


@tf_export.tf_export('experimental.numpy.isrealobj', v1=[])
@np_utils.np_doc('isrealobj')
def isrealobj(x):
  # [FUNCTION DOC]
  # Returns a Python boolean: the negation of iscomplexobj for x.
  return not iscomplexobj(x)


@tf_export.tf_export('experimental.numpy.isnan', v1=[])
@np_utils.np_doc('isnan')
def isnan(x):
  # [FUNCTION DOC]
  # Elementwise not-a-number test. Integers are first converted to the float type (so always
  # false). Returns a boolean array.
  return _scalar(math_ops.is_nan, x, True)


def _make_nan_reduction(np_fun_name, reduction, init_val):
  # [FUNCTION DOC]
  # HELPER: factory for nansum and nanprod.
  # Inputs: np_fun_name, name of the NumPy function whose documentation is borrowed; reduction,
  # the underlying reduction function (sum or product); init_val, the neutral value that replaces
  # not-a-number entries. Returns a new reduction function taking a, axis, dtype and keepdims.
  """Helper to generate nan* functions."""

  @np_utils.np_doc(np_fun_name)
  def nan_reduction(a, axis=None, dtype=None, keepdims=False):
    # [FUNCTION DOC]
    # HELPER: generated reduction. Converts a to an array, replaces not-a-number entries by the
    # neutral value, and reduces over the axis (default all) with optional dtype and keepdims
    # (default false). Returns an array.
    a = np_array_ops.array(a)
    v = np_array_ops.array(init_val, dtype=a.dtype)
    return reduction(
        np_array_ops.where(isnan(a), v, a),
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
    )

  return nan_reduction


nansum = tf_export.tf_export('experimental.numpy.nansum', v1=[])(
    _make_nan_reduction('nansum', np_array_ops.sum, 0)
)
nanprod = tf_export.tf_export('experimental.numpy.nanprod', v1=[])(
    _make_nan_reduction('nanprod', np_array_ops.prod, 1)
)


@tf_export.tf_export('experimental.numpy.nanmean', v1=[])
@np_utils.np_doc('nanmean')
def nanmean(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Mean ignoring not-a-number entries. Inputs: a array-like; axis optional; dtype optional
  # (defaults to the input dtype); keepdims optional. Boolean and integer inputs use the ordinary
  # mean. Otherwise it divides nansum by the count of non-NaN entries. Returns an array (all-NaN
  # slices give not-a-number/division results).
  a = np_array_ops.array(a)
  if np.issubdtype(a.dtype.as_numpy_dtype, np.bool_) or np.issubdtype(
      a.dtype.as_numpy_dtype, np.integer
  ):
    return np_array_ops.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
  nan_mask = logical_not(isnan(a))
  if dtype is None:
    dtype = a.dtype.as_numpy_dtype
  normalizer = np_array_ops.sum(
      nan_mask, axis=axis, dtype=dtype, keepdims=keepdims
  )
  return nansum(a, axis=axis, dtype=dtype, keepdims=keepdims) / normalizer


@tf_export.tf_export('experimental.numpy.isfinite', v1=[])
@np_utils.np_doc('isfinite')
def isfinite(x):
  # [FUNCTION DOC]
  # Elementwise test that x is neither infinite nor not-a-number. Returns a boolean array.
  return _scalar(math_ops.is_finite, x, True)


@tf_export.tf_export('experimental.numpy.isinf', v1=[])
@np_utils.np_doc('isinf')
def isinf(x):
  # [FUNCTION DOC]
  # KNOWN ISSUE: reads the dtype attribute of x directly without converting it, so Python lists
  # or scalars raise an attribute error; for non-floating dtypes it returns a plain Python false
  # instead of an array of false values.
  # Elementwise infinity test (positive or negative) for floating tensors; returns a boolean
  # array for floating dtypes, otherwise the Python value false.
  if x.dtype.is_floating:
    return _scalar(math_ops.is_inf, x, True)
  return False


@tf_export.tf_export('experimental.numpy.isneginf', v1=[])
@np_utils.np_doc('isneginf')
def isneginf(x):
  # [FUNCTION DOC]
  # KNOWN ISSUE: reads the dtype attribute of x directly without converting it, so Python lists
  # or scalars raise an attribute error; for non-floating dtypes it returns a plain Python false
  # instead of an array of false values.
  # Elementwise test for negative infinity: compares floating x with a negative-infinity array of
  # the same shape. Otherwise returns the Python value false.
  if x.dtype.is_floating:
    return x == np_array_ops.full_like(x, -np.inf)
  return False


@tf_export.tf_export('experimental.numpy.isposinf', v1=[])
@np_utils.np_doc('isposinf')
def isposinf(x):
  # [FUNCTION DOC]
  # KNOWN ISSUE: reads the dtype attribute of x directly without converting it, so Python lists
  # or scalars raise an attribute error; for non-floating dtypes it returns a plain Python false
  # instead of an array of false values.
  # Elementwise test for positive infinity: compares floating x with a positive-infinity array of
  # the same shape. Otherwise returns the Python value false.
  if x.dtype.is_floating:
    return x == np_array_ops.full_like(x, np.inf)
  return False


@tf_export.tf_export('experimental.numpy.log2', v1=[])
@np_utils.np_doc('log2')
def log2(x):
  # [FUNCTION DOC]
  # Elementwise base-two logarithm computed as natural log divided by the natural log of two.
  return log(x) / np.log(2)


@tf_export.tf_export('experimental.numpy.log10', v1=[])
@np_utils.np_doc('log10')
def log10(x):
  # [FUNCTION DOC]
  # Elementwise base-ten logarithm computed as natural log divided by the natural log of ten.
  return log(x) / np.log(10)


@tf_export.tf_export('experimental.numpy.log1p', v1=[])
@np_utils.np_doc('log1p')
def log1p(x):
  # [FUNCTION DOC]
  # Elementwise natural log of one plus x, accurate for small x; integers become the float type.
  return _scalar(math_ops.log1p, x, True)


@tf_export.tf_export('experimental.numpy.positive', v1=[])
@np_utils.np_doc('positive')
def positive(x):
  # [FUNCTION DOC]
  # Elementwise numerical positive (identity); returns x converted to an array.
  return _scalar(lambda x: x, x)


@tf_export.tf_export('experimental.numpy.sinc', v1=[])
@np_utils.np_doc('sinc')
def sinc(x):
  # [FUNCTION DOC]
  # Normalized sinc function: sin(pi x) divided by pi x, with value one at x equal to zero.
  # Integers are converted to the configured float type. Returns an array.
  def f(x):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of sinc, selecting one where x is zero.
    pi_x = x * np.pi
    return array_ops.where_v2(
        x == 0, array_ops.ones_like(x), math_ops.sin(pi_x) / pi_x
    )

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.square', v1=[])
@np_utils.np_doc('square')
def square(x):
  # [FUNCTION DOC]
  # Elementwise square of x; dtype is preserved.
  return _scalar(math_ops.square, x)


@tf_export.tf_export('experimental.numpy.diff', v1=[])
@np_utils.np_doc('diff')
def diff(a, n=1, axis=-1):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: the negative-order error message says "Argument order ... Received: axis=n",
  # mislabeling the parameter (it is n, not axis).
  # n-th discrete difference along an axis. Inputs: a array-like (rank must be statically known);
  # n number of times to difference (default 1, must be non-negative); axis (default last).
  # Raises ValueError for unknown rank, out-of-range axis or negative n. For booleans uses not
  # equal instead of subtraction. Returns an array shorter by n along the axis.
  def f(a):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of diff: validates rank, axis and n, builds slices that offset by
    # one along the axis, and repeatedly combines shifted slices n times.
    # TODO(agarwal): transpose and reshape to N, H, 1 and do a 1D convolution
    # TODO(agarwal): avoid depending on static rank.
    nd = a.shape.rank
    if nd is None:
      raise ValueError(
          'Function `diff` currently requires a known rank for input `a`. '
          f'Received: a={a} (unknown rank)'
      )
    if (axis + nd if axis < 0 else axis) >= nd:
      raise ValueError(
          f'Argument `axis` (received axis={axis}) is out of bounds '
          f'for input {a} of rank {nd}.'
      )
    if n < 0:
      raise ValueError(
          f'Argument `order` must be a non-negative integer. Received: axis={n}'
      )
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    op = math_ops.not_equal if a.dtype == dtypes.bool else math_ops.subtract
    for _ in range(n):
      a = op(a[slice1], a[slice2])
    return a

  return _scalar(f, a)


def _wrap(f, reverse=False):
  # [FUNCTION DOC]
  # HELPER: builds operator-overload adapters for the ndarray class.
  # Input f: binary function; reverse (default false) swaps the operands first (for reflected
  # operators). Returns a function of two arguments that returns the not-implemented marker if
  # the other operand has a higher array priority than ndarray, otherwise calls f.
  """Wraps binary ops so they can be added as operator overloads on ndarray."""

  def _f(a, b):
    # [FUNCTION DOC]
    # HELPER: the generated operator adapter. Swaps operands if reverse was requested, defers
    # (not-implemented marker) to operands with higher array priority, else returns f(a, b).
    if reverse:
      a, b = b, a

    if (
        getattr(b, '__array_priority__', 0)
        > np_arrays.ndarray.__array_priority__
    ):
      return NotImplemented

    return f(a, b)

  return _f


def _comparison(tf_fun, x1, x2, cast_bool_to_int=False):
  # [FUNCTION DOC]
  # HELPER: shared implementation of the comparison functions (equal, less, ...).
  # Converts both inputs to arrays of their common result type, optionally (cast_bool_to_int,
  # default false) casts boolean pairs to 32-bit integers (needed for ordering ops), and applies
  # tf_fun. Returns a boolean array.
  """Helper function for comparision."""
  dtype = np_utils.result_type(x1, x2)
  # Cast x1 and x2 to the result_type if needed.
  x1 = np_array_ops.array(x1, dtype=dtype)
  x2 = np_array_ops.array(x2, dtype=dtype)
  if cast_bool_to_int and x1.dtype == dtypes.bool:
    x1 = math_ops.cast(x1, dtypes.int32)
    x2 = math_ops.cast(x2, dtypes.int32)
  return tf_fun(x1, x2)


@tf_export.tf_export('experimental.numpy.equal', v1=[])
@np_utils.np_doc('equal')
def equal(x1, x2):
  # [FUNCTION DOC]
  # Elementwise equality of x1 and x2 after conversion to a common type. Returns boolean array.
  return _comparison(math_ops.equal, x1, x2)


@tf_export.tf_export('experimental.numpy.not_equal', v1=[])
@np_utils.np_doc('not_equal')
def not_equal(x1, x2):
  # [FUNCTION DOC]
  # Elementwise inequality of x1 and x2 after conversion to a common type. Returns boolean array.
  return _comparison(math_ops.not_equal, x1, x2)


@tf_export.tf_export('experimental.numpy.greater', v1=[])
@np_utils.np_doc('greater')
def greater(x1, x2):
  # [FUNCTION DOC]
  # Elementwise x1 greater than x2 (booleans compared as integers). Returns boolean array.
  return _comparison(math_ops.greater, x1, x2, True)


@tf_export.tf_export('experimental.numpy.greater_equal', v1=[])
@np_utils.np_doc('greater_equal')
def greater_equal(x1, x2):
  # [FUNCTION DOC]
  # Elementwise x1 greater than or equal to x2 (booleans compared as integers). Boolean array.
  return _comparison(math_ops.greater_equal, x1, x2, True)


@tf_export.tf_export('experimental.numpy.less', v1=[])
@np_utils.np_doc('less')
def less(x1, x2):
  # [FUNCTION DOC]
  # Elementwise x1 less than x2 (booleans compared as integers). Returns boolean array.
  return _comparison(math_ops.less, x1, x2, True)


@tf_export.tf_export('experimental.numpy.less_equal', v1=[])
@np_utils.np_doc('less_equal')
def less_equal(x1, x2):
  # [FUNCTION DOC]
  # Elementwise x1 less than or equal to x2 (booleans compared as integers). Boolean array.
  return _comparison(math_ops.less_equal, x1, x2, True)


@tf_export.tf_export('experimental.numpy.array_equal', v1=[])
@np_utils.np_doc('array_equal')
def array_equal(a1, a2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Returns a scalar boolean array: true only if a1 and a2 have the same rank and shape and all
  # elements are equal (after conversion to a common type). Different shapes give false instead
  # of raising.
  def f(x1, x2):
    # [FUNCTION DOC]
    # HELPER: tensor-level body of array_equal: checks equal rank, then equal shape, then all
    # elements equal; any failed check yields a constant false.
    return np_utils.cond(
        math_ops.equal(array_ops.rank(x1), array_ops.rank(x2)),
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            np_utils.reduce_all(
                math_ops.equal(array_ops.shape(x1), array_ops.shape(x2))
            ),
            lambda: math_ops.reduce_all(math_ops.equal(x1, x2)),
            lambda: constant_op.constant(False),
        ),
        lambda: constant_op.constant(False),
    )

  return _comparison(f, a1, a2)


def _logical_binary_op(tf_fun, x1, x2):
  # [FUNCTION DOC]
  # HELPER: shared implementation of logical_and, logical_or and logical_xor.
  # Converts both inputs to boolean arrays (non-zero is true), then applies tf_fun. Returns a
  # boolean array.
  x1 = np_array_ops.array(x1, dtype=np.bool_)
  x2 = np_array_ops.array(x2, dtype=np.bool_)
  return tf_fun(x1, x2)


@tf_export.tf_export('experimental.numpy.logical_and', v1=[])
@np_utils.np_doc('logical_and')
def logical_and(x1, x2):
  # [FUNCTION DOC]
  # Elementwise logical and after converting both inputs to boolean arrays.
  return _logical_binary_op(math_ops.logical_and, x1, x2)


@tf_export.tf_export('experimental.numpy.logical_or', v1=[])
@np_utils.np_doc('logical_or')
def logical_or(x1, x2):
  # [FUNCTION DOC]
  # Elementwise logical or after converting both inputs to boolean arrays.
  return _logical_binary_op(math_ops.logical_or, x1, x2)


@tf_export.tf_export('experimental.numpy.logical_xor', v1=[])
@np_utils.np_doc('logical_xor')
def logical_xor(x1, x2):
  # [FUNCTION DOC]
  # Elementwise logical exclusive or after converting both inputs to boolean arrays.
  return _logical_binary_op(math_ops.logical_xor, x1, x2)


@tf_export.tf_export('experimental.numpy.logical_not', v1=[])
@np_utils.np_doc('logical_not')
def logical_not(x):
  # [FUNCTION DOC]
  # Elementwise logical negation after converting x to a boolean array.
  x = np_array_ops.array(x, dtype=np.bool_)
  return math_ops.logical_not(x)


@tf_export.tf_export('experimental.numpy.linspace', v1=[])
@np_utils.np_doc('linspace')
def linspace(  # pylint: disable=missing-docstring
    start, stop, num=50, endpoint=True, retstep=False, dtype=float, axis=0
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: with numpy 2 or newer installed, a truthy dtype argument is ignored and the
  # result dtype is inferred from start and stop instead; with older numpy it is honored.
  # Evenly spaced numbers over an interval. Inputs: start, stop (array-likes, converted to the
  # chosen dtype); num samples (default 50, must be non-negative, else ValueError); endpoint
  # (default true) includes stop; retstep (default false) also returns the spacing; dtype
  # (default the float type; a false value disables casting); axis (default 0) where samples are
  # placed. Without endpoint, the interval is computed manually with one more step. Integer
  # dtypes floor the result before casting. Returns the samples, or a pair of samples and step
  # (step is not-a-number when it is undefined).
  if dtype:
    # In numpy 2.x, the result type of np.linspace is based off of `start` and
    # `end`. We mimic the behavior.
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0.dev0':
      dtype = np_utils.result_type([start * 1.0, stop * 1.0])
    else:
      dtype = np_utils.result_type(dtype)
  start = np_array_ops.array(start, dtype=dtype)
  stop = np_array_ops.array(stop, dtype=dtype)
  if num < 0:
    raise ValueError(
        'Argument `num` (number of samples) must be a non-negative integer. '
        f'Received: num={num}'
    )
  step = ops.convert_to_tensor(np.nan)
  if endpoint:
    result = math_ops.linspace(start, stop, num, axis=axis)
    if num > 1:
      step = (stop - start) / (num - 1)
  else:
    # math_ops.linspace does not support endpoint=False so we manually handle it
    # here.
    if num > 0:
      step = (stop - start) / num
    if num > 1:
      new_stop = math_ops.cast(stop, step.dtype) - step
      start = math_ops.cast(start, new_stop.dtype)
      result = math_ops.linspace(start, new_stop, num, axis=axis)
    else:
      result = math_ops.linspace(start, stop, num, axis=axis)
  if dtype:
    if dtype.is_integer:
      # Since numpy 1.20, linspace's rounding is towards -inf instead of 0
      result = math_ops.floor(result)
    result = math_ops.cast(result, dtype)
  if retstep:
    return (result, step)
  else:
    return result


@tf_export.tf_export('experimental.numpy.logspace', v1=[])
@np_utils.np_doc('logspace')
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
  # [FUNCTION DOC]
  # KNOWN ISSUE: with numpy 2 or newer installed, the dtype argument is always overridden by a
  # type inferred from start and stop.
  # Numbers spaced evenly on a log scale: base raised to a linear sequence from start to stop.
  # Inputs: start, stop exponents; num (default 50); endpoint (default true); base (default 10);
  # dtype (optional output type); axis (default 0). Returns an array cast to the chosen dtype.
  # In numpy 2.x, the result type of np.logspace is based off of `start` and
  # `end`. We mimic the behavior.
  if np.lib.NumpyVersion(np.__version__) >= '2.0.0.dev0':
    dtype = np_utils.result_type([start * 1.0, stop * 1.0])
  else:
    dtype = np_utils.result_type(start, stop, dtype)
  result = linspace(
      start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
  )
  result = math_ops.pow(math_ops.cast(base, result.dtype), result)
  if dtype:
    result = math_ops.cast(result, dtype)
  return result


@tf_export.tf_export('experimental.numpy.geomspace', v1=[])
@np_utils.np_doc('geomspace')
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Numbers spaced evenly on a geometric progression from start to stop. Inputs: start, stop
  # (non-zero, may be negative or complex); num (default 50); endpoint (default true); dtype
  # (optional, otherwise inferred); axis (default 0, result axis moved there at the end).
  # Computes in at least single precision using logspace on base-ten logarithms, flipping signs
  # to follow NumPy's convention for negative endpoints. Returns an array cast to the dtype.
  dtype = (
      dtypes.as_dtype(dtype)  # pylint: disable=g-long-ternary
      if dtype
      else np_utils.result_type(
          start, stop, float(num), np_array_ops.zeros((), dtype)
      )
  )
  computation_dtype = np.promote_types(dtype.as_numpy_dtype, np.float32)
  start = np_array_ops.asarray(start, dtype=computation_dtype)
  stop = np_array_ops.asarray(stop, dtype=computation_dtype)
  # follow the numpy geomspace convention for negative and complex endpoints
  start_sign = 1 - np_array_ops.sign(np_array_ops.real(start))
  stop_sign = 1 - np_array_ops.sign(np_array_ops.real(stop))
  signflip = 1 - start_sign * stop_sign // 2
  res = signflip * logspace(
      log10(signflip * start),
      log10(signflip * stop),
      num,
      endpoint=endpoint,
      base=10.0,
      dtype=computation_dtype,
      axis=0,
  )
  if axis != 0:
    res = np_array_ops.moveaxis(res, 0, axis)
  return math_ops.cast(res, dtype)


@tf_export.tf_export('experimental.numpy.ptp', v1=[])
@np_utils.np_doc('ptp')
def ptp(a, axis=None, keepdims=None):
  # [FUNCTION DOC]
  # Range (maximum minus minimum) of values along an axis. Inputs: a array-like; axis optional
  # (default all); keepdims optional. Returns an array.
  return np_array_ops.amax(a, axis=axis, keepdims=keepdims) - np_array_ops.amin(
      a, axis=axis, keepdims=keepdims
  )


@tf_export.tf_export('experimental.numpy.concatenate', v1=[])
@np_utils.np_doc_only('concatenate')
def concatenate(arys, axis=0):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Joins arrays along an existing axis. Inputs: arys, a list or tuple of array-likes (a single
  # non-sequence is wrapped in a list); axis (default 0). All are converted to their common
  # result type. Raises ValueError for an empty list. Returns one array.
  if not isinstance(arys, (list, tuple)):
    arys = [arys]
  if not arys:
    raise ValueError(
        'Need at least one array to concatenate. Received empty '
        f'input: arys={arys}'
    )
  dtype = np_utils.result_type(*arys)
  arys = [np_array_ops.array(array, dtype=dtype) for array in arys]
  return array_ops.concat(arys, axis)


@tf_export.tf_export('experimental.numpy.tile', v1=[])
@np_utils.np_doc_only('tile')
def tile(a, reps):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Repeats an array. Inputs: a array-like; reps, repetitions per axis (integer or sequence,
  # flattened). When reps is shorter than the rank of a, it is padded with leading ones; when a
  # has lower rank, its shape gets leading ones. Returns the tiled array.
  a = np_array_ops.array(a)
  reps = array_ops.reshape(np_array_ops.array(reps, dtype=dtypes.int32), [-1])

  a_rank = array_ops.rank(a)
  reps_size = array_ops.size(reps)
  reps = array_ops.pad(
      reps, [[math_ops.maximum(a_rank - reps_size, 0), 0]], constant_values=1
  )
  a_shape = array_ops.pad(
      array_ops.shape(a),
      [[math_ops.maximum(reps_size - a_rank, 0), 0]],
      constant_values=1,
  )
  a = array_ops.reshape(a, a_shape)

  return array_ops.tile(a, reps)


@tf_export.tf_export('experimental.numpy.count_nonzero', v1=[])
@np_utils.np_doc('count_nonzero')
def count_nonzero(a, axis=None):
  # [FUNCTION DOC]
  # Counts non-zero elements of array-like a, over all elements or along axis. Returns a tensor
  # (the framework result is not wrapped in an array conversion).
  return math_ops.count_nonzero(np_array_ops.array(a), axis)


@tf_export.tf_export('experimental.numpy.argsort', v1=[])
@np_utils.np_doc('argsort')
def argsort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Indices that would sort a. Inputs: a array-like (string tensors unsupported); axis (default
  # last; none flattens first); kind, either quicksort (default) or stable, which selects a stable
  # sort; order must be nothing. Raises ValueError for another kind or non-nothing order, and
  # TypeError for complex dtypes. A zero-dimensional input gives a one-element result containing
  # zero. Returns an integer array (default integer type if automatic dtype conversion is on,
  # otherwise pointer-sized integer).
  # TODO(nareshmodi): make string tensors also work.
  if kind not in ('quicksort', 'stable'):
    raise ValueError(
        'Invalid value for argument `kind`. '
        'Only kind="quicksort" and kind="stable" are supported. '
        f'Received: kind={kind}'
    )
  if order is not None:
    raise ValueError('The `order` argument is not supported. Pass order=None')
  stable = kind == 'stable'

  a = np_array_ops.array(a)

  if a.dtype in (dtypes.complex64, dtypes.complex128):
    raise TypeError(
        'argsort does not support complex64/complex128 dtypes. '
        f'Received dtype: {a.dtype}'
    )

  def _argsort(a, axis, stable):
    # [FUNCTION DOC]
    # HELPER: for argsort. Flattens a first when axis is none (sorting along the first axis of the
    # flattened data), then returns the sort indices, using the stable algorithm if requested.
    if axis is None:
      a = array_ops.reshape(a, [-1])
      axis = 0

    return sort_ops.argsort(a, axis, stable=stable)

  tf_ans = np_utils.cond(
      math_ops.equal(array_ops.rank(a), 0),
      lambda: constant_op.constant([0]),
      lambda: _argsort(a, axis, stable),
  )

  if ops.is_auto_dtype_conversion_enabled():
    return np_array_ops.array(tf_ans, dtype=int)
  else:
    return np_array_ops.array(tf_ans, dtype=np.intp)


@tf_export.tf_export('experimental.numpy.sort', v1=[])
@np_utils.np_doc('sort')
def sort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Returns a sorted copy of a. Inputs: a array-like; axis (default last; none flattens first);
  # kind must be quicksort; order must be nothing. Raises ValueError otherwise.
  if kind != 'quicksort':
    raise ValueError(
        'Invalid value for argument `kind`. '
        'Only kind="quicksort" is supported. '
        f'Received: kind={kind}'
    )
  if order is not None:
    raise ValueError('The `order` argument is not supported. Pass order=None')

  a = np_array_ops.array(a)

  if axis is None:
    return sort_ops.sort(array_ops.reshape(a, [-1]), 0)
  else:
    return sort_ops.sort(a, axis)


def _argminmax(fn, a, axis=None):
  # [FUNCTION DOC]
  # HELPER: shared implementation of argmax and argmin.
  # Inputs: fn, the framework arg-reduction op; a array-like; axis optional. With no axis the
  # array is flattened; with an axis, scalars are promoted to one-dimensional. Returns the index
  # tensor from fn.
  a = np_array_ops.array(a)
  if axis is None:
    # When axis is None numpy flattens the array.
    a_t = array_ops.reshape(a, [-1])
  else:
    a_t = np_array_ops.atleast_1d(a)
  return fn(input=a_t, axis=axis)


@tf_export.tf_export('experimental.numpy.argmax', v1=[])
@np_utils.np_doc('argmax')
def argmax(a, axis=None):
  # [FUNCTION DOC]
  # Index of the maximum value of a, over the flattened array or along axis.
  return _argminmax(math_ops.argmax, a, axis)


@tf_export.tf_export('experimental.numpy.argmin', v1=[])
@np_utils.np_doc('argmin')
def argmin(a, axis=None):
  # [FUNCTION DOC]
  # Index of the minimum value of a, over the flattened array or along axis.
  return _argminmax(math_ops.argmin, a, axis)


@tf_export.tf_export('experimental.numpy.append', v1=[])
@np_utils.np_doc('append')
def append(arr, values, axis=None):
  # [FUNCTION DOC]
  # Appends values to arr. With axis none (default) both are flattened and joined into a
  # one-dimensional array; otherwise they are concatenated along axis. Returns a new array.
  if axis is None:
    return concatenate([np_array_ops.ravel(arr), np_array_ops.ravel(values)], 0)
  else:
    return concatenate([arr, values], axis=axis)


@tf_export.tf_export('experimental.numpy.average', v1=[])
@np_utils.np_doc('average')
def average(a, axis=None, weights=None, returned=False):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Weighted average. Inputs: a array-like; axis, an integer or nothing (tuples unsupported:
  # raises ValueError); weights optional array-like, same shape as a or one-dimensional along the
  # axis (mismatches fail via framework assertions with invalid-argument errors); returned
  # (default false) also returns the sum of weights broadcast to the result shape. Integer data
  # is promoted to the float type. Returns the average, or a pair of average and weight sums.
  if axis is not None and not isinstance(axis, int):
    # TODO(wangpeng): Support tuple of ints as `axis`
    raise ValueError(
        'Argument `axis` must be an integer. '
        f'Received axis={axis} (of type {type(axis)})'
    )
  a = np_array_ops.array(a)
  default_float_type = np_utils.result_type(float)
  if weights is None:  # Treat all weights as 1
    if not np.issubdtype(a.dtype.as_numpy_dtype, np.inexact):
      a = a.astype(np_utils.result_type(a.dtype, default_float_type))
    avg = math_ops.reduce_mean(a, axis=axis)
    if returned:
      if axis is None:
        weights_sum = array_ops.size(a)
      else:
        weights_sum = array_ops.shape(a)[axis]
      weights_sum = math_ops.cast(weights_sum, a.dtype)
  else:
    if np.issubdtype(a.dtype.as_numpy_dtype, np.inexact):
      out_dtype = np_utils.result_type(a.dtype, weights)
    else:
      out_dtype = np_utils.result_type(a.dtype, weights, default_float_type)
    a = np_array_ops.array(a, out_dtype)
    weights = np_array_ops.array(weights, out_dtype)

    def rank_equal_case():
      # [FUNCTION DOC]
      # HELPER: for average, when weights has the same rank as a. Asserts equal shapes and
      # returns the weighted average and the sum of weights over the axis.
      control_flow_assert.Assert(
          math_ops.reduce_all(array_ops.shape(a) == array_ops.shape(weights)),
          [array_ops.shape(a), array_ops.shape(weights)],
      )
      weights_sum = math_ops.reduce_sum(weights, axis=axis)
      avg = math_ops.reduce_sum(a * weights, axis=axis) / weights_sum
      return avg, weights_sum

    if axis is None:
      avg, weights_sum = rank_equal_case()
    else:

      def rank_not_equal_case():
        # [FUNCTION DOC]
        # HELPER: for average, when weights has a different rank than a. Asserts weights is
        # one-dimensional, contracts a with weights along the axis, and returns the average and
        # the total weight.
        control_flow_assert.Assert(
            array_ops.rank(weights) == 1, [array_ops.rank(weights)]
        )
        weights_sum = math_ops.reduce_sum(weights)
        axes = ops.convert_to_tensor([[axis], [0]])
        avg = math_ops.tensordot(a, weights, axes) / weights_sum
        return avg, weights_sum

      # We condition on rank rather than shape equality, because if we do the
      # latter, when the shapes are partially unknown but the ranks are known
      # and different, np_utils.cond will run shape checking on the true branch,
      # which will raise a shape-checking error.
      avg, weights_sum = np_utils.cond(
          math_ops.equal(array_ops.rank(a), array_ops.rank(weights)),
          rank_equal_case,
          rank_not_equal_case,
      )

  avg = np_array_ops.array(avg)
  if returned:
    weights_sum = np_array_ops.broadcast_to(weights_sum, array_ops.shape(avg))
    return avg, weights_sum
  return avg


@tf_export.tf_export('experimental.numpy.trace', v1=[])
@np_utils.np_doc('trace')
def trace(a, offset=0, axis1=0, axis2=1, dtype=None):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Sum along a diagonal. Inputs: a array-like; offset diagonal offset (default 0); axis1, axis2
  # the axes spanning the diagonal (defaults 0 and 1); dtype optional accumulator type. Uses the
  # direct trace op when offset is zero and the axes are the last two of a known-rank input;
  # otherwise extracts the diagonal and sums it over the last axis. Returns an array.
  if dtype:
    dtype = np_utils.result_type(dtype)
  a = np_array_ops.asarray(a, dtype)

  if offset == 0:
    a_shape = a.shape
    if a_shape.rank is not None:
      rank = len(a_shape)
      if (axis1 == -2 or axis1 == rank - 2) and (
          axis2 == -1 or axis2 == rank - 1
      ):
        return math_ops.trace(a)

  a = np_array_ops.diagonal(a, offset, axis1, axis2)
  return np_array_ops.sum(a, -1, dtype)


@tf_export.tf_export('experimental.numpy.meshgrid', v1=[])
@np_utils.np_doc('meshgrid')
def meshgrid(*xi, **kwargs):
  # [FUNCTION DOC]
  # Builds coordinate matrices from coordinate vectors. Inputs: variadic array-likes xi; keyword
  # options sparse (must be false), copy (must be true) and indexing (default xy). Raises
  # ValueError if sparse is true or copy is false; unknown keywords are silently ignored.
  # Returns a list of arrays.
  """This currently requires copy=True and sparse=False."""
  sparse = kwargs.get('sparse', False)
  if sparse:
    raise ValueError(
        'Function `meshgrid` does not support returning sparse arrays yet. '
        f'Received: sparse={sparse}'
    )

  copy = kwargs.get('copy', True)
  if not copy:
    raise ValueError(
        f'Function `meshgrid` only supports copy=True. Received: copy={copy}'
    )

  indexing = kwargs.get('indexing', 'xy')

  xi = [np_array_ops.asarray(arg) for arg in xi]
  kwargs = {'indexing': indexing}

  outputs = array_ops.meshgrid(*xi, **kwargs)

  return outputs


# Uses np_doc_only here because np.einsum (in 1.16) doesn't have argument
# `subscripts`, even though the doc says it has.
@tf_export.tf_export('experimental.numpy.einsum', v1=[])
@np_utils.np_doc_only('einsum')
def einsum(subscripts, *operands, **kwargs):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # KNOWN ISSUE: NumPy's no-optimization default is mapped to the greedy strategy because the
  # framework has no such option, so contraction order may differ from NumPy.
  # Einstein summation. Inputs: subscripts string, variadic operands array-likes; keyword casting
  # ("safe" promotes operands to a common type, "no" only converts them; anything else raises
  # ValueError) and optimize (false/true/greedy use greedy, optimal uses optimal; other values
  # raise ValueError). Returns the result array.
  casting = kwargs.get('casting', 'safe')
  optimize = kwargs.get('optimize', False)
  if casting == 'safe':
    operands = np_array_ops._promote_dtype(*operands)  # pylint: disable=protected-access
  elif casting == 'no':
    operands = [np_array_ops.asarray(x) for x in operands]
  else:
    raise ValueError(
        'Invalid value for argument `casting`. '
        f'Expected casting="safe" or casting="no". Received: casting={casting}'
    )
  if not optimize:
    # TF doesn't have a "no optimization" option.
    # TODO(wangpeng): Print a warning that np and tf use different
    #   optimizations.
    tf_optimize = 'greedy'
  elif optimize == True:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
    tf_optimize = 'greedy'
  elif optimize == 'greedy':
    tf_optimize = 'greedy'
  elif optimize == 'optimal':
    tf_optimize = 'optimal'
  else:
    raise ValueError(
        'Invalid value for argument `optimize`. '
        'Expected one of {True, "greedy", "optimal"}. '
        f'Received: optimize={optimize}'
    )

  res = special_math_ops.einsum(subscripts, *operands, optimize=tf_optimize)
  return res


def _tensor_t(self):
  # [FUNCTION DOC]
  # HELPER: getter installed as the transpose property on tensors; returns the transposed tensor.
  """Returns a Tensor which is the transpose of this Tensor."""
  return self.transpose()


def _tensor_ndim(self):
  # [FUNCTION DOC]
  # HELPER: getter installed as the ndim property on tensors; returns the static rank (nothing
  # if unknown).
  """Returns the rank of the Tensor."""
  return self.shape.ndims


def _tensor_pos(self):
  # [FUNCTION DOC]
  # HELPER: implements unary plus on tensors; returns the tensor itself.
  """Returns self, for unary operator `+`."""
  return self


def _tensor_size(self):
  # [FUNCTION DOC]
  # HELPER: getter installed as the size property on tensors. Returns the product of the
  # dimensions if the shape is fully known, otherwise nothing.
  """Returns the number of elements in this Tensor, if fully known."""
  if not self.shape.is_fully_defined():
    return None
  return np.prod(self.shape.as_list())


def _tensor_tolist(self):
  # [FUNCTION DOC]
  # HELPER: implements the tolist method installed on tensors. Raises ValueError for symbolic
  # (graph) tensors; otherwise returns the eager value as nested Python lists.
  if ops.is_symbolic_tensor(self):
    raise ValueError('Symbolic Tensors do not support the tolist API.')

  return self._numpy().tolist()  # pylint: disable=protected-access


def _enable_numpy_methods(tensor_class):
  # [FUNCTION DOC]
  # HELPER: installs numpy-style members on a tensor class. Side effects: mutates tensor_class by
  # adding the properties T, ndim, size and data, and the members __pos__, tolist, transpose,
  # flatten, reshape, ravel, clip, astype, __round__, max, mean and min (overwriting any existing
  # ones). Returns nothing.
  """A helper method for adding additional NumPy methods."""
  t = property(_tensor_t)
  setattr(tensor_class, 'T', t)

  ndim = property(_tensor_ndim)
  setattr(tensor_class, 'ndim', ndim)

  size = property(_tensor_size)
  setattr(tensor_class, 'size', size)

  setattr(tensor_class, '__pos__', _tensor_pos)
  setattr(tensor_class, 'tolist', _tensor_tolist)

  # TODO(b/178540516): Make a custom `setattr` that changes the method's
  #   docstring to the TF one.
  setattr(tensor_class, 'transpose', np_array_ops.transpose)
  setattr(tensor_class, 'flatten', np_array_ops.flatten)
  setattr(tensor_class, 'reshape', np_array_ops._reshape_method_wrapper)  # pylint: disable=protected-access
  setattr(tensor_class, 'ravel', np_array_ops.ravel)
  setattr(tensor_class, 'clip', clip)
  setattr(tensor_class, 'astype', math_ops.cast)
  setattr(tensor_class, '__round__', np_array_ops.around)
  setattr(tensor_class, 'max', np_array_ops.amax)
  setattr(tensor_class, 'mean', np_array_ops.mean)
  setattr(tensor_class, 'min', np_array_ops.amin)

  # TODO(wangpeng): Remove `data` when all uses of it are removed
  data = property(lambda self: self)
  setattr(tensor_class, 'data', data)


def enable_numpy_methods_on_tensor():
  # [FUNCTION DOC]
  # Public entry point that adds the numpy-style methods and properties to the framework tensor
  # class (global side effect, idempotent in effect). Takes no inputs; returns nothing.
  """Adds additional NumPy methods on tf.Tensor class."""
  _enable_numpy_methods(tensor.Tensor)
