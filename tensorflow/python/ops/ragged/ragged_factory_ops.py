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
# [MODULE DOC]
# Factory operations for building ragged values from nested Python lists: ragged.constant (a
# RaggedTensor or dense tensor), the v1-only ragged.constant_value (a RaggedTensorValue or numpy
# array) and the v1-only ragged.placeholder. Both constant functions share a private
# implementation parameterized by factory callbacks, plus private helpers that measure nesting
# depth and infer inner shapes. Import-time side effect: public API export registration and
# dispatch support on the decorated functions.
"""Operations for constructing RaggedTensors."""

from typing import Union

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import dispatch
from tensorflow.python.util.numpy_compat import np_reshape
from tensorflow.python.util.tf_export import tf_export


# ===============================================================================
# Op to construct a constant RaggedTensor from a nested Python list.
# ===============================================================================
@tf_export("ragged.constant")
@dispatch.add_dispatch_support
def constant(
    pylist,
    dtype=None,
    ragged_rank=None,
    inner_shape=None,
    name=None,
    row_splits_dtype=dtypes.int64,
) -> Union[ragged_tensor.RaggedTensor, ops._EagerTensorBase, ops.Operation]:
  # [FUNCTION DOC]
  # Public ragged.constant. Inputs: pylist (nested list, tuple or numpy array whose leaves are
  # scalars), dtype (optional element type, inferred when omitted), ragged_rank (optional
  # nonnegative int), inner_shape (optional tuple of ints for dense inner values), name (optional
  # scope name), row_splits_dtype (int32 or int64, default int64).
  # Defines a local factory that turns row split lists into constant tensors of the requested
  # type and wraps values with them, then inside a name scope calls the shared constant builder
  # with the standard tensor constant function for leaves. Returns a RaggedTensor, or a plain
  # tensor when the result has no ragged dimension. Raises a value error for inconsistent
  # nesting or incompatible ragged_rank / inner_shape, and a type error if pylist is already
  # ragged (raised by the shared builder). Creates constant ops in the current graph/eager context.
  """Constructs a constant RaggedTensor from a nested Python list.

  Example:

  >>> tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
  <tf.RaggedTensor [[1, 2], [3], [4, 5, 6]]>

  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensor` will have rank `K`.  If `pylist` contains no scalar
  values, then `K` is one greater than the maximum depth of empty lists in
  `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.

  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list`, `tuple` or `np.ndarray` must be a scalar value compatible
      with `dtype`.
    dtype: The type of elements for the returned `RaggedTensor`.  If not
      specified, then a default is chosen based on the scalar values in
      `pylist`. If there are no scalar values in `pylist`, then the default is
      `tf.float32`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensor`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K
      - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensor`.  Defaults to `()` if `ragged_rank`
      is not specified.  If `ragged_rank` is specified, then a default is chosen
      based on the contents of `pylist`.
    name: A name prefix for the returned tensor (optional).
    row_splits_dtype: data type for the constructed `RaggedTensor`'s row_splits.
      One of `tf.int32` or `tf.int64`.

  Returns:
    A potentially ragged tensor with rank `K` and the specified `ragged_rank`,
    containing the values from `pylist`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  def ragged_factory(values, row_splits):
    # [FUNCTION DOC]
    # HELPER: callback passed to the shared constant builder by constant.
    # Converts a list of row split offsets into a constant tensor of row_splits_dtype and returns
    # a RaggedTensor made from the given values and those splits, without validation.
    row_splits = constant_op.constant(row_splits, dtype=row_splits_dtype)
    return ragged_tensor.RaggedTensor.from_row_splits(values, row_splits,
                                                      validate=False)

  with ops.name_scope(name, "RaggedConstant"):
    return _constant_value(ragged_factory, constant_op.constant, pylist, dtype,
                           ragged_rank, inner_shape)


@tf_export(v1=["ragged.constant_value"])
@dispatch.add_dispatch_support
def constant_value(
    pylist,
    dtype=None,
    ragged_rank=None,
    inner_shape=None,
    row_splits_dtype="int64",
) -> Union[ragged_tensor_value.RaggedTensorValue, np.ndarray]:
  # [FUNCTION DOC]
  # Public v1 ragged.constant_value. Inputs: pylist (nested list, tuple or numpy array), dtype
  # (optional; a TensorFlow dtype is converted to its numpy type), ragged_rank and inner_shape
  # (optional, as for constant) and row_splits_dtype (name or type, default int64).
  # Builds a numpy-backed result by calling the shared constant builder with numpy-based factory
  # callbacks. Returns a RaggedTensorValue, or a numpy array when no ragged dimension results.
  # Raises a value error for inconsistent nesting or incompatible ragged_rank / inner_shape.
  # No graph ops are created.
  """Constructs a RaggedTensorValue from a nested Python list.

  Warning: This function returns a `RaggedTensorValue`, not a `RaggedTensor`.
  If you wish to construct a constant `RaggedTensor`, use
  [`ragged.constant(...)`](constant.md) instead.

  Example:

  >>> tf.compat.v1.ragged.constant_value([[1, 2], [3], [4, 5, 6]])
  tf.RaggedTensorValue(values=array([1, 2, 3, 4, 5, 6]),
                       row_splits=array([0, 2, 3, 6]))

  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensorValue` will have rank `K`.  If `pylist` contains no
  scalar values, then `K` is one greater than the maximum depth of empty lists
  in `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.

  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list` or `tuple` must be a scalar value compatible with `dtype`.
    dtype: `numpy.dtype`.  The type of elements for the returned `RaggedTensor`.
      If not specified, then a default is chosen based on the scalar values in
      `pylist`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensorValue`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K
      - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensorValue`.  Defaults to `()` if
      `ragged_rank` is not specified.  If `ragged_rank` is specified, then a
      default is chosen based on the contents of `pylist`.
    row_splits_dtype: data type for the constructed `RaggedTensorValue`'s
      row_splits.  One of `numpy.int32` or `numpy.int64`.

  Returns:
    A `tf.RaggedTensorValue` or `numpy.array` with rank `K` and the specified
    `ragged_rank`, containing the values from `pylist`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  if dtype is not None and isinstance(dtype, dtypes.DType):
    dtype = dtype.as_numpy_dtype
  row_splits_dtype = dtypes.as_dtype(row_splits_dtype).as_numpy_dtype
  def _ragged_factory(values, row_splits):
    # [FUNCTION DOC]
    # HELPER: callback passed to the shared constant builder by constant_value.
    # Converts the row split list to a numpy array of the requested integer type and returns a
    # RaggedTensorValue wrapping the values and splits.
    row_splits = np.array(row_splits, dtype=row_splits_dtype)
    return ragged_tensor_value.RaggedTensorValue(values, row_splits)

  def _inner_factory(pylist, dtype, shape, name=None):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # HELPER: callback for leaf values, passed to the shared constant builder by constant_value.
    # Converts the nested list to a numpy array (cast to dtype when a concrete numeric dtype is
    # given; kept as-is for object or unspecified dtype) and reshapes it to the given shape.
    # The name argument is accepted for interface compatibility and ignored. Returns the array.
    if dtype is object or dtype is None:
      return np_reshape(np.array(pylist, dtype=dtype), shape)
    else:
      return np_reshape(np.array(pylist).astype(dtype), shape)

  return _constant_value(
      _ragged_factory, _inner_factory, pylist, dtype, ragged_rank, inner_shape
  )


def _constant_value(
    ragged_factory, inner_factory, pylist, dtype, ragged_rank, inner_shape
):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind constant and constant_value.
  # Inputs: ragged_factory (callback building a ragged value from values and row splits),
  # inner_factory (callback building a leaf tensor/array from a flat list, dtype, shape and name),
  # pylist (nested list, tuple or array), dtype, ragged_rank and inner_shape (each may be nothing).
  # Raises a type error if pylist is already ragged; a value error for negative ragged_rank,
  # empty-list nesting deeper than scalar nesting, max depth below ragged_rank, or ragged_rank
  # and inner_shape inconsistent with the depth. A scalar input is returned through the inner
  # factory (it must have no ragged_rank or inner_shape). If the result needs no ragged
  # dimension, the whole list goes to the inner factory. Otherwise defaults for inner_shape and
  # ragged_rank are inferred, the list is flattened one ragged level at a time recording row
  # split offsets, the flat values are built, and the ragged levels are wrapped innermost first.
  # Returns the value produced by the factories.
  """Constructs a constant RaggedTensor or RaggedTensorValue.

  Args:
    ragged_factory: A factory function with the signature:
      `ragged_factory(values, row_splits)`
    inner_factory: A factory function with the signature: `inner_factory(pylist,
      dtype, shape, name)`
    pylist: A nested `list`, `tuple` or `np.ndarray`.
    dtype: Data type for returned value.
    ragged_rank: Ragged rank for returned value.
    inner_shape: Inner value shape for returned value.

  Returns:
    A value returned by `ragged_factory` or `inner_factory`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  if ragged_tensor.is_ragged(pylist):
    raise TypeError("pylist may not be a RaggedTensor or RaggedTensorValue.")
  # np.ndim builds an array, so we short-circuit lists and tuples.
  if not isinstance(pylist, (list, tuple)) and np.ndim(pylist) == 0:
    # Scalar value
    if ragged_rank is not None and ragged_rank != 0:
      raise ValueError("Invalid pylist=%r: incompatible with ragged_rank=%d" %
                       (pylist, ragged_rank))
    if inner_shape is not None and inner_shape:
      raise ValueError(
          "Invalid pylist=%r: incompatible with dim(inner_shape)=%d" %
          (pylist, len(inner_shape)))
    return inner_factory(pylist, dtype, ())

  if ragged_rank is not None and ragged_rank < 0:
    raise ValueError(
        "Invalid ragged_rank=%r: must be nonnegative" % ragged_rank)

  # Find the depth of scalar values in `pylist`.
  scalar_depth, max_depth = _find_scalar_and_max_depth(pylist)
  if scalar_depth is not None:
    if max_depth > scalar_depth:
      raise ValueError("Invalid pylist=%r: empty list nesting is greater "
                       "than scalar value nesting" % pylist)
    if ragged_rank is not None and max_depth < ragged_rank:
      raise ValueError(f"Invalid pylist={pylist}, max depth smaller than "
                       f"ragged_rank={ragged_rank}")

  # If both inner_shape and ragged_rank were specified, then check that
  # they are compatible with pylist.
  if inner_shape is not None and ragged_rank is not None:
    expected_depth = ragged_rank + len(inner_shape) + 1
    if ((scalar_depth is not None and expected_depth != scalar_depth) or
        (scalar_depth is None and expected_depth < max_depth)):
      raise ValueError(
          "Invalid pylist=%r: incompatible with ragged_rank=%d "
          "and dim(inner_shape)=%d" % (pylist, ragged_rank, len(inner_shape)))

  # Check if the result is a `Tensor`.
  if (ragged_rank == 0 or
      (ragged_rank is None and
       ((max_depth < 2) or
        (inner_shape is not None and max_depth - len(inner_shape) < 2)))):
    return inner_factory(pylist, dtype, inner_shape)

  # Compute default value for inner_shape.
  if inner_shape is None:
    if ragged_rank is None:
      inner_shape = ()
    else:
      inner_shape = _default_inner_shape_for_pylist(pylist, ragged_rank)

  # Compute default value for ragged_rank.
  if ragged_rank is None:
    if scalar_depth is None:
      ragged_rank = max(1, max_depth - 1)
    else:
      ragged_rank = max(1, scalar_depth - 1 - len(inner_shape))

  # Build the splits for each ragged rank, and concatenate the inner values
  # into a single list.
  nested_splits = []
  values = pylist
  for dim in range(ragged_rank):
    nested_splits.append([0])
    concatenated_values = []
    for row in values:
      nested_splits[dim].append(nested_splits[dim][-1] + len(row))
      concatenated_values.extend(row)
    values = concatenated_values

  values = inner_factory(
      values, dtype=dtype, shape=(len(values),) + inner_shape, name="values")
  for row_splits in reversed(nested_splits):
    values = ragged_factory(values, row_splits)
  return values


def _find_scalar_and_max_depth(pylist):
  # [FUNCTION DOC]
  # HELPER: depth analysis for the shared constant builder.
  # Recursively walks a nested list/tuple/array. Returns a pair: the nesting depth of scalar
  # leaves (nothing if there are no scalars) and the maximum depth including empty lists (a scalar
  # has depths zero and zero). Raises a value error when scalar leaves occur at different depths.
  """Finds nesting depth of scalar values in pylist.

  Args:
    pylist: A nested python `list` or `tuple`.

  Returns:
    A tuple `(scalar_depth, max_depth)`.  `scalar_depth` is the nesting
    depth of scalar values in `pylist`, or `None` if `pylist` contains no
    scalars.  `max_depth` is the maximum depth of `pylist` (including
    empty lists).

  Raises:
    ValueError: If pylist has inconsistent nesting depths for scalars.
  """
  # Check if pylist is not scalar. np.ndim builds an array, so we
  # short-circuit lists and tuples.
  if isinstance(pylist, (list, tuple)) or np.ndim(pylist) != 0:
    scalar_depth = None
    max_depth = 1
    for child in pylist:
      child_scalar_depth, child_max_depth = _find_scalar_and_max_depth(child)
      if child_scalar_depth is not None:
        if scalar_depth is not None and scalar_depth != child_scalar_depth + 1:
          raise ValueError("all scalar values must have the same nesting depth")
        scalar_depth = child_scalar_depth + 1
      max_depth = max(max_depth, child_max_depth + 1)
    return (scalar_depth, max_depth)
  return (0, 0)


def _default_inner_shape_for_pylist(pylist, ragged_rank):
  # [FUNCTION DOC]
  # HELPER: infers the dense inner shape for the shared constant builder when ragged_rank is given
  # but inner_shape is not.
  # Flattens the first ragged_rank nesting levels of pylist, then takes the shape of the leftmost
  # inner item and verifies every other item matches. Returns the inner shape as a tuple (without
  # the leading flat-length dimension). Raises a value error if some element at a ragged level is
  # a scalar (scalar depth too small for ragged_rank) or if the inner values have inconsistent
  # shapes.
  """Computes a default inner shape for the given python list."""

  def get_inner_shape(item):
    # [FUNCTION DOC]
    # HELPER: shape probe for the default inner shape computation.
    # Recursively follows the first element of nested lists. Returns a tuple of lengths: empty
    # for a scalar, and a single zero for an empty list.
    """Returns the inner shape for a python list `item`."""
    if not isinstance(item, (list, tuple)) and np.ndim(item) == 0:
      return ()
    # Note that we need this check here in case `item` is not a Python list but
    # fakes as being one (pylist). For a scenario of this, see test added in
    # https://github.com/tensorflow/tensorflow/pull/48945
    elif len(item) > 0:  # pylint: disable=g-explicit-length-test
      return (len(item),) + get_inner_shape(item[0])
    return (0,)

  def check_inner_shape(item, shape):
    # [FUNCTION DOC]
    # HELPER: validator for the default inner shape computation.
    # Recursively checks that item is nested exactly as deeply as shape says and that each level
    # has the expected length. Raises a value error ("inner values have inconsistent shape")
    # otherwise. Returns nothing.
    """Checks that `item` has a consistent shape matching `shape`."""
    is_nested = isinstance(item, (list, tuple)) or np.ndim(item) != 0
    if is_nested != bool(shape):
      raise ValueError("inner values have inconsistent shape")
    if is_nested:
      if shape[0] != len(item):
        raise ValueError("inner values have inconsistent shape")
      for child in item:
        check_inner_shape(child, shape[1:])

  # Collapse the ragged layers to get the list of inner values.
  flat_values = pylist
  for dim in range(ragged_rank):
    if not all(
        isinstance(v, (list, tuple)) or np.ndim(v) != 0 for v in flat_values):
      raise ValueError("pylist has scalar values depth %d, but ragged_rank=%d "
                       "requires scalar value depth greater than %d" %
                       (dim + 1, ragged_rank, ragged_rank))
    flat_values = sum((list(v) for v in flat_values), [])

  # Compute the inner shape looking only at the leftmost elements; and then
  # use check_inner_shape to verify that other elements have the same shape.
  inner_shape = get_inner_shape(flat_values)
  check_inner_shape(flat_values, inner_shape)
  return inner_shape[1:]


@tf_export(v1=["ragged.placeholder"])
@dispatch.add_dispatch_support
def placeholder(dtype, ragged_rank, value_shape=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises a runtime error when eager execution is enabled,
  # but this function has no explicit check; failure would come from the placeholder op itself.
  # KNOWN ISSUE: value_shape defaults to nothing, which is only fine for ragged_rank zero or when
  # concatenation with an unknown shape is acceptable; the resulting flat shape is then unknown.
  # Creates a v1 graph placeholder for a ragged tensor. Inputs: dtype of the values, ragged_rank
  # (number of ragged dimensions), value_shape (shape of each flat value, optional), name
  # (optional). For ragged_rank zero returns a plain placeholder with value_shape. Otherwise, in
  # a name scope, creates a flat-values placeholder with an unknown leading dimension followed by
  # value_shape, and one int64 row_splits placeholder per ragged level, wrapping them (innermost
  # first) into a RaggedTensor without validation. Returns the ragged tensor handle, which must
  # be fed rather than evaluated. Adds placeholder ops to the current graph.
  """Creates a placeholder for a `tf.RaggedTensor` that will always be fed.

  **Important**: This ragged tensor will produce an error if evaluated.
  Its value must be fed using the `feed_dict` optional argument to
  `Session.run()`, `Tensor.eval()`, or `Operation.run()`.


  Args:
    dtype: The data type for the `RaggedTensor`.
    ragged_rank: The ragged rank for the `RaggedTensor`
    value_shape: The shape for individual flat values in the `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor` that may be used as a handle for feeding a value, but
    not evaluated directly.

  Raises:
    RuntimeError: if eager execution is enabled

  @compatibility(TF2)
  This API is not compatible with eager execution and `tf.function`. To migrate
  to TF2, rewrite the code to be compatible with eager execution. Check the
  [migration
  guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
  on replacing `Session.run` calls. In TF2, you can just pass tensors directly
  into ops and layers. If you want to explicitly set up your inputs, also see
  [Keras functional API](https://www.tensorflow.org/guide/keras/functional) on
  how to use `tf.keras.Input` to replace `tf.compat.v1.ragged.placeholder`.
  `tf.function` arguments also do the job of `tf.compat.v1.ragged.placeholder`.
  For more details please read [Better
  performance with tf.function](https://www.tensorflow.org/guide/function).
  @end_compatibility
  """
  if ragged_rank == 0:
    return array_ops.placeholder(dtype, value_shape, name)

  with ops.name_scope(name, "RaggedPlaceholder", []):
    flat_shape = tensor_shape.TensorShape([None]).concatenate(value_shape)
    result = array_ops.placeholder(dtype, flat_shape, "flat_values")
    for i in reversed(range(ragged_rank)):
      row_splits = array_ops.placeholder(dtypes.int64, [None],
                                         "row_splits_%d" % i)
      result = ragged_tensor.RaggedTensor.from_row_splits(result, row_splits,
                                                          validate=False)
    return result
