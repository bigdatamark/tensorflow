# [MODULE DOC]
# HELPER: shared utility module for the feature column implementation files.
# Provides sequence length computation from sparse tensors, validators for dtype and key arguments,
# and default value checking and normalization into nested tuples. No import-time side effects or
# registrations.
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
"""Defines functions common to multiple feature column files."""

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


def sequence_length_from_sparse_tensor(sp_tensor, num_elements=1):
  # [FUNCTION DOC]
  # HELPER: used by sequence feature columns to compute per-example lengths.
  # Inputs: a rank two sparse tensor (batch by time) and num_elements (default one), the number of
  # raw values that group into one timestep. Output: an int64 tensor of length batch size holding
  # for each row the maximum column index plus one, divided by num_elements and rounded up; rows
  # with no entries get zero (trailing empty rows are padded with zeros). Adds graph ops under a
  # name scope.
  """Returns a [batch_size] Tensor with per-example sequence length."""
  with ops.name_scope(None, 'sequence_length') as name_scope:
    row_ids = sp_tensor.indices[:, 0]
    column_ids = sp_tensor.indices[:, 1]
    # Add one to convert column indices to element length
    column_ids += array_ops.ones_like(column_ids)
    # Get the number of elements we will have per example/row
    seq_length = math_ops.segment_max(column_ids, segment_ids=row_ids)

    # The raw values are grouped according to num_elements;
    # how many entities will we have after grouping?
    # Example: orig tensor [[1, 2], [3]], col_ids = (0, 1, 1),
    # row_ids = (0, 0, 1), seq_length = [2, 1]. If num_elements = 2,
    # these will get grouped, and the final seq_length is [1, 1]
    seq_length = math_ops.cast(
        math_ops.ceil(seq_length / num_elements), dtypes.int64)

    # If the last n rows do not have ids, seq_length will have shape
    # [batch_size - n]. Pad the remaining values with zeros.
    n_pad = array_ops.shape(sp_tensor)[:1] - array_ops.shape(seq_length)[:1]
    padding = array_ops.zeros(n_pad, dtype=seq_length.dtype)
    return array_ops.concat([seq_length, padding], axis=0, name=name_scope)


def assert_string_or_int(dtype, prefix):
  # [FUNCTION DOC]
  # HELPER: argument validator. Inputs: a dtype and a prefix string for the message. Returns
  # nothing; raises a value error when the dtype is neither string nor integer.
  if (dtype != dtypes.string) and (not dtype.is_integer):
    raise ValueError(
        '{} dtype must be string or integer. dtype: {}.'.format(prefix, dtype))


def assert_key_is_string(key):
  # [FUNCTION DOC]
  # HELPER: argument validator. Input: a feature key. Returns nothing; raises a value error that
  # reports the type and value when the key is not a string.
  if not isinstance(key, six.string_types):
    raise ValueError(
        'key must be a string. Got: type {}. Given key: {}.'.format(
            type(key), key))


def check_default_value(shape, default_value, dtype, key):
  # [FUNCTION DOC]
  # HELPER: default value validation for numeric feature columns.
  # KNOWN ISSUE: the docstring says a value error is raised when dtype is not convertible to
  # float; no such check exists. Shape mismatches raise a value error, and incompatible values raise
  # a type error. A plain integer is accepted for any dtype without checking, and a zero-dimensional
  # numpy array converts to a scalar that is not nested, so it falls through to the type error.
  # Inputs: shape (sequence of integers), default_value (null, a number, or nested list or array),
  # dtype (target dtype), key (column name for messages). Output: null for a null default;
  # otherwise the default as a tuple nested to the given shape (scalars are broadcast to the full
  # shape). Floats are allowed only for floating dtypes; nested values must be all integers or
  # contain floats with a floating dtype. Raises a value error for shape mismatch and a type error
  # for values incompatible with the dtype.
  """Returns default value as tuple if it's valid, otherwise raises errors.

  This function verifies that `default_value` is compatible with both `shape`
  and `dtype`. If it is not compatible, it raises an error. If it is compatible,
  it casts default_value to a tuple and returns it. `key` is used only
  for error message.

  Args:
    shape: An iterable of integers specifies the shape of the `Tensor`.
    default_value: If a single value is provided, the same value will be applied
      as the default value for every item. If an iterable of values is
      provided, the shape of the `default_value` should be equal to the given
      `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    key: Column name, used only for error messages.

  Returns:
    A tuple which will be used as default value.

  Raises:
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  if default_value is None:
    return None

  if isinstance(default_value, int):
    return _create_tuple(shape, default_value)

  if isinstance(default_value, float) and dtype.is_floating:
    return _create_tuple(shape, default_value)

  if callable(getattr(default_value, 'tolist', None)):  # Handles numpy arrays
    default_value = default_value.tolist()

  if nest.is_nested(default_value):
    if not _is_shape_and_default_value_compatible(default_value, shape):
      raise ValueError(
          'The shape of default_value must be equal to given shape. '
          'default_value: {}, shape: {}, key: {}'.format(
              default_value, shape, key))
    # Check if the values in the list are all integers or are convertible to
    # floats.
    is_list_all_int = all(
        isinstance(v, int) for v in nest.flatten(default_value))
    is_list_has_float = any(
        isinstance(v, float) for v in nest.flatten(default_value))
    if is_list_all_int:
      return _as_tuple(default_value)
    if is_list_has_float and dtype.is_floating:
      return _as_tuple(default_value)
  raise TypeError('default_value must be compatible with dtype. '
                  'default_value: {}, dtype: {}, key: {}'.format(
                      default_value, dtype, key))


def _create_tuple(shape, value):
  # [FUNCTION DOC]
  # HELPER: broadcasting utility for default value checking. Inputs: a shape sequence and a
  # scalar value. Output: nested tuples following the shape, each leaf holding the value; an empty
  # shape returns the value itself.
  """Returns a tuple with given shape and filled with value."""
  if shape:
    return tuple([_create_tuple(shape[1:], value) for _ in range(shape[0])])
  return value


def _as_tuple(value):
  # [FUNCTION DOC]
  # HELPER: converts a possibly nested sequence to nested tuples recursively; a non-nested value
  # is returned unchanged.
  if not nest.is_nested(value):
    return value
  return tuple([_as_tuple(v) for v in value])


def _is_shape_and_default_value_compatible(default_value, shape):
  # [FUNCTION DOC]
  # HELPER: shape compatibility check for default values. Inputs: default value (scalar or nested
  # sequence) and shape sequence. Output: true when the nesting matches the shape at every level
  # (a scalar for an empty shape, sequences of the exact lengths otherwise), false otherwise.
  """Verifies compatibility of shape and default_value."""
  # Invalid condition:
  #  * if default_value is not a scalar and shape is empty
  #  * or if default_value is an iterable and shape is not empty
  if nest.is_nested(default_value) != bool(shape):
    return False
  if not shape:
    return True
  if len(default_value) != shape[0]:
    return False
  for i in range(shape[0]):
    if not _is_shape_and_default_value_compatible(default_value[i], shape[1:]):
      return False
  return True
