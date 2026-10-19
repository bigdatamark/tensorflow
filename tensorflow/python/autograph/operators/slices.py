# [MODULE DOC]
# AutoGraph slice operators: functional forms of reading (get_item) and writing (set_item) an
# element by index, used by converted code in place of subscript expressions.
# Dispatches on the target type: tensor arrays (read/write ops), tensor lists (variant tensors),
# string scalar tensors (substring, read only), other tensors (indexing / scatter update) and
# plain Python objects (ordinary indexing / assignment). Extended slices are not supported.
# No import-time side effects beyond defining the options tuple type.
#
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
"""Operators specific to slicing operations."""

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops


# TODO(mdan): Support extended slices.


class GetItemOpts(collections.namedtuple('GetItemOpts', ('element_dtype',))):
  # [CLASS DOC]
  # Immutable options record for get_item with a single field, element_dtype: the data type of
  # the elements, needed when reading from a tensor list (may be null otherwise).
  # Created by the converted code and passed as the opts argument of get_item.
  pass


def get_item(target, i, opts):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says ValueError is raised for unsupported targets, but unsupported
  # types simply fall through to ordinary Python indexing; the only value error comes from reading
  # a tensor list without an element type. The docstring also says the target may be mutated,
  # which is not true for the tensor paths.
  # Staged read of element i from target. Asserts that opts is a GetItemOpts. Tensor arrays use the
  # array read op; variant (list) tensors use the list get-item op and need opts.element_dtype;
  # scalar string tensors return the one-character substring at i; other tensors are indexed;
  # everything else uses Python indexing. Returns the element.
  """The slice read operator (i.e. __getitem__).

  Note: it is unspecified whether target will be mutated or not. In general,
  if target is mutable (like Python lists), it will be mutated.

  Args:
    target: An entity that supports getitem semantics.
    i: Index to read from.
    opts: A GetItemOpts object.

  Returns:
    The read element.

  Raises:
    ValueError: if target is not of a supported type.
  """
  assert isinstance(opts, GetItemOpts)

  if isinstance(target, tensor_array_ops.TensorArray):
    return _tf_tensorarray_get_item(target, i)
  elif tensor_util.is_tf_type(target):
    if target.dtype == dtypes.variant:
      return _tf_tensor_list_get_item(target, i, opts)
    elif target.dtype == dtypes.string and target.shape.ndims == 0:
      return _tf_tensor_string_get_item(target, i)
    else:
      return _tf_tensor_get_item(target, i)
  else:
    return _py_get_item(target, i)


def _tf_tensorarray_get_item(target, i):
  # [FUNCTION DOC]
  # HELPER: tensor-array branch of get_item.
  # Returns the element at index i read from the tensor array target.
  """Overload of get_item that stages a TensorArray read."""
  return target.read(i)


def _tf_tensor_list_get_item(target, i, opts):
  # [FUNCTION DOC]
  # HELPER: tensor-list branch of get_item.
  # Reads element i of the tensor list target using the element type in opts. Raises a value error
  # if opts has no element type. Returns the element tensor.
  """Overload of get_item that stages a Tensor list read."""
  if opts.element_dtype is None:
    raise ValueError('cannot retrieve from a list without knowing its '
                     'element type; use set_element_type to annotate it')
  x = list_ops.tensor_list_get_item(target, i, element_dtype=opts.element_dtype)
  return x


def _tf_tensor_get_item(target, i):
  # [FUNCTION DOC]
  # HELPER: ordinary-tensor branch of get_item.
  # Returns the tensor indexed by i.
  """Overload of get_item that stages a Tensor (not Tensor list) read."""
  return target[i]


def _tf_tensor_string_get_item(target, i):
  # [FUNCTION DOC]
  # HELPER: scalar-string-tensor branch of get_item.
  # Returns a string tensor holding the single character of target at position i.
  """Overload of get_item that stages a Tensor string read."""
  x = gen_string_ops.substr(target, i, 1)
  return x


def _py_get_item(target, i):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of get_item.
  # KNOWN ISSUE: the docstring mentions a list modification, but this only reads an element.
  # Returns the element of target at index i using ordinary indexing.
  """Overload of get_item that executes a Python list modification."""
  return target[i]


def set_item(target, i, x):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says ValueError is raised for unsupported targets, but unsupported
  # types fall through to ordinary Python item assignment and no value error is raised here.
  # Staged write of value x at index i of target. Tensor arrays use the array write op; variant
  # (list) tensors use the list set-item op; other tensors use a scatter update; everything else
  # is mutated in place with ordinary assignment. Returns the updated target (a new tensor or tensor
  # array for tensor types, the same object for Python types).
  """The slice write operator (i.e. __setitem__).

  Note: it is unspecified whether target will be mutated or not. In general,
  if target is mutable (like Python lists), it will be mutated.

  Args:
    target: An entity that supports setitem semantics.
    i: Index to modify.
    x: The new element value.

  Returns:
    Same as target, after the update was performed.

  Raises:
    ValueError: if target is not of a supported type.
  """
  if isinstance(target, tensor_array_ops.TensorArray):
    return _tf_tensorarray_set_item(target, i, x)
  elif tensor_util.is_tf_type(target):
    if target.dtype == dtypes.variant:
      return _tf_tensor_list_set_item(target, i, x)
    else:
      return _tf_tensor_set_item(target, i, x)
  else:
    return _py_set_item(target, i, x)


def _tf_tensorarray_set_item(target, i, x):
  # [FUNCTION DOC]
  # HELPER: tensor-array branch of set_item.
  # Returns the tensor array resulting from writing x at index i of target.
  """Overload of set_item that stages a TensorArray write."""
  return target.write(i, x)


def _tf_tensor_list_set_item(target, i, x):
  # [FUNCTION DOC]
  # HELPER: tensor-list branch of set_item.
  # Returns a new tensor list equal to target with element i replaced by x.
  """Overload of set_item that stages a Tensor list update."""
  return list_ops.tensor_list_set_item(target, i, x)


def _tf_tensor_set_item(target, i, x):
  # [FUNCTION DOC]
  # HELPER: ordinary-tensor branch of set_item.
  # Returns a new tensor equal to target with the slice at index i replaced by x, built with a
  # scatter update on a single index.
  """Overload of set_item that stages a Tensor scatter update."""
  return gen_array_ops.tensor_scatter_update(target, ((i,),), (x,))


def _py_set_item(target, i, x):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of set_item.
  # Assigns x to index i of target in place (side effect on target) and returns target.
  """Overload of set_item that executes a Python list modification."""
  target[i] = x
  return target
