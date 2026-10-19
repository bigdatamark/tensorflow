# [MODULE DOC]
# Typed-list support for AutoGraph. Provides dynamic_list_append, which appends an element to a
# Python list, a TensorFlow tensor list or a tensor array, and TensorList, a thin wrapper around
# a TensorFlow tensor list that mimics part of the Python list interface. dynamic_list_append is
# re-exported by the utils package. No import-time side effects.
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
"""A typed list in Python."""

from tensorflow.python.framework import tensor
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops


def dynamic_list_append(target, element):
  # [FUNCTION DOC]
  # Appends element to target and returns the resulting list-like object. If target is a tensor
  # array, returns a new array with the element written at the current size; if target is a
  # tensor (a tensor list handle), returns a new tensor list with the element pushed to the back;
  # otherwise calls the append method of target (mutating it) and returns the same object.
  """Converts a list append call inline."""
  if isinstance(target, tensor_array_ops.TensorArray):
    return target.write(target.size(), element)
  # TODO(mdan): What's the right way to check this?
  # TODO(mdan): We may not need this branch.
  # It may be possible to use TensorList alone if the loop body will not
  # require wrapping it, although we'd have to think about an autoboxing
  # mechanism for lists received as parameter.
  if isinstance(target, tensor.Tensor):
    return list_ops.tensor_list_push_back(target, element)

  # Python targets (including TensorList): fallback to their original append.
  target.append(element)
  return target


class TensorList(object):
  # [CLASS DOC]
  # Wrapper around a TensorFlow tensor list with a Python-list-like API (append, pop, clear,
  # count, indexing and index assignment). State: dtype and shape (element dtype and shape given
  # at construction) and list_ (the current underlying tensor list handle, replaced on each
  # mutation because tensor list operations are functional). Created with an element shape and dtype.
  """Tensor list wrapper API-compatible with Python built-in list."""

  def __init__(self, shape, dtype):
    # [FUNCTION DOC]
    # Stores the element shape and dtype, then creates an empty underlying tensor list via clear.
    self.dtype = dtype
    self.shape = shape
    self.clear()

  def append(self, value):
    # [FUNCTION DOC]
    # Pushes value onto the back of the underlying tensor list, replacing the stored handle.
    # Returns nothing.
    self.list_ = list_ops.tensor_list_push_back(self.list_, value)

  def pop(self):
    # [FUNCTION DOC]
    # Removes the last element of the underlying tensor list, replacing the stored handle, and
    # returns the removed element as a tensor of the list's dtype.
    self.list_, value = list_ops.tensor_list_pop_back(self.list_, self.dtype)
    return value

  def clear(self):
    # [FUNCTION DOC]
    # Replaces the underlying list with a new empty tensor list of the stored shape and dtype.
    # Returns nothing.
    self.list_ = list_ops.empty_tensor_list(self.shape, self.dtype)

  def count(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests the Python list method counting occurrences of a value, but
    # this takes no argument and returns the length of the list, as a scalar tensor.
    return list_ops.tensor_list_length(self.list_)

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Returns the element at index key (integer or integer tensor) as a tensor of the stored dtype.
    return list_ops.tensor_list_get_item(self.list_, key, self.dtype)

  def __setitem__(self, key, value):
    # [FUNCTION DOC]
    # Replaces the element at index key (integer or integer tensor) with value by creating an
    # updated tensor list and storing it. Returns nothing.
    self.list_ = list_ops.tensor_list_set_item(self.list_, key, value)
