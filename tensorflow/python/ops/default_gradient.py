# [MODULE DOC]
# HELPER: internal helper module used by gradient code to build default (zero or one) gradients.
# Utilities for computing default gradients for tensors, including resource-handle tensors whose
# real dtype and shape are stored in handle data rather than on the tensor itself. Exposes
# functions to get the effective dtype, shape and dtype pair, zeros and ones gradients, and a
# predicate telling whether a default gradient can be created. No import-time side effects.
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
"""Utilities for computing default gradients."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops


def get_zeros_dtype(t):
  # [FUNCTION DOC]
  # HELPER: used when creating default zero gradients for tensors.
  # Returns the dtype a default gradient for tensor t should have. For ordinary tensors this is
  # the tensor dtype; for resource tensors it is the dtype recorded in the handle data.
  # Input: t, a tensor. Output: a data type. Raises a value error if t is a resource tensor
  # whose handle data is missing, unset, or does not hold exactly one shape-and-type entry.
  # No side effects.
  """Return the dtype for the default gradient for a Tensor."""
  if t.dtype == dtypes.resource:
    handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
    if (handle_data is None or not handle_data.is_set or
        len(handle_data.shape_and_type) != 1):
      raise ValueError("Internal error: Tried to take gradients (or similar) "
                       "of a variable without handle data:\n%s" % str(t))
    return handle_data.shape_and_type[0].dtype
  return t.dtype


def shape_and_dtype(t):
  # [FUNCTION DOC]
  # HELPER: used to build default gradients for resource tensors.
  # Returns a pair of shape and dtype for the default gradient of tensor t. For ordinary tensors
  # it is the tensor's own shape and dtype; for resource tensors it is the shape (as a tensor
  # shape) and dtype (as a data type) stored in the handle data.
  # Raises a value error when a resource tensor lacks valid single-entry handle data.
  # No side effects.
  """Return the shape and dtype for the default gradient for a Tensor."""
  if t.dtype == dtypes.resource:
    handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
    if (handle_data is None or not handle_data.is_set or
        len(handle_data.shape_and_type) != 1):
      raise ValueError("Internal error: Tried to take gradients (or similar) "
                       "of a variable without handle data:\n%s" % str(t))
    shape_and_type = handle_data.shape_and_type[0]
    return (tensor_shape.TensorShape(shape_and_type.shape),
            dtypes.as_dtype(shape_and_type.dtype))
  return t.shape, t.dtype


def zeros_like(t):
  # [FUNCTION DOC]
  # HELPER: zero-gradient builder that understands resource handles.
  # Returns a tensor of zeros for t. For resource tensors it creates zeros using the shape and
  # dtype from the handle data; otherwise it delegates to the standard zeros-like operation.
  # Raises a value error (via the shape lookup) for resource tensors without handle data.
  # Adds ops to the current graph or runs eagerly.
  """Like array_ops.zeros_like, but respects resource handles."""
  if t.dtype == dtypes.resource:
    return array_ops.zeros(*shape_and_dtype(t))
  else:
    return array_ops.zeros_like(t)


def ones_like(t):
  # [FUNCTION DOC]
  # HELPER: ones-gradient builder that understands resource handles.
  # Returns a tensor of ones for t. For resource tensors it creates ones using the shape and
  # dtype from the handle data; otherwise it delegates to the standard ones-like operation.
  # Raises a value error (via the shape lookup) for resource tensors without handle data.
  # Adds ops to the current graph or runs eagerly.
  """Like array_ops.ones_like, but respects resource handles."""
  if t.dtype == dtypes.resource:
    return array_ops.ones(*shape_and_dtype(t))
  else:
    return array_ops.ones_like(t)


def supports_default_grad(t):
  # [FUNCTION DOC]
  # HELPER: guard used before creating default gradients.
  # Returns true unless t is a resource tensor whose handle data is missing, unset, or does not
  # contain exactly one shape-and-type entry, in which case it returns false. Never raises for
  # that case (unlike the shape and dtype lookups). Input: a tensor of a trainable type.
  # No side effects.
  """Whether tensor `t` supports creating a default gradient.

  This function assumes that `t` is of a trainable type.

  Args:
    t: Tensor

  Returns:
    Bool
  """
  if t.dtype == dtypes.resource:
    handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
    if (handle_data is None or not handle_data.is_set or
        len(handle_data.shape_and_type) != 1):
      return False
  return True
