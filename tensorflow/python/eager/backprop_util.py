# [MODULE DOC]
# HELPER: shared helper module for the eager backprop (gradient tape) machinery.
# Small utilities used when recording and aggregating gradients: deciding whether a tensor or
# dtype is differentiable (trainable), and combining gradients that may be sparse slice objects.
# Imports have no side effects beyond loading dependencies; nothing is registered here.
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
"""Shared utilities related to backprop."""

from tensorflow.core.config import flags
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops


def _DTypeFromTensor(tensor):
  # [FUNCTION DOC]
  # HELPER: support routine for the trainability check.
  # Returns the effective dtype of a tensor. For ordinary dtypes this is just the tensor dtype.
  # For variant tensors it inspects the static handle data (from the eager tensor itself, or from
  # the graph resource handle data) and, if every recorded element type is known and identical,
  # returns that element type so a variant holding non-trainable data is treated as non-trainable.
  # Unknown or mixed element types fall back to the variant dtype (conservatively trainable).
  # Input: a tensor. Output: a dtype or a raw type enum value. No side effects.
  """Extract either `tensor.dtype` or the unanimous sub-type of a variant."""
  dtype = tensor.dtype
  if dtype.base_dtype == dtypes.variant:
    # If we know statically that the data a variant points to is non-trainable
    # then the variant itself is non-trainable.
    if isinstance(tensor, ops.EagerTensor):
      handle_data = tensor._handle_data  # pylint: disable=protected-access
    else:
      handle_data = handle_data_util.get_resource_handle_data(tensor)
    if (handle_data is not None
        and handle_data.is_set
        and handle_data.shape_and_type):
      first_type = handle_data.shape_and_type[0].dtype
      # Some variants have statically unknown dtypes; we can't make inferences
      # about trainability, so we conservatively assume they're trainable
      # (which may waste memory passing zeros around, but will be correct).
      if (first_type != types_pb2.DT_INVALID
          and all(shape_and_type.dtype == first_type
                  for shape_and_type in handle_data.shape_and_type)):
        return first_type
  return dtype


def IsTrainable(tensor_or_dtype):
  # [FUNCTION DOC]
  # Decides whether gradients can flow through a value of the given type.
  # Input: either a tensor-like object or a dtype (anything convertible to a dtype). Tensors are
  # first reduced to their effective dtype (looking inside variants).
  # Output: true when the base dtype is a float (half, single, double, bfloat16), complex, resource
  # or variant type; quantized integer types are also included only when the global config flag
  # for training quantized dtypes is enabled. Otherwise false. Reads configuration; no mutation.
  """Determines whether a tensor or dtype supports infinitesimal changes."""
  if tensor_util.is_tf_type(tensor_or_dtype):
    dtype = _DTypeFromTensor(tensor_or_dtype)
  else:
    dtype = tensor_or_dtype
  dtype = dtypes.as_dtype(dtype)
  trainable_dtypes = [dtypes.float16, dtypes.float32, dtypes.float64,
                      dtypes.complex64, dtypes.complex128, dtypes.resource,
                      dtypes.variant, dtypes.bfloat16]
  if flags.config().enable_quantized_dtypes_training.value():
    trainable_dtypes.extend([dtypes.qint8, dtypes.qint16, dtypes.qint32,
                             dtypes.quint8, dtypes.quint16])
  return dtype.base_dtype in trainable_dtypes


def FlattenNestedIndexedSlices(grad):
  # [FUNCTION DOC]
  # Collapses an indexed-slices gradient whose values are themselves indexed slices (nested
  # sparse gradient) into a single-level indexed-slices object.
  # Input: grad, which must be an indexed-slices object (an assertion fails otherwise).
  # Output: grad itself if its values are a dense tensor; otherwise a new indexed-slices object
  # whose indices are the outer indices gathered by the flattened inner indices, with the inner
  # values and the original dense shape. Recurses through nesting; asserts the values are either
  # a tensor or indexed slices.
  assert isinstance(grad, indexed_slices.IndexedSlices)
  if isinstance(grad.values, tensor_lib.Tensor):
    return grad
  else:
    assert isinstance(grad.values, indexed_slices.IndexedSlices)
    g = FlattenNestedIndexedSlices(grad.values)
    return indexed_slices.IndexedSlices(
        g.values, array_ops.gather(grad.indices, g.indices), g.dense_shape)


def AggregateIndexedSlicesGradients(grads):
  # [FUNCTION DOC]
  # KNOWN ISSUE: if every gradient is null (more than one), the filtered list is empty and the
  # final steps would fail on the empty list; the docstring does not mention this.
  # Sums a list of gradients for one value, where some entries may be indexed-slices.
  # Input: grads, a list of gradients (tensors, indexed slices, or nothing/null values).
  # Output: null if the list is empty; the single element unchanged if the list has exactly one
  # entry (checked before nulls are removed, so a lone null is returned as is). Otherwise null
  # entries are dropped; if any remaining gradient is a dense tensor, all are summed with an
  # add-n into a dense tensor; if all are indexed slices, they are cast to a common index type,
  # flattened, and concatenated (values and indices) into one indexed-slices object that uses the
  # first gradient's dense shape. Duplicate indices are not merged.
  """Aggregates gradients containing `IndexedSlices`s."""
  if len(grads) < 1:
    return None
  if len(grads) == 1:
    return grads[0]
  grads = [g for g in grads if g is not None]
  # If any gradient is a `Tensor`, sum them up and return a dense tensor
  # object.
  if any(isinstance(g, tensor_lib.Tensor) for g in grads):
    return math_ops.add_n(grads)

  # The following `_as_indexed_slices_list` casts ids of IndexedSlices into
  # int64. It is to make sure the inputs of `concat` all have same the data
  # type.
  grads = math_ops._as_indexed_slices_list(grads)  # pylint: disable=protected-access

  grads = [FlattenNestedIndexedSlices(x) for x in grads]
  # Form IndexedSlices out of the concatenated values and indices.
  concat_grad = indexed_slices.IndexedSlices(
      array_ops.concat([x.values for x in grads], axis=0),
      array_ops.concat([x.indices for x in grads], axis=0),
      grads[0].dense_shape)

  return concat_grad

