# [MODULE DOC]
# HELPER: Helper module for StructuredTensor, which builds the shape object used to describe it.
# KNOWN ISSUE: the main structured tensor module defines its own function of the same name with
# richer behavior and does not use this one; this copy is only referenced by tests.
# Provides a single private function that builds a dynamic ragged shape from the pieces a
# structured tensor holds (field mapping, static shape, row count and row partitions). It is
# kept in its own module and imports a private dtype-finding helper from the structured tensor
# module. No registrations or global state are created at import time.
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
"""Dynamic shape for structured Tensors."""

from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.structured.structured_tensor import _find_shape_dtype


# pylint:disable=protected-access
def _dynamic_ragged_shape_init(fields, shape, nrows, row_partitions):
  # [FUNCTION DOC]
  # HELPER: Internal factory used when constructing a StructuredTensor's ragged shape.
  # KNOWN ISSUE: duplicate of a same-named function in the main module; unlike that one it does
  # not merge in shapes inferred from fields and requires nrows to be a tensor (not an integer).
  # Builds a dynamic ragged shape for a structured tensor. Inputs: fields (a dict of field
  # values), shape (a static tensor shape that must have known rank), nrows (nothing or a
  # scalar tensor with the outer row count), row_partitions (a tuple of row partitions). The
  # inputs are checked with assertions, so wrong types raise assertion errors. A shape with
  # unknown rank raises a type error. The index dtype is chosen from the fields, nrows and
  # row partitions. Rank 0 gives a shape with an empty inner shape; rank 1 uses the static
  # first dimension when known (else nrows) as the single inner dimension; higher ranks are
  # built from the row partitions. Returns the new shape object; no side effects.
  """Produce a DynamicRaggedShape for StructuredTensor."""
  assert isinstance(fields, dict), fields
  assert isinstance(shape, tensor_shape.TensorShape), shape
  assert nrows is None or isinstance(nrows, tensor.Tensor), nrows
  assert isinstance(row_partitions, tuple), row_partitions

  rank = shape.rank
  if rank is None:
    raise TypeError("StructuredTensor's shape must have known rank.")

  # TODO(martinz): figure out whether to validate.
  dtype = _find_shape_dtype(fields, nrows, row_partitions)
  if rank == 0:
    return dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(
        array_ops.zeros((0,), dtype=dtype))

  if rank == 1:
    alt_value = shape[0]
    if isinstance(alt_value, tensor_shape.Dimension):
      alt_value = alt_value.value
    if alt_value is not None:
      nrows = alt_value
    return dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(
        [nrows], dtype=dtype)

  return dynamic_ragged_shape.DynamicRaggedShape.from_row_partitions(
      row_partitions, dtype=dtype)
