# [MODULE DOC]
# HELPER: implementation module behind the public identity matrix function in linalg_ops.
# Holds the graph and eager construction logic for batched identity (or rectangular
# "identity") matrices, kept separate so other modules can use it without importing the full
# linear algebra module. No import-time side effects other than imports.
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
"""Operations for linear algebra."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat

# Names below are lower_case.
# pylint: disable=invalid-name


def eye(num_rows,
        num_columns=None,
        batch_shape=None,
        dtype=dtypes.float32,
        name=None):
  # [FUNCTION DOC]
  # HELPER: implementation of the public eye function in linalg_ops.
  # Builds an identity matrix, or a batch of them, optionally rectangular. Inputs: num_rows
  # (integer or integer tensor), num_columns (optional, defaults to num_rows), batch_shape
  # (optional list, tuple or integer tensor of leading dimensions, default empty), dtype
  # (default float32), name (optional scope name, default eye).
  # Behavior: if either dimension is a tensor the diagonal length is computed dynamically as
  # the minimum; otherwise both must be Python-style integers and the diagonal length and
  # squareness are computed statically. If batch shape or diagonal size is dynamic, shapes
  # are assembled by concatenating tensors; otherwise plain lists are used. A square result is
  # produced by turning a tensor of ones into a diagonal matrix; a non-square result sets the
  # diagonal of a zero matrix to ones. Returns the resulting tensor.
  # Raises a type error when a statically known row or column count is not an integer.
  # Opens a name scope; no other side effects.
  """Construct an identity matrix, or a batch of matrices.

  See `linalg_ops.eye`.
  """
  with ops.name_scope(
      name, default_name='eye', values=[num_rows, num_columns, batch_shape]):
    is_square = num_columns is None
    batch_shape = [] if batch_shape is None else batch_shape
    num_columns = num_rows if num_columns is None else num_columns

    # We cannot statically infer what the diagonal size should be:
    if (isinstance(num_rows, tensor.Tensor) or
        isinstance(num_columns, tensor.Tensor)):
      diag_size = math_ops.minimum(num_rows, num_columns)
    else:
      # We can statically infer the diagonal size, and whether it is square.
      if not isinstance(num_rows, compat.integral_types) or not isinstance(
          num_columns, compat.integral_types):
        raise TypeError(
            'Arguments `num_rows` and `num_columns` must be positive integer '
            f'values. Received: num_rows={num_rows}, num_columns={num_columns}')
      is_square = num_rows == num_columns
      diag_size = np.minimum(num_rows, num_columns)

    # We can not statically infer the shape of the tensor.
    if isinstance(batch_shape, tensor.Tensor) or isinstance(
        diag_size, tensor.Tensor
    ):
      batch_shape = ops.convert_to_tensor(
          batch_shape, name='shape', dtype=dtypes.int32
      )
      diag_shape = array_ops.concat((batch_shape, [diag_size]), axis=0)
      if not is_square:
        shape = array_ops.concat((batch_shape, [num_rows, num_columns]), axis=0)
    # We can statically infer everything.
    else:
      batch_shape = list(batch_shape)
      diag_shape = batch_shape + [diag_size]
      if not is_square:
        shape = batch_shape + [num_rows, num_columns]

    diag_ones = array_ops.ones(diag_shape, dtype=dtype)
    if is_square:
      return array_ops.matrix_diag(diag_ones)
    else:
      zero_matrix = array_ops.zeros(shape, dtype=dtype)
      return array_ops.matrix_set_diag(zero_matrix, diag_ones)

# pylint: enable=invalid-name,redefined-builtin
