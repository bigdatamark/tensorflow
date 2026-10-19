# [MODULE DOC]
# HELPER: helper module with math utilities for kernelized Keras layers and losses.
# Exposes pairwise inner products and exact Gaussian and Laplacian kernel computations over
# tensors of rows, built on TensorFlow array and math ops, plus private shape-alignment helpers.
# No import-time side effects other than importing ops modules.
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
"""Utility methods related to kernelized layers."""

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _to_matrix(u):
  # [FUNCTION DOC]
  # HELPER: shape normalizer for the kernel functions in this module.
  # Input: u, a tensor of rank 1 or 2. A rank-1 tensor is turned into a matrix with a single row;
  # a rank-2 tensor is returned unchanged. Raises a value error for any other rank.
  """If input tensor is a vector (i.e., has rank 1), converts it to matrix."""
  u_rank = len(u.shape)
  if u_rank not in [1, 2]:
    raise ValueError('The input tensor should have rank 1 or 2. Given rank: {}'
                     .format(u_rank))
  if u_rank == 1:
    return array_ops.expand_dims(u, 0)
  return u


def _align_matrices(x, y):
  # [FUNCTION DOC]
  # HELPER: prepares operands for pairwise row computations in the kernel functions.
  # Inputs: x and y, tensors of rank 1 or 2 (vectors are promoted to single-row matrices).
  # Returns two rank-3 tensors of shape (rows of x, rows of y, columns): x repeated along the
  # second axis and y repeated along the first, so that elementwise ops compare every row of x
  # with every row of y. Raises a value error if the column counts differ or the ranks are bad.
  """Aligns x and y tensors to allow computations over pairs of their rows."""
  x_matrix = _to_matrix(x)
  y_matrix = _to_matrix(y)
  x_shape = x_matrix.shape
  y_shape = y_matrix.shape
  if y_shape[1] != x_shape[1]:  # dimensions do not match.
    raise ValueError(
        'The outermost dimensions of the input tensors should match. Given: {} '
        'vs {}.'.format(y_shape[1], x_shape[1]))

  x_tile = array_ops.tile(
      array_ops.expand_dims(x_matrix, 1), [1, y_shape[0], 1])
  y_tile = array_ops.tile(
      array_ops.expand_dims(y_matrix, 0), [x_shape[0], 1, 1])
  return x_tile, y_tile


def inner_product(u, v):
  # [FUNCTION DOC]
  # Computes inner products between all pairs of rows of u and v.
  # Inputs: u and v, tensors of rank 1 or 2 (vectors become single-row matrices) with equal
  # column counts. Returns a matrix of shape (rows of u, rows of v) from a matrix product with
  # v transposed. Raises a value error for bad ranks; a mismatch of columns fails in the matrix
  # product op. No side effects.
  u = _to_matrix(u)
  v = _to_matrix(v)
  return math_ops.matmul(u, v, transpose_b=True)


def exact_gaussian_kernel(x, y, stddev):
  # [FUNCTION DOC]
  # Computes the exact Gaussian (RBF) kernel between all row pairs of x and y.
  # Inputs: x and y, tensors of rank 1 or 2 with the same number of columns; stddev, the kernel
  # width (number). Returns a matrix of shape (rows of x, rows of y) holding the exponential of
  # the negative squared Euclidean distance divided by twice the squared stddev (shape (1, 1)
  # for two vectors). Raises a value error if shapes are incompatible. No side effects.
  r"""Computes exact Gaussian kernel value(s) for tensors x and y and stddev.

  The Gaussian kernel for vectors u, v is defined as follows:
       K(u, v) = exp(-||u-v||^2 / (2* stddev^2))
  where the norm is the l2-norm. x, y can be either vectors or matrices. If they
  are vectors, they must have the same dimension. If they are matrices, they
  must have the same number of columns. In the latter case, the method returns
  (as a matrix) K(u, v) values for all pairs (u, v) where u is a row from x and
  v is a row from y.

  Args:
    x: a tensor of rank 1 or 2. It's shape should be either [dim] or [m, dim].
    y: a tensor of rank 1 or 2. It's shape should be either [dim] or [n, dim].
    stddev: The width of the Gaussian kernel.

  Returns:
    A single value (scalar) with shape (1, 1) (if x, y are vectors) or a matrix
      of shape (m, n) with entries K(u, v) (where K is the Gaussian kernel) for
      all (u,v) pairs where u, v are rows from x and y respectively.

  Raises:
    ValueError: if the shapes of x, y are not compatible.
  """
  x_aligned, y_aligned = _align_matrices(x, y)
  diff_squared_l2_norm = math_ops.reduce_sum(
      math_ops.squared_difference(x_aligned, y_aligned), 2)
  return math_ops.exp(-diff_squared_l2_norm / (2 * stddev * stddev))


def exact_laplacian_kernel(x, y, stddev):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring describes stddev as the width of the Gaussian kernel, but this is the
  # Laplacian kernel; the code uses the l1 distance (sum of absolute differences) divided by stddev.
  # Computes the exact Laplacian kernel between all row pairs of x and y.
  # Inputs: x and y, tensors of rank 1 or 2 with equal column counts; stddev, the width (number).
  # Returns a matrix of shape (rows of x, rows of y) holding the exponential of the negative l1
  # distance divided by stddev. Raises a value error if shapes are incompatible. No side effects.
  r"""Computes exact Laplacian kernel value(s) for tensors x and y using stddev.

  The Laplacian kernel for vectors u, v is defined as follows:
       K(u, v) = exp(-||u-v|| / stddev)
  where the norm is the l1-norm. x, y can be either vectors or matrices. If they
  are vectors, they must have the same dimension. If they are matrices, they
  must have the same number of columns. In the latter case, the method returns
  (as a matrix) K(u, v) values for all pairs (u, v) where u is a row from x and
  v is a row from y.

  Args:
    x: a tensor of rank 1 or 2. It's shape should be either [dim] or [m, dim].
    y: a tensor of rank 1 or 2. It's shape should be either [dim] or [n, dim].
    stddev: The width of the Gaussian kernel.

  Returns:
    A single value (scalar) with shape (1, 1)  if x, y are vectors or a matrix
    of shape (m, n) with entries K(u, v) (where K is the Laplacian kernel) for
    all (u,v) pairs where u, v are rows from x and y respectively.

  Raises:
    ValueError: if the shapes of x, y are not compatible.
  """
  x_aligned, y_aligned = _align_matrices(x, y)
  diff_l1_norm = math_ops.reduce_sum(
      math_ops.abs(math_ops.subtract(x_aligned, y_aligned)), 2)
  return math_ops.exp(-diff_l1_norm / stddev)
