# [MODULE DOC]
# Implementation of the public set operations (size, intersection, difference, union) over the
# last dimension of dense or sparse tensors. Builds on the generated set ops and sparse tensor
# conversion; the public sets module re-exports these. Import-time side effects: defines the
# set of supported element dtypes and marks the four set op kernels as not differentiable.
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
"""Implementation of tf.sets."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_set_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

_VALID_DTYPES = frozenset([
    dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8,
    dtypes.uint16, dtypes.string
])


@tf_export("sets.size", v1=["sets.size", "sets.set_size"])
@dispatch.add_dispatch_support
def set_size(a, validate_indices=True):
  # [FUNCTION DOC]
  # Public set size op. Input a: a sparse tensor (or convertible) with row-major sorted indices;
  # validate_indices (default true) asks the kernel to check index order and range. Raises a type
  # error if a is not a sparse tensor or its element dtype is unsupported (small integer types,
  # unsigned 8 and 16 bit, or string). Returns an int32 tensor of the number of unique values
  # along the last dimension (rank one lower than a).
  """Compute number of unique elements along last dimension of `a`.

  Args:
    a: `SparseTensor`, with indices sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
      in `a`. Note that setting this to `false` allows for undefined behavior
      when calling this function with invalid indices.

  Returns:
    `int32` `Tensor` of set sizes. For `a` ranked `n`, this is a `Tensor` with
    rank `n-1`, and the same 1st `n-1` dimensions as `a`. Each value is the
    number of unique elements in the corresponding `[0...n-1]` dimension of `a`.

  Raises:
    TypeError: If `a` is an invalid types.
  """
  a = sparse_tensor.convert_to_tensor_or_sparse_tensor(a, name="a")
  if not isinstance(a, sparse_tensor.SparseTensor):
    raise TypeError("Expected `SparseTensor`, got %s." % a)
  if a.values.dtype.base_dtype not in _VALID_DTYPES:
    raise TypeError(
        f"Invalid dtype `{a.values.dtype}` not in supported dtypes: "
        f"`{_VALID_DTYPES}`.")
  # pylint: disable=protected-access
  return gen_set_ops.set_size(a.indices, a.values, a.dense_shape,
                              validate_indices)


ops.NotDifferentiable("SetSize")

ops.NotDifferentiable("DenseToDenseSetOperation")
ops.NotDifferentiable("DenseToSparseSetOperation")
ops.NotDifferentiable("SparseToSparseSetOperation")


def _convert_to_tensors_or_sparse_tensors(a, b):
  # [FUNCTION DOC]
  # HELPER: input normalization for the binary set operations. Converts both inputs to tensors
  # or sparse tensors (named a and b). Raises a type error if a has an unsupported dtype or the
  # two dtypes differ. If a is sparse and b is dense, swaps them because only dense-first is
  # supported. Returns a triple of the (possibly swapped) first, second and a flag telling
  # whether they were swapped.
  """Convert to tensor types, and flip order if necessary.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`.
    b: `Tensor` or `SparseTensor` of the same type as `a`.

  Returns:
    Tuple of `(a, b, flipped)`, where `a` and `b` have been converted to
    `Tensor` or `SparseTensor`, and `flipped` indicates whether the order has
    been flipped to make it dense,sparse instead of sparse,dense (since the set
    ops do not support the latter).
  """
  a = sparse_tensor.convert_to_tensor_or_sparse_tensor(a, name="a")
  if a.dtype.base_dtype not in _VALID_DTYPES:
    raise TypeError(
        f"'a' has invalid dtype `{a.dtype}` not in supported dtypes: "
        f"`{_VALID_DTYPES}`.")
  b = sparse_tensor.convert_to_tensor_or_sparse_tensor(b, name="b")
  if b.dtype.base_dtype != a.dtype.base_dtype:
    raise TypeError("Types don't match, %s vs %s." % (a.dtype, b.dtype))
  if (isinstance(a, sparse_tensor.SparseTensor) and
      not isinstance(b, sparse_tensor.SparseTensor)):
    return b, a, True
  return a, b, False


def _set_operation(a, b, set_operation, validate_indices=True):
  # [FUNCTION DOC]
  # HELPER: dispatches a binary set operation to the matching generated kernel. Inputs: a and b
  # (dense or sparse), the operation name string (intersection, union, a-b or b-a), and the
  # validate_indices flag (default true). Sparse-sparse uses the sparse kernel, dense-sparse the
  # mixed kernel, dense-dense the dense kernel. Raises a value error if a is sparse and b dense.
  # Returns a sparse tensor built from the kernel's indices, values and shape.
  """Compute set operation of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
      `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be sorted
      in row-major order.
    set_operation: String indicating set operation. See
        SetOperationOp::SetOperationFromContext for valid values.
    validate_indices: Whether to validate the order and range of sparse indices
      in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the results
    of the set operation.

  Raises:
    TypeError: If inputs are invalid types.
    ValueError: If `a` is sparse and `b` is dense.
  """
  if isinstance(a, sparse_tensor.SparseTensor):
    if isinstance(b, sparse_tensor.SparseTensor):
      indices, values, shape = gen_set_ops.sparse_to_sparse_set_operation(
          a.indices, a.values, a.dense_shape, b.indices, b.values,
          b.dense_shape, set_operation, validate_indices)
    else:
      raise ValueError("Sparse,Dense is not supported, but Dense,Sparse is. "
                       "Please flip the order of your inputs.")
  elif isinstance(b, sparse_tensor.SparseTensor):
    indices, values, shape = gen_set_ops.dense_to_sparse_set_operation(
        a, b.indices, b.values, b.dense_shape, set_operation, validate_indices)
  else:
    indices, values, shape = gen_set_ops.dense_to_dense_set_operation(
        a, b, set_operation, validate_indices)
  return sparse_tensor.SparseTensor(indices, values, shape)


@tf_export(
    "sets.intersection", v1=["sets.intersection", "sets.set_intersection"])
@dispatch.add_dispatch_support
def set_intersection(a, b, validate_indices=True):
  # [FUNCTION DOC]
  # Public intersection of the sets along the last dimension of a and b. Inputs: two tensors or
  # sparse tensors of the same dtype (indices sorted if sparse), and validate_indices (default
  # true). Converts and possibly reorders inputs (type errors for bad dtypes or mismatches), then
  # runs the intersection operation. Returns a sparse tensor of the same rank. The flip does not
  # matter because intersection is symmetric.
  """Compute set intersection of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # Represent the following array of sets as a sparse tensor:
    # a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.sparse.SparseTensor(list(a.keys()), list(a.values()),
                               dense_shape=[2,2,2])

    # b = np.array([[{1}, {}], [{4}, {5, 6, 7, 8}]])
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.sparse.SparseTensor(list(b.keys()), list(b.values()),
                               dense_shape=[2, 2, 4])

    # `tf.sets.intersection` is applied to each aligned pair of sets.
    tf.sets.intersection(a, b)

    # The result will be equivalent to either of:
    #
    # np.array([[{1}, {}], [{4}, {5, 6}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 1),
    #     ((1, 0, 0), 4),
    #     ((1, 1, 0), 5),
    #     ((1, 1, 1), 6),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
      must be sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
      in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    intersections.
  """
  a, b, _ = _convert_to_tensors_or_sparse_tensors(a, b)
  return _set_operation(a, b, "intersection", validate_indices)


@tf_export("sets.difference", v1=["sets.difference", "sets.set_difference"])
@dispatch.add_dispatch_support
def set_difference(a, b, aminusb=True, validate_indices=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised when a is sparse and b is dense, but
  # the inputs are swapped by the conversion helper first, so that combination actually works
  # (the direction of the subtraction is inverted to compensate).
  # Public set difference along the last dimension. Inputs: a and b (tensors or sparse tensors
  # of equal dtype), aminusb (default true: elements of a not in b; false: elements of b not in
  # a), validate_indices (default true). Raises type errors for bad or mismatched dtypes. Returns
  # a sparse tensor of the same rank.
  """Compute set difference of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # Represent the following array of sets as a sparse tensor:
    # a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.sparse.SparseTensor(list(a.keys()), list(a.values()),
                               dense_shape=[2, 2, 2])

    # np.array([[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]])
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 3),
        ((0, 1, 0), 2),
        ((1, 0, 0), 4),
        ((1, 0, 1), 5),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.sparse.SparseTensor(list(b.keys()), list(b.values()),
                               dense_shape=[2, 2, 4])

    # `set_difference` is applied to each aligned pair of sets.
    tf.sets.difference(a, b)

    # The result will be equivalent to either of:
    #
    # np.array([[{2}, {3}], [{}, {}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 2),
    #     ((0, 1, 0), 3),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
      must be sorted in row-major order.
    aminusb: Whether to subtract `b` from `a`, vs vice versa.
    validate_indices: Whether to validate the order and range of sparse indices
      in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    differences.

  Raises:
    TypeError: If inputs are invalid types, or if `a` and `b` have
        different types.
    ValueError: If `a` is sparse and `b` is dense.
    errors_impl.InvalidArgumentError: If the shapes of `a` and `b` do not
        match in any dimension other than the last dimension.
  """
  a, b, flipped = _convert_to_tensors_or_sparse_tensors(a, b)
  if flipped:
    aminusb = not aminusb
  return _set_operation(a, b, "a-b" if aminusb else "b-a", validate_indices)


@tf_export("sets.union", v1=["sets.union", "sets.set_union"])
@dispatch.add_dispatch_support
def set_union(a, b, validate_indices=True):
  # [FUNCTION DOC]
  # Public union of the sets along the last dimension of a and b. Inputs: two tensors or sparse
  # tensors of the same dtype (indices sorted if sparse), and validate_indices (default true).
  # Converts and possibly reorders the inputs (type errors for bad dtypes or mismatches), then
  # runs the union operation. Returns a sparse tensor of the same rank.
  """Compute set union of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # [[{1, 2}, {3}], [{4}, {5, 6}]]
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.sparse.SparseTensor(list(a.keys()), list(a.values()),
                               dense_shape=[2, 2, 2])

    # [[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]]
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 3),
        ((0, 1, 0), 2),
        ((1, 0, 0), 4),
        ((1, 0, 1), 5),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.sparse.SparseTensor(list(b.keys()), list(b.values()),
                               dense_shape=[2, 2, 4])

    # `set_union` is applied to each aligned pair of sets.
    tf.sets.union(a, b)

    # The result will be a equivalent to either of:
    #
    # np.array([[{1, 2, 3}, {2, 3}], [{4, 5}, {5, 6, 7, 8}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 1),
    #     ((0, 0, 1), 2),
    #     ((0, 0, 2), 3),
    #     ((0, 1, 0), 2),
    #     ((0, 1, 1), 3),
    #     ((1, 0, 0), 4),
    #     ((1, 0, 1), 5),
    #     ((1, 1, 0), 5),
    #     ((1, 1, 1), 6),
    #     ((1, 1, 2), 7),
    #     ((1, 1, 3), 8),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
      must be sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
      in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    unions.
  """
  a, b, _ = _convert_to_tensors_or_sparse_tensors(a, b)
  return _set_operation(a, b, "union", validate_indices)
