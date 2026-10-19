# [MODULE DOC]
# HELPER: helper module with sparse tensor utilities for datasets. Provides class detection
# (any_sparse, get_classes), conversion of shapes and types to their physical dense/variant form
# (as_dense_shapes, as_dense_types), and serialization/deserialization of sparse tensors inside
# nested structures (single and batched). Used by dataset code that must carry sparse tensors
# as variant tensors. No import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Python dataset sparse tensor utility functions."""
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import sparse_ops


def any_sparse(classes):
  # [FUNCTION DOC]
  # Returns true if any element of the nested structure of item classes is the sparse tensor
  # class, false otherwise (including for an empty structure). No side effects.
  """Checks for sparse tensor.

  Args:
    classes: a structure of objects that identify the dataset item classes

  Returns:
    `True` if `classes` contains a sparse tensor type and `False` otherwise.
  """
  return any(c is sparse_tensor.SparseTensor for c in nest.flatten(classes))


def as_dense_shapes(shapes, classes):
  # [FUNCTION DOC]
  # Maps element shapes to their physical shapes: where classes marks a sparse tensor the shape
  # becomes a fully unknown shape, otherwise the shape is kept. shapes and classes are parallel
  # nested structures. Returns a structure with the nesting of shapes.
  """Converts sparse tensor shapes to their physical shapes.

  Args:
    shapes: a structure of shapes to convert.
    classes: a structure of objects that identify the dataset item classes

  Returns:
    a structure matching the nested structure of `shapes`, containing
    `tensor_shape.unknown_shape()` at positions where `classes` contains
    `tf.sparse.SparseTensor` and matching contents of `shapes` otherwise
  """
  ret = nest.pack_sequence_as(shapes, [
      tensor_shape.unknown_shape() if c is sparse_tensor.SparseTensor else shape
      for shape, c in zip(nest.flatten(shapes), nest.flatten(classes))
  ])
  return ret


def as_dense_types(types, classes):
  # [FUNCTION DOC]
  # Maps element types to their physical types: where classes marks a sparse tensor the type
  # becomes the variant type (serialized form), otherwise the type is kept. types and classes
  # are parallel nested structures. Returns a structure with the nesting of types.
  """Converts sparse tensor types to `dtypes.variant`.

  Args:
    types: a structure of types to convert.
    classes: a structure of objects that identify the dataset item classes

  Returns:
    a structure matching the nested structure of `types`, containing
    `dtypes.variant` at positions where `classes` contains
    `tf.sparse.SparseTensor` and matching contents of `types` otherwise
  """
  ret = nest.pack_sequence_as(types, [
      dtypes.variant if c is sparse_tensor.SparseTensor else ty
      for ty, c in zip(nest.flatten(types), nest.flatten(classes))
  ])
  return ret


def deserialize_sparse_tensors(tensors, types, shapes, classes):
  # [FUNCTION DOC]
  # Reverses serialization of sparse tensors. tensors, types, shapes and classes are parallel
  # nested structures (serialized values, element types, shapes, and item classes). Where the
  # class is sparse, the tensor is deserialized using its type and the rank of its shape;
  # other entries pass through. Returns a structure with the nesting of the types argument.
  # Adds ops to the current graph.
  """Deserializes sparse tensors.

  Args:
    tensors: a structure of tensors to deserialize.
    types: a structure that holds information about types of `tensors`
    shapes: a structure that holds information about shapes of `tensors`
    classes: a structure of objects that identify the dataset item classes

  Returns:
    `tensors` with any serialized sparse tensors replaced by their deserialized
    version.
  """
  ret = nest.pack_sequence_as(types, [
      sparse_ops.deserialize_sparse(tensor, dtype=ty, rank=shape.ndims)
      if c is sparse_tensor.SparseTensor else tensor
      for (tensor, ty, shape, c) in zip(
          nest.flatten(tensors), nest.flatten(types), nest.flatten(shapes),
          nest.flatten(classes))
  ])
  return ret


def get_classes(tensors):
  # [FUNCTION DOC]
  # Describes the kind of each element of a nested structure of tensors. Returns a structure of
  # the same nesting holding the sparse tensor class for sparse tensors and the dense tensor
  # class for everything else. No side effects.
  """Gets classes for a structure of tensors.

  Args:
    tensors: the tensor structure to get classes for.

  Returns:
    a structure matching the nested structure of `tensors`, containing
    `tf.sparse.SparseTensor` at positions where `tensors` contains a sparse
    tensor and `tf.Tensor` otherwise.
  """
  return nest.pack_sequence_as(tensors, [
      sparse_tensor.SparseTensor
      if isinstance(tensor, sparse_tensor.SparseTensor) else tensor_lib.Tensor
      for tensor in nest.flatten(tensors)
  ])


def serialize_many_sparse_tensors(tensors):
  # [FUNCTION DOC]
  # KNOWN ISSUE: sparse detection uses a general sparse check, whereas the single-tensor
  # variant checks for the sparse tensor class only.
  # Serializes each sparse tensor in a nested structure, treating its first dimension as a
  # batch, into a variant-typed tensor holding one serialized row per batch entry. Other tensors
  # are unchanged. Returns a structure with the same nesting. Adds ops to the current graph.
  """Serializes many sparse tensors into a batch.

  Args:
    tensors: a tensor structure to serialize.

  Returns:
    `tensors` with any sparse tensors replaced by the serialized batch.
  """

  ret = nest.pack_sequence_as(tensors, [
      sparse_ops.serialize_many_sparse(tensor, out_type=dtypes.variant)
      if sparse_tensor.is_sparse(tensor) else tensor
      for tensor in nest.flatten(tensors)
  ])
  return ret


def serialize_sparse_tensors(tensors):
  # [FUNCTION DOC]
  # Serializes each single sparse tensor in a nested structure into a variant-typed scalar
  # tensor, leaving other tensors unchanged. tensors is a nested structure of tensors. Returns
  # a structure with the same nesting. Adds serialization ops to the current graph.
  """Serializes sparse tensors.

  Args:
    tensors: a tensor structure to serialize.

  Returns:
    `tensors` with any sparse tensors replaced by their serialized version.
  """

  ret = nest.pack_sequence_as(tensors, [
      sparse_ops.serialize_sparse(tensor, out_type=dtypes.variant)
      if isinstance(tensor, sparse_tensor.SparseTensor) else tensor
      for tensor in nest.flatten(tensors)
  ])
  return ret
