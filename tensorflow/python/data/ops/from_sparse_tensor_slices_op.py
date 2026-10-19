# [MODULE DOC]
# HELPER: internal implementation module for the dataset from_sparse_tensor_slices constructor.
# Defines a private factory and a private source dataset that splits a sparse tensor into its
# rows, each element being an (indices, values, dense shape) triple. Used by dataset_ops. No
# import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of `tf.data.Dataset.from_sparse_tensor_slices`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops


def _from_sparse_tensor_slices(sparse_tensor):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: factory used by the dataset from_sparse_tensor_slices method.
  # Input: sparse_tensor, a sparse tensor. Returns the sliced dataset wrapped in the version-1
  # dataset adapter. Errors come from the dataset constructor.
  return dataset_ops.DatasetV1Adapter(_SparseTensorSliceDataset(sparse_tensor))


class _SparseTensorSliceDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: private source dataset that splits a sparse tensor along its first dimension.
  # Holds the sparse tensor and the element structure (indices, values, dense shape specs).
  """A `Dataset` that splits a rank-N `tf.sparse.SparseTensor` into its rows."""

  def __init__(self, sparse_tensor):
    # [FUNCTION DOC]
    # Builds the dataset. Input: sparse_tensor, a sparse tensor; raises a TypeError otherwise.
    # Computes the rank of each slice by merging the rank implied by the indices shape and by
    # the dense shape (may raise a shape-incompatibility error), sets the element structure and
    # creates the sparse-slice variant tensor. Returns nothing.
    """See `Dataset.from_sparse_tensor_slices()` for details."""
    if not isinstance(sparse_tensor, sparse_tensor_lib.SparseTensor):
      raise TypeError(f"Invalid `sparse_tensor`. `sparse_tensor` must be a "
                      f"`tf.sparse.SparseTensor`. Got {type(sparse_tensor)}.")
    self._sparse_tensor = sparse_tensor

    indices_shape = self._sparse_tensor.indices.get_shape()
    shape_shape = self._sparse_tensor.dense_shape.get_shape()
    rank = (indices_shape.dims[1] - 1).merge_with(shape_shape.dims[0] - 1)
    self._structure = (tensor_spec.TensorSpec([None, rank], dtypes.int64),
                       tensor_spec.TensorSpec([None],
                                              self._sparse_tensor.dtype),
                       tensor_spec.TensorSpec([rank], dtypes.int64))

    variant_tensor = gen_dataset_ops.sparse_tensor_slice_dataset(
        self._sparse_tensor.indices, self._sparse_tensor.values,
        self._sparse_tensor.dense_shape)
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the element structure: specs for indices, values and dense shape of one slice.
    return self._structure
