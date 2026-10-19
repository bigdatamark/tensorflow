# [MODULE DOC]
# Implementation module for the dataset dense-to-sparse batch transformation: batches dense
# single-component elements of varying size into sparse tensors. Exposes a private factory
# function and a dataset class. No import-time side effects beyond imports.
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
"""The implementation of `tf.data.Dataset.sparse_batch`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _sparse_batch(input_dataset, batch_size, row_shape, name=None):
  # [FUNCTION DOC]
  # HELPER: factory used by the public sparse batch method.
  # Inputs: input_dataset, batch_size, row_shape (maximum shape of each element), name (optional).
  # Returns a new dense-to-sparse batch dataset.
  return _DenseToSparseBatchDataset(input_dataset, batch_size, row_shape, name)


class _DenseToSparseBatchDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # Unary dataset that groups consecutive dense elements into sparse tensors whose first dimension
  # is the batch and remaining dimensions are bounded by a row shape. Holds the input, batch size,
  # row shape, sparse element spec and name.
  """A `Dataset` that batches ragged dense elements into `tf.sparse.SparseTensor`s."""

  def __init__(self, input_dataset, batch_size, row_shape, name=None):
    # [FUNCTION DOC]
    # Inputs: input_dataset (elements must have a single component), batch_size (integer or
    # tensor), row_shape (partial shape bounding each element), name (optional).
    # Raises a type error when the input elements are not a single component. Builds the sparse
    # element spec with an unknown batch dimension plus the row shape, creates the op and calls the
    # parent constructor.
    """See `Dataset.dense_to_sparse_batch()` for more details."""
    if not isinstance(
        dataset_ops.get_legacy_output_types(input_dataset), dtypes.DType):
      raise TypeError("`dense_to_sparse_batch` requires an input dataset whose "
                      "elements have a single component, but the given dataset "
                      "has the following component types: "
                      f"{dataset_ops.get_legacy_output_types(input_dataset)}.")
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    self._row_shape = row_shape
    self._element_spec = sparse_tensor.SparseTensorSpec(
        tensor_shape.TensorShape([None]).concatenate(self._row_shape),
        dataset_ops.get_legacy_output_types(input_dataset))
    self._name = name
    variant_tensor = ged_ops.dense_to_sparse_batch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._batch_size,
        row_shape=convert.partial_shape_to_tensor(self._row_shape),
        **self._flat_structure)
    super(_DenseToSparseBatchDataset, self).__init__(input_dataset,
                                                     variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the sparse tensor element spec computed at construction.
    return self._element_spec
