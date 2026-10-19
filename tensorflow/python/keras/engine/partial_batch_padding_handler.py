# [MODULE DOC]
# Defines PartialBatchPaddingHandler, a utility used by the prediction loops (for TPU strategies)
# to pad a final, smaller batch up to the full batch size, track which rows are real with a mask,
# and strip the padded rows out of the prediction results afterwards. No import-time side effects.
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
"""Utility object to handler partial batches for TPUStrategy."""
# pylint: disable=protected-access

import numpy as np

from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


class PartialBatchPaddingHandler(object):
  # [CLASS DOC]
  # Holds the state needed to handle partial batches during prediction: the full padded batch
  # size (initially zero, set by the caller), a running mask tensor marking real versus padded
  # rows, and the structure of the model outputs. Provides methods to measure a batch, extend the
  # mask, pad batch tensors, and remove padded predictions.
  """A container that holds info about partial batches for `predict()`."""

  def __init__(self, output_shape):
    # [FUNCTION DOC]
    # Constructor. Input: output_shape, a list of output shapes of the model (its length is used
    # to tell single-output from multi-output models). Sets the padded batch size to zero, the
    # padding mask to an empty float tensor (creates a tensor op), and stores the output shape.
    self.padded_batch_size = 0
    self.padding_mask = array_ops.zeros(0)
    self.output_shape = output_shape

  def get_real_batch_size(self, dataset_batch):
    # [FUNCTION DOC]
    # Returns the actual number of rows in a batch as a 64-bit integer tensor. Input:
    # dataset_batch, a tensor, nested structure or tuple/list (only the first element of a tuple or
    # list, the features, is examined). Asserts the structure is not empty. Takes the first tensor
    # found in the flattened structure and reads its leading dimension. Raises a value error if no
    # tensor is present.
    """Returns the number of elements in a potentially partial batch."""
    if isinstance(dataset_batch, (tuple, list)):
      dataset_batch = dataset_batch[0]

    assert nest.flatten(dataset_batch)

    def _find_any_tensor(batch_features):
      # [FUNCTION DOC]
      # HELPER: picks a representative tensor for batch size measurement. Input: batch_features,
      # a nested structure. Returns the first tensor in flattened order; raises a value error if
      # there is none.
      tensors = [
          x for x in nest.flatten(batch_features) if tensor_util.is_tf_type(x)
      ]
      if not tensors:
        raise ValueError('Cannot find any Tensor in features dict.')
      return tensors[0]

    return backend.cast(backend.shape(_find_any_tensor(dataset_batch))[0],
                        dtype='int64')

  def update_mask(self, padding_mask, dataset_batch):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it caches the padding amount, but it caches nothing; it
    # returns a new mask and leaves the stored mask unchanged (the caller must store the result).
    # Inputs: padding_mask, the mask tensor so far; dataset_batch, the batch being processed.
    # Computes the missing row count as padded batch size minus the real size, builds a mask of
    # ones for real rows followed by zeros for padded rows, and returns it appended to the given
    # mask. Does not mutate state.
    """Calculate and cache the amount of padding required for a batch."""
    original_batch_size = self.get_real_batch_size(dataset_batch)
    missing_count = self.padded_batch_size - original_batch_size
    mask = backend.concatenate([array_ops.ones(original_batch_size),
                                array_ops.zeros(missing_count)], axis=0)
    return backend.concatenate([padding_mask, mask], axis=0)

  def pad_batch(self, *dataset_batch_elements):
    # [FUNCTION DOC]
    # Pads each given batch element at the end of its first dimension with zeros up to the padded
    # batch size. Input: one or more batch elements, each a tensor or a mapping from names to
    # tensors or nested mappings. Returns the padded element if exactly one was given, otherwise a
    # tuple of padded elements. Asserts rank is above zero for tensors. Padding is the difference
    # between padded batch size and real size, so it must be nonnegative.
    """Pads out the batch dimension of a tensor to the complete batch size."""
    def _pad(batch):
      # [FUNCTION DOC]
      # HELPER: recursive padding routine for pad_batch. Mappings are padded value by value and
      # returned as a new mapping; tensors get zero padding after the last real row on the first
      # axis only. Asserts rank is positive. Returns the padded structure.
      """Helper function to pad nested data within each batch elements."""
      padded_dict_batch = {}
      if isinstance(batch, dict):
        for key, value in batch.items():
          padded_dict_batch[key] = _pad(value)
        return padded_dict_batch

      rank = len(batch.shape)
      assert rank > 0
      missing_count = (self.padded_batch_size -
                       self.get_real_batch_size(batch))
      padding = backend.stack([[0, missing_count]] + [[0, 0]] * (rank - 1))
      return array_ops.pad(batch, padding, 'constant')

    if len(dataset_batch_elements) == 1:
      return _pad(dataset_batch_elements[0])

    batch_elements = []
    for batch_element in dataset_batch_elements:
      batch_elements.append(_pad(batch_element))
    return tuple(batch_elements)

  def apply_mask(self, prediction_result):
    # [FUNCTION DOC]
    # Drops predictions that came from padded rows. Input: prediction_result, an array for a
    # single-output model or a list of arrays (one per output) otherwise. Evaluates the stored
    # padding mask (asserting it is one-dimensional), keeps the rows where the mask is nonzero
    # (using only as much mask as there are predictions), and returns the filtered array for a
    # single output (leading size-one axis removed) or a list of squeezed arrays for multiple
    # outputs. Note the multi-output branch squeezes all size-one axes, which can also remove real
    # size-one dimensions.
    """Removes prediction output that corresponds to padded input."""
    padding_mask = backend.get_value(self.padding_mask)
    assert len(padding_mask.shape) == 1

    if len(self.output_shape) == 1:
      prediction = np.take(prediction_result,
                           np.nonzero(
                               padding_mask[:len(prediction_result)]),
                           axis=0)
      if prediction.shape[0] == 1:
        prediction = np.squeeze(prediction, axis=0)
      return prediction

    else:
      predictions = []
      for i in range(len(self.output_shape)):
        prediction = prediction_result[i]
        prediction = np.take(prediction, np.nonzero(
            padding_mask[:len(prediction)]), axis=0)
        predictions.append(np.squeeze(prediction))

      return predictions
