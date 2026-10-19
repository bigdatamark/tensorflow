# [MODULE DOC]
# Implementation of the dataset concatenate operation: a private entry function and a dataset class
# that yields all elements of one dataset followed by all elements of another. Used by the public
# dataset class. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.concatenate`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import nest as tf_nest


def _concatenate(input_dataset, dataset_to_concatenate, name):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: implementation behind the public concatenate method of datasets.
  # Takes two datasets and an op name; returns a concatenation dataset of the first followed by
  # the second.
  return _ConcatenateDataset(input_dataset, dataset_to_concatenate, name)


class _ConcatenateDataset(dataset_ops.DatasetV2):
  # [CLASS DOC]
  # HELPER: internal dataset implementing concatenation of two datasets.
  # Holds both input datasets, the combined element structure and an optional name. Elements of
  # the first dataset come first, then those of the second.
  """A `Dataset` that concatenates its input with given dataset."""

  def __init__(self, input_dataset, dataset_to_concatenate, name=None):
    # [FUNCTION DOC]
    # Inputs: the two datasets and an optional name. Computes the element structure as the most
    # specific common supertype of the two element structures. Raises a type error if the
    # structures are incompatible. Creates the concatenate variant tensor and calls the parent
    # constructor. Returns nothing.
    """See `Dataset.concatenate()` for details."""
    self._input_dataset = input_dataset
    self._dataset_to_concatenate = dataset_to_concatenate

    def common_supertype(a, b):
      # [FUNCTION DOC]
      # HELPER: finds the common supertype of two element specs.
      # Returns the most specific common supertype of a and b; raises a type error if none exists.
      result = a.most_specific_common_supertype([b])
      if result is None:
        raise TypeError(f"No common supertype of {a} and {b}.")
      return result

    try:
      self._structure = tf_nest.map_structure(
          common_supertype, input_dataset.element_spec,
          dataset_to_concatenate.element_spec)
    except (TypeError, ValueError) as e:
      raise TypeError(f"Incompatible dataset elements:\n"
                      f"  {input_dataset.element_spec} vs. "
                      f"  {dataset_to_concatenate.element_spec}") from e

    self._input_datasets = [input_dataset, dataset_to_concatenate]
    self._name = name
    # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.concatenate_dataset(
        input_dataset._variant_tensor, dataset_to_concatenate._variant_tensor,
        **self._common_args)
    # pylint: enable=protected-access
    super().__init__(variant_tensor)

  def _inputs(self):
    # [FUNCTION DOC]
    # Returns the list of the two input datasets (used for dataset graph traversal).
    return self._input_datasets

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the combined element structure computed at construction.
    return self._structure
