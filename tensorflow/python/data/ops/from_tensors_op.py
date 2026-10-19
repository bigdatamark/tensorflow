# [MODULE DOC]
# HELPER: internal implementation module for the dataset from_tensors constructor.
# Defines a private factory and a private source dataset with exactly one element. Used by
# dataset_ops. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.from_tensors`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_dataset_ops


def _from_tensors(tensors, name):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: factory used by the dataset from_tensors method.
  # Inputs: tensors (nested structure forming the single element); name (string or none).
  # Returns the single-element dataset.
  return _TensorDataset(tensors, name)


class _TensorDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: private source dataset containing a single element.
  # Holds the element structure, its flattened tensor list and an optional name.
  """A `Dataset` with a single element."""

  def __init__(self, element, name=None):
    # [FUNCTION DOC]
    # Builds the dataset. Inputs: element (nested structure convertible to tensors); name
    # (optional string). Normalizes the element, records its structure and tensor list, and
    # creates the tensor variant tensor with output shapes and serialized metadata.
    # Returns nothing.
    """See `tf.data.Dataset.from_tensors` for details."""
    element = structure.normalize_element(element)
    self._structure = structure.type_spec_from_value(element)
    self._tensors = structure.to_tensor_list(self._structure, element)
    self._name = name
    variant_tensor = gen_dataset_ops.tensor_dataset(
        self._tensors,
        output_shapes=structure.get_flat_tensor_shapes(self._structure),
        metadata=self._metadata.SerializeToString())
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the structure of the single element.
    return self._structure
