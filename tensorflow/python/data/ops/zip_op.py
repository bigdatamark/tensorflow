# [MODULE DOC]
# HELPER: helper module holding the implementation behind the dataset zip transformation.
# Defines a private zip function and a private dataset class that pairs up elements of several
# datasets into nested structures. Used by the dataset class in the dataset ops module.
# No import-time side effects beyond imports.
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
"""The implementation of `tf.data.Dataset.zip`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.types import data as data_types


def _zip(datasets, name):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # HELPER: implements the public zip transformation of datasets.
  # Takes a nested structure of datasets and an optional name; returns the zipped dataset.
  # Errors (non-dataset inputs) come from the dataset constructor.
  return _ZipDataset(datasets, name)


class _ZipDataset(dataset_ops.DatasetV2):
  # [CLASS DOC]
  # HELPER: internal dataset class backing the public zip transformation.
  # Combines several datasets (given as a nested structure) element-wise into one dataset whose
  # elements have the same nested structure. Holds the datasets, the element structure and name.
  """A `Dataset` that zips its inputs together."""

  def __init__(self, datasets, name=None):
    # [FUNCTION DOC]
    # Constructor. datasets is a (nested) structure of dataset objects; name is an optional
    # operation name. Raises a type error if any leaf is not a dataset, with a specific message
    # when a list is used (tuples must be used instead). Records the structure, builds the
    # element structure from the element specs and creates the zip dataset op in the current
    # graph, then initializes the base dataset. Returns nothing.
    """See `Dataset.zip()` for details."""
    for ds in nest.flatten(datasets):
      if not isinstance(ds, data_types.DatasetV2):
        if isinstance(ds, list):
          raise TypeError(
              "Invalid input to `zip`. Inputs are expected to be (nested)"
              " structures of `tf.data.Dataset` objects. Python `list` is"
              " not supported and you should use `tuple` instead."
          )
        else:
          raise TypeError(
              "Invalid input to `zip`. Inputs are expected to be (nested)"
              " structures of `tf.data.Dataset` objects but"
              f" encountered object of type {type(ds)}."
          )
    self._datasets = datasets
    self._structure = nest.pack_sequence_as(
        self._datasets, [ds.element_spec for ds in nest.flatten(self._datasets)]
    )
    self._name = name
    variant_tensor = gen_dataset_ops.zip_dataset(
        [ds._variant_tensor for ds in nest.flatten(self._datasets)],
        **self._common_args,
    )
    super().__init__(variant_tensor)

  def _inputs(self):
    # [FUNCTION DOC]
    # HELPER: reports the input datasets to the dataset graph machinery.
    # Returns a flat list of all datasets in the (possibly nested) structure given at creation.
    return nest.flatten(self._datasets)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element specification: the same nesting as the input datasets
    # structure with each dataset replaced by its element specification. No arguments.
    return self._structure
