# [MODULE DOC]
# Experimental API for building a dataset from a Python list of elements. Defines the private
# list dataset source and the public (exported) from_list function that wraps it. Elements may
# be nested structures with differing shapes (non-rectangular input), unlike slicing-based
# constructors. No import-time side effects other than public symbol export.
#
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
"""Python API for creating a dataset from a list."""

import itertools

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export


class _ListDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: dataset source backing from_list. Holds the flattened tensors of all elements, the
  # inferred element structure (the most specific common supertype of all elements' specs, so
  # shapes may become partially unknown) and an optional op name. Produces the list's items one
  # by one.
  """A `Dataset` of elements from a list."""

  def __init__(self, elements, name=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: elements (non-empty list of elements with identical nested structure;
    # each is normalized to tensors) and name (optional operation name). Raises a value error if
    # elements is empty or not a list; the structure-equality check raises if elements differ in
    # nested structure. Computes each element's type spec, merges them into a common supertype
    # per component, keeps the flattened tensors (or the variant tensors for nested datasets),
    # and builds the list-dataset operation with the dataset metadata. Returns nothing.
    if not elements:
      raise ValueError(
          "Invalid `elements`. `elements` should not be empty. If you want an"
          " empty dataset, use `tf.data.Dataset.range(0)`."
      )
    if not isinstance(elements, list):
      raise ValueError("Invalid `elements`. `elements` must be a list.")

    elements = [structure.normalize_element(element) for element in elements]
    type_specs = [
        structure.type_spec_from_value(element) for element in elements
    ]

    # Check that elements have same nested structure.
    num_elements = len(elements)
    for i in range(1, num_elements):
      nest.assert_same_structure(type_specs[0], type_specs[i])

    # Infer elements' supershape.
    flattened_type_specs = [nest.flatten(type_spec) for type_spec in type_specs]
    num_tensors_per_element = len(flattened_type_specs[0])
    flattened_structure = [None] * num_tensors_per_element
    for i in range(num_tensors_per_element):
      flattened_structure[i] = flattened_type_specs[0][i]
      for j in range(1, num_elements):
        flattened_structure[i] = flattened_structure[
            i].most_specific_common_supertype([flattened_type_specs[j][i]])

    if not isinstance(type_specs[0], dataset_ops.DatasetSpec):
      self._tensors = list(
          itertools.chain.from_iterable(
              [nest.flatten(element) for element in elements]))
    else:
      self._tensors = [x._variant_tensor for x in elements]
    self._structure = nest.pack_sequence_as(type_specs[0], flattened_structure)
    self._name = name
    variant_tensor = gen_experimental_dataset_ops.list_dataset(
        self._tensors,
        output_types=self._flat_types,
        output_shapes=self._flat_shapes,
        metadata=self._metadata.SerializeToString())
    super(_ListDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the merged element structure inferred at construction.
    return self._structure


@tf_export("data.experimental.from_list")
def from_list(elements, name=None):
  # [FUNCTION DOC]
  # Public function (exported) creating a dataset from a list. Inputs: elements (non-empty list
  # whose items share one nested structure) and name (optional operation name). Returns a dataset
  # yielding the list items in order. Raises a value error for an empty or non-list input.
  """Creates a `Dataset` comprising the given list of elements.

  The returned dataset will produce the items in the list one by one. The
  functionality is identical to `Dataset.from_tensor_slices` when elements are
  scalars, but different when elements have structure. Consider the following
  example.

  >>> dataset = tf.data.experimental.from_list([(1, 'a'), (2, 'b'), (3, 'c')])
  >>> [(n.item(), s) for n, s in dataset.as_numpy_iterator()]
  [(1, b'a'), (2, b'b'), (3, b'c')]

  To get the same output with `from_tensor_slices`, the data needs to be
  reorganized:

  >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3], ['a', 'b', 'c']))
  >>> [(n.item(), s) for n, s in dataset.as_numpy_iterator()]
  [(1, b'a'), (2, b'b'), (3, b'c')]

  Unlike `from_tensor_slices`, `from_list` supports non-rectangular input:

  >>> dataset = tf.data.experimental.from_list([[1], [2, 3]])
  >>> list(dataset.as_numpy_iterator())
  [array([1], dtype=int32), array([2, 3], dtype=int32)]

  Achieving the same with `from_tensor_slices` requires the use of ragged
  tensors.

  `from_list` can be more performant than `from_tensor_slices` in some cases,
  since it avoids the need for data slicing each epoch. However, it can also be
  less performant, because data is stored as many small tensors rather than a
  few large tensors as in `from_tensor_slices`. The general guidance is to
  prefer `from_list` from a performance perspective when the number of elements
  is small (less than 1000).

  Args:
    elements: A list of elements whose components have the same nested
      structure.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    Dataset: A `Dataset` of the `elements`.
  """
  return _ListDataset(elements, name)
