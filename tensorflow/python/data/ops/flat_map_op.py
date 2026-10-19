# [MODULE DOC]
# HELPER: internal implementation module for the dataset flat_map transformation.
# Defines a private factory function and a private dataset class that maps a user function
# returning datasets over the input and flattens the results. Used by dataset_ops. No import-time
# side effects.
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
"""The implementation of `tf.data.Dataset.flat_map`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_dataset_ops


def _flat_map(input_dataset, map_func, name=None):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: factory used by the dataset flat_map method.
  # Inputs: input_dataset; map_func (callable returning a dataset); name (optional string).
  # Returns the flat-mapped dataset.
  """See `Dataset.flat_map()` for details."""
  return _FlatMapDataset(input_dataset, map_func, name)


class _FlatMapDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: private dataset that applies a dataset-returning function to each input element and
  # concatenates the resulting datasets. Holds the input dataset, the wrapped function, the inner
  # element structure and an optional name.
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func, name=None):
    # [FUNCTION DOC]
    # Builds the dataset. Inputs: input_dataset; map_func (callable returning a dataset); name
    # (optional string). Wraps map_func as a structured function; raises a TypeError if its
    # output is not a dataset. The element structure is that of the returned datasets. Creates
    # the flat-map variant tensor and initializes the base class. Returns nothing.

    self._input_dataset = input_dataset
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
      raise TypeError(
          "The `map_func` argument must return a `Dataset` object. Got "
          f"{dataset_ops.get_type(self._map_func.output_structure)!r}.")
    # pylint: disable=protected-access
    self._structure = self._map_func.output_structure._element_spec
    self._name = name
    variant_tensor = gen_dataset_ops.flat_map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # Returns a list holding the wrapped map function.
    return [self._map_func]

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the element spec of the flattened output (the inner datasets' element spec).
    return self._structure

  def _transformation_name(self):
    # [FUNCTION DOC]
    # Returns the display name of this transformation, used in error messages and tracing.
    return "Dataset.flat_map()"
