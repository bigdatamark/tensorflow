# [MODULE DOC]
# HELPER: helper module holding the implementation behind the dataset window transformation.
# Defines a private window function and a private unary dataset class that groups consecutive
# input elements into nested datasets (windows) using a window dataset op. It is used by the
# dataset class in the dataset ops module. No import-time side effects beyond imports.
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
"""The implementation of `tf.data.Dataset.window`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _window(input_dataset, size, shift, stride, drop_remainder, name):
  # [FUNCTION DOC]
  # HELPER: implements the public window transformation of datasets.
  # Takes the input dataset, window size, shift (nothing/null means shift equals size), stride,
  # drop_remainder flag and an optional name. Returns a new window dataset wrapping the input.
  # No validation here; errors come from the dataset constructor. No side effects.
  if shift is None:
    shift = size
  return _WindowDataset(
      input_dataset, size, shift, stride, drop_remainder, name=name)


class _WindowDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: internal dataset class backing the public window transformation.
  # Unary dataset whose each element is a nested dataset (a window) built from consecutive input
  # elements. Holds the input dataset, size, shift, stride and drop_remainder tensors, the
  # nested element structure and the optional name. Created only by the window helper function.
  """A dataset that creates window datasets from the input elements."""

  def __init__(self,
               input_dataset,
               size,
               shift,
               stride,
               drop_remainder,
               name=None):
    # [FUNCTION DOC]
    # Constructor. Parameters: input_dataset is the dataset to window; size, shift, stride are
    # scalar values (or scalar tensors) for window length, step between window starts and
    # spacing of elements inside a window; drop_remainder is a boolean saying whether to drop
    # trailing windows smaller than size; name is an optional operation name.
    # All numeric arguments are converted to 64-bit integer tensors (drop_remainder to a boolean
    # tensor), so invalid values raise the usual conversion errors. The element structure is
    # built by wrapping each input component's type information into a dataset specification.
    # Creates the window dataset op in the current graph and initializes the base unary dataset
    # with it. Returns nothing.
    """See `window()` for more details."""
    self._input_dataset = input_dataset
    self._size = ops.convert_to_tensor(size, dtype=dtypes.int64, name="size")
    self._shift = ops.convert_to_tensor(shift, dtype=dtypes.int64, name="shift")
    self._stride = ops.convert_to_tensor(
        stride, dtype=dtypes.int64, name="stride")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    self._structure = nest.pack_sequence_as(
        dataset_ops.get_legacy_output_classes(input_dataset),
        [
            dataset_ops.DatasetSpec(  # pylint: disable=g-complex-comprehension
                structure.convert_legacy_structure(output_type, output_shape,
                                                   output_class))
            for output_class, output_shape, output_type in zip(
                nest.flatten(
                    dataset_ops.get_legacy_output_classes(input_dataset)),
                nest.flatten(
                    dataset_ops.get_legacy_output_shapes(input_dataset)),
                nest.flatten(
                    dataset_ops.get_legacy_output_types(input_dataset)))
        ])
    self._name = name
    variant_tensor = gen_dataset_ops.window_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        size=self._size,
        shift=self._shift,
        stride=self._stride,
        drop_remainder=self._drop_remainder,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element specification of this dataset: the structure of the input
    # elements with every component replaced by a dataset specification describing the nested
    # window dataset that is produced for it. Takes no arguments; no side effects.
    return self._structure
