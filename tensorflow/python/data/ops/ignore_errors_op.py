# [MODULE DOC]
# HELPER: internal implementation module for the dataset ignore_errors transformation.
# Defines a private factory and a private dataset that silently drops elements whose production
# raised an error. Used by dataset_ops. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.ignore_errors`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops


def _ignore_errors(input_dataset, log_warning=False, name=None):
  # [FUNCTION DOC]
  # HELPER: factory used by the dataset ignore_errors method.
  # Inputs: input_dataset; log_warning (optional boolean, default false, log a warning for each
  # dropped error); name (optional string). Returns the error-skipping dataset.
  return _IgnoreErrorsDataset(input_dataset, log_warning, name)


class _IgnoreErrorsDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # HELPER: private dataset that skips input elements whose retrieval fails.
  # Holds the input dataset and an optional name; output structure equals the input's.
  """A `Dataset` that drops erroneous elements from its input."""

  def __init__(self, input_dataset, log_warning, name=None):
    # [FUNCTION DOC]
    # Builds the dataset. Inputs: input_dataset; log_warning (boolean, whether to log a warning
    # when an element is dropped); name (optional string). Creates the ignore-errors variant
    # tensor and initializes the base class. Returns nothing.
    """See `Dataset.ignore_errors` for details."""
    self._input_dataset = input_dataset
    self._name = name
    variant_tensor = (
        gen_experimental_dataset_ops.ignore_errors_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            log_warning=log_warning,
            **self._flat_structure))
    super().__init__(input_dataset, variant_tensor)
