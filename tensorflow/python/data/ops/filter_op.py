# [MODULE DOC]
# HELPER: internal implementation module for the dataset filter transformation.
# Defines a private factory function and a private dataset class that keeps only the elements for
# which a user predicate returns a scalar true boolean. Used by dataset_ops. No import-time side
# effects.
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
"""The implementation of `tf.data.Dataset.filter`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops


def _filter(input_dataset, predicate, name=None):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # HELPER: factory used by the dataset filter method.
  # Inputs: input_dataset (dataset to filter), predicate (callable mapping an element to a scalar
  # boolean tensor), name (optional string, default none). Returns the filtered dataset.
  return _FilterDataset(input_dataset, predicate, name=name)


class _FilterDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # HELPER: private dataset that drops input elements for which the predicate is false.
  # Holds the input dataset, the wrapped predicate function and an optional name. Output structure
  # is identical to the input's (unchanged-structure unary dataset).
  """A `Dataset` that filters its input according to a predicate function."""

  def __init__(self,
               input_dataset,
               predicate,
               use_legacy_function=False,
               name=None):
    # [FUNCTION DOC]
    # Builds the filter dataset. Inputs: input_dataset; predicate (callable returning a scalar
    # boolean tensor); use_legacy_function (optional boolean, default false) selecting the legacy
    # function tracing path; name (optional string).
    # Wraps the predicate as a structured function and raises a ValueError if its output is not a
    # scalar boolean. Creates the underlying filter variant tensor with the predicate's captured
    # inputs and initializes the base class. Returns nothing.
    """See `Dataset.filter` for details."""
    self._input_dataset = input_dataset
    wrapped_func = structured_function.StructuredFunctionWrapper(
        predicate,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    if not wrapped_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.bool)):
      raise ValueError(f"Invalid `predicate`. `predicate` must return a "
                       f"`tf.bool` scalar tensor, but its return type is "
                       f"{wrapped_func.output_structure}.")
    self._predicate = wrapped_func
    self._name = name
    variant_tensor = gen_dataset_ops.filter_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._predicate.function.captured_inputs,
        predicate=self._predicate.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # Returns a list holding the wrapped predicate function.
    return [self._predicate]

  def _transformation_name(self):
    # [FUNCTION DOC]
    # Returns the display name of this transformation, used in error messages and tracing.
    return "Dataset.filter()"
