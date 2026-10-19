# [MODULE DOC]
# Implementation module for the dataset take-while transformation: yields input elements until a
# predicate first returns false. Exposes a private factory function and a dataset class used by
# the main dataset module. No import-time side effects beyond imports.
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
"""The implementation of `tf.data.Dataset.take_while`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _take_while(input_dataset, predicate, name=None):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: factory used by the public dataset take-while method.
  # Inputs: input_dataset, predicate (callback returning a boolean scalar tensor), name
  # (optional). Returns a new take-while dataset.
  """See `Dataset.take_while()` for details."""
  return _TakeWhileDataset(input_dataset, predicate, name=name)


class _TakeWhileDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # Dataset that stops iteration the first time the predicate returns false. Element structure
  # unchanged. Holds the input, the wrapped predicate function and the name.
  """A dataset that stops iteration when `predicate` returns false."""

  def __init__(self, input_dataset, predicate, name=None):
    # [FUNCTION DOC]
    # Inputs: input_dataset, predicate (callback applied to each element), name (optional).
    # Wraps the predicate using the input's structure; raises a value error if it does not return
    # a boolean scalar tensor (the message lacks a space before the reported type). Creates the
    # take-while op and calls the parent constructor.
    """See `take_while()` for details."""

    self._input_dataset = input_dataset
    wrapped_func = structured_function.StructuredFunctionWrapper(
        predicate, self._transformation_name(), dataset=self._input_dataset)

    if not wrapped_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.bool)):
      raise ValueError(f"Invalid `predicate`. `predicate` must return a "
                       f"`tf.bool` scalar tensor but its return type is"
                       f"{wrapped_func.output_structure}.")

    self._predicate = wrapped_func
    self._name = name
    variant_tensor = ged_ops.take_while_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._predicate.function.captured_inputs,
        predicate=self._predicate.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # Returns a list holding the wrapped predicate, used for function tracking.
    return [self._predicate]

  def _transformation_name(self):
    # [FUNCTION DOC]
    # Returns the display name of this transformation used in error messages.
    return "Dataset.take_while()"
