# [MODULE DOC]
# HELPER: internal implementation module for the dataset group_by_window transformation.
# Defines a private factory (validating window-size arguments) and a private dataset class that
# groups input elements by an integer key and reduces each window of elements with a function
# returning a dataset. Used by dataset_ops. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.group_by_window`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _group_by_window(input_dataset,  # pylint: disable=unused-private-name
                     key_func,
                     reduce_func,
                     window_size=None,
                     window_size_func=None,
                     name=None):
  # [FUNCTION DOC]
  # HELPER: factory used by the dataset group_by_window method.
  # Inputs: input_dataset; key_func (maps an element to an int64 scalar key); reduce_func (maps
  # a key and a windowed dataset to a dataset); window_size (optional integer, constant window
  # size); window_size_func (optional callable mapping a key to an int64 window size); name
  # (optional string). Exactly one of window_size and window_size_func must be given, otherwise
  # a ValueError is raised. A constant window size is turned into a function returning that
  # constant. Returns the grouped dataset.
  """See `Dataset.group_by_window()` for details."""

  if (window_size is not None and window_size_func or
      not (window_size is not None or window_size_func)):
    raise ValueError("Either the `window_size` argument or the "
                     "`window_size_func` argument must be specified.")

  if window_size is not None:

    def constant_window_func(unused_key):
      # [FUNCTION DOC]
      # HELPER: window-size function that ignores the key and returns the fixed window size
      # converted to an int64 tensor.
      return ops.convert_to_tensor(window_size, dtype=dtypes.int64)

    window_size_func = constant_window_func

  assert window_size_func is not None

  return _GroupByWindowDataset(
      input_dataset, key_func, reduce_func, window_size_func, name=name)


class _GroupByWindowDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: private dataset that groups elements by key into windows and reduces each window.
  # Holds the input dataset, wrapped key, reduce and window-size functions, the output element
  # spec and an optional name.
  """A `Dataset` that groups its input and performs a windowed reduction."""

  def __init__(self,
               input_dataset,
               key_func,
               reduce_func,
               window_size_func,
               name=None):
    # [FUNCTION DOC]
    # Builds the dataset. Inputs: input_dataset; key_func, reduce_func, window_size_func
    # (callables as described for the factory); name (optional string). Wraps the three
    # functions (each helper may raise a ValueError or TypeError on bad output types), then
    # creates the group-by-window variant tensor with the functions and their captured inputs.
    # Returns nothing.
    """See `group_by_window()` for details."""
    self._input_dataset = input_dataset
    self._make_key_func(key_func, input_dataset)
    self._make_reduce_func(reduce_func, input_dataset)
    self._make_window_size_func(window_size_func)
    self._name = name
    variant_tensor = ged_ops.group_by_window_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._key_func.function.captured_inputs,
        self._reduce_func.function.captured_inputs,
        self._window_size_func.function.captured_inputs,
        key_func=self._key_func.function,
        reduce_func=self._reduce_func.function,
        window_size_func=self._window_size_func.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _make_window_size_func(self, window_size_func):
    # [FUNCTION DOC]
    # HELPER: wraps the window-size callable as a structured function stored on the instance.
    # Input: window_size_func, mapping an int64 scalar key to a window size. Raises a ValueError
    # if its output is not an int64 scalar. Returns nothing.
    """Make wrapping defun for window_size_func."""

    def window_size_func_wrapper(key):
      # [FUNCTION DOC]
      # HELPER: calls the user window-size function and converts the result to an int64 tensor.
      return ops.convert_to_tensor(window_size_func(key), dtype=dtypes.int64)

    self._window_size_func = structured_function.StructuredFunctionWrapper(
        window_size_func_wrapper,
        self._transformation_name(),
        input_structure=tensor_spec.TensorSpec([], dtypes.int64))
    if not self._window_size_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.int64)):
      raise ValueError(f"Invalid `window_size_func`. `window_size_func` must "
                       f"return a single `tf.int64` scalar tensor but its "
                       f"return type is "
                       f"{self._window_size_func.output_structure}.")

  def _make_key_func(self, key_func, input_dataset):
    # [FUNCTION DOC]
    # HELPER: wraps the key callable as a structured function stored on the instance.
    # Inputs: key_func (maps an element to an int64 scalar key); input_dataset (defines the
    # element structure). Raises a ValueError if the output is not an int64 scalar.
    # Returns nothing.
    """Make wrapping defun for key_func."""

    def key_func_wrapper(*args):
      # [FUNCTION DOC]
      # HELPER: calls the user key function on the element components and converts the result to
      # an int64 tensor.
      return ops.convert_to_tensor(key_func(*args), dtype=dtypes.int64)

    self._key_func = structured_function.StructuredFunctionWrapper(
        key_func_wrapper, self._transformation_name(), dataset=input_dataset)
    if not self._key_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.int64)):
      raise ValueError(f"Invalid `key_func`. `key_func` must return a single "
                       f"`tf.int64` scalar tensor but its return type is "
                       f"{self._key_func.output_structure}.")

  def _make_reduce_func(self, reduce_func, input_dataset):
    # [FUNCTION DOC]
    # HELPER: wraps the reduce callable as a structured function stored on the instance.
    # Inputs: reduce_func (takes an int64 key and a dataset of the window elements, returns a
    # dataset); input_dataset (supplies the window element spec). Raises a TypeError if the
    # output is not a dataset. Records the output element spec. Returns nothing.
    """Make wrapping defun for reduce_func."""
    nested_dataset = dataset_ops.DatasetSpec(input_dataset.element_spec)
    input_structure = (tensor_spec.TensorSpec([], dtypes.int64), nested_dataset)
    self._reduce_func = structured_function.StructuredFunctionWrapper(
        reduce_func,
        self._transformation_name(),
        input_structure=input_structure)
    if not isinstance(self._reduce_func.output_structure,
                      dataset_ops.DatasetSpec):
      raise TypeError(f"Invalid `reduce_func`. `reduce_func` must return a "
                      f"single `tf.data.Dataset` object but its return type "
                      f"is {self._reduce_func.output_structure}.")
    # pylint: disable=protected-access
    self._element_spec = (self._reduce_func.output_structure._element_spec)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the element spec of the datasets produced by the reduce function.
    return self._element_spec

  def _functions(self):
    # [FUNCTION DOC]
    # Returns the list of wrapped functions: key, reduce and window-size.
    return [self._key_func, self._reduce_func, self._window_size_func]

  def _transformation_name(self):
    # [FUNCTION DOC]
    # Returns the display name of this transformation, used in error messages and tracing.
    return "Dataset.group_by_window()"
