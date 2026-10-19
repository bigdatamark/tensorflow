# [MODULE DOC]
# Implementation of the dataset batch operation. Provides the private entry point that chooses
# between a sequential batch dataset and a parallel batch dataset, plus the two dataset classes
# that wrap the underlying batch kernels. Used by the public dataset class in the dataset ops
# module. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.batch`."""

import warnings

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops


def _batch(input_dataset,
           batch_size,
           drop_remainder=False,
           num_parallel_calls=None,
           deterministic=None,
           name=None):
  # [FUNCTION DOC]
  # HELPER: implementation behind the public batch method of datasets.
  # Inputs: input_dataset, batch_size (scalar count), drop_remainder (bool, default false),
  # num_parallel_calls (optional; nothing means sequential), deterministic (optional bool,
  # only meaningful with parallel calls), name (optional op name).
  # Returns a sequential batch dataset when num_parallel_calls is not given or debug mode is on,
  # otherwise a parallel batch dataset. Emits a warning if deterministic is given without
  # parallel calls (outside debug mode). Raises nothing itself.
  """See `Dataset.batch` for details."""
  if num_parallel_calls is None or debug_mode.DEBUG_MODE:
    if deterministic is not None and not debug_mode.DEBUG_MODE:
      warnings.warn("The `deterministic` argument has no effect unless the "
                    "`num_parallel_calls` argument is specified.")
    return _BatchDataset(input_dataset, batch_size, drop_remainder, name=name)
  else:
    return _ParallelBatchDataset(
        input_dataset,
        batch_size,
        drop_remainder,
        num_parallel_calls,
        deterministic,
        name=name)


class _BatchDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: internal dataset implementing sequential batching.
  # Wraps an input dataset and groups consecutive elements into batches via the batch kernel.
  # Holds the input dataset, batch size and drop-remainder tensors, the batched element
  # structure and an optional name.
  """A `Dataset` that batches contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, drop_remainder, name=None):
    # [FUNCTION DOC]
    # Inputs: input_dataset, batch_size (integer, converted to a 64-bit tensor), drop_remainder
    # (bool, converted to a bool tensor), name (optional).
    # Computes the batched element structure: if drop_remainder is statically true, the batch
    # dimension is the constant batch size, otherwise it is unknown. Creates the batch dataset
    # variant tensor and calls the parent constructor. Returns nothing.
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
    # pylint: disable=protected-access
    if constant_drop_remainder:
      # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
      # or `False` (explicitly retaining the remainder).
      # pylint: disable=g-long-lambda
      constant_batch_size = tensor_util.constant_value(self._batch_size)
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(constant_batch_size),
          input_dataset.element_spec)
    else:
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(None),
          input_dataset.element_spec)

    self._name = name
    variant_tensor = gen_dataset_ops.batch_dataset_v2(
        input_dataset._variant_tensor,
        batch_size=self._batch_size,
        drop_remainder=self._drop_remainder,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the batched element structure computed at construction.
    return self._structure


class _ParallelBatchDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: internal dataset implementing parallel batching.
  # Like the sequential batch dataset but also takes a parallelism level and a determinism
  # setting (default, true or false) and uses the parallel batch kernel.
  """A `Dataset` that batches contiguous elements from its input in parallel."""

  def __init__(self,
               input_dataset,
               batch_size,
               drop_remainder,
               num_parallel_calls,
               deterministic,
               name=None):
    # [FUNCTION DOC]
    # Inputs: input_dataset, batch_size, drop_remainder, num_parallel_calls (integer, converted
    # to a 64-bit tensor), deterministic (nothing means default setting, true or false), name
    # (optional).
    # Converts arguments to tensors, maps deterministic to a string setting, computes the batched
    # element structure (constant batch size only when drop_remainder is statically true),
    # creates the parallel batch variant tensor and calls the parent constructor. Returns nothing.
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    if deterministic is None:
      self._deterministic = "default"
    elif deterministic:
      self._deterministic = "true"
    else:
      self._deterministic = "false"

    constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
    # pylint: disable=protected-access
    if constant_drop_remainder:
      # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
      # or `False` (explicitly retaining the remainder).
      # pylint: disable=g-long-lambda
      constant_batch_size = tensor_util.constant_value(self._batch_size)
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(constant_batch_size),
          input_dataset.element_spec)
    else:
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(None),
          input_dataset.element_spec)

    self._name = name
    variant_tensor = gen_dataset_ops.parallel_batch_dataset(
        input_dataset._variant_tensor,
        batch_size=self._batch_size,
        num_parallel_calls=self._num_parallel_calls,
        drop_remainder=self._drop_remainder,
        deterministic=self._deterministic,
        **self._common_args)

    super().__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the batched element structure computed at construction.
    return self._structure
