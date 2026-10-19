# [MODULE DOC]
# Implementation module for the dataset prefetch transformation. It exposes a private factory
# function and a dataset class that wraps an input dataset so that its elements are produced
# asynchronously into a buffer ahead of consumption. It is used by the main dataset module to
# implement the public prefetch method. No import-time side effects beyond imports.
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
"""The implementation of `tf.data.Dataset.prefetch`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _prefetch(input_dataset, buffer_size, name=None):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: factory used by the public dataset prefetch method.
  # Inputs: input_dataset (the dataset to prefetch), buffer_size (number of elements to buffer,
  # or the autotune sentinel, or nothing meaning autotune in the dataset class), name (optional
  # string name for the operation).
  # Returns the input dataset unchanged when global debug mode is on; otherwise returns a new
  # prefetch dataset wrapping the input. Raises nothing itself.
  """See `Dataset.prefetch()` for details."""
  if debug_mode.DEBUG_MODE:
    return input_dataset
  return _PrefetchDataset(input_dataset, buffer_size, name=name)


class _PrefetchDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # Dataset that wraps one input dataset and asynchronously prefetches its elements into a
  # buffer. The element structure is unchanged from the input. Holds the input dataset, the
  # buffer size tensor and the optional name. Created by the prefetch factory function.
  """A `Dataset` that asynchronously prefetches its input."""

  def __init__(self, input_dataset, buffer_size, slack_period=None, name=None):
    # [FUNCTION DOC]
    # Builds the prefetch dataset variant tensor. Inputs: input_dataset (dataset to wrap),
    # buffer_size (integer or tensor; nothing is replaced by the autotune sentinel), slack_period
    # (optional integer passed through to the kernel, default nothing), name (optional string).
    # The buffer size is converted to a 64-bit integer tensor. The op is created colocated with
    # the input dataset's variant tensor. Legacy autotune is enabled when the buffer size equals
    # the autotune sentinel. Stores state on the instance and calls the parent constructor.
    """See `Dataset.prefetch()` for details."""
    self._input_dataset = input_dataset
    if buffer_size is None:
      buffer_size = dataset_ops.AUTOTUNE
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    self._name = name
    # pylint: disable=protected-access
    # We colocate the prefetch dataset with its input as this collocation only
    # happens automatically in graph mode.
    with ops.colocate_with(input_dataset._variant_tensor):
      variant_tensor = gen_dataset_ops.prefetch_dataset(
          input_dataset._variant_tensor,
          buffer_size=self._buffer_size,
          slack_period=slack_period,
          legacy_autotune=(buffer_size == dataset_ops.AUTOTUNE),
          **self._common_args)
    super().__init__(input_dataset, variant_tensor)
