# [MODULE DOC]
# Implementation of the dataset cache operation: a private entry function and the cache dataset
# class that caches input elements in memory or in a file. Used by the public dataset class in
# the dataset ops module. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.cache`."""

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _cache(input_dataset, filename, name):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: implementation behind the public cache method of datasets.
  # Takes the input dataset, a filename (string; empty means in-memory) and an op name, and
  # returns a new cache dataset wrapping the input.
  return CacheDataset(input_dataset, filename, name)


class CacheDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # Dataset that caches the elements of its input, in memory (empty filename) or in files with
  # the given filename prefix. Element structure is unchanged from the input. Holds the input
  # dataset, filename tensor and optional name.
  """A `Dataset` that caches elements of its input."""

  def __init__(self, input_dataset, filename, name=None):
    # [FUNCTION DOC]
    # Inputs: input_dataset, filename (string, converted to a string tensor), name (optional).
    # When the v2 behavior is enabled and running eagerly or inside a function, creates the
    # version 2 cache dataset using a fresh in-memory cache resource; otherwise creates the
    # original cache dataset. Then calls the parent constructor. Returns nothing.
    """See `Dataset.cache()` for details."""
    self._input_dataset = input_dataset
    self._filename = ops.convert_to_tensor(
        filename, dtype=dtypes.string, name="filename")
    self._name = name
    if tf2.enabled() and (context.executing_eagerly() or ops.inside_function()):
      variant_tensor = gen_dataset_ops.cache_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          filename=self._filename,
          cache=gen_dataset_ops.dummy_memory_cache(),
          **self._common_args)
    else:
      variant_tensor = gen_dataset_ops.cache_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          filename=self._filename,
          **self._common_args)
    super().__init__(input_dataset, variant_tensor)
