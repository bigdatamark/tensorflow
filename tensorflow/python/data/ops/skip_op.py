# [MODULE DOC]
# Implementation module for the dataset skip transformation: drops the first given number of
# input elements. Exposes a private factory function and a dataset class used by the main dataset
# module. No import-time side effects beyond imports.
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
"""The implementation of `tf.data.Dataset.skip`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _skip(self, count, name=None):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: factory used by the public dataset skip method (the first parameter is the input
  # dataset, named as an instance). Inputs: count (number of elements to skip), name (optional).
  # Returns a new skip dataset.
  return _SkipDataset(self, count, name)


class _SkipDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # Dataset that skips the first count elements of its input and yields the rest. Element
  # structure unchanged. Holds the input, the count tensor and the name.
  """A `Dataset` skipping the first `count` elements from its input."""

  def __init__(self, input_dataset, count, name=None):
    # [FUNCTION DOC]
    # Inputs: input_dataset, count (integer or tensor converted to a 64-bit integer tensor; a
    # negative value skips everything), name (optional). Creates the skip op and calls the parent
    # constructor.
    """See `Dataset.skip()` for details."""
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")
    self._name = name
    variant_tensor = gen_dataset_ops.skip_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)
