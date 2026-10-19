# [MODULE DOC]
# Implementation module for the dataset repeat transformation: exposes a private factory
# function and a dataset class that repeats the input a given number of times (or forever).
# Used by the main dataset module. No import-time side effects beyond imports.
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
"""The implementation of `tf.data.Dataset.repeat`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _repeat(input_dataset, count, name):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: factory used by the public dataset repeat method.
  # Inputs: input_dataset, count (number of repetitions; nothing or negative means forever),
  # name (string or nothing; required positional here). Returns a new repeat dataset.
  return _RepeatDataset(input_dataset, count, name)


class _RepeatDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # Dataset repeating its input a fixed number of times or indefinitely. Element structure is
  # unchanged. Holds the input dataset, the count tensor and the name.
  """A `Dataset` that repeats its input several times."""

  def __init__(self, input_dataset, count, name=None):
    # [FUNCTION DOC]
    # Inputs: input_dataset, count (integer or tensor; nothing becomes minus one meaning repeat
    # forever), name (optional string). Converts count to a 64-bit tensor, creates the repeat op
    # and calls the parent constructor.
    """See `Dataset.repeat()` for details."""
    self._input_dataset = input_dataset
    if count is None:
      self._count = constant_op.constant(-1, dtype=dtypes.int64, name="count")
    else:
      self._count = ops.convert_to_tensor(
          count, dtype=dtypes.int64, name="count")
    self._name = name
    variant_tensor = gen_dataset_ops.repeat_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)
