# [MODULE DOC]
# Holds configuration switches for ragged tensor behavior. Currently it exposes a
# single function reporting whether differing row-partition dtypes are auto-converted
# when ragged tensors are combined. It has no import-time side effects and is used as
# a helper by the other ragged modules.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Configuration parameters for RaggedTensors."""


def auto_cast_partition_dtype():
  # [FUNCTION DOC]
  # HELPER: configuration query used by ragged ops that combine row partitions.
  # Takes no inputs and returns a boolean. In this version it always returns false, meaning
  # operations mixing row-partitioning dtypes are not auto-cast and will error instead.
  # No side effects and no exceptions.
  """Whether incompatible row-partitioning dtypes should be auto-converted.

  If true, then operations that combine RaggedTensors but have different
  row-partitioning tensor dtypes will be automatically cast to a
  compatible dtype (`tf.int64`).  If false, then such operations will result
  in an error.

  Returns:
    `bool`
  """
  return False
