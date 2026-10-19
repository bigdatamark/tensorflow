# [MODULE DOC]
# Experimental dataset source that lists files matching glob patterns. Defines
# MatchingFilesDataset, used by the file-listing dataset utilities. No import-time side effects
# and no export registration.
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
"""Experimental API for matching input filenames."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


class MatchingFilesDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # Dataset source producing scalar strings, one per file path matching the given patterns.
  # Holds the patterns as a string tensor and builds the generated matching-files dataset
  # operation; the element structure is a scalar string tensor.
  """A `Dataset` that list the files according to the input patterns."""

  def __init__(self, patterns):
    # [FUNCTION DOC]
    # Input: patterns, a string or list of strings (glob patterns) converted to a string tensor
    # named patterns. Stores the tensor, creates the matching-files dataset variant tensor and
    # calls the parent constructor. Returns nothing.
    self._patterns = ops.convert_to_tensor(
        patterns, dtype=dtypes.string, name="patterns")
    variant_tensor = ged_ops.matching_files_dataset(self._patterns)
    super(MatchingFilesDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element structure: a scalar tensor spec of string type.
    return tensor_spec.TensorSpec([], dtypes.string)
