# [MODULE DOC]
# HELPER: Shared testing utilities for the quantization libraries' tests.
# Provides a generator of all parameter combinations for parameterized tests and helpers that
# compute the total size of directories/files and the size ratio of two paths. No import-time
# side effects.
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Common testing utilities for quantization libraries."""
import itertools
import os
from typing import Any, Mapping, Sequence


def parameter_combinations(
    test_parameters: Sequence[Mapping[str, Sequence[Any]]]
) -> Sequence[Mapping[str, Any]]:
  # [FUNCTION DOC]
  # HELPER: Builds parameter sets for parameterized tests.
  # test_parameters: list of mappings from parameter names to lists of candidate values. For
  # each mapping, takes the cartesian product of its value lists and builds one mapping per
  # combination (names to chosen values). Returns the combined list over all input mappings, in
  # order; an empty input gives an empty list and a mapping with an empty value list yields no
  # combinations. Does not modify its input.
  """Generate all combinations of test parameters.

  Args:
    test_parameters: List of dictionaries that maps parameter keys and values.

  Returns:
    real_parameters: All possible combinations of the parameters as list of
    dictionaries.
  """
  real_parameters = []
  for parameters in test_parameters:
    keys = parameters.keys()
    for curr in itertools.product(*parameters.values()):
      real_parameters.append(dict(zip(keys, curr)))
  return real_parameters


def get_dir_size(path: str = '.') -> int:
  # [FUNCTION DOC]
  # HELPER: Test utility computing storage size.
  # path: directory (or file) to measure; defaults to the current directory. Walks the tree and
  # sums the byte sizes of all files found. Returns the total in bytes.
  # KNOWN ISSUE: A path that is a single file is documented as supported, but the tree walk of a
  # file yields nothing, so the result is zero for a file (and for missing paths).
  """Get the total size of files and sub-directories under the path.

  Args:
    path: Path of a directory or a file to calculate the total size.

  Returns:
    Total size of the directory or a file.
  """
  total = 0
  for root, _, files in os.walk(path):
    for filename in files:
      total += os.path.getsize(os.path.join(root, filename))
  return total


def get_size_ratio(path_a: str, path_b: str) -> float:
  # [FUNCTION DOC]
  # HELPER: Test utility comparing storage sizes.
  # path_a: path of the numerator; path_b: path of the denominator. Returns the total size of
  # path_a divided by the total size of path_b, using the directory size helper. Raises a
  # division-by-zero error if path_b has total size zero.
  """Return the size ratio of the given paths.

  Args:
    path_a: Path of a directory or a file to be the nominator of the ratio.
    path_b: Path of a directory or a file to be the denominator of the ratio.

  Returns:
    Ratio of size of path_a / size of path_b.
  """
  size_a = get_dir_size(path_a)
  size_b = get_dir_size(path_b)
  return size_a / size_b
