# [MODULE DOC]
# HELPER: holds the shared maximum chunk size used by the proto splitter modules, kept just under
# 2GB because the split algorithm is imprecise. Exposes a debug setter that overrides the limit in
# module state and a getter that returns it.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Common constants used by proto splitter modules."""

# The splitter algorithm isn't extremely precise, so the max is set to a little
# less than 2GB.
#
# TODO: b/380463192 - Consider fixing the split algorithm to handle edge cases
# accurately and raising the max size to 2GB.
_MAX_SIZE = 1 << 30


def debug_set_max_size(value: int) -> None:
  # [FUNCTION DOC]
  # HELPER: debugging hook (for tests) to change the chunk size limit.
  # KNOWN ISSUE: the module comment says the default is a little less than 2GB, but the value is
  # one gibibyte (half of 2GB).
  # Input: value, new limit in bytes. Replaces the module-level limit, affecting all later calls
  # to the getter. Returns nothing.
  """Sets the max size allowed for each proto chunk (used for debugging only).

  Args:
    value: int byte size
  """
  global _MAX_SIZE
  _MAX_SIZE = value


def max_size() -> int:
  # [FUNCTION DOC]
  # Returns the current maximum size in bytes of each proto chunk (one gibibyte unless changed by
  # the debug setter). Takes no arguments and has no side effects.
  """Returns the maximum size each proto chunk."""
  return _MAX_SIZE
