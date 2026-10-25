# [MODULE DOC]
# HELPER: small constants module for the proto splitter. Holds the current version of the join
# implementation and a list of join versions known to be bad, with accessors for both. No
# import-time side effects.
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
"""Version constants."""

# Version of the Join implementation.
# This should be incremented almost whenever the joining implementation is
# updated. The only time this number should not be incremented is if the change
# is extremely trivial.
_JOIN_VERSION = 0

# Bad/Buggy join versions.
_BAD_VERSIONS = ()


def get_current_join_version() -> int:
  # [FUNCTION DOC]
  # Returns the current version number of the join implementation (an integer, currently zero).
  return _JOIN_VERSION


def get_bad_versions() -> tuple[int, ...]:
  # [FUNCTION DOC]
  # Returns the tuple of join versions known to be buggy (currently empty).
  return _BAD_VERSIONS
