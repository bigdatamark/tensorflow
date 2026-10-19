# [MODULE DOC]
# Defines the dispatch context used by AutoGraph operators: a small immutable record that carries
# extra option values to specific operator implementations, plus a shared empty instance (NO_CTX)
# used when no options are needed. Helper module; no import-time side effects beyond creating
# the empty instance.
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
"""Structures that allow uniform control over the dispatch process."""

import collections


# TODO(mdan): This is where macro override controls fit.


class DispatchContext(collections.namedtuple(
    'DispatchContext',
    ('options',))):
  # [CLASS DOC]
  # Named tuple with a single field, options: an optional mapping from option names to values
  # that specific operator implementations may need. Created by callers (or the shared empty
  # instance NO_CTX) and passed to dispatching operators; read through the option method.
  """Allows passing additional parameters to the specific implementations.

  Attributes:
    options: Optional dict of extra arguments that may be required by specific
      implementations.
  """

  def option(self, name):
    # [FUNCTION DOC]
    # Returns the value stored under the given option name in the options mapping. Raises a
    # key error if the name is absent, and fails if options is nothing/null.
    return self.options[name]


NO_CTX = DispatchContext(options={})
