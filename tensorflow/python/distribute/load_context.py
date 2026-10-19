# [MODULE DOC]
# Thread-local context for storing options used while loading a SavedModel. Exposes a context
# manager (exported as an internal API) that sets load options for its duration, plus accessors for
# the current options and whether a load is in progress. Creates one module-level thread-local
# context object at import time. Helper module for the loading machinery.

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
"""Context for storing options for loading a SavedModel."""

import contextlib
import threading

from tensorflow.python.util.tf_export import tf_export


class LoadContext(threading.local):
  # [CLASS DOC]
  # Thread-local holder of load state: a stack of markers recording nested entries into a load
  # context, and the current load options. Each thread sees its own state, initialized by the
  # constructor on first use in that thread. One module-level instance is used.
  """A context for loading a model."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor (runs per thread): initializes the empty stack of entered markers and sets the
    # load options to nothing.
    super().__init__()
    self._entered_load_context = []
    self._load_options = None

  def set_load_options(self, load_options):
    # [FUNCTION DOC]
    # Stores the given load options as current and pushes an entered marker. Returns nothing.
    self._load_options = load_options
    self._entered_load_context.append(True)

  def clear_load_options(self):
    # [FUNCTION DOC]
    # Resets the current load options to nothing and pops one entered marker (raises an index
    # error if none was pushed). Returns nothing.
    # KNOWN ISSUE: clears the options to nothing even when entries are nested, so leaving an inner
    # load context loses the outer context's options while the outer one is still active.
    self._load_options = None
    self._entered_load_context.pop()

  def load_options(self):
    # [FUNCTION DOC]
    # Returns the current load options (nothing when not loading).
    return self._load_options

  def in_load_context(self):
    # [FUNCTION DOC]
    # Returns the stack of entered markers itself (not a boolean): an empty list when not in a load
    # context, which is falsy, and a non-empty list (truthy) while inside one.
    return self._entered_load_context


_load_context = LoadContext()


@tf_export("__internal__.load_context", v1=[])
@contextlib.contextmanager
def load_context(load_options):
  # [FUNCTION DOC]
  # Context manager (exported as an internal API). Parameter: the load options to apply. On entry
  # stores them in the thread-local load context; yields nothing; on exit (even after an error)
  # clears the options in that context. Side effect: mutates thread-local module state.
  _load_context.set_load_options(load_options)
  try:
    yield
  finally:
    _load_context.clear_load_options()


def get_load_options():
  # [FUNCTION DOC]
  # Returns the load options of the current thread's load context, or nothing outside one.
  """Returns the load options under a load context."""
  return _load_context.load_options()


def in_load_context():
  # [FUNCTION DOC]
  # Returns the current thread's entered-marker stack (a list, truthy inside a load context and
  # empty otherwise), not strictly a boolean as the docstring suggests.
  """Returns whether under a load context."""
  return _load_context.in_load_context()
