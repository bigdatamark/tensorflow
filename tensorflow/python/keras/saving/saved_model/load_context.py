# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# HELPER: helper module for the SavedModel loading code.
# Keeps per-thread state recording whether a SavedModel load is in progress and which load
# options apply. Exposes a context manager that enters and leaves a load context, plus
# functions to read the current options and to ask whether a load context is active.
# Import-time side effect: creates the module-level thread-local context instance.
"""Context for storing options for loading a SavedModel."""

import contextlib
import threading


class LoadContext(threading.local):
  # [CLASS DOC]
  # Thread-local holder of load state. Each thread has its own stack of entered load contexts
  # (a list with one marker per nesting level) and the current load options (nothing when
  # unset). One module-level instance is used by the functions below.
  """A context for loading a model."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor (re-run for each thread using the instance): initializes an empty stack of
    # entered contexts and no load options.
    super(LoadContext, self).__init__()
    self._entered_load_context = []
    self._load_options = None

  def set_load_options(self, load_options):
    # [FUNCTION DOC]
    # Records load_options (the load options object, may be nothing) as current for this
    # thread and pushes a marker onto the stack of entered contexts. Returns nothing.
    # KNOWN ISSUE: nesting is not restored properly; clearing resets the options to nothing
    # rather than the outer level's options.
    self._load_options = load_options
    self._entered_load_context.append(True)

  def clear_load_options(self):
    # [FUNCTION DOC]
    # Resets the current options to nothing and pops one marker from the stack. Raises an
    # index error if called without a matching set. Returns nothing.
    self._load_options = None
    self._entered_load_context.pop()

  def load_options(self):
    # [FUNCTION DOC]
    # Returns the load options currently set for this thread, or nothing if none.
    return self._load_options

  def in_load_context(self):
    # [FUNCTION DOC]
    # Returns the stack of entered contexts for this thread: a non-empty list (truthy) while
    # inside a load context, an empty list (falsy) otherwise.
    # KNOWN ISSUE: despite the name suggesting a boolean, it returns the list itself.
    return self._entered_load_context


_load_context = LoadContext()


@contextlib.contextmanager
def load_context(load_options):
  # [FUNCTION DOC]
  # Context manager. Enters a load context for the current thread with the given
  # load_options for the duration of the with-block, and always clears it on exit, even if
  # the block raises. Yields nothing. Mutates the module-level thread-local state.
  _load_context.set_load_options(load_options)
  try:
    yield
  finally:
    _load_context.clear_load_options()


def get_load_options():
  # [FUNCTION DOC]
  # Returns the load options of the current thread's load context, or nothing if none is set.
  """Returns the load options under a load context."""
  return _load_context.load_options()


def in_load_context():
  # [FUNCTION DOC]
  # KNOWN ISSUE: documented as returning whether under a load context, but it returns the
  # internal stack list (truthy when inside, empty list when not), not a real boolean.
  # Reports whether the current thread is inside a load context.
  """Returns whether under a load context."""
  return _load_context.in_load_context()
