# [MODULE DOC]
# HELPER: internal state holder used by the SavedModel save machinery and option consumers.
# Tracks, per thread, whether a SavedModel save is currently in progress and which save options
# apply. Exposes a thread-local context class, a context manager that enters and leaves the
# save context, a query for whether the current thread is saving, and an accessor for the active
# save options. Import-time side effect: creates one module-level thread-local context instance
# shared by all callers; each thread sees its own independent state.
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
"""Context for building SavedModel."""

import contextlib
import threading


class SaveContext(threading.local):
  # [CLASS DOC]
  # HELPER: per-thread state container for the SavedModel save machinery.
  # Thread-local object holding two pieces of state: a flag saying whether this thread is inside
  # a save, and the save options object supplied when it entered. Because it derives from a
  # thread-local base, the constructor runs again for every new thread, so each thread starts
  # outside any save context and cannot see or change other threads' state. A single instance
  # is created at module level and used through the module functions.
  """A context for building a graph of SavedModel."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor, run once per thread that touches the instance. Takes no inputs, returns
    # nothing. Initializes the in-save flag to false and the options to nothing.
    super(SaveContext, self).__init__()
    self._in_save_context = False
    self._options = None

  def options(self):
    # [FUNCTION DOC]
    # Returns the save options stored for the current thread. Raises a value error if the
    # current thread is not inside a save context. No side effects.
    if not self.in_save_context():
      raise ValueError("Not in a SaveContext.")
    return self._options

  def enter_save_context(self, options):
    # [FUNCTION DOC]
    # Marks the current thread as inside a save context and remembers the given save options
    # object. Returns nothing. Does not check whether a context is already active (the module
    # level context manager does that check); a second call silently overwrites the options.
    self._in_save_context = True
    self._options = options

  def exit_save_context(self):
    # [FUNCTION DOC]
    # Clears the in-save flag and drops the stored options for the current thread. Returns
    # nothing; safe to call even when no context is active.
    self._in_save_context = False
    self._options = None

  def in_save_context(self):
    # [FUNCTION DOC]
    # Returns true if the current thread is inside a save context, false otherwise. Pure query.
    return self._in_save_context

_save_context = SaveContext()


@contextlib.contextmanager
def save_context(options):
  # [FUNCTION DOC]
  # Context manager that puts the current thread into the save context with the given save
  # options object for the duration of the managed block. Yields nothing. Raises a value error
  # if the thread is already inside a save context (nesting is not allowed). On exit, normal or
  # by exception, the context is always cleared. Side effect: mutates the thread-local state of
  # the module-level context.
  if in_save_context():
    raise ValueError("Already in a SaveContext.")
  _save_context.enter_save_context(options)
  try:
    yield
  finally:
    _save_context.exit_save_context()


def in_save_context():
  # [FUNCTION DOC]
  # Module-level query: returns true if the calling thread is currently inside a save context
  # entered through the context manager. Takes no inputs and has no side effects.
  """Returns whether under a save context."""
  return _save_context.in_save_context()


def get_save_options():
  # [FUNCTION DOC]
  # Returns the save options object of the calling thread's active save context. Takes no
  # inputs. Raises a value error saying the thread is not in a save context when none is active.
  """Returns the save options if under a save context."""
  return _save_context.options()
