# [MODULE DOC]
# Cancellation support for eager execution. Provides a manager object wrapping a native cancellation
# manager, a process-wide slot holding the currently active manager, an accessor for that slot, and
# a context manager that installs a manager in the slot while a cancelable function runs. Import has
# no side effects other than a module-level variable initialized to nothing.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Cancellation support for eager execution."""

from tensorflow.python import pywrap_tfe


class CancellationManager(object):
  # [CLASS DOC]
  # Wrapper around a native cancellation manager used to cancel blocking computation. Holds a single
  # private handle created in the constructor; slots are used so no other attributes can be set.
  # Create one, wrap concrete functions with it, and call start_cancel to abort registered blocking
  # operations.
  """A mechanism for cancelling blocking computation."""

  __slots__ = ["_impl"]

  def __init__(self):
    # [FUNCTION DOC]
    # Creates the underlying native cancellation manager and stores its handle. Takes no arguments.
    self._impl = pywrap_tfe.TFE_NewCancellationManager()

  @property
  def is_cancelled(self):
    # [FUNCTION DOC]
    # Property. Returns true if cancellation has been requested on this manager (via start_cancel),
    # else false.
    """Returns `True` if `CancellationManager.start_cancel` has been called."""
    return pywrap_tfe.TFE_CancellationManagerIsCancelled(self._impl)

  def start_cancel(self):
    # [FUNCTION DOC]
    # Requests cancellation of blocking operations registered with this manager. Returns nothing;
    # side effect on the native manager.
    """Cancels blocking operations that have been registered with this object."""
    pywrap_tfe.TFE_CancellationManagerStartCancel(self._impl)

  def get_cancelable_function(self, concrete_function):
    # [FUNCTION DOC]
    # Wraps a concrete function so it runs while this manager is the active one. Input:
    # concrete_function, a callable. Returns a new callable accepting any positional and keyword
    # arguments that enters a manager context for this manager, calls the original function and
    # returns its result.
    def cancellable(*args, **kwargs):
      # [FUNCTION DOC]
      # Inner wrapper: sets this manager as the active one, calls the wrapped function with the
      # given arguments, returns its result, and clears the active manager afterwards (even on
      # error).
      with CancellationManagerContext(self):
        return concrete_function(*args, **kwargs)
    return cancellable

_active_context = None


def context():
  # [FUNCTION DOC]
  # Returns the currently active cancellation manager, or nothing if none is active. Reads
  # module-level state.
  return _active_context


class CancellationManagerContext:
  # [CLASS DOC]
  # Context manager that installs a cancellation manager as the module-wide active one for the
  # duration of a with block. Holds the manager. Not re-entrant: leaving the block always clears the
  # active manager rather than restoring a previous one.
  """A Python context for wrapping a cancellable ConcreteFunction."""

  def __init__(self, cancellation_manager):
    # [FUNCTION DOC]
    # Stores the cancellation manager to install on entry. Input: cancellation_manager.
    self._cancellation_manager = cancellation_manager

  def __enter__(self):
    # [FUNCTION DOC]
    # Sets the module-level active manager to the stored manager. Returns nothing (so a with target
    # receives nothing).
    global _active_context
    _active_context = self._cancellation_manager

  def __exit__(self, exc_type, exc_value, exc_tb):
    # [FUNCTION DOC]
    # Clears the module-level active manager to nothing, regardless of exception information given
    # (exception type, value, traceback). Returns nothing, so exceptions propagate.
    # KNOWN ISSUE: it resets to nothing instead of restoring the previously active manager, so
    #   nested use loses the outer manager.
    global _active_context
    _active_context = None
