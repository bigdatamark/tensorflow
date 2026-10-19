# [MODULE DOC]
# Provides a small generic registry class used to register classes or functions under string
# names, typically through a decorator that calls the registration method. Each entry remembers
# the registered object and the source location (caller frame) of the registration so duplicate
# registrations can be reported with the earlier location. Other framework modules create their
# own Registry instances. Module-level constants name the two record keys. Registrations and
# lookups are logged at verbose level 1 through the TensorFlow logger. No import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Registry mechanism for "registering" classes/functions for general use.

This is typically used with a decorator that calls Register for adding
a class or function to a registry.
"""

import traceback

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


# Registry mechanism below is based on mapreduce.python.mrpython.Register.
_LOCATION_TAG = "location"
_TYPE_TAG = "type"


class Registry(object):
  # [CLASS DOC]
  # A named mapping from string keys to registered Python objects (classes or functions). State
  # is held in two slots: the registry's human readable name (used in messages) and a mapping
  # from key to a record containing the registered object and the location of the registration.
  # Create one instance per kind of thing being registered; use register to add, lookup to
  # fetch and list to enumerate. Registering the same key twice is an error. Not thread-safe.
  """Provides a registry for saving objects."""

  __slots__ = ["_name", "_registry"]

  def __init__(self, name):
    # [FUNCTION DOC]
    # Creates an empty registry. Input: name, a string used only in log and error messages to
    # say what kind of objects are registered. Returns nothing; has no side effects.
    """Creates a new registry."""
    self._name = name
    self._registry = {}

  def register(self, candidate, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if the call stack is too shallow the stored location falls back to a plain
    # tuple of placeholder strings; a later duplicate registration would then fail while
    # formatting the error, because it expects a frame object with name, filename and line
    # number attributes (a stack of fewer than one frame is practically impossible, so this is
    # mostly dead code). Also the stack index logic assumes a registration decorator call chain.
    # Adds candidate to the registry. Inputs: candidate, the object (class or function) to store;
    # name, optional string key (default none, in which case the candidate's own name attribute is
    # used; an empty string also falls back to it). Raises a key error naming the registry, the key
    # and the earlier registration's function, file and line if the key is already present.
    # Otherwise logs the registration at verbose level 1, captures the caller location from the
    # stack (the caller two levels up) and stores the object with that location. Returns nothing.
    # Side effect: mutates the registry.
    """Registers a Python object "candidate" for the given "name".

    Args:
      candidate: The candidate object to add to the registry.
      name: An optional string specifying the registry key for the candidate.
            If None, candidate.__name__ will be used.
    Raises:
      KeyError: If same name is used twice.
    """
    if not name:
      name = candidate.__name__
    if name in self._registry:
      frame = self._registry[name][_LOCATION_TAG]
      raise KeyError(
          "Registering two %s with name '%s'! "
          "(Previous registration was in %s %s:%d)" %
          (self._name, name, frame.name, frame.filename, frame.lineno))

    logging.vlog(1, "Registering %s (%s) in %s.", name, candidate, self._name)
    # stack trace is [this_function, Register(), user_function,...]
    # so the user function is #2.
    stack = traceback.extract_stack(limit=3)
    stack_index = min(2, len(stack) - 1)
    if stack_index >= 0:
      location_tag = stack[stack_index]
    else:
      location_tag = ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN")
    self._registry[name] = {_TYPE_TAG: candidate, _LOCATION_TAG: location_tag}

  def list(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a list of names, but it returns the live keys
    # view of the internal mapping (not a list); it reflects later registrations and cannot be
    # indexed.
    # Enumerates registered keys. No inputs. Returns a view of the registered names in insertion
    # order. No side effects.
    """Lists registered items.

    Returns:
      A list of names of registered objects.
    """
    return self._registry.keys()

  def lookup(self, name):
    # [FUNCTION DOC]
    # Fetches a registered object. Input: name, a string or bytes key, which is converted to a
    # text string first. Returns the object stored under that key. Raises a lookup error naming
    # the registry and key when nothing is registered under it. No side effects.
    """Looks up "name".

    Args:
      name: a string specifying the registry key for the candidate.
    Returns:
      Registered object if found
    Raises:
      LookupError: if "name" has not been registered.
    """
    name = compat.as_str(name)
    if name in self._registry:
      return self._registry[name][_TYPE_TAG]
    else:
      raise LookupError(
          "%s registry has no entry for: %s" % (self._name, name))
