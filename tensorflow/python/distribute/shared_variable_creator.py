# [MODULE DOC]
# Utility that lets per-device variable creator functions share variables: the creator for the
# first device really creates variables and records them in a shared store keyed by a canonical
# (de-uniquified) name; creators for other devices hand back the recorded variables in order.
# Exposes the creator factory function plus a private name canonicalization helper. Defines two
# compiled patterns at import time; no other side effects.
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
"""Utility to re-use variables created on first device on subsequent devices."""

import re

_VARIABLE_UNIQUIFYING_REGEX = re.compile(r"_\d/")
_VARIABLE_UNIQUIFYING_REGEX_AT_END = re.compile(r"_\d$")


def _canonicalize_variable_name(name):
  # [FUNCTION DOC]
  # HELPER: for the creator functions, to match variables that differ only by uniquifying
  # suffixes. Input: name, a string or nothing. Returns the default name "Variable" for nothing;
  # otherwise removes every underscore-digit suffix that precedes a slash (keeping the slash)
  # and any underscore-digit suffix at the very end, and returns the result.
  # If no name is specified, uses default name "Variable".
  if name is None:
    return "Variable"
  # Replace all instances of "_<num>/" with "/"
  name = _VARIABLE_UNIQUIFYING_REGEX.sub("/", name)
  # Replace any instances of "_<num>" at the end of the string with ""
  name = _VARIABLE_UNIQUIFYING_REGEX_AT_END.sub("", name)
  return name


def make_fn(shared_variable_store, device_id):
  # [FUNCTION DOC]
  # Factory for a variable creator function for one device. Inputs: shared_variable_store (mapping
  # from canonical variable names to lists of variables, mutated by the device-zero creator) and
  # device_id (integer; asserted to be an integer). Returns the creating function for device
  # zero and the re-using function for every other device. Keeps a per-call access counter
  # mapping canonical names to how many were already re-used.
  """Construct the variable creator function for device `device_id`.

  Constructs custom variable creator functions for the given device.
  On first device (device_id == 0), it creates the variable using the
  `next_creator`, and stores it in the provided `shared_variable_store`.
  On all other devices (device_id > 0), it tries to re-use the variable
  already created with the same name. If no such variable exists, it throws an
  error.
  Additionally, we de-uniquify variable names before checking for matches. This
  helps re-use variables which are intended to be the same but have different
  names due to variable uniquification happening upstream. Since this might
  mean we may have multiple variables with the same canonical name, we store
  them in a list per canonical name and return them in the same order as well.

  Args:
    shared_variable_store: A dictionary that we will use to store variables
      created on the first device, and re-used by creators for other devices.
    device_id: Integer index of the device whose creator should be
      constructed.

  Returns:
    An appropriate creator function based on device_id.

  """
  variable_scope_access_index = {}
  assert isinstance(device_id, int)

  def create_new_variable(next_creator, **kwargs):
    # [FUNCTION DOC]
    # HELPER: creator for device zero. Calls the next creator with all keyword arguments to build
    # the variable, appends it to the list stored under its canonical name in the shared store
    # (creating the list when missing) and returns the variable.
    """Create the variable using `next_creator` and store it."""
    canonical_name = _canonicalize_variable_name(kwargs.get("name"))
    v = next_creator(**kwargs)

    if canonical_name not in shared_variable_store:
      shared_variable_store[canonical_name] = []
    shared_variable_store[canonical_name].append(v)
    return v

  def reuse_variable(next_creator, **kwargs):
    # [FUNCTION DOC]
    # HELPER: creator for devices other than zero. Ignores the next creator. Looks up the
    # canonical form of the requested name in the shared store and returns the next not-yet-used
    # variable in that list, advancing a per-name counter. Raises a runtime error mentioning the
    # name and device if no such name or no remaining variable exists. Does not check shape or
    # dtype compatibility.
    """Re-use existing variable from store with same name (in order)."""
    del next_creator
    name = kwargs.get("name")
    canonical_name = _canonicalize_variable_name(name)

    try:
      variable_index = variable_scope_access_index.get(canonical_name, 0)
      v = shared_variable_store[canonical_name][variable_index]
      # TODO(priyag): Make this variable re-use more robust by adding checks
      # that the requested shape and dtype match the existing variable.
      variable_scope_access_index[canonical_name] = variable_index + 1
      return v
    except (KeyError, IndexError):
      raise RuntimeError(
          "Tried to create variable {} with mismatching name on device {}".
          format(name, device_id))

  if device_id == 0:
    return create_new_variable
  else:
    return reuse_variable
