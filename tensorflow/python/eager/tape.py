# [MODULE DOC]
# Thin Python helpers around the native gradient tape set used by eager execution. Exposes a
# wrapper class for a native tape object plus functions to create and push a new tape, push or pop
# an existing one, mark tensors or variables as watched, and notify the tape stack that variables
# were accessed. Keeps one module-level hook (a variable resolver, by default one that returns the
# variable itself) that distribution strategies may replace to map a variable to the list of
# underlying variables. Has no import-time side effects besides defining that hook.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Gradient tape utilities."""

from tensorflow.python import pywrap_tfe


class Tape(object):
  # [CLASS DOC]
  # Lightweight Python handle for a native gradient tape. Uses slots and holds a single private
  # attribute, the native tape object, created by the push-new-tape function and passed to the
  # other functions in this module. Offers a method to list the variables the tape has watched.
  """Represents a gradient propagation trace."""

  __slots__ = ["_tape"]

  def __init__(self, tape):
    # [FUNCTION DOC]
    # Constructor. Input: the native tape object to wrap; stored unchanged as the private
    # attribute. Returns nothing and has no other side effects.
    self._tape = tape

  def watched_variables(self):
    # [FUNCTION DOC]
    # Asks the native layer for the variables watched by the wrapped tape and returns them
    # (a sequence of variable objects). Takes no arguments; does not modify the tape.
    return pywrap_tfe.TFE_Py_TapeWatchedVariables(self._tape)


def push_new_tape(persistent=False, watch_accessed_variables=True):
  # [FUNCTION DOC]
  # Creates a new native tape and pushes it onto the global tape stack (the native creation call
  # registers it). Inputs: persistent (optional boolean, default false) allows the tape to be used
  # for several gradient computations; watch_accessed_variables (optional boolean, default true)
  # makes the tape automatically watch trainable variables as they are read. Returns a wrapper
  # object for the new tape. Side effect: modifies the global tape stack.
  """Pushes a new tape onto the tape stack."""
  tape = pywrap_tfe.TFE_Py_TapeSetNew(persistent, watch_accessed_variables)
  return Tape(tape)


def push_tape(tape):
  # [FUNCTION DOC]
  # Pushes an already created tape wrapper back onto the global tape stack. Input: a tape wrapper
  # whose native object is added. Returns nothing. Side effect: modifies the global tape stack.
  """Pushes an existing tape onto the tape stack."""
  pywrap_tfe.TFE_Py_TapeSetAdd(tape._tape)  # pylint: disable=protected-access


def watch(tape, tensor):
  # [FUNCTION DOC]
  # Tells the given tape to record operations on the given tensor so gradients can later be taken
  # with respect to it. Inputs: a tape wrapper and a tensor. Returns nothing; mutates the native
  # tape's watched set.
  """Marks this tensor to be watched by the given tape."""
  pywrap_tfe.TFE_Py_TapeWatch(tape._tape, tensor)  # pylint: disable=protected-access


def default_get_variables(variable):
  # [FUNCTION DOC]
  # HELPER: default variable resolver for the module-level hook used when watching variables.
  # Input: a variable. Returns a list containing just that variable (identity mapping).
  return [variable]

# Gets a list of changed variables. Can be overridden using
# register_variables_override. An example of overriding is for getting the
# variables within a distributed context.
_variables_override = default_get_variables


def register_watched_variable_resolver(resolver):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the explanatory comment above the module-level hook refers to a function called
  # register variables override, which does not exist; this function is the actual registration
  # point. It also enforces single registration with an assert, which is skipped under optimized
  # Python and gives no message.
  # Replaces the module-level variable resolver, used by the watch and access functions to expand
  # one variable into the list of variables to watch (for example per-replica components in a
  # distribution strategy). Input: a callable taking a variable and returning a list of variables.
  # Returns nothing. Raises an assertion error if a resolver was already registered. Side effect:
  # changes global module state.
  """Registers the resolver to be used to get the list of variables to watch.

  Args:
    resolver: callable, takes a Variable and returns a list of Variables that
      shall be watched.
  """
  global _variables_override
  assert _variables_override is default_get_variables
  _variables_override = resolver


def watch_variable(tape, variable):
  # [FUNCTION DOC]
  # Marks a variable as watched by the given tape. Inputs: a tape wrapper and a variable. The
  # variable is first expanded by the registered resolver (default: itself); each resulting
  # variable is added to the native tape's watched set and also reported to the variable watcher
  # as accessed. Returns nothing; mutates native tape and watcher state.
  """Marks this variable to be watched by the given tape."""
  variables = _variables_override(variable)
  for var in variables:
    pywrap_tfe.TFE_Py_TapeWatchVariable(tape._tape, var)  # pylint: disable=protected-access
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def variable_accessed(variable):
  # [FUNCTION DOC]
  # Reports to every tape on the stack (and to the variable watcher) that a variable was read so
  # tapes that auto-watch variables can start watching it. Input: a variable, expanded first with
  # the registered resolver. No trainability check is made here, unlike the plural variant.
  # Returns nothing; mutates native tape state.
  """Notifies all tapes in the stack that a variable has been accessed.

  Args:
    variable: variable to be watched.
  """
  variables = _variables_override(variable)
  for var in variables:
    pywrap_tfe.TFE_Py_TapeVariableAccessed(var)
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def variables_accessed(variables):
  # [FUNCTION DOC]
  # Plural version of the access notification. Input: an iterable of variables. Keeps only the
  # trainable ones, expands each with the registered resolver, and then reports every resulting
  # variable to all tapes and to the variable watcher as accessed. Returns nothing; mutates native
  # tape state. Non-trainable variables are ignored.
  """Notifies all tapes in the stack that variables have been accessed.

  Only trainable variables are marked as accessed.

  Args:
    variables: iterable of variables to mark as accessed.
  """
  accessed = []
  for variable in variables:
    if variable.trainable:
      accessed.extend(_variables_override(variable))

  for var in accessed:
    pywrap_tfe.TFE_Py_TapeVariableAccessed(var)
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def pop_tape(tape):
  # [FUNCTION DOC]
  # Removes the given tape from the global tape stack. Input: a tape wrapper. Returns nothing.
  # Side effect: modifies the global tape stack.
  """Pops the given tape in the stack."""
  pywrap_tfe.TFE_Py_TapeSetRemove(tape._tape)  # pylint: disable=protected-access
