# [MODULE DOC]
# Import router for the absl command-line flags library. Re-exports everything from absl flags
# (the define functions, the flag registry, errors, etc.), then wraps the string, boolean,
# integer and float define functions so the old keyword names (flag name, default value,
# docstring) are still accepted with a deprecation warning, and replaces the global flag
# registry with a wrapper that lazily parses the process arguments on first flag read.
# Import-time side effects: marks this module as not owning key flags, rebinds the exported
# define functions and the global flag registry name. Public-facing compatibility layer.

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

"""Import router for absl.flags. See https://github.com/abseil/abseil-py."""
import logging as _logging
import sys as _sys

# go/tf-wildcard-import

from absl.flags import *  # pylint: disable=wildcard-import

from tensorflow.python.util import tf_decorator


# Since we wrap absl.flags DEFINE functions, we need to declare this module
# does not affect key flags.
disclaim_key_flags()  # pylint: disable=undefined-variable


_RENAMED_ARGUMENTS = {
    'flag_name': 'name',
    'default_value': 'default',
    'docstring': 'help',
}


def _wrap_define_function(original_function):
  # [FUNCTION DOC]
  # HELPER: used at import time to wrap the absl define functions for backward compatibility.
  # Takes an absl define function and returns a decorated replacement (built so metadata of the
  # original is preserved and the decorator chain is recorded) that renames legacy keyword
  # arguments before delegating. No side effects at wrap time.
  """Wraps absl.flags's define functions so tf.flags accepts old names."""

  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: the replacement define function created by the wrapping helper.
    # KNOWN ISSUE: if both an old and the new spelling of a keyword are passed, the old value
    # silently overwrites the new one without any error.
    # Accepts any positional and keyword arguments. For each legacy keyword (flag name, default
    # value, docstring) present, removes it and re-adds its value under the new keyword (name,
    # default, help). Logs a deprecation warning once per call when any old
    # keyword was used. Returns whatever the original define function returns.
    """Wrapper function that turns old keyword names to new ones."""
    has_old_names = False
    for old_name, new_name in _RENAMED_ARGUMENTS.items():
      if old_name in kwargs:
        has_old_names = True
        value = kwargs.pop(old_name)
        kwargs[new_name] = value
    if has_old_names:
      _logging.warning(
          'Use of the keyword argument names (flag_name, default_value, '
          'docstring) is deprecated, please use (name, default, help) instead.')
    return original_function(*args, **kwargs)

  return tf_decorator.make_decorator(original_function, wrapper)


class _FlagValuesWrapper:
  # [CLASS DOC]
  # HELPER: private proxy installed as the public flag registry in this module.
  # Holds the wrapped absl flag registry in its instance dictionary under a private key and
  # forwards attribute access, item access, deletion, iteration, length, string conversion,
  # directory listing and calling to it. The only behavioral difference: reading a missing
  # attribute first parses the process command-line arguments if the registry has not been parsed
  # yet, instead of raising an unparsed-flags error. Created once at import time.
  """Wrapper class for absl.flags.FLAGS.

  The difference is that tf.flags.FLAGS implicitly parses flags with sys.argv
  when accessing the FLAGS values before it's explicitly parsed,
  while absl.flags.FLAGS raises an exception.
  """

  def __init__(self, flags_object):
    # [FUNCTION DOC]
    # Constructor. Stores the given absl flag registry object in the instance dictionary under the
    # private wrapped key (bypassing the forwarding attribute setter). Returns nothing.
    self.__dict__['__wrapped'] = flags_object

  def __getattribute__(self, name):
    # [FUNCTION DOC]
    # Attribute lookup hook. For the instance dictionary itself, uses the normal lookup so the
    # wrapped registry can be reached; for every other name, delegates to the wrapped registry's
    # own lookup. If that raises an attribute error, Python then falls back to the missing
    # attribute hook below. Input: attribute name. Returns the attribute value.
    if name == '__dict__':
      return super().__getattribute__(name)
    return self.__dict__['__wrapped'].__getattribute__(name)

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Fallback lookup used when normal lookup fails (typically for flag names). If the wrapped
    # registry has not been parsed, parses it with the process command-line arguments (side
    # effect: flags become parsed, and parse errors may be raised). Then returns the flag value
    # via the wrapped registry's own missing-attribute handling, which raises an attribute error
    # for unknown flags.
    wrapped = self.__dict__['__wrapped']
    # To maintain backwards compatibility, implicitly parse flags when reading
    # a flag.
    if not wrapped.is_parsed():
      wrapped(_sys.argv)
    return wrapped.__getattr__(name)

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Forwards attribute assignment to the wrapped registry, i.e. sets the named flag's value.
    # Inputs: flag name and new value. Returns whatever the wrapped setter returns (nothing).
    return self.__dict__['__wrapped'].__setattr__(name, value)

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # Forwards attribute deletion to the wrapped registry, removing the named flag. Raises an
    # attribute error from the wrapped registry if the flag is unknown.
    return self.__dict__['__wrapped'].__delattr__(name)

  def __dir__(self):
    # [FUNCTION DOC]
    # Returns the directory listing of the wrapped registry (its flag names), not the proxy's.
    return self.__dict__['__wrapped'].__dir__()

  def __getitem__(self, name):
    # [FUNCTION DOC]
    # Returns the flag object registered under the given name from the wrapped registry; does not
    # trigger implicit parsing. Raises a key error for unknown names.
    return self.__dict__['__wrapped'].__getitem__(name)

  def __setitem__(self, name, flag):
    # [FUNCTION DOC]
    # Registers a flag object under the given name in the wrapped registry (side effect: mutates
    # the registry). Errors such as duplicate-name conflicts come from the wrapped registry.
    return self.__dict__['__wrapped'].__setitem__(name, flag)

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of flags in the wrapped registry.
    return self.__dict__['__wrapped'].__len__()

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the flag names of the wrapped registry.
    return self.__dict__['__wrapped'].__iter__()

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the help-style text representation of the wrapped registry.
    return self.__dict__['__wrapped'].__str__()

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Calls the wrapped registry with the given arguments, which parses a command-line argument
    # list (first item is the program name) and returns the unparsed remainder. Side effect:
    # sets flag values and marks the registry parsed; may raise a flag parsing error.
    return self.__dict__['__wrapped'].__call__(*args, **kwargs)


# pylint: disable=invalid-name,used-before-assignment
# absl.flags APIs use `default` as the name of the default value argument.
# Allow the following functions continue to accept `default_value`.
DEFINE_string = _wrap_define_function(DEFINE_string)
DEFINE_boolean = _wrap_define_function(DEFINE_boolean)
DEFINE_bool = DEFINE_boolean
DEFINE_float = _wrap_define_function(DEFINE_float)
DEFINE_integer = _wrap_define_function(DEFINE_integer)
# pylint: enable=invalid-name,used-before-assignment

FLAGS = _FlagValuesWrapper(FLAGS)  # pylint: disable=used-before-assignment
