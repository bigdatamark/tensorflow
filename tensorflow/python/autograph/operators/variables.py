# [MODULE DOC]
# AutoGraph variable operators: load helpers (ld, ldu) and the placeholder types Undefined and
# UndefinedReturnValue that stand in for symbols that may not be defined yet in converted code
# (needed for the functional form of loops and conditionals). Pure Python, no import-time effects.
#
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
"""Utilities used to capture Python idioms."""


def ld(v):
  # [FUNCTION DOC]
  # Load-variable operator. If v is an Undefined placeholder, reading it raises an
  # unbound-local-variable error; otherwise v is returned unchanged.
  """Load variable operator."""
  if isinstance(v, Undefined):
    return v.read()
  return v


def ldu(load_v, name):
  # [FUNCTION DOC]
  # Load-or-undefined operator. load_v is a zero-argument callable that reads the symbol; name is
  # the human-readable symbol name. Returns the callable's result, or a new Undefined placeholder
  # carrying name if the callable raises a key, attribute or name error. Other errors propagate.
  """Load variable operator that returns Undefined when failing to evaluate.

  Note: the name ("load or return undefined") is abbreviated to minimize
  the amount of clutter in generated code.

  This variant of `ld` is useful when loading symbols that may be undefined at
  runtime, such as composite symbols, and whether they are defined or not cannot
  be determined statically. For example `d['a']` is undefined when `d` is an
  empty dict.

  Args:
    load_v: Lambda that executes the actual read.
    name: Human-readable name of the symbol being read.
  Returns:
    Either the value of the symbol, or Undefined, if the symbol is not fully
    defined.
  """
  try:
    # TODO(mdan): Use locals()/globals() here.
    return load_v()
  except (KeyError, AttributeError, NameError):
    return Undefined(name)


class Undefined(object):
  # [CLASS DOC]
  # Placeholder object for a symbol that is not defined at runtime. Holds one attribute,
  # symbol_name (the identifier). Created by ldu or by converted code when initializing loop state.
  # It is permissive: any missing attribute access or item access yields the same placeholder, so
  # chained accesses stay undefined; only read() raises an error.
  """Represents an undefined symbol in Python.

  This is used to reify undefined symbols, which is required to use the
  functional form of loops.
  Example:

    while n > 0:
      n = n - 1
      s = n
    return s  # Runtime error if n == 0

  This is valid Python code and will not result in an error as long as n
  is positive. The use of this class is to stay as close to Python semantics
  as possible for staged code of this nature.

  Converted version of the above showing the possible usage of this class:

    s = Undefined('s')
    init_state = (s,)
    s = while_loop(cond, body, init_state)
    return s  # s is an instance of Undefined if the loop never runs

  Attributes:
    symbol_name: Text, identifier for the undefined symbol
  """

  __slots__ = ('symbol_name',)

  def __init__(self, symbol_name):
    # [FUNCTION DOC]
    # Constructor: stores symbol_name (text identifier of the undefined symbol).
    self.symbol_name = symbol_name

  def read(self):
    # [FUNCTION DOC]
    # Always raises an unbound-local-variable error saying the symbol is used before assignment.
    raise UnboundLocalError("'{}' is used before assignment".format(
        self.symbol_name))

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the symbol name as the text representation.
    return self.symbol_name

  def __getattribute__(self, name):
    # [FUNCTION DOC]
    # Attribute access hook. Returns the real attribute if it exists (such as the symbol name or
    # the read method); for any other name returns the placeholder itself instead of raising.
    try:
      # If it's an existing attribute, return it.
      return object.__getattribute__(self, name)
    except AttributeError:
      # Otherwise return Undefined.
      return self

  def __getitem__(self, i):
    # [FUNCTION DOC]
    # Item access hook: ignores the index i and returns the placeholder itself.
    return self


# TODO(mdan): Refactor as a RetVal object, aggregating the value and do_return.
class UndefinedReturnValue(object):
  # [CLASS DOC]
  # Marker type for a function return value that was never set. Has no state or behavior; converted
  # code creates an instance as the initial value of the return slot and checks its type later.
  """Represents a return value that is undefined."""
  pass
