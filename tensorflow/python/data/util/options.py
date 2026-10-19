# [MODULE DOC]
# HELPER: helper module providing the machinery for tf.data option classes. Exposes the
# options base class, a factory for type-checked option properties, a function to merge options
# objects (recursively, later values win) and a factory for the graph rewrites record type.
# Used by the dataset options definitions. No import-time side effects beyond imports.
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
"""Utilities for tf.data options."""

import collections

from absl import logging


def _internal_attr_name(name):
  # [FUNCTION DOC]
  # HELPER: maps an option name to its private attribute name by prefixing an underscore.
  # Returns the new string. Appears unused within this module.
  return "_" + name


class OptionsBase:
  # [CLASS DOC]
  # Base class for sets of tf.data options. Subclasses declare options as type-checked
  # properties created with the option factory. Holds a private storage mapping of option names
  # to values and a mutable flag. Overrides equality (compares all set options), inequality and
  # attribute assignment (rejects unknown attributes and, once made immutable, all assignment).
  # Provides abstract protocol buffer conversion hooks for subclasses.
  """Base class for representing a set of tf.data options.

  Attributes:
    _options: Stores the option values.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor: creates an empty option storage mapping and marks the object mutable, writing
    # the attributes directly because attribute assignment is overridden. No arguments.
    # NOTE: Cannot use `self._options` here as we override `__setattr__`
    object.__setattr__(self, "_options", {})
    object.__setattr__(self, "_mutable", True)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality: returns the not-implemented marker if other is not an instance of this class;
    # otherwise compares every option name set on either object (reading unset ones yields the
    # default) and returns true only if all values match. Reading may populate defaults.
    if not isinstance(other, self.__class__):
      return NotImplemented
    for name in set(self._options) | set(other._options):  # pylint: disable=protected-access
      if getattr(self, name) != getattr(other, name):
        return False
    return True

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Inequality: the negation of equality when other is an instance of this class (or subclass);
    # otherwise returns the not-implemented marker so the comparison falls back.
    if isinstance(other, self.__class__):
      return not self.__eq__(other)
    else:
      return NotImplemented

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Guarded attribute assignment. Raises a value error if the object was made immutable (the
    # message says mutating the returned options has no effect and to use the dataset method for
    # applying options). Raises an attribute error if the name is not an existing attribute
    # (only declared option properties and existing attributes may be set). Otherwise sets it,
    # which for option properties runs the type-checking setter.
    if not self._mutable:
      raise ValueError("Mutating `tf.data.Options()` returned by "
                       "`tf.data.Dataset.options()` has no effect. Use "
                       "`tf.data.Dataset.with_options(options)` to set or "
                       "update dataset options.")
    if hasattr(self, name):
      object.__setattr__(self, name, value)
    else:
      raise AttributeError("Cannot set the property {} on {}.".format(
          name,
          type(self).__name__))

  def _set_mutable(self, mutable):
    # [FUNCTION DOC]
    # HELPER: turns attribute assignment on or off for this object (boolean mutable), bypassing
    # the guarded assignment hook. Returns nothing.
    """Change the mutability property to `mutable`."""
    object.__setattr__(self, "_mutable", mutable)

  def _to_proto(self):
    # [FUNCTION DOC]
    # Abstract hook: subclasses convert their options into a protocol buffer message. The base
    # implementation always raises a not-implemented error naming the class.
    """Convert options to protocol buffer."""
    raise NotImplementedError("{}._to_proto()".format(type(self).__name__))

  def _from_proto(self, pb):
    # [FUNCTION DOC]
    # Abstract hook: subclasses convert a protocol buffer message into option values. The base
    # implementation always raises a not-implemented error naming the class. pb is the message.
    """Convert protocol buffer to options."""
    raise NotImplementedError("{}._from_proto()".format(type(self).__name__))


# Creates a namedtuple with three keys for optimization graph rewrites settings.
def graph_rewrites():
  # [FUNCTION DOC]
  # HELPER: factory for the record type describing graph rewrite settings. Takes no arguments
  # and returns a new named tuple type called graph rewrites with fields enabled, disabled and
  # default (a new type is created on every call).
  return collections.namedtuple("GraphRewrites",
                                ["enabled", "disabled", "default"])


def create_option(name, ty, docstring, default_factory=lambda: None):
  # [FUNCTION DOC]
  # HELPER: builds a property for use in options classes. name is the key under which the value
  # is stored; ty is the required type of assigned values; docstring becomes the property's
  # documentation; default_factory is a callable without arguments producing the default (by
  # default returns nothing/null). Returns a property whose getter lazily stores and returns the
  # default and whose setter validates the type, raising a type error on mismatch.
  """Creates a type-checked property.

  Args:
    name: The name to use.
    ty: The type to use. The type of the property will be validated when it
      is set.
    docstring: The docstring to use.
    default_factory: A callable that takes no arguments and returns a default
      value to use if not set.

  Returns:
    A type-checked property.
  """

  def get_fn(option):
    # [FUNCTION DOC]
    # HELPER: getter of the created property. Takes the options object. If the option was never
    # set, stores the default produced by the default factory in the object's option storage
    # (a side effect), then returns the stored value.
    # pylint: disable=protected-access
    if name not in option._options:
      option._options[name] = default_factory()
    return option._options.get(name)

  def set_fn(option, value):
    # [FUNCTION DOC]
    # HELPER: setter of the created property. Takes the options object and the new value. Raises
    # a type error if the value is not an instance of the declared type; otherwise stores it in
    # the object's option storage. Returns nothing.
    if not isinstance(value, ty):
      raise TypeError(
          "Property \"{}\" must be of type {}, got: {} (type: {})".format(
              name, ty, value, type(value)))
    option._options[name] = value  # pylint: disable=protected-access

  return property(get_fn, set_fn, None, docstring)


def merge_options(*options_list):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists only a type error, but an empty argument list raises a
  # value error ("at least one options should be provided"). The docstring also omits the special
  # handling of the framework type option, whose list values are concatenated rather than
  # overridden.
  # Merges one or more options objects of the same class into a new instance of that class.
  # Takes the options objects as positional arguments. Raises a value error if none is given; a
  # type error if any object is not an instance of the first object's type, or if the first
  # object does not derive from the options base class. Walks the explicitly set options of each
  # input in order: values equal to the default are skipped; if the result still has the default,
  # the value is copied; if the current value is itself an options object, the two are merged
  # recursively; for the framework type option the lists are concatenated; otherwise a warning
  # is logged about the changed value and the later value wins. Returns the new merged object;
  # inputs are not modified (though reading unset options may populate defaults inside them).
  """Merges the given options, returning the result as a new options object.

  The input arguments are expected to have a matching type that derives from
  `tf.data.OptionsBase` (and thus each represent a set of options). The method
  outputs an object of the same type created by merging the sets of options
  represented by the input arguments.

  If an option is set to different values by different options objects, the
  result will match the setting of the options object that appears in the input
  list last.

  If an option is an instance of `tf.data.OptionsBase` itself, then this method
  is applied recursively to the set of options represented by this option.

  Args:
    *options_list: options to merge

  Raises:
    TypeError: if the input arguments are incompatible or not derived from
      `tf.data.OptionsBase`

  Returns:
    A new options object which is the result of merging the given options.
  """
  if len(options_list) < 1:
    raise ValueError("At least one options should be provided")
  result_type = type(options_list[0])

  for options in options_list:
    if not isinstance(options, result_type):
      raise TypeError(
          "Could not merge incompatible options of type {} and {}.".format(
              type(options), result_type))

  if not isinstance(options_list[0], OptionsBase):
    raise TypeError(
        "All options to be merged should inherit from `OptionsBase` but found "
        "option of type {} which does not.".format(type(options_list[0])))

  default_options = result_type()
  result = result_type()
  for options in options_list:
    # Iterate over all set options and merge them into the result.
    for name in options._options:  # pylint: disable=protected-access
      this = getattr(result, name)
      that = getattr(options, name)
      default = getattr(default_options, name)
      if that == default:
        continue
      elif this == default:
        setattr(result, name, that)
      elif isinstance(this, OptionsBase):
        setattr(result, name, merge_options(this, that))
      elif name == "framework_type":
        # Since, `framework_type`` is a repeated string field (list), the merged
        # result will be a combined list.
        setattr(result, name, this+that)
      elif this != that:
        logging.warning("Changing the value of option %s from %r to %r.", name,
                        this, that)
        setattr(result, name, that)
  return result
