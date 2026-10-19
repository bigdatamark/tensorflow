# [MODULE DOC]
# Global registry that associates globally unique string names with TypeSpec subclasses. Holds two
# module-level mappings (class to name, name to class) and a pattern requiring names of the form
# project name, dot, type name. Exposes a registration decorator factory, a function to get the
# registered name of a class, and a function to look a class up by name. Used by serialization
# code that needs stable names for custom type specs. Importing has no side effects beyond
# creating the empty registries; registration happens when decorated classes are defined.
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
"""Registry for custom TypeSpecs."""

import re

from tensorflow.python.types import internal


_TYPE_SPEC_TO_NAME = {}
_NAME_TO_TYPE_SPEC = {}

# Regular expression for valid TypeSpec names.
_REGISTERED_NAME_RE = re.compile(r"^(\w+\.)+\w+$")


# TODO(b/173744905) tf_export this as "tf.register_type_spec".  (And add a
# usage example to the docstring, once the API is public.)
#
# TODO(b/173744905) Update this decorator to apply to ExtensionType rather than
# TypeSpec (once we do refactoring to move to_components/from_components from
# TypeSpec to ExtensionType).
def register(name):
  # [FUNCTION DOC]
  # Builds a class decorator that registers a TypeSpec subclass under a globally unique name.
  # Input: name, a string of the form project name, dot, type name (more dotted segments allowed).
  # Output: a decorator function that returns the decorated class unchanged after registering it.
  # Raises a type error if name is not a string, and a value error if the name does not match
  # the required dotted pattern. Name validation happens immediately, not at decoration time.
  # Side effects: only when the returned decorator is applied, it updates both module-level
  # registries (class to name and name to class).
  """Decorator used to register a globally unique name for a TypeSpec subclass.

  Args:
    name: The name of the type spec.  Must be globally unique.  Must have the
      form `"{project_name}.{type_name}"`.  E.g. `"my_project.MyTypeSpec"`.

  Returns:
    A class decorator that registers the decorated class with the given name.
  """
  if not isinstance(name, str):
    raise TypeError("Expected `name` to be a string; got %r" % (name,))
  if not _REGISTERED_NAME_RE.match(name):
    raise ValueError(
        "Registered name must have the form '{project_name}.{type_name}' "
        "(e.g. 'my_project.MyTypeSpec'); got %r." % name)

  def decorator_fn(cls):
    # [FUNCTION DOC]
    # HELPER: inner decorator returned by the registration function.
    # Registers the class cls under the captured name and returns cls unchanged.
    # Raises a type error if cls is not a class derived from the internal TypeSpec base, and a
    # value error if the class is already registered or the name is already taken by any class.
    # Side effects: adds entries to both module-level registries.
    if not (isinstance(cls, type) and issubclass(cls, internal.TypeSpec)):
      raise TypeError("Expected `cls` to be a TypeSpec; got %r" % (cls,))
    if cls in _TYPE_SPEC_TO_NAME:
      raise ValueError("Class %s.%s has already been registered with name %s." %
                       (cls.__module__, cls.__name__, _TYPE_SPEC_TO_NAME[cls]))
    if name in _NAME_TO_TYPE_SPEC:
      raise ValueError("Name %s has already been registered for class %s.%s." %
                       (name, _NAME_TO_TYPE_SPEC[name].__module__,
                        _NAME_TO_TYPE_SPEC[name].__name__))
    _TYPE_SPEC_TO_NAME[cls] = name
    _NAME_TO_TYPE_SPEC[name] = cls
    return cls

  return decorator_fn


# TODO(edloper) tf_export this as "tf.get_type_spec_name" (or some similar name)
def get_name(cls):
  # [FUNCTION DOC]
  # Looks up the registered name of a TypeSpec subclass.
  # Input: cls, a class that must derive from the internal TypeSpec base.
  # Output: the registered name string.
  # Raises a type error if cls is not such a class, and a value error if it was never registered.
  # No side effects.
  """Returns the registered name for TypeSpec `cls`."""
  if not (isinstance(cls, type) and issubclass(cls, internal.TypeSpec)):
    raise TypeError("Expected `cls` to be a TypeSpec; got %r" % (cls,))
  if cls not in _TYPE_SPEC_TO_NAME:
    raise ValueError("TypeSpec %s.%s has not been registered." %
                     (cls.__module__, cls.__name__))
  return _TYPE_SPEC_TO_NAME[cls]


# TODO(edloper) tf_export this as "tf.lookup_type_spec" (or some similar name)
def lookup(name):
  # [FUNCTION DOC]
  # Finds the TypeSpec subclass registered under a name (inverse of the name getter).
  # Input: name, a string. Output: the registered class.
  # Raises a type error if name is not a string, and a value error if nothing is registered
  # under that name. No side effects.
  """Returns the TypeSpec that has been registered with name `name`."""
  if not isinstance(name, str):
    raise TypeError("Expected `name` to be a string; got %r" % (name,))
  if name not in _NAME_TO_TYPE_SPEC:
    raise ValueError("No TypeSpec has been registered with name %r" % (name,))
  return _NAME_TO_TYPE_SPEC[name]
