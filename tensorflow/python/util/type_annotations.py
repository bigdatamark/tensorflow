# [MODULE DOC]
# Small utilities for inspecting Python generic type annotations from the standard typing
# module: predicates telling whether an annotation is a parameterized union, tuple, list or
# mapping, a predicate for forward references (handling both newer and older interpreter
# internals), and an accessor for the type arguments of a generic annotation. Pure helper module
# used by TensorFlow's type-dispatch and extension-type code; the only import-time effect is
# choosing the argument accessor depending on whether the typing module provides its own.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for accessing Python generic type annotations (typing.*)."""

import collections.abc
import typing


def is_generic_union(tp):
  # [FUNCTION DOC]
  # HELPER: type-annotation predicate for the dispatch/extension-type code.
  # Input: tp, any object. Returns true only if tp is a parameterized union annotation (its
  # origin is the union form) and not the bare union form itself; false for everything else,
  # including non-type objects. No exceptions, no side effects.
  """Returns true if `tp` is a parameterized typing.Union value."""
  return (tp is not typing.Union and
          getattr(tp, '__origin__', None) is typing.Union)


def is_generic_tuple(tp):
  # [FUNCTION DOC]
  # HELPER: type-annotation predicate for the dispatch/extension-type code.
  # Input: tp, any object. Returns true only if tp is a parameterized tuple annotation (origin is
  # the built-in tuple or the typing tuple) and not the bare tuple type or typing tuple itself.
  # False otherwise. No exceptions, no side effects.
  """Returns true if `tp` is a parameterized typing.Tuple value."""
  return (tp not in (tuple, typing.Tuple) and
          getattr(tp, '__origin__', None) in (tuple, typing.Tuple))


def is_generic_list(tp):
  # [FUNCTION DOC]
  # HELPER: type-annotation predicate for the dispatch/extension-type code.
  # Input: tp, any object. Returns true only if tp is a parameterized list annotation (origin is
  # the built-in list or the typing list) and not the bare list type or typing list itself.
  # False otherwise. No exceptions, no side effects.
  """Returns true if `tp` is a parameterized typing.List value."""
  return (tp not in (list, typing.List) and
          getattr(tp, '__origin__', None) in (list, typing.List))


def is_generic_mapping(tp):
  # [FUNCTION DOC]
  # HELPER: type-annotation predicate for the dispatch/extension-type code.
  # Input: tp, any object. Returns true only if tp is a parameterized mapping annotation (origin
  # is the abstract mapping class or the typing mapping) and not the bare forms themselves.
  # False otherwise. No exceptions, no side effects.
  """Returns true if `tp` is a parameterized typing.Mapping value."""
  return (tp not in (collections.abc.Mapping, typing.Mapping) and getattr(
      tp, '__origin__', None) in (collections.abc.Mapping, typing.Mapping))


def is_forward_ref(tp):
  # [FUNCTION DOC]
  # HELPER: type-annotation predicate for the dispatch/extension-type code.
  # Input: tp, any object. Returns true if tp is an instance of the typing forward-reference
  # class. Uses the public class when the typing module has one, falls back to the older private
  # class name, and returns false if neither exists. No exceptions, no side effects.
  """Returns true if `tp` is a typing forward reference."""
  if hasattr(typing, 'ForwardRef'):
    return isinstance(tp, typing.ForwardRef)
  elif hasattr(typing, '_ForwardRef'):
    return isinstance(tp, typing._ForwardRef)  # pylint: disable=protected-access
  else:
    return False


# Note: typing.get_args was added in Python 3.8.
if hasattr(typing, 'get_args'):
  get_generic_type_args = typing.get_args
else:
  get_generic_type_args = lambda tp: tp.__args__
