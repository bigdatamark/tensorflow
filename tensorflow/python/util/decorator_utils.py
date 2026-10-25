# [MODULE DOC]
# HELPER: utilities for writing decorators that modify docstrings (used by the deprecation and
# similar decorator modules). Provides a qualified-name lookup, docstring normalization, a
# function that inserts a notice block into a docstring, a callable validator, and class
# property decorators (plain and cached). No import-time side effects other than copying the
# cached property class documentation onto the cached class property function.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Utility functions for writing decorators (which modify docstrings)."""
import sys


def get_qualified_name(function):
  # [FUNCTION DOC]
  # HELPER: gives a printable name for a function in decorator messages.
  # Input: function, a function or method object. Returns its qualified name if available;
  # otherwise (legacy Python 2 methods) the owning class name joined with the function name
  # when it has an owning class attribute, else just the function name. May raise an attribute
  # error for objects with no name.
  # Python 3
  if hasattr(function, '__qualname__'):
    return function.__qualname__

  # Python 2
  if hasattr(function, 'im_class'):
    return function.im_class.__name__ + '.' + function.__name__
  return function.__name__


def _normalize_docstring(docstring):
  # [FUNCTION DOC]
  # HELPER: cleans a docstring before a notice is inserted by add_notice_to_docstring.
  # Input: docstring, a string or empty/null value. Returns an empty string for empty input.
  # Otherwise expands tabs, strips the first line, removes the common indentation of the other
  # non-blank lines (trailing whitespace trimmed), drops leading and trailing blank lines, and
  # returns the lines joined by newlines. No side effects.
  """Normalizes the docstring.

  Replaces tabs with spaces, removes leading and trailing blanks lines, and
  removes any indentation.

  Copied from PEP-257:
  https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation

  Args:
    docstring: the docstring to normalize

  Returns:
    The normalized docstring
  """
  if not docstring:
    return ''
  # Convert tabs to spaces (following the normal Python rules)
  # and split into a list of lines:
  lines = docstring.expandtabs().splitlines()
  # Determine minimum indentation (first line doesn't count):
  # (we use sys.maxsize because sys.maxint doesn't exist in Python 3)
  indent = sys.maxsize
  for line in lines[1:]:
    stripped = line.lstrip()
    if stripped:
      indent = min(indent, len(line) - len(stripped))
  # Remove indentation (first line is special):
  trimmed = [lines[0].strip()]
  if indent < sys.maxsize:
    for line in lines[1:]:
      trimmed.append(line[indent:].rstrip())
  # Strip off trailing and leading blank lines:
  while trimmed and not trimmed[-1]:
    trimmed.pop()
  while trimmed and not trimmed[0]:
    trimmed.pop(0)
  # Return a single string:
  return '\n'.join(trimmed)


def add_notice_to_docstring(doc,
                            instructions,
                            no_doc_str,
                            suffix_str,
                            notice,
                            notice_type='Warning'):
  # [FUNCTION DOC]
  # HELPER: builds the docstring of a decorated function with a notice block inserted.
  # KNOWN ISSUE: the docstring lists only one raised error, but an unknown notice type also
  # raises a value error (checked first). It also does not say that the first element of the
  # notice list argument is modified in place (the notice type label is prepended to it), so
  # reusing the same list twice duplicates the label. The suffix is ignored when the original
  # docstring is empty.
  # Inputs: doc, original docstring (may be empty); instructions, text describing how to fix
  # the problem (omitted when empty); no_doc_str, text used when doc is empty; suffix_str, text
  # appended to the first docstring line; notice, a non-empty list of strings forming the
  # warning body; notice_type, one of Deprecated, Warning, Caution, Important, Note (default
  # Warning). Returns the new docstring: the notice is placed after the first line (after
  # keeping a blank-line gap before the body) or appended when the docstring has one line.
  # Raises a value error for an unknown notice type or an empty notice.
  """Adds a deprecation notice to a docstring.

  Args:
    doc: The original docstring.
    instructions: A string, describing how to fix the problem.
    no_doc_str: The default value to use for `doc` if `doc` is empty.
    suffix_str: Is added to the end of the first line.
    notice: A list of strings. The main notice warning body.
    notice_type: The type of notice to use. Should be one of `[Caution,
    Deprecated, Important, Note, Warning]`

  Returns:
    A new docstring, with the notice attached.

  Raises:
    ValueError: If `notice` is empty.
  """
  allowed_notice_types = ['Deprecated', 'Warning', 'Caution', 'Important',
                          'Note']
  if notice_type not in allowed_notice_types:
    raise ValueError(
        f'Unrecognized notice type. Should be one of: {allowed_notice_types}')

  if not doc:
    lines = [no_doc_str]
  else:
    lines = _normalize_docstring(doc).splitlines()
    lines[0] += ' ' + suffix_str

  if not notice:
    raise ValueError('The `notice` arg must not be empty.')

  notice[0] = f'{notice_type}: {notice[0]}'
  notice = [''] + notice + ([instructions] if instructions else [])

  if len(lines) > 1:
    # Make sure that we keep our distance from the main body
    if lines[1].strip():
      notice.append('')

    lines[1:1] = notice
  else:
    lines += notice

  return '\n'.join(lines)


def validate_callable(func, decorator_name):
  # [FUNCTION DOC]
  # HELPER: guards decorators against being applied to non-callables.
  # Inputs: func, the object being decorated; decorator_name, name of the decorator used in the
  # message. Raises a value error with a hint about decorator ordering for properties when the
  # object has no call method. Returns nothing.
  if not hasattr(func, '__call__'):
    raise ValueError(
        '%s is not a function. If this is a property, make sure'
        ' @property appears before @%s in your source code:'
        '\n\n@property\n@%s\ndef method(...)' % (
            func, decorator_name, decorator_name))


class classproperty(object):  # pylint: disable=invalid-name
  # [CLASS DOC]
  # Descriptor decorator turning a function of the class into a read-only property accessible
  # on the class itself. Stores the wrapped function; each access calls it with the owner class.
  # Nothing is cached, and the instance is ignored.
  """Class property decorator.

  Example usage:

  class MyClass(object):

    @classproperty
    def value(cls):
      return '123'

  > print MyClass.value
  123
  """

  def __init__(self, func):
    # [FUNCTION DOC]
    # Stores the function to be called on attribute access. Input: func, a function taking the
    # class as its only argument. Returns nothing.
    self._func = func

  def __get__(self, owner_self, owner_cls):
    # [FUNCTION DOC]
    # Descriptor getter. Inputs: owner_self, the instance (ignored, may be null); owner_cls,
    # the class through which the attribute was accessed. Returns the result of calling the
    # stored function with the class; recomputed on every access.
    return self._func(owner_cls)


class _CachedClassProperty(object):
  # [CLASS DOC]
  # HELPER: implementation behind the cached class property decorator.
  # Descriptor that computes a class-level value once per class on first access and keeps it
  # in a per-class cache (so subclasses compute their own value). Read-only: setting or
  # deleting raises an attribute error. Holds the wrapped function and the cache mapping from
  # classes to values.
  """Cached class property decorator.

  Transforms a class method into a property whose value is computed once
  and then cached as a normal attribute for the life of the class.  Example
  usage:

  >>> class MyClass(object):
  ...   @cached_classproperty
  ...   def value(cls):
  ...     print("Computing value")
  ...     return '<property of %s>' % cls.__name__
  >>> class MySubclass(MyClass):
  ...   pass
  >>> MyClass.value
  Computing value
  '<property of MyClass>'
  >>> MyClass.value  # uses cached value
  '<property of MyClass>'
  >>> MySubclass.value
  Computing value
  '<property of MySubclass>'

  This decorator is similar to `functools.cached_property`, but it adds a
  property to the class, not to individual instances.
  """

  def __init__(self, func):
    # [FUNCTION DOC]
    # HELPER: constructor. Input: func, a function taking the class and returning the value.
    # Stores it and creates an empty cache mapping classes to computed values.
    self._func = func
    self._cache = {}

  def __get__(self, obj, objtype):
    # [FUNCTION DOC]
    # Descriptor getter. Inputs: obj, the instance (ignored); objtype, the class accessed.
    # Computes the value by calling the stored function with the class the first time that
    # class is seen, stores it in the cache, and returns the cached value. Side effect:
    # updates the cache.
    if objtype not in self._cache:
      self._cache[objtype] = self._func(objtype)
    return self._cache[objtype]

  def __set__(self, obj, value):
    # [FUNCTION DOC]
    # Blocks assignment through instances. Inputs are ignored. Always raises an attribute
    # error saying the property is read-only. (Assignment on the class itself bypasses this.)
    raise AttributeError('property %s is read-only' % self._func.__name__)

  def __delete__(self, obj):
    # [FUNCTION DOC]
    # Blocks deletion through instances. Input is ignored. Always raises an attribute error
    # saying the property is read-only.
    raise AttributeError('property %s is read-only' % self._func.__name__)


def cached_classproperty(func):
  # [FUNCTION DOC]
  # Public decorator for cached class properties. Input: func, a function taking the class.
  # Returns a new cached class property descriptor wrapping it (value computed once per class).
  return _CachedClassProperty(func)


cached_classproperty.__doc__ = _CachedClassProperty.__doc__
