# [MODULE DOC]
# HELPER: Helper module offering decorator-aware replacements for the standard introspection
# module, used by Keras code that must see through TensorFlow decorators.
# Most functions first strip TensorFlow decorators from the given object (using the decorator
# utility) and then delegate to the standard introspection function; the argument-spec functions
# prefer the signature recorded on the outermost decorator. Also defines compatibility aliases
# for the argument-spec tuple types at import time (reusing the standard ones when they exist,
# otherwise a named tuple). No other import-time side effects.
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
"""TFDecorator-aware replacements for the inspect module."""

import collections
import functools
import inspect as _inspect

from tensorflow.python.util import tf_decorator

try:
  ArgSpec = _inspect.ArgSpec
except:
  pass


if hasattr(_inspect, 'FullArgSpec'):
  FullArgSpec = _inspect.FullArgSpec  # pylint: disable=invalid-name
else:
  FullArgSpec = collections.namedtuple('FullArgSpec', [
      'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
      'annotations'
  ])


def _convert_maybe_argspec_to_fullargspec(argspec):
  # [FUNCTION DOC]
  # HELPER: Normalizes argument-spec results for the argument-spec functions of this module.
  # Takes either a full argument spec or an old-style argument spec. A full spec is returned
  # unchanged; an old-style one is converted to a full spec (keywords becomes the var-keyword
  # field, with no keyword-only arguments, no keyword-only defaults and empty annotations).
  if isinstance(argspec, FullArgSpec):
    return argspec
  return FullArgSpec(
      args=argspec.args,
      varargs=argspec.varargs,
      varkw=argspec.keywords,
      defaults=argspec.defaults,
      kwonlyargs=[],
      kwonlydefaults=None,
      annotations={})

if hasattr(_inspect, 'getfullargspec'):
  _getfullargspec = _inspect.getfullargspec  # pylint: disable=invalid-name

  def _getargspec(target):
    # [FUNCTION DOC]
    # HELPER: Argument-spec fallback used by the public getargspec on modern Python.
    # Gets the full argument spec of target through this module's decorator-aware function and
    # repackages args, varargs, var-keyword and defaults into an old-style argspec when that type
    # exists in the interpreter, or into a full spec with empty keyword-only data otherwise.
    # Keyword-only arguments and annotations are dropped. Input: any callable. Returns the spec.
    """A python3 version of getargspec.

    Calls `getfullargspec` and assigns args, varargs,
    varkw, and defaults to a python 2/3 compatible `ArgSpec`.

    The parameter name 'varkw' is changed to 'keywords' to fit the
    `ArgSpec` struct.

    Args:
      target: the target object to inspect.

    Returns:
      An ArgSpec with args, varargs, keywords, and defaults parameters
      from FullArgSpec.
    """
    fullargspecs = getfullargspec(target)
    if hasattr(_inspect, 'ArgSpec'):
      argspecs = ArgSpec(
          args=fullargspecs.args,
          varargs=fullargspecs.varargs,
          keywords=fullargspecs.varkw,
          defaults=fullargspecs.defaults)
    else:
      argspecs = FullArgSpec(
          args=fullargspecs.args,
          varargs=fullargspecs.varargs,
          varkw=fullargspecs.varkw,
          defaults=fullargspecs.defaults,
          kwonlyargs=[],
          kwonlydefaults=None,
          annotations={})
    return argspecs
else:
  _getargspec = _inspect.getargspec

  def _getfullargspec(target):
    # [FUNCTION DOC]
    # HELPER: Legacy fallback, only defined when the interpreter lacks the standard full argument
    # spec function. Takes a callable and returns its old-style spec converted to a full spec.
    """A python2 version of getfullargspec.

    Args:
      target: the target object to inspect.

    Returns:
      A FullArgSpec with empty kwonlyargs, kwonlydefaults and annotations.
    """
    return _convert_maybe_argspec_to_fullargspec(getargspec(target))


def currentframe():
  # [FUNCTION DOC]
  # KNOWN ISSUE: Builds the entire call stack with source context just to pick one frame, which
  # is slow; and the frame returned is the caller of this function (index one of the stack), as
  # intended by the standard function, but the docstring's "TFDecorator-aware" claim is not
  # reflected in any decorator handling here.
  # Returns the frame object of whoever called this function. No inputs, no side effects.
  """TFDecorator-aware replacement for inspect.currentframe."""
  return _inspect.stack()[1][0]


def getargspec(obj):
  # [FUNCTION DOC]
  # Decorator-aware old-style argument spec. Input obj is a function, partial object, class or
  # callable object. Partial objects are handled by the partial-specific helper. Otherwise
  # decorators are unwrapped; if any decorator recorded an argspec, the first (outermost) one is
  # returned. Else the spec of the unwrapped target is tried, then for classes the constructor
  # initializer and the instance creator, finally the call method of the target's type.
  # Returns an argument spec. Raises a value error if the signature cannot be expressed as an
  # old-style spec (for example keyword-only arguments) and a type error for unsupported objects.
  """TFDecorator-aware replacement for `inspect.getargspec`.

  Note: `getfullargspec` is recommended as the python 2/3 compatible
  replacement for this function.

  Args:
    obj: A function, partial function, or callable object, possibly decorated.

  Returns:
    The `ArgSpec` that describes the signature of the outermost decorator that
    changes the callable's signature, or the `ArgSpec` that describes
    the object if not decorated.

  Raises:
    ValueError: When callable's signature can not be expressed with
      ArgSpec.
    TypeError: For objects of unsupported types.
  """
  if isinstance(obj, functools.partial):
    return _get_argspec_for_partial(obj)

  decorators, target = tf_decorator.unwrap(obj)

  spec = next((d.decorator_argspec
               for d in decorators
               if d.decorator_argspec is not None), None)
  if spec:
    return spec

  try:
    # Python3 will handle most callables here (not partial).
    return _getargspec(target)
  except TypeError:
    pass

  if isinstance(target, type):
    try:
      return _getargspec(target.__init__)
    except TypeError:
      pass

    try:
      return _getargspec(target.__new__)
    except TypeError:
      pass

  # The `type(target)` ensures that if a class is received we don't return
  # the signature of its __call__ method.
  return _getargspec(type(target).__call__)


def _get_argspec_for_partial(obj):
  # [FUNCTION DOC]
  # HELPER: Partial-object support for getargspec.
  # Computes the argument spec of a partially applied function: the arguments bound positionally
  # are removed, and keyword bindings become defaults that override the wrapped function's own
  # defaults. Raises a value error if a bound keyword is unknown and the function accepts no
  # arbitrary keywords, or if an argument without default follows one with a default (not
  # expressible in an old-style spec). Returns the spec, with defaults null if none exist.
  """Implements `getargspec` for `functools.partial` objects.

  Args:
    obj: The `functools.partial` object
  Returns:
    An `inspect.ArgSpec`
  Raises:
    ValueError: When callable's signature can not be expressed with
      ArgSpec.
  """
  # When callable is a functools.partial object, we construct its ArgSpec with
  # following strategy:
  # - If callable partial contains default value for positional arguments (ie.
  # object.args), then final ArgSpec doesn't contain those positional arguments.
  # - If callable partial contains default value for keyword arguments (ie.
  # object.keywords), then we merge them with wrapped target. Default values
  # from callable partial takes precedence over those from wrapped target.
  #
  # However, there is a case where it is impossible to construct a valid
  # ArgSpec. Python requires arguments that have no default values must be
  # defined before those with default values. ArgSpec structure is only valid
  # when this presumption holds true because default values are expressed as a
  # tuple of values without keywords and they are always assumed to belong to
  # last K arguments where K is number of default values present.
  #
  # Since functools.partial can give default value to any argument, this
  # presumption may no longer hold in some cases. For example:
  #
  # def func(m, n):
  #   return 2 * m + n
  # partialed = functools.partial(func, m=1)
  #
  # This example will result in m having a default value but n doesn't. This is
  # usually not allowed in Python and can not be expressed in ArgSpec correctly.
  #
  # Thus, we must detect cases like this by finding first argument with default
  # value and ensures all following arguments also have default values. When
  # this is not true, a ValueError is raised.

  n_prune_args = len(obj.args)
  partial_keywords = obj.keywords or {}

  args, varargs, keywords, defaults = getargspec(obj.func)

  # Pruning first n_prune_args arguments.
  args = args[n_prune_args:]

  # Partial function may give default value to any argument, therefore length
  # of default value list must be len(args) to allow each argument to
  # potentially be given a default value.
  no_default = object()
  all_defaults = [no_default] * len(args)

  if defaults:
    all_defaults[-len(defaults):] = defaults

  # Fill in default values provided by partial function in all_defaults.
  for kw, default in partial_keywords.items():
    if kw in args:
      idx = args.index(kw)
      all_defaults[idx] = default
    elif not keywords:
      raise ValueError('Function does not have **kwargs parameter, but '
                       'contains an unknown partial keyword.')

  # Find first argument with default value set.
  first_default = next(
      (idx for idx, x in enumerate(all_defaults) if x is not no_default), None)

  # If no default values are found, return ArgSpec with defaults=None.
  if first_default is None:
    return ArgSpec(args, varargs, keywords, None)

  # Checks if all arguments have default value set after first one.
  invalid_default_values = [
      args[i] for i, j in enumerate(all_defaults)
      if j is no_default and i > first_default
  ]

  if invalid_default_values:
    raise ValueError('Some arguments %s do not have default value, but they '
                     'are positioned after those with default values. This can '
                     'not be expressed with ArgSpec.' % invalid_default_values)

  return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))


def getfullargspec(obj):
  # [FUNCTION DOC]
  # Decorator-aware full argument spec. Input obj is a callable. Unwraps decorators; the first
  # decorator carrying a recorded argspec supplies the result (converted to a full spec if needed).
  # Otherwise returns the full spec of the unwrapped target. Errors from the standard
  # introspection (for example a type error for unsupported objects) propagate.
  """TFDecorator-aware replacement for `inspect.getfullargspec`.

  This wrapper emulates `inspect.getfullargspec` in[^)]* Python2.

  Args:
    obj: A callable, possibly decorated.

  Returns:
    The `FullArgSpec` that describes the signature of
    the outermost decorator that changes the callable's signature. If the
    callable is not decorated, `inspect.getfullargspec()` will be called
    directly on the callable.
  """
  decorators, target = tf_decorator.unwrap(obj)

  for d in decorators:
    if d.decorator_argspec is not None:
      return _convert_maybe_argspec_to_fullargspec(d.decorator_argspec)
  return _getfullargspec(target)


def getcallargs(*func_and_positional, **named):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Does not raise errors for missing or surplus arguments as the standard
  # function does, and positional values are matched to every argument name not given by keyword
  # without checking counts; extra positionals are silently dropped by the pairing.
  # Decorator-aware version of the call-argument binder. The first positional input is the
  # callable, the rest are the positional arguments, and keyword arguments are the named ones.
  # Uses the full argspec of the callable; for a bound method its owner is prepended to the
  # positionals. Fills remaining positional names, then defaults, then keyword-only defaults.
  # Returns a new mapping from parameter names to the values they would receive. No side effects.
  """TFDecorator-aware replacement for inspect.getcallargs.

  Args:
    *func_and_positional: A callable, possibly decorated, followed by any
      positional arguments that would be passed to `func`.
    **named: The named argument dictionary that would be passed to `func`.

  Returns:
    A dictionary mapping `func`'s named arguments to the values they would
    receive if `func(*positional, **named)` were called.

  `getcallargs` will use the argspec from the outermost decorator that provides
  it. If no attached decorators modify argspec, the final unwrapped target's
  argspec will be used.
  """
  func = func_and_positional[0]
  positional = func_and_positional[1:]
  argspec = getfullargspec(func)
  call_args = named.copy()
  this = getattr(func, 'im_self', None) or getattr(func, '__self__', None)
  if ismethod(func) and this:
    positional = (this,) + positional
  remaining_positionals = [arg for arg in argspec.args if arg not in call_args]
  call_args.update(dict(zip(remaining_positionals, positional)))
  default_count = 0 if not argspec.defaults else len(argspec.defaults)
  if default_count:
    for arg, value in zip(argspec.args[-default_count:], argspec.defaults):
      if arg not in call_args:
        call_args[arg] = value
  if argspec.kwonlydefaults is not None:
    for k, v in argspec.kwonlydefaults.items():
      if k not in call_args:
        call_args[k] = v
  return call_args


def getframeinfo(*args, **kwargs):
  # [FUNCTION DOC]
  # Pass-through to the standard frame-information function: forwards all positional and keyword
  # arguments unchanged (a frame or traceback, optional number of context lines) and returns its
  # result. No decorator unwrapping is performed.
  return _inspect.getframeinfo(*args, **kwargs)


def getdoc(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns the cleaned documentation string of the object using the standard function. The
  # object is deliberately not unwrapped, so the outermost decorated object's documentation is
  # used. Returns null if there is none.
  """TFDecorator-aware replacement for inspect.getdoc.

  Args:
    object: An object, possibly decorated.

  Returns:
    The docstring associated with the object.

  The outermost-decorated object is intended to have the most complete
  documentation, so the decorated parameter is not unwrapped.
  """
  return _inspect.getdoc(object)


def getfile(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns the source file path of an object after removing decorators. Special case: for a
  # frame whose globals have a file entry, that entry is returned (works around wrong paths when
  # only compiled files exist). Otherwise delegates to the standard function, which raises a
  # type error for built-in objects.
  """TFDecorator-aware replacement for inspect.getfile."""
  unwrapped_object = tf_decorator.unwrap(object)[1]

  # Work around for the case when object is a stack frame
  # and only .pyc files are used. In this case, getfile
  # might return incorrect path. So, we get the path from f_globals
  # instead.
  if (hasattr(unwrapped_object, 'f_globals') and
      '__file__' in unwrapped_object.f_globals):
    return unwrapped_object.f_globals['__file__']
  return _inspect.getfile(unwrapped_object)


def getmembers(object, predicate=None):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # KNOWN ISSUE: Despite the "TFDecorator-aware" docstring, the object is not unwrapped; this is
  # a plain pass-through to the standard function.
  # Returns a sorted list of name and value pairs of the object's members, optionally filtered by
  # the predicate callable (default: no filtering).
  """TFDecorator-aware replacement for inspect.getmembers."""
  return _inspect.getmembers(object, predicate)


def getmodule(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # KNOWN ISSUE: Despite the "TFDecorator-aware" docstring, the object is not unwrapped; plain
  # pass-through to the standard function.
  # Returns the module an object was defined in, or null if it cannot be determined.
  """TFDecorator-aware replacement for inspect.getmodule."""
  return _inspect.getmodule(object)


def getmro(cls):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Despite the "TFDecorator-aware" docstring, the class is not unwrapped; plain
  # pass-through to the standard function.
  # Returns the tuple of the class and its base classes in method resolution order.
  """TFDecorator-aware replacement for inspect.getmro."""
  return _inspect.getmro(cls)


def getsource(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators from the object and returns its source code as one text string. Raises an
  # OS error if the source is unavailable and a type error for built-in objects.
  """TFDecorator-aware replacement for inspect.getsource."""
  return _inspect.getsource(tf_decorator.unwrap(object)[1])


def getsourcefile(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators from the object and returns the path of the Python source file it was
  # defined in, or null if none is found. Raises a type error for built-in objects.
  """TFDecorator-aware replacement for inspect.getsourcefile."""
  return _inspect.getsourcefile(tf_decorator.unwrap(object)[1])


def getsourcelines(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators from the object and returns a pair: the list of its source lines and the
  # starting line number in its file. Raises an OS error if the source is unavailable.
  """TFDecorator-aware replacement for inspect.getsourcelines."""
  return _inspect.getsourcelines(tf_decorator.unwrap(object)[1])


def isbuiltin(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators, then returns true if the underlying object is a built-in function.
  """TFDecorator-aware replacement for inspect.isbuiltin."""
  return _inspect.isbuiltin(tf_decorator.unwrap(object)[1])


def isclass(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators, then returns true if the underlying object is a class.
  """TFDecorator-aware replacement for inspect.isclass."""
  return _inspect.isclass(tf_decorator.unwrap(object)[1])


def isfunction(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators, then returns true if the underlying object is a user-defined function.
  """TFDecorator-aware replacement for inspect.isfunction."""
  return _inspect.isfunction(tf_decorator.unwrap(object)[1])


def isframe(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring says it replaces the module check, but the code tests whether the
  # underlying object is a stack frame (copy-paste error in the docstring).
  # Unwraps decorators, then returns true if the underlying object is a frame.
  """TFDecorator-aware replacement for inspect.ismodule."""
  return _inspect.isframe(tf_decorator.unwrap(object)[1])


def isgenerator(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators, then returns true if the underlying object is a generator iterator.
  """TFDecorator-aware replacement for inspect.isgenerator."""
  return _inspect.isgenerator(tf_decorator.unwrap(object)[1])


def isgeneratorfunction(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators, then returns true if the underlying object is a generator function.
  """TFDecorator-aware replacement for inspect.isgeneratorfunction."""
  return _inspect.isgeneratorfunction(tf_decorator.unwrap(object)[1])


def ismethod(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators, then returns true if the underlying object is a bound method.
  """TFDecorator-aware replacement for inspect.ismethod."""
  return _inspect.ismethod(tf_decorator.unwrap(object)[1])


def ismodule(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators, then returns true if the underlying object is a module.
  """TFDecorator-aware replacement for inspect.ismodule."""
  return _inspect.ismodule(tf_decorator.unwrap(object)[1])


def isroutine(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Unwraps decorators, then returns true if the underlying object is any kind of function or
  # method (user-defined, built-in or method descriptor).
  """TFDecorator-aware replacement for inspect.isroutine."""
  return _inspect.isroutine(tf_decorator.unwrap(object)[1])


def stack(context=1):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Not decorator-aware despite the docstring; it only forwards to the standard
  # function and drops the first record (this function's own frame) from the result.
  # Input context is the number of source lines of context per frame (default one). Returns a
  # list of frame records for the callers of this function, innermost first.
  """TFDecorator-aware replacement for inspect.stack."""
  return _inspect.stack(context)[1:]
