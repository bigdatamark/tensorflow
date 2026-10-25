# [MODULE DOC]
# HELPER: small introspection utilities shared across TensorFlow.
# Provides functions to get argument names of function-like objects (including partials and
# callable instances), detect keyword-argument catch-alls, get a callable's display name and
# code object, and lazily build a serialized session configuration with the meta-optimizer
# disabled. Holds one module-level cache for that configuration. No import-time side effects.
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
"""Utility to retrieve function args."""

import functools

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def _is_bound_method(fn):
  # [FUNCTION DOC]
  # HELPER: used by fn_args to decide whether a leading self/cls argument must be dropped.
  # Input: fn, any callable (possibly wrapped by decorators). Unwraps decorators first, then
  # returns true only if the underlying object is a method with a non-null bound instance.
  # Returns a boolean; no exceptions or side effects.
  _, fn = tf_decorator.unwrap(fn)
  return tf_inspect.ismethod(fn) and (fn.__self__ is not None)


def _is_callable_object(obj):
  # [FUNCTION DOC]
  # HELPER: used by fn_args and has_kwargs to detect instances of classes defining a call method.
  # Input: obj, any object. Returns true if the object has a call attribute that is a method
  # (as opposed to a plain function or builtin); otherwise false. No side effects.
  return hasattr(obj, '__call__') and tf_inspect.ismethod(obj.__call__)


def fn_args(fn):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised when a partial has positionally
  # bound arguments, but the code never raises; it silently drops that many leading names.
  # Returns the tuple of argument names of a function-like object.
  # Input: fn, a function, bound method, callable instance, or partial object. For a partial,
  # recurses on the wrapped function, removes as many leading names as there are positional
  # bound values, and removes names bound by keyword. For a callable instance, inspects its call
  # method. For a bound method, the first positional name (self/cls) is removed if present.
  # Output: tuple of strings. Exceptions: whatever the argument-spec inspection raises (for
  # example for non-callables or unsupported builtins). No side effects.
  """Get argument names for function-like object.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `tuple` of string argument names.

  Raises:
    ValueError: if partial function has positionally bound arguments
  """
  if isinstance(fn, functools.partial):
    args = fn_args(fn.func)
    args = [a for a in args[len(fn.args):] if a not in (fn.keywords or [])]
  else:
    if _is_callable_object(fn):
      fn = fn.__call__
    args = tf_inspect.getfullargspec(fn).args
    if _is_bound_method(fn) and args:
      # If it's a bound method, it may or may not have a self/cls first
      # argument; for example, self could be captured in *args.
      # If it does have a positional argument, it is self/cls.
      args.pop(0)
  return tuple(args)


def has_kwargs(fn):
  # [FUNCTION DOC]
  # Reports whether a callable accepts arbitrary keyword arguments.
  # Input: fn, a function, callable instance, or partial object (a partial is replaced by its
  # wrapped function; a callable instance by its call method).
  # Output: true if the argument spec has a variadic keyword parameter, otherwise false.
  # Raises a type error if fn is not callable. No side effects.
  """Returns whether the passed callable has **kwargs in its signature.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `bool`: if `fn` has **kwargs in its signature.

  Raises:
     `TypeError`: If fn is not a Function, or function-like object.
  """
  if isinstance(fn, functools.partial):
    fn = fn.func
  elif _is_callable_object(fn):
    fn = fn.__call__
  elif not callable(fn):
    raise TypeError(
        'Argument `fn` should be a callable. '
        f'Received: fn={fn} (of type {type(fn)})')
  return tf_inspect.getfullargspec(fn).varkw is not None


def get_func_name(func):
  # [FUNCTION DOC]
  # Returns a display name for a callable, after unwrapping decorators.
  # Input: func, a callable. Output: for a plain function its name; for a method the class name
  # of the bound instance and the method name joined by a dot; for any other callable (such as
  # an instance with a call method) the string form of its type.
  # Raises a value error if the unwrapped object is not callable. No side effects.
  """Returns name of passed callable."""
  _, func = tf_decorator.unwrap(func)
  if callable(func):
    if tf_inspect.isfunction(func):
      return func.__name__
    elif tf_inspect.ismethod(func):
      return '%s.%s' % (
          func.__self__.__class__.__name__,
          func.__func__.__name__,
      )
    else:  # Probably a class instance with __call__
      return str(type(func))
  else:
    raise ValueError(
        'Argument `func` must be a callable. '
        f'Received func={func} (of type {type(func)})')


def get_func_code(func):
  # [FUNCTION DOC]
  # Returns the code object of a callable after unwrapping decorators.
  # Input: func, a callable. For functions and methods returns their code object; for other
  # callables (such as callable class instances) tries the code object of the call method and
  # returns nothing/null if that is unavailable (for example for partial objects).
  # Raises a value error if the unwrapped object is not callable. No side effects.
  """Returns func_code of passed callable, or None if not available."""
  _, func = tf_decorator.unwrap(func)
  if callable(func):
    if tf_inspect.isfunction(func) or tf_inspect.ismethod(func):
      return func.__code__
    # Since the object is not a function or method, but is a callable, we will
    # try to access the __call__method as a function.  This works with callable
    # classes but fails with functool.partial objects despite their __call__
    # attribute.
    try:
      return func.__call__.__code__
    except AttributeError:
      return None
  else:
    raise ValueError(
        'Argument `func` must be a callable. '
        f'Received func={func} (of type {type(func)})')


_rewriter_config_optimizer_disabled = None


def get_disabled_rewriter_config():
  # [FUNCTION DOC]
  # Returns a serialized session configuration (bytes) in which the graph meta-optimizer is
  # disabled. Takes no inputs. The value is computed lazily on first call and cached in the
  # module-level variable, so later calls return the same bytes. Side effect: sets that global.
  global _rewriter_config_optimizer_disabled
  if _rewriter_config_optimizer_disabled is None:
    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.disable_meta_optimizer = True
    _rewriter_config_optimizer_disabled = config.SerializeToString()
  return _rewriter_config_optimizer_disabled
