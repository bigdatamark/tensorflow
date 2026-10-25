# [MODULE DOC]
# KNOWN ISSUE: the module docstring says "Tensor utility functions", but the module contains
# deprecation decorators and helpers, not tensor utilities.
# Deprecation toolkit for TensorFlow Python code. Provides decorators that mark functions,
# classes, methods, specific arguments or specific argument values as deprecated (each logs a
# warning with the call location and update instructions, and rewrites the docstring with a
# deprecation notice), an alias helper for renamed symbols, a decorator that records deprecated
# endpoint names, a helper for renamed keyword arguments, a context manager that temporarily
# silences warnings, a hook factory for moved modules, and a property type that hides inherited
# attributes from the public API.
# Module state: a global switch that enables or disables printing warnings, and a global mapping
# that remembers which warnings were already printed (so warn-once behavior works). In strict
# mode (from the framework strict-mode module) a deprecation use logs an error and raises a
# runtime error instead of just warning. Imports only logging, decorator and inspection helpers;
# no registrations happen at import time other than creating the one module-level hidden
# attribute constant.
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
"""Tensor utility functions."""
import collections
import functools
import inspect
import re

from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls

# Allow deprecation warnings to be silenced temporarily with a context manager.
_PRINT_DEPRECATION_WARNINGS = True

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


class DeprecatedNamesAlreadySetError(Exception):
  # [CLASS DOC]
  # Exception type raised by the deprecated-endpoints decorator when a symbol already carries a
  # list of deprecated API names and the decorator is applied to it again. Plain exception
  # subclass with no extra state.
  """Raised when setting deprecated names multiple times for the same symbol."""


def _log_deprecation(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: single point through which all deprecation decorators emit their messages.
  # Inputs: msg, a logging format string; extra positional values for the format string; extra
  # keyword arguments passed through to the logger. Behavior: if strict mode is enabled, logs
  # the message as an error and then raises a runtime error saying the deprecated behavior is
  # an error in strict mode; otherwise logs the message as a warning and returns nothing.
  # Raises: runtime error in strict mode only. Side effect: writes to the TensorFlow log.
  """Raises errors for deprecated methods if in strict mode, warns otherwise."""
  if strict_mode.STRICT_MODE:
    logging.error(msg, *args, **kwargs)
    raise RuntimeError(
        'This behavior has been deprecated, which raises an error in strict'
        ' mode.'
    )
  else:
    logging.warning(msg, *args, **kwargs)


def _add_deprecated_function_notice_to_docstring(doc, date, instructions):
  # [FUNCTION DOC]
  # HELPER: builds the new docstring used by the function/class deprecation decorators.
  # Inputs: doc, the original docstring (text or nothing); date, the removal date text or
  # nothing; instructions, the update instructions text (may be empty). Builds the notice
  # "this function is deprecated, it will be removed after <date>" (or "in a future version"
  # when no date), adds an "Instructions for updating" line only when instructions are
  # non-empty, and delegates to the shared notice helper with the placeholder text
  # "DEPRECATED FUNCTION", the suffix "(deprecated)" and the notice type "Deprecated".
  # Returns the new docstring text. No side effects.
  """Adds a deprecation notice to a docstring for deprecated functions."""
  main_text = [
      'THIS FUNCTION IS DEPRECATED. It will be removed %s.'
      % ('in a future version' if date is None else ('after %s' % date))
  ]
  if instructions:
    main_text.append('Instructions for updating:')
  return decorator_utils.add_notice_to_docstring(
      doc,
      instructions,
      'DEPRECATED FUNCTION',
      '(deprecated)',
      main_text,
      notice_type='Deprecated')


def _add_deprecated_arg_notice_to_docstring(doc, date, instructions,
                                            deprecated_names):
  # [FUNCTION DOC]
  # HELPER: builds the new docstring used by the deprecated-arguments decorator.
  # Inputs: doc, the original docstring (text or nothing); date, removal date text or nothing;
  # instructions, update instructions text; deprecated_names, collection of deprecated argument
  # names. Sorts and joins the names into a comma separated list, writes the notice "some
  # arguments are deprecated: (<names>), they will be removed <when>" followed by "Instructions
  # for updating:", and delegates to the shared notice helper with placeholder "DEPRECATED
  # FUNCTION ARGUMENTS" and suffix "(deprecated arguments)". Returns the new docstring text.
  # No side effects.
  """Adds a deprecation notice to a docstring for deprecated arguments."""

  deprecation_string = ', '.join(sorted(deprecated_names))

  return decorator_utils.add_notice_to_docstring(
      doc,
      instructions,
      'DEPRECATED FUNCTION ARGUMENTS',
      '(deprecated arguments)', [
          'SOME ARGUMENTS ARE DEPRECATED: `(%s)`. '
          'They will be removed %s.' %
          (deprecation_string, 'in a future version' if date is None else
           ('after %s' % date)), 'Instructions for updating:'
      ],
      notice_type='Deprecated')


def _add_deprecated_arg_value_notice_to_docstring(doc, date, instructions,
                                                  deprecated_name_value_dict):
  # [FUNCTION DOC]
  # HELPER: builds the new docstring used by the deprecated-argument-values decorator.
  # KNOWN ISSUE: the docstring is copied from the deprecated-arguments helper and says
  # "deprecated arguments"; this function actually documents deprecated argument values.
  # Inputs: doc, the original docstring (text or nothing); date, removal date text or nothing;
  # instructions, update instructions text; deprecated_name_value_dict, mapping from argument
  # name to the deprecated value. Formats each pair as name equals the printable
  # representation of the value, sorted by name and comma separated, and writes the notice
  # "some argument values are deprecated: (<pairs>), they will be removed <when>" followed by
  # "Instructions for updating:". Delegates to the shared notice helper with placeholder
  # "DEPRECATED FUNCTION ARGUMENT VALUES" and suffix "(deprecated argument values)". Returns the
  # new docstring text. No side effects.
  """Adds a deprecation notice to a docstring for deprecated arguments."""

  deprecation_string = ', '.join(
      '%s=%r' % (key, value)
      for key, value in sorted(deprecated_name_value_dict.items()))

  when = 'in a future version' if date is None else ('after %s' % date)

  return decorator_utils.add_notice_to_docstring(
      doc,
      instructions,
      'DEPRECATED FUNCTION ARGUMENT VALUES',
      '(deprecated argument values)', [
          'SOME ARGUMENT VALUES ARE DEPRECATED: `(%s)`. '
          'They will be removed %s.' %
          (deprecation_string, when), 'Instructions for updating:'
      ],
      notice_type='Deprecated')


def _validate_deprecation_args(date, instructions):
  # [FUNCTION DOC]
  # HELPER: argument checking shared by all deprecation decorators.
  # KNOWN ISSUE: the date pattern is only matched at the start of the string and is loose
  # (months up to 19 and days up to 39 pass, and trailing text is accepted), so it is not a
  # strict YYYY-MM-DD validation as the error message implies.
  # Inputs: date, removal date text or nothing; instructions, update instructions text.
  # Raises a value error if a date is given but does not look like a year-month-day date, or if
  # instructions are empty or missing. Returns nothing.
  if date is not None and not re.match(r'20\d\d-[01]\d-[0123]\d', date):
    raise ValueError(f'Date must be in format YYYY-MM-DD. Received: {date}')
  if not instructions:
    raise ValueError(
        'Don\'t deprecate things without conversion instructions! Specify '
        'the `instructions` argument.')


def _call_location(outer=False):
  # [FUNCTION DOC]
  # HELPER: finds the user code location to cite in deprecation warnings.
  # Input: outer, boolean (default false). Walks the call stack two frames up from this function
  # (the caller of the function that called this one). If outer is true and one more frame
  # exists above that, uses that frame instead. Returns text of the form file name, colon,
  # line number for the chosen frame. Uses the standard inspection module directly rather than
  # the TensorFlow wrapper (to avoid test timeouts). May fail with an attribute error if the
  # stack is shallower than expected. No side effects.
  """Returns call location given level up from current call."""
  # Two up: <_call_location>, <_call_location's caller>
  # tf_inspect is not required here. Please ignore the lint warning by adding
  # DISABLE_IMPORT_INSPECT_CHECK=TRUE to your cl description. Using it caused
  # test timeouts (b/189384061).
  f = inspect.currentframe().f_back.f_back
  parent = f and f.f_back
  if outer and parent is not None:
    f = parent
  return '{}:{}'.format(f.f_code.co_filename, f.f_lineno)


def _safe_eq(a, b):
  # [FUNCTION DOC]
  # HELPER: equality test used by the deprecated-argument-values decorator.
  # Inputs: a and b, any values. If either is the null value, returns true only when both are
  # null; otherwise returns the result of the normal equality comparison (which may be a
  # non-boolean for array-like objects). No side effects.
  if a is None or b is None:
    return a is None and b is None
  return a == b


def _wrap_decorator(wrapped_function, decorator_name):
  # [FUNCTION DOC]
  # HELPER: creates a decorator factory used by the deprecation decorators.
  # Inputs: wrapped_function, the original function; decorator_name, text label recorded on the
  # resulting decorator object. Returns a function that takes a wrapper function and returns the
  # decorator-tracking object produced by the TensorFlow decorator utility, which links the
  # wrapper to the original function (so signatures and docs can be recovered). No side effects
  # on its own.
  """Indicate that one function wraps another.

  This decorator wraps a function using `tf_decorator.make_decorator`
  so that doc generation scripts can pick up original function
  signature.
  It would be better to use @functools.wrap decorator, but it would
  not update function signature to match wrapped function in Python 2.

  Args:
    wrapped_function: The function that decorated function wraps.
    decorator_name: The name of the decorator.

  Returns:
    Function that accepts wrapper function as an argument and returns
    `TFDecorator` instance.
  """

  def wrapper(wrapper_func):
    # [FUNCTION DOC]
    # HELPER: inner decorator returned by the wrapping factory. Takes the wrapper function and
    # returns the decorator-tracking object that ties it to the original function and name.
    return tf_decorator.make_decorator(wrapped_function, wrapper_func,
                                       decorator_name)

  return wrapper


def deprecated_alias(deprecated_name, name, func_or_class, warn_once=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: for classes the code carries a dead branch for Python 2 (the check for a
  # function-bearing method object never holds on Python 3), and the class alias sets its
  # module attribute to the call location text (file and line) rather than a module name.
  # Creates a backwards-compatible alias for a renamed class or function. Inputs:
  # deprecated_name, the old fully qualified name used in messages; name, the new name to use
  # instead; func_or_class, the class or callable to alias; warn_once, boolean (default true)
  # to warn only on the first use. For a class: returns a subclass whose docstring carries a
  # class deprecation notice, whose name equals the original and whose constructor copies the
  # original constructor docstring and logs a warning (unless warnings are silenced or already
  # printed) before running the original constructor. For a function: validates it is callable
  # (value error otherwise) and returns a wrapper that logs the warning on use and then calls
  # the original, with a function deprecation notice in its docstring. Side effects: records
  # printed warnings in the module-level mapping; in strict mode warning becomes an error.
  """Deprecate a symbol in favor of a new name with identical semantics.

  This function is meant to be used when defining a backwards-compatibility
  alias for a symbol which has been moved. For example:

  module1.py:
  ```python
  class NewNameForClass: pass
  ```

  module2.py:
  ```python
  import module1

  DeprecatedNameForClass = deprecated_alias(
    deprecated_name='module2.DeprecatedNameForClass',
    name='module1.NewNameForClass',
    func_or_class=module1.NewNameForClass)
  ```

  This function works for classes and functions.

  For classes, it creates a new class which is functionally identical (it
  inherits from the original, and overrides its constructor), but which prints
  a deprecation warning when an instance is created. It also adds a deprecation
  notice to the class' docstring.

  For functions, it returns a function wrapped by `tf_decorator.make_decorator`.
  That function prints a warning when used, and has a deprecation notice in its
  docstring. This is more or less equivalent (the deprecation warning has
  slightly different text) to writing:

  ```python
  @deprecated
  def deprecated_alias(original_args):
    real_function(original_args)
  ```

  Args:
    deprecated_name: The name of the symbol that is being deprecated, to be used
      in the warning message. This should be its fully qualified name to avoid
      confusion.
    name: The name of the symbol that is to be used instead of the deprecated
      name. This should be a fully qualified name to avoid confusion.
    func_or_class: The (non-deprecated) class or function for which a deprecated
      alias should be created.
    warn_once: If True (the default), only print a deprecation warning the first
      time this function is used, or the class is instantiated.

  Returns:
    A wrapped version of `func_or_class` which prints a deprecation warning on
    use and has a modified docstring.
  """
  if tf_inspect.isclass(func_or_class):

    # Make a new class with __init__ wrapped in a warning.
    class _NewClass(func_or_class):  # pylint: disable=missing-docstring
      # [CLASS DOC]
      # HELPER: dynamically created subclass returned by the alias helper for classes. Behaves
      # like the original class but has a deprecation notice in its docstring, reuses the
      # original name and logs a warning when an instance is created.
      __doc__ = decorator_utils.add_notice_to_docstring(
          func_or_class.__doc__,
          'Please use %s instead.' % name,
          'DEPRECATED CLASS',
          '(deprecated)', [('THIS CLASS IS DEPRECATED. '
                            'It will be removed in a future version. ')],
          notice_type='Deprecated')
      __name__ = func_or_class.__name__
      __module__ = _call_location(outer=True)

      @_wrap_decorator(func_or_class.__init__, 'deprecated_alias')
      def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: constructor of the deprecated alias class.
        # Takes any positional and keyword arguments and forwards them unchanged to the original
        # class constructor. First copies the original constructor docstring onto the alias
        # constructor. Then, if warnings are enabled and this alias was not warned about yet,
        # logs the "name is deprecated, use the new name" message (remembering it when warn_once
        # is set). Returns nothing; may raise a runtime error in strict mode.
        if hasattr(_NewClass.__init__, '__func__'):
          # Python 2
          _NewClass.__init__.__func__.__doc__ = func_or_class.__init__.__doc__
        else:
          # Python 3
          _NewClass.__init__.__doc__ = func_or_class.__init__.__doc__

        if _PRINT_DEPRECATION_WARNINGS:
          # We're making the alias as we speak. The original may have other
          # aliases, so we cannot use it to check for whether it's already been
          # warned about.
          if _NewClass.__init__ not in _PRINTED_WARNING:
            if warn_once:
              _PRINTED_WARNING[_NewClass.__init__] = True
            _log_deprecation(
                'From %s: The name %s is deprecated. Please use %s instead.\n',
                _call_location(), deprecated_name, name)
        super(_NewClass, self).__init__(*args, **kwargs)

    return _NewClass
  else:
    decorator_utils.validate_callable(func_or_class, 'deprecated')

    # Make a wrapper for the original
    @functools.wraps(func_or_class)
    def new_func(*args, **kwargs):  # pylint: disable=missing-docstring
      # [FUNCTION DOC]
      # HELPER: wrapper returned for function aliases. Forwards all arguments to the original
      # function and returns its result. Before calling, if warnings are enabled and this
      # wrapper has not warned yet, logs the "name is deprecated, use the new name" message
      # (remembering it when warn_once is set). May raise a runtime error in strict mode.
      if _PRINT_DEPRECATION_WARNINGS:
        # We're making the alias as we speak. The original may have other
        # aliases, so we cannot use it to check for whether it's already been
        # warned about.
        if new_func not in _PRINTED_WARNING:
          if warn_once:
            _PRINTED_WARNING[new_func] = True
          _log_deprecation(
              'From %s: The name %s is deprecated. Please use %s instead.\n',
              _call_location(), deprecated_name, name)
      return func_or_class(*args, **kwargs)

    return tf_decorator.make_decorator(
        func_or_class, new_func, 'deprecated',
        _add_deprecated_function_notice_to_docstring(
            func_or_class.__doc__, None, 'Please use %s instead.' % name))


def deprecated_endpoints(*args):
  # [FUNCTION DOC]
  # Creates a decorator that records deprecated API endpoint names on a symbol. Input: any
  # number of endpoint name texts. Returns a decorator; it prints no deprecation messages
  # (despite the name, it only tags the symbol). Applying the decorator stores the given names
  # (as a tuple) in a special attribute of the decorated symbol and returns the same symbol.
  # Raises DeprecatedNamesAlreadySetError at decoration time if the symbol already has that
  # attribute in its own attribute table. Side effect: mutates the decorated symbol.
  """Decorator for marking endpoints deprecated.

  This decorator does not print deprecation messages.
  TODO(annarev): eventually start printing deprecation warnings when
  @deprecation_endpoints decorator is added.

  Args:
    *args: Deprecated endpoint names.

  Returns:
    A function that takes symbol as an argument and adds
    _tf_deprecated_api_names to that symbol.
    _tf_deprecated_api_names would be set to a list of deprecated
    endpoint names for the symbol.
  """

  def deprecated_wrapper(func):
    # [FUNCTION DOC]
    # HELPER: decorator returned by the deprecated-endpoints factory. Input: func, the symbol to
    # tag. Raises DeprecatedNamesAlreadySetError if the symbol already has deprecated names set;
    # otherwise stores the endpoint names on it and returns the same symbol.
    # pylint: disable=protected-access
    if '_tf_deprecated_api_names' in func.__dict__:
      raise DeprecatedNamesAlreadySetError(
          f'Cannot set deprecated names for {func.__name__} to {args}. '
          'Deprecated names are already set to '
          f'{func._tf_deprecated_api_names}.')
    func._tf_deprecated_api_names = args
    # pylint: disable=protected-access
    return func

  return deprecated_wrapper


def deprecated(date, instructions, warn_once=True):
  # [FUNCTION DOC]
  # Decorator factory that marks a function, method or class as deprecated. Inputs: date, removal
  # date in year-month-day form or nothing; instructions, non-empty text telling users how to
  # update; warn_once, boolean (default true) to warn only on first call. Validates the inputs
  # immediately and raises a value error for a malformed date or empty instructions. Returns a
  # decorator. For a function: returns a wrapper that logs a warning (with caller location,
  # qualified name, module, removal timing and instructions) on calls when warnings are enabled
  # and not yet shown, then calls the original; the docstring gets a deprecation notice and the
  # signature of the original is preserved. For a class: wraps its constructor (the new method if
  # customized, otherwise the init method), installs the wrapper on the class, prepends the
  # notice to the class docstring and returns the class; if the inherited init is already
  # deprecated the class is returned unchanged. Marks the wrapper as deprecated for the doc
  # tooling. Side effects: mutates the class when decorating a class; records printed warnings
  # in the module-level mapping; strict mode turns the warning into a runtime error.
  """Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed. Must
      be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  """
  _validate_deprecation_args(date, instructions)

  def deprecated_wrapper(func_or_class):
    # [FUNCTION DOC]
    # HELPER: decorator returned by the deprecated factory. Input: func_or_class, a function,
    # method or class. Chooses the callable to wrap (the symbol itself, or for classes the new
    # method or the init method), validates it is callable (value error otherwise), builds the
    # warning wrapper, and returns either the wrapper (functions) or the mutated class. Returns
    # the class untouched if its inherited constructor is already deprecated.
    """Deprecation wrapper."""
    if isinstance(func_or_class, type):
      # If a class is deprecated, you actually want to wrap the constructor.
      cls = func_or_class
      if cls.__new__ is object.__new__:
        # If a class defaults to its parent's constructor, wrap that instead.
        func = cls.__init__
        constructor_name = '__init__'
        decorators, _ = tf_decorator.unwrap(func)
        for decorator in decorators:
          if decorator.decorator_name == 'deprecated':
            # If the parent is already deprecated, there's nothing to do.
            return cls
      else:
        func = cls.__new__
        constructor_name = '__new__'

    else:
      cls = None
      constructor_name = None
      func = func_or_class

    decorator_utils.validate_callable(func, 'deprecated')

    @_wrap_decorator(func, 'deprecated')
    def new_func(*args, **kwargs):  # pylint: disable=missing-docstring
      # [FUNCTION DOC]
      # HELPER: warning wrapper installed by the deprecated decorator. Forwards all positional
      # and keyword arguments to the original callable and returns its result. Before calling,
      # if warnings are enabled and neither the function nor its class was warned about, logs the
      # "is deprecated and will be removed" message with instructions; with warn_once it records
      # the function (and class) as warned. May raise a runtime error in strict mode.
      if _PRINT_DEPRECATION_WARNINGS:
        if func not in _PRINTED_WARNING and cls not in _PRINTED_WARNING:
          if warn_once:
            _PRINTED_WARNING[func] = True
            if cls:
              _PRINTED_WARNING[cls] = True
          _log_deprecation(
              'From %s: %s (from %s) is deprecated and will be removed %s.\n'
              'Instructions for updating:\n%s', _call_location(),
              decorator_utils.get_qualified_name(func),
              func_or_class.__module__,
              'in a future version' if date is None else ('after %s' % date),
              instructions)
      return func(*args, **kwargs)

    doc_controls.set_deprecated(new_func)
    new_func = tf_decorator.make_decorator(
        func, new_func, 'deprecated',
        _add_deprecated_function_notice_to_docstring(func.__doc__, date,
                                                     instructions))
    new_func.__signature__ = inspect.signature(func)

    if cls is None:
      return new_func
    else:
      # Insert the wrapped function as the constructor
      setattr(cls, constructor_name, new_func)

      # And update the docstring of the class.
      cls.__doc__ = _add_deprecated_function_notice_to_docstring(
          cls.__doc__, date, instructions)

      return cls

  return deprecated_wrapper


DeprecatedArgSpec = collections.namedtuple(
    'DeprecatedArgSpec', ['position', 'has_ok_value', 'ok_value'])


def deprecated_args(date, instructions, *deprecated_arg_names_or_tuples,
                    **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised when the second element of a tuple
  # is not a list, but no such check exists; any value is accepted as the "ok" value.
  # KNOWN ISSUE: the warning is only emitted in graph mode (not eagerly), which the docstring
  # does not mention.
  # Decorator factory that marks specific arguments of a function as deprecated. Inputs: date,
  # removal date (year-month-day text) or nothing; instructions, non-empty update text; any
  # number of deprecated argument specifications, each either a name or a pair of name and an
  # "ok" value that suppresses the warning when passed; optional keyword warn_once (default true).
  # Raises a value error for a bad date, empty instructions, no deprecated arguments given, or
  # any keyword other than warn_once. Returns a decorator. The decorator validates the target is
  # callable, checks every named argument exists in its signature (including variadic names) and
  # raises a value error listing missing ones. The wrapped function logs a warning, once per
  # function and argument unless warn_once is false, when a deprecated argument is supplied
  # positionally or by keyword (and not equal to its ok value), then calls the original. The
  # docstring gets a "deprecated arguments" notice. Side effect: records printed warnings.
  """Decorator for marking specific function arguments as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument. It has the following format:

    Calling <function> (from <module>) with <arg> is deprecated and will be
    removed after <date>. Instructions for updating:
      <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> includes the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed. Must
      be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    *deprecated_arg_names_or_tuples: String or 2-Tuple (String, ok_val).  The
      string is the deprecated argument name. Optionally, an ok-value may be
      provided.  If the user provided argument equals this value, the warning is
      suppressed.
    **kwargs: If `warn_once=False` is passed, every call with a deprecated
      argument will log a warning. The default behavior is to only warn the
      first time the function is called with any given deprecated argument. All
      other kwargs raise `ValueError`.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, instructions are
      empty, the deprecated arguments are not present in the function
      signature, the second element of a deprecated_tuple is not a
      list, or if a kwarg other than `warn_once` is passed.
  """
  _validate_deprecation_args(date, instructions)
  if not deprecated_arg_names_or_tuples:
    raise ValueError('Specify which argument is deprecated.')
  if kwargs and list(kwargs.keys()) != ['warn_once']:
    kwargs.pop('warn_once', None)
    raise ValueError(f'Illegal argument passed to deprecated_args: {kwargs}')
  warn_once = kwargs.get('warn_once', True)

  def _get_arg_names_to_ok_vals():
    # [FUNCTION DOC]
    # HELPER: normalizes the deprecated argument specifications. Takes no arguments (reads the
    # enclosing specification list). Returns a mapping from argument name to a spec record with
    # position minus one (unknown), a flag saying whether an ok value was given, and that value.
    """Returns a dict mapping arg_name to DeprecatedArgSpec w/o position."""
    d = {}
    for name_or_tuple in deprecated_arg_names_or_tuples:
      if isinstance(name_or_tuple, tuple):
        d[name_or_tuple[0]] = DeprecatedArgSpec(-1, True, name_or_tuple[1])
      else:
        d[name_or_tuple] = DeprecatedArgSpec(-1, False, None)
    return d

  def _get_deprecated_positional_arguments(names_to_ok_vals, arg_spec):
    # [FUNCTION DOC]
    # HELPER: resolves where each deprecated argument sits in the target signature.
    # Inputs: names_to_ok_vals, mapping from name to spec record; arg_spec, the full argument
    # specification of the target function. Considers regular then keyword-only arguments in
    # order. Returns a mapping, only for names found in the signature, to a spec record carrying
    # the zero-based position plus the original ok-value flag and value. No side effects.
    """Builds a dictionary from deprecated arguments to their spec.

    Returned dict is keyed by argument name.
    Each value is a DeprecatedArgSpec with the following fields:
       position: The zero-based argument position of the argument
         within the signature.  None if the argument isn't found in
         the signature.
       ok_values:  Values of this argument for which warning will be
         suppressed.

    Args:
      names_to_ok_vals: dict from string arg_name to a list of values, possibly
        empty, which should not elicit a warning.
      arg_spec: Output from tf_inspect.getfullargspec on the called function.

    Returns:
      Dictionary from arg_name to DeprecatedArgSpec.
    """
    # Extract argument list
    arg_space = arg_spec.args + arg_spec.kwonlyargs
    arg_name_to_pos = {name: pos for pos, name in enumerate(arg_space)}
    deprecated_positional_args = {}
    for arg_name, spec in iter(names_to_ok_vals.items()):
      if arg_name in arg_name_to_pos:
        pos = arg_name_to_pos[arg_name]
        deprecated_positional_args[arg_name] = DeprecatedArgSpec(
            pos, spec.has_ok_value, spec.ok_value)
    return deprecated_positional_args

  deprecated_arg_names = _get_arg_names_to_ok_vals()

  def deprecated_wrapper(func):
    # [FUNCTION DOC]
    # HELPER: decorator returned by the deprecated-arguments factory. Input: func, the function
    # to wrap. Validates it is callable, inspects its signature, and raises a value error if any
    # deprecated argument name is not part of the signature (listing the missing and known
    # names). Returns the wrapper (tracked as a deprecated decorator) whose docstring has the
    # deprecated-arguments notice.
    """Deprecation decorator."""
    decorator_utils.validate_callable(func, 'deprecated_args')

    arg_spec = tf_inspect.getfullargspec(func)
    deprecated_positions = _get_deprecated_positional_arguments(
        deprecated_arg_names, arg_spec)

    is_varargs_deprecated = arg_spec.varargs in deprecated_arg_names
    is_kwargs_deprecated = arg_spec.varkw in deprecated_arg_names

    if (len(deprecated_positions) + is_varargs_deprecated + is_kwargs_deprecated
        != len(deprecated_arg_names_or_tuples)):
      known_args = (
          arg_spec.args + arg_spec.kwonlyargs +
          [arg_spec.varargs, arg_spec.varkw])
      missing_args = [
          arg_name for arg_name in deprecated_arg_names
          if arg_name not in known_args
      ]
      raise ValueError('The following deprecated arguments are not present '
                       f'in the function signature: {missing_args}. '
                       'Expected arguments from the following list: '
                       f'{known_args}.')

    def _same_value(a, b):
      # [FUNCTION DOC]
      # HELPER: tolerant equality used to compare a passed value with an "ok" value.
      # Inputs: a and b, any values. Returns true if they are the identical object, or if their
      # equality comparison yields a real boolean true. Returns false when the comparison is
      # not a boolean (for example array-like results) or raises a type error. No side effects.
      """A comparison operation that works for multiple object types.

      Returns True for two empty lists, two numeric values with the
      same value, etc.

      Returns False for (pd.DataFrame, None), and other pairs which
      should not be considered equivalent.

      Args:
        a: value one of the comparison.
        b: value two of the comparison.

      Returns:
        A boolean indicating whether the two inputs are the same value
        for the purposes of deprecation.
      """
      if a is b:
        return True
      try:
        equality = a == b
        if isinstance(equality, bool):
          return equality
      except TypeError:
        return False
      return False

    @functools.wraps(func)
    def new_func(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: wrapper that emits deprecated-argument warnings. Forwards all arguments to the
      # original function and returns its result. Only when running in graph mode and warnings
      # are enabled does it work out which deprecated arguments were supplied (by position
      # within the passed positional arguments, by variadic use, or by keyword, skipping those
      # equal to an ok value) and logs one warning per argument not yet reported (remembering
      # it when warn_once). May raise a runtime error in strict mode.
      """Deprecation wrapper."""
      # TODO(apassos) figure out a way to have reasonable performance with
      # deprecation warnings and eager mode.
      if is_in_graph_mode.IS_IN_GRAPH_MODE() and _PRINT_DEPRECATION_WARNINGS:
        invalid_args = []
        named_args = tf_inspect.getcallargs(func, *args, **kwargs)
        for arg_name, spec in iter(deprecated_positions.items()):
          if (spec.position < len(args) and
              not (spec.has_ok_value and
                   _same_value(named_args[arg_name], spec.ok_value))):
            invalid_args.append(arg_name)
        if is_varargs_deprecated and len(args) > len(arg_spec.args):
          invalid_args.append(arg_spec.varargs)
        if is_kwargs_deprecated and kwargs:
          invalid_args.append(arg_spec.varkw)
        for arg_name in deprecated_arg_names:
          if (arg_name in kwargs and
              not (deprecated_positions[arg_name].has_ok_value and
                   _same_value(named_args[arg_name],
                               deprecated_positions[arg_name].ok_value))):
            invalid_args.append(arg_name)
        for arg_name in invalid_args:
          if (func, arg_name) not in _PRINTED_WARNING:
            if warn_once:
              _PRINTED_WARNING[(func, arg_name)] = True
            _log_deprecation(
                'From %s: calling %s (from %s) with %s is deprecated and will '
                'be removed %s.\nInstructions for updating:\n%s',
                _call_location(), decorator_utils.get_qualified_name(func),
                func.__module__, arg_name,
                'in a future version' if date is None else ('after %s' % date),
                instructions)
      return func(*args, **kwargs)

    doc = _add_deprecated_arg_notice_to_docstring(
        func.__doc__, date, instructions, sorted(deprecated_arg_names.keys()))
    return tf_decorator.make_decorator(func, new_func, 'deprecated', doc)

  return deprecated_wrapper


def deprecated_arg_values(date,
                          instructions,
                          warn_once=True,
                          **deprecated_kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says " (deprecated arguments)" is appended to the first docstring
  # line, but the code appends "(deprecated argument values)".
  # Decorator factory that marks specific argument values as deprecated. Inputs: date, removal
  # date (year-month-day text) or nothing; instructions, non-empty update text; warn_once,
  # boolean (default true); and keyword arguments mapping argument names to the deprecated values.
  # Raises a value error for a bad date, empty instructions, or when no deprecated values are
  # given. Returns a decorator. The decorator validates the target is callable and returns a
  # wrapper which, when warnings are enabled, resolves the call arguments to names and logs a
  # warning for each named argument equal (null-safe) to its deprecated value, once per function
  # and argument when warn_once is set, then calls the original. Docstring gets a notice about
  # deprecated argument values. Side effect: records printed warnings.
  """Decorator for marking specific function argument values as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument values. It has the following format:

    Calling <function> (from <module>) with <arg>=<value> is deprecated and
    will be removed after <date>. Instructions for updating:
      <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed. Must
      be ISO 8601 (YYYY-MM-DD), or None
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: If `True`, warn only the first time this function is called with
      deprecated argument values. Otherwise, every call (with a deprecated
      argument value) will log a warning.
    **deprecated_kwargs: The deprecated argument values.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  """
  _validate_deprecation_args(date, instructions)
  if not deprecated_kwargs:
    raise ValueError('Specify which argument values are deprecated.')

  def deprecated_wrapper(func):
    # [FUNCTION DOC]
    # HELPER: decorator returned by the deprecated-argument-values factory. Input: func, the
    # function to wrap. Raises a value error if it is not callable. Returns the wrapper (tracked
    # as a deprecated decorator) with the deprecated-argument-values notice in its docstring.
    """Deprecation decorator."""
    decorator_utils.validate_callable(func, 'deprecated_arg_values')

    @functools.wraps(func)
    def new_func(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: wrapper that emits deprecated-argument-value warnings. Forwards all arguments to
      # the original and returns its result. When warnings are enabled it maps the call
      # arguments to parameter names and logs a warning for each one equal to its deprecated
      # value and not yet reported (remembering it when warn_once). May raise a runtime error in
      # strict mode.
      """Deprecation wrapper."""
      if _PRINT_DEPRECATION_WARNINGS:
        named_args = tf_inspect.getcallargs(func, *args, **kwargs)
        for arg_name, arg_value in deprecated_kwargs.items():
          if arg_name in named_args and _safe_eq(named_args[arg_name],
                                                 arg_value):
            if (func, arg_name) not in _PRINTED_WARNING:
              if warn_once:
                _PRINTED_WARNING[(func, arg_name)] = True
              _log_deprecation(
                  'From %s: calling %s (from %s) with %s=%s is deprecated and '
                  'will be removed %s.\nInstructions for updating:\n%s',
                  _call_location(), decorator_utils.get_qualified_name(func),
                  func.__module__, arg_name, arg_value,
                  'in a future version' if date is None else
                  ('after %s' % date), instructions)
      return func(*args, **kwargs)

    doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,
                                                        instructions,
                                                        deprecated_kwargs)
    return tf_decorator.make_decorator(func, new_func, 'deprecated', doc)

  return deprecated_wrapper


def deprecated_argument_lookup(new_name, new_value, old_name, old_value):
  # [FUNCTION DOC]
  # Resolves a renamed argument. Inputs: new_name and old_name, argument names as text (used only
  # in the error message); new_value and old_value, the values supplied for each (null if not
  # used). Returns the old value if it is not null, otherwise the new value (possibly null).
  # Raises a value error if both values are non-null. No other side effects, and no deprecation
  # warning is logged.
  """Looks up deprecated argument name and ensures both are not used.

  Args:
    new_name: new name of argument
    new_value: value of new argument (or None if not used)
    old_name: old name of argument
    old_value: value of old argument (or None if not used)

  Returns:
    The effective argument that should be used.
  Raises:
    ValueError: if new_value and old_value are both non-null
  """
  if old_value is not None:
    if new_value is not None:
      raise ValueError(f"Cannot specify both '{old_name}' and '{new_name}'.")
    return old_value
  return new_value


def rewrite_argument_docstring(old_doc, old_argument, new_argument):
  # [FUNCTION DOC]
  # Renames an argument inside docstring text. Inputs: old_doc, docstring text or nothing;
  # old_argument and new_argument, argument names as text. Returns nothing if old_doc is
  # nothing; otherwise returns a copy where every backtick-quoted occurrence of the old name and
  # every occurrence of the old name followed by a colon is replaced with the new name. Plain
  # text replacement, so it may also change unrelated text ending in the old name and a colon.
  # No side effects.
  if old_doc is None:
    return None
  return old_doc.replace('`%s`' % old_argument,
                         '`%s`' % new_argument).replace('%s:' % old_argument,
                                                        '%s:' % new_argument)


@tf_contextlib.contextmanager
def silence():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the previous setting is restored after the yield without a protective
  # try/finally block, so if the body of the with-statement raises, deprecation warnings stay
  # disabled for the rest of the process.
  # Context manager (built with the TensorFlow context manager decorator) that temporarily turns
  # off deprecation warnings. Takes no arguments and yields nothing. On entry it saves the global
  # warning switch and sets it to off; after the body it restores the saved value. Side effect:
  # mutates the module-level warning switch, affecting all deprecation decorators.
  """Temporarily silence deprecation warnings."""
  global _PRINT_DEPRECATION_WARNINGS
  print_deprecation_warnings = _PRINT_DEPRECATION_WARNINGS
  _PRINT_DEPRECATION_WARNINGS = False
  yield
  _PRINT_DEPRECATION_WARNINGS = print_deprecation_warnings


def deprecate_moved_module(deprecated_name, new_module, deletion_version):
  # [FUNCTION DOC]
  # Builds a module-level attribute lookup hook for a module that moved. Inputs: deprecated_name,
  # name of the old module; new_module, the module object that replaces it; deletion_version,
  # text of the TensorFlow version in which the old module will be removed. Returns a hook that
  # takes an attribute name, logs (only the first time the hook is used, and only if warnings are
  # enabled) a "fix your imports, module has moved" message, and returns the attribute of the
  # new module with that name (the attribute lookup error from the new module propagates if it
  # is missing). Side effect of the hook: records itself in the printed-warnings mapping; strict
  # mode turns the warning into a runtime error.
  """Logs a warning when a module that has been moved to a new location is used.

  Copy the following code into the old module:

  ```
  import deprecation
  import new_module

  __getattr__ = deprecation.deprecate_moved_module(
      __name__, new_module, "2.9")  # adjust version number.
  ```

  Args:
    deprecated_name: Name of old module.
    new_module: Module to replace the old module.
    deletion_version: Version of TensorFlow in which the old module will be
      removed.

  Returns:
    A function that logs a warning and returns the symbol from the new module.
    Set this function as the module's `__getattr__`.
  """

  def getter(name):
    # [FUNCTION DOC]
    # HELPER: the attribute lookup hook returned by the moved-module factory. Input: name, the
    # attribute requested from the old module. Logs the move warning once (when enabled) and
    # returns the same-named attribute from the new module; raises an attribute error if the new
    # module lacks it.
    if getter not in _PRINTED_WARNING and _PRINT_DEPRECATION_WARNINGS:
      _PRINTED_WARNING[getter] = True
      _log_deprecation(
          'Please fix your imports. Module %s has been moved to %s. The old '
          'module will be deleted in version %s.', deprecated_name,
          new_module.__name__, deletion_version)
    return getattr(new_module, name)

  return getter


class HiddenTfApiAttribute(property):
  # [CLASS DOC]
  # Property subclass used to hide an inherited attribute or method from the public API. Its
  # getter always raises an attribute error carrying the message given at construction, and since
  # no setter or deleter is defined, assignment and deletion also fail. Assign an instance (or
  # the module alias hide_attribute_from_api) as a class attribute that shadows an inherited one.
  """Hides a class attribute from the public API.

  Attributes in public classes can be hidden from the API by having an '_' in
  front of the name (e.g. ClassName._variables). This doesn't work when
  attributes or methods are inherited from a parent class. To hide inherited
  attributes, set their values to be `deprecation.hide_attribute_from_api`.
  """

  def __init__(self, deprecation_message):
    # [FUNCTION DOC]
    # Constructor. Input: deprecation_message, the text used as the message of the attribute
    # error raised on any access. Builds a getter that ignores the instance and raises that
    # error, and passes it to the base property constructor. Returns nothing.

    def raise_error(unused_self):
      # [FUNCTION DOC]
      # HELPER: getter installed in the property. Ignores its instance argument and always
      # raises an attribute error with the deprecation message.
      raise AttributeError(deprecation_message)

    super(HiddenTfApiAttribute, self).__init__(raise_error)


hide_attribute_from_api = HiddenTfApiAttribute  # pylint: disable=invalid-name

# TODO(kathywu): Remove once cl/246395236 is submitted.
HIDDEN_ATTRIBUTE = HiddenTfApiAttribute('This attribute has been deprecated.')
