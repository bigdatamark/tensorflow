# [MODULE DOC]
# Drop-in replacements for functions of the standard inspect module that understand TensorFlow
# decorators (TFDecorator objects and functions built with make_decorator). Most predicates and
# source/file getters first unwrap the decorator chain and then delegate to the standard module;
# the argument-spec functions prefer an argument spec recorded on a decorator, handle partial
# objects, and convert between the older and fuller spec record types. Also exposes aliases for
# the standard parameter, signature and argument-spec types (defining compatible ones when missing).
# Import-time side effect: chooses implementations of the private spec helpers depending on
# whether the standard module offers getfullargspec. Widely used across TensorFlow.
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


# inspect.signature() is preferred over inspect.getfullargspec() in PY3.
# Note that while it can handle TFDecorators, it will ignore a TFDecorator's
# provided ArgSpec/FullArgSpec and instead return the signature of the
# inner-most function.
def signature(obj, *, follow_wrapped=True):
  # [FUNCTION DOC]
  # Returns the standard signature object of obj after unwrapping any TensorFlow decorators.
  # KNOWN ISSUE: because the innermost target is inspected, a signature override recorded on a
  # decorator is ignored, as the comment above the function says.
  # Inputs: obj, any callable; follow_wrapped, keyword-only flag (default true) passed through to
  # the standard function. Raises type or value errors from the standard function when no
  # signature can be found.
  """TFDecorator-aware replacement for inspect.signature."""
  return _inspect.signature(
      tf_decorator.unwrap(obj)[1], follow_wrapped=follow_wrapped)


Parameter = _inspect.Parameter
Signature = _inspect.Signature

if hasattr(_inspect, 'ArgSpec'):
  ArgSpec = _inspect.ArgSpec
else:
  ArgSpec = collections.namedtuple(
      'ArgSpec',
      [
          'args',
          'varargs',
          'keywords',
          'defaults',
      ],
  )


if hasattr(_inspect, 'FullArgSpec'):
  FullArgSpec = _inspect.FullArgSpec  # pylint: disable=invalid-name
else:
  FullArgSpec = collections.namedtuple('FullArgSpec', [
      'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
      'annotations'
  ])


def _convert_maybe_argspec_to_fullargspec(argspec):
  # [FUNCTION DOC]
  # HELPER: normalizes argument-spec records for getfullargspec and the fallback code.
  # Returns the input unchanged if it already is a full argument spec; otherwise builds a full
  # spec copying args, varargs, defaults and the keywords field (as the variable-keyword name),
  # with empty keyword-only names, no keyword-only defaults and empty annotations.
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
    # HELPER: version of getargspec built on getfullargspec, defined when the standard module
    # has getfullargspec. Input: target, an object to inspect (decorators are honored through
    # getfullargspec). Returns an ArgSpec whose names are the positional plus keyword-only
    # names, with the variable-keyword name as keywords, and defaults being positional defaults
    # followed by keyword-only default values (or null when there are none).
    # KNOWN ISSUE: keyword-only defaults are appended to the positional defaults without regard to
    # which names actually have defaults, so the result can misalign defaults with names.
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

    defaults = fullargspecs.defaults or ()
    if fullargspecs.kwonlydefaults:
      defaults += tuple(fullargspecs.kwonlydefaults.values())

    if not defaults:
      defaults = None

    argspecs = ArgSpec(
        args=fullargspecs.args + fullargspecs.kwonlyargs,
        varargs=fullargspecs.varargs,
        keywords=fullargspecs.varkw,
        defaults=defaults,
    )
    return argspecs
else:
  _getargspec = _inspect.getargspec

  def _getfullargspec(target):
    # [FUNCTION DOC]
    # HELPER: fallback defined only when the standard module lacks getfullargspec (old
    # interpreters). Input: target, the object to inspect. Returns a full spec built from the
    # older getargspec result with empty keyword-only names, defaults and annotations.
    # KNOWN ISSUE: effectively dead code on modern interpreters, where the other branch is used.
    """A python2 version of getfullargspec.

    Args:
      target: the target object to inspect.

    Returns:
      A FullArgSpec with empty kwonlyargs, kwonlydefaults and annotations.
    """
    return _convert_maybe_argspec_to_fullargspec(getargspec(target))


def currentframe():
  # [FUNCTION DOC]
  # Returns the frame object of the caller of this function, by taking the second entry of the
  # full interpreter stack. Takes no inputs and has no side effects. Note this builds the whole
  # stack with source context, which is slow compared with the standard direct frame lookup.
  """TFDecorator-aware replacement for inspect.currentframe."""
  return _inspect.stack()[1][0]


def getargspec(obj):
  # [FUNCTION DOC]
  # Returns an ArgSpec (names, variable positional name, keywords name, defaults) for a callable.
  # Input: obj, a function, partial object or callable object, possibly decorated. Partial objects
  # are delegated to a dedicated helper. Otherwise decorators are unwrapped and the first argument
  # spec recorded on a decorator wins; else the final target is inspected, falling back for classes
  # to the constructor and then the object-creation method, and finally to the call method of
  # the target's type. Raises ValueError when a signature cannot be expressed as an ArgSpec and
  # TypeError for unsupported objects. No side effects.
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
  # HELPER: implements getargspec for partial objects.
  # Input: obj, a partial object. Takes the argument spec of the wrapped function, drops as many
  # leading arguments as the partial has positional values, and applies the partial's keyword values
  # as defaults (they take precedence over the function's own defaults). Returns an ArgSpec; its
  # defaults are null when no argument has a default. Raises ValueError when a partial keyword
  # names an argument that does not exist and the function lacks variable keywords, or when an
  # argument without a default follows one with a default (not expressible in an ArgSpec).
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
  for kw, default in iter(partial_keywords.items()):
    if kw in args:
      idx = args.index(kw)
      all_defaults[idx] = default
    elif not keywords:
      raise ValueError(f'{obj} does not have a **kwargs parameter, but '
                       f'contains an unknown partial keyword {kw}.')

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
    raise ValueError(f'{obj} has some keyword-only arguments, which are not'
                     f' supported: {invalid_default_values}.')

  return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))


def getfullargspec(obj):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring contains a garbled fragment (a character-class pattern) where the
  # text about emulating the standard function in Python 2 should be.
  # Returns a full argument spec for a callable. Input: obj, a callable, possibly decorated.
  # Unwraps decorators; the first decorator with a recorded argument spec supplies the result
  # (converted to a full spec if needed); otherwise the final target is inspected directly.
  # Exceptions come from the standard inspection. No side effects.
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
  # Computes how the arguments of a call would be bound to parameter names. Inputs: the first
  # positional value is the callable (possibly decorated); the remaining positional values are the
  # call's positional arguments; named holds the call's keyword arguments. Uses the full argument
  # spec of the outermost recorded decorator or the final target. For bound methods the owning
  # instance is prepended to the positional values. Returns a new mapping from parameter names
  # to values, including defaults (positional and keyword-only) for parameters not supplied.
  # KNOWN ISSUE: unlike the standard function it does not raise for missing or surplus
  # arguments; extra positional values are silently dropped and missing ones are left out.
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
  # Thin pass-through to the standard frame information function; all positional and keyword
  # arguments are forwarded and its result returned. No decorator unwrapping is done.
  return _inspect.getframeinfo(*args, **kwargs)


def getdoc(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns the cleaned doc string of the given object using the standard function. The object
  # is deliberately not unwrapped because the outermost decorated object has the fullest docs.
  # Returns null if there is no doc. No side effects.
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
  # Returns the source file path for an object after unwrapping decorators. For frame objects whose
  # globals have a file entry, that entry is returned (works around wrong paths when only compiled
  # files exist); otherwise the standard function result is returned. Raises type errors for
  # built-in objects, as the standard function does.
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
  # Pass-through to the standard function listing the members of an object as name/value pairs
  # sorted by name, optionally filtered by the predicate callback (default none). No unwrapping,
  # despite the docstring's claim of decorator awareness.
  """TFDecorator-aware replacement for inspect.getmembers."""
  return _inspect.getmembers(object, predicate)


def getmodule(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Pass-through to the standard function returning the module that defines the object, or null if
  # unknown. No unwrapping is done.
  """TFDecorator-aware replacement for inspect.getmodule."""
  return _inspect.getmodule(object)


def getmro(cls):
  # [FUNCTION DOC]
  # Pass-through returning the tuple of base classes of cls in method resolution order.
  """TFDecorator-aware replacement for inspect.getmro."""
  return _inspect.getmro(cls)


def getsource(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns the source text of the object's innermost undecorated target. Raises OS or type errors
  # if the source is unavailable.
  """TFDecorator-aware replacement for inspect.getsource."""
  return _inspect.getsource(tf_decorator.unwrap(object)[1])


def getsourcefile(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns the name of the source file of the innermost undecorated target, or null if it cannot
  # be determined.
  """TFDecorator-aware replacement for inspect.getsourcefile."""
  return _inspect.getsourcefile(tf_decorator.unwrap(object)[1])


def getsourcelines(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns the pair of source lines and starting line number of the innermost undecorated
  # target. Raises OS or type errors if the source is unavailable.
  """TFDecorator-aware replacement for inspect.getsourcelines."""
  return _inspect.getsourcelines(tf_decorator.unwrap(object)[1])


def isbuiltin(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns true if the innermost undecorated target is a built-in function.
  """TFDecorator-aware replacement for inspect.isbuiltin."""
  return _inspect.isbuiltin(tf_decorator.unwrap(object)[1])


def isclass(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns true if the innermost undecorated target is a class.
  """TFDecorator-aware replacement for inspect.isclass."""
  return _inspect.isclass(tf_decorator.unwrap(object)[1])


def isfunction(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns true if the innermost undecorated target is a plain Python function.
  """TFDecorator-aware replacement for inspect.isfunction."""
  return _inspect.isfunction(tf_decorator.unwrap(object)[1])


def isframe(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it replaces the module predicate, but it actually checks for
  # a frame object.
  # Returns true if the innermost undecorated target is a stack frame.
  """TFDecorator-aware replacement for inspect.ismodule."""
  return _inspect.isframe(tf_decorator.unwrap(object)[1])


def isgenerator(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns true if the innermost undecorated target is a generator object.
  """TFDecorator-aware replacement for inspect.isgenerator."""
  return _inspect.isgenerator(tf_decorator.unwrap(object)[1])


def isgeneratorfunction(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns true if the innermost undecorated target is a generator function.
  """TFDecorator-aware replacement for inspect.isgeneratorfunction."""
  return _inspect.isgeneratorfunction(tf_decorator.unwrap(object)[1])


def ismethod(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns true if the innermost undecorated target is a bound method.
  """TFDecorator-aware replacement for inspect.ismethod."""
  return _inspect.ismethod(tf_decorator.unwrap(object)[1])


def isanytargetmethod(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring names the parameter obj but the real parameter is object, and the
  # name suggests only methods count while the final check is true for any callable that is not
  # a plain function (classes, callable instances), not just methods.
  # Decides whether the first argument of the callable's spec may be self or cls. Input: object,
  # a method, function or partial, possibly decorated. Returns true if any decorator's wrapped
  # target is a bound method; otherwise looks through partial objects to their function and
  # returns true when the result is callable but not a plain function. No side effects.
  # pylint: disable=g-doc-args,g-doc-return-or-yield
  """Checks if `object` or a TF Decorator wrapped target contains self or cls.

  This function could be used along with `tf_inspect.getfullargspec` to
  determine if the first argument of `object` argspec is self or cls. If the
  first argument is self or cls, it needs to be excluded from argspec when we
  compare the argspec to the input arguments and, if provided, the tf.function
  input_signature.

  Like `tf_inspect.getfullargspec` and python `inspect.getfullargspec`, it
  does not unwrap python decorators.

  Args:
    obj: An method, function, or functool.partial, possibly decorated by
    TFDecorator.

  Returns:
    A bool indicates if `object` or any target along the chain of TF decorators
    is a method.
  """
  decorators, target = tf_decorator.unwrap(object)
  for decorator in decorators:
    if _inspect.ismethod(decorator.decorated_target):
      return True

  # TODO(b/194845243): Implement the long term solution with inspect.signature.
  # A functools.partial object is not a function or method. But if the wrapped
  # func is a method, the argspec will contain self/cls.
  while isinstance(target, functools.partial):
    target = target.func

  # `target` is a method or an instance with __call__
  return callable(target) and not _inspect.isfunction(target)


def ismodule(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns true if the innermost undecorated target is a module.
  """TFDecorator-aware replacement for inspect.ismodule."""
  return _inspect.ismodule(tf_decorator.unwrap(object)[1])


def isroutine(object):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns true if the innermost undecorated target is a function, method or built-in routine.
  """TFDecorator-aware replacement for inspect.isroutine."""
  return _inspect.isroutine(tf_decorator.unwrap(object)[1])


def stack(context=1):
  # [FUNCTION DOC]
  # Returns the interpreter stack as records, with the first entry (this function's own frame)
  # removed so the list starts at the caller. Input: context, number of source lines of context
  # per frame (default 1).
  """TFDecorator-aware replacement for inspect.stack."""
  return _inspect.stack(context)[1:]
