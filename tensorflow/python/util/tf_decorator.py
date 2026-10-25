# [MODULE DOC]
# Core of TensorFlow's decorator introspection support. Provides the TFDecorator base class
# (records the wrapped target, decorator name, doc and argspec), make_decorator (attaches that
# metadata to a wrapper function and copies name, module, doc, dict and signature from the
# target), unwrap (peels a chain of decorators down to the final target), rewrap (swaps the
# innermost target), and helpers converting an argspec into an inspect signature.
# Used by tf_inspect, tf_export, tf_contextlib and many other utilities so that documentation and
# signature inspection are not obscured by generic wrapper signatures. No import-time side effects.
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
"""Base TFDecorator class and utility functions for working with decorators.

There are two ways to create decorators that TensorFlow can introspect into.
This is important for documentation generation purposes, so that function
signatures aren't obscured by the (*args, **kwds) signature that decorators
often provide.

1. Call `tf_decorator.make_decorator` on your wrapper function. If your
decorator is stateless, or can capture all of the variables it needs to work
with through lexical closure, this is the simplest option. Create your wrapper
function as usual, but instead of returning it, return
`tf_decorator.make_decorator(target, your_wrapper)`. This will attach some
decorator introspection metadata onto your wrapper and return it.

Example:

  def print_hello_before_calling(target):
    def wrapper(*args, **kwargs):
      print('hello')
      return target(*args, **kwargs)
    return tf_decorator.make_decorator(target, wrapper)

2. Derive from TFDecorator. If your decorator needs to be stateful, you can
implement it in terms of a TFDecorator. Store whatever state you need in your
derived class, and implement the `__call__` method to do your work before
calling into your target. You can retrieve the target via
`super(MyDecoratorClass, self).decorated_target`, and call it with whatever
parameters it needs.

Example:

  class CallCounter(tf_decorator.TFDecorator):
    def __init__(self, target):
      super(CallCounter, self).__init__('count_calls', target)
      self.call_count = 0

    def __call__(self, *args, **kwargs):
      self.call_count += 1
      return super(CallCounter, self).decorated_target(*args, **kwargs)

  def count_calls(target):
    return CallCounter(target)
"""
import inspect
from typing import Dict, Any


def _make_default_values(fullargspec: inspect.FullArgSpec) -> Dict[str, Any]:
  # [FUNCTION DOC]
  # HELPER: used by fullargspec_to_signature to line up default values with parameter names.
  # Takes an argument-spec record and returns a mapping from parameter name to default value. The
  # positional defaults are matched to the trailing positional argument names; keyword-only
  # defaults are merged in. Returns an empty mapping when there are no defaults. No side effects.
  """Returns default values from the function's fullargspec."""
  if fullargspec.defaults is not None:
    defaults = {
        name: value for name, value in zip(
            fullargspec.args[-len(fullargspec.defaults):], fullargspec.defaults)
    }
  else:
    defaults = {}

  if fullargspec.kwonlydefaults is not None:
    defaults.update(fullargspec.kwonlydefaults)

  return defaults


def fullargspec_to_signature(
    fullargspec: inspect.FullArgSpec) -> inspect.Signature:
  # [FUNCTION DOC]
  # Converts an argument-spec record into an equivalent signature object. Builds, in order, the
  # positional-or-keyword parameters (with defaults where present), the variable positional
  # parameter, the keyword-only parameters (with their defaults) and the variable keyword
  # parameter. Returns the new signature; raises whatever the signature classes raise if the
  # spec is inconsistent (for example an invalid parameter name). No side effects.
  """Repackages fullargspec information into an equivalent inspect.Signature."""
  defaults = _make_default_values(fullargspec)
  parameters = []

  for arg in fullargspec.args:
    parameters.append(
        inspect.Parameter(
            arg,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=defaults.get(arg, inspect.Parameter.empty),
        )
    )

  if fullargspec.varargs is not None:
    parameters.append(
        inspect.Parameter(fullargspec.varargs, inspect.Parameter.VAR_POSITIONAL)
    )

  for kwarg in fullargspec.kwonlyargs:
    parameters.append(
        inspect.Parameter(
            kwarg,
            inspect.Parameter.KEYWORD_ONLY,
            default=defaults.get(kwarg, inspect.Parameter.empty),
        )
    )

  if fullargspec.varkw is not None:
    parameters.append(
        inspect.Parameter(fullargspec.varkw, inspect.Parameter.VAR_KEYWORD)
    )

  return inspect.Signature(parameters)


def make_decorator(target,
                   decorator_func,
                   decorator_name=None,
                   decorator_doc='',
                   decorator_argspec=None):
  # [FUNCTION DOC]
  # Turns decorator_func (the wrapper) into an introspectable decorator of target and returns it.
  # Inputs: target, the callable being wrapped; decorator_func, the wrapper; decorator_name,
  # optional name (defaults to the name of the calling function found via the call stack);
  # decorator_doc, optional doc text specific to this application; decorator_argspec, optional
  # argument-spec record used to override the signature.
  # Side effects: mutates decorator_func by attaching a TFDecorator record, copying the name,
  # qualified name, module, doc and any missing dictionary entries from target, setting the
  # wrapped and original-wrapped references, and setting a signature (from the override, or
  # from the target when it can be inspected; failures to inspect are silently ignored). When the
  # target is bound to an instance whose signature starts with self, that first parameter is
  # dropped and the bound instance is recorded. Returns decorator_func.
  """Make a decorator from a wrapper and a target.

  Args:
    target: The final callable to be wrapped.
    decorator_func: The wrapper function.
    decorator_name: The name of the decorator. If `None`, the name of the
      function calling make_decorator.
    decorator_doc: Documentation specific to this application of
      `decorator_func` to `target`.
    decorator_argspec: Override the signature using FullArgSpec.

  Returns:
    The `decorator_func` argument with new metadata attached.
  """
  if decorator_name is None:
    decorator_name = inspect.currentframe().f_back.f_code.co_name
  decorator = TFDecorator(decorator_name, target, decorator_doc,
                          decorator_argspec)
  setattr(decorator_func, '_tf_decorator', decorator)
  # Objects that are callables (e.g., a functools.partial object) may not have
  # the following attributes.
  if hasattr(target, '__name__'):
    decorator_func.__name__ = target.__name__
  if hasattr(target, '__qualname__'):
    decorator_func.__qualname__ = target.__qualname__
  if hasattr(target, '__module__'):
    decorator_func.__module__ = target.__module__
  if hasattr(target, '__dict__'):
    # Copy dict entries from target which are not overridden by decorator_func.
    for name in target.__dict__:
      if name not in decorator_func.__dict__:
        decorator_func.__dict__[name] = target.__dict__[name]
  if hasattr(target, '__doc__'):
    decorator_func.__doc__ = decorator.__doc__
  decorator_func.__wrapped__ = target
  # Keeping a second handle to `target` allows callers to detect whether the
  # decorator was modified using `rewrap`.
  decorator_func.__original_wrapped__ = target
  if decorator_argspec:
    decorator_func.__signature__ = fullargspec_to_signature(
        decorator_argspec)
  elif callable(target):
    try:
      signature = inspect.signature(target)
    except (TypeError, ValueError):
      # Certain callables such as builtins can not be inspected for signature.
      pass
    else:
      bound_instance = _get_bound_instance(target)
      # Present the decorated func as a method as well
      if bound_instance and 'self' in signature.parameters:
        signature = inspect.Signature(list(signature.parameters.values())[1:])
        decorator_func.__self__ = bound_instance

      decorator_func.__signature__ = signature

  return decorator_func


def _get_bound_instance(target):
  # [FUNCTION DOC]
  # HELPER: used by make_decorator to find the instance that a wrapped bound method belongs to.
  # KNOWN ISSUE: the unwrapped final target returned by unwrap is assigned but never used, and the
  # function returns nothing at all (the null value) when no decorator wraps a bound method.
  # Unwraps the given target into its decorators and returns the owning instance of the first
  # decorator whose decorated target is a bound method. No side effects.
  """Returns the instance any of the targets is attached to."""
  decorators, target = unwrap(target)
  for decorator in decorators:
    if inspect.ismethod(decorator.decorated_target):
      return decorator.decorated_target.__self__


def _has_tf_decorator_attr(obj):
  # [FUNCTION DOC]
  # HELPER: used by unwrap and rewrap to detect decorated objects.
  # KNOWN ISSUE: the docstring has no return section; the function returns a boolean that is true
  # only when obj has a decorator-record attribute whose value really is a TFDecorator (so
  # mock objects with auto-created attributes are rejected). No side effects.
  """Checks if object has _tf_decorator attribute.

  This check would work for mocked object as well since it would
  check if returned attribute has the right type.

  Args:
    obj: Python object.
  """
  return (hasattr(obj, '_tf_decorator') and
          isinstance(getattr(obj, '_tf_decorator'), TFDecorator))


def rewrap(decorator_func, previous_target, new_target):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says decorator_func is the callable returned by "wrap", but no such
  # function exists; it is the result of make_decorator.
  # Replaces previous_target with new_target inside a decorator chain built by make_decorator.
  # Walks from decorator_func through the chain of decorator records until it reaches the record
  # whose decorated target is previous_target (or the innermost one), then sets that record's
  # target to new_target and updates the wrapped reference on the matching function (for bound
  # methods, on the underlying function). If decorator_func is not a decorator at all, it must be
  # identical to previous_target (checked with an assertion) and new_target is returned.
  # Returns the updated decorator_func (mutated in place) or new_target. Raises an assertion
  # error if a chain link has no target or the consistency check fails.
  """Injects a new target into a function built by make_decorator.

  This function allows replacing a function wrapped by `decorator_func`,
  assuming the decorator that wraps the function is written as described below.

  The decorator function must use `<decorator name>.__wrapped__` instead of the
  wrapped function that is normally used:

  Example:

      # Instead of this:
      def simple_parametrized_wrapper(*args, **kwds):
        return wrapped_fn(*args, **kwds)

      tf_decorator.make_decorator(simple_parametrized_wrapper, wrapped_fn)

      # Write this:
      def simple_parametrized_wrapper(*args, **kwds):
        return simple_parametrized_wrapper.__wrapped__(*args, **kwds)

      tf_decorator.make_decorator(simple_parametrized_wrapper, wrapped_fn)

  Note that this process modifies decorator_func.

  Args:
    decorator_func: Callable returned by `wrap`.
    previous_target: Callable that needs to be replaced.
    new_target: Callable to replace previous_target with.

  Returns:
    The updated decorator. If decorator_func is not a tf_decorator, new_target
    is returned.
  """
  # Because the process mutates the decorator, we only need to alter the
  # innermost function that wraps previous_target.
  cur = decorator_func
  innermost_decorator = None
  target = None
  while _has_tf_decorator_attr(cur):
    innermost_decorator = cur
    target = getattr(cur, '_tf_decorator')
    if target.decorated_target is previous_target:
      break
    cur = target.decorated_target
    assert cur is not None

  # If decorator_func is not a decorator, new_target replaces it directly.
  if innermost_decorator is None:
    # Consistency check. The caller should always pass the result of
    # tf_decorator.unwrap as previous_target. If decorator_func is not a
    # decorator, that will have returned decorator_func itself.
    assert decorator_func is previous_target
    return new_target

  target.decorated_target = new_target

  if inspect.ismethod(innermost_decorator):
    # Bound methods can't be assigned attributes. Thankfully, they seem to
    # be just proxies for their unbound counterpart, and we can modify that.
    if hasattr(innermost_decorator, '__func__'):
      innermost_decorator.__func__.__wrapped__ = new_target
    elif hasattr(innermost_decorator, 'im_func'):
      innermost_decorator.im_func.__wrapped__ = new_target
    else:
      innermost_decorator.__wrapped__ = new_target
  else:
    innermost_decorator.__wrapped__ = new_target

  return decorator_func


def unwrap(maybe_tf_decorator):
  # [FUNCTION DOC]
  # Peels off decorators from a callable. Input: any object. Follows TFDecorator instances or
  # objects carrying a decorator record, collecting each decorator from outermost to innermost,
  # stopping when a non-decorator (or a record without a decorated target) is reached.
  # Returns a pair: the list of decorators (empty if none) and the final undecorated target.
  # No exceptions or side effects.
  """Unwraps an object into a list of TFDecorators and a final target.

  Args:
    maybe_tf_decorator: Any callable object.

  Returns:
    A tuple whose first element is an list of TFDecorator-derived objects that
    were applied to the final callable target, and whose second element is the
    final undecorated callable target. If the `maybe_tf_decorator` parameter is
    not decorated by any TFDecorators, the first tuple element will be an empty
    list. The `TFDecorator` list is ordered from outermost to innermost
    decorators.
  """
  decorators = []
  cur = maybe_tf_decorator
  while True:
    if isinstance(cur, TFDecorator):
      decorators.append(cur)
    elif _has_tf_decorator_attr(cur):
      decorators.append(getattr(cur, '_tf_decorator'))
    else:
      break
    if not hasattr(decorators[-1], 'decorated_target'):
      break
    cur = decorators[-1].decorated_target
  return decorators, cur


class TFDecorator(object):
  # [CLASS DOC]
  # Base class for stateful TensorFlow decorators. Holds the wrapped target, the decorator name,
  # an optional decorator doc and optional argument-spec override, and mimics the target by copying
  # its name, qualified name, doc and signature. Instances are callable (forwarding to the target)
  # and act as descriptors so they work as methods. Subclass it and override the call method for
  # decorators that need state; make_decorator creates instances of it as metadata records.
  """Base class for all TensorFlow decorators.

  TFDecorator captures and exposes the wrapped target, and provides details
  about the current decorator.
  """

  def __init__(self,
               decorator_name,
               target,
               decorator_doc='',
               decorator_argspec=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: decorator_name (string name of the decorator), target (the wrapped
    # callable), decorator_doc (optional doc text, empty by default), decorator_argspec (optional
    # argument-spec override). Stores them as private state. Copies the target's name and qualified
    # name when present; sets the doc to the decorator doc if given, otherwise the target's doc,
    # otherwise an empty string. Sets the signature from the override or from the target if it can
    # be inspected (inspection failures are ignored). Returns nothing.
    self._decorated_target = target
    self._decorator_name = decorator_name
    self._decorator_doc = decorator_doc
    self._decorator_argspec = decorator_argspec
    if hasattr(target, '__name__'):
      self.__name__ = target.__name__
    if hasattr(target, '__qualname__'):
      self.__qualname__ = target.__qualname__
    if self._decorator_doc:
      self.__doc__ = self._decorator_doc
    elif hasattr(target, '__doc__') and target.__doc__:
      self.__doc__ = target.__doc__
    else:
      self.__doc__ = ''

    if decorator_argspec:
      self.__signature__ = fullargspec_to_signature(decorator_argspec)
    elif callable(target):
      try:
        self.__signature__ = inspect.signature(target)
      except (TypeError, ValueError):
        # Certain callables such as builtins can not be inspected for signature.
        pass

  def __get__(self, instance, owner):
    # [FUNCTION DOC]
    # Descriptor hook so the decorator behaves like a function when stored on a class. Delegates
    # to the decorated target's own descriptor method with the given instance and owner class and
    # returns the result (typically a bound method). Raises an attribute error if the target has
    # no such hook.
    return self._decorated_target.__get__(instance, owner)

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Calls the decorated target with the given positional and keyword arguments and returns its
    # result; exceptions from the target propagate. Subclasses override this to add behavior.
    return self._decorated_target(*args, **kwargs)

  @property
  def decorated_target(self):
    # [FUNCTION DOC]
    # Property getter returning the wrapped target callable.
    return self._decorated_target

  @decorated_target.setter
  def decorated_target(self, decorated_target):
    # [FUNCTION DOC]
    # Property setter replacing the wrapped target with the given callable (used by rewrap).
    # Does not update the copied name, doc or signature. Returns nothing.
    self._decorated_target = decorated_target

  @property
  def decorator_name(self):
    # [FUNCTION DOC]
    # Read-only property returning the decorator's name string.
    return self._decorator_name

  @property
  def decorator_doc(self):
    # [FUNCTION DOC]
    # Read-only property returning the decorator-specific doc text (empty string if none).
    return self._decorator_doc

  @property
  def decorator_argspec(self):
    # [FUNCTION DOC]
    # Read-only property returning the argument-spec override, or the null value if none.
    return self._decorator_argspec
