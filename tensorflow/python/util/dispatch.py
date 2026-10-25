# [MODULE DOC]
# Type-based and fallback dispatch machinery for TensorFlow's Python APIs (ops). Exposes the
# fallback system (operation dispatcher base classes, a registry of global dispatchers, a helper
# that tries each registered dispatcher, and decorators that add a dispatch list to an op) and the
# type-based system (decorators that register dispatch targets for an API given type signatures,
# registries of unary/binary/binary-assert elementwise APIs and their handlers, helpers that build
# native signature/type checkers, and the decorator that adds dispatch support to Python ops).
# Several functions are exported publicly via the export decorator (experimental dispatch
# decorators, unregister function, add_dispatch_support, dispatcher base classes).
# Import-time side effects: defines module-level registries (global dispatcher list, signature
# registry, elementwise API lists and handler/target tables, instance-checker cache) and registers
# exported symbols. Relies on the native API dispatcher extension module. Not a helper module;
# it is core infrastructure used by op definitions throughout the package.
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
"""Type-based dispatch for TensorFlow's Python APIs.

"Python APIs" refers to Python functions that have been exported with
`tf_export`, such as `tf.add` and `tf.linalg.matmul`; they are sometimes also
referred to as "ops".

There are currently two dispatch systems for TensorFlow:

  * The "fallback dispatch" system calls an API's standard implementation first,
    and only tries to perform dispatch if that standard implementation raises a
    TypeError (or ValueError) exception.

  * The "type-based dispatch" system checks the types of the parameters passed
    to an API, and performs dispatch if those types match any signatures that
    have been registered for dispatch.

The fallback dispatch system was the original dispatch system, but it was
somewhat brittle and had limitations, such as an inability to support dispatch
for some operations (like convert_to_tensor).  We plan to remove the fallback
dispatch system in favor of the type-based dispatch system, once all users have
been switched over to use it.

### Fallback Dispatch

The fallback dispatch system is based on "operation dispatchers", which can be
used to override the behavior for TensorFlow ops when they are called with
otherwise unsupported argument types.  In particular, when an operation is
called with arguments that would cause it to raise a TypeError, it falls back on
its registered operation dispatchers.  If any registered dispatchers can handle
the arguments, then its result is returned. Otherwise, the original TypeError is
raised.

### Type-based Dispatch

The main interface for the type-based dispatch system is the `dispatch_for_api`
decorator, which overrides the default implementation for a TensorFlow API.
The decorated function (known as the "dispatch target") will override the
default implementation for the API when the API is called with parameters that
match a specified type signature.

### Dispatch Support

By default, dispatch support is added to the generated op wrappers for any
visible ops by default.  APIs/ops that are implemented in Python can opt in to
dispatch support using the `add_dispatch_support` decorator.
"""

import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)

from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export


# Private function attributes used to store dispatchers on TensorFlow APIs.
FALLBACK_DISPATCH_ATTR = "_tf_fallback_dispatchers"
TYPE_BASED_DISPATCH_ATTR = "_tf_type_based_dispatcher"

# OpDispatchers which should be used for all operations.
_GLOBAL_DISPATCHERS = []


################################################################################
# Fallback Dispatch
################################################################################


@tf_export("__internal__.dispatch.OpDispatcher", v1=[])
class OpDispatcher(object):
  # [CLASS DOC]
  # Abstract base class for fallback dispatchers that override a single TensorFlow op. A
  # dispatcher is registered on one op and is consulted only when the op's normal implementation
  # raises a type or value error. Subclasses override the handle method and return the sentinel
  # NOT_SUPPORTED (a class-level unique object) when they cannot handle the arguments.
  # Holds no per-instance state in the base class.
  """Abstract base class for TensorFlow operator dispatchers.

  Each operation dispatcher acts as an override handler for a single
  TensorFlow operation, and its results are used when the handler indicates
  that it can handle the operation's arguments (by returning any value other
  than `OpDispatcher.NOT_SUPPORTED`).
  """

  # Sentinel value that can be returned to indicate that an operation
  # dispatcher does not support a given set of arguments.
  NOT_SUPPORTED = object()

  def handle(self, args, kwargs):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Default handler hook: inputs are the positional arguments (sequence) and keyword arguments
    # (mapping) the op was called with. This base version ignores them and always returns the
    # NOT_SUPPORTED sentinel. Subclasses override it to return a result (or raise) when they can
    # handle the call. No side effects.
    """Handle this dispatcher's operation with the specified arguments.

    If this operation dispatcher can handle the given arguments, then
    return an appropriate value (or raise an appropriate exception).

    Args:
      args: The arguments to the operation.
      kwargs: They keyword arguments to the operation.

    Returns:
      The result of the operation, or `OpDispatcher.NOT_SUPPORTED` if this
      dispatcher can not handle the given arguments.
    """
    return self.NOT_SUPPORTED

  def register(self, op):
    # [FUNCTION DOC]
    # Registers this dispatcher on the given op by appending it to the op's fallback dispatcher
    # list attribute. Input op: a function that already has a fallback dispatch list (added by
    # add_dispatch_support or generated op wrappers). Raises an assertion error if the op has no
    # such list. Returns nothing. Side effect: mutates the op's dispatcher list.
    """Register this dispatcher as a handler for `op`.

    Args:
      op: Python function: the TensorFlow operation that should be handled. Must
        have a dispatch list (which is added automatically for generated ops,
        and can be added to Python ops using the `add_dispatch_support`
        decorator).
    """
    if not hasattr(op, FALLBACK_DISPATCH_ATTR):
      raise AssertionError("Dispatching not enabled for %s" % op)
    getattr(op, FALLBACK_DISPATCH_ATTR).append(self)


@tf_export("__internal__.dispatch.GlobalOpDispatcher", v1=[])
class GlobalOpDispatcher(object):
  # [CLASS DOC]
  # Abstract base class for fallback dispatchers that apply to every op rather than a single one.
  # Its handle method receives the op as well as the arguments. Registered instances are kept in
  # the module-level global dispatcher list. Shares the NOT_SUPPORTED sentinel with the
  # per-op dispatcher class.
  """Abstract base class for TensorFlow global operator dispatchers."""

  NOT_SUPPORTED = OpDispatcher.NOT_SUPPORTED

  def handle(self, op, args, kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the base implementation has only a docstring, so it returns nothing (null)
    # rather than the NOT_SUPPORTED sentinel; a subclass that does not override it would be
    # treated by the dispatch helper as having handled the op with a null result.
    # Hook for subclasses. Inputs: the op being called, positional arguments, keyword arguments.
    # Returns nothing in this base class. No side effects.
    """Handle the specified operation with the specified arguments."""

  def register(self):
    # [FUNCTION DOC]
    # Adds this dispatcher to the module-level global dispatcher list so it is consulted for all
    # ops. Takes no inputs, returns nothing. Side effect: mutates global state; registering twice
    # adds the instance twice.
    """Register this dispatcher as a handler for all ops."""
    _GLOBAL_DISPATCHERS.append(self)


def dispatch(op, args, kwargs):
  # [FUNCTION DOC]
  # Runs fallback dispatch for an op. Inputs: the op (must have a fallback dispatcher list,
  # otherwise an attribute error is raised), positional arguments, keyword arguments. First asks
  # each dispatcher registered on the op, then each global dispatcher, in registration order, and
  # returns the first result that is not the NOT_SUPPORTED sentinel. Returns NOT_SUPPORTED if
  # none handles the call. Exceptions raised by handlers propagate. No other side effects.
  """Returns the result from the first successful dispatcher for a given op.

  Calls the `handle` method of each `OpDispatcher` that has been registered
  to handle `op`, and returns the value from the first successful handler.

  Args:
    op: Python function: the operation to dispatch for.
    args: The arguments to the operation.
    kwargs: They keyword arguments to the operation.

  Returns:
    The result of the operation, or `NOT_SUPPORTED` if no registered
    dispatcher can handle the given arguments.
  """
  for dispatcher in getattr(op, FALLBACK_DISPATCH_ATTR):
    result = dispatcher.handle(args, kwargs)
    if result is not OpDispatcher.NOT_SUPPORTED:
      return result
  for dispatcher in _GLOBAL_DISPATCHERS:
    result = dispatcher.handle(op, args, kwargs)
    if result is not OpDispatcher.NOT_SUPPORTED:
      return result
  return OpDispatcher.NOT_SUPPORTED


class _TypeBasedDispatcher(OpDispatcher):
  # [CLASS DOC]
  # HELPER: implements the legacy dispatch_for_types decorator.
  # Fallback dispatcher that delegates to an override function when any argument (or element of a
  # list or tuple argument) is an instance of one of the configured types. State: the tuple of
  # types and the override function.
  """Dispatcher that handles op if any arguments have a specified type.

  Checks the types of the arguments and keyword arguments (including elements
  of lists or tuples), and if any argument values have the indicated type(s),
  then delegates to an override function.
  """

  def __init__(self, override_func, types):
    # [FUNCTION DOC]
    # Constructor. Stores the override function (called with the original arguments when a match
    # is found) and the tuple of types to look for. Returns nothing.
    self._types = types
    self._override_func = override_func

  def _handles(self, args, kwargs):
    # [FUNCTION DOC]
    # HELPER: type-matching predicate for the handle method of this dispatcher.
    # Inputs: positional arguments and keyword arguments. Returns true if any positional or
    # keyword value is an instance of the configured types, or is a list or tuple containing such
    # an element (one level deep only); otherwise false. No side effects.
    for arg in itertools.chain(args, kwargs.values()):
      if (isinstance(arg, self._types) or
          (isinstance(arg, (list, tuple)) and
           any(isinstance(elt, self._types) for elt in arg))):
        return True
    return False

  def handle(self, args, kwargs):
    # [FUNCTION DOC]
    # Calls the override function with the original positional and keyword arguments and returns
    # its result if any argument matches the configured types; otherwise returns the
    # NOT_SUPPORTED sentinel. Exceptions from the override function propagate.
    if self._handles(args, kwargs):
      return self._override_func(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED


def _remove_annotation(sig):
  # [FUNCTION DOC]
  # HELPER: normalizes signatures so get_compatible_func can compare them.
  # Input: a signature object. Returns a copy with all parameter annotations and the return
  # annotation cleared. Does not modify the input.
  """Removes annotation from a python Signature."""
  parameters = [p.replace(annotation=p.empty) for p in sig.parameters.values()]
  return sig.replace(parameters=parameters, return_annotation=sig.empty)


def _get_required_param_names(sig):
  # [FUNCTION DOC]
  # HELPER: used by get_compatible_func to compare required arguments.
  # Input: a signature object. Returns, in order, the names of parameters that have no default
  # value and are neither variadic positional nor variadic keyword. No side effects.
  """Returns a list of required parameter names from a python Signature."""
  params = []
  for p in sig.parameters.values():
    if p.kind == p.VAR_POSITIONAL:
      continue
    if p.kind == p.VAR_KEYWORD:
      continue
    if p.default is not p.empty:
      continue
    params.append(p.name)
  return params


def get_compatible_func(op, func):
  # [FUNCTION DOC]
  # Adapts func so it can be called with the calling convention of op. Inputs: op, a callable
  # whose signature is the target; func, the callable to wrap. If the annotation-stripped
  # signatures are equal, returns func itself. Otherwise requires that the required (no default)
  # parameter names of both match in order, and that every op parameter missing from func has a
  # default. Raises an assertion error if either requirement fails. The returned wrapper binds
  # arguments to op's signature, drops the parameters func lacks, raises an assertion error if
  # a dropped parameter was given a non-default value, then calls func with the rest and returns
  # its result.
  """Returns a compatible function.

  Args:
    op: a callable with whose signature the returned function is compatible.
    func: a callable which is called by the returned function.

  Returns:
    a compatible function, which conducts the actions of `func` but can
    be called like `op`, given that:
      - the list of required arguments in `func` and `op` are the same.
      - there is no override of the default arguments of `op` that are not
        supported by `func`.
  """
  op_signature = _remove_annotation(tf_inspect.signature(op))
  func_signature = _remove_annotation(tf_inspect.signature(func))

  # Identical signatures, no need to apply compatibility fixes.
  if op_signature == func_signature:
    return func

  # When calling func:
  # - Positional args without default must be in the same order.
  # - Ignore missing optional arguments from op

  op_pos_names = _get_required_param_names(op_signature)
  func_pos_names = _get_required_param_names(func_signature)

  if op_pos_names != func_pos_names:
    raise AssertionError(
        "The decorated function's non-default arguments must be identical"
        " to that of the overridden op."
        f" func has {func_pos_names}. op has {op_pos_names}."
    )

  func_missing_params = {}

  for name in set(op_signature.parameters.keys()) - set(
      func_signature.parameters.keys()
  ):
    p = op_signature.parameters[name]
    if p.default is p.empty:
      raise AssertionError(
          "The decorated function's signature must implement all of the"
          f" non-default arguments of the overridden op. Argument `{name}` is"
          " unimplemented."
      )
    func_missing_params[name] = p

  def compatible_func(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper returned by get_compatible_func.
    # Binds the call arguments to the op's signature (raising a type error if they do not fit),
    # removes arguments for parameters the wrapped function does not support, raising an assertion
    # error if any was passed with a non-default value, then calls the wrapped function with the
    # remaining positional and keyword arguments and returns its result.
    bound = op_signature.bind(*args, **kwargs)
    for name, param in func_missing_params.items():
      if name not in bound.arguments:
        continue
      value = bound.arguments.pop(name)
      if value is not param.default:
        raise AssertionError(
            f"Dispatched op is called with argument `{name}` set to a"
            " non-default value, which is not supported by the decorated"
            " function"
        )
    return func(*bound.args, **bound.kwargs)

  return compatible_func


# pylint: disable=g-doc-return-or-yield
def dispatch_for_types(op, *types):
  # [FUNCTION DOC]
  # Legacy fallback-dispatch decorator factory. Inputs: op, the function to override, and any
  # number of types. Returns a decorator that, when applied to a function, builds a compatible
  # version of it for the op's signature (which may raise an assertion error), registers a
  # type-based dispatcher on the op, and returns the function unchanged. Side effect: mutates the
  # op's fallback dispatcher list; raises an assertion error if op has none.
  """Decorator to declare that a Python function overrides an op for a type.

  The decorated function is used to override `op` if any of the arguments or
  keyword arguments (including elements of lists or tuples) have one of the
  specified types.

  Example:

  ```python
  @dispatch_for_types(math_ops.add, RaggedTensor, RaggedTensorValue)
  def ragged_add(x, y, name=None): ...
  ```

  Args:
    op: Python function: the operation that should be overridden.
    *types: The argument types for which this function should be used.
  """

  def decorator(func):
    # [FUNCTION DOC]
    # Decorator returned by dispatch_for_types. Input: the override function. Registers a
    # type-based dispatcher wrapping it on the captured op and returns the function unchanged.
    # Raises an assertion error on incompatible signatures or if the op has no dispatch list.

    _TypeBasedDispatcher(get_compatible_func(op, func), types).register(op)
    return func

  return decorator


# pylint: enable=g-doc-return-or-yield


def add_fallback_dispatch_list(target):
  # [FUNCTION DOC]
  # Decorator that gives a function an empty fallback dispatcher list attribute. Input: the
  # target function. Returns the same function. Raises an assertion error if the function already
  # has such a list. Side effect: sets an attribute on the function.
  """Decorator that adds a dispatch_list attribute to an op."""
  if hasattr(target, FALLBACK_DISPATCH_ATTR):
    raise AssertionError("%s already has a dispatch list" % target)
  setattr(target, FALLBACK_DISPATCH_ATTR, [])
  return target


# Alias for backwards-compatibility.
add_dispatch_list = add_fallback_dispatch_list


################################################################################
# Type-based Dispatch
################################################################################


@tf_export("experimental.dispatch_for_api")
def dispatch_for_api(api, *signatures):
  # [FUNCTION DOC]
  # Public decorator factory for type-based dispatch. Inputs: api, a TensorFlow API function that
  # supports type-based dispatch; signatures, zero or more mappings from parameter names (or
  # positions) to type annotations. Raises a value error immediately if the api has no
  # type-based dispatcher attribute. Builds one native signature checker per signature (these may
  # raise type or value errors for malformed signatures). Returns a decorator that registers the
  # dispatch target for each checker (or, with no signatures, for a signature read from the
  # target's annotations) and records the signatures in the module registry, returning the
  # (possibly name-scope-wrapped) target. Side effects: mutates the API's native dispatcher and
  # the signature registry. The docstring's API list placeholder is filled in later by
  # update_docstrings_with_api_lists.
  """Decorator that overrides the default implementation for a TensorFlow API.

  The decorated function (known as the "dispatch target") will override the
  default implementation for the API when the API is called with parameters that
  match a specified type signature.  Signatures are specified using dictionaries
  that map parameter names to type annotations.  E.g., in the following example,
  `masked_add` will be called for `tf.add` if both `x` and `y` are
  `MaskedTensor`s:

  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor

  >>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor, 'y': MaskedTensor})
  ... def masked_add(x, y, name=None):
  ...   return MaskedTensor(x.values + y.values, x.mask & y.mask)

  >>> mt = tf.add(MaskedTensor([1, 2], [True, False]), MaskedTensor(10, True))
  >>> print(f"values={mt.values.numpy()}, mask={mt.mask.numpy()}")
  values=[11 12], mask=[ True False]

  If multiple type signatures are specified, then the dispatch target will be
  called if any of the signatures match.  For example, the following code
  registers `masked_add` to be called if `x` is a `MaskedTensor` *or* `y` is
  a `MaskedTensor`.

  >>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor}, {'y':MaskedTensor})
  ... def masked_add(x, y):
  ...   x_values = x.values if isinstance(x, MaskedTensor) else x
  ...   x_mask = x.mask if isinstance(x, MaskedTensor) else True
  ...   y_values = y.values if isinstance(y, MaskedTensor) else y
  ...   y_mask = y.mask if isinstance(y, MaskedTensor) else True
  ...   return MaskedTensor(x_values + y_values, x_mask & y_mask)

  The type annotations in type signatures may be type objects (e.g.,
  `MaskedTensor`), `typing.List` values, or `typing.Union` values.   For
  example, the following will register `masked_concat` to be called if `values`
  is a list of `MaskedTensor` values:

  >>> @dispatch_for_api(tf.concat, {'values': typing.List[MaskedTensor]})
  ... def masked_concat(values, axis):
  ...   return MaskedTensor(tf.concat([v.values for v in values], axis),
  ...                       tf.concat([v.mask for v in values], axis))

  Each type signature must contain at least one subclass of `tf.CompositeTensor`
  (which includes subclasses of `tf.ExtensionType`), and dispatch will only be
  triggered if at least one type-annotated parameter contains a
  `CompositeTensor` value.  This rule avoids invoking dispatch in degenerate
  cases, such as the following examples:

  * `@dispatch_for_api(tf.concat, {'values': List[MaskedTensor]})`: Will not
    dispatch to the decorated dispatch target when the user calls
    `tf.concat([])`.

  * `@dispatch_for_api(tf.add, {'x': Union[MaskedTensor, Tensor], 'y':
    Union[MaskedTensor, Tensor]})`: Will not dispatch to the decorated dispatch
    target when the user calls `tf.add(tf.constant(1), tf.constant(2))`.

  The dispatch target's signature must match the signature of the API that is
  being overridden.  In particular, parameters must have the same names, and
  must occur in the same order.  The dispatch target may optionally elide the
  "name" parameter, in which case it will be wrapped with a call to
  `tf.name_scope` when appropriate.

  Args:
    api: The TensorFlow API to override.
    *signatures: Dictionaries mapping parameter names or indices to type
      annotations, specifying when the dispatch target should be called.  In
      particular, the dispatch target will be called if any signature matches;
      and a signature matches if all of the specified parameters have types that
      match with the indicated type annotations.  If no signatures are
      specified, then a signature will be read from the dispatch target
      function's type annotations.

  Returns:
    A decorator that overrides the default implementation for `api`.

  #### Registered APIs

  The TensorFlow APIs that may be overridden by `@dispatch_for_api` are:

  <<API_LIST>>
  """
  dispatcher = getattr(api, TYPE_BASED_DISPATCH_ATTR, None)
  if dispatcher is None:
    raise ValueError(f"{api} does not support dispatch.")

  api_signature = tf_inspect.signature(api)
  signature_checkers = [
      _make_signature_checker(api_signature, signature)
      for signature in signatures
  ]

  def decorator(dispatch_target):
    # [FUNCTION DOC]
    # Decorator returned by dispatch_for_api. Input: the dispatch target, which must be callable
    # (otherwise a type error is raised). Wraps it to accept a name argument if the API has one,
    # checks its signature against the API (value error on mismatch), registers it with the
    # API's native dispatcher for every signature checker, and records the signatures. Returns
    # the wrapped target.
    """Decorator that registers the given dispatch target."""
    if not callable(dispatch_target):
      raise TypeError("Expected dispatch_target to be callable; "
                      f"got {dispatch_target!r}")
    dispatch_target = _add_name_scope_wrapper(dispatch_target, api_signature)
    _check_signature(api_signature, dispatch_target)

    for signature_checker in signature_checkers:
      dispatcher.Register(signature_checker, dispatch_target)
    _TYPE_BASED_DISPATCH_SIGNATURES[api][dispatch_target].extend(signatures)

    if not signature_checkers:
      signature = _signature_from_annotations(dispatch_target)
      checker = _make_signature_checker(api_signature, signature)
      dispatcher.Register(checker, dispatch_target)
      _TYPE_BASED_DISPATCH_SIGNATURES[api][dispatch_target].append(signature)

    return dispatch_target

  return decorator


# Nested dict mapping `api_func` -> `dispatch_target` -> `List[signature]`,
# which can be used for documentation generation and for improved error messages
# when APIs are called with unsupported types.
_TYPE_BASED_DISPATCH_SIGNATURES = {}


def apis_with_type_based_dispatch():
  # [FUNCTION DOC]
  # Takes no inputs. Returns a list of the API functions that have a registered type-based
  # dispatcher, sorted by module name plus function name. No side effects.
  """Returns a list of TensorFlow APIs that support type-based dispatch."""
  return sorted(
      _TYPE_BASED_DISPATCH_SIGNATURES,
      key=lambda api: f"{api.__module__}.{api.__name__}")


def type_based_dispatch_signatures_for(cls):
  # [FUNCTION DOC]
  # Documentation-generation helper. Input: cls, a class to search for. Scans the signature
  # registry and returns a mapping from each API to the list of its registered signatures that
  # mention cls anywhere in their annotations (recursing through generic list and union
  # annotations). APIs with no matching signature are omitted. No side effects.
  """Returns dispatch signatures that have been registered for a given class.

  This function is intended for documentation-generation purposes.

  Args:
    cls: The class to search for.  Type signatures are searched recursively, so
      e.g., if `cls=RaggedTensor`, then information will be returned for all
      dispatch targets that have `RaggedTensor` anywhere in their type
      annotations (including nested in `typing.Union` or `typing.List`.)

  Returns:
    A `dict` mapping `api` -> `signatures`, where `api` is a TensorFlow API
    function; and `signatures` is a list of dispatch signatures for `api`
    that include `cls`.  (Each signature is a dict mapping argument names to
    type annotations; see `dispatch_for_api` for more info.)
  """

  def contains_cls(x):
    # [FUNCTION DOC]
    # HELPER: recursive predicate for type_based_dispatch_signatures_for.
    # Input: a signature mapping, annotation, or any value. Returns true if it is the target
    # class, or if it is a mapping whose values contain it, or a generic list or union whose type
    # arguments contain it; otherwise false.
    """Returns true if `x` contains `cls`."""
    if isinstance(x, dict):
      return any(contains_cls(v) for v in x.values())
    elif x is cls:
      return True
    elif (type_annotations.is_generic_list(x) or
          type_annotations.is_generic_union(x)):
      type_args = type_annotations.get_generic_type_args(x)
      return any(contains_cls(arg) for arg in type_args)
    else:
      return False

  result = {}
  for api, api_signatures in _TYPE_BASED_DISPATCH_SIGNATURES.items():
    for _, signatures in api_signatures.items():
      filtered = list(filter(contains_cls, signatures))
      if filtered:
        result.setdefault(api, []).extend(filtered)
  return result


# TODO(edloper): Consider using a mechanism like this to automatically add
# the `name` argument to all TensorFlow APIs that are implemented in Python
# (so each Python function doesn't need to do it manually).
def _add_name_scope_wrapper(func, api_signature):
  # [FUNCTION DOC]
  # HELPER: lets dispatch targets omit the name parameter of an API.
  # Inputs: func, the dispatch target; api_signature, signature of the API. If the API has no name
  # parameter, or func already accepts a name (explicitly or via variadic keywords), returns func
  # unchanged. Otherwise returns a wrapper that extracts the name from positional or keyword
  # arguments, runs func inside a name scope when a name was given, and calls func directly
  # otherwise. The wrapper keeps func's identity via the decorator utility, has its signature
  # extended with the API's name parameter, and has its decorator marker attribute removed.
  """Wraps `func` to expect a "name" arg, and use it to call `ops.name_scope`.

  If `func` already expects a "name" arg, or if `api_signature` does not
  expect a "name" arg, then returns `func` as-is.

  Args:
    func: The function to wrap.  Signature must match `api_signature` (except
      the "name" parameter may be missing.
    api_signature: The signature of the original API (used to find the index for
      the "name" parameter).

  Returns:
    The wrapped function (or the original function if no wrapping is needed).
  """
  if "name" not in api_signature.parameters:
    return func  # no wrapping needed (API has no name parameter).

  func_signature = tf_inspect.signature(func)
  func_argspec = tf_inspect.getargspec(func)
  if "name" in func_signature.parameters or func_argspec.keywords is not None:
    return func  # No wrapping needed (already has name parameter).

  name_index = list(api_signature.parameters).index("name")

  def wrapped_func(*args, **kwargs):
    # [FUNCTION DOC]
    # Wrapper built by _add_name_scope_wrapper. Pulls the name out of the positional arguments
    # (at the API's name position) or from keyword arguments (defaulting to nothing), then calls
    # the wrapped function with the remaining arguments, inside a name scope if a name was given.
    # Returns the wrapped function's result.
    if name_index < len(args):
      name = args[name_index]
      args = args[:name_index] + args[name_index + 1:]
    else:
      name = kwargs.pop("name", None)
    if name is None:
      return func(*args, **kwargs)
    else:
      with ops.name_scope(name):
        return func(*args, **kwargs)

  wrapped_func = tf_decorator.make_decorator(func, wrapped_func)
  wrapped_func.__signature__ = func_signature.replace(
      parameters=(list(func_signature.parameters.values()) +
                  [api_signature.parameters["name"]]))
  del wrapped_func._tf_decorator
  return wrapped_func


@tf_export("experimental.unregister_dispatch_for")
def unregister_dispatch_for(dispatch_target):
  # [FUNCTION DOC]
  # Removes a previously registered dispatch target (mainly for tests). Input: the target
  # function. Looks it up in the type-based signature registry (unregistering from each API's
  # native dispatcher and deleting the record) and in the elementwise handler table (recursively
  # unregistering every generated per-API target and deleting the handler and target records).
  # Raises a value error if it was found in neither. Returns nothing.
  """Unregisters a function that was registered with `@dispatch_for_*`.

  This is primarily intended for testing purposes.

  Example:

  >>> # Define a type and register a dispatcher to override `tf.abs`:
  >>> class MyTensor(tf.experimental.ExtensionType):
  ...   value: tf.Tensor
  >>> @tf.experimental.dispatch_for_api(tf.abs)
  ... def my_abs(x: MyTensor):
  ...   return MyTensor(tf.abs(x.value))
  >>> tf.abs(MyTensor(5))
  MyTensor(value=<tf.Tensor: shape=(), dtype=int32, numpy=5>)

  >>> # Unregister the dispatcher, so `tf.abs` no longer calls `my_abs`.
  >>> unregister_dispatch_for(my_abs)
  >>> tf.abs(MyTensor(5))
  Traceback (most recent call last):
  ...
  ValueError: Attempt to convert a value ... to a Tensor.

  Args:
    dispatch_target: The function to unregister.

  Raises:
    ValueError: If `dispatch_target` was not registered using `@dispatch_for`,
      `@dispatch_for_unary_elementwise_apis`, or
      `@dispatch_for_binary_elementwise_apis`.
  """
  found = False

  # Check if dispatch_target registered by `@dispatch_for_api`
  for api, signatures in _TYPE_BASED_DISPATCH_SIGNATURES.items():
    if dispatch_target in signatures:
      dispatcher = getattr(api, TYPE_BASED_DISPATCH_ATTR)
      dispatcher.Unregister(dispatch_target)
      del signatures[dispatch_target]
      found = True

  # Check if dispatch_target registered by `@dispatch_for_*_elementwise_apis`
  elementwise_keys_to_delete = [
      key for (key, handler) in _ELEMENTWISE_API_HANDLERS.items()
      if handler is dispatch_target
  ]
  for key in set(elementwise_keys_to_delete):
    for _, target in _ELEMENTWISE_API_TARGETS[key]:
      unregister_dispatch_for(target)
    del _ELEMENTWISE_API_HANDLERS[key]
    del _ELEMENTWISE_API_TARGETS[key]
    found = True

  if not found:
    raise ValueError(f"Function {dispatch_target} was not registered using "
                     "a `@dispatch_for_*` decorator.")


def register_dispatchable_type(cls):
  # [FUNCTION DOC]
  # Registers a class with the native dispatcher so its instances trigger type-based dispatch.
  # Input: cls, the class (not a composite tensor or extension type, which are registered
  # automatically). Returns cls. Side effect: updates native global dispatcher state.
  """Class decorator that registers a type for use with type-based dispatch.

  Should *not* be used with subclasses of `CompositeTensor` or `ExtensionType`
  (which are automatically registered).

  Note: this function is intended to support internal legacy use cases (such
  as RaggedTensorValue), and will probably not be exposed as a public API.

  Args:
    cls: The class to register.

  Returns:
    `cls`.
  """
  _api_dispatcher.register_dispatchable_type(cls)
  return cls


def add_type_based_api_dispatcher(target):
  # [FUNCTION DOC]
  # Attaches a native API dispatcher to an API function. Input: target, the (possibly decorated)
  # function. Raises a value error if it already has one. If the unwrapped function takes
  # variadic positional or keyword arguments, returns target unchanged with no dispatcher
  # (unsupported, see the bug reference in the code). Otherwise sets the dispatcher attribute on
  # target (built from the function name, argument names and defaults), creates an empty
  # signature registry entry for it, and returns target.
  """Adds a PythonAPIDispatcher to the given TensorFlow API function."""
  if hasattr(target, TYPE_BASED_DISPATCH_ATTR):
    raise ValueError(f"{target} already has a type-based API dispatcher.")

  _, unwrapped = tf_decorator.unwrap(target)
  target_argspec = tf_inspect.getargspec(unwrapped)
  if target_argspec.varargs or target_argspec.keywords:
    # @TODO(b/194903203) Add v2 dispatch support for APIs that take varargs
    # and keywords.  Examples of APIs that take varargs and kwargs: meshgrid,
    # einsum, map_values, map_flat_values.
    return target

  setattr(
      target, TYPE_BASED_DISPATCH_ATTR,
      _api_dispatcher.PythonAPIDispatcher(unwrapped.__name__,
                                          target_argspec.args,
                                          target_argspec.defaults))
  _TYPE_BASED_DISPATCH_SIGNATURES[target] = collections.defaultdict(list)
  return target


def _check_signature(api_signature, func):
  # [FUNCTION DOC]
  # HELPER: validates dispatch targets in the decorator from dispatch_for_api.
  # Inputs: the API's signature and the candidate function. Returns nothing. A function taking
  # only variadic positional and keyword arguments is accepted without checks. Otherwise raises
  # a value error unless the parameter counts match and each corresponding parameter has the
  # same name and kind.
  """Checks that a dispatch target's signature is compatible with an API.

  Args:
    api_signature: The signature of the TensorFlow API.
    func: The dispatch target.

  Raises:
    ValueError: if the signatures are incompatible.  Two signatures are
      considered compatible if they have the same number of parameters, and all
      corresponding parameters have the same `name` and `kind`.  (Parameters
      are not required to have the same default value or the same annotation.)
  """
  # Special case: if func_signature is (*args, **kwargs), then assume it's ok.
  func_argspec = tf_inspect.getargspec(func)
  if (func_argspec.varargs is not None and func_argspec.keywords is not None
      and not func_argspec.args):
    return

  func_signature = tf_inspect.signature(func)
  ok = len(api_signature.parameters) == len(func_signature.parameters)
  if ok:
    for param_1, param_2 in zip(api_signature.parameters.values(),
                                func_signature.parameters.values()):
      if (param_1.name != param_2.name) or (param_1.kind != param_2.kind):
        ok = False
  if not ok:
    raise ValueError(f"Dispatch function's signature {func_signature} does "
                     f"not match API's signature {api_signature}.")


def _make_signature_checker(api_signature, signature):
  # [FUNCTION DOC]
  # HELPER: converts a user type signature into a native checker for dispatch_for_api.
  # Inputs: api_signature, the API's signature; signature, a mapping from parameter names or
  # integer positions to type annotations. Raises a type error if signature is not such a
  # mapping, a value error if a parameter is unknown or is not positional(-or-keyword). Integer
  # positions within range are converted to names. Returns a native signature checker built from
  # (parameter index, type checker) pairs.
  """Builds a PySignatureChecker for the given type signature.

  Args:
    api_signature: The `inspect.Signature` of the API whose signature is
      being checked.
    signature: Dictionary mapping parameter names to type annotations.

  Returns:
    A `PySignatureChecker`.
  """
  if not (isinstance(signature, dict) and
          all(isinstance(k, (str, int)) for k in signature)):
    raise TypeError("signatures must be dictionaries mapping parameter names "
                    "to type annotations.")
  checkers = []

  param_names = list(api_signature.parameters)
  for param_name, param_type in signature.items():
    # Convert positional parameters to named parameters.
    if (isinstance(param_name, int) and
        param_name < len(api_signature.parameters)):
      param_name = list(api_signature.parameters.values())[param_name].name

    # Check that the parameter exists, and has an appropriate kind.
    param = api_signature.parameters.get(param_name, None)
    if param is None:
      raise ValueError("signature includes annotation for unknown "
                       f"parameter {param_name!r}.")
    if param.kind not in (tf_inspect.Parameter.POSITIONAL_ONLY,
                          tf_inspect.Parameter.POSITIONAL_OR_KEYWORD):
      raise ValueError("Dispatch currently only supports type annotations "
                       "for positional parameters; can't handle annotation "
                       f"for {param.kind!r} parameter {param_name}.")

    checker = make_type_checker(param_type)
    index = param_names.index(param_name)
    checkers.append((index, checker))

  return _api_dispatcher.PySignatureChecker(checkers)


# Cache for InstanceTypeChecker objects (we only want to create one
# InstanceTypeChecker for each type, since each one uses an internal cache
# to avoid repeated calls back into Python's isinstance).
_is_instance_checker_cache = {}


def make_type_checker(annotation):
  # [FUNCTION DOC]
  # Builds a native type checker from a type annotation. Supported: unions (several plain types
  # are merged into one cached instance checker combined with checkers for the other members),
  # lists (exactly one element type, else assertion error), plain classes (cached instance
  # checker per class), and the null value (checks the null type). Any other annotation raises a
  # value error. Side effect: fills the module instance-checker cache.
  """Builds a PyTypeChecker for the given type annotation."""
  if type_annotations.is_generic_union(annotation):
    type_args = type_annotations.get_generic_type_args(annotation)

    # If the union contains two or more simple types, then use a single
    # InstanceChecker to check them.
    simple_types = [t for t in type_args if isinstance(t, type)]
    simple_types = tuple(sorted(simple_types, key=id))
    if len(simple_types) > 1:
      if simple_types not in _is_instance_checker_cache:
        checker = _api_dispatcher.MakeInstanceChecker(*simple_types)
        _is_instance_checker_cache[simple_types] = checker
      options = ([_is_instance_checker_cache[simple_types]] +
                 [make_type_checker(t) for t in type_args
                  if not isinstance(t, type)])
      return _api_dispatcher.MakeUnionChecker(options)

    options = [make_type_checker(t) for t in type_args]
    return _api_dispatcher.MakeUnionChecker(options)

  elif type_annotations.is_generic_list(annotation):
    type_args = type_annotations.get_generic_type_args(annotation)
    if len(type_args) != 1:
      raise AssertionError("Expected List[...] to have a single type parameter")
    elt_type = make_type_checker(type_args[0])
    return _api_dispatcher.MakeListChecker(elt_type)

  elif isinstance(annotation, type):
    if annotation not in _is_instance_checker_cache:
      checker = _api_dispatcher.MakeInstanceChecker(annotation)
      _is_instance_checker_cache[annotation] = checker
    return _is_instance_checker_cache[annotation]

  elif annotation is None:
    return make_type_checker(type(None))

  else:
    raise ValueError(f"Type annotation {annotation} is not currently supported"
                     " by dispatch.  Supported annotations: type objects, "
                     " List[...], and Union[...]")


def _signature_from_annotations(func):
  # [FUNCTION DOC]
  # HELPER: supplies the default signature for dispatch_for_api when none is given.
  # Input: a function. Returns a mapping from each annotated parameter name to its annotation.
  # Raises a value error if no parameter is annotated.
  """Builds a dict mapping from parameter names to type annotations."""
  func_signature = tf_inspect.signature(func)

  signature = dict([(name, param.annotation)
                    for (name, param) in func_signature.parameters.items()
                    if param.annotation != tf_inspect.Parameter.empty])
  if not signature:
    raise ValueError("The dispatch_for_api decorator must be called with at "
                     "least one signature, or applied to a function that "
                     "has type annotations on its parameters.")
  return signature


# Registries for elementwise APIs and API handlers.
#
# _*_ELEMENTWISE_APIS: A list of TensorFlow APIs that have been registered
# as elementwise operations using the `register_*_elementwise_api`
# decorators.
#
# _ELEMENTWISE_API_HANDLERS: Dicts mapping from argument type(s) to API
# handlers that have been registered with the `dispatch_for_*_elementwise_apis`
# decorators.
#
# _ELEMENTWISE_API_TARGETS: Dict mapping from argument type(s) to lists of
# `(api, dispatch_target)` pairs.  Used to implement
# `unregister_elementwise_api_handler`.
_UNARY_ELEMENTWISE_APIS = []
_BINARY_ELEMENTWISE_APIS = []
_BINARY_ELEMENTWISE_ASSERT_APIS = []
_ELEMENTWISE_API_HANDLERS = {}
_ELEMENTWISE_API_TARGETS = {}

_ASSERT_API_TAG = "ASSERT_API_TAG"


@tf_export("experimental.dispatch_for_unary_elementwise_apis")
def dispatch_for_unary_elementwise_apis(x_type):
  # [FUNCTION DOC]
  # Public decorator factory that overrides all unary elementwise APIs for one argument type.
  # Input: x_type, a type annotation (class, list or union form). Returns a decorator taking the
  # handler (a function receiving the API function and the first argument). The docstring's API
  # list placeholder is filled in later by update_docstrings_with_api_lists.
  """Decorator to override default implementation for unary elementwise APIs.

  The decorated function (known as the "elementwise api handler") overrides
  the default implementation for any unary elementwise API whenever the value
  for the first argument (typically named `x`) matches the type annotation
  `x_type`. The elementwise api handler is called with two arguments:

    `elementwise_api_handler(api_func, x)`

  Where `api_func` is a function that takes a single parameter and performs the
  elementwise operation (e.g., `tf.abs`), and `x` is the first argument to the
  elementwise api.

  The following example shows how this decorator can be used to update all
  unary elementwise operations to handle a `MaskedTensor` type:

  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_unary_elementwise_apis(MaskedTensor)
  ... def unary_elementwise_api_handler(api_func, x):
  ...   return MaskedTensor(api_func(x.values), x.mask)
  >>> mt = MaskedTensor([1, -2, -3], [True, False, True])
  >>> abs_mt = tf.abs(mt)
  >>> print(f"values={abs_mt.values.numpy()}, mask={abs_mt.mask.numpy()}")
  values=[1 2 3], mask=[ True False True]

  For unary elementwise operations that take extra arguments beyond `x`, those
  arguments are *not* passed to the elementwise api handler, but are
  automatically added when `api_func` is called.  E.g., in the following
  example, the `dtype` parameter is not passed to
  `unary_elementwise_api_handler`, but is added by `api_func`.

  >>> ones_mt = tf.ones_like(mt, dtype=tf.float32)
  >>> print(f"values={ones_mt.values.numpy()}, mask={ones_mt.mask.numpy()}")
  values=[1.0 1.0 1.0], mask=[ True False True]

  Args:
    x_type: A type annotation indicating when the api handler should be called.
      See `dispatch_for_api` for a list of supported annotation types.

  Returns:
    A decorator.

  #### Registered APIs

  The unary elementwise APIs are:

  <<API_LIST>>
  """

  def decorator(handler):
    # [FUNCTION DOC]
    # Decorator returned by the unary elementwise factory. Input: the handler. Raises a value
    # error if a unary handler is already registered for the same type. Otherwise records the
    # handler, registers a dispatch target for every currently known unary elementwise API, and
    # returns the handler. Side effects: mutates the handler registry, target registry and the
    # APIs' dispatchers.
    if (x_type,) in _ELEMENTWISE_API_HANDLERS:
      raise ValueError("A unary elementwise dispatch handler "
                       f"({_ELEMENTWISE_API_HANDLERS[(x_type,)]}) "
                       f"has already been registered for {x_type}.")
    _ELEMENTWISE_API_HANDLERS[(x_type,)] = handler
    for api in _UNARY_ELEMENTWISE_APIS:
      _add_dispatch_for_unary_elementwise_api(api, x_type, handler)

    return handler

  return decorator


@tf_export("experimental.dispatch_for_binary_elementwise_apis")
def dispatch_for_binary_elementwise_apis(x_type, y_type):
  # [FUNCTION DOC]
  # Public decorator factory that overrides all binary elementwise APIs for a pair of argument
  # types. Inputs: x_type and y_type, type annotations for the first two arguments. Returns a
  # decorator taking the handler (a function receiving the API function and both arguments). The
  # docstring's API list placeholder is filled in later by update_docstrings_with_api_lists.
  """Decorator to override default implementation for binary elementwise APIs.

  The decorated function (known as the "elementwise api handler") overrides
  the default implementation for any binary elementwise API whenever the value
  for the first two arguments (typically named `x` and `y`) match the specified
  type annotations.  The elementwise api handler is called with two arguments:

    `elementwise_api_handler(api_func, x, y)`

  Where `x` and `y` are the first two arguments to the elementwise api, and
  `api_func` is a TensorFlow function that takes two parameters and performs the
  elementwise operation (e.g., `tf.add`).

  The following example shows how this decorator can be used to update all
  binary elementwise operations to handle a `MaskedTensor` type:

  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
  ... def binary_elementwise_api_handler(api_func, x, y):
  ...   return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)
  >>> a = MaskedTensor([1, 2, 3, 4, 5], [True, True, True, True, False])
  >>> b = MaskedTensor([2, 4, 6, 8, 0], [True, True, True, False, True])
  >>> c = tf.add(a, b)
  >>> print(f"values={c.values.numpy()}, mask={c.mask.numpy()}")
  values=[ 3 6 9 12 5], mask=[ True True True False False]

  Args:
    x_type: A type annotation indicating when the api handler should be called.
    y_type: A type annotation indicating when the api handler should be called.

  Returns:
    A decorator.

  #### Registered APIs

  The binary elementwise APIs are:

  <<API_LIST>>
  """

  def decorator(handler):
    # [FUNCTION DOC]
    # Decorator returned by the binary elementwise factory. Input: the handler. Raises a value
    # error if a binary handler is already registered for the same type pair. Otherwise records
    # the handler, registers a dispatch target for every currently known binary elementwise API,
    # and returns the handler. Side effects: mutates registries and the APIs' dispatchers.
    if (x_type, y_type) in _ELEMENTWISE_API_HANDLERS:
      raise ValueError("A binary elementwise dispatch handler "
                       f"({_ELEMENTWISE_API_HANDLERS[x_type, y_type]}) "
                       f"has already been registered for ({x_type}, {y_type}).")
    _ELEMENTWISE_API_HANDLERS[x_type, y_type] = handler
    for api in _BINARY_ELEMENTWISE_APIS:
      _add_dispatch_for_binary_elementwise_api(api, x_type, y_type, handler)

    return handler

  return decorator


@tf_export("experimental.dispatch_for_binary_elementwise_assert_apis")
def dispatch_for_binary_elementwise_assert_apis(x_type, y_type):
  # [FUNCTION DOC]
  # Public decorator factory that overrides all binary elementwise assert APIs (such as equality
  # assertions) for a pair of argument types. Inputs: x_type and y_type, type annotations.
  # Returns a decorator taking the handler (a function receiving the assert function and both
  # arguments). The docstring's API list placeholder is filled in later by
  # update_docstrings_with_api_lists.
  """Decorator to override default implementation for binary elementwise assert APIs.

  The decorated function (known as the "elementwise assert handler")
  overrides the default implementation for any binary elementwise assert API
  whenever the value for the first two arguments (typically named `x` and `y`)
  match the specified type annotations.  The handler is called with two
  arguments:

    `elementwise_assert_handler(assert_func, x, y)`

  Where `x` and `y` are the first two arguments to the binary elementwise assert
  operation, and `assert_func` is a TensorFlow function that takes two
  parameters and performs the elementwise assert operation (e.g.,
  `tf.debugging.assert_equal`).

  The following example shows how this decorator can be used to update all
  binary elementwise assert operations to handle a `MaskedTensor` type:

  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_binary_elementwise_assert_apis(MaskedTensor, MaskedTensor)
  ... def binary_elementwise_assert_api_handler(assert_func, x, y):
  ...   merged_mask = tf.logical_and(x.mask, y.mask)
  ...   selected_x_values = tf.boolean_mask(x.values, merged_mask)
  ...   selected_y_values = tf.boolean_mask(y.values, merged_mask)
  ...   assert_func(selected_x_values, selected_y_values)
  >>> a = MaskedTensor([1, 1, 0, 1, 1], [False, False, True, True, True])
  >>> b = MaskedTensor([2, 2, 0, 2, 2], [True, True, True, False, False])
  >>> tf.debugging.assert_equal(a, b) # assert passed; no exception was thrown

  >>> a = MaskedTensor([1, 1, 1, 1, 1], [True, True, True, True, True])
  >>> b = MaskedTensor([0, 0, 0, 0, 2], [True, True, True, True, True])
  >>> tf.debugging.assert_greater(a, b)
  Traceback (most recent call last):
  ...
  InvalidArgumentError: Condition x > y did not hold.

  Args:
    x_type: A type annotation indicating when the api handler should be called.
    y_type: A type annotation indicating when the api handler should be called.

  Returns:
    A decorator.

  #### Registered APIs

  The binary elementwise assert APIs are:

  <<API_LIST>>
  """

  def decorator(handler):
    # [FUNCTION DOC]
    # Decorator returned by the assert factory. Input: the handler. The registry key is the type
    # pair plus an assert tag, so assert handlers do not clash with ordinary binary handlers.
    # Raises a value error if one is already registered for the key. Otherwise records the
    # handler, registers a dispatch target for every known binary assert API, and returns the
    # handler. Side effects: mutates registries and the APIs' dispatchers.
    api_handler_key = (x_type, y_type, _ASSERT_API_TAG)
    if api_handler_key in _ELEMENTWISE_API_HANDLERS:
      raise ValueError("A binary elementwise assert dispatch handler "
                       f"({_ELEMENTWISE_API_HANDLERS[api_handler_key]}) "
                       f"has already been registered for ({x_type}, {y_type}).")
    _ELEMENTWISE_API_HANDLERS[api_handler_key] = handler
    for api in _BINARY_ELEMENTWISE_ASSERT_APIS:
      _add_dispatch_for_binary_elementwise_api(api, x_type, y_type, handler)

    return handler

  return decorator


def register_unary_elementwise_api(func):
  # [FUNCTION DOC]
  # Decorator that marks an op as a unary elementwise API. Input: func. Appends it to the unary
  # API list and registers a dispatch target for it for every unary handler already registered.
  # Returns func. Side effects: mutates the list, registry and func's dispatcher.
  """Decorator that registers a TensorFlow op as a unary elementwise API."""
  _UNARY_ELEMENTWISE_APIS.append(func)
  for args, handler in _ELEMENTWISE_API_HANDLERS.items():
    if len(args) == 1:
      _add_dispatch_for_unary_elementwise_api(func, args[0], handler)
  return func


def register_binary_elementwise_api(func):
  # [FUNCTION DOC]
  # Decorator that marks an op as a binary elementwise API. Input: func. Appends it to the
  # binary API list and registers a dispatch target for it for every existing handler keyed by
  # exactly two types. Returns func. Side effects: mutates the list, registry and dispatcher.
  """Decorator that registers a TensorFlow op as a binary elementwise API."""
  _BINARY_ELEMENTWISE_APIS.append(func)
  for args, handler in _ELEMENTWISE_API_HANDLERS.items():
    if len(args) == 2:
      _add_dispatch_for_binary_elementwise_api(func, args[0], args[1], handler)
  return func


def register_binary_elementwise_assert_api(func):
  # [FUNCTION DOC]
  # Decorator that marks an op as a binary elementwise assert API. Input: func. Appends it to
  # the assert API list and registers a dispatch target for it for every existing handler whose
  # key carries the assert tag. Returns func. Side effects: mutates the list, registry and
  # dispatcher.
  """Decorator that registers a TensorFlow op as a binary elementwise assert API.

  Different from `dispatch_for_binary_elementwise_apis`, this decorator is used
  for assert apis, such as assert_equal, assert_none_equal, etc, which return
  None in eager mode and an op in graph mode.

  Args:
    func: The function that implements the binary elementwise assert API.

  Returns:
    `func`
  """
  _BINARY_ELEMENTWISE_ASSERT_APIS.append(func)
  for args, handler in _ELEMENTWISE_API_HANDLERS.items():
    if len(args) == 3 and args[2] is _ASSERT_API_TAG:
      _add_dispatch_for_binary_elementwise_api(func, args[0], args[1], handler)
  return func


def unary_elementwise_apis():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a list, but it returns an immutable tuple
  # snapshot of the registered unary elementwise APIs. Takes no inputs; no side effects.
  """Returns a list of APIs that have been registered as unary elementwise."""
  return tuple(_UNARY_ELEMENTWISE_APIS)


def binary_elementwise_apis():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a list, but it returns an immutable tuple
  # snapshot of the registered binary elementwise APIs (assert APIs are not included). Takes no
  # inputs; no side effects.
  """Returns a list of APIs that have been registered as binary elementwise."""
  return tuple(_BINARY_ELEMENTWISE_APIS)


def _add_dispatch_for_unary_elementwise_api(api, x_type,
                                            elementwise_api_handler):
  # [FUNCTION DOC]
  # HELPER: glue between unary elementwise handlers and type-based dispatch.
  # Inputs: api, the unary op; x_type, the type annotation for its first parameter;
  # elementwise_api_handler, the user handler. Registers (via the type-based decorator) a
  # generated dispatch target for the api keyed on the first parameter's type. The target
  # extracts the name argument and first argument, binds any extra arguments into a one-argument
  # function (or uses the api itself if it only takes the value and name), and calls the handler,
  # inside a name scope when a name was given. The target is renamed after the api, and recorded
  # in the elementwise target registry so it can be unregistered. Returns nothing.
  """Registers a unary elementwise handler as a dispatcher for a given API."""
  api_signature = tf_inspect.signature(api)
  x_name = list(api_signature.parameters)[0]
  name_index = _find_name_index(api_signature)

  need_to_bind_api_args = (
      len(api_signature.parameters) > 2 or
      "name" not in api_signature.parameters)

  @dispatch_for_api(api, {x_name: x_type})
  def dispatch_target(*args, **kwargs):
    # [FUNCTION DOC]
    # Generated dispatch target for one unary elementwise API. Takes the API's call arguments,
    # pulls out the name, takes the first argument from positional arguments or (if absent) from
    # the keyword argument with the first parameter's name (a missing one raises a key error),
    # then calls the elementwise handler with a one-argument version of the API and the value.
    # Returns the handler's result.
    args, kwargs, name = _extract_name_arg(args, kwargs, name_index)
    if args:
      x, args = args[0], args[1:]
    else:
      x = kwargs.pop(x_name)

    if need_to_bind_api_args:
      tensor_api = lambda v: api(v, *args, **kwargs)
    else:
      tensor_api = api

    if name is None:
      return elementwise_api_handler(tensor_api, x)
    else:
      with ops.name_scope(name, None, [x]):
        return elementwise_api_handler(tensor_api, x)

  dispatch_target.__name__ = "elementwise_dispatch_target_for_" + api.__name__
  dispatch_target.__qualname__ = dispatch_target.__name__
  # Keep track of what targets we've registered (so we can unregister them).
  target_list = _ELEMENTWISE_API_TARGETS.setdefault((x_type,), [])
  target_list.append((api, dispatch_target))


def _add_dispatch_for_binary_elementwise_api(api, x_type, y_type,
                                             elementwise_api_handler):
  # [FUNCTION DOC]
  # HELPER: glue between binary elementwise handlers and type-based dispatch.
  # Inputs: api, the binary op (or assert op); x_type and y_type, annotations for its first two
  # parameters; elementwise_api_handler, the user handler. Registers a generated dispatch target
  # keyed on both types, which extracts the name, takes the two values from positional or
  # keyword arguments (missing ones become the null value), binds extra arguments into a
  # two-argument function (or uses the api directly), and calls the handler, inside a name scope
  # when a name was given. The target is renamed after the api and recorded in the elementwise
  # target registry. Returns nothing.
  """Registers a binary elementwise handler as a dispatcher for a given API."""
  api_signature = tf_inspect.signature(api)
  x_name, y_name = list(api_signature.parameters)[:2]
  name_index = _find_name_index(api_signature)

  need_to_bind_api_args = (len(api_signature.parameters) > 3 or
                           "name" not in api_signature.parameters)

  @dispatch_for_api(api, {x_name: x_type, y_name: y_type})
  def dispatch_target(*args, **kwargs):
    # [FUNCTION DOC]
    # Generated dispatch target for one binary elementwise API. Takes the API's call arguments,
    # pulls out the name, takes the first two values from positional arguments, falling back to
    # keyword arguments named after the first two parameters (missing ones become the null
    # value), then calls the elementwise handler with a two-argument version of the API and both
    # values. Returns the handler's result.
    args, kwargs, name = _extract_name_arg(args, kwargs, name_index)
    if len(args) > 1:
      x, y, args = args[0], args[1], args[2:]
    elif args:
      x, args = args[0], args[1:]
      y = kwargs.pop(y_name, None)
    else:
      x = kwargs.pop(x_name, None)
      y = kwargs.pop(y_name, None)

    if need_to_bind_api_args:
      tensor_api = lambda v1, v2: api(v1, v2, *args, **kwargs)
    else:
      tensor_api = api

    if name is None:
      return elementwise_api_handler(tensor_api, x, y)
    else:
      with ops.name_scope(name, None, [x, y]):
        return elementwise_api_handler(tensor_api, x, y)

  dispatch_target.__name__ = "elementwise_dispatch_target_for_" + api.__name__
  dispatch_target.__qualname__ = dispatch_target.__name__
  # Keep track of what targets we've registered (so we can unregister them).
  target_list = _ELEMENTWISE_API_TARGETS.setdefault((x_type, y_type), [])
  target_list.append((api, dispatch_target))


def _find_name_index(signature):
  # [FUNCTION DOC]
  # HELPER: locates the name parameter for the elementwise dispatch targets.
  # Input: a signature object. Returns the position of the parameter called name, or -1 if there
  # is none. No side effects.
  """Returns the index of the `name` parameter, or -1 if it's not present."""
  try:
    return list(signature.parameters).index("name")
  except ValueError:
    return -1


def _extract_name_arg(args, kwargs, name_index):
  # [FUNCTION DOC]
  # HELPER: separates the name argument in the elementwise dispatch targets.
  # Inputs: positional arguments, keyword arguments, and the name position (negative if the API
  # has no name). Returns a triple of the positional arguments without the name, the keyword
  # arguments, and the name value (null if absent). Side effect: removes the name entry from the
  # passed keyword mapping in place when taken from keywords.
  """Extracts the parameter `name` and returns `(args, kwargs, name_value)`."""
  if name_index < 0:
    name_value = None
  elif name_index < len(args):
    name_value = args[name_index]
    args = args[:name_index] + args[name_index + 1:]
  else:
    name_value = kwargs.pop("name", None)
  return args, kwargs, name_value


def update_docstrings_with_api_lists():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists three decorators, but the code also updates the docstring
  # of the binary elementwise assert decorator.
  # Takes no inputs, returns nothing. Replaces the API list placeholder in the docstrings of the
  # unary, binary, binary assert and dispatch_for_api decorators with the registered APIs.
  # Intended to be called once after all ops are registered; a second call finds no placeholder.
  """Updates the docstrings of dispatch decorators with API lists.

  Updates docstrings for `dispatch_for_api`,
  `dispatch_for_unary_elementwise_apis`, and
  `dispatch_for_binary_elementwise_apis`, by replacing the string '<<API_LIST>>'
  with a list of APIs that have been registered for that decorator.
  """
  _update_docstring_with_api_list(
      dispatch_for_unary_elementwise_apis, _UNARY_ELEMENTWISE_APIS)
  _update_docstring_with_api_list(
      dispatch_for_binary_elementwise_apis, _BINARY_ELEMENTWISE_APIS)
  _update_docstring_with_api_list(
      dispatch_for_binary_elementwise_assert_apis,
      _BINARY_ELEMENTWISE_ASSERT_APIS)
  _update_docstring_with_api_list(
      dispatch_for_api, _TYPE_BASED_DISPATCH_SIGNATURES)


def _update_docstring_with_api_list(target, api_list):
  # [FUNCTION DOC]
  # HELPER: used by update_docstrings_with_api_lists.
  # Inputs: target, a function whose docstring contains the API list placeholder; api_list, a
  # list of API functions (or a mapping whose keys are the APIs). For each API with a canonical
  # exported name, adds a sorted bullet line showing the public name and parameter names. Side
  # effect: rewrites the target's docstring in place, replacing the placeholder with the lines.
  # Fails with an attribute error if the target has no docstring.
  """Replaces `<<API_LIST>>` in target.__doc__ with the given list of APIs."""
  lines = []
  for func in api_list:
    if isinstance(func, dict):
      func = list(func.keys())[0]
    name = tf_export_lib.get_canonical_name_for_symbol(
        func, add_prefix_to_v1_names=True
    )
    if name is not None:
      params = tf_inspect.signature(func).parameters.keys()
      lines.append(f"  * `tf.{name}({', '.join(params)})`")
  lines.sort()
  target.__doc__ = target.__doc__.replace("<<API_LIST>>", "\n".join(lines))


################################################################################
# Dispatch Support
################################################################################
@tf_export("__internal__.dispatch.add_dispatch_support", v1=[])
def add_dispatch_support(target=None, iterable_parameters=None):
  # [FUNCTION DOC]
  # Public decorator (usable bare or with arguments) that makes a Python API support both
  # dispatch systems. Inputs: target, the function to wrap (omit to get a decorator back);
  # iterable_parameters, optional list or tuple of parameter names that may receive one-shot
  # iterables (anything else raises a type error). The wrapper first tries type-based dispatch
  # (converting iterable parameters to lists), then calls the original function, and if that
  # raises a type or value error tries the fallback dispatchers, returning their result or
  # re-raising the original error. Tracebacks are filtered. Returns the wrapper, or a decorator
  # when no target was given. Side effects: gives the wrapper a fallback list and, when supported,
  # a native dispatcher registered in the signature registry.
  """Decorator that adds a dispatch handling wrapper to a TensorFlow Python API.

  This wrapper adds the decorated function as an API that can be overridden
  using the `@dispatch_for_api` decorator.  In the following example, we first
  define a new API (`double`) that supports dispatch, then define a custom type
  (`MaskedTensor`) and finally use `dispatch_for_api` to override the default
  implementation of `double` when called with `MaskedTensor` values:

  >>> @add_dispatch_support
  ... def double(x):
  ...   return x * 2
  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_api(double, {'x': MaskedTensor})
  ... def masked_double(x):
  ...   return MaskedTensor(x.values * 2, y.mask)

  The optional `iterable_parameter` argument can be used to mark parameters that
  can take arbitrary iterable values (such as generator expressions).  These
  need to be handled specially during dispatch, since just iterating over an
  iterable uses up its values.  In the following example, we define a new API
  whose second argument can be an iterable value; and then override the default
  implementation of that API when the iterable contains MaskedTensors:

  >>> @add_dispatch_support(iterable_parameters=['ys'])
  ... def add_tensor_to_list_of_tensors(x, ys):
  ...   return [x + y for y in ys]
  >>> @dispatch_for_api(add_tensor_to_list_of_tensors,
  ...               {'ys': typing.List[MaskedTensor]})
  ... def masked_add_tensor_to_list_of_tensors(x, ys):
  ...   return [MaskedTensor(x+y.values, y.mask) for y in ys]

  (Note: the only TensorFlow API that currently supports iterables is `add_n`.)

  Args:
    target: The TensorFlow API that should support dispatch.
    iterable_parameters: Optional list of parameter names that may be called
      with iterables (such as the `inputs` parameter for `tf.add_n`).

  Returns:
    A decorator.
  """

  if not (iterable_parameters is None or
          (isinstance(iterable_parameters, (list, tuple)) and
           all(isinstance(p, str) for p in iterable_parameters))):
    raise TypeError("iterable_parameters should be a list or tuple of string.")

  def decorator(dispatch_target):
    # [FUNCTION DOC]
    # Decorator built by add_dispatch_support. Input: the function to make dispatchable. Looks up
    # the position of each iterable parameter among the function's argument names (a missing name
    # raises a value error), builds the dispatching wrapper, attaches the fallback list and the
    # native type-based dispatcher, and returns the wrapper.

    # Get the name & index for each iterable parameter.
    if iterable_parameters is None:
      iterable_params = None
    else:
      arg_names = tf_inspect.getargspec(dispatch_target).args
      iterable_params = [
          (name, arg_names.index(name)) for name in iterable_parameters
      ]

    @traceback_utils.filter_traceback
    def op_dispatch_handler(*args, **kwargs):
      # [FUNCTION DOC]
      # The wrapper that replaces the API. Takes the API's call arguments. If a native dispatcher
      # exists, converts iterable parameters to lists and asks it to dispatch, returning its
      # result unless it signals not implemented. Otherwise calls the original function; on a
      # type or value error asks the fallback dispatchers and returns their result, or re-raises
      # the original error if none handles the call. Traceback frames are filtered on errors.
      """Call `dispatch_target`, performing dispatch when appropriate."""

      # Type-based dispatch system (dispatch v2):
      if api_dispatcher is not None:
        if iterable_params is not None:
          args, kwargs = replace_iterable_params(args, kwargs, iterable_params)
        result = api_dispatcher.Dispatch(args, kwargs)
        if result is not NotImplemented:
          return result

      # Fallback dispatch system (dispatch v1):
      try:
        return dispatch_target(*args, **kwargs)
      except (TypeError, ValueError):
        # Note: convert_to_eager_tensor currently raises a ValueError, not a
        # TypeError, when given unexpected types.  So we need to catch both.
        result = dispatch(op_dispatch_handler, args, kwargs)
        if result is not OpDispatcher.NOT_SUPPORTED:
          return result
        else:
          raise

    add_fallback_dispatch_list(op_dispatch_handler)
    op_dispatch_handler = tf_decorator.make_decorator(dispatch_target,
                                                      op_dispatch_handler)
    add_type_based_api_dispatcher(op_dispatch_handler)
    api_dispatcher = getattr(op_dispatch_handler, TYPE_BASED_DISPATCH_ATTR,
                             None)
    return op_dispatch_handler

  if target is None:
    return decorator
  else:
    return decorator(target)


def replace_iterable_params(args, kwargs, iterable_params):
  # [FUNCTION DOC]
  # Materializes one-shot iterables so dispatch can inspect them without consuming them.
  # Inputs: positional arguments, keyword arguments, and a list of (name, index) pairs. For each
  # pair, converts the positional value at that index to a list if present, else the keyword
  # value of that name if present. Returns a pair of the new positional tuple and the keyword
  # mapping. Side effect: the keyword mapping is modified in place.
  """Returns (args, kwargs) with any iterable parameters converted to lists.

  Args:
    args: Positional rguments to a function
    kwargs: Keyword arguments to a function.
    iterable_params: A list of (name, index) tuples for iterable parameters.

  Returns:
    A tuple (args, kwargs), where any positional or keyword parameters in
    `iterable_params` have their value converted to a `list`.
  """
  args = list(args)
  for name, index in iterable_params:
    if index < len(args):
      args[index] = list(args[index])
    elif name in kwargs:
      kwargs[name] = list(kwargs[name])
  return tuple(args), kwargs
