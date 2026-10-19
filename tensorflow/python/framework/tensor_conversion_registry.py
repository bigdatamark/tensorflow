# [MODULE DOC]
# Registry of functions that convert Python objects to tensors. Holds module-level state: a
# mapping from priority to a list of (base type, conversion function) pairs, a per-type cache of
# the ordered applicable functions, and a lock guarding both. Exposes registration (public and
# internal variants), lookup by type, and the main convert routine that tries applicable
# functions in priority order and validates results. Public numeric types (int, float, numpy
# scalars and arrays) may only be registered through the internal variant. Registers the public
# registration function in the exported API at import time.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Registry for tensor conversion functions."""
# pylint: disable=g-bad-name
import collections
import threading

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export


_tensor_conversion_func_registry = collections.defaultdict(list)
_tensor_conversion_func_cache = {}
_tensor_conversion_func_lock = threading.Lock()

# Instances of these types should only be converted by internally-registered
# conversion functions.
_CONSTANT_OP_CONVERTIBLES = (
    int,
    float,
    np.generic,
    np.ndarray,
)


# TODO(josh11b): Add ctx argument to conversion_func() signature.
def register_tensor_conversion_function_internal(base_type,
                                                 conversion_func,
                                                 priority=100):
  # [FUNCTION DOC]
  # HELPER: internal registration path used by the public registration function and by
  # framework code that needs to convert numeric types.
  # KNOWN ISSUE: the docstring says this variant allows types in an unconvertible-types tuple,
  # but no such tuple exists in this module; the restriction actually lives in a constants tuple
  # of numeric types checked only by the public function.
  # Validates that base_type is a type or tuple of types and conversion_func is callable,
  # raising a type error otherwise. Then, under the module lock, appends the pair to the list for
  # the given priority (smaller runs earlier; default 100) and clears the whole lookup cache.
  # Returns nothing. Side effect: mutates global registry state.
  """Internal version of register_tensor_conversion_function.

  See docstring of `register_tensor_conversion_function` for details.

  The internal version of the function allows registering conversions
  for types in the _UNCONVERTIBLE_TYPES tuple.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values run
      earlier than conversion functions with larger priority values. Defaults to
      100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
  """
  base_types = base_type if isinstance(base_type, tuple) else (base_type,)
  if any(not isinstance(x, type) for x in base_types):
    raise TypeError("Argument `base_type` must be a type or a tuple of types. "
                    f"Obtained: {base_type}")
  del base_types  # Only needed for validation.
  if not callable(conversion_func):
    raise TypeError("Argument `conversion_func` must be callable. Received "
                    f"{conversion_func}.")

  with _tensor_conversion_func_lock:
    _tensor_conversion_func_registry[priority].append(
        (base_type, conversion_func))
    _tensor_conversion_func_cache.clear()


@tf_export("register_tensor_conversion_function")
def register_tensor_conversion_function(base_type,
                                        conversion_func,
                                        priority=100):
  # [FUNCTION DOC]
  # Public API for registering a conversion function for a type. Inputs: base_type (a type or
  # tuple of types), conversion_func (callable taking the value, dtype, name and as-ref flag),
  # priority (optional integer, smaller runs first, default 100). Raises a type error if
  # base_type is not a type or tuple of types, or if any listed type is a subclass of the
  # protected numeric types (int, float, numpy scalars and arrays). Otherwise delegates to the
  # internal registration function, which also validates callability. Returns nothing; mutates
  # global registry state and invalidates the cache.
  """Registers a function for converting objects of `base_type` to `Tensor`.

  The conversion function must have the following signature:

  ```python
      def conversion_func(value, dtype=None, name=None, as_ref=False):
        # ...
  ```

  It must return a `Tensor` with the given `dtype` if specified. If the
  conversion function creates a new `Tensor`, it should use the given
  `name` if specified. All exceptions will be propagated to the caller.

  The conversion function may return `NotImplemented` for some
  inputs. In this case, the conversion process will continue to try
  subsequent conversion functions.

  If `as_ref` is true, the function must return a `Tensor` reference,
  such as a `Variable`.

  NOTE: The conversion functions will execute in order of priority,
  followed by order of registration. To ensure that a conversion function
  `F` runs before another conversion function `G`, ensure that `F` is
  registered with a smaller priority than `G`.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values run
      earlier than conversion functions with larger priority values. Defaults to
      100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
  """
  base_types = base_type if isinstance(base_type, tuple) else (base_type,)
  if any(not isinstance(x, type) for x in base_types):
    raise TypeError("Argument `base_type` must be a type or a tuple of types. "
                    f"Obtained: {base_type}")
  if any(issubclass(x, _CONSTANT_OP_CONVERTIBLES) for x in base_types):
    raise TypeError("Cannot register conversions for Python numeric types and "
                    "NumPy scalars and arrays.")
  del base_types  # Only needed for validation.
  register_tensor_conversion_function_internal(
      base_type, conversion_func, priority)


def get(query):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring mentions a class named cls, but the parameter is named query; it
  # also says the list is in increasing order of priority, which means smaller priority numbers
  # (earlier execution) first.
  # Looks up the conversion functions applicable to a type. Input: query, a type. Returns a list
  # of (base type, conversion function) pairs whose base type is a superclass of query, ordered
  # by ascending priority value then registration order. Uses a cache with double-checked
  # locking: on a miss, takes the module lock, rebuilds the list from the registry and stores it
  # in the cache. Side effect: populates the cache. Raises nothing itself.
  """Get conversion function for objects of `cls`.

  Args:
    query: The type to query for.

  Returns:
    A list of conversion functions in increasing order of priority.
  """
  conversion_funcs = _tensor_conversion_func_cache.get(query)
  if conversion_funcs is None:
    with _tensor_conversion_func_lock:
      # Has another thread populated the cache in the meantime?
      conversion_funcs = _tensor_conversion_func_cache.get(query)
      if conversion_funcs is None:
        conversion_funcs = []
        for _, funcs_at_priority in sorted(
            _tensor_conversion_func_registry.items()):
          conversion_funcs.extend(
              (base_type, conversion_func)
              for base_type, conversion_func in funcs_at_priority
              if issubclass(query, base_type))
        _tensor_conversion_func_cache[query] = conversion_funcs
  return conversion_funcs


def _add_error_prefix(msg, *, name=None):
  # [FUNCTION DOC]
  # HELPER: formats error messages for the convert routine.
  # Returns msg unchanged when name is absent, otherwise the name followed by a colon, a space
  # and the message. Pure; no side effects.
  return msg if name is None else f"{name}: {msg}"


def convert(value,
            dtype=None,
            name=None,
            as_ref=False,
            preferred_dtype=None,
            accepted_result_types=(core.Symbol,)):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring's Raises section refers to an element in values, but this function
  # converts a single value. Also the incompatible-dtype check after a preferred-dtype attempt
  # compares only base dtypes, while the later check for the requested dtype uses compatibility.
  # Core conversion routine. Inputs: value (object to convert), dtype (optional target type,
  # coerced to a dtype object), name (optional name), as_ref (flag passed through to conversion
  # functions, default false), preferred_dtype (optional soft type tried first only when dtype is
  # absent), accepted_result_types (types a result may have; default the symbolic tensor base).
  # If the value defines a to-tensor hook, it is called with dtype and name and its result is
  # returned immediately without validation. Otherwise each registered function for the value's
  # type is tried in order: with the preferred dtype first (type/value errors there are swallowed
  # and it falls back to the plain attempt); a runtime error is raised if the result has a
  # different base dtype than preferred. Functions returning the not-implemented marker are
  # skipped. Tensor-like results with a to-tensor hook are converted through it. A result not of an
  # accepted type, or with dtype incompatible with the requested dtype, raises a runtime error.
  # Returns the first valid result. Raises a type error if no function handles the value. Errors
  # carry the name as a prefix. Conversion functions may have their own side effects.
  """Converts `value` to a `Tensor` using registered conversion functions.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    as_ref: Optional boolean specifying if the returned value should be a
      reference-type `Tensor` (e.g. Variable). Pass-through to the registered
      conversion function. Defaults to `False`.
    preferred_dtype: Optional element type for the returned tensor.
      Used when dtype is None. In some cases, a caller may not have a dtype
      in mind when converting to a tensor, so `preferred_dtype` can be used
      as a soft preference. If the conversion to `preferred_dtype` is not
      possible, this argument has no effect.
    accepted_result_types: Optional collection of types as an allow-list
      for the returned value. If a conversion function returns an object
      which is not an instance of some type in this collection, that value
      will not be returned.

  Returns:
    A `Tensor` converted from `value`.

  Raises:
    ValueError: If `value` is a `Tensor` and conversion is requested
      to a `Tensor` with an incompatible `dtype`.
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """

  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  if preferred_dtype is not None:
    preferred_dtype = dtypes.as_dtype(preferred_dtype)

  overload = getattr(value, "__tf_tensor__", None)
  if overload is not None:
    return overload(dtype, name)  #  pylint: disable=not-callable

  for base_type, conversion_func in get(type(value)):
    # If dtype is None but preferred_dtype is not None, we try to
    # cast to preferred_dtype first.
    ret = None
    if dtype is None and preferred_dtype is not None:
      try:
        ret = conversion_func(
            value, dtype=preferred_dtype, name=name, as_ref=as_ref)
      except (TypeError, ValueError):
        # Could not coerce the conversion to use the preferred dtype.
        pass
      else:
        if (ret is not NotImplemented and
            ret.dtype.base_dtype != preferred_dtype.base_dtype):
          raise RuntimeError(
              _add_error_prefix(
                  f"Conversion function {conversion_func!r} for type "
                  f"{base_type} returned incompatible dtype: requested = "
                  f"{preferred_dtype.base_dtype.name}, "
                  f"actual = {ret.dtype.base_dtype.name}",
                  name=name))

    if ret is None:
      ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)

    if ret is NotImplemented:
      continue
    # Convert ret to Tensor if it is a core.Tensor type.
    if isinstance(ret, core.Tensor):
      to_tensor = getattr(ret, "__tf_tensor__", None)
      ret = (
          to_tensor()  #  pylint: disable=not-callable
          if to_tensor is not None
          else ret
      )
    if not isinstance(ret, accepted_result_types):
      raise RuntimeError(
          _add_error_prefix(
              f"Conversion function {conversion_func!r} for type "
              f"{base_type} returned non-Tensor: {ret!r}",
              name=name))
    if dtype and not dtype.is_compatible_with(ret.dtype):
      raise RuntimeError(
          _add_error_prefix(
              f"Conversion function {conversion_func} for type {base_type} "
              f"returned incompatible dtype: requested = {dtype.name}, "
              f"actual = {ret.dtype.name}",
              name=name))
    return ret
  raise TypeError(
      _add_error_prefix(
          f"Cannot convert {value!r} with type {type(value)} to Tensor: "
          f"no conversion function registered.",
          name=name))
