# [MODULE DOC]
# Implements the tf_export and keras_export decorators that mark functions, classes and constants
# as part of the public API and record the public names (separately for the V2 and V1 APIs).
# Exposes: constants naming the API families and the attribute names used to store export
# information, a global mapping from public API name to symbol, lookup helpers (symbol by name,
# canonical name, V1/V2 names and constants of a symbol or module), the api_export decorator
# class, a keyword-only argument wrapper, and the partial-application decorators tf_export and
# keras_export. API generation tools read the attributes written here. Import-time side effects:
# builds the attribute tables and the two decorator objects; the global name-to-symbol mapping is
# filled as other modules are imported and decorated.
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
"""Utilities for exporting TensorFlow symbols to the API.

Exporting a function or a class:

To export a function or a class use tf_export decorator. For e.g.:
```python
@tf_export('foo', 'bar.foo')
def foo(...):
  ...
```

If a function is assigned to a variable, you can export it by calling
tf_export explicitly. For e.g.:
```python
foo = get_foo(...)
tf_export('foo', 'bar.foo')(foo)
```


Exporting a constant
```python
foo = 1
tf_export('consts.foo').export_constant(__name__, 'foo')
```
"""
from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

KERAS_API_NAME = 'keras'
TENSORFLOW_API_NAME = 'tensorflow'

# List of subpackage names used by TensorFlow components. Have to check that
# TensorFlow core repo does not export any symbols under these names.
SUBPACKAGE_NAMESPACES = []


class _Attributes(NamedTuple):
  # [CLASS DOC]
  # HELPER: small immutable record holding the two attribute names (for API names and for API
  # constants) used to store export information on a symbol or module for one API family.
  names: str
  constants: str


# Attribute values must be unique to each API.
API_ATTRS = {
    TENSORFLOW_API_NAME: _Attributes('_tf_api_names', '_tf_api_constants'),
    KERAS_API_NAME: _Attributes('_keras_api_names', '_keras_api_constants'),
}

API_ATTRS_V1 = {
    TENSORFLOW_API_NAME: _Attributes(
        '_tf_api_names_v1', '_tf_api_constants_v1'
    ),
    KERAS_API_NAME: _Attributes(
        '_keras_api_names_v1', '_keras_api_constants_v1'
    ),
}


class InvalidSymbolNameError(Exception):
  # [CLASS DOC]
  # Error raised by the export decorator when a symbol would be exported under a name that is not
  # allowed for its API (for example under a subpackage reserved for another component).
  """Raised when trying to export symbol as an invalid or unallowed name."""


_NAME_TO_SYMBOL_MAPPING: dict[str, Any] = dict()


def get_symbol_from_name(name: str) -> Optional[Any]:
  # [FUNCTION DOC]
  # Looks up an exported symbol by its public API name in the global registry filled by the export
  # decorators. Input: name, a dotted string (V1-only names carry the compat.v1 prefix). Returns
  # the symbol, or nothing/null if the name is unknown. No side effects.
  return _NAME_TO_SYMBOL_MAPPING.get(name)


def get_canonical_name_for_symbol(
    symbol: Any,
    api_name: str = TENSORFLOW_API_NAME,
    add_prefix_to_v1_names: bool = False,
) -> Optional[str]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: when the symbol has V2 names attribute but no V1 one (or the reverse path), the
  # lookup of the V1 attribute uses a direct attribute read and raises an attribute error; also,
  # if no V1 canonical name exists and prefixing is requested, the result is the text
  # compat.v1.None instead of null.
  # Finds the preferred public name of an exported symbol. Inputs: symbol, a function or class;
  # api_name, which API family to use (default tensorflow); add_prefix_to_v1_names, whether a
  # name found only in the V1 API gets a compat.v1 prefix (default false). Unwraps any decorators
  # first. Returns null if the symbol has no dictionary or no export attribute; otherwise the
  # canonical V2 name, falling back to the canonical V1 name. Deprecated names are consulted so
  # non-deprecated names win. Raises a key error for an unknown api_name. No side effects.
  """Get canonical name for the API symbol.

  Example:
  ```python
  from tensorflow.python.util import tf_export
  cls = tf_export.get_symbol_from_name('keras.optimizers.Adam')

  # Gives `<class 'keras.optimizer_v2.adam.Adam'>`
  print(cls)

  # Gives `keras.optimizers.Adam`
  print(tf_export.get_canonical_name_for_symbol(cls, api_name='keras'))
  ```

  Args:
    symbol: API function or class.
    api_name: API name. Currently, only `tensorflow`.
    add_prefix_to_v1_names: Specifies whether a name available only in V1 should
      be prefixed with compat.v1.

  Returns:
    Canonical name for the API symbol (for e.g. initializers.zeros) if
    canonical name could be determined. Otherwise, returns None.
  """
  if not hasattr(symbol, '__dict__'):
    return None
  api_names_attr = API_ATTRS[api_name].names
  _, undecorated_symbol = tf_decorator.unwrap(symbol)
  if api_names_attr not in undecorated_symbol.__dict__:
    return None
  api_names = getattr(undecorated_symbol, api_names_attr)
  deprecated_api_names = undecorated_symbol.__dict__.get(
      '_tf_deprecated_api_names', []
  )

  canonical_name = get_canonical_name(api_names, deprecated_api_names)
  if canonical_name:
    return canonical_name

  # If there is no V2 canonical name, get V1 canonical name.
  api_names_attr = API_ATTRS_V1[api_name].names
  api_names = getattr(undecorated_symbol, api_names_attr)
  v1_canonical_name = get_canonical_name(api_names, deprecated_api_names)
  if add_prefix_to_v1_names:
    return 'compat.v1.%s' % v1_canonical_name
  return v1_canonical_name


def get_canonical_name(
    api_names: Sequence[str], deprecated_api_names: Sequence[str]
) -> Optional[str]:
  # [FUNCTION DOC]
  # Chooses the preferred endpoint name from a sequence of API names. Inputs: api_names, all
  # names; deprecated_api_names, names considered deprecated. Returns the first name that is not
  # deprecated, else the first name, else null when the sequence is empty. No side effects.
  """Get preferred endpoint name.

  Args:
    api_names: API names iterable.
    deprecated_api_names: Deprecated API names iterable.

  Returns:
    Returns one of the following in decreasing preference:
    - first non-deprecated endpoint
    - first endpoint
    - None
  """
  non_deprecated_name = next(
      (name for name in api_names if name not in deprecated_api_names), None
  )
  if non_deprecated_name:
    return non_deprecated_name
  if api_names:
    return api_names[0]
  return None


def get_v1_names(symbol: Any) -> Sequence[str]:
  # [FUNCTION DOC]
  # Collects the V1 API names recorded on a symbol by the export decorators, for both the
  # tensorflow and keras families, tensorflow names first. Input: symbol, any object. Returns a
  # new list (empty if the symbol has no dictionary or no names). No side effects.
  """Get a list of TF 1.* names for this symbol.

  Args:
    symbol: symbol to get API names for.

  Returns:
    List of all API names for this symbol.
  """
  names_v1 = []
  tensorflow_api_attr_v1 = API_ATTRS_V1[TENSORFLOW_API_NAME].names
  keras_api_attr_v1 = API_ATTRS_V1[KERAS_API_NAME].names

  if not hasattr(symbol, '__dict__'):
    return names_v1
  if tensorflow_api_attr_v1 in symbol.__dict__:
    names_v1.extend(getattr(symbol, tensorflow_api_attr_v1))
  if keras_api_attr_v1 in symbol.__dict__:
    names_v1.extend(getattr(symbol, keras_api_attr_v1))
  return names_v1


def get_v2_names(symbol: Any) -> Sequence[str]:
  # [FUNCTION DOC]
  # Collects the V2 API names recorded on a symbol by the export decorators, for both the
  # tensorflow and keras families, tensorflow names first. Input: symbol, any object. Returns a
  # new list (empty if the symbol has no dictionary or no names). No side effects.
  """Get a list of TF 2.0 names for this symbol.

  Args:
    symbol: symbol to get API names for.

  Returns:
    List of all API names for this symbol.
  """
  names_v2 = []
  tensorflow_api_attr = API_ATTRS[TENSORFLOW_API_NAME].names
  keras_api_attr = API_ATTRS[KERAS_API_NAME].names

  if not hasattr(symbol, '__dict__'):
    return names_v2
  if tensorflow_api_attr in symbol.__dict__:
    names_v2.extend(getattr(symbol, tensorflow_api_attr))
  if keras_api_attr in symbol.__dict__:
    names_v2.extend(getattr(symbol, keras_api_attr))
  return names_v2


def get_v1_constants(module: Any) -> Sequence[str]:
  # [FUNCTION DOC]
  # Returns a new list of the V1 API constant records stored on a module by export_constant (only
  # the tensorflow family is consulted). Input: module object. Returns an empty list if none.
  # KNOWN ISSUE: the annotation and docstring say a sequence of strings, but entries are pairs of
  # (names, constant name).
  """Get a list of TF 1.* constants in this module.

  Args:
    module: TensorFlow module.

  Returns:
    List of all API constants under the given module.
  """
  constants_v1 = []
  tensorflow_constants_attr_v1 = API_ATTRS_V1[TENSORFLOW_API_NAME].constants

  if hasattr(module, tensorflow_constants_attr_v1):
    constants_v1.extend(getattr(module, tensorflow_constants_attr_v1))
  return constants_v1


def get_v2_constants(module: Any) -> Sequence[str]:
  # [FUNCTION DOC]
  # Returns a new list of the V2 API constant records stored on a module by export_constant (only
  # the tensorflow family is consulted). Input: module object. Returns an empty list if none.
  # KNOWN ISSUE: the annotation and docstring say a sequence of strings, but entries are pairs of
  # (names, constant name).
  """Get a list of TF 2.0 constants in this module.

  Args:
    module: TensorFlow module.

  Returns:
    List of all API constants under the given module.
  """
  constants_v2 = []
  tensorflow_constants_attr = API_ATTRS[TENSORFLOW_API_NAME].constants

  if hasattr(module, tensorflow_constants_attr):
    constants_v2.extend(getattr(module, tensorflow_constants_attr))
  return constants_v2


T = TypeVar('T')


class api_export(object):  # pylint: disable=invalid-name
  # [CLASS DOC]
  # Decorator class behind tf_export and keras_export. An instance holds the V2 public names, the
  # V1 public names (defaulting to the V2 names) and the API family name. Calling it on a symbol
  # records the names on the undecorated symbol and registers the symbol in the global name
  # registry; export_constant records constants on a module. Names are validated at creation.
  """Provides ways to export symbols to the TensorFlow API."""

  _names: Sequence[str]
  _names_v1: Sequence[str]
  _api_name: str

  def __init__(
      self,
      *args: str,
      api_name: str = TENSORFLOW_API_NAME,
      v1: Optional[Sequence[str]] = None,
      allow_multiple_exports: bool = True,  # pylint: disable=unused-argument
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: positional strings naming the public endpoints in dotted form (the first
    # is canonical); api_name, the API family (default tensorflow); v1, optional names for the V1
    # API (defaults to the positional names); allow_multiple_exports, deprecated and ignored.
    # Stores the names and validates them. Raises InvalidSymbolNameError for disallowed names.
    """Export under the names *args (first one is considered canonical).

    Args:
      *args: API names in dot delimited format.
      api_name: API you want to generate Currently, only `tensorflow`.
      v1: Names for the TensorFlow V1 API. If not set, we will use V2 API names
        both for TensorFlow V1 and V2 APIs.
      allow_multiple_exports: Deprecated.
    """
    self._names = args
    self._names_v1 = v1 if v1 is not None else args
    self._api_name = api_name

    self._validate_symbol_names()

  def _validate_symbol_names(self) -> None:
    # [FUNCTION DOC]
    # HELPER: used by the constructor to check the requested names. For the tensorflow family it
    # rejects any name starting with a registered component subpackage namespace; for other
    # families it requires every name to start with the family name. Raises InvalidSymbolNameError
    # on violation; returns nothing.
    """Validate you are exporting symbols under an allowed package.

    We need to ensure things exported by tf_export, etc.
    export symbols under disjoint top-level package names.

    For TensorFlow, we check that it does not export anything under subpackage
    names used by components (keras, etc.).

    For each component, we check that it exports everything under its own
    subpackage.

    Raises:
      InvalidSymbolNameError: If you try to export symbol under disallowed name.
    """
    all_symbol_names = set(self._names) | set(self._names_v1)
    if self._api_name == TENSORFLOW_API_NAME:
      for subpackage in SUBPACKAGE_NAMESPACES:
        if any(n.startswith(subpackage) for n in all_symbol_names):
          raise InvalidSymbolNameError(
              '@tf_export is not allowed to export symbols under %s.*'
              % (subpackage)
          )
    else:
      if not all(n.startswith(self._api_name) for n in all_symbol_names):
        raise InvalidSymbolNameError(
            'Can only export symbols under package name of component.'
        )

  def __call__(self, func: T) -> T:
    # [FUNCTION DOC]
    # Applies the export to a symbol. Input: func, a function or class (possibly decorated).
    # Side effects: sets the V2 and V1 name attributes on the undecorated symbol, and registers the
    # symbol in the global registry under each V2 name and under each V1 name with a compat.v1
    # prefix. Returns the same symbol unchanged.
    """Calls this decorator.

    Args:
      func: decorated symbol (function or class).

    Returns:
      The input function with _tf_api_names attribute set.
    """
    api_names_attr = API_ATTRS[self._api_name].names
    api_names_attr_v1 = API_ATTRS_V1[self._api_name].names

    _, undecorated_func = tf_decorator.unwrap(func)
    self.set_attr(undecorated_func, api_names_attr, self._names)
    self.set_attr(undecorated_func, api_names_attr_v1, self._names_v1)

    for name in self._names:
      _NAME_TO_SYMBOL_MAPPING[name] = func
    for name_v1 in self._names_v1:
      _NAME_TO_SYMBOL_MAPPING['compat.v1.%s' % name_v1] = func

    return func

  def set_attr(
      self, func: Any, api_names_attr: str, names: Sequence[str]
  ) -> None:
    # [FUNCTION DOC]
    # Stores a names sequence as an attribute on the given object (overridable hook used by
    # __call__). Inputs: func, target object; api_names_attr, attribute name; names, the value.
    # Returns nothing; mutates func.
    setattr(func, api_names_attr, names)

  def export_constant(self, module_name: str, name: str) -> None:
    # [FUNCTION DOC]
    # Records a constant for API generation. Inputs: module_name, name of an already imported
    # module; name, the constant's name in that module. Side effects: creates (if missing) the V2
    # and V1 constants lists on the module and appends a pair of the exported names and the
    # constant name to each. Raises a key error if the module is not imported. Returns nothing.
    """Store export information for constants/string literals.

    Export information is stored in the module where constants/string literals
    are defined.

    e.g.
    ```python
    foo = 1
    bar = 2
    tf_export("consts.foo").export_constant(__name__, 'foo')
    tf_export("consts.bar").export_constant(__name__, 'bar')
    ```

    Args:
      module_name: (string) Name of the module to store constant at.
      name: (string) Current constant name.
    """
    module = sys.modules[module_name]
    api_constants_attr = API_ATTRS[self._api_name].constants
    api_constants_attr_v1 = API_ATTRS_V1[self._api_name].constants

    if not hasattr(module, api_constants_attr):
      setattr(module, api_constants_attr, [])
    # pylint: disable=protected-access
    getattr(module, api_constants_attr).append((self._names, name))

    if not hasattr(module, api_constants_attr_v1):
      setattr(module, api_constants_attr_v1, [])
    getattr(module, api_constants_attr_v1).append((self._names_v1, name))


def kwarg_only(f: Any) -> Any:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it throws away non-keyword arguments, but the wrapper actually
  # raises a type error when any positional argument is passed.
  # Decorator forcing keyword-only calls. Input: f, the callable. Captures its full argument spec
  # and returns a wrapper, built with make_decorator using that spec, that rejects positional
  # arguments and otherwise calls f with the keyword arguments.
  """A wrapper that throws away all non-kwarg arguments."""
  f_argspec = tf_inspect.getfullargspec(f)

  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper built by kwarg_only. Raises a type error naming the function and its
    # possible keyword names if any positional argument is given; otherwise calls the wrapped
    # function with the keyword arguments and returns its result.
    if args:
      raise TypeError(
          '{f} only takes keyword args (possible keys: {kwargs}). '
          'Please pass these args as kwargs instead.'.format(
              f=f.__name__, kwargs=f_argspec.args
          )
      )
    return f(**kwargs)

  return tf_decorator.make_decorator(f, wrapper, decorator_argspec=f_argspec)


class ExportType(Protocol):
  # [CLASS DOC]
  # HELPER: typing protocol describing the call signature of tf_export and keras_export for static
  # type checkers. Never instantiated at runtime.

  def __call__(
      self,
      *v2: str,
      v1: Optional[Sequence[str]] = None,
      allow_multiple_exports: bool = True,  # Deprecated, no-op
  ) -> api_export:
    # [FUNCTION DOC]
    # HELPER: typing-only signature. Takes positional API name strings, optional V1 names and a
    # deprecated no-op flag, and is declared to return an api_export object. Body is empty.
    ...


tf_export: ExportType = functools.partial(
    api_export, api_name=TENSORFLOW_API_NAME
)
keras_export: ExportType = functools.partial(
    api_export, api_name=KERAS_API_NAME
)
