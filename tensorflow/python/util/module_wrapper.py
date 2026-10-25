# [MODULE DOC]
# HELPER: wrapper machinery for TensorFlow API modules. Provides a module-type wrapper that
# lazily imports public symbols, issues limited deprecation warnings using the generated rename
# table, records compat.v1 usage in a monitoring gauge, and keeps caches consistent; plus helper
# functions for rename lookup and detecting existing deprecation decorators.
# Import-time side effects: creates a boolean monitoring gauge named for compat.v1 usage and
# obtains the fast module type class.
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
"""Provides wrapper for TensorFlow modules."""

import importlib

from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2

FastModuleType = fast_module_type.get_fast_module_type_class()
_PER_MODULE_WARNING_LIMIT = 1
compat_v1_usage_gauge = monitoring.BoolGauge('/tensorflow/api/compat/v1',
                                             'compat.v1 usage')


def get_rename_v2(name):
  # [FUNCTION DOC]
  # Looks up the recommended replacement for a renamed public symbol in the generated rename
  # table. Input: name, the full dotted symbol name (string). Returns the replacement name
  # string, or the null value if the symbol is not in the table. No side effects.
  if name not in all_renames_v2.symbol_renames:
    return None
  return all_renames_v2.symbol_renames[name]


def _call_location():
  # [FUNCTION DOC]
  # HELPER: used only by the wrapper's deprecation-warning method to report user code location.
  # KNOWN ISSUE: the docstring says it returns the caller location, but it actually walks up
  # three stack frames from itself (past the attribute hook and the warning method), which is
  # a heuristic that breaks if the call chain changes; it enforces this by asserting that its
  # direct caller has the expected name (the assertion is lost under optimized mode).
  # Returns a string of the form file name, colon, line number of the located frame (stops
  # early if the stack is shorter). Raises an assertion error if called from anywhere else.
  """Extracts the caller filename and line number as a string.

  Returns:
    A string describing the caller source location.
  """
  frame = tf_inspect.currentframe()
  assert frame.f_back.f_code.co_name == '_tfmw_add_deprecation_warning', (
      'This function should be called directly from '
      '_tfmw_add_deprecation_warning, as the caller is identified '
      'heuristically by chopping off the top stack frames.')

  # We want to get stack frame 3 frames up from current frame,
  # i.e. above __getattr__, _tfmw_add_deprecation_warning,
  # and _call_location calls.
  for _ in range(3):
    parent = frame.f_back
    if parent is None:
      break
    frame = parent
  return '{}:{}'.format(frame.f_code.co_filename, frame.f_lineno)


def contains_deprecation_decorator(decorators):
  # [FUNCTION DOC]
  # HELPER: used by has_deprecation_decorator.
  # Input: decorators, a list of decorator records (each with a decorator name). Returns true
  # if any record is named as the deprecation decorator, otherwise false.
  return any(d.decorator_name == 'deprecated' for d in decorators)


def has_deprecation_decorator(symbol):
  # [FUNCTION DOC]
  # Decides whether a symbol is already marked deprecated so the wrapper need not warn again.
  # Input: symbol, any object. Unwraps decorators; true if any is the deprecation decorator.
  # Otherwise, for plain functions and non-classes returns false; for a class, checks the
  # decorators on its constructor. Returns a boolean; no side effects.
  """Checks if given object has a deprecation decorator.

  We check if deprecation decorator is in decorators as well as
  whether symbol is a class whose __init__ method has a deprecation
  decorator.
  Args:
    symbol: Python object.

  Returns:
    True if symbol has deprecation decorator.
  """
  decorators, symbol = tf_decorator.unwrap(symbol)
  if contains_deprecation_decorator(decorators):
    return True
  if tf_inspect.isfunction(symbol):
    return False
  if not tf_inspect.isclass(symbol):
    return False
  if not hasattr(symbol, '__init__'):
    return False
  init_decorators, _ = tf_decorator.unwrap(symbol.__init__)
  return contains_deprecation_decorator(init_decorators)


class TFModuleWrapper(FastModuleType):
  # [CLASS DOC]
  # Module-type wrapper around a TensorFlow API module that (1) lazily imports public symbols
  # listed in a mapping of name to (module, attribute) location, (2) emits at most a limited
  # number of deprecation warnings for renamed symbols, (3) records compat.v1 usage once in a
  # monitoring gauge, and (4) keeps the wrapped module, the wrapper's dictionary, the export
  # list and the fast attribute cache consistent on set and delete. Internal state lives in
  # attributes prefixed with a reserved marker (wrapped module, module name, public API map,
  # warning flags/counters, has-lite flag, compat.v1 flag, checked-name set). The class attribute
  # compat_v1_usage_recorded ensures the usage gauge is set only once per process. It installs
  # its own attribute-access and fallback callbacks on the fast module type machinery.
  """Wrapper for TF modules to support deprecation messages and lazyloading."""
  # Ensures that compat.v1 API usage is recorded at most once
  compat_v1_usage_recorded = False

  def __init__(
      self,
      wrapped,
      module_name,
      public_apis=None,
      deprecation=True,
      has_lite=False):
    # [FUNCTION DOC]
    # Builds the wrapper. Inputs: wrapped, the module object to wrap; module_name, the short
    # name used when composing full symbol names for rename lookups (may be empty); public_apis,
    # optional mapping from symbol name to a pair of module path and attribute (an empty module
    # path means import the attribute itself as a module); deprecation, whether to print
    # deprecation warnings (default true); has_lite, whether the lite subpackage should be
    # loaded from the public API map (default false).
    # Initializes the base type with the wrapped module's name, registers the fallback and
    # attribute-access callbacks, copies the wrapped module's dictionary, stores the internal
    # state, and sets the export list: from the public API names if given, else the wrapped
    # module's list, else all its public (non-underscore) names (which is also stored on the
    # wrapped module). Initializes the checked-names set and warning counter.
    # Side effects: mutates the wrapped module (export list).
    super(TFModuleWrapper, self).__init__(wrapped.__name__)
    FastModuleType.set_getattr_callback(self, TFModuleWrapper._getattr)
    FastModuleType.set_getattribute_callback(self,
                                             TFModuleWrapper._getattribute)
    self.__dict__.update(wrapped.__dict__)
    # Prefix all local attributes with _tfmw_ so that we can
    # handle them differently in attribute access methods.
    self._tfmw_wrapped_module = wrapped
    self._tfmw_module_name = module_name
    self._tfmw_public_apis = public_apis
    self._tfmw_print_deprecation_warnings = deprecation
    self._tfmw_has_lite = has_lite
    self._tfmw_is_compat_v1 = (wrapped.__name__.endswith('.compat.v1'))
    # Set __all__ so that import * work for lazy loaded modules
    if self._tfmw_public_apis:
      self._tfmw_wrapped_module.__all__ = list(self._tfmw_public_apis.keys())
      self.__all__ = list(self._tfmw_public_apis.keys())
    else:
      if hasattr(self._tfmw_wrapped_module, '__all__'):
        self.__all__ = self._tfmw_wrapped_module.__all__
      else:
        self._tfmw_wrapped_module.__all__ = [
            attr for attr in dir(self._tfmw_wrapped_module)
            if not attr.startswith('_')
        ]
        self.__all__ = self._tfmw_wrapped_module.__all__

    # names we already checked for deprecation
    self._tfmw_deprecated_checked = set()
    self._tfmw_warning_count = 0

  def _tfmw_add_deprecation_warning(self, name, attr):
    # [FUNCTION DOC]
    # HELPER: used by the attribute-access hooks to warn about renamed symbols.
    # KNOWN ISSUE: the call location is computed twice (once for the check, once for the log
    # message); the second call happens from the same frame so it is harmless but redundant.
    # Inputs: name, the attribute name; attr, the attribute value. If the per-module warning
    # limit has not been reached and the name was not already checked, marks it checked, builds
    # the full dotted name, and if it has a rename and the value is not already decorated as
    # deprecated and the call location is not a synthetic one (starting with an angle bracket),
    # logs a warning naming the location, old name and replacement and increments the counter.
    # Returns true if a warning was logged, otherwise false.
    """Print deprecation warning for attr with given name if necessary."""
    if (self._tfmw_warning_count < _PER_MODULE_WARNING_LIMIT and
        name not in self._tfmw_deprecated_checked):

      self._tfmw_deprecated_checked.add(name)

      if self._tfmw_module_name:
        full_name = 'tf.%s.%s' % (self._tfmw_module_name, name)
      else:
        full_name = 'tf.%s' % name
      rename = get_rename_v2(full_name)
      if rename and not has_deprecation_decorator(attr):
        call_location = _call_location()
        # skip locations in Python source
        if not call_location.startswith('<'):
          logging.warning(
              'From %s: The name %s is deprecated. Please use %s instead.\n',
              _call_location(), full_name, rename)
          self._tfmw_warning_count += 1
          return True
    return False

  def _tfmw_import_module(self, name):
    # [FUNCTION DOC]
    # HELPER: lazy-import step used by the attribute hooks.
    # Input: name, a key that must exist in the public API map (otherwise a key error is
    # raised, or a type error if the map is null). If this wrapper is for compat.v1 and the name
    # is not "app" and usage was not yet recorded, sets the class flag and sets the monitoring
    # gauge to true. Then imports the module (or the attribute from the module) given by the
    # map, stores it on the wrapped module, in this wrapper's dictionary and in the fast cache.
    # Returns the imported object. Side effects: imports, gauge update, three caches updated.
    """Lazily loading the modules."""
    # We ignore 'app' because it is accessed in __init__.py of tf.compat.v1.
    # That way, if a user only imports tensorflow.compat.v1, it is not
    # considered v1 API usage.
    if (self._tfmw_is_compat_v1 and name != 'app' and
        not TFModuleWrapper.compat_v1_usage_recorded):
      TFModuleWrapper.compat_v1_usage_recorded = True
      compat_v1_usage_gauge.get_cell().set(True)

    symbol_loc_info = self._tfmw_public_apis[name]
    if symbol_loc_info[0]:
      module = importlib.import_module(symbol_loc_info[0])
      attr = getattr(module, symbol_loc_info[1])
    else:
      attr = importlib.import_module(symbol_loc_info[1])
    setattr(self._tfmw_wrapped_module, name, attr)
    self.__dict__[name] = attr
    # Cache the pair
    self._fastdict_insert(name, attr)
    return attr

  def _getattribute(self, name):
    # [FUNCTION DOC]
    # Callback installed in place of the normal attribute-access hook (see its docstring).
    # Input: name, the attribute being read. For the lite subpackage, when has-lite is set,
    # imports it through the public API map, sets it on the wrapped module, caches it and
    # returns it. Otherwise does a plain lookup (raising an attribute error if missing, which
    # lets the fallback callback run). Dunder names and the wrapper's own reserved-prefix names
    # are cached and returned. For other names, if deprecation printing is on and a warning was
    # issued, the value is not cached (so the warning limit logic can be revisited); otherwise
    # it is cached in the fast dictionary. Returns the attribute value.
    # pylint: disable=g-doc-return-or-yield,g-doc-args
    """Imports and caches pre-defined API.

    Warns if necessary.

    This method is a replacement for __getattribute__(). It will be added into
    the extended python module as a callback to reduce API overhead.
    """
    # Avoid infinite recursions
    func__fastdict_insert = object.__getattribute__(self, '_fastdict_insert')

    # Make sure we do not import from tensorflow/lite/__init__.py
    if name == 'lite':
      if self._tfmw_has_lite:
        attr = self._tfmw_import_module(name)
        setattr(self._tfmw_wrapped_module, 'lite', attr)
        func__fastdict_insert(name, attr)
        return attr
  # Placeholder for Google-internal contrib error

    attr = object.__getattribute__(self, name)

    # Return and cache dunders and our own members.
    # This is necessary to guarantee successful construction.
    # In addition, all the accessed attributes used during the construction must
    # begin with "__" or "_tfmw" or "_fastdict_".
    if name.startswith('__') or name.startswith('_tfmw_') or name.startswith(
        '_fastdict_'):
      func__fastdict_insert(name, attr)
      return attr

    # Print deprecations, only cache functions after deprecation warnings have
    # stopped.
    if not (self._tfmw_print_deprecation_warnings and
            self._tfmw_add_deprecation_warning(name, attr)):
      func__fastdict_insert(name, attr)

    return attr

  def _getattr(self, name):
    # [FUNCTION DOC]
    # Callback installed as the fallback hook, called when default lookup fails.
    # Input: name, the attribute name. First tries the wrapped module; if that raises an
    # attribute error, re-raises it when there is no public API map or the name is not in it,
    # otherwise lazily imports the symbol. Then, if deprecation printing is enabled, may log a
    # deprecation warning (result ignored). Returns the attribute value. Side effects: lazy
    # import and caching, possible warning log.
    # pylint: disable=g-doc-return-or-yield,g-doc-args
    """Imports and caches pre-defined API.

    Warns if necessary.

    This method is a replacement for __getattr__(). It will be added into the
    extended python module as a callback to reduce API overhead. Instead of
    relying on implicit AttributeError handling, this added callback function
    will
    be called explicitly from the extended C API if the default attribute lookup
    fails.
    """
    try:
      attr = getattr(self._tfmw_wrapped_module, name)
    except AttributeError:
    # Placeholder for Google-internal contrib error

      if not self._tfmw_public_apis:
        raise
      if name not in self._tfmw_public_apis:
        raise
      attr = self._tfmw_import_module(name)

    if self._tfmw_print_deprecation_warnings:
      self._tfmw_add_deprecation_warning(name, attr)
    return attr

  def __setattr__(self, arg, val):
    # [FUNCTION DOC]
    # Sets an attribute. Inputs: arg (name string) and val (any value). For names that do not
    # start with the reserved internal prefix, also sets the value on the wrapped module and in
    # this wrapper's dictionary, appends the name to the export list if absent (except the
    # export list itself), and refreshes the fast cache entry if the name is already cached.
    # In all cases then performs the base-class assignment. Returns nothing.
    if not arg.startswith('_tfmw_'):
      setattr(self._tfmw_wrapped_module, arg, val)
      self.__dict__[arg] = val
      if arg not in self.__all__ and arg != '__all__':
        self.__all__.append(arg)
      # Update the cache
      if self._fastdict_key_in(arg):
        self._fastdict_insert(arg, val)
    super(TFModuleWrapper, self).__setattr__(arg, val)

  def __dir__(self):
    # [FUNCTION DOC]
    # Returns the list of attribute names. With a public API map: the union of the map's names
    # and the wrapped module's public (non-underscore) names, in unspecified order. Without a
    # map: the wrapped module's full name listing.
    if self._tfmw_public_apis:
      return list(
          set(self._tfmw_public_apis.keys()).union(
              set([
                  attr for attr in dir(self._tfmw_wrapped_module)
                  if not attr.startswith('_')
              ])))
    else:
      return dir(self._tfmw_wrapped_module)

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for non-internal names, deletion fails with a key error if the name is not
    # in the wrapper's own dictionary or the fast cache (for example a lazily listed symbol that
    # was never loaded), after already deleting it from the wrapped module; a stale commented
    # out duplicate deletion line remains at the end of the method.
    # Deletes an attribute. Reserved-prefix names are deleted via the base class. Other names
    # are deleted from the wrapped module, the wrapper's dictionary, the export list (if
    # listed) and the fast cache. Input: name (string). Returns nothing.
    if name.startswith('_tfmw_'):
      super(TFModuleWrapper, self).__delattr__(name)
    else:
      delattr(self._tfmw_wrapped_module, name)
      self.__dict__.pop(name)
      if name in self.__all__:
        self.__all__.remove(name)
      self._fastdict_pop(name)
      # delattr(self._tfmw_wrapped_module, name)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the text form of the wrapped module, so the wrapper looks like the original.
    return self._tfmw_wrapped_module.__repr__()

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickle support: instructs the unpickler to import the module by this wrapper's name
    # instead of serializing it. Returns a pair of the import function and a one-item tuple
    # holding the module name.
    return importlib.import_module, (self.__name__,)
