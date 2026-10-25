# [MODULE DOC]
# Utilities for controlling which symbols a module exposes based on references written in
# module docstrings: building a public-names list from reference markers, removing (hiding)
# symbols that are not referenced, and restoring hidden symbols for internal tests. Keeps a
# module-level registry of hidden attributes. Exposes only the three public functions via the
# export list. No import-time side effects other than defining the registry and pattern.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Generate __all__ from a module docstring."""
import re as _re
import sys as _sys

from tensorflow.python.util import tf_inspect as _tf_inspect


_reference_pattern = _re.compile(r'^@@(\w+)$', flags=_re.MULTILINE)


def make_all(module_name, doc_string_modules=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring has a grammar slip ("must each a docstring"), and it says the
  # modules must have a docstring while modules without one are actually silently skipped.
  # Builds a list of public symbol names from reference markers in docstrings. Inputs:
  # module_name, the name of an already-imported module (looked up in the loaded modules);
  # doc_string_modules, optional list of modules whose docstrings are scanned (default: only
  # the named module). Collects every name written as a double-at marker on its own line in
  # those docstrings, keeping only names that currently exist as members of the named module.
  # Returns a list of the unique names in arbitrary order. Raises a key error if the named
  # module is not loaded. No side effects.
  """Generates `__all__` from the docstring of one or more modules.

  Usage: `make_all(__name__)` or
  `make_all(__name__, [sys.modules(__name__), other_module])`. The doc string
  modules must each a docstring, and `__all__` will contain all symbols with
  `@@` references, where that symbol currently exists in the module named
  `module_name`.

  Args:
    module_name: The name of the module (usually `__name__`).
    doc_string_modules: a list of modules from which to take docstring.
    If None, then a list containing only the module named `module_name` is used.

  Returns:
    A list suitable for use as `__all__`.
  """
  if doc_string_modules is None:
    doc_string_modules = [_sys.modules[module_name]]
  cur_members = set(
      name for name, _ in _tf_inspect.getmembers(_sys.modules[module_name]))

  results = set()
  for doc_module in doc_string_modules:
    if doc_module.__doc__ is None:
      continue
    results.update([m.group(1)
                    for m in _reference_pattern.finditer(doc_module.__doc__)
                    if m.group(1) in cur_members])
  return list(results)

# Hidden attributes are attributes that have been hidden by
# `remove_undocumented`. They can be re-instated by `reveal_undocumented`.
# This maps symbol names to a tuple, containing:
#   (module object, attribute value)
_HIDDEN_ATTRIBUTES = {}


def reveal_undocumented(symbol_name, target_module=None):
  # [FUNCTION DOC]
  # Restores a symbol previously hidden by remove_undocumented, intended for internal tests.
  # Inputs: symbol_name, the full dotted path of the symbol; target_module, optional module to
  # restore it into (default: the module it was removed from). Looks up the stored module and
  # value in the hidden-attribute registry and sets the attribute (last path component) on
  # the target module. Raises a lookup error if the symbol was never hidden. Returns nothing.
  # The registry entry is not removed.
  """Reveals a symbol that was previously removed by `remove_undocumented`.

  This should be used by tensorflow internal tests only. It explicitly
  defeats the encapsulation afforded by `remove_undocumented`.

  It throws an exception when the symbol was not hidden in the first place.

  Args:
    symbol_name: a string representing the full absolute path of the symbol.
    target_module: if specified, the module in which to restore the symbol.
  """
  if symbol_name not in _HIDDEN_ATTRIBUTES:
    raise LookupError('Symbol %s is not a hidden symbol' % symbol_name)
  symbol_basename = symbol_name.split('.')[-1]
  (original_module, attr_value) = _HIDDEN_ATTRIBUTES[symbol_name]
  if not target_module: target_module = original_module
  setattr(target_module, symbol_basename, attr_value)


def remove_undocumented(module_name, allowed_exception_list=None,
                        doc_string_modules=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring mentions a global allow-list function that does not exist in
  # this module, and the last paragraph is attached to the argument list; no allow-list is
  # consulted. Also, the allowed exception list is combined by adding to the list returned
  # from make_all, which is a fresh list so callers' lists are not mutated.
  # Hides every public symbol of a module that is not referenced in the scanned docstrings.
  # Inputs: module_name, name of a loaded module; allowed_exception_list, optional names to
  # keep anyway; doc_string_modules, optional modules whose docstrings define the referenced
  # names (default: the module itself). Names starting with an underscore are always kept.
  # For each removed symbol it records the module and original value in the hidden-attribute
  # registry under the dotted full name, then deletes the attribute from the module.
  # Returns nothing. Side effects: mutates the module and the registry.
  """Removes symbols in a module that are not referenced by a docstring.

  Args:
    module_name: the name of the module (usually `__name__`).
    allowed_exception_list: a list of names that should not be removed.
    doc_string_modules: a list of modules from which to take the docstrings.
    If None, then a list containing only the module named `module_name` is used.

    Furthermore, if a symbol previously added with `add_to_global_allowlist`,
    then it will always be allowed. This is useful for internal tests.

  Returns:
    None
  """
  current_symbols = set(dir(_sys.modules[module_name]))
  should_have = make_all(module_name, doc_string_modules)
  should_have += allowed_exception_list or []
  extra_symbols = current_symbols - set(should_have)
  target_module = _sys.modules[module_name]
  for extra_symbol in extra_symbols:
    # Skip over __file__, etc. Also preserves internal symbols.
    if extra_symbol.startswith('_'): continue
    fully_qualified_name = module_name + '.' + extra_symbol
    _HIDDEN_ATTRIBUTES[fully_qualified_name] = (target_module,
                                                getattr(target_module,
                                                        extra_symbol))
    delattr(target_module, extra_symbol)


__all__ = [
    'make_all',
    'remove_undocumented',
    'reveal_undocumented',
]
