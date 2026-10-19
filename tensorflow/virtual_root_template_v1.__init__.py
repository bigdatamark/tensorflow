# [PACKAGE DOC]
# Template for the virtual-root TensorFlow 1.x package initializer, used when the real code lives
# in a separate core package (tensorflow_core) and the tensorflow name only forwards to it.
# Import-time side effects: registers lazy forwarding modules in the module table for each
# top-level subpackage (api, python, tools, core, compiler, lite, keras, compat, summary,
# examples), mapping each to the same name in the core package; re-exports every public name
# from the core package; sets major API version 1; wraps this module in the module wrapper that
# prints deprecation messages (unless already wrapped); and removes core, python, compiler,
# tools and examples names from the namespace if present.
# Defines a private lazy-loader module class and a private forwarding function.
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
# LINT.IfChange
"""TensorFlow root package"""

import sys as _sys
import importlib as _importlib
import types as _types

from tensorflow.python.util import module_wrapper


# Since TensorFlow Python code now resides in tensorflow_core but TensorFlow
# ecosystem code  imports tensorflow, we need to do forwarding between the two.
# To do so, we use a lazy loader to load and forward the top level modules. We
# cannot use the LazyLoader defined by tensorflow at
# tensorflow/python/util/lazy_loader.py as to use that we would already need to
# import tensorflow. Hence, we define it inline.
class _LazyLoader(_types.ModuleType):
  # [CLASS DOC]
  # HELPER: private placeholder module used to forward old module names to the core package.
  # A module object that defers importing the real module until an attribute is read or the
  # directory listing is requested. Holds the local name, a reference to the parent's global
  # namespace mapping, and the full target name (as the module name). On first use it imports
  # the target, rebinds the local name in the parent namespace and copies the target's contents.
  # Supports pickling by reducing to an import of the target name.
  """Lazily import a module so that we can forward it."""

  # The lint error here is incorrect.
  def __init__(self, local_name, parent_module_globals, name):
    # [FUNCTION DOC]
    # Constructor. Inputs: local_name, the attribute name under which the module is bound in the
    # parent namespace; parent_module_globals, the parent's global name mapping (mutated on
    # load); name, the full dotted name of the module to import. Stores the first two and
    # initialises the base module object with the target name. No import happens yet.
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    super(_LazyLoader, self).__init__(name)

  def _load(self):
    # [FUNCTION DOC]
    # HELPER: performs the deferred import. Imports the module named by this object, stores it in
    # the parent namespace under the local name, copies the module's attributes into this
    # object's own dictionary, and returns the real module. Raises an import error if the target
    # cannot be imported.
    """Import the target module and insert it into the parent's namespace."""
    module = _importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module
    self.__dict__.update(module.__dict__)
    return module

  def __getattr__(self, item):
    # [FUNCTION DOC]
    # Called for attributes not found normally. Input: item, attribute name. Loads the real
    # module (triggering the import) and returns that attribute from it; raises an attribute
    # error if the real module lacks it.
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    # [FUNCTION DOC]
    # Supports listing of attribute names. Loads the real module and returns its attribute name
    # listing.
    module = self._load()
    return dir(module)

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickling support. Returns a recipe that re-imports the target module by its full name
    # rather than serialising this placeholder.
    return __import__, (self.__name__,)


# Forwarding a module is as simple as lazy loading the module from the new path
# and then registering it to sys.modules using the old path
def _forward_module(old_name):
  # [FUNCTION DOC]
  # HELPER: registers a lazy forwarding module for one top-level subpackage.
  # Input: old_name, dotted name such as the tensorflow python subpackage. Builds the core-package
  # name by appending the core suffix to the first dotted part, creates a lazy loader for it,
  # and records it in the interpreter's module table under the old name unless one is already
  # present. Returns the module registered under the old name (the new loader or the existing
  # one). Side effect: mutates the module table.
  # KNOWN ISSUE: the loader's rebinding target is this module's global namespace under the last
  # name part, which is only meaningful for single-level names like those in the list.
  parts = old_name.split(".")
  parts[0] = parts[0] + "_core"
  local_name = parts[-1]
  existing_name = ".".join(parts)
  _module = _LazyLoader(local_name, globals(), existing_name)
  return _sys.modules.setdefault(old_name, _module)


# This list should contain all modules _immediately_ under tensorflow
_top_level_modules = [
    "tensorflow._api",
    "tensorflow.python",
    "tensorflow.tools",
    "tensorflow.core",
    "tensorflow.compiler",
    "tensorflow.lite",
    "tensorflow.keras",
    "tensorflow.compat",
    "tensorflow.summary",  # tensorboard
    "tensorflow.examples",
]

# Lazy load all of the _top_level_modules, we don't need their names anymore
for _m in _top_level_modules:
  _forward_module(_m)

# We still need all the names that are toplevel on tensorflow_core
from tensorflow_core import *

_major_api_version = 1

# In V1 API we need to print deprecation messages
if not isinstance(_sys.modules[__name__], module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "")

# These should not be visible in the main tf module.
try:
  del core
except NameError:
  pass

try:
  del python
except NameError:
  pass

try:
  del compiler
except NameError:
  pass

try:
  del tools
except NameError:
  pass

try:
  del examples
except NameError:
  pass

# LINT.ThenChange(//tensorflow/virtual_root_template_v2.__init__.py.oss)
