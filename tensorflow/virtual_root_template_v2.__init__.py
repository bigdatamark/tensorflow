# [PACKAGE DOC]
# Template for the root "tensorflow" package of the TF2 API, used as a virtual root that forwards
# to the real implementation package named "tensorflow_core".
# On import it registers lazy-loading placeholder modules in the module registry for each listed
# top-level subpackage (api, python, tools, core, compiler, lite, keras, compat, summary,
# examples) under the old "tensorflow." names, pointing at the "tensorflow_core." equivalents.
# It then re-exports every public name from tensorflow_core, sets the major API version to 2,
# and removes the internal names core, python, compiler, tools and examples from the namespace
# (ignoring them if they are absent) so they are not visible in the main module.
# Import-time side effects: modifies the global module registry and this module's globals.
# The lint markers indicate it must be kept in sync with the version 1 template.
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


# Since TensorFlow Python code now resides in tensorflow_core but TensorFlow
# ecosystem code imports tensorflow, we need to do forwarding between the two.
# To do so, we use a lazy loader to load and forward the top level modules. We
# cannot use the LazyLoader defined by tensorflow at
# tensorflow/python/util/lazy_loader.py as to use that we would already need to
# import tensorflow. Hence, we define it inline.
class _LazyLoader(_types.ModuleType):
  # [CLASS DOC]
  # HELPER: Internal glue that lets the virtual root forward old module names to tensorflow_core.
  # A placeholder module object that defers importing the real module until an attribute is
  # first accessed. State: the local name under which it is stored in the parent namespace and
  # a reference to the parent's global mapping. Created by the forwarding function and
  # registered in the module registry. Once loaded it replaces itself in the parent namespace
  # and copies the real module's contents into its own dictionary.
  """Lazily import a module so that we can forward it."""

  # The lint error here is incorrect.
  def __init__(self, local_name, parent_module_globals, name):
    # [FUNCTION DOC]
    # Constructor. local_name: attribute name used in the parent namespace; parent_module_globals:
    # the mapping of global names of the parent module; name: full dotted name of the real
    # module to import later. Stores the first two, then initializes the module object with name.
    # Does not import anything yet. Returns nothing.
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    super(_LazyLoader, self).__init__(name)

  def _load(self):
    # [FUNCTION DOC]
    # HELPER: Performs the deferred import for the lazy loader.
    # Imports the real module by this object's name, stores it in the parent namespace under the
    # local name (replacing the placeholder), copies the real module's attributes into this
    # object's dictionary, and returns the real module. Side effects: import, parent globals
    # mutation. Raises whatever the import raises (e.g. import error) if the module is missing.
    """Import the target module and insert it into the parent's namespace."""
    module = _importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module
    self.__dict__.update(module.__dict__)
    return module

  def __getattr__(self, item):
    # [FUNCTION DOC]
    # Called only when normal attribute lookup fails. Triggers the deferred load and returns the
    # requested attribute (item, the attribute name) of the real module; raises an attribute
    # error if the real module lacks it.
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    # [FUNCTION DOC]
    # Triggers the deferred load and returns the list of attribute names of the real module, so
    # that directory listings and completion work on the placeholder.
    module = self._load()
    return dir(module)

  def __reduce__(self):
    # [FUNCTION DOC]
    # Serialization (pickling) support. Returns a recipe that re-imports the module by name
    # instead of serializing the placeholder's contents.
    return __import__, (self.__name__,)


# Forwarding a module is as simple as lazy loading the module from the new path
# and then registering it to sys.modules using the old path
def _forward_module(old_name):
  # [FUNCTION DOC]
  # HELPER: Registers a forwarding placeholder for one top-level module of this package.
  # old_name: dotted module name under the old root (for example the package name followed by a
  # submodule). The first name component gets the "_core" suffix to form the real module name,
  # a lazy loader is built for it (local name is the last component, parent mapping is this
  # module's globals), and it is registered in the module registry under old_name unless an
  # entry already exists. Returns the registered module (the existing one if present).
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

_major_api_version = 2

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

# LINT.ThenChange(//tensorflow/virtual_root_template_v1.__init__.py.oss)
