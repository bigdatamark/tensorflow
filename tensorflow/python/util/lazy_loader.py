# [MODULE DOC]
# HELPER: lazy-import utilities used by TensorFlow's API packages. Defines a module-type
# placeholder that imports the real module only when first used, and a specialised variant that
# chooses between the Keras 3, Keras 2 and legacy tf_keras packages at first use depending on
# an environment variable. Defines a reserved attribute prefix constant. No import-time imports
# of the lazily loaded modules happen here.
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

"""A LazyLoader class."""

import importlib
import os
import types
from tensorflow.python.platform import tf_logging as logging

_TENSORFLOW_LAZY_LOADER_PREFIX = "_tfll"


class LazyLoader(types.ModuleType):
  # [CLASS DOC]
  # A module-type placeholder that imports the real module only on first attribute access.
  # State: local name, the parent's globals mapping, and an optional one-time warning, all kept
  # in attributes with a reserved prefix. On load, the real module replaces the placeholder in
  # the parent's globals and its dictionary is copied into the placeholder. Attribute reads,
  # writes, deletes and directory listing are forwarded to the real module (triggering the
  # load); the text representation and pickling do not load.
  """Lazily import a module, mainly to avoid pulling in large dependencies.

  `contrib`, and `ffmpeg` are examples of modules that are large and not always
  needed, and this allows them to only be loaded when they are used.
  """

  # The lint error here is incorrect.
  def __init__(self, local_name, parent_module_globals, name, warning=None):
    # [FUNCTION DOC]
    # Creates a placeholder module object for a lazily imported module.
    # Inputs: local_name, the name under which the real module will be inserted into the parent
    # namespace; parent_module_globals, the mutable mapping of the importing module's globals;
    # name, the full dotted module name to import; warning, optional message logged once on
    # first load (default none).
    # Stores these as internal attributes (prefixed so setattr bypasses loading), and sets
    # the module-attribution and wrapped attributes directly through the base class so that
    # documentation-test discovery does not trigger a load. Then runs the base module
    # constructor with the name. Nothing is imported here.
    self._tfll_local_name = local_name
    self._tfll_parent_module_globals = parent_module_globals
    self._tfll_warning = warning

    # These members allows doctest correctly process this module member without
    # triggering self._load(). self._load() mutates parent_module_globals and
    # triggers a dict mutated during iteration error from doctest.py.
    # - for from_module()
    super().__setattr__("__module__", name.rsplit(".", 1)[0])
    # - for is_routine()
    super().__setattr__("__wrapped__", None)

    super().__init__(name)

  def _load(self):
    # [FUNCTION DOC]
    # HELPER: performs the actual import for the lazy loader; used by attribute hooks.
    # Imports the module named by this object, stores it in the parent globals under the local
    # name (replacing the placeholder), logs the configured warning once (then clears it), and
    # copies the module's dictionary into this object so later lookups are fast.
    # Returns the real module. Raises whatever the import raises. Side effects: mutates the
    # parent globals, may log a warning, mutates this object's dictionary.
    """Load the module and insert it into the parent's globals."""
    # Import the target module and insert it into the parent's namespace
    module = importlib.import_module(self.__name__)
    self._tfll_parent_module_globals[self._tfll_local_name] = module

    # Emit a warning if one was specified
    if self._tfll_warning:
      logging.warning(self._tfll_warning)
      # Make sure to only warn once.
      self._tfll_warning = None

    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)

    return module

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Called only when normal lookup fails. Loads the real module, then returns its attribute
    # of the given name (raising an attribute error if the module lacks it).
    module = self._load()
    return getattr(module, name)

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Sets an attribute. Names with the reserved internal prefix are stored on the placeholder
    # itself without loading. For any other name, loads the real module, sets the attribute on
    # it and on this object, and appends the name to the module's export list if it has one
    # and the name is not already listed (ignored when there is no such list).
    # Inputs: name (string) and value (any). Returns nothing.
    if name.startswith(_TENSORFLOW_LAZY_LOADER_PREFIX):
      super().__setattr__(name, value)
    else:
      module = self._load()
      setattr(module, name, value)
      self.__dict__[name] = value
      try:
        # check if the module has __all__
        if name not in self.__all__ and name != "__all__":
          self.__all__.append(name)
      except AttributeError:
        pass

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # Deletes an attribute. Names with the reserved internal prefix are deleted from the
    # placeholder without loading. For other names, loads the real module, deletes the attribute
    # from it, removes it from this object's dictionary (a missing entry raises a key error),
    # and removes the name from the module's export list if present.
    # Input: name (string). Returns nothing; raises an attribute error if the module lacks it.
    if name.startswith(_TENSORFLOW_LAZY_LOADER_PREFIX):
      super().__delattr__(name)
    else:
      module = self._load()
      delattr(module, name)
      self.__dict__.pop(name)
      try:
        # check if the module has __all__
        if name in self.__all__:
          self.__all__.remove(name)
      except AttributeError:
        pass

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form naming the module and its local name. Deliberately avoids loading the
    # module because it may be called in sensitive places.
    # Carefully to not trigger _load, since repr may be called in very
    # sensitive places.
    return f"<LazyLoader {self.__name__} as {self._tfll_local_name}>"

  def __dir__(self):
    # [FUNCTION DOC]
    # Loads the real module and returns the list of its attribute names.
    module = self._load()
    return dir(module)

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickle support: instead of serializing the placeholder, instructs the unpickler to import
    # the module by name, so unpickling yields the real module without loading it now.
    # Returns a pair of the import function and a one-item tuple of the module name.
    return importlib.import_module, (self.__name__,)


class KerasLazyLoader(LazyLoader):
  # [CLASS DOC]
  # Lazy loader specialised for Keras that picks the Keras implementation at first use: the
  # legacy tf_keras package when an environment variable asks for it, otherwise the installed
  # keras package (version 3 or 2 layouts). State: parent globals, mode ("v1" or "v2" API
  # flavour, optional), optional submodule path, local name, an initialized flag and the
  # resolved Keras version label. The base class is fully initialized lazily, not in the
  # constructor.
  """LazyLoader that handles routing to different Keras version."""

  def __init__(  # pylint: disable=super-init-not-called
      self, parent_module_globals, mode=None, submodule=None, name="keras"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: intentionally does not call the parent constructor (lint suppressed), so the
    # module-name and wrapped attributes are not set until initialization happens lazily.
    # Records the parent globals mapping, the mode (optional; "v1" selects the first-generation
    # API, anything else the second), an optional submodule path appended to the package path,
    # and the local name (default "keras"); marks the loader as not yet initialized.
    # Imports nothing and returns nothing.
    self._tfll_parent_module_globals = parent_module_globals
    self._tfll_mode = mode
    self._tfll_submodule = submodule
    self._tfll_name = name
    self._tfll_initialized = False

  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: resolves which Keras package to use, used by attribute lookup and listing.
    # Marks the loader initialized first. If the legacy-Keras environment variable is true
    # (true, True or 1), tries to import tf_keras and picks its v1 or v2 API package path by
    # mode; if the import fails only a warning is logged and nothing is selected. Otherwise
    # imports keras and picks the Keras 3 path (version starting with 3) or the Keras 2 v1/v2
    # path; an import failure raises an import error. Appends the submodule if given, records
    # the version label, and runs the base constructor with the local name, parent globals and
    # package path. Raises an import error if no Keras version could be selected (including the
    # tf_keras-missing case). Side effects: imports packages, may log a warning, sets state.
    """Resolve the Keras version to use and initialize the loader."""
    self._tfll_initialized = True
    package_name = None
    keras_version = None
    if os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"):
      try:
        import tf_keras  # pylint: disable=g-import-not-at-top,unused-import

        keras_version = "tf_keras"
        if self._tfll_mode == "v1":
          package_name = "tf_keras.api._v1.keras"
        else:
          package_name = "tf_keras.api._v2.keras"
      except ImportError:
        logging.warning(
            "Your environment has TF_USE_LEGACY_KERAS set to True, but you "
            "do not have the tf_keras package installed. You must install it "
            "in order to use the legacy tf.keras. Install it via: "
            "`pip install tf_keras`"
        )
    else:
      try:
        import keras  # pylint: disable=g-import-not-at-top

        if keras.__version__.startswith("3."):
          # This is the Keras 3.x case.
          keras_version = "keras_3"
          package_name = "keras._tf_keras.keras"
        else:
          # This is the Keras 2.x case.
          keras_version = "keras_2"
          if self._tfll_mode == "v1":
            package_name = "keras.api._v1.keras"
          else:
            package_name = "keras.api._v2.keras"
      except ImportError:
        raise ImportError(  # pylint: disable=raise-missing-from
            "Keras cannot be imported. Check that it is installed."
        )

    self._tfll_keras_version = keras_version
    if keras_version is not None:
      if self._tfll_submodule is not None:
        package_name += "." + self._tfll_submodule
      super().__init__(
          self._tfll_name, self._tfll_parent_module_globals, package_name
      )
    else:
      raise ImportError(  # pylint: disable=raise-missing-from
          "Keras cannot be imported. Check that it is installed."
      )

  def __getattr__(self, item):
    # [FUNCTION DOC]
    # Attribute lookup hook. The mode, initialized-flag and name attributes are served directly
    # from the object (bypassing module lookup). Otherwise initializes on first use, and under
    # Keras 3 raises an attribute error with an explanatory message for unsupported routes:
    # compat v1 access in v1 mode without submodule, compat v2 access in v2 mode without
    # submodule, and any attribute when the submodule path is a legacy internal one. Finally
    # loads the real module and returns the requested attribute.
    # Input: item, the attribute name (string).
    if item in ("_tfll_mode", "_tfll_initialized", "_tfll_name"):
      return super(types.ModuleType, self).__getattribute__(item)
    if not self._tfll_initialized:
      self._initialize()
    if self._tfll_keras_version == "keras_3":
      if (
          self._tfll_mode == "v1"
          and not self._tfll_submodule
          and item.startswith("compat.v1.")
      ):
        raise AttributeError(
            "`tf.compat.v1.keras` is not available with Keras 3. Keras 3 has "
            "no support for TF 1 APIs. You can install the `tf_keras` package "
            "as an alternative, and set the environment variable "
            "`TF_USE_LEGACY_KERAS=True` to configure TensorFlow to route "
            "`tf.compat.v1.keras` to `tf_keras`."
        )
      elif (
          self._tfll_mode == "v2"
          and not self._tfll_submodule
          and item.startswith("compat.v2.")
      ):
        raise AttributeError(
            "`tf.compat.v2.keras` is not available with Keras 3. Just use "
            "`import keras` instead."
        )
      elif self._tfll_submodule and self._tfll_submodule.startswith(
          "__internal__.legacy."
      ):
        raise AttributeError(
            f"`{item}` is not available with Keras 3."
        )
    module = self._load()
    return getattr(module, item)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form. Before initialization it is a fixed placeholder text; afterwards it
    # includes the Keras version label, module name, local name and mode. Never loads anything.
    if self._tfll_initialized:
      return (
          f"<KerasLazyLoader ({self._tfll_keras_version}) "
          f"{self.__name__} as {self._tfll_local_name} mode={self._tfll_mode}>"
      )
    return "<KerasLazyLoader>"

  def __dir__(self):
    # [FUNCTION DOC]
    # Initializes the loader if needed (selecting the Keras package), then returns the base
    # class listing, which loads the real module and lists its attribute names.
    if not self._tfll_initialized:
      self._initialize()
    return super().__dir__()
