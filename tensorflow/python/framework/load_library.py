# [MODULE DOC]
# Public functions for dynamically loading native TensorFlow extensions: custom op and kernel
# libraries (with generated Python wrappers), file system implementations, kernel-only libraries,
# PluggableDevice plugins, and file system plugins. Also holds a private helper that recognises
# shared-object file names per platform. Loading registers ops, kernels or devices in the process;
# several functions are exported under public TensorFlow names. No import-time side effects.
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

"""Function for loading TensorFlow plugins."""
import errno
import hashlib
import importlib
import os
import platform
import sys

from tensorflow.python.client import pywrap_tf_session as py_tf
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_op_gen
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export('load_op_library')
def load_op_library(library_filename):
  # [FUNCTION DOC]
  # Loads a native plugin library (path string, relative or absolute) containing custom ops and
  # kernels, registering them in the process. Gets the op list from the loaded library, generates
  # Python wrapper source for it, and always releases the native library handle afterwards. The
  # module name is the SHA-1 hex digest of the generated wrapper source; if a module with that name
  # is already in the loaded-modules table it is returned. Otherwise a new module is created, the
  # wrapper source is executed inside it, a marker attribute is set so AutoGraph recognises it as a
  # TensorFlow plugin, and it is stored in the loaded-modules table and returned.
  # Raises a runtime error when loading fails or wrappers cannot be produced.
  # Side effects: registers ops and kernels natively, mutates the global module table, executes code.
  """Loads a TensorFlow plugin, containing custom ops and kernels.

  Pass "library_filename" to a platform-specific mechanism for dynamically
  loading a library. The rules for determining the exact location of the
  library are platform-specific and are not documented here. When the
  library is loaded, ops and kernels registered in the library via the
  `REGISTER_*` macros are made available in the TensorFlow process. Note
  that ops with the same name as an existing op are rejected and not
  registered with the process.

  Args:
    library_filename: Path to the plugin.
      Relative or absolute filesystem path to a dynamic library file.

  Returns:
    A python module containing the Python wrappers for Ops defined in
    the plugin.

  Raises:
    RuntimeError: when unable to load the library or get the python wrappers.
  """
  lib_handle = py_tf.TF_LoadLibrary(library_filename)
  try:
    wrappers = _pywrap_python_op_gen.GetPythonWrappers(
        py_tf.TF_GetOpList(lib_handle))
  finally:
    # Delete the library handle to release any memory held in C
    # that are no longer needed.
    py_tf.TF_DeleteLibraryHandle(lib_handle)

  # Get a unique name for the module.
  module_name = hashlib.sha1(wrappers).hexdigest()
  if module_name in sys.modules:
    return sys.modules[module_name]
  module_spec = importlib.machinery.ModuleSpec(module_name, None)
  module = importlib.util.module_from_spec(module_spec)
  # pylint: disable=exec-used
  exec(wrappers, module.__dict__)
  # Allow this to be recognized by AutoGraph.
  setattr(module, '_IS_TENSORFLOW_PLUGIN', True)
  sys.modules[module_name] = module
  return module


@deprecation.deprecated(date=None,
                        instructions='Use `tf.load_library` instead.')
@tf_export(v1=['load_file_system_library'])
def load_file_system_library(library_filename):
  # [FUNCTION DOC]
  # Deprecated (decorator points users to the newer load library function). Loads the native
  # library at library_filename (path string) so a file system implementation it provides gets
  # registered. Returns nothing; raises a runtime error if loading fails. Side effect is native
  # registration. Note the library handle is neither inspected nor released here.
  """Loads a TensorFlow plugin, containing file system implementation.

  Pass `library_filename` to a platform-specific mechanism for dynamically
  loading a library. The rules for determining the exact location of the
  library are platform-specific and are not documented here.

  Args:
    library_filename: Path to the plugin.
      Relative or absolute filesystem path to a dynamic library file.

  Returns:
    None.

  Raises:
    RuntimeError: when unable to load the library.
  """
  py_tf.TF_LoadLibrary(library_filename)


def _is_shared_object(filename):
  # [FUNCTION DOC]
  # HELPER: file filter for the library and pluggable device loaders.
  # KNOWN ISSUE: on Linux, a name ending in a dot-so-dot suffix with nothing after it makes the
  # character lookup go past the end of the string and raise an index error instead of returning
  # false. The docstring also says only the extension is used, which holds.
  # Decides from the file name alone (no file access) whether it is a shared object for the current
  # platform: on Linux a name ending in .so or containing .so. followed by a digit (versioned);
  # on macOS ending in .dylib; on Windows ending in .dll; any other platform gives false.
  # Takes a file name string and returns a boolean.
  """Check the file to see if it is a shared object, only using extension."""
  if platform.system() == 'Linux':
    if filename.endswith('.so'):
      return True
    else:
      index = filename.rfind('.so.')
      if index == -1:
        return False
      else:
        # A shared object with the API version in filename
        return filename[index + 4].isdecimal()
  elif platform.system() == 'Darwin':
    return filename.endswith('.dylib')
  elif platform.system() == 'Windows':
    return filename.endswith('.dll')
  else:
    return False


@tf_export('load_library')
def load_library(library_location):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says that for a folder only shared objects named with the kernel
  # library prefix are loaded, but the code loads every file in the folder that looks like a shared
  # object, with no prefix check. Subdirectories are not searched.
  # Loads kernel libraries so their kernels are registered in the process. library_location is a
  # path string to one library file or to a folder. For a folder, lists its entries and loads each
  # one accepted by the shared-object check; for a file, loads that file as given (no extension
  # check). Returns nothing. Raises an OSError (file-not-found code) if the path does not exist;
  # the native loader may raise a runtime error. Side effect: native kernel registration.
  """Loads a TensorFlow plugin.

  "library_location" can be a path to a specific shared object, or a folder.
  If it is a folder, all shared objects that are named "libtfkernel*" will be
  loaded. When the library is loaded, kernels registered in the library via the
  `REGISTER_*` macros are made available in the TensorFlow process.

  Args:
    library_location: Path to the plugin or the folder of plugins.
      Relative or absolute filesystem path to a dynamic library file or folder.

  Returns:
    None

  Raises:
    OSError: When the file to be loaded is not found.
    RuntimeError: when unable to load the library.
  """
  if os.path.exists(library_location):
    if os.path.isdir(library_location):
      directory_contents = os.listdir(library_location)

      kernel_libraries = [
          os.path.join(library_location, f) for f in directory_contents
          if _is_shared_object(f)]
    else:
      kernel_libraries = [library_location]

    for lib in kernel_libraries:
      py_tf.TF_LoadLibrary(lib)

  else:
    raise OSError(
        errno.ENOENT,
        'The file or folder to load kernel libraries from does not exist.',
        library_location)


def load_pluggable_device_library(library_location):
  # [FUNCTION DOC]
  # Loads PluggableDevice plugins whose devices and kernels get registered via the native stream
  # executor and kernel/op registration interfaces. library_location is a path string to one
  # library or a folder; for a folder every entry passing the shared-object check is loaded (no
  # recursion). After loading, it tells the eager context to reinitialize its list of physical
  # devices so the new devices are visible. Returns nothing. Raises an OSError (file-not-found code)
  # when the path does not exist; the native loader may raise a runtime error.
  # Not exported under a public name in this module.
  """Loads a TensorFlow PluggableDevice plugin.

  "library_location" can be a path to a specific shared object, or a folder.
  If it is a folder, all shared objects will be loaded. when the library is
  loaded, devices/kernels registered in the library via StreamExecutor C API
  and Kernel/Op Registration C API are made available in TensorFlow process.

  Args:
    library_location: Path to the plugin or folder of plugins. Relative or
      absolute filesystem path to a dynamic library file or folder.

  Raises:
    OSError: When the file to be loaded is not found.
    RuntimeError: when unable to load the library.
  """
  if os.path.exists(library_location):
    if os.path.isdir(library_location):
      directory_contents = os.listdir(library_location)

      pluggable_device_libraries = [
          os.path.join(library_location, f)
          for f in directory_contents
          if _is_shared_object(f)
      ]
    else:
      pluggable_device_libraries = [library_location]

    for lib in pluggable_device_libraries:
      py_tf.TF_LoadPluggableDeviceLibrary(lib)
    # Reinitialized physical devices list after plugin registration.
    context.context().reinitialize_physical_devices()
  else:
    raise OSError(
        errno.ENOENT,
        'The file or folder to load pluggable device libraries from does not '
        'exist.', library_location)


@tf_export('experimental.register_filesystem_plugin')
def register_filesystem_plugin(plugin_location):
  # [FUNCTION DOC]
  # Registers a file system plugin from the dynamic library at plugin_location (path string).
  # If the path exists, passes it to the native filesystem plugin registration; the path is not
  # checked to be a file or to have a particular extension. Returns nothing. Raises an OSError
  # (file-not-found code) if the path does not exist; native registration may raise a runtime error.
  # Side effect: registers new file system schemes in the process.
  """Loads a TensorFlow FileSystem plugin.

  Args:
    plugin_location: Path to the plugin. Relative or absolute filesystem plugin
      path to a dynamic library file.

  Returns:
    None

  Raises:
    OSError: When the file to be loaded is not found.
    RuntimeError: when unable to load the library.
  """
  if os.path.exists(plugin_location):
    py_tf.TF_RegisterFilesystemPlugin(plugin_location)

  else:
    raise OSError(errno.ENOENT,
                  'The file to load file system plugin from does not exist.',
                  plugin_location)
