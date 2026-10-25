# [MODULE DOC]
# HELPER: import-time sanity check support for the TensorFlow package. Defines the key under which
# the build information lists required Microsoft C++ runtime library names, and a function that
# verifies the environment before the native extension is loaded. Import-time side effect:
# imports the generated build-info module and, if that import fails, raises an import error
# explaining that TensorFlow must not be imported from its source directory.

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

"""Platform-specific code for checking the integrity of the TensorFlow build."""
import ctypes
import os

MSVCP_DLL_NAMES = "msvcp_dll_names"

try:
  from tensorflow.python.platform import build_info
except ImportError:
  raise ImportError("Could not import tensorflow. Do not import tensorflow "
                    "from its source directory; change directory to outside "
                    "the TensorFlow source tree, and relaunch your Python "
                    "interpreter from there.")


def preload_check():
  # [FUNCTION DOC]
  # HELPER: called early during package import to give actionable errors before native loading.
  # KNOWN ISSUE: the docstring says it raises only when loading will fail; on non-Windows systems
  # it merely loads the CPU feature guard extension, which logs information (and can itself raise
  # an import error if the extension is missing) rather than checking configuration.
  # On Windows: for each comma-separated library name in the build information entry, tries to
  # load it; if any fail, raises an import error naming the missing libraries and how to install
  # the Visual C++ redistributable. On other systems: imports the CPU feature guard extension and
  # asks it to report CPU features compiled in but unused (side effect: may log a warning).
  # No inputs; returns nothing.
  """Raises an exception if the environment is not correctly configured.

  Raises:
    ImportError: If the check detects that the environment is not correctly
      configured, and attempting to load the TensorFlow runtime will fail.
  """
  if os.name == "nt":
    # Attempt to load any DLLs that the Python extension depends on before
    # we load the Python extension, so that we can raise an actionable error
    # message if they are not found.
    if MSVCP_DLL_NAMES in build_info.build_info:
      missing = []
      for dll_name in build_info.build_info[MSVCP_DLL_NAMES].split(","):
        try:
          ctypes.WinDLL(dll_name)
        except OSError:
          missing.append(dll_name)
      if missing:
        raise ImportError(
            "Could not find the DLL(s) %r. TensorFlow requires that these DLLs "
            "be installed in a directory that is named in your %%PATH%% "
            "environment variable. You may install these DLLs by downloading "
            '"Microsoft C++ Redistributable for Visual Studio 2015, 2017 and '
            '2019" for your platform from this URL: '
            "https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads"
            % " or ".join(missing))
  else:
    # Load a library that performs CPU feature guard checking.  Doing this here
    # as a preload check makes it more likely that we detect any CPU feature
    # incompatibilities before we trigger them (which would typically result in
    # SIGILL).
    from tensorflow.python.platform import _pywrap_cpu_feature_guard
    _pywrap_cpu_feature_guard.InfoAboutUnusedCPUFeatures()
