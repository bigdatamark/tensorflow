# [MODULE DOC]
# HELPER: small utilities for locating the directories of Python modules, used by tooling that
# needs file-system paths of imported packages.
# Exposes a function returning the parent directory of a loaded module and one that resolves a
# dotted module name to a directory path without importing it. No import-time side effects.
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
"""Helper functions for modules."""
import os

import importlib


def get_parent_dir(module):
  # [FUNCTION DOC]
  # Returns the absolute path of the directory one level above the directory containing the given
  # loaded module object's source file. Raises an attribute error if the module has no file
  # attribute (for example built-in modules).
  return os.path.abspath(os.path.join(os.path.dirname(module.__file__), ".."))


def get_parent_dir_for_name(module_name):
  # [FUNCTION DOC]
  # KNOWN ISSUE: only the top-level import package is imported; the code uses the importlib utility
  # submodule without importing it explicitly, so it works only if something else already loaded it.
  # Splits the dotted module_name, finds the import spec of its first component (without
  # importing it) and returns the directory of that spec's origin joined with the remaining
  # components except the last one. Returns the null value if the name has no components, the
  # spec lookup raises a value error, or no spec/origin is found.
  """Get parent directory for module with the given name.

  Args:
    module_name: Module name, e.g. tf_keras.api._v2.keras.

  Returns:
    Path to the parent directory if module is found and None otherwise.
    Given example above, it should return:
      /root_path/tf_keras/api/_v2.
  """
  name_split = module_name.split(".")
  if not name_split:
    return None

  try:
    spec = importlib.util.find_spec(name_split[0])
  except ValueError:
    return None
  if not spec or not spec.origin:
    return None
  base_path = os.path.dirname(spec.origin)
  return os.path.join(base_path, *name_split[1:-1])
