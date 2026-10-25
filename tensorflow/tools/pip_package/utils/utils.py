# [MODULE DOC]
# HELPER: shared utilities for the pip package builder script (platform checks and file
# operations used while staging the wheel contents). Exposes platform predicates, a function that
# creates package init files, an in-place text replacement over python files, and a file copy
# function with prefix stripping. No import-time side effects.
# Copyright 2023 The Tensorflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for pip package builder."""
import os
import shutil
import sys


def is_windows() -> bool:
  # [FUNCTION DOC]
  # HELPER: platform check for the wheel builder. Returns true when the running platform
  # identifier starts with the 32-bit-style Windows name used by Python, false otherwise.
  return sys.platform.startswith("win32")


def is_macos() -> bool:
  # [FUNCTION DOC]
  # HELPER: platform check for the wheel builder. Returns true when the running platform
  # identifier starts with the Darwin name, false otherwise.
  return sys.platform.startswith("darwin")


def create_init_files(dst_dir: str) -> None:
  # [FUNCTION DOC]
  # HELPER: makes the staged tree importable as packages.
  # Input: dst_dir, root of the tree to scan. For every directory containing a python or shared
  # object file, creates an empty package init file in that directory and in each parent up to
  # but excluding the root, unless one already exists. Writes files on disk. Returns nothing.
  """Create __init__.py files."""
  for root, _, files in os.walk(dst_dir):
    if any(file.endswith(".py") or file.endswith(".so") for file in files):
      curr_dir = root
      while curr_dir != dst_dir:
        init_path = os.path.join(curr_dir, "__init__.py")
        if not os.path.exists(init_path):
          open(init_path, "w").close()
        curr_dir = os.path.dirname(curr_dir)


def replace_inplace(directory, search, to_replace) -> None:
  # [FUNCTION DOC]
  # HELPER: bulk text substitution for staged python sources.
  # Inputs: directory (root to walk), search (text to find) and to_replace (replacement text).
  # In every python file below the root that contains the search text, replaces all occurrences
  # and rewrites the file. Files are read as UTF-8 but written with the platform default
  # encoding. Modifies files in place; returns nothing.
  """Traverse the directory and replace search phrase in each file."""
  for root, _, files in os.walk(directory):
    for file_name in files:
      if file_name.endswith(".py"):
        file_path = os.path.join(root, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
          filedata = file.read()
        if search in filedata:
          filedata = filedata.replace(search, to_replace)
          with open(file_path, "w") as file:
            file.write(filedata)


def copy_file(
    src_file: str,
    dst_dir: str,
    strip: str = None,
    dest_file: str = None,
) -> None:
  # [FUNCTION DOC]
  # HELPER: copies one file into the staging tree for the wheel builder.
  # Inputs: src_file (file to copy), dst_dir (destination root), strip (optional prefix removed
  # from the relative destination path), dest_file (optional destination path used instead of the
  # source path). The relative destination is the given destination path or the source path; a
  # build output prefix up to and including the bin directory is dropped for generated files,
  # then the strip prefix is removed. Creates missing directories, copies the file, and sets its
  # permissions to owner read-write and group/other read-only. Raises file errors on failure.
  # Returns nothing.
  """Copy a file to the destination directory.

  Args:
    src_file: file to be copied
    dst_dir: destination directory
    strip: prefix to strip before copying to destination
    dest_file: destanation file location if different from src_file
  """

  # strip `bazel-out/.../bin/` for generated files.
  dest = dest_file if dest_file else src_file
  if dest.startswith("bazel-out"):
    dest = dest[dest.index("bin") + 4 :]

  if strip:
    dest = dest.removeprefix(strip)
  dest_dir_path = os.path.join(dst_dir, os.path.dirname(dest))
  os.makedirs(dest_dir_path, exist_ok=True)
  shutil.copy(src_file, dest_dir_path)
  os.chmod(os.path.join(dst_dir, dest), 0o644)
