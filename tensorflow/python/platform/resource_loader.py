# [MODULE DOC]
# Utilities for locating and reading data files that ship alongside code, including under a
# Bazel runfiles tree. Offers: reading a resource's bytes, the directory of the calling script,
# a root directory that holds all declared data dependencies, the resolved path of a data file,
# and a no-op readahead helper. All are exported in the v1 API only. Import-time side effect:
# tries to import the Bazel runfiles library and falls back to a null placeholder if missing.

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
"""Resource management library."""
import os as _os
import sys as _sys

from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.tf_export import tf_export

# pylint: disable=g-import-not-at-top
try:
  from python.runfiles import runfiles
except ImportError:
  runfiles = None
# pylint: enable=g-import-not-at-top


@tf_export(v1=['resource_loader.load_resource'])
def load_resource(path):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the fallback path resolution in the data-file path function inspects the caller's
  # frame, and here the caller is this module, so a relative path that is not found in runfiles is
  # resolved against this module's directory, not the directory of the code calling this function.
  # Reads and returns the full contents (bytes) of the resource at a path relative to the
  # tensorflow directory. Input: path string. Raises an IO error if the file cannot be opened.
  """Load the resource at given path, where path is relative to tensorflow/.

  Args:
    path: a string resource path relative to tensorflow/.

  Returns:
    The contents of that resource.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  with open(get_path_to_datafile(path), 'rb') as f:
    return f.read()


# pylint: disable=protected-access
@tf_export(v1=['resource_loader.get_data_files_path'])
def get_data_files_path():
  # [FUNCTION DOC]
  # Returns the directory containing the source file of the immediate caller (found by inspecting
  # the call stack one level up), which is where data files declared in the build rule live. No
  # inputs; no side effects. Behavior depends on who calls it, so wrapping it changes the result.
  """Get a direct path to the data files colocated with the script.

  Returns:
    The directory where files specified in data attribute of py_test
    and py_binary are stored.
  """
  return _os.path.dirname(_inspect.getfile(_sys._getframe(1)))


@tf_export(v1=['resource_loader.get_root_dir_with_all_resources'])
def get_root_dir_with_all_resources():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns the path to a specified file, but it takes no
  # arguments and returns a directory. Also the directory examined is that of this module's
  # caller chain start: the data-files-path function is called from here, so the starting point
  # is this module's own directory rather than the caller's.
  # Walks up from the starting script directory, collecting parent directories, until it finds a
  # directory whose name contains the runfiles marker; the directory just below that (the
  # repository root in the runfiles tree) is returned. If the marker is never found, the root of
  # the filesystem is reached, or the script is directly inside runfiles, returns the starting
  # script directory instead. No inputs; no side effects.
  """Get a root directory containing all the data attributes in the build rule.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary. Falls back to returning the same as get_data_files_path if it
    fails to detect a bazel runfiles directory.
  """
  script_dir = get_data_files_path()

  # Create a history of the paths, because the data files are located relative
  # to the repository root directory, which is directly under runfiles
  # directory.
  directories = [script_dir]
  data_files_dir = ''

  while True:
    candidate_dir = directories[-1]
    current_directory = _os.path.basename(candidate_dir)
    if '.runfiles' in current_directory:
      # Our file should never be directly under runfiles.
      # If the history has only one item, it means we are directly inside the
      # runfiles directory, something is wrong, fall back to the default return
      # value, script directory.
      if len(directories) > 1:
        data_files_dir = directories[-2]

      break
    else:
      new_candidate_dir = _os.path.dirname(candidate_dir)
      # If we are at the root directory these two will be the same.
      if new_candidate_dir == candidate_dir:
        break
      else:
        directories.append(new_candidate_dir)

  return data_files_dir or script_dir


@tf_export(v1=['resource_loader.get_path_to_datafile'])
def get_path_to_datafile(path):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises an IO error if the path is not found, but it never
  # checks existence of the fallback path; it returns an unchecked old-style path instead. The
  # source comment itself calls the old-style fallback buggy but kept for compatibility.
  # Resolves a data file path (string relative to the tensorflow directory). If the runfiles
  # library is available, creates a runfiles locator and looks up the path under the tensorflow
  # directory; if the located file exists it is returned. Otherwise returns the path joined to the
  # directory of the immediate caller's source file. Side effect: creates a runfiles object each
  # call; may raise if the runfiles environment is not set up.
  """Get the path to the specified file in the data dependencies.

  The path is relative to tensorflow/

  Args:
    path: a string resource path relative to tensorflow/

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  # First, try finding in the new path.
  if runfiles:
    r = runfiles.Create()
    new_fpath = r.Rlocation(
        _os.path.abspath(_os.path.join('tensorflow', path)))
    if new_fpath is not None and _os.path.exists(new_fpath):
      return new_fpath

  # Then, the old style path, as people became dependent on this buggy call.
  old_filepath = _os.path.join(
      _os.path.dirname(_inspect.getfile(_sys._getframe(1))), path)
  return old_filepath


@tf_export(v1=['resource_loader.readahead_file_path'])
def readahead_file_path(path, readahead='128M'):  # pylint: disable=unused-argument
  # [FUNCTION DOC]
  # No-op stand-in for an internal read-ahead feature. Inputs: path (string) and readahead size
  # text (default 128M, ignored). Returns the path unchanged. No side effects.
  """Readahead files not implemented; simply returns given path."""
  return path
