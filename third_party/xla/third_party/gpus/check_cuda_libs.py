# [MODULE DOC]
# Deprecated helper script used by the legacy CUDA configure repository rule to verify that
# given shared libraries exist on disk and (optionally) that their SONAME matches the file name.
# Run as a script with pairs of arguments: library path and the word True or False for whether
# to check the SONAME. Exposes a configuration error type, a per-library check and main, which
# prints the verified paths one per line on success or writes the error to standard error and
# exits with status 1. No import-time side effects.
# HELPER: support script for the legacy CUDA configure rule.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Verifies that a list of libraries is installed on the system.

NB: DEPRECATED! This script is a part of the deprecated `cuda_configure` rule.
Please use `hermetic/cuda_configure` instead.

Takes a list of arguments with every two subsequent arguments being a logical
tuple of (path, check_soname). The path to the library and either True or False
to indicate whether to check the soname field on the shared library.

Example Usage:
./check_cuda_libs.py /path/to/lib1.so True /path/to/lib2.so False
"""
import os
import os.path
import platform
import shutil
import subprocess
import sys


class ConfigError(Exception):
  # [CLASS DOC]
  # Exception type signalling a configuration problem: a missing library, a SONAME mismatch or a
  # wrong argument count. Carries only the message.
  pass


def _is_windows():
  # [FUNCTION DOC]
  # HELPER: platform test for check_cuda_lib.
  # Returns true if the platform system name reports Windows, otherwise false.
  return platform.system() == "Windows"


def check_cuda_lib(path, check_soname=True):
  # [FUNCTION DOC]
  # Verifies one library. Inputs: path (string) of the library file and check_soname (boolean,
  # default true). Raises ConfigError if the path is not an existing regular file. If SONAME
  # checking is requested, the objdump tool is found on the search path and the platform is not
  # Windows, runs objdump to print the private headers, collects the last token of each SONAME
  # line, and raises ConfigError if none equals the file's base name. If objdump is missing or on
  # Windows, the SONAME check is silently skipped. Returns nothing; side effect: runs a
  # subprocess (which may raise a called-process error on failure).
  """Tests if a library exists on disk and whether its soname matches the filename.

  Args:
    path: the path to the library.
    check_soname: whether to check the soname as well.

  Raises:
    ConfigError: If the library does not exist or if its soname does not match
    the filename.
  """
  if not os.path.isfile(path):
    raise ConfigError("No library found under: " + path)
  objdump = shutil.which("objdump")
  if check_soname and objdump is not None and not _is_windows():
    # Decode is necessary as in py3 the return type changed from str to bytes
    output = subprocess.check_output([objdump, "-p", path]).decode("utf-8")
    output = [line for line in output.splitlines() if "SONAME" in line]
    sonames = [line.strip().split(" ")[-1] for line in output]
    if not any(soname == os.path.basename(path) for soname in sonames):
      raise ConfigError("None of the libraries match their SONAME: " + path)


def main():
  # [FUNCTION DOC]
  # Script entry point. Reads the process arguments (excluding the program name), which must come
  # in pairs of library path and a flag word; the flag enables the SONAME check only when it is
  # exactly the word True. Checks each library in order and, if all pass, prints the paths joined
  # by line separators to standard output. On a configuration error (including an odd number of
  # arguments) writes the message (no trailing newline) to standard error and exits with status 1.
  # Returns nothing otherwise.
  try:
    args = [argv for argv in sys.argv[1:]]
    if len(args) % 2 == 1:
      raise ConfigError("Expected even number of arguments")
    checked_paths = []
    for i in range(0, len(args), 2):
      path = args[i]
      check_cuda_lib(path, check_soname=args[i + 1] == "True")
      checked_paths.append(path)
    # pylint: disable=superfluous-parens
    print(os.linesep.join(checked_paths))
    # pylint: enable=superfluous-parens
  except ConfigError as e:
    sys.stderr.write(str(e))
    sys.exit(1)


if __name__ == "__main__":
  main()
