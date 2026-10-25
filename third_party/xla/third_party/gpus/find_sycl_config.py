# [MODULE DOC]
# HELPER: command-line helper script used by the SYCL repository rule of the GPU build setup.
# Discovers an Intel oneAPI/SYCL toolchain on the build machine and prints its configuration.
# The compiler toolkit directory comes from the SYCL_TOOLKIT_PATH environment variable (default
# is the "latest" compiler directory of the standard oneAPI install) and is resolved to a real
# path; the base kit directory and base kit version are derived from the part of the path
# around the compiler directory. It reads the SYCL library version from its version header,
# then finds the oneMKL header and library directories (searched under MKL_INSTALL_PATH, or
# the kit's mkl directory by default) and the Level Zero header and loader library directories
# (searched under L0_INSTALL_PATH, or the system prefix by default). Output is sorted
# "key: value" lines on standard output; on failure an error goes to standard error and the exit
# status is 1. Defines the ConfigError exception. No import-time side effects.
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Prints SYCL library and header directories and versions found on the system.

The script searches for SYCL library and header files on the system, inspects
them to determine their version and prints the configuration to stdout. The path
to inspect is specified through an environment variable (SYCL_PATH). If no valid
configuration is found, the script prints to stderr and returns an error code.
The script takes the directory specified by the SYCL_PATH environment variable.
The script looks for headers and library files in a hard-coded set of
subdirectories from base path of the specified directory. If SYCL_PATH is not
specified, then "/opt/sycl" is used as it default value
"""

import io
import os
import glob
import re
import sys


class ConfigError(Exception):
  # [CLASS DOC]
  # Exception type raised when a SYCL, MKL or Level Zero file cannot be found or a version
  # macro cannot be parsed. Holds only a message; the main entry point catches it, prints it to
  # standard error and exits with failure status.
  pass

def _get_default_sycl_toolkit_path():
  # [FUNCTION DOC]
  # HELPER: supplies the fallback SYCL compiler directory for the toolkit path lookup.
  # Takes no inputs. Returns the fixed default oneAPI compiler directory string. No side effects.
  return "/opt/intel/oneapi/compiler/latest"

def _get_toolkit_path():
  # [FUNCTION DOC]
  # HELPER: resolves the SYCL compiler toolkit directory used by the other lookups.
  # Takes no inputs. Uses the SYCL_TOOLKIT_PATH environment variable when set, otherwise the
  # default path, and returns it with symbolic links resolved to a real absolute path. Reads
  # the environment and the file system (for link resolution) only; never raises for a missing
  # directory.
  """Determines and returns the SYCL installation path."""
  sycl_toolkit_path = None
  sycl_toolkit_path = _get_default_sycl_toolkit_path()
  if "SYCL_TOOLKIT_PATH" in os.environ:
    sycl_toolkit_path = os.environ["SYCL_TOOLKIT_PATH"]
  return os.path.realpath(sycl_toolkit_path)

def _get_basekit_path():
  # [FUNCTION DOC]
  # HELPER: derives the oneAPI base kit root from the toolkit path.
  # Takes no inputs. Returns the part of the resolved toolkit path that precedes the compiler
  # directory component. If the path contains no such component the whole path is returned.
  return _get_toolkit_path().split("/compiler/")[0]

def _get_basekit_version():
  # [FUNCTION DOC]
  # HELPER: derives the oneAPI base kit version label from the toolkit path.
  # KNOWN ISSUE: uses a bare catch-all handler, which hides every error type, not only the
  # expected missing-component case.
  # Takes no inputs. Returns the directory name that follows the compiler directory component
  # in the resolved toolkit path, or the text "unknown" if the path has no such component.
  try:
    version = _get_toolkit_path().split("/compiler/")[1].split("/")[0]
  except:
    version = "unknown"
  return version

def _get_composite_version_number(major, minor, patch):
  # [FUNCTION DOC]
  # HELPER: packs a version triple into one comparable integer for the config output.
  # Inputs: major, minor and patch numbers (integers). Returns major times 10000 plus minor
  # times 100 plus patch. Pure function.
  return 10000 * major + 100 * minor + patch

def _list_from_env(env_name, default=[]):
  # [FUNCTION DOC]
  # HELPER: reads a comma-separated list setting from the environment.
  # KNOWN ISSUE: the default argument is a shared mutable list; it is returned as is, so a
  # caller that modifies the result would change the default for later calls.
  # Inputs: environment variable name and an optional default list (empty list by default).
  # Returns the variable's value split on commas (no trimming, an empty value gives a list with
  # one empty string) if the variable is set, otherwise the default. Reads the environment only.
  """Returns comma-separated list from environment variable."""
  if env_name in os.environ:
    return os.environ[env_name].split(",")
  return default

def _get_legacy_path(env_name, default=[]):
  # [FUNCTION DOC]
  # HELPER: reads a path-list environment variable, tolerating the legacy form that points at a
  # multiarch library directory.
  # KNOWN ISSUE: the default argument is a shared mutable list that may be returned directly.
  # Inputs: environment variable name and optional default list. If the variable is set and its
  # value is an absolute path ending in a library directory with a multiarch Linux triple
  # suffix, returns a one-item list holding the install prefix before the library directory.
  # Otherwise falls back to the comma-separated list reading (or the default if unset). Reads
  # the environment only.
  if env_name in os.environ:
    match = re.match(r"^(/[^/ ]*)+/lib/\w+-linux-gnu/?$", os.environ[env_name])
    if match:
      return [match.group(1)]
  return _list_from_env(env_name, default)

def _get_header_version(path, name):
  # [FUNCTION DOC]
  # HELPER: version-header parser used by the SYCL version lookup.
  # KNOWN ISSUE: the macro name is inserted into a regular expression without escaping, and the
  # file handle opened by the loop is never explicitly closed.
  # Inputs: path to a C/C++ header and the name of a preprocessor macro. Returns, as an
  # integer, the value of the first line that defines that macro as a decimal literal. Raises
  # ConfigError if there is no such line; file errors propagate. Read-only.
  """Returns preprocessor defines in C header file."""
  for line in io.open(path, "r", encoding="utf-8"):
    match = re.match(r"#define %s +(\d+)" % name, line)
    if match:
      value = match.group(1)
      return int(value)

  raise ConfigError(
      '#define "{}" is either\n'.format(name)
      + "  not present in file {} OR\n".format(path)
      + "  its value is not an integer literal"
  )


def _cartesian_product(first, second):
  # [FUNCTION DOC]
  # HELPER: builds candidate search paths for the file finder.
  # Inputs: a list of base directories and a list of relative sub-paths. Returns a list of every
  # base joined with every relative path, base-major order. Pure function.
  """Returns all path combinations of first and second."""
  return [os.path.join(f, s) for f in first for s in second]

def _header_paths():
  # [FUNCTION DOC]
  # HELPER: lists where header files are searched, for the file finder.
  # Takes no inputs. Returns a fixed list of relative directories: the base itself and the
  # include, Level Zero include and oneAPI include subdirectories.
  """Returns hard-coded set of relative paths to look for header files."""
  return [
      "",
      "include",
      "include/level_zero",
      "include/oneapi",
  ]

def _library_paths():
  # [FUNCTION DOC]
  # HELPER: lists where library files are searched, for the file finder.
  # Takes no inputs. Returns a fixed list of relative directories: the base itself, the 64-bit
  # library directory, the plain library directory and a wildcard multiarch library directory.
  """Returns hard-coded set of relative paths to look for library files."""
  return [
      "",
      "lib64",
      "lib",
      "lib/*-linux-gnu",
  ]

def _not_found_error(base_paths, relative_paths, filepattern):
  # [FUNCTION DOC]
  # HELPER: builds the failure exception for the file finder.
  # Inputs: collection of base directories, list of relative directories searched, and the file
  # name pattern that was sought. Returns (does not raise) a ConfigError whose message lists
  # the pattern, the relative directories and the sorted base directories.
  base_paths = "".join(["\n        '%s'" % path for path in sorted(base_paths)])
  relative_paths = "".join(["\n        '%s'" % path for path in relative_paths])
  return ConfigError(
      "Could not find any %s in any subdirectory:%s\nof:%s\n" %
      (filepattern, relative_paths, base_paths))

def _find_file(base_paths, relative_paths, filepattern):
  # [FUNCTION DOC]
  # HELPER: core search routine behind the header and library lookups.
  # Inputs: list of base directories, list of relative sub-directories and a file name pattern
  # that may contain wildcards. Tries each base/relative combination in order and returns the
  # first file matching the pattern. Raises ConfigError (built by the not-found helper) if
  # nothing matches. Read-only file system access.
  for path in _cartesian_product(base_paths, relative_paths):
    for file in glob.glob(os.path.join(path, filepattern)):
      return file
  raise _not_found_error(base_paths, relative_paths, filepattern)

def _find_header(base_paths, header_name):
  # [FUNCTION DOC]
  # HELPER: header lookup for the MKL and Level Zero probes.
  # Inputs: list of base directories and a header file name. Returns the path of the first match
  # under the standard header sub-directories. Raises ConfigError if not found.
  """Returns first valid path to a header."""
  return _find_file(base_paths, _header_paths(), header_name)

def _find_library(base_paths, library_name):
  # [FUNCTION DOC]
  # HELPER: shared-library lookup for the MKL and Level Zero probes.
  # Inputs: list of base directories and a library name without the prefix and suffix. Searches
  # for the shared object file with that name (any version suffix) under the standard library
  # sub-directories and returns the first matching path. Raises ConfigError if not found.
  """Returns first valid path to the requested library."""
  filepattern = ".".join(["lib" + library_name, "so"]) + "*"
  return _find_file(base_paths, _library_paths(), filepattern)

def _find_sycl_config(toolkit_path):
  # [FUNCTION DOC]
  # HELPER: SYCL version probe used by the main config function.
  # Input: compiler toolkit directory. Returns a mapping with the composite SYCL library version
  # number and the base kit version label. Raises ConfigError if the version header or a macro
  # is missing. Read-only.

  def sycl_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the SYCL version header.
    # Input: toolkit directory. Checks two candidate header locations and uses the first that
    # exists. Returns major, minor and patch library version numbers as a tuple. Raises
    # ConfigError if no file exists or a macro is absent.
    possible_version_files = [
        "linux/include/sycl/version.hpp",
        "include/sycl/version.hpp",
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "SYCL version file not found in {}".format(possible_version_files)
      )

    major = _get_header_version(version_file, "__LIBSYCL_MAJOR_VERSION")
    minor = _get_header_version(version_file, "__LIBSYCL_MINOR_VERSION")
    patch = _get_header_version(version_file, "__LIBSYCL_PATCH_VERSION")
    return major, minor, patch

  major, minor, patch = sycl_version_numbers(toolkit_path)

  sycl_config = {
    "sycl_version_number": _get_composite_version_number(major, minor, patch),
    "sycl_basekit_version_number": _get_basekit_version(),
  }

  return sycl_config

def _find_mkl_config(base_paths):
  # [FUNCTION DOC]
  # HELPER: oneMKL probe used by the main config function.
  # Input: list of candidate MKL install directories. Locates the MKL header and the MKL SYCL
  # library. Returns a mapping with the include root (the header's directory with its last path
  # component removed, i.e. the parent of the mkl header directory) and the library directory.
  # Raises ConfigError if either file is not found.
  header_path = os.path.dirname(_find_header(base_paths, "mkl.hpp"))
  library_path = os.path.dirname(_find_library(base_paths, "mkl_sycl"))
  return {
    "mkl_include_dir": "/".join(header_path.split("/")[:-1]),
    "mkl_library_dir": library_path,
  }

def _find_l0_config(base_paths):
  # [FUNCTION DOC]
  # HELPER: Level Zero probe used by the main config function.
  # Input: list of candidate Level Zero install directories. Locates the Level Zero API header
  # and the loader library. Returns a mapping with the header directory and the library
  # directory. Raises ConfigError if either file is not found.
  header_path = os.path.dirname(_find_header(base_paths, "ze_api.h"))
  library_path = os.path.dirname(_find_library(base_paths, "ze_loader"))
  return {
    "l0_include_dir": header_path,
    "l0_library_dir": library_path,
  }

def find_sycl_config():
  # [FUNCTION DOC]
  # Public entry point that gathers the whole SYCL configuration.
  # Takes no inputs; reads SYCL_TOOLKIT_PATH, MKL_INSTALL_PATH and L0_INSTALL_PATH from the
  # environment. Returns a mapping with the base kit path, toolkit path, SYCL version numbers,
  # MKL include and library directories and Level Zero include and library directories.
  # Raises ConfigError if any component is missing. Read-only.
  """Returns a dictionary of SYCL components config info."""
  basekit_path = _get_basekit_path()
  toolkit_path = _get_toolkit_path()

  result = {}
  result["sycl_basekit_path"] = basekit_path
  result["sycl_toolkit_path"] = toolkit_path
  result.update(_find_sycl_config(toolkit_path))

  default_mkl_path = [basekit_path + "/mkl/" + _get_basekit_version()]
  mkl_paths = _get_legacy_path("MKL_INSTALL_PATH", default_mkl_path)
  result.update(_find_mkl_config(mkl_paths))

  default_l0_path = ["/usr"]
  l0_paths = _get_legacy_path("L0_INSTALL_PATH", default_l0_path)
  result.update(_find_l0_config(l0_paths))

  return result


def main():
  # [FUNCTION DOC]
  # Command-line entry point. Takes no arguments. Prints each discovered setting as a name and
  # value line on standard output, sorted by name. On a ConfigError it writes an error message
  # to standard error and terminates the process with exit status 1.
  try:
    for key, value in sorted(find_sycl_config().items()):
      print("%s: %s" % (key, value))
  except ConfigError as e:
    sys.stderr.write("\nERROR: {}\n\n".format(str(e)))
    sys.exit(1)


if __name__ == "__main__":
  main()
