# [MODULE DOC]
# Deprecated script used by the legacy CUDA configure repository rule to discover CUDA, cuBLAS,
# cuSOLVER, cuRAND, cuFFT, cuSPARSE, cuDNN, NCCL and TensorRT installations. Library names are
# given as command line arguments; search roots and required versions come from environment
# variables (TF_CUDA_PATHS, per-library version variables and legacy install path variables).
# For each requested library it finds header and library directories by globbing a fixed set of
# relative subdirectories under the base paths, determines the version from header defines (or
# nvcc output), and prints sorted key: value lines to standard output. On a configuration error
# it writes the message to standard error and exits with status 1. Mostly private helpers plus
# find_cuda_config (returns a mapping) and main. No import-time side effects.
# HELPER: support script for the legacy CUDA configure rule.
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
"""Prints CUDA library and header directories and versions found on the system.

NB: DEPRECATED! This script is a part of the deprecated `cuda_configure` rule.
Please use `hermetic/cuda_configure` instead.

The script searches for CUDA library and header files on the system, inspects
them to determine their version and prints the configuration to stdout.
The paths to inspect and the required versions are specified through environment
variables. If no valid configuration is found, the script prints to stderr and
returns an error code.

The list of libraries to find is specified as arguments. Supported libraries are
CUDA (includes cuBLAS), cuDNN, NCCL, and TensorRT.

The script takes a list of base directories specified by the TF_CUDA_PATHS
environment variable as comma-separated glob list. The script looks for headers
and library files in a hard-coded set of subdirectories from these base paths.
If TF_CUDA_PATHS is not specified, a OS specific default is used:

  Linux:   /usr/local/cuda, /usr, and paths from 'ldconfig -p'.
  Windows: CUDA_PATH environment variable, or
           C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\*

For backwards compatibility, some libraries also use alternative base
directories from other environment variables if they are specified. List of
library-specific environment variables:

  Library   Version env variable  Additional base directories
  ----------------------------------------------------------------
  CUDA      TF_CUDA_VERSION       CUDA_TOOLKIT_PATH
  cuBLAS    TF_CUBLAS_VERSION     CUDA_TOOLKIT_PATH
  cuDNN     TF_CUDNN_VERSION      CUDNN_INSTALL_PATH
  NCCL      TF_NCCL_VERSION       NCCL_INSTALL_PATH, NCCL_HDR_PATH
  TensorRT  TF_TENSORRT_VERSION   TENSORRT_INSTALL_PATH

Versions environment variables can be of the form 'x' or 'x.y' to request a
specific version, empty or unspecified to accept any version.

The output of a found library is of the form:
tf_<library>_version: x.y.z
tf_<library>_header_dir: ...
tf_<library>_library_dir: ...
"""

import glob
import io
import os
import platform
import re
import shutil
import subprocess
import sys


class ConfigError(Exception):
  # [CLASS DOC]
  # Exception type signalling that a requested library, header or tool could not be found or the
  # discovered configuration is inconsistent. Carries a human readable message printed by main.
  pass


def _is_linux():
  # [FUNCTION DOC]
  # HELPER: platform test. Returns true if the platform system name is Linux.
  return platform.system() == "Linux"


def _is_windows():
  # [FUNCTION DOC]
  # HELPER: platform test. Returns true if the platform system name is Windows.
  return platform.system() == "Windows"


def _is_macos():
  # [FUNCTION DOC]
  # HELPER: platform test. Returns true if the platform system name is Darwin (macOS).
  return platform.system() == "Darwin"


def _matches_version(actual_version, required_version):
  # [FUNCTION DOC]
  # HELPER: version filter used when searching headers and tools.
  # Decides whether a detected version satisfies a requirement by prefix match after trimming
  # whitespace. Inputs: actual_version (string or nothing) and required_version (string; empty
  # means any). Returns false when the actual version is nothing, otherwise whether the actual
  # string starts with the required one (so "1" matches "1.1" but "1.2" does not match "1.3").
  # Note the prefix match also lets "1" match "10.0". No side effects.
  """Checks whether some version meets the requirements.

      All elements of the required_version need to be present in the
      actual_version.

          required_version  actual_version  result
          -----------------------------------------
          1                 1.1             True
          1.2               1               False
          1.2               1.3             False
                            1               True

      Args:
        required_version: The version specified by the user.
        actual_version: The version detected from the CUDA installation.
      Returns: Whether the actual version matches the required one.
  """
  if actual_version is None:
    return False

  # Strip spaces from the versions.
  actual_version = actual_version.strip()
  required_version = required_version.strip()
  return actual_version.startswith(required_version)


def _at_least_version(actual_version, required_version):
  # [FUNCTION DOC]
  # HELPER: numeric version comparison. Inputs: two dotted version strings of integers. Returns
  # true if the actual version, compared as a list of integers element by element, is greater
  # than or equal to the required one. Raises a value error if a component is not an integer.
  actual = [int(v) for v in actual_version.split(".")]
  required = [int(v) for v in required_version.split(".")]
  return actual >= required


def _get_header_version(path, name):
  # [FUNCTION DOC]
  # HELPER: reads a numeric define from a C header.
  # KNOWN ISSUE: the file handle is opened and never explicitly closed (relies on garbage
  # collection), and the docstring says "defines" but only a single define value is returned.
  # Inputs: path of a header file and name of a preprocessor define. Scans lines in order and
  # returns the digits of the first define of that name as a string; returns an empty string if
  # no such define is found. Raises a file error if the file cannot be read.
  """Returns preprocessor defines in C header file."""
  for line in io.open(path, "r", encoding="utf-8").readlines():
    match = re.match(r"\s*#\s*define %s\s+(\d+)" % name, line)
    if match:
      return match.group(1)
  return ""


def _cartesian_product(first, second):
  # [FUNCTION DOC]
  # HELPER: builds candidate directories. Inputs: two sequences of path strings. Returns a list
  # of all joins of each element of the first with each element of the second, first-major order.
  """Returns all path combinations of first and second."""
  return [os.path.join(f, s) for f in first for s in second]


def _get_ld_config_paths():
  # [FUNCTION DOC]
  # HELPER: contributes system library directories to the default search paths.
  # Returns an empty list when not on Linux. Otherwise runs the dynamic linker cache listing
  # command (from the search path, falling back to a standard location), parses lines of the form
  # name => path, and returns the sorted list of unique directories of those paths. Lines that are
  # not ASCII decodable are ignored. Raises a called-process or file error if the command cannot be
  # run. Side effect: spawns a subprocess.
  """Returns all directories from 'ldconfig -p'."""
  if not _is_linux():
    return []
  ldconfig_path = shutil.which("ldconfig") or "/sbin/ldconfig"
  output = subprocess.check_output([ldconfig_path, "-p"])
  pattern = re.compile(".* => (.*)")
  result = set()
  for line in output.splitlines():
    try:
      match = pattern.match(line.decode("ascii"))
    except UnicodeDecodeError:
      match = False
    if match:
      result.add(os.path.dirname(match.group(1)))
  return sorted(list(result))


def _get_default_cuda_paths(cuda_version):
  # [FUNCTION DOC]
  # HELPER: default search roots when no explicit path list is configured.
  # Input: cuda_version, the requested version string (empty means any). An empty version becomes
  # a wildcard and a version without a dot gets a wildcard minor part. On Windows returns a
  # one-element list: the CUDA path environment variable or the default toolkit install directory
  # for that version. Otherwise returns the versioned and unversioned local CUDA directories, the
  # system prefix, a local cuDNN directory, followed by directories from the linker cache.
  if not cuda_version:
    cuda_version = "*"
  elif not "." in cuda_version:
    cuda_version = cuda_version + ".*"

  if _is_windows():
    return [
        os.environ.get(
            "CUDA_PATH",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%s\\" %
            cuda_version)
    ]
  return ["/usr/local/cuda-%s" % cuda_version, "/usr/local/cuda", "/usr",
         "/usr/local/cudnn"] + _get_ld_config_paths()


def _header_paths():
  # [FUNCTION DOC]
  # HELPER: returns the fixed list of relative subdirectories (under each base path) where header
  # files are searched, including CUPTI and architecture specific include directories.
  """Returns hard-coded set of relative paths to look for header files."""
  return [
      "",
      "include",
      "include/cuda",
      "include/*-linux-gnu",
      "extras/CUPTI/include",
      "include/cuda/CUPTI",
      "local/cuda/extras/CUPTI/include",
      "targets/x86_64-linux/include",
  ]


def _library_paths():
  # [FUNCTION DOC]
  # HELPER: returns the fixed list of relative subdirectories (under each base path) where
  # library files are searched, including lib64, architecture specific and CUPTI directories.
  """Returns hard-coded set of relative paths to look for library files."""
  return [
      "",
      "lib64",
      "lib",
      "lib/*-linux-gnu",
      "lib/x64",
      "extras/CUPTI/*",
      "local/cuda/lib64",
      "local/cuda/extras/CUPTI/lib64",
  ]


def _not_found_error(base_paths, relative_paths, filepattern):
  # [FUNCTION DOC]
  # HELPER: builds (does not raise) the error for failed searches. Inputs: base paths, relative
  # subdirectories and the file pattern searched. Returns a ConfigError whose message names the
  # pattern, the searched subdirectories and the sorted base paths, one per line.
  base_paths = "".join(["\n        '%s'" % path for path in sorted(base_paths)])
  relative_paths = "".join(["\n        '%s'" % path for path in relative_paths])
  return ConfigError(
      "Could not find any %s in any subdirectory:%s\nof:%s\n" %
      (filepattern, relative_paths, base_paths))


def _find_file(base_paths, relative_paths, filepattern):
  # [FUNCTION DOC]
  # HELPER: finds the first file matching a glob pattern. Inputs: base paths, relative
  # subdirectories and a file pattern. Tries every base/relative combination in order and returns
  # the first glob match. Raises a ConfigError (via the not-found helper) when nothing matches.
  for path in _cartesian_product(base_paths, relative_paths):
    for file in glob.glob(os.path.join(path, filepattern)):
      return file
  raise _not_found_error(base_paths, relative_paths, filepattern)


def _find_library(base_paths, library_name, required_version):
  # [FUNCTION DOC]
  # HELPER: locates a shared library file. Inputs: base paths, library name without prefix and
  # a required version string (only its major part, before the first dot, is used). The file
  # pattern depends on the platform: Windows uses the import library suffix; macOS uses the
  # dynamic library suffix with the major version; others use the shared object suffix followed
  # by the major version and a wildcard. Searches the fixed library subdirectories and returns the
  # first match; raises ConfigError if none is found.
  """Returns first valid path to the requested library."""
  if _is_windows():
    filepattern = library_name + ".lib"
  elif _is_macos():
    filepattern = "%s*.dylib" % (".".join(["lib" + library_name] +
                                          required_version.split(".")[:1]))
  else:
    filepattern = ".".join(["lib" + library_name, "so"] +
                           required_version.split(".")[:1]) + "*"
  return _find_file(base_paths, _library_paths(), filepattern)


def _find_versioned_file(base_paths, relative_paths, filepatterns,
                         required_version, get_version):
  # [FUNCTION DOC]
  # HELPER: finds a file whose detected version satisfies the requirement.
  # Inputs: base paths, relative subdirectories, one file pattern or a list/tuple of patterns,
  # the required version string (empty for any), and get_version, a callback taking a file path
  # and returning its version string or nothing. Tries every combination, evaluating matches in
  # order, and returns a pair of the first file whose version matches (prefix match) and that
  # version. Raises ConfigError naming the patterns and required version if none matches. The
  # callback may run subprocesses or read files.
  """Returns first valid path to a file that matches the requested version."""
  if type(filepatterns) not in [list, tuple]:
    filepatterns = [filepatterns]
  for path in _cartesian_product(base_paths, relative_paths):
    for filepattern in filepatterns:
      for file in glob.glob(os.path.join(path, filepattern)):
        actual_version = get_version(file)
        if _matches_version(actual_version, required_version):
          return file, actual_version
  raise _not_found_error(
      base_paths, relative_paths,
      ", ".join(filepatterns) + " matching version '%s'" % required_version)


def _find_header(base_paths, header_name, required_version, get_version):
  # [FUNCTION DOC]
  # HELPER: header specialization of the versioned file search. Inputs: base paths, a header file
  # name or collection of names, required version string and a version-reading callback. Searches
  # the fixed header subdirectories and returns a pair of the header path and its version; raises
  # ConfigError if no header matches.
  """Returns first valid path to a header that matches the requested version."""
  return _find_versioned_file(base_paths, _header_paths(), header_name,
                              required_version, get_version)


def _find_cuda_config(base_paths, required_version):
  # [FUNCTION DOC]
  # HELPER: CUDA toolkit discovery for find_cuda_config.
  # KNOWN ISSUE: the CUPTI library is searched using the user's required version rather than the
  # detected CUDA version, unlike the runtime library; and the nested header reader converts the
  # define to an integer directly, so a header lacking the define raises a value error instead
  # of returning nothing as its "if not version" branch intends.
  # Inputs: base_paths (list of directories) and required_version (string, empty for any). Finds
  # the CUDA header and derives major.minor, then the runtime library, the compiler (version read
  # from its version output, must match), the device library directory, CUPTI header and library
  # and the management library header. Raises ConfigError if anything is missing or if the
  # toolkit root derived from the compiler directory differs from that derived from the device
  # library directory. Returns a mapping with the CUDA version and include, library, binary,
  # device-library, CUPTI, toolkit root and NVML directories. Side effect: runs the compiler
  # with its version flag.

  def get_header_version(path):
    # [FUNCTION DOC]
    # HELPER: nested version reader for the CUDA header. Input: header path. Reads the version
    # number define (integer like 12040) and returns "major.minor" text; returns nothing when the
    # number is zero.
    version = int(_get_header_version(path, "CUDA_VERSION"))
    if not version:
      return None
    return "%d.%d" % (version // 1000, version % 1000 // 10)

  cuda_header_path, header_version = _find_header(base_paths, "cuda.h",
                                                  required_version,
                                                  get_header_version)
  cuda_version = header_version  # x.y, see above.

  cuda_library_path = _find_library(base_paths, "cudart", cuda_version)

  def get_nvcc_version(path):
    # [FUNCTION DOC]
    # HELPER: nested version reader for the CUDA compiler. Input: path of the compiler binary.
    # Runs it with the version flag, decodes each output line as ASCII and returns the full
    # three-part version from the "compilation tools, release" line; returns nothing if no line
    # matches. Raises a called-process error if the run fails.
    pattern = r"Cuda compilation tools, release \d+\.\d+, V(\d+\.\d+\.\d+)"
    for line in subprocess.check_output([path, "--version"]).splitlines():
      match = re.match(pattern, line.decode("ascii"))
      if match:
        return match.group(1)
    return None

  nvcc_name = "nvcc.exe" if _is_windows() else "nvcc"
  nvcc_path, nvcc_version = _find_versioned_file(base_paths, [
      "",
      "bin",
      "local/cuda/bin",
  ], nvcc_name, cuda_version, get_nvcc_version)

  nvvm_path = _find_file(base_paths, [
      "nvvm/libdevice",
      "share/cuda",
      "lib/nvidia-cuda-toolkit/libdevice",
      "local/cuda/nvvm/libdevice",
  ], "libdevice*.10.bc")

  cupti_header_path = _find_file(base_paths, _header_paths(), "cupti.h")
  nvml_header_dir = _find_file(base_paths, _header_paths(), "nvml.h")
  cupti_library_path = _find_library(base_paths, "cupti", required_version)

  cuda_binary_dir = os.path.dirname(nvcc_path)
  nvvm_library_dir = os.path.dirname(nvvm_path)

  # XLA requires the toolkit path to find ptxas and libdevice.
  # TODO(csigg): pass in both directories instead.
  cuda_toolkit_paths = (
      os.path.normpath(os.path.join(cuda_binary_dir, "..")),
      os.path.normpath(os.path.join(nvvm_library_dir, "../..")),
  )
  if cuda_toolkit_paths[0] != cuda_toolkit_paths[1]:
    raise ConfigError("Inconsistent CUDA toolkit path: %s vs %s" %
                      cuda_toolkit_paths)

  return {
      "cuda_version": cuda_version,
      "cuda_include_dir": os.path.dirname(cuda_header_path),
      "cuda_library_dir": os.path.dirname(cuda_library_path),
      "cuda_binary_dir": cuda_binary_dir,
      "nvvm_library_dir": nvvm_library_dir,
      "cupti_include_dir": os.path.dirname(cupti_header_path),
      "cupti_library_dir": os.path.dirname(cupti_library_path),
      "cuda_toolkit_path": cuda_toolkit_paths[0],
      "nvml_header_dir": os.path.dirname(nvml_header_dir),
  }


def _find_cublas_config(base_paths, required_version, cuda_version):
  # [FUNCTION DOC]
  # HELPER: cuBLAS discovery for find_cuda_config.
  # Inputs: base_paths (directories), required_version (string, empty for any) and cuda_version
  # (detected CUDA major.minor). For CUDA 10.1 or newer, finds the cuBLAS API header whose
  # major.minor.patch version (from three header defines) matches the requirement and uses only
  # the major part to find the library. For older CUDA there is no version info: just finds the
  # header and uses the required version for the library search. Returns a mapping with the
  # cuBLAS version, include directory and library directory. Raises ConfigError if not found.

  if _at_least_version(cuda_version, "10.1"):

    def get_header_version(path):
      # [FUNCTION DOC]
      # HELPER: nested version reader. Input: header path. Returns the dotted major.minor.patch
      # string built from the three version defines (missing defines yield empty parts).
      version = (
          _get_header_version(path, name)
          for name in ("CUBLAS_VER_MAJOR", "CUBLAS_VER_MINOR",
                       "CUBLAS_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "cublas_api.h",
                                               required_version,
                                               get_header_version)
    # cuBLAS uses the major version only.
    cublas_version = header_version.split(".")[0]

  else:
    # There is no version info available before CUDA 10.1, just find the file.
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "cublas_api.h")
    # cuBLAS version is the same as CUDA version (x.y).
    cublas_version = required_version

  library_path = _find_library(base_paths, "cublas", cublas_version)

  return {
      "cublas_version": header_version,
      "cublas_include_dir": os.path.dirname(header_path),
      "cublas_library_dir": os.path.dirname(library_path),
  }


def _find_cusolver_config(base_paths, required_version, cuda_version):
  # [FUNCTION DOC]
  # HELPER: cuSOLVER discovery for find_cuda_config.
  # Same structure as the cuBLAS discovery but the version-aware path applies for CUDA 11.0 or
  # newer (header: the common cuSOLVER header). Inputs: base paths, required version, detected
  # CUDA version. Returns a mapping with the cuSOLVER version, include directory and library
  # directory. Raises ConfigError if the header or library is not found.

  if _at_least_version(cuda_version, "11.0"):

    def get_header_version(path):
      # [FUNCTION DOC]
      # HELPER: nested version reader. Input: header path. Returns the dotted
      # major.minor.patch string built from the three cuSOLVER version defines.
      version = (
          _get_header_version(path, name)
          for name in ("CUSOLVER_VER_MAJOR", "CUSOLVER_VER_MINOR",
                       "CUSOLVER_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "cusolver_common.h",
                                               required_version,
                                               get_header_version)
    cusolver_version = header_version.split(".")[0]

  else:
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "cusolver_common.h")
    cusolver_version = required_version

  library_path = _find_library(base_paths, "cusolver", cusolver_version)

  return {
      "cusolver_version": header_version,
      "cusolver_include_dir": os.path.dirname(header_path),
      "cusolver_library_dir": os.path.dirname(library_path),
  }


def _find_curand_config(base_paths, required_version, cuda_version):
  # [FUNCTION DOC]
  # HELPER: cuRAND discovery for find_cuda_config.
  # Same structure as the cuSOLVER discovery, using the cuRAND header. Inputs: base paths,
  # required version, detected CUDA version. Returns a mapping with the cuRAND version, include
  # directory and library directory. Raises ConfigError if not found.

  if _at_least_version(cuda_version, "11.0"):

    def get_header_version(path):
      # [FUNCTION DOC]
      # HELPER: nested version reader. Input: header path. Returns the dotted
      # major.minor.patch string built from the three cuRAND version defines.
      version = (
          _get_header_version(path, name)
          for name in ("CURAND_VER_MAJOR", "CURAND_VER_MINOR",
                       "CURAND_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "curand.h",
                                               required_version,
                                               get_header_version)
    curand_version = header_version.split(".")[0]

  else:
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "curand.h")
    curand_version = required_version

  library_path = _find_library(base_paths, "curand", curand_version)

  return {
      "curand_version": header_version,
      "curand_include_dir": os.path.dirname(header_path),
      "curand_library_dir": os.path.dirname(library_path),
  }


def _find_cufft_config(base_paths, required_version, cuda_version):
  # [FUNCTION DOC]
  # HELPER: cuFFT discovery for find_cuda_config.
  # Same structure as the cuSOLVER discovery, using the cuFFT header. Inputs: base paths,
  # required version, detected CUDA version. Returns a mapping with the cuFFT version, include
  # directory and library directory. Raises ConfigError if not found.

  if _at_least_version(cuda_version, "11.0"):

    def get_header_version(path):
      # [FUNCTION DOC]
      # HELPER: nested version reader. Input: header path. Returns the dotted
      # major.minor.patch string built from the three cuFFT version defines.
      version = (
          _get_header_version(path, name)
          for name in ("CUFFT_VER_MAJOR", "CUFFT_VER_MINOR", "CUFFT_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "cufft.h",
                                               required_version,
                                               get_header_version)
    cufft_version = header_version.split(".")[0]

  else:
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "cufft.h")
    cufft_version = required_version

  library_path = _find_library(base_paths, "cufft", cufft_version)

  return {
      "cufft_version": header_version,
      "cufft_include_dir": os.path.dirname(header_path),
      "cufft_library_dir": os.path.dirname(library_path),
  }


def _find_cudnn_config(base_paths, required_version):
  # [FUNCTION DOC]
  # HELPER: cuDNN discovery for find_cuda_config.
  # KNOWN ISSUE: the returned version entry holds only the major version, while the other
  # libraries (cuBLAS and similar) report the full version from the header.
  # Inputs: base paths and required version string. Searches for either of two cuDNN header
  # names whose major.minor.patchlevel version matches, then the library for the major version.
  # Returns a mapping with the cuDNN major version, include directory and library directory.
  # Raises ConfigError if not found.

  def get_header_version(path):
    # [FUNCTION DOC]
    # HELPER: nested version reader. Input: header path. Returns the dotted version string from
    # the major, minor and patch level defines, or nothing if the major define is missing.
    version = [
        _get_header_version(path, name)
        for name in ("CUDNN_MAJOR", "CUDNN_MINOR", "CUDNN_PATCHLEVEL")]
    return ".".join(version) if version[0] else None

  header_path, header_version = _find_header(base_paths,
                                             ("cudnn.h", "cudnn_version.h"),
                                             required_version,
                                             get_header_version)
  cudnn_version = header_version.split(".")[0]

  library_path = _find_library(base_paths, "cudnn", cudnn_version)

  return {
      "cudnn_version": cudnn_version,
      "cudnn_include_dir": os.path.dirname(header_path),
      "cudnn_library_dir": os.path.dirname(library_path),
  }


def _find_cusparse_config(base_paths, required_version, cuda_version):
  # [FUNCTION DOC]
  # HELPER: cuSPARSE discovery for find_cuda_config.
  # Same structure as the cuSOLVER discovery, using the cuSPARSE header. Inputs: base paths,
  # required version, detected CUDA version. Returns a mapping with the cuSPARSE version, include
  # directory and library directory. Raises ConfigError if not found.

  if _at_least_version(cuda_version, "11.0"):

    def get_header_version(path):
      # [FUNCTION DOC]
      # HELPER: nested version reader. Input: header path. Returns the dotted
      # major.minor.patch string built from the three cuSPARSE version defines.
      version = (
          _get_header_version(path, name)
          for name in ("CUSPARSE_VER_MAJOR", "CUSPARSE_VER_MINOR",
                       "CUSPARSE_VER_PATCH"))
      return ".".join(version)

    header_path, header_version = _find_header(base_paths, "cusparse.h",
                                               required_version,
                                               get_header_version)
    cusparse_version = header_version.split(".")[0]

  else:
    header_version = cuda_version
    header_path = _find_file(base_paths, _header_paths(), "cusparse.h")
    cusparse_version = required_version

  library_path = _find_library(base_paths, "cusparse", cusparse_version)

  return {
      "cusparse_version": header_version,
      "cusparse_include_dir": os.path.dirname(header_path),
      "cusparse_library_dir": os.path.dirname(library_path),
  }


def _find_nccl_config(base_paths, required_version):
  # [FUNCTION DOC]
  # HELPER: NCCL discovery for find_cuda_config.
  # KNOWN ISSUE: the returned version entry holds only the major version.
  # Inputs: base paths and required version string. Finds the NCCL header whose
  # major.minor.patch version matches, then the library for the major version. Returns a mapping
  # with the NCCL major version, include directory and library directory. Raises ConfigError if
  # not found.

  def get_header_version(path):
    # [FUNCTION DOC]
    # HELPER: nested version reader. Input: header path. Returns the dotted version string
    # from the three NCCL version defines.
    version = (
        _get_header_version(path, name)
        for name in ("NCCL_MAJOR", "NCCL_MINOR", "NCCL_PATCH"))
    return ".".join(version)

  header_path, header_version = _find_header(base_paths, "nccl.h",
                                             required_version,
                                             get_header_version)
  nccl_version = header_version.split(".")[0]

  library_path = _find_library(base_paths, "nccl", nccl_version)

  return {
      "nccl_version": nccl_version,
      "nccl_include_dir": os.path.dirname(header_path),
      "nccl_library_dir": os.path.dirname(library_path),
  }


def _find_tensorrt_config(base_paths, required_version):
  # [FUNCTION DOC]
  # HELPER: TensorRT discovery for find_cuda_config.
  # Inputs: base paths and required version string. Finds the version header whose
  # major.minor.patch version matches, then the inference library for the major version. Returns
  # a mapping with the full TensorRT version, include directory and library directory. Raises
  # ConfigError if not found.

  def get_header_version(path):
    # [FUNCTION DOC]
    # HELPER: nested version reader. Input: header path. Reads the three version defines and
    # returns the dotted string, or nothing if any part is missing or empty so the version
    # matching step rejects the file.
    version = (
        _get_header_version(path, name)
        for name in ("NV_TENSORRT_MAJOR", "NV_TENSORRT_MINOR",
                     "NV_TENSORRT_PATCH"))
    # `version` is a generator object, so we convert it to a list before using
    # it (muitiple times below).
    version = list(version)
    if not all(version):
      return None  # Versions not found, make _matches_version returns False.
    return ".".join(version)

  header_path, header_version = _find_header(base_paths, "NvInferVersion.h",
                                             required_version,
                                             get_header_version)

  tensorrt_version = header_version.split(".")[0]
  library_path = _find_library(base_paths, "nvinfer", tensorrt_version)

  return {
      "tensorrt_version": header_version,
      "tensorrt_include_dir": os.path.dirname(header_path),
      "tensorrt_library_dir": os.path.dirname(library_path),
  }


def _list_from_env(env_name, default=[]):
  # [FUNCTION DOC]
  # HELPER: environment parsing. KNOWN ISSUE: the default argument is a shared mutable list that
  # is returned as is (not copied), so callers mutating it would affect later calls.
  # Inputs: env_name (environment variable name) and default (list returned when unset). Returns
  # the variable's value split on commas if the variable is set (even if empty, yielding a list
  # with one empty string), otherwise the default.
  """Returns comma-separated list from environment variable."""
  if env_name in os.environ:
    return os.environ[env_name].split(",")
  return default


def _get_legacy_path(env_name, default=[]):
  # [FUNCTION DOC]
  # HELPER: handles legacy install path variables. KNOWN ISSUE: shared mutable default list, as
  # in the list-from-environment helper.
  # Inputs: env_name and default list. If the variable is set to a library directory under an
  # architecture-specific lib directory of a system prefix, returns a one-element list with that
  # prefix (the part before the lib directory), since such a setting once located both headers
  # and libraries. Otherwise returns the comma-split list or default via the list helper.
  """Returns a path specified by a legacy environment variable.

  CUDNN_INSTALL_PATH, NCCL_INSTALL_PATH, TENSORRT_INSTALL_PATH set to
  '/usr/lib/x86_64-linux-gnu' would previously find both library and header
  paths. Detect those and return '/usr', otherwise forward to _list_from_env().
  """
  if env_name in os.environ:
    match = re.match(r"^(/[^/ ]*)+/lib/\w+-linux-gnu/?$", os.environ[env_name])
    if match:
      return [match.group(1)]
  return _list_from_env(env_name, default)


def _normalize_path(path):
  # [FUNCTION DOC]
  # HELPER: path cleanup for the final output. Input: a path string. Resolves symbolic links to
  # an absolute path and, on Windows, replaces backslashes with forward slashes. Returns the
  # result.
  """Returns normalized path, with forward slashes on Windows."""
  path = os.path.realpath(path)
  if _is_windows():
    path = path.replace("\\", "/")
  return path


def find_cuda_config():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring calls the result a dictionary of library and header paths, but it
  # also contains version strings.
  # Main discovery routine. Inputs: none directly; library names come from the process command
  # line (lower-cased: cuda, cudnn, nccl, tensorrt), and environment variables supply the
  # requested CUDA version, base paths (comma list; default OS specific list; non-existing
  # directories dropped), CUDA toolkit path override, per-library versions and legacy install
  # paths. For cuda, finds the toolkit and then cuBLAS, cuSOLVER, cuRAND, cuFFT and cuSPARSE
  # (for CUDA older than 10.1 or 11.0, as applicable, those search the toolkit paths instead of
  # the base paths). Then optionally cuDNN, NCCL and TensorRT. Finally normalizes every entry
  # whose key ends with a directory or path suffix. Returns the combined mapping of names to
  # strings. Raises ConfigError if a requested library cannot be found. Side effects: reads
  # environment and file system, runs subprocesses.
  """Returns a dictionary of CUDA library and header file paths."""
  libraries = [argv.lower() for argv in sys.argv[1:]]
  cuda_version = os.environ.get("TF_CUDA_VERSION", "")
  base_paths = _list_from_env("TF_CUDA_PATHS",
                              _get_default_cuda_paths(cuda_version))
  base_paths = [path for path in base_paths if os.path.exists(path)]

  result = {}
  if "cuda" in libraries:
    cuda_paths = _list_from_env("CUDA_TOOLKIT_PATH", base_paths)
    res = _find_cuda_config(cuda_paths, cuda_version)

    result.update(res)

    cuda_version = result["cuda_version"]
    cublas_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (10, 1):
      # Before CUDA 10.1, cuBLAS was in the same directory as the toolkit.
      cublas_paths = cuda_paths
    cublas_version = os.environ.get("TF_CUBLAS_VERSION", "")
    result.update(
        _find_cublas_config(cublas_paths, cublas_version, cuda_version))

    cusolver_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (11, 0):
      cusolver_paths = cuda_paths
    cusolver_version = os.environ.get("TF_CUSOLVER_VERSION", "")
    result.update(
        _find_cusolver_config(cusolver_paths, cusolver_version, cuda_version))

    curand_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (11, 0):
      curand_paths = cuda_paths
    curand_version = os.environ.get("TF_CURAND_VERSION", "")
    result.update(
        _find_curand_config(curand_paths, curand_version, cuda_version))

    cufft_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (11, 0):
      cufft_paths = cuda_paths
    cufft_version = os.environ.get("TF_CUFFT_VERSION", "")
    result.update(_find_cufft_config(cufft_paths, cufft_version, cuda_version))

    cusparse_paths = base_paths
    if tuple(int(v) for v in cuda_version.split(".")) < (11, 0):
      cusparse_paths = cuda_paths
    cusparse_version = os.environ.get("TF_CUSPARSE_VERSION", "")
    result.update(
        _find_cusparse_config(cusparse_paths, cusparse_version, cuda_version))

  if "cudnn" in libraries:
    cudnn_paths = _get_legacy_path("CUDNN_INSTALL_PATH", base_paths)
    cudnn_version = os.environ.get("TF_CUDNN_VERSION", "")
    result.update(_find_cudnn_config(cudnn_paths, cudnn_version))

  if "nccl" in libraries:
    nccl_paths = _get_legacy_path("NCCL_INSTALL_PATH", base_paths)
    nccl_version = os.environ.get("TF_NCCL_VERSION", "")
    result.update(_find_nccl_config(nccl_paths, nccl_version))

  if "tensorrt" in libraries:
    tensorrt_paths = _get_legacy_path("TENSORRT_INSTALL_PATH", base_paths)
    tensorrt_version = os.environ.get("TF_TENSORRT_VERSION", "")
    result.update(_find_tensorrt_config(tensorrt_paths, tensorrt_version))

  for k, v in result.items():
    if k.endswith("_dir") or k.endswith("_path"):
      result[k] = _normalize_path(v)

  return result


def main():
  # [FUNCTION DOC]
  # Script entry point. Runs the discovery and prints each result as a key, colon, value line to
  # standard output, sorted by key. On a ConfigError writes the message and a newline to standard
  # error and exits with status 1. Returns nothing otherwise.
  try:
    for key, value in sorted(find_cuda_config().items()):
      print("%s: %s" % (key, value))
  except ConfigError as e:
    sys.stderr.write(str(e) + '\n')
    sys.exit(1)


if __name__ == "__main__":
  main()
