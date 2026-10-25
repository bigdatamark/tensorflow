# [MODULE DOC]
# HELPER: command-line helper script used by the ROCm repository rule of the GPU build setup.
# Discovers the ROCm installation on the build machine and reports the version numbers of its
# components. The install directory comes from the ROCM_PATH environment variable, with a
# default of the standard system ROCm location. For each component (ROCm core, HIP runtime,
# MIOpen, rocBLAS, rocRAND, rocFFT, hipFFT, roctracer, hipSPARSE, hipSOLVER, rocSOLVER) it
# locates a version header under one of two known relative locations (newer layout first, older
# layout second), reads the integer version macros from it and combines them into a single
# number. Major/minor/patch triples become major*10000 + minor*100 + patch; the HIP runtime
# uses major*100 + minor. hipFFT is only probed for ROCm 4.1 or newer and hipSOLVER only for
# ROCm 4.5 or newer. The result is printed as sorted "key: value" lines on standard output.
# On any problem it writes an error message to standard error and exits with status 1.
# Defines the ConfigError exception. No import-time side effects; main runs only when the file
# is executed as a script.
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
"""Prints ROCm library and header directories and versions found on the system.

The script searches for ROCm library and header files on the system, inspects
them to determine their version and prints the configuration to stdout.
The path to inspect is specified through an environment variable (ROCM_PATH).
If no valid configuration is found, the script prints to stderr and
returns an error code.

The script takes the directory specified by the ROCM_PATH environment variable.
The script looks for headers and library files in a hard-coded set of
subdirectories from base path of the specified directory. If ROCM_PATH is not
specified, then "/opt/rocm" is used as it default value

"""

import io
import os
import re
import sys


class ConfigError(Exception):
  # [CLASS DOC]
  # Exception type raised when the ROCm installation or one of its version headers cannot be
  # found or parsed. Carries only a human-readable message; the main entry point catches it,
  # prints the message to standard error and exits with a failure status.
  pass


def _get_default_rocm_path():
  # [FUNCTION DOC]
  # HELPER: supplies the fallback ROCm location for the install-path lookup.
  # Takes no inputs. Returns the fixed default ROCm installation directory as a string. No
  # side effects and no exceptions.
  return "/opt/rocm"


def _get_rocm_install_path():
  # [FUNCTION DOC]
  # HELPER: resolves the base directory that all ROCm component lookups start from.
  # Takes no inputs. Returns the value of the ROCM_PATH environment variable if it is set,
  # otherwise the default ROCm directory. The path is returned as given (it is not resolved
  # to a real path and its existence is not checked here). Reads the environment only.
  """Determines and returns the ROCm installation path."""
  rocm_install_path = _get_default_rocm_path()
  if "ROCM_PATH" in os.environ:
    rocm_install_path = os.environ["ROCM_PATH"]
  # rocm_install_path = os.path.realpath(rocm_install_path)
  return rocm_install_path


def _get_composite_version_number(major, minor, patch):
  # [FUNCTION DOC]
  # HELPER: packs a version triple into one comparable integer for the config output.
  # Inputs: major, minor and patch version numbers (integers). Returns major times 10000 plus
  # minor times 100 plus patch. Pure function, no side effects.
  return 10000 * major + 100 * minor + patch


def _get_header_version(path, name):
  # [FUNCTION DOC]
  # HELPER: shared version-header parser used by every component lookup.
  # KNOWN ISSUE: the macro name is inserted into a regular expression without escaping, and the
  # docstring says it returns "defines" in general while only integer-literal values are
  # supported. The file is opened with io.open and never explicitly closed.
  # Inputs: path to a C header file and the name of a preprocessor macro. Scans the file line
  # by line and returns, as an integer, the value of the first line that starts with a define
  # of that macro followed by spaces and decimal digits. Raises ConfigError when no such line
  # exists (macro missing or value not an integer literal). Raises the usual file errors if
  # the file cannot be opened. Read-only.
  """Returns preprocessor defines in C header file."""
  for line in io.open(path, "r", encoding="utf-8"):
    match = re.match(r"#define %s +(\d+)" % name, line)
    if match:
      value = match.group(1)
      return int(value)

  raise ConfigError('#define "{}" is either\n'.format(name) +
                    "  not present in file {} OR\n".format(path) +
                    "  its value is not an integer literal")


def _find_rocm_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for the ROCm core version, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the ROCm core version
  # as a composite integer. Raises ConfigError if no ROCm version header is found or a macro is
  # missing. Read-only.

  def rocm_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the ROCm core version header.
    # Input: installation directory. Tries the newer rocm-core header location first, then the
    # older one, and uses the first that exists. Returns the major, minor and patch numbers as
    # a tuple. Raises ConfigError if neither file exists or a version macro is absent.
    possible_version_files = [
        "include/rocm-core/rocm_version.h",  # ROCm 5.2
        "include/rocm_version.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "ROCm version file not found in {}".format(possible_version_files))

    major = _get_header_version(version_file, "ROCM_VERSION_MAJOR")
    minor = _get_header_version(version_file, "ROCM_VERSION_MINOR")
    patch = _get_header_version(version_file, "ROCM_VERSION_PATCH")
    return major, minor, patch

  major, minor, patch = rocm_version_numbers(rocm_install_path)

  rocm_config = {
      "rocm_version_number": _get_composite_version_number(major, minor, patch)
  }

  return rocm_config


def _find_hipruntime_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for the HIP runtime version, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the HIP runtime
  # version as major times 100 plus minor. Raises ConfigError if the header or a macro is
  # missing. Read-only.

  def hipruntime_version_number(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the HIP version header.
    # Input: installation directory. Checks the newer then the older header location, reads
    # the major and minor macros (the patch is ignored, and the header's own combined macro is
    # deliberately not parsed). Returns major times 100 plus minor. Raises ConfigError if the
    # file or a macro is missing.
    possible_version_files = [
        "include/hip/hip_version.h",  # ROCm 5.2
        "hip/include/hip/hip_version.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError("HIP Runtime version file not found in {}".format(
          possible_version_files))

    # This header file has an explicit #define for HIP_VERSION, whose value
    # is (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR)
    # Retreive the major + minor and re-calculate here, since we do not
    # want get into the business of parsing arith exprs
    major = _get_header_version(version_file, "HIP_VERSION_MAJOR")
    minor = _get_header_version(version_file, "HIP_VERSION_MINOR")
    return 100 * major + minor

  hipruntime_config = {
      "hipruntime_version_number": hipruntime_version_number(rocm_install_path)
  }

  return hipruntime_config


def _find_miopen_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for MIOpen, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the MIOpen version as
  # a composite integer. Raises ConfigError if the header or a macro is missing. Read-only.

  def miopen_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the MIOpen version header.
    # KNOWN ISSUE: when no candidate file exists, the error message formats the variable that
    # holds the chosen file, which is still the nothing value, instead of listing the
    # candidate paths, so the message is unhelpful.
    # Input: installation directory. Checks the newer then the older header location. Returns
    # the major, minor and patch numbers as a tuple. Raises ConfigError if no file exists or a
    # macro is absent.
    possible_version_files = [
        "include/miopen/version.h",  # ROCm 5.2 and prior
        "miopen/include/miopen/version.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          'MIOpen version file "{}" not found'.format(version_file))
    major = _get_header_version(version_file, "MIOPEN_VERSION_MAJOR")
    minor = _get_header_version(version_file, "MIOPEN_VERSION_MINOR")
    patch = _get_header_version(version_file, "MIOPEN_VERSION_PATCH")
    return major, minor, patch

  major, minor, patch = miopen_version_numbers(rocm_install_path)

  miopen_config = {
      "miopen_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return miopen_config


def _find_rocblas_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for rocBLAS, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the rocBLAS version as
  # a composite integer. Raises ConfigError if the header or a macro is missing. Read-only.

  def rocblas_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the rocBLAS version header.
    # Input: installation directory. Checks the newer then the older internal header location.
    # Returns the major, minor and patch numbers as a tuple. Raises ConfigError if no file
    # exists or a macro is absent.
    possible_version_files = [
        "include/rocblas/internal/rocblas-version.h",  # ROCm 5.2
        "rocblas/include/internal/rocblas-version.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "rocblas version file not found in {}".format(
              possible_version_files))
    major = _get_header_version(version_file, "ROCBLAS_VERSION_MAJOR")
    minor = _get_header_version(version_file, "ROCBLAS_VERSION_MINOR")
    patch = _get_header_version(version_file, "ROCBLAS_VERSION_PATCH")
    return major, minor, patch

  major, minor, patch = rocblas_version_numbers(rocm_install_path)

  rocblas_config = {
      "rocblas_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return rocblas_config


def _find_rocrand_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for rocRAND, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the rocRAND version
  # number exactly as defined in its header (not repacked). Raises ConfigError if the header or
  # macro is missing. Read-only.

  def rocrand_version_number(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the rocRAND version header.
    # Input: installation directory. Checks the newer then the older header location and reads
    # the single combined version macro. Returns it as an integer. Raises ConfigError if no
    # file exists or the macro is absent.
    possible_version_files = [
        "include/rocrand/rocrand_version.h",  # ROCm 5.1
        "rocrand/include/rocrand_version.h",  # ROCm 5.0 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "rocrand version file not found in {}".format(possible_version_files))
    version_number = _get_header_version(version_file, "ROCRAND_VERSION")
    return version_number

  rocrand_config = {
      "rocrand_version_number": rocrand_version_number(rocm_install_path)
  }

  return rocrand_config


def _find_rocfft_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for rocFFT, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the rocFFT version as
  # a composite integer. Raises ConfigError if the header or a macro is missing. Read-only.

  def rocfft_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the rocFFT version header.
    # Input: installation directory. Checks the newer then the older header location. Returns
    # the major, minor and patch numbers as a tuple (macro names are lower case in this
    # header). Raises ConfigError if no file exists or a macro is absent.
    possible_version_files = [
        "include/rocfft/rocfft-version.h",  # ROCm 5.2
        "rocfft/include/rocfft-version.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "rocfft version file not found in {}".format(possible_version_files))
    major = _get_header_version(version_file, "rocfft_version_major")
    minor = _get_header_version(version_file, "rocfft_version_minor")
    patch = _get_header_version(version_file, "rocfft_version_patch")
    return major, minor, patch

  major, minor, patch = rocfft_version_numbers(rocm_install_path)

  rocfft_config = {
      "rocfft_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return rocfft_config


def _find_hipfft_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for hipFFT, used by the main config function (only for ROCm 4.1+).
  # Input: ROCm installation directory. Returns a mapping with one entry, the hipFFT version as
  # a composite integer. Raises ConfigError if the header or a macro is missing. Read-only.

  def hipfft_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the hipFFT version header.
    # Input: installation directory. Checks the newer then the older header location. Returns
    # the major, minor and patch numbers as a tuple. Raises ConfigError if no file exists or a
    # macro is absent.
    possible_version_files = [
        "include/hipfft/hipfft-version.h",  # ROCm 5.2
        "hipfft/include/hipfft-version.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "hipfft version file not found in {}".format(possible_version_files))
    major = _get_header_version(version_file, "hipfftVersionMajor")
    minor = _get_header_version(version_file, "hipfftVersionMinor")
    patch = _get_header_version(version_file, "hipfftVersionPatch")
    return major, minor, patch

  major, minor, patch = hipfft_version_numbers(rocm_install_path)

  hipfft_config = {
      "hipfft_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return hipfft_config


def _find_roctracer_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for roctracer, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the roctracer version
  # as a composite integer (patch always zero). Raises ConfigError if the header or a macro is
  # missing. Read-only.

  def roctracer_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the roctracer header.
    # Input: installation directory. Checks the newer then the older header location. Reads
    # major and minor macros; the header has no patch macro so patch is fixed at zero. Returns
    # a tuple of major, minor, patch. Raises ConfigError if no file exists or a macro is absent.
    possible_version_files = [
        "include/roctracer/roctracer.h",  # ROCm 5.2
        "roctracer/include/roctracer.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError("roctracer version file not found in {}".format(
          possible_version_files))
    major = _get_header_version(version_file, "ROCTRACER_VERSION_MAJOR")
    minor = _get_header_version(version_file, "ROCTRACER_VERSION_MINOR")
    # roctracer header does not have a patch version number
    patch = 0
    return major, minor, patch

  major, minor, patch = roctracer_version_numbers(rocm_install_path)

  roctracer_config = {
      "roctracer_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return roctracer_config


def _find_hipsparse_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for hipSPARSE, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the hipSPARSE version
  # as a composite integer. Raises ConfigError if the header or a macro is missing. Read-only.

  def hipsparse_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the hipSPARSE version header.
    # Input: installation directory. Checks the newer then the older header location. Returns
    # the major, minor and patch numbers as a tuple. Raises ConfigError if no file exists or a
    # macro is absent.
    possible_version_files = [
        "include/hipsparse/hipsparse-version.h",  # ROCm 5.2
        "hipsparse/include/hipsparse-version.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError("hipsparse version file not found in {}".format(
          possible_version_files))
    major = _get_header_version(version_file, "hipsparseVersionMajor")
    minor = _get_header_version(version_file, "hipsparseVersionMinor")
    patch = _get_header_version(version_file, "hipsparseVersionPatch")
    return major, minor, patch

  major, minor, patch = hipsparse_version_numbers(rocm_install_path)

  hipsparse_config = {
      "hipsparse_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return hipsparse_config

def _find_hipsolver_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for hipSOLVER, used by the main config function (only for ROCm
  # 4.5+).
  # Input: ROCm installation directory. Returns a mapping with one entry, the hipSOLVER version
  # as a composite integer. Raises ConfigError if the header or a macro is missing. Read-only.

  def hipsolver_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the hipSOLVER version header.
    # Input: installation directory. Checks three candidate header locations in order (newest
    # layout first). Returns the major, minor and patch numbers as a tuple. Raises ConfigError
    # if no file exists or a macro is absent.
    possible_version_files = [
        "include/hipsolver/internal/hipsolver-version.h",  # ROCm 5.2
        "hipsolver/include/internal/hipsolver-version.h",  # ROCm 5.1
        "hipsolver/include/hipsolver-version.h",  # ROCm 5.0 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError("hipsolver version file not found in {}".format(
          possible_version_files))
    major = _get_header_version(version_file, "hipsolverVersionMajor")
    minor = _get_header_version(version_file, "hipsolverVersionMinor")
    patch = _get_header_version(version_file, "hipsolverVersionPatch")
    return major, minor, patch

  major, minor, patch = hipsolver_version_numbers(rocm_install_path)

  hipsolver_config = {
      "hipsolver_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return hipsolver_config


def _find_rocsolver_config(rocm_install_path):
  # [FUNCTION DOC]
  # HELPER: component probe for rocSOLVER, used by the main config function.
  # Input: ROCm installation directory. Returns a mapping with one entry, the rocSOLVER version
  # as a composite integer. Raises ConfigError if the header or a macro is missing. Read-only.

  def rocsolver_version_numbers(path):
    # [FUNCTION DOC]
    # HELPER: nested locator/parser for the rocSOLVER version header.
    # Input: installation directory. Checks the newer then the older header location. Returns
    # the major, minor and patch numbers as a tuple. Raises ConfigError if no file exists or a
    # macro is absent.
    possible_version_files = [
        "include/rocsolver/rocsolver-version.h",  # ROCm 5.2
        "rocsolver/include/rocsolver-version.h",  # ROCm 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError("rocsolver version file not found in {}".format(
          possible_version_files))
    major = _get_header_version(version_file, "ROCSOLVER_VERSION_MAJOR")
    minor = _get_header_version(version_file, "ROCSOLVER_VERSION_MINOR")
    patch = _get_header_version(version_file, "ROCSOLVER_VERSION_PATCH")
    return major, minor, patch

  major, minor, patch = rocsolver_version_numbers(rocm_install_path)

  rocsolver_config = {
      "rocsolver_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return rocsolver_config


def find_rocm_config():
  # [FUNCTION DOC]
  # Public entry point that gathers the whole ROCm configuration.
  # Takes no inputs; uses the ROCM_PATH environment variable (or the default location).
  # Returns a mapping from setting names to values: the toolkit path plus a version number for
  # ROCm, HIP runtime, MIOpen, rocBLAS, rocRAND, rocFFT, roctracer, hipSPARSE and rocSOLVER,
  # and additionally hipFFT (ROCm 4.1+) and hipSOLVER (ROCm 4.5+). Raises ConfigError if the
  # install directory does not exist or any component probe fails. Read-only.
  """Returns a dictionary of ROCm components config info."""
  rocm_install_path = _get_rocm_install_path()
  if not os.path.exists(rocm_install_path):
    raise ConfigError(
        'Specified ROCM_PATH "{}" does not exist'.format(rocm_install_path))

  result = {}

  result["rocm_toolkit_path"] = rocm_install_path
  result.update(_find_rocm_config(rocm_install_path))
  result.update(_find_hipruntime_config(rocm_install_path))
  result.update(_find_miopen_config(rocm_install_path))
  result.update(_find_rocblas_config(rocm_install_path))
  result.update(_find_rocrand_config(rocm_install_path))
  result.update(_find_rocfft_config(rocm_install_path))
  if result["rocm_version_number"] >= 40100:
    result.update(_find_hipfft_config(rocm_install_path))
  result.update(_find_roctracer_config(rocm_install_path))
  result.update(_find_hipsparse_config(rocm_install_path))
  if result["rocm_version_number"] >= 40500:
    result.update(_find_hipsolver_config(rocm_install_path))
  result.update(_find_rocsolver_config(rocm_install_path))

  return result


def main():
  # [FUNCTION DOC]
  # Command-line entry point. Takes no arguments. Prints each discovered setting as a name and
  # value line on standard output, sorted by name. If a ConfigError occurs it writes an error
  # message to standard error and terminates the process with exit status 1.
  try:
    for key, value in sorted(find_rocm_config().items()):
      print("%s: %s" % (key, value))
  except ConfigError as e:
    sys.stderr.write("\nERROR: {}\n\n".format(str(e)))
    sys.exit(1)


if __name__ == "__main__":
  main()
