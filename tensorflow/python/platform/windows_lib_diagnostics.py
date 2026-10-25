# [MODULE DOC]
# HELPER: Diagnostic helper for troubleshooting failures to load TensorFlow native libraries on Windows.
# Standalone utility that walks the import tables of a Windows PE binary (by default the TensorFlow
# internal pywrap extension) and tries to load each dependency, printing a human-readable
# explanation for every failure (module not found, bad format, missing procedure, access denied,
# initialization failure). It exposes the Windows error-code constants, a PE header parser, a
# recursive loader-checker and a top-level entry point. It deliberately avoids importing
# tensorflow so it can run when the package itself fails to import. It can be run as a script with
# an optional path argument. No import-time side effects other than standard library imports.
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
"""Utility to diagnose Windows DLL loading issues."""

import ctypes
import os
import struct
import sys

# Windows Error Codes
ERROR_ACCESS_DENIED = 0x05
ERROR_PROC_NOT_FOUND = 0x7F
ERROR_MOD_NOT_FOUND = 0x7E
ERROR_BAD_EXE_FORMAT = 0xC1
ERROR_DLL_INIT_FAILED = 0x45A


def get_dll_dependencies(path):
  # [FUNCTION DOC]
  # HELPER: Supplies the dependency list used by the recursive Windows DLL load diagnosis.
  # Reads the binary at the given path and parses its PE header by hand: checks the MZ and PE
  # signatures, reads the section count and optional header (supporting both 32-bit and 64-bit
  # layouts), locates the import directory, maps relative virtual addresses to file offsets using
  # the section table, and collects the ASCII names of every imported DLL.
  # Input: path, a string path to an executable or library file.
  # Output: a list of DLL file name strings; an empty list if the file is not a valid PE file, has
  # no imports, or any file/parse/decoding error occurs (those errors are silently swallowed).
  # Side effects: opens and reads the file only.
  """Parses PE header to find direct DLL dependencies."""
  deps = []
  try:
    with open(path, "rb") as f:
      if f.read(2) != b"MZ":
        return []
      f.seek(0x3C)
      header_offset_raw = f.read(4)
      if not header_offset_raw:
        return []
      pe_offset = struct.unpack("<I", header_offset_raw)[0]
      f.seek(pe_offset)
      if f.read(4) != b"PE\0\0":
        return []
      f.seek(2, 1)  # machine
      num_sections = struct.unpack("<H", f.read(2))[0]
      f.seek(12, 1)  # stamp etc.
      size_of_optional_header = struct.unpack("<H", f.read(2))[0]
      f.seek(2, 1)  # characteristics
      optional_header_start = f.tell()
      magic_raw = f.read(2)
      if not magic_raw:
        return []
      magic = struct.unpack("<H", magic_raw)[0]
      if magic == 0x20B:  # PE32+
        import_dir_entry_offset = optional_header_start + 120
      elif magic == 0x10B:  # PE32
        import_dir_entry_offset = optional_header_start + 104
      else:
        return []
      f.seek(import_dir_entry_offset)
      import_rva = struct.unpack("<I", f.read(4))[0]
      if import_rva == 0:
        return []
      f.seek(optional_header_start + size_of_optional_header)
      sections = []
      for _ in range(num_sections):
        _ = f.read(8)
        vs = struct.unpack("<I", f.read(4))[0]
        va = struct.unpack("<I", f.read(4))[0]
        _ = struct.unpack("<I", f.read(4))[0]
        rp = struct.unpack("<I", f.read(4))[0]
        f.seek(16, 1)
        sections.append((va, vs, rp))

      def rva_to_offset(rva):
        # [FUNCTION DOC]
        # HELPER: Address translation for the PE import-table parser.
        # Converts a relative virtual address (integer) into a file offset using the section list
        # captured from the enclosing function: finds the section whose virtual range contains the
        # address and returns the raw data pointer plus the offset inside the section.
        # Returns nothing (null value) if no section contains the address. No side effects.
        for va, vs, ptr in sections:
          if va <= rva < va + vs:
            return ptr + (rva - va)
        return None

      import_offset = rva_to_offset(import_rva)
      if import_offset is None:
        return []
      f.seek(import_offset)
      while True:
        entry = f.read(20)
        if not entry or len(entry) < 20:
          break
        name_rva = struct.unpack("<I", entry[12:16])[0]
        if name_rva == 0:
          break
        current_pos = f.tell()
        name_offset = rva_to_offset(name_rva)
        if name_offset:
          f.seek(name_offset)
          name = b""
          while True:
            char = f.read(1)
            if char == b"\0" or not char:
              break
            name += char
          deps.append(name.decode("ascii"))
        f.seek(current_pos)
  except (IOError, struct.error, UnicodeDecodeError):
    # Catch specific exceptions that can occur during file reading and parsing.
    pass
  return deps


def diagnose_dll_load(path, depth=0, visited=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: relies on the PATH environment variable key existing; a missing PATH raises a
  # lookup error that is not caught here. Also the Windows-only loader call raises an attribute
  # error on non-Windows platforms.
  # Core of the diagnostic: depth-first, leaf-first check of a binary's DLL dependencies.
  # Inputs: path, string path of the binary to analyze; depth, integer recursion level used for
  # output indentation (default 0; level 0 prints a header line); visited, optional set of
  # lower-cased names/paths already handled (created fresh when omitted) used to avoid cycles
  # and duplicate work.
  # Behavior: returns immediately if already visited, the file does not exist (assumed system
  # DLL) or it has no dependencies. Otherwise registers the file's directory as an extra DLL
  # search directory (when the operating system supports it, ignoring failures), then for each
  # dependency recurses into a matching file found in that directory or on the PATH environment
  # variable (skipping well-known system DLL prefixes), and afterwards attempts to load the
  # dependency through the Windows loader. Load failures are printed with a specific explanation
  # for known error codes (and hints for initialization failures) or a generic unknown-error
  # message, and the dependency is marked visited.
  # Returns nothing. Side effects: printing to standard output, mutating the visited set, adding
  # a DLL search directory, loading libraries into the process.
  """Recursively checks DLL dependencies and attempts loading them."""
  if visited is None:
    visited = set()
  if path.lower() in visited:
    return
  visited.add(path.lower())

  indent = "  " * depth
  if depth == 0:
    print(f"\n[TensorFlow DLL Diagnostic] Analyzing: {path}")

  # If file doesn't exist, it might be a system DLL
  if not os.path.exists(path):
    return

  deps = get_dll_dependencies(path)
  if not deps:
    return

  target_dir = os.path.dirname(path)
  # os.add_dll_directory is available on Windows in Python 3.8+
  add_dll_dir = getattr(os, "add_dll_directory", None)
  if add_dll_dir is not None and callable(add_dll_dir):
    try:
      add_dll_dir(target_dir)
    except OSError:
      pass

  for dll_name in deps:
    if dll_name.lower() in visited:
      continue

    # 1. Recurse first (Leaf-first)
    if not dll_name.lower().startswith(
        ("kernel32", "user32", "api-ms-win-", "msvcrt", "ucrtbase")
    ):
      for p in [target_dir] + os.environ["PATH"].split(os.pathsep):
        full_p = os.path.join(p, dll_name)
        if os.path.exists(full_p):
          diagnose_dll_load(full_p, depth + 1, visited)
          break

    # 2. After recursion, try to load at this level
    try:
      _ = ctypes.WinDLL(dll_name)
    except OSError as e:
      err = getattr(e, "winerror", None)
      print(f"{indent}[Error] Failed to load {dll_name}: ", end="")
      if err == ERROR_MOD_NOT_FOUND:
        print(
            "MODULE NOT FOUND (0x7E) - Check if the file or its dependencies"
            " exist."
        )
      elif err == ERROR_BAD_EXE_FORMAT:
        print("BAD EXE FORMAT (0xC1/193) - Likely a 32-bit vs 64-bit mismatch.")
      elif err == ERROR_PROC_NOT_FOUND:
        print(
            "PROC NOT FOUND (0x7F/127) - Version mismatch in a dependent DLL."
        )
      elif err == ERROR_ACCESS_DENIED:
        print("ACCESS DENIED (0x05) - Check file permissions.")
      elif err == ERROR_DLL_INIT_FAILED:
        print(
            "INITIALIZATION FAILED (0x45A) - The DLL's DllMain returned false."
        )
        print(
            f"{indent}    Hint: This often happens if your CPU lacks required"
            " instructions (like AVX/AVX2)"
        )
        print(
            f"{indent}    or if the Microsoft Visual C++ Redistributable is"
            " outdated/missing."
        )
      else:
        print(f"UNKNOWN ERROR ({err}): {e}")
      visited.add(dll_name.lower())


def run_diagnosis(path=None):
  # [FUNCTION DOC]
  # Entry point of the diagnostic. Input: path, optional string path of the primary DLL or
  # extension module; when omitted, it is derived from this file's location as the internal
  # TensorFlow pywrap extension in the parent python directory. If running on Windows and the
  # path exists, runs the recursive dependency diagnosis on it; otherwise prints an error
  # message saying the path is missing or the platform is not Windows. Operating-system errors
  # are caught and reported by printing. Returns nothing; side effect is output to standard
  # output plus those of the diagnosis.
  """Runs the Windows DLL load diagnosis.

  Args:
    path: The path to the primary DLL/PYD file to diagnose. If None, it defaults
      to '_pywrap_tensorflow_internal.pyd' located relative to this script.
  """
  try:
    if path is None:
      # Determine path relative to this file to avoid importing tensorflow
      # This file is typically at tensorflow/python/platform/
      platform_dir = os.path.dirname(os.path.abspath(__file__))
      python_dir = os.path.dirname(platform_dir)
      path = os.path.join(python_dir, "_pywrap_tensorflow_internal.pyd")

    if os.name == "nt" and os.path.exists(path):
      diagnose_dll_load(path)
    else:
      print(f"Error: Path does not exist or not on Windows: {path}")
  except OSError as e:
    print(f"Diagnostic failed: {e}")


if __name__ == "__main__":
  if len(sys.argv) > 1:
    run_diagnosis(sys.argv[1])
  else:
    run_diagnosis()
