# [MODULE DOC]
# HELPER: Build-time command-line tool that generates C++ headers embedding AMD GPU device
# library bitcode, used by the GPU LLVM backend build rules.
# Links the given LLVM bitcode files with the llvm-link program and writes a C++ header holding
# the linked bitcode as an escaped byte string constant plus an LLVM string reference constant
# wrapping it, inside a configurable namespace. Run as a script; nothing happens at import time.
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tool to generate C++ headers from LLVM bitcode files.

This tool links multiple LLVM bitcode files using llvm-link and then converts
the resulting bitcode into a C++ header file containing a byte array and an
llvm::StringRef.
"""

import argparse
import itertools
import subprocess


def main():
  # [FUNCTION DOC]
  # KNOWN ISSUE: with the default empty namespace the generated text reads "namespace  {", which
  # is an anonymous namespace rather than the global scope.
  # Command-line entry point. Arguments parsed from the process command line: the path of the
  # llvm-link binary (required), the output header filename (required), one or more input bitcode
  # files, an optional C++ namespace (default empty), and a required C++ identifier.
  # Runs llvm-link with force and standard-output options on an empty input plus each input file
  # given as an override, capturing its output; a failing tool raises a called-process error.
  # Formats every output byte as a hexadecimal escape in one string and writes (overwriting) the
  # header: pragma once, an include of the LLVM string reference header, and in the namespace a
  # raw character array and a string reference constant named by the identifier (size minus the
  # terminating character). Returns nothing.
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--llvm_link_bin", required=True, help="Path to the llvm-link binary"
  )
  parser.add_argument(
      "-o", "--output", required=True, help="Output filename for the C++ header"
  )
  parser.add_argument(
      "input_files", nargs="+", help="Variable number of input filenames"
  )
  parser.add_argument(
      "--cpp_namespace",
      default="",
      help="Namespace to be used when generating data",
  )
  parser.add_argument(
      "--cpp_identifier",
      required=True,
      help="Identifier to be used to refer to data",
  )

  args = parser.parse_args()
  llvm_link_bin = args.llvm_link_bin
  output_filename = args.output
  input_filenames = args.input_files
  cpp_namespace = args.cpp_namespace
  cpp_identifier = args.cpp_identifier

  result = subprocess.run(
      [llvm_link_bin, "-f", "-o", "-", "/dev/null"]
      + list(
          itertools.chain.from_iterable(
              ("--override", f) for f in input_filenames
          )
      ),
      capture_output=True,
      check=True,
  )

  llvm_output = result.stdout
  data_string = "".join("\\x{:02x}".format(byte) for byte in llvm_output)

  with open(output_filename, "w") as output_file:
    output_file.write(f"""\
#pragma once

#include "llvm/ADT/StringRef.h"

namespace {cpp_namespace} {{
  inline const char kRaw_{cpp_identifier}[] = "{data_string}";
  constexpr llvm::StringRef {cpp_identifier}{{kRaw_{cpp_identifier}, sizeof(kRaw_{cpp_identifier}) - 1}};
}} // namespace {cpp_namespace}
""")


if __name__ == "__main__":
  main()
