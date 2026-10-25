# [MODULE DOC]
# HELPER: build-time filter script used when generating OpenMP configuration headers from CMake
# template files.
# Reads a template from standard input and writes the expanded text to standard output.
# Variable definitions come from the command-line arguments, each of the form NAME=VALUE.
# Lines declaring a conditional define are turned into a define or a commented-out undefine
# depending on whether the variable is given; a flag-style define always becomes a define of
# zero or the given value; all other lines have dollar-brace and at-sign variable references
# replaced by their values (unknown variables become empty). No import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Expands CMake variables in a text file."""

import re
import sys

_CMAKE_DEFINE_REGEX = re.compile(r"\s*#cmakedefine\s+([A-Za-z_0-9]*)(\s.*)?$")
_CMAKE_DEFINE01_REGEX = re.compile(r"\s*#cmakedefine01\s+([A-Za-z_0-9]*)")
_CMAKE_VAR_REGEX = re.compile(r"\${([A-Za-z_0-9]*)}")
_CMAKE_ATVAR_REGEX = re.compile(r"@([A-Za-z_0-9]*)@")


def _parse_args(argv):
  # [FUNCTION DOC]
  # HELPER: command-line argument parser for the script entry point.
  # KNOWN ISSUE: each argument is split on every equals sign, so a value that itself contains an
  # equals sign makes the unpacking fail with a value error; an argument without an equals sign
  # fails the same way.
  # Input: list of strings of the form name=value. Returns a mapping from names to values; later
  # duplicates overwrite earlier ones. No side effects.
  """Parses arguments with the form KEY=VALUE into a dictionary."""
  result = {}
  for arg in argv:
    k, v = arg.split("=")
    result[k] = v
  return result


def _expand_variables(input_str, cmake_vars):
  # [FUNCTION DOC]
  # HELPER: variable substitution used for every line and for define suffixes.
  # Inputs: a string and a mapping from variable names to string values. Replaces each
  # dollar-brace reference first, then each at-sign reference in the result, by the mapped value,
  # or by an empty string when the name is not in the mapping. Returns the expanded string.
  # Pure function.
  """Expands ${VARIABLE}s and @VARIABLE@s in 'input_str', using dictionary 'cmake_vars'.

  Args:
    input_str: the string containing ${VARIABLE} or @VARIABLE@ expressions to expand.
    cmake_vars: a dictionary mapping variable names to their values.

  Returns:
    The expanded string.
  """
  def replace(match):
    # [FUNCTION DOC]
    # HELPER: substitution callback for the regular-expression replacement above.
    # Input: a match whose first group is a variable name. Returns the mapped value of that
    # variable from the enclosing mapping, or an empty string if it is not defined.
    if match.group(1) in cmake_vars:
      return cmake_vars[match.group(1)]
    return ""
  return _CMAKE_ATVAR_REGEX.sub(replace,_CMAKE_VAR_REGEX.sub(replace, input_str))


def _expand_cmakedefines(line, cmake_vars):
  # [FUNCTION DOC]
  # HELPER: per-line transformer used by the entry point.
  # Inputs: one text line (with its newline) and a mapping of variable names to values.
  # For a conditional-define line: if the name is in the mapping, returns a define of the name
  # followed by the variable-expanded remainder of the line; otherwise returns a commented-out
  # undefine. For a flag-style define line: returns a define of the name with its mapped value,
  # or zero when absent. Any other line is returned with variable references expanded.
  # Outputs always end with a newline for define cases. Pure function.
  """Expands #cmakedefine declarations, using a dictionary 'cmake_vars'."""

  # Handles #cmakedefine lines
  match = _CMAKE_DEFINE_REGEX.match(line)
  if match:
    name = match.group(1)
    suffix = match.group(2) or ""
    if name in cmake_vars:
      return "#define {}{}\n".format(name,
                                     _expand_variables(suffix, cmake_vars))
    else:
      return "/* #undef {} */\n".format(name)

  # Handles #cmakedefine01 lines
  match = _CMAKE_DEFINE01_REGEX.match(line)
  if match:
    name = match.group(1)
    value = cmake_vars.get(name, "0")
    return "#define {} {}\n".format(name, value)

  # Otherwise return the line unchanged.
  return _expand_variables(line, cmake_vars)


def main():
  # [FUNCTION DOC]
  # Script entry point. Takes the variable definitions from the command-line arguments (after the
  # program name), then reads standard input line by line and writes each transformed line to
  # standard output. Returns nothing. A malformed argument makes the argument parser helper raise
  # a value error.
  cmake_vars = _parse_args(sys.argv[1:])
  for line in sys.stdin:
    sys.stdout.write(_expand_cmakedefines(line, cmake_vars))


if __name__ == "__main__":
  main()
