#!/usr/bin/env python
# [MODULE DOC]
# HELPER: CI consistency check script for the system-libraries build configuration.
# Run as a script with the TensorFlow source folder as the first argument (missing argument gives
# an index error; a non-directory gives a value error). Verifies that the list of valid library
# names defined in the system libraries configuration bzl file matches the set of external
# repository rules (in the workspace bzl files of the tensorflow folder and of each third party
# folder) that declare a system build file, and that each referenced system build file exists.
# Prints the problems found and exits with status 1 if any, otherwise 0. All work happens at
# import time (no main guard): it builds path globals, validates inputs, scans files, prints and
# terminates the process.
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
#
# Checks that the options mentioned in syslibs_configure.bzl are consistent with
# the valid options in workspace.bzl
# Expects the tensorflow source folder as the first argument

import glob
import os
import sys

tf_source_path = sys.argv[1]

syslibs_configure_path = os.path.join(tf_source_path, 'third_party',
                                      'systemlibs', 'syslibs_configure.bzl')
workspace0_path = os.path.join(tf_source_path, 'tensorflow', 'workspace0.bzl')
workspace_glob = os.path.join(tf_source_path, 'tensorflow', 'workspace*.bzl')
third_party_path = os.path.join(tf_source_path, 'third_party')
third_party_glob = os.path.join(third_party_path, '*', 'workspace.bzl')

if not os.path.isdir(tf_source_path):
  raise ValueError('The path to the TensorFlow source must be passed as'
                   ' the first argument')

if not os.path.isfile(syslibs_configure_path):
  raise ValueError('Could not find syslibs_configure.bzl at %s' %
                   syslibs_configure_path)

if not os.path.isfile(workspace0_path):
  raise ValueError('Could not find workspace0.bzl at %s' % workspace0_path)

def extract_valid_libs(filepath):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the file content is executed as Python code (needs trusted input), and a file
  # without the valid libraries list raises a key error.
  # Evaluates the given configuration bzl file as Python source with a stubbed repository rule
  # function (which ignores its arguments) available, in separate global and local namespaces, and
  # returns the set of names found in the list named VALID_LIBS defined by that file.
  # Input: path of the file. Side effects: reads the file and executes its code.
  """Evaluate syslibs_configure.bzl, return the VALID_LIBS set from that file."""

  # Stub only
  def repository_rule(**kwargs):  # pylint: disable=unused-variable
    # [FUNCTION DOC]
    # HELPER: stub standing in for the build system's repository rule function while the
    # configuration file is evaluated. Accepts any keyword arguments, discards them, returns
    # nothing.
    del kwargs

  # Populates VALID_LIBS
  with open(filepath, 'r') as f:
    f_globals = {'repository_rule': repository_rule}
    f_locals = {}
    exec(f.read(), f_globals, f_locals)  # pylint: disable=exec-used

  return set(f_locals['VALID_LIBS'])


def extract_system_builds(filepath):
  # [FUNCTION DOC]
  # Scans a workspace bzl file line by line (stripped). A line starting with the name assignment
  # sets the current rule name (the text after the prefix without the final character and
  # quotes); a line starting with the system build file assignment appends the current name to
  # the library names and the target to the build file list. The target must start with a double
  # slash (checked by an assertion); it is converted to a relative path by dropping that prefix
  # and replacing the colon with a path separator. KNOWN ISSUE: if a system build file line comes
  # before any name line, the recorded library name is null.
  # Input: file path. Returns a pair of lists: library names and relative build file paths.
  """Extract the 'name' argument of all rules with a system_build_file argument."""
  lib_names = []
  system_build_files = []
  current_name = None
  with open(filepath, 'r') as f:
    for line in f:
      line = line.strip()
      if line.startswith('name = '):
        current_name = line[7:-1].strip('"')
      elif line.startswith('system_build_file = '):
        lib_names.append(current_name)
        # Split at '=' to extract rhs, then extract value between quotes
        system_build_spec = line.split('=')[-1].split('"')[1]
        assert system_build_spec.startswith('//')
        system_build_files.append(system_build_spec[2:].replace(':', os.sep))
  return lib_names, system_build_files


syslibs = extract_valid_libs(syslibs_configure_path)

syslibs_from_workspace = set()
system_build_files_from_workspace = []
for current_path in glob.glob(workspace_glob) + glob.glob(third_party_glob):
  cur_lib_names, build_files = extract_system_builds(current_path)
  syslibs_from_workspace.update(cur_lib_names)
  system_build_files_from_workspace.extend(build_files)

missing_build_files = [
    file for file in system_build_files_from_workspace
    if not os.path.isfile(os.path.join(tf_source_path, file))
]

has_error = False

if missing_build_files:
  has_error = True
  print('Missing system build files: ' + ', '.join(missing_build_files))

if syslibs != syslibs_from_workspace:
  has_error = True
  # Libs present in workspace files but not in the allowlist
  missing_syslibs = syslibs_from_workspace - syslibs
  if missing_syslibs:
    libs = ', '.join(sorted(missing_syslibs))
    print('Libs missing from syslibs_configure: ' + libs)
  # Libs present in the allow list but not in workspace files
  additional_syslibs = syslibs - syslibs_from_workspace
  if additional_syslibs:
    libs = ', '.join(sorted(additional_syslibs))
    print('Libs missing in workspace (or superfluous in syslibs_configure): ' +
          libs)

sys.exit(1 if has_error else 0)
