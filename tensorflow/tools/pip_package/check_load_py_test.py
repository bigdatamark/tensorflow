# [MODULE DOC]
# Script-style test that checks every pip-testable Python test target in the build has its build
# file loading the test rule from the TensorFlow macro file. It queries the build system for test
# targets, drops those tagged as not for pip, collects their build files, and asks the buildozer
# tool whether adding the load statement would change each file. Raises an error listing files
# that lack the load, otherwise prints a pass message.
# Import-time side effect: changes the working directory to the repository root.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests to check that py_test are properly loaded in BUILD files."""

import os
import subprocess


os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def check_output_despite_error(args):
  # [FUNCTION DOC]
  # HELPER: runs buildozer and similar shell commands for the main check.
  # Input: args, the command line passed to a subprocess run through a shell. Returns the
  # combined standard output and error with surrounding whitespace removed, as bytes; if the
  # command exits with an error the captured output is returned instead of raising.
  # KNOWN ISSUE: passing a list together with the shell option makes only the first list item the
  # shell command on POSIX systems, so the remaining items become shell arguments, not arguments
  # of the tool; the docstring also claims the result is a string though it is bytes.
  """Get output of args from command line, even if there are errors.

  Args:
    args: a list of command line args.

  Returns:
    output as string.
  """
  try:
    output = subprocess.check_output(args, shell=True, stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    output = e.output
  return output.strip()


def main():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the build query refers to a contrib directory that no longer exists in the
  # repository, and the output of the helper is compared as bytes with a string kind name, so the
  # filter on test kind can never match under Python 3.
  # Takes no arguments. Queries the build system for Python test targets, filters by kind and
  # the no-pip tag via buildozer, derives their build files, and checks each by running
  # buildozer to add the load statement and comparing with the current file content. Raises a
  # runtime error naming the files missing the load; otherwise prints a pass message.
  # Runs external processes and reads build files; writes nothing.
  # Get all py_test target, note bazel query result will also include
  # cuda_py_test etc.
  try:
    targets = subprocess.check_output([
        'bazel', 'query',
        'kind(py_test, //tensorflow/contrib/... + '
        '//tensorflow/python/... - '
        '//tensorflow/contrib/tensorboard/...)']).strip()
  except subprocess.CalledProcessError as e:
    targets = e.output
  targets = targets.decode("utf-8") if isinstance(targets, bytes) else targets

  # Only keep py_test targets, and filter out targets with 'no_pip' tag.
  valid_targets = []
  for target in targets.split('\n'):
    kind = check_output_despite_error(['buildozer', 'print kind', target])
    if kind == 'py_test':
      tags = check_output_despite_error(['buildozer', 'print tags', target])
      if 'no_pip' not in tags:
        valid_targets.append(target)

  # Get all BUILD files for all valid targets.
  build_files = set()
  for target in valid_targets:
    build_files.add(os.path.join(target[2:].split(':')[0], 'BUILD'))

  # Check if BUILD files load py_test.
  files_missing_load = []
  for build_file in build_files:
    updated_build_file = subprocess.check_output(
        ['buildozer', '-stdout', 'new_load //tensorflow:tensorflow.bzl py_test',
         build_file])
    with open(build_file, 'r') as f:
      if f.read() != updated_build_file:
        files_missing_load.append(build_file)

  if files_missing_load:
    raise RuntimeError('The following files are missing %s:\n %s' % (
        'load("//tensorflow:tensorflow.bzl", "py_test").\nThis load statement'
        ' is needed because otherwise pip tests will try to use their '
        'dependencies, which are not visible to them.',
        '\n'.join(files_missing_load)))
  else:
    print('TEST PASSED.')


if __name__ == '__main__':
  main()
