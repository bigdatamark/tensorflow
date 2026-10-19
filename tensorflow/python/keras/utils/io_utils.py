# [MODULE DOC]
# HELPER: helper module of small file-system utilities for the Keras utils package.
# Provides conversion of path-like objects to plain strings and an interactive prompt that asks
# the user whether an existing file may be overwritten. No import-time side effects other than
# importing the standard os module.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
"""Utilities related to disk I/O."""

import os


def path_to_string(path):
  # [FUNCTION DOC]
  # HELPER: used by file-saving and loading code to normalize path arguments.
  # Takes any object. If it is a path-like object (implements the OS path protocol) it is
  # converted to its string form; any other value (strings, open file objects, nothing) is
  # returned unchanged. Raises nothing itself and has no side effects.
  """Convert `PathLike` objects to their string representation.

  If given a non-string typed path object, converts it to its string
  representation.

  If the object passed to `path` is not among the above, then it is
  returned unchanged. This allows e.g. passthrough of file objects
  through this function.

  Args:
    path: `PathLike` object that represents a path

  Returns:
    A string representation of the path argument, if Python support exists.
  """
  if isinstance(path, os.PathLike):
    return os.fspath(path)
  return path


def ask_to_proceed_with_overwrite(filepath):
  # [FUNCTION DOC]
  # HELPER: interactive guard used before overwriting an existing model or weights file.
  # Input: filepath, the path of the file that already exists (only used in the message).
  # Prompts on standard input until the user answers y or n (case-insensitive, whitespace
  # ignored), re-prompting on any other reply. Returns False if the user answers n. If the
  # answer is y it prints a tip suggesting the overwrite option and returns True.
  # Side effects: reads from standard input and prints to standard output. Raises an end-of-file
  # error if input is closed.
  """Produces a prompt asking about overwriting a file.

  Args:
      filepath: the path to the file to be overwritten.

  Returns:
      True if we can proceed with overwrite, False otherwise.
  """
  overwrite = input('[WARNING] %s already exists - overwrite? '
                    '[y/n]' % (filepath)).strip().lower()
  while overwrite not in ('y', 'n'):
    overwrite = input('Enter "y" (overwrite) or "n" '
                      '(cancel).').strip().lower()
  if overwrite == 'n':
    return False
  print('[TIP] Next time specify overwrite=True!')
  return True
