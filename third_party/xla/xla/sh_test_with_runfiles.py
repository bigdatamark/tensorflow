# [MODULE DOC]
# HELPER: helper module for the lit test configuration. Defines a lit shell-test format that makes
# Bazel runfiles visible next to each test's execution directory by temporarily linking them in.
# It is used by the lit configuration script of the same directory. No import-time side effects.
# Copyright 2025 The OpenXLA Authors.
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
"""Lit runner configuration."""

import os
import pathlib

import lit.formats


class ShTestWithRunfiles(lit.formats.ShTest):
  # [CLASS DOC]
  # HELPER: lit test format used by the XLA lit configuration. Extends the standard shell test
  # format; before running each test it symlinks every entry of the XLA subdirectory of the Bazel
  # runfiles tree into the directory that contains the test's execution path, and removes those
  # links afterwards. Holds no state of its own; created with the same arguments as the base
  # format.
  """Used to symlink bazels runfiles subdirs into the lit tmp test dir."""

  def execute(self, test, lit_config):
    # [FUNCTION DOC]
    # KNOWN ISSUE: links are removed only if the base execution returns normally; an exception
    # from it leaves the symlinks behind because no cleanup guard is used. Entries that already
    # exist are skipped silently and are not removed afterwards (correct), but a link creation
    # failure other than an existing-file error propagates.
    # Overrides the base execute. Inputs: the test object and the lit configuration. If the
    # runfiles directory environment variable is unset or has no XLA subdirectory, simply runs the
    # base execution. Otherwise creates the test's parent directory if needed, symlinks each
    # runfiles entry into it (marking directory links appropriately), runs the base execution,
    # unlinks the links it created and returns the base result.
    runfiles = os.environ.get("RUNFILES_DIR")
    if not runfiles:
      return super().execute(test, lit_config)

    runfiles_dir = pathlib.Path(runfiles) / "xla"
    if not runfiles_dir.is_dir():
      return super().execute(test, lit_config)

    dst = pathlib.Path(test.getExecPath()).parent
    dst.mkdir(parents=True, exist_ok=True)
    runfiles = []
    for item in runfiles_dir.iterdir():
      target = dst / item.name
      try:
        target.symlink_to(item, target_is_directory=item.is_dir())
        runfiles.append(target)
      except FileExistsError:
        pass

    result = super().execute(test, lit_config)
    for target in runfiles:
      target.unlink()
    return result
