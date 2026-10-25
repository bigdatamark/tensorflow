# [MODULE DOC]
# Test that runs an HLO text file containing Transformer Engine custom calls through the
# multihost HLO runner Python binding. Registers the Transformer Engine custom call targets for
# the CUDA platform before the test, so it needs the transformer engine JAX package, a GPU
# platform and the test environment variables set by the Bazel test runner. Runs under absltest.
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for running HLO files."""

import os
import pathlib

from absl.testing import absltest
from transformer_engine import transformer_engine_jax

from xla.tools.multihost_hlo_runner import py_hlo_multihost_runner


def _register_transformer_engine_custom_calls():
  # [FUNCTION DOC]
  # HELPER: Test setup glue for the Transformer Engine test class.
  # Asks the Transformer Engine JAX module for its custom call registrations (mapping of names to
  # handlers) and registers each with the HLO runner for the CUDA platform using API version 1.
  # Side effect: mutates global custom call registry of the runner. Takes no arguments, returns
  # nothing; errors from the underlying modules propagate.
  for name, value in transformer_engine_jax.registrations().items():
    py_hlo_multihost_runner.register_custom_call_target(
        name, value, platform="CUDA", api_version=1
    )


def _get_test_hlo_path(file_name: str) -> str:
  # [FUNCTION DOC]
  # HELPER: Test utility locating data files.
  # Input: a file name. Reads three environment variables set by the Bazel test runner (source
  # directory, workspace name, test binary path; a missing one raises a key error) and returns the
  # path to that file inside the data directory next to the test binary's location.
  """Returns the path to a HLO file in the data directory."""
  test_srcdir = pathlib.Path(os.environ["TEST_SRCDIR"])
  test_workspace = os.environ["TEST_WORKSPACE"]
  test_binary = os.environ["TEST_BINARY"]
  return os.path.join(
      os.path.dirname(test_srcdir / test_workspace / test_binary),
      "data",
      file_name,
  )


class RunTEHloTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for running HLO files with Transformer Engine custom calls. The shared fixture is a
  # runner configuration created in set-up: text input format and uninitialized module arguments.
  """Tests for running custom calls from Transformer Engine."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture set-up. Calls the base set-up, registers the Transformer Engine custom calls
    # (global side effect), and stores in the instance a runner configuration whose input format
    # is text and whose argument mode is uninitialized (module arguments are not given real data).
    super().setUp()
    _register_transformer_engine_custom_calls()
    self.config = py_hlo_multihost_runner.PyHloRunnerConfig()
    self.config.input_format = py_hlo_multihost_runner.InputFormat.Text
    self.config.hlo_argument_mode = (
        py_hlo_multihost_runner.ModuleArgumentMode.Uninitialized
    )

  def test_run_custom_call_hlo(self):
    # [FUNCTION DOC]
    # Locates the Transformer Engine softmax HLO text file in the test data directory and runs it
    # with the runner using the configuration from set-up. The test passes if the run completes
    # without raising; no output is compared.
    hlo_file = _get_test_hlo_path("transformer_engine_softmax.hlo")
    py_hlo_multihost_runner.RunHloFiles([hlo_file], self.config)


def main():
  # [FUNCTION DOC]
  # HELPER: Script entry wrapper. Starts the absltest test runner, which discovers and runs the
  # tests and exits the process with the result status.
  absltest.main()


if __name__ == "__main__":
  main()
