# [MODULE DOC]
# Unit tests for the tool that inserts generated FileCheck expectations into HLO test files
# (the generate_hlo_test_checks module). The golden-file tests run the real HLO optimizer
# program, located in the parent directory, with two passes enabled on an input test file placed
# next to this module and compare the printed output with an expected output file in the same
# directory. Other tests cover argument parsing and the worker pool life cycle of the writer class
# as a context manager. Run directly it starts the absltest runner.
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
# ==============================================================================

import io
import os

from absl.testing import absltest

from xla.hlo.tools import generate_hlo_test_checks


class GenerateHloTestChecksTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for the test-check writer and the argument parser of the generator tool. The shared
  # fixture (set up before each test) records the test data directory, the path of the optimizer
  # program in the parent directory, and the paths of the golden input file and expected output
  # file in the data directory.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture run before each test: calls the base setup, then stores the directory of this file,
    # the optimizer program path (in the parent directory), and the paths of the input and
    # expected output golden files. Takes no arguments, returns nothing, touches no files.
    super().setUp()
    self._data_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(self._data_dir, os.pardir))
    self._optimizer_path = os.path.join(parent_dir, "hlo-opt")
    self._input_file_path = os.path.join(
        self._data_dir, "generate_hlo_test_checks_test_input.hlo"
    )
    self._output_file_path = os.path.join(
        self._data_dir, "generate_hlo_test_checks_test_output.hlo"
    )

  def test_parallel_mode_generates_expected_output_file(self):
    # [FUNCTION DOC]
    # Test: with an explicit worker count of eight (so the pool path is used even on a single-core
    # machine) and the logistic-expander and reshape-mover passes, the writer used as a context
    # manager prints the transformation of the input file into an in-memory stream. Asserts that
    # the text equals the contents of the expected output file. Needs the optimizer program.
    actual_output = io.StringIO()

    # The `worker_count` argument is normally inferred, but we specify it here
    # in order to ensure that we're testing the parallel code path even in the
    # unlikely event that `os.process_cpu_count() == 1` on the test machine.
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=[
            "{}",
            "--passes=logistic-expander,reshape-mover",
        ],
        worker_count=8,
    ) as writer:
      writer.transform_and_print_file(
          self._input_file_path,
          output_stream=actual_output,
      )

    actual_output.seek(0)

    with open(self._output_file_path, mode="r") as expected_output:
      self.assertEqual(actual_output.read(), expected_output.read())

  def test_sequential_mode_generates_expected_output_file(self):
    # [FUNCTION DOC]
    # Test: same scenario as the parallel test but the writer is used without a context manager
    # and with the default worker count, so processing is sequential. Asserts the printed text
    # equals the expected output file contents. Needs the optimizer program.
    actual_output = io.StringIO()

    writer = generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=[
            "{}",
            "--passes=logistic-expander,reshape-mover",
        ],
    )

    writer.transform_and_print_file(
        self._input_file_path,
        output_stream=actual_output,
    )

    actual_output.seek(0)

    with open(self._output_file_path, mode="r") as expected_output:
      self.assertEqual(actual_output.read(), expected_output.read())

  def test_custom_input_file_placeholder_string(self):
    # [FUNCTION DOC]
    # Test: uses a custom expansion token (percent s) both as the placeholder in the optimizer
    # arguments and as the expansion option of the writer, in a context manager. Asserts the printed
    # output equals the same expected output file, showing the token is configurable.
    actual_output = io.StringIO()

    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=[
            "%s",
            "--passes=logistic-expander,reshape-mover",
        ],
        expand_to_input="%s",
    ) as writer:
      writer.transform_and_print_file(
          self._input_file_path,
          output_stream=actual_output,
      )

    actual_output.seek(0)

    with open(self._output_file_path, mode="r") as expected_output:
      self.assertEqual(actual_output.read(), expected_output.read())

  def test_argument_parsing(self):
    # [FUNCTION DOC]
    # Test: parses two argument lists. The first has a test file, a separator and an optimizer
    # command with arguments; asserts in-place is false, the expansion token is the default brace
    # placeholder, and the command and its arguments are split correctly. The second adds the
    # in-place flag and a custom expansion token; asserts in-place is true and the token is the
    # custom one.
    args = generate_hlo_test_checks.parse_args(
        "/path/to/test_file.hlo -- /path/to/hlo-opt {} --passes=foo,bar".split()
    )

    self.assertEqual(args.test_file, "/path/to/test_file.hlo")
    self.assertEqual(args.in_place, False)
    self.assertEqual(args.expand_to_input, "{}")
    self.assertEqual(args.opt_cmd, "/path/to/hlo-opt")
    self.assertEqual(args.opt_args, ["{}", "--passes=foo,bar"])

    args = generate_hlo_test_checks.parse_args(
        "test_file.hlo -i -I%s -- hlo-opt %s --passes=foo,bar".split()
    )

    self.assertEqual(args.test_file, "test_file.hlo")
    self.assertEqual(args.in_place, True)
    self.assertEqual(args.expand_to_input, "%s")
    self.assertEqual(args.opt_cmd, "hlo-opt")
    self.assertEqual(args.opt_args, ["%s", "--passes=foo,bar"])

  def test_conditional_parallelism(self):
    # [FUNCTION DOC]
    # Test: a writer with a worker count of two has a worker pool while inside its context manager
    # and none after leaving; a writer with a count of one never has a pool; constructing a writer
    # with a count of zero raises a value error. Inspects the private pool attribute.
    # A writer with `worker_count > 1` should have a worker pool.
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
        worker_count=2,
    ) as parallel_writer:
      self.assertIsNotNone(parallel_writer._worker_pool)

    # The worker pool should be destroyed when exiting the context manager.
    self.assertIsNone(parallel_writer._worker_pool)

    # A writer with `worker_count == 1` should not have a worker pool.
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
        worker_count=1,
    ) as sequential_writer:
      self.assertIsNone(sequential_writer._worker_pool)

    # Attempting to construct a writer with `worker_count < 1` should result in
    # a `ValueError`.
    with self.assertRaises(ValueError):
      with generate_hlo_test_checks.TestCheckWriter(
          optimizer_path=self._optimizer_path,
          optimizer_args=["{}"],
          worker_count=0,
      ):
        pass

  def test_unscoped_writer(self):
    # [FUNCTION DOC]
    # Test: a writer with a worker count of four has no pool when created outside a context
    # manager, gets one inside a with block, and has none again afterwards. Inspects the private
    # pool attribute.
    # A writer with no active context manager shouldn't have a worker pool even
    # if `worker_count > 1`. This is because the worker pool is RAII-managed by
    # the context manager (i.e. the `with` block). In this situation, the writer
    # will operate sequentially as though `worker_count` were 1.
    writer = generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
        worker_count=4,
    )
    self.assertIsNone(writer._worker_pool)

    # However, entering a context manager for a previously constructed writer
    # with `worker_count > 1` should initialize a worker pool that lasts for the
    # duration of the context manager.
    with writer:
      self.assertIsNotNone(writer._worker_pool)

    # The worker pool should be destroyed when exiting the context manager.
    self.assertIsNone(writer._worker_pool)

  def test_prevent_conflicting_context_managers(self):
    # [FUNCTION DOC]
    # Test: entering a second with block on a writer that is already inside one raises a runtime
    # error.
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
    ) as writer:
      # Creating multiple overlapping context managers for the same
      # `TestCheckWriter` instance should result in a `RuntimeError`.
      with self.assertRaises(RuntimeError):
        with writer:
          pass

  def test_allow_non_conflicting_context_managers(self):
    # [FUNCTION DOC]
    # Test: two different writers can be in context managers at the same time, and a writer can be
    # re-entered once its earlier with block has finished (resources released and re-acquired).
    # Asserts nothing explicitly: it passes if no exception is raised.
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
    ) as writer:
      # Context managers for *separate* `TestCheckWriter` instances are allowed
      # to coexist since they're managing separate resources.
      with generate_hlo_test_checks.TestCheckWriter(
          optimizer_path=self._optimizer_path,
          optimizer_args=["{}"],
      ):
        pass

    # A given `TestCheckWriter` instance may be reused in multiple context
    # managers as long as their scopes never overlap with each other (i.e. the
    # resources are released before they're re-acquired).
    with writer:
      pass


if __name__ == "__main__":
  absltest.main()
