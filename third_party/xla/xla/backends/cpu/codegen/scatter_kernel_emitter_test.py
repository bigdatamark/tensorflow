# [MODULE DOC]
# Test module for the CPU scatter kernel emitter. Provides two helpers (a scalar-add update
# computation builder and a function that builds, wraps in a fusion, emits, compiles and returns
# a kernel runner for a scatter instruction) and a test case class with a basic scatter test with
# repeated indices (updates are accumulated) and an out-of-bounds test (the scatter is skipped).
# Uses the XLA CPU test library bindings and numpy. Runs the test main when executed as a script.
# Copyright 2024 The OpenXLA Authors.
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


from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu import testlib as testlib_cpu
from xla.backends.cpu.testlib import utilities
from xla.codegen import testlib as testlib_base
from xla.codegen.testlib import utilities as base_utilities
from xla.python import xla_extension


def create_sclar_add_computation(
    dtype: np.dtype,
) -> testlib_base.HloComputation:
  # [FUNCTION DOC]
  # HELPER: small shared test utility that builds the scatter update computation (the name has a
  # typo, "sclar", which is kept for existing callers).
  # Input: numpy data type of the scalars. Creates two scalar parameters of that type and an
  # addition over them, and returns an HLO computation with the addition as root. No side effects.
  scalar_shape = xla_extension.Shape.scalar_shape(dtype)
  param_0 = testlib_base.HloInstruction.create_parameter(
      0,
      scalar_shape,
      "update_param_0",
  )
  param_1 = testlib_base.HloInstruction.create_parameter(
      1,
      scalar_shape,
      "update_param_1",
  )
  root = testlib_base.HloInstruction.create_binary(
      scalar_shape,
      testlib_base.HloOpcode.add,
      param_0,
      param_1,
  )
  return testlib_base.build_hlo_computation(
      root,
      param_0,
      param_1,
  )


def create_scatter_runner(
    operand: xla_extension.Literal,
    scatter_indicies: xla_extension.Literal,
    updates: xla_extension.Literal,
    scatter_dimension_numbers: testlib_base.ScatterDimensionNumbers,
) -> testlib_cpu.KernelRunner:
  # [FUNCTION DOC]
  # HELPER: shared test utility that turns scatter inputs into a ready-to-run kernel.
  # KNOWN ISSUE: the update computation always uses float32 regardless of the dtype of the given
  # operand literal, so only float32 operands work; the parameter name for the indices is
  # misspelled ("indicies") throughout.
  # Inputs: operand literal, scatter indices literal, updates literal and the scatter dimension
  # numbers. Builds parameters from the literals' shapes, a scatter instruction with the add update
  # computation (neither indices-are-sorted nor unique-indices set), an HLO module, runs the
  # fusion wrapper pass, annotates the module to get a buffer assignment, emits the kernel with
  # the scatter emitter and creates a JIT compiler. Returns a kernel runner for the kernel.
  dtype = np.dtype(np.float32)

  operand_parameter = testlib_base.HloInstruction.create_parameter(
      0, operand.shape(), "operand"
  )
  scatter_indicies_parameter = testlib_base.HloInstruction.create_parameter(
      1, scatter_indicies.shape(), "scatter_indicies"
  )
  updates_parameter = testlib_base.HloInstruction.create_parameter(
      2, updates.shape(), "updates"
  )

  update_computation = create_sclar_add_computation(dtype)

  scatter_instruction = testlib_base.HloInstruction.create_scatter(
      operand.shape(),
      operand_parameter,
      scatter_indicies_parameter,
      updates_parameter,
      update_computation,
      scatter_dimension_numbers,
      False,
      False,
  )

  hlo_module, _ = utilities.build_hlo_module(
      scatter_instruction,
      operand_parameter,
      scatter_indicies_parameter,
      updates_parameter,
      extra_computations=[update_computation],
  )

  hlo_module = testlib_cpu.run_fusion_wrapper_pass(hlo_module)
  hlo_module, buffer_assignment = utilities.annotate_hlo_module(hlo_module)

  mlir_context = testlib_cpu.MLIRContext()
  scatter_emitter = testlib_cpu.ScatterKernelEmitter(
      hlo_module.get_root_instruction(),
      buffer_assignment,
      mlir_context,
  )
  kernel_definition = scatter_emitter.emit_kernel_definition()

  jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

  return testlib_cpu.KernelRunner.create(kernel_definition, jit_compiler)


class ScatterKernelTest(parameterized.TestCase):
  # [CLASS DOC]
  # Test case class covering the CPU scatter kernel emitter, using the shared runner-building
  # helper above. No shared fixtures.

  def test_basic_scatter(self):
    # [FUNCTION DOC]
    # Scenario: scatter of five two-dimensional index rows with updates of window size one by
    # three into a zero float32 operand of three by four, where one index row is repeated.
    # Setup: operand of zeros, indices array, updates array of consecutive numbers reshaped to
    # five by one by three. Action: build the runner and run it. Expected: operand row zero holds
    # the first update followed by zero; row one holds the third update plus the fourth update
    # shifted by one column; row two holds the sum of the second and fifth updates (accumulated for
    # the repeated index) followed by zero.
    dtype = np.dtype(np.float32)

    scatter_dimension_numbers = testlib_base.ScatterDimensionNumbers(
        update_window_dims=[1, 2],
        inserted_window_dims=[],
        scatter_dims_to_operand_dims=[0, 1],
        index_vector_dim=1,
    )

    operand_shape = [3, 4]
    operand = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    # Repeat the last dimension to test the update computation.
    scatter_indicies = base_utilities.create_literal_from_np(
        np.array([[0, 0], [2, 0], [1, 0], [1, 1], [2, 0]], dtype=np.int32)
    )

    updates = base_utilities.create_literal_from_np(
        np.arange(15, dtype=dtype).reshape([5, 1, 3])
    )

    runner = create_scatter_runner(
        operand, scatter_indicies, updates, scatter_dimension_numbers
    )

    runner.call([operand, scatter_indicies, updates])

    operand_np = np.asarray(operand)
    updates_np = np.asarray(updates).reshape([5, 3])

    np.testing.assert_array_equal(
        operand_np[0, :], np.concatenate((updates_np[0, :], [0]))
    )
    np.testing.assert_array_equal(
        operand_np[1, :],
        np.concatenate((updates_np[2, :], [0]))
        + (np.concatenate(([0], updates_np[3, :]))),
    )
    np.testing.assert_array_equal(
        operand_np[2, :],
        np.concatenate((updates_np[1, :] + updates_np[4, :], [0])),
    )

  def test_out_of_bounds_scatter(self):
    # [FUNCTION DOC]
    # Scenario: a single scatter index whose offset of two combined with a window of three columns
    # exceeds the four-column operand. Setup: zero operand of three by four, one index row, one
    # update of three consecutive numbers. Action: build the runner and run it. Expected: the
    # operand is unchanged (still all zeros), since out-of-bounds updates are skipped.
    dtype = np.dtype(np.float32)

    scatter_dimension_numbers = testlib_base.ScatterDimensionNumbers(
        update_window_dims=[1, 2],
        inserted_window_dims=[],
        scatter_dims_to_operand_dims=[0, 1],
        index_vector_dim=1,
    )

    operand_shape = [3, 4]
    operand = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    # Offset of 2 combined with window size of 3 results in oob.
    scatter_indicies = base_utilities.create_literal_from_np(
        np.array([[0, 2]], dtype=np.int32)
    )

    updates = base_utilities.create_literal_from_np(
        np.arange(3, dtype=dtype).reshape([1, 1, 3])
    )

    runner = create_scatter_runner(
        operand, scatter_indicies, updates, scatter_dimension_numbers
    )

    runner.call([operand, scatter_indicies, updates])

    np.testing.assert_array_equal(
        np.asarray(operand), np.zeros(operand_shape, dtype)
    )


if __name__ == "__main__":
  absltest.main()
