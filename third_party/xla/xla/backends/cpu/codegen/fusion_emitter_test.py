# [MODULE DOC]
# Test module for the CPU fusion kernel emitter. Two test case classes parse small HLO text
# modules containing a loop fusion, emit a kernel for the fusion with the emitter binding,
# compile it with the JIT compiler, run it through the kernel runner on numpy-backed literals and
# compare the outputs against numpy computations. Covers elementwise add and subtract, conversion
# through bfloat16, NaN propagation, constants with non-default layouts and removal of dead
# exponential computations. Runs the test main when executed as a script.
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
from xla.codegen.testlib import utilities as base_utilities


class LoopFusionTest(parameterized.TestCase):
  # [CLASS DOC]
  # Test case class covering CPU loop-fusion kernel emission for elementwise operations, type
  # conversion and layout handling. No shared fixtures; each test parses its own HLO text.

  def test_basic_add_sub(self):
    # [FUNCTION DOC]
    # Scenario: a loop fusion over three float32 matrices of 100 by 200 computes an addition and
    # then a subtraction, returning both as a tuple. Setup: parse the HLO text, create the JIT
    # compiler, a compilation context and the fusion kernel, plus random inputs and zero-filled
    # outputs. Action: run the kernel. Expected: the first output equals the sum of the first two
    # inputs and the second output equals that sum minus the third input, within numpy's
    # almost-equal tolerance.
    dtype = np.dtype(np.float32)

    hlo = """
      HloModule test_module

      fusion_computation {
        %param_0 = f32[100, 200] parameter(0)
        %param_1 = f32[100, 200] parameter(1)
        %param_2 = f32[100, 200] parameter(2)
        %add = f32[100, 200] add(%param_0, %param_1)
        %sub = f32[100, 200] subtract(%add, %param_2)
        ROOT %tuple = (f32[100, 200], f32[100, 200]) tuple(%add, %sub)
      }

      ENTRY main {
        %param_0 = f32[100, 200] parameter(0)
        %param_1 = f32[100, 200] parameter(1)
        %param_2 = f32[100, 200] parameter(2)
        ROOT %wrapped_fusion = (f32[100, 200], f32[100, 200])
                               fusion(%param_0, %param_1, %param_2),
                               kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    operand_shape = (100, 200)

    param_0 = base_utilities.create_literal_from_np(
        np.random.rand(*operand_shape).astype(dtype)
    )
    param_1 = base_utilities.create_literal_from_np(
        np.random.rand(*operand_shape).astype(dtype)
    )
    param_2 = base_utilities.create_literal_from_np(
        np.random.rand(*operand_shape).astype(dtype)
    )

    result_0 = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )
    result_1 = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    kernel_runner.call([param_0, param_1, param_2, result_0, result_1])

    np.testing.assert_array_almost_equal(
        np.asarray(result_0), np.add(param_0, param_1)
    )
    np.testing.assert_array_almost_equal(
        np.asarray(result_1), np.subtract(np.add(param_0, param_1), param_2)
    )

  def test_convert_f32_bf16_f32(self):
    # [FUNCTION DOC]
    # Scenario: a fusion converts a float32 array of 300 by 500 to bfloat16 and back. Setup: input
    # of random values biased and scaled to a large range, with the first element set to
    # infinity; zero-filled output. Action: run the emitted kernel. Expected: the output matches
    # the input within a relative tolerance of two to the minus eight (the precision of
    # bfloat16's 7-bit fraction), and infinity is preserved.
    dtype = np.dtype(np.float32)

    hlo = """
      HloModule test_module

      fusion_computation {
        %param = f32[300, 500] parameter(0)
        %convert_bf16 = bf16[300, 500] convert(%param)
        ROOT %convert_f32 = f32[300, 500] convert(%convert_bf16)
      }

      ENTRY main {
        %param = f32[300, 500] parameter(0)
        ROOT %wrapped_fusion = f32[300, 500]
                               fusion(%param),
                               kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    operand_shape = (300, 500)

    # np.rand produces values in the range [0, 1).
    # Bias and scale it to get more representative values.
    param = base_utilities.create_literal_from_np(
        (np.random.rand(*operand_shape).astype(dtype) - 0.5) * 1e9
    )
    # Also make sure inf is preserved.
    np.asarray(param)[0, 0] = np.inf

    result = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    kernel_runner.call([param, result])

    # Based on the 7-bit fraction of bf16.
    bf_16_rtol = 2**-8
    np.testing.assert_allclose(result, param, rtol=bf_16_rtol)

  # Check that user provided nans and nans produced by internal ops are
  # preserved.
  def test_convert_f32_bf16_f32_nan(self):
    # [FUNCTION DOC]
    # Scenario: a fusion divides a float32 vector of three elements by itself, then converts to
    # bfloat16 and back. Setup: inputs one, zero and not-a-number. Action: run the emitted kernel.
    # Expected: the first output is one, and the other two are not-a-number (zero divided by zero
    # produced internally, and the user-provided NaN), showing NaNs survive the conversions.
    dtype = np.dtype(np.float32)

    hlo = """
      HloModule test_module

      fusion_computation {
        %param = f32[3] parameter(0)
        %div = f32[3] divide(%param, %param)
        %convert_bf16 = bf16[3] convert(%div)
        ROOT %convert_f32 = f32[3] convert(%convert_bf16)
      }

      ENTRY main {
        %param = f32[3] parameter(0)
        ROOT %wrapped_fusion = f32[3]
                               fusion(%param),
                               kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    operand_shape = (3,)

    param = base_utilities.create_literal_from_np(
        np.ndarray(operand_shape, dtype)
    )
    np_param = np.asarray(param)
    np_param[0] = 1
    np_param[1] = 0
    np_param[2] = np.nan

    result = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    kernel_runner.call([param, result])

    np_result = np.asarray(result)

    self.assertEqual(np_result[0], 1)
    self.assertTrue(np.isnan(np_result[1]))
    self.assertTrue(np.isnan(np_result[2]))

  # Check that a constant with a layout is respected.
  def test_constant_with_layout(self):
    # [FUNCTION DOC]
    # Scenario: a fusion with no parameters holds a two-by-two float32 constant with a
    # column-major layout and copies it into a row-major result. Setup: zero-filled result
    # literal. Action: run the emitted kernel. Expected: the result reads zero, one, two, three in
    # logical row-major order, i.e. the constant's logical values are preserved across the layout
    # change.
    dtype = np.dtype(np.float32)

    hlo = """
      HloModule test_module

      fusion_computation {
        %constant = f32[2, 2]{0, 1} constant({{0, 1}, {2, 3}})
        ROOT %result = f32[2, 2]{1, 0} copy(%constant)
      }

      ENTRY main {
        ROOT %wrapped_fusion =
          f32[2, 2] fusion(), kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    shape = (2, 2)

    result = base_utilities.create_literal_from_np(np.zeros(shape, dtype))

    kernel_runner.call([result])

    np_result = np.asarray(result)

    self.assertEqual(np_result[0, 0], 0)
    self.assertEqual(np_result[0, 1], 1)
    self.assertEqual(np_result[1, 0], 2)
    self.assertEqual(np_result[1, 1], 3)


class FusionEmitterTest(parameterized.TestCase):
  # [CLASS DOC]
  # Test case class covering general fusion emitter behavior beyond plain elementwise loops
  # (currently dead-code elimination around exponentials). No shared fixtures.

  def test_exp_nan_dce(self):
    # [FUNCTION DOC]
    # Scenario: a fusion multiplies a float64 vector of three elements by a constant vector of
    # one, zero and not-a-number, then applies the exponential. Setup: inputs one, zero and
    # not-a-number. Action: run the emitted kernel. Expected: outputs are e to the one, e to the
    # zero (one) and not-a-number, i.e. the emitted code keeps the exponential correct for NaN and
    # is not broken by dead-code elimination.
    dtype = np.dtype(np.float64)

    hlo = """
      HloModule test_module

      fusion_computation {
        %param = f64[3] parameter(0)
        %multiplier = f64[3] constant({1, 0, nan})
        %mult = f64[3] multiply(%param, %multiplier)
        ROOT %exp = f64[3] exponential(%mult) 
      }

      ENTRY main {
        %param = f64[3] parameter(0)
        ROOT %wrapped_fusion = f64[3]
                               fusion(%param),
                               kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    operand_shape = (3,)

    param = base_utilities.create_literal_from_np(
        np.ndarray(operand_shape, dtype)
    )
    np_param = np.asarray(param)
    np_param[0] = 1
    np_param[1] = 0
    np_param[2] = np.nan

    result = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    kernel_runner.call([param, result])

    np_result = np.asarray(result)

    self.assertAlmostEqual(np_result[0], np.exp(1))
    self.assertAlmostEqual(np_result[1], np.exp(0))
    self.assertTrue(np.isnan(np_result[2]))


if __name__ == "__main__":
  absltest.main()
