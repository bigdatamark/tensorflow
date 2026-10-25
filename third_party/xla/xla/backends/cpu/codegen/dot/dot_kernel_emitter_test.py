# [MODULE DOC]
# Tests for the XLA CPU dot (matrix/vector product) kernel emitter. Each test builds a small HLO
# module containing a single dot instruction, emits a kernel with either the generic elemental
# emitter or the dedicated dot emitter, JIT-compiles it, runs it on literals and compares the
# result with numpy within a maximum ulp tolerance. Tests are parameterized over emitter type,
# operand shapes and many integer/float dtypes.
# Import-time side effect: sets the XLA_FLAGS environment variable to disable multi-threaded
# eigen, because the dot emitter refuses to emit for some shapes when it is enabled.
# Also defines module-level aliases and the list of emitters and dtypes under test.
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

from collections.abc import Sequence
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu import testlib as testlib_cpu
from xla.backends.cpu.testlib import utilities
from xla.codegen import testlib as testlib_base
from xla.codegen.testlib import utilities as base_utilities

# We have some checks in the dot emitter which will fail to emit for certain
# shapes if multi-threading is enabled.
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

create_literal = base_utilities.create_literal_from_np
HloInstruction = testlib_base.HloInstruction
HloOpcode = testlib_base.HloOpcode


def create_input(
    value_range: tuple[float, float],
    shape: Sequence[int],
    dtype: np.dtype,
) -> np.ndarray:
  # [FUNCTION DOC]
  # HELPER: builds deterministic test input arrays for the dot kernel tests.
  # Creates an evenly spaced sequence of values between the two ends of value_range (inclusive),
  # cast to the given dtype, and reshapes it to the given shape. The number of elements is the
  # product of the shape dimensions (one element for an empty shape, i.e. a scalar).
  # Inputs: value_range is a (start, stop) pair of floats; shape is a sequence of ints; dtype is
  # a numpy dtype. Returns the numpy array. No side effects; reshape errors propagate from numpy.
  size = np.prod(shape) if shape else 1
  result = np.linspace(
      value_range[0], value_range[1], size, dtype=dtype
  ).reshape(shape)

  return result


emitter_types = [
    testlib_cpu.ElementalKernelEmitter,
    testlib_cpu.DotKernelEmitter,
]


dtypes_to_test = [
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.uint64),
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
]


class DotKernelTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case covering the CPU dot kernel emitters. It holds no state; each test
  # method builds its own HLO module, emitter, JIT compiler and kernel runner, and compares
  # against numpy for the combinations of emitter type, shape and dtype.

  @parameterized.product(
      emitter_type=emitter_types,
      rhs_shape=[(4,), (4, 3), (4, 3, 10), (500, 10, 123)],
      dtype=dtypes_to_test,
  )
  def test_vector_matrix_dot(self, emitter_type, rhs_shape, dtype):
    # [FUNCTION DOC]
    # Test: vector-times-tensor dot. The left operand is a vector whose length equals the first
    # dimension of rhs_shape; the right operand has shape rhs_shape. Both are filled with evenly
    # spaced values in 0..20 of the given dtype. The dot contracts dimension 0 of both operands,
    # so the output has the remaining dimensions of the right operand.
    # Builds parameters, a dot instruction, an HLO module and buffer assignment, JIT-compiles,
    # creates the emitter of emitter_type, emits the kernel and runs it on the literals.
    # Asserts that the output equals numpy tensordot over axis 0 within 10 ulp.
    # Parameters come from the parameterized product of emitter type, rhs shape and dtype.
    value_range = (0.0, 20.0)
    lhs_np = create_input(value_range, rhs_shape[0], dtype)
    rhs_np = create_input(value_range, rhs_shape, dtype)

    lhs_literal = create_literal(lhs_np)
    rhs_literal = create_literal(rhs_np)

    output_literal = create_literal(np.ndarray(rhs_shape[1:], dtype=dtype))

    lhs_param = HloInstruction.create_parameter(0, lhs_literal.shape(), "lhs")
    rhs_param = HloInstruction.create_parameter(1, rhs_literal.shape(), "rhs")

    dot_dimension_numbers = testlib_base.DotDimensionNumbers([0], [0])
    hlo_op = HloInstruction.create_dot(
        output_literal.shape(), lhs_param, rhs_param, dot_dimension_numbers
    )

    hlo_module, buffer_assignment = utilities.build_hlo_module(
        hlo_op, lhs_param, rhs_param
    )
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

    emitter = emitter_type(
        hlo_module.get_root_instruction(),
        buffer_assignment,
        jit_compiler.get_target_machine(),
    )

    runner = testlib_cpu.KernelRunner.create(
        emitter.emit_kernel_definition(), jit_compiler
    )

    runner.call([lhs_literal, rhs_literal, output_literal])

    np_result = np.tensordot(lhs_np, rhs_np, axes=(0, 0))
    np.testing.assert_array_max_ulp(
        np.asarray(output_literal),
        np_result,
        maxulp=10,
    )

  @parameterized.product(
      emitter_type=emitter_types,
      shapes=[
          ((1, 1), (1, 1)),
          ((1, 1), (1, 10)),
          ((2, 2), (2, 2)),
          ((2, 2), (2, 3)),
          ((10, 10), (10, 10)),
          ((15, 13), (13, 17)),
      ],
      dtype=dtypes_to_test,
  )
  def test_matrix_multiplication(self, emitter_type, shapes, dtype):
    # [FUNCTION DOC]
    # Test: matrix-times-matrix dot for several small and medium shape pairs. Contracts the
    # last dimension of the left operand with the first of the right operand; the output shape
    # is the left shape without its last dimension followed by the right shape without its first.
    # Skips (via the test framework skip) the float16 case for the dedicated dot emitter, which
    # does not support it. Otherwise builds the HLO module, emits and JIT-compiles the kernel
    # with the given emitter type, runs it, and asserts the result matches numpy matrix
    # multiplication within 10 ulp. Parameters: emitter type, pair of shapes, dtype.
    if dtype == np.float16 and emitter_type is testlib_cpu.DotKernelEmitter:
      self.skipTest("float16 is not supported by the dot emitter")

    value_range = (0.0, 20.0)
    lhs_np = create_input(value_range, shapes[0], dtype)
    rhs_np = create_input(value_range, shapes[1], dtype)

    lhs_literal = create_literal(lhs_np)
    rhs_literal = create_literal(rhs_np)

    output_shape = shapes[0][:-1] + shapes[1][1:]
    output_literal = create_literal(np.ndarray(output_shape, dtype=dtype))

    lhs_param = HloInstruction.create_parameter(0, lhs_literal.shape(), "lhs")
    rhs_param = HloInstruction.create_parameter(1, rhs_literal.shape(), "rhs")

    dot_dimension_numbers = testlib_base.DotDimensionNumbers([1], [0])
    hlo_op = HloInstruction.create_dot(
        output_literal.shape(), lhs_param, rhs_param, dot_dimension_numbers
    )

    hlo_module, buffer_assignment = utilities.build_hlo_module(
        hlo_op, lhs_param, rhs_param
    )
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

    emitter = emitter_type(
        hlo_module.get_root_instruction(),
        buffer_assignment,
        jit_compiler.get_target_machine(),
    )

    kernel_definition = emitter.emit_kernel_definition()
    runner = testlib_cpu.KernelRunner.create(kernel_definition, jit_compiler)

    runner.call([lhs_literal, rhs_literal, output_literal])

    np_result = lhs_np @ rhs_np
    np.testing.assert_array_max_ulp(
        np.asarray(output_literal),
        np_result,
        maxulp=10,
    )


if __name__ == "__main__":
  absltest.main()
