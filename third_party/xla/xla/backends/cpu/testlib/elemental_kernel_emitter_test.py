# [MODULE DOC]
# Tests for the XLA CPU elemental kernel emitter. Three parameterized test classes check that
# kernels emitted for (1) elementwise unary/binary/ternary opcodes such as sine, add, power,
# clamp, (2) comparison instructions in all directions and (3) map and reduce computations given
# as HLO text produce the same results as numpy, across shapes and dtypes. Each test builds or
# parses an HLO module, emits a kernel with the elemental emitter, JIT-compiles and runs it.
# Also defines helpers: an input generator, an erf wrapper for numpy and a frozen dataclass that
# pairs an HLO opcode with its numpy reference, input range and decimal precision.
# No import-time side effects beyond defining module-level aliases.
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

from collections.abc import Callable, Sequence
import dataclasses
import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu import testlib as testlib_cpu
from xla.backends.cpu.testlib import utilities
from xla.codegen import testlib as testlib_base
from xla.codegen.testlib.utilities import opcode_arity
from xla.python import xla_extension

HloOpcode = testlib_base.HloOpcode
create_literal = testlib_base.utilities.create_literal_from_np
HloInstruction = testlib_base.HloInstruction
ComparisonDirection = testlib_base.ComparisonDirection
_inf = float("inf")


def create_input(
    value_range: tuple[float, float],
    shape: Sequence[int],
    dtype: np.dtype,
    shuffle: bool = False,
) -> np.ndarray:
  # [FUNCTION DOC]
  # HELPER: builds deterministic (or shuffled) test input arrays for the elemental emitter tests.
  # Creates evenly spaced values between the two ends of value_range (a pair of floats,
  # inclusive) cast to dtype and reshaped to shape (a sequence of ints; empty means a scalar
  # with one element). If shuffle is true (default false) and the result is not a scalar, the
  # values are shuffled in place along the first axis using the global numpy random state, so
  # the output is then non-deterministic. Returns the array.
  # Note that an infinite range, as used for the is-finite test, yields non-finite values.
  size = np.prod(shape) if shape else 1
  result = np.linspace(
      value_range[0], value_range[1], size, dtype=dtype
  ).reshape(shape)

  if shuffle and (np.ndim(result) != 0):
    np.random.shuffle(result)

  return result


def np_erf(x):
  # [FUNCTION DOC]
  # HELPER: numpy reference for the error function used by the erf opcode test.
  # Applies the standard-library error function elementwise over the numpy array x and returns
  # an array with the same element type as x. Raises a type error for non-numeric input.
  return np.vectorize(math.erf, otypes=[x.dtype])(x)


@dataclasses.dataclass(frozen=True)
class ElementalHloOpcodeDef:
  # [CLASS DOC]
  # HELPER: immutable test-case descriptor for the elemental opcode tests.
  # KNOWN ISSUE: the np_op annotation puts an ellipsis inside the Callable argument list, which
  # is not a valid way to say variable number of arguments.
  # Fields: op, the HLO opcode under test; np_op, the numpy reference callable taking one array
  # per opcode operand; input_ranges, the (low, high) range of generated input values (default
  # minus one to one); decimal_precision, number of decimals for the comparison (default 6).
  # Supports unpacking into those four values and a short readable representation used in the
  # generated test names.
  op: HloOpcode
  np_op: Callable[[np.ndarray, ...], np.ndarray]
  input_ranges: tuple[float, float] = (-1.0, 1.0)
  decimal_precision: int = 6

  # For simple unpacking
  def __iter__(self):
    # [FUNCTION DOC]
    # Iterates over the four fields in order (opcode, numpy reference, input range, decimal
    # precision) so an instance can be unpacked like a sequence. Returns an iterator; no side
    # effects.
    return iter(
        (self.op, self.np_op, self.input_ranges, self.decimal_precision)
    )

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a short string made of the opcode name followed by the input range in parentheses.
    # This string appears in the generated parameterized test names. No side effects.
    return f"{self.op.name}({self.input_ranges})"


@parameterized.product(
    op_def=[
        ElementalHloOpcodeDef(HloOpcode.sine, np.sin),
        ElementalHloOpcodeDef(HloOpcode.cosine, np.cos),
        ElementalHloOpcodeDef(HloOpcode.tan, np.tan),
        ElementalHloOpcodeDef(HloOpcode.exponential, np.exp),
        ElementalHloOpcodeDef(HloOpcode.log, np.log, (0.01, 10.0)),
        ElementalHloOpcodeDef(HloOpcode.log_plus_one, np.log1p),
        ElementalHloOpcodeDef(HloOpcode.sqrt, np.sqrt),
        ElementalHloOpcodeDef(
            HloOpcode.rsqrt, lambda x: np.reciprocal(np.sqrt(x))
        ),
        ElementalHloOpcodeDef(HloOpcode.cbrt, np.cbrt),
        ElementalHloOpcodeDef(HloOpcode.power, np.pow),
        ElementalHloOpcodeDef(HloOpcode.add, np.add),
        ElementalHloOpcodeDef(HloOpcode.subtract, np.subtract),
        ElementalHloOpcodeDef(HloOpcode.multiply, np.multiply),
        ElementalHloOpcodeDef(HloOpcode.divide, np.divide),
        ElementalHloOpcodeDef(HloOpcode.maximum, np.maximum),
        ElementalHloOpcodeDef(HloOpcode.minimum, np.minimum),
        ElementalHloOpcodeDef(HloOpcode.sign, np.sign),
        ElementalHloOpcodeDef(HloOpcode.negate, np.negative),
        ElementalHloOpcodeDef(HloOpcode.is_finite, np.isfinite, (-_inf, _inf)),
        ElementalHloOpcodeDef(HloOpcode.ceil, np.ceil, (-10.0, 10.0)),
        ElementalHloOpcodeDef(HloOpcode.floor, np.floor, (-5.0, 5.0)),
        ElementalHloOpcodeDef(HloOpcode.tanh, np.tanh),
        ElementalHloOpcodeDef(HloOpcode.atan2, np.arctan2),
        ElementalHloOpcodeDef(HloOpcode.erf, np_erf),
        ElementalHloOpcodeDef(HloOpcode.exponential_minus_one, np.expm1),
        # TODO(willfroom): Update to use better inputs for the following.
        ElementalHloOpcodeDef(HloOpcode.clamp, np.clip),
        # TODO(willfroom): Add complex ops.
        # ElementalHloOpcodeDef(HloOpcode.complex, np.complex),
        # ElementalHloOpcodeDef(HloOpcode.real, np.real),
        # ElementalHloOpcodeDef(HloOpcode.imag, np.imag),
        # TODO(willfroom): go through ElementalIrEmitter interface and ensure
        # that all ops are implemented.
        # ...
    ],
    shape=[(4,), (4, 3), (4, 3, 10)],
    dtype=[np.dtype(np.float32), np.dtype(np.float64)],
)
class ElementalKernelRunnerTest(parameterized.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the power entry uses the numpy pow alias that exists only in numpy 2 and later,
  # and the is-finite entry uses an infinite input range, so its inputs are not meaningful
  # linearly spaced finite values.
  # Parameterized test case (decorated to run over a product of opcode definitions, shapes and
  # float32/float64 dtypes) checking elementwise opcodes against numpy through the elemental
  # kernel emitter. The decorator list covers math, arithmetic, min/max, sign, rounding, tanh,
  # atan2, erf and clamp. Stateless.

  def test_elemental_kernel_emitter(
      self,
      op_def: ElementalHloOpcodeDef,
      shape: tuple[int, ...],
      dtype: np.dtype,
  ):
    # [FUNCTION DOC]
    # Test: for one opcode definition, shape and dtype. Determines the opcode arity, creates that
    # many identical evenly spaced inputs in the definition's range, computes the numpy expected
    # output, builds HLO parameters and a variadic instruction for the opcode, builds the module,
    # emits a kernel with the elemental emitter (asserted non-null), saves the kernel IR text
    # (because the runner consumes the kernel definition), runs the kernel and asserts the output
    # equals the expected values to the definition's decimal precision, with the IR text as the
    # failure message. Fails if the opcode has no known arity.

    [op, np_op, input_ranges, decimal_precision] = op_def

    num_inputs = opcode_arity(op)
    self.assertIsNotNone(num_inputs)

    np_inputs = [
        create_input(input_ranges, shape, dtype) for _ in range(num_inputs)
    ]
    input_literals = [create_literal(input_array) for input_array in np_inputs]

    expected_output = np_op(*np_inputs)
    output_literal = create_literal(
        np.ndarray(shape, dtype=expected_output.dtype)
    )

    hlo_parameters = [
        HloInstruction.create_parameter(idx, literal.shape(), f"input_{idx}")
        for [idx, literal] in enumerate(input_literals)
    ]

    hlo_op = HloInstruction.create_variadic(
        output_literal.shape(), op, hlo_parameters
    )

    hlo_module, buffer_assignment = utilities.build_hlo_module(
        hlo_op, *hlo_parameters
    )
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

    emitter = testlib_cpu.ElementalKernelEmitter(
        hlo_module.get_root_instruction(),
        buffer_assignment,
        jit_compiler.get_target_machine(),
    )

    kernel_definition = emitter.emit_kernel_definition()
    self.assertIsNotNone(kernel_definition)

    # kernel_definition is consumed by the runner, so we need to save the IR
    # string before passing it to the runner.
    ir_string = str(kernel_definition.source())

    runner = testlib_cpu.KernelRunner.create(kernel_definition, jit_compiler)

    runner.call(list(itertools.chain(input_literals, [output_literal])))
    np.testing.assert_array_almost_equal(
        np.asarray(output_literal),
        expected_output,
        decimal=decimal_precision,
        err_msg=ir_string,
    )


@parameterized.product(
    op_def=[
        (ComparisonDirection.kEq, np.equal),
        (ComparisonDirection.kNe, np.not_equal),
        (ComparisonDirection.kGe, np.greater_equal),
        (ComparisonDirection.kGt, np.greater),
        (ComparisonDirection.kLe, np.less_equal),
        (ComparisonDirection.kLt, np.less),
    ],
    shape=[(4,), (4, 3), (4, 3, 10)],
    dtype=[
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
    ],
)
class ElementalComparisonKernelRunnerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case (over six comparison directions with their numpy equivalents, three
  # shapes and eleven integer and float dtypes) for compare instructions emitted by the
  # elemental kernel emitter. Stateless.

  def test_elemental_comparision_kernel_emitter(self, op_def, shape, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name misspells comparison; also relies on the numpy bool alias.
    # Test: compares two independently shuffled evenly spaced arrays (range 0..20 for unsigned
    # types, minus ten to ten otherwise) elementwise with the given direction. Builds a compare
    # instruction with a boolean output, emits and runs the kernel, and asserts the boolean
    # output exactly equals the numpy comparison.
    [direction, np_op] = op_def

    is_unsigned = np.issubdtype(dtype, np.unsignedinteger)
    value_range = (0.0, 20.0) if is_unsigned else (-10.0, 10.0)
    lhs_np = create_input(value_range, shape, dtype, shuffle=True)
    rhs_np = create_input(value_range, shape, dtype, shuffle=True)

    lhs_literal = create_literal(lhs_np)
    rhs_literal = create_literal(rhs_np)

    output_literal = create_literal(np.ndarray(shape, dtype=np.bool))

    lhs_param = HloInstruction.create_parameter(0, lhs_literal.shape(), "lhs")
    rhs_param = HloInstruction.create_parameter(1, rhs_literal.shape(), "rhs")

    hlo_op = HloInstruction.create_compare(
        output_literal.shape(), lhs_param, rhs_param, direction
    )

    hlo_module, buffer_assignment = utilities.build_hlo_module(
        hlo_op, lhs_param, rhs_param
    )
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

    emitter = testlib_cpu.ElementalKernelEmitter(
        hlo_module.get_root_instruction(),
        buffer_assignment,
        jit_compiler.get_target_machine(),
    )

    runner = testlib_cpu.KernelRunner.create(
        emitter.emit_kernel_definition(), jit_compiler
    )

    runner.call([lhs_literal, rhs_literal, output_literal])
    np.testing.assert_equal(
        np.asarray(output_literal),
        np_op(lhs_np, rhs_np),
    )


@parameterized.product(
    input_dimensions=[(4,), (4, 3), (4, 3, 10)],
    dtype=[
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
    ],
)
class HloModuleKernelRunnerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case (over three input shapes and eleven dtypes) for kernels emitted from
  # HLO text containing map and reduce instructions with sub-computations, via the elemental
  # kernel emitter. Stateless.

  def test_map(self, input_dimensions, dtype):
    # [FUNCTION DOC]
    # Test: parses an HLO module whose entry maps a doubling computation (multiply by constant
    # two) over an array of the given dimensions and dtype, emits and runs the elemental kernel
    # on a shuffled input in range 0..10, and asserts the output exactly equals twice the input.
    # The HLO text is built with string formatting of shapes, hence doubled braces in the text.
    scalar_shape = xla_extension.Shape.scalar_shape(dtype)
    shape = xla_extension.Shape.array_shape(dtype, input_dimensions)

    # Please note the double curly braces is to escape the python string
    # formatting.
    hlo = """
      HloModule test_map

      double {{
        a = {scalar_shape} parameter(0)
        b = {scalar_shape} constant(2)
        ROOT doubled = {scalar_shape} multiply(a, b)
      }}

      ENTRY main {{
        a = {shape} parameter(0)
        ROOT mapped = {shape} map(a), to_apply=double
      }}
    """.format(scalar_shape=scalar_shape, shape=shape)

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

    emitter = testlib_cpu.ElementalKernelEmitter(
        hlo_module.get_root_instruction(),
        buffer_assignment,
        jit_compiler.get_target_machine(),
    )

    input_np = create_input([0, 10], input_dimensions, dtype, shuffle=True)

    input_literal = create_literal(input_np)

    output_literal = xla_extension.Literal(shape)

    runner = testlib_cpu.KernelRunner.create(
        emitter.emit_kernel_definition(), jit_compiler
    )

    runner.call([input_literal, output_literal])

    np.testing.assert_equal(
        np.asarray(output_literal),
        input_np * 2,
    )

  def test_reduce(self, input_dimensions, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: reduction dimension sets are only built with sizes from one up to rank minus
    # one, so full reductions are never tested and rank-one inputs run no iterations at all
    # (the test passes without checking anything).
    # KNOWN ISSUE: for the 8-bit signed dtype the output shape is declared as 16-bit signed while
    # the reducer computation and operands stay 8-bit, so the HLO types disagree.
    # Test: for each non-empty proper subset of dimensions, parses an HLO module that reduces the
    # input with an add computation and a scalar initial value parameter, emits and runs the
    # elemental kernel, and asserts the output is within 3 units in the last place of the numpy
    # add-reduction over those axes with the initial value. Inputs are evenly spaced 0..10.
    # Iterate over all combinations of reduce dimensions.
    for reduce_dimensions in itertools.chain.from_iterable(
        itertools.combinations(range(len(input_dimensions)), r)
        for r in range(1, len(input_dimensions))
    ):
      scalar_shape = xla_extension.Shape.scalar_shape(dtype)
      input_shape = xla_extension.Shape.array_shape(dtype, input_dimensions)

      output_dimensions = [
          dim
          for idx, dim in enumerate(input_dimensions)
          if idx not in reduce_dimensions
      ]
      # Result can overflow in int8 (which results in undefined behavior),
      # so we use int16 instead.
      output_dtype = np.dtype(np.int16) if (dtype == np.int8) else dtype
      output_shape = xla_extension.Shape.array_shape(
          output_dtype, output_dimensions
      )

      # Please note the double curly braces is to escape the python string
      # formatting.
      hlo = """
        HloModule test_reduce

        add_method {{
          a = {scalar_shape} parameter(0)
          b = {scalar_shape} parameter(1)
          ROOT add = {scalar_shape} add(a, b)
        }}

        ENTRY main {{
          array = {input_shape} parameter(0)
          initial_value = {scalar_shape} parameter(1)
          ROOT reduced = {output_shape} reduce(array, initial_value),
            dimensions={{{reduce_dimensions}}}, to_apply=add_method
        }}
      """.format(
          scalar_shape=scalar_shape,
          input_shape=input_shape,
          reduce_dimensions=",".join(map(str, reduce_dimensions)),
          output_shape=output_shape,
      )

      hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
      jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

      emitter = testlib_cpu.ElementalKernelEmitter(
          hlo_module.get_root_instruction(),
          buffer_assignment,
          jit_compiler.get_target_machine(),
      )

      input_np = create_input([0, 10], input_dimensions, dtype)
      input_literal = create_literal(input_np)

      initial_value_np = create_input([0, 10], (), dtype)
      initial_value_literal = create_literal(initial_value_np)

      output_literal = xla_extension.Literal(output_shape)

      runner = testlib_cpu.KernelRunner.create(
          emitter.emit_kernel_definition(), jit_compiler
      )

      runner.call([input_literal, initial_value_literal, output_literal])

      np.testing.assert_array_almost_equal_nulp(
          np.asarray(output_literal),
          np.add.reduce(
              input_np, axis=reduce_dimensions, initial=initial_value_np
          ),
          nulp=3,
      )


if __name__ == "__main__":
  absltest.main()
