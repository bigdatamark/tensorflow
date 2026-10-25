# [MODULE DOC]
# Tests for lowering tiled (xtile dialect) MLIR kernels on the XLA CPU backend. Each test supplies
# a small MLIR module text that extracts tiles from input buffers, applies operations (transpose,
# add, dot, reduce, broadcast, reshape, compare) and inserts the result into an output buffer;
# the shared compare_kernel helper emits the kernel, JIT-compiles it, runs it on generated inputs
# and compares the output with a numpy reference function.
# Contains the InputSpec helper class, the random-array and comparison helpers and the test class.
# No import-time side effects beyond defining the create_literal alias.
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

from collections.abc import Callable, Iterable
from typing import Optional

from absl.testing import absltest
import numpy as np

from xla.backends.cpu import testlib as cpu_testlib
from xla.codegen import testlib as base_testlib
from xla.codegen.testlib import utilities as testlib_utilities

create_literal = testlib_utilities.create_literal_from_np


class InputSpec:
  # [CLASS DOC]
  # HELPER: small value holder used by compare_kernel to describe the inputs of a test kernel.
  # Stores only the shape (tuple of dimension sizes) of one input array in the public attribute
  # shape. Used to generate the concrete input data for that input.

  def __init__(self, shape: tuple[int, ...]):
    # [FUNCTION DOC]
    # Constructor. Takes the shape (tuple of non-negative ints) of the input array and stores it
    # unchanged in the shape attribute. Returns nothing, performs no validation.
    """Initializes the InputSpec.

    Args:
      shape: The shape of the input array.
    """
    self.shape = shape


def get_random_array(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
  # [FUNCTION DOC]
  # HELPER: produces filler data for the output buffer in compare_kernel.
  # Returns a numpy array of the given shape with values drawn uniformly from -5 to 5 using a
  # freshly seeded (non-deterministic) random generator, then cast to dtype. Casting truncates
  # for integer types and maps values to 0/1-like booleans for bool. No other side effects.
  rng = np.random.default_rng()
  return rng.uniform(low=-5, high=5, size=shape).astype(dtype)


def compare_kernel(
    ir: str,
    kernel_name: str,
    num_workgroups: int,
    input_specs: Iterable[InputSpec],
    output_shape: tuple[int, ...],
    dtype,
    expected_output: Callable[[np.ndarray, ...], np.ndarray],
    maxulp: Optional[int] = None,
) -> None:
  # [FUNCTION DOC]
  # HELPER: shared driver used by every test in this file to emit, run and verify a tiled kernel.
  # KNOWN ISSUE: the type annotation of expected_output uses an ellipsis inside the argument list
  # of Callable, which is not a valid Callable signature for a variable number of arguments.
  # KNOWN ISSUE: relies on the numpy bool alias for dtype comparison, which only exists in newer
  # numpy versions.
  # Inputs: ir is the MLIR module text; kernel_name is the entry function name inside it;
  # num_workgroups is the number of workgroups (used as the x dimension of the grid, with y and z
  # equal to one); input_specs is an iterable of InputSpec describing each input array;
  # output_shape is the shape of the output buffer; dtype is the numpy element type used for
  # inputs and output; expected_output is a callback taking the numpy inputs and returning the
  # reference result; maxulp is optional: if absent the result must match exactly, otherwise it
  # must match within that many units in the last place.
  # Behavior: emits the kernel with the MLIR test kernel emitter, creates a kernel runner with a
  # JIT compiler built from a default HLO module config, generates inputs as ascending ranges
  # (for bool, alternating 0/1 values), fills the output with random data so unwritten elements
  # are detected, runs the kernel with inputs followed by the output, and compares.
  # Returns nothing; raises an assertion error on mismatch. Compilation errors propagate.
  mlir_emitter = cpu_testlib.MlirTestKernelEmitter(
      ir, kernel_name, (num_workgroups, 1, 1)
  )
  kernel_definition = mlir_emitter.emit_kernel_definition()

  runner = cpu_testlib.KernelRunner.create(
      kernel_definition,
      cpu_testlib.JitCompiler(base_testlib.HloModuleConfig()),
  )

  def get_input(spec: InputSpec):
    # [FUNCTION DOC]
    # HELPER: builds the deterministic input array for one InputSpec inside compare_kernel.
    # Returns an array of the spec's shape holding ascending integers 0..n-1 cast to the
    # enclosing dtype; for the bool dtype it instead returns alternating 0 and 1 values stored
    # as 8-bit integers. Uses the dtype of the enclosing call; no side effects.
    if dtype == np.bool:
      return (
          np.arange(np.prod(spec.shape), dtype=np.int8).reshape(spec.shape) % 2
      )
    return np.arange(np.prod(spec.shape), dtype=dtype).reshape(spec.shape)

  inputs = [get_input(spec) for spec in input_specs]

  input_tensors = [create_literal(input) for input in inputs]
  # Use a random array as the output to ensure all values are written to.
  output_tensor = create_literal(get_random_array(output_shape, dtype))
  runner.call(input_tensors + [output_tensor])

  output_np = np.asarray(output_tensor)
  expected_output_np = expected_output(*inputs)
  if maxulp is None:
    np.testing.assert_array_equal(output_np, expected_output_np)
  else:
    np.testing.assert_array_max_ulp(
        output_np, expected_output_np, maxulp=maxulp
    )


class XtileLoweringTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case covering lowering and execution of xtile-dialect kernels on the CPU backend. Stateless;
  # each test method embeds an MLIR module and a numpy reference and delegates to compare_kernel.

  def test_slice(self):
    # [FUNCTION DOC]
    # Test: a single tile of 64x64 is extracted from a 5x5 float32 buffer (larger than the buffer,
    # so the extract and insert must be masked), transposed, and inserted into a 5x5 output.
    # Runs with one workgroup and asserts the output equals the transposed input exactly.
    # Check that masked extract / insert works.
    ir = """
      module @tiled_slice {
        xtile.entry_func @tiled_slice(
            %input: memref<5x5xf32>,
            %output: memref<5x5xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %offset = arith.constant 0 : index
          %input_tile = xtile.extract %input[%offset, %offset][64, 64][1, 1] : memref<5x5xf32> -> tensor<64x64xf32>
          %transposed_tile = stablehlo.transpose %input_tile, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
          xtile.insert %transposed_tile into %output[%offset, %offset][64, 64][1, 1] : tensor<64x64xf32> -> memref<5x5xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "tiled_slice",
        1,
        [InputSpec((5, 5))],
        (5, 5),
        np.float32,
        lambda arg: arg.transpose(),
    )

  def test_strided(self):
    # [FUNCTION DOC]
    # Test: extracts a 4x32 tile with strides 21 and 2 from a 64x64 float32 buffer and inserts it
    # densely into a 4x32 output. One workgroup. Asserts the output equals the input sliced with
    # those strides, exactly.
    ir = """
      module @tiled_slice {
        xtile.entry_func @tiled_slice(
            %input: memref<64x64xf32>,
            %output: memref<4x32xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %input_tile = xtile.extract %input[%tile_id, %tile_id][4, 32][21, 2] : memref<64x64xf32> -> tensor<4x32xf32>
          xtile.insert %input_tile into %output[%tile_id, %tile_id][4, 32][1, 1] : tensor<4x32xf32> -> memref<4x32xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "tiled_slice",
        1,
        [InputSpec((64, 64))],
        (4, 32),
        np.float32,
        lambda arg: arg[::21, ::2],
    )

  def test_transpose(self):
    # [FUNCTION DOC]
    # Test: tiled transpose of a 4096x4096 float32 buffer. Each of 262144 tiles (8x8) computes its
    # offsets from the tile id through indexing maps, transposes the tile and writes it at the
    # swapped offsets. Run with 8 workgroups; asserts exact equality with the numpy transpose.
    ir = """
      module @tiled_transpose {
        xtile.entry_func @tiled_transpose(
            %input: memref<4096x4096xf32>,
            %output: memref<4096x4096xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:262144, tiles_per_workgroup:32768>} {
          %offset_0 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid mod 512) * 8), domain: tid in [0, 262144]">(%tile_id)
          %offset_1 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid floordiv 512) * 8), domain: tid in [0, 262144]">(%tile_id)
          %input_tile = xtile.extract %input[%offset_0, %offset_1][8, 8][1, 1] : memref<4096x4096xf32> -> tensor<8x8xf32>
          %transposed_tile = stablehlo.transpose %input_tile, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
          xtile.insert %transposed_tile into %output[%offset_1, %offset_0][8, 8][1, 1] : tensor<8x8xf32> -> memref<4096x4096xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "tiled_transpose",
        8,
        [InputSpec((4096, 4096))],
        (4096, 4096),
        np.float32,
        lambda arg: arg.transpose(),
    )

  def test_add_tranpose(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test name has a typo (tranpose instead of transpose); the same misspelling
    # is used in the MLIR module and kernel names, so it is harmless but misleading.
    # Test: for a 4096x4096 float32 buffer, each 8x8 tile transposes the tile at offsets
    # (a, b), adds the tile at the swapped offsets (b, a), and writes the sum at (b, a).
    # Run with 8 workgroups; asserts the output equals the input plus its transpose, exactly.
    ir = """
      module @add_tranpose {
        xtile.entry_func @add_tranpose(
            %input: memref<4096x4096xf32>,
            %output: memref<4096x4096xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:262144, tiles_per_workgroup:32768>} {
          %offset_0 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid mod 512) * 8), domain: tid in [0, 262144]">(%tile_id)
          %offset_1 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid floordiv 512) * 8), domain: tid in [0, 262144]">(%tile_id)
          %input_tile_0 = xtile.extract %input[%offset_0, %offset_1][8, 8][1, 1] : memref<4096x4096xf32> -> tensor<8x8xf32>
          %input_tile_1 = xtile.extract %input[%offset_1, %offset_0][8, 8][1, 1] : memref<4096x4096xf32> -> tensor<8x8xf32>
          %transposed_tile = stablehlo.transpose %input_tile_0, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
          %added_tile = arith.addf %input_tile_1, %transposed_tile : tensor<8x8xf32>
          xtile.insert %added_tile into %output[%offset_1, %offset_0][8, 8][1, 1] : tensor<8x8xf32> -> memref<4096x4096xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "add_tranpose",
        8,
        [InputSpec((4096, 4096))],
        (4096, 4096),
        np.float32,
        lambda arg: arg + arg.transpose(),
    )

  def test_dot_single_tile(self):
    # [FUNCTION DOC]
    # Test: one tile computes a general dot of an 8x16 and a 16x8 float32 operand (contracting
    # dimension 1 with 0) into an 8x8 output. One workgroup. Asserts the result matches numpy
    # matrix multiplication within 5 ulp.
    ir = """
      module @dot_single_tile {
        xtile.entry_func @dot_single_tile(
            %lhs: memref<8x16xf32>,
            %rhs: memref<16x8xf32>,
            %output: memref<8x8xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %offset = arith.constant 0 : index
          %lhs_tile = xtile.extract %lhs[%offset, %offset][8, 16][1, 1] : memref<8x16xf32> -> tensor<8x16xf32>
          %rhs_tile = xtile.extract %rhs[%offset, %offset][16, 8][1, 1] : memref<16x8xf32> -> tensor<16x8xf32>
          %result = stablehlo.dot_general %lhs_tile, %rhs_tile, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
          xtile.insert %result into %output[%offset, %offset][8, 8][1, 1] : tensor<8x8xf32> -> memref<8x8xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "dot_single_tile",
        1,
        [InputSpec((8, 16)), InputSpec((16, 8))],
        (8, 8),
        np.float32,
        lambda lhs, rhs: lhs @ rhs,
        maxulp=5,
    )

  def test_dot_scalar_output(self):
    # [FUNCTION DOC]
    # Test: a dot that contracts both dimensions of an 8x16 and a 16x8 float32 operand (pairing
    # dimension 1 with 0 and 0 with 1), giving a scalar result inserted into a rank-zero output.
    # One workgroup. Asserts the result matches the numpy tensor dot over those axes within 8 ulp.
    ir = """
      module @test_dot_scalar_output {
        xtile.entry_func @test_dot_scalar_output(
            %lhs: memref<8x16xf32>,
            %rhs: memref<16x8xf32>,
            %output: memref<f32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %offset = arith.constant 0 : index
          %lhs_tile = xtile.extract %lhs[%offset, %offset][8, 16][1, 1] : memref<8x16xf32> -> tensor<8x16xf32>
          %rhs_tile = xtile.extract %rhs[%offset, %offset][16, 8][1, 1] : memref<16x8xf32> -> tensor<16x8xf32>
          %result = stablehlo.dot_general %lhs_tile, %rhs_tile, contracting_dims = [1, 0] x [0, 1] : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<f32>
          xtile.insert %result into %output[][][] : tensor<f32> -> memref<f32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "test_dot_scalar_output",
        1,
        [InputSpec((8, 16)), InputSpec((16, 8))],
        (),
        np.float32,
        lambda lhs, rhs: np.tensordot(lhs, rhs, axes=[[1, 0], [0, 1]]),
        maxulp=8,
    )

  def test_dot_fusion_single_tile(self):
    # [FUNCTION DOC]
    # Test: fused computation in one tile: two 8x16 float32 inputs are added, the sum is
    # multiplied by a 16-element vector (taken from a 16x1 buffer) via a dot, and the hyperbolic
    # tangent of the 8-element result is written to an 8x1 output. One workgroup. Asserts the
    # result matches the numpy equivalent within 5 ulp.
    ir = """
      module @dot_fusion_single_tile {
        xtile.entry_func @dot_fusion_single_tile(
            %lhs_0: memref<8x16xf32>,
            %lhs_1: memref<8x16xf32>,
            %rhs: memref<16x1xf32>,
            %output: memref<8x1xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %offset = arith.constant 0 : index
          %lhs_0_tile = xtile.extract %lhs_0[%offset, %offset][8, 16][1, 1] : memref<8x16xf32> -> tensor<8x16xf32>
          %lhs_1_tile = xtile.extract %lhs_1[%offset, %offset][8, 16][1, 1] : memref<8x16xf32> -> tensor<8x16xf32>
          %add_lhs = arith.addf %lhs_0_tile, %lhs_1_tile : tensor<8x16xf32>
          %rhs_tile = xtile.extract %rhs[%offset, %offset][16, 1][1, 1] : memref<16x1xf32> -> tensor<16xf32>
          %result = stablehlo.dot_general %add_lhs, %rhs_tile, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16xf32>) -> tensor<8xf32>
          %tanh_result = math.tanh %result : tensor<8xf32>
          xtile.insert %tanh_result into %output[%offset, %offset][8, 1][1, 1] : tensor<8xf32> -> memref<8x1xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "dot_fusion_single_tile",
        1,
        [
            InputSpec((8, 16)),
            InputSpec((8, 16)),
            InputSpec((16, 1)),
        ],
        (8, 1),
        np.float32,
        lambda lhs_0, lhs_1, rhs: np.tanh((lhs_0 + lhs_1) @ rhs),
        maxulp=5,
    )

  def test_reduction_add_inner(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the MLIR module declares float32 buffers but the test runs the comparison with
    # the 32-bit integer dtype, so the generated integer inputs and output do not match the
    # kernel's declared element type; it also uses a 1024-row buffer with 4 workgroups although
    # the tiling info declares 128 tiles of 8 rows.
    # Test: each tile reduces an 8x32 slice over its inner (last) dimension with addition and
    # inserts an 8-element result into a 1024-element output. The init value is taken from a
    # scalar buffer. Asserts exact equality with the numpy row sums plus the init value.
    ir = """
      module @reduction_add_inner {
        xtile.entry_func @reduction_add_inner(
            %input: memref<1024x32xf32>,
            %init: memref<f32>,
            %output: memref<1024xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:128, tiles_per_workgroup:32>} {
          %c_0 = arith.constant 0 : index
          %c_8 = arith.constant 8 : index
          %init_tile = xtile.extract %init[][][] : memref<f32> -> tensor<f32>
          %index = arith.muli %tile_id, %c_8 : index
          %input_tile = xtile.extract %input[%index, %c_0][8, 32][1, 1] : memref<1024x32xf32> -> tensor<8x32xf32>
          %result = stablehlo.reduce(%input_tile init: %init_tile)
                    applies stablehlo.add across dimensions = [1]
                    : (tensor<8x32xf32>, tensor<f32>) -> tensor<8xf32>
          xtile.insert %result into %output[%index][8][1] : tensor<8xf32> -> memref<1024xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "reduction_add_inner",
        4,
        [InputSpec((1024, 32)), InputSpec((1,))],
        (1024,),
        np.int32,
        lambda input, init: np.sum(input, axis=1) + init,
    )

  def test_reduction_add_outer(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the numpy reference ignores the init value (it only sums over axis 0), which
    # matches the result only because the generated init input is zero.
    # Test: each of 4 tiles reduces a 1024x8 slice of a 1024x32 float32 buffer over its outer
    # (first) dimension using an explicit adding reducer region, writing an 8-element piece of
    # a 32-element output. Four workgroups. Asserts exact equality with the numpy column sums.
    ir = """
      module @reduction_add_outer {
        xtile.entry_func @reduction_add_outer(
            %input: memref<1024x32xf32>,
            %init: memref<f32>,
            %output: memref<32xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:4, tiles_per_workgroup:1>} {
          %c_0 = arith.constant 0 : index
          %c_8 = arith.constant 8 : index
          %init_tile = xtile.extract %init[][][] : memref<f32> -> tensor<f32>
          %index = arith.muli %tile_id, %c_8 : index
          %input_tile = xtile.extract %input[%c_0, %index][1024, 8][1, 1] : memref<1024x32xf32> -> tensor<1024x8xf32>
          %result = stablehlo.reduce(%input_tile init: %init_tile)
                    across dimensions = [0]
                    : (tensor<1024x8xf32>, tensor<f32>) -> tensor<8xf32>
            reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
              %add = arith.addf %arg0, %arg1 : tensor<f32>
              stablehlo.return %add : tensor<f32>
            }
          xtile.insert %result into %output[%index][8][1] : tensor<8xf32> -> memref<32xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "reduction_add_outer",
        4,
        [InputSpec((1024, 32)), InputSpec((1,))],
        (32,),
        np.float32,
        lambda input, init: np.sum(input, axis=0),
    )

  def test_reduction_middle(self):
    # [FUNCTION DOC]
    # Test: reduces the middle dimension of an 8x4x2 float32 tile with an adding reducer, giving
    # an 8x2 output. One workgroup. The init value is ignored by the numpy reference (it is zero
    # in the generated data). Asserts exact equality with the numpy sum over axis 1.
    ir = """
      module @reduction_add_middle {
        xtile.entry_func @reduction_add_middle(
            %input: memref<8x4x2xf32>,
            %init: memref<f32>,
            %output: memref<8x2xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %init_val = xtile.extract %init[][][] : memref<f32> -> tensor<f32>
          %input_tile = xtile.extract %input[%tile_id, %tile_id, %tile_id][8, 4, 2][1, 1, 1] : memref<8x4x2xf32> -> tensor<8x4x2xf32>
          %result = stablehlo.reduce(%input_tile init: %init_val)
                    across dimensions = [1]
                    : (tensor<8x4x2xf32>, tensor<f32>) -> tensor<8x2xf32>
            reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
              %add = arith.addf %arg0, %arg1 : tensor<f32>
              stablehlo.return %add : tensor<f32>
            }
          xtile.insert %result into %output[%tile_id, %tile_id][8, 2][1, 1] : tensor<8x2xf32> -> memref<8x2xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "reduction_add_middle",
        1,
        [InputSpec((8, 4, 2)), InputSpec((1,))],
        (8, 2),
        np.float32,
        lambda input, init: np.sum(input, axis=1),
    )

  def test_reduction_outer_inner(self):
    # [FUNCTION DOC]
    # Test: reduces the first and last dimensions of an 8x4x2 float32 tile at once with an adding
    # reducer, giving a 4-element output. One workgroup. Asserts exact equality with the numpy
    # sum over axes 0 and 2.
    ir = """
      module @reduction_add_outer_inner {
        xtile.entry_func @reduction_add_outer_inner(
            %input: memref<8x4x2xf32>,
            %init: memref<f32>,
            %output: memref<4xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %init_val = xtile.extract %init[][][] : memref<f32> -> tensor<f32>
          %input_tile = xtile.extract %input[%tile_id, %tile_id, %tile_id][8, 4, 2][1, 1, 1] : memref<8x4x2xf32> -> tensor<8x4x2xf32>
          %result = stablehlo.reduce(%input_tile init: %init_val)
                    across dimensions = [0, 2]
                    : (tensor<8x4x2xf32>, tensor<f32>) -> tensor<4xf32>
            reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
              %add = arith.addf %arg0, %arg1 : tensor<f32>
              stablehlo.return %add : tensor<f32>
            }
          xtile.insert %result into %output[%tile_id][4][1] : tensor<4xf32> -> memref<4xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "reduction_add_outer_inner",
        1,
        [InputSpec((8, 4, 2)), InputSpec((1,))],
        (4,),
        np.float32,
        lambda input, init: np.sum(input, axis=(0, 2)),
    )

  def test_broadcast_in_dim_inner(self):
    # [FUNCTION DOC]
    # Test: broadcasts a 4-element float32 tile into a 32x4 tile, mapping the input to the last
    # dimension, and writes it to a 32x4 output. One workgroup. Asserts the output equals the
    # numpy broadcast of the input to that shape.
    ir = """
      module @broadcast_in_dim_inner {
        xtile.entry_func @broadcast_in_dim_inner(
            %input: memref<4xf32>,
            %output: memref<32x4xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %input_tile = xtile.extract %input[%tile_id][4][1] : memref<4xf32> -> tensor<4xf32>
          %result = stablehlo.broadcast_in_dim %input_tile, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
          xtile.insert %result into %output[%tile_id, %tile_id][32,4][1,1] : tensor<32x4xf32> -> memref<32x4xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "broadcast_in_dim_inner",
        1,
        [InputSpec((4,))],
        (32, 4),
        np.float32,
        lambda input: np.broadcast_to(input, (32, 4)),
    )

  def test_broadcast_in_dim_outer(self):
    # [FUNCTION DOC]
    # Test: broadcasts a 4-element float32 tile into a 4x32 tile, mapping the input to the first
    # dimension, and writes it to a 4x32 output. One workgroup. Asserts the output equals the
    # transpose of the numpy broadcast of the input to 32x4.
    ir = """
      module @broadcast_in_dim_outer {
        xtile.entry_func @broadcast_in_dim_outer(
            %input: memref<4xf32>,
            %output: memref<4x32xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %input_tile = xtile.extract %input[%tile_id][4][1] : memref<4xf32> -> tensor<4xf32>
          %result = stablehlo.broadcast_in_dim %input_tile, dims = [0] : (tensor<4xf32>) -> tensor<4x32xf32>
          xtile.insert %result into %output[%tile_id, %tile_id][4,32][1,1] : tensor<4x32xf32> -> memref<4x32xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "broadcast_in_dim_outer",
        1,
        [InputSpec((4,))],
        (4, 32),
        np.float32,
        lambda input: np.transpose(np.broadcast_to(input, (32, 4))),
    )

  def test_i1_reshape_transpose(self):
    # [FUNCTION DOC]
    # Test: boolean (i1) handling in a bitcast-copy-like fusion. Extracts a 2x4 tile of 8-bit
    # integers from a 2x3 buffer (masked out of bounds), compares it with zero to get booleans,
    # reshapes and transposes the booleans, widens them back to 8-bit integers, and inserts them
    # into a 2x3x1 output. One workgroup, bool dtype (inputs alternate 0 and 1). Asserts the
    # output equals the input, exactly.
    ir = """
      module @__compute_module_bitcast_copy_fusion {
        xtile.entry_func @bitcast_copy_fusion(
            %arg0: memref<2x3xi8>, %arg1: memref<2x3x1xi8>,
            %arg2: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count : 1, tiles_per_workgroup : 1>} {
          %c0 = arith.constant 0 : index
          %1 = xtile.extract %arg0[%c0, %c0] [2, 4] [1, 1] : memref<2x3xi8> -> tensor<2x4xi8>
          %cst = arith.constant dense<0> : tensor<2x4xi8>
          %2 = arith.cmpi ne, %1, %cst : tensor<2x4xi8>
          %3 = stablehlo.reshape %2 : (tensor<2x4xi1>) -> tensor<1x2x4xi1>
          %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<1x2x4xi1>) -> tensor<2x4x1xi1>
          %5 = arith.extui %4 : tensor<2x4x1xi1> to tensor<2x4x1xi8>
          xtile.insert %5 into %arg1[%c0, %c0, %c0] [2, 4, 1] [1, 1, 1] : tensor<2x4x1xi8> -> memref<2x3x1xi8>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "bitcast_copy_fusion",
        1,
        [InputSpec((2, 3))],
        (2, 3),
        np.bool,
        lambda input: input,
    )


if __name__ == "__main__":
  absltest.main()
