# [MODULE DOC]
# Functional tests for the space-to-batch and batch-to-space operations and their N-dimensional
# variants. Includes a direct NumPy reference implementation of space-to-batch, tests of the
# Python wrapper and the raw generated (C++) op bindings via mixin classes, N-D padding cases,
# error handling (invalid block sizes, negative padding, mismatched shapes, overflow), static
# shape inference with unknown dimensions, gradient checks, a relationship check against
# space-to-depth, and the helper that computes required paddings and crops.
# No import-time side effects beyond imports; runs the test main entry point as a script.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for SpaceToBatch and BatchToSpace ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def space_to_batch_direct(input_array, block_shape, paddings):
  # [FUNCTION DOC]
  # HELPER: reference implementation used by the N-D space-to-batch tests.
  # Pure NumPy space-to-batch: zero-pads the spatial dimensions of the input array (given as any
  # array-like with batch first) by the paddings (num_block_dims by 2 begin/end pairs), splits
  # each padded block dimension into blocks of block_shape size, moves the within-block offsets
  # into the batch dimension, and returns an array whose batch size is the input batch times the
  # product of the block sizes. Trailing dimensions beyond the block dimensions are kept.
  # No side effects; does not mutate its inputs.
  """Direct Python implementation of space-to-batch conversion.

  This is used for tests only.

  Args:
    input_array: N-D array
    block_shape: 1-D array of shape [num_block_dims].
    paddings: 2-D array of shape [num_block_dims, 2].

  Returns:
    Converted tensor.
  """
  input_array = np.array(input_array)
  block_shape = np.array(block_shape)
  num_block_dims = len(block_shape)
  paddings = np.array(paddings).reshape((len(block_shape), 2))

  padded = np.pad(input_array,
                  pad_width=([[0, 0]] + list(paddings) + [[0, 0]] *
                             (input_array.ndim - 1 - num_block_dims)),
                  mode="constant")
  reshaped_padded_shape = [input_array.shape[0]]
  output_shape = [input_array.shape[0] * np.prod(block_shape)]
  for block_dim, block_shape_value in enumerate(block_shape):
    reduced_size = padded.shape[block_dim + 1] // block_shape_value
    reshaped_padded_shape.append(reduced_size)
    output_shape.append(reduced_size)
    reshaped_padded_shape.append(block_shape_value)
  reshaped_padded_shape.extend(input_array.shape[num_block_dims + 1:])
  output_shape.extend(input_array.shape[num_block_dims + 1:])

  reshaped_padded = padded.reshape(reshaped_padded_shape)
  permuted_reshaped_padded = np.transpose(reshaped_padded, (
      list(np.arange(num_block_dims) * 2 + 2) + [0] +
      list(np.arange(num_block_dims) * 2 + 1) + list(
          np.arange(input_array.ndim - num_block_dims - 1) + 1 + num_block_dims
          * 2)))
  return permuted_reshaped_padded.reshape(output_shape)


class PythonOpImpl(object):
  # [CLASS DOC]
  # HELPER: mixin for the space-to-batch test classes.
  # Supplies static methods space_to_batch and batch_to_space that forward to the public Python
  # wrapper functions, so test classes can be run against the Python API by inheriting from it.
  # Holds no state.

  @staticmethod
  def space_to_batch(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: static adapter for the test mixin.
    # Forwards all positional and keyword arguments unchanged to the Python wrapper function for
    # space-to-batch and returns its resulting tensor. Errors propagate from the wrapper.
    return array_ops.space_to_batch(*args, **kwargs)

  @staticmethod
  def batch_to_space(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: static adapter for the test mixin.
    # Forwards all arguments unchanged to the Python wrapper function for batch-to-space and
    # returns its resulting tensor. Errors propagate from the wrapper.
    return array_ops.batch_to_space(*args, **kwargs)


class CppOpImpl(object):
  # [CLASS DOC]
  # HELPER: mixin for the space-to-batch test classes.
  # KNOWN ISSUE: the Cpp variants list this mixin after a base that already inherits the Python
  # mixin, so method resolution finds the Python mixin methods first and this class's methods
  # are never used; the Cpp test classes effectively re-run the Python wrapper tests.
  # Supplies static methods space_to_batch and batch_to_space that forward to the raw generated
  # op bindings (the C++ kernels) rather than the Python wrappers. Intended for the Cpp variants
  # of the test classes. Holds no state.

  @staticmethod
  def space_to_batch(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: static adapter for the test mixin.
    # Forwards all arguments unchanged to the generated raw space-to-batch op and returns its
    # tensor. Errors propagate from the op.
    return gen_array_ops.space_to_batch(*args, **kwargs)

  @staticmethod
  def batch_to_space(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: static adapter for the test mixin.
    # Forwards all arguments unchanged to the generated raw batch-to-space op and returns its
    # tensor. Errors propagate from the op.
    return gen_array_ops.batch_to_space(*args, **kwargs)


class SpaceToBatchTest(test.TestCase, parameterized.TestCase, PythonOpImpl):
  # [CLASS DOC]
  # Tests input-output pairs of the 4-D space-to-batch and batch-to-space ops through the Python
  # compatibility wrapper (via the Python mixin). Each case checks the forward conversion and the
  # inverse conversion with a given block size and paddings. Some tests are parameterized over
  # dtypes. Subclassed by the Cpp variant.
  """Tests input-output pairs for the SpaceToBatch and BatchToSpace ops.

  This uses the Python compatibility wrapper that forwards to space_to_batch_nd.
  """

  def _testPad(self,
               inputs,
               paddings,
               block_size,
               outputs,
               dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: shared round-trip check for the space-to-batch test cases.
    # Inputs: inputs (nested list of values), paddings (2 by 2 integer array), block_size (integer),
    # outputs (expected converted nested list), dtype (element type used for casting, default
    # 32-bit float). Casts inputs, applies space-to-batch and asserts equality with outputs, then
    # casts outputs, applies batch-to-space and asserts equality with inputs. Returns nothing.
    with self.cached_session():
      # outputs = space_to_batch(inputs)
      x_tf = self.space_to_batch(
          math_ops.cast(inputs, dtype), paddings, block_size=block_size)
      self.assertAllEqual(x_tf, outputs)
      # inputs = batch_to_space(outputs)
      x_tf = self.batch_to_space(
          math_ops.cast(outputs, dtype), paddings, block_size=block_size)
      self.assertAllEqual(x_tf, inputs)

  def _testOne(self, inputs, block_size, outputs, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: same round-trip check as the padded helper, but with all-zero 2 by 2 paddings.
    # Inputs: inputs, block_size, expected outputs, and element dtype (default 32-bit float).
    # Returns nothing.
    paddings = np.zeros((2, 2), dtype=np.int32)
    self._testPad(inputs, paddings, block_size, outputs, dtype)

  # [1, 2, 2, 1] <-> [4, 1, 1, 1]
  @parameterized.parameters(dtypes.float32, dtypes.float16, dtypes.bfloat16,
                            dtypes.uint8)
  @test_util.run_deprecated_v1
  def testSmallInput2x2(self, dtype):
    # [FUNCTION DOC]
    # Test (parameterized over float32, float16, bfloat16, uint8): a 1x2x2x1 input with block
    # size 2 converts to 4x1x1x1 and back, with zero paddings. Returns nothing.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 2
    x_out = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    self._testOne(x_np, block_size, x_out, dtype)

  # [1, 2, 2, 1] <-> [1, 3, 3, 1] (padding) <-> [9, 1, 1, 1]
  @test_util.run_deprecated_v1
  def testSmallInput2x2Pad1x0(self):
    # [FUNCTION DOC]
    # Test: a 1x2x2x1 input padded by one at the beginning of each spatial dimension becomes 3x3,
    # block size 3 gives a 9x1x1x1 output (zeros where padding landed); checks round trip.
    # Returns nothing.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.array([[1, 0], [1, 0]], dtype=np.int32)
    block_size = 3
    x_out = [[[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[1]]], [[[2]]], [[[0]]],
             [[[3]]], [[[4]]]]
    self._testPad(x_np, paddings, block_size, x_out)

  # Test with depth larger than 1.
  # [1, 2, 2, 3] <-> [4, 1, 1, 3]
  @test_util.run_deprecated_v1
  def testDepthInput2x2(self):
    # [FUNCTION DOC]
    # Test: 1x2x2x3 input (depth 3), block size 2, no padding gives 4x1x1x3; checks round trip.
    # Returns nothing.
    x_np = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
    block_size = 2
    x_out = [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
    self._testOne(x_np, block_size, x_out)

  # Test for larger input dimensions.
  # [1, 4, 4, 1] <-> [4, 2, 2, 1]
  @test_util.run_deprecated_v1
  def testLargerInput2x2(self):
    # [FUNCTION DOC]
    # Test: 1x4x4x1 input with block size 2 gives 4x2x2x1; checks the spatial layout and round
    # trip. Returns nothing.
    x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]],
             [[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
    block_size = 2
    x_out = [[[[1], [3]], [[9], [11]]], [[[2], [4]], [[10], [12]]],
             [[[5], [7]], [[13], [15]]], [[[6], [8]], [[14], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Test with batch larger than 1.
  # [2, 2, 4, 1] <-> [8, 1, 2, 1]
  @test_util.run_deprecated_v1
  def testBatchInput2x2(self):
    # [FUNCTION DOC]
    # Test: 2x2x4x1 input (batch of two), block size 2 gives 8x1x2x1; checks the batch
    # interleaving and round trip. Returns nothing.
    x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
            [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
    block_size = 2
    x_out = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
             [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input spatial dimensions AND batch larger than 1, to ensure
  # that elements are correctly laid out spatially and properly interleaved
  # along the batch dimension.
  # [2, 4, 4, 1] <-> [8, 2, 2, 1]
  @test_util.run_deprecated_v1
  def testLargerInputBatch2x2(self):
    # [FUNCTION DOC]
    # Test: 2x4x4x1 input (larger spatial size and batch), block size 2 gives 8x2x2x1; checks the
    # spatial layout and the interleaving along the batch dimension, plus round trip. Returns
    # nothing.
    x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]],
             [[9], [10], [11], [12]], [[13], [14], [15], [16]]],
            [[[17], [18], [19], [20]], [[21], [22], [23], [24]],
             [[25], [26], [27], [28]], [[29], [30], [31], [32]]]]
    x_out = [[[[1], [3]], [[9], [11]]], [[[17], [19]], [[25], [27]]],
             [[[2], [4]], [[10], [12]]], [[[18], [20]], [[26], [28]]],
             [[[5], [7]], [[13], [15]]], [[[21], [23]], [[29], [31]]],
             [[[6], [8]], [[14], [16]]], [[[22], [24]], [[30], [32]]]]
    block_size = 2
    self._testOne(x_np, block_size, x_out)


class SpaceToBatchCppTest(SpaceToBatchTest, CppOpImpl):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says this uses the C++ ops, but since the first base already
  # inherits the Python mixin, method resolution picks the Python wrapper methods and the C++
  # mixin is never consulted, so it re-runs the same Python-wrapper tests.
  # Variant of the space-to-batch tests intended to run against the raw generated ops. Adds no
  # tests or state of its own; inherits all test methods.
  """Tests input-output pairs for the SpaceToBatch and BatchToSpace ops.

  This uses the C++ ops.
  """
  pass


class SpaceToBatchNDTest(test.TestCase):
  # [CLASS DOC]
  # Tests input-output pairs for the N-dimensional space-to-batch and batch-to-space ops, using
  # explicit expected outputs and a NumPy reference implementation, on both CPU and GPU sessions.
  # Covers zero block dimensions, one block dimension, and many dimensions with zero-size cases.
  """Tests input-output pairs for the SpaceToBatchND and BatchToSpaceND ops."""

  def _testPad(self, inputs, block_shape, paddings, outputs):
    # [FUNCTION DOC]
    # HELPER: shared round-trip check for the N-D tests.
    # Inputs: inputs and expected outputs (nested lists or arrays), block_shape (list of block
    # sizes), paddings (flattenable to num_block_dims by 2). For both a CPU and a GPU-allowed
    # session, casts to float32, applies N-D space-to-batch and asserts equality with outputs,
    # then applies N-D batch-to-space to the outputs and asserts equality with inputs.
    # Returns nothing.
    block_shape = np.array(block_shape)
    paddings = np.array(paddings).reshape((len(block_shape), 2))
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        # outputs = space_to_batch(inputs)
        x_tf = array_ops.space_to_batch_nd(
            math_ops.cast(inputs, dtypes.float32), block_shape, paddings)
        self.assertAllEqual(x_tf, outputs)
        # inputs = batch_to_space(outputs)
        x_tf = array_ops.batch_to_space_nd(
            math_ops.cast(outputs, dtypes.float32), block_shape, paddings)
        self.assertAllEqual(x_tf, inputs)

  def _testDirect(self, input_shape, block_shape, paddings):
    # [FUNCTION DOC]
    # HELPER: builds a float32 ramp array of the given input shape, computes the expected result
    # with the NumPy reference space-to-batch function, and runs the N-D round-trip check against
    # it. Inputs: input_shape, block_shape, paddings (lists). Returns nothing.
    inputs = np.arange(np.prod(input_shape), dtype=np.float32)
    inputs = inputs.reshape(input_shape)
    self._testPad(inputs, block_shape, paddings,
                  space_to_batch_direct(inputs, block_shape, paddings))

  @test_util.run_deprecated_v1
  def testZeroBlockDimsZeroRemainingDims(self):
    # [FUNCTION DOC]
    # Test: with no block dimensions and a 1-D input, the conversion is the identity. Returns
    # nothing.
    self._testPad(
        inputs=[1, 2],
        block_shape=[],
        paddings=[],
        outputs=[1, 2],)

  @test_util.run_deprecated_v1
  def testZeroBlockDimsOneRemainingDim(self):
    # [FUNCTION DOC]
    # Test: 2x2 input with no block dimensions, and again with one no-op block dimension of size
    # 1, is unchanged by the conversion. Returns nothing.
    self._testPad(
        inputs=[[1, 2], [3, 4]],
        block_shape=[],
        paddings=[],
        outputs=[[1, 2], [3, 4]])

    # Same thing, but with a no-op block dim.
    self._testPad(
        inputs=[[1, 2], [3, 4]],
        block_shape=[1],
        paddings=[[0, 0]],
        outputs=[[1, 2], [3, 4]])

  @test_util.run_deprecated_v1
  def testZeroBlockDimsTwoRemainingDims(self):
    # [FUNCTION DOC]
    # Test: 2x2x2 input with zero, one, or two no-op block dimensions (size 1, no padding) is
    # unchanged by the conversion. Returns nothing.
    self._testPad(
        inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        block_shape=[],
        paddings=[],
        outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # Same thing, but with a no-op block dim.
    self._testPad(
        inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        block_shape=[1],
        paddings=[[0, 0]],
        outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # Same thing, but with two no-op block dims.
    self._testPad(
        inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        block_shape=[1, 1],
        paddings=[[0, 0], [0, 0]],
        outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

  @test_util.run_deprecated_v1
  def testOneBlockDimZeroRemainingDims(self):
    # [FUNCTION DOC]
    # Test: 2x3 input, one block dimension of size 2, padding one at the start, yields a 4x2
    # result with zeros where padding landed; checks round trip. Returns nothing.
    self._testPad(
        inputs=[[1, 2, 3], [4, 5, 6]],
        block_shape=[2],
        paddings=[1, 0],
        outputs=[[0, 2], [0, 5], [1, 3], [4, 6]])

  @test_util.run_deprecated_v1
  def testOneBlockDimOneRemainingDim(self):
    # [FUNCTION DOC]
    # Test: 2x3x2 input, one block dimension of size 2, padding one at the start, yields a 4x2x2
    # result; the trailing dimension is carried through unchanged. Checks round trip. Returns
    # nothing.
    self._testPad(
        inputs=[[[1, 11], [2, 21], [3, 31]], [[4, 41], [5, 51], [6, 61]]],
        block_shape=[2],
        paddings=[1, 0],
        outputs=[[[0, 0], [2, 21]], [[0, 0], [5, 51]], [[1, 11], [3, 31]],
                 [[4, 41], [6, 61]]])

  @test_util.run_deprecated_v1
  def testDirect(self):
    # [FUNCTION DOC]
    # Test: compares the ops with the NumPy reference for several shapes: zero-size remaining
    # dimension, zero-size blocked dimension, padding up from zero size, and up to six block
    # dimensions in an 8-D input, including no-op block sizes. Each case runs the round-trip
    # check. Returns nothing.
    # Test with zero-size remaining dimension.
    self._testDirect(
        input_shape=[3, 1, 2, 0], block_shape=[3], paddings=[[0, 2]])

    # Test with zero-size blocked dimension.
    self._testDirect(
        input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[0, 0]])

    # Test with padding up from zero size.
    self._testDirect(
        input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[1, 2]])

    self._testDirect(
        input_shape=[3, 3, 4, 5, 2],
        block_shape=[3, 4, 2],
        paddings=[[1, 2], [0, 0], [3, 0]])

    self._testDirect(
        input_shape=[3, 3, 4, 5, 2],
        block_shape=[3, 4, 2, 2],
        paddings=[[1, 2], [0, 0], [3, 0], [0, 0]])

    self._testDirect(
        input_shape=[3, 2, 2, 3, 4, 5, 2, 5],
        block_shape=[1, 1, 3, 4, 2, 2],
        paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0]])

    self._testDirect(
        input_shape=[3, 2, 2, 3, 4, 5, 2, 5],
        block_shape=[1, 1, 3, 4, 2, 2, 1],
        paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0], [0, 0]])


class SpaceToBatchSpaceToDepth(test.TestCase, PythonOpImpl):
  # [CLASS DOC]
  # Tests the relationship between space-to-batch and space-to-depth through the Python mixin.
  # Subclassed by a Cpp variant.

  # Verifies that: space_to_batch(x) = transpose(space_to_depth(transpose(x)))
  @test_util.run_deprecated_v1
  def testSpaceToDepthTranspose(self):
    # [FUNCTION DOC]
    # Test: for a 5x10x16x7 ramp tensor and block size 2 with zero paddings, space-to-batch of x
    # must equal transposing (swap batch and depth axes), applying space-to-depth, and
    # transposing back. Asserts the two results are equal. Returns nothing.
    x = np.arange(5 * 10 * 16 * 7, dtype=np.float32).reshape([5, 10, 16, 7])
    block_size = 2
    paddings = np.zeros((2, 2), dtype=np.int32)
    y1 = self.space_to_batch(x, paddings, block_size=block_size)
    y2 = array_ops.transpose(
        array_ops.space_to_depth(
            array_ops.transpose(x, [3, 1, 2, 0]), block_size=block_size),
        [3, 1, 2, 0])
    with self.session():
      self.assertAllEqual(y1, y2)


class SpaceToBatchSpaceToDepthCpp(SpaceToBatchSpaceToDepth, CppOpImpl):
  # [CLASS DOC]
  # KNOWN ISSUE: intended to run the space-to-depth relationship test against the raw generated
  # op, but the first base already inherits the Python mixin so the C++ mixin is never used.
  # Variant of the parent test class with no body of its own; inherits all tests.
  pass


class SpaceToBatchErrorHandlingTest(test.TestCase, PythonOpImpl):
  # [CLASS DOC]
  # Tests that the 4-D space-to-batch wrapper rejects invalid input: missing batch dimension,
  # invalid or too-large block sizes, and block sizes that do not divide the spatial dimensions;
  # also checks static rank inference with unknown shapes. Uses the Python mixin.

  @test_util.run_deprecated_v1
  def testInputWrongDimMissingBatch(self):
    # [FUNCTION DOC]
    # Test: a 3-D input (batch dimension missing) must make space-to-batch raise a value error.
    # Returns nothing.
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]], [[3], [4]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 2
    with self.assertRaises(ValueError):
      _ = self.space_to_batch(x_np, paddings, block_size)

  @test_util.run_deprecated_v1
  def testBlockSize0(self):
    # [FUNCTION DOC]
    # Test: block size 0 must cause a value error when building or evaluating the op.
    # Returns nothing.
    # The block size is 0.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = self.space_to_batch(x_np, paddings, block_size)
      out_tf.eval()

  @test_util.run_deprecated_v1
  def testBlockSizeOne(self):
    # [FUNCTION DOC]
    # Test: block size 1 (must be greater than 1) must cause a value error. Returns nothing.
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = self.space_to_batch(x_np, paddings, block_size)
      out_tf.eval()

  @test_util.run_deprecated_v1
  def testBlockSizeLarger(self):
    # [FUNCTION DOC]
    # Test: block size 10 on a 2x2 spatial input (too large) must cause a value error on build or
    # evaluation. Returns nothing.
    # The block size is too large for this input.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 10
    with self.assertRaises(ValueError):
      out_tf = self.space_to_batch(x_np, paddings, block_size)
      self.evaluate(out_tf)

  @test_util.run_deprecated_v1
  def testBlockSizeNotDivisibleWidth(self):
    # [FUNCTION DOC]
    # Test: a 2-high, 3-wide spatial input with block size 3 (divides width but not height) must
    # raise a value error. Returns nothing.
    # The block size divides width but not height.
    x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(ValueError):
      _ = self.space_to_batch(x_np, paddings, block_size)

  @test_util.run_deprecated_v1
  def testBlockSizeNotDivisibleHeight(self):
    # [FUNCTION DOC]
    # Test: a 3x2 spatial input with block size 3 (divides height but not width) must raise a
    # value error. Returns nothing.
    # The block size divides height but not width.
    x_np = [[[[1], [2]], [[3], [4]], [[5], [6]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(ValueError):
      _ = self.space_to_batch(x_np, paddings, block_size)

  @test_util.run_deprecated_v1
  def testBlockSizeNotDivisibleBoth(self):
    # [FUNCTION DOC]
    # Test: a 2x2 spatial input with block size 3 (divides neither dimension) must raise a value
    # error. Returns nothing.
    # The block size does not divide neither width or height.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(ValueError):
      _ = self.space_to_batch(x_np, paddings, block_size)

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Test: with a placeholder of unknown shape for input and paddings and block size 4, the
    # result still has statically known rank 4. Returns nothing.
    t = self.space_to_batch(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32),
        block_size=4)
    self.assertEqual(4, t.get_shape().ndims)


class SpaceToBatchErrorHandlingCppTest(SpaceToBatchErrorHandlingTest,
                                       CppOpImpl):
  # [CLASS DOC]
  # KNOWN ISSUE: intended to run the error-handling tests against the raw generated ops, but the
  # first base already inherits the Python mixin so the C++ mixin is never used.
  # Variant of the parent test class with no body of its own; inherits all tests.
  pass


class SpaceToBatchNDErrorHandlingTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the N-D space-to-batch op rejects invalid block shapes and paddings, both when
  # sizes are known at graph construction time and when supplied through placeholders at run
  # time, and that static shape inference works with partially or fully unknown shapes.

  def _testStaticShape(self, input_shape, block_shape, paddings, error):
    # [FUNCTION DOC]
    # HELPER: asserts that applying N-D space-to-batch to a zero array of input_shape with the
    # given block_shape and paddings (lists) raises the expected error type at construction time.
    # Returns nothing.
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)

    # Try with sizes known at graph construction time.
    with self.assertRaises(error):
      _ = array_ops.space_to_batch_nd(
          np.zeros(input_shape, np.float32), block_shape, paddings)

  def _testDynamicShape(self, input_shape, block_shape, paddings):
    # [FUNCTION DOC]
    # HELPER: builds the N-D space-to-batch op from placeholders (input of unknown shape, block
    # shape with known length, paddings unknown) and evaluates it by feeding the given values;
    # asserts a value error is raised at evaluation. Returns nothing.
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)
    # Try with sizes unknown at graph construction time.
    input_placeholder = array_ops.placeholder(dtypes.float32)
    block_shape_placeholder = array_ops.placeholder(
        dtypes.int32, shape=block_shape.shape)
    paddings_placeholder = array_ops.placeholder(dtypes.int32)
    t = array_ops.space_to_batch_nd(input_placeholder, block_shape_placeholder,
                                    paddings_placeholder)

    with self.assertRaises(ValueError):
      _ = t.eval({
          input_placeholder: np.zeros(input_shape, np.float32),
          block_shape_placeholder: block_shape,
          paddings_placeholder: paddings
      })

  def _testShape(self, input_shape, block_shape, paddings, error):
    # [FUNCTION DOC]
    # HELPER: runs both the static-shape check (expecting the given error type) and the
    # dynamic-shape check (expecting a value error) for the same invalid arguments. Returns
    # nothing.
    self._testStaticShape(input_shape, block_shape, paddings, error)
    self._testDynamicShape(input_shape, block_shape, paddings)

  @test_util.run_deprecated_v1
  def testBlockSize0(self):
    # [FUNCTION DOC]
    # Test: a block shape containing 0 is rejected with a value error. Returns nothing.
    # The block size is 0.
    self._testShape([1, 2, 2], [0, 2], [[0, 0], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testBlockSizeNegative(self):
    # [FUNCTION DOC]
    # Test: a negative block size is rejected with a value error. Returns nothing.
    self._testShape([1, 2, 2], [-1, 2], [[0, 0], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testNegativePadding(self):
    # [FUNCTION DOC]
    # Test: a negative padding value is rejected with a value error. Returns nothing.
    # The padding is negative.
    self._testShape([1, 2, 2], [1, 1], [[0, -1], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testBlockSizeNotDivisible(self):
    # [FUNCTION DOC]
    # Test: when the padded spatial size is not divisible by the block size, a value error is
    # raised. Returns nothing.
    # The padded size is not divisible by the block size.
    self._testShape([1, 2, 3, 1], [3, 3], [[0, 0], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testBlockDimsMismatch(self):
    # [FUNCTION DOC]
    # Test: block shape of length two with paddings of one row raises a value error at
    # construction time (static check only). Returns nothing.
    # Shape of block_shape does not match shape of paddings.
    self._testStaticShape([1, 3, 3, 1], [3, 3], [[0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testUnknown(self):
    # [FUNCTION DOC]
    # Test: shape inference for N-D space-to-batch with progressively more known information:
    # fully unknown input shape and paddings (must not fail); known rank only (rank 4 result);
    # partially known dimensions (known depth, constant block shape giving a known batch size,
    # and known padded sizes where available); fully known dimensions. Asserts the inferred
    # static shapes. Returns nothing.
    # Verify that input shape and paddings shape can be unknown.
    _ = array_ops.space_to_batch_nd(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))

    # Only number of input dimensions is known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(None, None, None, None)),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))
    self.assertEqual(4, t.get_shape().ndims)

    # Dimensions are partially known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(None, None, None, 2)),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))
    self.assertEqual([None, None, None, 2], t.get_shape().as_list())

    # Dimensions are partially known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3, None, None, 2)), [2, 3],
        array_ops.placeholder(dtypes.int32))
    self.assertEqual([3 * 2 * 3, None, None, 2], t.get_shape().as_list())

    # Dimensions are partially known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3, None, 2, 2)), [2, 3], [[1, 1], [0, 1]])
    self.assertEqual([3 * 2 * 3, None, 1, 2], t.get_shape().as_list())

    # Dimensions are fully known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3, 2, 3, 2)), [2, 3], [[1, 1], [0, 0]])
    self.assertEqual([3 * 2 * 3, 2, 1, 2], t.get_shape().as_list())

  @test_util.run_in_graph_and_eager_modes
  def testInvalidBlockShape(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): a block shape of negative values on a 10x20x20 input raises
    # a value error or invalid-argument error saying block shape must be positive. Returns
    # nothing.
    tf_in = constant_op.constant(
        -3.5e+35, shape=[10, 20, 20], dtype=dtypes.float32)
    block_shape = constant_op.constant(-10, shape=[2], dtype=dtypes.int64)
    paddings = constant_op.constant(0, shape=[2, 2], dtype=dtypes.int32)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "block_shape must be positive"):
      array_ops.space_to_batch_nd(tf_in, block_shape, paddings)

  @test_util.run_in_graph_and_eager_modes
  def testOutputSizeOutOfBounds(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): a huge block shape (about 1.9 billion) makes the output batch
    # size overflow; expects a value error or invalid-argument error mentioning a negative
    # dimension size caused by overflow. Returns nothing.
    tf_in = constant_op.constant(
        -3.5e+35, shape=[10, 19, 22], dtype=dtypes.float32)
    block_shape = constant_op.constant(
        1879048192, shape=[2], dtype=dtypes.int64)
    paddings = constant_op.constant(0, shape=[2, 2], dtype=dtypes.int32)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Negative.* dimension size caused by overflow"):
      array_ops.space_to_batch_nd(tf_in, block_shape, paddings)


class SpaceToBatchGradientTest(test.TestCase, PythonOpImpl):
  # [CLASS DOC]
  # Numerical gradient checks for the 4-D space-to-batch op via the Python mixin: compares
  # theoretical and numerical Jacobians for small shapes, with and without padding.

  # Check the gradients.
  def _checkGrad(self, x, paddings, block_size):
    # [FUNCTION DOC]
    # HELPER: gradient check. Inputs: x (4-D float array, asserted to have 4 dimensions via an
    # assertion statement), paddings (2 by 2 integer array), block_size (integer). Applies
    # space-to-batch and compares theoretical and numerical Jacobians (epsilon 1e-5, relative
    # tolerance 1e-2). Returns nothing.
    assert 4 == x.ndim
    with self.cached_session():
      tf_x = ops.convert_to_tensor(x)
      tf_y = self.space_to_batch(tf_x, paddings, block_size)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for space_to_batch of x which is a four dimensional
  # tensor of shape [b, h * block_size, w * block_size, d].
  def _compare(self, b, h, w, d, block_size, pad_beg, pad_end):
    # [FUNCTION DOC]
    # HELPER: builds a random normal float32 tensor of shape batch b by height h times block size
    # by width w times block size by depth d, uses the same begin/end padding on both spatial
    # dimensions, and runs the gradient check. Returns nothing.
    block_size_sq = block_size * block_size
    x = np.random.normal(0, 1, b * h * w * d *
                         block_size_sq).astype(np.float32).reshape(
                             [b, h * block_size, w * block_size, d])
    paddings = np.array(
        [[pad_beg, pad_end], [pad_beg, pad_end]], dtype=np.int32)

    self._checkGrad(x, paddings, block_size)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  @test_util.run_deprecated_v1
  def testSmall(self):
    # [FUNCTION DOC]
    # Test: gradient check for a 1x4x6x5 input with block size 2 and no padding. Returns nothing.
    block_size = 2
    pad_beg = 0
    pad_end = 0
    self._compare(1, 2, 3, 5, block_size, pad_beg, pad_end)

  @test_util.run_deprecated_v1
  def testSmall2(self):
    # [FUNCTION DOC]
    # Test: gradient check for a 2x8x6x2 input with block size 2 and no padding. Returns nothing.
    block_size = 2
    pad_beg = 0
    pad_end = 0
    self._compare(2, 4, 3, 2, block_size, pad_beg, pad_end)

  @test_util.run_deprecated_v1
  def testSmallPad1x1(self):
    # [FUNCTION DOC]
    # Test: gradient check for a 1x4x6x5 input with block size 2 and padding of one before and
    # after each spatial dimension. Returns nothing.
    block_size = 2
    pad_beg = 1
    pad_end = 1
    self._compare(1, 2, 3, 5, block_size, pad_beg, pad_end)


class SpaceToBatchGradientCppTest(SpaceToBatchGradientTest, CppOpImpl):
  # [CLASS DOC]
  # KNOWN ISSUE: intended to run the gradient tests against the raw generated ops, but the first
  # base already inherits the Python mixin so the C++ mixin is never used.
  # Variant of the parent test class with no body of its own; inherits all tests.
  pass


class SpaceToBatchNDGradientTest(test.TestCase):
  # [CLASS DOC]
  # Numerical gradient checks for the N-D space-to-batch op with various block shapes and
  # paddings, including three block dimensions.

  # Check the gradients.
  def _checkGrad(self, x, block_shape, paddings):
    # [FUNCTION DOC]
    # HELPER: gradient check. Inputs: x (float array), block_shape (list), paddings (flattenable
    # to num_block_dims by 2). Applies N-D space-to-batch and compares theoretical and numerical
    # Jacobians (epsilon 1e-5, relative tolerance 1e-2). Returns nothing.
    block_shape = np.array(block_shape)
    paddings = np.array(paddings).reshape((len(block_shape), 2))
    with self.cached_session():
      tf_x = ops.convert_to_tensor(x)
      tf_y = array_ops.space_to_batch_nd(tf_x, block_shape, paddings)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  def _compare(self, input_shape, block_shape, paddings):
    # [FUNCTION DOC]
    # HELPER: builds a random normal float32 tensor of input_shape and runs the N-D gradient
    # check with the given block_shape and paddings. Returns nothing.
    x = np.random.normal(
        0, 1, np.prod(input_shape)).astype(np.float32).reshape(input_shape)
    self._checkGrad(x, block_shape, paddings)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  @test_util.run_deprecated_v1
  def testSmall(self):
    # [FUNCTION DOC]
    # Test: N-D gradient check, 1x4x6x5 input, block shape 2x2, no padding. Returns nothing.
    self._compare([1, 4, 6, 5], [2, 2], [[0, 0], [0, 0]])

  @test_util.run_deprecated_v1
  def testSmall2(self):
    # [FUNCTION DOC]
    # Test: N-D gradient check, 2x8x6x2 input, block shape 2x2, no padding. Returns nothing.
    self._compare([2, 8, 6, 2], [2, 2], [[0, 0], [0, 0]])

  @test_util.run_deprecated_v1
  def testSmallPad1(self):
    # [FUNCTION DOC]
    # Test: N-D gradient check, 2x4x6x2 input, block shape 2x2, padding of one on every side.
    # Returns nothing.
    self._compare([2, 4, 6, 2], [2, 2], [[1, 1], [1, 1]])

  @test_util.run_deprecated_v1
  def testSmallPadThreeBlockDims(self):
    # [FUNCTION DOC]
    # Test: N-D gradient check with a 5-D 2x2x4x3x2 input, three block dimensions of size 2 and
    # asymmetric padding. Returns nothing.
    self._compare([2, 2, 4, 3, 2], [2, 2, 2], [[1, 1], [1, 1], [1, 0]])


class RequiredSpaceToBatchPaddingsTest(test.TestCase):
  # [CLASS DOC]
  # Tests the function that computes the paddings and crops required to make an input shape
  # divisible by a block shape, for constant and placeholder (non-constant) arguments.

  def _checkProperties(self, input_shape, block_shape, base_paddings, paddings,
                       crops):
    # [FUNCTION DOC]
    # HELPER: invariant checker. Inputs: input_shape, block_shape, base_paddings (or nothing,
    # treated as zeros), and the computed paddings and crops (arrays of num_block_dims by 2).
    # Asserts shapes, that begin padding equals the base begin padding, that extra end padding is
    # in [0, block size), that the padded size is divisible by the block size, and that crops
    # begin at zero and end equal the extra padding. Returns nothing.
    """Checks that `paddings` and `crops` satisfy invariants."""
    num_block_dims = len(block_shape)
    self.assertEqual(len(input_shape), num_block_dims)
    if base_paddings is None:
      base_paddings = np.zeros((num_block_dims, 2), np.int32)
    self.assertEqual(base_paddings.shape, (num_block_dims, 2))
    self.assertEqual(paddings.shape, (num_block_dims, 2))
    self.assertEqual(crops.shape, (num_block_dims, 2))
    for i in range(num_block_dims):
      self.assertEqual(paddings[i, 0], base_paddings[i, 0])
      self.assertLessEqual(0, paddings[i, 1] - base_paddings[i, 1])
      self.assertLess(paddings[i, 1] - base_paddings[i, 1], block_shape[i])
      self.assertEqual(
          (input_shape[i] + paddings[i, 0] + paddings[i, 1]) % block_shape[i],
          0)
      self.assertEqual(crops[i, 0], 0)
      self.assertEqual(crops[i, 1], paddings[i, 1] - base_paddings[i, 1])

  def _test(self, input_shape, block_shape, base_paddings):
    # [FUNCTION DOC]
    # HELPER: runs the required-paddings function on constant arrays, checks the results are
    # statically constant and satisfy the invariants, then runs it on placeholders (feeding the
    # same values) and asserts the evaluated paddings and crops equal the constant results.
    # Inputs: input_shape, block_shape, optional base_paddings. Returns nothing.
    input_shape = np.array(input_shape)
    block_shape = np.array(block_shape)
    if base_paddings is not None:
      base_paddings = np.array(base_paddings)
    # Check with constants.
    paddings, crops = array_ops.required_space_to_batch_paddings(input_shape,
                                                                 block_shape,
                                                                 base_paddings)
    paddings_const = tensor_util.constant_value(paddings)
    crops_const = tensor_util.constant_value(crops)
    self.assertIsNotNone(paddings_const)
    self.assertIsNotNone(crops_const)
    self._checkProperties(input_shape, block_shape, base_paddings,
                          paddings_const, crops_const)
    # Check with non-constants.
    assignments = {}
    input_shape_placeholder = array_ops.placeholder(dtypes.int32)
    assignments[input_shape_placeholder] = input_shape
    block_shape_placeholder = array_ops.placeholder(dtypes.int32,
                                                    [len(block_shape)])
    assignments[block_shape_placeholder] = block_shape
    if base_paddings is not None:
      base_paddings_placeholder = array_ops.placeholder(dtypes.int32,
                                                        [len(block_shape), 2])
      assignments[base_paddings_placeholder] = base_paddings
    else:
      base_paddings_placeholder = None
    t_paddings, t_crops = array_ops.required_space_to_batch_paddings(
        input_shape_placeholder, block_shape_placeholder,
        base_paddings_placeholder)
    with self.cached_session():
      paddings_result = t_paddings.eval(assignments)
      crops_result = t_crops.eval(assignments)
    self.assertAllEqual(paddings_result, paddings_const)
    self.assertAllEqual(crops_result, crops_const)

  @test_util.run_deprecated_v1
  def testSimple(self):
    # [FUNCTION DOC]
    # Test: runs the paddings helper on empty shapes (with and without base paddings), single
    # and two-dimensional shapes, and several base paddings. Returns nothing.
    self._test(
        input_shape=np.zeros((0,), np.int32),
        block_shape=np.zeros((0,), np.int32),
        base_paddings=None)
    self._test(
        input_shape=np.zeros((0,), np.int32),
        block_shape=np.zeros((0,), np.int32),
        base_paddings=np.zeros((0, 2), np.int32))
    self._test(input_shape=[1], block_shape=[2], base_paddings=None)
    self._test(input_shape=[1], block_shape=[2], base_paddings=[[1, 0]])
    self._test(input_shape=[3], block_shape=[1], base_paddings=[[1, 2]])
    self._test(input_shape=[1], block_shape=[2], base_paddings=[[2, 3]])
    self._test(input_shape=[4, 5], block_shape=[3, 2], base_paddings=None)
    self._test(
        input_shape=[4, 5], block_shape=[3, 2], base_paddings=[[0, 0], [0, 1]])


if __name__ == "__main__":
  test.main()
