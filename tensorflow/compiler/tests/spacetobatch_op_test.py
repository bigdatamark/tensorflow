# [MODULE DOC]
# Tests for the space-to-batch and batch-to-space ops (both the fixed 2-D block-size versions and
# the N-dimensional versions) on the XLA test device. Includes a pure numpy reference
# implementation of space-to-batch used to generate expected outputs, plus error-handling tests
# for invalid block shapes. Run as a test binary via the test main entry point.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


def space_to_batch_direct(input_array, block_shape, paddings):
  # [FUNCTION DOC]
  # HELPER: numpy reference implementation of N-dimensional space-to-batch, used to compute
  # expected outputs in tests.
  # Inputs: input_array (N-dimensional array-like), block_shape (one-dimensional list of block
  # sizes for the leading spatial dimensions), paddings (list of before/after padding pairs, one
  # per block dimension). Zero-pads the spatial dimensions, splits each into blocks, moves the
  # block offsets into the batch dimension and returns the resulting numpy array. No side effects.
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


class SpaceToBatchTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the 4-D space-to-batch and batch-to-space ops with a single integer block size.
  # Each test supplies an input, block size (and optional paddings) and the expected rearranged
  # output; the ops are checked in both directions. No instance state.
  """Tests input-output pairs for the SpaceToBatch and BatchToSpace ops."""

  def _testPad(self, inputs, paddings, block_size, outputs):
    # [FUNCTION DOC]
    # HELPER: shared check for the tests in this class. For each floating-point type supported by
    # the device, applies space-to-batch with the given paddings and block size to inputs and
    # asserts the result equals outputs, then applies batch-to-space to outputs and asserts the
    # result equals inputs. Runs inside the XLA test device scope. Returns nothing.
    with self.session() as sess, self.test_scope():
      for dtype in self.float_types:
        # outputs = space_to_batch(inputs)
        placeholder = array_ops.placeholder(dtype)
        x_tf = gen_array_ops.space_to_batch(
            placeholder, paddings, block_size=block_size)
        self.assertAllEqual(sess.run(x_tf, {placeholder: inputs}), outputs)
        # inputs = batch_to_space(outputs)
        x_tf = gen_array_ops.batch_to_space(
            placeholder, paddings, block_size=block_size)
        self.assertAllEqual(sess.run(x_tf, {placeholder: outputs}), inputs)

  def _testOne(self, inputs, block_size, outputs):
    # [FUNCTION DOC]
    # HELPER: convenience wrapper around the padded check that uses all-zero paddings (a two by
    # two int32 array), so only the block rearrangement is tested.
    paddings = np.zeros((2, 2), dtype=np.int32)
    self._testPad(inputs, paddings, block_size, outputs)

  # [1, 2, 2, 1] <-> [4, 1, 1, 1]
  def testSmallInput2x2(self):
    # [FUNCTION DOC]
    # A 1x2x2x1 input with block size 2 and no padding maps to a 4x1x1x1 batch of single values.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 2
    x_out = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    self._testOne(x_np, block_size, x_out)

  # [1, 2, 2, 1] <-> [1, 3, 3, 1] (padding) <-> [9, 1, 1, 1]
  def testSmallInput2x2Pad1x0(self):
    # [FUNCTION DOC]
    # A 1x2x2x1 input padded by one before each spatial dimension (becoming 3x3) with block size 3
    # maps to a 9x1x1x1 batch; the expected output contains zeros for the padded positions.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.array([[1, 0], [1, 0]], dtype=np.int32)
    block_size = 3
    x_out = [[[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[1]]], [[[2]]], [[[0]]],
             [[[3]]], [[[4]]]]
    self._testPad(x_np, paddings, block_size, x_out)

  # Test with depth larger than 1.
  # [1, 2, 2, 3] <-> [4, 1, 1, 3]
  def testDepthInput2x2(self):
    # [FUNCTION DOC]
    # Same as the small 2x2 case but with depth 3: a 1x2x2x3 input with block size 2 maps to
    # 4x1x1x3 and back.
    x_np = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
    block_size = 2
    x_out = [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
    self._testOne(x_np, block_size, x_out)

  # Test for larger input dimensions.
  # [1, 4, 4, 1] <-> [4, 2, 2, 1]
  def testLargerInput2x2(self):
    # [FUNCTION DOC]
    # A 1x4x4x1 input with block size 2 maps to 4x2x2x1; checks spatial layout of the blocks.
    x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]],
             [[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
    block_size = 2
    x_out = [[[[1], [3]], [[9], [11]]], [[[2], [4]], [[10], [12]]],
             [[[5], [7]], [[13], [15]]], [[[6], [8]], [[14], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Test with batch larger than 1.
  # [2, 2, 4, 1] <-> [8, 1, 2, 1]
  def testBatchInput2x2(self):
    # [FUNCTION DOC]
    # A 2x2x4x1 input (batch of two) with block size 2 maps to 8x1x2x1; checks that block
    # offsets are placed in the batch dimension in the expected order.
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
  def testLargerInputBatch2x2(self):
    # [FUNCTION DOC]
    # A 2x4x4x1 input with block size 2 maps to 8x2x2x1; checks both spatial layout and the
    # interleaving along the batch dimension.
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


class SpaceToBatchNDErrorHandlingTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case verifying that the N-dimensional space-to-batch op rejects invalid block shapes
  # with an error at graph construction time on the XLA test device.

  def testInvalidBlockShape(self):
    # [FUNCTION DOC]
    # Builds space_to_batch_nd on a 10x20x20 float input with a block shape of negative ten for
    # both dimensions and zero padding. Expects a value error saying the block shape must be
    # positive.
    with self.assertRaisesRegex(ValueError, "block_shape must be positive"):
      with self.session() as sess, self.test_scope():
        tf_in = constant_op.constant(
            -3.5e+35, shape=[10, 20, 20], dtype=dtypes.float32)
        block_shape = constant_op.constant(-10, shape=[2], dtype=dtypes.int64)
        paddings = constant_op.constant(0, shape=[2, 2], dtype=dtypes.int32)
        sess.run(array_ops.space_to_batch_nd(tf_in, block_shape, paddings))

  def testOutputSizeOutOfBounds(self):
    # [FUNCTION DOC]
    # Builds space_to_batch_nd on a 10x19x22 float input with a huge block shape (1879048192 in
    # both dimensions) and zero padding. Expects a value error about a negative dimension size
    # caused by integer overflow in the computed output size.
    with self.assertRaisesRegex(ValueError,
                                "Negative.* dimension size caused by overflow"):
      with self.session() as sess, self.test_scope():
        tf_in = constant_op.constant(
            -3.5e+35, shape=[10, 19, 22], dtype=dtypes.float32)
        block_shape = constant_op.constant(
            1879048192, shape=[2], dtype=dtypes.int64)
        paddings = constant_op.constant(0, shape=[2, 2], dtype=dtypes.int32)
        sess.run(array_ops.space_to_batch_nd(tf_in, block_shape, paddings))


class SpaceToBatchNDTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the N-dimensional space-to-batch and batch-to-space ops with a list of block
  # sizes. Covers zero, one and several block dimensions, and compares against the numpy
  # reference implementation, including zero-sized dimensions.
  """Tests input-output pairs for the SpaceToBatchND and BatchToSpaceND ops."""

  def _testPad(self, inputs, block_shape, paddings, outputs):
    # [FUNCTION DOC]
    # HELPER: shared check for the N-dimensional tests. Reshapes paddings to one before/after pair
    # per block dimension. For each supported floating-point type (skipping bfloat16 due to a
    # known mismatch, and converting data to float16 when that type is used), applies
    # space_to_batch_nd to inputs and asserts the result equals outputs, then applies
    # batch_to_space_nd to outputs and asserts it equals inputs. Returns nothing.
    block_shape = np.array(block_shape)
    paddings = np.array(paddings).reshape((len(block_shape), 2))
    with self.session() as sess, self.test_scope():
      for dtype in self.float_types:
        # TODO(b/68813416): Skip bfloat16's as the input type for direct is
        # float32 and results in a mismatch, while making testDirect provide the
        # correctly typed input results in 'no fill-function for data-type'
        # error.
        if dtype == dtypes.bfloat16.as_numpy_dtype:
          continue
        if dtype == np.float16:
          actual_inputs = np.array(inputs).astype(dtype)
          actual_paddings = np.array(paddings).astype(dtype)
          expected_outputs = np.array(outputs).astype(dtype)
        else:
          actual_inputs = inputs
          actual_paddings = paddings
          expected_outputs = outputs
        placeholder = array_ops.placeholder(dtype)
        # outputs = space_to_batch(inputs)
        x_tf = array_ops.space_to_batch_nd(placeholder, block_shape,
                                           actual_paddings)
        self.assertAllEqual(
            sess.run(x_tf, {placeholder: actual_inputs}), expected_outputs)
        # inputs = batch_to_space(outputs)
        placeholder = array_ops.placeholder(dtype)
        x_tf = array_ops.batch_to_space_nd(placeholder, block_shape,
                                           actual_paddings)
        self.assertAllEqual(
            sess.run(x_tf, {placeholder: expected_outputs}), actual_inputs)

  def _testDirect(self, input_shape, block_shape, paddings):
    # [FUNCTION DOC]
    # HELPER: builds a float32 input holding consecutive numbers in the given shape, computes the
    # expected output with the numpy reference space-to-batch, and runs the shared padded check.
    inputs = np.arange(np.prod(input_shape), dtype=np.float32)
    inputs = inputs.reshape(input_shape)
    self._testPad(inputs, block_shape, paddings,
                  space_to_batch_direct(inputs, block_shape, paddings))

  def testZeroBlockDimsZeroRemainingDims(self):
    # [FUNCTION DOC]
    # With an empty block shape and paddings, a one-dimensional input passes through unchanged.
    self._testPad(
        inputs=[1, 2],
        block_shape=[],
        paddings=[],
        outputs=[1, 2],)

  def testZeroBlockDimsOneRemainingDim(self):
    # [FUNCTION DOC]
    # A two-dimensional input with no block dimensions, and again with one no-op block dimension
    # (size one, zero padding), passes through unchanged.
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

  def testZeroBlockDimsTwoRemainingDims(self):
    # [FUNCTION DOC]
    # A three-dimensional input with no block dimensions, with one no-op block dimension, and with
    # two no-op block dimensions passes through unchanged.
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

  def testOneBlockDimZeroRemainingDims(self):
    # [FUNCTION DOC]
    # A 2x3 input with block shape 2 and padding of one before the block dimension maps to a 4x2
    # output with zeros in the padded positions.
    self._testPad(
        inputs=[[1, 2, 3], [4, 5, 6]],
        block_shape=[2],
        paddings=[1, 0],
        outputs=[[0, 2], [0, 5], [1, 3], [4, 6]])

  def testOneBlockDimOneRemainingDim(self):
    # [FUNCTION DOC]
    # Like the previous case but with an extra trailing dimension of size 2: a 2x3x2 input maps to
    # a 4x2x2 output.
    self._testPad(
        inputs=[[[1, 11], [2, 21], [3, 31]], [[4, 41], [5, 51], [6, 61]]],
        block_shape=[2],
        paddings=[1, 0],
        outputs=[[[0, 0], [2, 21]], [[0, 0], [5, 51]], [[1, 11], [3, 31]],
                 [[4, 41], [6, 61]]])

  def testDirect0(self):
    # [FUNCTION DOC]
    # Compares against the reference for input shape 3x1x2x0 (zero-size trailing dimension),
    # block shape 3 and padding of 0 before and 2 after.
    # Test with zero-size remaining dimension.
    self._testDirect(
        input_shape=[3, 1, 2, 0], block_shape=[3], paddings=[[0, 2]])

  def testDirect1(self):
    # [FUNCTION DOC]
    # Compares against the reference for input shape 3x0x2x5 (zero-size blocked dimension),
    # block shape 3 and no padding.
    # Test with zero-size blocked dimension.
    self._testDirect(
        input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[0, 0]])

  def testDirect2(self):
    # [FUNCTION DOC]
    # Compares against the reference for input shape 3x0x2x5, block shape 3 and padding 1 before
    # and 2 after, i.e. padding up from a zero-size dimension.
    # Test with padding up from zero size.
    self._testDirect(
        input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[1, 2]])

  def testDirect3(self):
    # [FUNCTION DOC]
    # Compares against the reference for a five-dimensional input (3x3x4x5x2) with three block
    # dimensions (3, 4, 2) and mixed paddings.
    self._testDirect(
        input_shape=[3, 3, 4, 5, 2],
        block_shape=[3, 4, 2],
        paddings=[[1, 2], [0, 0], [3, 0]])

  def testDirect4(self):
    # [FUNCTION DOC]
    # Compares against the reference for the same five-dimensional input as the previous case but
    # with four block dimensions (3, 4, 2, 2) and mixed paddings.
    self._testDirect(
        input_shape=[3, 3, 4, 5, 2],
        block_shape=[3, 4, 2, 2],
        paddings=[[1, 2], [0, 0], [3, 0], [0, 0]])

  def testDirect5(self):
    # [FUNCTION DOC]
    # Compares against the reference for an eight-dimensional input with six block dimensions
    # (some of size one) and mixed paddings.
    self._testDirect(
        input_shape=[3, 2, 2, 3, 4, 5, 2, 5],
        block_shape=[1, 1, 3, 4, 2, 2],
        paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0]])

  def testDirect6(self):
    # [FUNCTION DOC]
    # Compares against the reference for the same eight-dimensional input with seven block
    # dimensions, the last of size one with zero padding.
    self._testDirect(
        input_shape=[3, 2, 2, 3, 4, 5, 2, 5],
        block_shape=[1, 1, 3, 4, 2, 2, 1],
        paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0], [0, 0]])


if __name__ == "__main__":
  test.main()
