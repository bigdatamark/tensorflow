# [MODULE DOC]
# Functional tests for the space-to-depth operation, which rearranges spatial blocks of an image
# batch into the channel (depth) dimension. Covers explicit input/output pairs for several block
# sizes, depths and batch sizes (NHWC, plus NCHW on GPU), error handling for invalid shapes and
# block sizes, static rank inference, comparison against a transpose-based reference for float and
# quantized 8-bit data (including the vectorized-channel layout on GPU), and numerical gradients.
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
"""Functional tests for SpacetoDepth op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class SpaceToDepthTest(test.TestCase):
  # [CLASS DOC]
  # Tests the space-to-depth op: explicit value cases, error handling, shape inference and a
  # comparison against a reshape/transpose reference implementation defined in this class.
  # Holds no state; helpers are methods of the class.

  def _testOne(self, inputs, block_size, outputs, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: shared check for the value tests.
    # Inputs: inputs (nested list or tensor in NHWC layout), block_size (integer), outputs
    # (expected nested list or tensor), dtype (element type for casting, default 32-bit float).
    # Applies space-to-depth in NHWC and asserts equality with outputs; if a GPU is available,
    # also converts to NCHW, applies the op with NCHW layout under a forced GPU device, converts
    # back and asserts equality. Returns nothing.
    input_nhwc = math_ops.cast(inputs, dtype)
    # test NHWC (default)
    x_tf = array_ops.space_to_depth(input_nhwc, block_size)
    self.assertAllEqual(self.evaluate(x_tf), outputs)

    if test_util.is_gpu_available():
      with test_util.force_gpu():
        # test NCHW on GPU
        input_nchw = test_util.NHWCToNCHW(input_nhwc)
        output_nchw = array_ops.space_to_depth(
            input_nchw, block_size, data_format="NCHW")
        output_nhwc = test_util.NCHWToNHWC(output_nchw)
        self.assertAllEqual(self.evaluate(output_nhwc), outputs)

  def testBasic(self):
    # [FUNCTION DOC]
    # Test: a 1x2x2x1 input with block size 2 gives 1x1x1x4 for float32, float16, bfloat16 and
    # uint8. Returns nothing.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 2
    x_out = [[[[1, 2, 3, 4]]]]
    for dtype in [
        dtypes.float32, dtypes.float16, dtypes.bfloat16, dtypes.uint8
    ]:
      self._testOne(x_np, block_size, x_out, dtype=dtype)


  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered spatially.
  def testLargerInput2x2(self):
    # [FUNCTION DOC]
    # Test: 1x4x4x1 input with block size 2 gives 1x2x2x4, checking spatial element ordering.
    # Returns nothing.
    x_np = [[[[1], [2], [5], [6]], [[3], [4], [7], [8]],
             [[9], [10], [13], [14]], [[11], [12], [15], [16]]]]
    block_size = 2
    x_out = [[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12],
                                             [13, 14, 15, 16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered in depth. Here, larger block size.
  def testLargerInput4x4(self):
    # [FUNCTION DOC]
    # Test: 1x4x4x1 input with block size 4 gives 1x1x1x16, checking the depth ordering.
    # Returns nothing.
    x_np = [[[[1], [2], [5], [6]], [[3], [4], [7], [8]],
             [[9], [10], [13], [14]], [[11], [12], [15], [16]]]]
    block_size = 4
    x_out = [[[[1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleaved(self):
    # [FUNCTION DOC]
    # Test: 1x2x2x2 input (depth 2), block size 2 gives 1x1x1x8 with channels interleaved per
    # spatial position. Returns nothing.
    x_np = [[[[1, 10], [2, 20]], [[3, 30], [4, 40]]]]
    block_size = 2
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths. Here an odd depth.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleavedDepth3(self):
    # [FUNCTION DOC]
    # Test: 1x2x2x3 input (odd depth 3), block size 2 gives 1x1x1x12 in order. Returns nothing.
    x_np = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
    block_size = 2
    x_out = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions AND for larger input depths.
  # To make sure elements are properly interleaved in depth and ordered
  # spatially.
  def testDepthInterleavedLarge(self):
    # [FUNCTION DOC]
    # Test: 1x4x4x2 input, block size 2 gives 1x2x2x8, checking both spatial order and depth
    # interleaving. Returns nothing.
    x_np = [[[[1, 10], [2, 20], [5, 50], [6, 60]],
             [[3, 30], [4, 40], [7, 70], [8, 80]],
             [[9, 90], [10, 100], [13, 130], [14, 140]],
             [[11, 110], [12, 120], [15, 150], [16, 160]]]]
    block_size = 2
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40], [5, 50, 6, 60, 7, 70, 8, 80]],
              [[9, 90, 10, 100, 11, 110, 12, 120],
               [13, 130, 14, 140, 15, 150, 16, 160]]]]
    self._testOne(x_np, block_size, x_out)

  def testBlockSize2Batch10(self):
    # [FUNCTION DOC]
    # Test: a batch of 10 inputs, each a 4x4 single-channel image scaled by the batch index, with
    # block size 2; the expected output is the scaled 2x2x4 result per batch element. Returns
    # nothing.
    block_size = 2

    def batch_input_elt(i):
      # [FUNCTION DOC]
      # HELPER: builds one 4x4x1 nested-list image whose values 1..16 (in the fixed pattern) are
      # multiplied by the integer i. Returns the nested list.
      return [[[1 * i], [2 * i], [5 * i], [6 * i]],
              [[3 * i], [4 * i], [7 * i], [8 * i]],
              [[9 * i], [10 * i], [13 * i], [14 * i]],
              [[11 * i], [12 * i], [15 * i], [16 * i]]]

    def batch_output_elt(i):
      # [FUNCTION DOC]
      # HELPER: builds the expected 2x2x4 nested-list result of space-to-depth with block size 2
      # for the image produced by the input builder with the same integer i. Returns it.
      return [[[1 * i, 2 * i, 3 * i, 4 * i], [5 * i, 6 * i, 7 * i, 8 * i]],
              [[9 * i, 10 * i, 11 * i, 12 * i],
               [13 * i, 14 * i, 15 * i, 16 * i]]]

    batch_size = 10
    x_np = [batch_input_elt(i) for i in range(batch_size)]
    x_out = [batch_output_elt(i) for i in range(batch_size)]
    self._testOne(x_np, block_size, x_out)

  def testBatchSize0(self):
    # [FUNCTION DOC]
    # Test: an empty batch (size 0) of 4x6x3 inputs with block size 2 gives a result of shape
    # 0x2x3x12. Returns nothing.
    block_size = 2
    batch_size = 0
    x_np = array_ops.ones([batch_size, 4, 6, 3])
    x_out = array_ops.ones([batch_size, 2, 3, 12])
    self._testOne(x_np, block_size, x_out)

  # Tests for different width and height.
  def testNonSquare(self):
    # [FUNCTION DOC]
    # Test: 1x6x2x2 input (height differs from width), block size 2 gives 1x3x1x8. Returns
    # nothing.
    x_np = [[[[1, 10], [2, 20]], [[3, 30], [4, 40]], [[5, 50], [6, 60]],
             [[7, 70], [8, 80]], [[9, 90], [10, 100]], [[11, 110], [12, 120]]]]
    block_size = 2
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40]], [[5, 50, 6, 60, 7, 70, 8, 80]],
              [[9, 90, 10, 100, 11, 110, 12, 120]]]]
    self._testOne(x_np, block_size, x_out)

  # Error handling:

  def testInputWrongDimMissingDepth(self):
    # [FUNCTION DOC]
    # Test: a 3-D input (depth dimension missing) must raise a value error or invalid-argument
    # error when built or evaluated. Returns nothing.
    # The input is missing the last dimension ("depth")
    x_np = [[[1, 2], [3, 4]]]
    block_size = 2
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      self.evaluate(out_tf)

  def testInputWrongDimMissingBatch(self):
    # [FUNCTION DOC]
    # Test: a 3-D input (batch dimension missing) must raise a value error or invalid-argument
    # error. Returns nothing.
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]], [[3], [4]]]
    block_size = 2
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      _ = array_ops.space_to_depth(x_np, block_size)

  def testBlockSize0(self):
    # [FUNCTION DOC]
    # Test: block size 0 must raise a value error or invalid-argument error. Returns nothing.
    # The block size is 0.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 0
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      self.evaluate(out_tf)

  def testBlockSizeOne(self):
    # [FUNCTION DOC]
    # Test: block size 1 (must exceed 1) must raise a value error or invalid-argument error.
    # Returns nothing.
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 1
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      self.evaluate(out_tf)

  def testBlockSizeLarger(self):
    # [FUNCTION DOC]
    # Test: block size 10 on a 2x2 image (too large) must raise a value error or
    # invalid-argument error. Returns nothing.
    # The block size is too large for this input.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 10
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      self.evaluate(out_tf)

  def testBlockSizeNotDivisibleWidth(self):
    # [FUNCTION DOC]
    # Test: a 2-high, 3-wide image with block size 3 (divides width but not height) must raise a
    # value error or invalid-argument error. Returns nothing.
    # The block size divides width but not height.
    x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
    block_size = 3
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      _ = array_ops.space_to_depth(x_np, block_size)

  def testBlockSizeNotDivisibleHeight(self):
    # [FUNCTION DOC]
    # Test: a 3-high, 2-wide image with block size 3 (divides height but not width) must raise a
    # value error or invalid-argument error. Returns nothing.
    # The block size divides height but not width.
    x_np = [[[[1], [2]], [[3], [4]], [[5], [6]]]]
    block_size = 3
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      _ = array_ops.space_to_depth(x_np, block_size)

  def testBlockSizeNotDivisibleBoth(self):
    # [FUNCTION DOC]
    # Test: a 2x2 image with block size 3 (divides neither dimension) must raise a value error or
    # invalid-argument error. Returns nothing.
    # The block size does not divide neither width or height.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 3
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      _ = array_ops.space_to_depth(x_np, block_size)

  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Test: in a fresh graph, space-to-depth of a placeholder with unknown shape and block size 4
    # still has statically known rank 4. Returns nothing.
    # Testing an unkown shape in graph.
    with ops.Graph().as_default():
      t = array_ops.space_to_depth(
          array_ops.placeholder(dtypes.float32), block_size=4)
      self.assertEqual(4, t.get_shape().ndims)

  def spaceToDepthUsingTranspose(self, tensor, block_size, data_format):
    # [FUNCTION DOC]
    # HELPER: reference implementation for the comparison test.
    # Inputs: tensor (4-D, float or 8-bit quantized), block_size (integer), data_format (NHWC or
    # NCHW). Quantized 8-bit input is bit-cast to plain int8 first and cast back at the end.
    # Reshapes to split height and width into blocks, transposes the block factors next to the
    # channels, and reshapes to the space-to-depth output layout. Asserts (assertion statements)
    # that height and width are divisible by the block size. Any other data_format value leaves
    # the tensor unchanged. Returns the converted tensor.
    block_size_sq = block_size * block_size

    dtype = tensor.dtype
    if dtype == dtypes.qint8:
      tensor = array_ops.bitcast(tensor, dtypes.int8)

    if data_format == "NHWC":
      b, ih, iw, ic = tensor.shape.as_list()
      assert ih % block_size == 0, (ih, block_size)
      assert iw % block_size == 0, (iw, block_size)
      ow, oh, oc = iw // block_size, ih // block_size, ic * block_size_sq
      tensor = array_ops.reshape(tensor,
                                 [b, oh, block_size, ow, block_size, ic])
      tensor = array_ops.transpose(tensor, [0, 1, 3, 2, 4, 5])
      tensor = array_ops.reshape(tensor, [b, oh, ow, oc])
    elif data_format == "NCHW":
      b, ic, ih, iw = tensor.shape.as_list()
      assert ih % block_size == 0, (ih, block_size)
      assert iw % block_size == 0, (iw, block_size)
      ow, oh, oc = iw // block_size, ih // block_size, ic * block_size_sq
      tensor = array_ops.reshape(tensor,
                                 [b, ic, oh, block_size, ow, block_size])
      tensor = array_ops.transpose(tensor, [0, 3, 5, 1, 2, 4])
      tensor = array_ops.reshape(tensor, [b, oc, oh, ow])

    if dtype == dtypes.qint8:
      tensor = array_ops.bitcast(tensor, dtype)
    return tensor

  def compareToTranspose(self, batch_size, out_height, out_width, in_channels,
                         block_size, data_format, data_type, use_gpu):
    # [FUNCTION DOC]
    # HELPER: compares the op against the transpose-based reference.
    # Inputs: batch_size, output height and width, in_channels, block_size (integers),
    # data_format (NHWC, NCHW or NCHW_VECT_C), data_type (32-bit float or quantized 8-bit; the
    # latter is required for the vectorized layout) and use_gpu (whether to run on GPU).
    # Builds an ascending-values input on the CPU (quantizing it for the 8-bit type), runs the
    # op on the chosen device (using bit-casts and layout conversions for the vectorized layout)
    # and the reference, evaluates both and asserts they are exactly equal. Returns nothing.
    in_height = out_height * block_size
    in_width = out_width * block_size
    nhwc_input_shape = [batch_size, in_height, in_width, in_channels]
    nchw_input_shape = [batch_size, in_channels, in_height, in_width]
    total_size = np.prod(nhwc_input_shape)

    # Construct the input tensor in data_type and NHWC.
    # force_cpu is needed because quantize_v2 runs on only CPU.
    with test_util.force_cpu():
      if data_type == dtypes.qint8:
        # Initialize the input tensor with qint8 values that circle -127..127.
        x = [((f + 128) % 255) - 127 for f in range(total_size)]
        t = constant_op.constant(
            x, shape=nhwc_input_shape, dtype=dtypes.float32)
        t, _, _ = gen_array_ops.quantize_v2(t, -128.0, 127.0, dtypes.qint8)
      else:
        assert data_type == dtypes.float32
        # Initialize the input tensor with ascending whole numbers as floats.
        x = [f * 1.0 for f in range(total_size)]
        shape = nchw_input_shape if data_format == "NCHW" else nhwc_input_shape
        t = constant_op.constant(x, shape=shape, dtype=dtypes.float32)

    with test_util.device(use_gpu):
      if data_format == "NCHW_VECT_C":
        assert data_type == dtypes.qint8

        # Convert to int8, then NHWCToNCHW_VECT_C, and then back to qint8.
        actual = array_ops.bitcast(t, dtypes.int8)
        actual = test_util.NHWCToNCHW_VECT_C(actual)
        actual = array_ops.bitcast(actual, dtypes.qint8)
        actual = array_ops.space_to_depth(
            actual, block_size, data_format=data_format)
        actual = array_ops.bitcast(actual, dtypes.int8)
        actual = test_util.NCHW_VECT_CToNHWC(actual)
        actual = array_ops.bitcast(actual, dtypes.qint8)

        expected = array_ops.bitcast(t, dtypes.int8)
        expected = math_ops.cast(expected, dtypes.float32)
        expected = self.spaceToDepthUsingTranspose(expected, block_size, "NHWC")
        expected = math_ops.cast(expected, dtypes.int8)
        expected = array_ops.bitcast(expected, dtypes.qint8)
      else:
        # Initialize the input tensor with ascending whole numbers as floats.
        actual = array_ops.space_to_depth(
            t, block_size, data_format=data_format)
        expected = self.spaceToDepthUsingTranspose(t, block_size, data_format)

      actual_vals, expected_vals = self.evaluate([actual, expected])
      self.assertTrue(np.array_equal(actual_vals, expected_vals))

  @test_util.disable_tfrt("b/169901260")
  def testAgainstTranspose(self):
    # [FUNCTION DOC]
    # Test (disabled under the TFRT runtime): runs the comparison helper on CPU for float32 and
    # quantized 8-bit data with several shapes. If no GPU is available it logs a skip message and
    # returns early; otherwise also runs GPU cases for NHWC, NCHW, and the vectorized channel
    # layout. Returns nothing.
    self.compareToTranspose(3, 2, 3, 1, 2, "NHWC", dtypes.float32, False)
    self.compareToTranspose(1, 2, 3, 2, 2, "NHWC", dtypes.float32, False)
    self.compareToTranspose(1, 2, 3, 2, 3, "NHWC", dtypes.float32, False)

    self.compareToTranspose(3, 2, 3, 1, 2, "NHWC", dtypes.qint8, False)
    self.compareToTranspose(1, 2, 3, 2, 2, "NHWC", dtypes.qint8, False)
    self.compareToTranspose(1, 2, 3, 2, 3, "NHWC", dtypes.qint8, False)

    if not test.is_gpu_available():
      tf_logging.info("skipping gpu tests since gpu not available")
      return

    self.compareToTranspose(3, 2, 3, 1, 2, "NHWC", dtypes.float32, True)
    self.compareToTranspose(3, 2, 3, 2, 2, "NHWC", dtypes.float32, True)
    self.compareToTranspose(3, 2, 3, 1, 2, "NCHW", dtypes.float32, True)
    self.compareToTranspose(3, 2, 3, 2, 3, "NCHW", dtypes.float32, True)
    self.compareToTranspose(5, 7, 11, 3, 2, "NCHW", dtypes.float32, True)

    self.compareToTranspose(3, 2, 3, 4, 2, "NCHW_VECT_C", dtypes.qint8, True)
    self.compareToTranspose(3, 2, 3, 8, 3, "NCHW_VECT_C", dtypes.qint8, True)
    self.compareToTranspose(5, 7, 11, 12, 2, "NCHW_VECT_C", dtypes.qint8, True)


class SpaceToDepthGradientTest(test.TestCase):
  # [CLASS DOC]
  # Numerical gradient checks for the space-to-depth op in NHWC and (GPU only) NCHW layouts.

  # Check the gradients.
  def _checkGrad(self, x, block_size, data_format):
    # [FUNCTION DOC]
    # HELPER: gradient check. Inputs: x (4-D float array, asserted to have four dimensions),
    # block_size (integer), data_format (NHWC or NCHW). Returns immediately for NCHW when no GPU
    # is available. Otherwise compares theoretical and numerical gradients of the op (relative and
    # absolute tolerance 1e-2) on a GPU-preferred session. Returns nothing.
    # NCHW is implemented for only GPU.
    if data_format == "NCHW" and not test.is_gpu_available():
      return

    assert 4 == x.ndim

    def func(x):
      # [FUNCTION DOC]
      # HELPER: closure passed to the gradient checker; applies space-to-depth with the enclosing
      # block size and data format to its input tensor and returns the result.
      return array_ops.space_to_depth(x, block_size, data_format=data_format)

    with test_util.use_gpu():
      with self.cached_session():
        theoretical, numerical = gradient_checker_v2.compute_gradient(
            func, [ops.convert_to_tensor(x)])
        self.assertAllClose(theoretical, numerical, rtol=1e-2, atol=1e-2)

  # Tests a gradient for space_to_depth of x which is a four dimensional
  # tensor of shape [b, h * block_size, w * block_size, d].
  def _compare(self, b, h, w, d, block_size, data_format):
    # [FUNCTION DOC]
    # HELPER: builds a random normal float32 tensor of batch b, height h times block size, width
    # w times block size and depth d (laid out as NHWC or NCHW per data_format), then runs the
    # gradient check. Returns nothing.
    block_size_sq = block_size * block_size
    data = np.random.normal(0, 1, b * h * w * d * block_size_sq).astype(
        np.float32)
    if data_format == "NHWC":
      x = data.reshape([b, h * block_size, w * block_size, d])
    else:
      x = data.reshape([b, d, h * block_size, w * block_size])

    self._checkGrad(x, block_size, data_format)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    # [FUNCTION DOC]
    # Test: gradient check with block size 2 for a small input in NHWC and NCHW layouts.
    # Returns nothing.
    block_size = 2
    self._compare(1, 2, 3, 5, block_size, "NHWC")
    self._compare(1, 2, 3, 5, block_size, "NCHW")

  @test_util.run_deprecated_v1
  def testSmall2(self):
    # [FUNCTION DOC]
    # Test: gradient check with block size 2 for a second small input shape (batch 2) in NHWC and
    # NCHW layouts. Returns nothing.
    block_size = 2
    self._compare(2, 4, 3, 2, block_size, "NHWC")
    self._compare(2, 4, 3, 2, block_size, "NCHW")


if __name__ == "__main__":
  test.main()
