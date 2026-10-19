# [MODULE DOC]
# Functional tests for the depth_to_space op, which rearranges channel data into spatial blocks.
# Covers basic and larger block sizes, batches including size zero, unequal height and width, depth
# interleaving, error cases (block size too large, zero, one, not dividing depth), unknown shapes,
# a comparison against a reshape/transpose reference for NHWC, NCHW and NCHW_VECT_C layouts, and
# numeric gradient checks. Parameterized over dtypes for the basic test. GPU checks only run when a
# GPU exists. No import-time side effects; running directly starts the test main entry point.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Functional tests for DepthToSpace op."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import device_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class DepthToSpaceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the depth_to_space op: functional cases with a shared CPU/GPU helper,
  # error handling, and a transpose-based reference implementation for comparison.

  def _testOne(self, inputs, block_size, outputs, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: shared check. Takes inputs (nested list in height-width-channel layout), block_size,
    # expected outputs and a dtype (default 32-bit float). Casts inputs, runs depth_to_space on CPU and
    # compares with outputs. If only CPU devices exist, it also checks that the channels-first layout
    # raises an invalid-argument error saying no kernel is registered. If a GPU is available, it checks
    # both layouts on GPU against the expected outputs.
    input_nhwc = math_ops.cast(inputs, dtype)
    with self.cached_session(use_gpu=False):
      # test NHWC (default) on CPU
      x_tf = array_ops.depth_to_space(input_nhwc, block_size)
      self.assertAllEqual(x_tf, outputs)

      # Run this test only if only CPU device is available
      if all(x.device_type == "CPU" for x in device_lib.list_local_devices()):
        input_nchw = test_util.NHWCToNCHW(input_nhwc)
        output_nchw = array_ops.depth_to_space(
            input_nchw, block_size, data_format="NCHW")
        output_nhwc = test_util.NCHWToNHWC(output_nchw)
        with self.assertRaisesRegex(
            errors_impl.InvalidArgumentError,
            "No OpKernel was registered to support Op 'DepthToSpace'"):
          self.evaluate(output_nhwc)

    if test.is_gpu_available():
      with self.cached_session():
        # test NHWC (default) on GPU
        x_tf = array_ops.depth_to_space(input_nhwc, block_size)
        self.assertAllEqual(x_tf, outputs)
        # test NCHW on GPU
        input_nchw = test_util.NHWCToNCHW(input_nhwc)
        output_nchw = array_ops.depth_to_space(
            input_nchw, block_size, data_format="NCHW")
        output_nhwc = test_util.NCHWToNHWC(output_nchw)
        self.assertAllEqual(output_nhwc, outputs)

  @parameterized.parameters(dtypes.float32, dtypes.float16, dtypes.bfloat16)
  @test_util.run_deprecated_v1
  def testBasic(self, dtype):
    # [FUNCTION DOC]
    # Tests the simplest case, parameterized over float32, float16 and bfloat16: a single pixel with
    # four channels and block size two becomes a two by two image, checked via the shared helper.
    x_np = [[[[1, 2, 3, 4]]]]
    block_size = 2
    x_out = [[[[1], [2]], [[3], [4]]]]
    self._testOne(x_np, block_size, x_out, dtype)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered spatially.
  @test_util.run_deprecated_v1
  def testBlockSize2(self):
    # [FUNCTION DOC]
    # Tests a two by two image with four channels and block size two; verifies elements are ordered
    # correctly in the spatial output.
    x_np = [[[[1, 2, 3, 4],
              [5, 6, 7, 8]],
             [[9, 10, 11, 12],
              [13, 14, 15, 16]]]]
    block_size = 2
    x_out = [[[[1], [2], [5], [6]],
              [[3], [4], [7], [8]],
              [[9], [10], [13], [14]],
              [[11], [12], [15], [16]]]]
    self._testOne(x_np, block_size, x_out)

  @test_util.run_deprecated_v1
  def testBlockSizeOverflow(self):
    # [FUNCTION DOC]
    # Tests in eager mode that an absurdly large block size (one hundred thousand) raises an
    # invalid-argument error.
    with context.eager_mode():
      x_np = [[[[1, 2, 3, 4]]]]
      block_size = 100000
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(array_ops.depth_to_space(x_np, block_size))

  @test_util.run_deprecated_v1
  def testBlockSize2Batch10(self):
    # [FUNCTION DOC]
    # Tests block size two on a batch of ten inputs, each element scaled by its batch index. Builds
    # input and expected output through two small local generators.
    block_size = 2
    def batch_input_elt(i):
      # [FUNCTION DOC]
      # HELPER: returns the two by two by four input image with every value multiplied by index i.
      return [[[1 * i, 2 * i, 3 * i, 4 * i],
               [5 * i, 6 * i, 7 * i, 8 * i]],
              [[9 * i, 10 * i, 11 * i, 12 * i],
               [13 * i, 14 * i, 15 * i, 16 * i]]]
    def batch_output_elt(i):
      # [FUNCTION DOC]
      # HELPER: returns the expected four by four by one output image with every value multiplied by index i.
      return [[[1 * i], [2 * i], [5 * i], [6 * i]],
              [[3 * i], [4 * i], [7 * i], [8 * i]],
              [[9 * i], [10 * i], [13 * i], [14 * i]],
              [[11 * i], [12 * i], [15 * i], [16 * i]]]
    batch_size = 10
    x_np = [batch_input_elt(i) for i in range(batch_size)]
    x_out = [batch_output_elt(i) for i in range(batch_size)]
    self._testOne(x_np, block_size, x_out)

  def testBatchSize0(self):
    # [FUNCTION DOC]
    # Tests an empty batch (size zero): the output static shape equals that of an expected zero-batch
    # tensor of four by six by three and evaluating does not fail, on CPU and, if available, GPU.
    block_size = 2
    batch_size = 0
    input_nhwc = array_ops.ones([batch_size, 2, 3, 12])
    x_out = array_ops.ones([batch_size, 4, 6, 3])

    with self.cached_session(use_gpu=False):
      # test NHWC (default) on CPU
      x_tf = array_ops.depth_to_space(input_nhwc, block_size)
      self.assertAllEqual(x_tf.shape, x_out.shape)
      self.evaluate(x_tf)
    if test.is_gpu_available():
      with self.cached_session():
        # test NHWC (default) on GPU
        x_tf = array_ops.depth_to_space(input_nhwc, block_size)
        self.assertAllEqual(x_tf.shape, x_out.shape)
        self.evaluate(x_tf)

  # Tests for different width and height.
  @test_util.run_deprecated_v1
  def testNonSquare(self):
    # [FUNCTION DOC]
    # Tests an image of different height and width (three rows, one column) with eight channels and
    # block size two.
    x_np = [[[[1, 10, 2, 20, 3, 30, 4, 40]],
             [[5, 50, 6, 60, 7, 70, 8, 80]],
             [[9, 90, 10, 100, 11, 110, 12, 120]]]]
    block_size = 2
    x_out = [[[[1, 10], [2, 20]],
              [[3, 30], [4, 40]],
              [[5, 50], [6, 60]],
              [[7, 70], [8, 80]],
              [[9, 90], [10, 100]],
              [[11, 110], [12, 120]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered spatially.
  @test_util.run_deprecated_v1
  def testBlockSize4FlatInput(self):
    # [FUNCTION DOC]
    # Tests a single pixel with sixteen channels and block size four; ordering of elements must match
    # the expected four by four image.
    x_np = [[[[1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]]]]
    block_size = 4
    x_out = [[[[1], [2], [5], [6]],
              [[3], [4], [7], [8]],
              [[9], [10], [13], [14]],
              [[11], [12], [15], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths.
  # To make sure elements are properly interleaved in depth.
  @test_util.run_deprecated_v1
  def testDepthInterleaved(self):
    # [FUNCTION DOC]
    # Tests that multi-channel output keeps interleaved depth: eight channels, block size two give pairs
    # of values in two by two blocks.
    x_np = [[[[1, 10, 2, 20, 3, 30, 4, 40]]]]
    block_size = 2
    x_out = [[[[1, 10], [2, 20]],
              [[3, 30], [4, 40]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths. Here an odd depth.
  # To make sure elements are properly interleaved in depth.
  @test_util.run_deprecated_v1
  def testDepthInterleavedDepth3(self):
    # [FUNCTION DOC]
    # Tests interleaving with an odd output depth of three: twelve channels and block size two.
    x_np = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    block_size = 2
    x_out = [[[[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths.
  # To make sure elements are properly interleaved in depth.
  @test_util.run_deprecated_v1
  def testDepthInterleavedLarger(self):
    # [FUNCTION DOC]
    # Tests interleaving for a larger two by two image with eight channels and block size two.
    x_np = [[[[1, 10, 2, 20, 3, 30, 4, 40],
              [5, 50, 6, 60, 7, 70, 8, 80]],
             [[9, 90, 10, 100, 11, 110, 12, 120],
              [13, 130, 14, 140, 15, 150, 16, 160]]]]
    block_size = 2
    x_out = [[[[1, 10], [2, 20], [5, 50], [6, 60]],
              [[3, 30], [4, 40], [7, 70], [8, 80]],
              [[9, 90], [10, 100], [13, 130], [14, 140]],
              [[11, 110], [12, 120], [15, 150], [16, 160]]]]
    self._testOne(x_np, block_size, x_out)

  # Error handling:

  # Tests for a block larger for the depth. In this case should raise an
  # exception.
  @test_util.run_deprecated_v1
  def testBlockSizeTooLarge(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment says a block of four does not fit into depth four; the depth must be
    # divisible by sixteen, so the error comes from the depth divisibility rule.
    # Tests that block size four on depth four raises a value error.
    x_np = [[[[1, 2, 3, 4],
              [5, 6, 7, 8]],
             [[9, 10, 11, 12],
              [13, 14, 15, 16]]]]
    block_size = 4
    # Raise an exception, since th depth is only 4 and needs to be
    # divisible by 16.
    with self.assertRaises(ValueError):
      out_tf = array_ops.depth_to_space(x_np, block_size)
      self.evaluate(out_tf)

  # Test when the block size is 0.
  @test_util.run_deprecated_v1
  def testBlockSize0(self):
    # [FUNCTION DOC]
    # Tests that block size zero raises a value error.
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = array_ops.depth_to_space(x_np, block_size)
      self.evaluate(out_tf)

  # Test when the block size is 1. The block size should be > 1.
  @test_util.run_deprecated_v1
  def testBlockSizeOne(self):
    # [FUNCTION DOC]
    # Tests that block size one (not greater than one) raises a value error.
    x_np = [[[[1, 1, 1, 1],
              [2, 2, 2, 2]],
             [[3, 3, 3, 3],
              [4, 4, 4, 4]]]]
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = array_ops.depth_to_space(x_np, block_size)
      self.evaluate(out_tf)

  @test_util.run_deprecated_v1
  def testBlockSizeLargerThanInput(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite living in the depth_to_space tests, this calls the space_to_depth op, so it
    # does not exercise depth_to_space at all.
    # Tests that a block size of ten, larger than the two by two input, raises a value error.
    # The block size is too large for this input.
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 10
    with self.assertRaises(ValueError):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      self.evaluate(out_tf)

  @test_util.run_deprecated_v1
  def testBlockSizeNotDivisibleDepth(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment says depth is not divisible by the block size squared, but the call is
    # space_to_depth, not depth_to_space, and its two by two spatial size is not divisible by block size
    # three, which is the actual reason for the error.
    # Tests that block size three raises a value error.
    # The depth is not divisible by the square of the block size.
    x_np = [[[[1, 1, 1, 1],
              [2, 2, 2, 2]],
             [[3, 3, 3, 3],
              [4, 4, 4, 4]]]]
    block_size = 3
    with self.assertRaises(ValueError):
      _ = array_ops.space_to_depth(x_np, block_size)

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Tests that depth_to_space on a placeholder of unknown shape with block size four still gives a
    # result with known rank four.
    t = array_ops.depth_to_space(
        array_ops.placeholder(dtypes.float32), block_size=4)
    self.assertEqual(4, t.get_shape().ndims)

  def depthToSpaceUsingTranspose(self, tensor, block_size, data_format):
    # [FUNCTION DOC]
    # HELPER: reference implementation of depth_to_space using reshape and transpose. Inputs: a four
    # dimensional tensor, block_size and data_format. For the channels-last layout or the channels-first
    # layout, it reshapes channels into block factors, transposes and reshapes into the output shape,
    # and asserts the channel count is divisible by the squared block size. Returns the rearranged
    # tensor; for any other layout name it returns the input unchanged.
    block_size_sq = block_size * block_size
    if data_format == "NHWC":
      b, ih, iw, ic = tensor.shape.as_list()
      assert ic % block_size_sq == 0, (ic, block_size_sq)
      ow, oh, oc = iw * block_size, ih * block_size, ic // block_size_sq
      tensor = array_ops.reshape(tensor,
                                 [b, ih, iw, block_size, block_size, oc])
      tensor = array_ops.transpose(tensor, [0, 1, 3, 2, 4, 5])
      tensor = array_ops.reshape(tensor, [b, oh, ow, oc])
    elif data_format == "NCHW":
      b, ic, ih, iw = tensor.shape.as_list()
      assert ic % block_size_sq == 0, (ic, block_size_sq)
      ow, oh, oc = iw * block_size, ih * block_size, ic // block_size_sq
      tensor = array_ops.reshape(tensor,
                                 [b, block_size, block_size, oc, ih, iw])
      tensor = array_ops.transpose(tensor, [0, 3, 4, 1, 5, 2])
      tensor = array_ops.reshape(tensor, [b, oc, oh, ow])
    return tensor

  def compareToTranspose(self, batch_size, in_height, in_width, out_channels,
                         block_size, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: compares depth_to_space with the reference. Inputs: batch size, input height and width,
    # output channels, block size, data format (channels-last, channels-first or the vectorized
    # quantized layout) and whether to use the GPU. For the vectorized layout, builds values that
    # cycle across the signed 8-bit range, converts layout, quantizes, runs the op and dequantizes;
    # otherwise uses ascending floats. Evaluates both results in a session and asserts equality.
    in_channels = out_channels * block_size * block_size
    nhwc_input_shape = [batch_size, in_height, in_width, in_channels]
    nchw_input_shape = [batch_size, in_channels, in_height, in_width]
    total_size = np.prod(nhwc_input_shape)

    if data_format == "NCHW_VECT_C":
      # Initialize the input tensor with qint8 values that circle -127..127.
      x = [((f + 128) % 255) - 127 for f in range(total_size)]
      t = constant_op.constant(x, shape=nhwc_input_shape, dtype=dtypes.float32)
      expected = self.depthToSpaceUsingTranspose(t, block_size, "NHWC")
      t = test_util.NHWCToNCHW_VECT_C(t)
      t, _, _ = gen_array_ops.quantize_v2(t, -128.0, 127.0, dtypes.qint8)
      t = array_ops.depth_to_space(t, block_size, data_format="NCHW_VECT_C")
      t = gen_array_ops.dequantize(t, -128, 127)
      actual = test_util.NCHW_VECT_CToNHWC(t)
    else:
      # Initialize the input tensor with ascending whole numbers as floats.
      x = [f * 1.0 for f in range(total_size)]
      shape = nchw_input_shape if data_format == "NCHW" else nhwc_input_shape
      t = constant_op.constant(x, shape=shape, dtype=dtypes.float32)
      expected = self.depthToSpaceUsingTranspose(t, block_size, data_format)
      actual = array_ops.depth_to_space(t, block_size, data_format=data_format)

    with self.session(use_gpu=use_gpu) as sess:
      actual_vals, expected_vals = self.evaluate([actual, expected])
      self.assertTrue(np.array_equal(actual_vals, expected_vals))

  def testAgainstTranspose(self):
    # [FUNCTION DOC]
    # Tests depth_to_space against the reference on CPU for several channels-last sizes. When no GPU is
    # available it logs a skip message and returns; otherwise also runs GPU cases for all three layouts
    # and many sizes.
    self.compareToTranspose(3, 2, 3, 1, 2, "NHWC", False)
    self.compareToTranspose(3, 2, 3, 2, 2, "NHWC", False)
    self.compareToTranspose(1, 2, 3, 2, 3, "NHWC", False)

    if not test.is_gpu_available():
      tf_logging.info("skipping gpu tests since gpu not available")
      return

    self.compareToTranspose(3, 2, 3, 1, 2, "NHWC", True)
    self.compareToTranspose(3, 2, 3, 2, 2, "NHWC", True)
    self.compareToTranspose(3, 2, 3, 1, 2, "NCHW", True)
    self.compareToTranspose(3, 2, 3, 2, 2, "NCHW", True)
    self.compareToTranspose(3, 2, 3, 1, 3, "NCHW", True)
    self.compareToTranspose(3, 2, 3, 2, 3, "NCHW", True)
    self.compareToTranspose(5, 7, 11, 3, 2, "NCHW", True)
    self.compareToTranspose(3, 200, 300, 32, 2, "NCHW", True)

    self.compareToTranspose(3, 2, 3, 8, 2, "NCHW_VECT_C", True)
    self.compareToTranspose(3, 2, 3, 4, 3, "NCHW_VECT_C", True)
    self.compareToTranspose(3, 2, 3, 8, 3, "NCHW_VECT_C", True)
    self.compareToTranspose(5, 7, 11, 12, 2, "NCHW_VECT_C", True)
    self.compareToTranspose(3, 200, 300, 32, 2, "NCHW_VECT_C", True)


class DepthToSpaceGradientTest(test.TestCase):
  # [CLASS DOC]
  # Test case that checks the numeric gradient of depth_to_space.

  # Check the gradients.
  def _checkGrad(self, x, block_size, data_format):
    # [FUNCTION DOC]
    # HELPER: gradient check for an input array x (four dimensional), block_size and data_format. Returns
    # immediately for channels-first when no GPU exists. Computes theoretical and numeric Jacobians with
    # a step of one hundredth and asserts they are close (tolerances 1e-2 relative, step size absolute).
    # NCHW is implemented for only GPU.
    if data_format == "NCHW" and not test.is_gpu_available():
      return

    assert 4 == x.ndim
    with self.cached_session():
      tf_x = ops.convert_to_tensor(x)
      tf_y = array_ops.depth_to_space(tf_x, block_size, data_format=data_format)

      epsilon = 1e-2
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)
      self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for depth_to_space of x which is a four dimensional
  # tensor of shape [b, h, w, d * block_size * block_size].
  def _compare(self, b, h, w, d, block_size, data_format):
    # [FUNCTION DOC]
    # HELPER: builds random normal float input of the requested layout from batch, height, width,
    # output depth and block size, then runs the gradient check. Returns nothing.
    block_size_sq = block_size * block_size
    data = np.random.normal(0, 1, b * h * w * d * block_size_sq).astype(
        np.float32)
    if data_format == "NHWC":
      x = data.reshape([b, h, w, d * block_size_sq])
    else:
      x = data.reshape([b, d * block_size_sq, h, w])

    self._checkGrad(x, block_size, data_format)

  # Don't use very large numbers as dimensions here, as the result is tensor
  # with cartesian product of the dimensions.
  @test_util.run_deprecated_v1
  def testSmall(self):
    # [FUNCTION DOC]
    # Tests the gradient for block size two with small dimensions in both layouts.
    block_size = 2
    self._compare(3, 2, 5, 3, block_size, "NHWC")
    self._compare(3, 2, 5, 3, block_size, "NCHW")

  @test_util.run_deprecated_v1
  def testSmall2(self):
    # [FUNCTION DOC]
    # Tests the gradient for block size three with small dimensions in both layouts.
    block_size = 3
    self._compare(1, 2, 3, 2, block_size, "NHWC")
    self._compare(1, 2, 3, 2, block_size, "NCHW")


if __name__ == "__main__":
  test.main()
