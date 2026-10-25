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
# [MODULE DOC]
# Functional tests for the 2-D grayscale morphological ops: dilation and erosion, both forward
# values and gradients. Contains two test classes (DilationTest, ErosionTest) that run small
# hand-computed cases (valid/same padding, depth, batch, non-square window, atrous rate, uneven
# stride) on CPU and GPU settings, and numerically check gradients with the v2 gradient checker.
# Importing it also pulls in the nn gradient registrations. Running as a script starts the
# standard test runner.
"""Functional tests for morphological filtering operations."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class DilationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the 2-D dilation op (forward values and gradients) in float32 and bfloat16,
  # parameterized by data type. Helper methods with a leading underscore build individual
  # scenarios; the public test methods loop over them for GPU and CPU placement.

  def _VerifyValues(self, image, kernel, strides, rates, padding, out, use_gpu,
                    dtype):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for the dilation forward-value scenarios.
    # KNOWN ISSUE: the docstring does not list the dtype parameter, which is required and is
    # the data type used to build both the image and kernel constants.
    # Wraps strides and rates (two element lists for height and width) with leading and
    # trailing ones, runs the dilation op on constants made from image and kernel in a cached
    # session on the requested device, and asserts the result is close to the expected out
    # with a tolerance chosen according to the data type. Returns nothing.
    """Verifies the output values of the dilation function.

    Args:
      image: Input tensor with shape: [batch, in_height, in_width, channels].
      kernel: Filter tensor with shape: [filter_height, filter_width, channels].
      strides: Output strides, specified as [stride_height, stride_width].
      rates: Atrous rates, specified as [rate_height, rate_width].
      padding: Padding type.
      out: Expected output.
      use_gpu: Whether we are running on GPU.
    """
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]

    with self.cached_session(use_gpu=use_gpu):
      out_tensor = nn_ops.dilation2d(
          constant_op.constant(image, dtype=dtype),
          constant_op.constant(kernel, dtype=dtype),
          strides=strides,
          rates=rates,
          padding=padding,
          name="dilation2d")
      self.assertAllCloseAccordingToType(out, self.evaluate(out_tensor))

  def _testDilationValidPadding(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: scenario for testDilation. A 2x2 single channel image with a 2x2 kernel,
    # unit strides and rates and valid padding; expects a single output value of 0.5.
    # [1, 2, 2, 1]
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    # [1, 1, 1, 1]
    out = [[[[.5]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationSamePadding(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: scenario for testDilation. Same 2x2 image and 2x2 kernel as the valid-padding
    # case but with same padding; expects a 2x2 output of 0.5, 0.6, 0.7, 0.8.
    # [1, 2, 2, 1]
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    # [1, 2, 2, 1]
    out = [[[[.5], [.6]], [[.7], [.8]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationSamePaddingDepth(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: scenario for testDilation. A 2x2 image with three channels and a 2x2x3 kernel,
    # same padding, unit strides and rates; each channel is dilated independently, and the
    # expected 2x2x3 output is compared.
    # [1, 2, 2, 3]
    image = [[[[.1, .2, .0], [.2, .3, .1]], [[.3, .4, .2], [.4, .5, .3]]]]
    # [2, 2, 3]
    kernel = [[[.4, .5, .3], [.3, .4, .2]], [[.1, .2, .0], [.0, .1, -.1]]]
    # [1, 2, 2, 3]
    out = [[[[.5, .7, .3], [.6, .8, .4]], [[.7, .9, .5], [.8, 1., .6]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationSamePaddingBatch(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: scenario for testDilation. A batch of two 2x2 single channel images with a 2x2
    # kernel and same padding; checks each batch item is dilated independently.
    # [2, 2, 2, 1]
    image = [[[[.1], [.2]], [[.3], [.4]]], [[[.2], [.3]], [[.4], [.5]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    # [2, 2, 2, 1]
    out = [[[[.5], [.6]], [[.7], [.8]]], [[[.6], [.7]], [[.8], [.9]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationValidPaddingNonSquareWindow(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: scenario for testDilation. A 2x2 image with a 1x2 (non square) kernel and valid
    # padding; expects a 2x1 output of 0.5 and 0.7.
    # [1, 2, 2, 1]
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    # [1, 2, 1]
    kernel = [[[.4], [.3]]]
    # [1, 2, 1, 1]
    out = [[[[.5]], [[.7]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationSamePaddingRate(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: scenario for testDilation. A 3x3 image with a 2x2 kernel using an atrous rate
    # of 2 in both directions (effective 3x3 kernel with zero gaps) and same padding; compares
    # against a hand computed 3x3 output.
    # [1, 3, 3, 1]
    image = [[[[.1], [.2], [.3]], [[.4], [.5], [.6]], [[.7], [.8], [.9]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.2]]]
    # Because rate = 2.0, the effective kernel is [3, 3, 1]:
    # kernel_eff = [[[.4], [.0], [.3]],
    #               [[.0], [.0], [.0]],
    #               [[.1], [.0], [.2]]]
    # [1, 3, 3, 1]
    out = [[[[.7], [.8], [.6]], [[1.0], [1.1], [.9]], [[.8], [.9], [.9]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[2, 2],
        padding="SAME",
        out=out,
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationValidPaddingUnevenStride(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: scenario for testDilation. A 3x4 image with a 2x2 kernel, strides of 1 in height
    # and 2 in width, valid padding; expects a 2x2 output.
    # KNOWN ISSUE: a comment in the code labels the image as shape 1x3x3x1 but it is actually
    # 1x3x4x1.
    # [1, 3, 3, 1]
    image = [[[[.1], [.2], [.3], [.4]], [[.5], [.6], [.7], [.8]],
              [[.9], [1.0], [1.1], [1.2]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.2]]]
    # [1, 2, 2, 1]
    out = [[[[.8], [1.0]], [[1.2], [1.4]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 2],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu,
        dtype=dtype)

  @parameterized.parameters(dtypes.float32, dtypes.bfloat16)
  def testDilation(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float32 and bfloat16. For both GPU and CPU placement, runs all seven
    # dilation forward-value scenarios (valid, same, depth, batch, non square window, rate,
    # uneven stride) and expects every result to match the hand computed values.
    for use_gpu in True, False:
      self._testDilationValidPadding(use_gpu, dtype)
      self._testDilationSamePadding(use_gpu, dtype)
      self._testDilationSamePaddingDepth(use_gpu, dtype)
      self._testDilationSamePaddingBatch(use_gpu, dtype)
      self._testDilationValidPaddingNonSquareWindow(use_gpu, dtype)
      self._testDilationSamePaddingRate(use_gpu, dtype)
      self._testDilationValidPaddingUnevenStride(use_gpu, dtype)

  def _ConstructAndTestGradient(self,
                                image_shape,
                                kernel_shape,
                                strides,
                                rates,
                                padding,
                                use_gpu,
                                dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: shared gradient checker for the dilation gradient scenarios.
    # KNOWN ISSUE: the docstring omits the dtype parameter (default float32). Also the random
    # image and kernel are always generated as float32 numpy arrays and then converted to
    # the constants of the requested dtype.
    # Inputs: image_shape (batch, height, width, channels), kernel_shape (height, width,
    # channels), strides and rates (two element lists), padding string, use_gpu flag, dtype.
    # Asserts the channel counts match (assertion error otherwise), seeds numpy randomness
    # with a fixed value, builds random constants, and computes the gradient error of the
    # dilation op with respect to both the image and the kernel using the v2 gradient
    # checker on the chosen device. Prints the larger error and asserts it is below 4.0 for
    # bfloat16 or 1e-4 otherwise. Returns nothing.
    """Verifies the gradients of the dilation function.

    Args:
      image_shape: Input shape, [batch, in_height, in_width, channels].
      kernel_shape: Filter shape, [filter_height, filter_width, channels].
      strides: Output strides, specified as [stride_height, stride_width].
      rates: Atrous rates, specified as [rate_height, rate_width].
      padding: Padding type.
      use_gpu: Whether we are running on GPU.
    """
    assert image_shape[3] == kernel_shape[2]

    np.random.seed(1)  # Make it reproducible.
    image = np.random.random_sample(image_shape).astype(np.float32)
    kernel = np.random.random_sample(kernel_shape).astype(np.float32)

    strides = [1] + strides + [1]
    rates = [1] + rates + [1]

    image_tensor = constant_op.constant(
        image, shape=image_shape, name="input", dtype=dtype)
    kernel_tensor = constant_op.constant(
        kernel, shape=kernel_shape, name="filter", dtype=dtype)

    def compute_dilation2d(image_tensor, kernel_tensor):
      # [FUNCTION DOC]
      # HELPER: closure used as the function under differentiation. Applies the dilation op
      # to the given image and kernel tensors using the enclosing strides, rates and padding,
      # and returns the output tensor.
      return nn_ops.dilation2d(
          image_tensor,
          kernel_tensor,
          strides=strides,
          rates=rates,
          padding=padding,
          name="dilation2d")

    with test_util.device(use_gpu=use_gpu):
      with self.cached_session():
        # Small delta is necessary for argmax to remain the same.
        err1 = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: compute_dilation2d(x, kernel_tensor), [image_tensor]))
        err2 = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: compute_dilation2d(image_tensor, x), [kernel_tensor]))
        err = max(err1, err2)

    print("Dilation gradient error = %f" % err)
    if dtype == dtypes.bfloat16:
      self.assertLess(err, 4.0)
    else:
      self.assertLess(err, 1e-4)

  def _testDilationGradValidPadding_1x1x1(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testDilationGrad. 3x3x1 image, 1x1x1 kernel, valid padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[1, 1, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationGradDeterminismError(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testDilationGrad that enables op determinism first.
    # When a CUDA GPU is requested and available, expects the gradient computation to raise an
    # unimplemented error saying determinism is not supported for the dilation input
    # backprop. Otherwise (CPU or no CUDA) expects the gradient check to succeed normally.
    # Side effect: turns global op determinism on and always turns it off again afterwards.
    if use_gpu and test.is_gpu_available(cuda_only=True):
      try:
        config.enable_op_determinism()
        with self.assertRaisesRegex(
            errors_impl.UnimplementedError, "Determinism is not yet supported "
            "for Dilation2DBackpropInput."):
          self._ConstructAndTestGradient(
              image_shape=[1, 3, 3, 1],
              kernel_shape=[1, 1, 1],
              strides=[1, 1],
              rates=[1, 1],
              padding="VALID",
              use_gpu=use_gpu,
              dtype=dtype)
      finally:
        config.disable_op_determinism()
    else:
      try:
        config.enable_op_determinism()
        self._ConstructAndTestGradient(
            image_shape=[1, 3, 3, 1],
            kernel_shape=[1, 1, 1],
            strides=[1, 1],
            rates=[1, 1],
            padding="VALID",
            use_gpu=use_gpu,
            dtype=dtype)
      finally:
        config.disable_op_determinism()

  def _testDilationGradSamePadding_1x1x1(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testDilationGrad. 3x3x1 image, 1x1x1 kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[1, 1, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationGradSamePadding_1x1x2(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testDilationGrad. 3x3x2 image, 1x1x2 kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 2],
        kernel_shape=[1, 1, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationGradValidPadding_2x2x1(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testDilationGrad. 3x3x1 image, 2x2x1 kernel, valid padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationGradSamePadding_2x2x1(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testDilationGrad. 3x3x1 image, 2x2x1 kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationGradSamePaddingBatch_2x2x1(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testDilationGrad. Batch of four 3x3x1 images, 2x2x1
    # kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[4, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu,
        dtype=dtype)

  def _testDilationGradSamePadding_2x2x4(self, use_gpu, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testDilationGrad. 3x3x4 image, 2x2x4 kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 4],
        kernel_shape=[2, 2, 4],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu,
        dtype=dtype)

  @parameterized.parameters(dtypes.float32, dtypes.bfloat16)
  def testDilationGrad(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float32 and bfloat16. For GPU and CPU placement, runs the determinism
    # scenario and the seven dilation gradient scenarios, expecting gradient error within the
    # tolerance of the shared checker.
    for use_gpu in True, False:
      self._testDilationGradDeterminismError(use_gpu, dtype)
      self._testDilationGradValidPadding_1x1x1(use_gpu, dtype)
      self._testDilationGradSamePadding_1x1x1(use_gpu, dtype)
      self._testDilationGradSamePadding_1x1x2(use_gpu, dtype)
      self._testDilationGradValidPadding_2x2x1(use_gpu, dtype)
      self._testDilationGradSamePadding_2x2x1(use_gpu, dtype)
      self._testDilationGradSamePaddingBatch_2x2x1(use_gpu, dtype)
      self._testDilationGradSamePadding_2x2x4(use_gpu, dtype)


class ErosionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the 2-D erosion op (forward values and gradients) in float32. Mirrors the
  # dilation tests: private helpers build scenarios and public tests loop over GPU and CPU.

  def _VerifyValues(self, image, kernel, strides, rates, padding, out, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for the erosion forward-value scenarios.
    # Wraps strides and rates (two element lists) with leading and trailing ones, runs the
    # erosion op on constants made from image and kernel in a cached session on the requested
    # device, and asserts the result is close to the expected out. Returns nothing.
    """Verifies the output values of the erosion function.

    Args:
      image: Input tensor with shape: [batch, in_height, in_width, channels].
      kernel: Filter tensor with shape: [filter_height, filter_width, channels].
      strides: Output strides, specified as [stride_height, stride_width].
      rates: Atrous rates, specified as [rate_height, rate_width].
      padding: Padding type.
      out: Expected output.
      use_gpu: Whether we are running on GPU.
    """
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]

    with self.cached_session(use_gpu=use_gpu):
      out_tensor = nn_ops.erosion2d(
          constant_op.constant(image),
          constant_op.constant(kernel),
          strides=strides,
          rates=rates,
          padding=padding,
          name="erosion2d")
      self.assertAllClose(out, self.evaluate(out_tensor))

  def _testErosionValidPadding(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testErosion. 2x2 image, 2x2 kernel, valid padding; expects 0.0.
    # [1, 2, 2, 1]
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    # [1, 1, 1, 1]
    out = [[[[.0]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)

  def _testErosionSamePadding(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testErosion. 2x2 image, 2x2 kernel, same padding; expects a 2x2
    # output of 0.0, 0.1, 0.3, 0.4.
    # [1, 2, 2, 1]
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    # [1, 2, 2, 1]
    out = [[[[.0], [.1]], [[.3], [.4]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)

  def _testErosionSamePaddingDepth(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testErosion. 2x2 image with three channels, 2x2x3 kernel, same
    # padding; each channel eroded independently and compared with expected 2x2x3 output.
    # [1, 2, 2, 3]
    image = [[[[.1, .2, .0], [.2, .3, .1]], [[.3, .4, .2], [.4, .5, .3]]]]
    # [2, 2, 3]
    kernel = [[[.4, .5, .3], [.3, .4, .2]], [[.1, .2, .0], [.0, .1, -.1]]]
    # [1, 2, 2, 3]
    out = [[[[.0, .0, .0], [.1, .1, .1]], [[.3, .3, .3], [.4, .4, .4]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)

  def _testErosionSamePaddingBatch(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testErosion. Batch of two 2x2 single channel images, 2x2 kernel,
    # same padding; checks each batch item is eroded independently.
    # [2, 2, 2, 1]
    image = [[[[.1], [.2]], [[.3], [.4]]], [[[.2], [.3]], [[.4], [.5]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.0]]]
    # [2, 2, 2, 1]
    out = [[[[.0], [.1]], [[.3], [.4]]], [[[.1], [.2]], [[.4], [.5]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)

  def _testErosionValidPaddingNonSquareWindow(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testErosion. 2x2 image with a 1x2 kernel, valid padding; expects a
    # 2x1 output of -0.2 and 0.0.
    # [1, 2, 2, 1]
    image = [[[[.1], [.2]], [[.3], [.4]]]]
    # [1, 2, 1]
    kernel = [[[.4], [.3]]]
    # [1, 2, 1, 1]
    out = [[[[-.2]], [[.0]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)

  def _testErosionSamePaddingRate(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testErosion. 3x3 image, 2x2 kernel, atrous rate 2 in both
    # directions (effective 3x3 kernel), same padding; compares to a hand computed output.
    # [1, 3, 3, 1]
    image = [[[[.1], [.2], [.3]], [[.4], [.5], [.6]], [[.7], [.8], [.9]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.2]]]
    # Because rate = 2.0, the effective kernel is [3, 3, 1]:
    # kernel_eff = [[[.4], [.0], [.3]],
    #               [[.0], [.0], [.0]],
    #               [[.1], [.0], [.2]]]
    # [1, 3, 3, 1]
    out = [[[[.1], [.1], [.2]], [[0.1], [-.1], [.0]], [[.4], [.2], [.3]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 1],
        rates=[2, 2],
        padding="SAME",
        out=out,
        use_gpu=use_gpu)

  def _testErosionValidPaddingUnevenStride(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testErosion. 3x4 image, 2x2 kernel, strides 1 by 2, valid padding;
    # expects a 2x2 output.
    # KNOWN ISSUE: a comment in the code labels the image as shape 1x3x3x1 but it is actually
    # 1x3x4x1.
    # [1, 3, 3, 1]
    image = [[[[.1], [.2], [.3], [.4]], [[.5], [.6], [.7], [.8]],
              [[.9], [1.0], [1.1], [1.2]]]]
    # [2, 2, 1]
    kernel = [[[.4], [.3]], [[.1], [.2]]]
    # [1, 2, 2, 1]
    out = [[[[-.1], [.1]], [[.3], [.5]]]]
    self._VerifyValues(
        image,
        kernel,
        strides=[1, 2],
        rates=[1, 1],
        padding="VALID",
        out=out,
        use_gpu=use_gpu)

  def testErosion(self):
    # [FUNCTION DOC]
    # For both GPU and CPU placement, runs all seven erosion forward-value scenarios and
    # expects every result to match the hand computed values.
    for use_gpu in True, False:
      self._testErosionValidPadding(use_gpu)
      self._testErosionSamePadding(use_gpu)
      self._testErosionSamePaddingDepth(use_gpu)
      self._testErosionSamePaddingBatch(use_gpu)
      self._testErosionValidPaddingNonSquareWindow(use_gpu)
      self._testErosionSamePaddingRate(use_gpu)
      self._testErosionValidPaddingUnevenStride(use_gpu)

  def _ConstructAndTestGradient(self, image_shape, kernel_shape, strides, rates,
                                padding, use_gpu):
    # [FUNCTION DOC]
    # HELPER: shared gradient checker for the erosion gradient scenarios.
    # Inputs: image_shape (batch, height, width, channels), kernel_shape (height, width,
    # channels), strides and rates (two element lists), padding string, use_gpu flag.
    # Asserts channel counts match, seeds numpy randomness with a fixed value, builds random
    # float32 constants, computes the gradient error of the erosion op with respect to both
    # the image and the kernel with the v2 gradient checker on the chosen device, prints the
    # larger error and asserts it is below 1e-4. Returns nothing.
    """Verifies the gradients of the erosion function.

    Args:
      image_shape: Input shape, [batch, in_height, in_width, channels].
      kernel_shape: Filter shape, [filter_height, filter_width, channels].
      strides: Output strides, specified as [stride_height, stride_width].
      rates: Atrous rates, specified as [rate_height, rate_width].
      padding: Padding type.
      use_gpu: Whether we are running on GPU.
    """
    assert image_shape[3] == kernel_shape[2]

    np.random.seed(1)  # Make it reproducible.
    image = np.random.random_sample(image_shape).astype(np.float32)
    kernel = np.random.random_sample(kernel_shape).astype(np.float32)

    strides = [1] + strides + [1]
    rates = [1] + rates + [1]

    image_tensor = constant_op.constant(image, shape=image_shape, name="input")
    kernel_tensor = constant_op.constant(
        kernel, shape=kernel_shape, name="filter")

    def compute_erosion2d(image_tensor, kernel_tensor):
      # [FUNCTION DOC]
      # HELPER: closure used as the function under differentiation. Applies the erosion op to
      # the given image and kernel tensors using the enclosing strides, rates and padding,
      # and returns the output tensor.
      return nn_ops.erosion2d(
          image_tensor,
          kernel_tensor,
          strides=strides,
          rates=rates,
          padding=padding,
          name="erosion2d")

    with test_util.device(use_gpu=use_gpu):
      with self.cached_session():
        # Small delta is necessary for argmax to remain the same.
        err1 = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: compute_erosion2d(x, kernel_tensor), [image_tensor]))
        err2 = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: compute_erosion2d(image_tensor, x), [kernel_tensor]))
        err = max(err1, err2)

    print("Erosion gradient error = %f" % err)
    self.assertLess(err, 1e-4)

  def _testErosionGradValidPadding_1x1x1(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testErosionGrad. 3x3x1 image, 1x1x1 kernel, valid padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[1, 1, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        use_gpu=use_gpu)

  def _testErosionGradSamePadding_1x1x1(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testErosionGrad. 3x3x1 image, 1x1x1 kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[1, 1, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)

  def _testErosionGradSamePadding_1x1x2(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testErosionGrad. 3x3x2 image, 1x1x2 kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 2],
        kernel_shape=[1, 1, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)

  def _testErosionGradValidPadding_2x2x1(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testErosionGrad. 3x3x1 image, 2x2x1 kernel, valid padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        use_gpu=use_gpu)

  def _testErosionGradSamePadding_2x2x1(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testErosionGrad. 3x3x1 image, 2x2x1 kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)

  def _testErosionGradSamePaddingBatch_2x2x1(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testErosionGrad. Batch of four 3x3x1 images, 2x2x1 kernel,
    # same padding.
    self._ConstructAndTestGradient(
        image_shape=[4, 3, 3, 1],
        kernel_shape=[2, 2, 1],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)

  def _testErosionGradSamePadding_2x2x4(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: gradient scenario for testErosionGrad. 3x3x4 image, 2x2x4 kernel, same padding.
    self._ConstructAndTestGradient(
        image_shape=[1, 3, 3, 4],
        kernel_shape=[2, 2, 4],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        use_gpu=use_gpu)

  def testErosionGrad(self):
    # [FUNCTION DOC]
    # For both GPU and CPU placement, runs the seven erosion gradient scenarios, expecting the
    # numerical gradient error of each to stay below the shared tolerance.
    for use_gpu in True, False:
      self._testErosionGradValidPadding_1x1x1(use_gpu)
      self._testErosionGradSamePadding_1x1x1(use_gpu)
      self._testErosionGradSamePadding_1x1x2(use_gpu)
      self._testErosionGradValidPadding_2x2x1(use_gpu)
      self._testErosionGradSamePadding_2x2x1(use_gpu)
      self._testErosionGradSamePaddingBatch_2x2x1(use_gpu)
      self._testErosionGradSamePadding_2x2x4(use_gpu)


if __name__ == "__main__":
  test.main()
