# [MODULE DOC]
# HELPER: shared test-case base classes for the image gradient tests; not run directly as a test.
# Reusable test base classes covering the forward shapes and gradients of the image ops: nearest
# neighbor, bilinear and bicubic resizing, scale-and-translate, crop-and-resize and RGB to HSV
# conversion. Gradients are checked numerically against analytical ones with the version 2 gradient
# checker, and several tests compare CPU against GPU results. The classes are meant to be
# subclassed by concrete test modules (for example CPU or GPU variants); this module also has a
# main entry point that runs the tests through the standard test runner. Importing the module has
# no side effects beyond the imports.
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
"""Tests for Python ops defined in image_grad.py."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.for_all_test_methods(test_util.disable_xla,
                                'align_corners=False not supported by XLA')
class ResizeNearestNeighborOpTestBase(test.TestCase):
  # [CLASS DOC]
  # Test base for the nearest neighbor resize op and its gradient. The class attribute TYPES lists
  # the numpy float types (half, single, double and bfloat16) every test loops over. All test
  # methods are decorated to be disabled under XLA because align_corners being false is not
  # supported there. Subclass to run the tests.

  TYPES = [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]

  def testShapeIsCorrectAfterOp(self):
    # [FUNCTION DOC]
    # Test: resizing a 1x2x2x1 image to a 4x6 spatial size with nearest neighbor, for each type in
    # TYPES, must give a static shape (checked on the graph tensor) and an evaluated shape both
    # equal to 1x4x6x1. No return value.
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    for nptype in self.TYPES:
      x = np.arange(0, 4).reshape(in_shape).astype(nptype)

      input_tensor = constant_op.constant(x, shape=in_shape)
      resize_out = image_ops.resize_nearest_neighbor(input_tensor,
                                                     out_shape[1:3])
      with self.cached_session():
        self.assertEqual(out_shape, list(resize_out.get_shape()))
        resize_out = self.evaluate(resize_out)
      self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    # [FUNCTION DOC]
    # Test: for each type in TYPES, numerically checks the gradient of nearest neighbor resizing a
    # 1x2x3x1 image up to 4x6 (finite-difference step one eighth). The maximum difference between
    # analytical and numerical Jacobians must be below 1e-3. No return value.
    in_shape = [1, 2, 3, 1]
    out_shape = (1, 4, 6, 1)

    for nptype in self.TYPES:
      x = np.arange(0, 6).reshape(in_shape).astype(nptype)

      def resize_nn(t, shape=out_shape):
        # [FUNCTION DOC]
        # HELPER: function under test for the gradient checker in the enclosing test.
        # Resizes the tensor t to the height and width given by the shape argument (default: the
        # enclosing target shape) with nearest neighbor; returns the resized tensor.
        return image_ops.resize_nearest_neighbor(t, shape[1:3])

      with self.cached_session():
        input_tensor = constant_op.constant(x, shape=in_shape)
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                resize_nn, [input_tensor], delta=1 / 8))
        self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    # [FUNCTION DOC]
    # Test: same as the larger-size gradient test but shrinking a 1x4x6x1 image to 2x3. For each
    # type in TYPES the maximum error between analytical and numerical gradients must be below
    # 1e-3. No return value.
    in_shape = [1, 4, 6, 1]
    out_shape = (1, 2, 3, 1)

    for nptype in self.TYPES:
      x = np.arange(0, 24).reshape(in_shape).astype(nptype)

      def resize_nn(t, shape=out_shape):
        # [FUNCTION DOC]
        # HELPER: function under test for the gradient checker in the enclosing test.
        # Resizes the tensor t to the height and width given by the shape argument with nearest
        # neighbor; returns the resized tensor.
        return image_ops.resize_nearest_neighbor(t, shape[1:3])

      with self.cached_session():
        input_tensor = constant_op.constant(x, shape=in_shape)
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                resize_nn, [input_tensor], delta=1 / 8))
        self.assertLess(err, 1e-3)

  def testCompareGpuVsCpu(self):
    # [FUNCTION DOC]
    # Test: for each type in TYPES and for align_corners true and false, computes the Jacobians
    # (analytical and numerical) of nearest neighbor upsampling of a 1x4x6x3 image to 8x16, once in
    # a session without GPU and once in a default session, and asserts both results are close
    # (relative and absolute tolerance 1e-5). No return value.
    in_shape = [1, 4, 6, 3]
    out_shape = (1, 8, 16, 3)

    for nptype in self.TYPES:
      x = np.arange(0, np.prod(in_shape)).reshape(in_shape).astype(nptype)
      for align_corners in [True, False]:

        def resize_nn(t, shape=out_shape, align_corners=align_corners):
          # [FUNCTION DOC]
          # HELPER: function under test for the gradient checker in the enclosing test.
          # Nearest neighbor resize of t to the height and width in shape, honoring the
          # align_corners flag; returns the resized tensor.
          return image_ops.resize_nearest_neighbor(
              t, shape[1:3], align_corners=align_corners)

        with self.cached_session(use_gpu=False):
          input_tensor = constant_op.constant(x, shape=in_shape)
          grad_cpu = gradient_checker_v2.compute_gradient(
              resize_nn, [input_tensor], delta=1 / 8)

        with self.cached_session():
          input_tensor = constant_op.constant(x, shape=in_shape)
          grad_gpu = gradient_checker_v2.compute_gradient(
              resize_nn, [input_tensor], delta=1 / 8)

        self.assertAllClose(grad_cpu, grad_gpu, rtol=1e-5, atol=1e-5)


class ResizeBilinearOpTestBase(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test base for the bilinear resize op and its gradient. Provides helpers that enumerate shape
  # and option combinations and compute Jacobians, plus parameterized tests for shapes, gradient
  # accuracy, dtypes, unsupported types and CPU versus GPU agreement. Subclass to run.

  def _itGen(self, smaller_shape, larger_shape):
    # [FUNCTION DOC]
    # HELPER: generates the option and shape combinations used by the bilinear tests.
    # Generator. Inputs: the smaller and larger image shapes. Builds three shape pairs (upsample,
    # downsample, pass-through) and yields, for each option set and each pair, a tuple of input
    # shape, output shape, align_corners flag and half_pixel_centers flag. Option sets are always
    # (align corners true, half pixel centers false); unless XLA is enabled it also includes
    # (false, true) and (false, false) since non-aligned corners are unsupported by XLA.
    up_sample = (smaller_shape, larger_shape)
    down_sample = (larger_shape, smaller_shape)
    pass_through = (larger_shape, larger_shape)
    shape_pairs = (up_sample, down_sample, pass_through)
    # Align corners is deprecated in TF2.0, but align_corners==False is not
    # supported by XLA.
    options = [(True, False)]
    if not test_util.is_xla_enabled():
      options += [(False, True), (False, False)]
    for align_corners, half_pixel_centers in options:
      for in_shape, out_shape in shape_pairs:
        yield in_shape, out_shape, align_corners, half_pixel_centers

  def _getJacobians(self,
                    in_shape,
                    out_shape,
                    align_corners=False,
                    half_pixel_centers=False,
                    dtype=np.float32,
                    use_gpu=False,
                    force_gpu=False):
    # [FUNCTION DOC]
    # HELPER: computes numerical and analytical Jacobians of bilinear resizing for the tests.
    # Inputs: in_shape and out_shape (4D shapes; only height and width of out_shape are used),
    # align_corners and half_pixel_centers flags (default false), dtype of the input (default
    # float32), use_gpu and force_gpu session placement flags (default false). Runs inside a cached
    # session, builds an input of increasing values (values do not influence gradients) and returns
    # the pair from the gradient checker: theoretical and numerical Jacobians.
    with self.cached_session(use_gpu=use_gpu, force_gpu=force_gpu):
      # Input values should not influence gradients
      x = np.arange(np.prod(in_shape)).reshape(in_shape).astype(dtype)
      input_tensor = constant_op.constant(x, shape=in_shape)

      def func(in_tensor):
        # [FUNCTION DOC]
        # HELPER: function handed to the gradient checker. Bilinear resize of in_tensor to the
        # enclosing output height and width using the enclosing option flags; returns the result.
        return image_ops.resize_bilinear(
            in_tensor,
            out_shape[1:3],
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers)

      return gradient_checker_v2.compute_gradient(func, [input_tensor])

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def _testShapesParameterized(self, use_tape):
    # [FUNCTION DOC]
    # HELPER: shared shape check, not collected as a test because of its leading underscore.
    # KNOWN ISSUE: the name starts with an underscore, so the test runner never executes it unless
    # a subclass calls it explicitly; the parameter set built from True and the eager flag is
    # also odd (a set of two booleans).
    # Input use_tape: whether to record with a real gradient tape or an abstract stand-in. For
    # batch/channel combinations and every shape combination, resizes with bilinear, asserts the
    # static and evaluated output shapes equal the target, takes the gradient with respect to the
    # input and asserts its static and evaluated shapes equal the input shape. No return value.

    TEST_CASES = [[1, 1], [2, 3], [5, 4]]  # pylint: disable=invalid-name

    for batch_size, channel_count in TEST_CASES:
      smaller_shape = [batch_size, 2, 3, channel_count]
      larger_shape = [batch_size, 4, 6, channel_count]
      for in_shape, out_shape, _, _ in self._itGen(smaller_shape, larger_shape):
        with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
          # Input values should not influence shapes
          x = np.arange(np.prod(in_shape)).reshape(in_shape).astype(np.float32)
          input_tensor = constant_op.constant(x, shape=in_shape)
          tape.watch(input_tensor)
          resized_tensor = image_ops.resize_bilinear(input_tensor,
                                                     out_shape[1:3])
          self.assertEqual(out_shape, list(resized_tensor.get_shape()))

        grad_tensor = tape.gradient(resized_tensor, input_tensor)
        self.assertEqual(in_shape, list(grad_tensor.get_shape()))
        with self.cached_session():
          resized_values = self.evaluate(resized_tensor)
          self.assertEqual(out_shape, list(resized_values.shape))
          grad_values = self.evaluate(grad_tensor)
          self.assertEqual(in_shape, list(grad_values.shape))

  @parameterized.parameters({
      'batch_size': 1,
      'channel_count': 1
  }, {
      'batch_size': 4,
      'channel_count': 3
  }, {
      'batch_size': 3,
      'channel_count': 2
  })
  def testGradients(self, batch_size, channel_count):
    # [FUNCTION DOC]
    # Parameterized test over (batch size, channel count) of (1,1), (4,3) and (3,2). For every
    # shape and option combination from the generator, computes the analytical and numerical
    # Jacobians of bilinear resize and asserts they are close within 5e-3. No return value.
    smaller_shape = [batch_size, 2, 3, channel_count]
    larger_shape = [batch_size, 5, 6, channel_count]
    for in_shape, out_shape, align_corners, half_pixel_centers in \
        self._itGen(smaller_shape, larger_shape):
      jacob_a, jacob_n = self._getJacobians(in_shape, out_shape, align_corners,
                                            half_pixel_centers)
      threshold = 5e-3
      self.assertAllClose(jacob_a, jacob_n, threshold, threshold)

  def testTypes(self):
    # [FUNCTION DOC]
    # Test: for CPU and GPU and for float16, float32, float64 and bfloat16, computes bilinear
    # downsampling Jacobians of a 1x4x6x1 input to 2x3. For half precision types the numerical
    # Jacobian is replaced by one computed in float32 (a weaker check, noted in the code). Asserts
    # analytical and numerical Jacobians agree within 1e-3, or 1e-5 for float64. No return value.
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]
    for use_gpu in [False, True]:
      for dtype in [
          np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
      ]:
        jacob_a, jacob_n = self._getJacobians(
            in_shape, out_shape, dtype=dtype, use_gpu=use_gpu)
        if dtype in (np.float16, dtypes.bfloat16.as_numpy_dtype):
          # Compare fp16/bf16 analytical gradients to fp32 numerical gradients,
          # since fp16/bf16 numerical gradients are too imprecise unless great
          # care is taken with choosing the inputs and the delta. This is
          # a weaker, but pragmatic, check (in particular, it does not test
          # the op itself, only its gradient).
          _, jacob_n = self._getJacobians(
              in_shape, out_shape, dtype=np.float32, use_gpu=use_gpu)
        threshold = 1e-3
        if dtype == np.float64:
          threshold = 1e-5
        self.assertAllClose(jacob_a, jacob_n, threshold, threshold)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradOnUnsupportedType(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized test (with and without a real tape). Resizes an unsigned 8-bit image with
    # bilinear and asks for the gradient with respect to the input; asserts the result is a list
    # holding a single null value, i.e. no gradient is defined for that type. No return value.
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      x = np.arange(0, 24).reshape(in_shape).astype(np.uint8)
      input_tensor = constant_op.constant(x, shape=in_shape)
      tape.watch(input_tensor)
      resize_out = image_ops.resize_bilinear(input_tensor, out_shape[1:3])
      with self.cached_session():
        grad = tape.gradient(resize_out, [input_tensor])
    self.assertEqual([None], grad)

  def _gpuVsCpuCase(self, in_shape, out_shape, align_corners,
                    half_pixel_centers, dtype):
    # [FUNCTION DOC]
    # HELPER: compares bilinear resize Jacobians on CPU and GPU.
    # Inputs: input and output shapes, align_corners and half_pixel_centers flags, and dtype.
    # Computes the Jacobian pair with and without GPU and asserts the two results are close
    # (tolerance 1e-4, covering both analytical and numerical parts). No return value.
    grad = {}
    for use_gpu in [False, True]:
      grad[use_gpu] = self._getJacobians(
          in_shape,
          out_shape,
          align_corners,
          half_pixel_centers,
          dtype=dtype,
          use_gpu=use_gpu)
    threshold = 1e-4
    # Note that this is comparing both analytical and numerical Jacobians
    self.assertAllClose(grad[False], grad[True], rtol=threshold, atol=threshold)

  @parameterized.parameters({
      'batch_size': 1,
      'channel_count': 1
  }, {
      'batch_size': 2,
      'channel_count': 3
  }, {
      'batch_size': 5,
      'channel_count': 4
  })
  def testCompareGpuVsCpu(self, batch_size, channel_count):
    # [FUNCTION DOC]
    # Parameterized test over (batch size, channel count) of (1,1), (2,3) and (5,4). For each shape
    # and option combination from the generator, checks float32 bilinear Jacobians agree between
    # CPU and GPU. No return value.
    smaller_shape = [batch_size, 4, 6, channel_count]
    larger_shape = [batch_size, 8, 16, channel_count]
    for params in self._itGen(smaller_shape, larger_shape):
      self._gpuVsCpuCase(*params, dtype=np.float32)

  def testCompareGpuVsCpuFloat64(self):
    # [FUNCTION DOC]
    # Test: compares CPU and GPU Jacobians of float64 bilinear upsampling from 5x7 to 9x11 with
    # align_corners true and half pixel centers false. Only float64 is used because no 16-bit
    # float kernel is registered on GPU. No return value.
    in_shape = [1, 5, 7, 1]
    out_shape = [1, 9, 11, 1]
    # Note that there is no 16-bit floating-point format registered for GPU
    self._gpuVsCpuCase(
        in_shape,
        out_shape,
        align_corners=True,
        half_pixel_centers=False,
        dtype=np.float64)


class ResizeBicubicOpTestBase(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test base for the bicubic resize op and its gradient (shape check, gradients when growing and
  # shrinking, and behavior for an unsupported input type). Subclass to run.
  """Tests resize bicubic ops."""

  def testShapeIsCorrectAfterOp(self):
    # [FUNCTION DOC]
    # Test: bicubic resize of a 1x2x2x1 float32 image to 4x6, with align_corners true and false,
    # must give static and evaluated shapes of 1x4x6x1. No return value.
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 4).reshape(in_shape).astype(np.float32)

    for align_corners in [True, False]:
      input_tensor = constant_op.constant(x, shape=in_shape)
      resize_out = image_ops.resize_bicubic(
          input_tensor, out_shape[1:3], align_corners=align_corners)
      with self.cached_session():
        self.assertEqual(out_shape, list(resize_out.get_shape()))
        resize_out = self.evaluate(resize_out)
        self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    # [FUNCTION DOC]
    # Test: gradient check of bicubic upsampling a 1x2x3x1 float32 image to 4x6, for align_corners
    # true and false; the maximum error between analytical and numerical Jacobians must be below
    # 1e-3. No return value.
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)
    input_tensor = constant_op.constant(x, shape=in_shape)

    for align_corners in [True, False]:

      def func(input_tensor, align_corners=align_corners):
        # [FUNCTION DOC]
        # HELPER: function handed to the gradient checker. Bicubic resize of input_tensor to the
        # enclosing target height and width with the given align_corners flag; returns the result.
        return image_ops.resize_bicubic(
            input_tensor, out_shape[1:3], align_corners=align_corners)

      with self.cached_session():
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(func, [input_tensor]))

      self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    # [FUNCTION DOC]
    # Test: gradient check of bicubic downsampling a 1x4x6x1 float32 image to 2x3, for
    # align_corners true and false; the maximum Jacobian error must be below 1e-3. No return value.
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    x = np.arange(0, 24).reshape(in_shape).astype(np.float32)
    input_tensor = constant_op.constant(x, shape=in_shape)

    for align_corners in [True, False]:

      def func(input_tensor, align_corners=align_corners):
        # [FUNCTION DOC]
        # HELPER: function handed to the gradient checker. Bicubic resize of input_tensor to the
        # enclosing target height and width with the given align_corners flag; returns the result.
        return image_ops.resize_bicubic(
            input_tensor, out_shape[1:3], align_corners=align_corners)

      with self.cached_session():
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(func, [input_tensor]))

      self.assertLess(err, 1e-3)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradOnUnsupportedType(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized test (with and without a real tape). Bicubic resizes an unsigned 8-bit image
    # and requests the gradient with respect to the input; asserts the result is a list holding a
    # single null value (no gradient for that type). No return value.
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      in_shape = [1, 4, 6, 1]
      out_shape = [1, 2, 3, 1]

      x = np.arange(0, 24).reshape(in_shape).astype(np.uint8)
      input_tensor = constant_op.constant(x, shape=in_shape)
      tape.watch(input_tensor)

      resize_out = image_ops.resize_bicubic(input_tensor, out_shape[1:3])
      with self.cached_session():
        grad = tape.gradient(resize_out, [input_tensor])
    self.assertEqual([None], grad)


class ScaleAndTranslateOpTestBase(test.TestCase):
  # [CLASS DOC]
  # Test base for the scale-and-translate image op: gradient accuracy across kernels, scales and
  # translations, and the identity-scale gradient. Subclass to run.
  """Tests scale and translate op."""

  def testGrads(self):
    # [FUNCTION DOC]
    # Test: for each of three scales, four translations, eight resampling kernels and antialias on
    # and off, numerically checks the gradient of scale-and-translate on a 1x2x3x1 float32 image
    # with output size 4x6. The maximum Jacobian error must be below 1e-3. No return value.
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)

    kernel_types = [
        'lanczos1', 'lanczos3', 'lanczos5', 'gaussian', 'box', 'triangle',
        'keyscubic', 'mitchellcubic'
    ]
    scales = [(1.0, 1.0), (0.37, 0.47), (2.1, 2.1)]
    translations = [(0.0, 0.0), (3.14, 1.19), (2.1, 3.1), (100.0, 200.0)]
    for scale in scales:
      for translation in translations:
        for kernel_type in kernel_types:
          for antialias in [True, False]:
            with self.cached_session():
              input_tensor = constant_op.constant(x, shape=in_shape)

              def scale_trans(input_tensor,
                              scale=scale,
                              translation=translation,
                              kernel_type=kernel_type,
                              antialias=antialias):
                # [FUNCTION DOC]
                # HELPER: function handed to the gradient checker. Applies scale-and-translate to
                # input_tensor with the given scale, translation, kernel type and antialias flag
                # (defaults bound from the loop variables) to the enclosing 4x6 output size and
                # returns the resampled image.
                # pylint: disable=cell-var-from-loop
                return image_ops.scale_and_translate(
                    input_tensor,
                    out_shape[1:3],
                    scale=constant_op.constant(scale),
                    translation=constant_op.constant(translation),
                    kernel_type=kernel_type,
                    antialias=antialias)

              err = gradient_checker_v2.max_error(
                  *gradient_checker_v2.compute_gradient(scale_trans,
                                                        [input_tensor]))

            self.assertLess(err, 1e-3)

  def testIdentityGrads(self):
    # [FUNCTION DOC]
    # Test: with unit scale, zero translation and antialias on, for kernels lanczos1, lanczos3,
    # lanczos5, triangle and keyscubic, records scale-and-translate on a 1x2x3x1 image with a
    # gradient tape and asserts the gradient of the first batch entry is all ones. No return value.
    """Tests that Gradients for 1.0 scale should be ones for some kernels."""
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)

    kernel_types = ['lanczos1', 'lanczos3', 'lanczos5', 'triangle', 'keyscubic']
    scale = (1.0, 1.0)
    translation = (0.0, 0.0)
    antialias = True
    for kernel_type in kernel_types:
      with self.cached_session():
        input_tensor = constant_op.constant(x, shape=in_shape)
        with backprop.GradientTape() as tape:
          tape.watch(input_tensor)
          scale_and_translate_out = image_ops.scale_and_translate(
              input_tensor,
              out_shape[1:3],
              scale=constant_op.constant(scale),
              translation=constant_op.constant(translation),
              kernel_type=kernel_type,
              antialias=antialias)
        grad = tape.gradient(scale_and_translate_out, input_tensor)[0]
        grad_v = self.evaluate(grad)
        self.assertAllClose(np.ones_like(grad_v), grad_v)


class CropAndResizeOpTestBase(test.TestCase):
  # [CLASS DOC]
  # Test base for the crop-and-resize op: output shape check and gradient checks against both the
  # image and the box coordinates using randomly generated boxes. Subclass to run.

  def testShapeIsCorrectAfterOp(self):
    # [FUNCTION DOC]
    # Test: crops two boxes from a batch of two 3x4x2 images into 4x5 crops and asserts the static
    # and evaluated shape equals number of boxes by crop height by crop width by depth. No return
    # value.
    batch = 2
    image_height = 3
    image_width = 4
    crop_height = 4
    crop_width = 5
    depth = 2
    num_boxes = 2

    image_shape = [batch, image_height, image_width, depth]
    crop_size = [crop_height, crop_width]
    crops_shape = [num_boxes, crop_height, crop_width, depth]

    image = np.arange(0, batch * image_height * image_width *
                      depth).reshape(image_shape).astype(np.float32)
    boxes = np.array([[0, 0, 1, 1], [.1, .2, .7, .8]], dtype=np.float32)
    box_ind = np.array([0, 1], dtype=np.int32)

    crops = image_ops.crop_and_resize(
        constant_op.constant(image, shape=image_shape),
        constant_op.constant(boxes, shape=[num_boxes, 4]),
        constant_op.constant(box_ind, shape=[num_boxes]),
        constant_op.constant(crop_size, shape=[2]))
    with self.session():
      self.assertEqual(crops_shape, list(crops.get_shape()))
      crops = self.evaluate(crops)
      self.assertEqual(crops_shape, list(crops.shape))

  def _randomUniformAvoidAnchors(self, low, high, anchors, radius, num_samples):
    # [FUNCTION DOC]
    # HELPER: draws random box coordinates away from non-differentiable points.
    # Inputs: low and high interval ends, anchors (list of anchor values), radius (minimum
    # distance), num_samples (count). Asserts low is below high, radius is non-negative and that
    # the forbidden zones cover less than a quarter of the interval (assertion failure
    # otherwise). Uses the global numpy random generator, rejecting samples within radius of any
    # anchor, and returns a list of num_samples accepted values.
    """Generate samples that are far enough from a set of anchor points.

    We generate uniform samples in [low, high], then reject those that are less
    than radius away from any point in anchors. We stop after we have accepted
    num_samples samples.

    Args:
      low: The lower end of the interval.
      high: The upper end of the interval.
      anchors: A list of length num_crops with anchor points to avoid.
      radius: Distance threshold for the samples from the anchors.
      num_samples: How many samples to produce.

    Returns:
      samples: A list of length num_samples with the accepted samples.
    """
    self.assertTrue(low < high)
    self.assertTrue(radius >= 0)
    num_anchors = len(anchors)
    # Make sure that at least half of the interval is not forbidden.
    self.assertTrue(2 * radius * num_anchors < 0.5 * (high - low))
    anchors = np.reshape(anchors, num_anchors)
    samples = []
    while len(samples) < num_samples:
      sample = np.random.uniform(low, high)
      if np.all(np.fabs(sample - anchors) > radius):
        samples.append(sample)
    return samples

  def testGradRandomBoxes(self):
    # [FUNCTION DOC]
    # Test: seeds numpy randomness (side effect on global state), then over many image widths,
    # crop sizes, depths and box counts builds images and random boxes kept away from pixel grid
    # lines. On a GPU with op determinism enabled it asserts the gradient computation raises an
    # unimplemented error for both image and boxes; otherwise it checks the gradient with respect
    # to image and boxes numerically and requires the larger error to be below 2e-3.
    """Test that the gradient is correct for randomly generated boxes.

    The mapping is piecewise differentiable with respect to the box coordinates.
    The points where the function is not differentiable are those which are
    mapped to image pixels, i.e., the normalized y coordinates in
    np.linspace(0, 1, image_height) and normalized x coordinates in
    np.linspace(0, 1, image_width). Make sure that the box coordinates are
    sufficiently far away from those rectangular grid centers that are points of
    discontinuity, so that the finite difference Jacobian is close to the
    computed one.
    """
    np.random.seed(1)  # Make it reproducible.
    delta = 1e-3
    radius = 2 * delta
    low, high = -0.5, 1.5  # Also covers the case of extrapolation.

    image_height = 4
    for image_width in range(1, 3):
      for crop_height in range(1, 3):
        for crop_width in range(2, 4):
          for depth in range(1, 3):
            for num_boxes in range(1, 3):

              batch = num_boxes
              image_shape = [batch, image_height, image_width, depth]
              crop_size = [crop_height, crop_width]

              image = np.arange(0, batch * image_height * image_width *
                                depth).reshape(image_shape).astype(np.float32)
              boxes = []
              for _ in range(num_boxes):
                # pylint: disable=unbalanced-tuple-unpacking
                y1, y2 = self._randomUniformAvoidAnchors(
                    low, high, np.linspace(0, 1, image_height), radius, 2)
                x1, x2 = self._randomUniformAvoidAnchors(
                    low, high, np.linspace(0, 1, image_width), radius, 2)
                # pylint: enable=unbalanced-tuple-unpacking
                boxes.append([y1, x1, y2, x2])

              boxes = np.array(boxes, dtype=np.float32)
              box_ind = np.arange(batch, dtype=np.int32)

              image_tensor = constant_op.constant(image, shape=image_shape)
              boxes_tensor = constant_op.constant(boxes, shape=[num_boxes, 4])
              box_ind_tensor = constant_op.constant(box_ind, shape=[num_boxes])

              def crop_resize(image_tensor, boxes_tensor):
                # [FUNCTION DOC]
                # HELPER: wraps crop-and-resize for the gradient checker. Takes an image tensor
                # and a boxes tensor, uses the enclosing box indices and crop size, and returns
                # the cropped and resized tensor.
                # pylint: disable=cell-var-from-loop
                return image_ops.crop_and_resize(
                    image_tensor, boxes_tensor, box_ind_tensor,
                    constant_op.constant(crop_size, shape=[2]))

              with test_util.device(use_gpu=True):
                with self.cached_session():
                  # pylint: disable=cell-var-from-loop
                  if (config.is_op_determinism_enabled() and
                      test_util.is_gpu_available()):
                    with self.assertRaises(errors_impl.UnimplementedError):
                      gradient_checker_v2.compute_gradient(
                          lambda x: crop_resize(x, boxes_tensor),
                          [image_tensor])
                    with self.assertRaises(errors_impl.UnimplementedError):
                      gradient_checker_v2.compute_gradient(
                          lambda x: crop_resize(image_tensor, x),
                          [boxes_tensor])
                  else:
                    err1 = gradient_checker_v2.max_error(
                        *gradient_checker_v2.compute_gradient(
                            lambda x: crop_resize(x, boxes_tensor),
                            [image_tensor]))
                    err2 = gradient_checker_v2.max_error(
                        *gradient_checker_v2.compute_gradient(
                            lambda x: crop_resize(image_tensor, x),
                            [boxes_tensor]))
                    err = max(err1, err2)
                    self.assertLess(err, 2e-3)


@test_util.run_all_in_graph_and_eager_modes
class RGBToHSVOpTestBase(test.TestCase):
  # [CLASS DOC]
  # Test base for the RGB to HSV conversion op and its gradient; every test runs in both graph and
  # eager modes. TYPES lists the float types (single and double) looped over by the shape test.
  # Subclass to run.

  TYPES = [np.float32, np.float64]

  def testShapeIsCorrectAfterOp(self):
    # [FUNCTION DOC]
    # Test: converting random 2x20x30x3 pixel data of each type in TYPES to HSV preserves the shape
    # both statically and after evaluation. No return value.
    in_shape = [2, 20, 30, 3]
    out_shape = [2, 20, 30, 3]

    for nptype in self.TYPES:
      x = np.random.randint(0, high=255, size=[2, 20, 30, 3]).astype(nptype)
      rgb_input_tensor = constant_op.constant(x, shape=in_shape)
      hsv_out = gen_image_ops.rgb_to_hsv(rgb_input_tensor)
      with self.cached_session():
        self.assertEqual(out_shape, list(hsv_out.get_shape()))
      hsv_out = self.evaluate(hsv_out)
      self.assertEqual(out_shape, list(hsv_out.shape))

  def testRGBToHSVGradSimpleCase(self):
    # [FUNCTION DOC]
    # Test: for each type in TYPES, uses a smooth 3x3 matrix of RGB values (no discontinuities) and
    # asserts numerical and analytical gradients of the conversion match within 1e-4. No return
    # value.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test for the gradient checker; converts RGB tensor x to HSV.
      return gen_image_ops.rgb_to_hsv(x)

    for nptype in self.TYPES:
      # Building a simple input tensor to avoid any discontinuity
      x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8,
                                                       0.9]]).astype(nptype)
      rgb_input_tensor = constant_op.constant(x, shape=x.shape)
      # Computing Analytical and Numerical gradients of f(x)
      analytical, numerical = gradient_checker_v2.compute_gradient(
          f, [rgb_input_tensor])
      self.assertAllClose(numerical, analytical, atol=1e-4)

  def testRGBToHSVGradRandomCase(self):
    # [FUNCTION DOC]
    # Test: seeds numpy randomness (global side effect), builds a random 1x5x5x3 float32 image and
    # asserts the maximum error between numerical and analytical gradients of the conversion is
    # below 1e-4. No return value.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test for the gradient checker; converts RGB tensor x to HSV.
      return gen_image_ops.rgb_to_hsv(x)

    np.random.seed(0)
    # Building a simple input tensor to avoid any discontinuity
    x = np.random.rand(1, 5, 5, 3).astype(np.float32)
    rgb_input_tensor = constant_op.constant(x, shape=x.shape)
    # Computing Analytical and Numerical gradients of f(x)
    self.assertLess(
        gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(f, [rgb_input_tensor])), 1e-4)

  def testRGBToHSVGradSpecialCaseRGreatest(self):
    # [FUNCTION DOC]
    # Test: on an image where red exceeds green exceeds blue everywhere, compares gradients of the
    # real conversion op with those of a hand-written reference built from primitive ops. Asserts
    # numerical versus analytical agreement for each function and that the analytical (and the
    # numerical) results of the two functions match, all within 1e-4. No return value.
    # This test tests a specific subset of the input space
    # with a dummy function implemented with native TF operations.
    in_shape = [2, 10, 20, 3]

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test for the gradient checker; converts RGB tensor x to HSV.
      return gen_image_ops.rgb_to_hsv(x)

    def f_dummy(x):
      # [FUNCTION DOC]
      # HELPER: reference implementation of RGB to HSV from basic ops, valid only when red is the
      # largest and blue the smallest channel. Takes an RGB tensor x and returns a tensor with hue
      # (scaled to the zero to one range), saturation and value stacked on the last axis.
      # This dummy function is a implementation of RGB to HSV using
      # primitive TF functions for one particular case when R>G>B.
      r = x[..., 0]
      g = x[..., 1]
      b = x[..., 2]
      # Since MAX = r and MIN = b, we get the following h,s,v values.
      v = r
      s = 1 - math_ops.div_no_nan(b, r)
      h = 60 * math_ops.div_no_nan(g - b, r - b)
      h = h / 360
      return array_ops_stack.stack([h, s, v], axis=-1)

    # Building a custom input tensor where R>G>B
    x_reds = np.ones((in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x_greens = 0.5 * np.ones(
        (in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x_blues = 0.2 * np.ones(
        (in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x = np.stack([x_reds, x_greens, x_blues], axis=-1)
    rgb_input_tensor = constant_op.constant(x, shape=in_shape)

    # Computing Analytical and Numerical gradients of f(x)
    analytical, numerical = gradient_checker_v2.compute_gradient(
        f, [rgb_input_tensor])
    # Computing Analytical and Numerical gradients of f_dummy(x)
    analytical_dummy, numerical_dummy = gradient_checker_v2.compute_gradient(
        f_dummy, [rgb_input_tensor])
    self.assertAllClose(numerical, analytical, atol=1e-4)
    self.assertAllClose(analytical_dummy, analytical, atol=1e-4)
    self.assertAllClose(numerical_dummy, numerical, atol=1e-4)


if __name__ == '__main__':
  test.main()
