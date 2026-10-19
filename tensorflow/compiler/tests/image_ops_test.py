# [MODULE DOC]
# Tests for the image operations when run through the XLA test harness (XLATestCase) on the
# registered XLA test device. Covers RGB/HSV conversion, contrast, hue and saturation
# adjustment, nearest-neighbor and bilinear resizing (forward and gradient, aligned and
# half-pixel-center variants) and non-max suppression (single and batched).
# Results are compared against hand-computed tables or NumPy/colorsys reference implementations.
# No import-time side effects; the bottom of the file runs the test main when executed directly.
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
"""Tests for image ops."""

import colorsys
import math
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


def _generate_numpy_random_rgb(shape):
  # [FUNCTION DOC]
  # HELPER: shared input generator for the RGB/HSV tests in this file.
  # Returns a NumPy array of the given shape whose entries are random integers from 0 to 255
  # divided by 256, so every value is exactly representable even in low-precision float types.
  # Uses the global NumPy random state (the caller seeds it); no other side effects.
  # Only generate floating points that are fractions like n / 256, since they
  # are RGB pixels. Some low-precision floating point types in this test can't
  # handle arbitrary precision floating points well.
  return np.random.randint(0, 256, shape) / 256.


class RGBToHSVTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests RGB to HSV and HSV to RGB conversion on the XLA test device for all float types.

  def testBatch(self):
    # [FUNCTION DOC]
    # Seeds NumPy, builds a random batch of RGB images, and for every float type converts it to
    # HSV and back, both as one batch and image by image (unstack, convert, stack).
    # Asserts the batched and per-image results agree and that the round trip recovers the input
    # within type-dependent tolerances.
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in self.float_types:
      inp = _generate_numpy_random_rgb(shape).astype(nptype)

      # Convert to HSV and back, as a batch and individually
      with self.session() as sess:
        batch0 = array_ops.placeholder(nptype, shape=shape)
        with self.test_scope():
          batch1 = image_ops.rgb_to_hsv(batch0)
          batch2 = image_ops.hsv_to_rgb(batch1)
        split0 = array_ops_stack.unstack(batch0)
        with self.test_scope():
          split1 = list(map(image_ops.rgb_to_hsv, split0))
          split2 = list(map(image_ops.hsv_to_rgb, split1))
        join1 = array_ops_stack.stack(split1)
        join2 = array_ops_stack.stack(split2)
        batch1, batch2, join1, join2 = sess.run([batch1, batch2, join1, join2],
                                                {batch0: inp})

      # Verify that processing batch elements together is the same as separate
      self.assertAllCloseAccordingToType(batch1, join1, half_rtol=0.000002)
      self.assertAllCloseAccordingToType(batch2, join2, half_rtol=0.000002)
      self.assertAllCloseAccordingToType(
          batch2, inp, bfloat16_atol=0.03, half_rtol=0.02)

  def testRGBToHSVRoundTrip(self):
    # [FUNCTION DOC]
    # Round-trip test: for each float type, converts a small fixed RGB image (values scaled to
    # the zero to one range) to HSV and back on the test device and asserts the result matches
    # the original within type-dependent tolerance.
    data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    for nptype in self.float_types:
      rgb_np = np.array(data, dtype=nptype).reshape([2, 2, 3]) / 255.
      with self.session():
        placeholder = array_ops.placeholder(nptype)
        with self.test_scope():
          hsv = image_ops.rgb_to_hsv(placeholder)
          rgb = image_ops.hsv_to_rgb(hsv)
          rgb_tf = rgb.eval(feed_dict={placeholder: rgb_np})
      self.assertAllCloseAccordingToType(rgb_tf, rgb_np, bfloat16_atol=0.03)

  def testRGBToHSVNumpy(self):
    # [FUNCTION DOC]
    # For each float type, converts random RGB pixels to HSV on the test device and compares with
    # the reference Python colorsys conversion computed in double precision; asserts closeness
    # according to type.
    """Tests the RGB to HSV conversion matches a reference implementation."""
    for nptype in self.float_types:
      rgb_flat = _generate_numpy_random_rgb((64, 3)).astype(nptype)
      rgb_np = rgb_flat.reshape(4, 4, 4, 3)  # pylint: disable=too-many-function-args
      hsv_np = np.array([
          colorsys.rgb_to_hsv(
              r.astype(np.float64), g.astype(np.float64), b.astype(np.float64))
          for r, g, b in rgb_flat
      ])
      hsv_np = hsv_np.reshape(4, 4, 4, 3)  # pylint: disable=too-many-function-args
      with self.session():
        placeholder = array_ops.placeholder(nptype)
        with self.test_scope():
          hsv_op = image_ops.rgb_to_hsv(placeholder)
        hsv_tf = hsv_op.eval(feed_dict={placeholder: rgb_np})
      self.assertAllCloseAccordingToType(hsv_tf, hsv_np)


class AdjustContrastTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests the contrast adjustment op on the XLA test device against known values and a NumPy
  # reference (per-image, per-channel mean scaling).

  def _testContrast(self, x_np, y_np, contrast_factor):
    # [FUNCTION DOC]
    # HELPER: shared check for the fixed-value contrast tests.
    # Inputs: x_np, an input array; y_np, the expected output array (same dtype as input);
    # contrast_factor, the scalar factor. Converts the input to float32, adjusts contrast on the
    # test device, converts back with saturation, and asserts closeness to the expected values
    # (tolerance 1e-6). Returns nothing.
    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_np.shape)
      flt_x = image_ops.convert_image_dtype(x, dtypes.float32)
      with self.test_scope():
        y = image_ops.adjust_contrast(flt_x, contrast_factor)
      y = image_ops.convert_image_dtype(y, x.dtype, saturate=True)
      y_tf = y.eval({x: x_np})
      self.assertAllClose(y_tf, y_np, 1e-6)

  def testFloatContrast(self):
    # [FUNCTION DOC]
    # Contrast test with float data (scaled to zero to one) and factor 2.0 on a 1x2x2x3 image,
    # compared with precomputed expected values.
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float32).reshape(x_shape) / 255.

    y_data = [
        -45.25, -90.75, -92.5, 62.75, 169.25, 333.5, 28.75, -84.75, 349.5,
        134.75, 409.25, -116.5
    ]
    y_np = np.array(y_data, dtype=np.float32).reshape(x_shape) / 255.

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testBatchContrast(self):
    # [FUNCTION DOC]
    # Contrast test on a two-image unsigned 8-bit batch with factor 2.0; expected output is
    # saturated to the 0 to 255 range.
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 81, 200, 255, 10, 0, 255, 116, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def _adjustContrastNp(self, x_np, contrast_factor):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for contrast adjustment.
    # Computes the mean over the spatial axes (height and width) of each image and channel, and
    # returns mean plus contrast_factor times the deviation from that mean. Inputs are a 4-D
    # array and a scalar factor.
    mean = np.mean(x_np, (1, 2), keepdims=True)
    y_np = mean + contrast_factor * (x_np - mean)
    return y_np

  def _adjustContrastTf(self, x_np, contrast_factor):
    # [FUNCTION DOC]
    # HELPER: runs the contrast adjustment op on the test device for a float32 input array and
    # the given scalar factor, and returns the evaluated result as an array.
    with self.session():
      x = array_ops.placeholder(np.float32)
      with self.test_scope():
        y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = y.eval({x: x_np})
    return y_tf

  def testRandomContrast(self):
    # [FUNCTION DOC]
    # For several random 4-D shapes, draws random pixel data and a random factor between 0.1 and
    # 2.1, and asserts the device result matches the NumPy reference within 1e-5.
    x_shapes = [
        [1, 2, 2, 3],
        [2, 1, 2, 3],
        [1, 2, 2, 3],
        [2, 5, 5, 3],
        [2, 1, 1, 3],
    ]
    for x_shape in x_shapes:
      x_np = np.random.rand(*x_shape) * 255.
      contrast_factor = np.random.rand() * 2.0 + 0.1
      y_np = self._adjustContrastNp(x_np, contrast_factor)
      y_tf = self._adjustContrastTf(x_np, contrast_factor)
      self.assertAllClose(y_tf, y_np, rtol=1e-5, atol=1e-5)


class AdjustHueTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests the hue adjustment op on the XLA test device, both with fixed 8-bit values and against
  # a colorsys-based reference on random data.

  def testAdjustNegativeHue(self):
    # [FUNCTION DOC]
    # Adjusts hue by negative 0.25 on a 2x2x3 uint8 image (converted to float and back with
    # saturation) and asserts exact equality with a precomputed expected image.
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = -0.25
    y_data = [0, 13, 1, 54, 226, 59, 8, 234, 150, 255, 39, 1]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      flt_x = image_ops.convert_image_dtype(x, dtypes.float32)
      with self.test_scope():
        y = gen_image_ops.adjust_hue(flt_x, delta)
      y = image_ops.convert_image_dtype(y, x.dtype, saturate=True)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def testAdjustPositiveHue(self):
    # [FUNCTION DOC]
    # Same as the negative-delta test but with a positive delta of 0.25; asserts exact equality
    # with precomputed expected 8-bit values.
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      flt_x = image_ops.convert_image_dtype(x, dtypes.float32)
      with self.test_scope():
        y = gen_image_ops.adjust_hue(flt_x, delta)
      y = image_ops.convert_image_dtype(y, x.dtype, saturate=True)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def testBatchAdjustHue(self):
    # [FUNCTION DOC]
    # Hue adjustment by 0.25 on a batch of two 1x2x3 uint8 images; the expected values equal
    # those of the single-image positive-delta test.
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      flt_x = image_ops.convert_image_dtype(x, dtypes.float32)
      with self.test_scope():
        y = gen_image_ops.adjust_hue(flt_x, delta)
      y = image_ops.convert_image_dtype(y, x.dtype, saturate=True)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def _adjustHueNp(self, x_np, delta_h):
    # [FUNCTION DOC]
    # HELPER: reference hue adjustment using Python colorsys.
    # Asserts the last dimension is 3, flattens to pixels, converts each pixel to HSV, adds the
    # hue delta, wraps into the zero to one range, converts back to RGB and reshapes to the
    # original shape. Returns the adjusted array.
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in range(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      h += delta_h
      h = math.fmod(h + 10.0, 1.0)
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def _adjustHueTf(self, x_np, delta_h):
    # [FUNCTION DOC]
    # HELPER: runs the generated hue adjustment op on the test device for a float32 placeholder
    # fed with x_np and scalar delta_h; returns the evaluated result.
    with self.session():
      x = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        y = gen_image_ops.adjust_hue(x, delta_h)
      y_tf = y.eval({x: x_np})
    return y_tf

  def testAdjustRandomHue(self):
    # [FUNCTION DOC]
    # Randomized comparison against the reference for several shapes and five channel-equality
    # styles (all random, or red equals green, red equals blue, green equals blue, all equal)
    # with a random delta in minus one to one. Raises an assertion error on an unknown style.
    # Asserts closeness with small tolerances.
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    for x_shape in x_shapes:
      for test_style in test_styles:
        x_np = np.random.rand(*x_shape) * 255.
        delta_h = np.random.rand() * 2.0 - 1.0
        if test_style == "all_random":
          pass
        elif test_style == "rg_same":
          x_np[..., 1] = x_np[..., 0]
        elif test_style == "rb_same":
          x_np[..., 2] = x_np[..., 0]
        elif test_style == "gb_same":
          x_np[..., 2] = x_np[..., 1]
        elif test_style == "rgb_same":
          x_np[..., 1] = x_np[..., 0]
          x_np[..., 2] = x_np[..., 0]
        else:
          raise AssertionError("Invalid test style: %s" % (test_style))
        y_np = self._adjustHueNp(x_np, delta_h)
        y_tf = self._adjustHueTf(x_np, delta_h)
        self.assertAllClose(y_tf, y_np, rtol=2e-5, atol=1e-4)

  def testInvalidShapes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: effectively disabled; a local flag is set to false and the test returns
    # immediately, so none of the assertions below it ever run.
    # Intended to check that a rank-2 input raises a shape error and that an input without 3
    # channels raises an op error.
    fused = False
    if not fused:
      # The tests are known to pass with the fused adjust_hue. We will enable
      # them when the fused implementation is the default.
      return
    x_np = np.random.rand(2, 3) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    fused = False
    with self.assertRaisesRegex(ValueError, "Shape must be at least rank 3"):
      self._adjustHueTf(x_np, delta_h)
    x_np = np.random.rand(4, 2, 4) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesOpError("input must have 3 channels"):
      self._adjustHueTf(x_np, delta_h)


class AdjustSaturationTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests the saturation adjustment op on the XLA test device with fixed 8-bit values and a
  # colorsys-based reference on random data.

  def _adjust_saturation(self, image, saturation_factor):
    # [FUNCTION DOC]
    # HELPER: converts the image to a tensor, converts to float32, applies the saturation
    # adjustment on the test device, converts back to the original dtype and returns the tensor.
    # Inputs: an image (array or tensor) and a scalar saturation factor.
    image = ops.convert_to_tensor(image, name="image")
    orig_dtype = image.dtype
    flt_image = image_ops.convert_image_dtype(image, dtypes.float32)
    with self.test_scope():
      saturation_adjusted_image = gen_image_ops.adjust_saturation(
          flt_image, saturation_factor)
    return image_ops.convert_image_dtype(saturation_adjusted_image, orig_dtype)

  def testHalfSaturation(self):
    # [FUNCTION DOC]
    # Saturation factor 0.5 on a 2x2x3 uint8 image; asserts exact equality with expected values.
    x_shape = [2, 2, 3]
    x_rgb_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_rgb_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_rgb_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_rgb_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      y = self._adjust_saturation(x, saturation_factor)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def testTwiceSaturation(self):
    # [FUNCTION DOC]
    # Saturation factor 2.0 on a 2x2x3 uint8 image; asserts exact equality with expected values.
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 2.0
    y_data = [0, 5, 13, 0, 106, 226, 30, 0, 234, 89, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      y = self._adjust_saturation(x, saturation_factor)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def _adjustSaturationNp(self, x_np, scale):
    # [FUNCTION DOC]
    # HELPER: reference saturation adjustment using Python colorsys.
    # Asserts the last dimension is 3, converts every pixel to HSV, multiplies saturation by the
    # scale clamped to the zero to one range, converts back and reshapes. Returns the array.
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in range(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      s *= scale
      s = min(1.0, max(0.0, s))
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def testAdjustRandomSaturation(self):
    # [FUNCTION DOC]
    # Randomized comparison with the reference over several shapes and five channel-equality
    # styles, using a random scale in zero to one. Raises an assertion error for an unknown
    # style. Asserts closeness of the device result to the reference.
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    with self.session():
      for x_shape in x_shapes:
        for test_style in test_styles:
          x_np = np.random.rand(*x_shape) * 255.
          scale = np.random.rand()
          if test_style == "all_random":
            pass
          elif test_style == "rg_same":
            x_np[..., 1] = x_np[..., 0]
          elif test_style == "rb_same":
            x_np[..., 2] = x_np[..., 0]
          elif test_style == "gb_same":
            x_np[..., 2] = x_np[..., 1]
          elif test_style == "rgb_same":
            x_np[..., 1] = x_np[..., 0]
            x_np[..., 2] = x_np[..., 0]
          else:
            raise AssertionError("Invalid test style: %s" % (test_style))
          y_baseline = self._adjustSaturationNp(x_np, scale)
          x = array_ops.placeholder(dtypes.float32, shape=x_shape)
          with self.test_scope():
            y_fused = self._adjust_saturation(x,
                                              scale).eval(feed_dict={x: x_np})
          self.assertAllClose(y_fused, y_baseline, rtol=2e-5, atol=1e-5)


class ResizeNearestNeighborTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests nearest-neighbor resize on the XLA test device with align-corners enabled, using
  # small hand-written inputs and expected outputs (including bfloat16 and uint8).
  # TODO(ilch): Wrap each test with `for dtype in self.float_types:`
  # Some work to understand how that should be done was presented here:
  # cl/227850213

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None,
                                      large_tolerance=False,
                                      align_corners=True):
    # [FUNCTION DOC]
    # HELPER: shared check for the aligned nearest-neighbor tests.
    # Inputs: image_np, a 2-D array; target_shape, the output height and width; expected, the
    # expected 2-D output (required, fails the test if missing); large_tolerance, loosens
    # tolerance; align_corners, default true. Adds batch and channel axes, resizes on the test
    # device and asserts closeness.
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_nearest_neighbor(
          image, target_shape, align_corners=align_corners)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=2e-4, atol=2e-4)
      else:
        self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  def testAlignCorners2x2To1x1(self):
    # [FUNCTION DOC]
    # Aligned resize of 2x2 to 1x1.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2], [3, 4]], dtype=np.float32), [1, 1],
        expected=np.array([[1]], dtype=np.float32))

  def testAlignCorners1x1To2x2(self):
    # [FUNCTION DOC]
    # Aligned resize of 1x1 to 2x2.
    self._assertForwardOpMatchesExpected(
        np.array([[1]], dtype=np.float32), [2, 2],
        expected=np.array([[1, 1], [1, 1]], dtype=np.float32))

  def testAlignCorners1x1To3x3(self):
    # [FUNCTION DOC]
    # Aligned resize of 1x1 to 3x3.
    self._assertForwardOpMatchesExpected(
        np.array([[1]], dtype=np.float32), [3, 3],
        expected=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32))

  def testAlignCorners2x2To3x3(self):
    # [FUNCTION DOC]
    # Aligned resize of 2x2 to 3x3.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2], [3, 4]], dtype=np.float32), [3, 3],
        expected=np.array([[1, 2, 2], [3, 4, 4], [3, 4, 4]], dtype=np.float32))

  def testAlignCorners2x2To4x4(self):
    # [FUNCTION DOC]
    # Aligned resize of 2x2 to 4x4 with large tolerance.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2], [3, 4]], dtype=np.float32), [4, 4],
        expected=np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
            dtype=np.float32),
        large_tolerance=True)

  def testAlignCorners3x3To2x2(self):
    # [FUNCTION DOC]
    # Aligned downscale of 3x3 to 2x2.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [2, 2],
        expected=np.array([[1, 3], [7, 9]], dtype=np.float32))

  def testAlignCorners4x4To3x3(self):
    # [FUNCTION DOC]
    # Aligned downscale of 4x4 to 3x3.
    self._assertForwardOpMatchesExpected(
        np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32), [3, 3],
        expected=np.array([[1, 3, 4], [9, 11, 12], [13, 15, 16]],
                          dtype=np.float32))

  def testAlignCorners3x3To4x4(self):
    # [FUNCTION DOC]
    # Aligned upscale of 3x3 to 4x4.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [4, 4],
        expected=np.array(
            [[1, 2, 2, 3], [4, 5, 5, 6], [4, 5, 5, 6], [7, 8, 8, 9]],
            dtype=np.float32))

  def testAlignCorners3x3To6x6(self):
    # [FUNCTION DOC]
    # Aligned upscale of 3x3 to 6x6.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [6, 6],
        expected=np.array(
            [[1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6],
             [4, 4, 5, 5, 6, 6], [7, 7, 8, 8, 9, 9], [7, 7, 8, 8, 9, 9]],
            dtype=np.float32))

  def testAlignCorners3x3To9x9(self):
    # [FUNCTION DOC]
    # Aligned upscale of 3x3 to 9x9; the expected matrix is uneven because the implementation
    # dilates and convolves iteratively.
    # The expected matrix might look uneven in terms of how many of each number
    # there is, but this is an artifact of doing the dilation and convolution
    # iteratively. The behavior is less esoteric in the 3x3To12x12 case below.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [9, 9],
        expected=np.array(
            [[1, 1, 2, 2, 2, 2, 3, 3, 3], [1, 1, 2, 2, 2, 2, 3, 3, 3],
             [4, 4, 5, 5, 5, 5, 6, 6, 6], [4, 4, 5, 5, 5, 5, 6, 6, 6],
             [4, 4, 5, 5, 5, 5, 6, 6, 6], [4, 4, 5, 5, 5, 5, 6, 6, 6],
             [7, 7, 8, 8, 8, 8, 9, 9, 9], [7, 7, 8, 8, 8, 8, 9, 9, 9],
             [7, 7, 8, 8, 8, 8, 9, 9, 9]],
            dtype=np.float32))

  def testAlignCorners3x3To12x12(self):
    # [FUNCTION DOC]
    # Aligned upscale of 3x3 to 12x12, where each source pixel maps to an even block.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [12, 12],
        expected=np.array([[1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9]],
                          dtype=np.float32))

  def testBFloat16(self):
    # [FUNCTION DOC]
    # Aligned upscale of a 3x3 bfloat16 image to 4x4; expected values given in float32.
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   dtype=dtypes.bfloat16.as_numpy_dtype)
    self._assertForwardOpMatchesExpected(
        img, [4, 4],
        expected=np.array(
            [[1, 2, 2, 3], [4, 5, 5, 6], [4, 5, 5, 6], [7, 8, 8, 9]],
            dtype=np.float32))

  def testAlignCorners3x3To12x12_uint8(self):
    # [FUNCTION DOC]
    # Aligned upscale of a 3x3 uint8 image to 12x12, ensuring the convolution-based resize works
    # for integer types.
    # Ensure that resize with convolution works on XLA/GPU for integer types
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8), [12, 12],
        expected=np.array([[1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9]],
                          dtype=np.uint8))


class ResizeNearestNeighborNonAlignCornersTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests nearest-neighbor resize with align-corners off, with and without half-pixel centers,
  # across all float types of the XLA test device.

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None,
                                      large_tolerance=False,
                                      align_corners=False,
                                      half_pixel_centers=True):
    # [FUNCTION DOC]
    # HELPER: shared check for the non-aligned nearest-neighbor tests.
    # Inputs: image_np, target_shape, expected (required), large_tolerance, align_corners
    # (default false) and half_pixel_centers (default true). Resizes on the test device with
    # batch and channel axes added and asserts closeness to expected.
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_nearest_neighbor(
          image,
          target_shape,
          align_corners=align_corners,
          half_pixel_centers=half_pixel_centers)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=2e-4, atol=2e-4)
      else:
        self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  def testNonAlignCorners1x1To2x2(self):
    # [FUNCTION DOC]
    # Half-pixel upscale of 1x1 to 2x2 for each float type.
    input_data = [[64]]
    expected_data = [[64, 64], [64, 64]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [2, 2],
          expected=np.array(expected_data, dtype=np.float32))

  def testNonAlignCorners2x2To1x1(self):
    # [FUNCTION DOC]
    # Half-pixel downscale of 2x2 to 1x1 for each float type.
    input_data = [[64, 32], [8, 16]]
    expected_data = [[16]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [1, 1],
          expected=np.array(expected_data, dtype=np.float32))

  def testNonAlignCorners3x3To2x2(self):
    # [FUNCTION DOC]
    # Half-pixel resize to 2x2 from a 2x3 input (the test name says 3x3), for each float type.
    input_data = [[64, 32, 128], [4, 8, 16]]
    expected_data = [[64, 128], [4, 16]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [2, 2],
          expected=np.array(expected_data, dtype=np.float32))

  def testNonAlignCorners2x2To3x3(self):
    # [FUNCTION DOC]
    # Half-pixel upscale of 2x2 to 3x3 for each float type.
    input_data = [[64, 32], [8, 16]]
    expected_data = [[64.0, 32.0, 32.0], [8.0, 16.0, 16.0], [8.0, 16.0, 16.0]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [3, 3],
          expected=np.array(expected_data, dtype=np.float32))

  def testNonAlignCorners2x2To4x4(self):
    # [FUNCTION DOC]
    # Half-pixel upscale of 2x2 to 4x4 for each float type.
    input_data = [[64, 32], [8, 16]]
    expected_data = [[64.0, 64.0, 32.0, 32.0], [64.0, 64.0, 32.0, 32.0],
                     [8.0, 8.0, 16.0, 16.0], [8.0, 8.0, 16.0, 16.0]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [4, 4],
          expected=np.array(expected_data, dtype=np.float32))

  def testNonAlignCorners3x2To5x3(self):
    # [FUNCTION DOC]
    # Half-pixel resize of a 3x2 input to 5x3 for each float type.
    input_data = [[64, 32], [8, 16], [50, 100]]
    expected_data = [[64.0, 32.0, 32.0], [64.0, 32.0, 32.0], [8.0, 16.0, 16.0],
                     [50.0, 100.0, 100.0], [50.0, 100.0, 100.0]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [5, 3],
          expected=np.array(expected_data, dtype=np.float32))

  def testNonAlignCornersNonHalfPixelCorners2x2To1x1(self):
    # [FUNCTION DOC]
    # Resize 2x2 to 1x1 with half-pixel centers disabled; the top-left value is selected.
    input_data = [[64, 32], [8, 16]]
    expected_data = [[64]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [1, 1],
          expected=np.array(expected_data, dtype=np.float32),
          half_pixel_centers=False)

  def testNonAlignCornersNonHalfPixelCorners3x2To5x3(self):
    # [FUNCTION DOC]
    # Resize 3x2 to 5x3 with half-pixel centers disabled.
    input_data = [[64, 32], [8, 16], [50, 100]]
    expected_data = [[64.0, 64.0, 32.0], [64.0, 64.0, 32.0], [8.0, 8.0, 16.0],
                     [8.0, 8.0, 16.0], [50.0, 50.0, 100.0]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [5, 3],
          expected=np.array(expected_data, dtype=np.float32),
          half_pixel_centers=False)

  def testNonAlignCorners3x2To6x4Batch2(self):
    # [FUNCTION DOC]
    # Half-pixel resize of a batch of two 3x2 images to 6x4 for each float type, built directly
    # without the helper; asserts closeness to the expected batch.
    input_data = [[[64, 32], [32, 64], [50, 100]], [[32, 16], [16, 32],
                                                    [25, 50]]]
    expected_data = [[[64.0, 64.0, 32.0, 32.0], [64.0, 64.0, 32.0, 32.0],
                      [32.0, 32.0, 64.0, 64.0], [32.0, 32.0, 64.0, 64.0],
                      [50.0, 50.0, 100.0, 100.0], [50.0, 50.0, 100.0, 100.0]],
                     [[32.0, 32.0, 16.0, 16.0], [32.0, 32.0, 16.0, 16.0],
                      [16.0, 16.0, 32.0, 32.0], [16.0, 16.0, 32.0, 32.0],
                      [25.0, 25.0, 50.0, 50.0], [25.0, 25.0, 50.0, 50.0]]]

    for dtype in self.float_types:
      input_image = np.array(input_data, dtype=dtype)
      expected = np.array(expected_data, dtype=dtype)
      with self.session() as sess, self.test_scope():
        image = array_ops.placeholder(input_image.dtype)
        resized = gen_image_ops.resize_nearest_neighbor(
            image, [6, 4], align_corners=False, half_pixel_centers=True)
        out = sess.run(resized, {image: input_image[:, :, :, np.newaxis]})
        self.assertAllClose(expected[:, :, :, np.newaxis], out)


class ResizeBilinearTest(parameterized.TestCase, xla_test.XLATestCase):
  # [CLASS DOC]
  # Parameterized tests of bilinear resize with align-corners on the XLA test device.

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None,
                                      large_tolerance=False,
                                      align_corners=True):
    # [FUNCTION DOC]
    # HELPER: shared check for the bilinear forward tests.
    # Inputs: image_np, target_shape, expected (required), large_tolerance (relative 0.1,
    # absolute 0.01), align_corners (default true). Adds batch and channel axes, resizes on the
    # test device and asserts closeness.
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_bilinear(
          image, target_shape, align_corners=align_corners)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=0.1, atol=0.01)
      else:
        self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  @parameterized.named_parameters(
      [("1x2To3x3", 1, 2, 3, 3), ("2x2To1x1", 2, 2, 1, 1),
       ("2x2To3x3", 2, 2, 3, 3), ("3x3To2x2", 3, 3, 2, 2),
       ("4x4To3x3", 4, 4, 3, 3), ("3x3To9x9", 3, 3, 9, 9),
       ("4x4To8x8", 4, 4, 8, 8), ("8x8To16x16", 8, 8, 16, 16),
       ("64x64To512x512", 64, 64, 512, 512),
       ("80x80To512x512", 80, 80, 512, 512),
       ("96x96To512x512", 96, 96, 512, 512),
       ("112x112To512x512", 112, 112, 512, 512),
       ("256x48To2048x384", 256, 48, 2048, 384),
       ("320x60To2048x384", 320, 60, 2048, 384),
       ("448x84To2048x384", 448, 84, 2048, 384),
       ("69x69To545x545", 69, 69, 545, 545),
       ("86x86To545x545", 86, 86, 545, 545),
       ("103x103To545x545", 103, 103, 545, 545),
       ("120x120To545x545", 120, 120, 545, 545),
       ("57x57To456x456", 57, 57, 456, 456),
       ("72x72To456x456", 72, 72, 456, 456),
       ("86x86To456x456", 86, 86, 456, 456),
       ("100x100To456x456", 100, 100, 456, 456),
       ("64x64To224x224", 64, 64, 224, 224),
       ("128x128To224x224", 128, 128, 224, 224),
       ("256x256To224x224", 256, 256, 224, 224),
       ("512x512To224x224", 512, 512, 224, 224),
       ("64x64To299x299", 64, 64, 299, 299),
       ("128x128To299x299", 128, 128, 299, 299),
       ("256x256To299x299", 256, 256, 299, 299),
       ("512x512To299x299", 512, 512, 299, 299),
       ("224x224To224x224", 224, 224, 224, 224)] +
      # On windows, initialization of the following or any larger np.arrays
      # where we set the dtype explicitly fails with:
      #   TypeError: expected number, got int
      ([] if os.name == "nt" else [("224x224To224x224-bfloat", 224, 224, 224,
                                    224, dtypes.bfloat16.as_numpy_dtype)]),
      # This test is disabled because it is very slow. It is slow because
      # 383 is prime, 383 and 2047 are coprime, and 2048 is large.
      # ("Disabled_384x72To2048x384", 384, 72, 2048, 384),
  )
  def test(self, src_y, src_x, dst_y, dst_x, dtype=np.float32):
    # [FUNCTION DOC]
    # Parameterized test over many source and destination sizes. Inputs: source height and
    # width, destination height and width, and optional dtype (default float32; one case uses
    # bfloat16 except on Windows). Builds a linear ramp input so that the aligned bilinear
    # result is a known ramp, and asserts closeness with large tolerance.
    max_y = max(src_y - 1, 1) * (dst_y - 1) + 1
    max_x = max(src_x - 1, 1) * (dst_x - 1) + 1

    input_data = [
        range(y * max_x, (y + 1) * max_x, max(dst_x - 1, 1))
        for y in range(0, max_y, max(dst_y - 1, 1))
    ]

    result = [
        range(y * max_x, (y + 1) * max_x, max(src_x - 1, 1))
        for y in range(0, max_y, max(src_y - 1, 1))
    ]

    self._assertForwardOpMatchesExpected(
        np.array(input_data, dtype=dtype), [dst_y, dst_x],
        expected=np.array(result, dtype=np.float32),
        large_tolerance=True)


class ResizeBilinearGradTest(parameterized.TestCase, xla_test.XLATestCase):
  # [CLASS DOC]
  # Parameterized tests of the bilinear resize gradient with align-corners on the test device.

  def _assertBackwardOpMatchesExpected(self,
                                       grads_np,
                                       input_shape=None,
                                       dtype=None,
                                       expected=None,
                                       large_tolerance=False):
    # [FUNCTION DOC]
    # HELPER: shared check for the gradient tests.
    # Inputs: grads_np (2-D incoming gradient), input_shape (source height and width, required),
    # dtype (default float32), expected (required), large_tolerance. Runs the gradient op with a
    # zero original image of the source size and asserts closeness to the expected gradient.
    if input_shape is None:
      self.fail("input_shape must be specified")
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      dtype = dtype or np.float32
      grads = array_ops.placeholder(np.float32)
      resized = gen_image_ops.resize_bilinear_grad(
          grads,
          np.zeros([1, input_shape[0], input_shape[1], 1], dtype=dtype),
          align_corners=True)
      out = sess.run(resized, {grads: grads_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=0.1, atol=0.01)
      else:
        self.assertAllCloseAccordingToType(
            expected[np.newaxis, :, :, np.newaxis], out)

  @parameterized.named_parameters(
      ("1x3To1x3", 1, 2, 1, 3),
      ("1x2To3x2", 1, 2, 3, 2),
      ("1x2To3x3", 1, 2, 3, 3),
      ("1x1To4x1", 1, 1, 4, 1),
      ("1x1To5x1", 1, 1, 5, 1),
      ("2x2To1x1", 2, 2, 1, 1),
      ("2x2To3x3", 2, 2, 3, 3),
      ("3x3To2x2", 3, 3, 2, 2),
      ("4x4To3x3", 4, 4, 3, 3),
      ("3x3To9x9", 3, 3, 9, 9),
      ("4x4To8x8", 4, 4, 8, 8),
      ("8x8To16x16", 8, 8, 16, 16),
      ("2x64To2x512", 2, 64, 2, 512),
      ("64x64To512x512", 64, 64, 512, 512),
      ("80x80To512x512", 80, 80, 512, 512),
      ("96x96To512x512", 96, 96, 512, 512),
      ("112x112To512x512", 112, 112, 512, 512),
      # ("Disabled_256x48To2048x384", 256, 48, 2048, 384),
      # ("Disabled_320x60To2048x384", 320, 60, 2048, 384),
      # ("Disabled_448x84To2048x384", 448, 84, 2048, 384),
      ("69x69To545x545", 69, 69, 545, 545),
      ("86x86To545x545", 86, 86, 545, 545),
      ("103x103To545x545", 103, 103, 545, 545),
      ("120x120To545x545", 120, 120, 545, 545),
      ("57x57To456x456", 57, 57, 456, 456),
      ("72x72To456x456", 72, 72, 456, 456),
      ("86x86To456x456", 86, 86, 456, 456),
      ("100x100To456x456", 100, 100, 456, 456),
      # This test is disabled because it is very slow. It is slow because
      # 383 is prime, 383 and 2047 are coprime, and 2048 is large.
      # ("Disabled_384x72To2048x384", 384, 72, 2048, 384),
  )
  def test(self, src_y, src_x, dst_y, dst_x):
    # [FUNCTION DOC]
    # Parameterized test over source and destination sizes. Builds a constant incoming gradient
    # and an expected result as the outer product of per-axis weight rows, then checks the
    # gradient op output with large tolerance.

    def GetRow(src, dst):
      # [FUNCTION DOC]
      # HELPER: computes the expected one-dimensional weight row of the aligned bilinear
      # gradient for a source and destination length. Returns a one-row array of weights.
      if src == 1:
        return np.array([[max(dst**2 - dst, 1)]])
      row = [0] * src
      for i in range(0, (dst - 1) * max(src - 1, 1) + 1, src - 1):
        prev = int(math.floor(i / max(dst - 1, 1)))
        row[prev] += max(dst - 1, 1) - i % max(dst - 1, 1)
        if prev + 1 < src:
          row[prev + 1] += i % max(dst - 1, 1)
      return np.array([row])

    input_element = max(dst_x - 1, 1) * max(dst_y - 1, 1)
    input_data = [[input_element] * dst_x] * dst_y
    result = GetRow(src_x, dst_x) * np.transpose(GetRow(src_y, dst_y))
    self._assertBackwardOpMatchesExpected(
        np.array(input_data, dtype=np.float32), [src_y, src_x],
        expected=np.array(result, dtype=np.float32),
        large_tolerance=True)


class ResizeBilinearNonAlignCornersTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests bilinear resize with align-corners disabled on the test device.

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None,
                                      large_tolerance=False,
                                      align_corners=True):
    # [FUNCTION DOC]
    # HELPER: same as the aligned bilinear helper but the align_corners default is true; the
    # tests pass false explicitly. Resizes and asserts closeness to expected.
    # KNOWN ISSUE: the half-pixel-centers option is not passed, so only the legacy behavior is
    # tested.
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_bilinear(
          image, target_shape, align_corners=align_corners)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=0.1, atol=0.01)
      else:
        self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  def testNonAlignCorners3x2To6x4(self):
    # [FUNCTION DOC]
    # Non-aligned bilinear resize of 3x2 to 6x4 for each float type.
    input_data = [[64, 32], [32, 64], [50, 100]]
    expected_data = [[64.0, 48.0, 32.0, 32.0], [48.0, 48.0, 48.0, 48.0],
                     [32.0, 48.0, 64.0, 64.0], [41.0, 61.5, 82.0, 82.0],
                     [50.0, 75.0, 100.0, 100.0], [50.0, 75.0, 100.0, 100.0]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [6, 4],
          expected=np.array(expected_data, dtype=np.float32),
          align_corners=False)

  def testNonAlignCorners6x4To3x2(self):
    # [FUNCTION DOC]
    # Non-aligned bilinear downscale of 6x4 to 3x2 for each float type.
    input_data = [[127, 127, 64, 64], [127, 127, 64, 64], [64, 64, 127, 127],
                  [64, 64, 127, 127], [50, 50, 100, 100], [50, 50, 100, 100]]
    expected_data = [[127, 64], [64, 127], [50, 100]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [3, 2],
          expected=np.array(expected_data, dtype=dtype),
          align_corners=False)

  def testNonAlignCorners3x2To6x4Batch2(self):
    # [FUNCTION DOC]
    # Non-aligned bilinear resize of a batch of two 3x2 images to 6x4 for each float type.
    input_data = [[[64, 32], [32, 64], [50, 100]], [[32, 16], [16, 32],
                                                    [25, 50]]]
    expected_data = [[[64.0, 48.0, 32.0, 32.0], [48.0, 48.0, 48.0, 48.0],
                      [32.0, 48.0, 64.0, 64.0], [41.0, 61.5, 82.0, 82.0],
                      [50.0, 75.0, 100.0, 100.0], [50.0, 75.0, 100.0, 100.0]],
                     [[32.0, 24.0, 16.0, 16.0], [24.0, 24.0, 24.0, 24.0],
                      [16.0, 24.0, 32.0, 32.0], [20.5, 30.75, 41.0, 41.0],
                      [25.0, 37.5, 50.0, 50.0], [25.0, 37.5, 50.0, 50.0]]]

    for dtype in self.float_types:
      input_image = np.array(input_data, dtype=dtype)
      expected = np.array(expected_data, dtype=dtype)
      with self.session() as sess, self.test_scope():
        image = array_ops.placeholder(input_image.dtype)
        resized = gen_image_ops.resize_bilinear(
            image, [6, 4], align_corners=False)
        out = sess.run(resized, {image: input_image[:, :, :, np.newaxis]})
        self.assertAllClose(expected[:, :, :, np.newaxis], out)


class ResizeBilinearGradHalfPixelCentersTest(
    parameterized.TestCase, xla_test.XLATestCase
):
  # [CLASS DOC]
  # Tests the bilinear resize gradient with half-pixel centers and align-corners off.

  def _assertBackwardOpMatchesExpected(self, grads_np, expected):
    # [FUNCTION DOC]
    # HELPER: runs the gradient op for given incoming gradients (4-D) with a zero original image
    # of the expected shape and asserts closeness to expected according to type.
    with self.session() as sess, self.test_scope():
      grads = array_ops.placeholder(np.float32)
      resized = gen_image_ops.resize_bilinear_grad(
          grads,
          np.zeros(expected.shape, dtype=np.float32),
          align_corners=False,
          half_pixel_centers=True,
      )
      out = sess.run(resized, {grads: grads_np})
      self.assertAllCloseAccordingToType(expected, out)

  def test3x2To6x4(self):
    # [FUNCTION DOC]
    # Gradient of a 3x2 to 6x4 resize for a single channel image.
    grad_data = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
    ]
    expected_data = [[12.5, 19.5], [42.5, 49.5], [72.5, 79.5]]
    self._assertBackwardOpMatchesExpected(
        np.array(grad_data, dtype=np.float32)[np.newaxis, :, :, np.newaxis],
        expected=np.array(expected_data, dtype=np.float32)[
            np.newaxis, :, :, np.newaxis
        ],
    )

  def test6x4To3x2(self):
    # [FUNCTION DOC]
    # Gradient of a 6x4 to 3x2 resize for a single channel image.
    grad_data = [[0, 1], [2, 3], [4, 5]]
    expected_data = [
        [0.0, 0.0, 0.25, 0.25],
        [0.0, 0.0, 0.25, 0.25],
        [0.5, 0.5, 0.75, 0.75],
        [0.5, 0.5, 0.75, 0.75],
        [1.0, 1.0, 1.25, 1.25],
        [1.0, 1.0, 1.25, 1.25],
    ]
    self._assertBackwardOpMatchesExpected(
        np.array(grad_data, dtype=np.float32)[np.newaxis, :, :, np.newaxis],
        expected=np.array(expected_data, dtype=np.float32)[
            np.newaxis, :, :, np.newaxis
        ],
    )

  def test3x2To6x4Batch2(self):
    # [FUNCTION DOC]
    # Gradient for a batch of two images, 3x2 to 6x4, single channel.
    dst_y = 6
    dst_x = 4
    grad_data = np.arange(2 * dst_y * dst_x).reshape(2, dst_y, dst_x, 1)
    expected_data = [
        [[[12.5], [19.5]], [[42.5], [49.5]], [[72.5], [79.5]]],
        [[[108.5], [115.5]], [[138.5], [145.5]], [[168.5], [175.5]]],
    ]
    self._assertBackwardOpMatchesExpected(
        np.array(grad_data, dtype=np.float32),
        expected=np.array(expected_data, dtype=np.float32),
    )

  def test3x2To6x4Batch2Channel3(self):
    # [FUNCTION DOC]
    # Gradient for a batch of two images with three channels, 3x2 to 6x4.
    dst_y = 6
    dst_x = 4
    grad_data = np.arange(2 * dst_y * dst_x * 3).reshape(2, dst_y, dst_x, 3)
    expected_data = [
        [
            [[37.5, 41.5, 45.5], [58.5, 62.5, 66.5]],
            [[127.5, 131.5, 135.5], [148.5, 152.5, 156.5]],
            [[217.5, 221.5, 225.5], [238.5, 242.5, 246.5]],
        ],
        [
            [[325.5, 329.5, 333.5], [346.5, 350.5, 354.5]],
            [[415.5, 419.5, 423.5], [436.5, 440.5, 444.5]],
            [[505.5, 509.5, 513.5], [526.5, 530.5, 534.5]],
        ],
    ]
    self._assertBackwardOpMatchesExpected(
        np.array(grad_data, dtype=np.float32),
        expected=np.array(expected_data, dtype=np.float32),
    )


class NonMaxSuppressionTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests non-max suppression ops (version 3 and padded) on the XLA test device with a fixed
  # set of six boxes, scores and thresholds.

  def testNMSV3(self):
    # [FUNCTION DOC]
    # Runs version 3 suppression with an IOU threshold of 0.5, no score threshold and a maximum
    # of six outputs; asserts three indices are returned: 3, 0, 5.
    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)
    max_output_size = 6
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_v3(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=float("-inf"))
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          iou_threshold: iou_threshold_np
      }
      (indices_tf) = sess.run(selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, 3)
      self.assertAllClose(indices_tf[:3], [3, 0, 5])

  def testNMS128From1024(self):
    # [FUNCTION DOC]
    # Runs padded suppression on 1024 random boxes and scores, selecting up to 128; asserts the
    # padded output holds exactly 128 entries.
    num_boxes = 1024
    boxes_np = np.random.normal(50, 10, (num_boxes, 4)).astype("f4")
    scores_np = np.random.normal(0.5, 0.1, (num_boxes,)).astype("f4")

    max_output_size = 128
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.0, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          score_threshold: score_threshold_np,
          iou_threshold: iou_threshold_np
      }
      (indices_tf, _) = sess.run(selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)

  def testNMS3From6Boxes(self):
    # [FUNCTION DOC]
    # Padded suppression with maximum three, threshold 0.5; asserts output size three, three
    # valid, indices 3, 0, 5.
    # Three boxes are selected based on IOU.
    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)

    max_output_size = 3
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.0, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          score_threshold: score_threshold_np,
          iou_threshold: iou_threshold_np
      }
      (indices_tf, num_valid) = sess.run(
          selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)
      self.assertEqual(num_valid, 3)
      self.assertAllClose(indices_tf[:num_valid], [3, 0, 5])

  def testNMS3Then2WithScoreThresh(self):
    # [FUNCTION DOC]
    # Padded suppression with a score threshold of 0.4; asserts two valid indices, 3 and 0.
    # Three boxes are selected based on IOU.
    # One is filtered out by score threshold.

    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)
    max_output_size = 3
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.4, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          iou_threshold: iou_threshold_np,
          score_threshold: score_threshold_np
      }
      (indices_tf, num_valid) = sess.run(
          selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)
      self.assertEqual(num_valid, 2)
      self.assertAllClose(indices_tf[:num_valid], [3, 0])

  def testNMS3Then1WithScoreMaxThresh(self):
    # [FUNCTION DOC]
    # Padded suppression with maximum output one and score threshold 0.4; asserts one valid
    # index, 3.
    # Three boxes are selected based on IOU.
    # One is filtered out by score threshold.
    # One is filtered out by max_output_size.

    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)
    max_output_size = 1
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.4, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          iou_threshold: iou_threshold_np,
          score_threshold: score_threshold_np
      }
      (indices_tf, num_valid) = sess.run(
          selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)
      self.assertEqual(num_valid, 1)
      self.assertAllClose(indices_tf[:num_valid], [3])

  def testSelectFromContinuousOverLap(self):
    # [FUNCTION DOC]
    # Verifies that a suppressed box does not itself suppress others, using overlapping boxes;
    # asserts three valid indices: 0, 2, 4.
    # Tests that a suppressed box does not itself suppress other boxes.

    boxes_data = [[0, 0, 1, 1], [0, 0.2, 1, 1.2], [0, 0.4, 1, 1.4],
                  [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 3]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.5, 0.4, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)
    max_output_size = 3
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.1, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          iou_threshold: iou_threshold_np,
          score_threshold: score_threshold_np
      }
      (indices_tf, num_valid) = sess.run(
          selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)
      self.assertEqual(num_valid, 3)
      self.assertAllClose(indices_tf[:num_valid], [0, 2, 4])


class BatchedNonMaxSuppressionCorrectnessTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests batched padded non-max suppression on the XLA test device: batch dimensions, sorted
  # and unsorted input, score thresholds, canonical and non-canonical coordinates, dynamic shape.

  def testBatchedNMSFrom6(self):
    # [FUNCTION DOC]
    # Two batches of six boxes, sorted input, maximum six; asserts per-batch padded indices
    # (padding index zero) and valid counts 5 and 4.
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                   [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)

      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([[0, 1, 2, 4, 5, invalid_index],
                         [0, 1, 3, 5, invalid_index, invalid_index]],
                        indices_output)
    self.assertAllEqual([5, 4], num_valid_output)

  def testBatchedNMSFrom6Max3(self):
    # [FUNCTION DOC]
    # Same boxes with maximum output three; asserts indices and valid counts 3 and 3.
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                   [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    self.assertAllEqual([[0, 1, 2], [0, 1, 3]], indices_output)
    self.assertAllEqual([3, 3], num_valid_output)

  def testBatchedNMSSingleFrom6Max3(self):
    # [FUNCTION DOC]
    # Unbatched six boxes, maximum three, padded; asserts indices 0, 1, 2 and count three.
    boxes_data = [[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                  [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]]
    scores_data = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    self.assertAllEqual([0, 1, 2], indices_output)
    self.assertAllEqual(3, num_valid_output)

  def testBatchedNMSSingleFrom6NoPad(self):
    # [FUNCTION DOC]
    # Unbatched six boxes, maximum six, without padding; asserts indices 0, 1, 2, 4, 5 and
    # count five.
    boxes_data = [[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                  [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]]
    scores_data = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    self.assertAllEqual([0, 1, 2, 4, 5], indices_output)
    self.assertAllEqual(5, num_valid_output)

  def testBatchedNMSBatchDimsFrom6Max3(self):
    # [FUNCTION DOC]
    # Inputs with an extra leading batch dimension, maximum three; asserts output shape and
    # indices and counts of 3 and 3.
    boxes_data = [[[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                    [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                   [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                    [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]]
    scores_data = [[[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    self.assertAllEqual([[[0, 1, 2], [0, 1, 3]]], indices_output)
    self.assertAllEqual([[3, 3]], num_valid_output)

  def testBatchedNMSScoreThresholdFrom6Max3(self):
    # [FUNCTION DOC]
    # Batched suppression with score threshold 0.5, maximum three; asserts counts 3 and 2 and
    # the padded indices.
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.4, 0.3, 0.2],
                   [0.8, 0.7, 0.6, 0.4, 0.3, 0.1]]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=0.5,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([3, 2], num_valid_output)
    self.assertAllEqual([[0, 1, 2], [0, 1, invalid_index]], indices_output)

  def testBatchedNMSUnsortedInputFrom6(self):
    # [FUNCTION DOC]
    # Unsorted input (sorting done by the op), maximum six; asserts the indices refer to the
    # original unsorted positions and counts 5 and 4.
    boxes_data = [[[0, 2, 1, 2], [3, 3, 4, 4], [0, 0, 1, 1], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8]],
                  [[0, 0.4, 1, 1.4], [0, 2, 1, 2], [0, 0.2, 1, 1.2],
                   [0, 0, 1, 1], [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8]]]
    scores_data = [[0.3, 0.7, 0.9, 0.6, 0.5, 0.4],
                   [0.5, 0.8, 0.4, 0.3, 0.6, 0.7]]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)

      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            canonicalized_coordinates=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([[2, 1, 3, 5, 0, invalid_index],
                         [1, 5, 0, 3, invalid_index, invalid_index]],
                        indices_output)
    self.assertAllEqual([5, 4], num_valid_output)

  def testBatchedNMSNoncanonicalizedInputFrom6(self):
    # [FUNCTION DOC]
    # Boxes with non-canonical corner order and the canonicalization flag left at its default;
    # asserts the same result as the canonical case: counts 5 and 4.
    boxes_data = [[[1, 0, 0, 1], [4, 3, 3, 4], [1, 0.4, 0, 1.4],
                   [1, 0.6, 0, 1.6], [1, 0.8, 0, 1.8], [1, 2, 0, 2]],
                  [[1, 2, 0, 2], [1, 0.8, 0, 1.8], [1, 0.6, 0, 1.6],
                   [1, 0.4, 0, 1.4], [1, 0.2, 0, 1.2], [1, 0, 0, 1]]]

    scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                   [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)

      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([[0, 1, 2, 4, 5, invalid_index],
                         [0, 1, 3, 5, invalid_index, invalid_index]],
                        indices_output)
    self.assertAllEqual([5, 4], num_valid_output)

  def testBatchedNMSScoreThresholdCanInputsFrom6Max3(self):
    # [FUNCTION DOC]
    # Score threshold 0.5 with canonicalized coordinates explicitly false; asserts counts 3 and
    # 2 and the padded indices.
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.4, 0.3, 0.2],
                   [0.8, 0.7, 0.6, 0.4, 0.3, 0.1]]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=0.5,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=False)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([3, 2], num_valid_output)
    self.assertAllEqual([[0, 1, 2], [0, 1, invalid_index]], indices_output)

  def testBatchedNMSFrom6DynamicInput(self):
    # [FUNCTION DOC]
    # Same as the first batched test but with placeholders of unknown shape; asserts the same
    # indices and valid counts 5 and 4.
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                   [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype)
      scores = array_ops.placeholder(scores_np.dtype)

      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {boxes: boxes_np, scores: scores_np}
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([[0, 1, 2, 4, 5, invalid_index],
                         [0, 1, 3, 5, invalid_index, invalid_index]],
                        indices_output)
    self.assertAllEqual([5, 4], num_valid_output)


if __name__ == "__main__":
  test.main()
