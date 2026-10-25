# [MODULE DOC]
# Tests that image_ops brightness and gamma adjustment accept weak tensors (WeakTensor) and return
# WeakTensor results with correct values, and that invalid (negative) gamma is rejected.
# Uses the shared weak tensor test utility to create weak inputs. Running as a script sets the
# dtype conversion mode to "all" before launching the test runner.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.image_ops on WeakTensor."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.platform import test

_get_weak_tensor = weak_tensor_test_util.get_weak_tensor


class AdjustBrightnessTest(test.TestCase):
  # [CLASS DOC]
  # Test case for image_ops.adjust_brightness applied to weak tensor inputs. Uses a shared helper
  # that checks result type and values; individual tests supply float32 and float64 data.

  def _testBrightness(self, x_np, y_np, delta, tol=1e-6):
    # [FUNCTION DOC]
    # HELPER: shared checker for the brightness tests in this class.
    # Inputs: x_np, a numpy image array; y_np, the expected numpy result; delta, the amount added
    # to brightness; tol, the comparison tolerance (default 1e-6). Wraps x_np as a weak tensor,
    # calls adjust_brightness within a cached session, evaluates it, and asserts the result is a
    # WeakTensor and close to y_np within tol. Returns nothing.
    with self.cached_session():
      x = _get_weak_tensor(x_np, shape=x_np.shape)
      y = image_ops.adjust_brightness(x, delta)
      y_tf = self.evaluate(y)

      self.assertIsInstance(y, WeakTensor)
      self.assertAllClose(y_tf, y_np, tol)

  def testPositiveDeltaFloat32(self):
    # [FUNCTION DOC]
    # Builds a 2x2x3 float32 image scaled to [0, 1], the expected image with 10/255 added to each
    # value (not clipped, so values can exceed 1), and runs the brightness helper with delta 10/255
    # at the default tolerance.
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float32).reshape(x_shape) / 255.0

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
    y_np = np.array(y_data, dtype=np.float32).reshape(x_shape) / 255.0

    self._testBrightness(x_np, y_np, delta=10.0 / 255.0)

  def testPositiveDeltaFloat64(self):
    # [FUNCTION DOC]
    # Same scenario as the float32 positive-delta test but with float64 data; the helper is run
    # with delta 10/255 and a looser tolerance of 1e-3.
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float64).reshape(x_shape) / 255.0

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
    y_np = np.array(y_data, dtype=np.float64).reshape(x_shape) / 255.0

    self._testBrightness(x_np, y_np, delta=10.0 / 255.0, tol=1e-3)


class AdjustGamma(test.TestCase):
  # [CLASS DOC]
  # Test case for image_ops.adjust_gamma on weak tensor inputs: error handling for negative gamma
  # and value/type checks for several gamma values through a shared helper.

  def test_adjust_gamma_less_zero_float32(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a white image is returned for gamma equal to zero, but the
    # test actually passes a negative gamma (-1) and asserts that an error is raised.
    # Creates a random 8x8 float32 weak tensor and asserts that adjust_gamma with gamma -1 raises
    # a value error or invalid-argument error with the message "Gamma should be a non-negative
    # real number".
    """White image should be returned for gamma equal to zero."""
    with self.cached_session():
      x_data = np.random.uniform(0, 1.0, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = _get_weak_tensor(x_np, shape=x_np.shape)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), err_msg):
        image_ops.adjust_gamma(x, gamma=-1)

  def test_adjust_gamma_less_zero_tensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring (copied from another test) says a white image is returned for
    # gamma equal to zero, but the test passes a negative gamma tensor and expects an error.
    # Creates a random 8x8 float32 weak tensor and a float32 constant tensor -1.0 as gamma, then
    # asserts that building and evaluating adjust_gamma raises a value error or invalid-argument
    # error with the message "Gamma should be a non-negative real number".
    """White image should be returned for gamma equal to zero."""
    with self.cached_session():
      x_data = np.random.uniform(0, 1.0, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = _get_weak_tensor(x_np, shape=x_np.shape)
      y = constant_op.constant(-1.0, dtype=dtypes.float32)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), err_msg):
        image = image_ops.adjust_gamma(x, gamma=y)
        self.evaluate(image)

  def _test_adjust_gamma_float32(self, gamma):
    # [FUNCTION DOC]
    # HELPER: shared checker for the float32 gamma tests in this class.
    # Input: gamma, a python float exponent. Wraps a random 8x8 array as a weak tensor, applies
    # adjust_gamma, evaluates it, asserts the result is a WeakTensor, and asserts it is close
    # (tolerance 1e-6) to the input raised to gamma and clipped to the range 0 to 1.
    # Returns nothing.
    """Verifying the output with expected results for gamma correction for float32 images."""
    with self.cached_session():
      x_np = np.random.uniform(0, 1.0, (8, 8))
      x = _get_weak_tensor(x_np, shape=x_np.shape)
      y = image_ops.adjust_gamma(x, gamma=gamma)
      y_tf = self.evaluate(y)

      self.assertIsInstance(y, WeakTensor)
      y_np = np.clip(np.power(x_np, gamma), 0, 1.0)

      self.assertAllClose(y_tf, y_np, 1e-6)

  def test_adjust_gamma_one_float32(self):
    # [FUNCTION DOC]
    # Runs the float32 gamma helper with gamma 1.0 (identity result expected).
    """Same image should be returned for gamma equal to one."""
    self._test_adjust_gamma_float32(1.0)

  def test_adjust_gamma_less_one_float32(self):
    # [FUNCTION DOC]
    # Runs the float32 gamma helper with gamma 0.5, a value below one.
    """Verifying the output with expected results for gamma correction with gamma equal to half for float32 images."""
    self._test_adjust_gamma_float32(0.5)

  def test_adjust_gamma_greater_one_float32(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says gamma equal to two, but the code runs the helper with gamma
    # 1.0, so this duplicates the gamma-one test and never exercises a gamma above one.
    # Runs the float32 gamma helper with gamma 1.0.
    """Verifying the output with expected results for gamma correction with gamma equal to two for float32 images."""
    self._test_adjust_gamma_float32(1.0)

  def test_adjust_gamma_zero_float32(self):
    # [FUNCTION DOC]
    # Runs the float32 gamma helper with gamma 0.0; the expected result is an all-ones (white)
    # image because any value raised to zero is one.
    """White image should be returned for gamma equal to zero for float32 images."""
    self._test_adjust_gamma_float32(0.0)


if __name__ == "__main__":
  ops.set_dtype_conversion_mode("all")
  test.main()
