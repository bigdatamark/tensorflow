# [MODULE DOC]
# Test module for the roll operation in the manipulation ops: compares against numpy's roll
# across integer, floating and complex types, checks gradients, negative axes, empty and very
# large inputs, and the errors raised for bad ranks, mismatched shift/axis sizes and
# out-of-range axes. Defines a module constant telling whether numpy supports multi-axis roll.
# Defines one test class; run via the test runner entry point when executed as a script.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for manip_ops."""
import numpy as np
from packaging.version import Version

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import manip_ops
from tensorflow.python.platform import test as test_lib

# numpy.roll for multiple shifts was introduced in numpy version 1.12.0
NP_ROLL_CAN_MULTISHIFT = Version(np.version.version) >= Version("1.12.0")


class RollTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the roll operation. Holds private helpers that compare roll against numpy
  # (and optionally numeric gradients) plus many tests covering dtypes and error handling.

  def _testRoll(self, np_input, shift, axis):
    # [FUNCTION DOC]
    # HELPER: shared check used by the roll tests via the combined helper.
    # Computes the expected result with numpy's roll for the given array, shift and axis, then
    # runs the roll operation in a cached session and asserts equality. Inputs: a numpy array,
    # a shift (integer or list), an axis (integer or list). Returns nothing.
    expected_roll = np.roll(np_input, shift, axis)
    with self.cached_session():
      roll = manip_ops.roll(np_input, shift, axis)
      self.assertAllEqual(roll, expected_roll)

  def _testGradient(self, np_input, shift, axis):
    # [FUNCTION DOC]
    # HELPER: gradient check used by the roll tests via the combined helper.
    # Wraps the input in a constant, applies roll, and uses the gradient checker to compare the
    # theoretical and numeric Jacobians (input and output shapes are identical) with relative
    # and absolute tolerance of 1e-5. Inputs: numpy array, shift, axis. Returns nothing.
    with self.cached_session():
      inx = constant_op.constant(np_input.tolist())
      xs = list(np_input.shape)
      y = manip_ops.roll(inx, shift, axis)
      # Expected y's shape to be the same
      ys = xs
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, y, ys, x_init_value=np_input)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testAll(self, np_input, shift, axis):
    # [FUNCTION DOC]
    # HELPER: combined check for the roll tests.
    # Always runs the value comparison against numpy; additionally runs the gradient check only
    # when the input dtype is 32-bit float. Inputs: numpy array, shift, axis. Returns nothing.
    self._testRoll(np_input, shift, axis)
    if np_input.dtype == np.float32:
      self._testGradient(np_input, shift, axis)

  @test_util.run_deprecated_v1
  def testIntTypes(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit integers, roll random arrays: a vector with a scalar
    # shift, and (only when numpy supports multi-shift) rank-three and rank-four arrays with
    # list shifts including negatives over several axes. Expects equality with numpy.
    # Runs in the deprecated graph mode. Returns nothing.
    for t in [np.int32, np.int64]:
      self._testAll(np.random.randint(-100, 100, (5)).astype(t), 3, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        self._testAll(
            np.random.randint(-100, 100, (4, 4, 3)).astype(t), [1, -2, 3],
            [0, 1, 2])
        self._testAll(
            np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t), [0, 1, -2],
            [1, 2, 3])

  @test_util.run_deprecated_v1
  def testFloatTypes(self):
    # [FUNCTION DOC]
    # Test scenario: same as the integer test but for 32-bit, 64-bit and bfloat16 floats with
    # random arrays of rank one to three; multi-shift cases only when numpy supports them.
    # 32-bit float inputs also get the gradient check. Graph mode. Returns nothing.
    for t in [np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]:
      self._testAll(np.random.rand(5).astype(t), 2, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        self._testAll(np.random.rand(3, 4).astype(t), [1, 2], [1, 0])
        self._testAll(np.random.rand(1, 3, 4).astype(t), [1, 0, -3], [0, 1, 2])

  @test_util.run_deprecated_v1
  def testComplexTypes(self):
    # [FUNCTION DOC]
    # Test scenario: for 64- and 128-bit complex types, build random complex arrays (real part
    # plus the same values times the imaginary unit) and roll them with scalar and list shifts
    # (list cases only when numpy supports multi-shift). Compares with numpy. Graph mode.
    # Returns nothing.
    for t in [np.complex64, np.complex128]:
      x = np.random.rand(4, 4).astype(t)
      self._testAll(x + 1j * x, 2, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        x = np.random.rand(2, 5).astype(t)
        self._testAll(x + 1j * x, [1, 2], [1, 0])
        x = np.random.rand(3, 2, 1, 1).astype(t)
        self._testAll(x + 1j * x, [2, 1, 1, 0], [0, 3, 1, 2])

  @test_util.run_deprecated_v1
  def testNegativeAxis(self):
    # [FUNCTION DOC]
    # Test scenario: rolling with axis minus one on a vector and minus two on a matrix matches
    # numpy. Then rolling a matrix with axis minus ten must raise an invalid-argument error
    # whose message says the axis is out of range. Graph mode. Returns nothing.
    self._testAll(np.random.randint(-100, 100, (5)).astype(np.int32), 3, -1)
    self._testAll(np.random.randint(-100, 100, (4, 4)).astype(np.int32), 3, -2)
    # Make sure negative axis should be 0 <= axis + dims < dims
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is out of range"):
        manip_ops.roll(np.random.randint(-100, 100, (4, 4)).astype(np.int32),
                       3, -10).eval()

  @test_util.run_deprecated_v1
  def testEmptyInput(self):
    # [FUNCTION DOC]
    # Test scenario: rolling empty arrays (zero-by-one and one-by-zero) by one along axis one
    # must match numpy and not fail. Graph mode. Returns nothing.
    self._testAll(np.zeros([0, 1]), 1, 1)
    self._testAll(np.zeros([1, 0]), 1, 1)

  @test_util.run_v2_only
  def testLargeInput(self):
    # [FUNCTION DOC]
    # Test scenario (v2 behavior only, forced onto the CPU): build an 8-bit array with a bit more
    # than 2^31 elements (reshaped to 128 by N by 33) so index arithmetic would overflow 32 bits.
    # For shifts from minus five to four along axis zero, rolls it and checks that the row at
    # the shift index equals the first input row and the first output row equals the input row
    # at the negated shift index, with the shift in the failure message. Returns nothing.
    with test_util.force_cpu():
      # Num elements just over INT_MAX for int32 to ensure no overflow
      np_input = np.arange(0, 128 * 524289 * 33, dtype=np.int8).reshape(
          128, -1, 33
      )

      for shift in range(-5, 5):
        roll = manip_ops.roll(np_input, shift, 0)
        self.assertAllEqual(roll[shift], np_input[0], msg=f"shift={shift}")
        self.assertAllEqual(roll[0], np_input[-shift], msg=f"shift={shift}")

  @test_util.run_deprecated_v1
  def testInvalidInputShape(self):
    # [FUNCTION DOC]
    # Test scenario: rolling a scalar input must raise a value error at graph construction
    # (shape function) saying the shape must be at least rank one but is rank zero.
    # Graph mode. Returns nothing.
    # The input should be 1-D or higher, checked in shape function.
    with self.assertRaisesRegex(
        ValueError, "Shape must be at least rank 1 but is rank 0"
    ):
      manip_ops.roll(7, 1, 0)

  @test_util.run_deprecated_v1
  def testRollInputMustVectorHigherRaises(self):
    # [FUNCTION DOC]
    # Test scenario: the input is a placeholder of unknown shape so the shape function cannot
    # reject it; feeding a scalar at run time must make the kernel raise an invalid-argument
    # error saying the input must be 1-D or higher. Graph mode. Returns nothing.
    # The input should be 1-D or higher, checked in kernel.
    tensor = array_ops.placeholder(dtype=dtypes.int32)
    shift = 1
    axis = 0
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "input must be 1-D or higher"):
        manip_ops.roll(tensor, shift, axis).eval(feed_dict={tensor: 7})

  @test_util.run_deprecated_v1
  def testInvalidAxisShape(self):
    # [FUNCTION DOC]
    # Test scenario: an axis argument of rank two must raise a value error at construction
    # saying the shape must be at most rank one but is rank two. Graph mode. Returns nothing.
    # The axis should be a scalar or 1-D, checked in shape function.
    with self.assertRaisesRegex(ValueError,
                                "Shape must be at most rank 1 but is rank 2"):
      manip_ops.roll([[1, 2], [3, 4]], 1, [[0, 1]])

  @test_util.run_deprecated_v1
  def testRollAxisMustBeScalarOrVectorRaises(self):
    # [FUNCTION DOC]
    # Test scenario: axis is a placeholder of unknown shape; feeding a rank-two value at run
    # time must make the kernel raise an invalid-argument error saying the axis must be a
    # scalar or a 1-D vector. Graph mode. Returns nothing.
    # The axis should be a scalar or 1-D, checked in kernel.
    tensor = [[1, 2], [3, 4]]
    shift = 1
    axis = array_ops.placeholder(dtype=dtypes.int32)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "axis must be a scalar or a 1-D vector"):
        manip_ops.roll(tensor, shift, axis).eval(feed_dict={axis: [[0, 1]]})

  @test_util.run_deprecated_v1
  def testInvalidShiftShape(self):
    # [FUNCTION DOC]
    # Test scenario: a shift argument of rank two must raise a value error at construction
    # saying the shape must be at most rank one but is rank two. Graph mode. Returns nothing.
    # The shift should be a scalar or 1-D, checked in shape function.
    with self.assertRaisesRegex(ValueError,
                                "Shape must be at most rank 1 but is rank 2"):
      manip_ops.roll([[1, 2], [3, 4]], [[0, 1]], 1)

  @test_util.run_deprecated_v1
  def testRollShiftMustBeScalarOrVectorRaises(self):
    # [FUNCTION DOC]
    # Test scenario: shift is a placeholder of unknown shape; feeding a rank-two value at run
    # time must make the kernel raise an invalid-argument error saying the shift must be a
    # scalar or a 1-D vector. Graph mode. Returns nothing.
    # The shift should be a scalar or 1-D, checked in kernel.
    tensor = [[1, 2], [3, 4]]
    shift = array_ops.placeholder(dtype=dtypes.int32)
    axis = 1
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "shift must be a scalar or a 1-D vector"):
        manip_ops.roll(tensor, shift, axis).eval(feed_dict={shift: [[0, 1]]})

  @test_util.run_deprecated_v1
  def testInvalidShiftAndAxisNotEqualShape(self):
    # [FUNCTION DOC]
    # Test scenario: a one-element shift with a two-element axis must raise a value error at
    # construction (shape function) saying both shapes must be equal. Graph mode. Returns nothing.
    # The shift and axis must be same size, checked in shape function.
    with self.assertRaisesRegex(ValueError, "both shapes must be equal"):
      manip_ops.roll([[1, 2], [3, 4]], [1], [0, 1])

  @test_util.run_deprecated_v1
  def testRollShiftAndAxisMustBeSameSizeRaises(self):
    # [FUNCTION DOC]
    # Test scenario: shift is a placeholder of unknown shape and axis a two-element list; feeding
    # a one-element shift at run time must make the kernel raise an invalid-argument error saying
    # shift and axis must have the same size. Graph mode. Returns nothing.
    # The shift and axis must be same size, checked in kernel.
    tensor = [[1, 2], [3, 4]]
    shift = array_ops.placeholder(dtype=dtypes.int32)
    axis = [0, 1]
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "shift and axis must have the same size"):
        manip_ops.roll(tensor, shift, axis).eval(feed_dict={shift: [1]})

  def testRollAxisOutOfRangeRaises(self):
    # [FUNCTION DOC]
    # Test scenario: rolling a two-element vector along axis one (valid axes are only zero)
    # must raise an invalid-argument error saying the axis is out of range when evaluated.
    # Returns nothing.
    tensor = [1, 2]
    shift = 1
    axis = 1
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is out of range"):
        manip_ops.roll(tensor, shift, axis).eval()


if __name__ == "__main__":
  test_lib.main()
