# [MODULE DOC]
# Functional tests for the quantized 2-D convolution op. Defines one test class that feeds
# small unsigned 8-bit quantized inputs and filters (incrementing values) through the quantized
# convolution with 32-bit quantized output, converts the result back to floating point and
# compares it to hand-computed totals. Also checks that malformed rank arguments are rejected.
# Not a helper module; no import-time side effects beyond the standard test main hook.
#
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
"""Functional tests for quantized convolutional operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class Conv2DTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the quantized 2-D convolution op. Holds no state; shared fixtures are the
  # helper methods that run the op on incrementing uint8 data, de-quantize the 32-bit output and
  # check the expected values, plus a helper checking that bad argument ranks raise errors.

  def __init__(self, method_name="runTest"):
    # [FUNCTION DOC]
    # Constructor that only forwards the test method name (default is the standard run method
    # name) to the base test case constructor. No other state or side effects.
    super(Conv2DTest, self).__init__(method_name)

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    expected):
    # [FUNCTION DOC]
    # HELPER: shared verification routine for the convolution value tests.
    # Inputs: tensor_in_sizes (four sizes: batch, rows, columns, depth), filter_in_sizes (four
    # sizes: kernel rows, kernel columns, input depth, output depth), stride (integer used for both
    # spatial strides), padding (string such as VALID or SAME) and expected (list of expected
    # floating-point outputs).
    # Builds uint8 input and filter data counting up from one, wraps them as quantized tensors with
    # range zero to 255, runs the quantized convolution with 32-bit quantized output on CPU,
    # converts the output to floats using the reported output range and asserts every value is
    # within 1.0 of the expected one and that the evaluated shape equals the static shape.
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = np.array([f for f in range(1, total_size_1 + 1)])
    x1 = x1.astype(np.uint8).reshape(tensor_in_sizes)
    x1_min = 0.0
    x1_max = 255.0
    x2 = np.array([f for f in range(1, total_size_2 + 1)]).astype(np.uint8)
    x2 = x2.astype(np.uint8).reshape(filter_in_sizes)
    x2_min = 0.0
    x2_max = 255.0
    with self.cached_session(use_gpu=False) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtypes.quint8)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtypes.quint8)
      conv = nn_ops.quantized_conv2d(
          t1,
          t2,
          out_type=dtypes.qint32,
          strides=[1, stride, stride, 1],
          padding=padding,
          min_input=x1_min,
          max_input=x1_max,
          min_filter=x2_min,
          max_filter=x2_max)
      value = self.evaluate(conv)
    quantized_output = value[0]
    output_min = value[1]
    output_max = value[2]
    float_output = self._QuantizedOutputToFloat(quantized_output, output_min,
                                                output_max)
    self.assertArrayNear(expected, float_output.flatten(), 1.0)
    self.assertEqual(value[0].shape, conv[0].get_shape())

  def _assertQuantizedArrayEquals(self, iarray1, iarray2):
    # [FUNCTION DOC]
    # HELPER: assertion utility for quantized arrays (not used by the tests in this file).
    # KNOWN ISSUE: pairs elements with zip, so extra elements of the longer array are silently
    # ignored and differing lengths are not detected.
    # Asserts that corresponding elements of the two sequences are equal; fails the test on the
    # first mismatch. Returns nothing.
    for i1, i2 in zip(iarray1, iarray2):
      self.assertTrue(i1 == i2)

  def _QuantizedOutputToFloat(self, quantized, quantized_min, quantized_max):
    # [FUNCTION DOC]
    # HELPER: de-quantizes the 32-bit quantized convolution output for comparison.
    # Inputs: quantized (array of 32-bit signed quantized values), quantized_min and quantized_max
    # (the float range the quantized values represent). Maps each value linearly from the signed
    # 32-bit range onto the float range (with a small range adjustment) using double precision.
    # Returns a flat one-dimensional array of doubles. No side effects.
    number_of_bits = 32
    number_of_steps = 1 << number_of_bits
    range_adjust = (number_of_steps / (number_of_steps - np.float64(1.0)))
    quantized_range = ((quantized_max - quantized_min) * range_adjust)
    range_scale = (quantized_range / number_of_steps)
    lowest_quantized = -(1 << (number_of_bits - 1))
    result = np.array([(quantized_min +
                        ((float(x) - lowest_quantized) * range_scale))
                       for x in quantized.flatten()], dtype=np.float64)
    return result

  def testConv2D1x1Filter(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comments in the body say some results are "clamped to 255", but the
    # expected list holds the unclamped values (258, 261, 312) and that is what is asserted.
    # Scenario: 1x2x3x3 input convolved with a 1x1 filter (3 in, 3 out channels), stride 1, VALID
    # padding. Expected: the hand-computed per-pixel dot products (30 up to 312), compared within a
    # tolerance of 1.0. Exercises the non-optimized reference path because the output shift is zero.
    # Our generated input is [batch, rows, cols, depth], and looks like this:
    # (1,2,3)    (4,5,6)    (7,8,9)
    # (10,11,12) (13,14,15) (16,17,18)
    # The filter data is:
    # (1,4,7) (2,5,8) (3,6,9)
    # That means the calculations are:
    # 1*1+2*4+3*7=30
    # 1*2+2*5+3*8=36
    # 1*3+2*6+3*9=42
    # 4*1+5*4+6*7=66
    # 4*2+5*5+6*8=81
    # 4*3+5*6+6*9=96
    # 7*1+5*8+6*9=102
    # 7*2+8*5+9*8=126
    # 7*3+8*6+9*9=150
    # 10*1+11*4+12*7=138
    # 10*2+11*5+12*8=171
    # 10*3+11*6+12*9=204
    # 13*1+14*4+15*7=174
    # 13*2+14*5+15*8=216
    # 13*3+14*6+15*9=258, clamped to 255
    # 16*1+17*4+18*7=210
    # 16*2+17*5+18*8=261, clamped to 255
    # 16*3+17*6+18*9=312, clamped to 255
    # Because the output shift is zero, we call the non-optimized reference
    # path for the convolution.
    expected_output = [
        30, 36, 42, 66, 81, 96, 102, 126, 150, 138, 171, 204, 174, 216, 258,
        210, 261, 312
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2Filter(self):
    # [FUNCTION DOC]
    # Scenario: 1x2x3x3 input with a 2x2 filter (3 in, 3 out channels), stride 1, VALID padding.
    # Expected: six outputs (two positions by three channels) equal to the hand-computed raw
    # accumulated totals (2271 through 3165), within tolerance.
    # Our generated input is [batch, rows, cols, depth], and looks like this:
    # (1,2,3)    (4,5,6)    (7,8,9)
    # (10,11,12) (13,14,15) (16,17,18)
    # The filter data is [filter_height, filter_width, depth, filter_count]:
    # ( 1, 4, 7) (10, 13, 16)
    # (19,22,25) (28, 31, 34)
    # -
    # ( 2, 5, 8) (11, 14, 17)
    # (20,23,26) (29, 32, 35)
    # -
    # ( 3, 6, 9) (12, 15, 18)
    # (21,24,27) (30, 33, 36)
    # The raw accumulated totals are:
    # 1*1+2*4+3*7+4*10+5*13+6*16+10*19+11*22+12*25+13*28+14*31+15*34=2271
    # 1*2+2*5+3*8+4*11+5*14+6*17+10*20+11*23+12*26+13*29+14*32+15*35=2367
    # 1*3+2*6+3*9+4*12+5*15+6*18+10*21+11*24+12*27+13*30+14*33+15*36=2463
    # 4*1+5*4+6*7+7*10+8*13+9*16+13*19+14*22+15*25+16*28+17*31+18*34=2901
    # 4*2+5*5+6*8+7*11+8*14+9*17+13*20+14*23+15*26+16*29+17*32+18*35=3033
    # 4*3+5*6+6*9+7*12+8*15+9*18+13*21+14*24+15*27+16*30+17*33+18*36=3165
    # The expected values are taken from the raw totals and rescaled to fit into
    # eight bits.
    expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2Filter(self):
    # [FUNCTION DOC]
    # Scenario: 1x2x3x3 input with a 1x2 filter, stride 1, VALID padding. Expected: twelve outputs
    # matching precomputed values (231 through 1029), within tolerance. The comment says this
    # configuration takes the optimized computation path.
    # The outputs are computed using third_party/py/IPython/notebook.
    # With a shift of 21, we should execute the optimized path here.
    expected_output = [
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2(self):
    # [FUNCTION DOC]
    # Scenario: 1x2x3x3 input with a 2x2 filter, stride 2, VALID padding. Expected: three outputs
    # (2271, 2367, 2463), within tolerance; uses the optimized path per the comment.
    # With a shift of 21, we should execute the optimized path here.
    expected_output = [2271.0, 2367.0, 2463.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=2,
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2Same(self):
    # [FUNCTION DOC]
    # Scenario: 1x2x3x3 input with a 2x2 filter, stride 2, SAME padding. Expected: six outputs
    # (2271, 2367, 2463, 1230, 1305, 1380), the second group reflecting the zero-padded border,
    # within tolerance.
    # With a shift of 21, we should execute the optimized path here.
    expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=2,
        padding="SAME",
        expected=expected_output)

  def _testBadInputSize(self,
                        tin=None,
                        tfilter=None,
                        min_input=None,
                        max_input=None,
                        min_filter=None,
                        max_filter=None,
                        error_regex=""):
    # [FUNCTION DOC]
    # HELPER: runs the quantized convolution with one deliberately malformed argument and asserts
    # that an error matching a pattern is raised.
    # Inputs (all optional): tin and tfilter (quantized tensors; default to ones of shape
    # 1x2x3x3), min_input, max_input, min_filter, max_filter (range tensors; default to scalar
    # zero floats) and error_regex (pattern the error text must match; default matches anything).
    # Uses unit strides, SAME padding and 32-bit quantized output. Expects either a value error or
    # an invalid-argument error whose message matches the pattern.
    strides = [1, 1, 1, 1]
    padding = "SAME"
    if tin is None:
      tin = math_ops.cast(
          constant_op.constant(1, shape=[1, 2, 3, 3]), dtype=dtypes.quint8)

    if tfilter is None:
      tfilter = math_ops.cast(
          constant_op.constant(1, shape=[1, 2, 3, 3]), dtype=dtypes.quint8)

    if min_input is None:
      min_input = constant_op.constant(0, shape=[], dtype=dtypes.float32)

    if max_input is None:
      max_input = constant_op.constant(0, shape=[], dtype=dtypes.float32)

    if min_filter is None:
      min_filter = constant_op.constant(0, shape=[], dtype=dtypes.float32)

    if max_filter is None:
      max_filter = constant_op.constant(0, shape=[], dtype=dtypes.float32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                error_regex):
      self.evaluate(
          nn_ops.quantized_conv2d(
              tin,
              tfilter,
              out_type=dtypes.qint32,
              strides=strides,
              padding=padding,
              min_input=min_input,
              max_input=max_input,
              min_filter=min_filter,
              max_filter=max_filter))

  def testBadInputSizes(self):
    # [FUNCTION DOC]
    # Scenario: six invalid calls, each changing one argument. A rank-2 input and a rank-2 filter
    # must raise errors containing "must be rank 4"; a rank-1 minimum or maximum for the input or
    # the filter must raise errors containing "must be rank 0".
    self._testBadInputSize(
        tin=math_ops.cast(
            constant_op.constant(1, shape=[1, 2]), dtype=dtypes.quint8),
        error_regex="must be rank 4")
    self._testBadInputSize(
        tfilter=math_ops.cast(
            constant_op.constant(1, shape=[1, 2]), dtype=dtypes.quint8),
        error_regex="must be rank 4")
    self._testBadInputSize(
        min_input=constant_op.constant(0, shape=[1], dtype=dtypes.float32),
        error_regex="must be rank 0")
    self._testBadInputSize(
        max_input=constant_op.constant(0, shape=[1], dtype=dtypes.float32),
        error_regex="must be rank 0")
    self._testBadInputSize(
        min_filter=constant_op.constant(0, shape=[1], dtype=dtypes.float32),
        error_regex="must be rank 0")
    self._testBadInputSize(
        max_filter=constant_op.constant(0, shape=[1], dtype=dtypes.float32),
        error_regex="must be rank 0")

if __name__ == "__main__":
  test.main()
