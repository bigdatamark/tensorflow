# [MODULE DOC]
# Functional tests for depthwise 2D convolution on the XLA test device. Compares the native
# depthwise convolution op (NHWC and NCHW, with and without dilation) against a CPU reference,
# checks a hand-computed example, and compares input-gradient and filter-gradient ops between XLA
# and CPU. Provides module-level helpers that generate lists of shape/stride/padding
# configurations and a slicing-based reference implementation. No import-time side effects other
# than importing the nn gradient module; run as a test via the test main.
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
"""Functional tests for depthwise convolutional operations."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


# Reference implementation of depthwise_conv2d
def ReferenceDepthwiseConv2D(input_tensor, filter_tensor, strides, padding,
                             data_format=None):
  # [FUNCTION DOC]
  # HELPER: CPU reference implementation used by the depthwise convolution tests.
  # Computes depthwise 2D convolution by slicing each input channel (along the channel axis
  # according to data_format, NCHW or otherwise channels-last), convolving it with the matching
  # filter slice (filter layout is height, width, input channels, depth multiplier) using regular
  # 2D convolution, and concatenating the results along the channel axis. Parameters: input_tensor
  # and filter_tensor (tensors), strides (four values), padding (SAME or VALID), data_format
  # (optional, treated as channels-last unless it is NCHW). Returns the concatenated tensor.
  # Adds one convolution op per channel to the graph.
  # Reference implementation of depthwise convolution that uses regular
  # convolution.
  convs = []
  in_channels = filter_tensor.shape[2]
  # Use a custom implementation of depthwise conv2d using slicing.
  for channel in range(in_channels):
    # Slice the input along channel
    if data_format == "NCHW":
      input_slice = input_tensor[:, channel:channel+1, :, :]
    else:
      input_slice = input_tensor[:, :, :, channel:channel+1]

    # Slice the filters.  Filters are  H, W, InC, DepthMultiplier
    filter_slice = filter_tensor[:, :, channel:channel+1, :]
    # Do conv
    convs.append(nn_ops.conv2d(input_slice, filter_slice,
                               strides, padding,
                               data_format=data_format,
                               name="depthwise_slice_%d" % channel))

  # Concat along dimension.
  if data_format == "NCHW":
    return array_ops.concat(convs, 1)
  else:
    return array_ops.concat(convs, 3)


def ConfigsToTest():
  # [FUNCTION DOC]
  # HELPER: generator of test configurations for the depthwise convolution tests.
  # KNOWN ISSUE: the docstring is accurate about the yielded tuple, but the padding list has 11
  # entries versus 10 for the other lists, so the extra one is silently dropped by pairing.
  # Yields tuples of input size, filter size, expected output size, stride and padding, for ten
  # configurations (one VALID, the rest SAME). Takes no arguments.
  """Iterator for different convolution shapes, strides and paddings.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the depthwise
    convolution parameters.
  """
  input_sizes = [[4, 5, 5, 48], [2, 5, 5, 48], [4, 8, 8, 84], [4, 17, 17, 48],
                 [4, 9, 27, 8], [4, 31, 31, 7], [4, 35, 35, 2],
                 [4, 147, 147, 2], [3, 299, 299, 3], [5, 183, 183, 1]]
  filter_sizes = [[1, 1, 48, 2], [2, 2, 48, 8], [1, 3, 84, 1], [3, 1, 48, 4],
                  [3, 3, 8, 1], [3, 3, 7, 1], [5, 5, 2, 1], [3, 3, 2, 8],
                  [2, 2, 3, 8], [5, 5, 1, 2]]
  out_sizes = [[4, 5, 5, 96], [2, 5, 5, 384], [4, 8, 8, 84], [4, 17, 17, 192],
               [4, 9, 27, 8], [4, 31, 31, 7], [4, 35, 35, 2], [4, 49, 49, 16],
               [3, 150, 150, 24], [5, 92, 92, 2]]
  strides = [1, 1, 1, 1, 1, 1, 1, 3, 2, 2]
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [SAME, SAME, SAME, SAME, SAME, SAME, SAME, VALID, SAME, SAME, SAME]
  for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                           paddings):
    yield i, f, o, s, p


def ConfigsWithDilationsToTest():
  # [FUNCTION DOC]
  # HELPER: generator of dilated test configurations for the depthwise convolution tests.
  # Yields tuples of input size, filter size, expected output size, stride, dilation and padding
  # for five configurations (the last uses VALID padding, others SAME). Takes no arguments.
  """Iterator for different convolution shapes, strides and paddings.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, dilation, padding), the
    depthwise
    convolution parameters.
  """
  input_sizes = [[4, 6, 6, 48], [4, 8, 8, 84], [4, 36, 36, 2], [4, 148, 148, 2],
                 [3, 300, 300, 3]]
  filter_sizes = [[1, 1, 48, 2], [1, 3, 84, 1], [5, 5, 2, 1], [4, 4, 2, 8],
                  [2, 2, 3, 8]]
  out_sizes = [[4, 6, 6, 96], [4, 8, 8, 84], [4, 36, 36, 2], [4, 74, 74, 16],
               [3, 296, 296, 24]]
  strides = [1, 1, 2, 2, 1]
  dilations = [2, 2, 4, 2, 4]
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [SAME, SAME, SAME, SAME, VALID]
  for i, f, o, s, d, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                              dilations, paddings):
    yield i, f, o, s, d, p


def CheckGradConfigsToTest():
  # [FUNCTION DOC]
  # HELPER: generator of small configurations intended for gradient checking.
  # KNOWN ISSUE: this generator is never used in this file (dead code).
  # Yields tuples of input size, filter size, expected output size, stride and padding for five
  # small configurations. Takes no arguments.
  """Iterator for different convolution shapes, strides and paddings.

  compute_gradient_error() is very expensive. So the configs should be
  relatively small.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the depthwise
    convolution parameters.
  """
  input_sizes = [[2, 5, 8, 1], [4, 5, 5, 1], [2, 4, 4, 2], [1, 15, 15, 2],
                 [2, 15, 16, 1]]
  filter_sizes = [[4, 4, 1, 2], [2, 2, 1, 2], [3, 1, 2, 2], [1, 3, 2, 1],
                  [3, 3, 1, 2]]
  out_sizes = [[2, 5, 8, 2], [4, 2, 2, 2], [2, 4, 4, 4], [1, 15, 15, 2],
               [2, 5, 5, 2]]
  strides = [1, 2, 1, 1, 3]
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [SAME, VALID, SAME, SAME, VALID]
  for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                           paddings):
    yield i, f, o, s, p


class DepthwiseConv2DTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for depthwise 2D convolution and its input and filter gradients on the XLA test
  # device. Each comparison test evaluates the XLA op and a CPU equivalent on identical data
  # and checks closeness within a tolerance; float32 only for most comparisons.

  # This is testing that depthwise_conv2d and depthwise_conv2d_native
  # produce the same results.  It also tests that NCHW and NWHC
  # formats agree, by comparing the depthwise_conv2d_native with
  # 'NCHW' format (with transposition) matches the 'NHWC' format using
  # the higher level interface.
  def _VerifyValues(self,
                    tensor_in_sizes,
                    filter_in_sizes,
                    stride,
                    padding,
                    data_type,
                    data_format="NHWC"):
    # [FUNCTION DOC]
    # HELPER: shared check comparing native depthwise convolution on the XLA device to the CPU
    # reference implementation.
    # Fills input and filter with values incrementing from 1, runs the native op (transposing to
    # and from channel-first layout when data_format is NCHW), runs the reference on CPU, prints
    # the maximum absolute difference, and asserts closeness with relative tolerance 1e-4 for
    # 32-bit floats or 1e-8 for 64-bit floats. Any other data type fails an assertion.
    # Parameters: sizes (lists), stride (integer), padding, data_type (numpy float type),
    # data_format (optional, NHWC default). Returns nothing.
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [filter_rows, filter_cols, input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      data_type: The data type to use.
      data_format: The data_format of the input. "NHWC" or "NCHW".
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input and filter tensor with numbers incrementing from 1.
    x1 = np.array([f * 1.0 for f in range(1, total_size_1 + 1)],
                  dtype=data_type).reshape(tensor_in_sizes)
    x2 = np.array([f * 1.0 for f in range(1, total_size_2 + 1)],
                  dtype=data_type).reshape(filter_in_sizes)
    with self.session() as sess:
      if data_type == np.float32:
        tolerance = 1e-4
      else:
        self.assertEqual(data_type, np.float64)
        tolerance = 1e-8

      t1 = array_ops.placeholder(shape=tensor_in_sizes, dtype=data_type)
      t2 = array_ops.placeholder(shape=filter_in_sizes, dtype=data_type)

      native_t1 = t1
      strides = [1, stride, stride, 1]
      if data_format == "NCHW":
        # Transpose from NWHC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        native_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
        strides = [1, 1, stride, stride]

      with self.test_scope():
        conv_native = nn_ops.depthwise_conv2d_native(
            native_t1,
            t2,
            strides=strides,
            data_format=data_format,
            padding=padding)

      if data_format == "NCHW":
        # Transpose back from NCHW to NHWC
        conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])

      with ops.device("CPU"):
        conv_interface = ReferenceDepthwiseConv2D(
            t1, t2, strides=[1, stride, stride, 1], padding=padding)

      native_result = sess.run(conv_native, {t1: x1, t2: x2})
      interface_result = sess.run(conv_interface, {t1: x1, t2: x2})

    print("data_type:", data_type, "max diff = ",
          np.amax(np.absolute(native_result - interface_result)))
    self.assertAllClose(
        np.ravel(native_result), np.ravel(interface_result), rtol=tolerance)

  @test_util.run_without_tensor_float_32(
      "DepthwiseConv2D may use TF32 when available.")
  def testDepthwiseConv2D(self):
    # [FUNCTION DOC]
    # Runs the native-versus-reference comparison in channels-last layout for every standard
    # configuration, only for 32-bit floats (the reference supports only that type).
    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2D,", index, "th config:", input_size, "*",
            filter_size, "stride:", stride, "padding:", padding)
      for data_type in self.float_types:
        # TODO(phawkins): the reference implementation only supports float32.
        if data_type == np.float32:
          self._VerifyValues(
              input_size, filter_size, stride, padding, data_type)

  @test_util.run_without_tensor_float_32(
      "DepthwiseConv2D may use TF32 when available.")
  def testDepthwiseConv2DFormat(self):
    # [FUNCTION DOC]
    # Same as the basic comparison test but feeds the native op channel-first (NCHW) data,
    # for every standard configuration and 32-bit floats only.
    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2DFormat,", index, "th config:", input_size,
            "*", filter_size, "stride:", stride, "padding:", padding)
      for data_type in self.float_types:
        # TODO(phawkins): the reference implementation only supports float32.
        if data_type == np.float32:
          self._VerifyValues(
              input_size,
              filter_size,
              stride,
              padding,
              data_type,
              data_format="NCHW")

# This is testing against hand calculated results.

  def _VerifyHandValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                        expected):
    # [FUNCTION DOC]
    # HELPER: checks the native depthwise convolution on the XLA device against hand-computed
    # values. Fills input and filter with 32-bit floats incrementing from 1, runs the op, prints
    # the result, asserts the flattened output is within 1e-4 of expected, and that the output
    # shape matches the op's static shape. Returns nothing.
    """Verifies the output values of the depthwise convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [filter_rows, filter_cols, input_depth, depth_multiplier].
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
    x1 = np.array([f * 1.0 for f in range(1, total_size_1 + 1)],
                  dtype=np.float32).reshape(tensor_in_sizes)
    x2 = np.array([f * 1.0 for f in range(1, total_size_2 + 1)],
                  dtype=np.float32).reshape(filter_in_sizes)
    with self.session() as sess:
      t1 = array_ops.placeholder(shape=tensor_in_sizes, dtype=np.float32)
      t2 = array_ops.placeholder(shape=filter_in_sizes, dtype=np.float32)
      with self.test_scope():
        conv = nn_ops.depthwise_conv2d_native(
            t1, t2, strides=[1, stride, stride, 1], padding=padding)
      value = sess.run(conv, {t1: x1, t2: x2})
    print("value = ", value)
    self.assertArrayNear(expected, np.ravel(value), 1e-4)
    self.assertShapeEqual(value, conv)

  def testConv2D2x2Filter(self):
    # [FUNCTION DOC]
    # Hand-computed example: input 1x2x3x2, filter 2x2x2x2, stride 1, VALID padding; the expected
    # eight outputs are worked out in the comments below. Delegates to the hand-value check.
    # The inputs look like this (it's a 3 x 2 matrix, each of depth 2):
    #
    # [ (1.0, 2.0), (3.0,  4.0), ( 5.0,  6.0) ]
    # [ (7.0, 8.0), (9.0, 10.0), (11.0, 12.0) ]
    #  We can view this as two inputs
    #
    #  input depth 0:
    #
    #  [ 1.0,  3.0,  5.0 ]
    #  [ 7.0,  9.0, 11.0 ]
    #
    #  input depth 1:
    #
    #  [ 2.0,  4.0,  6.0 ]
    #  [ 8.0, 10.0, 12.0 ]
    #
    # The filter looks like this (it has two 2 x 2 patches, each generating 2
    # depths):
    #
    #  filter #0:
    #
    #  [ (1.0,  3.0), ( 5.0,  7.0)]
    #  [ (9.0, 11.0), (13.0, 15.0)]
    #
    #  filter #1:
    #
    #  [ ( 2.0,  4.0), ( 6.0,  8.0)]
    #  [ (10.0, 12.0), (14.0, 16.0)]
    #
    # So the outputs are:
    #
    # (position 0, 0: in_depth 0, output_depth 0 -- using filter #0)
    #  1.0 * 1.0 + 7.0 * 9.0 + 3.0 * 5.0 + 9.0 * 13.0 = 196
    # (position 0, 0: in_depth 0, output_depth 1 -- using filter #1)
    #  1.0 * 2.0 + 7.0 * 10.0 + 3.0 * 6.0 + 9.0 * 14.0 = 216
    # (position 0, 0: in_depth 1, output_depth 2 -- using filter #0)
    #  2.0 * 3.0 + 8.0 * 11.0 + 4.0 * 7.0 + 10.0 * 15.0 = 272
    # (position 0, 0: in_depth 1, output_depth 3 -- using filter #1)
    #  2.0 * 4.0 + 8.0 * 12.0 + 4.0 * 8.0 + 10.0 * 16.0 = 296
    #
    # (position 1, 0: in_depth 0, output_depth 0 -- using filter #0)
    #  3.0 * 1.0 + 9.0 * 9.0 + 5.0 * 5.0 + 11.0 * 13.0 = 252
    # (position 1, 0: in_depth 0, output_depth 1 -- using filter #1)
    #  3.0 * 2.0 + 9.0 * 10.0 + 5.0 * 6.0 + 11.0 * 14.0 = 280
    # (position 1, 0: in_depth 1, output_depth 2 -- using filter #0)
    #  4.0 * 3.0 + 10.0 * 11.0 + 6.0 * 7.0 + 12.0 * 15.0 = 344
    # (position 1, 0: in_depth 1, output_depth 3 -- using filter #1)
    #  4.0 * 4.0 + 10.0 * 12.0 + 6.0 * 8.0 + 12.0 * 16.0 = 376
    expected_output = [196, 216, 272, 296, 252, 280, 344, 376]
    self._VerifyHandValues(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        stride=1,
        padding="VALID",
        expected=expected_output)

  # This is testing that depthwise_conv2d with dilation produces
  # the same results between CPU and TPU. It also tests that NCHW
  # and NWHC formats agree.
  def _VerifyValuesWithDilation(self,
                                tensor_in_sizes,
                                filter_in_sizes,
                                stride,
                                dilation,
                                padding,
                                data_type,
                                data_format="NHWC"):
    # [FUNCTION DOC]
    # HELPER: like the basic native-versus-reference check but with dilation. Uses the high-level
    # depthwise convolution with a dilation rate for both the XLA device (channel-first when
    # requested, with transposes) and the CPU (channels-last only). Tolerance is 1e-2 for 32-bit
    # floats (loose, noted as a TPU issue) or 1e-8 for 64-bit floats; other types fail an
    # assertion. Prints the maximum absolute difference. Returns nothing.
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [filter_rows, filter_cols,
        input_depth, depth_multiplier].
      stride: Stride.
      dilation: Dilation.
      padding: Padding type.
      data_type: The data type to use.
      data_format: The data_format of the input. "NHWC" or "NCHW".
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input and filter tensor with numbers incrementing from 1.
    x1 = np.array([f * 1.0 for f in range(1, total_size_1 + 1)],
                  dtype=data_type).reshape(tensor_in_sizes)
    x2 = np.array([f * 1.0 for f in range(1, total_size_2 + 1)],
                  dtype=data_type).reshape(filter_in_sizes)
    with self.session() as sess:
      if data_type == np.float32:
        # TODO(b/64210055): Tolerance for TPU is high.
        tolerance = 1e-2
      else:
        self.assertEqual(data_type, np.float64)
        tolerance = 1e-8

      t1 = array_ops.placeholder(shape=tensor_in_sizes, dtype=data_type)
      t2 = array_ops.placeholder(shape=filter_in_sizes, dtype=data_type)

      native_t1 = t1
      strides = [1, stride, stride, 1]
      dilations = [dilation, dilation]
      if data_format == "NCHW":
        # Transpose from NWHC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        native_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
        strides = [1, 1, stride, stride]

      with self.test_scope():
        conv_native = nn_impl.depthwise_conv2d(
            native_t1,
            t2,
            strides=strides,
            rate=dilations,
            data_format=data_format,
            padding=padding)

      if data_format == "NCHW":
        # Transpose back from NCHW to NHWC
        conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])

      with ops.device("CPU"):
        # CPU only support NHWC format
        strides = [1, stride, stride, 1]
        conv_interface = nn_impl.depthwise_conv2d(
            t1, t2, strides=strides, rate=dilations, padding=padding)

      native_result = sess.run(conv_native, {t1: x1, t2: x2})
      interface_result = sess.run(conv_interface, {t1: x1, t2: x2})

    print("data_type:", data_type, "max diff = ",
          np.amax(np.absolute(native_result - interface_result)))
    self.assertAllClose(
        np.ravel(native_result), np.ravel(interface_result), rtol=tolerance)

  def testDilationDepthwiseConv2DWith(self):
    # [FUNCTION DOC]
    # Runs the dilated comparison in channels-last layout for every dilated configuration, with
    # 32-bit floats only.
    for index, (input_size, filter_size, _, stride, dilation,
                padding) in enumerate(ConfigsWithDilationsToTest()):
      print("Testing DilationDepthwiseConv2D,", index, "th config:", input_size,
            "*", filter_size, "stride:", stride, "dilation: ", dilation,
            "padding:", padding)
      for data_type in self.float_types:
        # TODO(phawkins): the reference implementation only supports float32.
        if data_type == np.float32:
          self._VerifyValuesWithDilation(input_size, filter_size, stride,
                                         dilation, padding, data_type)

  def testDilationDepthwiseConv2DWithFormat(self):
    # [FUNCTION DOC]
    # Runs the dilated comparison with channel-first (NCHW) layout on the XLA device for every
    # dilated configuration, with 32-bit floats only.
    for index, (input_size, filter_size, _, stride, dilation,
                padding) in enumerate(ConfigsWithDilationsToTest()):
      print("Testing DilationDepthwiseConv2DFormat,", index, "th config:",
            input_size, "*", filter_size, "stride:", stride, "dilation:",
            dilation, "padding:", padding)
      for data_type in self.float_types:
        # TODO(phawkins): the reference implementation only supports float32.
        if data_type == np.float32:
          self._VerifyValuesWithDilation(
              input_size,
              filter_size,
              stride,
              dilation,
              padding,
              data_type,
              data_format="NCHW")

  def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes,
                            stride, padding):
    # [FUNCTION DOC]
    # HELPER: compares the depthwise convolution input-gradient op on the XLA device against the
    # same op on the default (CPU) device using random 32-bit float filter and output-gradient
    # data (not seeded). Asserts closeness with relative and absolute tolerance 1e-3. Returns
    # nothing.
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_xla):
      # [FUNCTION DOC]
      # HELPER: builds and evaluates the input-gradient op, under the XLA test scope when use_xla
      # is true, and returns the resulting array after asserting its shape matches the op.
      with self.session():
        t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
        t1 = array_ops.placeholder(np.float32, shape=filter_sizes)
        t2 = array_ops.placeholder(np.float32, shape=output_sizes)
        if use_xla:
          with self.test_scope():
            backprop = nn_ops.depthwise_conv2d_native_backprop_input(
                t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        else:
          backprop = nn_ops.depthwise_conv2d_native_backprop_input(
              t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)

        ret = backprop.eval({t1: x1, t2: x2})
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_xla=True)
    cpu_value = _GetVal(use_xla=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-3, atol=1e-3)

  def testDepthwiseConv2DInputGradCompare(self):
    # [FUNCTION DOC]
    # Runs the input-gradient XLA-versus-CPU comparison for every standard configuration.
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2DInputGradCompare,", index, "th config:",
            input_size, "*", filter_size, "stride:", stride, "padding:",
            padding)
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding)

  def _CompareBackpropFilter(self,
                             input_sizes,
                             filter_sizes,
                             output_sizes,
                             stride,
                             padding,
                             data_format="NHWC"):
    # [FUNCTION DOC]
    # HELPER: compares the depthwise convolution filter-gradient op on the XLA device (optionally
    # with channel-first data via transposes) against the CPU channels-last version, using random
    # 32-bit float input and output-gradient data (not seeded). Asserts closeness with tolerance
    # 1e-4. Parameters: sizes (lists), stride, padding, data_format (optional, NHWC default).
    x0 = np.random.rand(*input_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_xla):
      # [FUNCTION DOC]
      # HELPER: builds and evaluates the filter-gradient op on the XLA test scope (use_xla true) or
      # on the default device, and returns the array after asserting its shape matches the op.
      with self.session():
        t0 = array_ops.placeholder(np.float32, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = array_ops.placeholder(np.float32, shape=output_sizes)
        native_t0 = t0
        native_t2 = t2
        strides = [1, stride, stride, 1]

        if use_xla:
          if data_format == "NCHW":
            # Transpose from NWHC input to NCHW
            # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
            native_t0 = array_ops.transpose(t0, [0, 3, 1, 2])
            native_t2 = array_ops.transpose(t2, [0, 3, 1, 2])
            strides = [1, 1, stride, stride]
          with self.test_scope():
            backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
                native_t0,
                t1,
                native_t2,
                strides=strides,
                padding=padding,
                data_format=data_format)
        else:
          # For CPU, the format NCHW is not supported. Therefore we always use
          # NHWC here.
          backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
              native_t0, t1, native_t2, strides=strides, padding=padding)
        ret = backprop.eval({t0: x0, t2: x2})
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_xla=True)
    cpu_value = _GetVal(use_xla=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  @test_util.run_without_tensor_float_32(
      "DepthwiseConv2DFilterGrad may use TF32 when available.")
  def testDepthwiseConv2DFilterGradCompare(self):
    # [FUNCTION DOC]
    # Runs the filter-gradient XLA-versus-CPU comparison (channels-last) for every standard
    # configuration, with reduced-precision matmul modes disabled by the decorator.
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2DFilterGradCompare,", index, "th config:",
            input_size, "*", filter_size, "producing output", output_size,
            "stride:", stride, "padding:", padding)
      self._CompareBackpropFilter(input_size, filter_size, output_size,
                                  stride, padding)

  @test_util.run_without_tensor_float_32(
      "DepthwiseConv2DFilterGrad may use TF32 when available.")
  def testDepthwiseConv2DFilterGradFormatNCHWCompare(self):
    # [FUNCTION DOC]
    # Same as the filter-gradient comparison but with channel-first (NCHW) data on the XLA side.
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2DFilterGradFormatNCHWCompare,", index,
            "th config:", input_size, "*", filter_size, "producing output",
            output_size, "stride:", stride, "padding:", padding)
      self._CompareBackpropFilter(
          input_size,
          filter_size,
          output_size,
          stride,
          padding,
          data_format="NCHW")

  def _CompareBackpropInputWithDilation(self, input_sizes, filter_sizes,
                                        output_sizes, stride, dilation,
                                        padding):
    # [FUNCTION DOC]
    # HELPER: compares the dilated depthwise input-gradient op on XLA against a CPU emulation
    # (the CPU op lacks dilation, so the output gradient is rearranged with space-to-batch, the
    # op run with adjusted input sizes, and the result rearranged back with batch-to-space).
    # Random 32-bit float data, not seeded. Asserts closeness with relative tolerance 1e-2 and
    # absolute 1e-3 (loose, noted as a TPU issue). Only valid for stride 1.
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_xla):
      # [FUNCTION DOC]
      # HELPER: evaluates the dilated input-gradient op directly on XLA (use_xla true) or via the
      # space-to-batch emulation on the default device; returns the array after a shape assertion.
      with self.session():
        t1 = array_ops.placeholder(np.float32, shape=filter_sizes)
        t2 = array_ops.placeholder(np.float32, shape=output_sizes)
        if use_xla:
          with self.test_scope():
            t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
            backprop = nn_ops.depthwise_conv2d_native_backprop_input(
                t0,
                t1,
                t2,
                strides=[1, stride, stride, 1],
                dilations=[1, dilation, dilation, 1],
                padding=padding)
        else:
          # TODO(wangtao): figure out gradient with stride > 1.
          # depthwise_conv2d_native_backprop_input on CPU doesn't support
          # dilation.
          t3 = array_ops.space_to_batch(
              t2, block_size=dilation, paddings=[[0, 0], [0, 0]])
          input_sizes_transform = [
              input_sizes[0] * dilation * dilation, input_sizes[1] // dilation,
              input_sizes[2] // dilation, input_sizes[3]
          ]
          t0 = constant_op.constant(
              input_sizes_transform, shape=[len(input_sizes)])
          backprop_naive = nn_ops.depthwise_conv2d_native_backprop_input(
              t0, t1, t3, strides=[1, stride, stride, 1], padding=padding)
          backprop = array_ops.batch_to_space(
              backprop_naive, [[0, 0], [0, 0]], block_size=dilation)

        ret = backprop.eval({t1: x1, t2: x2})
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_xla=True)
    cpu_value = _GetVal(use_xla=False)

    # TODO (b/64210055): Tolerance for TPU is high.
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-2, atol=1e-3)

  def testDilationDepthwiseConv2DInputGradWithCompare(self):
    # [FUNCTION DOC]
    # Runs the dilated input-gradient comparison for each dilated configuration whose stride is 1
    # (larger strides are skipped because the CPU emulation does not support them).
    for index, (input_size, filter_size, output_size, stride, dilation,
                padding) in enumerate(ConfigsWithDilationsToTest()):
      print("Testing DilationDepthwiseConv2DInputGradWithDilationCompare,",
            index, "th config:", input_size, "*", filter_size, "stride:",
            stride, "dilation:", dilation, "padding:", padding)
      # TODO(wangtao): implement CPU grad computation with stride > 1.
      if stride == 1:
        self._CompareBackpropInputWithDilation(input_size, filter_size,
                                               output_size, stride, dilation,
                                               padding)

  def _CompareBackpropFilterWithDilation(self,
                                         input_sizes,
                                         filter_sizes,
                                         output_sizes,
                                         stride,
                                         dilation,
                                         padding,
                                         data_format="NHWC"):
    # [FUNCTION DOC]
    # HELPER: compares the dilated depthwise filter-gradient op on XLA (optionally channel-first
    # via transposes) against a CPU emulation that applies space-to-batch to the input and the
    # output gradient. Random 32-bit float data, not seeded. Asserts closeness with relative
    # tolerance 1e-3 and absolute 1e-4. Only meaningful for stride 1.
    x0 = np.random.rand(*input_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_xla):
      # [FUNCTION DOC]
      # HELPER: evaluates the dilated filter-gradient op on XLA (use_xla true) or via the
      # space-to-batch emulation on the default device; returns the array after a shape assertion.
      with self.session():
        t0 = array_ops.placeholder(np.float32, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = array_ops.placeholder(np.float32, shape=output_sizes)
        native_t0 = t0
        native_t2 = t2
        strides = [1, stride, stride, 1]
        dilations = [1, dilation, dilation, 1]

        if use_xla:
          if data_format == "NCHW":
            # Transpose from NWHC input to NCHW
            # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
            native_t0 = array_ops.transpose(t0, [0, 3, 1, 2])
            native_t2 = array_ops.transpose(t2, [0, 3, 1, 2])
            strides = [1, 1, stride, stride]
            dilations = [1, 1, dilation, dilation]
          with self.test_scope():
            backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
                native_t0,
                t1,
                native_t2,
                strides=strides,
                padding=padding,
                dilations=dilations,
                data_format=data_format)
        else:
          # For CPU, the format NCHW is not supported. Therefore we always use
          # NHWC here.
          # depthwise_conv2d_native_backprop_filter on CPU doesn't support
          # dilation.
          native_t3 = array_ops.space_to_batch(
              native_t2, block_size=dilation, paddings=[[0, 0], [0, 0]])
          native_t0_transform = array_ops.space_to_batch(
              native_t0, block_size=dilation, paddings=[[0, 0], [0, 0]])
          backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
              native_t0_transform,
              t1,
              native_t3,
              strides=strides,
              padding=padding)
        ret = backprop.eval({t0: x0, t2: x2})
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_xla=True)
    cpu_value = _GetVal(use_xla=False)
    # TODO(b/64210055): Tolerance for TPU is high.
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-3, atol=1e-4)

  def testDilationDepthwiseConv2DFilterGradCompare(self):
    # [FUNCTION DOC]
    # Runs the dilated filter-gradient comparison for each dilated configuration whose stride is 1.
    for index, (input_size, filter_size, output_size, stride, dilation,
                padding) in enumerate(ConfigsWithDilationsToTest()):
      print("Testing DilationDepthwiseConv2DFilterGradCompare,", index,
            "th config:", input_size, "*", filter_size, "producing output",
            output_size, "stride:", stride, "dilation:", dilation, "padding:",
            padding)
      if stride == 1:
        # TODO(wangtao): implement CPU grad computation with stride > 1.
        self._CompareBackpropFilterWithDilation(input_size, filter_size,
                                                output_size, stride, dilation,
                                                padding)

if __name__ == "__main__":
  test.main()
