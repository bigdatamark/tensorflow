# [MODULE DOC]
# Functional tests for the three-dimensional convolution operations (Conv3D and the generic Conv
# op with three spatial dimensions) and their gradients. Exposes module-level helpers (a function
# that lists data types to test, a function that lists data-format/device configurations), two
# lists of named test parameters (full and dilated) and one test class that compares results with
# hand-computed expected values, with the generic convolution routine, and with numerical
# gradients. Covers NDHWC and NCDHW layouts, CPU and GPU, several dtypes, strides, padding modes,
# dilations and expanded batch dimensions. No import-time side effects apart from imports and
# building the parameter lists. The file runs the TensorFlow test main when executed directly.
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
"""Functional tests for 3d convolutional operations."""

import math

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.eager import context


def DtypesToTest(use_gpu):
  # [FUNCTION DOC]
  # HELPER: small shared utility choosing the data types used by the convolution tests.
  # Input: use_gpu, a boolean saying whether the test runs on a GPU. Returns a list of data types.
  # Double precision is included unless TensorFlow is built for ROCm. On GPU, float32 is
  # returned (plus float16 when the GPU supports half precision matmul and convolution; float32
  # deliberately precedes float16 because its gradients are the reference for half). On CPU,
  # float32, float16 and bfloat16 are returned. No side effects.
  # double datatype is currently not supported for convolution ops
  # on the ROCm platform
  optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
  if use_gpu:
    if not test_util.GpuSupportsHalfMatMulAndConv():
      return optional_float64 + [dtypes.float32]
    else:
      # It is important that float32 comes before float16 here,
      # as we will be using its gradients as reference for fp16 gradients.
      return optional_float64 + [dtypes.float32, dtypes.float16]
  else:
    return optional_float64 + [dtypes.float32, dtypes.float16, dtypes.bfloat16]


TEST_PARAMS = [
    ("Conv3D_NDHWC_float_cpu", "NDHWC", dtypes.float32, False, "Conv3D"),
    ("Conv3D_NDHWC_half_cpu", "NDHWC", dtypes.float16, False, "Conv3D"),
    ("Conv3D_NDHWC_double_cpu", "NDHWC", dtypes.float64, False, "Conv3D"),
    ("Conv3D_NDHWC_bfloat16_cpu", "NDHWC", dtypes.bfloat16, False, "Conv3D"),
    ("Conv3D_NDHWC_float_gpu", "NDHWC", dtypes.float32, True, "Conv3D"),
    ("Conv3D_NDHWC_half_gpu", "NDHWC", dtypes.float16, True, "Conv3D"),
    ("Conv3D_NDHWC_double_gpu", "NDHWC", dtypes.float64, True, "Conv3D"),
    ("Conv3D_NDHWC_bfloat16_gpu", "NDHWC", dtypes.bfloat16, True, "Conv3D"),
    ("Conv3D_NCDHW_float_gpu", "NCDHW", dtypes.float32, True, "Conv3D"),
    ("Conv3D_NCDHW_half_gpu", "NCDHW", dtypes.float16, True, "Conv3D"),
    ("Conv3D_NCDHW_double_gpu", "NCDHW", dtypes.float64, True, "Conv3D"),
    ("Conv3D_NCDHW_bfloat16_gpu", "NCDHW", dtypes.bfloat16, True, "Conv3D"),
    ("Conv_NDHWC_float_cpu", "NDHWC", dtypes.float32, False, "Conv"),
    ("Conv_NDHWC_half_cpu", "NDHWC", dtypes.float16, False, "Conv"),
    ("Conv_NDHWC_double_cpu", "NDHWC", dtypes.float64, False, "Conv"),
    ("Conv_NDHWC_bfloat16_cpu", "NDHWC", dtypes.bfloat16, False, "Conv"),
    ("Conv_NDHWC_float_gpu", "NDHWC", dtypes.float32, True, "Conv"),
    ("Conv_NDHWC_half_gpu", "NDHWC", dtypes.float16, True, "Conv"),
    ("Conv_NDHWC_double_gpu", "NDHWC", dtypes.float64, True, "Conv"),
    ("Conv_NDHWC_bfloat16_gpu", "NDHWC", dtypes.bfloat16, True, "Conv"),
    ("Conv_NCDHW_float_gpu", "NCDHW", dtypes.float32, True, "Conv"),
    ("Conv_NCDHW_half_gpu", "NCDHW", dtypes.float16, True, "Conv"),
    ("Conv_NCDHW_double_gpu", "NCDHW", dtypes.float64, True, "Conv"),
    ("Conv_NCDHW_bfloat16_gpu", "NCDHW", dtypes.bfloat16, True, "Conv"),
]

DILATED_PARAMS = [
    ("Conv3D_NDHWC_cpu", "NDHWC", False, "Conv3D"),
    ("Conv3D_NDHWC_gpu", "NDHWC", True, "Conv3D"),
    ("Conv3D_NCDHW_gpu", "NCDHW", True, "Conv3D"),
    ("Conv_NDHWC_cpu", "NDHWC", False, "Conv"),
    ("Conv_NDHWC_gpu", "NDHWC", True, "Conv"),
    ("Conv_NCDHW_gpu", "NCDHW", True, "Conv"),
]


def GetTestConfigs():
  # [FUNCTION DOC]
  # HELPER: small shared utility listing the layout/device combinations to test.
  # Takes no inputs. Returns a list of pairs of data format name and a boolean use-GPU flag.
  # Always includes channels-last layout on both the non-GPU and GPU settings; adds the
  # channels-first layout (GPU only) when a CUDA GPU is available. Queries device availability.
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NDHWC", False), ("NDHWC", True)]
  if test.is_gpu_available(cuda_only=True):
    # "NCDHW" format is only supported on CUDA.
    test_configs += [("NCDHW", True)]
  return test_configs


@test_util.run_all_without_tensor_float_32(
    "Tests Conv3d, which in some cases is implemented with a matmul. With "
    "TensorFloat-32, tests fail in some of those cases (and as of August 13 "
    "2020, only those cases)")
class Conv3DTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Test case covering 3D convolution forward values, strides, padding, dilation, expanded batch
  # dimensions and gradients with respect to input and filter. Combines the parameterized test
  # case with the TensorFlow test case, so named parameter sets (TEST_PARAMS, DILATED_PARAMS)
  # generate one test per layout, dtype, device and op variant. The whole class is decorated to
  # run with TensorFloat-32 disabled, since some cases use a matmul that would otherwise fail
  # tolerance checks. Holds no instance state; helper methods build graphs and compare outputs.

  def _SetupValuesForDevice(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      stride,
      padding,
      data_format,
      dtype,
      use_gpu,
      op_name,
  ):
    # [FUNCTION DOC]
    # HELPER: builds the convolution under test for the verification helper.
    # KNOWN ISSUE: the name suggests it only sets up values, but it also builds and returns the
    # convolution result tensor. It has no docstring.
    # Inputs: tensor_in_sizes and filter_in_sizes (lists of dimensions), stride (a single integer
    # or an iterable of three), padding (string), data_format (channels-last or channels-first
    # name), dtype, use_gpu (boolean choosing the session device) and op_name (either the
    # specific 3D op name or the generic op name). Creates input values scaled into 0..1 so half
    # precision does not overflow, converts to channels-first when requested, runs the chosen op
    # and converts the result back to channels-last. Returns the output tensor. Raises a value
    # error for an unknown op name. Opens a cached session as a side effect.
    total_size_tensor = np.prod(tensor_in_sizes)
    total_size_filter = np.prod(filter_in_sizes)

    # Initializes the input tensor with array containing numbers from 0 to 1.

    # We keep the input tensor values fairly small to avoid overflowing float16
    # during the conv3d.
    x1 = [f * 1.0 / total_size_tensor for f in range(1, total_size_tensor + 1)]
    x2 = [f * 1.0 / total_size_filter for f in range(1, total_size_filter + 1)]
    with self.cached_session(use_gpu=use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)

      if isinstance(stride, collections_abc.Iterable):
        strides = [1] + list(stride) + [1]
      else:
        strides = [1, stride, stride, stride, 1]

      if data_format == "NCDHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)

      if op_name == "Conv3D":
        op = nn_ops.conv3d
        conv_format = data_format
      elif op_name == "Conv":
        op = gen_nn_ops.conv
        conv_format = (
            "CHANNELS_LAST" if data_format == "NDHWC" else "CHANNELS_FIRST"
        )
      else:
        raise ValueError("Invalid op name: %s" % op_name)
      conv = op(t1, t2, strides, padding=padding, data_format=conv_format)
      if data_format == "NCDHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv

  def _VerifyValues(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      stride,
      padding,
      expected,
      data_format,
      dtype,
      use_gpu,
      op_name,
  ):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine used by the forward value tests.
    # KNOWN ISSUE: the unused session variable is created but never used, and the results list
    # only ever holds one element, so the loop is redundant.
    # Skips the test when a GPU is requested but unavailable. Otherwise builds the convolution
    # via the setup helper, evaluates it and asserts that the flattened output is close to the
    # expected list of values within a tolerance depending on dtype. Logs expected and actual
    # values at debug level. Returns nothing.
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions [batch, input_x, input_y,
        input_z, input_depth].
      filter_in_sizes: Filter tensor dimensions [kernel_x, kernel_y, kernel_z,
        input_depth, output_depth].
      stride: [x_stride, y_stride, z_stride]
      padding: Padding type.
      expected: Value that the output of computation should match
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_gpu: True if the operations should be run on GPU
      op_name: Name of the op to be tested

    Returns:
      None
    """
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      self.skipTest("GPU not available")
    results = []
    result = self._SetupValuesForDevice(
        tensor_in_sizes,
        filter_in_sizes,
        stride,
        padding,
        data_format,
        dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )
    results.append(result)

    with self.cached_session() as sess:
      values = self.evaluate(results)
      for value in values:
        tf_logging.debug("expected = %s", expected)
        tf_logging.debug("actual = %s", value)
        self.assertAllCloseAccordingToType(expected, value.flatten())

  def _ComputeReferenceDilatedConv(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      stride,
      dilation,
      padding,
      data_format,
      use_gpu,
      op_name,
  ):
    # [FUNCTION DOC]
    # HELPER: computes a reference result and the op result for dilated convolution tests.
    # Inputs: tensor_in_sizes, filter_in_sizes, stride (integer or iterable of three), dilation
    # (list of three rates), padding, data_format, use_gpu, op_name (3D op or generic op).
    # Fills input and filter with incrementing numbers starting at one, computes the expected
    # output with the generic convolution routine and the computed output with the requested op
    # using full-length stride and dilation lists. Converts both back to channels-last if the
    # channels-first layout was used. Returns the pair (expected, computed). Raises a value error
    # for an unknown op name. Opens a cached session.
    total_size_tensor = np.prod(tensor_in_sizes)
    total_size_filter = np.prod(filter_in_sizes)

    # Initializes the input tensor with array containing incrementing

    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_tensor + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_filter + 1)]
    with self.cached_session(use_gpu=use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      if isinstance(stride, collections_abc.Iterable):
        strides = list(stride)
      else:
        strides = [stride, stride, stride]
      if data_format == "NCDHW":
        t1 = test_util.NHWCToNCHW(t1)
        full_strides = [1, 1] + strides
        full_dilation = [1, 1] + dilation
      else:
        full_strides = [1] + strides + [1]
        full_dilation = [1] + dilation + [1]
      expected = nn_ops.convolution(
          t1,
          t2,
          padding=padding,
          strides=strides,
          dilation_rate=dilation,
          data_format=data_format)
      if op_name == "Conv3D":
        op = nn_ops.conv3d
        conv_format = data_format
      elif op_name == "Conv":
        op = gen_nn_ops.conv
        conv_format = (
            "CHANNELS_LAST" if data_format == "NDHWC" else "CHANNELS_FIRST"
        )
      else:
        raise ValueError("Invalid op name: %s" % op_name)

      computed = op(
          t1,
          t2,
          strides=full_strides,
          dilations=full_dilation,
          padding=padding,
          data_format=conv_format,
      )
      if data_format == "NCDHW":
        expected = test_util.NCHWToNHWC(expected)
        computed = test_util.NCHWToNHWC(computed)
    return expected, computed

  def _VerifyDilatedConvValues(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      stride,
      padding,
      dilations,
      data_format,
      use_gpu,
      op_name,
  ):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for dilated convolution tests.
    # KNOWN ISSUE: the unused session variable is created but never used.
    # Skips if a GPU is requested but unavailable. When all dilation rates are one, or the GPU is
    # used (no CPU kernel exists for arbitrary dilation), computes the reference and op results
    # via the reference helper, evaluates both and asserts the flattened values are close
    # (absolute 1e-5, relative 1e-6). Otherwise does nothing. Returns nothing.
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      self.skipTest("GPU not available")
    expected_results = []

    computed_results = []
    default_dilations = (
        dilations[0] == 1 and dilations[1] == 1 and dilations[2] == 1)
    # If any dilation rate is larger than 1, only do test on the GPU
    # because we currently do not have a CPU implementation for arbitrary
    # dilation rates.
    if default_dilations or use_gpu:
      expected, computed = self._ComputeReferenceDilatedConv(
          tensor_in_sizes,
          filter_in_sizes,
          stride,
          dilations,
          padding,
          data_format,
          use_gpu,
          op_name,
      )
      expected_results.append(expected)
      computed_results.append(computed)
      with self.cached_session() as sess:
        expected_values = self.evaluate(expected_results)
        computed_values = self.evaluate(computed_results)
        for e_value, c_value in zip(expected_values, computed_values):
          tf_logging.debug("expected = %s", e_value)
          tf_logging.debug("actual = %s", c_value)
          self.assertAllCloseAccordingToType(
              e_value.flatten(), c_value.flatten(), atol=1e-5, rtol=1e-6
          )

  def _CreateNumpyTensor(self, sizes):
    # [FUNCTION DOC]
    # HELPER: creates test data for the expanded batch tests.
    # Input: sizes, a list of dimensions. Returns a single precision array of that shape filled
    # with incrementing numbers starting at one. No side effects.
    return np.asarray([f * 1.0 for f in range(1,
                                              np.prod(sizes) + 1)],
                      dtype=np.float32).reshape(sizes)

  @test_util.run_in_graph_and_eager_modes
  def testConv3DExpandedBatch(self):
    # [FUNCTION DOC]
    # Test: runs the version-two 3D convolution on an input with one batch dimension and on the
    # same data reshaped to two batch dimensions, using a one-by-one-by-one filter, valid padding.
    # Asserts the output shapes match the input shapes and that both outputs hold equal values.
    # Runs in graph and eager modes.
    tensor_in_sizes_batch = [10, 2, 3, 1, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 1, 3]
    filter_in_sizes = [1, 1, 1, 3, 3]
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)
    conv1 = nn_ops.conv3d_v2(
        x1, filter_in, strides=[1, 1, 1, 1, 1], padding="VALID")
    conv2 = nn_ops.conv3d_v2(
        x2, filter_in, strides=[1, 1, 1, 1, 1], padding="VALID")
    self.assertEqual(conv1.shape, tensor_in_sizes_batch)
    self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
    self.assertAllClose(conv1, self.evaluate(conv2).reshape(conv1.shape))

  @test_util.run_in_graph_and_eager_modes
  def testConvExpandedBatch(self):
    # [FUNCTION DOC]
    # Test: same scenario as the 3D expanded batch test but uses the generic convolution op,
    # passing an explicit batch dimension count of two for the reshaped input. Asserts output
    # shapes equal the input shapes and that both outputs hold equal values. Graph and eager.
    tensor_in_sizes_batch = [10, 2, 3, 1, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 1, 3]
    batch_dims = 2
    filter_in_sizes = [1, 1, 1, 3, 3]
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)
    conv1 = gen_nn_ops.conv(
        x1, filter_in, strides=[1, 1, 1, 1, 1], padding="VALID"
    )
    conv2 = gen_nn_ops.conv(
        x2,
        filter_in,
        strides=[1, 1, 1, 1, 1],
        padding="VALID",
        batch_dims=batch_dims,
    )
    self.assertEqual(conv1.shape, tensor_in_sizes_batch)
    self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
    self.assertAllClose(conv1, self.evaluate(conv2).reshape(conv1.shape))

  @test_util.run_in_graph_and_eager_modes
  def testConvolutionClass3DExpandedBatch(self):
    # [FUNCTION DOC]
    # Test: builds two convolution-class objects for one and two batch dimensions and asserts
    # their reported batch-dimension counts are one and two. Applies both to the matching inputs
    # and asserts output shapes and equal values. Graph and eager modes.
    tensor_in_sizes_batch = [10, 2, 3, 1, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 1, 3]
    filter_in_sizes = [1, 1, 1, 3, 3]
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)
    convolver1 = nn_ops.Convolution(
        input_shape=x1.shape,
        filter_shape=filter_in.shape,
        strides=[1, 1, 1],
        padding="VALID")
    self.assertEqual(convolver1.num_batch_dims, 1)
    convolver2 = nn_ops.Convolution(
        input_shape=x2.shape,
        filter_shape=filter_in.shape,
        strides=[1, 1, 1],
        padding="VALID")
    self.assertEqual(convolver2.num_batch_dims, 2)
    conv1 = convolver1(x1, filter_in)
    conv2 = convolver2(x2, filter_in)
    self.assertEqual(conv1.shape, tensor_in_sizes_batch)
    self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
    self.assertAllClose(conv1, self.evaluate(conv2).reshape(conv1.shape))

  @test_util.run_in_graph_and_eager_modes
  def testConvolutionWith2SpatialDimensionsAndExpandedBatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says two spatial dimensions, but the data and filter have three
    # spatial dimensions (rank five and six inputs), so it really tests 3D.
    # Test: calls the generic convolution function with one and two batch dimensions on the same
    # data, asserts output shapes match the input shapes and values are equal. Graph and eager.
    tensor_in_sizes_batch = [10, 2, 3, 1, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 1, 3]
    filter_in_sizes = [1, 1, 1, 3, 3]
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)
    conv1 = nn_ops.convolution(
        x1, filter_in, strides=[1, 1, 1], padding="VALID")
    conv2 = nn_ops.convolution(
        x2, filter_in, strides=[1, 1, 1], padding="VALID")
    self.assertEqual(conv1.shape, tensor_in_sizes_batch)
    self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
    self.assertAllClose(conv1, self.evaluate(conv2).reshape(conv1.shape))

  @parameterized.named_parameters(*TEST_PARAMS)
  def testConv3D1x1x1Filter(self, data_format, dtype, use_gpu, op_name):
    # [FUNCTION DOC]
    # Parameterized test (layout, dtype, device, op variant). Verifies a one-by-one-by-one filter
    # with stride one and valid padding on three input shapes that differ only in which spatial
    # axis has length two; all three must give the same expected output list, equivalent to a
    # two-dimensional one-by-one convolution.
    expected_output = [
        0.18518519, 0.22222222, 0.25925926, 0.40740741, 0.5, 0.59259259,
        0.62962963, 0.77777778, 0.92592593, 0.85185185, 1.05555556, 1.25925926,
        1.07407407, 1.33333333, 1.59259259, 1.2962963, 1.61111111, 1.92592593
    ]

    # These are equivalent to the Conv2D1x1 case.
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 1, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )
    self._VerifyValues(
        tensor_in_sizes=[1, 1, 2, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

  @parameterized.named_parameters(*DILATED_PARAMS)
  def testConv3D1x1x1Filter2x1x1Dilation(self, data_format, use_gpu, op_name):
    # [FUNCTION DOC]
    # Parameterized test (layout, device, op variant). Only runs its check when a CUDA GPU is
    # available, or when MKL is enabled and the context is not eager; otherwise it silently
    # passes. The check compares the op against the generic convolution routine for a
    # one-by-one-by-one filter, stride one, valid padding and dilation of two along the first
    # spatial axis.
    ctx = context.context()
    is_eager = ctx is not None and ctx.executing_eagerly()
    if test.is_gpu_available(cuda_only=True) or \
      (test_util.IsMklEnabled() and is_eager is False):
      self._VerifyDilatedConvValues(
          tensor_in_sizes=[1, 3, 6, 1, 1],
          filter_in_sizes=[1, 1, 1, 1, 1],
          stride=1,
          padding="VALID",
          dilations=[2, 1, 1],
          data_format=data_format,
          use_gpu=use_gpu,
          op_name=op_name,
      )

  # Expected values computed using scipy's correlate function.
  @parameterized.named_parameters(*TEST_PARAMS)
  def testConv3D2x2x2Filter(self, data_format, dtype, use_gpu, op_name):
    # [FUNCTION DOC]
    # Parameterized test (layout, dtype, device, op variant). Verifies a two-by-two-by-two filter
    # with stride one and valid padding on a small input against expected values that were
    # computed externally with a correlation routine from a scientific library.
    expected_output = [
        3.77199074, 3.85069444, 3.92939815, 4.2650463, 4.35763889, 4.45023148,
        6.73032407, 6.89236111, 7.05439815, 7.22337963, 7.39930556, 7.57523148,
        9.68865741, 9.93402778, 10.17939815, 10.18171296, 10.44097222,
        10.70023148
    ]
    # expected_shape = [1, 3, 1, 2, 5]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],  # b, z, y, x, fin
        filter_in_sizes=[2, 2, 2, 3, 3],  # z, y, x, fin, fout
        stride=1,
        padding="VALID",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

  @parameterized.named_parameters(*DILATED_PARAMS)
  def testConv3D2x2x2Filter1x2x1Dilation(self, data_format, use_gpu, op_name):
    # [FUNCTION DOC]
    # Parameterized test (layout, device, op variant). Runs only when a CUDA GPU is available or
    # MKL is enabled outside eager mode; otherwise silently passes. Compares the op with the
    # generic convolution routine for a two-by-two-by-two filter, stride one, valid padding and a
    # dilation of two along the second spatial axis.
    ctx = context.context()
    is_eager = ctx is not None and ctx.executing_eagerly()
    if test.is_gpu_available(cuda_only=True) or \
      (test_util.IsMklEnabled() and is_eager is False):
      self._VerifyDilatedConvValues(
          tensor_in_sizes=[1, 4, 6, 3, 1],
          filter_in_sizes=[2, 2, 2, 1, 1],
          stride=1,
          padding="VALID",
          dilations=[1, 2, 1],
          data_format=data_format,
          use_gpu=use_gpu,
          op_name=op_name,
      )

  @parameterized.named_parameters(*TEST_PARAMS)
  def testConv3DStrides(self, data_format, dtype, use_gpu, op_name):
    # [FUNCTION DOC]
    # Parameterized test (layout, dtype, device, op variant). Verifies same padding with a
    # different stride per spatial dimension (two, three, one) and a one-by-two-by-three filter
    # against a fixed list of expected output values.
    expected_output = [
        0.06071429, 0.08988095, 0.10238095, 0.11488095, 0.12738095, 0.13988095,
        0.08452381, 0.26071429, 0.35238095, 0.36488095, 0.37738095, 0.38988095,
        0.40238095, 0.23452381, 0.46071429, 0.61488095, 0.62738095, 0.63988095,
        0.65238095, 0.66488095, 0.38452381, 1.12738095, 1.48988095, 1.50238095,
        1.51488095, 1.52738095, 1.53988095, 0.88452381, 1.32738095, 1.75238095,
        1.76488095, 1.77738095, 1.78988095, 1.80238095, 1.03452381, 1.52738095,
        2.01488095, 2.02738095, 2.03988095, 2.05238095, 2.06488095, 1.18452381,
        2.19404762, 2.88988095, 2.90238095, 2.91488095, 2.92738095, 2.93988095,
        1.68452381, 2.39404762, 3.15238095, 3.16488095, 3.17738095, 3.18988095,
        3.20238095, 1.83452381, 2.59404762, 3.41488095, 3.42738095, 3.43988095,
        3.45238095, 3.46488095, 1.98452381
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 5, 8, 7, 1],
        filter_in_sizes=[1, 2, 3, 1, 1],
        stride=[2, 3, 1],  # different stride for each spatial dimension
        padding="SAME",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

  @parameterized.named_parameters(*TEST_PARAMS)
  def testConv3D2x2x2FilterStride2(self, data_format, dtype, use_gpu, op_name):
    # [FUNCTION DOC]
    # Parameterized test (layout, dtype, device, op variant). Verifies a two-by-two-by-two filter
    # with stride two and valid padding against six expected output values.
    expected_output = [
        3.77199074, 3.85069444, 3.92939815, 9.68865741, 9.93402778, 10.17939815
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2,
        padding="VALID",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

  @parameterized.named_parameters(*TEST_PARAMS)
  def testConv3DStride3(self, data_format, dtype, use_gpu, op_name):
    # [FUNCTION DOC]
    # Parameterized test (layout, dtype, device, op variant). Verifies stride three with valid
    # padding on a six-by-seven-by-eight, two-channel input and a three-by-two-by-one filter
    # with three output channels, against a fixed list of expected values.
    expected_output = [
        1.51140873, 1.57167659, 1.63194444, 1.56349206, 1.62673611, 1.68998016,
        1.6155754, 1.68179563, 1.74801587, 1.9280754, 2.01215278, 2.09623016,
        1.98015873, 2.0672123, 2.15426587, 2.03224206, 2.12227183, 2.21230159,
        4.4280754, 4.65500992, 4.88194444, 4.48015873, 4.71006944, 4.93998016,
        4.53224206, 4.76512897, 4.99801587, 4.84474206, 5.09548611, 5.34623016,
        4.8968254, 5.15054563, 5.40426587, 4.94890873, 5.20560516, 5.46230159
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 6, 7, 8, 2],
        filter_in_sizes=[3, 2, 1, 2, 3],
        stride=3,
        padding="VALID",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

  @parameterized.named_parameters(*TEST_PARAMS)
  def testConv3D2x2x2FilterStride2Same(
      self, data_format, dtype, use_gpu, op_name
  ):
    # [FUNCTION DOC]
    # Parameterized test (layout, dtype, device, op variant). Verifies a two-by-two-by-two filter
    # with stride two and same padding against twelve expected output values.
    expected_output = [
        3.77199074, 3.85069444, 3.92939815, 2.0162037, 2.06597222, 2.11574074,
        9.68865741, 9.93402778, 10.17939815, 4.59953704, 4.73263889, 4.86574074
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2,
        padding="SAME",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

  def _TestConv3DEmptyTensorOutputShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name starts with an underscore and not the test prefix, so the test
    # runner never collects or runs it; it is dead code. Its docstring "Args: none" is also
    # redundant.
    # For every dtype from the dtype helper (non-GPU setting), runs the 3D convolution on an
    # input with batch size zero, a three-cubed filter, same padding and unit strides/dilations,
    # and asserts the result shape is the zero-batch output shape with 64 channels.
    """Verifies the output shape of the Conv3D op when output tensor is empty.

    Args: none
    """
    input_shape = [0, 112, 112, 112, 32]
    filter_shape = [3, 3, 3, 32, 64]

    output_shape = [0, 112, 112, 112, 64]
    input_data = 1
    filter_data = 1
    for data_type in DtypesToTest(False):
      input_tensor = constant_op.constant(
          input_data, shape=input_shape, dtype=data_type, name="input")
      filter_tensor = constant_op.constant(
          filter_data, shape=filter_shape, dtype=data_type, name="filter")
      conv = nn_ops.conv3d(
          input_tensor,
          filter_tensor,
          strides=[1, 1, 1, 1, 1],
          dilations=[1, 1, 1, 1, 1],
          padding="SAME",
          data_format="NDHWC",
          name="conv")
      values = self.evaluate(conv)
      self.assertEqual(values.shape, tensor_shape.TensorShape(output_shape))

  @parameterized.named_parameters(*TEST_PARAMS)
  def testKernelSmallerThanStride(self, data_format, dtype, use_gpu, op_name):
    # [FUNCTION DOC]
    # Parameterized test (layout, dtype, device, op variant). Checks four cases where the filter
    # is smaller than the stride: a one-by-one-by-one filter with stride two on a three-cubed
    # input under same and valid padding, and a two-cubed filter with stride three on a
    # seven-cubed input under same and valid padding. Compares each with fixed expected values.
    expected_output = [
        0.03703704, 0.11111111, 0.25925926, 0.33333333, 0.7037037, 0.77777778,
        0.92592593, 1.
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2,
        padding="SAME",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2,
        padding="VALID",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

    expected_output = [
        0.54081633, 0.58017493, 0.28061224, 0.81632653, 0.85568513, 0.40306122,
        0.41873178, 0.4340379, 0.19642857, 2.46938776, 2.50874636, 1.1377551,
        2.74489796, 2.78425656, 1.26020408, 1.16873178, 1.1840379, 0.51785714,
        1.09511662, 1.10604956, 0.44642857, 1.17164723, 1.18258017, 0.47704082,
        0.3691691, 0.37244898, 0.125
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3,
        padding="SAME",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

    expected_output = [
        0.540816, 0.580175, 0.816327, 0.855685, 2.469388, 2.508746, 2.744898,
        2.784257
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3,
        padding="VALID",
        expected=expected_output,
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

  @parameterized.named_parameters(*TEST_PARAMS)
  def testKernelSizeMatchesInputSize(
      self, data_format, dtype, use_gpu, op_name
  ):
    # [FUNCTION DOC]
    # Parameterized test (layout, dtype, device, op variant). Verifies valid padding with stride
    # one when the filter spans the whole spatial input, giving two expected output values.
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 2, 1],
        filter_in_sizes=[2, 1, 2, 1, 2],
        stride=1,
        padding="VALID",
        expected=[1.5625, 1.875],
        data_format=data_format,
        dtype=dtype,
        use_gpu=use_gpu,
        op_name=op_name,
    )

  def testZeroSizedFilterThrowsIllegalArgument(self):
    # [FUNCTION DOC]
    # Test: runs the 3D convolution with a filter that has a zero-length dimension and asserts an
    # invalid-argument error whose message says the filter must not have zero elements or has a
    # non-positive dimension.
    tensor_in_sizes = [1, 1, 1, 1, 1]
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    filter_in = np.ones((1, 1, 0, 1, 1), dtype=np.float32)
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "filter must not have zero elements"
        "|has a non-positive dimension"):
      self.evaluate(
          nn_ops.conv3d(x1, filter_in, strides=[1, 1, 1, 1, 1], padding="SAME"))

  def _ConstructAndTestGradientForConfig(
      self, batch, input_shape, filter_shape, in_depth, out_depth, stride,
      padding, test_input, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: numerical gradient check for one layout/device configuration.
    # KNOWN ISSUE: the tolerance variable is only assigned for float64, float32, float16 and
    # bfloat16, and the reference gradient is only set when float32 or float64 is processed
    # first; the dtype ordering from the dtype helper is relied upon, otherwise a name error
    # would occur for half types. The progress message is printed with a print statement.
    # Inputs: batch size, input_shape and filter_shape (triples of spatial sizes), in_depth and
    # out_depth, stride (integer or iterable of three), padding, test_input (true checks the
    # gradient with respect to the input, false the filter), data_format, use_gpu.
    # Computes the expected output shape, then for each dtype builds scaled constant tensors,
    # runs the 3D convolution, asserts its static shape, computes theoretical and numerical
    # Jacobians and asserts the max error is below a dtype-specific tolerance (half types are
    # compared against the single precision theoretical gradient). Returns nothing.

    input_planes, input_rows, input_cols = input_shape

    filter_planes, filter_rows, filter_cols = filter_shape

    input_shape = [batch, input_planes, input_rows, input_cols, in_depth]
    filter_shape = [
        filter_planes, filter_rows, filter_cols, in_depth, out_depth
    ]

    if isinstance(stride, collections_abc.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]

    if padding == "VALID":
      output_planes = int(
          math.ceil((input_planes - filter_planes + 1.0) / strides[1]))
      output_rows = int(
          math.ceil((input_rows - filter_rows + 1.0) / strides[2]))
      output_cols = int(
          math.ceil((input_cols - filter_cols + 1.0) / strides[3]))
    else:
      output_planes = int(math.ceil(float(input_planes) / strides[1]))
      output_rows = int(math.ceil(float(input_rows) / strides[2]))
      output_cols = int(math.ceil(float(input_cols) / strides[3]))
    output_shape = [batch, output_planes, output_rows, output_cols, out_depth]
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]

    for data_type in DtypesToTest(use_gpu=use_gpu):
      # TODO(mjanusz): Modify gradient_checker to also provide max relative
      # error and synchronize the tolerance levels between the tests for forward
      # and backward computations.
      if data_type == dtypes.float64:
        tolerance = 1e-8
      elif data_type == dtypes.float32:
        tolerance = 5e-3
      elif data_type == dtypes.float16:
        tolerance = 5e-3 if test.is_built_with_rocm() else 1e-3
      elif data_type == dtypes.bfloat16:
        tolerance = 1e-2

      with self.cached_session(use_gpu=use_gpu):
        orig_input_tensor = constant_op.constant(
            input_data, shape=input_shape, dtype=data_type, name="input")
        filter_tensor = constant_op.constant(
            filter_data, shape=filter_shape, dtype=data_type, name="filter")

        if data_format == "NCDHW":
          input_tensor = test_util.NHWCToNCHW(orig_input_tensor)
          new_strides = test_util.NHWCToNCHW(strides)
        else:
          input_tensor = orig_input_tensor
          new_strides = strides

        conv = nn_ops.conv3d(
            input_tensor,
            filter_tensor,
            new_strides,
            padding,
            data_format=data_format,
            name="conv")

        if data_format == "NCDHW":
          conv = test_util.NCHWToNHWC(conv)

        self.assertEqual(conv.shape, tensor_shape.TensorShape(output_shape))

        if test_input:
          jacob_t, jacob_n = gradient_checker.compute_gradient(
              orig_input_tensor, input_shape, conv, output_shape)
        else:
          jacob_t, jacob_n = gradient_checker.compute_gradient(
              filter_tensor, filter_shape, conv, output_shape)

        if data_type != dtypes.float16 and data_type != dtypes.bfloat16:
          reference_jacob_t = jacob_t
          err = np.fabs(jacob_t - jacob_n).max()
        else:
          # Compare fp16/bf16 theoretical gradients to fp32 theoretical
          # gradients, since fp16/bf16 numerical gradients are too imprecise.
          err = np.fabs(jacob_t - reference_jacob_t).max()

      print("conv3d gradient error = ", err)
      self.assertLess(err, tolerance)

  def ConstructAndTestGradient(self, **kwargs):
    # [FUNCTION DOC]
    # HELPER: runs the gradient check for every configuration from the configuration helper.
    # Takes keyword arguments (batch, input_shape, filter_shape, in_depth, out_depth, stride,
    # padding, test_input) and forwards them together with each data format and GPU flag to the
    # per-configuration gradient checker. Returns nothing; failures are assertion errors.
    for data_format, use_gpu in GetTestConfigs():
      self._ConstructAndTestGradientForConfig(data_format=data_format,
                                              use_gpu=use_gpu, **kwargs)

  @test_util.run_deprecated_v1
  def testInputGradientValidPaddingStrideOne(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using valid padding
    # and stride one. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 5, 4),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientValidPaddingStrideOne(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the filter, using valid padding
    # and stride one. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=4,
        input_shape=(4, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientValidPaddingStrideTwo(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using valid padding
    # and stride two. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(6, 3, 5),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="VALID",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientValidPaddingStrideTwo(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the filter, using valid padding
    # and stride two. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(7, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="VALID",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientValidPaddingStrideThree(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using valid padding
    # and stride three. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 7, 6),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="VALID",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientValidPaddingStrideThree(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the filter, using valid padding
    # and stride three. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(4, 4, 7),
        filter_shape=(4, 4, 4),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="VALID",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientSamePaddingStrideOne(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using same padding
    # and stride one. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 2, 2),
        filter_shape=(3, 2, 1),
        in_depth=2,
        out_depth=1,
        stride=1,
        padding="SAME",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientSamePaddingStrideOne(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the filter, using same padding
    # and stride one. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="SAME",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientSamePaddingStrideTwo(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using same padding
    # and stride two. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(6, 3, 4),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="SAME",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientSamePaddingStrideTwo(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the filter, using same padding
    # and stride two. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=4,
        input_shape=(7, 3, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="SAME",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientSamePaddingStrideThree(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using same padding
    # and stride three. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(9, 3, 6),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="SAME",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientSamePaddingStrideThree(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the filter, using same padding
    # and stride three. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(9, 4, 7),
        filter_shape=(4, 4, 4),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="SAME",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientSamePaddingDifferentStrides(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using same padding
    # and a different stride per spatial dimension. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=1,
        input_shape=(5, 8, 7),
        filter_shape=(1, 2, 3),
        in_depth=2,
        out_depth=3,
        stride=[2, 3, 1],
        padding="SAME",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientKernelSizeMatchesInputSize(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the filter, using valid padding
    # and stride one where the filter covers the whole spatial input. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(5, 4, 3),
        filter_shape=(5, 4, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientKernelSizeMatchesInputSize(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using valid padding
    # and stride one where the filter covers the whole spatial input. Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(5, 4, 3),
        filter_shape=(5, 4, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=True)

  def disabledtestFilterGradientSamePaddingDifferentStrides(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name starts with disabled and not the test prefix, so it is never run; it is
    # intentionally disabled dead code.
    # Would check the filter gradient with same padding and a different stride per spatial
    # dimension via the gradient helper.
    self.ConstructAndTestGradient(
        batch=1,
        input_shape=(5, 8, 7),
        filter_shape=(1, 2, 3),
        in_depth=2,
        out_depth=3,
        stride=[2, 3, 1],
        padding="SAME",
        test_input=False)

  # Test the fast path in gemm_pack_rhs/gemm_pack_colmajor_block, when channel
  # dimension is a multiple of packet size.
  @test_util.run_deprecated_v1
  def testInputGradientValidPaddingStrideOneFastPath(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the input, using valid padding
    # and stride one with eight input channels (a multiple of the packet size, exercising the fast packing path). Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 5, 4),
        filter_shape=(2, 2, 2),
        in_depth=8,
        out_depth=2,
        stride=1,
        padding="VALID",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientValidPaddingStrideOneFastPath(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of the 3D convolution with respect to the filter, using valid padding
    # and stride one with eight input channels (a multiple of the packet size, exercising the fast packing path). Runs the gradient helper over all layout/device configurations and asserts the
    # theoretical and numerical gradients agree within tolerance. Graph mode (deprecated v1).
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(4, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=8,
        out_depth=2,
        stride=1,
        padding="VALID",
        test_input=False)

  # Testing for backprops
  def _RunAndVerifyBackprop(self, input_sizes, filter_sizes, output_sizes,
                            strides, dilations, padding, data_format, use_gpu,
                            err, mode):
    # [FUNCTION DOC]
    # HELPER: compares input or filter gradients of the 3D op against gradients of the generic
    # convolution routine.
    # KNOWN ISSUE: the output_sizes argument is never used, and the unused session variable is
    # created but not used. When dilation is above one and no GPU is used, nothing is checked.
    # Inputs: input_sizes, filter_sizes, output_sizes (lists), strides and dilations (lists of three),
    # padding, data_format, use_gpu, err (absolute tolerance), mode (input or filter, choosing the
    # gradient target). Builds both convolutions, takes gradients of each with respect to the
    # chosen tensor, evaluates them, asserts shapes and asserts the values are near within err.
    # Prints expected and actual values. Returns nothing.
    total_input_size = 1
    total_filter_size = 1
    for s in input_sizes:
      total_input_size *= s
    for s in filter_sizes:
      total_filter_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_input_size + 1)]
    x2 = [f * 1.0 for f in range(1, total_filter_size + 1)]
    default_dilations = (
        dilations[0] == 1 and dilations[1] == 1 and dilations[2] == 1)

    # If any dilation rate is larger than 1, only do test on the GPU
    # because we currently do not have a CPU implementation for arbitrary
    # dilation rates.
    if default_dilations or use_gpu:
      with self.cached_session(use_gpu=use_gpu) as sess:
        if data_format == "NCDHW":
          input_sizes = test_util.NHWCToNCHW(input_sizes)
        t1 = constant_op.constant(x1, shape=input_sizes)
        t2 = constant_op.constant(x2, shape=filter_sizes)
        full_strides = [1] + strides + [1]
        full_dilations = [1] + dilations + [1]
        if data_format == "NCDHW":
          full_strides = test_util.NHWCToNCHW(full_strides)
          full_dilations = test_util.NHWCToNCHW(full_dilations)
        actual = nn_ops.conv3d(
            t1,
            t2,
            strides=full_strides,
            dilations=full_dilations,
            padding=padding,
            data_format=data_format)
        expected = nn_ops.convolution(
            t1,
            t2,
            padding=padding,
            strides=strides,
            dilation_rate=dilations,
            data_format=data_format)
        if data_format == "NCDHW":
          actual = test_util.NCHWToNHWC(actual)
          expected = test_util.NCHWToNHWC(expected)
        actual_grad = gradients_impl.gradients(actual, t1
                                               if mode == "input" else t2)[0]
        expected_grad = gradients_impl.gradients(expected, t1
                                                 if mode == "input" else t2)[0]
        # "values" consists of two tensors for two backprops
        actual_value = self.evaluate(actual_grad)
        expected_value = self.evaluate(expected_grad)
        self.assertShapeEqual(actual_value, actual_grad)
        self.assertShapeEqual(expected_value, expected_grad)
      print("expected = ", expected_value)
      print("actual = ", actual_value)
      self.assertArrayNear(expected_value.flatten(), actual_value.flatten(),
                           err)

  @test_util.run_deprecated_v1
  def testConv3D2x2Depth3ValidBackpropFilterStride1x1Dilation2x1(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name mentions depth three and two-dimensional dilation, but the test uses
    # a three-dimensional input with a single channel and dilation of two on the first axis.
    # Test: only runs when a CUDA GPU is available; for each layout/device configuration checks the
    # filter gradient with stride one and dilation against the generic convolution gradient.
    # Graph mode (deprecated v1).
    if test.is_gpu_available(cuda_only=True):
      for (data_format, use_gpu) in GetTestConfigs():
        self._RunAndVerifyBackprop(
            input_sizes=[1, 3, 6, 1, 1],
            filter_sizes=[2, 2, 1, 1, 1],
            output_sizes=[1, 1, 5, 1, 1],
            strides=[1, 1, 1],
            dilations=[2, 1, 1],
            padding="VALID",
            data_format=data_format,
            use_gpu=use_gpu,
            err=1e-5,
            mode="filter")

  @test_util.run_deprecated_v1
  def testConv3D2x2Depth3ValidBackpropInputStride1x1Dilation2x1(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name mentions depth three and two-dimensional dilation, but the test uses
    # a three-dimensional input with a single channel and dilation of two on the first axis.
    # Test: only runs when a CUDA GPU is available; for each layout/device configuration checks the
    # input gradient with stride one and dilation against the generic convolution gradient.
    # Graph mode (deprecated v1).
    if test.is_gpu_available(cuda_only=True):
      for (data_format, use_gpu) in GetTestConfigs():
        self._RunAndVerifyBackprop(
            input_sizes=[1, 3, 6, 1, 1],
            filter_sizes=[2, 2, 1, 1, 1],
            output_sizes=[1, 1, 5, 1, 1],
            strides=[1, 1, 1],
            dilations=[2, 1, 1],
            padding="VALID",
            data_format=data_format,
            use_gpu=use_gpu,
            err=1e-5,
            mode="input")


if __name__ == "__main__":
  test.main()
