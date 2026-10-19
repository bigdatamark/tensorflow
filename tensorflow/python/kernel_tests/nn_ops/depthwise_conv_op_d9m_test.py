# [MODULE DOC]
# Tests that the depthwise 2-D convolution op is deterministic when TensorFlow op determinism is
# enabled. It defines one test case class that derives from the shared depthwise convolution base
# test class (so it also inherits all of that class's tests) and adds forward and backward
# determinism checks for CPU and GPU, including cuDNN-friendly shapes.
# Import-time side effects: importing the gradient functions so they are registered.
# When run as a script it enables op determinism globally before starting the test runner; this
# must happen in a separate test main from the non-deterministic tests to avoid cuDNN algorithm
# cache interference.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for determinsitic depthwise convolutional operations."""

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import depthwise_conv_op_base
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
# The following imports are required to register the gradient functions.
from tensorflow.python.ops.nn_grad import _DepthwiseConv2dNativeBackpropFilterGrad  # pylint: disable=unused-import
from tensorflow.python.ops.nn_grad import _DepthwiseConv2dNativeBackpropInputGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.run_all_without_tensor_float_32("Uses matmul")
class DepthwiseConv2DDeterministicTest(
    depthwise_conv_op_base.DepthwiseConv2DBase):
  # [CLASS DOC]
  # Test case verifying that the depthwise convolution forward pass and its gradients (to input and
  # filter) give bit-identical results across repeated runs when op determinism is enabled.
  # Derives from the shared depthwise convolution base test case, so it also inherits all of the
  # base value and gradient tests. Decorated to disable TensorFloat-32 so results are full precision.
  # Holds no state; the checks are done by generating large random inputs and comparing two runs.
  """Test determinism-related functionality of tf.nn.depthwise_conv2d."""

  def _genParams(self,
                 use_cudnn=False,
                 data_format="NHWC",
                 dtype=dtypes.float32,
                 seed=123):
    # [FUNCTION DOC]
    # HELPER: builds the random inputs shared by the forward and backward determinism tests.
    # Sets the global random seed to the given seed (side effect), then creates a random normal
    # input tensor and filter tensor. Parameters: use_cudnn (bool, default false) selects one input
    # channel, which lets cuDNN be used, versus two channels; data_format (text, NHWC default or
    # NCHW) selects the input and output layout; dtype (default 32-bit float) is the element type;
    # seed (integer, default 123) seeds the generator. The input is 2 x 500 x 1000 spatial with the
    # chosen channels; the filter is 7 x 7 with channel multiplier 10; strides are all one and the
    # padding is same. Returns a tuple of input data, filter data, strides list, padding name and
    # the expected output shape (same spatial size, channels multiplied by 10). Raises nothing.
    random_seed.set_seed(seed)
    batch_size = 2  # no interaction over batch, so make small
    if use_cudnn:
      # When op-determinism is not enabled, one input channel, plus a
      # cuDNN-supported filter size and number of output channels will result
      # in cuDNN being used for both backprop-to-input and backprop-to-filter on
      # cuDNN 7.6.3 and higher. When op-determnism is enabled, cuDNN is always
      # used for backprop-to-filter.
      input_channels = 1
    else:
      input_channels = 2  # no interaction over channels, so make small
    input_height = 500
    input_width = 1000
    if data_format == "NHWC":
      input_shape = (batch_size, input_height, input_width, input_channels)
    else:  # "NCHW"
      input_shape = (batch_size, input_channels, input_height, input_width)
    input_data = random_ops.random_normal(input_shape, dtype=dtype)
    # The following filter size results in nondeterminism being exercised in
    # cuDNN backprop (when determinism is not enabled) to both input and filter
    # as well as in the specialized (non-cuDNN) depthwise backprop to filter.
    filter_height = 7
    filter_width = 7
    channel_multiplier = 10
    filter_shape = (filter_height, filter_width, input_channels,
                    channel_multiplier)
    filter_data = random_ops.random_normal(filter_shape, dtype=dtype)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    output_height = input_height  # because same padding
    output_width = input_width  # because same padding
    output_channels = input_channels * channel_multiplier
    if data_format == "NHWC":
      output_shape = (batch_size, output_height, output_width, output_channels)
    else:  # "NCHW"
      output_shape = (batch_size, output_channels, output_height, output_width)
    return input_data, filter_data, strides, padding, output_shape

  def _testForwardDeterminismCase(self,
                                  use_cudnn=False,
                                  data_format="NHWC",
                                  dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: shared body of the forward determinism tests.
    # For each of five seeds, generates parameters (use_cudnn flag, data_format, dtype as given),
    # runs the version-2 depthwise convolution twice on identical input and filter, and asserts the
    # two results are exactly equal. Returns nothing; fails the test on any difference.
    for seed in range(5):
      p = self._genParams(use_cudnn, data_format, dtype, seed=seed)
      input_data, filter_data, strides, padding, _ = p

      result_a = nn_impl.depthwise_conv2d_v2(input_data, filter_data, strides,
                                             padding, data_format)
      result_b = nn_impl.depthwise_conv2d_v2(input_data, filter_data, strides,
                                             padding, data_format)

      self.assertAllEqual(result_a, result_b)

  @test_util.run_gpu_only
  def testForwardDeterminismGPU(self):
    # [FUNCTION DOC]
    # GPU-only test (skipped when no GPU). Runs the forward determinism check for every
    # combination of cuDNN-eligible or not, NHWC or NCHW layout, and half, single and double
    # precision. Asserts repeated forward runs give identical outputs.
    for use_cudnn in [False, True]:
      for data_format in ["NHWC", "NCHW"]:
        for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
          self._testForwardDeterminismCase(use_cudnn, data_format, dtype=dtype)

  def testForwardDeterminismCPU(self):
    # [FUNCTION DOC]
    # CPU forward determinism test. Skips itself if any physical GPU is present. Uses NHWC layout
    # only (the CPU op lacks NCHW) and checks bfloat16, single and double precision that two
    # forward runs on identical data are exactly equal.
    if tf_config.list_physical_devices("GPU"):
      self.skipTest("Test only runs when there is no GPU")
    data_format = "NHWC"  # CPU does not implement NCHW version of op
    for dtype in [dtypes.bfloat16.as_numpy_dtype, dtypes.float32,
                  dtypes.float64]:
      self._testForwardDeterminismCase(data_format=data_format, dtype=dtype)

  def _testBackwardDeterminismCase(self,
                                   using_gpu=False,
                                   use_cudnn=False,
                                   data_format="NHWC",
                                   dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: shared body of the backward determinism tests.
    # KNOWN ISSUE: the using_gpu parameter is accepted but never used in the body.
    # Generates parameters with seed 123 (use_cudnn, data_format and dtype as given), then for two
    # fixed seeds creates random upstream gradients of the output shape and computes the gradients
    # to the input and the filter twice via a gradient tape, asserting both pairs are exactly
    # equal. Returns nothing; fails the test on any difference.
    p = self._genParams(use_cudnn, data_format, dtype, seed=123)
    input_data, filter_data, strides, padding, output_shape = p

    def Gradients(upstream_gradients):
      # [FUNCTION DOC]
      # HELPER: computes input and filter gradients of the depthwise convolution for the test.
      # Takes upstream gradients (tensor shaped like the op output), records the convolution on a
      # tape watching the captured input and filter, weights the output by the upstream gradients,
      # and returns the list of gradients with respect to the input and the filter.
      with backprop.GradientTape() as tape:
        tape.watch(input_data)
        tape.watch(filter_data)
        op_output = nn_impl.depthwise_conv2d_v2(input_data, filter_data,
                                                strides, padding, data_format)
        gradient_injector_output = op_output * upstream_gradients
      return tape.gradient(gradient_injector_output, [input_data, filter_data])

    # Test only two seeds, since testing takes a long time
    for seed in (987, 988):
      upstream_gradients = random_ops.random_normal(
          output_shape, dtype=dtype, seed=seed)
      input_gradients_a, filter_gradients_a = Gradients(upstream_gradients)
      input_gradients_b, filter_gradients_b = Gradients(upstream_gradients)
      self.assertAllEqual(input_gradients_a, input_gradients_b)
      self.assertAllEqual(filter_gradients_a, filter_gradients_b)

  @test_util.run_gpu_only
  def testBackwardDeterminismGPU(self):
    # [FUNCTION DOC]
    # GPU-only test. Runs the backward determinism check over cuDNN-eligible or not, NHWC and
    # NCHW layouts, and half, single and double precision, asserting repeated gradient
    # computations to input and filter are identical.
    using_gpu = True
    for use_cudnn in [False, True]:
      for data_format in ["NHWC", "NCHW"]:
        for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
          self._testBackwardDeterminismCase(using_gpu, use_cudnn, data_format,
                                            dtype)

  def testBackwardDeterminismCPU(self):
    # [FUNCTION DOC]
    # CPU backward determinism test. Skips itself if a GPU is present. Uses NHWC layout and checks
    # bfloat16, single and double precision that repeated gradient computations are identical.
    if tf_config.list_physical_devices("GPU"):
      self.skipTest("Test only runs when there is no GPU")
    data_format = "NHWC"  # CPU does not implement NCHW version of op
    for dtype in [dtypes.bfloat16.as_numpy_dtype, dtypes.float32,
                  dtypes.float64]:
      self._testBackwardDeterminismCase(data_format=data_format, dtype=dtype)


if __name__ == "__main__":
  # The op-determinism setting can be enabled and disabled on-the-fly.
  # However, if cuDNN convolution is used (as it is for these tests) then its
  # setting at the time will influence which algorithm for a particular layer
  # configuration is cached (independently for XLA and non-XLA operation).
  #
  # The tests in this file must be run under a separate test.main from the
  # tests in depthwise_conv_op_test.py to prevent caching the selection of
  # nondeterminsitic algorithms, which would cause the tests defined in this
  # file to fail.
  #
  # Also because of this caching, the tests defined in depthwise_conv_op_base.py
  # should be run with and without op-determinism enabled in separate files.
  #
  # TODO(duncanriach): Implement cuDNN auto-tuning cache invalidation and
  # and execute when op-determinism setting is changed.
  tf_config.enable_op_determinism()
  test.main()
