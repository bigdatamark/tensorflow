# [MODULE DOC]
# Gradient tests for the filter-backprop op of 2-D convolution (conv2d_backprop_filter): checks
# numerically that the second-order gradient with respect to the input and the output backprop is
# accurate, for several paddings (named and explicit), strides, and a dilated variant that needs a
# CUDA GPU. The main test is currently skipped as flaky. Imports the nn gradient registrations
# for side effects.
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
"""Tests for convolution related functionality in tensorflow.ops.nn."""

import unittest
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.run_all_without_tensor_float_32(
    "Run Conv2D backprop without TF32 on GPU")
class Conv2DBackpropFilterGradTest(test.TestCase):
  # [CLASS DOC]
  # Test case for gradients of conv2d_backprop_filter. The class decorator turns off TF32 math on
  # GPU for every test so that precision does not affect the error tolerances. No other fixtures.

  # TODO(b/292002914): Enable this test after fixing its flakyness.
  @unittest.skip("Disable the flaky test.")
  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unconditionally skipped by a decorator because it is flaky (bug reference in the
    # source TODO), so it currently never runs.
    # Intended scenario: for SAME, VALID and two explicit padding lists and strides one and two,
    # with a seeded random input of shape five by eight by six by four and a three by three by four
    # by six filter, creates a random output backprop shaped like the forward output, runs
    # conv2d_backprop_filter, and asserts the numeric gradient error w.r.t. input and backprop is
    # below 3e-2 on GPU or 2e-3 otherwise. Prints the error; reseeds numpy each iteration.
    with self.cached_session():
      for padding in [
          "SAME",
          "VALID",
          [(0, 0), (1, 2), (3, 4), (0, 0)],
          [(0, 0), (0, 3), (4, 2), (0, 0)]
      ]:
        for stride in [1, 2]:
          np.random.seed(1)
          in_shape = [5, 8, 6, 4]
          in_val = constant_op.constant(
              2 * np.random.random_sample(in_shape) - 1, dtype=dtypes.float32)
          filter_shape = [3, 3, 4, 6]
          # Make a convolution op with the current settings, just to easily get
          # the shape of the output.
          conv_out = nn_ops.conv2d(
              in_val,
              array_ops.zeros(filter_shape),
              strides=[1, stride, stride, 1],
              padding=padding)
          out_backprop_shape = conv_out.get_shape().as_list()
          out_backprop_val = constant_op.constant(
              2 * np.random.random_sample(out_backprop_shape) - 1,
              dtype=dtypes.float32)
          output = nn_ops.conv2d_backprop_filter(
              in_val,
              filter_shape,
              out_backprop_val,
              strides=[1, stride, stride, 1],
              padding=padding)
          err = gradient_checker.compute_gradient_error(
              [in_val, out_backprop_val], [in_shape, out_backprop_shape],
              output, filter_shape)
          print("conv2d_backprop_filter gradient err = %g " % err)
          err_tolerance = 3e-2 if test.is_gpu_available() else 2e-3
          self.assertLess(
              err,
              err_tolerance,
              msg="padding={0},stride={1},".format(str(padding), stride))

  @test_util.run_deprecated_v1
  def testGradientDilatedConv(self):
    # [FUNCTION DOC]
    # Scenario: runs only when a CUDA GPU is available (otherwise silently passes). Same setup as
    # the main gradient test but with dilation of two in both spatial directions and different
    # explicit paddings, using the dilated convolution and backprop-filter ops. Asserts the
    # numeric gradient error is below 1e-2 for each padding and stride combination. Prints the
    # error and reseeds numpy each iteration.
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        for padding in [
            "SAME",
            "VALID",
            [(0, 0), (3, 5), (2, 1), (0, 0)],
            [(0, 0), (5, 2), (5, 1), (0, 0)]
        ]:
          for stride in [1, 2]:
            np.random.seed(1)
            in_shape = [5, 8, 6, 4]
            in_val = constant_op.constant(
                2 * np.random.random_sample(in_shape) - 1, dtype=dtypes.float32)
            filter_shape = [3, 3, 4, 6]
            # Make a convolution op with the current settings,
            # just to easily get the shape of the output.
            conv_out = nn_ops.conv2d(
                in_val,
                array_ops.zeros(filter_shape),
                dilations=[1, 2, 2, 1],
                strides=[1, stride, stride, 1],
                padding=padding)
            out_backprop_shape = conv_out.get_shape().as_list()
            out_backprop_val = constant_op.constant(
                2 * np.random.random_sample(out_backprop_shape) - 1,
                dtype=dtypes.float32)
            output = nn_ops.conv2d_backprop_filter(
                in_val,
                filter_shape,
                out_backprop_val,
                dilations=[1, 2, 2, 1],
                strides=[1, stride, stride, 1],
                padding=padding)
            err = gradient_checker.compute_gradient_error(
                [in_val, out_backprop_val], [in_shape, out_backprop_shape],
                output, filter_shape)
            print("conv2d_backprop_filter gradient err = %g " % err)
            err_tolerance = 1e-2
            self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  test.main()
