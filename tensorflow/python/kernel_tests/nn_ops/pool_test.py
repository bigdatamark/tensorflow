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
# Tests for the unified N-dimensional pooling function in the nn ops (max and average
# pooling with padding, dilation and strides, optionally with channel-first data formats).
# Provides two numpy reference implementations (single axis and multi axis pooling) used as
# ground truth, and one test class that compares the op against them for 1-D, 2-D and 3-D
# inputs and checks numerical gradients. Importing registers nn gradients. Running as a
# script starts the standard test runner.
"""Tests for unified pooling functionality in tensorflow.ops.nn."""

import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def pool_direct_single_axis(
    input,  # pylint: disable=redefined-builtin
    axis,
    window_size,
    pooling_type,
    padding,
    dilation_rate,
    stride):
  # [FUNCTION DOC]
  # HELPER: numpy reference pooling along one axis, used by pool_direct and the tests.
  # Inputs: input (numpy array), axis (which axis to pool), window_size (positive int),
  # pooling_type ("MAX" or "AVG"), padding ("SAME" or "VALID"), dilation_rate and stride
  # (positive ints). Computes the effective window size with dilation, derives the output
  # size and leading padding (SAME pads so that output size is ceil of input over stride, with
  # half of the padding before; VALID uses no padding), then for each output position takes the
  # max or mean over the dilated window slice clipped to the array bounds (padding cells are
  # simply excluded, not zero filled). Returns a new array with the pooled axis resized.
  # Raises a value error for an unsupported padding or pooling type. Does not modify input.
  """Numpy implementation of pooling along a single axis.

  This is intended for testing only, and therefore isn't particularly efficient.

  See pool_direct below for the meaning of the arguments.

  Args:
    input: numpy array.
    axis: axis along which to perform pooling.
    window_size: int >= 1.  Size of pooling window within axis.
    pooling_type: either "MAX" or "AVG".
    padding: either "SAME" or "VALID".
    dilation_rate: int >= 1.  Dilation factor for window, i.e. stride at which
      to sample input.
    stride: int >= 1.  Stride at which to generate output.

  Returns:
    pooling output array of rank N+2.

  Raises:
    ValueError: if arguments are invalid.
  """
  effective_window_size = (window_size - 1) * dilation_rate + 1
  input_size = input.shape[axis]
  if padding == "SAME":
    output_size = int(math.ceil(input_size / stride))
    total_padding_amount = max(0, (output_size - 1) * stride +
                               effective_window_size - input_size)
    before_padding = total_padding_amount // 2
  elif padding == "VALID":
    output_size = int(
        math.ceil((input_size - effective_window_size + 1) / stride))
    before_padding = 0
  else:
    raise ValueError("Unsupported padding type: %r" % (padding,))

  output_shape = input.shape[:axis] + (output_size,) + input.shape[axis + 1:]
  output = np.zeros(output_shape, input.dtype)
  initial_dim_selector = tuple(np.s_[:] for _ in range(axis))
  if pooling_type == "MAX":
    pooling_func = np.max
  elif pooling_type == "AVG":
    pooling_func = np.mean
  else:
    raise ValueError("Unsupported pooling type: %r" % (pooling_type,))
  for output_pos in range(output_size):
    input_start_pos = output_pos * stride - before_padding
    input_end_pos = min(input_start_pos + effective_window_size, input_size)
    if input_start_pos < 0:
      input_start_pos += dilation_rate
    input_slice = np.s_[input_start_pos:input_end_pos:dilation_rate]

    output[initial_dim_selector + (output_pos,)] = pooling_func(
        input[initial_dim_selector + (input_slice,)], axis=axis)
  return output


def pool_direct(
    input,  # pylint: disable=redefined-builtin
    window_shape,
    pooling_type,
    padding,  # pylint: disable=redefined-builtin
    dilation_rate,
    strides,
    data_format=None):
  # [FUNCTION DOC]
  # HELPER: numpy reference for N-dimensional pooling, used as ground truth by the tests.
  # Inputs: input (numpy array of rank N+2), window_shape, dilation_rate and strides (sequences
  # of N positive ints), pooling_type ("MAX" or "AVG"), padding ("SAME" or "VALID"), and an
  # optional data_format; a format starting with "NC" means spatial dims start at axis 2,
  # otherwise they start at axis 1. Applies the single-axis reference to each spatial axis in
  # turn and returns the resulting array. Errors from the single-axis helper (value error
  # for unsupported padding or pooling type) propagate. Does not modify the input.
  """Numpy implementation of pooling.

  This is intended for testing only, and therefore isn't particularly efficient.

  See tensorflow.nn.pool.

  Args:
    input: numpy array of rank N+2.
    window_shape: Sequence of N ints >= 1.
    pooling_type: either "MAX" or "AVG".
    padding: either "SAME" or "VALID".
    dilation_rate: Sequence of N ints >= 1.
    strides: Sequence of N ints >= 1.
    data_format: If specified and starts with "NC", indicates that second
      dimension, rather than the last dimension, specifies the channel.

  Returns:
    pooling output array of rank N+2.

  Raises:
    ValueError: if arguments are invalid.
  """
  if data_format is None or not data_format.startswith("NC"):
    spatial_start_dim = 1
  else:
    spatial_start_dim = 2
  output = input
  for i in range(len(window_shape)):
    output = pool_direct_single_axis(
        input=output,
        axis=i + spatial_start_dim,
        window_size=window_shape[i],
        pooling_type=pooling_type,
        padding=padding,
        dilation_rate=dilation_rate[i],
        stride=strides[i])
  return output


class PoolingTest(test.TestCase):
  # [CLASS DOC]
  # Test case comparing the unified pooling op to the numpy reference over many combinations
  # of padding, pooling type, window shape, dilation and strides in one to three spatial
  # dimensions, plus channel-first formats on CUDA and numerical gradient checks.

  def _test(self, input_shape, **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared forward comparison. Builds a float32 input of the given shape filled with
    # strictly negative decreasing numbers (so wrongly used zero padding would show up), runs the
    # numpy reference and the real pooling op with the extra keyword arguments (window shape,
    # pooling type, padding, dilation rate, strides, optional data format), and asserts
    # the results are close (tolerance 1e-2). Returns nothing.
    # Use negative numbers to make sure there isn't any zero padding getting
    # used.
    x = -np.arange(
        np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1
    y1 = pool_direct(input=x, **kwargs)
    y2 = nn_ops.pool(input=x, **kwargs)
    self.assertAllClose(y1, self.evaluate(y2), rtol=1e-2, atol=1e-2)

  def testPoolSimple(self):
    # [FUNCTION DOC]
    # Scenario: input of shape 1x1x10x1 with window 1x3 and strides 1x2, for both paddings and
    # both pooling types; expects the op to match the numpy reference.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          self._test(
              input_shape=[1, 1, 10, 1],
              window_shape=[1, 3],
              padding=padding,
              pooling_type=pooling_type,
              dilation_rate=[1, 1],
              strides=[1, 2])

  def testPool1D(self):
    # [FUNCTION DOC]
    # Sweeps one-dimensional pooling over paddings, pooling types, two input lengths, window
    # sizes 1 to 3, dilation rates (VALID padding only, with unit strides) and strides (skipping
    # strides larger than the window, with unit dilation); expects agreement with the reference.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          for input_shape in [[2, 9, 2], [2, 10, 2]]:
            for window_shape in [[1], [2], [3]]:
              if padding != "SAME":
                for dilation_rate in [[1], [2], [3]]:
                  self._test(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1])
              for strides in [[1], [2], [3]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1],
                    strides=strides)

  def testPool2D(self):
    # [FUNCTION DOC]
    # Same sweep as the 1-D test for two spatial dimensions, with several window shapes,
    # dilation rates (VALID only) and strides; expects agreement with the numpy reference.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          for input_shape in [[2, 9, 10, 2], [2, 10, 9, 2]]:
            for window_shape in [[1, 1], [2, 1], [2, 3]]:
              if padding != "SAME":
                for dilation_rate in [[1, 1], [2, 1], [1, 2], [2, 3]]:
                  self._test(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1, 1])
              for strides in [[1, 1], [2, 1], [1, 2], [2, 3]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1, 1],
                    strides=strides)

  def testPool3D(self):
    # [FUNCTION DOC]
    # Same sweep for three spatial dimensions with several window shapes, dilation rates (VALID
    # only) and strides; expects agreement with the numpy reference.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          for input_shape in [[2, 9, 10, 11, 2], [2, 10, 9, 11, 2]]:
            for window_shape in [[1, 1, 1], [2, 1, 2], [2, 3, 2]]:
              if padding != "SAME":
                for dilation_rate in [[1, 1, 1], [2, 1, 2], [1, 2, 2],
                                      [2, 3, 3]]:
                  self._test(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1, 1, 1])
              for strides in [[1, 1, 1], [2, 1, 2], [1, 2, 2], [2, 3, 3]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1, 1, 1],
                    strides=strides)

  def testPoolNC(self):
    # [FUNCTION DOC]
    # Channel-first formats (NCW, NCHW, NCDHW) are only supported on CUDA, so the test does
    # nothing unless a CUDA GPU is available. Then, for max pooling with both paddings and
    # various strides, checks the op against the reference; finally a VALID padding NCHW case
    # with dilation rate 2 is checked once.
    if test.is_gpu_available(cuda_only=True):
      # "NC*" format is currently only supported on CUDA.
      with self.session():
        for padding in ["SAME", "VALID"]:
          self._test(
              input_shape=[2, 2, 9],
              window_shape=[2],
              padding=padding,
              pooling_type="MAX",
              strides=[1],
              dilation_rate=[1],
              data_format="NCW")
          self._test(
              input_shape=[2, 2, 9],
              window_shape=[2],
              padding=padding,
              pooling_type="MAX",
              strides=[2],
              dilation_rate=[1],
              data_format="NCW")
          self._test(
              input_shape=[2, 2, 7, 9],
              window_shape=[2, 2],
              padding=padding,
              pooling_type="MAX",
              strides=[1, 2],
              dilation_rate=[1, 1],
              data_format="NCHW")
          self._test(
              input_shape=[2, 2, 7, 5, 3],
              window_shape=[2, 2, 2],
              padding=padding,
              pooling_type="MAX",
              strides=[1, 2, 1],
              dilation_rate=[1, 1, 1],
              data_format="NCDHW")
        self._test(
            input_shape=[2, 2, 7, 9],
            window_shape=[2, 2],
            padding="VALID",
            pooling_type="MAX",
            strides=[1, 1],
            dilation_rate=[2, 2],
            data_format="NCHW")

  def _test_gradient(self, input_shape, **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared numerical gradient check. Builds a float32 constant of negative decreasing
    # values with the given shape, applies the pooling op with the extra keyword arguments,
    # and uses the graph-mode gradient checker (with the same initial value) to compute the
    # error between analytic and numeric gradients; asserts it is below 1e-2. Returns nothing.
    x_val = -np.arange(
        np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1
    x = constant_op.constant(x_val, name="x", dtype=dtypes.float32)
    output = nn_ops.pool(input=x, **kwargs)
    y_shape = output.get_shape().as_list()
    err = gradient_checker.compute_gradient_error([x], [input_shape],
                                                  output,
                                                  y_shape,
                                                  x_init_value=[x_val])
    err_tolerance = 1e-2
    self.assertLess(err, err_tolerance)

  @test_util.run_deprecated_v1
  def testGradient1D(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the strided loop calls the forward-value helper instead of the gradient
    # helper, so gradients are only checked for the dilation cases (VALID padding, unit
    # strides); the strided cases just repeat the forward comparison.
    # Sweeps 1-D pooling over paddings, pooling types, small shapes and windows. Graph mode only.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["AVG", "MAX"]:
          for input_shape in [[2, 5, 2], [1, 4, 1]]:
            for window_shape in [[1], [2]]:
              if padding != "SAME":
                for dilation_rate in [[1], [2]]:
                  self._test_gradient(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1])
              for strides in [[1], [2]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1],
                    strides=strides)

  @test_util.run_deprecated_v1
  def testGradient2D(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: as in the 1-D test, the strided loop calls the forward-value helper instead of
    # the gradient helper, so strided cases are not gradient-checked.
    # Sweeps 2-D pooling over paddings, pooling types, shapes, windows and dilation rates
    # (VALID padding) with gradient checks. Graph mode only.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["AVG", "MAX"]:
          for input_shape in [[2, 4, 5, 2], [1, 5, 4, 1]]:
            for window_shape in [[1, 1], [2, 1], [2, 2]]:
              if padding != "SAME":
                for dilation_rate in [[1, 1], [2, 1], [2, 2]]:
                  self._test_gradient(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1, 1])
              for strides in [[1, 1], [2, 1], [1, 2], [2, 2]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1, 1],
                    strides=strides)

  @test_util.run_deprecated_v1
  def testGradient3D(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: as in the 1-D test, the strided loop calls the forward-value helper instead of
    # the gradient helper, so strided cases are not gradient-checked.
    # Sweeps 3-D pooling over paddings, pooling types, shapes, windows and dilation rates
    # (VALID padding) with gradient checks. Graph mode only.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["AVG", "MAX"]:
          for input_shape in [[1, 3, 5, 4, 1], [1, 5, 4, 3, 1]]:
            for window_shape in [[1, 1, 1], [2, 1, 2], [2, 2, 2]]:
              if padding != "SAME":
                for dilation_rate in [[1, 1, 1], [2, 1, 2], [2, 2, 2]]:
                  self._test_gradient(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=dilation_rate,
                      strides=[1, 1, 1])
              for strides in [[1, 1, 1], [2, 1, 2], [2, 2, 2]]:
                if np.any(np.array(strides) > window_shape):
                  continue
                self._test(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    padding=padding,
                    pooling_type=pooling_type,
                    dilation_rate=[1, 1, 1],
                    strides=strides)


if __name__ == "__main__":
  test.main()
