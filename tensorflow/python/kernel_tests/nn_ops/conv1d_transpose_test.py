# [MODULE DOC]
# Kernel tests for conv1d_transpose in channels-last and channels-first (NCW, CUDA only) layouts,
# with SAME and VALID padding and different strides, using all-ones inputs and filters so the
# expected output is computed by counting overlaps; plus a (currently skipped) gradient check.
# Imports the nn gradient registrations for side effects.
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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class Conv1DTransposeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for conv1d_transpose. Each test builds float32 all-ones tensors, runs the op and
  # compares against an expected array computed by hand. NCW tests are active only on CUDA GPUs.

  def testConv1DTransposeSingleStride(self):
    # [FUNCTION DOC]
    # Scenario: input batch two, width six, depth three; filter width three, output depth two;
    # stride one, SAME padding, output width six. Each output element should equal six for the
    # edge positions and nine for interior ones; asserted element by element.
    with self.cached_session():
      strides = [1, 1, 1]

      # Input, output: [batch, width, depth]
      x_shape = [2, 6, 3]
      y_shape = [2, 6, 2]

      # Filter: [kernel_width, output_depth, input_depth]
      f_shape = [3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv1d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      value = self.evaluate(output)

      for n in range(y_shape[0]):
        for w in range(y_shape[1]):
          for c in range(y_shape[2]):
            target = 2 * 3.0
            w_in = w > 0 and w < y_shape[1] - 1
            if w_in:
              target += 3.0
            self.assertAllClose(target, value[n, w, c])

  def testConv1DTransposeSame(self):
    # [FUNCTION DOC]
    # Scenario: input width four, stride two, SAME padding, output width eight. Expected value is
    # three, plus another three at interior positions divisible by the stride; asserted for every
    # element.
    with self.cached_session():
      strides = [1, 2, 1]

      # Input, output: [batch, width, depth]
      x_shape = [2, 4, 3]
      y_shape = [2, 8, 2]

      # Filter: [kernel_width, output_depth, input_depth]
      f_shape = [3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv1d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      value = self.evaluate(output)

      for n in range(x_shape[0]):
        for k in range(f_shape[1]):
          for w in range(y_shape[1]):
            target = 3.0
            # We add a case for locations divisible by the stride.
            w_in = w % strides[1] == 0 and w > 0 and w < y_shape[1] - 1
            if w_in:
              target += 3.0
            self.assertAllClose(target, value[n, w, k])

  def testConv1DTransposeValid(self):
    # [FUNCTION DOC]
    # Scenario: input width four, stride two, VALID padding, output width nine. Builds the
    # expected array by hand (interior values of three, doubled at interior stride-aligned
    # positions, borders copied from their neighbours) and asserts equality with the op output.
    # KNOWN ISSUE: the final statement in the loop assigns a slice of the expected array to
    # itself, which has no effect (dead code).
    with self.cached_session():
      strides = [1, 2, 1]

      # Input, output: [batch, width, depth]
      x_shape = [2, 4, 3]
      y_shape = [2, 9, 2]

      # Filter: [kernel_width, output_depth, input_depth]
      f_shape = [3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv1d_transpose(
          x, f, y_shape, strides=strides, padding="VALID")
      value = self.evaluate(output)

      cache_values = np.zeros(y_shape, dtype=np.float32)

      # The amount of padding added
      pad = 1

      for n in range(x_shape[0]):
        for k in range(f_shape[1]):
          for w in range(pad, y_shape[1] - pad):
            target = 3.0
            # We add a case for locations divisible by the stride.
            w_in = w % strides[1] == 0 and w > pad and w < y_shape[1] - 1 - pad
            if w_in:
              target += 3.0
            cache_values[n, w, k] = target

          # copy values in the border
          cache_values[n, 0, k] = cache_values[n, 1, k]
          cache_values[n, -1, k] = cache_values[n, -2, k]
          cache_values[n, :, k] = cache_values[n, :, k]

    self.assertAllClose(cache_values, value)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself immediately (bug reference for a GPU nightly build
    # failure), so the rest of the body never runs; also the random float64 arrays are converted
    # to float32 constants.
    # Intended scenario: seeded random input and filter, conv1d_transpose with stride two and SAME
    # padding, numeric-versus-analytic gradient error w.r.t. input and filter printed and asserted
    # below 5e-4.
    self.skipTest("b/262851489: Fix nightly build for GPU.")
    x_shape = [2, 4, 3]
    f_shape = [3, 2, 3]
    y_shape = [2, 8, 2]
    strides = [1, 2, 1]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    f_val = np.random.random_sample(f_shape).astype(np.float64)
    with self.cached_session():
      x = constant_op.constant(x_val, name="x", dtype=dtypes.float32)
      f = constant_op.constant(f_val, name="f", dtype=dtypes.float32)
      output = nn_ops.conv1d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      err = gradient_checker.compute_gradient_error([x, f], [x_shape, f_shape],
                                                    output, y_shape)
    print("conv1d_transpose gradient err = %g " % err)
    err_tolerance = 0.0005
    self.assertLess(err, err_tolerance)

  def testConv1DTransposeSingleStrideNCW(self):
    # [FUNCTION DOC]
    # Scenario: only runs (otherwise silently passes) when a CUDA GPU exists. Channels-first
    # input (batch two, depth three, width four), stride one, SAME padding, NCW data format.
    # Asserts each output element equals six at the edges and nine in the interior.
    # `NCW` data format is only supported for CUDA device.
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        strides = [1, 1, 1]

        # Input, output: [batch, depth, width]
        x_shape = [2, 3, 4]
        y_shape = [2, 2, 4]

        # Filter: [kernel_width, output_depth, input_depth]
        f_shape = [3, 2, 3]

        x = constant_op.constant(
            1.0, shape=x_shape, name="x", dtype=dtypes.float32)
        f = constant_op.constant(
            1.0, shape=f_shape, name="filter", dtype=dtypes.float32)

        output = nn_ops.conv1d_transpose(
            x, f, y_shape, strides=strides, padding="SAME", data_format="NCW")

        value = self.evaluate(output)
        for n in range(x_shape[0]):
          for k in range(f_shape[1]):
            for w in range(y_shape[2]):
              target = 2 * 3.0
              w_in = w > 0 and w < y_shape[2] - 1
              if w_in:
                target += 3.0
              self.assertAllClose(target, value[n, k, w])

  def testConv1DTransposeSameNCW(self):
    # [FUNCTION DOC]
    # Scenario: CUDA-only channels-first variant of the SAME test: width four, stride two, output
    # width eight, NCW format. Asserts three, plus three at interior stride-aligned positions.
    # `NCW` data format is only supported for CUDA device.
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        strides = [1, 1, 2]

        # Input, output: [batch, depth, width]
        x_shape = [2, 3, 4]
        y_shape = [2, 2, 8]

        # Filter: [kernel_width, output_depth, input_depth]
        f_shape = [3, 2, 3]

        x = constant_op.constant(
            1.0, shape=x_shape, name="x", dtype=dtypes.float32)
        f = constant_op.constant(
            1.0, shape=f_shape, name="filter", dtype=dtypes.float32)

        output = nn_ops.conv1d_transpose(
            x, f, y_shape, strides=strides, padding="SAME", data_format="NCW")

        value = self.evaluate(output)
        for n in range(x_shape[0]):
          for k in range(f_shape[1]):
            for w in range(y_shape[2]):
              target = 3.0
              # We add a case for locations divisible by the stride.
              w_in = w % strides[2] == 0 and w > 0 and w < y_shape[2] - 1
              if w_in:
                target += 3.0
              self.assertAllClose(target, value[n, k, w])

  def testConv1DTransposeValidNCW(self):
    # [FUNCTION DOC]
    # Scenario: CUDA-only channels-first variant of the VALID test: width four, stride two, output
    # width nine, NCW format. Compares against a hand-built expected array with copied borders.
    # KNOWN ISSUE: the last statement in the loop assigns a slice to itself (no-op, dead code).
    # `NCW` data format is only supported for CUDA device.
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        strides = [1, 1, 2]

        # Input, output: [batch, depth, width]
        x_shape = [2, 3, 4]
        y_shape = [2, 2, 9]

        # Filter: [kernel_width, output_depth, input_depth]
        f_shape = [3, 2, 3]

        x = constant_op.constant(
            1.0, shape=x_shape, name="x", dtype=dtypes.float32)
        f = constant_op.constant(
            1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
        output = nn_ops.conv1d_transpose(
            x, f, y_shape, strides=strides, padding="VALID", data_format="NCW")

        value = self.evaluate(output)
        cache_values = np.zeros(y_shape, dtype=np.float32)
        # The amount of padding added
        pad = 1
        for n in range(x_shape[0]):
          for k in range(f_shape[1]):
            for w in range(pad, y_shape[2] - pad):
              target = 3.0
              # We add a case for locations divisible by the stride.
              w_in = w % strides[2] == 0 and w > pad and \
                     w < y_shape[2] - 1 - pad
              if w_in:
                target += 3.0
              cache_values[n, k, w] = target

            # copy values in the border
            cache_values[n, k, 0] = cache_values[n, k, 1]
            cache_values[n, k, -1] = cache_values[n, k, -2]
            cache_values[n, k, :] = cache_values[n, k, :]

        self.assertAllClose(cache_values, value)


if __name__ == "__main__":
  test.main()
