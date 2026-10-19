# [MODULE DOC]
# Functional tests of the reduction ops (sum, max, mean, logical-and) on big inputs and a wide
# range of slice shapes designed to exercise every kernel invocation path, including a
# float32 versus bfloat16 mean comparison. Exposes one test class and a base class. No
# import-time side effects; tests run only in graph mode.
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
"""Functional tests for reduction ops."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BaseReductionTest(test.TestCase):
  # [CLASS DOC]
  # HELPER: abstract base test case declaring the reduction hook that subclasses must provide.
  # Holds no state.

  def _tf_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the only subclass here defines specific reduction wrappers and never overrides
    # this hook, so it is effectively unused.
    # HELPER: abstract hook for the reduction under test. Always raises a not-implemented error;
    # subclasses are meant to override it. Inputs: tensor, axes, keep-dimensions flag.
    raise NotImplementedError()


class BigReductionTest(BaseReductionTest):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring mentions only sum and boolean all, but max and mean are tested too.
  # Test case running reductions (sum, max, mean, logical-and) over large and many differently
  # shaped inputs to hit all kernel code paths. Provides thin wrapper methods for each reduction.
  """Test reductions for sum and boolean all over a wide range of shapes."""

  def _tf_reduce_max(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: wraps the maximum reduction. Inputs: tensor, axes to reduce, keep-dimensions flag.
    # Returns the reduced tensor.
    return math_ops.reduce_max(x, reduction_axes, keepdims)

  def _tf_reduce_all(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: wraps the logical-and reduction. Inputs: boolean tensor, axes, keep-dimensions flag.
    # Returns the reduced tensor.
    return math_ops.reduce_all(x, reduction_axes, keepdims)

  def _tf_reduce_mean(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: wraps the mean reduction. Inputs: tensor, axes, keep-dimensions flag. Returns the
    # reduced tensor.
    return math_ops.reduce_mean(x, reduction_axes, keepdims)

  def _tf_reduce_sum(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: wraps the sum reduction. Inputs: tensor, axes, keep-dimensions flag. Returns the
    # reduced tensor.
    return math_ops.reduce_sum(x, reduction_axes, keepdims)

  @test_util.run_deprecated_v1
  def testFloat32Bfloat16Mean(self):
    # [FUNCTION DOC]
    # Graph-mode CPU test: takes the full mean of a random 4105x4105 array as float32 and as its
    # bfloat16 cast (result cast back to float32), and asserts the two results are close. Checks
    # that the bfloat16 mean does not lose accuracy badly on large inputs. Returns nothing.
    arrfp32 = np.random.normal(size=[4105, 4105]).astype(np.float32)
    arrbf16 = arrfp32.astype(dtypes.bfloat16.as_numpy_dtype)
    with self.session(graph=ops.Graph(), use_gpu=False) as sess:
      arrfp32_placeholder = array_ops.placeholder(
          dtype=np.float32, shape=(4105, 4105)
      )
      arrbf16_placeholder = array_ops.placeholder(
          dtype=dtypes.bfloat16.as_numpy_dtype, shape=(4105, 4105)
      )
      tf_full_mean_fp32 = self._tf_reduce_mean(
          arrfp32_placeholder, [0, 1], False
      )
      tf_full_mean_bf16 = self._tf_reduce_mean(
          arrbf16_placeholder, [0, 1], False
      )
      tf_full_mean_bf16_cast = math_ops.cast(tf_full_mean_bf16, dtypes.float32)

      tf_out_full_f, tf_out_full_b = sess.run(
          [tf_full_mean_fp32, tf_full_mean_bf16_cast],
          {arrfp32_placeholder: arrfp32, arrbf16_placeholder: arrbf16},
      )
    self.assertAllClose(tf_out_full_f, tf_out_full_b)

  @test_util.run_deprecated_v1
  def testFloat32Sum(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the 3-D part the reductions actually performed are means, not sums (the mean
    # helper is called), so the sum kernel is not exercised there.
    # Graph-mode test of sum reduction on all-ones float32 arrays: a 2-D array sliced to many
    # sizes around kernel boundaries (row, column and full sums), then a 3-D array sliced to many
    # sizes. Expected values are the slice sizes (or ones).
    # Returns nothing.
    # make sure we test all possible kernel invocations
    # logic is the same for all ops, test just float32 for brevity
    arr_ = np.ones([4097, 4097], dtype=np.float32)
    for size_x in [
        1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025,
        4096, 4097
    ]:
      for size_y in [
          1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025,
          4096, 4097
      ]:
        arr = arr_[0:size_x, 0:size_y]
        col_sum = np.ones([size_y], dtype=np.float32) * size_x
        row_sum = np.ones([size_x], dtype=np.float32) * size_y
        full_sum = np.ones([], dtype=np.float32) * size_x * size_y

        with self.session(graph=ops.Graph(), use_gpu=True) as sess:
          arr_placeholder = array_ops.placeholder(dtype=np.float32,
                                                  shape=(size_x, size_y))
          tf_row_sum = self._tf_reduce_sum(arr_placeholder, 1, False)
          tf_col_sum = self._tf_reduce_sum(arr_placeholder, 0, False)
          tf_full_sum = self._tf_reduce_sum(arr_placeholder, [0, 1], False)
          tf_out_row, tf_out_col, tf_out_full = sess.run(
              [tf_row_sum, tf_col_sum, tf_full_sum], {arr_placeholder: arr})
        self.assertAllClose(col_sum, tf_out_col)
        self.assertAllClose(row_sum, tf_out_row)
        self.assertAllClose(full_sum, tf_out_full)

    arr_ = np.ones([130, 130, 130], dtype=np.float32)
    for size_x in range(1, 130, 13):
      for size_y in range(1, 130, 13):
        for size_z in range(1, 130, 13):
          arr = arr_[0:size_x, 0:size_y, 0:size_z]
          sum_y = np.ones([size_x, size_z], dtype=np.float32)
          sum_xz = np.ones([size_y], dtype=np.float32)

          with self.session(graph=ops.Graph(), use_gpu=True) as sess:
            arr_placeholder = array_ops.placeholder(
                dtype=np.float32, shape=(size_x, size_y, size_z))
            tf_sum_xz = self._tf_reduce_mean(arr_placeholder, [0, 2], False)
            tf_sum_y = self._tf_reduce_mean(arr_placeholder, 1, False)
            tf_out_sum_xz, tf_out_sum_y = sess.run([tf_sum_xz, tf_sum_y],
                                                   {arr_placeholder: arr})
          self.assertAllClose(sum_y, tf_out_sum_y)
          self.assertAllClose(sum_xz, tf_out_sum_xz)

  @test_util.run_deprecated_v1
  def testFloat32Max(self):
    # [FUNCTION DOC]
    # Graph-mode test of max reduction on random negative float32 arrays: a 2-D array sliced to
    # many sizes around kernel boundaries (row, column and full reductions) and a 3-D array
    # sliced to many sizes (reductions over axes zero and two, and axis one). Compares each
    # result with the numpy maximum. Returns nothing.
    # make sure we test all possible kernel invocations
    # logic is the same for all ops, test just float32 for brevity
    arr_ = np.random.uniform(
        low=-3, high=-1, size=[4105, 4105]).astype(np.float32)
    for size_x in [
        1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025,
        4096, 4097
    ]:
      for size_y in [
          1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025,
          4096, 4097
      ]:
        arr = arr_[0:size_x, 0:size_y]
        col_max = np.max(arr, axis=0)
        row_max = np.max(arr, axis=1)
        full_max = np.max(col_max)

        with self.session(graph=ops.Graph(), use_gpu=True) as sess:
          arr_placeholder = array_ops.placeholder(dtype=np.float32,
                                                  shape=(size_x, size_y))
          tf_row_max = self._tf_reduce_max(arr_placeholder, 1, False)
          tf_col_max = self._tf_reduce_max(arr_placeholder, 0, False)
          tf_full_max = self._tf_reduce_max(arr_placeholder, [0, 1], False)
          tf_out_row, tf_out_col, tf_out_full = sess.run(
              [tf_row_max, tf_col_max, tf_full_max], {arr_placeholder: arr})
        self.assertAllClose(col_max, tf_out_col)
        self.assertAllClose(row_max, tf_out_row)
        self.assertAllClose(full_max, tf_out_full)

    arr_ = np.random.uniform(
        low=-3, high=-1, size=[130, 130, 130]).astype(np.float32)
    for size_x in range(1, 130, 13):
      for size_y in range(1, 130, 13):
        for size_z in range(1, 130, 13):
          arr = arr_[0:size_x, 0:size_y, 0:size_z]
          sum_y = np.max(arr, axis=1)
          sum_xz = np.max(arr, axis=(0, 2))

          with self.session(graph=ops.Graph(), use_gpu=True) as sess:
            arr_placeholder = array_ops.placeholder(
                dtype=np.float32, shape=(size_x, size_y, size_z))
            tf_sum_xz = self._tf_reduce_max(arr_placeholder, [0, 2], False)
            tf_sum_y = self._tf_reduce_max(arr_placeholder, 1, False)
            tf_out_sum_xz, tf_out_sum_y = sess.run(
                [tf_sum_xz, tf_sum_y], {arr_placeholder: arr})
          self.assertAllClose(sum_y, tf_out_sum_y)
          self.assertAllClose(sum_xz, tf_out_sum_xz)

  @test_util.run_deprecated_v1
  def testBooleanAll(self):
    # [FUNCTION DOC]
    # Graph-mode test of logical-and reduction on all-true boolean arrays: first a 2-D array
    # sliced to many sizes around kernel boundaries, reducing over rows, columns and both axes;
    # then a 3-D array sliced to many sizes, reducing over the first and last axes and over the
    # middle axis. Each result from a CPU/GPU session is compared to an all-true expectation.
    # Returns nothing.
    # make sure we test all possible kernel invocations
    # test operation where T(0) is not the identity
    arr_ = np.ones([4097, 4097], dtype=np.bool_)
    for size_x in [
        1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025,
        4096, 4097
    ]:
      for size_y in [
          1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 128, 131, 256, 263, 1024, 1025,
          4096, 4097
      ]:
        arr = arr_[0:size_x, 0:size_y]
        col_sum = np.ones([size_y], dtype=np.bool_)
        row_sum = np.ones([size_x], dtype=np.bool_)
        full_sum = np.ones([1], dtype=np.bool_).reshape([])

        with self.session(graph=ops.Graph(), use_gpu=True) as sess:
          arr_placeholder = array_ops.placeholder(
              dtype=np.bool_, shape=(size_x, size_y))
          tf_row_sum = self._tf_reduce_all(arr_placeholder, 1, False)
          tf_col_sum = self._tf_reduce_all(arr_placeholder, 0, False)
          tf_full_sum = self._tf_reduce_all(arr_placeholder, [0, 1], False)
          tf_out_row, tf_out_col, tf_out_full = sess.run(
              [tf_row_sum, tf_col_sum, tf_full_sum], {arr_placeholder: arr})
        self.assertAllClose(col_sum, tf_out_col)
        self.assertAllClose(row_sum, tf_out_row)
        self.assertAllClose(full_sum, tf_out_full)

    arr_ = np.ones([130, 130, 130], dtype=np.bool_)
    for size_x in range(1, 130, 13):
      for size_y in range(1, 130, 13):
        for size_z in range(1, 130, 13):
          arr = arr_[0:size_x, 0:size_y, 0:size_z]
          sum_y = np.ones([size_x, size_z], dtype=np.bool_)
          sum_xz = np.ones([size_y], dtype=np.bool_)

          with self.session(graph=ops.Graph(), use_gpu=True) as sess:
            arr_placeholder = array_ops.placeholder(
                dtype=np.bool_, shape=(size_x, size_y, size_z))
            tf_sum_xz = self._tf_reduce_all(arr_placeholder, [0, 2], False)
            tf_sum_y = self._tf_reduce_all(arr_placeholder, 1, False)
            tf_out_sum_xz, tf_out_sum_y = sess.run(
                [tf_sum_xz, tf_sum_y], {arr_placeholder: arr})
          self.assertAllClose(sum_y, tf_out_sum_y)
          self.assertAllClose(sum_xz, tf_out_sum_xz)


if __name__ == "__main__":
  test.main()
