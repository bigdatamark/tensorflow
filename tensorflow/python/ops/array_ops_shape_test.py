# [MODULE DOC]
# Test module checking that the shape and size array operations produce 64-bit integer results
# when the global "shape default is int64" configuration flag is switched on. It covers both the
# version 2 and the legacy (version 1) entry points of each operation. The flag is expected to be
# enabled by the test build target, not by this file. Has no import-time side effects; running it
# as a script starts the standard test runner.
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
"""Tests for shape op int64 output."""

from tensorflow.core.config import flags
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ArrayOpShapeSizeTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the dtype of results from the shape and size operations (both the version 2
  # and legacy variants) under the shape-default-int64 flag. Holds no fixtures or state; each test
  # asserts the flag is on and then checks the result dtype is 64-bit integer.

  def testShapeInt64Flag(self):
    # [FUNCTION DOC]
    # Test: asserts the shape-default-int64 flag is enabled, calls the version 2 shape operation on
    # a zeros tensor of shape 1 by 2, and expects the resulting shape tensor to have 64-bit
    # integer dtype. Takes no inputs, returns nothing; fails the test on any mismatch.
    # The tf_shape_default_int64 flag should be set when this test runs
    self.assertTrue(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.shape_v2(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int64)

  def testShapeInt64FlagTf1(self):
    # [FUNCTION DOC]
    # Test: same as the version 2 shape test but uses the legacy shape operation; asserts the
    # int64 flag is on and that the shape of a 1 by 2 zeros tensor has 64-bit integer dtype.
    # Takes no inputs, returns nothing.
    # The tf_shape_default_int64 flag should be set when this test runs
    self.assertTrue(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.shape(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int64)

  def testSizeInt64Flag(self):
    # [FUNCTION DOC]
    # Test: asserts the shape-default-int64 flag is enabled, calls the version 2 size operation on
    # a 1 by 2 zeros tensor, and expects the resulting scalar to have 64-bit integer dtype.
    # Takes no inputs, returns nothing.
    # The tf_shape_default_int64 flag should be set when this test runs
    self.assertTrue(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.size_v2(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int64)

  def testSizeInt64FlagTf1(self):
    # [FUNCTION DOC]
    # Test: same as the version 2 size test but uses the legacy size operation; asserts the flag
    # is on and that the size of a 1 by 2 zeros tensor has 64-bit integer dtype. Takes no inputs,
    # returns nothing.
    # The tf_shape_default_int64 flag should be set when this test runs
    self.assertTrue(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.size(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int64)


if __name__ == "__main__":
  test.main()
