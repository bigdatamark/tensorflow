# [MODULE DOC]
# Test module for the bucketize operation (mapping values to bucket indices given sorted
# boundaries) when run on the XLA test device. Covers integer, float and two-dimensional inputs
# plus error cases for unsorted boundaries and non-list boundaries. No import-time side effects.
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
"""Tests for bucketize_op."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BucketizationOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the bucketize operation. Each test builds a placeholder, applies bucketize
  # inside the XLA test scope and runs it in a session. Holds no state of its own.

  def testInt(self):
    # [FUNCTION DOC]
    # Bucketizes an integer vector with boundaries 0, 3, 8, 11 and asserts the bucket indices
    # (values equal to a boundary fall into the bucket above it; below 0 gives 0, above 11 gives 4).
    # Returns nothing.
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = math_ops._bucketize(p, boundaries=[0, 3, 8, 11])
      expected_out = [0, 1, 1, 2, 2, 3, 3, 4, 4]
      self.assertAllEqual(expected_out,
                          sess.run(op, {p: [-5, 0, 2, 3, 5, 8, 10, 11, 12]}))

  def testFloat(self):
    # [FUNCTION DOC]
    # Same as the integer test but with float32 inputs and float boundaries; asserts the same
    # expected bucket indices. Returns nothing.
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        op = math_ops._bucketize(p, boundaries=[0., 3., 8., 11.])
      expected_out = [0, 1, 1, 2, 2, 3, 3, 4, 4]
      self.assertAllEqual(
          expected_out,
          sess.run(op, {p: [-5., 0., 2., 3., 5., 8., 10., 11., 12.]}))

  def test2DInput(self):
    # [FUNCTION DOC]
    # Bucketizes a 2x5 float32 matrix with integer boundaries and asserts the output keeps the
    # 2x5 shape with per-element bucket indices. Returns nothing.
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        op = math_ops._bucketize(p, boundaries=[0, 3, 8, 11])
      expected_out = [[0, 1, 1, 2, 2], [3, 3, 4, 4, 1]]
      self.assertAllEqual(
          expected_out, sess.run(op,
                                 {p: [[-5, 0, 2, 3, 5], [8, 10, 11, 12, 0]]}))

  @test_util.disable_mlir_bridge("Error handling")
  def testInvalidBoundariesOrder(self):
    # [FUNCTION DOC]
    # Error-path test (disabled for the MLIR bridge because of error handling differences).
    # Builds bucketize with unsorted boundaries and expects running it to raise an
    # invalid-argument error mentioning that sorted boundaries were expected. Returns nothing.
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = math_ops._bucketize(p, boundaries=[0, 8, 3, 11])
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Expected sorted boundaries"):
        sess.run(op, {p: [-5, 0]})

  def testBoundariesNotList(self):
    # [FUNCTION DOC]
    # Passes a plain integer instead of a list as the boundaries and expects a type error whose
    # message says a list was expected; raised at graph construction time. Returns nothing.
    with self.session():
      with self.assertRaisesRegex(TypeError, "Expected list.*"):
        p = array_ops.placeholder(dtypes.int32)
        with self.test_scope():
          math_ops._bucketize(p, boundaries=0)


if __name__ == "__main__":
  test.main()
