# [MODULE DOC]
# Test module for the internal bucketize op, which maps values to bucket indices given sorted
# boundaries. One test class covers integer, float, empty, two-dimensional and invalid inputs.
# No import-time side effects; running as a script starts the test runner.
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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BucketizationOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the internal bucketize op in the math ops module, which maps each input value
  # to the index of the bucket defined by a sorted list of boundaries. Covers integer, float,
  # empty, two-dimensional and invalid-argument cases.

  def testInt(self):
    # [FUNCTION DOC]
    # Buckets integer values against boundaries zero, three, eight and eleven and asserts the
    # bucket indices (a value equal to a boundary falls in the bucket above it).
    op = math_ops._bucketize(
        constant_op.constant([-5, 0, 2, 3, 5, 8, 10, 11, 12]),
        boundaries=[0, 3, 8, 11])
    expected_out = [0, 1, 1, 2, 2, 3, 3, 4, 4]
    with self.session():
      self.assertAllEqual(expected_out, self.evaluate(op))

  def testEmptyFloat(self):
    # [FUNCTION DOC]
    # Buckets an empty float32 tensor of shape zero by three with no boundaries and asserts the
    # result is an empty array of the same shape.
    op = math_ops._bucketize(
        array_ops.zeros([0, 3], dtype=dtypes.float32), boundaries=[])
    expected_out = np.zeros([0, 3], dtype=np.float32)
    with self.session():
      self.assertAllEqual(expected_out, self.evaluate(op))

  def testFloat(self):
    # [FUNCTION DOC]
    # Same scenario as the integer test but with float values and float boundaries; asserts the
    # same bucket indices.
    op = math_ops._bucketize(
        constant_op.constant([-5., 0., 2., 3., 5., 8., 10., 11., 12.]),
        boundaries=[0., 3., 8., 11.])
    expected_out = [0, 1, 1, 2, 2, 3, 3, 4, 4]
    with self.session():
      self.assertAllEqual(expected_out, self.evaluate(op))

  def test2DInput(self):
    # [FUNCTION DOC]
    # Buckets a two-row integer matrix and asserts the output keeps the shape with per-element
    # bucket indices.
    op = math_ops._bucketize(
        constant_op.constant([[-5, 0, 2, 3, 5], [8, 10, 11, 12, 0]]),
        boundaries=[0, 3, 8, 11])
    expected_out = [[0, 1, 1, 2, 2], [3, 3, 4, 4, 1]]
    with self.session():
      self.assertAllEqual(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def testInvalidBoundariesOrder(self):
    # [FUNCTION DOC]
    # Graph-mode test: boundaries that are not sorted ascending must make evaluation raise an
    # invalid-argument error whose message says sorted boundaries were expected.
    op = math_ops._bucketize(
        constant_op.constant([-5, 0]), boundaries=[0, 8, 3, 11])
    with self.session():
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Expected sorted boundaries"):
        self.evaluate(op)

  def testBoundariesNotList(self):
    # [FUNCTION DOC]
    # Passing a plain number instead of a list as boundaries must raise a type error whose
    # message begins by saying a list was expected.
    with self.assertRaisesRegex(TypeError, "Expected list.*"):
      math_ops._bucketize(constant_op.constant([-5, 0]), boundaries=0)


if __name__ == "__main__":
  test.main()
