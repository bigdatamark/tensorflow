# [MODULE DOC]
# Test module for the multi-input identity operation (identity_n), which returns each of its
# inputs unchanged. Covers mixed integer/string lists, shape preservation and static shape
# inference. Defines one test case class and has no import-time side effects beyond imports.
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
"""Tests for IdentityNOp."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class IdentityNOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the identity_n operation with several tensors of differing types/shapes.
  # No shared fixtures.

  def testInt32String_6(self):
    # [FUNCTION DOC]
    # Test scenario: pass one list of six integers and one list of seven byte strings to the
    # identity_n operation in a single call and evaluate. Expects the two outputs to equal the
    # respective inputs, showing differently typed and differently sized inputs are accepted.
    # Takes no inputs and returns nothing.
    value0, value1 = self.evaluate(
        array_ops.identity_n([[1, 2, 3, 4, 5, 6],
                              [b"a", b"b", b"C", b"d", b"E", b"f", b"g"]]))

    self.assertAllEqual(np.array([1, 2, 3, 4, 5, 6]), value0)
    self.assertAllEqual(
        np.array([b"a", b"b", b"C", b"d", b"E", b"f", b"g"]), value1)

  def testInt32_shapes(self):
    # [FUNCTION DOC]
    # Test scenario: three integer constants with shapes two-by-three, three-by-two and
    # five-by-three go through identity_n together. Expects each output to keep its input's
    # values and shape. Takes no inputs and returns nothing.
    inp0 = constant_op.constant([10, 20, 30, 40, 50, 60], shape=[2, 3])
    inp1 = constant_op.constant([11, 21, 31, 41, 51, 61], shape=[3, 2])
    inp2 = constant_op.constant(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], shape=[5, 3])
    value0, value1, value2 = self.evaluate(
        array_ops.identity_n([inp0, inp1, inp2]))

    self.assertAllEqual(np.array([[10, 20, 30], [40, 50, 60]]), value0)
    self.assertAllEqual(np.array([[11, 21], [31, 41], [51, 61]]), value1)
    self.assertAllEqual(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
        value2)

  def testString(self):
    # [FUNCTION DOC]
    # Test scenario: identity_n is given a list holding a single list of six byte strings.
    # Expects the single output to equal the input. Takes no inputs and returns nothing.
    source = [b"A", b"b", b"C", b"d", b"E", b"f"]
    [value] = self.evaluate(array_ops.identity_n([source]))
    self.assertAllEqual(source, value)

  def testIdentityShape(self):
    # [FUNCTION DOC]
    # Test scenario: in a cached session, build a two-by-three tensor and check that identity_n
    # applied to a one-element list holding the tensor, or holding the equivalent nested list,
    # reports the static shape two-by-three. Verifies static shape inference, not values.
    # Takes no inputs and returns nothing.
    with self.cached_session():
      shape = [2, 3]
      array_2x3 = [[1, 2, 3], [6, 5, 4]]
      tensor = constant_op.constant(array_2x3)
      self.assertEqual(shape, tensor.get_shape())
      self.assertEqual(shape, array_ops.identity_n([tensor])[0].get_shape())
      self.assertEqual(shape, array_ops.identity_n([array_2x3])[0].get_shape())


if __name__ == "__main__":
  test.main()
