# [MODULE DOC]
# Test module for the single-input identity operation: values, strings, static shape inference
# and composite (sparse) tensor support. Defines one test case class and has no import-time
# side effects beyond imports.
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
"""Tests for IdentityOp."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class IdentityOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the identity operation on dense, string and sparse inputs.
  # No shared fixtures.

  def testInt32_6(self):
    # [FUNCTION DOC]
    # Test scenario: apply identity to a plain list of six integers and evaluate. Expects the
    # result to equal the input. Takes no inputs and returns nothing.
    value = self.evaluate(array_ops.identity([1, 2, 3, 4, 5, 6]))
    self.assertAllEqual(np.array([1, 2, 3, 4, 5, 6]), value)

  def testInt32_2_3(self):
    # [FUNCTION DOC]
    # Test scenario: identity applied to a two-by-three integer constant. Expects the same
    # values and layout back. Takes no inputs and returns nothing.
    inp = constant_op.constant([10, 20, 30, 40, 50, 60], shape=[2, 3])
    value = self.evaluate(array_ops.identity(inp))
    self.assertAllEqual(np.array([[10, 20, 30], [40, 50, 60]]), value)

  def testString(self):
    # [FUNCTION DOC]
    # Test scenario: identity applied to a list of six byte strings. Expects equal output.
    # Takes no inputs and returns nothing.
    source = [b"A", b"b", b"C", b"d", b"E", b"f"]
    value = self.evaluate(array_ops.identity(source))
    self.assertAllEqual(source, value)

  def testIdentityShape(self):
    # [FUNCTION DOC]
    # Test scenario: in a cached session, check that the static shape of the identity output
    # is two-by-three whether the input is a tensor, a nested list or a numpy array.
    # Verifies shape inference only. Takes no inputs and returns nothing.
    with self.cached_session():
      shape = [2, 3]
      array_2x3 = [[1, 2, 3], [6, 5, 4]]
      tensor = constant_op.constant(array_2x3)
      self.assertEqual(shape, tensor.get_shape())
      self.assertEqual(shape, array_ops.identity(tensor).get_shape())
      self.assertEqual(shape, array_ops.identity(array_2x3).get_shape())
      self.assertEqual(shape,
                       array_ops.identity(np.array(array_2x3)).get_shape())

  def testCompositeTensor(self):
    # [FUNCTION DOC]
    # Test scenario: identity applied to a sparse tensor with a single entry at index three in
    # a dense shape of one hundred. Expects the copy to have equal indices, values and dense
    # shape, showing composite tensors are passed through component-wise. Returns nothing.
    original = sparse_tensor.SparseTensor([[3]], [1.0], [100])
    copied = array_ops.identity(original)
    self.assertAllEqual(original.indices, copied.indices)
    self.assertAllEqual(original.values, copied.values)
    self.assertAllEqual(original.dense_shape, copied.dense_shape)


if __name__ == "__main__":
  test.main()
