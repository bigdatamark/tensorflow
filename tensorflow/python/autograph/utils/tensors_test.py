# [MODULE DOC]
# Unit tests for the autograph tensors utility module (tensor array, tensor list and range
# tensor predicates). Run as a standard TensorFlow test file; no import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensors module."""

from tensorflow.python.autograph.utils import tensors
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class TensorsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the predicates in the autograph tensors module. Provides small helper
  # methods that build a tensor array, an empty tensor list and a plain list of tensors.

  def _simple_tensor_array(self):
    # [FUNCTION DOC]
    # HELPER: builds a fixture for the tests in this class.
    # Returns a new integer tensor array of size three.
    return tensor_array_ops.TensorArray(dtypes.int32, size=3)

  def _simple_tensor_list(self):
    # [FUNCTION DOC]
    # HELPER: builds a fixture for the tests in this class.
    # Returns an empty tensor list with integer elements of shape one.
    return list_ops.empty_tensor_list(
        element_shape=constant_op.constant([1]), element_dtype=dtypes.int32)

  def _simple_list_of_tensors(self):
    # [FUNCTION DOC]
    # HELPER: builds a fixture for the tests in this class.
    # Returns a plain Python list holding two scalar integer constant tensors.
    return [constant_op.constant(1), constant_op.constant(2)]

  def test_is_tensor_array(self):
    # [FUNCTION DOC]
    # Asserts the tensor array predicate is true for a tensor array and false for a tensor list,
    # a constant tensor, a list of tensors and a null value.
    self.assertTrue(tensors.is_tensor_array(self._simple_tensor_array()))
    self.assertFalse(tensors.is_tensor_array(self._simple_tensor_list()))
    self.assertFalse(tensors.is_tensor_array(constant_op.constant(1)))
    self.assertFalse(tensors.is_tensor_array(self._simple_list_of_tensors()))
    self.assertFalse(tensors.is_tensor_array(None))

  def test_is_tensor_list(self):
    # [FUNCTION DOC]
    # Asserts the tensor list predicate is false for a tensor array, true for an empty tensor
    # list, and false for a constant tensor, a list of tensors and a null value.
    self.assertFalse(tensors.is_tensor_list(self._simple_tensor_array()))
    self.assertTrue(tensors.is_tensor_list(self._simple_tensor_list()))
    self.assertFalse(tensors.is_tensor_list(constant_op.constant(1)))
    self.assertFalse(tensors.is_tensor_list(self._simple_list_of_tensors()))
    self.assertFalse(tensors.is_tensor_list(None))

  def is_range_tensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name lacks the test prefix, so the test runner never collects or runs it.
    # Intended to assert the range tensor predicate is true for range tensors built with one,
    # two and three arguments, and false for a null value and a constant built from a range.
    self.assertTrue(tensors.is_range_tensor(math_ops.range(1)))
    self.assertTrue(tensors.is_range_tensor(math_ops.range(1, 2)))
    self.assertTrue(tensors.is_range_tensor(math_ops.range(1, 2, 3)))
    self.assertFalse(tensors.is_range_tensor(None))
    self.assertFalse(tensors.is_range_tensor(constant_op.constant(range(1))))


if __name__ == '__main__':
  test.main()
