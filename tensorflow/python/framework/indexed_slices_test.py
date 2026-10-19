# [MODULE DOC]
# Unit tests for the IndexedSlices composite tensor type, specifically its support for the
# composite-tensor gradient protocol (flattening to and rebuilding from gradient components).
# Defines one test class and a standard test entry point; no import-time side effects beyond imports.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for third_party.tensorflow.python.framework.indexed_slices."""

from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import indexed_slices
from tensorflow.python.platform import test


class IndexedSlicesTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the IndexedSlices type. Currently holds a single test of its integration with
  # the composite-tensor gradient helpers. No shared fixtures.

  def testCompositeTensorGradient(self):
    # [FUNCTION DOC]
    # Builds an IndexedSlices with one row of values, index one and dense shape three by two, then
    # flattens it for gradient purposes and asserts the flat components equal the original list
    # holding that single IndexedSlices (it is its own gradient component). Then replaces the
    # component with a plain list of two floats and asserts the rebuilt structure is a list holding
    # exactly that replacement. Takes no arguments, returns nothing, no side effects.
    i = indexed_slices.IndexedSlices(values=constant_op.constant([[1., 2.]]),
                                     indices=constant_op.constant([1]),
                                     dense_shape=[3, 2])
    gradient_components = (
        composite_tensor_gradient.get_flat_tensors_for_gradients([i]))
    self.assertAllEqual(gradient_components, [i])

    t = [3., 4.]
    result = (
        composite_tensor_gradient.replace_flat_tensors_for_gradients([i], [t]))
    self.assertAllEqual(result, [t])


if __name__ == '__main__':
  test.main()
