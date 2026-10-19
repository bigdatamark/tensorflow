# [MODULE DOC]
# Tests for version 2 of the zero-out example op, including its gradient. Importing the gradient
# module registers the op's gradient function as an import-time side effect (hence the unused
# import). Contains one test case class and a standard main entry running the test runner.
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

"""Test for version 2 of the zero_out op."""

import tensorflow as tf


from tensorflow.examples.adding_an_op import zero_out_grad_2  # pylint: disable=unused-import
from tensorflow.examples.adding_an_op import zero_out_op_2


class ZeroOut2Test(tf.test.TestCase):
  # [CLASS DOC]
  # Test case covering version 2 of the zero-out example op (forward values and gradients).
  # Has no fixtures.

  def test(self):
    # [FUNCTION DOC]
    # Applies zero-out to the one-dimensional list five down to one and asserts only the first
    # element is kept, the rest being zero.
    result = zero_out_op_2.zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_2d(self):
    # [FUNCTION DOC]
    # Applies zero-out to a two-by-three matrix and asserts only the first element of the
    # flattened data (six) is kept and all other entries are zero.
    result = zero_out_op_2.zero_out([[6, 5, 4], [3, 2, 1]])
    self.assertAllEqual(result, [[6, 0, 0], [0, 0, 0]])

  def test_grad(self):
    # [FUNCTION DOC]
    # Builds a float vector five down to one and compares the theoretical and numerical gradients
    # of the zero-out op, asserting they are close.
    x = tf.constant([5, 4, 3, 2, 1], dtype=tf.float32)
    theoretical, numerical = tf.test.compute_gradient(zero_out_op_2.zero_out,
                                                      tuple([x]))
    self.assertAllClose(theoretical, numerical)

  def test_grad_2d(self):
    # [FUNCTION DOC]
    # Same gradient check as the one-dimensional case but with a float two-by-three matrix;
    # asserts theoretical and numerical gradients are close.
    x = tf.constant([[6, 5, 4], [3, 2, 1]], dtype=tf.float32)
    theoretical, numerical = tf.test.compute_gradient(zero_out_op_2.zero_out,
                                                      tuple([x]))
    self.assertAllClose(theoretical, numerical)


if __name__ == '__main__':
  tf.test.main()
