# [MODULE DOC]
# Test module for the XLA implementation of the sorted-sequence search op (lower and upper bound
# insertion positions). One test class compares against numpy for one-dimensional inputs and
# against documented examples for two-dimensional inputs. No import-time side effects beyond
# imports; as a script it starts the test runner.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test for XLA implementation of tf.searchsorted."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SearchSorteddOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class name contains a typo (doubled letter d); it has no functional effect.
  # XLA test case for the searchsorted op: checks insertion positions for both the left (lower
  # bound) and right (upper bound) sides. Holds no state of its own.

  def test1D(self):
    # [FUNCTION DOC]
    # Seeds numpy randomness with one. For both sides and for float and integer types, draws ten
    # random query values and twenty random values sorted into a sequence, computes the numpy
    # insertion positions, runs the device op in a session and asserts exact equality.
    # Returns nothing.
    # Test against NumPy implementation (which is 1D only).
    np.random.seed(1)
    for side in ['left', 'right']:
      for dtype in [np.float32, np.int32]:
        values = np.random.uniform(
            low=-1000, high=1000, size=(10,)).astype(dtype)
        unsorted = np.random.uniform(
            low=-1000, high=1000, size=(20,)).astype(dtype)

        sorted_sequence = np.sort(unsorted)
        np_ans = np.searchsorted(sorted_sequence, values, side=side)

        with self.session() as session:
          with self.test_scope():
            tf_ans = array_ops.searchsorted(sorted_sequence, values, side=side)
          tf_out = session.run(tf_ans)
          self.assertAllEqual(np_ans, tf_out)

  def _test2DExample(self, dtype, side, sorted_sequence, values, correct_ans):
    # [FUNCTION DOC]
    # HELPER: Shared checker for the two-dimensional example tests.
    # KNOWN ISSUE: the dtype parameter is accepted but never used.
    # Inputs: side is the string left or right; sorted_sequence is a numpy matrix with each row
    # sorted; values is a numpy matrix of queries; correct_ans is the expected positions. Runs
    # the device search in the test scope within a session and asserts exact equality with
    # correct_ans. Returns nothing.

    with self.session() as session:
      with self.test_scope():
        tf_ans = array_ops.searchsorted(sorted_sequence, values, side=side)
      tf_out = session.run(tf_ans)
      self.assertAllEqual(correct_ans, tf_out)

  def testLowerBound2DExample(self):
    # [FUNCTION DOC]
    # For every float and integer type supported on the device, searches the rows of the
    # documentation example matrices with the left side and expects the lower-bound positions
    # (one, two, two and zero, one, five). Returns nothing.
    # 2D TensorFlow documentation example.
    for dtype in self.float_types | self.int_types:
      sorted_sequence = np.array([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]], dtype)
      values = np.array([[2, 4, 9], [0, 2, 6]], dtype)
      correct_ans = np.array([[1, 2, 2], [0, 1, 5]], dtype)
      self._test2DExample(dtype, 'left', sorted_sequence, values, correct_ans)

  def testUpperBound2DExample(self):
    # [FUNCTION DOC]
    # For every float and integer type supported on the device, searches the rows of the
    # documentation example matrices with the right side and expects the upper-bound positions
    # (one, two, four and zero, two, five). Returns nothing.
    # 2D TensorFlow documentation example.
    for dtype in self.float_types | self.int_types:
      sorted_sequence = np.array([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]], dtype)
      values = np.array([[2, 4, 9], [0, 2, 6]], dtype)
      correct_ans = np.array([[1, 2, 4], [0, 2, 5]], dtype)
      self._test2DExample(dtype, 'right', sorted_sequence, values, correct_ans)


if __name__ == '__main__':
  test.main()
