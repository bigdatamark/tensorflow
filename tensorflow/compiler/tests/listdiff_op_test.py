# [MODULE DOC]
# Tests for the XLA list-difference operator (elements of x not present in y, plus their
# positions in x), run on the XLA test device with int32 and int64 values and index types.
# Contains one test class; running the file directly starts the test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA listdiff operator."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ListDiffTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests the list-difference op on the XLA test device with fixed and random inputs.

  def _testListDiff(self, x, y, out, idx):
    # [FUNCTION DOC]
    # HELPER: shared check used by all tests.
    # Inputs: x and y, lists of integers; out, expected values of x not in y (order and
    # duplicates kept); idx, expected positions of those values in x. For each combination of
    # value type and index type (int32 or int64) it runs the op on the test device and asserts
    # the outputs equal the expectations and that both results are rank one. Returns nothing.
    for dtype in [dtypes.int32, dtypes.int64]:
      for index_dtype in [dtypes.int32, dtypes.int64]:
        with self.session():
          x_tensor = ops.convert_to_tensor(x, dtype=dtype)
          y_tensor = ops.convert_to_tensor(y, dtype=dtype)
          with self.test_scope():
            out_tensor, idx_tensor = array_ops.listdiff(
                x_tensor, y_tensor, out_idx=index_dtype)
            tf_out, tf_idx = self.evaluate([out_tensor, idx_tensor])
        self.assertAllEqual(out, tf_out)
        self.assertAllEqual(idx, tf_idx)
        self.assertEqual(1, out_tensor.get_shape().ndims)
        self.assertEqual(1, idx_tensor.get_shape().ndims)

  def testBasic1(self):
    # [FUNCTION DOC]
    # Removes two leading values from a four-element list; expects the last two and indices 2, 3.
    self._testListDiff(x=[1, 2, 3, 4], y=[1, 2], out=[3, 4], idx=[2, 3])

  def testBasic2(self):
    # [FUNCTION DOC]
    # Removes a single value from the middle; expects the remaining three with their indices.
    self._testListDiff(x=[1, 2, 3, 4], y=[2], out=[1, 3, 4], idx=[0, 2, 3])

  def testBasic3(self):
    # [FUNCTION DOC]
    # Removes values from an unordered list; expects the output to keep the original order.
    self._testListDiff(x=[1, 4, 3, 2], y=[4, 2], out=[1, 3], idx=[0, 2])

  def testDuplicates(self):
    # [FUNCTION DOC]
    # Input has repeated values; expects retained duplicates to all appear, in original order.
    self._testListDiff(x=[1, 2, 4, 3, 2, 3, 3, 1],
                       y=[4, 2],
                       out=[1, 3, 3, 3, 1],
                       idx=[0, 3, 5, 6, 7])

  def testRandom(self):
    # [FUNCTION DOC]
    # Ten random trials: random-length lists of small integers; the expected outputs are
    # computed with plain Python filtering and compared with the op.
    num_random_tests = 10
    int_low = -7
    int_high = 8
    max_size = 50
    for _ in range(num_random_tests):
      x_size = np.random.randint(max_size + 1)
      x = np.random.randint(int_low, int_high, size=x_size)
      y_size = np.random.randint(max_size + 1)
      y = np.random.randint(int_low, int_high, size=y_size)
      out_idx = [(entry, pos) for pos, entry in enumerate(x) if entry not in y]
      if out_idx:
        out, idx = map(list, zip(*out_idx))
      else:
        out = []
        idx = []
      self._testListDiff(list(x), list(y), out, idx)

  def testFullyOverlapping(self):
    # [FUNCTION DOC]
    # y equals x; expects empty outputs.
    self._testListDiff(x=[1, 2, 3, 4], y=[1, 2, 3, 4], out=[], idx=[])

  def testNonOverlapping(self):
    # [FUNCTION DOC]
    # x and y share no values; expects all of x and indices 0 to 3.
    self._testListDiff(x=[1, 2, 3, 4],
                       y=[5, 6],
                       out=[1, 2, 3, 4],
                       idx=[0, 1, 2, 3])

  def testEmptyX(self):
    # [FUNCTION DOC]
    # Empty x; expects empty outputs.
    self._testListDiff(x=[], y=[1, 2], out=[], idx=[])

  def testEmptyY(self):
    # [FUNCTION DOC]
    # Empty y; expects all of x with identity indices.
    self._testListDiff(x=[1, 2, 3, 4], y=[], out=[1, 2, 3, 4], idx=[0, 1, 2, 3])

  def testEmptyXY(self):
    # [FUNCTION DOC]
    # Both inputs empty; expects empty outputs.
    self._testListDiff(x=[], y=[], out=[], idx=[])


if __name__ == "__main__":
  test.main()
