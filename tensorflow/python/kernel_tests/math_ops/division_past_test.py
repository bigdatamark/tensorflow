# [MODULE DOC]
# Test module checking that TensorFlow tensor division (true and floor) matches NumPy results
# for integer and floating dtypes, with the future division import intentionally omitted. Twin of
# the "future" division test. No import-time side effects; running it as a script starts the
# TensorFlow test runner.
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
"""Tests for division with division imported from __future__.

This file should be exactly the same as division_future_test.py except
for the __future__ division line.
"""

# from __future__ import division  # Intentionally skip this import
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class DivisionTestCase(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that tensor division and floor division operators agree with NumPy
  # scalar division in value and dtype, without the future division import. Holds no state; one
  # test method, no shared fixtures.

  def testDivision(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The module docstring says division is imported from the future, but this file
    # deliberately omits that import (copy-paste docstring). The session object bound by the
    # context manager is never used.
    # Scenario: for each dtype (8/16/32/64-bit integers and floats) and each pair of values from
    # a small list, operands are plain NumPy scalars or constant tensors. True division and floor
    # division results from NumPy are compared with tensor results (any operand mix). True
    # division is skipped for 8 and 16-bit integers because NumPy promotes those results to
    # 64-bit floats, which disagrees with TensorFlow. All pairs are evaluated in one call; each
    # pair must have equal dtype and close values.
    """Test all the different ways to divide."""
    values = [1, 2, 7, 11]
    functions = (lambda x: x), constant_op.constant
    dtypes = np.int8, np.int16, np.int32, np.int64, np.float32, np.float64

    tensors = []
    checks = []

    def check(x, y):
      # [FUNCTION DOC]
      # HELPER: Records one expected/actual pair for the single batched evaluation in the test.
      # Takes x (a NumPy scalar result) and y (a TensorFlow tensor result), converts both to
      # tensors, appends the pair to the enclosing list of tensors and appends a comparison
      # callback to the enclosing list of checks. Returns nothing; mutates those two lists.
      x = ops.convert_to_tensor(x)
      y = ops.convert_to_tensor(y)
      tensors.append((x, y))
      def f(x, y):
        # [FUNCTION DOC]
        # HELPER: Deferred comparison used by the division test after the single evaluation.
        # Takes an expected value x (computed with NumPy) and an evaluated TensorFlow result y.
        # Asserts that both have the same data type and that their values are close. Returns
        # nothing; raises a test assertion failure on mismatch. Called once per recorded pair.
        self.assertEqual(x.dtype, y.dtype)
        self.assertAllClose(x, y)
      checks.append(f)

    with self.cached_session() as sess:
      for dtype in dtypes:
        for x in map(dtype, values):
          for y in map(dtype, values):
            for fx in functions:
              for fy in functions:
                tf_x = fx(x)
                tf_y = fy(y)
                div = x / y
                tf_div = tf_x / tf_y
                # In NumPy 1.23, np.int16(x) / np.int16(y) now has np.float64
                # type, which disagrees with TF.
                if x.dtype not in (np.int8, np.int16):
                  check(div, tf_div)
                floordiv = x // y
                tf_floordiv = tf_x // tf_y
                check(floordiv, tf_floordiv)
      # Do only one sess.run for speed
      for f, (x, y) in zip(checks, self.evaluate(tensors)):
        f(x, y)


if __name__ == "__main__":
  test.main()
