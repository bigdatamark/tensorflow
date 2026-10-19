# [MODULE DOC]
# KNOWN ISSUE: The module docstring says "Tests for PrecisionOp", but the file actually tests the
# in-top-k op (the old precision op name).
# Test module for the neural-network in-top-k op: for score matrices and target class vectors it
# checks the boolean membership result. No import-time side effects; running it as a script
# starts the TensorFlow test runner.
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
"""Tests for PrecisionOp."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class InTopKTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the in-top-k op (checks whether each target class is among the k highest
  # predictions of its row). Covers plain, tied, int64-target, non-finite, out-of-range target,
  # empty and tensor-valued k cases through one shared validation helper.

  def _validateInTopK(self, predictions, target, k, expected):
    # [FUNCTION DOC]
    # HELPER: Shared assertion routine for all in-top-k tests in this class.
    # Inputs: predictions (matrix of scores, examples by classes), target (vector of true class
    # indices), k (integer or scalar tensor), expected (list of booleans). Runs the in-top-k op
    # in a cached session, evaluates it, and asserts the result is close to the expected boolean
    # array and has the same shape. Returns nothing; fails the test on mismatch.
    np_ans = np.array(expected, np.bool_)
    with self.cached_session():
      precision = nn_ops.in_top_k(predictions, target, k)
      out = self.evaluate(precision)
      self.assertAllClose(np_ans, out)
      self.assertShapeEqual(np_ans, precision)

  def testInTop1(self):
    # [FUNCTION DOC]
    # Test: k of 1 with targets 3 and 2. In the first row class 3 has the highest score (true);
    # in the second row the highest is class 3, not 2 (false). Expects [True, False].
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [3, 2]
    self._validateInTopK(predictions, target, 1, [True, False])

  def testInTop2(self):
    # [FUNCTION DOC]
    # Test: two examples with four classes, k of 2, both targeted at class 2. Class 2 is third
    # in the first row (not in top 2) and second in the second row. Expects [False, True].
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [2, 2]
    self._validateInTopK(predictions, target, 2, [False, True])

  def testInTop2Tie(self):
    # [FUNCTION DOC]
    # Test: ties at the k-th position. Classes 2 and 3 have equal scores, tying for second place,
    # so with k of 2 both targets (2 and 3) count as being in the top 2. Expects [True, True].
    # Class 2 and 3 tie for 2nd, so both are considered in top 2.
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 3]
    self._validateInTopK(predictions, target, 2, [True, True])

  def testInTop2_int64Target(self):
    # [FUNCTION DOC]
    # Test: targets given as a 64-bit integer array instead of a list, k of 2, targets 0 and 2.
    # Class 0 is the lowest-scored in the first row (not in top 2); class 2 is second in the
    # second row. Expects [False, True].
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = np.asarray([0, 2]).astype(np.int64)
    self._validateInTopK(predictions, target, 2, [False, True])

  def testInTopNan(self):
    # [FUNCTION DOC]
    # Test: rows containing non-finite predictions (a not-a-number entry in the first row and an
    # infinity in the second). With k of 2 and targets 1 and 3, both rows must be reported as not
    # in the top k. Expects [False, False].
    predictions = [[0.1, float("nan"), 0.2, 0.4], [0.1, 0.2, 0.3, float("inf")]]
    target = [1, 3]
    self._validateInTopK(predictions, target, 2, [False, False])

  def testBadTarget(self):
    # [FUNCTION DOC]
    # Test: the second target (4) is outside the valid class range of four classes. With k of 2,
    # the op must report false for that row rather than fail; the first row (target 2, tied for
    # second) is true. Expects [True, False].
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 4]  # must return False for invalid target
    self._validateInTopK(predictions, target, 2, [True, False])

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: zero examples (an empty predictions matrix with five classes and an empty integer
    # target vector) with k of 2. Expects an empty boolean result of matching shape.
    predictions = np.empty([0, 5])
    target = np.empty([0], np.int32)
    self._validateInTopK(predictions, target, 2, [])

  def testTensorK(self):
    # [FUNCTION DOC]
    # Test: k is supplied as a constant scalar tensor (value 3) instead of a Python integer. With
    # target classes 0 and 2, the first row (class 0 has the lowest score) is not in the top 3 and
    # the second row is. Expects [False, True].
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [0, 2]
    k = constant_op.constant(3)
    self._validateInTopK(predictions, target, k, [False, True])


if __name__ == "__main__":
  test.main()
