# [MODULE DOC]
# XLA compiler test module for the roll manipulation op: runs it on the XLA test device and
# compares to numpy. Runs the test runner when executed as a script. No import-time side effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for manip ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.platform import googletest


class ManipOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case covering the roll manipulation op on the XLA test device, across numeric,
  # floating point and complex element types supplied by the base class.
  """Test cases for manip ops."""

  def _testRoll(self, a, shift, axis):
    # [FUNCTION DOC]
    # HELPER: shared check for the roll op tests.
    # KNOWN ISSUE: a placeholder is created and fed, but the roll op is applied to the numpy
    # array a directly, not to the placeholder, so the feed is unused and the op is built
    # from a constant.
    # Inputs: a (numpy array), shift (integer or list), axis (integer or list). In the test
    # scope, builds the roll op, runs it in a session and asserts the result equals numpy roll.
    with self.session() as session:
      with self.test_scope():
        p = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        output = manip_ops.roll(a, shift, axis)
      result = session.run(output, {p: a})
      self.assertAllEqual(result, np.roll(a, shift, axis))

  def testNumericTypes(self):
    # [FUNCTION DOC]
    # For each numeric type, rolls random integer arrays: a vector with a single shift, a
    # three-dimensional and a four-dimensional array with per-axis shifts including negative
    # and larger-than-size values. Expects results equal to numpy roll.
    for t in self.numeric_types:
      self._testRoll(np.random.randint(-100, 100, (5)).astype(t), 3, 0)
      self._testRoll(
          np.random.randint(-100, 100, (4, 4, 3)).astype(t), [1, -6, 6],
          [0, 1, 2])
      self._testRoll(
          np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t), [0, 1, -2],
          [1, 2, 3])

  def testFloatTypes(self):
    # [FUNCTION DOC]
    # For each floating point type, rolls random arrays of one, two and three dimensions with
    # single or multiple shifts and axes (including negative shifts). Expects numpy roll results.
    for t in self.float_types:
      self._testRoll(np.random.rand(5).astype(t), 2, 0)
      self._testRoll(np.random.rand(3, 4).astype(t), [1, 2], [1, 0])
      self._testRoll(np.random.rand(1, 3, 4).astype(t), [1, 0, -3], [0, 1, 2])

  def testComplexTypes(self):
    # [FUNCTION DOC]
    # For each complex type, builds complex arrays from random real parts (real plus imaginary
    # equal parts) of two to four dimensions and rolls them with various shifts and axis
    # orders. Expects numpy roll results.
    for t in self.complex_types:
      x = np.random.rand(4, 4).astype(t)
      self._testRoll(x + 1j * x, 2, 0)
      x = np.random.rand(2, 5).astype(t)
      self._testRoll(x + 1j * x, [1, 2], [1, 0])
      x = np.random.rand(3, 2, 1, 1).astype(t)
      self._testRoll(x + 1j * x, [2, 1, 1, 0], [0, 3, 1, 2])


if __name__ == "__main__":
  googletest.main()
