# [MODULE DOC]
# XLA compiler test module for operators with no arguments (no-op and constants of many types
# and shapes). Runs the test runner when executed as a script.
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
"""Test cases for operators with no arguments."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest


class NullaryOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for operators with no arguments: the no-op and constants of many numeric and
  # complex types and shapes (including empty arrays), run on the XLA test device.

  def _testNullary(self, op, expected):
    # [FUNCTION DOC]
    # HELPER: shared runner. Inputs: op (callable with no arguments that builds the output) and
    # expected (value to compare with). Builds the op in the test scope, runs it in a session
    # and asserts closeness with relative tolerance 0.001.
    with self.session() as session:
      with self.test_scope():
        output = op()
      result = session.run(output)
      self.assertAllClose(result, expected, rtol=1e-3)

  def testNoOp(self):
    # [FUNCTION DOC]
    # Scenario: build a no-op in the test scope and run it; expects it not to crash.
    with self.session():
      with self.test_scope():
        output = control_flow_ops.no_op()
      # This should not crash.
      output.run()

  def testConstants(self):
    # [FUNCTION DOC]
    # Scenario: for each numeric type, constants of many shapes (scalar, empty, vectors, 2-D,
    # 3-D, 4-D and empty nested arrays) are created and fetched; expects the same values.
    for dtype in self.numeric_types:
      constants = [
          dtype(42),
          np.array([], dtype=dtype),
          np.array([1, 2], dtype=dtype),
          np.array([7, 7, 7, 7, 7], dtype=dtype),
          np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
          np.array([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]],
                   dtype=dtype),
          np.array([[[]], [[]]], dtype=dtype),
          np.array([[[[1]]]], dtype=dtype),
      ]
      for c in constants:
        self._testNullary(lambda c=c: constant_op.constant(c), expected=c)

  def testComplexConstants(self):
    # [FUNCTION DOC]
    # Scenario: same as the numeric constants test but for each complex type with values that
    # have nonzero imaginary parts; expects the same values back.
    for dtype in self.complex_types:
      constants = [
          dtype(42 + 3j),
          np.array([], dtype=dtype),
          np.ones([50], dtype=dtype) * (3 + 4j),
          np.array([1j, 2 + 1j], dtype=dtype),
          np.array([[1, 2j, 7j], [4, 5, 6]], dtype=dtype),
          np.array([[[1, 2], [3, 4 + 6j], [5, 6]],
                    [[10 + 7j, 20], [30, 40], [50, 60]]],
                   dtype=dtype),
          np.array([[[]], [[]]], dtype=dtype),
          np.array([[[[1 + 3j]]]], dtype=dtype),
      ]
      for c in constants:
        self._testNullary(lambda c=c: constant_op.constant(c), expected=c)


if __name__ == "__main__":
  googletest.main()
