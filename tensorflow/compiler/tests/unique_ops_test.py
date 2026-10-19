# [MODULE DOC]
# Test for the unique op (version 2, with an axis argument) on an XLA test device. Currently
# contains one regression test for negative axis handling. Runs the standard test runner when
# executed as a script.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for unique ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import googletest


class UniqueTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the unique op; holds a single test and no shared fixtures.

  def testNegativeAxis(self):
    # [FUNCTION DOC]
    # Scenario: unique along axis minus one of a 2x1x1 float tensor holding two distinct values.
    # Runs the unique-v2 op in the XLA scope and asserts the unique values equal the input and the
    # index output is a single zero, i.e. the negative axis was normalised to a positive axis.
    """Verifies that an axis with negative index is converted to positive."""
    with self.session() as session:
      with self.test_scope():
        px = array_ops.placeholder(dtypes.float32, [2, 1, 1], name="x")
        axis = constant_op.constant([-1], dtype=dtypes.int32)
        output = gen_array_ops.unique_v2(px, axis)
      result = session.run(
          output, {px: np.array([[[-2.0]], [[10.0]]], dtype=np.float32)}
      )
      self.assertAllEqual(
          result.y, np.array([[[-2.0]], [[10.0]]], dtype=np.float32)
      )
      self.assertAllEqual(result.idx, np.array([0], dtype=np.int32))


if __name__ == "__main__":
  googletest.main()
