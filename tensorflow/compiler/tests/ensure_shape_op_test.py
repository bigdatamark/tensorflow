# [MODULE DOC]
# Test module for the ensure-shape op on XLA test devices: one test class checking that a
# compatible shape passes data through and an incompatible shape raises an invalid-argument
# error. No import-time side effects; runs the test runner when executed as a script.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ensure_shape_op."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.platform import test


class EnsureShapeOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the ensure-shape op; each test feeds a three-by-three integer matrix into a
  # placeholder and runs the op in a graph session on the test device.

  def testEnsureShape(self):
    # [FUNCTION DOC]
    # Ensures a placeholder has shape (unknown, 3) and feeds a three-by-three matrix; asserts the
    # output equals the input.
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = check_ops.ensure_shape(p, (None, 3))
      expected_out = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
      self.assertAllEqual(expected_out,
                          sess.run(op, {p: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}))

  def testInvalidEnsureShape(self):
    # [FUNCTION DOC]
    # Ensures shape (unknown, 3, 3) (rank three) but feeds a rank-two matrix; asserts running the
    # op raises an invalid-argument error whose message says the shape is not compatible with the
    # expected shape.
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = check_ops.ensure_shape(p, (None, 3, 3))
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is not compatible with expected shape"):
        sess.run(op, {p: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]})


if __name__ == "__main__":
  test.main()
