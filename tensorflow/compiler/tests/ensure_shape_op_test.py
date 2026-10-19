# [MODULE DOC]
# Tests for the ensure-shape check op on XLA devices: one test class covering a compatible and an
# incompatible shape. No import-time side effects.
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
  # Tests ensure_shape running in the XLA test scope.

  def testEnsureShape(self):
    # [FUNCTION DOC]
    # Feeds a 3x3 integer matrix into ensure_shape with expected shape (any, 3); asserts the output
    # equals the input.
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = check_ops.ensure_shape(p, (None, 3))
      expected_out = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
      self.assertAllEqual(expected_out,
                          sess.run(op, {p: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}))

  def testInvalidEnsureShape(self):
    # [FUNCTION DOC]
    # Feeds a 3x3 matrix where a rank-3 shape is required; asserts an invalid-argument error whose
    # message says the shape is not compatible with the expected shape.
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = check_ops.ensure_shape(p, (None, 3, 3))
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is not compatible with expected shape"):
        sess.run(op, {p: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]})


if __name__ == "__main__":
  test.main()
