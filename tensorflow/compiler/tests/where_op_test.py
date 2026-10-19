# [MODULE DOC]
# Tests for the single-argument "where" operation (returning indices of non-zero elements) when
# compiled through XLA. Defines one test class run via the standard test main entry point.
# Covers boolean, integer, float and complex inputs, 1-D and 2-D shapes, and dynamic-shaped outputs
# feeding into gather and reduction operations. No import-time side effects.
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
"""Tests for where op."""

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu
# pylint: enable=g-direct-tensorflow-import


class WhereOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the index-returning form of the where operation under the XLA test device.
  # Inherits session and test-scope helpers from the XLA test base class. No shared fixtures other
  # than TPU system initialization in the constructor.

  def __init__(self, method_name="runTest"):
    # [FUNCTION DOC]
    # Constructor. Takes the name of the test method to run (default is the generic run method) and
    # forwards it to the base class. If any logical TPU devices exist, opens a session and runs the
    # TPU system initialization operation. Returns nothing.
    super(WhereOpTest, self).__init__(method_name)
    if config.list_logical_devices("TPU"):
      with self.session() as sess:
        sess.run(tpu.initialize_system())

  def testWhere(self):
    # [FUNCTION DOC]
    # Feeds a 2x3 boolean matrix with three true entries to where on the XLA device and asserts the
    # result is the three row/column index pairs of the true entries in row-major order.
    """Test first form of where (return indices)."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.bool)
        true_vals = array_ops.where(x)

      # Output of the computation is dynamic.
      feed = [[True, False, False], [False, True, True]]
      self.assertAllEqual([[0, 0], [1, 1], [1, 2]],
                          sess.run(true_vals, {x: feed}))

  def testWhereGather(self):
    # [FUNCTION DOC]
    # Applies where to a 2x2 boolean mask, then gathers elements of a constant 2x2 float matrix at
    # the resulting indices. Asserts the gathered values are zero, two and three.
    """Test where followed by a gather."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.bool)
        value = array_ops.constant([[0, 1], [2, 3]], dtypes.float32)
        true_vals = array_ops.where(x)

        # Gather 0, 2, 3.
        gathered = array_ops.gather_nd(value, true_vals)

      feed = [[True, False], [True, True]]
      self.assertAllEqual([0, 2, 3], sess.run(gathered, {x: feed}))

  def testWhereGatherReduce(self):
    # [FUNCTION DOC]
    # Same as the gather test but additionally sums the gathered values. Asserts the total is five,
    # checking that dynamic-shaped where output flows through gather and reduction.
    """Test where followed by a gather and a reduce."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.bool)
        value = array_ops.constant([[0, 1], [2, 3]], dtypes.float32)
        indices = array_ops.where(x)

        # Reduce to 5
        gathered = array_ops.gather_nd(value, indices)
        reduction = math_ops.reduce_sum(gathered)

      feed = [[True, False], [True, True]]
      self.assertAllEqual(5, sess.run(reduction, {x: feed}))

  def testWhere1D(self):
    # [FUNCTION DOC]
    # Runs where on a 1-D boolean vector (true, false, true) and asserts the output is a column of
    # indices zero and two (shape two by one).
    """Test first form of where (return indices)."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.bool)
        result = array_ops.where(x)

      # Output of the computation is dynamic.
      feed = [True, False, True]
      self.assertAllEqual([[0], [2]], sess.run(result, {x: feed}))

  def testWhereInt(self):
    # [FUNCTION DOC]
    # Runs where on an int32 vector (-1, 0, 1); non-zero counts as true. Asserts indices zero, two.
    """Test Where with integers."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.int32)
        result = array_ops.where(x)

      # Output of the computation is dynamic.
      feed = [-1, 0, 1]
      self.assertAllEqual([[0], [2]], sess.run(result, {x: feed}))

  def testWhereFloat(self):
    # [FUNCTION DOC]
    # Runs where on a float32 vector containing negative and positive zero and non-zero values.
    # Asserts both zeros are treated as false, so only indices zero and three are returned.
    """Test Where with floats."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.float32)
        result = array_ops.where(x)

      # Output of the computation is dynamic.
      feed = [-1.0, -0.0, 0.0, 1.0]
      self.assertAllEqual([[0], [3]], sess.run(result, {x: feed}))

  def testWhereComplex(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says "floats" but the test uses complex64 values.
    # Runs where on a complex64 vector; only elements with both parts zero count as false.
    # Asserts the returned indices are zero, three, four and five.
    """Test Where with floats."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.complex64)
        result = array_ops.where(x)

      # Output of the computation is dynamic.
      feed = [
          -1.0 + 0.0j, -0.0 + 0.0j, 0.0 - 0.0j, 1.0 - 1.0j, 1.0 + 0.0j,
          0.0 + 1.0j
      ]
      self.assertAllEqual([[0], [3], [4], [5]], sess.run(result, {x: feed}))

if __name__ == "__main__":
  test.main()
