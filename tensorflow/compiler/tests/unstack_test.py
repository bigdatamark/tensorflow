# [MODULE DOC]
# Test for the unstack (unpack) op on an XLA test device with a very large number of outputs.
# Contains one parameterized-capable test case class with one size-driven helper and one test.
# Runs the standard test runner when executed as a script.
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
"""Tests for tensorflow.compiler.tests.unstack."""

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.platform import test


class UnstackOpTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # XLA test case (also a parameterized test case) for the unstack op. Provides a size-driven
  # helper used by the single test method.

  def _test(self, size):
    # [FUNCTION DOC]
    # HELPER: shared check for unstack. Input: size, the number of rows of a float matrix with 512
    # columns. Feeds an all-zero matrix through a placeholder, unstacks it in the XLA scope, and
    # asserts there are exactly size outputs, each entirely zero. Returns nothing.
    with self.session() as sess:
      x_tf = array_ops.placeholder(np.float32, shape=[size, 512])
      with self.test_scope():
        ret = array_ops_stack.unstack(x_tf)
      ret_vals = sess.run([ret], feed_dict={x_tf: np.zeros([size, 512])})
      self.assertLen(ret_vals[0], size)
      for ret_val in ret_vals[0]:
        self.assertTrue(np.all(ret_val == 0.))

  def testLarge2000(self):
    # [FUNCTION DOC]
    # Scenario: unstacks a 2000-row matrix, i.e. produces 2000 outputs; delegates to the helper
    # which expects 2000 all-zero rows.
    self._test(2000)


if __name__ == "__main__":
  test.main()
