# [MODULE DOC]
# XLA compiler test for the array repeat op under jit compilation, checking that repeating with
# per-element counts (a bounded-dynamic-shape result) works along both axes. No import-time side
# effects.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.array_ops.repeat."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class RepeatTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the repeat op compiled with XLA; holds a single test and no fixtures.

  def test(self):
    # [FUNCTION DOC]
    # Defines a jit-compiled function that repeats a tensor with given repeat counts along an
    # axis. On the test device repeats the int32 matrix [[1,2],[3,4]] with counts [1,2] along
    # axis 0 and axis 1 and asserts the results [[1,2],[3,4],[3,4]] and [[1,2,2],[3,4,4]]. The
    # in-code comment says it verifies that a bounded dynamic result from the Where op can be
    # reshaped correctly.

    # Verifies that bounded dynamic result generated from the Where op can be
    # Reshaped correctly.
    @def_function.function(jit_compile=True)
    def repeat(values, repeats, axis):
      # [FUNCTION DOC]
      # HELPER: jit-compiled wrapper that applies the repeat op to values with repeat counts along
      # the given axis and returns the resulting tensor.
      return array_ops.repeat(values, repeats, axis)

    with self.session() as sess:
      with self.test_scope():
        values = array_ops.constant([[1, 2], [3, 4]], dtype=dtypes.int32)
        repeats = array_ops.constant([1, 2], dtype=dtypes.int32)
        y1 = repeat(values, repeats, 0)
        y2 = repeat(values, repeats, 1)
      actual1, actual2 = sess.run([y1, y2])

    self.assertAllEqual(actual1, [[1, 2], [3, 4], [3, 4]])
    self.assertAllEqual(actual2, [[1, 2, 2], [3, 4, 4]])


if __name__ == "__main__":
  test.main()
