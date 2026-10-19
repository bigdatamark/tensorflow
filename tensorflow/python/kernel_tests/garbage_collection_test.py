# [MODULE DOC]
# Tests that create eager resource variables and tensor arrays under a decorator which enables
# garbage collector debug retention and asserts no uncollectable garbage (reference cycles) is
# left behind. Isolated in its own module because that collector flag is sticky.
# Defines one test class and runs the TensorFlow test main when executed as a script.
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
"""Tests which set DEBUG_SAVEALL and assert no garbage was created.

This flag seems to be sticky, so these tests have been isolated for now.
"""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class NoReferenceCycleTests(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case whose methods are wrapped so they fail if running them leaves garbage objects
  # (reference cycles) behind. No shared fixtures.

  @test_util.assert_no_garbage_created
  def testEagerResourceVariables(self):
    # [FUNCTION DOC]
    # In eager mode, creates a resource variable named "a" with initial value one and discards
    # it; the wrapper asserts no garbage was created. Returns nothing.
    with context.eager_mode():
      resource_variable_ops.ResourceVariable(1.0, name="a")

  @test_util.assert_no_garbage_created
  def testTensorArrays(self):
    # [FUNCTION DOC]
    # In eager mode, builds a size-3 float tensor array without shape inference, writes three
    # differently shaped values, reads them back, and asserts the values match what was written.
    # The wrapper additionally asserts no garbage (reference cycles) was created. Returns nothing.
    with context.eager_mode():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)


if __name__ == "__main__":
  test.main()
