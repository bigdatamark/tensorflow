# [MODULE DOC]
# Test module for the dense variable update operations (assign, assign add, assign subtract)
# covering behaviors that only exist in TF1 graph mode: assignment without shape validation and
# the requirement that variables be initialized before use. Defines one test case class and runs
# the test main entry point when executed as a script. No import-time side effects.
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
"""Tests for dense_update_ops that only work in V1."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test as test_lib


class AssignOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for dense assign operations restricted to V1 behavior. Each test is decorated to
  # run only in V1 mode; no shared fixtures.

  @test_util.run_v1_only("Non-strict shape assignment only in V1.")
  # Bypassing Shape validation in assign with validate_shape only works in V1.
  # TF V2 raises error:
  # ValueError: Cannot assign value to variable ' Variable:0': Shape mismatch.
  # The variable shape (1,), and the assigned value shape (1024, 1024) are
  # incompatible.
  # Thus, test is enabled only for V1.
  def testAssignNonStrictShapeChecking(self):
    # [FUNCTION DOC]
    # Scenario: a variable holding a one-element list is assigned a 1024 by 1024 tensor with shape
    # validation turned off, then a 10 by 10 tensor the same way. After each assignment the
    # variable is expected to equal the assigned data, proving the shape was replaced. Returns
    # nothing; fails via assertion if values differ.
    data = array_ops.fill([1024, 1024], 0)
    p = variable_v1.VariableV1([1])
    a = state_ops.assign(p, data, validate_shape=False)
    self.evaluate(a)
    self.assertAllEqual(p, self.evaluate(data))

    # Assign to yet another shape
    data2 = array_ops.fill([10, 10], 1)
    a2 = state_ops.assign(p, data2, validate_shape=False)
    self.evaluate(a2)
    self.assertAllEqual(p, self.evaluate(data2))

  @test_util.run_v1_only("Variables need initialization only in V1,")
  def testInitRequiredAssignAdd(self):
    # [FUNCTION DOC]
    # Scenario: create an integer variable without running its initializer, build an assign-add
    # of zeros, and evaluate it. Expects an op error mentioning use of an uninitialized value.
    p = variable_v1.VariableV1(array_ops.fill([1024, 1024], 1), dtypes.int32)
    a = state_ops.assign_add(p, array_ops.fill([1024, 1024], 0))
    with self.assertRaisesOpError("use uninitialized"):
      self.evaluate(a)

  @test_util.run_v1_only("Variables need initialization only in V1.")
  def testInitRequiredAssignSub(self):
    # [FUNCTION DOC]
    # Scenario: same as the assign-add variant but with assign-subtract: an uninitialized integer
    # variable is updated and evaluated, expecting an op error about using an uninitialized value.
    p = variable_v1.VariableV1(array_ops.fill([1024, 1024], 1), dtypes.int32)
    a = state_ops.assign_sub(p, array_ops.fill([1024, 1024], 0))
    with self.assertRaisesOpError("use uninitialized"):
      self.evaluate(a)


if __name__ == "__main__":
  test_lib.main()
