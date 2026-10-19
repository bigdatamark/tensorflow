# [MODULE DOC]
# Test module for switch/case control flow compiled by XLA (jit-compiled functions). Checks that
# the right branch is selected, and that branches are pruned when the index is a compile-time
# constant. When run as a script it enables eager execution first.
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
"""Tests for case statements in XLA."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test


class CaseTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for switch-case control flow inside jit-compiled functions. Uses the device
  # name provided by the XLA test base class. Holds no state of its own.

  def testCaseBasic(self):
    # [FUNCTION DOC]
    # Builds a jit-compiled switch-case with branches for indices 0 and 1 and a default branch,
    # then calls it with indices 0, 1, 2 and 3 on the test device and asserts results 17, 31, -1
    # and -1 (indices without a branch use the default). Returns nothing.

    @def_function.function(jit_compile=True)
    def switch_case_test(branch_index):
      # [FUNCTION DOC]
      # Jit-compiled function under test: dispatches on the given integer branch index to one of
      # the branch functions (0 gives 17, 1 gives 31, anything else the default, -1) and returns
      # the resulting scalar integer tensor.

      def f1():
        # [FUNCTION DOC]
        # HELPER: branch body for the switch-case test. Returns the constant 17.
        return array_ops.constant(17)

      def f2():
        # [FUNCTION DOC]
        # HELPER: branch body for the switch-case test. Returns the constant 31.
        return array_ops.constant(31)

      def f3():
        # [FUNCTION DOC]
        # HELPER: default branch body for the switch-case test. Returns the constant -1.
        return array_ops.constant(-1)

      return control_flow_switch_case.switch_case(
          branch_index, branch_fns={
              0: f1,
              1: f2
          }, default=f3)

    with ops.device(self.device):
      self.assertEqual(switch_case_test(array_ops.constant(0)).numpy(), 17)
      self.assertEqual(switch_case_test(array_ops.constant(1)).numpy(), 31)
      self.assertEqual(switch_case_test(array_ops.constant(2)).numpy(), -1)
      self.assertEqual(switch_case_test(array_ops.constant(3)).numpy(), -1)

  def testBranchIsPruned(self):
    # [FUNCTION DOC]
    # Verifies XLA does not compile unused branches when the branch index is a trivial constant.
    # One branch contains operations XLA cannot compile (reading and decoding an image file); the
    # constant index 0 selects the other branch, so compilation must succeed and the result must
    # be 17. Returns nothing.

    @def_function.function(jit_compile=True)
    def switch_case_test():
      # [FUNCTION DOC]
      # Jit-compiled function under test with no inputs: uses a constant branch index of 0 and
      # a switch-case whose default and branch 1 are the uncompilable function. Returns a scalar
      # integer tensor (17 when pruning works).
      branch_index = array_ops.constant(0)

      def f1():
        # [FUNCTION DOC]
        # HELPER: selected branch body. Returns the constant 17.
        return array_ops.constant(17)

      def f2():
        # [FUNCTION DOC]
        # HELPER: branch body containing operations XLA cannot compile (file read plus image
        # decode of a fixed temporary path); must never be compiled. Returns the constant 31.
        # Some operations that XLA cannot compile.
        image_ops.decode_image(io_ops.read_file('/tmp/bmp'))
        return array_ops.constant(31)

      # This tests that we do not try to compile all branches if the branch
      # index in trivially constant.
      return control_flow_switch_case.switch_case(
          branch_index, branch_fns={
              0: f1,
              1: f2
          }, default=f2)

    with ops.device(self.device):
      self.assertEqual(switch_case_test().numpy(), 17)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
