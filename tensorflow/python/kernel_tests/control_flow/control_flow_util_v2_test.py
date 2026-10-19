# [MODULE DOC]
# Unit test for the version 2 control flow utility module: checks that its in-defun query reports
# true only while code is being traced inside a function graph (here, within while loop and conditional
# branches). The fixture toggles the global enable-control-flow-v2 flag. No import-time side effects.
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

"""Tests for tensorflow.python.ops.control_flow_util_v2."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


class ControlFlowUtilV2Test(test.TestCase):
  # [CLASS DOC]
  # Test case for the version 2 control flow utility module, specifically its check of whether code is
  # currently being built inside a defun (function graph). Its fixture forces the global switch that
  # enables version 2 control flow on for each test and restores the previous value afterwards.
  # Derives from the framework test case.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Saves the current value of the global flag enabling control flow version 2 on the
    # instance and then sets that module-level flag to true. Side effect: mutates global state in the
    # control flow utility module until tearDown restores it. Does not call the parent setUp.
    self._enable_control_flow_v2_old = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture teardown. Restores the global flag enabling control flow version 2 to the value saved in
    # setUp. Side effect: mutates module-level state. Does not call the parent tearDown.
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = self._enable_control_flow_v2_old

  def _create_control_flow(self, expect_in_defun):
    # [FUNCTION DOC]
    # HELPER: builds the control flow used by testInDefun.
    # Takes a boolean expect_in_defun. Builds a while loop running four iterations (starting from zero)
    # whose body creates a conditional on a constant true; the true branch asserts that the in-defun
    # check equals expect_in_defun and returns the loop value plus one, the false branch returns zero.
    # Returns the while loop result. Fails an assertion inside the branch when the expectation differs.
    """Helper method for testInDefun."""
    def body(i):
      # [FUNCTION DOC]
      # HELPER: while loop body for the control flow built by the enclosing helper.
      # Takes the loop counter i and returns the result of a conditional on a constant true, using the
      # nested branch function for the true case and a function returning zero for the false case.
      def branch():
        # [FUNCTION DOC]
        # HELPER: true branch of the conditional in the enclosing helper.
        # Asserts that the in-defun check of the control flow version 2 utility equals the expectation
        # passed to the helper, then returns the loop counter plus one. Fails the test if the assertion
        # does not hold.
        self.assertEqual(control_flow_util_v2.in_defun(), expect_in_defun)
        return i + 1
      return cond.cond(constant_op.constant(True),
                       branch, lambda: 0)

    return while_loop.while_loop(lambda i: i < 4, body,
                                 [constant_op.constant(0)])

  @test_util.run_in_graph_and_eager_modes
  def testInDefun(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes) of the in-defun check. First builds the control flow directly,
    # expecting the check to report false inside the branch. Then defines a compiled function (via the
    # function decorator) that builds the same control flow expecting true, and calls it. Finally
    # asserts that the check reports false again outside any function.
    self._create_control_flow(False)

    @def_function.function
    def defun():
      # [FUNCTION DOC]
      # HELPER: compiled function used by testInDefun. Takes no inputs; when traced and called it builds
      # the control flow with the expectation that code is running inside a function graph. Returns nothing.
      self._create_control_flow(True)

    defun()
    self.assertFalse(control_flow_util_v2.in_defun())


if __name__ == "__main__":
  test.main()
