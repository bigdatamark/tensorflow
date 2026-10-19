# [MODULE DOC]
# Unit test module checking that eager execution can be enabled and disabled at process level
# through the framework ops module, and that repeating either call is harmless. Exposes one test
# case class and runs the standard test main when executed directly. No import-time side effects.
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
"""Tests enabling eager execution at process level."""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest


class OpsEnableAndDisableEagerTest(googletest.TestCase):
  # [CLASS DOC]
  # Test case covering process-level enabling and disabling of eager execution through the
  # framework ops module. All checks live in the per-test setup (enable, twice) and teardown
  # (disable, twice) methods; the single test method is an empty placeholder. Instances are
  # created by the test runner; the class holds no extra state.

  def setUp(self):
    # [FUNCTION DOC]
    # Test setup that verifies enabling eager execution. Takes no inputs and returns nothing.
    # Calls the base setup, enables eager execution at process level and asserts that eager
    # execution is active. Enables it a second time and asserts again, checking that repeated
    # enabling raises no error. Side effect: changes the process-wide eager execution state.
    super().setUp()
    # test for enable eager test
    ops.enable_eager_execution()
    self.assertTrue(context.executing_eagerly())

    # Calling enable eager execution a second time should not cause an error.
    ops.enable_eager_execution()
    self.assertTrue(context.executing_eagerly())

  def testEnableDisableEagerExecution(self):
    # [FUNCTION DOC]
    # Placeholder test method that does nothing itself. The real scenario runs in the setup
    # method (enable eager execution, twice) and the teardown method (disable it, twice), so this
    # test passes as long as those two steps succeed. Takes no inputs and returns nothing.
    # The entirety of the test runs in setUp/tearDown methods
    pass

  def tearDown(self):
    # [FUNCTION DOC]
    # Test teardown that verifies disabling eager execution. Takes no inputs and returns nothing.
    # Calls the base teardown, then disables eager execution at process level and asserts that
    # eager execution is no longer active. Disables it a second time and asserts again that it is
    # still inactive, checking that repeated disabling is harmless. Side effect: changes the
    # process-wide eager execution state back to graph mode. Fails the test on assertion mismatch.
    super().tearDown()
    # test for disable eager test
    ops.disable_eager_execution()
    self.assertFalse(context.executing_eagerly())

    # Calling disable eager execution a second time should not cause an error.
    ops.disable_eager_execution()
    self.assertFalse(context.executing_eagerly())


if __name__ == '__main__':
  googletest.main()
