# [MODULE DOC]
# Test module checking that enabling and disabling TF2 behavior is reflected consistently in
# both the Python-level TF2 flag and the native (pywrap) TF2 flag.
# Contains one test class with four tests covering every combination of starting in TF1 or TF2
# mode and then calling the enable/disable v2 behavior switches. Run as a script it starts the
# standard TensorFlow test runner. No import-time side effects beyond imports.
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
"""Tests for enabling and disabling TF2 behavior."""

from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.platform import _pywrap_tf2
from tensorflow.python.platform import test


class EnablingTF2Behavior(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the global TF2 behavior switch. Tests are generated for either only the
  # TF1 (graph) mode or only the TF2 (eager) mode via the dataset test-base combinations, and
  # each one toggles the switch through the v2 compat module while checking both the Python
  # flag and the native flag. It holds no state except a seed option set in the constructor.

  def __init__(self, methodName):
    # [FUNCTION DOC]
    # Constructor. Input is the name of the test method to run, passed to the parent test case.
    # Also sets an instance flag that turns off the default random seed setting for these tests
    # (so the base class does not set a default seed). Returns nothing; no exceptions.
    super().__init__(methodName)
    self._set_default_seed = False

  @combinations.generate(test_base.v1_only_combinations())
  def test_tf1_enable_tf2_behaviour(self):
    # [FUNCTION DOC]
    # Runs only in TF1 mode. Starts with both TF2 flags reporting disabled, enables v2 behavior
    # and asserts both flags report enabled, then disables v2 behavior and asserts both report
    # disabled again. Side effect: toggles the process-wide TF2 switch.
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

    v2_compat.enable_v2_behavior()
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

    v2_compat.disable_v2_behavior()
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

  @combinations.generate(test_base.v1_only_combinations())
  def test_tf1_disable_tf2_behaviour(self):
    # [FUNCTION DOC]
    # Runs only in TF1 mode. Starts disabled, disables v2 behavior and asserts both flags stay
    # disabled (a no-op), then enables v2 behavior and asserts both flags report enabled.
    # Side effect: toggles the process-wide TF2 switch.
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

    v2_compat.disable_v2_behavior()
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

    v2_compat.enable_v2_behavior()
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

  @combinations.generate(test_base.v2_only_combinations())
  def test_tf2_enable_tf2_behaviour(self):
    # [FUNCTION DOC]
    # Runs only in TF2 mode. Starts with both flags enabled, enables v2 behavior and asserts
    # they stay enabled (a no-op), then disables v2 behavior and asserts both report disabled.
    # Side effect: toggles the process-wide TF2 switch.
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

    v2_compat.enable_v2_behavior()
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

    v2_compat.disable_v2_behavior()
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

  @combinations.generate(test_base.v2_only_combinations())
  def test_tf2_disable_tf2_behaviour(self):
    # [FUNCTION DOC]
    # Runs only in TF2 mode. Starts enabled, disables v2 behavior and asserts both flags report
    # disabled, then enables v2 behavior and asserts both report enabled again.
    # Side effect: toggles the process-wide TF2 switch.
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

    v2_compat.disable_v2_behavior()
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

    v2_compat.enable_v2_behavior()
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())


if __name__ == '__main__':
  test.main()
