# [MODULE DOC]
# Test module checking that, when the environment requests TF2 behavior but explicitly turns control
# flow v2 off, the control flow v2 switch stays off. Sets two environment variables at import time
# (before importing the TensorFlow modules that read them): TF2 behavior on, control flow v2 off.
# Exposes a single test class and runs the googletest main entry point when executed as a script.
# Not a helper module.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests that TF2_BEHAVIOR=1 and TF_ENABLE_CONTROL_FLOW_V2=0 disables cfv2."""

import os
os.environ["TF2_BEHAVIOR"] = "1"
os.environ["TF_ENABLE_CONTROL_FLOW_V2"] = "0"

from tensorflow.python import tf2  # pylint: disable=g-import-not-at-top
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class ControlFlowV2DisableTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the interaction between the TF2 behavior environment variable and the
  # explicit control-flow-v2 opt-out environment variable. Has no shared fixtures or state.

  def testIsDisabled(self):
    # [FUNCTION DOC]
    # Verifies the opt-out wins: asserts that the TF2 module reports TF2 behavior as enabled while
    # the module-level control flow v2 flag in the control flow utilities is false. Takes no
    # inputs, returns nothing, fails the test on mismatch, no side effects.
    self.assertTrue(tf2.enabled())
    self.assertFalse(control_flow_util.ENABLE_CONTROL_FLOW_V2)


if __name__ == "__main__":
  googletest.main()
