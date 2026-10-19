# [MODULE DOC]
# Test that switching off v2 behavior turns eager tensors into graph tensors and turns the
# v2 flag off. When run as a script it first enables v2 behavior, then runs the tests.
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
"""Tests for forward and backwards compatibility utilities."""

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import _pywrap_tf2
from tensorflow.python.platform import test


class DisableV2BehaviorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for disabling v2 behavior.

  def test_basic(self):
    # [FUNCTION DOC]
    # Asserts a constant is an eager tensor and the v2 flag is on; calls the disable function;
    # then asserts a new constant is not an eager tensor and the v2 flag is off. Leaves v2
    # behavior disabled for the rest of the process.
    t = constant_op.constant([1, 2, 3])  # creates a hidden context
    self.assertTrue(isinstance(t, ops.EagerTensor))
    t = _pywrap_tf2.is_enabled()
    self.assertTrue(t)
    v2_compat.disable_v2_behavior()
    t = constant_op.constant([1, 2, 3])
    self.assertFalse(isinstance(t, ops.EagerTensor))
    t = _pywrap_tf2.is_enabled()
    self.assertFalse(t)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
