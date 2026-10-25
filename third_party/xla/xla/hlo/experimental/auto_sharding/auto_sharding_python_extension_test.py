# [MODULE DOC]
# Tests for the auto-sharding Python extension, which registers or unregisters the auto-sharding
# functionality in the process. Two tests check the registered state query together with the
# register and clear operations. No import-time side effects beyond importing the extension.
# Copyright 2025 The OpenXLA Authors
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
"""Tests for AutoSharding Python extension."""

from absl.testing import absltest
from xla.hlo.experimental.auto_sharding import auto_sharding_python_extension


class AutoShardingPythonExtensionTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for the auto-sharding extension's registration state. Stateless in Python, but
  # the tests change process-global registration state of the extension.
  # KNOWN ISSUE: the register test assumes nothing registered it earlier and does not clear
  # afterwards, so the tests depend on execution order and shared global state.

  def test_register(self):
    # [FUNCTION DOC]
    # Test: asserts the extension starts unregistered, registers it, then asserts it reports
    # registered. Side effect: leaves the extension registered.
    self.assertFalse(auto_sharding_python_extension.is_registered())
    auto_sharding_python_extension.register()
    self.assertTrue(auto_sharding_python_extension.is_registered())

  def test_clear(self):
    # [FUNCTION DOC]
    # Test: registers the extension, asserts it reports registered, clears it, then asserts it
    # reports unregistered. Side effect: leaves the extension unregistered.
    auto_sharding_python_extension.register()
    self.assertTrue(auto_sharding_python_extension.is_registered())
    auto_sharding_python_extension.clear()
    self.assertFalse(auto_sharding_python_extension.is_registered())


if __name__ == '__main__':
  absltest.main()
