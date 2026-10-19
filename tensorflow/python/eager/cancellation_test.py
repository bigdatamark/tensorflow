# [MODULE DOC]
# Tests for the eager cancellation manager. Contains one test class checking that a new manager is
# not cancelled and becomes cancelled after a cancel request. Run directly as a test binary.
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

from tensorflow.python.eager import cancellation
from tensorflow.python.platform import test


class CancellationTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering CancellationManager state transitions. No shared fixtures.

  def testStartCancel(self):
    # [FUNCTION DOC]
    # Scenario: create a new cancellation manager. Checks that it reports not cancelled initially,
    # calls the cancel request, then asserts it reports cancelled.
    manager = cancellation.CancellationManager()
    self.assertFalse(manager.is_cancelled)
    manager.start_cancel()
    self.assertTrue(manager.is_cancelled)


if __name__ == '__main__':
  test.main()
