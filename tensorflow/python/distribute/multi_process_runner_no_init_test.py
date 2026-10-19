# [MODULE DOC]
# Test module checking that the multi-process runner refuses to run when the test file did not
# use the runner's own test entry point. It deliberately calls the regular test main so the
# runner is never initialized, and expects the not-initialized error. No import-time side effects.
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
"""Tests for `multi_process_runner` for non-initialization."""

from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import test


class MultiProcessRunnerNoInitTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the multi-process runner's behavior when it was never initialized through
  # its dedicated test entry point. No shared fixtures.

  def test_not_calling_correct_main(self):
    # [FUNCTION DOC]
    # Scenario: run a trivial function on a one-worker cluster without initialization.
    # Expected: the not-initialized error is raised with a message saying the runner is not
    # initialized.

    def simple_func():
      # [FUNCTION DOC]
      # HELPER: trivial task function for the test; returns a fixed string and is never actually
      # run in a child because the runner rejects initialization first.
      return 'foobar'

    with self.assertRaisesRegex(multi_process_runner.NotInitializedError,
                                '`multi_process_runner` is not initialized.'):
      multi_process_runner.run(
          simple_func,
          multi_worker_test_base.create_cluster_spec(num_workers=1))


if __name__ == '__main__':
  # Intentionally not using `multi_process_runner.test_main()` so the error
  # would occur.
  test.main()
