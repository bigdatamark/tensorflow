# [MODULE DOC]
# Tests for the watchdog module: checks that the watchdog fires its callback and writes stack
# traces after inactivity, that reporting activity delays firing, and that stopping it ends
# firing. Running as a script starts the standard test runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for watchdog.py."""

import os
import time

from absl.testing import parameterized

from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import test


class WatchDogTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the watchdog class; has no shared fixtures.

  @parameterized.parameters(True, False)
  def testWatchDogTimeout(self, use_env_var):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the environment variable set in the first parameterization is never removed, so
    # it leaks into later tests and into the second parameterization. Also takes about thirty
    # seconds of real sleeping.
    # Test, run twice: with the three second timeout given through the environment variable (true)
    # or the constructor (false). Creates a watchdog writing to a temporary file with a counting
    # callback. After six seconds expects at least one trigger; after reporting activity and a
    # second of waiting expects still at least one; after five more seconds expects at least two;
    # stops the watchdog and, after waiting, expects the count to stay constant over ten seconds.
    # Finally closes the file and expects it to contain the dumped current-thread header.
    tmp_file = self.create_tempfile()
    f = open(tmp_file, "w+")

    triggerred_count = [0]

    def on_triggered_fn():
      # [FUNCTION DOC]
      # HELPER: callback that increments the shared trigger counter. Returns nothing.
      triggerred_count[0] += 1

    timeout = 3
    if use_env_var:
      os.environ["TF_CLUSTER_COORDINATOR_WATCH_DOG_TIMEOUT"] = str(timeout)
      wd = watchdog.WatchDog(traceback_file=f, on_triggered=on_triggered_fn)
    else:
      wd = watchdog.WatchDog(
          timeout=timeout, traceback_file=f, on_triggered=on_triggered_fn)
    time.sleep(6)

    self.assertGreaterEqual(triggerred_count[0], 1)
    wd.report_closure_done()
    time.sleep(1)
    self.assertGreaterEqual(triggerred_count[0], 1)
    time.sleep(5)
    self.assertGreaterEqual(triggerred_count[0], 2)

    wd.stop()
    time.sleep(5)
    last_triggered_count = triggerred_count[0]
    time.sleep(10)
    self.assertEqual(last_triggered_count, triggerred_count[0])

    f.close()
    with open(tmp_file) as f:
      self.assertIn("Current thread", f.read())


if __name__ == "__main__":
  test.main()
