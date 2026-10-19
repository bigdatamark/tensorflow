# [MODULE DOC]
# Fault tolerance tests for parameter server training when failures are detected through the
# coordination service. Defines a base test class that reuses the shared fault tolerance test
# base with the coordination service enabled and adds stricter checks (a parameter server failure
# must surface as a PS-unavailable error from join and schedule), plus single-worker and
# multi-worker concrete test classes. When run as a script it enables v2 behavior and starts the
# multi-process test main.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Fault tolerance tests for coordination service-based failure handling."""

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.distribute.coordinator import fault_tolerance_test_base
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test


class BaseCoordinationServiceTest(
    fault_tolerance_test_base.BaseFaultToleranceTest):
  # [CLASS DOC]
  # Mixin-style base test class (it is not itself a test case; concrete subclasses combine it
  # with the test case class). Inherits all fault tolerance tests from the shared base and
  # overrides setup to enable the coordination service, and overrides two tests with stricter
  # expectations.
  """Modify some tests to have stronger checks."""

  def setUp(self, num_workers, num_ps):
    # [FUNCTION DOC]
    # Setup with parameters: num_workers and num_ps (integer cluster sizes). Calls the base setup
    # with the same sizes and the coordination service flag turned on. Returns nothing.
    super().setUp(num_workers=num_workers, num_ps=num_ps, use_cs=True)

  def testJoinRaisesUnavailableErrorAtPsFailure(self):
    # [FUNCTION DOC]
    # Scenario: a parameter server task is run and killed via the base helper. Action: join the
    # cluster coordinator. Expected: a PS-unavailable error is raised.
    self._run_and_kill_ps_task()
    with self.assertRaises(cluster_coordinator.PSUnavailableError):
      self.cluster_coord.join()

  def testScheduleRaisesUnavailableErrorAtPsFailure(self):
    # [FUNCTION DOC]
    # Scenario: a parameter server task is run and killed via the base helper. Action: schedule a
    # trivial no-op function on the coordinator. Expected: a PS-unavailable error is raised.
    self._run_and_kill_ps_task()
    with self.assertRaises(cluster_coordinator.PSUnavailableError):
      self.cluster_coord.schedule(def_function.function(lambda: None))


class SingleWorkerCoordinationServiceTest(
    BaseCoordinationServiceTest, test.TestCase
):
  # [CLASS DOC]
  # Concrete test case running the coordination service fault tolerance tests against a cluster
  # of one worker and one parameter server.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the cluster fixture with one worker and one parameter server. Returns nothing.
    super().setUp(num_workers=1, num_ps=1)


class MultiWorkerCoordinationServiceTest(
    BaseCoordinationServiceTest, test.TestCase
):
  # [CLASS DOC]
  # Concrete test case running the coordination service fault tolerance tests against a cluster
  # of two workers and two parameter servers.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the cluster fixture with two workers and two parameter servers. Returns nothing.
    super().setUp(num_workers=2, num_ps=2)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
