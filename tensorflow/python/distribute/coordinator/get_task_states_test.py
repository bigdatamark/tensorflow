# [MODULE DOC]
# Test module for the task-state query of the eager context against a multi-process parameter
# server cluster. Defines a mixin test class with scenarios that kill and restart workers and
# parameter servers (including the coordination service leader) while a repeated timer polls the
# task states, plus a concrete test case with two workers and two parameter servers. Defines
# module constants for the polling interval and the coordination error payload key. Running the
# module as a script enables v2 behavior and starts the multi-process runner.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Get task states test for parameter server strategy in TF2."""

import time

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.training import server_lib

_PULL_FREQ_IN_SEC = 2
_COORDINATION_ERROR_PAYLOAD_KEY = (
    "type.googleapis.com/tensorflow.CoordinationServiceError"
)


class GetTaskStatesTest(object):  # pylint: disable=missing-docstring
  # [CLASS DOC]
  # Mixin with the task state test cases. Concrete subclasses must also derive from a test case
  # and call setUp with the worker and parameter server counts. Fixtures: a multi-process cluster
  # with an added chief, a coordination service configured on parameter server 0 with a short
  # heartbeat timeout, a parameter server strategy, a cluster coordinator, and a repeated timer
  # that refreshes the latest task states.

  def setUp(self, num_workers, num_ps):
    # [FUNCTION DOC]
    # Builds a multi-process gRPC cluster with the given worker and parameter server counts
    # (integers), adds a chief on an unused local port, configures a standalone coordination
    # service led by parameter server 0 (heartbeat timeout equal to the polling interval, new
    # incarnations may reconnect), creates the strategy and cluster coordinator, and starts a
    # repeated timer that refreshes the stored task states every polling interval. Returns nothing.
    super().setUp()

    self._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=num_workers, num_ps=num_ps, rpc_layer="grpc")
    self._cluster_def = self._cluster.cluster_resolver.cluster_spec().as_dict()
    self._cluster_def["chief"] = [
        "localhost:%d" % multi_worker_test_base.pick_unused_port()
    ]
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        server_lib.ClusterSpec(self._cluster_def), rpc_layer="grpc")

    context.context().configure_coordination_service(
        service_type="standalone",
        service_leader="/job:ps/replica:0/task:0",
        heartbeat_timeout_in_ms=_PULL_FREQ_IN_SEC * 1000,
        allow_new_incarnation_to_reconnect=True)
    self.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        cluster_resolver)
    self.cluster_coord = cluster_coordinator.ClusterCoordinator(self.strategy)

    self.num_workers = num_workers
    self.num_ps = num_ps

    self.states = None
    self.polling_thread = utils.RepeatedTimer(
        interval=_PULL_FREQ_IN_SEC, function=self.get_task_states)

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent tearDown, stops the polling timer, stops the cluster and drops the
    # reference to it. Returns nothing.
    super().tearDown()
    self.polling_thread.stop()
    self._cluster.stop()
    self._cluster = None

  def get_task_states(self):
    # [FUNCTION DOC]
    # HELPER: callback for the polling timer.
    # Queries the eager context for the state of every worker and parameter server task and stores
    # the resulting list (one entry per task, empty when healthy, an error otherwise) on the
    # test object. Returns nothing.
    self.states = context.context().get_task_states([("worker",
                                                      self.num_workers),
                                                     ("ps", self.num_ps)])

  def testAllTasksHealthy(self):
    # [FUNCTION DOC]
    # Test: after waiting about one and a half polling intervals, the state list has one entry per
    # task and all entries are empty (healthy).
    time.sleep(_PULL_FREQ_IN_SEC * 1.5)
    self.assertLen(self.states, self.num_workers + self.num_ps)
    for state in self.states:
      self.assertIsNone(state)

  def testWorkerPreempted(self):
    # [FUNCTION DOC]
    # Test: kills worker 0, waits, and expects its entry to be an unavailable error naming the
    # task, carrying the unavailable code and the coordination service error payload, while every
    # other entry is healthy. Then restarts the worker, updates the server definition, waits, and
    # expects all tasks healthy again.
    self._cluster.kill_task("worker", 0)
    time.sleep(_PULL_FREQ_IN_SEC * 2)
    self.assertLen(self.states, self.num_workers + self.num_ps)
    self.assertIsInstance(self.states[0], errors.UnavailableError)
    self.assertIn("/job:worker/replica:0/task:0", self.states[0]._message)
    self.assertEqual(self.states[0]._error_code, error_codes_pb2.UNAVAILABLE)
    self.assertIn(_COORDINATION_ERROR_PAYLOAD_KEY,
                  self.states[0]._experimental_payloads)
    for i in range(1, self.num_workers + self.num_ps):
      self.assertIsNone(self.states[i])
    self._cluster.start_task("worker", 0)
    context.context().update_server_def(context.get_server_def())
    time.sleep(_PULL_FREQ_IN_SEC * 2)
    for state in self.states:
      self.assertIsNone(state)

  def testPSPreempted(self):
    # [FUNCTION DOC]
    # Test: kills parameter server 1 and expects its entry to be an unavailable error naming the
    # task. Then simulates restart of the whole cluster (kill the rest, start all), updates the
    # server definition, waits, and expects the full list of tasks to be healthy.
    self._cluster.kill_task("ps", 1)
    time.sleep(_PULL_FREQ_IN_SEC * 2)
    self.assertLen(self.states, self.num_workers + self.num_ps)
    state_ix = self.num_workers + 1
    self.assertIsInstance(self.states[state_ix], errors.UnavailableError)
    self.assertIn("/job:ps/replica:0/task:1", self.states[state_ix]._message)
    self.assertEqual(self.states[state_ix]._error_code,
                     error_codes_pb2.UNAVAILABLE)
    # Simulate the restart of all the tasks.
    self._cluster.kill_task("ps", 0)
    for index in range(2, self.num_ps):
      self._cluster.kill_task("ps", index)
    for index in range(self.num_workers):
      self._cluster.kill_task("worker", index)
    for index in range(self.num_ps):
      self._cluster.start_task("ps", index)
    for index in range(self.num_workers):
      self._cluster.start_task("worker", index)
    context.context().update_server_def(context.get_server_def())
    time.sleep(_PULL_FREQ_IN_SEC * 2)
    self.assertLen(self.states, self.num_workers + self.num_ps)
    for state in self.states:
      self.assertIsNone(state)

  def testCoordinationServicePreempted(self):
    # [FUNCTION DOC]
    # Test: kills parameter server 0 (the coordination service leader), expects the stored states
    # to be empty because the service is unreachable, then restarts all tasks, updates the server
    # definition and expects a full list of healthy states.
    self._cluster.kill_task("ps", 0)
    time.sleep(_PULL_FREQ_IN_SEC * 2)
    # `states` is None since Coordination Service is not available.
    self.assertIsNone(self.states)
    # Simulate the restart of all the tasks.
    for index in range(1, self.num_ps):
      self._cluster.kill_task("ps", index)
    for index in range(self.num_workers):
      self._cluster.kill_task("worker", index)
    for index in range(self.num_ps):
      self._cluster.start_task("ps", index)
    for index in range(self.num_workers):
      self._cluster.start_task("worker", index)
    context.context().update_server_def(context.get_server_def())
    time.sleep(_PULL_FREQ_IN_SEC * 2)
    self.assertLen(self.states, self.num_workers + self.num_ps)
    for state in self.states:
      self.assertIsNone(state)


class MultiWorkerGetTaskStatesTest(GetTaskStatesTest, test.TestCase):
  # [CLASS DOC]
  # Concrete test case running the task state scenarios with two workers and two parameter
  # servers.
  """This covers the cases where multiple workers and PS are used."""

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the mixin setUp with two workers and two parameter servers. Returns nothing.
    super().setUp(2, 2)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
