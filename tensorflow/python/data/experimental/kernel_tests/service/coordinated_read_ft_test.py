# [MODULE DOC]
# Fault tolerance tests for coordinated (round robin) reads in the tf.data service: adding workers,
# restarting a worker, and repeated stop/start of workers plus a dispatcher restart while consumers
# keep reading. Contains one test class and a standard test main entry point.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Fault tolerance tests for tf.data service coordinated reads."""

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class CoordinatedReadFTTest(data_service_test_base.TestBase,
                            parameterized.TestCase):
  # [CLASS DOC]
  # Fault-tolerance test case for coordinated (round robin) reads of the tf.data service. Starts
  # in-process test clusters and checks that consumers keep receiving consistent coordinated read
  # groups while workers are added, stopped, restarted, or the dispatcher is restarted.

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(workers_to_add=[1, 3, 10])))
  def testAddWorkers(self, workers_to_add):
    # [FUNCTION DOC]
    # Scenario (eager only): a cluster of three workers and seven consumers; 25 elements are read,
    # then 1, 3 or 10 workers are added. Reading continues until a zero has been seen from every
    # worker (old and new, i.e. all joined), then 25 more elements. Checks the results form valid
    # coordinated read groups for the consumer count, then stops the workers.
    starting_workers = 3
    cluster = data_service_test_base.TestCluster(num_workers=starting_workers)
    num_consumers = 7
    ds = self.make_coordinated_read_dataset(cluster, num_consumers)

    get_next = self.getNext(ds, requires_initialization=True)
    results = []
    zeros_seen = 0
    for _ in range(25):
      results.append(self.evaluate(get_next()))
      if results[-1] == 0:
        zeros_seen += 1
    for _ in range(workers_to_add):
      cluster.add_worker()
    # Read until all new workers have joined.
    while zeros_seen < starting_workers + workers_to_add:
      results.append(self.evaluate(get_next()))
      if results[-1] == 0:
        zeros_seen += 1
    # Read some more.
    for _ in range(25):
      results.append(self.evaluate(get_next()))

    self.checkCoordinatedReadGroups(results, num_consumers)
    cluster.stop_workers()

  @combinations.generate(test_base.eager_only_combinations())
  def testRestartWorker(self):
    # [FUNCTION DOC]
    # Scenario (eager only): three workers with a two-second shutdown quiet period (so workers do
    # not stop mid round) and five consumers. Reads 20 elements, stops worker 1, reads 20 more,
    # restarts it, reads until a zero (from the restarted worker) is seen, reads 20 more. Checks the
    # results form valid coordinated read groups and stops the workers.
    num_workers = 3
    # Set a shutdown quiet period to prevent workers from shutting down partway
    # through a round.
    cluster = data_service_test_base.TestCluster(
        num_workers, worker_shutdown_quiet_period_ms=2000)
    num_consumers = 5
    ds = self.make_coordinated_read_dataset(cluster, num_consumers)

    get_next = self.getNext(ds, requires_initialization=True)
    results = []

    self.read(get_next, results, 20)
    cluster.workers[1].stop()
    # Check that we can continue to read even with a worker stopped.
    self.read(get_next, results, 20)
    cluster.workers[1].restart()
    # Read until we get results from the restarted worker, then read some more.
    while results[-1] != 0:
      results.append(self.evaluate(get_next()))
    self.read(get_next, results, 20)
    self.checkCoordinatedReadGroups(results, num_consumers)
    cluster.stop_workers()

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.OFF,
              data_service_ops.ShardingPolicy.DYNAMIC
          ])))
  def testMultiStartStop(self, sharding_policy):
    # [FUNCTION DOC]
    # Scenario (eager only): OFF or DYNAMIC sharding, three workers with a two-second quiet
    # period, five consumers. Reads 20 elements; then for each worker stops it, reads 20, restarts
    # it, reads 20. Then adds a worker, restarts the dispatcher, stops all original workers and
    # reads 20 more. Checks the results form valid coordinated read groups; stops the workers.
    num_workers = 3
    # Set a shutdown quiet period to prevent workers from shutting down partway
    # through a round.
    cluster = data_service_test_base.TestCluster(
        num_workers, worker_shutdown_quiet_period_ms=2000)
    num_consumers = 5
    ds = self.make_coordinated_read_dataset(cluster, num_consumers,
                                            sharding_policy)

    get_next = self.getNext(ds, requires_initialization=True)
    results = []

    self.read(get_next, results, 20)
    for i in range(num_workers):
      cluster.workers[i].stop()
      self.read(get_next, results, 20)
      cluster.workers[i].restart()
      self.read(get_next, results, 20)

    cluster.add_worker()
    cluster.restart_dispatcher()
    for i in range(num_workers):
      cluster.workers[i].stop()
    self.read(get_next, results, 20)

    self.checkCoordinatedReadGroups(results, num_consumers)
    cluster.stop_workers()

if __name__ == "__main__":
  test.main()
