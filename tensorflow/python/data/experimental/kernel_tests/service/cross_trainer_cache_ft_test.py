# [MODULE DOC]
# Fault tolerance tests for the tf.data service cross-trainer cache. Starts in-process test
# clusters with a cache-enabled worker, reads from a distributed dataset, restarts the worker or
# the dispatcher mid-read, and checks the element sequence that follows. Runs under the standard
# test runner via the main entry point. No import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Fault tolerance tests for tf.data service cross-trainer cache."""

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class CrossTrainerCacheFtTest(data_service_test_base.TestBase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering worker and dispatcher restarts while a trainer reads through the
  # cross-trainer cache. Uses the shared data service test base for cluster and dataset helpers;
  # each test builds its own cluster with a single cache-enabled worker.
  """Fault tolerance tests for tf.data service cross-trainer cache."""

  @combinations.generate(test_base.default_test_combinations())
  def testWorkerRestart(self):
    # [FUNCTION DOC]
    # Scenario: one worker, an endless range dataset read as trainer "Trainer 1" with the cache.
    # Reads 100 elements and expects 0..99. Restarts the worker, reads until a zero is seen
    # (the restarted worker starting over), then expects the next 100 elements to be 1..100.
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    distributed_dataset = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))

    get_next = self.getNext(distributed_dataset)
    elements = self._get_next(get_next, 100)
    self.assertEqual(elements, list(range(100)))

    cluster.workers[0].restart()

    # Read until we get results from the restarted worker, then read some more.
    while self.evaluate(get_next()) != 0:
      pass

    elements = self._get_next(get_next, 100)
    self.assertEqual(elements, list(range(1, 101)))

  @combinations.generate(test_base.default_test_combinations())
  def testDispatcherRestart(self):
    # [FUNCTION DOC]
    # Scenario: same setup as the worker restart test. Reads 100 elements (expects 0..99),
    # restarts the dispatcher, then reads 100 more and expects 100..199, i.e. the workers and
    # the reader are unaffected by the dispatcher restart.
    cluster = self._create_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10000000).repeat()
    distributed_dataset = self.make_distributed_dataset(
        dataset,
        cluster,
        job_name="job",
        cross_trainer_cache=data_service_ops.CrossTrainerCache(
            trainer_id="Trainer 1"))

    get_next = self.getNext(distributed_dataset)
    elements = self._get_next(get_next, 100)
    self.assertEqual(elements, list(range(100)))

    cluster.restart_dispatcher()

    # Dispatcher restart should not affect the workers.
    elements = self._get_next(get_next, 100)
    self.assertEqual(elements, list(range(100, 200)))

  def _get_next(self, get_next, num_elements):
    # [FUNCTION DOC]
    # HELPER: reading utility for the restart tests in this class.
    # Calls the iterator callback the given number of times, evaluating each result, and returns
    # the list of evaluated elements.
    return [self.evaluate(get_next()) for _ in range(num_elements)]

  def _create_cluster(self,
                      num_workers,
                      cross_trainer_cache_size_bytes=10 * (2**30)):
    # [FUNCTION DOC]
    # HELPER: cluster fixture builder for the tests in this class.
    # Creates a test cluster with a dispatcher and no workers, then starts num_workers workers
    # (no shutdown quiet period, cache size given by cross_trainer_cache_size_bytes, default 10
    # GiB) and appends them to the cluster's worker list. Returns the cluster.
    cluster = data_service_test_base.TestCluster(num_workers=0)
    for _ in range(num_workers):
      worker = data_service_test_base.TestWorker(
          dispatcher_address=cluster.dispatcher_address(),
          shutdown_quiet_period_ms=0,
          cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes)
      worker.start()
      cluster.workers.append(worker)
    return cluster


if __name__ == "__main__":
  test.main()
