# [MODULE DOC]
# Tests for the multi-process test cluster helper: verifies that local and remote workers start
# and produce data, and that distributing a dataset does not block when some workers are stuck.
# Test module only.
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
"""Tests tf.data service cluster with local and remote workers."""

from absl.testing import parameterized

from tensorflow.core.protobuf import data_service_pb2
from tensorflow.python.data.experimental.kernel_tests.service import multi_process_cluster
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_ops
from tensorflow.python.ops import math_ops


class MultiProcessClusterTest(data_service_test_base.TestBase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests exercising the multi-process cluster fixture with varying numbers of
  # local and remote workers.
  """Verifies the local and remote workers are running and producing data."""

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[0, 1, 3], num_remote_workers=[0, 1, 3])))
  def testCluster(self, num_local_workers, num_remote_workers):
    # [FUNCTION DOC]
    # Parameters: number of local and remote workers. Builds the cluster; if there are no workers
    # at all the test returns early. Otherwise reads a ten element distributed range and expects a
    # full copy per worker (local plus remote), in any order.
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    num_workers = num_local_workers + num_remote_workers
    if num_workers == 0:
      return
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(
        dataset,
        num_workers * list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeNonblockingWithStuckWorkers(self):
    # [FUNCTION DOC]
    # Scenario: six remote workers (no local ones, remote deployment mode). The dataset's map
    # function makes whichever worker sees element zero sleep for a very long time. With hint
    # sharding and at most one outstanding request, reading a hundred elements must still
    # succeed and never contain zero, showing other workers keep serving.
    num_workers = 6
    # Avoids using local workers because it will stall the teardown
    # while separate worker processes can be killed easily.
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=0,
        num_remote_workers=num_workers,
        worker_addresses=["localhost"] * num_workers,
        deployment_mode=data_service_pb2.DEPLOYMENT_MODE_REMOTE,
    )

    num_elements = 10

    def force_one_worker_to_stall_map(x):
      # [FUNCTION DOC]
      # Map function: for element zero sleeps for a very long time (simulating a stuck worker)
      # and returns zero as a 64-bit integer; any other element is returned unchanged.
      # Simulates having a stuck worker.
      if math_ops.equal(x, 0):
        test_ops.sleep_op(sleep_seconds=10000)
        return math_ops.cast(0, dtypes.int64)
      else:
        return x

    dataset = dataset_ops.Dataset.range(num_elements, dtype=dtypes.int64)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = dataset.map(force_one_worker_to_stall_map)
    dataset = dataset.repeat()

    dataset = self.make_distributed_dataset(
        dataset,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT,
        # Makes sure only there is one client request at most.
        max_outstanding_requests=1,
    )

    get_next = self.getNext(dataset, requires_initialization=False)

    results = []
    for _ in range(100):
      results.append(self.evaluate(get_next()))

    self.assertNotIn(
        0,
        results,
        "The worker producing 0 should be sleeping so 0 should not show up in"
        " the results.",
    )


if __name__ == "__main__":
  multi_process_cluster.test_main()
