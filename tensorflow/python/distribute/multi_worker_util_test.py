# [MODULE DOC]
# Unit tests for the multi-worker utility module: normalization of cluster descriptions, chief
# detection, worker counting, unique task ids, collective leader selection, and the cluster
# validation helper. Pure logic tests with small hand-written cluster descriptions; no servers.

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
"""Tests for multi_worker_util."""

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.eager import test
from tensorflow.python.training import server_lib


class NormalizeClusterSpecTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the cluster description normalization function with the supported input kinds.

  def assert_same_cluster(self, lhs, rhs):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Converts both inputs to cluster specs and asserts their mapping
    # forms are equal.
    self.assertEqual(
        server_lib.ClusterSpec(lhs).as_dict(),
        server_lib.ClusterSpec(rhs).as_dict())

  def testDictAsInput(self):
    # [FUNCTION DOC]
    # Scenario: chief, worker and ps jobs given as a plain mapping. Asserts the normalized result
    # describes the same cluster.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assert_same_cluster(
        cluster_spec, multi_worker_util.normalize_cluster_spec(cluster_spec))

  def testClusterDefAsInput(self):
    # [FUNCTION DOC]
    # Scenario: the same cluster built as a cluster definition message. Asserts the normalized
    # result describes the same cluster.
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = "chief"
    job.tasks[0] = "127.0.0.1:1234"

    job = cluster_def.job.add()
    job.name = "worker"
    job.tasks[0] = "127.0.0.1:8964"
    job.tasks[1] = "127.0.0.1:2333"

    job = cluster_def.job.add()
    job.name = "ps"
    job.tasks[0] = "127.0.0.1:1926"
    job.tasks[1] = "127.0.0.1:3141"

    self.assert_same_cluster(
        cluster_def, multi_worker_util.normalize_cluster_spec(cluster_def))

  def testClusterSpecAsInput(self):
    # [FUNCTION DOC]
    # Scenario: input already a cluster spec object. Asserts normalization yields the same cluster.
    cluster_spec = server_lib.ClusterSpec({
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    })
    self.assert_same_cluster(
        cluster_spec, multi_worker_util.normalize_cluster_spec(cluster_spec))

  def testUnexpectedInput(self):
    # [FUNCTION DOC]
    # Scenario: a list of addresses is given. Asserts a value error with the message saying a
    # mapping, cluster spec or cluster definition is expected.
    cluster_spec = ["127.0.0.1:8964", "127.0.0.1:2333"]

    with self.assertRaisesRegex(
        ValueError,
        "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
        "`tf.train.ClusterDef` object"):
      multi_worker_util.normalize_cluster_spec(cluster_spec)


class IsChiefTest(test.TestCase):
  # [CLASS DOC]
  # Tests for chief detection, given explicit cluster, task type and task id.

  def testClusterWithChief(self):
    # [FUNCTION DOC]
    # Scenario: cluster with a chief job. Asserts the chief is chief and worker zero is not.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertTrue(multi_worker_util.is_chief(cluster_spec, "chief", 0))
    self.assertFalse(multi_worker_util.is_chief(cluster_spec, "worker", 0))

  def testClusterWithoutChief(self):
    # [FUNCTION DOC]
    # Scenario: workers only. Asserts worker zero is chief and worker one is not; asking about a
    # chief task raises a value error saying the type is not in the cluster; a task id beyond the
    # worker count raises a value error about exceeding the maximum id.
    cluster_spec = {"worker": ["127.0.0.1:8964", "127.0.0.1:2333"]}
    self.assertTrue(multi_worker_util.is_chief(cluster_spec, "worker", 0))
    self.assertFalse(multi_worker_util.is_chief(cluster_spec, "worker", 1))

    with self.assertRaisesRegex(
        ValueError, "`task_type` 'chief' not found in cluster_spec."):
      multi_worker_util.is_chief(cluster_spec, "chief", 0)

    with self.assertRaisesRegex(
        ValueError, "The `task_id` 2 exceeds the maximum id of worker."):
      multi_worker_util.is_chief(cluster_spec, "worker", 2)

  def testEvaluatorIsChief(self):
    # [FUNCTION DOC]
    # Scenario: cluster with an evaluator job. Asserts the evaluator is considered chief.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "evaluator": ["127.0.0.1:2019"]
    }
    self.assertTrue(multi_worker_util.is_chief(cluster_spec, "evaluator", 0))


class NumWorkersTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the worker counting function.

  def testCountWorker(self):
    # [FUNCTION DOC]
    # Scenario: one chief, two workers, two ps. Asserts the count is three for both chief and worker
    # task types (chief counts as a worker; ps excluded).
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.worker_count(cluster_spec, task_type="chief"), 3)
    self.assertEqual(
        multi_worker_util.worker_count(cluster_spec, task_type="worker"), 3)

  def testCountEvaluator(self):
    # [FUNCTION DOC]
    # Scenario: cluster with chief, workers and one evaluator. Asserts the count for the evaluator
    # type is one (chief and workers are not counted).
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "evaluator": ["127.0.0.1:7566"]
    }
    self.assertEqual(
        multi_worker_util.worker_count(cluster_spec, task_type="evaluator"), 1)

  def testTaskTypeNotFound(self):
    # [FUNCTION DOC]
    # Scenario: empty cluster, task type worker. Asserts a value error saying the type is not found
    # in the cluster.
    cluster_spec = {}
    with self.assertRaisesRegex(
        ValueError, "`task_type` 'worker' not found in cluster_spec."):
      multi_worker_util.worker_count(cluster_spec, task_type="worker")

  def testCountPs(self):
    # [FUNCTION DOC]
    # Scenario: asking for the count with task type ps. Asserts a value error about an unexpected
    # task type.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    # A "ps" job shouldn't call this method.
    with self.assertRaisesRegex(ValueError, "Unexpected `task_type` 'ps'"):
      multi_worker_util.worker_count(cluster_spec, task_type="ps")


class IdInClusterTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the function assigning unique ids to tasks within the cluster.

  def testChiefId(self):
    # [FUNCTION DOC]
    # Scenario: cluster with chief. Asserts the chief id is zero.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.id_in_cluster(cluster_spec, "chief", 0), 0)

  def testWorkerId(self):
    # [FUNCTION DOC]
    # Scenario: worker one in a cluster with a chief gets id two; in a cluster without a chief it
    # gets id one. Asserts both.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.id_in_cluster(cluster_spec, "worker", 1), 2)

    cluster_spec = {
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.id_in_cluster(cluster_spec, "worker", 1), 1)

  def testEvaluatorId(self):
    # [FUNCTION DOC]
    # Scenario: evaluator zero in a cluster with chief and workers. Asserts its id is zero.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "evaluator": ["127.0.0.1:7566"]
    }
    self.assertEqual(
        multi_worker_util.id_in_cluster(cluster_spec, "evaluator", 0), 0)

  def testPsId(self):
    # [FUNCTION DOC]
    # Scenario: id requested for a ps task. Asserts a value error saying there is no id for ps.
    cluster_spec = {"chief": ["127.0.0.1:1234"], "ps": ["127.0.0.1:7566"]}
    with self.assertRaisesRegex(ValueError,
                                "There is no id for task_type 'ps'"):
      multi_worker_util.id_in_cluster(cluster_spec, "ps", 0)

  def testMultipleChiefs(self):
    # [FUNCTION DOC]
    # Scenario: cluster with two chief tasks. Asserts a value error saying there must be at most one
    # chief job.
    cluster_spec = {
        "chief": ["127.0.0.1:8258", "127.0.0.1:7566"],
    }
    with self.assertRaisesRegex(ValueError,
                                "There must be at most one 'chief' job."):
      multi_worker_util.id_in_cluster(cluster_spec, "chief", 0)


class CollectiveLeaderTest(test.TestCase):
  # [CLASS DOC]
  # Tests for choosing the collective ops leader task name.

  def testChiefAsLeader(self):
    # [FUNCTION DOC]
    # Scenario: cluster with a chief, asked for worker zero. Asserts the leader is chief task zero.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.collective_leader(cluster_spec, "worker", 0),
        "/job:chief/replica:0/task:0")

  def testWorkerAsLeader(self):
    # [FUNCTION DOC]
    # Scenario: no chief, asked for worker one. Asserts the leader is worker task zero.
    cluster_spec = {
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    self.assertEqual(
        multi_worker_util.collective_leader(cluster_spec, "worker", 1),
        "/job:worker/replica:0/task:0")

  def testLeaderForEvaluator(self):
    # [FUNCTION DOC]
    # Scenario: asked for the evaluator task. Asserts the leader is the empty string.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"],
        "evaluator": ["127.0.0.1:2019"]
    }
    self.assertEqual(
        multi_worker_util.collective_leader(cluster_spec, "evaluator", 0), "")

  def testLocalLeader(self):
    # [FUNCTION DOC]
    # Scenario: empty cluster and no task type. Asserts the leader is the empty string.
    cluster_spec = {}
    self.assertEqual(
        multi_worker_util.collective_leader(cluster_spec, None, 0), "")


# Most of the validation logic is tested by above tests except for some.
class ClusterSpecValidationTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the private cluster validation helper, covering cases not hit by other tests.

  def testEvaluatorNotInCluster(self):
    # [FUNCTION DOC]
    # Scenario: cluster without evaluator. Asserts validation passes (no error) for chief, worker,
    # ps and evaluator task types, since an evaluator may be absent.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "worker": ["127.0.0.1:8964", "127.0.0.1:2333"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    multi_worker_util._validate_cluster_spec(cluster_spec, "chief", 0)
    multi_worker_util._validate_cluster_spec(cluster_spec, "worker", 0)
    multi_worker_util._validate_cluster_spec(cluster_spec, "ps", 0)
    multi_worker_util._validate_cluster_spec(cluster_spec, "evaluator", 0)

  def testWorkerNotInCluster(self):
    # [FUNCTION DOC]
    # Scenario: cluster with chief and ps only. Asserts validation of an evaluator passes while
    # validation of a worker raises a value error saying the type is not found.
    cluster_spec = {
        "chief": ["127.0.0.1:1234"],
        "ps": ["127.0.0.1:1926", "127.0.0.1:3141"]
    }
    multi_worker_util._validate_cluster_spec(cluster_spec, "evaluator", 0)
    with self.assertRaisesRegex(
        ValueError, "`task_type` 'worker' not found in cluster_spec."):
      multi_worker_util._validate_cluster_spec(cluster_spec, "worker", 0)


if __name__ == "__main__":
  test.main()
