# [MODULE DOC]
# Unit tests for the generic cluster resolver classes (base class accelerator counting, the simple
# resolver and the union resolver). Defines a minimal mock resolver subclass and two test classes.
# One test class mocks device listing to check per-type accelerator counts; the other checks
# property overrides, master address formatting and merging of cluster specifications.
# No import-time side effects other than aliasing the mock library.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Cluster Resolvers."""

from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


class MockBaseClusterResolver(cluster_resolver.ClusterResolver):
  # [CLASS DOC]
  # HELPER: minimal concrete resolver used by the base-class tests below.
  # Concrete subclass of the abstract cluster resolver with trivial implementations of the
  # abstract methods, so that inherited behavior (accelerator counting) can be tested.

  def cluster_spec(self):
    # [FUNCTION DOC]
    # HELPER: stub implementation of the abstract method for tests.
    # Takes no arguments and returns a null value instead of a cluster specification.
    return None

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    # [FUNCTION DOC]
    # HELPER: stub implementation of the abstract method for tests.
    # Accepts optional task type, task index and RPC layer, ignores them and returns an empty
    # string as the master address.
    return ""

  def environment(self):
    # [FUNCTION DOC]
    # HELPER: stub implementation of the abstract method for tests.
    # Takes no arguments and returns an empty string as the environment name.
    return ""


@test_util.run_all_in_graph_and_eager_modes
class BaseClusterResolverTest(test.TestCase):
  # [CLASS DOC]
  # Tests the accelerator counting logic inherited from the base cluster resolver. Runs in both
  # graph and eager modes. Each test mocks both the eager logical-device listing and the session
  # device listing so the resolver sees a fixed set of devices.

  @mock.patch.object(config, "list_logical_devices")
  @mock.patch.object(session.BaseSession, "list_devices")
  def testNumAcceleratorsSuccess(self, mock_list_devices,
                                 mock_eager_list_devices):
    # [FUNCTION DOC]
    # Scenario: four GPUs on one worker task are reported by the mocked device listings.
    # Action: ask the mock resolver for its accelerator counts without filters.
    # Expected: a mapping with GPU count of four.
    devices = [
        context.LogicalDevice("/job:worker/task:0/device:GPU:0", "GPU"),
        context.LogicalDevice("/job:worker/task:0/device:GPU:1", "GPU"),
        context.LogicalDevice("/job:worker/task:0/device:GPU:2", "GPU"),
        context.LogicalDevice("/job:worker/task:0/device:GPU:3", "GPU"),
    ]
    device_list = [
        session._DeviceAttributes(d.name, d.device_type, 1024, 0)
        for d in devices
    ]
    mock_eager_list_devices.return_value = devices
    mock_list_devices.return_value = device_list

    resolver = MockBaseClusterResolver()
    self.assertEqual(resolver.num_accelerators(), {"GPU": 4})

  @mock.patch.object(config, "list_logical_devices")
  @mock.patch.object(session.BaseSession, "list_devices")
  def testNumAcceleratorsMultiDeviceSuccess(self, mock_list_devices,
                                            mock_eager_list_devices):
    # [FUNCTION DOC]
    # Scenario: one worker task reports four TPUs and four GPUs through the mocked listings.
    # Action: ask the resolver for accelerator counts without filters.
    # Expected: a mapping with four for TPU and four for GPU.
    devices = [
        context.LogicalDevice("/job:worker/task:0/device:TPU:0", "TPU"),
        context.LogicalDevice("/job:worker/task:0/device:TPU:1", "TPU"),
        context.LogicalDevice("/job:worker/task:0/device:TPU:2", "TPU"),
        context.LogicalDevice("/job:worker/task:0/device:TPU:3", "TPU"),
        context.LogicalDevice("/job:worker/task:0/device:GPU:0", "GPU"),
        context.LogicalDevice("/job:worker/task:0/device:GPU:1", "GPU"),
        context.LogicalDevice("/job:worker/task:0/device:GPU:2", "GPU"),
        context.LogicalDevice("/job:worker/task:0/device:GPU:3", "GPU"),
    ]
    device_list = [
        session._DeviceAttributes(d.name, d.device_type, 1024, 0)
        for d in devices
    ]
    mock_eager_list_devices.return_value = devices
    mock_list_devices.return_value = device_list

    resolver = MockBaseClusterResolver()
    self.assertEqual(resolver.num_accelerators(), {"TPU": 4, "GPU": 4})

  @mock.patch.object(config, "list_logical_devices")
  @mock.patch.object(session.BaseSession, "list_devices")
  def testNumAcceleratorsFilterTasks(self, mock_list_devices,
                                     mock_eager_list_devices):
    # [FUNCTION DOC]
    # Scenario: devices spread over two jobs and several tasks are reported by the mocks.
    # Action: query accelerator counts filtered by task type and task index three times.
    # Expected: worker1 task 0 has two TPUs and two GPUs; worker2 tasks 3 and 4 each have one GPU.
    devices = [
        context.LogicalDevice("/job:worker1/task:0/device:TPU:0", "TPU"),
        context.LogicalDevice("/job:worker1/task:0/device:TPU:1", "TPU"),
        context.LogicalDevice("/job:worker1/task:0/device:GPU:0", "GPU"),
        context.LogicalDevice("/job:worker1/task:0/device:GPU:1", "GPU"),
        context.LogicalDevice("/job:worker2/task:1/device:TPU:2", "TPU"),
        context.LogicalDevice("/job:worker2/task:2/device:TPU:3", "TPU"),
        context.LogicalDevice("/job:worker2/task:3/device:GPU:2", "GPU"),
        context.LogicalDevice("/job:worker2/task:4/device:GPU:3", "GPU"),
    ]
    device_list = [
        session._DeviceAttributes(d.name, d.device_type, 1024, 0)
        for d in devices
    ]
    mock_eager_list_devices.return_value = devices
    mock_list_devices.return_value = device_list

    resolver = MockBaseClusterResolver()
    self.assertEqual(resolver.num_accelerators(task_type="worker1", task_id=0),
                     {"TPU": 2, "GPU": 2})
    self.assertEqual(resolver.num_accelerators(task_type="worker2", task_id=3),
                     {"GPU": 1})
    self.assertEqual(resolver.num_accelerators(task_type="worker2", task_id=4),
                     {"GPU": 1})


class UnionClusterResolverTest(test.TestCase):
  # [CLASS DOC]
  # Tests the simple cluster resolver and the union cluster resolver: property getters and
  # setters, master address formatting and merging of cluster specifications (list-based and
  # sparse dictionary-based jobs, overlaps and conflicts).
  # TODO(frankchn): Transform to parameterized test after it is included in the
  # TF open source codebase.

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the merge tests in this class.
    # Checks that the cluster definition of the given specification matches the expected text
    # proto, and that the same holds after rebuilding a cluster specification from the specification
    # itself, from its cluster definition and from its dictionary form. Returns nothing.
    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

  def testSingleClusterResolver(self):
    # [FUNCTION DOC]
    # Scenario: a union resolver wraps a single simple resolver with ps and worker jobs.
    # Expected: the union cluster specification equals the original (two ps, three workers).
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    simple_resolver = cluster_resolver.SimpleClusterResolver(base_cluster_spec)
    union_resolver = cluster_resolver.UnionClusterResolver(simple_resolver)

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """
    actual_cluster_spec = union_resolver.cluster_spec()
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testInitSimpleClusterResolver(self):
    # [FUNCTION DOC]
    # Scenario: a simple resolver is constructed with task type, task index, environment,
    # accelerator counts and RPC layer.
    # Expected: each property getter returns the value supplied at construction.
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = cluster_resolver.SimpleClusterResolver(
        base_cluster_spec, task_type="ps",
        task_id=1, environment="cloud",
        num_accelerators={"GPU": 8},
        rpc_layer="grpc",
    )

    self.assertEqual(simple_resolver.task_type, "ps")
    self.assertEqual(simple_resolver.task_id, 1)
    self.assertEqual(simple_resolver.environment, "cloud")
    self.assertEqual(simple_resolver.num_accelerators(), {"GPU": 8})
    self.assertEqual(simple_resolver.rpc_layer, "grpc")

  def testOverrideSimpleClusterResolver(self):
    # [FUNCTION DOC]
    # Scenario: a simple resolver built with ps, task one and grpc.
    # Action: assign new task type, task index and RPC layer through the property setters.
    # Expected: the getters return the newly assigned values.
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = cluster_resolver.SimpleClusterResolver(
        base_cluster_spec, task_type="ps",
        task_id=1, environment="cloud",
        num_accelerators={"GPU": 8},
        rpc_layer="grpc",
    )

    simple_resolver.task_type = "worker"
    simple_resolver.task_id = 2
    simple_resolver.rpc_layer = "http"

    self.assertEqual(simple_resolver.task_type, "worker")
    self.assertEqual(simple_resolver.task_id, 2)
    self.assertEqual(simple_resolver.rpc_layer, "http")

  def testSimpleOverrideMasterWithTaskIndexZero(self):
    # [FUNCTION DOC]
    # Scenario: simple resolver with ps and worker jobs; master requested for worker task zero
    # with the grpc RPC layer.
    # Expected: the address is the first worker address prefixed with the grpc scheme.
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = cluster_resolver.SimpleClusterResolver(base_cluster_spec)
    actual_master = simple_resolver.master("worker", 0, rpc_layer="grpc")
    self.assertEqual(actual_master, "grpc://worker0:2222")

  def testSimpleOverrideMasterWithRpcLayer(self):
    # [FUNCTION DOC]
    # Scenario: simple resolver; master requested for worker task two with the grpc RPC layer.
    # Expected: the third worker address prefixed with the grpc scheme.
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = cluster_resolver.SimpleClusterResolver(base_cluster_spec)
    actual_master = simple_resolver.master("worker", 2, rpc_layer="grpc")
    self.assertEqual(actual_master, "grpc://worker2:2222")

  def testSimpleOverrideMaster(self):
    # [FUNCTION DOC]
    # Scenario: simple resolver; master requested for worker task two without an RPC layer.
    # Expected: the plain third worker address with no scheme prefix.
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    simple_resolver = cluster_resolver.SimpleClusterResolver(base_cluster_spec)
    actual_master = simple_resolver.master("worker", 2)
    self.assertEqual(actual_master, "worker2:2222")

  def testUnionClusterResolverGetProperties(self):
    # [FUNCTION DOC]
    # Scenario: a union of two simple resolvers with different task types, indices, environments,
    # accelerator counts and RPC layers.
    # Expected: the union reports the properties of the first resolver. After assigning new task
    # type, task index and RPC layer on the union, the getters return the assigned values.
    cluster_spec_1 = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    resolver1 = cluster_resolver.SimpleClusterResolver(
        cluster_spec_1, task_type="ps",
        task_id=1, environment="cloud",
        num_accelerators={"GPU": 8},
        rpc_layer="grpc",
    )

    cluster_spec_2 = server_lib.ClusterSpec({
        "ps": ["ps2:2222", "ps3:2222"],
        "worker": ["worker3:2222", "worker4:2222", "worker5:2222"]
    })
    resolver2 = cluster_resolver.SimpleClusterResolver(
        cluster_spec_2, task_type="worker",
        task_id=2, environment="local",
        num_accelerators={"GPU": 16},
        rpc_layer="http",
    )

    union_resolver = cluster_resolver.UnionClusterResolver(resolver1, resolver2)

    self.assertEqual(union_resolver.task_type, "ps")
    self.assertEqual(union_resolver.task_id, 1)
    self.assertEqual(union_resolver.environment, "cloud")
    self.assertEqual(union_resolver.num_accelerators(), {"GPU": 8})
    self.assertEqual(union_resolver.rpc_layer, "grpc")

    union_resolver.task_type = "worker"
    union_resolver.task_id = 2
    union_resolver.rpc_layer = "http"

    self.assertEqual(union_resolver.task_type, "worker")
    self.assertEqual(union_resolver.task_id, 2)
    self.assertEqual(union_resolver.rpc_layer, "http")

  def testTwoNonOverlappingJobMergedClusterResolver(self):
    # [FUNCTION DOC]
    # Scenario: two resolvers, one with only a ps job and one with only a worker job.
    # Expected: the merged specification contains both jobs unchanged.
    cluster_spec_1 = server_lib.ClusterSpec({
        "ps": [
            "ps0:2222",
            "ps1:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": [
            "worker0:2222",
            "worker1:2222",
            "worker2:2222"
        ]
    })
    cluster_resolver_1 = cluster_resolver.SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = cluster_resolver.SimpleClusterResolver(cluster_spec_2)

    union_cluster = cluster_resolver.UnionClusterResolver(
        cluster_resolver_1, cluster_resolver_2)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)

  def testMergedClusterResolverMaster(self):
    # [FUNCTION DOC]
    # Scenario: union of a ps-only and a worker-only resolver.
    # Expected: master with no arguments is an empty string; for worker one it is the plain
    # address; with the grpc RPC layer it carries the grpc scheme prefix.
    cluster_spec_1 = server_lib.ClusterSpec({
        "ps": [
            "ps0:2222",
            "ps1:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": [
            "worker0:2222",
            "worker1:2222",
            "worker2:2222"
        ]
    })
    cluster_resolver_1 = cluster_resolver.SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = cluster_resolver.SimpleClusterResolver(cluster_spec_2)

    union_cluster = cluster_resolver.UnionClusterResolver(
        cluster_resolver_1, cluster_resolver_2)

    unspecified_master = union_cluster.master()
    self.assertEqual(unspecified_master, "")

    specified_master = union_cluster.master("worker", 1)
    self.assertEqual(specified_master, "worker1:2222")

    rpc_master = union_cluster.master("worker", 1, rpc_layer="grpc")
    self.assertEqual(rpc_master, "grpc://worker1:2222")

  def testOverlappingJobMergedClusterResolver(self):
    # [FUNCTION DOC]
    # Scenario: two resolvers both define a worker job as lists (two and three addresses).
    # Expected: the merged worker job concatenates them in resolver order, giving five tasks
    # numbered zero to four.
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": [
            "worker4:2222",
            "worker5:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": [
            "worker0:2222",
            "worker1:2222",
            "worker2:2222"
        ]
    })
    cluster_resolver_1 = cluster_resolver.SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = cluster_resolver.SimpleClusterResolver(cluster_spec_2)

    union_cluster = cluster_resolver.UnionClusterResolver(
        cluster_resolver_1, cluster_resolver_2)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: 'worker4:2222' }
                         tasks { key: 1 value: 'worker5:2222' }
                         tasks { key: 2 value: 'worker0:2222' }
                         tasks { key: 3 value: 'worker1:2222' }
                         tasks { key: 4 value: 'worker2:2222' } }
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)

  def testOverlappingSparseJobMergedClusterResolverThrowError(self):
    # [FUNCTION DOC]
    # Scenario: two sparse worker jobs that both use task index seven.
    # Expected: computing the merged specification raises a key error because of the collision.
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": {
            7: "worker4:2222",
            9: "worker5:2222"
        }
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": {
            3: "worker0:2222",
            6: "worker1:2222",
            7: "worker2:2222"
        }
    })
    cluster_resolver_1 = cluster_resolver.SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = cluster_resolver.SimpleClusterResolver(cluster_spec_2)

    union_cluster = cluster_resolver.UnionClusterResolver(
        cluster_resolver_1, cluster_resolver_2)
    self.assertRaises(KeyError, union_cluster.cluster_spec)

  def testOverlappingDictAndListThrowError(self):
    # [FUNCTION DOC]
    # Scenario: one worker job given as a list (indices zero and one) and another as a sparse
    # mapping using indices one to three, so index one collides.
    # Expected: computing the merged specification raises a key error.
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": [
            "worker4:2222",
            "worker5:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": {
            1: "worker0:2222",
            2: "worker1:2222",
            3: "worker2:2222"
        }
    })
    cluster_resolver_1 = cluster_resolver.SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = cluster_resolver.SimpleClusterResolver(cluster_spec_2)

    union_cluster = cluster_resolver.UnionClusterResolver(
        cluster_resolver_1, cluster_resolver_2)
    self.assertRaises(KeyError, union_cluster.cluster_spec)

  def testOverlappingJobNonOverlappingKey(self):
    # [FUNCTION DOC]
    # Scenario: two sparse worker jobs with disjoint task indices (five, nine and three, six,
    # seven).
    # Expected: the merged job keeps all five indices with their original addresses, sorted.
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": {
            5: "worker4:2222",
            9: "worker5:2222"
        }
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": {
            3: "worker0:2222",
            6: "worker1:2222",
            7: "worker2:2222"
        }
    })
    cluster_resolver_1 = cluster_resolver.SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = cluster_resolver.SimpleClusterResolver(cluster_spec_2)

    union_cluster = cluster_resolver.UnionClusterResolver(
        cluster_resolver_1, cluster_resolver_2)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'worker' tasks { key: 3 value: 'worker0:2222' }
                         tasks { key: 5 value: 'worker4:2222' }
                         tasks { key: 6 value: 'worker1:2222' }
                         tasks { key: 7 value: 'worker2:2222' }
                         tasks { key: 9 value: 'worker5:2222' }}
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)

  def testMixedModeNonOverlappingKey(self):
    # [FUNCTION DOC]
    # Scenario: one worker job as a list (indices zero and one) and one as a sparse mapping with
    # indices three, six and seven.
    # Expected: the merged job contains all five tasks under their respective indices.
    cluster_spec_1 = server_lib.ClusterSpec({
        "worker": [
            "worker4:2222",
            "worker5:2222"
        ]
    })
    cluster_spec_2 = server_lib.ClusterSpec({
        "worker": {
            3: "worker0:2222",
            6: "worker1:2222",
            7: "worker2:2222"
        }
    })
    cluster_resolver_1 = cluster_resolver.SimpleClusterResolver(cluster_spec_1)
    cluster_resolver_2 = cluster_resolver.SimpleClusterResolver(cluster_spec_2)

    union_cluster = cluster_resolver.UnionClusterResolver(
        cluster_resolver_1, cluster_resolver_2)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: 'worker4:2222' }
                         tasks { key: 1 value: 'worker5:2222' }
                         tasks { key: 3 value: 'worker0:2222' }
                         tasks { key: 6 value: 'worker1:2222' }
                         tasks { key: 7 value: 'worker2:2222' }}
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)

  def testRetainSparseJobWithNoMerging(self):
    # [FUNCTION DOC]
    # Scenario: a union over a single resolver whose worker job is sparse (indices one, three,
    # five).
    # Expected: the sparse task indices are preserved rather than renumbered.
    base_cluster_spec = server_lib.ClusterSpec({
        "worker": {
            1: "worker0:2222",
            3: "worker1:2222",
            5: "worker2:2222"
        }
    })

    base_cluster_resolver = cluster_resolver.SimpleClusterResolver(
        base_cluster_spec)
    union_cluster = cluster_resolver.UnionClusterResolver(base_cluster_resolver)
    cluster_spec = union_cluster.cluster_spec()

    expected_proto = """
    job { name: 'worker' tasks { key: 1 value: 'worker0:2222' }
                         tasks { key: 3 value: 'worker1:2222' }
                         tasks { key: 5 value: 'worker2:2222' } }
    """
    self._verifyClusterSpecEquality(cluster_spec, expected_proto)


# TODO(saeta): Include tests for master resolution

if __name__ == "__main__":
  test.main()
