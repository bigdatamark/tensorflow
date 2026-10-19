# [MODULE DOC]
# Unit tests for the TF_CONFIG based cluster resolver. Each test sets the TF_CONFIG environment
# variable to a JSON document and checks the cluster specification, master address, task type,
# task index and rpc layer reported by the resolver, including overrides and type casting, plus
# accelerator counting with mocked device listings. Side effect: the tests overwrite the process
# TF_CONFIG environment variable and never restore it.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for TFCONFIGClusterResolver."""

import os

from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


@test_util.run_all_in_graph_and_eager_modes
class TFConfigClusterResolverTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the TF_CONFIG based cluster resolver, run in both graph and eager modes.
  # Each test sets the TF_CONFIG environment variable itself; the class has a shared helper that
  # compares cluster specifications against an expected protocol-buffer text.

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the cluster-spec tests in this file.
    # Asserts that the given cluster specification, and specifications rebuilt from it (from the
    # specification itself, from its cluster definition message and from its dictionary form),
    # all equal the expected protocol-buffer text. Returns nothing; fails the test on mismatch.
    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

  def testNormalClusterSpecRead(self):
    # [FUNCTION DOC]
    # Sets TF_CONFIG with ps and worker jobs (dense task lists) and task ps 0, then asserts the
    # resolver's cluster spec matches the expected two-job layout.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """
    actual_cluster_spec = cluster_resolver.cluster_spec()
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testSparseClusterSpecRead(self):
    # [FUNCTION DOC]
    # Sets TF_CONFIG where the worker job is a sparse mapping (only task index 1 present) and
    # asserts the cluster spec keeps that sparse task numbering.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": {"1": "worker1:2222"}
      },
      "task": {
        "type": "worker",
        "index": 1
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 1 value: 'worker1:2222' } }
    """
    actual_cluster_spec = cluster_resolver.cluster_spec()
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testAutomaticMasterRead(self):
    # [FUNCTION DOC]
    # With TF_CONFIG naming task ps 0, asserts the master address is auto-detected as the
    # address of that task (the first ps host).
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual('ps0:2222', cluster_resolver.master())

  def testSpecifiedTaskTypeAndIndexMasterRead(self):
    # [FUNCTION DOC]
    # Asks for the master of worker task 1 explicitly while TF_CONFIG names ps 0, and asserts
    # the explicit arguments win and the worker 1 address is returned.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual('worker1:2222', cluster_resolver.master('worker', 1))

  def testSessionMasterRead(self):
    # [FUNCTION DOC]
    # TF_CONFIG contains a session master entry; asserts the master address returned is exactly
    # that entry, taking precedence over the cluster layout.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "session_master": "sessionmaster:2222",
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual('sessionmaster:2222', cluster_resolver.master())

  def testRpcLayerRead(self):
    # [FUNCTION DOC]
    # TF_CONFIG sets the rpc layer to grpc; asserts the master address is prefixed with the grpc
    # protocol.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual('grpc://ps0:2222', cluster_resolver.master())

  def testTaskTypeIndexRpcRead(self):
    # [FUNCTION DOC]
    # Asserts the task type, task index and rpc layer properties report the values from TF_CONFIG
    # (ps, 0, grpc).
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "ps",
        "index": 0
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual('ps', cluster_resolver.task_type)
    self.assertEqual(0, cluster_resolver.task_id)
    self.assertEqual('grpc', cluster_resolver.rpc_layer)

  def testParameterOverrides(self):
    # [FUNCTION DOC]
    # TF_CONFIG says ps 1; the resolver is built with constructor overrides ps 0 and the test
    # asserts the master, task type and index follow the overrides. It then changes task type,
    # task index and rpc layer through the property setters and asserts the master and the
    # properties reflect the new values.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "ps",
        "index": 1
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver(
        task_type='ps', task_id=0)

    self.assertEqual('grpc://ps0:2222', cluster_resolver.master())
    self.assertEqual('ps', cluster_resolver.task_type)
    self.assertEqual(0, cluster_resolver.task_id)

    cluster_resolver.task_type = 'worker'
    cluster_resolver.task_id = 1
    cluster_resolver.rpc_layer = 'test'

    self.assertEqual('test://worker1:2222', cluster_resolver.master())
    self.assertEqual('worker', cluster_resolver.task_type)
    self.assertEqual(1, cluster_resolver.task_id)
    self.assertEqual('test', cluster_resolver.rpc_layer)

  def testTaskTypeCastToString(self):
    # [FUNCTION DOC]
    # TF_CONFIG gives the task type as a number; asserts the task type property returns it as a
    # string.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "123456": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": 123456,
        "index": 0
      }
    }
    """
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual('123456', cluster_resolver.task_type)

  def testTaskIndexCastToInteger(self):
    # [FUNCTION DOC]
    # TF_CONFIG gives the task index as a string; asserts the task id property returns an
    # integer.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "ps",
        "index": "1"
      }
    }
    """
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual(1, cluster_resolver.task_id)

  def testTaskIndexOverride(self):
    # [FUNCTION DOC]
    # TF_CONFIG names worker index 0; the resolver is built with task id 1 and the test asserts
    # the task id property returns the override.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "worker": ["worker0:2222", "worker1:2222"]
      },
      "task": {
        "type": "worker",
        "index": "0"
      }
    }
    """
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver(
        task_id=1)
    self.assertEqual(1, cluster_resolver.task_id)

  def testZeroItemsInClusterSpecMasterRead(self):
    # [FUNCTION DOC]
    # TF_CONFIG is an empty document; asserts the master address is an empty string.
    os.environ['TF_CONFIG'] = """
    {}
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual('', cluster_resolver.master())

  def testOneItemInClusterSpecMasterRead(self):
    # [FUNCTION DOC]
    # TF_CONFIG has a single job with a single task; asserts the master address is an empty
    # string.
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "worker": ["worker0:2222"]
      }
    }
    """

    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    self.assertEqual('', cluster_resolver.master())

  @mock.patch.object(config, 'list_logical_devices')
  @mock.patch.object(session.BaseSession, 'list_devices')
  def testNumAcceleratorsFilterTasksByEnvVar(self, mock_list_devices,
                                             mock_eager_list_devices):
    # [FUNCTION DOC]
    # Mocks the eager logical-device listing and the session device listing (mock arguments are
    # injected by the patch decorators) with TPU and GPU devices spread over two jobs. TF_CONFIG
    # names task worker1 0. Asserts accelerator counts default to that task (two TPUs, two GPUs)
    # and that explicit task type and id arguments (worker2, 3) override it (one GPU).
    os.environ['TF_CONFIG'] = """
    {
      "cluster": {
        "worker1": ["w10:2222"],
        "worker2": ["w21:2222", "w22:2222", "w23:2222", "w24:2222"]
      },
      "rpc_layer": "grpc",
      "task": {
        "type": "worker1",
        "index": "0"
      }
    }
    """

    devices = [
        context.LogicalDevice('/job:worker1/task:0/device:TPU:0', 'TPU'),
        context.LogicalDevice('/job:worker1/task:0/device:TPU:1', 'TPU'),
        context.LogicalDevice('/job:worker1/task:0/device:GPU:0', 'GPU'),
        context.LogicalDevice('/job:worker1/task:0/device:GPU:1', 'GPU'),
        context.LogicalDevice('/job:worker2/task:1/device:TPU:2', 'TPU'),
        context.LogicalDevice('/job:worker2/task:2/device:TPU:3', 'TPU'),
        context.LogicalDevice('/job:worker2/task:3/device:GPU:2', 'GPU'),
        context.LogicalDevice('/job:worker2/task:4/device:GPU:3', 'GPU'),
    ]
    device_list = [
        session._DeviceAttributes(d.name, d.device_type, 1024, 0)
        for d in devices
    ]
    mock_eager_list_devices.return_value = devices
    mock_list_devices.return_value = device_list

    resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()

    # By default we read from TF_CONFIG
    self.assertEqual(resolver.num_accelerators(), {'TPU': 2, 'GPU': 2})

    # Override still works when we want it to
    self.assertEqual(resolver.num_accelerators(task_type='worker2', task_id=3),
                     {'GPU': 1})


if __name__ == '__main__':
  test.main()
