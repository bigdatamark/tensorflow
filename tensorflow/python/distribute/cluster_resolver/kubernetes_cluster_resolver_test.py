# [MODULE DOC]
# Unit tests for the Kubernetes cluster resolver using a mocked Kubernetes client that returns
# canned pod lists per label selector. Covers address ordering, multiple selectors and jobs,
# master computation, non-running pods and invalid executable location.
# Contains small helper functions that build the mock objects. Note that one helper replaces the
# Kubernetes module in the global module registry for the rest of the process.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for K8sClusterResolver."""

import sys

from tensorflow.python.distribute.cluster_resolver.kubernetes_cluster_resolver import ExecutableLocation
from tensorflow.python.distribute.cluster_resolver.kubernetes_cluster_resolver import KubernetesClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


def _mock_kubernetes_module():
  # [FUNCTION DOC]
  # HELPER: makes the Kubernetes library importable in tests.
  # Replaces the Kubernetes entry in the global module registry with a mock object. The
  # replacement is never undone, so it persists for the rest of the process. Returns nothing.
  sys.modules['kubernetes'] = mock.MagicMock()


def _mock_kubernetes_client(ret):
  # [FUNCTION DOC]
  # HELPER: builds a fake Kubernetes client. Input: mapping from label selector string to a pod
  # list object. The returned mock's list-pods-for-all-namespaces call looks up the pod list by
  # the label selector keyword argument (raising a key error for unknown selectors).
  mock_client = mock.MagicMock()
  mock_client.list_pod_for_all_namespaces.side_effect = (
      lambda *args, **kwargs: ret[kwargs['label_selector']])
  return mock_client


def _get_mock_pod_item(name, phase, host_ip):
  # [FUNCTION DOC]
  # HELPER: builds a fake pod object. Inputs: pod name, phase string (such as running or failed)
  # and host IP address. Returns a mock with a metadata part holding the name and a status part
  # holding the phase and host IP.
  mock_status = mock.Mock()
  mock_status.configure_mock(phase=phase, host_ip=host_ip)

  mock_metadata = mock.Mock()
  mock_metadata.configure_mock(name=name)

  mock_item = mock.Mock()
  mock_item.configure_mock(status=mock_status, metadata=mock_metadata)
  return mock_item


def _create_pod_list(*args):
  # [FUNCTION DOC]
  # HELPER: builds a fake pod list response. Inputs: any number of triples of pod name, phase and
  # host IP. Returns a mock whose items attribute is the list of fake pods.
  return mock.MagicMock(items=[_get_mock_pod_item(*x) for x in args])


class KubernetesClusterResolverTest(test.TestCase):
  # [CLASS DOC]
  # Tests the Kubernetes cluster resolver with an injected fake client (no real cluster needed).

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests in this class.
    # KNOWN ISSUE: the docstring mentions the TPU cluster resolver, but this test file is about
    # the Kubernetes resolver (copy-paste leftover).
    # Checks the cluster definition against the expected text proto, also after rebuilding a
    # cluster specification from the specification, its cluster definition and its dictionary
    # form. Returns nothing.
    """Verifies that the ClusterSpec generates the correct proto.

    We are testing this four different ways to ensure that the ClusterSpec
    returned by the TPUClusterResolver behaves identically to a normal
    ClusterSpec when passed into the generic ClusterSpec libraries.

    Args:
      cluster_spec: ClusterSpec returned by the TPUClusterResolver
      expected_proto: Expected protobuf
    """
    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(expected_proto,
                           server_lib.ClusterSpec(
                               cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(expected_proto,
                           server_lib.ClusterSpec(
                               cluster_spec.as_dict()).as_cluster_def())

  def testSingleItemSuccessfulRetrievalInCluster(self):
    # [FUNCTION DOC]
    # Scenario: one running pod for the default selector; resolver created with an injected client
    # and the within-cluster location.
    # Expected: a worker job with a single task at the pod host IP and port 8470.
    ret = _create_pod_list(
        ('tensorflow-abc123', 'Running', '10.1.2.3'),
    )

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client({'job-name=tensorflow': ret}),
        executable_location=ExecutableLocation.WITHIN_CLUSTER,
    )

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.3:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))

  def testValueErrorRaisedOnInvalidExecutableLocation(self):
    # [FUNCTION DOC]
    # Scenario: the Kubernetes library is replaced by a mock (permanently, for the process) and
    # the resolver is created with a null executable location and no injected client.
    # Expected: a value error (any message). Uses a deprecated regex-assertion alias.

    _mock_kubernetes_module()

    with self.assertRaisesRegexp(ValueError, '.*'):
      KubernetesClusterResolver(executable_location=None)

  def testSingleItemSuccessfulRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: one running pod for the default selector, injected client, default location.
    # Expected: a worker job with one task at the pod host IP and port 8470.
    ret = _create_pod_list(('tensorflow-abc123', 'Running', '10.1.2.3'),)

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client(
            {'job-name=tensorflow': ret}))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.3:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))

  def testSuccessfulRetrievalWithSort(self):
    # [FUNCTION DOC]
    # Scenario: three running pods whose names are not in sorted order.
    # Expected: tasks are numbered following the pods sorted by name.
    ret = _create_pod_list(
        ('tensorflow-abc123', 'Running', '10.1.2.3'),
        ('tensorflow-def456', 'Running', '10.1.2.4'),
        ('tensorflow-999999', 'Running', '10.1.2.5'))

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client(
            {'job-name=tensorflow': ret}))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.5:8470' }
      tasks { key: 1 value: '10.1.2.3:8470' }
      tasks { key: 2 value: '10.1.2.4:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))

  def testGetMasterWithOverrideParameters(self):
    # [FUNCTION DOC]
    # Scenario: three running worker pods; task type and index set as attributes (worker, zero).
    # Expected: attributes read back; master with no arguments is the first pod address with the
    # grpc scheme; master with explicit worker and index two is the third pod address.
    ret = _create_pod_list(
        ('worker-0', 'Running', '10.1.2.3'),
        ('worker-1', 'Running', '10.1.2.4'),
        ('worker-2', 'Running', '10.1.2.5'))

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client(
            {'job-name=tensorflow': ret}))
    cluster_resolver.task_type = 'worker'
    cluster_resolver.task_id = 0
    self.assertEqual(cluster_resolver.task_type, 'worker')
    self.assertEqual(cluster_resolver.task_id, 0)
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')
    self.assertEqual(cluster_resolver.master('worker', 2),
                     'grpc://10.1.2.5:8470')

  def testNonRunningPod(self):
    # [FUNCTION DOC]
    # Scenario: the only pod is in the failed phase.
    # Expected: requesting the cluster specification raises a runtime error naming the pod and
    # its phase.
    ret = _create_pod_list(('tensorflow-abc123', 'Failed', '10.1.2.3'),)

    cluster_resolver = KubernetesClusterResolver(
        override_client=_mock_kubernetes_client(
            {'job-name=tensorflow': ret}))

    error_msg = 'Pod "tensorflow-abc123" is not running; phase: "Failed"'
    with self.assertRaisesRegex(RuntimeError, error_msg):
      cluster_resolver.cluster_spec()

  def testMultiplePodSelectorsAndWorkers(self):
    # [FUNCTION DOC]
    # Scenario: a worker job with two selectors (three pods each) and a ps job with one selector
    # (two pods), all running.
    # Expected: ps job with two tasks; worker job with six tasks, pods sorted by name within each
    # selector and the selectors' results concatenated in order.
    worker1 = _create_pod_list(
        ('tensorflow-abc123', 'Running', '10.1.2.3'),
        ('tensorflow-def456', 'Running', '10.1.2.4'),
        ('tensorflow-999999', 'Running', '10.1.2.5'))
    worker2 = _create_pod_list(
        ('tensorflow-abc124', 'Running', '10.1.2.6'),
        ('tensorflow-def457', 'Running', '10.1.2.7'),
        ('tensorflow-999990', 'Running', '10.1.2.8'))
    ps = _create_pod_list(
        ('tensorflow-ps-1', 'Running', '10.1.2.1'),
        ('tensorflow-ps-2', 'Running', '10.1.2.2'))

    cluster_resolver = KubernetesClusterResolver(
        job_to_label_mapping={
            'worker': ['job-name=worker1', 'job-name=worker2'],
            'ps': ['job-name=ps']
        },
        override_client=_mock_kubernetes_client({
            'job-name=worker1': worker1,
            'job-name=worker2': worker2,
            'job-name=ps': ps
        }))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'ps'
      tasks { key: 0 value: '10.1.2.1:8470' }
      tasks { key: 1 value: '10.1.2.2:8470' }
    }
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.5:8470' }
      tasks { key: 1 value: '10.1.2.3:8470' }
      tasks { key: 2 value: '10.1.2.4:8470' }
      tasks { key: 3 value: '10.1.2.8:8470' }
      tasks { key: 4 value: '10.1.2.6:8470' }
      tasks { key: 5 value: '10.1.2.7:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))


if __name__ == '__main__':
  test.main()
