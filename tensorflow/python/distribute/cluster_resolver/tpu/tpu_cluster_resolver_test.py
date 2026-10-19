# [MODULE DOC]
# Unit tests for the TPU cluster resolver. Uses mock Cloud TPU service clients (fake node lookup
# tables, fake metadata server responses, fake GCE detection) to check cluster specifications,
# master addresses, coordinator handling, pod resolution, GKE environment variables, local TPU
# mode, accelerator counting with mocked devices, topology storage and the environment property.
# Defines small mock helper classes and functions used only by these tests. Side effects: some
# tests set and then delete an environment variable naming TPU endpoints, and some start a local
# TensorFlow server through the resolver.
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
"""Tests for TPUClusterResolver."""

import os

import six
from six.moves.urllib.error import URLError

from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver.tpu import tpu_cluster_resolver as resolver
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

mock = test.mock

try:
  from cloud_tpu_client import client  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.debug(
      'Falling back to TensorFlow client; we recommended you install the Cloud '
      'TPU client directly with pip install cloud-tpu-client.')
  from tensorflow.python.tpu.client import client  # pylint: disable=g-import-not-at-top


class MockRequestClass(object):
  # [CLASS DOC]
  # HELPER: fake API request object for the TPU resolver tests. Holds a node name and a mapping
  # from node names to node description records; executing it returns the record or fails.

  def __init__(self, name, tpu_map):
    # [FUNCTION DOC]
    # Stores the requested node name and the mapping from node names to node records.
    self._name = name
    self._tpu_map = tpu_map

  def execute(self):
    # [FUNCTION DOC]
    # Simulates running the request: returns the record for the stored name if present in the
    # mapping, otherwise raises a key error saying the resource was not found.
    if self._name in self._tpu_map:
      return self._tpu_map[self._name]
    else:
      raise KeyError('Resource %s was not found' % self._name)


class MockNodeClass(object):
  # [CLASS DOC]
  # HELPER: fake "nodes" resource of the Cloud TPU API for the tests; holds a mapping from node
  # names to node records and hands out fake request objects.

  def __init__(self, tpu_map):
    # [FUNCTION DOC]
    # Stores the mapping from node names to node records.
    self._tpu_map = tpu_map

  def get(self, name):
    # [FUNCTION DOC]
    # Returns a fake request object for the given node name, bound to the stored mapping.
    return MockRequestClass(name, self._tpu_map)


def mock_request_compute_metadata(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: replacement for the client's metadata-server request function in tests. Keyword
  # arguments are ignored. Given a metadata path as first argument, returns a fixed project id
  # for the project path, a fixed zone path for the zone path, a fixed internal IP address for
  # the network-interface path, and an empty string for anything else.
  del kwargs  # Unused.
  if args[0] == 'project/project-id':
    return 'test-project'
  elif args[0] == 'instance/zone':
    return 'projects/test-project/locations/us-central1-c'
  elif args[0] == 'instance/network-interfaces/0/ip':
    return '10.128.1.2'
  return ''


def mock_is_running_in_gce():
  # [FUNCTION DOC]
  # HELPER: test replacement for the GCE detection function; always returns true.
  return True


def mock_is_not_running_in_gce():
  # [FUNCTION DOC]
  # HELPER: test replacement for the GCE detection function; always returns false.
  return False


def mock_running_in_gce_urlopen(cls, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: fake URL-opening function simulating a reachable GCE metadata server. All arguments
  # are ignored; returns a mock response whose header info says the metadata flavor is Google.
  del cls, args, kwargs  # Unused.
  mock_response = mock.MagicMock()
  mock_response.info.return_value = {'Metadata-Flavor': 'Google'}
  return mock_response


def mock_not_running_in_gce_urlopen(cls, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: fake URL-opening function simulating an unreachable metadata server. All arguments
  # are ignored; always raises a URL error saying the host does not exist.
  del cls, args, kwargs  # Unused.
  raise URLError(reason='Host does not exist.')


@test_util.run_all_in_graph_and_eager_modes
class TPUClusterResolverTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the TPU cluster resolver, run in graph and eager modes. Uses a mock Cloud TPU
  # service client built by a shared helper, plus helpers for comparing cluster specifications
  # and for checking whether a TPU setting triggers API resolution.

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the cluster-spec tests. Asserts the cluster specification and
    # specifications rebuilt from it (itself, its cluster definition message, its dictionary
    # form) all equal the expected protocol-buffer text.
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
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

  def mock_service_client(self, tpu_map=None):
    # [FUNCTION DOC]
    # HELPER: builds a fake Cloud TPU service client for the tests. Input: tpu_map (optional
    # mapping from full node names to node records; defaults to empty). Builds a chain of mock
    # objects (client to projects to locations to nodes) whose final node lookup is backed by the
    # mapping, and returns the top-level mock client.

    if tpu_map is None:
      tpu_map = {}

    mock_locations = mock.MagicMock()
    mock_locations.nodes.return_value = MockNodeClass(tpu_map)

    mock_project = mock.MagicMock()
    mock_project.locations.return_value = mock_locations

    mock_client = mock.MagicMock()
    mock_client.projects.return_value = mock_project

    return mock_client

  @mock.patch.object(resolver, 'is_running_in_gce', mock_is_running_in_gce)
  def testCheckRunningInGceWithNoTpuName(self):
    # [FUNCTION DOC]
    # With GCE detection mocked to true, creating a resolver with an empty TPU name is expected to
    # raise a value error asking for a TPU name.
    with self.assertRaisesRegex(ValueError,
                                'Please provide a TPU Name to connect to.*'):
      resolver.TPUClusterResolver(tpu='')

  @mock.patch.object(six.moves.urllib.request, 'urlopen',
                     mock_running_in_gce_urlopen)
  def testIsRunningInGce(self):
    # [FUNCTION DOC]
    # With URL opening mocked to look like a GCE metadata server, asserts the GCE detection
    # function returns true.
    self.assertTrue(resolver.is_running_in_gce())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRetrieveProjectAndZoneFromMetadata(self):
    # [FUNCTION DOC]
    # Creates a resolver with no project or zone (taken from mocked metadata), one TPU name and a
    # coordinator name, backed by a mock service with one ready TPU. Asserts the cluster spec has
    # a coordinator job (local IP and the auto-started server port) and a worker job with the TPU
    # address, and that the master is the TPU's grpc address.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': 'HEALTHY'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project=None,
        zone=None,
        tpu=['test-tpu-1'],
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map),
        coordinator_name='coordinator')

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'coordinator'
      tasks { key: 0 value: '10.128.1.2:%s' }
    }
    job {
      name: 'worker'
      tasks { key: 0 value: '10.1.2.3:8470' }
    }
    """ % cluster_resolver._coordinator_port
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRetrieveProjectAndZoneFromMetadataNoCoordinator(self):
    # [FUNCTION DOC]
    # Same as the metadata test but without a coordinator name; asserts the cluster spec has only
    # the worker job and the master is the TPU's grpc address.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': 'HEALTHY'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project=None,
        zone=None,
        tpu=['test-tpu-1'],
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: '10.1.2.3:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testNotReadyCloudTpu(self):
    # [FUNCTION DOC]
    # The mock TPU is in the creating state; asserts that asking for the cluster spec raises a
    # runtime error.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'CREATING'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project=None,
        zone=None,
        tpu='test-tpu-1',
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    with self.assertRaises(RuntimeError):
      cluster_resolver.cluster_spec()

  def testSimpleSuccessfulRetrieval(self):
    # [FUNCTION DOC]
    # Explicit project, zone, TPU, coordinator name and address with a ready mock TPU; asserts the
    # cluster spec has the given coordinator address plus the worker address, and the master is
    # the worker grpc address.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': 'HEALTHY'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu=['test-tpu-1'],
        coordinator_name='coordinator',
        coordinator_address='10.128.1.5:10203',
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'coordinator' tasks { key: 0 value: '10.128.1.5:10203' } }
    job { name: 'worker' tasks { key: 0 value: '10.1.2.3:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')

  def testFailedMetadata(self):
    # [FUNCTION DOC]
    # Asks for a TPU name absent from the mock service; asserts the cluster spec request raises a
    # value error whose message says the TPU metadata could not be looked up.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'health': 'HEALTHY'
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='nonexistent-tpu',
        coordinator_name='coordinator',
        coordinator_address='10.128.1.5:10203',
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    with self.assertRaises(ValueError) as error_context:
      cluster_resolver.cluster_spec()

    self.assertIn('Could not lookup TPU metadata', str(error_context.exception))

  def testNewNetworkEndpointFormat(self):
    # [FUNCTION DOC]
    # The mock TPU record uses the newer list-of-network-endpoints format; asserts the cluster
    # spec contains the coordinator and the endpoint as the worker, and the master is its grpc
    # address.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'state': 'READY',
            'health': 'HEALTHY',
            'networkEndpoints': [{
                'ipAddress': '10.2.3.4',
                'port': 8470,
            }]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='test-tpu-1',
        coordinator_name='coordinator',
        coordinator_address='10.128.1.5:10203',
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'coordinator' tasks { key: 0 value: '10.128.1.5:10203' } }
    job { name: 'worker' tasks { key: 0 value: '10.2.3.4:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual('grpc://10.2.3.4:8470', cluster_resolver.master())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testPodResolution(self):
    # [FUNCTION DOC]
    # A mock TPU pod with four network endpoints and a coordinator name (metadata mocked);
    # asserts the cluster spec has the coordinator job and four numbered worker tasks, and the
    # master is the first worker's grpc address.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'state': 'READY',
            'health':
                'HEALTHY',
            'networkEndpoints': [
                {
                    'ipAddress': '10.2.3.4',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.5',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.6',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.7',
                    'port': 8470,
                },
            ]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        tpu='test-tpu-1',
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map),
        coordinator_name='coordinator')

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'coordinator',
      tasks { key: 0 value: '10.128.1.2:%s'}
    }
    job {
      name: 'worker'
      tasks { key: 0 value: '10.2.3.4:8470' }
      tasks { key: 1 value: '10.2.3.5:8470' }
      tasks { key: 2 value: '10.2.3.6:8470' }
      tasks { key: 3 value: '10.2.3.7:8470' }
    }
    """ % cluster_resolver._coordinator_port
    self._verifyClusterSpecEquality(actual_cluster_spec, str(expected_proto))
    self.assertEqual(cluster_resolver.master(), 'grpc://10.2.3.4:8470')

  def testPodResolutionNoCoordinator(self):
    # [FUNCTION DOC]
    # Four-endpoint pod without a coordinator; asserts the cluster spec has only four worker tasks
    # and the master is the first worker's grpc address.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'state': 'READY',
            'health':
                'HEALTHY',
            'networkEndpoints': [
                {
                    'ipAddress': '10.2.3.4',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.5',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.6',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.7',
                    'port': 8470,
                },
            ]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='test-tpu-1',
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.2.3.4:8470' }
      tasks { key: 1 value: '10.2.3.5:8470' }
      tasks { key: 2 value: '10.2.3.6:8470' }
      tasks { key: 3 value: '10.2.3.7:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual(cluster_resolver.master(), 'grpc://10.2.3.4:8470')

  def testGetMasterNoEntries(self):
    # [FUNCTION DOC]
    # Creating a resolver with an empty list of TPUs against an empty mock service is expected to
    # raise a value error.
    tpu_map = {}

    with self.assertRaises(ValueError):
      resolver.TPUClusterResolver(
          project='test-project',
          zone='us-central1-c',
          tpu=[],
          coordinator_name=None,
          credentials=None,
          service=self.mock_service_client(tpu_map=tpu_map))

  # TODO(saeta): Convert to parameterized test when included in OSS TF.
  def verifyShouldResolve(self, tpu, should_resolve):
    # [FUNCTION DOC]
    # HELPER: shared by the should-resolve tests. Creates a resolver for the given tpu setting
    # (name, address or list) against an empty mock service and asserts the client's API
    # availability flag equals should_resolve (boolean), with the setting in the failure message.
    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu=tpu,
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map={}))
    self.assertEqual(should_resolve,
                     cluster_resolver._cloud_tpu_client.api_available(),
                     "TPU: '%s'" % tpu)

  def testShouldResolveGrpc(self):
    # [FUNCTION DOC]
    # A grpc address as TPU setting should not trigger API resolution (expects false).
    self.verifyShouldResolve('grpc://10.1.2.3:8470', False)

  def testShouldResolveName(self):
    # [FUNCTION DOC]
    # A plain TPU name should trigger API resolution (expects true).
    self.verifyShouldResolve('mytpu', True)

  def testShouldResolveList(self):
    # [FUNCTION DOC]
    # A list holding a TPU name should trigger API resolution (expects true).
    self.verifyShouldResolve(['myothertpu'], True)

  def testShouldResolveGrpcPrefix(self):
    # [FUNCTION DOC]
    # A name that merely starts with the letters grpc but is not an address should still trigger
    # API resolution (expects true).
    self.verifyShouldResolve('grpctpu', True)

  def testNoCallComputeMetadata(self):
    # [FUNCTION DOC]
    # Creates a resolver from an explicit grpc address (no service mocks); asserts the master is
    # that address and the cluster spec has a single worker with the address host and port.
    cluster_resolver = resolver.TPUClusterResolver(tpu='grpc://10.1.2.3:8470')
    self.assertEqual('grpc://10.1.2.3:8470', cluster_resolver.master())
    self.assertEqual(
        server_lib.ClusterSpec({
            'worker': ['10.1.2.3:8470']
        }).as_dict(),
        cluster_resolver.cluster_spec().as_dict())

  def testGkeEnvironmentForDonut(self):
    # [FUNCTION DOC]
    # Sets the environment variable listing Kubernetes TPU endpoints to a single address, creates
    # a resolver with no arguments, and asserts the master and a one-worker cluster spec match
    # that address. Removes the environment variable at the end (not on failure).
    os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'] = 'grpc://10.120.27.5:8470'

    self.assertIn('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS', os.environ)

    cluster_resolver = resolver.TPUClusterResolver()
    self.assertEqual(
        compat.as_bytes('grpc://10.120.27.5:8470'),
        compat.as_bytes(cluster_resolver.master()))
    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.120.27.5:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

    del os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS']

  def testGkeEnvironmentForPod(self):
    # [FUNCTION DOC]
    # Sets the Kubernetes TPU endpoints environment variable to four comma-separated addresses,
    # creates a resolver with no arguments, and asserts the master is the first address and the
    # cluster spec has four worker tasks. Removes the variable at the end (not on failure).
    os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'] = ('grpc://10.120.27.5:8470,'
                                                     'grpc://10.120.27.6:8470,'
                                                     'grpc://10.120.27.7:8470,'
                                                     'grpc://10.120.27.8:8470')

    self.assertIn('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS', os.environ)
    cluster_resolver = resolver.TPUClusterResolver()
    self.assertEqual(
        compat.as_bytes('grpc://10.120.27.5:8470'),
        compat.as_bytes(cluster_resolver.master()))
    actual_cluster_spec = cluster_resolver.cluster_spec()
    expected_proto = """
    job {
      name: 'worker'
      tasks { key: 0 value: '10.120.27.5:8470' }
      tasks { key: 1 value: '10.120.27.6:8470' }
      tasks { key: 2 value: '10.120.27.7:8470' }
      tasks { key: 3 value: '10.120.27.8:8470' }
    }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

    del os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS']

  def testRpcDetectionForGrpcString(self):
    # [FUNCTION DOC]
    # A resolver created from a grpc address string should report that address as master.
    cluster_resolver = resolver.TPUClusterResolver(
        tpu='grpc://10.1.2.3:8470')
    self.assertEqual(cluster_resolver.master(), 'grpc://10.1.2.3:8470')

  def testOverrideTaskTypeAndIndexAndGetMaster(self):
    # [FUNCTION DOC]
    # With a four-endpoint mock pod, asserts the default master is the first worker, then sets the
    # task type to worker and task id to 3 on the resolver and asserts the master becomes the
    # fourth worker's grpc address.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'state': 'READY',
            'health':
                'HEALTHY',
            'networkEndpoints': [
                {
                    'ipAddress': '10.2.3.4',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.5',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.6',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.7',
                    'port': 8470,
                },
            ]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='test-tpu-1',
        coordinator_name=None,
        credentials=None,
        service=self.mock_service_client(tpu_map=tpu_map))

    self.assertEqual(cluster_resolver.master(), 'grpc://10.2.3.4:8470')

    cluster_resolver.task_type = 'worker'
    cluster_resolver.task_id = 3
    self.assertEqual(cluster_resolver.master(), 'grpc://10.2.3.7:8470')

  def testGetDeviceDictAndCoresWithTPUs(self):
    # [FUNCTION DOC]
    # Feeds eight TPU device attribute objects over four hosts to the device-grouping helper and
    # asserts eight total cores and a map from each host id to its two core ids.
    devices = [
        '/job:tpu_worker/task:0/device:TPU:0',
        '/job:tpu_worker/task:1/device:TPU:1',
        '/job:tpu_worker/task:2/device:TPU:0',
        '/job:tpu_worker/task:3/device:TPU:1',
        '/job:tpu_worker/task:0/device:TPU:4',
        '/job:tpu_worker/task:1/device:TPU:5',
        '/job:tpu_worker/task:2/device:TPU:4',
        '/job:tpu_worker/task:3/device:TPU:5',
    ]
    device_list = [
        session._DeviceAttributes(name, 'TPU', 1024, 0) for name in devices
    ]

    device_details = resolver.TPUClusterResolver._get_device_dict_and_cores(
        device_list)
    self.assertEqual(device_details.total_cores, 8)
    self.assertEqual(device_details.device_map,
                     {'0': ['0', '4'],
                      '1': ['1', '5'],
                      '2': ['0', '4'],
                      '3': ['1', '5']})

  def testGetDeviceDictAndCoresWithCPUsAndGPUs(self):
    # [FUNCTION DOC]
    # Feeds CPU and GPU device attribute objects to the device-grouping helper and asserts zero
    # cores and an empty host map (nothing matches the TPU pattern).
    devices = [
        '/job:tpu_worker/task:0/device:CPU:0',
        '/job:tpu_worker/task:1/device:CPU:0',
        '/job:tpu_worker/task:2/device:CPU:0',
        '/job:tpu_worker/task:3/device:CPU:0',
        '/job:tpu_worker/task:0/device:GPU:1',
        '/job:tpu_worker/task:1/device:GPU:1',
        '/job:tpu_worker/task:2/device:GPU:1',
        '/job:tpu_worker/task:3/device:GPU:1',
    ]
    device_list = [
        session._DeviceAttributes(name, 'XLA', 1024, 0) for name in devices
    ]

    device_dict, num_cores =\
        resolver.TPUClusterResolver._get_device_dict_and_cores(device_list)
    self.assertEqual(num_cores, 0)
    self.assertEqual(device_dict, {})

  def testVerifySameCoreCount(self):
    # [FUNCTION DOC]
    # Asserts the same-core-count helper returns 8 for one host with eight cores, 2 for two hosts
    # with two cores each, and raises a runtime error when hosts have different core counts.
    self.assertEqual(
        resolver.TPUClusterResolver
        ._verify_and_return_same_core_count({0: [0, 1, 2, 3, 4, 5, 6, 7]}), 8)
    self.assertEqual(
        resolver.TPUClusterResolver
        ._verify_and_return_same_core_count({
            0: [0, 1],
            1: [2, 3]
        }), 2)
    with self.assertRaises(RuntimeError):
      resolver.TPUClusterResolver._verify_and_return_same_core_count(
          {
              0: [0],
              1: [1, 2]
          })

  @mock.patch.object(config, 'list_logical_devices')
  @mock.patch.object(session.BaseSession, 'list_devices')
  @mock.patch.object(resolver, 'is_running_in_gce', mock_is_not_running_in_gce)
  def testNumAcceleratorsSuccess(self, mock_list_devices,
                                 mock_eager_list_devices):
    # [FUNCTION DOC]
    # Mock arguments are injected by patch decorators. Mocks the device listings with eight TPU
    # cores over four hosts and a four-endpoint mock pod, creates a resolver, and asserts the
    # accelerator count mapping reports two TPU cores per host.
    devices = [
        context.LogicalDevice('/job:tpu_worker/task:0/device:TPU:0', 'TPU'),
        context.LogicalDevice('/job:tpu_worker/task:1/device:TPU:1', 'TPU'),
        context.LogicalDevice('/job:tpu_worker/task:2/device:TPU:0', 'TPU'),
        context.LogicalDevice('/job:tpu_worker/task:3/device:TPU:1', 'TPU'),
        context.LogicalDevice('/job:tpu_worker/task:0/device:TPU:4', 'TPU'),
        context.LogicalDevice('/job:tpu_worker/task:1/device:TPU:5', 'TPU'),
        context.LogicalDevice('/job:tpu_worker/task:2/device:TPU:4', 'TPU'),
        context.LogicalDevice('/job:tpu_worker/task:3/device:TPU:5', 'TPU'),
    ]
    device_list = [
        session._DeviceAttributes(d.name, d.device_type, 1024, 0)
        for d in devices
    ]
    mock_eager_list_devices.return_value = devices
    mock_list_devices.return_value = device_list

    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'state': 'READY',
            'health':
                'HEALTHY',
            'networkEndpoints': [
                {
                    'ipAddress': '10.2.3.4',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.5',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.6',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.7',
                    'port': 8470,
                },
            ]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='test-tpu-1',
        service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual(cluster_resolver.num_accelerators(), {'TPU': 2})

  @mock.patch.object(config, 'list_logical_devices')
  @mock.patch.object(session.BaseSession, 'list_devices')
  @mock.patch.object(resolver, 'is_running_in_gce', mock_is_not_running_in_gce)
  def testNumAcceleratorsRetryFailure(self, mock_list_devices,
                                      mock_eager_list_devices):
    # [FUNCTION DOC]
    # Mock arguments are injected by patch decorators. Makes both device listings always raise a
    # deadline-exceeded error and asserts the accelerator count request eventually raises a
    # runtime error after retries (the test runs the full retry loop).
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/test-tpu-1': {
            'health':
                'HEALTHY',
            'networkEndpoints': [
                {
                    'ipAddress': '10.2.3.4',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.5',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.6',
                    'port': 8470,
                },
                {
                    'ipAddress': '10.2.3.7',
                    'port': 8470,
                },
            ]
        }
    }

    cluster_resolver = resolver.TPUClusterResolver(
        project='test-project',
        zone='us-central1-c',
        tpu='test-tpu-1',
        service=self.mock_service_client(tpu_map=tpu_map))
    mock_list_devices.side_effect = errors.DeadlineExceededError(
        None, None, 'timeout')
    mock_eager_list_devices.side_effect = errors.DeadlineExceededError(
        None, None, 'timeout')
    with self.assertRaises(RuntimeError):
      cluster_resolver.num_accelerators()

  def testLocalTpuResolver(self):
    # [FUNCTION DOC]
    # A resolver for the local TPU setting should return an empty master string.
    cr = resolver.TPUClusterResolver(tpu='local')
    self.assertEqual(cr.get_master(), '')

  def testTpuTopology(self):
    # [FUNCTION DOC]
    # With a local resolver, asserts the stored topology starts as null, then sets a serialized
    # topology message and asserts the hardware feature property is a hardware-feature message.
    cluster_resolver = resolver.TPUClusterResolver(tpu='local')
    self.assertIsNone(cluster_resolver._tpu_topology)

    # Test set with tpu topology proto.
    cluster_resolver.set_tpu_topology(
        serialized_tpu_topology=topology_pb2.TopologyProto(
            mesh_shape=[1, 1, 1, 1]).SerializeToString())
    self.assertIsInstance(cluster_resolver.tpu_hardware_feature,
                          topology_pb2.TPUHardwareFeature)

  def testEnvironment(self):
    # [FUNCTION DOC]
    # A local resolver should report an empty string as its environment.
    cluster_resolver = resolver.TPUClusterResolver(tpu='local')
    self.assertEqual(cluster_resolver.environment, '')


if __name__ == '__main__':
  test.main()
