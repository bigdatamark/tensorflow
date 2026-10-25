# [MODULE DOC]
# Unit tests for the Cloud TPU client module. Uses mock replacements for the GCE metadata
# request, the current-time source, the Cloud TPU API service (an in-memory map from node resource
# names to metadata records), time and sleep, and the URL opener. Covers environment helper
# functions, client construction in its variants, endpoint lookup, recoverability (state and
# out-of-memory symptoms with the exit flags), health, version and accelerator getters, waiting for
# health, and TPU version configuration. Runs under the TensorFlow test main.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for cloud tpu client."""

import datetime
import json
import os
import time
import urllib

from absl import flags

from tensorflow.python.platform import test
from tensorflow.python.tpu.client import client

FLAGS = flags.FLAGS

mock = test.mock

_UTCNOW_STR = '2000-01-01T00:30:00'


def mock_utcnow():
  # [FUNCTION DOC]
  # HELPER: replacement time source for the OOM tests. Takes nothing and returns a fixed
  # datetime (the first of January 2000 at 00:30:00) so symptom ages are deterministic.
  return datetime.datetime.strptime(_UTCNOW_STR, '%Y-%m-%dT%H:%M:%S')


def mock_request_compute_metadata(path):
  # [FUNCTION DOC]
  # HELPER: replacement for the GCE metadata request. Input: metadata path. Returns the test
  # project id, the test zone path, or a local ip address for the three known paths, and an
  # empty string otherwise.
  if path == 'project/project-id':
    return 'test-project'
  elif path == 'instance/zone':
    return 'projects/test-project/locations/us-central1-c'
  elif path == 'instance/network-interfaces/0/ip':
    return '10.128.1.2'
  return ''


class MockRequestClass:
  # [CLASS DOC]
  # HELPER: fake Cloud TPU API request object for tests. Holds a node resource name and a map
  # from names to metadata records, and returns the record when executed.

  def __init__(self, name, tpu_map):
    # [FUNCTION DOC]
    # Stores the requested node name and the mapping of known node names to metadata records.
    self._name = name
    self._tpu_map = tpu_map

  def execute(self):
    # [FUNCTION DOC]
    # Returns a copy of the record for the stored name. If its health field is a list it is
    # treated as a time series: the health is looked up at the current time value (the mocked
    # clock) and replaced by that single entry. Raises a key error when the name is unknown.
    if self._name in self._tpu_map:
      tpu_dict = self._tpu_map[self._name].copy()
      if isinstance(tpu_dict.get('health'), list):
        # Do extraction of health list to a single health string based on time.
        time_now = time.time()
        health_now = tpu_dict.get('health')[time_now]
        tpu_dict['health'] = health_now
      return tpu_dict
    else:
      raise KeyError('Resource %s was not found' % self._name)


class MockNodeClass:
  # [CLASS DOC]
  # HELPER: fake nodes collection of the Cloud TPU API for tests. Holds a map from node resource
  # names to metadata records and hands out fake request objects.

  def __init__(self, tpu_map):
    # [FUNCTION DOC]
    # Stores the mapping from node names to metadata records.
    self._tpu_map = tpu_map

  def get(self, name):
    # [FUNCTION DOC]
    # Input: node resource name. Returns a fake request object bound to that name and the map.
    return MockRequestClass(name, self._tpu_map)


class CloudTpuClientTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Cloud TPU client: environment helpers, construction, endpoints,
  # recoverability, health and version getters, waiting for health and version configuration.
  # Shared fixtures: a fake API service built from a node map, and a fake clock advanced by
  # the fake sleep.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: removes the discovery URL and TPU name environment variables if set,
    # resets the fake clock to zero, and registers cleanup that stops all active patches.
    super().setUp()
    if 'TPU_API_DISCOVERY_URL' in os.environ:
      del os.environ['TPU_API_DISCOVERY_URL']
    if 'TPU_NAME' in os.environ:
      del os.environ['TPU_NAME']
    self._time_now = 0
    self.addCleanup(mock.patch.stopall)

  def _mock_time(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: fake wall clock. Ignores its arguments and returns the current fake time value.
    return self._time_now

  def _mock_sleep(self, secs):
    # [FUNCTION DOC]
    # HELPER: fake sleep. Input: seconds; advances the fake clock by that amount without waiting.
    self._time_now += secs

  def mock_service_client(self, tpu_map=None):
    # [FUNCTION DOC]
    # HELPER: builds a fake Cloud TPU API service. Input: optional mapping from node resource
    # names to metadata records (empty if missing). Returns a mock whose chain of project,
    # locations and nodes calls ends in the fake nodes collection backed by that mapping.
    if tpu_map is None:
      tpu_map = {}

    mock_locations = mock.MagicMock()
    mock_locations.nodes.return_value = MockNodeClass(tpu_map)

    mock_project = mock.MagicMock()
    mock_project.locations.return_value = mock_locations

    mock_client = mock.MagicMock()
    mock_client.projects.return_value = mock_project
    return mock_client

  def testEnvironmentDiscoveryUrl(self):
    # [FUNCTION DOC]
    # Sets the discovery URL environment variable and asserts the helper returns that value.
    os.environ['TPU_API_DISCOVERY_URL'] = 'https://{api}.internal/{apiVersion}'
    self.assertEqual('https://{api}.internal/{apiVersion}',
                     (client._environment_discovery_url()))

  def testEnvironmentGCEDefault(self):
    # [FUNCTION DOC]
    # With no metadata overrides set, asserts the metadata endpoint is the default Google
    # internal metadata host with the http scheme.
    self.assertEqual(
        'http://metadata.google.internal', client._gce_metadata_endpoint()
    )

  @mock.patch.dict(os.environ, {'GCE_METADATA_IP': '1.2.3.4'})
  def testEnvironmentGCEIPOverride(self):
    # [FUNCTION DOC]
    # With the metadata IP environment variable patched in, asserts the endpoint uses that
    # address with the http scheme.
    self.assertEqual('http://1.2.3.4', client._gce_metadata_endpoint())

  @mock.patch.dict(os.environ, {'GCE_METADATA_HOST': 'foo.bar'})
  def testEnvironmentGCEHostOverride(self):
    # [FUNCTION DOC]
    # With the metadata host environment variable patched in, asserts the endpoint uses that
    # host with the http scheme.
    self.assertEqual('http://foo.bar', client._gce_metadata_endpoint())

  def testEnvironmentVarToNetworkEndpointsSingleIp(self):
    # [FUNCTION DOC]
    # Parses a single address with a port and asserts one endpoint with that ip and port.
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '1234'}],
        list(client._environment_var_to_network_endpoints(
            '1.2.3.4:1234')))

  def testEnvironmentVarToNetworkEndpointsSingleGrpcAddress(self):
    # [FUNCTION DOC]
    # Parses one address with the grpc scheme prefix and asserts the prefix is stripped.
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '2000'}],
        list(
            client._environment_var_to_network_endpoints(
                'grpc://1.2.3.4:2000')))

  def testEnvironmentVarToNetworkEndpointsMultipleIps(self):
    # [FUNCTION DOC]
    # Parses a comma separated list of two address-port pairs and asserts two endpoints in order.
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '2000'},
         {'ipAddress': '5.6.7.8', 'port': '1234'}],
        list(
            client._environment_var_to_network_endpoints(
                '1.2.3.4:2000,5.6.7.8:1234')))

  def testEnvironmentVarToNetworkEndpointsMultipleGrpcAddresses(self):
    # [FUNCTION DOC]
    # Parses a list of two grpc-prefixed addresses and asserts both are returned without prefix.
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '2000'},
         {'ipAddress': '5.6.7.8', 'port': '1234'}],
        list(client._environment_var_to_network_endpoints(
            'grpc://1.2.3.4:2000,grpc://5.6.7.8:1234')))

  def testEnvironmentVarToNetworkEndpointsMissingPortAndMixed(self):
    # [FUNCTION DOC]
    # Parses a mix of a plain address with port and a grpc address without port; asserts the
    # missing port becomes the default TPU port.
    self.assertEqual(
        [{'ipAddress': '1.2.3.4', 'port': '2000'},
         {'ipAddress': '5.6.7.8', 'port': '8470'}],
        list(client._environment_var_to_network_endpoints(
            '1.2.3.4:2000,grpc://5.6.7.8')))

  def testInitializeNoArguments(self):
    # [FUNCTION DOC]
    # Creates a client with no arguments and no environment hints; asserts a value error asking
    # for a TPU name.
    with self.assertRaisesRegex(
        ValueError, 'Please provide a TPU Name to connect to.'):
      client.Client()

  def testInitializeMultiElementTpuArray(self):
    # [FUNCTION DOC]
    # Creates a client with a two-element TPU list; asserts a not-implemented error about using
    # multiple TPUs in one session.
    with self.assertRaisesRegex(
        NotImplementedError,
        'Using multiple TPUs in a single session is not yet implemented'):
      client.Client(tpu=['multiple', 'elements'])

  def assertClientContains(self, c):
    # [FUNCTION DOC]
    # HELPER: shared assertion for construction tests. Input: a client. Asserts it uses the API,
    # has the name tpu_name, the test project and zone, no credentials or discovery URL, and
    # that its network endpoints are the single ten-dot-one-dot-two-dot-three address on port 8470.
    self.assertEqual('tpu_name', c._tpu)
    self.assertEqual(True, c._use_api)
    self.assertIsNone(c._credentials)
    self.assertEqual('test-project', c._project)
    self.assertEqual('us-central1-c', c._zone)
    self.assertIsNone(c._discovery_url)
    self.assertEqual([{
        'ipAddress': '10.1.2.3',
        'port': '8470'
    }], c.network_endpoints())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testNetworkEndpointsNotReadyWithApi(self):
    # [FUNCTION DOC]
    # Fake node record has no state. Asserts asking for network endpoints raises a runtime
    # error saying the TPU is not yet ready with state none.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertRaisesRegex(
        RuntimeError, 'TPU .* is not yet ready; state: "None"',
        c.network_endpoints)

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testInitializeNoArgumentsWithEnvironmentVariable(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: sets the TPU name environment variable without restoring it; only the setup
    # method removes it before the next test, so leakage is limited to the end of the run.
    # Sets the TPU name environment variable and creates a client with only a fake service;
    # asserts the name, project, zone and endpoints are resolved as expected.
    os.environ['TPU_NAME'] = 'tpu_name'
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': 'HEALTHY',
        }
    }
    c = client.Client(
        service=self.mock_service_client(tpu_map=tpu_map))
    self.assertClientContains(c)

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testInitializeNoArgumentsWithTPUEnvironmentVariableTPUConfig(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: sets the TPU config environment variable and never clears it (setup only
    # clears the name and discovery variables), so later tests that rely on no name being
    # available may see this config.
    # Sets the TPU config variable with project, zone and node name, creates a client with only a
    # fake service, and asserts the values are taken from the config.
    os.environ['TPU_CONFIG'] = json.dumps({
        'project': 'test-project',
        'zone': 'us-central1-c',
        'tpu_node_name': 'tpu_name',
    })
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': 'HEALTHY',
        }
    }
    c = client.Client(service=self.mock_service_client(tpu_map=tpu_map))
    self.assertClientContains(c)

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testInitializeTpuName(self):
    # [FUNCTION DOC]
    # Creates a client with an explicit name and a fake service; asserts the metadata-derived
    # project and zone and the endpoints.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': 'HEALTHY',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertClientContains(c)

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testInitializeIpAddress(self):
    # [FUNCTION DOC]
    # Creates a client from a grpc address; asserts the API is not used, project, zone, service,
    # credentials and discovery URL are unset, and the endpoint is parsed from the address.
    c = client.Client(tpu='grpc://1.2.3.4:8470')
    self.assertEqual('grpc://1.2.3.4:8470', c._tpu)
    self.assertEqual(False, c._use_api)
    self.assertIsNone(c._service)
    self.assertIsNone(c._credentials)
    self.assertIsNone(c._project)
    self.assertIsNone(c._zone)
    self.assertIsNone(c._discovery_url)
    self.assertEqual([{
        'ipAddress': '1.2.3.4',
        'port': '8470'
    }], c.network_endpoints())

  def testInitializeWithoutMetadata(self):
    # [FUNCTION DOC]
    # Creates a client with explicit name, project and zone and no metadata mock; asserts the
    # given values are stored and the other fields are unset, meaning no metadata call is needed.
    c = client.Client(
        tpu='tpu_name', project='project', zone='zone')
    self.assertEqual('tpu_name', c._tpu)
    self.assertEqual(True, c._use_api)
    self.assertIsNone(c._service)
    self.assertIsNone(c._credentials)
    self.assertEqual('project', c._project)
    self.assertEqual('zone', c._zone)
    self.assertIsNone(c._discovery_url)

  def testRecoverableNoApiAccess(self):
    # [FUNCTION DOC]
    # A client for a direct grpc address cannot query state; asserts it is considered recoverable.
    c = client.Client(tpu='grpc://1.2.3.4:8470')
    self.assertEqual(True, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRecoverableNoState(self):
    # [FUNCTION DOC]
    # Fake node has no state field; asserts the TPU is considered recoverable.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual(True, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRecoverableReady(self):
    # [FUNCTION DOC]
    # Fake node in the ready state; asserts the TPU is considered recoverable.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual(True, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRecoverablePreempted(self):
    # [FUNCTION DOC]
    # Fake node in the preempted state; asserts the TPU is not considered recoverable.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'PREEMPTED',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual(False, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  @mock.patch.object(client, '_utcnow', mock_utcnow)
  def testRecoverableOOM(self):
    # [FUNCTION DOC]
    # With the clock fixed at 00:30:00, runs several fake nodes with different symptom lists.
    # Asserts not recoverable only when an out-of-memory symptom is newer than ninety seconds
    # (30 seconds old), and recoverable with no symptoms, an old one (100 seconds), or only
    # other symptom types.
    test_cases = [
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
            }
        }, True),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'OUT_OF_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, False),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:28:20.123456Z',
                    'symptomType': 'OUT_OF_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:28:40.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'OUT_OF_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:40.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, False),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:28:20.123456Z',
                    'symptomType': 'OUT_OF_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:40.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:29:00.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:10.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:20.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:40.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True)
    ]

    for tpu_map, want in test_cases:
      c = client.Client(tpu='tpu_name',
                        service=self.mock_service_client(tpu_map=tpu_map))
      self.assertEqual(want, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  @mock.patch.object(client, '_utcnow', mock_utcnow)
  def testRecoverableOOMDisabled(self):
    # [FUNCTION DOC]
    # Turns the runtime-OOM exit flag off, uses a node with a fresh OOM symptom, and asserts the
    # TPU is still recoverable. Restores the flag to on afterwards (not restored if an assertion
    # fails first).
    test_cases = [
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'OUT_OF_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True),
    ]

    FLAGS.runtime_oom_exit = False
    for tpu_map, want in test_cases:
      c = client.Client(tpu='tpu_name',
                        service=self.mock_service_client(tpu_map=tpu_map))
      self.assertEqual(want, c.recoverable())
    FLAGS.runtime_oom_exit = True

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  @mock.patch.object(client, '_utcnow', mock_utcnow)
  def testRecoverableOOMNoAPI(self):
    # [FUNCTION DOC]
    # Client for a direct grpc address with a fake service holding a fresh OOM symptom; asserts
    # it is recoverable because symptoms are not read without the API.
    test_cases = [
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'OUT_OF_MEMORY',
                    'details': 'The TPU runtime has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True),
    ]

    for tpu_map, want in test_cases:
      c = client.Client(tpu='grpc://1.2.3.4:8470',
                        service=self.mock_service_client(tpu_map=tpu_map))
      self.assertEqual(want, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  @mock.patch.object(client, '_utcnow', mock_utcnow)
  def testRecoverableHBMOOM(self):
    # [FUNCTION DOC]
    # Same scenarios as the runtime OOM test but with high-bandwidth-memory out-of-memory
    # symptoms. Asserts not recoverable only when such a symptom is newer than ninety seconds.
    test_cases = [
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
            }
        }, True),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'HBM_OUT_OF_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, False),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:28:20.123456Z',
                    'symptomType': 'HBM_OUT_OF_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:28:40.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'HBM_OUT_OF_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:40.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, False),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:28:20.123456Z',
                    'symptomType': 'HBM_OUT_OF_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:40.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True),
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:29:00.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:10.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:20.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }, {
                    'createTime': '2000-01-01T00:29:40.123456Z',
                    'symptomType': 'LOW_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True)
    ]

    for tpu_map, want in test_cases:
      c = client.Client(tpu='tpu_name',
                        service=self.mock_service_client(tpu_map=tpu_map))
      self.assertEqual(want, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  @mock.patch.object(client, '_utcnow', mock_utcnow)
  def testRecoverableHBMOOMDisabled(self):
    # [FUNCTION DOC]
    # Turns the HBM-OOM exit flag off with a fresh HBM OOM symptom; asserts the TPU is still
    # recoverable, then restores the flag to on.
    test_cases = [
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'HBM_OUT_OF_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True),
    ]

    FLAGS.hbm_oom_exit = False
    for tpu_map, want in test_cases:
      c = client.Client(tpu='tpu_name',
                        service=self.mock_service_client(tpu_map=tpu_map))
      self.assertEqual(want, c.recoverable())
    FLAGS.hbm_oom_exit = True

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  @mock.patch.object(client, '_utcnow', mock_utcnow)
  def testRecoverableHBMOOMNoAPI(self):
    # [FUNCTION DOC]
    # Client for a direct grpc address with a fresh HBM OOM symptom in the fake service; asserts
    # it is recoverable since symptoms are not read without the API.
    test_cases = [
        ({
            'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
                'state':
                    'READY',
                'symptoms': [{
                    'createTime': '2000-01-01T00:29:30.123456Z',
                    'symptomType': 'HBM_OUT_OF_MEMORY',
                    'details': 'The TPU HBM has run OOM at timestamp '
                               '2020-05-29T04:51:32.038721+00:00',
                    'workerId': '0'
                }]
            }
        }, True),
    ]

    for tpu_map, want in test_cases:
      c = client.Client(tpu='grpc://1.2.3.4:8470',
                        service=self.mock_service_client(tpu_map=tpu_map))
      self.assertEqual(want, c.recoverable())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testHealthApi(self):
    # [FUNCTION DOC]
    # Fake node with health healthy; asserts the health getter returns that string.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'PREEMPTED',
            'health': 'HEALTHY',
            'acceleratorType': 'v3-8',
            'tensorflowVersion': 'nightly',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual('HEALTHY', c.health())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testRuntimeVersionApi(self):
    # [FUNCTION DOC]
    # Fake node with a nightly software version; asserts the runtime version getter returns it.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'PREEMPTED',
            'health': 'HEALTHY',
            'acceleratorType': 'v3-8',
            'tensorflowVersion': 'nightly',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual('nightly', c.runtime_version())

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testAcceleratorTypeApi(self):
    # [FUNCTION DOC]
    # Fake node with accelerator type v3-8; asserts the accelerator type getter returns it.
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'PREEMPTED',
            'health': 'HEALTHY',
            'acceleratorType': 'v3-8',
            'tensorflowVersion': 'nightly',
        }
    }
    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))
    self.assertEqual('v3-8', c.accelerator_type())

  def testHandlesByteStrings(self):
    # [FUNCTION DOC]
    # Builds one client from text arguments and one from byte string arguments; asserts both
    # produce the same full resource name.
    self.assertEqual(
        client.Client(
            tpu='tpu_name', zone='zone', project='project')._full_name(),
        client.Client(
            tpu=b'tpu_name', zone=b'zone', project=b'project')._full_name(),
    )

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testWaitForHealthy(self):
    # [FUNCTION DOC]
    # Patches time and sleep with the fake clock. The fake node's health is a time series that
    # becomes healthy at time 60. Waiting with timeout 80 and interval 5 must not raise; asserts
    # the final time is 60 and sleep was called twelve times.
    time_mock = mock.patch.object(time, 'time', autospec=True).start()
    time_mock.side_effect = self._mock_time
    sleep_mock = mock.patch.object(time, 'sleep', autospec=True).start()
    sleep_mock.side_effect = self._mock_sleep

    health_timeseries = (['UNHEALTHY_MAINTENANCE']*30 + ['TIMEOUT']*10
                         + [None]*20 + ['HEALTHY']*30)
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': health_timeseries,
        },
    }

    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))

    # Doesn't throw RuntimeError as TPU becomes HEALTHY before timeout
    timeout = 80
    interval = 5
    return_time = 60
    c.wait_for_healthy(timeout_s=timeout, interval=interval)
    self.assertEqual(time.time(), return_time)
    self.assertEqual(sleep_mock.call_count, return_time/interval)

  @mock.patch.object(client, '_request_compute_metadata',
                     mock_request_compute_metadata)
  def testWaitForHealthyRaisesError(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment inside the test says it does not raise, but the test asserts it does.
    # Patches time and sleep with the fake clock; the health series never becomes healthy within
    # the timeout. Asserts waiting with timeout 80 and interval 5 raises a runtime error about
    # timing out.
    time_mock = mock.patch.object(time, 'time', autospec=True).start()
    time_mock.side_effect = self._mock_time
    sleep_mock = mock.patch.object(time, 'sleep', autospec=True).start()
    sleep_mock.side_effect = self._mock_sleep

    # Mock timeseries where takes longer than timeout.
    health_timeseries = ['UNHEALTHY_MAINTENANCE']*50 + ['TIMEOUT']*50
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'ipAddress': '10.1.2.3',
            'port': '8470',
            'state': 'READY',
            'health': health_timeseries,
        },
    }

    c = client.Client(
        tpu='tpu_name', service=self.mock_service_client(tpu_map=tpu_map))

    # Doesn't throw RuntimeError as TPU becomes HEALTHY before timeout
    with self.assertRaisesRegex(
        RuntimeError,
        'Timed out waiting for TPU .* to become healthy'):
      c.wait_for_healthy(timeout_s=80, interval=5)

  def baseConfigureTpuVersion(self):
    # [FUNCTION DOC]
    # HELPER: builds the client for the version configuration tests. Returns a client backed by
    # a fake ready node with two network endpoints (1.2.3.4 and 5.6.7.8).
    tpu_map = {
        'projects/test-project/locations/us-central1-c/nodes/tpu_name': {
            'state':
                'READY',
            'networkEndpoints': [
                {
                    'ipAddress': '1.2.3.4'
                },
                {
                    'ipAddress': '5.6.7.8'
                },
            ]
        }
    }
    return client.Client(
        tpu='tpu_name',
        project='test-project',
        zone='us-central1-c',
        service=self.mock_service_client(tpu_map=tpu_map))

  @mock.patch.object(urllib.request, 'urlopen')
  def testConfigureTpuVersion(self, urlopen):
    # [FUNCTION DOC]
    # With URL opening mocked, configures version 1.15 with the default restart type; asserts a
    # request was made to each worker's version switcher with restart type always.
    c = self.baseConfigureTpuVersion()
    c.configure_tpu_version('1.15')
    paths = [call[0][0].full_url for call in urlopen.call_args_list]
    self.assertCountEqual([
        'http://1.2.3.4:8475/requestversion/1.15?restartType=always',
        'http://5.6.7.8:8475/requestversion/1.15?restartType=always'
    ], sorted(paths))

  @mock.patch.object(urllib.request, 'urlopen')
  def testConfigureTpuVersionRestartIfneeded(self, urlopen):
    # [FUNCTION DOC]
    # Same as the default version configuration test but with the restart type set to only if
    # needed; asserts both worker requests carry that restart type.
    c = self.baseConfigureTpuVersion()
    c.configure_tpu_version('1.15', restart_type='ifNeeded')
    paths = [call[0][0].full_url for call in urlopen.call_args_list]
    self.assertCountEqual([
        'http://1.2.3.4:8475/requestversion/1.15?restartType=ifNeeded',
        'http://5.6.7.8:8475/requestversion/1.15?restartType=ifNeeded'
    ], sorted(paths))

  @mock.patch.object(urllib.request, 'urlopen')
  def testGetTpuVersion(self, urlopen):
    # [FUNCTION DOC]
    # Client for a direct grpc address with URL opening mocked to return an empty JSON object and
    # then one with a current version. Asserts the first call returns nothing, the second returns
    # the version, and both requests went to the version switcher URL of that host.
    c = client.Client(
        tpu='grpc://1.2.3.4:8470')
    resp = mock.Mock()
    resp.read.side_effect = ['{}', '{"currentVersion": "someVersion"}']
    urlopen.return_value = resp
    self.assertIsNone(c.runtime_version(), 'Missing key should be handled.')
    self.assertEqual(
        'someVersion', c.runtime_version(), 'Should return configured version.')
    paths = [call[0][0].full_url for call in urlopen.call_args_list]
    self.assertCountEqual([
        'http://1.2.3.4:8475/requestversion',
        'http://1.2.3.4:8475/requestversion',
    ], sorted(paths))


if __name__ == '__main__':
  test.main()
