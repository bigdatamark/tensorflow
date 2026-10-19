# [MODULE DOC]
# Unit tests for the GCE cluster resolver. The GCE API client is replaced by nested mock objects
# (instance groups and instances) so no network access is needed. Tests cover cluster
# specification retrieval, master address computation, parameter overriding and merging with the
# union resolver.
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
"""Tests for GCEClusterResolver."""

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import UnionClusterResolver
from tensorflow.python.distribute.cluster_resolver.gce_cluster_resolver import GCEClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


mock = test.mock


class GCEClusterResolverTest(test.TestCase):
  # [CLASS DOC]
  # Tests the GCE cluster resolver against a fake GCE service client built from mocks. Provides
  # helper methods that build the fake instance-group listing, instance details and a combined
  # service client from simple name-to-address data.

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests in this class.
    # Checks that the cluster definition matches the expected text proto, also after rebuilding a
    # cluster specification from the specification, its cluster definition and its dictionary
    # form. Returns nothing.
    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

  def standard_mock_instance_groups(self, instance_map=None):
    # [FUNCTION DOC]
    # HELPER: builds the fake instance-groups part of the GCE client.
    # Input: optional list of instance entries (each with an instance URL); defaults to a single
    # instance named gce-instance-1. Returns a mock whose list-instances call yields a request
    # whose execution returns those items, and whose next-page call returns nothing (one page).
    if instance_map is None:
      instance_map = [
          {'instance': 'https://gce.example.com/res/gce-instance-1'}
      ]

    mock_instance_group_request = mock.MagicMock()
    mock_instance_group_request.execute.return_value = {
        'items': instance_map
    }

    service_attrs = {
        'listInstances.return_value': mock_instance_group_request,
        'listInstances_next.return_value': None,
    }
    mock_instance_groups = mock.Mock(**service_attrs)
    return mock_instance_groups

  def standard_mock_instances(self, instance_to_ip_map=None):
    # [FUNCTION DOC]
    # HELPER: builds the fake instances part of the GCE client.
    # Input: optional mapping from instance name to IP address; defaults to one instance with
    # address 10.123.45.67. Returns a mock whose get call returns a request that executes to a
    # network-interface record with the mapped address, and raises a runtime error for an unknown
    # instance name. (The initially built default request object is unused.)
    if instance_to_ip_map is None:
      instance_to_ip_map = {
          'gce-instance-1': '10.123.45.67'
      }

    mock_get_request = mock.MagicMock()
    mock_get_request.execute.return_value = {
        'networkInterfaces': [
            {'networkIP': '10.123.45.67'}
        ]
    }

    def get_side_effect(project, zone, instance):
      # [FUNCTION DOC]
      # HELPER: fake implementation of the instance get call. Ignores project and zone; for a
      # known instance name returns a mock request whose execution gives its network address,
      # otherwise raises a runtime error stating the instance was not found.
      del project, zone  # Unused

      if instance in instance_to_ip_map:
        mock_get_request = mock.MagicMock()
        mock_get_request.execute.return_value = {
            'networkInterfaces': [
                {'networkIP': instance_to_ip_map[instance]}
            ]
        }
        return mock_get_request
      else:
        raise RuntimeError('Instance %s not found!' % instance)

    service_attrs = {
        'get.side_effect': get_side_effect,
    }
    mock_instances = mock.MagicMock(**service_attrs)
    return mock_instances

  def standard_mock_service_client(
      self,
      mock_instance_groups=None,
      mock_instances=None):
    # [FUNCTION DOC]
    # HELPER: assembles a fake GCE service client from optional instance-group and instance mocks
    # (standard defaults are built when omitted). Returns a mock client whose instance-groups and
    # instances accessors return those mocks.

    if mock_instance_groups is None:
      mock_instance_groups = self.standard_mock_instance_groups()
    if mock_instances is None:
      mock_instances = self.standard_mock_instances()

    mock_client = mock.MagicMock()
    mock_client.instanceGroups.return_value = mock_instance_groups
    mock_client.instances.return_value = mock_instances
    return mock_client

  def gen_standard_mock_service_client(self, instances=None):
    # [FUNCTION DOC]
    # HELPER: builds a fake GCE service client from a list of records each holding an instance
    # name and an IP address. Despite the default of null, the list is required because it is
    # iterated unconditionally. Creates the name-to-address mapping and instance list, builds
    # the instance and instance-group mocks from them and returns the combined mock client.
    name_to_ip = {}
    instance_list = []
    for instance in instances:
      name_to_ip[instance['name']] = instance['ip']
      instance_list.append({
          'instance': 'https://gce.example.com/gce/res/' + instance['name']
      })

    mock_instance = self.standard_mock_instances(name_to_ip)
    mock_instance_group = self.standard_mock_instance_groups(instance_list)

    return self.standard_mock_service_client(mock_instance_group, mock_instance)

  def testSimpleSuccessfulRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: resolver over the default fake client with one instance.
    # Expected: cluster specification has a worker job with one task at that address and port 8470.
    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        port=8470,
        credentials=None,
        service=self.standard_mock_service_client())

    actual_cluster_spec = gce_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: '10.123.45.67:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testMasterRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: resolver with task index zero over the default fake client.
    # Expected: master with no arguments returns the instance address with the default grpc scheme.
    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_id=0,
        port=8470,
        credentials=None,
        service=self.standard_mock_service_client())
    self.assertEqual(gce_cluster_resolver.master(), 'grpc://10.123.45.67:8470')

  def testMasterRetrievalWithCustomTasks(self):
    # [FUNCTION DOC]
    # Scenario: three fake instances; master requested for worker task two with RPC layer test.
    # Expected: the third instance address prefixed with the test scheme.
    name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    self.assertEqual(
        gce_cluster_resolver.master('worker', 2, 'test'),
        'test://10.3.4.5:8470')

  def testOverrideParameters(self):
    # [FUNCTION DOC]
    # Scenario: resolver with custom task type over three instances; task index and RPC layer are
    # reassigned through the setters.
    # Expected: getters return the new values (task type unchanged) and master returns the second
    # instance address with the test scheme.
    name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='testworker',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    gce_cluster_resolver.task_id = 1
    gce_cluster_resolver.rpc_layer = 'test'

    self.assertEqual(gce_cluster_resolver.task_type, 'testworker')
    self.assertEqual(gce_cluster_resolver.task_id, 1)
    self.assertEqual(gce_cluster_resolver.rpc_layer, 'test')
    self.assertEqual(gce_cluster_resolver.master(), 'test://10.2.3.4:8470')

  def testOverrideParametersWithZeroOrEmpty(self):
    # [FUNCTION DOC]
    # Scenario: resolver configured with an empty task type and task index one.
    # Action: ask for master with an explicit empty task type and index zero.
    # Expected: the explicit falsy values are honored (not replaced by stored ones), giving the
    # first instance address with the grpc scheme.
    name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='',
        task_id=1,
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    self.assertEqual(gce_cluster_resolver.master(
        task_type='', task_id=0), 'grpc://10.1.2.3:8470')

  def testCustomJobNameAndPortRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: resolver with job name custom and port 2222 over the default fake client.
    # Expected: cluster specification has a job named custom with the address using port 2222.
    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='custom',
        port=2222,
        credentials=None,
        service=self.standard_mock_service_client())

    actual_cluster_spec = gce_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'custom' tasks { key: 0 value: '10.123.45.67:2222' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testMultipleInstancesRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: three fake instances with distinct addresses.
    # Expected: worker job with three tasks, numbered in sorted address order.
    name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    actual_cluster_spec = gce_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: '10.1.2.3:8470' }
                         tasks { key: 1 value: '10.2.3.4:8470' }
                         tasks { key: 2 value: '10.3.4.5:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testUnionMultipleInstanceRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: two worker-type resolvers (three instances each, port 8470) and a ps resolver
    # (two instances, port 2222) combined in a union resolver.
    # Expected: ps job with two tasks and a worker job concatenating both worker resolvers (six
    # tasks in resolver order).
    worker1_name_to_ip = [
        {'name': 'instance1', 'ip': '10.1.2.3'},
        {'name': 'instance2', 'ip': '10.2.3.4'},
        {'name': 'instance3', 'ip': '10.3.4.5'},
    ]

    worker2_name_to_ip = [
        {'name': 'instance4', 'ip': '10.4.5.6'},
        {'name': 'instance5', 'ip': '10.5.6.7'},
        {'name': 'instance6', 'ip': '10.6.7.8'},
    ]

    ps_name_to_ip = [
        {'name': 'ps1', 'ip': '10.100.1.2'},
        {'name': 'ps2', 'ip': '10.100.2.3'},
    ]

    worker1_gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='worker',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(worker1_name_to_ip))

    worker2_gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='worker',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(worker2_name_to_ip))

    ps_gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='ps',
        port=2222,
        credentials=None,
        service=self.gen_standard_mock_service_client(ps_name_to_ip))

    union_cluster_resolver = UnionClusterResolver(worker1_gce_cluster_resolver,
                                                  worker2_gce_cluster_resolver,
                                                  ps_gce_cluster_resolver)

    actual_cluster_spec = union_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: '10.100.1.2:2222' }
                     tasks { key: 1 value: '10.100.2.3:2222' } }
    job { name: 'worker' tasks { key: 0 value: '10.1.2.3:8470' }
                         tasks { key: 1 value: '10.2.3.4:8470' }
                         tasks { key: 2 value: '10.3.4.5:8470' }
                         tasks { key: 3 value: '10.4.5.6:8470' }
                         tasks { key: 4 value: '10.5.6.7:8470' }
                         tasks { key: 5 value: '10.6.7.8:8470' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testSettingTaskTypeRaiseError(self):
    # [FUNCTION DOC]
    # Scenario: resolver over three fake instances.
    # Action: try to assign a new task type.
    # Expected: a runtime error whose message says the task type cannot be reset after creation.
    name_to_ip = [
        {
            'name': 'instance1',
            'ip': '10.1.2.3'
        },
        {
            'name': 'instance2',
            'ip': '10.2.3.4'
        },
        {
            'name': 'instance3',
            'ip': '10.3.4.5'
        },
    ]

    gce_cluster_resolver = GCEClusterResolver(
        project='test-project',
        zone='us-east1-d',
        instance_group='test-instance-group',
        task_type='testworker',
        port=8470,
        credentials=None,
        service=self.gen_standard_mock_service_client(name_to_ip))

    with self.assertRaisesRegex(
        RuntimeError, 'You cannot reset the task_type '
        'of the GCEClusterResolver after it has '
        'been created.'):
      gce_cluster_resolver.task_type = 'foobar'


if __name__ == '__main__':
  test.main()
