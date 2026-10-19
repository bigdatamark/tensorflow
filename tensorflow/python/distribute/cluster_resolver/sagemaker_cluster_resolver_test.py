# [MODULE DOC]
# Unit tests for the SageMaker cluster resolver. Each test sets the SageMaker host-list and
# current-host environment variables directly in the process environment (never restoring them)
# and checks the resulting cluster specification, master address and property overrides.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SageMakerClusterResolver."""

import os

from tensorflow.python.distribute.cluster_resolver.sagemaker_cluster_resolver import SageMakerClusterResolver
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


@test_util.run_all_in_graph_and_eager_modes
class SageMakerClusterResolverTest(test.TestCase):
  # [CLASS DOC]
  # Tests the SageMaker resolver, in graph and eager modes. Environment variables are set per
  # test and leak to later tests.

  def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
    # [FUNCTION DOC]
    # HELPER: shared assertion. Checks the cluster definition against the expected text proto,
    # also after rebuilding a cluster specification from the specification, its cluster
    # definition and its dictionary form. Returns nothing.
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

  def testNormalClusterSpecRead(self):
    # [FUNCTION DOC]
    # Scenario: two hosts, current host is the second.
    # Expected: a worker job with both hosts, sorted, using the default port 2223.
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-2'

    cluster_resolver = SageMakerClusterResolver()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: 'algo-1:2223' }
                         tasks { key: 1 value: 'algo-2:2223' } }
    """
    actual_cluster_spec = cluster_resolver.cluster_spec()
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  def testAutomaticMasterRead(self):
    # [FUNCTION DOC]
    # Scenario: two hosts, current host is the first; no overrides.
    # Expected: master is the first host address without a scheme prefix.
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-1'

    cluster_resolver = SageMakerClusterResolver()
    self.assertEqual('algo-1:2223', cluster_resolver.master())

  def testSpecifiedTaskTypeAndIndexMasterRead(self):
    # [FUNCTION DOC]
    # Scenario: two hosts, current host is the second.
    # Expected: master for worker task one is the second host address.
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-2'

    cluster_resolver = SageMakerClusterResolver()
    self.assertEqual('algo-2:2223', cluster_resolver.master('worker', 1))

  def testRpcLayerRead(self):
    # [FUNCTION DOC]
    # Scenario: two hosts, current host is the first; resolver built with RPC layer grpc.
    # Expected: master is the first host address prefixed with the grpc scheme.
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-1'

    cluster_resolver = SageMakerClusterResolver(rpc_layer='grpc')
    self.assertEqual('grpc://algo-1:2223', cluster_resolver.master())

  def testParameterOverrides(self):
    # [FUNCTION DOC]
    # Scenario: resolver built with task type worker and index zero, then task index and RPC layer
    # reassigned.
    # Expected: master and property getters reflect the initial values, then the new ones (second
    # host address with the test scheme, index one, RPC layer test).
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-1'

    cluster_resolver = SageMakerClusterResolver(task_type='worker', task_id=0)

    self.assertEqual('algo-1:2223', cluster_resolver.master())
    self.assertEqual('worker', cluster_resolver.task_type)
    self.assertEqual(0, cluster_resolver.task_id)

    cluster_resolver.task_type = 'worker'
    cluster_resolver.task_id = 1
    cluster_resolver.rpc_layer = 'test'

    self.assertEqual('test://algo-2:2223', cluster_resolver.master())
    self.assertEqual('worker', cluster_resolver.task_type)
    self.assertEqual(1, cluster_resolver.task_id)
    self.assertEqual('test', cluster_resolver.rpc_layer)

  def testTaskIndexOverride(self):
    # [FUNCTION DOC]
    # Scenario: current host is the second of two, resolver built with task index one.
    # Expected: the task index property returns one.
    os.environ['SM_HOSTS'] = '["algo-1","algo-2"]'
    os.environ['SM_CURRENT_HOST'] = 'algo-2'

    cluster_resolver = SageMakerClusterResolver(task_id=1)
    self.assertEqual(1, cluster_resolver.task_id)

  def testZeroItemsInClusterSpecMasterRead(self):
    # [FUNCTION DOC]
    # Scenario: both environment variables are empty text.
    # Expected: master is an empty string.
    os.environ['SM_HOSTS'] = ''
    os.environ['SM_CURRENT_HOST'] = ''

    cluster_resolver = SageMakerClusterResolver()
    self.assertEqual('', cluster_resolver.master())

  def testOneItemInClusterSpecMasterRead(self):
    # [FUNCTION DOC]
    # Scenario: host list has one host but the current host is empty text (not in the list).
    # Expected: master is an empty string (the configuration is empty, so no cluster is found).
    os.environ['SM_HOSTS'] = '["algo-1"]'
    os.environ['SM_CURRENT_HOST'] = ''

    cluster_resolver = SageMakerClusterResolver()
    self.assertEqual('', cluster_resolver.master())


if __name__ == '__main__':
  test.main()
