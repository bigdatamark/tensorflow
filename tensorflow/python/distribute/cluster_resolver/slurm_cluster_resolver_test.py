# [MODULE DOC]
# Unit tests for the Slurm cluster resolver and its host-list and tasks-per-node parsing
# functions. Resolver tests temporarily patch the process environment with Slurm variables
# (restored after each test) and check cluster specifications, master addresses, accelerator
# counts and the visible CUDA devices variable.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SlurmClusterResolver."""

import os

from tensorflow.python.distribute.cluster_resolver.slurm_cluster_resolver import expand_hostlist
from tensorflow.python.distribute.cluster_resolver.slurm_cluster_resolver import expand_tasks_per_node
from tensorflow.python.distribute.cluster_resolver.slurm_cluster_resolver import SlurmClusterResolver
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

mock = test.mock


class SlurmClusterResolverTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the Slurm resolver and its parsing helpers. Resolver tests use a decorator that
  # patches the environment variables for the duration of each test.

  def test_expand_hostlist(self):
    # [FUNCTION DOC]
    # Checks host list expansion for a single name, a bracketed list, a range, mixed groups with
    # several bracket expressions, and zero-padded ranges (padding preserved).
    self.assertEqual(expand_hostlist('n1'), ['n1'])
    self.assertEqual(expand_hostlist('n[1,3]'), ['n1', 'n3'])
    self.assertEqual(expand_hostlist('n[1-3]'), ['n1', 'n2', 'n3'])
    self.assertEqual(
        expand_hostlist('n[1-2],m5,o[3-4,6,7-9]'),
        ['n1', 'n2', 'm5', 'o3', 'o4', 'o6', 'o7', 'o8', 'o9'])
    self.assertEqual(
        expand_hostlist('n[0001-0003],m5,o[009-011]'),
        ['n0001', 'n0002', 'n0003', 'm5', 'o009', 'o010', 'o011'])

  def test_expand_tasks_per_node(self):
    # [FUNCTION DOC]
    # Checks tasks-per-node expansion for a single count, a list, repetition markers, multi-digit
    # counts with repetition, and a long repetition (ten nodes with thirteen tasks each).
    self.assertEqual(expand_tasks_per_node('2'), [2])
    self.assertEqual(expand_tasks_per_node('2,1,3'), [2, 1, 3])
    self.assertEqual(expand_tasks_per_node('3(x2),2,1'), [3, 3, 2, 1])
    self.assertEqual(
        expand_tasks_per_node('3(x2),2,11(x4)'), [3, 3, 2, 11, 11, 11, 11])
    self.assertEqual(
        expand_tasks_per_node('13(x10)'),
        [13, 13, 13, 13, 13, 13, 13, 13, 13, 13])

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

  @mock.patch.dict(
      os.environ, {
          'SLURM_PROCID': '0',
          'SLURM_STEP_NUM_TASKS': '3',
          'SLURM_STEP_TASKS_PER_NODE': '1(x3)',
          'SLURM_STEP_NODELIST': 't02n13,t02n41,t02n43',
          'CUDA_VISIBLE_DEVICES': '0',
      })
  def testSimpleRetrievalFromEnv(self):
    # [FUNCTION DOC]
    # Scenario: rank zero of three tasks on three nodes, one task each, one visible GPU; all
    # values come from the patched environment and constructor defaults.
    # Expected: a worker job with three tasks on port 8888; master for worker zero with grpc is
    # the first node; one GPU per task; the visible devices variable remains zero.
    slurm_cluster_resolver = SlurmClusterResolver()

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'worker' tasks { key: 0 value: 't02n13:8888' }
                         tasks { key: 1 value: 't02n41:8888' }
                         tasks { key: 2 value: 't02n43:8888' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    self.assertEqual(
        slurm_cluster_resolver.master('worker', 0, rpc_layer='grpc'),
        'grpc://t02n13:8888')
    self.assertEqual(slurm_cluster_resolver.num_accelerators(), {'GPU': 1})
    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')

  @mock.patch.dict(
      os.environ, {
          'SLURM_PROCID': '0',
          'SLURM_STEP_NUM_TASKS': '3',
          'SLURM_STEP_NODELIST': 't02n13,t02n41,t02n43',
      })
  def testSimpleSuccessfulRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: jobs ps (one task) and worker (two tasks), one task per node given explicitly, one
    # GPU per node and task, no automatic GPU setting.
    # Expected: ps on the first node and workers on the other two, all on port 8888.
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 2
        },
        port_base=8888,
        tasks_per_node=1,
        gpus_per_node=1,
        gpus_per_task=1,
        auto_set_gpu=False)

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { value: 't02n13:8888' } }
    job { name: 'worker' tasks { key: 0 value: 't02n41:8888' }
                         tasks { key: 1 value: 't02n43:8888' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  @mock.patch.dict(
      os.environ, {
          'SLURM_PROCID': '0',
          'SLURM_STEP_NUM_TASKS': '3',
          'SLURM_STEP_NODELIST': 't02n13,t02n41,t02n43',
      })
  def testSimpleMasterRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: same layout as the simple retrieval test; task type and index set to worker one.
    # Expected: master with no arguments is the third node with the default grpc scheme; after
    # setting RPC layer ab, master for ps zero uses that scheme; an explicit RPC layer argument
    # overrides the stored one.
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 2
        },
        port_base=8888,
        tasks_per_node=1,
        gpus_per_node=1,
        gpus_per_task=1,
        auto_set_gpu=False)

    slurm_cluster_resolver.task_type = 'worker'
    slurm_cluster_resolver.task_id = 1
    self.assertEqual(slurm_cluster_resolver.master(), 'grpc://t02n43:8888')

    slurm_cluster_resolver.rpc_layer = 'ab'
    self.assertEqual(slurm_cluster_resolver.master('ps', 0), 'ab://t02n13:8888')
    self.assertEqual(
        slurm_cluster_resolver.master('ps', 0, rpc_layer='test'),
        'test://t02n13:8888')

  @mock.patch.dict(
      os.environ, {
          'SLURM_PROCID': '0',
          'SLURM_STEP_NUM_TASKS': '3',
          'SLURM_STEP_TASKS_PER_NODE': '1(x3)',
          'SLURM_STEP_NODELIST': 't02n13,t02n41,t02n43',
      })
  def testTaskPerNodeNotSetRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: tasks per node is not given, so it is read from the environment (one task on each
    # of three nodes).
    # Expected: the same cluster specification as the explicit one-task-per-node case.
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 2
        },
        port_base=8888,
        gpus_per_node=1,
        gpus_per_task=1,
        auto_set_gpu=False)

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { value: 't02n13:8888' } }
    job { name: 'worker' tasks { key: 0 value: 't02n41:8888' }
                         tasks { key: 1 value: 't02n43:8888' } }
    """
    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

  @mock.patch.dict(
      os.environ, {
          'SLURM_PROCID': '1',
          'SLURM_STEP_NUM_TASKS': '5',
          'SLURM_STEP_TASKS_PER_NODE': '2(x2),1',
          'SLURM_STEP_NODELIST': 't02n13,t02n41,t02n43',
          'CUDA_VISIBLE_DEVICES': '',
      })
  def testMultiTaskPerNodeRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: rank one of five tasks over nodes hosting two, two and one tasks; two GPUs per
    # node, one per task; automatic GPU setting on.
    # Expected: ps on the first node port 8888; workers on first node 8889, second node 8888 and
    # 8889, third node 8888; the visible devices variable becomes one (second GPU block).
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 4
        },
        port_base=8888,
        gpus_per_node=2,
        gpus_per_task=1,
        auto_set_gpu=True)

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { value: 't02n13:8888' } }
    job { name: 'worker' tasks { key: 0 value: 't02n13:8889' }
                         tasks { key: 1 value: 't02n41:8888' }
                         tasks { key: 2 value: 't02n41:8889' }
                         tasks { key: 3 value: 't02n43:8888' } }
    """

    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'

  @mock.patch.dict(
      os.environ, {
          'SLURM_PROCID': '1',
          'SLURM_STEP_NUM_TASKS': '5',
          'SLURM_STEP_TASKS_PER_NODE': '2(x2),1',
          'SLURM_STEP_NODELIST': 't02n13,t02n41,t02n43',
          'CUDA_VISIBLE_DEVICES': '',
      })
  def testMultipleGpusPerTaskRetrieval(self):
    # [FUNCTION DOC]
    # Scenario: same layout as the multi-task test but four GPUs per node and two per task, rank
    # one, automatic GPU setting on.
    # Expected: same addresses; the visible devices variable becomes two and three.
    slurm_cluster_resolver = SlurmClusterResolver(
        jobs={
            'ps': 1,
            'worker': 4
        },
        port_base=8888,
        gpus_per_node=4,
        gpus_per_task=2,
        auto_set_gpu=True)

    actual_cluster_spec = slurm_cluster_resolver.cluster_spec()
    expected_proto = """
    job { name: 'ps' tasks { value: 't02n13:8888' } }
    job { name: 'worker' tasks { key: 0 value: 't02n13:8889' }
                         tasks { key: 1 value: 't02n41:8888' }
                         tasks { key: 2 value: 't02n41:8889' }
                         tasks { key: 3 value: 't02n43:8888' } }
    """

    self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'


if __name__ == '__main__':
  test.main()
