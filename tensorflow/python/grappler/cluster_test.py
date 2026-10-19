# [MODULE DOC]
# Unit tests for the Grappler cluster wrapper (the Python binding of the C++ cluster classes).
# Covers measuring per-op costs on a real cluster, disabling detailed statistics, peak memory
# estimation, a virtual cluster built from fake device properties, the Provision context manager,
# listing available ops and querying which devices support each node. Defines one test class and
# has no import-time side effects; running it as a script starts the TensorFlow test runner.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the swig wrapper of clusters."""

from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.grappler import cluster
from tensorflow.python.grappler import item
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ClusterTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the cluster wrapper and its Provision context manager. Each test builds a
  # small graph of random values added together, wraps it in a meta graph and a Grappler item,
  # and checks cost, memory or device-support results. No shared fixtures beyond the base class.

  def testBasic(self):
    # [FUNCTION DOC]
    # Builds a graph adding two random scalars, registers the sum as the train op, and measures
    # costs on a real cluster with detailed stats and timeline enabled. Expects a positive run
    # time, exactly four per-op performance entries and non-empty per-device step statistics.
    with ops.Graph().as_default() as g:
      a = random_ops.random_uniform(shape=())
      b = random_ops.random_uniform(shape=())
      c = a + b
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      grappler_cluster = cluster.Cluster(
          disable_detailed_stats=False, disable_timeline=False)
      op_perfs, run_time, step_stats = grappler_cluster.MeasureCosts(
          grappler_item)
      self.assertTrue(run_time > 0)
      self.assertEqual(len(op_perfs), 4)
      self.assertTrue(step_stats.dev_stats)

  def testNoDetailedStats(self):
    # [FUNCTION DOC]
    # Same graph as the basic test, but the cluster is created with detailed stats disabled.
    # Expects a positive run time while the per-op performance list and the per-device step
    # statistics are both empty.
    with ops.Graph().as_default() as g:
      a = random_ops.random_uniform(shape=())
      b = random_ops.random_uniform(shape=())
      c = a + b
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      grappler_cluster = cluster.Cluster(disable_detailed_stats=True)

      op_perfs, run_time, step_stats = grappler_cluster.MeasureCosts(
          grappler_item)
      self.assertTrue(run_time > 0)
      self.assertEqual(len(op_perfs), 0)
      self.assertEqual(len(step_stats.dev_stats), 0)

  def testMemoryEstimates(self):
    # [FUNCTION DOC]
    # Places the two random scalars and their sum on the local CPU device, then asks the cluster
    # to determine peak memory usage. Expects at least one device entry; for the CPU device the
    # peak usage must be 12 bytes and the snapshot must list five live tensors.
    with ops.Graph().as_default() as g:
      with ops.device('/job:localhost/replica:0/task:0/device:CPU:0'):
        a = random_ops.random_uniform(shape=())
        b = random_ops.random_uniform(shape=())
        c = a + b
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(c)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        grappler_cluster = cluster.Cluster(
            disable_detailed_stats=True, disable_timeline=True)
        peak_mem = grappler_cluster.DeterminePeakMemoryUsage(grappler_item)
        self.assertLessEqual(1, len(peak_mem))
        snapshot = peak_mem['/job:localhost/replica:0/task:0/device:CPU:0']
        peak_usage = snapshot[0]
        self.assertEqual(12, peak_usage)
        live_tensors = snapshot[1]
        self.assertEqual(5, len(live_tensors))

  def testVirtualCluster(self):
    # [FUNCTION DOC]
    # Adds two 1024 by 1024 random matrices on a GPU device name, and builds a virtual cluster
    # from a fake GPU device description (frequency 1000, 60 cores, architecture 7). Measures
    # costs and expects a simulated run time of exactly 0.000209 and five op performance entries,
    # then expects the estimated device performance to be 7680.0. No real GPU is required.
    with ops.Graph().as_default() as g:
      with ops.device('/device:GPU:0'):
        a = random_ops.random_uniform(shape=[1024, 1024])
        b = random_ops.random_uniform(shape=[1024, 1024])
        c = a + b
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      device_properties = device_properties_pb2.DeviceProperties(
          type='GPU',
          frequency=1000,
          num_cores=60,
          environment={'architecture': '7'})
      named_device = device_properties_pb2.NamedDevice(
          properties=device_properties, name='/device:GPU:0')
      grappler_cluster = cluster.Cluster(
          disable_detailed_stats=False,
          disable_timeline=False,
          devices=[named_device])
      op_perfs, run_time, _ = grappler_cluster.MeasureCosts(grappler_item)
      self.assertEqual(run_time, 0.000209)
      self.assertEqual(len(op_perfs), 5)

      estimated_perf = grappler_cluster.EstimatePerformance(named_device)
      self.assertEqual(7680.0, estimated_perf)

  def testContext(self):
    # [FUNCTION DOC]
    # Builds the scalar-add graph and item outside the cluster, then measures costs inside a
    # Provision context manager with detailed stats and timeline enabled. Expects positive run
    # time, four op performance entries and non-empty device statistics.
    with ops.Graph().as_default() as g:
      a = random_ops.random_uniform(shape=())
      b = random_ops.random_uniform(shape=())
      c = a + b
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)

    with cluster.Provision(
        disable_detailed_stats=False, disable_timeline=False) as gcluster:
      op_perfs, run_time, step_stats = gcluster.MeasureCosts(grappler_item)
      self.assertTrue(run_time > 0)
      self.assertEqual(len(op_perfs), 4)
      self.assertTrue(step_stats.dev_stats)

  def testAvailableOps(self):
    # [FUNCTION DOC]
    # Provisions a default cluster and lists the available op names. Expects the list to include
    # the add and matrix multiply ops and to be sorted alphabetically.
    with cluster.Provision() as gcluster:
      op_names = gcluster.ListAvailableOps()
      self.assertTrue('Add' in op_names)
      self.assertTrue('MatMul' in op_names)
      self.assertEqual(op_names, sorted(op_names))

  def testSupportDevices(self):
    # [FUNCTION DOC]
    # Builds a graph with an add and a sum reduction whose axis comes from a range op. First uses
    # a virtual cluster with one fake CPU and one fake GPU and expects add, sum and range nodes to
    # be supported on both. Then uses a real cluster: with a GPU available, add and sum support
    # CPU and GPU while range is CPU only (the axis tensor must be on the host); without a GPU,
    # add is supported on the CPU only.
    with ops.Graph().as_default() as g:
      a = random_ops.random_uniform(shape=(2, 3))
      b = random_ops.random_uniform(shape=(2, 3))
      c = a + b
      dims = math_ops.range(0, array_ops.rank(c), 1)
      d = math_ops.reduce_sum(a, axis=dims)
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(d)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)

      device_properties = device_properties_pb2.DeviceProperties(
          type='GPU', frequency=1000, num_cores=60)
      named_gpu = device_properties_pb2.NamedDevice(
          properties=device_properties, name='/GPU:0')
      device_properties = device_properties_pb2.DeviceProperties(
          type='CPU', frequency=3000, num_cores=6)
      named_cpu = device_properties_pb2.NamedDevice(
          properties=device_properties, name='/CPU:0')
      virtual_cluster = cluster.Cluster(devices=[named_cpu, named_gpu])
      supported_dev = virtual_cluster.GetSupportedDevices(grappler_item)
      self.assertEqual(supported_dev['add'], ['/CPU:0', '/GPU:0'])
      self.assertEqual(supported_dev['Sum'], ['/CPU:0', '/GPU:0'])
      self.assertEqual(supported_dev['range'], ['/CPU:0', '/GPU:0'])

      real_cluster = cluster.Cluster()
      supported_dev = real_cluster.GetSupportedDevices(grappler_item)
      if test.is_gpu_available():
        self.assertEqual(supported_dev['add'], [
            '/job:localhost/replica:0/task:0/device:CPU:0',
            '/job:localhost/replica:0/task:0/device:GPU:0'
        ])
        self.assertEqual(supported_dev['Sum'], [
            '/job:localhost/replica:0/task:0/device:CPU:0',
            '/job:localhost/replica:0/task:0/device:GPU:0'
        ])
        # The axis tensor must reside on the host
        self.assertEqual(supported_dev['range'],
                         ['/job:localhost/replica:0/task:0/device:CPU:0'])
      else:
        self.assertEqual(supported_dev['add'],
                         ['/job:localhost/replica:0/task:0/device:CPU:0'])


if __name__ == '__main__':
  test.main()
