# [MODULE DOC]
# Tests and benchmarks for creating in-process RPC clusters on localhost with the test platform
# helper that builds a local cluster of worker and parameter server servers. Contains one test
# case (variables placed on parameter servers are visible across worker sessions), a benchmark
# timing cluster creation with one worker and ten parameter servers, and a benchmark reading a
# partitioned variable spread over a hundred parameter servers. Running the file as a main program
# starts the standard test runner. No import-time side effects beyond imports.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for creating RPC clusters on localhost."""

import time

import numpy as np

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter


class CreateLocalClusterTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering creation of a local multi-server cluster. It has no shared fixtures; the
  # single test builds its own cluster.

  @test_util.run_v1_only("b/120545219")
  def testCreateLocalCluster(self):
    # [FUNCTION DOC]
    # Test (graph mode only, skipped under TF2 behavior). Scenario: create a local cluster with
    # two workers and two parameter servers and open a session on each worker. Variables are
    # created pinned to each parameter server task (zero and one by worker session zero, two and
    # three by worker session one) and initialized through the worker sessions. Asserts that each
    # variable evaluated in the opposite worker's session returns the initial value it was given
    # (0.0, 1.0, 2.0, 3.0), proving state lives on the shared parameter servers.
    workers, _ = test.create_local_cluster(num_workers=2, num_ps=2)
    worker_sessions = [session_lib.Session(w.target) for w in workers]
    with ops.device("/job:ps/task:0"):
      var0 = variables.Variable(0.0)
    with ops.device("/job:ps/task:1"):
      var1 = variables.Variable(1.0)
    worker_sessions[0].run([var0.initializer, var1.initializer])
    with ops.device("/job:ps/task:0"):
      var2 = variables.Variable(2.0)
    with ops.device("/job:ps/task:1"):
      var3 = variables.Variable(3.0)
    worker_sessions[1].run([var2.initializer, var3.initializer])

    # Read values back in the opposite session
    self.assertAllEqual(0.0, var0.eval(session=worker_sessions[1]))
    self.assertAllEqual(1.0, var1.eval(session=worker_sessions[1]))
    self.assertAllEqual(2.0, var2.eval(session=worker_sessions[0]))
    self.assertAllEqual(3.0, var3.eval(session=worker_sessions[0]))


class CreateLocalClusterBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class measuring how long it takes to start a local cluster.

  def benchmarkCreateLocalCluster(self):
    # [FUNCTION DOC]
    # Benchmark. Creates a local cluster with one worker and ten parameter servers five times,
    # recording wall time of each creation, then prints and reports the median wall time and
    # iteration count under the name for one worker and ten parameter servers. Returns nothing;
    # side effects are starting servers, printing to standard output and reporting the benchmark.
    deltas = []
    iters = 5
    for _ in range(iters):
      start_time = time.time()
      test.create_local_cluster(num_workers=1, num_ps=10)
      end_time = time.time()
      deltas.append(end_time - start_time)

    median_deltas = np.median(deltas)
    print("\n\nbenchmark_create_local_cluster_1_worker_10_ps.  "
          "iterations: %d, median wall time: %g\n\n" % (iters, median_deltas))
    self.report_benchmark(
        iters=iters,
        wall_time=median_deltas,
        name="benchmark_create_local_cluster_1_worker_10_ps")


class PartitionedVariablesBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class measuring reads of partitioned variables whose shards live on many servers.

  def benchmark_create_1000_partitions_with_100_parameter_servers(self):
    # [FUNCTION DOC]
    # Benchmark. Starts a local cluster of one worker and a hundred parameter servers. For each
    # of four partition sizes (1, 512, 32768 and 131072 floats) it creates a float32 partitioned
    # variable of one thousand partitions, placed round-robin by the replica device setter and
    # sized by a maximum-shard-bytes partitioner, and keeps its concatenated tensor. After
    # initializing all global variables in the worker session it runs an op benchmark that reads
    # and concatenates each variable, named by partition size. Returns nothing; prints progress
    # and reports benchmark results.
    workers, _ = test.create_local_cluster(num_workers=1, num_ps=100)
    worker_sessions = [session_lib.Session(w.target) for w in workers]
    worker = worker_sessions[0]
    partition_sizes = (1, 512, 1024 * 32, 1024 * 128)

    partitioned = []

    for partition_size in partition_sizes:
      # max_shard_bytes is 4, shape is 1000*partition_size float32s which should
      # partition into 1000 shards, each containing partition_size float32s.
      print("Building partitioned variable with %d floats per partition" %
            partition_size)
      with ops.device(device_setter.replica_device_setter(ps_tasks=100)):
        partitioned_ix = variable_scope.get_variable(
            "partitioned_%d" % partition_size,
            shape=[1000 * partition_size],
            dtype=dtypes.float32,
            # Each partition to have exactly N float32s
            partitioner=partitioned_variables.variable_axis_size_partitioner(
                max_shard_bytes=4 * partition_size))
        # Concatenates along axis 0
        partitioned.append(ops.convert_to_tensor(partitioned_ix))

    variables.global_variables_initializer().run(session=worker)

    for ix, partition_size in enumerate(partition_sizes):
      print("Running benchmark having partitions with %d floats" %
            partition_size)
      self.run_op_benchmark(
          worker,
          partitioned[ix],
          name=("read_concat_1000_partitions_from_"
                "100_parameter_servers_partsize_%d_floats" % partition_size))


if __name__ == "__main__":
  test.main()
