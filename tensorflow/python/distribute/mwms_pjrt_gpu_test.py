# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# GPU test for multi-worker mirrored strategy style setups on the PJRT runtime: two worker
# processes, each expected to own one GPU, enable collective ops and a standalone coordination
# service, then run a compiled (XLA) function doing two all-reduce operations. Skipped unless at
# least two GPUs and XLA are available. Also defines two small helpers that configure collective
# ops and the coordination service for a worker.

import copy

from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops


# Based on collective_ops_multi_worker_test.py
def enable_collective_ops(cluster_resolver):
  # [FUNCTION DOC]
  # HELPER: turns on collective ops in the eager context for this worker. Input: a cluster
  # resolver giving cluster, task type, task id and RPC layer (grpc if unset). Sets worker zero as
  # collective leader, builds a server definition from a copy of the context's configuration and
  # the resolver, and enables collective ops with it. Global side effect on the eager context.
  context.context().configure_collective_ops(
      collective_leader="/job:worker/replica:0/task:0")
  config_proto = copy.deepcopy(context.context().config)
  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_resolver.cluster_spec().as_cluster_def(),
      default_session_config=config_proto,
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol=cluster_resolver.rpc_layer or "grpc")
  context.context().enable_collective_ops(server_def)


def configure_coordination_service():
  # [FUNCTION DOC]
  # HELPER: configures the eager context to use a standalone coordination service whose leader is
  # worker task zero, with health checking disabled. Takes no input, returns nothing; global side
  # effect on the eager context.
  context.context().configure_coordination_service(
      service_type="standalone",
      service_leader="/job:worker/replica:0/task:0",
      enable_health_check=False,
  )


class MultiWorkerMirroredStrategyPjRtRemoteGpuTest(
    multi_worker_test_base.MultiWorkerTestBase, test.TestCase
):
  # [CLASS DOC]
  # Test case for two-worker, one-GPU-per-worker collective execution under the PJRT runtime. Uses
  # the multi-worker test base class (cluster fixtures) together with the eager test case.

  def testRemoteGpusFound(self):
    # [FUNCTION DOC]
    # Scenario: skipped unless at least two GPUs and XLA are available. Starts two worker processes
    # without sharing GPUs, each running a worker function that configures the coordination service
    # and collective ops and runs the compiled function. Asserts two results are returned and, for
    # each, the all-reduced tensors equal eight in every element of the ten-element tensors.

    def worker_fn():
      # [FUNCTION DOC]
      # Body run in each worker process: configures coordination service and collective ops from
      # the environment's cluster resolver, initializes the context, builds a constant tensor
      # filled with task id plus one, and returns the result of the compiled function that doubles
      # and adds one, then all-reduces (sum) two copies with different instance keys across a
      # group of two. Returns the pair of reduced tensors.

      configure_coordination_service()
      cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
      enable_collective_ops(cluster_resolver=cluster_resolver)
      context.context().ensure_initialized()

      group_size = 2
      group_key = 1
      instance_key1 = 1
      instance_key2 = 2
      tensor_size = 10

      # cluster_resolver.task_id is int 0 or 1.
      tensor_val = [cluster_resolver.task_id + 1.] * tensor_size
      constant = constant_op.constant(tensor_val)

      @def_function.function(jit_compile=True)
      def g():
        # [FUNCTION DOC]
        # Compiled function: applies the inner scaling function to the constant twice, wraps each
        # in an identity, and all-reduces each with the sum operation over a group of two using
        # distinct instance keys. Returns the two reduced tensors.

        def f(x):
          # [FUNCTION DOC]
          # HELPER: computes twice the input plus one.
          return 2 * x + 1

        input_tensor1 = array_ops.identity(f(constant))
        input_tensor2 = array_ops.identity(f(constant))

        reduced_tensor1 = collective_ops.all_reduce_v2(
            input_tensor1, group_size, group_key, instance_key1, "Add", "Id")
        reduced_tensor2 = collective_ops.all_reduce_v2(
            input_tensor2, group_size, group_key, instance_key2, "Add", "Id")
        return reduced_tensor1, reduced_tensor2

      return g()

    num_gpus = len(tf_config.list_physical_devices("GPU"))
    skip_flag = (num_gpus < 2) or not test_util.is_xla_enabled()
    if skip_flag:
      self.skipTest(
          "This test is intended to test the 2 GPU (1 per worker with 2"
          " workers) with XLA case (%d GPUs found, using XLA = %s)."
          % (num_gpus, test_util.is_xla_enabled())
      )

    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    mpr = multi_process_runner.MultiProcessRunner(
        worker_fn, cluster_spec, share_gpu=False
    )
    mpr.start()
    mpr_result = mpr.join()
    self.assertLen(mpr_result.return_value, 2)
    for rval in mpr_result.return_value:
      for t in rval:
        # for IDs 0 and 1: (2*(0+1)+1) + (2*(1+1)+1) = 8
        self.assertAllClose(t.numpy(), [8., 8., 8., 8., 8., 8., 8., 8., 8., 8.])


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
