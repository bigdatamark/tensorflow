# [MODULE DOC]
# Test module for multi-worker collective operations (all-reduce, broadcast) and the collective
# health-check and abort facilities of the eager context. Tests launch several worker processes
# through the multi-process runner and exercise peer health checking, peer failure detection and
# aborting of in-flight collectives at the communication, group resolution and instance
# resolution stages. Defines small helpers that enable collective ops on a worker from its
# cluster resolver, plus a module-level combination of device and communication settings and a
# shared two-worker process pool runner (created at import time). When run as a script it starts
# the multi-process test main.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for multi worker Collective Operations."""

import copy
import os
import threading
import time

from absl.testing import parameterized

from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import collective_ops


def enable_collective_ops(cluster_resolver):
  # [FUNCTION DOC]
  # HELPER: shared setup used by the multi-worker collective tests in this module.
  # Configures the eager context so that worker task 0 is the collective leader, copies the
  # context's session config, builds a server definition from the given cluster resolver (cluster
  # spec, task type, task id, and its rpc layer defaulting to the gRPC protocol) and enables
  # collective ops with it. Input: a cluster resolver describing the cluster. Returns nothing.
  # Side effects: mutates the global eager context (collective leader, server definition).
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


def enable_collective_ops_with_barrier(cluster_resolver):
  # [FUNCTION DOC]
  # HELPER: setup used by the abort tests so all workers enable collective ops in lockstep.
  # Waits on the multi-process runner barrier, enables collective ops from the given cluster
  # resolver, then waits on the barrier again so no worker proceeds before every worker has
  # (re)enabled them. Returns nothing. Side effects: resets the collective executor of the eager
  # context; must run inside a multi-process runner worker where the barrier exists.
  multi_process_runner.get_barrier().wait()
  enable_collective_ops(cluster_resolver)
  multi_process_runner.get_barrier().wait()


device_combination = (
    combinations.combine(device="CPU", communication="RING", required_gpus=0) +
    combinations.combine(
        device="GPU", communication=["RING", "NCCL"], required_gpus=1))


class CollectiveOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the collective peer health check of the eager context. Each test starts a small
  # cluster of worker processes with the multi-process runner and checks health results for
  # live, dead, restarted and invalid peers. No shared fixtures.

  def testCheckHealth(self):
    # [FUNCTION DOC]
    # Scenario: two workers are started. Each enables collective ops and repeatedly checks the
    # health of both workers, retrying on unavailable or deadline-exceeded errors until every
    # check succeeds, then waits on a barrier. Expected: the runner joins with no error.

    def worker_fn():
      # [FUNCTION DOC]
      # Worker body: enables collective ops, loops checking health of worker tasks 0 and 1
      # (one second timeout) until no unavailable/deadline error occurs, then waits on the
      # barrier so neither process exits early. Returns nothing.
      enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
      # There may be some delays before the server startup. Check health should
      # eventually be OK.
      while True:
        try:
          for task in [
              "/job:worker/replica:0/task:0",
              "/job:worker/replica:0/task:1",
          ]:
            context.context().check_collective_ops_peer_health(
                task, timeout_in_ms=1000)
        except (errors.UnavailableError, errors.DeadlineExceededError):
          continue
        break
      multi_process_runner.get_barrier().wait()

    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
    mpr.start()
    mpr.join()

  def testCheckHealthPeerDown(self):
    # [FUNCTION DOC]
    # Scenario: a two-worker cluster is defined but only worker 0 is started. Worker 0 checks
    # the health of the never-started worker 1. Expected: joining the runner raises an
    # unavailable or deadline-exceeded error.

    def worker_fn():
      # [FUNCTION DOC]
      # Worker body: enables collective ops, then checks health of worker task 1 with a one
      # second timeout; the check is expected to fail because that peer is down. Returns nothing.
      enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
      context.context().check_collective_ops_peer_health(
          "/job:worker/replica:0/task:1", timeout_in_ms=1000)

    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
    mpr.start_single_process("worker", 0)
    with self.assertRaises(
        (errors.UnavailableError, errors.DeadlineExceededError)):
      mpr.join()

  def testCheckHealthPeerRestart(self):
    # [FUNCTION DOC]
    # Scenario: a cluster with one chief and one worker is started with automatic restart on.
    # The worker exits abruptly after a first all-reduce and is restarted by the runner; the chief
    # polls worker 0 health until it sees a failed-precondition error (meaning the peer came back
    # with a new incarnation). Expected: the runner joins without error.

    def worker_fn():
      # [FUNCTION DOC]
      # Worker body shared by chief and worker. Enables collective ops and runs one all-reduce
      # of a constant over a group of two. A worker task then terminates its process
      # immediately with exit status 1 (to be auto restarted). The chief loops: sleeps one
      # second, checks health of worker 0, ignores unavailable errors and stops when a
      # failed-precondition error is raised. Returns nothing.
      cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
      enable_collective_ops(cluster_resolver)

      collective_ops.all_reduce(
          constant_op.constant(1.),
          group_size=2,
          group_key=100,
          instance_key=100,
          merge_op="Add",
          final_op="Id",
          communication_hint="ring")

      if cluster_resolver.task_type == "worker":
        # MultiProcessRunner will auto restart worker-0.
        os._exit(1)  # pylint: disable=protected-access
      else:
        # chief should eventually gets FailedPreconditionError after worker-0
        # has restarted.
        while True:
          time.sleep(1)
          try:
            context.context().check_collective_ops_peer_health(
                "/job:worker/replica:0/task:0", timeout_in_ms=1000)
          except errors.UnavailableError:
            pass
          except errors.FailedPreconditionError:
            break

    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=True, num_workers=1)
    mpr = multi_process_runner.MultiProcessRunner(
        worker_fn, cluster_spec, auto_restart=True)
    mpr.start()
    mpr.join()

  def testCheckHealthInvalidPeer(self):
    # [FUNCTION DOC]
    # Scenario: worker 0 of a two-worker cluster checks the health of a peer address that is
    # not a valid task name. Expected: joining the runner raises an invalid-argument error.

    def worker_fn():
      # [FUNCTION DOC]
      # Worker body: enables collective ops, then checks health of a host-and-port string that
      # is not a cluster task name, with a one second timeout. Expected to raise an
      # invalid-argument error. Returns nothing.
      enable_collective_ops(cluster_resolver_lib.TFConfigClusterResolver())
      context.context().check_collective_ops_peer_health(
          "localhost:12345", timeout_in_ms=1000)

    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    mpr = multi_process_runner.MultiProcessRunner(worker_fn, cluster_spec)
    mpr.start_single_process("worker", 0)
    with self.assertRaises(errors.InvalidArgumentError):
      mpr.join()


two_worker_pool_runner = multi_process_runner.MultiProcessPoolRunner(
    multi_worker_test_base.create_cluster_spec(num_workers=2),
    initializer=lambda: enable_collective_ops(cluster_resolver_lib.
                                              TFConfigClusterResolver()))


@combinations.generate(
    combinations.times(
        combinations.combine(
            mode="eager", num_workers=2, runner=two_worker_pool_runner),
        device_combination))
class AbortCollectiveOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for aborting collective ops across two workers, run in eager mode on the
  # shared two-worker pool runner for each device and communication combination (CPU with ring;
  # GPU with ring or NCCL). Each test enables collectives, aborts them from one worker, checks the
  # failure, then re-enables collectives to confirm recovery.

  def testAbortCommunication(self, device, communication):
    # [FUNCTION DOC]
    # Scenario: skipped for NCCL (multi-worker NCCL untestable). Both workers enable collectives
    # with a barrier and run an all-reduce to complete group and instance resolution. On task 1 a
    # background thread aborts collectives with an unavailable error "peer down" after two
    # seconds while another all-reduce is pending; expected: unavailable error mentioning
    # "peer down", and a later all-reduce fails immediately the same way. Then collectives are
    # re-enabled with a barrier and a final all-reduce must succeed. Inputs: device name and
    # communication hint from the parameterization.
    if communication == "NCCL":
      self.skipTest("b/171358086: cannot test multi worker NCCL")
    dev0 = "/device:%s:0" % device
    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    enable_collective_ops_with_barrier(cluster_resolver)
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    # First perform a normal all-reduce to complete the group and instance
    # resolution.
    with ops.device(dev0):
      collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          communication_hint=communication)

    if cluster_resolver.task_id == 1:

      def abort_fn():
        # [FUNCTION DOC]
        # Thread body: sleeps two seconds, then aborts all collective ops in the eager context
        # with an unavailable error carrying the message "peer down". Returns nothing.
        time.sleep(2)
        context.context().abort_collective_ops(errors.UNAVAILABLE, "peer down")

      t = threading.Thread(target=abort_fn)
      t.start()

      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        with ops.device(dev0):
          collective_ops.all_reduce(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              communication_hint=communication)

      # After abortion, subsequent collectives should fail immediately.
      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        with ops.device(dev0):
          collective_ops.all_reduce(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              communication_hint=communication)

      t.join()

    # Enable collective ops again in order to reset the collective executor.
    enable_collective_ops_with_barrier(cluster_resolver)
    with ops.device(dev0):
      collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          communication_hint=communication)

  def testAbortGroupParamsResolution(self, device, communication):
    # [FUNCTION DOC]
    # Scenario: skipped for NCCL. After enabling collectives with a barrier, task 1 launches one
    # all-reduce for a group of two, which hangs in group parameter resolution. A thread aborts
    # collectives after two seconds with unavailable "peer down". Expected: that all-reduce and a
    # following one raise unavailable errors mentioning "peer down". Collectives are then
    # re-enabled with a barrier and a final all-reduce on all workers succeeds. Note the
    # communication hint is not passed to the collectives here.
    if communication == "NCCL":
      self.skipTest("b/171358086: cannot test multi worker NCCL")
    dev0 = "/device:%s:0" % device
    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    enable_collective_ops_with_barrier(cluster_resolver)
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    if cluster_resolver.task_id == 1:

      def abort_fn():
        # [FUNCTION DOC]
        # Thread body: sleeps two seconds, then aborts all collective ops in the eager context
        # with an unavailable error carrying the message "peer down". Returns nothing.
        time.sleep(2)
        context.context().abort_collective_ops(errors.UNAVAILABLE, "peer down")

      t = threading.Thread(target=abort_fn)
      t.start()

      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        # This hangs on params resolution since we're only launching one
        # collective for a group size of 2.
        with ops.device(dev0):
          collective_ops.all_reduce(in_tensor, group_size, group_key,
                                    instance_key)

      # After abortion, subsequent collectives should fail immediately.
      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        with ops.device(dev0):
          collective_ops.all_reduce(in_tensor, group_size, group_key,
                                    instance_key)

      t.join()

    # Enable collective ops again in order to reset the collective executor.
    enable_collective_ops_with_barrier(cluster_resolver)
    with ops.device(dev0):
      collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key)

  def testAbortInstanceParamsResolution(self, device, communication):
    # [FUNCTION DOC]
    # Scenario: skipped for NCCL. All workers enable collectives with a barrier and complete
    # group resolution with an all-reduce. Task 1 then switches to a new instance key and calls a
    # broadcast send for a group of two, which hangs in instance resolution; a thread aborts
    # collectives after two seconds with unavailable "peer down". Expected: that call and a
    # repeat both raise unavailable errors mentioning "peer down". After re-enabling collectives
    # with a barrier and restoring the original instance key, task 0 broadcasts and the other
    # task receives, which must succeed.
    if communication == "NCCL":
      self.skipTest("b/171358086: cannot test multi worker NCCL")
    dev0 = "/device:%s:0" % device
    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    enable_collective_ops_with_barrier(cluster_resolver)
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    # First perform a normal all-reduce to complete the group resolution.
    with ops.device(dev0):
      collective_ops.all_reduce(in_tensor, group_size, group_key, instance_key)

    # We use broadcast to test aborting instance resolution since only broadcast
    # waits for the group.

    if cluster_resolver.task_id == 1:

      def abort_fn():
        # [FUNCTION DOC]
        # Thread body: sleeps two seconds, then aborts all collective ops in the eager context
        # with an unavailable error carrying the message "peer down". Returns nothing.
        time.sleep(2)
        context.context().abort_collective_ops(errors.UNAVAILABLE, "peer down")

      t = threading.Thread(target=abort_fn)
      t.start()

      # Use a different instance key to trigger another instance resolution.
      instance_key = 101
      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        # This hangs on params resolution since we're only launching one
        # collective for a group size of 2.
        with ops.device(dev0):
          collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32,
                                        group_size, group_key, instance_key)

      # After abortion, subsequent collectives should fail immediately.
      with self.assertRaisesRegex(errors.UnavailableError, "peer down"):
        with ops.device(dev0):
          collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32,
                                        group_size, group_key, instance_key)

      t.join()

    # Enable collective ops again in order to reset the collective executor.
    enable_collective_ops_with_barrier(cluster_resolver)
    # Reassign instance_key so that it's the same on each worker.
    instance_key = 100
    with ops.device(dev0):
      if cluster_resolver.task_id == 0:
        collective_ops.broadcast_send(in_tensor, (1,), dtypes.float32,
                                      group_size, group_key, instance_key)
      else:
        collective_ops.broadcast_recv((1,), dtypes.float32, group_size,
                                      group_key, instance_key)


if __name__ == "__main__":
  multi_process_runner.test_main()
