# [MODULE DOC]
# Tests for dynamic cluster updates in remote eager execution: adding, removing and replacing
# servers in a running cluster via update-server-def, with plain ops, compiled functions,
# distributed functions, multi-threaded launches, multiple executors and liveness checks.
# Uses four in-process local servers. The main guard enables eager execution and runs the tests.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for remote eager execution."""

import os
import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.core.distributed_runtime.preemption import gen_check_preemption_op
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib

JOB_NAME = "remote_device"


def get_server_def(job_name, local_server_port, remote_server_addresses,
                   task_index):
  # [FUNCTION DOC]
  # HELPER: builds server definitions for the dynamic cluster tests.
  # Inputs: job_name (string), local_server_port (integer used for task 0 on localhost),
  # remote_server_addresses (list of host-port strings that become tasks 1..n), task_index
  # (which task this process is). Returns a server definition message with one job, grpc
  # protocol, and the coordination service type set to standalone in the default session config.
  """Returns a server def with a single job + multiple tasks."""
  cluster_def = cluster_pb2.ClusterDef()
  job_def = cluster_def.job.add()
  job_def.name = job_name
  job_def.tasks[0] = "localhost:%d" % local_server_port

  for i, remote_server_address in enumerate(remote_server_addresses, start=1):
    job_def.tasks[i] = remote_server_address

  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_def,
      job_name=job_name,
      task_index=task_index,
      protocol="grpc")
  server_def.default_session_config.experimental.coordination_config.service_type = "standalone"

  return server_def


class DynamicClusterTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering dynamic updates of a remote eager cluster. Fixtures: four cached local
  # servers and their targets, prebuilt server definitions for several server combinations (s1;
  # s1+s2; s1+s3; s4+s3; s1+s2+s3; s1..s4), and device name strings for the local task and
  # tasks 1 through 4. Each test starts from a cluster of the local task plus servers 1 and 2.

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructs the test case, starts four in-process local servers (side effect: opens ports),
    # stores their targets without the grpc prefix, builds the server definitions listed in the
    # class doc (all with local port zero and task index zero) and the device name strings.
    super(DynamicClusterTest, self).__init__(methodName)
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server2 = server_lib.Server.create_local_server()
    self._cached_server3 = server_lib.Server.create_local_server()
    self._cached_server4 = server_lib.Server.create_local_server()

    self._cached_server1_target = self._cached_server1.target[len("grpc://"):]
    self._cached_server2_target = self._cached_server2.target[len("grpc://"):]
    self._cached_server3_target = self._cached_server3.target[len("grpc://"):]
    self._cached_server4_target = self._cached_server4.target[len("grpc://"):]

    self.server_def_s1 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[self._cached_server1_target],
        task_index=0)
    self.server_def_s1_s2 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server1_target, self._cached_server2_target
        ],
        task_index=0)
    self.server_def_s1_s3 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server1_target, self._cached_server3_target
        ],
        task_index=0)
    self.server_def_s4_s3 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server4_target, self._cached_server3_target
        ],
        task_index=0)
    self.server_def_s1_s2_s3 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server1_target, self._cached_server2_target,
            self._cached_server3_target
        ],
        task_index=0)
    self.server_def_s1_s2_s3_s4 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server1_target, self._cached_server2_target,
            self._cached_server3_target, self._cached_server4_target
        ],
        task_index=0)

    self.device_local = "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME
    self.device_t1 = "/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME
    self.device_t2 = "/job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME
    self.device_t3 = "/job:%s/replica:0/task:3/device:CPU:0" % JOB_NAME
    self.device_t4 = "/job:%s/replica:0/task:4/device:CPU:0" % JOB_NAME

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: disables eager client streaming enqueue through an environment variable (global
    # side effect, never restored), picks an unused local port and installs a server definition
    # for the local task plus servers 1 and 2 on the eager context.
    super(DynamicClusterTest, self).setUp()
    os.environ["TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"] = str(False)
    local_port = pywrap_tfe.TF_PickUnusedPortOrDie()
    context.set_server_def(
        server_def=get_server_def(
            JOB_NAME,
            local_server_port=local_port,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture: after the base teardown, resets the device scope by entering a null device scope
    # (never exited) and resets the global eager context so the next test starts clean.
    super(DynamicClusterTest, self).tearDown()
    ops.device(None).__enter__()
    context._reset_context()

  def testCheckPreemption(self):
    # [FUNCTION DOC]
    # Scenario: a check-preemption op on task 1 succeeds when no notice is set. After setting the
    # default preemption notice key to point at worker task 0 in the config store, running the op
    # on task 2 must raise an aborted error whose payload of the worker-preemption type equals
    # the encoded preempted task name.
    preemption_key = "TF_DEFAULT_PREEMPTION_NOTICE_KEY"
    preemption_task = "/job:worker/task:0"

    with ops.device(self.device_t1):
      gen_check_preemption_op.check_preemption(preemption_key=preemption_key)

    # Simulate a preemption notifier callback invocation.
    context.context().set_config_key_value(preemption_key, preemption_task)
    with self.assertRaises(errors.AbortedError) as cm:
      with ops.device(self.device_t2):
        gen_check_preemption_op.check_preemption(preemption_key=preemption_key)
    self.assertEqual(
        cm.exception.experimental_payloads.get(
            b"type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption"
        ), preemption_task.encode())

  @test_util.run_in_async_and_sync_mode
  def testServerAdded(self):
    # [FUNCTION DOC]
    # Scenario (run in async and sync modes): create a ones matrix on task 1, update the cluster
    # to add server 3, create another on task 3. Asserts a matrix product on the new task 3 of
    # tensors from the old and new servers equals the expected all-twos matrix, and the same
    # product on task 2 (old server) also works.
    """Add a server to cluster, and run remote ops on it."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      x2 = array_ops.ones([2, 2])

    # Test new server accessing resources on old server
    with ops.device(self.device_t3):
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Test old server accessing resources on new server
    with ops.device(self.device_t2):
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testServerRemoved(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): with two servers, multiply tensors from tasks 1 and 2 on
    # task 1 and check the result. Update the cluster to only server 1; ops on task 1 still work,
    # but running an op on removed task 2 must raise an invalid-argument error whose message
    # contains "unknown device".
    """Remove a server from cluster, and run ops on cluster."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])
    with ops.device(self.device_t2):
      x2 = array_ops.ones([2, 2])

    with ops.device(self.device_t1):
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    context.update_server_def(server_def=self.server_def_s1)
    with ops.device(self.device_t1):
      y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Running ops on removed server s2 throws an exception
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device(self.device_t2):
        y = math_ops.matmul(x1, x2)
    self.assertIn("unknown device", cm.exception.message)

    # TODO(haoyuzhang): raise and catch exception when accessing tensors on
    # the removed servers.

  @test_util.run_in_async_and_sync_mode
  def testServerReplaced(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): create a tensor on task 1, replace server 2 by server 3 in
    # the cluster, then multiply the tensor with itself on task 2 (now the new server) and check
    # the all-twos result.
    """Replace remote host_port for a task, and run ops on cluster."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    context.update_server_def(server_def=self.server_def_s1_s3)
    with ops.device(self.device_t2):
      y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testFunctionServerAdded(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): trace and register a matrix-squaring function with a
    # tensor from task 1, add server 3 to the cluster, then run the function on task 3 and on
    # task 1 (with an input created on task 3). Asserts both results are the all-twos matrix.
    """Add a server to cluster, and run remote function on it."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function returning the matrix product of its input with itself.
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    with ops.device(self.device_t3):
      x2 = array_ops.ones([2, 2])
    with ops.device(self.device_t1):
      y = worker_fn(x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testFunctionRunsAtMostOnceInvokedOnce(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): define a compiled squaring function carrying the
    # run-at-most-once attribute, call it once on a local ones matrix and check the all-twos
    # result. Purpose: ensure the kernel tears down cleanly after the function handle was already
    # released.
    """Run a function decorated with `function_runs_at_most_once`."""
    @def_function.function(
        experimental_attributes={"function_runs_at_most_once": True}
    )
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function (run-at-most-once attribute) returning the input matrix squared.
      return math_ops.matmul(i, i)

    x = array_ops.ones([2, 2])
    y = worker_fn(x)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())
    # The kernel will be destroyed at this point. The purpose of this test is
    # to ensure it tears down cleanly because we have already released the
    # function handle.

  @test_util.run_in_async_and_sync_mode
  def testFunctionRunsAtMostOnceInvokedTwice(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): call a compiled run-at-most-once squaring function once
    # successfully (all-twos result), then call it again and expect a not-found error whose
    # message matches a missing handle; in async mode waits for pending async work inside the
    # asserted block so the error surfaces.
    """Run a function decorated with `function_runs_at_most_once` twice."""
    @def_function.function(
        experimental_attributes={"function_runs_at_most_once": True}
    )
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function (run-at-most-once attribute) returning the input matrix squared.
      return math_ops.matmul(i, i)

    x = array_ops.ones([2, 2])
    y = worker_fn(x)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Since the worker_fn can run at most once a second invocation should fail.
    with self.assertRaisesRegex(errors.NotFoundError, "Handle.* not found"):
      worker_fn(x)
      if context.is_async():
        context.async_wait()

  @test_util.run_in_async_and_sync_mode
  def testFunctionServerRemoved(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): trace and register a squaring function with a task 1
    # tensor, update the cluster to only server 1, check the function still runs on task 1, and
    # that running it on removed task 2 raises an invalid-argument error whose message contains
    # "unknown device".
    """Remove a server from cluster, and run ops on cluster."""

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function returning the matrix product of its input with itself.
      return math_ops.matmul(i, i)

    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1)

    with ops.device(self.device_t1):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Running functions on removed server s2 throws an exception
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device(self.device_t2):
        y = worker_fn(x1)
    self.assertIn(" unknown device", cm.exception.message)

    # TODO(haoyuzhang): raise and catch exception when accessing tensors on
    # the removed servers.

  @test_util.run_in_async_and_sync_mode
  def testFunctionServerRemovedAddedBack(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): register a squaring function, then cycle the cluster
    # through servers 1+2+3, 1+2, and 1+2+3 again, running the function on task 3, task 2 and
    # task 3 respectively and asserting the all-twos result each time.
    """Add and remove a server, and run functions on cluster."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function returning the matrix product of its input with itself.
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    context.update_server_def(server_def=self.server_def_s1_s2)
    with ops.device(self.device_t2):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testFunctionServerReplaced(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): register a squaring function with a task 1 tensor, replace
    # server 2 by server 3, run the function on task 2 and assert the all-twos result.
    """Replace remote host_port for a task, and run functions on cluster."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function returning the matrix product of its input with itself.
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1_s3)
    with ops.device(self.device_t2):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testFunctionRegisteredAndRemoved(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): a background thread alternately updates the cluster
    # between servers 1+2 and 1+3 thirty times while the main thread repeatedly defines a
    # squaring function, traces it into a concrete function and discards both. Asserts that no
    # exception was raised in the background thread (via joining the coordinator).
    """Update cluster when other function are registered and removed."""
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    num_calls = 30
    self._coord = coordinator.Coordinator()

    def update_server_def_fn():
      # [FUNCTION DOC]
      # Thread body: inside the coordinator's exception guard, updates the cluster definition
      # num_calls times, alternating between the 1+2 and 1+3 server definitions.
      with self._coord.stop_on_exception():
        for i in range(num_calls):
          context.update_server_def(
              server_def=(self.server_def_s1_s2 if i %
                          2 == 0 else self.server_def_s1_s3))

    t = threading.Thread(target=update_server_def_fn)
    t.start()

    for _ in range(num_calls):

      @def_function.function
      def worker_fn(i):
        # [FUNCTION DOC]
        # Compiled function returning the matrix product of its input with itself; only traced,
        # never run, in this test.
        return math_ops.matmul(i, i)

      concrete_fn = worker_fn.get_concrete_function(x1)
      del concrete_fn
      del worker_fn

    # No exception should be thrown from the thread
    self._coord.join([t])

  def testPendingNodesServerReplaced(self):
    # [FUNCTION DOC]
    # Scenario: register a squaring function, launch ten calls on task 1 so they are pending,
    # then replace server 2 by server 3 and run the function on task 2. Asserts all ten pending
    # results and the new task 2 result equal the all-twos matrix.
    """Update cluster when nodes are still pending on remote workers."""
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function returning the matrix product of its input with itself.
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    # Add enough ops so they are pending when changing the cluster
    num_nodes = 10
    ret = [None] * num_nodes
    for i in range(num_nodes):
      with ops.device(self.device_t1):
        ret[i] = worker_fn(x1)
    # While nodes are still pending on worker s1, replace worker s2 with s3.
    context.update_server_def(server_def=self.server_def_s1_s3)
    with ops.device(self.device_t2):
      y = worker_fn(x1)
    for i in range(num_nodes):
      np.testing.assert_array_equal([[2, 2], [2, 2]], ret[i].numpy())
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testMultiThreadPendingNodesServerReplaced(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): two threads each call a squaring function ten times on
    # task 1 and task 2 respectively while a third thread alternately updates the cluster between
    # servers 1+2 and 1+3; a shared lock serializes every call and every update. After joining
    # all threads, asserts every collected result is the all-twos matrix.
    """Update cluster when other remote function calls are being launched."""
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    num_calls = 10
    lock = threading.Lock()

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function returning the matrix product of its input with itself.
      return math_ops.matmul(i, i)

    def thread_fn(device, results):
      # [FUNCTION DOC]
      # Thread body: num_calls times, under the shared lock, runs the function on the given
      # device and stores the numpy result at the loop index in the results list (mutated).
      for i in range(num_calls):
        lock.acquire()
        with ops.device(device):
          y = worker_fn(x1)
        results[i] = y.numpy()
        lock.release()

    def update_server_def_fn():
      # [FUNCTION DOC]
      # Thread body: num_calls times, under the shared lock, updates the cluster definition,
      # alternating between the 1+2 and 1+3 server definitions.
      for i in range(num_calls):
        lock.acquire()
        context.update_server_def(
            server_def=(self.server_def_s1_s2 if i %
                        2 == 0 else self.server_def_s1_s3))
        lock.release()

    t1_results = [None] * num_calls
    t2_results = [None] * num_calls
    threads = []
    threads.append(
        threading.Thread(target=thread_fn, args=(self.device_t1, t1_results)))
    threads.append(
        threading.Thread(target=thread_fn, args=(self.device_t2, t2_results)))
    threads.append(threading.Thread(target=update_server_def_fn))
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    for result in t1_results + t2_results:
      np.testing.assert_array_equal([[2, 2], [2, 2]], result)

  def testMultiThreadPendingNodesLockFree(self):
    # [FUNCTION DOC]
    # Scenario: like the locked multi-thread test but without a lock. Two threads call a traced
    # squaring function ten times each on tasks 1 and 2 while a third thread updates the cluster
    # thirty times with the 1+2 definition; exceptions are captured by a coordinator. After the
    # coordinator joins, asserts every result is the all-twos matrix.
    """Update cluster when other remote function calls are being launched."""

    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    num_calls = 10
    self._coord = coordinator.Coordinator()

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function returning the matrix product of its input with itself.
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    def thread_fn(device, results):
      # [FUNCTION DOC]
      # Thread body: inside the coordinator guard, runs the function num_calls times on the given
      # device and stores each numpy result into the results list (mutated).
      for i in range(num_calls):
        with self._coord.stop_on_exception():
          with ops.device(device):
            results[i] = worker_fn(x1).numpy()

    def update_server_def_fn():
      # [FUNCTION DOC]
      # Thread body: thirty times, inside the coordinator guard, re-applies the 1+2 server
      # definition to the cluster.
      for _ in range(30):
        with self._coord.stop_on_exception():
          context.update_server_def(self.server_def_s1_s2)

    t1_results = [None] * num_calls
    t2_results = [None] * num_calls
    threads = []
    threads.append(
        threading.Thread(target=thread_fn, args=(self.device_t1, t1_results)))
    threads.append(
        threading.Thread(target=thread_fn, args=(self.device_t2, t2_results)))
    threads.append(threading.Thread(target=update_server_def_fn))
    for t in threads:
      t.start()
    self._coord.join(threads)
    for result in t1_results + t2_results:
      np.testing.assert_array_equal([[2, 2], [2, 2]], result)

  @test_util.run_in_async_and_sync_mode
  def testDistributedFunctionServerAdded(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): register a function that squares its input on task 2 and
    # subtracts zeros, add server 3 to the cluster, run the function on task 3 and assert the
    # all-twos result.
    """Add a server to cluster, and run distributed function on it."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function that squares its input matrix on task 2 and returns the product minus
      # a zeros tensor of the same shape (an identity that forces cross-device data flow).
      with ops.device(self.device_t2):
        mul = math_ops.matmul(i, i)
      return mul - array_ops.zeros_like(mul)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testDistributedFunctionServerRemovedAddedBack(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name and docstring say a server is removed and added back, but the test
    # only shrinks the cluster to server 1 and then grows it to servers 1+2; it never re-adds
    # the same removed server after removal beyond that.
    # Scenario (async and sync modes): register a function that squares its input on task 1,
    # reduce the cluster to server 1 and run it on task 1, then update to servers 1+2 and run it
    # on task 2; each result must be the all-twos matrix.
    """Add then remove a server, and run distributed function on cluster."""
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function that squares its input matrix on task 1 and returns the product minus
      # a zeros tensor of the same shape.
      with ops.device(self.device_t1):
        mul = math_ops.matmul(i, i)
      return mul - array_ops.zeros_like(mul)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1)
    with ops.device(self.device_t1):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    context.update_server_def(server_def=self.server_def_s1_s2)
    with ops.device(self.device_t2):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testDistributedFunctionBothServersReplaced(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): register a function that squares on task 1, adds the
    # input on task 2 and subtracts the input. First replace server 2 by server 3 (servers 1+3),
    # run the function from tasks 1 and 2 and check the all-twos result; then replace server 1
    # by server 4 (servers 4+3) and repeat. Verifies existing and restarted workers get their
    # context view IDs updated correctly.
    """Tests that replacing servers works correctly.

    We create two servers, t1 and t2. We first replace t2, then we replace t1.

    Among other things, this ensures that both already existing, and
    restarted workers have the context view IDs correctly updated.
    """
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function: squares the input on task 1, adds the input on task 2, subtracts the
      # input and returns the result (numerically the squared matrix).
      with ops.device(self.device_t1):
        mul = math_ops.matmul(i, i)
      with ops.device(self.device_t2):
        add = mul + i
      return add - i

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    # Replace task2
    context.update_server_def(server_def=self.server_def_s1_s3)
    for device in (self.device_t1, self.device_t2):
      with ops.device(device):
        y = worker_fn(x1)
      np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Then replace task1
    context.update_server_def(server_def=self.server_def_s4_s3)
    for device in (self.device_t1, self.device_t2):
      with ops.device(device):
        y = worker_fn(x1)
      np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  def testDistributedFunctionPendingNodesServerReplaced(self):
    # [FUNCTION DOC]
    # Scenario: register a function that squares on task 1 and adds then subtracts the input on
    # task 2. One thread runs it ten times on task 1 while another alternately updates the
    # cluster between servers 1+2+3 and 1+2, with exceptions captured by a coordinator. After
    # joining, asserts every result is the all-twos matrix.
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      # [FUNCTION DOC]
      # Compiled function: squares the input on task 1, adds the input on task 2, subtracts it
      # and returns the squared matrix.
      with ops.device(self.device_t1):
        mul = math_ops.matmul(i, i)
      with ops.device(self.device_t2):
        add = mul + i
      return add - i

    worker_fn.get_concrete_function(x1)

    num_calls = 10
    self._coord = coordinator.Coordinator()

    def thread_fn(device, results):
      # [FUNCTION DOC]
      # Thread body: inside the coordinator guard, runs the function num_calls times on the given
      # device, storing each numpy result into the results list (mutated).
      with self._coord.stop_on_exception():
        for i in range(num_calls):
          with ops.device(device):
            y = worker_fn(x1)
          results[i] = y.numpy()

    def update_server_def_fn():
      # [FUNCTION DOC]
      # Thread body: inside the coordinator guard, updates the cluster num_calls times,
      # alternating between servers 1+2+3 and servers 1+2.
      with self._coord.stop_on_exception():
        for i in range(num_calls):
          context.update_server_def(
              server_def=(self.server_def_s1_s2_s3 if i %
                          2 == 0 else self.server_def_s1_s2))

    results = [None] * num_calls
    threads = []
    threads.append(
        threading.Thread(target=thread_fn, args=(self.device_t1, results)))
    threads.append(threading.Thread(target=update_server_def_fn))
    for t in threads:
      t.start()
    self._coord.join(threads)
    for result in results:
      np.testing.assert_array_equal([[2, 2], [2, 2]], result)

  def testParameterServerMultiExecutors(self):
    # [FUNCTION DOC]
    # Scenario: update the cluster to four servers, create variables on tasks 1 and 2, and
    # register a function that reads both, applies a tenth of their sum as a gradient (adding to
    # one, subtracting from the other) and returns the sum. Two threads, each with its own
    # synchronous executor, run the function ten times on tasks 3 and 4 while another thread
    # repeatedly re-applies the four-server definition. Asserts the summed results are positive
    # (values are non-deterministic, so only validity, not exact numbers, is checked).
    context.update_server_def(server_def=self.server_def_s1_s2_s3_s4)

    with ops.device(self.device_t1):
      v1 = variables.Variable(initial_value=0.)
    with ops.device(self.device_t2):
      v2 = variables.Variable(initial_value=10.)

    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # Compiled function: reads both variables, computes a tenth of their sum, adds it to the
      # first variable and subtracts it from the second (side effects on the variables), and
      # returns the sum of the two variables.
      x1 = v1.read_value()
      x2 = v2.read_value()
      grad = (x1 + x2) * 0.1
      v1.assign_add(grad)
      v2.assign_sub(grad)
      return v1 + v2

    worker_fn.get_concrete_function()

    executor_t3 = executor.new_executor(enable_async=False)
    executor_t4 = executor.new_executor(enable_async=False)

    num_calls = 10
    self._coord = coordinator.Coordinator()

    def thread_fn(executor_obj, device, results):
      # [FUNCTION DOC]
      # Thread body: inside the coordinator guard, num_calls times runs the function under the
      # given executor and device, storing the results into the results list (mutated).
      with self._coord.stop_on_exception():
        for i in range(num_calls):
          with context.executor_scope(executor_obj):
            with ops.device(device):
              results[i] = worker_fn()

    def update_server_def_fn():
      # [FUNCTION DOC]
      # Thread body: inside the coordinator guard, thirty times re-applies the four-server
      # definition to the cluster.
      with self._coord.stop_on_exception():
        for _ in range(30):
          context.update_server_def(self.server_def_s1_s2_s3_s4)

    t3_results = [None] * num_calls
    t4_results = [None] * num_calls
    threads = []
    threads.append(
        threading.Thread(
            target=thread_fn, args=(executor_t3, self.device_t3, t3_results)))
    threads.append(
        threading.Thread(
            target=thread_fn, args=(executor_t4, self.device_t4, t4_results)))
    threads.append(threading.Thread(target=update_server_def_fn))
    for t in threads:
      t.start()
    self._coord.join(threads)

    # Cannot assert individual values since the results are non-deterministic.
    # By summing up the value we ensure that there are all reasonable and valid
    # numbers (not `None` or `NaN`).
    total = np.sum(t3_results + t4_results)
    self.assertGreater(total, 0)

  def testCheckAlive(self):
    # [FUNCTION DOC]
    # Scenario: checking liveness before the context is initialized must raise a value error
    # saying the context is not initialized. After initializing the context, tasks 0 and 1 of the
    # remote job are alive (true), while task 10 raises an invalid-argument error about being
    # unable to find the worker interface.
    with self.assertRaisesRegex(ValueError, "Context is not initialized."):
      context.check_alive("/job:remote_device/task:0")
    context.context().ensure_initialized()

    self.assertTrue(context.check_alive("/job:remote_device/replica:0/task:0"))
    self.assertTrue(context.check_alive("/job:remote_device/replica:0/task:1"))

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Unable to find worker interface"):
      context.check_alive("/job:remote_device/replica:0/task:10")


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
