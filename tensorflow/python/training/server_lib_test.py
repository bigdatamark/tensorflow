# [MODULE DOC]
# Main test module for the server library (the training server, cluster spec and server
# definition handling). Contains three test cases: one running real local gRPC servers
# (sessions, resets, large tensors, cancellation, config, restarts, session-state isolation),
# one checking server definition protocol buffers built from cluster specs, and one checking
# cluster spec conversions, accessors, truthiness and equality. Other server tests that need
# several differently configured servers live in sibling files because they were flaky here.
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
"""Tests for tf.GrpcServer."""

import time

import numpy as np

from xla.tsl.protobuf import rpc_options_pb2
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_ops
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import server_lib


class GrpcServerTest(test.TestCase):
  # [CLASS DOC]
  # Test case exercising real in-process gRPC servers through sessions. Shared fixture: every
  # test instance creates one local server in its constructor and reuses it as the cached
  # server; tests that need a different server or a port clash create their own.

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructor. Input: methodName, the name of the test method this instance runs (default
    # the standard run-test name). Calls the base constructor, then starts a local single-task
    # gRPC server and stores it as the cached server shared by this instance's test. Side
    # effect: one server per test instance is started when the test suite is loaded.
    super(GrpcServerTest, self).__init__(methodName)
    self._cached_server = server_lib.Server.create_local_server()

  def testRunStep(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, builds a product of two small constant matrices and runs it
    # in a session connected to the cached server. Expected: the result is the single value 4.
    server = self._cached_server
    with ops.Graph().as_default():
      with session.Session(server.target) as sess:
        c = constant_op.constant([[2, 1]])
        d = constant_op.constant([[1], [2]])
        e = math_ops.matmul(c, d)
        self.assertAllEqual([[4]], sess.run(e))
      # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  def testMultipleSessions(self):
    # [FUNCTION DOC]
    # Scenario: builds a matrix product graph and runs it in two simultaneous sessions
    # connected to the cached server, then closes both. Expected: both sessions return the
    # single value 4.
    server = self._cached_server
    with ops.Graph().as_default():
      c = constant_op.constant([[2, 1]])
      d = constant_op.constant([[1], [2]])
      e = math_ops.matmul(c, d)

      sess_1 = session.Session(server.target)
      sess_2 = session.Session(server.target)

      self.assertAllEqual([[4]], sess_1.run(e))
      self.assertAllEqual([[4]], sess_2.run(e))

      sess_1.close()
      sess_2.close()
      # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  # Verifies various reset failures.
  def testResetFails(self):
    # [FUNCTION DOC]
    # Scenario: creates a variable in a named container and one in the default container.
    # Expected: resetting a non-existent target raises a not-found error; resetting a target
    # with no server listening, with a very short operation timeout in the config, raises a
    # deadline-exceeded error; resetting the cached server with a container name that matches
    # nothing leaves both variables valid (values 1.0 and 2.0 before and after, using a new
    # session after the reset).
    with ops.Graph().as_default():
      # Creates variable with container name.
      with ops.container("test0"):
        v0 = variable_v1.VariableV1(1.0, name="v0")
      # Creates variable with default container.
      v1 = variable_v1.VariableV1(2.0, name="v1")
      # Verifies resetting the non-existent target returns error.
      with self.assertRaises(errors_impl.NotFoundError):
        session.Session.reset("nonexistent", ["test0"])

      # Verifies resetting with config.
      # Verifies that resetting target with no server times out.
      with self.assertRaises(errors_impl.DeadlineExceededError):
        session.Session.reset(
            "grpc://localhost:0",
            ["test0"],
            config=config_pb2.ConfigProto(operation_timeout_in_ms=5),
        )

      # Verifies no containers are reset with non-existent container.
      server = self._cached_server
      sess = session.Session(server.target)
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual(1.0, sess.run(v0))
      self.assertAllEqual(2.0, sess.run(v1))
      # No container is reset, but the server is reset.
      session.Session.reset(server.target, ["test1"])
      # Verifies that both variables are still valid.
      sess = session.Session(server.target)
      self.assertAllEqual(1.0, sess.run(v0))
      self.assertAllEqual(2.0, sess.run(v1))

  def _useRPCConfig(self):
    # [FUNCTION DOC]
    # HELPER: Supplies the session configuration used by the large-tensor, cancellation and
    # timeout tests.
    # Returns a session configuration whose RPC options force the gRPC transport even when the
    # client and master are in the same process. No inputs, no side effects.
    """Return a `tf.compat.v1.ConfigProto` that ensures we use the RPC stack for tests.

    This configuration ensures that we continue to exercise the gRPC
    stack when testing, rather than using the in-process optimization,
    which avoids using gRPC as the transport between a client and
    master in the same process.

    Returns:
      A `tf.compat.v1.ConfigProto`.
    """
    return config_pb2.ConfigProto(
        rpc_options=rpc_options_pb2.RPCOptions(
            use_rpc_for_inprocess_master=True
        )
    )

  def testLargeConstant(self):
    # [FUNCTION DOC]
    # Scenario: embeds a large (10000 by 3000 float32) constant in the graph and fetches its
    # shape through a session forced to use the RPC stack. Expected: the shape equals the
    # original dimensions, showing large graphs can be sent to the server.
    server = self._cached_server
    with session.Session(server.target, config=self._useRPCConfig()) as sess:
      const_val = np.empty([10000, 3000], dtype=np.float32)
      const_val.fill(0.5)
      c = constant_op.constant(const_val)
      shape_t = array_ops.shape(c)
      self.assertAllEqual([10000, 3000], sess.run(shape_t))

  def testLargeFetch(self):
    # [FUNCTION DOC]
    # Scenario: fetches a large (10000 by 3000) tensor filled with 0.5 over the RPC stack.
    # Expected: the fetched array equals the locally built array of the same value and shape.
    server = self._cached_server
    with session.Session(server.target, config=self._useRPCConfig()) as sess:
      c = array_ops.fill([10000, 3000], 0.5)
      expected_val = np.empty([10000, 3000], dtype=np.float32)
      expected_val.fill(0.5)
      self.assertAllEqual(expected_val, sess.run(c))

  def testLargeFeed(self):
    # [FUNCTION DOC]
    # Scenario: feeds a large (10000 by 3000) array filled with 0.5 into a placeholder and
    # computes its minimum and maximum over the RPC stack. Expected: both equal 0.5, showing
    # large feeds are transmitted intact.
    server = self._cached_server
    with session.Session(server.target, config=self._useRPCConfig()) as sess:
      feed_val = np.empty([10000, 3000], dtype=np.float32)
      feed_val.fill(0.5)
      p = array_ops.placeholder(dtypes.float32, shape=[10000, 3000])
      min_t = math_ops.reduce_min(p)
      max_t = math_ops.reduce_max(p)
      min_val, max_val = sess.run([min_t, max_t], feed_dict={p: feed_val})
      self.assertEqual(0.5, min_val)
      self.assertEqual(0.5, max_val)

  def testCloseCancelsBlockingOperation(self):
    # [FUNCTION DOC]
    # Scenario: a queue is filled with one element and drained once, then a background thread
    # runs a dequeue on the now empty queue (it blocks). After a short sleep the session is
    # closed. Expected: the blocked dequeue fails with a cancelled error whose message mentions
    # the session close, and the thread finishes.
    server = self._cached_server
    with ops.Graph().as_default():
      sess = session.Session(server.target, config=self._useRPCConfig())

      q = data_flow_ops.FIFOQueue(10, [dtypes.float32])
      enqueue_op = q.enqueue(37.0)
      dequeue_t = q.dequeue()

      sess.run(enqueue_op)
      sess.run(dequeue_t)

      def blocking_dequeue():
        # [FUNCTION DOC]
        # HELPER: Thread body for the close-cancels-blocking-operation test.
        # Runs a dequeue on the empty queue and asserts that it raises a cancelled error
        # mentioning the session close. No inputs, returns nothing.
        with self.assertRaisesRegex(
            errors_impl.CancelledError, "Session::Close"
        ):
          sess.run(dequeue_t)

      blocking_thread = self.checkedThread(blocking_dequeue)
      blocking_thread.start()
      time.sleep(0.5)
      sess.close()
      blocking_thread.join()

  def testInteractiveSession(self):
    # [FUNCTION DOC]
    # Scenario: opens an interactive session on the cached server (the native layer warns that
    # an option is unsupported but ignores it), evaluates a constant and closes the session.
    # Expected: the value 42.0 is returned.
    server = self._cached_server
    # Session creation will warn (in C++) that the place_pruned_graph option
    # is not supported, but it should successfully ignore it.
    sess = session.InteractiveSession(server.target)
    c = constant_op.constant(42.0)
    self.assertEqual(42.0, self.evaluate(c))
    sess.close()

  def testSetConfiguration(self):
    # [FUNCTION DOC]
    # Scenario: builds a session config limiting GPU memory fraction to 0.1 and creates
    # (without starting) a server first through the local-server factory and then from an
    # explicit server definition with the config override. Expected: in both cases the server's
    # definition holds that memory fraction in its default session config.
    config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.1)
    )

    # Configure a server using the default local server options.
    server = server_lib.Server.create_local_server(config=config, start=False)
    self.assertEqual(
        0.1,
        server.server_def.default_session_config.gpu_options.per_process_gpu_memory_fraction,
    )

    # Configure a server using an explicit ServerDefd with an
    # overridden config.
    cluster_def = server_lib.ClusterSpec(
        {"localhost": ["localhost:0"]}
    ).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name="localhost", task_index=0, protocol="grpc"
    )
    server = server_lib.Server(server_def, config=config, start=False)
    self.assertEqual(
        0.1,
        server.server_def.default_session_config.gpu_options.per_process_gpu_memory_fraction,
    )

  def testRestartedMaster(self):
    # [FUNCTION DOC]
    # Scenario: two master servers (old and new) share the same master task name and use the
    # cached server as worker. A session on the old master and a session on the new master are
    # created, each with a cluster definition naming its master and the worker, and each is
    # checked to list both CPU devices. A constant doubled on the worker runs through the new
    # session. Expected: the new session returns 2.0, and the old session then fails with an
    # aborted error saying the session handle is not found (its worker session was collected
    # because the new master has a different incarnation). Both sessions are closed at the end.
    master_old = server_lib.Server.create_local_server()
    master_new = server_lib.Server.create_local_server()
    worker = self._cached_server

    def get_cluster_def(master, worker):
      # [FUNCTION DOC]
      # HELPER: Builds the cluster definition used by the restarted-master test.
      # Inputs: master and worker, two servers. Returns a cluster definition with a master job
      # and a worker job, each with one task whose address is the server's target with the
      # transport prefix removed.
      cluster_def = cluster_pb2.ClusterDef()
      job = cluster_def.job.add()
      job.name = "master"
      job.tasks[0] = master.target[len("grpc://") :]
      job = cluster_def.job.add()
      job.name = "worker"
      job.tasks[0] = worker.target[len("grpc://") :]
      return cluster_def

    def check_session_devices(sess):
      # [FUNCTION DOC]
      # HELPER: Assertion helper for the restarted-master test.
      # Input: sess, a session. Lists the session's devices and asserts that the CPU devices of
      # task zero of both the master and worker jobs are present. Returns nothing.
      # Make sure we have the correct set of cluster devices
      devices = sess.list_devices()
      device_names = set(d.name for d in devices)
      self.assertIn("/job:master/replica:0/task:0/device:CPU:0", device_names)
      self.assertIn("/job:worker/replica:0/task:0/device:CPU:0", device_names)

    with ops.Graph().as_default():
      # Construct a simple graph that runs ops on remote worker
      with ops.device("/job:worker/replica:0/task:0/device:CPU:0"):
        a = constant_op.constant([1.0])
        b = a + a

      config = config_pb2.ConfigProto(
          cluster_def=get_cluster_def(master_old, worker)
      )
      sess_old = session.Session(master_old.target, config=config)
      check_session_devices(sess_old)

      # Create a session with the new master and the worker.
      # The new master has the same task name ('/job:master/replica:0/task:0')
      # as the old master, but is initiated from a different server thus has a
      # different incarnation. This triggers the WorkerSession on worker with
      # the old master incarnation to be garbage collected.

      config = config_pb2.ConfigProto(
          cluster_def=get_cluster_def(master_new, worker)
      )
      sess_new = session.Session(master_new.target, config=config)
      check_session_devices(sess_new)

      # Running on worker with the new session should work as expected
      v = sess_new.run(b)
      self.assertAllEqual(v, [2.0])

      # Running on worker with the old session should raise an exception since
      # the WorkerSession of the old session has been garbage collected
      with self.assertRaisesRegex(
          errors_impl.AbortedError, "Session handle is not found"
      ):
        sess_old.run(b)

    sess_old.close()
    sess_new.close()

  def testInvalidHostname(self):
    # [FUNCTION DOC]
    # Scenario: creates a server whose only task address has a host but no port. Expected: an
    # invalid-argument error whose message mentions the port.
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError, "port"):
      _ = server_lib.Server(
          {"local": ["localhost"]}, job_name="local", task_index=0
      )

  def testTimeoutRaisesException(self):
    # [FUNCTION DOC]
    # Scenario: runs a dequeue on an empty queue with a one second timeout in the run options,
    # once with a default session and once with a session forced to use the RPC stack.
    # Expected: both raise a deadline-exceeded error.
    server = self._cached_server
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(1, [dtypes.float32])
      blocking_t = q.dequeue()

      with session.Session(server.target) as sess:
        with self.assertRaises(errors_impl.DeadlineExceededError):
          sess.run(
              blocking_t, options=config_pb2.RunOptions(timeout_in_ms=1000)
          )

      with session.Session(server.target, config=self._useRPCConfig()) as sess:
        with self.assertRaises(errors_impl.DeadlineExceededError):
          sess.run(
              blocking_t, options=config_pb2.RunOptions(timeout_in_ms=1000)
          )

  def testTwoServersSamePort(self):
    # [FUNCTION DOC]
    # Scenario: tries to start a second server bound to the same address as the cached server.
    # Expected: creation fails with an unknown error.
    # Starting a server with the same target as the cached server should fail.
    server = self._cached_server
    with self.assertRaises(errors_impl.UnknownError):
      _ = server_lib.Server({"local_2": [server.target[len("grpc://") :]]})

  def testExtendAfterQueueRunners(self):
    # [FUNCTION DOC]
    # Scenario: in a session on the cached server, creates an input producer queue, then adds a
    # non-trainable variable, initializes variables, starts the queue runners and assigns a new
    # value to the variable (extending the graph after queue runners started). Expected: all
    # steps succeed without error; the only explicit assertion is that the queue exists.
    server = self._cached_server
    with session.Session(server.target) as sess:
      input_queue = input_ops.input_producer(
          constant_op.constant([0.0], dtype=dtypes.float32)
      )
      self.assertIsNotNone(input_queue)

      var = variable_v1.VariableV1(
          1.0, dtype=dtypes.float32, trainable=False, name="var"
      )

      sess.run(variables.global_variables_initializer())
      queue_runner_impl.start_queue_runners(sess)
      sess.run(var.assign(3.0))

  def testIsolateSessionState(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: A comment in the body says that initially all variables are initialized, but
    # the assertions expect failed-precondition errors, i.e. they are uninitialized.
    # Scenario: a variable with a feedable initial value and unchecked shape is used from four
    # sessions on the cached server: two sharing session state and two with isolated state.
    # Expected: initially every session fails with a failed-precondition error. Initializing in
    # one sharing session makes the value visible to the other sharing session only. A shape
    # change through the other sharing session is also shared. Initializing in an isolated
    # session affects only that session, and isolated sessions may hold different values and
    # shapes for the same variable.
    server = self._cached_server

    with ops.Graph().as_default():
      init_value = array_ops.placeholder(dtypes.int32)
      v = variable_v1.VariableV1(init_value, validate_shape=False, name="v")

      sharing_config = config_pb2.ConfigProto(isolate_session_state=False)
      sharing_sess_0 = session.Session(server.target, config=sharing_config)
      sharing_sess_1 = session.Session(server.target, config=sharing_config)

      isolate_config = config_pb2.ConfigProto(isolate_session_state=True)
      isolate_sess_0 = session.Session(server.target, config=isolate_config)
      isolate_sess_1 = session.Session(server.target, config=isolate_config)

      # Initially all variables are initialized.
      for sess in [
          sharing_sess_0,
          sharing_sess_1,
          isolate_sess_0,
          isolate_sess_1,
      ]:
        with self.assertRaises(errors_impl.FailedPreconditionError):
          sess.run(v)

      # Shared sessions will see each other's updates, but isolated sessions
      # will not.
      sharing_sess_0.run(v.initializer, feed_dict={init_value: 86})
      self.assertAllEqual(86, sharing_sess_0.run(v))
      self.assertAllEqual(86, sharing_sess_1.run(v))
      with self.assertRaises(errors_impl.FailedPreconditionError):
        isolate_sess_0.run(v)
      with self.assertRaises(errors_impl.FailedPreconditionError):
        isolate_sess_1.run(v)

      # Changing the shape works because `validate_shape` is False.
      sharing_sess_1.run(v.initializer, feed_dict={init_value: [86, 99]})
      self.assertAllEqual([86, 99], sharing_sess_0.run(v))
      self.assertAllEqual([86, 99], sharing_sess_1.run(v))
      with self.assertRaises(errors_impl.FailedPreconditionError):
        isolate_sess_0.run(v)
      with self.assertRaises(errors_impl.FailedPreconditionError):
        isolate_sess_1.run(v)

      # Initializing in an isolated session will only affect the state in that
      # session.
      isolate_sess_0.run(v.initializer, feed_dict={init_value: 37})
      self.assertAllEqual([86, 99], sharing_sess_0.run(v))
      self.assertAllEqual([86, 99], sharing_sess_1.run(v))
      self.assertAllEqual(37, isolate_sess_0.run(v))
      with self.assertRaises(errors_impl.FailedPreconditionError):
        isolate_sess_1.run(v)

      # Isolated sessions can have different shapes for the same variable.
      isolate_sess_1.run(v.initializer, feed_dict={init_value: [19, 86]})
      self.assertAllEqual([86, 99], sharing_sess_0.run(v))
      self.assertAllEqual([86, 99], sharing_sess_1.run(v))
      self.assertAllEqual(37, isolate_sess_0.run(v))
      self.assertAllEqual([19, 86], isolate_sess_1.run(v))

  def testShapeChangingIsolateState(self):
    # [FUNCTION DOC]
    # Scenario: in separate graphs, variables with the same name are created. With shared
    # session state: a vector variable is first uninitialized (failed-precondition error), then
    # initialized to 1,2,3; a later graph reading the same-named variable sees 1,2,3 and can
    # re-initialize it to 4,5,6. With isolated session state: a scalar variable of the same
    # name is independent, fails when uninitialized and can be initialized to 37.
    server = self._cached_server
    sharing_config = config_pb2.ConfigProto(isolate_session_state=False)
    isolate_config = config_pb2.ConfigProto(isolate_session_state=True)

    with ops.Graph().as_default():
      w_vector = variable_v1.VariableV1([1, 2, 3], name="w")
      with session.Session(server.target, config=sharing_config) as sess:
        with self.assertRaises(errors_impl.FailedPreconditionError):
          sess.run(w_vector)
        sess.run(w_vector.initializer)
        self.assertAllEqual([1, 2, 3], sess.run(w_vector))

    with ops.Graph().as_default():
      w_vector = variable_v1.VariableV1([4, 5, 6], name="w")
      with session.Session(server.target, config=sharing_config) as sess:
        self.assertAllEqual([1, 2, 3], sess.run(w_vector))
        sess.run(w_vector.initializer)
        self.assertAllEqual([4, 5, 6], sess.run(w_vector))

    with ops.Graph().as_default():
      w_scalar = variable_v1.VariableV1(37, name="w")
      with session.Session(server.target, config=isolate_config) as sess:
        with self.assertRaises(errors_impl.FailedPreconditionError):
          sess.run(w_scalar)
        sess.run(w_scalar.initializer)
        self.assertAllEqual(37, sess.run(w_scalar))


class ServerDefTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering server definition protocol buffers assembled from cluster definitions
  # produced by a cluster spec, including the round trip back through a cluster spec. No shared
  # fixtures.

  def testLocalServer(self):
    # [FUNCTION DOC]
    # Scenario: builds a one-task job named local and a server definition for task zero over
    # grpc. Expected: the definition matches the expected text proto, and converting its cluster
    # definition to a cluster spec and back yields an equal cluster definition.
    cluster_def = server_lib.ClusterSpec(
        {"local": ["localhost:2222"]}
    ).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name="local", task_index=0, protocol="grpc"
    )

    self.assertProtoEquals(
        """
    cluster {
      job { name: 'local' tasks { key: 0 value: 'localhost:2222' } }
    }
    job_name: 'local' task_index: 0 protocol: 'grpc'
    """,
        server_def,
    )

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = server_lib.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testTwoProcesses(self):
    # [FUNCTION DOC]
    # Scenario: a job with two tasks and a server definition for task one. Expected: the
    # definition matches the expected text proto with both tasks, and the cluster definition
    # survives a round trip through a cluster spec unchanged.
    cluster_def = server_lib.ClusterSpec(
        {"local": ["localhost:2222", "localhost:2223"]}
    ).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name="local", task_index=1, protocol="grpc"
    )

    self.assertProtoEquals(
        """
    cluster {
      job { name: 'local' tasks { key: 0 value: 'localhost:2222' }
                          tasks { key: 1 value: 'localhost:2223' } }
    }
    job_name: 'local' task_index: 1 protocol: 'grpc'
    """,
        server_def,
    )

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = server_lib.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testTwoJobs(self):
    # [FUNCTION DOC]
    # Scenario: a cluster with a two-task parameter-server job and a three-task worker job,
    # and a server definition for worker task two. Expected: the definition matches the
    # expected text proto (jobs in sorted order) and the cluster definition round-trips
    # through a cluster spec unchanged.
    cluster_def = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"],
    }).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name="worker", task_index=2, protocol="grpc"
    )

    self.assertProtoEquals(
        """
    cluster {
      job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                       tasks { key: 1 value: 'ps1:2222' } }
      job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                           tasks { key: 1 value: 'worker1:2222' }
                           tasks { key: 2 value: 'worker2:2222' } }
    }
    job_name: 'worker' task_index: 2 protocol: 'grpc'
    """,
        server_def,
    )

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = server_lib.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testDenseAndSparseJobs(self):
    # [FUNCTION DOC]
    # Scenario: a dense parameter-server job and a worker job defined sparsely with task
    # indices zero and two, and a server definition for worker task two. Expected: the
    # definition matches the expected text proto (the missing index one is absent) and the
    # cluster definition round-trips through a cluster spec unchanged.
    cluster_def = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": {0: "worker0:2222", 2: "worker2:2222"},
    }).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name="worker", task_index=2, protocol="grpc"
    )

    self.assertProtoEquals(
        """
    cluster {
      job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                       tasks { key: 1 value: 'ps1:2222' } }
      job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                           tasks { key: 2 value: 'worker2:2222' } }
    }
    job_name: 'worker' task_index: 2 protocol: 'grpc'
    """,
        server_def,
    )

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = server_lib.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())


class ClusterSpecTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the cluster spec class: string form, equivalence of the dictionary,
  # protocol buffer and spec input forms, accessors, truthiness and equality. No shared fixtures.

  def testStringConversion(self):
    # [FUNCTION DOC]
    # Scenario: a two-job cluster spec converted to a string. Expected: the text shows the
    # class name and the jobs sorted by name with their address lists.
    cluster_spec = server_lib.ClusterSpec(
        {"ps": ["ps0:1111"], "worker": ["worker0:3333", "worker1:4444"]}
    )

    expected_str = (
        "ClusterSpec({'ps': ['ps0:1111'], 'worker': ['worker0:3333', "
        "'worker1:4444']})"
    )
    self.assertEqual(expected_str, str(cluster_spec))

  def testProtoDictDefEquivalences(self):
    # [FUNCTION DOC]
    # Scenario: a dense two-job cluster spec. Expected: its cluster definition, and the ones
    # from a spec built from the spec, from its cluster definition and from its dictionary
    # form, all match the same expected text proto.
    cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"],
    })

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """

    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def()
    )
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def(),
    )
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def(),
    )

  def testProtoDictDefEquivalencesWithStringTaskIndex(self):
    # [FUNCTION DOC]
    # Scenario: a worker job given as a dictionary whose task index is the text one instead of
    # an integer. Expected: the index is converted to an integer, and all four construction
    # routes (direct, from spec, from cluster definition, from dictionary form) match the same
    # expected text proto.
    cluster_spec = server_lib.ClusterSpec(
        {"ps": ["ps0:2222", "ps1:2222"], "worker": {"1": "worker1:2222"}}
    )

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 1 value: 'worker1:2222' } }
    """

    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def()
    )
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def(),
    )
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def(),
    )

  def testProtoDictDefEquivalencesWithZeroWorker(self):
    # [FUNCTION DOC]
    # Scenario: a cluster where the worker job has an empty task list. Expected: the worker job
    # still appears (with no tasks) in the cluster definition, and all four construction routes
    # match the same expected text proto.
    cluster_spec = server_lib.ClusterSpec(
        {"ps": ["ps0:2222", "ps1:2222"], "worker": []}
    )

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' }
    """

    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def()
    )
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def(),
    )
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def(),
    )

  def testClusterSpecAccessors(self):
    # [FUNCTION DOC]
    # Scenario: a spec with a two-task job, a three-task job and a sparse job (indices zero and
    # three). Expected: the dictionary form equals the original input; task counts are 2, 3 and
    # 2; task addresses are returned for valid indices; task index lists are 0 and 1, 0 to 2,
    # and 0 and 3; and the job task lists are dense lists, with the sparse job padded by null
    # values between indices. Unknown jobs raise value errors for counts, addresses, indices
    # and job tasks, and an undefined task index in the sparse job raises a value error.
    original_dict = {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"],
        "sparse": {0: "sparse0:2222", 3: "sparse3:2222"},
    }
    cluster_spec = server_lib.ClusterSpec(original_dict)

    self.assertEqual(original_dict, cluster_spec.as_dict())

    self.assertEqual(2, cluster_spec.num_tasks("ps"))
    self.assertEqual(3, cluster_spec.num_tasks("worker"))
    self.assertEqual(2, cluster_spec.num_tasks("sparse"))
    with self.assertRaises(ValueError):
      cluster_spec.num_tasks("unknown")

    self.assertEqual("ps0:2222", cluster_spec.task_address("ps", 0))
    self.assertEqual("sparse0:2222", cluster_spec.task_address("sparse", 0))
    with self.assertRaises(ValueError):
      cluster_spec.task_address("unknown", 0)
    with self.assertRaises(ValueError):
      cluster_spec.task_address("sparse", 2)

    self.assertEqual([0, 1], cluster_spec.task_indices("ps"))
    self.assertEqual([0, 1, 2], cluster_spec.task_indices("worker"))
    self.assertEqual([0, 3], cluster_spec.task_indices("sparse"))
    with self.assertRaises(ValueError):
      cluster_spec.task_indices("unknown")

    # NOTE(mrry): `ClusterSpec.job_tasks()` is not recommended for use
    # with sparse jobs.
    self.assertEqual(["ps0:2222", "ps1:2222"], cluster_spec.job_tasks("ps"))
    self.assertEqual(
        ["worker0:2222", "worker1:2222", "worker2:2222"],
        cluster_spec.job_tasks("worker"),
    )
    self.assertEqual(
        ["sparse0:2222", None, None, "sparse3:2222"],
        cluster_spec.job_tasks("sparse"),
    )
    with self.assertRaises(ValueError):
      cluster_spec.job_tasks("unknown")

  def testEmptyClusterSpecIsFalse(self):
    # [FUNCTION DOC]
    # Scenario: a spec built from an empty dictionary. Expected: it evaluates as false.
    self.assertFalse(server_lib.ClusterSpec({}))

  def testNonEmptyClusterSpecIsTrue(self):
    # [FUNCTION DOC]
    # Scenario: a spec with one job and one task. Expected: it evaluates as true.
    self.assertTrue(server_lib.ClusterSpec({"job": ["host:port"]}))

  def testEq(self):
    # [FUNCTION DOC]
    # Scenario: pairs of cluster specs. Expected: two empty specs are equal; two specs with the
    # same single job are equal; a spec given as an index-keyed dictionary equals one given as
    # a list with the same address.
    self.assertEqual(server_lib.ClusterSpec({}), server_lib.ClusterSpec({}))
    self.assertEqual(
        server_lib.ClusterSpec({"job": ["host:2222"]}),
        server_lib.ClusterSpec({"job": ["host:2222"]}),
    )
    self.assertEqual(
        server_lib.ClusterSpec({"job": {0: "host:2222"}}),
        server_lib.ClusterSpec({"job": ["host:2222"]}),
    )

  def testNe(self):
    # [FUNCTION DOC]
    # Scenario: pairs of cluster specs that differ. Expected: they compare as not equal when
    # one is empty, when job names differ, when an address differs, and when the same
    # addresses appear in a different task order.
    self.assertNotEqual(
        server_lib.ClusterSpec({}),
        server_lib.ClusterSpec({"job": ["host:2223"]}),
    )
    self.assertNotEqual(
        server_lib.ClusterSpec({"job1": ["host:2222"]}),
        server_lib.ClusterSpec({"job2": ["host:2222"]}),
    )
    self.assertNotEqual(
        server_lib.ClusterSpec({"job": ["host:2222"]}),
        server_lib.ClusterSpec({"job": ["host:2223"]}),
    )
    self.assertNotEqual(
        server_lib.ClusterSpec({"job": ["host:2222", "host:2223"]}),
        server_lib.ClusterSpec({"job": ["host:2223", "host:2222"]}),
    )


if __name__ == "__main__":
  test.main()
