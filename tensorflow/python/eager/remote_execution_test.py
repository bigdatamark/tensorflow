# [MODULE DOC]
# Tests for basic remote eager execution against two in-process local servers: matmul on remote
# tasks (plain and via compiled function), GPU-to-remote copies, py-func placement, remote
# variable reads (also under a gradient tape), changing the server definition, connecting to a
# remote host, and context device updates. The main guard enables eager execution and runs tests.
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

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

JOB_NAME = "remote_device"
ALT_JOB_NAME = "alt_remote_device"


def get_server_def(job_name, local_server_port, remote_server_addresses,
                   task_index):
  # [FUNCTION DOC]
  # HELPER: builds server definitions for the remote execution tests.
  # Inputs: job_name (string), local_server_port (integer for task 0 on localhost),
  # remote_server_addresses (list of host-port strings assigned to tasks 1..n), task_index (this
  # process's task). Returns a server definition message with one job using the grpc protocol.
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

  return server_def


class RemoteExecutionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for remote eager execution. Fixtures: two cached local servers and their targets;
  # before each test the eager context is configured with a cluster of the local task plus both
  # servers under the remote_device job.

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructs the test case and starts two in-process local servers (side effect: opens
    # ports), storing their targets without the grpc prefix.
    super(RemoteExecutionTest, self).__init__(methodName)
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server2 = server_lib.Server.create_local_server()

    self._cached_server1_target = self._cached_server1.target[len("grpc://"):]
    self._cached_server2_target = self._cached_server2.target[len("grpc://"):]

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: picks an unused local port and installs a server definition on the eager context
    # with the local task plus the two cached servers under the remote_device job.
    super(RemoteExecutionTest, self).setUp()
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
    # Fixture: after the base teardown, clears the device scope (enters a null device scope,
    # never exited) and resets the global eager context to avoid polluting other tests.
    super(RemoteExecutionTest, self).tearDown()

    # Clear the current device scope and reset the context to avoid polluting
    # other test cases.
    ops.device(None).__enter__()
    context._reset_context()

  @test_util.run_in_async_and_sync_mode
  @test_util.run_gpu_only
  def testGpuToRemoteCopy(self):
    # [FUNCTION DOC]
    # Scenario (GPU-only, async and sync modes): create a ones matrix on the GPU, copy it to the
    # CPU of remote task 1, and assert the copied values equal the original.
    """Tests that the remote copy happens satisfactorily."""
    x1 = array_ops.ones([2, 2]).gpu()
    with ops.device("/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x2 = x1._copy()  # pylint: disable=protected-access

    np.testing.assert_array_equal(x1.numpy(), x2.numpy())

  @test_util.run_in_async_and_sync_mode
  @test_util.run_gpu_only
  def testGpuToRemoteOp(self):
    # [FUNCTION DOC]
    # Scenario (GPU-only, async and sync modes): create a ones matrix on the local GPU and run a
    # matrix multiplication of it with itself on remote task 1 CPU; asserts the all-twos result.
    with ops.device("gpu:0"):
      x = array_ops.ones([2, 2])
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      y = math_ops.matmul(x, x)

    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testDefunMatmul(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): wrap matmul in a compiled function, create ones matrices
    # on tasks 1 and 2, call the compiled function under task 2's device scope and assert the
    # all-twos result.
    """Basic remote eager execution with defun."""

    mm_defun = def_function.function(math_ops.matmul)
    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    with ops.device("job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME):
      x2 = array_ops.ones([2, 2])
      y = mm_defun(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testSimpleMatmul(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): create ones matrices on remote tasks 1 and 2, multiply
    # them with an eager matmul under task 2's device scope, and assert the all-twos result.
    """Basic remote eager execution."""

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    with ops.device("job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME):
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  def testEagerPyFuncPlacement(self):
    # [FUNCTION DOC]
    # Scenario: returns immediately (passes) unless eager execution is active outside functions.
    # Otherwise, under remote task 1's device scope, creates a constant and wraps a squaring
    # callback with an eager py-func op. Asserts the py-func op is placed on the local task 0
    # CPU (callbacks must run in the local address space) and that its evaluated value is nine.
    if not ops.executing_eagerly_outside_functions():
      return

    def f(x):
      # [FUNCTION DOC]
      # Python callback returning the elementwise square of the input tensor.
      return math_ops.square(x)

    with ops.device("/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      const_op = constant_op.constant(3.0, dtype=dtypes.float32)
      # PyFuncOp should be placed on the localhost's address space.
      py_func_op = script_ops.eager_py_func(
          func=f, inp=[const_op], Tout=dtypes.float32)
      self.assertEqual(py_func_op.device,
                       "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME)
      self.assertEqual(self.evaluate(py_func_op), 9.0)

  @test_util.run_in_async_and_sync_mode
  def testSimpleWeightRead(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): create a one-element resource variable (value two) on
    # remote task 1, compute its square, and assert the loss equals four.
    """Basic remote eager weight read."""

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      w = resource_variable_ops.ResourceVariable([[2.0]])
      loss = w * w
    np.testing.assert_array_equal([[4.0]], loss.numpy())

  @test_util.run_in_async_and_sync_mode
  def testTapeWeightRead(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): create a one-element variable (value three) on remote
    # task 1, compute its square under a gradient tape, and take the gradient. Asserts the loss
    # is nine and the gradient is six.
    """Remote eager weight read in a tape."""

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      w = resource_variable_ops.ResourceVariable([[3.0]])
      with backprop.GradientTape() as tape:
        loss = w * w

      grad = tape.gradient(loss, w)
    np.testing.assert_array_equal([[9.0]], loss.numpy())
    np.testing.assert_array_equal([[6.0]], grad.numpy())

  @test_util.run_in_async_and_sync_mode
  def testServerDefChanged(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): replace the server definition with one using an alternate
    # job name over the same servers, run a matmul using a task 1 tensor under the alternate job
    # and check the all-twos result; then set the original job name back and repeat the check.
    """Update server def, and run ops on new cluster."""
    context.set_server_def(
        server_def=get_server_def(
            ALT_JOB_NAME,
            local_server_port=0,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % ALT_JOB_NAME):
      x1 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Set the server def back to JOB_NAME
    context.set_server_def(
        server_def=get_server_def(
            JOB_NAME,
            local_server_port=0,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

    with ops.device("job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME):
      x1 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testConnectToRemoteServer(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): reset the eager context, connect to the first local server
    # as a remote host (default job worker), and run a matmul of ones matrices on worker task 0;
    # asserts the all-twos result.
    """Basic server connection."""
    context._reset_context()
    remote.connect_to_remote_host(self._cached_server1_target)

    with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
      x1 = array_ops.ones([2, 2])
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testContextDeviceUpdated(self):
    # [FUNCTION DOC]
    # Scenario (async and sync modes): under the plain local CPU device scope, multiply two ones
    # matrices and assert the all-twos result and that the output's device is the local task 0
    # CPU of the remote_device job, showing the context's local device name was updated.
    """Tests that the context device is correctly updated."""

    with ops.device("cpu:0"):
      x1 = array_ops.ones([2, 2])
      x2 = array_ops.ones([2, 2])
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # `y` is placed on the local CPU as expected.
    self.assertEqual(y.device,
                     "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
