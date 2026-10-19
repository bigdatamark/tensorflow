# [MODULE DOC]
# Tests for remote execution in eager mode against local in-process clusters. Covers a single
# worker (multi-device functions, streaming, executors, shape errors, remote call, timeouts),
# asynchronous error handling with datasets, multiple workers (packed inputs and variables,
# cancellation, while loops, parameter server), and multiple jobs (cluster connection through
# specs and resolvers, device filters, error cases). Each test class builds a local cluster in its
# setup and resets the eager context in teardown. The main guard runs the test runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for remote execution."""

import os
import random
import time

from absl.testing import parameterized
import numpy as np
import portpicker

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.training import server_lib
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util import compat


class SingleWorkerTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for eager execution against a cluster with a single remote worker (job worker, task
  # 0), reached through connecting to a remote host. Setup creates the local cluster; teardown
  # clears the device scope and resets the eager context.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a local cluster with one worker and no parameter servers and connects the
    # eager context to that worker as a remote host.
    super(SingleWorkerTest, self).setUp()

    workers, _ = test_util.create_local_cluster(1, 0)
    remote.connect_to_remote_host(workers[0].target)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture: clears the device scope (enters a null device scope, never exited) and resets the
    # global eager context so later tests are unaffected.
    super(SingleWorkerTest, self).tearDown()

    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  def testMultiDeviceFunctionBasic(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function places one addition on the local CPU and a constant on the
    # remote worker CPU, then sums them. Asserts calling it with two gives five and with one
    # gives four.

    @def_function.function
    def basic(i):
      # [FUNCTION DOC]
      # Adds the input to the constant two on the local CPU, creates the constant one on the
      # remote worker CPU, and returns the sum of both.
      with ops.device('/job:localhost/replica:0/task:0/cpu:0'):
        a = constant_op.constant([2]) + i
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        b = constant_op.constant([1])

      return a + b

    self.assertAllEqual(basic(constant_op.constant([2])).numpy(), [5])
    self.assertAllEqual(basic(constant_op.constant([1])).numpy(), [4])

  def testMultiDeviceFunctionVariable(self):
    # [FUNCTION DOC]
    # Scenario: create a variable on the remote worker, wait for async work, then call a local
    # compiled function that adds the remote variable to its input; asserts two plus one is three.
    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      variable_b = variables.Variable(1)

    # Add a sync point to avoid the out-of-order issue of eager async execution
    # (b/155789951).
    context.async_wait()

    @def_function.function
    def with_variable(i):
      # [FUNCTION DOC]
      # Returns the input plus the captured remote variable.
      return i + variable_b

    self.assertAllEqual(with_variable(constant_op.constant([2])).numpy(), [3])

  def testMultiDeviceFunctionRemoteOutput(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function returns one output computed locally and one computed on the
    # remote worker from a remote variable. Asserts the values (two and two) and that the first
    # output's backing device is the local CPU while the second's is the remote worker CPU.
    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      variable_b = variables.Variable(1)

    @def_function.function
    def remote_output(i):
      # [FUNCTION DOC]
      # Returns a pair: the input plus the remote variable, and, computed on the remote worker
      # CPU, the variable plus one.
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        c = variable_b + 1
      return i + variable_b, c

    rets = remote_output(constant_op.constant([1]))
    self.assertAllEqual(rets[0].numpy(), [2])
    self.assertAllEqual(rets[1].numpy(), 2)
    self.assertEqual(rets[0].backing_device,
                     '/job:localhost/replica:0/task:0/device:CPU:0')
    self.assertEqual(rets[1].backing_device,
                     '/job:worker/replica:0/task:0/device:CPU:0')

  def testStreaming(self):
    # [FUNCTION DOC]
    # Scenario: a mini stress test issuing 200 back-to-back additions on the remote worker; with
    # probability about one in ten each iteration also reads the intermediate shape to exercise
    # waiting for remote shapes. Asserts the final tensor holds the iteration count everywhere.
    # Uses unseeded randomness.
    """A mini stress test for streaming - issuing many RPCs back to back."""
    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
      x = array_ops.ones([2, 2])
      y = array_ops.zeros([2, 2])
      num_iters = 200
      for _ in range(num_iters):
        y = x + y
        # Ask for y's shape after every 10 additions on average.
        # This exercises waiting for remote shape logic in TensorHandle.
        if random.randint(1, 10) == 1:
          _ = y.shape
    np.testing.assert_array_equal(
        [[num_iters, num_iters], [num_iters, num_iters]], y.numpy())

  def testTwoExecutors(self):
    # [FUNCTION DOC]
    # Scenario: run a division by zero on the remote worker via the default executor with
    # streaming enqueue, so the error is deferred. Then run a valid division under a second
    # synchronous executor with streaming disabled and assert it returns two (the deferred error
    # is not carried over). Finally waiting on the context must raise an invalid-argument error
    # mentioning division by zero.
    # Run an op on the main executor that by default uses StreamingEnqueue to
    # schedule the op to run on the remote async executor. This op produces an
    # error, i.e., division by zero, but will not be immediately caught due to
    # streaming enqueue.
    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
      a = constant_op.constant(3)
      b = constant_op.constant(0)
      math_ops.div(a, b)

    # Run another op using another executor that disables streaming enqueue,
    # which would run the op using the tf_compute thread pool in the remote
    # worker. Since the op is not run in the same remotes async executor, it
    # will not carry back that error produced by the op above, even though this
    # op is executed synchronously.
    with context.executor_scope(
        executor.new_executor(
            enable_async=False, enable_streaming_enqueue=False)):
      with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
        c = constant_op.constant(4)
        d = constant_op.constant(2)
        self.assertEqual(math_ops.div(c, d).numpy(), 2)

    # Sync on the context to force to catch the error produced by the first op.
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      context.async_wait()
    self.assertIn('division by zero', cm.exception.message)

  def testShapeError_OpByOp(self):
    # [FUNCTION DOC]
    # Scenario: multiplying a two-by-three matrix with a two-by-two matrix on the remote worker
    # in op-by-op mode must raise an invalid-argument error whose message says dimensions must
    # be equal.
    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
      x = array_ops.ones([2, 3])
      y = array_ops.zeros([2, 2])
      with self.assertRaises(errors.InvalidArgumentError) as cm:
        math_ops.matmul(x, y)

    self.assertIn('Dimensions must be equal', cm.exception.message)

  def testShapeError_Function(self):
    # [FUNCTION DOC]
    # Scenario: calling a compiled matmul function with incompatible shapes under the remote
    # worker device scope must raise a value error (raised at trace time) whose first argument
    # says dimensions must be equal.

    @def_function.function
    def matmul_func(x, y):
      # [FUNCTION DOC]
      # Returns the matrix product of the two inputs.
      return math_ops.matmul(x, y)

    x = array_ops.ones([2, 3])
    y = array_ops.zeros([2, 2])

    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
      with self.assertRaises(ValueError) as cm:
        matmul_func(x, y)

    self.assertIn('Dimensions must be equal', cm.exception.args[0])

  def testClientVarible(self):
    # [FUNCTION DOC]
    # Scenario: a variable lives on the client (local job); a compiled function reads it under
    # the local device scope and adds one; the call is made under the remote worker scope.
    # Asserts the result is one.
    var = variables.Variable(initial_value=0)

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Reads the client-side variable on the local task and returns the value plus one.
      with ops.device('/job:localhost/task:0'):
        read = var.read_value()
      return read + 1

    with ops.device('/job:worker/task:0'):
      self.assertAllEqual(func(), 1)

  def testRemoteCall(self):
    # [FUNCTION DOC]
    # Scenario: a concrete function adding one to an integer scalar is invoked through the
    # remote-call op targeting the worker, from a compiled function called under the local scope.
    # Asserts calling it with one returns a single-element list holding two.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def _remote_fn(x):
      # [FUNCTION DOC]
      # HELPER: function to be called remotely; returns the integer input plus one.
      return constant_op.constant(1) + x

    remote_fn = _remote_fn.get_concrete_function()

    @def_function.function
    def func(x):
      # [FUNCTION DOC]
      # Invokes the concrete function on the worker task through the remote-call op, passing the
      # input, and returns its integer output list.
      return functional_ops.remote_call(
          args=[x],
          Tout=[dtypes.int32],
          f=remote_fn,
          target='/job:worker/task:0')

    with ops.device('/job:localhost/task:0'):
      self.assertAllEqual(func(constant_op.constant(1)), [2])

  def testOperationTimeout(self):
    # [FUNCTION DOC]
    # Scenario: reset the context, set a 10 ms operation timeout, create a fresh one-worker
    # cluster and connect. A function dequeuing from an empty queue on the worker must raise a
    # deadline-exceeded error (after waiting for async errors to surface).
    context._reset_context()
    context.context().operation_timeout_in_ms = 10
    workers, _ = test_util.create_local_cluster(1, 0)
    remote.connect_to_remote_host(workers[0].target)

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Dequeues one element from the captured queue; blocks forever as the queue is empty.
      return q.dequeue()

    with self.assertRaises(errors.DeadlineExceededError):
      with ops.device('/job:worker/replica:0/task:0'):
        f()
      # If streaming RPC is enabled, fetch remote errors before end of execution
      context.async_wait()


class RemoteAsyncTest(test.TestCase):
  # [CLASS DOC]
  # Tests for asynchronous remote execution error handling when a remote dataset iterator runs
  # out of data. Setup creates a one-worker cluster and connects to it; teardown resets the
  # eager context.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a local cluster with one worker and connects the eager context to it.
    super(RemoteAsyncTest, self).setUp()

    workers, _ = test_util.create_local_cluster(1, 0)
    remote.connect_to_remote_host(workers[0].target)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context after the base teardown.
    super(RemoteAsyncTest, self).tearDown()

    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  def test_out_of_range_with_while_loop(self):
    # [FUNCTION DOC]
    # Scenario: on the worker, a two-element dataset batched by one and a variable of one are
    # created. A training step takes the next batch and adds its mean to the variable; it is run
    # in an unbounded loop on the worker until an out-of-range or internal error appears, which
    # is cleared via clearing the async error. Asserts the variable ends at four.

    with ops.device('/job:worker/task:0'):
      dataset = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0])
      dataset = dataset.batch(1, drop_remainder=False)
      iterator = iter(dataset)
      v = variables.Variable(1.0)

    @def_function.function
    def train_step(iterator):
      # [FUNCTION DOC]
      # Fetches the next batch from the iterator (raises out-of-range when exhausted) and adds
      # the batch mean to the captured variable; returns nothing.
      i = next(iterator)
      v.assign_add(math_ops.reduce_mean(i))

    while True:
      try:
        with ops.device('/job:worker/task:0'):
          train_step(iterator)
      except (errors.OutOfRangeError, errors.InternalError):
        context.async_clear_error()
        break

    self.assertAllEqual(v.numpy(), 4.0)

  def test_out_of_range_with_for_loop(self):
    # [FUNCTION DOC]
    # Scenario: same dataset and variable as the while-loop variant, but the training step runs
    # in a loop of three steps; on the last step the test waits for async work. An out-of-range
    # error stops the loop and is cleared. Asserts the variable ends at four.

    with ops.device('/job:worker/task:0'):
      dataset = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0])
      dataset = dataset.batch(1, drop_remainder=False)
      iterator = iter(dataset)
      v = variables.Variable(1.0)

    @def_function.function
    def train_step(iterator):
      # [FUNCTION DOC]
      # Fetches the next batch from the iterator (raises out-of-range when exhausted) and adds
      # the batch mean to the captured variable; returns nothing.
      i = next(iterator)
      v.assign_add(math_ops.reduce_mean(i))

    num_steps = 3
    for i in range(num_steps):
      try:
        with ops.device('/job:worker/task:0'):
          train_step(iterator)
        if i == num_steps - 1:
          context.async_wait()
      except errors.OutOfRangeError:
        context.async_clear_error()
        break

    self.assertAllEqual(v.numpy(), 4.0)

  def test_out_of_range_with_async_scope(self):
    # [FUNCTION DOC]
    # Scenario: same dataset and variable, but three training steps run inside an async scope
    # whose exit surfaces any out-of-range error; the error is caught and cleared. Asserts the
    # variable ends at four.

    with ops.device('/job:worker/task:0'):
      dataset = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0])
      dataset = dataset.batch(1, drop_remainder=False)
      iterator = iter(dataset)
      v = variables.Variable(1.0)

    @def_function.function
    def train_step(iterator):
      # [FUNCTION DOC]
      # Fetches the next batch from the iterator (raises out-of-range when exhausted) and adds
      # the batch mean to the captured variable; returns nothing.
      i = next(iterator)
      v.assign_add(math_ops.reduce_mean(i))

    num_steps = 3
    try:
      with context.async_scope():
        for _ in range(num_steps):
          with ops.device('/job:worker/task:0'):
            train_step(iterator)
    except errors.OutOfRangeError:
      context.async_clear_error()

    self.assertAllEqual(v.numpy(), 4.0)


class MultiWorkersTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for eager execution against a cluster with three workers (tasks 0-2 of job worker):
  # multi-device functions, packed inputs and variables, cancellation of remote functions, while
  # loops and a simple parameter-server pattern.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a local cluster with three workers and connects the eager context to all
    # three as remote hosts.
    super(MultiWorkersTest, self).setUp()

    workers, _ = test_util.create_local_cluster(3, 0)
    remote.connect_to_remote_host(
        [workers[0].target, workers[1].target, workers[2].target])

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture: clears the device scope (never exited) and resets the global eager context.
    super(MultiWorkersTest, self).tearDown()

    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  def testReturnRemoteArgument(self):
    # [FUNCTION DOC]
    # Scenario: a tensor is created on worker task 0 and passed to an identity function called
    # under worker task 1's scope; asserts the returned value equals the input.

    @def_function.function
    def local_func(i):
      # [FUNCTION DOC]
      # Identity: returns its input unchanged.
      return i

    with ops.device('/job:worker/replica:0/task:0'):
      x = constant_op.constant([2, 1])

    with ops.device('/job:worker/replica:0/task:1'):
      self.assertAllEqual(local_func(x), [2, 1])

  def testMultiDeviceFunctionAmbiguousDevice(self):
    # [FUNCTION DOC]
    # Scenario: a function places an addition under the bare worker job (matching three tasks).
    # Calling it must raise an invalid-argument error saying the output node must match exactly
    # one device.

    @def_function.function
    def ambiguous_device(i):
      # [FUNCTION DOC]
      # Adds the constant two to the input under an ambiguous worker-job device scope; this
      # ambiguity triggers the error the test expects.
      with ops.device('/job:worker'):
        # Multiple worker tasks, thus ambiguous device found error will be
        # raised.
        return i + constant_op.constant([2])

    with self.assertRaises(errors.InvalidArgumentError) as cm:
      ambiguous_device(constant_op.constant([2])).numpy()

    self.assertIn('the output node must match exactly one device',
                  cm.exception.message)

  # Note that the following tests for remote function cancellation only works
  # when non-streaming RPC. We need to disable streaming explicitly and restore
  # this config to its initial value at the end of each test case.
  def testCancelRemoteFunctionBeforeExecution(self):
    # [FUNCTION DOC]
    # Scenario: disable streaming enqueue via an environment variable (saving the old value),
    # wrap a queue-dequeuing function as a cancelable function, cancel before calling it, and
    # assert calling it on worker task 1 raises a cancelled error. The original environment
    # value is restored (or removed) at the end, but not if an assertion fails earlier.
    remote_async_env_var = 'TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'
    default_streaming = os.environ.get(remote_async_env_var)
    os.environ[remote_async_env_var] = str(False)

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Dequeues one element from the captured empty queue (would block indefinitely).
      return q.dequeue()

    c_mgr = cancellation.CancellationManager()
    cancelable_func = c_mgr.get_cancelable_function(f.get_concrete_function())

    c_mgr.start_cancel()
    with self.assertRaises(errors.CancelledError):
      with ops.device('/job:worker/replica:0/task:1'):
        cancelable_func()

    if default_streaming is None:
      del os.environ[remote_async_env_var]
    else:
      os.environ[remote_async_env_var] = default_streaming

  def testCancelRemoteFunctionDuringExecution(self):
    # [FUNCTION DOC]
    # Scenario: disable streaming enqueue via an environment variable (saving the old value),
    # wrap a queue-dequeuing function as cancelable, start a thread that cancels after half a
    # second, and call the function on worker task 1 (it blocks). Asserts a cancelled error is
    # raised, joins the thread, then restores the environment variable (skipped on failure).
    remote_async_env_var = 'TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'
    default_streaming = os.environ.get(remote_async_env_var)
    os.environ[remote_async_env_var] = str(False)

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Dequeues one element from the captured empty queue (blocks until cancelled).
      return q.dequeue()

    c_mgr = cancellation.CancellationManager()
    cancelable_func = c_mgr.get_cancelable_function(f.get_concrete_function())

    def cancel_thread():
      # [FUNCTION DOC]
      # Thread body: sleeps half a second then triggers cancellation on the cancellation manager.
      time.sleep(0.5)
      c_mgr.start_cancel()

    t = self.checkedThread(cancel_thread)
    t.start()
    with self.assertRaises(errors.CancelledError):
      with ops.device('/job:worker/replica:0/task:1'):
        cancelable_func()
    t.join()

    if default_streaming is None:
      del os.environ[remote_async_env_var]
    else:
      os.environ[remote_async_env_var] = default_streaming

  def testMultiDeviceFunctionOnLocalDevice(self):
    # [FUNCTION DOC]
    # Scenario: a variable lives on worker task 1; a function adds it to the input on worker
    # task 0 and then adds one on the default (local) device. Calling locally with one gives
    # three.
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):
      # [FUNCTION DOC]
      # Adds the input to the task 1 variable on worker task 0, then adds one on the default
      # device, and returns the result.
      with ops.device('/job:worker/replica:0/task:0'):
        a = i + variable_b
      c = a + 1.0
      return c

    self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  def testMultiDeviceFunctionExecutionOrderingWithPackedInput(self):
    # [FUNCTION DOC]
    # Scenario: send twenty requests to worker task 2 to load it, create an input there, pack it
    # into a packed eager tensor, and call a function from worker task 1 that adds one on task 2
    # and one more. Asserts the packed input is ready when the component function arrives, i.e.
    # the result is a vector of twos.
    shape = [2]
    with ops.device('/job:worker/replica:0/task:2/device:CPU:0'):
      # Send 20 remote requests to simulate heavy load on worker:2.
      unused_values = []
      for _ in range(20):
        unused_values.append(array_ops.zeros(shape))
      func_input = array_ops.zeros(shape)

    packed_input = ops.pack_eager_tensors([func_input])

    @def_function.function
    def func(packed_input):
      # [FUNCTION DOC]
      # Adds one to the packed input on worker task 2, then adds one more, returning the result.
      # When worker:2 receives the component function request, packed_input
      # should be ready on worker:2.
      with ops.device('/job:worker/replica:0/task:2/device:CPU:0'):
        ret = packed_input + constant_op.constant(1.0)
      return ret + constant_op.constant(1.0)

    # Run the function on a worker:1
    with ops.device('/job:worker/replica:0/task:1/device:CPU:0'):
      self.assertAllEqual(func(packed_input).numpy(),
                          array_ops.ones(shape).numpy() * 2)

  def testMultiDeviceFunctionWithPackedVariable(self):
    # [FUNCTION DOC]
    # Scenario: variables on worker tasks 0 and 1 have their handles packed into one composite
    # tensor, whose device and backing device must both be the local composite device. A function
    # reads the packed handle on each task and sums the values; asserts the sum is three both
    # when called under a remote worker scope and when called locally.
    with ops.device('/job:worker/replica:0/task:0/device:CPU:0'):
      var0 = resource_variable_ops.ResourceVariable(1.0)
    with ops.device('/job:worker/replica:0/task:1/device:CPU:0'):
      var1 = resource_variable_ops.ResourceVariable(2.0)

    packed_var = ops.pack_eager_tensors([var0.handle, var1.handle])
    self.assertEqual(packed_var.device,
                     '/job:localhost/replica:0/task:0/device:COMPOSITE:0')
    self.assertEqual(packed_var.backing_device,
                     '/job:localhost/replica:0/task:0/device:COMPOSITE:0')

    @def_function.function
    def add_variables():
      # [FUNCTION DOC]
      # Reads the packed variable handle on worker task 0 and on worker task 1 (each reading the
      # component local to that device) and returns the sum of both float values.
      with ops.device('/job:worker/replica:0/task:0/device:CPU:0'):
        read0 = resource_variable_ops.read_variable_op(
            packed_var, dtype=dtypes.float32)
      with ops.device('/job:worker/replica:0/task:1/device:CPU:0'):
        read1 = resource_variable_ops.read_variable_op(
            packed_var, dtype=dtypes.float32)

      return read0 + read1

    # Run the function on a remote device
    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(add_variables().numpy(), 3.0)

    # Run the function on a local worker
    self.assertAllEqual(add_variables().numpy(), 3.0)

  def testMultiDeviceFunctionOnRemoteDeviceWithWait(self):
    # [FUNCTION DOC]
    # Scenario: a variable on worker task 1 is mutated by a slow function launched on task 0 and
    # then by a faster function launched on task 2. Because both are side-effecting and need the
    # remote variable, the second must wait for the first (a sync point). Asserts the second
    # returns seven (initial one plus two plus three, plus one).
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable([1.0])

    @def_function.function
    def remote_function(i):
      # [FUNCTION DOC]
      # Slow side-effecting function: burns time squaring a large matrix a thousand times, adds
      # the input to the remote variable (mutating it) and returns one plus the variable.
      x = array_ops.ones([1000, 1000])
      for _ in range(1, 1000):
        x = x * x
      variable_b.assign_add(i)
      a = 1.0 + variable_b
      return a

    @def_function.function
    def remote_function2(i):
      # [FUNCTION DOC]
      # Fast side-effecting function: adds the input to the remote variable (mutating it) and
      # returns one plus the variable.
      variable_b.assign_add(i)
      a = 1.0 + variable_b
      return a

    # Runs first function:
    # - on remote device
    # - needs remote input
    # - is side impacting
    # - runs much slower
    with ops.device('/job:worker/replica:0/task:0'):
      remote_function(constant_op.constant([2.0]))

    # Runs second function:
    # - on remote device
    # - is side impacting
    # There should be a sync point here and the next function will be executed
    # only after the first function has completed.
    with ops.device('/job:worker/replica:0/task:2'):
      self.assertAllEqual(remote_function2(constant_op.constant([3.0])), [7.0])

  def testMultiDeviceFunctionOnRemoteDevice(self):
    # [FUNCTION DOC]
    # Scenario: a variable on worker task 1 is read by a function placing an addition on worker
    # task 0. The function is called under worker task 0's scope and asserted to return three;
    # if a GPU is available it is also called under task 0's GPU scope with the same assertion.
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):
      # [FUNCTION DOC]
      # Adds the input to the task 1 variable on worker task 0, then adds one and returns it.
      with ops.device('/job:worker/replica:0/task:0'):
        a = i + variable_b
      c = a + 1.0
      return c

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  def testMultiDeviceFunctionRemoteOutput(self):
    # [FUNCTION DOC]
    # Scenario: a variable lives on worker task 1; a function called under worker task 0 returns
    # one output on task 0 and one computed on task 1. Asserts each output's backing device and
    # the values (two and two).
    with ops.device('/job:worker/replica:0/task:1/cpu:0'):
      variable_b = variables.Variable(1)

    @def_function.function
    def remote_output(i):
      # [FUNCTION DOC]
      # Returns a pair: the input plus the variable (placed with the caller) and, computed on
      # worker task 1 CPU, the variable plus one.
      with ops.device('/job:worker/replica:0/task:1/cpu:0'):
        c = variable_b + 1
      return i + variable_b, c

    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      rets = remote_output(constant_op.constant([1]))
    self.assertEqual(rets[0].backing_device,
                     '/job:worker/replica:0/task:0/device:CPU:0')
    self.assertEqual(rets[1].backing_device,
                     '/job:worker/replica:0/task:1/device:CPU:0')
    self.assertAllEqual(rets[0].numpy(), [2])
    self.assertAllEqual(rets[1].numpy(), 2)

  def testMultiDeviceWhileLoopOnRemoteDevice(self):
    # [FUNCTION DOC]
    # Scenario: a variable on worker task 1 is read inside a one-iteration while loop whose body
    # runs on worker task 0. The function is called under task 0 and asserted to return three;
    # also under task 0's GPU scope when a GPU is available.
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):
      # [FUNCTION DOC]
      # Runs a while loop (condition: the counter below one) starting from the input and a
      # counter of zero, and returns the loop-carried value.

      def body(i, _):
        # [FUNCTION DOC]
        # Loop body: adds the task 1 variable to the value on worker task 0, then returns the
        # sum plus one together with a counter of one, which ends the loop.
        with ops.device('/job:worker/replica:0/task:0'):
          a = i + variable_b
        return a + 1.0, 1

      return while_loop.while_loop_v2(lambda _, d: d < 1, body, [i, 0])[0]

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  def testSimpleParameterServer(self):
    # [FUNCTION DOC]
    # Scenario: two variables (zero and ten) live on worker task 2 acting as a parameter server;
    # a function increments the first, decrements the second by two, and returns their sum.
    # Called from worker task 0 it returns nine, then from task 1 it returns eight.

    with ops.device('/job:worker/task:2/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
      v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # Adds one to the first variable, subtracts two from the second (mutating both) and returns
      # the sum of their current values.
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)

    with ops.device('/job:worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)


_GRPC_PREFIX = 'grpc://'


class MultiJobsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for connecting to clusters with several jobs (two workers named my_worker and two
  # parameter servers named my_ps) through cluster specs and resolvers. Fixtures: the cluster
  # spec and a simple resolver whose master is the first parameter server.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a local cluster of two workers and two parameter servers, builds a cluster
    # specification with jobs my_worker and my_ps from the server addresses (grpc prefix
    # stripped), and a simple resolver using the first parameter server as master. Does not
    # connect yet.
    super(MultiJobsTest, self).setUp()

    workers, ps = test_util.create_local_cluster(num_workers=2, num_ps=2)
    cluster = {
        'my_worker': [_strip_prefix(t.target, _GRPC_PREFIX) for t in workers],
        'my_ps': [_strip_prefix(t.target, _GRPC_PREFIX) for t in ps],
    }
    self._cluster = server_lib.ClusterSpec(cluster)
    self._cluster_resolver = SimpleClusterResolver(
        cluster_spec=self._cluster, master=ps[0].target)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture: clears the device scope (never exited) and resets the global eager context.
    super(MultiJobsTest, self).tearDown()

    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  def testMultipleDeviceFoundCheck(self):
    # [FUNCTION DOC]
    # Scenario: connect to the cluster; a function places a device-placement op under a plain
    # CPU scope and upper-cases its string output. Called under worker task 1 and ps task 1, the
    # first output must name that task's CPU (the parent device scope wins despite multiple CPU
    # matches) and the upper-cased output must contain the upper-cased job name.
    remote.connect_to_cluster(self._cluster)

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Under a plain CPU scope, runs the device-placement op, upper-cases its string output,
      # and returns the pair stacked into one string tensor.
      with ops.device('cpu:0'):
        # Multiple CPU:0 devices match would be found, but the CPU:0 from the
        # parent device scope should be picked.
        x = test_ops.device_placement_op()
        y = string_ops.string_upper(x)
        packed_var_0 = array_ops_stack.stack([x, y], 0)
        return packed_var_0

    with ops.device('/job:my_worker/task:1'):
      output = self.evaluate(func())
      self.assertEqual(
          compat.as_bytes('/job:my_worker/replica:0/task:1/device:CPU:0'),
          output[0])
      self.assertIn(compat.as_bytes('/JOB:MY_WORKER'), output[1])
    with ops.device('/job:my_ps/task:1'):
      output = self.evaluate(func())
      self.assertEqual(
          compat.as_bytes('/job:my_ps/replica:0/task:1/device:CPU:0'),
          output[0])
      self.assertIn(compat.as_bytes('/JOB:MY_PS'), output[1])

  def testSimpleParameterServer(self):
    # [FUNCTION DOC]
    # Scenario: connect to the cluster, place two variables (zero and ten) on ps task 0, and run
    # a function that increments the first, decrements the second by two and returns their sum
    # from worker task 0 (expects nine) and worker task 1 (expects eight).
    remote.connect_to_cluster(self._cluster)

    with ops.device('/job:my_ps/task:0/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
      v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # Adds one to the first variable, subtracts two from the second (mutating both) and returns
      # the sum of their values.
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)

    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)

  def testResetClusterWithDifferentJobNames(self):
    # [FUNCTION DOC]
    # Scenario: connect to a one-task cluster whose only job is the local job name and create and
    # increment a variable on it; then connect to a different one-task cluster whose job is
    # named worker (with local job name worker) and do the same. Asserts only that neither step
    # raises. Uses freshly picked unused ports.
    addr = 'localhost:%s' % portpicker.pick_unused_port()
    cluster = server_lib.ClusterSpec({'localhost': [addr]})
    remote.connect_to_cluster(cluster, job_name='localhost')
    with ops.device('/job:localhost/task:0/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
      v1.assign_add(1)

    # Replace job name from 'localhost' to 'worker' in the cluster.
    addr = 'localhost:%s' % portpicker.pick_unused_port()
    cluster = server_lib.ClusterSpec({'worker': [addr]})
    remote.connect_to_cluster(cluster, job_name='worker')

    with ops.device('/job:worker/task:0/device:CPU:0'):
      v2 = variables.Variable(initial_value=0)
      v2.assign_add(1)

  # TODO(b/152224115): Re-enable this test.
  def DISABLED_testSimpleParameterServerWithDeviceFilters(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this test is disabled (its name has a DISABLED prefix so the runner never
    # collects it, pending bug 152224115), so the device-filter behavior is currently untested.
    # Scenario: set device filters so each worker sees only ps tasks and each ps sees only
    # workers, connect with them, place variables on the two ps tasks and run an
    # increment/decrement function from both workers (expecting nine and eight). Running it
    # from a ps task must raise an invalid-argument error naming the other ps task as an unknown
    # device. Afterwards both workers see seven and six; finally the variables are deleted
    # explicitly to avoid errors during garbage collection in later tests.
    cluster_device_filters = server_lib.ClusterDeviceFilters()
    for i in range(2):
      cluster_device_filters.set_device_filters('my_worker', i, ['/job:my_ps'])
      cluster_device_filters.set_device_filters('my_ps', i, ['/job:my_worker'])
    remote.connect_to_cluster(
        self._cluster, cluster_device_filters=cluster_device_filters)

    with ops.device('/job:my_ps/task:0/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
    with ops.device('/job:my_ps/task:1/device:CPU:0'):
      v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # Adds one to the first variable, subtracts two from the second (mutating both) and returns
      # the sum of their values.
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)
    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)

    # The following remote call would fail because the ps nodes cannot see each
    # other due to the device filters.
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:my_ps/task:0/device:CPU:0'):
        worker_fn().numpy()
    self.assertIn('/job:my_ps/replica:0/task:1/device:CPU:0 unknown device',
                  cm.exception.message)

    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:my_ps/task:1/device:CPU:0'):
        worker_fn().numpy()
    self.assertIn('/job:my_ps/replica:0/task:0/device:CPU:0 unknown device',
                  cm.exception.message)

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 7)
    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 6)
    # Explicitly delete variables to avoid triggering errors when being GC'ed in
    # subsequent tests.
    del v1, v2

  def testConnectWithClusterResolver(self):
    # [FUNCTION DOC]
    # Scenario: connect using the resolver (master is the first ps, so its device becomes the
    # default); create two variables (zero and ten) on the default device, and run an
    # increment/decrement function from worker task 0 (expects nine) and worker task 1 (eight).
    remote.connect_to_cluster(self._cluster_resolver)

    v1 = variables.Variable(initial_value=0)
    v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # Adds one to the first variable, subtracts two from the second (mutating both) and returns
      # the sum of their values.
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)

    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)

  def testConnectToClusterTwiceOk(self):
    # [FUNCTION DOC]
    # Scenario: connecting to the same cluster through the same resolver twice in a row must not
    # raise; nothing else is asserted.
    remote.connect_to_cluster(self._cluster_resolver)
    remote.connect_to_cluster(self._cluster_resolver)

  def testConnectToClusterOnMismatchedDevice(self):
    # [FUNCTION DOC]
    # Scenario: connect with the resolver, then manually enter a different device scope (a
    # worker CPU, never exited here; teardown clears it); connecting again must raise a value
    # error because the current scope differs from the master device scope.
    remote.connect_to_cluster(self._cluster_resolver)

    # enter into another device scope.
    ops.device('/job:my_worker/task:0/device:CPU:0').__enter__()

    with self.assertRaises(ValueError):
      remote.connect_to_cluster(self._cluster_resolver)

  def testConnectToClusterWithLocalMaster(self):
    # [FUNCTION DOC]
    # Scenario: a resolver with an empty cluster spec and master "local" is passed to connect;
    # the call must return silently without raising (it is a no-op for a local master).
    local_resolver = SimpleClusterResolver(ClusterSpec({}), master='local')
    remote.connect_to_cluster(local_resolver)

  def testConnectToClusterInGraphModeWillFail(self):
    # [FUNCTION DOC]
    # Scenario: disable eager execution, assert connecting to the cluster raises a value error,
    # then re-enable eager execution (skipped if the assertion fails, since it is not in a
    # cleanup block).
    ops.disable_eager_execution()
    with self.assertRaises(ValueError):
      remote.connect_to_cluster(self._cluster_resolver)
    ops.enable_eager_execution()

  def testConnectToClusterWithoutLocalGpu(self):
    # [FUNCTION DOC]
    # Scenario: hide all local GPUs (only remote workers have GPUs), connect with the resolver
    # but without making the master device default, and assert no default device name is set.
    # Then create a variable, increment it and assert its value is one.
    # Only remote workers have GPU devices
    context.context().set_visible_devices([], 'GPU')
    # Ensure that no default device is set in eager context
    remote.connect_to_cluster(self._cluster_resolver,
                              make_master_device_default=False)
    self.assertEmpty(context.get_device_name())

    v1 = variables.Variable(initial_value=0)
    v1.assign_add(1)
    self.assertAllEqual(v1.read_value(), 1)

  # TODO(b/249134783): Add a test for task failures by introducing an Op for
  # reporting errors.
  def testGetTaskStatesAllOK(self):
    # [FUNCTION DOC]
    # Scenario: configure a standalone coordination service led by ps task 0, connect, initialize
    # the context, and query task states for two workers and two ps tasks. Asserts four states
    # come back and each is null (no error reported, i.e. all tasks healthy).
    context.context().configure_coordination_service(
        service_type='standalone', service_leader='/job:my_ps/replica:0/task:0')
    remote.connect_to_cluster(self._cluster)
    context.context().ensure_initialized()

    states = context.context().get_task_states([('my_worker', 2), ('my_ps', 2)])
    self.assertLen(states, 4)
    for state in states:
      self.assertIsNone(state)


def _strip_prefix(s, prefix):
  # [FUNCTION DOC]
  # HELPER: used by the multi-job test fixture to turn target addresses into plain host-port
  # strings. Returns s without the leading prefix when present, otherwise s unchanged.
  return s[len(prefix):] if s.startswith(prefix) else s


if __name__ == '__main__':
  test.main()
