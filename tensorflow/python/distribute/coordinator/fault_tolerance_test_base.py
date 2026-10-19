# [MODULE DOC]
# HELPER: shared base for the parameter-server fault tolerance tests (concrete test modules mix it in).
# Defines a small trainable Model used with a cluster coordinator, and BaseFaultToleranceTest, a
# mixin of test methods that kill and restart worker and parameter-server tasks of a multi-process
# cluster while closures are scheduled, then check that the coordinator recovers or raises the
# expected error types. Defines only module constants (RPC error markers, thread names); no import
# time registration or global side effects beyond that.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Fault tolerance test base class for parameter server training in TF2."""

import gc
import os
import sys
import threading
import time

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator as thread_coordinator
from tensorflow.python.training import server_lib


_RPC_ERROR_FROM_WORKER = "GRPC error information from remote target /job:worker"
_RPC_ERROR_FROM_PS = "GRPC error information from remote target /job:ps"
_WORKER_PREEMPTION_THREAD_NAME = "WorkerPreemptionHandler"
_WORKER_THREAD_PREFIX = "WorkerClosureProcessingLoop"


class Model(object):
  # [CLASS DOC]
  # HELPER: small training model used by the fault tolerance tests.
  # Holds a cluster coordinator, its strategy, a weight matrix variable, an iteration counter, a
  # flag variable that makes a training step loop forever, and two per-worker dataset iterators.
  # Built under the strategy scope; used to schedule training closures and join them.

  def __init__(self, coordinator):
    # [FUNCTION DOC]
    # Stores the given cluster coordinator and its strategy, then builds the model variables and
    # iterators inside the strategy scope. Returns nothing.
    self.cluster_coord = coordinator
    self.strategy = self.cluster_coord.strategy
    with self.cluster_coord.strategy.scope():
      self.build()

  def build(self):
    # [FUNCTION DOC]
    # Creates a random 10 by 10 float weight variable, an integer iteration counter starting at
    # zero, and a boolean flag (initially false) that makes the training function run forever, then
    # creates the dataset iterators. Returns nothing; sets attributes on the model.
    self.w = variables.Variable(
        initial_value=random_ops.random_uniform((10, 10)), dtype=dtypes.float32)
    self.iterations = variables.Variable(initial_value=0, dtype=dtypes.int32)
    # Allow external control to make the model run its train_fn in an infinite
    # loop. This allows us to reliably test worker preemption in the middle of
    # function execution.
    self.do_infinite_step = variables.Variable(False)

    self.rebuild_iterators()

  def rebuild_iterators(self, use_dataset_fn=True):
    # [FUNCTION DOC]
    # Recreates the two per-worker dataset iterators. Parameter use_dataset_fn (boolean, default
    # true): when true the per-worker datasets are built from a function that distributes a dataset
    # of repeated random 10 by 10 tensors through the strategy; when false a dataset object is
    # created directly and handed to the coordinator. Returns nothing; replaces both iterators.
    if use_dataset_fn:

      def dataset_fn():
        # [FUNCTION DOC]
        # HELPER: dataset factory for rebuild_iterators.
        # Returns an endlessly repeating dataset whose single element is a random 10 by 10 tensor.
        data = random_ops.random_uniform((10, 10))
        dataset = dataset_ops.DatasetV2.from_tensors([data]).repeat()
        return dataset

      def distribute_dataset_fn():
        # [FUNCTION DOC]
        # HELPER: per-worker dataset function for rebuild_iterators.
        # Returns a distributed dataset created from a function (ignoring its input context) that
        # calls the dataset factory.
        return self.cluster_coord.strategy.distribute_datasets_from_function(
            lambda _: dataset_fn())

      self.iterator = iter(
          self.cluster_coord.create_per_worker_dataset(distribute_dataset_fn))
      self.iterator2 = iter(
          self.cluster_coord.create_per_worker_dataset(distribute_dataset_fn))
    else:
      data = random_ops.random_uniform((10, 10))
      dataset = dataset_ops.DatasetV2.from_tensors([data]).repeat()

      self.iterator = iter(
          self.cluster_coord.create_per_worker_dataset(dataset))
      self.iterator2 = iter(
          self.cluster_coord.create_per_worker_dataset(dataset))

  def _train_fn_internal(self, iterator, iterator2):
    # [FUNCTION DOC]
    # HELPER: one training step body shared by the training function.
    # Takes the next element from each of two iterators, multiplies them with the weight and a
    # random matrix, and adds the product to the weight variable. Returns nothing.
    x = math_ops.matmul(array_ops.squeeze(next(iterator)), self.w)
    x = math_ops.matmul(array_ops.squeeze(next(iterator2)), x)
    x = math_ops.matmul(random_ops.random_uniform((10, 10)), x)
    self.w.assign_add(x)

  @def_function.function
  def train_fn(self, iterator, iterator2):
    # [FUNCTION DOC]
    # Compiled function executed on workers. Runs one training step, keeps running steps while the
    # infinite-step flag is true, then increments the iteration counter. Parameters: two dataset
    # iterators. Returns nothing.
    self._train_fn_internal(iterator, iterator2)
    while self.do_infinite_step:
      self._train_fn_internal(iterator, iterator2)
    self.iterations.assign_add(1)

  def schedule_training_functions(self, num_steps):
    # [FUNCTION DOC]
    # Schedules the training function num_steps times (integer) on the cluster coordinator under
    # the strategy scope, passing both iterators. Returns nothing.
    with self.strategy.scope():
      for _ in range(num_steps):
        self.cluster_coord.schedule(
            self.train_fn, args=(self.iterator, self.iterator2))

  def join_training_functions(self):
    # [FUNCTION DOC]
    # Turns the infinite-step flag off and then blocks until all scheduled closures finish; errors
    # recorded by the coordinator are raised from the join. Returns nothing.
    self.do_infinite_step.assign(False)
    self.cluster_coord.join()


class BaseFaultToleranceTest(object):  # pylint: disable=missing-docstring
  # [CLASS DOC]
  # Mixin holding the fault tolerance test cases for parameter server training. A concrete test
  # class must also derive from a test case type and call setUp with worker and parameter-server
  # counts. Fixtures: a multi-process cluster, a chief address, a parameter server strategy, a
  # cluster coordinator and a thread coordinator. Tests kill and restart tasks and assert on the
  # resulting errors, recovery and thread cleanup.

  def setUp(self, num_workers, num_ps, use_cs=False):
    # [FUNCTION DOC]
    # Creates a multi-process gRPC cluster with the given numbers of workers and parameter servers
    # (integers), adds a chief task on an unused local port, and builds a cluster resolver, a
    # parameter server strategy, a cluster coordinator and a thread coordinator. If use_cs
    # (boolean, default false) is true, sets the environment variable that enables the
    # coordination service for parameter servers (side effect on the process environment, never
    # reset here). Stores the worker and ps counts. Returns nothing.
    super(BaseFaultToleranceTest, self).setUp()

    self._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=num_workers,
        num_ps=num_ps,
        rpc_layer="grpc",
        stream_output=True,
    )
    self._cluster_def = self._cluster.cluster_resolver.cluster_spec().as_dict()
    self._cluster_def["chief"] = [
        "localhost:%d" % multi_worker_test_base.pick_unused_port()
    ]
    cluster_resolver = SimpleClusterResolver(
        server_lib.ClusterSpec(self._cluster_def), rpc_layer="grpc"
    )

    if use_cs:
      os.environ["TF_PSS_ENABLE_COORDINATION_SERVICE"] = "1"
    # The strategy's constructor would connect to the cluster.
    self.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        cluster_resolver
    )
    self.cluster_coord = cluster_coordinator.ClusterCoordinator(self.strategy)

    self.thread_coord = thread_coordinator.Coordinator(
        clean_stop_exception_types=[]
    )
    self.num_workers = num_workers
    self.num_ps = num_ps

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent tearDown, stops the multi-process cluster and drops the reference to it.
    # Returns nothing.
    super(BaseFaultToleranceTest, self).tearDown()
    self._cluster.stop()
    self._cluster = None

  def _restart(self, downtime_secs, job):
    # [FUNCTION DOC]
    # HELPER: kills and restarts a task for the failure tests.
    # Kills task 0 of the given job (string), sleeps for downtime_secs, asserts the task is not
    # alive, restarts it and polls once per second until it reports alive. Returns nothing.
    """Kills `job` (index: 0) and restarts it after `downtime_secs`.

    Args:
      downtime_secs: secs before restarting the job.
      job: a string specifying the job to restart.
    """
    self._cluster.kill_task(job, 0)
    time.sleep(downtime_secs)
    self.assertFalse(context.check_alive("/job:%s/replica:0/task:0" % job))
    self._cluster.start_task(job, 0)
    while not context.check_alive("/job:%s/replica:0/task:0" % job):
      time.sleep(1)

  def _restart_in_thread(self, downtime_secs, restart_job):
    # [FUNCTION DOC]
    # HELPER: runs the restart helper in a background thread.
    # Starts a thread that restarts task 0 of restart_job after downtime_secs, reporting any
    # exception to the thread coordinator. Returns the started thread object.

    def _restart_fn():
      # [FUNCTION DOC]
      # HELPER: thread body for the restart helper. Restarts the task, routing any exception to the
      # thread coordinator so it stops cleanly. Returns nothing.
      with self.thread_coord.stop_on_exception():
        self._restart(downtime_secs, restart_job)

    restart_thread = threading.Thread(target=_restart_fn)
    restart_thread.start()
    return restart_thread

  def _ensure_threads_closed(self):
    # [FUNCTION DOC]
    # HELPER: verifies the coordinator cleans up its threads.
    # Asserts that the worker closure threads and the preemption handler thread exist, optionally
    # prints the reference graph of the coordinator if it appears to leak, drops the coordinator
    # and strategy, runs garbage collection, waits a second, and asserts both kinds of threads are
    # gone. Side effect: sets the coordinator and strategy attributes to nothing.
    """Ensures worker and preemption threads are closed."""
    # Worker and preemption threads should exist before releasing
    # ClusterCoordinator.
    running_threads = test_util.get_running_threads()
    self.assertTrue(
        test_util.has_thread(_WORKER_THREAD_PREFIX, running_threads))
    self.assertIn(_WORKER_PREEMPTION_THREAD_NAME, running_threads)

    # Print object graph if ClusterCoordinator may leak.
    if sys.getrefcount(self.cluster_coord) > 2:
      try:
        test_util.show_backref(self.cluster_coord)
      except:  # pylint: disable=bare-except
        pass

    # Wait for threads to close.
    self.cluster_coord = None
    self.strategy = None
    gc.collect()
    time.sleep(1)

    # Verify thread names.
    running_threads = test_util.get_running_threads()
    self.assertNotIn(_WORKER_PREEMPTION_THREAD_NAME, running_threads)
    self.assertFalse(
        test_util.has_thread(_WORKER_THREAD_PREFIX, running_threads),
        "Worker thread is not stopped properly.")

  def _create_model_and_run_indefinitely(self):
    # [FUNCTION DOC]
    # HELPER: builds a Model, turns on infinite stepping, schedules ten training closures and waits
    # until as many closures are in flight as there are workers. Returns the model.
    model = Model(self.cluster_coord)
    model.do_infinite_step.assign(True)
    model.schedule_training_functions(10)
    # Model does infinite training step, so at this moment, we expect to have
    # `self.num_workers` infinite closures inflight, and `10-self.num_workers`
    # closures in the queue.
    while (self.cluster_coord._cluster.closure_queue._inflight_closure_count <
           self.num_workers):
      time.sleep(0.1)
    return model

  def testClusterCoordinatorDestroyed(self):
    # [FUNCTION DOC]
    # Test: right after setup, releasing the coordinator and strategy must stop all worker and
    # preemption threads (checked by the thread closing helper).
    self._ensure_threads_closed()

  def testWorkerPreemptionBetweenFunctions(self):
    # [FUNCTION DOC]
    # Test: trains two steps and checks the iteration count is two, restarts a worker for two
    # seconds while idle, trains two more steps and checks the count is four.
    model = Model(self.cluster_coord)
    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 2)

    self._restart(downtime_secs=2, job="worker")

    model.schedule_training_functions(2)
    model.join_training_functions()
    self.assertEqual(model.iterations.numpy(), 4)

  def testWorkerPreemptionMidstFunction(self):
    # [FUNCTION DOC]
    # Test: with infinite steps enabled and four closures scheduled, waits until each worker has an
    # in-flight closure, asserts the coordinator is not done, restarts a worker, joins, and checks
    # the iteration count is at least four.
    model = Model(self.cluster_coord)
    model.do_infinite_step.assign(True)

    model.schedule_training_functions(4)
    # Model does infinite training step, so at this moment, we expect to have
    # `self.num_workers` infinite closures inflight, and `4-self.num_workers`
    # closures in the queue.
    while (self.cluster_coord._cluster.closure_queue._inflight_closure_count <
           self.num_workers):
      time.sleep(0.1)
    self.assertFalse(self.cluster_coord.done())
    self._restart(downtime_secs=2, job="worker")
    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 4)

  def testOneWorkerPreemptionWithCancellation(self):
    # [FUNCTION DOC]
    # Test: schedules three normal closures, one long-running closure and one closure that fails an
    # assertion, restarts a worker, then expects join to raise an invalid-argument error and
    # fetching the long closure result to raise a cancelled error. Schedules three more normal
    # closures, joins, then stops the failure handler and waits for its thread to exit.

    @def_function.function
    def normal_function():
      # [FUNCTION DOC]
      # Compiled function returning the mean of the product of two small random matrices.
      x = random_ops.random_uniform((2, 10))
      y = random_ops.random_uniform((10, 2))
      return math_ops.reduce_mean(math_ops.matmul(x, y))

    @def_function.function
    def error_function():
      # [FUNCTION DOC]
      # Compiled function that asserts the sum of a random matrix product is non-positive (which
      # normally fails with an invalid-argument error) and returns the first random matrix.
      x = random_ops.random_uniform((2, 10))
      y = random_ops.random_uniform((10, 2))
      check_ops.assert_non_positive_v2(
          math_ops.reduce_sum(math_ops.matmul(x, y)))
      return x

    @def_function.function
    def long_function():
      # [FUNCTION DOC]
      # Compiled function running a long loop of ten thousand large random matrix products so it
      # is still running when the worker is killed. Returns the accumulated matrix.
      x = random_ops.random_uniform((1000, 1000))
      for _ in math_ops.range(10000):
        a = random_ops.random_uniform((1000, 1000))
        b = random_ops.random_uniform((1000, 1000))
        x += math_ops.matmul(a, b)
      return x

    for _ in range(3):
      self.cluster_coord.schedule(normal_function)
    long_function_result = self.cluster_coord.schedule(long_function)
    self.cluster_coord.schedule(error_function)

    time.sleep(1)  # Let it run a couple steps.
    self._restart(2, "worker")

    # InvalidArgumentError thrown from the error_function.
    with self.assertRaises(errors.InvalidArgumentError):
      self.cluster_coord.join()

    # CancelledError thrown by ClusterCoordinator after cancelling due to user
    # error.
    with self.assertRaises(errors.CancelledError):
      long_function_result.fetch()

    for _ in range(3):
      self.cluster_coord.schedule(normal_function)
    self.cluster_coord.join()

    # The cluster is likely still being recovered since `join` returned early
    # due to the error_function.
    failure_handler = self.cluster_coord._cluster.failure_handler
    failure_handler.stop()
    failure_handler._preemption_handler_thread.join()

  def testHandleDatasetCreationFailureWithDatasetFn(self):
    # [FUNCTION DOC]
    # Test: restarts a worker in a background thread while training steps are scheduled and the
    # iterators (built from a dataset function) are rebuilt twice. Joins everything and checks the
    # iteration count is at least three.
    model = Model(self.cluster_coord)

    restart_thread = self._restart_in_thread(5, "worker")

    model.schedule_training_functions(3)
    model.rebuild_iterators()
    model.schedule_training_functions(3)
    model.rebuild_iterators()
    model.schedule_training_functions(3)

    model.join_training_functions()

    self.thread_coord.join([restart_thread])
    self.assertGreaterEqual(model.iterations.numpy(), 3)

  # TODO(yuefengz): consider using combinations when there is more code
  # duplication.
  def testHandleDatasetCreationFailureWithDataset(self):
    # [FUNCTION DOC]
    # Test: same scenario as the dataset-function variant, but the iterators are rebuilt from a
    # dataset object instead of a function. Checks the iteration count is at least three.
    model = Model(self.cluster_coord)

    restart_thread = self._restart_in_thread(5, "worker")

    model.schedule_training_functions(3)
    model.rebuild_iterators(use_dataset_fn=False)
    model.schedule_training_functions(3)
    model.rebuild_iterators(use_dataset_fn=False)
    model.schedule_training_functions(3)

    model.join_training_functions()

    self.thread_coord.join([restart_thread])
    self.assertGreaterEqual(model.iterations.numpy(), 3)

  def testWorkerPreemptionErrorType(self):
    # [FUNCTION DOC]
    # Test: runs compiled functions directly on worker 0 from a thread, restarts that worker, and if
    # the thread fails with an unavailable or aborted error checks the message names the worker
    # (not the parameter server) and matches a known connection failure text.

    @def_function.function
    def worker_train_fn():
      # [FUNCTION DOC]
      # Compiled function returning the mean of the product of two small random matrices.
      x = random_ops.random_uniform((2, 10))
      y = random_ops.random_uniform((10, 2))
      return math_ops.reduce_mean(math_ops.matmul(x, y))

    def run_fn():
      # [FUNCTION DOC]
      # Thread body: on worker 0, runs the worker function three times in three rounds with five
      # second pauses, reporting exceptions to the thread coordinator.
      with self.thread_coord.stop_on_exception():
        with ops.device("/job:worker/replica:0/task:0"):
          for _ in range(3):
            for _ in range(3):
              worker_train_fn()
            time.sleep(5)

    run_thread = threading.Thread(target=run_fn)
    run_thread.start()
    time.sleep(1)  # Let it run a couple steps.
    self._restart(2, "worker")

    try:
      self.thread_coord.join([run_thread])
    except (errors.UnavailableError, errors.AbortedError) as e:
      logging.info("Got exception %r, error message is %s", e, e)

      self.assertIn(_RPC_ERROR_FROM_WORKER, str(e))  # pylint: disable=g-assert-in-except
      self.assertNotIn(_RPC_ERROR_FROM_PS, str(e))

      self.assertTrue("failed to connect to all addresses" in str(e) or
                      "Unable to find a context_id" in str(e) or
                      "Socket closed" in str(e) or
                      "Connection reset by peer" in str(e) or
                      "Transport closed" in str(e))

  def testWorkerPreemptionErrorTypeWithPythonFunction(self):
    # [FUNCTION DOC]
    # Test: same as the worker preemption error type test, but the function run on the worker is a
    # plain Python function instead of a compiled one.

    def worker_train_fn():
      # [FUNCTION DOC]
      # Plain Python function returning the mean of the product of two small random matrices.
      x = random_ops.random_uniform((2, 10))
      y = random_ops.random_uniform((10, 2))
      return math_ops.reduce_mean(math_ops.matmul(x, y))

    def run_fn():
      # [FUNCTION DOC]
      # Thread body: on worker 0, runs the worker function three times in three rounds with five
      # second pauses, reporting exceptions to the thread coordinator.
      with self.thread_coord.stop_on_exception():
        with ops.device("/job:worker/replica:0/task:0"):
          for _ in range(3):
            for _ in range(3):
              worker_train_fn()
            time.sleep(5)

    run_thread = threading.Thread(target=run_fn)
    run_thread.start()
    time.sleep(1)  # Let it run a couple steps.
    self._restart(2, "worker")

    try:
      self.thread_coord.join([run_thread])
    except (errors.UnavailableError, errors.AbortedError) as e:
      logging.info("Got exception %r, error message is %s", e, e)

      self.assertIn(_RPC_ERROR_FROM_WORKER, str(e))  # pylint: disable=g-assert-in-except
      self.assertNotIn(_RPC_ERROR_FROM_PS, str(e))

      self.assertTrue("failed to connect to all addresses" in str(e) or
                      "Unable to find a context_id" in str(e) or
                      "Socket closed" in str(e) or
                      "Connection reset by peer" in str(e) or
                      "Transport closed" in str(e))

  def testPSPreemptionErrorType(self):
    # [FUNCTION DOC]
    # Test: places a variable on parameter server 0, runs a function on worker 0 using it from a
    # thread, restarts the parameter server after a short delay, and if the thread fails checks the
    # error names the parameter server, matches known messages for unavailable or aborted errors,
    # then verifies threads are closed.

    with ops.device("/job:ps/replica:0/task:0"):
      v = variables.Variable(
          initial_value=random_ops.random_uniform((2, 10)),
          dtype=dtypes.float32)

    @def_function.function
    def worker_train_fn():
      # [FUNCTION DOC]
      # Compiled function returning the mean of the product of the parameter-server variable and a
      # random matrix.
      y = random_ops.random_uniform((10, 2))
      return math_ops.reduce_mean(math_ops.matmul(v, y))

    def run_fn():
      # [FUNCTION DOC]
      # Thread body: on worker 0, runs the worker function three times in three rounds with five
      # second pauses, reporting exceptions to the thread coordinator.
      with self.thread_coord.stop_on_exception():
        with ops.device("/job:worker/replica:0/task:0"):
          for _ in range(3):
            for _ in range(3):
              worker_train_fn()
            time.sleep(5)

    run_thread = threading.Thread(target=run_fn)
    run_thread.start()
    time.sleep(1)  # Let it run a couple steps.

    # Use a short restart delay to cover the case that RPC channel is reused
    self._restart(1, "ps")

    try:
      self.thread_coord.join([run_thread])
    except (errors.UnavailableError, errors.AbortedError) as e:
      logging.info("Got exception %r, error message is %s", e, e)
      self.assertIn(_RPC_ERROR_FROM_PS, str(e))  # pylint: disable=g-assert-in-except

      if isinstance(e, errors.UnavailableError):
        self.assertTrue("failed to connect to all addresses" in str(e) or
                        "Socket closed" in str(e) or
                        "Connection reset by peer" in str(e) or
                        "Transport closed" in str(e))

      if isinstance(e, errors.AbortedError):
        self.assertTrue(
            "RecvTensor expects a different device incarnation" in str(e) or
            "Unable to find a context_id" in str(e))
      self._ensure_threads_closed()

  def testTwoWorkersPreempted(self):
    # [FUNCTION DOC]
    # Test: skipped with fewer than two workers. Runs infinite closures, kills workers 0 and 1,
    # checks they are dead, restarts both, checks they are alive, joins and checks at least ten
    # iterations completed.
    if self.num_workers < 2:
      self.skipTest("Worker number is less than 2.")
    model = self._create_model_and_run_indefinitely()

    self.assertFalse(self.cluster_coord.done())
    self._cluster.kill_task("worker", 0)
    self._cluster.kill_task("worker", 1)
    time.sleep(2)
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:1"))
    self._cluster.start_task("worker", 0)
    self._cluster.start_task("worker", 1)
    time.sleep(2)
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:1"))

    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 10)

  def testWorkerContinuousFailure(self):
    # [FUNCTION DOC]
    # Test: with infinite closures running, kills and restarts worker 0 twice in a row, checking
    # liveness after each step, then joins and checks at least ten iterations completed.
    model = self._create_model_and_run_indefinitely()

    self.assertFalse(self.cluster_coord.done())
    self._cluster.kill_task("worker", 0)
    time.sleep(2)
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))
    self._cluster.start_task("worker", 0)
    time.sleep(2)
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))
    self._cluster.kill_task("worker", 0)
    time.sleep(2)
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))
    self._cluster.start_task("worker", 0)
    time.sleep(2)
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))

    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 10)

  def testPSFailureWhileRecoveryFromWokerFailure(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name misspells Worker, and the nested kill helper calls a sleep method that
    # the test class does not appear to define (it should use the time module), so those threads
    # would raise an attribute error after killing the task and never restart it.
    # Test: with infinite closures running, kills a worker and a parameter server concurrently in
    # two threads and expects join to raise an unavailable or invalid-argument error.
    model = self._create_model_and_run_indefinitely()

    time.sleep(1)
    self.assertFalse(self.cluster_coord.done())

    def kill(task):
      # [FUNCTION DOC]
      # KNOWN ISSUE: calls a sleep method on the test object that is likely undefined; the task is
      # therefore probably never restarted.
      # HELPER: thread body that kills task 0 of the given job name, sleeps, and restarts it.
      self._cluster.kill_task(task, 0)
      self.sleep(1)
      self._cluster.start_task(task, 0)

    kill_thread_1 = threading.Thread(target=kill, args=("worker",))
    kill_thread_2 = threading.Thread(target=kill, args=("ps",))
    kill_thread_1.start()
    kill_thread_2.start()
    kill_thread_1.join()
    kill_thread_2.join()

    with self.assertRaises(
        (errors.UnavailableError, errors.InvalidArgumentError)):
      model.join_training_functions()

  def testNumpyFetchedAfterWorkerFailure(self):
    # [FUNCTION DOC]
    # Test: schedules a function returning two values derived from a variable, fetches the result
    # (expected pair of one and minus one), kills worker 0, and fetches again expecting the same.

    with self.strategy.scope():
      v = variables.Variable(initial_value=0, dtype=dtypes.int32)

    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # Compiled function returning the variable plus one and the variable minus one.
      return v + 1, v - 1

    remote_value = self.cluster_coord.schedule(worker_fn)
    # Attempt to fetch before killing worker task should succeed.
    self.assertEqual((1, -1), remote_value.fetch())
    self._cluster.kill_task("worker", 0)
    # So should attempt to fetch after killing worker task.
    self.assertEqual((1, -1), remote_value.fetch())

  def testTensorGotAfterWorkerFailure(self):
    # [FUNCTION DOC]
    # Test: schedules a function and gets its result. Checks the local copy is a tensor on the
    # chief CPU and the remote values are tensors on workers. Kills all workers, then checks the
    # local copy is still readable while reading the remote tensor raises an unavailable error
    # that mentions connection failure and a worker task.

    with self.strategy.scope():
      v = variables.Variable(initial_value=0, dtype=dtypes.int32)

    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # Compiled function returning the variable plus one and the variable minus one.
      return v + 1, v - 1

    remote_value = self.cluster_coord.schedule(worker_fn)

    # Attempt to fetch before killing worker task should succeed.
    fetched = remote_value.get()[0]
    self.assertIsInstance(fetched, tensor.Tensor)
    self.assertEqual(fetched.device, "/job:chief/replica:0/task:0/device:CPU:0")
    self.assertEqual((1, -1), remote_value.get())
    remote_value.get()[0].numpy()

    # As well as the remote tensors that point to worker0 or worker1.
    values = remote_value._values[0]
    self.assertIsInstance(values, tensor.Tensor)
    self.assertRegex(values.device,
                     "/job:worker/replica:0/task:[0-1]/device:CPU:0")
    self.assertEqual((1, -1), remote_value._values)
    remote_value._values[0].numpy()

    # Terminate the workers and wait a little so that they are indeed killed.
    for i in range(self.num_workers):
      self._cluster.kill_task("worker", i)
    time.sleep(5)

    # Attempt to fetch after killing worker tasks should succeed as well.
    remote_value.get()[0].numpy()
    self.assertEqual((1, -1), remote_value.get())

    # Attempting to copy the tensor from worker now should fail.
    with self.assertRaises(errors.UnavailableError) as cm:
      remote_value._values[0].numpy()
    self.assertIn("failed to connect to all addresses", cm.exception.message)
    self.assertIn("/job:worker/replica:0/task:", cm.exception.message)

  def testFetchFromPSAfterWorkerFailure(self):
    # [FUNCTION DOC]
    # Test: with an infinite step scheduled, a background thread kills and restarts worker 0 after
    # delays while the main thread keeps reading the iteration variable from the parameter server
    # in a loop (at most 200 rounds, stopping after a few rounds following the restart). Passes if
    # no flaky errors are raised.
    # Test for flaky failures when reading from a parameter server while a
    # worker is recovering.
    # Place some variables on PSes, kill a worker, and continuously poll one of
    # those variables.

    model = Model(self.cluster_coord)

    # kill the worker after a delay to make sure variable reading runs while
    # worker is up, while it's down, and while it restarts
    def kill_after_delay():
      # [FUNCTION DOC]
      # Thread body: waits three seconds, kills worker 0, waits a second, restarts it, logging each
      # step. Returns nothing.
      time.sleep(3)
      logging.info("Killing worker 0")
      self._cluster.kill_task("worker", 0)
      time.sleep(1)
      logging.info("Restarting worker 0")
      self._cluster.start_task("worker", 0)

    kill_thread = threading.Thread(target=kill_after_delay)
    kill_thread.start()

    model.do_infinite_step.assign(True)
    model.schedule_training_functions(1)

    num_reads = 0
    num_reads_after_restart = 0
    read_interval_secs = 0.1
    worker_has_stopped = False
    # limit runtime of the test: stop after doing a few reads after worker
    # is back up, or after a fixed maximum number of reads
    while num_reads_after_restart <= 5 and num_reads < 200:
      worker_up = context.check_alive("/job:worker/replica:0/task:0")
      if not worker_up:
        worker_has_stopped = True
      if worker_up and worker_has_stopped:
        num_reads_after_restart += 1

      model.join_training_functions()
      start = time.time()
      while time.time() < start + read_interval_secs:
        model.iterations.read_value()

      num_reads += 1
      # run another epoch
      model.do_infinite_step.assign(True)
      model.schedule_training_functions(1)

  def testClusterStateNotDisrupted(self):
    # [FUNCTION DOC]
    # Test: reruns the worker preemption midst-function test and the worker preemption error type
    # test with fresh thread coordinators to check that cluster state is not disrupted. Other
    # sub-tests are disabled as commented-out TODOs.
    # This test has side effects and can disrupt other tests, even if the
    # resource created by it will not be used in following tests.
    # TODO(b/155209534): enable this test.
    # self.testPSPreemptionErrorType()

    self.thread_coord = thread_coordinator.Coordinator(
        clean_stop_exception_types=[])
    self.testWorkerPreemptionMidstFunction()

    self.thread_coord = thread_coordinator.Coordinator(
        clean_stop_exception_types=[])
    self.testWorkerPreemptionErrorType()

    # In previous tests, workers may fail after training is done. But the
    # following tests start with creating resources where failure is not
    # handled.
    # TODO(b/153888707): enable the following two tests.
    # self.testTwoWorkersPreempted()
    # self.testWorkerContinuousFailure()

  def _run_and_kill_ps_task(self):
    # [FUNCTION DOC]
    # HELPER: runs the infinite model, kills parameter server 0 and waits until the closure queue
    # records an error. Returns nothing.
    self._create_model_and_run_indefinitely()
    self._cluster.kill_task("ps", 0)
    while self.cluster_coord._cluster.closure_queue._error is None:
      time.sleep(1)
    logging.info("Trying to join, expecting error")

  def testJoinRaisesUnavailableErrorAtPsFailure(self):
    # [FUNCTION DOC]
    # Test: after a parameter server failure, join must raise unavailable, not-found or
    # failed-precondition error.
    self._run_and_kill_ps_task()
    with self.assertRaises((errors.UnavailableError, errors.NotFoundError,
                            errors.FailedPreconditionError)):
      self.cluster_coord.join()

  def testScheduleRaisesUnavailableErrorAtPsFailure(self):
    # [FUNCTION DOC]
    # Test: after a parameter server failure, scheduling a new closure must raise unavailable,
    # not-found or failed-precondition error.
    self._run_and_kill_ps_task()
    with self.assertRaises((errors.UnavailableError, errors.NotFoundError,
                            errors.FailedPreconditionError)):
      self.cluster_coord.schedule(def_function.function(lambda: None))

  def testWorkerExecutionAfterPsFailureRaisesExpectedError(self):
    # [FUNCTION DOC]
    # Test: kills all parameter servers, waits for the error, then runs a trivial function on each
    # worker directly; each attempt must fail with a parameter server failure (checked by the
    # coordinator helper), otherwise an assertion error is raised.
    model = self._create_model_and_run_indefinitely()
    for i in range(self.num_ps):
      self._cluster.kill_task("ps", i)
    while self.cluster_coord._cluster.closure_queue._error is None:
      time.sleep(1)

    @def_function.function
    def trivial_function():
      # [FUNCTION DOC]
      # Compiled function returning the model iteration counter plus one.
      return model.iterations + 1

    for i in range(self.num_workers):
      try:
        with ops.device("/job:worker/replica:0/task:{}".format(i)):
          trivial_function()
      except Exception as e:  # pylint: disable=broad-except
        if cluster_coordinator._is_ps_failure(e):  # pylint: disable=protected-access
          if i < self.num_workers - 1:
            continue
          return
      raise AssertionError("Executing a function after PS fails, should "
                           "result in a PS failure.")

  def testAsyncWaitIsNoOp(self):
    # [FUNCTION DOC]
    # Test: skipped with fewer than two workers. Kills worker 0 while infinite closures run, checks
    # the global async wait does not raise despite the dead worker, joins, checks at least ten
    # iterations, and finally restarts the worker.
    if self.num_workers < 2:
      self.skipTest("Worker number is less than 2.")
    model = self._create_model_and_run_indefinitely()

    self.assertFalse(self.cluster_coord.done())
    self._cluster.kill_task("worker", 0)
    time.sleep(2)
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))
    # Should pass without exception even with failed remote workers
    context.async_wait()

    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 10)

    self._cluster.start_task("worker", 0)
