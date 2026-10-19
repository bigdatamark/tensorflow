# [MODULE DOC]
# Test module for the distribute coordinator. Provides mock strategy, extended-strategy and
# server objects plus a shared test base class that creates an in-process cluster of workers
# and parameter servers. Tests cover standalone client mode (in-graph and between-graph
# replication, worker context contents, strategy properties, chief and evaluator handling),
# independent worker mode (one coordinator thread per task, with real or mocked standard
# servers), the "google" environment and the RPC-layer environment setting, how strategies
# configure the session config for servers and sessions, and starting a server from TF_CONFIG.
# Import-time side effects: records the original system exit function and defines task-type
# and mode aliases. When run as a script it patches system exit to a hard exit, shortens the
# session manager recovery wait, and runs the tests.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Distribute Coordinator."""

import contextlib
import copy
import json
import os
import sys
import threading
import time

import six

# pylint: disable=g-import-not-at-top
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_manager


CHIEF = distribute_coordinator._TaskType.CHIEF
WORKER = distribute_coordinator._TaskType.WORKER
PS = distribute_coordinator._TaskType.PS
EVALUATOR = distribute_coordinator._TaskType.EVALUATOR

STANDALONE_CLIENT = distribute_coordinator.CoordinatorMode.STANDALONE_CLIENT
INDEPENDENT_WORKER = distribute_coordinator.CoordinatorMode.INDEPENDENT_WORKER

NUM_WORKERS = 3
NUM_PS = 2

original_sys_exit = sys.exit


def _bytes_to_str(maybe_bytes):
  # [FUNCTION DOC]
  # HELPER: test utility converting a server target that may be bytes into text.
  # Returns the input unchanged if it is already a string, otherwise decodes it as UTF-8.
  if isinstance(maybe_bytes, six.string_types):
    return maybe_bytes
  else:
    return str(maybe_bytes, "utf-8")


def _strip_protocol(target):
  # [FUNCTION DOC]
  # HELPER: test utility turning a target address into a bare host and port string, as cluster
  # specs expect. If the text contains a scheme separator, returns the part after it; otherwise
  # returns the input unchanged.
  # cluster_spec expects "host:port" strings.
  if "//" in target:
    return target.split("//")[1]
  else:
    return target


class MockExtended(object):
  # [CLASS DOC]
  # HELPER: test double for a strategy's extended object. A plain holder of the settings the
  # coordinator reads: whether replication is between-graph, whether to run init ops, whether
  # to checkpoint and whether to save summaries. The last three may be null until the mock
  # strategy's configure method fills them in.

  def __init__(self,
               between_graph=False,
               should_init=None,
               should_checkpoint=None,
               should_save_summary=None):
    # [FUNCTION DOC]
    # Constructor. Stores the four settings (between_graph default false; the other three
    # optional booleans default null, meaning "decide later") as public attributes. Returns
    # nothing.
    self.experimental_between_graph = between_graph
    self.experimental_should_init = should_init
    self.should_checkpoint = should_checkpoint
    self.should_save_summary = should_save_summary


class MockStrategy(object):
  # [CLASS DOC]
  # HELPER: test double for a distribution strategy. Holds a mock extended object and offers
  # a configure method that mimics real strategies: it fills unset flags based on whether the
  # task id is zero and edits a session configuration (thread counts and device filters) so
  # tests can see which configuration path was taken.

  def __init__(self,
               between_graph=False,
               should_init=None,
               should_checkpoint=None,
               should_save_summary=None):
    # [FUNCTION DOC]
    # Constructor. Takes the same four settings as the mock extended class and builds the
    # extended attribute from them. Returns nothing.
    self.extended = MockExtended(between_graph, should_init, should_checkpoint,
                                 should_save_summary)

  def configure(self,
                session_config=None,
                cluster_spec=None,
                task_type=None,
                task_id=None):
    # [FUNCTION DOC]
    # Mock of the strategy configure hook. Inputs (all optional): session_config (mutated),
    # cluster_spec, task_type, task_id. Any of should_init, should_checkpoint and
    # should_save_summary still null are set to true when the task id is zero, else false.
    # Then, if a session config is given: when a cluster spec, task type and task id are all
    # present and replication is between-graph, it increments the intra-op thread count and,
    # for chief or worker tasks, adds device filters for the task itself and the ps job;
    # otherwise it increments the inter-op thread count and adds a filter for a dummy job.
    # Returns nothing.
    if self.extended.experimental_should_init is None:
      if task_id == 0:
        self.extended.experimental_should_init = True
      else:
        self.extended.experimental_should_init = False
    if self.extended.should_checkpoint is None:
      if task_id == 0:
        self.extended.should_checkpoint = True
      else:
        self.extended.should_checkpoint = False
    if self.extended.should_save_summary is None:
      if task_id == 0:
        self.extended.should_save_summary = True
      else:
        self.extended.should_save_summary = False

    if session_config:
      if (cluster_spec and task_type and task_id is not None and
          self.extended.experimental_between_graph):
        session_config.intra_op_parallelism_threads += 1
        if task_type in ["chief", "worker"]:
          session_config.device_filters.extend(
              ["/job:%s/task:%d" % (task_type, task_id), "/job:ps"])
      else:
        session_config.inter_op_parallelism_threads += 1
        session_config.device_filters.append("/job:somejob")


class MockServer(object):
  # [CLASS DOC]
  # HELPER: test double for a TensorFlow server. Records whether start and join were called,
  # without any networking; join may be called at most once.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Marks the server as neither joined nor started. Returns nothing.
    self._joined = False
    self._started = False

  def start(self):
    # [FUNCTION DOC]
    # Marks the server as started. Returns nothing.
    self._started = True

  def join(self):
    # [FUNCTION DOC]
    # Marks the server as joined without blocking. Fails an assertion if it was already
    # joined. Returns nothing.
    assert not self._joined
    self._joined = True

  @property
  def joined(self):
    # [FUNCTION DOC]
    # Property. True once join has been called.
    return self._joined

  @property
  def started(self):
    # [FUNCTION DOC]
    # Property. True once start has been called.
    return self._started


class DistributeCoordinatorTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the coordinator tests. Creates (once per class) a global
  # in-process cluster of workers and parameter servers, and per test sets up counters, a
  # lock, dictionaries recording worker contexts, strategy properties and mock std servers, a
  # barrier and a thread coordinator. Provides worker functions and utilities to run the
  # coordinator in threads. Has no tests itself.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture. Creates a local in-process cluster with the configured number of
    # workers and parameter servers (it cannot be terminated once created) and stores the
    # workers, the parameter servers and a cluster spec mapping worker and ps jobs to their
    # host and port strings (protocol stripped). Returns nothing.
    # We have to create a global in-process cluster because once an in-process
    # tensorflow server is created, there is no way to terminate it. Please see
    # multi_worker_test_base.py for more details.
    # TODO(yuefengz): use the utitliy from multi_worker_test_base.
    cls._workers, cls._ps = test_util.create_local_cluster(
        NUM_WORKERS, num_ps=NUM_PS)
    cls._cluster_spec = {
        WORKER: [
            _strip_protocol(_bytes_to_str(w.target)) for w in cls._workers
        ],
        PS: [_strip_protocol(_bytes_to_str(ps.target)) for ps in cls._ps]
    }

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture. Resets the correct-result counter, creates a lock, empty dictionaries
    # for recorded worker contexts, strategy properties and mock std servers, a barrier sized
    # to the worker count and a fresh thread coordinator. Returns nothing.
    self._result_correct = 0
    self._lock = threading.Lock()
    self._worker_context = {}
    self._strategy_property = {}
    self._std_servers = {}
    self._barrier = distribute_coordinator._Barrier(NUM_WORKERS)
    self._coord = coordinator.Coordinator()

  @contextlib.contextmanager
  def _test_session(self, target):
    # [FUNCTION DOC]
    # HELPER: context manager yielding a session connected to the given target. Builds a
    # configuration with soft placement and graph optimization disabled, opens a session on
    # the default graph with that target, yields it, and closes it on exit.
    config = config_pb2.ConfigProto(allow_soft_placement=True)
    config.graph_options.optimizer_options.opt_level = -1
    with session.Session(graph=None, config=config, target=target) as sess:
      yield sess

  # TODO(yuefengz): use the utitliy from multi_worker_test_base.
  def _create_cluster_spec(self,
                           has_chief=False,
                           num_workers=1,
                           num_ps=0,
                           has_eval=False):
    # [FUNCTION DOC]
    # HELPER: builds a cluster spec mapping of local addresses on freshly picked unused ports.
    # Inputs: has_chief (add one chief, default false), num_workers (default 1), num_ps
    # (default 0), has_eval (add one evaluator, default false). Returns the mapping from job
    # names to lists of host and port strings.
    cluster_spec = {}
    if has_chief:
      cluster_spec[CHIEF] = ["localhost:%s" % test_util.pick_unused_port()]
    if num_workers:
      cluster_spec[WORKER] = [
          "localhost:%s" % test_util.pick_unused_port()
          for _ in range(num_workers)
      ]
    if num_ps:
      cluster_spec[PS] = [
          "localhost:%s" % test_util.pick_unused_port() for _ in range(num_ps)
      ]
    if has_eval:
      cluster_spec[EVALUATOR] = ["localhost:%s" % test_util.pick_unused_port()]
    return cluster_spec

  def _in_graph_worker_fn(self, strategy):
    # [FUNCTION DOC]
    # HELPER: worker function for in-graph tests. Asserts a worker context is active, opens a
    # session on its master target, creates one variable per worker pinned to that worker's
    # device and increments each by its index, sums them on worker 0, initializes variables,
    # runs the sum and asserts it equals the expected total. Increments the correct-result
    # counter when it matches. The strategy argument is unused.
    context = distribute_coordinator_context.get_current_worker_context()
    self.assertTrue(context is not None)
    with self._test_session(target=context.master_target) as sess:
      xs = []
      expected = 0.0
      for i in range(context.num_workers):
        with ops.device("/job:worker/task:%d" % i):
          x = variable_scope.get_variable("x_%d" % i, initializer=10.0)
          x_add = x.assign_add(float(i))
          xs.append(x_add)
          expected += i + 10.0

      with ops.device("/job:worker/task:0"):
        result = math_ops.add_n(xs)

      self.evaluate(variables.global_variables_initializer())
      result_value = sess.run(result)
    self.assertEqual(result_value, expected)
    if result_value == expected:
      self._result_correct += 1

  def _wrapped_worker_fn(self, worker_fn):
    # [FUNCTION DOC]
    # HELPER: wraps a worker function so errors it raises stop the shared thread coordinator.
    # Input: worker_fn callable. Returns a new callable that runs worker_fn under the
    # coordinator's stop-on-exception scope and returns its result.
    def wrapped(*args, **kwargs):
      # [FUNCTION DOC]
      # Runs the wrapped worker function with the given arguments inside the coordinator's
      # stop-on-exception scope and returns its result.
      with self._coord.stop_on_exception():
        return worker_fn(*args, **kwargs)
    return wrapped

  def _run_coordinator_in_thread(self, worker_fn, strategy, **kwargs):
    # [FUNCTION DOC]
    # HELPER: starts a thread that runs the distribute coordinator with the wrapped worker
    # function and the strategy, passing extra keyword arguments through. Returns the started
    # thread.
    t = threading.Thread(
        target=distribute_coordinator.run_distribute_coordinator,
        args=(self._wrapped_worker_fn(worker_fn), strategy),
        kwargs=kwargs)
    t.start()
    return t

  def _run_multiple_coordinator_in_threads(self, worker_fn, strategy,
                                           cluster_spec, **kwargs):
    # [FUNCTION DOC]
    # HELPER: simulates independent worker mode by starting one coordinator thread for every
    # task in the cluster spec, each told its own task type and id. Extra keyword arguments are
    # passed to every coordinator. Returns a mapping from task type to the list of started
    # threads indexed by task id.
    threads = {}
    for task_type in cluster_spec.keys():
      threads[task_type] = []
      for task_id in range(len(cluster_spec[task_type])):
        t = self._run_coordinator_in_thread(
            worker_fn,
            strategy,
            cluster_spec=cluster_spec,
            task_type=task_type,
            task_id=task_id,
            **kwargs)
        threads[task_type].append(t)
    return threads

  def _join_threads(self, threads):
    # [FUNCTION DOC]
    # HELPER: joins the given threads through the coordinator, re-raising their errors. If an
    # unknown error says a gRPC server could not be started, the test is skipped; any other
    # error is re-raised.
    try:
      self._coord.join(threads)
    except errors.UnknownError as e:
      if "Could not start gRPC server" in e.message:
        self.skipTest("Cannot start std servers.")
      else:
        raise

  def _between_graph_worker_fn(self, strategy):
    # [FUNCTION DOC]
    # HELPER: worker function for between-graph tests. Asserts a worker context exists, opens a
    # session on its master target, creates two resource variables on the two parameter
    # servers (10 and 20), builds an op adding 2 to the first and subtracting 2 from the
    # second. The chief initializes variables; workers then synchronize (via the context
    # barrier, or by polling for uninitialized variables). After running the op everyone
    # synchronizes again, reads both values and asserts they are 16 and 14, then increments the
    # correct-result counter under a lock. The strategy argument is unused.
    context = distribute_coordinator_context.get_current_worker_context()
    self.assertTrue(context is not None)
    with self._test_session(target=context.master_target) as sess:
      with ops.device("/job:ps/task:0"):
        # TODO(yuefengz): investigate why not using resource variable will make
        # the test flaky.
        x = variable_scope.get_variable(
            "x", initializer=10.0, use_resource=True)
      with ops.device("/job:ps/task:1"):
        y = variable_scope.get_variable(
            "y", initializer=20.0, use_resource=True)

      x_add = x.assign_add(2.0)
      y_sub = y.assign_sub(2.0)
      train_op = control_flow_ops.group([x_add, y_sub])

      if context.is_chief:
        self.evaluate(variables.global_variables_initializer())

      # Synchronize workers after initialization.
      if context.has_barrier:
        context.wait_for_other_workers()
      else:
        while True:
          uninit_vars = sess.run(variables.report_uninitialized_variables())
          # pylint: disable=g-explicit-length-test
          if len(uninit_vars) == 0:
            break

      sess.run(train_op)

      # Synchronize workers after one step to make sure they all have finished
      # training.
      if context.has_barrier:
        context.wait_for_other_workers()
      else:
        self._barrier.wait()

      x_val, y_val = sess.run([x, y])

      self.assertEqual(x_val, 16.0)
      self.assertEqual(y_val, 14.0)
      if x_val == 16.0 and y_val == 14.0:
        with self._lock:
          self._result_correct += 1

  def _between_graph_with_monitored_session(self, strategy):
    # [FUNCTION DOC]
    # HELPER: like the between-graph worker function but uses a monitored session, which runs
    # init or ready ops itself. Creates the two parameter-server variables, runs the add/subtract
    # op, synchronizes workers (context barrier or test barrier), reads the values, asserts
    # they are 16 and 14, and increments the correct-result counter under a lock. The strategy
    # argument is unused.
    context = distribute_coordinator_context.get_current_worker_context()
    self.assertTrue(context is not None)
    with ops.device("/job:ps/task:0"):
      # TODO(yuefengz): investigate why not using resource variable will make
      # the test flaky.
      x = variable_scope.get_variable("xx", initializer=10.0, use_resource=True)
    with ops.device("/job:ps/task:1"):
      y = variable_scope.get_variable("yy", initializer=20.0, use_resource=True)

    x_add = x.assign_add(2.0)
    y_sub = y.assign_sub(2.0)
    train_op = control_flow_ops.group([x_add, y_sub])

    # The monitored session will run init or ready ops.
    with monitored_session.MonitoredSession() as sess:
      sess.run(train_op)

      # Synchronize workers after one step to make sure they all have finished
      # training.
      if context.has_barrier:
        context.wait_for_other_workers()
      else:
        self._barrier.wait()

      x_val, y_val = sess.run([x, y])

    self.assertEqual(x_val, 16.0)
    self.assertEqual(y_val, 14.0)
    if x_val == 16.0 and y_val == 14.0:
      with self._lock:
        self._result_correct += 1

  def _dump_worker_context(self, strategy):
    # [FUNCTION DOC]
    # HELPER: worker function that records its worker context for later assertions. Asserts a
    # context is active, then under the lock stores, in the worker-context mapping keyed by the
    # task type text (null becomes the text "None") and indexed by task id (null becomes 0), a
    # tuple of master target, number of workers, is-chief and distributed-mode. The strategy
    # argument is unused.
    """Dumps the properties of each worker context.

    It dumps the context properties to a dict mapping from task_type to a list
    of tuples of master_target, num_workers, is_chief and distribute_mode, where
    the list is indexed by the task_id.

    Args:
      strategy: a `DistributionStrategy` object.
    """
    context = distribute_coordinator_context.get_current_worker_context()
    self.assertTrue(context is not None)
    task_type = str(context.task_type)
    task_id = context.task_id or 0
    with self._lock:
      if task_type not in self._worker_context:
        self._worker_context[task_type] = []
      while len(self._worker_context[task_type]) <= task_id:
        self._worker_context[task_type].append(None)
      self._worker_context[task_type][task_id] = (context.master_target,
                                                  context.num_workers,
                                                  context.is_chief,
                                                  context.distributed_mode)

  def _dump_strategy_property(self, strategy):
    # [FUNCTION DOC]
    # HELPER: worker function that records strategy-derived properties. Asserts a context is
    # active and that the context's strategy flags and its should-checkpoint and
    # should-save-summary properties equal those of the strategy passed in. Then under the lock
    # stores, keyed by task type text and indexed by task id, a tuple of should-init,
    # should-checkpoint and should-save-summary.
    context = distribute_coordinator_context.get_current_worker_context()
    self.assertTrue(context is not None)

    self.assertEqual(context._strategy.extended.experimental_should_init,
                     strategy.extended.experimental_should_init)
    self.assertEqual(context.should_checkpoint,
                     strategy.extended.should_checkpoint)
    self.assertEqual(context.should_save_summary,
                     strategy.extended.should_save_summary)

    task_type = str(context.task_type)
    task_id = context.task_id or 0
    with self._lock:
      if task_type not in self._strategy_property:
        self._strategy_property[task_type] = []
      while len(self._strategy_property[task_type]) <= task_id:
        self._strategy_property[task_type].append(None)
      self._strategy_property[task_type][task_id] = (
          context._strategy.extended.experimental_should_init,
          context.should_checkpoint,
          context.should_save_summary)

  def _run_mock_std_server(self,
                           session_config=None,
                           cluster_spec=None,
                           task_type=None,
                           task_id=None,
                           rpc_layer=None,
                           environment=None):
    # [FUNCTION DOC]
    # HELPER: replacement for the coordinator's standard-server runner, patched in by tests.
    # Accepts the same optional arguments (most ignored). Under the lock records a new mock
    # server in the std-server mapping keyed by task type text and indexed by task id (null
    # becomes 0) and returns it. The server is not started.
    task_type = str(task_type)
    task_id = task_id or 0
    with self._lock:
      if task_type not in self._std_servers:
        self._std_servers[task_type] = []
      while len(self._std_servers[task_type]) <= task_id:
        self._std_servers[task_type].append(None)

      server = MockServer()
      self._std_servers[task_type][task_id] = server
    return server


class DistributeCoordinatorTestStandaloneMode(DistributeCoordinatorTestBase):
  # [CLASS DOC]
  # Tests the coordinator in standalone client mode against the shared in-process cluster:
  # in-graph and between-graph training, monitored sessions, and the contents of worker
  # contexts and strategy properties for various cluster layouts.

  def testInGraphStandaloneMode(self):
    # [FUNCTION DOC]
    # Runs the in-graph worker function with an in-graph mock strategy against the cluster.
    # Expects the correct-result counter to be exactly one.
    """Test it runs in-graph replication in standalone client mode."""
    distribute_coordinator.run_distribute_coordinator(
        self._in_graph_worker_fn,
        MockStrategy(between_graph=False),
        cluster_spec=self._cluster_spec)
    self.assertEqual(self._result_correct, 1)

  def testBetweenGraph(self):
    # [FUNCTION DOC]
    # Runs the between-graph worker function with a between-graph mock strategy in standalone
    # mode. Expects every worker to report success, i.e. the counter equals the worker count.
    """Test it runs between-graph replication in standalone client mode."""
    distribute_coordinator.run_distribute_coordinator(
        self._between_graph_worker_fn,
        MockStrategy(between_graph=True),
        cluster_spec=self._cluster_spec)

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  @test_util.run_v1_only("MonitoredSession removed from v2")
  def testBetweenGraphWithMonitoredSession(self):
    # [FUNCTION DOC]
    # Runs the monitored-session worker function in standalone between-graph mode (graph-mode
    # only test). Expects the counter to equal the worker count.
    """Test monitored session in standalone client mode."""
    distribute_coordinator.run_distribute_coordinator(
        self._between_graph_with_monitored_session,
        MockStrategy(between_graph=True),
        cluster_spec=self._cluster_spec)

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  def testBetweenGraphContext(self):
    # [FUNCTION DOC]
    # Runs the context-dumping worker function between-graph in standalone mode. Expects one
    # task type (worker) with three tasks, each with its own server target as master target,
    # three workers, chief only for task 0, and distributed mode true.
    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        MockStrategy(between_graph=True),
        cluster_spec=self._cluster_spec)

    # There is only one type of task and there three such tasks.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue(WORKER in self._worker_context)
    self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(
        self._worker_context[WORKER][0],
        (_bytes_to_str(self._workers[0].target), NUM_WORKERS, True, True))
    self.assertEqual(
        self._worker_context[WORKER][1],
        (_bytes_to_str(self._workers[1].target), NUM_WORKERS, False, True))
    self.assertEqual(
        self._worker_context[WORKER][2],
        (_bytes_to_str(self._workers[2].target), NUM_WORKERS, False, True))

  def testBetweenGraphStrategyProperties(self):
    # [FUNCTION DOC]
    # Runs the strategy-property-dumping worker function between-graph in standalone mode with
    # should-init forced true. Expects three worker entries: all should-init true, and
    # checkpoint and summary flags true only for worker 0.
    # Dumps properties of the strategy objects.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_strategy_property,
        MockStrategy(between_graph=True, should_init=True),
        cluster_spec=self._cluster_spec)

    # There is only one type of task and there three such tasks.
    self.assertEqual(len(self._strategy_property), 1)
    self.assertTrue(WORKER in self._strategy_property)
    self.assertEqual(len(self._strategy_property[WORKER]), NUM_WORKERS)

    # Check whether each task has the right properties of should_init,
    # should_checkpoint and should_save_summary.
    self.assertEqual(self._strategy_property[WORKER][0], (True, True, True))
    self.assertEqual(self._strategy_property[WORKER][1], (True, False, False))
    self.assertEqual(self._strategy_property[WORKER][2], (True, False, False))

  def testInGraphContext(self):
    # [FUNCTION DOC]
    # Runs the context-dumping worker function in-graph in standalone mode. Expects a single
    # context under the "None" task type whose master target is worker 0's target, with three
    # workers, chief true and distributed mode true.
    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        MockStrategy(between_graph=False),
        cluster_spec=self._cluster_spec)

    # There is only a "None" task in the dumped task context.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue("None" in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(
        self._worker_context["None"][0],
        (_bytes_to_str(self._workers[0].target), NUM_WORKERS, True, True))

  def testLocalContext(self):
    # [FUNCTION DOC]
    # Runs the context-dumping worker function with no cluster spec (local training). Expects
    # a single "None" task whose master target is empty, zero workers, chief true and
    # distributed mode false.
    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        MockStrategy(between_graph=False),
        cluster_spec=None)

    # There is only a "None" task.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue("None" in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context["None"][0], ("", 0, True, False))

  def testBetweenGraphContextWithChief(self):
    # [FUNCTION DOC]
    # Adds a fake chief job to a copy of the cluster spec and runs the context-dumping worker
    # between-graph with the grpc RPC layer. Expects one chief with master target prefixed by
    # the grpc scheme, four workers counted, chief true; and three workers with their own
    # targets, four workers counted, chief false.
    # Adds a chief node, so there are NUM_WORKERS + 1 workers in total.
    cluster_spec = copy.deepcopy(self._cluster_spec)
    cluster_spec[CHIEF] = ["fake_chief"]

    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        MockStrategy(between_graph=True),
        cluster_spec=cluster_spec,
        rpc_layer="grpc")

    # There are one CHIEF and three workers.
    self.assertEqual(len(self._worker_context), 2)
    self.assertTrue(CHIEF in self._worker_context)
    self.assertTrue(WORKER in self._worker_context)
    self.assertEqual(len(self._worker_context[CHIEF]), 1)
    self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context[CHIEF][0],
                     ("grpc://fake_chief", 4, True, True))
    self.assertEqual(
        self._worker_context[WORKER][0],
        (_bytes_to_str(self._workers[0].target), NUM_WORKERS + 1, False, True))
    self.assertEqual(
        self._worker_context[WORKER][1],
        (_bytes_to_str(self._workers[1].target), NUM_WORKERS + 1, False, True))
    self.assertEqual(
        self._worker_context[WORKER][2],
        (_bytes_to_str(self._workers[2].target), NUM_WORKERS + 1, False, True))

  def testInGraphContextWithEval(self):
    # [FUNCTION DOC]
    # Adds a fake evaluator job to a copy of the cluster spec and runs the context-dumping
    # worker in-graph in standalone mode with no RPC layer. Expects a "None" task context with
    # worker 0's stripped target, three workers, chief true, distributed true, and an
    # evaluator context with empty master target, three workers, chief true, distributed false.
    # Adds a EVALUATOR job.
    cluster_spec = copy.deepcopy(self._cluster_spec)
    cluster_spec[EVALUATOR] = ["fake_evaluator"]

    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        MockStrategy(between_graph=False),
        cluster_spec=cluster_spec,
        rpc_layer=None)

    # There are one "None" task and one EVALUATOR task.
    self.assertEqual(len(self._worker_context), 2)
    self.assertTrue("None" in self._worker_context)
    self.assertTrue(EVALUATOR in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)
    self.assertEqual(len(self._worker_context[EVALUATOR]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context["None"][0], (_strip_protocol(
        _bytes_to_str(self._workers[0].target)), 3, True, True))
    self.assertEqual(self._worker_context[EVALUATOR][0], ("", 3, True, False))


class DistributeCoordinatorTestIndependentWorkerMode(
    DistributeCoordinatorTestBase):
  # [CLASS DOC]
  # Tests the coordinator in independent worker mode, where one coordinator thread runs per
  # task of a locally created cluster spec, using either real in-process servers or a mocked
  # standard-server runner. Covers in-graph, between-graph, contexts, strategy properties,
  # evaluator handling, the "google" environment and the RPC-layer setting.

  def testInGraph(self):
    # [FUNCTION DOC]
    # Starts a coordinator thread per worker task with the in-graph worker function and joins
    # the first worker's thread. Expects exactly one correct result.
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    threads = self._run_multiple_coordinator_in_threads(
        self._in_graph_worker_fn,
        MockStrategy(between_graph=False),
        cluster_spec,
        mode=INDEPENDENT_WORKER)
    self._join_threads([threads[WORKER][0]])
    self.assertEqual(self._result_correct, 1)

  def testBetweenGraph(self):
    # [FUNCTION DOC]
    # Starts coordinator threads for three workers and two parameter servers running the
    # between-graph worker function in independent mode, joins the worker threads, and expects
    # the correct-result counter to equal the worker count.
    cluster_spec = self._create_cluster_spec(
        num_workers=NUM_WORKERS, num_ps=NUM_PS)
    threads = self._run_multiple_coordinator_in_threads(
        self._between_graph_worker_fn,
        MockStrategy(between_graph=True),
        cluster_spec,
        mode=INDEPENDENT_WORKER)
    self._join_threads(threads[WORKER])

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  @test_util.run_v1_only("MonitoredSession removed from v2")
  def testBetweenGraphWithMonitoredSession(self):
    # [FUNCTION DOC]
    # Same as the independent between-graph test but the workers use a monitored session
    # (graph-mode only). Expects the counter to equal the worker count.
    cluster_spec = self._create_cluster_spec(
        num_workers=NUM_WORKERS, num_ps=NUM_PS)
    threads = self._run_multiple_coordinator_in_threads(
        self._between_graph_with_monitored_session,
        MockStrategy(between_graph=True),
        cluster_spec,
        mode=INDEPENDENT_WORKER)
    self._join_threads(threads[WORKER])

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  def testBetweenGraphContext(self):
    # [FUNCTION DOC]
    # Runs the context-dumping worker between-graph in independent mode with the standard
    # server runner mocked and no RPC layer. Expects three worker contexts whose master target
    # is each task's own address, three workers, chief only for task 0, distributed true; and
    # one mock server per worker, none of which was joined.
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    # Dumps the task contexts and std server arguments.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_worker_context,
          MockStrategy(between_graph=True),
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          rpc_layer=None)
      self._join_threads(threads[WORKER])

    # There is only one type of task and three such tasks.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue(WORKER in self._worker_context)
    self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(
        self._worker_context[WORKER][0],
        (_bytes_to_str(cluster_spec[WORKER][0]), NUM_WORKERS, True, True))
    self.assertEqual(
        self._worker_context[WORKER][1],
        (_bytes_to_str(cluster_spec[WORKER][1]), NUM_WORKERS, False, True))
    self.assertEqual(
        self._worker_context[WORKER][2],
        (_bytes_to_str(cluster_spec[WORKER][2]), NUM_WORKERS, False, True))

    # Make sure each worker runs a std server.
    self.assertEqual(len(self._std_servers), 1)
    self.assertTrue(WORKER in self._std_servers)
    self.assertEqual(len(self._std_servers[WORKER]), 3)
    self.assertFalse(self._std_servers[WORKER][0].joined)
    self.assertFalse(self._std_servers[WORKER][1].joined)
    self.assertFalse(self._std_servers[WORKER][2].joined)

  def testBetweenGraphStrategyProperties(self):
    # [FUNCTION DOC]
    # Runs the strategy-property-dumping worker between-graph in independent mode with the
    # server runner mocked and should-init forced true. Expects three worker entries with
    # should-init true everywhere and checkpoint and summary flags true only for worker 0.
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    # Dumps properties of the strategy objects.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_strategy_property,
          MockStrategy(between_graph=True, should_init=True),
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          rpc_layer=None)
      self._join_threads(threads[WORKER])

    # There is only one type of task and there three such tasks.
    self.assertEqual(len(self._strategy_property), 1)
    self.assertTrue(WORKER in self._strategy_property)
    self.assertEqual(len(self._strategy_property[WORKER]), NUM_WORKERS)

    # Check whether each task has the right properties of should_init,
    # should_checkpoint and should_save_summary.
    self.assertEqual(self._strategy_property[WORKER][0], (True, True, True))
    self.assertEqual(self._strategy_property[WORKER][1], (True, False, False))
    self.assertEqual(self._strategy_property[WORKER][2], (True, False, False))

  def testInGraphContext(self):
    # [FUNCTION DOC]
    # Runs the context-dumping worker in-graph in independent mode with the server runner
    # mocked. Expects a single "None" context (only the chief-like worker 0 runs the function)
    # targeting worker 0 with three workers, chief true, distributed true; and three mock
    # servers where worker 0's was not joined while workers 1 and 2 joined theirs.
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    # Dumps the task contexts and std server arguments.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_worker_context,
          MockStrategy(between_graph=False),
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          rpc_layer=None)
      self._join_threads(threads[WORKER])

    # There is only a "None" task in the dumped task context.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue("None" in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(
        self._worker_context["None"][0],
        (_bytes_to_str(cluster_spec[WORKER][0]), NUM_WORKERS, True, True))

    # Make sure each worker runs a std server.
    self.assertEqual(len(self._std_servers), 1)
    self.assertTrue(WORKER in self._std_servers)
    self.assertEqual(len(self._std_servers[WORKER]), 3)
    self.assertFalse(self._std_servers[WORKER][0].joined)
    self.assertTrue(self._std_servers[WORKER][1].joined)
    self.assertTrue(self._std_servers[WORKER][2].joined)

  def testInGraphContextWithEval(self):
    # [FUNCTION DOC]
    # Like the independent in-graph context test but the cluster also has an evaluator. Runs
    # all task threads with the mocked server runner and joins the worker and evaluator
    # threads. Expects a "None" context for worker 0 (three workers, chief, distributed) and
    # an evaluator context with empty master target, three workers, chief true, distributed
    # false; plus three worker mock servers with only worker 0's left unjoined.
    # Adds a EVALUATOR job.
    cluster_spec = self._create_cluster_spec(
        num_workers=NUM_WORKERS, has_eval=True)

    # Dumps the task contexts and std server arguments.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_worker_context,
          MockStrategy(between_graph=False),
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          rpc_layer=None)
      self._join_threads(threads[WORKER])
      self._join_threads([threads[EVALUATOR][0]])

    # There are one "None" task and one EVALUATOR task.
    self.assertEqual(len(self._worker_context), 2)
    self.assertTrue("None" in self._worker_context)
    self.assertTrue(EVALUATOR in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)
    self.assertEqual(len(self._worker_context[EVALUATOR]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context["None"][0],
                     (_bytes_to_str(cluster_spec[WORKER][0]), 3, True, True))
    self.assertEqual(self._worker_context[EVALUATOR][0], ("", 3, True, False))

    # Make sure each worker runs a std server.
    self.assertEqual(len(self._std_servers), 1)
    self.assertTrue(WORKER in self._std_servers)
    self.assertEqual(len(self._std_servers[WORKER]), 3)
    self.assertFalse(self._std_servers[WORKER][0].joined)
    self.assertTrue(self._std_servers[WORKER][1].joined)
    self.assertTrue(self._std_servers[WORKER][2].joined)

  def testRunStdServerInGoogleEnvironment(self):
    # [FUNCTION DOC]
    # Sets TF_CONFIG with environment "google", then runs the coordinator in a thread as
    # parameter server 0 in independent mode, which uses the fake server whose join sleeps
    # forever. The sleep function is replaced so that it records that joining happened and
    # then exits the thread. Expects the recorded flag to be true.
    cluster_spec = {"worker": ["fake_worker"], "ps": ["localhost:0"]}
    tf_config = {"cluster": cluster_spec, "environment": "google"}

    joined = [False]

    def _fake_sleep(_):
      # [FUNCTION DOC]
      # HELPER: replacement for the sleep function. Ignores its argument, records that the
      # fake server was joined, and ends the calling thread by invoking the original exit.
      joined[0] = True
      original_sys_exit(0)

    def _thread_fn(cluster_spec):
      # [FUNCTION DOC]
      # HELPER: thread body that runs the coordinator with no worker function, a between-graph
      # mock strategy, independent mode, the given cluster spec, as parameter server task 0.
      distribute_coordinator.run_distribute_coordinator(
          None,
          MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="ps",
          task_id=0)

    with test.mock.patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), test.mock.patch.object(
            time, "sleep", _fake_sleep):
      t = threading.Thread(target=_thread_fn, args=(cluster_spec,))
      t.start()
      t.join()
    self.assertTrue(joined[0])

  def testRpcLayerEnvironmentVariable(self):
    # [FUNCTION DOC]
    # Sets TF_CONFIG with an RPC layer named "cake", mocks the standard-server runner to record
    # the RPC layer it receives, and runs the coordinator as parameter server 0 in independent
    # mode. Expects the recorded RPC layer to be "cake", i.e. TF_CONFIG overrides the default.
    cluster_spec = {"worker": ["fake_worker"], "ps": ["fake_ps"]}
    tf_config = {"cluster": cluster_spec, "rpc_layer": "cake"}

    rpc_layer_from_coordinator = [None]

    def _run_mock_server(cluster_spec=None,
                         task_type=None,
                         task_id=None,
                         session_config=None,
                         rpc_layer=None,
                         environment=None):
      # [FUNCTION DOC]
      # HELPER: mock server runner. Discards all arguments except the RPC layer, stores that in
      # the shared record and returns a mock server.
      del cluster_spec, task_type, task_id, session_config, environment
      rpc_layer_from_coordinator[0] = rpc_layer
      return MockServer()

    with test.mock.patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), test.mock.patch.object(
            distribute_coordinator, "_run_std_server", _run_mock_server):
      distribute_coordinator.run_distribute_coordinator(
          None,
          MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="ps",
          task_id=0)
    self.assertEqual(rpc_layer_from_coordinator[0], "cake")


class StrategyConfigureTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the coordinator lets the strategy configure the session configuration used for
  # standard servers, for sessions created in worker contexts, and for the evaluator. Records
  # device filters and thread counts seen by mocked servers or worker functions.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture. Clears the recorded device filters and the intra-op and inter-op thread
    # counts, then runs the parent fixture. Returns nothing.
    self._device_filters = []
    self._intra_op_parallelism_threads = None
    self._inter_op_parallelism_threads = None
    super(StrategyConfigureTest, self).setUp()

  def _dump_device_filters(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: mock standard-server runner. Takes arbitrary arguments, reads the session config
    # keyword argument, appends its device filters to the recorded list, records its intra-op
    # and inter-op thread counts, and returns a mock server.
    session_config = kwargs.get("session_config", None)
    self._device_filters.extend(session_config.device_filters)
    self._intra_op_parallelism_threads = (
        session_config.intra_op_parallelism_threads)
    self._inter_op_parallelism_threads = (
        session_config.inter_op_parallelism_threads)
    return MockServer()

  def _worker_fn(self, strategy):
    # [FUNCTION DOC]
    # HELPER: worker function that records the session config of the current worker context:
    # appends its device filters to the recorded list and stores its intra-op and inter-op
    # thread counts. Returns a mock server (the return value is not otherwise used). The
    # strategy argument is unused.
    worker_context = distribute_coordinator_context.get_current_worker_context()
    session_config = worker_context._session_config
    self._device_filters.extend(session_config.device_filters)
    self._intra_op_parallelism_threads = (
        session_config.intra_op_parallelism_threads)
    self._inter_op_parallelism_threads = (
        session_config.inter_op_parallelism_threads)
    return MockServer()

  def test_session_config_in_std_server(self):
    # [FUNCTION DOC]
    # Runs the coordinator in independent mode as worker 0 with a between-graph mock strategy,
    # TF_CONFIG set, and the server runner mocked to record the session config. Expects the
    # strategy to have been configured before the server was started: intra-op thread count 1
    # and inter-op count 0 (device filters are cleared for servers).
    cluster_spec = {"worker": ["fake_worker"], "ps": ["fake_ps"]}
    tf_config = {"cluster": cluster_spec}

    with test.mock.patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), test.mock.patch.object(
            distribute_coordinator, "_run_std_server",
            self._dump_device_filters):
      distribute_coordinator.run_distribute_coordinator(
          lambda _: None,
          MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="worker",
          task_id=0)
    self.assertEqual(self._intra_op_parallelism_threads, 1)
    self.assertEqual(self._inter_op_parallelism_threads, 0)

  def test_session_config_in_session_creator(self):
    # [FUNCTION DOC]
    # Resets the coordinator's saved server state, then runs the real coordinator in
    # independent mode as worker 0 on a local address with the config-recording worker function.
    # Expects the worker context's session config to hold the device filters for worker 0 and
    # the ps job, intra-op thread count 2 (configured once for the server and once for the
    # worker) and inter-op count 0. Side effect: replaces a module-level thread-local holder.
    cluster_spec = {"worker": ["localhost:0"]}
    tf_config = {"cluster": cluster_spec}

    # Reset the saved Server state.
    distribute_coordinator._thread_local = threading.local()  # pylint: disable=protected-access

    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(tf_config)}):
      distribute_coordinator.run_distribute_coordinator(
          self._worker_fn,
          MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="worker",
          task_id=0)
    self.assertEqual(self._device_filters, ["/job:worker/task:0", "/job:ps"])
    self.assertEqual(self._intra_op_parallelism_threads, 2)
    self.assertEqual(self._inter_op_parallelism_threads, 0)

  def test_eval_strategy_configure(self):
    # [FUNCTION DOC]
    # Runs the coordinator in independent mode as an evaluator with a no-op worker function,
    # an in-graph main strategy, the config-recording function as eval function and a
    # between-graph eval strategy. Expects the evaluation strategy's non-cluster configure path:
    # device filters equal to the dummy job only, intra-op count 0 and inter-op count 2.
    cluster_spec = {"evaluator": ["localhost:0"]}
    tf_config = {"cluster": cluster_spec}

    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(tf_config)}):
      distribute_coordinator.run_distribute_coordinator(
          lambda _: None,
          MockStrategy(between_graph=False),
          eval_fn=self._worker_fn,
          eval_strategy=MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="evaluator",
          task_id=0)
    self.assertEqual(self._device_filters, ["/job:somejob"])
    self.assertEqual(self._intra_op_parallelism_threads, 0)
    self.assertEqual(self._inter_op_parallelism_threads, 2)


class RunStandardTensorflowServerTest(test.TestCase):
  # [CLASS DOC]
  # Tests the function that starts a standard server from the TF_CONFIG environment variable,
  # checking the arguments passed to the underlying server runner.

  def test_std_server_arguments(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the task entry in the test's TF_CONFIG uses the key "id" rather than the key
    # "index" that the code reads, so the task id passes the check only because it defaults to 0.
    # Sets TF_CONFIG with a worker and ps cluster and a ps task, mocks the server runner, and
    # calls the function with a session config whose intra-op threads are 1. The mock asserts
    # the cluster spec, task type "ps", task id 0, the collective group leader set to worker
    # task 0 (no chief exists), intra-op threads still 1 and RPC layer "grpc". Expects the
    # returned server to have been started.
    cs = {"worker": ["fake_worker"], "ps": ["fake_ps"]}
    tf_config = {"cluster": cs, "task": {"type": "ps", "id": 0}}

    def _mock_run_std_server(cluster_spec=None,
                             task_type=None,
                             task_id=None,
                             session_config=None,
                             rpc_layer=None):
      # [FUNCTION DOC]
      # HELPER: mock server runner that asserts the arguments described by the enclosing test
      # and returns a mock server.
      self.assertEqual(cluster_spec.as_dict(), cs)
      self.assertEqual(task_type, "ps")
      self.assertEqual(task_id, 0)
      self.assertEqual(session_config.experimental.collective_group_leader,
                       "/job:worker/replica:0/task:0")
      self.assertEqual(session_config.intra_op_parallelism_threads, 1)
      self.assertEqual(rpc_layer, "grpc")

      return MockServer()

    with test.mock.patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), test.mock.patch.object(
            distribute_coordinator, "_run_std_server", _mock_run_std_server):
      session_config = config_pb2.ConfigProto()
      session_config.intra_op_parallelism_threads = 1
      mock_server = distribute_coordinator.run_standard_tensorflow_server(
          session_config)
      self.assertTrue(mock_server.started)


if __name__ == "__main__":
  # TODO(yuefengz): find a smart way to terminate std server threads.
  with test.mock.patch.object(sys, "exit", os._exit):
    # Reduce `recovery_wait_secs` from 30 seconds so the test completes quickly.
    orig_init = session_manager.SessionManager.__init__

    def new_init(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: replacement session manager constructor installed when run as a script. Removes
      # any recovery-wait argument, forces it to half a second so tests finish quickly, and
      # calls the original constructor with the remaining arguments. Returns nothing.
      kwargs.pop("recovery_wait_secs", None)
      kwargs["recovery_wait_secs"] = 0.5
      orig_init(*args, **kwargs)

    session_manager.SessionManager.__init__ = new_init

    test.main()
