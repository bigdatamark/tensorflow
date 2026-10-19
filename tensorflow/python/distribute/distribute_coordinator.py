# [MODULE DOC]
# Implements the legacy "distribute coordinator": a component that runs user training code
# (a worker function, plus an optional evaluation function) for a distributed TensorFlow
# cluster. Depending on the mode it either acts as a standalone client that drives remote
# standard servers from one process (between-graph: one thread per chief/worker task;
# in-graph: a single thread), or runs independently on every task, starting a standard server
# and then running the worker function for that task. Cluster layout and task role come from
# arguments or the TF_CONFIG environment variable.
# Exposes: CoordinatorMode (the two modes), run_distribute_coordinator (main entry point) and
# run_standard_tensorflow_server (start a server from TF_CONFIG). Internal pieces: task type
# names, a reusable thread barrier, the per-task worker context (published through the
# distribute_coordinator_context module), and server / client helpers.
# Import-time side effects: only creates a module-level thread-local holder used to remember
# an already started server per thread. Closely related to the context module and to
# multi-worker utilities and distribution strategies, which it configures.
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
"""A component for running distributed TensorFlow."""

import copy
import json
import os
import threading
import time

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib


_thread_local = threading.local()


class _TaskType(object):
  # [CLASS DOC]
  # HELPER: namespace of task-role name constants for the distribute coordinator.
  # Holds the string names of the possible task roles: parameter server, worker, chief,
  # evaluator and client. Never instantiated; only its class attributes are read.
  PS = "ps"
  WORKER = "worker"
  CHIEF = "chief"
  EVALUATOR = "evaluator"
  CLIENT = "client"


# TODO(yuefengz): support another mode where the client colocates with one
# worker.
class CoordinatorMode(object):
  # [CLASS DOC]
  # Namespace of string constants naming how the coordinator runs: a standalone client that
  # connects to remote servers, or an independent worker where each task runs its own
  # coordinator with a local server. Never instantiated; passed as the mode argument to
  # run_distribute_coordinator.
  """Specify how distribute coordinator runs."""
  # The default mode where distribute coordinator will run as a standalone
  # client and connects to remote servers for training.  Each remote server can
  # use the distribute coordinator binary with task_type set correctly which
  # will then turn into standard servers.
  STANDALONE_CLIENT = "standalone_client"

  # The distribute coordinator runs on each worker. It will run a standard
  # server on each worker and optionally run the `worker_fn` that is configured
  # to talk to its standard server.
  INDEPENDENT_WORKER = "independent_worker"


class _Barrier(object):
  # [CLASS DOC]
  # HELPER: synchronization primitive for the worker threads started by the coordinator.
  # A reusable (sense-reversing) barrier: a fixed number of participating threads each call
  # wait, and all are released once the last one arrives; it can then be used again. State:
  # participant count, arrival counter, a global flag, a per-thread local sense value, a lock
  # and a condition variable.
  """A reusable barrier class for worker synchronization."""

  def __init__(self, num_participants):
    # [FUNCTION DOC]
    # Constructor. num_participants is the integer number of threads that must call wait before
    # any of them is released. Initializes the arrival counter to zero, the flag to false, and
    # creates the thread-local sense holder, a lock and a condition variable. Returns nothing.
    """Initializes the barrier object.

    Args:
      num_participants: an integer which is the expected number of calls of
        `wait` pass to through this barrier.
    """
    self._num_participants = num_participants
    self._counter = 0
    self._flag = False
    self._local_sense = threading.local()
    self._lock = threading.Lock()
    self._condition = threading.Condition()

  def wait(self):
    # [FUNCTION DOC]
    # Blocks the calling thread until all participants have called wait. Each caller records
    # the opposite of the current global flag as its local sense, increments the shared counter
    # under a lock; the last arrival resets the counter and flips the global flag to its sense.
    # Everyone waits on the condition until the flag matches their sense, then wakes the others.
    # Takes no inputs and returns nothing; mutates the barrier state; blocks indefinitely if
    # fewer participants than expected ever arrive.
    """Waits until all other callers reach the same wait call."""
    self._local_sense.value = not self._flag
    with self._lock:
      self._counter += 1
      if self._counter == self._num_participants:
        self._counter = 0
        self._flag = self._local_sense.value
    with self._condition:
      while self._flag != self._local_sense.value:
        self._condition.wait()
      self._condition.notify_all()


def _get_num_workers(cluster_spec):
  # [FUNCTION DOC]
  # HELPER: computes the barrier / worker-context worker count.
  # Takes a cluster specification object (may be null or empty). Returns zero when it is
  # null or empty; otherwise returns the number of "worker" tasks plus the number of "chief"
  # tasks in it (missing jobs count as zero). No side effects.
  """Gets number of workers including chief."""
  if not cluster_spec:
    return 0
  return len(cluster_spec.as_dict().get(_TaskType.WORKER, [])) + len(
      cluster_spec.as_dict().get(_TaskType.CHIEF, []))


class _WorkerContext(object):
  # [CLASS DOC]
  # Per-task configuration object handed to the user worker function. Holds the distribution
  # strategy, cluster specification, task type and id, session configuration, RPC protocol,
  # an optional worker barrier, and derived values: master target address, number of workers
  # and whether the task is the chief. Created once per worker function invocation and used
  # as a context manager: entering publishes it as the current thread's worker context (and
  # fails if one is already active), exiting clears it. Exposes read-only properties and
  # helpers for creating sessions and waiting for other workers.
  """The worker context class.

  This context object provides configuration information for each task. One
  context manager with a worker context object will be created per
  invocation to the `worker_fn` where `get_current_worker_context` can be called
  to access the worker context object.
  """

  def __init__(self,
               strategy,
               cluster_spec,
               task_type,
               task_id,
               session_config=None,
               rpc_layer="grpc",
               worker_barrier=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: strategy (distribution strategy object, may be null for evaluator);
    # cluster_spec (cluster specification, may be null or empty for local training); task_type
    # (string role, null for local or in-graph training); task_id (integer, same nullability);
    # session_config (optional session configuration message, default null); rpc_layer (RPC
    # protocol string, default "grpc"; empty or null means hosts are used directly);
    # worker_barrier (optional barrier shared by workers). Stores them and computes the master
    # target, the worker count and whether this task is chief. Returns nothing.
    """Initialize the worker context object.

    Args:
      strategy: a `DistributionStrategy` object.
      cluster_spec: a ClusterSpec object. It can be empty or None in the local
        training case.
      task_type: a string indicating the role of the corresponding task, such as
        "worker" or "ps". It can be None if it is local training or in-graph
        replicated training.
      task_id: an integer indicating id of the corresponding task. It can be
        None if it is local training or in-graph replicated training.
      session_config: an optional `tf.compat.v1.ConfigProto` object.
      rpc_layer: optional string specifying the RPC protocol for communication
        with worker masters. If None or empty, hosts in the `cluster_spec` will
        be used directly.
      worker_barrier: optional, the barrier object for worker synchronization.
    """
    self._strategy = strategy
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._session_config = session_config
    self._worker_barrier = worker_barrier
    self._rpc_layer = rpc_layer
    self._master_target = self._get_master_target()
    self._num_workers = _get_num_workers(cluster_spec)
    self._is_chief_node = self._is_chief()

  def _debug_message(self):
    # [FUNCTION DOC]
    # HELPER: builds a text used in error messages. Returns a string describing the cluster
    # spec, task type and task id when a cluster spec is set, or the text "local" otherwise.
    if self._cluster_spec:
      return "[cluster_spec: %r, task_type: %r, task_id: %r]" % (
          self._cluster_spec, self.task_type, self.task_id)
    else:
      return "[local]"

  def __enter__(self):
    # [FUNCTION DOC]
    # Context-manager entry. Raises a value error (including the debug message) if a worker
    # context is already active on this thread, i.e. the coordinator is being run inside a
    # worker function. Otherwise installs this object as the current thread's worker context.
    # Returns nothing (so a with-as target would be null).
    old_context = distribute_coordinator_context.get_current_worker_context()
    if old_context:
      raise ValueError(
          "You cannot run distribute coordinator in a `worker_fn`.\t" +
          self._debug_message())
    # pylint: disable=protected-access
    distribute_coordinator_context._worker_context.current = self

  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    # [FUNCTION DOC]
    # Context-manager exit. Ignores the exception information (never suppresses exceptions)
    # and clears the current thread's worker context. Returns nothing.
    # pylint: disable=protected-access
    distribute_coordinator_context._worker_context.current = None

  def _get_master_target(self):
    # [FUNCTION DOC]
    # HELPER: computes the session master address for this task.
    # Returns an empty string (local master) when there is no cluster spec or the task is an
    # evaluator. When the task type is null (in-graph training) it uses the first chief task if
    # a chief job exists, otherwise the first worker (asserting a worker job exists). Otherwise
    # it uses this task's own type and id. The result is the task address, prefixed with the
    # RPC protocol and a scheme separator when an RPC layer is set. A null id is treated as 0.
    """Return the master target for a task."""
    # If cluster_spec is None or empty, we use local master.
    if not self._cluster_spec or self._task_type == _TaskType.EVALUATOR:
      return ""

    # If task_type is None, then it is in-graph replicated training. In this
    # case we use the chief or first worker's master target.
    if not self._task_type:
      if _TaskType.CHIEF in self._cluster_spec.jobs:
        task_type = _TaskType.CHIEF
        task_id = 0
      else:
        assert _TaskType.WORKER in self._cluster_spec.jobs
        task_type = _TaskType.WORKER
        task_id = 0
    else:
      task_type = self._task_type
      task_id = self._task_id

    prefix = ""
    if self._rpc_layer:
      prefix = self._rpc_layer + "://"
    return prefix + self._cluster_spec.job_tasks(task_type)[task_id or 0]

  def _is_chief(self):
    # [FUNCTION DOC]
    # HELPER: decides whether this task is the chief. Returns true when there is no cluster
    # spec, or the task type is chief, evaluator or null; also true for worker 0 when the
    # cluster has no chief job. Returns false otherwise.
    """Return whether the task is the chief worker."""
    if (not self._cluster_spec or
        self._task_type in [_TaskType.CHIEF, _TaskType.EVALUATOR, None]):
      return True

    # If not local and chief not in the cluster_spec, use the first worker as
    # chief.
    if (_TaskType.CHIEF not in self._cluster_spec.jobs and
        self._task_type == _TaskType.WORKER and self._task_id == 0):
      return True
    return False

  def wait_for_other_workers(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises a value error when no barrier was given, but
    # the code silently returns without waiting (a to-do notes an error is wanted only in
    # independent worker mode).
    # Blocks until every worker has reached the same call, using the shared barrier. If no
    # barrier was supplied to the context it returns immediately. Takes no inputs, returns
    # nothing.
    """Waits for other workers to reach the same call to this method.

    Raises:
      ValueError: if `worker_barrier` is not passed to the __init__ method.
    """
    if not self._worker_barrier:
      # TODO(yuefengz): we should throw an error in independent worker mode.
      return
    self._worker_barrier.wait()

  def session_creator(self,
                      scaffold=None,
                      config=None,
                      checkpoint_dir=None,
                      checkpoint_filename_with_path=None,
                      max_wait_secs=7200):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when a config argument is given it is copied and merged with the context's
    # session config, which fails if that stored config is null; the log messages print the
    # config argument rather than the merged configuration actually used.
    # Builds a monitored-session creator targeting this task's master. Inputs (all optional):
    # scaffold (supportive-ops holder, default created internally), config (session
    # configuration merged on top of a copy of the context's config), checkpoint_dir (directory
    # to restore from), checkpoint_filename_with_path (full checkpoint file; use only one of the
    # two), max_wait_secs (maximum wait for the session to be ready, default 7200, used only by
    # the worker creator). Returns a chief session creator (runs init ops) when there is no
    # strategy or the strategy says this task should initialize; otherwise a worker session
    # creator (waits for the chief). Logs which kind it made.
    """Returns a session creator.

    The returned session creator will be configured with the correct master
    target and session configs. It will also run either init ops or ready ops
    by querying the `strategy` object when `create_session` is called on it.

    Args:
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      config: `ConfigProto` proto used to configure the session.
      checkpoint_dir: A string. Optional path to a directory where to restore
        variables.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
        Only one of `checkpoint_dir` or `checkpoint_filename_with_path` can be
        specified.
      max_wait_secs: Maximum time to wait for the session to become available.

    Returns:
      a descendant of SessionCreator.
    """
    if config:
      session_config = copy.deepcopy(config)
      session_config.MergeFrom(self._session_config)
    else:
      session_config = self._session_config

    if not self._strategy or self._strategy.extended.experimental_should_init:
      logging.info("Creating chief session creator with config: %r", config)
      return monitored_session.ChiefSessionCreator(
          scaffold,
          master=self.master_target,
          config=session_config,
          checkpoint_dir=checkpoint_dir,
          checkpoint_filename_with_path=checkpoint_filename_with_path)
    else:
      logging.info("Creating worker session creator with config: %r", config)
      return monitored_session.WorkerSessionCreator(
          scaffold,
          master=self.master_target,
          config=session_config,
          max_wait_secs=max_wait_secs)

  @property
  def session_config(self):
    # [FUNCTION DOC]
    # Property. Returns an independent deep copy of the session configuration (may be null).
    return copy.deepcopy(self._session_config)

  @property
  def has_barrier(self):
    # [FUNCTION DOC]
    # Property. True when a worker barrier was supplied, false otherwise.
    """Whether the barrier is set or not."""
    return self._worker_barrier is not None

  @property
  def distributed_mode(self):
    # [FUNCTION DOC]
    # Property. True when a non-empty cluster spec exists and the task is not an evaluator
    # (evaluators run single-machine), false otherwise.
    """Whether it is distributed training or not."""
    return bool(self._cluster_spec) and self._task_type != _TaskType.EVALUATOR

  @property
  def cluster_spec(self):
    # [FUNCTION DOC]
    # Property. Returns an independent deep copy of the cluster specification (may be null).
    """Returns a copy of the cluster_spec object."""
    return copy.deepcopy(self._cluster_spec)

  @property
  def task_type(self):
    # [FUNCTION DOC]
    # Property. Returns the task role string given at construction (may be null).
    """Returns the role of the corresponding task."""
    return self._task_type

  @property
  def task_id(self):
    # [FUNCTION DOC]
    # Property. Returns the integer task index given at construction (may be null).
    """Returns the id or index of the corresponding task."""
    return self._task_id

  @property
  def master_target(self):
    # [FUNCTION DOC]
    # Property. Returns the precomputed session master address string (empty for local).
    """Returns the session master for the corresponding task to connect to."""
    return self._master_target

  @property
  def is_chief(self):
    # [FUNCTION DOC]
    # Property. Returns the precomputed boolean saying whether this task is the chief.
    """Returns whether the task is a chief node."""
    return self._is_chief_node

  @property
  def num_workers(self):
    # [FUNCTION DOC]
    # Property. Returns the precomputed number of workers (workers plus chief; zero if local).
    """Returns number of workers in the cluster, including chief."""
    return self._num_workers

  @property
  def experimental_should_init(self):
    # [FUNCTION DOC]
    # Property. Delegates to the strategy's extended object and returns whether init ops
    # should be run. Fails if the strategy is null.
    """Whether to run init ops."""
    return self._strategy.extended.experimental_should_init

  @property
  def should_checkpoint(self):
    # [FUNCTION DOC]
    # Property. Delegates to the strategy's extended object and returns whether this task
    # should save checkpoints. Fails if the strategy is null.
    """Whether to save checkpoint."""
    return self._strategy.extended.should_checkpoint

  @property
  def should_save_summary(self):
    # [FUNCTION DOC]
    # Property. Delegates to the strategy's extended object and returns whether this task
    # should save summaries. Fails if the strategy is null.
    """Whether to save summaries."""
    return self._strategy.extended.should_save_summary


def _run_single_worker(worker_fn,
                       strategy,
                       cluster_spec,
                       task_type,
                       task_id,
                       session_config,
                       rpc_layer="",
                       worker_barrier=None,
                       coord=None):
  # [FUNCTION DOC]
  # HELPER: runs one task's user function; used by the client runners, by the independent
  # worker path and by the local case.
  # Inputs: worker_fn (callable taking the strategy), strategy (distribution strategy, may be
  # null only for an evaluator), cluster_spec, task_type, task_id, session_config (message;
  # copied, so the caller's is not mutated), rpc_layer (default empty), worker_barrier
  # (optional), coord (optional thread coordinator). Deep-copies the session config and
  # strategy, configures the strategy (evaluator: session config only, if a strategy exists;
  # others: asserts a strategy and passes cluster spec, type and id), builds a worker context
  # and calls worker_fn with the strategy inside it, wrapped in the coordinator's
  # stop-on-exception scope when coord is given. Returns whatever worker_fn returns.
  """Runs a single worker by calling `worker_fn` under context."""
  session_config = copy.deepcopy(session_config)
  strategy = copy.deepcopy(strategy)
  # If there is an EVALUATOR task, we run single-machine eval on that task.
  if task_type == _TaskType.EVALUATOR:
    # It is possible to not have a strategy object for EVALUATOR task.
    if strategy:
      strategy.configure(session_config)
  else:
    assert strategy
    strategy.configure(session_config, cluster_spec, task_type, task_id)

  context = _WorkerContext(
      strategy,
      cluster_spec,
      task_type,
      task_id,
      session_config=session_config,
      rpc_layer=rpc_layer,
      worker_barrier=worker_barrier)
  with context:
    if coord:
      with coord.stop_on_exception():
        return worker_fn(strategy)
    else:
      return worker_fn(strategy)


def _split_cluster_for_evaluator(cluster_spec, task_type):
  # [FUNCTION DOC]
  # HELPER: isolates the evaluator from other tasks when starting a server.
  # Normalizes the cluster spec to a mapping. For an evaluator task (asserting an evaluator
  # job exists) the result contains only the evaluator job; for any other task the evaluator
  # job is removed. Returns the normalized cluster spec object; inputs are not modified.
  """Split the cluster for evaluator since it needn't talk to other tasks."""
  # Splitting the cluster is important to prevent the evaluator from talking to
  # other tasks in the cluster. Since we allow evaluator not to use
  # distribution strategies and as a result ops in the evaluator task may have
  # unspecified devices. Those ops may end up on other tasks if we don't split
  # the cluster.
  # Note: if you bypass distribute coordinator and bring the cluster yourself,
  # you can equivalently set device filters to split clusters. This is already
  # done by distribution strategy's `update_config_proto` method.
  new_cluster_spec = multi_worker_util.normalize_cluster_spec(
      cluster_spec).as_dict()
  if task_type == _TaskType.EVALUATOR:
    assert _TaskType.EVALUATOR in new_cluster_spec
    new_cluster_spec = {
        _TaskType.EVALUATOR: new_cluster_spec[_TaskType.EVALUATOR]
    }
  else:
    new_cluster_spec.pop(_TaskType.EVALUATOR, None)
  return multi_worker_util.normalize_cluster_spec(new_cluster_spec)


def _run_std_server(cluster_spec=None,
                    task_type=None,
                    task_id=None,
                    session_config=None,
                    rpc_layer=None,
                    environment=None):
  # [FUNCTION DOC]
  # HELPER: starts (or reuses) the standard TensorFlow server for this thread's task.
  # Inputs (all optional but cluster_spec and task_type/id are needed to start): cluster_spec,
  # task_type, task_id, session_config, rpc_layer, environment (the value "google" selects a
  # fake server that just opens a remote session). If a server was already started on this
  # thread, asserts that all arguments are unchanged and returns it. Otherwise records the
  # arguments in thread-local state (not thread-safe), asserts a cluster spec, builds the
  # target address (prefixed by the RPC layer if set), creates and starts the server (the real
  # one uses the evaluator-split cluster) and stores it. Returns the started server.
  """Runs a standard server."""
  # Check if the Server is already running. If so, assert that no configuration
  # options have changed, and return the existing Server. This allows us to
  # call `run_distribute_coordinator` multiple times.
  if getattr(_thread_local, "server", None) is not None:
    assert _thread_local.cluster_spec == cluster_spec
    assert _thread_local.task_type == task_type
    assert _thread_local.task_id == task_id
    assert _thread_local.session_config_str == repr(session_config)
    assert _thread_local.rpc_layer == rpc_layer
    assert _thread_local.environment == environment
    return _thread_local.server
  else:
    # This method is not thread-safe.
    _thread_local.server_started = True
    _thread_local.cluster_spec = cluster_spec
    _thread_local.task_type = task_type
    _thread_local.task_id = task_id
    _thread_local.session_config_str = repr(session_config)
    _thread_local.rpc_layer = rpc_layer
    _thread_local.environment = environment

  assert cluster_spec
  target = cluster_spec.task_address(task_type, task_id)
  if rpc_layer:
    target = rpc_layer + "://" + target

  class _FakeServer(object):
    # [CLASS DOC]
    # HELPER: stand-in server used when the environment is "google". Starting it opens a
    # remote session to the target, which makes the real server start remotely; joining it
    # sleeps forever. Closes over the target and session configuration of the enclosing call.
    """A fake server that runs a master session."""

    def start(self):
      # [FUNCTION DOC]
      # Logs and creates a session to the target with the session configuration, which
      # triggers the remote server to start. Returns nothing.
      # A tensorflow server starts when a remote session is created.
      logging.info(
          "Creating a remote session to start a TensorFlow server, "
          "target = %r, session_config=%r", target, session_config)
      session.Session(target=target, config=session_config)

    def join(self):
      # [FUNCTION DOC]
      # Blocks forever, sleeping in five-second intervals; never returns.
      while True:
        time.sleep(5)

  if environment == "google":
    server = _FakeServer()
  else:
    if session_config:
      logging.info(
          "Starting standard TensorFlow server, target = %r, session_config= "
          "%r", target, session_config)
    else:
      logging.info("Starting standard TensorFlow server, target = %r", target)
    cluster_spec = _split_cluster_for_evaluator(cluster_spec, task_type)
    server = server_lib.Server(
        cluster_spec,
        job_name=task_type,
        task_index=task_id,
        config=session_config,
        protocol=rpc_layer)

  server.start()
  _thread_local.server = server
  return server


def _run_between_graph_client(worker_fn, strategy, eval_fn, eval_strategy,
                              cluster_spec, session_config, rpc_layer):
  # [FUNCTION DOC]
  # HELPER: standalone-client path for between-graph replication.
  # Inputs: worker_fn and strategy for chief/worker tasks, eval_fn and eval_strategy for the
  # evaluator, cluster_spec, session_config, rpc_layer. Starts one thread for the evaluator (if
  # the cluster has one) and one thread per chief and worker task, each running the single
  # worker runner with a shared coordinator and a shared barrier sized to the worker count,
  # then joins all threads through the coordinator (which re-raises thread errors). Returns
  # nothing (null); results of individual workers are discarded.
  """Runs a standalone client for between-graph replication."""
  coord = coordinator.Coordinator()
  eval_thread = None
  if _TaskType.EVALUATOR in cluster_spec.jobs:
    eval_thread = threading.Thread(
        target=_run_single_worker,
        args=(eval_fn, eval_strategy, cluster_spec, _TaskType.EVALUATOR, 0,
              session_config),
        kwargs={
            "rpc_layer": rpc_layer,
            "coord": coord,
        })
    eval_thread.start()

  threads = []
  worker_barrier = _Barrier(_get_num_workers(cluster_spec))
  for task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
    for task_id in range(len(cluster_spec.as_dict().get(task_type, []))):
      t = threading.Thread(
          target=_run_single_worker,
          args=(worker_fn, strategy, cluster_spec, task_type, task_id,
                session_config),
          kwargs={
              "rpc_layer": rpc_layer,
              "worker_barrier": worker_barrier,
              "coord": coord,
          })
      t.start()
      threads.append(t)

  if eval_thread:
    # TODO(yuefengz): is it necessary to join eval thread?
    threads_to_join = threads + [eval_thread]
  else:
    threads_to_join = threads
  coord.join(threads_to_join)

  # TODO(yuefengz): we probably want to return results from all workers?
  return None


def _run_in_graph_client(worker_fn, strategy, eval_fn, eval_strategy,
                         cluster_spec, session_config, rpc_layer):
  # [FUNCTION DOC]
  # HELPER: standalone-client path for in-graph replication.
  # Same inputs as the between-graph variant. Starts an evaluator thread if the cluster has an
  # evaluator, then runs worker_fn once in the calling thread with a null task type and id
  # (so the context targets the chief or first worker), then joins the evaluator thread.
  # Returns the value returned by worker_fn.
  """Runs a standalone client for in-graph replication."""
  coord = coordinator.Coordinator()
  eval_thread = None
  if _TaskType.EVALUATOR in cluster_spec.jobs:
    eval_thread = threading.Thread(
        target=_run_single_worker,
        args=(eval_fn, eval_strategy, cluster_spec, _TaskType.EVALUATOR, 0,
              session_config),
        kwargs={
            "rpc_layer": rpc_layer,
            "coord": coord,
        })
    eval_thread.start()

  worker_result = _run_single_worker(
      worker_fn,
      strategy,
      cluster_spec,
      None,
      None,
      session_config,
      rpc_layer=rpc_layer,
      coord=coord)

  if eval_thread:
    coord.join([eval_thread])

  return worker_result


def _configure_session_config_for_std_servers(
    strategy, eval_strategy, session_config, cluster_spec, task_type, task_id):
  # [FUNCTION DOC]
  # HELPER: prepares the session configuration used when bringing up a standard server.
  # Inputs: strategy and eval_strategy (distribution strategies; eval one may be null),
  # session_config (message, mutated in place), cluster_spec, task_type, task_id. For an
  # evaluator it configures the eval strategy (if present) with only the session config;
  # otherwise it configures a deep copy of the strategy with config, cluster spec, type and
  # id. Finally clears the device filters from the session config so the server can be shared
  # by other strategies. Returns nothing.
  # pylint: disable=g-doc-args
  """Call strategy's `configure` to mutate the session_config.

  The session_config is currently needed as default config for a TensorFlow
  server. In the future, we should be able to remove this method and only pass
  the session config to a client session.
  """
  if task_type == _TaskType.EVALUATOR:
    if eval_strategy:
      eval_strategy.configure(session_config=session_config)
  else:
    # The strategy may be shared in standalone client mode.
    strategy = copy.deepcopy(strategy)
    strategy.configure(
        session_config=session_config,
        cluster_spec=cluster_spec,
        task_type=task_type,
        task_id=task_id)
  # Remove the device filters specific to the strategy, so that the
  # TensorFlow server brought up with one strategy can be used by other
  # strategies. The device filters can be set in the client side as well.
  del session_config.device_filters[:]


def run_standard_tensorflow_server(session_config=None):
  # [FUNCTION DOC]
  # Starts a standard server described by the TF_CONFIG environment variable (JSON text).
  # Input: optional session_config (default: a new empty configuration; it is mutated to set
  # the collective group leader to the chief task 0 if a chief job exists, else worker task 0).
  # Reads the cluster, task type, task index (default 0) and RPC layer (default "grpc").
  # Raises a value error when the cluster or task entry, or its type, is missing, or when the
  # cluster has neither chief nor worker jobs. Returns the started server. Side effects: reads
  # the environment, records the server in thread-local state, starts the server.
  """Starts a standard TensorFlow server.

  This method parses configurations from "TF_CONFIG" environment variable and
  starts a TensorFlow server. The "TF_CONFIG" is typically a json string and
  must have information of the cluster and the role of the server in the
  cluster. One example is:

  TF_CONFIG='{
      "cluster": {
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "worker", "index": 1}
  }'

  This "TF_CONFIG" specifies there are 3 workers and 2 ps tasks in the cluster
  and the current role is worker 1.

  Valid task types are "chief", "worker", "ps" and "evaluator" and you can have
  at most one "chief" and at most one "evaluator".

  An optional key-value can be specified is "rpc_layer". The default value is
  "grpc".

  Args:
    session_config: an optional `tf.compat.v1.ConfigProto` object. Users can
      pass in the session config object to configure server-local devices.

  Returns:
    a `tf.distribute.Server` object which has already been started.

  Raises:
    ValueError: if the "TF_CONFIG" environment is not complete.
  """
  tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
  if "cluster" not in tf_config:
    raise ValueError("\"cluster\" is not found in TF_CONFIG.")
  cluster_spec = multi_worker_util.normalize_cluster_spec(tf_config["cluster"])
  if "task" not in tf_config:
    raise ValueError("\"task\" is not found in TF_CONFIG.")
  task_env = tf_config["task"]
  if "type" not in task_env:
    raise ValueError(
        "\"task_type\" is not found in the `task` part of TF_CONFIG.")
  task_type = task_env["type"]
  task_id = int(task_env.get("index", 0))

  rpc_layer = tf_config.get("rpc_layer", "grpc")

  session_config = session_config or config_pb2.ConfigProto()
  # Set the collective group leader for collective ops to initialize collective
  # ops when server starts.
  if "chief" in cluster_spec.jobs:
    session_config.experimental.collective_group_leader = (
        "/job:chief/replica:0/task:0")
  else:
    if "worker" not in cluster_spec.jobs:
      raise ValueError(
          "You must have `chief` or `worker` jobs in the `cluster_spec`.")
    session_config.experimental.collective_group_leader = (
        "/job:worker/replica:0/task:0")

  server = _run_std_server(
      cluster_spec=cluster_spec,
      task_type=task_type,
      task_id=task_id,
      session_config=session_config,
      rpc_layer=rpc_layer)
  server.start()
  return server


# TODO(yuefengz): propagate cluster_spec in the STANDALONE_CLIENT mode.
# TODO(yuefengz): we may need a smart way to figure out whether the current task
# is the special task when we support cluster_spec propagation.
def run_distribute_coordinator(worker_fn,
                               strategy,
                               eval_fn=None,
                               eval_strategy=None,
                               mode=CoordinatorMode.STANDALONE_CLIENT,
                               cluster_spec=None,
                               task_type=None,
                               task_id=None,
                               session_config=None,
                               rpc_layer="grpc"):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring's Raises section mentions only an invalid cluster spec type, but
  # the code also raises a value error for an unexpected coordinator mode or an unexpected task
  # type; nothing validates the cluster spec type itself. Also the INDEPENDENT_WORKER path for
  # a parameter-server or non-chief in-graph task never returns a value after joining.
  # Main entry point. Inputs: worker_fn (callable taking a strategy), strategy, optional
  # eval_fn and eval_strategy (evaluator falls back to worker_fn / no strategy with warnings),
  # mode (default standalone client), cluster_spec (mapping, cluster definition or spec),
  # task_type, task_id, session_config (default: soft placement enabled), rpc_layer.
  # TF_CONFIG overrides rpc_layer, supplies the environment and, when no cluster spec is given,
  # the cluster and task; else a cluster resolver on the strategy may supply them.
  # Behavior: with no cluster it runs worker_fn then eval_fn locally. In standalone client mode
  # a client (or null task type) runs the between-graph or in-graph client runner depending on
  # the strategy; any other task type configures the session config, starts a standard server
  # and joins it. In independent worker mode every task configures the session config, starts a
  # server (unless evaluator or already started), then: chief/worker run worker_fn (all tasks
  # if between-graph; only the chief if in-graph, others join the server), evaluators run
  # eval_fn, parameter servers join the server. Returns the worker function's result in
  # in-graph or independent worker cases, otherwise null.
  """Runs the coordinator for distributed TensorFlow.


  This function runs a split coordinator for distributed TensorFlow in its
  default mode, i.e the STANDALONE_CLIENT mode. Given a `cluster_spec`
  specifying server addresses and their roles in a cluster, this coordinator
  will figure out how to set them up, give the underlying function the right
  targets for master sessions via a scope object and coordinate their training.
  The cluster consisting of standard servers needs to be brought up either with
  the standard server binary or with a binary running distribute coordinator
  with `task_type` set to non-client type which will then turn into standard
  servers.

  In addition to be the distribute coordinator, this is also the source of
  configurations for each job in the distributed training. As there are multiple
  ways to configure a distributed TensorFlow cluster, its context object
  provides these configurations so that users or higher-level APIs don't have to
  figure out the configuration for each job by themselves.

  In the between-graph replicated training, this coordinator will create
  multiple threads and each calls the `worker_fn` which is supposed to create
  its own graph and connect to one worker master given by its context object. In
  the in-graph replicated training, it has only one thread calling this
  `worker_fn`.

  Another mode is the INDEPENDENT_WORKER mode where each server runs a
  distribute coordinator which will start a standard server and optionally runs
  `worker_fn` depending whether it is between-graph training or in-graph
  replicated training.

  The `strategy` object is expected to be a DistributionStrategy object which
  has implemented methods needed by distributed coordinator such as
  `configure(session_config, cluster_spec, task_type, task_id)` which configures
  the strategy object for a specific task and `experimental_should_init`
  property which instructs the distribute coordinator whether to run init ops
  for a task. The distribute coordinator will make a copy of the `strategy`
  object, call its `configure` method and pass it to `worker_fn` as an argument.

  The `worker_fn` defines the training logic and is called under its own
  worker context which can be accessed to via `get_current_worker_context`. A
  worker context provides access to configurations for each task, e.g. the
  task_type, task_id, master target and so on. Since `worker_fn` will be called
  in a thread and possibly multiple times, caller should be careful when it
  accesses global data. For example, it is unsafe to define flags in a
  `worker_fn` or to define different environment variables for different
  `worker_fn`s.

  The `worker_fn` for the between-graph replication is defined as if there is
  only one worker corresponding to the `worker_fn` and possibly ps jobs. For
  example, when training with parameter servers, it assigns variables to
  parameter servers and all other operations to that worker. In the in-graph
  replication case, the `worker_fn` has to define operations for all worker
  jobs. Using a distribution strategy can simplify the `worker_fn` by not having
  to worry about the replication and device assignment of variables and
  operations.

  This method is intended to be invoked by high-level APIs so that users don't
  have to explicitly call it to run this coordinator. For those who don't use
  high-level APIs, to change a program to use this coordinator, wrap everything
  in a the program after global data definitions such as commandline flag
  definition into the `worker_fn` and get task-specific configurations from
  the worker context.

  The `cluster_spec` can be either passed by the argument or parsed from the
  "TF_CONFIG" environment variable. Example of a TF_CONFIG:
  ```
    cluster = {'chief': ['host0:2222'],
               'ps': ['host1:2222', 'host2:2222'],
               'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
    os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster})
  ```

  If `cluster_spec` is not given in any format, it becomes local training and
  this coordinator will connect to a local session.

  For evaluation, if "evaluator" exists in the cluster_spec, a separate thread
  will be created to call `eval_fn` with its `task_type` set to "evaluator". If
  `eval_fn` is not defined, fall back to `worker_fn`. This implies that
  evaluation will be done on a single machine if there is an "evaluator" task.
  If "evaluator" doesn't exist in the cluster_spec, it entirely depends on the
  `worker_fn` for how to do evaluation.

  Args:
    worker_fn: the function to be called. The function should accept a
      `strategy` object and will be given access to a context object via a
      context manager scope.
    strategy: a DistributionStrategy object specifying whether it should
      run between-graph replicated training or not, whether to run init ops,
      etc. This object will also be configured given `session_config`,
      `cluster_spec`, `task_type` and `task_id`.
    eval_fn: optional function for "evaluator" task. If `eval_fn` is not passed
      in but a "evaluator" task is found in the `cluster_spec`, the `worker_fn`
      will be used for this task.
    eval_strategy: optional DistributionStrategy object for "evaluator" task.
    mode: in which mode this distribute coordinator runs.
    cluster_spec: a dict, ClusterDef or ClusterSpec specifying servers and roles
      in a cluster. If not set or empty, fall back to local training.
    task_type: the current task type, optional if this is a client.
    task_id: the current task id, optional if this is a client.
    session_config: an optional `tf.compat.v1.ConfigProto` object which will be
      passed to `strategy`'s `configure` method and used to create a session.
    rpc_layer: optional string, the protocol for RPC, e.g. "grpc".

  Raises:
    ValueError: if `cluster_spec` is supplied but not a dict or a ClusterDef or
      a ClusterSpec.

  Returns:
    In the client job, return the value returned by `worker_fn` if
    it is in-graph replication or INDEPENDENT_WORKER mode; return None
    otherwise.
  """
  tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
  rpc_layer = tf_config.get("rpc_layer", rpc_layer)
  environment = tf_config.get("environment", None)

  if not cluster_spec:
    cluster_spec = tf_config.get("cluster", {})
    task_env = tf_config.get("task", {})
    if task_env:
      task_type = task_env.get("type", task_type)
      task_id = int(task_env.get("index", task_id))

  if cluster_spec:
    # TODO(yuefengz): validate cluster_spec.
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
  elif hasattr(strategy.extended, "_cluster_resolver"):
    cluster_resolver = strategy.extended._cluster_resolver  # pylint: disable=protected-access
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    rpc_layer = cluster_resolver.rpc_layer or rpc_layer
    environment = cluster_resolver.environment
    cluster_spec = cluster_resolver.cluster_spec()

  # Setting the session config is necessary for some strategies such as
  # CollectiveAllReduceStrategy.
  session_config = session_config or config_pb2.ConfigProto(
      allow_soft_placement=True)

  if cluster_spec:
    logging.info(
        "Running Distribute Coordinator with mode = %r, cluster_spec = %r, "
        "task_type = %r, task_id = %r, environment = %r, rpc_layer = %r", mode,
        cluster_spec.as_dict(), task_type, task_id, environment, rpc_layer)

  if not cluster_spec:
    # `mode` is ignored in the local case.
    logging.info("Running local Distribute Coordinator.")
    _run_single_worker(worker_fn, strategy, None, None, None, session_config,
                       rpc_layer)
    if eval_fn:
      _run_single_worker(eval_fn, eval_strategy, None, None, None,
                         session_config, rpc_layer)
    else:
      logging.warning("Skipped evaluation since `eval_fn` is not passed in.")
  elif mode == CoordinatorMode.STANDALONE_CLIENT:
    if not eval_fn:
      logging.warning("`eval_fn` is not passed in. The `worker_fn` will be "
                      "used if an \"evaluator\" task exists in the cluster.")
    eval_fn = eval_fn or worker_fn
    if not eval_strategy:
      logging.warning("`eval_strategy` is not passed in. No distribution "
                      "strategy will be used for evaluation.")

    # The client must know the cluster but servers in the cluster don't have to
    # know the client.
    if task_type in [_TaskType.CLIENT, None]:
      if strategy.extended.experimental_between_graph:
        return _run_between_graph_client(worker_fn, strategy, eval_fn,
                                         eval_strategy, cluster_spec,
                                         session_config, rpc_layer)
      else:
        return _run_in_graph_client(worker_fn, strategy, eval_fn, eval_strategy,
                                    cluster_spec, session_config, rpc_layer)
    else:
      # If not a client job, run the standard server.
      _configure_session_config_for_std_servers(strategy, eval_strategy,
                                                session_config, cluster_spec,
                                                task_type, task_id)
      server = _run_std_server(
          cluster_spec=cluster_spec,
          task_type=task_type,
          task_id=task_id,
          session_config=session_config,
          rpc_layer=rpc_layer,
          environment=environment)
      server.join()
  else:
    if mode != CoordinatorMode.INDEPENDENT_WORKER:
      raise ValueError("Unexpected coordinator mode: %r" % mode)

    if not eval_fn:
      logging.warning("`eval_fn` is not passed in. The `worker_fn` will be "
                      "used if an \"evaluator\" task exists in the cluster.")
    eval_fn = eval_fn or worker_fn
    if not eval_strategy:
      logging.warning("`eval_strategy` is not passed in. No distribution "
                      "strategy will be used for evaluation.")

    # Every one starts a standard server, get session config from `configure`
    # method.
    _configure_session_config_for_std_servers(strategy, eval_strategy,
                                              session_config, cluster_spec,
                                              task_type, task_id)

    if (task_type != _TaskType.EVALUATOR and
        not getattr(strategy.extended, "_std_server_started", False)):
      # Right now, with eager mode, context is configured with a std server at
      # the very beginning while with graph mode the std server is started when
      # distribute coordinator is called. We should consolidate these two paths.
      server = _run_std_server(
          cluster_spec=cluster_spec,
          task_type=task_type,
          task_id=task_id,
          session_config=session_config,
          rpc_layer=rpc_layer,
          environment=environment)
    if task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
      if strategy.extended.experimental_between_graph:
        # All jobs run `worker_fn` if between-graph.
        return _run_single_worker(worker_fn, strategy, cluster_spec, task_type,
                                  task_id, session_config, rpc_layer)
      else:
        # Only one node runs `worker_fn` if in-graph.
        context = _WorkerContext(strategy, cluster_spec, task_type, task_id)
        if context.is_chief:
          return _run_single_worker(worker_fn, strategy, cluster_spec, None,
                                    None, session_config, rpc_layer)
        else:
          server.join()
    elif task_type == _TaskType.EVALUATOR:
      return _run_single_worker(eval_fn, eval_strategy, cluster_spec, task_type,
                                task_id, session_config, rpc_layer)
    else:
      if task_type != _TaskType.PS:
        raise ValueError("Unexpected task_type: %r" % task_type)
      server.join()
