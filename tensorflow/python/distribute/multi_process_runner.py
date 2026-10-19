# [MODULE DOC]
# Multi-process runner used by distributed-training tests to simulate a cluster. It launches one
# child process per task in a cluster description, sets the cluster-config environment variable
# in each, streams or collects their output, supervises them with a watchdog thread, and reports
# return values or re-raises errors in the parent. It also provides a pool-based variant that
# reuses worker processes, a barrier accessor for child tasks, a shared multiprocessing manager,
# and the test entry point that must be used by test files relying on this runner.
# Exposes: MultiProcessRunner, MultiProcessPoolRunner, run, get_barrier, manager, test_main,
# is_oss, and the error types for timeouts, unexpected exits and missing initialization (several
# are exported under the internal distribute namespace).
# Import-time side effects: tries to import the optional dill and tblib packages (and installs
# tblib pickling support so tracebacks can cross processes), and creates a weak set of active pool
# runners plus a lock for the lazily created manager. Not a helper module; it is a test utility.
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
"""Multi-process runner for testing purpose."""

import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time
import unittest
import weakref

from absl import logging
import six
from six.moves import queue as Queue

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.util.tf_export import tf_export

multiprocessing = multi_process_lib.multiprocessing

# pylint: disable=g-import-not-at-top
try:
  # `faulthandler` is not available in py2.
  import faulthandler
except ImportError:
  faulthandler = None

# TODO(b/150264776): Remove after resolving CI issue.
try:
  import dill
except ImportError:
  dill = None

# TODO(b/150264776): Remove after resolving CI issue.
try:
  import tblib.pickling_support
  # For pickling traceback objects.
  tblib.pickling_support.install()
except ImportError:
  pass


# _ProcessStatusInfo contains process status information. When is_successful
# attribute is True, the subprocess has ended successfully, or if False, the
# exception stack trace info is stored in exc_info to pass on to parent process
# to be re-raised.
_ProcessStatusInfo = collections.namedtuple(
    '_ProcessStatusInfo',
    ['task_type', 'task_id', 'is_successful', 'exc_info', 'return_value'])

# Information returned from a successful MultiProcessRunner run.
MultiProcessRunnerResult = collections.namedtuple('MultiProcessRunnerResult',
                                                  ['return_value', 'stdout'])

# visible_gpus: If not None, CUDA_VISIBLE_DEVICES is set to visible_gpus.
TestEnvironment = collections.namedtuple('TestEnvironment', [
    'task_type', 'task_id', 'cluster_spec', 'rpc_layer', 'grpc_fail_fast',
    'v2_enabled', 'executing_eagerly', 'visible_gpus'
])

# Resources for communication between worker processes and the main process.
#
# `process_status_queue` is used by `multi_process_runner` internally for
#   communication from subprocesses to the parent process for whether it's been
#   successful, and if not what the error stack trace is.
# `parent_to_sub_queue` is used for communications from parent to subprocess.
#   Currently this is only used to terminate subprocesses.
# TODO(rchao): Remove this once subprocess is terminated by SIGKILL.
# `streaming_pipe_w` is to stream stdout and stderr from subprocesses to parent
#   process.
# `barrier` is a barrier for the party of all subprocesses.
Resources = collections.namedtuple('Resources', [
    'process_status_queue', 'parent_to_sub_queue', 'streaming_pipe_w', 'barrier'
])

# Default time out sec is selected so that it's handled before the default
# "medium" timeout of the test runs.
_DEFAULT_TIMEOUT_SEC = 200

# The timeout in seconds to wait to force kill a child process. When a child
# process times out we first try to SIGTERM it so that it has a chance to dump
# stacktraces. However dumping stacktrace can take a long time.
_FORCE_KILL_WAIT_SEC = 30


class MultiProcessRunner(object):
  # [CLASS DOC]
  # Test utility that starts one child process per task of a cluster description to simulate a
  # multi-worker cluster. Holds the user function and its arguments, the cluster description, RPC
  # layer, run-time limit, output streaming/collection options, restart and chief-dependence
  # policies, GPU sharing settings, a lock-guarded mapping from (task type, task id) to process,
  # a set of deliberately terminated tasks, reader threads, a watchdog thread, and manager-backed
  # queues and a barrier used for communication between parent and children.
  # Created directly or through the run function; used by calling start (or start_in_process_as),
  # optionally terminating tasks, then join to obtain results. Not thread-safe. Children inherit
  # the TF2 and eager-execution behavior of the parent.
  """A utility class to start multiple processes to simulate a cluster.

  We need to use multiple processes to simulate a cluster in TF 2.0 tests
  because TF 2.0 has some process-global data structures that have to be
  separated by processes. We also need child processes to test out our fault
  tolerance because shutting down a standard TensorFlow server within its
  process is not supported.

  Note: the main test program that uses this runner class must run main program
  via `test_main` defined in this file. Using this runner in non-test binaries
  is not supported yet.

  This class is not thread-safe. Child processes will inherit TF2 behavior flag.
  """

  def __init__(self,
               fn,
               cluster_spec,
               rpc_layer=None,
               max_run_time=None,
               grpc_fail_fast=None,
               stream_output=True,
               return_output=False,
               use_dill_for_args=True,
               daemon=False,
               dependence_on_chief=True,
               auto_restart=False,
               share_gpu=True,
               args=None,
               kwargs=None):
    # [FUNCTION DOC]
    # Constructor. Parameters: fn (callable run in every child), cluster_spec (mapping from task
    # type to list of addresses; required), rpc_layer (optional, defaults to grpc), max_run_time
    # (optional seconds before children are force-terminated through an alarm), grpc_fail_fast
    # (optional flag exported to children), stream_output (print child output in parent, default
    # true), return_output (collect child output for the result, default false), use_dill_for_args
    # (serialize args with dill, default true), daemon (start children as daemons), dependence_on_chief
    # (terminate cluster when chief exits, default true), auto_restart (restart children that
    # exit non-zero), share_gpu (default true; otherwise GPUs are assigned round-robin), args and
    # kwargs (optional, default to empty).
    # Raises a skip-test exception if thread sanitizer is enabled, a value error if more than one
    # chief exists or fn is not callable, and the not-initialized error if test_main was not used.
    # Side effects: queries physical GPUs, records TF2/eager state, creates (or reuses) the shared
    # manager and creates status/parent-to-child/streaming queues and a barrier sized to the total
    # number of tasks. Starts no processes. Returns nothing.
    """Instantiation of a `MultiProcessRunner`.

    Args:
      fn: Function to be run on child processes. This will be run on processes
        for all task types.
      cluster_spec: Dict for cluster spec. The utility function
        `tf.__internal__.distribute.multi_process_runner.create_cluster_spec`
        can be conveniently used to create such dict. The following is an
        example of cluster with three workers and two ps's.
        {"worker": ["worker0.example.com:2222",
                    "worker1.example.com:2222",
                    "worker2.example.com:2222"],
         "ps": ["ps0.example.com:2222",
                "ps1.example.com:2222"]}
      rpc_layer: RPC layer to use. Default value is 'grpc'.
      max_run_time: `None` or integer. If not `None`, child processes are forced
        to exit at approximately this many seconds after this utility is called.
        We achieve this through `signal.alarm()` api. Note that this is best
        effort at Python level since Python signal handler does not get executed
        when it runs lower level C/C++ code. So it can be delayed for
        arbitrarily long time. If any of the child process is still running when
        `max_run_time` is up, they will be force-terminated and an
        `UnexpectedSubprocessExitError` may be raised. If `None`, child
        processes are not forced to exit.
      grpc_fail_fast: Whether GRPC connection between processes should fail
        without retrying. Defaults to None, in which case the environment
        variable is not explicitly set.
      stream_output: True if the output/error from the subprocesses should be
        streamed to be printed in parent process' log. Defaults to True.
      return_output: If True, the output/error from the subprocesses should be
        collected to be attached to the resulting namedtuple returned from
        `join()`. The list of output can be retrieved via `stdout` attribute.
        Defaults to False.
      use_dill_for_args: Whether to use dill to pickle `args` and `kwargs`. dill
        can pickle more objects, but doesn't work with types in
        `multiprocessing` library like `Mutex`.
      daemon: Whether to start processes as daemons.
      dependence_on_chief: Whether to terminates the cluster if the chief exits.
        If auto_restart is True, it only terminates the cluster if the chief
        exits with a zero exit code.
      auto_restart: Whether to automatically restart processes that exit with
        non-zero exit code.
      share_gpu: Whether to share GPUs among workers. If False, each worker is
        assigned different GPUs in a roundrobin fashion. This should be True
        whenever possible for better test execution coverage; some situations
        that need it to be False are tests that runs NCCL.
      args: Positional arguments to be sent to `fn` run on subprocesses.
      kwargs: Keyword arguments to be sent to `fn` run on subprocesses.

    Raises:
      RuntimeError: if `multi_process_runner.test_main()` is not called.
      ValueError: if there are more than one chief in the `cluster_spec`.
      SkipTest: if thread sanitizer is enabled (which is incompatible with MPR).
    """
    if test_util.is_tsan_enabled():
      raise unittest.SkipTest(
          'ThreadSanitizer is not compatible with MultiProcessRunner.')

    assert cluster_spec is not None
    if 'chief' in cluster_spec and len(cluster_spec['chief']) > 1:
      raise ValueError('If chief exists in the cluster, there must be at most '
                       'one chief. Current `cluster_spec` has {} chiefs.'
                       .format(len(cluster_spec['chief'])))
    _check_initialization()
    if not callable(fn):
      raise ValueError('fn is not a callable')

    self._fn = fn
    self._cluster_spec = cluster_spec
    self._rpc_layer = rpc_layer or 'grpc'
    self._max_run_time = max_run_time
    self._grpc_fail_fast = grpc_fail_fast
    self._stream_output = stream_output
    # TODO(rchao): Revisit return_output argument to consider other solution.
    self._return_output = return_output
    self._dependence_on_chief = dependence_on_chief
    self._use_dill_for_args = use_dill_for_args
    self._daemon = daemon
    self._auto_restart = auto_restart
    self._args = args or ()
    self._kwargs = kwargs or {}

    self._share_gpu = share_gpu
    self._total_gpu = len(context.context().list_physical_devices('GPU'))

    # Child processes should have the same v2 and eager behavior.
    self._v2_enabled = tf2.enabled()
    self._executing_eagerly = context.executing_eagerly()

    self._joined = False
    self._process_lock = threading.Lock()
    # Guarded by self._process_lock.
    self._processes = {}
    # Record which processes are terminated. Due to a bug in Python<3.7,
    # terminated processes return 255 exit code, which should cause an exception
    # in join().
    # https://bugs.python.org/issue30589
    # Guarded by self._process_lock.
    self._terminated = set()
    self._reading_threads = []

    self._manager = manager()
    self._process_status_queue = self._manager.Queue()
    self._parent_to_sub_queue = self._manager.Queue()
    parties = sum(len(addresses) for addresses in self._cluster_spec.values())
    self._barrier = self._manager.Barrier(parties)

    # We use a queue to collect outputs from worker processes since it's thread
    # safe.
    self._streaming_queue = self._manager.Queue()

    self._watchdog_thread = None

  def set_args(self, args=None, kwargs=None):
    # [FUNCTION DOC]
    # Replaces the positional and keyword arguments later passed to the user function. Each is
    # optional; a missing or empty value keeps the existing one. Returns nothing.
    self._args = args or self._args
    self._kwargs = kwargs or self._kwargs

  def _continuously_readline_from_sub(self, pipe_r, task_type, task_id):
    # [FUNCTION DOC]
    # HELPER: reader-thread body for MultiProcessRunner that relays child output to the parent.
    # Reads lines from the read end of a child's pipe until it closes (inputs: the pipe, task type,
    # task id), prefixing each line with a padded task label. Prints it if output streaming is
    # enabled and queues it for the final result if output return is enabled. Returns nothing.
    """Function to continuously read lines from subprocesses."""
    with os.fdopen(pipe_r.fileno(), 'r', closefd=False) as reader:
      for line in reader:
        task_string = '[{}-{}]:'.format(task_type, task_id)
        formatted_line = '{} {}'.format(task_string.ljust(14), line)
        if self._stream_output:
          # TODO(rchao): Use a lock here to ensure the printed lines are not
          # broken.
          print(formatted_line, end='', flush=True)
        if self._return_output:
          self._streaming_queue.put(formatted_line)

  def _start_subprocess_and_reading_thread(self,
                                           task_type,
                                           task_id,
                                           cluster_spec=None,
                                           fn=None,
                                           args=None,
                                           kwargs=None):
    # [FUNCTION DOC]
    # HELPER: internal routine of MultiProcessRunner used by start and the watchdog to launch a task.
    # Inputs: task type, task id, optional cluster spec / function / args / kwargs (defaults to
    # the ones given at construction; if no function is given all three come from the instance).
    # Raises a skip-test exception when dill is unavailable. When GPU sharing is off, computes a
    # round-robin list of visible GPUs. Builds the test environment and resources (queues, write
    # end of a new pipe, barrier), pickles the function (always) and arguments (if enabled) with
    # dill, creates and starts a child process wrapper, records it in the process mapping, clears
    # any terminated mark for the task, starts a reader thread for its output, and starts the
    # watchdog thread if absent or dead. Returns nothing. Caller is expected to hold the lock.
    """Start a subprocess and a thread the reads lines from the subprocess."""

    if dill is None:
      raise unittest.SkipTest(
          'TODO(b/150264776): Resolve dependency issue in CI')

    cluster_spec = cluster_spec or self._cluster_spec
    visible_gpus = None
    if not self._share_gpu and self._total_gpu > 0:
      # Assign GPUs in a roundrobin fashion.
      id_in_cluster = multi_worker_util.id_in_cluster(cluster_spec, task_type,
                                                      task_id)
      worker_count = multi_worker_util.worker_count(cluster_spec, task_type)
      visible_gpus = list(range(id_in_cluster, self._total_gpu, worker_count))

    test_env = TestEnvironment(
        task_type=task_type,
        task_id=task_id,
        cluster_spec=cluster_spec,
        rpc_layer=self._rpc_layer,
        grpc_fail_fast=self._grpc_fail_fast,
        v2_enabled=self._v2_enabled,
        executing_eagerly=self._executing_eagerly,
        visible_gpus=visible_gpus,
    )
    pipe_r, pipe_w = multiprocessing.Pipe(duplex=False)
    resources = Resources(
        process_status_queue=self._process_status_queue,
        parent_to_sub_queue=self._parent_to_sub_queue,
        streaming_pipe_w=pipe_w,
        barrier=self._barrier,
    )
    if fn is None:
      fn, args, kwargs = self._fn, self._args, self._kwargs
    # Always use dill to pickle fn so that we support more callable
    # types, e.g. lambda.
    fn = dill.dumps(fn, dill.HIGHEST_PROTOCOL)
    if self._use_dill_for_args:
      args = dill.dumps(args, dill.HIGHEST_PROTOCOL)
      kwargs = dill.dumps(kwargs, dill.HIGHEST_PROTOCOL)

    p = _Process(
        test_env=test_env,
        target=_ProcFunc(),
        args=(resources, test_env, fn, args, kwargs, self._use_dill_for_args),
        daemon=self._daemon)
    logging.info(
        '[%s-%d] Starting subprocess (PID will be assigned)', task_type, task_id
    )
    start_time = time.time()
    p.start()
    startup_time = time.time() - start_time
    logging.info(
        '[%s-%d] Subprocess started in %.2fs, PID: %d',
        task_type,
        task_id,
        startup_time,
        p.pid,
    )
    self._processes[(task_type, task_id)] = p
    self._terminated.discard((task_type, task_id))

    # For each subprocess, we dedicate a thread continuously reading lines
    # from them.
    thread = threading.Thread(  # pylint: disable=unexpected-keyword-arg
        target=self._continuously_readline_from_sub,
        args=(pipe_r, task_type, task_id))
    thread.start()
    self._reading_threads.append(thread)

    if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
      self._watchdog_thread = threading.Thread(target=self._process_watchdog)
      self._watchdog_thread.start()

  def start(self):
    # [FUNCTION DOC]
    # Starts one child process per task in the cluster description, under the process lock.
    # Raises a value error if processes were already started or if join was already called.
    # If a maximum run time was set, installs an alarm-signal handler in the parent that terminates
    # all children and schedules the alarm (a process-global side effect). Returns nothing.
    """Starts processes, one for each task in `cluster_spec`.

    Note that this is best effort by the applicable multiprocessing library,
    and it may take up to seconds for a subprocess to be successfully started.
    """
    with self._process_lock:
      if self._processes:
        raise ValueError('MultiProcessRunner already started.')
      if self._joined:
        raise ValueError('cannot start new processes after'
                         'MultiProcessRunner.join() is called')

      for task_type, addresses in self._cluster_spec.items():
        for task_id, _ in enumerate(addresses):
          self._start_subprocess_and_reading_thread(task_type, task_id)

    # TODO(rchao): Remove the need of using SIGALRM if possible. At this time,
    # without this the tests become very flaky.
    if self._max_run_time is not None:

      def handler(signum, frame):
        # [FUNCTION DOC]
        # HELPER: alarm-signal callback installed by start. Ignores its signal-number and frame
        # inputs and terminates all child processes. Returns nothing.
        del signum, frame
        self.terminate_all()

      signal.signal(signal.SIGALRM, handler)
      signal.alarm(self._max_run_time)

  def start_in_process_as(self, as_task_type, as_task_id):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the already-started check happens before taking the lock, unlike start.
    # Like start, but the task identified by as_task_type and as_task_id runs in the calling
    # (main) process and blocks until the user function returns. All other tasks start as child
    # processes. Raises a value error if already started or joined. Sets the cluster-config
    # environment variable in the current process for that task (side effect), then calls the user
    # function with the stored args and kwargs. Unlike start, the maximum run time is not applied
    # and the function's return value is discarded. Returns nothing.
    """Start the processes, with the specified task run in main process.

    This is similar to `start()` except that the task with task_type
    `as_task_type` and task_id `as_task_id` is run in the main process.
    This method is particularly useful when debugging tool such as `pdb` is
    needed in some specific task. Note that since this method is blocking until
    that specific task exits, additional actions would need a thread to be
    called:

    ```python
    def fn():
      # user code to be run
      import pdb; pdb.set_trace()

    def follow_ups():
      time.sleep(5)
      mpr.start_single_process(
          task_type='evaluator',
          task_id=0)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1))
    threading.Thread(target=follow_ups).start()
    mpr.start_in_process_as(as_task_type='chief', as_task_id=0)
    mpr.join()
    ```

    Note that if `return_output=True`, the logs/stdout by task
    run by the main process is not available in result.stdout.

    Args:
      as_task_type: The task type to be run in the main process.
      as_task_id: The task id to be run in the main process.
    """
    if self._processes:
      raise ValueError('MultiProcessRunner already started.')
    with self._process_lock:
      if self._joined:
        raise ValueError('cannot start new processes after'
                         'MultiProcessRunner.join() is called')
      for task_type, addresses in self._cluster_spec.items():
        for task_id, _ in enumerate(addresses):
          if not (task_type == as_task_type and task_id == as_task_id):
            self._start_subprocess_and_reading_thread(task_type, task_id)

    _set_tf_config(as_task_type, as_task_id, self._cluster_spec,
                   self._rpc_layer)
    self._fn(*self._args, **self._kwargs)

  def start_single_process(self,
                           task_type,
                           task_id,
                           cluster_spec=None,
                           fn=None,
                           args=None,
                           kwargs=None):
    # [FUNCTION DOC]
    # Starts one child process for the given task type and task id under the process lock.
    # Optional cluster_spec, fn, args and kwargs override the ones given at construction (args and
    # kwargs default to empty when omitted). Raises a value error if join was already called.
    # Only the newly started process sees an updated cluster spec. Returns nothing.
    """Starts a single process.

    This starts a process in the cluster with the task type, task id, and the
    process function (`fn`). If process function is `None`, the function
    provided at `__init__` will be used. If `cluster_spec` is `None`, the
    cluster spec provided at `__init__` will be used.

    TODO(rchao): It is meant that all subprocesses will be updated with the new
    cluster spec, but this has yet to be implemented. At this time only the
    newly started subprocess picks up this updated cluster spec.

    Args:
      task_type: The task type.
      task_id: The task id.
      cluster_spec: The cluster spec to be used on the newly started
        process. If `None`, the cluster spec provided at `__init__` will be
        used.
      fn: The process function to be run on the newly started
        process. If specified, specify `args` and `kwargs` as well. If `None`,
        the function provided at `__init__` will be used.
      args: Optional positional arguments to be supplied in `fn`.
      kwargs: Optional keyword arguments to be supplied in `fn`.
    """
    with self._process_lock:
      if self._joined:
        raise ValueError('cannot start new processes after'
                         'MultiProcessRunner.join() is called')
      self._start_subprocess_and_reading_thread(
          task_type,
          task_id,
          cluster_spec=cluster_spec,
          fn=fn,
          args=args or (),
          kwargs=kwargs or {})

  def _queue_to_list(self, queue_to_convert):
    # [FUNCTION DOC]
    # HELPER: drains a queue without blocking. Input: a queue. Returns a list of all items that
    # could be fetched until the queue reported empty; the queue is emptied as a side effect.
    """Convert `queue.Queue` to `list`."""
    list_to_return = []
    # Calling `queue.empty()` is not reliable.
    while True:
      try:
        list_to_return.append(queue_to_convert.get(block=False))
      except Queue.Empty:
        break
    return list_to_return

  def _get_process_statuses(self):
    # [FUNCTION DOC]
    # HELPER: collects child status reports for join. Drains the status queue and returns a mapping
    # from (task type, task id) to the last status record received for that task. Consumes the
    # queue contents.
    # One worker may have multiple statuses. We only keep the last one.
    statuses = {}
    for status in self._queue_to_list(self._process_status_queue):
      statuses[(status.task_type, status.task_id)] = status
    return statuses

  def get_process_id(self, task_type, task_id):
    # [FUNCTION DOC]
    # Looks up the child for the given task type and task id under the lock. Returns its operating
    # system process id, or nothing if no such task was started.
    """Returns the subprocess id given the task type and task id."""
    with self._process_lock:
      p = self._processes.get((task_type, task_id), None)
    return p.pid if p else None

  def get_process_exit_code(self, task_type, task_id):
    # [FUNCTION DOC]
    # Returns the exit code of the child for the given task type and id, or nothing if it has not
    # exited yet. Raises a key error if no such task was started.
    """Returns the subprocess exit code given the task type and task id.

    Args:
      task_type: The task type.
      task_id: The task id.

    Returns:
      The subprocess exit code; `None` if the subprocess has not exited yet.

    Raises:
      KeyError: If the corresponding subprocess is not found with `task_type`
        and `task_id`.
    """
    with self._process_lock:
      p = self._processes[(task_type, task_id)]
    return p.exitcode if p else None

  def process_exists(self, task_type, task_id):
    # [FUNCTION DOC]
    # Returns true if the child for the task has not exited (its exit code is still unset), false
    # otherwise. Raises a key error if the task was never started.
    """Returns whether the subprocess still exists given the task type and id.

    Args:
      task_type: The task type.
      task_id: The task id.

    Returns:
      Boolean; whether the subprocess still exists. If the subprocess has
      exited, this returns False.
    """
    return self.get_process_exit_code(task_type, task_id) is None

  def _process_watchdog(self):
    # [FUNCTION DOC]
    # HELPER: body of the watchdog thread started by MultiProcessRunner. Loops once per second
    # under the process lock. If the chief has exited and dependence on the chief is on, and
    # either it exited with zero or auto restart is off, gives the others a few seconds to exit,
    # terminates all remaining processes, waits for them and ends. Otherwise, if auto restart is
    # on, restarts every process that exited non-zero and keeps looping. Ends when all processes
    # have exited. Takes no inputs and returns nothing; side effects are process termination,
    # restarts and logging.
    """Simulates a cluster management system.

    - If auto_restart is True, it restarts processes that exit with a non-zero
      exit code. Note that when join() times out it overrides auto_restart to
      False.
    - If dependence_on_chief is True, it terminates all processes once the chief
      exits. If auto_restart is also True, it only terminates all processes if
      the chief exit with a zero exit code, otherwise it restarts the chief.

    This runs in self._watchdog_thread.
    """
    while True:
      time.sleep(1)
      with self._process_lock:
        chief = self._processes.get(('chief', 0), None)
        # Terminate the cluster when _dependence_on_chief is True if either:
        # - chief has exited with zero exit code.
        # - chief has exited with non-zero exit code and self._auto_restart is
        #   False.
        if chief and self._dependence_on_chief and chief.exitcode is not None:
          if chief.exitcode == 0 or (not self._auto_restart):
            for p in self._processes.values():
              # Give other processes a chance to exit on their own.
              p.join(timeout=3)
            self._terminate_all()
            for p in self._processes.values():
              p.join()
            return

        # Auto restart failed processes if self._auto_restart is True.
        if self._auto_restart:
          has_failure = False
          for (task_type, task_id), p in self._processes.items():
            if p.exitcode is not None and p.exitcode != 0:
              has_failure = True
              logging.info('Restarting failed %s-%d', task_type, task_id)
              self._start_subprocess_and_reading_thread(task_type, task_id)
          if has_failure:
            continue

        # Exit the thread if all processes have exited at this point.
        if all(p.exitcode is not None for p in self._processes.values()):
          return

  def _reraise_if_subprocess_error(self, process_statuses):
    # [FUNCTION DOC]
    # HELPER: error propagation for join. Input: mapping of task to status record. For the first
    # unsuccessful status found, attaches the run result (return values and output) to the stored
    # exception as an attribute and re-raises that exception with its original traceback. Asserts
    # every entry is a status record. Returns nothing when all tasks succeeded.
    for process_status in process_statuses.values():
      assert isinstance(process_status, _ProcessStatusInfo)
      if not process_status.is_successful:
        process_status.exc_info[1].mpr_result = self._get_mpr_result(
            process_statuses)
        six.reraise(*process_status.exc_info)

  def join(self, timeout=_DEFAULT_TIMEOUT_SEC):
    # [FUNCTION DOC]
    # Waits for the watchdog thread (and thereby all children) to finish. Input: timeout in
    # seconds (integer or nothing for no timeout; default is the module default). Raises a value
    # error if timeout is not an integer (a zero or missing value skips the check) or if called
    # twice. On timeout: disables auto restart, sends a terminate signal, waits a fixed grace
    # period, then force-kills; re-raises any child error, otherwise raises the timeout error
    # carrying the partial result. On normal completion: re-raises child errors, raises the
    # unexpected-exit error for children with positive exit codes that were not deliberately
    # terminated, joins reader threads, clears the alarm, and returns a result with the list of
    # child return values and collected output lines.
    """Joins all the processes with timeout.

    If any of the subprocesses does not exit approximately after `timeout`
    seconds has passed after `join` call, this raises a
    `SubprocessTimeoutError`.

    Note: At timeout, it uses SIGTERM to terminate the subprocesses, in order to
    log the stack traces of the subprocesses when they exit. However, this
    results in timeout when the test runs with tsan (thread sanitizer); if tsan
    is being run on the test targets that rely on timeout to assert information,
    `MultiProcessRunner.terminate_all()` must be called after `join()`, before
    the test exits, so the subprocesses are terminated with SIGKILL, and data
    race is removed.

    Args:
      timeout: optional integer or `None`. If provided as an integer, and not
      all processes report status within roughly `timeout` seconds, a
      `SubprocessTimeoutError` exception will be raised. If `None`, `join` never
      times out.

    Returns:
      A `MultiProcessRunnerResult` object, which has two attributes,
      `return_value` and `stdout`. `return_value` always contains a list of
      return values from the subprocesses, although the order is not meaningful.
      If `return_output` argument is True at `__init__`, `stdout` is available
      that contains a list of all messages from subprocesses' stdout and stderr.

    Raises:
      SubprocessTimeoutError: if not all processes report status approximately
        within `timeout` seconds. When this is raised, a
        `MultiProcessRunnerResult` object can be retrieved by
        `SubprocessTimeoutError`'s mpr_result attribute, which has the same
        structure as above 'Returns' section describes.
      UnexpectedSubprocessExitError: If any of the subprocesses did not exit
        properly (for example, they exit on SIGTERM or SIGKILL signal). When
        this is raised, a `MultiProcessRunnerResult` object can be retrieved by
        `UnexpectedSubprocessExitError`'s mpr_result attribute, which has the
        same structure as above 'Returns' section describes. If `max_run_time`
        is not `None`, it is expected that some subprocesses may be
        force-killed when `max_run_time` is up, and this is raised in those
        cases.
      Exception: if there is an Exception propagated from any subprocess. When
        this is raised, a `MultiProcessRunnerResult` object can be retrieved by
        `UnexpectedSubprocessExitError`'s mpr_result attribute, which has the
        same structure as above 'Returns' section describes.
    """
    if timeout and not isinstance(timeout, int):
      raise ValueError('`timeout` must be an integer or `None`.')
    with self._process_lock:
      if self._joined:
        raise ValueError("MultiProcessRunner can't be joined twice.")
      self._joined = True

    self._watchdog_thread.join(timeout)
    if self._watchdog_thread.is_alive():
      # Timeout. Force termination to dump worker processes stack trace.
      with self._process_lock:
        self._auto_restart = False
      logging.error('Timeout when joining for child processes. Terminating...')
      self.terminate_all(sig=signal.SIGTERM)
      # Wait for the processes to terminate by themselves first, so they have a
      # chance to dump stacktraces. After _FORCE_KILL_WAIT_SEC, we SIGKILL them.
      self._watchdog_thread.join(_FORCE_KILL_WAIT_SEC)
      if self._watchdog_thread.is_alive():
        logging.error('Timeout when waiting for child processes to '
                      'print stacktrace. Sending SIGKILL...')
        self.terminate_all()
        self._watchdog_thread.join()
      process_statuses = self._get_process_statuses()
      self._reraise_if_subprocess_error(process_statuses)
      raise SubprocessTimeoutError(
          'One or more subprocesses timed out, where timeout was set to {}s. '
          'Please change the `timeout` argument for '
          '`MultiProcessRunner.join()` or `multi_process_runner.run()` '
          'if it should be adjusted.'.format(timeout),
          self._get_mpr_result(process_statuses))

    for (task_type, task_id), p in self._processes.items():
      logging.info('%s-%d exit code: %s', task_type, task_id, p.exitcode)

    process_statuses = self._get_process_statuses()
    self._reraise_if_subprocess_error(process_statuses)

    # Checking all the processes that are expected to exit properly.
    for (task_type, task_id), p in self._processes.items():
      # Successfully exiting process has exit code 0. We ignore processes that
      # are terminated.
      assert p.exitcode is not None
      if (p.exitcode > 0 and (task_type, task_id) not in self._terminated):
        raise UnexpectedSubprocessExitError(
            'Subprocess %s-%d exited with exit code %s. See logs for details.'
            % (task_type, task_id, p.exitcode),
            self._get_mpr_result(process_statuses))

    logging.info('Joining log reading threads.')
    for thread in self._reading_threads:
      thread.join()
    logging.info('Joined log reading threads.')

    # Clear the alarm.
    signal.alarm(0)

    return self._get_mpr_result(process_statuses)

  def _get_mpr_result(self, process_statuses):
    # [FUNCTION DOC]
    # HELPER: builds the result record for join and error paths. Input: mapping of task to status
    # record. Drains the streaming output queue into a list and gathers all non-null return
    # values. Returns a result with return values and output lines.
    stdout = self._queue_to_list(self._streaming_queue)
    return_values = []
    for process_status in process_statuses.values():
      if process_status.return_value is not None:
        return_values.append(process_status.return_value)
    return MultiProcessRunnerResult(stdout=stdout, return_value=return_values)

  def terminate(self, task_type, task_id):
    # [FUNCTION DOC]
    # Asks one child to terminate. Inputs: task type and task id. Raises a value error if no such
    # task exists. Marks the task as deliberately terminated, sends a terminate message through the
    # parent-to-child queue, and waits for the process to exit. Returns nothing.
    # KNOWN ISSUE: the docstring mentions restarting when auto restart is on, but this method itself
    # does no restart; that is left to the watchdog thread.
    """Terminates the process with `task_type` and `task_id`.

    If auto_retart=True, the terminated task will be restarted unless the chief
    has already exited with zero exit code.

    Args:
      task_type: the task type.
      task_id: the task id.

    """
    with self._process_lock:
      p = self._processes.get((task_type, task_id), None)
      if p is None:
        raise ValueError('{}-{} does not exist'.format(task_type, task_id))
      self._terminated.add((task_type, task_id))
      # TODO(crccw): change to use Process.terminate() as well.
      self._parent_to_sub_queue.put('terminate {} {}'.format(
          task_type, task_id))
      p.join()

  def _terminate_all(self, sig=None):
    # [FUNCTION DOC]
    # HELPER: lock-free implementation of terminate_all; the caller must hold the process lock.
    # Input: optional signal (default is the kill signal, or the terminate signal on platforms
    # without it). Sends it to each process that has not exited, marking it as deliberately
    # terminated, and logs; tolerates processes that no longer exist. Returns nothing.
    """Terminates all subprocesses.

    The caller is required to hold self._process_lock.

    Args:
      sig: the signal used to terminate the process. The default is SIGKILL.
    """

    # Use SIGKILL as default. In systems where that's unavailable such as
    # windows, use SIGTERM.
    sig = sig or getattr(signal, 'SIGKILL', signal.SIGTERM)
    for (task_type, task_id), p in self._processes.items():
      if p.exitcode is not None:
        logging.info('%s-%d has already exited. Not terminating.', task_type,
                     task_id)
        continue
      try:
        os.kill(p.pid, sig)
        self._terminated.add((task_type, task_id))
        logging.info('%s-%d terminated with signal %r.', task_type, task_id,
                     sig)
      except ProcessLookupError:
        logging.info('Attempting to kill %s-%d but it does not exist.',
                     task_type, task_id)

  def terminate_all(self, sig=None):
    # [FUNCTION DOC]
    # Terminates all child processes while holding the process lock. Optional input: the signal to
    # send (defaults to kill, or terminate where kill is unavailable). Returns nothing.
    """Terminates all subprocesses."""
    with self._process_lock:
      self._terminate_all(sig)


class _Process(multi_process_lib.Process):
  # [CLASS DOC]
  # HELPER: process class used internally by MultiProcessRunner. Extends the library process type
  # so that, in the child, environment variables (fail-fast flag, visible GPUs, cluster config) are
  # set before the target runs. Holds the test environment record and the original run method.
  """A modified `multiprocessing.Process` that can set up environment variables."""

  # TODO(crccw): consider moving other logics in _ProcFunc to _Process.

  def __init__(self, test_env, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Input: the test environment record plus any keyword arguments forwarded to the
    # base process constructor. Saves the environment and the original run method, then replaces
    # the instance's run with the environment-setting wrapper. Returns nothing.
    super(_Process, self).__init__(**kwargs)
    self._test_env = test_env
    self._actual_run = getattr(self, 'run')
    self.run = self._run_with_setenv

  def _run_with_setenv(self):
    # [FUNCTION DOC]
    # HELPER: replacement run method of the child process. Sets the fail-fast environment variable
    # if specified, sets the visible-GPU environment variable (comma-joined) if any GPUs were
    # assigned, writes the cluster-config variable for the task, then runs the original run method
    # and returns its result. Environment changes happen first because setting them is not
    # thread-safe.
    # We need to set environment variables before doing anything because
    # setenv() is not thread-safe.
    test_env = self._test_env
    if test_env.grpc_fail_fast is not None:
      os.environ['GRPC_FAIL_FAST'] = str(test_env.grpc_fail_fast)
    if test_env.visible_gpus:
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
          [str(i) for i in test_env.visible_gpus])
    _set_tf_config(test_env.task_type, test_env.task_id, test_env.cluster_spec,
                   test_env.rpc_layer)
    return self._actual_run()


class _ProcFunc(object):
  # [CLASS DOC]
  # HELPER: callable object used as the target of each child process. Calling it performs the
  # child-side setup (dill loading, fault handler, stream redirection, message-checking thread,
  # TF2 behavior, eager or graph mode), runs the user function in a contained way, reports status
  # to the parent and exits. Holds the shared resources record once called.
  """Represents a callable to run in a subprocess."""

  @contextlib.contextmanager
  def _runtime_mode(self, executing_eagerly):
    # [FUNCTION DOC]
    # HELPER: context manager that enters eager mode if the flag is true, otherwise graph mode,
    # for the duration of the with block. Yields nothing.
    if executing_eagerly:
      with context.eager_mode():
        yield
    else:
      with context.graph_mode():
        yield

  def _message_checking_func(self, task_type, task_id):
    # [FUNCTION DOC]
    # HELPER: child-side thread body that polls the parent-to-child queue. Inputs: this task's
    # type and id. Raises a value error for any message not starting with the terminate word.
    # A terminate message addressed to this task ends the loop; messages for other tasks are put
    # back after a one second pause; an empty queue causes a short sleep. On termination it
    # reports a successful status with no return value to the parent and force-exits the process
    # immediately with exit code one. Never returns normally.
    """A function that regularly checks messages from parent process."""
    # TODO(rchao): Remove this once parent uses SIGKILL to terminate subprocess.
    while True:
      try:
        message = self._resources.parent_to_sub_queue.get(block=False)

        # Currently the only possible message is termination.
        if not message.startswith('terminate'):
          raise ValueError('Unrecognized message: {}'.format(message))

        if message == 'terminate {} {}'.format(task_type, task_id):
          break
        else:
          # If the message is not targeting this process, put it back to the
          # queue.
          self._resources.parent_to_sub_queue.put(message)
          time.sleep(1)
      except Queue.Empty:
        time.sleep(0.1)
    self._resources.process_status_queue.put(
        _ProcessStatusInfo(
            task_type=task_type,
            task_id=task_id,
            is_successful=True,
            exc_info=None,
            return_value=None))
    # `os._exit(1)` is used to more reliably terminate a subprocess.
    os._exit(1)  # pylint: disable=protected-access

  def _close_streaming(self):
    # [FUNCTION DOC]
    # HELPER: flushes and closes standard output and error and the write end of the streaming
    # pipe so the parent's reader thread can finish early. Returns nothing.
    """Close stdout, stderr and streaming pipe.

    We need to explicitly close them since Tensorflow may take a while to exit,
    so that the reading threads in the main process can exit more quickly.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout.close()
    sys.stderr.close()
    self._resources.streaming_pipe_w.close()

  def __call__(self, resources, test_env, fn, args, kwargs, use_dill_for_args):
    # [FUNCTION DOC]
    # Entry point run inside each child process. Inputs: the resources record (queues, pipe,
    # barrier), the test environment record, the dill-pickled function, the arguments and keyword
    # arguments (pickled if the flag is set), and that flag. Stores resources, publishes the
    # barrier in a module global, unpickles everything, enables the fault handler (and registers
    # it for the terminate signal) where available, sets logging to stderr, redirects standard
    # output and error to the streaming pipe, logs the task and cluster config, starts a daemon
    # thread to watch for parent messages, enables TF2 behavior if the parent had it, then runs
    # the function in the chosen eager/graph mode via the contained runner and puts the resulting
    # status on the status queue. On failure it re-raises the exception for logging; on success
    # it closes the streams and exits the process with code zero.
    """The wrapper function that actually gets run in child process(es)."""

    global _barrier

    self._resources = resources
    _barrier = self._resources.barrier
    fn = dill.loads(fn)
    if use_dill_for_args:
      args = dill.loads(args)
      kwargs = dill.loads(kwargs)

    if faulthandler is not None:
      faulthandler.enable()
      faulthandler.register(signal.SIGTERM, chain=True)

    # All logging should go to stderr to be streamed to the main process.
    logging.set_stderrthreshold(logging.DEBUG)

    # Assign sys.stdout and sys.stderr as duplicates of `streaming_pipe_w` so
    # print() and logging.*() write directly to `streaming_pipe_w`.
    # Unfortunately since we cannot prepend task_type and task_id information to
    # the streamed logs we will need a thread per subprocess to distinguish
    # where the piece of message is from.
    os.dup2(resources.streaming_pipe_w.fileno(), sys.stdout.fileno())
    os.dup2(resources.streaming_pipe_w.fileno(), sys.stderr.fileno())

    pid = os.getpid()
    logging.info('Subprocess with PID %d (%s, %d) is now being started.', pid,
                 test_env.task_type, test_env.task_id)
    logging.info('TF_CONFIG: %r', os.environ['TF_CONFIG'])

    # The thread will be dedicated to checking messages from the parent process.
    threading.Thread(  # pylint: disable=unexpected-keyword-arg
        target=self._message_checking_func,
        args=(test_env.task_type, test_env.task_id),
        daemon=True).start()

    if test_env.v2_enabled:
      v2_compat.enable_v2_behavior()

    with self._runtime_mode(test_env.executing_eagerly):
      info = _run_contained(test_env.task_type, test_env.task_id, fn, args,
                            kwargs)
      self._resources.process_status_queue.put(info)

      # Re-raise the exception in addition to reporting it to the parent
      # process, so that even if `--test_timeout` flag is set and the
      # error doesn't make it to be shown in parent process before bazel's
      # timeout, the log would still show what happens in this subprocess,
      # instead of silently suppressing the error due to early bazel
      # timeout. Raising an error in the subprocess produces stack trace in
      # the log, but the program continues running.
      if not info.is_successful:
        six.reraise(*info.exc_info)

      self._close_streaming()

    # Exit with code 0 as it's considered successful exit at this point.
    sys.exit(0)


# Active MultiProcessPoolRunner. We need to shut them down when the program
# exits, and this is by setting the `tearDownModule` of the module containing
# `__main__`. Note this it set in both the parent process and the subprocesses.
_active_pool_runners = weakref.WeakSet()


def _shutdown_all_pool_runners():
  # [FUNCTION DOC]
  # HELPER: shuts down every live pool runner in the module's weak set. Installed into the test
  # module's module-teardown hook by test_main. Returns nothing.
  for pool in _active_pool_runners:
    pool.shutdown()


def is_oss():
  # [FUNCTION DOC]
  # Returns true if the first command-line argument mentions the bazel build tool (a heuristic for
  # the open-source environment), false otherwise. Reads global process arguments.
  """Returns whether the test is run under OSS."""
  return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]


class MultiProcessPoolRunner(object):
  # [CLASS DOC]
  # Test utility holding a long-lived pool of worker processes (one per task) so repeated runs
  # avoid TensorFlow's start-up cost. State: cluster description, optional initializer, GPU
  # sharing flag, a mapping from task to parent-side pipe connection, and an underlying
  # MultiProcessRunner started lazily on the first run. Registers itself in a module-wide weak set
  # so test_main can shut it down at module teardown. Use run to execute functions on all tasks.
  """A utility class to start a process pool to simulate a cluster.

  It's similar to MultiProcessRunner, but uses a pool of processes to avoid the
  expensive initialization cost of Tensorflow.
  """

  def __init__(self, cluster_spec, initializer=None, share_gpu=True):
    # [FUNCTION DOC]
    # Constructor. Inputs: cluster_spec (mapping from task type to addresses), initializer
    # (optional callable run once at worker start-up) and share_gpu (default true). Registers the
    # instance in the global weak set of pool runners and initializes empty connection and runner
    # state; starts nothing. Returns nothing. Nothing is validated here despite the docstring's
    # listed errors.
    """Creates a multi-process pool runner.

    Args:
      cluster_spec: Dict for cluster spec. The following is an example of
        cluster with three workers.
        {"worker": ["worker0.example.com:2222",
                    "worker1.example.com:2222",
                    "worker2.example.com:2222"]}
      initializer: a callable to called at the startup of worker processes.
      share_gpu: Whether to share GPUs among workers. If False, each worker is
        assigned different GPUs in a roundrobin fashion.

    Raises:
      RuntimeError: if `multi_process_runner.test_main()` is not called.
      ValueError: if there are more than one chief in the `cluster_spec`.
    """
    _active_pool_runners.add(self)
    self._cluster_spec = cluster_spec
    self._initializer = initializer
    self._share_gpu = share_gpu
    self._conn = {}
    self._runner = None

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: shuts down the worker pool when the object is garbage collected. Returns nothing.
    self.shutdown()

  def shutdown(self):
    # [FUNCTION DOC]
    # Closes all parent-side connections (which makes workers leave their loops), clears the
    # connection mapping, and joins the underlying runner if any. A skip-test exception from the
    # join is re-raised; any other exception is logged and ignored. Afterwards the runner is
    # cleared so a later run restarts the pool. Returns nothing.
    """Shuts down the worker pool."""
    for conn in self._conn.values():
      conn.close()
    self._conn = {}
    if self._runner is not None:
      try:
        self._runner.join()
      except unittest.SkipTest:
        raise
      except Exception as e:  # pylint: disable=broad-except
        logging.exception(
            'Ignoring exception when shutting down MultiProcessPoolRunner: %s',
            e,
        )
      self._runner = None

  def _start(self):
    # [FUNCTION DOC]
    # HELPER: lazily starts the pool for run. Raises a skip-test exception if dill is missing.
    # Creates an underlying MultiProcessRunner with a no-op function (no dill for args), pickles
    # the initializer if given, and for each task creates a duplex pipe, stores the parent end,
    # and starts a single child running the pool worker loop with the task identity, initializer
    # and child end. Returns nothing.
    """Starts the worker pool."""
    # We need different arguments for different processes so we're passing a
    # no-op fn here and use start_single_process instead.

    if dill is None:
      raise unittest.SkipTest(
          'TODO(b/150264776): Resolve dependency issue in CI')

    self._runner = MultiProcessRunner(
        fn=lambda: None,
        cluster_spec=self._cluster_spec,
        use_dill_for_args=False,
        share_gpu=self._share_gpu)
    if self._initializer:
      initializer = dill.dumps(self._initializer, dill.HIGHEST_PROTOCOL)
    else:
      initializer = None
    for task_type, addresses in self._cluster_spec.items():
      for task_id, _ in enumerate(addresses):
        conn1, conn2 = multiprocessing.Pipe(duplex=True)
        self._conn[(task_type, task_id)] = conn1
        self._runner.start_single_process(
            task_type,
            task_id,
            fn=_pool_runner_worker,
            args=(task_type, task_id, initializer, conn2))

  def run(self, fn, args=None, kwargs=None):
    # [FUNCTION DOC]
    # Runs a function on every pool worker and collects results. Inputs: fn (callable), optional
    # positional args and keyword args (default empty). Checks initialization (not-initialized
    # error otherwise), starts the pool if needed, pickles fn with dill and sends the triple to
    # every worker. Then, per worker, waits for a reply in a helper thread with a per-worker cap of
    # 120 seconds within a 300 second total budget. Raises a runtime error on total timeout, worker
    # timeout (after logging process state), unexpected end-of-stream (after shutting down the
    # pool) or missing reply; re-raises other receive errors. Re-raises the first failed worker's
    # stored exception. Returns the list of non-null worker return values.
    # KNOWN ISSUE: the timeout diagnostic formats the exit code as a number although it is unset
    # for a running process, and the comment says the process is terminated but it is not.
    """Runs `fn` with `args` and `kwargs` on all jobs.

    Args:
      fn: The function to be run.
      args: Optional positional arguments to be supplied in `fn`.
      kwargs: Optional keyword arguments to be supplied in `fn`.

    Returns:
      A list of return values.
    """
    _check_initialization()
    # TODO(b/150264776): skip in OSS until it's implemented.
    multi_process_lib.Process()
    if self._runner is None:
      self._start()

    fn = dill.dumps(fn, dill.HIGHEST_PROTOCOL)
    for conn in self._conn.values():
      conn.send((fn, args or [], kwargs or {}))

    process_statuses = []
    total_timeout = 300  # 5 minutes total timeout for all processes
    start_time = time.time()

    for (task_type, task_id), conn in self._conn.items():
      remaining_time = total_timeout - (time.time() - start_time)
      if remaining_time <= 0:
        raise RuntimeError(
            f'Total timeout of {total_timeout} s exceeded '
            'while waiting for processes'
        )

      logging.info(
          'Waiting for the result from %s-%d (timeout: %.1fs)',
          task_type,
          task_id,
          remaining_time,
      )

      # Use threading to implement timeout for conn.recv()
      result_container = []
      error_container = []

      def recv_with_timeout():
        # [FUNCTION DOC]
        # HELPER: receiver thread body for run. Reads one message from the worker connection and
        # appends it to the shared result list; records an end-of-stream condition or any other
        # exception, tagged by kind, in the shared error list. Returns nothing.
        try:
          result = conn.recv()
          result_container.append(result)
        except EOFError as e:
          error_container.append(('eof', e))
        except Exception as e:
          error_container.append(('error', e))

      recv_thread = threading.Thread(target=recv_with_timeout)
      recv_thread.daemon = True
      recv_thread.start()
      recv_thread.join(
          timeout=min(remaining_time, 120)
      )  # Max 2 minutes run per process

      if recv_thread.is_alive():
        # Timeout occurred
        logging.info(
            '[%s-%d] Timeout waiting for process result after %d s',
            task_type,
            task_id,
            min(remaining_time, 120),
        )
        # Try to get process status for debugging
        with self._runner._process_lock if self._runner else threading.Lock():
          if self._runner and (task_type, task_id) in self._runner._processes:
            process = self._runner._processes[(task_type, task_id)]
            logging.info(
                '[%s-%d] Process state: exitcode=%d, pid=%d',
                task_type,
                task_id,
                process.exitcode,
                process.pid,
            )
        # Terminate the process if it is still running
        raise RuntimeError(
            f'Timeout waiting for {task_type}-{task_id} '
            f'after {min(remaining_time, 120):.1f}s. '
            'This often indicates a subprocess is stuck '
            'in initialization or pipe reading. '
            'Try increasing Docker container resources '
            'or check for deadlocks.'
        )

      if error_container:
        error_type, error = error_container[0]
        if error_type == 'eof':
          # This shouldn't happen due to exceptions in fn. This usually
          # means bugs in the runner.
          self.shutdown()
          raise RuntimeError(
              'Unexpected EOF. Worker process may have died. '
              'Please report a bug'
          )
        else:
          raise error

      if result_container:
        process_statuses.append(result_container[0])
      else:
        raise RuntimeError(f'No result received from {task_type}-{task_id}')

    return_values = []
    for process_status in process_statuses:
      assert isinstance(process_status, _ProcessStatusInfo)
      if not process_status.is_successful:
        six.reraise(*process_status.exc_info)
      if process_status.return_value is not None:
        return_values.append(process_status.return_value)

    return return_values


def _pool_runner_worker(task_type, task_id, initializer, conn):
  # [FUNCTION DOC]
  # HELPER: worker loop for MultiProcessPoolRunner child processes. Inputs: task type, task id,
  # a dill-pickled initializer (or nothing) and the child end of a duplex pipe. Runs the
  # initializer once if present, then repeatedly receives a (pickled function, args, kwargs)
  # triple, runs it in a contained way, flushes output and sends the status record back, until the
  # pipe reaches end-of-stream. Returns nothing.
  """Function that runs on the workers in a pool.

  It listens for callables to run and returns the result until `conn` is closed.
  It captures the exceptions during executing the callable and return it through
  `conn`.

  Args:
    task_type: the task type.
    task_id: the task index.
    initializer: a callable to execute during startup.
    conn: a multiprocessing.Connection object to listen for tasks and send
      results.
  """
  if initializer:
    initializer = dill.loads(initializer)
    initializer()
  while True:
    try:
      fn, args, kwargs = conn.recv()
    except EOFError:
      break
    fn = dill.loads(fn)
    info = _run_contained(task_type, task_id, fn, args, kwargs)
    sys.stdout.flush()
    sys.stderr.flush()
    conn.send(info)


def _run_contained(task_type, task_id, fn, args, kwargs):
  # [FUNCTION DOC]
  # HELPER: runs the user function and captures the outcome instead of propagating. Inputs: task
  # type, task id, function, positional args and keyword args. Returns a status record with
  # success flag, exception info (or nothing) and return value. Ordinary exceptions are captured
  # (a failed run has no return value); exit requests are not caught.
  """Runs `fn` with `args` and `kwargs`.

  The function returns _ProcessStatusInfo which captures the return value and
  the exception.

  Args:
    task_type: the task type.
    task_id: the task index.
    fn: the function to be run.
    args: optional positional arguments to be supplied in `fn`.
    kwargs: optional keyword arguments to be supplied in `fn`.

  Returns:
    a _ProcessStatusInfo.

  """
  is_successful = False
  return_value = None
  exc_info = None
  try:
    return_value = fn(*args, **kwargs)
    is_successful = True
    return _ProcessStatusInfo(
        task_type=task_type,
        task_id=task_id,
        is_successful=is_successful,
        exc_info=exc_info,
        return_value=return_value)

  # If `fn` ends up exiting with `sys.exit()`, the `SystemExit` is not
  # handled here.
  except Exception:  # pylint: disable=broad-except
    exc_info = sys.exc_info()
    return _ProcessStatusInfo(
        task_type=task_type,
        task_id=task_id,
        is_successful=is_successful,
        exc_info=exc_info,
        return_value=return_value)


@tf_export('__internal__.distribute.multi_process_runner'
           '.SubprocessTimeoutError',
           v1=[])
class SubprocessTimeoutError(RuntimeError):
  # [CLASS DOC]
  # Runtime error raised by join or run when child processes do not finish in time. Carries the
  # partial run result in an attribute. Exported under the internal distribute namespace.
  """An error that indicates there is at least one subprocess timing out.

  When this is raised, a namedtuple object representing the multi-process run
  result can be retrieved by
  `tf.__internal__.distribute.multi_process_runner.SubprocessTimeoutError`'s
  `mpr_result` attribute. See
  `tf.__internal__.distribute.multi_process_runner.run` for more information.
  """

  def __init__(self, msg, mpr_result):
    # [FUNCTION DOC]
    # Constructor. Inputs: message text and the run result record (return values and output).
    # Stores the result in an attribute. Returns nothing.
    super(SubprocessTimeoutError, self).__init__(msg)
    self.mpr_result = mpr_result


@tf_export('__internal__.distribute.multi_process_runner'
           '.UnexpectedSubprocessExitError',
           v1=[])
class UnexpectedSubprocessExitError(RuntimeError):
  # [CLASS DOC]
  # Runtime error raised by join when a child exits with a positive exit code without having been
  # deliberately terminated (or is force-killed at the maximum run time). Carries the run result
  # in an attribute. Exported under the internal distribute namespace.
  """An error indicating there is at least one subprocess with unexpected exit.

  When this is raised, a namedtuple object representing the multi-process run
  result can be retrieved by
  `tf.__internal__.distribute.multi_process_runner
  .UnexpectedSubprocessExitError`'s
  `mpr_result` attribute. See
  `tf.__internal__.distribute.multi_process_runner.run` for more information.
  """

  def __init__(self, msg, mpr_result):
    # [FUNCTION DOC]
    # Constructor. Inputs: message text and the run result record. Stores the result in an
    # attribute. Returns nothing.
    super(UnexpectedSubprocessExitError, self).__init__(msg)
    self.mpr_result = mpr_result


@tf_export(
    '__internal__.distribute.multi_process_runner.NotInitializedError', v1=[])
class NotInitializedError(RuntimeError):
  # [CLASS DOC]
  # Runtime error raised when the runner is used without the test entry point having initialized
  # multi-process support. Has no extra state. Exported under the internal distribute namespace.
  """An error indicating `multi_process_runner.run` is used without init.

  When this is raised, user is supposed to call
  `tf.__internal__.distribute.multi_process_runner.test_main()` within
  `if __name__ == '__main__':` block to properly initialize
  `multi_process_runner.run`.
  """
  pass


def _check_initialization():
  # [FUNCTION DOC]
  # HELPER: guard used by the runner classes and manager. Raises the not-initialized error with
  # instructions to call the test entry point if multi-process support is not initialized.
  # Returns nothing otherwise.
  if not multi_process_lib.initialized():
    raise NotInitializedError(
        '`multi_process_runner` is not initialized. '
        'Please call `tf.__internal__.distribute.multi_process_runner.'
        'test_main()` within `if __name__ == \'__main__\':` block '
        'in your python module to properly initialize '
        '`multi_process_runner`.')


def _set_tf_config(task_type, task_id, cluster_spec, rpc_layer=None):
  # [FUNCTION DOC]
  # HELPER: writes the cluster-config environment variable for the current process. Inputs: task
  # type, task id, cluster spec mapping, and optional RPC layer (included only if given). The
  # value is a serialized description with the cluster and the task type and index. Returns
  # nothing; mutates the process environment.
  """Set TF_CONFIG environment variable."""
  tf_config_dict = {
      'cluster': cluster_spec,
      'task': {
          'type': task_type,
          'index': task_id,
      },
  }
  if rpc_layer is not None:
    tf_config_dict['rpc_layer'] = rpc_layer
  os.environ['TF_CONFIG'] = json.dumps(tf_config_dict)


@tf_export('__internal__.distribute.multi_process_runner.run', v1=[])
def run(fn,
        cluster_spec,
        rpc_layer=None,
        max_run_time=None,
        return_output=False,
        timeout=_DEFAULT_TIMEOUT_SEC,
        args=None,
        kwargs=None):
  # [FUNCTION DOC]
  # Convenience entry point: creates a MultiProcessRunner from fn, cluster_spec, rpc_layer,
  # max_run_time, return_output, args and kwargs, starts it, and joins with the given timeout
  # (default module constant). Returns the run result with return values and optional output.
  # Raises whatever construction, start and join raise (timeout, unexpected exit, child errors,
  # not-initialized, invalid cluster). Note that the run does not expose the other runner options
  # such as output streaming or restart policies.
  """Run `fn` in multiple processes according to `cluster_spec`.

  Given a callable `fn`, `tf.__internal__.distribute.multi_process_runner.run`
  launches multiple processes, each of which runs `fn`. These processes are
  referred to as "subprocesses" or "child processes". Each of those subprocesses
  will have their `TF_CONFIG` environment variable set, according to
  `cluster_spec` and their task types. The stdout of the subprocesses are
  streamed to the main process' and thus available in logs (if `stream_output`
  is True), with [type-id] prefix.

  `tf.__internal__.distribute.multi_process_runner.run` will block until all
  subprocesses have successfully exited, and return a namedtuple object that
  represents the run result. This object has a `return_value` attribute, which
  is a list that contains subprocesses `fn`'s return values, for those
  subprocesses that successfully returned from `fn`. The order of `return_value`
  list is not meaningful. If an optional arg `return_output` (default to False)
  is set to True, the namedtuple object will have an additional attribute
  `stdout`, which is a list containing the stdout of the subprocesses. If any
  subprocess' `fn` ends up raising an error, that error will be reraised from
  `tf.__internal__.distribute.multi_process_runner.run`, and the aforementioned
  namedtuple object will be available through the exception's
  `mpr_result` attribute.

  This utility is used for simulating running TensorFlow programs across
  multiple task types, and each of the task type may contain more than one task
  (except for "chief" where more than one task is prohibited). Test coverage of
  multi-worker training is the main application of this utility, where code
  written for multi-worker training can be realistically covered in unit tests.

  Any test module that uses
  `tf.__internal__.distribute.multi_process_runner.run()` must call
  `tf.__internal__.distribute.multi_process_runner.test_main()` instead of
  regular `test.main()` inside `if __name__ == '__main__':` block for proper
  initialization.

  Args:
    fn: Function to be run on child processes. This will be run on processes for
      all task types.
    cluster_spec: Dict for cluster spec. The utility function
      `tf.__internal__.distribute.multi_process_runner.create_cluster_spec` can
      be conveniently used to create such dict. The following is an example of
      cluster with three workers and two ps's.
      {"worker": ["worker0.example.com:2222",
                  "worker1.example.com:2222",
                  "worker2.example.com:2222"],
       "ps": ["ps0.example.com:2222",
              "ps1.example.com:2222"]}
    rpc_layer: RPC layer to use. Default value is 'grpc'.
    max_run_time: `None` or integer. If not `None`, child processes are forced
      to exit at approximately this many seconds after this utility is called.
      We achieve this through `signal.alarm()` api. Note that this is best
      effort at Python level since Python signal handler does not get executed
      when it runs lower level C/C++ code. So it can be delayed for arbitrarily
      long time. If any of the child process is still running when
      `max_run_time` is up, they will be force-terminated and an
      `tf.__internal__.distribute.multi_process_runner
      .UnexpectedSubprocessExitError`
      may be raised. If `None`, child processes are not forced to exit.
    return_output: If True, the output/error from the subprocesses should be
      collected to be attached to the resulting namedtuple returned from this
      utility. The list of output can be retrieved via `stdout` attribute.
      Defaults to False.
    timeout: optional integer or `None`. If provided as an integer, and not all
      processes report status within roughly `timeout` seconds, a
      `tf.__internal__.distribute.multi_process_runner.SubprocessTimeoutError`
      exception will be raised. If `None`,
      `tf.__internal__.distribute.multi_process_runner.run` never times out.
      Defaults to the constant `_DEFAULT_TIMEOUT_SEC` defined in
      `multi_process_runner` module.
    args: Positional arguments to be sent to `fn` run on subprocesses.
    kwargs: Keyword arguments to be sent to `fn` run on subprocesses.

  Returns:
      A namedtuple object, which has two attributes,
      `return_value` and `stdout`. `return_value` always contains a list of
      returnvalues from the subprocesses, although the order is not meaningful.
      If `return_output` argument is True, `stdout` is available that contains a
      list of all messages from subprocesses' stdout and stderr, and the order
      is mostly chronological.

  Raises:
    RuntimeError: if
    `tf.__internal__.distribute.multi_process_runner.test_main()` is
      not called in test's `if __name__ == '__main__':` block.
    ValueError: if there are more than one chief in the `cluster_spec`.
    tf.__internal__.distribute.multi_process_runner.SubprocessTimeoutError: if
      not all processes report status approximately
      within `timeout` seconds. When this is raised, a
      namedtuple object can be retrieved by
      `tf.__internal__.distribute.multi_process_runner.SubprocessTimeoutError`'s
      `mpr_result` attribute, which has the same
      structure as above 'Returns' section describes.
    tf.__internal__.distribute.multi_process_runner
    .UnexpectedSubprocessExitError:
      If any of the subprocesses did not exit
      properly (for example, they exit on SIGTERM or SIGKILL signal). When
      this is raised, a namedtuple object can be retrieved by
      `tf.__internal__.distribute.multi_process_runner
      .UnexpectedSubprocessExitError`'s
      `mpr_result` attribute, which has the
      same structure as above 'Returns' section describes. If `max_run_time`
      is not `None`, it is expected that some subprocesses may be
      force-killed when `max_run_time` is up, and this is raised in those
      cases.
    Exception: if there is an Exception propagated from any subprocess. When
      this is raised, a namedtuple object can be retrieved by
      `tf.__internal__.distribute.multi_process_runner
      .UnexpectedSubprocessExitError`
      `mpr_result` attribute, which has the
      same structure as above 'Returns' section describes.

  Examples:

  ```python
  class SimpleMultiProcessTest(tf.test.TestCase):

    def test_simple_printing_and_return(self):

      def fn():
        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

        # This will print "[chief-0]:     Task type: chief , task id: 0"
        # for chief, for example.
        logging.info('Task type: %s, task id: %d',
                     resolver.task_type, resolver.task_id)

        return resolver.task_type

      result = tf.__internal__.distribute.multi_process_runner.run(
          fn=fn,
          cluster_spec=(
              tf.__internal__
              .distribute.multi_process_runner.create_cluster_spec(
                  has_chief=True, num_workers=2)))
      assert sorted(result.return_value) == ['chief', 'worker', 'worker']

    def test_error_from_fn(self):

      def fn():
        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        raise ValueError('Task type {}, task id {} is errors out'.format(
            resolver.task_type, resolver.task_id))

      with self.assertRaisesRegex(ValueError,
                                   'Task type worker, task id 0 is errors out'):
        cluster_spec = (
            tf.__internal__.distribute.multi_process_runner.create_cluster_spec(
                num_workers=1))
        tf.__internal__.distribute.multi_process_runner.run(
            fn=fn, cluster_spec=cluster_spec)


  if __name__ == '__main__':
    tf.__internal__.distribute.multi_process_runner.test_main()
  ```
  """
  runner = MultiProcessRunner(
      fn,
      cluster_spec,
      rpc_layer,
      max_run_time=max_run_time,
      return_output=return_output,
      args=args,
      kwargs=kwargs)
  runner.start()
  return runner.join(timeout)


# This is set by MultiProcessRunner in worker processes.
_barrier = None


@tf_export('__internal__.distribute.multi_process_runner.get_barrier', v1=[])
def get_barrier():
  # [FUNCTION DOC]
  # Returns the multiprocessing barrier shared by all child tasks. Raises a value error if called
  # where the barrier was never set (such as the main process). Reads a module global that child
  # processes assign during start-up.
  """Returns a `multiprocessing.Barrier` for `multi_process_runner.run`.

  `tf.__internal__.distribute.multi_process_runner.get_barrier()` returns
  a `multiprocessing.Barrier` object which can be used within `fn` of
  `tf.__internal__.distribute.multi_process_runner` to wait with
  `barrier.wait()` call until all other tasks have also reached the
  `barrier.wait()` call, before they can proceed individually.

  Note that all tasks (subprocesses) have to reach `barrier.wait()` call to
  proceed. Currently it is not supported to block on only a subset of tasks
  in the cluster.

  Example:
  ```python

  def fn():
    some_work_to_be_done_by_all_tasks()

    tf.__internal__.distribute.multi_process_runner.get_barrier().wait()

    # The barrier guarantees that at this point, all tasks have finished
    # `some_work_to_be_done_by_all_tasks()`
    some_other_work_to_be_done_by_all_tasks()

  result = tf.__internal__.distribute.multi_process_runner.run(
      fn=fn,
      cluster_spec=(
          tf.__internal__
          .distribute.multi_process_runner.create_cluster_spec(
              num_workers=2)))
  ```


  Returns:
    A `multiprocessing.Barrier` for `multi_process_runner.run`.
  """
  if _barrier is None:
    raise ValueError(
        'barrier is not defined. It is likely because you are calling '
        'get_barrier() in the main process. get_barrier() can only be called '
        'in the subprocesses.'
    )
  return _barrier


_manager = None
_manager_lock = threading.Lock()


def manager():
  # [FUNCTION DOC]
  # Returns the process-wide multiprocessing manager, creating it lazily under a lock on first
  # use. Raises the not-initialized error if the test entry point was not used. Side effect:
  # starts a manager server process on first call.
  """Returns the multiprocessing manager object for concurrency tools.

  The manager object is useful as it controls a server process that holds
  the python objects that can be shared across processes. This can be used
  for parent-subprocess communication:

  ```python
  manager = multi_process_runner.manager()
  some_event_happening_in_subprocess = manager.Event()
  mpr = multi_process_runner.MultiProcessRunner(fn, cluster_spec,
      args=(some_event_happening_in_subprocess,))
  mpr.start()
  some_event_happening_in_subprocess.wait()
  # Do something that only should after some event happens in subprocess.
  ```

  Note that the user of multi_process_runner should not create additional
  `multiprocessing.Manager()` objects; doing so can result in segfault in
  some cases.

  This method should only be called after multi_process_runner.test_main() is
  called.
  """
  _check_initialization()
  global _manager
  with _manager_lock:
    if _manager is None:
      _manager = multiprocessing.Manager()
    return _manager


@tf_export('__internal__.distribute.multi_process_runner.test_main', v1=[])
def test_main():
  # [FUNCTION DOC]
  # Test entry point replacing the regular test main. Wraps the main module's module-teardown hook
  # (keeping any existing one) so all pool runners are shut down first, installs the wrapper on the
  # main module, and then delegates to the library's test main to initialize multi-process
  # support and run the tests. No inputs; side effects on the main module's attributes.
  """Main function to be called within `__main__` of a test file.

  Any test module that uses
  `tf.__internal__.distribute.multi_process_runner.run()`
  must call this instead of regular `test.main()` inside
  `if __name__ == '__main__':` block, or an error will be raised when
  `tf.__internal__.distribute.multi_process_runner.run()` is used. This method
  takes
  care of needed initialization for launching multiple subprocesses.

  Example:
  ```python
  class MyTestClass(tf.test.TestCase):
    def testSomething(self):
      # Testing code making use of
      # `tf.__internal__.distribute.multi_process_runner.run()`.

  if __name__ == '__main__':
    tf.__internal__.distribute.multi_process_runner.test_main()
  ```
  """
  # Inject tearDownModule() to shut down all pool runners. Active pool runners
  # will block the program from exiting. This is necessary for global pool
  # runners. We tried atexit in the past, and it doesn't work in some
  # deployment.
  old_tear_down_module = getattr(sys.modules['__main__'], 'tearDownModule',
                                 None)

  def tear_down_module():
    # [FUNCTION DOC]
    # HELPER: replacement module-teardown hook. Shuts down all pool runners, then calls the
    # previously installed hook if one existed. Returns nothing.
    _shutdown_all_pool_runners()
    if old_tear_down_module is not None:
      old_tear_down_module()

  setattr(sys.modules['__main__'], 'tearDownModule', tear_down_module)
  multi_process_lib.test_main()
