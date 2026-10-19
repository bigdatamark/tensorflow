# [MODULE DOC]
# Implementation of the cluster coordinator used with parameter server training. It defines the
# public ClusterCoordinator class (exported as the distribute coordinator API) together with the
# internal machinery behind it: Closure types that wrap a scheduled function and its arguments,
# a thread-safe coordinated closure queue (global queue plus per-worker tagged queues), a Worker
# class running one background thread per remote worker, a Cluster class owning the workers, the
# queue and a failure handler, and two preemption handlers (one based on error inspection, one
# based on polling task states from the coordination service) that detect worker preemption,
# wait for recovery and refresh the server definition. It also holds helper functions that
# classify errors as parameter server failures or worker failures, plus the error wrapper types
# ClosureInputError, ClosureAbortedError and PSUnavailableError.
# Module-level constants control the maximum recovery wait, the task-state polling frequency, the
# closure queue size and the error-message markers. It re-exports the RemoteValue related types
# under local aliases. Import time side effects: only the registration of the public API export
# for ClusterCoordinator; no threads are started until a coordinator is constructed.
# Several behaviors are tuned through environment variables read when a Cluster or Worker is
# created (transient failure thresholds, staggered start delays, coordination service usage).
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Module for `ClusterCoordinator` and relevant cluster-worker related library.

This is currently under development and the API is subject to change.
"""

import collections
import contextlib
import os
import re
import threading
import time
import weakref

from six.moves import queue

from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# Maximum time for failed worker to come back is 1 hour
_WORKER_MAXIMUM_RECOVERY_SEC = 3600
# How often to poll task states from the coordination service. In testing, a
# value of 1 led to some spurious reports of unavailability, so a higher value
# is used. Refer to the discussion in b/249134783 for more.
_POLL_FREQ_IN_SEC = 5

# Maximum size for queued closures, "infinite" if set to 0.
# When the maximum queue size is reached, further schedule calls will become
# blocking until some previously queued closures are executed on workers.
# Note that using an "infinite" queue size can take a non-trivial portion of
# memory, and even lead to coordinator OOM. Modify the size to a smaller value
# for coordinator with constrained memory resource (only recommended for
# advanced users). Also used in unit tests to ensure the correctness when the
# queue is full.
_CLOSURE_QUEUE_MAX_SIZE = 256 * 1024

# RPC error message from PS
_RPC_ERROR_FROM_PS = "GRPC error information from remote target /job:ps"

# InvalidArgumentError (unknown device) will not have "GRPC error..." string.
_JOB_WORKER_STRING_IDENTIFIER = "/job:worker"


RemoteValueStatus = remote_value.RemoteValueStatus
RemoteValue = remote_value.RemoteValue
RemoteValueImpl = values_lib.RemoteValueImpl
RemoteVariable = values_lib.RemoteVariable
PerWorkerValues = values_lib.PerWorkerValues


class ClosureInputError(Exception):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring lists an attribute named original_exception with no description.
  # Exception raised when a closure finds that one of its input remote values already holds an
  # error. It wraps the underlying exception so the worker failure handler can treat it
  # uniformly. If the wrapped exception is itself a closure input or aborted error, the innermost
  # original exception is kept to avoid nesting. State: original_exception, the unwrapped error.
  """Wrapper for errors from resource building.

  When a closure starts, it first checks for errors in any of its inputs, which
  are RemoteValues from resource closures. If there were any errors, it wraps
  the exception in this class and raises so it can be handled by the worker
  failure handler.

  Attributes:
    original_exception:
  """

  def __init__(self, original_exception):
    # [FUNCTION DOC]
    # Builds the wrapper error. Parameter original_exception: the exception to wrap (any
    # exception object). If it is already a closure input or closure aborted error, its own
    # original exception is stored instead. Stores it as the original_exception attribute,
    # initializes the base exception with a message that includes the representation and text of
    # the original error, and copies the original traceback onto the new exception. Returns
    # nothing.
    # Avoid doubly-nested errors
    if isinstance(original_exception,
                  (ClosureInputError, ClosureAbortedError)):
      self.original_exception = original_exception.original_exception
    else:
      self.original_exception = original_exception
    message = ("Input has an error, the original exception is %r, "
               "error message is %s." %
               (self.original_exception, str(self.original_exception)))
    super().__init__(message)
    self.with_traceback(original_exception.__traceback__)


class ClosureAbortedError(Exception):
  # [CLASS DOC]
  # Exception attached to resource remote values that were aborted because a training closure
  # failed or a worker was lost, so any later reader sees why the value is unavailable. Wraps the
  # original exception; if the original is already a closure input or aborted error, the
  # innermost original exception is kept. State: original_exception.
  """Wrapper for errors from training closures, to attach to resource closures.

  This wrapper is used when a dependent training closure fails to set errors on
  its required resource closures.

  Attributes:
    original_exception: The Exception to wrap
  """

  def __init__(self, original_exception):
    # [FUNCTION DOC]
    # Builds the wrapper error. Parameter original_exception: the exception to wrap. If it is
    # already a closure input or closure aborted error, its own original exception is stored
    # instead. Stores it as the original_exception attribute, initializes the base exception with
    # a message saying another function failed and the value is unavailable (including the
    # original error), and copies the original traceback. Returns nothing.
    # Avoid doubly-nested errors
    if isinstance(original_exception,
                  (ClosureInputError, ClosureAbortedError)):
      self.original_exception = original_exception.original_exception
    else:
      self.original_exception = original_exception
    message = ("Other function has an execution error, as a result, the "
               "current value is not available. The original exception is %r, "
               "error message is %s." %
               (self.original_exception, str(self.original_exception)))
    super().__init__(message)
    self.with_traceback(original_exception.__traceback__)


class PSUnavailableError(errors.UnavailableError):
  # [CLASS DOC]
  # Unavailable-error subclass that marks the failure as coming from a parameter server rather
  # than a worker, so failure classification can treat it as unrecoverable. Holds the original
  # unavailable error as original_exception.
  """Specifies that a parameter server is the unavailable task."""

  def __init__(self, original_exception):
    # [FUNCTION DOC]
    # Parameter original_exception: an unavailable error (asserted; an assertion failure occurs
    # otherwise). Keeps it as an attribute so the error can be serialized, then initializes the
    # base unavailable error with the original's node definition, operation and message. Returns
    # nothing.
    assert isinstance(original_exception, errors.UnavailableError)
    # TF Errors should have init args set as attributes for serialization.
    self.original_exception = original_exception
    super().__init__(
        original_exception.node_def,
        original_exception.op,
        original_exception.message,
    )


def _get_error_from_remote_values(structure):
  # [FUNCTION DOC]
  # HELPER: used by Closure.execute_on to detect failed inputs before running a closure.
  # KNOWN ISSUE: the docstring mentions rebuilding values, but the code only collects errors.
  # Walks a nested structure of values and gathers the error stored in each remote value.
  # Parameter structure: any nested structure. Returns the first error found, or nothing/null
  # if no remote value holds an error.
  """Attempts to return errors from `RemoteValue`s. Rebuilds them if needed."""
  errors_in_structure = []

  def _get_error(val):
    # [FUNCTION DOC]
    # HELPER: visitor applied to each leaf of the structure. If the leaf is a remote value that
    # has an error, appends the error to the enclosing list. Returns nothing.
    if isinstance(val, RemoteValue):
      error = val._get_error()  # pylint: disable=protected-access
      if error:
        errors_in_structure.append(error)

  nest.map_structure(_get_error, structure)
  if errors_in_structure:
    return errors_in_structure[0]
  else:
    return None


def _maybe_as_type_spec(val):
  # [FUNCTION DOC]
  # HELPER: used by Closure to build argument specs for tracing a function.
  # Converts a remote value or per-worker value into its type specification; any other value is
  # returned unchanged. Raises a value error if a remote or per-worker value has no type spec
  # (output of a scheduled function that is not a compiled function).
  if isinstance(val, (RemoteValue, PerWorkerValues)):
    if val._type_spec is None:  # pylint: disable=protected-access
      raise ValueError("Output of a scheduled function that is not "
                       "tf.function cannot be the input of another function.")
    return val._type_spec  # pylint: disable=protected-access
  else:
    return val


def _select_worker_slice(worker_id, structured):
  # [FUNCTION DOC]
  # HELPER: used by Closure to pick per-worker arguments.
  # Parameters: worker_id, integer index of the worker; structured, nested structure of values.
  # Returns a structure of the same shape in which every per-worker value is replaced by the
  # component for the given worker and all other items are kept as they are.
  """Selects the worker slice of each of the items in `structured`."""

  def _get(x):
    # [FUNCTION DOC]
    # HELPER: leaf visitor for the worker slice selection. Returns the component of a
    # per-worker value for the enclosing worker id, or the item itself otherwise.
    return x._values[worker_id] if isinstance(x, PerWorkerValues) else x  # pylint: disable=protected-access

  return nest.map_structure(_get, structured)


def _disallow_remote_value_as_input(structured):
  # [FUNCTION DOC]
  # HELPER: argument validation for Closure construction.
  # Parameter structured: nested structure of arguments. Raises a value error if any leaf is a
  # remote value, since remote values are not yet supported as inputs. Returns nothing.
  """Raises if any element of `structured` is a RemoteValue."""

  def _raise_if_remote_value(x):
    # [FUNCTION DOC]
    # HELPER: leaf visitor for the validation. Raises a value error if the item is a remote
    # value; otherwise does nothing.
    if isinstance(x, RemoteValue):
      raise ValueError(
          "`tf.distribute.experimental.coordinator.RemoteValue` used "
          "as an input to scheduled function is not yet "
          "supported.")

  nest.map_structure(_raise_if_remote_value, structured)


class Closure(object):
  # [CLASS DOC]
  # A unit of work to be run on a worker: a callable plus its positional and keyword arguments.
  # If the callable is a traced-function wrapper or a concrete function, the closure keeps the
  # concrete function, derives the output type spec and wraps the function so it can be
  # cancelled; otherwise it keeps the plain callable and has no output type spec. It holds only a
  # weak reference to its output remote value. Subclassed by ResourceClosure.
  """Hold a function to be scheduled and its arguments."""

  def __init__(self, function, cancellation_mgr, args=None, kwargs=None):
    # [FUNCTION DOC]
    # Parameters: function, a callable (traced function, concrete function or plain callable);
    # cancellation_mgr, the cancellation manager used to make concrete functions cancelable;
    # args, optional positional argument structure (default empty); kwargs, optional keyword
    # argument mapping (default empty). Raises a value error if function is not callable or if
    # any argument is a remote value. For a traced function it gets a concrete function using the
    # first worker's argument slice, timing the tracing in a metric. Sets the output type spec
    # and cancelable function, and clears the output remote value reference. Returns nothing.
    if not callable(function):
      raise ValueError("Function passed to `ClusterCoordinator.schedule` must "
                       "be a callable object.")
    self._args = args or ()
    self._kwargs = kwargs or {}

    _disallow_remote_value_as_input(self._args)
    _disallow_remote_value_as_input(self._kwargs)

    if isinstance(function, def_function.Function):
      replica_args = _select_worker_slice(0, self._args)
      replica_kwargs = _select_worker_slice(0, self._kwargs)

      # Note: no need to handle function registration failure since this kind of
      # failure will not raise exceptions as designed in the runtime. The
      # coordinator has to rely on subsequent operations that raise to catch
      # function registration failure.

      # Record the function tracing overhead. Note that we pass in the tracing
      # count of the def_function.Function as a state tracker, so that metrics
      # will only record the time for actual function tracing (i.e., excluding
      # function cache lookups).
      with metric_utils.monitored_timer(
          "function_tracing", state_tracker=function._get_tracing_count):  # pylint: disable=protected-access
        self._concrete_function = function.get_concrete_function(
            *nest.map_structure(_maybe_as_type_spec, replica_args),
            **nest.map_structure(_maybe_as_type_spec, replica_kwargs))
    elif isinstance(function, tf_function.ConcreteFunction):
      self._concrete_function = function

    if hasattr(self, "_concrete_function"):
      # If we have a concrete function, we get to retrieve the output type spec
      # via the structured_output.
      self._output_type_spec = func_graph.convert_structure_to_signature(
          self._concrete_function.structured_outputs)
      self._function = cancellation_mgr.get_cancelable_function(
          self._concrete_function)
    else:
      # Otherwise (i.e. what is passed in is a regular python function), we have
      # no such information.
      self._output_type_spec = None
      self._function = function

    self._output_remote_value_ref = None

  def build_output_remote_value(self):
    # [FUNCTION DOC]
    # Creates the remote value that will receive this closure's output, remembering only a weak
    # reference to it. Takes no parameters. Returns the new remote value. Raises a value error if
    # an output remote value was already built for this closure.
    if self._output_remote_value_ref is None:
      ret = RemoteValueImpl(None, self._output_type_spec)
      self._output_remote_value_ref = weakref.ref(ret)
      return ret
    else:
      raise ValueError(
          "The output of the Closure cannot be built more than once.")

  def maybe_call_with_output_remote_value(self, method):
    # [FUNCTION DOC]
    # Parameter method: callback taking the output remote value. If an output remote value was
    # built and is still alive, calls the callback with it and returns its result; otherwise does
    # nothing and returns nothing/null.
    if self._output_remote_value_ref is None:
      return None
    output_remote_value = self._output_remote_value_ref()
    if output_remote_value is not None:
      return method(output_remote_value)
    return None

  def mark_cancelled(self):
    # [FUNCTION DOC]
    # Sets a cancelled error (asking the user to reschedule the function) on the output remote
    # value, if one exists and is alive. Takes no parameters, returns nothing.
    e = errors.CancelledError(
        None, None, "The corresponding function is "
        "cancelled. Please reschedule the function.")
    self.maybe_call_with_output_remote_value(lambda r: r._set_error(e))  # pylint: disable=protected-access

  def execute_on(self, worker):
    # [FUNCTION DOC]
    # Runs the closure on one worker. Parameter worker: a Worker object supplying its index,
    # device name and executor. Selects that worker's slice of the arguments; if any argument
    # remote value holds an error, raises it (wrapped as a closure input error if not already).
    # Otherwise runs the function under the worker's device, executor and dispatch context
    # (timed by a metric), resolving remote value arguments to local values, and stores the
    # outputs in the output remote value if it is alive. Returns nothing; side effects are the
    # remote execution and setting the remote value.
    """Executes the closure on the given worker.

    Args:
      worker: a `Worker` object.
    """
    replica_args = _select_worker_slice(worker.worker_index, self._args)
    replica_kwargs = _select_worker_slice(worker.worker_index, self._kwargs)

    e = (
        _get_error_from_remote_values(replica_args) or
        _get_error_from_remote_values(replica_kwargs))
    if e:
      if not isinstance(e, ClosureInputError):
        e = ClosureInputError(e)
      raise e

    with ops.device(worker.device_name):
      with context.executor_scope(worker.executor):
        with coordinator_context.with_dispatch_context(worker):
          with metric_utils.monitored_timer("closure_execution"):
            output_values = self._function(
                *nest.map_structure(coordinator_context.maybe_get_remote_value,
                                    replica_args),
                **nest.map_structure(coordinator_context.maybe_get_remote_value,
                                     replica_kwargs))
    self.maybe_call_with_output_remote_value(
        lambda r: r._set_values(output_values))  # pylint: disable=protected-access


class ResourceClosure(Closure):
  # [CLASS DOC]
  # Closure subclass for building per-worker resources (such as datasets). Its output remote
  # value keeps a strong reference back to the closure so the coordinator can re-run it on a
  # worker after recovery. Overrides how the output remote value is created and reused.
  """A closure that builds a resource on a worker.

  ResourceClosures keep a reference to the closure object, which is used to
  rerun the closure upon recovery to ensure  workers have access to the
  resources they need.
  """

  def _init_remote_value(self):
    # [FUNCTION DOC]
    # HELPER: factory hook for build_output_remote_value. Returns a new remote value that
    # remembers this closure and the output type spec.
    return RemoteValueImpl(self, self._output_type_spec)

  def build_output_remote_value(self):
    # [FUNCTION DOC]
    # Returns the output remote value, creating it (with a weak reference kept) on first call.
    # Unlike the base class it does not raise on repeated calls; it returns the existing value
    # through the weak reference, which may be nothing/null if that value was already freed.
    if self._output_remote_value_ref is None:
      # We need to remember the Closure object in the `RemoteValue` here.
      ret = self._init_remote_value()
      self._output_remote_value_ref = weakref.ref(ret)
      return ret
    else:
      return self._output_remote_value_ref()


class PerWorkerVariableClosure(ResourceClosure):
  # [CLASS DOC]
  # Resource closure whose output is a remote variable instead of a plain remote value; used to
  # create per-worker variables.

  def _init_remote_value(self):
    # [FUNCTION DOC]
    # HELPER: factory hook override. Returns a new remote variable that remembers this closure
    # and the output type spec.
    return RemoteVariable(self, self._output_type_spec)


class _CoordinatedClosureQueue(object):
  # [CLASS DOC]
  # HELPER: internal queue used by Cluster to hand closures to worker threads.
  # Thread-safe container of closures waiting to run. It has a global bounded queue plus a
  # per-tag queue (tag is a worker index) for resource closures that must run on one worker. It
  # tracks the number of in-flight closures, the first error reported by a failed closure, a
  # cancellation manager for in-flight functions, a watchdog that logs state when no closure
  # finishes for a while, and several condition variables sharing one lock (queue not empty,
  # slot free, all done, no in-flight closures). A separate lock prevents puts during waits.
  """Manage a queue of closures, inflight count and errors from execution.

  This class is thread-safe.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Creates empty queues, the lock and condition variables, a fresh cancellation manager, an
    # in-flight count of zero, no stored error and a watchdog that reports state when triggered.
    # Logs a warning if the module's maximum queue size is zero or negative (infinite queue).
    # Updates the queued-closures metric. Takes no parameters, returns nothing.
    # `self._inflight_closure_count` only tracks the number of inflight closures
    # that are "in generation". Once an error occurs, error generation is
    # incremented and all subsequent arriving closures (from inflight) are
    # considered "out of generation".
    self.inflight_closure_count = 0

    self._queue_lock = threading.Lock()

    # Condition indicating that all pending closures (either queued or inflight)
    # have been processed, failed, or cancelled.
    self._stop_waiting_condition = threading.Condition(self._queue_lock)

    # Condition indicating that an item becomes available in queue (not empty).
    self._closures_queued_condition = threading.Condition(self._queue_lock)
    self._should_process_closures = True

    # Condition indicating that a queue slot becomes available (not full).
    # Note that even with "infinite" queue size, there is still a "practical"
    # size limit for the queue depending on host memory capacity, and thus the
    # queue will eventually become full with a lot of enqueued closures.
    self._queue_free_slot_condition = threading.Condition(self._queue_lock)

    # Condition indicating there is no inflight closures.
    self._no_inflight_closure_condition = threading.Condition(self._queue_lock)

    # Use to cancel in-flight closures.
    self._cancellation_mgr = cancellation.CancellationManager()

    if _CLOSURE_QUEUE_MAX_SIZE <= 0:
      logging.warning(
          "In a `ClusterCoordinator`, creating an infinite closure queue can "
          "consume a significant amount of memory and even lead to OOM.")
    self._queue = queue.Queue(maxsize=_CLOSURE_QUEUE_MAX_SIZE)
    metric_utils.monitor_int("queued_closures", self._queue.qsize())
    self._tagged_queue = collections.defaultdict(queue.Queue)
    self._error = None

    # The following is a lock to make sure when `wait` is called and before it
    # returns no `put` can be executed during this period. It is because `wait`
    # won't know what to do with newly put closures. This lock adds an cutoff
    # for `wait` so that closures put into the queue while waiting would not be
    # taken responsible by this `wait`.
    #
    # We cannot reuse the `self._queue_lock` since when `wait` waits for a
    # condition, the `self._queue_lock` will be released.
    #
    # We don't use a reader/writer's lock on purpose to reduce the complexity
    # of the code.
    self._put_wait_lock = threading.Lock()

    self._watchdog = watchdog.WatchDog(on_triggered=self._on_watchdog_timeout)

  def _on_watchdog_timeout(self):
    # [FUNCTION DOC]
    # HELPER: callback for the watchdog. Logs the current in-flight closure count and the
    # current stored error. Returns nothing.
    logging.info("inflight_closure_count is %d", self._inflight_closure_count)
    logging.info("current error is %s:%r", self._error, self._error)

  @property
  def inflight_closure_count(self):
    # [FUNCTION DOC]
    # Property getter. Returns the number of closures currently being executed by workers.
    return self._inflight_closure_count

  @inflight_closure_count.setter
  def inflight_closure_count(self, value):
    # [FUNCTION DOC]
    # Property setter. Parameter value: the new in-flight count (integer). Stores it and
    # updates the in-flight closures metric. Returns nothing.
    self._inflight_closure_count = value
    metric_utils.monitor_int("inflight_closures", self._inflight_closure_count)

  def stop(self):
    # [FUNCTION DOC]
    # Shuts the queue down: under the lock, stops closure processing, starts cancellation of
    # in-flight functions and wakes all threads waiting for closures (they then get nothing).
    # Then stops the watchdog. Takes no parameters, returns nothing.
    with self._queue_lock:
      self._should_process_closures = False
      self._cancellation_mgr.start_cancel()
      self._closures_queued_condition.notify_all()
    self._watchdog.stop()

  def _cancel_all_closures(self):
    # [FUNCTION DOC]
    # HELPER: error cleanup for put, wait and done. Caller must hold the queue lock.
    # Starts cancellation, waits until no closures are in flight, then drains the global queue
    # marking each drained closure as cancelled and updating the metric, and finally replaces the
    # cancellation manager with a fresh one because a cancelled manager cannot be reused.
    # Tagged (per-worker) queues are not drained here. Returns nothing.
    """Clears the queue and sets remaining closures cancelled error.

    This method expects self._queue_lock to be held prior to entry.
    """
    self._cancellation_mgr.start_cancel()
    logging.info("Canceling all closures: waiting for inflight closures to "
                 "finish")
    while self._inflight_closure_count > 0:
      self._no_inflight_closure_condition.wait()
    logging.info("Canceling all closures: canceling remaining closures on the "
                 "queue")
    while True:
      try:
        closure = self._queue.get(block=False)
        metric_utils.monitor_int("queued_closures", self._queue.qsize())
        self._queue_free_slot_condition.notify()
        closure.mark_cancelled()
      except queue.Empty:
        break
    # The cancellation manager cannot be reused once cancelled. After all
    # closures (queued or inflight) are cleaned up, recreate the cancellation
    # manager with clean state.
    # Note on thread-safety: this is triggered when one of theses
    # ClusterCoordinator APIs are called: `schedule`, `wait`, and `done`. At the
    # same time, no new closures can be constructed (which reads the
    # _cancellation_mgr to get cancellable functions).
    self._cancellation_mgr = cancellation.CancellationManager()

  def _raise_if_error(self):
    # [FUNCTION DOC]
    # HELPER: shared by put, wait and done. Caller must hold the queue lock.
    # If an error was recorded by a failed closure, logs it, cancels all queued closures, raises
    # the error and clears it (so it is reported only once). Does nothing otherwise.
    """Raises the error if one exists.

    If an error exists, cancel the closures in queue, raises it, and clear
    the error.

    This method expects self._queue_lock to be held prior to entry.
    """
    if self._error:
      logging.error("Start cancelling closures due to error %r: %s",
                    self._error, self._error)
      self._cancel_all_closures()
      try:
        raise self._error  # pylint: disable=raising-bad-type
      finally:
        self._error = None

  def put(self, closure, tag=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a prior mark_failed error is raised by put, but that only
    # happens for untagged closures; a put with a tag never checks for or raises a stored error.
    # Parameters: closure, the closure to enqueue (its tag attribute is set to the given tag);
    # tag, optional worker index. A tagged closure goes to that tag's queue (no blocking, wakes
    # all waiting getters). An untagged closure waits for a free slot in the bounded global queue
    # (blocking while full), is added, then any stored error is raised after cancelling queued
    # closures, and one getter is woken. Untagged puts are excluded while a wait is in progress.
    """Put a closure into the queue for later execution.

    If `mark_failed` was called before `put`, the error from the first
    invocation of `mark_failed` will be raised.

    Args:
      closure: The `Closure` to put into the queue.
      tag: if not None, put into a queue with the given tag.
    """
    closure.tag = tag
    if tag is not None:
      with self._queue_lock:
        self._tagged_queue[tag].put(closure, block=False)
        self._closures_queued_condition.notify_all()
    else:
      with self._put_wait_lock, self._queue_lock:
        self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
        self._queue.put(closure, block=False)
        metric_utils.monitor_int("queued_closures", self._queue.qsize())
        self._raise_if_error()
        self._closures_queued_condition.notify()

  def get(self, timeout=None, tag=None):
    # [FUNCTION DOC]
    # Parameters: timeout, optional seconds to wait for a closure (default wait forever); tag,
    # optional worker index whose queue is checked first. Blocks until a closure is available or
    # processing is stopped. Returns nothing/null on timeout or after stop. A closure from the
    # tagged queue is returned without counting it as in flight; a closure from the global queue
    # frees a slot, increments the in-flight count and is returned.
    """Return a closure from the queue to be executed.

    It will try to fetch an item from the queue with the given tag. If this
    queue is empty, it will then check the global queue.

    Args:
      timeout: timeout when waiting for a closure to be put.
      tag: optional tag to specify which queue to query first before querying
        the global queue.

    Returns:
      a closure or None after timeout.
    """
    with self._queue_lock:
      while (self._should_process_closures and self._queue.empty() and
             (tag is None or self._tagged_queue[tag].empty())):
        if not self._closures_queued_condition.wait(timeout=timeout):
          return None
      if not self._should_process_closures:
        return None
      if tag is not None and not self._tagged_queue[tag].empty():
        closure = self._tagged_queue[tag].get(block=False)
        return closure
      closure = self._queue.get(block=False)
      metric_utils.monitor_int("queued_closures", self._queue.qsize())
      assert closure.tag is None
      assert tag is None or self._tagged_queue[tag].empty()
      self._queue_free_slot_condition.notify()
      self.inflight_closure_count += 1
      return closure

  def mark_finished(self):
    # [FUNCTION DOC]
    # Records that an in-flight closure succeeded: decrements the in-flight count and wakes
    # waiters when nothing is in flight or everything is done, then reports progress to the
    # watchdog. Raises an assertion error if no closure is in flight. Returns nothing.
    """Let the queue know that a closure has been successfully executed."""
    with self._queue_lock:
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to mark_finished.")
      self.inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notify_all()
      if self._queue.empty() and self._inflight_closure_count == 0:
        self._stop_waiting_condition.notify_all()
      self._watchdog.report_closure_done()

  def put_back(self, closure):
    # [FUNCTION DOC]
    # Parameter closure: an untagged in-flight closure (asserted) that could not be executed
    # properly. If an error is already stored the closure is marked cancelled; otherwise it is
    # waited-for-slot and re-queued. Either way the in-flight count drops and waiters are woken
    # if none remain. Raises an assertion error if no closure is in flight. Returns nothing.
    """Put the closure back into the queue as it was not properly executed."""
    assert closure.tag is None
    with self._queue_lock:
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to put_back.")
      if self._error:
        closure.mark_cancelled()
      else:
        self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
        self._queue.put(closure, block=False)
        metric_utils.monitor_int("queued_closures", self._queue.qsize())
        self._closures_queued_condition.notify()
      self.inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notify_all()

  def wait(self, timeout=None):
    # [FUNCTION DOC]
    # Blocks until the global queue is empty and nothing is in flight, or an error is stored.
    # Parameter timeout: optional seconds (float). Returns false if the wait timed out; if an
    # error is stored it is raised (after cancelling queued closures); otherwise returns true.
    # Puts are blocked meanwhile so later closures are not covered by this wait.
    """Wait for all closures to be finished before returning.

    If `mark_failed` was called before or during `wait`, the error from the
    first invocation of `mark_failed` will be raised.

    Args:
      timeout: A float specifying a timeout for the wait in seconds.

    Returns:
      True unless the given timeout expired, in which case it returns False.
    """
    with self._put_wait_lock, self._queue_lock:
      logging.info("Waiting for all global closures to be finished.")
      while (not self._error and
             (not self._queue.empty() or self._inflight_closure_count > 0)):
        if not self._stop_waiting_condition.wait(timeout=timeout):
          return False
      self._raise_if_error()
      return True

  def mark_failed(self, e):
    # [FUNCTION DOC]
    # Parameter e: the exception from a failed in-flight closure. Keeps it as the stored error
    # only if no error is stored yet (first error wins), decrements the in-flight count and wakes
    # waiters. Raises an assertion error if no closure is in flight. Returns nothing.
    """Sets error and unblocks any wait() call."""
    with self._queue_lock:
      # TODO(yuefengz): maybe record all failure and give users more
      # information?
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to mark_failed.")
      if self._error is None:
        self._error = e
      self.inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notify_all()
      self._stop_waiting_condition.notify_all()

  def done(self):
    # [FUNCTION DOC]
    # Raises any stored error (after cancelling queued closures); otherwise returns true when the
    # global queue is empty and nothing is in flight. Takes no parameters.
    """Returns true if the queue is empty and there is no inflight closure.

    If `mark_failed` was called before `done`, the error from the first
    invocation of `mark_failed` will be raised.
    """
    with self._queue_lock:
      self._raise_if_error()
      return self._queue.empty() and self._inflight_closure_count == 0

  def clear_tag_unlocked(self, tag):
    # [FUNCTION DOC]
    # Discards all closures in the queue for the given tag (worker index) by replacing it with
    # an empty queue. Does not take the lock itself, so the caller must ensure no concurrent
    # access. Discarded closures are not marked cancelled. Returns nothing.
    self._tagged_queue[tag] = queue.Queue()


class CoordinationServicePreemptionHandler(object):
  # [CLASS DOC]
  # Failure handler used when the coordination service integration is enabled by environment
  # variable. A repeating timer thread polls task states of workers and parameter servers; a
  # daemon thread updates the server definition when a worker needs recovery. State: server
  # definition, owning cluster, a lock and event signalling that a cluster update is due, a
  # condition for worker recovery, a condition notified on each new task state, the latest task
  # states, an error from recovery and worker and PS counts. Provides the same interface as the
  # other handler (a stop method and a context manager for failures) so Cluster can use either.
  """Handles preemptions of workers and parameter servers.

  Starts a thread to regularly poll the coordination service (hosted on PS 0)
  for task states. When a worker's task state reflects an error, it inspects the
  error. If the error is recoverable (i.e. a preemption), it waits for the
  worker to recover, then updates the server def. Otherwise, it raises the error
  to the user.

  A worker error is detected to be recoverable if it is the result of missing a
  heartbeat that workers regularly send to the coordination service.

  The thread also checks for parameter server errors. If these are detected, the
  thread and coordinator shutdown. To resume training in this case, the whole
  job must be restarted and resumed from the latest checkpoint.
  """

  def __init__(self, server_def, cluster):
    # [FUNCTION DOC]
    # Parameters: server_def, the server definition used to refresh the cluster after worker
    # recovery; cluster, the owning Cluster (provides worker and PS counts, closure queue and
    # transient-timeout bookkeeping). Creates the locks, events and conditions, starts a repeating
    # timer that polls task states at the module polling interval and starts the daemon
    # preemption handler thread. Returns nothing.
    self._server_def = server_def
    self._cluster = cluster
    self._cluster_update_lock = threading.Lock()
    self._cluster_due_for_update_or_finish = threading.Event()
    self._worker_up_cond = threading.Condition(self._cluster_update_lock)

    self._next_task_state_cond = threading.Condition()
    self._task_states = None

    self._error_from_recovery = None
    self._should_preemption_thread_run = True
    self._task_state_poller_thread = utils.RepeatedTimer(
        interval=_POLL_FREQ_IN_SEC,
        function=self._get_task_states)
    self._preemption_handler_thread = threading.Thread(
        target=self._preemption_handler,
        name="WorkerPreemptionHandler",
        daemon=True)
    self._preemption_handler_thread.start()

    self._num_workers = self._cluster._num_workers
    self._num_ps = self._cluster._num_ps

  def stop(self):
    # [FUNCTION DOC]
    # Stops the task-state poller, tells the preemption thread to exit and wakes it. The thread
    # is not joined (see the existing comment about termination). Returns nothing.
    """Ensure the worker preemption thread is closed."""
    self._task_state_poller_thread.stop()
    self._should_preemption_thread_run = False
    with self._cluster_update_lock:
      self._cluster_due_for_update_or_finish.set()
    # TODO(yuefengz): The preemption handler thread shouldn't be terminated
    # asynchronously since it touches eager context which is a process-wide
    # singleton. The problem is in OSS unit tests will time out.

  @contextlib.contextmanager
  def wait_on_failure(self,
                      on_failure_fn=None,
                      on_transient_failure_fn=None,
                      on_recovery_fn=None,
                      worker_device_name="(unknown)"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: with the default worker_device_name of the unknown placeholder, an error in
    # the guarded block makes the worker index parsing fail with a value error instead of
    # handling the original error; callers must always pass a real device name.
    # Context manager wrapping closure execution. Parameters: on_failure_fn, optional callback
    # taking the error, run on preemption; on_transient_failure_fn, optional callback run when a
    # failure is ignored as transient; on_recovery_fn, optional callback run after the worker
    # recovered; worker_device_name, device string ending in the worker index. Yields nothing.
    # On an op error or closure input/aborted error it waits two polling rounds for fresh task
    # states, then: if the coordination service is down or a PS failed, stops and raises a PS
    # unavailable error; if this worker's state shows an error, raises a cancelled error when
    # closures are being cancelled (or re-raises the cancelled error), otherwise runs failure and
    # recovery handling and returns; if the error is an ignorable transient timeout it logs and
    # calls the transient callback; any other error is re-raised.
    """Catches errors during closure execution and handles them.

    Args:
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
      on_recovery_fn: an optional function to run when a worker is recovered
        from preemption.
      worker_device_name: the device name of the worker instance that is passing
        through the failure.

    Yields:
      None.
    """
    assert self._should_preemption_thread_run
    try:
      yield
    except (errors.OpError, ClosureInputError,
            ClosureAbortedError) as e:
      # The next state could reflect stale heartbeats, so wait for two rounds.
      # Example:
      # - Worker sends healthy heartbeat at T=0.
      # - Coordination service receives healthy heartbeat at T=0.
      # - Worker gets preempted at T=0.1.
      # - Coordinator catches error at T=0.2, and waits here for next states.
      # - Coordinator polls states at T=1.9. Heartbeat time has not elapsed yet,
      #   so coordination service does not know it is down yet.
      # - Coordination service learns of worker unavailability at T=2, the next
      #   heartbeat.
      # - Coordinator polls states at T=3.9 and learns of worker unavailability.
      with self._next_task_state_cond:
        # Give some buffer time to make sure task states are updated during the
        # wait interval
        self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 1.25)
      with self._next_task_state_cond:
        self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 1.25)

      # Check for coordination service failure
      if not self._task_states:
        self._log_ps_failure_and_raise(e, 0)

      worker_states = self._task_states[:self._num_workers]
      ps_states = self._task_states[self._num_workers:]

      # Check for PS failure
      if any(ps_states):
        failed_ps_index = [
            ix for ix, ps_state in enumerate(ps_states) if ps_state
        ]
        self._log_ps_failure_and_raise(e, failed_ps_index[0])

      # Check for preemption of this worker
      worker_ix = int(worker_device_name.split(":")[-1])
      if worker_states[worker_ix]:
        # Raise error if all closures are being cancelled
        if self._cluster.closure_queue._cancellation_mgr.is_cancelled:  # pylint: disable=protected-access
          if isinstance(e, errors.CancelledError):
            raise e
          # It's possible the caught error `e` here is due to worker preemption
          # and is thus not a `CancelledError`, because a different
          # unrecoverable error on another worker caused closure cancellation,
          # while this thread was waiting for task states. So raise a new
          # CancelledError.
          else:
            raise errors.CancelledError(
                None, None, "The corresponding function was cancelled while "
                "attempting to recover from worker failure.")
        # Else, preemption
        self._handle_failure_and_recovery(e, on_failure_fn,
                                          on_transient_failure_fn,
                                          on_recovery_fn, worker_device_name)
        return

      #  else, if timeout: log
      if self._cluster._record_and_ignore_transient_timeouts(e):  # pylint: disable=protected-access
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "This derived error is ignored and not reported to users.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return
      raise e

  def _handle_failure_and_recovery(self,
                                   e,
                                   on_failure_fn,
                                   on_transient_failure_fn,
                                   on_recovery_fn,
                                   worker_device_name):
    # [FUNCTION DOC]
    # HELPER: part of wait_on_failure. Parameters as in wait_on_failure, plus e, the caught
    # error. Calls the failure callback with the error, signals the handler thread that a cluster
    # update is due and waits (up to the maximum recovery time) for the update to finish. If the
    # recovery stored an error it is raised once and cleared. Then logs recovery and, if given,
    # runs the recovery callback inside another failure-handling context. Returns nothing.
    """Call failure fn, wait for cluster to recover, then call recovery fn.

    Args:
      e: the Exception thrown during closure execution.
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
      on_recovery_fn: an optional function to run when a worker is recovered
        from preemption.
      worker_device_name: the device name of the worker instance that is passing
        through the failure.
    """
    if on_failure_fn:
      on_failure_fn(e)
    # update server def
    with self._cluster_update_lock:
      self._cluster_due_for_update_or_finish.set()
      self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
      if self._error_from_recovery:
        # TODO(yuefengz): there is only one worker that will get this error.
        # Ideally we should let all workers notified by `_worker_up_cond` get
        # this error.
        try:
          raise self._error_from_recovery
        finally:
          self._error_from_recovery = None
      logging.info("Worker %s has been recovered.", worker_device_name)

    if on_recovery_fn:
      logging.info("Worker %s calling on_recovery_fn", worker_device_name)
      with self.wait_on_failure(
          on_recovery_fn=on_recovery_fn,
          on_transient_failure_fn=on_transient_failure_fn,
          worker_device_name=worker_device_name):
        on_recovery_fn()

  def _log_ps_failure_and_raise(self, e, ps_index):
    # [FUNCTION DOC]
    # HELPER: part of wait_on_failure. Parameters: e, the caught unavailable error; ps_index,
    # integer index of the failed parameter server (0 when the coordination service itself is
    # unreachable). Logs it, stops this handler and always raises a PS unavailable error wrapping
    # e (the wrapper asserts e is an unavailable error).
    logging.info("Parameter server failure detected at PS task %d", ps_index)
    self.stop()
    raise PSUnavailableError(e)

  def _get_task_states(self):
    # [FUNCTION DOC]
    # HELPER: called periodically by the poller timer thread. Queries the eager context for the
    # task states of all workers and parameter servers and stores them. On unavailable or
    # internal errors it stores nothing/null (coordination service down); an internal error whose
    # message does not say the service is disabled is re-raised. Always wakes all threads waiting
    # for a new state. Returns nothing.
    """Get task states and reset to None if coordination service is down."""
    try:
      self._task_states = context.context().get_task_states(
          [("worker", self._num_workers), ("ps", self._num_ps)]
      )
    except (errors.UnavailableError, errors.InternalError) as e:
      if isinstance(
          e, errors.InternalError
      ) and "coordination service is not enabled" not in str(e).lower():
        raise
      # Coordination service is down
      self._task_states = None
    with self._next_task_state_cond:
      self._next_task_state_cond.notify_all()

  def _preemption_handler(self):
    # [FUNCTION DOC]
    # HELPER: body of the daemon preemption thread. Loops: waits for the update-due signal and
    # exits when the handler was stopped. Otherwise refreshes the server definition in the eager
    # context; on success notifies waiting workers. On failure waits for new task states and
    # records the error for workers only if task states are unavailable or a parameter server
    # failed, then notifies and logs that the update failed (the loop retries). Returns nothing.
    """A loop that handles preemption.

    This loop waits for signal of worker preemption and upon worker preemption,
    it waits until all workers are back and updates the cluster about the
    restarted workers.
    """
    assert self._should_preemption_thread_run
    while True:
      self._cluster_due_for_update_or_finish.wait()
      if not self._should_preemption_thread_run:
        logging.info("Stopping the failure handing thread.")
        break

      with self._cluster_update_lock:
        try:
          # TODO(haoyuzhang): support partial cluster recovery
          logging.info("Cluster now being recovered.")
          context.context().update_server_def(self._server_def)

          # Cluster updated successfully, clear the update signal, and notify
          # all workers that they are recovered from failure.
          logging.info("Cluster successfully recovered.")
          self._notify_cluster_update()
        except Exception as e:  # pylint: disable=broad-except
          logging.info("Error occurred while updating server def: %s", e)
          # Wait for the next set of states from the task state poller
          with self._next_task_state_cond:
            self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 2)
          # If a PS is preempted, set the error
          if not self._task_states:
            self._error_from_recovery = e
          else:
            ps_states = self._task_states[self._num_workers:]
            # Check for PS failure
            if any(ps_states):
              self._error_from_recovery = e
          # Else, likely another worker failed. Just log and retry
          self._notify_cluster_update()
          # NOTE: Since the first RPC (GetStatus) of update_server_def is
          # currently blocking by default, error should only happen if:
          # (1) More workers failed while waiting for the previous workers to
          #     come back;
          # (2) Worker failed when exchanging subsequent RPCs after the first
          #     RPC returns.
          # Consider adding backoff retry logic if we see the error logged
          # too frequently.
          logging.error("Cluster update failed with error: %s. Retrying...", e)

  def _notify_cluster_update(self):
    # [FUNCTION DOC]
    # HELPER: wakes every worker waiting for recovery and, unless the handler is stopping, clears
    # the update-due signal. Called with the update lock held. Returns nothing.
    self._worker_up_cond.notify_all()
    # The check for _should_preemption_thread_run is necessary since the
    # `stop` may have already set _cluster_due_for_update_or_finish.
    if self._should_preemption_thread_run:
      self._cluster_due_for_update_or_finish.clear()


class WorkerPreemptionHandler(object):
  # [CLASS DOC]
  # Default failure handler. Classifies errors raised while a worker runs a closure: transient
  # PS connectivity errors, transient timeouts and derived cancellations are ignored; worker
  # unavailability triggers a wait for the worker to return and a server definition update done
  # by a daemon thread; everything else (such as PS failure) is re-raised. State: server
  # definition, owning cluster, update lock, update-due event, recovery condition and an error
  # from recovery.
  """Handles worker preemptions."""

  def __init__(self, server_def, cluster):
    # [FUNCTION DOC]
    # Parameters: server_def, the server definition used to refresh the cluster after recovery;
    # cluster, the owning Cluster. Creates the locks, event and condition and starts the daemon
    # preemption handler thread. Returns nothing.
    self._server_def = server_def
    self._cluster = cluster
    self._cluster_update_lock = threading.Lock()
    self._cluster_due_for_update_or_finish = threading.Event()
    self._worker_up_cond = threading.Condition(self._cluster_update_lock)
    self._error_from_recovery = None
    self._should_preemption_thread_run = True
    self._preemption_handler_thread = threading.Thread(
        target=self._preemption_handler,
        name="WorkerPreemptionHandler",
        daemon=True)
    self._preemption_handler_thread.start()

  def stop(self):
    # [FUNCTION DOC]
    # Tells the preemption thread to exit and wakes it. The thread is not joined. Returns
    # nothing.
    """Ensure the worker preemption thread is closed."""
    self._should_preemption_thread_run = False
    with self._cluster_update_lock:
      self._cluster_due_for_update_or_finish.set()
    # TODO(yuefengz): The preemption handler thread shouldn't be terminated
    # asynchronously since it touches eager context which is a process-wide
    # singleton. The problem is in OSS unit tests will time out.

  def _validate_preemption_failure(self, e):
    # [FUNCTION DOC]
    # HELPER: part of wait_on_failure and the handler thread. Parameter e: an exception. If it
    # looks like a worker failure and closures are not being cancelled, increments the worker
    # failures counter metric and returns nothing; otherwise re-raises e.
    """Validates that the given exception represents worker preemption."""

    # Only categorize the failure as a worker preemption if the cancellation
    # manager did not attempt to cancel the blocking operations.
    if _is_worker_failure(e) and (
        not self._cluster.closure_queue._cancellation_mgr.is_cancelled):  # pylint: disable=protected-access
      metric_utils.monitor_increment_counter("worker_failures")
      return
    raise e

  @contextlib.contextmanager
  def wait_on_failure(self,
                      on_failure_fn=None,
                      on_transient_failure_fn=None,
                      on_recovery_fn=None,
                      worker_device_name="(unknown)"):
    # [FUNCTION DOC]
    # Context manager wrapping closure execution. Parameters: on_failure_fn, optional callback
    # taking the error, run on worker failure; on_transient_failure_fn, optional callback run when
    # a failure is ignored; on_recovery_fn, optional callback run after recovery;
    # worker_device_name, name used in logs. Yields nothing. Catches op errors, closure input or
    # aborted errors and type errors. It ignores (logs, calls the transient callback, returns)
    # transient PS connectivity failures, ignorable timeouts and derived cancelled errors that
    # mention a job. Otherwise it validates the error (re-raising unless it is a worker failure),
    # runs the failure callback, signals an update and waits up to the maximum recovery time,
    # raises (once) any error stored by the recovery thread, and finally runs the recovery
    # callback inside another failure-handling context.
    """Catches worker preemption error and wait until failed workers are back.

    Args:
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
      on_recovery_fn: an optional function to run when a worker is recovered
        from preemption.
      worker_device_name: the device name of the worker instance that is passing
        through the failure.

    Yields:
      None.
    """
    assert self._should_preemption_thread_run
    try:
      yield
    except (errors.OpError, ClosureInputError,
            ClosureAbortedError, TypeError) as e:
      # If the error is due to temporary connectivity issues between worker and
      # ps, put back closure, ignore error and do not mark worker as failure.
      if self._cluster._record_and_ignore_transient_ps_failure(e):  # pylint: disable=protected-access
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "It is treated as a transient connectivity failure for now.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return

      # If the error is due to temporary connectivity issues that cause the
      # server-side RPCs to be cancelled, TF might not abort the step and the
      # closure might timeout. The coordinator ignores certain amount of such
      # failures without marking worker as failure.
      if self._cluster._record_and_ignore_transient_timeouts(e):  # pylint: disable=protected-access
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "This derived error is ignored and not reported to users.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return

      # Ignoring derived CancelledErrors to tolerate transient failures in
      # PS-worker communication, which initially exposed as an UnavailableError
      # and then lead to sub-function cancellation, subsequently getting
      # reported from worker to chief as CancelledError.
      # We do not mark either worker or PS as failed due to only CancelledError.
      # If there are real (non-transient) failures, they must also be reported
      # as other errors (UnavailableError most likely) in closure executions.
      if isinstance(e, errors.CancelledError) and "/job:" in str(e):
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "This derived error is ignored and not reported to users.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return

      # This reraises the error, if it's not considered recoverable; otherwise,
      # the following failure recovery logic run. At this time, only worker
      # unavailability is recoverable. PS unavailability as well as other
      # errors in the user function is not recoverable.
      self._validate_preemption_failure(e)

      logging.error("Worker %s failed with %r:%s", worker_device_name, e, e)
      if on_failure_fn:
        on_failure_fn(e)

      with self._cluster_update_lock:
        self._cluster_due_for_update_or_finish.set()
        self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
        if self._error_from_recovery:
          # TODO(yuefengz): there is only one worker that will get this error.
          # Ideally we should let all workers notified by `_worker_up_cond` get
          # this error.
          try:
            raise self._error_from_recovery
          finally:
            self._error_from_recovery = None
        logging.info("Worker %s has been recovered.", worker_device_name)

      if on_recovery_fn:
        logging.info("Worker %s calling on_recovery_fn", worker_device_name)
        with self.wait_on_failure(
            on_recovery_fn=on_recovery_fn,
            on_transient_failure_fn=on_transient_failure_fn,
            worker_device_name=worker_device_name):
          on_recovery_fn()

  def _preemption_handler(self):
    # [FUNCTION DOC]
    # HELPER: body of the daemon preemption thread. Loops: waits for the update-due signal and
    # exits when stopped. Otherwise refreshes the server definition in the eager context (timed
    # by a metric); on success wakes the waiting workers and clears the signal. On failure,
    # validates the error: if it is not a worker failure (for example a PS failure) it is stored
    # as the recovery error, waiting workers are woken and the signal cleared. Logs that the
    # update failed and keeps looping. Returns nothing.
    """A loop that handles preemption.

    This loop waits for signal of worker preemption and upon worker preemption,
    it waits until all workers are back and updates the cluster about the
    restarted workers.
    """
    assert self._should_preemption_thread_run
    while True:
      self._cluster_due_for_update_or_finish.wait()
      if not self._should_preemption_thread_run:
        logging.info("Stopping the failure handing thread.")
        break

      with self._cluster_update_lock:
        try:
          # TODO(haoyuzhang): support partial cluster recovery
          logging.info("Cluster now being recovered.")
          with metric_utils.monitored_timer("server_def_update"):
            context.context().update_server_def(self._server_def)

          # Cluster updated successfully, clear the update signal, and notify
          # all workers that they are recovered from failure.
          logging.info("Cluster successfully recovered.")
          self._worker_up_cond.notify_all()
          # The check for _should_preemption_thread_run is necessary since the
          # `stop` may have already set _cluster_due_for_update_or_finish.
          if self._should_preemption_thread_run:
            self._cluster_due_for_update_or_finish.clear()
        except Exception as e:  # pylint: disable=broad-except
          logging.info("Error occurred while updating server def: %s", e)
          try:
            self._validate_preemption_failure(e)
          except Exception as ps_e:  # pylint: disable=broad-except
            logging.info("Error that occurred while updating server def is not "
                         "a worker failure. So set it as _error_from_recovery")
            # In this case, a parameter server fails. So we raise this error to
            # the caller of `wait_on_failure`.
            self._error_from_recovery = ps_e
            self._worker_up_cond.notify_all()
            if self._should_preemption_thread_run:
              self._cluster_due_for_update_or_finish.clear()
          # NOTE: Since the first RPC (GetStatus) of update_server_def is
          # currently blocking by default, error should only happen if:
          # (1) More workers failed while waiting for the previous workers to
          #     come back;
          # (2) Worker failed when exchanging subsequent RPCs after the first
          #     RPC returns.
          # Consider adding backoff retry logic if we see the error logged
          # too frequently.
          logging.error("Cluster update failed with error: %s. Retrying...", e)


class Worker(object):
  # [CLASS DOC]
  # Coordinator-side representation of one remote worker. Each instance owns an executor for
  # remote execution and one daemon thread that loops fetching closures (from its tagged queue
  # and the global queue) and running them under the cluster's failure handler. It also tracks
  # weak references to the resource remote values created for it so they can be rebuilt after the
  # worker recovers, and whether it is currently dead with an error.
  """A worker in a cluster.

  Attributes:
    worker_index: The index of the worker in the cluster.
    device_name: The device string of the worker, e.g. "/job:worker/task:1".
    executor: The worker's executor for remote function execution.
    failure_handler: The failure handler used to handler worker preemption
      failure.
  """

  def __init__(self, worker_index, device_name, cluster):
    # [FUNCTION DOC]
    # Parameters: worker_index, integer index of the worker; device_name, its device string;
    # cluster, the owning Cluster (supplies the failure handler and queues). Creates a
    # synchronous executor, initializes bookkeeping and starts the daemon closure-processing
    # thread last, after initialization completes. Returns nothing.
    self.worker_index = worker_index
    self.device_name = device_name
    self.executor = executor.new_executor(enable_async=False)
    self.failure_handler = cluster.failure_handler
    self._cluster = cluster
    self._resource_tracking_lock = threading.Lock()
    self._resource_remote_value_refs = []
    self._is_dead_with_error = None
    self._should_worker_thread_run = True

    # Worker threads need to start after `Worker`'s initialization.
    threading.Thread(target=self._process_queue,
                     name="WorkerClosureProcessingLoop-%d" % self.worker_index,
                     daemon=True).start()

  def stop(self):
    # [FUNCTION DOC]
    # Asks the worker thread to exit after its current iteration; does not join it. Returns
    # nothing.
    """Ensure the worker thread is closed."""
    self._should_worker_thread_run = False

  def _schedule_resource(self, closure):
    # [FUNCTION DOC]
    # HELPER: puts a resource closure into the tagged queue of this worker, so only this worker
    # runs it. Parameter closure: the resource closure. Returns nothing.
    self._cluster.closure_queue.put(closure, tag=self.worker_index)

  def _set_resources_aborted(self, e):
    # [FUNCTION DOC]
    # HELPER: failure handling. Parameter e: the causing exception. Marks every still-alive
    # resource remote value of this worker as aborted with a closure aborted error wrapping e, so
    # later readers see an error. Caller should hold the resource tracking lock. Returns nothing.
    """Set the resource ABORTED and add an error to it."""
    # TODO(yuefengz): maybe we can query whether a tensor is valid or not
    # instead of marking a tensor aborted?
    logging.info("[Worker %d] Clearing all resources.", self.worker_index)
    for weakref_resource in self._resource_remote_value_refs:
      resource = weakref_resource()
      if resource:
        # It is important to set an error on an aborted RemoteValue from a
        # ResourceClosure because its failure will not trigger the worker thread
        # to raise error immediately and the worker may continue executing
        # closures taking it as an input. The error will then be correctly
        # reported to users.
        resource._set_aborted(ClosureAbortedError(e))  # pylint: disable=protected-access

  def _on_closure_failure(self, closure, e):
    # [FUNCTION DOC]
    # HELPER: failure callback for normal closures. Parameters: closure, the failed closure; e,
    # the error. Puts the closure back into the global queue, then under the tracking lock
    # records that the worker is dead with error e and aborts its resources. Returns nothing.
    logging.info("[Worker %d] Putting back a closure after it failed.",
                 self.worker_index)
    self._cluster.closure_queue.put_back(closure)

    with self._resource_tracking_lock:
      self._is_dead_with_error = e
      self._set_resources_aborted(e)

  def _on_resource_closure_failure(self, e):
    # [FUNCTION DOC]
    # HELPER: failure callback for resource closures. Under the tracking lock, records the dead
    # state with error e, discards this worker's tagged queue and aborts its resources (they are
    # rebuilt on recovery). Returns nothing.
    """Clear tagged queue to ensure resource closures are rebuilt.

    Args:
      e: The exception arisen from the resource closure.
    """
    logging.info("[Worker %d] Clearing tagged queue after resource closure "
                 "failure.", self.worker_index)
    with self._resource_tracking_lock:
      self._is_dead_with_error = e
      # No locking on queue is needed since
      #  * get will not happen concurrently here.
      #  * put to the specific tagged queue will be guarded by
      #    `self._resource_tracking_lock`.
      self._cluster.closure_queue.clear_tag_unlocked(self.worker_index)
      self._set_resources_aborted(e)

  def _on_worker_recovery(self):
    # [FUNCTION DOC]
    # HELPER: recovery callback. Under the tracking lock, reschedules the closure of every
    # still-alive resource remote value on this worker and clears the dead-with-error state.
    # Returns nothing.
    logging.info("[Worker %d] calling _on_worker_recovery", self.worker_index)
    with self._resource_tracking_lock:
      for weakref_resource in self._resource_remote_value_refs:
        resource = weakref_resource()
        if resource:
          self._schedule_resource(resource._closure)  # pylint: disable=protected-access
      self._is_dead_with_error = False

  def _process_closure(self, closure):
    # [FUNCTION DOC]
    # HELPER: worker thread step for a normal closure. Parameter closure: the closure taken from
    # the queue. Runs it under the failure handler (failure puts it back, transient failure puts
    # it back, recovery reschedules resources), then fetches its output to the coordinator (timed
    # by a metric) and marks it finished. Any other exception is logged unless it is a
    # cancellation, stored on the output remote value and reported to the queue as failed.
    # Returns nothing.
    """Runs a closure with preemption handling."""
    try:
      with self.failure_handler.wait_on_failure(
          on_failure_fn=lambda e: self._on_closure_failure(closure, e),
          on_transient_failure_fn=(
              lambda: self._cluster.closure_queue.put_back(closure)),
          on_recovery_fn=self._on_worker_recovery,
          worker_device_name=self.device_name):
        closure.execute_on(self)
        with metric_utils.monitored_timer("remote_value_fetch"):
          # Copy the remote tensor to local (the coordinator) in case worker
          # becomes unavailable at a later time.
          closure.maybe_call_with_output_remote_value(lambda r: r.get())
        self._cluster.closure_queue.mark_finished()
    except Exception as e:  # pylint: disable=broad-except
      # Avoid logging the derived cancellation error
      if not isinstance(e, errors.CancelledError):
        logging.error(
            " /job:worker/task:%d encountered the following error when "
            "processing closure: %r:%s", self.worker_index, e, e)
      closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))  # pylint: disable=protected-access
      self._cluster.closure_queue.mark_failed(e)

  def _process_resource_closure(self, closure):
    # [FUNCTION DOC]
    # HELPER: worker thread step for a resource closure. Parameter closure: a resource closure
    # whose tag must equal this worker's index (asserted). Runs it under the failure handler
    # (transient failure re-runs it recursively). Other exceptions are logged (errors only if not
    # cancellations) and stored on the output remote value; the queue is not told it failed.
    # Returns nothing.
    """Run the given resource closure with preemption handling."""
    assert closure.tag == self.worker_index
    try:
      with self.failure_handler.wait_on_failure(
          on_failure_fn=self._on_resource_closure_failure,
          on_transient_failure_fn=(
              lambda: self._process_resource_closure(closure)),
          on_recovery_fn=self._on_worker_recovery,
          worker_device_name=self.device_name):
        closure.execute_on(self)
    except Exception as e:  # pylint: disable=broad-except
      # Avoid logging the derived cancellation error
      logging.info("[Worker %d] got an exception when processing resource "
                   "closure", self.worker_index)
      if not isinstance(e, errors.CancelledError):
        logging.error(
            " /job:worker/task:%d encountered the following error when "
            "processing resource closure: %r:%s", self.worker_index, e, e)
      closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))  # pylint: disable=protected-access

  def _maybe_delay(self):
    # [FUNCTION DOC]
    # HELPER: staggers worker start. Reads two environment variables (start delay per worker
    # index and its maximum, both integer seconds, default zero), multiplies the delay by this
    # worker's index, caps it by the maximum when that is nonzero, logs when positive and sleeps
    # for that many seconds. Returns nothing.
    """Delay if corresponding env vars are set."""
    # If the following two env vars variables are set. Scheduling for workers
    # will start in a staggered manner. Worker i will wait for
    # `TF_COORDINATOR_SCHEDULE_START_DELAY` * i seconds, not exceeding
    # `TF_COORDINATOR_SCHEDULE_START_DELAY_MAX`.
    delay_secs = int(os.environ.get("TF_COORDINATOR_SCHEDULE_START_DELAY", "0"))
    delay_secs *= self.worker_index
    delay_cap = int(
        os.environ.get("TF_COORDINATOR_SCHEDULE_START_DELAY_MAX", "0"))
    if delay_cap:
      delay_secs = min(delay_secs, delay_cap)
    if delay_secs > 0:
      logging.info(" Worker %d sleeping for %d seconds before running function",
                   self.worker_index, delay_secs)
    time.sleep(delay_secs)

  def _process_queue(self):
    # [FUNCTION DOC]
    # HELPER: main loop of the worker thread. After the optional start delay, repeatedly gets a
    # closure (tagged queue of this worker first, then global) and dispatches it as resource or
    # normal closure. If the worker was stopped or nothing is returned, it cancels any closure
    # obtained and exits. Drops its reference to each processed closure so the coordinator can be
    # garbage collected. Returns nothing.
    """Function running in a worker thread to process closure queues."""
    self._maybe_delay()
    while self._should_worker_thread_run:
      closure = self._cluster.closure_queue.get(tag=self.worker_index)
      if not self._should_worker_thread_run or closure is None:
        if closure is not None:
          closure.mark_cancelled()
        return
      if isinstance(closure, ResourceClosure):
        self._process_resource_closure(closure)
      else:
        self._process_closure(closure)
      # To properly stop the worker and preemption threads, it is important that
      # `ClusterCoordinator` object is not held onto so its `__del__` can be
      # called. By removing the reference to the `closure` that has already been
      # processed, we ensure that the `closure` object is released, while
      # getting the next `closure` at above `self._cluster.closure_queue.get()`
      # call.
      del closure

  def create_resource(self, function, args=None, kwargs=None):
    # [FUNCTION DOC]
    # Parameters: function, the resource-building function (traced function, concrete function
    # or plain callable); args and kwargs, optional arguments. Wraps them into a resource
    # closure using the cluster's resource cancellation manager, then registers and schedules it
    # on this worker. Returns the remote value for the resource.
    """Asynchronously creates a per-worker resource represented by a `RemoteValue`.

    Args:
      function: the resource function to be run remotely. It should be a
        `tf.function`, a concrete function or a Python function.
      args: positional arguments to be passed to the function.
      kwargs: keyword arguments to be passed to the function.

    Returns:
      one or several RemoteValue objects depending on the function return
      values.
    """
    closure = ResourceClosure(
        function,
        self._cluster.resource_cancellation_mgr,
        args=args,
        kwargs=kwargs)
    return self._register_and_schedule_resource_closure(closure)

  def create_variable_resource(self, function, args=None, kwargs=None):
    # [FUNCTION DOC]
    # Same as create_resource but uses a per-worker variable closure, so the returned remote
    # value is a remote variable. Parameters: function, args (optional), kwargs (optional).
    # Returns the remote variable.
    """Create a per-worker variable."""
    closure = PerWorkerVariableClosure(
        function,
        self._cluster.resource_cancellation_mgr,
        args=args,
        kwargs=kwargs)
    return self._register_and_schedule_resource_closure(closure)

  def _register_and_schedule_resource_closure(self, closure):
    # [FUNCTION DOC]
    # HELPER: shared by create_resource and create_variable_resource. Parameter closure: a
    # resource closure. Builds its output remote value and, under the tracking lock, registers it
    # for rebuilding; if the worker is currently dead with an error the value is marked aborted
    # with that error, otherwise the closure is scheduled on this worker. Returns the remote
    # value.
    """Build remote value for, register for reconstruction, and schedule."""
    # Some notes about the concurrency: currently all the activities related to
    # the same worker such as creating resources, setting resources' aborted
    # status, and executing closures happen on the same thread. This allows us
    # to have simpler logic of concurrency.

    resource_remote_value = closure.build_output_remote_value()
    with self._resource_tracking_lock:
      self._register_resource(resource_remote_value)
      if self._is_dead_with_error:
        resource_remote_value._set_aborted(  # pylint: disable=protected-access
            ClosureAbortedError(self._is_dead_with_error))
      else:
        self._schedule_resource(closure)
    return resource_remote_value

  def _register_resource(self, resource_remote_value):
    # [FUNCTION DOC]
    # HELPER: remembers a weak reference to a resource remote value for rebuilding after
    # recovery. Parameter resource_remote_value: must be a remote value, else a value error is
    # raised. Returns nothing.
    if not isinstance(resource_remote_value, RemoteValue):
      raise ValueError("Resource being registered is not of type "
                       "`tf.distribute.experimental.coordinator.RemoteValue`.")
    self._resource_remote_value_refs.append(weakref.ref(resource_remote_value))


class Cluster(object):
  # [CLASS DOC]
  # Owns everything needed to run closures on a set of workers: one Worker per worker task, the
  # coordinated closure queue, a failure handler (chosen by an environment variable) and a
  # cancellation manager for resource closures. Also keeps counters that let it ignore a limited
  # number of transient PS failure reports and worker timeouts. Created once per cluster
  # coordinator from a parameter server strategy.
  """A cluster with workers.

  We assume all function errors are fatal and based on this assumption our
  error reporting logic is:
  1) Both `schedule` and `join` can raise a non-retryable error which is the
  first error seen by the coordinator from any previously scheduled functions.
  2) When an error is raised, there is no guarantee on how many previously
  scheduled functions have been executed; functions that have not been executed
  will be thrown away and marked as cancelled.
  3) After an error is raised, the internal state of error will be cleared.
  I.e. functions can continue to be scheduled and subsequent calls of `schedule`
  or `join` will not raise the same error again.

  Attributes:
    failure_handler: The failure handler used to handler worker preemption
      failure.
    workers: a list of `Worker` objects in the cluster.
    closure_queue: the global Closure queue.
    resource_cancellation_mgr: the cancellation manager used to cancel resource
      closures.
  """

  def __init__(self, strategy):
    # [FUNCTION DOC]
    # Parameter strategy: a parameter server strategy giving the worker and PS counts. Reads
    # environment variables: the threshold of ignored transient PS failure reports (default 3),
    # the threshold of ignored transient timeouts (default a tenth of the worker count) and a flag
    # selecting the coordination-service failure handler. Creates the closure queue, the failure
    # handler (using the current server definition), one Worker per worker (each starts its
    # thread) and the resource cancellation manager. Returns nothing.
    """Initializes the cluster instance."""

    self._num_workers = strategy._num_workers
    self._num_ps = strategy._num_ps

    # Ignore PS failures reported by workers due to transient connection errors.
    # Transient connectivity issues between workers and PS are relayed by the
    # workers to the coordinator, leading the coordinator to believe that there
    # are PS failures. The difference between transient vs. permanent PS failure
    # is the number of reports from the workers. When this env var is set to a
    # positive integer K, the coordinator ignores up to K reports of a failed PS
    # task, i.e., only when there are more than K trials of executing closures
    # fail due to errors from the same PS instance do we consider the PS
    # instance encounters a failure.
    # TODO(b/164279603): Remove this workaround when the underlying connectivity
    # issue in gRPC server is resolved.
    self._transient_ps_failures_threshold = int(
        os.environ.get("TF_COORDINATOR_IGNORE_TRANSIENT_PS_FAILURES", 3))
    self._potential_ps_failures_lock = threading.Lock()
    self._potential_ps_failures_count = [0] * self._num_ps

    # Ignore worker timeouts due to transient connection errors.
    # Transient connectivity issues might cause the server side to unexpectedly
    # cancel RPC handling logic, leading to closure execution timeouts. When
    # the _transient_timeout_threshold is set to a positive number, the cluster
    # coordinator ignores DeadlineExceeded errors from workers for the specified
    # times before raising the error to users.
    self._transient_timeouts_threshold = int(
        os.environ.get("TF_COORDINATOR_IGNORE_TRANSIENT_TIMEOUTS",
                       self._num_workers // 10))
    self._transient_timeouts_lock = threading.Lock()
    self._transient_timeouts_count = 0

    self.closure_queue = _CoordinatedClosureQueue()
    # Set this environment variable to use an experimental
    # integration with the runtime coordination service to aid in failure
    # detection and handling. This will not affect the functionality of
    # the strategy or cluster coordinator, but is off by default.
    if os.getenv("TF_PSS_ENABLE_COORDINATION_SERVICE"):
      self.failure_handler = CoordinationServicePreemptionHandler(
          context.get_server_def(), self,
      )
    else:
      self.failure_handler = WorkerPreemptionHandler(context.get_server_def(),
                                                     self)
    worker_device_strings = [
        "/job:worker/replica:0/task:%d" % i for i in range(self._num_workers)
    ]
    self.workers = [
        Worker(i, w, self) for i, w in enumerate(worker_device_strings)
    ]

    # Cancellation manager for all resource closures.
    self.resource_cancellation_mgr = cancellation.CancellationManager()

  def stop(self):
    # [FUNCTION DOC]
    # Shuts the cluster down in order: stops the failure handler, every worker and the closure
    # queue, then starts cancelling resource-building functions. Logs each stage. Returns
    # nothing.
    """Stop worker, worker preemption threads, and the closure queue."""
    logging.info("Stopping cluster, starting with failure handler")
    self.failure_handler.stop()

    logging.info("Stopping workers")
    for worker in self.workers:
      worker.stop()
    logging.info("Stopping queue")
    self.closure_queue.stop()
    logging.info("Start cancelling remote resource-building functions")
    self.resource_cancellation_mgr.start_cancel()

  def _record_and_ignore_transient_ps_failure(self, e):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the per-task failure counts are never reset, so occasional transient reports
    # accumulate over the whole run until the threshold is reached.
    # HELPER: used by the failure handler. Parameter e: the caught error. Returns false if the
    # threshold is not positive or e is not a PS failure. Otherwise increments the counter of
    # every PS task named in the error text and returns false if any counter reaches the
    # threshold; returns true (ignore) otherwise.
    """Records potential PS failures and return if failure should be ignored."""
    if self._transient_ps_failures_threshold <= 0 or not _is_ps_failure(e):
      return False

    ps_tasks = _extract_failed_ps_instances(str(e))
    with self._potential_ps_failures_lock:
      for t in ps_tasks:
        self._potential_ps_failures_count[t] += 1
        # The number of UnavailableError encountered on this PS task exceeds the
        # maximum number of ignored error
        if (self._potential_ps_failures_count[t] >=
            self._transient_ps_failures_threshold):
          return False
    return True

  def _record_and_ignore_transient_timeouts(self, e):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the timeout count is never reset, so timeouts accumulate over the whole run.
    # HELPER: used by the failure handlers. Parameter e: the caught error. Returns false if the
    # threshold is not positive or e is not a deadline-exceeded error. Otherwise increments the
    # count under a lock and returns false once it reaches the threshold; true means ignore.
    """Records observed timeout error and return if it should be ignored."""
    if self._transient_timeouts_threshold <= 0:
      return False
    if not isinstance(e, errors.DeadlineExceededError):
      return False
    with self._transient_timeouts_lock:
      self._transient_timeouts_count += 1
      if self._transient_timeouts_count >= self._transient_timeouts_threshold:
        return False
    return True

  def schedule(self, function, args, kwargs):
    # [FUNCTION DOC]
    # Parameters: function, the callable to run on a worker; args, positional arguments; kwargs,
    # keyword arguments (both required here, may be null). Builds a closure with the queue's
    # current cancellation manager, creates its output remote value and puts the closure in the
    # global queue (may block when full or raise a stored earlier error). Returns the remote
    # value.
    """Schedules `function` to be dispatched to a worker for execution.

    Args:
      function: The function to be dispatched to a worker for execution
        asynchronously.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `RemoteValue` object.
    """
    closure = Closure(
        function,
        self.closure_queue._cancellation_mgr,  # pylint: disable=protected-access
        args=args,
        kwargs=kwargs)
    ret = closure.build_output_remote_value()
    self.closure_queue.put(closure)
    return ret

  def join(self):
    # [FUNCTION DOC]
    # Waits (without timeout) until the closure queue is drained and nothing is in flight; raises
    # the first stored closure error if any. Returns nothing.
    """Blocks until all scheduled functions are executed."""
    self.closure_queue.wait()

  def done(self):
    # [FUNCTION DOC]
    # Returns true if no closure is queued or in flight; raises a stored closure error if one
    # exists.
    """Returns true if all scheduled functions are executed."""
    return self.closure_queue.done()


@tf_export("distribute.experimental.coordinator.ClusterCoordinator",
           "distribute.coordinator.ClusterCoordinator", v1=[])
class ClusterCoordinator(object):
  # [CLASS DOC]
  # Public entry point for parameter server training. It is a per-strategy singleton: creating
  # it again for the same strategy returns the existing instance, which is initialized only once.
  # It wraps a Cluster and exposes scheduling of functions, joining, polling completion, creating
  # per-worker datasets and fetching remote values. Exported in the public API under two names.
  # Destroying it stops the cluster.
  """An object to schedule and coordinate remote function execution.

  This class is used to create fault-tolerant resources and dispatch functions
  to remote TensorFlow servers.

  Currently, this class is not supported to be used in a standalone manner. It
  should be used in conjunction with a `tf.distribute` strategy that is designed
  to work with it. The `ClusterCoordinator` class currently only works
  `tf.distribute.experimental.ParameterServerStrategy`.

  __The `schedule`/`join` APIs__

  The most important APIs provided by this class is the `schedule`/`join` pair.
  The `schedule` API is non-blocking in that it queues a `tf.function` and
  returns a `RemoteValue` immediately. The queued functions will be dispatched
  to remote workers in background threads and their `RemoteValue`s will be
  filled asynchronously. Since `schedule` doesn’t require worker assignment, the
  `tf.function` passed in can be executed on any available worker. If the worker
  it is executed on becomes unavailable before its completion, it will be
  migrated to another worker. Because of this fact and function execution is not
  atomic, a function may be executed more than once.

  __Handling Task Failure__

  This class when used with
  `tf.distribute.experimental.ParameterServerStrategy`, comes with built-in
  fault tolerance for worker failures. That is, when some workers are not
  available for any reason to be reached from the coordinator, the training
  progress continues to be made with the remaining workers. Upon recovery of a
  failed worker, it will be added for function execution after datasets created
  by `create_per_worker_dataset` are re-built on it.

  When a parameter server fails, a `tf.errors.UnavailableError` is raised by
  `schedule`, `join` or `done`. In this case, in addition to bringing back the
  failed parameter server, users should restart the coordinator so that it
  reconnects to workers and parameter servers, re-creates the variables, and
  loads checkpoints. If the coordinator fails, after the user brings it back,
  the program will automatically connect to workers and parameter servers, and
  continue the progress from a checkpoint.

  It is thus essential that in user's program, a checkpoint file is periodically
  saved, and restored at the start of the program. If an
  `tf.keras.optimizers.Optimizer` is checkpointed, after restoring from a
  checkpoiont, its `iterations` property roughly indicates the number of steps
  that have been made. This can be used to decide how many epochs and steps are
  needed before the training completion.

  See `tf.distribute.experimental.ParameterServerStrategy` docstring for an
  example usage of this API.

  This is currently under development, and the API as well as implementation
  are subject to changes.
  """

  def __new__(cls, strategy):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the singleton check is not thread-safe (see the existing TODO), so concurrent
    # construction could create two instances.
    # Parameter strategy: the strategy object. Returns the coordinator cached on the strategy,
    # creating and caching a new bare instance first if none exists. Initialization happens in
    # the constructor.
    # `ClusterCoordinator` is kept as a single instance to a given `Strategy`.
    # TODO(rchao): Needs a lock for thread-safety
    if strategy._cluster_coordinator is None:
      strategy._cluster_coordinator = super(
          ClusterCoordinator, cls).__new__(cls)
    return strategy._cluster_coordinator

  def __init__(self, strategy):
    # [FUNCTION DOC]
    # Runs only on the first initialization of the instance. Parameter strategy: must be a
    # parameter server strategy (version 2); otherwise a value error is raised. Remembers it,
    # marks its extended object as used with a coordinator and creates the Cluster (which starts
    # worker and handler threads). Repeated calls do nothing. Returns nothing.
    """Initialization of a `ClusterCoordinator` instance.

    Args:
      strategy: a supported `tf.distribute.Strategy` object. Currently, only
        `tf.distribute.experimental.ParameterServerStrategy` is supported.

    Raises:
      ValueError: if the strategy being used is not supported.
    """
    if not getattr(self, "_has_initialized", False):
      if not hasattr(strategy, "_is_parameter_server_strategy_v2"):
        raise ValueError(
            "Only `tf.distribute.experimental.ParameterServerStrategy` "
            "is supported to work with "
            "`tf.distribute.experimental.coordinator.ClusterCoordinator` "
            "currently.")
      self._strategy = strategy
      self.strategy.extended._used_with_coordinator = True
      self._cluster = Cluster(strategy)
      self._has_initialized = True

  def __del__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if construction failed before the cluster was created (unsupported strategy),
    # destruction fails with a missing attribute error because the cluster does not exist.
    # Destructor: logs and stops the cluster (handler, workers, queue). Returns nothing.
    logging.info("ClusterCoordinator destructor: stopping cluster")
    self._cluster.stop()

  @property
  def strategy(self):
    # [FUNCTION DOC]
    # Property returning the strategy this coordinator was created with.
    """Returns the `Strategy` associated with the `ClusterCoordinator`."""
    return self._strategy

  def schedule(self, fn, args=None, kwargs=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the flag marking the strategy as currently scheduling is reset only on
    # success; if scheduling raises, the flag stays set because there is no cleanup block.
    # Parameters: fn, a traced function or concrete function (anything else raises a type
    # error); args, optional positional arguments; kwargs, optional keyword arguments; they may
    # contain per-worker values. Inside the strategy scope, with the scheduling flag set, hands
    # the function to the cluster. Returns the remote value for the output. May raise an error
    # from earlier scheduled functions.
    """Schedules `fn` to be dispatched to a worker for asynchronous execution.

    This method is non-blocking in that it queues the `fn` which will be
    executed later and returns a
    `tf.distribute.experimental.coordinator.RemoteValue` object immediately.
    `fetch` can be called on it to wait for the function execution to finish
    and retrieve its output from a remote worker. On the other hand, call
    `tf.distribute.experimental.coordinator.ClusterCoordinator.join` to wait for
    all scheduled functions to finish.

    `schedule` guarantees that `fn` will be executed on a worker at least once;
    it could be more than once if its corresponding worker fails in the middle
    of its execution. Note that since worker can fail at any point when
    executing the function, it is possible that the function is partially
    executed, but `tf.distribute.experimental.coordinator.ClusterCoordinator`
    guarantees that in those events, the function will eventually be executed on
    any worker that is available.

    If any previously scheduled function raises an error, `schedule` will raise
    any one of those errors, and clear the errors collected so far. What happens
    here, some of the previously scheduled functions may have not been executed.
    User can call `fetch` on the returned
    `tf.distribute.experimental.coordinator.RemoteValue` to inspect if they have
    executed, failed, or cancelled, and reschedule the corresponding function if
    needed.

    When `schedule` raises, it guarantees that there is no function that is
    still being executed.

    At this time, there is no support of worker assignment for function
    execution, or priority of the workers.

    `args` and `kwargs` are the arguments passed into `fn`, when `fn` is
    executed on a worker. They can be
    `tf.distribute.experimental.coordinator.PerWorkerValues` and in this case,
    the argument will be substituted with the corresponding component on the
    target worker. Arguments that are not
    `tf.distribute.experimental.coordinator.PerWorkerValues` will be passed into
    `fn` as-is. Currently, `tf.distribute.experimental.coordinator.RemoteValue`
    is not supported to be input `args` or `kwargs`.

    Args:
      fn: A `tf.function`; the function to be dispatched to a worker for
        execution asynchronously. Regular python function is not supported to be
        scheduled.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `tf.distribute.experimental.coordinator.RemoteValue` object that
      represents the output of the function scheduled.

    Raises:
      Exception: one of the exceptions caught by the coordinator from any
        previously scheduled function, since the last time an error was thrown
        or since the beginning of the program.
    """
    if not isinstance(fn,
                      (def_function.Function, tf_function.ConcreteFunction)):
      raise TypeError(
          "`tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`"
          " only accepts a `tf.function` or a concrete function.")
    # Slot variables are usually created during function tracing time; thus
    # `schedule` needs to be called within the `strategy.scope()`.
    with self.strategy.scope():
      self.strategy.extended._being_scheduled = True  # pylint: disable=protected-access
      schedule_remote_value = self._cluster.schedule(
          fn, args=args, kwargs=kwargs)
      self.strategy.extended._being_scheduled = False  # pylint: disable=protected-access
      return schedule_remote_value

  def join(self):
    # [FUNCTION DOC]
    # Blocks until every scheduled function has finished; raises one stored error from earlier
    # functions (clearing it). Returns nothing.
    """Blocks until all the scheduled functions have finished execution.

    If any previously scheduled function raises an error, `join` will fail by
    raising any one of those errors, and clear the errors collected so far. If
    this happens, some of the previously scheduled functions may have not been
    executed. Users can call `fetch` on the returned
    `tf.distribute.experimental.coordinator.RemoteValue` to inspect if they have
    executed, failed, or cancelled. If some that have been cancelled need to be
    rescheduled, users should call `schedule` with the function again.

    When `join` returns or raises, it guarantees that there is no function that
    is still being executed.

    Raises:
      Exception: one of the exceptions caught by the coordinator by any
        previously scheduled function since the last time an error was thrown or
        since the beginning of the program.
    """
    self._cluster.join()

  def done(self):
    # [FUNCTION DOC]
    # Returns true if no function is pending or running; raises a stored error from earlier
    # functions if one exists.
    """Returns whether all the scheduled functions have finished execution.

    If any previously scheduled function raises an error, `done` will fail by
    raising any one of those errors.

    When `done` returns True or raises, it guarantees that there is no function
    that is still being executed.

    Returns:
      Whether all the scheduled functions have finished execution.
    Raises:
      Exception: one of the exceptions caught by the coordinator by any
        previously scheduled function since the last time an error was thrown or
        since the beginning of the program.
    """
    return self._cluster.done()

  def create_per_worker_dataset(self, dataset_fn):
    # [FUNCTION DOC]
    # Parameter dataset_fn: a dataset, distributed dataset or function returning a dataset.
    # Delegates to the values module helper, passing this coordinator, which builds a dataset on
    # every worker. Returns the per-worker dataset collection object; iterating it gives
    # per-worker iterators.
    """Create dataset on each worker.

    This creates dataset on workers from the input which can be either a
    `tf.data.Dataset`, a `tf.distribute.DistributedDataset` or a function which
    returns a dataset, and returns an object that represents the collection of
    those individual datasets. Calling `iter` on such collection of datasets
    returns a `tf.distribute.experimental.coordinator.PerWorkerValues`, which is
    a collection of iterators, where the iterators have been placed on
    respective workers.

    Calling `next` on a `PerWorkerValues` of iterator is unsupported. The
    iterator is meant to be passed as an argument into
    `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`. When
    the scheduled function is about to be executed by a worker, the
    function will receive the individual iterator that corresponds to the
    worker. The `next` method can be called on an iterator inside a
    scheduled function when the iterator is an input of the function.

    Currently the `schedule` method assumes workers are all the same and thus
    assumes the datasets on different workers are the same, except they may be
    shuffled differently if they contain a `dataset.shuffle` operation and a
    random seed is not set. Because of this, we also recommend the datasets to
    be repeated indefinitely and schedule a finite number of steps instead of
    relying on the `OutOfRangeError` from a dataset.


    Example:

    ```python
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver=...)
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy=strategy)

    @tf.function
    def worker_fn(iterator):
      return next(iterator)

    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(
          lambda x: tf.data.Dataset.from_tensor_slices([3] * 3))

    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iter = iter(per_worker_dataset)
    remote_value = coordinator.schedule(worker_fn, args=(per_worker_iter,))
    assert remote_value.fetch() == 3
    ```

    Args:
      dataset_fn: The dataset function that returns a dataset. This is to be
        executed on the workers.

    Returns:
      An object that represents the collection of those individual
      datasets. `iter` is expected to be called on this object that returns
      a `tf.distribute.experimental.coordinator.PerWorkerValues` of the
      iterators (that are on the workers).
    """
    return values_lib.get_per_worker_dataset(dataset_fn, self)

  def _create_per_worker_resources(self, fn, args=None, kwargs=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says synchronously, but each worker only schedules the resource
    # closure and returns a remote value immediately; nothing waits for completion.
    # HELPER: used internally for per-worker datasets. Parameters: fn, resource function; args,
    # kwargs optional. Creates a resource on every worker. Returns a per-worker value holding the
    # tuple of remote values, one per worker.
    """Synchronously create resources on the workers.

    The resources are represented by
    `tf.distribute.experimental.coordinator.RemoteValue`s.

    Args:
      fn: The function to be dispatched to all workers for execution
        asynchronously.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `tf.distribute.experimental.coordinator.PerWorkerValues` object, which
      wraps a tuple of `tf.distribute.experimental.coordinator.RemoteValue`
      objects.
    """
    results = []
    for w in self._cluster.workers:
      results.append(w.create_resource(fn, args=args, kwargs=kwargs))
    return PerWorkerValues(tuple(results))

  def _create_per_worker_variables(self, fn, args=None, kwargs=None):
    # [FUNCTION DOC]
    # HELPER: used internally to create per-worker variables. Parameters: fn, variable creation
    # function; args, kwargs optional. Schedules creation on every worker without waiting.
    # Returns a per-worker value holding the tuple of remote variables.
    """Asynchronously create variables on workers."""
    results = []
    for w in self._cluster.workers:
      results.append(w.create_variable_resource(fn, args=args, kwargs=kwargs))
    return PerWorkerValues(tuple(results))

  def fetch(self, val):
    # [FUNCTION DOC]
    # Parameter val: any nested structure. Replaces every remote value by its fetched result
    # (blocking until ready; may raise the value's stored error) and leaves other items as they
    # are. Returns the structure of fetched values.
    """Blocking call to fetch results from the remote values.

    This is a wrapper around
    `tf.distribute.experimental.coordinator.RemoteValue.fetch` for a
    `RemoteValue` structure; it returns the execution results of
    `RemoteValue`s. If not ready, wait for them while blocking the caller.

    Example:
    ```python
    strategy = ...
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy)

    def dataset_fn():
      return tf.data.Dataset.from_tensor_slices([1, 1, 1])

    with strategy.scope():
      v = tf.Variable(initial_value=0)

    @tf.function
    def worker_fn(iterator):
      def replica_fn(x):
        v.assign_add(x)
        return v.read_value()
      return strategy.run(replica_fn, args=(next(iterator),))

    distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
    distributed_iterator = iter(distributed_dataset)
    result = coordinator.schedule(worker_fn, args=(distributed_iterator,))
    assert coordinator.fetch(result) == 1
    ```

    Args:
      val: The value to fetch the results from. If this is structure of
        `tf.distribute.experimental.coordinator.RemoteValue`, `fetch()` will be
        called on the individual
        `tf.distribute.experimental.coordinator.RemoteValue` to get the result.

    Returns:
      If `val` is a `tf.distribute.experimental.coordinator.RemoteValue` or a
      structure of `tf.distribute.experimental.coordinator.RemoteValue`s,
      return the fetched `tf.distribute.experimental.coordinator.RemoteValue`
      values immediately if they are available, or block the call until they are
      available, and return the fetched
      `tf.distribute.experimental.coordinator.RemoteValue` values with the same
      structure. If `val` is other types, return it as-is.
    """

    def _maybe_fetch(val):
      # [FUNCTION DOC]
      # HELPER: leaf visitor for fetch. Returns the fetched value of a remote value, or the
      # item unchanged.
      if isinstance(val, RemoteValue):
        return val.fetch()
      else:
        return val

    # TODO(yuefengz): we should fetch values in a batch.
    return nest.map_structure(_maybe_fetch, val)


def _extract_failed_ps_instances(err_msg):
  # [FUNCTION DOC]
  # HELPER: used by Cluster for transient PS failure tracking. Parameter err_msg: error text.
  # Finds every parameter server task name in it and returns the set of their integer task
  # indices.
  """Return a set of potentially failing ps instances from error message."""
  tasks = re.findall("/job:ps/replica:0/task:[0-9]+", err_msg)
  return set(int(t.split(":")[-1]) for t in tasks)


def _is_ps_failure(error):
  # [FUNCTION DOC]
  # HELPER: error classification. Parameter error: an exception. Returns true for a PS
  # unavailable error, or (after unwrapping closure input or aborted errors) for an error whose
  # text carries the PS RPC marker and which is an unavailable or aborted error, or an invalid
  # argument error about an unknown device or missing remote tensor handle. False otherwise.
  """Whether the error is considered a parameter server failure."""
  if isinstance(error, PSUnavailableError):
    return True

  # For an `ClosureInputError` or `ClosureAbortedError`, extract
  # the original error and assess it accordingly.
  if isinstance(error, (ClosureInputError, ClosureAbortedError)):
    error = error.original_exception

  if _RPC_ERROR_FROM_PS not in str(error):
    return False

  if isinstance(error, (errors.UnavailableError, errors.AbortedError)):
    return True

  # The following error could happen when the remote task fails and restarts
  # in a very short interval during which no RPCs were exchanged to detect the
  # failure. In that case, gRPC allows channel (which is different from a
  # connection) to be reused for a replaced server listening to same address.
  if isinstance(error, errors.InvalidArgumentError):
    if ("unknown device" in str(error).lower() or
        "Unable to find the relevant tensor remote_handle" in str(error)):
      return True

  return False


def _handle_graph_execution_error_as_worker_failure():
  # [FUNCTION DOC]
  # HELPER: reads an environment variable (integer, default zero) and returns true if it is
  # positive, meaning unknown graph execution errors should be treated as worker failures.
  return int(os.environ.get("TF_PS_HANDLE_UNKNOWN_ERROR", "0")) > 0


def _is_worker_failure(error):
  # [FUNCTION DOC]
  # HELPER: error classification used by the worker preemption handler. Parameter error: an
  # exception. Returns true if the error should be treated as a worker failure: an unknown
  # graph execution error when enabled by environment variable (logged); otherwise, after
  # unwrapping closure input or aborted errors, an error whose text mentions a worker job and
  # not the PS RPC marker and which is unavailable, aborted, an invalid argument about an
  # unknown device, non-remote primary device or missing remote handle, a not-found error about
  # an unregistered function, a cancelled error, or a type error about binding inputs to a
  # compiled function. False otherwise (note a cancelled error counts only if the worker job
  # text is present).
  """Whether the error is considered a worker failure."""

  # TODO(b/216666282): Understand why worker failure can manifest as a
  # "Graph execution error" `UnknownError`.
  if (_handle_graph_execution_error_as_worker_failure() and
      isinstance(error, errors.UnknownError) and
      "Graph execution error" in str(error)):
    logging.info(f"Handling {type(error)}: {str(error)} as worker failure.")
    return True

  # For an `ClosureInputError` or `ClosureAbortedError`, extract
  # the original error and assess it accordingly.
  if isinstance(error, (ClosureInputError, ClosureAbortedError)):
    error = error.original_exception

  if _JOB_WORKER_STRING_IDENTIFIER not in str(error):
    return False
  if _RPC_ERROR_FROM_PS in str(error):
    return False

  # TODO(haoyuzhang): Consider using special status code if error from a
  # remote is derived from RPC errors originated from other hosts.
  if isinstance(error, (errors.UnavailableError, errors.AbortedError)):
    return True

  # The following error could happen when the remote task fails and restarts
  # in a very short interval during which no RPCs were exchanged to detect the
  # failure. In that case, gRPC allows channel (which is different from a
  # connection) to be reused for a replaced server listening to same address.
  if isinstance(error, errors.InvalidArgumentError):
    if ("unknown device" in str(error).lower() or
        "Primary device is not remote" in str(error) or
        "Unable to find the relevant tensor remote_handle" in str(error)):
      return True

  # TODO(b/162541228): The following 2 types of errors are very rare and only
  # observed in large-scale testing. The types of errors should be reduced.
  # This could happen when the function registration fails. In the observed
  # cases this only happens to the dataset related functions.
  if isinstance(error, errors.NotFoundError):
    if ("is neither a type of a primitive operation nor a name of a function "
        "registered" in str(error)):
      return True

  # NOTE(b/179061495): During worker preemptions, if multiple functions are
  # running concurrently (especially with subfunctions spanning chief/PS),
  # CancelledError can be returned due to chief/PS cancelling outstanding RPCs
  # to the failing workers.
  if isinstance(error, errors.CancelledError):
    return True

  # This can occur when preparing closures for execution when doing exact
  # evaluation, because the iterator creation, which occurs within the
  # tf.function, needs to access the worker device, so it fails if the worker is
  # down.
  if isinstance(error, TypeError) and "Binding inputs to tf.function" in str(
      error):
    return True

  return False
