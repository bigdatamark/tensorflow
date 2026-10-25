# [MODULE DOC]
# Deprecated v1 queue-runner machinery: a class that runs a queue's enqueue operations in
# background threads (optionally coordinated by a coordinator that propagates stop requests and
# errors), plus functions to add queue runners to a graph collection and to start all collected
# runners. Not usable under eager execution. Import-time side effects: registers the queue runner
# collection key with the graph protocol-buffer conversion machinery (to and from the queue runner
# definition message) and registers public exports. Re-exported by the queue_runner module.
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

"""Create threads to run multiple enqueue ops."""
import threading
import weakref

from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

_DEPRECATION_INSTRUCTION = (
    "To construct input pipelines, use the `tf.data` module.")


@tf_export(v1=["train.queue_runner.QueueRunner", "train.QueueRunner"])
class QueueRunner:
  # [CLASS DOC]
  # Holds a queue, a list of enqueue operations, a close operation and a cancel operation, and
  # creates one thread per enqueue operation (plus an optional close-on-stop thread when a
  # coordinator is used). State: the queue, enqueue ops, close and cancel ops, the tuple of
  # exception types meaning "queue closed", a lock, a weak per-session count of running threads,
  # and a list of captured exceptions. Built from explicit arguments or from a serialized
  # definition message; can be serialized back. Deprecated; unsupported in eager mode.
  """Holds a list of enqueue operations for a queue, each to be run in a thread.

  Queues are a convenient TensorFlow mechanism to compute tensors
  asynchronously using multiple threads. For example in the canonical 'Input
  Reader' setup one set of threads generates filenames in a queue; a second set
  of threads read records from the files, processes them, and enqueues tensors
  on a second queue; a third set of threads dequeues these input records to
  construct batches and runs them through training operations.

  There are several delicate issues when running multiple threads that way:
  closing the queues in sequence as the input is exhausted, correctly catching
  and reporting exceptions, etc.

  The `QueueRunner`, combined with the `Coordinator`, helps handle these issues.

  @compatibility(TF2)
  QueueRunners are not compatible with eager execution. Instead, please
  use [tf.data](https://www.tensorflow.org/guide/data) to get data into your
  model.
  @end_compatibility
  """

  @deprecation.deprecated(None, _DEPRECATION_INSTRUCTION)
  def __init__(self, queue=None, enqueue_ops=None, close_op=None,
               cancel_op=None, queue_closed_exception_types=None,
               queue_runner_def=None, import_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says ValueError is raised when queue_runner_def and queue are both
    # given, but the code also raises it when enqueue_ops is given together with queue_runner_def.
    # Constructor (deprecated). Inputs: queue, enqueue_ops, close_op, cancel_op,
    # queue_closed_exception_types (all optional, used when building from arguments) or
    # queue_runner_def (a serialized definition) with optional import_scope name scope. Raises a
    # runtime error under eager execution; a value error if a definition is combined with a queue or
    # enqueue ops; the argument path may raise value or type errors. Delegates to the proto or
    # argument initializer, then creates the lock, a weak mapping from session to running-thread
    # count, and an empty exceptions list. Adds close and cancel ops to the graph if not supplied.
    """Create a QueueRunner.

    On construction the `QueueRunner` adds an op to close the queue.  That op
    will be run if the enqueue ops raise exceptions.

    When you later call the `create_threads()` method, the `QueueRunner` will
    create one thread for each op in `enqueue_ops`.  Each thread will run its
    enqueue op in parallel with the other threads.  The enqueue ops do not have
    to all be the same op, but it is expected that they all enqueue tensors in
    `queue`.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Optional tuple of Exception types that
        indicate that the queue has been closed when raised during an enqueue
        operation.  Defaults to `(tf.errors.OutOfRangeError,)`.  Another common
        case includes `(tf.errors.OutOfRangeError, tf.errors.CancelledError)`,
        when some of the enqueue ops may dequeue from other Queues.
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer. If specified,
        recreates the QueueRunner from its contents. `queue_runner_def` and the
        other arguments are mutually exclusive.
      import_scope: Optional `string`. Name scope to add. Only used when
        initializing from protocol buffer.

    Raises:
      ValueError: If both `queue_runner_def` and `queue` are both specified.
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
      RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
      raise RuntimeError(
          "QueueRunners are not supported when eager execution is enabled. "
          "Instead, please use tf.data to get data into your model.")

    if queue_runner_def:
      if queue or enqueue_ops:
        raise ValueError("queue_runner_def and queue are mutually exclusive.")
      self._init_from_proto(queue_runner_def,
                            import_scope=import_scope)
    else:
      self._init_from_args(
          queue=queue, enqueue_ops=enqueue_ops,
          close_op=close_op, cancel_op=cancel_op,
          queue_closed_exception_types=queue_closed_exception_types)
    # Protect the count of runs to wait for.
    self._lock = threading.Lock()
    # A map from a session object to the number of outstanding queue runner
    # threads for that session.
    self._runs_per_session = weakref.WeakKeyDictionary()
    # List of exceptions raised by the running threads.
    self._exceptions_raised = []

  def _init_from_args(self, queue=None, enqueue_ops=None, close_op=None,
                      cancel_op=None, queue_closed_exception_types=None):
    # [FUNCTION DOC]
    # HELPER: argument-based initializer used by the constructor.
    # Stores queue, enqueue ops, close op and cancel op. Raises a value error if queue or enqueue ops
    # are missing or empty; raises a type error if closed-exception types is given but is not a
    # non-empty tuple of operation-error subclasses. Missing close op becomes a plain close of the
    # queue; missing cancel op becomes a close that cancels pending enqueues (both create graph ops).
    # Closed-exception types default to the out-of-range error, otherwise are stored as a tuple.
    """Create a QueueRunner from arguments.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Tuple of exception types, which indicate
        the queue has been safely closed.

    Raises:
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
      TypeError: If `queue_closed_exception_types` is provided, but is not
        a non-empty tuple of error types (subclasses of `tf.errors.OpError`).
    """
    if not queue or not enqueue_ops:
      raise ValueError("Must provide queue and enqueue_ops.")
    self._queue = queue
    self._enqueue_ops = enqueue_ops
    self._close_op = close_op
    self._cancel_op = cancel_op
    if queue_closed_exception_types is not None:
      if (not isinstance(queue_closed_exception_types, tuple)
          or not queue_closed_exception_types
          or not all(issubclass(t, errors.OpError)
                     for t in queue_closed_exception_types)):
        raise TypeError(
            "queue_closed_exception_types, when provided, "
            "must be a tuple of tf.error types, but saw: %s"
            % queue_closed_exception_types)
    self._queue_closed_exception_types = queue_closed_exception_types
    # Close when no more will be produced, but pending enqueues should be
    # preserved.
    if self._close_op is None:
      self._close_op = self._queue.close()
    # Close and cancel pending enqueues since there was an error and we want
    # to unblock everything so we can cleanly exit.
    if self._cancel_op is None:
      self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
    if not self._queue_closed_exception_types:
      self._queue_closed_exception_types = (errors.OutOfRangeError,)
    else:
      self._queue_closed_exception_types = tuple(
          self._queue_closed_exception_types)

  def _init_from_proto(self, queue_runner_def, import_scope=None):
    # [FUNCTION DOC]
    # HELPER: definition-message initializer used by the constructor.
    # Asserts the input is a queue runner definition message (assertion error otherwise). Looks up,
    # in the current default graph, the queue, enqueue ops, close op and cancel op by name, each
    # prefixed with the optional import scope, and converts stored error codes to exception types.
    # If the definition has no exception types (older definitions), defaults to out-of-range error.
    # Raises graph lookup errors if a named element is missing.
    """Create a QueueRunner from `QueueRunnerDef`.

    Args:
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
    assert isinstance(queue_runner_def, queue_runner_pb2.QueueRunnerDef)
    g = ops.get_default_graph()
    self._queue = g.as_graph_element(
        ops.prepend_name_scope(queue_runner_def.queue_name, import_scope))
    self._enqueue_ops = [g.as_graph_element(
        ops.prepend_name_scope(op, import_scope))
                         for op in queue_runner_def.enqueue_op_name]
    self._close_op = g.as_graph_element(ops.prepend_name_scope(
        queue_runner_def.close_op_name, import_scope))
    self._cancel_op = g.as_graph_element(ops.prepend_name_scope(
        queue_runner_def.cancel_op_name, import_scope))
    self._queue_closed_exception_types = tuple(
        errors.exception_type_from_error_code(code)
        for code in queue_runner_def.queue_closed_exception_types)
    # Legacy support for old QueueRunnerDefs created before this field
    # was added.
    if not self._queue_closed_exception_types:
      self._queue_closed_exception_types = (errors.OutOfRangeError,)

  @property
  def queue(self):
    # [FUNCTION DOC]
    # Read-only property returning the queue whose enqueue ops this runner runs.
    return self._queue

  @property
  def enqueue_ops(self):
    # [FUNCTION DOC]
    # Read-only property returning the list of enqueue operations, one thread each.
    return self._enqueue_ops

  @property
  def close_op(self):
    # [FUNCTION DOC]
    # Read-only property returning the operation that closes the queue while preserving pending
    # enqueues.
    return self._close_op

  @property
  def cancel_op(self):
    # [FUNCTION DOC]
    # Read-only property returning the operation that closes the queue and cancels pending enqueues.
    return self._cancel_op

  @property
  def queue_closed_exception_types(self):
    # [FUNCTION DOC]
    # Read-only property returning the tuple of exception types that signal the queue was closed.
    return self._queue_closed_exception_types

  @property
  def exceptions_raised(self):
    # [FUNCTION DOC]
    # Read-only property returning the list of exceptions captured from worker threads when no
    # coordinator was supplied. The list is reset each time threads are newly created.
    """Exceptions raised but not handled by the `QueueRunner` threads.

    Exceptions raised in queue runner threads are handled in one of two ways
    depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `QueueRunner`.
    * Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
      made available in this `exceptions_raised` property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    """
    return self._exceptions_raised

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the name string of the underlying queue.
    """The string name of the underlying Queue."""
    return self._queue.name

  # pylint: disable=broad-except
  def _run(self, sess, enqueue_op, coord=None):
    # [FUNCTION DOC]
    # HELPER: thread body started by the thread-creation method, one per enqueue op.
    # Inputs: sess, the session; enqueue_op, the operation to run repeatedly; coord, optional
    # coordinator. Builds a cached callable and runs it in a loop until the coordinator says stop or
    # a queue-closed exception occurs. On a closed exception it decrements this session's running
    # count under the lock, and the last thread to finish runs the close op (errors ignored, logged
    # at verbose level). Any other exception is reported to the coordinator, or, without one, logged,
    # appended to the captured list and re-raised. The count is always decremented on exit.
    """Execute the enqueue op in a loop, close the queue in case of error.

    Args:
      sess: A Session.
      enqueue_op: The Operation to run.
      coord: Optional Coordinator object for reporting errors and checking
        for stop conditions.
    """
    decremented = False
    try:
      # Make a cached callable from the `enqueue_op` to decrease the
      # Python overhead in the queue-runner loop.
      enqueue_callable = sess.make_callable(enqueue_op)
      while True:
        if coord and coord.should_stop():
          break
        try:
          enqueue_callable()
        except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
          # This exception indicates that a queue was closed.
          with self._lock:
            self._runs_per_session[sess] -= 1
            decremented = True
            if self._runs_per_session[sess] == 0:
              try:
                sess.run(self._close_op)
              except Exception as e:
                # Intentionally ignore errors from close_op.
                logging.vlog(1, "Ignored exception: %s", str(e))
            return
    except Exception as e:
      # This catches all other exceptions.
      if coord:
        coord.request_stop(e)
      else:
        logging.error("Exception in QueueRunner: %s", str(e))
        with self._lock:
          self._exceptions_raised.append(e)
        raise
    finally:
      # Make sure we account for all terminations: normal or errors.
      if not decremented:
        with self._lock:
          self._runs_per_session[sess] -= 1

  def _close_on_stop(self, sess, cancel_op, coord):
    # [FUNCTION DOC]
    # HELPER: thread body that cancels the queue when the coordinator requests stop.
    # Blocks until the coordinator signals stop, then runs the cancel op in the session; any error
    # from that run is ignored and only logged at verbose level. Returns nothing.
    """Close the queue when the Coordinator requests stop.

    Args:
      sess: A Session.
      cancel_op: The Operation to run.
      coord: Coordinator.
    """
    coord.wait_for_stop()
    try:
      sess.run(cancel_op)
    except Exception as e:
      # Intentionally ignore errors from cancel_op.
      logging.vlog(1, "Ignored exception: %s", str(e))
  # pylint: enable=broad-except

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    # [FUNCTION DOC]
    # Inputs: sess, session for running ops; coord, optional coordinator; daemon, whether threads
    # are daemon threads (default false); start, whether to start them (default false). Under the
    # lock, if threads for this session are still counted as running, returns an empty list;
    # otherwise records the number of enqueue ops as the session's running count and clears captured
    # exceptions. Returns a list of unstarted-or-started threads: one per enqueue op, plus a
    # close-on-stop thread when a coordinator is given. Each thread is registered with the
    # coordinator if present, marked daemon and started as requested. Names embed queue and op names.
    """Create threads to run the enqueue ops for the given session.

    This method requires a session in which the graph was launched.  It creates
    a list of threads, optionally starting them.  There is one thread for each
    op passed in `enqueue_ops`.

    The `coord` argument is an optional coordinator that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to close the queue when the
    coordinator requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: Optional `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: Boolean.  If `True` make the threads daemon threads.
      start: Boolean.  If `True` starts the threads.  If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    """
    with self._lock:
      try:
        if self._runs_per_session[sess] > 0:
          # Already started: no new threads to return.
          return []
      except KeyError:
        # We haven't seen this session yet.
        pass
      self._runs_per_session[sess] = len(self._enqueue_ops)
      self._exceptions_raised = []

    ret_threads = []
    for op in self._enqueue_ops:
      name = "QueueRunnerThread-{}-{}".format(self.name, op.name)
      ret_threads.append(threading.Thread(target=self._run,
                                          args=(sess, op, coord),
                                          name=name))
    if coord:
      name = "QueueRunnerThread-{}-close_on_stop".format(self.name)
      ret_threads.append(threading.Thread(target=self._close_on_stop,
                                          args=(sess, self._cancel_op, coord),
                                          name=name))
    for t in ret_threads:
      if coord:
        coord.register_thread(t)
      if daemon:
        t.daemon = True
      if start:
        t.start()
    return ret_threads

  def to_proto(self, export_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring mentions a Variable not being in the name scope; it is actually
    # the queue that must be in the scope.
    # Input: optional export_scope name scope to strip from names. Returns a queue runner definition
    # message holding the queue name, enqueue op names, close and cancel op names and error codes of
    # the closed-exception types, or nothing if an export scope is given and the queue's name does
    # not start with it. Has no side effects on the runner.
    """Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `QueueRunnerDef` protocol buffer, or `None` if the `Variable` is not in
      the specified name scope.
    """
    if (export_scope is None or
        self.queue.name.startswith(export_scope)):
      queue_runner_def = queue_runner_pb2.QueueRunnerDef()
      queue_runner_def.queue_name = ops.strip_name_scope(
          self.queue.name, export_scope)
      for enqueue_op in self.enqueue_ops:
        queue_runner_def.enqueue_op_name.append(
            ops.strip_name_scope(enqueue_op.name, export_scope))
      queue_runner_def.close_op_name = ops.strip_name_scope(
          self.close_op.name, export_scope)
      queue_runner_def.cancel_op_name = ops.strip_name_scope(
          self.cancel_op.name, export_scope)
      queue_runner_def.queue_closed_exception_types.extend([
          errors.error_code_from_exception_type(cls)
          for cls in self._queue_closed_exception_types])
      return queue_runner_def
    else:
      return None

  @staticmethod
  def from_proto(queue_runner_def, import_scope=None):
    # [FUNCTION DOC]
    # Static factory. Inputs: queue_runner_def, a definition message; import_scope, optional name
    # scope to prepend. Returns a new runner built from the definition in the current default graph
    # (so the constructor's deprecation warning and eager-mode error apply). Used for collection
    # deserialization.
    """Returns a `QueueRunner` object created from `queue_runner_def`."""
    return QueueRunner(queue_runner_def=queue_runner_def,
                       import_scope=import_scope)


@tf_export(v1=["train.queue_runner.add_queue_runner", "train.add_queue_runner"])
@deprecation.deprecated(None, _DEPRECATION_INSTRUCTION)
def add_queue_runner(qr, collection=ops.GraphKeys.QUEUE_RUNNERS):
  # [FUNCTION DOC]
  # Inputs: qr, a queue runner; collection, graph collection key (default the queue runners key).
  # Adds the runner to that collection of the default graph. Returns nothing. Deprecated.
  """Adds a `QueueRunner` to a collection in the graph.

  When building a complex model that uses many queues it is often difficult to
  gather all the queue runners that need to be run.  This convenience function
  allows you to add a queue runner to a well known collection in the graph.

  The companion method `start_queue_runners()` can be used to start threads for
  all the collected queue runners.

  @compatibility(TF2)
  QueueRunners are not compatible with eager execution. Instead, please
  use [tf.data](https://www.tensorflow.org/guide/data) to get data into your
  model.
  @end_compatibility

  Args:
    qr: A `QueueRunner`.
    collection: A `GraphKey` specifying the graph collection to add
      the queue runner to.  Defaults to `GraphKeys.QUEUE_RUNNERS`.
  """
  ops.add_to_collection(collection, qr)


@tf_export(v1=["train.queue_runner.start_queue_runners",
               "train.start_queue_runners"])
@deprecation.deprecated(None, _DEPRECATION_INSTRUCTION)
def start_queue_runners(sess=None, coord=None, daemon=True, start=True,
                        collection=ops.GraphKeys.QUEUE_RUNNERS):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring has two Raises sections and omits that a monitored-session object
  # yields an empty list; it lists only value error for a missing session, but a type error is
  # raised for non-session objects and a runtime error under eager execution.
  # Inputs: sess (default: the default session), coord (optional coordinator), daemon (default
  # true), start (default true), collection (default queue runners key). Raises a runtime error in
  # eager mode, a value error with no session available, a type error if sess is not a session
  # (except monitored session classes, which return an empty list). Logs a warning if the collection
  # is empty. Within the session's graph, creates (and optionally starts) threads for every runner
  # in the collection and returns the combined list of threads.
  """Starts all queue runners collected in the graph.

  This is a companion method to `add_queue_runner()`.  It just starts
  threads for all queue runners collected in the graph.  It returns
  the list of all threads.

  @compatibility(TF2)
  QueueRunners are not compatible with eager execution. Instead, please
  use [tf.data](https://www.tensorflow.org/guide/data) to get data into your
  model.
  @end_compatibility

  Args:
    sess: `Session` used to run the queue ops.  Defaults to the
      default session.
    coord: Optional `Coordinator` for coordinating the started threads.
    daemon: Whether the threads should be marked as `daemons`, meaning
      they don't block program exit.
    start: Set to `False` to only create the threads, not start them.
    collection: A `GraphKey` specifying the graph collection to
      get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.

  Raises:
    ValueError: if `sess` is None and there isn't any default session.
    TypeError: if `sess` is not a `tf.compat.v1.Session` object.

  Returns:
    A list of threads.

  Raises:
    RuntimeError: If called with eager execution enabled.
    ValueError: If called without a default `tf.compat.v1.Session` registered.
  """
  if context.executing_eagerly():
    raise RuntimeError("Queues are not compatible with eager execution.")
  if sess is None:
    sess = ops.get_default_session()
    if not sess:
      raise ValueError("Cannot start queue runners: No default session is "
                       "registered. Use `with sess.as_default()` or pass an "
                       "explicit session to tf.start_queue_runners(sess=sess)")

  if not isinstance(sess, session.SessionInterface):
    # Following check is due to backward compatibility. (b/62061352)
    if sess.__class__.__name__ in [
        "MonitoredSession", "SingularMonitoredSession"]:
      return []
    raise TypeError("sess must be a `tf.Session` object. "
                    "Given class: {}".format(sess.__class__))

  queue_runners = ops.get_collection(collection)
  if not queue_runners:
    logging.warning(
        "`tf.train.start_queue_runners()` was called when no queue runners "
        "were defined. You can safely remove the call to this deprecated "
        "function.")

  with sess.graph.as_default():
    threads = []
    for qr in ops.get_collection(collection):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=daemon,
                                       start=start))
  return threads


ops.register_proto_function(ops.GraphKeys.QUEUE_RUNNERS,
                            proto_type=queue_runner_pb2.QueueRunnerDef,
                            to_proto=QueueRunner.to_proto,
                            from_proto=QueueRunner.from_proto)
