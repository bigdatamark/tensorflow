# [MODULE DOC]
# Implementation of the TF2 summary writing API: thread-local default writer/step/recording
# state, summary writer classes (resource-backed, trackable, legacy and no-op), file writer
# factories, generic write functions, legacy contrib-style scalar/histogram/image/audio ops,
# graph and run-metadata summaries, and trace on/off/export. Import-time side effects: creates the
# thread-local state object and trace lock and registers exported public symbols.
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

"""Operations to emit summaries."""

import abc
import collections
import functools
import os
import re
import threading

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.dtensor.python import api as dtensor_api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as _profiler
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


# Name for graph collection of summary writer init ops, which is only exposed
# as a legacy API for tf.contrib.summary in TF 1.x.
_SUMMARY_WRITER_INIT_COLLECTION_NAME = "_SUMMARY_WRITER_V2"


class _SummaryState(threading.local):
  # [CLASS DOC]
  # HELPER: per-thread holder of the global summary settings.
  # Thread-local state with fields: is_recording (the recording condition: boolean, tensor,
  # callable or nothing), is_recording_distribution_strategy (condition set by distribution
  # strategies, default true), writer (the current default summary writer or nothing) and step
  # (the default summary step or nothing). A single module-level instance is used.

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the thread-local state: recording condition unset, distribution-strategy
    # condition true, no default writer and no default step. Called once per thread on first use.
    super(_SummaryState, self).__init__()
    self.is_recording = None
    # TODO(slebedev): why a separate flag for DS and is it on by default?
    self.is_recording_distribution_strategy = True
    self.writer = None
    self.step = None


_summary_state = _SummaryState()


class _SummaryContextManager:
  # [CLASS DOC]
  # HELPER: context manager returned by the writer's default-enabling method.
  # Installs a writer (and optionally a step) as the thread's defaults on entry and restores the
  # previous writer and step on exit. Implemented as a class rather than a generator-based
  # manager so that the set-as-default method can just enter it without a deferred exit.
  """Context manager to implement SummaryWriter.as_default()."""
  # Note: this is a class so that it's possible to implement `set_as_default()`
  # simply via `as_default().__enter__()`. We can't do that with @contextmanager
  # because the `finally` block will be executed when the generator is GCed.

  def __init__(self, writer, step=None):
    # [FUNCTION DOC]
    # Stores the writer and optional step to install; remembers no previous values yet.
    self._writer = writer
    self._step = step
    self._old_writer = None
    self._old_step = None

  def __enter__(self):
    # [FUNCTION DOC]
    # Saves the current thread's default writer and installs this writer; if a step was given,
    # also saves and replaces the default step. Returns the writer. Mutates thread-local state.
    self._old_writer = _summary_state.writer
    _summary_state.writer = self._writer
    if self._step is not None:
      self._old_step = _summary_state.step
      _summary_state.step = self._step
    return self._writer

  def __exit__(self, *exc):
    # [FUNCTION DOC]
    # KNOWN ISSUE: flushes the writer currently installed in the thread state (not necessarily the
    # one this manager installed) and fails if that is nothing.
    # Flushes the current default writer, then restores the previous default writer, and the
    # previous step if a step had been set. Exception details are ignored; returns false so
    # exceptions propagate.
    # Flushes the summary writer in eager mode or in graph functions, but
    # not in legacy graph mode (you're on your own there).
    _summary_state.writer.flush()
    _summary_state.writer = self._old_writer
    if self._step is not None:
      _summary_state.step = self._old_step
    return False


def _should_record_summaries_internal(default_state):
  # [FUNCTION DOC]
  # HELPER: shared implementation of the "should record" checks.
  # KNOWN ISSUE: the docstring describes the argument loosely; the code returns a constant false
  # tensor when there is no writer or a static recording condition is false, otherwise the
  # logical and of the distribution-strategy condition and the recording condition.
  # Input: default_state is the value used when no recording condition was set (true for the
  # public API, false for the legacy contrib one). Callable conditions are invoked here.
  # Returns a boolean tensor.
  """Returns boolean Tensor if summaries should/shouldn't be recorded.

  Now the summary condition is decided by logical "and" of below conditions:
  First, summary writer must be set. Given this constraint is met,
  ctx.summary_recording and ctx.summary_recording_distribution_strategy.
  The former one is usually set by user, and the latter one is controlled
  by DistributionStrategy (tf.distribute.ReplicaContext).

  Args:
    default_state: can be True or False. The default summary behavior when
    summary writer is set and the user does not specify
    ctx.summary_recording and ctx.summary_recording_distribution_strategy
    is True.
  """
  if _summary_state.writer is None:
    return constant_op.constant(False)

  if not callable(_summary_state.is_recording):
    static_cond = tensor_util.constant_value(_summary_state.is_recording)
    if static_cond is not None and not static_cond:
      return constant_op.constant(False)

  resolve = lambda x: x() if callable(x) else x
  cond_distributed = resolve(_summary_state.is_recording_distribution_strategy)
  cond = resolve(_summary_state.is_recording)
  if cond is None:
    cond = default_state
  return math_ops.logical_and(cond_distributed, cond)


@tf_export("summary.should_record_summaries", v1=[])
def should_record_summaries():
  # [FUNCTION DOC]
  # Public query: returns a boolean tensor, false when there is no default writer or recording is
  # off, and true by default when a writer exists and no condition was set. Takes no inputs.
  """Returns boolean Tensor which is True if summaries will be recorded.

  If no default summary writer is currently registered, this always returns
  False. Otherwise, this reflects the recording condition has been set via
  `tf.summary.record_if()` (except that it may return False for some replicas
  when using `tf.distribute.Strategy`). If no recording condition is active,
  it defaults to True.
  """
  return _should_record_summaries_internal(default_state=True)


# Legacy symbol used by tf.contrib.summary.should_record_summaries.
def _legacy_contrib_should_record_summaries():
  # [FUNCTION DOC]
  # HELPER: legacy variant for the old contrib summary API. Same as the public check but, when no
  # recording condition was set, defaults to not recording. Returns a boolean tensor.
  """Returns boolean Tensor which is true if summaries should be recorded."""
  return _should_record_summaries_internal(default_state=False)


def is_recording_summaries():
  # [FUNCTION DOC]
  # KNOWN ISSUE: despite the docstring promising a plain boolean, it returns the raw recording
  # condition object (which may be a tensor or callable) when a writer and a condition exist.
  # Returns false when there is no default writer or no condition was set; otherwise returns the
  # stored condition itself. No side effects.
  """Returns non-Tensor boolean indicating if summaries are being recorded."""
  if _summary_state.writer is None:
    return False
  if _summary_state.is_recording is None:
    return False
  return _summary_state.is_recording


@tf_export("summary.record_if", v1=[])
@tf_contextlib.contextmanager
def record_if(condition):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a context manager; it is itself a generator-based
  # context manager that yields nothing.
  # Context manager that sets the thread's summary-recording condition on entry and restores the
  # previous condition on exit (even on error). condition may be a boolean, a scalar boolean
  # tensor, or a callable evaluated on demand. Mutates thread-local state.
  """Sets summary recording on or off per the provided boolean value.

  The provided value can be a python boolean, a scalar boolean Tensor, or
  or a callable providing such a value; if a callable is passed it will be
  invoked on-demand to determine whether summary writing will occur.  Note that
  when calling record_if() in an eager mode context, if you intend to provide a
  varying condition like `step % 100 == 0`, you must wrap this in a
  callable to avoid immediate eager evaluation of the condition.  In particular,
  using a callable is the only way to have your condition evaluated as part of
  the traced body of an @tf.function that is invoked from within the
  `record_if()` context.

  Args:
    condition: can be True, False, a bool Tensor, or a callable providing such.

  Yields:
    Returns a context manager that sets this value on enter and restores the
    previous value on exit.
  """
  old = _summary_state.is_recording
  try:
    _summary_state.is_recording = condition
    yield
  finally:
    _summary_state.is_recording = old


def has_default_writer():
  # [FUNCTION DOC]
  # Returns true if the current thread has a default summary writer installed, false otherwise.
  """Returns a boolean indicating whether a default summary writer exists."""
  return _summary_state.writer is not None


# TODO(apassos) consider how to handle local step here.
def record_summaries_every_n_global_steps(n, global_step=None):
  # [FUNCTION DOC]
  # Returns a recording-condition context manager that records only when the global step is a
  # multiple of n. global_step defaults to the (created if missing) global step. On the CPU
  # device, builds an equality check; in eager mode it stays a callable evaluated lazily, in
  # graph mode it is evaluated into a tensor immediately. May create the global step variable.
  """Sets the should_record_summaries Tensor to true if global_step % n == 0."""
  if global_step is None:
    global_step = training_util.get_or_create_global_step()
  with ops.device("cpu:0"):
    should = lambda: math_ops.equal(global_step % n, 0)
    if not context.executing_eagerly():
      should = should()
  return record_if(should)


def always_record_summaries():
  # [FUNCTION DOC]
  # Returns a context manager that forces summary recording on for its duration.
  """Sets the should_record_summaries Tensor to always true."""
  return record_if(True)


def never_record_summaries():
  # [FUNCTION DOC]
  # Returns a context manager that forces summary recording off for its duration.
  """Sets the should_record_summaries Tensor to always false."""
  return record_if(False)


@tf_export("summary.experimental.get_step", v1=[])
def get_step():
  # [FUNCTION DOC]
  # Returns the default summary step stored for the current thread, or nothing if none was set.
  """Returns the default summary step for the current thread.

  Returns:
    The step set by `tf.summary.experimental.set_step()` if one has been set,
    otherwise None.
  """
  return _summary_state.step


@tf_export("summary.experimental.set_step", v1=[])
def set_step(step):
  # [FUNCTION DOC]
  # Sets the current thread's default summary step (an int64-castable value, variable, or nothing
  # to unset). Returns nothing. Inside traced functions a non-variable step is captured at trace
  # time.
  """Sets the default summary step for the current thread.

  For convenience, this function sets a default value for the `step` parameter
  used in summary-writing functions elsewhere in the API so that it need not
  be explicitly passed in every such invocation. The value can be a constant
  or a variable, and can be retrieved via `tf.summary.experimental.get_step()`.

  Note: when using this with @tf.functions, the step value will be captured at
  the time the function is traced, so changes to the step outside the function
  will not be reflected inside the function unless using a `tf.Variable` step.

  Args:
    step: An `int64`-castable default step value, or None to unset.
  """
  _summary_state.step = step


@tf_export("summary.SummaryWriter", v1=[])
class SummaryWriter(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Public interface for stateful summary writers (exported as the summary writer type).
  # Declares init, flush and close (unimplemented here) and provides the default-writer helpers
  # set_as_default and as_default, which install the writer into thread-local state. Concrete
  # subclasses below are backed by resources or do nothing.
  """Interface representing a stateful summary writer object."""

  def set_as_default(self, step=None):
    # [FUNCTION DOC]
    # Makes this writer the current thread's default writer without automatic restoration, by
    # entering the context manager from the default-enabling method and never exiting it.
    # step is an optional int64-castable default step; when given it replaces the current step.
    """Enables this summary writer for the current thread.

    For convenience, if `step` is not None, this function also sets a default
    value for the `step` parameter used in summary-writing functions elsewhere
    in the API so that it need not be explicitly passed in every such
    invocation. The value can be a constant or a variable.

    Note: when setting `step` in a @tf.function, the step value will be
    captured at the time the function is traced, so changes to the step outside
    the function will not be reflected inside the function unless using
    a `tf.Variable` step.

    Args:
      step: An `int64`-castable default step value, or `None`. When not `None`,
        the current step is modified to the given value. When `None`, the
        current step is not modified.
    """
    self.as_default(step).__enter__()

  def as_default(self, step=None):
    # [FUNCTION DOC]
    # Returns a context manager that installs this writer as the thread default (and sets the
    # default step when step is given) and restores the previous writer and step on exit; the
    # writer is flushed on exit. step is an optional int64-castable value. Nothing happens until
    # the manager is entered.
    """Returns a context manager that enables summary writing.

    For convenience, if `step` is not None, this function also sets a default
    value for the `step` parameter used in summary-writing functions elsewhere
    in the API so that it need not be explicitly passed in every such
    invocation. The value can be a constant or a variable.

    Note: when setting `step` in a @tf.function, the step value will be
    captured at the time the function is traced, so changes to the step outside
    the function will not be reflected inside the function unless using
    a `tf.Variable` step.

    For example, `step` can be used as:

    ```python
    with writer_a.as_default(step=10):
      tf.summary.scalar(tag, value)   # Logged to writer_a with step 10
      with writer_b.as_default(step=20):
        tf.summary.scalar(tag, value) # Logged to writer_b with step 20
      tf.summary.scalar(tag, value)   # Logged to writer_a with step 10
    ```

    Args:
      step: An `int64`-castable default step value, or `None`. When not `None`,
        the current step is captured, replaced by a given one, and the original
        one is restored when the context manager exits. When `None`, the current
        step is not modified (and not restored when the context manager exits).

    Returns:
      The context manager.
    """
    return _SummaryContextManager(self, step)

  def init(self):
    # [FUNCTION DOC]
    # Abstract hook for initializing the writer; the base implementation always raises a
    # not-implemented error.
    """Initializes the summary writer."""
    raise NotImplementedError()

  def flush(self):
    # [FUNCTION DOC]
    # Abstract hook for flushing buffered data; the base implementation always raises a
    # not-implemented error.
    """Flushes any buffered data."""
    raise NotImplementedError()

  def close(self):
    # [FUNCTION DOC]
    # Abstract hook for flushing and closing the writer; the base implementation always raises a
    # not-implemented error.
    """Flushes and closes the summary writer."""
    raise NotImplementedError()


class _ResourceSummaryWriter(SummaryWriter):
  # [CLASS DOC]
  # HELPER: concrete writer backed by a summary-writer resource handle.
  # Holds the resource, its initialization op, a closed flag (eager mode) and an optional
  # DTensor mesh. In graph mode the init op is added to a graph collection for later retrieval;
  # in eager mode a deleter releases the resource. Subclassed by the trackable variant.
  """Implementation of SummaryWriter using a SummaryWriterInterface resource."""

  def __init__(self, create_fn, init_op_fn, mesh=None):
    # [FUNCTION DOC]
    # Creates the resource by calling create_fn (a no-argument callable) and the init op by
    # calling init_op_fn with the resource; when a mesh is given, both happen under that mesh's
    # host mesh as default. In eager mode sets up a resource deleter; in graph mode adds the init
    # op to the writer-init collection. Remembers the mesh and marks the writer as open.
    if mesh is not None:
      with dtensor_api.default_mesh(mesh.host_mesh()):
        self._resource = create_fn()
        self._init_op = init_op_fn(self._resource)
    else:
      self._resource = create_fn()
      self._init_op = init_op_fn(self._resource)

    self._closed = False
    if context.executing_eagerly():
      self._set_up_resource_deleter()
    else:
      ops.add_to_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME, self._init_op)

    self._mesh = mesh

  # Extension point to be overridden by subclasses to customize deletion.

  def _set_up_resource_deleter(self):
    # [FUNCTION DOC]
    # HELPER: extension point; creates an eager resource deleter on the CPU for the resource
    # handle so the resource is freed when the writer is garbage collected.
    self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
        handle=self._resource, handle_device="cpu:0")

  def set_as_default(self, step=None):
    # [FUNCTION DOC]
    # Same as the base method but, in eager mode, raises a runtime error if the writer was closed.
    """See `SummaryWriter.set_as_default`."""
    if context.executing_eagerly() and self._closed:
      raise RuntimeError(f"SummaryWriter {self!r} is already closed")
    super().set_as_default(step)

  def as_default(self, step=None):
    # [FUNCTION DOC]
    # Same as the base method but, in eager mode, raises a runtime error if the writer was closed.
    # Returns the context manager from the base class.
    """See `SummaryWriter.as_default`."""
    if context.executing_eagerly() and self._closed:
      raise RuntimeError(f"SummaryWriter {self!r} is already closed")
    return super().as_default(step)

  def init(self):
    # [FUNCTION DOC]
    # Returns the stored initialization op (already created in the constructor). In eager mode
    # raises a runtime error if the writer is closed.
    """See `SummaryWriter.init`."""
    if context.executing_eagerly() and self._closed:
      raise RuntimeError(f"SummaryWriter {self!r} is already closed")
    return self._init_op

  def flush(self):
    # [FUNCTION DOC]
    # Flushes buffered events to disk by running the flush op on the CPU and returns its result
    # (an op in graph mode). Does nothing when the writer is closed in eager mode.
    """See `SummaryWriter.flush`."""
    if context.executing_eagerly() and self._closed:
      return
    with ops.device("cpu:0"):
      return gen_summary_ops.flush_summary_writer(self._resource)

  def close(self):
    # [FUNCTION DOC]
    # Flushes, then closes the writer resource on the CPU, returning the close op/result. In eager
    # mode, does nothing if already closed and marks the writer closed afterwards (even if the
    # close fails).
    """See `SummaryWriter.close`."""
    if context.executing_eagerly() and self._closed:
      return
    try:
      with ops.control_dependencies([self.flush()]):
        with ops.device("cpu:0"):
          return gen_summary_ops.close_summary_writer(self._resource)
    finally:
      if context.executing_eagerly():
        self._closed = True


class _MultiMetaclass(
    type(_ResourceSummaryWriter), type(resource.TrackableResource)):
  # [CLASS DOC]
  # HELPER: metaclass combining the metaclasses of the abstract writer base and the trackable
  # resource base so the trackable writer can inherit from both. Adds no behavior.
  pass


class _TrackableResourceSummaryWriter(
    _ResourceSummaryWriter,
    resource.TrackableResource,
    metaclass=_MultiMetaclass):
  # [CLASS DOC]
  # Resource-backed writer that is also a trackable resource, so it can be saved in a SavedModel
  # when held as a module attribute. The resource handle is created lazily through the trackable
  # machinery, and its destruction is handled by the trackable base rather than an eager deleter.
  """A `_ResourceSummaryWriter` subclass that implements `TrackableResource`."""

  def __init__(self, create_fn, init_op_fn, mesh=None):
    # [FUNCTION DOC]
    # Initializes the trackable resource base on the CPU, stores the creation and init callables,
    # then initializes the writer base using the cached resource handle as the creation function
    # so only one handle exists. mesh is the optional DTensor mesh.
    # Resolve multiple inheritance via explicit calls to __init__() on parents.
    resource.TrackableResource.__init__(self, device="/CPU:0")
    self._create_fn = create_fn
    self._init_op_fn = init_op_fn
    # Pass .resource_handle into _ResourceSummaryWriter parent class rather than
    # create_fn, to ensure it accesses the resource handle only through the
    # cached property so that everything is using a single resource handle.
    _ResourceSummaryWriter.__init__(
        self,
        create_fn=lambda: self.resource_handle,
        init_op_fn=init_op_fn,
        mesh=mesh,
    )

  # Override for TrackableResource implementation.
  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: trackable-resource hook; creates the resource handle by calling the stored creation
    # callable. Returns the handle.
    return self._create_fn()

  # Override for TrackableResource implementation.
  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: trackable-resource hook; runs the stored init-op function on the resource handle and
    # returns its result.
    return self._init_op_fn(self.resource_handle)

  # Override for TrackableResource implementation.
  def _destroy_resource(self):
    # [FUNCTION DOC]
    # HELPER: trackable-resource hook; destroys the resource, ignoring the error if it was already
    # gone. Returns nothing.
    gen_resource_variable_ops.destroy_resource_op(
        self.resource_handle, ignore_lookup_error=True)

  def _set_up_resource_deleter(self):
    # [FUNCTION DOC]
    # HELPER: intentionally does nothing; the trackable resource base cleans up on deletion.
    # Override to suppress ResourceSummaryWriter implementation; we don't need
    # the deleter since TrackableResource.__del__() handles it for us.
    pass


class _LegacyResourceSummaryWriter(SummaryWriter):
  # [CLASS DOC]
  # HELPER: legacy writer wrapping an existing resource handle, used by the old contrib-style
  # file-writer factory. Has no closed-state tracking. Holds the resource and an init function;
  # registers the init op in a graph collection (graph mode) or an eager deleter (eager mode).
  """Legacy resource-backed SummaryWriter for tf.contrib.summary."""

  def  __init__(self, resource, init_op_fn):
    # [FUNCTION DOC]
    # Stores the resource handle and init function, builds the init op right away and, in eager
    # mode, creates a deleter that frees the resource on the CPU; in graph mode adds the init op
    # to the writer-init collection.
    self._resource = resource
    self._init_op_fn = init_op_fn
    init_op = self.init()
    if context.executing_eagerly():
      self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._resource, handle_device="cpu:0")
    else:
      ops.add_to_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME, init_op)

  def init(self):
    # [FUNCTION DOC]
    # Builds and returns a new initialization op by applying the init function to the resource.
    """See `SummaryWriter.init`."""
    return self._init_op_fn(self._resource)

  def flush(self):
    # [FUNCTION DOC]
    # Runs the flush op for the resource on the CPU and returns its result.
    """See `SummaryWriter.flush`."""
    with ops.device("cpu:0"):
      return gen_summary_ops.flush_summary_writer(self._resource)

  def close(self):
    # [FUNCTION DOC]
    # Flushes, then (after the flush) closes the resource on the CPU and returns the close result.
    """See `SummaryWriter.close`."""
    with ops.control_dependencies([self.flush()]):
      with ops.device("cpu:0"):
        return gen_summary_ops.close_summary_writer(self._resource)


class _NoopSummaryWriter(SummaryWriter):
  # [CLASS DOC]
  # Writer that does nothing; every method is a no-op. Returned by the no-op writer factory and
  # by the legacy file-writer factory when no log directory is given, usable as a placeholder
  # context manager.
  """A summary writer that does nothing, for create_noop_writer()."""

  def set_as_default(self, step=None):
    # [FUNCTION DOC]
    # Does nothing (does not install a default writer or step). Returns nothing.
    pass

  @tf_contextlib.contextmanager
  def as_default(self, step=None):
    # [FUNCTION DOC]
    # Context manager that does nothing: yields once without changing the default writer or step.
    yield

  def init(self):
    # [FUNCTION DOC]
    # Does nothing and returns nothing.
    pass

  def flush(self):
    # [FUNCTION DOC]
    # Does nothing and returns nothing.
    pass

  def close(self):
    # [FUNCTION DOC]
    # Does nothing and returns nothing.
    pass


@tf_export(v1=["summary.initialize"])
def initialize(
    graph=None,  # pylint: disable=redefined-outer-name
    session=None):
  # [FUNCTION DOC]
  # Graph-mode initialization of summary writing; returns immediately when executing eagerly.
  # Inputs: graph is an optional graph or graph definition to be written to the event file (at
  # step zero); session is the session used to run ops, defaulting to the default session.
  # Behavior: runs the initializer ops of all writers in the session and, if graph is given,
  # serializes it and runs the graph-writing op with it fed through a string placeholder.
  # Raises a runtime error if there is no default writer, and a value error if no session was
  # given and no default session exists.
  """Initializes summary writing for graph execution mode.

  This operation is a no-op when executing eagerly.

  This helper method provides a higher-level alternative to using
  `tf.contrib.summary.summary_writer_initializer_op` and
  `tf.contrib.summary.graph`.

  Most users will also want to call `tf.compat.v1.train.create_global_step`
  which can happen before or after this function is called.

  Args:
    graph: A `tf.Graph` or `tf.compat.v1.GraphDef` to output to the writer.
      This function will not write the default graph by default. When
      writing to an event log file, the associated step will be zero.
    session: So this method can call `tf.Session.run`. This defaults
      to `tf.compat.v1.get_default_session`.

  Raises:
    RuntimeError: If  the current thread has no default
      `tf.contrib.summary.SummaryWriter`.
    ValueError: If session wasn't passed and no default session.
  """
  if context.executing_eagerly():
    return
  if _summary_state.writer is None:
    raise RuntimeError("No default tf.contrib.summary.SummaryWriter found")
  if session is None:
    session = ops.get_default_session()
    if session is None:
      raise ValueError("Argument `session must be passed if no default "
                       "session exists")
  session.run(summary_writer_initializer_op())
  if graph is not None:
    data = _serialize_graph(graph)
    x = array_ops.placeholder(dtypes.string)
    session.run(graph_v1(x, 0), feed_dict={x: data})


@tf_export("summary.create_file_writer", v1=[])
def create_file_writer_v2(
    logdir,
    max_queue=None,
    flush_millis=None,
    filename_suffix=None,
    name=None,
    experimental_trackable=False,
    experimental_mesh=None,
):
  # [FUNCTION DOC]
  # Creates a resource-backed summary file writer (TF2 API).
  # Inputs: logdir (required directory string); max_queue (queue size, default 10); flush_millis
  # (flush interval, default two minutes); filename_suffix (default ".v2"); name (op name scope);
  # experimental_trackable (return a trackable writer usable in SavedModel); experimental_mesh
  # (DTensor mesh whose host mesh is used for creation).
  # Output: a summary writer object. Raises a value error when logdir is nothing. In eager mode
  # validates arguments against an earlier writer with the same settings created inside a
  # function. Creation runs on the CPU in an init scope; in eager mode each writer gets a unique
  # shared name, in graph mode one derived from the scope.
  """Creates a summary file writer for the given log directory.

  Args:
    logdir: a string specifying the directory in which to write an event file.
    max_queue: the largest number of summaries to keep in a queue; will flush
      once the queue gets bigger than this. Defaults to 10.
    flush_millis: the largest interval between flushes. Defaults to 120,000.
    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.
    name: a name for the op that creates the writer.
    experimental_trackable: a boolean that controls whether the returned writer
      will be a `TrackableResource`, which makes it compatible with SavedModel
      when used as a `tf.Module` property.
    experimental_mesh: a `tf.experimental.dtensor.Mesh` instance. When running
      with DTensor, the mesh (experimental_mesh.host_mesh()) will be used for
      bringing all the DTensor logging from accelerator to CPU mesh.

  Returns:
    A SummaryWriter object.
  """
  # TODO(b/291655717): Revisit the experimental_mesh once we have soft placment.
  if logdir is None:
    raise ValueError("Argument `logdir` cannot be None")
  inside_function = ops.inside_function()
  with ops.name_scope(name, "create_file_writer") as scope, ops.device("cpu:0"):
    # Run init inside an init_scope() to hoist it out of tf.functions.
    with ops.init_scope():
      if context.executing_eagerly():
        _check_create_file_writer_args(
            inside_function,
            logdir=logdir,
            max_queue=max_queue,
            flush_millis=flush_millis,
            filename_suffix=filename_suffix)
      logdir = ops.convert_to_tensor(logdir, dtype=dtypes.string)
      if max_queue is None:
        max_queue = constant_op.constant(10)
      if flush_millis is None:
        flush_millis = constant_op.constant(2 * 60 * 1000)
      if filename_suffix is None:
        filename_suffix = constant_op.constant(".v2")

      def create_fn():
        # [FUNCTION DOC]
        # HELPER: creates the summary writer resource handle for the enclosing factory. Uses a
        # unique anonymous shared name in eager mode (no sharing), otherwise a name derived from
        # the scope so TF1 SavedModel loading works. Returns the resource handle.
        # Use unique shared_name to prevent resource sharing in eager mode, but
        # otherwise use a fixed shared_name to allow SavedModel TF 1.x loading.
        if context.executing_eagerly():
          shared_name = context.anonymous_name()
        else:
          shared_name = ops.name_from_scope_name(scope)  # pylint: disable=protected-access
        return gen_summary_ops.summary_writer(
            shared_name=shared_name, name=name)

      init_op_fn = functools.partial(
          gen_summary_ops.create_summary_file_writer,
          logdir=logdir,
          max_queue=max_queue,
          flush_millis=flush_millis,
          filename_suffix=filename_suffix)
      if experimental_trackable:
        return _TrackableResourceSummaryWriter(
            create_fn=create_fn, init_op_fn=init_op_fn, mesh=experimental_mesh
        )
      else:
        return _ResourceSummaryWriter(
            create_fn=create_fn, init_op_fn=init_op_fn, mesh=experimental_mesh
        )


def create_file_writer(logdir,
                       max_queue=None,
                       flush_millis=None,
                       filename_suffix=None,
                       name=None):
  # [FUNCTION DOC]
  # Legacy (contrib-style) file writer factory. If logdir is nothing, returns a no-op writer.
  # Otherwise converts logdir to a string and, on the CPU, fills in defaults (queue 10, flush
  # every two minutes, suffix ".v2", shared name "logdir:" plus the directory), creates the
  # writer resource with that shared name, and returns a legacy writer wrapping it with a
  # file-writer init function. An existing resource with the same shared name is reused.
  """Creates a summary file writer in the current context under the given name.

  Args:
    logdir: a string, or None. If a string, creates a summary file writer
     which writes to the directory named by the string. If None, returns
     a mock object which acts like a summary writer but does nothing,
     useful to use as a context manager.
    max_queue: the largest number of summaries to keep in a queue; will
     flush once the queue gets bigger than this. Defaults to 10.
    flush_millis: the largest interval between flushes. Defaults to 120,000.
    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.
    name: Shared name for this SummaryWriter resource stored to default
      Graph. Defaults to the provided logdir prefixed with `logdir:`. Note: if a
      summary writer resource with this shared name already exists, the returned
      SummaryWriter wraps that resource and the other arguments have no effect.

  Returns:
    Either a summary writer or an empty object which can be used as a
    summary writer.
  """
  if logdir is None:
    return _NoopSummaryWriter()
  logdir = str(logdir)
  with ops.device("cpu:0"):
    if max_queue is None:
      max_queue = constant_op.constant(10)
    if flush_millis is None:
      flush_millis = constant_op.constant(2 * 60 * 1000)
    if filename_suffix is None:
      filename_suffix = constant_op.constant(".v2")
    if name is None:
      name = "logdir:" + logdir
    resource = gen_summary_ops.summary_writer(shared_name=name)
    return _LegacyResourceSummaryWriter(
        resource=resource,
        init_op_fn=functools.partial(
            gen_summary_ops.create_summary_file_writer,
            logdir=logdir,
            max_queue=max_queue,
            flush_millis=flush_millis,
            filename_suffix=filename_suffix))


@tf_export("summary.create_noop_writer", v1=[])
def create_noop_writer():
  # [FUNCTION DOC]
  # Returns a new writer object whose operations all do nothing; useful as a placeholder context
  # manager.
  """Returns a summary writer that does nothing.

  This is useful as a placeholder in code that expects a context manager.
  """
  return _NoopSummaryWriter()


def _cleanse_string(name, pattern, value):
  # [FUNCTION DOC]
  # HELPER: validates string arguments for writer creation checks. If value is a Python string
  # that does not contain a match for the compiled pattern, raises a value error naming the
  # argument. Returns value converted to a string tensor (non-strings are converted without the
  # pattern check).
  if isinstance(value, str) and pattern.search(value) is None:
    raise ValueError(f"{name} ({value}) must match {pattern.pattern}")
  return ops.convert_to_tensor(value, dtypes.string)


def _nothing():
  # [FUNCTION DOC]
  # HELPER: else-branch for conditional summary writing when recording is off; returns a constant
  # false tensor.
  """Convenient else branch for when summaries do not record."""
  return constant_op.constant(False)


@tf_export(v1=["summary.all_v2_summary_ops"])
def all_v2_summary_ops():
  # [FUNCTION DOC]
  # Returns the list of V2-style summary ops stored in the default graph's summary collection, or
  # nothing when executing eagerly. Excludes TF1 summary ops.
  """Returns all V2-style summary ops defined in the current default graph.

  This includes ops from TF 2.0 tf.summary and TF 1.x tf.contrib.summary (except
  for `tf.contrib.summary.graph` and `tf.contrib.summary.import_event`), but
  does *not* include TF 1.x tf.summary ops.

  Returns:
    List of summary ops, or None if called under eager execution.
  """
  if context.executing_eagerly():
    return None
  return ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access


def summary_writer_initializer_op():
  # [FUNCTION DOC]
  # Graph mode only: returns the list of initializer ops of all summary writers created so far
  # (from the graph collection). Raises a runtime error when executing eagerly.
  """Graph-mode only. Returns the list of ops to create all summary writers.

  Returns:
    The initializer ops.

  Raises:
    RuntimeError: If in Eager mode.
  """
  if context.executing_eagerly():
    raise RuntimeError(
        "tf.contrib.summary.summary_writer_initializer_op is only "
        "supported in graph mode.")
  return ops.get_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME)


_INVALID_SCOPE_CHARACTERS = re.compile(r"[^-_/.A-Za-z0-9]")


@tf_export("summary.experimental.summary_scope", v1=[])
@tf_contextlib.contextmanager
def summary_scope(name, default_name="summary", values=None):
  # [FUNCTION DOC]
  # Context manager for defining custom summary ops. name is the summary name (falls back to
  # default_name when empty); values is optional tensors for the name scope. The tag is the name
  # prefixed by the current name scope (not uniquified, characters unrestricted); the scope name
  # has illegal characters removed. Enters a name scope (also in eager mode) and yields a pair
  # of tag and scope.
  """Experimental context manager for use when defining a custom summary op.

  This behaves similarly to `tf.name_scope`, except that it returns a generated
  summary tag in addition to the scope name. The tag is structurally similar to
  the scope name - derived from the user-provided name, prefixed with enclosing
  name scopes if any - but we relax the constraint that it be uniquified, as
  well as the character set limitation (so the user-provided name can contain
  characters not legal for scope names; in the scope name these are removed).

  This makes the summary tag more predictable and consistent for the user.

  For example, to define a new summary op called `my_op`:

  ```python
  def my_op(name, my_value, step):
    with tf.summary.summary_scope(name, "MyOp", [my_value]) as (tag, scope):
      my_value = tf.convert_to_tensor(my_value)
      return tf.summary.write(tag, my_value, step=step)
  ```

  Args:
    name: string name for the summary.
    default_name: Optional; if provided, used as default name of the summary.
    values: Optional; passed as `values` parameter to name_scope.

  Yields:
    A tuple `(tag, scope)` as described above.
  """
  name = name or default_name
  current_scope = ops.get_name_scope()
  tag = current_scope + "/" + name if current_scope else name
  # Strip illegal characters from the scope name, and if that leaves nothing,
  # use None instead so we pick up the default name.
  name = _INVALID_SCOPE_CHARACTERS.sub("", name) or None
  with ops.name_scope(name, default_name, values, skip_on_eager=False) as scope:
    yield tag, scope


@tf_export("summary.write", v1=[])
def write(tag, tensor, step=None, metadata=None, name=None):
  # [FUNCTION DOC]
  # Writes a generic summary to the default writer, if any. tag identifies the summary; tensor is
  # the data or a callable producing it (called only when writing); step is an int64-castable
  # step (defaults to the default step); metadata is an optional summary metadata proto or
  # serialized bytes; name is an optional op name.
  # Returns a true value on success, or a constant false if there is no default writer. Raises a
  # value error if a writer exists but no step is available. Writing is conditional on the
  # recording condition and runs the tensor computation on the CPU.
  """Writes a generic summary to the default SummaryWriter if one exists.

  This exists primarily to support the definition of type-specific summary ops
  like scalar() and image(), and is not intended for direct use unless defining
  a new type-specific summary op.

  Args:
    tag: string tag used to identify the summary (e.g. in TensorBoard), usually
      generated with `tf.summary.summary_scope`
    tensor: the Tensor holding the summary data to write or a callable that
      returns this Tensor. If a callable is passed, it will only be called when
      a default SummaryWriter exists and the recording condition specified by
      `record_if()` is met.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    metadata: Optional SummaryMetadata, as a proto or serialized bytes
    name: Optional string name for this op.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  if step is not None:
    _choose_step(step)
  with ops.name_scope(name, "write_summary") as scope:
    if _summary_state.writer is None:
      return constant_op.constant(False)
    if step is None:
      step = get_step()
    if metadata is None:
      serialized_metadata = b""
    elif hasattr(metadata, "SerializeToString"):
      serialized_metadata = metadata.SerializeToString()
    else:
      serialized_metadata = metadata

    def record():
      # [FUNCTION DOC]
      # HELPER: the actual write performed when recording is enabled. Raises a value error if no
      # step is set; copies the tensor to the CPU (or calls it if callable), converts it for
      # DTensor writers if needed, and emits the write op. Returns true.
      """Record the actual summary and return True."""
      if step is None:
        raise ValueError("No step set. Please specify one either through the "
                         "`step` argument or through "
                         "tf.summary.experimental.set_step()")

      # Note the identity to move the tensor to the CPU.
      with ops.device("cpu:0"):
        summary_tensor = tensor() if callable(tensor) else array_ops.identity(
            tensor)
        # For DTensor, the device scope above doesn't work, we need to
        # explicitly copy the resource tensor to host mesh, which is a cpu
        # mesh.
        writer = _summary_state.writer
        summary_value = _maybe_convert_tensor_to_dtensor(writer, summary_tensor)
        step_value = _maybe_convert_tensor_to_dtensor(writer, step)

        write_summary_op = gen_summary_ops.write_summary(
            writer._resource,  # pylint: disable=protected-access
            step_value,
            summary_value,
            tag,
            serialized_metadata,
            name=scope,
        )
        with ops.control_dependencies([write_summary_op]):
          return constant_op.constant(True)

    op = smart_cond.smart_cond(
        should_record_summaries(), record, _nothing, name="summary_cond")
    if not context.executing_eagerly():
      ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)  # pylint: disable=protected-access
    return op


@tf_export("summary.experimental.write_raw_pb", v1=[])
def write_raw_pb(tensor, step=None, name=None):
  # [FUNCTION DOC]
  # Experimental: writes one or more serialized V1-style summary protocol buffers (given as a
  # string tensor) to the default writer. step is an int64-castable step (defaults to the default
  # step); name is an optional op name.
  # Returns a true value on success or a constant false if there is no default writer. Raises a
  # value error if a writer exists but no step is available. The write is conditional on the
  # recording condition; the op is added to the graph summary collection in graph mode.
  """Writes a summary using raw `tf.compat.v1.Summary` protocol buffers.

  Experimental: this exists to support the usage of V1-style manual summary
  writing (via the construction of a `tf.compat.v1.Summary` protocol buffer)
  with the V2 summary writing API.

  Args:
    tensor: the string Tensor holding one or more serialized `Summary` protobufs
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    name: Optional string name for this op.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  if step is not None:
    _choose_step(step)
  with ops.name_scope(name, "write_raw_pb") as scope:
    if _summary_state.writer is None:
      return constant_op.constant(False)
    if step is None:
      step = get_step()
      if step is None:
        raise ValueError("No step set. Please specify one either through the "
                         "`step` argument or through "
                         "tf.summary.experimental.set_step()")

    def record():
      # [FUNCTION DOC]
      # HELPER: performs the raw protocol buffer write on the CPU once recording is enabled, then
      # returns a true constant that depends on the write op.
      """Record the actual summary and return True."""
      # Note the identity to move the tensor to the CPU.
      with ops.device("cpu:0"):
        raw_summary_op = gen_summary_ops.write_raw_proto_summary(
            _summary_state.writer._resource,  # pylint: disable=protected-access
            step,
            array_ops.identity(tensor),
            name=scope)
        with ops.control_dependencies([raw_summary_op]):
          return constant_op.constant(True)

    with ops.device("cpu:0"):
      op = smart_cond.smart_cond(
          should_record_summaries(), record, _nothing, name="summary_cond")
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)  # pylint: disable=protected-access
      return op


def summary_writer_function(name, tensor, function, family=None):
  # [FUNCTION DOC]
  # HELPER: shared glue for the legacy contrib-style summary functions.
  # Inputs: name of the summary; tensor, the main data (used for name scope values); function,
  # a callback taking tag and scope that returns the write op; family, optional tag prefix.
  # If there is no default writer returns a no-op. Otherwise, on the CPU, conditionally (by the
  # legacy recording condition, off by default) runs the callback inside the re-entered name
  # scope and a summary scope and returns a true constant depending on it. In graph mode the
  # result is added to the summary collection. Returns that op/value.
  """Helper function to write summaries.

  Args:
    name: name of the summary
    tensor: main tensor to form the summary
    function: function taking a tag and a scope which writes the summary
    family: optional, the summary's family

  Returns:
    The result of writing the summary.
  """
  name_scope = ops.get_name_scope()
  if name_scope:
    # Add a slash to allow reentering the name scope.
    name_scope += "/"
  def record():
    # [FUNCTION DOC]
    # HELPER: re-enters the saved name scope and a summary scope, runs the caller's write
    # callback with the resulting tag and scope, and returns a true constant after it completes.
    with ops.name_scope(name_scope), summary_op_util.summary_scope(
        name, family, values=[tensor]) as (tag, scope):
      with ops.control_dependencies([function(tag, scope)]):
        return constant_op.constant(True)

  if _summary_state.writer is None:
    return control_flow_ops.no_op()
  with ops.device("cpu:0"):
    op = smart_cond.smart_cond(
        _legacy_contrib_should_record_summaries(), record, _nothing, name="")
    if not context.executing_eagerly():
      ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)  # pylint: disable=protected-access
  return op


def generic(name, tensor, metadata=None, family=None, step=None):
  # [FUNCTION DOC]
  # Legacy API: writes a tensor summary if a default writer exists. name is the summary name;
  # tensor the data; metadata an optional summary metadata proto or serialized value; family an
  # optional tag prefix; step an optional step (defaults to the global step). Validates step
  # eagerly. Returns the write op (a no-op if no writer).
  """Writes a tensor summary if possible."""
  if step is not None:
    _choose_step(step)
  def function(tag, scope):
    # [FUNCTION DOC]
    # HELPER: callback building the write op for tag and scope: serializes metadata (empty if
    # none), copies the tensor to the CPU via identity and writes it with the chosen step.
    if metadata is None:
      serialized_metadata = constant_op.constant("")
    elif hasattr(metadata, "SerializeToString"):
      serialized_metadata = constant_op.constant(metadata.SerializeToString())
    else:
      serialized_metadata = metadata
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        array_ops.identity(tensor),
        tag,
        serialized_metadata,
        name=scope)
  return summary_writer_function(name, tensor, function, family=family)


def scalar(name, tensor, family=None, step=None):
  # [FUNCTION DOC]
  # Legacy API: writes a scalar summary if a default writer exists. name is the summary name;
  # tensor a numeric scalar tensor; family optional tag prefix; step optional step (defaults to
  # the global step, validated up front). Returns the write op, or a no-op if no writer or
  # recording is disabled.
  """Writes a scalar summary if possible.

  Unlike `tf.contrib.summary.generic` this op may change the dtype
  depending on the writer, for both practical and efficiency concerns.

  Args:
    name: An arbitrary name for this summary.
    tensor: A `tf.Tensor` Must be one of the following types:
      `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`,
      `int8`, `uint16`, `half`, `uint32`, `uint64`.
    family: Optional, the summary's family.
    step: The `int64` monotonic step variable, which defaults
      to `tf.compat.v1.train.get_global_step`.

  Returns:
    The created `tf.Operation` or a `tf.no_op` if summary writing has
    not been enabled for this context.
  """
  if step is not None:
    _choose_step(step)
  def function(tag, scope):
    # [FUNCTION DOC]
    # HELPER: callback that emits the scalar write op for the given tag and scope, copying the
    # tensor to the CPU and using the chosen step. Returns the op.
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_scalar_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        tag,
        array_ops.identity(tensor),
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def histogram(name, tensor, family=None, step=None):
  # [FUNCTION DOC]
  # Legacy API: writes a histogram summary of tensor under name if a default writer exists.
  # family is an optional tag prefix; step an optional step (defaults to the global step).
  # Returns the write op or a no-op.
  """Writes a histogram summary if possible."""
  if step is not None:
    _choose_step(step)
  def function(tag, scope):
    # [FUNCTION DOC]
    # HELPER: callback that emits the histogram write op for the tag and scope. Returns the op.
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_histogram_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        tag,
        array_ops.identity(tensor),
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def image(name, tensor, bad_color=None, max_images=3, family=None, step=None):
  # [FUNCTION DOC]
  # Legacy API: writes an image summary of tensor (a batch of images) if a default writer
  # exists. bad_color is the color used for invalid pixels (default opaque red); max_images
  # (default 3) caps how many images of the batch are written; family and step as in the other
  # legacy writers. Returns the write op or a no-op.
  """Writes an image summary if possible."""
  if step is not None:
    _choose_step(step)
  def function(tag, scope):
    # [FUNCTION DOC]
    # HELPER: callback that emits the image write op for the tag and scope, substituting the
    # default red bad color when none was given. Returns the op.
    bad_color_ = (constant_op.constant([255, 0, 0, 255], dtype=dtypes.uint8)
                  if bad_color is None else bad_color)
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_image_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        tag,
        array_ops.identity(tensor),
        bad_color_,
        max_images,
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def audio(name, tensor, sample_rate, max_outputs, family=None, step=None):
  # [FUNCTION DOC]
  # Legacy API: writes an audio summary of tensor if a default writer exists. sample_rate is the
  # sampling rate in Hz; max_outputs caps how many clips of the batch are written; family and
  # step as in the other legacy writers. Returns the write op or a no-op.
  """Writes an audio summary if possible."""
  if step is not None:
    _choose_step(step)
  def function(tag, scope):
    # [FUNCTION DOC]
    # HELPER: callback that emits the audio write op for the tag and scope. Returns the op.
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_audio_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        tag,
        array_ops.identity(tensor),
        sample_rate=sample_rate,
        max_outputs=max_outputs,
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def graph_v1(param, step=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the error message for a bad param refers to graph() rather than this function.
  # Legacy API: writes a serialized graph to the default writer, ignoring the recording
  # condition. param is a string tensor holding a graph proto (in eager mode a graph or graph
  # definition is also accepted and serialized); step is optional (defaults to the global step);
  # name is an optional op name. Raises a type error in graph mode if param is not a tensor.
  # Returns the write op, or a no-op if there is no default writer.
  """Writes a TensorFlow graph to the summary interface.

  The graph summary is, strictly speaking, not a summary. Conditions
  like `tf.summary.should_record_summaries` do not apply. Only
  a single graph can be associated with a particular run. If multiple
  graphs are written, then only the last one will be considered by
  TensorBoard.

  When not using eager execution mode, the user should consider passing
  the `graph` parameter to `tf.compat.v1.summary.initialize` instead of
  calling this function. Otherwise special care needs to be taken when
  using the graph to record the graph.

  Args:
    param: A `tf.Tensor` containing a serialized graph proto. When
      eager execution is enabled, this function will automatically
      coerce `tf.Graph`, `tf.compat.v1.GraphDef`, and string types.
    step: The global step variable. This doesn't have useful semantics
      for graph summaries, but is used anyway, due to the structure of
      event log files. This defaults to the global step.
    name: A name for the operation (optional).

  Returns:
    The created `tf.Operation` or a `tf.no_op` if summary writing has
    not been enabled for this context.

  Raises:
    TypeError: If `param` isn't already a `tf.Tensor` in graph mode.
  """
  if step is not None:
    _choose_step(step)

  if not context.executing_eagerly() and not isinstance(
      param, tensor_lib.Tensor
  ):
    raise TypeError(
        "graph() needs a argument `param` to be tf.Tensor "
        "(e.g. tf.placeholder) in graph mode, but received "
        f"param={param} of type {type(param).__name__}."
    )
  writer = _summary_state.writer
  if writer is None:
    return control_flow_ops.no_op()
  with ops.device("cpu:0"):
    if isinstance(param, (ops.Graph, graph_pb2.GraphDef)):
      tensor = ops.convert_to_tensor(_serialize_graph(param), dtypes.string)
    else:
      tensor = array_ops.identity(param)
    return gen_summary_ops.write_graph_summary(
        writer._resource, _choose_step(step), tensor, name=name)  # pylint: disable=protected-access


@tf_export("summary.graph", v1=[])
def graph(graph_data):
  # [FUNCTION DOC]
  # Writes a graph or graph definition as a summary (eager mode only), at step zero.
  # Input: graph_data is a graph object or graph definition. Raises a value error when called in
  # graph mode or when graph_data is of another type (the type check happens only after the
  # writer and recording checks pass). Returns a constant false if there is no default writer or
  # recording is off, otherwise writes and returns a constant true.
  """Writes a TensorFlow graph summary.

  Write an instance of `tf.Graph` or `tf.compat.v1.GraphDef` as summary only
  in an eager mode. Please prefer to use the trace APIs (`tf.summary.trace_on`,
  `tf.summary.trace_off`, and `tf.summary.trace_export`) when using
  `tf.function` which can automatically collect and record graphs from
  executions.

  Usage Example:
  ```py
  writer = tf.summary.create_file_writer("/tmp/mylogs")

  @tf.function
  def f():
    x = constant_op.constant(2)
    y = constant_op.constant(3)
    return x**y

  with writer.as_default():
    tf.summary.graph(f.get_concrete_function().graph)

  # Another example: in a very rare use case, when you are dealing with a TF v1
  # graph.
  graph = tf.Graph()
  with graph.as_default():
    c = tf.constant(30.0)
  with writer.as_default():
    tf.summary.graph(graph)
  ```

  Args:
    graph_data: The TensorFlow graph to write, as a `tf.Graph` or a
      `tf.compat.v1.GraphDef`.

  Returns:
    True on success, or False if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: `graph` summary API is invoked in a graph mode.
  """
  if not context.executing_eagerly():
    raise ValueError("graph() cannot be invoked inside a graph context.")
  writer = _summary_state.writer
  if writer is None:
    return constant_op.constant(False)
  with ops.device("cpu:0"):
    if not should_record_summaries():
      return constant_op.constant(False)

    if isinstance(graph_data, (ops.Graph, graph_pb2.GraphDef)):
      tensor = ops.convert_to_tensor(
          _serialize_graph(graph_data), dtypes.string)
    else:
      raise ValueError("Argument 'graph_data' is not tf.Graph or "
                       "tf.compat.v1.GraphDef. Received graph_data="
                       f"{graph_data} of type {type(graph_data).__name__}.")

    gen_summary_ops.write_graph_summary(
        writer._resource,  # pylint: disable=protected-access
        # Graph does not have step. Set to 0.
        0,
        tensor,
    )
    return constant_op.constant(True)


def import_event(tensor, name=None):
  # [FUNCTION DOC]
  # Writes a serialized event protocol buffer (string tensor) directly into the default writer,
  # ignoring the recording condition. name is an optional op name. Returns the created op.
  # Fails with an attribute error if there is no default writer.
  """Writes a `tf.compat.v1.Event` binary proto.

  This can be used to import existing event logs into a new summary writer sink.
  Please note that this is lower level than the other summary functions and
  will ignore the `tf.summary.should_record_summaries` setting.

  Args:
    tensor: A `tf.Tensor` of type `string` containing a serialized
      `tf.compat.v1.Event` proto.
    name: A name for the operation (optional).

  Returns:
    The created `tf.Operation`.
  """
  return gen_summary_ops.import_event(
      _summary_state.writer._resource, tensor, name=name)  # pylint: disable=protected-access


@tf_export("summary.flush", v1=[])
def flush(writer=None, name=None):
  # [FUNCTION DOC]
  # Forces a summary writer to send buffered data to storage. writer is the writer to flush
  # (default: the current default writer; if none, returns a no-op). name is ignored. Returns
  # the writer's flush result. Raises a value error if writer is not a summary writer object.
  """Forces summary writer to send any buffered data to storage.

  This operation blocks until that finishes.

  Args:
    writer: The `tf.summary.SummaryWriter` to flush. If None, the current
      default writer will be used instead; if there is no current writer, this
      returns `tf.no_op`.
    name: Ignored legacy argument for a name for the operation.

  Returns:
    The created `tf.Operation`.
  """
  del name  # unused
  if writer is None:
    writer = _summary_state.writer
    if writer is None:
      return control_flow_ops.no_op()
  if isinstance(writer, SummaryWriter):
    return writer.flush()
  raise ValueError("Invalid argument to flush(): %r" % (writer,))


def legacy_raw_flush(writer=None, name=None):
  # [FUNCTION DOC]
  # Legacy flush that also accepts a raw resource tensor. For nothing or a writer object it
  # forwards to the regular flush; otherwise it runs the flush op on the resource on the CPU with
  # the given name. Returns the flush op/result. Not part of the public API.
  """Legacy version of flush() that accepts a raw resource tensor for `writer`.

  Do not use this function in any new code. Not supported and not part of the
  public TF APIs.

  Args:
    writer: The `tf.summary.SummaryWriter` to flush. If None, the current
      default writer will be used instead; if there is no current writer, this
      returns `tf.no_op`. For this legacy version only, also accepts a raw
      resource tensor pointing to the underlying C++ writer resource.
    name: Ignored legacy argument for a name for the operation.

  Returns:
    The created `tf.Operation`.
  """
  if writer is None or isinstance(writer, SummaryWriter):
    # Forward to the TF2 implementation of flush() when possible.
    return flush(writer, name)
  else:
    # Legacy fallback in case we were passed a raw resource tensor.
    with ops.device("cpu:0"):
      return gen_summary_ops.flush_summary_writer(writer, name=name)


def eval_dir(model_dir, name=None):
  # [FUNCTION DOC]
  # HELPER: builds the log directory path for evaluation summaries: the model directory joined
  # with "eval", or "eval_" plus name when a non-empty name is given. Returns the path string.
  """Construct a logdir for an eval summary writer."""
  return os.path.join(model_dir, "eval" if not name else "eval_" + name)


@deprecation.deprecated(date=None,
                        instructions="Renamed to create_file_writer().")
def create_summary_file_writer(*args, **kwargs):
  # [FUNCTION DOC]
  # Deprecated alias kept for compatibility: logs a deprecation warning and forwards all
  # arguments to the legacy file-writer factory, returning its writer.
  """Please use `tf.contrib.summary.create_file_writer`."""
  logging.warning("Deprecation Warning: create_summary_file_writer was renamed "
                  "to create_file_writer")
  return create_file_writer(*args, **kwargs)


def _serialize_graph(arbitrary_graph):
  # [FUNCTION DOC]
  # HELPER: serializes a graph for graph summaries. A graph object is converted to a graph
  # definition with shapes added first; a graph definition is serialized directly. Returns bytes.
  if isinstance(arbitrary_graph, ops.Graph):
    return arbitrary_graph.as_graph_def(add_shapes=True).SerializeToString()
  else:
    return arbitrary_graph.SerializeToString()

def _choose_step(step):
  # [FUNCTION DOC]
  # HELPER: normalizes a step argument. Nothing yields the global step (creating it if absent);
  # otherwise the value is converted to an int64 tensor. Raises a value error if the result is
  # not a single-element tensor. Returns the step tensor.
  if step is None:
    return training_util.get_or_create_global_step()

  step = ops.convert_to_tensor(step, dtypes.int64)

  if step.shape.num_elements() != 1:
    raise ValueError(
        "Argument `step` must be a scalar. "
        f"Received step with shape {step.shape}."
    )
  return step

def _check_create_file_writer_args(inside_function, **kwargs):
  # [FUNCTION DOC]
  # HELPER: validation for the file-writer factory. inside_function says whether the call is in a
  # traced function; the keyword arguments are checked by name. Raises a value error (with a
  # message depending on inside_function) if any argument is a non-eager graph tensor. Returns
  # nothing.
  """Helper to check the validity of arguments to a create_file_writer() call.

  Args:
    inside_function: whether the create_file_writer() call is in a tf.function
    **kwargs: the arguments to check, as kwargs to give them names.

  Raises:
    ValueError: if the arguments are graph tensors.
  """
  for arg_name, arg in kwargs.items():
    if not isinstance(arg, ops.EagerTensor) and tensor_util.is_tf_type(arg):
      if inside_function:
        raise ValueError(
            f"Invalid graph Tensor argument '{arg_name}={arg}' to "
            "create_file_writer() inside an @tf.function. The create call will "
            "be lifted into the outer eager execution context, so it cannot "
            "consume graph tensors defined inside the function body.")
      else:
        raise ValueError(
            f"Invalid graph Tensor argument '{arg_name}={arg}' to eagerly "
            "executed create_file_writer().")


def run_metadata(name, data, step=None):
  # [FUNCTION DOC]
  # Writes an entire run-metadata proto as a summary. name is the summary name (tag gets the
  # active name scopes as prefix); data is the run-metadata proto; step is an optional step.
  # Tags the summary with the graph run-metadata plugin (version 1), serializes the proto into a
  # string tensor on the CPU and delegates to the generic write. Returns its result (true, or
  # false if no writer). Raises a value error if no step is available while a writer exists.
  """Writes entire RunMetadata summary.

  A RunMetadata can contain DeviceStats, partition graphs, and function graphs.
  Please refer to the proto for definition of each field.

  Args:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A RunMetadata proto to write.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = summary_pb2.SummaryMetadata()
  # Hard coding a plugin name. Please refer to go/tb-plugin-name-hardcode for
  # the rationale.
  summary_metadata.plugin_data.plugin_name = "graph_run_metadata"
  # version number = 1
  summary_metadata.plugin_data.content = b"1"

  with summary_scope(name,
                     "graph_run_metadata_summary",
                     [data, step]) as (tag, _):
    with ops.device("cpu:0"):
      tensor = constant_op.constant(data.SerializeToString(),
                                    dtype=dtypes.string)
    return write(
        tag=tag,
        tensor=tensor,
        step=step,
        metadata=summary_metadata)


def run_metadata_graphs(name, data, step=None):
  # [FUNCTION DOC]
  # Writes only the graphs (function graphs and partition graphs) from a run-metadata proto as a
  # summary. Same inputs and outputs as the full run-metadata writer, but copies just those two
  # fields into a fresh proto and tags it with the graph-only plugin name (version 1).
  # Returns the write result; raises a value error if a writer exists but no step is available.
  """Writes graphs from a RunMetadata summary.

  Args:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A RunMetadata proto to write.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = summary_pb2.SummaryMetadata()
  # Hard coding a plugin name. Please refer to go/tb-plugin-name-hardcode for
  # the rationale.
  summary_metadata.plugin_data.plugin_name = "graph_run_metadata_graph"
  # version number = 1
  summary_metadata.plugin_data.content = b"1"

  data = config_pb2.RunMetadata(
      function_graphs=data.function_graphs,
      partition_graphs=data.partition_graphs)

  with summary_scope(name,
                     "graph_run_metadata_graph_summary",
                     [data, step]) as (tag, _):
    with ops.device("cpu:0"):
      tensor = constant_op.constant(data.SerializeToString(),
                                    dtype=dtypes.string)
    return write(
        tag=tag,
        tensor=tensor,
        step=step,
        metadata=summary_metadata)


_TraceContext = collections.namedtuple("TraceContext", ("graph", "profiler"))
_current_trace_context_lock = threading.Lock()
_current_trace_context = None


@tf_export("summary.trace_on", v1=[])
def trace_on(graph=True, profiler=False, profiler_outdir=None):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # Starts a global trace (eager mode only) collecting graphs and optionally profiler data.
  # Inputs: graph (default true) enables graph collection; profiler (default false) enables the
  # profiler; profiler_outdir is the profiler output directory, needed when profiler is on.
  # Behavior: logs a warning and returns if called inside a function, outside eager mode, or if a
  # trace is already active. With profiler but no output directory it only warns and the profiler
  # is not enabled (but the trace context is still recorded). Updates the module-level trace
  # context under a lock and changes runtime state. Returns nothing.
  """Starts a trace to record computation graphs and profiling information.

  Must be invoked in eager mode.

  When enabled, TensorFlow runtime will collect information that can later be
  exported and consumed by TensorBoard. The trace is activated across the entire
  TensorFlow runtime and affects all threads of execution.

  To stop the trace and export the collected information, use
  `tf.summary.trace_export`. To stop the trace without exporting, use
  `tf.summary.trace_off`.

  Args:
    graph: If True, enables collection of executed graphs. It includes ones from
      tf.function invocation and ones from the legacy graph mode. The default is
      True.
    profiler: If True, enables the advanced profiler. Enabling profiler
      implicitly enables the graph collection. The profiler may incur a high
      memory overhead. The default is False.
    profiler_outdir: Output directory for profiler. It is required when profiler
      is enabled when trace was started. Otherwise, it is ignored.
  """
  if ops.inside_function():
    logging.warn("Cannot enable trace inside a tf.function.")
    return
  if not context.executing_eagerly():
    logging.warn("Must enable trace in eager mode.")
    return

  global _current_trace_context
  with _current_trace_context_lock:
    if _current_trace_context:
      logging.warn("Trace already enabled")
      return

    if graph and not profiler:
      context.context().enable_graph_collection()
    if profiler:
      if profiler_outdir is None:
        # TODO(b/149431324): Change this to throw a ValueError when Tensorflow
        # major version advances. (current version is 2.15)
        logging.warn(
            "No `profiler_outdir` passed to trace_on(). Profiler won't be"
            " enabled."
        )
      else:
        context.context().enable_run_metadata()
        _profiler.start(profiler_outdir)

    _current_trace_context = _TraceContext(graph=graph, profiler=profiler)


# TODO(b/149431324): Delete `profiler_outdir` arg when Tensorflow major version
# advances. (current version is 2.15)
@tf_export("summary.trace_export", v1=[])
def trace_export(name, step=None, profiler_outdir=None):
  # [FUNCTION DOC]
  # Stops the active trace and exports its metadata as a summary. name is the summary name; step
  # an optional step (validated first); profiler_outdir is ignored with a warning if given.
  # Warns and returns when called inside a function or outside eager mode. Raises a value error
  # if no trace was enabled. Writes only graphs if graph collection was on without the profiler,
  # otherwise the full run-metadata; stops the profiler if it was on; finally turns the trace off.
  # Writes through the default writer, if any. Returns nothing.
  """Stops and exports the active trace as a Summary and/or profile file.

  Stops the trace and exports all metadata collected during the trace to the
  default SummaryWriter, if one has been set.

  Args:
    name: A name for the summary to be written.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    profiler_outdir: This arg is a no-op. Please set this in trace_on().

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  # TODO(stephanlee): See if we can remove profiler_outdir and infer it from
  # the SummaryWriter's logdir.
  global _current_trace_context
  if step is not None:
    _choose_step(step)

  if ops.inside_function():
    logging.warn("Cannot export trace inside a tf.function.")
    return
  if not context.executing_eagerly():
    logging.warn("Can only export trace while executing eagerly.")
    return

  with _current_trace_context_lock:
    if _current_trace_context is None:
      raise ValueError("Must enable trace before export through "
                       "tf.summary.trace_on.")
    graph, profiler = _current_trace_context  # pylint: disable=redefined-outer-name

  run_meta = context.context().export_run_metadata()

  if graph and not profiler:
    run_metadata_graphs(name, run_meta, step)
  else:
    run_metadata(name, run_meta, step)

  if profiler:
    if profiler_outdir:
      logging.warn(
          "Ignoring `profiler_outdir` passed to trace_export(). Please pass it"
          " to trace_on() instead."
      )
    _profiler.stop()

  trace_off()


@tf_export("summary.trace_off", v1=[])
def trace_off():
  # [FUNCTION DOC]
  # Stops the current trace and discards collected data. Does nothing if no trace is active.
  # Clears the module-level trace context under a lock, disables run-metadata (and thus graph)
  # collection if graph tracing was on, and stops the profiler if it was on, logging a warning
  # instead of raising if stopping fails. Returns nothing.
  """Stops the current trace and discards any collected information."""
  global _current_trace_context
  with _current_trace_context_lock:
    if _current_trace_context is None:
      return  # tracing already off
    graph, profiler = _current_trace_context  # pylint: disable=redefined-outer-name, unpacking-non-sequence
    _current_trace_context = None

  if graph:
    # Disabling run_metadata disables graph collection as well.
    context.context().disable_run_metadata()

  if profiler:
    try:
      _profiler.stop()
    except Exception as e:  # pylint: disable=broad-except
      logging.warn("Error while stopping profiler: %s", e)
      pass


def _maybe_convert_tensor_to_dtensor(writer, tensor):
  # [FUNCTION DOC]
  # HELPER: for writers created with a DTensor mesh, copies the tensor to the mesh's host mesh
  # with a fully replicated layout so it can be written; otherwise returns the tensor unchanged.
  # Inputs: writer (any writer, may lack a mesh) and tensor. Returns the possibly converted
  # tensor.
  if getattr(writer, "_mesh", None) is not None:
    mesh = writer._mesh.host_mesh()  # pylint: disable=protected-access
    tensor = dtensor_api.copy_to_mesh(
        tensor, layout_lib.Layout.replicated(mesh, rank=tensor.shape.rank)
    )
  return tensor
