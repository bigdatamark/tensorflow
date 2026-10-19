# [MODULE DOC]
# Execution context support for the parameter server ClusterCoordinator. It keeps a per-thread
# "dispatch context" recording which worker is currently running a closure, and exposes helpers
# to read it: a context manager that installs the context, an accessor for the current context,
# a helper that unwraps remote values, and the public function that returns the current worker
# index from inside a function running on a worker.
# Import-time side effects: creates a module-level thread-local storage object and registers the
# worker index function with the public API export mechanism (not exported for version 1).
# Not a helper module; it is used by the coordinator and by user code that needs the worker index.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""The execution context for ClusterCoordinator."""

import contextlib
import threading

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

_dispatch_context = threading.local()


def get_current_dispatch_context():
  # [FUNCTION DOC]
  # HELPER: internal accessor for the thread-local dispatch context used by the coordinator.
  # Takes no inputs. Returns the dispatch context object installed for the calling thread by the
  # dispatch context manager, or the null value if no context was ever set on this thread.
  # No side effects.
  try:
    return _dispatch_context.current
  except AttributeError:
    return None


@contextlib.contextmanager
def with_dispatch_context(worker_obj):
  # [FUNCTION DOC]
  # HELPER: internal context manager used by the coordinator when running a closure on a worker.
  # KNOWN ISSUE: the previous context is restored only after the yield with no protective
  # cleanup block, so if the body raises, the new dispatch context stays installed on the thread.
  # Input: worker_obj, the worker object (must have a worker index attribute) on whose behalf
  # code runs. Saves the thread's current dispatch context, installs a new one wrapping the
  # worker, yields nothing, and afterwards restores the saved context (possibly the null value).
  # Side effect: mutates the thread-local module state.
  previous_context = getattr(_dispatch_context, "current", None)
  _dispatch_context.current = DispatchContext(worker_obj)
  yield
  _dispatch_context.current = previous_context


class DispatchContext(object):
  # [CLASS DOC]
  # Lightweight record of the worker on which a closure is currently being executed. Holds the
  # worker object and its index. Instances are created by the dispatch context manager and stored
  # in thread-local state; read-only properties expose the worker and its index.
  """Context entered when executing a closure on a given worker."""

  def __init__(self, worker_obj):
    # [FUNCTION DOC]
    # Constructor. Input: worker_obj, the worker object, which must expose a worker index
    # attribute. Stores the worker and copies its index. Returns nothing.
    self._worker = worker_obj
    self._worker_index = worker_obj.worker_index

  @property
  def worker(self):
    # [FUNCTION DOC]
    # Read-only property returning the worker object this context was created for.
    return self._worker

  @property
  def worker_index(self):
    # [FUNCTION DOC]
    # Read-only property returning the integer index of the worker, captured at construction.
    return self._worker_index

  def maybe_get_remote_value(self, ret):
    # [FUNCTION DOC]
    # Convenience method delegating to the module-level function of the same name. Input: ret,
    # any value, possibly a remote value. Returns the underlying values if it is a ready remote
    # value, otherwise the input unchanged. May raise an assertion error (see the module function).
    return maybe_get_remote_value(ret)


def maybe_get_remote_value(val):
  # [FUNCTION DOC]
  # Unwraps a remote value. Input: val, any object. If it is not a remote value it is returned
  # unchanged. If it is a remote value that holds an error, raises an assertion error mentioning
  # the error; if it has not reached the ready status, raises an assertion error saying it has not
  # been executed; otherwise returns the fetched local values of the remote value.
  """Gets the value of `val` if it is a `RemoteValue`."""
  if isinstance(val, remote_value.RemoteValue):
    error = val._get_error()  # pylint: disable=protected-access
    if error:
      raise AssertionError(
          "RemoteValue doesn't have a value because it has error %r:%s" %
          (error, error))
    elif val._status is not remote_value.RemoteValueStatus.READY:  # pylint: disable=protected-access
      raise AssertionError("The input RemoteValue has not been executed.")
    else:
      return val._get_values()  # pylint: disable=protected-access
  else:
    return val


@tf_export("distribute.coordinator.experimental_get_current_worker_index",
           v1=[])
def get_current_worker_index():
  # [FUNCTION DOC]
  # Public API (exported as the experimental current-worker-index function of the coordinator
  # package, version 2 only). Takes no inputs. Must be called while tracing a function; raises a
  # runtime error otherwise. Returns a scalar 64-bit integer tensor, captured as a call-time value
  # in the default graph, whose value is resolved each time the function is run by reading the
  # worker index from the thread's dispatch context; at run time it raises a runtime error if no
  # dispatch context is present (i.e. not running on a worker closure). The tensor's producing
  # operation is tagged with the user-specified name "worker_index".
  """Returns the current worker index, when called within a worker closure.

  Some parameter server training workloads may require the worker to know its
  index, for example for data sharding for reduced-variance training.

  This method may be used within a `tf.function` that is executed on a worker.
  That is, either a `dataset_fn` that runs via
  `ClusterCoordinator.create_per_worker_dataset`, or any other function
  scheduled via `ClusterCoordinator.schedule`.

  Example (sharding data by worker):

  ```python
  strategy = tf.distribute.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.coordinator.ClusterCoordinator(strategy))

  def dataset_fn(context):
    dataset = tf.data.Dataset.range(10)
    worker_index = (
        tf.distribute.coordinator.experimental_get_current_worker_index()
    )
    dataset = dataset.shard(
        num_shards=num_workers,
        index=worker_index,
    )
    return dataset

  @tf.function
  def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)

  per_worker_dataset = coordinator.create_per_worker_dataset(
      per_worker_dataset_fn)
  ```

  Raises:
    RuntimeError: if called from outside a `tf.function` or outside of a remote
      closure execution context (that is, on a non-worker machine).
  """

  msg = ("Cannot retrieve the worker index. `get_worker_idx_and_num_workers` "
         "should be called from within a tf.function being executed on a "
         "worker. This method should only be called from either a dataset_fn "
         "that is passed into `ClusterCoordinator.create_per_worker_dataset`, "
         "or a tf.function that is passed into `ClusterCoordinator.schedule`.")
  if not ops.inside_function():
    raise RuntimeError(msg)

  def call_time_worker_index():
    # [FUNCTION DOC]
    # HELPER: call-time value producer for the worker index tensor. Takes no inputs. Looks up the
    # current thread's dispatch context; raises a runtime error with the outer message if none;
    # otherwise returns the worker index of that context.
    dispatch_context = get_current_dispatch_context()
    if not dispatch_context:
      raise RuntimeError(msg)
    return dispatch_context.worker_index

  worker_index = ops.get_default_graph().capture_call_time_value(
      call_time_worker_index, tensor.TensorSpec([], dtype=dtypes.int64))
  worker_index.op._set_attr(  # pylint: disable=protected-access
      "_user_specified_name",
      attr_value_pb2.AttrValue(s=compat.as_bytes("worker_index")))
  return worker_index
