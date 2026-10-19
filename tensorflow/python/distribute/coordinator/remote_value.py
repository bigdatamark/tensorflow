# [MODULE DOC]
# Defines the public, abstract RemoteValue interface returned by scheduling a function on the
# cluster coordinator, and an enumeration of the states such a value can be in (not ready,
# aborted, ready). The concrete implementations live in the sibling values module. The
# RemoteValue class is registered with the public API export mechanism under two names at import
# time.
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
"""RemoteValue interface class."""

import enum

from tensorflow.python.util.tf_export import tf_export


class RemoteValueStatus(enum.Enum):
  # [CLASS DOC]
  # Enumeration of the lifecycle states of a remote value: not ready (no value yet), aborted (the
  # producing closure was aborted by a task failure but can be retried) and ready (has a value or
  # a non-retryable error). Members are string valued; transitions are described in the
  # docstring below.
  """The status of a `RemoteValue` object.

  A `RemoteValue` object can have three states:
    1) not ready: no value, no non-retryable error and not aborted;
    2) aborted: i.e. the execution of function was aborted because of task
       failure, but can be retried;
    3) ready: i.e. has value or has non-tryable error;

  The initial state of a `RemoteValue` is "not ready". When its corresponding
  closure has
  been executed at least once, it will become aborted or ready. The state
  transitions are:
    1) not ready -> 2) aborted:
      when the corresponding closure is aborted due to worker failure, and the
      worker failure is not immediately handled.
    1) not ready -> 3) ready:
      when the corresponding closure has been executed successfully.
    2) aborted -> 3) ready:
      when the `RemoteValue` is rebuilt by rerunning the corresponding closure
      and the closure has been executed successfully.
    3) ready -> 2) aborted:
      when the corresponding closure had been executed successfully but later
      the corresponding remote worker failed. This is currently only implemented
      for resource `RemoteValue` like iterators.
  """
  NOT_READY = "NOT_READY"
  ABORTED = "ABORTED"
  READY = "READY"


@tf_export("distribute.experimental.coordinator.RemoteValue",
           "distribute.coordinator.RemoteValue", v1=[])
class RemoteValue(object):
  # [CLASS DOC]
  # Abstract public interface for a value that becomes available asynchronously after a scheduled
  # function has run on a remote worker. Returned by the cluster coordinator's schedule call. The
  # base class holds no state; subclasses (see the values module) implement fetching the value as
  # numpy data or as local tensors. Exported in the public distribute API.
  """An asynchronously available value of a scheduled function.

  This class is used as the return value of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` where
  the underlying value becomes available at a later time once the function has
  been executed.

  Using `tf.distribute.experimental.coordinator.RemoteValue` as an input to
  a subsequent function scheduled with
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` is
  currently not supported.

  Example:

  ```python
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))

  with strategy.scope():
    v1 = tf.Variable(initial_value=0.0)
    v2 = tf.Variable(initial_value=1.0)

  @tf.function
  def worker_fn():
    v1.assign_add(0.1)
    v2.assign_sub(0.2)
    return v1.read_value() / v2.read_value()

  result = coordinator.schedule(worker_fn)
  # Note that `fetch()` gives the actual result instead of a `tf.Tensor`.
  assert result.fetch() == 0.125

  for _ in range(10):
    # `worker_fn` will be run on arbitrary workers that are available. The
    # `result` value will be available later.
    result = coordinator.schedule(worker_fn)
  ```
  """

  def fetch(self):
    # [FUNCTION DOC]
    # Abstract: subclasses wait for the value and return it as numpy data (a single value or a
    # structure). The base implementation takes no input and always raises a not-implemented
    # error.
    """Wait for the result of `RemoteValue` and return the numpy result.

    This makes the value concrete by copying the remote value to local.

    Returns:
      The numpy array structure of the actual output of the `tf.function`
      associated with this `RemoteValue`, previously returned by a
      `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` call.
      This can be a single value, or a structure of values, depending on the
      output of the `tf.function`.

    Raises:
      tf.errors.CancelledError: If the function that produces this `RemoteValue`
        is aborted or cancelled due to failure.
    """
    raise NotImplementedError("Must be implemented in subclasses.")

  def get(self):
    # [FUNCTION DOC]
    # Abstract: subclasses wait for the value and return it as local tensors (a single tensor or a
    # structure). The base implementation takes no input and always raises a not-implemented
    # error.
    """Wait for the result of `RemoteValue` and return the tensor result.

    This makes the value concrete by copying the remote tensor to local.

    Returns:
      The actual output (in the form of `tf.Tensor`s) of the `tf.function`
      associated with this `RemoteValue`, previously returned by a
      `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` call.
      This can be a single Tensor, or a structure of Tensors, depending on the
      output of the `tf.function`.

    Raises:
      tf.errors.CancelledError: If the function that produces this `RemoteValue`
        is aborted or cancelled due to failure.
    """
    raise NotImplementedError("Must be implemented in subclasses.")
