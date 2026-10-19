# coding=utf-8
# [MODULE DOC]
# Utilities for collective operations in distribution strategies: the enumeration of cross-device
# communication implementations (auto, ring, NCCL) with a legacy alias, the options object that
# carries hints (pack size, timeout, implementation) for collectives, a public facade class that
# only exists to expose the options API, and a deprecated hints class. Public classes are
# registered with the API export decorator at import time. Used by the collective all-reduce
# strategy and cross-device ops. Not a helper module.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for collectives."""

import copy
import enum

from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# TODO(b/170340570): print deprecation warning for CollectiveCommunication.
@tf_export("distribute.experimental.CommunicationImplementation",
           "distribute.experimental.CollectiveCommunication")
class CommunicationImplementation(enum.Enum):
  # [CLASS DOC]
  # Enumeration of preferred cross-device communication implementations for collectives. Members
  # (string valued, equal to their names): automatic choice, ring algorithms, and NCCL. Exported
  # publicly under two names (the older one is a deprecated alias); the module also defines a
  # plain alias for it named CollectiveCommunication. Instances are looked up by upper-case name
  # when options are given as strings.
  """Cross device communication implementation.

  Warning: The alias `tf.distribute.experimental.CollectiveCommunication` is
  deprecated and will be removed in a future version. Use
  `tf.distribute.experimental.CommunicationImplementation` instead.

  * `AUTO`: Automatically chosen by Tensorflow.
  * `RING`: TensorFlow's ring algorithms for all-reduce and
    all-gather.
  * `NCCL`: NVIDIA®'s NCCL library. This is now only used for all-reduce on
    GPUs; all-reduce on CPU, all-gather and broadcast fallbacks to RING.
  """
  AUTO = "AUTO"
  RING = "RING"
  NCCL = "NCCL"
  # TODO(ayushd): add ncclAllGather implementation.


CollectiveCommunication = CommunicationImplementation


@tf_export("distribute.experimental.CommunicationOptions")
class _OptionsExported(object):
  # [CLASS DOC]
  # HELPER: public API facade for the options class; exists only so the exported documentation
  # and signature are separate from the internal implementation. Constructing it never yields an
  # instance of itself: the object-creation hook returns an instance of the internal options
  # class, so its initializer is never run. Holds no state.
  """Options for cross device communications like All-reduce.

  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.

  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.

  Examples:

  ```python
  options = tf.distribute.experimental.CommunicationOptions(
      bytes_per_pack=50 * 1024 * 1024,
      timeout_seconds=120.0,
      implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
  )
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, options=options)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```

  """

  def __new__(cls, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: redirects construction of the public facade to the internal options class.
    # Takes any positional and keyword arguments and forwards them unchanged. Returns a new
    # instance of the internal options class (not of this class); validation errors therefore
    # come from that class.
    # We expose a dummy class so that we can separate internal and public APIs.
    # Note that __init__ won't be called on the returned object if it's a
    # different class [1].
    # [1] https://docs.python.org/3/reference/datamodel.html#object.__new__
    return Options(*args, **kwargs)

  def __init__(self,
               bytes_per_pack=0,
               timeout_seconds=None,
               implementation=CommunicationImplementation.AUTO):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it creates a CollectiveHints and raises a value error, but
    # the body does nothing and is never run, because the object-creation hook returns an
    # instance of the internal options class. It exists only to carry the public signature and
    # documentation (the internal class copies this docstring).
    # Parameters: pack size in bytes (default zero, automatic), timeout in seconds (default
    # none), and preferred communication implementation (default automatic). Returns nothing.
    """Creates a CollectiveHints.

    Args:
      bytes_per_pack: a non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, the value is determined
        automatically. This hint is respected by all multi-replica strategies
        except `TPUStrategy`.
      timeout_seconds: a float or None, timeout in seconds. If not None, the
        collective raises `tf.errors.DeadlineExceededError` if it takes longer
        than this timeout. Zero disables timeout. This can be useful when
        debugging hanging issues.  This should only be used for debugging since
        it creates a new thread for each collective, i.e. an overhead of
        `timeout_seconds * num_collectives_per_second` more threads. This only
        works for `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
      implementation: a
        `tf.distribute.experimental.CommunicationImplementation`. This is a hint
        on the preferred communication implementation. Possible values include
        `AUTO`, `RING`, and `NCCL`. NCCL is generally more performant for GPU,
        but doesn't work for CPU. This only works for
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

    Raises:
      ValueError: When arguments have invalid value.
    """
    pass


class Options(object):
  # [CLASS DOC]
  # Internal implementation of the collective communication options. Holds three public
  # attributes: pack size in bytes (non-negative integer, zero meaning automatic), timeout in
  # seconds (number or nothing; nothing means unset) and the preferred communication
  # implementation. Created directly or via the public facade class; supports merging with
  # another options object and a readable string form.
  """Implementation of OptionsInterface."""

  def __init__(self,
               bytes_per_pack=0,
               timeout_seconds=None,
               implementation=CommunicationImplementation.AUTO):
    # [FUNCTION DOC]
    # Validates and stores the options. Parameters: bytes_per_pack (integer, default zero,
    # must not be negative), timeout_seconds (number or nothing, default nothing, stored
    # unchecked), implementation (an enumeration member or a case-insensitive string naming one;
    # default automatic). Strings are converted to the enumeration member (an unknown name raises
    # a value error from the enumeration). Raises a value error if the pack size is negative or
    # if the implementation is not an enumeration member after conversion. Sets the three
    # attributes; returns nothing. The class-level docstring of this initializer is copied from
    # the public facade.
    if bytes_per_pack < 0:
      raise ValueError(
          f"Argument `bytes_per_pack` must be >=0, Received {bytes_per_pack}.")
    if isinstance(implementation, str):
      implementation = CommunicationImplementation(implementation.upper())
    if not isinstance(implementation, CommunicationImplementation):
      raise ValueError(
          "Argument `implementation` must be instance of "
          "`tf.distribute.experimental.CommunicationImplementation`.")
    self.bytes_per_pack = bytes_per_pack
    self.timeout_seconds = timeout_seconds
    self.implementation = implementation

  __init__.__doc__ = _OptionsExported.__init__.__doc__

  def merge(self, options):
    # [FUNCTION DOC]
    # Combines this options object with another. Makes a deep copy of this object; if the other
    # is nothing, returns the copy. Otherwise a field of the other overrides the copy only when it
    # is not its default (pack size not zero, timeout not nothing, implementation not automatic).
    # Consequently an explicit zero pack size or automatic implementation in the other cannot
    # override a non-default value here. Returns the new options object; does not mutate either
    # input.
    """Merges with another options and returns a new one.

    Values specified in the `options` takes precedence if they're not the
    default.

    Args:
      options: a `tf.distribute.experimental.CollectiveCommunication`.

    Returns:
      A new `tf.distribute.experimental.CollectiveCommunication`.
    """
    merged = copy.deepcopy(self)
    if options is None:
      return merged
    if options.bytes_per_pack != 0:
      merged.bytes_per_pack = options.bytes_per_pack
    if options.timeout_seconds is not None:
      merged.timeout_seconds = options.timeout_seconds
    if options.implementation != CommunicationImplementation.AUTO:
      merged.implementation = options.implementation
    return merged

  def __str__(self):
    # [FUNCTION DOC]
    # Returns a human-readable string listing the pack size, timeout and implementation.
    # Note the separator after the pack size has no space while the following one does.
    return (f"Options(bytes_per_pack={self.bytes_per_pack},"
            f"timeout_seconds={self.timeout_seconds}, "
            f"implementation={self.implementation})")


@tf_export("distribute.experimental.CollectiveHints")
class Hints(object):
  # [CLASS DOC]
  # Deprecated public facade (older name for the options API, supporting only pack size and
  # timeout). Like the options facade, constructing it returns an instance of the internal
  # options class, so the initializer here is never run. Holds no state.
  """Hints for collective operations like AllReduce.

  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.

  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.

  Examples:

  - bytes_per_pack

  ```python
  hints = tf.distribute.experimental.CollectiveHints(
      bytes_per_pack=50 * 1024 * 1024)
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, experimental_hints=hints)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```

  - timeout_seconds

  ```python
  strategy = tf.distribute.MirroredStrategy()
  hints = tf.distribute.experimental.CollectiveHints(
      timeout_seconds=120.0)
  try:
    strategy.reduce("sum", v, axis=None, experimental_hints=hints)
  except tf.errors.DeadlineExceededError:
    do_something()
  ```

  """

  @deprecation.deprecated(
      None, "use distribute.experimental.CommunicationOptions instead")
  def __new__(cls, bytes_per_pack=0, timeout_seconds=None):
    # [FUNCTION DOC]
    # Deprecated construction hook (a deprecation warning is logged on each call, pointing to
    # the communication options). Takes pack size in bytes (default zero) and timeout in seconds
    # (default nothing). Returns a new instance of the internal options class with those values
    # and the default implementation; a negative pack size raises a value error there.
    return Options(
        bytes_per_pack=bytes_per_pack, timeout_seconds=timeout_seconds)

  def __init__(self, bytes_per_pack=0, timeout_seconds=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: documented as raising a value error and creating hints, but the body does
    # nothing and is never run, since the object-creation hook returns an instance of a
    # different class. It only carries the public signature and documentation.
    # Parameters: pack size in bytes and timeout in seconds. Returns nothing.
    """Creates a CollectiveHints.

    Args:
      bytes_per_pack: a non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, the value is determined
        automatically. This only applies to all-reduce with
        `MultiWorkerMirroredStrategy` currently.
      timeout_seconds: a float or None, timeout in seconds. If not None, the
        collective raises `tf.errors.DeadlineExceededError` if it takes longer
        than this timeout. This can be useful when debugging hanging issues.
        This should only be used for debugging since it creates a new thread for
        each collective, i.e. an overhead of `timeout_seconds *
        num_collectives_per_second` more threads.  This only works for
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

    Raises:
      ValueError: When arguments have invalid value.
    """
    pass
