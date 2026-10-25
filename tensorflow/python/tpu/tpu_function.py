# [MODULE DOC]
# HELPER: helper module for TPU compilation code.
# Keeps a thread-local record of the number of shards of the TPU computation currently being
# built, a context manager that sets it, an accessor for it, and a decorator that marks a function
# as an on-device training loop. Import creates one module-level context object.
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
# =============================================================================

"""Helper library for functions used during TPU compilation."""

import contextlib
import threading


class TpuContext(threading.local):
  # [CLASS DOC]
  # Thread-local holder of state about the TPU computation under construction. Currently only
  # holds the number of shards (null when no computation is being built). Each thread sees its
  # own value. One instance exists at module level.
  """A context object holding state about the TPU computation being built."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor; initialises the number of shards to null for the (per thread) context.
    """Creates a new TpuContext."""
    self._number_of_shards = None

  @property
  def number_of_shards(self):
    # [FUNCTION DOC]
    # Read-only property: the current number of shards, or null if no shard context is active.
    return self._number_of_shards

  def set_number_of_shards(self, number_of_shards):
    # [FUNCTION DOC]
    # Stores the given shard count (an integer or null to clear) in this thread's context.
    # No validation; returns nothing.
    self._number_of_shards = number_of_shards


# The Tpu context holds the number of shards when a sharded computation is
# being built, or None if no computation is being built.
_current_tpu_context = TpuContext()


@contextlib.contextmanager
def tpu_shard_context(number_of_shards):
  # [FUNCTION DOC]
  # Context manager that sets the module-level shard count for the duration of the with block and
  # resets it to null on exit (even on error). Input: number_of_shards, an integer.
  # Side effect: mutates the thread-local TPU context. Raises a not-implemented error if a shard
  # context is already active (nesting is unsupported). Yields nothing.
  """A context manager setting current number of shards."""
  if _current_tpu_context.number_of_shards is not None:
    raise NotImplementedError("tpu_shard_context cannot be nested")
  try:
    _current_tpu_context.set_number_of_shards(number_of_shards)
    yield
  finally:
    _current_tpu_context.set_number_of_shards(None)


def get_tpu_context():
  # [FUNCTION DOC]
  # Returns the module-level thread-local TPU context object (not a copy), whose shard count
  # reflects the active shard context in the calling thread.
  return _current_tpu_context


# Decorator function for tpu computation func that was passed to tpu.rewrite()
# if there is an embedded training loop in this func, trace tools will generate
# step markers for each iteration.
def on_device_training_loop(func):
  # [FUNCTION DOC]
  # Decorator that tags a function so the compiler places the step marker at the top-level while
  # loop. Input: func, any function object. Side effect: sets an attribute on the function.
  # Returns the same function.
  # Value for this attribute is from xla.DebugOptions.StepMarkerLocation.
  setattr(func, "step_marker_location", "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP")
  return func
