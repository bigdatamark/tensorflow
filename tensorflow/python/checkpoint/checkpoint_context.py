# [MODULE DOC]
# Thread-local flags describing the checkpoint-saving context: whether the current thread is saving
# upon preemption, and whether async-checkpoint metrics recording is being controlled. Exposes
# context managers (preemption_save_context, async_metrics_context) and query functions. Creates
# two module-level thread-local singletons at import time. Helper module for checkpoint saving code.
#
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Context for saving checkpoint."""

import contextlib
import threading


class PreemptionSaveContext(threading.local):
  # [CLASS DOC]
  # Thread-local holder of a boolean flag telling whether the current thread is saving a
  # checkpoint because of preemption. The flag starts false in every thread.
  """A context for saving checkpoint upon preemption."""

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the thread-local state; the preemption flag starts false (per thread).
    super().__init__()
    self._in_preemption_save_context = False

  def enter_preemption_save_context(self):
    # [FUNCTION DOC]
    # Sets the preemption-save flag for the current thread to true. Returns nothing.
    self._in_preemption_save_context = True

  def exit_preemption_save_context(self):
    # [FUNCTION DOC]
    # Clears the preemption-save flag for the current thread. Returns nothing.
    self._in_preemption_save_context = False

  def in_preemption_save_context(self):
    # [FUNCTION DOC]
    # Returns the boolean preemption-save flag of the current thread.
    return self._in_preemption_save_context


_preemption_save_context = PreemptionSaveContext()


@contextlib.contextmanager
def preemption_save_context():
  # [FUNCTION DOC]
  # Context manager that marks the current thread as inside a preemption save for the duration of
  # the block and always clears the flag on exit, even on error. Takes no inputs, yields nothing.
  _preemption_save_context.enter_preemption_save_context()
  try:
    yield
  finally:
    _preemption_save_context.exit_preemption_save_context()


def in_preemption_save_context():
  # [FUNCTION DOC]
  # Returns true if the current thread is inside a preemption save context.
  return _preemption_save_context.in_preemption_save_context()


class AsyncMetricsContext(threading.local):
  # [CLASS DOC]
  # Thread-local holder of a boolean flag used to control metrics recording while asynchronous
  # checkpointing is in use. The flag starts false in every thread.
  """A context for controlling metrics recording when async checkpoint is used.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the thread-local state; the async-metrics flag starts false (per thread).
    super().__init__()
    self._in_async_metrics_context = False

  def enter_async_metrics_context(self):
    # [FUNCTION DOC]
    # Sets the async-metrics flag for the current thread to true. Returns nothing.
    self._in_async_metrics_context = True

  def exit_async_metrics_context(self):
    # [FUNCTION DOC]
    # Clears the async-metrics flag for the current thread. Returns nothing.
    self._in_async_metrics_context = False

  def in_async_metrics_context(self):
    # [FUNCTION DOC]
    # Returns the boolean async-metrics flag of the current thread.
    return self._in_async_metrics_context


_async_metrics_context = AsyncMetricsContext()


@contextlib.contextmanager
def async_metrics_context():
  # [FUNCTION DOC]
  # Context manager that marks the current thread as inside the async-metrics context for the
  # duration of the block and always clears the flag on exit. Takes no inputs, yields nothing.
  _async_metrics_context.enter_async_metrics_context()
  try:
    yield
  finally:
    _async_metrics_context.exit_async_metrics_context()


def in_async_metrics_context():
  # [FUNCTION DOC]
  # Returns true if the current thread is inside the async-metrics context.
  return _async_metrics_context.in_async_metrics_context()
