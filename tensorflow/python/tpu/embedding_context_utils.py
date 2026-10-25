# [MODULE DOC]
# HELPER: helper module for TPU embedding pipelining control.
# Holds a per-thread flag saying whether embedding pipelining is enabled, a module level
# instance of it, and a context manager that temporarily disables pipelining. Import time
# creates the module level state object (enabled by default for each thread).
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions for TPU Embedding Contexts."""

import threading
from absl import logging


class _EmbeddingPipeliningState(threading.local):
  # [CLASS DOC]
  # HELPER: thread local holder for the embedding pipelining flag.
  # Each thread sees its own enabled attribute, initialized to true on first access in that
  # thread. A single module level instance is shared by the helpers in this module.

  def __init__(self):
    # [FUNCTION DOC]
    # Initializer run once per thread on first use: calls the parent initializer and sets the
    # enabled attribute to true. Takes no inputs, returns nothing.
    super().__init__()
    self.enabled = True

embedding_pipelining_state = _EmbeddingPipeliningState()


class SequentialEmbeddingContext:
  # [CLASS DOC]
  # Context manager that turns off embedding pipelining in the current thread while active.
  # It remembers the flag value seen at construction and restores it on exit. State is the
  # saved original flag. Used with a with statement; changes the module level per-thread flag.
  """Disables embedding pipelining for all ops created in the scope."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no inputs, returns nothing. Records the current thread's pipelining
    # flag value for restoration later. Note the value is captured at construction time, not
    # at entry, so creating the context in one state and entering it later restores the older
    # value.
    self._original_embedding_pipelining_state_enabled = (
        embedding_pipelining_state.enabled
    )

  def __enter__(self):
    # [FUNCTION DOC]
    # Enters the scope: sets the current thread's pipelining flag to false and logs an info
    # message. Takes no inputs and returns nothing (no value is bound by the with statement).
    embedding_pipelining_state.enabled = False
    logging.info("Entering SequentialEmbeddingContext.")

  def __exit__(self, exc_type, exc_val, exc_tb):
    # [FUNCTION DOC]
    # Leaves the scope: restores the pipelining flag to the value saved at construction and
    # logs an info message. Takes the standard exception type, value and traceback arguments
    # (ignored). Returns nothing, so exceptions raised inside the scope propagate.
    embedding_pipelining_state.enabled = (
        self._original_embedding_pipelining_state_enabled
    )
    logging.info("Exiting SequentialEmbeddingContext.")
