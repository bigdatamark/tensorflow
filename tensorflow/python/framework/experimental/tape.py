# [MODULE DOC]
# Experimental, test-only gradient tape built on the native unified-API tape bindings. Exposes
# one class, a gradient tape with a small interface (watch, gradient, context manager) similar to
# the regular gradient tape. Relies on the thread-local default context stack and the global
# gradient registry.
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
"""Experimental impl for GradientTape using unified APIs, for testing only."""

from tensorflow.python.framework.experimental import _tape
from tensorflow.python.framework.experimental import context_stack
from tensorflow.python.framework.experimental import gradient_registry
from tensorflow.python.util import nest


class GradientTape(object):
  # [CLASS DOC]
  # Gradient tape for the unified API. Holds a native tape, a tape context wrapping the default
  # context at creation time, and the active context manager (set while inside a with-block).
  # Used as a context manager: operations performed inside are recorded; gradients are computed
  # afterwards through the gradient method.
  """GradientTape using the unified API."""

  def __init__(self, persistent=False):
    # [FUNCTION DOC]
    # Constructor. Input: persistent, whether the tape may be queried more than once (default
    # false). Creates the native tape and a tape context that wraps the current default context,
    # the native tape and the global gradient registry. Returns nothing.
    self._c_tape = _tape.Tape(persistent)
    ctx = context_stack.get_default()
    self._tape_context = _tape.TapeContext(
        ctx, self._c_tape, gradient_registry.get_global_registry())
    self._ctx_manager = None

  def watch(self, t):
    # [FUNCTION DOC]
    # Marks a tensor so operations on it are tracked for gradients. Input: t, a unified-API
    # tensor handle. Returns nothing; mutates the native tape.
    self._c_tape.Watch(t)

  # TODO(srbs): Add support for unconnected_gradients.
  def gradient(self, targets, sources, output_gradients=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the regular gradient tape, there is no way to choose how unconnected
    # sources are reported (see the to-do above); also an empty list of output gradients is
    # treated the same as none because of the truthiness test.
    # Computes gradients. Inputs: targets, a tensor handle or nested structure of them to
    # differentiate; sources, a handle or nested structure of handles to differentiate with
    # respect to; output_gradients, optional initial gradients for the targets (default none,
    # meaning an empty list is passed). Uses the current default context. Returns the gradients
    # packed in the same structure as sources. Errors come from the native tape.
    ctx = context_stack.get_default()
    flat_targets = nest.flatten(targets)
    flat_sources = nest.flatten(sources)
    out_grads = self._c_tape.ComputeGradient(ctx, flat_targets, flat_sources,
                                             output_gradients or [])
    return nest.pack_sequence_as(sources, out_grads)

  def __enter__(self):
    # [FUNCTION DOC]
    # Context manager entry. Makes the tape's context the thread's default context so operations
    # are recorded, remembering the manager so exit can undo it. Returns the tape itself.
    """Enters a context inside which operations are recorded on this tape."""
    self._ctx_manager = context_stack.set_default(self._tape_context)
    self._ctx_manager.__enter__()
    return self

  def __exit__(self, typ, value, traceback):
    # [FUNCTION DOC]
    # Context manager exit. Inputs: the exception type, value and traceback (or nothing), passed
    # through to the stored context manager, which pops the tape's context from the default stack.
    # Clears the stored manager. Returns nothing, so exceptions are not suppressed.
    self._ctx_manager.__exit__(typ, value, traceback)
    self._ctx_manager = None
