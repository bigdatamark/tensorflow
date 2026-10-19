# [MODULE DOC]
# HELPER: internal helper module for the experimental unified-API framework code.
# Keeps a module-level, per-thread stack of execution contexts and exposes two functions: one
# returns the context currently on top of the stack (the default context), the other temporarily
# pushes a context for the duration of a with-block. Import-time side effect: creates the global
# thread-local stack instance.
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
"""Thread-local context manager stack."""

import contextlib

from tensorflow.python.framework.experimental import thread_local_stack

_default_ctx_stack = thread_local_stack.ThreadLocalStack()


def get_default():
  # [FUNCTION DOC]
  # Returns the execution context currently on top of the calling thread's stack. Takes no
  # inputs. If the stack is empty, the underlying stack returns nothing (a null value).
  """Returns the default execution context."""
  return _default_ctx_stack.peek()


@contextlib.contextmanager
def set_default(ctx):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the push happens inside the try block, so if the push itself fails, the finally
  # clause still pops the stack and removes an entry that this call never pushed.
  # Context manager. Input: ctx, the execution context to make the default. On entry pushes ctx
  # onto the calling thread's stack; on exit (normal or via an exception) pops the top entry.
  # Yields nothing. Side effect: modifies the thread-local global stack for the with-block.
  """Returns a contextmanager with `ctx` as the default execution context."""
  try:
    _default_ctx_stack.push(ctx)
    yield
  finally:
    _default_ctx_stack.pop()
