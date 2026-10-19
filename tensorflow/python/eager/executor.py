# [MODULE DOC]
# Python wrapper around the native eager executor. An executor schedules and runs eagerly
# dispatched ops, either synchronously or asynchronously. Exposes the Executor class, which
# owns a native handle, and a factory function that creates a new native executor and wraps it.
# Used by the eager context to set a per-thread executor. No import-time side effects.

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Executor for eager execution."""

from tensorflow.python import pywrap_tfe


class Executor(object):
  # [CLASS DOC]
  # KNOWN ISSUE: the example in the docstring is not valid as written (it ends a call statement
  # with a colon and uses a local name that shadows the executor module).
  # Owning wrapper for a native eager executor handle. Holds a single slot, the opaque native
  # handle, and nothing else. Normally created through the module-level factory function rather
  # than directly. Offers query of async mode, access to the handle, waiting for pending ops and
  # clearing errors. Deleting the wrapper waits for pending work and frees the native executor.
  """A class for handling eager execution.

  The default behavior for asynchronous execution is to serialize all ops on
  a single thread. Having different `Executor` objects in different threads
  enables executing ops asynchronously in parallel:

  ```python
  def thread_function():
    executor = executor.Executor(enable_async=True):
    context.set_executor(executor)

  a = threading.Thread(target=thread_function)
  a.start()
  b = threading.Thread(target=thread_function)
  b.start()
  ```
  """

  __slots__ = ["_handle"]

  def __init__(self, handle):
    # [FUNCTION DOC]
    # Constructor. Takes the opaque native executor handle (obtained from the native layer) and
    # stores it. Takes ownership: the native executor is freed when this object is deleted.
    # Returns nothing.
    self._handle = handle

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer. Waits for all pending ops of the executor to finish, then deletes the native
    # executor. A type error (for example when the native module or the handle was already torn
    # down at interpreter shutdown) is silently suppressed so no warnings are printed.
    # Side effects: blocks until pending nodes finish; releases native resources. Returns nothing.
    try:
      self.wait()
      pywrap_tfe.TFE_DeleteExecutor(self._handle)
    except TypeError:
      # Suppress some exceptions, mainly for the case when we're running on
      # module deletion. Things that can go wrong include the pywrap module
      # already being unloaded, self._handle. no longer being
      # valid, and so on. Printing warnings in these cases is silly
      # (exceptions raised from __del__ are printed as warnings to stderr).
      pass  # 'NoneType' object is not callable when the handle has been
      # partially unloaded.

  def is_async(self):
    # [FUNCTION DOC]
    # Reports whether this executor runs ops asynchronously. No inputs. Returns a boolean
    # obtained from the native layer.
    return pywrap_tfe.TFE_ExecutorIsAsync(self._handle)

  def handle(self):
    # [FUNCTION DOC]
    # Returns the opaque native executor handle held by this wrapper. No inputs, no side effects.
    return self._handle

  def wait(self):
    # [FUNCTION DOC]
    # Blocks until every op dispatched on this executor has finished. Errors from the native
    # layer propagate to the caller. No inputs; returns nothing.
    """Waits for ops dispatched in this executor to finish."""
    pywrap_tfe.TFE_ExecutorWaitForAllPendingNodes(self._handle)

  def clear_error(self):
    # [FUNCTION DOC]
    # Clears any error recorded by this executor during earlier execution so later ops can run.
    # No inputs; returns nothing.
    """Clears errors raised in this executor during execution."""
    pywrap_tfe.TFE_ExecutorClearError(self._handle)


def new_executor(enable_async,
                 enable_streaming_enqueue=True,
                 in_flight_nodes_limit=0):
  # [FUNCTION DOC]
  # Factory that creates a new native eager executor and wraps it in an Executor object.
  # Inputs: enable_async (boolean, required) selects asynchronous execution;
  # enable_streaming_enqueue (boolean, default true) lets ops be streamed to remote workers
  # without waiting; in_flight_nodes_limit (integer, default zero meaning no limit) caps the
  # number of nodes allowed in flight. Returns the new Executor wrapper.
  handle = pywrap_tfe.TFE_NewExecutor(enable_async, enable_streaming_enqueue,
                                      in_flight_nodes_limit)
  return Executor(handle)
