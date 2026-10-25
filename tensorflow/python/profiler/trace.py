# [MODULE DOC]
# Python tracing API for the profiler. Exposes the enabled check (a direct, low-overhead binding
# to the native check that the profiler is on), the Trace context manager that emits a named trace
# event with metadata, and trace_wrapper, a decorator form of the same.
# Trace events are produced only while a profiler session is active.
# Import-time side effects: binds the enabled name and registers the Trace class export.
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
"""Trace allows the profiler to trace Python events."""

import functools

from tensorflow.python.profiler.internal import _pywrap_traceme
from tensorflow.python.util.tf_export import tf_export

# This is a low-overhead function that directly calls C++ to check if the
# profiler is enabled.
enabled = _pywrap_traceme.traceme_enabled


@tf_export('profiler.experimental.Trace', v1=[])
class Trace(object):
  # [CLASS DOC]
  # Context manager that records one named trace event in the profiler timeline. The event starts
  # when constructed (the native clock starts at creation) and stops on exit. Holds a native trace
  # object when the profiler is enabled at construction time, otherwise nothing, making all
  # operations no-ops. Supports adding metadata after creation.
  """Context manager that generates a trace event in the profiler.

  A trace event will start when entering the context, and stop and save the
  result to the profiler when exiting the context. Open TensorBoard Profile tab
  and choose trace viewer to view the trace event in the timeline.

  Trace events are created only when the profiler is enabled. More information
  on how to use the profiler can be found at
  https://tensorflow.org/guide/profiler

  Example usage:
  ```python
  tf.profiler.experimental.start('logdir')
  for step in range(num_steps):
    # Creates a trace event for each training step with the step number.
    with tf.profiler.experimental.Trace("Train", step_num=step, _r=1):
      train_fn()
  tf.profiler.experimental.stop()
  ```
  """

  def __init__(self, name, **kwargs):
    # [FUNCTION DOC]
    # Creates the trace event. Inputs: name (event name) and keyword arguments (metadata whose keys
    # and values are convertible to strings). If the profiler is enabled, creates the native trace
    # object, which starts the clock immediately; otherwise stores nothing. Returns nothing.
    """Creates a trace event in the profiler.

    Args:
      name: The name of the trace event.
      **kwargs: Keyword arguments added to the trace event.
                Both the key and value are of types that
                can be converted to strings, which will be
                interpreted by the profiler according to the
                traceme name.

      Example usage:

      ```python

        tf.profiler.experimental.start('logdir')
        for step in range(num_steps):
          # Creates a trace event for each training step with the
          # step number.
          with tf.profiler.experimental.Trace("Train", step_num=step):
            train_fn()
        tf.profiler.experimental.stop()

      ```
      The example above uses the keyword argument "step_num" to specify the
      training step being traced.
    """
    if enabled():
      # Creating _pywrap_traceme.TraceMe starts the clock.
      self._traceme = _pywrap_traceme.TraceMe(name, **kwargs)
    else:
      self._traceme = None

  def __enter__(self):
    # [FUNCTION DOC]
    # Entry hook of the context manager. Does no work (the clock already started in the
    # constructor, to avoid an extra native call). Returns this trace object.
    # Starting the TraceMe clock here would require an extra Python->C++ call.
    return self

  def set_metadata(self, **kwargs):
    # [FUNCTION DOC]
    # Adds metadata to the event after creation. Input: keyword arguments of key-value pairs.
    # Forwards them to the native trace object only if the profiler was enabled at construction
    # and at least one pair is given; otherwise does nothing. Returns nothing.
    """Sets metadata in this trace event.

    Args:
      **kwargs: metadata in key-value pairs.

    This method enables setting metadata in a trace event after it is
    created.

    Example usage:

    ```python

      def call(function):
        with tf.profiler.experimental.Trace("call",
             function_name=function.name) as tm:
          binary, in_cache = jit_compile(function)
          tm.set_metadata(in_cache=in_cache)
          execute(binary)

    ```
    In this example, we want to trace how much time spent on
    calling a function, which includes compilation and execution.
    The compilation can be either getting a cached copy of the
    binary or actually generating the binary, which is indicated
    by the boolean "in_cache" returned by jit_compile(). We need
    to use set_metadata() to pass in_cache because we did not know
    the in_cache value when the trace was created (and we cannot
    create the trace after jit_compile(), because we want
    to measure the entire duration of call()).
    """
    if self._traceme and kwargs:
      self._traceme.SetMetadata(**kwargs)

  def __exit__(self, exc_type, exc_val, exc_tb):
    # [FUNCTION DOC]
    # Exit hook: stops the native trace event (recording its duration) if one exists. Exception
    # arguments are ignored and exceptions are not suppressed. Returns nothing.
    if self._traceme:
      self._traceme.Stop()


def trace_wrapper(trace_name, **trace_kwargs):
  # [FUNCTION DOC]
  # Decorator form of the Trace context manager. Inputs: trace_name (either an event name, or
  # directly a callable to decorate, in which case the name comes from its qualified name, else
  # its plain name, else a placeholder) and keyword arguments added to each event.
  # Returns a decorator when given a name, or the decorated function when given a callable.
  # The wrapped function checks at call time whether the profiler is enabled and only then
  # runs inside a Trace scope; otherwise it calls the function directly (faster). The wrapper
  # keeps the original function metadata and passes through arguments and the return value.
  """Decorator alternative to `with Trace(): ...`.  It's faster.

  Args:
    trace_name: The name of the trace event, or a callable to be traced, in
      which case the name is inferred from qualname or name of the callable.
    **trace_kwargs: Keyword arguments added to the trace event. Both the key and
      value are of types that can be converted to strings, which will be
      interpreted by the profiler according to the traceme name.

  Returns:
    A decorator that can wrap a function and apply `Trace` scope if needed,
    or a decorated function if used as a decorator directly.

  Example usage:
    ```python

    @trace_wrapper('trace_name')
    def func(x, y, z):
      pass  # code to execute and apply `Trace` if needed.

    # Equivalent to
    # with Trace('trace_name'):
    #   func(1, 2, 3)
    func(1, 2, 3)
    ```

  or
    ```python

    @trace_wrapper
    def func(x, y, z):
      pass  # code to execute and apply `Trace` if needed.

    # Equivalent to
    # with Trace(func.__qualname__):
    #   func(1, 2, 3)
    func(1, 2, 3)
    ```

  """

  if callable(trace_name):
    func = trace_name
    name = getattr(func, '__qualname__', None)
    if not name:
      name = getattr(func, '__name__', 'unknown function')

    return trace_wrapper(name)(func)

  def inner_wrapper(func):
    # [FUNCTION DOC]
    # HELPER: Decorator produced by trace_wrapper for a given event name.
    # Input: func, the function to wrap. Returns the wrapped function that adds a trace scope
    # around calls when the profiler is enabled.

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: The wrapper returned by the decorator. Accepts any positional and keyword
      # arguments, forwards them to the wrapped function and returns its result. If the profiler
      # is enabled when called, the call is wrapped in a trace event with the configured name and
      # metadata; exceptions from the function propagate.
      if enabled():
        with Trace(trace_name, **trace_kwargs):
          return func(*args, **kwargs)
      return func(*args, **kwargs)

    return wrapped

  return inner_wrapper
