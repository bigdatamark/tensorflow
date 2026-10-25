# [MODULE DOC]
# Public TensorFlow 2.x profiler API. Exposes the options tuple type, start, stop, warmup,
# start_server and the Profile context manager (most exported under the experimental profiler
# namespace for v2). Wraps the native profiler session and server bindings.
# Module state: one global profiler session handle guarded by a lock, so only one programmatic
# profiling session can be active per process.
# Import-time side effects: creates the lock and registers exported symbols; nothing else.
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
"""TensorFlow 2.x Profiler.

The profiler has two modes:
- Programmatic Mode: start(logdir), stop(), and Profiler class. Profiling starts
                     when calling start(logdir) or create a Profiler class.
                     Profiling stops when calling stop() to save to
                     TensorBoard logdir or destroying the Profiler class.
- Sampling Mode: start_server(). It will perform profiling after receiving a
                 profiling request.

NOTE: Only one active profiler session is allowed. Use of simultaneous
Programmatic Mode and Sampling Mode is undefined and will likely fail.

NOTE: The Keras TensorBoard callback will automatically perform sampled
profiling. Before enabling customized profiling, set the callback flag
"profile_batches=[]" to disable automatic sampled profiling.
"""

import collections
import threading

from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util.tf_export import tf_export

_profiler = None
_profiler_lock = threading.Lock()


@tf_export('profiler.experimental.ProfilerOptions', v1=[])
class ProfilerOptions(
    collections.namedtuple('ProfilerOptions', [
        'host_tracer_level', 'python_tracer_level', 'device_tracer_level',
        'delay_ms'
    ])):
  # [CLASS DOC]
  # Immutable named tuple of profiler settings with four fields: host tracer level, python tracer
  # level, device tracer level and delay in milliseconds. Created by users and passed to start
  # (or the Profile context manager, or the client trace call), where it is converted to a mapping.
  # The overridden constructor supplies defaults: host level 2, python level 0, device level 1 and
  # no delay (nothing, letting the profiler pick). No validation of values is performed here.
  """Options for finer control over the profiler.

  Use `tf.profiler.experimental.ProfilerOptions` to control `tf.profiler`
  behavior.

  Fields:
    host_tracer_level: Adjust CPU tracing level. Values are: `1` - critical info
      only, `2` - info, `3` - verbose. [default value is `2`]
    python_tracer_level: Toggle tracing of Python function calls. Values are:
      `1` - enabled, `0` - disabled [default value is `0`]
    device_tracer_level: Adjust device (TPU/GPU) tracing level. Values are:
      `1` - enabled, `0` - disabled [default value is `1`]
    delay_ms: Requests for all hosts to start profiling at a timestamp that is
      `delay_ms` away from the current time. `delay_ms` is in milliseconds. If
      zero, each host will start profiling immediately upon receiving the
      request. Default value is `None`, allowing the profiler guess the best
      value.
  """

  def __new__(cls,
              host_tracer_level=2,
              python_tracer_level=0,
              device_tracer_level=1,
              delay_ms=None):
    # [FUNCTION DOC]
    # Constructor for the options tuple. Inputs (all optional): host_tracer_level (default 2),
    # python_tracer_level (default 0), device_tracer_level (default 1), delay_ms (default
    # nothing). Returns a new immutable options instance holding these values. No validation.
    return super(ProfilerOptions,
                 cls).__new__(cls, host_tracer_level, python_tracer_level,
                              device_tracer_level, delay_ms)


@tf_export('profiler.experimental.start', v1=[])
def start(logdir, options=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: If the native start reports that a session already exists, the freshly created
  # global session handle is not cleared before re-raising, so the module keeps a handle and
  # later start calls fail until stop is called.
  # Starts a profiling session. Inputs: logdir (directory for results), options (optional options
  # tuple, converted to a mapping; nothing means defaults). Under the module lock: raises an
  # already-exists error if a session is already held; otherwise creates a native session and
  # starts it. On already-exists from native code it logs a warning about mixing server and API
  # and raises already-exists; on any other failure it clears the session handle and re-raises.
  # Returns nothing. Side effect: sets the module-level session.
  """Start profiling TensorFlow performance.

  Args:
    logdir: Profiling results log directory.
    options: `ProfilerOptions` namedtuple to specify miscellaneous profiler
      options. See example usage below.

  Raises:
    AlreadyExistsError: If a profiling session is already running.

  Example usage:
  ```python
  options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                                     python_tracer_level = 1,
                                                     device_tracer_level = 1)
  tf.profiler.experimental.start('logdir_path', options = options)
  # Training code here
  tf.profiler.experimental.stop()
  ```

  To view the profiling results, launch TensorBoard and point it to `logdir`.
  Open your browser and go to `localhost:6006/#profile` to view profiling
  results.

  """
  global _profiler
  with _profiler_lock:
    if _profiler is not None:
      raise errors.AlreadyExistsError(None, None,
                                      'Another profiler is running.')
    _profiler = _pywrap_profiler.ProfilerSession()
    try:
      # support for namedtuple in pybind11 is missing, we change it to
      # dict type first.
      opts = dict(options._asdict()) if options is not None else {}
      _profiler.start(logdir, opts)
    except errors.AlreadyExistsError:
      logging.warning('Another profiler session is running which is probably '
                      'created by profiler server. Please avoid using profiler '
                      'server and profiler APIs at the same time.')
      raise errors.AlreadyExistsError(None, None,
                                      'Another profiler is running.')
    except Exception:
      _profiler = None
      raise


@tf_export('profiler.experimental.stop', v1=[])
def stop(save=True):
  # [FUNCTION DOC]
  # Stops the active profiling session. Input: save (default true) whether to export results to
  # the TensorBoard log directory. Under the module lock: raises an unavailable error if no
  # session is running; if save is true exports results (on failure clears the handle and
  # re-raises). Always clears the module-level session handle afterwards. Returns nothing.
  """Stops the current profiling session.

  The profiler session will be stopped and profile results can be saved.

  Args:
    save: An optional variable to save the results to TensorBoard. Default True.

  Raises:
    UnavailableError: If there is no active profiling session.
  """
  global _profiler
  with _profiler_lock:
    if _profiler is None:
      raise errors.UnavailableError(
          None, None,
          'Cannot export profiling results. No profiler is running.')
    if save:
      try:
        _profiler.export_to_tb()
      except Exception:
        _profiler = None
        raise
    _profiler = None


def warmup():
  # [FUNCTION DOC]
  # Warms up the profiler by starting a session with an empty log directory and stopping it
  # without saving, so that one-time setup (such as loading the GPU tracing library) happens
  # before real profiling. No inputs, returns nothing. Raises already-exists if a session is
  # active. Not exported as a public symbol in this module.
  """Warm-up the profiler session.

  The profiler session will set up profiling context, including loading CUPTI
  library for GPU profiling. This is used for improving the accuracy of
  the profiling results.

  """
  start('')
  stop(save=False)


@tf_export('profiler.experimental.server.start', v1=[])
def start_server(port):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring example is garbled (an unterminated code fence run into prose).
  # Starts a gRPC profiler server in the background listening on the given port (integer). The
  # server lives until the process exits; profiling happens when a client requests it.
  # Returns nothing; errors come from the native layer. Side effect: opens a network port.
  """Start a profiler grpc server that listens to given port.

  The profiler server will exit when the process finishes. The service is
  defined in tensorflow/core/profiler/profiler_service.proto.

  Args:
    port: port profiler server listens to.
  Example usage: ```python tf.profiler.experimental.server.start(6009) # do
    your training here.
  """
  _pywrap_profiler.start_server(port)


@tf_export('profiler.experimental.Profile', v1=[])
class Profile(object):
  # [CLASS DOC]
  # Context manager wrapping start and stop. Holds the log directory and optional options given at
  # construction. Entering starts profiling; exiting stops it and saves results to the log
  # directory. Only one profiling session may be active at a time.
  """Context-manager profile API.

  Profiling will start when entering the scope, and stop and save the results to
  the logdir when exits the scope. Open TensorBoard profile tab to view results.

  Example usage:
  ```python
  with tf.profiler.experimental.Profile("/path/to/logdir"):
    # do some work
  ```
  """

  def __init__(self, logdir, options=None):
    # [FUNCTION DOC]
    # Stores the log directory (where results are saved) and optional profiler options for later
    # use on entry. Does not start profiling. Returns nothing.
    """Creates a context manager object for profiler API.

    Args:
      logdir: profile data will save to this directory.
      options: An optional `tf.profiler.experimental.ProfilerOptions` can be
        provided to fine tune the profiler's behavior.
    """
    self._logdir = logdir
    self._options = options

  def __enter__(self):
    # [FUNCTION DOC]
    # Starts profiling with the stored log directory and options. Raises already-exists if a
    # session is already active. Returns nothing (not the context object).
    start(self._logdir, self._options)

  def __exit__(self, typ, value, tb):
    # [FUNCTION DOC]
    # Stops profiling and saves results; exception arguments are ignored and exceptions are not
    # suppressed. Raises an unavailable error if no session is active. Returns nothing.
    stop()
