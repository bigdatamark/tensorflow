# [MODULE DOC]
# AutoGraph logging and debugging utilities. Holds module-level state: the verbosity level
# (overrides the AUTOGRAPH_VERBOSITY environment variable when set; default level 0) and a flag
# that mirrors log messages to standard output (enabled at import time when Python is running
# interactively). Provides public set_verbosity and trace (exported as API under the autograph
# namespace) plus get_verbosity, has_verbosity, error, log and warning used throughout AutoGraph.
# Helper module for the rest of the AutoGraph package.
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
# ==============================================================================
"""Logging and debugging utilities."""

import os
import sys
import traceback

# TODO(mdan): Use a custom logger class.
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

VERBOSITY_VAR_NAME = 'AUTOGRAPH_VERBOSITY'
DEFAULT_VERBOSITY = 0

verbosity_level = None  # vlog-like. Takes precedence over the env variable.
echo_log_to_stdout = False

# In interactive Python, logging echo is enabled by default.
if hasattr(sys, 'ps1') or hasattr(sys, 'ps2'):
  echo_log_to_stdout = True


@tf_export('autograph.set_verbosity')
def set_verbosity(level, alsologtostdout=False):
  # [FUNCTION DOC]
  # Public API. Sets the module-level verbosity level to level (integer; larger is more verbose, 0
  # disables logging) and sets whether messages are also echoed to standard output
  # (alsologtostdout, boolean, default false). Both values are stored unconditionally, so calling it
  # without the flag turns echoing off even in interactive mode. Returns nothing.
  # Side effect: modifies module globals, which take precedence over the environment variable.
  """Sets the AutoGraph verbosity level.

  _Debug logging in AutoGraph_

  More verbose logging is useful to enable when filing bug reports or doing
  more in-depth debugging.

  There are two means to control the logging verbosity:

   * The `set_verbosity` function

   * The `AUTOGRAPH_VERBOSITY` environment variable

  `set_verbosity` takes precedence over the environment variable.

  For example:

  ```python
  import os
  import tensorflow as tf

  os.environ['AUTOGRAPH_VERBOSITY'] = '5'
  # Verbosity is now 5

  tf.autograph.set_verbosity(0)
  # Verbosity is now 0

  os.environ['AUTOGRAPH_VERBOSITY'] = '1'
  # No effect, because set_verbosity was already called.
  ```

  Logs entries are output to [absl](https://abseil.io)'s
  [default output](https://abseil.io/docs/python/guides/logging),
  with `INFO` level.
  Logs can be mirrored to stdout by using the `alsologtostdout` argument.
  Mirroring is enabled by default when Python runs in interactive mode.

  Args:
    level: int, the verbosity level; larger values specify increased verbosity;
      0 means no logging. When reporting bugs, it is recommended to set this
      value to a larger number, like 10.
    alsologtostdout: bool, whether to also output log messages to `sys.stdout`.
  """
  global verbosity_level
  global echo_log_to_stdout
  verbosity_level = level
  echo_log_to_stdout = alsologtostdout


@tf_export('autograph.trace')
def trace(*args):
  # [FUNCTION DOC]
  # Public API. Prints all positional arguments to standard output; intended as a debugging aid
  # that runs at graph-construction (tracing) time. Returns nothing.
  """Traces argument information at compilation time.

  `trace` is useful when debugging, and it always executes during the tracing
  phase, that is, when the TF graph is constructed.

  _Example usage_

  ```python
  import tensorflow as tf

  for i in tf.range(10):
    tf.autograph.trace(i)
  # Output: <Tensor ...>
  ```

  Args:
    *args: Arguments to print to `sys.stdout`.
  """
  print(*args)


def get_verbosity():
  # [FUNCTION DOC]
  # Returns the effective verbosity level as an integer: the value set via set_verbosity if any,
  # otherwise the integer parsed from the AUTOGRAPH_VERBOSITY environment variable, defaulting
  # to 0. Raises a value error if the environment variable is not an integer.
  global verbosity_level
  if verbosity_level is not None:
    return verbosity_level
  return int(os.getenv(VERBOSITY_VAR_NAME, DEFAULT_VERBOSITY))


def has_verbosity(level):
  # [FUNCTION DOC]
  # Returns true when the effective verbosity is at least level (integer).
  return get_verbosity() >= level


def _output_to_stdout(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: echoes a log message to standard output for the logging functions. Prints msg
  # formatted with args by percent-style substitution; if the keyword exc_info is present and
  # true, also prints the current exception traceback. Other keyword arguments are ignored.
  # Returns nothing.
  print(msg % args)
  if kwargs.get('exc_info', False):
    traceback.print_exc()


def error(level, msg, *args, **kwargs):
  # [FUNCTION DOC]
  # Logs an error-severity message through TensorFlow logging if the verbosity is at least level.
  # msg is a format string, args its substitution values, kwargs passed on to the logger. When
  # echoing is enabled the message, prefixed with an error marker, is also printed. Returns nothing.
  if has_verbosity(level):
    logging.error(msg, *args, **kwargs)
    if echo_log_to_stdout:
      _output_to_stdout('ERROR: ' + msg, *args, **kwargs)


def log(level, msg, *args, **kwargs):
  # [FUNCTION DOC]
  # Logs an info-severity message through TensorFlow logging if the verbosity is at least level.
  # msg is a format string, args its values, kwargs passed on to the logger. When echoing is
  # enabled the message is also printed to standard output. Returns nothing.
  if has_verbosity(level):
    logging.info(msg, *args, **kwargs)
    if echo_log_to_stdout:
      _output_to_stdout(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # Always logs a warning-severity message through TensorFlow logging (no verbosity check).
  # When echoing is enabled also prints the message with a warning prefix and flushes standard
  # output. Returns nothing.
  logging.warning(msg, *args, **kwargs)
  if echo_log_to_stdout:
    _output_to_stdout('WARNING: ' + msg, *args, **kwargs)
    sys.stdout.flush()
