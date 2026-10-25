# [MODULE DOC]
# TensorFlow's logging facade over the standard logging library. Provides a lazily created,
# lock-protected singleton logger named tensorflow (with a custom caller finder so reported file
# and line numbers skip logging internals, and a stream handler attached only when the root
# logger has none), v1 convenience functions (log, debug, info, warn, warning, error, fatal,
# vlog, frequency-limited and conditional variants, verbosity get and set), a prefix formatter
# in the google2 style, and level constants re-exported. Module state: the singleton logger and
# its lock, a per-call-site counter mapping, level name mapping, thread id mask and the log
# prefix function. Import-time side effects: registers exports of functions and constants.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Logging utilities."""
# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
# pylint: disable=invalid-name
import logging as _logging
import os as _os
import sys as _sys
import _thread
import time as _time
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading

from tensorflow.python.util.tf_export import tf_export

# Don't use this directly. Use get_logger() instead.
_logger = None
_logger_lock = threading.Lock()


def error_log(error_msg, level=ERROR):
  # [FUNCTION DOC]
  # HELPER: placeholder hook for internal error logging.
  # Accepts an error message and a level (default error) and discards both; does nothing and
  # returns nothing.
  """Empty helper method."""
  del error_msg, level


def _get_caller(offset=3):
  # [FUNCTION DOC]
  # HELPER: stack inspection for the logger's caller finder and the file-and-line function.
  # Starts at the stack frame offset levels above this one (default 3), remembers its source file,
  # then walks outward to the first frame from a different file. Returns a pair of that frame's
  # code object and frame object, or a pair of null values if none exists. Avoids building a
  # traceback object.
  """Returns a code and frame object for the lowest non-logging stack frame."""
  # Use sys._getframe().  This avoids creating a traceback object.
  # pylint: disable=protected-access
  f = _sys._getframe(offset)
  # pylint: enable=protected-access
  our_file = f.f_code.co_filename
  f = f.f_back
  while f:
    code = f.f_code
    if code.co_filename != our_file:
      return code, f
    f = f.f_back
  return None, None

# The definition of `findCaller` changed in Python 3.2,
# and further changed in Python 3.8
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 8:

  def _logger_find_caller(stack_info=False, stacklevel=1):  # pylint: disable=g-wrong-blank-lines
    # [FUNCTION DOC]
    # HELPER: replacement caller-finder installed on the logger for Python 3.8 and later.
    # KNOWN ISSUE: the stack level argument is accepted for signature compatibility but ignored.
    # Inputs: stack_info flag (default false) and stack level. Returns a four-item tuple of file
    # name, line number, function name and stack text (stack text is null unless requested, then
    # the formatted current stack). If no caller outside the logging module is found, returns
    # placeholders for unknown file and function and line zero.
    code, frame = _get_caller(4)
    sinfo = None
    if stack_info:
      sinfo = '\n'.join(_traceback.format_stack())
    if code:
      return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
    else:
      return '(unknown file)', 0, '(unknown function)', sinfo
elif _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:

  def _logger_find_caller(stack_info=False):  # pylint: disable=g-wrong-blank-lines
    # [FUNCTION DOC]
    # HELPER: replacement caller-finder installed on the logger for Python 3.2 through 3.7.
    # Input: stack_info flag (default false). Returns a four-item tuple of file name, line number,
    # function name and stack text (null unless requested, then the formatted current stack), or
    # placeholders for unknown file and function with line zero when no outside caller exists.
    code, frame = _get_caller(4)
    sinfo = None
    if stack_info:
      sinfo = '\n'.join(_traceback.format_stack())
    if code:
      return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
    else:
      return '(unknown file)', 0, '(unknown function)', sinfo
else:
  def _logger_find_caller():  # pylint: disable=g-wrong-blank-lines
    # [FUNCTION DOC]
    # HELPER: replacement caller-finder installed on the logger for Python versions before 3.2.
    # KNOWN ISSUE: this branch is effectively dead; the package requires a far newer Python, so
    # only the first two variants can ever be defined.
    # Takes no inputs. Returns a three-item tuple of file name, line number and function name of
    # the first caller outside the logging module, or placeholders (unknown file, line zero,
    # unknown function) if none is found.
    code, frame = _get_caller(4)
    if code:
      return (code.co_filename, frame.f_lineno, code.co_name)
    else:
      return '(unknown file)', 0, '(unknown function)'


@tf_export('get_logger')
def get_logger():
  # [FUNCTION DOC]
  # Returns the process-wide TensorFlow logger, creating it on first use with double-checked
  # locking so concurrent callers create it only once. Creation: gets the standard logger named
  # tensorflow, replaces its caller-finder with the module's version, and, only if the root
  # logger has no handlers, determines whether the session is interactive (interactive prompt
  # attribute or interpreter interactive flag); if so sets the level to info and logs to standard
  # output, otherwise logs to standard error; then attaches a stream handler using the basic
  # format. Stores it in module state. No inputs. Side effects: configures logging global state.
  # KNOWN ISSUE: the docstring example sets the level using an unqualified level name that is not
  # available to users of the public logger function.
  """Return TF logger instance.

  Returns:
    An instance of the Python logging library Logger.

  See Python documentation (https://docs.python.org/3/library/logging.html)
  for detailed API. Below is only a summary.

  The logger has 5 levels of logging from the most serious to the least:

  1. FATAL
  2. ERROR
  3. WARN
  4. INFO
  5. DEBUG

  The logger has the following methods, based on these logging levels:

  1. fatal(msg, *args, **kwargs)
  2. error(msg, *args, **kwargs)
  3. warn(msg, *args, **kwargs)
  4. info(msg, *args, **kwargs)
  5. debug(msg, *args, **kwargs)

  The `msg` can contain string formatting.  An example of logging at the `ERROR`
  level
  using string formatting is:

  >>> tf.get_logger().error("The value %d is invalid.", 3)

  You can also specify the logging verbosity.  In this case, the
  WARN level log will not be emitted:

  >>> tf.get_logger().setLevel(ERROR)
  >>> tf.get_logger().warn("This is a warning.")
  """
  global _logger

  # Use double-checked locking to avoid taking lock unnecessarily.
  if _logger:
    return _logger

  _logger_lock.acquire()

  try:
    if _logger:
      return _logger

    # Scope the TensorFlow logger to not conflict with users' loggers.
    logger = _logging.getLogger('tensorflow')

    # Override findCaller on the logger to skip internal helper functions
    logger.findCaller = _logger_find_caller

    # Don't further configure the TensorFlow logger if the root logger is
    # already configured. This prevents double logging in those cases.
    if not _logging.getLogger().handlers:
      # Determine whether we are in an interactive environment
      _interactive = False
      try:
        # This is only defined in interactive shells.
        if _sys.ps1:
          _interactive = True
      except AttributeError:
        # Even now, we may be in an interactive shell with `python -i`.
        _interactive = _sys.flags.interactive

      # If we are in an interactive environment (like Jupyter), set loglevel
      # to INFO and pipe the output to stdout.
      if _interactive:
        logger.setLevel(INFO)
        _logging_target = _sys.stdout
      else:
        _logging_target = _sys.stderr

      # Add the output handler.
      _handler = _logging.StreamHandler(_logging_target)
      _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
      logger.addHandler(_handler)

    _logger = logger
    return _logger

  finally:
    _logger_lock.release()


@tf_export(v1=['logging.log'])
def log(level, msg, *args, **kwargs):
  # [FUNCTION DOC]
  # v1 logging function. Logs a message at the given numeric level through the TensorFlow logger;
  # msg is a format string substituted with the extra arguments. Keyword arguments pass through.
  # Returns nothing.
  get_logger().log(level, msg, *args, **kwargs)


@tf_export(v1=['logging.debug'])
def debug(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # v1 logging function: logs at debug level through the TensorFlow logger, with format-string
  # substitution of extra arguments. Returns nothing.
  get_logger().debug(msg, *args, **kwargs)


@tf_export(v1=['logging.error'])
def error(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # v1 logging function: logs at error level through the TensorFlow logger, with format-string
  # substitution of extra arguments. Returns nothing.
  get_logger().error(msg, *args, **kwargs)


@tf_export(v1=['logging.fatal'])
def fatal(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: despite the name, this only logs at critical level; it does not terminate the
  # process.
  # v1 logging function: logs at fatal (critical) level through the TensorFlow logger, with
  # format-string substitution of extra arguments. Returns nothing.
  get_logger().fatal(msg, *args, **kwargs)


@tf_export(v1=['logging.info'])
def info(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # v1 logging function: logs at info level through the TensorFlow logger, with format-string
  # substitution of extra arguments. Returns nothing.
  get_logger().info(msg, *args, **kwargs)


@tf_export(v1=['logging.warn'])
def warn(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # v1 logging function (legacy spelling): logs at warning level through the TensorFlow logger,
  # using the non-deprecated warning method internally. Returns nothing.
  get_logger().warning(msg, *args, **kwargs)


@tf_export(v1=['logging.warning'])
def warning(msg, *args, **kwargs):
  # [FUNCTION DOC]
  # v1 logging function: logs at warning level through the TensorFlow logger, with format-string
  # substitution of extra arguments. Returns nothing.
  get_logger().warning(msg, *args, **kwargs)


_level_names = {
    FATAL: 'FATAL',
    ERROR: 'ERROR',
    WARN: 'WARN',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
}

# Mask to convert integer thread ids to unsigned quantities for logging
# purposes
_THREAD_ID_MASK = 2 * _sys.maxsize + 1

_log_prefix = None  # later set to google2_log_prefix

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}


@tf_export(v1=['logging.TaskLevelStatusMessage'])
def TaskLevelStatusMessage(msg):
  # [FUNCTION DOC]
  # v1 compatibility function for task status messages. In this build it simply logs the message
  # at error level. Returns nothing.
  error(msg)


@tf_export(v1=['logging.flush'])
def flush():
  # [FUNCTION DOC]
  # KNOWN ISSUE: exported as a public v1 function but not implemented; every call fails.
  # Always raises a not-implemented error. No inputs.
  raise NotImplementedError()


# Code below is taken from pyglib/logging
@tf_export(v1=['logging.vlog'])
def vlog(level, msg, *args, **kwargs):
  # [FUNCTION DOC]
  # v1 verbose-logging function, derived from an internal logging library. Behaves identically to
  # the generic log function: logs at the given numeric level with format-string substitution.
  # Returns nothing.
  get_logger().log(level, msg, *args, **kwargs)


def _GetNextLogCountPerToken(token):
  # [FUNCTION DOC]
  # HELPER: call-site counter for the frequency-limited logging functions.
  # Increments and returns the counter stored in module state for the given token (any hashable
  # key, here a file and line pair). The first call for a token returns zero. Side effect:
  # mutates the module-level counter mapping. Not thread safe.
  """Wrapper for _log_counter_per_token.

  Args:
    token: The token for which to look up the count.

  Returns:
    The number of times this function has been called with
    *token* as an argument (starting at 0)
  """
  global _log_counter_per_token  # pylint: disable=global-variable-not-assigned
  _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
  return _log_counter_per_token[token]


@tf_export(v1=['logging.log_every_n'])
def log_every_n(level, msg, n, *args):
  # [FUNCTION DOC]
  # KNOWN ISSUE: if n is zero the modulo operation raises a zero-division error; the counter is
  # keyed on the call site found by stack inspection, so wrapping this function changes keys.
  # Logs the message at the given level on the first call from a given source location and then
  # every n-th call after it. Inputs: numeric level, format string, n (positive integer), and
  # format arguments. Side effect: increments the per-call-site counter. Returns nothing.
  """Log 'msg % args' at level 'level' once per 'n' times.

  Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
  Not threadsafe.

  Args:
    level: The level at which to log.
    msg: The message to be logged.
    n: The number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
  """
  count = _GetNextLogCountPerToken(_GetFileAndLine())
  log_if(level, msg, not (count % n), *args)


@tf_export(v1=['logging.log_first_n'])
def log_first_n(level, msg, n, *args):  # pylint: disable=g-bad-name
  # [FUNCTION DOC]
  # Logs the message at the given level only for the first n calls from a given source location
  # (as identified by stack inspection). Inputs: numeric level, format string, n (integer), and
  # format arguments. Side effect: increments the per-call-site counter. Returns nothing.
  """Log 'msg % args' at level 'level' only first 'n' times.

  Not threadsafe.

  Args:
    level: The level at which to log.
    msg: The message to be logged.
    n: The number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
  """
  count = _GetNextLogCountPerToken(_GetFileAndLine())
  log_if(level, msg, count < n, *args)


@tf_export(v1=['logging.log_if'])
def log_if(level, msg, condition, *args):
  # [FUNCTION DOC]
  # Logs the message through the verbose logging function only when the condition is true.
  # Inputs: numeric level, format string, truth value, format arguments. Returns nothing.
  """Log 'msg % args' at level 'level' only if condition is fulfilled."""
  if condition:
    vlog(level, msg, *args)


def _GetFileAndLine():
  # [FUNCTION DOC]
  # HELPER: identifies the call site for counters and log prefixes.
  # Finds the first stack frame outside this file and returns a pair of its file name and current
  # line number, or the placeholder name unknown with line zero if none is found. No inputs.
  """Returns (filename, linenumber) for the stack frame."""
  code, f = _get_caller()
  if not code:
    return ('<unknown>', 0)
  return (code.co_filename, f.f_lineno)


def google2_log_prefix(level, timestamp=None, file_and_line=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the module-level prefix variable and this function are not used by the logger
  # itself (which uses the standard basic format), so the prefix is only available to callers.
  # Builds a log line prefix: a severity letter (first letter of the level's name, or I when the
  # level is unknown), month and day, hour:minute:second.microseconds, the thread id, and the
  # source file base name and line, ending with a closing bracket and space. Inputs: numeric
  # level; optional timestamp in seconds since the epoch (default now; note a zero value counts
  # as missing); optional pair of file name and line (default: found by stack inspection).
  # Returns the prefix string. No other side effects.
  """Assemble a logline prefix using the google2 format."""
  # pylint: disable=global-variable-not-assigned
  global _level_names
  # pylint: enable=global-variable-not-assigned

  # Record current time
  now = timestamp or _time.time()
  now_tuple = _time.localtime(now)
  now_microsecond = int(1e6 * (now % 1.0))

  (filename, line) = file_and_line or _GetFileAndLine()
  basename = _os.path.basename(filename)

  # Severity string
  severity = 'I'
  if level in _level_names:
    severity = _level_names[level][0]

  s = '%c%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] ' % (
      severity,
      now_tuple[1],  # month
      now_tuple[2],  # day
      now_tuple[3],  # hour
      now_tuple[4],  # min
      now_tuple[5],  # sec
      now_microsecond,
      _get_thread_id(),
      basename,
      line)

  return s


@tf_export(v1=['logging.get_verbosity'])
def get_verbosity():
  # [FUNCTION DOC]
  # Returns the effective numeric log level of the TensorFlow logger (taking parent loggers into
  # account). No inputs.
  """Return how much logging output will be produced."""
  return get_logger().getEffectiveLevel()


@tf_export(v1=['logging.set_verbosity'])
def set_verbosity(v):
  # [FUNCTION DOC]
  # Sets the TensorFlow logger's threshold to v (a numeric level or level name accepted by the
  # standard logging library); messages below it are dropped. Side effect: changes global logger
  # state. Returns nothing.
  """Sets the threshold for what messages will be logged."""
  get_logger().setLevel(v)


def _get_thread_id():
  # [FUNCTION DOC]
  # HELPER: used by the log prefix builder.
  # Returns the current thread's identifier masked to an unsigned integer. No inputs.
  """Get id of current thread, suitable for logging as an unsigned quantity."""
  thread_id = _thread.get_ident()
  return thread_id & _THREAD_ID_MASK


_log_prefix = google2_log_prefix

tf_export(v1=['logging.DEBUG']).export_constant(__name__, 'DEBUG')
tf_export(v1=['logging.ERROR']).export_constant(__name__, 'ERROR')
tf_export(v1=['logging.FATAL']).export_constant(__name__, 'FATAL')
tf_export(v1=['logging.INFO']).export_constant(__name__, 'INFO')
tf_export(v1=['logging.WARN']).export_constant(__name__, 'WARN')
