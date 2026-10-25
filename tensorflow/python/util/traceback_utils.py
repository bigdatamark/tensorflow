# [MODULE DOC]
# Utilities for prettifying exception stack traces by hiding TensorFlow-internal frames.
# Keeps a thread-local switch (default: filtering enabled) and a tuple of excluded directory
# paths (the TensorFlow python package directory). Exposes, as public TensorFlow API under the
# debugging namespace, functions to query, enable and disable the filtering; plus a frame
# inclusion test, a traceback rebuilding helper, and the decorator that re-raises exceptions of
# a wrapped function with a filtered traceback. Import-time side effects: creates the
# thread-local state and computes the excluded path from this file's location, and registers
# the three public functions for export.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities related to TensorFlow exception stack trace prettifying."""

import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export


_ENABLE_TRACEBACK_FILTERING = threading.local()
_EXCLUDED_PATHS = (
    os.path.abspath(os.path.join(__file__, '..', '..')),
)


@tf_export('debugging.is_traceback_filtering_enabled')
def is_traceback_filtering_enabled():
  # [FUNCTION DOC]
  # Reports whether traceback filtering is on for the calling thread. Takes no inputs. Reads the
  # thread-local flag, defaulting to true if the thread never set it. Returns a boolean. No side
  # effects.
  """Check whether traceback filtering is currently enabled.

  See also `tf.debugging.enable_traceback_filtering()` and
  `tf.debugging.disable_traceback_filtering()`. Note that filtering out
  internal frames from the tracebacks of exceptions raised by TensorFlow code
  is the default behavior.

  Returns:
    True if traceback filtering is enabled
    (e.g. if `tf.debugging.enable_traceback_filtering()` was called),
    and False otherwise (e.g. if `tf.debugging.disable_traceback_filtering()`
    was called).
  """
  value = getattr(_ENABLE_TRACEBACK_FILTERING, 'value', True)
  return value


@tf_export('debugging.enable_traceback_filtering')
def enable_traceback_filtering():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the minimum-version check (Python 3.7) is effectively dead code on supported
  # interpreters, and the first part of the error message is an f-string with no placeholders.
  # Turns filtering on for the calling thread only (the flag is thread-local, so other threads
  # keep their own setting). Takes no inputs, returns nothing. Raises a runtime error if the
  # interpreter is not Python 3.7 or newer.
  """Enable filtering out TensorFlow-internal frames in exception stack traces.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  If you have previously disabled traceback filtering via
  `tf.debugging.disable_traceback_filtering()`, you can re-enable it via
  `tf.debugging.enable_traceback_filtering()`.

  Raises:
    RuntimeError: If Python version is not at least 3.7.
  """
  if sys.version_info.major != 3 or sys.version_info.minor < 7:
    raise RuntimeError(
        f'Traceback filtering is only available with Python 3.7 or higher. '
        f'This Python version: {sys.version}')
  global _ENABLE_TRACEBACK_FILTERING
  _ENABLE_TRACEBACK_FILTERING.value = True


@tf_export('debugging.disable_traceback_filtering')
def disable_traceback_filtering():
  # [FUNCTION DOC]
  # Turns filtering off for the calling thread only (thread-local flag), so subsequent exceptions
  # from filtered functions carry full tracebacks including internal frames. Takes no inputs,
  # returns nothing, raises nothing.
  """Disable filtering out TensorFlow-internal frames in exception stack traces.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  Calling `tf.debugging.disable_traceback_filtering` disables this filtering
  mechanism, meaning that TensorFlow exceptions stack traces will include
  all frames, in particular TensorFlow-internal ones.

  **If you are debugging a TensorFlow-internal issue, you need to call
  `tf.debugging.disable_traceback_filtering`**.
  To re-enable traceback filtering afterwards, you can call
  `tf.debugging.enable_traceback_filtering()`.
  """
  global _ENABLE_TRACEBACK_FILTERING
  _ENABLE_TRACEBACK_FILTERING.value = False


def include_frame(fname):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the test is a substring match, not a path-prefix match, so any file name that
  # merely contains the excluded directory text anywhere is treated as internal.
  # Decides whether a frame from the given file should be kept in a filtered traceback. Input:
  # fname (string file name of the frame's code). Returns false if any excluded directory path
  # occurs inside it, true otherwise. No side effects.
  """Determines whether a frame should be included in filtered tracebacks.

  This function checks if the given filename contains any of the excluded
  paths. Frames from excluded paths (typically TensorFlow internal code) are
  filtered out from stack traces to make them more readable for end users.

  Args:
    fname: A string representing the filename of the frame to check.

  Returns:
    True if the frame should be included in the traceback (i.e., it's not
    from an excluded path), False otherwise.
  """

  for exclusion in _EXCLUDED_PATHS:
    if exclusion in fname:
      return False
  return True


def _process_traceback_frames(tb):
  # [FUNCTION DOC]
  # HELPER: used by the traceback-filtering decorator to rebuild a traceback.
  # Input: tb, a traceback object (may be the null value). Walks all frames, keeps those whose
  # code file passes the frame-inclusion test, and links new traceback entries in the original
  # order. If every frame was excluded but the traceback was not empty, keeps only the innermost
  # frame so the error location is not lost. Returns the new traceback (null if the input had
  # no frames). Does not modify the input.
  new_tb = None
  tb_list = list(traceback.walk_tb(tb))
  for f, line_no in reversed(tb_list):
    if include_frame(f.f_code.co_filename):
      new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
  if new_tb is None and tb_list:
    f, line_no = tb_list[-1]
    new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
  return new_tb


def filter_traceback(fn):
  # [FUNCTION DOC]
  # Decorator that makes exceptions escaping fn carry a traceback without TensorFlow-internal
  # frames. Input: fn, the callable to decorate. Returns fn unchanged on pythons older than 3.7;
  # otherwise returns a wrapper (created through the tf decorator maker so the target is
  # recorded). The wrapper calls fn directly when filtering is disabled for the thread;
  # otherwise calls fn and, on any ordinary exception, re-raises the same exception object with
  # a filtered traceback and with the original cause suppressed.
  """Decorator to filter out TF-internal stack trace frames in exceptions.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  Arguments:
    fn: The function or method to decorate. Any exception raised within the
      function will be reraised with its internal stack trace frames filtered
      out.

  Returns:
    Decorated function or method.
  """
  if sys.version_info.major != 3 or sys.version_info.minor < 7:
    return fn

  def error_handler(*args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when filtering is disabled the call to fn happens inside a try block that
    # catches name errors; a name error raised by fn itself would therefore cause fn to be
    # called a second time.
    # HELPER: wrapper returned by the traceback-filtering decorator. Takes any positional and
    # keyword arguments and forwards them to the decorated function, returning its result. If
    # filtering is enabled, any exception from the call is re-raised with filtered traceback
    # frames (original cause suppressed); the stored filtered traceback is deleted afterwards
    # to avoid reference cycles.
    try:
      if not is_traceback_filtering_enabled():
        return fn(*args, **kwargs)
    except NameError:
      # In some very rare cases,
      # `is_traceback_filtering_enabled` (from the outer scope) may not be
      # accessible from inside this function
      return fn(*args, **kwargs)

    filtered_tb = None
    try:
      return fn(*args, **kwargs)
    except Exception as e:
      filtered_tb = _process_traceback_frames(e.__traceback__)
      raise e.with_traceback(filtered_tb) from None
    finally:
      del filtered_tb

  return tf_decorator.make_decorator(fn, error_handler)
