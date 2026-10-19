# [MODULE DOC]
# Core support for TensorFlow's eager mode. Defines the internal exception types used between
# the native eager layer and Python (a wrapper for a non-OK native status, a signal for falling
# back from the fast path to the slow path, and one for symbolic tensors used during eager
# execution) plus a helper that converts a native status into the matching framework error.
# Import-time side effects: registers the not-OK-status exception class and the fallback
# exception class with the native layer so it can raise them. Holds a module-level slot for an
# active trace, initially empty.

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
"""Experimental API for TensorFlow's "Eager" mode of execution."""

from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging

# Trace of execution and memory usage.
_active_trace = None


def _status_to_exception(status):
  # [FUNCTION DOC]
  # HELPER: turns a native non-OK status into a framework error for the eager execution paths.
  # Input: status, an object with a numeric code, message and payloads. Returns an error instance
  # of the class matching the code; if the code has no known class, returns an unknown-error
  # instance carrying the code. Side effect: logs the resulting error through the error logger.
  # It returns the error rather than raising it; callers raise it.
  try:
    error_class = errors.exception_type_from_error_code(status.code)
    e = error_class(None, None, status.message, status.payloads)
    logging.error_log("%s: %s" % (e.__class__.__name__, e))
    return e
  except KeyError:
    e = errors.UnknownError(
        None, None, status.message, status.code, status.payloads
    )
    logging.error_log("%s: %s" % (e.__class__.__name__, e))
    return e


class _NotOkStatusException(Exception):
  # [CLASS DOC]
  # HELPER: internal exception the native layer raises for a non-OK status; registered with the
  # native layer at import. Holds the message, numeric status code and payloads. Eager execution
  # code catches it and converts it to a public framework error.
  """Exception class to handle not ok Status."""

  def __init__(self, message, code, payloads):
    # [FUNCTION DOC]
    # Constructor. Inputs: message (string), code (integer status code), payloads (mapping of
    # status payloads). Initializes the base exception without arguments and stores the three
    # values as attributes. Returns nothing.
    super(_NotOkStatusException, self).__init__()
    self.message = message
    self.code = code
    self.payloads = payloads

  def __str__(self):
    # [FUNCTION DOC]
    # Builds the text form by converting this exception into the matching framework error (which
    # also logs it) and returning its class name and message separated by a colon.
    e = _status_to_exception(self)
    return "%s: %s" % (e.__class__.__name__, e)


pywrap_tfe.TFE_Py_RegisterExceptionClass(_NotOkStatusException)


class _FallbackException(Exception):
  # [CLASS DOC]
  # HELPER: internal signal exception, registered with the native layer at import, raised when
  # the fast path for executing an op cannot be used so the slower, more complete path runs.
  # Carries no state of its own.
  """Exception class to handle fallback from the fastpath.

  The fastpath that we refer to here is the one implemented to reduce per-op
  overheads (TFE_Py_FastPathExecute_C). If the conditions for executing the op
  on the fastpath are not met, we fallback to a safer (and more complete)
  slowpath, and this Exception is raised to signal that transition.
  """
  pass


class _SymbolicException(Exception):
  # [CLASS DOC]
  # HELPER: internal exception raised when a symbolic (Keras graph) tensor is passed to an op
  # executed eagerly, so generated op functions can fall back to building graphs. Carries no
  # state of its own. Not registered with the native layer.
  """Exception class to handle use of symbolic tensors when executing eagerly.

  `keras.Input()` creates symbolic tensors (in a FuncGraph managed by the
  Keras backend) while in eager execution. This exception is used to
  identify this case (raised in `convert_to_tensor` cause generated functions
  for ops to construct graphs instead of executing the kernel).
  """
  pass


pywrap_tfe.TFE_Py_RegisterFallbackExceptionClass(_FallbackException)
