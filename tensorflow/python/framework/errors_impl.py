# [MODULE DOC]
# Implementation of TensorFlow's exception hierarchy. Defines the base operation error class, one
# subclass per canonical status code (cancelled, unknown, invalid argument, deadline exceeded, not
# found, already exists, permission denied, unauthenticated, resource exhausted, failed precondition,
# aborted, out of range, unimplemented, internal, unavailable, data loss), module-level integer
# constants for those codes (exported to the public errors namespace), lookup tables between codes and
# exception classes, conversion helpers, and a context manager that raises the matching exception when
# a C API status is not OK. The public errors module re-exports everything here.
# Import-time side effects: registers the code-to-exception mapping with the native exception
# registry so the C++ layer can raise these Python types, and registers API exports.
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
"""Exception types for TensorFlow errors."""

import traceback
import warnings

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import _pywrap_py_exception_registry
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


class InaccessibleTensorError(ValueError):
  # [CLASS DOC]
  # Value error raised when a tensor created inside a graph or function scope is used outside of it.
  # Carries no extra state or behavior; it exists as a distinct catchable type. Not exported publicly.
  pass


@tf_export("errors.OperatorNotAllowedInGraphError", v1=[])
class OperatorNotAllowedInGraphError(TypeError):
  # [CLASS DOC]
  # Type error raised when an operator that needs a concrete Python value, such as using a tensor as a
  # boolean or iterating over its elements, is applied inside graph execution. Carries no extra state
  # or behavior; exported publicly as part of the errors API.
  """Raised when an unsupported operator is present in Graph execution.

  For example, using a `tf.Tensor` as a Python `bool` inside a Graph will
  raise `OperatorNotAllowedInGraphError`. Iterating over values inside a
  `tf.Tensor` is also not supported in Graph execution.

  Example:
  >>> @tf.function
  ... def iterate_over(t):
  ...   a,b,c = t
  ...   return a
  >>>
  >>> iterate_over(tf.constant([1, 2, 3]))
  Traceback (most recent call last):
  ...
  OperatorNotAllowedInGraphError: ...

  """
  pass


@tf_export("errors.OpError", v1=["errors.OpError", "OpError"])
@deprecation.deprecated_endpoints("OpError")
class OpError(Exception):
  # [CLASS DOC]
  # Base class of all errors raised by TensorFlow operations. Holds the node definition proto of the
  # failed op, the failed operation object (absent in eager mode), a message, an integer status code and
  # an optional mapping of experimental payload details. Exposes them through read-only properties and
  # renders a message plus the original stack trace of the op in its string form. Subclasses fix the
  # status code and accept fewer constructor arguments; pickling support rebuilds an instance from its
  # constructor argument names. Exported publicly, also under a deprecated alias.
  """The base class for TensorFlow exceptions.

  Usually, TensorFlow will raise a more specific subclass of `OpError` from the
  `tf.errors` module.
  """

  def __init__(self, node_def, op, message, error_code, *args):
    # [FUNCTION DOC]
    # Constructor of the base error. Inputs: node definition proto of the failed op (or nothing/null),
    # the failed operation (or nothing/null; always null in eager mode), the message text, the integer
    # status code, and optional extra positional arguments of which only the first is used, as a mapping
    # of payload details (default: empty mapping). Stores them on the instance; raises nothing.
    """Creates a new `OpError` indicating that a particular op failed.

    Args:
      node_def: The `node_def_pb2.NodeDef` proto representing the op that
        failed, if known; otherwise None.
      op: The `ops.Operation` that failed, if known; otherwise None. During
        eager execution, this field is always `None`.
      message: The message string describing the failure.
      error_code: The `error_codes_pb2.Code` describing the error.
      *args: If not empty, it should contain a dictionary describing details
        about the error. This argument is inspired by Abseil payloads:
        https://github.com/abseil/abseil-cpp/blob/master/absl/status/status.h
    """
    super(OpError, self).__init__()
    self._node_def = node_def
    self._op = op
    self._message = message
    self._error_code = error_code
    if args:
      self._experimental_payloads = args[0]
    else:
      self._experimental_payloads = {}

  def __reduce__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: only the constructor arguments named in the signature are saved, so the experimental
    # payloads (taken from extra positional arguments) are not preserved when an error is pickled or copied.
    # Supports pickling/copying. Inspects the constructor signature of the actual class, reads the
    # attributes of the same names from the instance (skipping the first), and returns the class together
    # with that tuple of values so subclasses with shorter constructors can be rebuilt.
    # Allow the subclasses to accept less arguments in their __init__.
    init_argspec = tf_inspect.getargspec(self.__class__.__init__)
    args = tuple(getattr(self, arg) for arg in init_argspec.args[1:])
    return self.__class__, args

  @property
  def message(self):
    # [FUNCTION DOC]
    # Read-only property returning the message string describing the error.
    """The error message that describes the error."""
    return self._message

  @property
  def op(self):
    # [FUNCTION DOC]
    # Read-only property returning the operation that failed, or nothing/null when unknown (for example in
    # eager mode or for ops synthesized at runtime such as send and receive).
    """The operation that failed, if known.

    *N.B.* If the failed op was synthesized at runtime, e.g. a `Send`
    or `Recv` op, there will be no corresponding
    `tf.Operation`
    object.  In that case, this will return `None`, and you should
    instead use the `tf.errors.OpError.node_def` to
    discover information about the op.

    Returns:
      The `Operation` that failed, or None.
    """
    return self._op

  @property
  def error_code(self):
    # [FUNCTION DOC]
    # Read-only property returning the integer status code of the error.
    """The integer error code that describes the error."""
    return self._error_code

  @property
  def node_def(self):
    # [FUNCTION DOC]
    # Read-only property returning the node definition proto of the failed op (may be nothing/null).
    """The `NodeDef` proto representing the op that failed."""
    return self._node_def

  @property
  def experimental_payloads(self):
    # [FUNCTION DOC]
    # Read-only property returning the mapping of experimental payload details (empty if none were given).
    """A dictionary describing the details of the error."""
    return self._experimental_payloads

  def __str__(self):
    # [FUNCTION DOC]
    # String form of the error. With no associated op, returns just the message. With an op, returns the
    # message followed by the formatted original stack trace of that op, then, for each earlier op it
    # was copied or derived from, a line saying which op it was originally created as plus that op's
    # stack trace. When consecutive traces share a long common run of identical lines, the repeated run is
    # collapsed and replaced by a note giving the number of elided identical lines. Pure; no side effects.
    if self._op is not None:
      output = [
          "%s\n\nOriginal stack trace for %r:\n" % (
              self.message,
              self._op.name,
          )
      ]
      curr_traceback_list = traceback.format_list(self._op.traceback or [])
      output.extend(curr_traceback_list)
      # pylint: disable=protected-access
      original_op = self._op._original_op
      # pylint: enable=protected-access
      while original_op is not None:
        output.append(
            "\n...which was originally created as op %r, defined at:\n" %
            (original_op.name,))
        prev_traceback_list = curr_traceback_list
        curr_traceback_list = traceback.format_list(original_op.traceback or [])

        # Attempt to elide large common subsequences of the subsequent
        # stack traces.
        #
        # TODO(mrry): Consider computing the actual longest common subsequence.
        is_eliding = False
        elide_count = 0
        last_elided_line = None
        for line, line_in_prev in zip(curr_traceback_list, prev_traceback_list):
          if line == line_in_prev:
            if is_eliding:
              elide_count += 1
              last_elided_line = line
            else:
              output.append(line)
              is_eliding = True
              elide_count = 0
          else:
            if is_eliding:
              if elide_count > 0:
                output.extend([
                    "[elided %d identical lines from previous traceback]\n" %
                    (elide_count - 1,), last_elided_line
                ])
              is_eliding = False
            output.extend(line)

        # pylint: disable=protected-access
        original_op = original_op._original_op
        # pylint: enable=protected-access
      return "".join(output)
    else:
      return self.message


OK = error_codes_pb2.OK
tf_export("errors.OK").export_constant(__name__, "OK")
CANCELLED = error_codes_pb2.CANCELLED
tf_export("errors.CANCELLED").export_constant(__name__, "CANCELLED")
UNKNOWN = error_codes_pb2.UNKNOWN
tf_export("errors.UNKNOWN").export_constant(__name__, "UNKNOWN")
INVALID_ARGUMENT = error_codes_pb2.INVALID_ARGUMENT
tf_export("errors.INVALID_ARGUMENT").export_constant(__name__,
                                                     "INVALID_ARGUMENT")
DEADLINE_EXCEEDED = error_codes_pb2.DEADLINE_EXCEEDED
tf_export("errors.DEADLINE_EXCEEDED").export_constant(__name__,
                                                      "DEADLINE_EXCEEDED")
NOT_FOUND = error_codes_pb2.NOT_FOUND
tf_export("errors.NOT_FOUND").export_constant(__name__, "NOT_FOUND")
ALREADY_EXISTS = error_codes_pb2.ALREADY_EXISTS
tf_export("errors.ALREADY_EXISTS").export_constant(__name__, "ALREADY_EXISTS")
PERMISSION_DENIED = error_codes_pb2.PERMISSION_DENIED
tf_export("errors.PERMISSION_DENIED").export_constant(__name__,
                                                      "PERMISSION_DENIED")
UNAUTHENTICATED = error_codes_pb2.UNAUTHENTICATED
tf_export("errors.UNAUTHENTICATED").export_constant(__name__, "UNAUTHENTICATED")
RESOURCE_EXHAUSTED = error_codes_pb2.RESOURCE_EXHAUSTED
tf_export("errors.RESOURCE_EXHAUSTED").export_constant(__name__,
                                                       "RESOURCE_EXHAUSTED")
FAILED_PRECONDITION = error_codes_pb2.FAILED_PRECONDITION
tf_export("errors.FAILED_PRECONDITION").export_constant(__name__,
                                                        "FAILED_PRECONDITION")
ABORTED = error_codes_pb2.ABORTED
tf_export("errors.ABORTED").export_constant(__name__, "ABORTED")
OUT_OF_RANGE = error_codes_pb2.OUT_OF_RANGE
tf_export("errors.OUT_OF_RANGE").export_constant(__name__, "OUT_OF_RANGE")
UNIMPLEMENTED = error_codes_pb2.UNIMPLEMENTED
tf_export("errors.UNIMPLEMENTED").export_constant(__name__, "UNIMPLEMENTED")
INTERNAL = error_codes_pb2.INTERNAL
tf_export("errors.INTERNAL").export_constant(__name__, "INTERNAL")
UNAVAILABLE = error_codes_pb2.UNAVAILABLE
tf_export("errors.UNAVAILABLE").export_constant(__name__, "UNAVAILABLE")
DATA_LOSS = error_codes_pb2.DATA_LOSS
tf_export("errors.DATA_LOSS").export_constant(__name__, "DATA_LOSS")


@tf_export("errors.CancelledError")
class CancelledError(OpError):
  # [CLASS DOC]
  # Exception type for the CANCELLED status code; a thin subclass of the base operation error. Raised when an operation, such as a queue enqueue or a function call, is cancelled.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when an operation is cancelled.

  For example, a long-running operation e.g.`tf.queue.QueueBase.enqueue`, or a
  `tf.function` call may be cancelled by either running another operation e.g.
  `tf.queue.QueueBase.close` or a remote worker failure.

  This long-running operation will fail by raising `CancelledError`.

  Example:
  >>> q = tf.queue.FIFOQueue(10, tf.float32, ((),))
  >>> q.enqueue((10.0,))
  >>> q.close()
  >>> q.enqueue((10.0,))
  Traceback (most recent call last):
    ...
  CancelledError: ...

  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the CANCELLED code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates a `CancelledError`."""
    super(CancelledError, self).__init__(node_def, op, message, CANCELLED,
                                         *args)


@tf_export("errors.UnknownError")
class UnknownError(OpError):
  # [CLASS DOC]
  # Exception type for the UNKNOWN status code; a thin subclass of the base operation error. Used for statuses from unknown error spaces or APIs lacking error detail.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Unknown error.

  An example of where this error may be returned is if a Status value
  received from another address space belongs to an error-space that
  is not known to this address space. Also, errors raised by APIs that
  do not return enough error information may be converted to this
  error.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the UNKNOWN code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `UnknownError`."""
    super(UnknownError, self).__init__(node_def, op, message, UNKNOWN, *args)


@tf_export("errors.InvalidArgumentError")
class InvalidArgumentError(OpError):
  # [CLASS DOC]
  # Exception type for the INVALID_ARGUMENT status code; a thin subclass of the base operation error. Raised when an operation receives mismatched or invalid arguments.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when an operation receives an invalid argument.

  This error is typically raised when an op receives mismatched arguments.

  Example:

  >>> tf.reshape([1, 2, 3], (2,))
  Traceback (most recent call last):
     ...
  InvalidArgumentError: ...
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the INVALID_ARGUMENT code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `InvalidArgumentError`."""
    super(InvalidArgumentError, self).__init__(node_def, op, message,
                                               INVALID_ARGUMENT, *args)


@tf_export("errors.DeadlineExceededError")
class DeadlineExceededError(OpError):
  # [CLASS DOC]
  # Exception type for the DEADLINE_EXCEEDED status code; a thin subclass of the base operation error. Documented as not currently used.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when a deadline expires before an operation could complete.

  This exception is not currently used.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the DEADLINE_EXCEEDED code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates a `DeadlineExceededError`."""
    super(DeadlineExceededError, self).__init__(node_def, op, message,
                                                DEADLINE_EXCEEDED, *args)


@tf_export("errors.NotFoundError")
class NotFoundError(OpError):
  # [CLASS DOC]
  # Exception type for the NOT_FOUND status code; a thin subclass of the base operation error. Raised when a requested entity such as a file is missing.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when a requested entity (e.g., a file or directory) was not found.

  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `NotFoundError` if it receives the name of a file that
  does not exist.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the NOT_FOUND code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates a `NotFoundError`."""
    super(NotFoundError, self).__init__(node_def, op, message, NOT_FOUND, *args)


@tf_export("errors.AlreadyExistsError")
class AlreadyExistsError(OpError):
  # [CLASS DOC]
  # Exception type for the ALREADY_EXISTS status code; a thin subclass of the base operation error. Raised when creating an entity that already exists.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when an entity that we attempted to create already exists.

  An API raises this this error to avoid overwriting an existing resource,
  value, etc. Calling a creation API multiple times with the same arguments
  could raise this error if the creation API is not idempotent.

  For example, running an operation that saves a file
  (e.g. `tf.saved_model.save`)
  could potentially raise this exception if an explicit filename for an
  existing file was passed.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the ALREADY_EXISTS code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `AlreadyExistsError`."""
    super(AlreadyExistsError, self).__init__(node_def, op, message,
                                             ALREADY_EXISTS, *args)


@tf_export("errors.PermissionDeniedError")
class PermissionDeniedError(OpError):
  # [CLASS DOC]
  # Exception type for the PERMISSION_DENIED status code; a thin subclass of the base operation error. Raised when the caller lacks permission for an operation.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when the caller does not have permission to run an operation.

  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `PermissionDeniedError` if it receives the name of a
  file for which the user does not have the read file permission.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the PERMISSION_DENIED code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates a `PermissionDeniedError`."""
    super(PermissionDeniedError, self).__init__(node_def, op, message,
                                                PERMISSION_DENIED, *args)


@tf_export("errors.UnauthenticatedError")
class UnauthenticatedError(OpError):
  # [CLASS DOC]
  # Exception type for the UNAUTHENTICATED status code; a thin subclass of the base operation error. Documented as not currently used.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when the request does not have valid authentication credentials.

  This exception is not currently used.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the UNAUTHENTICATED code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `UnauthenticatedError`."""
    super(UnauthenticatedError, self).__init__(node_def, op, message,
                                               UNAUTHENTICATED, *args)


@tf_export("errors.ResourceExhaustedError")
class ResourceExhaustedError(OpError):
  # [CLASS DOC]
  # Exception type for the RESOURCE_EXHAUSTED status code; a thin subclass of the base operation error. Raised when a resource such as memory, quota or disk space is exhausted.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when some resource has been exhausted while running operation.

  For example, this error might be raised if a per-user quota is
  exhausted, or perhaps the entire file system is out of space. If running into
  `ResourceExhaustedError` due to out of memory (OOM), try to use smaller batch
  size or reduce dimension size of model weights.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the RESOURCE_EXHAUSTED code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates a `ResourceExhaustedError`."""
    super(ResourceExhaustedError, self).__init__(node_def, op, message,
                                                 RESOURCE_EXHAUSTED, *args)


@tf_export("errors.FailedPreconditionError")
class FailedPreconditionError(OpError):
  # [CLASS DOC]
  # Exception type for the FAILED_PRECONDITION status code; a thin subclass of the base operation error. Raised when preconditions are unmet, e.g. reading an uninitialized variable.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when some prerequisites are not met when running an operation.

  This typically indicates that system is not in state to execute the operation
  and requires preconditions to be met before successfully executing current
  operation.

  For example, this exception is commonly raised when running an operation
  that reads a `tf.Variable` before it has been initialized.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the FAILED_PRECONDITION code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates a `FailedPreconditionError`."""
    super(FailedPreconditionError, self).__init__(node_def, op, message,
                                                  FAILED_PRECONDITION, *args)


@tf_export("errors.AbortedError")
class AbortedError(OpError):
  # [CLASS DOC]
  # Exception type for the ABORTED status code; a thin subclass of the base operation error. Raised when an operation is aborted, typically by a concurrent action.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when an operation was aborted, typically due to a concurrent action.

  For example, running a
  `tf.queue.QueueBase.enqueue`
  operation may raise `AbortedError` if a
  `tf.queue.QueueBase.close` operation
  previously ran.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the ABORTED code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `AbortedError`."""
    super(AbortedError, self).__init__(node_def, op, message, ABORTED, *args)


@tf_export("errors.OutOfRangeError")
class OutOfRangeError(OpError):
  # [CLASS DOC]
  # Exception type for the OUT_OF_RANGE status code; a thin subclass of the base operation error. Raised when an operation iterates past a valid range, e.g. the end of a dataset.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when an operation iterates past the valid range.

  Unlike `InvalidArgumentError`, this error indicates a problem may be fixed if
  the system state changes. For example, if a list grows and the operation is
  now within the valid range. `OutOfRangeError` overlaps with
  `FailedPreconditionError` and should be preferred as the more specific error
  when iterating or accessing a range.

  For example, iterating a TF dataset past the last item in the dataset will
  raise this error.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the OUT_OF_RANGE code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `OutOfRangeError`."""
    super(OutOfRangeError, self).__init__(node_def, op, message, OUT_OF_RANGE,
                                          *args)


@tf_export("errors.UnimplementedError")
class UnimplementedError(OpError):
  # [CLASS DOC]
  # Exception type for the UNIMPLEMENTED status code; a thin subclass of the base operation error. Raised when an operation or option is not implemented.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when an operation has not been implemented.

  Some operations may raise this error when passed otherwise-valid
  arguments that it does not currently support. For example, running
  the `tf.nn.max_pool2d` operation
  would raise this error if pooling was requested on the batch dimension,
  because this is not yet supported.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the UNIMPLEMENTED code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `UnimplementedError`."""
    super(UnimplementedError, self).__init__(node_def, op, message,
                                             UNIMPLEMENTED, *args)


@tf_export("errors.InternalError")
class InternalError(OpError):
  # [CLASS DOC]
  # Exception type for the INTERNAL status code; a thin subclass of the base operation error. Raised when a runtime invariant has been broken.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when the system experiences an internal error.

  This exception is raised when some invariant expected by the runtime
  has been broken. Catching this exception is not recommended.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the INTERNAL code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `InternalError`."""
    super(InternalError, self).__init__(node_def, op, message, INTERNAL, *args)


@tf_export("errors.UnavailableError")
class UnavailableError(OpError):
  # [CLASS DOC]
  # Exception type for the UNAVAILABLE status code; a thin subclass of the base operation error. Documented as not currently used.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when the runtime is currently unavailable.

  This exception is not currently used.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the UNAVAILABLE code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates an `UnavailableError`."""
    super(UnavailableError, self).__init__(node_def, op, message, UNAVAILABLE,
                                           *args)


@tf_export("errors.DataLossError")
class DataLossError(OpError):
  # [CLASS DOC]
  # Exception type for the DATA_LOSS status code; a thin subclass of the base operation error. Raised on unrecoverable data loss or corruption such as a truncated file.
  # Instances are created by the runtime (through the code-to-class registry) or directly by callers
  # with a node definition, an operation and a message; the status code is fixed by the class.
  """Raised when unrecoverable data loss or corruption is encountered.

  This could be due to:
  * A truncated file.
  * A corrupted file.
  * Specifying the wrong data format.

  For example, this may be raised by running a
  `tf.WholeFileReader.read`
  operation, if the file is truncated while it is being read.
  """

  def __init__(self, node_def, op, message, *args):
    # [FUNCTION DOC]
    # Constructor. Takes the node definition proto of the failed op (or nothing/null), the failed operation
    # object (or nothing/null), the message text, and optionally one extra positional argument that is a
    # mapping of payload details. Forwards them to the base class with the DATA_LOSS code filled in.
    # Raises nothing itself; stores the values on the new instance.
    """Creates a `DataLossError`."""
    super(DataLossError, self).__init__(node_def, op, message, DATA_LOSS, *args)


_CODE_TO_EXCEPTION_CLASS = {
    CANCELLED: CancelledError,
    UNKNOWN: UnknownError,
    INVALID_ARGUMENT: InvalidArgumentError,
    DEADLINE_EXCEEDED: DeadlineExceededError,
    NOT_FOUND: NotFoundError,
    ALREADY_EXISTS: AlreadyExistsError,
    PERMISSION_DENIED: PermissionDeniedError,
    UNAUTHENTICATED: UnauthenticatedError,
    RESOURCE_EXHAUSTED: ResourceExhaustedError,
    FAILED_PRECONDITION: FailedPreconditionError,
    ABORTED: AbortedError,
    OUT_OF_RANGE: OutOfRangeError,
    UNIMPLEMENTED: UnimplementedError,
    INTERNAL: InternalError,
    UNAVAILABLE: UnavailableError,
    DATA_LOSS: DataLossError,
}

_pywrap_py_exception_registry.PyExceptionRegistry_Init(_CODE_TO_EXCEPTION_CLASS)

_EXCEPTION_CLASS_TO_CODE = {
    class_: code for code, class_ in _CODE_TO_EXCEPTION_CLASS.items()
}


@tf_export(v1=["errors.exception_type_from_error_code"])
def exception_type_from_error_code(error_code):
  # [FUNCTION DOC]
  # Looks up the exception class registered for an integer status code. Input: the status code.
  # Returns the exception class. Raises a key error for codes not in the table (for example the OK code).
  # Exported in the version 1 API only.
  return _CODE_TO_EXCEPTION_CLASS[error_code]


@tf_export(v1=["errors.error_code_from_exception_type"])
def error_code_from_exception_type(cls):
  # [FUNCTION DOC]
  # KNOWN ISSUE: on an unknown class it returns a newly built unknown-error exception instance (with a
  # null message payload argument) instead of an integer code, so the return type is inconsistent.
  # Looks up the status code for an exception class. Input: the exception class. Returns its integer
  # code. For classes not in the table, emits a warning and returns the instance described above.
  # Exported in the version 1 API only.
  try:
    return _EXCEPTION_CLASS_TO_CODE[cls]
  except KeyError:
    warnings.warn("Unknown class exception")
    return UnknownError(None, None, "Unknown class exception", None)


def _make_specific_exception(node_def, op, message, error_code):
  # [FUNCTION DOC]
  # HELPER: converts a status code into an exception instance for the status-checking context manager.
  # Inputs: node definition (or null), op (or null), message text, integer code. Returns an instance of
  # the exception class registered for the code, or, for an unregistered code, emits a warning and
  # returns an unknown-error instance. Does not raise; the caller raises the result.
  try:
    exc_type = exception_type_from_error_code(error_code)
    return exc_type(node_def, op, message)
  except KeyError:
    warnings.warn("Unknown error code: %d" % error_code)
    return UnknownError(node_def, op, message, error_code)


# Named like a function for backwards compatibility with the
# @tf_contextlib.contextmanager version, which was switched to a class to avoid
# some object creation overhead.
# TODO(b/77295559): expand use of TF_Status* SWIG typemap and deprecate this.
@tf_export(v1=["errors.raise_exception_on_not_ok_status"])  # pylint: disable=invalid-name
class raise_exception_on_not_ok_status(object):
  # [CLASS DOC]
  # Context manager that checks a native C API status. Entering creates a scoped status object and
  # returns the underlying status handle for the body to pass to native calls; leaving checks it.
  # Named like a function for backward compatibility with an older generator-based implementation.
  # Exported in the version 1 API only.
  """Context manager to check for C API status."""

  def __enter__(self):
    # [FUNCTION DOC]
    # Context entry. Creates a scoped status object, keeps it on the instance, and returns the raw status
    # handle to be filled in by native calls. Takes no inputs; raises nothing.
    self.status = c_api_util.ScopedTFStatus()
    return self.status.status

  def __exit__(self, type_arg, value_arg, traceback_arg):
    # [FUNCTION DOC]
    # Context exit. Inputs: the usual exception type, value and traceback of the with body (unused).
    # If the status code is not OK, raises the matching specific exception built from the status message.
    # Always deletes the stored status object afterwards so it is not kept alive by the traceback.
    # Otherwise returns false, so exceptions from the body are never suppressed.
    try:
      if c_api.TF_GetCode(self.status.status) != 0:
        raise _make_specific_exception(
            None, None, compat.as_text(c_api.TF_Message(self.status.status)),
            c_api.TF_GetCode(self.status.status))
    # Delete the underlying status object from memory otherwise it stays alive
    # as there is a reference to status from this from the traceback due to
    # raise.
    finally:
      del self.status
    return False  # False values do not suppress exceptions
