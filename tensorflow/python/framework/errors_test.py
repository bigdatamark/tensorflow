# [MODULE DOC]
# Unit tests for the TensorFlow error classes: the mapping between status codes and exception
# classes, warnings for unknown codes, absence of status-object leaks, pickling round trips, and the
# experimental payloads carried from native statuses and their default value. Uses a native test
# helper extension to raise errors from C++ statuses. Running as a script starts the test runner.
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
"""Tests for tensorflow.python.framework.errors."""

import gc
import pickle
import warnings

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python.framework import _errors_test_helper
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ErrorsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the exception hierarchy in the errors modules. Has one shared helper that counts
  # live objects of a type; each test is otherwise independent.

  def _CountReferences(self, typeof):
    # [FUNCTION DOC]
    # HELPER: used by the status-leak test to count live objects.
    # Input: a class. Walks every object tracked by the garbage collector and counts those that are
    # instances of the class, ignoring objects whose weak references have died. Returns the integer count.
    """Count number of references to objects of type |typeof|."""
    objs = gc.get_objects()
    ref_count = 0
    for o in objs:
      try:
        if isinstance(o, typeof):
          ref_count += 1
      # Certain versions of python keeps a weakref to deleted objects.
      except ReferenceError:
        pass
    return ref_count

  def testUniqueClassForEachErrorCode(self):
    # [FUNCTION DOC]
    # Scenario: for each of the sixteen status codes paired with its expected exception class, builds an
    # exception from the code with the internal factory and asserts it is an instance of that class, and
    # that the code-from-class lookup returns the same code. Returns nothing.
    for error_code, exc_type in [
        (errors.CANCELLED, errors_impl.CancelledError),
        (errors.UNKNOWN, errors_impl.UnknownError),
        (errors.INVALID_ARGUMENT, errors_impl.InvalidArgumentError),
        (errors.DEADLINE_EXCEEDED, errors_impl.DeadlineExceededError),
        (errors.NOT_FOUND, errors_impl.NotFoundError),
        (errors.ALREADY_EXISTS, errors_impl.AlreadyExistsError),
        (errors.PERMISSION_DENIED, errors_impl.PermissionDeniedError),
        (errors.UNAUTHENTICATED, errors_impl.UnauthenticatedError),
        (errors.RESOURCE_EXHAUSTED, errors_impl.ResourceExhaustedError),
        (errors.FAILED_PRECONDITION, errors_impl.FailedPreconditionError),
        (errors.ABORTED, errors_impl.AbortedError),
        (errors.OUT_OF_RANGE, errors_impl.OutOfRangeError),
        (errors.UNIMPLEMENTED, errors_impl.UnimplementedError),
        (errors.INTERNAL, errors_impl.InternalError),
        (errors.UNAVAILABLE, errors_impl.UnavailableError),
        (errors.DATA_LOSS, errors_impl.DataLossError),
    ]:
      # pylint: disable=protected-access
      self.assertTrue(
          isinstance(
              errors_impl._make_specific_exception(None, None, None,
                                                   error_code), exc_type))
      # error_code_from_exception_type and exception_type_from_error_code should
      # be consistent with operation result.
      self.assertEqual(error_code,
                       errors_impl.error_code_from_exception_type(exc_type))
      # pylint: enable=protected-access

  def testKnownErrorClassForEachErrorCodeInProto(self):
    # [FUNCTION DOC]
    # Scenario: iterates every code defined in the status-code proto except the OK code and the reserved
    # placeholder. For each, builds an exception with the factory while recording warnings. Asserts no
    # warning was emitted, the result is an operation error, and that base class is a direct base of the
    # concrete class.
    for error_code in error_codes_pb2.Code.values():
      # pylint: disable=line-too-long
      if error_code in (
          error_codes_pb2.OK, error_codes_pb2.
          DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_
      ):
        continue
      # pylint: enable=line-too-long
      with warnings.catch_warnings(record=True) as w:
        # pylint: disable=protected-access
        exc = errors_impl._make_specific_exception(None, None, None, error_code)
        # pylint: enable=protected-access
      self.assertEqual(0, len(w))  # No warning is raised.
      self.assertTrue(isinstance(exc, errors_impl.OpError))
      self.assertTrue(errors_impl.OpError in exc.__class__.__bases__)

  def testUnknownErrorCodeCausesWarning(self):
    # [FUNCTION DOC]
    # Scenario: builds an exception from an unregistered code (37) and looks up the code of a non-class
    # value. Each time asserts exactly one warning is recorded with the expected text (unknown error code,
    # unknown class exception) and that the returned object is an operation error instance.
    with warnings.catch_warnings(record=True) as w:
      # pylint: disable=protected-access
      exc = errors_impl._make_specific_exception(None, None, None, 37)
      # pylint: enable=protected-access
    self.assertEqual(1, len(w))
    self.assertTrue("Unknown error code: 37" in str(w[0].message))
    self.assertTrue(isinstance(exc, errors_impl.OpError))

    with warnings.catch_warnings(record=True) as w:
      # pylint: disable=protected-access
      exc = errors_impl.error_code_from_exception_type("Unknown")
      # pylint: enable=protected-access
    self.assertEqual(1, len(w))
    self.assertTrue("Unknown class exception" in str(w[0].message))
    self.assertTrue(isinstance(exc, errors_impl.OpError))

  def testStatusDoesNotLeak(self):
    # [FUNCTION DOC]
    # Scenario: triggers a failing native file deletion of a nonexistent path (any exception swallowed),
    # forces garbage collection, then asserts no scoped status objects remain alive.
    try:
      _pywrap_file_io.DeleteFile(compat.as_bytes("/DOES_NOT_EXIST/"))
    except:
      pass
    gc.collect()
    self.assertEqual(0, self._CountReferences(c_api_util.ScopedTFStatus))

  def testPickleable(self):
    # [FUNCTION DOC]
    # Scenario: for each status code builds an exception, pickles and unpickles it, and asserts the node
    # definition, op, message and error code are equal on the copy.
    for error_code in [
        errors.CANCELLED,
        errors.UNKNOWN,
        errors.INVALID_ARGUMENT,
        errors.DEADLINE_EXCEEDED,
        errors.NOT_FOUND,
        errors.ALREADY_EXISTS,
        errors.PERMISSION_DENIED,
        errors.UNAUTHENTICATED,
        errors.RESOURCE_EXHAUSTED,
        errors.FAILED_PRECONDITION,
        errors.ABORTED,
        errors.OUT_OF_RANGE,
        errors.UNIMPLEMENTED,
        errors.INTERNAL,
        errors.UNAVAILABLE,
        errors.DATA_LOSS,
    ]:
      # pylint: disable=protected-access
      exc = errors_impl._make_specific_exception(None, None, None, error_code)
      # pylint: enable=protected-access
      unpickled = pickle.loads(pickle.dumps(exc))
      self.assertEqual(exc.node_def, unpickled.node_def)
      self.assertEqual(exc.op, unpickled.op)
      self.assertEqual(exc.message, unpickled.message)
      self.assertEqual(exc.error_code, unpickled.error_code)

  def testErrorPayloadsFromStatus(self):
    # [FUNCTION DOC]
    # Scenario: for each numeric status code with its expected exception class, uses the native test helper
    # to raise an error from a status carrying payloads. Asserts the expected class is raised and that the
    # payload mapping holds the two expected key and value pairs.
    for code, expected_exception in [
        (1, errors.CancelledError),
        (2, errors.UnknownError),
        (3, errors.InvalidArgumentError),
        (4, errors.DeadlineExceededError),
        (5, errors.NotFoundError),
        (6, errors.AlreadyExistsError),
        (7, errors.PermissionDeniedError),
        (16, errors.UnauthenticatedError),
        (8, errors.ResourceExhaustedError),
        (9, errors.FailedPreconditionError),
        (10, errors.AbortedError),
        (11, errors.OutOfRangeError),
        (12, errors.UnimplementedError),
        (13, errors.InternalError),
        (14, errors.UnavailableError),
        (15, errors.DataLossError),
    ]:
      with self.assertRaises(expected_exception) as error:
        _errors_test_helper.TestRaiseFromStatus(code)
      self.assertEqual(error.exception.experimental_payloads[b"key1"],
                       b"value1")
      self.assertEqual(error.exception.experimental_payloads[b"key2"],
                       b"value2")

  def testErrorPayloadsFromTFStatus(self):
    # [FUNCTION DOC]
    # Same scenario as the status payload test but the native helper raises from the TensorFlow status
    # wrapper type. Asserts the expected exception class and the two payload entries for every code.
    for code, expected_exception in [
        (1, errors.CancelledError),
        (2, errors.UnknownError),
        (3, errors.InvalidArgumentError),
        (4, errors.DeadlineExceededError),
        (5, errors.NotFoundError),
        (6, errors.AlreadyExistsError),
        (7, errors.PermissionDeniedError),
        (16, errors.UnauthenticatedError),
        (8, errors.ResourceExhaustedError),
        (9, errors.FailedPreconditionError),
        (10, errors.AbortedError),
        (11, errors.OutOfRangeError),
        (12, errors.UnimplementedError),
        (13, errors.InternalError),
        (14, errors.UnavailableError),
        (15, errors.DataLossError),
    ]:
      with self.assertRaises(expected_exception) as error:
        _errors_test_helper.TestRaiseFromTFStatus(code)
      self.assertEqual(error.exception.experimental_payloads[b"key1"],
                       b"value1")
      self.assertEqual(error.exception.experimental_payloads[b"key2"],
                       b"value2")

  def testErrorPayloadsDefaultValue(self):
    # [FUNCTION DOC]
    # Scenario: constructs each exception class with null node definition, op and message and no payload.
    # Asserts the payload property is a plain mapping and is empty.
    for exception_type in [
        (errors.CancelledError),
        (errors.UnknownError),
        (errors.InvalidArgumentError),
        (errors.DeadlineExceededError),
        (errors.NotFoundError),
        (errors.AlreadyExistsError),
        (errors.PermissionDeniedError),
        (errors.UnauthenticatedError),
        (errors.ResourceExhaustedError),
        (errors.FailedPreconditionError),
        (errors.AbortedError),
        (errors.OutOfRangeError),
        (errors.UnimplementedError),
        (errors.InternalError),
        (errors.UnavailableError),
        (errors.DataLossError),
    ]:
      e = exception_type(None, None, None)
      self.assertEqual(type(e.experimental_payloads), dict)
      self.assertEqual(len(e.experimental_payloads), 0)


if __name__ == "__main__":
  test.main()
