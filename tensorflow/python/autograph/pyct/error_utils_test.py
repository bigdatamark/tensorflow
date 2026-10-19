# [MODULE DOC]
# Unit tests for the error_utils module: building replacement exceptions from error metadata and
# formatting the translated, user-readable stack trace message (plain frames, converted frames
# mapped through a source map, and allowlisted call-overload frames).
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
"""Tests for error_utils module."""

import re

from tensorflow.python.autograph.pyct import error_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.platform import test


class ErrorMetadataBaseTest(test.TestCase):
  # [CLASS DOC]
  # Tests for ErrorMetadataBase: exception re-creation and message formatting. No shared
  # fixtures; each test constructs its own metadata object with an empty or small traceback.

  def test_create_exception_default_constructor(self):
    # [FUNCTION DOC]
    # Scenario: a custom exception type using the default constructor. Creates metadata with an
    # empty traceback and message "test message", then asks it to re-create the exception.
    # Expects an instance of the same custom type whose text contains the message.

    class CustomError(Exception):
      # [CLASS DOC]
      # Test-only exception type with the default constructor and no extra behavior.
      pass

    em = error_utils.ErrorMetadataBase(
        callsite_tb=(),
        cause_metadata=None,
        cause_message='test message',
        source_map={},
        converter_filename=None)
    exc = em.create_exception(CustomError())
    self.assertIsInstance(exc, CustomError)
    self.assertIn('test message', str(exc))

  def test_create_exception_custom_constructor(self):
    # [FUNCTION DOC]
    # Scenario: a custom exception type with its own no-argument constructor. Expects the
    # metadata object to be unable to rebuild it and to return nothing/null.

    class CustomError(Exception):
      # [CLASS DOC]
      # Test-only exception type with a custom constructor that fixes its own message, so it
      # cannot be rebuilt from a message string.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor taking no arguments; passes a fixed message to the base class.
        super(CustomError, self).__init__('test_message')

    em = error_utils.ErrorMetadataBase(
        callsite_tb=(),
        cause_metadata=None,
        cause_message='test message',
        source_map={},
        converter_filename=None)
    exc = em.create_exception(CustomError())
    self.assertIsNone(exc)

  def test_get_message_no_code(self):
    # [FUNCTION DOC]
    # Scenario: two traceback entries (one without code text), an empty source map and no
    # converter file. Expects the message to list both frames with file, line and function, in
    # order, followed by the cause message.
    callsite_tb = [
        ('/path/one.py', 11, 'test_fn_1', None),
        ('/path/two.py', 171, 'test_fn_2', 'test code'),
    ]
    cause_message = 'Test message'
    em = error_utils.ErrorMetadataBase(
        callsite_tb=callsite_tb,
        cause_metadata=None,
        cause_message=cause_message,
        source_map={},
        converter_filename=None)
    self.assertRegex(
        em.get_message(),
        re.compile(('"/path/one.py", line 11, in test_fn_1.*'
                    '"/path/two.py", line 171, in test_fn_2.*'
                    'Test message'), re.DOTALL))

  def test_get_message_converted_code(self):
    # [FUNCTION DOC]
    # Scenario: three traceback entries where the second maps through the source map to original
    # code. Expects the converted frame to show the original function name with a star marker,
    # followed by the third frame and the cause message, and the first (older) frame omitted.
    callsite_tb = [
        ('/path/one.py', 11, 'test_fn_1', 'test code 1'),
        ('/path/two.py', 171, 'test_fn_2', 'test code 2'),
        ('/path/three.py', 171, 'test_fn_3', 'test code 3'),
    ]
    cause_message = 'Test message'
    em = error_utils.ErrorMetadataBase(
        callsite_tb=callsite_tb,
        cause_metadata=None,
        cause_message=cause_message,
        source_map={
            origin_info.LineLocation(filename='/path/two.py', lineno=171):
                origin_info.OriginInfo(
                    loc=origin_info.LineLocation(
                        filename='/path/other_two.py', lineno=13),
                    function_name='converted_fn',
                    source_code_line='converted test code',
                    comment=None)
        },
        converter_filename=None)
    result = em.get_message()
    self.assertRegex(
        result,
        re.compile((r'converted_fn  \*.*'
                    r'"/path/three.py", line 171, in test_fn_3.*'
                    r'Test message'), re.DOTALL))
    self.assertNotRegex(result, re.compile('test_fn_1'))

  def test_get_message_call_overload(self):
    # [FUNCTION DOC]
    # Scenario: three traceback entries where the middle one belongs to the converter file.
    # Expects that frame to be dropped, the preceding (more recent) frame to be marked with two
    # stars (allowlisted), and the other frame and cause message to appear in order.

    callsite_tb = [
        ('/path/one.py', 11, 'test_fn_1', 'test code 1'),
        ('/path/two.py', 0, 'test_fn_2', 'test code 2'),
        ('/path/three.py', 171, 'test_fn_3', 'test code 3'),
    ]
    cause_message = 'Test message'
    em = error_utils.ErrorMetadataBase(
        callsite_tb=callsite_tb,
        cause_metadata=None,
        cause_message=cause_message,
        source_map={},
        converter_filename='/path/two.py')
    self.assertRegex(
        em.get_message(),
        re.compile((r'"/path/one.py", line 11, in test_fn_1.*'
                    r'"/path/three.py", line 171, in test_fn_3  \*\*.*'
                    r'Test message'), re.DOTALL))


if __name__ == '__main__':
  test.main()
