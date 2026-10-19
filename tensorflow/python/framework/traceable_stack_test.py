# [MODULE DOC]
# Unit tests for the traceable stack module: the wrapper that records file name and line number
# of a caller, and the stack built from such wrappers. Line-number checks are done relative to
# a placeholder helper defined just above the call under test, and the tests that depend on
# this must keep those two lines adjacent. Defines module-level constants holding this test
# file's name, derived through source inspection at import time.
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
"""Tests for tensorflow.python.framework.traceable_stack."""

from tensorflow.python.framework import test_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.platform import googletest
from tensorflow.python.util import tf_inspect as inspect

_LOCAL_OBJECT = lambda x: x
_THIS_FILENAME = inspect.getsourcefile(_LOCAL_OBJECT)


class TraceableObjectTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the wrapper class that records a source file and line: using the caller's frame,
  # honoring the offset, and handling an absurdly large offset. No shared fixtures.

  def testSetFilenameAndLineFromCallerUsesCallersStack(self):
    # [FUNCTION DOC]
    # Creates a wrapper, then captures location with the default offset. Expects the recorded
    # line to be the line right after a placeholder definition, the recorded file to be this
    # test file, and the result code to be success.
    t_obj = traceable_stack.TraceableObject(17)

    # Do not separate placeholder from the set_filename_and_line_from_caller()
    # call one line below it as it is used to calculate the latter's line
    # number.
    placeholder = lambda x: x
    result = t_obj.set_filename_and_line_from_caller()

    expected_lineno = inspect.getsourcelines(placeholder)[1] + 1
    self.assertEqual(expected_lineno, t_obj.lineno)
    self.assertEqual(_THIS_FILENAME, t_obj.filename)
    self.assertEqual(t_obj.SUCCESS, result)

  def testSetFilenameAndLineFromCallerRespectsOffset(self):
    # [FUNCTION DOC]
    # Calls the capture method with offset one from inside a nested helper, so the recorded
    # location must be that of the helper's caller (the line after a placeholder), and the
    # result code must be success.

    def call_set_filename_and_line_from_caller(t_obj):
      # [FUNCTION DOC]
      # HELPER: nested helper for the offset test, adding one extra stack frame.
      # Input: a wrapper object. Calls its capture method with offset one so the location of
      # this helper's caller is recorded, and returns the resulting code.
      # We expect to retrieve the line number from _our_ caller.
      return t_obj.set_filename_and_line_from_caller(offset=1)

    t_obj = traceable_stack.TraceableObject(None)
    # Do not separate placeholder from the
    # call_set_filename_and_line_from_caller() call one line below it as it is
    # used to calculate the latter's line number.
    placeholder = lambda x: x
    result = call_set_filename_and_line_from_caller(t_obj)

    expected_lineno = inspect.getsourcelines(placeholder)[1] + 1
    self.assertEqual(expected_lineno, t_obj.lineno)
    self.assertEqual(t_obj.SUCCESS, result)

  def testSetFilenameAndLineFromCallerHandlesRidiculousOffset(self):
    # [FUNCTION DOC]
    # Captures location with an offset of 300, far beyond the real stack depth. Expects no
    # failure and that the result code is the heuristic-used code. File and line values are
    # deliberately not asserted because they depend on the test runner.
    t_obj = traceable_stack.TraceableObject('The quick brown fox.')
    # This line shouldn't die.
    result = t_obj.set_filename_and_line_from_caller(offset=300)

    # We expect a heuristic to be used because we are not currently 300 frames
    # down on the stack.  The filename and lineno of the outermost frame are not
    # predictable -- in some environments the filename is this test file, but in
    # other environments it is not (e.g. due to a test runner calling this
    # file).  Therefore we only test that the called function knows it applied a
    # heuristic for the ridiculous stack offset.
    self.assertEqual(t_obj.HEURISTIC_USED, result)


class TraceableStackTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the stack class: push, peek and pop ordering, and the file/line information
  # recorded on push (including with an offset). Each test builds its own stack.

  def testPushPeekPopObj(self):
    # [FUNCTION DOC]
    # Pushes a number and a string, expects the object iterator to list them newest first, then
    # pops twice and expects the string and then the number back.
    t_stack = traceable_stack.TraceableStack()
    t_stack.push_obj(42.0)
    t_stack.push_obj('hope')

    expected_lifo_peek = ['hope', 42.0]
    self.assertEqual(expected_lifo_peek, list(t_stack.peek_objs()))

    self.assertEqual('hope', t_stack.pop_obj())
    self.assertEqual(42.0, t_stack.pop_obj())

  def testPushPeekTopObj(self):
    # [FUNCTION DOC]
    # Pushes a number and then a string and expects peeking at the top object to return the
    # string (the latest push).
    t_stack = traceable_stack.TraceableStack()
    t_stack.push_obj(42.0)
    t_stack.push_obj('hope')
    self.assertEqual('hope', t_stack.peek_top_obj())

  def testPushPopPreserveLifoOrdering(self):
    # [FUNCTION DOC]
    # Pushes the integers zero through three, pops four times, and expects them back in reverse
    # order (three, two, one, zero), confirming last-in first-out behavior.
    t_stack = traceable_stack.TraceableStack()
    t_stack.push_obj(0)
    t_stack.push_obj(1)
    t_stack.push_obj(2)
    t_stack.push_obj(3)

    obj_3 = t_stack.pop_obj()
    obj_2 = t_stack.pop_obj()
    obj_1 = t_stack.pop_obj()
    obj_0 = t_stack.pop_obj()

    self.assertEqual(3, obj_3)
    self.assertEqual(2, obj_2)
    self.assertEqual(1, obj_1)
    self.assertEqual(0, obj_0)

  def testPushObjSetsFilenameAndLineInfoForCaller(self):
    # [FUNCTION DOC]
    # Pushes one object directly and another through a nested helper using offset one, then
    # reads the wrappers newest first. Expects the first object's line to be the direct push
    # line and the second object's line to be the line where the helper was invoked, each
    # computed as the line after a placeholder definition.
    t_stack = traceable_stack.TraceableStack()

    # We expect that the line number recorded for the 1-object will come from
    # the call to t_stack.push_obj(1).  Do not separate the next two lines!
    placeholder_1 = lambda x: x
    t_stack.push_obj(1)

    # We expect that the line number recorded for the 2-object will come from
    # the call to call_push_obj() and _not_ the call to t_stack.push_obj().
    def call_push_obj(obj):
      # [FUNCTION DOC]
      # HELPER: nested helper adding one stack frame for the offset push test.
      # Input: an object. Pushes it on the enclosing stack with offset one so the recorded line
      # is that of this helper's caller. Returns nothing.
      t_stack.push_obj(obj, offset=1)

    # Do not separate the next two lines!
    placeholder_2 = lambda x: x
    call_push_obj(2)

    expected_lineno_1 = inspect.getsourcelines(placeholder_1)[1] + 1
    expected_lineno_2 = inspect.getsourcelines(placeholder_2)[1] + 1

    t_obj_2, t_obj_1 = t_stack.peek_traceable_objs()
    self.assertEqual(expected_lineno_2, t_obj_2.lineno)
    self.assertEqual(expected_lineno_1, t_obj_1.lineno)


if __name__ == '__main__':
  googletest.main()
