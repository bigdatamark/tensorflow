# [MODULE DOC]
# Unit tests for the tf_stack module. Exercises stack extraction through the public function:
# equality and hashing of captured frames, the last-user-frame and user-frames accessors,
# indexing with out-of-range errors, remapping locations through the native source map, and
# round-tripping stacks through the graph debug info builder and the trace loader. Runs as a
# test binary; no import-time side effects.
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
"""Tests for functions used to extract and analyze stacks."""

from tensorflow.python.platform import test
from tensorflow.python.util import tf_stack


class TFStackTest(test.TestCase):
  # [CLASS DOC]
  # Test case for stack extraction, frame comparison/hashing, frame accessors, source-map
  # remapping and debug-info round trips. No shared fixtures.

  def testFrameSummaryEquality(self):
    # [FUNCTION DOC]
    # Extracts the stack twice. Asserts the first two frames of one stack differ, a frame equals
    # itself, and the outermost frame of two separate extractions are equal.
    frames1 = tf_stack.extract_stack()
    frames2 = tf_stack.extract_stack()

    self.assertNotEqual(frames1[0], frames1[1])
    self.assertEqual(frames1[0], frames1[0])
    self.assertEqual(frames1[0], frames2[0])

  def testFrameSummaryEqualityAndHash(self):
    # [FUNCTION DOC]
    # Extracts two stacks from the same source line so they are identical. Asserts equal length,
    # frame-by-frame equality and equal hashes, equality of the whole stacks, and equal hashes of
    # their tuple forms.
    # Both defined on the same line to produce identical stacks.
    frame1, frame2 = tf_stack.extract_stack(), tf_stack.extract_stack()
    self.assertEqual(len(frame1), len(frame2))
    for f1, f2 in zip(frame1, frame2):
      self.assertEqual(f1, f2)
      self.assertEqual(hash(f1), hash(f1))
      self.assertEqual(hash(f1), hash(f2))
    self.assertEqual(frame1, frame2)
    self.assertEqual(hash(tuple(frame1)), hash(tuple(frame2)))

  def testLastUserFrame(self):
    # [FUNCTION DOC]
    # Extracts a stack and asserts the text form of its last user frame mentions this test's
    # name.
    trace = tf_stack.extract_stack()
    frame = trace.last_user_frame()
    self.assertRegex(repr(frame), 'testLastUserFrame')

  def testGetUserFrames(self):
    # [FUNCTION DOC]
    # Calls a nested function that extracts the stack and returns the list of user frames.
    # Asserts the last frame mentions the nested function and the one before it mentions this
    # test.

    def func():
      # [FUNCTION DOC]
      # HELPER: extracts the current stack and returns its user frames as a list.
      trace = tf_stack.extract_stack()  # COMMENT
      frames = list(trace.get_user_frames())
      return frames

    frames = func()  # CALLSITE

    self.assertRegex(repr(frames[-1]), 'func')
    self.assertRegex(repr(frames[-2]), 'testGetUserFrames')

  def testGetItem(self):
    # [FUNCTION DOC]
    # Builds a stack through five levels of recursion and asserts the last item mentions the
    # recursive function. Asserts indexing one before the first and at the length raises an
    # index error.
    def func(n):
      # [FUNCTION DOC]
      # HELPER: recursive function; input n counts the remaining levels. At zero returns the
      # extracted stack, otherwise returns the result of recursing with n minus one.
      if n == 0:
        return tf_stack.extract_stack()  # COMMENT
      else:
        return func(n - 1)

    trace = func(5)
    self.assertIn('func', repr(trace[-1]))

    with self.assertRaises(IndexError):
      _ = trace[-len(trace) - 1]

    with self.assertRaises(IndexError):
      _ = trace[len(trace)]

  def testSourceMap(self):
    # [FUNCTION DOC]
    # Creates a native source map and captures a stack with it. Then maps the location of the
    # first captured frame to a fake file name, line 42 and function name, captures again and
    # asserts the first frame prints as that fake location.
    source_map = tf_stack._tf_stack.PyBindSourceMap()

    def func(n):
      # [FUNCTION DOC]
      # HELPER: recursive function (input n levels remaining); at zero extracts a stack using
      # the test's source map and an empty file set, otherwise recurses with n minus one.
      if n == 0:
        return tf_stack._tf_stack.extract_stack(
            source_map, tf_stack._tf_stack.PyBindFileSet()
        )
      else:
        return func(n - 1)

    trace = func(5)
    source_map.update_to((
        (
            (trace[0].filename, trace[0].lineno),
            ("filename", 42, "function_name"),
        ),
    ))
    trace = list(func(5))
    self.assertEqual(
        str(trace[0]), 'File "filename", line 42, in function_name'
    )

  def testStackTraceBuilder(self):
    # [FUNCTION DOC]
    # Captures three stacks, accumulates them in a debug info builder under distinct function
    # and node names, builds the debug info, loads the traces back and asserts the keys are the
    # node-at-function names and that each trace's text form mentions this test file.
    stack1 = tf_stack.extract_stack()
    stack2 = tf_stack.extract_stack()
    stack3 = tf_stack.extract_stack()

    builder = tf_stack.GraphDebugInfoBuilder()
    builder.AccumulateStackTrace('func1', 'node1', stack1)
    builder.AccumulateStackTrace('func2', 'node2', stack2)
    builder.AccumulateStackTrace('func3', 'node3', stack3)
    debug_info = builder.Build()

    trace_map = tf_stack.LoadTracesFromDebugInfo(debug_info)
    self.assertSameElements(
        trace_map.keys(), ['node1@func1', 'node2@func2', 'node3@func3']
    )

    for trace in trace_map.values():
      self.assertRegex(repr(trace), 'tf_stack_test.py', trace)


if __name__ == "__main__":
  test.main()
