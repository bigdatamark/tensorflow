# [MODULE DOC]
# Unit tests for the error message interpolation helpers in the framework error_interpolation module.
# The tests cover device and colocation summary formatting, locating the user-defining frame in an
# operation traceback, replacing node tags in error messages with "defined at" location text, the
# traceback text produced when a graph function fails at run time, and the check that classifies file
# names as framework files. Module-level helpers build mock stack frames with rewritten file names.
# Importing the module has no side effects; running it as a script starts the TensorFlow test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import collections
import os
import re

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


# A mock for ``tf_stack.FrameSummary``.
FrameSummary = collections.namedtuple(
    "StackFrame", ["filename", "lineno", "name", "line"]
)

# TODO(feyu): convert tests to tf function from graph when appropriate.


def _make_frame_with_filename(tb, idx, filename):
  # [FUNCTION DOC]
  # HELPER: builds fake stack frames for the traceback-rewriting tests in this file.
  # Takes a traceback (sequence of frame records), an integer index, and a replacement file name.
  # Returns a new frame record copying the line number, function name and source line of the frame at
  # that index but with the file name replaced. Raises an index error if idx is out of range.
  # Does not modify the input traceback.
  """Return a copy of an existing stack frame with a new filename."""
  frame = tb[idx]
  return FrameSummary(filename, frame.lineno, frame.name, frame.line)


def _modify_op_stack_with_filenames(
    tb, num_user_frames, user_filename, num_inner_tf_frames
):
  # [FUNCTION DOC]
  # HELPER: builds a synthetic operation traceback for the defining-frame tests in this file.
  # KNOWN ISSUE: the user_filename argument is overwritten at the top of the body, so the caller value is ignored.
  # Inputs: a traceback (list of frame records, outermost first), the number of frames
  # to relabel as user frames (taken just before the framework frames), a user file name (accepted but immediately replaced by a fixed
  # per-frame user file name pattern), and the number of innermost frames to relabel as framework frames
  # using the framework path prefix. Keeps the leading frames untouched, then relabels user frames, then
  # framework frames, each file name carrying the frame index. Returns the new list of frames.
  # Fails an assertion (not an exception type of its own) when more frames are requested than exist.
  """Replace traceback with a new traceback using special filenames."""
  tf_filename = error_interpolation._FRAMEWORK_PATH_PREFIXES[0] + "%d.py"
  user_filename = os.path.join("%d", "my_favorite_file.py")

  num_requested_frames = num_user_frames + num_inner_tf_frames
  num_actual_frames = len(tb)
  num_outer_frames = num_actual_frames - num_requested_frames
  assert num_requested_frames <= num_actual_frames, "Too few real frames."

  # The op's traceback has outermost frame at index 0.
  stack = []
  for idx in range(0, num_outer_frames):
    stack.append(tb[idx])
  for idx in range(len(stack), len(stack) + num_user_frames):
    stack.append(_make_frame_with_filename(tb, idx, user_filename % idx))
  for idx in range(len(stack), len(stack) + num_inner_tf_frames):
    stack.append(_make_frame_with_filename(tb, idx, tf_filename % idx))
  return stack


class ComputeDeviceSummaryFromOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the helper that formats the device-assignment part of an error message summary.
  # Covers the case with active device assignments and the case with none. Has no shared fixtures.

  def testCorrectFormatWithActiveDeviceAssignments(self):
    # [FUNCTION DOC]
    # Scenario: two traceable device assignments (a cpu device set at a fake file and line, a gpu device
    # at another) are passed with a node name to the device-summary helper using a two-space prefix.
    # Asserts the returned summary text mentions the node name, each device in device-call form, and each
    # file-and-line location. Returns nothing; fails on assertion error.
    assignments = []
    assignments.append(
        traceable_stack.TraceableObject("/cpu:0", filename="hope.py", lineno=24)
    )
    assignments.append(
        traceable_stack.TraceableObject(
            "/gpu:2", filename="please.py", lineno=42
        )
    )

    summary = error_interpolation._compute_device_summary_from_list(
        "nodename", assignments, prefix="  "
    )

    self.assertIn("nodename", summary)
    self.assertIn("tf.device(/cpu:0)", summary)
    self.assertIn("<hope.py:24>", summary)
    self.assertIn("tf.device(/gpu:2)", summary)
    self.assertIn("<please.py:42>", summary)

  def testCorrectFormatWhenNoColocationsWereActive(self):
    # [FUNCTION DOC]
    # Scenario: the device-summary helper is given a node name and an empty list of device assignments.
    # Asserts the summary mentions the node name and says there are no device assignments.
    device_assignment_list = []
    summary = error_interpolation._compute_device_summary_from_list(
        "nodename", device_assignment_list, prefix="  "
    )
    self.assertIn("nodename", summary)
    self.assertIn("No device assignments", summary)


class ComputeColocationSummaryFromOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the helper that formats the colocation part of an error message summary from a mapping of
  # colocated node names to traceable objects. Covers the populated and the empty mapping.

  def testCorrectFormatWithActiveColocations(self):
    # [FUNCTION DOC]
    # Scenario: a mapping of two node names to traceable objects carrying fake file names and line numbers
    # is given to the colocation-summary helper. Asserts the summary contains the node name, each
    # colocation in colocate-with form and each file-and-line location.
    t_obj_1 = traceable_stack.TraceableObject(
        None, filename="test_1.py", lineno=27
    )
    t_obj_2 = traceable_stack.TraceableObject(
        None, filename="test_2.py", lineno=38
    )
    colocation_dict = {
        "test_node_1": t_obj_1,
        "test_node_2": t_obj_2,
    }
    summary = error_interpolation._compute_colocation_summary_from_dict(
        "node_name", colocation_dict, prefix="  "
    )
    self.assertIn("node_name", summary)
    self.assertIn("colocate_with(test_node_1)", summary)
    self.assertIn("<test_1.py:27>", summary)
    self.assertIn("colocate_with(test_node_2)", summary)
    self.assertIn("<test_2.py:38>", summary)

  def testCorrectFormatWhenNoColocationsWereActive(self):
    # [FUNCTION DOC]
    # Scenario: the colocation-summary helper is called with a node name and an empty mapping.
    # Asserts the summary mentions the node name and says there are no node-device colocations.
    colocation_dict = {}
    summary = error_interpolation._compute_colocation_summary_from_dict(
        "node_name", colocation_dict, prefix="  "
    )
    self.assertIn("node_name", summary)
    self.assertIn("No node-device colocations", summary)


class InterpolateFilenamesAndLineNumbersTest(test.TestCase):
  # [CLASS DOC]
  # Tests for finding the user-defining frame of an operation traceback and for interpolating node tags
  # in message strings into text describing where graph nodes were defined. Each test builds its own
  # default graph; there are no shared fixtures.

  def testFindIndexOfDefiningFrameForOp(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph a constant operation is created and its traceback is rewritten so three
    # frames look like user frames and five innermost frames look like framework frames. Asserts the
    # defining-frame finder returns the index of the last user frame, which is sixth from the end.
    with ops.Graph().as_default():
      local_op = constant_op.constant(42).op
      user_filename = "hope.py"
      modified_tb = _modify_op_stack_with_filenames(
          local_op.traceback,
          num_user_frames=3,
          user_filename=user_filename,
          num_inner_tf_frames=5,
      )
      idx = error_interpolation._find_index_of_defining_frame(modified_tb)
      # Expected frame is 6th from the end because there are 5 inner frames with
      # TF filenames.
      expected_frame = len(modified_tb) - 6
      self.assertEqual(expected_frame, idx)

  def testFindIndexOfDefiningFrameForOpReturnsZeroOnError(self):
    # [FUNCTION DOC]
    # Scenario: an operation traceback is truncated to seven frames and all seven are relabeled as
    # framework frames, so no user frame exists. Asserts the defining-frame finder falls back to index zero.
    with ops.Graph().as_default():
      local_op = constant_op.constant(43).op
      # Ensure all frames look like TF frames.
      modified_tb = _modify_op_stack_with_filenames(
          local_op.traceback[:7],  # Truncate stack to known length.
          num_user_frames=0,
          user_filename="user_file.py",
          num_inner_tf_frames=7,
      )
      idx = error_interpolation._find_index_of_defining_frame(modified_tb)
      self.assertEqual(0, idx)

  def testNothingToDo(self):
    # [FUNCTION DOC]
    # Scenario: a plain string with no node tags is interpolated against a graph holding one constant.
    # Asserts the original text is still contained in the result.
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      normal_string = "This is just a normal string"
      interpolated_string = error_interpolation.interpolate_graph(
          normal_string, ops.get_default_graph()
      )
      self.assertIn(normal_string, interpolated_string)

  def testOneTagWithAFakeNameResultsInPlaceholders(self):
    # [FUNCTION DOC]
    # Scenario: a node tag naming an operation that does not exist in the graph is interpolated.
    # Asserts the tag text is left in the output unchanged rather than raising an error.
    with ops.Graph().as_default():
      one_tag_string = "{{node MinusOne}}"
      interpolated_string = error_interpolation.interpolate_graph(
          one_tag_string, ops.get_default_graph()
      )
      self.assertIn(one_tag_string, interpolated_string)

  def testOneTagWithAFakeFunctionTag(self):
    # [FUNCTION DOC]
    # Scenario: a function-node tag for a nonexistent function followed by a node tag for an existing
    # constant is interpolated in a graph with two constants. Asserts the output describes the named node
    # with a "defined at" location in this test file, drops the function-node tag text, and does not
    # mention the other, unreferenced node.
    defined_at = r"defined at.*error_interpolation_test\.py"
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      one_tag_with_a_fake_function_tag = "{{function_node fake}}{{node One}}"
      interpolated_string = error_interpolation.interpolate_graph(
          one_tag_with_a_fake_function_tag, ops.get_default_graph()
      )
      # Fragments the expression to avoid matching the pattern itself.
      expected_regex = re.compile(rf"node 'One'.*{defined_at}", re.DOTALL)
      self.assertRegex(interpolated_string, expected_regex)
      self.assertNotIn("function_node", interpolated_string)
      self.assertNotIn("node 'Two'", interpolated_string)

  def testTwoTagsNoSeps(self):
    # [FUNCTION DOC]
    # Scenario: two adjacent node tags with no separator, naming the first and third of three constants.
    # Asserts both nodes are described in order, each with a "defined at" location in this test file.
    defined_at = r"defined at.*error_interpolation_test\.py"
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      constant_op.constant(3, name="Three")
      two_tags_no_seps = "{{node One}}{{node Three}}"
      interpolated_string = error_interpolation.interpolate_graph(
          two_tags_no_seps, ops.get_default_graph()
      )
      # Fragments the expression to avoid matching the pattern itself.
      expected_regex = re.compile(
          rf"node 'One'.*{defined_at}.*node 'Three'.*{defined_at}", re.DOTALL
      )
      self.assertRegex(interpolated_string, expected_regex)

  def testTwoTagsWithSeps(self):
    # [FUNCTION DOC]
    # Scenario: two node tags separated by punctuation and surrounded by punctuation. Asserts both named
    # nodes are described in order with "defined at" locations in this test file.
    defined_at = r"defined at.*error_interpolation_test\.py"
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      constant_op.constant(3, name="Three")
      two_tags_with_seps = ";;;{{node Two}},,,{{node Three}};;;"
      interpolated_string = error_interpolation.interpolate_graph(
          two_tags_with_seps, ops.get_default_graph()
      )
      # Fragments the expression to avoid matching the pattern itself.
      expected_regex = re.compile(
          rf"node 'Two'.*{defined_at}.*node 'Three'.*{defined_at}", re.DOTALL
      )
      self.assertRegex(interpolated_string, expected_regex)

  def testNewLine(self):
    # [FUNCTION DOC]
    # Scenario: a string beginning with blank lines and containing one node tag is interpolated.
    # Asserts the named node is described with a "defined at" location in this test file.
    defined_at = r"defined at.*error_interpolation_test\.py"
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      newline = "\n\n;;;{{node One}};;;"
      interpolated_string = error_interpolation.interpolate_graph(
          newline, ops.get_default_graph()
      )
      expected_regex = re.compile(rf"node 'One'.*{defined_at}", re.DOTALL)
      self.assertRegex(interpolated_string, expected_regex)


class OperationDefinedAtTraceTest(test.TestCase):
  # [CLASS DOC]
  # Tests, run only in version-2 behavior, that errors raised while running a compiled function include
  # a "defined at" traceback naming the Python functions that created the failing operation.

  @test_util.run_v2_only
  def testSimpleCall(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function builds a matrix multiplication of a one-by-three constant with the
    # output of a Python callback also shaped one-by-three, which is invalid. Calling it must raise an
    # invalid-argument error whose message mentions "defined at" followed by this test and the function.
    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Nested compiled function used by the test: creates a constant and a callback-produced tensor and
      # returns their matrix product, which fails at run time because the shapes are incompatible.
      x = constant_op.constant([[1, 2, 3]])
      y = script_ops.eager_py_func(lambda: [[1, 2, 3]], (), dtypes.int32)
      return math_ops.matmul(x, y)

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        re.compile(
            r"defined at.*" r"in testSimpleCall.*" r"in func", re.DOTALL
        ),
    ):
      func()

  @test_util.run_v2_only
  def testNestedCall(self):
    # [FUNCTION DOC]
    # Scenario: like the simple-call test, but the failing matrix multiplication is created in an inner
    # plain function called from a compiled function. Expects an invalid-argument error whose message lists
    # the test, the compiled function and the inner function as "defined at" locations in that order.
    def inner():
      # [FUNCTION DOC]
      # Nested helper used by the test: builds a constant and a callback tensor and returns their invalid
      # matrix product. Takes no arguments.
      x = constant_op.constant([[1, 2, 3]])
      y = script_ops.eager_py_func(lambda: [[1, 2, 3]], (), dtypes.int32)
      return math_ops.matmul(x, y)

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Nested compiled function used by the test: simply calls and returns the inner function result.
      return inner()

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        re.compile(
            r"defined at.*" r"in testNestedCall.*" r"in func.*" r"in inner",
            re.DOTALL,
        ),
    ):
      func()

  @test_util.run_v2_only
  def testAssert(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function runs an assertion op whose condition is false. Expects an
    # invalid-argument error whose message mentions "defined at" followed by this test and the function.
    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Nested compiled function used by the test: asserts a false condition and returns nothing.
      control_flow_assert.Assert(False, [False])
      return

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        re.compile(r"defined at.*" r"in testAssert.*" r"in func", re.DOTALL),
    ):
      func()

  @test_util.run_v2_only
  def testControlFlow(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function whose false-condition branch builds an invalid matrix multiplication.
    # Expects an invalid-argument error whose message mentions "defined at" with this test and the function.
    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Nested compiled function used by the test: branches on a constant false value; the taken branch
      # creates constants and a callback tensor and returns their invalid matrix product.
      if constant_op.constant(False):
        return constant_op.constant(1)

      else:
        x = constant_op.constant([[1, 2, 3]])
        y = script_ops.eager_py_func(lambda: [[1, 2, 3]], (), dtypes.int32)
        return math_ops.matmul(x, y)

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        re.compile(
            r"defined at.*" r"in testControlFlow.*" r"in func", re.DOTALL
        ),
    ):
      func()


class IsFrameworkFilenameTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the predicate that decides whether a file name belongs to the framework and so should be
  # skipped when choosing a user-defining frame.

  def testAllowsUnitTests(self):
    # [FUNCTION DOC]
    # Scenario: a file name made of the framework path prefix plus a name ending in a test suffix.
    # Asserts the predicate returns false, i.e. unit test files count as user code.
    self.assertFalse(
        error_interpolation._is_framework_filename(
            error_interpolation._FRAMEWORK_PATH_PREFIXES[0] + "foobar_test.py"
        )
    )

  def testFrameworkPythonFile(self):
    # [FUNCTION DOC]
    # Scenario: the path of the error interpolation module itself is checked. Asserts the predicate
    # returns true.
    self.assertTrue(
        error_interpolation._is_framework_filename(error_interpolation.__file__)
    )

  def testEmbedded(self):
    # [FUNCTION DOC]
    # Scenario: a file name from an embedded standard library marker is checked. Asserts the predicate
    # returns true.
    self.assertTrue(
        error_interpolation._is_framework_filename(
            "<embedded stdlib>/context_lib.py"
        )
    )


if __name__ == "__main__":
  test.main()
