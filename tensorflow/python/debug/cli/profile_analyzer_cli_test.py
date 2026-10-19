# [MODULE DOC]
# Unit tests for the tfdbg profile analyzer command-line module. One test class checks the
# list-profile command (empty data, single and multiple devices, a real session run, sorting,
# filtering and time-unit selection) using hand-built run metadata and mocked graphs. The other
# class checks the print-source command against a real while-loop run, including cost bars,
# clickable menu links, filters, error messages and the initial scroll position. Also holds small
# line-matching helper functions. Tests are restricted to graph-and-session (v1) mode.
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
"""Tests for profile_analyzer_cli."""

import re

from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import profile_analyzer_cli
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def no_rewrite_session_config():
  # [FUNCTION DOC]
  # HELPER: builds a session configuration for these tests. Returns a configuration message with
  # graph pruning and constant folding turned off so that every op executes and appears in the
  # profile. No side effects.
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True,
      constant_folding=rewriter_config_pb2.RewriterConfig.OFF)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


def _line_number_above():
  # [FUNCTION DOC]
  # HELPER: lets tests record the source line of the statement just above a call. Returns the
  # line number of the caller's current line minus one, read from the call stack.
  return tf_inspect.stack()[1][2] - 1


def _at_least_one_line_matches(pattern, lines):
  # [FUNCTION DOC]
  # HELPER: search utility for assertions. pattern is a regular expression string; lines is a
  # list of strings. Returns a pair: true and the index of the first line containing a match, or
  # false and nothing when no line matches.
  pattern_re = re.compile(pattern)
  for i, line in enumerate(lines):
    if pattern_re.search(line):
      return True, i
  return False, None


def _assert_at_least_one_line_matches(pattern, lines):
  # [FUNCTION DOC]
  # HELPER: assertion utility. Raises an assertion error showing the pattern and lines if no line
  # matches the regular expression pattern; returns nothing otherwise.
  any_match, _ = _at_least_one_line_matches(pattern, lines)
  if not any_match:
    raise AssertionError(
        "%s does not match any line in %s." % (pattern, str(lines)))


def _assert_no_lines_match(pattern, lines):
  # [FUNCTION DOC]
  # HELPER: assertion utility. Raises an assertion error if any line matches the regular
  # expression pattern; returns nothing otherwise.
  any_match, _ = _at_least_one_line_matches(pattern, lines)
  if any_match:
    raise AssertionError(
        "%s matched at least one line in %s." % (pattern, str(lines)))


@test_util.run_v1_only("Requires tf.Session")
class ProfileAnalyzerListProfileTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the list-profile command of the profile analyzer. Most tests build run metadata with
  # hand-made node stats and a mocked graph whose operations carry fake tracebacks and types, then
  # check the text of the returned table lines. Runs only in session (v1) mode.

  def testNodeInfoEmpty(self):
    # [FUNCTION DOC]
    # An empty graph and empty run metadata are given; list-profile is expected to return only a
    # single empty line.
    graph = ops.Graph()
    run_metadata = config_pb2.RunMetadata()

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)
    prof_output = prof_analyzer.list_profile([]).lines
    self.assertEqual([""], prof_output)

  def testSingleDevice(self):
    # [FUNCTION DOC]
    # One device with two nodes (add and mul) and a mocked graph. Expects the device heading
    # and one row per node showing op type, op time (2us, 1us) and exec time (4us, 3us).
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)

    node2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        op_start_rel_micros=1,
        op_end_rel_micros=2,
        all_end_rel_micros=3)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1, node2])

    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = "Add/123"
    op1.traceback = [("a/b/file1", 10, "some_var")]
    op1.type = "add"
    op2 = test.mock.MagicMock()
    op2.name = "Mul/456"
    op2.traceback = [("a/b/file1", 11, "some_var")]
    op2.type = "mul"
    graph.get_operations.return_value = [op1, op2]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)
    prof_output = prof_analyzer.list_profile([]).lines

    _assert_at_least_one_line_matches(r"Device 1 of 1: deviceA", prof_output)
    _assert_at_least_one_line_matches(r"^Add/123.*add.*2us.*4us", prof_output)
    _assert_at_least_one_line_matches(r"^Mul/456.*mul.*1us.*3us", prof_output)

  def testMultipleDevices(self):
    # [FUNCTION DOC]
    # Two devices share the same node stats. Expects headings "Device 1 of 2" and "Device 2 of 2";
    # then with the device filter set to the second device, only that device section remains.
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=3)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1])

    device2 = run_metadata.step_stats.dev_stats.add()
    device2.device = "deviceB"
    device2.node_stats.extend([node1])

    graph = test.mock.MagicMock()
    op = test.mock.MagicMock()
    op.name = "Add/123"
    op.traceback = [("a/b/file1", 10, "some_var")]
    op.type = "abc"
    graph.get_operations.return_value = [op]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)
    prof_output = prof_analyzer.list_profile([]).lines

    _assert_at_least_one_line_matches(r"Device 1 of 2: deviceA", prof_output)
    _assert_at_least_one_line_matches(r"Device 2 of 2: deviceB", prof_output)

    # Try filtering by device.
    prof_output = prof_analyzer.list_profile(["-d", "deviceB"]).lines
    _assert_at_least_one_line_matches(r"Device 2 of 2: deviceB", prof_output)
    _assert_no_lines_match(r"Device 1 of 2: deviceA", prof_output)

  def testWithSession(self):
    # [FUNCTION DOC]
    # Runs an add of two constants in a real session with full tracing and no graph rewrites, then
    # lists the profile. Expects a device heading, the column headers in order, a row for the add
    # node and the device total row.
    options = config_pb2.RunOptions()
    options.trace_level = config_pb2.RunOptions.FULL_TRACE
    run_metadata = config_pb2.RunMetadata()

    with session.Session(config=no_rewrite_session_config()) as sess:
      a = constant_op.constant([1, 2, 3])
      b = constant_op.constant([2, 2, 1])
      result = math_ops.add(a, b)

      sess.run(result, options=options, run_metadata=run_metadata)

      prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(
          sess.graph, run_metadata)
      prof_output = prof_analyzer.list_profile([]).lines

      _assert_at_least_one_line_matches("Device 1 of", prof_output)
      expected_headers = [
          "Node", r"Start Time \(us\)", r"Op Time \(.*\)", r"Exec Time \(.*\)",
          r"Filename:Lineno\(function\)"]
      _assert_at_least_one_line_matches(
          ".*".join(expected_headers), prof_output)
      _assert_at_least_one_line_matches(r"^Add/", prof_output)
      _assert_at_least_one_line_matches(r"Device Total", prof_output)

  def testSorting(self):
    # [FUNCTION DOC]
    # Two nodes with different start times, op times, exec times and source files. Checks the
    # row order under: default (start time), reverse, by node name, by op time, by exec time and
    # by file line, asserting the order of the two node names in the joined output.
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        all_start_micros=123,
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)

    node2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        all_start_micros=122,
        op_start_rel_micros=1,
        op_end_rel_micros=2,
        all_end_rel_micros=5)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1, node2])

    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = "Add/123"
    op1.traceback = [("a/b/file2", 10, "some_var")]
    op1.type = "add"
    op2 = test.mock.MagicMock()
    op2.name = "Mul/456"
    op2.traceback = [("a/b/file1", 11, "some_var")]
    op2.type = "mul"
    graph.get_operations.return_value = [op1, op2]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)

    # Default sort by start time (i.e. all_start_micros).
    prof_output = prof_analyzer.list_profile([]).lines
    self.assertRegex("".join(prof_output), r"Mul/456.*Add/123")
    # Default sort in reverse.
    prof_output = prof_analyzer.list_profile(["-r"]).lines
    self.assertRegex("".join(prof_output), r"Add/123.*Mul/456")
    # Sort by name.
    prof_output = prof_analyzer.list_profile(["-s", "node"]).lines
    self.assertRegex("".join(prof_output), r"Add/123.*Mul/456")
    # Sort by op time (i.e. op_end_rel_micros - op_start_rel_micros).
    prof_output = prof_analyzer.list_profile(["-s", "op_time"]).lines
    self.assertRegex("".join(prof_output), r"Mul/456.*Add/123")
    # Sort by exec time (i.e. all_end_rel_micros).
    prof_output = prof_analyzer.list_profile(["-s", "exec_time"]).lines
    self.assertRegex("".join(prof_output), r"Add/123.*Mul/456")
    # Sort by line number.
    prof_output = prof_analyzer.list_profile(["-s", "line"]).lines
    self.assertRegex("".join(prof_output), r"Mul/456.*Add/123")

  def testFiltering(self):
    # [FUNCTION DOC]
    # Two nodes (add and mul). Applies filters by node name, op type, file name, execution time
    # interval and op time interval, asserting that only the expected node appears each time.
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        all_start_micros=123,
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)

    node2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        all_start_micros=122,
        op_start_rel_micros=1,
        op_end_rel_micros=2,
        all_end_rel_micros=5)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1, node2])

    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = "Add/123"
    op1.traceback = [("a/b/file2", 10, "some_var")]
    op1.type = "add"
    op2 = test.mock.MagicMock()
    op2.name = "Mul/456"
    op2.traceback = [("a/b/file1", 11, "some_var")]
    op2.type = "mul"
    graph.get_operations.return_value = [op1, op2]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)

    # Filter by name
    prof_output = prof_analyzer.list_profile(["-n", "Add"]).lines
    _assert_at_least_one_line_matches(r"Add/123", prof_output)
    _assert_no_lines_match(r"Mul/456", prof_output)
    # Filter by op_type
    prof_output = prof_analyzer.list_profile(["-t", "mul"]).lines
    _assert_at_least_one_line_matches(r"Mul/456", prof_output)
    _assert_no_lines_match(r"Add/123", prof_output)
    # Filter by file name.
    prof_output = prof_analyzer.list_profile(["-f", ".*file2"]).lines
    _assert_at_least_one_line_matches(r"Add/123", prof_output)
    _assert_no_lines_match(r"Mul/456", prof_output)
    # Filter by execution time.
    prof_output = prof_analyzer.list_profile(["-e", "[5, 10]"]).lines
    _assert_at_least_one_line_matches(r"Mul/456", prof_output)
    _assert_no_lines_match(r"Add/123", prof_output)
    # Filter by op time.
    prof_output = prof_analyzer.list_profile(["-o", ">=2"]).lines
    _assert_at_least_one_line_matches(r"Add/123", prof_output)
    _assert_no_lines_match(r"Mul/456", prof_output)

  def testSpecifyingTimeUnit(self):
    # [FUNCTION DOC]
    # Lists the profile with the time unit forced to milliseconds. Expects the rows and the device
    # total row to show op times in milliseconds (0.002, 0.005 and total 0.009).
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        all_start_micros=123,
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)

    node2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        all_start_micros=122,
        op_start_rel_micros=1,
        op_end_rel_micros=2,
        all_end_rel_micros=5)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1, node2])

    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = "Add/123"
    op1.traceback = [("a/b/file2", 10, "some_var")]
    op1.type = "add"
    op2 = test.mock.MagicMock()
    op2.name = "Mul/456"
    op2.traceback = [("a/b/file1", 11, "some_var")]
    op2.type = "mul"
    graph.get_operations.return_value = [op1, op2]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)

    # Force time unit.
    prof_output = prof_analyzer.list_profile(["--time_unit", "ms"]).lines
    _assert_at_least_one_line_matches(r"Add/123.*add.*0\.002ms", prof_output)
    _assert_at_least_one_line_matches(r"Mul/456.*mul.*0\.005ms", prof_output)
    _assert_at_least_one_line_matches(r"Device Total.*0\.009ms", prof_output)


@test_util.run_v1_only("Requires tf.Session")
class ProfileAnalyzerPrintSourceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the print-source command of the profile analyzer. The fixture runs a small while
  # loop in a real traced session from this very test file and records the line numbers of the
  # loop condition, body, constant and loop statements and an analyzer over the resulting graph and
  # run metadata. Tests print this file with annotations. Runs only in session (v1) mode.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Builds a while loop (condition less than ten, body adds one, starting at zero) in a
    # new session with full tracing, recording the source line numbers of each defining statement
    # on the test instance, checks the loop result is ten, and stores a profile analyzer built from
    # the session graph and run metadata.
    super(ProfileAnalyzerPrintSourceTest, self).setUp()

    options = config_pb2.RunOptions()
    options.trace_level = config_pb2.RunOptions.FULL_TRACE
    run_metadata = config_pb2.RunMetadata()
    with session.Session() as sess:
      loop_cond = lambda x: math_ops.less(x, 10)
      self.loop_cond_lineno = _line_number_above()
      loop_body = lambda x: math_ops.add(x, 1)
      self.loop_body_lineno = _line_number_above()
      x = constant_op.constant(0, name="x")
      self.x_lineno = _line_number_above()
      loop = while_loop.while_loop(loop_cond, loop_body, [x])
      self.loop_lineno = _line_number_above()
      self.assertEqual(
          10, sess.run(loop, options=options, run_metadata=run_metadata))

      self.prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(
          sess.graph, run_metadata)

  def tearDown(self):
    # [FUNCTION DOC]
    # Cleanup. Resets the default graph, then runs the parent cleanup.
    ops.reset_default_graph()
    super(ProfileAnalyzerPrintSourceTest, self).tearDown()

  def testPrintSourceForWhileLoop(self):
    # [FUNCTION DOC]
    # Prints this test file's annotated source. Expects cost-bar lines with node and execution
    # counts 2(22), 2(20) and 7(55) on the lines of the loop condition, body and loop statements.
    prof_output = self.prof_analyzer.print_source([__file__])

    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*2\(22\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*2\(20\) .*L%d.*(\S)+" % self.loop_body_lineno,
        prof_output.lines)
    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*7\(55\) .*L%d.*(\S)+" % self.loop_lineno,
        prof_output.lines)

  def testPrintSourceOutputContainsClickableLinks(self):
    # [FUNCTION DOC]
    # Prints the annotated source and finds the loop-condition line. Expects its font segments to
    # include a menu item whose command starts with the list-profile alias and a file path filter,
    # with minimum line equal to that line and maximum line one greater.
    prof_output = self.prof_analyzer.print_source([__file__])
    any_match, line_index = _at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*2\(22\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    self.assertTrue(any_match)
    any_menu_item_match = False
    for seg in prof_output.font_attr_segs[line_index]:
      if (isinstance(seg[2][1], debugger_cli_common.MenuItem) and
          seg[2][1].content.startswith("lp --file_path_filter ") and
          "--min_lineno %d" % self.loop_cond_lineno in seg[2][1].content and
          "--max_lineno %d" % (self.loop_cond_lineno + 1) in seg[2][1].content):
        any_menu_item_match = True
        break
    self.assertTrue(any_menu_item_match)

  def testPrintSourceWithNonDefaultTimeUnit(self):
    # [FUNCTION DOC]
    # Prints with the time unit set to milliseconds. Expects the same annotated lines as the
    # default case but with costs shown in milliseconds.
    prof_output = self.prof_analyzer.print_source([
        __file__, "--time_unit", "ms"])

    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*ms .*2\(22\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*ms .*2\(20\) .*L%d.*(\S)+" % self.loop_body_lineno,
        prof_output.lines)
    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*ms .*7\(55\) .*L%d.*(\S)+" % self.loop_lineno,
        prof_output.lines)

  def testPrintSourceWithNodeNameFilter(self):
    # [FUNCTION DOC]
    # Prints with a node name filter matching only the constant node. Expects only the constant's
    # line annotated (1 node, 1 execution), the loop lines absent, and the clickable link's
    # command to carry the node name filter and the one-line range.
    prof_output = self.prof_analyzer.print_source([
        __file__, "--node_name_filter", "x$"])

    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*1\(1\) .*L%d.*(\S)+" % self.x_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*us .*2\(22\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*us .*2\(20\) .*L%d.*(\S)+" % self.loop_body_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*ms .*7\(55\) .*L%d.*(\S)+" % self.loop_lineno,
        prof_output.lines)

    # Check clickable link.
    _, line_index = _at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*1\(1\) .*L%d.*(\S)+" % self.x_lineno,
        prof_output.lines)
    any_menu_item_match = False
    for seg in prof_output.font_attr_segs[line_index]:
      if (isinstance(seg[2][1], debugger_cli_common.MenuItem) and
          seg[2][1].content.startswith("lp --file_path_filter ") and
          "--node_name_filter x$" in seg[2][1].content and
          "--min_lineno %d" % self.x_lineno in seg[2][1].content and
          "--max_lineno %d" % (self.x_lineno + 1) in seg[2][1].content):
        any_menu_item_match = True
        break
    self.assertTrue(any_menu_item_match)

  def testPrintSourceWithOpTypeFilter(self):
    # [FUNCTION DOC]
    # Prints with an op type filter for the less-than op. Expects the loop-condition line with
    # 1 node and 11 executions, and the body and loop lines absent.
    prof_output = self.prof_analyzer.print_source([
        __file__, "--op_type_filter", "Less"])

    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*1\(11\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*us .*2\(20\) .*L%d.*(\S)+" % self.loop_body_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*us .*7\(55\) .*L%d.*(\S)+" % self.loop_lineno,
        prof_output.lines)

  def testPrintSourceWithNonexistentDeviceGivesCorrectErrorMessage(self):
    # [FUNCTION DOC]
    # Prints with a device filter naming a nonexistent device. Expects the "no profile
    # information" message and a line echoing the device filter.
    prof_output = self.prof_analyzer.print_source([
        __file__, "--device_name_filter", "foo_device"])

    _assert_at_least_one_line_matches(
        r"The source file .* does not contain any profile information for the "
        "previous Session run", prof_output.lines)
    _assert_at_least_one_line_matches(
        r".*--device_name_filter: foo_device", prof_output.lines)

  def testPrintSourceWithUnrelatedFileShowsCorrectErrorMessage(self):
    # [FUNCTION DOC]
    # Prints a source file unrelated to the run (the inspection utility module). Expects the "no
    # profile information" message.
    prof_output = self.prof_analyzer.print_source([tf_inspect.__file__])
    _assert_at_least_one_line_matches(
        r"The source file .* does not contain any profile information for the "
        "previous Session run", prof_output.lines)

  def testPrintSourceOutputContainsInitScrollPosAnnotation(self):
    # [FUNCTION DOC]
    # Prints with the initial line set to the loop-condition line. Expects the initial scroll
    # position annotation to equal that line number plus one, accounting for the two header lines.
    prof_output = self.prof_analyzer.print_source([
        __file__, "--init_line", str(self.loop_cond_lineno)])
    self.assertEqual(
        self.loop_cond_lineno + 1,  # The extra line is due to the head lines.
        prof_output.annotations[debugger_cli_common.INIT_SCROLL_POS_KEY])


if __name__ == "__main__":
  googletest.main()
