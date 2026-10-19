# [MODULE DOC]
# Tests of the analyzer command-line backend of the TensorFlow debugger (tfdbg). The module
# contains module-level assertion and construction helpers (for checking tensor lists, node
# information, error output, main menus and menu items, and for building an analyzer plus a
# command registry), and four test classes: one on a simple multiply-add graph (listing,
# sorting, filtering, node info, printing tensors, evaluating expressions, tensor filters,
# source annotation and source listing), one on printing large tensors, one on graphs with
# control dependencies (node info, list inputs, list recipients), and one on a while-loop graph
# with multiple dumps per tensor. The graph tests run under session-based (version 1) mode,
# write debug dumps to temporary directories and delete them afterwards. Running the file as a
# script starts the standard test main.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests of the Analyzer CLI Backend."""
import os
import tempfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import cli_test_utils
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import while_loop as while_loop_tf
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


# Helper function to accommodate MKL-enabled TensorFlow:
# MatMul op is supported by MKL for some data types and its name is prefixed
# with "_Mkl" during the MKL graph rewrite pass.
def _matmul_op_name():
  # [FUNCTION DOC]
  # HELPER: lets the tests expect the right matrix-multiply op name.
  # Returns the expected op type name of the matrix multiplication: the MKL-prefixed name when
  # MKL is enabled and the graph data type is one MKL supports, otherwise the plain name.
  if (test_util.IsMklEnabled() and
      _get_graph_matmul_dtype() in _mkl_matmul_supported_types()):
    return "_MklMatMul"
  else:
    return "MatMul"


# Helper function to get MklMatMul supported types
def _mkl_matmul_supported_types():
  # [FUNCTION DOC]
  # HELPER: supplies the data type names MKL's matrix multiply supports, for the op-name helper.
  # Returns a set of type name strings (32-bit float and bfloat16).
  return {"float32", "bfloat16"}


# Helper function to get dtype used in the graph of SetUpClass()
def _get_graph_matmul_dtype():
  # [FUNCTION DOC]
  # HELPER: reports the data type of the matrix multiply in the shared test graph.
  # Returns the type name string for 64-bit float, the default type of the constants used.
  # default dtype of matmul op created is float64
  return "float64"


def _cli_config_from_temp_file():
  # [FUNCTION DOC]
  # HELPER: gives tests an isolated debugger configuration.
  # Creates a new temporary directory and returns a CLI configuration object backed by a config
  # file inside it, so the user's real config file is never touched. Side effect: creates the
  # directory and writes the config file there.
  return cli_config.CLIConfig(
      config_file_path=os.path.join(tempfile.mkdtemp(), ".tfdbg_config"))


def no_rewrite_session_config():
  # [FUNCTION DOC]
  # HELPER: session setup for tests whose graph nodes must stay unchanged.
  # Builds and returns a session configuration whose graph optimizer is set to disable model
  # pruning, constant folding, arithmetic optimization, dependency optimization and pin-to-host
  # optimization, so that the dumped nodes match the graph as constructed.
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True,
      constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
      arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
      dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
      pin_to_host_optimization=rewriter_config_pb2.RewriterConfig.OFF)

  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


def line_number_above():
  # [FUNCTION DOC]
  # HELPER: lets tests record the source line of the statement just before a call.
  # Inspects the call stack and returns the line number of the caller's current line minus one,
  # as an integer. Used to learn where graph nodes were created in this file.
  return tf_inspect.stack()[1][2] - 1


def parse_op_and_node(line):
  # [FUNCTION DOC]
  # HELPER: parsing step of the node-info output checker.
  # Takes one text line of the form bracketed op type followed by a node name (extra trailing
  # items are tolerated). Returns a pair of strings: the op type without brackets and the node
  # name (the second whitespace-separated item).
  """Parse a line containing an op node followed by a node name.

  For example, if the line is
    "  [Variable] hidden/weights",
  this function will return ("Variable", "hidden/weights")

  Args:
    line: The line to be parsed, as a str.

  Returns:
    Name of the parsed op type.
    Name of the parsed node.
  """

  op_type = line.strip().split(" ")[0].replace("[", "").replace("]", "")

  # Not using [-1], to tolerate any other items that might be present behind
  # the node name.
  node_name = line.strip().split(" ")[1]

  return op_type, node_name


def assert_column_header_command_shortcut(tst,
                                          command,
                                          reverse,
                                          node_name_regex,
                                          op_type_regex,
                                          tensor_filter_name):
  # [FUNCTION DOC]
  # HELPER: assertion helper for the tensor-list checker.
  # Takes the test case, a column-header command string, the reverse flag, node name regex, op
  # type regex and tensor filter name. Asserts that when reverse sorting is active the command
  # does not carry the reverse option (clicking a header toggles it), and that when a regex or
  # filter name is unset no option text built from an unset value appears in the command.
  # Only the negative direction is checked, so it passes trivially in many cases. Returns nothing.
  tst.assertFalse(reverse and "-r" in command)

  tst.assertFalse(not(op_type_regex) and ("-t %s" % op_type_regex) in command)
  tst.assertFalse(
      not(node_name_regex) and ("-t %s" % node_name_regex) in command)
  tst.assertFalse(
      not(tensor_filter_name) and ("-t %s" % tensor_filter_name) in command)


def assert_listed_tensors(tst,
                          out,
                          expected_tensor_names,
                          expected_op_types,
                          node_name_regex=None,
                          op_type_regex=None,
                          tensor_filter_name=None,
                          sort_by="timestamp",
                          reverse=False):
  # [FUNCTION DOC]
  # HELPER: shared output checker for the list-tensors tests.
  # Verifies a rich-text output of the list-tensors command: the first line gives a count of
  # dumped tensors (at least the expected number); optional regex filter lines appear when
  # filters were given; then a blank line and a header with time, size, op type and tensor name
  # columns whose segments carry command shortcuts for sorting (checked for position and bold
  # style). The listed rows are parsed and verified to be sorted by the requested key (timestamp,
  # dump size, op type or tensor name; reversed if asked; any other key fails the test), and
  # every expected tensor name must be listed with its expected op type. Takes the test case,
  # output object, expected names and op types, optional filters, sort key and reverse flag.
  # Returns nothing.
  """Check RichTextLines output for list_tensors commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    expected_tensor_names: (list of str) Expected tensor names in the list.
    expected_op_types: (list of str) Expected op types of the tensors, in the
      same order as the expected_tensor_names.
    node_name_regex: Optional: node name regex filter.
    op_type_regex: Optional: op type regex filter.
    tensor_filter_name: Optional: name of the tensor filter.
    sort_by: (str) (timestamp | op_type | tensor_name) the field by which the
      tensors in the list are sorted.
    reverse: (bool) whether the sorting is in reverse (i.e., descending) order.
  """

  line_iter = iter(out.lines)
  attr_segs = out.font_attr_segs
  line_counter = 0

  num_dumped_tensors = int(next(line_iter).split(" ")[0])
  line_counter += 1
  tst.assertGreaterEqual(num_dumped_tensors, len(expected_tensor_names))

  if op_type_regex is not None:
    tst.assertEqual("Op type regex filter: \"%s\"" % op_type_regex,
                    next(line_iter))
    line_counter += 1

  if node_name_regex is not None:
    tst.assertEqual("Node name regex filter: \"%s\"" % node_name_regex,
                    next(line_iter))
    line_counter += 1

  tst.assertEqual("", next(line_iter))
  line_counter += 1

  # Verify the column heads "t (ms)", "Op type" and "Tensor name" are present.
  line = next(line_iter)
  tst.assertIn("t (ms)", line)
  tst.assertIn("Op type", line)
  tst.assertIn("Tensor name", line)

  # Verify the command shortcuts in the top row.
  attr_segs = out.font_attr_segs[line_counter]
  attr_seg = attr_segs[0]
  tst.assertEqual(0, attr_seg[0])
  tst.assertEqual(len("t (ms)"), attr_seg[1])
  command = attr_seg[2][0].content
  tst.assertIn("-s timestamp", command)
  assert_column_header_command_shortcut(
      tst, command, reverse, node_name_regex, op_type_regex,
      tensor_filter_name)
  tst.assertEqual("bold", attr_seg[2][1])

  idx0 = line.index("Size")
  attr_seg = attr_segs[1]
  tst.assertEqual(idx0, attr_seg[0])
  tst.assertEqual(idx0 + len("Size (B)"), attr_seg[1])
  command = attr_seg[2][0].content
  tst.assertIn("-s dump_size", command)
  assert_column_header_command_shortcut(tst, command, reverse, node_name_regex,
                                        op_type_regex, tensor_filter_name)
  tst.assertEqual("bold", attr_seg[2][1])

  idx0 = line.index("Op type")
  attr_seg = attr_segs[2]
  tst.assertEqual(idx0, attr_seg[0])
  tst.assertEqual(idx0 + len("Op type"), attr_seg[1])
  command = attr_seg[2][0].content
  tst.assertIn("-s op_type", command)
  assert_column_header_command_shortcut(
      tst, command, reverse, node_name_regex, op_type_regex,
      tensor_filter_name)
  tst.assertEqual("bold", attr_seg[2][1])

  idx0 = line.index("Tensor name")
  attr_seg = attr_segs[3]
  tst.assertEqual(idx0, attr_seg[0])
  tst.assertEqual(idx0 + len("Tensor name"), attr_seg[1])
  command = attr_seg[2][0].content
  tst.assertIn("-s tensor_name", command)
  assert_column_header_command_shortcut(
      tst, command, reverse, node_name_regex, op_type_regex,
      tensor_filter_name)
  tst.assertEqual("bold", attr_seg[2][1])

  # Verify the listed tensors and their timestamps.
  tensor_timestamps = []
  dump_sizes_bytes = []
  op_types = []
  tensor_names = []
  for line in line_iter:
    items = line.split(" ")
    items = [item for item in items if item]

    rel_time = float(items[0][1:-1])
    tst.assertGreaterEqual(rel_time, 0.0)

    tensor_timestamps.append(rel_time)
    dump_sizes_bytes.append(command_parser.parse_readable_size_str(items[1]))
    op_types.append(items[2])
    tensor_names.append(items[3])

  # Verify that the tensors should be listed in ascending order of their
  # timestamps.
  if sort_by == "timestamp":
    sorted_timestamps = sorted(tensor_timestamps)
    if reverse:
      sorted_timestamps.reverse()
    tst.assertEqual(sorted_timestamps, tensor_timestamps)
  elif sort_by == "dump_size":
    sorted_dump_sizes_bytes = sorted(dump_sizes_bytes)
    if reverse:
      sorted_dump_sizes_bytes.reverse()
    tst.assertEqual(sorted_dump_sizes_bytes, dump_sizes_bytes)
  elif sort_by == "op_type":
    sorted_op_types = sorted(op_types)
    if reverse:
      sorted_op_types.reverse()
    tst.assertEqual(sorted_op_types, op_types)
  elif sort_by == "tensor_name":
    sorted_tensor_names = sorted(tensor_names)
    if reverse:
      sorted_tensor_names.reverse()
    tst.assertEqual(sorted_tensor_names, tensor_names)
  else:
    tst.fail("Invalid value in sort_by: %s" % sort_by)

  # Verify that the tensors are all listed.
  for tensor_name, op_type in zip(expected_tensor_names, expected_op_types):
    tst.assertIn(tensor_name, tensor_names)
    index = tensor_names.index(tensor_name)
    tst.assertEqual(op_type, op_types[index])


def assert_node_attribute_lines(tst,
                                out,
                                node_name,
                                op_type,
                                device,
                                input_op_type_node_name_pairs,
                                ctrl_input_op_type_node_name_pairs,
                                recipient_op_type_node_name_pairs,
                                ctrl_recipient_op_type_node_name_pairs,
                                attr_key_val_pairs=None,
                                num_dumped_tensors=None,
                                show_stack_trace=False,
                                stack_trace_available=False):
  # [FUNCTION DOC]
  # HELPER: shared output checker for the node-info tests.
  # KNOWN ISSUE: the attribute section only collects key and value pairs and never compares
  # them to the expected pairs, so attribute contents are effectively unchecked.
  # Walks the output lines of a node-info command and asserts, in order: node name line, op type,
  # device, input and control input counts and listings, recipient and control recipient counts
  # and listings (recipients compared ignoring order), optional attribute section, optional
  # dumped-tensor section (count, line format, timestamps ascending), and optional stack-trace
  # section (either an unavailable message, or repeated five-line records of depth, line,
  # function, text and a blank line). Takes the test case, output, expected node name, op type,
  # device, the four lists of op type and node name pairs, and optional attributes, dump count,
  # stack-trace flags. Returns nothing.
  """Check RichTextLines output for node_info commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    node_name: Name of the node.
    op_type: Op type of the node, as a str.
    device: Name of the device on which the node resides.
    input_op_type_node_name_pairs: A list of 2-tuples of op type and node name,
      for the (non-control) inputs to the node.
    ctrl_input_op_type_node_name_pairs: A list of 2-tuples of op type and node
      name, for the control inputs to the node.
    recipient_op_type_node_name_pairs: A list of 2-tuples of op type and node
      name, for the (non-control) output recipients to the node.
    ctrl_recipient_op_type_node_name_pairs: A list of 2-tuples of op type and
      node name, for the control output recipients to the node.
    attr_key_val_pairs: Optional: attribute key-value pairs of the node, as a
      list of 2-tuples.
    num_dumped_tensors: Optional: number of tensor dumps from the node.
    show_stack_trace: (bool) whether the stack trace of the node's
      construction is asserted to be present.
    stack_trace_available: (bool) whether Python stack trace is available.
  """

  line_iter = iter(out.lines)

  tst.assertEqual("Node %s" % node_name, next(line_iter))
  tst.assertEqual("", next(line_iter))
  tst.assertEqual("  Op: %s" % op_type, next(line_iter))
  tst.assertEqual("  Device: %s" % device, next(line_iter))
  tst.assertEqual("", next(line_iter))
  tst.assertEqual("  %d input(s) + %d control input(s):" %
                  (len(input_op_type_node_name_pairs),
                   len(ctrl_input_op_type_node_name_pairs)), next(line_iter))

  # Check inputs.
  tst.assertEqual("    %d input(s):" % len(input_op_type_node_name_pairs),
                  next(line_iter))
  for op_type, node_name in input_op_type_node_name_pairs:
    tst.assertEqual("      [%s] %s" % (op_type, node_name), next(line_iter))

  tst.assertEqual("", next(line_iter))

  # Check control inputs.
  if ctrl_input_op_type_node_name_pairs:
    tst.assertEqual("    %d control input(s):" %
                    len(ctrl_input_op_type_node_name_pairs), next(line_iter))
    for op_type, node_name in ctrl_input_op_type_node_name_pairs:
      tst.assertEqual("      [%s] %s" % (op_type, node_name), next(line_iter))

    tst.assertEqual("", next(line_iter))

  tst.assertEqual("  %d recipient(s) + %d control recipient(s):" %
                  (len(recipient_op_type_node_name_pairs),
                   len(ctrl_recipient_op_type_node_name_pairs)),
                  next(line_iter))

  # Check recipients, the order of which is not deterministic.
  tst.assertEqual("    %d recipient(s):" %
                  len(recipient_op_type_node_name_pairs), next(line_iter))

  t_recs = []
  for _ in recipient_op_type_node_name_pairs:
    line = next(line_iter)

    op_type, node_name = parse_op_and_node(line)
    t_recs.append((op_type, node_name))

  tst.assertItemsEqual(recipient_op_type_node_name_pairs, t_recs)

  # Check control recipients, the order of which is not deterministic.
  if ctrl_recipient_op_type_node_name_pairs:
    tst.assertEqual("", next(line_iter))

    tst.assertEqual("    %d control recipient(s):" %
                    len(ctrl_recipient_op_type_node_name_pairs),
                    next(line_iter))

    t_ctrl_recs = []
    for _ in ctrl_recipient_op_type_node_name_pairs:
      line = next(line_iter)

      op_type, node_name = parse_op_and_node(line)
      t_ctrl_recs.append((op_type, node_name))

    tst.assertItemsEqual(ctrl_recipient_op_type_node_name_pairs, t_ctrl_recs)

  # The order of multiple attributes can be non-deterministic.
  if attr_key_val_pairs:
    tst.assertEqual("", next(line_iter))

    tst.assertEqual("Node attributes:", next(line_iter))

    kv_pairs = []
    for key, val in attr_key_val_pairs:
      key = next(line_iter).strip().replace(":", "")

      val = next(line_iter).strip()

      kv_pairs.append((key, val))

      tst.assertEqual("", next(line_iter))

  if num_dumped_tensors is not None:
    tst.assertEqual("%d dumped tensor(s):" % num_dumped_tensors,
                    next(line_iter))
    tst.assertEqual("", next(line_iter))

    dump_timestamps_ms = []
    for _ in range(num_dumped_tensors):
      line = next(line_iter)

      tst.assertStartsWith(line.strip(), "Slot 0 @ DebugIdentity @")
      tst.assertTrue(line.strip().endswith(" ms"))

      dump_timestamp_ms = float(line.strip().split(" @ ")[-1].replace("ms", ""))
      tst.assertGreaterEqual(dump_timestamp_ms, 0.0)

      dump_timestamps_ms.append(dump_timestamp_ms)

    tst.assertEqual(sorted(dump_timestamps_ms), dump_timestamps_ms)

  if show_stack_trace:
    tst.assertEqual("", next(line_iter))
    tst.assertEqual("", next(line_iter))
    tst.assertEqual("Traceback of node construction:", next(line_iter))
    if stack_trace_available:
      try:
        depth_counter = 0
        while True:
          for i in range(5):
            line = next(line_iter)
            if i == 0:
              tst.assertEqual(depth_counter, int(line.split(":")[0]))
            elif i == 1:
              tst.assertStartsWith(line, "  Line:")
            elif i == 2:
              tst.assertStartsWith(line, "  Function:")
            elif i == 3:
              tst.assertStartsWith(line, "  Text:")
            elif i == 4:
              tst.assertEqual("", line)

          depth_counter += 1
      except StopIteration:
        tst.assertEqual(0, i)
    else:
      tst.assertEqual("(Unavailable because no Python graph has been loaded)",
                      next(line_iter))


def check_syntax_error_output(tst, out, command_prefix):
  # [FUNCTION DOC]
  # HELPER: shared checker for invalid-syntax tests.
  # Asserts the output lines are exactly the two-line message saying there was a syntax error for
  # the command prefix and suggesting the help command for it. Returns nothing.
  """Check RichTextLines output for valid command prefix but invalid syntax."""

  tst.assertEqual([
      "Syntax error for command: %s" % command_prefix,
      "For help, do \"help %s\"" % command_prefix
  ], out.lines)


def check_error_output(tst, out, command_prefix, args):
  # [FUNCTION DOC]
  # HELPER: shared checker for erroneous-command tests.
  # Asserts the output has more than two lines and the first line starts with the standard error
  # message naming the command prefix and the space-joined arguments. Returns nothing.
  """Check RichTextLines output from invalid/erroneous commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    command_prefix: The command prefix of the command that caused the error.
    args: The arguments (excluding prefix) of the command that caused the error.
  """

  tst.assertGreater(len(out.lines), 2)
  tst.assertStartsWith(out.lines[0],
                       "Error occurred during handling of command: %s %s" %
                       (command_prefix, " ".join(args)))


def check_main_menu(tst,
                    out,
                    list_tensors_enabled=False,
                    node_info_node_name=None,
                    print_tensor_node_name=None,
                    list_inputs_node_name=None,
                    list_outputs_node_name=None):
  # [FUNCTION DOC]
  # HELPER: shared main-menu checker for the command tests.
  # Asserts the output carries a main-menu annotation, that the list-tensors item is enabled or
  # not as requested, that each of the node-info, print-tensor, list-inputs and list-outputs
  # items is enabled and ends with the given node name when one is supplied (otherwise disabled),
  # and that the run-info and help items are enabled. Returns nothing.
  """Check the main menu annotation of an output."""

  tst.assertIn(debugger_cli_common.MAIN_MENU_KEY, out.annotations)

  menu = out.annotations[debugger_cli_common.MAIN_MENU_KEY]
  tst.assertEqual(list_tensors_enabled,
                  menu.caption_to_item("list_tensors").is_enabled())

  menu_item = menu.caption_to_item("node_info")
  if node_info_node_name:
    tst.assertTrue(menu_item.is_enabled())
    tst.assertTrue(menu_item.content.endswith(node_info_node_name))
  else:
    tst.assertFalse(menu_item.is_enabled())

  menu_item = menu.caption_to_item("print_tensor")
  if print_tensor_node_name:
    tst.assertTrue(menu_item.is_enabled())
    tst.assertTrue(menu_item.content.endswith(print_tensor_node_name))
  else:
    tst.assertFalse(menu_item.is_enabled())

  menu_item = menu.caption_to_item("list_inputs")
  if list_inputs_node_name:
    tst.assertTrue(menu_item.is_enabled())
    tst.assertTrue(menu_item.content.endswith(list_inputs_node_name))
  else:
    tst.assertFalse(menu_item.is_enabled())

  menu_item = menu.caption_to_item("list_outputs")
  if list_outputs_node_name:
    tst.assertTrue(menu_item.is_enabled())
    tst.assertTrue(menu_item.content.endswith(list_outputs_node_name))
  else:
    tst.assertFalse(menu_item.is_enabled())

  tst.assertTrue(menu.caption_to_item("run_info").is_enabled())
  tst.assertTrue(menu.caption_to_item("help").is_enabled())


def check_menu_item(tst, out, line_index, expected_begin, expected_end,
                    expected_command):
  # [FUNCTION DOC]
  # HELPER: shared menu-item checker for the command tests.
  # Looks through the styled segments of the given output line for a clickable menu item and
  # asserts its start and end columns and its command text match the expectations; fails if no
  # menu item is found. Returns nothing.
  attr_segs = out.font_attr_segs[line_index]

  found_menu_item = False
  for begin, end, attribute in attr_segs:
    attributes = [attribute] if not isinstance(attribute, list) else attribute
    menu_item = [attribute for attribute in attributes if
                 isinstance(attribute, debugger_cli_common.MenuItem)]
    if menu_item:
      tst.assertEqual(expected_begin, begin)
      tst.assertEqual(expected_end, end)
      tst.assertEqual(expected_command, menu_item[0].content)
      found_menu_item = True
      break
  tst.assertTrue(found_menu_item)


def create_analyzer_cli(dump):
  # [FUNCTION DOC]
  # HELPER: fixture builder shared by all test classes.
  # Takes a debug dump directory object. Creates an analyzer on it with a temporary-file
  # configuration, creates a command handler registry, and registers handlers (with short
  # aliases) for list tensors, node info, list inputs, list outputs, print tensor, print source,
  # list source and eval. Returns a pair: the analyzer and the registry.
  """Create an analyzer CLI.

  Args:
    dump: A `DebugDumpDir` object to base the analyzer CLI on.

  Returns:
    1) A `DebugAnalyzer` object created based on `dump`.
    2) A `CommandHandlerRegistry` that is based on the `DebugAnalyzer` object
       and has the common tfdbg commands, e.g., lt, ni, li, lo, registered.
  """
  # Construct the analyzer.
  analyzer = analyzer_cli.DebugAnalyzer(dump, _cli_config_from_temp_file())

  # Construct the handler registry.
  registry = debugger_cli_common.CommandHandlerRegistry()

  # Register command handlers.
  registry.register_command_handler(
      "list_tensors",
      analyzer.list_tensors,
      analyzer.get_help("list_tensors"),
      prefix_aliases=["lt"])
  registry.register_command_handler(
      "node_info",
      analyzer.node_info,
      analyzer.get_help("node_info"),
      prefix_aliases=["ni"])
  registry.register_command_handler(
      "list_inputs",
      analyzer.list_inputs,
      analyzer.get_help("list_inputs"),
      prefix_aliases=["li"])
  registry.register_command_handler(
      "list_outputs",
      analyzer.list_outputs,
      analyzer.get_help("list_outputs"),
      prefix_aliases=["lo"])
  registry.register_command_handler(
      "print_tensor",
      analyzer.print_tensor,
      analyzer.get_help("print_tensor"),
      prefix_aliases=["pt"])
  registry.register_command_handler(
      "print_source",
      analyzer.print_source,
      analyzer.get_help("print_source"),
      prefix_aliases=["ps"])
  registry.register_command_handler(
      "list_source",
      analyzer.list_source,
      analyzer.get_help("list_source"),
      prefix_aliases=["ls"])
  registry.register_command_handler(
      "eval",
      analyzer.evaluate_expression,
      analyzer.get_help("eval"),
      prefix_aliases=["ev"])

  return analyzer, registry


@test_util.run_v1_only("b/120545219")
class AnalyzerCLISimpleMulAddTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests of the analyzer commands on a small graph that multiplies two variables and adds the
  # result to itself. Class-level fixtures hold the dump directories, main device name, this
  # file's path, the session, the debug dump, the analyzer, the command registry, and the source
  # line numbers where the nodes were created. Covers tensor listing, sorting and filtering, node
  # info, tensor printing, expression evaluation, tensor filter management, and source printing
  # and listing. Runs only under session (version 1) mode.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture. Creates two temporary dump directories, detects whether a GPU exists
    # to choose the expected device name, records this file's path, builds a graph with two
    # variables, a matrix multiply and an add (recording the source line of each creation),
    # initializes the variables, runs the add with debug-identity watches on all nodes writing
    # to the first dump directory, and builds the debug dump, analyzer and command registry from
    # the result. Returns nothing.
    cls._dump_root = tempfile.mkdtemp()
    cls._dump_root_for_unique = tempfile.mkdtemp()

    cls._is_gpu_available = test.is_gpu_available()
    if cls._is_gpu_available:
      gpu_name = test_util.gpu_device_name()
      cls._main_device = "/job:localhost/replica:0/task:0" + gpu_name
    else:
      cls._main_device = "/job:localhost/replica:0/task:0/device:CPU:0"

    cls._curr_file_path = os.path.abspath(
        tf_inspect.getfile(tf_inspect.currentframe()))

    cls._sess = session.Session(config=no_rewrite_session_config())
    with cls._sess as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      v_init_val = np.array([[2.0], [-1.0]])

      u_name = "simple_mul_add/u"
      v_name = "simple_mul_add/v"

      u_init = constant_op.constant(u_init_val, shape=[2, 2], name="u_init")
      u = variable_v1.VariableV1(u_init, name=u_name)
      cls._u_line_number = line_number_above()

      v_init = constant_op.constant(v_init_val, shape=[2, 1], name="v_init")
      v = variable_v1.VariableV1(v_init, name=v_name)
      cls._v_line_number = line_number_above()

      w = math_ops.matmul(u, v, name="simple_mul_add/matmul")
      cls._w_line_number = line_number_above()

      x = math_ops.add(w, w, name="simple_mul_add/add")
      cls._x_line_number = line_number_above()

      a = variable_v1.VariableV1([1, 3, 3, 7], name="a")

      u.initializer.run()
      v.initializer.run()
      a.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls="file://%s" % cls._dump_root)

      # Invoke Session.run().
      run_metadata = config_pb2.RunMetadata()
      sess.run([x], options=run_options, run_metadata=run_metadata)
      cls._debug_dump = debug_data.DebugDumpDir(
          cls._dump_root, partition_graphs=run_metadata.partition_graphs)
      cls._analyzer, cls._registry = create_analyzer_cli(cls._debug_dump)

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Class-level cleanup. Recursively deletes both temporary dump directories. Returns nothing.
    # Tear down temporary dump directory.
    file_io.delete_recursively(cls._dump_root)
    file_io.delete_recursively(cls._dump_root_for_unique)

  def testMeasureTensorListColumnWidthsGivesRightAnswerForEmptyData(self):
    # [FUNCTION DOC]
    # Scenario: measure tensor-list column widths for an empty list of dumps. Expects each
    # width to equal the header text length of its column (time, size, op type) plus one.
    timestamp_col_width, dump_size_col_width, op_type_col_width = (
        self._analyzer._measure_tensor_list_column_widths([]))
    self.assertEqual(len("t (ms)") + 1, timestamp_col_width)
    self.assertEqual(len("Size (B)") + 1, dump_size_col_width)
    self.assertEqual(len("Op type") + 1, op_type_col_width)

  def testMeasureTensorListColumnWidthsGivesRightAnswerForData(self):
    # [FUNCTION DOC]
    # Scenario: measure column widths for the first dumped tensor (a small variable). Expects
    # the size column width to be set by its header (plus one) because the size text is short,
    # and the op type column width to be set by the longer variable op type name (plus one).
    dump = self._debug_dump.dumped_tensor_data[0]
    self.assertLess(dump.dump_size_bytes, 1000)
    self.assertEqual(
        "VariableV2", self._debug_dump.node_op_type(dump.node_name))
    _, dump_size_col_width, op_type_col_width = (
        self._analyzer._measure_tensor_list_column_widths([dump]))
    # The length of str(dump.dump_size_bytes) is less than the length of
    # "Size (B)" (8). So the column width should be determined by the length of
    # "Size (B)".
    self.assertEqual(len("Size (B)") + 1, dump_size_col_width)
    # The length of "VariableV2" is greater than the length of "Op type". So the
    # column should be determined by the length of "VariableV2".
    self.assertEqual(len("VariableV2") + 1, op_type_col_width)

  def testListTensors(self):
    # [FUNCTION DOC]
    # Scenario: run the list-tensors command via its short alias with no arguments. Expects all
    # six dumped tensors with their op types, sorted by timestamp, and a main menu with
    # list-tensors disabled.
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", [])

    assert_listed_tensors(self, out, [
        "simple_mul_add/u:0", "simple_mul_add/v:0", "simple_mul_add/u/read:0",
        "simple_mul_add/v/read:0", "simple_mul_add/matmul:0",
        "simple_mul_add/add:0"
    ], [
        "VariableV2", "VariableV2", "Identity", "Identity",
        _matmul_op_name(), "AddV2"
    ])

    # Check the main menu.
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInReverseTimeOrderWorks(self):
    # [FUNCTION DOC]
    # Scenario: list tensors sorted by timestamp with the reverse flag. Expects all six tensors
    # in descending timestamp order and the standard main menu.
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "timestamp", "-r"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ], [
            "VariableV2", "VariableV2", "Identity", "Identity",
            _matmul_op_name(), "AddV2"
        ],
        sort_by="timestamp",
        reverse=True)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInDumpSizeOrderWorks(self):
    # [FUNCTION DOC]
    # Scenario: list tensors sorted by dump size. Expects all six tensors in ascending dump size
    # order and the standard main menu.
    out = self._registry.dispatch_command("lt", ["-s", "dump_size"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ], [
            "VariableV2", "VariableV2", "Identity", "Identity",
            _matmul_op_name(), "AddV2"
        ],
        sort_by="dump_size")
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInReverseDumpSizeOrderWorks(self):
    # [FUNCTION DOC]
    # Scenario: list tensors sorted by dump size with the reverse flag. Expects all six tensors
    # in descending dump size order and the standard main menu.
    out = self._registry.dispatch_command("lt", ["-s", "dump_size", "-r"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ], [
            "VariableV2", "VariableV2", "Identity", "Identity",
            _matmul_op_name(), "AddV2"
        ],
        sort_by="dump_size",
        reverse=True)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsWithInvalidSortByFieldGivesError(self):
    # [FUNCTION DOC]
    # Scenario: list tensors with an unsupported sort key. Expects the output lines to include a
    # value-error message naming the unsupported key.
    out = self._registry.dispatch_command("lt", ["-s", "foobar"])
    self.assertIn("ValueError: Unsupported key to sort tensors by: foobar",
                  out.lines)

  def testListTensorsInOpTypeOrderWorks(self):
    # [FUNCTION DOC]
    # Scenario: list tensors sorted by op type. Expects all six tensors in ascending op type
    # order and the standard main menu.
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "op_type"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ], [
            "VariableV2", "VariableV2", "Identity", "Identity",
            _matmul_op_name(), "AddV2"
        ],
        sort_by="op_type",
        reverse=False)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInReverseOpTypeOrderWorks(self):
    # [FUNCTION DOC]
    # Scenario: list tensors sorted by op type with the reverse flag. Expects all six tensors in
    # descending op type order and the standard main menu.
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "op_type", "-r"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ], [
            "VariableV2", "VariableV2", "Identity", "Identity",
            _matmul_op_name(), "AddV2"
        ],
        sort_by="op_type",
        reverse=True)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInTensorNameOrderWorks(self):
    # [FUNCTION DOC]
    # Scenario: list tensors sorted by tensor name. Expects all six tensors in ascending name
    # order and the standard main menu.
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "tensor_name"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ], [
            "VariableV2", "VariableV2", "Identity", "Identity",
            _matmul_op_name(), "AddV2"
        ],
        sort_by="tensor_name",
        reverse=False)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInReverseTensorNameOrderWorks(self):
    # [FUNCTION DOC]
    # Scenario: list tensors sorted by tensor name with the reverse flag. Expects all six tensors
    # in descending name order and the standard main menu.
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "tensor_name", "-r"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ], [
            "VariableV2", "VariableV2", "Identity", "Identity",
            _matmul_op_name(), "AddV2"
        ],
        sort_by="tensor_name",
        reverse=True)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsFilterByNodeNameRegex(self):
    # [FUNCTION DOC]
    # Scenario: list tensors with a node-name regex filter, first via the long option (matching
    # the two read-identity tensors), then via the short option with a pattern matching nothing.
    # Expects the matching tensors (then an empty list) with the filter line shown, and the
    # standard main menu.
    out = self._registry.dispatch_command("list_tensors",
                                          ["--node_name_filter", ".*read.*"])
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/u/read:0", "simple_mul_add/v/read:0"],
        ["Identity", "Identity"],
        node_name_regex=".*read.*")

    out = self._registry.dispatch_command("list_tensors", ["-n", "^read"])
    assert_listed_tensors(self, out, [], [], node_name_regex="^read")
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorFilterByOpTypeRegex(self):
    # [FUNCTION DOC]
    # Scenario: list tensors with an op-type regex filter, first matching identity ops, then an
    # alternation of add and matrix-multiply. Expects only the matching tensors listed, with the
    # op type filter line shown, and the standard main menu.
    out = self._registry.dispatch_command("list_tensors",
                                          ["--op_type_filter", "Identity"])
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/u/read:0", "simple_mul_add/v/read:0"],
        ["Identity", "Identity"],
        op_type_regex="Identity")

    out = self._registry.dispatch_command(
        "list_tensors", ["-t", "(Add|" + _matmul_op_name() + ")"])
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/add:0", "simple_mul_add/matmul:0"],
        ["AddV2", _matmul_op_name()],
        op_type_regex=("(Add|" + _matmul_op_name() + ")"))
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorFilterByNodeNameRegexAndOpTypeRegex(self):
    # [FUNCTION DOC]
    # Scenario: list tensors with both an op-type regex and a node-name regex. Expects only the
    # add tensor listed, with both filter lines shown, and the standard main menu.
    out = self._registry.dispatch_command(
        "list_tensors", ["-t", "(Add|MatMul)", "-n", ".*add$"])
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/add:0"], ["AddV2"],
        node_name_regex=".*add$",
        op_type_regex="(Add|MatMul)")
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorWithFilterAndNodeNameExclusionWorks(self):
    # [FUNCTION DOC]
    # Scenario: register a tensor filter accepting tensors of shape two by one, then list
    # tensors with that filter plus a node-name exclusion pattern matching names containing the
    # letter v. Expects only the matrix-multiply and add tensors listed and the standard main
    # menu. Side effect: registers the filter on the shared analyzer.
    # First, create and register the filter.
    def is_2x1_vector(datum, tensor):
      # [FUNCTION DOC]
      # Tensor filter used by this test: ignores the dump datum and returns true when the tensor
      # shape is two by one.
      del datum  # Unused.
      return list(tensor.shape) == [2, 1]
    self._analyzer.add_tensor_filter("is_2x1_vector", is_2x1_vector)

    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command(
        "lt", ["-f", "is_2x1_vector", "--filter_exclude_node_names", ".*v.*"])

    # If the --filter_exclude_node_names were not used, then the matching
    # tensors would be:
    #   - simple_mul_add/v:0
    #   - simple_mul_add/v/read:0
    #   - simple_mul_add/matmul:0
    #   - simple_mul_add/add:0
    #
    # With the --filter_exclude_node_names option, only the last two should
    # show up in the result.
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/matmul:0", "simple_mul_add/add:0"],
        [_matmul_op_name(), "AddV2"],
        tensor_filter_name="is_2x1_vector")

    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsFilterNanOrInf(self):
    # [FUNCTION DOC]
    # Scenario: register the built-in infinity-or-NaN tensor filter and list tensors with it.
    # Expects an empty list (the run produced no bad values) with the filter name shown, and the
    # standard main menu. Side effect: registers the filter on the shared analyzer.
    """Test register and invoke a tensor filter."""

    # First, register the filter.
    self._analyzer.add_tensor_filter("has_inf_or_nan",
                                     debug_data.has_inf_or_nan)

    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-f", "has_inf_or_nan"])

    # This TF graph run did not generate any bad numerical values.
    assert_listed_tensors(
        self, out, [], [], tensor_filter_name="has_inf_or_nan")
    # TODO(cais): A test with some actual bad numerical values.

    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorNonexistentFilter(self):
    # [FUNCTION DOC]
    # Scenario: list tensors with a filter name that was never registered. Expects a single
    # error line saying there is no such tensor filter, and the standard main menu.
    """Test attempt to use a nonexistent tensor filter."""

    out = self._registry.dispatch_command("lt", ["-f", "foo_filter"])

    self.assertEqual(["ERROR: There is no tensor filter named \"foo_filter\"."],
                     out.lines)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInvalidOptions(self):
    # [FUNCTION DOC]
    # Scenario: run list-tensors with an unknown option. Expects the standard syntax-error
    # output for that command.
    out = self._registry.dispatch_command("list_tensors", ["--bar"])
    check_syntax_error_output(self, out, "list_tensors")

  def testNodeInfoByNodeName(self):
    # [FUNCTION DOC]
    # Scenario: run node-info on the matrix-multiply node by node name. Expects correct op type,
    # device, two identity inputs, two add recipients, a main menu with node-specific items
    # enabled, and the node name shown in bold on the first line.
    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", [node_name])

    recipients = [("AddV2", "simple_mul_add/add"),
                  ("AddV2", "simple_mul_add/add")]

    assert_node_attribute_lines(self, out, node_name, _matmul_op_name(),
                                self._main_device,
                                [("Identity", "simple_mul_add/u/read"),
                                 ("Identity", "simple_mul_add/v/read")], [],
                                recipients, [])
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

    # Verify that the node name is bold in the first line.
    self.assertEqual(
        [(len(out.lines[0]) - len(node_name), len(out.lines[0]), "bold")],
        out.font_attr_segs[0])

  def testNodeInfoShowAttributes(self):
    # [FUNCTION DOC]
    # Scenario: run node-info with the attributes option on the matrix-multiply node. Expects the
    # usual node description plus an attribute section (transpose flags and data type, plus a
    # kernel attribute when MKL is enabled) and the node-specific main menu.
    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", ["-a", node_name])

    test_attr_key_val_pairs = [("transpose_a", "b: false"),
                               ("transpose_b", "b: false"),
                               ("T", "type: DT_DOUBLE")]
    if test_util.IsMklEnabled():
      test_attr_key_val_pairs.append(("_kernel", 's: "MklNameChangeOp"'))

    assert_node_attribute_lines(
        self,
        out,
        node_name,
        _matmul_op_name(),
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("AddV2", "simple_mul_add/add"), ("AddV2", "simple_mul_add/add")], [],
        attr_key_val_pairs=test_attr_key_val_pairs)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testNodeInfoShowDumps(self):
    # [FUNCTION DOC]
    # Scenario: run node-info with the dumps option on the matrix-multiply node. Expects the
    # usual description plus one dumped tensor entry, the node-specific main menu, and a menu
    # item on output line 16 whose command prints the first dump of the node's output tensor.
    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", ["-d", node_name])

    assert_node_attribute_lines(
        self,
        out,
        node_name,
        _matmul_op_name(),
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("AddV2", "simple_mul_add/add"), ("AddV2", "simple_mul_add/add")], [],
        num_dumped_tensors=1)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 16,
                    len(out.lines[16]) - len(out.lines[16].strip()),
                    len(out.lines[16]), "pt %s:0 -n 0" % node_name)

  def testNodeInfoShowStackTraceUnavailableIsIndicated(self):
    # [FUNCTION DOC]
    # Scenario: clear the Python graph on the dump, then run node-info with the stack-trace
    # option. Expects the output to say the traceback is unavailable because no Python graph was
    # loaded, plus the usual description and main menu. Side effect: modifies the shared dump.
    self._debug_dump.set_python_graph(None)

    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", ["-t", node_name])

    assert_node_attribute_lines(
        self,
        out,
        node_name,
        _matmul_op_name(),
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("AddV2", "simple_mul_add/add"), ("AddV2", "simple_mul_add/add")], [],
        show_stack_trace=True,
        stack_trace_available=False)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testNodeInfoShowStackTraceAvailableWorks(self):
    # [FUNCTION DOC]
    # Scenario: set the session's Python graph on the dump, then run node-info with the
    # stack-trace option. Expects a traceback section of well-formed multi-line records plus the
    # usual description and main menu. Side effect: modifies the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)

    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", ["-t", node_name])

    assert_node_attribute_lines(
        self,
        out,
        node_name,
        _matmul_op_name(),
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("AddV2", "simple_mul_add/add"), ("AddV2", "simple_mul_add/add")], [],
        show_stack_trace=True,
        stack_trace_available=True)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testNodeInfoByTensorName(self):
    # [FUNCTION DOC]
    # Scenario: run node-info giving a tensor name (node name plus output index) for the read
    # identity node. Expects the node's description with one variable input and the matrix
    # multiply recipient, and the node-specific main menu.
    node_name = "simple_mul_add/u/read"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command("node_info", [tensor_name])

    assert_node_attribute_lines(self, out, node_name, "Identity",
                                self._main_device,
                                [("VariableV2", "simple_mul_add/u")], [],
                                [(_matmul_op_name(), "simple_mul_add/matmul")],
                                [])
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testNodeInfoNonexistentNodeName(self):
    # [FUNCTION DOC]
    # Scenario: run node-info on a node name that does not exist. Expects a single error line
    # colored red over its whole length, and a main menu with list-tensors enabled.
    out = self._registry.dispatch_command("node_info", ["bar"])
    self.assertEqual(
        ["ERROR: There is no node named \"bar\" in the partition graphs"],
        out.lines)
    # Check color indicating error.
    self.assertEqual({0: [(0, 59, cli_shared.COLOR_RED)]}, out.font_attr_segs)
    check_main_menu(self, out, list_tensors_enabled=True)

  def testPrintTensor(self):
    # [FUNCTION DOC]
    # Scenario: print the matrix-multiply output tensor by tensor name. Expects a header line,
    # data type and shape lines, a blank line and the array values; expects tensor-metadata and
    # per-line annotations, and the node-specific main menu.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name], screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\":" % tensor_name,
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorAndWriteToNpyFile(self):
    # [FUNCTION DOC]
    # Scenario: print the matrix-multiply output tensor with the write-to-file option pointing at
    # a numpy file in the dump directory. Expects the header, type and shape lines, then a line
    # saying where the value was saved, and the saved file loads with values close to the
    # expected array. Side effect: writes a file in the dump directory.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    npy_path = os.path.join(self._dump_root, "matmul.npy")
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "-w", npy_path],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\":" % tensor_name,
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
    ], out.lines[:4])
    self.assertTrue(out.lines[4].startswith("Saved value to: %s (" % npy_path))
    # Load the numpy file and verify its contents.
    self.assertAllClose([[7.0], [-2.0]], np.load(npy_path))

  def testPrintTensorHighlightingRanges(self):
    # [FUNCTION DOC]
    # Scenario: print the tensor with the ranges option, first a single range of non-positive
    # values, then two ranges. Expects the header to state the highlighted ranges and count
    # (one of two elements, fifty percent), the values unchanged, annotations present, and bold
    # font segments on the line of the highlighted element (and none on the other in the second
    # case); the main menu is checked at the end.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "--ranges", "[-inf, 0.0]"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\": " % tensor_name +
        "Highlighted([-inf, 0.0]): 1 of 2 element(s) (50.00%)",
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)
    self.assertEqual([(8, 11, "bold")], out.font_attr_segs[5])

    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "--ranges", "[[-inf, -5.5], [5.5, inf]]"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\": " % tensor_name +
        "Highlighted([[-inf, -5.5], [5.5, inf]]): "
        "1 of 2 element(s) (50.00%)",
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)
    self.assertEqual([(9, 11, "bold")], out.font_attr_segs[4])
    self.assertNotIn(5, out.font_attr_segs)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorHighlightingRangesAndIncludingNumericSummary(self):
    # [FUNCTION DOC]
    # Scenario: print the tensor with the ranges option and the numeric-summary option. Expects
    # the highlighted-range header, a numeric summary block (counts of negative and positive
    # values, total, minimum, maximum, mean and standard deviation), the array values,
    # annotations at the expected line indices and a bold segment on the highlighted element line.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "--ranges", "[-inf, 0.0]", "-s"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\": " % tensor_name +
        "Highlighted([-inf, 0.0]): 1 of 2 element(s) (50.00%)",
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "Numeric summary:",
        "| - + | total |",
        "| 1 1 |     2 |",
        "|  min  max mean  std |",
        "| -2.0  7.0  2.5  4.5 |",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(10, out.annotations)
    self.assertIn(11, out.annotations)
    self.assertEqual([(8, 11, "bold")], out.font_attr_segs[11])

  def testPrintTensorWithSlicing(self):
    # [FUNCTION DOC]
    # Scenario: print the tensor with a slicing suffix selecting its second row. Expects the
    # header to include the slice, shape of one element, the sliced values, a metadata
    # annotation and the node-specific main menu.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name + "[1, :]"], screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity[1, :]\":" % tensor_name, "  dtype: float64",
        "  shape: (1,)", "", "array([-2.])"
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorInvalidSlicingString(self):
    # [FUNCTION DOC]
    # Scenario: print the tensor with a slicing suffix that contains a function call. Expects the
    # generic error header naming the command and a value-error line saying the tensor-slicing
    # string is invalid.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name + "[1, foo()]"], screen_info={"cols": 80})

    self.assertEqual("Error occurred during handling of command: print_tensor "
                     + tensor_name + "[1, foo()]:", out.lines[0])
    self.assertEqual("ValueError: Invalid tensor-slicing string.",
                     out.lines[-2])

  def testPrintTensorValidExplicitNumber(self):
    # [FUNCTION DOC]
    # Scenario: print the tensor with the dump-number option set to zero. Expects the same output
    # as the default printing, annotations present and the node-specific main menu.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "-n", "0"], screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\":" % tensor_name,
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorInvalidExplicitNumber(self):
    # [FUNCTION DOC]
    # Scenario: print the tensor with the dump-number option set to one though only one dump
    # exists. Expects a single error line about the invalid number, no tensor-metadata
    # annotation and the node-specific main menu.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "-n", "1"], screen_info={"cols": 80})

    self.assertEqual([
        "ERROR: Invalid number (1) for tensor simple_mul_add/matmul:0, "
        "which generated one dump."
    ], out.lines)

    self.assertNotIn("tensor_metadata", out.annotations)

    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorMissingOutputSlotLeadsToOnlyDumpedTensorPrinted(self):
    # [FUNCTION DOC]
    # Scenario: print using only a node name without an output slot. Expects the node's single
    # dumped tensor (slot zero) to be printed with its full header and values, and the
    # node-specific main menu.
    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("print_tensor", [node_name])

    self.assertEqual([
        "Tensor \"%s:0:DebugIdentity\":" % node_name, "  dtype: float64",
        "  shape: (2, 1)", "", "array([[ 7.],", "       [-2.]])"
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorNonexistentNodeName(self):
    # [FUNCTION DOC]
    # Scenario: print a tensor whose node does not exist. Expects a single error line saying the
    # node is not in the partition graphs, and a main menu with list-tensors enabled.
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul/foo:0"])

    self.assertEqual([
        "ERROR: Node \"simple_mul_add/matmul/foo\" does not exist in partition "
        "graphs"
    ], out.lines)
    check_main_menu(self, out, list_tensors_enabled=True)

  def testEvalExpression(self):
    # [FUNCTION DOC]
    # Scenario: evaluate an expression multiplying the output tensor by its transpose, with the
    # tensor referenced inside backquotes. Expects a header quoting the expression, type and
    # shape lines, a numeric summary, and the summary statistics and array values close to the
    # expected numbers (compared ignoring whitespace).
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "eval", ["np.matmul(`%s`, `%s`.T)" % (tensor_name, tensor_name)],
        screen_info={"cols": 80})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["Tensor \"from eval of expression "
         "'np.matmul(`simple_mul_add/matmul:0`, "
         "`simple_mul_add/matmul:0`.T)'\":",
         "  dtype: float64",
         "  shape: (2, 2)",
         "",
         "Numeric summary:",
         "| - + | total |",
         "| 2 2 |     4 |",
         "|           min           max          mean           std |"],
        out.lines[:8])
    cli_test_utils.assert_array_lines_close(
        self, [-14.0, 49.0, 6.25, 25.7524270701], out.lines[8:9])
    cli_test_utils.assert_array_lines_close(
        self, [[49.0, -14.0], [-14.0, 4.0]], out.lines[10:])

  def testEvalExpressionAndWriteToNpyFile(self):
    # [FUNCTION DOC]
    # Scenario: evaluate the same expression with the write-to-file option. Expects the header,
    # type and shape lines, a saved-to line, and the numpy file loads with values close to the
    # expected matrix. Side effect: writes a file in the dump directory.
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    npy_path = os.path.join(self._dump_root, "matmul_eval.npy")
    out = self._registry.dispatch_command(
        "eval",
        ["np.matmul(`%s`, `%s`.T)" % (tensor_name, tensor_name), "-w",
         npy_path], screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"from eval of expression "
        "'np.matmul(`simple_mul_add/matmul:0`, "
        "`simple_mul_add/matmul:0`.T)'\":",
        "  dtype: float64",
        "  shape: (2, 2)",
        ""], out.lines[:4])

    self.assertTrue(out.lines[4].startswith("Saved value to: %s (" % npy_path))
    # Load the numpy file and verify its contents.
    self.assertAllClose([[49.0, -14.0], [-14.0, 4.0]], np.load(npy_path))

  def testAddGetTensorFilterLambda(self):
    # [FUNCTION DOC]
    # Scenario: add an anonymous-function tensor filter to a fresh analyzer and get it back by
    # name. Expects the retrieved filter to be callable and return true.
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump,
                                          _cli_config_from_temp_file())
    analyzer.add_tensor_filter("foo_filter", lambda x, y: True)
    self.assertTrue(analyzer.get_tensor_filter("foo_filter")(None, None))

  def testAddGetTensorFilterNestedFunction(self):
    # [FUNCTION DOC]
    # Scenario: add a filter defined as a nested named function to a fresh analyzer and get it
    # back. Expects the retrieved filter to return true.
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump,
                                          _cli_config_from_temp_file())

    def foo_filter(unused_arg_0, unused_arg_1):
      # [FUNCTION DOC]
      # Test tensor filter that ignores both arguments and always returns true.
      return True

    analyzer.add_tensor_filter("foo_filter", foo_filter)
    self.assertTrue(analyzer.get_tensor_filter("foo_filter")(None, None))

  def testAddTensorFilterEmptyName(self):
    # [FUNCTION DOC]
    # Scenario: add a tensor filter with an empty name. Expects a value error saying the filter
    # name cannot be empty.
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump,
                                          _cli_config_from_temp_file())

    with self.assertRaisesRegex(ValueError,
                                "Input argument filter_name cannot be empty."):
      analyzer.add_tensor_filter("", lambda datum, tensor: True)

  def testAddTensorFilterNonStrName(self):
    # [FUNCTION DOC]
    # Scenario: add a tensor filter whose name is an integer. Expects a type error saying the
    # filter name must be a string.
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump,
                                          _cli_config_from_temp_file())

    with self.assertRaisesRegex(
        TypeError, "Input argument filter_name is expected to be str, "
        "but is not"):
      analyzer.add_tensor_filter(1, lambda datum, tensor: True)

  def testAddGetTensorFilterNonCallable(self):
    # [FUNCTION DOC]
    # Scenario: add a tensor filter whose callable argument is a string. Expects a type error
    # saying the filter must be callable.
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump,
                                          _cli_config_from_temp_file())

    with self.assertRaisesRegex(
        TypeError, "Input argument filter_callable is expected to be callable, "
        "but is not."):
      analyzer.add_tensor_filter("foo_filter", "bar")

  def testGetNonexistentTensorFilter(self):
    # [FUNCTION DOC]
    # Scenario: register one filter, then get a filter under a different name. Expects a value
    # error saying there is no tensor filter with that name.
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump,
                                          _cli_config_from_temp_file())

    analyzer.add_tensor_filter("foo_filter", lambda datum, tensor: True)
    with self.assertRaisesRegex(ValueError,
                                "There is no tensor filter named \"bar\""):
      analyzer.get_tensor_filter("bar")

  def _findSourceLine(self, annotated_source, line_number):
    # [FUNCTION DOC]
    # HELPER: locates an annotated source line for the print-source tests.
    # Takes the annotated-source rich-text output and a one-based source line number. Returns the
    # zero-based index of the first output line that begins with the line-number label for that
    # number, or nothing if none does.
    """Find line of given line number in annotated source.

    Args:
      annotated_source: (debugger_cli_common.RichTextLines) the annotated source
      line_number: (int) 1-based line number

    Returns:
      (int) If line_number is found, 0-based line index in
        annotated_source.lines. Otherwise, None.
    """

    index = None
    for i, line in enumerate(annotated_source.lines):
      if line.startswith("L%d " % line_number):
        index = i
        break
    return index

  def testPrintSourceForOpNamesWholeFileWorks(self):
    # [FUNCTION DOC]
    # Scenario: with the session graph attached to the dump, print this test file's source
    # annotated with op names. For the lines creating the two variables, the matrix multiply and
    # the add, expects the source line followed by the op names created there, each with a
    # print-tensor menu command; the unused assign op is colored blue. Side effect: sets the
    # Python graph on the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "print_source", [self._curr_file_path], screen_info={"cols": 80})

    # Verify the annotation of the line that creates u.
    index = self._findSourceLine(out, self._u_line_number)
    self.assertEqual(
        ["L%d         u = variable_v1.VariableV1(u_init, name=u_name)" %
         self._u_line_number,
         "    simple_mul_add/u",
         "    simple_mul_add/u/Assign",
         "    simple_mul_add/u/read"],
        out.lines[index : index + 4])
    self.assertEqual("pt simple_mul_add/u",
                     out.font_attr_segs[index + 1][0][2].content)
    # simple_mul_add/u/Assign is not used in this run because the Variable has
    # already been initialized.
    self.assertEqual(cli_shared.COLOR_BLUE, out.font_attr_segs[index + 2][0][2])
    self.assertEqual("pt simple_mul_add/u/read",
                     out.font_attr_segs[index + 3][0][2].content)

    # Verify the annotation of the line that creates v.
    index = self._findSourceLine(out, self._v_line_number)
    self.assertEqual(
        ["L%d         v = variable_v1.VariableV1(v_init, name=v_name)" %
         self._v_line_number,
         "    simple_mul_add/v"],
        out.lines[index : index + 2])
    self.assertEqual("pt simple_mul_add/v",
                     out.font_attr_segs[index + 1][0][2].content)

    # Verify the annotation of the line that creates w.
    index = self._findSourceLine(out, self._w_line_number)
    self.assertEqual(
        ["L%d         " % self._w_line_number +
         "w = math_ops.matmul(u, v, name=\"simple_mul_add/matmul\")",
         "    simple_mul_add/matmul"],
        out.lines[index : index + 2])
    self.assertEqual("pt simple_mul_add/matmul",
                     out.font_attr_segs[index + 1][0][2].content)

    # Verify the annotation of the line that creates x.
    index = self._findSourceLine(out, self._x_line_number)
    self.assertEqual(
        ["L%d         " % self._x_line_number +
         "x = math_ops.add(w, w, name=\"simple_mul_add/add\")",
         "    simple_mul_add/add"],
        out.lines[index : index + 2])
    self.assertEqual("pt simple_mul_add/add",
                     out.font_attr_segs[index + 1][0][2].content)

  def testPrintSourceForTensorNamesWholeFileWorks(self):
    # [FUNCTION DOC]
    # Scenario: print this file's source annotated with tensor names (tensors option). For the
    # line creating the first variable, expects the two tensor names listed below it, each with a
    # print-tensor menu command. Side effect: sets the Python graph on the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "print_source",
        [self._curr_file_path, "--tensors"],
        screen_info={"cols": 80})

    # Verify the annotation of the line that creates u.
    index = self._findSourceLine(out, self._u_line_number)
    self.assertEqual(
        ["L%d         u = variable_v1.VariableV1(u_init, name=u_name)" %
         self._u_line_number,
         "    simple_mul_add/u/read:0",
         "    simple_mul_add/u:0"],
        out.lines[index : index + 3])
    self.assertEqual("pt simple_mul_add/u/read:0",
                     out.font_attr_segs[index + 1][0][2].content)
    self.assertEqual("pt simple_mul_add/u:0",
                     out.font_attr_segs[index + 2][0][2].content)

  def testPrintSourceForOpNamesStartingAtSpecifiedLineWorks(self):
    # [FUNCTION DOC]
    # Scenario: print source annotated with op names, starting at line three. Expects the initial
    # scroll position annotation to be two, and the variable creation line still annotated with
    # its op names and menu commands. Side effect: sets the Python graph on the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "print_source",
        [self._curr_file_path, "-b", "3"],
        screen_info={"cols": 80})

    self.assertEqual(
        2, out.annotations[debugger_cli_common.INIT_SCROLL_POS_KEY])

    index = self._findSourceLine(out, self._u_line_number)
    self.assertEqual(
        ["L%d         u = variable_v1.VariableV1(u_init, name=u_name)" %
         self._u_line_number,
         "    simple_mul_add/u",
         "    simple_mul_add/u/Assign",
         "    simple_mul_add/u/read"],
        out.lines[index : index + 4])
    self.assertEqual("pt simple_mul_add/u",
                     out.font_attr_segs[index + 1][0][2].content)
    # simple_mul_add/u/Assign is not used in this run because the Variable has
    # already been initialized.
    self.assertEqual(cli_shared.COLOR_BLUE, out.font_attr_segs[index + 2][0][2])
    self.assertEqual("pt simple_mul_add/u/read",
                     out.font_attr_segs[index + 3][0][2].content)

  def testPrintSourceForOpNameSettingMaximumElementCountWorks(self):
    # [FUNCTION DOC]
    # Scenario: print source with a maximum of one annotation element per line. Expects the
    # variable creation line to show one op name and an omitted-count line, whose last menu
    # command re-runs print-source for this file with a larger maximum. Side effect: sets the
    # Python graph on the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "print_source",
        [self._curr_file_path, "-m", "1"],
        screen_info={"cols": 80})

    index = self._findSourceLine(out, self._u_line_number)
    self.assertEqual(
        ["L%d         u = variable_v1.VariableV1(u_init, name=u_name)" %
         self._u_line_number,
         "    simple_mul_add/u",
         "    (... Omitted 2 of 3 op(s) ...) +5"],
        out.lines[index : index + 3])
    self.assertEqual("pt simple_mul_add/u",
                     out.font_attr_segs[index + 1][0][2].content)
    more_elements_command = out.font_attr_segs[index + 2][-1][2].content
    self.assertStartsWith(more_elements_command,
                          "ps %s " % self._curr_file_path)
    self.assertIn(" -m 6", more_elements_command)

  def testListSourceWorks(self):
    # [FUNCTION DOC]
    # Scenario: run list-source with no filters. Expects this test file among the listed
    # non-library source files, and every line after the library-files header to be colored gray.
    # Side effect: sets the Python graph on the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command("list_source", [])

    non_tf_lib_files_start = [
        i for i in range(len(out.lines))
        if out.lines[i].startswith("Source file path")
    ][0] + 1
    non_tf_lib_files_end = [
        i for i in range(len(out.lines))
        if out.lines[i].startswith("TensorFlow Python library file(s):")
    ][0] - 1
    non_tf_lib_files = [
        line.split(" ")[0] for line
        in out.lines[non_tf_lib_files_start : non_tf_lib_files_end]]
    self.assertIn(self._curr_file_path, non_tf_lib_files)

    # Check that the TF library files are marked with special color attribute.
    for i in range(non_tf_lib_files_end + 1, len(out.lines)):
      if not out.lines[i]:
        continue
      for attr_seg in  out.font_attr_segs[i]:
        self.assertTrue(cli_shared.COLOR_GRAY in attr_seg[2] or
                        attr_seg[2] == cli_shared.COLOR_GRAY)

  def testListSourceWithNodeNameFilterWithMatchesWorks(self):
    # [FUNCTION DOC]
    # Scenario: run list-source with a node-name filter that matches read nodes. Expects the
    # filter echoed on the second line, this test file among the listed files, and library files
    # colored gray. Side effect: sets the Python graph on the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command("list_source", ["-n", ".*/read"])

    self.assertStartsWith(out.lines[1], "Node name regex filter: \".*/read\"")

    non_tf_lib_files_start = [
        i for i in range(len(out.lines))
        if out.lines[i].startswith("Source file path")
    ][0] + 1
    non_tf_lib_files_end = [
        i for i in range(len(out.lines))
        if out.lines[i].startswith("TensorFlow Python library file(s):")
    ][0] - 1
    non_tf_lib_files = [
        line.split(" ")[0] for line
        in out.lines[non_tf_lib_files_start : non_tf_lib_files_end]]
    self.assertIn(self._curr_file_path, non_tf_lib_files)

    # Check that the TF library files are marked with special color attribute.
    for i in range(non_tf_lib_files_end + 1, len(out.lines)):
      if not out.lines[i]:
        continue
      for attr_seg in  out.font_attr_segs[i]:
        self.assertTrue(cli_shared.COLOR_GRAY in attr_seg[2] or
                        attr_seg[2] == cli_shared.COLOR_GRAY)

  def testListSourceWithNodeNameFilterWithNoMatchesWorks(self):
    # [FUNCTION DOC]
    # Scenario: run list-source with a node-name filter that matches nothing. Expects exactly
    # four lines: title, filter echo, blank, and a no-source-information message. Side effect:
    # sets the Python graph on the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command("list_source", ["-n", "^$"])

    self.assertEqual([
        "List of source files that created nodes in this run",
        "Node name regex filter: \"^$\"", "",
        "[No source file information.]"], out.lines)

  def testListSourceWithPathAndNodeNameFiltersWorks(self):
    # [FUNCTION DOC]
    # Scenario: run list-source with both a file-path filter (this file) and a node-name filter.
    # Expects the first four lines to be the title, both filter echoes and a blank line. Side
    # effect: sets the Python graph on the shared dump.
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "list_source", ["-p", self._curr_file_path, "-n", ".*read"])

    self.assertEqual([
        "List of source files that created nodes in this run",
        "File path regex filter: \"%s\"" % self._curr_file_path,
        "Node name regex filter: \".*read\"", ""], out.lines[:4])

  def testListSourceWithCompiledPythonSourceWorks(self):
    # [FUNCTION DOC]
    # Scenario: replace the source-file listing function with a fake returning two compiled
    # files and one regular file, then run list-source. Expects the compiled files shown in white
    # with no navigation command, and the regular file in white with a print-source command
    # starting at its first-node line. Side effect: temporarily patches the source utilities.
    def fake_list_source_files_against_dump(dump,
                                            path_regex_allowlist=None,
                                            node_name_regex_allowlist=None):
      # [FUNCTION DOC]
      # Fake replacement for the source-file listing function: ignores all arguments and returns
      # a fixed list of tuples describing two compiled files and one regular source file.
      del dump, path_regex_allowlist, node_name_regex_allowlist
      return [("compiled_1.pyc", False, 10, 20, 30, 4),
              ("compiled_2.pyo", False, 10, 20, 30, 5),
              ("uncompiled.py", False, 10, 20, 30, 6)]

    with test.mock.patch.object(
        source_utils, "list_source_files_against_dump",
        side_effect=fake_list_source_files_against_dump):
      out = self._registry.dispatch_command("list_source", [])

      self.assertStartsWith(out.lines[4], "compiled_1.pyc")
      self.assertEqual((0, 14, [cli_shared.COLOR_WHITE]),
                       out.font_attr_segs[4][0])
      self.assertStartsWith(out.lines[5], "compiled_2.pyo")
      self.assertEqual((0, 14, [cli_shared.COLOR_WHITE]),
                       out.font_attr_segs[5][0])
      self.assertStartsWith(out.lines[6], "uncompiled.py")
      self.assertEqual(0, out.font_attr_segs[6][0][0])
      self.assertEqual(13, out.font_attr_segs[6][0][1])
      self.assertEqual(cli_shared.COLOR_WHITE, out.font_attr_segs[6][0][2][0])
      self.assertEqual("ps uncompiled.py -b 6",
                       out.font_attr_segs[6][0][2][1].content)

  def testListInputInvolvingNodesWithMultipleOutputs(self):
    # [FUNCTION DOC]
    # Scenario: in a new session, build a variable feeding a unique op (which has two outputs)
    # and a multiply of its second output, run with debug watches writing to the second dump
    # directory, build an analyzer on the dump, and list inputs of the multiply. Expects the
    # result to name the non-first output slot of the unique op as the input.
    """List an input tree containing tensors from non-:0 output slot."""

    with session.Session(config=no_rewrite_session_config()) as sess:
      with ops.device("CPU:0"):
        x = variable_v1.VariableV1([1, 3, 3, 7], name="x")
        _, idx = array_ops.unique(x, name="x_unique")
        idx_times_two = math_ops.multiply(idx, 2, name="idx_times_two")
        self.evaluate(x.initializer)

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_utils.watch_graph(
            run_options,
            sess.graph,
            debug_ops=["DebugIdentity"],
            debug_urls="file://%s" % self._dump_root_for_unique)
        run_metadata = config_pb2.RunMetadata()
        self.assertAllEqual([0, 2, 2, 4],
                            sess.run(
                                idx_times_two,
                                options=run_options,
                                run_metadata=run_metadata))
        debug_dump = debug_data.DebugDumpDir(
            self._dump_root_for_unique,
            partition_graphs=run_metadata.partition_graphs)
        _, registry = create_analyzer_cli(debug_dump)

        out = registry.dispatch_command("li", ["idx_times_two"])
        self.assertEqual([
            "Inputs to node \"idx_times_two\" (Depth limit = 1):",
            "|- (1) x_unique:1"
        ], out.lines[:2])


class AnalyzerCLIPrintLargeTensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests of printing a tensor large enough (2400 elements) to exceed the default summarization
  # threshold. Fixtures: a temporary dump directory, the debug dump, the analyzer and the
  # command registry, built once for the class.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture. Creates a temporary dump directory, builds a graph with one large
    # constant of zeros, runs it with a debug-identity watch writing to the directory, and
    # builds the debug dump, analyzer and command registry. Returns nothing.
    cls._dump_root = tempfile.mkdtemp()

    with session.Session(config=no_rewrite_session_config()) as sess:
      # 2400 elements should exceed the default threshold (2000).
      x = constant_op.constant(np.zeros([300, 8]), name="large_tensors/x")

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls="file://%s" % cls._dump_root)

      # Invoke Session.run().
      run_metadata = config_pb2.RunMetadata()
      sess.run(x, options=run_options, run_metadata=run_metadata)

    cls._debug_dump = debug_data.DebugDumpDir(
        cls._dump_root, partition_graphs=run_metadata.partition_graphs)

    # Construct the analyzer and command registry.
    cls._analyzer, cls._registry = create_analyzer_cli(cls._debug_dump)

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Class-level cleanup. Recursively deletes the temporary dump directory. Returns nothing.
    # Tear down temporary dump directory.
    file_io.delete_recursively(cls._dump_root)

  def testPrintLargeTensorWithoutAllOption(self):
    # [FUNCTION DOC]
    # Scenario: print the large tensor without the all option, once whole and once sliced to
    # 2100 elements (still above the threshold). Expects an ellipsis marker in the value text.
    out = self._registry.dispatch_command(
        "print_tensor", ["large_tensors/x:0"], screen_info={"cols": 80})

    # Assert that ellipses are present in the tensor value printout.
    self.assertIn("...,", out.lines[4])

    # 2100 still exceeds 2000.
    out = self._registry.dispatch_command(
        "print_tensor", ["large_tensors/x:0[:, 0:7]"],
        screen_info={"cols": 80})

    self.assertIn("...,", out.lines[4])

  def testPrintLargeTensorWithAllOption(self):
    # [FUNCTION DOC]
    # Scenario: print the large tensor with the all option (short form, then long form on a
    # slice). Expects no ellipsis marker in the value text.
    out = self._registry.dispatch_command(
        "print_tensor", ["large_tensors/x:0", "-a"],
        screen_info={"cols": 80})

    # Assert that ellipses are not present in the tensor value printout.
    self.assertNotIn("...,", out.lines[4])

    out = self._registry.dispatch_command(
        "print_tensor", ["large_tensors/x:0[:, 0:7]", "--all"],
        screen_info={"cols": 80})
    self.assertNotIn("...,", out.lines[4])


@test_util.run_v1_only("b/120545219")
class AnalyzerCLIControlDepTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests of node-info, list-inputs and list-outputs commands on a graph with control
  # dependencies (a variable, an add, a multiply and identity nodes with control inputs).
  # Fixtures: a temporary dump directory, the main device name and the command registry, built
  # once for the class. Runs only under session (version 1) mode.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture. Creates a temporary dump directory, detects whether a GPU exists to
    # choose the expected device name, builds a graph in which an add and a multiply are wrapped
    # by identity nodes that carry control dependencies on the variable and earlier nodes, runs
    # it with debug-identity watches, and builds the command registry from the dump. Returns
    # nothing.
    cls._dump_root = tempfile.mkdtemp()

    cls._is_gpu_available = test.is_gpu_available()
    if cls._is_gpu_available:
      gpu_name = test_util.gpu_device_name()
      cls._main_device = "/job:localhost/replica:0/task:0" + gpu_name
    else:
      cls._main_device = "/job:localhost/replica:0/task:0/device:CPU:0"

    with session.Session(config=no_rewrite_session_config()) as sess:
      x_init_val = np.array([5.0, 3.0])
      x_init = constant_op.constant(x_init_val, shape=[2])
      x = variable_v1.VariableV1(x_init, name="control_deps/x")

      y = math_ops.add(x, x, name="control_deps/y")
      y = control_flow_ops.with_dependencies(
          [x], y, name="control_deps/ctrl_dep_y")

      z = math_ops.multiply(x, y, name="control_deps/z")

      z = control_flow_ops.with_dependencies(
          [x, y], z, name="control_deps/ctrl_dep_z")

      x.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls="file://%s" % cls._dump_root)

      # Invoke Session.run().
      run_metadata = config_pb2.RunMetadata()
      sess.run(z, options=run_options, run_metadata=run_metadata)

    debug_dump = debug_data.DebugDumpDir(
        cls._dump_root, partition_graphs=run_metadata.partition_graphs)

    # Construct the analyzer and command handler registry.
    _, cls._registry = create_analyzer_cli(debug_dump)

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Class-level cleanup. Recursively deletes the temporary dump directory. Returns nothing.
    # Tear down temporary dump directory.
    file_io.delete_recursively(cls._dump_root)

  def testNodeInfoWithControlDependencies(self):
    # [FUNCTION DOC]
    # Scenario: run node-info on a node with a control input, then on the variable node which has
    # control recipients. Expects the control inputs and recipients to be listed correctly, and
    # menu items on the recipient lines whose commands re-run node-info with all options for
    # that node (the order of the two control recipients may vary, so the line positions are
    # determined at run time).
    # Call node_info on a node with control inputs.
    out = self._registry.dispatch_command("node_info",
                                          ["control_deps/ctrl_dep_y"])

    assert_node_attribute_lines(self, out, "control_deps/ctrl_dep_y",
                                "Identity", self._main_device,
                                [("AddV2", "control_deps/y")],
                                [("VariableV2", "control_deps/x")],
                                [("Mul", "control_deps/z")],
                                [("Identity", "control_deps/ctrl_dep_z")])

    # Call node info on a node with control recipients.
    out = self._registry.dispatch_command("ni", ["control_deps/x"])

    assert_node_attribute_lines(self, out, "control_deps/x", "VariableV2",
                                self._main_device, [], [],
                                [("Identity", "control_deps/x/read")],
                                [("Identity", "control_deps/ctrl_dep_y"),
                                 ("Identity", "control_deps/ctrl_dep_z")])

    # Verify the menu items (command shortcuts) in the output.
    check_menu_item(self, out, 10,
                    len(out.lines[10]) - len("control_deps/x/read"),
                    len(out.lines[10]), "ni -a -d -t control_deps/x/read")
    if out.lines[13].endswith("control_deps/ctrl_dep_y"):
      y_line = 13
      z_line = 14
    else:
      y_line = 14
      z_line = 13
    check_menu_item(self, out, y_line,
                    len(out.lines[y_line]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[y_line]),
                    "ni -a -d -t control_deps/ctrl_dep_y")
    check_menu_item(self, out, z_line,
                    len(out.lines[z_line]) - len("control_deps/ctrl_dep_z"),
                    len(out.lines[z_line]),
                    "ni -a -d -t control_deps/ctrl_dep_z")

  def testListInputsNonRecursiveNoControl(self):
    # [FUNCTION DOC]
    # Scenario: list inputs of the multiply node non-recursively without control inputs, first
    # without and then with op types. Expects depth-one trees with truncated deeper levels and a
    # legend (with an op-type legend line in the second case), the node-specific main menu, a
    # bold node name on the first line, and menu items on the input lines that re-run list-inputs
    # recursively with control inputs.
    """List inputs non-recursively, without any control inputs."""

    # Do not include node op types.
    node_name = "control_deps/z"
    out = self._registry.dispatch_command("list_inputs", [node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 1):" % node_name,
        "|- (1) control_deps/x/read", "|  |- ...",
        "|- (1) control_deps/ctrl_dep_y", "   |- ...", "", "Legend:",
        "  (d): recursion depth = d."
    ], out.lines)

    # Include node op types.
    out = self._registry.dispatch_command("li", ["-t", node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 1):" % node_name,
        "|- (1) [Identity] control_deps/x/read", "|  |- ...",
        "|- (1) [Identity] control_deps/ctrl_dep_y", "   |- ...", "", "Legend:",
        "  (d): recursion depth = d.", "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

    # Verify that the node name has bold attribute.
    self.assertEqual([(16, 16 + len(node_name), "bold")], out.font_attr_segs[0])

    # Verify the menu items (command shortcuts) in the output.
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/x/read"),
                    len(out.lines[1]), "li -c -r control_deps/x/read")
    check_menu_item(self, out, 3,
                    len(out.lines[3]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[3]), "li -c -r control_deps/ctrl_dep_y")

  def testListInputsNonRecursiveNoControlUsingTensorName(self):
    # [FUNCTION DOC]
    # Scenario: list inputs of the multiply node giving its output tensor name instead of the
    # node name. Expects the same output as with the node name (depth-one tree, legend), the
    # node-specific main menu, and the same menu items on the input lines.
    """List inputs using the name of an output tensor of the node."""

    # Do not include node op types.
    node_name = "control_deps/z"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command("list_inputs", [tensor_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 1):" % node_name,
        "|- (1) control_deps/x/read", "|  |- ...",
        "|- (1) control_deps/ctrl_dep_y", "   |- ...", "", "Legend:",
        "  (d): recursion depth = d."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/x/read"),
                    len(out.lines[1]), "li -c -r control_deps/x/read")
    check_menu_item(self, out, 3,
                    len(out.lines[3]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[3]), "li -c -r control_deps/ctrl_dep_y")

  def testListInputsNonRecursiveWithControls(self):
    # [FUNCTION DOC]
    # Scenario: list inputs of the final control-dependency identity node with op types and
    # control inputs included, non-recursively. Expects regular and control inputs marked, the
    # extended legend, the node-specific main menu, and menu items on the three input lines.
    """List inputs non-recursively, with control inputs."""
    node_name = "control_deps/ctrl_dep_z"
    out = self._registry.dispatch_command("li", ["-t", node_name, "-c"])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 1, " % node_name +
        "control inputs included):", "|- (1) [Mul] control_deps/z", "|  |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y", "|  |- ...",
        "|- (1) (Ctrl) [VariableV2] control_deps/x", "", "Legend:",
        "  (d): recursion depth = d.", "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/z"),
                    len(out.lines[1]), "li -c -r control_deps/z")
    check_menu_item(self, out, 3,
                    len(out.lines[3]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[3]), "li -c -r control_deps/ctrl_dep_y")
    check_menu_item(self, out, 5,
                    len(out.lines[5]) - len("control_deps/x"),
                    len(out.lines[5]), "li -c -r control_deps/x")

  def testListInputsRecursiveWithControls(self):
    # [FUNCTION DOC]
    # Scenario: list inputs of the final control-dependency identity node recursively with
    # control inputs and op types, using the default depth limit of twenty. Expects the full
    # indented dependency tree with depth numbers, control markers and legend, the node-specific
    # main menu, and menu items on selected lines.
    """List inputs recursively, with control inputs."""
    node_name = "control_deps/ctrl_dep_z"
    out = self._registry.dispatch_command("li", ["-c", "-r", "-t", node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 20, " % node_name +
        "control inputs included):", "|- (1) [Mul] control_deps/z",
        "|  |- (2) [Identity] control_deps/x/read",
        "|  |  |- (3) [VariableV2] control_deps/x",
        "|  |- (2) [Identity] control_deps/ctrl_dep_y",
        "|     |- (3) [AddV2] control_deps/y",
        "|     |  |- (4) [Identity] control_deps/x/read",
        "|     |  |  |- (5) [VariableV2] control_deps/x",
        "|     |  |- (4) [Identity] control_deps/x/read",
        "|     |     |- (5) [VariableV2] control_deps/x",
        "|     |- (3) (Ctrl) [VariableV2] control_deps/x",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- (2) [AddV2] control_deps/y",
        "|  |  |- (3) [Identity] control_deps/x/read",
        "|  |  |  |- (4) [VariableV2] control_deps/x",
        "|  |  |- (3) [Identity] control_deps/x/read",
        "|  |     |- (4) [VariableV2] control_deps/x",
        "|  |- (2) (Ctrl) [VariableV2] control_deps/x",
        "|- (1) (Ctrl) [VariableV2] control_deps/x", "", "Legend:",
        "  (d): recursion depth = d.", "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/z"),
                    len(out.lines[1]), "li -c -r control_deps/z")
    check_menu_item(self, out, 11,
                    len(out.lines[11]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[11]), "li -c -r control_deps/ctrl_dep_y")
    check_menu_item(self, out, 18,
                    len(out.lines[18]) - len("control_deps/x"),
                    len(out.lines[18]), "li -c -r control_deps/x")

  def testListInputsRecursiveWithControlsWithDepthLimit(self):
    # [FUNCTION DOC]
    # Scenario: same as the recursive control-input listing but with a depth limit of two.
    # Expects a tree cut at depth two with continuation markers, the header stating the limit,
    # the node-specific main menu, and menu items on selected lines.
    """List inputs recursively, with control inputs and a depth limit."""
    node_name = "control_deps/ctrl_dep_z"
    out = self._registry.dispatch_command(
        "li", ["-c", "-r", "-t", "-d", "2", node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 2, " % node_name +
        "control inputs included):", "|- (1) [Mul] control_deps/z",
        "|  |- (2) [Identity] control_deps/x/read", "|  |  |- ...",
        "|  |- (2) [Identity] control_deps/ctrl_dep_y", "|     |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- (2) [AddV2] control_deps/y", "|  |  |- ...",
        "|  |- (2) (Ctrl) [VariableV2] control_deps/x",
        "|- (1) (Ctrl) [VariableV2] control_deps/x", "", "Legend:",
        "  (d): recursion depth = d.", "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/z"),
                    len(out.lines[1]), "li -c -r control_deps/z")
    check_menu_item(self, out, 10,
                    len(out.lines[10]) - len("control_deps/x"),
                    len(out.lines[10]), "li -c -r control_deps/x")

  def testListInputsNodeWithoutInputs(self):
    # [FUNCTION DOC]
    # Scenario: list inputs recursively (with control inputs and op types) for the variable node,
    # which has no inputs. Expects a header, a none marker line, a blank line, the legend, and the
    # node-specific main menu.
    """List the inputs to a node without any input."""
    node_name = "control_deps/x"
    out = self._registry.dispatch_command("li", ["-c", "-r", "-t", node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 20, control " % node_name +
        "inputs included):", "  [None]", "", "Legend:",
        "  (d): recursion depth = d.", "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testListInputsNonexistentNode(self):
    # [FUNCTION DOC]
    # Scenario: list inputs of a node name that does not exist. Expects a single error line
    # saying there is no such node in the partition graphs.
    out = self._registry.dispatch_command(
        "list_inputs", ["control_deps/z/foo"])

    self.assertEqual([
        "ERROR: There is no node named \"control_deps/z/foo\" in the "
        "partition graphs"], out.lines)

  def testListRecipientsRecursiveWithControlsWithDepthLimit(self):
    # [FUNCTION DOC]
    # Scenario: list outputs (recipients) of the variable node recursively with control
    # recipients, op types and a depth limit of one. Expects one regular recipient and two
    # control recipients with continuation markers, a legend, menu items on the three recipient
    # lines that re-run list-outputs, and a bold node name in the header.
    """List recipients recursively, with control inputs and a depth limit."""

    out = self._registry.dispatch_command(
        "lo", ["-c", "-r", "-t", "-d", "1", "control_deps/x"])

    self.assertEqual([
        "Recipients of node \"control_deps/x\" (Depth limit = 1, control "
        "recipients included):",
        "|- (1) [Identity] control_deps/x/read",
        "|  |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_z",
        "", "Legend:", "  (d): recursion depth = d.",
        "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."], out.lines)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/x/read"),
                    len(out.lines[1]), "lo -c -r control_deps/x/read")
    check_menu_item(self, out, 3,
                    len(out.lines[3]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[3]), "lo -c -r control_deps/ctrl_dep_y")
    check_menu_item(self, out, 5,
                    len(out.lines[5]) - len("control_deps/ctrl_dep_z"),
                    len(out.lines[5]), "lo -c -r control_deps/ctrl_dep_z")

    # Verify the bold attribute of the node name.
    self.assertEqual([(20, 20 + len("control_deps/x"), "bold")],
                     out.font_attr_segs[0])


@test_util.run_v1_only("b/120545219")
class AnalyzerCLIWhileLoopTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests of printing a tensor that is dumped many times (once per iteration of a while loop).
  # Fixtures: a temporary dump directory, the debug dump, the analyzer and the command registry,
  # built once for the class. Runs only under session (version 1) mode.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture. Creates a temporary dump directory and a graph with a while loop
    # counting from zero to ten with one parallel iteration; watches the loop's identity node
    # with the debug-identity op, runs the loop, and builds the debug dump, analyzer and command
    # registry. Returns nothing.
    cls._dump_root = tempfile.mkdtemp()

    with session.Session(config=no_rewrite_session_config()) as sess:
      loop_var = constant_op.constant(0, name="while_loop_test/loop_var")
      cond = lambda loop_var: math_ops.less(loop_var, 10)
      body = lambda loop_var: math_ops.add(loop_var, 1)
      while_loop = while_loop_tf.while_loop(
          cond, body, [loop_var], parallel_iterations=1)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_url = "file://%s" % cls._dump_root

      watch_opts = run_options.debug_options.debug_tensor_watch_opts

      # Add debug tensor watch for "while/Identity".
      watch = watch_opts.add()
      watch.node_name = "while/Identity"
      watch.output_slot = 0
      watch.debug_ops.append("DebugIdentity")
      watch.debug_urls.append(debug_url)

      # Invoke Session.run().
      run_metadata = config_pb2.RunMetadata()
      sess.run(while_loop, options=run_options, run_metadata=run_metadata)

    cls._debug_dump = debug_data.DebugDumpDir(
        cls._dump_root, partition_graphs=run_metadata.partition_graphs)

    cls._analyzer, cls._registry = create_analyzer_cli(cls._debug_dump)

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Class-level cleanup. Recursively deletes the temporary dump directory. Returns nothing.
    # Tear down temporary dump directory.
    file_io.delete_recursively(cls._dump_root)

  def testMultipleDumpsPrintTensorNoNumber(self):
    # [FUNCTION DOC]
    # Scenario: print the loop identity tensor without a dump number. Expects a header saying it
    # generated ten dumps, ten numbered lines ending with the dump's tensor name, and a final
    # hint explaining how to choose a dump with the number option and an example command.
    output = self._registry.dispatch_command("pt", ["while/Identity:0"])

    self.assertEqual("Tensor \"while/Identity:0\" generated 10 dumps:",
                     output.lines[0])

    for i in range(10):
      self.assertTrue(output.lines[i + 1].startswith("#%d" % i))
      self.assertTrue(output.lines[i + 1].endswith(
          " ms] while/Identity:0:DebugIdentity"))

    self.assertEqual(
        "You can use the -n (--number) flag to specify which dump to print.",
        output.lines[-3])
    self.assertEqual("For example:", output.lines[-2])
    self.assertEqual("  print_tensor while/Identity:0 -n 0", output.lines[-1])

  def testMultipleDumpsPrintTensorWithNumber(self):
    # [FUNCTION DOC]
    # Scenario: print the loop identity tensor for each dump number from zero to four. Expects
    # the header to name the dump number, integer type, scalar shape, a blank line and a value
    # line starting with the array of that iteration's value.
    for i in range(5):
      output = self._registry.dispatch_command(
          "pt", ["while/Identity:0", "-n", "%d" % i])

      self.assertEqual("Tensor \"while/Identity:0:DebugIdentity (dump #%d)\":" %
                       i, output.lines[0])
      self.assertEqual("  dtype: int32", output.lines[1])
      self.assertEqual("  shape: ()", output.lines[2])
      self.assertEqual("", output.lines[3])
      self.assertTrue(output.lines[4].startswith("array(%d" % i))
      self.assertTrue(output.lines[4].endswith(")"))

  def testMultipleDumpsPrintTensorInvalidNumber(self):
    # [FUNCTION DOC]
    # Scenario: print the loop identity tensor with dump number ten when only ten dumps (zero to
    # nine) exist. Expects a single error line saying the number exceeds the available dumps.
    output = self._registry.dispatch_command("pt",
                                             ["while/Identity:0", "-n", "10"])

    self.assertEqual([
        "ERROR: Specified number (10) exceeds the number of available dumps "
        "(10) for tensor while/Identity:0"
    ], output.lines)


if __name__ == "__main__":
  googletest.main()
