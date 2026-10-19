# [MODULE DOC]
# Profiling display for the tfdbg command-line interface. Turns a graph plus run metadata (step
# statistics per device) into text tables shown by two commands: a per-op profile listing with
# sorting and filtering by device, node name, op type, file, line range and time intervals, and an
# annotated source-file view with per-line cost bars. Exposes a table-view class, an analyzer class
# that parses command arguments and builds rich text output, filter and sort-key helpers, the
# sort-mode constants, and a factory that creates a UI with both commands registered (with short
# aliases). Import-time side effects: defines constants only; registration happens in the factory.
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
"""Formats and displays profiling information."""

import argparse
import os
import re

import numpy as np

from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import profiling
from tensorflow.python.debug.lib import source_utils

RL = debugger_cli_common.RichLine

SORT_OPS_BY_OP_NAME = "node"
SORT_OPS_BY_OP_TYPE = "op_type"
SORT_OPS_BY_OP_TIME = "op_time"
SORT_OPS_BY_EXEC_TIME = "exec_time"
SORT_OPS_BY_START_TIME = "start_time"
SORT_OPS_BY_LINE = "line"

_DEVICE_NAME_FILTER_FLAG = "device_name_filter"
_NODE_NAME_FILTER_FLAG = "node_name_filter"
_OP_TYPE_FILTER_FLAG = "op_type_filter"


class ProfileDataTableView(object):
  # [CLASS DOC]
  # Table view over a list of profile records for one device. Holds the record list, per-row
  # formatted start time, op time and exec time (in a chosen time unit), the six column headers
  # and the sort identifier associated with each column. Provides cell access (with a clickable
  # menu item on the last column), row and column counts, names and sort ids.
  """Table View of profiling data."""

  def __init__(self, profile_datum_list, time_unit=cli_shared.TIME_UNIT_US):
    # [FUNCTION DOC]
    # Constructor. profile_datum_list is a list of profile records; time_unit is a unit name that
    # must be one of the supported time units (default microseconds). Precomputes formatted
    # start, op and exec time columns, the column header names (unit shown in the time headers)
    # and the column sort ids. Returns nothing; no validation of time_unit is done here.
    """Constructor.

    Args:
      profile_datum_list: List of `ProfileDatum` objects.
      time_unit: must be in cli_shared.TIME_UNITS.
    """
    self._profile_datum_list = profile_datum_list
    self.formatted_start_time = [
        datum.start_time for datum in profile_datum_list]
    self.formatted_op_time = [
        cli_shared.time_to_readable_str(datum.op_time,
                                        force_time_unit=time_unit)
        for datum in profile_datum_list]
    self.formatted_exec_time = [
        cli_shared.time_to_readable_str(
            datum.node_exec_stats.all_end_rel_micros,
            force_time_unit=time_unit)
        for datum in profile_datum_list]

    self._column_names = ["Node",
                          "Op Type",
                          "Start Time (us)",
                          "Op Time (%s)" % time_unit,
                          "Exec Time (%s)" % time_unit,
                          "Filename:Lineno(function)"]
    self._column_sort_ids = [SORT_OPS_BY_OP_NAME, SORT_OPS_BY_OP_TYPE,
                             SORT_OPS_BY_START_TIME, SORT_OPS_BY_OP_TIME,
                             SORT_OPS_BY_EXEC_TIME, SORT_OPS_BY_LINE]

  def value(self,
            row,
            col,
            device_name_filter=None,
            node_name_filter=None,
            op_type_filter=None):
    # [FUNCTION DOC]
    # Returns the content of one table cell as a rich text line. row and col are integer indexes.
    # Columns 0 to 4 give node name, op type, start time, op time and exec time as text. Column 5
    # gives the file, line and function text, wrapped in a clickable menu item whose command is the
    # print-source command for that file with an initial line, carrying over any non-empty device,
    # node name and op type filters (regex strings, optional). Raises an index error for a column
    # other than 0 to 5 (and a plain index error from the list for a bad row). No side effects.
    """Get the content of a cell of the table.

    Args:
      row: (int) row index.
      col: (int) column index.
      device_name_filter: Regular expression to filter by device name.
      node_name_filter: Regular expression to filter by node name.
      op_type_filter: Regular expression to filter by op type.

    Returns:
      A debuggre_cli_common.RichLine object representing the content of the
      cell, potentially with a clickable MenuItem.

    Raises:
      IndexError: if row index is out of range.
    """
    menu_item = None
    if col == 0:
      text = self._profile_datum_list[row].node_exec_stats.node_name
    elif col == 1:
      text = self._profile_datum_list[row].op_type
    elif col == 2:
      text = str(self.formatted_start_time[row])
    elif col == 3:
      text = str(self.formatted_op_time[row])
    elif col == 4:
      text = str(self.formatted_exec_time[row])
    elif col == 5:
      command = "ps"
      if device_name_filter:
        command += " --%s %s" % (_DEVICE_NAME_FILTER_FLAG,
                                 device_name_filter)
      if node_name_filter:
        command += " --%s %s" % (_NODE_NAME_FILTER_FLAG, node_name_filter)
      if op_type_filter:
        command += " --%s %s" % (_OP_TYPE_FILTER_FLAG, op_type_filter)
      command += " %s --init_line %d" % (
          self._profile_datum_list[row].file_path,
          self._profile_datum_list[row].line_number)
      menu_item = debugger_cli_common.MenuItem(None, command)
      text = self._profile_datum_list[row].file_line_func
    else:
      raise IndexError("Invalid column index %d." % col)

    return RL(text, font_attr=menu_item)

  def row_count(self):
    # [FUNCTION DOC]
    # Returns the number of rows, i.e. the number of profile records in the table.
    return len(self._profile_datum_list)

  def column_count(self):
    # [FUNCTION DOC]
    # Returns the number of columns (six).
    return len(self._column_names)

  def column_names(self):
    # [FUNCTION DOC]
    # Returns the list of column header strings (the internal list, not a copy).
    return self._column_names

  def column_sort_id(self, col):
    # [FUNCTION DOC]
    # Returns the sort identifier string for the column at integer index col. Raises an index
    # error if col is out of range.
    return self._column_sort_ids[col]


def _list_profile_filter(
    profile_datum,
    node_name_regex,
    file_path_regex,
    op_type_regex,
    op_time_interval,
    exec_time_interval,
    min_lineno=-1,
    max_lineno=-1):
  # [FUNCTION DOC]
  # HELPER: row filter for the list-profile command of ProfileAnalyzer.
  # Decides whether a profile record is kept. Parameters: the record; compiled patterns for node
  # name, file path and op type (each may be nothing, meaning no filtering; matching is anchored at
  # the start of the text); op time and exec time intervals (nothing means no filtering); optional
  # inclusive lower and exclusive upper line number bounds, ignored when not positive. Returns
  # false if any active condition fails (a record with no file path fails an active file filter;
  # records with no line number or no op type skip the line and op type checks), true otherwise.
  # No side effects.
  """Filter function for list_profile command.

  Args:
    profile_datum: A `ProfileDatum` object.
    node_name_regex: Regular expression pattern object to filter by name.
    file_path_regex: Regular expression pattern object to filter by file path.
    op_type_regex: Regular expression pattern object to filter by op type.
    op_time_interval: `Interval` for filtering op time.
    exec_time_interval: `Interval` for filtering exec time.
    min_lineno: Lower bound for 1-based line number, inclusive.
      If <= 0, has no effect.
    max_lineno: Upper bound for 1-based line number, exclusive.
      If <= 0, has no effect.
    # TODO(cais): Maybe filter by function name.

  Returns:
    True iff profile_datum should be included.
  """
  if node_name_regex and not node_name_regex.match(
      profile_datum.node_exec_stats.node_name):
    return False
  if file_path_regex:
    if (not profile_datum.file_path or
        not file_path_regex.match(profile_datum.file_path)):
      return False
  if (min_lineno > 0 and profile_datum.line_number and
      profile_datum.line_number < min_lineno):
    return False
  if (max_lineno > 0 and profile_datum.line_number and
      profile_datum.line_number >= max_lineno):
    return False
  if (profile_datum.op_type is not None and op_type_regex and
      not op_type_regex.match(profile_datum.op_type)):
    return False
  if op_time_interval is not None and not op_time_interval.contains(
      profile_datum.op_time):
    return False
  if exec_time_interval and not exec_time_interval.contains(
      profile_datum.node_exec_stats.all_end_rel_micros):
    return False
  return True


def _list_profile_sort_key(profile_datum, sort_by):
  # [FUNCTION DOC]
  # HELPER: provides the sort key for the list-profile command of ProfileAnalyzer.
  # Given a profile record and a sort mode string, returns the matching field: node name, op type,
  # file-line-function text, op time, or exec time (end relative micros). Any other value,
  # including the start-time mode, falls through to the start time of the node (no error is raised
  # for unknown modes). No side effects.
  """Get a profile_datum property to sort by in list_profile command.

  Args:
    profile_datum: A `ProfileDatum` object.
    sort_by: (string) indicates a value to sort by.
      Must be one of SORT_BY* constants.

  Returns:
    profile_datum property to sort by.
  """
  if sort_by == SORT_OPS_BY_OP_NAME:
    return profile_datum.node_exec_stats.node_name
  elif sort_by == SORT_OPS_BY_OP_TYPE:
    return profile_datum.op_type
  elif sort_by == SORT_OPS_BY_LINE:
    return profile_datum.file_line_func
  elif sort_by == SORT_OPS_BY_OP_TIME:
    return profile_datum.op_time
  elif sort_by == SORT_OPS_BY_EXEC_TIME:
    return profile_datum.node_exec_stats.all_end_rel_micros
  else:  # sort by start time
    return profile_datum.node_exec_stats.all_start_micros


class ProfileAnalyzer(object):
  # [CLASS DOC]
  # Analyzer that implements the profiling commands of the debugger UI. Holds the graph, the run
  # metadata with per-device step stats, and a mapping from command name to its argument parser
  # (list-profile and print-source). Command handlers take argument lists and return rich text
  # lines. Created by create_profiler_ui, which registers the handlers.
  """Analyzer for profiling data."""

  def __init__(self, graph, run_metadata):
    # [FUNCTION DOC]
    # Constructor. graph is the Python graph whose operations carry creation tracebacks;
    # run_metadata is the run metadata message (must be truthy). Raises a value error if
    # run_metadata is empty or nothing. Builds and stores two argument parsers: one for
    # list-profile (device, node name, op type, file filters, line bounds, exec/op time intervals,
    # sort field, reverse flag, time unit) and one for print-source (source path, cost type,
    # time unit, device/node/op filters, initial line). Usage text is suppressed.
    """ProfileAnalyzer constructor.

    Args:
      graph: (tf.Graph) Python graph object.
      run_metadata: A `RunMetadata` protobuf object.

    Raises:
      ValueError: If run_metadata is None.
    """
    self._graph = graph
    if not run_metadata:
      raise ValueError("No RunMetadata passed for profile analysis.")
    self._run_metadata = run_metadata
    self._arg_parsers = {}
    ap = argparse.ArgumentParser(
        description="List nodes profile information.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-d",
        "--%s" % _DEVICE_NAME_FILTER_FLAG,
        dest=_DEVICE_NAME_FILTER_FLAG,
        type=str,
        default="",
        help="filter device name by regex.")
    ap.add_argument(
        "-n",
        "--%s" % _NODE_NAME_FILTER_FLAG,
        dest=_NODE_NAME_FILTER_FLAG,
        type=str,
        default="",
        help="filter node name by regex.")
    ap.add_argument(
        "-t",
        "--%s" % _OP_TYPE_FILTER_FLAG,
        dest=_OP_TYPE_FILTER_FLAG,
        type=str,
        default="",
        help="filter op type by regex.")
    # TODO(annarev): allow file filtering at non-stack top position.
    ap.add_argument(
        "-f",
        "--file_path_filter",
        dest="file_path_filter",
        type=str,
        default="",
        help="filter by file name at the top position of node's creation "
             "stack that does not belong to TensorFlow library.")
    ap.add_argument(
        "--min_lineno",
        dest="min_lineno",
        type=int,
        default=-1,
        help="(Inclusive) lower bound for 1-based line number in source file. "
             "If <= 0, has no effect.")
    ap.add_argument(
        "--max_lineno",
        dest="max_lineno",
        type=int,
        default=-1,
        help="(Exclusive) upper bound for 1-based line number in source file. "
             "If <= 0, has no effect.")
    ap.add_argument(
        "-e",
        "--execution_time",
        dest="execution_time",
        type=str,
        default="",
        help="Filter by execution time interval "
             "(includes compute plus pre- and post -processing time). "
             "Supported units are s, ms and us (default). "
             "E.g. -e >100s, -e <100, -e [100us,1000ms]")
    ap.add_argument(
        "-o",
        "--op_time",
        dest="op_time",
        type=str,
        default="",
        help="Filter by op time interval (only includes compute time). "
             "Supported units are s, ms and us (default). "
             "E.g. -e >100s, -e <100, -e [100us,1000ms]")
    ap.add_argument(
        "-s",
        "--sort_by",
        dest="sort_by",
        type=str,
        default=SORT_OPS_BY_START_TIME,
        help=("the field to sort the data by: (%s)" %
              " | ".join([SORT_OPS_BY_OP_NAME, SORT_OPS_BY_OP_TYPE,
                          SORT_OPS_BY_START_TIME, SORT_OPS_BY_OP_TIME,
                          SORT_OPS_BY_EXEC_TIME, SORT_OPS_BY_LINE])))
    ap.add_argument(
        "-r",
        "--reverse",
        dest="reverse",
        action="store_true",
        help="sort the data in reverse (descending) order")
    ap.add_argument(
        "--time_unit",
        dest="time_unit",
        type=str,
        default=cli_shared.TIME_UNIT_US,
        help="Time unit (" + " | ".join(cli_shared.TIME_UNITS) + ")")

    self._arg_parsers["list_profile"] = ap

    ap = argparse.ArgumentParser(
        description="Print a Python source file with line-level profile "
                    "information",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "source_file_path",
        type=str,
        help="Path to the source_file_path")
    ap.add_argument(
        "--cost_type",
        type=str,
        choices=["exec_time", "op_time"],
        default="exec_time",
        help="Type of cost to display")
    ap.add_argument(
        "--time_unit",
        dest="time_unit",
        type=str,
        default=cli_shared.TIME_UNIT_US,
        help="Time unit (" + " | ".join(cli_shared.TIME_UNITS) + ")")
    ap.add_argument(
        "-d",
        "--%s" % _DEVICE_NAME_FILTER_FLAG,
        dest=_DEVICE_NAME_FILTER_FLAG,
        type=str,
        default="",
        help="Filter device name by regex.")
    ap.add_argument(
        "-n",
        "--%s" % _NODE_NAME_FILTER_FLAG,
        dest=_NODE_NAME_FILTER_FLAG,
        type=str,
        default="",
        help="Filter node name by regex.")
    ap.add_argument(
        "-t",
        "--%s" % _OP_TYPE_FILTER_FLAG,
        dest=_OP_TYPE_FILTER_FLAG,
        type=str,
        default="",
        help="Filter op type by regex.")
    ap.add_argument(
        "--init_line",
        dest="init_line",
        type=int,
        default=0,
        help="The 1-based line number to scroll to initially.")

    self._arg_parsers["print_source"] = ap

  def list_profile(self, args, screen_info=None):
    # [FUNCTION DOC]
    # Command handler that lists per-op profile information. args is the list of argument strings
    # (without the command name); screen_info is an optional mapping that may hold the screen
    # width under the columns key (default 80). Parses the arguments (the parser may exit or raise
    # on bad flags), converts the time interval strings and filter strings into intervals and
    # compiled patterns, then for each device whose name matches the device filter: collects its
    # profile records, filters them, sorts them by the chosen field (optionally reversed) and
    # appends a formatted table for that device. Returns a rich text lines object. No other side
    # effects.
    """Command handler for list_profile.

    List per-operation profile information.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """
    screen_cols = 80
    if screen_info and "cols" in screen_info:
      screen_cols = screen_info["cols"]

    parsed = self._arg_parsers["list_profile"].parse_args(args)
    op_time_interval = (command_parser.parse_time_interval(parsed.op_time)
                        if parsed.op_time else None)
    exec_time_interval = (
        command_parser.parse_time_interval(parsed.execution_time)
        if parsed.execution_time else None)
    node_name_regex = (re.compile(parsed.node_name_filter)
                       if parsed.node_name_filter else None)
    file_path_regex = (re.compile(parsed.file_path_filter)
                       if parsed.file_path_filter else None)
    op_type_regex = (re.compile(parsed.op_type_filter)
                     if parsed.op_type_filter else None)

    output = debugger_cli_common.RichTextLines([""])
    device_name_regex = (re.compile(parsed.device_name_filter)
                         if parsed.device_name_filter else None)
    data_generator = self._get_profile_data_generator()
    device_count = len(self._run_metadata.step_stats.dev_stats)
    for index in range(device_count):
      device_stats = self._run_metadata.step_stats.dev_stats[index]
      if not device_name_regex or device_name_regex.match(device_stats.device):
        profile_data = [
            datum for datum in data_generator(device_stats)
            if _list_profile_filter(
                datum, node_name_regex, file_path_regex, op_type_regex,
                op_time_interval, exec_time_interval,
                min_lineno=parsed.min_lineno, max_lineno=parsed.max_lineno)]
        profile_data = sorted(
            profile_data,
            key=lambda datum: _list_profile_sort_key(datum, parsed.sort_by),
            reverse=parsed.reverse)
        output.extend(
            self._get_list_profile_lines(
                device_stats.device, index, device_count,
                profile_data, parsed.sort_by, parsed.reverse, parsed.time_unit,
                device_name_filter=parsed.device_name_filter,
                node_name_filter=parsed.node_name_filter,
                op_type_filter=parsed.op_type_filter,
                screen_cols=screen_cols))
    return output

  def _get_profile_data_generator(self):
    # [FUNCTION DOC]
    # HELPER: builds the record generator used by list_profile and print_source.
    # Walks all graph operations; for each, scans its traceback from the innermost-last entry
    # backwards until it finds a frame outside the TensorFlow library (if every frame is inside the
    # library the last examined frame is used), and records file path, line number, function name
    # and op type by op name. Returns a function that takes one device's step stats and yields a
    # profile record for each node (skipping the source and sink pseudo nodes), using empty or zero
    # defaults for nodes not in the graph. Fails if an operation has an empty traceback.
    """Get function that generates `ProfileDatum` objects.

    Returns:
      A function that generates `ProfileDatum` objects.
    """
    node_to_file_path = {}
    node_to_line_number = {}
    node_to_func_name = {}
    node_to_op_type = {}
    for op in self._graph.get_operations():
      for trace_entry in reversed(op.traceback):
        file_path = trace_entry[0]
        line_num = trace_entry[1]
        func_name = trace_entry[2]
        if not source_utils.guess_is_tensorflow_py_library(file_path):
          break
      node_to_file_path[op.name] = file_path
      node_to_line_number[op.name] = line_num
      node_to_func_name[op.name] = func_name
      node_to_op_type[op.name] = op.type

    def profile_data_generator(device_step_stats):
      # [FUNCTION DOC]
      # HELPER: generator returned by the enclosing helper. Takes one device's step stats and
      # yields a profile record per node, skipping the source and sink pseudo nodes, filling file
      # path, line number, function name and op type from the precomputed per-node maps.
      for node_stats in device_step_stats.node_stats:
        if node_stats.node_name == "_SOURCE" or node_stats.node_name == "_SINK":
          continue
        yield profiling.ProfileDatum(
            device_step_stats.device,
            node_stats,
            node_to_file_path.get(node_stats.node_name, ""),
            node_to_line_number.get(node_stats.node_name, 0),
            node_to_func_name.get(node_stats.node_name, ""),
            node_to_op_type.get(node_stats.node_name, ""))
    return profile_data_generator

  def _get_list_profile_lines(
      self, device_name, device_index, device_count,
      profile_datum_list, sort_by, sort_reverse, time_unit,
      device_name_filter=None, node_name_filter=None, op_type_filter=None,
      screen_cols=80):
    # [FUNCTION DOC]
    # HELPER: formats the table for one device on behalf of list_profile.
    # KNOWN ISSUE: the docstring mentions a memory sort constant that does not exist in this
    # module. Also the device total row has only four cells (label, empty, op time total, exec time
    # total) so the totals appear under the start-time and op-time columns, not under the op-time
    # and exec-time columns.
    # Inputs: device name, zero-based device index, device count, list of profile records, sort
    # field name, reverse flag, time unit, optional device/node/op type filter strings (passed to
    # cell menu commands) and screen width (default 80). Computes totals, column widths (with a
    # two-character margin), then builds lines: a separator, a device heading, a blank line, a
    # clickable header row (clicking sorts, toggling reverse when already sorted ascending by that
    # column), one row per record, a blank line and the totals row. Returns a rich text lines
    # object. No side effects.
    """Get `RichTextLines` object for list_profile command for a given device.

    Args:
      device_name: (string) Device name.
      device_index: (int) Device index.
      device_count: (int) Number of devices.
      profile_datum_list: List of `ProfileDatum` objects.
      sort_by: (string) Identifier of column to sort. Sort identifier
          must match value of SORT_OPS_BY_OP_NAME, SORT_OPS_BY_OP_TYPE,
          SORT_OPS_BY_EXEC_TIME, SORT_OPS_BY_MEMORY or SORT_OPS_BY_LINE.
      sort_reverse: (bool) Whether to sort in descending instead of default
          (ascending) order.
      time_unit: time unit, must be in cli_shared.TIME_UNITS.
      device_name_filter: Regular expression to filter by device name.
      node_name_filter: Regular expression to filter by node name.
      op_type_filter: Regular expression to filter by op type.
      screen_cols: (int) Number of columns available on the screen (i.e.,
        available screen width).

    Returns:
      `RichTextLines` object containing a table that displays profiling
      information for each op.
    """
    profile_data = ProfileDataTableView(profile_datum_list, time_unit=time_unit)

    # Calculate total time early to calculate column widths.
    total_op_time = sum(datum.op_time for datum in profile_datum_list)
    total_exec_time = sum(datum.node_exec_stats.all_end_rel_micros
                          for datum in profile_datum_list)
    device_total_row = [
        "Device Total", "",
        cli_shared.time_to_readable_str(total_op_time,
                                        force_time_unit=time_unit),
        cli_shared.time_to_readable_str(total_exec_time,
                                        force_time_unit=time_unit)]

    # Calculate column widths.
    column_widths = [
        len(column_name) for column_name in profile_data.column_names()]
    for col in range(len(device_total_row)):
      column_widths[col] = max(column_widths[col], len(device_total_row[col]))
    for col in range(len(column_widths)):
      for row in range(profile_data.row_count()):
        column_widths[col] = max(
            column_widths[col], len(profile_data.value(
                row,
                col,
                device_name_filter=device_name_filter,
                node_name_filter=node_name_filter,
                op_type_filter=op_type_filter)))
      column_widths[col] += 2  # add margin between columns

    # Add device name.
    output = [RL("-" * screen_cols)]
    device_row = "Device %d of %d: %s" % (
        device_index + 1, device_count, device_name)
    output.append(RL(device_row))
    output.append(RL())

    # Add headers.
    base_command = "list_profile"
    row = RL()
    for col in range(profile_data.column_count()):
      column_name = profile_data.column_names()[col]
      sort_id = profile_data.column_sort_id(col)
      command = "%s -s %s" % (base_command, sort_id)
      if sort_by == sort_id and not sort_reverse:
        command += " -r"
      head_menu_item = debugger_cli_common.MenuItem(None, command)
      row += RL(column_name, font_attr=[head_menu_item, "bold"])
      row += RL(" " * (column_widths[col] - len(column_name)))

    output.append(row)

    # Add data rows.
    for row in range(profile_data.row_count()):
      new_row = RL()
      for col in range(profile_data.column_count()):
        new_cell = profile_data.value(
            row,
            col,
            device_name_filter=device_name_filter,
            node_name_filter=node_name_filter,
            op_type_filter=op_type_filter)
        new_row += new_cell
        new_row += RL(" " * (column_widths[col] - len(new_cell)))
      output.append(new_row)

    # Add stat totals.
    row_str = ""
    for width, row in zip(column_widths, device_total_row):
      row_str += ("{:<%d}" % width).format(row)
    output.append(RL())
    output.append(RL(row_str))
    return debugger_cli_common.rich_text_lines_from_rich_line_list(output)

  def _measure_list_profile_column_widths(self, profile_data):
    # [FUNCTION DOC]
    # HELPER: intended column width calculator for list-profile tables.
    # KNOWN ISSUE: appears to be dead code. Nothing in this file calls it, and it calls a row
    # values method that the table view class does not define, so calling it would fail with an
    # attribute error.
    # Would return, per column, the maximum text length over header and rows plus two.
    """Determine the maximum column widths for each data list.

    Args:
      profile_data: list of ProfileDatum objects.

    Returns:
      List of column widths in the same order as columns in data.
    """
    num_columns = len(profile_data.column_names())
    widths = [len(column_name) for column_name in profile_data.column_names()]
    for row in range(profile_data.row_count()):
      for col in range(num_columns):
        widths[col] = max(
            widths[col], len(str(profile_data.row_values(row)[col])) + 2)
    return widths

  _LINE_COST_ATTR = cli_shared.COLOR_CYAN
  _LINE_NUM_ATTR = cli_shared.COLOR_YELLOW
  _NUM_NODES_HEAD = "#nodes"
  _NUM_EXECS_SUB_HEAD = "(#execs)"
  _LINENO_HEAD = "lineno"
  _SOURCE_HEAD = "source"

  def print_source(self, args, screen_info=None):
    # [FUNCTION DOC]
    # Command handler that prints a source file annotated with line-level profile costs. args is
    # the list of argument strings (without the command name); screen_info is accepted but ignored.
    # Parses the arguments, gathers profile records from devices matching the device filter, and
    # annotates the (home-expanded) source file against them using the node and op type filters.
    # If no line has profile information, returns a short message listing the filters. Otherwise
    # loads the source, finds the maximum line cost for the chosen cost type, and builds a header
    # and sub-header plus one line per source line: with a cost bar, formatted total cost and a
    # clickable node/exec count (which runs the list-profile command restricted to that line) when
    # the line is annotated, else blank padding; then the line number and source text. If the
    # initial line argument matches a line, records its position as the initial scroll position.
    # Returns a rich text lines object. Reads the source file; raises a value error indirectly
    # for an unsupported cost type.
    """Print a Python source file with line-level profile information.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """
    del screen_info

    parsed = self._arg_parsers["print_source"].parse_args(args)

    device_name_regex = (re.compile(parsed.device_name_filter)
                         if parsed.device_name_filter else None)

    profile_data = []
    data_generator = self._get_profile_data_generator()
    device_count = len(self._run_metadata.step_stats.dev_stats)
    for index in range(device_count):
      device_stats = self._run_metadata.step_stats.dev_stats[index]
      if device_name_regex and not device_name_regex.match(device_stats.device):
        continue
      profile_data.extend(data_generator(device_stats))

    source_annotation = source_utils.annotate_source_against_profile(
        profile_data,
        os.path.expanduser(parsed.source_file_path),
        node_name_filter=parsed.node_name_filter,
        op_type_filter=parsed.op_type_filter)
    if not source_annotation:
      return debugger_cli_common.RichTextLines(
          ["The source file %s does not contain any profile information for "
           "the previous Session run under the following "
           "filters:" % parsed.source_file_path,
           "  --%s: %s" % (_DEVICE_NAME_FILTER_FLAG, parsed.device_name_filter),
           "  --%s: %s" % (_NODE_NAME_FILTER_FLAG, parsed.node_name_filter),
           "  --%s: %s" % (_OP_TYPE_FILTER_FLAG, parsed.op_type_filter)])

    max_total_cost = 0
    for line_index in source_annotation:
      total_cost = self._get_total_cost(source_annotation[line_index],
                                        parsed.cost_type)
      max_total_cost = max(max_total_cost, total_cost)

    source_lines, line_num_width = source_utils.load_source(
        parsed.source_file_path)

    cost_bar_max_length = 10
    total_cost_head = parsed.cost_type
    column_widths = {
        "cost_bar": cost_bar_max_length + 3,
        "total_cost": len(total_cost_head) + 3,
        "num_nodes_execs": len(self._NUM_EXECS_SUB_HEAD) + 1,
        "line_number": line_num_width,
    }

    head = RL(
        " " * column_widths["cost_bar"] +
        total_cost_head +
        " " * (column_widths["total_cost"] - len(total_cost_head)) +
        self._NUM_NODES_HEAD +
        " " * (column_widths["num_nodes_execs"] - len(self._NUM_NODES_HEAD)),
        font_attr=self._LINE_COST_ATTR)
    head += RL(self._LINENO_HEAD, font_attr=self._LINE_NUM_ATTR)
    sub_head = RL(
        " " * (column_widths["cost_bar"] +
               column_widths["total_cost"]) +
        self._NUM_EXECS_SUB_HEAD +
        " " * (column_widths["num_nodes_execs"] -
               len(self._NUM_EXECS_SUB_HEAD)) +
        " " * column_widths["line_number"],
        font_attr=self._LINE_COST_ATTR)
    sub_head += RL(self._SOURCE_HEAD, font_attr="bold")
    lines = [head, sub_head]

    output_annotations = {}
    for i, line in enumerate(source_lines):
      lineno = i + 1
      if lineno in source_annotation:
        annotation = source_annotation[lineno]
        cost_bar = self._render_normalized_cost_bar(
            self._get_total_cost(annotation, parsed.cost_type), max_total_cost,
            cost_bar_max_length)
        annotated_line = cost_bar
        annotated_line += " " * (column_widths["cost_bar"] - len(cost_bar))

        total_cost = RL(cli_shared.time_to_readable_str(
            self._get_total_cost(annotation, parsed.cost_type),
            force_time_unit=parsed.time_unit),
                        font_attr=self._LINE_COST_ATTR)
        total_cost += " " * (column_widths["total_cost"] - len(total_cost))
        annotated_line += total_cost

        file_path_filter = re.escape(parsed.source_file_path) + "$"
        command = "lp --file_path_filter %s --min_lineno %d --max_lineno %d" % (
            file_path_filter, lineno, lineno + 1)
        if parsed.device_name_filter:
          command += " --%s %s" % (_DEVICE_NAME_FILTER_FLAG,
                                   parsed.device_name_filter)
        if parsed.node_name_filter:
          command += " --%s %s" % (_NODE_NAME_FILTER_FLAG,
                                   parsed.node_name_filter)
        if parsed.op_type_filter:
          command += " --%s %s" % (_OP_TYPE_FILTER_FLAG,
                                   parsed.op_type_filter)
        menu_item = debugger_cli_common.MenuItem(None, command)
        num_nodes_execs = RL("%d(%d)" % (annotation.node_count,
                                         annotation.node_exec_count),
                             font_attr=[self._LINE_COST_ATTR, menu_item])
        num_nodes_execs += " " * (
            column_widths["num_nodes_execs"] - len(num_nodes_execs))
        annotated_line += num_nodes_execs
      else:
        annotated_line = RL(
            " " * sum(column_widths[col_name] for col_name in column_widths
                      if col_name != "line_number"))

      line_num_column = RL(" L%d" % (lineno), self._LINE_NUM_ATTR)
      line_num_column += " " * (
          column_widths["line_number"] - len(line_num_column))
      annotated_line += line_num_column
      annotated_line += line
      lines.append(annotated_line)

      if parsed.init_line == lineno:
        output_annotations[
            debugger_cli_common.INIT_SCROLL_POS_KEY] = len(lines) - 1

    return debugger_cli_common.rich_text_lines_from_rich_line_list(
        lines, annotations=output_annotations)

  def _get_total_cost(self, aggregated_profile, cost_type):
    # [FUNCTION DOC]
    # HELPER: selects a cost for print_source. Given aggregated per-line profile data and a cost
    # type string, returns its total exec time for the exec-time type or total op time for the
    # op-time type. Raises a value error for any other cost type.
    if cost_type == "exec_time":
      return aggregated_profile.total_exec_time
    elif cost_type == "op_time":
      return aggregated_profile.total_op_time
    else:
      raise ValueError("Unsupported cost type: %s" % cost_type)

  def _render_normalized_cost_bar(self, cost, max_cost, length):
    # [FUNCTION DOC]
    # HELPER: draws the cost bar for print_source. cost is the absolute cost, max_cost the
    # normalizing maximum (must be non-zero), length the bar width in characters. The number of
    # filled ticks is the rounded-up proportion of the length, at least one. Returns a rich text
    # line: an opening bracket, bold tick marks padded with spaces, and a closing bracket.
    # KNOWN ISSUE: the docstring says the result is a rich text line type with a different name
    # than the one actually used; behavior otherwise matches.
    """Render a text bar representing a normalized cost.

    Args:
      cost: the absolute value of the cost.
      max_cost: the maximum cost value to normalize the absolute cost with.
      length: (int) length of the cost bar, in number of characters, excluding
        the brackets on the two ends.

    Returns:
      An instance of debugger_cli_common.RichTextLine.
    """
    num_ticks = int(np.ceil(float(cost) / max_cost * length))
    num_ticks = num_ticks or 1  # Minimum is 1 tick.
    output = RL("[", font_attr=self._LINE_COST_ATTR)
    output += RL("|" * num_ticks + " " * (length - num_ticks),
                 font_attr=["bold", self._LINE_COST_ATTR])
    output += RL("]", font_attr=self._LINE_COST_ATTR)
    return output

  def get_help(self, handler_name):
    # [FUNCTION DOC]
    # Returns the formatted help text of the argument parser for the named command handler
    # (list-profile or print-source). Raises a key error for an unknown handler name.
    return self._arg_parsers[handler_name].format_help()


def create_profiler_ui(graph,
                       run_metadata,
                       ui_type="readline",
                       on_ui_exit=None,
                       config=None):
  # [FUNCTION DOC]
  # Factory for the profiler command-line UI. graph is the Python graph, run_metadata the run
  # metadata message (must be non-empty), ui_type the UI kind name (default the line-based
  # reader UI), on_ui_exit an optional exit callback, config an ignored placeholder.
  # Builds a ProfileAnalyzer, asks the UI factory for a UI, registers the list-profile command
  # (alias lp) and print-source command (alias ps) with their help text, and returns the UI.
  # Raises a value error if run_metadata is empty. Side effect: registers command handlers.
  """Create an instance of ReadlineUI based on a `tf.Graph` and `RunMetadata`.

  Args:
    graph: Python `Graph` object.
    run_metadata: A `RunMetadata` protobuf object.
    ui_type: (str) requested UI type, e.g., "readline".
    on_ui_exit: (`Callable`) the callback to be called when the UI exits.
    config: An instance of `cli_config.CLIConfig`.

  Returns:
    (base_ui.BaseUI) A BaseUI subtype object with a set of standard analyzer
      commands and tab-completions registered.
  """
  del config  # Currently unused.

  analyzer = ProfileAnalyzer(graph, run_metadata)

  cli = ui_factory.get_ui(ui_type, on_ui_exit=on_ui_exit)
  cli.register_command_handler(
      "list_profile",
      analyzer.list_profile,
      analyzer.get_help("list_profile"),
      prefix_aliases=["lp"])
  cli.register_command_handler(
      "print_source",
      analyzer.print_source,
      analyzer.get_help("print_source"),
      prefix_aliases=["ps"])

  return cli
