# [MODULE DOC]
# Small data structures for debugger profiling: a record of one op execution (device, execution
# statistics, creating source location, op type, timing) and an aggregate that sums op and
# execution times and counts executions over many such records. Used by the debugger's profile
# analysis. No import-time side effects.
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
"""Data structures and algorithms for profiling information."""

import os


class ProfileDatum(object):
  # [CLASS DOC]
  # One profile data point for a single node execution. Attributes: device name, the node
  # execution statistics message, source file path, line number and function name of the op's
  # creation site, a combined file:line(function) string (empty without a file path), op type,
  # start time (microseconds) and op time (end minus start relative micros). Created by profile
  # analysis code; exec_time is exposed as a property.
  """Profile data point."""

  def __init__(self,
               device_name,
               node_exec_stats,
               file_path,
               line_number,
               func_name,
               op_type):
    # [FUNCTION DOC]
    # Stores device name (string), node execution statistics message, source file path, line
    # number (integer), function name and op type (string). Derives a display string made of the
    # file's base name, line number and function name (empty when the file path is empty or
    # nothing), the start time from the statistics, and the op time as the relative end minus
    # relative start micros.
    """Constructor.

    Args:
      device_name: (string) name of the device.
      node_exec_stats: `NodeExecStats` proto.
      file_path: path to the source file involved in creating the op.
      line_number: line number in the file involved in creating the op.
      func_name: name of the function that the line belongs to.
      op_type: (string) Operation type.
    """
    self.device_name = device_name
    self.node_exec_stats = node_exec_stats
    self.file_path = file_path
    self.line_number = line_number
    self.func_name = func_name
    if self.file_path:
      self.file_line_func = "%s:%d(%s)" % (
          os.path.basename(self.file_path), self.line_number, self.func_name)
    else:
      self.file_line_func = ""
    self.op_type = op_type
    self.start_time = self.node_exec_stats.all_start_micros
    self.op_time = (self.node_exec_stats.op_end_rel_micros -
                    self.node_exec_stats.op_start_rel_micros)

  @property
  def exec_time(self):
    # [FUNCTION DOC]
    # Property: total execution time in microseconds, including pre- and post-processing, taken
    # from the statistics' all-end relative micros field.
    """Op execution time plus pre- and post-processing."""
    return self.node_exec_stats.all_end_rel_micros


class AggregateProfile(object):
  # [CLASS DOC]
  # Summary of several ProfileDatum records. Keeps total op time, total execution time and a
  # private mapping from "device:node name" to how many times that node was seen. Created from
  # the first record and extended with the add method; exposes distinct node count and total
  # execution count as properties.
  """Profile summary data for aggregating a number of ProfileDatum."""

  def __init__(self, profile_datum):
    # [FUNCTION DOC]
    # Initializes the totals from a first ProfileDatum: total op time, total execution time, and a
    # count mapping with one entry (device name and node name joined) set to one.
    """Constructor.

    Args:
      profile_datum: (`ProfileDatum`) an instance of `ProfileDatum` to
        initialize this object with.
    """

    self.total_op_time = profile_datum.op_time
    self.total_exec_time = profile_datum.exec_time
    device_and_node = "%s:%s" % (profile_datum.device_name,
                                 profile_datum.node_exec_stats.node_name)
    self._node_to_exec_count = {device_and_node: 1}

  def add(self, profile_datum):
    # [FUNCTION DOC]
    # Adds another ProfileDatum: increases the total op time and total execution time and
    # increments the execution count for its device-and-node key (creating it at one). The key is
    # computed twice in a redundant way with the same result. Returns nothing.
    """Accumulate a new instance of ProfileDatum.

    Args:
      profile_datum: (`ProfileDatum`) an instance of `ProfileDatum` to
        accumulate to this object.
    """

    self.total_op_time += profile_datum.op_time
    self.total_exec_time += profile_datum.exec_time
    device_and_node = "%s:%s" % (profile_datum.device_name,
                                 profile_datum.node_exec_stats.node_name)

    device_and_node = "%s:%s" % (profile_datum.device_name,
                                 profile_datum.node_exec_stats.node_name)
    if device_and_node in self._node_to_exec_count:
      self._node_to_exec_count[device_and_node] += 1
    else:
      self._node_to_exec_count[device_and_node] = 1

  @property
  def node_count(self):
    # [FUNCTION DOC]
    # Property: number of distinct device-and-node keys aggregated so far.
    return len(self._node_to_exec_count)

  @property
  def node_exec_count(self):
    # [FUNCTION DOC]
    # Property: total number of executions aggregated (sum of per-node counts).
    return sum(self._node_to_exec_count.values())
