# [MODULE DOC]
# Converts TensorFlow graph and run metadata (collected with full tracing) into profiles in the
# pprof protocol buffer format. Defines small bookkeeping classes that deduplicate strings,
# functions, locations and samples (StringTable, Functions, Locations, Samples), the PprofProfiler
# class that builds one profile per device from per-node execution statistics and the creating
# Python tracebacks of graph operations, and two public functions: get_profiles returns the
# protos per device, and profile also writes them as compressed files in an output directory
# or prints them. Used by the option builder's pprof output. Imports the pprof profile proto
# from a top-level package named proto at import time.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Profiler for TensorFlow models that outputs data in pprof format.

See https://github.com/google/pprof/blob/master/proto/profile.proto for pprof
profile format.
The following needs to be set for profiler to work:
  * trace_level needs to be set to FULL_TRACE
  * run_metadata object should be passed in to session.run call

Sample usage:
  options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.compat.v1.RunMetadata()

  with tf.compat.v1.Session as sess:
    ...
    sess.run(computation, run_metadata=run_metadata, options=options)
  pprof_profiler.profile(sess.graph, run_metadata, output_dir)


  The code above would output a pprof profile to separate output_dir/.*.pb.gz
  file for each device. These files can be passed to pprof for formatting.
  For e.g.:
     pprof -png --nodecount=100 --sample_index=1 output_dir/profile_output.pb.gz
"""
from collections import defaultdict
from collections import namedtuple
import gzip
import os
import string
import sys
import time

from proto import profile_pb2


if sys.version_info < (3,):
  maketrans = string.maketrans
else:
  maketrans = str.maketrans


ProfileDatum = namedtuple('ProfileDatum', [
    'node_exec_stats', 'op_type', 'traceback'])


class StringTable(object):
  # [CLASS DOC]
  # HELPER: Bookkeeping helper for building the pprof profile.
  # Interns strings: holds an ordered list of unique strings (the first entry is always the empty
  # string, as pprof requires) and a mapping from each string to its position, so each distinct
  # string gets a stable index in the profile's string table.
  """Keeps track of strings to add to string_table in pprof proto."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates the table containing only the empty string at index zero. No inputs, no outputs.
    # Pprof requires first entry in string_table to be ''.
    self._string_table = ['']
    self._string_to_index = {'': 0}

  def index_of(self, value_str):
    # [FUNCTION DOC]
    # Looks up or interns a string. Input: value_str, a string (a null value is treated as the
    # empty string). Returns the integer index of the string, appending it at the end of the table
    # first if it was not present. Side effect: may grow the table and the index mapping.
    """Get index of value_str in the string table.

    If value_str is not in the string table, we will add it at the end
    and then return the new index.
    Args:
      value_str: (string) Value to lookup/add in/to the string table.

    Returns:
      Index of value_str in the string table.
    """
    if value_str is None:
      value_str = ''
    if value_str in self._string_to_index:
      return self._string_to_index[value_str]
    index = len(self._string_table)
    self._string_table.append(value_str)
    self._string_to_index[value_str] = index
    return index

  def next_index(self):
    # [FUNCTION DOC]
    # Returns the integer index that the next newly added string would receive (the current table
    # length). Does not modify the table.
    """Gets index that would be assigned to the next added string.

    Returns:
      Index of the next string if it was added.
    """
    return len(self._string_table)

  def string_table(self):
    # [FUNCTION DOC]
    # Returns the underlying list of strings in index order (the live list, not a copy), ready to
    # be stored as the profile's string table.
    """Returns a list of strings to store in pprof's string_table."""
    return self._string_table


class Functions(object):
  # [CLASS DOC]
  # HELPER: Bookkeeping helper for building the pprof profile.
  # Registry of function records for the profile. Holds a reference to the shared string table
  # and a mapping from (file path, function name, start line) to the function proto, assigning
  # identifiers starting at one in insertion order.
  """Keeps track of `Function` protos for pprof profile."""

  def __init__(self, string_table):
    # [FUNCTION DOC]
    # Creates an empty registry. Input: string_table, the shared string table object used to
    # intern function names and file paths. No outputs.
    """Constructor.

    Args:
      string_table: A `StringTable` object.
    """
    self._string_table = string_table
    # Maps tuples in the form (file_path, function_name, start_line_number)
    # to `Function` protos.
    self._function_key_to_function = {}

  def index_of(self, file_path, function_name, function_start_line):
    # [FUNCTION DOC]
    # Looks up or creates a function record. Inputs: file_path, string path where the function is
    # defined; function_name, string; function_start_line, integer start line. Returns the
    # function's integer identifier (starting at one). A new record stores the interned name and
    # file name and the start line. Side effects: may add a record and strings to the string table.
    """Returns index of the function, adding the function if needed.

    Args:
      file_path: (string) Path to file where the function is defined.
      function_name: (string) Function name.
      function_start_line: (integer) Start line number of function definition.

    Returns:
      Function index.
    """
    function_key = (file_path, function_name, function_start_line)
    if function_key in self._function_key_to_function:
      return self._function_key_to_function[function_key].id
    else:
      # Function indexes should start from 1
      function_index = len(self._function_key_to_function) + 1
      function = profile_pb2.Function()
      function.id = function_index
      function.name = self._string_table.index_of(function_name)
      function.filename = self._string_table.index_of(file_path)
      function.start_line = function_start_line
      self._function_key_to_function[function_key] = function
      return function_index

  def function_protos(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring promises a list, but this returns a dictionary values view, which
    # is not indexable (callers here only extend a repeated field with it).
    # Returns all registered function protos in insertion order. Does not modify state.
    """Returns list of `profile_pb2.Function` protos."""
    return self._function_key_to_function.values()


class Locations(object):
  # [CLASS DOC]
  # HELPER: Bookkeeping helper for building the pprof profile.
  # Registry of call-location records. Holds the function registry and a mapping from (calling
  # file path, called function name, line number) to the location proto; identifiers start at
  # one in insertion order, and each location has one line entry pointing to a function record.
  """Keeps track of `Location` protos for pprof profile.

  `Locations` store information about function call locations.
  """

  def __init__(self, functions):
    # [FUNCTION DOC]
    # Creates an empty registry. Input: functions, the Functions registry used to register called
    # functions. No outputs.
    """Constructor.

    Args:
      functions: A `Functions` object.
    """
    self._functions = functions
    # Maps tuples in the form (file_path, called_function_name, line_number)
    # to `Location` protos.
    self._location_key_to_location = {}

  def index_of(
      self, file_path, line_number, called_function_name, called_file_path,
      called_function_start_line):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the lookup key omits the called file path and start line, so two calls from the
    # same file and line to same-named functions in different files share one location.
    # Looks up or creates a location. Inputs: file_path, string path of the calling file;
    # line_number, integer call line; called_function_name, string; called_file_path, string path
    # where the called function is defined; called_function_start_line, integer. Returns the
    # location's integer identifier. A new location registers the called function and records
    # the call line. Side effects: may add a location and function.
    """Returns index of the location, adding the location if needed.

    Args:
      file_path: (string) Path to file that makes the call.
      line_number: (integer) Call line number.
      called_function_name: (string) Function name of the function called at
        `file_path` and `line_number`.
      called_file_path: (string) Path to file where the called function is
        defined.
      called_function_start_line: (integer) Start line number of called
        function definition in `called_file_path` file.

    Returns:
      Index of location.
    """
    location_key = (file_path, called_function_name, line_number)
    if location_key in self._location_key_to_location:
      location = self._location_key_to_location[location_key]
      return location.id
    else:
      # Location indexes should start from 1
      location_index = len(self._location_key_to_location) + 1
      location = profile_pb2.Location()
      location.id = location_index
      self._location_key_to_location[location_key] = location

      line = location.line.add()
      line.function_id = self._functions.index_of(
          called_file_path, called_function_name, called_function_start_line)
      line.line = line_number
      return location_index

  def location_protos(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring promises a list, but this returns a dictionary values view.
    # Returns all registered location protos in insertion order. Does not modify state.
    """Returns list of `profile_pb2.Location` protos."""
    return self._location_key_to_location.values()


class Samples(object):
  # [CLASS DOC]
  # HELPER: Bookkeeping helper for building the pprof profile.
  # Registry of sample records, one per graph node name. Each sample holds three values (count,
  # total time and operation time) and labels for node name and operation type; repeated
  # executions of the same node are aggregated into the same sample.
  """Keeps track of `Sample` protos for pprof profile.

  Samples store the following statistics in order:
  count, all_time, op_time
  """

  def __init__(self, string_table):
    # [FUNCTION DOC]
    # Creates an empty sample registry. Input: string_table, the shared string table used for
    # label keys and values. Keeps a mapping from graph node name to its sample proto.
    """Constructor.

    Args:
      string_table: A `StringTable` object.
    """
    self._string_table = string_table
    # TODO(annarev): figure out if location is unique for each node name.
    # If not, also key this dictionary based on location ids.
    self._node_name_to_sample = {}

  def add(self, datum, location_ids):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when a sample for the node already exists, its location list is extended again
    # with the new ids, so repeated executions duplicate the same locations in the sample.
    # Records one execution of a node. Inputs: datum, a profile datum (execution statistics, op
    # type, traceback); location_ids, list of integer location identifiers. Creates the sample
    # with zero values and node name / op type labels if new, then increments the count by one,
    # adds the total end time and adds the operation duration (end minus start) to the values.
    """Adds a sample data point.

    Args:
      datum: `ProfileDatum` to add a sample for.
      location_ids: List of numeric location ids for this sample.
    """
    node_name = datum.node_exec_stats.node_name
    if node_name in self._node_name_to_sample:
      sample = self._node_name_to_sample[node_name]
      sample.location_id.extend(location_ids)
    else:
      sample = profile_pb2.Sample()
      # Sample stores 3 values: count, all_time, op_time
      sample.value.extend([0, 0, 0])

      label = sample.label.add()
      label.key = self._string_table.index_of('node_name')
      label.str = self._string_table.index_of(node_name)
      label = sample.label.add()
      label.key = self._string_table.index_of('op_type')
      label.str = self._string_table.index_of(datum.op_type)
      self._node_name_to_sample[node_name] = sample
    sample.value[0] += 1
    sample.value[1] += datum.node_exec_stats.all_end_rel_micros
    sample.value[2] += (
        datum.node_exec_stats.op_end_rel_micros -
        datum.node_exec_stats.op_start_rel_micros)

  def get_sample_protos(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring promises a list, but this returns a dictionary values view.
    # Returns all sample protos, one per node name, in insertion order. Does not modify state.
    """Returns list of `Sample` protos for pprof profile."""
    return self._node_name_to_sample.values()


class PprofProfiler(object):
  # [CLASS DOC]
  # Builds pprof profiles from a graph and run metadata. Holds the graph, the run metadata and
  # shared string table, function and location registries (shared across devices, so identifiers
  # and strings accumulate over one instance). Use profile to get one proto per device.
  """Creates profiles in pprof format."""

  def __init__(self, graph, run_metadata):
    # [FUNCTION DOC]
    # Creates a profiler. Inputs: graph, the graph whose operations supply op types and creation
    # tracebacks; run_metadata, run metadata whose step statistics hold per-device node timings
    # (the docstring says a list, but the code reads the step statistics of a single metadata
    # object). Initializes empty string, function and location registries.
    """Constructor.

    Args:
      graph: A `Graph` instance.
      run_metadata: A list of `RunMetadata` objects.
    """
    self._graph = graph
    self._run_metadata = run_metadata
    self._string_table = StringTable()
    self._functions = Functions(self._string_table)
    self._locations = Locations(self._functions)

  def profile(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the constructor docstring says run_metadata is a list, but a single object is
    # used. Also string table and registries are shared across devices, so each device's proto
    # contains entries from earlier devices, and calling this twice accumulates state.
    # Generates a pprof proto per device in the run metadata. For each device builds the proto from
    # that device's node statistics; if it has no samples, prints a message suggesting that run
    # metadata be passed to the session run and skips it; otherwise appends a comment string
    # "Device i of n: name" to the proto. Returns a dictionary from device name to proto.
    # Side effect: printing to standard output.
    """Generates pprof profiles.

    Returns:
      Dictionary mapping from device name to proto in `profile_pb2.Profile`
      format.
    """
    profiles = {}
    data_generator_func = self._get_profile_data_generator()
    for device_index, device_stats in enumerate(
        self._run_metadata.step_stats.dev_stats):
      # Create profile
      pprof_proto = self._get_pprof_proto(data_generator_func(device_stats))
      if not pprof_proto.sample:
        print(
            'Not enough data to create profile for device %s. Did you pass '
            'RunMetadata to session.run call?' % device_stats.device)
        continue
      # Add device name comment
      device_count = len(self._run_metadata.step_stats.dev_stats)
      device_description = (
          'Device %d of %d: %s' %
          (device_index + 1, device_count, device_stats.device))
      device_description_str_index = self._string_table.next_index()
      pprof_proto.string_table.append(device_description)
      pprof_proto.comment.append(device_description_str_index)
      profiles[device_stats.device] = pprof_proto
    return profiles

  def _get_pprof_proto(self, profile_datum_generator):
    # [FUNCTION DOC]
    # HELPER: Core proto builder used by profile for a single device.
    # Input: profile_datum_generator, iterable of profile data (node stats, op type, traceback).
    # For each datum with a non-empty traceback, walks the traceback from the innermost frame
    # outward, skipping frames until the op-application function is passed, and registers a
    # location for each remaining frame (the frame's file and line calling the previous frame's
    # function; the called function's start line is always recorded as minus one). Adds a sample
    # for the datum. Then declares sample types count, all_time and op_time (units count and
    # nanoseconds) and fills string table, samples, functions and locations. Returns the profile
    # proto. Side effects: grows shared string table, function and location registries.
    # KNOWN ISSUE: the timing units are declared as nanoseconds but the values come from
    # microsecond fields of the statistics, so the unit label is likely wrong.
    """Returns profile data in pprof proto format.

    Args:
      profile_datum_generator: Generator outputting `ProfileDatum` objects.

    Returns:
      A proto in pprof format.
    """
    pprof_profile = profile_pb2.Profile()
    samples = Samples(self._string_table)

    for datum in profile_datum_generator:
      if not datum.traceback:
        continue

      stack_frame = datum.traceback[-1]
      after_apply_op = False
      location_ids = []

      # We add locations from stack trace in bottom-up order.
      for stack_frame_index in reversed(range(len(datum.traceback) - 1)):
        prev_stack_frame = stack_frame
        stack_frame = datum.traceback[stack_frame_index]

        # Call at current frame calls function at previous frame.
        prev_file_path = prev_stack_frame[0]
        prev_function = prev_stack_frame[2]
        prev_function_start_line = -1
        curr_file_path = stack_frame[0]
        curr_line_number = stack_frame[1]

        # Skip all calls up to apply_op since they are the same for all ops.
        if not after_apply_op:
          if prev_function == 'apply_op':
            after_apply_op = True
          continue
        location_index = self._locations.index_of(
            curr_file_path, curr_line_number,
            prev_function, prev_file_path, prev_function_start_line)
        location_ids.append(location_index)
      samples.add(datum, location_ids)

    sample_type_description = 'count'
    sample_type = pprof_profile.sample_type.add()
    sample_type.type = self._string_table.index_of(sample_type_description)
    sample_type.unit = self._string_table.index_of('count')
    sample_type_description = 'all_time'
    sample_type = pprof_profile.sample_type.add()
    sample_type.type = self._string_table.index_of(sample_type_description)
    sample_type.unit = self._string_table.index_of('nanoseconds')
    sample_type_description = 'op_time'
    sample_type = pprof_profile.sample_type.add()
    sample_type.type = self._string_table.index_of(sample_type_description)
    sample_type.unit = self._string_table.index_of('nanoseconds')

    pprof_profile.string_table.extend(self._string_table.string_table())
    pprof_profile.sample.extend(samples.get_sample_protos())
    pprof_profile.function.extend(self._functions.function_protos())
    pprof_profile.location.extend(self._locations.location_protos())
    return pprof_profile

  def _get_profile_data_generator(self):
    # [FUNCTION DOC]
    # HELPER: Prepares per-operation lookups for the profile generation.
    # Scans the graph's operations to map each operation name to its creation traceback and type
    # (missing names default to an empty list and empty string). Returns a function that takes a
    # device's step statistics and yields profile data (execution stats, op type, traceback) for
    # each node, skipping the synthetic source and sink nodes.
    """Get function that generates `ProfileDatum` objects.

    Returns:
      A function that generates `ProfileDatum` objects.
    """
    node_to_traceback = defaultdict(list)
    node_to_op_type = defaultdict(str)
    for op in self._graph.get_operations():
      node_to_traceback[op.name] = op.traceback
      node_to_op_type[op.name] = op.type

    def profile_data_generator(device_step_stats):
      # [FUNCTION DOC]
      # HELPER: Generator returned by the enclosing method. Input: device_step_stats, a device's
      # step statistics. Yields one profile datum per node statistic, pairing it with the
      # operation type and traceback looked up by node name; skips the source and sink nodes.
      for node_stats in device_step_stats.node_stats:
        if node_stats.node_name == '_SOURCE' or node_stats.node_name == '_SINK':
          continue
        yield ProfileDatum(
            node_stats,
            node_to_op_type[node_stats.node_name],
            node_to_traceback[node_stats.node_name])

    return profile_data_generator


def get_profiles(graph, run_metadata):
  # [FUNCTION DOC]
  # Public entry point returning pprof protos. Inputs: graph, a graph object; run_metadata, run
  # metadata with step statistics from a fully traced run. Creates a profiler and returns the
  # dictionary from device name to profile proto (devices without samples are omitted, with a
  # message printed to standard output).
  """Generate profiles in pprof format.

  See https://github.com/google/pprof/blob/master/proto/profile.proto
  for pprof proto format.

  Args:
    graph: A `Graph` object.
    run_metadata: A `RunMetadata` proto.

  Returns:
    A dictionary mapping from device name to pprof proto for that device.
  """
  return PprofProfiler(graph, run_metadata).profile()


def profile(graph, run_metadata, output_dir=None):
  # [FUNCTION DOC]
  # Generates pprof profiles and writes or prints them. Inputs: graph and run_metadata as for
  # get_profiles; output_dir, optional directory string. If given, creates it when missing and
  # writes one gzip-compressed serialized proto per device, named with the sanitized device name
  # (slashes and colons replaced by underscores) and a timestamp suffix, printing each path.
  # Without a directory, prints a notice and each proto to standard output.
  # Returns the list of files created (empty when no directory is given).
  # KNOWN ISSUE: the timestamp has one-second resolution, so repeated calls in the same second
  # overwrite earlier files.
  """Generate profiles in pprof format.

  See https://github.com/google/pprof/blob/master/proto/profile.proto
  for pprof proto format.

  Args:
    graph: A `Graph` object.
    run_metadata: A `RunMetadata` proto.
    output_dir: (string) Directory to output pprof profile to.
      Profile files for each device will be stored in compressed
      serialized proto format. If output_dir is None, profile protos
      will be printed to stdout instead.

  Returns:
    List of output files created by this profile call.
    (Note: this list will be empty if output_dir is None)
  """
  profiles = get_profiles(graph, run_metadata)
  output_file_template = None
  if output_dir:
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    time_suffix = time.strftime('%Y%m%d%H%M%S')
    output_file_template = os.path.join(
        output_dir, '%s_' + time_suffix + '.pb.gz')

  profile_files = []
  for device, pprof_proto in profiles.items():
    if output_file_template is None:
      print('No output directory specified, printing to stdout instead.')
      print(pprof_proto)
    else:
      device_name = str(device).strip('/').translate(
          maketrans('/:', '__'))
      profile_file = output_file_template % device_name
      profile_files.append(profile_file)
      with gzip.open(profile_file, 'w') as output_file:
        print('Writing profile to %s...' % profile_file)
        output_file.write(pprof_proto.SerializeToString())
  return profile_files
