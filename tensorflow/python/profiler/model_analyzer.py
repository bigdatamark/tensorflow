# [MODULE DOC]
# Public entry points of the legacy TensorFlow profiler (tfprof) for analyzing a model.
# Exposes the Profiler class (multi-step profiling of a graph), the profile function (one-shot
# profiling by operation type, name scope, graph or Python code view), the advise function
# (automatic anomaly detection) and the ALL_ADVICE default checker configuration. Internally it
# converts Python option mappings into option protocol buffers, serializes graphs and run
# metadata, and calls the native tfprof extension, parsing the returned serialized protos.
# The public names are exported to the version 1 API through registration decorators at import.
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
"""Model Analyzer.

Analyze model, including shape, params, time, memory, structure, etc.
"""
import sys

from google.protobuf import message
from tensorflow.core.profiler import tfprof_options_pb2
from tensorflow.core.profiler import tfprof_output_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util import _pywrap_tfprof as print_mdl
from tensorflow.python.util.tf_export import tf_export

_DEFAULT_PROFILE_OPTIONS = 0
_DEFAULT_ADVISE_OPTIONS = 0

# The following options are for 'advise' cmd.
# Show all advice.
ALL_ADVICE = {
    'ExpensiveOperationChecker': {},
    'AcceleratorUtilizationChecker': {},
    'JobChecker': {},  # Only available internally.
    'OperationChecker': {},
}


def _graph_string(graph):
  # [FUNCTION DOC]
  # HELPER: Serialization glue for the native profiler calls in this module.
  # Input: graph, a graph object or nothing/null. Output: the serialized graph definition bytes
  # (with tensor shapes added) when a graph is given, otherwise empty bytes. No side effects.
  """Helper to serialize a graph to string."""
  if graph:
    return graph.as_graph_def(add_shapes=True).SerializeToString()
  else:
    return b''


def _build_options(options):
  # [FUNCTION DOC]
  # HELPER: Converts user option mappings into the protocol buffer the native profiler expects.
  # Input: options, a mapping from option names to values (depth, minimum thresholds for bytes,
  # time, parameters, float ops and occurrences, step id, ordering key, lists of regular
  # expressions for accounting/start/trim/show/hide, a flag to account only displayed ops, the
  # list of fields to select, output kind and dump file). Missing keys get defaults (maximum
  # depth 10, thresholds 0, step -1, order by name, empty lists, output to standard output,
  # empty dump file name). Output: a populated options protocol buffer. Unknown keys are ignored.
  # No side effects; raises type errors if values do not match the proto field types.
  """Build tfprof.OptionsProto.

  Args:
    options: A dictionary of options.

  Returns:
    tfprof.OptionsProto.
  """
  opts = tfprof_options_pb2.OptionsProto()
  opts.max_depth = options.get('max_depth', 10)
  opts.min_bytes = options.get('min_bytes', 0)
  opts.min_peak_bytes = options.get('min_peak_bytes', 0)
  opts.min_residual_bytes = options.get('min_residual_bytes', 0)
  opts.min_output_bytes = options.get('min_output_bytes', 0)
  opts.min_micros = options.get('min_micros', 0)
  opts.min_accelerator_micros = options.get('min_accelerator_micros', 0)
  opts.min_cpu_micros = options.get('min_cpu_micros', 0)
  opts.min_params = options.get('min_params', 0)
  opts.min_float_ops = options.get('min_float_ops', 0)
  opts.min_occurrence = options.get('min_occurrence', 0)

  opts.step = options.get('step', -1)

  opts.order_by = options.get('order_by', 'name')

  for p in options.get('account_type_regexes', []):
    opts.account_type_regexes.append(p)
  for p in options.get('start_name_regexes', []):
    opts.start_name_regexes.append(p)
  for p in options.get('trim_name_regexes', []):
    opts.trim_name_regexes.append(p)
  for p in options.get('show_name_regexes', []):
    opts.show_name_regexes.append(p)
  for p in options.get('hide_name_regexes', []):
    opts.hide_name_regexes.append(p)
  opts.account_displayed_op_only = options.get('account_displayed_op_only',
                                               False)

  for p in options.get('select', []):
    opts.select.append(p)

  opts.output = options.get('output', 'stdout')
  opts.dump_to_file = options.get('dump_to_file', '')

  return opts


def _build_advisor_options(options):
  # [FUNCTION DOC]
  # HELPER: Builds the advisor options protocol buffer for the advise calls.
  # Input: options, a mapping from checker names to mappings of checker settings, or nothing/null.
  # Output: an advisor options proto; empty when options is null, otherwise each checker's
  # settings are copied into a checker option message and merged into the checkers map.
  # KNOWN ISSUE: assigns settings by indexing the checker option message directly, while that
  # message type is a nested protocol buffer message rather than a mapping; this only works if
  # the generated type supports item assignment, which for non-empty checker settings may fail.
  """Build tfprof.AdvisorOptionsProto.

  Args:
    options: A dictionary of options. See ALL_ADVICE example.

  Returns:
    tfprof.AdvisorOptionsProto.
  """
  opts = tfprof_options_pb2.AdvisorOptionsProto()
  if options is None:
    return opts
  for checker, checker_opts in options.items():
    checker_ops_pb = tfprof_options_pb2.AdvisorOptionsProto.CheckerOption()
    for k, v in checker_opts.items():
      checker_ops_pb[k] = v
    opts.checkers[checker].MergeFrom(checker_ops_pb)
  return opts


@tf_export(v1=['profiler.Profiler'])
class Profiler:
  # [CLASS DOC]
  # Multi-step profiler wrapping the process-wide native tfprof profiler.
  # State: the graph being profiled and a coverage figure returned by the last added step. A
  # single native profiler exists per process, created in the constructor and destroyed when
  # the instance is garbage collected. Typical use: construct, add steps with run metadata,
  # then query by Python code, operation type, name scope, graph view, or ask for advice, and
  # optionally serialize the profile. Registered in the version 1 public API.
  """TensorFlow multi-step profiler.


  ```python
  Typical use case:
    # Currently we are only allowed to create 1 profiler per process.
    profiler = Profiler(sess.graph)

    for i in range(total_steps):
      if i % 10000 == 0:
        run_meta = tf.compat.v1.RunMetadata()
        _ = sess.run(...,
                     options=tf.compat.v1.RunOptions(
                         trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_meta)
        profiler.add_step(i, run_meta)

        # Profile the parameters of your model.
        profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
            .trainable_variables_parameter()))

        # Or profile the timing of your model operations.
        opts = option_builder.ProfileOptionBuilder.time_and_memory()
        profiler.profile_operations(options=opts)

        # Or you can generate a timeline:
        opts = (option_builder.ProfileOptionBuilder(
                option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(i)
                .with_timeline_output(filename).build())
        profiler.profile_graph(options=opts)
      else:
        _ = sess.run(...)
    # Auto detect problems and generate advice.
    profiler.advise()
  ```
  """

  def __init__(self, graph=None, op_log=None):
    # [FUNCTION DOC]
    # Creates the native profiler for a graph. Inputs: graph, optional graph (if null and eager
    # execution is off, the default graph is used; in eager mode it stays null); op_log, optional
    # op log protocol buffer defining extra operation types. Merges the op log with defaults
    # derived from the graph, then creates the process-wide native profiler with the serialized
    # graph and op log. Side effects: replaces/creates global native profiler state; sets the
    # coverage to zero.
    """Constructor.

    Args:
      graph: tf.Graph. If None and eager execution is not enabled, use default
        graph.
      op_log: optional. tensorflow::tfprof::OpLogProto proto. Used to define
        extra op types.
    """
    if not graph and not context.executing_eagerly():
      graph = ops.get_default_graph()
    self._coverage = 0.0
    self._graph = graph
    # pylint: disable=protected-access
    op_log = tfprof_logger.merge_default_with_oplog(self._graph, op_log=op_log)
    # pylint: enable=protected-access
    print_mdl.NewProfiler(
        _graph_string(self._graph), op_log.SerializeToString())

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: deletes the process-wide native profiler when the instance is destroyed.
    # Takes no inputs, returns nothing. Side effect: frees global native state, so any other
    # Profiler instance would lose its backing profiler as well.
    print_mdl.DeleteProfiler()

  def add_step(self, step, run_meta):
    # [FUNCTION DOC]
    # Records the statistics of one session run. Inputs: step, integer id used to group run
    # metadata together for later queries; run_meta, run metadata protocol buffer with the
    # collected statistics. Merges an op log built from the run metadata, passes the serialized
    # graph, run metadata and op log to the native profiler, and stores the returned coverage
    # value on the instance. Returns nothing. Side effect: updates native profiler state.
    """Add statistics of a step.

    Args:
      step: int, An id used to group one or more different `run_meta` together.
        When profiling with the profile_xxx APIs, user can use the `step` id in
        the `options` to profile these `run_meta` together.
      run_meta: RunMetadata proto that contains statistics of a session run.
    """
    # pylint: disable=protected-access
    op_log = tfprof_logger.merge_default_with_oplog(
        self._graph, run_meta=run_meta)
    # pylint: enable=protected-access
    # TODO(xpan): P1: Better to find the current graph.
    self._coverage = print_mdl.AddStep(step, _graph_string(self._graph),
                                       run_meta.SerializeToString(),
                                       op_log.SerializeToString())

  def profile_python(self, options):
    # [FUNCTION DOC]
    # Profiles by Python code view (call stacks). Input: options, mapping of profiler options.
    # Builds the options proto, asks the native profiler for the code view and parses the result
    # into a multi-graph node proto. Output: that proto; if the returned bytes cannot be parsed,
    # an error message is written to standard error and a possibly empty/partial proto is returned.
    """Profile the statistics of the Python codes.

      By default, it shows the call stack from root. To avoid
      redundant output, you may use options to filter as below
        options['show_name_regexes'] = ['.*my_code.py.*']

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.

    Returns:
      a MultiGraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
    try:
      tfprof_node.ParseFromString(
          print_mdl.Profile('code'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as e:
      sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    return tfprof_node

  def profile_operations(self, options):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring's first sentence is broken across the summary line and body
    # ("e.g." split), so the generated summary is truncated; behavior itself is unaffected.
    # Profiles by operation type view. Input: options, mapping of profiler options. Builds the
    # options proto, queries the native profiler for the op view and parses the result into a
    # multi-graph node proto, which is returned. On a parse failure an error message is written
    # to standard error and the possibly empty proto is returned.
    """Profile the statistics of the Operation types (e.g.

    MatMul, Conv2D).

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.

    Returns:
      a MultiGraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
    try:
      tfprof_node.ParseFromString(
          print_mdl.Profile('op'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as e:
      sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    return tfprof_node

  def profile_name_scope(self, options):
    # [FUNCTION DOC]
    # Profiles graph nodes organized by name scope. Input: options, mapping of profiler options.
    # Queries the native profiler for the scope view and parses it into a graph node proto,
    # which is returned. On a parse failure writes an error message to standard error and returns
    # the possibly empty proto.
    """Profile the statistics of graph nodes, organized by name scope.

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.

    Returns:
      a GraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.GraphNodeProto()
    try:
      tfprof_node.ParseFromString(
          print_mdl.Profile('scope'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as e:
      sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    return tfprof_node

  def profile_graph(self, options):
    # [FUNCTION DOC]
    # Profiles graph nodes organized by dataflow graph (inputs/outputs). Input: options, mapping
    # of profiler options. Queries the native profiler for the graph view and parses it into a
    # graph node proto, which is returned. On a parse failure writes an error message to
    # standard error and returns the possibly empty proto.
    """Profile the statistics of graph nodes, organized by dataflow graph.

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.

    Returns:
      a GraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.GraphNodeProto()
    try:
      tfprof_node.ParseFromString(
          print_mdl.Profile('graph'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as e:
      sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    return tfprof_node

  def advise(self, options):
    # [FUNCTION DOC]
    # Runs automatic problem detection on the accumulated profile. Input: options, mapping from
    # checker names to checker settings (null gives empty options). Output: an advice proto with
    # reports from all checkers. Unlike the profile methods, a parse failure is not caught and
    # raises the protocol buffer decode error.
    """Automatically detect problems and generate reports.

    Args:
      options: A dict of options. See ALL_ADVICE example above.

    Returns:
      An Advise proto that contains the reports from all checkers.
    """
    advise_pb = tfprof_output_pb2.AdviceProto()
    opts = _build_advisor_options(options)
    advise_pb.ParseFromString(
        print_mdl.Profile('advise'.encode('utf-8'), opts.SerializeToString()))
    return advise_pb

  def serialize_to_string(self):
    # [FUNCTION DOC]
    # Serializes the native profiler's accumulated profile proto. Takes no inputs; returns the
    # binary string, suitable for writing to a file for offline analysis. No side effects.
    """Serialize the ProfileProto to a binary string.

      Users can write it to file for offline analysis by tfprof commandline
      or graphical interface.

    Returns:
      ProfileProto binary string.
    """
    return print_mdl.SerializeToString()

  def _write_profile(self, filename):
    # [FUNCTION DOC]
    # HELPER: Private utility for dumping the profile to disk (used by tooling and tests).
    # Input: filename, string path. Asks the native profiler to write its profile there.
    # Returns nothing; side effect is creating or overwriting that file.
    """Writes the profile to a file."""
    print_mdl.WriteProfile(filename)


@tf_export(v1=['profiler.profile'])
def profile(graph=None,
            run_meta=None,
            op_log=None,
            cmd='scope',
            options=_DEFAULT_PROFILE_OPTIONS):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring names the option "accounted_type_regexes" but the options builder
  # reads the key "account_type_regexes", so following the docstring has no effect.
  # One-shot model profiling. Inputs: graph, optional graph (default graph when null and not
  # eager); run_meta, optional run metadata proto enabling time/memory data; op_log, optional op
  # log proto assigning types to nodes; cmd, one of op, scope, graph or code selecting the view
  # (default scope); options, mapping of options (default means trainable-variable parameter
  # profile). Merges the op log (with traces for the code view), builds options, calls the native
  # analysis and parses the result. Output: a multi-graph node proto for op and code, a graph
  # node proto for graph and scope. Raises an invalid-argument error for an unknown cmd. A parse
  # failure is reported on standard error. Side effect: output to standard output, a file or a
  # timeline file depending on the output option.
  """Profile model.

    Tutorials and examples can be found in:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use default
      graph.
    run_meta: optional tensorflow.RunMetadata proto. It is necessary to
      support run time information profiling, such as time and memory.
    op_log: tensorflow.tfprof.OpLogProto proto. User can assign "types" to graph
      nodes with op_log. "types" allow user to flexibly group and account
      profiles using options['accounted_type_regexes'].
    cmd: string. Either 'op', 'scope', 'graph' or 'code'. 'op' view organizes
      profile using operation type. (e.g. MatMul) 'scope' view organizes profile
      using graph node name scope. 'graph' view organizes profile using graph
      node inputs/outputs. 'code' view organizes profile using Python call
      stack.
    options: A dict of options. See core/profiler/g3doc/options.md.

  Returns:
    If cmd is 'scope' or 'graph', returns GraphNodeProto proto.
    If cmd is 'op' or 'code', returns MultiGraphNodeProto proto.
    Side effect: stdout/file/timeline.json depending on options['output']
  """
  if not graph and not context.executing_eagerly():
    graph = ops.get_default_graph()

  if options == _DEFAULT_PROFILE_OPTIONS:
    options = (
        option_builder.ProfileOptionBuilder.trainable_variables_parameter())
  # pylint: disable=protected-access
  op_log = tfprof_logger.merge_default_with_oplog(
      graph, op_log, run_meta, add_trace=cmd == 'code')
  # pylint: enable=protected-access

  opts = _build_options(options)

  run_meta_str = run_meta.SerializeToString() if run_meta else b''

  graph_str = _graph_string(graph)

  if cmd == 'code' or cmd == 'op':
    tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
    ret = print_mdl.PrintModelAnalysis(graph_str, run_meta_str,
                                       op_log.SerializeToString(),
                                       cmd.encode('utf-8'),
                                       opts.SerializeToString())
    try:
      tfprof_node.ParseFromString(ret)
    except message.DecodeError as e:
      sys.stderr.write('Cannot parse returned proto: %s.\n' % e)

  elif cmd == 'graph' or cmd == 'scope':
    tfprof_node = tfprof_output_pb2.GraphNodeProto()
    ret = print_mdl.PrintModelAnalysis(graph_str, run_meta_str,
                                       op_log.SerializeToString(),
                                       cmd.encode('utf-8'),
                                       opts.SerializeToString())
    try:
      tfprof_node.ParseFromString(ret)
    except message.DecodeError as e:
      sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
  else:
    raise errors.InvalidArgumentError(None, None, 'unknown cmd: %s\n' % cmd)

  return tfprof_node


@tf_export(v1=['profiler.advise'])
def advise(graph=None, run_meta=None, options=_DEFAULT_ADVISE_OPTIONS):
  # [FUNCTION DOC]
  # Builds profiles and runs anomaly checkers. Inputs: graph, optional graph (default graph when
  # null and not eager); run_meta, optional run metadata proto; options, mapping of checker names
  # to settings (default means all checkers in the default advice configuration). Merges an op
  # log with traces, calls the native analysis in advise mode and parses the result.
  # Output: an advice proto. A parse failure is not caught and raises the decode error.
  """Auto profile and advise.

    Builds profiles and automatically check anomalies of various
    aspects. For more details:
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/README.md

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use default
      graph.
    run_meta: optional tensorflow.RunMetadata proto. It is necessary to
      support run time information profiling, such as time and memory.
    options: see ALL_ADVICE example above. Default checks everything.

  Returns:
    Returns AdviceProto proto
  """
  if not graph and not context.executing_eagerly():
    graph = ops.get_default_graph()

  if options == _DEFAULT_ADVISE_OPTIONS:
    options = ALL_ADVICE.copy()

  # pylint: disable=protected-access
  op_log = tfprof_logger.merge_default_with_oplog(
      graph, None, run_meta, add_trace=True)
  # pylint: enable=protected-access

  run_meta_str = run_meta.SerializeToString() if run_meta else b''

  opts = _build_advisor_options(options)
  ret = tfprof_output_pb2.AdviceProto()
  ret.ParseFromString(
      print_mdl.PrintModelAnalysis(
          _graph_string(graph), run_meta_str, op_log.SerializeToString(),
          'advise'.encode('utf-8'), opts.SerializeToString()))
  return ret
