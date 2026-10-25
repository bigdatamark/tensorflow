# [MODULE DOC]
# HELPER: Helper module supplying profiler options to the model analyzer and profile context.
# Provides the fluent builder for profiler option mappings used by the legacy profiling API.
# Exposes one class, ProfileOptionBuilder, with pre-built option sets (trainable variable
# parameters, float operations, time and memory) and chainable setters for depth, minimum
# thresholds, type and name filters, ordering, selected attributes, step and output destination
# (standard output, none, file, timeline, pprof). Results are plain mappings consumed by the
# model analyzer. Registered in the version 1 public API at import time.
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
"""Utilities for building profiler options."""
import copy

from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=['profiler.ProfileOptionBuilder'])
class ProfileOptionBuilder(object):
  # [CLASS DOC]
  # Fluent builder of profiler option mappings. Holds a private mapping of option names to
  # values, initialized from defaults or a copy of a given mapping. Static methods return
  # ready-made option mappings; instance setter methods modify the held mapping and return the
  # builder for chaining; build returns a deep copy of the result. Registered in the version 1
  # public API.
  # pylint: disable=line-too-long
  """Option Builder for Profiling API.

  For tutorial on the options, see
  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/options.md

  ```python
  # Users can use pre-built options:
  opts = (
      tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

  # Or, build your own options:
  opts = (tf.compat.v1.profiler.ProfileOptionBuilder()
      .with_max_depth(10)
      .with_min_micros(1000)
      .select(['accelerator_micros'])
      .with_stdout_output()
      .build()

  # Or customize the pre-built options:
  opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
      tf.profiler.ProfileOptionBuilder.time_and_memory())
      .with_displaying_options(show_name_regexes=['.*rnn.*'])
      .build())

  # Finally, profiling with the options:
  _ = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(),
                          run_meta=run_meta,
                          cmd='scope',
                          options=opts)
  ```
  """
  # pylint: enable=line-too-long

  def __init__(self, options=None):
    # [FUNCTION DOC]
    # Creates a builder. Input: options, optional mapping of options to start from; it is deeply
    # copied so the caller's mapping is never changed. When omitted, starts from defaults: depth
    # 100, zero minimums, order by name, account all types, start and show everything, hide
    # nothing, account all ops, select time, step -1 (average) and output to standard output.
    """Constructor.

    Args:
      options: Optional initial option dict to start with.
    """
    if options is not None:
      self._options = copy.deepcopy(options)
    else:
      self._options = {'max_depth': 100,
                       'min_bytes': 0,
                       'min_micros': 0,
                       'min_params': 0,
                       'min_float_ops': 0,
                       'min_occurrence': 0,
                       'order_by': 'name',
                       'account_type_regexes': ['.*'],
                       'start_name_regexes': ['.*'],
                       'trim_name_regexes': [],
                       'show_name_regexes': ['.*'],
                       'hide_name_regexes': [],
                       'account_displayed_op_only': False,
                       'select': ['micros'],
                       'step': -1,
                       'output': 'stdout'}

  @staticmethod
  def trainable_variables_parameter():
    # [FUNCTION DOC]
    # Static factory. Takes no inputs; returns a new mapping of options for profiling trainable
    # variable parameters (depth 10000, accounts only the trainable variables type, only
    # displayed ops accounted, selects parameters, standard output). Normally used with the scope
    # view. No side effects.
    """Options used to profile trainable variable parameters.

    Normally used together with 'scope' view.

    Returns:
      A dict of profiling options.
    """
    return {'max_depth': 10000,
            'min_bytes': 0,
            'min_micros': 0,
            'min_params': 0,
            'min_float_ops': 0,
            'min_occurrence': 0,
            'order_by': 'name',
            'account_type_regexes': [tfprof_logger.TRAINABLE_VARIABLES],
            'start_name_regexes': ['.*'],
            'trim_name_regexes': [],
            'show_name_regexes': ['.*'],
            'hide_name_regexes': [],
            'account_displayed_op_only': True,
            'select': ['params'],
            'step': -1,
            'output': 'stdout'}

  @staticmethod
  def float_operation():
    # [FUNCTION DOC]
    # Static factory. Takes no inputs; returns a new mapping of options for profiling float
    # operations (depth 10000, minimum float ops 1, ordered by float ops, selects float ops,
    # accounts all types, standard output). No side effects.
    # pylint: disable=line-too-long
    """Options used to profile float operations.

    Please see https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md
    on the caveats of calculating float operations.

    Returns:
      A dict of profiling options.
    """
    # pylint: enable=line-too-long
    return {'max_depth': 10000,
            'min_bytes': 0,
            'min_micros': 0,
            'min_params': 0,
            'min_float_ops': 1,
            'min_occurrence': 0,
            'order_by': 'float_ops',
            'account_type_regexes': ['.*'],
            'start_name_regexes': ['.*'],
            'trim_name_regexes': [],
            'show_name_regexes': ['.*'],
            'hide_name_regexes': [],
            'account_displayed_op_only': True,
            'select': ['float_ops'],
            'step': -1,
            'output': 'stdout'}

  @staticmethod
  def time_and_memory(min_micros=1, min_bytes=1, min_accelerator_micros=0,
                      min_cpu_micros=0, min_peak_bytes=0, min_residual_bytes=0,
                      min_output_bytes=0):
    # [FUNCTION DOC]
    # Static factory for time and memory profiling options. Inputs (all optional numbers): the
    # minimum total execution micros (default 1), minimum requested bytes (default 1), minimum
    # accelerator micros, cpu micros, peak bytes, residual bytes and output bytes (default 0).
    # Returns a new mapping with depth 10000, ordered by micros, selecting micros and bytes,
    # accounting all types and only displayed ops, output to standard output. No side effects.
    """Show operation time and memory consumptions.

    Args:
      min_micros: Only show profiler nodes with execution time
          no less than this. It sums accelerator and cpu times.
      min_bytes: Only show profiler nodes requested to allocate no less bytes
          than this.
      min_accelerator_micros: Only show profiler nodes spend no less than
          this time on accelerator (e.g. GPU).
      min_cpu_micros: Only show profiler nodes spend no less than
          this time on cpu.
      min_peak_bytes: Only show profiler nodes using no less than this bytes
          at peak (high watermark). For profiler nodes consist of multiple
          graph nodes, it sums the graph nodes' peak_bytes.
      min_residual_bytes: Only show profiler nodes have no less than
          this bytes not being de-allocated after Compute() ends. For
          profiler nodes consist of multiple graph nodes, it sums the
          graph nodes' residual_bytes.
      min_output_bytes: Only show profiler nodes have no less than this bytes
          output. The output are not necessarily allocated by this profiler
          nodes.
    Returns:
      A dict of profiling options.
    """
    return {'max_depth': 10000,
            'min_bytes': min_bytes,
            'min_peak_bytes': min_peak_bytes,
            'min_residual_bytes': min_residual_bytes,
            'min_output_bytes': min_output_bytes,
            'min_micros': min_micros,
            'min_accelerator_micros': min_accelerator_micros,
            'min_cpu_micros': min_cpu_micros,
            'min_params': 0,
            'min_float_ops': 0,
            'min_occurrence': 0,
            'order_by': 'micros',
            'account_type_regexes': ['.*'],
            'start_name_regexes': ['.*'],
            'trim_name_regexes': [],
            'show_name_regexes': ['.*'],
            'hide_name_regexes': [],
            'account_displayed_op_only': True,
            'select': ['micros', 'bytes'],
            'step': -1,
            'output': 'stdout'}

  def build(self):
    # [FUNCTION DOC]
    # Returns a deep copy of the option mapping built so far, so later builder changes do not
    # affect the returned value. No inputs, no side effects.
    """Build a profiling option.

    Returns:
      A dict of profiling options.
    """
    return copy.deepcopy(self._options)

  def with_max_depth(self, max_depth):
    # [FUNCTION DOC]
    # Sets the maximum display depth option. Input: max_depth, integer. Returns the builder
    # itself for chaining. Mutates the held options.
    """Set the maximum depth of display.

    The depth depends on profiling view. For 'scope' view, it's the
    depth of name scope hierarchy (tree), for 'op' view, it's the number
    of operation types (list), etc.

    Args:
      max_depth: Maximum depth of the data structure to display.
    Returns:
      self
    """
    self._options['max_depth'] = max_depth
    return self

  def with_min_memory(self,
                      min_bytes=0,
                      min_peak_bytes=0,
                      min_residual_bytes=0,
                      min_output_bytes=0):
    # [FUNCTION DOC]
    # Sets the four memory thresholds: requested bytes, peak bytes, residual bytes and output
    # bytes (each a number, default 0). Always overwrites all four options, even when left at
    # their defaults. Returns the builder itself for chaining. Mutates the held options.
    """Only show profiler nodes consuming no less than 'min_bytes'.

    Args:
      min_bytes: Only show profiler nodes requested to allocate no less bytes
          than this.
      min_peak_bytes: Only show profiler nodes using no less than this bytes
          at peak (high watermark). For profiler nodes consist of multiple
          graph nodes, it sums the graph nodes' peak_bytes.
      min_residual_bytes: Only show profiler nodes have no less than
          this bytes not being de-allocated after Compute() ends. For
          profiler nodes consist of multiple graph nodes, it sums the
          graph nodes' residual_bytes.
      min_output_bytes: Only show profiler nodes have no less than this bytes
          output. The output are not necessarily allocated by this profiler
          nodes.
    Returns:
      self
    """
    self._options['min_bytes'] = min_bytes
    self._options['min_peak_bytes'] = min_peak_bytes
    self._options['min_residual_bytes'] = min_residual_bytes
    self._options['min_output_bytes'] = min_output_bytes
    return self

  def with_min_execution_time(self,
                              min_micros=0,
                              min_accelerator_micros=0,
                              min_cpu_micros=0):
    # [FUNCTION DOC]
    # Sets the execution time thresholds: total micros, accelerator micros and cpu micros (each a
    # number, default 0). Always overwrites all three options. Returns the builder itself for
    # chaining. Mutates the held options.
    """Only show profiler nodes consuming no less than 'min_micros'.

    Args:
      min_micros: Only show profiler nodes with execution time
          no less than this. It sums accelerator and cpu times.
      min_accelerator_micros: Only show profiler nodes spend no less than
          this time on accelerator (e.g. GPU).
      min_cpu_micros: Only show profiler nodes spend no less than
          this time on cpu.
    Returns:
      self
    """
    self._options['min_micros'] = min_micros
    self._options['min_accelerator_micros'] = min_accelerator_micros
    self._options['min_cpu_micros'] = min_cpu_micros
    return self

  def with_min_parameters(self, min_params):
    # [FUNCTION DOC]
    # Sets the minimum parameter count threshold. Input: min_params, number. Returns the builder
    # itself for chaining. Mutates the held options.
    """Only show profiler nodes holding no less than 'min_params' parameters.

    'Parameters' normally refers the weights of in TensorFlow variables.
    It reflects the 'capacity' of models.

    Args:
      min_params: Only show profiler nodes holding number parameters
          no less than this.
    Returns:
      self
    """
    self._options['min_params'] = min_params
    return self

  def with_min_occurrence(self, min_occurrence):
    # [FUNCTION DOC]
    # Sets the minimum number of graph nodes a profiler node must include to be shown. Input:
    # min_occurrence, integer. Returns the builder itself for chaining. Mutates held options.
    # pylint: disable=line-too-long
    """Only show profiler nodes including no less than 'min_occurrence' graph nodes.

    A "node" means a profiler output node, which can be a python line
    (code view), an operation type (op view), or a graph node
    (graph/scope view). A python line includes all graph nodes created by that
    line, while an operation type includes all graph nodes of that type.

    Args:
      min_occurrence: Only show nodes including no less than this.
    Returns:
      self
    """
    # pylint: enable=line-too-long
    self._options['min_occurrence'] = min_occurrence
    return self

  def with_min_float_operations(self, min_float_ops):
    # [FUNCTION DOC]
    # Sets the minimum float operation count threshold. Input: min_float_ops, number. Returns the
    # builder itself for chaining. Mutates the held options.
    # pylint: disable=line-too-long
    """Only show profiler nodes consuming no less than 'min_float_ops'.

    Please see https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md
    on the caveats of calculating float operations.

    Args:
      min_float_ops: Only show profiler nodes with float operations
          no less than this.
    Returns:
      self
    """
    # pylint: enable=line-too-long
    self._options['min_float_ops'] = min_float_ops
    return self

  def with_accounted_types(self, account_type_regexes):
    # [FUNCTION DOC]
    # Sets which node types are counted in the statistics. Input: account_type_regexes, list of
    # regular expression strings matched against node properties such as device and operation
    # type; the list is shallow-copied. Returns the builder itself. Mutates the held options.
    """Selectively counting statistics based on node types.

    Here, 'types' means the profiler nodes' properties. Profiler by default
    consider device name (e.g. /job:xx/.../device:GPU:0) and operation type
    (e.g. MatMul) as profiler nodes' properties. User can also associate
    customized 'types' to profiler nodes through OpLogProto proto.

    For example, user can select profiler nodes placed on gpu:0 with:
    `account_type_regexes=['.*gpu:0.*']`

    If none of a node's properties match the specified regexes, the node is
    not displayed nor accounted.

    Args:
      account_type_regexes: A list of regexes specifying the types.
    Returns:
      self.
    """
    self._options['account_type_regexes'] = copy.copy(account_type_regexes)
    return self

  def with_node_names(self,
                      start_name_regexes=None,
                      show_name_regexes=None,
                      hide_name_regexes=None,
                      trim_name_regexes=None):
    # [FUNCTION DOC]
    # Sets the node-name filters. Inputs (each an optional list of regular expression strings,
    # default null meaning leave unchanged): start_name_regexes where display begins,
    # show_name_regexes nodes to display, hide_name_regexes nodes to hide, trim_name_regexes
    # nodes where searching a branch stops. Provided lists are shallow-copied. Returns the builder
    # itself. Mutates the held options.
    """Regular expressions used to select profiler nodes to display.

    After 'with_accounted_types' is evaluated, 'with_node_names' are
    evaluated as follows:

      For a profile data structure, profiler first finds the profiler
      nodes matching 'start_name_regexes', and starts displaying profiler
      nodes from there. Then, if a node matches 'show_name_regexes' and
      doesn't match 'hide_name_regexes', it's displayed. If a node matches
      'trim_name_regexes', profiler stops further searching that branch.

    Args:
      start_name_regexes: list of node name regexes to start displaying.
      show_name_regexes: list of node names regexes to display.
      hide_name_regexes: list of node_names regexes that should be hidden.
      trim_name_regexes: list of node name regexes from where to stop.
    Returns:
      self
    """
    if start_name_regexes is not None:
      self._options['start_name_regexes'] = copy.copy(start_name_regexes)
    if show_name_regexes is not None:
      self._options['show_name_regexes'] = copy.copy(show_name_regexes)
    if hide_name_regexes is not None:
      self._options['hide_name_regexes'] = copy.copy(hide_name_regexes)
    if trim_name_regexes is not None:
      self._options['trim_name_regexes'] = copy.copy(trim_name_regexes)
    return self

  def account_displayed_op_only(self, is_true):
    # [FUNCTION DOC]
    # Sets whether only statistics of displayed nodes are accounted. Input: is_true, boolean.
    # Returns the builder itself. Mutates the held options.
    """Whether only account the statistics of displayed profiler nodes.

    Args:
      is_true: If true, only account statistics of nodes eventually
          displayed by the outputs.
          Otherwise, a node's statistics are accounted by its parents
          as long as it's types match 'account_type_regexes', even if
          it is hidden from the output, say, by hide_name_regexes.
    Returns:
      self
    """
    self._options['account_displayed_op_only'] = is_true
    return self

  def with_empty_output(self):
    # [FUNCTION DOC]
    # Sets the output destination to none so profiling produces no side-effect output. Returns
    # the builder itself. Mutates the held options.
    """Do not generate side-effect outputs."""
    self._options['output'] = 'none'
    return self

  def with_stdout_output(self):
    # [FUNCTION DOC]
    # Sets the output destination to standard output. Returns the builder itself. Mutates the
    # held options.
    """Print the result to stdout."""
    self._options['output'] = 'stdout'
    return self

  def with_file_output(self, outfile):
    # [FUNCTION DOC]
    # Sets the output destination to a text file. Input: outfile, string path (not validated).
    # Returns the builder itself. Mutates the held options; the file is written later when
    # profiling runs.
    """Print the result to a file."""
    self._options['output'] = 'file:outfile=%s' % outfile
    return self

  def with_timeline_output(self, timeline_file):
    # [FUNCTION DOC]
    # Sets the output destination to a timeline JSON file. Input: timeline_file, string path
    # prefix (the profiler appends the step number). Returns the builder itself. Mutates options.
    """Generate a timeline json file."""
    self._options['output'] = 'timeline:outfile=%s' % timeline_file
    return self

  def with_pprof_output(self, pprof_file):
    # [FUNCTION DOC]
    # Sets the output destination to a gzip-compressed pprof profile. Input: pprof_file, string
    # path. Returns the builder itself. Mutates the held options; the file is written later.
    """Generate a pprof profile gzip file.

    To use the pprof file:
      pprof -png --nodecount=100 --sample_index=1 <pprof_file>

    Args:
      pprof_file: filename for output, usually suffixed with .pb.gz.
    Returns:
      self.
    """
    self._options['output'] = 'pprof:outfile=%s' % pprof_file
    return self

  def order_by(self, attribute):
    # [FUNCTION DOC]
    # Sets the attribute used to order displayed nodes. Input: attribute, string such as micros,
    # bytes, occurrence or params. Returns the builder itself. Mutates the held options.
    # pylint: disable=line-too-long
    """Order the displayed profiler nodes based on a attribute.

    Supported attribute includes micros, bytes, occurrence, params, etc.
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/options.md

    Args:
      attribute: An attribute the profiler node has.
    Returns:
      self
    """
    # pylint: enable=line-too-long
    self._options['order_by'] = attribute
    return self

  def select(self, attributes):
    # [FUNCTION DOC]
    # Sets which attributes are displayed. Input: attributes, list of attribute name strings
    # (shallow-copied). Returns the builder itself. Mutates the held options.
    # pylint: disable=line-too-long
    """Select the attributes to display.

    See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/options.md
    for supported attributes.

    Args:
      attributes: A list of attribute the profiler node has.
    Returns:
      self
    """
    # pylint: enable=line-too-long
    self._options['select'] = copy.copy(attributes)
    return self

  def with_step(self, step):
    # [FUNCTION DOC]
    # Chooses which profile step to use. Input: step, integer step id as given when steps were
    # added; negative one means the average of all steps. Returns the builder itself. Mutates
    # the held options.
    """Which profile step to use for profiling.

    The 'step' here refers to the step defined by `Profiler.add_step()` API.

    Args:
      step: When multiple steps of profiles are available, select which step's
         profile to use. If -1, use average of all available steps.
    Returns:
      self
    """
    self._options['step'] = step
    return self
