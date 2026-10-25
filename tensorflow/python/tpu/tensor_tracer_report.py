# [MODULE DOC]
# Report generation for Tensor Tracer. Provides a graph topological sort, a wrapper that gives
# consistent operation and tensor orderings, a configuration holder, a tensor-to-cache-index
# mapping, a context manager for the report file, a proto fingerprint helper and TTReportHandle,
# which writes the human readable text report (sections for config, ops, tensors, trace points,
# cache index map, reasons, graph) and builds/writes the report protocol buffer. Defines only
# constants at import time; no registrations.
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
# ========================================================================
"""Tensor Tracer report generation utilities."""

import collections
import hashlib
import os

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2

_TRACER_LOG_PREFIX = ' [>>>TT>>>]'
_MARKER_SECTION_BEGIN = '!!!!!!! section-begin:'
_MARKER_SECTION_END = '!!!!!!! section-end:'

_SECTION_NAME_CONFIG = 'configuration'
_SECTION_NAME_REASON = 'reason'
_SECTION_NAME_OP_LIST = 'op-list'
_SECTION_NAME_TENSOR_LIST = 'tensor-list'
_SECTION_NAME_CACHE_INDEX_MAP = 'cache-index-map'
_SECTION_NAME_GRAPH = 'graph'
_SECTION_NAME_TENSOR_TRACER_CHECKPOINT = 'tensor_tracer_checkpoint'

_FIELD_NAME_VERSION = 'version:'
_FIELD_NAME_DEVICE = 'device:'
_FIELD_NAME_TRACE_MODE = 'trace-mode:'
_FIELD_NAME_SUBMODE = 'submode:'
_FIELD_NAME_NUM_REPLICAS = 'num-replicas:'
_FIELD_NAME_NUM_REPLICAS_PER_HOST = 'num-replicas-per-host:'
_FIELD_NAME_NUM_HOSTS = 'num-hosts:'
_FIELD_NAME_NUM_OPS = 'number-of-ops:'
_FIELD_NAME_NUM_TENSORS = 'number-of-tensors:'
_FIELD_NAME_NUM_CACHE_INDICES = 'number-of-indices:'
_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED = 'topological-sort-succeed:'

_CURRENT_VERSION = 'use-outside-compilation'

_TT_REPORT_PROTO = 'tensor_tracer_report.report_pb'


def topological_sort(g):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the first element is true when there is no cycle, but the
  # code returns true when a cycle is found (leftover ops with nonzero in-degree) and false when
  # the sort succeeded, i.e. the flag actually means "contains a cycle".
  # Input g: a graph exposing its list of operations. Computes in-degrees (ignoring edges from
  # loop-back next-iteration ops, which act as sinks), then repeatedly removes zero-degree ops
  # in name order, decrementing consumers. Returns a pair: (true, set of ops left over) when a
  # cycle prevents completion, else (false, list of ops in topological order).
  # Raises a value error if a consumer's in-degree goes negative.
  """Performs topological sort on the given graph.

  Args:
     g: the graph.

  Returns:
     A pair where the first element indicates if the topological
     sort succeeded (True if there is no cycle found; False if a
     cycle is found) and the second element is either the sorted
     list of nodes or the cycle of nodes found.
  """
  def _is_loop_edge(op):
    # [FUNCTION DOC]
    # HELPER: edge classifier for the topological sort.
    # Returns true if the operation's type is next-iteration, i.e. it closes a loop cycle.
    """Returns true if the op is the end of a while-loop creating a cycle."""
    return op.type in ['NextIteration']

  def _in_op_degree(op):
    # [FUNCTION DOC]
    # HELPER: in-degree computation for the topological sort.
    # Counts the control inputs and data-input producers of the operation, skipping those that
    # are next-iteration ops. Returns the count as an integer.
    # KNOWN ISSUE: the loop variable reuses the name of the parameter op, shadowing it; harmless
    # since the sequence is built before the loop starts.
    """Returns the number of incoming edges to the given op.

    The edge calculation skips the edges that come from 'NextIteration' ops.
    NextIteration creates a cycle in the graph. We break cycles by treating
    this op as 'sink' and ignoring all outgoing edges from it.
    Args:
      op: Tf.Operation
    Returns:
      the number of incoming edges.
    """
    count = 0
    for op in op.control_inputs + [in_tensor.op for in_tensor in op.inputs]:
      if not _is_loop_edge(op):
        count += 1
    return count

  sorted_ops = []
  op_in_degree = {op: _in_op_degree(op) for op in g.get_operations()}

  frontier = [op for (op, degree) in op_in_degree.items() if degree == 0]
  frontier.sort(key=lambda op: op.name)
  while frontier:
    op = frontier.pop()
    # Remove the op from graph, and remove its outgoing edges.
    sorted_ops.append(op)
    if _is_loop_edge(op):
      continue
    # pylint: disable=protected-access
    consumers = list(op._control_outputs)
    # pylint: enable=protected-access
    for out_tensor in op.outputs:
      consumers += [consumer_op for consumer_op in out_tensor.consumers()]
    consumers.sort(key=lambda op: op.name)
    for consumer in consumers:
      # For each deleted edge shift the bucket of the vertex.
      op_in_degree[consumer] -= 1
      if op_in_degree[consumer] == 0:
        frontier.append(consumer)
      if op_in_degree[consumer] < 0:
        raise ValueError('consumer:%s degree mismatch'%consumer.name)

  left_ops = set(op for (op, degree) in op_in_degree.items() if degree > 0)
  if left_ops:
    return (True, left_ops)
  else:
    assert len(g.get_operations()) == len(sorted_ops)
    return (False, sorted_ops)


class TensorTracerConfig(object):
  # [CLASS DOC]
  # Plain holder for the run environment recorded in the report: tracer version, device type,
  # number of replicas, replicas per host and number of hosts. Fields other than version start
  # as null and are filled in by the tracer.
  """Tensor Tracer config object."""

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the version to the current version string and the device type, replica count,
    # replicas per host and host count to null. Takes no inputs, returns nothing.
    self.version = _CURRENT_VERSION
    self.device_type = None
    self.num_replicas = None
    self.num_replicas_per_host = None
    self.num_hosts = None


class TensorTraceOrder(object):
  # [CLASS DOC]
  # Assigns each traced tensor a consecutive cache index. Holds the graph ordering wrapper, the
  # list of traced tensors, a mapping from tensor name to cache index and a list mapping cache
  # index to the tensor's index in the graph ordering.
  """Class that is responsible from storing the trace-id of the tensors."""

  def __init__(self, graph_order, traced_tensors):
    # [FUNCTION DOC]
    # Inputs: graph_order, the wrapper from the sorting function, and traced_tensors, the list of
    # tensors to trace. Stores both and builds the index maps. Raises errors as described for
    # the map-building helper. Returns nothing.
    self.graph_order = graph_order
    self.traced_tensors = traced_tensors
    self._create_tensor_maps()

  def _create_tensor_maps(self):
    # [FUNCTION DOC]
    # HELPER: constructor helper building the cache index maps.
    # Walks the traced tensors in order, giving each the next cache index and recording the
    # tensor's graph-order index. Sets the name-to-cache-index mapping and the cache-index-to-
    # tensor-index list on the instance. Raises a value error for a duplicate tensor name or a
    # name missing from the graph ordering, and a runtime error if the two maps disagree in size.
    """Creates tensor to cache id maps."""
    self.tensorname_to_cache_idx = {}
    self.cache_idx_to_tensor_idx = []
    for out_tensor in self.traced_tensors:
      tensor_name = out_tensor.name
      if tensor_name in self.tensorname_to_cache_idx:
        raise ValueError('Tensor name {} should not be already in '
                         'tensorname_to_cache_idx'.format(tensor_name))
      if tensor_name not in self.graph_order.tensor_to_idx:
        raise ValueError(
            'Tensor name {} is not in the tensor_to_idx, tensor_to_idx={} '
            .format(tensor_name, self.graph_order.tensor_to_idx))
      tensor_idx = self.graph_order.tensor_to_idx[tensor_name]
      cache_idx = len(self.tensorname_to_cache_idx)
      self.tensorname_to_cache_idx[tensor_name] = cache_idx
      self.cache_idx_to_tensor_idx.append(tensor_idx)
      if len(self.tensorname_to_cache_idx) != len(
          self.cache_idx_to_tensor_idx):
        raise RuntimeError(
            'len(self.tensorname_to_cache_idx) must equal'
            'len(self.cache_idx_to_tensor_idx), got '
            'len(self.tensorname_to_cache_idx)={}, '
            'len(self.cache_idx_to_tensor_idx)={}'
            .format(
                len(self.tensorname_to_cache_idx),
                len(self.cache_idx_to_tensor_idx)))


def sort_tensors_and_ops(graph):
  # [FUNCTION DOC]
  # Input graph: a graph with operations. Runs the topological sort; uses its order if it
  # succeeded, otherwise the graph's native operation order. Builds name-to-index maps for ops
  # and for output tensors. Returns a named tuple with fields: graph, operations, op_to_idx,
  # tensors, tensor_to_idx, contains_cycle and topological_order_or_cycle.
  # KNOWN ISSUE: the code names the first result of the sort "contains_cycle" and negates it
  # correctly only because the sort returns true for a cycle (contrary to its docstring).
  """Returns a wrapper that has consistent tensor and op orders."""
  graph_wrapper = collections.namedtuple('GraphWrapper',
                                         ['graph', 'operations', 'op_to_idx',
                                          'tensors', 'tensor_to_idx',
                                          'contains_cycle',
                                          'topological_order_or_cycle'])
  contains_cycle, topological_order_or_cycle = topological_sort(graph)
  if not contains_cycle:
    operations = topological_order_or_cycle
  else:
    operations = graph.get_operations()
  op_to_idx = {op.name: index for index, op
               in enumerate(operations)}
  tensors = []
  for op in operations:
    tensors.extend(op.outputs)
  tensor_to_idx = {tensor.name: index for index, tensor in
                   enumerate(tensors)}
  return graph_wrapper(graph=graph, operations=operations, op_to_idx=op_to_idx,
                       tensors=tensors, tensor_to_idx=tensor_to_idx,
                       contains_cycle=contains_cycle,
                       topological_order_or_cycle=topological_order_or_cycle)


class OpenReportFile(object):
  # [CLASS DOC]
  # Context manager that opens the report file for writing when a report path is configured and
  # yields the open file (or null when no path is set). Closes the file on exit.
  """Context manager for writing report file."""

  def __init__(self, tt_parameters):
    # [FUNCTION DOC]
    # Input tt_parameters: parameters object with a report file path. If the path is empty the
    # stored file is null; otherwise opens the file for writing (side effect: creates or
    # truncates it). Re-raises any I/O error from opening. Returns nothing.
    # KNOWN ISSUE: the try/except only re-raises the same error, so it has no effect.
    if not tt_parameters.report_file_path:
      self._report_file = None
      return
    try:
      self._report_file = gfile.Open(tt_parameters.report_file_path, 'w')
    except IOError as e:
      raise e

  def __enter__(self):
    # [FUNCTION DOC]
    # Returns the open report file, or null when no report path was configured.
    return self._report_file

  def __exit__(self, unused_type, unused_value, unused_traceback):
    # [FUNCTION DOC]
    # Closes the report file if one is open. Exception details are ignored and never suppressed.
    if self._report_file:
      self._report_file.close()


def proto_fingerprint(message_proto):
  # [FUNCTION DOC]
  # Input: a protocol buffer message. Serializes it and returns the hex digest string of its
  # SHA-256 hash. No side effects.
  serialized_message = message_proto.SerializeToString()
  hasher = hashlib.sha256(serialized_message)
  return hasher.hexdigest()


class TTReportHandle(object):
  # [CLASS DOC]
  # Collects explanations ("instrument records") for why ops or tensors were or were not traced,
  # and produces the tracer report: a text report written to the report file (or to the log when
  # no file is open) and a report protocol buffer. State: a mapping from names to explanation
  # strings and the currently open report file.
  """Utility class responsible from creating a tensor tracer report."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates an empty mapping of instrument records and no open report file. Returns nothing.
    self.instrument_records = {}
    self._report_file = None

  def instrument(self, name, explanation):
    # [FUNCTION DOC]
    # Records the explanation string for the given name, replacing any earlier explanation.
    # Mutates the instrument records. Returns nothing.
    self.instrument_records[name] = explanation

  def instrument_op(self, op, explanation):
    # [FUNCTION DOC]
    # Records the explanation under the operation's name. Returns nothing.
    self.instrument(op.name, explanation)

  def instrument_tensor(self, tensor, explanation):
    # [FUNCTION DOC]
    # Records the explanation under the tensor's name. Returns nothing.
    self.instrument(tensor.name, explanation)

  def create_report_proto(self, tt_config, tt_parameters, tensor_trace_order,
                          tensor_trace_points, collected_signature_types):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the text report writer for trace points iterates pairs of tensor and name,
    # while this method treats tensor_trace_points as a mapping from tensor name to trace point
    # name; the two callers' expectations of that argument differ.
    # Builds the report protocol buffer. Fills the config (version, device, cores, hosts, cores
    # per host, submode, trace mode), the signature names ordered by their index, and one
    # definition per graph tensor (name, traced flag, cache index, trace point name, explanation
    # looked up by tensor name then op name). Untraced tensors are skipped when the fingerprint
    # subdirectory option is on. Then sets the fingerprint (logged) and copies the graph
    # definition in. Returns the report proto.
    """Creates and returns a proto that stores tensor tracer configuration.

    Args:
      tt_config: TensorTracerConfig object holding information about the run
        environment (device, # cores, # hosts), and tensor tracer version
        information.
      tt_parameters: TTParameters objects storing the user provided parameters
        for tensor tracer.
      tensor_trace_order: TensorTraceOrder object storing a topological order of
        the graph.
      tensor_trace_points: Progromatically added trace_points/checkpoints.
      collected_signature_types: The signature types collected, e,g, norm,
        max, min, mean...
    Returns:
      TensorTracerReport proto.
    """
    report = tensor_tracer_pb2.TensorTracerReport()
    report.config.version = tt_config.version
    report.config.device = tt_config.device_type
    report.config.num_cores = tt_config.num_replicas
    report.config.num_hosts = tt_config.num_hosts
    report.config.num_cores_per_host = tt_config.num_replicas_per_host
    report.config.submode = tt_parameters.submode
    report.config.trace_mode = tt_parameters.trace_mode

    for signature_name, _ in sorted(collected_signature_types.items(),
                                    key=lambda x: x[1]):
      report.config.signatures.append(signature_name)

    for tensor in tensor_trace_order.graph_order.tensors:
      tensor_def = tensor_tracer_pb2.TensorTracerReport.TracedTensorDef()
      tensor_def.name = tensor.name
      if tensor.name in tensor_trace_order.tensorname_to_cache_idx:
        tensor_def.is_traced = True
        tensor_def.cache_index = (
            tensor_trace_order.tensorname_to_cache_idx[tensor.name])
      else:
        # To prevent small changes affecting the fingerprint calculation, avoid
        # writing the untraced tensors to metadata. Fingerprints will be
        # different only when the list of the traced tensors are different.
        if tt_parameters.use_fingerprint_subdir:
          continue
        tensor_def.is_traced = False

      if tensor.name in tensor_trace_points:
        tensor_def.trace_point_name = tensor_trace_points[tensor.name]
      if tensor.name in self.instrument_records:
        tensor_def.explanation = self.instrument_records[tensor.name]
      elif tensor.op.name in self.instrument_records:
        tensor_def.explanation = self.instrument_records[tensor.op.name]
      report.tensordef[tensor.name].CopyFrom(tensor_def)
    report.fingerprint = proto_fingerprint(report)
    logging.info('TensorTracerProto fingerprint is %s.',
                 report.fingerprint)
    tf_graph = tensor_trace_order.graph_order.graph
    report.graphdef.CopyFrom(tf_graph.as_graph_def())
    return report

  def report_proto_path(self, trace_dir, summary_tag_name):
    # [FUNCTION DOC]
    # Inputs: trace directory string and summary tag name. Returns the path inside the trace
    # directory of the report proto file, named with the fixed report prefix plus the tag name
    # with slashes replaced by underscores. No file access.
    """Returns the path where report proto should be written.

    Args:
      trace_dir: String denoting the trace directory.
      summary_tag_name: Name of the unique tag that relates to
                        the report.
    Returns:
      A string denoting the path to the report proto.
    """
    filename = _TT_REPORT_PROTO + '.' + summary_tag_name.replace('/', '_')
    return os.path.join(trace_dir, filename)

  def write_report_proto(self, report_path, report_proto, tt_parameters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it writes under the trace directory, but the file is written
    # at report_path as given; only the trace directory is created.
    # Creates the trace directory from the parameters, then writes the serialized proto bytes to
    # report_path (binary, overwriting). Side effects: directory and file creation. Returns
    # nothing; file system errors propagate.
    """Writes the given report proto under trace_dir."""
    gfile.MakeDirs(tt_parameters.trace_dir)
    with gfile.GFile(report_path, 'wb') as f:
      f.write(report_proto.SerializeToString())

  def create_report(self, tt_config, tt_parameters,
                    tensor_trace_order, tensor_trace_points):
    # [FUNCTION DOC]
    # Opens the report file per the parameters (or uses logging if none) and writes, in order,
    # the config, op list, tensor list, trace points, cache index map, reason and graph
    # sections. Inputs: tracer config, parameters, tensor trace order and trace points.
    # Side effects: writes the file or logs; sets the handle's current report file. Returns
    # nothing.
    """Creates a report file and writes the trace information."""
    with OpenReportFile(tt_parameters) as self._report_file:
      self._write_config_section(tt_config, tt_parameters)
      self._write_op_list_section(tensor_trace_order.graph_order)
      self._write_tensor_list_section(tensor_trace_order.graph_order)
      self._write_trace_points(tensor_trace_points)
      self._write_cache_index_map_section(tensor_trace_order)
      self._write_reason_section()
      self._write_graph_section(tensor_trace_order.graph_order)

  def _write_trace_points(self, tensor_trace_points):
    # [FUNCTION DOC]
    # HELPER: report section writer for the create-report method.
    # Writes the checkpoint section: begin marker, one line per (tensor, checkpoint name) pair
    # giving tensor name and checkpoint name, end marker. Expects an iterable of pairs.
    """Writes the list of checkpoints."""
    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _SECTION_NAME_TENSOR_TRACER_CHECKPOINT))
    for (tensor, checkpoint_name) in tensor_trace_points:
      self._write_report('%s %s\n'%(tensor.name, checkpoint_name))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _SECTION_NAME_TENSOR_TRACER_CHECKPOINT))

  def _write_report(self, content):
    # [FUNCTION DOC]
    # HELPER: low-level output for all report section writers.
    # Prefixes content with the tracer log prefix and writes it to the open report file, or logs
    # it at info level when no file is open. Returns nothing.
    """Writes the given content to the report."""

    line = '%s %s'%(_TRACER_LOG_PREFIX, content)
    if self._report_file:
      self._report_file.write(line)
    else:
      logging.info(line)

  def _write_config_section(self, tt_config, tt_parameters):
    # [FUNCTION DOC]
    # HELPER: report section writer.
    # Writes the configuration section: version, device, trace mode, submode, replica count,
    # replicas per host and host count between begin and end markers.
    """Writes the config section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_CONFIG))
    self._write_report('%s %s\n'%(_FIELD_NAME_VERSION, tt_config.version))
    self._write_report('%s %s\n'%(_FIELD_NAME_DEVICE, tt_config.device_type))
    self._write_report('%s %s\n'%(_FIELD_NAME_TRACE_MODE,
                                  tt_parameters.trace_mode))
    self._write_report('%s %s\n'%(_FIELD_NAME_SUBMODE,
                                  tt_parameters.submode))
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_REPLICAS,
                                  tt_config.num_replicas))
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_REPLICAS_PER_HOST,
                                  tt_config.num_replicas_per_host))
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_HOSTS, tt_config.num_hosts))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_CONFIG))

  def _write_reason_section(self):
    # [FUNCTION DOC]
    # HELPER: report section writer.
    # Writes the reason section: one line per instrument record, sorted by name, giving the quoted
    # name and its explanation.
    """Writes the reason section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_REASON))
    for key in sorted(self.instrument_records):
      self._write_report('"%s" %s\n'%(key, self.instrument_records[key]))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_REASON))

  def _write_op_list_section(self, graph_order):
    # [FUNCTION DOC]
    # HELPER: report section writer.
    # Writes the op count, then per operation its index, quoted name, type and the indices of its
    # output tensors. Raises a value error if an output tensor is missing from the tensor index
    # mapping.
    """Writes the Op-list section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_OP_LIST))
    self._write_report('%s %d\n'%(_FIELD_NAME_NUM_OPS,
                                  len(graph_order.operations)))
    for i in range(0, len(graph_order.operations)):
      op = graph_order.operations[i]
      line = '%d "%s" %s'%(i, op.name, op.type)
      for out_tensor in op.outputs:
        if out_tensor.name not in graph_order.tensor_to_idx:
          raise ValueError(
              'out_tensor is not in tensor_to_idx. out_tensor={}, '
              'tensor_to_idx={}'
              .format(out_tensor.name, graph_order.tensor_to_idx))
        line += ' %d'%graph_order.tensor_to_idx[out_tensor.name]
      line += '\n'
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_OP_LIST))

  def _write_tensor_list_section(self, graph_order):
    # [FUNCTION DOC]
    # HELPER: report section writer.
    # Writes the tensor count, then per tensor its index, quoted name and the indices of its
    # consumer ops (sorted by op name). Raises a value error if a consumer op is missing from
    # the op index mapping.
    """Writes the tensor-list section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _SECTION_NAME_TENSOR_LIST))
    self._write_report('%s %d\n'%(_FIELD_NAME_NUM_TENSORS,
                                  len(graph_order.tensors)))
    for i in range(0, len(graph_order.tensors)):
      tensor = graph_order.tensors[i]
      line = '%d "%s"'%(i, tensor.name)
      consumers = tensor.consumers()
      consumers.sort(key=lambda op: op.name)
      for consumer_op in consumers:
        if consumer_op.name not in graph_order.op_to_idx:
          raise ValueError(
              'consumer_op is not in op_to_idx.  '
              'got consumer_op={}, op_to_idx={}'
              .format(consumer_op.name, graph_order.op_to_idx))
        line += ' %d'%graph_order.op_to_idx[consumer_op.name]
      line += '\n'
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _SECTION_NAME_TENSOR_LIST))

  def _write_cache_index_map_section(self, tensor_trace_order):
    # [FUNCTION DOC]
    # HELPER: report section writer.
    # Writes the number of cache indices, then one line per cache index giving the cache index
    # and the corresponding tensor index.
    """Writes the mapping from cache index to tensor index to the report."""
    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _SECTION_NAME_CACHE_INDEX_MAP))
    self._write_report('%s %d\n'%(
        _FIELD_NAME_NUM_CACHE_INDICES,
        len(tensor_trace_order.cache_idx_to_tensor_idx)))
    for cache_idx in range(0, len(tensor_trace_order.cache_idx_to_tensor_idx)):
      tensor_idx = tensor_trace_order.cache_idx_to_tensor_idx[cache_idx]
      line = '%d %d\n'%(cache_idx, tensor_idx)
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _SECTION_NAME_CACHE_INDEX_MAP))

  def _write_graph_section(self, graph_order):
    # [FUNCTION DOC]
    # HELPER: report section writer.
    # Writes whether the topological sort succeeded (the negation of the cycle flag), then each
    # op of the topological order (or the cycle ops) with its position and quoted name.
    """Writes the graph section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_GRAPH))
    self._write_report('%s %s\n'%(_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED,
                                  not graph_order.contains_cycle))
    l = list(graph_order.topological_order_or_cycle)
    for i in range(0, len(l)):
      self._write_report('%d "%s"\n'%(i, l[i].name))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_GRAPH))
