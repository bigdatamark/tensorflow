# [MODULE DOC]
# HELPER: Library behind the print_selective_registration_header executable wrapper.
# Computes the C++ header (ops_to_register.h) used with the SELECTIVE_REGISTRATION build option.
# It collects the set of (op name, kernel class) pairs needed by one or more model files (raw or
# text GraphDef protos, or JSON ops-list files), adds default ops, and renders header text that
# defines macros deciding which ops, kernels and the symbolic gradient get registered.
# Exposes get_ops_from_nodedef, get_ops_and_kernels, get_header_from_ops_and_kernels and
# get_header. Module-level constants: an allowlist of ops that need no kernel, and the Flex
# prefix constants. Kernel lookup uses the native kernel registry extension. No other import-time
# side effects.
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
r"""Computes a header file to be used with SELECTIVE_REGISTRATION.

See the executable wrapper, print_selective_registration_header.py, for more
information.
"""

import json
import os
import sys

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_kernel_registry

# Usually, we use each graph node to induce registration of an op and
# corresponding kernel; nodes without a corresponding kernel (perhaps due to
# attr types) generate a warning but are otherwise ignored. Ops in this set are
# registered even if there's no corresponding kernel.
OPS_WITHOUT_KERNEL_ALLOWLIST = frozenset([
    # AccumulateNV2 is rewritten away by AccumulateNV2RemovePass; see
    # core/common_runtime/accumulate_n_optimizer.cc.
    'AccumulateNV2'
])
FLEX_PREFIX = b'Flex'
FLEX_PREFIX_LENGTH = len(FLEX_PREFIX)


def _get_ops_from_ops_list(input_file):
  # [FUNCTION DOC]
  # HELPER: Input reader for get_ops_and_kernels when the format is an ops list.
  # Reads a JSON file of (op, kernel) pairs at input_file and returns a set of tuples, with an
  # empty kernel name turned into a null value. Raises a generic Exception if the file content is
  # empty; invalid JSON raises the JSON decoding error. The file handle is never explicitly closed.
  """Gets the ops and kernels needed from the ops list file."""
  ops = set()
  ops_list_str = gfile.GFile(input_file, 'r').read()
  if not ops_list_str:
    raise Exception('Input file should not be empty')
  ops_list = json.loads(ops_list_str)
  for op, kernel in ops_list:
    op_and_kernel = (op, kernel if kernel else None)
    ops.add(op_and_kernel)
  return ops


def _get_ops_from_graphdef(graph_def):
  # [FUNCTION DOC]
  # HELPER: Graph traversal for get_ops_and_kernels.
  # Takes a GraphDef proto and returns the set of (op, kernel) pairs for all top-level nodes and
  # for the nodes of every function in its function library. Nodes with no kernel are skipped.
  """Gets the ops and kernels needed from the tensorflow model."""
  ops = set()
  ops.update(_get_ops_from_nodedefs(graph_def.node))

  for function in graph_def.library.function:
    ops.update(_get_ops_from_nodedefs(function.node_def))
  return ops


def get_ops_from_nodedef(node_def):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns None only when no kernel is found and the op is
  # not allowlisted; the kernel half of the returned pair is also None for allowlisted ops.
  # Resolves the op and kernel class for one NodeDef. Side effect: if the node has no device set,
  # it is mutated to the first CPU device before lookup. Asks the native kernel registry for the
  # kernel class of the serialized node. Returns a pair of op name and kernel class name (null
  # kernel for allowlisted ops without kernel), or nothing and logs a warning if no kernel exists
  # and the op is not allowlisted.
  """Gets the op and kernel needed from the given NodeDef.

  Args:
    node_def: TF NodeDef to get op/kernel information.

  Returns:
    A tuple of (op_name, kernel_name). If the op is not in the allowlist of ops
    without kernel and there is no kernel found, then return None.
  """
  if not node_def.device:
    node_def.device = '/cpu:0'
  kernel_class = _pywrap_kernel_registry.TryFindKernelClass(
      node_def.SerializeToString())
  op = str(node_def.op)
  if kernel_class or op in OPS_WITHOUT_KERNEL_ALLOWLIST:
    return (op, str(kernel_class.decode('utf-8')) if kernel_class else None)
  else:
    tf_logging.warning('Warning: no kernel found for op %s', op)
    return None


def _get_ops_from_nodedefs(node_defs):
  # [FUNCTION DOC]
  # HELPER: Per-node loop used by the graph traversal helper.
  # Takes an iterable of NodeDefs and returns the set of (op, kernel) pairs found, skipping nodes
  # for which no kernel is found (and the op is not allowlisted). May mutate node devices and log
  # warnings via get_ops_from_nodedef.
  """Gets the ops and kernels needed from the list of NodeDef.

  If a NodeDef's op is not in the allowlist of ops without kernel and there is
  no kernel found for this NodeDef, then skip that NodeDef and proceed to the
  next one.

  Args:
    node_defs: list of NodeDef's to get op/kernel information.

  Returns:
    A set of (op_name, kernel_name) tuples.
  """
  ops = set()
  for node_def in node_defs:
    op_and_kernel = get_ops_from_nodedef(node_def)
    if op_and_kernel:
      ops.add(op_and_kernel)
  return ops


def get_ops_and_kernels(proto_fileformat, proto_files, default_ops_str):
  # [FUNCTION DOC]
  # Collects the combined set of needed ops and kernels from several model files. Inputs:
  # proto_fileformat, one of ops_list (JSON list of pairs), rawproto (binary GraphDef) or
  # textproto (text GraphDef); proto_files, a list of paths; default_ops_str, a comma separated
  # list of op:kernel pairs always added (empty, null or the word all adds nothing).
  # Returns a sorted list of unique (op, kernel) pairs. Any other format value trips an assertion
  # error when a file is not an ops list; a default entry not of the form op:kernel raises a
  # value error. Logs each loaded file; reads files without closing handles explicitly.
  """Gets the ops and kernels needed from the model files."""
  ops = set()

  for proto_file in proto_files:
    tf_logging.info('Loading proto file %s', proto_file)
    # Load ops list file.
    if proto_fileformat == 'ops_list':
      ops = ops.union(_get_ops_from_ops_list(proto_file))
      continue

    # Load GraphDef.
    file_data = gfile.GFile(proto_file, 'rb').read()
    if proto_fileformat == 'rawproto':
      graph_def = graph_pb2.GraphDef.FromString(file_data)
    else:
      assert proto_fileformat == 'textproto'
      graph_def = text_format.Parse(file_data, graph_pb2.GraphDef())
    ops = ops.union(_get_ops_from_graphdef(graph_def))

  # Add default ops.
  if default_ops_str and default_ops_str != 'all':
    for s in default_ops_str.split(','):
      op, kernel = s.split(':')
      op_and_kernel = (op, kernel)
      if op_and_kernel not in ops:
        ops.add(op_and_kernel)

  return sorted(ops)


def get_header_from_ops_and_kernels(ops_and_kernels,
                                    include_all_ops_and_kernels):
  # [FUNCTION DOC]
  # Renders the C++ header text for selective registration. Inputs: ops_and_kernels, an iterable
  # of (op, kernel class) pairs, and include_all_ops_and_kernels, a flag; when true the pairs are
  # ignored and macros that accept every op, kernel and gradient are emitted. Otherwise the header
  # embeds helper constexpr functions, the list of needed kernel classes (null kernels skipped),
  # a function accepting only the listed ops, and a gradient macro that is true only if the
  # symbolic gradient op is present. The first line records the running script name taken from
  # the process arguments. Returns the header as one newline-joined string; no files are written.
  """Returns a header for use with tensorflow SELECTIVE_REGISTRATION.

  Args:
    ops_and_kernels: a set of (op_name, kernel_class_name) pairs to include.
    include_all_ops_and_kernels: if True, ops_and_kernels is ignored and all op
      kernels are included.

  Returns:
    the string of the header that should be written as ops_to_register.h.
  """
  ops_and_kernels = sorted(ops_and_kernels)
  ops = set(op for op, _ in ops_and_kernels)
  result_list = []

  def append(s):
    # [FUNCTION DOC]
    # HELPER: Nested line collector for building the header text.
    # Appends the given string to the enclosing result list; returns nothing.
    result_list.append(s)

  _, script_name = os.path.split(sys.argv[0])
  append('// This file was autogenerated by %s' % script_name)
  append('#ifndef OPS_TO_REGISTER')
  append('#define OPS_TO_REGISTER')

  if include_all_ops_and_kernels:
    append('#define SHOULD_REGISTER_OP(op) true')
    append('#define SHOULD_REGISTER_OP_KERNEL(clz) true')
    append('#define SHOULD_REGISTER_OP_GRADIENT true')
  else:
    line = """
    namespace {
      constexpr const char* skip(const char* x) {
        return (*x) ? (*x == ' ' ? skip(x + 1) : x) : x;
      }

      constexpr bool isequal(const char* x, const char* y) {
        return (*skip(x) && *skip(y))
                   ? (*skip(x) == *skip(y) && isequal(skip(x) + 1, skip(y) + 1))
                   : (!*skip(x) && !*skip(y));
      }

      template<int N>
      struct find_in {
        static constexpr bool f(const char* x, const char* const y[N]) {
          return isequal(x, y[0]) || find_in<N - 1>::f(x, y + 1);
        }
      };

      template<>
      struct find_in<0> {
        static constexpr bool f(const char* x, const char* const y[]) {
          return false;
        }
      };
    }  // end namespace
    """
    line += 'constexpr const char* kNecessaryOpKernelClasses[] = {\n'
    for _, kernel_class in ops_and_kernels:
      if kernel_class is None:
        continue
      line += '"%s",\n' % kernel_class
    line += '};'
    append(line)
    append('#define SHOULD_REGISTER_OP_KERNEL(clz) '
           '(find_in<sizeof(kNecessaryOpKernelClasses) '
           '/ sizeof(*kNecessaryOpKernelClasses)>::f(clz, '
           'kNecessaryOpKernelClasses))')
    append('')

    append('constexpr inline bool ShouldRegisterOp(const char op[]) {')
    append('  return false')
    for op in sorted(ops):
      append('     || isequal(op, "%s")' % op)
    append('  ;')
    append('}')
    append('#define SHOULD_REGISTER_OP(op) ShouldRegisterOp(op)')
    append('')

    append('#define SHOULD_REGISTER_OP_GRADIENT ' +
           ('true' if 'SymbolicGradient' in ops else 'false'))

  append('#endif')
  return '\n'.join(result_list)


def get_header(graphs,
               proto_fileformat='rawproto',
               default_ops='NoOp:NoOp,_Recv:RecvOp,_Send:SendOp'):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring promises a header string, but when no ops or kernels are found it
  # prints an error message and returns the integer 1 instead.
  # Top-level entry point: builds the selective registration header for a list of model files.
  # Inputs: graphs, list of file paths; proto_fileformat, textproto, rawproto (default) or
  # ops_list; default_ops, comma separated op:kernel pairs always included, or the word all to
  # include everything. Returns the header text (see the header builder) and prints to standard
  # output on failure. Exceptions from file reading and parsing propagate.
  """Computes a header for use with tensorflow SELECTIVE_REGISTRATION.

  Args:
    graphs: a list of paths to GraphDef files to include.
    proto_fileformat: optional format of proto file, either 'textproto',
      'rawproto' (default) or ops_list. The ops_list is the file contain the
      list of ops in JSON format, Ex: "[["Transpose", "TransposeCpuOp"]]".
    default_ops: optional comma-separated string of operator:kernel pairs to
      always include implementation for. Pass 'all' to have all operators and
      kernels included. Default: 'NoOp:NoOp,_Recv:RecvOp,_Send:SendOp'.

  Returns:
    the string of the header that should be written as ops_to_register.h.
  """
  ops_and_kernels = get_ops_and_kernels(proto_fileformat, graphs, default_ops)
  if not ops_and_kernels:
    print('Error reading graph!')
    return 1

  return get_header_from_ops_and_kernels(ops_and_kernels, default_ops == 'all')
