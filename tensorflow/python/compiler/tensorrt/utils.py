# [MODULE DOC]
# Utilities for TF-TRT: disabling non-TRT Grappler optimizers, TensorRT version checks (linked
# and loaded), experimental-feature flag lookup via environment variable, inspection helpers
# for graph nodes and TRT engine ops (dtypes, shapes, op counts) and a GraphViz exporter.
# Helper module used by the TF-TRT converter and its tests. Imports the native TF-TRT
# extension; no other import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Exposes the Python wrapper conversion to trt_graph."""

import collections
import os
import re

from packaging import version

from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes


def disable_non_trt_optimizers_in_rewriter_config(rewriter_config):
  # [FUNCTION DOC]
  # Turns off every Grappler optimizer in a rewriter configuration except what TF-TRT needs.
  # Input: rewriter_config, a rewriter configuration protocol buffer, mutated in place.
  # Sets arithmetic, auto mixed precision, auto parallel, constant folding, debug stripper,
  # dependency, function, implementation selector, layout, loop, memory, pin-to-host, remapping,
  # scoped allocator and shape optimizers to off, enables the meta optimizer (required for
  # TF-TRT), disables model pruning and sets the minimum graph node count to -1.
  # Returns nothing.
  """Modifies rewriter_config to disable all non-TRT optimizations."""
  off = rewriter_config_pb2.RewriterConfig.OFF

  rewriter_config.arithmetic_optimization = off
  rewriter_config.auto_mixed_precision = off
  rewriter_config.auto_parallel.enable = False
  rewriter_config.constant_folding = off
  rewriter_config.debug_stripper = off
  rewriter_config.dependency_optimization = off
  # This one needs to be ON to allow TF-TRT
  rewriter_config.disable_meta_optimizer = False
  rewriter_config.disable_model_pruning = True
  rewriter_config.function_optimization = off
  rewriter_config.implementation_selector = off
  rewriter_config.layout_optimizer = off
  rewriter_config.loop_optimization = off
  rewriter_config.memory_optimization = (
      rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
  rewriter_config.min_graph_nodes = -1
  rewriter_config.pin_to_host_optimization = off
  rewriter_config.remapping = off
  rewriter_config.scoped_allocator_optimization = off
  rewriter_config.shape_optimization = off


def version_tuple_to_string(ver_tuple):
  # [FUNCTION DOC]
  # Formats a three-part version tuple as a dotted version string.
  # Input: ver_tuple, a tuple of exactly three items (major, minor, patch).
  # Output: string with the items joined by dots. Raises an assertion error if the input is
  # not a tuple or does not have exactly three items.
  assert isinstance(ver_tuple, tuple)
  assert len(ver_tuple) == 3

  ver_tuple = [str(x) for x in ver_tuple]
  return ".".join(ver_tuple)


def _is_tensorrt_version_greater_equal(trt_ver, target_ver):
  # [FUNCTION DOC]
  # HELPER: shared comparison for the linked/loaded TensorRT version checks.
  # Inputs: trt_ver and target_ver, three-item version tuples. Output: true if trt_ver is at
  # least target_ver, compared as parsed version objects. Asserts on malformed tuples.
  trt_ver = version.Version(version_tuple_to_string(trt_ver))
  target_ver = version.Version(version_tuple_to_string(target_ver))

  return trt_ver >= target_ver


def is_linked_tensorrt_version_greater_equal(major, minor=0, patch=0):
  # [FUNCTION DOC]
  # Checks whether the TensorRT version TensorFlow was linked (built) against is at least the
  # given version. Inputs: major (required integer), minor and patch (optional, default 0).
  # Output: boolean. Queries the native extension for the linked version.
  ver = _pywrap_py_utils.get_linked_tensorrt_version()
  return _is_tensorrt_version_greater_equal(ver, (major, minor, patch))


def is_loaded_tensorrt_version_greater_equal(major, minor=0, patch=0):
  # [FUNCTION DOC]
  # Checks whether the TensorRT library actually loaded at runtime is at least the given
  # version. Inputs: major (required integer), minor and patch (optional, default 0).
  # Output: boolean. Queries the native extension for the loaded version.
  ver = _pywrap_py_utils.get_loaded_tensorrt_version()
  return _is_tensorrt_version_greater_equal(ver, (major, minor, patch))


def is_experimental_feature_activated(feature_name):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring has no Returns section; it returns a boolean.
  # Reads the comma-separated environment variable listing experimental TF-TRT features
  # (empty if unset) and returns true if feature_name is one of the entries (exact match, no
  # whitespace trimming). Input: feature_name, a string.
  """Determines if a TF-TRT experimental feature is enabled.

  This helper function checks if an experimental feature was enabled using
  the environment variable `TF_TRT_EXPERIMENTAL_FEATURES=feature_1,feature_2`.

  Args:
    feature_name: Name of the feature being tested for activation.
  """

  return (feature_name
          in os.environ.get("TF_TRT_EXPERIMENTAL_FEATURES",
                            default="").split(","))


def _convert_dtype_id_to_str(dtype):
  # [FUNCTION DOC]
  # HELPER: converts dtype enum values to names for the node/engine inspection functions.
  # Input: dtype, an integer dtype id or an iterable of ids. Output: the name string for an
  # integer, otherwise a list of names. Uses the private dtype-to-string table, so an unknown
  # id raises a key error.
  """Helper function to convert a dtype id to a corresponding string name."""
  if isinstance(dtype, int):
    return dtypes._TYPE_TO_STRING[dtype]
  else:
    return [dtypes._TYPE_TO_STRING[d] for d in dtype]


def get_node_compute_dtype(node):
  # [FUNCTION DOC]
  # KNOWN ISSUE: returns nothing (null) if no attribute yields a type, e.g. a node with none of
  # the inspected attributes; the broad exception handler swallows lookup errors silently.
  # Determines the compute dtype name of a graph node definition. Tries attributes in order:
  # precision mode (TRT engine ops; FP32/FP16/INT8 map to float32/float16/int8, other values
  # to "unknown", an empty value is skipped), destination type (casts), dtype (placeholders),
  # then the generic type attribute. Input: node, a node definition. Output: dtype name string.
  """Returns the compute DType of a GraphDef Node."""
  # Note: Order is important, by default TF Node compute dtype is mentioned
  # under `T` key, unless these nodes are one of ["TRTEngineOP", "Cast", "Plh"].
  for type_key in [
      "precision_mode",  # TRTEngineOp
      "DstT",  # Cast Nodes
      "dtype",  # Placeholder
      "T",  # Everything Else
  ]:
    try:
      precision_val = node.attr[type_key]
      if type_key == "precision_mode":
        precision_val = precision_val.s.decode("utf-8")
        if precision_val == "":
          continue
        if precision_val == "FP32":
          return "float32"
        elif precision_val == "FP16":
          return "float16"
        elif precision_val == "INT8":
          return "int8"
        else:
          return "unknown"
      else:
        return _convert_dtype_id_to_str(precision_val.type)
    except Exception as e:
      continue


def get_node_io_shapes(node, key):
  # [FUNCTION DOC]
  # Reads a list-of-shapes attribute of a node definition. Inputs: node, a node definition;
  # key, the attribute name (e.g. input or output shapes). Output: list with one list of
  # dimension sizes per shape.
  """Returns the input/output shapes of a GraphDef Node."""
  out_shape = []
  for shape in node.attr[key].list.shape:
    out_shape.append([dim.size for dim in shape.dim])
  return out_shape


def get_trtengineop_io_dtypes(node, key):
  # [FUNCTION DOC]
  # Gets dtype names of a TRT engine op's inputs or outputs. Inputs: node, the engine op node
  # definition; key, the attribute name holding the dtype list. Output: list of dtype names.
  """Returns the input/output dtypes of a TRTEngineOp."""
  return _convert_dtype_id_to_str(node.attr[key].list.type)


def get_trtengineop_io_nodes_count(node, key):
  # [FUNCTION DOC]
  # Counts the inputs or outputs of a TRT engine op. Inputs: node, the engine op node
  # definition; key, the attribute name holding the type list. Output: integer length.
  """Returns the number of input/output nodes of a TRTEngineOp."""
  return len(node.attr[key].list.type)


def get_trtengineop_node_op_count(graphdef, node_name):
  # [FUNCTION DOC]
  # KNOWN ISSUE: if the native segment function is not found in the graph library, the node
  # count variable is never assigned and an unbound-variable error is raised.
  # Finds the library function named after the engine node with the native-segment suffix and
  # counts its nodes and per-op-type occurrences. Inputs: graphdef, a graph definition with a
  # function library; node_name, the engine op name. Output: pair of (node count, mapping from
  # op type name to count).
  """Counts the number of nodes and OP types of a given TRTEngineOp."""
  ops_in_engine = collections.defaultdict(int)
  for func in graphdef.library.function:
    if f"{node_name}_native_segment" == func.signature.name:
      node_count = len(func.node_def)
      for node in func.node_def:
        ops_in_engine[node.op] += 1
      break
  return node_count, ops_in_engine


class DTypeIndex(dict):
  # [CLASS DOC]
  # HELPER: assigns graph colors to dtypes for the GraphViz export function.
  # A mapping from dtype names to consecutive integer indices starting at 1, in first-seen
  # order. Created empty and filled lazily through get_dtype_index.
  """Helper class to create an index of dtypes with incremental values."""

  def get_dtype_index(self, dtype):
    # [FUNCTION DOC]
    # Returns the index of dtype, assigning the next unused integer (current size plus one) on
    # first sight. Mutates the mapping. Input: dtype, a hashable dtype name.
    if dtype not in self:
      self[dtype] = len(self) + 1
    return self[dtype]


def draw_graphdef_as_graphviz(graphdef, dot_output_filename):
  # [FUNCTION DOC]
  # Writes a GraphViz dot file visualizing a graph definition and prints usage hints.
  # Inputs: graphdef, a graph definition; dot_output_filename, path of the file to write
  # (overwritten). Each node is drawn with op type (TRT engine ops also show their native
  # segment node count), compute dtype color and device; edges follow inputs (control-input
  # markers and output indices stripped). A legend cluster lists dtypes, and invisible edges
  # align the legend with nodes lacking inputs. Side effects: file write and printing to
  # standard output. Returns nothing.
  """Exports a GraphDef to GraphViz format.

  - Step 1: Drawing Each Node of the compute GraphDef.
  - Step 2: Create nodes for each collected dtype in the graph.
  - Step 3: Creating invisible links to align properly the legend.

  Each node consequently mentions:
  - Op Type
  - Compute Dtype
  - Compute Device
  """

  dtype_index = DTypeIndex()

  with open(dot_output_filename, "w") as f:
    print("digraph tftrt_converted_graph {", file=f)

    print("  graph [fontsize=10 fontname=\"Verdana\"];", file=f)
    # ColorScheme Documentation: https://graphviz.org/doc/info/colors.html
    print(
        "  node [style=filled height=0.55 colorscheme=set312 shape=box];",
        file=f)

    # Step 1: Parsing the graph and drawing OPs one by one.
    print("\n  subgraph tensorflow_graph {", file=f)
    print("    node [width=1.35];", file=f)
    nodes_with_no_inputs = []
    for node in graphdef.node:
      output_name = node.name

      node_precision = get_node_compute_dtype(node)
      color_idx = dtype_index.get_dtype_index(node_precision)

      device_key = node.device.split("/")[-1]
      if not device_key:
        device_key = "device:Unspecified"

      if node.op == "TRTEngineOp":
        node_count, _ = get_trtengineop_node_op_count(graphdef, output_name)
        node_label = f"{output_name} [{node_count}]"
      else:
        node_label = f"{node.op}"

      # Note: double space before <br/> is necessary for formatting.
      node_label = f"<b>{node_label}</b>  <br/><i>{device_key}</i>"

      print(
          f"    \"{output_name}\" [label=<{node_label}> "
          f"fillcolor={color_idx}];",
          file=f)

      if len(node.input):
        for input_full_name in node.input:
          parts = input_full_name.split(":")
          input_name = re.sub(r"^\^", "", parts[0])
          print(f"  \"{input_name}\" -> \"{output_name}\";", file=f)
      else:
        nodes_with_no_inputs.append(output_name)
    print("  }", file=f)

    # Step 2: Creating the DType Nodes previously found in Step 1.
    print("\n  subgraph cluster_legend {", file=f)
    print("    label=\"Compute Dtype Legend\";", file=f)
    print("    margin=\"30\";", file=f)
    print("    node [width=2];", file=f)

    for dtype, color_idx in dtype_index.items():
      print(
          f"    {dtype} [fillcolor={color_idx} label=<<b>{dtype}</b>>];",
          file=f)

    print("  }", file=f)

    # Step 3: Alignment of the legend with the graph.
    print("\n  edge[style=\"invisible\", dir=\"none\"];", file=f)
    for dtype in dtype_index.keys():
      for node_name in nodes_with_no_inputs:
        print(f"  \"{dtype}\" -> \"{node_name}\"", file=f)

    print("}", file=f)

  print("\n===================================================================")
  print(f"Graph Visualization Exported to: `{dot_output_filename}`.")
  print("We recommend using https://edotor.net/ to visualize the .dot file.")
  print("You can also use `graphviz` utility to convert them to PNG format:")
  print("  - `sudo apt install -y graphviz`")
  print("  - `dot -Tpng <input_filename>.dot -o <output_filename>.png`")
  print("===================================================================\n")
