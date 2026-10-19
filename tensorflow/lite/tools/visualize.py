#!/usr/bin/env python
# [MODULE DOC]
# Command line tool and library that renders a TensorFlow Lite flatbuffer model as a single
# self-contained HTML page. The page lists model metadata, per-subgraph tensor and operator
# tables, a d3-based interactive graph drawing, the buffers and the operator codes.
# Exposes create_html (main entry point), flatbuffer-to-dictionary conversion helpers, small
# cell-formatting mapper classes and the main function for command line use.
# Import-time side effect: chooses the schema module from the tensorflow package or from the
# tflite_runtime package depending on where this file is installed. No flags are defined.
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
"""This tool creates an html visualization of a TensorFlow Lite graph.

Example usage:

python visualize.py foo.tflite foo.html
"""

import json
import os
import re
import sys
import numpy as np

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("tflite_runtime", "visualize")):
  # This file is part of tensorflow package.
  from tensorflow.lite.python import schema_py_generated as schema_fb
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import schema_py_generated as schema_fb

# A CSS description for making the visualizer
_CSS = """
<html>
<head>
<style>
body {font-family: sans-serif; background-color: #fa0;}
table {background-color: #eca;}
th {background-color: black; color: white;}
/* Constrain table cells to a max size and make them scrollable. */
.data-table td {
  max-width: 900px;
}
.data-table .cell-content {
  max-height: 200px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
h1 {
  background-color: ffaa00;
  padding:5px;
  color: black;
}

svg {
  margin: 10px;
  border: 2px;
  border-style: solid;
  border-color: black;
  background: white;
}

div {
  border-radius: 5px;
  background-color: #fec;
  padding:5px;
  margin:5px;
}

.tooltip {color: blue;}
.tooltip .tooltipcontent  {
    visibility: hidden;
    color: black;
    background-color: yellow;
    padding: 5px;
    border-radius: 4px;
    position: absolute;
    z-index: 1;
}
.tooltip:hover .tooltipcontent {
    visibility: visible;
}

.edges line {
  stroke: #333;
}

text {
  font-weight: bold;
}

.nodes text {
  color: black;
  pointer-events: none;
  font-family: sans-serif;
  font-size: 11px;
}
</style>

<script src="https://d3js.org/d3.v4.min.js"></script>

</head>
<body>
"""

_D3_HTML_TEMPLATE = """
  <script>
    function buildGraph() {
      // Build graph data
      var graph = %s;

      var svg = d3.select("#subgraph%d")
      var width = svg.attr("width");
      var height = svg.attr("height");
      // Make the graph scrollable.
      svg = svg.call(d3.zoom().on("zoom", function() {
        svg.attr("transform", d3.event.transform);
      })).append("g");


      var color = d3.scaleOrdinal(d3.schemeDark2);

      var simulation = d3.forceSimulation()
          .force("link", d3.forceLink().id(function(d) {return d.id;}))
          .force("charge", d3.forceManyBody())
          .force("center", d3.forceCenter(0.5 * width, 0.5 * height));

      var edge = svg.append("g").attr("class", "edges").selectAll("line")
        .data(graph.edges).enter().append("path").attr("stroke","black").attr("fill","none")

      // Make the node group
      var node = svg.selectAll(".nodes")
        .data(graph.nodes)
        .enter().append("g")
        .attr("x", function(d){return d.x})
        .attr("y", function(d){return d.y})
        .attr("transform", function(d) {
          return "translate( " + d.x + ", " + d.y + ")"
        })
        .attr("class", "nodes")
          .call(d3.drag()
              .on("start", function(d) {
                if(!d3.event.active) simulation.alphaTarget(1.0).restart();
                d.fx = d.x;d.fy = d.y;
              })
              .on("drag", function(d) {
                d.fx = d3.event.x; d.fy = d3.event.y;
              })
              .on("end", function(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = d.fy = null;
              }));
      // Within the group, draw a box for the node position and text
      // on the side.

      var node_width = 150;
      var node_height = 30;

      node.append("rect")
          .attr("r", "5px")
          .attr("width", node_width)
          .attr("height", node_height)
          .attr("rx", function(d) { return d.group == 1 ? 1 : 10; })
          .attr("stroke", "#000000")
          .attr("fill", function(d) { return d.group == 1 ? "#dddddd" : "#000000"; })
      node.append("text")
          .text(function(d) { return d.name; })
          .attr("x", 5)
          .attr("y", 20)
          .attr("fill", function(d) { return d.group == 1 ? "#000000" : "#eeeeee"; })
      // Setup force parameters and update position callback


      var node = svg.selectAll(".nodes")
        .data(graph.nodes);

      // Bind the links
      var name_to_g = {}
      node.each(function(data, index, nodes) {
        console.log(data.id)
        name_to_g[data.id] = this;
      });

      function proc(w, t) {
        return parseInt(w.getAttribute(t));
      }
      edge.attr("d", function(d) {
        function lerp(t, a, b) {
          return (1.0-t) * a + t * b;
        }
        var x1 = proc(name_to_g[d.source],"x") + node_width /2;
        var y1 = proc(name_to_g[d.source],"y") + node_height;
        var x2 = proc(name_to_g[d.target],"x") + node_width /2;
        var y2 = proc(name_to_g[d.target],"y");
        var s = "M " + x1 + " " + y1
            + " C " + x1 + " " + lerp(.5, y1, y2)
            + " " + x2 + " " + lerp(.5, y1, y2)
            + " " + x2  + " " + y2
      return s;
    });

  }
  buildGraph()
</script>
"""


def TensorTypeToName(tensor_type):
  # [FUNCTION DOC]
  # HELPER: converts enum values for the HTML tables built by create_html.
  # Takes a numeric tensor type code and returns the name of the matching member of the schema
  # tensor type enumeration, or nothing/null if no member has that value. Scans the enumeration
  # linearly; if several names share a value the first one found is returned. No side effects.
  """Converts a numerical enum to a readable tensor type."""
  for name, value in schema_fb.TensorType.__dict__.items():
    if value == tensor_type:
      return name
  return None


def BuiltinCodeToName(code):
  # [FUNCTION DOC]
  # HELPER: converts enum values for the HTML tables built by create_html.
  # Takes a numeric builtin operator code and returns the name of the matching member of the
  # schema builtin operator enumeration, or nothing/null if none matches. Linear scan; the first
  # match wins. No side effects.
  """Converts a builtin op code enum to a readable name."""
  for name, value in schema_fb.BuiltinOperator.__dict__.items():
    if value == code:
      return name
  return None


def NameListToString(name_list):
  # [FUNCTION DOC]
  # HELPER: decodes names stored as integer arrays in the flatbuffer for display.
  # Input is either a string (returned unchanged), a list of integer character codes, or
  # nothing/null. A list is turned into a string by converting each element to a character; a
  # null input yields an empty string. Raises an error if an element cannot be converted to an
  # integer or is out of the valid character range.
  """Converts a list of integers to the equivalent ASCII string."""
  if isinstance(name_list, str):
    return name_list
  else:
    result = ""
    if name_list is not None:
      for val in name_list:
        result = result + chr(int(val))
    return result


class OpCodeMapper:
  # [CLASS DOC]
  # HELPER: cell formatter used by create_html and GenerateGraph.
  # Callable object that maps an operator-code index of a model to a display string made of the
  # operator name and the index. Holds a mapping from index to name built once at construction.
  # Custom operators are shown by their custom code string instead of the generic name.
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    # [FUNCTION DOC]
    # Constructor. Takes the model dictionary (as produced by the flatbuffer conversion) and
    # builds the index-to-name table from its list of operator codes: each entry's builtin code
    # is converted to a name, and when that name is the custom marker the custom code (a name
    # list) is decoded to a string instead. Raises a key error if the expected fields are missing.
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      self.code_to_name[idx] = BuiltinCodeToName(d["builtin_code"])
      if self.code_to_name[idx] == "CUSTOM":
        self.code_to_name[idx] = NameListToString(d["custom_code"])

  def __call__(self, x):
    # [FUNCTION DOC]
    # Takes an operator-code index and returns text of the form name followed by the index in
    # parentheses. Unknown indices use a placeholder name meaning unknown. Names that could not
    # be resolved (null) are formatted as the text "None".
    if x not in self.code_to_name:
      s = "<UNKNOWN>"
    else:
      s = self.code_to_name[x]
    return "%s (%d)" % (s, x)


class DataSizeMapper:
  # [CLASS DOC]
  # HELPER: cell formatter used for the buffer data column in create_html.
  # Stateless callable that turns a buffer's contents into a human readable byte count.
  """For buffers, report the number of bytes."""

  def __call__(self, x):
    # [FUNCTION DOC]
    # Takes buffer contents (anything with a length) or nothing/null. Returns text giving the
    # number of bytes, or a double dash placeholder when the value is null.
    if x is not None:
      return "%d bytes" % len(x)
    else:
      return "--"


class TensorMapper:
  # [CLASS DOC]
  # HELPER: cell formatter used by create_html for operator input and output columns.
  # Callable object bound to one subgraph dictionary. Given a list of tensor indices it produces
  # HTML showing the list, with a hover tooltip describing each tensor (index, name, type, shape
  # and shape signature).
  """Maps a list of tensor indices to a tooltip hoverable indicator of more."""

  def __init__(self, subgraph_data):
    # [FUNCTION DOC]
    # Constructor. Stores the subgraph dictionary (which must contain a list of tensors) for
    # later lookups. No other effects.
    self.data = subgraph_data

  def __call__(self, x):
    # [FUNCTION DOC]
    # Takes a list of tensor indices or nothing/null and returns an HTML fragment. Null gives an
    # empty string. Otherwise the fragment has a tooltip span listing, for each index, the tensor
    # name, type name, shape and shape signature (empty brackets when absent), followed by the
    # visible text form of the index list. Raises an error for out-of-range indices or when a
    # tensor type name cannot be resolved (null cannot be concatenated to text).
    html = ""
    if x is None:
      return html

    html += "<span class='tooltip'><span class='tooltipcontent'>"
    for i in x:
      tensor = self.data["tensors"][i]
      html += str(i) + " "
      html += NameListToString(tensor["name"]) + " "
      html += TensorTypeToName(tensor["type"]) + " "
      html += (repr(tensor["shape"]) if "shape" in tensor else "[]")
      html += (repr(tensor["shape_signature"])
               if "shape_signature" in tensor else "[]") + "<br>"
    html += "</span>"
    html += repr(x)
    html += "</span>"
    return html


def QuantizationMapper(q):
  # [FUNCTION DOC]
  # HELPER: cell formatter for the quantization column of the tensor table.
  # Takes a quantization mapping (or empty/null) and returns a brace-delimited text of
  # key-and-value pairs. Empty input gives an empty string. List values longer than 20 entries
  # are abbreviated to their first 10 and last 10 elements with an ellipsis between them.
  """Pretty-print the quantization dictionary, truncating large arrays."""
  if not q:
    return ""

  items_str = []
  for key, value in q.items():
    key_str = repr(key)
    # In TFLite, quantization arrays can be large.
    if isinstance(value, list) and len(value) > 20:
      head = value[:10]
      tail = value[-10:]
      value_str = (f"[{', '.join(map(repr, head))}, ..., "
                   f"{', '.join(map(repr, tail))}]")
    else:
      value_str = repr(value)
    items_str.append(f"{key_str}: {value_str}")

  return f"{{{', '.join(items_str)}}}"


def GenerateGraph(subgraph_idx, g, opcode_mapper):
  # [FUNCTION DOC]
  # HELPER: builds the graph drawing section of the page for create_html.
  # Inputs: the subgraph index (used to target the matching svg element), the subgraph
  # dictionary (with operators and tensors), and an operator-code mapper callable. Builds one
  # node per operator (group 2) and one per tensor (group 1) with initial layout coordinates
  # derived from the first operator that consumes or produces the tensor, plus edges from input
  # tensors to operators and from operators to output tensors. Returns the d3 script HTML with
  # the graph serialized as JSON embedded in it. Pure function, no side effects.
  # KNOWN ISSUE: tensor node labels read a shape attribute from a dictionary using attribute
  # access, so the shape is always the empty default and real shapes are never shown.
  """Produces the HTML required to have a d3 visualization of the dag."""

  def TensorName(idx):
    # [FUNCTION DOC]
    # HELPER: node id naming. Returns the id text for a tensor node: the letter t followed by
    # the index.
    return "t%d" % idx

  def OpName(idx):
    # [FUNCTION DOC]
    # HELPER: node id naming. Returns the id text for an operator node: the letter o followed
    # by the index.
    return "o%d" % idx

  edges = []
  nodes = []
  first = {}
  second = {}
  pixel_mult = 200  # TODO(aselle): multiplier for initial placement
  width_mult = 170  # TODO(aselle): multiplier for initial placement
  for op_index, op in enumerate(g["operators"] or []):
    if op["inputs"] is not None:
      for tensor_input_position, tensor_index in enumerate(op["inputs"]):
        if tensor_index not in first:
          first[tensor_index] = ((op_index - 0.5 + 1) * pixel_mult,
                                 (tensor_input_position + 1) * width_mult)
        edges.append({
            "source": TensorName(tensor_index),
            "target": OpName(op_index)
        })
    if op["outputs"] is not None:
      for tensor_output_position, tensor_index in enumerate(op["outputs"]):
        if tensor_index not in second:
          second[tensor_index] = ((op_index + 0.5 + 1) * pixel_mult,
                                  (tensor_output_position + 1) * width_mult)
        edges.append({
            "target": TensorName(tensor_index),
            "source": OpName(op_index)
        })

    nodes.append({
        "id": OpName(op_index),
        "name": opcode_mapper(op["opcode_index"]),
        "group": 2,
        "x": pixel_mult,
        "y": (op_index + 1) * pixel_mult
    })
  for tensor_index, tensor in enumerate(g["tensors"]):
    initial_y = (
        first[tensor_index] if tensor_index in first else
        second[tensor_index] if tensor_index in second else (0, 0))

    nodes.append({
        "id": TensorName(tensor_index),
        "name": "%r (%d)" % (getattr(tensor, "shape", []), tensor_index),
        "group": 1,
        "x": initial_y[1],
        "y": initial_y[0]
    })
  graph_str = json.dumps({"nodes": nodes, "edges": edges})

  html = _D3_HTML_TEMPLATE % (graph_str, subgraph_idx)
  return html


def GenerateTableHtml(items, keys_to_print, display_index=True):
  # [FUNCTION DOC]
  # HELPER: builds the HTML tables for create_html.
  # Builds an HTML table with one row per item. Columns come from keys_to_print, a list of pairs
  # of a key name and an optional formatter callable (null means show the raw value); a missing
  # key yields a null value which is still passed to the formatter. If display_index is true a
  # leading index column is added (default true). Cell contents are not escaped. Returns the
  # table as text; no side effects.
  """Given a list of object values and keys to print, make an HTML table.

  Args:
    items: Items to print an array of dicts.
    keys_to_print: (key, display_fn). `key` is a key in the object. i.e.
      items[0][key] should exist. display_fn is the mapping function on display.
      i.e. the displayed html cell will have the string returned by
      `mapping_fn(items[0][key])`.
    display_index: add a column which is the index of each row in `items`.

  Returns:
    An html table.
  """
  html = ""
  # Print the list of items
  html += "<table class='data-table'>\n"
  html += "<tr>\n"
  if display_index:
    html += "<th>index</th>"
  for h, mapper in keys_to_print:
    html += "<th>%s</th>" % h
  html += "</tr>\n"
  for idx, tensor in enumerate(items):
    html += "<tr>\n"
    if display_index:
      html += "<td>%d</td>" % idx
    # print tensor.keys()
    for h, mapper in keys_to_print:
      val = tensor[h] if h in tensor else None
      val = val if mapper is None else mapper(val)
      html += "<td><div class='cell-content'>%s</div></td>\n" % val

    html += "</tr>\n"
  html += "</table>\n"
  return html


def CamelCaseToSnakeCase(camel_case_input):
  # [FUNCTION DOC]
  # HELPER: renames flatbuffer attribute names for FlatbufferToDict.
  # Converts an identifier in camel case to lower-case words joined by underscores, inserting an
  # underscore before each capitalized word or lower-to-upper transition. Returns the new text.
  """Converts an identifier in CamelCase to snake_case."""
  s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
  return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def FlatbufferToDict(fb, preserve_as_numpy):
  # [FUNCTION DOC]
  # HELPER: recursive converter used by CreateDictFromFlatbuffer.
  # Inputs: a flatbuffer object tree node (or primitive/array/sequence), and a flag saying
  # whether numeric arrays below it stay as numpy arrays (true) or become plain lists (false).
  # Primitives (integer, float, string) are returned as is. Objects become mappings of
  # snake-case attribute names to converted values, skipping callables and underscore-prefixed
  # attributes; the buffers attribute always preserves numpy arrays regardless of the flag.
  # Numpy arrays are kept or listified per the flag; other sized sequences become lists of
  # converted entries; anything else is returned unchanged.
  """Converts a hierarchy of FB objects into a nested dict.

  We avoid transforming big parts of the flat buffer into python arrays. This
  speeds conversion from ten minutes to a few seconds on big graphs.

  Args:
    fb: a flat buffer structure. (i.e. ModelT)
    preserve_as_numpy: true if all downstream np.arrays should be preserved.
      false if all downstream np.array should become python arrays
  Returns:
    A dictionary representing the flatbuffer rather than a flatbuffer object.
  """
  if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
    return fb
  elif hasattr(fb, "__dict__"):
    result = {}
    for attribute_name in dir(fb):
      attribute = fb.__getattribute__(attribute_name)
      if not callable(attribute) and attribute_name[0] != "_":
        snake_name = CamelCaseToSnakeCase(attribute_name)
        preserve = True if attribute_name == "buffers" else preserve_as_numpy
        result[snake_name] = FlatbufferToDict(attribute, preserve)
    return result
  elif isinstance(fb, np.ndarray):
    return fb if preserve_as_numpy else fb.tolist()
  elif hasattr(fb, "__len__"):
    return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
  else:
    return fb


def CreateDictFromFlatbuffer(buffer_data):
  # [FUNCTION DOC]
  # HELPER: parsing step for create_html.
  # Takes the raw bytes of a TFLite flatbuffer, parses the root model, unpacks it to the
  # object-API form and converts it to nested mappings and lists (numpy arrays turned into
  # lists except buffer data). Returns the resulting model dictionary. Raises whatever the
  # flatbuffer parser raises on malformed input.
  model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
  model = schema_fb.ModelT.InitFromObj(model_obj)
  return FlatbufferToDict(model, preserve_as_numpy=False)


def create_html(tflite_input, input_is_filepath=True):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # KNOWN ISSUE: the page title is opened with a level-1 heading tag but closed with a level-2
  # closing tag, producing malformed HTML. A json input file handle is also never closed.
  # Main entry point. Inputs: a model file path (ending in tflite, bin or json) or an in-memory
  # model byte buffer, and a flag (default true) saying which of the two it is. Reads and parses
  # the model, then builds one HTML page: styling, a metadata table (filename, version,
  # description), and for each subgraph its input/output table, tensor table, operator table
  # (when it has operators) and the graph drawing. Then adds buffer and operator-code tables.
  # Also normalizes each operator code's builtin code to the larger of the builtin and
  # deprecated builtin codes. Returns the page as text. Raises a runtime error when the file
  # does not exist or has an unsupported extension. Reads a file; no other side effects.
  """Returns html description with the given tflite model.

  Args:
    tflite_input: TFLite flatbuffer model path or model object.
    input_is_filepath: Tells if tflite_input is a model path or a model object.

  Returns:
    Dump of the given tflite model in HTML format.

  Raises:
    RuntimeError: If the input is not valid.
  """

  # Convert the model into a JSON flatbuffer using flatc (build if doesn't
  # exist.
  if input_is_filepath:
    if not os.path.exists(tflite_input):
      raise RuntimeError("Invalid filename %r" % tflite_input)
    if tflite_input.endswith(".tflite") or tflite_input.endswith(".bin"):
      with open(tflite_input, "rb") as file_handle:
        file_data = bytearray(file_handle.read())
      data = CreateDictFromFlatbuffer(file_data)
    elif tflite_input.endswith(".json"):
      data = json.load(open(tflite_input))
    else:
      raise RuntimeError("Input file was not .tflite or .json")
  else:
    data = CreateDictFromFlatbuffer(tflite_input)
  html = ""
  html += _CSS
  html += "<h1>TensorFlow Lite Model</h2>"

  data["filename"] = tflite_input if input_is_filepath else (
      "Null (used model object)")  # Avoid special case

  toplevel_stuff = [("filename", None), ("version", None),
                    ("description", None)]

  html += "<table class='data-table'>\n"
  for key, mapping in toplevel_stuff:
    if not mapping:
      mapping = lambda x: x
    val = mapping(data.get(key))
    html += ("<tr><th>%s</th><td><div class='cell-content'>%s</div></td></tr>\n"
             % (key, val))
  html += "</table>\n"

  # Spec on what keys to display
  buffer_keys_to_display = [("data", DataSizeMapper())]
  operator_keys_to_display = [("builtin_code", BuiltinCodeToName),
                              ("custom_code", NameListToString),
                              ("version", None)]

  # Update builtin code fields.
  for d in data["operator_codes"]:
    d["builtin_code"] = max(d["builtin_code"], d["deprecated_builtin_code"])

  for subgraph_idx, g in enumerate(data["subgraphs"]):
    # Subgraph local specs on what to display
    html += "<div class='subgraph'>"
    tensor_mapper = TensorMapper(g)
    opcode_mapper = OpCodeMapper(data)
    op_keys_to_display = [("inputs", tensor_mapper), ("outputs", tensor_mapper),
                          ("builtin_options", None),
                          ("opcode_index", opcode_mapper)]
    tensor_keys_to_display = [("name", NameListToString),
                              ("type", TensorTypeToName), ("shape", None),
                              ("shape_signature", None), ("buffer", None),
                              ("quantization", QuantizationMapper)]

    html += "<h2>Subgraph %d</h2>\n" % subgraph_idx

    # Inputs and outputs.
    html += "<h3>Inputs/Outputs</h3>\n"
    html += GenerateTableHtml([{
        "inputs": g["inputs"],
        "outputs": g["outputs"]
    }], [("inputs", tensor_mapper), ("outputs", tensor_mapper)],
                              display_index=False)

    # Print the tensors.
    html += "<h3>Tensors</h3>\n"
    html += GenerateTableHtml(g["tensors"], tensor_keys_to_display)

    # Print the ops.
    if g["operators"]:
      html += "<h3>Ops</h3>\n"
      html += GenerateTableHtml(g["operators"], op_keys_to_display)

    # Visual graph.
    html += "<svg id='subgraph%d' width='1600' height='900'></svg>\n" % (
        subgraph_idx,)
    html += GenerateGraph(subgraph_idx, g, opcode_mapper)
    html += "</div>"

  # Buffers have no data, but maybe in the future they will
  html += "<h2>Buffers</h2>\n"
  html += GenerateTableHtml(data["buffers"], buffer_keys_to_display)

  # Operator codes
  html += "<h2>Operator Codes</h2>\n"
  html += GenerateTableHtml(data["operator_codes"], operator_keys_to_display)

  html += "</body></html>\n"

  return html


def main(argv):
  # [FUNCTION DOC]
  # Command line entry point. Takes the argument list: program name, input model path, output
  # html path. If either argument is missing it prints a usage message and returns nothing.
  # Otherwise it generates the HTML for the model and writes it to the output file (overwriting
  # it). Errors from create_html propagate.
  try:
    tflite_input = argv[1]
    html_output = argv[2]
  except IndexError:
    print("Usage: %s <input tflite> <output html>" % (argv[0]))
  else:
    html = create_html(tflite_input)
    with open(html_output, "w") as output_file:
      output_file.write(html)


if __name__ == "__main__":
  main(sys.argv)
