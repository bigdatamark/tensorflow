# [MODULE DOC]
# Helper module for the example generator: produces an HTML report table showing, for every
# attempted model conversion, whether plain TensorFlow and the TFLite converter succeeded, with
# clickable log display. Exposes the status constants (failed, success, not run), the function
# make_report_table and the log sanitizer escape_and_normalize. No import-time side effects.
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
"""Make HTML tables that report where TF and TFLite failed to convert models.

This is primarily used by generate_examples.py. See it or
`make_report_table` for more details on usage.
"""
import html
import json
import re

FAILED = "FAILED"
SUCCESS = "SUCCESS"
NOTRUN = "NOTRUN"


def make_report_table(fp, title, reports):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring example uses the status text FAILURE, but the module constant (and
  # the colouring logic) uses FAILED. The title is inserted into the HTML without escaping.
  # Writes a complete HTML page to the file-like object fp. Parameters: title (heading text) and
  # reports (list of pairs of a parameter mapping and a result mapping with TensorFlow status,
  # TFLite converter status, and their logs). The list is sorted in place: first by converter
  # status, then by TensorFlow status in reverse. The page embeds the normalized logs as script
  # data, and has a table with one column per parameter key seen in any record plus two coloured
  # status cells per row; clicking a status cell shows its log in a side panel. Returns nothing.
  """Make an HTML report of the success/failure reports.

  Args:
    fp: File-like object in which to put the html.
    title: "Title of the zip file this pertains to."
    reports: a list of conversion attempts. (report_args, report_vals) i.e.
      ({"shape": [1,2,3], "type": "tf.float32"},
       {"tf": "SUCCESS", "tflite_converter": "FAILURE",
        "tf_log": "", "tflite_converter_log": "Unsupported type."})
  """
  # sort reports by if TFLite converter failure and then TF failure (reversed)
  reports.sort(key=lambda x: x[1]["tflite_converter"], reverse=False)
  reports.sort(key=lambda x: x[1]["tf"], reverse=True)
  def result_cell(x, row, col):
    # [FUNCTION DOC]
    # HELPER: row rendering for make_report_table.
    # Writes one table cell showing the escaped text of status x, green when successful, red when
    # failed, grey otherwise, with a click handler that shows the log for the given row and
    # column index (0 for TensorFlow, 1 for converter). Returns nothing.
    """Produce a cell with the condition string `x`."""
    s = html.escape(repr(x), quote=True)
    color = "#44ff44" if x == SUCCESS else (
        "#ff4444" if x == FAILED else "#eeeeee")
    handler = "ShowLog(%d, %d)" % (row, col)
    fp.write("<td style='background-color: %s' onclick='%s'>%s</td>\n" % (
        color, handler, s))

  fp.write("""<html>
<head>
<title>tflite report</title>
<style>
body { font-family: Arial; }
th { background-color: #555555; color: #eeeeee; }
td { vertical-align: top; }
td.horiz {width: 50%;}
pre { white-space: pre-wrap; word-break: keep-all; }
table {width: 100%;}
</style>
</head>
""")
  # Write the log data to a javascript variable and also make a function
  # in javascript to show the log when an item is clicked.
  fp.write("<script> \n")
  fp.write("""
function ShowLog(row, col) {

var log = document.getElementById("log");
log.innerHTML = "<pre>" + data[row][col]  + "</pre>";
}
""")
  fp.write("var data = \n")
  logs = json.dumps([[escape_and_normalize(x[1]["tf_log"]),
                      escape_and_normalize(x[1]["tflite_converter_log"])
                     ] for x in reports])
  fp.write(logs)
  fp.write(";</script>\n")

  # Write the main table and use onclick on the items that have log items.
  fp.write("""
<body>
<h1>TensorFlow Lite Conversion</h1>
<h2>%s</h2>
""" % title)

  # Get a list of keys that are in any of the records.
  param_keys = {}
  for params, _ in reports:
    for k in params.keys():
      param_keys[k] = True

  fp.write("<table>\n")
  fp.write("<tr><td class='horiz'>\n")
  fp.write("<div style='height:1000px; overflow:auto'>\n")
  fp.write("<table>\n")
  fp.write("<tr>\n")
  for p in param_keys:
    fp.write("<th>%s</th>\n" % html.escape(p, quote=True))
  fp.write("<th>TensorFlow</th>\n")
  fp.write("<th>TensorFlow Lite Converter</th>\n")
  fp.write("</tr>\n")
  for idx, (params, vals) in enumerate(reports):
    fp.write("<tr>\n")
    for p in param_keys:
      fp.write("  <td>%s</td>\n" %
               html.escape(repr(params.get(p, None)), quote=True))

    result_cell(vals["tf"], idx, 0)
    result_cell(vals["tflite_converter"], idx, 1)
    fp.write("</tr>\n")
  fp.write("</table>\n")
  fp.write("</div>\n")
  fp.write("</td>\n")
  fp.write("<td class='horiz' id='log'></td></tr>\n")
  fp.write("</table>\n")
  fp.write("<script>\n")
  fp.write("</script>\n")
  fp.write("""
    </body>
    </html>
    """)


def escape_and_normalize(log):
  # [FUNCTION DOC]
  # HELPER: log sanitizer for make_report_table.
  # Replaces temporary file paths (those starting with a tmp directory and followed by a space)
  # and build work directory prefixes with fixed placeholders so output is deterministic, then
  # HTML-escapes the text. Takes a log string and returns the cleaned string.
  # These logs contain paths like /tmp/tmpgmypg3xa that are inconsistent between
  # builds. This replaces these inconsistent paths with a consistent placeholder
  # so the output is deterministic.
  log = re.sub(r"/tmp/[^ ]+ ", "/NORMALIZED_TMP_FILE_PATH ", log)
  log = re.sub(r"/build/work/[^/]+", "/NORMALIZED_BUILD_PATH", log)
  return html.escape(log, quote=True)
