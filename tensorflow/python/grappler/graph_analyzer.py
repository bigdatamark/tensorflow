# [MODULE DOC]
# Command line tool that finds all subgraphs of a given size in a TensorFlow graph, counts their
# occurrences and reports them sorted, considering only the transitive fanin of the fetch nodes.
# Thin wrapper over the native graph analyzer extension. Flags: input (meta graph file path) and
# n (subgraph size) are parsed with argparse and the absl app runner is started only when the
# file runs as the main program; there are no import-time side effects.
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
# =============================================================================
"""A tool that finds all subgraphs of a given size in a TF graph.

The subgraph patterns are sorted by occurrence, and only the transitive fanin
part of the graph with regard to the fetch nodes is considered.
"""

import argparse
import sys

from absl import app

from tensorflow.python.grappler import _pywrap_graph_analyzer as tf_wrap


def main(_):
  # [FUNCTION DOC]
  # KNOWN ISSUE: relies on a module-level flags object created only in main mode, and passes
  # unset flags (null values) straight through without validation.
  # Entry point of the tool; the single argument (remaining command line arguments) is ignored.
  # Hands the input file path and the subgraph size from the command line flags to the native
  # graph analyzer, which reads the meta graph and reports the subgraph patterns (output is
  # produced by the native code). Returns nothing.
  tf_wrap.GraphAnalyzer(FLAGS.input, FLAGS.n)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input",
      type=str,
      default=None,
      help="Input file path for a TensorFlow MetaGraphDef.")
  parser.add_argument(
      "--n", type=int, default=None, help="The size of the subgraphs.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
