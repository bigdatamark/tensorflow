# [MODULE DOC]
# Command-line tool that loads a SavedModel's graph and writes it as a TensorBoard graph summary
# so it can be visualized. Exposes the import function; when run as a script it parses the model
# directory, log directory and tag set flags and runs the app. Import-time side effect: tries to
# import the TensorRT engine op library so such ops can be resolved, ignoring import failure.
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
# ================================
"""Imports a protobuf model as a graph in Tensorboard."""

import argparse
import sys

from absl import app

from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.summary import summary
from tensorflow.python.tools import saved_model_utils

# Try importing TensorRT ops if available
# TODO(aaroey): ideally we should import everything from contrib, but currently
# tensorrt module would cause build errors when being imported in
# tensorflow/contrib/__init__.py. Fix it.
# pylint: disable=unused-import,g-import-not-at-top,wildcard-import
try:
  from tensorflow.contrib.tensorrt.ops.gen_trt_engine_op import *
except ImportError:
  pass
# pylint: enable=unused-import,g-import-not-at-top,wildcard-import


def import_to_tensorboard(model_dir, log_dir, tag_set):
  # [FUNCTION DOC]
  # Loads the graph definition of the meta graph matching tag_set (a comma-separated string of
  # tags; all tags of a multi-tag set must be given) from the SavedModel directory model_dir,
  # imports it into a fresh graph inside a session, and writes that graph to a summary file writer
  # at log_dir. Prints a message telling the user how to launch TensorBoard. Returns nothing.
  # Raises errors from the loader if the directory or tag set is not found. Writes event files.
  """View an SavedModel as a graph in Tensorboard.

  Args:
    model_dir: The directory containing the SavedModel to import.
    log_dir: The location for the Tensorboard log to begin visualization from.
    tag_set: Group of tag(s) of the MetaGraphDef to load, in string format,
      separated by ','. For tag-set contains multiple tags, all tags must be
      passed in.
  Usage: Call this function with your SavedModel location and desired log
    directory. Launch Tensorboard by pointing it to the log directory. View your
    imported SavedModel as a graph.
  """
  with session.Session(graph=ops.Graph()) as sess:
    input_graph_def = saved_model_utils.get_meta_graph_def(model_dir,
                                                           tag_set).graph_def
    importer.import_graph_def(input_graph_def)

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))


def main(_):
  # [FUNCTION DOC]
  # Application main; the argument (remaining args) is ignored. Calls the import function with the
  # model directory, log directory and tag set flag values.
  # KNOWN ISSUE: FLAGS is not a module-level name unless the module is run as a script (it is
  # assigned only in the script block), so calling main from an importing module would fail with
  # a name error.
  import_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir, FLAGS.tag_set)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      required=True,
      help="The directory containing the SavedModel to import.")
  parser.add_argument(
      "--log_dir",
      type=str,
      default="",
      required=True,
      help="The location for the Tensorboard log to begin visualization from.")
  parser.add_argument(
      "--tag_set",
      type=str,
      default="serve",
      required=False,
      help='tag-set of graph in SavedModel to load, separated by \',\'')
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
