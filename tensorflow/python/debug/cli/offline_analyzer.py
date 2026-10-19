# [MODULE DOC]
# Command-line entry point for offline analysis of a tfdbg dump directory. When run as a script it
# parses flags (dump directory, usage logging, UI type, graph validation), loads the dump, creates
# the analyzer command-line UI with a tensor filter for infinities or NaNs, and runs it with the
# list-tensors command as the initial command. Importing defines only the main function; the flag
# parser is created only under the script guard.
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
"""Offline dump analyzer of TensorFlow Debugger (tfdbg)."""
import argparse
import sys

from absl import app

from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.lib import debug_data


def main(_):
  # [FUNCTION DOC]
  # KNOWN ISSUE: reads a module-level flags object that is only assigned in the script guard at
  # the bottom of the file, so calling this when the module is imported raises a name error.
  # Application main. The single argument (leftover command-line args) is ignored. If the dump
  # directory flag is empty it prints an error to standard error and exits with status 1.
  # Otherwise it prints the directory, loads the dump (validating against graph definitions per
  # the validate flag), builds the analyzer UI of the requested type with an inf-or-nan tensor
  # filter, and runs it with a title naming the directory and the list-tensors command first.
  # Returns nothing; side effects are console output and the interactive UI session.
  if not FLAGS.dump_dir:
    print("ERROR: dump_dir flag is empty.", file=sys.stderr)
    sys.exit(1)

  print("tfdbg offline: FLAGS.dump_dir = %s" % FLAGS.dump_dir)

  debug_dump = debug_data.DebugDumpDir(
      FLAGS.dump_dir, validate=FLAGS.validate_graph)
  cli = analyzer_cli.create_analyzer_ui(
      debug_dump,
      tensor_filters={"has_inf_or_nan": debug_data.has_inf_or_nan},
      ui_type=FLAGS.ui_type)

  title = "tfdbg offline @ %s" % FLAGS.dump_dir
  cli.run_ui(title=title, title_color="black_on_white", init_command="lt")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--dump_dir", type=str, default="", help="tfdbg dump directory to load")
  parser.add_argument(
      "--log_usage",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="Whether the usage of this tool is to be logged")
  parser.add_argument(
      "--ui_type",
      type=str,
      default="readline",
      help="Command-line user interface type (only readline is supported)")
  parser.add_argument(
      "--validate_graph",
      nargs="?",
      const=True,
      type="bool",
      default=True,
      help="""\
      Whether the dumped tensors will be validated against the GraphDefs\
      """)
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
