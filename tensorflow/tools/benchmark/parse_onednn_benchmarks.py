# [MODULE DOC]
# Command line script that summarizes the output log of the oneDNN benchmark shell script.
# It reads the log file named by the first command line argument, extracts the average runtime
# (microseconds) for each combination of model, batch size and oneDNN setting (off or on), and
# prints a comma separated table with columns model, batch, vanilla runtime, oneDNN runtime and
# speedup (vanilla divided by oneDNN), using a question mark where a result is unavailable.
# Module-level state: a results mapping keyed by (model, batch, oneDNN flag), a set of model names,
# a set of batch sizes and a two-value parser state enumeration. The parsing function fills these
# globals as a side effect. Importing has no side effects beyond creating them. Not a helper module.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Parses results from run_onednn_benchmarks.sh.

Example results:
Showing runtimes in microseconds. `?` means not available.
               Model,  Batch,        Vanilla,         oneDNN,    Speedup
          bert-large,      1,              x,              y,        x/y
          bert-large,     16,            ...,            ...,        ...
           inception,      1,            ...,            ...,        ...
           inception,     16,            ...,            ...,        ...
                                        ⋮
        ssd-resnet34,      1,              ?,            ...,          ?
        ssd-resnet34,     16,              ?,            ...,          ?

Vanilla TF can't run ssd-resnet34 on CPU because it doesn't support NCHW format.
"""

import enum
import re
import sys

db = dict()
models = set()
batch_sizes = set()
State = enum.Enum("State", "FIND_CONFIG_OR_MODEL FIND_RUNNING_TIME")


def parse_results(lines):
  # [FUNCTION DOC]
  # Runs a two-state line scanner over the log lines. In the find-config-or-model state, a line
  # announcing a batch size and oneDNN flag updates the current batch and flag (and records the
  # batch size); any other line must name a model graph file, whose name becomes the current model
  # (recorded in the models set; an assertion fails if no model name is found) and the scanner
  # moves to the find-running-time state. In that state a line with an average time after the
  # no-stats marker stores the float under (model, batch, oneDNN flag) in the global results
  # mapping (asserting none of these are unset); a line without it is re-examined in the first state
  # (the index is not advanced) because models that cannot run produce no result line.
  # Input: list of text lines. Raises a runtime error if the state is unknown. Returns nothing;
  # mutates the module globals for results, models and batch sizes.
  """Parses benchmark results from run_onednn_benchmarks.sh.

  Stores results in a global dict.

  Args:
    lines: Array of strings corresponding to each line of the output from
      run_onednn_benchmarks.sh

  Raises:
    RuntimeError: If the program reaches an unknown state.
  """
  idx = 0
  batch, onednn, model = None, None, None
  state = State.FIND_CONFIG_OR_MODEL
  while idx < len(lines):
    if state is State.FIND_CONFIG_OR_MODEL:
      config = re.match(
          r"\+ echo 'BATCH=(?P<batch>[\d]+), ONEDNN=(?P<onednn>[\d]+)",
          lines[idx])
      if config:
        batch = int(config.group("batch"))
        onednn = int(config.group("onednn"))
        batch_sizes.add(batch)
      else:
        model_re = re.search(r"tf-graphs\/(?P<model>[\w\d_-]+).pb", lines[idx])
        assert model_re
        model = model_re.group("model")
        models.add(model)
        state = State.FIND_RUNNING_TIME
    elif state is State.FIND_RUNNING_TIME:
      match = re.search(r"no stats: (?P<avg>[\d.]+)", lines[idx])
      state = State.FIND_CONFIG_OR_MODEL
      if match:
        avg = float(match.group("avg"))
        key = (model, batch, onednn)
        assert None not in key
        db[key] = avg
      else:
        # Some models such as ssd-resnet34 can't run on CPU with vanilla TF and
        # won't have results. This line contains either a config or model name.
        continue
    else:
      raise RuntimeError("Reached the unreachable code.")
    idx = idx + 1


def main():
  # [FUNCTION DOC]
  # Script entry point. Reads all lines of the file whose path is the first command line argument
  # (an index error if missing), parses them into the global results, then prints a header and one
  # line per sorted model and sorted batch size with the vanilla (flag 0) and oneDNN (flag 1)
  # runtimes, and their ratio with two decimals when both exist; missing values print as a question
  # mark. Returns nothing; writes to standard output.
  filename = sys.argv[1]
  with open(filename, "r") as f:
    lines = f.readlines()
  parse_results(lines)
  print("Showing runtimes in microseconds. `?` means not available.")
  print("%20s, %6s, %14s, %14s, %10s" %
        ("Model", "Batch", "Vanilla", "oneDNN", "Speedup"))
  for model in sorted(models):
    for batch in sorted(batch_sizes):
      key = (model, batch, 0)
      eigen = db[key] if key in db else "?"
      key = (model, batch, 1)
      onednn = db[key] if key in db else "?"
      speedup = "%10.2f" % (eigen / onednn) if "?" not in (eigen,
                                                           onednn) else "?"
      print("%20s, %6d, %14s, %14s, %10s" %
            (model, batch, str(eigen), str(onednn), speedup))


if __name__ == "__main__":
  main()
