# [MODULE DOC]
# Command-line tool that prints selected metrics from an NVIDIA Nsight Compute report file
# (ncu-rep). It runs the ncu binary (found on the path, otherwise a default CUDA install
# location) to export the report as CSV in raw page format, parses the table, optionally filters
# kernels, aggregates the requested metrics across the selected kernels and prints them as
# markdown, CSV, JSON or raw text. The parsing, filtering and formatting logic lives in the
# companion library module so it can be unit tested.
# Import-time side effects: defines absl command-line flags (metric names, output format, kernel
# filter, list kernels, list metrics), resolves the ncu binary path into a module-level variable
# and logs it at info level.
# Entry point: the main function, run through the absl application runner when executed.
# Copyright 2025 The OpenXLA Authors.
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
"""Print metrics from ncu-rep file.

Usage:
  ncu_rep <ncu-rep-file> [--list_kernels] [--list_metrics]
    [--filter <filter kernels>]
    [-f <format>] [-k <kernel name>]
    [-m metric1] [-m metric2]f metrics: print all metric names
"""

from collections.abc import Sequence
import csv
import logging
import os
import shutil
import subprocess
import sys
from absl import app
from absl import flags
from xla.backends.gpu.codegen.tools import ncu_rep_lib

_METRICS = flags.DEFINE_multi_string(
    "m",
    [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.max",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "launch__registers_per_thread",
    ],
    "Names of metrics to print",
)
_FORMAT = flags.DEFINE_enum(
    "f",
    "md",
    ["md", "csv", "json", "raw"],
    "Output format: md (default), csv, json or plain text",
)
_KERNEL_FILTER = flags.DEFINE_list(
    "filter",
    None,
    "kernel filter: comma-separated list of kernel predicates:\n-"
    " 'name:<regex>' matches part of the name, case sensitive. Use"
    " 'name:^<regex>$' to match the whole string\n- 'id:<number>' - numeric"
    " kernel id\n- 'after:<matcher>' - match all kernels after the all matces"
    " of the matcher\nFilters are applied in order they are specified",
)
_LIST_KERNELS = flags.DEFINE_bool(
    "list_kernels", None, "print kernel names and exit", required=False
)
_LIST_METRICS = flags.DEFINE_bool(
    "list_metrics", None, "print metric names and exit", required=False
)

ncu_bin = shutil.which("ncu")
if not ncu_bin:
  ncu_bin = "/usr/local/cuda/bin/ncu"
logging.info("ncu binary: %s", ncu_bin)


def main(argv: Sequence[str]) -> None:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the usage text in the module docstring is garbled (it runs the metric option
  # into the words about printing all metric names).
  # KNOWN ISSUE: builds a mapping from column names to indexes that is never used, and leaves a
  # commented-out alternative locale line behind.
  # Entry point of the tool. Input: argv, the command-line arguments after flag parsing; it must
  # contain exactly the program name and one report file path, otherwise a usage error is raised;
  # a usage error is also raised if the file does not exist or if filters match no kernel.
  # Runs the ncu binary on the file with CSV output, base units and the raw page, forcing the
  # locale to US English UTF-8 in the child environment for consistent number formatting; a
  # failing ncu process raises a called-process error. Parses the CSV rows; if the list-metrics
  # flag is set prints every column name and returns. Otherwise extracts per-kernel metrics,
  # applies each kernel filter in order, and, if the list-kernels flag is set, prints the id and
  # name of each remaining kernel and returns. If several kernels remain, writes an
  # aggregation notice to standard error. Aggregates the requested metrics and writes them to
  # standard output in the chosen format (csv, json, raw, or markdown by default).
  # Returns nothing.
  if len(argv) != 2:
    raise app.UsageError("provide .ncu-rep file path")
  input_file_name = argv[1]
  if not os.path.exists(input_file_name):
    raise app.UsageError(f"file '{input_file_name}' does not exist")
  cmd = [
      ncu_bin,
      "-i",
      input_file_name,
      "--csv",
      "--print-units",
      "base",
      "--page",
      "raw",
  ]
  env_with_locale = os.environ.copy()
  # Force locale to en_US.UTF-8 to get consistent output.
  env_with_locale["LC_ALL"] = "en_US.UTF-8"
  # env_with_locale["LC_ALL"] = "de_DE.UTF-8"
  out = subprocess.check_output(cmd, text=True, env=env_with_locale).strip()
  rows = list(csv.reader(out.splitlines()))
  name_index = {}
  for i, name in enumerate(rows[0]):
    name_index[name] = i

  if _LIST_METRICS.value:
    for name in rows[0]:
      print(name)
    return

  all_kernels = ncu_rep_lib.get_metrics_by_kernel(rows)
  filtered_kernels = all_kernels
  for f in _KERNEL_FILTER.value or []:
    filtered_kernels = ncu_rep_lib.filter_kernels(filtered_kernels, f)
  if not filtered_kernels:
    raise app.UsageError(
        "No kernels matched the filter, use --list_kernels without --filter to"
        " see all kernels"
    )

  if _LIST_KERNELS.value:
    for row in filtered_kernels:
      print(
          row[ncu_rep_lib.KERNEL_ID_FIELD][0],
          row[ncu_rep_lib.KERNEL_NAME_FIELD][0],
      )
    return

  if len(filtered_kernels) > 1:
    sys.stderr.write(
        f"aggregating {len(filtered_kernels)} kernels\n",
    )

  metrics = ncu_rep_lib.aggregate_kernel_metrics(
      _METRICS.value, filtered_kernels
  )

  fmt = _FORMAT.value
  if fmt == "csv":
    ncu_rep_lib.write_metrics_csv(sys.stdout, metrics)
  elif fmt == "json":
    ncu_rep_lib.write_metrics_json(sys.stdout, metrics)
  elif fmt == "raw":
    ncu_rep_lib.write_metrics_raw(sys.stdout, metrics)
  else:
    ncu_rep_lib.write_metrics_markdown(sys.stdout, metrics)


if __name__ == "__main__":
  app.run(main)
