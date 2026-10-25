# [MODULE DOC]
# HELPER: library behind the ncu_rep command-line tool, split out so it can be unit tested.
# KNOWN ISSUE: the original docstring refers to a file named ncu_rep_main.py, but the tool is
# in the file ncu_rep.py.
# Works on the table (list of rows of strings) obtained from the CSV export of a Nsight Compute
# report: the first row holds column names, the second row units, and the rest one row per kernel.
# Provides converting the table into per-kernel metric mappings, aggregating metrics over several
# kernels (max, min, sum or first value depending on the metric name suffix), filtering kernels
# by id, name pattern or position, and writing metrics as markdown, CSV, JSON or raw text.
# Defines the names of the kernel name and id columns. No import-time side effects.
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
"""Library to work with ncu-rep CSV output.

Separate library from ncu_rep_main.py to enable unit tests.
"""

import csv
import itertools
import json
import logging
import re
from typing import TextIO
from absl import app

KERNEL_NAME_FIELD = "Kernel Name"
KERNEL_ID_FIELD = "ID"


def get_metrics_by_kernel(
    rows: list[list[str]],
) -> list[dict[str, tuple[str, str]]]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a dictionary of metrics by kernel, but it returns
  # a list with one mapping per kernel. It also builds a name-to-index mapping that is never used.
  # Input: rows, a list of string lists; row 0 is the column names, row 1 the units, and each
  # following row one kernel. Returns a list with one mapping per kernel row, from column name to
  # a pair of the cell value and the unit of that column. Raises an index error if the table has
  # fewer than two rows or rows shorter than the header. No side effects.
  """Converts ncu-rep table to a dictionary of metrics by kernel.

  Args:
    rows: ncu-rep table rows

  Returns:
    dictionary of metrics by kernel
  """
  name_index = {}
  units = rows[1]
  for i, name in enumerate(rows[0]):
    name_index[name] = i
  results = []
  for kernel in rows[2:]:
    values = {}
    for idx, name in enumerate(rows[0]):
      values[name] = (kernel[idx], units[idx])
    results.append(values)
  return results


def aggregate(metric_values: list[float], metric_name: str):
  # [FUNCTION DOC]
  # HELPER: combines the per-kernel values of one metric for aggregate_kernel_metrics.
  # Inputs: metric_values, a non-empty list of floats; metric_name, the metric name. If the name
  # ends with the max suffix returns the maximum, with the min suffix the minimum, with the sum
  # suffix the total; any other name returns the first value. An empty list raises a value error
  # (or an index error for the first-value case). No side effects.
  """Aggregates metric values using a function based on the metric name.

  If metric name does not match any of the known patterns, the first value from
  the input list is returned.

  Args:
    metric_values: list of metric values, floats
    metric_name: metric name

  Returns:
    aggregated metric value
  """
  if str.endswith(metric_name, ".max"):
    return max(metric_values)
  if str.endswith(metric_name, ".min"):
    return min(metric_values)
  if str.endswith(metric_name, ".sum"):
    return sum(metric_values)
  return metric_values[0]


def aggregate_kernel_metrics(
    metrics: list[str], kernel_metrics: list[dict[str, tuple[str, str]]]
) -> list[list[str]]:
  # [FUNCTION DOC]
  # Inputs: metrics, the names of metrics to report; kernel_metrics, a list of per-kernel
  # mappings from metric name to a (value text, unit) pair. Raises a usage error if the kernel
  # list is empty, a requested metric is missing in some kernel, or the unit of a metric differs
  # between kernels. Values are parsed as floats after removing thousands separators (commas).
  # Each metric's values are combined by the aggregate helper (max, min, sum or first value by
  # name suffix). Returns a list of three-element rows (name, value text, unit), in order of first
  # appearance of the requested metrics; values that are whole numbers are printed as integers,
  # others rounded to two decimals. Duplicate names in metrics are collected once per kernel.
  # The parameter kernel_metrics is rebound locally to the output list; inputs are not mutated.
  """Aggregates and returns the metrics for the given kernels.

  Args:
    metrics: list of metrics names to print
    kernel_metrics: dictionary of metrics by kernel

  Returns:
    list of rows [name, value, unit] per metric.
  """
  if not kernel_metrics:
    raise app.UsageError("no metrics found")
  results: dict[str, tuple[list[float], str]] = {}  # name -> (float[], unit)
  for vals in kernel_metrics:
    for name in metrics:
      if name not in vals:
        raise app.UsageError(f"metric '{name}' is not found")
      value, unit = vals[name]
      if name not in results:
        results[name] = ([], unit)
      if results[name][1] != unit:
        # That should not happen with `--print-units base` but left as a
        # safety check.
        raise app.UsageError(f"unit mismatch for metric '{name}'")
      # Replace ',' in value to parse it as a float. It is printed in
      # en_US.UTF-8 locale but we don't want to change the runtime locale just
      # for this as that might affect downstream.
      results[name][0].append(float(value.replace(",", "")))
  kernel_metrics = []
  for name, (values, unit) in results.items():
    a = aggregate(values, name)
    if round(a) == a:
      kernel_metrics.append([name, f"{round(a)}", unit])
    else:
      kernel_metrics.append([name, f"{round(a, 2)}", unit])
  return kernel_metrics


def filter_kernels(
    kernels: list[dict[str, tuple[str, str]]], condition: str
) -> list[dict[str, tuple[str, str]]]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring's description of the supported expressions is run together (the
  # three forms are not separated); also the after filter compares ids as text and, when the
  # matching kernels are found, keeps everything following the last match.
  # Inputs: kernels, a list of per-kernel mappings (as made by the table conversion function);
  # condition, a text filter. Forms: "id:" followed by an id keeps kernels whose id text equals
  # it; "name:" followed by a regular expression keeps kernels whose name contains a match;
  # "after:" followed by another filter keeps the kernels that come after the last kernel
  # matched by that inner filter (if the inner filter matches nothing it logs a warning and
  # returns all kernels unchanged). Returns the filtered list. Raises a usage error for any other
  # prefix. A malformed regular expression raises a regex error.
  """Filters kernels by a condition.

  Args:
    kernels: list of kernel tuples, extracted from ncu-rep CSV
    condition: filter condition. Supported filter expressions: 'id:<value>' -
      kernel with an ID 'name:<regex>' - kernel with a name matching the regex
      'after:<filter>' - kernels after the last kernel matching the filter.

  Returns:
    matching kernels
  """
  if condition.startswith("id:"):
    i = condition.removeprefix("id:")
    return [v for v in kernels if v[KERNEL_ID_FIELD][0] == i]
  if condition.startswith("name:"):
    r = condition.removeprefix("name:")
    return [v for v in kernels if re.search(r, v[KERNEL_NAME_FIELD][0])]
  if condition.startswith("after:"):
    r = condition.removeprefix("after:")
    sub = filter_kernels(kernels, r)
    if not sub:
      logging.warning("no kernels matched '%s', 'after:' has no effect", r)
      return kernels
    after_id = sub[-1][KERNEL_ID_FIELD][0]
    return list(
        itertools.dropwhile(
            lambda v: v[KERNEL_ID_FIELD][0] != after_id, kernels
        )
    )[1:]
  raise app.UsageError(f"unsupported filter: {condition}")


def write_metrics_markdown(out: TextIO, metrics: list[list[str]]):
  # [FUNCTION DOC]
  # Writes the metrics to the text stream out as a markdown-like table with a header line
  # (Metric, Value, Unit), a separator line and one line per metric, padding the name column on
  # the right and the value column on the left to the widest entries. Input metrics is a list of
  # three-element rows (name, value, unit). Raises a value error if metrics is empty. Returns
  # nothing; side effect is writing to out.
  """Formats metrics in markdown."""
  name_width = max(len(m[0]) for m in metrics)
  value_width = max(max(len(m[1]) for m in metrics), len("value"))
  unit_width = max(max(len(m[2]) for m in metrics), len("unit"))
  out.write(
      f"{'Metric'.ljust(name_width)} | {'Value'.rjust(value_width)} | Unit\n"
  )
  out.write("-" * name_width)
  out.write("-|-")
  out.write("-" * value_width)
  out.write("-|-")
  out.write("-" * unit_width + "\n")
  for name, value, unit in metrics:
    out.write(
        f"{name.ljust(name_width)} | {value.rjust(value_width)} | {unit}\n"
    )


def write_metrics_csv(out: TextIO, metrics: list[list[str]]):
  # [FUNCTION DOC]
  # Writes the metrics to the text stream out as CSV with newline line endings: a header row
  # (metric, value, unit) followed by each (name, value, unit) row of metrics. Returns nothing;
  # side effect is writing to out.
  """Formats metrics in csv."""
  writer = csv.writer(out, lineterminator="\n")
  writer.writerow(["metric", "value", "unit"])
  writer.writerows(metrics)


def write_metrics_json(out: TextIO, metrics: list[list[str]]):
  # [FUNCTION DOC]
  # Writes the metrics to the text stream out as one JSON object mapping each metric name to an
  # object with its value text and unit, keys sorted, followed by a newline. A repeated metric
  # name keeps the last entry. Returns nothing; side effect is writing to out.
  """Formats metrics in JSON."""
  data = {}
  for name, value, unit in metrics:
    data[name] = {"value": value, "unit": unit}
  json.dump(data, out, sort_keys=True)
  out.write("\n")


def write_metrics_raw(out: TextIO, metrics: list[list[str]]):
  # [FUNCTION DOC]
  # Writes only the value and unit of each metric row to the text stream out, one line per
  # metric as the value, a space and the unit (names are omitted). Returns nothing; side effect
  # is writing to out.
  """Formats metrics in raw."""
  for _, value, unit in metrics:
    out.write(f"{value} {unit}\n")
