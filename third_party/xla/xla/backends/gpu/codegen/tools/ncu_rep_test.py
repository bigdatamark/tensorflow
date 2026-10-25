# [MODULE DOC]
# Unit tests for the ncu_rep_lib library used by the Nsight Compute report tool: conversion of
# the CSV table into per-kernel metric mappings, metric aggregation, kernel filtering and the
# markdown, CSV and raw output writers. Uses small in-memory data and string streams; no
# import-time side effects.
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

import io
from absl.testing import absltest
from xla.backends.gpu.codegen.tools import ncu_rep_lib


class NcuRepTest(absltest.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: there is no test for the JSON writer, nor for the error paths (unsupported
  # filter, missing metric, unit mismatch).
  # Test case for the ncu_rep_lib functions. Stateless; each test builds its own data.

  def test_get_metrics_by_kernel(self):
    # [FUNCTION DOC]
    # Test: converts a small table (header row, units row, two kernel rows) and asserts the
    # result is a list of two mappings from column name to a (value, unit) pair.
    # That is a typical format of ncu-rep CSV output.
    by_kernel = ncu_rep_lib.get_metrics_by_kernel([
        ["Kernel Name", "Metric 1", "Metric 2"],
        ["", "s", "Gb"],
        ["kernel1", "1", "2"],
        ["kernel2", "3", "4"],
    ])
    self.assertEqual(
        by_kernel,
        [
            {
                "Kernel Name": ("kernel1", ""),
                "Metric 1": ("1", "s"),
                "Metric 2": ("2", "Gb"),
            },
            {
                "Kernel Name": ("kernel2", ""),
                "Metric 1": ("3", "s"),
                "Metric 2": ("4", "Gb"),
            },
        ],
    )

  def test_aggregate_kernel_metrics(self):
    # [FUNCTION DOC]
    # Test: aggregates four metrics over two kernels. Checks sum (with thousands separators and a
    # fractional result rounded to two decimals), max, min and first-value behavior, and that
    # unrequested metrics are left out and whole numbers print without decimals.
    data = [
        {
            "ID": ("1", ""),
            "Kernel Name": ("kernel1", ""),
            "a.sum": ("12,345", "s"),
            "b.max": ("2", "registers"),
            "c.min": ("3", "b"),
            "d": ("4", "b"),
            "e": ("10", "b"),
        },
        {
            "ID": ("2", ""),
            "Kernel Name": ("kernel2", ""),
            "a.sum": ("345,678.1", "s"),
            "b.max": ("4", "registers"),
            "c.min": ("5.0", "b"),
            "d": ("6", "b"),
            "e": ("11", "b"),
        },
    ]
    self.assertEqual(
        ncu_rep_lib.aggregate_kernel_metrics(
            ["a.sum", "b.max", "c.min", "d"], data
        ),
        [
            ["a.sum", "358023.1", "s"],
            ["b.max", "4", "registers"],
            ["c.min", "3", "b"],
            ["d", "4", "b"],
        ],
    )

  def test_filter_kernels(self):
    # [FUNCTION DOC]
    # Test: filters two kernels by id, by anchored and unanchored name patterns, by a pattern
    # matching all kernels, and by the after form combined with an id filter (which keeps only the
    # kernel following the matched one). Asserts the expected sublists.
    data = [
        {
            "ID": ("1", ""),
            "Kernel Name": ("kernel1", ""),
            "a.sum": ("1,000.0", "s"),
            "b.max": ("2", "registers"),
            "c.min": ("3", "b"),
            "d": ("4", "b"),
            "e": ("10", "b"),
        },
        {
            "ID": ("2", ""),
            "Kernel Name": ("kernel2", ""),
            "a.sum": ("3.0", "s"),
            "b.max": ("4", "registers"),
            "c.min": ("5.0", "b"),
            "d": ("6", "b"),
            "e": ("11", "b"),
        },
    ]
    self.assertEqual(ncu_rep_lib.filter_kernels(data, "id:1"), [data[0]])
    self.assertEqual(ncu_rep_lib.filter_kernels(data, "name:^k.*2$"), [data[1]])
    self.assertEqual(ncu_rep_lib.filter_kernels(data, "name:2"), [data[1]])
    self.assertEqual(ncu_rep_lib.filter_kernels(data, "name:kernel"), data)
    self.assertEqual(ncu_rep_lib.filter_kernels(data, "after:id:1"), [data[1]])

  def test_write_metrics_markdown(self):
    # [FUNCTION DOC]
    # Test: writes two metrics with different name, value and unit widths to a string stream and
    # asserts the exact padded markdown-style table text (header, separator, two rows).
    with io.StringIO() as f:
      ncu_rep_lib.write_metrics_markdown(
          f,
          [
              ["Long Metric 1", "1.0000000000", "s"],
              ["Metric 2", "2", "Long Unit"],
          ],
      )
      self.assertEqual(
          f.getvalue(),
          """Metric        |        Value | Unit
--------------|--------------|----------
Long Metric 1 | 1.0000000000 | s
Metric 2      |            2 | Long Unit
""",
      )

  def test_write_metrics_csv(self):
    # [FUNCTION DOC]
    # Test: writes two metrics to a string stream and asserts the exact CSV text: header line
    # followed by one line per metric.
    with io.StringIO() as f:
      ncu_rep_lib.write_metrics_csv(
          f,
          [
              ["Long Metric 1", "1.0000000000", "s"],
              ["Metric 2", "2", "Long Unit"],
          ],
      )
      self.assertEqual(
          f.getvalue(),
          """metric,value,unit
Long Metric 1,1.0000000000,s
Metric 2,2,Long Unit
""",
      )

  def test_write_metrics_raw(self):
    # [FUNCTION DOC]
    # Test: writes two metrics to a string stream and asserts the exact raw text: one line per
    # metric with the value then the unit and no names.
    with io.StringIO() as f:
      ncu_rep_lib.write_metrics_raw(
          f,
          [
              ["Long Metric 1", "1.0000000000", "s"],
              ["Metric 2", "2", "Long Unit"],
          ],
      )
      self.assertEqual(
          f.getvalue(),
          """1.0000000000 s
2 Long Unit
""",
      )


if __name__ == "__main__":
  absltest.main()
