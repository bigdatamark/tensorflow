# [MODULE DOC]
# Benchmark module for CSV parsing in tf.data: compares decoding CSV lines with a map over a text
# line dataset against the fused experimental CSV dataset reader, for float and string columns and
# for 4, 64 and 256 column files. Writes temporary CSV files, then deletes them. Not a library
# module; run directly it starts the benchmark runner. No import-time side effects.
#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for `tf.data.experimental.CsvDataset`."""

import os
import string
import tempfile

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest


class CsvDatasetBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite for CSV reading. Class constants give a float text value and a long string
  # value used to fill cells. Instance state set by the set-up helper: a temporary directory, the
  # list of column counts, elements per iteration, and the generated file names. Four benchmarks
  # cover map-with-decode versus fused CSV dataset, each for floats and for strings.
  """Benchmarks for `tf.data.experimental.CsvDataset`."""

  FLOAT_VAL = '1.23456E12'
  STR_VAL = string.ascii_letters * 10

  def _set_up(self, str_val):
    # [FUNCTION DOC]
    # HELPER: fixture creation for the benchmarks (this class is not a test case, so no automatic
    # set-up). Creates the temp directory, then writes one CSV file per column count (4, 64, 256)
    # holding 100 identical rows whose cells all equal str_val (text). Sets the temp dir, column
    # counts, per-iteration element count (5000) and file name list on the instance. Returns nothing.
    # Since this isn't test.TestCase, have to manually create a test dir
    gfile.MakeDirs(googletest.GetTempDir())
    self._temp_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())

    self._num_cols = [4, 64, 256]
    self._num_per_iter = 5000
    self._filenames = []
    for n in self._num_cols:
      fn = os.path.join(self._temp_dir, 'file%d.csv' % n)
      with open(fn, 'w') as f:
        # Just write 100 rows and use `repeat`... Assumes the cost
        # of creating an iterator is not significant
        row = ','.join(str_val for _ in range(n))
        f.write('\n'.join(row for _ in range(100)))
      self._filenames.append(fn)

  def _tear_down(self):
    # [FUNCTION DOC]
    # HELPER: removes the temporary directory created by the set-up helper, recursively. Returns
    # nothing.
    gfile.DeleteRecursively(self._temp_dir)

  def _run_benchmark(self, dataset, num_cols, prefix, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: shared runner. Runs and reports a benchmark of the given dataset: warmup, 10
    # iterations, 5000 elements each. The report name is the prefix plus the column count; extras
    # carry a model name from benchmark_id (int) and the column count as parameter. Returns nothing.

    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=self._num_per_iter,
        name='%s_with_cols_%d' % (prefix, num_cols),
        iters=10,
        extras={
            'model_name': 'csv.benchmark.%d' % benchmark_id,
            'parameters': '%d' % num_cols,
        },
        warmup=True)

  def benchmark_map_with_floats(self):
    # [FUNCTION DOC]
    # Benchmark: for each generated float-valued CSV file, reads lines with a repeating text line
    # dataset and parses each line with the decode-CSV op inside a map, defaults being float zero per
    # column. Creates and removes the temp files. Returns nothing.
    self._set_up(self.FLOAT_VAL)
    for i in range(len(self._filenames)):
      num_cols = self._num_cols[i]
      kwargs = {'record_defaults': [[0.0]] * num_cols}
      dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
      dataset = dataset.map(lambda l: parsing_ops.decode_csv(l, **kwargs))  # pylint: disable=cell-var-from-loop
      self._run_benchmark(
          dataset=dataset,
          num_cols=num_cols,
          prefix='csv_float_map_decode_csv',
          benchmark_id=1)
    self._tear_down()

  def benchmark_map_with_strings(self):
    # [FUNCTION DOC]
    # Benchmark: same as the float map benchmark but with long string cells and empty-string
    # defaults, parsing each line with the decode-CSV op inside a map. Creates and removes the temp
    # files. Returns nothing.
    self._set_up(self.STR_VAL)
    for i in range(len(self._filenames)):
      num_cols = self._num_cols[i]
      kwargs = {'record_defaults': [['']] * num_cols}
      dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
      dataset = dataset.map(lambda l: parsing_ops.decode_csv(l, **kwargs))  # pylint: disable=cell-var-from-loop
      self._run_benchmark(
          dataset=dataset,
          num_cols=num_cols,
          prefix='csv_strings_map_decode_csv',
          benchmark_id=2)
    self._tear_down()

  def benchmark_csv_dataset_with_floats(self):
    # [FUNCTION DOC]
    # Benchmark: reads each float-valued CSV file with the fused experimental CSV dataset (repeated),
    # float zero defaults per column. Creates and removes the temp files. Returns nothing.
    # KNOWN ISSUE: a text line dataset is also built first in each loop pass but is immediately
    # replaced by the CSV dataset, so it is dead code with no effect on the measurement.
    self._set_up(self.FLOAT_VAL)
    for i in range(len(self._filenames)):
      num_cols = self._num_cols[i]
      kwargs = {'record_defaults': [[0.0]] * num_cols}
      dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
      dataset = readers.CsvDataset(self._filenames[i], **kwargs).repeat()  # pylint: disable=cell-var-from-loop
      self._run_benchmark(
          dataset=dataset,
          num_cols=num_cols,
          prefix='csv_float_fused_dataset',
          benchmark_id=3)
    self._tear_down()

  def benchmark_csv_dataset_with_strings(self):
    # [FUNCTION DOC]
    # Benchmark: reads each string-valued CSV file with the fused experimental CSV dataset
    # (repeated), empty-string defaults per column. Creates and removes the temp files. Returns
    # nothing.
    # KNOWN ISSUE: a text line dataset is also built first in each loop pass but is immediately
    # replaced by the CSV dataset, so it is dead code with no effect on the measurement.
    self._set_up(self.STR_VAL)
    for i in range(len(self._filenames)):
      num_cols = self._num_cols[i]
      kwargs = {'record_defaults': [['']] * num_cols}
      dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
      dataset = readers.CsvDataset(self._filenames[i], **kwargs).repeat()  # pylint: disable=cell-var-from-loop
      self._run_benchmark(
          dataset=dataset,
          num_cols=num_cols,
          prefix='csv_strings_fused_dataset',
          benchmark_id=4)
    self._tear_down()


if __name__ == '__main__':
  benchmark_base.test.main()
