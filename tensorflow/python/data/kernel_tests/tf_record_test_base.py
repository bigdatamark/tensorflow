# [MODULE DOC]
# HELPER: helper module of base test classes for reader datasets, used by the TFRecord dataset
# tests and by experimental reader tests.
# Defines FeaturesTestBase (writes record files holding serialized example messages with file,
# record, keywords and label features, and checks batched-feature datasets against expected
# batches) and TFRecordTestBase (writes record files of simple text records and checks batches of
# raw records, including interleaving, final-batch dropping and parser-stripped records). No
# import-time side effects beyond imports.
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
"""Base class for testing reader datasets."""

import os

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat


class FeaturesTestBase(test_base.DatasetTestBase):
  # [CLASS DOC]
  # HELPER: base class for tests of batched feature datasets over record files. Holds the number
  # of files (two), records per file (seven) and the written file names (set in setUp), plus
  # helpers to build the dataset, compute expected batches and compare with the actual ones.
  # Subclasses are expected to set an attribute named outputs (a callable returning the next
  # batch) before using the actual-batch helpers.
  """Base class for testing TFRecord-based features."""

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent set-up, then sets two files with seven records each and writes them into the
    # test temp directory (side effect: creates files), keeping the names in an attribute.
    super(FeaturesTestBase, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self._filenames = self._createFiles()

  def make_batch_feature(self,
                         filenames,
                         num_epochs,
                         batch_size,
                         label_key=None,
                         reader_num_threads=1,
                         parser_num_threads=1,
                         shuffle=False,
                         shuffle_seed=None,
                         drop_final_batch=False):
    # [FUNCTION DOC]
    # HELPER: builds a batched features dataset reading record files. Inputs: filenames (file
    # pattern or list), num_epochs, batch_size, label_key (optional name of the feature to split
    # off as label), reader_num_threads and parser_num_threads (default one), shuffle (default
    # false), shuffle_seed (optional) and drop_final_batch (default false). Parses fixed-length
    # integer file and record features, a variable-length string keywords feature and a
    # fixed-length string label. Side effect: stores filenames, num_epochs and batch_size as
    # attributes on the test. Returns the dataset.
    self.filenames = filenames
    self.num_epochs = num_epochs
    self.batch_size = batch_size

    return readers.make_batched_features_dataset(
        file_pattern=self.filenames,
        batch_size=self.batch_size,
        features={
            "file": parsing_ops.FixedLenFeature([], dtypes.int64),
            "record": parsing_ops.FixedLenFeature([], dtypes.int64),
            "keywords": parsing_ops.VarLenFeature(dtypes.string),
            "label": parsing_ops.FixedLenFeature([], dtypes.string),
        },
        label_key=label_key,
        reader=core_readers.TFRecordDataset,
        num_epochs=self.num_epochs,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        reader_num_threads=reader_num_threads,
        parser_num_threads=parser_num_threads,
        drop_final_batch=drop_final_batch)

  def _record(self, f, r, l):
    # [FUNCTION DOC]
    # HELPER: builds one serialized example message for file number f, record number r and label
    # text l, with file and record integer features, a keywords feature from the keyword helper and
    # a label bytes feature. Returns the serialized bytes.
    example = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                "file":
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[f])),
                "record":
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[r])),
                "keywords":
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=self._get_keywords(f, r))),
                "label":
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[compat.as_bytes(l)]))
            }))
    return example.SerializeToString()

  def _get_keywords(self, f, r):
    # [FUNCTION DOC]
    # HELPER: returns the list of keyword byte strings for file f and record r: one keyword if the
    # sum of the indices is even, two if odd, named keyword0 and keyword1.
    num_keywords = 1 + (f + r) % 2
    keywords = []
    for index in range(num_keywords):
      keywords.append(compat.as_bytes("keyword%d" % index))
    return keywords

  def _sum_keywords(self, num_files):
    # [FUNCTION DOC]
    # HELPER: returns the total number of keywords across the first num_files files and all their
    # records, following the same one-or-two rule as the keyword helper.
    sum_keywords = 0
    for i in range(num_files):
      for j in range(self._num_records):
        sum_keywords += 1 + (i + j) % 2
    return sum_keywords

  def _createFiles(self):
    # [FUNCTION DOC]
    # HELPER: writes the fixture record files, one per configured file, into the test temp
    # directory (side effect: creates or overwrites files), each with the configured number of
    # serialized example records labeled with a fixed fake label. Returns the list of file paths.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for j in range(self._num_records):
        writer.write(self._record(i, j, "fake-label"))
      writer.close()
    return filenames

  def _run_actual_batch(self, outputs, label_key_provided=False):
    # [FUNCTION DOC]
    # HELPER: fetches one batch by evaluating the callable outputs. If label_key_provided (default
    # false) the result is a pair of feature mapping and label; otherwise the label is taken from
    # the label feature of the mapping. Returns a list of: file values, keyword sparse indices,
    # keyword sparse values, keyword dense shape, record values and labels.
    if label_key_provided:
      # outputs would be a tuple of (feature dict, label)
      features, label = self.evaluate(outputs())
    else:
      features = self.evaluate(outputs())
      label = features["label"]
    file_out = features["file"]
    keywords_indices = features["keywords"].indices
    keywords_values = features["keywords"].values
    keywords_dense_shape = features["keywords"].dense_shape
    record = features["record"]
    return ([
        file_out, keywords_indices, keywords_values, keywords_dense_shape,
        record, label
    ])

  def _next_actual_batch(self, label_key_provided=False):
    # [FUNCTION DOC]
    # HELPER: fetches the next actual batch from the attribute named outputs (which the test must
    # have set) using the batch-fetching helper; label_key_provided defaults to false. Returns the
    # same list as that helper.
    return self._run_actual_batch(self.outputs, label_key_provided)

  def _interleave(self, iterators, cycle_length):
    # [FUNCTION DOC]
    # HELPER: generator that mimics cycling through input iterators. Opens up to cycle_length
    # iterators from the list iterators, then repeatedly takes one item from each open iterator in
    # turn; when one is exhausted it is replaced by the next pending iterator, or closed if none
    # remain. Yields the items until all are exhausted. Side effect: removes entries from the
    # passed list.
    pending_iterators = iterators
    open_iterators = []
    num_open = 0
    for i in range(cycle_length):
      if pending_iterators:
        open_iterators.append(pending_iterators.pop(0))
        num_open += 1

    while num_open:
      for i in range(min(cycle_length, len(open_iterators))):
        if open_iterators[i] is None:
          continue
        try:
          yield next(open_iterators[i])
        except StopIteration:
          if pending_iterators:
            open_iterators[i] = pending_iterators.pop(0)
          else:
            open_iterators[i] = None
            num_open -= 1

  def _next_expected_batch(self,
                           file_indices,
                           batch_size,
                           num_epochs,
                           cycle_length=1):
    # [FUNCTION DOC]
    # HELPER: generator computing the expected batches. Inputs: file_indices (which files to
    # read), batch_size, num_epochs, cycle_length (default one; above one interleaves files).
    # Walks the records of the files for each epoch, accumulating per-batch lists of file numbers,
    # keyword sparse indices and values, record numbers and labels. Yields lists of: files,
    # keyword indices, keyword values, keyword dense shape (batch size and longest keyword list),
    # records, labels; a final smaller batch is yielded if records remain.

    def _next_record(file_indices):
      # [FUNCTION DOC]
      # HELPER: generator yielding, for each file index in order and each record of that file, a
      # triple of file number, record number and the fixed fake label bytes.
      for j in file_indices:
        for i in range(self._num_records):
          yield j, i, compat.as_bytes("fake-label")

    def _next_record_interleaved(file_indices, cycle_length):
      # [FUNCTION DOC]
      # HELPER: returns a generator of the same triples as the plain record generator, but
      # interleaved across the files with the given cycle length.
      return self._interleave([_next_record([i]) for i in file_indices],
                              cycle_length)

    file_batch = []
    keywords_batch_indices = []
    keywords_batch_values = []
    keywords_batch_max_len = 0
    record_batch = []
    batch_index = 0
    label_batch = []
    for _ in range(num_epochs):
      if cycle_length == 1:
        next_records = _next_record(file_indices)
      else:
        next_records = _next_record_interleaved(file_indices, cycle_length)
      for record in next_records:
        f = record[0]
        r = record[1]
        label_batch.append(record[2])
        file_batch.append(f)
        record_batch.append(r)
        keywords = self._get_keywords(f, r)
        keywords_batch_values.extend(keywords)
        keywords_batch_indices.extend(
            [[batch_index, i] for i in range(len(keywords))])
        batch_index += 1
        keywords_batch_max_len = max(keywords_batch_max_len, len(keywords))
        if len(file_batch) == batch_size:
          yield [
              file_batch, keywords_batch_indices, keywords_batch_values,
              [batch_size, keywords_batch_max_len], record_batch, label_batch
          ]
          file_batch = []
          keywords_batch_indices = []
          keywords_batch_values = []
          keywords_batch_max_len = 0
          record_batch = []
          batch_index = 0
          label_batch = []
    if file_batch:
      yield [
          file_batch, keywords_batch_indices, keywords_batch_values,
          [len(file_batch), keywords_batch_max_len], record_batch, label_batch
      ]

  def _verify_records(self,
                      batch_size,
                      file_index=None,
                      num_epochs=1,
                      label_key_provided=False,
                      interleave_cycle_length=1):
    # [FUNCTION DOC]
    # HELPER: compares actual batches with expected ones. Inputs: batch_size; file_index (optional
    # single file; all files when absent); num_epochs (default one); label_key_provided (default
    # false); interleave_cycle_length (default one). For every expected batch it fetches the next
    # actual batch and asserts each component is equal. Returns nothing; does not check that no
    # extra batches remain.
    if file_index is not None:
      file_indices = [file_index]
    else:
      file_indices = range(self._num_files)

    for expected_batch in self._next_expected_batch(
        file_indices,
        batch_size,
        num_epochs,
        cycle_length=interleave_cycle_length):
      actual_batch = self._next_actual_batch(
          label_key_provided=label_key_provided)
      for i in range(len(expected_batch)):
        self.assertAllEqual(expected_batch[i], actual_batch[i])


class TFRecordTestBase(test_base.DatasetTestBase):
  # [CLASS DOC]
  # HELPER: base class for TFRecord dataset tests. Holds the number of files (two), records per
  # file (seven) and the written file names (set in setUp), and provides helpers to write record
  # files of simple text records and to compare dataset batches with expected record batches.
  """Base class for TFRecord-based tests."""

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent set-up, then sets two files with seven records each and writes them into the
    # test temp directory (side effect: creates files), keeping the names in an attribute.
    super(TFRecordTestBase, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self._filenames = self._createFiles()

  def _interleave(self, iterators, cycle_length):
    # [FUNCTION DOC]
    # HELPER: generator that mimics cycling through input iterators. Opens up to cycle_length
    # iterators from the list iterators, then repeatedly takes one item from each open iterator in
    # turn; when one is exhausted it is replaced by the next pending iterator, or closed if none
    # remain. Yields the items until all are exhausted. Side effect: removes entries from the
    # passed list.
    pending_iterators = iterators
    open_iterators = []
    num_open = 0
    for i in range(cycle_length):
      if pending_iterators:
        open_iterators.append(pending_iterators.pop(0))
        num_open += 1

    while num_open:
      for i in range(min(cycle_length, len(open_iterators))):
        if open_iterators[i] is None:
          continue
        try:
          yield next(open_iterators[i])
        except StopIteration:
          if pending_iterators:
            open_iterators[i] = pending_iterators.pop(0)
          else:
            open_iterators[i] = None
            num_open -= 1

  def _next_expected_batch(self, file_indices, batch_size, num_epochs,
                           cycle_length, drop_final_batch, use_parser_fn):
    # [FUNCTION DOC]
    # HELPER: generator computing the expected batches of raw records. Inputs: file_indices,
    # batch_size, num_epochs, cycle_length (one means sequential, otherwise interleaved),
    # drop_final_batch (skip the last partial batch) and use_parser_fn (strip the first byte of
    # every record, imitating a parsing function). Yields lists of record byte strings of length
    # batch_size, plus a final shorter list when allowed.

    def _next_record(file_indices):
      # [FUNCTION DOC]
      # HELPER: generator yielding, for each file index in order and each record of that file, a
      # pair of file number and record number.
      for j in file_indices:
        for i in range(self._num_records):
          yield j, i

    def _next_record_interleaved(file_indices, cycle_length):
      # [FUNCTION DOC]
      # HELPER: returns a generator of the same pairs as the plain record generator, but
      # interleaved across the files with the given cycle length.
      return self._interleave([_next_record([i]) for i in file_indices],
                              cycle_length)

    record_batch = []
    batch_index = 0
    for _ in range(num_epochs):
      if cycle_length == 1:
        next_records = _next_record(file_indices)
      else:
        next_records = _next_record_interleaved(file_indices, cycle_length)
      for f, r in next_records:
        record = self._record(f, r)
        if use_parser_fn:
          record = record[1:]
        record_batch.append(record)
        batch_index += 1
        if len(record_batch) == batch_size:
          yield record_batch
          record_batch = []
          batch_index = 0
    if record_batch and not drop_final_batch:
      yield record_batch

  def _verify_records(self, outputs, batch_size, file_index, num_epochs,
                      interleave_cycle_length, drop_final_batch, use_parser_fn):
    # [FUNCTION DOC]
    # HELPER: compares dataset output with expected record batches. Inputs: outputs (callable
    # returning the next batch), batch_size, file_index (a single index, a list of indices, or
    # nothing for all files), num_epochs, interleave_cycle_length, drop_final_batch and
    # use_parser_fn. For every expected batch evaluates outputs and asserts equality. Returns
    # nothing; does not check that no extra batches remain.
    if file_index is not None:
      if isinstance(file_index, list):
        file_indices = file_index
      else:
        file_indices = [file_index]
    else:
      file_indices = range(self._num_files)

    for expected_batch in self._next_expected_batch(
        file_indices, batch_size, num_epochs, interleave_cycle_length,
        drop_final_batch, use_parser_fn):
      actual_batch = self.evaluate(outputs())
      self.assertAllEqual(expected_batch, actual_batch)

  def _record(self, f, r):
    # [FUNCTION DOC]
    # HELPER: returns the expected bytes of record r in file f, a text naming the record and file.
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _createFiles(self):
    # [FUNCTION DOC]
    # HELPER: writes the fixture record files into the test temp directory (side effect: creates
    # or overwrites files), one per configured file, each holding the configured number of simple
    # text records. Returns the list of file paths.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for j in range(self._num_records):
        writer.write(self._record(i, j))
      writer.close()
    return filenames

  def _writeFile(self, name, data):
    # [FUNCTION DOC]
    # HELPER: writes a record file named name in the test temp directory (side effect: creates or
    # overwrites it), with one record per item of data, each item converted to text and then
    # bytes. Returns the file path.
    filename = os.path.join(self.get_temp_dir(), name)
    writer = python_io.TFRecordWriter(filename)
    for d in data:
      writer.write(compat.as_bytes(str(d)))
    writer.close()
    return filename

