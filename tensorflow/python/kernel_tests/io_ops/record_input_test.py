# [MODULE DOC]
# Test module for the RecordInput op (the multi-threaded, shuffling TFRecord batch reader in the
# data flow ops). The tests write small numbered TFRecord files into the test temp directory,
# build a record input yield operation over a file pattern and check the yielded records:
# plain, gzip and zlib compressed input, no duplicates within an epoch, repeated runs not
# deadlocking, an empty file match raising a not-found error, and invalid raw op parameters
# raising an invalid-argument error. Running the file as a script starts the TensorFlow test
# runner. No import-time side effects.
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
"""Tests for record_input_op."""

import os

from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class RecordInputOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the RecordInput op. Holds no state of its own; each test generates its
  # own numbered TFRecord files in the test temp directory through a shared data generator and
  # reads them back with a RecordInput yield operation in a cached session.

  def generateTestData(self,
                       prefix,
                       n,
                       m,
                       compression_type=tf_record.TFRecordCompressionType.NONE):
    # [FUNCTION DOC]
    # HELPER: writes the input files used by the RecordInput tests.
    # KNOWN ISSUE: the writer is closed only once, after the file loop (the close call is
    # dedented out of the loop), so only the last file's writer is explicitly closed and earlier
    # writers rely on garbage collection to flush; with zero files the writer name would be
    # undefined and an error would be raised.
    # Inputs: prefix is the file name prefix; n the number of files; m the number of records
    # per file; compression_type the TFRecord compression type (default none). Each file is
    # named prefix, a dot and its index, in the test temp directory, and holds m records that
    # are the global record index (file index times m plus record index) zero padded to ten
    # digits and encoded as bytes. Returns nothing.
    options = tf_record.TFRecordOptions(compression_type)
    for i in range(n):
      f = os.path.join(self.get_temp_dir(), prefix + "." + str(i))
      w = tf_record.TFRecordWriter(f, options=options)

      for j in range(m):
        w.write("{0:0{width}}".format(i * m + j, width=10).encode("utf-8"))

    w.close()

  def testRecordInputSimple(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the session handle bound in the context manager is unused.
    # Scenario: one file with one record, read with a record input op of parallelism, buffer and
    # batch size one. Asserts the single yielded value equals the ten digit zero record.
    with self.cached_session() as sess:
      self.generateTestData("basic", 1, 1)

      yield_op = data_flow_ops.RecordInput(
          file_pattern=os.path.join(self.get_temp_dir(), "basic.*"),
          parallelism=1,
          buffer_size=1,
          batch_size=1,
          name="record_input").get_yield_op()

      self.assertEqual(self.evaluate(yield_op), b"0000000000")

  def testRecordInputSimpleGzip(self):
    # [FUNCTION DOC]
    # Scenario: same as the simple test but the file is written with gzip compression and the
    # record input op is told to use gzip. Asserts the yielded value is the expected record.
    with self.cached_session() as sess:
      self.generateTestData(
          "basic",
          1,
          1,
          compression_type=tf_record.TFRecordCompressionType.GZIP)

      yield_op = data_flow_ops.RecordInput(
          file_pattern=os.path.join(self.get_temp_dir(), "basic.*"),
          parallelism=1,
          buffer_size=1,
          batch_size=1,
          name="record_input",
          compression_type=tf_record.TFRecordCompressionType.GZIP).get_yield_op(
          )

      self.assertEqual(self.evaluate(yield_op), b"0000000000")

  def testRecordInputSimpleZlib(self):
    # [FUNCTION DOC]
    # Scenario: same as the simple test but with zlib compression on both file and op. Asserts
    # the yielded value is the expected record.
    with self.cached_session() as sess:
      self.generateTestData(
          "basic",
          1,
          1,
          compression_type=tf_record.TFRecordCompressionType.ZLIB)

      yield_op = data_flow_ops.RecordInput(
          file_pattern=os.path.join(self.get_temp_dir(), "basic.*"),
          parallelism=1,
          buffer_size=1,
          batch_size=1,
          name="record_input",
          compression_type=tf_record.TFRecordCompressionType.ZLIB).get_yield_op(
          )

      self.assertEqual(self.evaluate(yield_op), b"0000000000")

  @test_util.run_deprecated_v1
  def testRecordInputEpochs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the batch-count assertion compares lengths by identity rather than equality,
    # which only works because small integers are cached; the session handle is unused.
    # Scenario: one hundred files of one hundred records each are read with two reader threads,
    # a large buffer, a shift ratio and fixed seed, producing two batches per run. Over three
    # epochs (each of files times records divided by batches runs) asserts every run yields the
    # requested number of batches and that no record appears twice within one epoch.
    files = 100
    records_per_file = 100
    batches = 2
    with self.cached_session() as sess:
      self.generateTestData("basic", files, records_per_file)

      records = data_flow_ops.RecordInput(
          file_pattern=os.path.join(self.get_temp_dir(), "basic.*"),
          parallelism=2,
          buffer_size=2000,
          batch_size=1,
          shift_ratio=0.33,
          seed=10,
          name="record_input",
          batches=batches)

      yield_op = records.get_yield_op()

      # cycle over 3 epochs and make sure we never duplicate
      for _ in range(3):
        epoch_set = set()
        for _ in range(int(files * records_per_file / batches)):
          op_list = self.evaluate(yield_op)
          self.assertTrue(len(op_list) is batches)
          for r in op_list:
            self.assertTrue(r[0] not in epoch_set)
            epoch_set.add(r[0])

  @test_util.run_deprecated_v1
  def testDoesNotDeadlock(self):
    # [FUNCTION DOC]
    # Scenario: thirty times, creates a fresh session and a one file, one record data set and
    # evaluates the yield operation fifty times (many more than the records available, so the
    # reader wraps around). Passes if nothing hangs or raises; there are no value assertions.
    # Iterate multiple times to cause deadlock if there is a chance it can occur
    for _ in range(30):
      with self.cached_session() as sess:
        self.generateTestData("basic", 1, 1)

        records = data_flow_ops.RecordInput(
            file_pattern=os.path.join(self.get_temp_dir(), "basic.*"),
            parallelism=1,
            buffer_size=100,
            batch_size=1,
            name="record_input")

        yield_op = records.get_yield_op()
        for _ in range(50):
          self.evaluate(yield_op)

  @test_util.run_deprecated_v1
  def testEmptyGlob(self):
    # [FUNCTION DOC]
    # Scenario: a record input op over a file pattern that matches nothing. After initializing
    # global variables, asserts evaluating the yield operation raises a not-found error.
    with self.cached_session() as sess:
      record_input = data_flow_ops.RecordInput(file_pattern="foo")
      yield_op = record_input.get_yield_op()
      self.evaluate(variables.global_variables_initializer())
      with self.assertRaises(errors_impl.NotFoundError):
        self.evaluate(yield_op)

  @test_util.run_deprecated_v1
  def testBufferTooSmall(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests a buffer smaller than the data, but the buffer size
    # (two thousand) is larger than the total number of records (one hundred), so this is just
    # the epochs scenario on a smaller data set.
    # Scenario: ten files of ten records, two reader threads, two batches per run, three epochs.
    # Asserts each run yields the requested number of batches (checked by identity, not
    # equality) and no record repeats within an epoch.
    files = 10
    records_per_file = 10
    batches = 2
    with self.cached_session() as sess:
      self.generateTestData("basic", files, records_per_file)

      records = data_flow_ops.RecordInput(
          file_pattern=os.path.join(self.get_temp_dir(), "basic.*"),
          parallelism=2,
          buffer_size=2000,
          batch_size=1,
          shift_ratio=0.33,
          seed=10,
          name="record_input",
          batches=batches)

      yield_op = records.get_yield_op()

      # cycle over 3 epochs and make sure we never duplicate
      for _ in range(3):
        epoch_set = set()
        for _ in range(int(files * records_per_file / batches)):
          op_list = self.evaluate(yield_op)
          self.assertTrue(len(op_list) is batches)
          for r in op_list:
            self.assertTrue(r[0] not in epoch_set)
            epoch_set.add(r[0])

  def testInvalidParams(self):
    # [FUNCTION DOC]
    # Scenario: calls the raw record input op with a bogus file pattern, negative buffer size,
    # parallelism, shift ratio, batch size and seed, and an unknown compression type. Asserts
    # that evaluating it raises an invalid-argument error.
    with self.session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(
            data_flow_ops.gen_data_flow_ops.record_input(
                file_pattern="nan",
                file_buffer_size=-90,
                file_parallelism=-438,
                file_shuffle_shift_ratio=-784,
                batch_size=-933,
                file_random_seed=-678,
                compression_type="nan",
            )
        )


if __name__ == "__main__":
  test.main()
