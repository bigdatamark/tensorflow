# [MODULE DOC]
# Test module for the legacy (graph-mode, queue-based) reader ops exposed through the io_ops
# module: IdentityReader, WholeFileReader, TextLineReader, FixedLengthRecordReader and
# TFRecordReader. Each test class creates small input files (or queue entries) in the test temp
# directory, feeds file names or strings through a FIFO queue, reads records with the reader,
# and asserts on keys, values, counters, state serialization, compression handling and the
# error raised once a closed queue is exhausted. Most tests are decorated to run only in
# deprecated graph (v1) mode because the readers and queues are not eager compatible.
# It also contains a small shared base class that writes and (de)compresses TFRecord files and a
# test that readers blocked on queues in separate threads do not deadlock the session.
# Defines a module-level path prefix, a TFRecord compression alias and a sample text constant.
# Running the file as a script starts the TensorFlow test runner. No import-time side effects
# beyond constant definitions.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Reader ops from io_ops."""

import collections
import gzip
import os
import threading
import zlib

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import compat

prefix_path = "tensorflow/core/lib"

# pylint: disable=invalid-name
TFRecordCompressionType = tf_record.TFRecordCompressionType
# pylint: enable=invalid-name

# Edgar Allan Poe's 'Eldorado'
_TEXT = b"""Gaily bedight,
    A gallant knight,
    In sunshine and in shadow,
    Had journeyed long,
    Singing a song,
    In search of Eldorado.

    But he grew old
    This knight so bold
    And o'er his heart a shadow
    Fell as he found
    No spot of ground
    That looked like Eldorado.

   And, as his strength
   Failed him at length,
   He met a pilgrim shadow
   'Shadow,' said he,
   'Where can it be
   This land of Eldorado?'

   'Over the Mountains
    Of the Moon'
    Down the Valley of the Shadow,
    Ride, boldly ride,'
    The shade replied,
    'If you seek for Eldorado!'
    """


class TFCompressionTestCase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the TFRecord and fixed-length reader tests below.
  # Test fixture base class (it defines no tests itself). Holds the number of files and records
  # per file and provides helpers that write TFRecord files, build the expected record
  # contents, and compress or decompress files with zlib or gzip inside the test temp directory.
  # Subclasses inherit these helpers and may override the record generator and file creation.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: calls the parent setup, then sets the number of files to create (two) and
    # the number of records per file (seven). Takes no inputs and returns nothing.
    super(TFCompressionTestCase, self).setUp()
    self._num_files = 2
    self._num_records = 7

  def _Record(self, f, r):
    # [FUNCTION DOC]
    # HELPER: builds expected record contents for the TFRecord tests.
    # Inputs: f is the file index and r is the record index (integers). Returns the bytes of a
    # human readable string naming the record number and file number. No side effects.
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _CreateFiles(self, options=None, prefix=""):
    # [FUNCTION DOC]
    # HELPER: creates the input TFRecord files for the tests.
    # Inputs: options is an optional TFRecord options object (for example selecting a
    # compression type), default none meaning uncompressed; prefix is a string prepended to each
    # file name (default empty). Writes the configured number of files, each holding the
    # configured number of records produced by the record generator, into the test temp
    # directory. Returns the list of file paths in creation order.
    filenames = []
    for i in range(self._num_files):
      name = prefix + "tfrecord.%d.txt" % i
      records = [self._Record(i, j) for j in range(self._num_records)]
      fn = self._WriteRecordsToFile(records, name, options)
      filenames.append(fn)
    return filenames

  def _WriteRecordsToFile(self, records, name="tfrecord", options=None):
    # [FUNCTION DOC]
    # HELPER: low-level TFRecord writer used by the file creation helper.
    # Inputs: records is an iterable of bytes; name is the file name inside the test temp
    # directory (default "tfrecord"); options is an optional TFRecord options object. Writes each
    # record with a TFRecord writer (closed automatically) and returns the full file path.
    fn = os.path.join(self.get_temp_dir(), name)
    with tf_record.TFRecordWriter(fn, options=options) as writer:
      for r in records:
        writer.write(r)
    return fn

  def _ZlibCompressFile(self, infile, name="tfrecord.z"):
    # [FUNCTION DOC]
    # HELPER: produces zlib-compressed copies of test files.
    # Inputs: infile is the path of an existing file; name is the output file name inside the
    # test temp directory (default "tfrecord.z"). Reads the whole input, compresses it with zlib
    # and writes it to the output (overwriting if it exists). Returns the output path.
    # zlib compress the file and write compressed contents to file.
    with open(infile, "rb") as f:
      cdata = zlib.compress(f.read())

    zfn = os.path.join(self.get_temp_dir(), name)
    with open(zfn, "wb") as f:
      f.write(cdata)
    return zfn

  def _GzipCompressFile(self, infile, name="tfrecord.gz"):
    # [FUNCTION DOC]
    # HELPER: produces gzip-compressed copies of test files.
    # Inputs: infile is the path of an existing file; name is the output file name inside the
    # test temp directory (default "tfrecord.gz"). Reads the whole input, writes it through a
    # gzip writer to the output (overwriting if it exists). Returns the output path.
    # gzip compress the file and write compressed contents to file.
    with open(infile, "rb") as f:
      cdata = f.read()

    gzfn = os.path.join(self.get_temp_dir(), name)
    with gzip.GzipFile(gzfn, "wb") as f:
      f.write(cdata)
    return gzfn

  def _ZlibDecompressFile(self, infile, name="tfrecord"):
    # [FUNCTION DOC]
    # HELPER: inverse of the zlib compression helper for test files.
    # Inputs: infile is a zlib-compressed file path; name is the output file name inside the test
    # temp directory (default "tfrecord"). Decompresses the input and writes the plain bytes to
    # the output. Returns the output path. Raises a zlib error if the input is not valid.
    with open(infile, "rb") as f:
      cdata = zlib.decompress(f.read())
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn

  def _GzipDecompressFile(self, infile, name="tfrecord"):
    # [FUNCTION DOC]
    # HELPER: inverse of the gzip compression helper for test files.
    # Inputs: infile is a gzip-compressed file path; name is the output file name inside the test
    # temp directory (default "tfrecord"). Reads the decompressed bytes and writes them to the
    # output. Returns the output path. Raises an I/O error if the input is not valid gzip.
    with gzip.GzipFile(infile, "rb") as f:
      cdata = f.read()
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn


class IdentityReaderTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the identity reader op, which simply returns each queue element as both key and
  # value. Covers single and repeated epochs over a FIFO queue, the work-completed and
  # records-produced counters, state serialization and restoration (including malformed state),
  # and resetting. Uses a shared helper that reads one pair and checks it.

  def _ExpectRead(self, key, value, expected):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the identity reader tests.
    # Inputs: key and value are the tensors returned by a reader read; expected is the expected
    # element. Evaluates both tensors together (one read) and asserts that the key and the value
    # each equal the expected element. Returns nothing; fails the test on mismatch.
    k, v = self.evaluate([key, value])
    self.assertAllEqual(expected, k)
    self.assertAllEqual(expected, v)

  @test_util.run_deprecated_v1
  def testOneEpoch(self):
    # [FUNCTION DOC]
    # Scenario: an identity reader consumes one queue holding three strings that is then closed.
    # Checks that the work-completed, records-produced and queue-size counters start at zero,
    # become three after enqueueing (queue size), increase as each element is read in order,
    # and that a further read on the closed, empty queue raises the insufficient-elements
    # error. Finally asserts the counters show three units completed, three records produced
    # and an empty queue.
    reader = io_ops.IdentityReader("test_reader")
    work_completed = reader.num_work_units_completed()
    produced = reader.num_records_produced()
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    queued_length = queue.size()
    key, value = reader.read(queue)

    self.assertAllEqual(0, self.evaluate(work_completed))
    self.assertAllEqual(0, self.evaluate(produced))
    self.assertAllEqual(0, self.evaluate(queued_length))

    self.evaluate(queue.enqueue_many([["A", "B", "C"]]))
    self.evaluate(queue.close())
    self.assertAllEqual(3, self.evaluate(queued_length))

    self._ExpectRead(key, value, b"A")
    self.assertAllEqual(1, self.evaluate(produced))

    self._ExpectRead(key, value, b"B")

    self._ExpectRead(key, value, b"C")
    self.assertAllEqual(3, self.evaluate(produced))
    self.assertAllEqual(0, self.evaluate(queued_length))

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])

    self.assertAllEqual(3, self.evaluate(work_completed))
    self.assertAllEqual(3, self.evaluate(produced))
    self.assertAllEqual(0, self.evaluate(queued_length))

  @test_util.run_deprecated_v1
  def testMultipleEpochs(self):
    # [FUNCTION DOC]
    # Scenario: the same two-element enqueue operation is run three times against one identity
    # reader. Asserts that after each enqueue the reader returns the two elements in order
    # again, and that once the queue is closed and drained a read raises the
    # insufficient-elements error.
    reader = io_ops.IdentityReader("test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    enqueue = queue.enqueue_many([["DD", "EE"]])
    key, value = reader.read(queue)

    self.evaluate(enqueue)
    self._ExpectRead(key, value, b"DD")
    self._ExpectRead(key, value, b"EE")
    self.evaluate(enqueue)
    self._ExpectRead(key, value, b"DD")
    self._ExpectRead(key, value, b"EE")
    self.evaluate(enqueue)
    self._ExpectRead(key, value, b"DD")
    self._ExpectRead(key, value, b"EE")
    self.evaluate(queue.close())
    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])

  @test_util.run_deprecated_v1
  def testSerializeRestore(self):
    # [FUNCTION DOC]
    # Scenario: reads one element, serializes the reader state, reads the rest, re-enqueues the
    # last two elements, closes the queue and restores the saved state. Asserts the
    # records-produced counter goes back to one, the remaining elements are read again, the
    # exhausted closed queue raises the insufficient-elements error and the serialized state is
    # a bytes object. Also checks that restoring from an empty list or a list of two states
    # raises a value error, and that truncated, extended, prefixed or otherwise corrupted
    # state raises an op error saying the state of the named identity reader could not be parsed.
    reader = io_ops.IdentityReader("test_reader")
    produced = reader.num_records_produced()
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    self.evaluate(queue.enqueue_many([["X", "Y", "Z"]]))
    key, value = reader.read(queue)

    self._ExpectRead(key, value, b"X")
    self.assertAllEqual(1, self.evaluate(produced))
    state = self.evaluate(reader.serialize_state())

    self._ExpectRead(key, value, b"Y")
    self._ExpectRead(key, value, b"Z")
    self.assertAllEqual(3, self.evaluate(produced))

    self.evaluate(queue.enqueue_many([["Y", "Z"]]))
    self.evaluate(queue.close())
    self.evaluate(reader.restore_state(state))
    self.assertAllEqual(1, self.evaluate(produced))
    self._ExpectRead(key, value, b"Y")
    self._ExpectRead(key, value, b"Z")
    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])
    self.assertAllEqual(3, self.evaluate(produced))

    self.assertEqual(bytes, type(state))

    with self.assertRaises(ValueError):
      reader.restore_state([])

    with self.assertRaises(ValueError):
      reader.restore_state([state, state])

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state[1:]))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state[:-1]))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state + b"ExtraJunk"))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(b"PREFIX" + state))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(b"BOGUS" + state[5:]))

  @test_util.run_deprecated_v1
  def testReset(self):
    # [FUNCTION DOC]
    # Scenario: reads two of three queued elements, checking the counters are positive and the
    # queue has shrunk, then resets the reader. Asserts that work completed and records produced
    # return to zero, that the reader releases its current work item back (queue size one), that
    # the next read returns the third element, and that a freshly enqueued element is read after.
    reader = io_ops.IdentityReader("test_reader")
    work_completed = reader.num_work_units_completed()
    produced = reader.num_records_produced()
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    queued_length = queue.size()
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([["X", "Y", "Z"]]))
    self._ExpectRead(key, value, b"X")
    self.assertLess(0, self.evaluate(queued_length))
    self.assertAllEqual(1, self.evaluate(produced))

    self._ExpectRead(key, value, b"Y")
    self.assertLess(0, self.evaluate(work_completed))
    self.assertAllEqual(2, self.evaluate(produced))

    self.evaluate(reader.reset())
    self.assertAllEqual(0, self.evaluate(work_completed))
    self.assertAllEqual(0, self.evaluate(produced))
    self.assertAllEqual(1, self.evaluate(queued_length))
    self._ExpectRead(key, value, b"Z")

    self.evaluate(queue.enqueue_many([["K", "L"]]))
    self._ExpectRead(key, value, b"K")


class WholeFileReaderTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the whole-file reader op, which returns each file name as the key and the entire
  # file contents as the value. The fixture creates three small files with different contents in
  # the temp directory and removes them afterwards. Covers a single epoch over a closed queue and
  # repeated enqueueing of the same file names.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: after the parent setup, creates three files named by index in the test temp
    # directory and writes distinct byte contents to each; stores the file names and contents
    # for later comparison. Side effect: files are created on disk.
    super(WholeFileReaderTest, self).setUp()
    self._filenames = [
        os.path.join(self.get_temp_dir(), "whole_file.%d.txt" % i)
        for i in range(3)
    ]
    self._content = [b"One\na\nb\n", b"Two\nC\nD", b"Three x, y, z"]
    for fn, c in zip(self._filenames, self._content):
      with open(fn, "wb") as h:
        h.write(c)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: removes the three files created during setup (raises an OS error if one
    # is missing), then calls the parent cleanup. Takes no inputs and returns nothing.
    for fn in self._filenames:
      os.remove(fn)
    super(WholeFileReaderTest, self).tearDown()

  def _ExpectRead(self, key, value, index):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the whole-file reader tests.
    # Inputs: key and value are the reader output tensors; index selects which fixture file is
    # expected. Evaluates both together and asserts the key equals that file's name (as bytes)
    # and the value equals its full contents. Returns nothing.
    k, v = self.evaluate([key, value])
    self.assertAllEqual(compat.as_bytes(self._filenames[index]), k)
    self.assertAllEqual(self._content[index], v)

  @test_util.run_deprecated_v1
  def testOneEpoch(self):
    # [FUNCTION DOC]
    # Scenario: the three file names are enqueued and the queue closed. Asserts the reader
    # returns each file's name and content in order, and then a read on the exhausted closed
    # queue raises the insufficient-elements error.
    reader = io_ops.WholeFileReader("test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    self.evaluate(queue.enqueue_many([self._filenames]))
    self.evaluate(queue.close())
    key, value = reader.read(queue)

    self._ExpectRead(key, value, 0)
    self._ExpectRead(key, value, 1)
    self._ExpectRead(key, value, 2)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])

  @test_util.run_deprecated_v1
  def testInfiniteEpochs(self):
    # [FUNCTION DOC]
    # Scenario: the same enqueue operation (all three file names) is run several times without
    # closing the queue. Asserts the reader keeps cycling through the files, returning the right
    # name and contents each time in queue order (files 0,1 then 2,0,1 then 2,0).
    reader = io_ops.WholeFileReader("test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    enqueue = queue.enqueue_many([self._filenames])
    key, value = reader.read(queue)

    self.evaluate(enqueue)
    self._ExpectRead(key, value, 0)
    self._ExpectRead(key, value, 1)
    self.evaluate(enqueue)
    self._ExpectRead(key, value, 2)
    self._ExpectRead(key, value, 0)
    self._ExpectRead(key, value, 1)
    self.evaluate(enqueue)
    self._ExpectRead(key, value, 2)
    self._ExpectRead(key, value, 0)


class TextLineReaderTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the text line reader op, which returns one line per read keyed by "file:line
  # number". Fixture holds the number of files and lines per file. Covers LF and CRLF line
  # endings, a file without a trailing newline, and skipping header lines.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: calls the parent setup and sets two files with five lines each. Returns
    # nothing.
    super(TextLineReaderTest, self).setUp()
    self._num_files = 2
    self._num_lines = 5

  def _LineText(self, f, l):
    # [FUNCTION DOC]
    # HELPER: builds the expected text of one line for the text line reader tests.
    # Inputs: f is the file index and l the line index. Returns the bytes of a string made of
    # the two numbers separated by a colon and a space. No side effects.
    return compat.as_bytes("%d: %d" % (f, l))

  def _CreateFiles(self, crlf=False):
    # [FUNCTION DOC]
    # HELPER: creates the text input files for the tests.
    # Input: crlf, a boolean (default false) choosing carriage-return plus line-feed endings
    # instead of line feed. Writes the configured number of files into the test temp directory,
    # each with the configured number of lines. Every line is terminated except the last line
    # of the second file, which deliberately has no trailing newline. Returns the file paths.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        for j in range(self._num_lines):
          f.write(self._LineText(i, j))
          # Always include a newline after the record unless it is
          # at the end of the file, in which case we include it sometimes.
          if j + 1 != self._num_lines or i == 0:
            f.write(b"\r\n" if crlf else b"\n")
    return filenames

  def _testOneEpoch(self, files):
    # [FUNCTION DOC]
    # HELPER: shared body of the line-ending tests.
    # Input: files, a list of text file paths. Creates a text line reader and a queue, enqueues
    # the files and closes the queue, then reads every line asserting the key is "file:line
    # number" (one based) and the value is the expected line text. Finally asserts that another
    # read raises the insufficient-elements error from the closed queue.
    reader = io_ops.TextLineReader(name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_lines):
        k, v = self.evaluate([key, value])
        self.assertAllEqual("%s:%d" % (files[i], j + 1), compat.as_text(k))
        self.assertAllEqual(self._LineText(i, j), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  @test_util.run_deprecated_v1
  def testOneEpochLF(self):
    # [FUNCTION DOC]
    # Scenario: files with line-feed endings are read fully; delegates to the shared one-epoch
    # check, which asserts keys, values and end-of-queue error.
    self._testOneEpoch(self._CreateFiles(crlf=False))

  @test_util.run_deprecated_v1
  def testOneEpochCRLF(self):
    # [FUNCTION DOC]
    # Scenario: files with carriage-return plus line-feed endings are read fully; asserts the
    # reader strips the endings so values match the plain line text (shared one-epoch check).
    self._testOneEpoch(self._CreateFiles(crlf=True))

  @test_util.run_deprecated_v1
  def testSkipHeaderLines(self):
    # [FUNCTION DOC]
    # Scenario: a text line reader configured to skip one header line per file reads two files.
    # Asserts only the remaining lines are returned, with keys counting lines from two (the
    # skipped line still counts) and values matching the corresponding later lines, followed by
    # the insufficient-elements error on the closed, drained queue.
    files = self._CreateFiles()
    reader = io_ops.TextLineReader(skip_header_lines=1, name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_lines - 1):
        k, v = self.evaluate([key, value])
        self.assertAllEqual("%s:%d" % (files[i], j + 2), compat.as_text(k))
        self.assertAllEqual(self._LineText(i, j + 1), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])


class FixedLengthRecordReaderTest(TFCompressionTestCase):
  # [CLASS DOC]
  # Tests for the fixed-length record reader op. Inherits the compression helpers from the shared
  # base class. The fixture defines a header of five bytes, records of three bytes, a footer of
  # two bytes and a hop of two bytes (used for overlapping records). Files are built from a run
  # of header bytes, records (optionally separated by gap bytes) and footer bytes, optionally
  # compressed with gzip or zlib. Covers plain, gzip and zlib input, with and without a hop
  # size that makes consecutive records overlap, including empty (zero record) files.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: calls the parent setup, then sets two files, a five byte header, three byte
    # records, a two byte footer and a two byte hop. Returns nothing.
    super(FixedLengthRecordReaderTest, self).setUp()
    self._num_files = 2
    self._header_bytes = 5
    self._record_bytes = 3
    self._footer_bytes = 2

    self._hop_bytes = 2

  def _Record(self, f, r):
    # [FUNCTION DOC]
    # HELPER: overrides the base class record generator for the fixed-length tests.
    # Inputs: f is the file index, r the record index. Returns bytes made of the decimal digits
    # of twice the file index plus the record index, repeated record-size times (so the result is
    # not guaranteed to be exactly the record size when that number has several digits, and
    # different file/record pairs can yield identical content).
    return compat.as_bytes(str(f * 2 + r) * self._record_bytes)

  def _OverlappedRecord(self, f, r):
    # [FUNCTION DOC]
    # HELPER: computes the expected content of an overlapping record.
    # Inputs: f is the file index (unused in the result, all files share the same content),
    # r is the record index. Returns the bytes formed by the first digit of each integer in the
    # window starting at record index times hop size and of record-size length. No side effects.
    record_str = "".join([
        str(i)[0]
        for i in range(r * self._hop_bytes,
                       r * self._hop_bytes + self._record_bytes)
    ])
    return compat.as_bytes(record_str)

  # gap_bytes=hop_bytes-record_bytes
  def _CreateFiles(self, num_records, gap_bytes):
    # [FUNCTION DOC]
    # HELPER: overrides the base class file creation with fixed-length record files.
    # Inputs: num_records is the number of records per file (may be zero); gap_bytes is the
    # number of filler bytes written between consecutive records (zero for back-to-back).
    # Writes the configured number of files into the test temp directory, each holding header
    # filler bytes, the records (separated by gap filler bytes) and footer filler bytes.
    # Returns the list of file paths. Note the signature differs from the base class version.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        f.write(b"H" * self._header_bytes)
        if num_records > 0:
          f.write(self._Record(i, 0))
        for j in range(1, num_records):
          if gap_bytes > 0:
            f.write(b"G" * gap_bytes)
          f.write(self._Record(i, j))
        f.write(b"F" * self._footer_bytes)
    return filenames

  def _CreateOverlappedRecordFiles(self, num_overlapped_records):
    # [FUNCTION DOC]
    # HELPER: creates uncompressed input files for the hop-size tests.
    # Input: num_overlapped_records is the number of overlapping records the file must contain
    # (may be zero). For each file writes header filler, then one continuous string of the first
    # digits of consecutive integers long enough to hold that many records at the hop spacing,
    # then footer filler. Returns the list of file paths in the test temp directory.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(),
                        "fixed_length_overlapped_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        f.write(b"H" * self._header_bytes)
        if num_overlapped_records > 0:
          all_records_str = "".join([
              str(i)[0]
              for i in range(self._record_bytes + self._hop_bytes *
                             (num_overlapped_records - 1))
          ])
          f.write(compat.as_bytes(all_records_str))
        f.write(b"F" * self._footer_bytes)
    return filenames

  # gap_bytes=hop_bytes-record_bytes
  def _CreateGzipFiles(self, num_records, gap_bytes):
    # [FUNCTION DOC]
    # HELPER: gzip variant of the fixed-length file creation.
    # Inputs as for the plain creation helper (record count and gap bytes). Creates the plain
    # files then gzip-compresses each in place (overwriting it). Returns the file paths.
    filenames = self._CreateFiles(num_records, gap_bytes)
    for fn in filenames:
      # compress inplace.
      self._GzipCompressFile(fn, fn)
    return filenames

  # gap_bytes=hop_bytes-record_bytes
  def _CreateZlibFiles(self, num_records, gap_bytes):
    # [FUNCTION DOC]
    # HELPER: zlib variant of the fixed-length file creation.
    # Inputs as for the plain creation helper (record count and gap bytes). Creates the plain
    # files then zlib-compresses each in place (overwriting it). Returns the file paths.
    filenames = self._CreateFiles(num_records, gap_bytes)
    for fn in filenames:
      # compress inplace.
      self._ZlibCompressFile(fn, fn)
    return filenames

  def _CreateGzipOverlappedRecordFiles(self, num_overlapped_records):
    # [FUNCTION DOC]
    # HELPER: gzip variant of the overlapped-record file creation.
    # Input: num_overlapped_records, the number of overlapping records (may be zero). Writes the
    # same header, digit string and footer content as the plain variant but directly through a
    # gzip writer, using the same file names as the plain variant. Returns the file paths.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(),
                        "fixed_length_overlapped_record.%d.txt" % i)
      filenames.append(fn)
      with gzip.GzipFile(fn, "wb") as f:
        f.write(b"H" * self._header_bytes)
        if num_overlapped_records > 0:
          all_records_str = "".join([
              str(i)[0]
              for i in range(self._record_bytes + self._hop_bytes *
                             (num_overlapped_records - 1))
          ])
          f.write(compat.as_bytes(all_records_str))
        f.write(b"F" * self._footer_bytes)
    return filenames

  def _CreateZlibOverlappedRecordFiles(self, num_overlapped_records):
    # [FUNCTION DOC]
    # HELPER: zlib variant of the overlapped-record file creation.
    # KNOWN ISSUE: the intermediate uncompressed files (the target name plus a temporary suffix)
    # are written and never deleted, so they are left behind in the test temp directory.
    # Input: num_overlapped_records, the number of overlapping records (may be zero). Writes the
    # plain content to a temporary file for each target, then zlib-compresses it into the final
    # name. Returns the list of final file paths.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(),
                        "fixed_length_overlapped_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn + ".tmp", "wb") as f:
        f.write(b"H" * self._header_bytes)
        if num_overlapped_records > 0:
          all_records_str = "".join([
              str(i)[0]
              for i in range(self._record_bytes + self._hop_bytes *
                             (num_overlapped_records - 1))
          ])
          f.write(compat.as_bytes(all_records_str))
        f.write(b"F" * self._footer_bytes)
      self._ZlibCompressFile(fn + ".tmp", fn)
    return filenames

  # gap_bytes=hop_bytes-record_bytes
  def _TestOneEpoch(self, files, num_records, gap_bytes, encoding=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the non-overlapping one-epoch tests.
    # Inputs: files is the list of input paths; num_records the expected records per file;
    # gap_bytes the filler between records (zero means the hop is left at zero, otherwise the
    # hop is record size plus gap); encoding an optional compression name (for example gzip or
    # zlib), default none. Builds a fixed-length reader with the fixture sizes, enqueues the
    # files, closes the queue, reads every record asserting key "file:record index" and the
    # expected value, then asserts the insufficient-elements error on the drained queue.
    hop_bytes = 0 if gap_bytes == 0 else self._record_bytes + gap_bytes
    reader = io_ops.FixedLengthRecordReader(
        header_bytes=self._header_bytes,
        record_bytes=self._record_bytes,
        footer_bytes=self._footer_bytes,
        hop_bytes=hop_bytes,
        encoding=encoding,
        name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(num_records):
        k, v = self.evaluate([key, value])
        self.assertAllEqual("%s:%d" % (files[i], j), compat.as_text(k))
        self.assertAllEqual(self._Record(i, j), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  def _TestOneEpochWithHopBytes(self,
                                files,
                                num_overlapped_records,
                                encoding=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the overlapping-record one-epoch tests.
    # Inputs: files is the list of input paths; num_overlapped_records the expected number of
    # records per file; encoding an optional compression name, default none. Builds a reader
    # using the fixture's fixed hop size, enqueues and closes, reads all records asserting the
    # key "file:record index" and the overlapped expected value, then asserts the
    # insufficient-elements error on the drained queue.
    reader = io_ops.FixedLengthRecordReader(
        header_bytes=self._header_bytes,
        record_bytes=self._record_bytes,
        footer_bytes=self._footer_bytes,
        hop_bytes=self._hop_bytes,
        encoding=encoding,
        name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(num_overlapped_records):
        k, v = self.evaluate([key, value])
        self.assertAllEqual("%s:%d" % (files[i], j), compat.as_text(k))
        self.assertAllEqual(self._OverlappedRecord(i, j), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  @test_util.run_deprecated_v1
  def testOneEpoch(self):
    # [FUNCTION DOC]
    # Scenario: plain files with zero or seven records, each with no gap or a one byte gap
    # between records. Asserts the reader returns every record with correct key and value.
    for num_records in [0, 7]:
      # gap_bytes=0: hop_bytes=0
      # gap_bytes=1: hop_bytes=record_bytes+1
      for gap_bytes in [0, 1]:
        files = self._CreateFiles(num_records, gap_bytes)
        self._TestOneEpoch(files, num_records, gap_bytes)

  @test_util.run_deprecated_v1
  def testGzipOneEpoch(self):
    # [FUNCTION DOC]
    # Scenario: same as the plain one-epoch test but the files are gzip compressed and the
    # reader is told to use gzip decoding. Asserts all records are read correctly.
    for num_records in [0, 7]:
      # gap_bytes=0: hop_bytes=0
      # gap_bytes=1: hop_bytes=record_bytes+1
      for gap_bytes in [0, 1]:
        files = self._CreateGzipFiles(num_records, gap_bytes)
        self._TestOneEpoch(files, num_records, gap_bytes, encoding="GZIP")

  @test_util.run_deprecated_v1
  def testZlibOneEpoch(self):
    # [FUNCTION DOC]
    # Scenario: same as the plain one-epoch test but the files are zlib compressed and the
    # reader is told to use zlib decoding. Asserts all records are read correctly.
    for num_records in [0, 7]:
      # gap_bytes=0: hop_bytes=0
      # gap_bytes=1: hop_bytes=record_bytes+1
      for gap_bytes in [0, 1]:
        files = self._CreateZlibFiles(num_records, gap_bytes)
        self._TestOneEpoch(files, num_records, gap_bytes, encoding="ZLIB")

  @test_util.run_deprecated_v1
  def testOneEpochWithHopBytes(self):
    # [FUNCTION DOC]
    # Scenario: plain files containing zero or two overlapping records (hop smaller than record
    # size). Asserts the reader returns the overlapping records with correct keys and values.
    for num_overlapped_records in [0, 2]:
      files = self._CreateOverlappedRecordFiles(num_overlapped_records)
      self._TestOneEpochWithHopBytes(files, num_overlapped_records)

  @test_util.run_deprecated_v1
  def testGzipOneEpochWithHopBytes(self):
    # [FUNCTION DOC]
    # Scenario: gzip-compressed files with zero or two overlapping records, read with gzip
    # decoding. Asserts the overlapped records are returned correctly.
    for num_overlapped_records in [0, 2]:
      files = self._CreateGzipOverlappedRecordFiles(num_overlapped_records,)
      self._TestOneEpochWithHopBytes(
          files, num_overlapped_records, encoding="GZIP")

  @test_util.run_deprecated_v1
  def testZlibOneEpochWithHopBytes(self):
    # [FUNCTION DOC]
    # Scenario: zlib-compressed files with zero or two overlapping records, read with zlib
    # decoding. Asserts the overlapped records are returned correctly.
    for num_overlapped_records in [0, 2]:
      files = self._CreateZlibOverlappedRecordFiles(num_overlapped_records)
      self._TestOneEpochWithHopBytes(
          files, num_overlapped_records, encoding="ZLIB")


class TFRecordReaderTest(TFCompressionTestCase):
  # [CLASS DOC]
  # Tests for the TFRecord reader op using the inherited file creation helpers. Covers reading
  # one epoch of uncompressed files, reading up to a batch size of records at a time, and
  # reading zlib and gzip compressed files via reader options.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: only delegates to the parent setup (two files, seven records each).
    super(TFRecordReaderTest, self).setUp()

  @test_util.run_deprecated_v1
  def testOneEpoch(self):
    # [FUNCTION DOC]
    # Scenario: reads two uncompressed TFRecord files through a queue that is then closed.
    # Asserts each key starts with the file name and each value equals the expected record, and
    # that a read on the drained closed queue raises the insufficient-elements error.
    files = self._CreateFiles()
    reader = io_ops.TFRecordReader(name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_records):
        k, v = self.evaluate([key, value])
        self.assertTrue(compat.as_text(k).startswith("%s:" % files[i]))
        self.assertAllEqual(self._Record(i, j), v)

    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      k, v = self.evaluate([key, value])

  @test_util.run_deprecated_v1
  def testReadUpTo(self):
    # [FUNCTION DOC]
    # Scenario: reads the records with the batched read-up-to operation using a batch size of
    # three until the out-of-range error signals the end. Asserts each batch of keys and of
    # values has at most the batch size elements and that the total number read equals all
    # records in all files.
    files = self._CreateFiles()
    reader = io_ops.TFRecordReader(name="test_reader")
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    batch_size = 3
    key, value = reader.read_up_to(queue, batch_size)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    num_k = 0
    num_v = 0

    while True:
      try:
        k, v = self.evaluate([key, value])
        # Test reading *up to* batch_size records
        self.assertLessEqual(len(k), batch_size)
        self.assertLessEqual(len(v), batch_size)
        num_k += len(k)
        num_v += len(v)
      except errors_impl.OutOfRangeError:
        break

    # Test that we have read everything
    self.assertEqual(self._num_files * self._num_records, num_k)
    self.assertEqual(self._num_files * self._num_records, num_v)

  @test_util.run_deprecated_v1
  def testReadZlibFiles(self):
    # [FUNCTION DOC]
    # Scenario: files written with zlib compression are read by a reader given matching
    # options. Asserts keys start with the file name and values equal the expected records.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    files = self._CreateFiles(options)

    reader = io_ops.TFRecordReader(name="test_reader", options=options)
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_records):
        k, v = self.evaluate([key, value])
        self.assertTrue(compat.as_text(k).startswith("%s:" % files[i]))
        self.assertAllEqual(self._Record(i, j), v)

  @test_util.run_deprecated_v1
  def testReadGzipFiles(self):
    # [FUNCTION DOC]
    # Scenario: files written with gzip compression are read by a reader given matching
    # options. Asserts keys start with the file name and values equal the expected records.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    files = self._CreateFiles(options)

    reader = io_ops.TFRecordReader(name="test_reader", options=options)
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    key, value = reader.read(queue)

    self.evaluate(queue.enqueue_many([files]))
    self.evaluate(queue.close())
    for i in range(self._num_files):
      for j in range(self._num_records):
        k, v = self.evaluate([key, value])
        self.assertTrue(compat.as_text(k).startswith("%s:" % files[i]))
        self.assertAllEqual(self._Record(i, j), v)


class AsyncReaderTest(test.TestCase):
  # [CLASS DOC]
  # Test class checking that asynchronous reader ops blocked on empty queues in separate threads
  # do not starve each other when the session has a single thread per pool.

  @test_util.run_deprecated_v1
  def testNoDeadlockFromQueue(self):
    # [FUNCTION DOC]
    # Scenario: a session limited to one inter-op and one intra-op thread runs three text line
    # readers, each with its own queue, in three background threads; all block waiting for
    # queue entries. The test then writes one small file per reader (in reverse order of the
    # threads), enqueues its name into the corresponding queue and joins that thread. Asserts
    # each thread finishes (no deadlock) and its saved output is the single expected line.
    # Side effects: creates files in the temp directory and starts and joins threads.
    """Tests that reading does not block main execution threads."""
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    with self.session(config=config) as sess:
      thread_data_t = collections.namedtuple("thread_data_t",
                                             ["thread", "queue", "output"])
      thread_data = []

      # Create different readers, each with its own queue.
      for i in range(3):
        queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
        reader = io_ops.TextLineReader()
        _, line = reader.read(queue)
        output = []
        t = threading.Thread(
            target=AsyncReaderTest._RunSessionAndSave,
            args=(sess, [line], output))
        thread_data.append(thread_data_t(t, queue, output))

      # Start all readers. They are all blocked waiting for queue entries.
      self.evaluate(variables.global_variables_initializer())
      for d in thread_data:
        d.thread.start()

      # Unblock the readers.
      for i, d in enumerate(reversed(thread_data)):
        fname = os.path.join(self.get_temp_dir(), "deadlock.%s.txt" % i)
        with open(fname, "wb") as f:
          f.write(("file-%s" % i).encode())
        self.evaluate(d.queue.enqueue_many([[fname]]))
        d.thread.join()
        self.assertEqual([[("file-%s" % i).encode()]], d.output)

  @staticmethod
  def _RunSessionAndSave(sess, args, output):
    # [FUNCTION DOC]
    # HELPER: thread target for the deadlock test (a static method).
    # Inputs: sess is a session; args the tensors to fetch; output a list that is mutated by
    # appending the fetched result. Blocks until the fetch finishes. Returns nothing.
    output.append(sess.run(args))


if __name__ == "__main__":
  test.main()
