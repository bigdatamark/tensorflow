# [MODULE DOC]
# Unit tests for the TFRecord file writer, reader iterator and random reader in the tf_record
# module. They cover writing and reading with no compression, zlib and gzip, interoperability
# with the standard zlib and gzip libraries, the effect of compression options on file size,
# iterator behavior on growing, truncated and replaced files (needed by TensorBoard), random
# access reads, and close/flush semantics of the writer. Defines a sample text constant and a
# shared base test case with file helpers. No import-time side effects other than defining tests.
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
"""Tests for tf_record.TFRecordWriter and tf_record.tf_record_iterator."""

import gzip
import os
import random
import string
import zlib

import six

from tensorflow.python.framework import errors_impl
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import test
from tensorflow.python.util import compat

TFRecordCompressionType = tf_record.TFRecordCompressionType

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
  # HELPER: shared base class for the TFRecord tests in this module.
  # Test case base providing fixtures and file helpers: record generation, writing record files
  # with given options, and zlib/gzip compress and decompress of whole files into the temp
  # directory. Holds the number of files and records per file. Contains no tests itself.
  """TFCompression Test"""

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp, then sets the number of files to two and records per file to seven.
    super(TFCompressionTestCase, self).setUp()
    self._num_files = 2
    self._num_records = 7

  def _Record(self, f, r):
    # [FUNCTION DOC]
    # HELPER: builds deterministic record payloads for the tests.
    # Takes a file index and a record index and returns a bytes value naming both.
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _CreateFiles(self, options=None, prefix=""):
    # [FUNCTION DOC]
    # HELPER: creates the standard set of TFRecord files for the tests.
    # Writes one file per configured file count, each holding the configured number of records,
    # using optional writer options (default none) and a file name prefix (default empty).
    # Returns the list of written file paths. Side effect: files in the temp directory.
    filenames = []
    for i in range(self._num_files):
      name = prefix + "tfrecord.%d.txt" % i
      records = [self._Record(i, j) for j in range(self._num_records)]
      fn = self._WriteRecordsToFile(records, name, options)
      filenames.append(fn)
    return filenames

  def _WriteRecordsToFile(self, records, name="tfrecord", options=None):
    # [FUNCTION DOC]
    # HELPER: writes a list of byte records to a TFRecord file in the temp directory.
    # Inputs: records (list of bytes), name (file name, default "tfrecord"), options (optional
    # writer options). Uses the writer as a context manager so it is closed on exit.
    # Returns the full path of the file written.
    fn = os.path.join(self.get_temp_dir(), name)
    with tf_record.TFRecordWriter(fn, options=options) as writer:
      for r in records:
        writer.write(r)
    return fn

  def _ZlibCompressFile(self, infile, name="tfrecord.z"):
    # [FUNCTION DOC]
    # HELPER: compresses a whole file with the standard zlib library.
    # Reads the input file path, writes the compressed bytes to a new file named name (default
    # "tfrecord.z") in the temp directory and returns that path.
    # zlib compress the file and write compressed contents to file.
    with open(infile, "rb") as f:
      cdata = zlib.compress(f.read())

    zfn = os.path.join(self.get_temp_dir(), name)
    with open(zfn, "wb") as f:
      f.write(cdata)
    return zfn

  def _GzipCompressFile(self, infile, name="tfrecord.gz"):
    # [FUNCTION DOC]
    # HELPER: compresses a whole file with the standard gzip library.
    # Reads the input file path, writes a gzip file named name (default "tfrecord.gz") in the
    # temp directory and returns that path.
    # gzip compress the file and write compressed contents to file.
    with open(infile, "rb") as f:
      cdata = f.read()

    gzfn = os.path.join(self.get_temp_dir(), name)
    with gzip.GzipFile(gzfn, "wb") as f:
      f.write(cdata)
    return gzfn

  def _ZlibDecompressFile(self, infile, name="tfrecord"):
    # [FUNCTION DOC]
    # HELPER: decompresses a zlib-compressed file into a new file named name (default
    # "tfrecord") in the temp directory and returns its path. Fails if the input is not zlib data.
    with open(infile, "rb") as f:
      cdata = zlib.decompress(f.read())
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn

  def _GzipDecompressFile(self, infile, name="tfrecord"):
    # [FUNCTION DOC]
    # HELPER: decompresses a gzip file into a new file named name (default "tfrecord") in the
    # temp directory and returns its path. Fails if the input is not gzip data.
    with gzip.GzipFile(infile, "rb") as f:
      cdata = f.read()
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn


class TFRecordWriterTest(TFCompressionTestCase):
  # [CLASS DOC]
  # Tests for the TFRecord writer: output compatibility with manual zlib and gzip compression,
  # compression type option validation and the effect of individual zlib options on file size.
  """TFRecordWriter Test"""

  def _AssertFilesEqual(self, a, b, equal):
    # [FUNCTION DOC]
    # HELPER: pairwise compares the contents of two lists of files.
    # For each pair, asserts the bytes are equal when equal is true, otherwise asserts they differ.
    for an, bn in zip(a, b):
      with open(an, "rb") as af, open(bn, "rb") as bf:
        if equal:
          self.assertEqual(af.read(), bf.read())
        else:
          self.assertNotEqual(af.read(), bf.read())

  def _CompressionSizeDelta(self, records, options_a, options_b):
    # [FUNCTION DOC]
    # HELPER: measures how option sets compare in compressed file size.
    # Writes the records twice with the two option sets, reads each back with the record iterator
    # and asserts the records round-trip unchanged. Returns the size of the first file minus the
    # size of the second (negative means the first option set compressed better).
    """Validate compression with options_a and options_b and return size delta.

    Compress records with options_a and options_b. Uncompress both compressed
    files and assert that the contents match the original records. Finally
    calculate how much smaller the file compressed with options_a was than the
    file compressed with options_b.

    Args:
      records: The records to compress
      options_a: First set of options to compress with, the baseline for size.
      options_b: Second set of options to compress with.

    Returns:
      The difference in file size when using options_a vs options_b. A positive
      value means options_a was a better compression than options_b. A negative
      value means options_b had better compression than options_a.

    """

    fn_a = self._WriteRecordsToFile(records, "tfrecord_a", options=options_a)
    test_a = list(tf_record.tf_record_iterator(fn_a, options=options_a))
    self.assertEqual(records, test_a, options_a)

    fn_b = self._WriteRecordsToFile(records, "tfrecord_b", options=options_b)
    test_b = list(tf_record.tf_record_iterator(fn_b, options=options_b))
    self.assertEqual(records, test_b, options_b)

    # Negative number => better compression.
    return os.path.getsize(fn_a) - os.path.getsize(fn_b)

  def testWriteReadZLibFiles(self):
    # [FUNCTION DOC]
    # Scenario: records written uncompressed and then compressed manually with zlib must differ
    # from the uncompressed files, yet equal files written directly with zlib compression; and
    # decompressing the zlib output must give back the uncompressed files.
    """test Write Read ZLib Files"""
    # Write uncompressed then compress manually.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.NONE)
    files = self._CreateFiles(options, prefix="uncompressed")
    zlib_files = [
        self._ZlibCompressFile(fn, "tfrecord_%s.z" % i)
        for i, fn in enumerate(files)
    ]
    self._AssertFilesEqual(files, zlib_files, False)

    # Now write compressd and verify same.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    compressed_files = self._CreateFiles(options, prefix="compressed")
    self._AssertFilesEqual(compressed_files, zlib_files, True)

    # Decompress compress and verify same.
    uncompressed_files = [
        self._ZlibDecompressFile(fn, "tfrecord_%s.z" % i)
        for i, fn in enumerate(compressed_files)
    ]
    self._AssertFilesEqual(uncompressed_files, files, True)

  def testWriteReadGzipFiles(self):
    # [FUNCTION DOC]
    # Scenario: manually gzip-compressed files differ from uncompressed ones; files written with
    # gzip compression are not compared to them (header differs) but must decompress to the
    # original uncompressed files.
    """test Write Read Gzip Files"""
    # Write uncompressed then compress manually.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.NONE)
    files = self._CreateFiles(options, prefix="uncompressed")
    gzip_files = [
        self._GzipCompressFile(fn, "tfrecord_%s.gz" % i)
        for i, fn in enumerate(files)
    ]
    self._AssertFilesEqual(files, gzip_files, False)

    # Now write compressd and verify same.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    compressed_files = self._CreateFiles(options, prefix="compressed")

    # Note: Gzips written by TFRecordWriter add 'tfrecord_0' so
    # compressed_files can't be compared with gzip_files

    # Decompress compress and verify same.
    uncompressed_files = [
        self._GzipDecompressFile(fn, "tfrecord_%s.gz" % i)
        for i, fn in enumerate(compressed_files)
    ]
    self._AssertFilesEqual(uncompressed_files, files, True)

  def testNoCompressionType(self):
    # [FUNCTION DOC]
    # Checks that default options and an empty string both give an empty compression type string,
    # and that an integer or an unknown name such as BZ2 raises a value error.
    """test No Compression Type"""
    self.assertEqual(
        "",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions()))

    self.assertEqual(
        "",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions("")))

    with self.assertRaises(ValueError):
      tf_record.TFRecordOptions(5)

    with self.assertRaises(ValueError):
      tf_record.TFRecordOptions("BZ2")

  def testZlibCompressionType(self):
    # [FUNCTION DOC]
    # Checks the compression type string is ZLIB when options are built from the string name,
    # from the enum constant, or from another options object wrapping the constant.
    """test Zlib Compression Type"""
    zlib_t = tf_record.TFRecordCompressionType.ZLIB

    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions("ZLIB")))

    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions(zlib_t)))

    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions(tf_record.TFRecordOptions(zlib_t))))

  def testCompressionOptions(self):
    # [FUNCTION DOC]
    # Builds a mix of random and repeated-text records, then for each listed zlib option (level,
    # flush mode, buffer sizes, window bits, strategy) compares compressed size against default
    # zlib options. Asserts the size difference has the expected sign: equal, smaller, larger, or
    # less-or-equal as listed. Also verifies records round-trip through each option set.
    """Create record with mix of random and repeated data to test compression on."""
    rnd = random.Random(123)
    random_record = compat.as_bytes(
        "".join(rnd.choice(string.digits) for _ in range(10000)))
    repeated_record = compat.as_bytes(_TEXT)
    for _ in range(10000):
      start_i = rnd.randint(0, len(_TEXT))
      length = rnd.randint(10, 200)
      repeated_record += _TEXT[start_i:start_i + length]
    records = [random_record, repeated_record, random_record]

    tests = [
        ("compression_level", 2, "LE"),  # Lower compression is worse or equal.
        ("compression_level", 6, 0),  # Default compression_level is equal.
        ("flush_mode", zlib.Z_FULL_FLUSH, 1),  # A few less bytes.
        ("flush_mode", zlib.Z_NO_FLUSH, 0),  # NO_FLUSH is the default.
        ("input_buffer_size", 4096, 0),  # Increases time not size.
        ("output_buffer_size", 4096, 0),  # Increases time not size.
        ("window_bits", 8, -1),  # Smaller than default window increases size.
        ("compression_strategy", zlib.Z_HUFFMAN_ONLY, -1),  # Worse.
        ("compression_strategy", zlib.Z_FILTERED, "LE"),  # Worse or equal.
    ]

    compression_type = tf_record.TFRecordCompressionType.ZLIB
    options_a = tf_record.TFRecordOptions(compression_type)
    for prop, value, delta_sign in tests:
      options_b = tf_record.TFRecordOptions(
          compression_type=compression_type, **{prop: value})
      delta = self._CompressionSizeDelta(records, options_a, options_b)
      if delta_sign == "LE":
        self.assertLessEqual(
            delta, 0,
            "Setting {} = {}, file was {} smaller didn't match sign of {}"
            .format(prop, value, delta, delta_sign))
      else:
        self.assertTrue(
            delta == 0 if delta_sign == 0 else delta // delta_sign > 0,
            "Setting {} = {}, file was {} smaller didn't match sign of {}"
            .format(prop, value, delta, delta_sign))


class TFRecordWriterZlibTest(TFCompressionTestCase):
  # [CLASS DOC]
  # Tests interoperability of zlib and gzip compressed TFRecord files with the standard
  # libraries, including large records and unusual flush patterns.
  """TFRecordWriter Zlib test"""

  def testZLibFlushRecord(self):
    # [FUNCTION DOC]
    # Writes one small record, recompresses the file bytes one byte at a time with many full
    # flushes and trailing flushes, overwrites the file, and asserts the zlib-option iterator still
    # reads back the original record.
    """test ZLib Flush Record"""
    original = [b"small record"]
    fn = self._WriteRecordsToFile(original, "small_record")
    with open(fn, "rb") as h:
      buff = h.read()

    # creating more blocks and trailing blocks shouldn't break reads
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS)

    output = b""
    for c in buff:
      if isinstance(c, int):
        c = six.int2byte(c)
      output += compressor.compress(c)
      output += compressor.flush(zlib.Z_FULL_FLUSH)

    output += compressor.flush(zlib.Z_FULL_FLUSH)
    output += compressor.flush(zlib.Z_FULL_FLUSH)
    output += compressor.flush(zlib.Z_FINISH)

    # overwrite the original file with the compressed data
    with open(fn, "wb") as h:
      h.write(output)

    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(fn, options=options))
    self.assertEqual(actual, original)

  def testZlibReadWrite(self):
    # [FUNCTION DOC]
    # Writes two small records uncompressed, compresses the file with the zlib library, and
    # asserts the iterator with zlib options reads the original records.
    """Verify that files produced are zlib compatible."""
    original = [b"foo", b"bar"]
    fn = self._WriteRecordsToFile(original, "zlib_read_write.tfrecord")
    zfn = self._ZlibCompressFile(fn, "zlib_read_write.tfrecord.z")

    # read the compressed contents and verify.
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(zfn, options=options))
    self.assertEqual(actual, original)

  def testZlibReadWriteLarge(self):
    # [FUNCTION DOC]
    # Same as the small zlib read test but with a single record of about five megabytes.
    """Verify that writing large contents also works."""

    # Make it large (about 5MB)
    original = [_TEXT * 10240]
    fn = self._WriteRecordsToFile(original, "zlib_read_write_large.tfrecord")
    zfn = self._ZlibCompressFile(fn, "zlib_read_write_large.tfrecord.z")

    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(zfn, options=options))
    self.assertEqual(actual, original)

  def testGzipReadWrite(self):
    # [FUNCTION DOC]
    # Writes two records uncompressed, compresses the file with the gzip library, and asserts the
    # iterator with gzip options reads the original records.
    """Verify that files produced are gzip compatible."""
    original = [b"foo", b"bar"]
    fn = self._WriteRecordsToFile(original, "gzip_read_write.tfrecord")
    gzfn = self._GzipCompressFile(fn, "tfrecord.gz")

    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    actual = list(tf_record.tf_record_iterator(gzfn, options=options))
    self.assertEqual(actual, original)


class TFRecordIteratorTest(TFCompressionTestCase):
  # [CLASS DOC]
  # Tests for the TFRecord iterator: reading compressed and uncompressed files, end of file
  # behavior, handling of appended, truncated and replaced files, and reopening.
  """TFRecordIterator test"""

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp and sets the number of records per file to seven.
    super(TFRecordIteratorTest, self).setUp()
    self._num_records = 7

  def testIterator(self):
    # [FUNCTION DOC]
    # Writes zlib-compressed records, iterates with the matching options asserting each record in
    # order, and asserts the iterator then raises the stop-iteration signal.
    """test Iterator"""
    records = [self._Record(0, i) for i in range(self._num_records)]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(records, "compressed_records", options)

    reader = tf_record.tf_record_iterator(fn, options)
    for expected in records:
      record = next(reader)
      self.assertEqual(expected, record)
    with self.assertRaises(StopIteration):
      record = next(reader)

  def testWriteZlibRead(self):
    # [FUNCTION DOC]
    # Writes records with zlib compression, decompresses the file with the zlib library, and
    # asserts the iterator over the decompressed file yields the original records.
    """Verify compression with TFRecordWriter is zlib library compatible."""
    original = [b"foo", b"bar"]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(original, "write_zlib_read.tfrecord.z",
                                  options)

    zfn = self._ZlibDecompressFile(fn, "write_zlib_read.tfrecord")
    actual = list(tf_record.tf_record_iterator(zfn))
    self.assertEqual(actual, original)

  def testWriteZlibReadLarge(self):
    # [FUNCTION DOC]
    # Same as the zlib write-then-read test but with one record of about five megabytes.
    """Verify compression for large records is zlib library compatible."""
    # Make it large (about 5MB)
    original = [_TEXT * 10240]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(original, "write_zlib_read_large.tfrecord.z",
                                  options)
    zfn = self._ZlibDecompressFile(fn, "write_zlib_read_large.tfrecord")
    actual = list(tf_record.tf_record_iterator(zfn))
    self.assertEqual(actual, original)

  def testWriteGzipRead(self):
    # [FUNCTION DOC]
    # Writes two records with gzip compression, decompresses with the gzip library and asserts the
    # iterator over the decompressed file yields the original records.
    original = [b"foo", b"bar"]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    fn = self._WriteRecordsToFile(original, "write_gzip_read.tfrecord.gz",
                                  options)

    gzfn = self._GzipDecompressFile(fn, "write_gzip_read.tfrecord")
    actual = list(tf_record.tf_record_iterator(gzfn))
    self.assertEqual(actual, original)

  def testReadGrowingFile_preservesReadOffset(self):
    # [FUNCTION DOC]
    # Scenario: a writer keeps a file open and flushes two records; the iterator reads them and
    # then raises stop-iteration repeatedly at end of file. After a third record is written and
    # flushed, the iterator returns it, preserving its offset, and then stops again.
    """Verify that tf_record_iterator preserves read offset even after EOF.

    When a file is iterated to EOF, the iterator should raise StopIteration but
    not actually close the reader. Then if later new data is appended, the
    iterator should start returning that new data on the next call to next(),
    preserving the read offset. This behavior is required by TensorBoard.
    """
    # Start the file with a good record.
    fn = os.path.join(self.get_temp_dir(), "file.tfrecord")
    with tf_record.TFRecordWriter(fn) as writer:
      writer.write(b"one")
      writer.write(b"two")
      writer.flush()
      iterator = tf_record.tf_record_iterator(fn)
      self.assertEqual(b"one", next(iterator))
      self.assertEqual(b"two", next(iterator))
      # Iterating at EOF results in StopIteration repeatedly.
      with self.assertRaises(StopIteration):
        next(iterator)
      with self.assertRaises(StopIteration):
        next(iterator)
      # Retrying after adding a new record successfully returns the new record,
      # preserving the prior read offset.
      writer.write(b"three")
      writer.flush()
      self.assertEqual(b"three", next(iterator))
      with self.assertRaises(StopIteration):
        next(iterator)

  def testReadTruncatedFile_preservesReadOffset(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the iterator "throws an exception on bad TFRecords" but the
    # test mainly checks that reading resumes correctly once a truncated record is completed.
    # Scenario: a file holds a good record plus a record missing its last byte. The iterator
    # returns the good record, then raises a data-loss error on repeated attempts. After the
    # missing byte is appended, the completed record is returned and then iteration stops.
    """Verify that tf_record_iterator throws an exception on bad TFRecords.

    When a truncated record is completed, the iterator should return that new
    record on the next attempt at iteration, preserving the read offset. This
    behavior is required by TensorBoard.
    """
    # Write out a record and read it back it to get the raw bytes.
    fn = os.path.join(self.get_temp_dir(), "temp_file")
    with tf_record.TFRecordWriter(fn) as writer:
      writer.write(b"truncated")
    with open(fn, "rb") as f:
      record_bytes = f.read()
    # Start the file with a good record.
    fn_truncated = os.path.join(self.get_temp_dir(), "truncated_file")
    with tf_record.TFRecordWriter(fn_truncated) as writer:
      writer.write(b"good")
    with open(fn_truncated, "ab", buffering=0) as f:
      # Cause truncation by omitting the last byte from the record.
      f.write(record_bytes[:-1])
      iterator = tf_record.tf_record_iterator(fn_truncated)
      # Good record appears first.
      self.assertEqual(b"good", next(iterator))
      # Truncated record repeatedly causes DataLossError upon iteration.
      with self.assertRaises(errors_impl.DataLossError):
        next(iterator)
      with self.assertRaises(errors_impl.DataLossError):
        next(iterator)
      # Retrying after completing the record successfully returns the rest of
      # the file contents, preserving the prior read offset.
      f.write(record_bytes[-1:])
      self.assertEqual(b"truncated", next(iterator))
      with self.assertRaises(StopIteration):
        next(iterator)

  def testReadReplacedFile_preservesReadOffset_afterReopen(self):
    # [FUNCTION DOC]
    # Scenario: the iterator reads two records and hits end of file; the file is then replaced by
    # a longer version with a third record. The stale iterator still stops; after closing and
    # reopening it returns the new record at the preserved offset. On Windows the iterator is
    # closed before the replacement because open files cannot be replaced.
    """Verify that tf_record_iterator allows reopening at the same read offset.

    In some cases, data will be logically "appended" to a file by replacing the
    entire file with a new version that includes the additional data. For
    example, this can happen with certain GCS implementations (since GCS has no
    true append operation), or when using rsync without the `--inplace` option
    to transfer snapshots of a growing file. Since the iterator retains a handle
    to a stale version of the file, it won't return any of the new data.

    To force this to happen, callers can check for a replaced file (e.g. via a
    stat call that reflects an increased file size) and opt to close and reopen
    the iterator. When iteration is next attempted, this should result in
    reading from the newly opened file, while preserving the read offset. This
    behavior is required by TensorBoard.
    """
    def write_records_to_file(filename, records):
      # [FUNCTION DOC]
      # HELPER: writes a list of byte records to the named file with a plain writer and closes it.
      writer = tf_record.TFRecordWriter(filename)
      for record in records:
        writer.write(record)
      writer.close()

    fn = os.path.join(self.get_temp_dir(), "orig_file")
    write_records_to_file(fn, [b"one", b"two"])
    iterator = tf_record.tf_record_iterator(fn)
    self.assertEqual(b"one", next(iterator))
    self.assertEqual(b"two", next(iterator))
    # Iterating at EOF results in StopIteration repeatedly.
    with self.assertRaises(StopIteration):
      next(iterator)
    with self.assertRaises(StopIteration):
      next(iterator)
    # Add a new record to the end of the file by overwriting it.
    fn2 = os.path.join(self.get_temp_dir(), "new_file")
    write_records_to_file(fn2, [b"one", b"two", b"three"])
    # Windows disallows replacing files while in use, so close iterator early.
    if os.name == "nt":
      iterator.close()
    os.replace(fn2, fn)
    # Iterating at EOF still results in StopIteration; new data is not shown.
    with self.assertRaises(StopIteration):
      next(iterator)
    with self.assertRaises(StopIteration):
      next(iterator)
    # Retrying after close and reopen successfully returns the new record,
    # preserving the prior read offset.
    iterator.close()
    iterator.reopen()
    self.assertEqual(b"three", next(iterator))
    with self.assertRaises(StopIteration):
      next(iterator)


class TFRecordRandomReaderTest(TFCompressionTestCase):
  # [CLASS DOC]
  # Tests for the random-access TFRecord reader: reading by offset, invalid offsets and use
  # after closing.

  def testRandomReaderReadingWorks(self):
    # [FUNCTION DOC]
    # Reads all records forward, collecting offsets; asserts reading at the end raises an index
    # error about being out of range; then reads backward by saved offsets, asserting each record
    # and the returned next offset.
    """Test read access to random offsets in the TFRecord file."""
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)

    offset = 0
    offsets = [offset]
    # Do a pass of forward reading.
    for i in range(self._num_records):
      record, offset = reader.read(offset)
      self.assertEqual(record, records[i])
      offsets.append(offset)
    # Reading off the bound should lead to error.
    with self.assertRaisesRegex(IndexError, r"Out of range.*offset"):
      reader.read(offset)
    # Do a pass of backward reading.
    for i in range(self._num_records - 1, 0, -1):
      record, offset = reader.read(offsets[i])
      self.assertEqual(offset, offsets[i + 1])
      self.assertEqual(record, records[i])

  def testRandomReaderThrowsErrorForInvalidOffset(self):
    # [FUNCTION DOC]
    # Asserts that reading at offset one, which is not a record boundary, raises a data-loss
    # error mentioning a corrupted record.
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)
    with self.assertRaisesRegex(errors_impl.DataLossError, r"corrupted record"):
      reader.read(1)  # 1 is guaranteed to be an invalid offset.

  def testClosingRandomReaderCausesErrorsForFurtherReading(self):
    # [FUNCTION DOC]
    # Closes the random reader, then asserts reading raises a failed-precondition error that
    # mentions the reader being closed.
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)
    reader.close()
    with self.assertRaisesRegex(errors_impl.FailedPreconditionError, r"closed"):
      reader.read(0)


class TFRecordWriterCloseAndFlushTests(test.TestCase):
  # [CLASS DOC]
  # Tests for writer close and flush behavior with no compression. Each test gets a fresh writer
  # created in setUp. Subclasses rerun the same tests with gzip and zlib compression.
  """TFRecordWriter close and flush tests"""

  # pylint: disable=arguments-differ
  def setUp(self, compression_type=TFRecordCompressionType.NONE):
    # [FUNCTION DOC]
    # Creates a file path in the temp directory, options for the given compression type (default
    # none) and an open writer on that file, and sets the record count to twenty. The extra
    # optional parameter lets subclasses choose the compression type.
    super(TFRecordWriterCloseAndFlushTests, self).setUp()
    self._fn = os.path.join(self.get_temp_dir(), "tf_record_writer_test.txt")
    self._options = tf_record.TFRecordOptions(compression_type)
    self._writer = tf_record.TFRecordWriter(self._fn, self._options)
    self._num_records = 20

  def _Record(self, r):
    # [FUNCTION DOC]
    # HELPER: returns deterministic bytes naming record index r.
    return compat.as_bytes("Record %d" % r)

  def testWriteAndLeaveOpen(self):
    # [FUNCTION DOC]
    # Writes all records without closing the writer; the test only verifies nothing crashes.
    records = list(map(self._Record, range(self._num_records)))
    for record in records:
      self._writer.write(record)

    # Verify no segfault if writer isn't explicitly closed.

  def testWriteAndRead(self):
    # [FUNCTION DOC]
    # Writes records, closes the writer, reads the file back and asserts the records match.
    records = list(map(self._Record, range(self._num_records)))
    for record in records:
      self._writer.write(record)
    self._writer.close()

    actual = list(tf_record.tf_record_iterator(self._fn, self._options))
    self.assertListEqual(actual, records)

  def testFlushAndRead(self):
    # [FUNCTION DOC]
    # Writes records and flushes without closing, then reads the file and asserts all records
    # are visible.
    records = list(map(self._Record, range(self._num_records)))
    for record in records:
      self._writer.write(record)
    self._writer.flush()

    actual = list(tf_record.tf_record_iterator(self._fn, self._options))
    self.assertListEqual(actual, records)

  def testDoubleClose(self):
    # [FUNCTION DOC]
    # Closing the writer twice must not raise an error.
    self._writer.write(self._Record(0))
    self._writer.close()
    self._writer.close()

  def testFlushAfterCloseIsError(self):
    # [FUNCTION DOC]
    # Asserts flushing a closed writer raises a failed-precondition error.
    self._writer.write(self._Record(0))
    self._writer.close()

    with self.assertRaises(errors_impl.FailedPreconditionError):
      self._writer.flush()

  def testWriteAfterCloseIsError(self):
    # [FUNCTION DOC]
    # Asserts writing to a closed writer raises a failed-precondition error.
    self._writer.write(self._Record(0))
    self._writer.close()

    with self.assertRaises(errors_impl.FailedPreconditionError):
      self._writer.write(self._Record(1))


class TFRecordWriterCloseAndFlushGzipTests(TFRecordWriterCloseAndFlushTests):
  # [CLASS DOC]
  # Reruns the inherited close and flush tests with gzip compression.
  # pylint: disable=arguments-differ
  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent setUp with the gzip compression type.
    super(TFRecordWriterCloseAndFlushGzipTests,
          self).setUp(TFRecordCompressionType.GZIP)


class TFRecordWriterCloseAndFlushZlibTests(TFRecordWriterCloseAndFlushTests):
  # [CLASS DOC]
  # Reruns the inherited close and flush tests with zlib compression.
  # pylint: disable=arguments-differ
  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent setUp with the zlib compression type.
    super(TFRecordWriterCloseAndFlushZlibTests,
          self).setUp(TFRecordCompressionType.ZLIB)


if __name__ == "__main__":
  test.main()
