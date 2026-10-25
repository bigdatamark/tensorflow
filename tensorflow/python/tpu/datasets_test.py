# [MODULE DOC]
# Tests for the TPU streaming-files dataset helper (the datasets module in this package).
# Each test starts two local in-process servers (a worker and a coordinator), builds a
# streaming dataset over temporary files, reads it through a session bound to the worker and
# checks that the values read match what was written, or that bad arguments are rejected.
# Defines two module constants for the number of files and entries per file. No import-time
# side effects beyond imports. Runs under the TensorFlow test main entry point.
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
"""TPU datasets tests."""

import os

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.lib.io import python_io
from tensorflow.python.platform import test
from tensorflow.python.tpu import datasets
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

_NUM_FILES = 10
_NUM_ENTRIES = 20


class DatasetsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the streaming files dataset from the TPU datasets module. The fixture
  # (setUp) creates a local worker server, a local coordinator server, a cluster definition
  # naming both, a session on the worker and the worker device name; tests use these to
  # iterate the dataset on the worker. Covers text files, record files, a dataset of file
  # names, custom reader functions and invalid argument handling.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture. Takes no inputs and returns nothing. Calls the parent setUp, starts two
    # local servers (worker and coordinator), builds a cluster definition with a worker job and
    # a coordinator job each pointing at its server address (scheme prefix stripped), creates
    # a session connected to the worker with that cluster configuration, and stores the
    # worker device name string. Side effects: starts servers and stores state on the instance.
    super(DatasetsTest, self).setUp()
    self._coord = server_lib.Server.create_local_server()
    self._worker = server_lib.Server.create_local_server()

    self._cluster_def = cluster_pb2.ClusterDef()
    worker_job = self._cluster_def.job.add()
    worker_job.name = 'worker'
    worker_job.tasks[0] = self._worker.target[len('grpc://'):]
    coord_job = self._cluster_def.job.add()
    coord_job.name = 'coordinator'
    coord_job.tasks[0] = self._coord.target[len('grpc://'):]

    session_config = config_pb2.ConfigProto(cluster_def=self._cluster_def)

    self._sess = session.Session(self._worker.target, config=session_config)
    self._worker_device = '/job:' + worker_job.name

  def testTextLineDataset(self):
    # [FUNCTION DOC]
    # Test: writes several temporary text files, each holding many newline separated entries,
    # and builds a streaming dataset over a wildcard pattern with the text file type. Reads
    # four times the total entry count from an initialized iterator on the worker device.
    # Asserts the set of values read equals the set of values written (repeats are allowed).
    all_contents = []
    for i in range(_NUM_FILES):
      filename = os.path.join(self.get_temp_dir(), 'text_line.%d.txt' % i)
      contents = []
      for j in range(_NUM_ENTRIES):
        contents.append(compat.as_bytes('%d: %d' % (i, j)))
      with open(filename, 'wb') as f:
        f.write(b'\n'.join(contents))
      all_contents.extend(contents)

    dataset = datasets.StreamingFilesDataset(
        os.path.join(self.get_temp_dir(), 'text_line.*.txt'), filetype='text')

    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(4 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))

    self.assertEqual(set(all_contents), set(retrieved_values))

  def testTFRecordDataset(self):
    # [FUNCTION DOC]
    # Test: writes several temporary record files with a record writer, then builds a streaming
    # dataset over a wildcard pattern with the tfrecord file type. Reads four times the total
    # record count through an iterator on the worker device and asserts the set of values read
    # equals the set of records written.
    all_contents = []
    for i in range(_NUM_FILES):
      filename = os.path.join(self.get_temp_dir(), 'tf_record.%d' % i)
      writer = python_io.TFRecordWriter(filename)
      for j in range(_NUM_ENTRIES):
        record = compat.as_bytes('Record %d of file %d' % (j, i))
        writer.write(record)
        all_contents.append(record)
      writer.close()

    dataset = datasets.StreamingFilesDataset(
        os.path.join(self.get_temp_dir(), 'tf_record*'), filetype='tfrecord')

    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(4 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))

    self.assertEqual(set(all_contents), set(retrieved_values))

  def testTFRecordDatasetFromDataset(self):
    # [FUNCTION DOC]
    # Test: like the record dataset test, but the file names are passed as a dataset of strings
    # (built from the list of names) instead of a pattern. Reads four times the record count
    # and asserts the set of values read equals the set of records written.
    filenames = []
    all_contents = []
    for i in range(_NUM_FILES):
      filename = os.path.join(self.get_temp_dir(), 'tf_record.%d' % i)
      filenames.append(filename)
      writer = python_io.TFRecordWriter(filename)
      for j in range(_NUM_ENTRIES):
        record = compat.as_bytes('Record %d of file %d' % (j, i))
        writer.write(record)
        all_contents.append(record)
      writer.close()

    filenames = dataset_ops.Dataset.from_tensor_slices(filenames)

    dataset = datasets.StreamingFilesDataset(filenames, filetype='tfrecord')

    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(4 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))

    self.assertEqual(set(all_contents), set(retrieved_values))

  def testArbitraryReaderFunc(self):
    # [FUNCTION DOC]
    # Test: writes files of fixed length records and passes a custom reader function (building
    # a fixed length record dataset) as the file type. Reads four times the record count and
    # asserts the set of values read equals the set of records written.

    def MakeRecord(i, j):
      # [FUNCTION DOC]
      # HELPER: builds fixed width test records for the reader function test.
      # Takes a file index and entry index and returns a byte string of both numbers padded
      # to four digits and joined with a dash. Pure, no side effects.
      return compat.as_bytes('%04d-%04d' % (i, j))

    record_bytes = len(MakeRecord(10, 200))

    all_contents = []
    for i in range(_NUM_FILES):
      filename = os.path.join(self.get_temp_dir(), 'fixed_length.%d' % i)
      with open(filename, 'wb') as f:
        for j in range(_NUM_ENTRIES):
          record = MakeRecord(i, j)
          f.write(record)
          all_contents.append(record)

    def FixedLengthFile(filename):
      # [FUNCTION DOC]
      # HELPER: custom reader function passed to the streaming dataset as the file type.
      # Takes a file name and returns a fixed length record dataset over it using the record
      # size computed in the enclosing test. No side effects.
      return readers.FixedLengthRecordDataset(filename, record_bytes)

    dataset = datasets.StreamingFilesDataset(
        os.path.join(self.get_temp_dir(), 'fixed_length*'),
        filetype=FixedLengthFile)

    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(4 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))

    self.assertEqual(set(all_contents), set(retrieved_values))

  def testArbitraryReaderFuncFromDatasetGenerator(self):
    # [FUNCTION DOC]
    # Test: uses a dataset of integers as the input and a reader function that returns a
    # generator based dataset yielding a pair of a scalar and a ten element vector. Takes one
    # element from the iterator and asserts it is a list or tuple of two items whose first is
    # one and whose second holds ten ones.

    def my_generator():
      # [FUNCTION DOC]
      # HELPER: generator for the test dataset. Takes nothing and yields a single pair: the
      # integer one and a list of ten ones.
      yield (1, [1] * 10)

    def gen_dataset(dummy):
      # [FUNCTION DOC]
      # HELPER: reader function for the test. Ignores its argument and returns a dataset built
      # from the generator with int64 element types and shapes scalar and length ten.
      return dataset_ops.Dataset.from_generator(
          my_generator, (dtypes.int64, dtypes.int64),
          (tensor_shape.TensorShape([]), tensor_shape.TensorShape([10])))

    dataset = datasets.StreamingFilesDataset(
        dataset_ops.Dataset.range(10), filetype=gen_dataset)

    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = self._sess.run(get_next)

    self.assertIsInstance(retrieved_values, (list, tuple))
    self.assertEqual(len(retrieved_values), 2)
    self.assertEqual(retrieved_values[0], 1)
    self.assertItemsEqual(retrieved_values[1], [1] * 10)

  def testUnexpectedFiletypeString(self):
    # [FUNCTION DOC]
    # Test: constructing the streaming dataset with an unknown file type string must raise a
    # value error.
    with self.assertRaises(ValueError):
      datasets.StreamingFilesDataset(
          os.path.join(self.get_temp_dir(), '*'), filetype='foo')

  def testUnexpectedFiletypeType(self):
    # [FUNCTION DOC]
    # Test: constructing the streaming dataset with a file type that is an integer (neither a
    # string nor a callable) must raise a value error.
    with self.assertRaises(ValueError):
      datasets.StreamingFilesDataset(
          os.path.join(self.get_temp_dir(), '*'), filetype=3)

  def testUnexpectedFilesType(self):
    # [FUNCTION DOC]
    # Test: constructing the streaming dataset with an integer as the files argument (not a
    # pattern, list or dataset) must raise a value error.
    with self.assertRaises(ValueError):
      datasets.StreamingFilesDataset(123, filetype='tfrecord')


if __name__ == '__main__':
  test.main()
