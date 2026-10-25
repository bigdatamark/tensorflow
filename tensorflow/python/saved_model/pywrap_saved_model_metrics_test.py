# [MODULE DOC]
# Tests for the Python bindings of the native SavedModel and checkpoint metrics: counters for
# reads and writes and their API labels, checkpoint write/read duration histograms, saved
# training time, checkpoint size, file size calculation, fingerprint, path and singleprint
# metrics (with argument validation), found-fingerprint status, and sharding callback metrics.
# Each test first asserts the metric starts at its empty value; the metrics are process-global.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SavedModel and Checkpoint metrics Python bindings."""

import os

from tensorflow.core.framework import summary_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.eager import test
from tensorflow.python.saved_model.pywrap_saved_model import metrics


class MetricsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the native metrics bindings. Each test checks the initial value of a
  # process-global metric, updates it through the binding, and checks the new value. Two helper
  # methods build histogram and fingerprint inputs. Tests depend on the metrics not having been
  # touched earlier in the same process.

  def _get_histogram_proto(self, proto_bytes):
    # [FUNCTION DOC]
    # HELPER: decodes histogram data returned by the metrics bindings.
    # Input: proto_bytes, serialized histogram bytes. Returns the parsed histogram message.
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(proto_bytes)
    return histogram_proto

  def _get_serialized_fingerprint_def(self):
    # [FUNCTION DOC]
    # HELPER: builds sample fingerprint input for the fingerprint metric tests.
    # Returns the serialized bytes of a fingerprint message with checksum 1, program hash 2,
    # signature hash 3, object graph hash 4, checkpoint hash 5 and producer version 6.
    return fingerprint_pb2.FingerprintDef(
        saved_model_checksum=1,
        graph_def_program_hash=2,
        signature_def_hash=3,
        saved_object_graph_hash=4,
        checkpoint_hash=5,
        version=versions_pb2.VersionDef(producer=6)).SerializeToString()

  def test_SM_increment_write(self):
    # [FUNCTION DOC]
    # Scenario: the version 1 write counter starts at zero; incrementing the write API counter
    # labelled foo makes it one, and incrementing the version 1 write counter makes that one.
    self.assertEqual(metrics.GetWrite(write_version="1"), 0)
    metrics.IncrementWriteApi("foo")
    self.assertEqual(metrics.GetWriteApi("foo"), 1)
    metrics.IncrementWrite(write_version="1")
    self.assertEqual(metrics.GetWrite(write_version="1"), 1)

  def test_SM_increment_read(self):
    # [FUNCTION DOC]
    # Scenario: the version 2 read counter starts at zero; incrementing the read API counter
    # labelled bar makes it one, and incrementing the version 2 read counter makes that one.
    self.assertEqual(metrics.GetRead(write_version="2"), 0)
    metrics.IncrementReadApi("bar")
    self.assertEqual(metrics.GetReadApi("bar"), 1)
    metrics.IncrementRead(write_version="2")
    self.assertEqual(metrics.GetRead(write_version="2"), 1)

  def test_checkpoint_add_write_duration(self):
    # [FUNCTION DOC]
    # Scenario: the checkpoint write duration histogram for label foo starts empty; after adding
    # 100 and 200 microseconds it has two samples with minimum 100 and maximum 200.
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointWriteDurations(api_label="foo")).num, 0)

    metrics.AddCheckpointWriteDuration(api_label="foo", microseconds=100)
    metrics.AddCheckpointWriteDuration(api_label="foo", microseconds=200)

    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointWriteDurations(api_label="foo")).num, 2)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointWriteDurations(api_label="foo")).min, 100)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointWriteDurations(api_label="foo")).max, 200)

  def test_async_checkpoint_add_write_duration(self):
    # [FUNCTION DOC]
    # Scenario: the asynchronous checkpoint write duration histogram for label foo starts empty;
    # after adding 20 and 50 microseconds it has two samples with minimum 20 and maximum 50.
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetAsyncCheckpointWriteDurations(api_label="foo")).num, 0)

    metrics.AddAsyncCheckpointWriteDuration(api_label="foo", microseconds=20)
    metrics.AddAsyncCheckpointWriteDuration(api_label="foo", microseconds=50)

    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetAsyncCheckpointWriteDurations(api_label="foo")).num, 2)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetAsyncCheckpointWriteDurations(api_label="foo")).min, 20)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetAsyncCheckpointWriteDurations(api_label="foo")).max, 50)

  def test_checkpoint_add_read_duration(self):
    # [FUNCTION DOC]
    # Scenario: the checkpoint read duration histogram for label bar starts empty; after adding
    # 200 and 20000 microseconds it has two samples with minimum 200 and maximum 20000.
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointReadDurations(api_label="bar")).num, 0)

    metrics.AddCheckpointReadDuration(api_label="bar", microseconds=200)
    metrics.AddCheckpointReadDuration(api_label="bar", microseconds=20000)

    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointReadDurations(api_label="bar")).num, 2)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointReadDurations(api_label="bar")).min, 200)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointReadDurations(api_label="bar")).max, 20000)

  def test_training_time_saved(self):
    # [FUNCTION DOC]
    # Scenario: saved training time for label baz starts at zero; adding 1000 microseconds makes
    # the stored value 1000.
    self.assertEqual(metrics.GetTrainingTimeSaved(api_label="baz"), 0)
    metrics.AddTrainingTimeSaved(api_label="baz", microseconds=1000)
    self.assertEqual(metrics.GetTrainingTimeSaved(api_label="baz"), 1000)

  def test_checkpoint_size(self):
    # [FUNCTION DOC]
    # Scenario: the checkpoint size counter for label baz and size 100 starts at zero; recording
    # that size twice makes the count two.
    self.assertEqual(
        metrics.GetCheckpointSize(api_label="baz", filesize=100), 0)
    metrics.RecordCheckpointSize(api_label="baz", filesize=100)
    metrics.RecordCheckpointSize(api_label="baz", filesize=100)
    self.assertEqual(
        metrics.GetCheckpointSize(api_label="baz", filesize=100), 2)

  def test_filesize(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test name suggests a size check, but the file written holds eight bytes
    # of text while the assertion expects the calculated size to be zero; the expectation depends
    # on how the native function treats this file and is not explained in the test.
    # Scenario: writes a small text file into the temp dir and asserts the file size calculation
    # returns zero for it.
    filename = os.path.join(self.get_temp_dir(), "test.txt")
    with open(filename, "w") as file:
      file.write("Hello! \n")
    self.assertEqual(metrics.CalculateFileSize(filename), 0)

  def test_invalid_file(self):
    # [FUNCTION DOC]
    # Scenario: the file size calculation for a path that does not exist returns minus one.
    self.assertEqual(metrics.CalculateFileSize("not_a_file.txt"), -1)

  def test_SM_read_fingerprint(self):
    # [FUNCTION DOC]
    # Scenario: the read fingerprint metric starts as an empty string; after setting it from the
    # sample serialized fingerprint, the returned JSON text contains the five hash fields with
    # values 1 through 5.
    self.assertEqual(metrics.GetReadFingerprint(), "")
    metrics.SetReadFingerprint(
        fingerprint=self._get_serialized_fingerprint_def())
    read_fingerprint = metrics.GetReadFingerprint()
    self.assertIn('"saved_model_checksum" : 1', read_fingerprint)
    self.assertIn('"graph_def_program_hash" : 2', read_fingerprint)
    self.assertIn('"signature_def_hash" : 3', read_fingerprint)
    self.assertIn('"saved_object_graph_hash" : 4', read_fingerprint)
    self.assertIn('"checkpoint_hash" : 5', read_fingerprint)

  def test_SM_write_fingerprint(self):
    # [FUNCTION DOC]
    # Scenario: the write fingerprint metric starts as an empty string; after setting it from
    # the sample serialized fingerprint, the JSON text contains the five hash fields 1 to 5.
    self.assertEqual(metrics.GetWriteFingerprint(), "")
    metrics.SetWriteFingerprint(
        fingerprint=self._get_serialized_fingerprint_def())
    write_fingerprint = metrics.GetWriteFingerprint()
    self.assertIn('"saved_model_checksum" : 1', write_fingerprint)
    self.assertIn('"graph_def_program_hash" : 2', write_fingerprint)
    self.assertIn('"signature_def_hash" : 3', write_fingerprint)
    self.assertIn('"saved_object_graph_hash" : 4', write_fingerprint)
    self.assertIn('"checkpoint_hash" : 5', write_fingerprint)

  def test_SM_read_path(self):
    # [FUNCTION DOC]
    # Scenario: the read path metric starts empty; after setting it to foo it returns foo.
    self.assertEqual(metrics.GetReadPath(), "")
    metrics.SetReadPath(saved_model_path="foo")
    self.assertEqual(metrics.GetReadPath(), "foo")

  def test_SM_write_path(self):
    # [FUNCTION DOC]
    # Scenario: the write path metric starts empty; after setting it to foo it returns foo.
    self.assertEqual(metrics.GetWritePath(), "")
    metrics.SetWritePath(saved_model_path="foo")
    self.assertEqual(metrics.GetWritePath(), "foo")

  def test_SM_read_path_and_singleprint(self):
    # [FUNCTION DOC]
    # Scenario: the read path-and-singleprint metric starts as a pair of empty strings; after
    # setting path foo and singleprint bar it returns that pair.
    self.assertEqual(metrics.GetReadPathAndSingleprint(), ("", ""))
    metrics.SetReadPathAndSingleprint(path="foo", singleprint="bar")
    self.assertEqual(metrics.GetReadPathAndSingleprint(), ("foo", "bar"))

  def test_SM_read_invalid_path_and_singleprint(self):
    # [FUNCTION DOC]
    # Scenario: setting the read path-and-singleprint metric with an empty path, and then with
    # an empty singleprint, each raises the metric exception whose message names the empty
    # argument.
    with self.assertRaises(metrics.MetricException) as excinfo:
      metrics.SetReadPathAndSingleprint(path="", singleprint="bar")
    self.assertRegex(str(excinfo.exception),
                     "Invalid path_and_singleprint argument. Empty path.")

    with self.assertRaises(metrics.MetricException) as excinfo:
      metrics.SetReadPathAndSingleprint(path="foo", singleprint="")
    self.assertRegex(
        str(excinfo.exception),
        "Invalid path_and_singleprint argument. Empty singleprint.")

  def test_SM_write_path_and_singleprint(self):
    # [FUNCTION DOC]
    # Scenario: the write path-and-singleprint metric starts as a pair of empty strings; after
    # setting path foo and singleprint bar it returns that pair.
    self.assertEqual(metrics.GetWritePathAndSingleprint(), ("", ""))
    metrics.SetWritePathAndSingleprint(path="foo", singleprint="bar")
    self.assertEqual(metrics.GetWritePathAndSingleprint(), ("foo", "bar"))

  def test_SM_write_invalid_path_and_singleprint(self):
    # [FUNCTION DOC]
    # Scenario: setting the write path-and-singleprint metric with an empty path, and then with
    # an empty singleprint, each raises the metric exception whose message names the empty
    # argument.
    with self.assertRaises(metrics.MetricException) as excinfo:
      metrics.SetWritePathAndSingleprint(path="", singleprint="bar")
    self.assertRegex(str(excinfo.exception),
                     "Invalid path_and_singleprint argument. Empty path.")

    with self.assertRaises(metrics.MetricException) as excinfo:
      metrics.SetWritePathAndSingleprint(path="foo", singleprint="")
    self.assertRegex(
        str(excinfo.exception),
        "Invalid path_and_singleprint argument. Empty singleprint.")

  def test_SM_found_fingerprint_on_load(self):
    # [FUNCTION DOC]
    # Scenario: setting the found-fingerprint status to each of the found, not-found and error
    # constants makes the metric read back FOUND, NOT_FOUND and ERROR respectively.
    metrics.SetFoundFingerprintOnLoad(found_status=metrics.kFingerprintFound)
    self.assertEqual(metrics.GetFoundFingerprintOnLoad(), "FOUND")

    metrics.SetFoundFingerprintOnLoad(found_status=metrics.kFingerprintNotFound)
    self.assertEqual(metrics.GetFoundFingerprintOnLoad(), "NOT_FOUND")

    metrics.SetFoundFingerprintOnLoad(found_status=metrics.kFingerprintError)
    self.assertEqual(metrics.GetFoundFingerprintOnLoad(), "ERROR")

  def test_invalid_SM_found_fingerprint_on_load(self):
    # [FUNCTION DOC]
    # Scenario: setting the found-fingerprint status to an unknown string, and to a lowercase
    # variant of a valid one, results in the metric reading back as an empty string (the value
    # is not an accepted constant, so it is not recorded). No exception is raised.
    metrics.SetFoundFingerprintOnLoad(found_status="absolute nonsense")
    self.assertEqual(metrics.GetFoundFingerprintOnLoad(), "")

    metrics.SetFoundFingerprintOnLoad(found_status="found")
    self.assertEqual(metrics.GetFoundFingerprintOnLoad(), "")

  def test_checkpoint_sharding_callback_duration(self):
    # [FUNCTION DOC]
    # Scenario: the sharding callback duration starts at zero; adding a duration of 100 makes it
    # 100.
    self.assertEqual(metrics.GetShardingCallbackDuration(), 0)
    metrics.AddShardingCallbackDuration(callback_duration=100)
    self.assertEqual(metrics.GetShardingCallbackDuration(), 100)

  def test_num_checkpoint_shards_written(self):
    # [FUNCTION DOC]
    # Scenario: the number of checkpoint shards written starts at zero; adding ten makes it ten.
    self.assertEqual(metrics.GetNumCheckpointShardsWritten(), 0)
    metrics.AddNumCheckpointShardsWritten(num_shards=10)
    self.assertEqual(metrics.GetNumCheckpointShardsWritten(), 10)

  def test_sharding_callback_description(self):
    # [FUNCTION DOC]
    # Scenario: the sharding callback description starts as an empty string; after setting it to
    # foo it returns foo.
    self.assertEqual(metrics.GetShardingCallbackDescription(), "")
    metrics.SetShardingCallbackDescription(description="foo")
    self.assertEqual(metrics.GetShardingCallbackDescription(), "foo")


if __name__ == "__main__":
  test.main()
