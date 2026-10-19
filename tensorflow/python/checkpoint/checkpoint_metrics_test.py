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
# [MODULE DOC]
# Tests that saving and restoring checkpoints (the object-based V2 checkpoint and the V1
# checkpoint) update the checkpoint metrics: write-duration and read-duration histograms,
# training-time-saved, and checkpoint size counters. Test module; runs via the test main entry.
# ==============================================================================
"""Tests for checking the checkpoint reading and writing metrics."""

import os
import time

from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.eager import context
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.saved_model.pywrap_saved_model import metrics


class CheckpointMetricTests(test.TestCase):
  # [CLASS DOC]
  # Test case covering checkpoint metrics for the V2 and V1 checkpoint APIs, with small helpers
  # that read the global metric values for a given API label.

  def _get_write_histogram_proto(self, api_label):
    # [FUNCTION DOC]
    # HELPER: reads checkpoint write metrics for the tests.
    # Fetches the serialized write-duration histogram for the API label and returns it parsed
    # into a histogram message.
    proto_bytes = metrics.GetCheckpointWriteDurations(api_label=api_label)
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(proto_bytes)
    return histogram_proto

  def _get_read_histogram_proto(self, api_label):
    # [FUNCTION DOC]
    # HELPER: reads checkpoint read metrics for the tests.
    # Fetches the serialized read-duration histogram for the API label and returns it parsed
    # into a histogram message.
    proto_bytes = metrics.GetCheckpointReadDurations(api_label=api_label)
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(proto_bytes)
    return histogram_proto

  def _get_time_saved(self, api_label):
    # [FUNCTION DOC]
    # HELPER: reads the training-time-saved metric for the tests.
    # Returns the accumulated training time saved (number of seconds) for the API label.
    return metrics.GetTrainingTimeSaved(api_label=api_label)

  def _get_checkpoint_size(self, api_label, filesize):
    # [FUNCTION DOC]
    # HELPER: reads the checkpoint-size metric for the tests.
    # Returns the count recorded for the API label and the given file size.
    return metrics.GetCheckpointSize(api_label=api_label, filesize=filesize)

  def test_metrics_v2(self):
    # [FUNCTION DOC]
    # Scenario: in eager mode, create a V2 checkpoint holding one variable. Initially time saved
    # and write count are zero. Write the checkpoint three times (sleeping a second each time) and
    # assert the size counter grows by one each time and time saved increases. Then expect three
    # write durations and zero read durations; restore once and expect one read duration while
    # time saved stays unchanged (restoring in the same job does not add saved time).
    api_label = util._CHECKPOINT_V2
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')

    with context.eager_mode():
      ckpt = util.Checkpoint(v=variables_lib.Variable(1.))
      self.assertEqual(self._get_time_saved(api_label), 0.0)
      self.assertEqual(self._get_write_histogram_proto(api_label).num, 0.0)

      for i in range(3):
        time_saved = self._get_time_saved(api_label)
        time.sleep(1)
        ckpt_path = ckpt.write(file_prefix=prefix)
        filesize = util._get_checkpoint_size(ckpt_path)
        self.assertEqual(self._get_checkpoint_size(api_label, filesize), i + 1)
        self.assertGreater(self._get_time_saved(api_label), time_saved)

    self.assertEqual(self._get_write_histogram_proto(api_label).num, 3.0)
    self.assertEqual(self._get_read_histogram_proto(api_label).num, 0.0)

    time_saved = self._get_time_saved(api_label)
    with context.eager_mode():
      ckpt.restore(ckpt_path)
    self.assertEqual(self._get_read_histogram_proto(api_label).num, 1.0)
    # Restoring a checkpoint in the same "job" does not increase training time
    # saved.
    self.assertEqual(self._get_time_saved(api_label), time_saved)

  def test_metrics_v1(self):
    # [FUNCTION DOC]
    # Scenario: same as the V2 test but using the V1 checkpoint inside a cached session with an
    # initialized variable attached. Three writes raise the write count to three, the size counter
    # by one each time and time saved; a restore adds one read duration and leaves time saved
    # unchanged.
    api_label = util._CHECKPOINT_V1
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')

    with self.cached_session():
      ckpt = util.CheckpointV1()
      v = variables_lib.Variable(1.)
      self.evaluate(v.initializer)
      ckpt.v = v
      self.assertEqual(self._get_time_saved(api_label), 0.0)
      self.assertEqual(self._get_write_histogram_proto(api_label).num, 0.0)

      for i in range(3):
        time_saved = self._get_time_saved(api_label)
        time.sleep(1)
        ckpt_path = ckpt.write(file_prefix=prefix)
        filesize = util._get_checkpoint_size(ckpt_path)
        self.assertEqual(self._get_checkpoint_size(api_label, filesize), i + 1)
        self.assertGreater(self._get_time_saved(api_label), time_saved)

    self.assertEqual(self._get_write_histogram_proto(api_label).num, 3.0)

    self.assertEqual(self._get_read_histogram_proto(api_label).num, 0.0)
    time_saved = self._get_time_saved(api_label)
    ckpt.restore(ckpt_path)
    self.assertEqual(self._get_read_histogram_proto(api_label).num, 1.0)
    # Restoring a checkpoint in the same "job" does not increase training time
    # saved.
    self.assertEqual(self._get_time_saved(api_label), time_saved)


if __name__ == '__main__':
  test.main()
