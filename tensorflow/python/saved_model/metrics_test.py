# [MODULE DOC]
# Tests that SavedModel save and load code paths update the native instrumentation metrics:
# write/read counters per version, API usage counters, fingerprint, path and singleprint
# metrics, and the sharding callback metrics. Exposes one test case class; no import-time effects.
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
"""Tests for SavedModel instrumentation for Python reading/writing code.

These tests verify that the counters are incremented correctly after SavedModel
API calls.
"""

import os

from google.protobuf import json_format

from tensorflow.python.checkpoint.sharding import sharding_policies
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import autotrackable


class MetricsTests(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that saving and loading SavedModels (V1 builder, V2 save, V2 load, V1
  # loaded through V2, and the V1 loader) increment the global native metrics. Each test reads a
  # metric before the action and checks it afterwards. Helpers create models in the temp dir.

  def _create_save_v2_model(self):
    # [FUNCTION DOC]
    # HELPER: creates the SavedModel directory used by the metric tests.
    # Saves an empty trackable object with the V2 save API into a subdirectory of the test temp
    # dir and returns that directory path. Side effect: increments the V2 write metrics.
    root = autotrackable.AutoTrackable()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    return save_dir

  def _create_save_v1_model(self):
    # [FUNCTION DOC]
    # HELPER: creates the V1-format SavedModel used by the metric tests.
    # Uses the SavedModelBuilder in a graph with one constant, adds a meta graph tagged foo with
    # variables, saves, and returns the directory path. Side effect: increments V1 write metrics.
    save_dir = os.path.join(self.get_temp_dir(), "builder")
    builder_ = builder.SavedModelBuilder(save_dir)

    with ops.Graph().as_default():
      with self.session(graph=ops.Graph()) as sess:
        constant_op.constant(5.0)
        builder_.add_meta_graph_and_variables(sess, ["foo"])
      builder_.save()
    return save_dir

  def test_python_save(self):
    # [FUNCTION DOC]
    # Scenario: records the version 2 write counter and the V2 save API counter, saves a V2
    # model, and asserts both counters increased by exactly one.
    write_count = metrics.GetWrite(write_version="2")
    save_api_count = metrics.GetWriteApi(save._SAVE_V2_LABEL)
    _ = self._create_save_v2_model()

    self.assertEqual(
        metrics.GetWriteApi(save._SAVE_V2_LABEL), save_api_count + 1)
    self.assertEqual(metrics.GetWrite(write_version="2"), write_count + 1)

  def test_builder_save(self):
    # [FUNCTION DOC]
    # Scenario: records the version 1 write counter and the builder save API counter, saves a V1
    # model with the builder, and asserts both counters increased by exactly one.
    write_count = metrics.GetWrite(write_version="1")
    save_builder_count = metrics.GetWriteApi(builder_impl._SAVE_BUILDER_LABEL)
    _ = self._create_save_v1_model()

    self.assertEqual(
        metrics.GetWriteApi(builder_impl._SAVE_BUILDER_LABEL),
        save_builder_count + 1)
    self.assertEqual(metrics.GetWrite(write_version="1"), write_count + 1)

  def test_load_v2(self):
    # [FUNCTION DOC]
    # Scenario: saves a V2 model, records the version 2 read counter and the V2 load API
    # counter, loads the model, and asserts both counters increased by exactly one.
    save_dir = self._create_save_v2_model()

    read_count = metrics.GetRead(write_version="2")
    load_v2_count = metrics.GetReadApi(load._LOAD_V2_LABEL)
    load.load(save_dir)

    self.assertEqual(metrics.GetReadApi(load._LOAD_V2_LABEL), load_v2_count + 1)
    self.assertEqual(metrics.GetRead(write_version="2"), read_count + 1)

  def test_load_v1_in_v2(self):
    # [FUNCTION DOC]
    # Scenario: saves a V1 model and loads it through the V2 load entry point. Asserts the V1
    # read counter and the V1-in-V2 API counter went up by one, while the plain V2 load API
    # counter and the version 2 read counter did not change.
    save_dir = self._create_save_v1_model()
    read_v1_count = metrics.GetRead(write_version="1")
    read_v2_count = metrics.GetRead(write_version="2")
    load_v2_count = metrics.GetReadApi(load._LOAD_V2_LABEL)
    load_v1_v2_count = metrics.GetReadApi(load_v1_in_v2._LOAD_V1_V2_LABEL)
    load.load(save_dir)

    # Check that `load_v2` was *not* incremented.
    self.assertEqual(metrics.GetReadApi(load._LOAD_V2_LABEL), load_v2_count)
    self.assertEqual(metrics.GetRead(write_version="2"), read_v2_count)

    self.assertEqual(
        metrics.GetReadApi(load_v1_in_v2._LOAD_V1_V2_LABEL),
        load_v1_v2_count + 1)
    self.assertEqual(metrics.GetRead(write_version="1"), read_v1_count + 1)

  def test_loader_v1(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disables eager execution globally and re-enables it only after the load; if
    # loading raises, eager execution stays disabled for later tests. It also asserts the loader
    # API counter equals an absolute value of one rather than an increment over a baseline, so it
    # depends on no other test having used the V1 loader earlier in the process.
    # Scenario: saves a V1 model, loads it with the V1 SavedModelLoader in a session with the
    # foo tag, and asserts the loader API counter is one and the V1 read counter went up by one.
    ops.disable_eager_execution()
    save_dir = self._create_save_v1_model()

    read_count = metrics.GetRead(write_version="1")
    loader = loader_impl.SavedModelLoader(save_dir)
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"])
    ops.enable_eager_execution()

    self.assertEqual(metrics.GetReadApi(loader_impl._LOADER_LABEL), 1)
    self.assertEqual(metrics.GetRead(write_version="1"), read_count + 1)

  def test_save_sets_write_fingerprint_metric(self):
    # [FUNCTION DOC]
    # Scenario: saves a V2 model, reads its fingerprint from disk, parses the write fingerprint
    # metric (a JSON string) into a fingerprint, and asserts the two fingerprints are equal.
    exported_dir = self._create_save_v2_model()
    fingerprint = fingerprinting.read_fingerprint(exported_dir)
    fingerprint_metric = fingerprinting.Fingerprint.from_proto(
        json_format.Parse(metrics.GetWriteFingerprint(),
                          fingerprinting.fingerprint_pb2.FingerprintDef()))
    self.assertEqual(fingerprint, fingerprint_metric)

  def test_load_sets_read_fingerprint_metric(self):
    # [FUNCTION DOC]
    # Scenario: saves and then loads a V2 model, reads its fingerprint from disk, parses the read
    # fingerprint metric (a JSON string), and asserts it equals the on-disk fingerprint.
    exported_dir = self._create_save_v2_model()
    load.load(exported_dir)
    fingerprint = fingerprinting.read_fingerprint(exported_dir)
    fingerprint_metric = fingerprinting.Fingerprint.from_proto(
        json_format.Parse(metrics.GetReadFingerprint(),
                          fingerprinting.fingerprint_pb2.FingerprintDef()))
    self.assertEqual(fingerprint, fingerprint_metric)

  def test_save_sets_write_path_metric(self):
    # [FUNCTION DOC]
    # Scenario: saves a V2 model and asserts the write path metric equals the export directory.
    exported_dir = self._create_save_v2_model()

    self.assertEqual(metrics.GetWritePath(), exported_dir)

  def test_load_sets_read_path_metric(self):
    # [FUNCTION DOC]
    # Scenario: saves then loads a V2 model and asserts the read path metric equals the export
    # directory.
    exported_dir = self._create_save_v2_model()
    load.load(exported_dir)

    self.assertEqual(metrics.GetReadPath(), exported_dir)

  def test_save_sets_write_path_and_singleprint_metric(self):
    # [FUNCTION DOC]
    # Scenario: saves a V2 model, computes the singleprint of its on-disk fingerprint, and
    # asserts the write path-and-singleprint metric is the pair of export directory and
    # singleprint.
    exported_dir = self._create_save_v2_model()
    singleprint = fingerprinting.read_fingerprint(exported_dir).singleprint()
    path_and_singleprint_metric = metrics.GetWritePathAndSingleprint()
    self.assertEqual(path_and_singleprint_metric, (exported_dir, singleprint))

  def test_save_sets_read_path_and_singleprint_metric(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says save but the test saves and then loads the model, and checks
    # the read metric, so it really covers loading.
    # Scenario: saves and loads a V2 model, computes the singleprint of the on-disk fingerprint,
    # and asserts the read path-and-singleprint metric is the pair of directory and singleprint.
    exported_dir = self._create_save_v2_model()
    load.load(exported_dir)
    singleprint = fingerprinting.read_fingerprint(exported_dir).singleprint()
    path_and_singleprint_metric = metrics.GetReadPathAndSingleprint()
    self.assertEqual(path_and_singleprint_metric, (exported_dir, singleprint))

  def test_save_sets_sharding_callback_duration_metric(self):
    # [FUNCTION DOC]
    # Scenario: saves a V2 model and asserts the sharding callback duration metric is positive.
    self._create_save_v2_model()
    sharding_callback_duration_metric = metrics.GetShardingCallbackDuration()
    self.assertGreater(sharding_callback_duration_metric, 0)

  def test_save_sets_num_checkpoint_shards_written_metric(self):
    # [FUNCTION DOC]
    # Scenario: saves a V2 model and asserts the number of checkpoint shards written metric is
    # greater than zero.
    self._create_save_v2_model()
    num_shards_written_metric = metrics.GetNumCheckpointShardsWritten()
    self.assertGreater(num_shards_written_metric, 0)

  def test_save_sets_sharding_callback_description_metric(self):
    # [FUNCTION DOC]
    # Scenario: saves a V2 model and asserts the sharding callback description metric equals the
    # description of the default shard-by-task policy.
    self._create_save_v2_model()
    callback_description_metric = metrics.GetShardingCallbackDescription()
    self.assertEqual(callback_description_metric,
                     sharding_policies.ShardByTaskPolicy().description)


if __name__ == "__main__":
  test.main()
