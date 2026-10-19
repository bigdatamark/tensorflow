# [MODULE DOC]
# Unit tests for the speech commands models module. Checks that model settings can be prepared and
# that models of several architectures (conv, low_latency_conv, single_fc, tiny_conv) can be built
# in training and inference modes, and that an unknown architecture name is rejected.
# Runs under the TensorFlow test runner when executed as a script. No import-time side effects.
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
"""Tests for speech commands models."""

import tensorflow as tf

from tensorflow.examples.speech_commands import models
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ModelsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering model-settings preparation and model construction in the speech commands
  # models module. Shares a helper that builds a standard settings mapping (10 labels, 16 kHz,
  # 1000 ms clips, 20 ms window, 10 ms stride, 40 feature bins, mfcc preprocessing).

  def _modelSettings(self):
    # [FUNCTION DOC]
    # HELPER: builds the shared model settings used by the tests in this class.
    # Takes no arguments. Returns the settings mapping produced by preparing model settings with
    # fixed values (10 labels, 16000 Hz, 1000 ms clip, 20 ms window, 10 ms stride, 40 feature
    # bins, mfcc preprocessing). No side effects.
    return models.prepare_model_settings(
        label_count=10,
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=20,
        window_stride_ms=10,
        feature_bin_count=40,
        preprocess="mfcc")

  def testPrepareModelSettings(self):
    # [FUNCTION DOC]
    # Scenario: prepare model settings with typical audio parameters (10 labels, 16 kHz, 1 s clip,
    # 20 ms window, 10 ms stride, 40 bins, mfcc). Asserts that a non-null result is returned.
    self.assertIsNotNone(
        models.prepare_model_settings(
            label_count=10,
            sample_rate=16000,
            clip_duration_ms=1000,
            window_size_ms=20,
            window_stride_ms=10,
            feature_bin_count=40,
            preprocess="mfcc"))

  @test_util.run_deprecated_v1
  def testCreateModelConvTraining(self):
    # [FUNCTION DOC]
    # Scenario: build the "conv" architecture in training mode on a zero fingerprint input of batch
    # size one, inside a cached session (legacy graph mode). Asserts that logits and a dropout-rate
    # placeholder are returned and that both can be looked up by name in the session graph.
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_rate = models.create_model(
          fingerprint_input, model_settings, "conv", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_rate)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_rate.name))

  @test_util.run_deprecated_v1
  def testCreateModelConvInference(self):
    # [FUNCTION DOC]
    # Scenario: build the "conv" architecture in inference mode (not training) on a zero
    # fingerprint input. Asserts that only logits are returned (no dropout rate) and that the
    # logits tensor can be found by name in the session graph.
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits = models.create_model(fingerprint_input, model_settings, "conv",
                                   False)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))

  @test_util.run_deprecated_v1
  def testCreateModelLowLatencyConvTraining(self):
    # [FUNCTION DOC]
    # Scenario: build the "low_latency_conv" architecture in training mode on a zero fingerprint
    # input. Asserts logits and dropout rate are returned and both exist by name in the graph.
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_rate = models.create_model(
          fingerprint_input, model_settings, "low_latency_conv", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_rate)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_rate.name))

  @test_util.run_deprecated_v1
  def testCreateModelFullyConnectedTraining(self):
    # [FUNCTION DOC]
    # Scenario: build the "single_fc" (single fully connected layer) architecture in training mode
    # on a zero fingerprint input. Asserts logits and dropout rate are returned and both exist by
    # name in the graph.
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_rate = models.create_model(
          fingerprint_input, model_settings, "single_fc", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_rate)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_rate.name))

  def testCreateModelBadArchitecture(self):
    # [FUNCTION DOC]
    # Scenario: request a model with an unknown architecture name in training mode. Asserts that an
    # exception is raised and that its message contains the text "not recognized". Not marked as a
    # legacy graph-mode test.
    model_settings = self._modelSettings()
    with self.cached_session():
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      with self.assertRaises(Exception) as e:
        models.create_model(fingerprint_input, model_settings,
                            "bad_architecture", True)
      self.assertIn("not recognized", str(e.exception))

  @test_util.run_deprecated_v1
  def testCreateModelTinyConvTraining(self):
    # [FUNCTION DOC]
    # Scenario: build the "tiny_conv" architecture in training mode on a zero fingerprint input.
    # Asserts logits and dropout rate are returned and both exist by name in the graph.
    model_settings = self._modelSettings()
    with self.cached_session() as sess:
      fingerprint_input = tf.zeros([1, model_settings["fingerprint_size"]])
      logits, dropout_rate = models.create_model(
          fingerprint_input, model_settings, "tiny_conv", True)
      self.assertIsNotNone(logits)
      self.assertIsNotNone(dropout_rate)
      self.assertIsNotNone(sess.graph.get_tensor_by_name(logits.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(dropout_rate.name))


if __name__ == "__main__":
  test.main()
