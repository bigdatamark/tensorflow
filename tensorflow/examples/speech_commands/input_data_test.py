# [MODULE DOC]
# Unit tests for the speech commands input_data module: word list preparation, stable partition
# assignment, data index and background data preparation, wav load/save, processing graph
# construction, batch retrieval for each preprocessing mode, unprocessed data retrieval, feature
# extraction for a wav, and feature ranges. Tests write synthetic silent wav files into
# temporary directories.
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
"""Tests for data input for speech commands."""

import os

import numpy as np
import tensorflow as tf


from tensorflow.examples.speech_commands import input_data
from tensorflow.examples.speech_commands import models
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class InputDataTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the input_data module. Shared fixtures are helper methods that create
  # encoded silent wav bytes, write wav files, build folders of label-named wav files, and give a
  # default model settings mapping. Each test uses the framework temporary directory.

  def _getWavData(self):
    # [FUNCTION DOC]
    # HELPER: fixture creating wav bytes for the tests.
    # Encodes 32000 frames of two-channel silence at 16000 Hz and returns the encoded bytes.
    with self.cached_session():
      sample_data = tf.zeros([32000, 2])
      wav_encoder = tf.audio.encode_wav(sample_data, 16000)
      wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _saveTestWavFile(self, filename, wav_data):
    # [FUNCTION DOC]
    # HELPER: fixture writing bytes to a file.
    # Writes the given wav bytes to the given path in binary mode, overwriting it. Returns nothing.
    with open(filename, "wb") as f:
      f.write(wav_data)

  def _saveWavFolders(self, root_dir, labels, how_many):
    # [FUNCTION DOC]
    # HELPER: fixture building a fake data set layout.
    # For each label creates a subfolder of root_dir (which must not already contain it) and writes
    # how_many silent wav files named "some_audio_N.wav" into it. Returns nothing.
    wav_data = self._getWavData()
    for label in labels:
      dir_name = os.path.join(root_dir, label)
      os.mkdir(dir_name)
      for i in range(how_many):
        file_path = os.path.join(dir_name, "some_audio_%d.wav" % i)
        self._saveTestWavFile(file_path, wav_data)

  def _model_settings(self):
    # [FUNCTION DOC]
    # HELPER: fixture returning a small model settings mapping (160 desired samples, fingerprint
    # size 40, 4 labels, window size and stride 100 samples, fingerprint width 40, mfcc mode).
    return {
        "desired_samples": 160,
        "fingerprint_size": 40,
        "label_count": 4,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "fingerprint_width": 40,
        "preprocess": "mfcc",
    }

  def _runGetDataTest(self, preprocess, window_length_ms):
    # [FUNCTION DOC]
    # HELPER: shared body of the get-data tests.
    # Builds a data set with folders a, b, c (100 wavs each) plus a background noise folder of 10
    # wavs, creates model settings for the given preprocess mode and window length (4 labels,
    # 16000 Hz, 1000 ms clips, 20 ms stride, 40 bins), makes an AudioProcessor, and requests a
    # training batch of 10 with background frequency 0.3, volume 0.1 and time shift 100. Asserts
    # that exactly 10 data rows and 10 labels come back.
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    background_dir = os.path.join(wav_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._getWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._saveTestWavFile(file_path, wav_data)
    model_settings = models.prepare_model_settings(
        4, 16000, 1000, window_length_ms, 20, 40, preprocess)
    with self.cached_session() as sess:
      audio_processor = input_data.AudioProcessor(
          "", wav_dir, 10, 10, ["a", "b"], 10, 10, model_settings, tmp_dir)
      result_data, result_labels = audio_processor.get_data(
          10, 0, model_settings, 0.3, 0.1, 100, "training", sess)
      self.assertEqual(10, len(result_data))
      self.assertEqual(10, len(result_labels))

  def testPrepareWordsList(self):
    # [FUNCTION DOC]
    # Asserts the prepared word list is longer than the two-word input list (special labels added).
    words_list = ["a", "b"]
    self.assertGreater(
        len(input_data.prepare_words_list(words_list)), len(words_list))

  def testWhichSet(self):
    # [FUNCTION DOC]
    # Asserts partition assignment is deterministic for the same name, and that names differing
    # only after the "_nohash_" marker land in the same partition.
    self.assertEqual(
        input_data.which_set("foo.wav", 10, 10),
        input_data.which_set("foo.wav", 10, 10))
    self.assertEqual(
        input_data.which_set("foo_nohash_0.wav", 10, 10),
        input_data.which_set("foo_nohash_1.wav", 10, 10))

  @test_util.run_deprecated_v1
  def testPrepareDataIndex(self):
    # [FUNCTION DOC]
    # Scenario: folders a, b, c with 100 wavs each, wanted words a and b. Creates an AudioProcessor
    # and asserts the training set is non-empty, the three partitions exist in the index, and the
    # unwanted word c maps to the unknown word index.
    tmp_dir = self.get_temp_dir()
    self._saveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    audio_processor = input_data.AudioProcessor("", tmp_dir, 10, 10,
                                                ["a", "b"], 10, 10,
                                                self._model_settings(), tmp_dir)
    self.assertLess(0, audio_processor.set_size("training"))
    self.assertIn("training", audio_processor.data_index)
    self.assertIn("validation", audio_processor.data_index)
    self.assertIn("testing", audio_processor.data_index)
    self.assertEqual(input_data.UNKNOWN_WORD_INDEX,
                     audio_processor.word_to_index["c"])

  def testPrepareDataIndexEmpty(self):
    # [FUNCTION DOC]
    # Scenario: label folders exist but contain no wavs. Creating an AudioProcessor must raise an
    # exception whose message contains "No .wavs found".
    tmp_dir = self.get_temp_dir()
    self._saveWavFolders(tmp_dir, ["a", "b", "c"], 0)
    with self.assertRaises(Exception) as e:
      _ = input_data.AudioProcessor("", tmp_dir, 10, 10, ["a", "b"], 10, 10,
                                    self._model_settings(), tmp_dir)
    self.assertIn("No .wavs found", str(e.exception))

  def testPrepareDataIndexMissing(self):
    # [FUNCTION DOC]
    # Scenario: wanted words include d, which has no folder. Creating an AudioProcessor must raise
    # an exception whose message contains "Expected to find".
    tmp_dir = self.get_temp_dir()
    self._saveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    with self.assertRaises(Exception) as e:
      _ = input_data.AudioProcessor("", tmp_dir, 10, 10, ["a", "b", "d"], 10,
                                    10, self._model_settings(), tmp_dir)
    self.assertIn("Expected to find", str(e.exception))

  @test_util.run_deprecated_v1
  def testPrepareBackgroundData(self):
    # [FUNCTION DOC]
    # Scenario: a background noise folder with 10 wavs plus label folders. After creating the
    # AudioProcessor, asserts exactly 10 background clips were loaded.
    tmp_dir = self.get_temp_dir()
    background_dir = os.path.join(tmp_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._getWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._saveTestWavFile(file_path, wav_data)
    self._saveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    audio_processor = input_data.AudioProcessor("", tmp_dir, 10, 10,
                                                ["a", "b"], 10, 10,
                                                self._model_settings(), tmp_dir)
    self.assertEqual(10, len(audio_processor.background_data))

  def testLoadWavFile(self):
    # [FUNCTION DOC]
    # Writes a silent wav file and loads it with the load function; asserts a non-null result.
    tmp_dir = self.get_temp_dir()
    file_path = os.path.join(tmp_dir, "load_test.wav")
    wav_data = self._getWavData()
    self._saveTestWavFile(file_path, wav_data)
    sample_data = input_data.load_wav_file(file_path)
    self.assertIsNotNone(sample_data)

  def testSaveWavFile(self):
    # [FUNCTION DOC]
    # Saves 16000 zero samples at 16000 Hz as a wav, loads it back, and asserts the loaded data
    # is non-null and has 16000 samples.
    tmp_dir = self.get_temp_dir()
    file_path = os.path.join(tmp_dir, "load_test.wav")
    save_data = np.zeros([16000, 1])
    input_data.save_wav_file(file_path, save_data, 16000)
    loaded_data = input_data.load_wav_file(file_path)
    self.assertIsNotNone(loaded_data)
    self.assertEqual(16000, len(loaded_data))

  @test_util.run_deprecated_v1
  def testPrepareProcessingGraph(self):
    # [FUNCTION DOC]
    # Scenario: data set with background noise and mfcc model settings. Creates an AudioProcessor
    # and asserts all graph placeholders (filename, foreground volume, time shift padding and
    # offset, background data and volume) and the output tensor attributes are non-null.
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    background_dir = os.path.join(wav_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._getWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._saveTestWavFile(file_path, wav_data)
    model_settings = {
        "desired_samples": 160,
        "fingerprint_size": 40,
        "label_count": 4,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "fingerprint_width": 40,
        "preprocess": "mfcc",
    }
    audio_processor = input_data.AudioProcessor("", wav_dir, 10, 10, ["a", "b"],
                                                10, 10, model_settings, tmp_dir)
    self.assertIsNotNone(audio_processor.wav_filename_placeholder_)
    self.assertIsNotNone(audio_processor.foreground_volume_placeholder_)
    self.assertIsNotNone(audio_processor.time_shift_padding_placeholder_)
    self.assertIsNotNone(audio_processor.time_shift_offset_placeholder_)
    self.assertIsNotNone(audio_processor.background_data_placeholder_)
    self.assertIsNotNone(audio_processor.background_volume_placeholder_)
    self.assertIsNotNone(audio_processor.output_)

  @test_util.run_deprecated_v1
  def testGetDataAverage(self):
    # [FUNCTION DOC]
    # Runs the shared get-data scenario with average preprocessing and a 10 ms window.
    self._runGetDataTest("average", 10)

  @test_util.run_deprecated_v1
  def testGetDataAverageLongWindow(self):
    # [FUNCTION DOC]
    # Runs the shared get-data scenario with average preprocessing and a 30 ms window.
    self._runGetDataTest("average", 30)

  @test_util.run_deprecated_v1
  def testGetDataMfcc(self):
    # [FUNCTION DOC]
    # Runs the shared get-data scenario with mfcc preprocessing and a 30 ms window.
    self._runGetDataTest("mfcc", 30)

  @test_util.run_deprecated_v1
  def testGetDataMicro(self):
    # [FUNCTION DOC]
    # Runs the shared get-data scenario with micro frontend preprocessing and a 20 ms window.
    # Requires the micro frontend op to be available, otherwise processor creation raises.
    self._runGetDataTest("micro", 20)

  @test_util.run_deprecated_v1
  def testGetUnprocessedData(self):
    # [FUNCTION DOC]
    # Scenario: data set of three label folders, mfcc settings. Requests 10 unprocessed training
    # samples and asserts 10 data rows and 10 labels are returned.
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 100)
    model_settings = {
        "desired_samples": 160,
        "fingerprint_size": 40,
        "label_count": 4,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "fingerprint_width": 40,
        "preprocess": "mfcc",
    }
    audio_processor = input_data.AudioProcessor("", wav_dir, 10, 10, ["a", "b"],
                                                10, 10, model_settings, tmp_dir)
    result_data, result_labels = audio_processor.get_unprocessed_data(
        10, model_settings, "training")
    self.assertEqual(10, len(result_data))
    self.assertEqual(10, len(result_labels))

  @test_util.run_deprecated_v1
  def testGetFeaturesForWav(self):
    # [FUNCTION DOC]
    # Scenario: average preprocessing with 1600 desired samples. Saves a test wav containing a
    # repeating pattern of zero, minus one, zero, one (a quarter-rate tone), then computes its
    # features. Asserts the first result has shape 1 by 16 by 11, the value at the first bin is
    # about 0 and the value at bin 5 is about 200 (both within 0.1).
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ["a", "b", "c"], 1)
    desired_samples = 1600
    model_settings = {
        "desired_samples": desired_samples,
        "fingerprint_size": 40,
        "label_count": 4,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "fingerprint_width": 40,
        "average_window_width": 6,
        "preprocess": "average",
    }
    with self.cached_session() as sess:
      audio_processor = input_data.AudioProcessor(
          "", wav_dir, 10, 10, ["a", "b"], 10, 10, model_settings, tmp_dir)
      sample_data = np.zeros([desired_samples, 1])
      for i in range(desired_samples):
        phase = i % 4
        if phase == 0:
          sample_data[i, 0] = 0
        elif phase == 1:
          sample_data[i, 0] = -1
        elif phase == 2:
          sample_data[i, 0] = 0
        elif phase == 3:
          sample_data[i, 0] = 1
      test_wav_path = os.path.join(tmp_dir, "test_wav.wav")
      input_data.save_wav_file(test_wav_path, sample_data, 16000)

      results = audio_processor.get_features_for_wav(test_wav_path,
                                                     model_settings, sess)
      spectrogram = results[0]
      self.assertEqual(1, spectrogram.shape[0])
      self.assertEqual(16, spectrogram.shape[1])
      self.assertEqual(11, spectrogram.shape[2])
      self.assertNear(0, spectrogram[0, 0, 0], 0.1)
      self.assertNear(200, spectrogram[0, 0, 5], 0.1)

  def testGetFeaturesRange(self):
    # [FUNCTION DOC]
    # Asserts the feature minimum for average mode is about 0.
    model_settings = {
        "preprocess": "average",
    }
    features_min, _ = input_data.get_features_range(model_settings)
    self.assertNear(0.0, features_min, 1e-5)

  def testGetMfccFeaturesRange(self):
    # [FUNCTION DOC]
    # Asserts the mfcc mode feature minimum is less than its maximum.
    model_settings = {
        "preprocess": "mfcc",
    }
    features_min, features_max = input_data.get_features_range(model_settings)
    self.assertLess(features_min, features_max)


if __name__ == "__main__":
  test.main()
