# [MODULE DOC]
# KNOWN ISSUE: the module docstring says "data input" tests, but this file tests the training
# script's main routine.
# End-to-end smoke test of the speech commands training script. Generates dummy wav data in a
# temporary directory, supplies a stand-in flags object, runs two training steps and checks that
# the graph, label list and a checkpoint file are written. Includes a decorator for skipping tests
# that need contrib, and a small class that turns a mapping into an attribute holder.
# Runs under the TensorFlow test runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import unittest

import tensorflow as tf

from tensorflow.examples.speech_commands import train
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


def requires_contrib(test_method):
  # [FUNCTION DOC]
  # HELPER: test decorator that skips tests needing the removed contrib package.
  # KNOWN ISSUE: it is defined but never applied in this file.
  # Input: a test method. If accessing contrib on the TensorFlow module fails with an attribute
  # error, wraps the method so it is skipped with an install hint message. Returns the
  # (possibly wrapped) method.
  try:
    _ = tf.contrib
  except AttributeError:
    test_method = unittest.skip(
        'This test requires tf.contrib:\n    `pip install tensorflow<=1.15`')(
            test_method)

  return test_method


# Used to convert a dictionary into an object, for mocking parsed flags.
class DictStruct(object):
  # [CLASS DOC]
  # HELPER: lightweight attribute holder used to mock parsed command line flags in the test.
  # Each keyword given at construction becomes an attribute.

  def __init__(self, **entries):
    # [FUNCTION DOC]
    # Constructor. Takes arbitrary named values and stores each as an attribute on the instance.
    # Returns nothing.
    self.__dict__.update(entries)


class TrainTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the training script. Provides helpers that create dummy wav files laid out in
  # label folders plus a background noise folder, and a default flags object; the single test
  # runs the training entry point on that data.

  def _getWavData(self):
    # [FUNCTION DOC]
    # HELPER: produces encoded wav bytes for tests.
    # Encodes two seconds of silent stereo audio at 16000 Hz (32000 frames, 2 channels) in a
    # cached session and returns the resulting bytes.
    with self.cached_session():
      sample_data = tf.zeros([32000, 2])
      wav_encoder = tf.audio.encode_wav(sample_data, 16000)
      wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _saveTestWavFile(self, filename, wav_data):
    # [FUNCTION DOC]
    # HELPER: writes bytes to disk. Takes a file path and the bytes; writes them in binary mode,
    # overwriting any existing file. Returns nothing.
    with open(filename, 'wb') as f:
      f.write(wav_data)

  def _saveWavFolders(self, root_dir, labels, how_many):
    # [FUNCTION DOC]
    # HELPER: creates dummy label folders. For each label in the list, makes a subfolder of the
    # root directory (which must exist; folder must not already exist) and writes the given number
    # of identical silent wav files into it. Returns nothing.
    wav_data = self._getWavData()
    for label in labels:
      dir_name = os.path.join(root_dir, label)
      os.mkdir(dir_name)
      for i in range(how_many):
        file_path = os.path.join(dir_name, 'some_audio_%d.wav' % i)
        self._saveTestWavFile(file_path, wav_data)

  def _prepareDummyTrainingData(self):
    # [FUNCTION DOC]
    # HELPER: builds a dummy dataset in the test temp directory. Creates a wavs folder with three
    # label folders (a, b, c) of 100 silent clips each, and a background noise folder with 10
    # silent clips. Takes no arguments; returns the path of the wavs folder.
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, 'wavs')
    os.mkdir(wav_dir)
    self._saveWavFolders(wav_dir, ['a', 'b', 'c'], 100)
    background_dir = os.path.join(wav_dir, '_background_noise_')
    os.mkdir(background_dir)
    wav_data = self._getWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, 'background_audio_%d.wav' % i)
      self._saveTestWavFile(file_path, wav_data)
    return wav_dir

  def _getDefaultFlags(self):
    # [FUNCTION DOC]
    # HELPER: builds a stand-in flags object for the training script. Prepares the dummy data and
    # returns an attribute holder with test-friendly values: wanted words a,b,c, 16 kHz, 1 s
    # clips, 30 ms window, 20 ms stride, 40 bins, mfcc, a two-step schedule with learning rate
    # 0.01, conv architecture, batch size 1, evaluation and save every step, gradient descent,
    # no quantization, summaries and train directories under the temp directory.
    flags = {
        'data_url': '',
        'data_dir': self._prepareDummyTrainingData(),
        'wanted_words': 'a,b,c',
        'sample_rate': 16000,
        'clip_duration_ms': 1000,
        'window_size_ms': 30,
        'window_stride_ms': 20,
        'feature_bin_count': 40,
        'preprocess': 'mfcc',
        'silence_percentage': 25,
        'unknown_percentage': 25,
        'validation_percentage': 10,
        'testing_percentage': 10,
        'summaries_dir': os.path.join(self.get_temp_dir(), 'summaries'),
        'train_dir': os.path.join(self.get_temp_dir(), 'train'),
        'time_shift_ms': 100,
        'how_many_training_steps': '2',
        'learning_rate': '0.01',
        'quantize': False,
        'model_architecture': 'conv',
        'check_nans': False,
        'start_checkpoint': '',
        'batch_size': 1,
        'background_volume': 0.25,
        'background_frequency': 0.8,
        'eval_step_interval': 1,
        'save_step_interval': 1,
        'verbosity': tf.compat.v1.logging.INFO,
        'optimizer': 'gradient_descent'
    }
    return DictStruct(**flags)

  @test_util.run_deprecated_v1
  def testTrain(self):
    # [FUNCTION DOC]
    # Scenario: assign the default stand-in flags to the training module and run its main routine
    # (legacy graph mode). Asserts that the graph text file, the label list file and the
    # checkpoint metadata file for step one exist in the train directory.
    train.FLAGS = self._getDefaultFlags()
    train.main('')
    self.assertTrue(
        gfile.Exists(
            os.path.join(train.FLAGS.train_dir,
                         train.FLAGS.model_architecture + '.pbtxt')))
    self.assertTrue(
        gfile.Exists(
            os.path.join(train.FLAGS.train_dir,
                         train.FLAGS.model_architecture + '_labels.txt')))
    self.assertTrue(
        gfile.Exists(
            os.path.join(train.FLAGS.train_dir,
                         train.FLAGS.model_architecture + '.ckpt-1.meta')))


if __name__ == '__main__':
  test.main()
