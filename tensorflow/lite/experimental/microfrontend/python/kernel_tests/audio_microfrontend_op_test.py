# [MODULE DOC]
# Tests for the audio microfrontend operation. Feeds a repeating synthetic int16 waveform (a
# square-like pattern of full-scale values) at 1000 Hz with 25 ms windows, 10 ms steps and two
# channels, and checks the produced filterbank frames for the basic case, float scaled output,
# frame stacking with left/right context, stride, and zero padding. Module constants hold the
# shared settings. Runs under the TensorFlow test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for AudioMicrofrontend."""

import tensorflow as tf

from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow.python.framework import ops

SAMPLE_RATE = 1000
WINDOW_SIZE = 25
WINDOW_STEP = 10
NUM_CHANNELS = 2
UPPER_BAND_LIMIT = 450.0
LOWER_BAND_LIMIT = 8.0
SMOOTHING_BITS = 10


class AudioFeatureGenerationTest(tf.test.TestCase):
  # [CLASS DOC]
  # Test case for the microfrontend operation. Disables eager execution in setup so tests run in
  # graph mode with a session. Each test builds a short synthetic audio tensor and compares the
  # operation output to expected filterbank values.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Runs the parent setup and then globally disables eager execution (a process-wide
    # side effect). Returns nothing.
    super(AudioFeatureGenerationTest, self).setUp()
    ops.disable_eager_execution()

  def testSimple(self):
    # [FUNCTION DOC]
    # Scenario: audio is a repeating four-sample pattern (0, max, 0, min) long enough for four
    # frames; run the operation with the shared settings and PCAN enabled. Asserts the output
    # equals four frames of two integer channel values.
    with self.test_session():
      audio = tf.constant(
          [0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4),
          tf.int16)
      filterbanks = frontend_op.audio_microfrontend(
          audio,
          sample_rate=SAMPLE_RATE,
          window_size=WINDOW_SIZE,
          window_step=WINDOW_STEP,
          num_channels=NUM_CHANNELS,
          upper_band_limit=UPPER_BAND_LIMIT,
          lower_band_limit=LOWER_BAND_LIMIT,
          smoothing_bits=SMOOTHING_BITS,
          enable_pcan=True)
      self.assertAllEqual(
          self.evaluate(filterbanks),
          [[479, 425], [436, 378], [410, 350], [391, 325]])

  def testSimpleFloatScaled(self):
    # [FUNCTION DOC]
    # Scenario: same audio as the simple test but with output scale 64 and float output type.
    # Asserts the four frames equal the integer result divided by 64 (fractional values).
    with self.test_session():
      audio = tf.constant(
          [0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4),
          tf.int16)
      filterbanks = frontend_op.audio_microfrontend(
          audio,
          sample_rate=SAMPLE_RATE,
          window_size=WINDOW_SIZE,
          window_step=WINDOW_STEP,
          num_channels=NUM_CHANNELS,
          upper_band_limit=UPPER_BAND_LIMIT,
          lower_band_limit=LOWER_BAND_LIMIT,
          smoothing_bits=SMOOTHING_BITS,
          enable_pcan=True,
          out_scale=64,
          out_type=tf.float32)
      self.assertAllEqual(
          self.evaluate(filterbanks),
          [[7.484375, 6.640625], [6.8125, 5.90625], [6.40625, 5.46875],
           [6.109375, 5.078125]])

  def testStacking(self):
    # [FUNCTION DOC]
    # Scenario: same audio with right context of one frame and frame stride of two. Asserts two
    # output rows, each holding two consecutive frames concatenated (four values).
    with self.test_session():
      audio = tf.constant(
          [0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4),
          tf.int16)
      filterbanks = frontend_op.audio_microfrontend(
          audio,
          sample_rate=SAMPLE_RATE,
          window_size=WINDOW_SIZE,
          window_step=WINDOW_STEP,
          num_channels=NUM_CHANNELS,
          upper_band_limit=UPPER_BAND_LIMIT,
          lower_band_limit=LOWER_BAND_LIMIT,
          smoothing_bits=SMOOTHING_BITS,
          enable_pcan=True,
          right_context=1,
          frame_stride=2)
      self.assertAllEqual(
          self.evaluate(filterbanks),
          [[479, 425, 436, 378], [410, 350, 391, 325]])

  def testStackingWithOverlap(self):
    # [FUNCTION DOC]
    # Scenario: same audio with left and right context of one frame each and default stride.
    # Asserts four rows of six values (previous, current, next frame); at the edges the first or
    # last frame is repeated since zero padding is off.
    with self.test_session():
      audio = tf.constant(
          [0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4),
          tf.int16)
      filterbanks = frontend_op.audio_microfrontend(
          audio,
          sample_rate=SAMPLE_RATE,
          window_size=WINDOW_SIZE,
          window_step=WINDOW_STEP,
          num_channels=NUM_CHANNELS,
          upper_band_limit=UPPER_BAND_LIMIT,
          lower_band_limit=LOWER_BAND_LIMIT,
          smoothing_bits=SMOOTHING_BITS,
          enable_pcan=True,
          left_context=1,
          right_context=1)
      self.assertAllEqual(
          self.evaluate(filterbanks),
          [[479, 425, 479, 425, 436, 378], [479, 425, 436, 378, 410, 350],
           [436, 378, 410, 350, 391, 325], [410, 350, 391, 325, 391, 325]])

  def testStackingDropFrame(self):
    # [FUNCTION DOC]
    # Scenario: same audio with left context of one frame and frame stride of two. Asserts two
    # rows of four values, showing that every second stacked frame is dropped and the first
    # frame is repeated as left context.
    with self.test_session():
      audio = tf.constant(
          [0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4),
          tf.int16)
      filterbanks = frontend_op.audio_microfrontend(
          audio,
          sample_rate=SAMPLE_RATE,
          window_size=WINDOW_SIZE,
          window_step=WINDOW_STEP,
          num_channels=NUM_CHANNELS,
          upper_band_limit=UPPER_BAND_LIMIT,
          lower_band_limit=LOWER_BAND_LIMIT,
          smoothing_bits=SMOOTHING_BITS,
          enable_pcan=True,
          left_context=1,
          frame_stride=2)
      self.assertAllEqual(
          self.evaluate(filterbanks),
          [[479, 425, 479, 425], [436, 378, 410, 350]])

  def testZeroPadding(self):
    # [FUNCTION DOC]
    # Scenario: longer audio with left context of two frames, stride of three and zero padding
    # enabled. Asserts three rows of six values, where out-of-range left context frames at the
    # start are filled with zeros.
    with self.test_session():
      audio = tf.constant(
          [0, 32767, 0, -32768] * ((WINDOW_SIZE + 7 * WINDOW_STEP) // 4),
          tf.int16)
      filterbanks = frontend_op.audio_microfrontend(
          audio,
          sample_rate=SAMPLE_RATE,
          window_size=WINDOW_SIZE,
          window_step=WINDOW_STEP,
          num_channels=NUM_CHANNELS,
          upper_band_limit=UPPER_BAND_LIMIT,
          lower_band_limit=LOWER_BAND_LIMIT,
          smoothing_bits=SMOOTHING_BITS,
          enable_pcan=True,
          left_context=2,
          frame_stride=3,
          zero_padding=True)
      self.assertAllEqual(
          self.evaluate(filterbanks),
          [[0, 0, 0, 0, 479, 425], [436, 378, 410, 350, 391, 325],
           [374, 308, 362, 292, 352, 275]])


if __name__ == '__main__':
  tf.test.main()
