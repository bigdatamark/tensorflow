# [MODULE DOC]
# Unit test module for generate_streaming_test_wav. Contains a single test case checking that
# mixing a sample into a track applies the fade-in/fade-out envelope correctly.
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
"""Tests for test file generation for speech commands."""

import numpy as np

from tensorflow.examples.speech_commands import generate_streaming_test_wav
from tensorflow.python.platform import test


class GenerateStreamingTestWavTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the mix_in_audio_sample helper of the streaming test wav generator.
  # No shared fixtures.

  def testMixInAudioSample(self):
    # [FUNCTION DOC]
    # Scenario: a track of 10000 zeros and a sample of 1000 ones are mixed at track offset 2000
    # with clip duration 1000, volume 1.0 and 100-sample ramps in and out. Asserts the track value
    # at index 2500 (middle of the clip, full volume) is about 1.0 and at index 3500 (past the
    # clip end, untouched) is about 0.0, within 0.0001. Returns nothing.
    track_data = np.zeros([10000])
    sample_data = np.ones([1000])
    generate_streaming_test_wav.mix_in_audio_sample(
        track_data, 2000, sample_data, 0, 1000, 1.0, 100, 100)
    self.assertNear(1.0, track_data[2500], 0.0001)
    self.assertNear(0.0, track_data[3500], 0.0001)


if __name__ == "__main__":
  test.main()
