# [MODULE DOC]
# Tests for the legacy (v1) audio summary op: serialized summary metadata for random audio
# with different channel counts. No import-time side effects.

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
"""Tests for summary V1 audio op."""

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.summary import summary


class SummaryV1AudioOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy audio summary op: builds summaries from random multi-channel audio and checks
  # the serialized summary message (tags, content type, rate, channels, frames).

  def _AsSummary(self, s):
    # [FUNCTION DOC]
    # HELPER: parses a serialized summary string (bytes) and returns the summary message.
    summ = summary_pb2.Summary()
    summ.ParseFromString(s)
    return summ

  def _CheckProto(self, audio_summ, sample_rate, num_channels, length_frames):
    # [FUNCTION DOC]
    # HELPER: checks the non-audio parts of a summary message.
    # Clears the encoded audio data from each value, then asserts the message equals the expected text
    # for the first three outputs: tags snd/audio/0..2, wav content type, the given sample rate,
    # channel count and frame count. Fails the test on any mismatch.
    """Verify that the non-audio parts of the audio_summ proto match shape."""
    # Only the first 3 sounds are returned.
    for v in audio_summ.value:
      v.audio.ClearField("encoded_audio_string")
    expected = "\n".join("""
        value {
          tag: "snd/audio/%d"
          audio { content_type: "audio/wav" sample_rate: %d
                  num_channels: %d length_frames: %d }
        }""" % (i, sample_rate, num_channels, length_frames) for i in range(3))
    self.assertProtoEquals(expected, audio_summ)

  def testAudioSummary(self):
    # [FUNCTION DOC]
    # For 1, 2, 5 and 8 channels, summarizes random audio (batch of 4, 7 frames, values in minus one
    # to one) with at most three outputs and rate 8000 in a fresh graph. Asserts the summary is a
    # scalar and that the parsed message matches the expected metadata.
    np.random.seed(7)
    for channels in (1, 2, 5, 8):
      with self.session(graph=ops.Graph()) as sess:
        num_frames = 7
        shape = (4, num_frames, channels)
        # Generate random audio in the range [-1.0, 1.0).
        const = 2.0 * np.random.random(shape) - 1.0

        # Summarize
        sample_rate = 8000
        summ = summary.audio(
            "snd", const, max_outputs=3, sample_rate=sample_rate)
        value = self.evaluate(summ)
        self.assertEqual([], summ.get_shape())
        audio_summ = self._AsSummary(value)

        # Check the rest of the proto
        self._CheckProto(audio_summ, sample_rate, channels, num_frames)


if __name__ == "__main__":
  test.main()
