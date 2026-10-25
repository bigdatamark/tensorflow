# [MODULE DOC]
# Test module for the serialization helpers: checks JSON round trips through get_json_type.
# Runs through the standard TensorFlow test main. No import-time side effects.

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
"""Tests for serialization functions."""

import json

from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test
from tensorflow.python.util import serialization


class SerializationTests(test.TestCase):
  # [CLASS DOC]
  # Tests for the serialization module's JSON conversion hook. No shared fixtures.

  def test_serialize_shape(self):
    # [FUNCTION DOC]
    # Scenario: a tensor shape with an unknown first dimension and known dimensions 2 and 3 is
    # dumped to JSON text using get_json_type as the default handler, then parsed back.
    # Asserts the first entry is null (unknown dimension) and the second equals 2.
    round_trip = json.loads(json.dumps(
        tensor_shape.TensorShape([None, 2, 3]),
        default=serialization.get_json_type))
    self.assertIs(round_trip[0], None)
    self.assertEqual(round_trip[1], 2)


if __name__ == "__main__":
  test.main()
