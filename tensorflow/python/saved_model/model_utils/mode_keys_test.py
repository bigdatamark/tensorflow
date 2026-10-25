# [MODULE DOC]
# Unit tests for the mode-keyed immutable mapping in the mode_keys module: lookup by equivalent
# Keras and Estimator mode names, missing and invalid keys, standard mapping methods,
# immutability, and rejection of duplicate equivalent keys. No import-time side effects.
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
"""ModeKey Tests."""

from tensorflow.python.platform import test
from tensorflow.python.saved_model.model_utils import mode_keys


class ModeKeyMapTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the mode-keyed mapping class. No shared fixtures.

  def test_map(self):
    # [FUNCTION DOC]
    # Scenario: a map built with Keras predict mapped to 3 and Keras test mapped to 1. Asserts that
    # lookups work with both Keras and Estimator names (infer, eval), that the absent train mode
    # raises a key error under either name, that an unknown string raises a value error mentioning
    # an invalid mode, that length is two with values 1 and 3 and the supplied keys, and that
    # item assignment raises a type error because the map is immutable.
    mode_map = mode_keys.ModeKeyMap(**{
        mode_keys.KerasModeKeys.PREDICT: 3,
        mode_keys.KerasModeKeys.TEST: 1
    })

    # Test dictionary __getitem__
    self.assertEqual(3, mode_map[mode_keys.KerasModeKeys.PREDICT])
    self.assertEqual(3, mode_map[mode_keys.EstimatorModeKeys.PREDICT])
    self.assertEqual(1, mode_map[mode_keys.KerasModeKeys.TEST])
    self.assertEqual(1, mode_map[mode_keys.EstimatorModeKeys.EVAL])
    with self.assertRaises(KeyError):
      _ = mode_map[mode_keys.KerasModeKeys.TRAIN]
    with self.assertRaises(KeyError):
      _ = mode_map[mode_keys.EstimatorModeKeys.TRAIN]
    with self.assertRaisesRegex(ValueError, 'Invalid mode'):
      _ = mode_map['serve']

    # Test common dictionary methods
    self.assertLen(mode_map, 2)
    self.assertEqual({1, 3}, set(mode_map.values()))
    self.assertEqual(
        {mode_keys.KerasModeKeys.TEST, mode_keys.KerasModeKeys.PREDICT},
        set(mode_map.keys()))

    # Map is immutable
    with self.assertRaises(TypeError):
      mode_map[mode_keys.KerasModeKeys.TEST] = 1  # pylint: disable=unsupported-assignment-operation

  def test_invalid_init(self):
    # [FUNCTION DOC]
    # Scenario: construct the map with both Keras predict and Estimator infer, which are the same
    # mode. Expects a value error saying multiple keys/values were found.
    with self.assertRaisesRegex(ValueError, 'Multiple keys/values found'):
      _ = mode_keys.ModeKeyMap(**{
          mode_keys.KerasModeKeys.PREDICT: 3,
          mode_keys.EstimatorModeKeys.PREDICT: 1
      })


if __name__ == '__main__':
  test.main()
