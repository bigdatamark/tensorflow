# [MODULE DOC]
# Test module verifying that Keras code loaded lazily by SavedModel loading registers the
# optimizer identifier. Run as a standard TensorFlow test main; no import-time side effects.
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
"""Keras injection tests."""

import tensorflow as tf

from tensorflow.python.eager import test


# Some of the Keras code load should be triggered so that it will inject proper
# functionality like registering the optimizer class for SavedModel.
class KerasInjectionTest(tf.test.TestCase):
  # [CLASS DOC]
  # Test case checking that loading a SavedModel triggers Keras-side registration of the optimizer
  # class without the test touching Keras code directly. No shared fixtures.

  def test_keras_optimizer_injected(self):
    # [FUNCTION DOC]
    # Scenario: load the checked-in test SavedModel containing an optimizer with slot variables
    # (located through the test source directory helper). Asserts that the registered identifiers
    # of the SavedModel loader include the optimizer identifier, showing Keras injected it.
    save_path = test.test_src_dir_path(
        'cc/saved_model/testdata/OptimizerSlotVariableModule')
    _ = tf.saved_model.load(save_path)
    # Make sure keras optimizers are registed without accessing keras code
    # when loading a model with optimizers
    self.assertIn(
        'optimizer', tf.__internal__.saved_model.load.registered_identifiers()
    )


if __name__ == '__main__':
  tf.test.main()
