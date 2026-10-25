# [MODULE DOC]
# Integration test that the public tf.experimental.numpy module can be used through a module
# alias and gives results matching real numpy for a simple addition. Runs eagerly when executed
# as a script.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests using module `tf.experimental.numpy` via an alias."""

import numpy as onp
import tensorflow as tf


np = tf.experimental.numpy


class PublicSymbolTest(tf.test.TestCase):
  # [CLASS DOC]
  # Test case for the public tf.experimental.numpy alias defined at module level. No fixtures.

  def testSimple(self):
    # [FUNCTION DOC]
    # Scenario: adds two Python floats with real numpy and with the tf-numpy add reached through
    # the alias. Asserts the two results are numerically close.
    a = 0.1
    b = 0.2
    self.assertAllClose(onp.add(a, b), np.add(a, b))


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
