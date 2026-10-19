# [MODULE DOC]
# Reference test for autograph conversion of conditional expressions (inline if-else) nested in
# if statements. Each sample function is run natively and converted, with Python numbers and
# tensors, and results are compared. No import-time side effects.
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
"""Basic if conditional.

The loop is converted to tf.cond.
"""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def consecutive_conds(x):
  # [FUNCTION DOC]
  # Function under test: two consecutive if/else statements whose branches reassign x with
  # conditional expressions (negate or scale x when below five). Returns the final x.
  if x > 0:
    x = -x if x < 5 else x
  else:
    x = -2 * x if x < 5 else 1
  if x > 0:
    x = -x if x < 5 else x
  else:
    x = (3 * x) if x < 5 else x
  return x


def cond_with_multiple_values(x):
  # [FUNCTION DOC]
  # Function under test: an if/else statement where each branch computes x, y and z using
  # conditional expressions. Returns the tuple of x, y and z.
  if x > 0:
    x = -x if x < 5 else x
    y = 2 * x if x < 5 else x
    z = -y if y < 5 else y
  else:
    x = 2 * x if x < 5 else x
    y = -x if x < 5 else x
    z = -y if y < 5 else y
  return x, y, z


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Reference test case for conditional-expression conversion.

  def test_basic(self):
    # [FUNCTION DOC]
    # For Python numbers and equivalent tensors (negative, small and large), asserts both sample
    # functions give the same results natively and when compiled.
    for x in [-1, 1, 5, tf.constant(-1), tf.constant(1), tf.constant(5)]:
      self.assertFunctionMatchesEager(consecutive_conds, x)
      self.assertFunctionMatchesEager(cond_with_multiple_values, x)


if __name__ == '__main__':
  tf.test.main()
