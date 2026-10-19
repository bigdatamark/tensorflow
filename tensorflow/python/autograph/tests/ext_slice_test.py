# [MODULE DOC]
# Reference tests for extended slicing (multi-dimensional slices, new axes, slicing the result of
# a call) inside AutoGraph-converted functions, compared against eager execution.
# Contains three subject functions and one test class. No import-time side effects.
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
"""Extended slice operations."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def basic_ext_slice(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Takes a 2-D tensor n and returns three slices: everything, first row, first column.
  return n[:, :], n[0, :], n[:, 0]


def basic_expand_dims(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Subtracts the row-vector view of n from its column-vector view (new axes), by broadcasting.
  return n[:, tf.newaxis] - n[tf.newaxis, :]


def slice_of_application(n, x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Calls callable n on x twice, adds a new axis to each result in opposite positions and
  # returns their difference.
  return n(x)[:, tf.newaxis] - n(x)[tf.newaxis, :]


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Test case checking that each subject function gives identical results in graph and eager mode.

  def test_basic_ext_slice(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the multi-dimensional slices of a 3x3 identity.
    self.assertFunctionMatchesEager(basic_ext_slice, tf.eye(3))

  def test_basic_expand_dims(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for new-axis broadcasting on a 3x3 identity.
    self.assertFunctionMatchesEager(basic_expand_dims, tf.eye(3))

  def test_slice_of_application(self):
    # [FUNCTION DOC]
    # Passes an identity callable and a 3x3 identity; asserts graph and eager results match.
    self.assertFunctionMatchesEager(slice_of_application, lambda x: x,
                                    tf.eye(3))


if __name__ == '__main__':
  tf.test.main()
