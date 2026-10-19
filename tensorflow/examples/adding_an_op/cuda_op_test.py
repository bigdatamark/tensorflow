# [MODULE DOC]
# Test for the CUDA add-one example op wrapper. Contains one test case class and a standard
# main entry that runs the TensorFlow test runner. KNOWN ISSUE: the module docstring says it
# tests version 1 of the zero-out op, but it actually tests the CUDA add-one op.
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
"""Test for version 1 of the zero_out op."""

import tensorflow as tf
from tensorflow.examples.adding_an_op import cuda_op


class AddOneTest(tf.test.TestCase):
  # [CLASS DOC]
  # Test case covering the CUDA add-one example op. Has no fixtures.

  def test(self):
    # [FUNCTION DOC]
    # Only when TensorFlow is built with CUDA: calls the add-one op on the list five, four,
    # three, two, one and asserts the result equals six, five, four, three, two. On non-CUDA
    # builds the test does nothing and passes.
    if tf.test.is_built_with_cuda():
      result = cuda_op.add_one([5, 4, 3, 2, 1])
      self.assertAllEqual(result, [6, 5, 4, 3, 2])


if __name__ == '__main__':
  tf.test.main()
