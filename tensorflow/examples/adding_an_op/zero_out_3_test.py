# [MODULE DOC]
# Test module for version 3 of the tutorial zero_out custom op. It loads the op through the
# zero_out_op_3 wrapper module and checks the op behavior, including its preserve_index
# attribute and the error cases. No import-time side effects beyond importing TensorFlow and the
# wrapper (which loads a compiled shared library). Run directly, it starts the TensorFlow test
# runner.
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

"""Test for version 3 of the zero_out op."""

import tensorflow as tf
from tensorflow.examples.adding_an_op import zero_out_op_3


class ZeroOut3Test(tf.test.TestCase):
  # [CLASS DOC]
  # Test case covering the version 3 zero_out op (zero_out_op_3). Has no shared fixtures; each
  # test calls the op on the same five-element integer list.

  def test(self):
    # [FUNCTION DOC]
    # Calls the op with default arguments on a five-element list (5, 4, 3, 2, 1) and asserts the
    # result keeps only the first element: 5 followed by four zeros. Returns nothing.
    result = zero_out_op_3.zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_attr(self):
    # [FUNCTION DOC]
    # Calls the op with the preserve_index attribute set to 3 and asserts that only the element at
    # index 3 (value 2) is kept and all others are zero. Returns nothing.
    result = zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=3)
    self.assertAllEqual(result, [0, 0, 0, 2, 0])

  def test_negative(self):
    # [FUNCTION DOC]
    # Calls the op with preserve_index of -1 and evaluates it; asserts an op error is raised whose
    # message says preserve_index must be non-negative. Returns nothing.
    with self.assertRaisesOpError("Need preserve_index >= 0, got -1"):
      self.evaluate(zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=-1))

  def test_large(self):
    # [FUNCTION DOC]
    # Calls the op with preserve_index of 17 (beyond the input length of five) and evaluates it;
    # asserts an op error is raised saying preserve_index is out of range. Returns nothing.
    with self.assertRaisesOpError("preserve_index out of range"):
      self.evaluate(zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=17))


if __name__ == "__main__":
  tf.test.main()
