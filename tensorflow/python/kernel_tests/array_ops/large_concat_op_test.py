# [MODULE DOC]
# Test module holding concat-operation tests that use tensors larger than 2^31 elements and so
# are kept apart from the regular concat tests. Defines one test case class with one test; no
# import-time side effects beyond imports.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for Concat Op."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class LargeConcatOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for concat on very large tensors; no shared fixtures.
  """Tests that belong in concat_op_test.py, but run over large tensors."""

  def testConcatLargeTensors(self):
    # [FUNCTION DOC]
    # Test scenario: on the CPU, create an 8-bit integer tensor of ones with 2^31 + 6 elements
    # and one of zeros with 1024 elements, concatenate them along axis zero, then evaluate in a
    # CPU-only session. Only checks that evaluation completes without crashing; no values are
    # compared (a TODO notes more validation is wanted). Takes no inputs and returns nothing.
    # CPU-only test, because it fails on GPUs with <= 4GB memory.
    with ops.device("/cpu:0"):
      a = array_ops.ones([2**31 + 6], dtype=dtypes.int8)
      b = array_ops.zeros([1024], dtype=dtypes.int8)
      onezeros = array_ops.concat([a, b], 0)
    with self.session(use_gpu=False):
      # TODO(dga):  Add more depth to this test to validate correctness,
      # not just non-crashingness, once other large tensor fixes have gone in.
      _ = self.evaluate(onezeros)


if __name__ == "__main__":
  test.main()
