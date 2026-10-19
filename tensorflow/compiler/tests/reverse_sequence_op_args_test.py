# [MODULE DOC]
# Test module checking argument validation of the sequence-reversal array op when it is compiled
# with XLA. It defines one test class and, when run as a script, enables v2 behavior and starts
# the test runner. It has no import-time side effects beyond imports.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.reverse_sequence_op."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ReverseSequenceArgsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case covering rejection of invalid axis arguments by the reverse-sequence op.
  # It holds no fixtures of its own and relies on the XLA test base class.
  """Tests argument verification of array_ops.reverse_sequence."""

  def testInvalidArguments(self):
    # [FUNCTION DOC]
    # Checks that negative axis arguments are rejected. First, a compiled function calling
    # reverse-sequence with sequence lengths of two and two and a negative sequence axis is
    # invoked on a two-by-two list; an invalid-argument or value error mentioning that the
    # sequence dimension must be non-negative is expected. Second, the same with a valid sequence
    # axis but a negative batch axis must raise a value error saying the batch dimension must be
    # non-negative. Returns nothing.
    # seq_axis negative
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError), "seq_dim must be >=0"
    ):

      @def_function.function(jit_compile=True)
      def f(x):
        # [FUNCTION DOC]
        # HELPER: Inner function compiled with XLA for the negative sequence axis check.
        # Takes a tensor-like input x and returns it reversed with the fixed sequence lengths
        # and a negative sequence axis, which is invalid.
        return array_ops.reverse_sequence(x, [2, 2], seq_axis=-1)

      f([[1, 2], [3, 4]])

    # batch_axis negative
    with self.assertRaisesRegex(ValueError, "batch_dim must be >=0"):

      @def_function.function(jit_compile=True)
      def g(x):
        # [FUNCTION DOC]
        # HELPER: Inner function compiled with XLA for the negative batch axis check.
        # Takes a tensor-like input x and returns it reversed with fixed sequence lengths,
        # sequence axis one and a negative batch axis, which is invalid.
        return array_ops.reverse_sequence(x, [2, 2], seq_axis=1, batch_axis=-1)

      g([[1, 2], [3, 4]])


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
