# [MODULE DOC]
# Test module for the sequence-reversal array op running under the XLA test harness on the
# device under test. It defines one test class; running it as a script starts the test runner.
# No import-time side effects beyond imports.
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
"""Tests for tensorflow.ops.reverse_sequence_op."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ReverseSequenceTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case verifying that reversing variable-length prefixes along a sequence axis,
  # per batch entry, gives the expected values for all supported element and length types.
  # Holds no state of its own; uses the base class for sessions, test scope and type sets.

  def _testReverseSequence(self,
                           x,
                           batch_axis,
                           seq_axis,
                           seq_lengths,
                           truth,
                           expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: Shared checker for the reverse-sequence tests in this class.
    # Inputs: x is a numpy array to reverse; batch_axis and seq_axis are the integer axes;
    # seq_lengths is a numpy integer array giving the prefix length to reverse per batch entry;
    # truth is the expected result; expected_err_re is an optional regular expression (default
    # none). Builds placeholders matching the dtypes, runs the op inside the test device scope,
    # and evaluates it with fed values. If no error pattern is given, asserts the result is close
    # to truth; otherwise asserts evaluation raises an op error matching the pattern.
    # Returns nothing.
    with self.session():
      p = array_ops.placeholder(dtypes.as_dtype(x.dtype))
      lengths = array_ops.placeholder(dtypes.as_dtype(seq_lengths.dtype))
      with self.test_scope():
        ans = array_ops.reverse_sequence(
            p, batch_axis=batch_axis, seq_axis=seq_axis, seq_lengths=lengths)
      if expected_err_re is None:
        tf_ans = ans.eval(feed_dict={p: x, lengths: seq_lengths})
        self.assertAllClose(tf_ans, truth, atol=1e-10)
      else:
        with self.assertRaisesOpError(expected_err_re):
          ans.eval(feed_dict={p: x, lengths: seq_lengths})

  def testSimple(self):
    # [FUNCTION DOC]
    # Reverses a three-by-three integer matrix along the column axis with batch axis zero and
    # per-row lengths one, three and two. Expects the first row unchanged, the second fully
    # reversed and the third with only its first two entries swapped. Returns nothing.
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    expected = np.array([[1, 2, 3], [6, 5, 4], [8, 7, 9]], dtype=np.int32)
    self._testReverseSequence(
        x,
        batch_axis=0,
        seq_axis=1,
        seq_lengths=np.array([1, 3, 2], np.int32),
        truth=expected)

  def _testBasic(self, dtype, len_dtype):
    # [FUNCTION DOC]
    # HELPER: Builds the five-dimensional test case used by the sequence-length test.
    # Inputs: dtype is the numpy element type of the data; len_dtype is the numpy integer type of
    # the sequence lengths. Constructs a data tensor with axes permuted so that the sequence axis
    # is zero and the batch axis is two, lengths of three, zero and four (partial, none, full
    # reversal), and the matching expected tensor, then calls the shared checker. Returns nothing.
    x = np.asarray(
        [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]],
         [[17, 18, 19, 20], [21, 22, 23, 24]]],
        dtype=dtype)
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    # reverse dim 2 up to (0:3, none, 0:4) along dim=0
    seq_lengths = np.asarray([3, 0, 4], dtype=len_dtype)

    truth_orig = np.asarray(
        [
            [[3, 2, 1, 4], [7, 6, 5, 8]],  # reverse 0:3
            [[9, 10, 11, 12], [13, 14, 15, 16]],  # reverse none
            [[20, 19, 18, 17], [24, 23, 22, 21]]
        ],  # reverse 0:4 (all)
        dtype=dtype)
    truth_orig = truth_orig.reshape(3, 2, 4, 1, 1)
    truth = truth_orig.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    seq_axis = 0  # permute seq_axis and batch_axis (originally 2 and 0, resp.)
    batch_axis = 2
    self._testReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth)

  def testSeqLength(self):
    # [FUNCTION DOC]
    # Runs the basic reversal scenario for every supported element type combined with each
    # supported length type that is 32-bit or 64-bit integer. Expects each combination to match
    # the precomputed result. Returns nothing.
    for dtype in self.all_types:
      for seq_dtype in self.all_types & {np.int32, np.int64}:
        self._testBasic(dtype, seq_dtype)


if __name__ == "__main__":
  test.main()
