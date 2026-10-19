# [MODULE DOC]
# Tests for the reverse-sequence op, which reverses variable-length prefixes of a tensor along a
# sequence axis for each entry along a batch axis. One test case checks many element types
# (including strings) with permuted axes and both integer length types, gradient correctness,
# shape inference with unknown input shape, and error handling for invalid lengths and axes.
# No import-time side effects besides running the test main when executed as a script.
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

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@test_util.with_eager_op_as_function
class ReverseSequenceTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the reverse-sequence op, decorated so eager ops also run as compiled functions.
  # Private helpers run the op and compare with an expected array on CPU and GPU; a shared basic
  # scenario is reused with different element types by the public test methods.

  def _testReverseSequence(self,
                           x,
                           batch_axis,
                           seq_axis,
                           seq_lengths,
                           truth,
                           use_gpu=False,
                           expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: runs the reverse-sequence op on x with the given batch axis, sequence axis and
    # per-batch lengths in a cached session (GPU placement when use_gpu, default false). If no
    # error pattern is given, evaluates and asserts the result equals truth (exact comparison for
    # strings, near-equality otherwise) and that the static shape matches truth. If expected_err_re
    # is given, asserts evaluation fails with an op error matching that pattern.
    with self.cached_session(use_gpu=use_gpu):
      ans = array_ops.reverse_sequence(
          x, batch_axis=batch_axis, seq_axis=seq_axis, seq_lengths=seq_lengths)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        if ans.dtype == dtypes.string:
          self.assertAllEqual(tf_ans, truth)
        else:
          self.assertAllClose(tf_ans, truth, atol=1e-10)
        self.assertShapeEqual(truth, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBothReverseSequence(self,
                               x,
                               batch_axis,
                               seq_axis,
                               seq_lengths,
                               truth,
                               expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: runs the single-device reverse-sequence check twice, first with GPU placement and
    # then on CPU, forwarding the input, axes, lengths, expected result and optional error pattern.
    self._testReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth, True,
                              expected_err_re)
    self._testReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth,
                              False, expected_err_re)

  def _testBasic(self, dtype, len_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: shared scenario. Builds a three by two by four array of numbers 1 to 24 (turned into
    # single-character byte strings when dtype is a string type), permutes it so the sequence axis
    # is zero and batch axis is two, uses lengths three, zero and four (of integer type len_dtype,
    # default 64-bit) and expects each batch entry reversed over its prefix length. Verifies on CPU
    # and GPU via the shared check. Inputs: numpy element type and length integer type.
    x_values = [
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24]]
    ]
    truth_values = [
        [[3, 2, 1, 4], [7, 6, 5, 8]],  # reverse 0:3
        [[9, 10, 11, 12], [13, 14, 15, 16]],  # reverse none
        [[20, 19, 18, 17], [24, 23, 22, 21]]  # reverse 0:4 (all)
    ]
    if np.dtype(dtype).kind == "S":
      def int_to_utf8(i):
        # [FUNCTION DOC]
        # HELPER: converts an integer to the single-character byte string with that code point,
        # used to turn the numeric test data into string test data.
        return chr(i).encode("utf8")
      x_values = nest.map_structure(int_to_utf8, x_values)
      truth_values = nest.map_structure(int_to_utf8, truth_values)
    x = np.asarray(x_values, dtype=dtype)
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    # reverse dim 2 up to (0:3, none, 0:4) along dim=0
    seq_lengths = np.asarray([3, 0, 4], dtype=len_dtype)

    truth_orig = np.asarray(truth_values, dtype=dtype)
    truth_orig = truth_orig.reshape(3, 2, 4, 1, 1)
    truth = truth_orig.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    seq_axis = 0  # permute seq_axis and batch_axis (originally 2 and 0, resp.)
    batch_axis = 2
    self._testBothReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth)

  def testSeqLengthInt32(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with float32 data and 32-bit integer sequence lengths.
    self._testBasic(np.float32, np.int32)

  def testFloatBasic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with float32 data and 64-bit lengths.
    self._testBasic(np.float32)

  def testFloat16Basic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with float16 data.
    self._testBasic(np.float16)

  def testBfloat16Basic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with bfloat16 data.
    self._testBasic(dtypes.bfloat16.as_numpy_dtype)

  def testDoubleBasic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with float64 data.
    self._testBasic(np.float64)

  def testInt32Basic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with int32 data.
    self._testBasic(np.int32)

  def testInt64Basic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with int64 data.
    self._testBasic(np.int64)

  def testComplex64Basic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with complex64 data.
    self._testBasic(np.complex64)

  def testComplex128Basic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with complex128 data.
    self._testBasic(np.complex128)

  def testStringBasic(self):
    # [FUNCTION DOC]
    # Runs the basic scenario with one-byte string data, exercising string elements.
    self._testBasic(np.dtype("S1"))

  def testFloatReverseSequenceGrad(self):
    # [FUNCTION DOC]
    # Gradient test on a permuted float64 array with sequence axis zero, batch axis two and
    # lengths three, zero, four. Uses the version-2 gradient checker to compare theoretical and
    # numerical Jacobians and asserts the maximum error is below 1e-8.
    x = np.asarray(
        [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]],
         [[17, 18, 19, 20], [21, 22, 23, 24]]],
        dtype=np.float64)
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # transpose axes 0 <=> 2

    # reverse dim 0 up to (0:3, none, 0:4) along dim=2
    seq_axis = 0
    batch_axis = 2
    seq_lengths = np.asarray([3, 0, 4], dtype=np.int64)

    def reverse_sequence(x):
      # [FUNCTION DOC]
      # HELPER: closure applying the reverse-sequence op to the given tensor using the enclosing
      # axes and a constant tensor of lengths; the function under test for the gradient checker.
      seq_lengths_t = constant_op.constant(seq_lengths, shape=seq_lengths.shape)
      return array_ops.reverse_sequence(
          x,
          batch_axis=batch_axis,
          seq_axis=seq_axis,
          seq_lengths=seq_lengths_t)

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(reverse_sequence, [x]))
      self.assertLess(err, 1e-8)

  def testShapeFunctionEdgeCases(self):
    # [FUNCTION DOC]
    # In graph mode, applies the op to a placeholder of unknown shape with a 32-length lengths
    # placeholder and asserts the inferred output rank is unknown.
    # Enter graph mode since we want to test partial shapes
    with context.graph_mode():
      t = array_ops.reverse_sequence(
          array_ops.placeholder(dtypes.float32, shape=None),
          seq_lengths=array_ops.placeholder(dtypes.int64, shape=(32,)),
          batch_axis=0,
          seq_axis=1)
      self.assertIs(t.get_shape().ndims, None)

  def testInvalidArguments(self):
    # [FUNCTION DOC]
    # Asserts errors (value error or invalid-argument error) for: lengths vector too long or too
    # short for the batch size, lengths of rank two, sequence axis out of range, batch axis out of
    # range, and, at evaluation time, batch axis equal to sequence axis (both zero).
    # Batch size mismatched between input and seq_lengths.
    # seq_length too long
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                (r"Dimensions must be equal|"
                                 r"Length of seq_lengths != input.dims\(0\)")):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [2, 2, 2], seq_axis=1)

    # seq_length too short
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                (r"Dimensions must be equal|"
                                 r"Length of seq_lengths != input.dims\(0\)")):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [2], seq_axis=1)

    # Invalid seq_length shape
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                ("Shape must be rank 1 but is rank 2|"
                                 "seq_lengths must be 1-dim")):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [[2, 2]], seq_axis=1)

    # seq_axis out of bounds.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "seq_dim must be < input rank"):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [2, 2], seq_axis=2)

    # batch_axis out of bounds.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "batch_dim must be < input rank"):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [2, 2],
                                 seq_axis=1,
                                 batch_axis=3)

    with self.assertRaisesRegex((errors.OpError, errors.InvalidArgumentError),
                                "batch_dim == seq_dim == 0"):
      output = array_ops.reverse_sequence([[1, 2], [3, 4]], [2, 2], seq_axis=0)
      self.evaluate(output)


if __name__ == "__main__":
  test.main()
