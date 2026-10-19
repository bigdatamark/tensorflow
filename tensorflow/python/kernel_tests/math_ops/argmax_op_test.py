# [MODULE DOC]
# Test module for the argmax and argmin ops in the math ops module. Defines one parameterized
# test class comparing results with numpy across dtypes, axes, ties, output types, empty inputs
# and the default axis. No import-time side effects; running as a script starts the test runner.
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
"""Tests for tensorflow.ops.argmax_op."""
import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ArgMaxTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the argmax and argmin ops. Checks results against numpy for many dtypes, tie
  # breaking, all axes of a high-rank array, int32 output type, empty inputs and the default axis.
  # Private underscore methods are shared helpers; parameterized so one test covers many dtypes.

  def _testArg(self,
               method,
               x,
               axis,
               expected_values,
               use_gpu=False,
               expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: core checker for the argmax/argmin tests.
    # Applies the given reduction method (argmax or argmin) to array x along axis inside a
    # session, on GPU if use_gpu is true. If expected_err_re is nothing, evaluates the result and
    # asserts the dtype is 64-bit integer (the default output type), the values equal
    # expected_values and the static shape matches. Otherwise asserts evaluation raises an op
    # error whose message matches expected_err_re. Returns nothing.
    with self.session(use_gpu=use_gpu):
      ans = method(x, axis=axis)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        # Defaults to int64 output.
        self.assertEqual(np.int64, tf_ans.dtype)
        self.assertAllEqual(tf_ans, expected_values)
        self.assertShapeEqual(expected_values, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBothArg(self,
                   method,
                   x,
                   axis,
                   expected_values,
                   expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: runs the checker on GPU and, unless XLA is enabled (compile time is too large with
    # XLA CPU autojit), also on CPU, with the same method, input, axis and expectations.
    # Returns nothing; failures surface as assertion errors from the checker.
    self._testArg(method, x, axis, expected_values, True, expected_err_re)
    # Compilation time is too large with XLA/CPU autojit.
    if not test_util.is_xla_enabled():
      self._testArg(method, x, axis, expected_values, False, expected_err_re)

  def _testBasic(self, dtype):
    # [FUNCTION DOC]
    # HELPER: for the given numpy dtype, builds a shuffled vector of 200 distinct values and
    # checks argmax and argmin along axis 0 equal numpy's results on CPU/GPU. Uses the global
    # numpy random state for shuffling.
    x = np.arange(200, dtype=np.float32).astype(dtype)
    np.random.shuffle(x)

    # Check that argmin and argmax match numpy along the primary axis
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())

  def _testTieBreaking(self, dtype):
    # [FUNCTION DOC]
    # HELPER: checks tie-breaking behavior for the given numpy dtype. An all-zero vector and a
    # small matrix with repeated values are used, and argmax/argmin must match numpy (first
    # occurrence wins) along axis 0 and axis 1 respectively.
    x = np.zeros(200, dtype=dtype)

    # Check that argmin and argmax match numpy along the primary axis for
    # breaking ties.
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())

    # Check that argmin and argmax match numpy along axis=1 for
    # breaking ties.
    x = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]], dtype=dtype)
    self._testBothArg(math_ops.argmax, x, 1, x.argmax(axis=1))
    self._testBothArg(math_ops.argmin, x, 1, x.argmin(axis=1))

  def _testDim(self, dtype):
    # [FUNCTION DOC]
    # HELPER: for the given numpy dtype, builds a shuffled seven-dimensional array and checks
    # argmax and argmin match numpy for every axis from -7 through 6, including negative axes.
    # Each call goes through the CPU/GPU double-check helper.
    shape = (3, 2, 4, 5, 6, 3, 7)
    x = np.arange(
        functools.reduce(lambda x, y: x * y, shape),
        dtype=np.float32).astype(dtype)
    np.random.shuffle(x)
    x = x.reshape(shape)

    # Check that argmin and argmax match numpy along all axes
    for axis in range(-7, 7):
      self._testBothArg(math_ops.argmax, x, axis, x.argmax(axis))
      self._testBothArg(math_ops.argmin, x, axis, x.argmin(axis))

  @parameterized.parameters(np.float16, np.float32, np.float64, np.int16,
                            np.int32, np.int64, np.bool_,
                            dtypes.bfloat16.as_numpy_dtype)
  def testTypes(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float16/32/64, int16/32/64, bool and bfloat16 numpy dtypes. For each
    # dtype runs the basic, tie-breaking and all-axes helper checks against numpy.
    self._testBasic(dtype,)
    self._testTieBreaking(dtype)
    self._testDim(dtype)

  def testFloatInt32Output(self):
    # [FUNCTION DOC]
    # Uses a random float32 vector and requests 32-bit integer output from argmax and argmin.
    # Asserts the output dtype is 32-bit integer and the value equals numpy's argmax/argmin.
    x = np.asarray(100 * np.random.randn(200), dtype=np.float32)
    expected_values = x.argmax()
    with self.session():
      ans = math_ops.argmax(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      # The values are equal when comparing int32 to int64 because
      # the values don't have a range that exceeds 32-bit integers.
      self.assertAllEqual(tf_ans, expected_values)
    expected_values = x.argmin()
    with self.session():
      ans = math_ops.argmin(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      self.assertAllEqual(tf_ans, expected_values)

  def testEmpty(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test calls an eval method on the op result but has no graph-mode
    # decorator; in eager mode the error is raised while building the op, so the eval call is
    # never reached and the test only works because the error comes from op construction.
    # Applies argmin and argmax to an empty list along axis 0 and asserts an op error stating
    # the reduction axis 0 is empty in shape zero is raised.
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        with self.assertRaisesOpError(
            r"Reduction axis 0 is empty in shape \[0\]"):
          op([], 0).eval()

  @test_util.run_deprecated_v1
  def testDefaultAxis(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated-v1 decorator): argmin and argmax of a one-element list with
    # no axis given must evaluate to zero, showing the default axis is 0.
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        ans = op([1]).eval()
        self.assertAllEqual(ans, 0)

  @test_util.run_deprecated_v1
  def testOutputEmpty(self):
    # [FUNCTION DOC]
    # Graph-mode test: argmin and argmax over the last axis of a zeros tensor of shape one by
    # zero by two produce an empty result; asserts the evaluated shape is one by zero.
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        ret = op(array_ops.zeros(shape=[1, 0, 2]), axis=-1).eval()
        self.assertEqual(ret.shape, (1, 0))


if __name__ == "__main__":
  test.main()
