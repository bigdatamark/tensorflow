# [MODULE DOC]
# Unit tests for the softmax and log-softmax ops (nn_ops.softmax and nn_ops.log_softmax). Compares
# results with a NumPy reference for several dtypes, ranks and axes, and checks overflow handling,
# shape inference, empty input, invalid axes and large inputs. GPU-only tests are skipped unless
# TensorFlow is built with GPU support. No import-time side effects; running as a script starts
# the TensorFlow test runner.
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
"""Tests for SoftmaxOp and LogSoftmaxOp."""

import unittest

import numpy as np


from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class SoftmaxTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the softmax and log-softmax ops. Compares op output with a NumPy reference for
  # several dtypes (float32, float16, float64, bfloat16), tensor ranks and reduction axes, checks
  # overflow robustness, shape inference, empty inputs, invalid axes and large inputs. GPU-only
  # variants are skipped unless TensorFlow was built with GPU support and a CUDA GPU is present.

  def _npSoftmax(self, features, dim=-1, log=False):
    # [FUNCTION DOC]
    # HELPER: NumPy reference implementation of softmax / log-softmax for the tests.
    # Inputs: a NumPy array of features, the axis to normalise over (default last axis; negative
    # one means the last axis), and a flag selecting log-softmax instead of softmax. Subtracts the
    # per-axis maximum for stability, exponentiates and normalises; float16 input is computed in
    # float32 and cast back. Returns an array of the same shape. No side effects.
    if dim == -1:
      dim = len(features.shape) - 1
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    is_fp16 = features.dtype == np.float16
    if is_fp16:
      # Do the compute in fp32 and cast the input back to fp32.
      features = features.astype(np.float32)
    e = np.exp(features - np.reshape(
        np.amax(
            features, axis=dim), one_only_on_dim))
    softmax = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    if log:
      res = np.log(softmax)
    else:
      res = softmax
    if is_fp16:
      res = res.astype(np.float16)
    return res

  def _testSoftmax(self,
                   np_features,
                   dim=-1,
                   log=False,
                   dtype=None,
                   use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: shared checker comparing the softmax / log-softmax op against the NumPy reference.
    # Inputs: a NumPy feature array, axis (default last), log flag selecting log-softmax, optional
    # dtype to cast the input to inside the session (default none), and a flag requesting a
    # GPU-enabled session. Runs the op with an arbitrary op name (so the implementation cannot rely
    # on the name to choose log versus non-log), asserts closeness according to the dtype and that
    # the static shape matches. For plain softmax without a dtype cast it also asserts that values
    # sum to one along the axis. Returns nothing; raises on assertion failure.
    # A previous version of the code checked the op name rather than the op type
    # to distinguish between log and non-log.  Use an arbitrary name to catch
    # this bug in future.
    name = "arbitrary"
    np_softmax = self._npSoftmax(np_features, dim=dim, log=log)
    with self.cached_session(use_gpu=use_gpu):
      if dtype is not None:
        np_features = math_ops.cast(np_features, dtype=dtype)

      if log:
        tf_softmax = nn_ops.log_softmax(np_features, axis=dim, name=name)
      else:
        tf_softmax = nn_ops.softmax(np_features, axis=dim, name=name)
      out = self.evaluate(tf_softmax)
    self.assertAllCloseAccordingToType(np_softmax, out)
    self.assertShapeEqual(np_softmax, tf_softmax)
    if not log and dtype is None:
      # Bonus check: the softmaxes should add to one in dimension dim.
      sum_along_dim = np.sum(out, axis=dim)
      self.assertAllCloseAccordingToType(
          np.ones(sum_along_dim.shape), sum_along_dim)

  def _testAll(self, features, dtype=None):
    # [FUNCTION DOC]
    # HELPER: runs the common GPU-requested check set for one feature array.
    # KNOWN ISSUE: the overflow check it calls does not use the given features or dtype, so it is
    # repeated unchanged for every dtype.
    # Inputs: a NumPy feature array and an optional dtype to cast to. Checks softmax, then
    # log-softmax, with a GPU-enabled session, then the overflow check. Returns nothing.
    self._testSoftmax(features, dtype=dtype, use_gpu=True)
    self._testSoftmax(features, dtype=dtype, log=True, use_gpu=True)
    self._testOverflow(use_gpu=True)

  def testNpSoftmax(self):
    # [FUNCTION DOC]
    # Scenario: validates the NumPy reference itself on a 2x4 array (a row of ones and a row of
    # one to four). Asserts softmax and log-softmax equal hand-computed constants within 1e-5.
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    # Batch 0: All exps are 1.  The expected result is
    # Softmaxes = [0.25, 0.25, 0.25, 0.25]
    # LogSoftmaxes = [-1.386294, -1.386294, -1.386294, -1.386294]
    #
    # Batch 1:
    # exps = [1., 2.718, 7.389, 20.085]
    # sum = 31.192
    # Softmaxes = exps / sum = [0.0320586, 0.08714432, 0.23688282, 0.64391426]
    # LogSoftmaxes = [-3.44019 , -2.44019 , -1.44019 , -0.44019]
    np_sm = self._npSoftmax(np.array(features))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.0320586, 0.08714432, 0.23688282, 0.64391426]]),
        np_sm,
        rtol=1.e-5,
        atol=1.e-5)
    np_lsm = self._npSoftmax(np.array(features), log=True)
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [-3.4401897, -2.4401897, -1.4401897, -0.4401897]]),
        np_lsm,
        rtol=1.e-5,
        atol=1.e-5)

  def _testOverflow(self, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: shared overflow check for log-softmax.
    # Input: a flag; when true uses float32, otherwise float64. Feeds a 2x4 array whose second row
    # contains the largest finite value of the dtype and asserts log-softmax gives finite expected
    # values (zero for the maximum and minus the maximum for the others) within 1e-5. Returns
    # nothing; raises on assertion failure.
    if use_gpu:
      type = np.float32  # pylint: disable=redefined-builtin
    else:
      type = np.float64  # pylint: disable=redefined-builtin
    max = np.finfo(type).max  # pylint: disable=redefined-builtin
    features = np.array([[1., 1., 1., 1.], [max, 1., 2., 3.]]).astype(type)
    with self.cached_session(use_gpu=use_gpu):
      tf_log_softmax = nn_ops.log_softmax(features)
      out = self.evaluate(tf_log_softmax)
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [0, -max, -max, -max]]),
        out,
        rtol=1.e-5,
        atol=1.e-5)

  def testFloat(self):
    # [FUNCTION DOC]
    # Scenario: full softmax / log-softmax / overflow check set on a small 2x4 float32 array.
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32))

  @unittest.skipUnless(test.is_built_with_gpu_support(),
                       "Test only applicable when running on GPUs")
  def testFloatGPU(self):
    # [FUNCTION DOC]
    # Scenario (skipped unless built with GPU support; does nothing if no CUDA GPU is available):
    # runs the full check set on random float32 matrices whose sizes are powers of two plus a
    # random offset below 16 (three row/column pairs), logging each shape. Uses unseeded random
    # data.
    if test.is_gpu_available(cuda_only=True):
      rows = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      cols = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      for row, col in zip(rows, cols):
        logging.info("Testing softmax float dtype in shape [%d, %d]", row, col)
        data = np.random.rand(row, col)
        self._testAll(data.astype(np.float32))

  def testHalf(self):
    # [FUNCTION DOC]
    # Scenario: full check set on a small 2x4 float16 array.
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16))

  @unittest.skipUnless(test.is_built_with_gpu_support(),
                       "Test only applicable when running on GPUs")
  def testHalfGPU(self):
    # [FUNCTION DOC]
    # Scenario (skipped unless built with GPU support; does nothing if no CUDA GPU is available):
    # full check set on random float16 matrices with randomised power-of-two based shapes (three
    # row/column pairs), logging each shape. Uses unseeded random data.
    if test.is_gpu_available(cuda_only=True):
      rows = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      cols = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      for row, col in zip(rows, cols):
        logging.info("Testing softmax half dtype in shape [%d, %d]", row, col)
        data = np.random.rand(row, col)
        self._testAll(data.astype(np.float16))

  def testDouble(self):
    # [FUNCTION DOC]
    # Scenario: softmax (non-log, CPU session) on a small 2x4 float64 array, followed by the
    # float64 overflow check.
    self._testSoftmax(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64))
    self._testOverflow()

  @unittest.skipUnless(test.is_built_with_gpu_support(),
                       "Test only applicable when running on GPUs")
  def testDoubleGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the log message says "float dtype" although the data is float64 (copy-paste).
    # Scenario (skipped unless built with GPU support; does nothing if no CUDA GPU is available):
    # full check set on random float64 matrices with randomised power-of-two based shapes (three
    # row/column pairs), logging each shape. Uses unseeded random data.
    if test.is_gpu_available(cuda_only=True):
      rows = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      cols = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      for row, col in zip(rows, cols):
        logging.info("Testing softmax float dtype in shape [%d, %d]", row, col)
        data = np.random.rand(row, col)
        self._testAll(data.astype(np.float64))

  def testBfloat16(self):
    # [FUNCTION DOC]
    # Scenario: full check set on a small 2x4 float32 array cast to bfloat16 inside the session.
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32),
        dtype=dtypes.bfloat16)

  @unittest.skipUnless(test.is_built_with_gpu_support(),
                       "Test only applicable when running on GPUs")
  def testBfloat16GPU(self):
    # [FUNCTION DOC]
    # Scenario (skipped unless built with GPU support; does nothing if no CUDA GPU is available):
    # full check set on random bfloat16 matrices with randomised power-of-two based shapes (three
    # row/column pairs), logging each shape. Uses unseeded random data.
    if test.is_gpu_available(cuda_only=True):
      rows = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      cols = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      for row, col in zip(rows, cols):
        logging.info("Testing softmax bfloat16 dtype in shape [%d, %d]", row,
                     col)
        data = np.random.rand(row, col)
        self._testAll(data.astype(dtypes.bfloat16.as_numpy_dtype))

  def test1DTensorAsInput(self):
    # [FUNCTION DOC]
    # Scenario: softmax on a rank-1 float64 array with a CPU session, then the CPU overflow check.
    self._testSoftmax(
        np.array([3., 2., 3., 9.]).astype(np.float64), use_gpu=False)
    self._testOverflow(use_gpu=False)

  def test1DTensorAsInputNoReshape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the body is identical to the test of the same name without the no-reshape
    # suffix, so it does not exercise any distinct no-reshape path.
    # Scenario: softmax on a rank-1 float64 array with a CPU session, then the CPU overflow check.
    self._testSoftmax(
        np.array([3., 2., 3., 9.]).astype(np.float64), use_gpu=False)
    self._testOverflow(use_gpu=False)

  def test3DTensorAsInput(self):
    # [FUNCTION DOC]
    # Scenario: softmax over the last axis of a 3x2x4 float32 array with a CPU session, then the
    # CPU overflow check.
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def test3DTensorAsInputNoReshape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the body is identical to the test of the same name without the no-reshape
    # suffix, so it does not exercise any distinct no-reshape path.
    # Scenario: softmax over the last axis of a 3x2x4 float32 array with a CPU session, then the
    # CPU overflow check.
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def testAlongFirstDimension(self):
    # [FUNCTION DOC]
    # Scenario: softmax over axis zero of a 3x2x4 float32 array (CPU), then the CPU overflow check.
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        dim=0,
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def testAlongSecondDimension(self):
    # [FUNCTION DOC]
    # Scenario: softmax over axis one of a 3x2x4 float32 array (CPU), then the CPU overflow check.
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        dim=1,
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def testAlongNegativeDimension(self):
    # [FUNCTION DOC]
    # Scenario: softmax over axis minus two (the middle axis) of a 3x2x4 float32 array (CPU), then
    # the CPU overflow check.
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        dim=-2,
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def testShapeInference(self):
    # [FUNCTION DOC]
    # Scenario: builds softmax on a nested-list 3x2x4 constant and asserts the op's static shape is
    # three by two by four.
    op = nn_ops.softmax([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                         [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                         [[5., 4., 3., 2.], [1., 2., 3., 4.]]])
    self.assertEqual([3, 2, 4], op.get_shape())

  def testEmptyInput(self):
    # [FUNCTION DOC]
    # Scenario: softmax over axis zero of a float32 tensor of shape zero by three. Asserts the
    # input has zero elements and the result equals an empty array of the same shape.
    x = array_ops.ones(shape=[0, 3], dtype=dtypes.float32)
    y = np.zeros(shape=[0, 3], dtype=np.float32)
    self.assertEqual(0, self.evaluate(array_ops.size(x)))
    self.assertAllEqual(y, self.evaluate(nn_ops.softmax(x, axis=0)))

  def testDimTooLarge(self):
    # [FUNCTION DOC]
    # Scenario: softmax of a rank-1 list with the axis supplied through a placeholder defaulting
    # to 100, so the failure occurs at run time rather than shape inference. Expects an
    # invalid-argument error when evaluated.
    with self.cached_session():
      # Use placeholder to make sure we get runtime error instead of shape
      # inference error.
      dim = array_ops.placeholder_with_default(100, shape=[])
      with self.assertRaises(errors_impl.InvalidArgumentError):
        nn_ops.softmax([1., 2., 3., 4.], axis=dim).eval()

  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Scenario (regression for a GitHub issue): softmax of a 2x3 tensor of ones with axis two,
    # which is out of range for rank two. Expects an invalid-argument error when evaluated.
    # Test case for GitHub issue 22793.
    with self.cached_session():
      ones = array_ops.ones(shape=[2, 3])
      with self.assertRaises(errors_impl.InvalidArgumentError):
        nn_ops.softmax(ones, axis=2).eval()

  def testLargeDims(self):
    # [FUNCTION DOC]
    # Scenario (regression for a GitHub issue on large inputs): random square float32 matrices of
    # side 129 and 256, softmax computed on both GPU-enabled and CPU sessions and compared with the
    # NumPy reference. Uses unseeded random data.
    # Make sure that we properly handle large inputs. See
    # https://github.com/tensorflow/tensorflow/issues/4425 for details
    for dims in [129, 256]:
      ones = np.random.rand(dims, dims).astype(np.float32)
      np_softmax = self._npSoftmax(ones)

      for use_gpu in [True, False]:
        with self.cached_session(use_gpu=use_gpu):
          x = constant_op.constant(ones)
          y = nn_ops.softmax(x)
          tf_softmax = self.evaluate(y)
        self.assertAllClose(tf_softmax, np_softmax)


if __name__ == "__main__":
  test.main()
