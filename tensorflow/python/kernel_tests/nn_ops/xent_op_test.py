# [MODULE DOC]
# Concrete tests and benchmarks for the raw softmax-cross-entropy-with-logits op
# (gen_nn_ops.softmax_cross_entropy_with_logits). The XentOpTest class extends the shared base test
# case (xent_op_test_base) with tests for rank limits, feature broadcasting and non-matrix input.
# XentBenchmark measures speed of the public API on degenerate shapes. Running as a script starts
# the TensorFlow test runner. No import-time side effects.
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
"""Tests for SoftmaxCrossEntropyWithLogits op."""

import itertools
import sys

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import xent_op_test_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class XentOpTest(xent_op_test_base.XentOpTestBase):
  # [CLASS DOC]
  # Test case for the raw softmax-cross-entropy op. Inherits the common value and gradient tests
  # from the shared base class and adds tests for rank-3 input rejection, broadcasting of the
  # features along the batch dimension, and rejection of non-matrix input. Stateless.

  @test_util.run_deprecated_v1
  def testRankTooLarge(self):
    # [FUNCTION DOC]
    # Scenario: for float16 and float32, calls the raw op with rank-3 features and labels.
    # Expects a value error mentioning that rank 2 was expected but rank 3 was given. Graph-mode
    # only (deprecated v1).
    for dtype in np.float16, np.float32:
      np_features = np.array([[[1., 1., 1., 1.]], [[1., 2., 3.,
                                                    4.]]]).astype(dtype)
      np_labels = np.array([[[0., 0., 0., 1.]], [[0., .5, .5,
                                                  0.]]]).astype(dtype)
      self.assertRaisesRegex(ValueError, "rank 2, but is rank 3",
                             gen_nn_ops.softmax_cross_entropy_with_logits,
                             np_features, np_labels)

  def testFeaturesBroadcast(self):
    # [FUNCTION DOC]
    # Scenario: first, a 1x4 float32 logits row is broadcast against 2x4 labels; the op loss and
    # gradient must match the NumPy reference computed with explicitly tiled logits. Second, a
    # 1x1 logits tensor is broadcast against two 1x1 labels; loss must be zeros and gradient
    # zeros. Uses dtype-aware closeness assertions.
    np_f = np.array([[1., 2., 3., 4.],
                     [1., 2., 3., 4.]]).astype(np.float32)
    np_l = np.array([[0., 0., 0., 1.],
                     [0., .5, .5, 0.]]).astype(np.float32)
    np_loss, np_gradient = self._npXent(labels=np_l, logits=np_f)
    tf_f = constant_op.constant(
        np.array([[1., 2., 3., 4.]]).astype(np.float32))
    tf_l = constant_op.constant(
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32))
    tf_loss, tf_gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
        tf_f, tf_l)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

    tf_f = constant_op.constant(np.array([[1.]]).astype(np.float32))
    tf_l = constant_op.constant(np.array([[1.], [1.]]).astype(np.float32))
    tf_loss, tf_gradient = gen_nn_ops.softmax_cross_entropy_with_logits(
        tf_f, tf_l)
    self.assertAllClose([0, 0], tf_loss)
    self.assertAllCloseAccordingToType([[0], [0]], tf_gradient)

  @test_util.run_deprecated_v1
  def testNotMatrix(self):
    # [FUNCTION DOC]
    # Scenario: calls the raw op with plain rank-1 lists for logits and labels. Expects a value
    # error because the op requires matrices. Graph-mode only (deprecated v1).
    with self.cached_session():
      with self.assertRaises(ValueError):
        gen_nn_ops.softmax_cross_entropy_with_logits([0., 1., 2., 3.],
                                                     [0., 1., 0., 1.])


class XentBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class timing the public softmax cross-entropy API for degenerate input shapes
  # (zero-sized batch and single class). Each benchmark method loops over a parameter grid, builds
  # a graph, runs it in a session and prints timing; results are reported through the benchmark
  # framework.

  def benchmarkZeroDimension(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the loop variables n, p and k only affect the benchmark name and the throughput
    # arithmetic; the graph always uses fixed zero-by-two-by-four inputs, so every grid point
    # measures the same op, and the throughput figure is not based on data actually processed.
    # Scenario: for each combination of batch 128, class counts and fraction values (CPU only),
    # skips zero k, builds labels and logits of shape zero by two by four on the chosen device,
    # runs the cross-entropy op for at least 100 iterations via the benchmark helper, then prints
    # wall time and a derived throughput and flushes standard output. Returns nothing.
    for (m, n, p, use_gpu) in itertools.product(
        [128],
        [10, 100, 1000, 10000, 100000],
        [0.001, 0.01, 0.5, 0.99, 1.0],
        [False]):
      k = int(p * n)
      if k == 0:
        continue
      name = "zero_dimension_m_%d_n_%d_k_%g_use_gpu_%s" % (m, n, k, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          labels = array_ops.zeros([0, 2, 4], dtype=dtypes.float32)
          logits = array_ops.zeros([0, 2, 4], dtype=dtypes.float32)
          op = nn_ops.softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
        with session.Session() as sess:
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          throughput = gb_processed_input / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()

  def benchmarkSingleClass(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the loop variables n, p and k only affect the benchmark name and the throughput
    # arithmetic; the graph always uses fixed three-by-one constants, so every grid point measures
    # the same op, and the throughput figure is not based on data actually processed.
    # Scenario: same parameter grid as the zero-dimension benchmark (CPU only), but the graph uses
    # fixed 3x1 float32 labels and logits constants. Runs the cross-entropy op for at least 100
    # iterations via the benchmark helper, prints wall time and derived throughput, flushes
    # standard output. Returns nothing.
    for (m, n, p, use_gpu) in itertools.product(
        [128],
        [10, 100, 1000, 10000, 100000],
        [0.001, 0.01, 0.5, 0.99, 1.0],
        [False]):
      k = int(p * n)
      if k == 0:
        continue
      name = "single_class_m_%d_n_%d_k_%g_use_gpu_%s" % (m, n, k, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          labels = constant_op.constant([[1.], [-1.], [0.]],
                                        dtype=dtypes.float32)
          logits = constant_op.constant([[-1.], [0.], [1.]],
                                        dtype=dtypes.float32)
          op = nn_ops.softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
        with session.Session() as sess:
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          throughput = gb_processed_input / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()


if __name__ == "__main__":
  test.main()
