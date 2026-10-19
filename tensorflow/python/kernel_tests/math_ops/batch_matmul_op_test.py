# [MODULE DOC]
# Test and benchmark module for batched matrix multiplication. Defines a random input helper,
# forward and gradient test classes, factory functions that create test methods, and a CPU
# benchmark of built-in versus manual broadcasting. Test methods are generated only when run as
# a script: the entry point loops over dtypes, adjoint flags and static/dynamic shapes and attaches
# methods to the classes before starting the test runner. Importing the module registers none.
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
"""Tests for tensorflow.ops.tf.BatchMatMul."""

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def GetRandomNormalInput(shape, dtype):
  # [FUNCTION DOC]
  # HELPER: shared random input generator for the batch matmul tests.
  # Returns a numpy array of the given shape and dtype filled with normally distributed values
  # drawn from the global numpy random state (location minus ten and scale ten, or 0.1 and 0.1
  # for 16-bit float to stay in range). For complex dtypes an independent imaginary part is
  # added, drawn the same way. Side effect: advances the global numpy random state.
  # float16 has limited range so we reduce the variance of the scalars.
  scale = 10.0 if dtype != np.float16 else 0.1
  loc = -10.0 if dtype != np.float16 else 0.1
  vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
  if dtype in (np.complex64, np.complex128):
    imag = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    vals += 1j * imag
  return vals.reshape(shape)


class BatchMatmulOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the batched matrix multiplication op. It provides helpers that compare the op
  # with numpy for non-empty, empty and broadcasting shapes. The actual test methods are not
  # defined here: at the bottom of the module, when run as a script, test methods are attached
  # to this class for every dtype, adjoint flag and static/dynamic shape combination.

  # Uses numpy to compute batch_matmul(x, y, adjoint_a, adjoint_b).
  def _npBatchMatmul(self, x, y, adjoint_a, adjoint_b):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation. Optionally replaces x and/or y by the conjugate
    # transpose of their last two axes (when the adjoint flags are set) and returns their numpy
    # matrix product with broadcasting over leading batch dimensions.
    # output's shape depends on adj[0] and adj[1]
    if adjoint_a:
      x = np.conjugate(np.swapaxes(x, -1, -2))
    if adjoint_b:
      y = np.conjugate(np.swapaxes(y, -1, -2))
    return np.matmul(x, y)

  # Compares TensorFlow BatchMatmul with NumPy's matmul.
  def _compare(self, x_in, y_in, adjoint_a, adjoint_b, static_shape):
    # [FUNCTION DOC]
    # HELPER: runs the matmul op and compares with the numpy reference.
    # KNOWN ISSUE: when an adjoint flag is set the input is reshaped (not transposed) to the
    # swapped shape, so the data are reinterpreted rather than transposed; the comparison still
    # works because the numpy reference applies the same adjoint to the reshaped array.
    # Takes two numpy arrays, two adjoint flags and a flag saying whether to feed constants
    # (static shapes) or graph placeholders (unknown shapes). Chooses a tolerance of one hundred
    # machine epsilons for floating types (a fixed epsilon for bfloat16) and zero for integers,
    # runs the op on GPU for floating types, and asserts closeness with the reference.
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
    y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
    is_floating = x.dtype != np.int32
    # np.finfo doesn't support bfloat16. So, we manually compute the eps which
    # defines the difference between 1.0 and the next smallest representable
    # float larger than 1.0. For bfloat16, the difference is 1/128.
    if x.dtype == dtypes.bfloat16.as_numpy_dtype:
      epsilon = 0.0078125
    elif is_floating:
      epsilon = np.finfo(x.dtype).eps
    tol = 100 * epsilon if is_floating else 0
    with self.cached_session(use_gpu=is_floating) as sess:
      if static_shape:
        z0 = math_ops.matmul(x, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        z0_val = self.evaluate(z0)
      else:
        x_ph = array_ops.placeholder(x.dtype)
        y_ph = array_ops.placeholder(y.dtype)
        z0 = math_ops.matmul(
            x_ph, y_ph, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        z0_val = sess.run(z0, feed_dict={x_ph: x, y_ph: y})
      z1 = self._npBatchMatmul(x, y, adjoint_a, adjoint_b)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

  def _testNonEmpty(self, dtype, adjoint_a, adjoint_b, use_static_shape):
    # [FUNCTION DOC]
    # HELPER: compares the op with numpy for nine pairs of same-batch non-empty shapes (rank
    # three and four, including singleton dimensions and a larger 64 by 75 by 30 case), using
    # random inputs of the given dtype and the given adjoint and static-shape settings.

    def CompareNonEmpty(self, a_shape, b_shape):
      # [FUNCTION DOC]
      # HELPER: generates random inputs of the two shapes and runs the comparison with the
      # enclosing adjoint and static-shape settings.
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 1])
    CompareNonEmpty(self, [1, 1, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [7, 1, 3], [7, 3, 5])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 1])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 5])
    CompareNonEmpty(self, [10, 64, 75], [10, 75, 30])
    CompareNonEmpty(self, [5, 7, 2, 3], [5, 7, 3, 5])

  def _testBroadcasting(self, dtype, adjoint_a, adjoint_b, use_static_shape):
    # [FUNCTION DOC]
    # HELPER: compares the op with numpy for seven pairs of shapes whose batch dimensions differ
    # in rank or have singleton dimensions, so broadcasting is required (up to rank eight).

    def CompareNonEmpty(self, a_shape, b_shape):
      # [FUNCTION DOC]
      # HELPER: generates random inputs of the two shapes and runs the comparison with the
      # enclosing adjoint and static-shape settings.
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareNonEmpty(self, [2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [3, 5])
    CompareNonEmpty(self, [5, 1, 2, 3], [1, 7, 3, 5])
    CompareNonEmpty(self, [5, 2, 2, 3], [3, 5])
    CompareNonEmpty(self, [2, 3], [5, 2, 3, 5])
    CompareNonEmpty(self, [4, 5, 1, 2, 3], [1, 1, 3, 5])
    CompareNonEmpty(self, [1, 2, 1, 4, 2, 1, 3, 4], [3, 2, 1, 1, 1, 2, 4, 2])

  def _testEmpty(self, dtype, adjoint_a, adjoint_b, use_static_shape):
    # [FUNCTION DOC]
    # HELPER: compares the op with numpy for three cases containing zero-sized dimensions (empty
    # batch, empty rows, empty output columns) using all-zero inputs of the given dtype.

    def CompareEmpty(self, a_shape, b_shape):
      # [FUNCTION DOC]
      # HELPER: builds all-zero inputs of the two shapes and runs the comparison with the
      # enclosing adjoint and static-shape settings.
      self._compare(
          np.zeros(a_shape).astype(dtype),
          np.zeros(b_shape).astype(dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareEmpty(self, [0, 3, 2], [0, 2, 4])
    CompareEmpty(self, [3, 0, 2], [3, 2, 5])
    CompareEmpty(self, [3, 3, 2], [3, 2, 0])


def _GetBatchMatmulOpTest(dtype, adjoint_a, adjoint_b, use_static_shape):
  # [FUNCTION DOC]
  # HELPER: test-method factory used by the script entry point. Returns a test method bound to
  # the given dtype, adjoint flags and static-shape setting that checks the non-empty and empty
  # shape cases.

  @test_util.run_without_tensor_float_32("Tests batch matmul")
  def Test(self):
    # [FUNCTION DOC]
    # HELPER: generated test body. Seeds the global numpy random state with 42 (side effect),
    # then runs the non-empty and empty comparisons with the captured parameters; TensorFloat-32
    # is disabled for its duration.
    np.random.seed(42)
    self._testNonEmpty(dtype, adjoint_a, adjoint_b, use_static_shape)
    self._testEmpty(dtype, adjoint_a, adjoint_b, use_static_shape)

  return Test


def _GetBatchMatmulOpBroadcastingTest(dtype, adjoint_a, adjoint_b,
                                      use_static_shape):
  # [FUNCTION DOC]
  # HELPER: test-method factory used by the script entry point. Returns a test method bound to
  # the given dtype, adjoint flags and static-shape setting that checks the broadcasting cases.

  @test_util.run_without_tensor_float_32("Tests batch matmul")
  def Test(self):
    # [FUNCTION DOC]
    # HELPER: generated test body. Seeds the global numpy random state with 42 (side effect),
    # then runs the broadcasting comparison with the captured parameters; TensorFloat-32 is
    # disabled for its duration.
    np.random.seed(42)
    self._testBroadcasting(dtype, adjoint_a, adjoint_b, use_static_shape)

  return Test


class BatchMatmulGradientTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the gradients of batched matmul. Helpers run the numeric gradient checker on a
  # sum-of-product loss. Test methods are attached dynamically at the bottom of the module (when
  # run as a script) for every non-integer dtype and adjoint combination, with and without
  # broadcasting.

  # loss = sum(batch_matmul(x, y)). Verify dl/dx and dl/dy via the
  # gradient checker.
  def _checkGrad(self, x_in, y_in, adjoint_a, adjoint_b):
    # [FUNCTION DOC]
    # HELPER: compares theoretical and numerical Jacobians of the summed matmul with respect to
    # both inputs using the v2 gradient checker. Reshapes inputs when adjoint flags are set
    # (same reinterpretation as the forward test), picks epsilon by dtype, a step size of ten
    # times the cube root of epsilon, and a tolerance of ten times the step. Asserts the
    # Jacobians are close for both inputs. Returns nothing.
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
    y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
    # np.finfo doesn't support bfloat16. So, we manually compute the eps which
    # defines the difference between 1.0 and the next smallest representable
    # float larger than 1.0. For bfloat16, the difference is 1/128.
    if x.dtype == dtypes.bfloat16.as_numpy_dtype:
      epsilon = 0.0078125
    else:
      epsilon = np.finfo(x.dtype).eps
    # Since our gradient is linear, a larger delta decreases the error.
    delta = 10 * epsilon**(1.0 / 3.0)

    def Loss(x, y):
      # [FUNCTION DOC]
      # HELPER: scalar loss for the gradient checker; multiplies the two inputs with the
      # enclosing adjoint flags, casts bfloat16 results to float32 to avoid large sum errors,
      # and returns the sum of all elements.
      z = math_ops.matmul(x, y, adjoint_a, adjoint_b)
      # To avoid the high error when reduce_sum over the bfloat16 values, we
      # cast the results to float32.
      if z.dtype == dtypes.bfloat16:
        z = math_ops.cast(z, dtype=dtypes.float32)
      return math_ops.reduce_sum(z)

    with self.cached_session():
      ((x_jacob_t, y_jacob_t),
       (x_jacob_n, y_jacob_n)) = gradient_checker_v2.compute_gradient(
           Loss, [x, y], delta=delta)
      tol = 10 * delta
      self.assertAllClose(x_jacob_t, x_jacob_n, rtol=tol, atol=tol)
      self.assertAllClose(y_jacob_t, y_jacob_n, rtol=tol, atol=tol)

  # Tests gradients of a batched matmul of x, and y
  def _compare(self, a_shape, b_shape, dtype, adjoint_a, adjoint_b):
    # [FUNCTION DOC]
    # HELPER: seeds the global numpy random state with 42 (side effect), generates random inputs
    # of the two shapes and dtype, and runs the gradient check with the adjoint flags.
    np.random.seed(42)
    x = GetRandomNormalInput(a_shape, dtype)
    y = GetRandomNormalInput(b_shape, dtype)
    self._checkGrad(x, y, adjoint_a, adjoint_b)


def _GetBatchMatmulGradientTest(dtype, adjoint_a, adjoint_b):
  # [FUNCTION DOC]
  # HELPER: test-method factory used by the script entry point. Returns a test method that
  # checks gradients for two same-batch shape pairs with the given dtype and adjoint flags.

  def Test(self):
    # [FUNCTION DOC]
    # HELPER: generated test body; runs the gradient comparison for shapes one by two by three
    # with one by three by five, and three by four by seven with three by seven by ten.
    def CheckGradients(self, a_shape, b_shape):
      # [FUNCTION DOC]
      # HELPER: runs the gradient comparison for the given shapes with the captured dtype and
      # adjoint flags.
      self._compare(a_shape, b_shape, dtype, adjoint_a, adjoint_b)

    CheckGradients(self, [1, 2, 3], [1, 3, 5])
    CheckGradients(self, [3, 4, 7], [3, 7, 10])

  return Test


def _GetBatchMatmulGradientWithBroadcastingTest(dtype, adjoint_a, adjoint_b):
  # [FUNCTION DOC]
  # HELPER: test-method factory used by the script entry point. Returns a test method that
  # checks gradients for several broadcasting shape pairs with the given dtype and adjoint flags.

  def Test(self):
    # [FUNCTION DOC]
    # HELPER: generated test body; runs the gradient comparison for seven shape pairs whose
    # batch dimensions need broadcasting (ranks from two up to eight).
    def CheckGradients(self, a_shape, b_shape):
      # [FUNCTION DOC]
      # HELPER: runs the gradient comparison for the given shapes with the captured dtype and
      # adjoint flags.
      self._compare(a_shape, b_shape, dtype, adjoint_a, adjoint_b)

    CheckGradients(self, [1, 5, 2, 3], [7, 1, 3, 2])
    CheckGradients(self, [2, 3], [1, 3, 5])
    CheckGradients(self, [2, 3], [5, 3, 5])
    CheckGradients(self, [5, 2, 5], [5, 3])
    CheckGradients(self, [5, 2, 2, 3], [3, 5])
    CheckGradients(self, [4, 5, 1, 2, 3], [1, 1, 3, 5])
    CheckGradients(self, [1, 2, 1, 4, 2, 1, 3, 4], [3, 2, 1, 1, 1, 2, 4, 2])

  return Test


class BatchMatMulBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark comparing the batch matmul op's internal broadcasting with manually broadcasting
  # the inputs first, on CPU, for fully connected, square, matrix-vector and vector-matrix
  # shapes. The class attribute shape_pairs lists the shapes (batch sizes are 512).
  # Batch sizes are 512.
  shape_pairs = [
      # Typical fully connected layer.
      ((4, 8, 4, 2, 1, 1024), (1024, 1024)),
      ((4, 1, 4, 1, 1, 1024), (1, 8, 1, 2, 1024, 1024)),
      # Square matmul.
      ((4, 8, 4, 2, 512, 512), (512, 512)),
      ((4, 1, 4, 1, 512, 512), (1, 8, 1, 2, 512, 512)),
      # Matrix-vector multiplies.
      ((4, 8, 4, 2, 10000, 200), (200, 1)),
      ((4, 1, 4, 1, 10000, 200), (1, 8, 1, 2, 200, 1)),
      # Vector-matrix multiplies.
      ((4, 8, 4, 2, 1, 200), (200, 10000)),
      ((4, 1, 4, 1, 1, 200), (1, 8, 1, 2, 200, 10000)),
  ]

  def benchmarkBatchMatMulBroadcast(self):
    # [FUNCTION DOC]
    # For each shape pair, builds a fresh graph and session pinned to the CPU, creates two
    # float32 variables with random values, initializes them, and benchmarks (at least 50
    # iterations) the matmul op with built-in broadcasting, then a matmul of manually broadcast
    # inputs. Reports results through the benchmark framework under names including the shapes.
    # Returns nothing.
    for (a_shape, b_shape) in self.shape_pairs:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix_a = variables.Variable(
            GetRandomNormalInput(a_shape, np.float32))
        matrix_b = variables.Variable(
            GetRandomNormalInput(b_shape, np.float32))
        self.evaluate(variables.global_variables_initializer())

        # Use batch matmul op's internal broadcasting.
        self.run_op_benchmark(
            sess,
            math_ops.matmul(matrix_a, matrix_b),
            min_iters=50,
            name="batch_matmul_cpu_{}_{}".format(a_shape, b_shape))

        # Manually broadcast the input matrices using the broadcast_to op.
        broadcasted_batch_shape = array_ops.broadcast_static_shape(
            matrix_a.shape[:-2], matrix_b.shape[:-2])
        broadcasted_a_shape = broadcasted_batch_shape.concatenate(
            matrix_a.shape[-2:])
        broadcasted_b_shape = broadcasted_batch_shape.concatenate(
            matrix_b.shape[-2:])
        self.run_op_benchmark(
            sess,
            math_ops.matmul(
                array_ops.broadcast_to(matrix_a, broadcasted_a_shape),
                array_ops.broadcast_to(matrix_b, broadcasted_b_shape)),
            min_iters=50,
            name="batch_matmul_manual_broadcast_cpu_{}_{}".format(
                a_shape, b_shape))


if __name__ == "__main__":
  dtypes_to_test = [
      np.float16, np.float32, np.float64, np.int32, np.complex64, np.complex128,
      dtypes.bfloat16.as_numpy_dtype
  ]

  for dtype_ in dtypes_to_test:
    for adjoint_a_ in False, True:
      for adjoint_b_ in False, True:
        name = "%s_%s_%s" % (dtype_.__name__, adjoint_a_, adjoint_b_)
        # TF2 does not support placeholders under eager so we skip it.
        for use_static_shape_ in set([True, tf2.enabled()]):
          setattr(
              BatchMatmulOpTest,
              "testBatchMatmulOp_" + name + "_{}".format(use_static_shape_),
              test_util.xla_allow_fallback(
                  "TODO(b/134526360): XLA:CPU hasn't implemented int32 dot.")(
                      _GetBatchMatmulOpTest(dtype_, adjoint_a_, adjoint_b_,
                                            use_static_shape_)))
          # Broadcasting is supported only in v2.
          setattr(
              BatchMatmulOpTest, "testBatchMatmulBroadcasting_" + name +
              ("_%s" % use_static_shape_),
              test_util.xla_allow_fallback(
                  "TODO(b/134526360): XLA:CPU hasn't implemented int32 dot.")(
                      _GetBatchMatmulOpBroadcastingTest(dtype_, adjoint_a_,
                                                        adjoint_b_,
                                                        use_static_shape_)))
        if dtype_ == np.int32:
          continue
        setattr(BatchMatmulGradientTest, "testBatchMatmulGradient_" + name,
                _GetBatchMatmulGradientTest(dtype_, adjoint_a_, adjoint_b_))
        # Broadcasting is supported only in v2.
        setattr(
            BatchMatmulGradientTest,
            "testBatchMatmulGradientWithBroadcasting_" + name,
            _GetBatchMatmulGradientWithBroadcastingTest(dtype_, adjoint_a_,
                                                        adjoint_b_))
  test.main()
