# [MODULE DOC]
# Tests and a benchmark for the matrix solve op (linalg_ops.matrix_solve). The tests compare the
# op against numpy for real and complex types, with and without adjoint, with batches and with
# placeholders in graph mode; check errors for non-square matrices, mismatched dimensions and
# singular input; and check that identical concurrent solves give identical results. The benchmark
# times the op over many shapes, right-hand-side counts and adjoint settings on CPU and GPU.
# No import-time side effects other than imports.
#
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
"""Tests for tensorflow.ops.math_ops.matrix_solve."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class MatrixSolveOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the matrix solve op: a shared verification helper, a random complex matrix
  # generator, and tests for plain, batched, erroneous, singular and concurrent solves.

  def _verifySolve(self, x, y, batch_dims=None):
    # [FUNCTION DOC]
    # HELPER: checks the solve op against numpy. Inputs: x (complex matrix), y (right-hand side)
    # and batch_dims (optional list of leading batch sizes to tile over, default none). For four
    # dtypes (two real, two complex) and for adjoint on and off, casts the inputs (real types use
    # the real parts), builds the numpy reference with the matching transpose or conjugate
    # transpose, solves with TensorFlow (also via placeholders when not executing eagerly) and
    # asserts result shapes and values (tolerance 1e-5 for 32-bit types, 1e-12 otherwise).
    # Returns nothing.
    for np_type in [np.float32, np.float64, np.complex64, np.complex128]:
      if np_type == np.float32 or np_type == np.complex64:
        tol = 1e-5
      else:
        tol = 1e-12
      for adjoint in False, True:
        if np_type in (np.float32, np.float64):
          a = x.real.astype(np_type)
          b = y.real.astype(np_type)
          a_np = np.transpose(a) if adjoint else a
        else:
          a = x.astype(np_type)
          b = y.astype(np_type)
          a_np = np.conj(np.transpose(a)) if adjoint else a
        if batch_dims is not None:
          a = np.tile(a, batch_dims + [1, 1])
          a_np = np.tile(a_np, batch_dims + [1, 1])
          b = np.tile(b, batch_dims + [1, 1])
        np_ans = np.linalg.solve(a_np, b)
        for use_placeholder in set((False, not context.executing_eagerly())):
          if use_placeholder:
            a_ph = array_ops.placeholder(dtypes.as_dtype(np_type))
            b_ph = array_ops.placeholder(dtypes.as_dtype(np_type))
            tf_ans = linalg_ops.matrix_solve(a_ph, b_ph, adjoint=adjoint)
            with self.cached_session() as sess:
              out = sess.run(tf_ans, {a_ph: a, b_ph: b})
          else:
            tf_ans = linalg_ops.matrix_solve(a, b, adjoint=adjoint)
            out = self.evaluate(tf_ans)
            self.assertEqual(tf_ans.get_shape(), out.shape)
          self.assertEqual(np_ans.shape, out.shape)
          self.assertAllClose(np_ans, out, atol=tol, rtol=tol)

  def _generateMatrix(self, m, n):
    # [FUNCTION DOC]
    # HELPER: returns an m by n complex128 numpy matrix whose real and imaginary parts are drawn
    # independently from a normal distribution with mean minus five and standard deviation five
    # (the arguments are mean and scale, so the values are not centered on zero). Uses the global
    # numpy random state (unseeded here, so results vary between runs).
    matrix = (np.random.normal(-5, 5,
                               m * n).astype(np.complex128).reshape([m, n]))
    matrix.imag = (np.random.normal(-5, 5, m * n).astype(np.complex128).reshape(
        [m, n]))
    return matrix

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testSolve(self):
    # [FUNCTION DOC]
    # Test: for sizes 1, 2, 4 and 9 and right-hand-side column counts 1, 2 and n, random complex
    # systems are solved and verified against numpy. Graph and eager modes, GPU allowed.
    for n in 1, 2, 4, 9:
      matrix = self._generateMatrix(n, n)
      for nrhs in 1, 2, n:
        rhs = self._generateMatrix(n, nrhs)
        self._verifySolve(matrix, rhs)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testSolveBatch(self):
    # [FUNCTION DOC]
    # Test: random complex systems of size 2 and 5 with one or n right-hand sides are tiled over
    # batch shapes (2), (2, 2) and (7, 4) and verified against numpy.
    for n in 2, 5:
      matrix = self._generateMatrix(n, n)
      for nrhs in 1, n:
        rhs = self._generateMatrix(n, nrhs)
        for batch_dims in [[2], [2, 2], [7, 4]]:
          self._verifySolve(matrix, rhs, batch_dims=batch_dims)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNonSquareMatrix(self):
    # [FUNCTION DOC]
    # Test: solving with a 2x3 matrix as both operands raises a value error (graph mode) or an
    # invalid-argument error (eager mode).
    # When the solve of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      matrix = constant_op.constant([[1., 2., 3.], [3., 4., 5.]])
      self.evaluate(linalg_ops.matrix_solve(matrix, matrix))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # Test: raises a value error or invalid-argument error for (1) a 2x2 matrix with a
    # right-hand side that has a different number of rows, and (2) batched inputs whose batch
    # dimensions differ (2 by 6 versus 2 by 3).
    # The matrix and right-hand sides should have the same number of rows.
    matrix = constant_op.constant([[1., 0.], [0., 1.]])
    rhs = constant_op.constant([[1., 0.]])
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      self.evaluate(linalg_ops.matrix_solve(matrix, rhs))

    # The matrix and right-hand side should have the same batch dimensions
    matrix = np.random.normal(size=(2, 6, 2, 2))
    rhs = np.random.normal(size=(2, 3, 2, 2))
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      self.evaluate(linalg_ops.matrix_solve(matrix, rhs))

  def testNotInvertible(self):
    # [FUNCTION DOC]
    # Test: a singular 3x3 matrix (rows summing to zero) makes the solve fail at evaluation with
    # an op error saying the input matrix is not invertible.
    # The input should be invertible.
    with self.assertRaisesOpError("Input matrix is not invertible."):
      # All rows of the matrix below add to zero
      matrix = constant_op.constant([[1., 0., -1.], [-1., 1., 0.],
                                     [0., -1., 1.]])
      self.evaluate(linalg_ops.matrix_solve(matrix, matrix))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testConcurrent(self):
    # [FUNCTION DOC]
    # Test: for adjoint off and on, builds two identical solves (same stateless seeds for the
    # matrix and right-hand side), evaluates all together and asserts each pair of results is
    # identical.
    seed = [42, 24]
    matrix_shape = [3, 3]
    all_ops = []
    for adjoint_ in False, True:
      lhs1 = stateless_random_ops.stateless_random_normal(
          matrix_shape, seed=seed)
      lhs2 = stateless_random_ops.stateless_random_normal(
          matrix_shape, seed=seed)
      rhs1 = stateless_random_ops.stateless_random_normal(
          matrix_shape, seed=seed)
      rhs2 = stateless_random_ops.stateless_random_normal(
          matrix_shape, seed=seed)
      s1 = linalg_ops.matrix_solve(lhs1, rhs1, adjoint=adjoint_)
      s2 = linalg_ops.matrix_solve(lhs2, rhs2, adjoint=adjoint_)
      all_ops += [s1, s2]
    val = self.evaluate(all_ops)
    for i in range(0, len(all_ops), 2):
      self.assertAllEqual(val[i], val[i + 1])


class MatrixSolveBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark for the matrix solve op. The class attribute matrix_shapes lists the square matrix
  # shapes (with optional batch dimension) to time.

  matrix_shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1001, 1001),
      (1024, 1024),
      (2048, 2048),
      (513, 4, 4),
      (513, 16, 16),
      (513, 256, 256),
  ]

  def _GenerateTestData(self, matrix_shape, num_rhs):
    # [FUNCTION DOC]
    # HELPER: builds benchmark inputs. Inputs: matrix_shape (optional leading batch entries then
    # equal rows and columns; raises an assertion error if not square) and num_rhs (columns of
    # the right-hand side). Makes a float32 well-conditioned matrix (identity plus a small
    # constant) and a ones right-hand side, tiles both over the batch and wraps them as
    # non-trainable variables. Returns the pair (matrix variable, rhs variable).
    batch_shape = matrix_shape[:-2]
    matrix_shape = matrix_shape[-2:]
    assert matrix_shape[0] == matrix_shape[1]
    n = matrix_shape[0]
    matrix = (np.ones(matrix_shape).astype(np.float32) /
              (2.0 * n) + np.diag(np.ones(n).astype(np.float32)))
    rhs = np.ones([n, num_rhs]).astype(np.float32)
    matrix = variables.Variable(
        np.tile(matrix, batch_shape + (1, 1)), trainable=False)
    rhs = variables.Variable(
        np.tile(rhs, batch_shape + (1, 1)), trainable=False)
    return matrix, rhs

  def benchmarkMatrixSolveOp(self):
    # [FUNCTION DOC]
    # Benchmark: for each adjoint setting, shape, and right-hand-side count of 1, 2 or the number
    # of columns, builds data and times the solve for at least 25 iterations in a fresh graph and
    # session on CPU (memory usage not stored); repeats on the GPU when one is available. Side
    # effects: creates graphs and sessions and emits benchmark reports.
    run_gpu_test = test.is_gpu_available(True)
    for adjoint in False, True:
      for matrix_shape in self.matrix_shapes:
        for num_rhs in 1, 2, matrix_shape[-1]:

          with ops.Graph().as_default(), \
              session.Session(config=benchmark.benchmark_config()) as sess, \
              ops.device("/cpu:0"):
            matrix, rhs = self._GenerateTestData(matrix_shape, num_rhs)
            x = linalg_ops.matrix_solve(matrix, rhs, adjoint=adjoint)
            self.evaluate(variables.global_variables_initializer())
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(x),
                min_iters=25,
                store_memory_usage=False,
                name=("matrix_solve_cpu_shape_{matrix_shape}_num_rhs_{num_rhs}_"
                      "adjoint_{adjoint}").format(
                          matrix_shape=matrix_shape,
                          num_rhs=num_rhs,
                          adjoint=adjoint))

          if run_gpu_test:
            with ops.Graph().as_default(), \
                session.Session(config=benchmark.benchmark_config()) as sess, \
                ops.device("/gpu:0"):
              matrix, rhs = self._GenerateTestData(matrix_shape, num_rhs)
              x = linalg_ops.matrix_solve(matrix, rhs, adjoint=adjoint)
              self.evaluate(variables.global_variables_initializer())
              self.run_op_benchmark(
                  sess,
                  control_flow_ops.group(x),
                  min_iters=25,
                  store_memory_usage=False,
                  name=("matrix_solve_gpu_shape_{matrix_shape}_num_rhs_"
                        "{num_rhs}_adjoint_{adjoint}").format(
                            matrix_shape=matrix_shape, num_rhs=num_rhs,
                            adjoint=adjoint))


if __name__ == "__main__":
  test.main()
