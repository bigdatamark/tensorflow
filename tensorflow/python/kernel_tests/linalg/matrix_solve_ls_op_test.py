# [MODULE DOC]
# Tests and a benchmark for the least-squares matrix solve op (linalg_ops.matrix_solve_ls). Defines
# data-generation and numpy reference helpers, a test case with a shared verification routine and a
# few direct tests (wrong dimensions, empty inputs, batch result shape), and factory functions that
# generate many parameterized test methods (small and large cases over dtype, placeholder use, fast
# versus slow path and L2 regularization). When run as a script it attaches the generated tests to
# the test class (refusing duplicate names) and starts the test runner. The benchmark times the op
# over many shapes and right-hand-side counts on CPU and (when available) GPU.
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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test as test_lib


def _AddTest(test, op_name, testcase_name, fn):
  # [FUNCTION DOC]
  # HELPER: attaches a generated test function to a test class. Inputs: test (the class to
  # modify), op_name and testcase_name (strings joined with a test prefix, by underscores, to
  # form the method name) and fn (the test function). Side effect: sets the method on the class.
  # Raises a runtime error if the class already has an attribute with that name.
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


def _GenerateTestData(matrix_shape, num_rhs):
  # [FUNCTION DOC]
  # HELPER: builds benchmark inputs. Inputs: matrix_shape (tuple: optional leading batch
  # dimensions, then rows and columns) and num_rhs (number of right-hand-side columns). Seeds the
  # global numpy random state with 1, draws a single uniform random float32 matrix in minus one to
  # one, makes a right-hand side of ones with one row per matrix row, tiles both across the batch
  # and wraps them as non-trainable variables. Returns the pair (matrix variable, rhs variable).
  batch_shape = matrix_shape[:-2]
  matrix_shape = matrix_shape[-2:]
  m = matrix_shape[-2]
  np.random.seed(1)
  matrix = np.random.uniform(
      low=-1.0, high=1.0,
      size=np.prod(matrix_shape)).reshape(matrix_shape).astype(np.float32)
  rhs = np.ones([m, num_rhs]).astype(np.float32)
  matrix = variables.Variable(
      np.tile(matrix, batch_shape + (1, 1)), trainable=False)
  rhs = variables.Variable(np.tile(rhs, batch_shape + (1, 1)), trainable=False)
  return matrix, rhs


def _SolveWithNumpy(matrix, rhs, l2_regularizer=0):
  # [FUNCTION DOC]
  # HELPER: numpy reference solution of the (regularized) least-squares problem. Inputs: matrix
  # and rhs (numpy arrays for a single system, no batching) and l2_regularizer (default zero).
  # With zero regularization it uses the numpy least-squares solver and returns the solution.
  # Otherwise, for at least as many rows as columns it solves the regularized normal equations
  # (Gram matrix plus regularizer times identity); for fewer rows it solves the regularized
  # system in row space and maps back through the conjugate transpose (minimum norm solution).
  # Returns the solution array. Does not modify its arguments.
  if l2_regularizer == 0:
    np_ans, _, _, _ = np.linalg.lstsq(matrix, rhs)
    return np_ans
  else:
    rows = matrix.shape[-2]
    cols = matrix.shape[-1]
    if rows >= cols:
      preconditioner = l2_regularizer * np.identity(cols)
      gramian = np.dot(np.conj(matrix.T), matrix) + preconditioner
      rhs = np.dot(np.conj(matrix.T), rhs)
      return np.linalg.solve(gramian, rhs)
    else:
      preconditioner = l2_regularizer * np.identity(rows)
      gramian = np.dot(matrix, np.conj(matrix.T)) + preconditioner
      z = np.linalg.solve(gramian, rhs)
      return np.dot(np.conj(matrix.T), z)


@test_util.with_eager_op_as_function
class MatrixSolveLsOpTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the least-squares solve op. Has a shared verification method, a few direct
  # tests, and receives many additional generated test methods when the module is run as a script.

  def _verifySolve(self,
                   x,
                   y,
                   dtype,
                   use_placeholder,
                   fast,
                   l2_regularizer,
                   batch_shape=()):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the lines after the tolerance assignment in the else branch are indented under
    # that else, so the whole comparison against numpy runs only for 64-bit float and 128-bit
    # complex types; for 32-bit float and 64-bit complex the method sets a tolerance and does
    # nothing else, so those cases are never actually verified.
    # HELPER: shared verification for generated tests. Inputs: x (matrix), y (right-hand side),
    # dtype (numpy dtype to cast to), use_placeholder (feed through placeholders in graph mode),
    # fast (use the fast path), l2_regularizer (regularization strength), batch_shape (tuple of
    # leading batch sizes to tile over, default none). Returns early (no check) for the slow path
    # with nonzero regularization and for placeholders under eager execution. For complex dtypes
    # sets the imaginary parts equal to the real parts. Computes the numpy reference solution and
    # residual norm, tiles over the batch, runs the TensorFlow solve and asserts the shape and
    # values agree (tolerance scaled by the largest dimension), and, when unregularized, asserts
    # the least-squares optimality residual norm matches the reference. Returns nothing.
    if not fast and l2_regularizer != 0:
      # The slow path does not support regularization.
      return
    if use_placeholder and context.executing_eagerly():
      return
    maxdim = np.max(x.shape)
    if dtype == np.float32 or dtype == np.complex64:
      tol = maxdim * 5e-4
    else:
      tol = maxdim * 5e-7
      a = x.astype(dtype)
      b = y.astype(dtype)
      if dtype in [np.complex64, np.complex128]:
        a.imag = a.real
        b.imag = b.real
      # numpy.linalg.lstqr does not batching, so we just solve a single system
      # and replicate the solution. and residual norm.
      np_ans = _SolveWithNumpy(x, y, l2_regularizer=l2_regularizer)
      np_r = np.dot(np.conj(a.T), b - np.dot(a, np_ans))
      np_r_norm = np.sqrt(np.sum(np.conj(np_r) * np_r))
      if batch_shape != ():
        a = np.tile(a, batch_shape + (1, 1))
        b = np.tile(b, batch_shape + (1, 1))
        np_ans = np.tile(np_ans, batch_shape + (1, 1))
        np_r_norm = np.tile(np_r_norm, batch_shape)
      if use_placeholder:
        a_ph = array_ops.placeholder(dtypes.as_dtype(dtype))
        b_ph = array_ops.placeholder(dtypes.as_dtype(dtype))
        feed_dict = {a_ph: a, b_ph: b}
        tf_ans = linalg_ops.matrix_solve_ls(
            a_ph, b_ph, fast=fast, l2_regularizer=l2_regularizer)
      else:
        tf_ans = linalg_ops.matrix_solve_ls(
            a, b, fast=fast, l2_regularizer=l2_regularizer)
        feed_dict = None
        self.assertEqual(np_ans.shape, tf_ans.get_shape())
      if feed_dict:
        with self.session() as sess:
          tf_ans_val = sess.run(tf_ans, feed_dict=feed_dict)
      else:
        tf_ans_val = self.evaluate(tf_ans)
      self.assertEqual(np_ans.shape, tf_ans_val.shape)
      self.assertAllClose(np_ans, tf_ans_val, atol=2 * tol, rtol=2 * tol)

      if l2_regularizer == 0:
        # The least squares solution should satisfy A^H * (b - A*x) = 0.
        tf_r = b - math_ops.matmul(a, tf_ans)
        tf_r = math_ops.matmul(a, tf_r, adjoint_a=True)
        tf_r_norm = linalg_ops.norm(tf_r, ord="fro", axis=[-2, -1])
        if feed_dict:
          with self.session() as sess:
            tf_ans_val, tf_r_norm_val = sess.run([tf_ans, tf_r_norm],
                                                 feed_dict=feed_dict)
        else:
          tf_ans_val, tf_r_norm_val = self.evaluate([tf_ans, tf_r_norm])
        self.assertAllClose(np_r_norm, tf_r_norm_val, atol=tol, rtol=tol)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # Test: a 2x2 matrix with a right-hand side having a different number of rows (1) makes the
    # solve raise a value error or an invalid-argument error. Graph and eager modes, GPU allowed.
    # The matrix and right-hand sides should have the same number of rows.
    with self.session():
      matrix = constant_op.constant([[1., 0.], [0., 1.]])
      rhs = constant_op.constant([[1., 0.]])
      with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
        linalg_ops.matrix_solve_ls(matrix, rhs)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: for both fast and slow paths, solves with empty matrices or right-hand sides (zero
    # columns or zero rows) and asserts the result shapes are 0x0, 0x2, 2x0 and 2x2 for the four
    # combinations.
    full = np.array([[1., 2.], [3., 4.], [5., 6.]])
    empty0 = np.empty([3, 0])
    empty1 = np.empty([0, 2])
    for fast in [True, False]:
      tf_ans = self.evaluate(
          linalg_ops.matrix_solve_ls(empty0, empty0, fast=fast))
      self.assertEqual(tf_ans.shape, (0, 0))
      tf_ans = self.evaluate(
          linalg_ops.matrix_solve_ls(empty0, full, fast=fast))
      self.assertEqual(tf_ans.shape, (0, 2))
      tf_ans = self.evaluate(
          linalg_ops.matrix_solve_ls(full, empty0, fast=fast))
      self.assertEqual(tf_ans.shape, (2, 0))
      tf_ans = self.evaluate(
          linalg_ops.matrix_solve_ls(empty1, empty1, fast=fast))
      self.assertEqual(tf_ans.shape, (2, 2))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testBatchResultSize(self):
    # [FUNCTION DOC]
    # Test: with a batch of three 3x3 identity matrices and 3x1 right-hand sides, both the plain
    # solve and the least-squares solve report a static result shape of 3x3x1.
    # 3x3x3 matrices, 3x3x1 right-hand sides.
    matrix = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.] * 3).reshape(3, 3, 3)  # pylint: disable=too-many-function-args
    rhs = np.array([1., 2., 3.] * 3).reshape(3, 3, 1)  # pylint: disable=too-many-function-args
    answer = linalg_ops.matrix_solve(matrix, rhs)
    ls_answer = linalg_ops.matrix_solve_ls(matrix, rhs)
    self.assertEqual(ls_answer.get_shape(), [3, 3, 1])
    self.assertEqual(answer.get_shape(), [3, 3, 1])


def _GetSmallMatrixSolveLsOpTests(dtype, use_placeholder, fast, l2_regularizer):
  # [FUNCTION DOC]
  # HELPER: factory of small test methods. Inputs: dtype, use_placeholder, fast and
  # l2_regularizer, captured by the generated tests and forwarded to the shared verification.
  # Returns a tuple of three functions (square, overdetermined, underdetermined), each taking the
  # test instance, to be attached to the test class under generated names.

  def Square(self):
    # [FUNCTION DOC]
    # HELPER: generated test body for a square 2x2 system with a 2x3 right-hand side. Verifies the
    # solve without batching and with batch shape (2, 3) using the captured settings.
    # 2x2 matrices, 2x3 right-hand sides.
    matrix = np.array([[1., 2.], [3., 4.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    for batch_shape in (), (2, 3):
      self._verifySolve(
          matrix,
          rhs,
          dtype,
          use_placeholder,
          fast,
          l2_regularizer,
          batch_shape=batch_shape)

  def Overdetermined(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment copied from the square case says 2x2 matrices, but the
    # matrix used is 3x2 (more equations than unknowns) with a 3x3 right-hand side.
    # HELPER: generated test body for an overdetermined system. Verifies the solve without
    # batching and with batch shape (2, 3) using the captured settings.
    # 2x2 matrices, 2x3 right-hand sides.
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    for batch_shape in (), (2, 3):
      self._verifySolve(
          matrix,
          rhs,
          dtype,
          use_placeholder,
          fast,
          l2_regularizer,
          batch_shape=batch_shape)

  def Underdetermined(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment copied from the square case says 2x2 matrices, but the
    # matrix used is 2x3 (fewer equations than unknowns) with a 2x3 right-hand side.
    # HELPER: generated test body for an underdetermined system. Verifies the solve without
    # batching and with batch shape (2, 3) using the captured settings.
    # 2x2 matrices, 2x3 right-hand sides.
    matrix = np.array([[1., 2., 3], [4., 5., 6.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    for batch_shape in (), (2, 3):
      self._verifySolve(
          matrix,
          rhs,
          dtype,
          use_placeholder,
          fast,
          l2_regularizer,
          batch_shape=batch_shape)

  return (Square, Overdetermined, Underdetermined)


def _GetLargeMatrixSolveLsOpTests(dtype, use_placeholder, fast, l2_regularizer):
  # [FUNCTION DOC]
  # HELPER: factory of large batched test methods. Inputs: dtype, use_placeholder, fast and
  # l2_regularizer, captured by the generated tests and forwarded to the shared verification.
  # Returns a tuple of three functions (large-batch square, overdetermined, underdetermined),
  # each taking the test instance.

  def LargeBatchSquare(self):
    # [FUNCTION DOC]
    # HELPER: generated test body. Seeds the global numpy random state with 1, builds a random
    # 127x127 float32 matrix and a ones right-hand side with one column, and verifies the solve
    # tiled over batch shape (16, 8).
    np.random.seed(1)
    num_rhs = 1
    matrix_shape = (127, 127)
    matrix = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(matrix_shape)).reshape(matrix_shape).astype(np.float32)
    rhs = np.ones([matrix_shape[0], num_rhs]).astype(np.float32)
    self._verifySolve(
        matrix,
        rhs,
        dtype,
        use_placeholder,
        fast,
        l2_regularizer,
        batch_shape=(16, 8))

  def LargeBatchOverdetermined(self):
    # [FUNCTION DOC]
    # HELPER: generated test body. Same as the large square case but with a random 127x64 matrix
    # (overdetermined); seeds the global numpy random state with 1; verifies over batch (16, 8).
    np.random.seed(1)
    num_rhs = 1
    matrix_shape = (127, 64)
    matrix = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(matrix_shape)).reshape(matrix_shape).astype(np.float32)
    rhs = np.ones([matrix_shape[0], num_rhs]).astype(np.float32)
    self._verifySolve(
        matrix,
        rhs,
        dtype,
        use_placeholder,
        fast,
        l2_regularizer,
        batch_shape=(16, 8))

  def LargeBatchUnderdetermined(self):
    # [FUNCTION DOC]
    # HELPER: generated test body. Same as the large square case but with a random 64x127 matrix
    # (underdetermined); seeds the global numpy random state with 1; verifies over batch (16, 8).
    np.random.seed(1)
    num_rhs = 1
    matrix_shape = (64, 127)
    matrix = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(matrix_shape)).reshape(matrix_shape).astype(np.float32)
    rhs = np.ones([matrix_shape[0], num_rhs]).astype(np.float32)
    self._verifySolve(
        matrix,
        rhs,
        dtype,
        use_placeholder,
        fast,
        l2_regularizer,
        batch_shape=(16, 8))

  return (LargeBatchSquare, LargeBatchOverdetermined, LargeBatchUnderdetermined)


class MatrixSolveLsBenchmark(test_lib.Benchmark):
  # [CLASS DOC]
  # Benchmark for the least-squares solve op. The class attribute matrix_shapes lists square,
  # tall and wide shapes (some with a leading batch dimension) to time.

  matrix_shapes = [
      (4, 4),
      (8, 4),
      (4, 8),
      (10, 10),
      (10, 8),
      (8, 10),
      (16, 16),
      (16, 10),
      (10, 16),
      (101, 101),
      (101, 31),
      (31, 101),
      (256, 256),
      (256, 200),
      (200, 256),
      (1001, 1001),
      (1001, 501),
      (501, 1001),
      (1024, 1024),
      (1024, 128),
      (128, 1024),
      (2048, 2048),
      (2048, 64),
      (64, 2048),
      (513, 4, 4),
      (513, 4, 2),
      (513, 2, 4),
      (513, 16, 16),
      (513, 16, 10),
      (513, 10, 16),
      (513, 256, 256),
      (513, 256, 128),
      (513, 128, 256),
  ]

  def benchmarkMatrixSolveLsOp(self):
    # [FUNCTION DOC]
    # Benchmark: for every shape and for right-hand-side counts of 1, 2 and the number of
    # columns, builds the data, runs the solve with L2 regularization 1.0 in a fresh graph and
    # session on CPU, and times it for at least 25 iterations (memory usage not stored). Repeats
    # on the GPU if available, but only for unbatched shapes or batch size below 513 (so the large
    # batched shapes are CPU only). Side effects: creates graphs and sessions, emits reports.
    run_gpu_test = test_lib.is_gpu_available(True)
    regularizer = 1.0
    for matrix_shape in self.matrix_shapes:
      for num_rhs in 1, 2, matrix_shape[-1]:

        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/cpu:0"):
          matrix, rhs = _GenerateTestData(matrix_shape, num_rhs)
          x = linalg_ops.matrix_solve_ls(matrix, rhs, regularizer)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(x),
              min_iters=25,
              store_memory_usage=False,
              name=("matrix_solve_ls_cpu_shape_{matrix_shape}_num_rhs_{num_rhs}"
                   ).format(matrix_shape=matrix_shape, num_rhs=num_rhs))

        if run_gpu_test and (len(matrix_shape) < 3 or matrix_shape[0] < 513):
          with ops.Graph().as_default(), \
                session.Session(config=benchmark.benchmark_config()) as sess, \
                ops.device("/gpu:0"):
            matrix, rhs = _GenerateTestData(matrix_shape, num_rhs)
            x = linalg_ops.matrix_solve_ls(matrix, rhs, regularizer)
            self.evaluate(variables.global_variables_initializer())
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(x),
                min_iters=25,
                store_memory_usage=False,
                name=("matrix_solve_ls_gpu_shape_{matrix_shape}_num_rhs_"
                      "{num_rhs}").format(
                          matrix_shape=matrix_shape, num_rhs=num_rhs))


if __name__ == "__main__":
  dtypes_to_test = [np.float32, np.float64, np.complex64, np.complex128]
  for dtype_ in dtypes_to_test:
    for use_placeholder_ in set([False, True]):
      for fast_ in [True, False]:
        l2_regularizers = [0] if dtype_ == np.complex128 else [0, 0.1]
        for l2_regularizer_ in l2_regularizers:
          for test_case in _GetSmallMatrixSolveLsOpTests(
              dtype_, use_placeholder_, fast_, l2_regularizer_):
            name = "%s_%s_placeholder_%s_fast_%s_regu_%s" % (test_case.__name__,
                                                             dtype_.__name__,
                                                             use_placeholder_,
                                                             fast_,
                                                             l2_regularizer_)
            _AddTest(MatrixSolveLsOpTest, "MatrixSolveLsOpTest", name,
                     test_case)
  for dtype_ in dtypes_to_test:
    for test_case in _GetLargeMatrixSolveLsOpTests(dtype_, False, True, 0.0):
      name = "%s_%s" % (test_case.__name__, dtype_.__name__)
      _AddTest(MatrixSolveLsOpTest, "MatrixSolveLsOpTest", name, test_case)

  test_lib.main()
