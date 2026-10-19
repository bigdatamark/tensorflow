# [MODULE DOC]
# KNOWN ISSUE: the module docstring says it tests matrix_inverse, but the file tests the QR
# decomposition op (linalg_ops.qr), its gradient, and a benchmark.
# Unit tests and benchmark for the QR decomposition op. Contains a test-registration helper, a
# class with static tests (rank validation and concurrent execution), a factory generating
# value tests across dtypes, shapes, full-or-reduced matrices and static or dynamic shapes, a
# gradient test class and generator comparing analytic to numerical gradients, and a benchmark
# class timing the op on CPU and, if available, GPU. The script entry point generates and
# registers the dynamic tests and starts the test runner.
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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def _AddTest(test_class, op_name, testcase_name, fn):
  # [FUNCTION DOC]
  # HELPER: registers a dynamically generated test method on a test class. Inputs: the test class,
  # an operation name, a test case name and the test function. The attribute name joins the word
  # test, the operation name and the case name. Raises a runtime error if a method with that name
  # already exists; otherwise sets the attribute on the class (mutating it) and returns nothing.
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)


class QrOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the QR op. Defines static tests for invalid ranks and concurrent execution;
  # value-comparison tests are attached dynamically by the script entry point.

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # Scenario: QR is applied to a scalar and to a rank-one vector. Expects a value error or an
    # invalid-argument error whose message mentions the required rank of at least two and the
    # actual rank (zero, then one). Runs in graph and eager modes, GPU allowed.
    # The input to qr should be a tensor of at least rank 2.
    scalar = constant_op.constant(1.)
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "rank.* 2.*0"):
      linalg_ops.qr(scalar)
    vector = constant_op.constant([1., 2.])
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "rank.* 2.*1"):
      linalg_ops.qr(vector)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Scenario: for full and reduced modes and several matrix shapes (4 or 5 rows and columns),
    # two identical matrices are drawn from the same fixed seed (asserted equal) and each is
    # decomposed by QR. All factors are evaluated together; expects each pair of results from the
    # two identical decompositions to be close, showing concurrent kernels do not interfere.
    # Runs in graph and eager modes, GPU allowed.
    seed = [42, 24]
    all_ops = []
    for full_matrices_ in True, False:
      for rows_ in 4, 5:
        for cols_ in 4, 5:
          matrix_shape = [rows_, cols_]
          matrix1 = stateless_random_ops.stateless_random_normal(
              matrix_shape, seed)
          matrix2 = stateless_random_ops.stateless_random_normal(
              matrix_shape, seed)
          self.assertAllEqual(matrix1, matrix2)
          q1, r1 = linalg_ops.qr(matrix1, full_matrices=full_matrices_)
          q2, r2 = linalg_ops.qr(matrix2, full_matrices=full_matrices_)
          all_ops += [q1, q2, r1, r2]
    val = self.evaluate(all_ops)
    for i in range(0, len(val), 2):
      self.assertAllClose(val[i], val[i + 1])


def _GetQrOpTest(dtype_, shape_, full_matrices_, use_static_shape_):
  # [FUNCTION DOC]
  # HELPER: factory producing one value test for the QR op. Inputs: numpy dtype, full shape
  # (batch dimensions then rows and columns), whether to compute full matrices, and whether to
  # use a constant (static shape) or a placeholder. Defines tolerance-aware nested checks
  # (single precision uses looser tolerances) and returns the test function that closes over them.

  is_complex = dtype_ in (np.complex64, np.complex128)
  is_single = dtype_ in (np.float32, np.complex64)

  def CompareOrthogonal(self, x, y, rank):
    # [FUNCTION DOC]
    # HELPER: nested check that two orthogonal factors agree up to sign or complex phase. Inputs:
    # test case, two arrays x and y of orthogonal factors, and rank (number of leading columns
    # to compare, since the remaining columns are an arbitrary basis). Normalizes the phase of x
    # by the ratio of summed column ratios, then asserts close with a tolerance depending on
    # precision. Note it modifies x in place via the phase multiplication of the sliced view.
    if is_single:
      atol = 5e-4
    else:
      atol = 5e-14
    # We only compare the first 'rank' orthogonal vectors since the
    # remainder form an arbitrary orthonormal basis for the
    # (row- or column-) null space, whose exact value depends on
    # implementation details. Notice that since we check that the
    # matrices of singular vectors are unitary elsewhere, we do
    # implicitly test that the trailing vectors of x and y span the
    # same space.
    x = x[..., 0:rank]
    y = y[..., 0:rank]
    # Q is only unique up to sign (complex phase factor for complex matrices),
    # so we normalize the sign first.
    sum_of_ratios = np.sum(np.divide(y, x), -2, keepdims=True)
    phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
    x *= phases
    self.assertAllClose(x, y, atol=atol)

  def CheckApproximation(self, a, q, r):
    # [FUNCTION DOC]
    # HELPER: nested check that the factors reconstruct the input. Multiplies q and r (TF32
    # disabled) and asserts the product is close to the original matrix a, with tolerance 1e-5
    # for single precision and 1e-14 otherwise. Returns nothing.
    if is_single:
      tol = 1e-5
    else:
      tol = 1e-14
    # Tests that a ~= q*r.
    a_recon = test_util.matmul_without_tf32(q, r)
    self.assertAllClose(a_recon, a, rtol=tol, atol=tol)

  def CheckUnitary(self, x):
    # [FUNCTION DOC]
    # HELPER: nested check that the columns of x are orthonormal. Multiplies the conjugate
    # transpose of x by x (TF32 disabled) and asserts the result is close to the identity,
    # with tolerance 1e-5 for single precision and 1e-14 otherwise. Returns nothing.
    # Tests that x[...,:,:]^H * x[...,:,:] is close to the identity.
    xx = test_util.matmul_without_tf32(x, x, adjoint_a=True)
    identity = array_ops.matrix_band_part(array_ops.ones_like(xx), 0, 0)
    if is_single:
      tol = 1e-5
    else:
      tol = 1e-14
    self.assertAllClose(identity, xx, atol=tol)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def Test(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the whole verification body is indented under the complex-dtype condition, so
    # for the real dtypes (float32, float64) the generated test builds the random input but
    # performs no QR computation and asserts nothing; only complex dtypes are really tested.
    # Generated test body. Returns early when a placeholder is requested under eager execution.
    # Seeds numpy with a fixed value, builds a uniform random matrix of the captured shape and
    # dtype (adding a random imaginary part for complex), and for complex inputs runs QR (constant
    # or placeholder fed in a session), computes a reference orthogonal factor per matrix of the
    # flattened batch using numpy in complete or reduced mode, then compares orthogonal factors
    # (leading columns up to the smaller matrix dimension), checks the reconstruction and checks
    # the factor is unitary.
    if not use_static_shape_ and context.executing_eagerly():
      return
    np.random.seed(1)
    x_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    if is_complex:
      x_np += 1j * np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)

      if use_static_shape_:
        x_tf = constant_op.constant(x_np)
      else:
        x_tf = array_ops.placeholder(dtype_)
      q_tf, r_tf = linalg_ops.qr(x_tf, full_matrices=full_matrices_)

      if use_static_shape_:
        q_tf_val, r_tf_val = self.evaluate([q_tf, r_tf])
      else:
        with self.session() as sess:
          q_tf_val, r_tf_val = sess.run([q_tf, r_tf], feed_dict={x_tf: x_np})

      q_dims = q_tf_val.shape
      np_q = np.ndarray(q_dims, dtype_)
      np_q_reshape = np.reshape(np_q, (-1, q_dims[-2], q_dims[-1]))
      new_first_dim = np_q_reshape.shape[0]

      x_reshape = np.reshape(x_np, (-1, x_np.shape[-2], x_np.shape[-1]))
      for i in range(new_first_dim):
        if full_matrices_:
          np_q_reshape[i, :, :], _ = np.linalg.qr(
              x_reshape[i, :, :], mode="complete")
        else:
          np_q_reshape[i, :, :], _ = np.linalg.qr(
              x_reshape[i, :, :], mode="reduced")
      np_q = np.reshape(np_q_reshape, q_dims)
      CompareOrthogonal(self, np_q, q_tf_val, min(shape_[-2:]))
      CheckApproximation(self, x_np, q_tf_val, r_tf_val)
      CheckUnitary(self, q_tf_val)

  return Test


class QrGradOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the QR gradient. Holds one static test for the unsupported configuration; the
  # numerical gradient tests are attached dynamically by the script entry point.

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNotImplementedCheck(self):
    # [FUNCTION DOC]
    # Scenario: the gradient of QR with full matrices is requested for a 5x2 random float matrix
    # (taller than wide). Expects a not-implemented error with a message stating the gradient is
    # not implemented when rows exceed columns and full matrices is true. Runs in graph and eager
    # modes, GPU allowed.
    # Test that the correct message is issued
    np.random.seed(42)
    matrix = constant_op.constant(
        np.random.uniform(low=-1.0, high=1.0, size=(5, 2)).astype(np.float32))

    def _NoGrad(x):
      # [FUNCTION DOC]
      # HELPER: nested function that watches input x on a gradient tape, applies QR with full
      # matrices, and returns the gradient of the outputs with respect to x; raises the
      # not-implemented error expected by the enclosing test.
      with backprop.GradientTape() as tape:
        tape.watch(x)
        ret = linalg_ops.qr(x, full_matrices=True)
      return tape.gradient(ret, x)

    m = r"QrGrad not implemented when nrows > ncols and full_matrices is true."
    with self.assertRaisesRegex(NotImplementedError, m):
      _NoGrad(matrix)


def _GetQrGradOpTest(dtype_, shape_, full_matrices_):
  # [FUNCTION DOC]
  # HELPER: factory producing one gradient test for the QR op. Inputs: numpy dtype, full shape,
  # and whether to compute full matrices. Returns the test function closing over them.

  def RandomInput():
    # [FUNCTION DOC]
    # HELPER: nested generator of a random input matrix of the captured shape and dtype with
    # uniform values in minus one to one, plus a random imaginary part for complex dtypes.
    # Unseeded here (the caller seeds numpy). Returns the numpy array.
    a = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
    if dtype_ in [np.complex64, np.complex128]:
      a += 1j * np.random.uniform(
          low=-1.0, high=1.0, size=shape_).astype(dtype_)
    return a

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32("Tests Qr gradient, which calls matmul"
                                        )
  def Test(self):
    # [FUNCTION DOC]
    # Generated gradient test body. Seeds numpy, picks a finite-difference step from the dtype's
    # machine epsilon and a tolerance (loose for single precision, tight for double), defines one
    # function for each of the two QR outputs (the orthogonal and the triangular factor), and for
    # each compares the theoretical and numerical gradients computed on a random input.
    # TF32 is disabled; runs in graph and eager modes, GPU allowed.
    np.random.seed(42)
    # Optimal stepsize for central difference is O(epsilon^{1/3}).
    epsilon = np.finfo(dtype_).eps
    delta = 0.1 * epsilon**(1.0 / 3.0)
    if dtype_ in [np.float32, np.complex64]:
      tol = 3e-2
    else:
      tol = 1e-6
    # TODO(b/157171666): Sadly we have to double the computation because
    # gradient_checker_v2.compute_gradient expects a list of functions.
    funcs = [
        lambda a: linalg_ops.qr(a, full_matrices=full_matrices_)[0],
        lambda a: linalg_ops.qr(a, full_matrices=full_matrices_)[1]
    ]
    for f in funcs:
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [RandomInput()], delta=delta)
      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


class QRBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class timing the QR op on a fixed list of matrix shapes (square, tall and batched),
  # stored in the class attribute named shapes.

  shapes = [
      (4, 4),
      (8, 8),
      (16, 16),
      (101, 101),
      (256, 256),
      (1024, 1024),
      (2048, 2048),
      (1024, 2),
      (1024, 32),
      (1024, 128),
      (1024, 512),
      (1, 8, 8),
      (10, 8, 8),
      (100, 8, 8),
      (1, 256, 256),
      (10, 256, 256),
      (100, 256, 256),
  ]

  def benchmarkQROp(self):
    # [FUNCTION DOC]
    # For each listed shape, builds a graph holding a variable initialized with random float32
    # values and its QR decomposition, pinned to the CPU, initializes variables and reports a
    # benchmark run (at least 25 iterations) named by device and shape. If a GPU is available,
    # repeats the same on the first GPU. Side effects: creates graphs and sessions and emits
    # benchmark results. Returns nothing.
    for shape_ in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix_value = np.random.uniform(
            low=-1.0, high=1.0, size=shape_).astype(np.float32)
        matrix = variables.Variable(matrix_value)
        q, r = linalg_ops.qr(matrix)
        self.evaluate(variables.global_variables_initializer())
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(q, r),
            min_iters=25,
            name="QR_cpu_{shape}".format(shape=shape_))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/device:GPU:0"):
          matrix_value = np.random.uniform(
              low=-1.0, high=1.0, size=shape_).astype(np.float32)
          matrix = variables.Variable(matrix_value)
          q, r = linalg_ops.qr(matrix)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(q, r),
              min_iters=25,
              name="QR_gpu_{shape}".format(shape=shape_))


if __name__ == "__main__":
  for dtype in np.float32, np.float64, np.complex64, np.complex128:
    for rows in 1, 2, 5, 10, 32, 100:
      for cols in 1, 2, 5, 10, 32, 100:
        for full_matrices in False, True:
          for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
            # TF2 does not support placeholders under eager so we skip it
            for use_static_shape in [True, False]:
              shape = batch_dims + (rows, cols)
              name = "%s_%s_full_%s_static_%s" % (dtype.__name__,
                                                  "_".join(map(str, shape)),
                                                  full_matrices,
                                                  use_static_shape)
              _AddTest(QrOpTest, "Qr", name,
                       _GetQrOpTest(dtype, shape, full_matrices,
                                    use_static_shape))

  # TODO(pfau): Get working with full_matrices when rows > cols
  # TODO(pfau): Get working with shapeholders (dynamic shapes)
  for full_matrices in False, True:
    for dtype in np.float32, np.float64, np.complex64, np.complex128:
      for rows in 1, 2, 5, 10:
        for cols in 1, 2, 5, 10:
          if rows <= cols or (not full_matrices and rows > cols):
            for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
              shape = batch_dims + (rows, cols)
              name = "%s_%s_full_%s" % (dtype.__name__,
                                        "_".join(map(str, shape)),
                                        full_matrices)
              _AddTest(QrGradOpTest, "QrGrad", name,
                       _GetQrGradOpTest(dtype, shape, full_matrices))
  test.main()
