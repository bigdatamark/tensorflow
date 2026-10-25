# [MODULE DOC]
# Gradient tests for the sparse-times-dense matrix multiplication op. For random shapes and
# several value types (half, single and double precision floats, and two complex types) and index
# types (32-bit and 64-bit), with every combination of transposing-and-conjugating either operand,
# it compares analytic and numeric gradients with respect to the dense operand and the values of
# the sparse operand. Importing the module registers the sparse gradient functions; running it as
# a script starts the TensorFlow test runner.
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
"""Tests for the gradient of `tf.sparse.sparse_dense_matmul()`."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SparseTensorDenseMatMulGradientTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking the gradient of the sparse-dense matmul op. It holds no state; helpers
  # create random dense and sparse operands, and the tests run gradient error checks over all
  # type and adjoint combinations.

  def _sparsify(self, x, indices_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: turns a dense numpy matrix into a sparse tensor by dropping small entries.
    # Inputs: x, a two-dimensional numpy array; indices_dtype, numpy integer type for the indices
    # (default 64-bit). Side effect: x is modified in place, every entry below one half is set to
    # zero (for complex arrays this relies on numpy's ordering comparison of complex numbers).
    # Returns a pair: the sparse tensor holding the remaining non-zero entries with the shape of
    # x, and the count of stored entries.
    x[x < 0.5] = 0

    non_zero = np.where(x)
    x_indices = np.vstack(non_zero).astype(indices_dtype).T
    x_values = x[non_zero]
    x_shape = x.shape

    return sparse_tensor.SparseTensor(
        indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)

  def _randomTensor(self,
                    size,
                    values_dtype,
                    adjoint=False,
                    sparse=False,
                    indices_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: creates a random matrix operand, dense or sparse, for the gradient tests.
    # Inputs: size, pair of rows and columns; values_dtype, numpy element type; adjoint, when true
    # the generated matrix is transposed and conjugated before use (so its shape is columns by
    # rows); sparse, when true the result is sparsified; indices_dtype, index type for the sparse
    # case. Values are standard normal (complex types also get a random imaginary part).
    # Returns a pair of sparse tensor and stored-entry count when sparse is true, otherwise a
    # constant dense tensor. Uses (and advances) numpy's global random state.
    n, m = size
    x = np.random.randn(n, m).astype(values_dtype)
    if values_dtype in (np.complex64, np.complex128):
      x.imag = np.random.randn(n, m)

    if adjoint:
      x = x.transpose().conj()

    if sparse:
      return self._sparsify(x, indices_dtype=indices_dtype)
    else:
      return constant_op.constant(x, dtype=values_dtype)

  def _testGradients(self, adjoint_a, adjoint_b, name, values_dtype,
                     indices_dtype):
    # [FUNCTION DOC]
    # HELPER: runs one gradient check of the sparse-dense matmul for a given configuration.
    # Inputs: adjoint_a and adjoint_b, booleans choosing whether each operand is used adjointed;
    # name, the op name (also printed); values_dtype and indices_dtype, numpy types.
    # Picks random dimensions between 1 and 9, builds a sparse and a dense operand, applies the
    # matmul, then compares analytic and numeric gradients with respect to the dense operand and
    # the sparse values. Half precision uses a larger step and a looser tolerance. Prints the
    # error and asserts it is below the tolerance. Uses numpy global randomness.
    n, k, m = np.random.randint(1, 10, size=3)
    sp_t, nnz = self._randomTensor(
        [n, k],
        values_dtype,
        adjoint=adjoint_a,
        sparse=True,
        indices_dtype=indices_dtype)
    dense_t = self._randomTensor([k, m], values_dtype, adjoint=adjoint_b)

    matmul = sparse_ops.sparse_tensor_dense_matmul(
        sp_t, dense_t, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name)

    with self.cached_session():
      dense_t_shape = [m, k] if adjoint_b else [k, m]
      sp_t_val_shape = [nnz]
      delta = 1 / 16. if values_dtype == np.float16 else 1e-3
      tolerance = delta / 2. if values_dtype == np.float16 else 1e-3
      err = gradient_checker.compute_gradient_error(
          [dense_t, sp_t.values], [dense_t_shape, sp_t_val_shape],
          matmul, [n, m],
          delta=delta)
      print("%s gradient err = %s" % (name, err))
      self.assertLess(err, tolerance)

  def _testGradientsType(self, values_dtype, indices_dtype):
    # [FUNCTION DOC]
    # HELPER: runs the gradient check for one value type and one index type over all four
    # combinations of the two adjoint flags, generating a descriptive op name for each.
    # Inputs: values_dtype and indices_dtype, numpy type classes. Returns nothing; failures are
    # test assertion failures.
    for adjoint_a in [True, False]:
      for adjoint_b in [True, False]:
        name = "sparse_tensor_dense_matmul_%s_%s_%s_%s" % (
            adjoint_a, adjoint_b, values_dtype.__name__, indices_dtype.__name__)
        self._testGradients(adjoint_a, adjoint_b, name, values_dtype,
                            indices_dtype)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Scenario: seed numpy's random generator with a fixed value (to avoid flakiness) and run the
    # type-level gradient check for half, single, double, complex64 and complex128 values with
    # 64-bit indices, plus single and complex64 values with 32-bit indices. Graph mode only.
    # Expected: all gradient errors are below the per-type tolerance.
    np.random.seed(5)  # Fix seed to avoid flakiness
    self._testGradientsType(np.float16, np.int64)
    self._testGradientsType(np.float32, np.int64)
    self._testGradientsType(np.float64, np.int64)
    self._testGradientsType(np.complex64, np.int64)
    self._testGradientsType(np.complex128, np.int64)
    self._testGradientsType(np.float32, np.int32)
    self._testGradientsType(np.complex64, np.int32)


if __name__ == "__main__":
  test.main()
