# [MODULE DOC]
# Tests for deterministic-mode behavior of the sparse-times-dense matrix multiplication op. One
# test class checks that, when op determinism is enabled, the GPU path raises an unimplemented
# error; another checks that repeated CPU evaluations give identical results. Contains a
# module-level data generator helper. When run as a script, it enables op determinism globally
# before starting the TensorFlow test runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ========================================================================
"""Functional tests for deterministic SparseTensorDenseMatMul."""

import random

import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def _gen_data(m, k, n, nnz, row_occupied_rate, data_type, seed):
  # [FUNCTION DOC]
  # HELPER: generates reproducible random operands for the deterministic matmul tests.
  # KNOWN ISSUE: the docstring lists only some parameters (data_type and seed are undocumented),
  # has a typo ("col could"), and calls row_occupied_rate a probability although it is used as the
  # fraction of rows that are sampled to hold entries. It also says the matmul "returns" but this
  # function returns the two operands.
  # Inputs: m, k, n, the dimensions (sparse operand m by k, dense operand k by n); nnz, number of
  # coordinates generated; row_occupied_rate, fraction of the m rows allowed to contain entries;
  # data_type, numpy element type; seed, used to seed both the python and numpy random generators.
  # Coordinates are drawn at random from the occupied rows and any column, so duplicates are
  # possible. Values are standard normal; complex types get a matching imaginary part. Returns a
  # pair: the sparse tensor and the dense constant tensor. Side effect: reseeds global generators.
  """Generate valid input data for tf.sparse.sparse_dense_matmul

  returns sparse matrix a (type SparseTensor), dense matrix b (type Tensor)

  Parameters:
    m: row count of dense version of matrix a / row count of output matrix
    k: col count of dense version of matrix a / row count of matrix b
    n: col could of matrix b / col count of output matrix
    nnz: number of non-zero elements in matrix a
    row_occupied_rate: prob that row in a has one or more non-zero element
  """
  random.seed(seed)
  np.random.seed(seed)
  occupied_rows = random.sample(range(m), int(m * row_occupied_rate))
  sparse_input_dense_shape = [m, k]
  dense_input_shape = (k, n)
  indices = []
  for _ in range(nnz):
    row = random.choice(occupied_rows)
    col = random.randint(0, k - 1)
    indices.append([row, col])

  def maybe_complex(x):
    # [FUNCTION DOC]
    # HELPER: gives complex arrays a non-zero imaginary part. Input x is a numpy array. If it has
    # a complex type, returns the array scaled to have imaginary part equal to its real part
    # (divided by two); otherwise returns x unchanged. No side effects.
    if x.dtype.kind == "c":  # complex
      return (x + 1j * x) / 2
    return x

  sparse_values = maybe_complex(
      np.random.normal(size=len(indices)).astype(data_type))
  dense_values = maybe_complex(
      np.random.normal(size=dense_input_shape).astype(data_type))
  sparse_input = sparse_tensor.SparseTensor(indices, sparse_values,
                                            sparse_input_dense_shape)
  dense_input = constant_op.constant(dense_values)
  return sparse_input, dense_input


class SparseTensorDenseMatmulOpDeterminismExceptionsTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that the GPU kernel of the sparse-dense matmul op reports an
  # unimplemented-deterministic-implementation error when op determinism is enabled. Holds no
  # state.
  """Test d9m-unimplemented exceptions from SparseTensorDenseMatmulOp.

  Test that tf.errors.UnimplementedError is thrown, as appropriate, by the
  GPU-specific code-paths through SparseTensorDenseMatmulOp when deterministic
  ops are enabled.

  This test assumes that sparse_tensor_dense_matmul_op_test.py runs equivalent
  test cases when deterministic ops are not enabled and will therefore detect
  erroneous exception throwing in those cases.
  """

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testExceptionThrowing(self):
    # [FUNCTION DOC]
    # Scenario: only runs when a GPU exists, in graph and eager modes. For half, single, double,
    # complex64 and complex128 data, generate a 5x10 sparse operand with 20 entries and a 10x7
    # dense operand, force GPU placement, multiply and evaluate.
    # Expected: an unimplemented error stating that a deterministic GPU implementation of the op
    # is not currently available. Relies on determinism having been enabled by the script entry
    # point (when run through other runners the error may not occur).
    with self.session(), test_util.force_gpu():
      for data_type in [
          np.float16, np.float32, np.float64, np.complex64, np.complex128
      ]:
        sparse_input, dense_input = _gen_data(
            m=5,
            k=10,
            n=7,
            nnz=20,
            row_occupied_rate=0.9,
            data_type=data_type,
            seed=456)
        with self.assertRaisesRegex(
            errors.UnimplementedError,
            "A deterministic GPU implementation of SparseTensorDenseMatmulOp" +
            " is not currently available."):
          result = sparse_ops.sparse_tensor_dense_matmul(
              sparse_input, dense_input)
          self.evaluate(result)


class SparseTensorDenseMatmulOpDeterministicTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring has a typo in the op name (SparseTensorDenseMatul).
  # Test case verifying that the sparse-dense matmul op gives bit-identical results on repeated
  # CPU runs. Holds no state.
  """Test that SparseTensorDenseMatul operates reproducibly (on CPU only)."""

  @test_util.run_in_graph_and_eager_modes
  def testForward(self):
    # [FUNCTION DOC]
    # Scenario: for half, single, double, complex64 and complex128 data (integer and bfloat16 are
    # skipped), generate a large sparse operand (2430 by 615 with 65779 entries in about 2 percent
    # of the rows) and a dense operand with 857 columns, force CPU placement, multiply once and
    # then five more times.
    # Expected: each of the five repeated results is exactly equal to the first one.
    for data_type in [
        np.float16, np.float32, np.float64, np.complex64, np.complex128
    ]:  # skipping int32 and bfloat16
      sparse_input, dense_input = _gen_data(
          m=2430,
          k=615,
          n=857,
          nnz=(1 << 16) + 243,
          row_occupied_rate=0.02,
          data_type=data_type,
          seed=123)
      with self.session(), test_util.force_cpu():
        result_a = sparse_ops.sparse_tensor_dense_matmul(
            sparse_input, dense_input)
        for _ in range(5):
          result_b = sparse_ops.sparse_tensor_dense_matmul(
              sparse_input, dense_input)
          self.assertAllEqual(result_a, result_b)


if __name__ == "__main__":
  # TODO(reedwm): Merge this file with sparse_tensor_dense_matmul_test.py
  config.enable_op_determinism()
  test.main()
