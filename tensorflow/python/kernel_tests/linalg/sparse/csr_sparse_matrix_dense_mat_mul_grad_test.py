# [MODULE DOC]
# Gradient tests for the matrix-multiplication of a CSR sparse matrix with a dense matrix. A test
# class holds a private gradient-checking method; at import time the module generates one test
# method per combination of dtype (single precision real or complex), transpose/adjoint flags for
# both operands, output transpose/conjugate flags and batched or unbatched inputs, and attaches them
# to the class. These were split out from the CSR sparse matrix gradient test to keep that file
# medium sized. Importing the module registers the sparse matrix gradient functions (unused-import
# side effect) and mutates the test class. Standalone test module, not a helper.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""CSR sparse matrix tests."""

import itertools

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_grad  # pylint: disable=unused-import
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _add_test(test, op_name, testcase_name, fn):  # pylint: disable=redefined-outer-name
  # [FUNCTION DOC]
  # HELPER: attaches a generated test method to a test class.
  # Inputs: test (class to modify), op_name and testcase_name (strings joined with a test prefix into
  # the method name), fn (function or the null value). If fn is null nothing happens. Raises a runtime
  # error if a method of the same name already exists. Otherwise sets fn as an attribute of the class
  # (side effect). Returns nothing.
  if fn is None:
    return
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class CSRSparseMatrixDenseMatMulGradTest(test.TestCase):
  # [CLASS DOC]
  # Test case that verifies gradients of sparse (CSR) matrix times dense matrix multiplication.
  # Defines a class-level setup recording GPU availability and one private gradient-check method that
  # is invoked by the generated test methods added at import time. Derives from the TensorFlow test
  # case.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the recorded GPU flag is never used by the tests in this file.
    # Class-level fixture. Calls the parent class setup and records whether a GPU is available in a
    # class attribute (_gpu_available). Takes the class; returns nothing.
    super(CSRSparseMatrixDenseMatMulGradTest, cls).setUpClass()
    cls._gpu_available = test_util.is_gpu_available()

  # TODO(penporn): Make these tests runnable on eager mode.
  # (tf.gradients and gradient_checker only run in graph mode.)
  @test_util.run_deprecated_v1
  def _testLargeBatchSparseMatrixMatMulGrad(
      self,
      datatype,
      transpose_a,
      transpose_b,
      adjoint_a,
      adjoint_b,
      transpose_output,
      conjugate_output,
      batched_inputs,
  ):
    # [FUNCTION DOC]
    # HELPER: gradient checker run by the generated test methods.
    # KNOWN ISSUE: the deprecated graph-mode decorator is applied to this private helper rather than to
    # the generated test methods that call it, and the file's to-do notes the tests cannot run eagerly.
    # Inputs: datatype (numpy dtype, real or complex single precision), transpose_a, transpose_b,
    # adjoint_a, adjoint_b, transpose_output, conjugate_output (booleans passed to the multiplication)
    # and batched_inputs (boolean selecting batch-of-3 shapes versus a single pair of matrices).
    # Behavior: draws random complex matrices of shape five by eleven and eleven by thirteen (with a
    # batch dimension of three when batched), keeps only positive entries of the first to make it
    # sparse, casts to the dtype, then transposes and conjugates the stored values as needed for the
    # flags. Converts both to tensors, converts the first to a CSR sparse matrix at its nonzero
    # locations, multiplies it by the second with all flags, and for each of the two inputs computes
    # theoretical and numerical gradients with the gradient checker (delta 1e-3). Asserts they agree to
    # within 1e-3 absolute and relative tolerance. Logs which input is being checked. Returns nothing.
    if batched_inputs:
      a_shape = (3, 5, 11)
      b_shape = (3, 11, 13)
      transpose = lambda x: np.transpose(x, (0, 2, 1))
    else:
      a_shape = (5, 11)
      b_shape = (11, 13)
      transpose = np.transpose

    sparsify = lambda m: m * (m > 0)
    a_mats_val = sparsify(
        np.random.randn(*a_shape) +
        1.j * np.random.randn(*a_shape)).astype(datatype)
    if transpose_a or adjoint_a:
      a_mats_val = transpose(a_mats_val)
    if adjoint_a:
      a_mats_val = np.conj(a_mats_val)
    b_mats_val = (np.random.randn(*b_shape) +
                  1.j * np.random.randn(*b_shape)).astype(datatype)
    if transpose_b or adjoint_b:
      b_mats_val = transpose(b_mats_val)
    if adjoint_b:
      b_mats_val = np.conj(b_mats_val)
    with self.test_session():
      a_mats = ops.convert_to_tensor(a_mats_val, dtype=datatype)
      b_mats = ops.convert_to_tensor(b_mats_val, dtype=datatype)
      locs = array_ops.where(abs(a_mats_val) > 0)
      a_sm = sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(a_mats, locs)
      c_mats = sparse_csr_matrix_ops.sparse_matrix_mat_mul(
          a_sm,
          b_mats,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b,
          transpose_output=transpose_output,
          conjugate_output=conjugate_output)
      for [ten, val, nn] in [[a_mats, a_mats_val, "a"],
                             [b_mats, b_mats_val, "b"]]:
        tf_logging.info("Testing gradients for %s" % nn)
        theoretical, numerical = gradient_checker.compute_gradient(
            ten,
            ten.get_shape().as_list(),
            c_mats,
            c_mats.get_shape().as_list(),
            x_init_value=val,
            delta=1e-3)
        self.assertAllClose(theoretical, numerical, atol=1e-3, rtol=1e-3)


# These tests are refactored from sparse_csr_matrix_grad_test to keep its size
# "medium".
dtypes_to_test = [np.float32, np.complex64]
for dtype in dtypes_to_test:
  for (t_a, t_b, adj_a, adj_b, t_out,
       conj_out, batched) in itertools.product(*(([False, True],) * 7)):

    def create_mat_mul_test_fn(dtype_, t_a_, t_b_, adj_a_, adj_b_, t_out_,
                               conj_out_, batched_):
      # [FUNCTION DOC]
      # HELPER: factory of the generated test methods (defined in the import-time loop).
      # Inputs: the dtype and the flags (transpose a/b, adjoint a/b, transpose output, conjugate output,
      # batched) for one combination. Returns the null value for invalid combinations: transposing and
      # taking the adjoint of the same operand, or conjugation flags with real single-precision data.
      # Otherwise returns a test method that calls the gradient-check method with those values.
      # Skip invalid cases.
      if (t_a_ and adj_a_) or (t_b_ and adj_b_):
        return
      # Skip cases where we conjugate real matrices.
      if dtype_ == np.float32 and (adj_a_ or adj_b_ or conj_out_):
        return

      def test_fn(self):
        # [FUNCTION DOC]
        # Generated test method (closure). Takes the test instance and runs the gradient-check method with
        # the dtype and flags captured from the factory. Returns nothing.
        self._testLargeBatchSparseMatrixMatMulGrad(dtype_, t_a_, t_b_, adj_a_,
                                                   adj_b_, t_out_, conj_out_,
                                                   batched_)

      return test_fn

    name = (
        "_testLargeBatchSparseMatrixMatMulGrad_dtype_%s_t_a_%s_t_b_%s_adj_a_%s_"
        "adj_b_%s_t_out_%s_conj_out_%s_batched_%s" %
        (dtype.__name__, t_a, t_b, adj_a, adj_b, t_out, conj_out, batched))

    _add_test(
        CSRSparseMatrixDenseMatMulGradTest, "CSRSparseMatrixGradTest", name,
        create_mat_mul_test_fn(dtype, t_a, t_b, adj_a, adj_b, t_out, conj_out,
                               batched))

if __name__ == "__main__":
  test.main()
