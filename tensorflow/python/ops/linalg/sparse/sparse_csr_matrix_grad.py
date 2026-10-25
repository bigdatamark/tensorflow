# [MODULE DOC]
# Gradient registrations for the CSR sparse matrix (variant tensor) ops. Importing this module
# registers gradient functions for the conversion ops (dense, COO sparse tensor <-> CSR), sparse
# matrix add, transpose, softmax, sparse-times-dense matmul and sparse-times-sparse matmul, and
# marks the nnz and zeros ops as not differentiable. Gradients for the CSR matrices are
# themselves CSR matrices pruned to the sparsity pattern of the corresponding forward input.
# Also contains private pruning helpers used by those gradients. It relies on the forward op
# wrappers in sparse_csr_matrix_ops. Import-time side effect: global gradient registry updates.
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
"""CSR Sparse Matrix Gradients."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops


@ops.RegisterGradient("DenseToCSRSparseMatrix")
def _DenseToCSRSparseMatrixGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the dense-to-CSR conversion. Takes the forward op and the incoming gradient (a
  # CSR sparse matrix), converts the gradient back to a dense tensor using the forward op's
  # element type attribute, and returns a pair: the dense gradient for the dense input and
  # nothing for the indices input. No side effects beyond adding graph ops.
  """Gradient for dense_to_csr_sparse_matrix op."""
  grad_values = (
      sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          grad, type=op.get_attr("T")))
  # inputs to fw op were: params, indices.
  return (grad_values, None)


@ops.RegisterGradient("CSRSparseMatrixToDense")
def _CSRSparseMatrixToDenseGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the CSR-to-dense conversion. Extracts the nonzero coordinates of the forward
  # input CSR matrix (using the gradient's element type), gathers the dense gradient values at
  # those coordinates, and rebuilds a CSR matrix with the dense gradient's static shape. Returns
  # a single CSR matrix gradient for the one input. Requires a fully defined gradient shape.
  """Gradient for csr_sparse_matrix_to_dense op."""
  coo_sparse_tensor = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
      op.inputs[0], type=grad.dtype)
  return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
      indices=coo_sparse_tensor.indices,
      values=array_ops.gather_nd(grad, coo_sparse_tensor.indices),
      dense_shape=grad.shape)


@ops.RegisterGradient("SparseTensorToCSRSparseMatrix")
def _SparseTensorToCSRSparseMatrixGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the COO-sparse-tensor-to-CSR conversion. Converts the incoming CSR gradient to
  # a COO sparse tensor using the forward op's element type and keeps only its values. Returns
  # a triple: nothing for indices, the values gradient, and nothing for the dense shape.
  """Gradient for sparse_tensor_to_csr_sparse_matrix op."""
  grad_values = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
      grad, type=op.get_attr("T")).values
  return (None, grad_values, None)


@ops.RegisterGradient("CSRSparseMatrixToSparseTensor")
def _CSRSparseMatrixToSparseTensorGrad(op: ops.Operation, *grads):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Only the values gradient (second incoming gradient) is used; gradients for the
  # indices and dense shape outputs are ignored, which is correct since they are integers.
  # Gradient of the CSR-to-COO conversion. Builds a CSR matrix from the forward op's output
  # indices, the incoming values gradient and the forward output dense shape, and returns it as
  # the gradient of the single CSR input.
  """Gradient for csr_sparse_matrix_to_sparse_tensor op."""
  return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
      indices=op.outputs[0], values=grads[1], dense_shape=op.outputs[2])


ops.NotDifferentiable("SparseMatrixNNZ")

ops.NotDifferentiable("SparseMatrixZeros")


def _PruneSparseTensor(unpruned, pruned_pattern):
  # [FUNCTION DOC]
  # HELPER: Used by the CSR gradients (via the CSR pruning helper) to restrict a gradient to the
  # sparsity pattern of a forward input.
  # KNOWN ISSUE: The docstring says it returns indices, values and dense shape, which is true,
  # but the algorithm flattens both tensors to one dimension and assumes the flattened indices
  # are sorted (it uses a sorted search); unsorted inputs give wrong matches. The docstring
  # also admits it is not optimal.
  # Takes two COO sparse tensors. Flattens both to rank one, finds for each index of the
  # pattern the matching entry in the unpruned tensor, and keeps only entries whose index is
  # present in both. Returns a tuple of (kept indices, kept values, dense shape of the pattern).
  """Helper function to prune COO sparse tensor.

  Given two sparse tensors 'unpruned' and 'pruned_pattern', generates another
  sparse tensor with indices and values fron 'unpruned' only if its indices also
  occur in pruned_pattern.

  Args:
    unpruned: COO matrix with unpruned indices
    pruned_pattern: COO matrix with pruned pattern.

  TODO(tabakg): This is far from optimal. Consider a C++ implementation.

  Returns:
    Indices, values, and dense_shape of the pruned matrix.
  """
  pruned_indices = sparse_ops.sparse_reshape(
      pruned_pattern, shape=(-1,)).indices[..., 0]
  unpruned_indices = sparse_ops.sparse_reshape(
      unpruned, shape=(-1,)).indices[..., 0]
  best_match = array_ops.searchsorted(unpruned_indices, pruned_indices)
  keep_indices = array_ops.gather(
      best_match,
      array_ops.where(
          math_ops.equal(
              array_ops.gather(unpruned_indices, best_match), pruned_indices)))
  return (array_ops.gather_nd(unpruned.indices, keep_indices),
          array_ops.gather_nd(unpruned.values,
                              keep_indices), pruned_pattern.dense_shape)


def _PruneCSRMatrix(unpruned, pruned_pattern):
  # [FUNCTION DOC]
  # HELPER: Used by the add and sparse-sparse matmul gradients to mask gradients.
  # KNOWN ISSUE: The docstring is only a TODO note and does not describe the behavior.
  # Takes two CSR matrices. Reads the element type of the pattern, converts both to COO sparse
  # tensors, prunes the first to the indices present in the pattern, and converts the result
  # back to a CSR matrix which is returned.
  """TODO(tabakg): Consider re-writing in C++."""
  _, dtype = sparse_csr_matrix_ops.dense_shape_and_type(pruned_pattern)
  coo_unpruned = sparse_tensor.SparseTensor(
      *sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
          unpruned, type=dtype))
  coo_pruned_pattern = sparse_tensor.SparseTensor(
      *sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
          pruned_pattern, type=dtype))
  return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
      *_PruneSparseTensor(coo_unpruned, coo_pruned_pattern))


@ops.RegisterGradient("SparseMatrixAdd")
def _SparseMatrixAddGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Gradients with respect to the scalar inputs alpha and beta are not implemented
  # (returned as nothing), and a comment in the code admits the transposes may be wrong for
  # those terms.
  # Gradient of the sparse matrix add (alpha * a + beta * b). The forward inputs are two CSR
  # matrices and two scalars. The incoming CSR gradient is pruned to each input's sparsity
  # pattern and scaled by alpha and beta respectively. Returns a four-tuple: gradient for a,
  # gradient for b, nothing, nothing.
  """Gradient for sparse_matrix_add op."""
  # input to sparse_matrix_add is (a, b, alpha, beta)
  # with a, b CSR and alpha beta scalars.
  # output is: alpha * a + beta * b

  # d(a*A + b*B)/dA . grad = a * grad

  # May have gotten the transposes wrong below.
  # d(a*A + b*B)/da . grad = tr(A' . grad)

  # For now, only implement gradients w.r.t. A and B.
  # TODO(ebrevdo): Implement reduce_sum for SparseMatrix so that we
  # can implement gradients w.r.t. a and b.
  (a_csr, b_csr, alpha, beta) = op.inputs
  return (sparse_csr_matrix_ops.sparse_matrix_mul(
      _PruneCSRMatrix(grad, a_csr), alpha),
          sparse_csr_matrix_ops.sparse_matrix_mul(
              _PruneCSRMatrix(grad, b_csr), beta), None, None)


def _PrunedDenseMatrixMultiplication(a,
                                     b,
                                     indices,
                                     transpose_a=False,
                                     adjoint_a=False,
                                     transpose_b=False,
                                     adjoint_b=False):
  # [FUNCTION DOC]
  # HELPER: Used by the sparse-times-dense matmul gradient to compute gradients only at the
  # nonzero positions of the sparse operand.
  # KNOWN ISSUE: Only ranks 2 and 3 are handled; for any other rank the row/column/gather
  # variables are never assigned and a name error would occur. The docstring claims these
  # inputs must have matching rank but this is not checked.
  # Multiplies two dense matrices (or batches of matrices) but computes output values only at
  # the given indices. Parameters: a and b are the dense operands; indices is an integer tensor
  # of selected output coordinates (row, column, or batch, row, column); transpose_a/b and
  # adjoint_a/b optionally transpose or conjugate-transpose the operands (adjoint implies
  # transpose). Gathers the needed rows of a and columns of b, multiplies and sums them, and
  # returns a CSR matrix with those indices, the computed values and the product's dense shape.
  """Multiplies two dense matrices at selected indices.

  The two inputs `a` and `b` must have matching rank (2 or 3). If using rank 3,
  the first rank is used for the batch number. The last two dimensions should
  also be compatible for matrix multiplication.

  TODO(tabakg): Consider C++ implementation. There is also a more efficient way
  to handle transposes here.

  Args:
    a: The left dense matrix (or batched matrices).
    b: The right dense matrix (or batched matrices).
    indices: The selected output indices where values should be produced. Other
      indices will be pruned (not computed in the first place). Indices are
      specified as a tensor of shape (length, rank), where length is the number
      of entries and rank is the rank of the dense inputs (2 or 3).
    transpose_a: Whether to transpose a.
    adjoint_a: Whether to take the conjugate transpose of a.
    transpose_b: Whether to transpose b.
    adjoint_b: Whether to take the conjugate transpose of b.

  Returns:
    A CSR matrix.
  """
  transpose_a = transpose_a or adjoint_a
  transpose_b = transpose_b or adjoint_b

  a = math_ops.conj(a) if adjoint_a else a
  b = math_ops.conj(b) if adjoint_b else b

  rank = len(a.shape)
  dense_shape = (a.shape[-1] if transpose_a else a.shape[-2],
                 b.shape[-2] if transpose_b else b.shape[-1])
  if rank == 2:
    rows = indices[:, 0]
    cols = indices[:, 1]
    transpose = array_ops.transpose
    gather_op = array_ops.gather
  elif rank == 3:
    dense_shape = (a.shape[0],) + dense_shape
    rows = indices[:, :2]
    cols = array_ops_stack.stack([indices[:, 0], indices[:, 2]], axis=1)
    transpose = lambda x: array_ops.transpose(x, perm=[0, 2, 1])
    gather_op = array_ops.gather_nd

  a_rows = gather_op(transpose(a) if transpose_a else a, indices=rows)
  b_cols = gather_op(b if transpose_b else transpose(b), indices=cols)
  values = math_ops.reduce_sum(a_rows * b_cols, axis=1)

  return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
      indices=indices, values=values, dense_shape=dense_shape)


@ops.RegisterGradient("SparseMatrixTranspose")
def _SparseMatrixTransposeGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the sparse matrix transpose: transposing the incoming CSR gradient again, with
  # the same element type and conjugate setting as the forward op. Returns the CSR gradient.
  """Gradient for sparse_matrix_transpose op."""
  return sparse_csr_matrix_ops.sparse_matrix_transpose(
      grad, type=op.get_attr("type"), conjugate=op.get_attr("conjugate"))


@ops.RegisterGradient("SparseMatrixSoftmax")
def _SparseMatrixSoftmaxGrad(op: ops.Operation, grad_softmax):
  # [FUNCTION DOC]
  # Gradient of the sparse matrix softmax. Uses the forward output (the softmax result) and the
  # incoming gradient with the dedicated softmax-gradient op, using the forward element type.
  # Returns the CSR gradient for the single input.
  """Gradient for sparse_matrix_softmax op."""
  softmax = op.outputs[0]
  return sparse_csr_matrix_ops.sparse_matrix_softmax_grad(
      softmax, grad_softmax, type=op.get_attr("type"))


@ops.RegisterGradient("SparseMatrixMatMul")
def _SparseMatrixMatMulGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Attribute combinations where adjoint and transpose flags are both set for the
  # same operand, or both operands use mixed forms not covered by the case analysis, fall
  # through all branches and leave the gradients unassigned, causing a name error.
  # Gradient of the sparse (CSR) times dense matrix multiply. Reads the transpose, adjoint,
  # transpose-output and conjugate-output attributes of the forward op, then in a large case
  # analysis computes the gradient for the sparse operand A (a dense product pruned to A's
  # nonzero pattern, returned as a CSR matrix) and for the dense operand B (another sparse
  # times dense multiply with flipped flags). Returns a pair (gradient A, gradient B).
  # Contains a nested helper that wraps the pruned dense multiplication with A's indices.
  """Gradient for sparse_matrix_mat_mul op."""
  # input to sparse_matrix_mat_mul is (A, B) with CSR A and dense B.
  # Output is dense:
  #   C = opA(A) . opB(B) if transpose_output = false
  #   C = (opA(A) . opB(B))' = opB(B)' . opA(A)' if transpose_output = true.
  # where opA = transpose if transpose_a = True else identity
  # and   opB = transpose if transpose_b = True else identity

  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  adj_a = op.get_attr("adjoint_a")
  adj_b = op.get_attr("adjoint_b")
  transpose_output = op.get_attr("transpose_output")
  conjugate_output = op.get_attr("conjugate_output")
  a = op.inputs[0]  # sparse matrix
  b = op.inputs[1]  # dense matrix
  conj = math_ops.conj
  sparse_matmul = sparse_csr_matrix_ops.sparse_matrix_mat_mul

  def matmul(x, y, **kwargs):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: Local shortcut for the matmul gradient above.
    # Multiplies dense x and y, producing values only at the nonzero indices of the sparse
    # operand A (taken from the enclosing scope, extracted using x's element type). Extra
    # keyword arguments (transpose and adjoint flags) are forwarded. Returns a CSR matrix.
    return _PrunedDenseMatrixMultiplication(
        x,
        y,
        indices=sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
            a, type=x.dtype).indices,
        **kwargs)

  if conjugate_output:
    grad = conj(grad)
  if not transpose_output:
    # C = opA(A) . opB(B)
    if not adj_a and not adj_b:
      a = conj(a)
      b = conj(b)
      if not t_a:
        grad_a = matmul(grad, b, transpose_b=not t_b)
      else:
        grad_a = matmul(b, grad, transpose_a=t_b, transpose_b=True)
      grad_b = sparse_matmul(a, grad, transpose_a=not t_a, transpose_output=t_b)
    elif not t_a and not t_b:
      if not adj_a:
        grad_a = matmul(grad, b, adjoint_b=not adj_b)
      else:
        grad_a = matmul(b, grad, adjoint_a=adj_b, adjoint_b=True)
      grad_b = sparse_matmul(
          a,
          grad,
          adjoint_a=not adj_a,
          transpose_output=adj_b,
          conjugate_output=adj_b)
    elif adj_a and t_b:
      grad_a = matmul(b, grad, transpose_a=True, adjoint_b=True)
      grad_b = sparse_matmul(a, grad, transpose_output=True)
    elif t_a and adj_b:
      grad_a = matmul(b, grad, transpose_a=True, transpose_b=True)
      grad_b = sparse_matmul(
          conj(a), grad, transpose_output=True, conjugate_output=True)
  else:
    # C = (opA(A) . opB(B))' =  opB(B)' . opA(A)'
    if not adj_a and not adj_b:
      a = conj(a)
      b = conj(b)
      if not t_a:
        grad_a = matmul(grad, b, transpose_a=True, transpose_b=not t_b)
      else:
        grad_a = matmul(b, grad, transpose_a=t_b)
      grad_b = sparse_matmul(
          a, grad, transpose_a=not t_a, transpose_b=True, transpose_output=t_b)
    elif not t_a and not t_b:
      if not adj_a:
        grad_a = matmul(grad, b, transpose_a=True, adjoint_b=not adj_b)
      else:
        grad_a = matmul(b, conj(grad), adjoint_a=adj_b)
      grad_b = sparse_matmul(
          a,
          grad,
          adjoint_a=not adj_a,
          transpose_b=True,
          transpose_output=adj_b,
          conjugate_output=adj_b)
    elif adj_a and t_b:
      grad_a = matmul(b, conj(grad), transpose_a=True)
      grad_b = sparse_matmul(a, grad, transpose_b=True, transpose_output=True)
    elif t_a and adj_b:
      grad_a = matmul(b, grad, transpose_a=True)
      grad_b = sparse_matmul(a, grad, adjoint_b=True, transpose_output=True)

  return (grad_a, grad_b)


@ops.RegisterGradient("SparseMatrixSparseMatMul")
def _SparseMatrixSparseMatMulGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Combinations with both adjoint and transpose flags set on the same operand
  # (for example adjoint_a with transpose_a) are not covered and would leave gradients
  # unassigned, raising a name error. A TODO notes pruning after multiplication is wasteful.
  # Gradient of the sparse-times-sparse matrix multiply. Reads transpose, adjoint and type
  # attributes of the forward op and, by case analysis, computes the gradient for each operand
  # with further sparse-sparse multiplications (conjugating where required). Each result is
  # then pruned to the sparsity pattern of its corresponding input. Returns a pair of CSR
  # matrices (gradient A, gradient B).
  """Gradient for sparse_matrix_sparse_mat_mul op."""
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  adj_a = op.get_attr("adjoint_a")
  adj_b = op.get_attr("adjoint_b")
  dtype = op.get_attr("type")

  # input to sparse_matrix_sparse_mat_mul is (A, B) with CSR A and B.
  # Output is CSR:
  #   C = opA(A) . opB(B)
  # where opA = transpose if transpose_a = True else identity
  # and   opB = transpose if transpose_b = True else identity
  a = op.inputs[0]
  b = op.inputs[1]
  conj = math_ops.conj
  matmul = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul
  if not t_a and not t_b:
    if not adj_a:
      if not adj_b:
        grad_a = matmul(grad, b, adjoint_b=True, type=dtype)
        grad_b = matmul(a, grad, adjoint_a=True, type=dtype)
      else:
        grad_a = matmul(grad, b, type=dtype)
        grad_b = matmul(grad, a, adjoint_a=True, type=dtype)
    else:
      if not adj_b:
        grad_a = matmul(b, grad, adjoint_b=True, type=dtype)
        grad_b = matmul(a, grad, type=dtype)
      else:
        grad_a = matmul(b, grad, adjoint_a=True, adjoint_b=True, type=dtype)
        grad_b = matmul(grad, a, adjoint_a=True, adjoint_b=True, type=dtype)
  elif not adj_a and not adj_b:
    if not t_a and t_b:
      grad_a = matmul(grad, conj(b), type=dtype)
      grad_b = matmul(grad, conj(a), transpose_a=True, type=dtype)
    elif t_a and not t_b:
      grad_a = matmul(conj(b), grad, transpose_b=True, type=dtype)
      grad_b = matmul(conj(a), grad, type=dtype)
    else:
      grad_a = matmul(b, grad, adjoint_a=True, transpose_b=True, type=dtype)
      grad_b = matmul(grad, a, transpose_a=True, adjoint_b=True, type=dtype)
  elif adj_a and t_b:
    grad_a = matmul(b, grad, transpose_a=True, adjoint_b=True, type=dtype)
    grad_b = matmul(grad, a, transpose_a=True, transpose_b=True, type=dtype)
  elif t_a and adj_b:
    grad_a = matmul(b, grad, transpose_a=True, transpose_b=True, type=dtype)
    grad_b = matmul(grad, a, adjoint_a=True, transpose_b=True, type=dtype)

  # TODO(tabakg): There should be a C++ function for sparse-sparse
  # multiplication with pre-determined indices, instead of pruning after the
  # multiplication.
  return (_PruneCSRMatrix(grad_a, a), _PruneCSRMatrix(grad_b, b))


@ops.RegisterGradient("SparseMatrixMul")
def _SparseMatrixMulGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring claims to be a gradient but this is an unimplemented stub: it
  # discards its arguments and always raises.
  # Registered gradient for the sparse matrix elementwise multiply. Always raises a
  # not-implemented error, so differentiating through this op fails.
  """Gradient for sparse_matrix_mul op."""
  # input to sparse_matrix_mul is (A, B) with CSR A and dense B.
  # Output is CSR:
  #   C = A .* B
  del op
  del grad
  raise NotImplementedError
