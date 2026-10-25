# [MODULE DOC]
# Python front end for CSR (compressed sparse row) sparse matrices stored in variant tensors.
# Re-exports all generated raw ops from the generated sparse CSR matrix ops module, and defines
# the abstract SparseMatrix base class, the concrete CSRSparseMatrix class, a matmul function
# that dispatches between dense and sparse operands, and helpers that read or build the "handle
# data" (dense shape and dtype carried alongside the variant tensor), which is needed in eager
# mode where C++ shape inference does not run. The public export list contains SparseMatrix,
# CSRSparseMatrix, matmul, dense_shape_and_type and every public generated op name. Gradients
# are registered in the companion sparse_csr_matrix_grad module.
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
"""CSR Sparse Matrix Operations."""

import abc
import collections

# pylint: disable=g-direct-tensorflow-import, wildcard-import
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops as sm_ops
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import *


__all__ = [
    "SparseMatrix",
    "CSRSparseMatrix",
    "matmul",
    "dense_shape_and_type",
]
# pylint: disable=invalid-name
__all__ += [_x for _x in dir(sm_ops) if not _x.startswith("_")]


class DenseShapeAndType(
    collections.namedtuple("DenseShapeAndType", ("shape", "dtype"))):
  # [CLASS DOC]
  # Immutable pair (named tuple) holding the dense shape (a tensor shape) and the element type of
  # a sparse matrix. Returned by the function that reads a CSR matrix's handle data. No extra
  # behavior beyond the two named fields.
  pass


def _get_handle_data(tensor):
  # [FUNCTION DOC]
  # HELPER: Internal accessor used when reading CSR matrix shape and type metadata.
  # Returns the handle data (shape and type protobuf) attached to a tensor, in an eager-safe
  # way, or nothing if none is available.
  return resource_variable_ops.get_eager_safe_handle_data(tensor)


def _create_handle_data_proto(shape_proto, dtype_enum):
  # [FUNCTION DOC]
  # HELPER: Builds metadata for CSR matrices, used by the constructor and matmul shape inference.
  # Given a shape protobuf and a data type enumeration value, creates a handle data message that
  # is marked as set and contains exactly one shape-and-type entry. Returns that message.
  """Create handle data based on shape and dtype protos."""
  variant_shape_and_type_data = \
    cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
  variant_shape_and_type_data.is_set = True
  # NOTE(ebrevdo): shape_and_type lacks append() in some versions of protobuf.
  variant_shape_and_type_data.shape_and_type.extend([
      cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
          shape=shape_proto, dtype=dtype_enum)
  ])
  return variant_shape_and_type_data


def _make_handle_data(tensor):
  # [FUNCTION DOC]
  # HELPER: Used by the CSR matrix constructor in eager mode.
  # Creates handle data from a dense tensor's static shape and element type. Returns the handle
  # data message.
  """Create handle data based on tensor shape and dtype."""
  return _create_handle_data_proto(tensor.shape.as_proto(),
                                   tensor.dtype.as_datatype_enum)


def get_shape_and_type(matrix):
  # [FUNCTION DOC]
  # Returns the single shape-and-type entry from the matrix tensor's cached handle data
  # attribute, or nothing if the tensor has no handle data. Raises a value error if the handle
  # data has a number of entries other than one. Reads only; no side effects.
  """Return matrix's shape and type if available."""
  handle_data = getattr(matrix, "_handle_data", None)
  if handle_data is None:
    return None
  if len(handle_data.shape_and_type) != 1:
    raise ValueError(
        "shape_and_type array in _handle_data must have length one, but saw: %d"
        % len(handle_data.shape_and_type))
  return handle_data.shape_and_type[0]


def dense_shape_and_type(matrix):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring says it returns a ShapeAndType; the code actually returns an
  # instance of the DenseShapeAndType named tuple defined in this module.
  # Reads the dense shape and element type of a variant tensor holding a sparse matrix. Input:
  # a tensor of variant type. Output: DenseShapeAndType with a tensor shape and a dtype. Raises
  # a type error if the input is not a tensor or is not of variant type, and a value error if
  # handle data is missing, unset, or does not have exactly one entry.
  """Get dense shape and dtype of the tf.Tensor containing the matrix.

  Args:
    matrix: A `tf.Tensor` of type `tf.variant` storing a sparse matrix.

  Returns:
    An instance of `ShapeAndType` with properties `shape` (a `tf.TensorShape`)
    and `dtype` (a `tf.DType`).

  Raises:
    TypeError: if `matrix` is not a tensor or its dtype is not variant.
    ValueError: if `matrix` lacks static handle data containing the dense
      shape and dtype.
  """
  if not isinstance(matrix, tensor_lib.Tensor):
    raise TypeError("matrix should be a tensor, but saw: %s" % (matrix,))
  if matrix.dtype != dtypes.variant:
    raise TypeError(
        "expected matrix to be type tf.variant, but saw: %s" % (matrix.dtype,))
  handle_data = _get_handle_data(matrix)
  if not handle_data or not handle_data.is_set:
    raise ValueError("matrix has missing handle data: %s" % (matrix,))
  if len(handle_data.shape_and_type) != 1:
    raise ValueError("len(matrix.handle_data.shape_and_type) != 1: '%s'" %
                     (handle_data.shape_and_type,))
  return DenseShapeAndType(
      tensor_shape.TensorShape(handle_data.shape_and_type[0].shape),
      dtypes.DType(handle_data.shape_and_type[0].dtype))


def matmul_shape_inference(a, b, c, transpose_a, transpose_b, adjoint_a,
                           adjoint_b):
  # [FUNCTION DOC]
  # HELPER: Used by the matmul function when multiplying two sparse matrices in eager mode.
  # Computes handle data (dense shape and type) for the product of sparse matrices a and b.
  # If the result tensor c already has handle data it is returned unchanged. Otherwise, if both
  # inputs have handle data, the result shape is the batch dimensions of a followed by the row
  # count of a (or its column count when transposed or adjoint) and the column count of b (or
  # its row count when transposed or adjoint); the type is a's type. If either input lacks
  # handle data, returns whatever c has (possibly nothing). Does not modify its arguments.
  """Helper function for matmul to set the result matrix's handle data."""
  c_handle = getattr(c, "_handle_data", None)
  a_shape_and_type = get_shape_and_type(a)
  b_shape_and_type = get_shape_and_type(b)
  if (c_handle is None and a_shape_and_type is not None and
      b_shape_and_type is not None):

    transpose_a = transpose_a or adjoint_a
    transpose_b = transpose_b or adjoint_b

    a_shape = a_shape_and_type.shape
    b_shape = b_shape_and_type.shape
    rank = len(a_shape.dim)

    # Creates the output shape.
    c_rows = a_shape.dim[rank - (1 if transpose_a else 2)].size
    c_cols = b_shape.dim[rank - (2 if transpose_b else 1)].size
    c_shape = tensor_shape.TensorShape(a_shape)
    c_shape = tensor_shape.TensorShape(c_shape[:rank - 2] + [c_rows, c_cols])
    c_handle = _create_handle_data_proto(c_shape.as_proto(),
                                         a_shape_and_type.dtype)
  return c_handle


def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           name=None):
  # [FUNCTION DOC]
  # Matrix multiply that accepts dense tensors and sparse matrix objects. Inputs a and b are
  # dense tensors or sparse matrices of rank 2 or 3; the transpose and adjoint flags choose
  # the operation applied to each operand (adjoint implies conjugate transpose); name is an
  # optional name scope. Behavior: if neither operand is sparse, delegates to the ordinary dense
  # matrix multiply. If both are sparse, they must be of related types (one a subclass of the
  # other) or a type error is raised; the sparse-sparse product op is run, output handle data
  # is inferred (needed in eager mode) and a new sparse matrix of a's kind is returned. If only
  # a is sparse, the sparse-times-dense op is used and a dense tensor is returned. If only b is
  # sparse, the product is computed through the transposed identity with transposed output
  # (conjugating as needed for adjoints) and a dense tensor is returned.
  """Perform a sparse matrix matmul between `a` and `b`.

  Performs a contraction between `a` and `b` along the two innermost dimensions.
  If both `a` and `b` are instances of `SparseMatrix`, returns a new instance
  of `SparseMatrix` (same type as `a`).  If one is not an instance of
  `SparseMatrix`, returns a dense `Tensor`:

  ```
  c = opA(a) . opB(b)
  ```
  where `opA` (resp. `opB`) is the transpose or hermitian transpose depending
  on the values of `transpose_a` (resp. `transpose_b`) and `adjoint_a`
  (resp. `adjoint_b`).

  Args:
    a: `Tensor` or `SparseMatrix`, having rank `2` or `3`.
    b: `Tensor` or `SparseMatrix`, having rank `2` or `3`.
    transpose_a: Python `bool`.
    transpose_b: Python `bool`.
    adjoint_a: Python `bool`.
    adjoint_b: Python `bool`.
    name: Optional name to use when creating ops.

  Returns:
    A `SparseMatrix` if both `a` and `b` are instances of `SparseMatrix`,
    otherwise a dense `Tensor`.
  """
  if not isinstance(a, SparseMatrix) and not isinstance(b, SparseMatrix):
    return math_ops.matmul(
        a,
        b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b,
        name=name)

  # pylint: disable=protected-access
  a_matrix = a._matrix if isinstance(a, SparseMatrix) else a
  b_matrix = b._matrix if isinstance(b, SparseMatrix) else b
  with ops.name_scope(name, "SparseMatrixMatMul", [a_matrix, b_matrix]):
    if isinstance(a, SparseMatrix) and isinstance(b, SparseMatrix):
      if not (isinstance(a, type(b)) or isinstance(b, type(a))):
        raise TypeError("SparseMatrix types don't inherit from each other: "
                        "%s and %s" % (type(a), type(b)))
      c = sm_ops.sparse_matrix_sparse_mat_mul(
          a_matrix,
          b_matrix,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b,
          type=a.dtype)

      # In eager mode, shape inference functions are not called, and the output
      # shape is not set. We have to infer the output shape here.
      # TODO(penporn): Set this from the C++ kernel instead.
      c_handle = matmul_shape_inference(a_matrix, b_matrix, c, transpose_a,
                                        transpose_b, adjoint_a, adjoint_b)
      return a._from_matrix(c, handle_data=c_handle)

    elif isinstance(a, SparseMatrix):
      return sm_ops.sparse_matrix_mat_mul(
          a_matrix,
          b,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
    else:
      # opA(A) . opB(B) = t(nopB(B) . nopA(A))
      if not adjoint_a and not adjoint_b:
        return sm_ops.sparse_matrix_mat_mul(
            b_matrix,
            a,
            transpose_a=not transpose_b,
            transpose_b=not transpose_a,
            transpose_output=True)
      elif not transpose_a and not transpose_b:
        return sm_ops.sparse_matrix_mat_mul(
            b_matrix,
            a,
            adjoint_a=not adjoint_b,
            adjoint_b=not adjoint_a,
            transpose_output=True,
            conjugate_output=True)
      else:
        return sm_ops.sparse_matrix_mat_mul(
            b_matrix,
            math_ops.conj(a),
            transpose_output=True,
            conjugate_output=adjoint_b)


class SparseMatrix(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class for sparse matrix types backed by a variant tensor. Subclasses must
  # supply the underlying variant tensor, a way to wrap a new variant tensor into the same
  # kind of object, and conversion to dense and COO sparse tensor forms. The base class records
  # whether it was created in eager mode and provides shared operations: graph, shape and dtype
  # properties, conjugate, hermitian transpose, transpose, stored-value count and sorted indices.
  """Abstract class for sparse matrix types."""

  @abc.abstractmethod
  def __init__(self):
    # [FUNCTION DOC]
    # Abstract constructor. Takes no arguments and records whether the program is currently
    # executing eagerly in an internal flag used later to decide whether handle data must be
    # tracked. Subclasses are expected to call it.
    self._eager_mode = context.executing_eagerly()

  @abc.abstractproperty
  def _matrix(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Declared with the abstract-property decorator, which is deprecated in favor
    # of combining the abstract-method and property decorators.
    # Abstract property: subclasses return the variant tensor holding the sparse matrix.
    pass

  @abc.abstractmethod
  def _from_matrix(self, matrix, handle_data=None):
    # [FUNCTION DOC]
    # Abstract method: subclasses wrap a variant tensor (matrix) into a new instance of their
    # own kind, optionally using handle data (needed in eager mode). Returns the new instance.
    pass

  @abc.abstractmethod
  def to_dense(self):
    # [FUNCTION DOC]
    # Abstract method: subclasses return the matrix as a dense tensor.
    pass

  @abc.abstractmethod
  def to_sparse_tensor(self):
    # [FUNCTION DOC]
    # Abstract method: subclasses return the matrix as a COO sparse tensor.
    pass

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Property returning the graph of the underlying variant tensor.
    return self._matrix.graph

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property returning the dense shape (tensor shape) taken from the underlying tensor's handle
    # data. Raises the errors of the shape-and-type reader if handle data is missing.
    return dense_shape_and_type(self._matrix).shape

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the element type taken from the underlying tensor's handle data.
    # Raises the errors of the shape-and-type reader if handle data is missing.
    return dense_shape_and_type(self._matrix).dtype

  @property
  def eager_handle_data(self):
    # [FUNCTION DOC]
    # Property returning the underlying tensor's handle data when the matrix was created in
    # eager mode, otherwise nothing.
    """Return the matrix's handle data iff in eager mode."""
    return _get_handle_data(self._matrix) if self._eager_mode else None

  def conj(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Applies the generic elementwise conjugate math op directly to the variant
    # tensor rather than a sparse-matrix-specific op; this relies on a conjugate kernel
    # registered for the variant type.
    # Returns a new sparse matrix of the same kind containing the complex conjugate of this one.
    return self._from_matrix(
        math_ops.conj(self._matrix), self.eager_handle_data)

  def hermitian_transpose(self):
    # [FUNCTION DOC]
    # Returns a new sparse matrix of the same kind holding the conjugate transpose of this one,
    # computed with the sparse transpose op with conjugation enabled.
    """Return the hermitian transpose of the matrix."""
    return self._from_matrix(
        sm_ops.sparse_matrix_transpose(
            self._matrix, conjugate=True, type=self.dtype),
        self.eager_handle_data)

  def nnz(self):
    # [FUNCTION DOC]
    # Returns an integer tensor with the number of stored values (per batch), counting any
    # explicitly stored zeros. The alias named nonzero refers to this same method.
    """Number of stored values, including explicit zeros."""
    return sm_ops.sparse_matrix_nnz(self._matrix)

  nonzero = nnz

  def sorted_indices(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The name suggests an explicit sort, but it just converts to a COO sparse
    # tensor and returns its indices, relying on the conversion producing row-major order. A
    # TODO notes it may be inefficient.
    # Returns the integer tensor of coordinates of stored values.
    # TODO(ebrevdo): A more efficient implementation?
    return self.to_sparse_tensor().indices

  def transpose(self):
    # [FUNCTION DOC]
    # Returns a new sparse matrix of the same kind holding the (non-conjugated) transpose of
    # this one, using the sparse transpose op and preserving eager handle data.
    return self._from_matrix(
        sm_ops.sparse_matrix_transpose(self._matrix, type=self.dtype),
        self.eager_handle_data)


class CSRSparseMatrix(SparseMatrix):
  # [CLASS DOC]
  # Concrete sparse matrix in compressed sparse row format, optionally batched, stored in a
  # variant tensor. Created from a dense rank 2 or 3 tensor (with optional nonzero indices) or
  # from a COO sparse tensor. Holds the CSR variant tensor and the element type; in eager mode
  # also sets handle data on the tensor. Implements the abstract conversions to dense and COO.
  """(Optionally batched) CSR Sparse Matrix."""

  def __init__(self, value, indices=None, name=None):
    # [FUNCTION DOC]
    # Builds the CSR matrix. value is a dense rank 2 or 3 tensor or a COO sparse tensor; indices
    # (optional) are the nonzero coordinates when value is dense, defaulting to the coordinates
    # where value is nonzero (computed with gradient flow stopped); name is ignored. For a sparse
    # tensor value the indices, values and dense shape are converted directly. Raises a value
    # error if value is a sparse tensor and indices is given. In eager mode, sets handle data
    # (shape and dtype) on the resulting variant tensor. Stores the element type and the tensor.
    """Construct a CSRSparseMatrix from a dense matrix or SparseTensor.

    Args:
      value: A dense `2D` or `3D` Tensor or `SparseTensor`.
      indices: The nonzero indices of `value`
        (if `value` is not a `SparseTensor`).
      name: Optional op name.

    Raises:
      ValueError: if `value` is a `SparseTensor` and `indices` is not `None`.
    """
    del name  # Unused.
    super(CSRSparseMatrix, self).__init__()
    if isinstance(value, sparse_tensor.SparseTensor):
      if indices is not None:
        raise ValueError("indices must be None if value is a SparseTensor.")
      self._dtype = value.dtype
      self._csr_matrix = sm_ops.sparse_tensor_to_csr_sparse_matrix(
          indices=value.indices,
          values=value.values,
          dense_shape=value.dense_shape)
    else:
      value = ops.convert_to_tensor(value)
      self._dtype = value.dtype
      if indices is not None:
        indices = ops.convert_to_tensor(indices, dtype=dtypes.int64)
      else:
        indices = array_ops.stop_gradient(array_ops.where(value))
      self._csr_matrix = sm_ops.dense_to_csr_sparse_matrix(value, indices)

    # Eager mode doesn't call shape inference functions, so we have to set the
    # shape and dtype handle data directly.
    if self._eager_mode:
      # pylint: disable=protected-access
      self._csr_matrix._handle_data = _make_handle_data(value)
      # pylint: enable=protected-access

  @property
  def _matrix(self):
    # [FUNCTION DOC]
    # Property returning the stored CSR variant tensor.
    return self._csr_matrix

  def _from_matrix(self, matrix, handle_data=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Validates input with assertions (disabled under optimized Python) instead of
    # raising proper errors; and in eager mode it reads the private handle data attribute of
    # the tensor, failing with an assertion if neither it nor handle_data is available.
    # Wraps a variant tensor into a new instance of the same class without running the
    # constructor, copying the element type. matrix must be a variant tensor. In eager mode, if
    # the tensor lacks handle data, handle_data is attached to it (mutating the tensor). Returns
    # the new instance.
    assert (
        isinstance(matrix, tensor_lib.Tensor) and matrix.dtype == dtypes.variant
    )
    ret = type(self).__new__(type(self))
    # pylint: disable=protected-access
    ret._dtype = self._dtype
    if self._eager_mode:
      if matrix._handle_data is None:
        matrix._handle_data = handle_data
      assert matrix._handle_data is not None
    ret._csr_matrix = matrix
    # pylint: enable=protected-access
    return ret

  def to_dense(self):
    # [FUNCTION DOC]
    # Converts the matrix to a dense tensor of the matrix's element type and returns it.
    return sm_ops.csr_sparse_matrix_to_dense(self._matrix, type=self.dtype)

  def to_sparse_tensor(self):
    # [FUNCTION DOC]
    # Converts the matrix to a COO sparse tensor (indices, values, dense shape) and returns it.
    r = sm_ops.csr_sparse_matrix_to_sparse_tensor(self._matrix, type=self.dtype)
    return sparse_tensor.SparseTensor(
        indices=r.indices, values=r.values, dense_shape=r.dense_shape)
