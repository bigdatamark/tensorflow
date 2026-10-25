# [MODULE DOC]
# Unit tests for the sparse tensor operations in the sparse ops module (sparse_eye, from_dense,
# sparse_to_dense, transpose, expand_dims, unary op dispatch on sparse tensors, dense-by-sparse
# matmul, map_values, concat) and for a few raw generated sparse kernels such as
# SparseFillEmptyRowsGrad. Tests run in both graph and eager modes through a class decorator.
# Importing the module also imports the array and sparse gradient modules so that gradients for
# Identity and SparseToDense are registered. Running it as a script starts the test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sparse ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
# Need array_grad to register gradient for Identity.
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gradient_checker_v2 as gradient_checker
from tensorflow.python.ops import math_ops
# Need sparse_grad to register gradient for SparseToDense.
from tensorflow.python.ops import sparse_grad  # pylint: disable=unused-import
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class SparseOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the public sparse ops (eye, conversion to and from dense, transpose,
  # expand_dims, unary dispatch, gradients, matmul, map_values). Decorated so that every test
  # method runs in both graph and eager modes. Holds no shared fixtures.

  def testSparseEye(self):
    # [FUNCTION DOC]
    # Checks sparse_eye against numpy's identity matrix for even sizes 2 to 8 in both
    # dimensions, with sizes given as Python ints and as constant tensors. The sparse result is
    # densified and compared for equality. Returns nothing; fails on mismatch.
    def test_one(n, m, as_tensors):
      # [FUNCTION DOC]
      # HELPER: nested helper of testSparseEye running one (rows, columns, as-tensors) case.
      # Builds the expected dense identity, optionally wraps sizes as tensors, builds the sparse
      # identity, densifies it and asserts equality with the expected matrix.
      expected = np.eye(n, m)
      if as_tensors:
        m = constant_op.constant(m)
        n = constant_op.constant(n)
      s = sparse_ops.sparse_eye(n, m)
      d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
      self.assertAllEqual(self.evaluate(d), expected)

    for n in range(2, 10, 2):
      for m in range(2, 10, 2):
        # Test with n and m as both constants and tensors.
        test_one(n, m, True)
        test_one(n, m, False)

  def testDenseFromConstantToSparse(self):
    # [FUNCTION DOC]
    # Converts a constant integer tensor of shape 3 by 4 by 2 to sparse with from_dense, converts
    # it back with sparse_to_dense, and asserts the round trip equals the original array.
    expected_constant = np.reshape(np.arange(24, dtype=np.int64), (3, 4, 2))
    tensor = constant_op.constant(expected_constant)
    sparse = sparse_ops.from_dense(tensor)
    dense = sparse_ops.sparse_to_dense(sparse.indices, sparse.dense_shape,
                                       sparse.values)
    constant = self.evaluate(dense)
    self.assertAllEqual(expected_constant, constant)

  def testTransposePreservesShape(self):
    # [FUNCTION DOC]
    # In a fresh graph, builds a sparse tensor with dense shape 3 by 4, transposes it and asserts
    # the static shape of the result is 4 by 3.
    with ops.Graph().as_default():
      t = sparse_tensor.SparseTensor(indices=[[0, 0]],
                                     values=[0.],
                                     dense_shape=[3, 4])
      self.assertTrue(t.shape.is_fully_defined)
      transposed = sparse_ops.sparse_transpose(t)
      self.assertAllEqual(transposed.shape, [4, 3])

  def testSparseExpandDims(self):
    # [FUNCTION DOC]
    # For ranks 1 to 3, builds a random sparse array (about 20 percent non-zero), then for every
    # valid axis (negative and positive) and with the axis given both as an int and as a tensor,
    # applies sparse_expand_dims, densifies and asserts equality with numpy's expand_dims.
    for rank in range(1, 4):
      # Create a dummy input. When rank=3, shape=[2, 4, 6].
      shape = np.arange(1, rank + 1) * 2
      before = np.arange(np.prod(shape)).reshape(shape)

      # Make entries sparse.
      before *= np.random.binomial(1, .2, before.shape)
      dense_shape = before.shape
      indices = np.array(np.where(before)).T
      values = before[before != 0]

      # Try every possible valid value of axis.
      for axis in range(-rank - 1, rank):
        expected_after = np.expand_dims(before, axis)

        for axis_as_tensor in [False, True]:
          dense_shape_t = constant_op.constant(dense_shape, dtype=dtypes.int64)
          indices_t = constant_op.constant(indices)
          values_t = constant_op.constant(values)
          before_t = sparse_tensor.SparseTensor(
              indices=indices_t, values=values_t, dense_shape=dense_shape_t)

          if axis_as_tensor:
            axis = constant_op.constant(axis)

          s = sparse_ops.sparse_expand_dims(before_t, axis)
          d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
          self.assertAllEqual(self.evaluate(d), expected_after)

  @parameterized.parameters([
      (math_ops.abs, [1.0, -1.0, 3.0, -4.0], [1.0, 1.0, 3.0, 4.0]),
      (math_ops.negative, [1.0, -1.0, 3.0, -4.0], [-1.0, 1.0, -3.0, 4.0]),
      (math_ops.sign, [3.0, -2.0, 0.0, -4.0], [1.0, -1.0, 0.0, -1.0]),
      (math_ops.square, [1.0, -1.0, 3.0, -4.0], [1.0, 1.0, 9.0, 16.0]),
      (
          math_ops.asinh,
          [1.0, -1.0, 3.0, -4.0],
          [0.8813736, -0.8813736, 1.8184465, -2.0947125],
      ),
      (
          math_ops.sin,
          [1.0, -1.0, 3.0, -4.0],
          [0.84147096, -0.84147096, 0.14112, 0.7568025],
      ),
      (
          math_ops.asin,
          [1.0, -1.0, 0.4, -0.5],
          [1.5707964, -1.5707964, 0.41151685, -0.5235988],
      ),
      (
          math_ops.tan,
          [1.0, -1.0, 0.4, -0.5],
          [1.5574077, -1.5574077, 0.42279324, -0.5463025],
      ),
      (
          math_ops.atan,
          [0.4, -0.4, 1.0, 0.5],
          [0.3805064, -0.3805064, 0.7853982, 0.4636476],
      ),
      (
          math_ops.atanh,
          [0.4, -0.4, -0.5, 0.5],
          [0.42364895, -0.42364895, -0.54930615, 0.54930615],
      ),
  ])
  def testUnarySparseDispatch(self, op, values, expected):
    # [FUNCTION DOC]
    # Parameterized by a unary math op (abs, negative, sign, square, asinh, sin, asin, tan, atan,
    # atanh), input values and expected outputs. Applies the op directly to a 3 by 6 sparse tensor
    # with four entries and asserts the result is sparse with unchanged indices and dense shape
    # and with values close to the expected ones.
    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [2, 0], [2, 4]],
        values=values,
        dense_shape=[3, 6])
    result = op(st)
    result_value = self.evaluate(result)
    self.assertAllEqual(result_value.indices, st.indices)
    self.assertAllClose(result_value.values, expected)
    self.assertAllEqual(result_value.dense_shape, st.dense_shape)

  def testSparseToDenseGradient(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of sparse_tensor_to_dense with respect to both the sparse
    # values and the default value, asserting the maximum error is below 1e-4.

    def f(sparse_values, default_value):
      # [FUNCTION DOC]
      # HELPER: function under gradient check for testSparseToDenseGradient. Builds a 3 by 6 by 9
      # sparse tensor with three entries from the given values and returns its dense form using
      # the given default value for missing entries.
      st = sparse_tensor.SparseTensor(
          indices=[[0, 3, 6], [1, 4, 7], [2, 5, 8]],
          values=sparse_values,
          dense_shape=[3, 6, 9])
      return sparse_ops.sparse_tensor_to_dense(st, default_value)

    grads = gradient_checker.compute_gradient(
        f, [constant_op.constant([1.0, 2.0, 3.0]),
            constant_op.constant(0.0)])
    epsilon = 1e-4
    self.assertLess(gradient_checker.max_error(*grads), epsilon)

  def testSparseTensorToDenseString(self):
    # [FUNCTION DOC]
    # Densifies a string-valued sparse tensor and asserts missing entries become empty byte
    # strings while present entries keep their values.
    sp = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=['a', 'b'], dense_shape=[2, 3])
    dense = sparse_ops.sparse_tensor_to_dense(sp)
    expected_dense = [[b'a', b'', b''], [b'', b'', b'b']]
    result_dense = self.evaluate(dense)
    self.assertAllEqual(expected_dense, result_dense)

  def testDenseSparseTensorMatMul(self):
    # [FUNCTION DOC]
    # Seeds numpy, builds a random 3 by 3 dense matrix and a 3 by 3 sparse matrix with two
    # entries, then multiplies dense by sparse for all four combinations of adjoint_a and
    # adjoint_b. Each result is asserted equal to a regular matmul of the dense matrix with the
    # densified (and, where requested, transposed) operands.

    np.random.seed(42)
    dense_numpy_array = np.random.rand(3, 3)
    independent_dense_tf = constant_op.constant(
        dense_numpy_array, dtype='float32')

    sp = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[4., 8.], dense_shape=[3, 3])
    dense_of_sparse = sparse_ops.sparse_to_dense(sp.indices, sp.shape,
                                                 sp.values)

    result = sparse_ops.sparse_tensor_dense_matmul(
        independent_dense_tf, sp, adjoint_a=False, adjoint_b=False)
    expected = math_ops.matmul(independent_dense_tf, dense_of_sparse)
    self.assertAllEqual(expected, result)

    result = sparse_ops.sparse_tensor_dense_matmul(
        independent_dense_tf, sp, adjoint_a=False, adjoint_b=True)
    expected = math_ops.matmul(independent_dense_tf,
                               array_ops.transpose(dense_of_sparse))
    self.assertAllEqual(expected, result)

    result = sparse_ops.sparse_tensor_dense_matmul(
        independent_dense_tf, sp, adjoint_a=True, adjoint_b=False)
    expected = math_ops.matmul(
        array_ops.transpose(independent_dense_tf), dense_of_sparse)
    self.assertAllEqual(expected, result)

    result = sparse_ops.sparse_tensor_dense_matmul(
        independent_dense_tf, sp, adjoint_a=True, adjoint_b=True)
    expected = math_ops.matmul(
        array_ops.transpose(independent_dense_tf),
        array_ops.transpose(dense_of_sparse))
    self.assertAllEqual(expected, result)

  def testMapValues(self):
    # [FUNCTION DOC]
    # Tests map_values: passing no sparse tensor raises a value error; a single sparse argument
    # with abs gives the expected sparse result; two sparse arguments plus a keyword argument are
    # passed through correctly; sparse inputs with mismatched indices or mismatched dense shapes
    # raise an invalid-argument or value error (at construction or evaluation).
    # supplying no sparse tensor should result in ValueError
    with self.assertRaises(ValueError):
      sparse_ops.map_values(math_ops.abs, 0.0)

    sp = sparse_ops.from_dense([[0.0, 1.0, 0.0], [-2.0, 1.0, 0.0]])

    # helper function to check equality of sparse tensor
    def assert_sparse_equal(expected, result):
      # [FUNCTION DOC]
      # HELPER: asserts two sparse tensors have equal values, indices and dense shapes, with a
      # distinct failure message for each component. Returns nothing.
      self.assertAllEqual(expected.values, result.values, msg='Values differ')
      self.assertAllEqual(
          expected.indices, result.indices, msg='Indices differ')
      self.assertAllEqual(
          expected.dense_shape, result.dense_shape, msg='Shapes differ')

    # check for a single sparse argument
    expected = sparse_ops.from_dense([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    result = sparse_ops.map_values(math_ops.abs, sp)
    assert_sparse_equal(expected, result)

    # check correct passing of keyword argument, and handling of two sparse
    # arguments at the same time
    def mapping(arg1, arg2, kwarg):
      # [FUNCTION DOC]
      # HELPER: mapping function for testMapValues. Asserts the keyword argument arrived with the
      # expected string value and returns the sum of the two value arguments.
      self.assertEqual(kwarg, 'kwarg')
      return arg1 + arg2

    result = sparse_ops.map_values(mapping, sp, sp, kwarg='kwarg')
    expected = sparse_ops.from_dense([[0.0, 2.0, 0.0], [-4.0, 2.0, 0.0]])
    assert_sparse_equal(expected, result)

    # check that index mismatches are correctly detected even if the `value`s
    # have compatible shape
    sp_incomp = sparse_ops.from_dense([[0.0, 1.0, 0.0], [-2.0, 0.0, 1.0]])
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      result = sparse_ops.map_values(mapping, sp, sp_incomp, kwarg='kwarg')
      self.evaluate(result)

    # check that shape mismatches are correctly detected
    sp_incomp = sparse_tensor.SparseTensor(sp.indices, sp.values, (25, 25))
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      result = sparse_ops.map_values(mapping, sp, sp_incomp, kwarg='kwarg')
      self.evaluate(result)

  def testConstantStringToSparse(self):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: converting a constant string vector of five characters
    # with from_dense yields indices 0 to 4, the same string values and dense shape of five.
    # Test case for GitHub issue 40633.
    tensor = constant_op.constant(list('ababa'))
    sparse = sparse_ops.from_dense(tensor)
    result = self.evaluate(sparse)
    self.assertAllEqual([[0], [1], [2], [3], [4]], result.indices)
    self.assertAllEqual([b'a', b'b', b'a', b'b', b'a'], result.values)
    self.assertAllEqual([5], result.dense_shape)

  def testSparseTensorToDenseQint(self):
    # [FUNCTION DOC]
    # For each quantized dtype (qint8, qint16, quint8, quint16), builds a 2 by 3 sparse tensor of
    # that dtype, densifies it and asserts the result equals the expected dense array after both
    # are cast to the matching numpy dtype.
    x = np.asarray([1, 2])
    y = np.asarray([[1, 0, 0], [0, 0, 2]])
    for dtype in [dtypes.qint8, dtypes.qint16, dtypes.quint8, dtypes.quint16]:
      sp = sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]],
          values=x.astype(dtype.as_numpy_dtype),
          dense_shape=[2, 3])
      v = self.evaluate(sparse_ops.sparse_tensor_to_dense(sp))
      self.assertAllEqual(
          y.astype(dtype.as_numpy_dtype), v.astype(dtype.as_numpy_dtype))


@test_util.run_all_in_graph_and_eager_modes
class RawOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for raw generated sparse kernels: the gradient of SparseFillEmptyRowsGrad
  # (valid and invalid inputs) and static shape inference of sparse concat. Runs in both graph
  # and eager modes through the class decorator. Holds no shared fixtures.

  def testSparseFillEmptyRowsGrad(self):
    # [FUNCTION DOC]
    # Runs the raw fill-empty-rows gradient op with a reverse index map of two entries and four
    # gradient values; asserts the value gradient is the gathered values (2, 1) and the default
    # value gradient is the sum of the unused values (3).
    reverse_index_map = [2, 1]
    grad_values = [0, 1, 2, 3]
    d_values, d_default_value = self.evaluate(
        gen_sparse_ops.SparseFillEmptyRowsGrad(
            reverse_index_map=reverse_index_map, grad_values=grad_values))
    self.assertAllEqual([2, 1], d_values)
    self.assertEqual(3, d_default_value)

  def testSparseFillEmptyRowsGradNegativeIndexMapValue(self):
    # [FUNCTION DOC]
    # Passes a reverse index map containing a negative entry and asserts an invalid-argument
    # error saying reverse index elements must lie in the range zero to four (exclusive).
    reverse_index_map = [2, -1]
    grad_values = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r'Elements in reverse index must be in \[0, 4\)'):
      self.evaluate(
          gen_sparse_ops.SparseFillEmptyRowsGrad(
              reverse_index_map=reverse_index_map, grad_values=grad_values))

  def testSparseFillEmptyRowsGradLargeIndexMapValue(self):
    # [FUNCTION DOC]
    # Passes a reverse index map containing an entry beyond the number of gradient values and
    # asserts the same invalid-argument error about the allowed index range.
    reverse_index_map = [2, 10]
    grad_values = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r'Elements in reverse index must be in \[0, 4\)'):
      self.evaluate(
          gen_sparse_ops.SparseFillEmptyRowsGrad(
              reverse_index_map=reverse_index_map, grad_values=grad_values))

  def testSparseFillEmptyRowsGradMatrix(self):
    # [FUNCTION DOC]
    # Passes a rank-2 gradient values input where a vector is required. In eager mode asserts an
    # invalid-argument error saying grad_values must be a vector; in graph mode asserts a value
    # error from shape checking saying the shape must be rank 1 but is rank 2.
    reverse_index_map = [0, 1]
    grad_values = [[0, 1], [2, 3]]
    # Note: Eager mode and graph mode throw different errors here. Graph mode
    # will fail with a ValueError from the shape checking logic, while Eager
    # will fail with an InvalidArgumentError from the kernel itself.
    if context.executing_eagerly():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  r'grad_values must be a vector'):
        self.evaluate(
            gen_sparse_ops.SparseFillEmptyRowsGrad(
                reverse_index_map=reverse_index_map, grad_values=grad_values))
    else:
      with self.assertRaisesRegex(ValueError,
                                  r'Shape must be rank 1 but is rank 2'):
        self.evaluate(
            gen_sparse_ops.SparseFillEmptyRowsGrad(
                reverse_index_map=reverse_index_map, grad_values=grad_values))

  def testSparseConcatStaticShape(self):
    # [FUNCTION DOC]
    # Graph-only test (skipped in eager mode since sparse placeholders need graph context).
    # Concatenates sparse placeholders of shapes 2 by 1 and 2 by 2 along axis 1 and asserts the
    # static result shape is 2 by 3.
    if context.executing_eagerly():
      self.skipTest('sparse_spaceholder is only available in graph context.')
    input_a = array_ops.sparse_placeholder(dtypes.float32, shape=(2, 1))
    input_b = array_ops.sparse_placeholder(dtypes.float32, shape=(2, 2))

    result = sparse_ops.sparse_concat_v2(axis=1, sp_inputs=[input_a, input_b])
    self.assertEqual(result.shape, [2, 3])


if __name__ == '__main__':
  googletest.main()
