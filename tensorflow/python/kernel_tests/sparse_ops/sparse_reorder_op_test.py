# [MODULE DOC]
# Unit tests for the sparse reorder operation, which sorts the entries of a sparse tensor into
# canonical row-major order of their indices. The module defines one parameterized test class
# that covers: preservation of static shape information (eagerly and inside traced functions),
# already-ordered and shuffled inputs (with and without placeholders in graph mode), numerical
# gradients with respect to the values, a regression for dense-shape overflow, and rejection of
# inconsistent raw inputs. It only runs tests and has no import-time side effects other than
# importing the sparse gradient registrations.
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
"""Tests for SparseReorder."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SparseReorderTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the sparse reorder op. It uses a fixed 5x6 sparse fixture with six
  # entries (built by a helper that can permute the entries and choose the value dtype) and
  # compares reordered results against the canonical ordering. Supports parameterized tests.

  def _SparseTensorPlaceholder(self):
    # [FUNCTION DOC]
    # HELPER: builds graph-mode placeholder inputs for the feed-based reorder tests.
    # Takes no arguments and returns a sparse tensor whose indices (64-bit integer), values
    # (64-bit float) and dense shape (64-bit integer) are all unconstrained placeholders, to be
    # fed with concrete sparse values when running a session. Only meaningful in graph mode.
    return sparse_tensor.SparseTensor(
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.float64),
        array_ops.placeholder(dtypes.int64))

  def _SparseTensorValue_5x6(self, permutation, dtype=dtypes.float64):
    # [FUNCTION DOC]
    # HELPER: creates the shared 5x6 sparse test fixture as a concrete (numpy-backed) value.
    # The canonical data has six entries in sorted index order with values 0, 10, 13, 14, 32, 33.
    # Parameters: permutation is an integer array of positions used to reorder both the indices
    # and the values consistently (identity gives canonical order); dtype is the TensorFlow
    # dtype of the values, defaulting to 64-bit float. Returns a sparse tensor value with
    # 64-bit indices, values converted to the numpy equivalent of dtype, and dense shape [5, 6].
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2],
                    [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(dtype.as_numpy_dtype)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([5, 6]).astype(np.int64)
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def testStaticShapeInfoPreserved(self):
    # [FUNCTION DOC]
    # Test: wraps the in-order fixture as a sparse tensor, checks its static shape is 5 by 6,
    # applies sparse reorder, and asserts the output's static shape is still 5 by 6.
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_5x6(np.arange(6)))
    self.assertAllEqual((5, 6), sp_input.get_shape())
    sp_output = sparse_ops.sparse_reorder(sp_input)
    self.assertAllEqual((5, 6), sp_output.get_shape())

  @parameterized.named_parameters([
      ("FullyStatic", [5, 6], None, [5, 6]),
      ("Partial1", [None, 6], None, [None, 6]),
      ("Partial2", [5, None], None, [5, None]),
      ("RankFromStaticDenseShape", [None, None], None, [None, None]),
      ("RankFromDenseShapeDim", None, [2], [None, None]),
      ("FullyDynamic", None, [None], None),
  ])
  def testStaticShapeInfoPreservedInFunction(self,
                                             static_dense_shape,
                                             dense_shape_shape,
                                             expected_output_shape):
    # [FUNCTION DOC]
    # Parameterized test: checks how much static shape information sparse reorder keeps when
    # traced inside a function. Parameters (from the named cases): static_dense_shape is a list
    # of known dimensions with unknown ones as null, or null when nothing is known;
    # dense_shape_shape is the static shape given to the dense-shape input spec (decides whether
    # the rank is known); expected_output_shape is the static shape the output must report.
    # It traces the function with concrete input specs and the assertion runs during tracing.

    @def_function.function
    def func(indices, values, dense_shape):
      # [FUNCTION DOC]
      # Traced body used by the enclosing test. Takes indices, values and a dense-shape tensor.
      # If static dimensions were supplied, it replaces dense-shape entries with those static
      # values where known (falling back to the dynamic entry otherwise), builds a sparse tensor,
      # reorders it, asserts its static shape equals the expected shape from the enclosing
      # test, and returns the reordered sparse tensor.
      if static_dense_shape:
        dense_shape = [
            static_dim or dense_shape[i]
            for i, static_dim in enumerate(static_dense_shape)
        ]
      sp_input = sparse_tensor.SparseTensor(indices, values, dense_shape)
      sp_output = sparse_ops.sparse_reorder(sp_input)
      self.assertEqual(expected_output_shape, sp_output.get_shape())
      return sp_output

    _ = func.get_concrete_function(
        tensor_spec.TensorSpec([6, 2], dtypes.int64),
        tensor_spec.TensorSpec([6], dtypes.float64),
        tensor_spec.TensorSpec(dense_shape_shape, dtypes.int64))

  def testAlreadyInOrder(self):
    # [FUNCTION DOC]
    # Test: reorders an already canonical sparse value and asserts indices, values and dense
    # shape are all unchanged.
    with self.session() as sess:
      input_val = self._SparseTensorValue_5x6(np.arange(6))
      sp_output = sparse_ops.sparse_reorder(input_val)

      output_val = self.evaluate(sp_output)
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @test_util.run_deprecated_v1
  def testFeedAlreadyInOrder(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): feeds an already ordered sparse value through
    # placeholders into sparse reorder and asserts the fetched indices, values and dense shape
    # equal the input.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6(np.arange(6))
      sp_output = sparse_ops.sparse_reorder(sp_input)

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @parameterized.parameters(dtypes.bfloat16, dtypes.float64)
  def testOutOfOrder(self, dtype):
    # [FUNCTION DOC]
    # Parameterized test over bfloat16 and 64-bit float values. Five times, shuffles the fixture
    # entries with a random permutation, reorders, and asserts indices, values and dense shape
    # match the canonical fixture. Parameter dtype is the value dtype under test.
    expected_output_val = self._SparseTensorValue_5x6(np.arange(6), dtype)
    with self.session() as sess:
      for _ in range(5):  # To test various random permutations
        input_val = self._SparseTensorValue_5x6(np.random.permutation(6), dtype)
        sp_output = sparse_ops.sparse_reorder(input_val)

        output_val = self.evaluate(sp_output)
        self.assertAllEqual(output_val.indices, expected_output_val.indices)
        self.assertAllEqual(output_val.values, expected_output_val.values)
        self.assertAllEqual(output_val.dense_shape,
                            expected_output_val.dense_shape)

  @test_util.run_deprecated_v1
  def testFeedOutOfOrder(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): five times, creates placeholder inputs, feeds a
    # randomly permuted fixture, and asserts the reordered result equals the canonical fixture
    # in indices, values and dense shape.
    expected_output_val = self._SparseTensorValue_5x6(np.arange(6))
    with self.session() as sess:
      for _ in range(5):  # To test various random permutations
        sp_input = self._SparseTensorPlaceholder()
        input_val = self._SparseTensorValue_5x6(np.random.permutation(6))
        sp_output = sparse_ops.sparse_reorder(sp_input)

        output_val = sess.run(sp_output, {sp_input: input_val})
        self.assertAllEqual(output_val.indices, expected_output_val.indices)
        self.assertAllEqual(output_val.values, expected_output_val.values)
        self.assertAllEqual(output_val.dense_shape,
                            expected_output_val.dense_shape)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 only): for five random permutations of the fixture, reorders
    # the sparse tensor and uses the numerical gradient checker to compare analytic and
    # numeric gradients of output values with respect to input values; asserts the error is
    # below 1e-11.
    with self.session():
      for _ in range(5):  # To test various random permutations
        input_val = self._SparseTensorValue_5x6(np.random.permutation(6))
        sp_input = sparse_tensor.SparseTensor(input_val.indices,
                                              input_val.values,
                                              input_val.dense_shape)
        sp_output = sparse_ops.sparse_reorder(sp_input)

        err = gradient_checker.compute_gradient_error(
            sp_input.values,
            input_val.values.shape,
            sp_output.values,
            input_val.values.shape,
            x_init_value=input_val.values)
        self.assertLess(err, 1e-11)

  def testShapeOverflow(self):
    # [FUNCTION DOC]
    # Regression test for a reported overflow issue: builds a six-dimensional sparse tensor with
    # each dimension 4096 (element count exceeds 64-bit range if multiplied) and one entry, and
    # asserts the static shape is reported correctly both before and after sparse reorder.
    # Test case for GitHub issue 45392
    sp_input = sparse_tensor.SparseTensor(
        indices=[[0, 0, 0, 0, 0, 0]],
        values=[0.0],
        dense_shape=[4096, 4096, 4096, 4096, 4096, 4096])
    self.assertAllEqual((4096, 4096, 4096, 4096, 4096, 4096),
                        sp_input.get_shape())
    sp_output = sparse_ops.sparse_reorder(sp_input)
    self.assertAllEqual((4096, 4096, 4096, 4096, 4096, 4096),
                        sp_output.get_shape())

  def testInvalidSparseInput(self):
    # [FUNCTION DOC]
    # Test: calls the raw reorder kernel with one index row of rank three, three values, and a
    # dense shape of rank two, so the element counts disagree. Asserts evaluation raises either
    # a value error or an invalid-argument error whose message says the number of elements do
    # not match.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Number of elements .* do not match",
    ):
      self.evaluate(
          gen_sparse_ops.sparse_reorder(
              input_indices=[[0, 0, 0]],
              input_values=[0, 1, 2],
              input_shape=[3, 3],
          )
      )


if __name__ == "__main__":
  test.main()
