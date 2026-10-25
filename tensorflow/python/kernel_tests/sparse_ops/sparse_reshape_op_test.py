# [MODULE DOC]
# Unit tests for the sparse reshape operation, which changes the dense shape of a sparse tensor
# and remaps its indices without changing the values. Two test classes: one covering static
# shape propagation, inferred (negative one) dimensions, error cases (several inferred
# dimensions, mismatched element counts, overflow, negative sizes), index remapping when the
# rank grows, shrinks or stays the same, partial shape inference with placeholders, and a
# randomized comparison with dense reshape; and one covering empty (zero-sized) tensors, comparing
# sparse behavior with dense reshape. Many tests are graph-mode only (deprecated v1 decorator).
# No import-time side effects.
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
"""Tests for SparseReshape."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SparseReshapeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the sparse reshape op on non-empty inputs. Provides two fixtures: a 5x6
  # sparse value with six entries and a 2x3x4 sparse value with seven entries, plus a helper
  # that builds placeholder sparse tensors for feed-based graph tests.

  def _SparseTensorPlaceholder(self):
    # [FUNCTION DOC]
    # HELPER: builds graph-mode placeholder inputs for the feed-based reshape tests.
    # Takes no arguments; returns a sparse tensor whose indices, values and dense shape are
    # placeholders of 64-bit integer, 64-bit float and 64-bit integer type respectively.
    return sparse_tensor.SparseTensor(
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.float64),
        array_ops.placeholder(dtypes.int64))

  def _SparseTensorValue_5x6(self):
    # [FUNCTION DOC]
    # HELPER: returns the shared 5x6 fixture as a numpy-backed sparse tensor value. No inputs.
    # It holds six entries at rows 0, 1, 1, 1, 3, 3 with values 0, 10, 13, 14, 32, 33 (64-bit
    # float), 64-bit integer indices, and dense shape [5, 6].
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2],
                    [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.float64)
    shape = np.array([5, 6]).astype(np.int64)
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_2x3x4(self):
    # [FUNCTION DOC]
    # HELPER: returns the shared 2x3x4 fixture as a numpy-backed sparse tensor value. No inputs.
    # It holds seven entries in row-major order with values 1, 10, 12, 103, 111, 113, 122 and
    # dense shape [2, 3, 4]. Unlike the 5x6 fixture, arrays use default integer dtypes
    # (not explicitly converted to 64-bit) for indices, values and shape.
    ind = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 2], [1, 0, 3], [1, 1, 1],
                    [1, 1, 3], [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 111, 113, 122])
    shape = np.array([2, 3, 4])
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def testStaticShapeInfoPreserved(self):
    # [FUNCTION DOC]
    # Test: wraps the 5x6 fixture as a sparse tensor (static shape 5 by 6), reshapes it to
    # 1 by 5 by 2 by 3, and asserts the output's static shape is that new shape.
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_5x6())
    self.assertAllEqual((5, 6), sp_input.get_shape())
    sp_output = sparse_ops.sparse_reshape(sp_input, shape=(1, 5, 2, 3))
    self.assertAllEqual((1, 5, 2, 3), sp_output.get_shape())

  def testStaticShapeInfoPreservedWithInferredDims(self):
    # [FUNCTION DOC]
    # Test: reshapes the 2x3x4 fixture to the target shape of 2 and an inferred dimension, and
    # asserts the static output shape resolves the inferred dimension to 12 (3 times 4).
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_2x3x4())
    self.assertAllEqual((2, 3, 4), sp_input.get_shape())
    sp_output = sparse_ops.sparse_reshape(sp_input, shape=(2, -1))
    self.assertAllEqual((2, 3 * 4), sp_output.get_shape())

  @test_util.run_deprecated_v1
  def testRaisesIfMoreThanOneInferredDim(self):
    # [FUNCTION DOC]
    # Graph-mode test: asks to reshape the 2x3x4 fixture to a target with two inferred
    # (negative one) dimensions and asserts a value error whose message says at most one
    # dimension can be inferred, raised at graph construction time.
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_2x3x4())
    with self.assertRaisesRegex(ValueError, "At most one dimension can"):
      sparse_ops.sparse_reshape(sp_input, shape=(-1, 2, -1))

  @test_util.run_deprecated_v1
  def testRaisesIfInferredShapeNotPossible(self):
    # [FUNCTION DOC]
    # Graph-mode test: reshapes the 24-element 2x3x4 fixture to a target of an inferred
    # dimension and 7, which does not divide 24, and asserts a value error mentioning that the
    # tensor cannot be reshaped.
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_2x3x4())
    with self.assertRaisesRegex(ValueError, "Cannot reshape"):
      sparse_ops.sparse_reshape(sp_input, shape=(-1, 7))

  @test_util.run_deprecated_v1
  def testPropagatesFullyKnownDenseShapeWhenShapePartiallyKnown(self):
    # [FUNCTION DOC]
    # Graph-mode test: the target shape is built at graph time by concatenating a constant 2
    # with a one-element placeholder (so only partly known). Since the input shape 2x3x4 is
    # fully known, asserts the output static shape is nevertheless inferred as 2 by 12.
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_2x3x4())
    self.assertAllEqual((2, 3, 4), sp_input.shape)
    sp_output = sparse_ops.sparse_reshape(
        sp_input, shape=array_ops.concat(
            (constant_op.constant([2], dtype=dtypes.int64),
             array_ops.placeholder(dtype=dtypes.int64, shape=[1])),
            axis=0))
    self.assertAllEqual((2, 3 * 4), sp_output.shape)

  def testSameShape(self):
    # [FUNCTION DOC]
    # Test: reshapes the 5x6 fixture to the identical shape and asserts indices, values and dense
    # shape of the evaluated result equal the input.
    with self.session() as sess:
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(input_val, [5, 6])

      output_val = self.evaluate(sp_output)
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  def testReshapeIntegeroverflow(self):
    # [FUNCTION DOC]
    # Test: calls the raw reshape kernel with an input dense shape of 2 to the 32 by 2 to the 31
    # (element count overflows 64-bit signed range) and a target shape of 1 by 1, and asserts
    # evaluation raises an invalid-argument error.
    # KNOWN ISSUE: the assignment of the op result sits inside the assertion block, so the
    # error could in principle be raised at construction rather than evaluation; either passes.
    with self.session():
      with self.assertRaises(errors.InvalidArgumentError):
        sp_output = sparse_ops.gen_sparse_ops.sparse_reshape(
            input_indices=[[0, 0]],
            input_shape=[2**32, 2**31],
            new_shape=[1, 1],
            name=None)

        self.evaluate(sp_output)

  def testReshapeNegativeShape(self):
    # [FUNCTION DOC]
    # Test: calls the raw reshape kernel with an input dense shape containing a negative entry
    # (1 and negative 1) and target shape 1 by 1, and asserts an invalid-argument error is
    # raised (during construction or evaluation).
    with self.session():
      with self.assertRaises(errors.InvalidArgumentError):
        sp_output = sparse_ops.gen_sparse_ops.sparse_reshape(
            input_indices=[[0, 0]],
            input_shape=[1, -1],
            new_shape=[1, 1],
            name=None)

        self.evaluate(sp_output)

  @test_util.run_deprecated_v1
  def testFeedSameShape(self):
    # [FUNCTION DOC]
    # Graph-mode test: feeds the 5x6 fixture through placeholders, reshapes to the same 5 by 6
    # shape, and asserts the fetched indices, values and dense shape equal the input.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [5, 6])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @test_util.run_deprecated_v1
  def testWorksWellWithTfShape(self):
    # [FUNCTION DOC]
    # Graph-mode test: uses the shape operation on the sparse placeholder (which yields a 32-bit
    # integer result) as the target shape for sparse reshape, feeds the 5x6 fixture, and asserts
    # the output is identical to the input, showing mixed integer types are accepted.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      shape = array_ops.shape(sp_input)  # tf.shape generates int32 output
      sp_output = sparse_ops.sparse_reshape(sp_input, shape)

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @test_util.run_deprecated_v1
  def testFeedSameShapeWithInferredDim(self):
    # [FUNCTION DOC]
    # Graph-mode test: feeds the 5x6 fixture and reshapes to an inferred first dimension and 6;
    # asserts the inferred result is the same shape and the output equals the input.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [-1, 6])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @test_util.run_deprecated_v1
  def testFeedNewShapeSameRank(self):
    # [FUNCTION DOC]
    # Graph-mode test: feeds the 5x6 fixture and reshapes to 3 by 10 (same rank). Asserts the
    # flat positions are remapped to the expected six indices, values are unchanged, and the
    # dense shape is 3 by 10.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [3, 10])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0], [0, 6], [0, 9], [1, 0], [2, 0],
                                    [2, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [3, 10])

  @test_util.run_deprecated_v1
  def testFeedNewShapeSameRankWithInferredDim(self):
    # [FUNCTION DOC]
    # Graph-mode test: same as the same-rank reshape test but the target is 3 and an inferred
    # dimension; asserts the same remapped indices, unchanged values and a dense shape of 3 by 10.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [3, -1])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0], [0, 6], [0, 9], [1, 0], [2, 0],
                                    [2, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [3, 10])

  def testUpRank(self):
    # [FUNCTION DOC]
    # Test: reshapes the 5x6 fixture to the rank-three shape 2 by 3 by 5 and asserts the
    # remapped three-column indices, unchanged values and the new dense shape.
    with self.session() as sess:
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(input_val, [2, 3, 5])

      output_val = self.evaluate(sp_output)
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0],
                                    [1, 1, 0], [1, 1, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [2, 3, 5])

  @test_util.run_deprecated_v1
  def testFeedUpRank(self):
    # [FUNCTION DOC]
    # Graph-mode test: feeds the 5x6 fixture through placeholders and reshapes to 2 by 3 by 5;
    # asserts remapped indices, unchanged values and dense shape [2, 3, 5].
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [2, 3, 5])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0],
                                    [1, 1, 0], [1, 1, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [2, 3, 5])

  @test_util.run_deprecated_v1
  def testFeedUpRankWithInferredDim(self):
    # [FUNCTION DOC]
    # Graph-mode test: like the up-rank feed test but the middle target dimension is inferred;
    # asserts it resolves to 3 giving the same indices and dense shape [2, 3, 5].
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [2, -1, 5])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0],
                                    [1, 1, 0], [1, 1, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [2, 3, 5])

  @test_util.run_deprecated_v1
  def testFeedDownRank(self):
    # [FUNCTION DOC]
    # Graph-mode test: feeds the 2x3x4 fixture and reshapes to 6 by 4 (lower rank); asserts the
    # seven remapped two-column indices, unchanged values and dense shape [6, 4].
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_2x3x4()
      sp_output = sparse_ops.sparse_reshape(sp_input, [6, 4])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 1], [1, 0], [1, 2], [3, 3], [4, 1],
                                    [4, 3], [5, 2]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [6, 4])

  @test_util.run_deprecated_v1
  def testFeedDownRankWithInferredDim(self):
    # [FUNCTION DOC]
    # Graph-mode test: like the down-rank feed test but the second target dimension is inferred;
    # asserts it resolves to 4 with identical indices, values and dense shape [6, 4].
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_2x3x4()
      sp_output = sparse_ops.sparse_reshape(sp_input, [6, -1])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 1], [1, 0], [1, 2], [3, 3], [4, 1],
                                    [4, 3], [5, 2]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [6, 4])

  @test_util.run_deprecated_v1
  def testFeedMultipleInferredDims(self):
    # [FUNCTION DOC]
    # Graph-mode test: builds a reshape of the placeholder to a target with two inferred
    # dimensions (graph construction succeeds because the target is a constant list that is
    # not validated statically here), then asserts running it raises an op error saying only
    # one output dimension may be negative one.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [4, -1, -1])
      with self.assertRaisesOpError("only one output dimension may be -1"):
        sess.run(sp_output, {sp_input: input_val})

  @test_util.run_deprecated_v1
  def testProvideStaticallyMismatchedSizes(self):
    # [FUNCTION DOC]
    # Graph-mode test: the input has static shape 5 by 6 (30 elements) and the target is 4 by 7
    # (28 elements); asserts a value error mentioning that it cannot reshape is raised during
    # graph construction.
    input_val = self._SparseTensorValue_5x6()
    sp_input = sparse_tensor.SparseTensor.from_value(input_val)
    with self.assertRaisesRegex(ValueError, "Cannot reshape"):
      sparse_ops.sparse_reshape(sp_input, [4, 7])

  @test_util.run_deprecated_v1
  def testFeedMismatchedSizes(self):
    # [FUNCTION DOC]
    # Graph-mode test: with an unknown-shape placeholder, reshapes to 4 by 7 and feeds the 30
    # element fixture; asserts running raises an op error saying the input has 30 dense values.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [4, 7])
      with self.assertRaisesOpError(
          "Input to reshape is a tensor with 30 dense values"):
        sess.run(sp_output, {sp_input: input_val})

  @test_util.run_deprecated_v1
  def testFeedMismatchedSizesWithInferredDim(self):
    # [FUNCTION DOC]
    # Graph-mode test: reshapes the placeholder to 4 and an inferred dimension, feeding 30
    # elements (not divisible by 4); asserts running raises an op error saying the requested
    # shape requires a multiple.
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [4, -1])
      with self.assertRaisesOpError("requested shape requires a multiple"):
        sess.run(sp_output, {sp_input: input_val})

  @test_util.run_deprecated_v1
  def testFeedPartialShapes(self):
    # [FUNCTION DOC]
    # Graph-mode test of static shape inference with partly unknown inputs. Case one: with
    # unconstrained placeholders and a known target of rank three, the output indices have
    # shape unknown by 3 and the dense shape has shape 3. Case two: when the input indices are
    # known to have 5 rows, the output indices have shape 5 by 3. Case three: when the target
    # shape is an unconstrained placeholder, the output indices have shape 5 by unknown and the
    # dense shape has unknown shape.
    with self.session():
      # Incorporate new rank into shape information if known
      sp_input = self._SparseTensorPlaceholder()
      sp_output = sparse_ops.sparse_reshape(sp_input, [2, 3, 5])
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [None, 3])
      self.assertListEqual(sp_output.dense_shape.get_shape().as_list(), [3])

      # Incorporate known shape information about input indices in output
      # indices
      sp_input = self._SparseTensorPlaceholder()
      sp_input.indices.set_shape([5, None])
      sp_output = sparse_ops.sparse_reshape(sp_input, [2, 3, 5])
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [5, 3])
      self.assertListEqual(sp_output.dense_shape.get_shape().as_list(), [3])

      # Even if new_shape has no shape information, we know the ranks of
      # output indices and shape
      sp_input = self._SparseTensorPlaceholder()
      sp_input.indices.set_shape([5, None])
      new_shape = array_ops.placeholder(dtypes.int64)
      sp_output = sparse_ops.sparse_reshape(sp_input, new_shape)
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [5, None])
      self.assertListEqual(sp_output.dense_shape.get_shape().as_list(), [None])

  @test_util.run_deprecated_v1
  def testFeedDenseReshapeSemantics(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: an inline comment says the shapes are rank 5, but the code draws the original
    # and new ranks randomly between 2 and 6 (upper bound exclusive 7), so ranks vary.
    # Randomized graph-mode test. Distributes twelve prime factors (four 2s, four 3s, four 5s,
    # 810000 elements in total) among a random number of dimensions to form random original and
    # new shapes. Builds a random dense array, keeps entries below 0.5 as the sparse input,
    # computes the expected sparse form from a dense reshape, feeds the input and asserts the
    # sparse reshape output's indices, values and dense shape equal the expected ones.
    with self.session() as sess:
      # Compute a random rank-5 initial shape and new shape, randomly sparsify
      # it, and check that the output of SparseReshape has the same semantics
      # as a dense reshape.
      factors = np.array([2] * 4 + [3] * 4 + [5] * 4)  # 810k total elements
      orig_rank = np.random.randint(2, 7)
      orig_map = np.random.randint(orig_rank, size=factors.shape)
      orig_shape = [np.prod(factors[orig_map == d]) for d in range(orig_rank)]
      new_rank = np.random.randint(2, 7)
      new_map = np.random.randint(new_rank, size=factors.shape)
      new_shape = [np.prod(factors[new_map == d]) for d in range(new_rank)]

      orig_dense = np.random.uniform(size=orig_shape)
      orig_indices = np.transpose(np.nonzero(orig_dense < 0.5))
      orig_values = orig_dense[orig_dense < 0.5]

      new_dense = np.reshape(orig_dense, new_shape)
      new_indices = np.transpose(np.nonzero(new_dense < 0.5))
      new_values = new_dense[new_dense < 0.5]

      sp_input = self._SparseTensorPlaceholder()
      input_val = sparse_tensor.SparseTensorValue(orig_indices, orig_values,
                                                  orig_shape)
      sp_output = sparse_ops.sparse_reshape(sp_input, new_shape)

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, new_indices)
      self.assertAllEqual(output_val.values, new_values)
      self.assertAllEqual(output_val.dense_shape, new_shape)


class EmptySparseTensorReshapeTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for reshaping zero-sized tensors, run for both a dense and a sparse
  # variant so the sparse behavior (output dense shape) can be compared with dense reshape,
  # including inference of a negative-one dimension when another dimension is zero.
  """Tests for reshaping 0-sized SparseTensors, compared w/ dense tensors."""

  def _MakeAndReshapeTensor(self, tensor_class, original_shape, target_shape):
    # [FUNCTION DOC]
    # HELPER: builds an empty tensor of the requested kind, reshapes it and returns the
    # resulting shape. Parameters: tensor_class is the text "sparse" or anything else for dense;
    # original_shape is a list of the empty tensor's dimensions; target_shape is the requested
    # shape, possibly with an inferred dimension. For sparse, builds a zero-entry sparse value,
    # runs sparse reshape and returns its dense shape array; for dense, reshapes a zeros
    # tensor and returns the result's shape. Errors from the kernel propagate to the caller.
    if tensor_class == "sparse":
      ind = np.zeros([0, len(original_shape)]).astype(np.int64)
      val = np.array([]).astype(np.float64)
      shape = np.array(original_shape).astype(np.int64)
      sp_input = sparse_tensor.SparseTensorValue(ind, val, shape)
      sp_output = self.evaluate(
          sparse_ops.sparse_reshape(sp_input, target_shape))
      return sp_output.dense_shape
    else:
      dense_input = array_ops.zeros(original_shape)
      dense_output = self.evaluate(array_ops.reshape(dense_input, target_shape))
      return dense_output.shape

  @parameterized.named_parameters([
      ("Dense", "dense"),
      ("Sparse", "sparse"),
  ])
  def testImpliedReshapeEmpty1DTensor(self, tensor_class):
    # [FUNCTION DOC]
    # Parameterized test (dense and sparse): a one-dimensional empty tensor of length zero is
    # reshaped with an inferred dimension to 0 by 1 and to 0 by 1 by 2; asserts the resulting
    # shapes. Parameter tensor_class selects the tensor kind.
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [0], [-1, 1]), [0, 1])
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [0], [-1, 1, 2]), [0, 1, 2])

  @parameterized.named_parameters([
      ("Dense", "dense"),
      ("Sparse", "sparse"),
  ])
  def testImpliedReshapeEmpty2DTensor(self, tensor_class):
    # [FUNCTION DOC]
    # Parameterized test (dense and sparse): an empty 1 by 0 tensor is reshaped with an inferred
    # leading dimension to 0 by 1 and to 0 by 2 by 3; asserts the resulting shapes.
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [1, 0], [-1, 1]), [0, 1])
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [1, 0], [-1, 2, 3]), [0, 2, 3])

  @parameterized.named_parameters([
      ("Dense", "dense"),
      ("Sparse", "sparse"),
  ])
  def testImpliedReshapeEmpty3DTensor(self, tensor_class):
    # [FUNCTION DOC]
    # Parameterized test (dense and sparse): an empty 1 by 0 by 0 tensor is reshaped to an
    # inferred dimension, 2 and 3; asserts the result shape is 0 by 2 by 3.
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [1, 0, 0], [-1, 2, 3]),
        [0, 2, 3])

  @parameterized.named_parameters([
      ("Dense", "dense"),
      ("Sparse", "sparse"),
  ])
  def testImpliedReshapeEmpty4DTensor(self, tensor_class):
    # [FUNCTION DOC]
    # Parameterized test (dense and sparse): an empty 2 by 4 by 0 by 6 tensor is reshaped to an
    # inferred dimension, 4, 6 and 2; asserts the result shape is 0 by 4 by 6 by 2.
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [2, 4, 0, 6], [-1, 4, 6, 2]),
        [0, 4, 6, 2])

  def testImpliedDimTogetherWithZeroDimCausesError(self):
    # [FUNCTION DOC]
    # Test: for four empty sparse inputs of different ranks, reshaping to a target that has both
    # an inferred dimension and a zero dimension must raise an invalid-argument error, because
    # the inferred size is ambiguous. The comment inside notes dense reshape behaves differently,
    # which is why only the sparse variant is exercised.
    # NOTE: When implied dimensions and zero dimensions coexist in the target
    # shape, the behavior currently differs between sparse and regular tensors.
    with self.assertRaises(errors.InvalidArgumentError):
      self._MakeAndReshapeTensor("sparse", [0], [-1, 0])
    with self.assertRaises(errors.InvalidArgumentError):
      self._MakeAndReshapeTensor("sparse", [1, 0], [-1, 0])
    with self.assertRaises(errors.InvalidArgumentError):
      self._MakeAndReshapeTensor("sparse", [1, 2, 0], [2, -1, 0])
    with self.assertRaises(errors.InvalidArgumentError):
      self._MakeAndReshapeTensor("sparse", [1, 2, 3, 0], [2, 0, -1, 3])


if __name__ == "__main__":
  test.main()
