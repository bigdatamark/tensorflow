# [MODULE DOC]
# Tests for creating a dataset from the rows (slices) of a sparse tensor. Covers iteration over
# slices in order (graph mode), rejection of reverse-ordered indices, empty sparse tensors,
# invalid sparse inputs, the error on non-sparse input in eager mode, and checkpoint/restore of the
# iterator. No import-time side effects; runs the test main entry point when executed directly.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.from_sparse_tensor_slices()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class FromSparseTensorSlicesTest(test_base.DatasetTestBase,
                                 parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the sparse-tensor-slices dataset constructor. Uses the shared dataset test base
  # class and mostly graph-mode (version 1) combinations because it feeds placeholders.

  @combinations.generate(
      combinations.times(
          combinations.combine(tf_api_version=1, mode=["graph"]),
          combinations.combine(slices=[[
              [1., 2., 3.], [1.], [1.], [1., 2.], [], [1., 2.], [], [], []
          ], [[1., 2.], [], [1., 2.], [1.], [1., 2.], [], [1., 2.]]])))
  def testFromSparseTensorSlices(self, slices):
    # [FUNCTION DOC]
    # Graph-mode test. Parameter slices is a list of rows, each a list of floats (possibly empty).
    # Builds a sparse tensor placeholder, a dataset of its slices and an initializable iterator.
    # Feeds a sparse tensor whose indices and values are derived from the slices in row-major
    # order, then reads each element and asserts its values, indices (a single column of
    # positions) and dense shape (the feed shape without its first dimension) match the row.
    # Finally asserts that one more read raises an out-of-range error.
    """Test a dataset based on slices of a `tf.sparse.SparseTensor`."""
    st = array_ops.sparse_placeholder(dtypes.float64)
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_sparse_tensor_slices(st))
    init_op = iterator.initializer
    get_next = sparse_tensor.SparseTensor(*iterator.get_next())

    with self.cached_session() as sess:
      # Test with sparse tensor in the appropriate order.
      # pylint: disable=g-complex-comprehension
      indices = np.array(
          [[i, j] for i in range(len(slices)) for j in range(len(slices[i]))])
      values = np.array([val for s in slices for val in s])
      # pylint: enable=g-complex-comprehension
      dense_shape = np.array([len(slices), max(len(s) for s in slices) + 1])
      sparse_feed = sparse_tensor.SparseTensorValue(indices, values,
                                                    dense_shape)
      sess.run(init_op, feed_dict={st: sparse_feed})
      for i, s in enumerate(slices):
        results = sess.run(get_next)
        self.assertAllEqual(s, results.values)
        expected_indices = np.array(
            [[j] for j in range(len(slices[i]))]).reshape([-1, 1])
        self.assertAllEqual(expected_indices, results.indices)
        self.assertAllEqual(dense_shape[1:], results.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @combinations.generate(
      combinations.times(
          combinations.combine(tf_api_version=1, mode=["graph"]),
          combinations.combine(slices=[[
              [1., 2., 3.], [1.], [1.], [1., 2.], [], [1., 2.], [], [], []
          ], [[1., 2.], [], [1., 2.], [1.], [1., 2.], [], [1., 2.]]])))
  def testFromSparseTensorSlicesInReverse(self, slices):
    # [FUNCTION DOC]
    # Graph-mode test. Parameter slices is a list of rows of floats. Feeds a sparse tensor whose
    # indices and values are in reverse (non-canonical) order and asserts that initializing the
    # iterator raises an unimplemented error, since reverse ordering is not supported.
    """Test a dataset based on slices of a `tf.sparse.SparseTensor` in reverse order."""
    st = array_ops.sparse_placeholder(dtypes.float64)
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_sparse_tensor_slices(st))
    init_op = iterator.initializer

    with self.cached_session() as sess:
      # pylint: disable=g-complex-comprehension
      indices = np.array(
          [[i, j] for i in range(len(slices)) for j in range(len(slices[i]))])
      values = np.array([val for s in slices for val in s])
      # pylint: enable=g-complex-comprehension
      dense_shape = np.array([len(slices), max(len(s) for s in slices) + 1])
      # Test with sparse tensor in the reverse order, which is not
      # currently supported.
      reverse_order_indices = indices[::-1, :]
      reverse_order_values = values[::-1]
      sparse_feed = sparse_tensor.SparseTensorValue(
          reverse_order_indices, reverse_order_values, dense_shape)
      with self.assertRaises(errors.UnimplementedError):
        sess.run(init_op, feed_dict={st: sparse_feed})

  @combinations.generate(combinations.combine(tf_api_version=1, mode=["graph"]))
  def testEmptySparseTensorSlices(self):
    # [FUNCTION DOC]
    # Graph-mode test. Feeds an empty sparse tensor (no indices or values, dense shape with a
    # leading zero dimension) and asserts the first read from the iterator raises an out-of-range
    # error, i.e. the dataset is empty.
    """Test a dataset based on slices of an empty `tf.sparse.SparseTensor`."""
    st = array_ops.sparse_placeholder(dtypes.float64)
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_sparse_tensor_slices(st))
    init_op = iterator.initializer
    get_next = sparse_tensor.SparseTensor(*iterator.get_next())

    with self.cached_session() as sess:
      # Test with an empty sparse tensor.
      empty_indices = np.empty((0, 4), dtype=np.int64)
      empty_values = np.empty((0,), dtype=np.float64)
      empty_dense_shape = [0, 4, 37, 9]
      sparse_feed = sparse_tensor.SparseTensorValue(empty_indices, empty_values,
                                                    empty_dense_shape)
      sess.run(init_op, feed_dict={st: sparse_feed})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @combinations.generate(combinations.combine(tf_api_version=1, mode=["graph"]))
  def testEmptySparseTensorSlicesInvalid(self):
    # [FUNCTION DOC]
    # Graph-mode test. Feeds a sparse tensor with no indices but four values and a zero-sized
    # dense shape, and asserts that initializing the iterator raises an invalid-argument error.
    """Test a dataset based on invalid `tf.sparse.SparseTensor`."""
    st = array_ops.sparse_placeholder(dtypes.float64)
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_sparse_tensor_slices(st))
    init_op = iterator.initializer

    with self.cached_session() as sess:
      # Test with an empty sparse tensor but with non empty values.
      empty_indices = np.empty((0, 4), dtype=np.int64)
      non_empty_values = [1, 2, 3, 4]
      empty_dense_shape = [0, 4, 37, 9]
      sparse_feed = sparse_tensor.SparseTensorValue(empty_indices,
                                                    non_empty_values,
                                                    empty_dense_shape)
      # Here, we expect the test to fail when running the feed.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(init_op, feed_dict={st: sparse_feed})

  @combinations.generate(combinations.combine(tf_api_version=1, mode=["graph"]))
  def testEmptySparseTensorSlicesInvalid2(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment says "non empty values" but the fed values are empty; the
    # invalid part is the malformed (single empty row) indices with dense shape of one by one.
    # Graph-mode test. Feeds a sparse tensor with a malformed empty index row and no values, and
    # asserts that initializing the iterator raises an invalid-argument error.
    """Test a dataset based on invalid `tf.sparse.SparseTensor`."""
    st = array_ops.sparse_placeholder(dtypes.float64)
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_sparse_tensor_slices(st))
    init_op = iterator.initializer

    with self.cached_session() as sess:
      # Test with an empty sparse tensor but with non empty values.
      empty_indices = [[]]
      empty_values = []
      dense_shape = [1, 1]
      sparse_feed = sparse_tensor.SparseTensorValue(empty_indices, empty_values,
                                                    dense_shape)
      # Here, we expect the test to fail when running the feed.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(init_op, feed_dict={st: sparse_feed})

  @combinations.generate(combinations.combine(tf_api_version=2, mode=["eager"]))
  def testFromSparseTensorSlicesError(self):
    # [FUNCTION DOC]
    # Eager-mode test. Passing a null value instead of a sparse tensor must raise an attribute
    # error.
    with self.assertRaises(AttributeError):
      dataset_ops.Dataset.from_sparse_tensor_slices(None)


class FromSparseTensorSlicesCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint test case for the sparse-tensor-slices dataset. Inherits the checkpoint test base
  # class, which supplies the verification function that saves and restores iterator state.

  def _build_sparse_tensor_slice_dataset(self, slices):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test.
    # Parameter slices is a list of rows of floats. Converts them to a sparse tensor (64-bit
    # indices, double values, dense shape of row count by longest row plus one) and returns the
    # dataset of its slices.
    # pylint: disable=g-complex-comprehension
    indices = np.array(
        [[i, j] for i in range(len(slices)) for j in range(len(slices[i]))],
        dtype=np.int64)
    values = np.array([val for s in slices for val in s], dtype=np.float64)
    # pylint: enable=g-complex-comprehension
    dense_shape = np.array(
        [len(slices), max(len(s) for s in slices) + 1], dtype=np.int64)
    sparse_components = sparse_tensor.SparseTensor(indices, values, dense_shape)
    return dataset_ops.Dataset.from_sparse_tensor_slices(sparse_components)

  @combinations.generate(
      combinations.times(test_base.v1_only_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Checkpoint test (graph-version-1 only). Parameter verify_fn is the verification callback from
    # the checkpoint test base. Runs it on a dataset built from nine fixed slices, expecting nine
    # outputs and flagging that elements are sparse tensors.
    slices = [[1., 2., 3.], [1.], [1.], [1., 2.], [], [1., 2.], [], [], []]

    verify_fn(
        self,
        lambda: self._build_sparse_tensor_slice_dataset(slices),
        num_outputs=9,
        sparse_tensors=True)


if __name__ == "__main__":
  test.main()
