# [MODULE DOC]
# Unit tests for the dataset method that creates a numpy iterator (as_numpy_iterator). Covers basic
# iteration, read-only results, nested structures, graph mode failure, sparse, ragged, none and
# unsupported (dataset) elements, and checkpoint compatibility of the numpy iterator.
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
"""Tests for `tf.data.Dataset.numpy()`."""
import collections
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import test
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops


class AsNumpyIteratorTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the as_numpy_iterator method of datasets. Most tests run in eager
  # mode only because the numpy iterator is unavailable in graph mode.

  @combinations.generate(test_base.eager_only_combinations())
  def testBasic(self):
    # [FUNCTION DOC]
    # Iterates a range dataset of three elements through the numpy iterator and asserts the
    # produced values are zero, one, two.
    ds = dataset_ops.Dataset.range(3)
    self.assertEqual([0, 1, 2], list(ds.as_numpy_iterator()))

  @combinations.generate(test_base.eager_only_combinations())
  def testImmutable(self):
    # [FUNCTION DOC]
    # Takes the first element (an array) from a numpy iterator and tries to assign into it.
    # Asserts a value error saying the assignment destination is read-only.
    ds = dataset_ops.Dataset.from_tensors([1, 2, 3])
    arr = next(ds.as_numpy_iterator())
    with self.assertRaisesRegex(ValueError,
                                'assignment destination is read-only'):
      arr[0] = 0

  @combinations.generate(test_base.eager_only_combinations())
  def testNestedStructure(self):
    # [FUNCTION DOC]
    # Builds a dataset from slices of a mapping holding a tuple, a plain list and a named tuple, and
    # asserts the numpy iterator yields the same nested structure (mapping, tuple, named tuple)
    # with numeric values for each of the two elements.
    point = collections.namedtuple('Point', ['x', 'y'])
    ds = dataset_ops.Dataset.from_tensor_slices({
        'a': ([1, 2], [3, 4]),
        'b': [5, 6],
        'c': point([7, 8], [9, 10])
    })
    self.assertEqual([{
        'a': (1, 3),
        'b': 5,
        'c': point(7, 9)
    }, {
        'a': (2, 4),
        'b': 6,
        'c': point(8, 10)
    }], list(ds.as_numpy_iterator()))

  @combinations.generate(test_base.graph_only_combinations())
  def testNonEager(self):
    # [FUNCTION DOC]
    # Graph-mode only: asserts that requesting a numpy iterator raises a runtime error.
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaises(RuntimeError):
      ds.as_numpy_iterator()

  def _testInvalidElement(self, element):
    # [FUNCTION DOC]
    # HELPER: shared check for tests of unsupported element types.
    # Input: element, a value to wrap in a single-element dataset. Asserts that requesting a numpy
    # iterator raises a type error stating the operation is not supported for datasets that
    # contain such elements. Returns nothing.
    ds = dataset_ops.Dataset.from_tensors(element)
    with self.assertRaisesRegex(TypeError,
                                'is not supported for datasets that'):
      ds.as_numpy_iterator()

  @combinations.generate(test_base.eager_only_combinations())
  def testSparseElement(self):
    # [FUNCTION DOC]
    # Creates a sparse tensor, slices it into a dataset and asserts the numpy iterator yields the
    # same values as the dense conversion of the sparse tensor.
    st = sparse_tensor.SparseTensor(
        indices=[(0, 0), (1, 1), (2, 2)], values=[1, 2, 3], dense_shape=(3, 3))
    ds = dataset_ops.Dataset.from_tensor_slices(st)
    dt = sparse_ops.sparse_tensor_to_dense(st)
    self.assertAllEqual(list(ds.as_numpy_iterator()), dt.numpy())

  @combinations.generate(test_base.eager_only_combinations())
  def testRaggedElement(self):
    # [FUNCTION DOC]
    # Creates a dataset from a ragged tensor holding one ragged row and asserts the numpy iterator
    # yields, for it, a sequence of integer arrays equal to the original rows (compared by length
    # and element-wise equality).
    lst = [[1, 2], [3], [4, 5, 6]]
    rt = ragged_factory_ops.constant([lst])
    # This dataset consists of exactly one ragged tensor.
    ds = dataset_ops.Dataset.from_tensor_slices(rt)
    expected = np.array([
        np.array([1, 2], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([4, 5, 6], dtype=np.int32)
    ], dtype=object)
    for actual in ds.as_numpy_iterator():
      self.assertEqual(len(actual), len(expected))
      for actual_arr, expected_arr in zip(actual, expected):
        self.assertTrue(np.array_equal(actual_arr, expected_arr),
                        f'{actual_arr} != {expected_arr}')

  @combinations.generate(test_base.eager_only_combinations())
  def testDatasetElement(self):
    # [FUNCTION DOC]
    # Asserts that a dataset whose element is itself a dataset is rejected by the numpy iterator
    # (via the invalid element helper).
    self._testInvalidElement(dataset_ops.Dataset.range(3))

  @combinations.generate(test_base.eager_only_combinations())
  def testNestedNonTensorElement(self):
    # [FUNCTION DOC]
    # Asserts that a tuple element that contains a dataset next to a tensor is rejected by the
    # numpy iterator (via the invalid element helper).
    tuple_elem = (constant_op.constant([1, 2, 3]), dataset_ops.Dataset.range(3))
    self._testInvalidElement(tuple_elem)

  @combinations.generate(test_base.eager_only_combinations())
  def testNoneElement(self):
    # [FUNCTION DOC]
    # Builds a dataset whose single element is a pair of a number and nothing, and asserts the
    # numpy iterator yields that pair with the null value preserved.
    ds = dataset_ops.Dataset.from_tensors((2, None))
    self.assertAllEqual(list(ds.as_numpy_iterator()), [(2, None)])

  @combinations.generate(combinations.times(
      test_base.eager_only_combinations(),
      combinations.combine(enable_async_ckpt=[True, False])
  ))
  def testCompatibleWithCheckpoint(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Parameter: enable_async_ckpt, boolean selecting asynchronous checkpoint writing.
    # Advances a numpy iterator five times, saves a checkpoint of it, then asserts the next two
    # values are five and six. Creates a fresh iterator, restores the checkpoint (syncing first
    # if asynchronous) and asserts the first value after restoring is five.
    # Side effect: writes checkpoint files under the test temporary directory.
    ds = dataset_ops.Dataset.range(10)
    iterator = ds.as_numpy_iterator()
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    for _ in range(5):
      next(iterator)

    prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    save_path = ckpt.save(prefix, options=ckpt_options)
    self.assertEqual(5, next(iterator))
    self.assertEqual(6, next(iterator))
    restore_iter = ds.as_numpy_iterator()
    restore_ckpt = trackable_utils.Checkpoint(iterator=restore_iter)
    if enable_async_ckpt:
      ckpt.sync()  # Otherwise save may not finish yet
    restore_ckpt.restore(save_path)
    self.assertEqual(5, next(restore_iter))


if __name__ == '__main__':
  test.main()
