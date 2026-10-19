# [MODULE DOC]
# Kernel tests for the experimental dense-to-sparse batching transformation. Covers basic batching
# of variable-length rows into sparse tensors, unknown row dimensions, invalid row shapes, shape
# error reporting, and checkpoint/restore of the iterator. Defines two test classes; no
# import-time side effects beyond test registration via the test main entry point.
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
"""Tests for `tf.data.experimental.dense_to_sparse_batch()."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DenseToSparseBatchTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the dense-to-sparse batch transformation on datasets of variable-length
  # rows, run under the default eager/graph test combinations.

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    # [FUNCTION DOC]
    # Test: builds a dataset of 100 random small integers, maps each to a row of that length
    # filled with itself, and batches with batch size 4 and row shape 12. For every batch it
    # asserts the sparse indices, values and dense shape (last batch may be smaller), and that
    # the iterator then raises an out-of-range error.
    components = np.random.randint(12, size=(100,)).astype(np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices(
        components).map(lambda x: array_ops.fill([x], x)).apply(
            batching.dense_to_sparse_batch(4, [12]))
    get_next = self.getNext(dataset)

    for start in range(0, len(components), 4):
      results = self.evaluate(get_next())
      self.assertAllEqual([[i, j]
                           for i, c in enumerate(components[start:start + 4])
                           for j in range(c)], results.indices)
      self.assertAllEqual(
          [c for c in components[start:start + 4] for _ in range(c)],
          results.values)
      self.assertAllEqual([min(4,
                               len(components) - start), 12],
                          results.dense_shape)

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testWithUnknownShape(self):
    # [FUNCTION DOC]
    # Test: rows are square (length by length) matrices filled with their length, batched with
    # row shape of 5 and an unknown second dimension. Asserts indices, values and a dense shape
    # whose last dimension is the maximum length in the batch, then out-of-range at the end.
    components = np.random.randint(5, size=(40,)).astype(np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices(
        components).map(lambda x: array_ops.fill([x, x], x)).apply(
            batching.dense_to_sparse_batch(4, [5, None]))

    get_next = self.getNext(dataset)

    for start in range(0, len(components), 4):
      results = self.evaluate(get_next())
      self.assertAllEqual([[i, j, z]
                           for i, c in enumerate(components[start:start + 4])
                           for j in range(c)
                           for z in range(c)], results.indices)
      self.assertAllEqual([
          c for c in components[start:start + 4] for _ in range(c)
          for _ in range(c)
      ], results.values)
      self.assertAllEqual([
          min(4,
              len(components) - start), 5,
          np.max(components[start:start + 4])
      ], results.dense_shape)

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testWithInvalidShape(self):
    # [FUNCTION DOC]
    # Test: applying the batching transformation with a negative row dimension must raise a
    # value error stating the dimension must be non-negative.
    input_tensor = array_ops.constant([[1]])
    with self.assertRaisesRegex(ValueError, "Dimension -2 must be >= 0"):
      dataset_ops.Dataset.from_tensors(input_tensor).apply(
          batching.dense_to_sparse_batch(4, [-2]))

  @combinations.generate(test_base.default_test_combinations())
  def testShapeErrors(self):
    # [FUNCTION DOC]
    # Test: feeds single elements that do not fit the row shape of 12. A rank-mismatched input
    # must produce an invalid-argument error ("incompatible with the row shape"); an input with
    # 13 elements must produce a data-loss error ("larger than the row shape").

    def dataset_fn(input_tensor):
      # [FUNCTION DOC]
      # HELPER: builds the dataset used by testShapeErrors.
      # Wraps the given tensor in a single-element dataset and applies sparse batching with batch
      # size 4 and row shape 12; returns the resulting dataset.
      return dataset_ops.Dataset.from_tensors(input_tensor).apply(
          batching.dense_to_sparse_batch(4, [12]))

    # Initialize with an input tensor of incompatible rank.
    get_next = self.getNext(dataset_fn([[1]]))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "incompatible with the row shape"):
      self.evaluate(get_next())

    # Initialize with an input tensor that is larger than `row_shape`.
    get_next = self.getNext(dataset_fn(np.int32(range(13))))
    with self.assertRaisesRegex(errors.DataLossError,
                                "larger than the row shape"):
      self.evaluate(get_next())


class DenseToSparseBatchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                       parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint test for the dense-to-sparse batch dataset: verifies iterator save/restore
  # behavior via the shared checkpoint test base.

  def _build_dataset(self, components):
    # [FUNCTION DOC]
    # HELPER: dataset factory for the checkpoint test.
    # Takes an array of small integers; builds slices, maps each to a row of that length filled
    # with itself, and applies sparse batching with batch size 4 and row shape 12. Returns it.
    return dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.fill([x], x)).apply(
            batching.dense_to_sparse_batch(4, [12]))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Test: creates 40 random small integers and expects 10 batches of 4. Calls the supplied
    # verification callback with the dataset factory and expected output count, which checks
    # iterator checkpoint save and restore at various points.
    components = np.random.randint(5, size=(40,)).astype(np.int32)

    num_outputs = len(components) // 4
    verify_fn(self, lambda: self._build_dataset(components), num_outputs)


if __name__ == "__main__":
  test.main()
