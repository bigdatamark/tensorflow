# [MODULE DOC]
# Unit tests for the dataset sparse batch transformation, which batches variable-length dense
# elements into sparse tensors with a given row shape: basic batching, unknown row dimensions,
# invalid shapes, shape errors at run time, and iterator checkpointing. No import time side
# effects; runs the test main when executed directly.
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
"""Tests for `tf.data.Dataset.sparse_batch`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DenseToSparseBatchTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for batching dense elements into sparse tensors using the dataset sparse batch method.

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    # [FUNCTION DOC]
    # Maps 100 random integers to vectors of that length filled with the value, sparse-batches by
    # four with row shape twelve. For each batch asserts the sparse indices, values and dense
    # shape (last dimension twelve), and that the iterator then raises out-of-range.
    components = np.random.randint(12, size=(100,)).astype(np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.fill([x], x)).sparse_batch(4, [12])
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
    # Maps each integer to a square matrix of that size filled with the value and sparse-batches
    # by four with a row shape whose second dimension is unknown. Asserts per batch the indices,
    # values and dense shape (the unknown dimension becomes the batch maximum), then out-of-range.
    components = np.random.randint(5, size=(40,)).astype(np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.fill([x, x], x)).sparse_batch(4, [5, None])

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
    # A negative row shape dimension must raise a value error mentioning that the dimension must
    # be non-negative when the sparse batch is constructed.
    input_tensor = array_ops.constant([[1]])
    with self.assertRaisesRegex(ValueError, "Dimension -2 must be >= 0"):
      dataset_ops.Dataset.from_tensors(input_tensor).sparse_batch(4, [-2])

  @combinations.generate(test_base.default_test_combinations())
  def testShapeErrors(self):
    # [FUNCTION DOC]
    # Sparse-batches a one-element dataset with row shape twelve. An input of incompatible rank
    # must raise an invalid-argument error about the row shape; an input of 13 elements, larger
    # than the row shape, must raise a data-loss error.

    def dataset_fn(input_tensor):
      # [FUNCTION DOC]
      # HELPER: wraps the given tensor in a one-element dataset and sparse-batches it by four with
      # row shape twelve. Returns the dataset.
      return dataset_ops.Dataset.from_tensors(input_tensor).sparse_batch(
          4, [12])

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
  # Checkpoint tests for the sparse batch dataset, built on the checkpoint test base.

  def _build_dataset(self, components):
    # [FUNCTION DOC]
    # HELPER: dataset factory for the checkpoint test. Takes an array of small integers, maps
    # each to a vector of that length filled with the value, and sparse-batches by four with row
    # shape twelve. Returns the dataset.
    return dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.fill([x], x)).sparse_batch(4, [12])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Runs the checkpoint verification function on the sparse batch dataset built from 40 random
    # small integers, expecting ten outputs (forty divided by the batch size four).
    components = np.random.randint(5, size=(40,)).astype(np.int32)

    num_outputs = len(components) // 4
    verify_fn(self, lambda: self._build_dataset(components), num_outputs)


if __name__ == "__main__":
  test.main()
