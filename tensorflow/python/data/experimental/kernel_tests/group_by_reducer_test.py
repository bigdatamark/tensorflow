# [MODULE DOC]
# Tests for the experimental group-by-reducer dataset transformation. Covers sum, average, string
# concat, sparse inputs, state whose shape changes, tuple state, type/shape/key validation errors,
# and checkpointing of the grouped dataset. No import-time side effects.
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
"""Tests for `tf.data.experimental.group_by_reducer()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class GroupByReducerTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests grouping elements by key and reducing each group with a reducer (init, reduce, finalize).

  @combinations.generate(test_base.default_test_combinations())
  def testSum(self):
    # [FUNCTION DOC]
    # For ranges of size 2 to 20, groups by parity with a summing reducer and asserts the two
    # group sums are produced as scalars.
    reducer = grouping.Reducer(
        init_func=lambda _: np.int64(0),
        reduce_func=lambda x, y: x + y,
        finalize_func=lambda x: x)
    for i in range(1, 11):
      dataset = dataset_ops.Dataset.range(2 * i).apply(
          grouping.group_by_reducer(lambda x: x % 2, reducer))
      self.assertDatasetProduces(
          dataset,
          expected_shapes=tensor_shape.TensorShape([]),
          expected_output=[(i - 1) * i, i * i])

  @combinations.generate(test_base.default_test_combinations())
  def testAverage(self):
    # [FUNCTION DOC]
    # Groups a range by parity with a running-average reducer whose state is the pair of average
    # and count; the finalize step returns the average. Asserts per-group averages.

    def reduce_fn(x, y):
      # [FUNCTION DOC]
      # HELPER: reduce step. Takes state (average, count) and a new value; returns the updated
      # average and incremented count.
      return (x[0] * x[1] + math_ops.cast(y, dtypes.float32)) / (
          x[1] + 1), x[1] + 1

    reducer = grouping.Reducer(
        init_func=lambda _: (0.0, 0.0),
        reduce_func=reduce_fn,
        finalize_func=lambda x, _: x)
    for i in range(1, 11):
      dataset = dataset_ops.Dataset.range(2 * i).apply(
          grouping.group_by_reducer(
              lambda x: math_ops.cast(x, dtypes.int64) % 2, reducer))
      self.assertDatasetProduces(
          dataset,
          expected_shapes=tensor_shape.TensorShape([]),
          expected_output=[i - 1, i])

  @combinations.generate(test_base.default_test_combinations())
  def testConcat(self):
    # [FUNCTION DOC]
    # Zips letters with a range and groups by parity of the range value, with a reducer that
    # concatenates the first element of each item. Asserts the concatenated byte strings.
    components = np.array(list("abcdefghijklmnopqrst")).view(np.char.chararray)
    reducer = grouping.Reducer(
        init_func=lambda x: "",
        reduce_func=lambda x, y: x + y[0],
        finalize_func=lambda x: x)
    for i in range(1, 11):
      dataset = dataset_ops.Dataset.zip(
          (dataset_ops.Dataset.from_tensor_slices(components),
           dataset_ops.Dataset.range(2 * i))).apply(
               grouping.group_by_reducer(lambda x, y: y % 2, reducer))
      self.assertDatasetProduces(
          dataset,
          expected_shapes=tensor_shape.TensorShape([]),
          expected_output=[b"acegikmoqs"[:i], b"bdfhjlnprt"[:i]])

  @combinations.generate(test_base.default_test_combinations())
  def testSparseSum(self):
    # [FUNCTION DOC]
    # Maps a range to single-value sparse tensors, groups by parity of the value, and sums using
    # sparse state; asserts the scalar sums per group.
    def _sparse(i):
      # [FUNCTION DOC]
      # HELPER: builds a one-element sparse tensor value holding the integer i at index (0, 0)
      # with dense shape (1, 1).
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1], dtype=np.int64)),
          dense_shape=np.array([1, 1]))

    reducer = grouping.Reducer(
        init_func=lambda _: _sparse(np.int64(0)),
        reduce_func=lambda x, y: _sparse(x.values[0] + y.values[0]),
        finalize_func=lambda x: x.values[0])
    for i in range(1, 11):
      dataset = dataset_ops.Dataset.range(2 * i).map(_sparse).apply(
          grouping.group_by_reducer(lambda x: x.values[0] % 2, reducer))
      self.assertDatasetProduces(
          dataset,
          expected_shapes=tensor_shape.TensorShape([]),
          expected_output=[(i - 1) * i, i * i])

  @combinations.generate(test_base.default_test_combinations())
  def testChangingStateShape(self):
    # [FUNCTION DOC]
    # Uses a reducer whose state doubles in length and gains a rank each step. Asserts the output
    # shapes are unknown length and unknown rank, that the values have the expected size and
    # nesting, and that the iterator is then exhausted.

    def reduce_fn(x, _):
      # [FUNCTION DOC]
      # HELPER: reduce step that concatenates the first state component with itself and adds a
      # leading dimension to the second; ignores the input value.
      # Statically known rank, but dynamic length.
      larger_dim = array_ops.concat([x[0], x[0]], 0)
      # Statically unknown rank.
      larger_rank = array_ops.expand_dims(x[1], 0)
      return larger_dim, larger_rank

    reducer = grouping.Reducer(
        init_func=lambda x: ([0], 1),
        reduce_func=reduce_fn,
        finalize_func=lambda x, y: (x, y))

    for i in range(1, 11):
      dataset = dataset_ops.Dataset.from_tensors(np.int64(0)).repeat(i).apply(
          grouping.group_by_reducer(lambda x: x, reducer))
      dataset_output_shapes = dataset_ops.get_legacy_output_shapes(dataset)
      self.assertEqual([None], dataset_output_shapes[0].as_list())
      self.assertIs(None, dataset_output_shapes[1].ndims)
      get_next = self.getNext(dataset)
      x, y = self.evaluate(get_next())
      self.assertAllEqual([0] * (2**i), x)
      self.assertAllEqual(np.array(1, ndmin=i), y)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testTypeMismatch(self):
    # [FUNCTION DOC]
    # Reducer init returns 32-bit ints while reduce returns 64-bit; asserts applying it raises a
    # type error.
    reducer = grouping.Reducer(
        init_func=lambda x: constant_op.constant(1, dtype=dtypes.int32),
        reduce_func=lambda x, y: constant_op.constant(1, dtype=dtypes.int64),
        finalize_func=lambda x: x)

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(TypeError):
      dataset.apply(
          grouping.group_by_reducer(lambda _: np.int64(0), reducer))

  # TODO(b/78665031): Remove once non-scalar keys are supported.
  @combinations.generate(test_base.default_test_combinations())
  def testInvalidKeyShape(self):
    # [FUNCTION DOC]
    # Key function returns a non-scalar key; asserts applying the transformation raises a value
    # error.
    reducer = grouping.Reducer(
        init_func=lambda x: np.int64(0),
        reduce_func=lambda x, y: x + y,
        finalize_func=lambda x: x)

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(ValueError):
      dataset.apply(
          grouping.group_by_reducer(lambda _: np.int64((0, 0)), reducer))

  # TODO(b/78665031): Remove once non-int64 keys are supported.
  @combinations.generate(test_base.default_test_combinations())
  def testInvalidKeyType(self):
    # [FUNCTION DOC]
    # Key function returns a string instead of a 64-bit integer; asserts a value error.
    reducer = grouping.Reducer(
        init_func=lambda x: np.int64(0),
        reduce_func=lambda x, y: x + y,
        finalize_func=lambda x: x)

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(ValueError):
      dataset.apply(
          grouping.group_by_reducer(lambda _: "wrong", reducer))

  @combinations.generate(test_base.default_test_combinations())
  def testTuple(self):
    # [FUNCTION DOC]
    # Zips two ranges and puts everything in one group with a reducer whose state is a pair:
    # a growing list of first values and a running sum of second values. Asserts the list holds
    # 0..9 and the sum is 45.
    def init_fn(_):
      # [FUNCTION DOC]
      # HELPER: initial state: an empty 64-bit integer array and a zero sum.
      return np.array([], dtype=np.int64), np.int64(0)

    def reduce_fn(state, value):
      # [FUNCTION DOC]
      # HELPER: appends the first value component to the list and adds the second to the sum.
      s1, s2 = state
      v1, v2 = value
      return array_ops.concat([s1, [v1]], 0), s2 + v2

    def finalize_fn(s1, s2):
      # [FUNCTION DOC]
      # HELPER: returns the final state components unchanged.
      return s1, s2

    reducer = grouping.Reducer(init_fn, reduce_fn, finalize_fn)
    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(10), dataset_ops.Dataset.range(10))).apply(
            grouping.group_by_reducer(lambda x, y: np.int64(0), reducer))
    get_next = self.getNext(dataset)
    x, y = self.evaluate(get_next())
    self.assertAllEqual(x, np.asarray([x for x in range(10)]))
    self.assertEqual(y, 45)


class GroupByReducerCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                   parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for datasets produced by group-by-reducer.

  def _build_dataset(self, components):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test. From the given array of integers, groups by value
    # modulo five and sums each group. Returns the dataset.
    reducer = grouping.Reducer(
        init_func=lambda _: np.int64(0),
        reduce_func=lambda x, y: x + y,
        finalize_func=lambda x: x)

    return dataset_ops.Dataset.from_tensor_slices(components).apply(
        grouping.group_by_reducer(lambda x: x % 5, reducer))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Runs the checkpoint verification callback on the grouped dataset built from ten integers,
    # expecting five outputs.
    components = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    verify_fn(
        self,
        lambda: self._build_dataset(components),
        num_outputs=5)


if __name__ == "__main__":
  test.main()
