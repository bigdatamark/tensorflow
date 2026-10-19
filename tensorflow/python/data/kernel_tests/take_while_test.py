# [MODULE DOC]
# Unit tests for the dataset take_while transformation, which yields elements until a predicate
# first returns false and then ends. Covers batched predicates, range and string datasets, short
# circuiting, interaction with repeat, naming, and checkpointing with and without symbolic
# checkpoints. Not a helper module; defines test classes only.
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
"""Tests for `tf.data.Dataset.take_while()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TakeWhileTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Behavior tests for take_while on small in-memory datasets.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_elements=[14, 15], window_size=[2]) +
          combinations.combine(num_elements=[100], window_size=[3])))
  def testTakeWhileDataset(self, num_elements, window_size):
    # [FUNCTION DOC]
    # Parameterized over element count and batch size. Batches a range dataset, then takes batches
    # while the batch is full (its first dimension is at least window_size), and flattens the
    # batches back to elements. Asserts the output is the leading elements up to the last full
    # batch (the largest multiple of window_size not exceeding num_elements).

    def _predicate_func(elem):
      # [FUNCTION DOC]
      # HELPER: predicate for the test. Returns a boolean tensor that is true when the batch
      # elem has at least window_size entries (captured from the enclosing test).
      return array_ops.shape(elem)[0] > (window_size - 1)

    dataset = dataset_ops.Dataset.range(num_elements).batch(window_size)
    dataset = dataset.take_while(predicate=_predicate_func).flat_map(
        dataset_ops.Dataset.from_tensor_slices)

    expected_num_elements = int(num_elements / window_size) * window_size
    self.assertDatasetProduces(dataset, np.arange(expected_num_elements))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_elements=[10], upper_bound=[2]) +
          combinations.combine(num_elements=[16], upper_bound=[7]) +
          combinations.combine(num_elements=[100], upper_bound=[99]) +
          combinations.combine(num_elements=[100], upper_bound=[101]) +
          combinations.combine(num_elements=[0], upper_bound=[1])))
  def testTakeWhileDatasetRange(self, num_elements, upper_bound):
    # [FUNCTION DOC]
    # Parameterized over range size and upper bound (including empty range and bound past the
    # end). Takes elements while they are below upper_bound and asserts the output is the
    # integers from zero up to the smaller of the two numbers.
    dataset = dataset_ops.Dataset.range(num_elements).take_while(
        lambda x: x < upper_bound)

    self.assertDatasetProduces(dataset,
                               np.arange(min(num_elements, upper_bound)))

  @combinations.generate(test_base.default_test_combinations())
  def testTakeWhileDatasetString(self):
    # [FUNCTION DOC]
    # Uses a dataset of six words and a predicate that is true while the element differs from the
    # word "test". Asserts the first three words ("this", "is", "the") are produced as byte
    # strings and the fourth read raises an out-of-range error.

    def not_equal(string):
      # [FUNCTION DOC]
      # HELPER: predicate factory. Given a Python string, returns a function mapping an element
      # to a boolean tensor that is true when the element is not equal to that string.
      return lambda x: math_ops.not_equal(x, constant_op.constant(string))

    string = ["this", "is", "the", "test", "for", "strings"]
    dataset = dataset_ops.Dataset.from_tensor_slices(string).take_while(
        predicate=not_equal("test"))

    next_element = self.getNext(dataset)
    self.assertEqual(b"this", self.evaluate(next_element()))
    self.assertEqual(b"is", self.evaluate(next_element()))
    self.assertEqual(b"the", self.evaluate(next_element()))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(size=[5], index=[3]) +
          combinations.combine(size=[10], index=[0]) +
          combinations.combine(size=[100], index=[5]) +
          combinations.combine(size=[8], index=[7])))
  def testTakewhileDatasetShortCircuit(self, size, index):
    # [FUNCTION DOC]
    # Parameterized over list size and position of a single false value. Builds a dataset of
    # booleans that are all true except at index, and takes while the element is true. Asserts
    # exactly index true values are read, then the next read raises an out-of-range error
    # (iteration stops at the first false even though later values may be true).

    def _predicate_func(data_elem):
      # [FUNCTION DOC]
      # HELPER: identity predicate; returns the boolean element unchanged.
      return data_elem

    boolean_array = [True] * size
    boolean_array[index] = False
    dataset = dataset_ops.Dataset.from_tensor_slices(boolean_array).take_while(
        predicate=_predicate_func)

    next_element = self.getNext(dataset)

    for _ in range(index):
      self.assertTrue(self.evaluate(next_element()))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testTakeWhileDatasetWithRepeat(self):
    # [FUNCTION DOC]
    # Takes elements below two from a ten-element range and repeats the result five times.
    # Asserts the output is the sequence zero, one tiled five times.
    dataset = dataset_ops.Dataset.range(10).take_while(
        predicate=lambda x: x < 2).repeat(5)
    self.assertDatasetProduces(dataset, np.tile([0, 1], 5))

  @combinations.generate(test_base.default_test_combinations())
  def testTakeWhileDatasetStops(self):
    # [FUNCTION DOC]
    # Takes elements of a ten-element range while they are not equal to five. Asserts the output
    # is zero through four.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.take_while(
        lambda x: math_ops.logical_not(math_ops.equal(x, 5)))
    self.assertDatasetProduces(dataset, range(5))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Checks that take_while accepts a name argument: an always-true predicate over a single
    # element dataset holding 42 yields exactly that value.
    dataset = dataset_ops.Dataset.from_tensors(42).take_while(
        lambda _: True, name="take_while")
    self.assertDatasetProduces(dataset, [42])


class TakeWhileCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for a range dataset followed by take_while, using the shared
  # checkpoint verification fixture.

  def _build_dataset(self, num_elements, upper_bound, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test.
    # Returns a range dataset of num_elements filtered by take_while with a predicate true below
    # upper_bound. If options (optional dataset options) is given it is applied first.
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.take_while(predicate=lambda x: x < upper_bound)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True]),
          combinations.combine(num_elements=[10, 23], upper_bound=[10, 23])))
  def test(self, verify_fn, symbolic_checkpoint, num_elements, upper_bound):
    # [FUNCTION DOC]
    # Parameterized checkpoint test. Inputs: verify_fn (checkpoint verification callback),
    # symbolic_checkpoint (whether symbolic checkpointing is enabled), num_elements and
    # upper_bound. Asks verify_fn to check that saving and restoring the take_while dataset
    # yields min(num_elements, upper_bound) elements.
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self,
              lambda: self._build_dataset(num_elements, upper_bound, options),
              min(num_elements, upper_bound))


if __name__ == "__main__":
  test.main()
