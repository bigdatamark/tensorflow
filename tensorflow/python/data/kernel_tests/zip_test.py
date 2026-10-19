# [MODULE DOC]
# Unit tests for the dataset zip operation. Covers zipping equal and unequal length datasets,
# nested structures, named tuples, data classes, attrs classes, naming, argument validation,
# checkpointing (including symbolic checkpoints), random access by index, and global shuffling
# of zipped datasets. Defines a small dataset-building helper and a data class used as a test
# structure. Has no import-time side effects other than an optional import of the attrs package.
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
"""Tests for `tf.data.Dataset.zip()`."""
import collections
import dataclasses
from typing import Callable, Tuple

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


def _dataset_factory(components):
  # [FUNCTION DOC]
  # HELPER: builds the zipped datasets used by the zip tests.
  # Takes a list of components (each a list or array). Builds one dataset of slices per component,
  # zips them into a tuple-structured dataset and returns it. No side effects.
  datasets = tuple([
      dataset_ops.Dataset.from_tensor_slices(component)
      for component in components
  ])
  return dataset_ops.Dataset.zip(datasets)


@dataclasses.dataclass
class MaskedNdarrayPair:
  # [CLASS DOC]
  # HELPER: custom composite structure used to test zipping of data classes.
  # Data class holding a boolean mask and two values. Implements flatten and unflatten protocol
  # methods so that it is treated as a structure with the mask as metadata and the two values as
  # components.
  mask: bool
  value1: np.ndarray
  value2: np.ndarray

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # HELPER: structure-flattening protocol hook for the test data class.
    # Returns a pair: metadata (a tuple holding the mask) and components (a tuple of the two
    # values). Takes no arguments, no side effects.
    metadata = (self.mask,)
    components = (self.value1, self.value2)
    return metadata, components

  def __tf_unflatten__(self, metadata, components):
    # [FUNCTION DOC]
    # HELPER: structure-rebuilding protocol hook for the test data class.
    # Takes the metadata tuple (whose first item is the mask) and the components pair (two
    # values) and returns a new instance with the mask and both values restored.
    mask = metadata[0]
    value1, value2 = components
    return MaskedNdarrayPair(mask=mask, value1=value1, value2=value2)


class ZipTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering basic behavior of the dataset zip operation across test combinations
  # (graph and eager modes): equal/unequal lengths, nesting, named tuples, data classes, attrs
  # classes, naming and invalid arguments.

  @combinations.generate(test_base.default_test_combinations())
  def testZipEqual(self):
    # [FUNCTION DOC]
    # Scenario: zip three datasets of equal length (4 elements, different shapes and dtypes).
    # Iterates the zipped dataset and asserts each element's parts equal the source rows, then
    # asserts that the iterator raises an out-of-range error when exhausted, twice in a row.
    components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    ]
    get_next = self.getNext(_dataset_factory(components))
    for i in range(4):
      results = self.evaluate(get_next())
      for component, result_component in zip(components, results):
        self.assertAllEqual(component[i], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testZipUnequal(self):
    # [FUNCTION DOC]
    # Scenario: zip datasets of lengths 4, 5 and 2. Asserts only the first 2 elements (the
    # shortest length) are produced with correct values, then that the iterator raises an
    # out-of-range error, twice in a row.
    components = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1.0, 2.0]]
    get_next = self.getNext(_dataset_factory(components))
    for i in range(2):
      results = self.evaluate(get_next())
      for component, result_component in zip(components, results):
        self.assertAllEqual(component[i], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testNested(self):
    # [FUNCTION DOC]
    # Scenario: zip with a nested structure (first dataset, then a pair of the other two).
    # Asserts the legacy output shapes are as expected, that each element unpacks to the three
    # matching values, and that exhaustion raises an out-of-range error, twice in a row.

    components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    ]
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(component)
        for component in components
    ]
    dataset = dataset_ops.Dataset.zip((datasets[0], (datasets[1], datasets[2])))

    self.assertEqual(
        dataset_ops.get_legacy_output_shapes(dataset),
        (tensor_shape.TensorShape([20]),
         (tensor_shape.TensorShape([22]), tensor_shape.TensorShape([]))))

    get_next = self.getNext(dataset)
    for i in range(4):
      result1, (result2, result3) = self.evaluate(get_next())
      self.assertAllEqual(components[0][i], result1)
      self.assertAllEqual(components[1][i], result2)
      self.assertAllEqual(components[2][i], result3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testNamedTuple(self):
    # [FUNCTION DOC]
    # Scenario: zip a named tuple whose fields are two range datasets. Asserts the dataset
    # produces named tuples with the paired values (0,3), (1,4), (2,5).
    Foo = collections.namedtuple("Foo", ["x", "y"])
    x = Foo(x=dataset_ops.Dataset.range(3), y=dataset_ops.Dataset.range(3, 6))
    dataset = dataset_ops.Dataset.zip(x)
    expected = [Foo(x=0, y=3), Foo(x=1, y=4), Foo(x=2, y=5)]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testDataclass(self):
    # [FUNCTION DOC]
    # Scenario: zip an instance of the test data class whose two values are range datasets.
    # Asserts the dataset yields data class instances with the same mask and paired values.
    mtp = MaskedNdarrayPair(
        mask=True,
        value1=dataset_ops.Dataset.range(3),
        value2=dataset_ops.Dataset.range(3, 6),
    )
    dataset = dataset_ops.Dataset.zip(mtp)
    expected = [
        MaskedNdarrayPair(mask=True, value1=0, value2=3),
        MaskedNdarrayPair(mask=True, value1=1, value2=4),
        MaskedNdarrayPair(mask=True, value1=2, value2=5),
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testAttrs(self):
    # [FUNCTION DOC]
    # Scenario: zip an instance of a locally defined attrs class holding two range datasets.
    # Skips the test when the attrs package is unavailable. Asserts the dataset yields attrs
    # instances with the paired values.
    if attr is None:
      self.skipTest("attr module is not available.")

    @attr.s
    class Foo:
      # [CLASS DOC]
      # Local attrs class with two fields (x and y) used as the structure being zipped.
      x = attr.ib()
      y = attr.ib()

    x = Foo(x=dataset_ops.Dataset.range(3), y=dataset_ops.Dataset.range(3, 6))
    dataset = dataset_ops.Dataset.zip(x)
    expected = [Foo(x=0, y=3), Foo(x=1, y=4), Foo(x=2, y=5)]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: zip two single-element datasets while passing a name argument. Asserts the
    # result produces the single pair (4, 2).
    x = dataset_ops.Dataset.from_tensors(4)
    y = dataset_ops.Dataset.from_tensors(2)
    dataset = dataset_ops.Dataset.zip((x, y), name="zip")
    self.assertDatasetProduces(dataset, [(4, 2)])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations())
  )
  def testZipWithArgsAndDataset(self):
    # [FUNCTION DOC]
    # Scenario: call zip with both positional datasets and the datasets keyword. Asserts a
    # type error stating both cannot be set.
    with self.assertRaisesRegex(
        TypeError, r"Both `\*args` and `datasets` cannot be set."
    ):
      dataset_ops.Dataset.zip(
          dataset_ops.Dataset.range(1, 4),
          dataset_ops.Dataset.range(4, 7),
          datasets=(
              dataset_ops.Dataset.range(1, 4),
              dataset_ops.Dataset.range(4, 7),
          ),
      )

  @combinations.generate(
      combinations.times(test_base.default_test_combinations())
  )
  def testZipBasicWithNoInput(self):
    # [FUNCTION DOC]
    # Scenario: call zip with no datasets. Asserts a type error saying at least one dataset
    # must be passed.
    with self.assertRaisesRegex(
        TypeError, r"Must pass at least one dataset to `zip`."
    ):
      dataset_ops.Dataset.zip()

  @combinations.generate(
      combinations.times(test_base.default_test_combinations())
  )
  def InvalidZipInputList(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name does not start with "test", so the test runner never collects or
    # runs this method; it is effectively dead test code.
    # Scenario: zip called with plain lists of numbers instead of datasets. Intended to assert a
    # type error saying lists are unsupported and tuples should be used.
    with self.assertRaisesRegex(
        TypeError,
        r"Invalid input to `zip`. Inputs are expected to be (nested)"
        r" structures of `tf.data.Dataset` objects. Python `list` is"
        r" not supported and you should use `tuple` instead.",
    ):
      dataset_ops.Dataset.zip([1, 2, 3], [4, 5, 6])


class ZipCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):
  # [CLASS DOC]
  # Checkpoint test case for the zip operation: verifies saving and restoring iterator state
  # of a nested zipped dataset, with and without symbolic checkpoints.

  def _build_dataset(self, arr, options=None):
    # [FUNCTION DOC]
    # HELPER: dataset builder for the checkpoint test.
    # Takes arr (values for the third component) and optional dataset options. Zips the first
    # dataset with a pair of the other two (nested), applies the options if given, and returns
    # the dataset.
    components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array(arr)
    ]
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(component)
        for component in components
    ]
    dataset = dataset_ops.Dataset.zip((datasets[0], (datasets[1], datasets[2])))
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(elements=[[37.0, 38.0, 39.0, 40.0], [1.0, 2.0]]),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, elements, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Scenario: for two element lists and symbolic checkpoint on/off, builds the nested zipped
    # dataset with the experimental symbolic checkpoint option set and runs the verification
    # function, expecting as many outputs as elements. Checkpoint save/restore must be correct.
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self._build_dataset(elements, options),
              len(elements))


class ZipRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering random access (fetching an element by index) on zipped datasets, for
  # valid and invalid indices and various structures.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 3, 4])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # Scenario: zip two 3-element range datasets and fetch by index -1, 3 or 4. Asserts an
    # out-of-range error.
    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(1, 4), dataset_ops.Dataset.range(4, 7)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 0])))
  def testEmptyDataset(self, index):
    # [FUNCTION DOC]
    # Scenario: zip two empty datasets and fetch by index -1 or 0. Asserts an out-of-range error.
    dataset = dataset_ops.Dataset.zip(
        datasets=(dataset_ops.Dataset.from_tensor_slices([]),
                  dataset_ops.Dataset.from_tensor_slices([])))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testZipBasic(self):
    # [FUNCTION DOC]
    # Scenario: zip two range datasets given as a tuple; fetch indices 0-2 by random access and
    # assert they equal (1,4), (2,5), (3,6).
    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(1, 4), dataset_ops.Dataset.range(4, 7)))
    expected_dataset = [(1, 4), (2, 5), (3, 6)]
    for i in range(3):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)),
          expected_dataset[i])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testZipBasicWithoutTuple(self):
    # [FUNCTION DOC]
    # Scenario: same as the basic random-access test but datasets are passed as separate
    # positional arguments rather than a tuple. Asserts the same paired values by index.
    dataset = dataset_ops.Dataset.zip(
        dataset_ops.Dataset.range(1, 4), dataset_ops.Dataset.range(4, 7)
    )
    expected_dataset = [(1, 4), (2, 5), (3, 6)]
    for i in range(3):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), expected_dataset[i]
      )

  @combinations.generate(
      combinations.times(test_base.default_test_combinations())
  )
  def testZipEqual(self):
    # [FUNCTION DOC]
    # Scenario: random access over three equal-length zipped datasets. Asserts each index 0-3
    # returns the matching row of every component, and index 4 raises an out-of-range error.
    components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    ]
    dataset = _dataset_factory(components)
    for i in range(4):
      results = self.evaluate(random_access.at(dataset, index=i))
      for component, result_component in zip(components, results):
        self.assertAllEqual(component[i], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=4))

  @combinations.generate(test_base.default_test_combinations())
  def testZipUnequal(self):
    # [FUNCTION DOC]
    # Scenario: random access over zipped datasets of lengths 4, 5 and 2. Asserts indices 0-1
    # return correct values and index 2 raises an out-of-range error.
    components = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1.0, 2.0]]
    dataset = _dataset_factory(components)
    for i in range(2):
      results = self.evaluate(random_access.at(dataset, index=i))
      for component, result_component in zip(components, results):
        self.assertAllEqual(component[i], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=2))

  @combinations.generate(test_base.default_test_combinations())
  def testNested(self):
    # [FUNCTION DOC]
    # Scenario: random access over a nested zip (tuple containing a pair). Asserts each index
    # 0-3 unpacks to the right three values and index 4 raises an out-of-range error.
    components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    ]
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(component)
        for component in components
    ]
    dataset = dataset_ops.Dataset.zip((datasets[0], (datasets[1], datasets[2])))
    for i in range(4):
      result1, (result2,
                result3) = self.evaluate(random_access.at(dataset, index=i))
      self.assertAllEqual(components[0][i], result1)
      self.assertAllEqual(components[1][i], result2)
      self.assertAllEqual(components[2][i], result3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=4))

  @combinations.generate(test_base.default_test_combinations())
  def testNestedWithoutTuple(self):
    # [FUNCTION DOC]
    # Scenario: like the nested random-access test but the datasets are passed as positional
    # arguments. Asserts correct values for indices 0-3 and an out-of-range error at index 4.
    components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0]),
    ]
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(component)
        for component in components
    ]
    dataset = dataset_ops.Dataset.zip(datasets[0], (datasets[1], datasets[2]))
    for i in range(4):
      result1, (result2, result3) = self.evaluate(
          random_access.at(dataset, index=i)
      )
      self.assertAllEqual(components[0][i], result1)
      self.assertAllEqual(components[1][i], result2)
      self.assertAllEqual(components[2][i], result3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=4))

  @combinations.generate(test_base.default_test_combinations())
  def testNamedTuple(self):
    # [FUNCTION DOC]
    # Scenario: random access over a zipped named tuple of two range datasets. Asserts indices
    # 0-2 match the expected named tuples and index 4 raises an out-of-range error.
    Foo = collections.namedtuple("Foo", ["x", "y"])
    x = Foo(x=dataset_ops.Dataset.range(3), y=dataset_ops.Dataset.range(3, 6))
    dataset = dataset_ops.Dataset.zip(x)
    expected = [Foo(x=0, y=3), Foo(x=1, y=4), Foo(x=2, y=5)]
    for i in range(3):
      self.assertAllEqual(
          self.evaluate(random_access.at(dataset, index=i)), expected[i])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=4))

  @combinations.generate(test_base.default_test_combinations())
  def testAttrs(self):
    # [FUNCTION DOC]
    # Scenario: random access over a zipped locally defined attrs class. Skips when attrs is
    # unavailable. Asserts indices 0-2 match the expected instances and index 4 raises an
    # out-of-range error.
    if attr is None:
      self.skipTest("attr module is not available.")

    @attr.s
    class Foo:
      # [CLASS DOC]
      # Local attrs class with two fields (x and y) used as the structure being zipped.
      x = attr.ib()
      y = attr.ib()

    x = Foo(x=dataset_ops.Dataset.range(3), y=dataset_ops.Dataset.range(3, 6))
    dataset = dataset_ops.Dataset.zip(x)
    expected = [Foo(x=0, y=3), Foo(x=1, y=4), Foo(x=2, y=5)]
    for i in range(3):
      self.assertAllEqual(
          self.evaluate(random_access.at(dataset, index=i)), expected[i])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=4))


class ZipGlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering global shuffling of zipped datasets (v2 only), for equal and unequal
  # input lengths.

  @combinations.generate(
      combinations.times(
          # Tested only on v2 because map v1 does not preserve cardinality.
          test_base.v2_only_combinations(),
          combinations.combine(dataset_range=[5, 6, 7]),
      )
  )
  def testZipV2SameLengthInputs(self, dataset_range: int):
    # [FUNCTION DOC]
    # Scenario: zip a doubled range dataset with a plain range dataset of the same length
    # (5, 6 or 7), prefetch, and globally shuffle with two different seeds. Asserts both outputs
    # have the full length and contain the expected pairs, the two orders differ, and in every
    # pair the first element is twice the second.
    first_dataset = dataset_ops.Dataset.range(dataset_range)
    first_dataset = first_dataset.map(lambda x: x * 2)
    second_dataset = dataset_ops.Dataset.range(dataset_range)

    dataset = dataset_ops.Dataset.zip(first_dataset, second_dataset)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    shuffle_dataset1 = global_shuffle_op._global_shuffle(dataset, seed=10)
    shuffle_dataset2 = global_shuffle_op._global_shuffle(dataset, seed=11)

    dataset_output1 = self.getDatasetOutput(
        shuffle_dataset1, requires_initialization=True
    )
    dataset_output2 = self.getDatasetOutput(
        shuffle_dataset2, requires_initialization=True
    )
    expected = [(x * 2, x) for x in range(dataset_range)]

    self.assertLen(dataset_output1, dataset_range)
    self.assertLen(dataset_output2, dataset_range)
    self.assertCountEqual(dataset_output1, expected)
    self.assertCountEqual(dataset_output2, expected)

    self.assertNotAllEqual(
        dataset_output1,
        dataset_output2,
        "Different seeds should generate different orders of outputs.",
    )

    for x_first, x_second in dataset_output1:
      self.assertEqual(x_first, x_second * 2)

    for x_first, x_second in dataset_output2:
      self.assertEqual(x_first, x_second * 2)

  @combinations.generate(
      combinations.times(
          test_base.v2_only_combinations(),
          combinations.combine(
              dataset_ranges=[(10, 8), (9, 5), (4, 7), (5, 8)]
          ),
      )
  )
  def testZipV2DifferentLengthInputs(self, dataset_ranges: Tuple[int, int]):
    # [FUNCTION DOC]
    # Scenario: like the same-length shuffle test but with two different lengths. Asserts both
    # shuffled outputs have the shorter length and contain the expected pairs, orders differ for
    # different seeds, and every pair keeps the first-is-twice-second relation.
    first_dataset = dataset_ops.Dataset.range(dataset_ranges[0])
    first_dataset = first_dataset.map(lambda x: x * 2)
    second_dataset = dataset_ops.Dataset.range(dataset_ranges[1])

    dataset = dataset_ops.Dataset.zip(first_dataset, second_dataset)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    shuffle_dataset1 = global_shuffle_op._global_shuffle(dataset, seed=10)
    shuffle_dataset2 = global_shuffle_op._global_shuffle(dataset, seed=11)

    dataset_output1 = self.getDatasetOutput(
        shuffle_dataset1, requires_initialization=True
    )
    dataset_output2 = self.getDatasetOutput(
        shuffle_dataset2, requires_initialization=True
    )
    expected = [(x * 2, x) for x in range(min(dataset_ranges))]

    self.assertLen(dataset_output1, min(dataset_ranges))
    self.assertLen(dataset_output2, min(dataset_ranges))
    self.assertCountEqual(dataset_output1, expected)
    self.assertCountEqual(dataset_output2, expected)
    self.assertNotAllEqual(
        dataset_output1,
        dataset_output2,
        "Different seeds should generate different orders of outputs.",
    )

    for x_first, x_second in dataset_output1:
      self.assertEqual(x_first, x_second * 2)

    for x_first, x_second in dataset_output2:
      self.assertEqual(x_first, x_second * 2)


class ZipGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):
  # [CLASS DOC]
  # Checkpoint test case for globally shuffled zipped datasets (v2 only).

  @combinations.generate(
      combinations.times(
          test_base.v2_only_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(10, 8), (9, 5), (4, 7), (5, 8)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testZipV2(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, int],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    # [FUNCTION DOC]
    # Scenario: for several length pairs, reshuffle and symbolic-checkpoint settings, builds a
    # zipped, prefetched, globally shuffled dataset and runs the verification function expecting
    # as many outputs as the shorter length, comparing items regardless of order when reshuffling
    # each iteration.

    def _build_dataset():
      # [FUNCTION DOC]
      # HELPER: dataset builder for the checkpoint verification.
      # Builds the zipped, prefetched and globally shuffled dataset (seed 10) and returns it with
      # default optimizations disabled and the symbolic checkpoint option applied.
      first_dataset = dataset_ops.Dataset.range(dataset_ranges[0])
      first_dataset = first_dataset.map(lambda x: x * 2)
      second_dataset = dataset_ops.Dataset.range(dataset_ranges[1])

      dataset = dataset_ops.Dataset.zip(first_dataset, second_dataset)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=min(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )


if __name__ == "__main__":
  test.main()
