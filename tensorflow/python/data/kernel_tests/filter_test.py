# [MODULE DOC]
# Tests for the filter transformation of the tf.data API, including the legacy-function variant
# in the older API version. Contains functional tests (values, shapes, dictionaries, sparse
# tensors, step containers, short-circuiting, parallel iterators, naming, error context), checkpoint
# tests, and tests of interaction with global shuffling (which filter does not support upstream of
# the shuffle). A module-level helper builds the parameter combinations over both filter flavors.
# No import-time side effects; running as a script starts the TensorFlow test runner.
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
"""Tests for `tf.data.Dataset.filter()`."""

from typing import Callable

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _test_combinations():
  # [FUNCTION DOC]
  # HELPER: supplies the parameter combinations shared by the functional filter tests.
  # Takes no inputs. Returns combinations that supply an apply-filter callback: the regular
  # filter in both API versions and both eager/graph modes, plus the legacy-function filter in
  # API version 1 only (both modes).

  def filter_fn(dataset, predicate):
    # [FUNCTION DOC]
    # HELPER: adapter for the combinations; applies the regular filter transformation with the
    # given predicate to the dataset and returns the filtered dataset.
    return dataset.filter(predicate)

  def legacy_filter_fn(dataset, predicate):
    # [FUNCTION DOC]
    # HELPER: adapter for the combinations; applies the legacy-function filter transformation
    # with the given predicate to the dataset and returns the filtered dataset.
    return dataset.filter_with_legacy_function(predicate)

  filter_combinations = combinations.combine(
      tf_api_version=[1, 2],
      mode=["eager", "graph"],
      apply_filter=combinations.NamedObject("filter_fn", filter_fn))

  legacy_filter_combinations = combinations.combine(
      tf_api_version=1,
      mode=["eager", "graph"],
      apply_filter=combinations.NamedObject("legacy_filter_fn",
                                            legacy_filter_fn))

  return filter_combinations + legacy_filter_combinations


class FilterTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests of the filter transformation, parameterized over the apply-filter callback
  # (regular and legacy filter) from the module combinations helper. No shared fixtures.

  @combinations.generate(_test_combinations())
  def testFilterDataset(self, apply_filter):
    # [FUNCTION DOC]
    # Scenario: three tensor components (integers, integer vectors, floats) of 7 slices are
    # squared by a map and repeated. A filter keeps elements whose first component is divisible by
    # a modulus. Checks legacy output shapes equal the component slice shapes, that the expected
    # squared values come out in order for each repetition, and that reading past the end raises
    # an out-of-range error. Runs with (14 repeats, modulus 2), (4, 18) and an empty dataset
    # (0, 1). Input is the apply-filter callback. Returns nothing.
    components = (np.arange(7, dtype=np.int64),
                  np.array([[1, 2, 3]], dtype=np.int64) *
                  np.arange(7, dtype=np.int64)[:, np.newaxis],
                  np.array(37.0, dtype=np.float64) * np.arange(7))

    def _map_fn(x, y, z):
      # [FUNCTION DOC]
      # HELPER: map function for testFilterDataset; returns the element-wise square of each of
      # the three components as a tuple.
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    def do_test(count, modulus):  # pylint: disable=missing-docstring
      # [FUNCTION DOC]
      # HELPER: runs one scenario of testFilterDataset. Builds the squared dataset repeated
      # count times, filters to elements whose first component is divisible by modulus, then
      # verifies shapes, the expected values per repetition, and the out-of-range error at the
      # end. Inputs are the repeat count and the modulus (both integers). Returns nothing.
      dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
          _map_fn).repeat(count)
      # pylint: disable=g-long-lambda
      dataset = apply_filter(
          dataset,
          lambda x, _y, _z: math_ops.equal(math_ops.mod(x, modulus), 0))
      # pylint: enable=g-long-lambda
      self.assertEqual(
          [c.shape[1:] for c in components],
          [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
      get_next = self.getNext(dataset)
      for _ in range(count):
        for i in [x for x in range(7) if x**2 % modulus == 0]:
          result = self.evaluate(get_next())
          for component, result_component in zip(components, result):
            self.assertAllEqual(component[i]**2, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

    do_test(14, 2)
    do_test(4, 18)

    # Test an empty dataset.
    do_test(0, 1)

  @combinations.generate(_test_combinations())
  def testFilterRange(self, apply_filter):
    # [FUNCTION DOC]
    # Scenario: a range of 4 filtered to drop elements whose remainder modulo 3 is 2. Expects
    # 0, 1 and 3. Input is the apply-filter callback. Returns nothing.
    dataset = dataset_ops.Dataset.range(4)
    dataset = apply_filter(dataset,
                           lambda x: math_ops.not_equal(math_ops.mod(x, 3), 2))
    self.assertDatasetProduces(dataset, expected_output=[0, 1, 3])

  @combinations.generate(_test_combinations())
  def testFilterDict(self, apply_filter):
    # [FUNCTION DOC]
    # Scenario: a range of 10 mapped to mappings with a doubled value and a squared value; the
    # filter keeps elements whose squared value is even; a final map sums the two values. Expects
    # the sums for the even-squared elements only. Input is the apply-filter callback. Returns
    # nothing.
    dataset = dataset_ops.Dataset.range(10).map(
        lambda x: {"foo": x * 2, "bar": x**2})
    dataset = apply_filter(dataset, lambda d: math_ops.equal(d["bar"] % 2, 0))
    dataset = dataset.map(lambda d: d["foo"] + d["bar"])
    self.assertDatasetProduces(
        dataset,
        expected_output=[(i * 2 + i**2) for i in range(10) if not (i**2) % 2])

  @combinations.generate(_test_combinations())
  def testUseStepContainerInFilter(self, apply_filter):
    # [FUNCTION DOC]
    # Scenario: a dataset of two integer vectors filtered with a predicate that itself uses a
    # map-over-elements op (a step container) inside the filter function. The predicate keeps only
    # the vector whose sum of squares is 14, so only the first vector is expected. Input is the
    # apply-filter callback. Returns nothing.
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    # Define a predicate that returns true for the first element of
    # the sequence and not the second, and uses `tf.map_fn()`.
    def _predicate(xs):
      # [FUNCTION DOC]
      # HELPER: filter predicate for this test; squares the entries of the vector with a map
      # over elements, sums them, and returns whether the sum equals 14 (a boolean scalar tensor).
      squared_xs = map_fn.map_fn(lambda x: x * x, xs)
      summed = math_ops.reduce_sum(squared_xs)
      return math_ops.equal(summed, 1 + 4 + 9)

    dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
    dataset = apply_filter(dataset, _predicate)
    self.assertDatasetProduces(dataset, expected_output=[input_data[0]])

  @combinations.generate(_test_combinations())
  def testSparse(self, apply_filter):
    # [FUNCTION DOC]
    # Scenario: a range of 10 mapped to pairs of a 1x1 sparse value (holding the number) and the
    # number itself; the filter keeps pairs with even numbers; a final map keeps the sparse part.
    # Expects the sparse values for 0, 2, 4, 6, 8. Input is the apply-filter callback. Returns
    # nothing.

    def _map_fn(i):
      # [FUNCTION DOC]
      # HELPER: map function; turns a number into a pair of a 1x1 sparse tensor value holding that
      # number and the number itself.
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1])), i

    def _filter_fn(_, i):
      # [FUNCTION DOC]
      # HELPER: filter predicate; ignores the sparse part and returns whether the number is even.
      return math_ops.equal(i % 2, 0)

    dataset = dataset_ops.Dataset.range(10).map(_map_fn)
    dataset = apply_filter(dataset, _filter_fn)
    dataset = dataset.map(lambda x, i: x)
    self.assertDatasetProduces(
        dataset, expected_output=[_map_fn(i * 2)[0] for i in range(5)])

  @combinations.generate(_test_combinations())
  def testShortCircuit(self, apply_filter):
    # [FUNCTION DOC]
    # Scenario: a range of 10 zipped with an infinitely repeated true value, filtered by a
    # predicate that just returns the second component. All ten pairs must pass through. Input is
    # the apply-filter callback. Returns nothing.
    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(10),
         dataset_ops.Dataset.from_tensors(True).repeat(None)))
    dataset = apply_filter(dataset, lambda x, y: y)
    self.assertDatasetProduces(
        dataset, expected_output=[(i, True) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testParallelFilters(self, apply_filter):
    # [FUNCTION DOC]
    # Scenario: a range of 10 filtered to even numbers; ten independent iterators are created and
    # each reads its first element in one evaluation. Asserts all ten first elements are zero.
    # Input is the apply-filter callback. Returns nothing.
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_filter(dataset, lambda x: math_ops.equal(x % 2, 0))
    next_elements = [self.getNext(dataset) for _ in range(10)]
    self.assertEqual([0 for _ in range(10)],
                     self.evaluate(
                         [next_element() for next_element in next_elements]))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: filters a single-element dataset (42) with an always-true predicate and an
    # explicit operation name; asserts the dataset still produces 42. Returns nothing.
    dataset = dataset_ops.Dataset.from_tensors(42).filter(
        lambda x: True, name="filter")
    self.assertDatasetProduces(dataset, [42])

  @combinations.generate(test_base.default_test_combinations())
  def testPredicateFailWithErrorContext(self):
    # [FUNCTION DOC]
    # Scenario: the predicate performs an integer division by zero. Reading the first element
    # must raise an invalid-argument error whose message mentions the error in the user-defined
    # function passed to the transformation and the root iterator path. Returns nothing.
    dataset = dataset_ops.Dataset.from_tensors(42).filter(
        lambda x: (x // 0) > 0, name="filter")
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r".*Error in user-defined function passed to .* transformation with "
        r"iterator: Iterator::Root::.*"):
      self.evaluate(get_next())


class FilterCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for the filter transformation; datasets are built by helper methods and
  # verified by the checkpoint test base class (save, restore, continue iterating).

  def _build_filter_range_dataset(self, div, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset for the range checkpoint test. A range of 100 filtered to drop
    # elements whose remainder modulo div equals 2. If options (optional) is given, applies it.
    # Returns the dataset.
    dataset = dataset_ops.Dataset.range(100).filter(
        lambda x: math_ops.not_equal(math_ops.mod(x, div), 2))
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Checkpoint scenario: with the symbolic-checkpoint option set per parameter, calls the
    # verification callback with a factory for the filtered range (divisor 3) and the expected
    # number of outputs (count of numbers below 100 whose remainder modulo 3 is not 2). Returns
    # nothing.
    div = 3
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    num_outputs = sum(x % 3 != 2 for x in range(100))
    verify_fn(self, lambda: self._build_filter_range_dataset(div, options),
              num_outputs)

  def _build_filter_dict_dataset(self):
    # [FUNCTION DOC]
    # HELPER: builds the dataset for the dictionary checkpoint test: range of 10 mapped to
    # mappings (doubled, squared), filtered to even squares, then mapped to the sum of the two
    # values. Takes no inputs; returns the dataset.
    return dataset_ops.Dataset.range(10).map(lambda x: {
        "foo": x * 2,
        "bar": x**2
    }).filter(lambda d: math_ops.equal(d["bar"] % 2, 0)).map(
        lambda d: d["foo"] + d["bar"])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testDict(self, verify_fn):
    # [FUNCTION DOC]
    # Checkpoint scenario for the dictionary dataset: calls the verification callback with the
    # dataset factory and the expected count of outputs (numbers below 10 with even square).
    # Returns nothing.
    num_outputs = sum((x**2) % 2 == 0 for x in range(10))
    verify_fn(self, self._build_filter_dict_dataset, num_outputs)

  def _build_sparse_filter_dataset(self):
    # [FUNCTION DOC]
    # HELPER: builds the dataset for the sparse checkpoint test: range of 10 mapped to a pair of
    # a 1x1 sparse tensor and the number, filtered to even numbers, then reduced to the sparse
    # part. Takes no inputs; returns the dataset (5 elements).

    def _map_fn(i):
      # [FUNCTION DOC]
      # HELPER: map function; returns a pair of a 1x1 sparse tensor holding the number and the
      # number itself.
      return sparse_tensor.SparseTensor(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[1, 1]), i

    def _filter_fn(_, i):
      # [FUNCTION DOC]
      # HELPER: filter predicate; ignores the sparse part and returns whether the number is even.
      return math_ops.equal(i % 2, 0)

    return dataset_ops.Dataset.range(10).map(_map_fn).filter(_filter_fn).map(
        lambda x, i: x)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testSparse(self, verify_fn):
    # [FUNCTION DOC]
    # Checkpoint scenario for the sparse dataset: calls the verification callback with the
    # dataset factory and an expected output count of 5. Returns nothing.
    verify_fn(self, self._build_sparse_filter_dataset, num_outputs=5)


class FilterGlobalShuffleTest(
    test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the interaction of filter with global shuffling: shuffle-then-filter works, while
  # filter-then-shuffle is rejected because filter does not support random access.

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleFilter(self):
    # [FUNCTION DOC]
    # Scenario: a range of 100 globally shuffled and then filtered to even numbers. Asserts the
    # dataset produces the even numbers 0 to 98 in any order, with explicit initialization
    # required. Returns nothing.
    dataset = dataset_ops.Dataset.range(100)
    dataset = global_shuffle_op._global_shuffle(dataset)
    dataset = dataset.filter(lambda x: math_ops.equal(x % 2, 0))
    self.assertDatasetProduces(
        dataset,
        list(range(0, 100, 2)),
        requires_initialization=True,
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFilterShuffle(self):
    # [FUNCTION DOC]
    # Scenario: a range of 100 filtered to even numbers and then globally shuffled. Reading the
    # output must raise a failed-precondition error saying global shuffle requires all upstream
    # transformations to be compatible with random access. Returns nothing.
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.filter(lambda x: math_ops.equal(x % 2, 0))
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "`global_shuffle` requires all upstream transformations be compatible "
        "with random access."):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)


class FilterGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for a global-shuffle followed by filter pipeline, parameterized over the
  # reshuffle-each-iteration and symbolic-checkpoint flags.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testShuffleFilter(
      self,
      verify_fn: Callable[..., None],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Checkpoint scenario: builds a range of 10 globally shuffled (fixed seed 42, reshuffle flag
    # per parameter) and filtered to even numbers, optionally with the symbolic-checkpoint
    # option. Calls the verification callback with the factory, expecting 5 outputs; order is
    # ignored only when reshuffling each iteration. Returns nothing.

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: factory for the dataset under test; builds the shuffled and filtered range,
      # applies the symbolic-checkpoint option when enabled, and returns the dataset.
      dataset = dataset_ops.Dataset.range(10)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      dataset = dataset.filter(lambda x: math_ops.equal(x % 2, 0))
      if symbolic_checkpoint:
        options = options_lib.Options()
        options.experimental_symbolic_checkpoint = symbolic_checkpoint
        dataset = dataset.with_options(options)
      return dataset

    verify_fn(
        self,
        _build_dataset,
        num_outputs=5,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
