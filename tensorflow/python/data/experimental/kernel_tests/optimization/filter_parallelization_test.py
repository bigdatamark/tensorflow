# [MODULE DOC]
# Tests for the filter parallelization graph optimization, which rewrites filter into a parallel
# filter when enabled (and when autotuning is on). Covers regular and legacy filter functions on
# tuples, ranges, dictionaries, sparse values, short-circuit predicates, step containers, naming,
# out-of-range signalling from a Python function, autotune setting, and checkpointing.
# No import-time side effects besides imports.
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
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


def _test_combinations():
  # [FUNCTION DOC]
  # HELPER: generates parameter combinations for the tests. Returns a list that covers both API
  # versions in eager and graph modes using the regular filter, plus version 1 combinations using
  # the legacy filter function; each combination supplies a named apply-filter callback.

  def filter_fn(dataset, predicate):
    # [FUNCTION DOC]
    # HELPER: applies the regular filter with the predicate to the dataset and returns the result.
    return dataset.filter(predicate)

  def legacy_filter_fn(dataset, predicate):
    # [FUNCTION DOC]
    # HELPER: applies the legacy-function filter with the predicate and returns the result.
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


class FilterParallelizationTest(test_base.DatasetTestBase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests of filter parallelization: each test enables the option and asserts via an
  # assert-next check that a parallel filter appears, then checks the produced elements.

  def enableFilterParallelization(self, dataset):
    # [FUNCTION DOC]
    # HELPER: returns the dataset with the filter parallelization optimization option turned on.
    options = options_lib.Options()
    options.experimental_optimization.filter_parallelization = True
    return dataset.with_options(options)

  @combinations.generate(_test_combinations())
  def testFilterDataset(self, apply_filter):
    # [FUNCTION DOC]
    # Builds a dataset from three numeric components, squares them, repeats it, filters by whether
    # the first component modulo a value is zero, and verifies output shapes and values for
    # several repeat/modulus pairs, including an empty dataset, ending with an out-of-range error.
    components = (np.arange(7, dtype=np.int64),
                  np.array([[1, 2, 3]], dtype=np.int64) *
                  np.arange(7, dtype=np.int64)[:, np.newaxis],
                  np.array(37.0, dtype=np.float64) * np.arange(7))

    def _map_fn(x, y, z):
      # [FUNCTION DOC]
      # HELPER: map callback returning the elementwise square of each of the three components.
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    def do_test(count, modulus):  # pylint: disable=missing-docstring
      # [FUNCTION DOC]
      # HELPER: runs one scenario. Inputs: repeat count and modulus. Builds, squares, repeats and
      # filters the dataset with parallelization enabled (asserting a parallel filter), checks the
      # component shapes, then reads the expected rows and asserts values and final exhaustion.
      dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
          _map_fn).repeat(count)
      dataset = self.enableFilterParallelization(dataset)
      dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
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
    # Filters range 0..3 keeping elements whose value modulo 3 is not 2; expects 0, 1, 3 and a
    # parallel filter in the pipeline.
    dataset = dataset_ops.Dataset.range(4)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset,
                           lambda x: math_ops.not_equal(math_ops.mod(x, 3), 2))
    self.assertDatasetProduces(dataset, expected_output=[0, 1, 3])

  @combinations.generate(_test_combinations())
  def testFilterDict(self, apply_filter):
    # [FUNCTION DOC]
    # Maps range 0..9 to dictionaries of double and square, filters on the square being even, maps
    # to the sum of the two fields, and asserts the expected sums.
    dataset = dataset_ops.Dataset.range(10).map(
        lambda x: {"foo": x * 2, "bar": x**2})
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset, lambda d: math_ops.equal(d["bar"] % 2, 0))
    dataset = dataset.map(lambda d: d["foo"] + d["bar"])
    self.assertDatasetProduces(
        dataset,
        expected_output=[(i * 2 + i**2) for i in range(10) if not (i**2) % 2])

  @combinations.generate(_test_combinations())
  def testUseStepContainerInFilter(self, apply_filter):
    # [FUNCTION DOC]
    # Filters two rows with a predicate that uses a map-over-elements helper (which makes it
    # stateful for the legacy filter). Expects a parallel filter for the regular filter but a plain
    # filter for the legacy one, and only the first row (sum of squares equals 14) to pass.
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    # Define a predicate that returns true for the first element of
    # the sequence and not the second, and uses `tf.map_fn()`.
    def _predicate(xs):
      # [FUNCTION DOC]
      # Predicate callback: squares each entry of the row via a map helper, sums them, and returns
      # whether the sum equals 14.
      squared_xs = map_fn.map_fn(lambda x: x * x, xs)
      summed = math_ops.reduce_sum(squared_xs)
      return math_ops.equal(summed, 1 + 4 + 9)

    dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
    dataset = self.enableFilterParallelization(dataset)
    # Calling `legacy_filter_fn` with `_predicate` makes the predicate passed to
    # `Filter` stateful and therefore not parallelizable.
    if repr(apply_filter) != "legacy_filter_fn":
      dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    else:
      dataset = dataset.apply(testing.assert_next(["Filter"]))
    dataset = apply_filter(dataset, _predicate)
    self.assertDatasetProduces(dataset, expected_output=[input_data[0]])

  @combinations.generate(_test_combinations())
  def testSparse(self, apply_filter):
    # [FUNCTION DOC]
    # Maps range 0..9 to pairs of (sparse value, index), filters on the even index, drops the index
    # and asserts the sparse values for indices 0, 2, 4, 6, 8, with a parallel filter in the plan.

    def _map_fn(i):
      # [FUNCTION DOC]
      # HELPER: map callback turning an integer into a pair of a one-entry sparse value (whose
      # value is the integer) and the integer itself.
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1])), i

    def _filter_fn(_, i):
      # [FUNCTION DOC]
      # HELPER: predicate on a pair ignoring the sparse part; true when the index is even.
      return math_ops.equal(i % 2, 0)

    dataset = dataset_ops.Dataset.range(10).map(_map_fn)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset, _filter_fn)
    dataset = dataset.map(lambda x, i: x)
    self.assertDatasetProduces(
        dataset, expected_output=[_map_fn(i * 2)[0] for i in range(5)])

  @combinations.generate(_test_combinations())
  def testShortCircuit(self, apply_filter):
    # [FUNCTION DOC]
    # Zips range 0..9 with an infinite stream of true values and filters by the second component
    # (a predicate that just returns an input); expects all ten pairs and a parallel filter.
    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(10),
         dataset_ops.Dataset.from_tensors(True).repeat(None)))
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset, lambda x, y: y)
    self.assertDatasetProduces(
        dataset, expected_output=[(i, True) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testParallelFilters(self, apply_filter):
    # [FUNCTION DOC]
    # Creates ten independent iterators over the same even-number filter of range 0..9 and
    # evaluates the first element of all at once; asserts each yields 0.
    dataset = dataset_ops.Dataset.range(10)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = apply_filter(dataset, lambda x: math_ops.equal(x % 2, 0))
    next_elements = [self.getNext(dataset) for _ in range(10)]
    self.assertEqual([0 for _ in range(10)],
                     self.evaluate(
                         [next_element() for next_element in next_elements]))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Applies an always-true filter with an explicit name and asserts the single element 42 is
    # produced.
    dataset = dataset_ops.Dataset.from_tensors(42).filter(
        lambda x: True, name="filter")
    self.assertDatasetProduces(dataset, [42])

  @combinations.generate(test_base.default_test_combinations())
  def testInputOutOfRange(self):
    # [FUNCTION DOC]
    # Filters with a predicate calling a Python function that signals end of iteration; asserts
    # reading raises an invalid-argument error (the stop signal is not treated as end of input).

    def py_fn(_):
      # [FUNCTION DOC]
      # HELPER: Python callback that ignores its input and always raises a stop-iteration signal.
      raise StopIteration()

    dataset = dataset_ops.Dataset.range(5)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    dataset = dataset.filter(
        lambda x: script_ops.py_func(py_fn, [x], dtypes.bool, stateful=False))
    get_next = self.getNext(dataset)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(autotune=[False, True])))
  def testAutotuneSetting(self, autotune):
    # [FUNCTION DOC]
    # Parameterized on whether autotuning is enabled. Enables filter parallelization and sets the
    # autotune option; expects a parallel filter when autotune is on and a plain filter when off;
    # output must be 0, 1, 3.
    dataset = dataset_ops.Dataset.range(4)
    options = options_lib.Options()
    options.experimental_optimization.filter_parallelization = True
    options.autotune.enabled = autotune
    dataset = dataset.with_options(options)
    if autotune:
      dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    else:
      dataset = dataset.apply(testing.assert_next(["Filter"]))
    dataset = dataset.filter(
        lambda x: math_ops.not_equal(math_ops.mod(x, 3), 2))
    self.assertDatasetProduces(dataset, expected_output=[0, 1, 3])


class FilterCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint (save/restore) tests for datasets using parallel filter, built with helper
  # builders for range, dictionary and sparse pipelines.

  def enableFilterParallelization(self, dataset):
    # [FUNCTION DOC]
    # HELPER: returns the dataset with the filter parallelization option turned on.
    options = options_lib.Options()
    options.experimental_optimization.filter_parallelization = True
    return dataset.with_options(options)

  def _build_filter_range_graph(self, div):
    # [FUNCTION DOC]
    # HELPER: builds range 0..99 with parallel filter enabled (asserted by next-op check) that
    # drops elements whose value modulo div equals 2; returns the dataset.
    dataset = dataset_ops.Dataset.range(100)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    return dataset.filter(
        lambda x: math_ops.not_equal(math_ops.mod(x, div), 2))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Runs the checkpoint verifier on the range filter pipeline (divisor 3) with the count of
    # elements not congruent to 2 modulo 3 among 0..99 as the expected output count.
    div = 3
    num_outputs = sum(x % 3 != 2 for x in range(100))
    verify_fn(self, lambda: self._build_filter_range_graph(div), num_outputs)

  def _build_filter_dict_graph(self):
    # [FUNCTION DOC]
    # HELPER: builds range 0..9 mapped to dictionaries of double and square, with parallel filter
    # on a even-square predicate, then mapped to the sum of the fields; returns the dataset.
    # pylint: disable=g-long-lambda
    dataset = dataset_ops.Dataset.range(10).map(lambda x: {
        "foo": x * 2,
        "bar": x**2
    })
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    return dataset.filter(lambda d: math_ops.equal(d["bar"] % 2, 0)).map(
        lambda d: d["foo"] + d["bar"])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testDict(self, verify_fn):
    # [FUNCTION DOC]
    # Runs the checkpoint verifier on the dictionary pipeline, expecting as many outputs as
    # elements of 0..9 whose square is even.
    num_outputs = sum((x**2) % 2 == 0 for x in range(10))
    # pylint: disable=unnecessary-lambda
    verify_fn(self, lambda: self._build_filter_dict_graph(), num_outputs)

  def _build_sparse_filter(self):
    # [FUNCTION DOC]
    # HELPER: builds range 0..9 mapped to pairs of a sparse tensor and the index, with parallel
    # filter on even indices, then drops the index; returns the dataset of sparse tensors.

    def _map_fn(i):
      # [FUNCTION DOC]
      # HELPER: map callback returning a pair of a one-entry sparse tensor holding the integer
      # and the integer itself.
      return sparse_tensor.SparseTensor(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[1, 1]), i

    def _filter_fn(_, i):
      # [FUNCTION DOC]
      # HELPER: predicate ignoring the sparse part; true when the index is even.
      return math_ops.equal(i % 2, 0)

    dataset = dataset_ops.Dataset.range(10).map(_map_fn)
    dataset = self.enableFilterParallelization(dataset)
    dataset = dataset.apply(testing.assert_next(["ParallelFilter"]))
    return dataset.filter(_filter_fn).map(lambda x, i: x)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testSparse(self, verify_fn):
    # [FUNCTION DOC]
    # Runs the checkpoint verifier on the sparse filter pipeline expecting five outputs.
    # pylint: disable=unnecessary-lambda
    verify_fn(self, lambda: self._build_sparse_filter(), num_outputs=5)


if __name__ == "__main__":
  test.main()
