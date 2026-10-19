# [MODULE DOC]
# Tests for the experimental parallel interleave dataset transformation (deprecated in favor of
# the standard interleave with parallelism). Contains a test case that exercises sloppy and
# deterministic modes, empty inputs, sparse inputs, error propagation, shutdown races and
# determinism options, plus a checkpoint test case that verifies save/restore of the iterator.
# A pure-Python reference interleave implementation is used to compute expected outputs.
# No import-time side effects; the module runs the test main when executed directly.
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
"""Tests for `tf.data.experimental.parallel_interleave()`."""
import itertools
import math
import threading
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class ParallelInterleaveTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the parallel interleave transformation. Holds per-test state: an optional error to
  # inject from the mapping function, a repeat count, and two mappings from the input values
  # (4, 5, 6) to threading primitives (semaphores for reads, events for writes) that let tests
  # deterministically control when map worker threads may produce elements.

  def setUp(self):
    # [FUNCTION DOC]
    # Resets the injected error to nothing, sets the repeat count to two and creates, for each of
    # the input values 4, 5 and 6, a read semaphore (initially zero) and an unset write event.

    self.error = None
    self.repeat_count = 2

    # Set up threading events used to sequence when items are produced that
    # are subsequently interleaved. These events allow us to deterministically
    # simulate slowdowns and force sloppiness.
    self.read_coordination_events = {}
    self.write_coordination_events = {}
    # input values [4, 5, 6] are the common case for the tests; set defaults
    for i in range(4, 7):
      self.read_coordination_events[i] = threading.Semaphore(0)
      self.write_coordination_events[i] = threading.Event()

  def dataset_fn(self, input_values, cycle_length, block_length, sloppy,
                 buffer_output_elements, prefetch_input_elements):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the parallel interleave tests.
    # Builds a dataset from the given input values, repeated by the repeat count, and applies
    # parallel interleave with the given cycle length, block length, sloppy flag, output buffer
    # size and input prefetch count. Each input value x yields x copies of x, which are mapped
    # through a Python function that blocks on the write event for x, signals the read semaphore
    # for x, optionally raises the injected error (once), and returns the square of x.
    # Returns the resulting dataset.

    def map_py_fn(x):
      # [FUNCTION DOC]
      # Blocks until the write event for x is set, clears it, releases the read semaphore for x,
      # raises and resets the injected error if one is pending, otherwise returns x squared.
      self.write_coordination_events[x].wait()
      self.write_coordination_events[x].clear()
      self.read_coordination_events[x].release()
      if self.error:
        err = self.error
        self.error = None
        raise err  # pylint: disable=raising-bad-type
      return x * x

    def map_fn(x):
      # [FUNCTION DOC]
      # Wraps the Python coordination function as a graph op preserving the dtype of x.
      return script_ops.py_func(map_py_fn, [x], x.dtype)

    def interleave_fn(x):
      # [FUNCTION DOC]
      # Turns an element x into a dataset repeating x, x times, mapped through the
      # coordination function. Returns that dataset.
      dataset = dataset_ops.Dataset.from_tensors(x)
      dataset = dataset.repeat(x)
      return dataset.map(map_fn)

    return dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
        self.repeat_count).apply(
            interleave_ops.parallel_interleave(
                interleave_fn, cycle_length, block_length, sloppy,
                buffer_output_elements, prefetch_input_elements))

  def _interleave(self, lists, cycle_length, block_length):
    # [FUNCTION DOC]
    # HELPER: reference interleave implementation used to compute expected outputs in tests.
    # Generator. Takes a list of lists, a cycle length (number of concurrently open inputs) and
    # a block length (elements taken from each open input per turn). Round-robins over the open
    # iterators taking up to block length items each, replacing exhausted iterators with the
    # next unopened list. Yields the interleaved elements one by one.
    """Python implementation of interleave used for testing."""
    num_open = 0

    # `all_iterators` acts as a queue of iterators over each element of `lists`.
    all_iterators = [iter(l) for l in lists]

    # `open_iterators` are the iterators whose elements are currently being
    # interleaved.
    open_iterators = []
    for i in range(cycle_length):
      if all_iterators:
        open_iterators.append(all_iterators.pop(0))
        num_open += 1
      else:
        open_iterators.append(None)

    while num_open or all_iterators:
      for i in range(cycle_length):
        if open_iterators[i] is None:
          if all_iterators:
            open_iterators[i] = all_iterators.pop(0)
            num_open += 1
          else:
            continue
        for _ in range(block_length):
          try:
            yield next(open_iterators[i])
          except StopIteration:
            open_iterators[i] = None
            num_open -= 1
            break

  @combinations.generate(
      combinations.times(
          combinations.combine(
              input_lists=[[[4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6],
                            [4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]]],
              expected_elements=[[
                  4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 5, 5,
                  5, 5, 5, 6, 6, 6, 6, 6, 6
              ]],
              cycle_length=1,
              block_length=1) +
          combinations.combine(
              input_lists=[[[4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6],
                            [4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]]],
              expected_elements=[[
                  4, 5, 4, 5, 4, 5, 4, 5, 5, 6, 6, 4, 6, 4, 6, 4, 6, 4, 6, 5, 6,
                  5, 6, 5, 6, 5, 6, 5, 6, 6
              ]],
              cycle_length=2,
              block_length=1) + combinations.combine(
                  input_lists=[[[4] * 4, [5] * 5, [6] * 6] * 2],
                  expected_elements=[[
                      4, 4, 5, 5, 4, 4, 5, 5, 5, 6, 6, 4, 4, 6, 6, 4, 4, 6, 6,
                      5, 5, 6, 6, 5, 5, 6, 6, 5, 6, 6
                  ]],
                  cycle_length=2,
                  block_length=2) +
          combinations.combine(
              input_lists=[[[4, 4, 4, 4], [], [6, 6, 6, 6, 6, 6], [4, 4, 4, 4],
                            [], [6, 6, 6, 6, 6, 6]]],
              expected_elements=[[
                  4, 4, 6, 4, 6, 4, 6, 6, 4, 6, 4, 6, 4, 4, 6, 6, 6, 6, 6, 6
              ]],
              cycle_length=2,
              block_length=1)))
  def testPythonImplementation(self, input_lists, expected_elements,
                               cycle_length, block_length):
    # [FUNCTION DOC]
    # Sanity-checks the Python reference interleave: for several parameter combinations (cycle
    # length one or two, block length one or two, with and without empty inputs) compares the
    # generated sequence element by element against the expected sequence, asserting equality
    # and that both have the same length.
    for index, (expected, produced) in enumerate(
        itertools.zip_longest(
            expected_elements,
            self._interleave(input_lists, cycle_length, block_length))):
      self.assertEqual(expected, produced, "Values differ at %s. %s != %s" %
                       (index, expected, produced))

  def _clear_coordination_events(self):
    # [FUNCTION DOC]
    # HELPER: resets coordination state between phases of a test.
    # For input values 4 to 6, replaces the read semaphore with a fresh zero-count one and
    # clears the write event. Returns nothing.
    for i in range(4, 7):
      self.read_coordination_events[i] = threading.Semaphore(0)
      self.write_coordination_events[i].clear()

  def _allow_all_map_threads(self):
    # [FUNCTION DOC]
    # HELPER: lets all map worker threads proceed.
    # Sets the write event for each of the input values 4 to 6. Returns nothing.
    for i in range(4, 7):
      self.write_coordination_events[i].set()

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(sloppy=[False, True])))
  def testEmptyInput(self, sloppy):
    # [FUNCTION DOC]
    # Scenario: parallel interleave over an empty int64 input, parameterized on sloppy mode.
    # Expects the first fetch of an element to raise an out-of-range error.
    # Empty input.
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([]),
            cycle_length=2,
            block_length=3,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(sloppy=[False, True])))
  def _testNonEmptyInputIntoEmptyOutputs(self, sloppy):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the leading underscore means the test runner does not collect it, so this
    # test is effectively disabled.
    # Scenario: input values of zero produce empty inner datasets; expects the first fetch to
    # raise an out-of-range error, for both sloppy modes.
    # Non-empty input leading to empty output.
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([0, 0, 0]),
            cycle_length=2,
            block_length=3,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(sloppy=[False, True])))
  def testTooManyReaders(self, sloppy=False):
    # [FUNCTION DOC]
    # Scenario: cycle length (16) exceeds the number of input elements (3 values repeated).
    # Reads 30 elements and asserts they match, ignoring order, the reference interleave output
    # (each value v repeated v times, block length two).

    def interleave_fn(x):
      # [FUNCTION DOC]
      # Returns a dataset repeating element x, x times (x cast to a 64-bit integer count).
      dataset = dataset_ops.Dataset.from_tensors(x)
      dataset = dataset.repeat(math_ops.cast(x, dtype=dtypes.int64))
      return dataset

    dataset = dataset_ops.Dataset.from_tensor_slices([4, 5, 6])
    dataset = dataset.repeat(self.repeat_count)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(
            interleave_fn, cycle_length=16, block_length=2, sloppy=sloppy))
    get_next = self.getNext(dataset)
    output_values = []
    for _ in range(30):
      output_values.append(self.evaluate(get_next()))

    expected_values = self._interleave(
        [[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 1, 2)
    self.assertCountEqual(output_values, expected_values)

  @combinations.generate(test_base.default_test_combinations())
  def testSparse(self):
    # [FUNCTION DOC]
    # Scenario: a range dataset is mapped to 2x2 sparse tensors (diagonal values i and -i) and
    # interleaved with cycle length one, converting each to dense and slicing rows.
    # Asserts rows come out as expected for all ten inputs, then end of data is signaled.
    def _map_fn(i):
      # [FUNCTION DOC]
      # Builds a 2x2 sparse tensor with entries i at (0,0) and -i at (1,1).
      return sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      # [FUNCTION DOC]
      # Densifies sparse tensor x and returns a dataset of its rows.
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).apply(
        interleave_ops.parallel_interleave(_interleave_fn, cycle_length=1))
    get_next = self.getNext(dataset)

    for i in range(10):
      for j in range(2):
        expected = [i, 0] if j % 2 == 0 else [0, -i]
        self.assertAllEqual(expected, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testErrorsInInputFn(self):
    # [FUNCTION DOC]
    # Scenario: the input dataset is mapped through a Python function that raises for value 5,
    # then repeated and parallel-interleaved (cycle length two, block length one, not sloppy).
    # Walks the reference interleave sequence: at each position of value 5 expects an
    # invalid-argument error, otherwise the matching element; finally expects out-of-range.

    def map_py_fn(x):
      # [FUNCTION DOC]
      # Raises a value error when x is 5, otherwise returns x unchanged.
      if x == 5:
        raise ValueError()
      return x

    def map_fn(x):
      # [FUNCTION DOC]
      # Wraps the Python function as a graph op preserving x's dtype.
      return script_ops.py_func(map_py_fn, [x], x.dtype)

    def interleave_fn(x):
      # [FUNCTION DOC]
      # Returns a dataset repeating x, x times.
      dataset = dataset_ops.Dataset.from_tensors(x)
      dataset = dataset.repeat(x)
      return dataset

    def dataset_fn(input_values, cycle_length, block_length, sloppy,
                   buffer_output_elements, prefetch_input_elements):
      # [FUNCTION DOC]
      # Builds the dataset under test: input values mapped through the failing function,
      # repeated, then parallel-interleaved with the given settings. Returns the dataset.
      return dataset_ops.Dataset.from_tensor_slices(input_values).map(
          map_fn).repeat(self.repeat_count).apply(
              interleave_ops.parallel_interleave(
                  interleave_fn, cycle_length, block_length, sloppy,
                  buffer_output_elements, prefetch_input_elements))

    next_element = self.getNext(
        dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=False,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5], [6] * 6] * self.repeat_count, 2, 1)):
      if expected_element == 5:
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(next_element())
      else:
        actual_element = self.evaluate(next_element())
        self.assertEqual(
            expected_element, actual_element,
            "At index %s: %s expected, got: %s" % (i, expected_element,
                                                   actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testErrorsInInterleaveFn(self):
    # [FUNCTION DOC]
    # Scenario: the interleave function itself calls a Python function that raises for value 5.
    # Same verification as the input-function error test: invalid-argument error exactly where
    # value 5 would appear, matching elements elsewhere, then out-of-range at the end.

    def map_py_fn(x):
      # [FUNCTION DOC]
      # Raises a value error when x is 5, otherwise returns x unchanged.
      if x == 5:
        raise ValueError()
      return x

    def interleave_fn(x):
      # [FUNCTION DOC]
      # Computes the repeat count through the possibly failing Python function and returns a
      # dataset repeating x that many times.
      dataset = dataset_ops.Dataset.from_tensors(x)
      y = script_ops.py_func(map_py_fn, [x], x.dtype)
      dataset = dataset.repeat(y)
      return dataset

    def dataset_fn(input_values, cycle_length, block_length, sloppy,
                   buffer_output_elements, prefetch_input_elements):
      # [FUNCTION DOC]
      # Builds the dataset under test: input values repeated and parallel-interleaved with the
      # given settings. Returns the dataset.
      return dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
          self.repeat_count).apply(
              interleave_ops.parallel_interleave(
                  interleave_fn, cycle_length, block_length, sloppy,
                  buffer_output_elements, prefetch_input_elements))

    next_element = self.getNext(
        dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=False,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5], [6] * 6] * self.repeat_count, 2, 1)):
      if expected_element == 5:
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(next_element())
      else:
        actual_element = self.evaluate(next_element())
        self.assertEqual(
            expected_element, actual_element,
            "At index %s: %s expected, got: %s" % (i, expected_element,
                                                   actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testShutdownRace(self):
    # [FUNCTION DOC]
    # Scenario: iterate a parallel-interleaved, batched dataset to exhaustion twice, creating a
    # new iterator each time. Asserts both passes produce identical results (no race at
    # iterator shutdown/restart).
    dataset = dataset_ops.Dataset.range(20)
    map_fn = lambda x: dataset_ops.Dataset.range(20 * x, 20 * (x + 1))
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(
            map_fn,
            cycle_length=3,
            sloppy=False,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    dataset = dataset.batch(32)

    results = []
    for _ in range(2):
      elements = []
      next_element = self.getNext(dataset)
      try:
        while True:
          elements.extend(self.evaluate(next_element()))
      except errors.OutOfRangeError:
        pass
      results.append(elements)
    self.assertAllEqual(results[0], results[1])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              sloppy=[None, True, False], global_determinism=[True, False])))
  def testDeterminismConfiguration(self, sloppy, global_determinism):
    # [FUNCTION DOC]
    # Scenario: parameterized over the sloppy argument (unset, true, false) and the global
    # deterministic option. Expected determinism is the global option when sloppy is unset,
    # otherwise the opposite of sloppy. Builds an interleave of 1000 elements where element 0
    # is delayed, and uses the base-class determinism checker to assert the output order is
    # deterministic or not as expected.
    if sloppy is None:
      expect_determinism = global_determinism
    else:
      expect_determinism = not sloppy
    elements = list(range(1000))

    def dataset_fn(delay_ms):
      # [FUNCTION DOC]
      # Builds the dataset for the determinism check; delay_ms is the artificial delay in
      # milliseconds applied to element 0. Applies parallel interleave (cycle length 10) and the
      # global deterministic option. Returns the dataset.

      def interleave_fn(x):
        # [FUNCTION DOC]
        # Wraps x in a one-element dataset, sleeping for the delay when x equals zero and not
        # sleeping otherwise. Returns the dataset.
        ds = dataset_ops.Dataset.from_tensors(x)
        if math_ops.equal(x, 0):
          ds = ds.apply(testing.sleep(delay_ms * 1000))
        else:
          ds = ds.apply(testing.sleep(0))
        return ds

      dataset = dataset_ops.Dataset.from_tensor_slices(elements)
      dataset = dataset.apply(
          interleave_ops.parallel_interleave(
              interleave_fn, cycle_length=10, sloppy=sloppy))

      opts = options_lib.Options()
      opts.deterministic = global_determinism
      dataset = dataset.with_options(opts)
      return dataset

    self.checkDeterminism(dataset_fn, expect_determinism, elements)


class ParallelInterleaveCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint (save/restore) tests for parallel interleave. Fixtures: input values 2 and 3,
  # repeated twice, so the total number of outputs is twice the sum of the inputs.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the base setUp, then stores the input values (2 and 3), the repeat count (2) and the
    # expected number of outputs.
    super(ParallelInterleaveCheckpointTest, self).setUp()
    self.input_values = np.array([2, 3], dtype=np.int64)
    self.num_repeats = 2
    self.num_outputs = np.sum(self.input_values) * 2

  def _build_ds(self, cycle_length, block_length, sloppy=False):
    # [FUNCTION DOC]
    # HELPER: builds the dataset checkpointed by the tests.
    # Repeats the input values and applies parallel interleave where each value x expands to
    # the integer range from 10x to 11x. Takes cycle length, block length and an optional sloppy
    # flag (default false). Returns the dataset.
    return (dataset_ops.Dataset.from_tensor_slices(self.input_values).repeat(
        self.num_repeats).apply(
            interleave_ops.parallel_interleave(
                lambda x: dataset_ops.Dataset.range(10 * x, 11 * x),
                cycle_length, block_length, sloppy)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(cycle_length=[1, 2], block_length=[1, 3])))
  def test(self, verify_fn, cycle_length, block_length):
    # [FUNCTION DOC]
    # Scenario: runs the standard checkpoint verification function over the non-sloppy dataset
    # for cycle length 1 or 2 and block length 1 or 3, expecting the full output count.
    verify_fn(self, lambda: self._build_ds(cycle_length, block_length),
              self.num_outputs)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(cycle_length=[1, 2], block_length=[1, 3])))
  def testWithSloppy(self, cycle_length, block_length):
    # [FUNCTION DOC]
    # Scenario: sloppy dataset generated with checkpoint/restore at several break points.
    # Since order is not guaranteed, asserts the sorted outputs equal the expected elements
    # (each range repeated per the repeat count, as a sorted-compatible list).
    break_points = self.gen_break_points(self.num_outputs, 10)
    expected_outputs = np.repeat(
        np.concatenate([np.arange(10 * x, 11 * x) for x in self.input_values]),
        self.num_repeats).tolist()

    actual = self.gen_outputs(
        lambda: self._build_ds(cycle_length, block_length, True),
        break_points, self.num_outputs)
    self.assertSequenceEqual(sorted(actual), expected_outputs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testSparse(self, verify_fn):
    # [FUNCTION DOC]
    # Scenario: checkpoint verification of a parallel interleave over sparse values (2x2
    # diagonal), expecting twenty outputs.

    def _map_fn(i):
      # [FUNCTION DOC]
      # Builds a sparse tensor value with entries i and -i on the diagonal of a 2x2 shape.
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      # [FUNCTION DOC]
      # Densifies sparse value x and returns a dataset of its rows.
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    def _build_dataset():
      # [FUNCTION DOC]
      # Returns the range dataset mapped to sparse values and interleaved with cycle length one.
      return dataset_ops.Dataset.range(10).map(_map_fn).apply(
          interleave_ops.parallel_interleave(_interleave_fn, 1))

    verify_fn(self, _build_dataset, num_outputs=20)


if __name__ == "__main__":
  test.main()
