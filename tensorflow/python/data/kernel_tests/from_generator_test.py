# [MODULE DOC]
# Tests for creating a dataset from a Python generator (the from-generator constructor). Covers
# generator functions, lists, arrays and generator expressions, concurrent generators through
# interleave, implicit dtype conversion, strings, nested structures, type/shape/structure errors,
# early termination, destructor and finalize-function behavior, extra arguments, shared names,
# ragged and sparse outputs through an output signature, argument validation errors and operation
# names. One test is disabled by name (it is not collected). No import-time side effects; runs the
# test main entry point when executed directly.
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
"""Tests for tf.data.Dataset.from_generator()."""
import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import from_generator_op
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class FromGeneratorTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for the from-generator dataset constructor, covering many kinds of
  # generators, error handling and lifecycle behavior. Uses the shared dataset test base class.

  def _testFromGenerator(self, generator, elem_sequence, num_repeats,
                         requires_initialization):
    # [FUNCTION DOC]
    # HELPER: shared check used by the generator-kind tests.
    # Parameters: generator (callable returning an iterable of 64-bit integer elements),
    # elem_sequence (list of expected elements for one pass), num_repeats (how many times the
    # dataset is repeated), requires_initialization (whether the iterator needs explicit
    # initialization). Builds a dataset from the generator, repeats it and prefetches five, then
    # asserts it produces the sequence repeated num_repeats times, checked over two iterations.
    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64).repeat(num_repeats).prefetch(5)
    self.assertDatasetProduces(
        dataset,
        elem_sequence * num_repeats,
        requires_initialization=requires_initialization,
        num_test_iterations=2)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_repeats=[1, 2], requires_initialization=[True, False])))
  def testFromGeneratorUsingFn(self, num_repeats, requires_initialization):
    # [FUNCTION DOC]
    # Parameters: num_repeats (repeat count), requires_initialization (iterator initialization
    # flag). Uses a generator function yielding lists where value i appears i times for i from 1
    # to 99, and verifies the dataset output through the shared helper.

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding, for each i from 1 to 99, a list of i copies of i.
      for i in range(1, 100):
        yield [i] * i

    elem_sequence = list(generator())
    self._testFromGenerator(
        generator,
        elem_sequence,
        num_repeats=num_repeats,
        requires_initialization=requires_initialization)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_repeats=[1, 2], requires_initialization=[True, False])))
  def testFromGeneratorUsingList(self, num_repeats, requires_initialization):
    # [FUNCTION DOC]
    # Parameters: num_repeats, requires_initialization. Uses a callable returning a list of lists
    # (value i repeated i times, i from 1 to 99) as the generator and verifies the output through
    # the shared helper.
    generator = lambda: [[i] * i for i in range(1, 100)]
    elem_sequence = list(generator())
    self._testFromGenerator(
        generator,
        elem_sequence,
        num_repeats=num_repeats,
        requires_initialization=requires_initialization)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_repeats=[1, 2], requires_initialization=[True, False])))
  def testFromGeneratorUsingNdarray(self, num_repeats, requires_initialization):
    # [FUNCTION DOC]
    # Parameters: num_repeats, requires_initialization. Uses a callable returning a 64-bit integer
    # array of 0 to 99 as the generator and verifies the output through the shared helper.
    generator = lambda: np.arange(100, dtype=np.int64)
    elem_sequence = list(generator())
    self._testFromGenerator(
        generator,
        elem_sequence,
        num_repeats=num_repeats,
        requires_initialization=requires_initialization)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_repeats=[1, 2], requires_initialization=[True, False])))
  def testFromGeneratorUsingGeneratorExpression(self, num_repeats,
                                                requires_initialization):
    # [FUNCTION DOC]
    # Parameters: num_repeats, requires_initialization. Uses a callable that returns a generator
    # expression (wrapping it makes it repeatable) yielding value i repeated i times, and
    # verifies the output through the shared helper.
    # NOTE(mrry): Generator *expressions* are not repeatable (or in general
    # reusable), because they eagerly evaluate the `for` expression as
    # `iter(range(1, 100))` and discard the means of reconstructing
    # `range(1, 100)`. Wrapping the generator expression in a `lambda` makes
    # it repeatable.
    generator = lambda: ([i] * i for i in range(1, 100))
    elem_sequence = list(generator())
    self._testFromGenerator(
        generator,
        elem_sequence,
        num_repeats=num_repeats,
        requires_initialization=requires_initialization)

  @combinations.generate(test_base.default_test_combinations())
  def testFromMultipleConcurrentGenerators(self):
    # [FUNCTION DOC]
    # Places a from-generator dataset (repeated five times and prefetched) inside an interleave of
    # 20 inputs with cycle length 10, so several generator iterators are active at once. Asserts
    # that each of the 100 passes yields the nine expected pairs in order, then an out-of-range
    # error.
    num_inner_repeats = 5
    num_outer_repeats = 20

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding, for i from 1 to 9, a pair of a list of i copies of i and the
      # list i, i squared, i cubed.
      for i in range(1, 10):
        yield ([i] * i, [i, i ** 2, i ** 3])
    input_list = list(generator())

    # The interleave transformation is essentially a flat map that draws from
    # multiple input datasets concurrently (in a cyclic fashion). By placing
    # `Dataset.from_generator()` inside an interleave, we test its behavior when
    # multiple iterators are active at the same time; by additionally
    # prefetching inside the interleave, we create the possibility of concurrent
    # invocations to several iterators created by the same dataset.
    def interleave_fn(_):
      # [FUNCTION DOC]
      # HELPER: ignores its input and returns the generator dataset (pair of 64-bit integers with
      # shapes unknown-length and length three), repeated and prefetched.
      return (dataset_ops.Dataset.from_generator(
          generator, output_types=(dtypes.int64, dtypes.int64),
          output_shapes=([None], [3]))
              .repeat(num_inner_repeats).prefetch(5))

    dataset = dataset_ops.Dataset.range(num_outer_repeats).interleave(
        interleave_fn, cycle_length=10, block_length=len(input_list))
    get_next = self.getNext(dataset)
    for _ in range(num_inner_repeats * num_outer_repeats):
      for elem in input_list:
        val0, val1 = self.evaluate(get_next())
        self.assertAllEqual(elem[0], val0)
        self.assertAllEqual(elem[1], val1)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def DISABLED_testFromGeneratorsRunningInParallel(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled by its name prefix (not collected as a test) and it also skips itself
    # immediately; the body after the skip call never runs under the test runner.
    # Intended scenario: three generator iterators run concurrently, each yielding 0, waiting at a
    # lock-and-condition barrier until all three have arrived, then yielding 1. An interleave with
    # prefetch drives them; asserts the output is three zeros then three ones, then out-of-range.
    self.skipTest("b/67868766")
    num_parallel_iterators = 3

    # Define shared state that multiple iterator instances will access to
    # demonstrate their concurrent activity.
    lock = threading.Lock()
    condition = threading.Condition(lock)
    next_ticket = [0]  # GUARDED_BY(lock)

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding 0, then taking a ticket under the shared condition; the last of
      # the parallel iterators to arrive wakes the others, earlier ones wait until all have
      # arrived; then releases the condition and yields 1.
      # NOTE(mrry): We yield one element before the barrier, because
      # the current implementation of `Dataset.interleave()` must
      # fetch one element from each incoming dataset to start the
      # prefetching.
      yield 0

      # Define a barrier that `num_parallel_iterators` iterators must enter
      # before any can proceed. Demonstrates that multiple iterators may be
      # active at the same time.
      condition.acquire()
      ticket = next_ticket[0]
      next_ticket[0] += 1
      if ticket == num_parallel_iterators - 1:
        # The last iterator to join the barrier notifies the others.
        condition.notify_all()
      else:
        # Wait until the last iterator enters the barrier.
        while next_ticket[0] < num_parallel_iterators:
          condition.wait()
      condition.release()

      yield 1

    # As in `testFromMultipleConcurrentGenerators()`, we use a combination of
    # `Dataset.interleave()` and `Dataset.prefetch()` to cause multiple
    # iterators to be active concurrently.
    def interleave_fn(_):
      # [FUNCTION DOC]
      # HELPER: ignores its input and returns the barrier generator as a scalar 64-bit integer
      # dataset with a prefetch of two.
      return dataset_ops.Dataset.from_generator(
          generator, output_types=dtypes.int64, output_shapes=[]).prefetch(2)

    dataset = dataset_ops.Dataset.range(num_parallel_iterators).interleave(
        interleave_fn, cycle_length=num_parallel_iterators, block_length=1)
    get_next = self.getNext(dataset)

    for elem in [0, 1]:
      for _ in range(num_parallel_iterators):
        self.assertAllEqual(elem, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorImplicitConversion(self):
    # [FUNCTION DOC]
    # For each of 8, 32 and 64-bit integer output types, builds a dataset from a generator that
    # yields the lists [1], [2], [3] with output shape of length one. Asserts each element's numpy
    # dtype matches the requested type and its value is right, then an out-of-range error.
    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding the one-item lists [1], [2] and [3].
      yield [1]
      yield [2]
      yield [3]

    for dtype in [dtypes.int8, dtypes.int32, dtypes.int64]:
      dataset = dataset_ops.Dataset.from_generator(
          generator, output_types=dtype, output_shapes=[1])
      get_next = self.getNext(dataset)

      for expected in [[1], [2], [3]]:
        next_val = self.evaluate(get_next())
        self.assertEqual(dtype.as_numpy_dtype, next_val.dtype)
        self.assertAllEqual(expected, next_val)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorString(self):
    # [FUNCTION DOC]
    # Builds a scalar string dataset from a generator yielding a text string, a byte string and a
    # unicode string. Asserts the output is the three values as byte strings.
    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding the strings foo, bar and baz in text, byte and unicode forms.
      yield "foo"
      yield b"bar"
      yield u"baz"

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.string, output_shapes=[])
    self.assertDatasetProduces(
        dataset, expected_output=[b"foo", b"bar", b"baz"])

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorDatastructures(self):
    # [FUNCTION DOC]
    # Builds a dataset from a generator yielding dictionaries with a string, an integer list and
    # an integer tuple, with matching output types and shapes. Asserts the three expected
    # dictionaries (strings as bytes, tuples as lists) are produced.
    # Tests multiple datastructures.
    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding three dictionaries with keys a, b and c.
      yield {"a": "foo", "b": [1, 2], "c": (9,)}
      yield {"a": "bar", "b": [3], "c": (7, 6)}
      yield {"a": "baz", "b": [5, 6], "c": (5, 4)}

    dataset = dataset_ops.Dataset.from_generator(
        generator,
        output_types={"a": dtypes.string, "b": dtypes.int32, "c": dtypes.int32},
        output_shapes={"a": [], "b": [None], "c": [None]})
    self.assertDatasetProduces(
        dataset,
        expected_output=[{"a": b"foo", "b": [1, 2], "c": [9]},
                         {"a": b"bar", "b": [3], "c": [7, 6]},
                         {"a": b"baz", "b": [5, 6], "c": [5, 4]}])

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorTypeError(self):
    # [FUNCTION DOC]
    # The generator yields two valid arrays, then a string, then another valid array, while the
    # declared type is 64-bit integer with shape of length three. Asserts the first two reads
    # succeed, the third raises an invalid-argument error, the fourth recovers, then out-of-range.
    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding two valid arrays, the string ERROR, then one more valid array.
      yield np.array([1, 2, 3], dtype=np.int64)
      yield np.array([4, 5, 6], dtype=np.int64)
      yield "ERROR"
      yield np.array([7, 8, 9], dtype=np.int64)

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64, output_shapes=[3])

    get_next = self.getNext(dataset)

    self.assertAllEqual([1, 2, 3], self.evaluate(get_next()))
    self.assertAllEqual([4, 5, 6], self.evaluate(get_next()))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.assertAllEqual([7, 8, 9], self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorShapeError(self):
    # [FUNCTION DOC]
    # The generator yields an array of length four among arrays of length three while the declared
    # shape is length three. Asserts the two valid reads succeed, the bad one raises an
    # invalid-argument error, the following valid one is produced, then out-of-range.
    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding arrays of lengths three, three, four and three.
      yield np.array([1, 2, 3], dtype=np.int64)
      yield np.array([4, 5, 6], dtype=np.int64)
      yield np.array([7, 8, 9, 10], dtype=np.int64)
      yield np.array([11, 12, 13], dtype=np.int64)

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64, output_shapes=[3])
    get_next = self.getNext(dataset)

    self.assertAllEqual([1, 2, 3], self.evaluate(get_next()))
    self.assertAllEqual([4, 5, 6], self.evaluate(get_next()))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.assertAllEqual([11, 12, 13], self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorStructureError(self):
    # [FUNCTION DOC]
    # The generator yields pairs but also a scalar and a triple while two 64-bit integer outputs
    # are declared. Asserts the two valid pairs are produced, the next two reads each raise an
    # invalid-argument error, the final valid pair is produced, then out-of-range.
    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding pairs (1,2), (3,4), then a scalar 5, a triple, then (9,10).
      yield 1, 2
      yield 3, 4
      yield 5
      yield 6, 7, 8
      yield 9, 10

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=(dtypes.int64, dtypes.int64))
    get_next = self.getNext(dataset)

    self.assertEqual((1, 2), self.evaluate(get_next()))
    self.assertEqual((3, 4), self.evaluate(get_next()))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.assertEqual((9, 10), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorHeterogeneous(self):
    # [FUNCTION DOC]
    # The generator yields a scalar and then a list with no declared shape. Asserts the dataset
    # produces both, i.e. unknown shapes allow heterogeneous element shapes.
    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding the scalar 1 and the list [2, 3].
      yield 1
      yield [2, 3]

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64)
    self.assertDatasetProduces(dataset, expected_output=[1, [2, 3]])

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorStopShort(self):
    # [FUNCTION DOC]
    # Reads only the first two of three generator elements and stops, asserting the two values;
    # checks that abandoning an iterator early does not fail.

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding 0, 1 and 2.
      yield 0
      yield 1
      yield 2

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64)
    get_next = self.getNext(dataset)
    self.assertAllEqual(0, self.evaluate(get_next()))
    self.assertAllEqual(1, self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorDestructorCalled(self):
    # [FUNCTION DOC]
    # Uses a wrapper class whose iterator endlessly yields 42 and sets a thread event when it is
    # destroyed, as the generator, taking two elements. Asserts both are 42, then out-of-range, and
    # that the event is set, i.e. the generator object is destroyed when the iterator terminates.
    # Use an `Event` to signal that the generator has been deleted.
    event = threading.Event()

    class GeneratorWrapper:
      # [CLASS DOC]
      # HELPER: endless iterator yielding 42 that signals the enclosing test's event when it is
      # garbage collected, used to observe generator destruction.

      def __iter__(self):
        # [FUNCTION DOC]
        # Returns itself as the iterator.
        return self

      def next(self):
        # [FUNCTION DOC]
        # Legacy iterator method; delegates to the standard next-item method.
        return self.__next__()

      def __next__(self):
        # [FUNCTION DOC]
        # Returns the constant 42 on every call (never ends).
        return 42

      def __del__(self):
        # [FUNCTION DOC]
        # Destructor: sets the enclosing test's event to signal the object was destroyed.
        event.set()

    dataset = dataset_ops.Dataset.from_generator(
        GeneratorWrapper, output_types=dtypes.int64).take(2)
    get_next = self.getNext(dataset)

    self.assertAllEqual(42, self.evaluate(get_next()))
    self.assertAllEqual(42, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    # Test that `GeneratorWrapper` object is destroyed when the
    # iterator terminates (and the generator iterator is deleted).
    self.assertTrue(event.is_set())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorWithArgs(self):
    # [FUNCTION DOC]
    # Flat-maps a range of five into generator datasets that receive the element as an argument.
    # Asserts the output is each n (1 to 4) repeated n times; zero yields nothing.

    def flat_map_fn(elem):
      # [FUNCTION DOC]
      # HELPER: returns a generator dataset of 64-bit scalars, passing elem as the generator
      # argument.

      def generator_with_arg(n):
        # [FUNCTION DOC]
        # HELPER: generator yielding the number n, n times.
        for _ in range(n):
          yield np.array(n, dtype=np.int64)

      return dataset_ops.Dataset.from_generator(
          generator_with_arg, output_types=dtypes.int64, output_shapes=(),
          args=(elem,))

    dataset = dataset_ops.Dataset.range(5).flat_map(flat_map_fn)
    self.assertDatasetProduces(
        dataset, expected_output=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorWithTwoArgs(self):
    # [FUNCTION DOC]
    # Zips a range of five with an endlessly repeated string and flat-maps each pair into a
    # generator dataset given both values as arguments. Asserts the output is, for each n, the
    # pairs (0..n-1, "Hi!").

    def flat_map_fn(elem, message):
      # [FUNCTION DOC]
      # HELPER: returns a generator dataset of (integer, string) pairs, passing elem and message as
      # the generator arguments.

      def generator_with_arg(n, msg):
        # [FUNCTION DOC]
        # HELPER: generator yielding pairs of i and msg for i from 0 to n minus one.
        for i in range(n):
          yield i, msg

      return dataset_ops.Dataset.from_generator(
          generator_with_arg, output_types=(dtypes.int64, dtypes.string),
          output_shapes=((), ()), args=(elem, message))

    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(5),
         dataset_ops.Dataset.from_tensors("Hi!").repeat(None)
        )).flat_map(flat_map_fn)

    self.assertDatasetProduces(
        dataset,
        expected_output=[(0, b"Hi!"), (0, b"Hi!"), (1, b"Hi!"), (0, b"Hi!"),
                         (1, b"Hi!"), (2, b"Hi!"), (0, b"Hi!"), (1, b"Hi!"),
                         (2, b"Hi!"), (3, b"Hi!")])

  @combinations.generate(test_base.default_test_combinations())
  def testGeneratorDatasetFinalizeFunctionCalled(self):
    # [FUNCTION DOC]
    # Uses the internal generator dataset class directly with a constant initial state of 37,
    # identity init/next functions and a finalize function that runs a stateful Python function
    # setting a thread event. Takes two elements, asserts both are 37, then out-of-range. (The
    # event is set by the finalize function; it is not asserted afterwards in this test.)
    # NOTE(mrry): This test tests the internal `_GeneratorDataset`,
    # which affords more control over what the finalize function can do than
    # the `Dataset.from_generator()` wrapper.

    # Use an `Event` to signal that the generator has been deleted.
    event = threading.Event()

    def finalize_fn(_):
      # [FUNCTION DOC]
      # HELPER: finalize function; ignores its input and returns a stateful Python-function op that
      # sets the event and returns zero.
      def finalize_py_func():
        # [FUNCTION DOC]
        # HELPER: Python function that sets the thread event and returns 0.
        event.set()
        return 0
      return script_ops.py_func(finalize_py_func, [], [dtypes.int64],
                                stateful=True)

    dummy = constant_op.constant(37)

    dataset = from_generator_op._GeneratorDataset(
        dummy, lambda x: x, lambda x: x, finalize_fn,
        tensor_spec.TensorSpec((), dtypes.int32))

    dataset = dataset.take(2)
    get_next = self.getNext(dataset)

    self.assertAllEqual(37, self.evaluate(get_next()))
    self.assertAllEqual(37, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testSharedName(self):
    # [FUNCTION DOC]
    # Builds a generator dataset of ten [20] elements and gets an iterator with initialization and a
    # shared name. Asserts the first element is [20].

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding the list [20] ten times.
      for _ in range(10):
        yield [20]

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=(dtypes.int64))
    get_next = self.getNext(
        dataset, requires_initialization=True, shared_name="shared_dataset")

    self.assertAllEqual([20], self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorRaggedTensor(self):
    # [FUNCTION DOC]
    # Builds a dataset from a generator yielding a ragged tensor, with a ragged output signature.
    # Asserts the first element is a ragged tensor equal to [[1, 2], [3]].

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding one ragged tensor with rows [1, 2] and [3].
      yield ragged_factory_ops.constant([[1, 2], [3]])

    dataset = dataset_ops.Dataset.from_generator(
        generator,
        output_signature=ragged_tensor.RaggedTensorSpec(
            shape=(2, None), dtype=dtypes.int32))
    get_next = self.getNext(dataset)

    ret = get_next()

    self.assertIsInstance(ret, ragged_tensor.RaggedTensor)
    self.assertAllEqual([[1, 2], [3]], ret)

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorSparseTensor(self):
    # [FUNCTION DOC]
    # Builds a dataset from a generator yielding a three-by-four sparse tensor, with a sparse
    # output signature. Asserts the first element is a sparse tensor whose dense form has ones and
    # twos at the expected positions.

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding one sparse tensor with values 1 and 2 at positions (0,0), (1,2).
      yield sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]],
          values=constant_op.constant([1, 2], dtype=dtypes.int64),
          dense_shape=[3, 4])

    dataset = dataset_ops.Dataset.from_generator(
        generator,
        output_signature=sparse_tensor.SparseTensorSpec([3, 4], dtypes.int64))

    get_next = self.getNext(dataset)

    ret = get_next()

    self.assertIsInstance(ret, sparse_tensor.SparseTensor)
    self.assertAllEqual([[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
                        sparse_ops.sparse_tensor_to_dense(ret))

  @combinations.generate(test_base.default_test_combinations())
  def testTypeIsListError(self):
    # [FUNCTION DOC]
    # Passing the output types as a list containing a type (instead of a single type or tuple)
    # must raise a type error saying the argument cannot be converted to a data type.

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding the list [20] ten times (never actually run).
      for _ in range(10):
        yield [20]

    with self.assertRaisesRegex(
        TypeError, r"Cannot convert the argument `type_value`: "
        r"\[tf.int64\] to a TensorFlow DType"):
      dataset_ops.Dataset.from_generator(
          generator, output_types=[dtypes.int64])

  @combinations.generate(test_base.default_test_combinations())
  def testDimensionIsListError(self):
    # [FUNCTION DOC]
    # Passing output shapes as a list of lists (a dimension that is a list) must raise a type error
    # saying a dimension must be an integer or nothing.

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding the list [20] ten times (never actually run).
      for _ in range(10):
        yield [20]

    with self.assertRaisesRegex(TypeError,
                                r"Dimension value must be integer or None"):
      dataset_ops.Dataset.from_generator(
          generator, output_types=(dtypes.int64), output_shapes=[[1]])

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Verifies the constructor accepts an operation name together with output type and shape;
    # builds the dataset and asserts nothing further (passes if no error is raised).

    def generator():
      # [FUNCTION DOC]
      # HELPER: generator yielding the single value 42.
      yield 42

    dataset_ops.Dataset.from_generator(
        generator,
        output_types=(dtypes.int64),
        output_shapes=[1],
        name="from_generator")


if __name__ == "__main__":
  test.main()
