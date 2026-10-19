# [MODULE DOC]
# Kernel tests for the multi-device iterator classes MultiDeviceIterator (graph/eager, with an
# explicit initializer) and OwnedMultiDeviceIterator (eager and tf.function only, owns its
# resource). Covers cancellation of get_next on one device, empty datasets, empty device lists,
# buffer-size combinations, round-robin sharding of elements across devices, uneven consumption,
# optional-returning get_next, repeated initialization, optimization options, use inside
# functions, retracing limits and argument validation errors. Needs three logical devices, which
# are configured by the shared test base. Runs through the standard TensorFlow test main entry
# point; defines a module-level list of parameter combinations over the two iterator classes.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `MultiDeviceIterator` and `OwnedMultiDeviceIterator` API."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import from_generator_op
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


cls_combination = combinations.combine(cls=[
    combinations.NamedObject("MultiDeviceIterator",
                             multi_device_iterator_ops.MultiDeviceIterator),
    combinations.NamedObject("OwnedMultiDeviceIterator",
                             multi_device_iterator_ops.OwnedMultiDeviceIterator)
])


class MultiDeviceIteratorCommonTest(test_base.DatasetTestBase,
                                    parameterized.TestCase):
  # [CLASS DOC]
  # Test case holding tests whose behavior must be identical for both MultiDeviceIterator and
  # OwnedMultiDeviceIterator; each test is parameterized over the iterator class and runs in
  # eager mode only. Fixture: three logical devices stored on the instance in setUp.
  """Tests that are common to MultiDeviceIterator and OwnedMultiDeviceIterator."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setUp and configures three logical devices for multi-device
    # testing, saving the resulting device name list on the instance. Takes and returns nothing.
    super().setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(), cls_combination))
  def testCancelGetNextWithDevice(self, cls):
    # [FUNCTION DOC]
    # Scenario: a dataset whose map function blocks on a queue is wrapped by the iterator class
    # under test (prefetch buffer size zero) with two target devices. A get_next on device one
    # is issued as a cancelable function under an asynchronous executor, then cancelled midway.
    # Expected: waiting on the executor raises a cancelled error, and cancelling one device's
    # get_next does not break the iterator: a later get_next on the other device still returns
    # the expected element (value 1). Also drops a function reference and collects garbage as
    # a workaround for a sanitizer error. Parameter cls is the iterator class.
    ping = data_flow_ops.FIFOQueue(capacity=2, dtypes=dtypes.int64)
    ping = data_flow_ops.FIFOQueue(capacity=2, dtypes=dtypes.int64)
    pong = data_flow_ops.FIFOQueue(capacity=2, dtypes=dtypes.int64)

    @def_function.function
    def map_fn(v):
      # [FUNCTION DOC]
      # HELPER: map function for testCancelGetNextWithDevice that makes the pipeline block.
      # Dequeues a token from the ping queue, enqueues it to the pong queue (signalling that the
      # pipeline is mid get_next), then dequeues another ping token and returns the input value
      # plus that token. Input v is a scalar element; returns a scalar.
      ball = ping.dequeue()
      with ops.control_dependencies([pong.enqueue(ball)]):
        return v + ping.dequeue()

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(map_fn)

    # We need to set prefetch_buffer_size=0 so that we can cancel the
    # MultiDeviceIteratorGetNextFromShardOp from eager. If
    # prefetch_buffer_size>0, that op runs in the background threads of the
    # prefetch and can only be cancelled by deleting the iterator.
    multi_device_iterator = cls(
        dataset, [self._devices[1], self._devices[2]], prefetch_buffer_size=0)

    @def_function.function
    def get_next_device1():
      # [FUNCTION DOC]
      # HELPER: traced function for testCancelGetNextWithDevice that fetches the next element of
      # the enclosing multi-device iterator for the second configured device. Takes no
      # arguments; returns that element.
      return multi_device_iterator.get_next(self._devices[1])

    async_executor = executor.new_executor(enable_async=True)
    with context.executor_scope(async_executor):
      cancel_mgr = cancellation.CancellationManager()
      cancel_mgr.get_cancelable_function(
          get_next_device1.get_concrete_function())()
    # Make sure we cancel in the middle of get_next.
    ping.enqueue(0)
    pong.dequeue()
    cancel_mgr.start_cancel()
    with self.assertRaises(errors.CancelledError):
      async_executor.wait()
    # Note that fetching from upstream iterator is not cancelled with the
    # cancellation of get_next.
    ping.enqueue(0)

    # Cancelling a get_next on one device shouldn't cancel the
    # multi_device_iterator and iterators on other devices.
    ping.enqueue(0)
    ping.enqueue(0)
    self.assertEqual(1,
                     multi_device_iterator.get_next(self._devices[2]).numpy())
    # FIXME(b/209534797): Workaround an asan error caused by this test.
    # Remove the dangling reference from tf.function to ensure queue objects
    # are not freed before they are flushed.
    import gc  # pylint: disable=g-import-not-at-top
    del get_next_device1
    gc.collect()

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(), cls_combination))
  def testEmptyDataset(self, cls):
    # [FUNCTION DOC]
    # Scenario: an empty dataset (range of zero) is wrapped by the iterator class cls with two
    # devices. Expected: calling get_next with no device raises an out-of-range error.
    dataset = dataset_ops.Dataset.range(0)
    multi_device_iterator = cls(
        dataset, devices=[self._devices[1], self._devices[2]])
    with self.assertRaises(errors.OutOfRangeError):
      multi_device_iterator.get_next()

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(), cls_combination))
  def testEmptyDeviceList(self, cls):
    # [FUNCTION DOC]
    # Scenario: the iterator class cls is constructed with a ten-element dataset and an empty
    # device list. Expected: an invalid-argument error stating that the devices attribute must
    # have at least one entry.
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Length for attr 'devices' of 0 must be at least minimum 1"):
      cls(dataset, devices=[])


class MultiDeviceIteratorTest(test_base.DatasetTestBase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the classic MultiDeviceIterator (explicit initializer, graph and eager modes):
  # initialization, buffer sizes, sharding across devices, uneven consumption, optional results
  # and graph optimization. Fixture: three logical devices stored in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setUp and configures three logical devices, saving the
    # device name list on the instance. Takes and returns nothing.
    super(MultiDeviceIteratorTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_inits=[0, 1, 42])))
  def testInitOnly(self, num_inits):
    # [FUNCTION DOC]
    # Scenario: a MultiDeviceIterator over a ten-element range and two devices has its
    # initializer evaluated num_inits times (zero, one or many). Expected: no error, i.e.
    # repeated initialization is allowed.
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    for _ in range(num_inits):
      self.evaluate(multi_device_iterator.initializer)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              max_buffer_size=[0, 1, 10], prefetch_buffer_size=[0, 1, 10])))
  def testBasic(self, prefetch_buffer_size, max_buffer_size):
    # [FUNCTION DOC]
    # Scenario: MultiDeviceIterator over a ten-element range with two devices and the given
    # maximum and prefetch buffer sizes (parameterized over zero, one and ten). Expected:
    # elements are dealt round-robin, so each get_next yields (i, i+1) for devices one and
    # two in order, and after ten elements an out-of-range error is raised.
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]],
        max_buffer_size=max_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size)

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testOneOnSameDevice(self):
    # [FUNCTION DOC]
    # Scenario: twelve-element range dealt over three devices, one of which is the device the
    # dataset lives on (the first). Expected: each get_next yields three consecutive values in
    # device order, and an out-of-range error is raised once exhausted.
    dataset = dataset_ops.Dataset.range(12)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[0], self._devices[1], self._devices[2]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 12, 3):
      elem_on_0, elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_0))
      self.assertEqual(i + 1, self.evaluate(elem_on_1))
      self.assertEqual(i + 2, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_0, elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_0)
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatDevices(self):
    # [FUNCTION DOC]
    # Scenario: the same device is listed twice in the device list for a ten-element range.
    # Expected: elements are still dealt in pairs (i, i+1) and an out-of-range error is raised
    # after exhaustion.
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[1]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elements = multi_device_iterator.get_next()
      elem_on_1, elem_on_2 = elements
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elements = multi_device_iterator.get_next()
      elem_on_1, elem_on_2 = elements
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testNotFullyDivisible(self):
    # [FUNCTION DOC]
    # Scenario: nine-element range over two devices (odd count). Expected: four full pairs are
    # read, then a get_next for the first device alone returns the last value (8), and a
    # following combined get_next raises an out-of-range error.
    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 8, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    elem_on_1 = multi_device_iterator.get_next(self._devices[1])
    self.assertEqual(8, self.evaluate(elem_on_1))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testGetNextAsOptional(self):
    # [FUNCTION DOC]
    # Scenario: get_next_as_optional on a ten-element range over two devices. Expected: while
    # data remains, both optionals have values equal to the round-robin elements; after
    # exhaustion both report no value, and fetching the value of either raises an
    # invalid-argument error.
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next_as_optional()
      has_elem_1, get_elem_1 = self.evaluate(
          [elem_on_1.has_value(), elem_on_1.get_value()])
      has_elem_2, get_elem_2 = self.evaluate(
          [elem_on_2.has_value(), elem_on_2.get_value()])
      self.assertTrue(has_elem_1)
      self.assertEqual(i, get_elem_1)
      self.assertTrue(has_elem_2)
      self.assertEqual(i + 1, get_elem_2)
    elem_on_1, elem_on_2 = multi_device_iterator.get_next_as_optional()
    has_elem_1 = elem_on_1.has_value()
    has_elem_2 = elem_on_2.has_value()
    self.assertFalse(self.evaluate(has_elem_1))
    self.assertFalse(self.evaluate(has_elem_2))
    with self.assertRaises(errors.InvalidArgumentError):
      elem_1 = elem_on_1.get_value()
      self.evaluate(elem_1)
    with self.assertRaises(errors.InvalidArgumentError):
      elem_2 = elem_on_2.get_value()
      self.evaluate(elem_2)

  @combinations.generate(test_base.default_test_combinations())
  def testUneven(self):
    # [FUNCTION DOC]
    # Scenario: iterator with a per-device buffer of four; all five elements of the first device
    # are consumed before any of the second device. Expected: first device yields even values,
    # second device yields odd values, then a combined get_next raises an out-of-range error.
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]], max_buffer_size=4)

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1 = multi_device_iterator.get_next(self._devices[1])
      self.assertEqual(i, self.evaluate(elem_on_1))
    for i in range(0, 10, 2):
      elem_on_2 = multi_device_iterator.get_next(self._devices[2])
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.graph_only_combinations())
  def testMultipleInitializationsGraph(self):
    # [FUNCTION DOC]
    # Graph-mode scenario: a zipped pair of thousand-element ranges feeds the iterator with
    # prefetch four; the initializer is run five times, each followed by evaluating both device
    # outputs. Expected: every re-initialization restarts from the beginning, giving the first
    # pairs (0, 0) and (1, 1).
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset_ops.Dataset.range(1000)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]], prefetch_buffer_size=4)
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()

    for _ in range(5):
      self.evaluate(multi_device_iterator.initializer)
      self.assertEqual([(0, 0), (1, 1)], self.evaluate([elem_on_1, elem_on_2]))

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleInitializationsEager(self):
    # [FUNCTION DOC]
    # Eager-mode scenario: five times in a row a new iterator is built on the same zipped
    # dataset, initialized and read once. Expected: each fresh iterator starts at the beginning
    # and yields (0, 0) and (1, 1).
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset_ops.Dataset.range(1000)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))

    for _ in range(5):
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, [self._devices[1], self._devices[2]], prefetch_buffer_size=4)
      self.evaluate(multi_device_iterator.initializer)
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual([(0, 0), (1, 1)], self.evaluate([elem_on_1, elem_on_2]))

  @combinations.generate(test_base.default_test_combinations())
  def testOptimization(self):
    # [FUNCTION DOC]
    # Scenario: a pipeline asserting that its next transformation is the in-memory cache, with a
    # zero-skip that should be removed by the no-op elimination option, is fed to the iterator.
    # Expected: the optimization applies (the assertion passes) and elements are still dealt
    # round-robin across two devices, ending with an out-of-range error.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # this should be optimized away
    dataset = dataset.cache()

    options = options_lib.Options()
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)

    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)


class OwnedMultiDeviceIteratorTest(test_base.DatasetTestBase,
                                   parameterized.TestCase):
  # [CLASS DOC]
  # Test case for OwnedMultiDeviceIterator (eager and function mode only): buffer sizes, use
  # inside functions, resource cleanup on errors, re-creation, retracing and constructor
  # argument validation. Fixture: three logical devices stored in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setUp and configures three logical devices, saving the
    # device name list on the instance. Takes and returns nothing.
    super(OwnedMultiDeviceIteratorTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(
              max_buffer_size=[0, 1, 10], prefetch_buffer_size=[0, 1, 10])))
  def testBasic(self, max_buffer_size, prefetch_buffer_size):
    # [FUNCTION DOC]
    # Scenario: OwnedMultiDeviceIterator over a thousand-element range with two devices and the
    # parameterized buffer sizes, consumed by Python iteration. Expected: the i-th step yields
    # the pair (2i, 2i+1) as numpy values.
    dataset = dataset_ops.Dataset.range(1000)

    mdi = multi_device_iterator_ops.OwnedMultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]],
        max_buffer_size=max_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size)

    for i, el in enumerate(mdi):
      self.assertEqual([i * 2, i * 2 + 1], [el[0].numpy(), el[1].numpy()])

  @combinations.generate(test_base.eager_only_combinations())
  def testBasicFunction(self):
    # [FUNCTION DOC]
    # Scenario: inside a traced function the dataset is created on the first device, wrapped in
    # an OwnedMultiDeviceIterator over devices two and three, and five pairs are read and
    # enqueued into a shared queue. Expected: after calling the function the queue yields the
    # values zero through nine in order.
    queue = data_flow_ops.FIFOQueue(10, dtypes.int64)

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: traced function for testBasicFunction. Builds a ten-element range on the first
      # device, wraps it in an owned multi-device iterator, and five times reads a pair of
      # elements and enqueues both into the enclosing queue. Takes and returns nothing.
      with ops.device(self._devices[0]):
        dataset = dataset_ops.Dataset.range(10)
      iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset, [self._devices[1], self._devices[2]])
      for _ in range(5):
        el0, el1 = next(iterator)
        queue.enqueue(el0)
        queue.enqueue(el1)

    fn()

    for i in range(10):
      self.assertEqual(queue.dequeue().numpy(), i)

  @combinations.generate(test_base.eager_only_combinations())
  def testFunctionError(self):
    # [FUNCTION DOC]
    # Scenario: a traced function builds a generator-backed dataset whose next-function reads
    # from an empty dataset, wraps it in an owned multi-device iterator and calls next.
    # Expected: calling the function raises an out-of-range error, and the generator's finalize
    # function still runs (queue size ends at two), proving the iterator resource was freed
    # despite the error.
    # In this test we verify that a function that raises an error ends up
    # properly deallocating the iterator resource.

    queue = data_flow_ops.FIFOQueue(10, dtypes.int64)
    queue.enqueue(0)

    def init_fn(n):
      # [FUNCTION DOC]
      # HELPER: generator initialization function for testFunctionError; returns its input
      # unchanged.
      return n

    def next_fn(_):
      # [FUNCTION DOC]
      # HELPER: generator next function for testFunctionError; ignores its argument, creates an
      # empty dataset and requests its next element, which raises an out-of-range error.
      ds = dataset_ops.Dataset.range(0)
      return next(iter(ds))

    def finalize_fn(n):
      # [FUNCTION DOC]
      # HELPER: generator finalize function for testFunctionError; enqueues a zero into the
      # enclosing queue (so the test can observe that cleanup ran) and returns its input.
      queue.enqueue(0)
      return n

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: traced function for testFunctionError. Builds the generator dataset from the
      # init, next and finalize functions, wraps it in an owned multi-device iterator over two
      # devices and requests one element, which fails with an out-of-range error.
      dataset = from_generator_op._GeneratorDataset(
          1,
          init_fn,
          next_fn,
          finalize_fn,
          output_signature=tensor_spec.TensorSpec([], dtypes.int64))
      iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset, [self._devices[1], self._devices[2]])
      next(iterator)

    with self.assertRaises(errors.OutOfRangeError):
      fn()

    self.assertEqual(queue.size().numpy(), 2)

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleInitializations(self):
    # [FUNCTION DOC]
    # Scenario: five times, a new owned iterator is built over the same thousand-element
    # dataset and fully iterated. Expected: every iteration yields the pairs (2i, 2i+1) from the
    # start.
    dataset = dataset_ops.Dataset.range(1000)

    for _ in range(5):
      multi_device_iterator = (
          multi_device_iterator_ops.OwnedMultiDeviceIterator(
              dataset, [self._devices[1], self._devices[2]]))
      for i, el in enumerate(multi_device_iterator):
        self.assertEqual([i * 2, i * 2 + 1], [el[0].numpy(), el[1].numpy()])

  @combinations.generate(test_base.eager_only_combinations())
  def testLimitedRetracing(self):
    # [FUNCTION DOC]
    # Scenario: a traced function that sums five element pairs from an iterator argument is
    # called with ten different owned iterators over two datasets of differing length.
    # Expected: each call returns 45 and the function is traced only once (trace counter is one),
    # so iterators of the same structure do not cause retracing.
    trace_count = [0]

    @def_function.function
    def f(iterator):
      # [FUNCTION DOC]
      # HELPER: traced function for testLimitedRetracing. Increments the enclosing trace counter
      # (so retraces are counted), reads five element pairs from the given iterator and returns
      # the sum of all ten values as a 64-bit integer.
      trace_count[0] += 1
      counter = np.int64(0)
      for _ in range(5):
        elem = next(iterator)
        counter += elem[0]
        counter += elem[1]
      return counter

    dataset = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(20)

    for _ in range(10):
      multi_device_iterator = (
          multi_device_iterator_ops.OwnedMultiDeviceIterator(
              dataset, [self._devices[1], self._devices[2]]))
      self.assertEqual(self.evaluate(f(multi_device_iterator)), 45)
      multi_device_iterator2 = (
          multi_device_iterator_ops.OwnedMultiDeviceIterator(
              dataset2, [self._devices[1], self._devices[2]]))
      self.assertEqual(self.evaluate(f(multi_device_iterator2)), 45)
      self.assertEqual(trace_count[0], 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testMissingDevices(self):
    # [FUNCTION DOC]
    # Scenario: constructing an owned iterator with a dataset but no devices. Expected: a value
    # error saying the devices must be provided.
    dataset = dataset_ops.Dataset.range(1000)
    with self.assertRaisesRegex(ValueError, "`devices` must be provided."):
      multi_device_iterator_ops.OwnedMultiDeviceIterator(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testMissingInput(self):
    # [FUNCTION DOC]
    # Scenario: constructing an owned iterator with devices but neither a dataset nor
    # components/element spec. Expected: a value error saying that without a dataset both
    # components and element spec must be given.
    with self.assertRaisesRegex(
        ValueError,
        "When `dataset` is not provided, both `components` and `element_spec` "
        "must be specified."):
      multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset=None, devices=[self._devices[1], self._devices[2]])

  @combinations.generate(test_base.eager_only_combinations())
  def testExtraElementSpecInput(self):
    # [FUNCTION DOC]
    # Scenario: constructing an owned iterator with both a dataset and an explicit element spec.
    # Expected: a value error saying that element spec and components must not be given when a
    # dataset is provided.
    dataset = dataset_ops.Dataset.range(1000)
    with self.assertRaisesRegex(
        ValueError,
        "When `dataset` is provided, `element_spec` and `components` must "
        "not be specified."):
      multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset, devices=[self._devices[1], self._devices[2]],
          element_spec=dataset.element_spec)

  @combinations.generate(test_base.graph_only_combinations())
  def testGraphMode(self):
    # [FUNCTION DOC]
    # Graph-mode scenario: constructing an owned iterator outside a function with eager
    # execution disabled. Expected: a runtime error saying the iterator is only supported inside
    # a function or with eager execution enabled.
    dataset = dataset_ops.Dataset.range(1000)
    with self.assertRaisesRegex(
        RuntimeError,
        "OwnedMultiDeviceIterator is only supported inside of tf.function or "
        "when eager execution is enabled."):
      multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset, devices=[self._devices[1], self._devices[2]])


if __name__ == "__main__":
  test.main()
