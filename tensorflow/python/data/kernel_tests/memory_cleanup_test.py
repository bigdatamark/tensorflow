# [MODULE DOC]
# Tests that dataset pipelines do not leak memory or native tensor objects in eager mode. Defines one
# test class that checks multi-device iterator memory growth (when the optional memory profiler
# package is installed) and that consuming various transformations (filter, flat map, generator,
# map, interleave, including legacy variants) leaves no live tensor objects after garbage collection.
# Import-time side effects: tries to import the optional memory profiler and falls back to a null
# value if missing. Run as a script it starts the test main loop.
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
"""Verify that memory usage is minimal in eager mode."""

import gc
import time
import weakref

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal


# memory_profiler might not be available in the OSS version of TensorFlow.
try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


class MemoryCleanupTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case verifying memory cleanup of dataset pipelines in eager mode. Fixture: setUp configures
  # three logical devices stored on the instance. Provides two assertion helpers (memory not
  # increasing, no leaked tensors) used by the test methods.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then configures three devices for multi-device testing and stores the
    # returned device names in the instance attribute for use by the multi-device iterator tests.
    super(MemoryCleanupTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)

  def assertMemoryNotIncreasing(self, f, num_iters, max_increase_mb):
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the multi-device memory tests.
    # Inputs: f (callable to run repeatedly), num_iters (how many measured repetitions) and
    # max_increase_mb (allowed growth in megabytes). Warms up by calling f once, sleeps four seconds so
    # background threads allocate, records process memory, calls f num_iters times, and logs the
    # increase. Fails an assertion with a descriptive message if the increase reaches the maximum. Needs
    # the memory profiler package. Note it uses a bare assertion statement, not a test assertion method.
    """Assert memory usage doesn't increase beyond given threshold for f."""

    # Warm up.
    f()
    # Wait for background threads to start up and allocate memory.
    time.sleep(4)
    initial = memory_profiler.memory_usage(-1)[0]
    for _ in range(num_iters):
      f()
    increase = memory_profiler.memory_usage(-1)[0] - initial
    logging.info("Memory increase observed: %f MB" % increase)
    assert increase < max_increase_mb, (
        "Increase is too high. Initial memory usage: %f MB. Increase: %f MB. "
        "Maximum allowed increase: %f") % (initial, increase, max_increase_mb)

  def assertNoMemoryLeak(self, dataset_fn):
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the transformation tests.
    # Input: dataset_fn, a callable that builds a dataset. Ten times it builds the dataset and reads 100
    # elements, then forces garbage collection and scans all live objects for native tensor-like objects
    # (skipping weak reference proxies, which raise errors when type checked). Asserts that none remain.
    """Assert consuming elements from the dataset does not leak memory."""

    def run():
      # [FUNCTION DOC]
      # HELPER: builds one dataset with the supplied function and reads 100 elements from it, evaluating
      # each.
      get_next = self.getNext(dataset_fn())
      for _ in range(100):
        self.evaluate(get_next())

    for _ in range(10):
      run()

    gc.collect()

    def is_native_object(o):
      # [FUNCTION DOC]
      # HELPER: returns whether an object is a native tensor object. Weak reference proxies yield false
      # without being inspected because checking an expired proxy raises a reference error.
      # First check if `o` is a weakref proxy. Calling
      # `isinstance(o, internal.NativeObject)` on an expired weak reference
      # proxy will raise a ReferenceError.
      if isinstance(o, weakref.ProxyTypes): return False
      return isinstance(o, internal.NativeObject)

    tensors = [
        o for o in gc.get_objects() if is_native_object(o)
    ]
    self.assertEmpty(tensors, "%d Tensors are still alive." % len(tensors))

  @combinations.generate(test_base.eager_only_combinations())
  def testEagerMemoryUsageWithReset(self):
    # [FUNCTION DOC]
    # Eager only; skipped without the memory profiler. Scenario: one multi-device iterator over a range
    # is repeatedly asked for an element and reset. Asserts memory growth stays below 250 MB over 50
    # iterations.
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    def f():
      # [FUNCTION DOC]
      # HELPER: one iteration of the workload: fetch the next element then reset the iterator.
      self.evaluate(multi_device_iterator.get_next())
      multi_device_iterator._eager_reset()

    self.assertMemoryNotIncreasing(f, num_iters=50, max_increase_mb=250)

  @combinations.generate(test_base.eager_only_combinations())
  def testEagerMemoryUsageWithRecreation(self):
    # [FUNCTION DOC]
    # Eager only; skipped without the memory profiler. Scenario: a new multi-device iterator is created,
    # read once and deleted in every iteration. Asserts growth stays below 250 MB over 50 iterations. The
    # threshold is large because of a known open bug.
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    dataset = dataset_ops.Dataset.range(10)

    def f():
      # [FUNCTION DOC]
      # HELPER: one iteration of the workload: create an iterator, fetch one element and delete it.
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, [self._devices[1], self._devices[2]])
      self.evaluate(multi_device_iterator.get_next())
      del multi_device_iterator

    # TODO(b/123316347): Reduce threshold once bug is fixed.
    self.assertMemoryNotIncreasing(f, num_iters=50, max_increase_mb=250)

  @combinations.generate(test_base.eager_only_combinations())
  def testFilter(self):
    # [FUNCTION DOC]
    # Eager only. Scenario: a filter that accepts everything. Asserts no native tensors leak after
    # consuming it.

    def get_dataset():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset filtered with the always-true predicate.

      def fn(_):
        # [FUNCTION DOC]
        # Predicate: ignores the input and returns true.
        return True

      return dataset_ops.Dataset.range(0, 100).filter(fn)

    self.assertNoMemoryLeak(get_dataset)

  @combinations.generate(combinations.combine(tf_api_version=1, mode="eager"))
  def testFilterLegacy(self):
    # [FUNCTION DOC]
    # Eager mode, API version 1. Same as the filter test but using the legacy filter function variant.

    def get_dataset():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset filtered with the legacy filter method.

      def fn(_):
        # [FUNCTION DOC]
        # Predicate: ignores the input and returns true.
        return True

      return dataset_ops.Dataset.range(0, 100).filter_with_legacy_function(fn)

    self.assertNoMemoryLeak(get_dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testFlatMap(self):
    # [FUNCTION DOC]
    # Eager only. Scenario: flat map producing a single-element dataset of the square. Asserts no leaks.

    def get_dataset():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset flat-mapped with the function.

      def fn(x):
        # [FUNCTION DOC]
        # Returns a one-element dataset holding the input squared.
        return dataset_ops.Dataset.from_tensors(x * x)

      return dataset_ops.Dataset.range(0, 100).flat_map(fn)

    self.assertNoMemoryLeak(get_dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testFromGenerator(self):
    # [FUNCTION DOC]
    # Eager only. Scenario: dataset from a generator yielding a range of 100 as floats. Asserts no leaks.

    def get_dataset():
      # [FUNCTION DOC]
      # HELPER: returns a dataset built from the generator function with float output type.

      def fn():
        # [FUNCTION DOC]
        # Generator function: returns a range of 100 integers.
        return range(100)

      return dataset_ops.Dataset.from_generator(fn, output_types=dtypes.float32)

    self.assertNoMemoryLeak(get_dataset)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_parallel_calls=[None, 10])))
  def testMap(self, num_parallel_calls):
    # [FUNCTION DOC]
    # Eager only, with and without parallel calls. Scenario: map squaring a range. Asserts no leaks.

    def get_dataset():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset mapped by squaring with the test's parallelism setting.

      def fn(x):
        # [FUNCTION DOC]
        # Mapped function: returns the input squared.
        return x * x

      return dataset_ops.Dataset.range(0, 100).map(
          fn, num_parallel_calls=num_parallel_calls)

    self.assertNoMemoryLeak(get_dataset)

  @combinations.generate(
      combinations.combine(
          tf_api_version=1, mode="eager", num_parallel_calls=[None, 10]))
  def testMapLegacy(self, num_parallel_calls):
    # [FUNCTION DOC]
    # Eager, API version 1, with and without parallelism. Same as the map test using the legacy map
    # variant.

    def get_dataset():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset mapped by squaring with the legacy map method.

      def fn(x):
        # [FUNCTION DOC]
        # Mapped function: returns the input squared.
        return x * x

      return dataset_ops.Dataset.range(0, 100).map_with_legacy_function(
          fn, num_parallel_calls=num_parallel_calls)

    self.assertNoMemoryLeak(get_dataset)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_parallel_calls=[None, 10])))
  def testInterleave(self, num_parallel_calls):
    # [FUNCTION DOC]
    # Eager only, with and without parallel calls. Scenario: interleave of single-element datasets with
    # cycle length ten. Asserts no leaks.

    def get_dataset():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset interleaved with the function, cycle length ten.

      def fn(x):
        # [FUNCTION DOC]
        # Returns a one-element dataset holding the input squared.
        return dataset_ops.Dataset.from_tensors(x * x)

      return dataset_ops.Dataset.range(0, 100).interleave(
          fn, num_parallel_calls=num_parallel_calls, cycle_length=10)

    self.assertNoMemoryLeak(get_dataset)


if __name__ == "__main__":
  test.main()
