# [MODULE DOC]
# Unit tests for the dataset map transformation (map and the legacy map-with-legacy-function variant).
# Contains the main map test class (plain, parallel, stateful, captured-resource, nested-structure,
# ragged/sparse/tensor-array, determinism, device and error scenarios), plus checkpoint, random-access
# and global-shuffle test classes for map. Also defines small helper types (dataclasses with custom
# flatten/unflatten) and helper functions that build parameterized test combinations over eager and
# graph modes, API versions 1 and 2, and the new versus legacy map entry points.
# Import-time side effects: tries to import the optional attr package and falls back to a null value
# when it is missing; no flags or registrations. Run as a script it starts the test main loop.
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
"""Tests for `tf.data.Dataset.map()`."""
import collections
import dataclasses
import functools
import threading
import time
from typing import Callable
import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


def _test_combinations_with_mode_v1(mode):
  # [FUNCTION DOC]
  # HELPER: builds parameterized test combinations used by the tests in this module.
  # Builds combinations for API version 1 in the given execution mode (eager or graph) with a named
  # apply_map callable, once using the regular map method and once using the legacy map method.
  # Input: mode, a string naming the execution mode. Returns the concatenated combinations list.

  def new_map_fn(dataset, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: callable wrapped into a named combination object for the tests.
    # Applies the regular map method to the given dataset, forwarding all extra positional and keyword
    # arguments, and returns the resulting dataset.
    return dataset.map(*args, **kwargs)

  def legacy_map_fn(dataset, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: callable wrapped into a named combination object for the tests.
    # Applies the legacy map-with-legacy-function method to the given dataset, forwarding all extra
    # positional and keyword arguments, and returns the resulting dataset.
    return dataset.map_with_legacy_function(*args, **kwargs)

  new_map_combinations = combinations.combine(
      tf_api_version=1,
      mode=mode,
      apply_map=combinations.NamedObject("map_fn", new_map_fn))

  legacy_map_combinations = combinations.combine(
      tf_api_version=1,
      mode=mode,
      apply_map=combinations.NamedObject("legacy_map_fn", legacy_map_fn))

  return new_map_combinations + legacy_map_combinations


def _test_combinations_with_mode_v2(mode):
  # [FUNCTION DOC]
  # HELPER: builds parameterized test combinations used by the tests in this module.
  # Builds combinations for API version 2 in the given execution mode, with only the regular map method
  # as the apply_map callable. Input: mode string. Returns the combinations list.

  def new_map_fn(dataset, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: callable wrapped into a named combination object for the tests.
    # Applies the regular map method to the given dataset, forwarding all extra positional and keyword
    # arguments, and returns the resulting dataset.
    return dataset.map(*args, **kwargs)

  return combinations.combine(
      tf_api_version=2,
      mode=mode,
      apply_map=combinations.NamedObject("map_fn", new_map_fn))


def _test_combinations_with_mode(mode):
  # [FUNCTION DOC]
  # HELPER: builds parameterized test combinations used by the tests in this module.
  # Concatenates the version 1 and version 2 combinations for the given execution mode.
  # Input: mode string. Returns the combined list.
  return _test_combinations_with_mode_v1(
      mode) + _test_combinations_with_mode_v2(mode)


def _test_combinations():
  # [FUNCTION DOC]
  # HELPER: builds parameterized test combinations used by the tests in this module.
  # Takes no inputs. Returns the version 1 and version 2 combinations for both the eager and graph
  # modes joined together; this is the default parameterization for most map tests.
  return _test_combinations_with_mode("eager") + _test_combinations_with_mode(
      "graph")


def _short_circuit_test_cases():
  # [FUNCTION DOC]
  # HELPER: supplies the parameter cases for the short-circuit map test.
  # Takes no inputs. Returns combinations of an element structure and a map function for four simple
  # functions: identity, replicate (return the input twice), swap (swap two inputs) and project (keep
  # the first of two inputs). Each function is wrapped in a named object so test names are readable.
  # The cases are folded together with a reduce over the case list.
  cases = [
      ("Identity", None, lambda x: x),
      ("Replicate", None, lambda x: (x, x)),
      ("Swap", (None, None), lambda x, y: (y, x)),
      ("Project", (None, None), lambda x, y: x)
  ]

  def reduce_fn(x, y):
    # [FUNCTION DOC]
    # HELPER: accumulator step for building the short-circuit cases.
    # Inputs: the accumulated list of combinations and a case triple of name, structure and function.
    # Returns the accumulated list extended with one combination for that case.
    name, structure, fn = y
    return x + combinations.combine(
        structure=structure, fn=combinations.NamedObject(name, fn))

  return functools.reduce(reduce_fn, cases, [])


class Foo:
  # [CLASS DOC]
  # Dummy class used as an invalid return value from a map function so the type check on map outputs
  # fails. It holds no state and is never meaningfully instantiated by the tests.
  """Dummy class used for invalid return value tests."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor that does nothing; takes no arguments besides the instance and sets no attributes.
    pass


@dataclasses.dataclass
class MyDataclass:
  # [CLASS DOC]
  # Small data class with two tensor fields, used to test mapping over user-defined composite types.
  # It implements the flatten and unflatten protocol so the data pipeline can decompose it into its
  # two components and rebuild it. No metadata is carried.
  value1: tensor.Tensor
  value2: tensor.Tensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # Part of the flatten protocol. Returns a pair: empty metadata and a tuple holding the two field
    # values in order. Takes only the instance.
    metadata = tuple()
    components = (self.value1, self.value2)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    # [FUNCTION DOC]
    # Class method forming the inverse of the flatten protocol. Inputs: metadata (ignored and discarded)
    # and the components sequence. Returns a new instance whose two fields come from the first and second
    # components.
    del metadata
    return cls(value1=components[0], value2=components[1])


@dataclasses.dataclass
class MaskedTensor:
  # [CLASS DOC]
  # Data class holding a boolean mask and a tensor value, used to test mapping over composite types
  # whose metadata (the mask) is carried separately from the tensor component. Implements the flatten
  # and unflatten protocol.
  mask: bool
  value: tensor.Tensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # Part of the flatten protocol. Returns a pair: metadata containing the mask, and a one-element tuple
    # containing the value.
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    # [FUNCTION DOC]
    # Class method inverse of the flatten protocol. Inputs: metadata (holding the mask) and components
    # (holding the value). Returns a new masked tensor built from them.
    mask = metadata[0]
    value = components[0]
    return MaskedTensor(mask=mask, value=value)


@dataclasses.dataclass
class NestedMaskedTensor:
  # [CLASS DOC]
  # Data class holding a boolean mask and a masked-tensor value, used to test nested composite types
  # in map functions. Implements the flatten and unflatten protocol and value equality.
  mask: bool
  value: MaskedTensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # Part of the flatten protocol. Returns a pair: metadata containing the mask, and a one-element tuple
    # containing the inner masked tensor.
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    # [FUNCTION DOC]
    # Class method inverse of the flatten protocol. Inputs: metadata (holding the mask) and components
    # (holding the inner value). Returns a new nested masked tensor built from them.
    mask = metadata[0]
    value = components[0]
    return NestedMaskedTensor(mask=mask, value=value)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality: true when the other object has the same mask and an equal inner value. Takes one other
    # object; assumes it has mask and value attributes (otherwise an attribute error would be raised).
    return self.mask == other.mask and self.value == other.value


class MapTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for the map transformation. Parameterized over eager and graph modes and over the
  # regular and legacy map entry points through an apply_map argument. Covers sequential and parallel
  # map, multi-threaded iteration, error handling, captured resources (iterators, tables, queues,
  # variables), statefulness and seeding, dataclass/namedtuple/attrs structures, sparse, ragged and
  # tensor-array values, determinism, short-circuiting, cancellation, devices and naming.

  def _map_dataset_factory(self, components, apply_map, count):
    # [FUNCTION DOC]
    # HELPER: builds the dataset for the basic map tests.
    # Inputs: components (tuple of numpy arrays sliced along the first axis), apply_map (callable applying
    # map) and count (number of repeats). Builds a dataset from the slices, maps a function that squares
    # each of the three components, repeats it count times, and asserts the static output shapes equal
    # the per-element component shapes. Returns the dataset.

    def _map_fn(x, y, z):
      # [FUNCTION DOC]
      # Mapped function: takes three tensors and returns a tuple of their element-wise squares.
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(dataset, _map_fn).repeat(count)
    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    return dataset

  @combinations.generate(_test_combinations())
  def testMapDataset(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map a squaring function over three components of a sliced dataset and repeat it 14 times.
    # Reads the dataset sequentially with a single iterator and asserts each element equals the square of
    # the matching input slice, in order, then asserts the end of the sequence raises an out-of-range
    # error.
    """Test an dataset that maps a TF function across its input elements."""
    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(count).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    # Test single-threaded access to the iterator.
    get_next = self.getNext(
        self._map_dataset_factory(components, apply_map, count=14))
    for _ in range(14):
      for i in range(7):
        result = self.evaluate(get_next())
        for component, result_component in zip(components, result):
          self.assertAllEqual(component[i]**2, result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # TODO(b/117581999): add eager coverage
  @combinations.generate(_test_combinations_with_mode("graph"))
  def testMapDatasetMultiThreaded(self, apply_map):
    # [FUNCTION DOC]
    # Graph-mode only. Scenario: the same squaring map dataset repeated 18 times is consumed by eight
    # threads sharing one iterator. Results are collected, sorted by the first component and each input
    # square is asserted to appear exactly 18 times. Order is not deterministic so sorting is used.
    # Test multi-threaded access to the same iterator.
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    get_next = self.getNext(
        self._map_dataset_factory(components, apply_map, count=18))
    results = []
    with self.cached_session() as sess:
      def iterator_thread():
        # [FUNCTION DOC]
        # HELPER: thread body for the multi-threaded consumption test.
        # Loops fetching the next element from the shared iterator within the session and appends it to the
        # shared results list until an out-of-range error is raised, then returns.
        while True:
          try:
            results.append(sess.run(get_next()))
          except errors.OutOfRangeError:
            return
      threads = [self.checkedThread(target=iterator_thread) for _ in range(8)]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      # `results` will contain the same elements components**2
      # repeated 18 times, but in a non-deterministic order. Sort the
      # results, and assert that each element of components**2 is
      # produced 18 times.
      results.sort(key=lambda x: x[0])
      for i in range(7):
        for j in range(18):
          for component, result_component in zip(components,
                                                 results[i * 18 + j]):
            self.assertAllEqual(component[i]**2, result_component)

  def _parallel_map_dataset_factory(self, components, apply_map, count,
                                    num_parallel_calls, buffer_size,
                                    use_unbounded_threadpool=False):
    # [FUNCTION DOC]
    # HELPER: builds the dataset for the parallel map tests.
    # Inputs: components, apply_map, count, num_parallel_calls, buffer_size and optional
    # use_unbounded_threadpool (default off). Slices the components, applies a parallel squaring map
    # with the given settings, prefetches buffer_size elements, repeats count times and asserts the
    # static output shapes match the component shapes. Returns the dataset.

    def _map_fn(x, y, z):
      # [FUNCTION DOC]
      # Mapped function: takes three tensors and returns a tuple of their element-wise squares.
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(dataset, _map_fn, num_parallel_calls=num_parallel_calls,
                        use_unbounded_threadpool=use_unbounded_threadpool)
    dataset = dataset.prefetch(buffer_size).repeat(count)

    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    return dataset

  @combinations.generate(
      combinations.times(
          _test_combinations(),
          combinations.combine(num_parallel_calls=1, buffer_size=1) +
          combinations.combine(num_parallel_calls=1, buffer_size=2) +
          combinations.combine(num_parallel_calls=2, buffer_size=2) +
          combinations.combine(num_parallel_calls=2, buffer_size=4) +
          combinations.combine(num_parallel_calls=8, buffer_size=8) +
          combinations.combine(num_parallel_calls=8, buffer_size=16),
          combinations.combine(use_unbounded_threadpool=[None, True, False])))
  def testParallelMapDataset(self, apply_map, num_parallel_calls, buffer_size,
                             use_unbounded_threadpool):
    # [FUNCTION DOC]
    # Scenario: parallel map with a grid of parallelism, prefetch buffer and unbounded-thread-pool
    # settings over the squaring pipeline repeated 14 times. Consumes it sequentially and asserts every
    # element equals the square of the input slice in order and that the sequence then ends with an
    # out-of-range error.
    """Test an dataset that maps a TF function across its input elements."""

    # The pipeline is TensorSliceDataset -> ParallelMapDataset(square_3) ->
    # RepeatDataset(count).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    # Test single-threaded access to the iterator.
    get_next = self.getNext(
        self._parallel_map_dataset_factory(components, apply_map, 14,
                                           num_parallel_calls, buffer_size,
                                           use_unbounded_threadpool))
    for _ in range(14):
      for i in range(7):
        result = self.evaluate(get_next())
        for component, result_component in zip(components, result):
          self.assertAllEqual(component[i]**2, result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # TODO(b/117581999): add eager coverage
  @combinations.generate(
      combinations.times(
          _test_combinations_with_mode("graph"),
          combinations.combine(num_parallel_calls=1, buffer_size=1) +
          combinations.combine(num_parallel_calls=1, buffer_size=2) +
          combinations.combine(num_parallel_calls=2, buffer_size=2) +
          combinations.combine(num_parallel_calls=2, buffer_size=4) +
          combinations.combine(num_parallel_calls=8, buffer_size=8) +
          combinations.combine(num_parallel_calls=8, buffer_size=16)))
  def testParallelMapDatasetMultiThreaded(self, apply_map, num_parallel_calls,
                                          buffer_size):
    # [FUNCTION DOC]
    # Graph-mode only. Scenario: parallel map with several parallelism and buffer settings, repeated 18
    # times, consumed by 64 threads on one shared iterator. Sorts the collected results and asserts each
    # squared input appears 18 times.

    # Test multi-threaded access to the same iterator.
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    get_next = self.getNext(
        self._parallel_map_dataset_factory(components, apply_map, 18,
                                           num_parallel_calls, buffer_size))
    results = []
    with self.cached_session() as sess:

      def iterator_thread():
        # [FUNCTION DOC]
        # HELPER: thread body for the multi-threaded consumption test.
        # Repeatedly fetches the next element from the shared iterator within the session, appends it to the
        # shared results list and stops on an out-of-range error.
        while True:
          try:
            results.append(sess.run(get_next()))
          except errors.OutOfRangeError:
            return

      threads = [self.checkedThread(target=iterator_thread) for _ in range(64)]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      # `results` will contain the same elements components**2
      # repeated 18 times, but in a non-deterministic order. Sort the
      # results, and assert that each element of components**2 is
      # produced 18 times.
      results.sort(key=lambda x: x[0])
      for i in range(7):
        for j in range(18):
          for component, result_component in zip(components,
                                                 results[i * 18 + j]):
            self.assertAllEqual(component[i]**2, result_component)

  @combinations.generate(_test_combinations())
  def testImplicitDisposeParallelMapDataset(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a large parallel map pipeline with prefetching is only partly consumed (three elements)
    # and then abandoned. Passes if building and partial consumption succeed and cleanup of background
    # threads does not hang or crash; there are no explicit assertions.
    # Tests whether a parallel map dataset will be cleaned up correctly when
    # the pipeline does not run it until exhaustion.
    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(1000).
    components = (np.arange(1000),
                  np.array([[1, 2, 3]]) * np.arange(1000)[:, np.newaxis],
                  np.array(37.0) * np.arange(1000))

    dataset = self._parallel_map_dataset_factory(components, apply_map, 1000,
                                                 100, 100)
    # NOTE(mrry): Also test that the prefetching thread is cancelled correctly.
    dataset = dataset.prefetch(100)
    get_next = self.getNext(dataset)

    for _ in range(3):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testParallelMapUnspecifiedOutputSize(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: parallel map (two calls) applies a numeric check to a float vector containing a not-a-number
    # value. Only the first three elements, which precede the bad value, are fetched; nothing is asserted
    # beyond those fetches succeeding.
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(
        dataset,
        lambda x: array_ops.check_numerics(x, "message"),
        num_parallel_calls=2)
    get_next = self.getNext(dataset)

    for _ in range(3):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testParallelMapError(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: parallel map applies a numeric check to values with a not-a-number fourth element.
    # Asserts the first three elements succeed, the fourth raises an invalid-argument error, the next
    # fetch succeeds (iteration continues after the error) and the end raises out-of-range.
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(
        dataset,
        lambda x: array_ops.check_numerics(x, "message"),
        num_parallel_calls=2)
    get_next = self.getNext(dataset)

    for _ in range(3):
      self.evaluate(get_next())
    # The 4th element is NaN, so `array_ops.check_numerics()` should fail.
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testPrefetchError(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: like the parallel error test but with a plain map followed by prefetch of two. Asserts
    # three good elements, an invalid-argument error for the not-a-number element, a successful next
    # fetch and then an out-of-range error.
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(
        dataset, lambda x: array_ops.check_numerics(x, "message")).prefetch(2)
    get_next = self.getNext(dataset)

    for _ in range(3):
      self.evaluate(get_next())
    # The 4th element is NaN, so `array_ops.check_numerics()` should fail.
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaptureIterator(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: the map function captures an iterator over a range and multiplies each element by the
    # captured iterator's next value. Asserts results are squares of 0 through 9 and the end raises an
    # out-of-range error. In graph mode the captured iterator is initialized first.

    def _build_ds(iterator):
      # [FUNCTION DOC]
      # HELPER: builds the mapped dataset for the capture-iterator test.
      # Input: the iterator to capture. Returns a mapped range dataset whose function multiplies each
      # element by the next value of that iterator.

      def _map_fn(x):
        # [FUNCTION DOC]
        # Mapped function: gets the next value from the captured iterator and returns the input times it.
        get_next = iterator.get_next()
        return x * get_next

      return apply_map(dataset_ops.Dataset.range(10), _map_fn)

    def _build_graph():
      # [FUNCTION DOC]
      # HELPER: creates the captured iterator and mapped dataset.
      # Takes no inputs. In eager mode uses a Python iterator over a range, otherwise an initializable
      # iterator. Returns a pair of the captured iterator and the mapped dataset.
      if context.executing_eagerly():
        captured_iterator = iter(dataset_ops.Dataset.range(10))
      else:
        captured_iterator = dataset_ops.make_initializable_iterator(
            dataset_ops.Dataset.range(10))
      ds = _build_ds(captured_iterator)
      return captured_iterator, ds

    captured_iter, ds = _build_graph()
    if not context.executing_eagerly():
      self.evaluate(captured_iter.initializer)
    get_next = self.getNext(ds, requires_initialization=True)
    for i in range(10):
      self.assertEqual(i * i, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaptureHashTable(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a lookup hash table is captured by a second map. First map splits sentences into words,
    # second maps words to table ids. Initializes the table, fetches both elements and asserts the end
    # of the sequence raises out-of-range.
    # NOTE(mrry): We must use the V2 variants of `HashTable`
    # etc. because these produce a `tf.resource`-typed output that is
    # compatible with the in-graph function implementation.
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.HashTable(
        lookup_ops.KeyValueTensorInitializer(keys, values), default_val)

    input_sentences = dataset_ops.Dataset.from_tensor_slices(
        ["brain brain tank salad surgery", "surgery brain"])

    dataset = apply_map(input_sentences,
                        lambda x: string_ops.string_split([x]).values)
    dataset = apply_map(dataset, table.lookup)

    get_next = self.getNext(dataset, requires_initialization=True)

    self.evaluate(table.initializer)
    self.evaluate(get_next())
    self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations_with_mode("graph"))
  def testCaptureQueue(self, apply_map):
    # [FUNCTION DOC]
    # Graph-mode only. Scenario: the map function dequeues from a captured queue filled with 200 random
    # numbers and closed. Asserts the elements come out in order. After exhaustion the dequeue's
    # out-of-range condition surfaces as an invalid-argument error under version 2 (cardinality
    # preserved) and as an out-of-range error under version 1.
    elements = np.random.randint(100, size=[200])
    queue = data_flow_ops.FIFOQueue(200, dtypes.int64, shapes=[])
    enqueue_op = queue.enqueue_many(elements)
    close_op = queue.close()
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(-1)
    dataset = apply_map(dataset, lambda _: queue.dequeue())

    get_next = self.getNext(dataset, requires_initialization=True)
    self.evaluate(enqueue_op)
    self.evaluate(close_op)

    for element in elements:
      self.assertEqual(element, self.evaluate(get_next()))
    # When the map function in `MapDataset` raises an OutOfRange error, TF1 and
    # TF2 behave differently. TF1 raises an OutOfRangeError to signal the end of
    # sequence while TF2 raises an InvalidArgumentError. This behavior is
    # controlled by the `preserve_cardinality` argument of `map` transformation
    # which is set to `True` for TF2 and `False` for TF1, which is for backward
    # compatibility.
    if tf2.enabled():
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(get_next())
    else:
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  # TODO(b/117581999): Possible deadlock in eager mode, debug.
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testCaptureSameResourceMultipleTimes(self, apply_map):
    # [FUNCTION DOC]
    # Graph-mode, API version 1 only. Scenario: two queue handles share a name and are both dequeued by
    # the map function. Asserts the paired outputs match consecutive element pairs of the queue (compared
    # without regard to order inside a pair) for 100 rounds, then an out-of-range error.
    elements = np.random.randint(100, size=[200])
    queue = data_flow_ops.FIFOQueue(
        200, dtypes.int64, shapes=[], shared_name="shared_queue")
    queue_2 = data_flow_ops.FIFOQueue(
        200, dtypes.int64, shapes=[], shared_name="shared_queue")

    enqueue_op = queue.enqueue_many(elements)
    close_op = queue.close()

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(-1)
    dataset = apply_map(dataset, lambda _: (queue.dequeue(), queue_2.dequeue()))

    self.evaluate(enqueue_op)
    self.evaluate(close_op)
    get_next = self.getNext(dataset, requires_initialization=True)
    for i in range(100):
      self.assertCountEqual([elements[i * 2], elements[i * 2 + 1]],
                            self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testSeededStatefulOperatorIsProperlyStateful(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a map with a seeded random function batched by two. Asserts ten values are produced, that
    # they are not all equal (stateful, not constant), and that a second iteration after exhaustion
    # yields the same values (repeatable given the seed).
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
    fn = lambda _: random_ops.random_uniform((), seed=11)
    dataset = apply_map(dataset, fn).batch(2)

    get_next = self.getNext(dataset, requires_initialization=True)
    random_values = []
    with self.assertRaises(errors.OutOfRangeError):
      while True:
        random_values.extend(self.evaluate(get_next()))
    self.assertLen(random_values, 10)
    self.assertGreater(np.abs(np.diff(random_values)).max(), 1e-6)

    get_next = self.getNext(dataset, requires_initialization=True)
    random_values_2 = []
    with self.assertRaises(errors.OutOfRangeError):
      while True:
        random_values_2.extend(self.evaluate(get_next()))

    # Randomness is repeatable given same seed
    self.assertAllClose(random_values, random_values_2)

  @combinations.generate(_test_combinations())
  def testStatefulMapKeepsStateAcrossIterators(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a seeded random map is repeated 1000 times and batched. Asserts that within the first 99
    # following batches at least one differs from the first, showing random state persists across
    # repeats.
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
    fn = lambda _: random_ops.random_uniform((), seed=11)
    dataset = apply_map(dataset, fn).repeat(1000).batch(10)

    get_next = self.getNext(dataset)
    random_values = self.evaluate(get_next())

    # Assert that one of the next 99 batches yielded by the iterator is
    # different from the first.
    i = 0
    while i < 99:
      if np.any(random_values != self.evaluate(get_next())):
        break
      i += 1
    self.assertLess(i, 99)

  @combinations.generate(_test_combinations())
  def testStatefulOperationInShortCircuit(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map function increments a counter variable and returns its input unchanged, with
    # prefetch injection disabled. Asserts the counter equals the number of elements consumed after each
    # step, so the stateful side effect is not optimized away, and that it stays at ten after the end.
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)

    def increment_fn(x):
      # [FUNCTION DOC]
      # Mapped function: increments the captured counter variable by one and returns the input unchanged.
      counter_var.assign_add(1)
      return x

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, increment_fn)
    options = options_lib.Options()
    options.experimental_optimization.inject_prefetch = False
    dataset = dataset.with_options(options)

    get_next = self.getNext(dataset, requires_initialization=True)

    self.evaluate(counter_var.initializer)
    for i in range(10):
      self.assertEqual(i, self.evaluate(counter_var))
      self.assertEqual(i, self.evaluate(get_next()))
    self.assertEqual(10, self.evaluate(counter_var))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertEqual(10, self.evaluate(counter_var))

  @combinations.generate(_test_combinations())
  def testMapDict(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map a range into a dictionary of two keys, then map the dictionary to the sum of its
    # values. Asserts the output equals twice the input plus its square.
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: {"foo": x * 2, "bar": x**2})
    dataset = apply_map(dataset, lambda d: d["foo"] + d["bar"])
    self.assertDatasetProduces(
        dataset, expected_output=[i * 2 + i**2 for i in range(10)])

  @combinations.generate(_test_combinations())
  def testMapDataclass(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map integers into the two-field data class and then map the class to the sum of its
    # fields. Asserts outputs are three times the input.
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: MyDataclass(value1=x, value2=2 * x))
    dataset = apply_map(dataset, lambda x: x.value1 + x.value2)
    self.assertDatasetProduces(
        dataset,
        expected_output=[3 * x for x in range(10)],
    )

  @combinations.generate(_test_combinations())
  def testMapMaskedTensor(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map integers into a masked tensor and then to three times its value. Asserts the outputs
    # are three times the inputs.
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: MaskedTensor(mask=True, value=x))
    dataset = apply_map(dataset, lambda x: 3 * x.value)
    self.assertDatasetProduces(
        dataset,
        expected_output=[3 * x for x in range(10)],
    )

  @combinations.generate(_test_combinations())
  def testMapDataclassWithInputAndOutput(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a dataset whose single element is a data class is mapped to a tuple, then to a masked
    # tensor, then to a nested masked tensor. Asserts the single output equals the expected nested
    # object with value seven, comparing via value equality.
    dataset = dataset_ops.Dataset.from_tensors(MyDataclass(value1=1, value2=2))
    dataset = apply_map(dataset, lambda x: (x.value1 * 5, x.value2))
    dataset = apply_map(
        dataset, lambda x, y: MaskedTensor(mask=True, value=x + y)
    )
    dataset = apply_map(
        dataset, lambda m: NestedMaskedTensor(mask=False, value=m)
    )
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            NestedMaskedTensor(
                mask=False, value=MaskedTensor(mask=True, value=7)
            )
        ],
    )

  @combinations.generate(_test_combinations())
  def testMapListOfDataclassObjects(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map returns a list of two data class objects and the next map takes them as separate
    # arguments and sums selected fields. Asserts outputs are three times the input.
    dataset = dataset_ops.Dataset.range(10)

    # Creates a list of dataclass objects.
    dataset = apply_map(
        dataset,
        lambda x: [  # pylint: disable=g-long-lambda
            MyDataclass(value1=x, value2=1),
            MyDataclass(value1=2, value2=2 * x),
        ],
    )

    # Takes a list of dataclass objects as input.
    dataset = apply_map(dataset, lambda *x: x[0].value1 + x[1].value2)

    self.assertDatasetProduces(
        dataset,
        expected_output=[3 * x for x in range(10)],
    )

  @combinations.generate(_test_combinations())
  def testMapDictOfDataclassValues(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map returns a dictionary of data class values, and a following map consumes the
    # dictionary. Asserts outputs are three times the input.
    dataset = dataset_ops.Dataset.range(10)

    # Creates a dict of {str -> dataclass}.
    dataset = apply_map(
        dataset,
        lambda x: {  # pylint: disable=g-long-lambda
            "a": MyDataclass(value1=x, value2=1),
            "b": MyDataclass(value1=2, value2=2 * x),
        },
    )
    # Takes a dict of dataclass values as input.
    dataset = apply_map(dataset, lambda x: x["a"].value1 + x["b"].value2)
    self.assertDatasetProduces(
        dataset,
        expected_output=[3 * x for x in range(10)],
    )

  @combinations.generate(_test_combinations())
  def testMapNestedMaskedTensorWithDataclassInput(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map a range to masked tensors, then to nested masked tensors, then to five times the
    # inner value. Asserts the outputs equal five times the input.
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: MaskedTensor(mask=True, value=x))
    dataset = apply_map(
        dataset,
        # Takes a MaskedTensor as input.
        lambda x: NestedMaskedTensor(mask=False, value=x),
    )
    dataset = apply_map(dataset, lambda x: 5 * x.value.value)
    self.assertDatasetProduces(
        dataset,
        expected_output=[5 * x for x in range(10)],
    )

  @combinations.generate(_test_combinations())
  def testMapNestedMaskedTensorWithDataclassOutput(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map creates nested masked tensors, then returns the inner masked tensor, then returns
    # seven times its value. Asserts outputs equal seven times the input.
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(
        dataset,
        lambda x: NestedMaskedTensor(  # pylint: disable=g-long-lambda
            mask=False, value=MaskedTensor(mask=True, value=x)
        ),
    )

    # Return a MaskedTensor as the return value.
    dataset = apply_map(dataset, lambda x: x.value)
    dataset = apply_map(dataset, lambda x: 7 * x.value)
    self.assertDatasetProduces(
        dataset,
        expected_output=[7 * x for x in range(10)],
    )

  @combinations.generate(_test_combinations())
  def testMapNamedtuple(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a dataset of tuples is converted into a dataset of named tuples through map and both are
    # preprocessed (the image field doubled). Asserts both datasets produce equal data, matching the
    # expected label and doubled negative image, and the named tuple dataset then ends out-of-range.
    # construct dataset of tuples
    labels = dataset_ops.Dataset.range(10)
    images = apply_map(labels, lambda l: -l)
    dataset_tuple = dataset_ops.Dataset.zip((labels, images))

    # convert dataset of tuples to dataset of namedtuples
    example = collections.namedtuple("Example", ["label", "image"])
    dataset_namedtuple = apply_map(dataset_tuple, example)

    def preprocess_tuple(label, image):
      # [FUNCTION DOC]
      # Mapped function: takes label and image, doubles the image and returns the pair.
      image = 2 * image
      return label, image

    def preprocess_namedtuple(example):
      # [FUNCTION DOC]
      # Mapped function: takes an example named tuple and returns a copy with the image field doubled.
      return example._replace(image=2 * example.image)

    # preprocess both datasets
    dataset_tuple = apply_map(dataset_tuple, preprocess_tuple)
    dataset_namedtuple = apply_map(dataset_namedtuple, preprocess_namedtuple)

    next_tuple = self.getNext(dataset_tuple)
    next_namedtuple = self.getNext(dataset_namedtuple)

    # make sure both datasets contain the same data
    for i in range(10):
      tuple_, namedtuple_ = self.evaluate([next_tuple(), next_namedtuple()])
      self.assertEqual(tuple_, namedtuple_)
      self.assertEqual(tuple_, (i, -2 * i))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_namedtuple())

  @combinations.generate(_test_combinations())
  def testMapAttrs(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: skipped when the attrs package is unavailable. Zips labels with negated labels, maps
    # them into an attrs class, then maps a function that mutates the image field. Asserts each output
    # equals the expected object and the end raises out-of-range.
    if attr is None:
      self.skipTest("attr module is not available.")

    # construct dataset of tuples
    labels = dataset_ops.Dataset.range(10)
    images = apply_map(labels, lambda l: -l)
    dataset = dataset_ops.Dataset.zip((labels, images))

    @attr.s(cmp=True)
    class Example:
      # [CLASS DOC]
      # Local attrs class with two attributes, label and image, with comparison enabled; used as the
      # element type in the attrs mapping test.
      label = attr.ib()
      image = attr.ib()

    dataset = apply_map(dataset, Example)

    def preprocess(example):
      # [FUNCTION DOC]
      # Mapped function: doubles the image attribute of the example in place and returns it.
      example.image = 2 * example.image
      return example

    dataset = apply_map(dataset, preprocess)
    get_next = self.getNext(dataset)

    for i in range(10):
      data = self.evaluate(get_next())
      self.assertEqual(data, Example(i, -2 * i))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testUseStepContainerInMap(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a map function itself calls the functional map-over-elements operation over a row of six
    # numbers, squaring each. Asserts the single output equals the squared row.
    row = np.arange(6)
    dataset = dataset_ops.Dataset.from_tensors(row)
    dataset = apply_map(dataset,
                        lambda elems: map_fn.map_fn(lambda x: x * x, elems))
    self.assertDatasetProduces(dataset, expected_output=[row**2])

  @combinations.generate(_test_combinations())
  def testCaseAndCondInMap(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map function uses a case construct whose branch contains a conditional, for several
    # selector values. Asserts the outputs match the expected arithmetic (doubling or halving) for each
    # selector and that each dataset ends with out-of-range.

    def control_map_fn(x, y):
      # [FUNCTION DOC]
      # HELPER: builds the control flow used by the test.
      # Inputs: x (element) and y (selector). Defines branch functions to double x, halve x (integer) and a
      # conditional between them based on x being even, and returns the result of a case construct that
      # chooses the conditional when y is 2 or 3 and doubling otherwise.

      def multiply():
        # [FUNCTION DOC]
        # Returns the captured x doubled.
        return x * 2

      def divide():
        # [FUNCTION DOC]
        # Returns the captured x integer-divided by two.
        return x // 2

      def defaults_two():
        # [FUNCTION DOC]
        # Returns a conditional: doubles the captured x if it is even, otherwise halves it.
        return cond.cond(
            math_ops.equal(math_ops.mod(x, 2), 0),
            multiply,
            divide,
            name="cond_mult")

      pred_fn_pairs = [
          (math_ops.logical_or(math_ops.equal(y, 2),
                               math_ops.equal(y, 3)), defaults_two),
      ]

      return control_flow_case.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    def build_dataset(row, num):
      # [FUNCTION DOC]
      # HELPER: creates the mapped dataset for the test.
      # Inputs: row (numpy array) and num (selector). Slices the row into a dataset and maps the control
      # function with that selector. Returns the dataset.
      dataset = dataset_ops.Dataset.from_tensor_slices(row)
      return apply_map(dataset, lambda x: control_map_fn(x, num))

    row = np.arange(6)
    for num in [2, 3, 4]:
      get_next = self.getNext(build_dataset(row, num))
      for i in range(6):
        self.assertEqual(
            (i // 2 if i % 2 else i * 2) if (num == 2 or num == 3) else i * 2,
            self.evaluate(get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaseInWhileInMap(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a case construct inside a functional map over a whole row inside a dataset map. Asserts
    # the output equals the row halved for selectors 2 and 3, otherwise doubled, followed by out-of-range.

    def control_map_fn(x, y):
      # [FUNCTION DOC]
      # HELPER: builds the control flow used by the test.
      # Inputs: x and y (selector). Returns a case construct that halves x when y is 2 or 3, otherwise
      # doubles x.

      def multiply():
        # [FUNCTION DOC]
        # Returns the captured x doubled.
        return x * 2

      def divide():
        # [FUNCTION DOC]
        # Returns the captured x integer-divided by two.
        return x // 2

      pred_fn_pairs = [
          (math_ops.logical_or(math_ops.equal(y, 2),
                               math_ops.equal(y, 3)), divide),
      ]

      return control_flow_case.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    def build_dataset(row, num):
      # [FUNCTION DOC]
      # HELPER: creates the mapped dataset for the test.
      # Inputs: row and num. Makes a single-element dataset of the row and maps a functional map applying
      # the control function to each row entry. Returns the dataset.
      dataset = dataset_ops.Dataset.from_tensors(row)
      return apply_map(
          dataset,
          lambda elems: map_fn.map_fn(lambda x: control_map_fn(x, num), elems))

    row = np.arange(6)
    for num in [2, 3, 4]:
      get_next = self.getNext(build_dataset(row, num))
      self.assertAllEqual(
          [x // 2 if (num == 2 or num == 3) else x * 2 for x in row],
          self.evaluate(get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaseAndCondInWhileInMap(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a case with an inner conditional used inside a functional map in a dataset map, for
    # selector 2. Asserts the single output equals the expected halved or doubled values by parity, then
    # out-of-range.

    def control_map_fn(x, y):
      # [FUNCTION DOC]
      # HELPER: builds the control flow used by the test.
      # Inputs: x and y (selector). Returns a case construct selecting a parity conditional when y is 2 or 3
      # and doubling otherwise.

      def multiply():
        # [FUNCTION DOC]
        # Returns the captured x doubled.
        return x * 2

      def divide():
        # [FUNCTION DOC]
        # Returns the captured x integer-divided by two.
        return x // 2

      def defaults_two():
        # [FUNCTION DOC]
        # Returns a conditional: doubles the captured x if even, otherwise halves it.
        return cond.cond(
            math_ops.equal(math_ops.mod(x, 2), 0),
            multiply,
            divide,
            name="cond_mult")

      pred_fn_pairs = [
          (math_ops.logical_or(math_ops.equal(y, 2),
                               math_ops.equal(y, 3)), defaults_two),
      ]

      return control_flow_case.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    row = np.arange(6)
    num = 2
    dataset = dataset_ops.Dataset.from_tensors(row)
    dataset = apply_map(
        dataset,
        lambda elems: map_fn.map_fn(lambda x: control_map_fn(x, num), elems))
    get_next = self.getNext(dataset)

    self.assertAllEqual([(x // 2 if x % 2 else x * 2) if
                         (num == 2 or num == 3) else x * 2 for x in row],
                        self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testNestedListMapDataset(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map returns a nested list-and-scalar structure from a three-element vector. Asserts
    # every output is a two-element array and the scalar value one, ten times.
    dataset = dataset_ops.Dataset.from_tensors([0, 1, 2]).repeat(10)
    dataset = apply_map(dataset, lambda a: ([a[1], a[0] + a[2]], a[1]))
    expected_output = [(np.array([1, 2]), 1)] * 10
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(
      combinations.times(_test_combinations(),
                         combinations.combine(buffer_size=[1, 2, 3, 4])))
  def testPrefetch(self, apply_map, buffer_size):
    # [FUNCTION DOC]
    # Scenario: a map function implemented as a Python function call sets an event when it sees input 5.
    # With different prefetch buffer sizes, asserts the event is not set until the expected number of
    # elements has been consumed (derived from the buffer size), waits for it, then checks all squares
    # and the final out-of-range error. Shows prefetch depth indirectly.
    # We will use this event to test that `_map_py_func()` has been invoked a
    # certain number of times (6 times, to be exact) after consuming fewer
    # elements from the iterator.
    ev = threading.Event()

    set_event_during_invocation = 5

    def _map_py_func(x):
      # [FUNCTION DOC]
      # HELPER: Python function wrapped as an op.
      # Sets the event when the input equals the trigger value and returns the input squared.
      if x == set_event_during_invocation:
        ev.set()
      return x * x

    def _map_fn(x):
      # [FUNCTION DOC]
      # Mapped function: wraps the Python function as a graph op over the input tensor and returns it.
      return script_ops.py_func(_map_py_func, [x], x.dtype)

    # We can indirectly observe that varying the buffer size has the intended
    # effect by observing when `ev` is set (on the 6th invocation of
    # `_map_py_func()`).
    # NOTE(mrry): We do not test with `buffer_size ==
    # set_event_during_invocation`, because we must consume at least one element
    # to start the prefetching.
    dataset = dataset_ops.Dataset.range(100)
    dataset = apply_map(dataset, _map_fn).prefetch(buffer_size)
    get_next = self.getNext(dataset)

    event_will_be_set_after_consuming = (
        set_event_during_invocation - buffer_size + 1)

    ev.clear()
    for i in range(event_will_be_set_after_consuming):
      self.assertFalse(ev.is_set())
      self.assertEqual(i * i, self.evaluate(get_next()))
    ev.wait()
    for i in range(event_will_be_set_after_consuming, 100):
      self.assertEqual(i * i, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testReturnList(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map returns a list of an element and a constant. Asserts the list is converted to a tuple
    # element pair of the input and 37.
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: [x, constant_op.constant(37.0)])
    self.assertDatasetProduces(
        dataset, expected_output=[(i, 37.0) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testMultiOutputPyFunc(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a Python function wrapped as an op returns two outputs of different types. Asserts the
    # dataset produces the tuple of input and 37.
    # The `tf.py_func()` op returns a list of tensors for its outputs.
    def _map_fn(x_tensor):
      # [FUNCTION DOC]
      # Mapped function: wraps a Python function as a graph op with two output types and returns its outputs.
      def _map_py_func(x):
        # [FUNCTION DOC]
        # Python function returning its input and a float constant 37.
        return x, np.array(37.0, dtype=np.float64)
      return script_ops.py_func(
          _map_py_func, [x_tensor], [dtypes.int64, dtypes.float64])

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _map_fn)
    self.assertDatasetProduces(
        dataset, expected_output=[(i, 37.0) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testSparse(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map produces a sparse tensor value from each integer. Asserts outputs equal the expected
    # sparse values.

    def _sparse(i):
      # [FUNCTION DOC]
      # HELPER: builds a one-element sparse tensor value (single index at the origin, value scaled by the
      # input, dense shape of one by one) from an integer.
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _sparse)
    self.assertDatasetProduces(
        dataset, expected_output=[_sparse(i) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testSparseChain(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: one map produces sparse tensors and a second asserts its input is sparse and
    # concatenates it with itself. Asserts outputs equal the same computation done directly.

    def _sparse(i):
      # [FUNCTION DOC]
      # HELPER: builds a one-element sparse tensor value from an integer.
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def _check(i):
      # [FUNCTION DOC]
      # HELPER: mapped function that asserts its input is sparse and returns the input concatenated with
      # itself along the first axis.
      self.assertTrue(sparse_tensor.is_sparse(i))
      return sparse_ops.sparse_concat(0, [i, i])

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _sparse)
    dataset = apply_map(dataset, _check)

    self.assertDatasetProduces(
        dataset,
        expected_output=[self.evaluate(_check(_sparse(i))) for i in range(10)])

  @combinations.generate(_test_combinations_with_mode("eager"))
  def testSparseMapShapeInference(self, apply_map):
    # [FUNCTION DOC]
    # Eager mode only. Scenario: a ragged tensor converted to sparse is batched by 32 with remainder
    # dropped and passed through an identity map. Asserts the element spec shape is the fully known 32
    # by 3.
    row_lengths = np.random.randint(0, 4, size=128)
    values = np.ones(np.sum(row_lengths))
    sparse = ragged_tensor.RaggedTensor.from_row_lengths(
        values, row_lengths).to_sparse()
    dataset = dataset_ops.Dataset.from_tensor_slices(sparse)
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = apply_map(dataset, lambda x: x)
    self.assertEqual((32, 3), dataset.element_spec.shape)

  @combinations.generate(_test_combinations_with_mode("eager"))
  def testSparseMapShapeInferencePartial(self, apply_map):
    # [FUNCTION DOC]
    # Eager mode only. Scenario: like the previous test but the remainder is kept. Asserts the element
    # spec shape has an unknown first dimension and a known second dimension of three.
    row_lengths = np.random.randint(0, 4, size=128)
    values = np.ones(np.sum(row_lengths))
    sparse = ragged_tensor.RaggedTensor.from_row_lengths(
        values, row_lengths).to_sparse()
    dataset = dataset_ops.Dataset.from_tensor_slices(sparse)
    dataset = dataset.batch(32, drop_remainder=False)
    dataset = apply_map(dataset, lambda x: x)
    self.assertEqual([None, 3], dataset.element_spec.shape.as_list())

  @combinations.generate(_test_combinations())
  def testTensorArray(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map builds a tensor array from each integer holding the numbers below it. Asserts the
    # outputs equal the expected lists.

    def _tensor_array(i):
      # [FUNCTION DOC]
      # HELPER: converts the input to a 32-bit integer and returns a tensor array of that size unstacked
      # from the range of that length.
      i = math_ops.cast(i, dtypes.int32)
      return (
          tensor_array_ops.TensorArray(dtypes.int32, element_shape=(), size=i)
          .unstack(math_ops.range(i, dtype=dtypes.int32)))

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _tensor_array)
    self.assertDatasetProduces(
        dataset, expected_output=[list(range(i)) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testTensorArrayChain(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a map produces tensor arrays and a second map asserts its input is a tensor array and
    # returns its identity. Asserts the outputs are the expected lists.

    def _tensor_array(i):
      # [FUNCTION DOC]
      # HELPER: converts the input to a 32-bit integer and returns a tensor array unstacked from the range.
      i = math_ops.cast(i, dtypes.int32)
      return (
          tensor_array_ops.TensorArray(dtypes.int32, element_shape=(), size=i)
          .unstack(math_ops.range(i, dtype=dtypes.int32)))

    def _check(x):
      # [FUNCTION DOC]
      # HELPER: asserts the input is a tensor array and returns its identity copy.
      self.assertIsInstance(x, tensor_array_ops.TensorArray)
      return x.identity()

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _tensor_array)
    dataset = apply_map(dataset, _check)

    self.assertDatasetProduces(
        dataset,
        expected_output=[list(range(i)) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testRagged(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map builds a ragged tensor from each integer. Asserts outputs equal the expected ragged
    # constants.

    def _ragged(i):
      # [FUNCTION DOC]
      # HELPER: builds a ragged tensor from the integer multiplied by a one-by-one list.
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1]])

    dataset = dataset_ops.Dataset.range(5)
    dataset = apply_map(dataset, _ragged)
    self.assertDatasetProduces(
        dataset,
        expected_output=[ragged_factory_ops.constant([[i]]) for i in range(5)])

  @combinations.generate(_test_combinations())
  def testRaggedChain(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: one map creates ragged tensors and a second asserts raggedness and concatenates each with
    # itself. Asserts the outputs equal the same computation done directly.

    def _ragged(i):
      # [FUNCTION DOC]
      # HELPER: builds a ragged tensor from the integer times a one-by-one list.
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1]])

    def _concat(i):
      # [FUNCTION DOC]
      # HELPER: asserts its input is ragged and returns it concatenated with itself along the first axis.
      self.assertTrue(ragged_tensor.is_ragged(i))
      return ragged_concat_ops.concat([i, i], 0)

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _ragged)
    dataset = apply_map(dataset, _concat)

    self.assertDatasetProduces(
        dataset,
        expected_output=[
            self.evaluate(_concat(ragged_factory_ops.constant([[i]])))
            for i in range(10)
        ])

  @combinations.generate(_test_combinations_with_mode("graph"))
  def testParallelMapOutOfRangeError(self, apply_map):
    # [FUNCTION DOC]
    # Graph-mode only. Scenario: parallel map calls a Python function that raises a stop-iteration at
    # element 100. Asserts the first 100 values pass, then the end is signaled as an invalid-argument
    # error under version 2 and as an out-of-range error under version 1.

    def raising_py_func(i):
      # [FUNCTION DOC]
      # Python function used as an op: raises a stop-iteration when given 100, otherwise returns the input.
      if i == 100:
        raise StopIteration()
      else:
        return i

    dataset = dataset_ops.Dataset.range(105)
    dataset = apply_map(
        dataset,
        lambda x: script_ops.py_func(raising_py_func, [x], dtypes.int64),
        num_parallel_calls=2)
    get_next = self.getNext(dataset)
    for i in range(100):
      self.assertEqual(i, self.evaluate(get_next()))
    # When the map function in `MapDataset` raises an OutOfRange error, TF1 and
    # TF2 behave differently. TF1 raises an OutOfRangeError to signal the end of
    # sequence while TF2 raises an InvalidArgumentError. This behavior is
    # controlled by the `preserve_cardinality` argument of `map` transformation
    # which is set to `True` for TF2 and `False` for TF1, which is for backward
    # compatibility.
    if tf2.enabled():
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(get_next())
    else:
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testConstantOutput(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map returns a list mixing the input, a string and an integer constant. Asserts outputs are
    # tuples of the input, the byte string and ten.
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: [x, "hello", 10])
    self.assertDatasetProduces(dataset, [(i, b"hello", 10) for i in range(10)])

  @combinations.generate(test_base.graph_only_combinations())
  def testWarnOnSeedFromOuterGraph(self):
    # [FUNCTION DOC]
    # Graph-only test. Sets a graph-level seed and records warnings. Asserts that a warning about
    # explicitly setting a seed is issued only when the map function uses randomness without its own seed
    # and the graph seed is set: not for a seed-free function, issued for an unseeded shuffle, not when
    # the function clears the graph seed, and not when a seed is passed explicitly.
    with ops.Graph().as_default() as g:
      g.seed = 10
      warnings.simplefilter("always")

      def _check_warning(caught_warnings, expected_result):
        # [FUNCTION DOC]
        # HELPER: checks whether any caught warning contains the seed advice text and asserts that this
        # equals the expected boolean. Inputs: the list of caught warnings and the expected result.
        found_warning = False
        for warning in caught_warnings:
          if ("Explicitly set the seed in the function if this is not the "
              "intended behavior" in str(warning)):
            found_warning = True
            break
        self.assertEqual(found_warning, expected_result)

      # map_fun doesn't use seed, so no warning is generated.
      with warnings.catch_warnings(record=True) as w:
        _ = dataset_ops.Dataset.range(10).map(math_ops.square)
      _check_warning(w, False)

      def random_func(x):
        # [FUNCTION DOC]
        # Mapped function: adds one, shuffles a list of the value and its square (result unused) and returns
        # the incremented value. Uses randomness without a seed.
        x = math_ops.add(x, 1)
        random_ops.random_shuffle([x, math_ops.square(x)])
        return x

      with warnings.catch_warnings(record=True) as w:
        _ = dataset_ops.Dataset.range(10).map(random_func)
      _check_warning(w, True)

      def random_func_seeded(x):
        # [FUNCTION DOC]
        # Mapped function: clears the default graph seed, shuffles the input and returns the input unchanged.
        # Modifies global graph state.
        ops.get_default_graph().seed = None
        random_ops.random_shuffle(x)
        return x

      with warnings.catch_warnings(record=True) as w:
        _ = dataset_ops.Dataset.range(10).batch(2).map(random_func_seeded)
      _check_warning(w, False)

      with warnings.catch_warnings(record=True) as w:
        _ = dataset_ops.Dataset.range(10).batch(2).map(
            lambda x: random_ops.random_shuffle(x, seed=37))
      _check_warning(w, False)

  @combinations.generate(_test_combinations())
  def testNestedDatasetMap(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map turns a vector into a dataset of slices, a second map batches each dataset by three,
    # then flat-map flattens. Asserts the single output equals the original vector.
    dataset = dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0])
    dataset = apply_map(dataset, dataset_ops.Dataset.from_tensor_slices)
    dataset = apply_map(dataset, lambda ds: ds.batch(3)).flat_map(lambda x: x)

    self.assertDatasetProduces(dataset, expected_output=[[1.0, 2.0, 3.0]])

  @combinations.generate(_test_combinations())
  def testReturnValueError(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map function returns an unsupported plain class object. Asserts the call to map raises a
    # type error whose message states the return value from the function passed to map is unsupported.
    dataset = dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0])
    with self.assertRaisesRegex(
        TypeError, r"Unsupported return value from function passed to "
        r"Dataset.map\(\)"):
      _ = apply_map(dataset, lambda x: Foo)

  @combinations.generate(test_base.default_test_combinations())
  def testBrokenFunctionErrorOnInitialization(self):
    # [FUNCTION DOC]
    # Scenario: a map function builds a constant node whose declared value type disagrees with its dtype.
    # Asserts reading the dataset raises an invalid-argument error mentioning a type mismatch.
    dataset = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0, 3.0])

    def broken_function(_):
      # [FUNCTION DOC]
      # Mapped function deliberately built to fail: creates a malformed empty constant graph node with a
      # mismatched type and returns its output tensor.
      """A function deliberately designed to fail on instantiation."""
      value = []
      tensor_value = attr_value_pb2.AttrValue()
      tensor_value.tensor.CopyFrom(
          tensor_util.make_tensor_proto(
              value, dtype=dtypes.float32, shape=[0], verify_shape=False))
      dtype_value = attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum)

      # Create a "Const" op with a `tf.float32` value and a `tf.int32` type.
      const_tensor = ops.get_default_graph().create_op(
          "Const", [], [dtypes.int32],
          attrs={
              "value": tensor_value,
              "dtype": dtype_value
          },
          name="BrokenConst").outputs[0]
      return const_tensor

    dataset = dataset.map(broken_function)
    self.assertDatasetProduces(
        dataset, expected_error=(errors.InvalidArgumentError, "Type mismatch"))

  @combinations.generate(
      combinations.times(
          _test_combinations_with_mode("graph"),
          combinations.combine(num_parallel_calls=[None, 12])))
  def testNoInterOpParallelism(self, apply_map, num_parallel_calls):
    # [FUNCTION DOC]
    # Graph-mode only. Scenario: the map function runs ten Python function calls and inter-op parallelism
    # is disabled on the dataset's op. Asserts all calls report the same thread identifier.
    dataset = dataset_ops.Dataset.from_tensors(0)

    def _get_tid():
      # [FUNCTION DOC]
      # Python function used as an op: returns the current thread identifier as a 64-bit integer.
      return np.int64(threading.current_thread().ident)

    def _map_fn(_):
      # [FUNCTION DOC]
      # Mapped function: returns a list of ten Python function op results (thread ids).
      tids = []
      for _ in range(10):
        tids.append(script_ops.py_func(_get_tid, [], dtypes.int64))
      return tids

    dataset = apply_map(dataset, _map_fn)
    dataset._variant_tensor.op._set_attr("use_inter_op_parallelism",
                                         attr_value_pb2.AttrValue(b=False))
    get_next = self.getNext(dataset)

    tids = self.evaluate(get_next())
    self.assertTrue(all(tids[0] == tid for tid in tids))

  @combinations.generate(
      combinations.times(_test_combinations(), _short_circuit_test_cases(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testShortCircuit(self, apply_map, structure, fn, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map with an identity, replicate, swap or project function (short-circuit candidates),
    # with and without parallel calls, over a repeated structured dataset. Asserts the first output
    # equals the function applied directly to the structured element.
    dataset = self.structuredDataset(structure).repeat()
    dataset = apply_map(dataset, fn, num_parallel_calls=num_parallel_calls)
    get_next = self.getNext(dataset)

    if isinstance(structure, tuple):
      expected = fn(*self.evaluate(self.structuredElement(structure)))
    else:
      expected = fn(self.evaluate(self.structuredElement(structure)))
    self.assertEqual(expected, self.evaluate(get_next()))

  @combinations.generate(
      combinations.times(_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testShortCircuitCapturedInput(self, apply_map, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map function ignores its input and returns a captured variable. Asserts the output is the
    # variable's value, 42.
    captured_t = variables.Variable(42)
    dataset = self.structuredDataset(None).repeat()
    dataset = apply_map(
        dataset, lambda x: captured_t, num_parallel_calls=num_parallel_calls)
    self.evaluate(variables.global_variables_initializer())
    get_next = self.getNext(dataset, requires_initialization=True)

    self.assertEqual(42, self.evaluate(get_next()))

  @combinations.generate(
      combinations.combine(
          tf_api_version=2,
          mode=["eager", "graph"],
          num_parallel_calls=[None, 12]))
  def testPreserveCardinality(self, num_parallel_calls):
    # [FUNCTION DOC]
    # API version 2 only. Scenario: a Python function raises stop-iteration inside a map, with and
    # without parallelism. Asserts an invalid-argument error is raised instead of end of sequence.

    def py_fn(_):
      # [FUNCTION DOC]
      # Python function used as an op: always raises a stop-iteration.
      raise StopIteration()

    dataset = dataset_ops.Dataset.from_tensors(0).map(
        lambda x: script_ops.py_func(py_fn, [x], dtypes.int64),
        num_parallel_calls=num_parallel_calls)
    get_next = self.getNext(dataset)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations_with_mode("graph"))
  def testCollectionCopy(self, apply_map):
    # [FUNCTION DOC]
    # Graph mode only. Scenario: a variable exists in the global variable collection. The map function
    # asserts that collection still contains the variable when traced. Only builds the dataset.
    w = variable_scope.get_variable("w", [])
    self.assertIn(w, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))

    def func(x):
      # [FUNCTION DOC]
      # Mapped function: asserts the captured variable is in the global variables collection and returns
      # the input.
      self.assertIn(w, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
      return x

    dataset = dataset_ops.Dataset.from_tensors(constant_op.constant(1.0))
    _ = apply_map(dataset, func)

  @combinations.generate(
      combinations.times(
          _test_combinations_with_mode_v1("graph"),
          combinations.combine(num_parallel_calls=[None, 12])))
  def testMapCancellation(self, apply_map, num_parallel_calls):
    # [FUNCTION DOC]
    # Graph mode, API version 1 only. Scenario: the map function dequeues from an empty queue, blocking.
    # A helper thread waits on the next element, then the session is closed. Asserts the pending operation
    # is cancelled.
    # Checks that a cancellation of is threaded through to map transformation.
    queue = data_flow_ops.FIFOQueue(10, dtypes.int32, ())

    def fn(_):
      # [FUNCTION DOC]
      # Mapped function: ignores the input and dequeues from the captured queue.
      return queue.dequeue()

    dataset = dataset_ops.Dataset.range(1)
    dataset = apply_map(dataset, fn, num_parallel_calls=num_parallel_calls)
    get_next = self.getNext(dataset, requires_initialization=True)

    with self.cached_session() as sess:
      thread = self.checkedThread(self.assert_op_cancelled, args=(get_next(),))
      thread.start()
      time.sleep(0.2)
      sess.close()
      thread.join()

  # TODO(b/126553094): map doesnt work with variable defined inside function in
  # eager mode, possible Graph tensors leak out of the function building context
  # from function graph in eager mode as variables are created in init_scope.
  @combinations.generate(test_base.graph_only_combinations())
  def testCreateVariableInsideFunctionWithGetter(self):
    # [FUNCTION DOC]
    # Graph-only test. Scenario: a map function creates a variable inside the function and increments it.
    # For the legacy function path asserts an attribute error because the resource is captured as a
    # tensor. For the regular map, asserts the outputs are the counter values one through ten and then
    # out-of-range.

    def func(_):
      # [FUNCTION DOC]
      # Mapped function: creates or reuses a resource variable named counter in a scope and returns the
      # result of incrementing it.
      with variable_scope.variable_scope(
          "variable", reuse=variable_scope.AUTO_REUSE):
        counter_var = variable_scope.get_variable(
            "counter", (), dtypes.int32, use_resource=True)
      return counter_var.assign_add(1)

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)

    if hasattr(dataset, "map_with_legacy_function"):
      # NOTE: In the legacy function, resource is captured by value.
      with self.assertRaisesWithPredicateMatch(
          AttributeError, ".*Tensor.* object has no attribute 'assign_add'"
      ):
        dataset.map_with_legacy_function(func)

    dataset = dataset.map(func)
    self.evaluate(variables.global_variables_initializer())

    get_next = self.getNext(dataset, requires_initialization=True)

    for i in range(10):
      self.assertEqual(i + 1, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaptureVariable(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: map increments a captured resource variable, prefetch injection disabled. Asserts the
    # variable and output values track the step count each iteration and the variable stays at ten after
    # the end.
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
    dataset = apply_map(dataset, lambda _: counter_var.assign_add(1))
    options = options_lib.Options()
    options.experimental_optimization.inject_prefetch = False
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset, requires_initialization=True)

    self.evaluate(counter_var.initializer)

    for i in range(10):
      self.assertEqual(i, self.evaluate(counter_var))
      self.assertEqual(i + 1, self.evaluate(get_next()))
    self.assertEqual(10, self.evaluate(counter_var))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertEqual(10, self.evaluate(counter_var))

  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testCaptureUninitializedVariableError(self, apply_map):
    # [FUNCTION DOC]
    # Graph mode, API version 1. Scenario: map increments a variable that was never initialized. Asserts
    # fetching an element raises a not-found error.
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
    dataset = apply_map(dataset, lambda _: counter_var.assign_add(1))

    get_next = self.getNext(dataset, requires_initialization=True)
    with self.assertRaises(errors.NotFoundError):
      self.evaluate(get_next())

  # TODO(b/121264236): add eager mode coverage when we have multi-device setup.
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testCaptureConstantsWithConflictingDevices(self, apply_map):
    # [FUNCTION DOC]
    # Graph mode, API version 1. Scenario: two constants are placed on different CPU devices and added
    # in the map function. Asserts every output is 8.
    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.cached_session(config=config):
      with ops.device("/device:CPU:0"):
        a = constant_op.constant(3.0)
      with ops.device("/device:CPU:1"):
        b = constant_op.constant(5.0)

      def func(_):
        # [FUNCTION DOC]
        # Mapped function: ignores the input and returns the sum of the two captured constants.
        return math_ops.add(a, b)

      dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
      dataset = apply_map(dataset, func)
      expected_output = [8.0] * 10
      self.assertDatasetProduces(dataset, expected_output=expected_output)

  # TODO(b/121264236): add eager mode coverage when we have multi-device setup.
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testReferenceVariablesWithMultipleDevices(self, apply_map):
    # [FUNCTION DOC]
    # Graph mode, API version 1. Scenario: two reference variables placed on different CPUs are added in
    # the map function. After initialization asserts all outputs are 8.
    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.cached_session(config=config):

      with ops.device("/device:CPU:0"):
        a = variable_v1.VariableV1(3.0)
      with ops.device("/device:CPU:1"):
        b = variable_v1.VariableV1(5.0)

      def func(_):
        # [FUNCTION DOC]
        # Mapped function: ignores the input and returns the sum of the two captured variables (the nonlocal
        # declaration is unnecessary, since they are only read).
        nonlocal a, b
        return math_ops.add(a, b)

      # NOTE: Use the legacy function implementation as eager function will
      # convert RefVariables to ResourceVariables.
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
      dataset = apply_map(dataset, func)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [8.0] * 10
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  # TODO(b/121264236): add eager mode coverage when we have multi-device setup.
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testResourceVariablesWithMultipleDevices(self, apply_map):
    # [FUNCTION DOC]
    # Graph mode, API version 1. Scenario: the map function creates two resource variables on different
    # CPU devices and adds them (one plus one). In a dedicated graph and session asserts all outputs are 1.
    config = config_pb2.ConfigProto(device_count={"CPU": 3})

    def func(_):
      # [FUNCTION DOC]
      # Mapped function: creates two scoped resource variables on different devices, adds one to the
      # first, and returns their sum.
      with variable_scope.variable_scope(
          "variable", reuse=variable_scope.AUTO_REUSE):
        with ops.device("/device:CPU:0"):
          a_var = variable_scope.get_variable(
              "a", (), dtypes.int32, use_resource=True)
          a_var = math_ops.add(a_var, 1)
        with ops.device("/device:CPU:1"):
          b_var = variable_scope.get_variable(
              "b", (), dtypes.int32, use_resource=True)
      return math_ops.add(a_var, b_var)

    g = ops.Graph()
    with self.session(config=config, graph=g):
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
      dataset = apply_map(dataset, func)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [1] * 10
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(
          _test_combinations(),
          combinations.combine(
              local_determinism=[None, True, False],
              global_determinism=[True, False])))
  def testDeterminismConfiguration(self, apply_map, local_determinism,
                                   global_determinism):
    # [FUNCTION DOC]
    # Scenario: parallel map (two calls) with a per-op determinism setting and a global option. The first
    # element is delayed with a sleeping Python function. Expected determinism is the local setting if
    # given, else the global option. Uses the determinism checker to assert order is deterministic or
    # not accordingly.
    expect_determinism = local_determinism or (local_determinism is None and
                                               global_determinism)
    elements = list(range(1000))

    def dataset_fn(delay_ms):
      # [FUNCTION DOC]
      # HELPER: creates the dataset for the determinism check.
      # Input: delay in milliseconds applied to element zero. Builds a sliced dataset, applies the
      # parallel map with the local determinism flag, sets the global deterministic option and returns it.

      def sleep(x):
        # [FUNCTION DOC]
        # Python function used as an op: sleeps for the captured delay and returns the input.
        time.sleep(delay_ms / 1000)
        return x

      def map_function(x):
        # [FUNCTION DOC]
        # Mapped function: for element zero calls the sleeping function as an op, otherwise returns the
        # input as is.
        if math_ops.equal(x, 0):
          return script_ops.py_func(sleep, [x], x.dtype)
        else:
          return x

      dataset = dataset_ops.Dataset.from_tensor_slices(elements)
      dataset = apply_map(
          dataset,
          map_function,
          num_parallel_calls=2,
          deterministic=local_determinism)
      opts = options_lib.Options()
      opts.deterministic = global_determinism
      dataset = dataset.with_options(opts)
      return dataset

    self.checkDeterminism(
        dataset_fn, expect_determinism, expected_elements=elements)

  @combinations.generate(_test_combinations())
  def testNoneComponent(self, apply_map):
    # [FUNCTION DOC]
    # Scenario: a dataset element contains a number and a null component. The map function returns half
    # the number if the second argument is null. Asserts the output is 21.
    dataset = dataset_ops.Dataset.from_tensors((42, None))

    def map_function(x, y):
      # [FUNCTION DOC]
      # Mapped function: takes x and y; if y is null returns x halved, otherwise x.
      if y is None:
        return x / 2
      return x

    dataset = apply_map(dataset, map_function)
    self.assertDatasetProduces(dataset, expected_output=[21])

  @combinations.generate(test_base.eager_only_combinations())
  def testCheckpointLargeBuffer(self):
    # [FUNCTION DOC]
    # Eager mode only; skipped under memory sanitizers. Scenario: parallel map buffers several very
    # large (512 MB) tensors beyond the protocol buffer size limit. Fetches one element, waits, saves a
    # checkpoint, deletes the iterator and restores or initializes. Passes if no exception occurs.
    if (pywrap_sanitizers.is_asan_enabled() or
        pywrap_sanitizers.is_tsan_enabled() or
        pywrap_sanitizers.is_msan_enabled()):
      self.skipTest("Skip to avoid OOM when using sanitizers.")
    dataset = dataset_ops.Dataset.range(10).batch(2)
    dataset = dataset.map(
        # Create tensors of size 512M.
        lambda seed: stateless_random_ops.stateless_random_uniform(
            (128, 1024, 1024), seed, dtype=dtypes.float32
        )
    )
    # Set parallelism to 5 to exceed the 2GB protobuf limit
    dataset = dataset.map(lambda x: x * 2, num_parallel_calls=5)
    iterator = iter(dataset)
    next(iterator)  # Request an element to fill the parallel map buffer
    time.sleep(1)  # Give buffers some time to fill
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=1)
    manager.save()
    del dataset
    del iterator
    manager.restore_or_initialize()

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(
                             use_unbounded_threadpool=[True, False])))
  def testAutotuneUseUnboundedThreadpool(self, use_unbounded_threadpool):
    # [FUNCTION DOC]
    # Scenario: map with autotuned parallelism, optional unbounded thread pool, deterministic order and a
    # name. Asserts outputs equal doubled inputs for a range of 100.
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.map(
        lambda x: x * 2,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        use_unbounded_threadpool=use_unbounded_threadpool,
        deterministic=True,
        name="map")
    self.assertDatasetProduces(dataset, [x * 2 for x in range(100)])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 1])))
  def testName(self, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map with a name, with and without parallel calls. Asserts the output of doubling 21 is 42.
    dataset = dataset_ops.Dataset.from_tensors(21).map(
        lambda x: x * 2, num_parallel_calls=num_parallel_calls, name="map")
    self.assertDatasetProduces(dataset, [42])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 1])))
  def testStatusMessage(self, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map function divides by zero, default optimizations disabled. Asserts reading raises an
    # invalid-argument error whose message reports an error in the user-defined function for the
    # transformation with the root iterator path.
    dataset = dataset_ops.Dataset.from_tensors(21).map(
        lambda x: x // 0, num_parallel_calls=num_parallel_calls, name="map")
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r".*Error in user-defined function passed to .* transformation with "
        r"iterator: Iterator::Root::.*"):
      self.evaluate(get_next())


class MapCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                        parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for map: saving and restoring iterator state for plain and parallel map, and
  # errors when saving with stateful functions or captured variables. Uses the checkpoint test base.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              num_parallel_calls=[None, 2], symbolic_checkpoint=[False, True])))
  def testCore(self, verify_fn, num_parallel_calls, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Scenario: parallel (or not) map squares sliced components, repeated twice, with the symbolic
    # checkpoint option on or off. Delegates to the verification function with the dataset builder and
    # the total number of expected outputs. Inputs: the verification function, parallelism and
    # symbolic-checkpoint flag.

    tensor_slice_len = 7
    num_epochs = 2
    multiplier = 37.0

    def _build_ds():
      # [FUNCTION DOC]
      # HELPER: builds the mapped, repeated dataset with the symbolic checkpoint option applied.

      components = (np.arange(tensor_slice_len), np.array([[1, 2, 3]]) *
                    np.arange(tensor_slice_len)[:, np.newaxis],
                    np.array(multiplier) * np.arange(tensor_slice_len))

      def _map_fn(x, y, z):
        # [FUNCTION DOC]
        # Mapped function: returns the element-wise squares of three tensors.
        return math_ops.square(x), math_ops.square(y), math_ops.square(z)

      dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
          _map_fn, num_parallel_calls=num_parallel_calls).repeat(num_epochs)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(self, _build_ds, tensor_slice_len * num_epochs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testSaveStatefulFunction(self, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map uses a random (stateful) function. Asserts saving a checkpoint fails with a
    # failed-precondition error at the 15th step.

    def _build_ds():
      # [FUNCTION DOC]
      # HELPER: builds a range dataset mapped with the random function.

      def _map_fn(x):
        # [FUNCTION DOC]
        # Mapped function: returns a random integer below ten times the input cast to integer.
        return random_ops.random_uniform(
            (), 0, 10, dtype=dtypes.int32) * math_ops.cast(x, dtypes.int32)

      return dataset_ops.Dataset.range(100).map(
          _map_fn, num_parallel_calls=num_parallel_calls)

    self.verify_error_on_save(_build_ds, 15, errors.FailedPreconditionError)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testCaptureVariableInMapFn(self, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map captures and increments a variable. Asserts saving a checkpoint fails with a
    # failed-precondition error.

    def _build_ds():
      # [FUNCTION DOC]
      # HELPER: creates a resource variable and returns a repeated dataset mapped by incrementing it.
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      return (dataset_ops.Dataset.from_tensors(0).repeat(10).map(
          lambda _: counter_var.assign_add(1),
          num_parallel_calls=num_parallel_calls))

    self.verify_error_on_save(_build_ds, 15, errors.FailedPreconditionError)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testCaptureConstantInMapFn(self, verify_fn, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map adds a captured constant. Verifies, through the checkpoint verification function,
    # that save and restore works for ten outputs.
    num_outputs = 10

    def _build_ds():
      # [FUNCTION DOC]
      # HELPER: returns a repeated dataset mapped by adding a captured constant.
      constant_var = constant_op.constant(5)
      return (dataset_ops.Dataset.from_tensors(0).repeat(10).map(
          lambda x: x + constant_var, num_parallel_calls=num_parallel_calls))

    verify_fn(self, _build_ds, num_outputs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testCaptureDefunInMapFn(self, verify_fn, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map uses a function declared through the legacy function decorator. Verifies save and
    # restore for ten outputs.
    num_outputs = 10

    def _build_ds():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset mapped with the decorated function.

      @function.Defun(dtypes.int64)
      def defun_fn(x):
        # [FUNCTION DOC]
        # Decorated function: returns 1000 plus the input cast to 32-bit integer.
        return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

      return dataset_ops.Dataset.range(num_outputs).map(
          defun_fn, num_parallel_calls=num_parallel_calls)

    verify_fn(self, _build_ds, num_outputs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testBuildDefunInMapFn(self, verify_fn, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map uses a decorated function that itself calls another decorated function. Verifies save
    # and restore for ten outputs.
    num_outputs = 10

    def _build_ds():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset mapped with the outer decorated function.

      @function.Defun(dtypes.int64)
      def defun_fn(x):
        # [FUNCTION DOC]
        # Decorated function: casts the input and returns 11000 plus the inner function's result.

        @function.Defun(dtypes.int32)
        def defun_fn_deep(x):
          # [FUNCTION DOC]
          # Inner decorated function: returns 1000 plus the input cast to 32-bit integer.
          return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

        return constant_op.constant(11000) + defun_fn_deep(
            math_ops.cast(x, dtypes.int32))

      return dataset_ops.Dataset.range(num_outputs).map(
          defun_fn, num_parallel_calls=num_parallel_calls)

    verify_fn(self, _build_ds, num_outputs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testSparse(self, verify_fn, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map producing sparse tensors from a range dataset. Verifies checkpoint save and restore
    # for ten outputs.

    def _sparse(i):
      # [FUNCTION DOC]
      # HELPER: builds a one-element sparse tensor value from an integer.
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def _build_ds(num_outputs):
      # [FUNCTION DOC]
      # HELPER: input num_outputs; returns a range dataset mapped with the sparse builder.
      return dataset_ops.Dataset.range(num_outputs).map(
          _sparse, num_parallel_calls=num_parallel_calls)

    num_outputs = 10
    verify_fn(self, lambda: _build_ds(num_outputs), num_outputs=num_outputs)


class MapRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests random access (element at an index) on datasets containing map, version 2 only.

  @combinations.generate(
      combinations.times(test_base.v2_only_combinations(),
                         combinations.combine(index=[-1, 4, 5])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # Scenario: dataset of four elements mapped by doubling. Asserts accessing indices -1, 4 or 5 raises
    # an out-of-range error.
    dataset = dataset_ops.Dataset.from_tensor_slices([-1, 0, 1,
                                                      2]).map(lambda x: x * 2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.v2_only_combinations(),
                         combinations.combine(index=[-1, 0])))
  def testEmptyDataset(self, index):
    # [FUNCTION DOC]
    # Scenario: an empty dataset mapped by halving. Asserts accessing index -1 or 0 raises out-of-range.
    dataset = dataset_ops.Dataset.from_tensor_slices([]).map(lambda x: x // 2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(combinations.times(test_base.v2_only_combinations()))
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: six elements mapped by tripling. Asserts accessing indices 0 to 4 returns triple the index.
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3, 4,
                                                      5]).map(lambda x: x * 3)
    for i in range(5):
      self.assertEqual(self.evaluate(random_access.at(dataset, index=i)), i * 3)

  @combinations.generate(
      combinations.times(
          test_base.v2_only_combinations(),
          combinations.combine(
              elements=[0, 10, 20, 40], num_parallel_calls=[None, 2])))
  def testMultipleCombinations(self, elements, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: ranges of several lengths mapped by halving, with and without parallelism. Asserts every
    # index returns the expected halved value.
    dataset = dataset_ops.Dataset.range(elements).map(
        lambda x: x // 2, num_parallel_calls=num_parallel_calls)
    for i in range(elements):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), i // 2)

  @combinations.generate(
      combinations.times(
          test_base.v2_only_combinations(),
          combinations.combine(
              elements=[0, 10, 20, 40], num_parallel_calls=[None, 2])))
  def testMapFnInFunction(self, elements, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: map with a compiled function decorated for graph execution that squares. Asserts random
    # access at each index equals the square.

    @def_function.function
    def _map_fn(x):
      # [FUNCTION DOC]
      # Compiled function: returns the element squared.
      return math_ops.square(x)

    dataset = dataset_ops.Dataset.range(elements).map(
        _map_fn, num_parallel_calls=num_parallel_calls)
    for i in range(elements):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)),
          self.evaluate(math_ops.square(i)))


class MapGlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that map works with global shuffle (shuffling of a dataset by random index mapping), for
  # different parallelism and determinism settings.

  @combinations.generate(
      combinations.times(
          test_base.v2_only_combinations(),
          combinations.combine(
              dataset_range=[100],
              num_parallel_calls=[None, 2, dataset_ops.AUTOTUNE],
              deterministic=[True, False])))
  def testMapV2(  # V2 API preserves cardinality by default.
      self, dataset_range: int, num_parallel_calls: int, deterministic: bool):
    # [FUNCTION DOC]
    # Version 2 only. Scenario: map doubling, prefetch, global shuffle, with default optimizations and
    # warm start disabled so both map variants are tested. Asserts the output has the expected elements
    # (as a multiset) but not in the original order. Inputs: range size, parallel calls, determinism.
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.map(
        lambda x: x * 2,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = global_shuffle_op._global_shuffle(dataset)
    # Disables optimizations (e.g.: `map_parallelization`), to make sure we test
    # both `Map` and `ParallelMap`.
    # TODO(b/325112575): Support warm-start. With warm-start, prefetching uses
    # the unintended IteratorContext here:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/data/prefetch_dataset_op.cc#L197-L199.
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_warm_start = False
    dataset = dataset.with_options(options)

    expected = list(range(0, dataset_range * 2, 2))
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[100],
              num_parallel_calls=[None, 2, dataset_ops.AUTOTUNE],
              deterministic=[True, False])))
  def testMapV1AndV2(
      self, dataset_range: int, num_parallel_calls: int, deterministic: bool):
    # [FUNCTION DOC]
    # Scenario: like the previous test but a cardinality assertion is applied after map, to cover
    # version 1 as well. Asserts the same elements in shuffled order.
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset_cardinality = dataset.cardinality()
    dataset = dataset.map(
        lambda x: x * 2,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic)
    dataset = dataset.apply(cardinality.assert_cardinality(dataset_cardinality))
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = global_shuffle_op._global_shuffle(dataset)

    expected = list(range(0, dataset_range * 2, 2))
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)


class MapGlobalShuffleCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                     parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for map combined with global shuffle, using the checkpoint test base.

  @combinations.generate(
      combinations.times(
          test_base.v2_only_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              num_parallel_calls=[None, 2, dataset_ops.AUTOTUNE],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testMapV2(  # V2 API preserves cardinality by default.
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      num_parallel_calls: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Version 2 only. Scenario: deterministic doubling map, prefetch and seeded global shuffle with
    # optional reshuffling and symbolic checkpoint. The verification function checks save and restore
    # over the whole range, treating items as unordered when reshuffling each iteration.

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the dataset described above, applying the options (optimizations and warm start off,
      # symbolic checkpoint as given). Returns the dataset.
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset = dataset.map(
          lambda x: x * 2,
          num_parallel_calls=num_parallel_calls,
          deterministic=True)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_warm_start = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=dataset_range,
        assert_items_equal=reshuffle_each_iteration)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              num_parallel_calls=[None, 2, dataset_ops.AUTOTUNE],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testMapV1AndV2(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      num_parallel_calls: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Like the previous test but with a cardinality assertion after the map, covering version 1 as well;
    # warm start is left at its default.

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the dataset with the cardinality assertion, prefetch, seeded global shuffle and
      # options (default optimizations off, symbolic checkpoint as given). Returns the dataset.
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset_cardinality = dataset.cardinality()
      dataset = dataset.map(
          lambda x: x * 2,
          num_parallel_calls=num_parallel_calls,
          deterministic=True)
      dataset = dataset.apply(
          cardinality.assert_cardinality(dataset_cardinality))
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=dataset_range,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
