# [MODULE DOC]
# HELPER: shared test utilities for the tf.data kernel tests.
# Provides the default test-combination generators (API version 1 and/or 2, eager and/or graph
# mode) and the DatasetTestBase class, a test case base with helpers for iterating datasets in both
# eager and graph mode, asserting dataset outputs, errors and equality, random-access checks,
# lookup table initializers, graph round trips, determinism checks and multi-device setup.
# Importing has no side effects beyond imports; DatasetTestBase.setUp turns on tf.data test mode.
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
"""Test utilities for tf.data functionality."""
import os
import random
import re

from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import test_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


def default_test_combinations():
  # [FUNCTION DOC]
  # HELPER: shared by tf.data tests to parameterize over API versions and execution modes.
  # Takes no inputs. Returns the test combinations covering API versions 1 and 2 crossed with
  # eager and graph modes.
  """Returns the default test combinations for tf.data tests."""
  return combinations.combine(tf_api_version=[1, 2], mode=["eager", "graph"])


def eager_only_combinations():
  # [FUNCTION DOC]
  # HELPER: shared test parameterization. Returns combinations for API versions 1 and 2 in eager
  # mode only.
  """Returns the default test combinations for eager mode only tf.data tests."""
  return combinations.combine(tf_api_version=[1, 2], mode="eager")


def graph_only_combinations():
  # [FUNCTION DOC]
  # HELPER: shared test parameterization. Returns combinations for API versions 1 and 2 in graph
  # mode only.
  """Returns the default test combinations for graph mode only tf.data tests."""
  return combinations.combine(tf_api_version=[1, 2], mode="graph")


def v1_only_combinations():
  # [FUNCTION DOC]
  # HELPER: shared test parameterization. Returns combinations for API version 1 only, in both
  # eager and graph modes.
  """Returns the default test combinations for v1 only tf.data tests."""
  return combinations.combine(tf_api_version=1, mode=["eager", "graph"])


def v2_only_combinations():
  # [FUNCTION DOC]
  # HELPER: shared test parameterization. Returns combinations for API version 2 only, in both
  # eager and graph modes.
  """Returns the default test combinations for v2 only tf.data tests."""
  return combinations.combine(tf_api_version=2, mode=["eager", "graph"])


def v2_eager_only_combinations():
  # [FUNCTION DOC]
  # HELPER: shared test parameterization. Returns combinations for API version 2 in eager mode
  # only.
  """Returns the default test combinations for v2 eager only tf.data tests."""
  return combinations.combine(tf_api_version=2, mode="eager")


class DatasetTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: base class for tf.data tests. Extends the framework test case with utilities to
  # iterate datasets in eager or graph mode (getNext, getDatasetOutput, getIteratorOutput), assert
  # on dataset contents, errors and equality, test random access, build lookup table initializers,
  # round-trip a dataset through a graph, check determinism and configure logical devices. Holds
  # no state of its own; setUp enables tf.data test mode.
  """Base class for dataset tests."""

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent set-up, then switches on the tf.data test mode (global state in the data ops
    # test-mode module). Takes no inputs and returns nothing.
    super().setUp()
    test_mode.toggle_test_mode(True)

  def assert_op_cancelled(self, op):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Evaluates op (a tensor or operation) and asserts that doing so
    # raises a cancelled error. Returns nothing; fails the test otherwise.
    with self.assertRaises(errors.CancelledError):
      self.evaluate(op)

  def assertValuesEqual(self, expected, actual):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Compares two evaluated values structurally. Mappings are compared
    # by key set (ignoring order) and then recursively per key; sparse tensors are compared by
    # indices, values and dense shape; anything else is compared as array-like values for exact
    # equality. Returns nothing; fails the test on mismatch.
    """Asserts that two values are equal."""
    if isinstance(expected, dict):
      self.assertItemsEqual(list(expected.keys()), list(actual.keys()))
      for k in expected.keys():
        self.assertValuesEqual(expected[k], actual[k])
    elif sparse_tensor.is_sparse(expected):
      self.assertAllEqual(expected.indices, actual.indices)
      self.assertAllEqual(expected.values, actual.values)
      self.assertAllEqual(expected.dense_shape, actual.dense_shape)
    else:
      self.assertAllEqual(expected, actual)

  def getNext(self, dataset, requires_initialization=False, shared_name=None):
    # [FUNCTION DOC]
    # HELPER: central iteration helper for tf.data tests. Inputs: dataset to iterate;
    # requires_initialization (default false) selects an initializable iterator in graph mode
    # (needed for stateful datasets); shared_name (optional) names the iterator for sharing across
    # sessions. Returns a callable producing the next element; tensor-array outputs are stacked.
    # In eager mode or inside a function, an anonymous Python iterator is used. In graph mode an
    # initializable iterator is created and its initializer is run immediately (side effect: runs
    # the initializer in the session), otherwise a one-shot iterator is used.
    """Returns a callable that returns the next element of the dataset.

    Example use:
    ```python
    # In both graph and eager modes
    dataset = ...
    get_next = self.getNext(dataset)
    result = self.evaluate(get_next())
    ```

    Args:
      dataset: A dataset whose elements will be returned.
      requires_initialization: Indicates that when the test is executed in graph
        mode, it should use an initializable iterator to iterate through the
        dataset (e.g. when it contains stateful nodes). Defaults to False.
      shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the same
        devices (e.g. when using a remote server).
    Returns:
      A callable that returns the next element of `dataset`. Any `TensorArray`
      objects `dataset` outputs are stacked.
    """
    def ta_wrapper(gn):
      # [FUNCTION DOC]
      # HELPER: wraps a get-next callable gn so that a tensor-array result is stacked into a
      # tensor. Returns the wrapping callable.
      def _wrapper():
        # [FUNCTION DOC]
        # HELPER: calls the wrapped callable and returns its result, stacking it if it is a tensor
        # array; any other result is returned unchanged.
        r = gn()
        if isinstance(r, tensor_array_ops.TensorArray):
          return r.stack()
        else:
          return r
      return _wrapper

    # Create an anonymous iterator if we are in eager-mode or are graph inside
    # of a tf.function.
    if context.executing_eagerly() or ops.inside_function():
      iterator = iter(dataset)
      return ta_wrapper(iterator._next_internal)  # pylint: disable=protected-access
    else:
      if requires_initialization:
        iterator = dataset_ops.make_initializable_iterator(dataset, shared_name)
        self.evaluate(iterator.initializer)
      else:
        iterator = dataset_ops.make_one_shot_iterator(dataset)
      get_next = iterator.get_next()
      return ta_wrapper(lambda: get_next)

  def _compareOutputToExpected(self, result_values, expected_values,
                               assert_items_equal):
    # [FUNCTION DOC]
    # HELPER: comparison step used by assertDatasetProduces. If assert_items_equal is true, only
    # checks both lists hold the same items regardless of order. Otherwise, for each position in
    # result_values, asserts the nested structures match and each flattened component is equal
    # (via assertValuesEqual). Only positions up to the length of result_values are checked.
    # Returns nothing; fails the test on mismatch.
    if assert_items_equal:
      # TODO(shivaniagrawal): add support for nested elements containing sparse
      # tensors when needed.
      self.assertItemsEqual(result_values, expected_values)
      return
    for i in range(len(result_values)):
      nest.assert_same_structure(result_values[i], expected_values[i])
      for result_value, expected_value in zip(
          nest.flatten(result_values[i]), nest.flatten(expected_values[i])):
        self.assertValuesEqual(expected_value, result_value)

  def getDatasetOutput(self, dataset, requires_initialization=False):
    # [FUNCTION DOC]
    # HELPER: iterates dataset to the end and returns the list of produced elements. Inputs:
    # dataset and requires_initialization (default false; use an initializable iterator in graph
    # mode). Built from getNext and getIteratorOutput.
    get_next = self.getNext(
        dataset, requires_initialization=requires_initialization)
    return self.getIteratorOutput(get_next)

  def getIteratorOutput(self, get_next):
    # [FUNCTION DOC]
    # HELPER: repeatedly evaluates the callable get_next and collects each result into a list
    # until an out-of-range error signals the end of input, which is swallowed. Returns the list.
    # Other errors propagate. Loops forever on an infinite dataset.
    """Evaluates `get_next` until end of input, returning the results."""
    results = []
    while True:
      try:
        results.append(self.evaluate(get_next()))
      except errors.OutOfRangeError:
        break
    return results

  def assertDatasetProduces(self,
                            dataset,
                            expected_output=None,
                            expected_shapes=None,
                            expected_error=None,
                            requires_initialization=False,
                            num_test_iterations=1,
                            assert_items_equal=False,
                            expected_error_iter=1):
    # [FUNCTION DOC]
    # HELPER: the main assertion helper for dataset tests. Inputs: dataset; expected_output (list
    # of expected elements); expected_shapes (optional, compared with the legacy output shapes);
    # expected_error (optional pair of exception type and a predicate or regex);
    # requires_initialization; num_test_iterations (default 1, how many full passes to make);
    # assert_items_equal (ignore order); expected_error_iter (how many elements to fetch before
    # the error is expected). At least one of expected_output or expected_error must be given,
    # and with an error the output must be absent. In error mode it asserts the error is raised
    # while fetching and returns. Otherwise, for each pass it asserts exactly the expected number
    # of elements come out (raising an assertion error with the partial output if the dataset ends
    # early), compares them to the expectation, and asserts two further reads both raise
    # out-of-range. Returns nothing.
    # Note: a pass reads only len(expected_output) elements then expects the end.
    """Asserts that a dataset produces the expected output / error.

    Args:
      dataset: A dataset to check for the expected output / error.
      expected_output: A list of elements that the dataset is expected to
        produce.
      expected_shapes: A list of TensorShapes which is expected to match
        output_shapes of dataset.
      expected_error: A tuple `(type, predicate)` identifying the expected error
        `dataset` should raise. The `type` should match the expected exception
        type, while `predicate` should either be 1) a unary function that inputs
        the raised exception and returns a boolean indicator of success or 2) a
        regular expression that is expected to match the error message
        partially.
      requires_initialization: Indicates that when the test is executed in graph
        mode, it should use an initializable iterator to iterate through the
        dataset (e.g. when it contains stateful nodes). Defaults to False.
      num_test_iterations: Number of times `dataset` will be iterated. Defaults
        to 1.
      assert_items_equal: Tests expected_output has (only) the same elements
        regardless of order.
      expected_error_iter: How many times to iterate before expecting an error,
        if an error is expected.
    """
    self.assertTrue(
        expected_error is not None or expected_output is not None,
        "Exactly one of expected_output or expected error should be provided.")
    if expected_error:
      self.assertTrue(
          expected_output is None,
          "Exactly one of expected_output or expected error should be provided."
      )
      with self.assertRaisesWithPredicateMatch(expected_error[0],
                                               expected_error[1]):
        get_next = self.getNext(
            dataset, requires_initialization=requires_initialization)
        for _ in range(expected_error_iter):
          self.evaluate(get_next())
      return
    if expected_shapes:
      self.assertEqual(expected_shapes,
                       dataset_ops.get_legacy_output_shapes(dataset))
    self.assertGreater(num_test_iterations, 0)
    for _ in range(num_test_iterations):
      get_next = self.getNext(
          dataset, requires_initialization=requires_initialization)
      result = []
      for _ in range(len(expected_output)):
        try:
          result.append(self.evaluate(get_next()))
        except errors.OutOfRangeError:
          raise AssertionError(
              "Dataset ended early, producing %d elements out of %d. "
              "Dataset output: %s" %
              (len(result), len(expected_output), str(result)))
      self._compareOutputToExpected(result, expected_output, assert_items_equal)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  def assertDatasetsEqual(self, dataset1, dataset2):
    # [FUNCTION DOC]
    # HELPER: asserts two datasets are equal in eager or graph mode. First asserts their element
    # structures are compatible. Then iterates both in lockstep: strings and sparse or ragged
    # components are compared exactly, other components with numeric closeness. When dataset1
    # ends it asserts dataset2 also raises out-of-range. Returns nothing.
    # Note: if dataset2 ends first, the out-of-range error from it propagates uncaught.
    """Checks that datasets are equal. Supports both graph and eager mode."""
    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(dataset1),
            dataset_ops.get_structure(dataset2)))

    flattened_types = nest.flatten(
        dataset_ops.get_legacy_output_types(dataset1))

    next1 = self.getNext(dataset1)
    next2 = self.getNext(dataset2)

    while True:
      try:
        op1 = self.evaluate(next1())
      except errors.OutOfRangeError:
        with self.assertRaises(errors.OutOfRangeError):
          self.evaluate(next2())
        break
      op2 = self.evaluate(next2())

      op1 = nest.flatten(op1)
      op2 = nest.flatten(op2)
      assert len(op1) == len(op2)
      for i in range(len(op1)):
        if sparse_tensor.is_sparse(op1[i]) or ragged_tensor.is_ragged(op1[i]):
          self.assertValuesEqual(op1[i], op2[i])
        elif flattened_types[i] == dtypes.string:
          self.assertAllEqual(op1[i], op2[i])
        else:
          self.assertAllClose(op1[i], op2[i])

  def assertDatasetsRaiseSameError(self,
                                   dataset1,
                                   dataset2,
                                   exception_class,
                                   replacements=None):
    # [FUNCTION DOC]
    # HELPER: asserts two datasets raise the same error on their first element fetch. Inputs: two
    # datasets, exception_class (expected error type) and replacements (optional list of triples
    # of old text, new text and maximum replacement count applied to the first error message
    # before comparing). The first dataset must raise exception_class, otherwise a value error is
    # raised; the second must raise the same type with a message matching the (escaped) adjusted
    # first message. Returns nothing.
    """Checks that datasets raise the same error on the first get_next call."""
    if replacements is None:
      replacements = []
    next1 = self.getNext(dataset1)
    next2 = self.getNext(dataset2)
    try:
      self.evaluate(next1())
      raise ValueError(
          "Expected dataset to raise an error of type %s, but it did not." %
          repr(exception_class))
    except exception_class as e:
      expected_message = e.message
      for old, new, count in replacements:
        expected_message = expected_message.replace(old, new, count)
      # Check that the first segment of the error messages are the same.
      with self.assertRaisesRegex(exception_class, re.escape(expected_message)):
        self.evaluate(next2())

  def structuredDataset(self, dataset_structure, shape=None,
                        dtype=dtypes.int64):
    # [FUNCTION DOC]
    # HELPER: builds a test dataset. dataset_structure is nothing (a single zero tensor) or a
    # sequence of nested sub-structures, which are zipped as a tuple recursively. shape (default
    # scalar) and dtype (default 64-bit integer) describe the zero tensors. Returns a dataset.
    """Returns a singleton dataset with the given structure."""
    if shape is None:
      shape = []
    if dataset_structure is None:
      return dataset_ops.Dataset.from_tensors(
          array_ops.zeros(shape, dtype=dtype))
    else:
      return dataset_ops.Dataset.zip(
          tuple([
              self.structuredDataset(substructure, shape, dtype)
              for substructure in dataset_structure
          ]))

  def verifyRandomAccess(self, dataset, expected):
    # [FUNCTION DOC]
    # HELPER: verifies random access on a finite dataset. Runs the infinite-cardinality check
    # against expected (list of expected elements) and then asserts access at index equal to the
    # expected length raises an out-of-range error. Returns nothing.
    self.verifyRandomAccessInfiniteCardinality(dataset, expected)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=len(expected)))

  def verifyRandomAccessInfiniteCardinality(self, dataset, expected):
    # [FUNCTION DOC]
    # HELPER: checks random access element by element. Reads every index in the expected list
    # twice in shuffled order (uses the global random module, so order varies), then reads each
    # distinct index again, asserting each value equals the expected element. Returns nothing.
    # KNOWN ISSUE: the comment says the second pass accesses elements in order, but the indices
    # are converted to a set, which gives no guaranteed ordering, so the pass is not truly ordered.
    """Tests randomly accessing elements of a dataset."""
    # Tests accessing the elements in a shuffled order with repeats.
    len_expected = len(expected)
    indices = list(range(len_expected)) * 2
    random.shuffle(indices)
    for i in indices:
      self.assertAllEqual(expected[i],
                          self.evaluate(random_access.at(dataset, i)))

    # Tests accessing the elements in order.
    indices = set(sorted(indices))
    for i in indices:
      self.assertAllEqual(expected[i],
                          self.evaluate(random_access.at(dataset, i)))

  def textFileInitializer(self, vals):
    # [FUNCTION DOC]
    # HELPER: lookup-table initializer factory. Writes the values, one per line, to a file named
    # for the initializer inside the test temp directory (side effect: creates or overwrites that
    # file), and returns a text-file initializer mapping line number (64-bit integer) to the whole
    # line (as 64-bit integer).
    file = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with open(file, "w") as f:
      f.write("\n".join(str(v) for v in vals) + "\n")
    return lookup_ops.TextFileInitializer(file, dtypes.int64,
                                          lookup_ops.TextFileIndex.LINE_NUMBER,
                                          dtypes.int64,
                                          lookup_ops.TextFileIndex.WHOLE_LINE)

  def keyValueTensorInitializer(self, vals):
    # [FUNCTION DOC]
    # HELPER: lookup-table initializer factory. Uses keys zero to len(vals)-1 as 64-bit integers
    # and vals (a list) as constant tensor values. Returns a key-value tensor initializer.
    keys_tensor = constant_op.constant(
        list(range(len(vals))), dtype=dtypes.int64)
    vals_tensor = constant_op.constant(vals)
    return lookup_ops.KeyValueTensorInitializer(keys_tensor, vals_tensor)

  def datasetInitializer(self, vals):
    # [FUNCTION DOC]
    # HELPER: lookup-table initializer factory. Zips a range dataset of keys with a dataset of
    # vals (a list) and returns a dataset-based lookup initializer.
    keys = dataset_ops.Dataset.range(len(vals))
    values = dataset_ops.Dataset.from_tensor_slices(vals)
    ds = dataset_ops.Dataset.zip((keys, values))
    return data_lookup_ops.DatasetInitializer(ds)

  def lookupTableInitializer(self, init_source, vals):
    # [FUNCTION DOC]
    # HELPER: dispatcher for lookup-table initializers. init_source selects the kind and vals
    # gives the values (keys are their positions). Returns the initializer from the matching
    # factory; raises a value error for an unknown source.
    # KNOWN ISSUE: the docstring lists the source name "keyvalue", but the code only recognizes
    # "keyvaluetensor" (plus "textfile" and "dataset"), so "keyvalue" raises a value error.
    """Returns a lookup table initializer for the given source and values.

    Args:
      init_source: One of ["textfile", "keyvalue", "dataset"], indicating what
        type of initializer to use.
      vals: The initializer values. The keys will be `range(len(vals))`.
    """
    if init_source == "textfile":
      return self.textFileInitializer(vals)
    elif init_source == "keyvaluetensor":
      return self.keyValueTensorInitializer(vals)
    elif init_source == "dataset":
      return self.datasetInitializer(vals)
    else:
      raise ValueError("Unrecognized init_source: " + init_source)

  def graphRoundTrip(self, dataset, allow_stateful=False):
    # [FUNCTION DOC]
    # HELPER: serializes dataset to a graph definition (optionally permitting stateful ops when
    # allow_stateful is true; default false) and rebuilds a dataset from it using the original
    # element spec. Returns the rebuilt dataset.
    """Converts a dataset to a graph and back."""
    graph = gen_dataset_ops.dataset_to_graph(
        dataset._variant_tensor, allow_stateful=allow_stateful)  # pylint: disable=protected-access
    return dataset_ops.from_variant(
        gen_experimental_dataset_ops.dataset_from_graph(graph),
        dataset.element_spec)

  def structuredElement(self, element_structure, shape=None,
                        dtype=dtypes.int64):
    # [FUNCTION DOC]
    # HELPER: builds a test element. element_structure is nothing (one zero tensor) or a sequence
    # of nested sub-structures turned into a nested tuple. shape defaults to scalar and dtype to
    # 64-bit integer. Returns the tensor or tuple.
    """Returns an element with the given structure."""
    if shape is None:
      shape = []
    if element_structure is None:
      return array_ops.zeros(shape, dtype=dtype)
    else:
      return tuple([
          self.structuredElement(substructure, shape, dtype)
          for substructure in element_structure
      ])

  def checkDeterminism(self, dataset_fn, expect_determinism, expected_elements):
    # [FUNCTION DOC]
    # HELPER: tests deterministic versus non-deterministic production order. dataset_fn is a
    # callable taking a delay in milliseconds (delays the first element) and returning a dataset;
    # expect_determinism says which behavior is expected; expected_elements is the deterministic
    # order. If determinism is expected, runs with a 100 ms delay and asserts the output equals
    # the expected list exactly. Otherwise tries increasing delays up to 100 seconds, asserting
    # the same elements appear (any order), and returns as soon as one run differs in order; fails
    # the test if every run is in the deterministic order. Can be slow on failure.
    """Tests whether a dataset produces its elements deterministically.

    `dataset_fn` takes a delay_ms argument, which tells it how long to delay
    production of the first dataset element. This gives us a way to trigger
    out-of-order production of dataset elements.

    Args:
      dataset_fn: A function taking a delay_ms argument.
      expect_determinism: Whether to expect deterministic ordering.
      expected_elements: The elements expected to be produced by the dataset,
        assuming the dataset produces elements in deterministic order.
    """
    if expect_determinism:
      dataset = dataset_fn(100)
      actual = self.getDatasetOutput(dataset)
      self.assertAllEqual(expected_elements, actual)
      return

    # We consider the test a success if it succeeds under any delay_ms. The
    # delay_ms needed to observe non-deterministic ordering varies across
    # test machines. Usually 10 or 100 milliseconds is enough, but on slow
    # machines it could take longer.
    for delay_ms in [10, 100, 1000, 20000, 100000]:
      dataset = dataset_fn(delay_ms)
      actual = self.getDatasetOutput(dataset)
      self.assertCountEqual(expected_elements, actual)
      for i in range(len(actual)):
        if actual[i] != expected_elements[i]:
          return
    self.fail("Failed to observe nondeterministic ordering")

  def configureDevicesForMultiDeviceTest(self, num_devices):
    # [FUNCTION DOC]
    # HELPER: for multi-device tests. Splits the first physical CPU into num_devices logical
    # devices (side effect: changes global device configuration, which must happen before devices
    # are initialized). Returns the list of device names: num_devices-1 CPU names, plus a final
    # name that is the first GPU if any physical GPU exists, otherwise the last CPU.
    """Configures number of logical devices for multi-device tests.

    It returns a list of device names. If invoked in GPU-enabled runtime, the
    last device name will be for a GPU device. Otherwise, all device names will
    be for a CPU device.

    Args:
      num_devices: The number of devices to configure.

    Returns:
      A list of device names to use for a multi-device test.
    """
    cpus = config.list_physical_devices("CPU")
    gpus = config.list_physical_devices("GPU")
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration() for _ in range(num_devices)
    ])
    devices = ["/device:CPU:" + str(i) for i in range(num_devices - 1)]
    if gpus:
      devices.append("/device:GPU:0")
    else:
      devices.append("/device:CPU:" + str(num_devices - 1))
    return devices
