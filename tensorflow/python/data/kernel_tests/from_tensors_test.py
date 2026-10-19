# [MODULE DOC]
# Tests for creating a single-element dataset from a structure of tensors (the from-tensors
# constructor). Covers tuples, nested datasets, tensor arrays, sparse, ragged and mixed
# components, named tuples, attrs classes, data classes with custom flatten support, nested
# structures and dictionaries, a two-device pipeline, random access, iterator checkpointing and
# global shuffling. Defines a small data class used as a test input. The optional attrs package is
# imported with a fallback to nothing. Runs the test main entry point when executed directly.
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
"""Tests for `tf.data.Dataset.from_tensors()."""
import collections
import dataclasses
from typing import Callable, Optional

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


@dataclasses.dataclass
class MaskedTensor:
  # [CLASS DOC]
  # HELPER: small data class used as a test input. Holds a boolean mask and an array value, and
  # implements the flatten/unflatten protocol so the dataset machinery treats the mask as static
  # metadata and the value as the only tensor component.
  mask: bool
  value: np.ndarray

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # Flatten protocol: returns a pair of the metadata (a one-item tuple holding the mask) and the
    # components (a one-item tuple holding the value).
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  def __tf_unflatten__(self, metadata, components):
    # [FUNCTION DOC]
    # Unflatten protocol: rebuilds a new masked tensor from the metadata (mask is its first item)
    # and the components (value is its first item). Does not use the current instance's fields.
    mask = metadata[0]
    value = components[0]
    return MaskedTensor(mask=mask, value=value)


class FromTensorsTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for the from-tensors dataset constructor with many kinds of element
  # structures. Uses the shared dataset test base class.

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensors(self):
    # [FUNCTION DOC]
    # Builds a dataset from a tuple of three arrays. Asserts the flattened legacy output shapes
    # equal the array shapes and that the dataset yields exactly that one tuple.
    """Test a dataset that represents a single tuple of tensors."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertEqual(
        [c.shape for c in components],
        nest.flatten(dataset_ops.get_legacy_output_shapes(dataset)))

    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsDataset(self):
    # [FUNCTION DOC]
    # Wraps a range-of-ten dataset as the single element of a dataset, flattens it with an
    # identity flat-map and asserts the output is the range of ten.
    """Test a dataset that represents a dataset."""
    dataset = dataset_ops.Dataset.from_tensors(dataset_ops.Dataset.range(10))
    dataset = dataset.flat_map(lambda x: x)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsTensorArray(self):
    # [FUNCTION DOC]
    # Uses a tensor array holding the values one and two as the element and asserts the dataset
    # produces the stacked list of those two values (iterator needs initialization).
    """Test a dataset that represents a TensorArray."""
    components = (
        tensor_array_ops.TensorArray(dtypes.float32, element_shape=(), size=2)
        .unstack([1.0, 2.0]))

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertDatasetProduces(
        dataset, expected_output=[[1.0, 2.0]], requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsSparse(self):
    # [FUNCTION DOC]
    # Uses a tuple of two sparse tensor values as the element. Asserts the legacy output shapes
    # equal the dense shapes and that the dataset yields exactly that tuple.
    """Test a dataset that represents a single tuple of tensors."""
    components = (sparse_tensor.SparseTensorValue(
        indices=np.array([[0]]),
        values=np.array([0]),
        dense_shape=np.array([1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1]]),
                      values=np.array([-1, 1]),
                      dense_shape=np.array([2, 2])))

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertEqual(
        [tensor_shape.TensorShape(c.dense_shape) for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsMixed(self):
    # [FUNCTION DOC]
    # Uses a tuple of three dense arrays and two sparse tensor values as the element. Asserts the
    # legacy output shapes (dense shape for sparse, shape otherwise) and that the dataset yields
    # exactly that tuple.
    """Test an dataset that represents a single tuple of tensors."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0]]),
                      values=np.array([0]),
                      dense_shape=np.array([1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1]]),
                      values=np.array([-1, 1]),
                      dense_shape=np.array([2, 2])))

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEqual([
        tensor_shape.TensorShape(c.dense_shape)
        if sparse_tensor.is_sparse(c) else c.shape for c in components
    ], [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])

    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsRagged(self):
    # [FUNCTION DOC]
    # Uses a tuple of two ragged tensor values as the element and asserts the dataset yields
    # exactly that tuple.
    components = (
        ragged_factory_ops.constant_value([[[0]], [[1]], [[2]]]),
        ragged_factory_ops.constant_value([[[3]], [[4]], [[5]]]),
    )

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsNamedTuple(self):
    # [FUNCTION DOC]
    # Uses a locally defined named tuple with two integer fields as the element and asserts the
    # dataset yields it unchanged.
    Foo = collections.namedtuple("Foo", ["x", "y"])
    element = Foo(x=1, y=2)
    dataset = dataset_ops.Dataset.from_tensors(element)
    self.assertDatasetProduces(dataset, expected_output=[element])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsAttrs(self):
    # [FUNCTION DOC]
    # Skips if the optional attrs package is unavailable. Otherwise defines a local attrs class
    # with two fields, uses an instance as the element and asserts the dataset yields it unchanged.
    if attr is None:
      self.skipTest("attr module is not available.")

    @attr.s
    class Foo:
      # [CLASS DOC]
      # HELPER: local attrs-decorated record with two attributes, x and y, used as test input.
      x = attr.ib()
      y = attr.ib()

    element = Foo(x=1, y=2)
    dataset = dataset_ops.Dataset.from_tensors(element)
    self.assertDatasetProduces(dataset, expected_output=[element])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsDataclass(self):
    # [FUNCTION DOC]
    # Uses a masked tensor data class instance as the element and asserts the dataset yields an
    # equal instance.
    mt = MaskedTensor(mask=True, value=np.array([1]))
    dataset = dataset_ops.Dataset.from_tensors(mt)
    self.assertDatasetProduces(dataset, expected_output=[mt])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsMixedRagged(self):
    # [FUNCTION DOC]
    # Uses a tuple mixing dense, sparse and ragged values as the element and asserts the dataset
    # yields exactly that tuple.
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0]]),
                      values=np.array([0]),
                      dense_shape=np.array([1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1]]),
                      values=np.array([-1, 1]),
                      dense_shape=np.array([2, 2])),
                  ragged_factory_ops.constant_value([[[0]], [[1]], [[2]]]))

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(
      combinations.combine(
          tf_api_version=[1],
          mode=["graph"],
          components=(np.array([1, 2, 3], dtype=np.int64),
                      (np.array([4., 5.]), np.array(
                          [6., 7.])), np.array([8, 9, 10], dtype=np.int64)),
          expected_shapes=[[[None, 3], [None, 3], [None, 2], [None, 2]]]) +
      combinations.combine(
          tf_api_version=[1],
          mode=["eager"],
          components=(np.array([1, 2, 3], dtype=np.int64),
                      (np.array([4., 5.]), np.array(
                          [6., 7.])), np.array([8, 9, 10], dtype=np.int64)),
          expected_shapes=[[[1, 3], [1, 3], [1, 2], [1, 2]]]))
  def testNestedStructure(self, components, expected_shapes):
    # [FUNCTION DOC]
    # Parameters: components (an integer vector, a pair of float vectors and another integer
    # vector), expected_shapes (the expected shapes of the four batched outputs; unknown batch
    # dimension in graph mode, one in eager mode). Maps the element into a nested pair of pairs,
    # rebuilds it with a flat-map and batches by 32. Reads the first element twice from fresh
    # iterators and asserts the dtypes (two 64-bit integers, two 64-bit floats) and shapes.
    dataset = dataset_ops.Dataset.from_tensors(components)
    dataset = dataset.map(lambda x, y, z: ((x, z), (y[0], y[1])))

    dataset = dataset.flat_map(
        lambda x, y: dataset_ops.Dataset.from_tensors(
            ((x[0], x[1]), (y[0], y[1])))).batch(32)

    get_next = self.getNext(dataset)
    (w, x), (y, z) = get_next()
    self.assertEqual(dtypes.int64, w.dtype)
    self.assertEqual(dtypes.int64, x.dtype)
    self.assertEqual(dtypes.float64, y.dtype)
    self.assertEqual(dtypes.float64, z.dtype)
    self.assertEqual(expected_shapes, [
        w.shape.as_list(),
        x.shape.as_list(),
        y.shape.as_list(),
        z.shape.as_list()
    ])

    get_next = self.getNext(dataset)
    (w, x), (y, z) = get_next()
    self.assertEqual(dtypes.int64, w.dtype)
    self.assertEqual(dtypes.int64, x.dtype)
    self.assertEqual(dtypes.float64, y.dtype)
    self.assertEqual(dtypes.float64, z.dtype)
    self.assertEqual(expected_shapes, [
        w.shape.as_list(),
        x.shape.as_list(),
        y.shape.as_list(),
        z.shape.as_list()
    ])

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDict(self):
    # [FUNCTION DOC]
    # Builds a dataset from a nested dictionary and asserts the legacy output types (32-bit
    # integer, float, integer) and shapes (scalar, length two, length three) at each leaf.
    components = {"a": {"aa": 1, "ab": [2.0, 2.0]}, "b": [3, 3, 3]}
    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEqual(dtypes.int32,
                     dataset_ops.get_legacy_output_types(dataset)["a"]["aa"])
    self.assertEqual(dtypes.float32,
                     dataset_ops.get_legacy_output_types(dataset)["a"]["ab"])
    self.assertEqual(dtypes.int32,
                     dataset_ops.get_legacy_output_types(dataset)["b"])
    self.assertEqual([],
                     dataset_ops.get_legacy_output_shapes(dataset)["a"]["aa"])
    self.assertEqual([2],
                     dataset_ops.get_legacy_output_shapes(dataset)["a"]["ab"])
    self.assertEqual([3],
                     dataset_ops.get_legacy_output_shapes(dataset)["b"])

  @combinations.generate(test_base.default_test_combinations())
  def testNonSequenceNestedStructure(self):
    # [FUNCTION DOC]
    # Starts from a single 64-bit integer vector and asserts the element type and shape are
    # preserved through a filter, a map that stacks the vector twice (shape two by three) and a
    # flat-map that slices it back to vectors of three. Finally reads an element and asserts its
    # dtype and shape.
    components = np.array([1, 2, 3], dtype=np.int64)

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual([3], dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.filter(
        lambda x: math_ops.reduce_all(math_ops.equal(x, components)))
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual([3], dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.map(lambda x: array_ops_stack.stack([x, x]))
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual([2, 3], dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.flat_map(
        lambda x: dataset_ops.Dataset.from_tensor_slices(x))
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual([3], dataset_ops.get_legacy_output_shapes(dataset))

    get_next = self.getNext(dataset)
    self.assertEqual(dtypes.int64, get_next().dtype)
    self.assertEqual([3], get_next().shape)

  # TODO(b/121264236): needs mechanism for multiple device in eager mode.
  @combinations.generate(test_base.graph_only_combinations())
  def testSplitPipeline(self):
    # [FUNCTION DOC]
    # Graph-only test (eager multi-device support is a pending to-do). Opens a session configured
    # with two CPU devices, builds a dataset of the scalar zero, and maps it twice, each map adding
    # a resource variable placed on a different CPU device (variables initialized before the
    # iterator is created to avoid placement overriding colocation). Asserts the first element is 2.
    with session.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:

      dataset = dataset_ops.Dataset.from_tensors(0)

      # Define a pipeline that attempts to use variables on two
      # different devices.
      #
      # Initialize the variables before creating to iterator, to avoid the
      # placement algorithm overriding the DT_RESOURCE colocation constraints.
      with ops.device("/cpu:0"):
        var_0 = resource_variable_ops.ResourceVariable(initial_value=1)
      dataset = dataset.map(lambda x: x + var_0.read_value())
      sess.run(var_0.initializer)

      with ops.device("/cpu:1"):
        var_1 = resource_variable_ops.ResourceVariable(initial_value=1)
      dataset = dataset.map(lambda x: x + var_1.read_value())
      sess.run(var_1.initializer)

      iterator = dataset_ops.make_initializable_iterator(dataset)
      sess.run(iterator.initializer)

      self.assertEqual(sess.run(iterator.get_next()), 2)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Verifies the constructor accepts an operation name: builds a dataset from the scalar 42 with
    # a name and asserts it produces that element.
    dataset = dataset_ops.Dataset.from_tensors(42, name="from_tensors")
    self.assertDatasetProduces(dataset, [42])


class FromTensorsCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for the from-tensors dataset iterator. Inherits the checkpoint test base
  # class, which provides the verification callback that saves and restores iterator state.

  def _build_tensor_dataset(self, variable_array, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test.
    # Parameters: variable_array (array used as the first component alongside a fixed vector and a
    # scalar), options (optional dataset options applied when provided). Returns the dataset.
    components = (variable_array, np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components)
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
    # Checkpoint test. Parameters: verify_fn (verification callback), symbolic_checkpoint (whether
    # to enable the symbolic checkpoint option). Runs verification on a dataset built from a scalar
    # array, expecting exactly one output.
    arr = np.array(1)
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self, lambda: self._build_tensor_dataset(arr, options), num_outputs=1)


class FromTensorsRandomAccessTest(test_base.DatasetTestBase,
                                  parameterized.TestCase):
  # [CLASS DOC]
  # Tests random access (reading an element by index) on from-tensors datasets, using the
  # experimental random-access helper.

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidIndex(self):
    # [FUNCTION DOC]
    # Asserts that reading index minus one or index one from a single-element dataset raises an
    # out-of-range error.
    dataset = dataset_ops.Dataset.from_tensors([1, 2, 3])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, -1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 1))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testBasic(self):
    # [FUNCTION DOC]
    # Asserts random access at index zero returns the whole element (a range of four) and index one
    # raises an out-of-range error.
    dataset = dataset_ops.Dataset.from_tensors(range(4))
    self.assertAllEqual(self.evaluate(random_access.at(dataset, 0)), range(4))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 1))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testWithOptions(self):
    # [FUNCTION DOC]
    # Same as the basic random-access test, but the dataset first has the map-and-batch fusion
    # optimization option enabled; asserts index zero returns the element and index one is out of
    # range.
    dataset = dataset_ops.Dataset.from_tensors(range(4))
    options = options_lib.Options()
    options.experimental_optimization.map_and_batch_fusion = True
    dataset = dataset.with_options(options)

    self.assertAllEqual(self.evaluate(random_access.at(dataset, 0)), range(4))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 1))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testEmptyDataset(self):
    # [FUNCTION DOC]
    # The element is an empty list. Asserts random access at index zero returns an empty result
    # and index one raises an out-of-range error.
    dataset = dataset_ops.Dataset.from_tensors([])
    self.assertAllEqual(self.evaluate(random_access.at(dataset, 0)), [])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 1))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testNumpyArray(self):
    # [FUNCTION DOC]
    # The element is a tuple of three arrays. Asserts random access at index zero returns the
    # three arrays unchanged and index one raises an out-of-range error.
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components)
    result = self.evaluate(random_access.at(dataset, 0))
    for i in range(3):
      self.assertAllEqual(result[i], components[i])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 1))

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsNestedDataset(self):
    # [FUNCTION DOC]
    # The element is a range-of-ten dataset. Random access at index zero yields the nested dataset,
    # and random access on it at each index from zero to nine must return that index.
    dataset = dataset_ops.Dataset.from_tensors(dataset_ops.Dataset.range(10))
    result = random_access.at(dataset, 0)
    for i in range(10):
      self.assertEqual(self.evaluate(random_access.at(result, i)), i)


class FromTensorsGlobalShuffleTest(
    test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the experimental global shuffle transformation applied to a from-tensors dataset.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              repetitions=[1, 3],
              seed=[None, 19],
              reshuffle_each_iteration=[True, False])))
  def testTensorsDataset(
      self,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    # [FUNCTION DOC]
    # Parameters: repetitions (repeat count), seed (shuffle seed or nothing), reshuffle_each_iteration
    # (whether order changes per epoch). Builds a single-element dataset from a tuple of arrays,
    # optionally repeats it, applies the global shuffle, and asserts the output equals the element
    # repeated the given number of times and that its length equals the dataset cardinality.
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components)
    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = [components] * repetitions
    self.assertDatasetProduces(
        dataset, expected_output=expected, requires_initialization=True)
    self.assertLen(expected, self.evaluate(dataset.cardinality()))


class FromTensorsGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for a globally shuffled from-tensors dataset, with and without the symbolic
  # checkpoint option.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              repetitions=[1, 3],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testGlobalShuffleTensorsDataset(
      self,
      verify_fn: Callable[..., None],
      repetitions: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Parameters: verify_fn (verification callback), repetitions (repeat count),
    # reshuffle_each_iteration (reshuffle per epoch), symbolic_checkpoint (value assigned to the
    # symbolic checkpoint option). Builds a globally shuffled dataset and runs the verification
    # expecting ten times repetitions outputs, ignoring order when reshuffling each iteration.

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: wraps a range-of-ten dataset as one element, optionally repeats it, globally
      # shuffles it with a fixed seed of 42, flattens it, sets the symbolic checkpoint option and
      # returns the dataset.
      dataset = dataset_ops.Dataset.from_tensors(dataset_ops.Dataset.range(10))
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      dataset = dataset.flat_map(lambda x: x)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=10 * repetitions,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
