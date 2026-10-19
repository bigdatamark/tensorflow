# [MODULE DOC]
# General tests for the tf.data Dataset base class and related machinery. Covers graph
# serialization (including stateful maps and lookup tables), tracing a dataset into a variant
# creation function, the reported input datasets and functions of each source and transformation,
# debug strings, element-spec round trips through the dataset spec type, same-graph errors,
# eager iteration, datasets as arguments of graph functions (trace reuse), legacy structure
# accessors, none components, restructuring errors, named-tuple specs, and a debug-mode test class
# that toggles eager/sequential execution of dataset functions. No import-time side effects;
# running as a script starts the TensorFlow test runner.
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
"""Tests for `tf.data.Dataset`."""

import collections
import os
import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class DatasetTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test class for the Dataset base class: serialization, input/function introspection,
  # element specs, graph/eager behavior and structure handling. Has helper methods shared by the
  # input-introspection tests (unary, binary, variadic, interleave) and the spec round-trip tests.

  @combinations.generate(test_base.default_test_combinations())
  def testAsSerializedGraph(self):
    # [FUNCTION DOC]
    # Scenario: serializes a range dataset to a graph definition and parses it. Asserts that a
    # node with the range dataset operation exists. Returns nothing.
    dataset = dataset_ops.Dataset.range(10)
    graph = graph_pb2.GraphDef().FromString(
        self.evaluate(dataset._as_serialized_graph()))
    self.assertTrue(any(node.op == "RangeDataset" for node in graph.node))

  def testAsSerializedGraphStateful(self):
    # [FUNCTION DOC]
    # Scenario: a range dataset mapped with a random-number function (stateful). Serializing with
    # an external-state policy of fail must raise a failed-precondition error. Note this test has
    # no combinations decorator, so it runs once in the default mode. Returns nothing.
    dataset = dataset_ops.Dataset.range(10).map(
        lambda _: random_ops.random_uniform(()))
    with self.assertRaises(errors.FailedPreconditionError):
      self.evaluate(
          dataset._as_serialized_graph(external_state_policy=options_lib
                                       .ExternalStatePolicy.FAIL))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              init_source=["textfile", "keyvaluetensor", "dataset"])))
  def testLookupTableGraphSerialization(self, init_source):
    # [FUNCTION DOC]
    # Scenario: builds a static hash table (values 10 and 11, default -1) from one of three
    # initializer sources (text file, key/value tensors, dataset), maps a range of 3 through the
    # lookup, initializes tables, round-trips the dataset through graph serialization, deletes
    # the original table and dataset, and asserts the restored dataset produces 10, 11, -1.
    # Input is the initializer source name. Returns nothing.
    vals = [10, 11]
    initializer = self.lookupTableInitializer(init_source, vals)
    table = lookup_ops.StaticHashTable(initializer, -1)
    dataset = dataset_ops.Dataset.range(3)
    dataset = dataset.map(table.lookup)
    self.evaluate(lookup_ops.tables_initializer())
    round_tripped = self.graphRoundTrip(dataset)
    del table
    del dataset
    self.assertDatasetProduces(
        round_tripped, [10, 11, -1], requires_initialization=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testAsFunctionWithMap(self):
    # [FUNCTION DOC]
    # Eager-only scenario: on the CPU, traces a mapped range dataset into a variant-creating
    # function, calls it, wraps the resulting variant in a revived dataset with the original
    # element spec, and asserts it produces the even numbers 0 to 8. Returns nothing.
    with ops.device("CPU"):
      original_dataset = dataset_ops.Dataset.range(5).map(lambda x: x * 2)
      fn = original_dataset._trace_variant_creation()
      variant = fn()

      revived_dataset = dataset_ops._VariantDataset(
          variant, original_dataset.element_spec)
      self.assertDatasetProduces(revived_dataset, range(0, 10, 2))

  @combinations.generate(test_base.eager_only_combinations())
  def testAsFunctionWithMapInFlatMap(self):
    # [FUNCTION DOC]
    # Eager-only scenario: like the map variant but the dataset is a flat-map whose inner dataset
    # itself contains a map. Traces to a variant, revives it, and asserts the revived dataset
    # yields the same elements as iterating the original. Returns nothing.
    with ops.device("CPU"):
      original_dataset = dataset_ops.Dataset.range(5).flat_map(
          lambda x: dataset_ops.Dataset.range(5).map(lambda x: x * 2))
      fn = original_dataset._trace_variant_creation()
      variant = fn()

      revived_dataset = dataset_ops._VariantDataset(
          variant, original_dataset.element_spec)
      self.assertDatasetProduces(revived_dataset, list(original_dataset))

  @combinations.generate(test_base.eager_only_combinations())
  def testAsFunctionFromReader(self):
    # [FUNCTION DOC]
    # Eager-only scenario: writes a gzip-compressed record file in the test temp directory with
    # three records, reads it with a record dataset, traces that dataset to a variant, revives it
    # and asserts it produces the three records. Side effect: creates a file. Returns nothing.
    with ops.device("CPU"):
      file_path = os.path.join(self.get_temp_dir(),
                               "{}.tfrecord.gz".format("tf_record_asset"))
      with tf_record.TFRecordWriter(file_path, "GZIP") as f:
        for v in ["a", "aa", "aaa"]:
          f.write(str(v))
      original_dataset = readers.TFRecordDataset([file_path],
                                                 compression_type="GZIP")
      fn = original_dataset._trace_variant_creation()
      variant = fn()

      revived_dataset = dataset_ops._VariantDataset(
          variant, original_dataset.element_spec)
      self.assertDatasetProduces(revived_dataset, ["a", "aa", "aaa"])

  def _testNumInputs(self, dataset, num_inputs):
    # [FUNCTION DOC]
    # HELPER: asserts that the dataset reports exactly num_inputs input datasets. Returns
    # nothing.
    self.assertLen(dataset._inputs(), num_inputs)

  @combinations.generate(test_base.default_test_combinations())
  def testFixedLengthRecordInputs(self):
    # [FUNCTION DOC]
    # Scenario: a fixed-length record reader dataset is a source and reports 0 inputs. Returns
    # nothing.
    dataset = readers.FixedLengthRecordDataset("", 42)
    self._testNumInputs(dataset, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorInputs(self):
    # [FUNCTION DOC]
    # Scenario: a dataset built from a generator yielding 42 reports exactly 1 input. Returns
    # nothing.
    def gen():
      # [FUNCTION DOC]
      # HELPER: generator that yields the single value 42.
      yield 42

    dataset = dataset_ops.Dataset.from_generator(gen, dtypes.int32)
    self._testNumInputs(dataset, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsInputs(self):
    # [FUNCTION DOC]
    # Scenario: a dataset from a single tensor value reports 0 inputs. Returns nothing.
    dataset = dataset_ops.Dataset.from_tensors([42])
    self._testNumInputs(dataset, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testRangeInputs(self):
    # [FUNCTION DOC]
    # Scenario: a range dataset reports 0 inputs. Returns nothing.
    dataset = dataset_ops.Dataset.range(10)
    self._testNumInputs(dataset, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testTextLineInputs(self):
    # [FUNCTION DOC]
    # Scenario: a text-line reader dataset over an empty file name reports 0 inputs. Returns
    # nothing.
    dataset = readers.TextLineDataset("")
    self._testNumInputs(dataset, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordInputs(self):
    # [FUNCTION DOC]
    # Scenario: a record-file reader dataset over an empty file name reports 1 input (it is built
    # on an inner dataset of file names). Returns nothing.
    dataset = readers.TFRecordDataset("")
    self._testNumInputs(dataset, 1)

  @combinations.generate(
      combinations.combine(tf_api_version=1, mode=["eager", "graph"]))
  def testDatasetComplexSourceInputs(self):
    # [FUNCTION DOC]
    # Scenario (API version 1 only): a dataset from sparse tensor slices has no inputs; asserts
    # the input list is empty. Returns nothing.
    dataset_fn = dataset_ops.Dataset.from_sparse_tensor_slices(
        sparse_tensor.SparseTensor(
            indices=np.array([[0, 0], [1, 0], [2, 0]]),
            values=np.array([0, 0, 0]),
            dense_shape=np.array([3, 1])))
    self.assertEmpty(dataset_fn._inputs())

  def _testUnaryInputs(self, dataset_fn):
    # [FUNCTION DOC]
    # HELPER: for the unary transformation input tests. Applies dataset_fn (a callback mapping a
    # dataset to a transformed dataset) to an empty range dataset and asserts the result's input
    # list is exactly that one range dataset. Returns nothing.
    input_dataset = dataset_ops.Dataset.range(0)
    self.assertEqual([input_dataset], dataset_fn(input_dataset)._inputs())

  @combinations.generate(test_base.default_test_combinations())
  def testBatchInputs(self):
    # [FUNCTION DOC]
    # Scenario: batch has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.batch(10))

  @combinations.generate(test_base.default_test_combinations())
  def testCacheInputs(self):
    # [FUNCTION DOC]
    # Scenario: cache has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.cache())

  @combinations.generate(test_base.default_test_combinations())
  def testFilterInputs(self):
    # [FUNCTION DOC]
    # Scenario: filter has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.filter(lambda x: True))

  @combinations.generate(test_base.default_test_combinations())
  def testFlatMapInputs(self):
    # [FUNCTION DOC]
    # Scenario: flat-map has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(
        lambda x: x.flat_map(lambda x: dataset_ops.Dataset.range(0)))

  @combinations.generate(test_base.default_test_combinations())
  def testMapInputs(self):
    # [FUNCTION DOC]
    # Scenario: map has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.map(lambda x: x))

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchInputs(self):
    # [FUNCTION DOC]
    # Scenario: padded batch has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.padded_batch(10, []))

  @combinations.generate(test_base.default_test_combinations())
  def testParallelMapInputs(self):
    # [FUNCTION DOC]
    # Scenario: map with 2 parallel calls has exactly its upstream dataset as input. Returns
    # nothing.
    self._testUnaryInputs(lambda x: x.map(lambda x: x, num_parallel_calls=2))

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatInputs(self):
    # [FUNCTION DOC]
    # Scenario: repeat has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.repeat())

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleInputs(self):
    # [FUNCTION DOC]
    # Scenario: shuffle has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.shuffle(10))

  @combinations.generate(test_base.default_test_combinations())
  def testSkipInputs(self):
    # [FUNCTION DOC]
    # Scenario: skip has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.skip(1))

  @combinations.generate(test_base.default_test_combinations())
  def testTakeInputs(self):
    # [FUNCTION DOC]
    # Scenario: take has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.take(1))

  @combinations.generate(test_base.default_test_combinations())
  def testWindowInputs(self):
    # [FUNCTION DOC]
    # Scenario: window has exactly its upstream dataset as input. Returns nothing.
    self._testUnaryInputs(lambda x: x.window(10))

  @combinations.generate(test_base.default_test_combinations())
  def testUnaryTransformationInputsApply(self):
    # [FUNCTION DOC]
    # Scenario: a transformation applied via the apply method (here cache) still has the
    # upstream dataset as its only input. Returns nothing.
    input_dataset = dataset_ops.Dataset.range(0)
    dataset = input_dataset.apply(lambda dataset: dataset.cache())

    self.assertEqual([input_dataset], dataset._inputs())

  def _testInputsWithInterleaveFn(self, dataset_fn, interleave_parallelism):
    # [FUNCTION DOC]
    # HELPER: for the interleave input tests.
    # KNOWN ISSUE: the dataset_fn parameter is never used; callers pass a callback that is
    # ignored.
    # Builds an empty range, interleaves it with an inner empty range (cycle length 2, the given
    # parallelism, possibly none for sequential) and asserts the only input is the upstream
    # range. Returns nothing.
    input_dataset = dataset_ops.Dataset.range(0)
    dataset = input_dataset.interleave(
        lambda x: dataset_ops.Dataset.range(0),
        cycle_length=2,
        num_parallel_calls=interleave_parallelism)
    self.assertEqual([input_dataset], dataset._inputs())

  @combinations.generate(test_base.default_test_combinations())
  def testParallelInterleaveInputs(self):
    # [FUNCTION DOC]
    # Scenario: parallel interleave (parallelism 2) has only its upstream dataset as input.
    # Returns nothing.
    self._testInputsWithInterleaveFn(lambda: dataset_ops.range(0), 2)

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveInputs(self):
    # [FUNCTION DOC]
    # Scenario: sequential interleave has only its upstream dataset as input. Returns nothing.
    self._testInputsWithInterleaveFn(lambda: dataset_ops.range(0), None)

  @combinations.generate(test_base.default_test_combinations())
  def testDebugString(self):
    # [FUNCTION DOC]
    # Scenario: range mapped then filtered; the debug string must mention the range, map and
    # filter transformations. Returns nothing.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(lambda x: x**2)
    dataset = dataset.filter(lambda x: x > 10)
    debug_string = dataset.__debug_string__()
    for transformation in ["Range", "Map", "Filter"]:
      self.assertIn(transformation, debug_string)

  @combinations.generate(test_base.default_test_combinations())
  def testNoWarnings(self):
    # [FUNCTION DOC]
    # Scenario: with the warnings function temporarily replaced by a mock, builds an interleave
    # of empty ranges and asserts no warning was issued. Returns nothing.
    with test.mock.patch.object(warnings, "warn") as mock_log:
      dataset_ops.Dataset.range(0).interleave(
          lambda x: dataset_ops.Dataset.range(0), cycle_length=2)
      self.assertEmpty(mock_log.call_args_list)

  def _testBinaryInputs(self, dataset_fn):
    # [FUNCTION DOC]
    # HELPER: applies dataset_fn (a callback combining two datasets) to a range of 0 and a range
    # of 1 and asserts the result's inputs are exactly those two datasets in order. Returns
    # nothing.
    input1 = dataset_ops.Dataset.range(0)
    input2 = dataset_ops.Dataset.range(1)
    self.assertEqual([input1, input2], dataset_fn(input1, input2)._inputs())

  @combinations.generate(test_base.default_test_combinations())
  def testConcatenateInputs(self):
    # [FUNCTION DOC]
    # Scenario: concatenate has both operand datasets as inputs, in order. Returns nothing.
    self._testBinaryInputs(lambda x, y: x.concatenate(y))

  def _testVariadicInputs(self, dataset_fn, input_datasets):
    # [FUNCTION DOC]
    # HELPER: applies dataset_fn (here the zip factory) to a possibly nested structure of
    # datasets and asserts the result's inputs equal the flattened list of the datasets. Returns
    # nothing.
    self.assertEqual(
        nest.flatten(input_datasets),
        dataset_fn(input_datasets)._inputs())

  @combinations.generate(test_base.default_test_combinations())
  def testZipOneInputs(self):
    # [FUNCTION DOC]
    # Scenario: zip of a single dataset has that dataset as its input. Returns nothing.
    input_datasets = dataset_ops.Dataset.range(0)
    self._testVariadicInputs(dataset_ops.Dataset.zip, input_datasets)

  @combinations.generate(test_base.default_test_combinations())
  def testZipNestInputs(self):
    # [FUNCTION DOC]
    # Scenario: zip of a nested tuple of three datasets has the flattened three datasets as
    # inputs. Returns nothing.
    input_datasets = (dataset_ops.Dataset.range(0),
                      (dataset_ops.Dataset.range(1),
                       dataset_ops.Dataset.range(2)))
    self._testVariadicInputs(dataset_ops.Dataset.zip, input_datasets)

  @combinations.generate(test_base.default_test_combinations())
  def testZipTupleInputs(self):
    # [FUNCTION DOC]
    # Scenario: zip of a tuple of two datasets has both as inputs. Returns nothing.
    input_datasets = (dataset_ops.Dataset.range(0),
                      dataset_ops.Dataset.range(1))
    self._testVariadicInputs(dataset_ops.Dataset.zip, input_datasets)

  @combinations.generate(test_base.default_test_combinations())
  def testFunctions(self):
    # [FUNCTION DOC]
    # Scenario: a mapped range dataset reports exactly one user function. Returns nothing.
    dataset = dataset_ops.Dataset.range(5).map(lambda x: x * 2)
    self.assertLen(dataset._functions(), 1)

  @combinations.generate(test_base.default_test_combinations())
  def testCollectInputs(self):
    # [FUNCTION DOC]
    # Scenario: builds a graph of datasets where a base range is concatenated with itself and
    # then zipped with itself and the concatenation. A breadth-first traversal over the input
    # lists (counting repeated visits) must visit the base 5 times, the concatenation twice and
    # the zip once. Returns nothing.
    ds1 = dataset_ops.Dataset.range(0)
    ds2 = ds1.concatenate(ds1)
    ds3 = dataset_ops.Dataset.zip((ds2, ds1, ds2))

    inputs = []
    queue = [ds3]
    while queue:
      ds = queue[0]
      queue = queue[1:]
      queue.extend(ds._inputs())
      inputs.append(ds)

    self.assertEqual(5, inputs.count(ds1))
    self.assertEqual(2, inputs.count(ds2))
    self.assertEqual(1, inputs.count(ds3))

  def _testDatasetSpec(self, tf_value, expected_element_structure):
    # [FUNCTION DOC]
    # HELPER: shared checker for the spec tests. Builds a one-element dataset whose element is
    # tf_value (a tensor, sparse tensor, nested structure, dataset or optional) and asserts the
    # dataset's own type spec is a dataset spec, its element structure is compatible with
    # expected_element_structure, and its flat tensor type and shape are a single scalar variant.
    # Then round-trips the dataset through the spec's tensor-list conversion and asserts the
    # restored dataset produces the same value (for dataset values compared via flattening, for
    # optionals via their contained value). Returns nothing.
    dataset = dataset_ops.Dataset.from_tensors(0).map(lambda _: tf_value)
    dataset_structure = structure.type_spec_from_value(dataset)
    self.assertIsInstance(dataset_structure, dataset_ops.DatasetSpec)

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(dataset), expected_element_structure))
    self.assertEqual([dtypes.variant],
                     structure.get_flat_tensor_types(dataset_structure))
    self.assertEqual([tensor_shape.TensorShape([])],
                     structure.get_flat_tensor_shapes(dataset_structure))

    # Assert that the `Dataset` survives a round-trip via _from_tensor_list()
    # and _to_tensor_list().
    round_trip_dataset = dataset_structure._from_tensor_list(
        dataset_structure._to_tensor_list(dataset))

    value = tf_value

    if isinstance(value, dataset_ops.Dataset):
      self.assertDatasetsEqual(value, dataset.flat_map(lambda x: x))
    elif isinstance(value, optional_ops.Optional):
      self.assertDatasetProduces(
          round_trip_dataset.map(lambda opt: opt.get_value()),
          [self.evaluate(value.get_value())],
          requires_initialization=True)
    else:
      self.assertDatasetProduces(
          round_trip_dataset, [self.evaluate(tf_value)],
          requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testTensorDatasetSpec(self):
    # [FUNCTION DOC]
    # Scenario: runs the spec checker on a scalar float tensor with a scalar float tensor spec.
    # Returns nothing.
    self._testDatasetSpec(
        constant_op.constant(37.0), tensor_spec.TensorSpec([], dtypes.float32))

  @combinations.generate(test_base.default_test_combinations())
  def testSparseTensorDatasetSpec(self):
    # [FUNCTION DOC]
    # Scenario: runs the spec checker on a one-entry sparse tensor of shape 1 with the matching
    # sparse tensor spec. Returns nothing.
    self._testDatasetSpec(
        sparse_tensor.SparseTensor(
            indices=[[0]],
            values=constant_op.constant([0], dtype=dtypes.int32),
            dense_shape=[1]), sparse_tensor.SparseTensorSpec([1], dtypes.int32))

  @combinations.generate(test_base.default_test_combinations())
  def testNestDatasetSpec(self):
    # [FUNCTION DOC]
    # Scenario: runs the spec checker on a mapping holding a float scalar and a pair of string
    # tensors, with the matching nested spec. Returns nothing.
    self._testDatasetSpec(
        {
            "a": constant_op.constant(37.0),
            "b": (constant_op.constant(["Foo"]), constant_op.constant("Bar"))
        }, {
            "a":
                tensor_spec.TensorSpec([], dtypes.float32),
            "b": (
                tensor_spec.TensorSpec([1], dtypes.string),
                tensor_spec.TensorSpec([], dtypes.string),
            )
        })

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetDatasetSpec(self):
    # [FUNCTION DOC]
    # Scenario: runs the spec checker on a dataset as the element value, with a dataset spec of
    # scalar 32-bit integers. Returns nothing.
    self._testDatasetSpec(
        dataset_ops.Dataset.from_tensor_slices(
            constant_op.constant([1, 2, 3])),
        dataset_ops.DatasetSpec(tensor_spec.TensorSpec([], dtypes.int32)))

  @combinations.generate(test_base.default_test_combinations())
  def testOptionalDatasetSpec(self):
    # [FUNCTION DOC]
    # Scenario: runs the spec checker on an optional holding a float value, with the matching
    # optional spec. Returns nothing.
    self._testDatasetSpec(
        optional_ops.Optional.from_value(37.0),
        optional_ops.OptionalSpec(tensor_spec.TensorSpec([], dtypes.float32)))

  @combinations.generate(test_base.graph_only_combinations())
  def testSameGraphError(self):
    # [FUNCTION DOC]
    # Graph-only scenario: a dataset is created in one graph and batch is applied inside a
    # different default graph; asserts a value error saying it must be from the same graph.
    # Returns nothing.
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError, "must be from the same graph"):
        dataset = dataset.batch(2)

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testSameGraphErrorOneShot(self):
    # [FUNCTION DOC]
    # Scenario (API version 1, graph mode): creating a one-shot iterator in a different graph
    # from the dataset's raises a value error telling that the dataset must be created in the
    # same graph as the iterator. Returns nothing.
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  "Make sure that the dataset is created in "
                                  "the same graph as the iterator"):
        _ = dataset_ops.make_one_shot_iterator(dataset)

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testSameGraphErrorInitializable(self):
    # [FUNCTION DOC]
    # Scenario (API version 1, graph mode): creating an initializable iterator in a different
    # graph from the dataset's raises a value error telling that the dataset must be created in
    # the same graph as the iterator. Returns nothing.
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  "Make sure that the dataset is created in "
                                  "the same graph as the iterator"):
        _ = dataset_ops.make_initializable_iterator(dataset)

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(execution_mode=[context.ASYNC, context.SYNC])))
  def testEagerIteration(self, execution_mode):
    # [FUNCTION DOC]
    # Eager-only scenario: under the given execution mode (asynchronous or synchronous),
    # iterates a range of 10 with a plain loop and asserts each element equals the running
    # counter. Input is the execution mode. Returns nothing.
    with context.execution_mode(execution_mode):
      val = 0
      dataset = dataset_ops.Dataset.range(10)
      for foo in dataset:
        self.assertEqual(val, foo.numpy())
        val += 1

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetAsFunctionArgument(self):
    # [FUNCTION DOC]
    # Scenario: a graph function that sums a dataset is called with ranges of 10 and 11 (sums 45
    # and 55) on the CPU. Asserts the dataset is not a captured input, that the two datasets of
    # the same structure share one concrete function (trace reuse), and that a zipped dataset
    # with a different structure gets a different concrete function. Returns nothing.

    @def_function.function
    def _uses_dataset(d):
      # [FUNCTION DOC]
      # HELPER: graph function under test; iterates the dataset d and returns the 64-bit integer
      # sum of its elements.
      accumulator = array_ops.zeros([], dtype=dtypes.int64)
      for value in d:
        accumulator += value
      return accumulator

    with ops.device("CPU"):
      first_dataset = dataset_ops.Dataset.range(10)
      self.assertEqual(45, self.evaluate(_uses_dataset(first_dataset)))
      second_dataset = dataset_ops.Dataset.range(11)
      self.assertEqual(55, self.evaluate(_uses_dataset(second_dataset)))
      first_concrete = _uses_dataset.get_concrete_function(first_dataset)
      # The dataset should not be a captured input
      self.assertEmpty(first_concrete.graph.captures)
      # The two datasets have the same structure and so should re-use a trace.
      self.assertIs(first_concrete,
                    _uses_dataset.get_concrete_function(second_dataset))
      # With a different structure we should use a different trace.
      self.assertIsNot(
          first_concrete,
          _uses_dataset.get_concrete_function(
              dataset_ops.Dataset.zip((first_dataset, second_dataset))))

  @combinations.generate(test_base.default_test_combinations())
  def testLimitedRetracing(self):
    # [FUNCTION DOC]
    # Scenario: a graph function that sums a dataset is called ten times alternately with ranges
    # of 5 and 10 (sums 10 and 45). Asserts that it was traced only once overall, so datasets of
    # the same structure do not trigger retracing. Returns nothing.
    trace_count = [0]

    @def_function.function
    def f(ds):
      # [FUNCTION DOC]
      # HELPER: graph function under test; increments the trace counter (so tracing can be
      # counted), sums the dataset elements and returns the sum.
      trace_count[0] += 1
      counter = np.int64(0)
      for elem in ds:
        counter += elem
      return counter

    dataset = dataset_ops.Dataset.range(5)
    dataset2 = dataset_ops.Dataset.range(10)

    for _ in range(10):
      self.assertEqual(self.evaluate(f(dataset)), 10)
      self.assertEqual(self.evaluate(f(dataset2)), 45)
      self.assertEqual(trace_count[0], 1)

  # pylint: disable=g-long-lambda,unnecessary-lambda
  @combinations.generate(test_base.default_test_combinations())
  def testLegacyStructureAPI(self):
    # [FUNCTION DOC]
    # Scenario: starting from a single element holding a nested tuple (integer vector, pair of
    # float vectors, integer vector), repeatedly applies shuffle, repeat, filter, take, map,
    # flat-map and batch, and after each step asserts the legacy output types and shapes (shapes
    # become batch-prefixed with unknown leading size after batching). Finally builds a dataset
    # from slices of a similar structure and asserts scalar shapes. Returns nothing.
    components = (np.array([1, 2, 3], dtype=np.int64), (np.array([4., 5.]),
                                                        np.array([6., 7.])),
                  np.array([8, 9, 10], dtype=np.int64))

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.shuffle(10, 10)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.repeat(-1)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.filter(lambda x, y, z: True)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.take(5)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.map(lambda x, y, z: ((x, z), (y[0], y[1])))
    self.assertEqual(
        ((dtypes.int64, dtypes.int64), (dtypes.float64, dtypes.float64)),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual((([3], [3]), ([2], [2])),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.flat_map(lambda x, y: dataset_ops.Dataset.from_tensors(
        ((x[0], x[1]), (y[0], y[1]))))
    self.assertEqual(
        ((dtypes.int64, dtypes.int64), (dtypes.float64, dtypes.float64)),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual((([3], [3]), ([2], [2])),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.batch(32)
    self.assertEqual(
        ((dtypes.int64, dtypes.int64), (dtypes.float64, dtypes.float64)),
        dataset_ops.get_legacy_output_types(dataset))
    dataset_output_shapes = dataset_ops.get_legacy_output_shapes(dataset)
    self.assertEqual(
        (([None, 3], [None, 3]), ([None, 2], [None, 2])),
        nest.pack_sequence_as(
            dataset_output_shapes,
            [s.as_list() for s in nest.flatten(dataset_output_shapes)]))

    # Define a separate set of components with matching leading
    # dimension for the from-slices constructor.
    components_for_slices = (np.array([1, 2, 3],
                                      dtype=np.int64), (np.array([4., 5., 6.]),
                                                        np.array([7., 8., 9.])),
                             np.array([10, 11, 12], dtype=np.int64))

    dataset = dataset_ops.Dataset.from_tensor_slices(components_for_slices)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([], ([], []), []),
                     dataset_ops.get_legacy_output_shapes(dataset))

  @combinations.generate(test_base.default_test_combinations())
  def testNoneComponent(self):
    # [FUNCTION DOC]
    # Scenario: a dataset with an element of 42 and a null component. In eager mode asserts the
    # dataset produces that pair; in graph mode uses a one-shot iterator, asserts the second
    # component is null and runs the first to get 42. Returns nothing.
    dataset = dataset_ops.Dataset.from_tensors((42, None))
    if context.executing_eagerly():
      self.assertDatasetProduces(dataset, expected_output=[(42, None)])
    else:
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      next_first, next_second = iterator.get_next()
      self.assertIsNone(next_second)
      with self.cached_session() as sess:
        self.assertEqual(sess.run(next_first), 42)

  @combinations.generate(test_base.default_test_combinations())
  def testNoneComponentInFunction(self):
    # [FUNCTION DOC]
    # Scenario: a range of 10 (32-bit) mapped to pairs with a null second component is consumed
    # inside a graph function that sums the first components. Asserts the sum is 45. Returns
    # nothing.

    @def_function.function
    def fn(ds):
      # [FUNCTION DOC]
      # HELPER: graph function under test; iterates the dataset, ignoring the null second
      # component, and returns the sum of the first components.
      total = 0
      it = iter(ds)
      for elem in it:
        x, _ = elem
        total += x
      return total

    dataset = dataset_ops.Dataset.range(
        10, output_type=dtypes.int32).map(lambda x: (x, None))
    self.assertEqual(self.evaluate(fn(dataset)), 45)

  @combinations.generate(test_base.default_test_combinations())
  def testIncorrectPythonStructure(self):
    # [FUNCTION DOC]
    # Scenario: a range of 10 is deliberately restructured to claim a two-component element
    # structure, then mapped with a function taking two arguments. Reading the output must raise
    # an op error rather than crashing. Returns nothing.
    # Tests that an exception is raised (as opposed to a segfault) when the
    # Python structure assigned to a dataset is incorrect.
    dataset = dataset_ops.Dataset.range(10)
    spec = tensor_spec.TensorSpec([], dtypes.int64)
    new_structure = (spec, spec)
    dataset = dataset_ops._RestructuredDataset(dataset, new_structure)
    dataset = dataset.map(lambda x, y: y)

    with self.assertRaisesOpError(""):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testNamedTupleStructure(self):
    # [FUNCTION DOC]
    # Scenario: builds a dataset whose elements are datasets of a named tuple with an integer and
    # a string field. Asserts the string form of its element spec equals the expected dataset spec
    # text with the named tuple's field specs and a scalar shape. Returns nothing.
    Foo = collections.namedtuple("Foo", ["a", "b"])
    x = Foo(a=3, b="test")
    dataset = dataset_ops.Dataset.from_tensors(x)
    dataset = dataset_ops.Dataset.from_tensor_slices([dataset, dataset])
    self.assertEqual(
        str(dataset.element_spec),
        "DatasetSpec(Foo(a=TensorSpec(shape=(), dtype=tf.int32, name=None), "
        "b=TensorSpec(shape=(), dtype=tf.string, name=None)), TensorShape([]))")

  @combinations.generate(test_base.eager_only_combinations())
  def testIterationError(self):
    # [FUNCTION DOC]
    # Eager-only scenario: a graph function with automatic code conversion disabled tries to loop
    # over a dataset argument with a plain loop. Calling it must raise a value error. Returns
    # nothing.

    @def_function.function(autograph=False)
    def fn(ds):
      # [FUNCTION DOC]
      # HELPER: graph function under test; iterates the dataset with a plain loop and does
      # nothing with the elements. Fails when traced because plain iteration is unsupported
      # without code conversion.
      for _ in ds:
        pass

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(ValueError):
      self.evaluate(fn(dataset))


class DebugDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests of the dataset debug mode, in which dataset functions run eagerly and in sequence.
  # The fixture switches debug mode on before each test and off afterwards (global state).

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: calls the parent setup and then turns the global dataset debug mode on.
    # Returns nothing.
    super(DebugDatasetTest, self).setUp()
    debug_mode.toggle_debug_mode(True)

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: turns the global dataset debug mode off, then calls the parent
    # teardown. Returns nothing.
    debug_mode.toggle_debug_mode(False)
    super(DebugDatasetTest, self).tearDown()

  @combinations.generate(test_base.eager_only_combinations())
  def testDebugModeEagerExecution(self):
    # [FUNCTION DOC]
    # Eager-only scenario: maps a function with a side effect (appending to a list) over a range
    # of 10 in debug mode. Asserts the output is 0 to 9 and that the function body ran 11 times
    # (once per element plus one tracing run to discover output types and shapes). Returns
    # nothing.
    counter = []
    ds = dataset_ops.Dataset.range(10)

    def map_fn(x):
      # [FUNCTION DOC]
      # HELPER: map function with a side effect; records one call in the enclosing list and
      # returns its input unchanged.
      counter.append(1)
      return x

    ds = ds.map(map_fn)
    self.assertDatasetProduces(ds, list(range(10)))

    # The body of `map_fn` will be executed 11 times since the implementation
    # traces the function to figure out what the types and shapes of its
    # outputs are.
    self.assertLen(counter, 11)

  @combinations.generate(test_base.eager_only_combinations())
  def testDebugModeGenerator(self):
    # [FUNCTION DOC]
    # Eager-only scenario: in debug mode, a dataset from a generator yielding 0 to 9 with a scalar
    # 64-bit integer output signature produces 0 to 9. Returns nothing.
    def gen():
      # [FUNCTION DOC]
      # HELPER: generator yielding the integers 0 through 9.
      yield from range(10)

    ds = dataset_ops.Dataset.from_generator(
        gen,
        output_signature=tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64))
    self.assertDatasetProduces(ds, list(range(10)))

  @combinations.generate(test_base.eager_only_combinations())
  def testDebugModeGeneratorTwoComponents(self):
    # [FUNCTION DOC]
    # Eager-only scenario: in debug mode, a dataset from a generator yielding ten pairs (n, n+1)
    # with a two-component scalar 64-bit integer signature produces exactly those pairs. Returns
    # nothing.
    data = [(n, n+1) for n in range(10)]

    def gen():
      # [FUNCTION DOC]
      # HELPER: generator yielding each pair from the enclosing list of ten pairs.
      yield from data

    ds = dataset_ops.Dataset.from_generator(
        gen,
        output_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64),
                          tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64)))
    self.assertDatasetProduces(ds, data)

  @combinations.generate(test_base.eager_only_combinations())
  def testDebugModeSequentialExecution(self):
    # [FUNCTION DOC]
    # Eager-only scenario: builds a pipeline over a range of 10 with an assertion that the next
    # transformations are interleave, map, batch and a finite take; then interleave (cycle 10,
    # 10 parallel calls), map (squares, 10 parallel calls), batch of 5 (2 parallel calls),
    # prefetch 2 and take 2. In debug mode parallelism is turned sequential. Asserts the output
    # is the two batches [0,1,4,9,16] and [25,36,49,64,81]. Returns nothing.
    ds = dataset_ops.Dataset.range(10)
    ds = ds.apply(
        testing.assert_next(["Interleave", "Map", "Batch", "FiniteTake"]))
    ds = ds.interleave(
        dataset_ops.Dataset.from_tensors,
        cycle_length=10,
        num_parallel_calls=10)
    ds = ds.map(lambda x: x * x, num_parallel_calls=10)
    ds = ds.batch(batch_size=5, num_parallel_calls=2)
    ds = ds.prefetch(buffer_size=2)
    ds = ds.take(2)
    self.assertDatasetProduces(ds, [[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]])


if __name__ == "__main__":
  test.main()
