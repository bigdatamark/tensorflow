# [MODULE DOC]
# Unit tests for Grappler shape inference through tf.data datasets. For many dataset sources
# (tensors, slices, generators, range, iterator structure and string handles) and transformations
# (concatenate, prefetch, repeat, shuffle, cache, take, skip, shard, filter, batch, padded batch,
# flat map, interleave, map), builds a graph with a one-shot iterator, wraps it in a Grappler
# item, and checks that the inferred output shape of the iterator get-next op matches the
# expected shape. Defines one test class; no import-time side effects.
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
"""Tests for the datasets shape inference."""

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.grappler import item
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class GrapplerTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking Grappler's static shape inference for the iterator get-next op over many
  # dataset constructions. Most tests loop over a list of tensor and expected-shape pairs, build
  # a graph per case and compare shapes. Offers a shared transformation helper and a shape
  # conversion helper.

  def testFromTensors(self):
    # [FUNCTION DOC]
    # For a scalar, a vector and a matrix constant, builds a dataset from the whole tensor and
    # expects the inferred get-next shape to equal the tensor shape.
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testFromTensorSlices(self):
    # [FUNCTION DOC]
    # For vector, matrix and rank-3 inputs, builds a dataset of slices along the first dimension
    # and expects the inferred element shape to be the tensor shape without its first dimension.
    test_cases = [{
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[[1, 2, 3]]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensor_slices(test_case['tensor'])
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testFromGenerator(self):
    # [FUNCTION DOC]
    # For scalar, vector and matrix values, builds a dataset from a generator yielding that value
    # with int64 type and an explicit output shape, and expects the inferred get-next shape to
    # equal the declared shape.
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:

      def make_generator(tensor):
        # [FUNCTION DOC]
        # HELPER: factory for the generator dataset test.
        # Takes a value and returns a no-argument generator function that yields that value once.

        def generator():
          # [FUNCTION DOC]
          # HELPER: generator source for the generator dataset test.
          # Takes no arguments; yields the captured value once.
          yield tensor

        return generator

      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_generator(
            make_generator(test_case['tensor']),
            dtypes.int64,
            output_shapes=test_case['shape'])
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testRange(self):
    # [FUNCTION DOC]
    # Builds a range dataset of 42 elements and expects the inferred get-next shape to be scalar.
    with ops.Graph().as_default() as g:
      dataset = dataset_ops.Dataset.range(42)
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      get_next = iterator.get_next()
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(get_next)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      op_properties = grappler_item.GetOpProperties()
      self.assertEqual(
          tensor_shape.TensorShape([]),
          op_properties['IteratorGetNext'][0].shape)

  def _testTransformation(self, fn):
    # [FUNCTION DOC]
    # HELPER: shared body of the shape-preserving transformation tests.
    # KNOWN ISSUE: the scalar case builds the expected shape from an empty mapping rather than an
    # empty list; it works only because an empty collection yields a scalar shape.
    # Input fn is a callable taking the dataset, the source tensor and the expected shape and
    # returning the transformed dataset. For scalar, vector and matrix constants, builds a
    # dataset from the tensor, applies fn, builds a one-shot iterator graph, and expects the
    # inferred get-next shape to equal the expected shape. Returns nothing.
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape({})
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        dataset = fn(dataset, test_case['tensor'], test_case['shape'])
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testConcatenate(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with a function that concatenates the dataset with
    # another dataset made from the same tensor; the shape must be unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the concatenate test. Ignores the shape argument and returns
      # the dataset concatenated with a single-element dataset of the given tensor.
      del shape
      return dataset.concatenate(dataset_ops.Dataset.from_tensors(tensor))

    self._testTransformation(fn)

  def testPrefetch(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with a prefetch of 42 elements; shape must be unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the prefetch test. Ignores tensor and shape and returns the
      # dataset with a prefetch buffer of 42.
      del tensor, shape
      return dataset.prefetch(42)

    self._testTransformation(fn)

  def testRepeat(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with repeating the dataset 42 times; shape unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the repeat test. Ignores tensor and shape and returns the
      # dataset repeated 42 times.
      del tensor, shape
      return dataset.repeat(42)

    self._testTransformation(fn)

  def testShuffle(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with a shuffle buffer of 42; shape unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the shuffle test. Ignores tensor and shape and returns the
      # dataset shuffled with a buffer of 42.
      del tensor, shape
      return dataset.shuffle(42)

    self._testTransformation(fn)

  def testCache(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with in-memory caching of the dataset; shape unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the cache test. Ignores tensor and shape and returns the
      # dataset with in-memory caching.
      del tensor, shape
      return dataset.cache()

    self._testTransformation(fn)

  def testTake(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with taking the first 42 elements; shape unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the take test. Ignores tensor and shape and returns the
      # dataset limited to its first 42 elements.
      del tensor, shape
      return dataset.take(42)

    self._testTransformation(fn)

  def testSkip(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with skipping the first 42 elements; shape unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the skip test. Ignores tensor and shape and returns the
      # dataset without its first 42 elements.
      del tensor, shape
      return dataset.skip(42)

    self._testTransformation(fn)

  def testShard(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with sharding into 42 shards and keeping shard 0;
    # shape unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the shard test. Ignores tensor and shape and returns shard
      # index 0 of 42 shards of the dataset.
      del tensor, shape
      return dataset.shard(42, 0)

    self._testTransformation(fn)

  def testFilter(self):
    # [FUNCTION DOC]
    # Runs the shared transformation check with a filter that keeps every element; shape unchanged.

    def fn(dataset, tensor, shape):
      # [FUNCTION DOC]
      # HELPER: transformation for the filter test. Ignores tensor and shape and returns the
      # dataset filtered by a predicate that always accepts.
      del tensor, shape
      return dataset.filter(lambda x: True)

    self._testTransformation(fn)

  def as_tensor_shape(self, proto_with_symbolic_values):
    # [FUNCTION DOC]
    # HELPER: shape conversion for the batch tests.
    # KNOWN ISSUE: not prefixed as private and does not start with the test prefix; it also
    # mutates its argument in place rather than working on a copy.
    # Takes a tensor shape proto whose unknown dimensions may be encoded as symbolic values below
    # minus one; rewrites every such dimension size to minus one (mutating the proto) and returns
    # a tensor shape built from it.
    for i in range(len(proto_with_symbolic_values.dim)):
      if proto_with_symbolic_values.dim[i].size < -1:
        proto_with_symbolic_values.dim[i].size = -1
    return tensor_shape.TensorShape(proto_with_symbolic_values)

  def testBatch(self):
    # [FUNCTION DOC]
    # For scalar, vector and matrix inputs, batches the dataset with batch size 42 and expects the
    # inferred shape (after converting symbolic dimensions) to have a compatible unknown first
    # dimension and the original shape in the remaining dimensions.
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([None])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([None, 3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([None, 1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        dataset = dataset.batch(42)
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        inferred_shape = self.as_tensor_shape(
            op_properties['IteratorGetNext'][0].shape)
        self.assertTrue(test_case['shape'].dims[0].is_compatible_with(
            inferred_shape[0]))
        self.assertEqual(test_case['shape'][1:], inferred_shape[1:])

  def testPaddedBatch(self):
    # [FUNCTION DOC]
    # Same as the batch test but uses padded batching with batch size 42 and padded shapes taken
    # from the expected shape without its first dimension (so padded dimensions become 4 or 2 and
    # 4). Expects a compatible unknown first dimension and the padded remaining dimensions.
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([None])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([None, 4])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([None, 2, 4])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        dataset = dataset.padded_batch(42, padded_shapes=test_case['shape'][1:])
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        inferred_shape = self.as_tensor_shape(
            op_properties['IteratorGetNext'][0].shape)
        self.assertTrue(test_case['shape'].dims[0].is_compatible_with(
            inferred_shape[0]))
        self.assertEqual(test_case['shape'][1:], inferred_shape[1:])

  def testFlatMap(self):
    # [FUNCTION DOC]
    # For scalar, vector and matrix values, flat-maps a range dataset of 42 into datasets that
    # repeat the value n times, and expects the inferred get-next shape to equal the value shape.
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.range(42)

        def make_dataset(tensor):
          # [FUNCTION DOC]
          # HELPER: factory for the flat map test. Takes a value and returns a function mapping a
          # count to a dataset repeating that value that many times.

          def dataset_fn(n):
            # [FUNCTION DOC]
            # HELPER: mapping function for the flat map test. Takes a count and returns a
            # dataset of the captured value repeated that many times.
            return dataset_ops.Dataset.from_tensors(tensor).repeat(n)

          return dataset_fn

        dataset = dataset.flat_map(make_dataset(test_case['tensor']))
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testInterleave(self):
    # [FUNCTION DOC]
    # Like the flat map test but interleaves the per-element datasets with a cycle length of 42;
    # expects the inferred get-next shape to equal the value shape.
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.range(42)

        def make_dataset(tensor):
          # [FUNCTION DOC]
          # HELPER: factory for the interleave test. Takes a value and returns a function mapping
          # a count to a dataset repeating that value that many times.

          def dataset_fn(n):
            # [FUNCTION DOC]
            # HELPER: mapping function for the interleave test. Takes a count and returns a
            # dataset of the captured value repeated that many times.
            return dataset_ops.Dataset.from_tensors(tensor).repeat(n)

          return dataset_fn

        dataset = dataset.interleave(
            make_dataset(test_case['tensor']), cycle_length=42)
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testMap(self):
    # [FUNCTION DOC]
    # Maps a transpose over a one-element dataset for scalar, vector, matrix and rank-3 values
    # and expects the inferred get-next shape to be the reversed dimension order.
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([3, 1])
    }, {
        'tensor': np.array([[[1, 2, 3], [4, 5, 6]]]),
        'shape': tensor_shape.TensorShape([3, 2, 1])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        dataset = dataset.map(array_ops.transpose)
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testFromStructure(self):
    # [FUNCTION DOC]
    # For shapes of rank zero to three, creates an iterator from an int64 structure with that
    # output shape and expects the inferred get-next shape to equal it.
    test_cases = [{
        'shape': tensor_shape.TensorShape([])
    }, {
        'shape': tensor_shape.TensorShape([3])
    }, {
        'shape': tensor_shape.TensorShape([1, 2])
    }, {
        'shape': tensor_shape.TensorShape([1, 2, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        iterator = iterator_ops.Iterator.from_structure(
            dtypes.int64, output_shapes=test_case['shape'])
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testFromStringHandle(self):
    # [FUNCTION DOC]
    # For shapes of rank zero to three, creates an int64 iterator, takes its string handle and
    # builds a second iterator from the handle with the given output shape; expects the inferred
    # get-next shape to equal that shape.
    test_cases = [{
        'shape': tensor_shape.TensorShape([])
    }, {
        'shape': tensor_shape.TensorShape([3])
    }, {
        'shape': tensor_shape.TensorShape([1, 2])
    }, {
        'shape': tensor_shape.TensorShape([1, 2, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        iterator = iterator_ops.Iterator.from_structure(dtypes.int64)
        handle = iterator.string_handle()
        iterator = iterator_ops.Iterator.from_string_handle(
            handle, dtypes.int64, output_shapes=test_case['shape'])
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)


if __name__ == '__main__':
  test.main()
