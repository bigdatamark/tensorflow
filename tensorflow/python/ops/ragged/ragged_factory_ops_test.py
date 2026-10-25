# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Test module checking that ragged tensors (int64 and string valued, plus dense string tensors for
# comparison) work with map_fn, multi-device iterators and distributed datasets over GPUs, where
# strings must stay on the CPU. Defines three small factory helpers building the shared 8-row
# example data and one parameterized test case class. Many tests require GPUs or TF2 only.
"""Tests that ragged tensors work with GPU, such as placement of int and string.

Test using ragged tensors with map_fn and distributed dataset. Since GPU does
not support strings, ragged tensors containing string should always be placed
on CPU.
"""

from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import map_fn
from tensorflow.python.ops.math_ops import reduce_sum
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.string_ops import string_to_hash_bucket
from tensorflow.python.platform import test


def ragged_int64():
  # [FUNCTION DOC]
  # HELPER: shared test data factory.
  # Returns a ragged constant of int64 values with eight rows (some empty) of up to four numbers.
  return ragged_factory_ops.constant(
      [
          [3, 1, 4, 1],
          [],
          [5, 9, 2],
          [6],
          [],
          [3, 1, 4, 1],
          [3, 1],
          [2, 1, 4, 1],
      ],
      dtype=dtypes.int64,
  )


def ragged_str():
  # [FUNCTION DOC]
  # HELPER: shared test data factory.
  # Returns a ragged constant of strings with the same eight-row layout as the int64 example,
  # each value being the digit as text.
  return ragged_factory_ops.constant([
      ['3', '1', '4', '1'],
      [],
      ['5', '9', '2'],
      ['6'],
      [],
      ['3', '1', '4', '1'],
      ['3', '1'],
      ['2', '1', '4', '1'],
  ])


def dense_str():
  # [FUNCTION DOC]
  # HELPER: shared test data factory.
  # Returns a dense 8 by 4 string tensor with the same content as the ragged string example, with
  # empty strings padding the shorter rows.
  return constant_op.constant([
      ['3', '1', '4', '1'],
      ['', '', '', ''],
      ['5', '9', '2', ''],
      ['6', '', '', ''],
      ['', '', '', ''],
      ['3', '1', '4', '1'],
      ['3', '1', '', ''],
      ['2', '1', '4', '1'],
  ])


class RaggedFactoryOpsTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case verifying ragged tensors with map_fn, multi-device iterators and
  # distributed (mirrored strategy) datasets, using the module's example factories. Several
  # tests are restricted to GPU machines or TF2 behavior via decorators.

  @parameterized.named_parameters(
      ('Int64', ragged_int64,),
      ('Str', ragged_str,),
  )
  def testRaggedWithMapFn(self, ragged_factory):
    # [FUNCTION DOC]
    # Builds the ragged example from the given factory, runs a map_fn with an identity function
    # inside a compiled function, and asserts that the flat values of the result equal those of
    # the input. Returns nothing.

    @def_function.function
    def map_fn_producer(inputs):
      # [FUNCTION DOC]
      # HELPER: compiled wrapper for the test. Applies an identity map over the rows of the
      # ragged input and returns the mapped ragged result.
      return map_fn.map_fn_v2(lambda x: x, inputs)

    t = ragged_factory()
    result = self.evaluate(map_fn_producer(t))
    self.assertAllEqual(t.values, result.values)

  # When drop_remainder=True, the IteratorGetNext op is used and the type
  # of the iterator's output is the type of the dataset elements.
  # When drop_remainder=False the IteratorGetNextAsOptional op is used and
  # the type of the iterator's output is an Optional that contains
  # the type of the dataset elements. Optionals are encoded using variants.
  @parameterized.named_parameters(
      ('Int64Drop', ragged_int64, True),
      ('StrDrop', ragged_str, True),
      ('Int64NoDrop', ragged_int64, False),
      ('StrNoDrop', ragged_str, False),
  )
  def testRaggedWithMultiDeviceIterator(self, ragged_factory, drop_remainder):
    # [FUNCTION DOC]
    # Builds the ragged example, slices it into batches of two (dropping the remainder or not as
    # parameterized), reads it through a multi-device iterator on the first GPU as an optional
    # value, and asserts the first row of the input equals the first row of the fetched batch.
    # Returns nothing.

    @def_function.function
    def dataset_producer(t):
      # [FUNCTION DOC]
      # HELPER: compiled wrapper for the test. Creates a batched dataset from the ragged tensor,
      # a multi-device iterator over it on the first GPU, and returns the next element as an
      # optional placed on that GPU.
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(
          2, drop_remainder)
      it = multi_device_iterator_ops.MultiDeviceIterator(ragged_ds, ['GPU:0'])
      with ops.device_v2('GPU:0'):
        return it.get_next_as_optional()

    t = ragged_factory()
    result = dataset_producer(t)
    self.assertAllEqual(
        self.evaluate(t[0]), self.evaluate(result[0].get_value()[0]))

  @parameterized.named_parameters(
      ('Int65Drop', ragged_int64, True),
      ('StrDrop', ragged_str, True),
      ('Int65NoDrop', ragged_int64, False),
      ('StrNoDrop', ragged_str, False),
  )
  @test_util.run_gpu_only
  def testRaggedWithDistributedDatasetIter(self, ragged_factory,
                                           drop_remainder):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the drop_remainder parameter is never used (batching ignores it), so the
    # drop and no-drop variants run identically; the case names also misspell the int64 label.
    # GPU-only test. Runs a compiled function that distributes a batched ragged dataset over two
    # GPUs with a mirrored strategy, iterates it and returns the per-replica results for four
    # steps. Asserts that each of the eight input rows equals the matching replica's first element
    # in order. Returns nothing.

    @def_function.function
    def distributed_dataset_producer(t):
      # [FUNCTION DOC]
      # HELPER: compiled wrapper for the test. Creates a mirrored strategy on two GPUs, batches
      # the ragged tensor by two, distributes it, takes four elements as per-replica local
      # results, drains the rest of the iterator, and returns the four results.
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)
      ds = iter(dist_dataset)
      result0 = strategy.experimental_local_results(next(ds))
      result1 = strategy.experimental_local_results(next(ds))
      result2 = strategy.experimental_local_results(next(ds))
      result3 = strategy.experimental_local_results(next(ds))
      # Reach the end of the iterator
      for ignore in ds:  # pylint: disable=unused-variable
        pass
      return result0, result1, result2, result3

    t = ragged_factory()

    result0, result1, result2, result3 = distributed_dataset_producer(t)
    self.assertAllEqual(self.evaluate(t[0]), self.evaluate(result0[0][0]))
    self.assertAllEqual(self.evaluate(t[1]), self.evaluate(result0[1][0]))
    self.assertAllEqual(self.evaluate(t[2]), self.evaluate(result1[0][0]))
    self.assertAllEqual(self.evaluate(t[3]), self.evaluate(result1[1][0]))
    self.assertAllEqual(self.evaluate(t[4]), self.evaluate(result2[0][0]))
    self.assertAllEqual(self.evaluate(t[5]), self.evaluate(result2[1][0]))
    self.assertAllEqual(self.evaluate(t[6]), self.evaluate(result3[0][0]))
    self.assertAllEqual(self.evaluate(t[7]), self.evaluate(result3[1][0]))

  @parameterized.named_parameters(
      ('Int65Drop', ragged_int64, True),
      ('StrDrop', ragged_str, True),
      ('Int65NoDrop', ragged_int64, False),
      ('StrNoDrop', ragged_str, False),
  )
  @test_util.run_v2_only
  def testRaggedWithDistributedDatasetReplicaFn(self, ragged_factory,
                                                drop_remainder):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the drop_remainder parameter is never used, so parameterized variants are
    # identical; the case names also misspell the int64 label.
    # TF2-only test. Distributes a batched ragged dataset over two GPUs, runs an identity
    # replica function on every batch with the strategy, and asserts that each input row equals
    # the first element of the matching replica's values. Returns nothing.

    def distributed_dataset_producer(t):
      # [FUNCTION DOC]
      # HELPER: test driver. Builds a two-GPU mirrored strategy and distributed batched dataset
      # from the ragged tensor, runs a replica function through the strategy for each batch and
      # returns the list of results.
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)

      @def_function.function
      def replica_fn(elem):
        # [FUNCTION DOC]
        # HELPER: per-replica function for the test. Returns its input element unchanged.
        return elem

      result = []
      for x in dist_dataset:
        result.append(strategy.run(replica_fn, args=(x,)))
      return result

    t = ragged_factory()
    result = distributed_dataset_producer(t)

    self.assertAllEqual(
        self.evaluate(t[0]), self.evaluate(result[0].values[0][0]))
    self.assertAllEqual(
        self.evaluate(t[1]), self.evaluate(result[0].values[1][0]))
    self.assertAllEqual(
        self.evaluate(t[2]), self.evaluate(result[1].values[0][0]))
    self.assertAllEqual(
        self.evaluate(t[3]), self.evaluate(result[1].values[1][0]))
    self.assertAllEqual(
        self.evaluate(t[4]), self.evaluate(result[2].values[0][0]))
    self.assertAllEqual(
        self.evaluate(t[5]), self.evaluate(result[2].values[1][0]))
    self.assertAllEqual(
        self.evaluate(t[6]), self.evaluate(result[3].values[0][0]))
    self.assertAllEqual(
        self.evaluate(t[7]), self.evaluate(result[3].values[1][0]))

  @parameterized.named_parameters(
      ('DenseDrop', dense_str, True),
      ('RaggedDrop', ragged_str, True),
      ('DenseNoDrop', dense_str, False),
      ('RaggedNoDrop', ragged_str, False),
  )
  @test_util.run_gpu_only
  def testIntStringWithDistributedDataset(self, string_factory, drop_remainder):
    # [FUNCTION DOC]
    # GPU-only test. Packs a ragged int64 tensor and a string tensor (dense or ragged) into a
    # mapping, distributes it batched by two (honoring drop_remainder) over two GPUs, and
    # asserts that for each of the eight rows both the integer and string entries equal the
    # corresponding replica result. Verifies strings stay on CPU while ints may use GPU.
    # Returns nothing.

    @def_function.function
    def distributed_dataset_producer(t):
      # [FUNCTION DOC]
      # HELPER: compiled wrapper for the test. Creates a two-GPU mirrored strategy, distributes
      # the batched dataset made from the mapping, collects four local results, drains the
      # iterator and returns the four results.
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(
          2, drop_remainder)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)
      ds = iter(dist_dataset)
      result0 = strategy.experimental_local_results(next(ds))
      result1 = strategy.experimental_local_results(next(ds))
      result2 = strategy.experimental_local_results(next(ds))
      result3 = strategy.experimental_local_results(next(ds))
      # Reach the end of the iterator
      for ignore in ds:  # pylint: disable=unused-variable
        pass
      return result0, result1, result2, result3

    ds_dict = {'int': ragged_int64(), 'str': string_factory()}
    result0, result1, result2, result3 = distributed_dataset_producer(ds_dict)
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][0]), self.evaluate(result0[0]['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][0]), self.evaluate(result0[0]['str'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][1]), self.evaluate(result0[1]['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][1]), self.evaluate(result0[1]['str'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][2]), self.evaluate(result1[0]['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][2]), self.evaluate(result1[0]['str'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][3]), self.evaluate(result1[1]['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][3]), self.evaluate(result1[1]['str'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][4]), self.evaluate(result2[0]['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][4]), self.evaluate(result2[0]['str'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][5]), self.evaluate(result2[1]['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][5]), self.evaluate(result2[1]['str'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][6]), self.evaluate(result3[0]['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][6]), self.evaluate(result3[0]['str'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][7]), self.evaluate(result3[1]['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][7]), self.evaluate(result3[1]['str'][0]))

  @parameterized.named_parameters(
      ('DenseDrop', dense_str, True),
      ('RaggedDrop', ragged_str, True),
      ('DenseNoDrop', dense_str, False),
      ('RaggedNoDrop', ragged_str, True),
  )
  @test_util.run_v2_only
  def testOpsWithDistributedDataset(self, string_factory, drop_remainder):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the case named RaggedNoDrop passes True for drop_remainder, so it duplicates
    # the RaggedDrop case instead of testing the no-drop path.
    # TF2-only test. Distributes a mapping holding a dense or ragged string tensor over two GPUs,
    # runs a replica function that hashes the strings into ten buckets and multiplies by 1000,
    # and asserts the per-replica values equal the expected numbers truncated to each row's
    # length. Returns nothing.

    def distributed_dataset_producer(t):
      # [FUNCTION DOC]
      # HELPER: test driver. Builds the two-GPU strategy and distributed batched dataset from the
      # mapping, runs the replica function for each batch and returns the list of results.
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(
          2, drop_remainder)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)

      @def_function.function
      def replica_fn(elem):
        # [FUNCTION DOC]
        # HELPER: per-replica function for the test. Hashes the string feature into ten buckets
        # and returns the bucket ids multiplied by 1000.
        # Example of typical preprocessing of string to numeric feature
        hashed = string_to_hash_bucket(elem['str'], 10)
        return 1000 * hashed

      result = []
      for x in dist_dataset:
        result.append(strategy.run(replica_fn, args=(x,)))
      return result

    ds_dict = {'str': string_factory()}
    result = distributed_dataset_producer(ds_dict)
    expect_length = [len(i) for i in ds_dict['str']]
    self.assertAllEqual([[5000, 3000, 5000, 3000][:expect_length[0]]],
                        self.evaluate(result[0].values[0]))
    self.assertAllEqual([[9000, 9000, 9000, 9000][:expect_length[1]]],
                        self.evaluate(result[0].values[1]))
    self.assertAllEqual([[0, 3000, 2000, 9000][:expect_length[2]]],
                        self.evaluate(result[1].values[0]))
    self.assertAllEqual([[2000, 9000, 9000, 9000][:expect_length[3]]],
                        self.evaluate(result[1].values[1]))
    self.assertAllEqual([[9000, 9000, 9000, 9000][:expect_length[4]]],
                        self.evaluate(result[2].values[0]))
    self.assertAllEqual([[5000, 3000, 5000, 3000][:expect_length[5]]],
                        self.evaluate(result[2].values[1]))
    self.assertAllEqual([[5000, 3000, 9000, 9000][:expect_length[6]]],
                        self.evaluate(result[3].values[0]))
    self.assertAllEqual([[2000, 3000, 5000, 3000][:expect_length[7]]],
                        self.evaluate(result[3].values[1]))

  @parameterized.named_parameters(
      ('DenseDrop', dense_str, True),
      ('RaggedDrop', ragged_str, True),
      ('DenseNoDrop', dense_str, False),
      ('RaggedNoDrop', ragged_str, False),
  )
  @test_util.run_v2_only
  def testIntStringOpsWithDistributedDataset(self, string_factory,
                                             drop_remainder):
    # [FUNCTION DOC]
    # TF2-only test. Builds a mapping with the ragged int64 example, a dense or ragged string
    # tensor and the row sizes, distributes it batched by two (honoring drop_remainder) over two
    # GPUs, and runs a replica function combining the hashed strings with the integers. Asserts
    # the eight per-replica scalar sums equal the expected numbers. Returns nothing.

    ri = ragged_int64()
    # To ease testing both dense and ragged strings, pass in the ragged sizes
    # so the dense strings can be sliced to match.
    element_sizes = [len(i) for i in ri]
    ds_dict = {'int': ri, 'str': string_factory(), 'size': element_sizes}

    def distributed_dataset_producer(t):
      # [FUNCTION DOC]
      # HELPER: test driver. Builds the two-GPU strategy and distributed batched dataset from the
      # mapping, runs the replica function for each batch and returns the list of results.
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(
          2, drop_remainder)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)

      @def_function.function
      def replica_fn(elem):
        # [FUNCTION DOC]
        # HELPER: per-replica function for the test. Hashes the string feature into ten buckets,
        # slices it to the first row's true size (so dense strings match ragged ones), and
        # returns the sum of those buckets plus one hundred times the sum of the integer feature.
        # Example of typical preprocessing of string to numeric feature
        hashed = string_to_hash_bucket(elem['str'], 10)
        # For dense string case, slice it to size of ragged int
        hashed_sliced = hashed[:, :elem['size'][0]]
        # Computation with both feature from string and numeric dataset output
        return reduce_sum(hashed_sliced) + 100 * reduce_sum(elem['int'])

      result = []
      for x in dist_dataset:
        result.append(strategy.run(replica_fn, args=(x,)))
      return result

    result = distributed_dataset_producer(ds_dict)
    self.assertAllEqual(916, self.evaluate(result[0].values[0]))
    self.assertAllEqual(0, self.evaluate(result[0].values[1]))
    self.assertAllEqual(1605, self.evaluate(result[1].values[0]))
    self.assertAllEqual(602, self.evaluate(result[1].values[1]))
    self.assertAllEqual(0, self.evaluate(result[2].values[0]))
    self.assertAllEqual(916, self.evaluate(result[2].values[1]))
    self.assertAllEqual(408, self.evaluate(result[3].values[0]))
    self.assertAllEqual(813, self.evaluate(result[3].values[1]))


if __name__ == '__main__':
  test.main()
