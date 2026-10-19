# [MODULE DOC]
# Unit tests for the dataset bucket-by-sequence-length transformation, which groups variable-length
# sequences into buckets by length and batches each bucket. Covers dropping partial batches,
# regular bucketing with padding or sparse output, padding up to the bucket boundary (and error on
# over-long sequences), tuple elements, sparse bucketing with expected batch contents, and the
# infinite cardinality of the result. Also defines small module-level helpers that build records,
# types and shapes in dense or sparse form for the generator-based datasets.
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
"""Tests for `tf.data.Dataset.bucket_by_sequence_length()."""
import random

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def _element_length_fn(x, y=None):
  # [FUNCTION DOC]
  # HELPER: length function passed to the bucketing transformation by the tests.
  # Inputs: x (a sequence tensor or sparse dense-shape-bearing value) and y (optional, ignored, so
  # that tuple elements with a label also work). Returns the size of the first dimension of x.
  del y
  return array_ops.shape(x)[0]


def _to_sparse_tensor(record):
  # [FUNCTION DOC]
  # HELPER: converts a record mapping (values, indices, dense shape) into a sparse tensor.
  # Input: record, mapping with those three entries. Returns the sparse tensor.
  return sparse_tensor.SparseTensor(**record)


def _format_record(array, sparse):
  # [FUNCTION DOC]
  # HELPER: shapes a list of numbers for the generator datasets.
  # Inputs: array (list of numbers) and sparse (boolean). When sparse, returns a mapping with the
  # values, the index list (one index per value) and the dense shape (a one-element tuple holding
  # the length); otherwise returns the list unchanged.
  if sparse:
    return {
        "values": array,
        "indices": [[i] for i in range(len(array))],
        "dense_shape": (len(array),)
    }
  return array


def _get_record_type(sparse):
  # [FUNCTION DOC]
  # HELPER: describes the element data type for generator datasets. Input sparse (boolean).
  # Returns a mapping of three 64-bit integer types (values, indices, dense shape) when sparse,
  # otherwise the 32-bit integer type.
  if sparse:
    return {
        "values": dtypes.int64,
        "indices": dtypes.int64,
        "dense_shape": dtypes.int64
    }
  return dtypes.int32


def _get_record_shape(sparse):
  # [FUNCTION DOC]
  # HELPER: describes the element shape for generator datasets. Input sparse (boolean).
  # Returns a mapping of tensor shapes (values: unknown length; indices: unknown by one;
  # dense shape: length one) when sparse, otherwise a rank-one shape of unknown length.
  if sparse:
    return {
        "values": tensor_shape.TensorShape([
            None,
        ]),
        "indices": tensor_shape.TensorShape([None, 1]),
        "dense_shape": tensor_shape.TensorShape([
            1,
        ])
    }
  return tensor_shape.TensorShape([None])


class BucketBySequenceLengthTest(test_base.DatasetTestBase,
                                 parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for bucket_by_sequence_length. Each test builds a generator-based
  # dataset of sequences of different lengths and inspects the resulting batches.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(param_no_padding=[True, False])))
  def testBucketDropReminder(self, param_no_padding):
    # [FUNCTION DOC]
    # Parameter: param_no_padding (true selects sparse output without padding, false dense padded).
    # Sets up four length buckets with specified element counts per bucket, computes expected total
    # sums, batch sizes and sequence lengths when partial batches are dropped, then builds the
    # dataset, reads the five expected batches, asserts the next read is out of range, and compares
    # the generated sums, batch sizes and lengths per bucket with the expectations.
    # KNOWN ISSUE: the test name misspells remainder as reminder; behavior is unaffected.

    boundaries = [10, 20, 30]
    batch_sizes = [10, 8, 4, 2]
    lengths = [8, 13, 25, 35]

    n_bucket_elements = [28, 7, 6, 5]
    n_expected_batches = 5

    # Expected sequence lengths of the individual batches.
    expected_lengths = []

    # Expected sum of all batches with an equal sequence length.
    # <seq-length>: <expected-total-sum>
    expected_sums = {}

    # Expected batch sizes of batches depending on the sequence length.
    # <seq-length>: [batch1_size, ..., batchN_size]
    expected_batch_sizes = {}

    for length, batch_size, bucket_elements in zip(lengths, batch_sizes,
                                                   n_bucket_elements):
      # Calculate the expected sum across all batches of a specific sequence
      # length.
      expected_sums[length] = \
          (bucket_elements - bucket_elements % batch_size) * length
      # Calculate the expected occurrence of individual batch sizes.
      expected_batch_sizes[length] = \
          [batch_size] * (bucket_elements // batch_size)
      # Calculate the expected occurrence of individual sequence lengths.
      expected_lengths.extend([length] * (bucket_elements // batch_size))

    def build_dataset(sparse):
      # [FUNCTION DOC]
      # HELPER: builds the generator dataset of the test. Input sparse (boolean). Elements are
      # all-ones sequences with the full bucket length, as many per bucket as configured, shuffled;
      # sparse input is converted to sparse tensors by a map. Returns the dataset.

      def _generator():
        # [FUNCTION DOC]
        # HELPER: generator of the dataset; yields one-element tuples holding the shuffled
        # sequences, formatted as dense or sparse records. Uses randomness.
        # Produce 1 batch for each bucket
        elements = []
        for bucket_elements, length in zip(n_bucket_elements, lengths):
          # Using only full sequences (opposed to the strategy employed in
          # `testBucket`) makes checking the sum a lot easier.
          record_len = length
          for _ in range(bucket_elements):
            elements.append([1] * record_len)
        random.shuffle(elements)
        for el in elements:
          yield (_format_record(el, sparse),)

      dataset = dataset_ops.Dataset.from_generator(_generator,
                                                   (_get_record_type(sparse),),
                                                   (_get_record_shape(sparse),))
      if sparse:
        dataset = dataset.map(lambda x: (_to_sparse_tensor(x),))
      return dataset

    def _test_bucket_by_padding(no_padding):
      # [FUNCTION DOC]
      # HELPER: runs the check described above. Input no_padding (boolean, also selects sparse
      # data). Applies the bucketing with dropped remainders, collects the batches and asserts
      # per-length sums, batch sizes and lengths match the expectations.
      dataset = build_dataset(sparse=no_padding)
      dataset = dataset.bucket_by_sequence_length(
          element_length_func=_element_length_fn,
          bucket_boundaries=boundaries,
          bucket_batch_sizes=batch_sizes,
          no_padding=no_padding,
          drop_remainder=True)

      get_next = self.getNext(dataset)
      batches = []
      for _ in range(n_expected_batches):
        batch, = self.evaluate(get_next())
        batches.append(batch)

      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

      generated_lengths = []

      # <seq-length>: <total-sum>
      generated_sums = {}

      # <seq-length>: [<batch_size>, ...]
      generated_batch_sizes = {}

      for length, batch_size, bucket_elements in zip(lengths, batch_sizes,
                                                     n_bucket_elements):
        # Initialize the sum across all batches.
        generated_sums[length] = 0
        # Initialize the individual batch sizes.
        generated_batch_sizes[length] = []

      for batch in batches:
        shape = batch.dense_shape if no_padding else batch.shape
        length = shape[1]
        generated_lengths.append(length)

        batch_size = shape[0]
        generated_batch_sizes[length].append(batch_size)

        batch_sum = batch.values.sum() if no_padding else batch.sum()
        generated_sums[length] += batch_sum

      for l in lengths:
        # Make sure the sum of the batch contents is correct for the individual
        # sequence lengths.
        self.assertEqual(
            generated_sums[l], expected_sums[l], "Tensor sums did not match! "
            "expected: {}, generated: {}".format(expected_sums, generated_sums))

        # Make sure the individual batch sizes are generated as expected.
        self.assertEqual(
            sorted(generated_batch_sizes[l]), sorted(expected_batch_sizes[l]),
            "Batch-sizes did not match! "
            "expected: {}, generated: {}".format(
                sorted(expected_batch_sizes[l]),
                sorted(generated_batch_sizes[l])))

      # Make sure the generated sequence lengths appear as often as expected.
      self.assertEqual(
          sorted(generated_lengths), sorted(expected_lengths),
          "The generated sequence lengths did not match! "
          "expected: {}, generated: {}".format(
              sorted(expected_lengths), sorted(generated_lengths)))

    _test_bucket_by_padding(param_no_padding)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(param_no_padding=[True, False])))
  def testBucket(self, param_no_padding):
    # [FUNCTION DOC]
    # Parameter: param_no_padding (true selects sparse output without padding).
    # Builds a dataset with one full batch worth of elements per bucket (the first element of each
    # bucket one shorter than the others), buckets it, reads four batches, asserts the next read is
    # out of range, and checks the batch sizes and lengths match the configuration. In graph mode
    # also checks the sum of each batch.

    boundaries = [10, 20, 30]
    batch_sizes = [10, 8, 4, 2]
    lengths = [8, 13, 25, 35]

    def build_dataset(sparse):
      # [FUNCTION DOC]
      # HELPER: builds the generator dataset of the test. Input sparse (boolean). For each bucket
      # produces batch-size sequences of all ones, shuffled; sparse input is converted by a map.
      # Returns the dataset.

      def _generator():
        # [FUNCTION DOC]
        # HELPER: generator of the dataset; yields one-element tuples holding shuffled sequences,
        # formatted as dense or sparse records. Uses randomness.
        # Produce 1 batch for each bucket
        elements = []
        for batch_size, length in zip(batch_sizes, lengths):
          record_len = length - 1
          for _ in range(batch_size):
            elements.append([1] * record_len)
            record_len = length
        random.shuffle(elements)
        for el in elements:
          yield (_format_record(el, sparse),)

      dataset = dataset_ops.Dataset.from_generator(_generator,
                                                   (_get_record_type(sparse),),
                                                   (_get_record_shape(sparse),))
      if sparse:
        dataset = dataset.map(lambda x: (_to_sparse_tensor(x),))
      return dataset

    def _test_bucket_by_padding(no_padding):
      # [FUNCTION DOC]
      # HELPER: runs the check described above. Input no_padding (boolean, also selects sparse
      # data). Applies the bucketing, reads four batches, asserts end of data and compares batch
      # sizes and lengths (and sums in graph mode).
      dataset = build_dataset(sparse=no_padding)
      dataset = dataset.bucket_by_sequence_length(
          element_length_func=_element_length_fn,
          bucket_boundaries=boundaries,
          bucket_batch_sizes=batch_sizes,
          no_padding=no_padding)
      get_next = self.getNext(dataset)
      batches = []
      for _ in range(4):
        batch, = self.evaluate(get_next())
        batches.append(batch)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

      batch_sizes_val = []
      lengths_val = []
      for batch in batches:
        shape = batch.dense_shape if no_padding else batch.shape
        batch_size = shape[0]
        length = shape[1]
        batch_sizes_val.append(batch_size)
        lengths_val.append(length)
        if not context.executing_eagerly():
          sum_check = batch.values.sum() if no_padding else batch.sum()
          self.assertEqual(sum_check, batch_size * length - 1)
      self.assertEqual(sum(batch_sizes_val), sum(batch_sizes))
      self.assertEqual(sorted(batch_sizes), sorted(batch_sizes_val))
      self.assertEqual(sorted(lengths), sorted(lengths_val))

    _test_bucket_by_padding(param_no_padding)

  @combinations.generate(test_base.default_test_combinations())
  def testPadToBoundary(self):
    # [FUNCTION DOC]
    # Builds sequences of lengths eight, thirteen and twenty-five plus several longer than the
    # largest boundary, buckets with padding to the bucket boundary, reads three batches and asserts
    # that the fourth read fails with an error mentioning bucket boundaries. Asserts batch sizes
    # match and that batch lengths equal each boundary minus one.

    boundaries = [10, 20, 30]
    batch_sizes = [10, 8, 4, 2]
    lengths = [8, 13, 25]

    def element_gen():
      # [FUNCTION DOC]
      # HELPER: generator of the dataset; yields shuffled all-ones sequences for the first three
      # buckets, then several sequences longer than the largest boundary.
      # Produce 1 batch for each bucket
      elements = []
      for batch_size, length in zip(batch_sizes[:-1], lengths):
        for _ in range(batch_size):
          elements.append([1] * length)
      random.shuffle(elements)
      for el in elements:
        yield (el,)
      for _ in range(batch_sizes[-1]):
        el = [1] * (boundaries[-1] + 5)
        yield (el,)

    element_len = lambda el: array_ops.shape(el)[0]
    dataset = dataset_ops.Dataset.from_generator(element_gen, (dtypes.int64,),
                                                 ([None],))
    dataset = dataset.bucket_by_sequence_length(
        element_length_func=element_len,
        bucket_boundaries=boundaries,
        bucket_batch_sizes=batch_sizes,
        pad_to_bucket_boundary=True)
    get_next = self.getNext(dataset)

    batches = []
    for _ in range(3):
      batch, = self.evaluate(get_next())
      batches.append(batch)
    with self.assertRaisesOpError("bucket_boundaries"):
      self.evaluate(get_next())

    batch_sizes_val = []
    lengths_val = []
    for batch in batches:
      batch_size = batch.shape[0]
      length = batch.shape[1]
      batch_sizes_val.append(batch_size)
      lengths_val.append(length)
    batch_sizes = batch_sizes[:-1]
    self.assertEqual(sum(batch_sizes_val), sum(batch_sizes))
    self.assertEqual(sorted(batch_sizes), sorted(batch_sizes_val))
    self.assertEqual([boundary - 1 for boundary in sorted(boundaries)],
                     sorted(lengths_val))

  @combinations.generate(test_base.default_test_combinations())
  def testPadToBoundaryNoExtraneousPadding(self):
    # [FUNCTION DOC]
    # Buckets all-ones sequences of lengths one to ten with padding to the bucket boundary and
    # asserts five batches whose padding (zeros) reaches only the needed length, then end of data.

    boundaries = [3, 7, 11]
    batch_sizes = [2, 2, 2, 2]
    lengths = range(1, 11)

    def element_gen():
      # [FUNCTION DOC]
      # HELPER: generator of the dataset; yields one-element tuples of all-ones sequences with
      # lengths one to ten.
      for length in lengths:
        yield ([1] * length,)

    element_len = lambda element: array_ops.shape(element)[0]
    dataset = dataset_ops.Dataset.from_generator(element_gen, (dtypes.int64,),
                                                 ([None],))
    dataset = dataset.bucket_by_sequence_length(
        element_length_func=element_len,
        bucket_boundaries=boundaries,
        bucket_batch_sizes=batch_sizes,
        pad_to_bucket_boundary=True)
    get_next = self.getNext(dataset)

    batches = []
    for _ in range(5):
      batch, = self.evaluate(get_next())
      batches.append(batch)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    self.assertAllEqual(batches[0], [[1, 0], [1, 1]])
    self.assertAllEqual(batches[1], [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    self.assertAllEqual(batches[2], [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    self.assertAllEqual(
        batches[3],
        [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    self.assertAllEqual(
        batches[4],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(param_no_padding=[True, False])))
  def testTupleElements(self, param_no_padding):
    # [FUNCTION DOC]
    # Parameter: param_no_padding (true selects sparse sequences).
    # Builds a dataset of (sequence, label) pairs, applies the bucketing, and asserts the static
    # output shapes: the sequence component has two unknown dimensions and the label has one.

    def build_dataset(sparse):
      # [FUNCTION DOC]
      # HELPER: builds the dataset of text-like sequences with integer labels from a generator;
      # sparse sequences are converted to sparse tensors by a map. Input sparse (boolean).

      def _generator():
        # [FUNCTION DOC]
        # HELPER: generator yielding pairs of a formatted sequence and its label for four fixed
        # examples.
        text = [[1, 2, 3], [3, 4, 5, 6, 7], [1, 2], [8, 9, 0, 2, 3]]
        label = [1, 2, 1, 2]
        for x, y in zip(text, label):
          yield (_format_record(x, sparse), y)

      dataset = dataset_ops.Dataset.from_generator(
          generator=_generator,
          output_types=(_get_record_type(sparse), dtypes.int32),
          output_shapes=(_get_record_shape(sparse),
                         tensor_shape.TensorShape([])))
      if sparse:
        dataset = dataset.map(lambda x, y: (_to_sparse_tensor(x), y))
      return dataset

    def _test_tuple_elements_by_padding(no_padding):
      # [FUNCTION DOC]
      # HELPER: runs the shape check described above. Input no_padding (boolean).
      dataset = build_dataset(sparse=no_padding)
      dataset = dataset.bucket_by_sequence_length(
          element_length_func=_element_length_fn,
          bucket_batch_sizes=[2, 2, 2],
          bucket_boundaries=[0, 8],
          no_padding=no_padding)
      shapes = dataset_ops.get_legacy_output_shapes(dataset)
      self.assertEqual([None, None], shapes[0].as_list())
      self.assertEqual([None], shapes[1].as_list())

    _test_tuple_elements_by_padding(param_no_padding)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(param_drop_remainder=[True, False])))
  def testBucketSparse(self, param_drop_remainder):  # pylint: disable=g-doc-args
    # [FUNCTION DOC]
    # Parameter: param_drop_remainder (boolean).
    # Builds a dataset of one hundred sparse sequences of growing length (0 up to 99), buckets
    # them with no padding in groups of ten lengths and batches of seven, with or without dropping
    # remainders. Collects all produced batches into a set of index and value tuples and asserts it
    # equals the set of expected batches computed independently.
    """Tests bucketing of sparse tensors (case where `no_padding` == True).

    Test runs on following dataset:
      [
        [0],
        [0, 1],
        [0, 1, 2]
        ...
        [0, ..., max_len - 1]
      ]
    Sequences are bucketed by length and batched with
      `batch_size` < `bucket_size`.
    """

    min_len = 0
    max_len = 100
    batch_size = 7
    bucket_size = 10

    def _build_dataset():
      # [FUNCTION DOC]
      # HELPER: builds the sparse sequence dataset from a generator and a map to sparse tensors.
      # Returns the dataset.
      input_data = [range(i + 1) for i in range(min_len, max_len)]

      def generator_fn():
        # [FUNCTION DOC]
        # HELPER: generator yielding each input sequence as a sparse-format record.
        for record in input_data:
          yield _format_record(record, sparse=True)

      dataset = dataset_ops.Dataset.from_generator(
          generator=generator_fn, output_types=_get_record_type(sparse=True))
      dataset = dataset.map(_to_sparse_tensor)
      return dataset

    def _compute_expected_batches(drop_remainder):
      # [FUNCTION DOC]
      # HELPER: computes the expected batches. Input drop_remainder (boolean). For every bucket,
      # derives each batch from its start and end lengths (one batch only when dropping
      # remainders, else several) and returns a set of pairs of index tuples and value tuples.
      """Computes expected batch outputs and stores in a set."""
      all_expected_sparse_tensors = set()
      for bucket_start_len in range(min_len, max_len, bucket_size):
        if drop_remainder:
          batch_offsets = [0]
        else:
          batch_offsets = range(0, bucket_size, batch_size)

        for batch_offset in batch_offsets:
          batch_start_len = bucket_start_len + batch_offset
          batch_end_len = min(batch_start_len + batch_size,
                              bucket_start_len + bucket_size)
          expected_indices = []
          expected_values = []
          for length in range(batch_start_len, batch_end_len):
            for val in range(length + 1):
              expected_indices.append((length - batch_start_len, val))
              expected_values.append(val)
          expected_sprs_tensor = (tuple(expected_indices),
                                  tuple(expected_values))
          all_expected_sparse_tensors.add(expected_sprs_tensor)
      return all_expected_sparse_tensors

    def _compute_batches(dataset):
      # [FUNCTION DOC]
      # HELPER: reads the dataset until the end, asserting the end is signaled by an out-of-range
      # error. Input dataset. Returns the set of produced batches as pairs of index tuples and
      # value tuples.
      """Computes actual batch outputs of dataset and stores in a set."""
      batch = self.getNext(dataset)
      all_sparse_tensors = set()
      with self.assertRaises(errors.OutOfRangeError):
        while True:
          output = self.evaluate(batch())
          sprs_tensor = (tuple([tuple(idx) for idx in output.indices]),
                         tuple(output.values))
          all_sparse_tensors.add(sprs_tensor)

      return all_sparse_tensors

    dataset = _build_dataset()
    boundaries = range(min_len + bucket_size + 1, max_len, bucket_size)
    dataset = dataset.bucket_by_sequence_length(
        element_length_func=_element_length_fn,
        bucket_boundaries=boundaries,
        bucket_batch_sizes=[batch_size] * (len(boundaries) + 1),
        no_padding=True,
        drop_remainder=param_drop_remainder)
    batches = _compute_batches(dataset)
    expected_batches = _compute_expected_batches(param_drop_remainder)
    self.assertEqual(batches, expected_batches)

  @combinations.generate(test_base.default_test_combinations())
  def testCardinality(self):
    # [FUNCTION DOC]
    # Repeats a generator dataset forever, buckets it with padding to the boundary and asserts the
    # resulting dataset cardinality is infinite.

    boundaries = [3, 7, 11]
    batch_sizes = [2, 2, 2, 2]
    lengths = range(1, 11)

    def element_gen():
      # [FUNCTION DOC]
      # HELPER: generator of the dataset; yields one-element tuples of all-ones sequences with
      # lengths one to ten.
      for length in lengths:
        yield ([1] * length,)

    element_len = lambda element: array_ops.shape(element)[0]
    dataset = dataset_ops.Dataset.from_generator(element_gen, (dtypes.int64,),
                                                 ([None],)).repeat()
    dataset = dataset.bucket_by_sequence_length(
        element_length_func=element_len,
        bucket_boundaries=boundaries,
        bucket_batch_sizes=batch_sizes,
        pad_to_bucket_boundary=True)
    self.assertEqual(self.evaluate(dataset.cardinality()), dataset_ops.INFINITE)


if __name__ == "__main__":
  test.main()
