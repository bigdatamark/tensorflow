# -*- coding: utf-8 -*-
# [MODULE DOC]
# Kernel tests for the padded-batch dataset transformation. Covers batching variable-length
# sequences with fixed, unknown (dynamic) and default padded shapes, drop-remainder behavior,
# non-default padding values for nested structures including strings, unicode string data,
# shape specifications given as shape objects, lists or tensors, error cases (short padding,
# sparse or ragged or dataset elements, incompatible or malformed padded shapes, unknown rank),
# low-precision float elements, and the name argument. Also includes a checkpoint test class
# verifying that iterator state saves and restores correctly. No import-time side effects other
# than the test entry point.
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
"""Tests for `tf.data.Dataset.padded_batch()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class PaddedBatchTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the padded-batch transformation: padding shapes and values, error handling
  # for unsupported element types and invalid shapes, and miscellaneous options.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              count=[32, 34],
              padded_shapes=[[None], [25]],
              drop_remainder=[True, False])))
  def testPaddedBatchDataset(self, count, padded_shapes, drop_remainder):
    # [FUNCTION DOC]
    # Scenario: random sequence lengths (count of them, thirty-two or thirty-four) are turned
    # into sequences of that length filled with the length value, then padded-batched with batch
    # size four. Parameters: count, padded_shapes (unknown dimension or fixed twenty-five) and
    # drop_remainder. Expected: full batches have shape batch size by padded length (the batch
    # maximum when dynamic), each row holds its values followed by zero padding; a final
    # partial batch exists only when remainder is kept; further reads raise out-of-range.
    seq_lens = np.random.randint(20, size=(count,)).astype(np.int32)
    batch_size = 4
    dataset = dataset_ops.Dataset.from_tensor_slices(seq_lens).map(
        lambda x: array_ops.fill([x], x)).padded_batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            padded_shapes=padded_shapes)

    num_full_batches = len(seq_lens) // batch_size
    get_next = self.getNext(dataset)
    for i in range(num_full_batches):
      result = self.evaluate(get_next())
      padded_len = padded_shapes[0]
      if padded_len is None or padded_len == -1:
        padded_len = np.max(result) if result.size > 0 else 0
      self.assertEqual((batch_size, padded_len), result.shape)
      for j in range(batch_size):
        seq_len = seq_lens[(i * batch_size) + j]
        self.assertAllEqual(result[j, :seq_len], [seq_len] * seq_len)
        self.assertAllEqual(result[j, seq_len:], [0] * (padded_len - seq_len))

    if not drop_remainder and len(seq_lens) % batch_size > 0:
      result = self.evaluate(get_next())
      padded_len = padded_shapes[0]
      if padded_len is None or padded_len == -1:
        padded_len = np.max(result) if result.size > 0 else 0
      self.assertEqual((len(seq_lens) % batch_size, padded_len), result.shape)
      for j in range(len(seq_lens) % batch_size):
        seq_len = seq_lens[num_full_batches * batch_size + j]
        self.assertAllEqual(result[j, :seq_len], [seq_len] * seq_len)
        self.assertAllEqual(result[j, seq_len:], [0] * (padded_len - seq_len))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchShortPadding(self):
    # [FUNCTION DOC]
    # Scenario: sequences of length six and five padded to a fixed length of five. Expected:
    # iteration fails with a data-loss error because the first sequence is longer than the
    # padded shape.
    dataset = (
        dataset_ops.Dataset.from_tensor_slices(
            [6, 5, 5, 5, 5]).map(lambda x: array_ops.fill([x], x)).padded_batch(
                batch_size=4, padded_shapes=[5]))
    self.assertDatasetProduces(
        dataset, expected_error=(errors.DataLossError, ''))

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchEmptyTensors(self):
    # [FUNCTION DOC]
    # Scenario: four zero-length sequences padded-batched with an unknown padded length.
    # Expected: a single batch of four empty rows.
    dataset = (
        dataset_ops.Dataset.from_tensor_slices(
            [0, 0, 0, 0]).map(lambda x: array_ops.fill([x], x)).padded_batch(
                batch_size=4, padded_shapes=[-1]))
    self.assertDatasetProduces(dataset, expected_output=[[[], [], [], []]])

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchWithDifferetElementTypes(self):
    # [FUNCTION DOC]
    # Scenario: a dataset of pairs of an integer and a string, padded-batched by two. Expected:
    # two batches, each a pair of an integer list and a string list.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        ([0, 1, 2, 3], ['a', 'b', 'c', 'd']))
    dataset = dataset.padded_batch(2)
    self.assertDatasetProduces(dataset, [([0, 1], ['a', 'b']),
                                         ([2, 3], ['c', 'd'])])

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchWithEmptyTuple(self):
    # [FUNCTION DOC]
    # Scenario: elements are pairs of an integer and an empty tuple, padded-batched by two.
    # Expected: batches pair integer lists with empty tuples.
    dataset = dataset_ops.Dataset.from_tensor_slices(([0, 1, 2, 3], ()))
    dataset = dataset.padded_batch(2)
    self.assertDatasetProduces(dataset, [([0, 1], ()), ([2, 3], ())])

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchWithNoneElement(self):
    # [FUNCTION DOC]
    # Scenario: elements are pairs of an integer and a null value. Expected: padded-batching
    # raises a type error.
    dataset = dataset_ops.Dataset.from_tensor_slices(([0, 1, 2, 3], None))
    with self.assertRaises(TypeError):
      dataset = dataset.padded_batch(2)

  @combinations.generate(test_base.default_test_combinations())
  def testDefaultPaddedShapes(self):
    # [FUNCTION DOC]
    # Scenario: sequences of lengths one to four, padded-batched by two without specifying
    # shapes. Expected: each batch is padded to its own longest sequence with zeros.

    def fill(x):
      # [FUNCTION DOC]
      # HELPER: map function creating a vector of length x filled with the value x.
      return array_ops.fill([x], x)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(
            [1, 2, 3, 4]).map(fill).padded_batch(batch_size=2))
    self.assertDatasetProduces(
        dataset,
        expected_output=[[[1, 0], [2, 2]], [[3, 3, 3, 0], [4, 4, 4, 4]]])

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDefaultPaddedShapes(self):
    # [FUNCTION DOC]
    # Scenario: elements are pairs of a scalar and a variable-length vector, padded-batched by
    # two with default shapes. Expected: scalars batch into vectors and vectors are padded with
    # zeros to the longest in each batch.

    def fill_tuple(x):
      # [FUNCTION DOC]
      # HELPER: map function returning a pair of x and a vector of length x filled with x.
      return (x, array_ops.fill([x], x))

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(
            [1, 2, 3, 4]).map(fill_tuple).padded_batch(batch_size=2))
    self.assertDatasetProduces(
        dataset,
        expected_output=[([1, 2], [[1, 0], [2, 2]]),
                         ([3, 4], [[3, 3, 3, 0], [4, 4, 4, 4]])])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              padding_values=[(-1, '<end>', {'structure': ''}),
                              (-1, '<end>', None)])))
  def testPaddedBatchDatasetNonDefaultPadding(self, padding_values):
    # [FUNCTION DOC]
    # Scenario: random-length sequences mapped into a triple of an integer vector, its string
    # form and a mapping with a string vector, padded-batched by four with explicit padded
    # shapes and the given padding values (with or without a value for the mapping entry).
    # Expected: eight batches; integer rows padded with minus one, string rows with the end
    # marker, mapping entries with the empty string; then out-of-range.

    def fill_tuple(x):
      # [FUNCTION DOC]
      # HELPER: map function creating a vector of length x filled with x and returning it with
      # its string conversion and a mapping holding another copy of the string vector.
      filled = array_ops.fill([x], x)
      return (filled, string_ops.as_string(filled), {
          'structure': string_ops.as_string(filled)
      })

    random_seq_lens = np.random.randint(20, size=(32,)).astype(np.int32)
    dataset = (
        dataset_ops.Dataset.from_tensor_slices(random_seq_lens).map(fill_tuple)
        .padded_batch(
            4, padded_shapes=([-1], [-1], {'structure': [-1]}),
            padding_values=padding_values))

    get_next = self.getNext(dataset)
    for i in range(8):
      result = self.evaluate(get_next())
      padded_len = np.max(result[0])
      self.assertEqual((4, padded_len), result[0].shape)
      self.assertEqual((4, padded_len), result[1].shape)
      self.assertEqual((4, padded_len), result[2]['structure'].shape)
      for j in range(4):
        seq_len = random_seq_lens[(i * 4) + j]
        self.assertAllEqual(result[0][j, :seq_len], [seq_len] * seq_len)
        self.assertAllEqual(result[0][j, seq_len:],
                            [-1] * (padded_len - seq_len))
        self.assertAllEqual(result[1][j, :seq_len],
                            [compat.as_bytes(str(seq_len))] * seq_len)
        self.assertAllEqual(result[1][j, seq_len:],
                            [b'<end>'] * (padded_len - seq_len))
        self.assertAllEqual(result[2]['structure'][j, :seq_len],
                            [compat.as_bytes(str(seq_len))] * seq_len)
        self.assertAllEqual(result[2]['structure'][j, seq_len:],
                            [b''] * (padded_len - seq_len))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchDatasetUnicode(self):
    # [FUNCTION DOC]
    # Scenario (regression for a reported issue): a generator dataset yields lists of Russian
    # unicode strings with integer lists, padded-batched by two with empty-string and zero
    # padding. Expected: the first batch can be fetched without error.
    # See GitHub issue 16149
    def generator():
      # [FUNCTION DOC]
      # HELPER: generator for the unicode test; yields pairs of a list of unicode words and the
      # integer list zero to three, one pair per word list.
      data = [[u'Простой', u'тест', u'юникода'],
              [u'никогда', u'не', u'бывает', u'простым']]

      for seq in data:
        yield seq, [0, 1, 2, 3]

    dataset = dataset_ops.Dataset.from_generator(
        generator, (dtypes.string, dtypes.int32),
        (tensor_shape.TensorShape([None]), tensor_shape.TensorShape([None])))
    padded_dataset = dataset.padded_batch(
        2, padded_shapes=([None], [None]), padding_values=('', 0))
    next_element = self.getNext(padded_dataset)
    self.evaluate(next_element())

  @combinations.generate(test_base.graph_only_combinations())
  def testPaddedBatchDatasetShapeSpecifications(self):
    # [FUNCTION DOC]
    # Graph-only scenario: a dataset of placeholders is padded-batched using padded shapes given
    # as shape objects, lists, lists with minus one for unknown, and tensors. Expected: in every
    # case the static output shapes are unknown batch plus unknown dims, with the fixed 37
    # dimension kept for the third component.
    int_placeholder = array_ops.placeholder(dtypes.int32)
    float_placeholder = array_ops.placeholder(dtypes.float32)
    string_placeholder = array_ops.placeholder(dtypes.string)
    input_dataset = dataset_ops.Dataset.from_tensors(
        (int_placeholder, float_placeholder, string_placeholder))

    # Test different ways of specifying the `padded_shapes` argument.
    dynamic_padding_from_tensor_shapes = input_dataset.padded_batch(
        32,
        padded_shapes=(tensor_shape.TensorShape([None]),
                       tensor_shape.TensorShape([None, None]),
                       tensor_shape.TensorShape([37])))
    dynamic_padding_from_lists = input_dataset.padded_batch(
        32, padded_shapes=([None], [None, None], [37]))
    dynamic_padding_from_lists_with_minus_one = input_dataset.padded_batch(
        32, padded_shapes=([-1], [-1, -1], [37]))
    dynamic_padding_from_tensors = input_dataset.padded_batch(
        32,
        padded_shapes=(constant_op.constant([-1], dtype=dtypes.int64),
                       constant_op.constant([-1, -1], dtype=dtypes.int64),
                       constant_op.constant([37], dtype=dtypes.int64)))

    for dataset in [
        dynamic_padding_from_tensor_shapes, dynamic_padding_from_lists,
        dynamic_padding_from_lists_with_minus_one, dynamic_padding_from_tensors
    ]:
      dataset_output_shapes = dataset_ops.get_legacy_output_shapes(dataset)
      self.assertEqual([None, None], dataset_output_shapes[0].as_list())
      self.assertEqual([None, None, None], dataset_output_shapes[1].as_list())
      self.assertEqual([None, 37], dataset_output_shapes[2].as_list())

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchSparseError(self):
    # [FUNCTION DOC]
    # Scenario: padded-batching a dataset of sparse tensors. Expected: a type error saying
    # padded batching only supports tensor elements and naming the sparse type spec.

    st = sparse_tensor.SparseTensorValue(
        indices=[[0, 0]], values=([42]), dense_shape=[1, 1])

    with self.assertRaisesRegex(
        TypeError, r'`padded_batch` is only supported for '
        r'datasets that produce tensor elements but type spec of elements in '
        r'the input dataset is not a subclass of TensorSpec: '
        r'`SparseTensorSpec.*`\.$'):
      _ = dataset_ops.Dataset.from_tensors(st).repeat(10).padded_batch(10)

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchRaggedError(self):
    # [FUNCTION DOC]
    # Scenario: padded-batching a dataset of ragged tensors. Expected: a type error saying
    # padded batching only supports tensor elements and naming the ragged type spec.

    rt = ragged_tensor_value.RaggedTensorValue(
        np.array([0, 42]), np.array([0, 2], dtype=np.int64))

    with self.assertRaisesRegex(
        TypeError, r'`padded_batch` is only supported for '
        r'datasets that produce tensor elements but type spec of elements in '
        r'the input dataset is not a subclass of TensorSpec: '
        r'`RaggedTensorSpec.*`\.$'):
      _ = dataset_ops.Dataset.from_tensors(rt).repeat(10).padded_batch(10)

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchDatasetsError(self):
    # [FUNCTION DOC]
    # Scenario: padded-batching a dataset whose elements are datasets. Expected: a type error
    # saying padded batch is not supported for datasets of datasets.

    ds = dataset_ops.Dataset.range(10).map(
        lambda x: dataset_ops.Dataset.range(1))

    with self.assertRaisesRegex(
        TypeError, r'`padded_batch` is not supported for datasets of datasets'):
      _ = ds.padded_batch(3)

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchShapeErrorWrongRank(self):
    # [FUNCTION DOC]
    # Scenario: scalar elements padded with a rank-one padded shape. Expected: a value error
    # saying the padded shape is not compatible with the scalar component shape.
    with self.assertRaisesRegex(
        ValueError, r'The padded shape \(1,\) is not compatible with the '
        r'shape \(\) of the corresponding input component.'):
      _ = dataset_ops.Dataset.range(10).padded_batch(5, padded_shapes=[1])

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchShapeErrorTooSmall(self):
    # [FUNCTION DOC]
    # Scenario: a length-three vector element padded with a padded shape of length one.
    # Expected: a value error saying the padded shape is not compatible with the component.
    with self.assertRaisesRegex(
        ValueError, r'The padded shape \(1,\) is not compatible with the '
        r'shape \(3,\) of the corresponding input component.'):
      _ = dataset_ops.Dataset.from_tensors([1, 2, 3]).padded_batch(
          5, padded_shapes=[1])

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchShapeErrorShapeNotRank1(self):
    # [FUNCTION DOC]
    # Scenario: a padded shape given as a two-by-two nested list. Expected: a value error saying
    # the padded shape must be a 64-bit integer vector but has shape two by two.
    with self.assertRaisesRegex(
        ValueError, r'Padded shape .* must be a `tf.int64` vector tensor, '
        r'but its shape was \(2, 2\).'):
      _ = dataset_ops.Dataset.from_tensors([1, 2, 3]).padded_batch(
          5, padded_shapes=[[1, 1], [1, 1]])

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchShapeErrorShapeNotInt(self):
    # [FUNCTION DOC]
    # Scenario: a padded shape given as a float tensor. Expected: a type error saying the shape
    # must be a 64-bit integer vector but its element type is 32-bit float.
    with self.assertRaisesRegex(
        TypeError, r'Padded shape .* must be a `tf.int64` vector tensor, '
        r'but its element type was float32.'):
      _ = dataset_ops.Dataset.from_tensors([1, 2, 3]).padded_batch(
          5, padded_shapes=constant_op.constant([1.5, 2., 3.]))

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchShapeErrorWrongRankFromTensor(self):
    # [FUNCTION DOC]
    # Scenario: scalar elements padded with a padded shape supplied as a constant integer
    # tensor of length one. Expected: a value error about incompatibility with the scalar shape.
    with self.assertRaisesRegex(
        ValueError, r'The padded shape \(1,\) is not compatible with the '
        r'shape \(\) of the corresponding input component.'):
      shape_as_tensor = constant_op.constant([1], dtype=dtypes.int64)
      _ = dataset_ops.Dataset.range(10).padded_batch(
          5, padded_shapes=shape_as_tensor)

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchShapeErrorDefaultShapeWithUnknownRank(self):
    # [FUNCTION DOC]
    # Scenario: a generator dataset with unknown element rank padded-batched with default
    # shapes. Expected: a value error mentioning that padded shapes cannot be derived for
    # unknown rank.
    with self.assertRaisesRegex(ValueError, r'`padded_shapes`.*unknown rank'):
      ds = dataset_ops.Dataset.from_generator(
          lambda: iter([1, 2, 3]), output_types=dtypes.int32)
      ds.padded_batch(2)

  @combinations.generate(test_base.graph_only_combinations())
  def testPaddedBatchShapeErrorPlaceholder(self):
    # [FUNCTION DOC]
    # Graph-only scenario: scalar elements padded with a length-two placeholder as padded shape.
    # Expected: a value error saying the padded shape (unknown, unknown) is not compatible with
    # the scalar component.
    with self.assertRaisesRegex(
        ValueError,
        r'The padded shape \((\?|None), (\?|None)\) is not compatible with the '
        r'shape \(\) of the corresponding input component.'):
      shape_as_tensor = array_ops.placeholder(dtypes.int64, shape=[2])
      _ = dataset_ops.Dataset.range(10).padded_batch(
          5, padded_shapes=shape_as_tensor)

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchBfloat16(self):
    # [FUNCTION DOC]
    # Scenario: range values cast to the bfloat16 type and padded-batched by ten. Expected: one
    # batch holding 0.0 through 4.0.
    ds = dataset_ops.Dataset.range(5)
    ds = ds.map(lambda x: math_ops.cast(x, dtypes.bfloat16))
    ds = ds.padded_batch(10)
    self.assertDatasetProduces(
        ds, expected_output=[[0.0, 1.0, 2.0, 3.0, 4.0]])

  @combinations.generate(test_base.default_test_combinations())
  def testDefaultPaddedValueShapes(self):
    # [FUNCTION DOC]
    # Scenario: two zipped datasets of variable-length sequences padded-batched by two with a
    # single scalar padding value of minus one. Expected: the scalar padding is applied to
    # every component of the zipped structure.

    def fill(x):
      # [FUNCTION DOC]
      # HELPER: map function creating a vector of length x filled with the value x.
      return array_ops.fill([x], x)

    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4]).map(fill),
         dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4]).map(fill)))
    dataset = dataset.padded_batch(batch_size=2, padding_values=-1)
    self.assertDatasetProduces(
        dataset,
        expected_output=[([[1, -1], [2, 2]], [[1, -1], [2, 2]]),
                         ([[3, 3, 3, -1], [4, 4, 4, 4]], [[3, 3, 3, -1],
                                                          [4, 4, 4, 4]])])

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: padded batch with an explicit name over a range of five. Expected: one batch
    # holding zero through four.
    dataset = dataset_ops.Dataset.range(5).padded_batch(5, name='padded_batch')
    self.assertDatasetProduces(dataset, [list(range(5))])


class PaddedBatchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint test case for the padded-batch transformation; uses the shared checkpoint test
  # base to verify that an iterator over a padded-batch pipeline can be saved and restored.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Scenario: padded-batch pipeline over thirty-two random-length sequences, batch size four,
    # with symbolic checkpointing on or off. Parameters: verify_fn (the checkpoint verification
    # strategy from the base class) and symbolic_checkpoint flag. Expected: the verification
    # function confirms checkpoint save/restore produces the same eight outputs.

    def build_dataset(seq_lens):
      # [FUNCTION DOC]
      # HELPER: builds the pipeline for the checkpoint test. Maps each length to a vector of
      # that length, padded-batches by four with unknown padded length and sets the symbolic
      # checkpoint option from the enclosing test parameter. Returns the dataset.
      dataset = dataset_ops.Dataset.from_tensor_slices(seq_lens).map(
          lambda x: array_ops.fill([x], x)).padded_batch(
              batch_size=4, padded_shapes=[-1])
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      dataset = dataset.with_options(options)
      return dataset

    seq_lens = np.random.randint(1, 20, size=(32,)).astype(np.int32)
    verify_fn(self, lambda: build_dataset(seq_lens), num_outputs=8)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testNonDefaultPadding(self, verify_fn):
    # [FUNCTION DOC]
    # Scenario: checkpoint test of a padded-batch pipeline with a pair of an integer vector and
    # its string form and explicit padding values (minus one and the end marker). Parameter:
    # verify_fn, the checkpoint verification strategy. Expected: save/restore reproduces the
    # eight output batches.

    def build_dataset(seq_lens):
      # [FUNCTION DOC]
      # HELPER: builds the pipeline for this checkpoint test from a list of lengths; maps to
      # vector/string pairs and padded-batches by four with the explicit padding values.

      def fill_tuple(x):
        # [FUNCTION DOC]
        # HELPER: map function returning a vector of length x filled with x and its string
        # conversion.
        filled = array_ops.fill([x], x)
        return (filled, string_ops.as_string(filled))

      padded_shape = [-1]
      return dataset_ops.Dataset.from_tensor_slices(seq_lens).map(
          fill_tuple).padded_batch(
              batch_size=4,
              padded_shapes=(padded_shape, padded_shape),
              padding_values=(-1, '<end>'))

    seq_lens = np.random.randint(1, 20, size=(32,)).astype(np.int32)
    verify_fn(self, lambda: build_dataset(seq_lens), num_outputs=8)


if __name__ == '__main__':
  test.main()
