# [MODULE DOC]
# Unit tests for the dataset unbatch transformation, which splits each batched element along its
# first dimension into individual elements. Covers unknown-rank, scalar, nested dataset, string,
# sparse, ragged, tuple, empty, unsigned-integer and none-valued components, static and dynamic
# shape mismatch errors, naming, checkpointing, and random access. Not a helper module.
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
"""Tests for `tf.data.Dataset.unbatch()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class UnbatchTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Behavior tests for unbatch across element types, shapes and error cases.

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchWithUnknownRankInput(self):
    # [FUNCTION DOC]
    # Unbatches a single element holding the list zero to three and asserts the four individual
    # values are produced.
    dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3]).unbatch()
    self.assertDatasetProduces(dataset, range(4))

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchScalarDataset(self):
    # [FUNCTION DOC]
    # Builds a dataset of triples of ranges, batches by two and unbatches. Asserts the element
    # types are three 32-bit integers both before and after unbatching, and that the output is
    # ten triples of repeated integers.
    data = tuple([math_ops.range(10) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = (dtypes.int32,) * 3
    data = data.batch(2)
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))
    data = data.unbatch()
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))

    self.assertDatasetProduces(data, [(i,) * 3 for i in range(10)])

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchNestedDataset(self):
    # [FUNCTION DOC]
    # Unbatches a single element that is a list of ten range datasets, then flattens each nested
    # dataset. Asserts the range of ten appears ten times.
    data = dataset_ops.Dataset.from_tensors(
        [dataset_ops.Dataset.range(10) for _ in range(10)])
    data = data.unbatch().flat_map(lambda x: x)
    self.assertDatasetProduces(data, list(range(10)) * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchDatasetWithStrings(self):
    # [FUNCTION DOC]
    # Builds triples where the middle component is converted to a string, batches by two and
    # unbatches. Asserts the types (integer, string, integer) before and after, and the output
    # triples with byte-string middle values.
    data = tuple([math_ops.range(10) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    data = data.map(lambda x, y, z: (x, string_ops.as_string(y), z))
    expected_types = (dtypes.int32, dtypes.string, dtypes.int32)
    data = data.batch(2)
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))
    data = data.unbatch()
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))

    self.assertDatasetProduces(
        data, [(i, compat.as_bytes(str(i)), i) for i in range(10)])

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchDatasetWithSparseTensor(self):
    # [FUNCTION DOC]
    # Starts from a ten-by-ten diagonal sparse tensor, unbatches, batches by five, and unbatches
    # again. Asserts ten sparse rows are produced, each with one entry at its own position.
    st = sparse_tensor.SparseTensorValue(
        indices=[[i, i] for i in range(10)],
        values=list(range(10)),
        dense_shape=[10, 10])
    data = dataset_ops.Dataset.from_tensors(st)
    data = data.unbatch()
    data = data.batch(5)
    data = data.unbatch()
    expected_output = [
        sparse_tensor.SparseTensorValue([[i]], [i], [10]) for i in range(10)
    ]
    self.assertDatasetProduces(data, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchDatasetWithDenseSparseAndRaggedTensor(self):
    # [FUNCTION DOC]
    # Combines a dense list, a diagonal sparse tensor and a ragged tensor in one element,
    # unbatches, batches by five and unbatches again. Asserts ten triples are produced with the
    # dense value, a one-entry sparse row and a one-entry ragged row.
    st = sparse_tensor.SparseTensorValue(
        indices=[[i, i] for i in range(10)],
        values=list(range(10)),
        dense_shape=[10, 10])
    rt = ragged_factory_ops.constant_value([[[0]], [[1]], [[2]], [[3]], [[4]],
                                            [[5]], [[6]], [[7]], [[8]], [[9]]])
    data = dataset_ops.Dataset.from_tensors((list(range(10)), st, rt))
    data = data.unbatch()
    data = data.batch(5)
    data = data.unbatch()
    expected_output = [(i, sparse_tensor.SparseTensorValue([[i]], [i], [10]),
                        ragged_factory_ops.constant_value([[i]]))
                       for i in range(10)]
    self.assertDatasetProduces(
        data, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchDatasetWithRaggedTensor(self):
    # [FUNCTION DOC]
    # Takes a ragged tensor of ten rows, unbatches, batches by five, batches by two, and unbatches.
    # Asserts two ragged batches of five rows each are produced.
    rt = ragged_factory_ops.constant_value([[[0]], [[1]], [[2]], [[3]], [[4]],
                                            [[5]], [[6]], [[7]], [[8]], [[9]]])
    data = dataset_ops.Dataset.from_tensors(rt)
    data = data.unbatch()
    data = data.batch(5)
    data = data.batch(2)
    data = data.unbatch()
    expected_output = [
        ragged_factory_ops.constant_value([[[0]], [[1]], [[2]], [[3]], [[4]]]),
        ragged_factory_ops.constant_value([[[5]], [[6]], [[7]], [[8]], [[9]]]),
    ]
    self.assertDatasetProduces(
        data, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchSingleElementTupleDataset(self):
    # [FUNCTION DOC]
    # Uses elements that are triples of one-element tuples of ranges; batches by two and
    # unbatches. Asserts the nested types are preserved before and after and the output is ten
    # nested tuples.
    data = tuple([(math_ops.range(10),) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = ((dtypes.int32,),) * 3
    data = data.batch(2)
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))
    data = data.unbatch()
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))

    self.assertDatasetProduces(data, [((i,),) * 3 for i in range(10)])

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchMultiElementTupleDataset(self):
    # [FUNCTION DOC]
    # Uses triples of pairs (integer range, string fill of "hi"); batches by two and unbatches.
    # Asserts the types are preserved before and after and the output matches the expected nested
    # tuples of integers and byte strings.
    data = tuple([(math_ops.range(10 * i, 10 * i + 10),
                   array_ops.fill([10], "hi")) for i in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = ((dtypes.int32, dtypes.string),) * 3
    data = data.batch(2)
    self.assertAllEqual(expected_types,
                        dataset_ops.get_legacy_output_types(data))
    data = data.unbatch()
    self.assertAllEqual(expected_types,
                        dataset_ops.get_legacy_output_types(data))

    self.assertDatasetProduces(
        data,
        [((i, b"hi"), (10 + i, b"hi"), (20 + i, b"hi")) for i in range(10)])

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchEmpty(self):
    # [FUNCTION DOC]
    # Unbatches a single element whose components all have zero length in the first dimension and
    # asserts the result is an empty dataset.
    data = dataset_ops.Dataset.from_tensors(
        (constant_op.constant([]), constant_op.constant([], shape=[0, 4]),
         constant_op.constant([], shape=[0, 4, 0])))
    data = data.unbatch()
    self.assertDatasetProduces(data, [])

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchStaticShapeMismatch(self):
    # [FUNCTION DOC]
    # Builds an element with three components of different lengths (seven, eight, nine) and
    # asserts that unbatching raises a value error at construction time.
    data = dataset_ops.Dataset.from_tensors((np.arange(7), np.arange(8),
                                             np.arange(9)))
    with self.assertRaises(ValueError):
      data.unbatch()

  @combinations.generate(test_base.graph_only_combinations())
  def testUnbatchDynamicShapeMismatch(self):
    # [FUNCTION DOC]
    # Graph mode only. Uses two placeholders (one with unknown length, one of unknown rank) and an
    # initializable iterator. In a cached session, feeds arrays of lengths seven and eight and
    # asserts reading raises an invalid-argument error; then re-initializes with an array and a
    # scalar and asserts the same error.
    ph1 = array_ops.placeholder(dtypes.int32, shape=[None])
    ph2 = array_ops.placeholder(dtypes.int32, shape=None)
    data = dataset_ops.Dataset.from_tensors((ph1, ph2))
    data = data.unbatch()
    iterator = dataset_ops.make_initializable_iterator(data)
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      # Mismatch in the 0th dimension.
      sess.run(
          iterator.initializer,
          feed_dict={
              ph1: np.arange(7).astype(np.int32),
              ph2: np.arange(8).astype(np.int32)
          })
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(next_element)

      # No 0th dimension (i.e. scalar value) for one component.
      sess.run(
          iterator.initializer,
          feed_dict={
              ph1: np.arange(7).astype(np.int32),
              ph2: 7
          })
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(next_element)

  @combinations.generate(test_base.default_test_combinations())
  def testUnbatchDatasetWithUintDtypes(self):
    # [FUNCTION DOC]
    # Uses four components of unsigned integer types (8, 16, 32, 64 bits), including values just
    # beyond the previous type's range; batches by two and unbatches. Asserts the types are kept
    # and the four resulting tuples equal the original rows.
    components = (
        np.tile(np.array([[0], [1], [2], [3]], dtype=np.uint8), 2),
        np.tile(np.array([[1], [2], [3], [256]], dtype=np.uint16), 2),
        np.tile(np.array([[2], [3], [4], [65536]], dtype=np.uint32), 2),
        np.tile(np.array([[3], [4], [5], [4294967296]], dtype=np.uint64), 2),
    )
    expected_types = (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
    expected_output = [tuple([c[i] for c in components]) for i in range(4)]

    data = dataset_ops.Dataset.from_tensor_slices(components)
    data = data.batch(2)
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))

    data = data.unbatch()
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))
    self.assertDatasetProduces(data, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNoneComponent(self):
    # [FUNCTION DOC]
    # Unbatches an element with a list and a null component, then maps to keep only the first
    # component. Asserts the values zero to nine are produced.
    dataset = dataset_ops.Dataset.from_tensors(
        (list(range(10)), None)).unbatch().map(lambda x, y: x)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Checks that unbatch accepts a name argument: unbatching a one-element list holding 42 yields
    # that value.
    dataset = dataset_ops.Dataset.from_tensors([42]).unbatch(name="unbatch")
    self.assertDatasetProduces(dataset, [42])


class UnbatchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                            parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for a batch-then-unbatch dataset, using the shared checkpoint
  # verification fixture.

  def build_dataset(self,
                    multiplier=15.0,
                    tensor_slice_len=2,
                    batch_size=2,
                    options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test. Creates three components of length tensor_slice_len
    # (default two): a range, a matrix row product with a fixed row, and the range scaled by
    # multiplier (default 15.0); slices them, batches by batch_size (default two) and unbatches.
    # If options (optional dataset options) is given, applies them. Returns the dataset.
    components = (np.arange(tensor_slice_len), np.array([[1, 2, 3]]) *
                  np.arange(tensor_slice_len)[:, np.newaxis],
                  np.array(multiplier) * np.arange(tensor_slice_len))

    dataset = dataset_ops.Dataset.from_tensor_slices(components).batch(
        batch_size).unbatch()
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
    # Parameterized checkpoint test over symbolic checkpoint on/off. Uses eight slices and batch
    # size two, and asks verify_fn (checkpoint verification callback) to check save/restore of
    # the dataset yields eight elements.
    tensor_slice_len = 8
    batch_size = 2
    num_outputs = tensor_slice_len
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self.build_dataset(15.0, tensor_slice_len, batch_size, options),
        num_outputs)


class UnbatchRandomAccessTest(test_base.DatasetTestBase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Tests random access (reading an element by index) on unbatched datasets.
  @combinations.generate(test_base.default_test_combinations())
  def test(self):
    # [FUNCTION DOC]
    # Batches a ten-element range by four dropping the remainder, unbatches, and asserts random
    # access at each of the eight positions returns the element equal to its index.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.batch(4, drop_remainder=True)
    dataset = dataset.unbatch()
    for i in range(8):
      self.assertEqual(self.evaluate(random_access.at(dataset, i)), i)

  @combinations.generate(test_base.default_test_combinations())
  def testNotDropRemainder(self):
    # [FUNCTION DOC]
    # Batches a ten-element range by four keeping the remainder, unbatches, and asserts random
    # access raises a failed-precondition error (the size is not statically known).
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.batch(4, drop_remainder=False)
    dataset = dataset.unbatch()
    with self.assertRaises(errors.FailedPreconditionError):
      self.evaluate(random_access.at(dataset, 0))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 100])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # Parameterized over invalid indices (negative and far beyond the end). Uses an unbatched
    # dataset of eight elements and asserts random access raises an out-of-range error.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.batch(4, drop_remainder=True)
    dataset = dataset.unbatch()
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))


if __name__ == "__main__":
  test.main()
