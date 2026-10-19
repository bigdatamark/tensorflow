# [MODULE DOC]
# Kernel tests for the ragged-batch dataset transformation, which batches elements of varying
# shape into ragged tensors. Provides a set of private dataset factories (scalars, vectors,
# matrices, five-dimensional tensors, ragged tensors, mappings and tuples of differing element
# shapes) that parameterize the main test, plus a conversion helper that turns tensors or
# ragged values into nested lists for comparison. Tests check batching with and without
# dropping the remainder and with structured (dense, ragged, sparse) elements. No import-time
# side effects other than the test entry point.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.ragged_batch`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def _make_scalar_ds(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory and base of the other factories. Parameter nrows is the number
  # of rows. Returns a dataset of the integers zero up to nrows minus one as scalar elements.
  """Create a test dataset with scalar elements."""
  return dataset_ops.Dataset.from_tensor_slices(np.arange(nrows))


def _make_vector_ds(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory. Parameter nrows is the number of rows. Returns a dataset in
  # which row i is a vector of length i filled with the value i.
  """Create a test dataset with vector elements (of varying size)."""
  return _make_scalar_ds(nrows).map(lambda x: array_ops.fill([x], x))


def _make_matrix_ds1(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory. Parameter nrows is the number of rows. Returns a dataset in
  # which row i is a matrix with i rows and two columns filled with the value i (varying first
  # dimension).
  """Create a test dataset with matrix elements (of varying size)."""
  return _make_scalar_ds(nrows).map(lambda x: array_ops.fill([x, 2], x))


def _make_matrix_ds2(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory. Parameter nrows is the number of rows. Returns a dataset in
  # which row i is a matrix with two rows and i columns filled with the value i (varying
  # second dimension).
  """Create a test dataset with matrix elements (of varying size)."""
  return _make_scalar_ds(nrows).map(lambda x: array_ops.fill([2, x], x))


def _make_matrix_ds_fully_defined(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory. Parameter nrows is the number of rows. Returns a dataset in
  # which every row is a two-by-three matrix filled with the row index (fixed shape, so a
  # regular batch could be used).
  """Create a test dataset with matrix elements (of varying size)."""
  return _make_scalar_ds(nrows).map(lambda x: array_ops.fill([2, 3], x))


def _make_5dtensor_ds(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory. Parameter nrows is the number of rows. Returns a dataset in
  # which row x is a five-dimensional tensor of shape two, x, three, twice x, four, filled with
  # x (several varying dimensions). The shared docstring wording says matrix elements, which
  # is inexact.
  """Create a test dataset with matrix elements (of varying size)."""
  return _make_scalar_ds(nrows).map(
      lambda x: array_ops.fill([2, x, 3, 2*x, 4], x))


def _make_ragged_ds(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory. Parameter nrows is the number of rows. Builds nested lists of
  # varying length from index arithmetic, converts them into one ragged tensor and returns a
  # dataset whose elements are the ragged tensor's rows (each itself ragged).
  """Create a test dataset with RaggedTensor elements (of varying size)."""
  values = [[[i] * (i % 3) for i in range(j)] * (j % 3) for j in range(nrows)]
  rt = ragged_factory_ops.constant(values)
  return dataset_ops.Dataset.from_tensor_slices(rt)


def _make_dict_ds(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory. Parameter nrows is the number of rows. Returns a dataset of
  # mappings with five entries of different shapes (scalar, vector, two matrix kinds and a
  # five-dimensional tensor), all derived from the row index.
  """Create a test set with various element shapes."""
  def transform(x):
    # [FUNCTION DOC]
    # HELPER: map function building the mapping of differently shaped tensors from the scalar
    # row index x.
    return {
        'shape=[]': ops.convert_to_tensor(x),
        'shape=[x]': math_ops.range(x),
        'shape=[x, 2]': array_ops.fill([x, 2], x),
        'shape=[2, x]': array_ops.fill([2, x], x),
        'shape=[2, x, 3, 2x, 4]': array_ops.fill([2, x, 3, 2*x, 4], x)
    }
  return _make_scalar_ds(nrows).map(transform)


def _make_tuple_ds(nrows):
  # [FUNCTION DOC]
  # HELPER: test dataset factory. Parameter nrows is the number of rows. Returns a dataset of
  # three-element tuples: the scalar index, a vector of its range and a matrix with x rows and
  # two columns.
  """Create a test set with various element shapes."""
  def transform(x):
    # [FUNCTION DOC]
    # HELPER: map function building the tuple of differently shaped tensors from the scalar row
    # index x.
    return (ops.convert_to_tensor(x),
            math_ops.range(x),
            array_ops.fill([x, 2], x))
  return _make_scalar_ds(nrows).map(transform)


def _to_list(v):
  # [FUNCTION DOC]
  # HELPER: converts a value to a plain nested list for comparison. Parameter v is a ragged
  # value (has a list conversion method) or a numpy-style array. Returns the nested list.
  return v.to_list() if hasattr(v, 'to_list') else v.tolist()


class RaggedBatchTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the ragged-batch transformation across many element kinds and structures.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              make_dataset=[
                  _make_scalar_ds, _make_vector_ds, _make_matrix_ds1,
                  _make_matrix_ds2, _make_ragged_ds, _make_5dtensor_ds,
                  _make_dict_ds, _make_tuple_ds, _make_matrix_ds_fully_defined,
              ],
              nrows=[0, 20, 23],
              batch_size=[4],
              drop_remainder=[True, False])))
  def testBasic(self, make_dataset, nrows, batch_size, drop_remainder):
    # [FUNCTION DOC]
    # Scenario: for each dataset factory (scalars, vectors, matrices, five-dimensional tensors,
    # ragged tensors, mappings, tuples, fixed-shape matrices), row count (zero, twenty, twenty
    # three), batch size four and drop-remainder flag. The unbatched rows are first collected as
    # nested lists; then the dataset is ragged-batched. Expected: each batch equals the matching
    # slice of rows (compared structure by structure), a trailing partial batch appears only
    # when the remainder is kept, and finally an out-of-range error is raised.
    dataset = make_dataset(nrows)

    # Get the unbatched rows (so we can check expected values).
    get_next = self.getNext(dataset)
    rows = [nest.map_structure(_to_list, self.evaluate(get_next()))
            for _ in range(nrows)]

    # Batch the dataset, and check that batches match slices from `rows`.
    batched_dataset = dataset.ragged_batch(batch_size, drop_remainder)
    get_next = self.getNext(batched_dataset)
    for start_row in range(0, nrows, batch_size):
      end_row = start_row + batch_size
      if end_row > nrows and drop_remainder:
        break
      end_row = min(end_row, nrows)
      result = self.evaluate(get_next())

      # Use nest for potentially nested datasets.
      nest.map_structure_up_to(
          result, lambda a, *b: self.assertAllEqual(a, list(b)),
          result, *rows[start_row:end_row])

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testWithStructuredElements(self):
    # [FUNCTION DOC]
    # Scenario: twenty rows whose elements are mappings holding a dense vector, a ragged tensor
    # and a sparse tensor, ragged-batched by four. Expected: dense entries become ragged lists
    # of the right lengths, ragged entries stack as expected, and sparse entries get batch
    # indices, the values and a dense shape of four by one hundred.
    nrows = 20
    batch_size = 4

    def make_structure(x):
      # [FUNCTION DOC]
      # HELPER: map function turning scalar x into a mapping with a dense vector of length x, a
      # ragged tensor with rows of lengths one and two, and a sparse tensor with a single entry
      # at position x in a vector of length one hundred.
      return {
          'dense':
              array_ops.fill([x], x),
          'ragged':
              ragged_concat_ops.stack(
                  [array_ops_stack.stack([x]),
                   array_ops_stack.stack([x, x])]),
          'sparse':
              sparse_tensor.SparseTensor([[x]], [x], [100])
      }

    dataset = dataset_ops.Dataset.from_tensor_slices(np.arange(nrows))
    dataset = dataset.map(make_structure)
    dataset = dataset.ragged_batch(batch_size)
    get_next = self.getNext(dataset)

    for i in range(0, nrows, batch_size):
      result = self.evaluate(get_next())
      rows = range(i, i + batch_size)
      self.assertAllEqual(result['dense'], [[r] * r for r in rows])
      self.assertAllEqual(result['ragged'], [[[r], [r, r]] for r in rows])
      self.assertAllEqual(result['sparse'].indices, list(enumerate(rows)))
      self.assertAllEqual(result['sparse'].values, rows)
      self.assertAllEqual(result['sparse'].dense_shape, [4, 100])


if __name__ == '__main__':
  test.main()
