# [MODULE DOC]
# Test module for the sparse feature-cross operations (sparse_cross, sparse_cross_hashed and the
# raw generated sparse_cross_v2 / sparse_cross_hashed kernels). It builds string or int64 sparse
# and dense feature columns, crosses them, and compares the resulting sparse tensor (indices,
# values, dense shape) with hand-written expectations, including hashed outputs whose exact
# fingerprints are pinned to catch accidental hash changes. It also checks empty columns, large
# batches (multithreading), custom separators and input validation errors.
# Defines three test classes: a shared base class with helpers, the legacy-API test class and the
# V2 / hashed raw-op test classes. No import-time side effects besides the test main entry point.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sparse_cross_op."""

import numpy

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class BaseSparseCrossOpTest(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class giving the sparse-cross test classes common fixtures.
  # Base test case that only provides helper methods (building a sparse tensor from nested
  # lists, comparing two sparse tensors, asserting that one is empty). It holds no state and
  # defines no tests of its own; the V2 and hashed test classes inherit from it.

  def _sparse_tensor(self, data, batch_size=-1):
    # [FUNCTION DOC]
    # HELPER: builds the sparse input tensors used by the cross-op tests.
    # Converts a list (one entry per batch row) of lists of feature values into a sparse tensor
    # with a two-dimensional shape. Inputs: data, nested list of strings or integers; batch_size,
    # optional number of rows (default means use the length of data, useful when trailing rows
    # are empty). The value dtype is string when there are no values or the first value is a
    # string, otherwise 64-bit integer. The column count is the longest row. Returns the sparse
    # tensor; raises nothing itself and has no side effects.
    """Generates a SparseTensor.

    Args:
      data: Should be a list of list of strings or int64. Each item of the outer
        list represents a batch. Each item of the batch is a feature of a
        specific feature column.
      batch_size: optional batch size, especially for cases when data has no
        entry for some batches.

    Returns:
     A SparseTensor.
    """
    indices = []
    values = []
    max_col_count = 0
    for batch, batch_ix in zip(data, range(len(data))):
      for column, column_ix in zip(batch, range(len(batch))):
        indices.append([batch_ix, column_ix])
        values.append(column)
        max_col_count = max(max_col_count, column_ix + 1)
    shape = [batch_size if batch_size != -1 else len(data), max_col_count]
    value_type = (
        dtypes.string
        if not values or isinstance(values[0], str) else dtypes.int64)
    return sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64, [len(indices), 2]),
        constant_op.constant(values, value_type, [len(indices)]),
        constant_op.constant(shape, dtypes.int64))

  def _assert_sparse_tensor_equals(self, sp1, sp2):
    # [FUNCTION DOC]
    # HELPER: comparison helper for the cross-op tests.
    # Asserts that two evaluated sparse tensors (sp1 and sp2) have equal indices, values and
    # dense shapes. Fails the test with the usual array-comparison assertion error on any
    # difference. Returns nothing.
    self.assertAllEqual(sp1.indices, sp2.indices)
    self.assertAllEqual(sp1.values, sp2.values)
    self.assertAllEqual(sp1.dense_shape, sp2.dense_shape)

  def _assert_sparse_tensor_empty(self, sp):
    # [FUNCTION DOC]
    # HELPER: emptiness check for the cross-op tests.
    # Asserts that the evaluated sparse tensor sp has no indices, no values and a second
    # dense-shape dimension of zero (the first dimension is deliberately not checked). Fails the
    # test otherwise. Returns nothing.
    self.assertEqual(0, sp.indices.size)
    self.assertEqual(0, sp.values.size)
    # TODO(zakaria): check if we can ignore the first dim of the shape.
    self.assertEqual(0, sp.dense_shape[1])


class SparseCrossOpTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: derives directly from the test case base instead of the shared base class and
  # re-declares copies of the three helper methods that the base class already provides.
  # Tests the legacy sparse_cross and sparse_cross_hashed Python wrappers: simple, dense-only,
  # mixed string/integer, sparse plus dense, permutations, large batches, empty columns, hashed
  # outputs with pinned values, and validation errors. It has its own copies of the helpers.

  @test_util.run_deprecated_v1
  def test_simple(self):
    # [FUNCTION DOC]
    # Crosses two sparse string columns with one and two features per batch row. Expects each
    # row to contain the cartesian product of its features joined with the default separator.
    # Compares the evaluated result with the hand-built expected sparse tensor.
    """Tests a simple scenario."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([['batch1-FC1-F1'],
                             ['batch2-FC1-F1', 'batch2-FC1-F2']]),
        self._sparse_tensor([['batch1-FC2-F1'],
                             ['batch2-FC2-F1', 'batch2-FC2-F2']])
    ])
    expected_out = self._sparse_tensor([['batch1-FC1-F1_X_batch1-FC2-F1'], [
        'batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
        'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
    ]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_dense(self):
    # [FUNCTION DOC]
    # Crosses two dense string matrices (two features per row each) with the legacy wrapper.
    # Expects four crossed strings per row, in row-major product order, in a sparse result.
    """Tests only dense inputs."""
    op = sparse_ops.sparse_cross([
        constant_op.constant([['batch1-FC1-F1', 'batch1-FC1-F2'],
                              ['batch2-FC1-F1', 'batch2-FC1-F2']],
                             dtypes.string),
        constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                              ['batch2-FC2-F1', 'batch2-FC2-F2']],
                             dtypes.string),
    ])
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1', 'batch1-FC1-F1_X_batch1-FC2-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1', 'batch1-FC1-F2_X_batch1-FC2-F2'
    ], [
        'batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
        'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
    ]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_integer_mixed_string_sparse(self):
    # [FUNCTION DOC]
    # Crosses a sparse integer column with a sparse string column. Expects integers to be
    # rendered as decimal text inside the crossed strings, in product order per row.
    """Tests mixed type."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([[11], [333, 55555]]),
        self._sparse_tensor([['batch1-FC2-F1'],
                             ['batch2-FC2-F1', 'batch2-FC2-F2']])
    ])
    expected_out = self._sparse_tensor([['11_X_batch1-FC2-F1'], [
        '333_X_batch2-FC2-F1', '333_X_batch2-FC2-F2', '55555_X_batch2-FC2-F1',
        '55555_X_batch2-FC2-F2'
    ]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_integer_mixed_string_dense(self):
    # [FUNCTION DOC]
    # Crosses a dense 64-bit integer matrix with a dense string matrix. Expects four crossed
    # strings per row with integers rendered as decimal text.
    """Tests mixed dense inputs."""
    op = sparse_ops.sparse_cross([
        constant_op.constant([[11, 333], [55555, 999999]], dtypes.int64),
        constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                              ['batch2-FC2-F1', 'batch2-FC2-F2']],
                             dtypes.string),
    ])
    expected_out = self._sparse_tensor([[
        '11_X_batch1-FC2-F1', '11_X_batch1-FC2-F2', '333_X_batch1-FC2-F1',
        '333_X_batch1-FC2-F2'
    ], [
        '55555_X_batch2-FC2-F1', '55555_X_batch2-FC2-F2',
        '999999_X_batch2-FC2-F1', '999999_X_batch2-FC2-F2'
    ]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_sparse_cross_dense(self):
    # [FUNCTION DOC]
    # Crosses one sparse string column with one dense string matrix. Expects the per-row
    # product of the sparse features (one or two) with the two dense features.
    """Tests sparse and dense inputs."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([['batch1-FC1-F1'],
                             ['batch2-FC1-F1', 'batch2-FC1-F2']]),
        constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                              ['batch2-FC2-F1', 'batch2-FC2-F2']],
                             dtypes.string),
    ])
    expected_out = self._sparse_tensor(
        [['batch1-FC1-F1_X_batch1-FC2-F1', 'batch1-FC1-F1_X_batch1-FC2-F2'], [
            'batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
            'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
        ]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_integer_sparse_input(self):
    # [FUNCTION DOC]
    # Crosses a sparse integer column with a dense string matrix. Expects integers rendered as
    # decimal text joined to each dense string, per row, in product order.
    """Tests mixed type sparse and dense inputs."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([[11], [333, 5555]]),
        constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                              ['batch2-FC2-F1', 'batch2-FC2-F2']],
                             dtypes.string),
    ])
    expected_out = self._sparse_tensor(
        [['11_X_batch1-FC2-F1', '11_X_batch1-FC2-F2'], [
            '333_X_batch2-FC2-F1', '333_X_batch2-FC2-F2',
            '5555_X_batch2-FC2-F1', '5555_X_batch2-FC2-F2'
        ]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_permutation_3x3x3(self):
    # [FUNCTION DOC]
    # Crosses three sparse columns of three features each in a single row. Expects all 27
    # combinations in lexicographic order, joined with the default separator.
    """Tests 3x3x3 permutation."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor(
            [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']]),
        self._sparse_tensor(
            [['batch1-FC2-F1', 'batch1-FC2-F2', 'batch1-FC2-F3']]),
        self._sparse_tensor(
            [['batch1-FC3-F1', 'batch1-FC3-F2', 'batch1-FC3-F3']])
    ])
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F3'
    ]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_permutation_3x1x2(self):
    # [FUNCTION DOC]
    # Crosses three sparse columns with three, one and two features in one row. Expects the six
    # combinations in lexicographic order.
    """Tests 3x1x2 permutation."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor(
            [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']]),
        self._sparse_tensor([['batch1-FC2-F1']]),
        self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    ])
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F2'
    ]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_large_batch(self):
    # [FUNCTION DOC]
    # Builds three sparse columns for 5000 batch rows (three, one and two features per row) and
    # crosses them with the legacy wrapper. Expects six crossed strings per row, built in plain
    # Python for comparison. The large batch is meant to exercise multithreaded kernel execution.
    """Tests with large batch size to force multithreading."""
    batch_size = 5000
    col1 = []
    col2 = []
    col3 = []
    for b in range(batch_size):
      col1.append(
          ['batch%d-FC1-F1' % b, 'batch%d-FC1-F2' % b, 'batch%d-FC1-F3' % b])
      col2.append(['batch%d-FC2-F1' % b])
      col3.append(['batch%d-FC3-F1' % b, 'batch%d-FC3-F2' % b])

    op = sparse_ops.sparse_cross([
        self._sparse_tensor(col1),
        self._sparse_tensor(col2),
        self._sparse_tensor(col3)
    ])

    col_out = []
    for b in range(batch_size):
      col_out.append([
          'batch%d-FC1-F1_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F1_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b),
          'batch%d-FC1-F2_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F2_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b),
          'batch%d-FC1-F3_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F3_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b)
      ])

    expected_out = self._sparse_tensor(col_out)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_one_column_empty(self):
    # [FUNCTION DOC]
    # Crosses three columns where the middle one has no entries (batch size one). Expects the
    # crossed result to be completely empty (no indices or values, zero columns).
    """Tests when one column is empty.

    The crossed tensor should be empty.
    """
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']]),
        self._sparse_tensor([], 1),
        self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    ])
    with self.cached_session():
      self._assert_sparse_tensor_empty(self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_some_columns_empty(self):
    # [FUNCTION DOC]
    # Crosses three columns over a batch of two rows where only the middle column has data in the
    # second row. Expects the first row to contain the four crossed strings and the second row
    # to produce nothing, with the dense shape keeping both rows.
    """Tests when more than one columns are empty.

    Cross for the corresponding batch should be empty.
    """
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']], 2),
        self._sparse_tensor([['batch1-FC2-F1'], ['batch2-FC2-F1']], 2),
        self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']], 2)
    ])
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2'
    ]], 2)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_all_columns_empty(self):
    # [FUNCTION DOC]
    # Crosses three completely empty sparse columns with the legacy wrapper and expects an
    # empty sparse result.
    """Tests when all columns are empty.

    The crossed tensor should be empty.
    """
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([]),
        self._sparse_tensor([]),
        self._sparse_tensor([])
    ])
    with self.cached_session():
      self._assert_sparse_tensor_empty(self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_hashed_zero_bucket_no_hash_key(self):
    # [FUNCTION DOC]
    # Hash-crosses three single-feature columns with the default bucket count (zero, meaning no
    # modulo) and the default hash key. Expects the exact pinned 64-bit fingerprint so that
    # unintended hashing changes are caught.
    op = sparse_ops.sparse_cross_hashed([
        self._sparse_tensor([['batch1-FC1-F1']]),
        self._sparse_tensor([['batch1-FC2-F1']]),
        self._sparse_tensor([['batch1-FC3-F1']])
    ])
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[1971693436396284976]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_hashed_zero_bucket(self):
    # [FUNCTION DOC]
    # Same hash-cross as the default case but with the default hash key incremented by one and
    # zero buckets. Expects a different exact pinned fingerprint.
    op = sparse_ops.sparse_cross_hashed(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        hash_key=sparse_ops._DEFAULT_HASH_KEY + 1)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[4847552627144134031]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  # TODO(sibyl-Aix6ihai): Add benchmark to compare Hashed vs Non-hashed.
  @test_util.run_deprecated_v1
  def test_hashed_no_hash_key(self):
    # [FUNCTION DOC]
    # Hash-crosses three single-feature columns into 100 buckets using the default hash key.
    # Expects the pinned bucket index.
    op = sparse_ops.sparse_cross_hashed(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        num_buckets=100)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[83]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_hashed_output(self):
    # [FUNCTION DOC]
    # Hash-crosses three single-feature columns into 100 buckets with a custom hash key (default
    # plus one). Expects the pinned bucket index.
    op = sparse_ops.sparse_cross_hashed(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        num_buckets=100,
        hash_key=sparse_ops._DEFAULT_HASH_KEY + 1)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[31]])
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_hashed__has_no_collision(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name contains a doubled underscore (likely a typo).
    # Hash-crosses a ten-feature dense column with a single value that differs between two rows
    # only above the bucket-mask bits (359 versus 1024 plus 359), into 1024 buckets, and converts
    # to dense. Expects every entry of the two rows to differ, i.e. no collisions.
    """Tests that fingerprint concatenation has no collisions."""
    # Although the last 10 bits of 359 and 1024+359 are identical.
    # As a result, all the crosses shouldn't collide.
    t1 = constant_op.constant([[359], [359 + 1024]])
    t2 = constant_op.constant([list(range(10)), list(range(10))])
    cross = sparse_ops.sparse_cross_hashed(
        [t2, t1], num_buckets=1024, hash_key=sparse_ops._DEFAULT_HASH_KEY + 1)
    cross_dense = sparse_ops.sparse_tensor_to_dense(cross)
    with session.Session():
      values = self.evaluate(cross_dense)
      self.assertTrue(numpy.not_equal(values[0], values[1]).all())

  def test_hashed_3x1x2(self):
    # [FUNCTION DOC]
    # Hash-crosses columns of three, one and two features into 1000 buckets. Expects six
    # entries in one row with consecutive column indices, all values in range, and all distinct.
    """Tests 3x1x2 permutation with hashed output."""
    op = sparse_ops.sparse_cross_hashed(
        [
            self._sparse_tensor(
                [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
        ],
        num_buckets=1000)
    with self.cached_session():
      out = self.evaluate(op)
      self.assertEqual(6, len(out.values))
      self.assertAllEqual([[0, i] for i in range(6)], out.indices)
      self.assertTrue(all(x < 1000 and x >= 0 for x in out.values))
      all_values_are_different = len(out.values) == len(set(out.values))
      self.assertTrue(all_values_are_different)

  def _assert_sparse_tensor_empty(self, sp):
    # [FUNCTION DOC]
    # HELPER: emptiness check for the cross-op tests.
    # KNOWN ISSUE: duplicate of the identically named method in the shared base class.
    # Asserts the evaluated sparse tensor sp has no indices, no values and a zero second
    # dense-shape dimension. Returns nothing.
    self.assertEqual(0, sp.indices.size)
    self.assertEqual(0, sp.values.size)
    # TODO(zakaria): check if we can ignore the first dim of the shape.
    self.assertEqual(0, sp.dense_shape[1])

  def _assert_sparse_tensor_equals(self, sp1, sp2):
    # [FUNCTION DOC]
    # HELPER: comparison helper for the cross-op tests.
    # KNOWN ISSUE: duplicate of the identically named method in the shared base class.
    # Asserts that two evaluated sparse tensors have equal indices, values and dense shapes.
    # Returns nothing.
    self.assertAllEqual(sp1.indices, sp2.indices)
    self.assertAllEqual(sp1.values, sp2.values)
    self.assertAllEqual(sp1.dense_shape, sp2.dense_shape)

  def _sparse_tensor(self, data, batch_size=-1):
    # [FUNCTION DOC]
    # HELPER: builds the sparse input tensors used by the cross-op tests.
    # KNOWN ISSUE: duplicate of the identically named method in the shared base class.
    # Converts a list of per-row feature lists (strings or integers) into a two-dimensional
    # sparse tensor. batch_size optionally overrides the number of rows (default: length of
    # data). Value dtype is string when empty or the first value is a string, else 64-bit
    # integer. Returns the sparse tensor; no side effects.
    """Generates a SparseTensor.

    Args:
      data: Should be a list of list of strings or int64. Each item of the outer
          list represents a batch. Each item of the batch is a feature of a
          specific feature column.
      batch_size: optional batch size, especially for cases when data has no
          entry for some batches.

    Returns:
     A SparseTensor.
    """
    indices = []
    values = []
    max_col_count = 0
    for batch, batch_ix in zip(data, range(len(data))):
      for column, column_ix in zip(batch, range(len(batch))):
        indices.append([batch_ix, column_ix])
        values.append(column)
        max_col_count = max(max_col_count, column_ix + 1)
    shape = [batch_size if batch_size != -1 else len(data), max_col_count]
    value_type = (dtypes.string if not values or isinstance(values[0], str) else
                  dtypes.int64)
    return sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64, [len(indices), 2]),
        constant_op.constant(values, value_type, [len(indices)]),
        constant_op.constant(shape, dtypes.int64))

  def test_invalid_sparse_tensors(self):
    # [FUNCTION DOC]
    # Corrupts valid sparse tensors by overwriting their indices, values or dense shape with
    # scalar tensors, then runs the cross. Expects an invalid-argument or value error whose
    # message says indices must be a matrix, values a vector, or shapes a vector respectively.
    # Test validation of invalid SparseTensors.  The SparseTensor constructor
    # prevents us from creating invalid SparseTensors (eps. in eager mode),
    # so we create valid SparseTensors and then modify them to be invalid.

    st1 = sparse_tensor.SparseTensor([[0, 0]], [0], [2, 2])
    st1._indices = array_ops.zeros([], dtypes.int64)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Input indices should be a matrix'):
      self.evaluate(sparse_ops.sparse_cross([st1]))

    st2 = sparse_tensor.SparseTensor([[0, 0]], [0], [2, 2])
    st2._values = array_ops.zeros([], dtypes.int64)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Input values should be a vector'):
      self.evaluate(sparse_ops.sparse_cross([st2]))

    st3 = sparse_tensor.SparseTensor([[0, 0]], [0], [2, 2])
    st3._dense_shape = array_ops.zeros([], dtypes.int64)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Input shapes should be a vector'):
      self.evaluate(sparse_ops.sparse_cross([st3]))

  def test_bad_tensor_shapes(self):
    # [FUNCTION DOC]
    # Passes a one-dimensional sparse tensor and a one-dimensional dense tensor to the cross.
    # Expects an invalid-argument or value error saying the index rank must be two, and that
    # dense inputs must be matrices.
    # All inputs must be 2D.
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Expected D2 of index to be 2'):
      st = sparse_tensor.SparseTensor([[0]], [0], [10])  # 1D SparseTensor
      self.evaluate(sparse_ops.sparse_cross([st]))

    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Dense inputs should be a matrix'):
      dt = array_ops.zeros([0])  # 1D DenseTensor.
      self.evaluate(sparse_ops.sparse_cross([dt]))

  def test_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Crosses inputs with different batch sizes (10, 7 and 5 rows) in two combinations. Expects an
    # invalid-argument or value error mentioning the expected batch size each time.
    st1 = sparse_tensor.SparseTensor([[0, 0]], [0], [10, 10])  # batch size 10
    st2 = sparse_tensor.SparseTensor([[0, 0]], [0], [7, 10])  # batch size 7
    dt = array_ops.zeros([5, 0])  # batch size 5
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Expected batch size'):
      self.evaluate(sparse_ops.sparse_cross([st1, dt]))
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Expected batch size'):
      self.evaluate(sparse_ops.sparse_cross([st1, st2]))


class SparseCrossV2OpTest(BaseSparseCrossOpTest):
  # [CLASS DOC]
  # Tests the raw generated sparse_cross_v2 kernel wrapper, which takes indices, values, shapes
  # and dense inputs as separate lists plus a separator string, and returns the three components
  # of the crossed sparse tensor. Mirrors the legacy tests (sparse, dense, mixed, permutations,
  # large batch, empty columns) and adds custom-separator and invalid-separator cases. Inherits
  # the helpers from the shared base class.

  @test_util.run_deprecated_v1
  def test_sparse(self):
    # [FUNCTION DOC]
    # Crosses two sparse string columns with the V2 kernel and the standard separator. Expects
    # the per-row cartesian products.
    """Tests a simple scenario."""
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1'],
                                    ['batch2-FC1-F1', 'batch2-FC1-F2']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1'],
                                    ['batch2-FC2-F1', 'batch2-FC2-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices],
        values=[sp_inp_1.values, sp_inp_2.values],
        shapes=[sp_inp_1.dense_shape, sp_inp_2.dense_shape],
        dense_inputs=[],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['batch1-FC1-F1_X_batch1-FC2-F1'],
        ['batch2-FC1-F1_X_batch2-FC2-F1',
         'batch2-FC1-F1_X_batch2-FC2-F2',
         'batch2-FC1-F2_X_batch2-FC2-F1',
         'batch2-FC1-F2_X_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_sparse_sep(self):
    # [FUNCTION DOC]
    # Same as the simple sparse V2 cross but with a different separator string. Expects the
    # crossed strings to be joined with that separator.
    """Tests a simple scenario."""
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1'],
                                    ['batch2-FC1-F1', 'batch2-FC1-F2']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1'],
                                    ['batch2-FC2-F1', 'batch2-FC2-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices],
        values=[sp_inp_1.values, sp_inp_2.values],
        shapes=[sp_inp_1.dense_shape, sp_inp_2.dense_shape],
        dense_inputs=[],
        sep='_Y_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['batch1-FC1-F1_Y_batch1-FC2-F1'],
        ['batch2-FC1-F1_Y_batch2-FC2-F1',
         'batch2-FC1-F1_Y_batch2-FC2-F2',
         'batch2-FC1-F2_Y_batch2-FC2-F1',
         'batch2-FC1-F2_Y_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_dense(self):
    # [FUNCTION DOC]
    # Crosses two dense string matrices with the V2 kernel (no sparse inputs) and the standard
    # separator. Expects four crossed strings per row.
    """Tests only dense inputs."""
    dense_inp_1 = constant_op.constant([['batch1-FC1-F1', 'batch1-FC1-F2'],
                                        ['batch2-FC1-F1', 'batch2-FC1-F2']],
                                       dtypes.string)
    dense_inp_2 = constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                                        ['batch2-FC2-F1', 'batch2-FC2-F2']],
                                       dtypes.string)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[],
        values=[],
        shapes=[],
        dense_inputs=[dense_inp_1, dense_inp_2],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['batch1-FC1-F1_X_batch1-FC2-F1', 'batch1-FC1-F1_X_batch1-FC2-F2',
         'batch1-FC1-F2_X_batch1-FC2-F1', 'batch1-FC1-F2_X_batch1-FC2-F2'
        ],
        ['batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
         'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_dense_sep(self):
    # [FUNCTION DOC]
    # Dense-only V2 cross using a single underscore as separator. Expects the crossed strings
    # to be joined with only that separator.
    """Tests only dense inputs."""
    dense_inp_1 = constant_op.constant([['batch1-FC1-F1', 'batch1-FC1-F2'],
                                        ['batch2-FC1-F1', 'batch2-FC1-F2']],
                                       dtypes.string)
    dense_inp_2 = constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                                        ['batch2-FC2-F1', 'batch2-FC2-F2']],
                                       dtypes.string)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[],
        values=[],
        shapes=[],
        dense_inputs=[dense_inp_1, dense_inp_2],
        sep='_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['batch1-FC1-F1_batch1-FC2-F1', 'batch1-FC1-F1_batch1-FC2-F2',
         'batch1-FC1-F2_batch1-FC2-F1', 'batch1-FC1-F2_batch1-FC2-F2'
        ],
        ['batch2-FC1-F1_batch2-FC2-F1', 'batch2-FC1-F1_batch2-FC2-F2',
         'batch2-FC1-F2_batch2-FC2-F1', 'batch2-FC1-F2_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_integer_mixed_string_sparse(self):
    # [FUNCTION DOC]
    # V2 cross of a sparse integer column with a sparse string column. Expects integers rendered
    # as decimal text in the crossed strings.
    """Tests mixed type."""
    sp_inp_1 = self._sparse_tensor([[11], [333, 55555]])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1'],
                                    ['batch2-FC2-F1', 'batch2-FC2-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices],
        values=[sp_inp_1.values, sp_inp_2.values],
        shapes=[sp_inp_1.dense_shape, sp_inp_2.dense_shape],
        dense_inputs=[],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['11_X_batch1-FC2-F1'],
        ['333_X_batch2-FC2-F1', '333_X_batch2-FC2-F2',
         '55555_X_batch2-FC2-F1', '55555_X_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_integer_mixed_string_dense(self):
    # [FUNCTION DOC]
    # V2 cross of a dense integer matrix with a dense string matrix. Expects four crossed
    # strings per row with integers as decimal text.
    """Tests mixed dense inputs."""
    dense_inp_1 = constant_op.constant([[11, 333], [55555, 999999]],
                                       dtypes.int64)
    dense_inp_2 = constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                                        ['batch2-FC2-F1', 'batch2-FC2-F2']],
                                       dtypes.string)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[],
        values=[],
        shapes=[],
        dense_inputs=[dense_inp_1, dense_inp_2],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['11_X_batch1-FC2-F1', '11_X_batch1-FC2-F2',
         '333_X_batch1-FC2-F1', '333_X_batch1-FC2-F2'
        ],
        ['55555_X_batch2-FC2-F1', '55555_X_batch2-FC2-F2',
         '999999_X_batch2-FC2-F1', '999999_X_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_sparse_cross_dense(self):
    # [FUNCTION DOC]
    # V2 cross of one sparse string column with one dense string matrix. Expects the per-row
    # product of sparse features with the two dense features.
    """Tests sparse and dense inputs."""
    sp_inp = self._sparse_tensor([['batch1-FC1-F1'],
                                  ['batch2-FC1-F1', 'batch2-FC1-F2']])
    dense_inp = constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                                      ['batch2-FC2-F1', 'batch2-FC2-F2']],
                                     dtypes.string)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp.indices],
        values=[sp_inp.values],
        shapes=[sp_inp.dense_shape],
        dense_inputs=[dense_inp],
        sep='_X_')
    expected_out = self._sparse_tensor(
        [['batch1-FC1-F1_X_batch1-FC2-F1', 'batch1-FC1-F1_X_batch1-FC2-F2'],
         [
             'batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
             'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
         ]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_permutation_3x3x3(self):
    # [FUNCTION DOC]
    # V2 cross of three sparse columns with three features each in one row. Expects all 27
    # combinations in lexicographic order.
    """Tests 3x3x3 permutation."""
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor(
        [['batch1-FC2-F1', 'batch1-FC2-F2', 'batch1-FC2-F3']])
    sp_inp_3 = self._sparse_tensor(
        [['batch1-FC3-F1', 'batch1-FC3-F2', 'batch1-FC3-F3']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F3'
    ]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_permutation_3x1x2(self):
    # [FUNCTION DOC]
    # V2 cross of three sparse columns with three, one and two features. Expects six
    # combinations in lexicographic order.
    """Tests 3x1x2 permutation."""
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F2'
    ]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_large_batch(self):
    # [FUNCTION DOC]
    # V2 cross of three sparse columns over 5000 rows (three, one and two features per row).
    # Expects six crossed strings per row, built in plain Python for comparison; the batch size
    # is meant to exercise multithreaded execution.
    """Tests with large batch size to force multithreading."""
    batch_size = 5000
    col1 = []
    col2 = []
    col3 = []
    for b in range(batch_size):
      col1.append(
          ['batch%d-FC1-F1' % b,
           'batch%d-FC1-F2' % b,
           'batch%d-FC1-F3' % b])
      col2.append(['batch%d-FC2-F1' % b])
      col3.append(['batch%d-FC3-F1' % b, 'batch%d-FC3-F2' % b])
    sp_inp_1 = self._sparse_tensor(col1)
    sp_inp_2 = self._sparse_tensor(col2)
    sp_inp_3 = self._sparse_tensor(col3)

    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')

    col_out = []
    for b in range(batch_size):
      col_out.append([
          'batch%d-FC1-F1_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F1_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b),
          'batch%d-FC1-F2_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F2_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b),
          'batch%d-FC1-F3_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F3_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b)
      ])

    expected_out = self._sparse_tensor(col_out)
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_one_column_empty(self):
    # [FUNCTION DOC]
    # V2 cross where the middle of three sparse columns is empty. Expects an empty result.
    """Tests when one column is empty.

    The crossed tensor should be empty.
    """
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']])
    sp_inp_2 = self._sparse_tensor([], 1)
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_empty(self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_some_columns_empty(self):
    # [FUNCTION DOC]
    # V2 cross over two rows where only row one has data in all columns. Expects four crossed
    # strings in row one and nothing for row two, with the dense shape keeping both rows.
    """Tests when more than one columns are empty.

    Cross for the corresponding batch should be empty.
    """
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']], 2)
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1'], ['batch2-FC2-F1']], 2)
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']], 2)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2'
    ]], 2)
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_all_columns_empty(self):
    # [FUNCTION DOC]
    # V2 cross of three completely empty sparse columns. Expects an empty result.
    """Tests when all columns are empty.

    The crossed tensor should be empty.
    """
    sp_inp_1 = self._sparse_tensor([])
    sp_inp_2 = self._sparse_tensor([])
    sp_inp_3 = self._sparse_tensor([])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_empty(self.evaluate(out))

  def testNonScalarInput(self):
    # [FUNCTION DOC]
    # Calls the legacy cross wrapper with no inputs and a separator that is a vector of two
    # strings. Expects an invalid-argument error saying the separator must be a scalar.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Input separator should be a scalar.'):
      self.evaluate(sparse_ops.sparse_cross(
          inputs=[],
          name='a',
          separator=constant_op.constant(['a', 'b'], dtype=dtypes.string)))


class SparseCrossHashedOpTest(BaseSparseCrossOpTest):
  # [CLASS DOC]
  # Tests the raw generated sparse_cross_hashed kernel wrapper with explicit bucket counts,
  # salts and the strong-hash flag: pinned hash outputs, collision-freedom of fingerprint
  # concatenation, salt sensitivity, and independence from any separator. Inherits helpers from
  # the shared base class.

  @test_util.run_deprecated_v1
  def test_hashed_zero_bucket_no_hash_key(self):
    # [FUNCTION DOC]
    # Hash-crosses three single-feature columns with zero buckets, salt of one and one, and the
    # weak hash. Expects the pinned 64-bit value, then repeats with a different salt and expects
    # the same value because salt is ignored when the strong hash is off.
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        num_buckets=0,
        salt=[1, 1],
        strong_hash=False)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[9186962005966787372]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

    # salt is not being used when `strong_hash` is False.
    inds_2, vals_2, shapes_2 = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        num_buckets=0,
        salt=[137, 173],
        strong_hash=False)
    out_2 = sparse_tensor.SparseTensor(inds_2, vals_2, shapes_2)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out_2))

  @test_util.run_deprecated_v1
  def test_hashed_output(self):
    # [FUNCTION DOC]
    # Hash-crosses three single-feature columns into 100 buckets with the weak hash. Expects the
    # pinned bucket index.
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        num_buckets=100,
        salt=[137, 173],
        strong_hash=False)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[79]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_hashed_has_no_collision(self):
    # [FUNCTION DOC]
    # Hash-crosses a ten-feature dense integer column with a single value that differs between
    # two rows only above the bucket-mask bits, into 1024 buckets. Converts to dense and expects
    # every entry of the two rows to differ (no collisions).
    """Tests that fingerprint concatenation has no collisions."""
    # Although the last 10 bits of 359 and 1024+359 are identical.
    # As a result, all the crosses shouldn't collide.
    t1 = constant_op.constant([[359], [359 + 1024]], dtype=dtypes.int64)
    t2 = constant_op.constant(
        [list(range(10)), list(range(10))], dtype=dtypes.int64)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[],
        values=[],
        shapes=[],
        dense_inputs=[t2, t1],
        num_buckets=1024,
        salt=[137, 173],
        strong_hash=False)
    cross = sparse_tensor.SparseTensor(inds, vals, shapes)
    cross_dense = sparse_ops.sparse_tensor_to_dense(cross)
    with session.Session():
      values = self.evaluate(cross_dense)
      self.assertTrue(numpy.not_equal(values[0], values[1]).all())

  def test_hashed_3x1x2(self):
    # [FUNCTION DOC]
    # Hash-crosses columns of three, one and two features into 1000 buckets. Expects six entries
    # in one row with consecutive column indices, all values in range, and all distinct.
    """Tests 3x1x2 permutation with hashed output."""
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        num_buckets=1000,
        salt=[137, 173],
        strong_hash=False)
    output = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      out = self.evaluate(output)
      self.assertEqual(6, len(out.values))
      self.assertAllEqual([[0, i] for i in range(6)], out.indices)
      self.assertTrue(all(x < 1000 and x >= 0 for x in out.values))
      all_values_are_different = len(out.values) == len(set(out.values))
      self.assertTrue(all_values_are_different)

  def test_hashed_different_salt(self):
    # [FUNCTION DOC]
    # Hash-crosses the same inputs twice with different salts (one weak-hash, one strong-hash
    # call). Expects identical indices but different hashed values.
    # KNOWN ISSUE: the two calls also differ in the strong-hash flag, so the test does not
    # isolate the effect of salt alone.
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        strong_hash=False,
        num_buckets=1000,
        salt=[137, 173])
    output = sparse_tensor.SparseTensor(inds, vals, shapes)
    inds_2, vals_2, shapes_2 = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        strong_hash=True,
        num_buckets=1000,
        salt=[137, 1])
    output_2 = sparse_tensor.SparseTensor(inds_2, vals_2, shapes_2)
    with self.cached_session():
      out = self.evaluate(output)
      out_2 = self.evaluate(output_2)
      self.assertAllEqual(out.indices, out_2.indices)
      self.assertNotAllEqual(out.values, out_2.values)

  def test_sep_ignored_in_hashed_out(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, no separator is passed anywhere; the test just runs the
    # same strong-hash cross twice and checks determinism.
    # Hash-crosses the same inputs twice with identical settings (strong hash, 1000 buckets,
    # same salt). Expects identical indices and identical values.
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        strong_hash=True,
        num_buckets=1000,
        salt=[137, 173])
    output = sparse_tensor.SparseTensor(inds, vals, shapes)
    inds_2, vals_2, shapes_2 = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        strong_hash=True,
        num_buckets=1000,
        salt=[137, 173])
    output_2 = sparse_tensor.SparseTensor(inds_2, vals_2, shapes_2)
    with self.cached_session():
      out = self.evaluate(output)
      out_2 = self.evaluate(output_2)
      self.assertAllEqual(out.indices, out_2.indices)
      self.assertAllEqual(out.values, out_2.values)

  def test_sparse_cross_hashed_empty_seed(self):
    # [FUNCTION DOC]
    # Calls the raw hashed-cross kernel with all-empty input lists, a negative bucket count and
    # an empty salt. Expects an invalid-argument error saying the salt must have length two.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, r'Input "salt" must have length 2'
    ):
      indices = []
      values = []
      shapes = []
      dense_inputs = []
      num_buckets = -461
      salt = []
      strong_hash = False
      op = gen_sparse_ops.sparse_cross_hashed(
          indices=indices,
          values=values,
          shapes=shapes,
          dense_inputs=dense_inputs,
          num_buckets=num_buckets,
          salt=salt,
          strong_hash=strong_hash,
      )
      self.evaluate(op)

if __name__ == '__main__':
  test.main()
