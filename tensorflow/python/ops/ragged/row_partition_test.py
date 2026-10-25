# [MODULE DOC]
# Unit tests for the row-partition class and its spec class from the ragged package. Covers the
# private constructor, the factory methods (from value row ids, row splits, row starts, row limits,
# row lengths, uniform row length), string representation, validation errors in graph and eager
# modes, precomputed encodings and merging them, spec construction, comparison, compatibility,
# merging with specs, broadcasting, offsets, dtype conversion, copying and tracing support.
# Run through the test runner entry point at the bottom; not a helper module.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.ragged.RowPartition."""

import copy
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.platform import googletest


def _get_specified_row_partition():
  # [FUNCTION DOC]
  # HELPER: Fixture factory for the merge-with-spec tests.
  # Builds a row partition (via the private constructor key) with row splits zero, three, eight,
  # and explicitly set row count two and value count eight, all as 64-bit integers. Normally the
  # value count is not set, which is why a special builder is needed. Takes no arguments.
  """Needed for merge_with_spec tests. Normally, nvals isn't set."""
  return RowPartition(
      row_splits=constant_op.constant([0, 3, 8], dtype=dtypes.int64),
      nrows=constant_op.constant(2, dtype=dtypes.int64),
      nvals=constant_op.constant(8, dtype=dtypes.int64),
      internal=row_partition._row_partition_factory_key)


@test_util.run_all_in_graph_and_eager_modes
class RowPartitionTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for the row-partition class. Decorated to run in graph and eager modes. Tests
  # are grouped by feature (constructor, factories, string forms, validation, precomputed
  # encodings, merging, broadcasting, conversion). Stateless; uses parameterized testing.
  # =============================================================================
  # RowPartition class docstring examples
  # =============================================================================

  def testClassDocStringExamples(self):
    # [FUNCTION DOC]
    # Re-runs the examples from the class documentation: builds the same partition from row
    # splits, lengths, value row ids, starts and limits and asserts all give splits zero, four,
    # four, seven, eight, eight; also builds nested partitions (no assertion on those).
    # From section: "Component Tensors"
    rp = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(rp.row_splits(), [0, 4, 4, 7, 8, 8])
    del rp

    # From section: "Alternative Row-Partitioning Schemes"
    rt1 = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    rt2 = RowPartition.from_row_lengths(row_lengths=[4, 0, 3, 1, 0])
    rt3 = RowPartition.from_value_rowids(
        value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
    rt4 = RowPartition.from_row_starts(row_starts=[0, 4, 4, 7, 8], nvals=8)
    rt5 = RowPartition.from_row_limits(row_limits=[4, 4, 7, 8, 8])
    for rp in (rt1, rt2, rt3, rt4, rt5):
      self.assertAllEqual(rp.row_splits(), [0, 4, 4, 7, 8, 8])
    del rt1, rt2, rt3, rt4, rt5

    # From section: "Multiple Ragged Dimensions"
    inner_rt = RowPartition.from_row_splits(row_splits=[0, 4, 4, 7, 8, 8])
    outer_rt = RowPartition.from_row_splits(row_splits=[0, 3, 3, 5])
    del inner_rt, outer_rt

  # =============================================================================
  # RowPartition Constructor (private)
  # =============================================================================

  def testRowPartitionConstruction(self):
    # [FUNCTION DOC]
    # Builds a partition directly with the private constructor key and 64-bit row splits and
    # asserts the splits are returned unchanged.
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rp = RowPartition(
        row_splits=row_splits,
        internal=row_partition._row_partition_factory_key)
    self.assertAllEqual(rp.row_splits(), [0, 2, 2, 5, 6, 7])

  def testRowPartitionConstructionErrors(self):
    # [FUNCTION DOC]
    # Checks constructor errors: missing private key gives a value error saying the constructor
    # is private; a list instead of a tensor gives a type error; rank-two splits give a value
    # error about rank; a non-tensor cached row lengths gives a type error; mixed integer dtypes
    # between splits and row count give a value error about inconsistent dtype.
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    with self.assertRaisesRegex(ValueError,
                                'RowPartition constructor is private'):
      RowPartition(row_splits=row_splits)

    with self.assertRaisesRegex(TypeError,
                                'Row-partitioning argument must be a Tensor'):
      RowPartition(
          row_splits=[0, 2, 2, 5, 6, 7],
          internal=row_partition._row_partition_factory_key)

    with self.assertRaisesRegex(ValueError, r'Shape \(6, 1\) must have rank 1'):
      RowPartition(
          row_splits=array_ops.expand_dims(row_splits, 1),
          internal=row_partition._row_partition_factory_key)

    with self.assertRaisesRegex(TypeError,
                                'Cached value must be a Tensor or None.'):
      RowPartition(
          row_splits=row_splits,
          row_lengths=[2, 3, 4],
          internal=row_partition._row_partition_factory_key)

    with self.assertRaisesRegex(ValueError, 'Inconsistent dtype'):
      RowPartition(
          row_splits=constant_op.constant([0, 3], dtypes.int64),
          nrows=constant_op.constant(1, dtypes.int32),
          internal=row_partition._row_partition_factory_key)

  # =============================================================================
  # RowPartition Factory Ops
  # =============================================================================

  def testFromValueRowIdsWithDerivedNRows(self):
    # [FUNCTION DOC]
    # Factory from value row ids without a row count. Expects 64-bit dtype, the same row ids
    # object returned, derived row count five and the splits zero, two, two, five, six, seven.
    # nrows is known at graph creation time.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    # TODO(martinz): add nrows
    rp = RowPartition.from_value_rowids(value_rowids, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_splits = rp.row_splits()
    rp_value_rowids = rp.value_rowids()
    rp_nrows = rp.nrows()

    self.assertIs(rp_value_rowids, value_rowids)  # value_rowids
    self.assertAllEqual(rp_value_rowids, value_rowids)
    self.assertAllEqual(rp_nrows, 5)
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromValueRowIdsWithDerivedNRowsDynamic(self):
    # [FUNCTION DOC]
    # Same as the derived row count test but the row ids pass through a placeholder with unknown
    # shape so the count is only known at run time. Expects the same row ids object and count five.
    # nrows is not known at graph creation time.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    value_rowids = array_ops.placeholder_with_default(value_rowids, shape=None)

    rp = RowPartition.from_value_rowids(value_rowids, validate=False)

    rp_value_rowids = rp.value_rowids()
    rp_nrows = rp.nrows()

    self.assertIs(rp_value_rowids, value_rowids)  # value_rowids
    self.assertAllEqual(rp_value_rowids, value_rowids)
    self.assertAllEqual(rp_nrows, 5)

  def testFromValueRowIdsWithExplicitNRows(self):
    # [FUNCTION DOC]
    # Factory from row ids with an explicit row count of seven (larger than derived). Expects the
    # same row ids and count objects returned and splits padded with trailing repeats of seven.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(7, dtypes.int64)

    rp = RowPartition.from_value_rowids(value_rowids, nrows, validate=False)

    rp_value_rowids = rp.value_rowids()
    rp_nrows = rp.nrows()
    rp_row_splits = rp.row_splits()

    self.assertIs(rp_value_rowids, value_rowids)  # value_rowids
    self.assertIs(rp_nrows, nrows)  # nrows
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7, 7, 7])

  def testFromValueRowIdsWithExplicitNRowsEqualToDefault(self):
    # [FUNCTION DOC]
    # Factory from row ids with an explicit row count equal to the derived one (five). Expects
    # identity of the provided objects, equal values and the usual splits.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    rp = RowPartition.from_value_rowids(value_rowids, nrows, validate=False)

    rp_value_rowids = rp.value_rowids()
    rp_nrows = rp.nrows()
    rp_row_splits = rp.row_splits()

    self.assertIs(rp_value_rowids, value_rowids)  # value_rowids
    self.assertIs(rp_nrows, nrows)  # nrows
    self.assertAllEqual(rp_value_rowids, value_rowids)
    self.assertAllEqual(rp_nrows, nrows)
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromValueRowIdsWithEmptyValues(self):
    # [FUNCTION DOC]
    # Factory from an empty list of row ids. Expects 64-bit dtype, row ids of shape zero and a row
    # count of zero.
    rp = RowPartition.from_value_rowids([])
    rp_nrows = rp.nrows()
    self.assertEqual(rp.dtype, dtypes.int64)
    self.assertEqual(rp.value_rowids().shape.as_list(), [0])
    self.assertAllEqual(rp_nrows, 0)

  def testFromRowSplits(self):
    # [FUNCTION DOC]
    # Factory from a 64-bit splits tensor without validation. Expects 64-bit dtype, the same
    # splits object returned and a derived row count of five.
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    rp = RowPartition.from_row_splits(row_splits, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_splits = rp.row_splits()
    rp_nrows = rp.nrows()

    self.assertIs(rp_row_splits, row_splits)
    self.assertAllEqual(rp_nrows, 5)

  def testFromRowSplitsWithDifferentSplitTypes(self):
    # [FUNCTION DOC]
    # Builds partitions from a list, 64-bit and 32-bit arrays and 64-bit and 32-bit tensors.
    # Expects lists and 64-bit inputs to give 64-bit splits and 32-bit inputs to keep 32 bits.
    splits1 = [0, 2, 2, 5, 6, 7]
    splits2 = np.array([0, 2, 2, 5, 6, 7], np.int64)
    splits3 = np.array([0, 2, 2, 5, 6, 7], np.int32)
    splits4 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    splits5 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int32)
    rt1 = RowPartition.from_row_splits(splits1)
    rt2 = RowPartition.from_row_splits(splits2)
    rt3 = RowPartition.from_row_splits(splits3)
    rt4 = RowPartition.from_row_splits(splits4)
    rt5 = RowPartition.from_row_splits(splits5)
    self.assertEqual(rt1.row_splits().dtype, dtypes.int64)
    self.assertEqual(rt2.row_splits().dtype, dtypes.int64)
    self.assertEqual(rt3.row_splits().dtype, dtypes.int32)
    self.assertEqual(rt4.row_splits().dtype, dtypes.int64)
    self.assertEqual(rt5.row_splits().dtype, dtypes.int32)

  def testFromRowSplitsWithEmptySplits(self):
    # [FUNCTION DOC]
    # Passing an empty splits list must raise a value error stating splits may not be empty.
    err_msg = 'row_splits tensor may not be empty'
    with self.assertRaisesRegex(ValueError, err_msg):
      RowPartition.from_row_splits([])

  def testFromRowStarts(self):
    # [FUNCTION DOC]
    # Factory from row starts and a value count of seven (no validation). Expects 64-bit dtype,
    # five rows, the starts returned as given and splits with the count appended.
    nvals = constant_op.constant(7)
    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)

    rp = RowPartition.from_row_starts(row_starts, nvals, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_starts = rp.row_starts()
    rp_row_splits = rp.row_splits()
    rp_nrows = rp.nrows()

    self.assertAllEqual(rp_nrows, 5)
    self.assertAllEqual(rp_row_starts, row_starts)
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromRowLimits(self):
    # [FUNCTION DOC]
    # Factory from row limits (no validation). Expects 64-bit dtype, five rows, limits returned
    # as given and splits with a leading zero.
    row_limits = constant_op.constant([2, 2, 5, 6, 7], dtypes.int64)

    rp = RowPartition.from_row_limits(row_limits, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_limits = rp.row_limits()
    rp_row_splits = rp.row_splits()
    rp_nrows = rp.nrows()

    self.assertAllEqual(rp_nrows, 5)
    self.assertAllEqual(rp_row_limits, row_limits)
    self.assertAllEqual(rp_row_splits, [0, 2, 2, 5, 6, 7])

  def testFromRowLengths(self):
    # [FUNCTION DOC]
    # Factory from row lengths (no validation). Expects 64-bit dtype, the same lengths object
    # returned and a derived row count of five.
    row_lengths = constant_op.constant([2, 0, 3, 1, 1], dtypes.int64)

    rp = RowPartition.from_row_lengths(row_lengths, validate=False)
    self.assertEqual(rp.dtype, dtypes.int64)

    rp_row_lengths = rp.row_lengths()
    rp_nrows = rp.nrows()

    self.assertIs(rp_row_lengths, row_lengths)  # nrows
    self.assertAllEqual(rp_nrows, 5)
    self.assertAllEqual(rp_row_lengths, row_lengths)

  def testFromUniformRowLength(self):
    # [FUNCTION DOC]
    # Factory from a value count of sixteen and row length two. Expects the uniform row length two
    # and a derived row count of eight.
    nvals = 16
    a1 = RowPartition.from_uniform_row_length(
        nvals=nvals, uniform_row_length=2)
    self.assertAllEqual(a1.uniform_row_length(), 2)
    self.assertAllEqual(a1.nrows(), 8)

  def testFromUniformRowLengthWithEmptyValues(self):
    # [FUNCTION DOC]
    # Uniform row length zero with zero values and ten rows. Expects value count zero and row
    # count ten after evaluation.
    a = RowPartition.from_uniform_row_length(
        nvals=0, uniform_row_length=0, nrows=10)
    self.assertEqual(self.evaluate(a.nvals()), 0)
    self.assertEqual(self.evaluate(a.nrows()), 10)

  def testFromUniformRowLengthWithPlaceholders1(self):
    # [FUNCTION DOC]
    # Value count supplied through a placeholder of unknown shape, row length three. Expects the
    # evaluated value count to be six.
    nvals = array_ops.placeholder_with_default(
        constant_op.constant(6, dtype=dtypes.int64), None)
    rt1 = RowPartition.from_uniform_row_length(
        nvals=nvals, uniform_row_length=3)
    const_nvals1 = self.evaluate(rt1.nvals())
    self.assertEqual(const_nvals1, 6)

  def testFromUniformRowLengthWithPlaceholders2(self):
    # [FUNCTION DOC]
    # Both value count and row length come from placeholders. Expects the evaluated value count
    # to be six.
    nvals = array_ops.placeholder_with_default(6, None)
    ph_rowlen = array_ops.placeholder_with_default(3, None)
    rt2 = RowPartition.from_uniform_row_length(
        nvals=nvals, uniform_row_length=ph_rowlen)
    const_nvals2 = self.evaluate(rt2.nvals())
    self.assertEqual(const_nvals2, 6)

  def testFromValueRowIdsWithBadNRows(self):
    # [FUNCTION DOC]
    # Invalid row count arguments to the row-ids factory must raise value errors: negative count,
    # count smaller than last id plus one (two cases, including four), rank-two row ids and a
    # rank-one row count. Messages are matched against expected text.
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    with self.assertRaisesRegex(ValueError, r'Expected nrows >= 0; got -2'):
      RowPartition.from_value_rowids(
          value_rowids=array_ops.placeholder_with_default(value_rowids, None),
          nrows=-2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=2, '
        r'value_rowids\[-1\]=4'):
      RowPartition.from_value_rowids(value_rowids=value_rowids, nrows=2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=4, '
        r'value_rowids\[-1\]=4'):
      RowPartition.from_value_rowids(value_rowids=value_rowids, nrows=4)

    with self.assertRaisesRegex(ValueError, r'Shape \(7, 1\) must have rank 1'):
      RowPartition.from_value_rowids(
          value_rowids=array_ops.expand_dims(value_rowids, 1), nrows=nrows)

    with self.assertRaisesRegex(ValueError, r'Shape \(1,\) must have rank 0'):
      RowPartition.from_value_rowids(
          value_rowids=value_rowids, nrows=array_ops.expand_dims(nrows, 0))

  # =============================================================================
  # RowPartition.__str__
  # =============================================================================
  def testRowPartitionStr(self):
    # [FUNCTION DOC]
    # Checks the text and debug representation of a partition built from row splits. In eager mode
    # it shows the splits values; in graph mode it shows the named symbolic tensor. Both forms
    # must be equal to the expected text.
    row_splits = [0, 2, 5, 6, 6, 7]
    rp = RowPartition.from_row_splits(row_splits, validate=False)
    if context.executing_eagerly():
      expected_repr = 'tf.RowPartition(row_splits=[0 2 5 6 6 7])'
    else:
      expected_repr = ('tf.RowPartition(row_splits='
                       'Tensor("RowPartitionFromRowSplits/row_splits:0", '
                       'shape=(6,), dtype=int64))')
    self.assertEqual(repr(rp), expected_repr)
    self.assertEqual(str(rp), expected_repr)

  def testRowPartitionStrUniformRowLength(self):
    # [FUNCTION DOC]
    # Same representation check for a uniform-row-length partition: shows row count and uniform
    # row length (values in eager mode, symbolic tensors in graph mode).
    rp = RowPartition.from_uniform_row_length(5, nvals=10, nrows=2)
    if context.executing_eagerly():
      expected_repr = ('tf.RowPartition(nrows=2, uniform_row_length=5)')
    else:
      expected_repr = (
          'tf.RowPartition(nrows='
          'Tensor("RowPartitionFromUniformRowLength/'
          'nrows:0", shape=(), dtype=int64), '
          'uniform_row_length=Tensor("RowPartitionFromUniformRowLength/'
          'uniform_row_length:0", shape=(), dtype=int64))')
    self.assertEqual(repr(rp), expected_repr)
    self.assertEqual(str(rp), expected_repr)

  @parameterized.parameters([
      # from_value_rowids
      {
          'descr': 'bad rank for value_rowids',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [[1, 2], [3, 4]],
          'nrows': 10,
      },
      {
          'descr': 'bad rank for nrows',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [1, 2, 3, 4],
          'nrows': [10],
      },
      {
          'descr': 'negative value_rowid',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [-5, 2, 3, 4],
          'nrows': 10,
      },
      {
          'descr': 'non-monotonic-increasing value_rowid',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [4, 3, 2, 1],
          'nrows': 10,
      },
      {
          'descr': 'value_rowid > nrows',
          'factory': RowPartition.from_value_rowids,
          'value_rowids': [1, 2, 3, 4],
          'nrows': 2,
      },
      # from_row_splits
      {
          'descr': 'bad rank for row_splits',
          'factory': RowPartition.from_row_splits,
          'row_splits': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_splits[0] != 0',
          'factory': RowPartition.from_row_splits,
          'row_splits': [2, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_splits',
          'factory': RowPartition.from_row_splits,
          'row_splits': [0, 3, 2, 4],
      },
      # from_row_lengths
      {
          'descr': 'bad rank for row_lengths',
          'factory': RowPartition.from_row_lengths,
          'row_lengths': [[1, 2], [1, 0]],
      },
      {
          'descr': 'negative row_lengths',
          'factory': RowPartition.from_row_lengths,
          'row_lengths': [3, -1, 2],
      },
      # from_row_starts
      {
          'descr': 'bad rank for row_starts',
          'factory': RowPartition.from_row_starts,
          'nvals': 2,
          'row_starts': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_starts[0] != 0',
          'factory': RowPartition.from_row_starts,
          'nvals': 5,
          'row_starts': [2, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_starts',
          'factory': RowPartition.from_row_starts,
          'nvals': 4,
          'row_starts': [0, 3, 2, 4],
      },
      {
          'descr': 'row_starts[0] > nvals',
          'factory': RowPartition.from_row_starts,
          'nvals': 4,
          'row_starts': [0, 2, 3, 5],
      },
      # from_row_limits
      {
          'descr': 'bad rank for row_limits',
          'factory': RowPartition.from_row_limits,
          'row_limits': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_limits[0] < 0',
          'factory': RowPartition.from_row_limits,
          'row_limits': [-1, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_limits',
          'factory': RowPartition.from_row_limits,
          'row_limits': [0, 3, 2, 4],
      },
      # from_uniform_row_length
      {
          'descr': 'rowlen * nrows != nvals (1)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 5,
          'uniform_row_length': 3,
      },
      {
          'descr': 'rowlen * nrows != nvals (2)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 5,
          'uniform_row_length': 6,
      },
      {
          'descr': 'rowlen * nrows != nvals (3)',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 6,
          'uniform_row_length': 3,
          'nrows': 3,
      },
      {
          'descr': 'rowlen must be a scalar',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 4,
          'uniform_row_length': [2],
      },
      {
          'descr': 'rowlen must be nonnegative',
          'factory': RowPartition.from_uniform_row_length,
          'nvals': 4,
          'uniform_row_length': -1,
      },
  ])
  def testFactoryValidation(self, descr, factory, **kwargs):
    # [FUNCTION DOC]
    # Parameterized over invalid inputs for each factory (descr is only a label). The factory is
    # called with the keyword arguments and splits evaluated; expects an invalid-argument or value
    # error (static detection). In graph mode it repeats with all arguments wrapped in
    # unknown-shape placeholders and expects an invalid-argument error at run time.
    # When input tensors have shape information, some of these errors will be
    # detected statically.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      partition = factory(**kwargs)
      self.evaluate(partition.row_splits())

    # Remove shape information (by wrapping tensors in placeholders), and check
    # that we detect the errors when the graph is run.
    if not context.executing_eagerly():

      def wrap_arg(v):
        # [FUNCTION DOC]
        # HELPER: Strips static shape information in the validation test.
        # Converts a value to a 64-bit constant and returns a placeholder with that default and
        # fully unknown shape.
        return array_ops.placeholder_with_default(
            constant_op.constant(v, dtype=dtypes.int64),
            tensor_shape.TensorShape(None))

      kwargs = dict((k, wrap_arg(v)) for (k, v) in kwargs.items())

      with self.assertRaises(errors.InvalidArgumentError):
        partition = factory(**kwargs)
        self.evaluate(partition.row_splits())

  @parameterized.named_parameters([
      ('FromRowSplits', lambda: RowPartition.from_row_splits([0, 2, 8]),
       ['row_splits']),
      ('FromRowLengths', lambda: RowPartition.from_row_lengths([3, 0, 8]),
       ['row_splits', 'row_lengths']),
      ('FromValueRowIds',
       lambda: RowPartition.from_value_rowids([0, 0, 3, 4, 4, 4]),
       ['row_splits', 'value_rowids', 'row_lengths', 'nrows']),
      ('FromRowStarts',
       lambda: RowPartition.from_row_starts([0, 3, 7], nvals=10),
       ['row_splits']),
      ('FromRowLimits', lambda: RowPartition.from_row_limits([3, 7, 10]),
       ['row_splits']),
  ])
  def testPrecomputedSplits(self, rp_factory, expected_encodings):
    # [FUNCTION DOC]
    # Parameterized over each factory. Builds a partition and checks which encodings (splits,
    # lengths, row ids, row count) are cached right after construction against the expected list.
    rp = rp_factory()
    self.assertEqual(rp._has_precomputed_row_splits(),
                     'row_splits' in expected_encodings)
    self.assertEqual(rp._has_precomputed_row_lengths(),
                     'row_lengths' in expected_encodings)
    self.assertEqual(rp._has_precomputed_value_rowids(),
                     'value_rowids' in expected_encodings)
    self.assertEqual(rp._has_precomputed_nrows(), 'nrows' in expected_encodings)

  def testWithPrecomputedSplits(self):
    # [FUNCTION DOC]
    # From a splits-only partition, requests copies with row splits, lengths, row ids, row count
    # and value count precomputed. Asserts the original lacks each cache and each copy has it.
    rp = RowPartition.from_row_splits([0, 2, 8])

    rp_with_row_splits = rp._with_precomputed_row_splits()
    self.assertTrue(rp_with_row_splits._has_precomputed_row_splits())

    self.assertFalse(rp._has_precomputed_row_lengths())
    rp_with_row_lengths = rp._with_precomputed_row_lengths()
    self.assertTrue(rp_with_row_lengths._has_precomputed_row_lengths())

    self.assertFalse(rp._has_precomputed_value_rowids())
    rp_with_value_rowids = rp._with_precomputed_value_rowids()
    self.assertTrue(rp_with_value_rowids._has_precomputed_value_rowids())

    self.assertFalse(rp._has_precomputed_nrows())
    rp_with_nrows = rp._with_precomputed_nrows()
    self.assertTrue(rp_with_nrows._has_precomputed_nrows())

    self.assertFalse(rp._has_precomputed_nvals())
    rp_with_nvals = rp._with_precomputed_nvals()
    self.assertTrue(rp_with_nvals._has_precomputed_nvals())

  @parameterized.named_parameters([
      dict(
          testcase_name='FromRowSplitsAndRowSplits',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_splits([0, 3, 8]),
          expected_encodings=['row_splits']),
      dict(
          testcase_name='FromRowSplitsAndUniformRowLength',
          x=lambda: RowPartition.from_row_splits([0, 3, 6]),
          y=lambda: RowPartition.from_uniform_row_length(3, nvals=6),
          expected_encodings=['row_splits', 'uniform_row_length', 'nrows']),
      dict(
          testcase_name='FromRowSplitsAndRowLengths',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_lengths([3, 5]),
          expected_encodings=['row_splits', 'row_lengths']),
      dict(
          testcase_name='FromRowSplitsAndValueRowIds',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_value_rowids([0, 0, 0, 1, 1, 1, 1, 1]),
          expected_encodings=[
              'row_splits', 'row_lengths', 'value_rowids', 'nrows'
          ]),
      dict(
          testcase_name='FromRowSplitsAndRowSplitsPlusNRows',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_splits([0, 3, 8]).
          _with_precomputed_nrows(),
          expected_encodings=['row_splits', 'nrows']),
  ])
  def testMergePrecomputedEncodings(self, x, y, expected_encodings):
    # [FUNCTION DOC]
    # Parameterized with two partition builders and the expected set of cached encodings of the
    # merge. For validation on and off, merges the cached encodings of y into x and asserts which
    # encodings the result caches, and that wherever both a source and the result cache an
    # encoding their values are equal.
    x = x()
    y = y()
    for validate in (True, False):
      result = x._merge_precomputed_encodings(y, validate)
      self.assertEqual(result._has_precomputed_row_splits(),
                       'row_splits' in expected_encodings)
      self.assertEqual(result._has_precomputed_row_lengths(),
                       'row_lengths' in expected_encodings)
      self.assertEqual(result._has_precomputed_value_rowids(),
                       'value_rowids' in expected_encodings)
      self.assertEqual(result._has_precomputed_nrows(),
                       'nrows' in expected_encodings)
      self.assertEqual(result.uniform_row_length() is not None,
                       'uniform_row_length' in expected_encodings)
      for r in (x, y):
        if (r._has_precomputed_row_splits() and
            result._has_precomputed_row_splits()):
          self.assertAllEqual(r.row_splits(), result.row_splits())
        if (r._has_precomputed_row_lengths() and
            result._has_precomputed_row_lengths()):
          self.assertAllEqual(r.row_lengths(), result.row_lengths())
        if (r._has_precomputed_value_rowids() and
            result._has_precomputed_value_rowids()):
          self.assertAllEqual(r.value_rowids(), result.value_rowids())
        if r._has_precomputed_nrows() and result._has_precomputed_nrows():
          self.assertAllEqual(r.nrows(), result.nrows())
        if (r.uniform_row_length() is not None and
            result.uniform_row_length() is not None):
          self.assertAllEqual(r.uniform_row_length(),
                              result.uniform_row_length())

  def testMergePrecomputedEncodingsFastPaths(self):
    # [FUNCTION DOC]
    # Merging a partition with itself, or with one that shares the same splits tensor, must
    # return the first partition object unchanged (identity, not just equality).
    # Same object: x gets returned as-is.
    x = RowPartition.from_row_splits([0, 3, 8, 8])
    self.assertIs(x._merge_precomputed_encodings(x), x)

    # Same encoding tensor objects: x gets returned as-is.
    y = RowPartition.from_row_splits(x.row_splits(), validate=False)
    self.assertIs(x._merge_precomputed_encodings(y), x)

  def testMergePrecomputedEncodingsWithMatchingTensors(self):
    # [FUNCTION DOC]
    # A partition built from row ids has a superset of the cached encodings of one built from its
    # splits, sharing the splits tensor. Merging in either order must return the richer
    # partition object itself, and the two partitions must be distinct objects.
    # The encoding tensors for `a` are a superset of the encoding tensors
    # for `b`, and where they overlap, they the same tensor objects.
    a = RowPartition.from_value_rowids([0, 0, 3, 4, 4, 4])
    b = RowPartition.from_row_splits(a.row_splits(), validate=False)
    self.assertIs(a._merge_precomputed_encodings(b), a)
    self.assertIs(b._merge_precomputed_encodings(a), a)
    self.assertIsNot(a, b)

  @parameterized.named_parameters([
      dict(
          testcase_name='RowSplitMismatch',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_splits([0, 3, 8, 9]),
          message='incompatible row_splits'),
      dict(
          testcase_name='RowLengthMismatch',
          x=lambda: RowPartition.from_row_lengths([2, 0, 2]),
          y=lambda: RowPartition.from_row_lengths([2, 0, 2, 1]),
          message='incompatible row_splits'),  # row_splits is checked first
      dict(
          testcase_name='ValueRowIdMismatch',
          x=lambda: RowPartition.from_value_rowids([0, 3, 3, 4]),
          y=lambda: RowPartition.from_value_rowids([0, 3, 4]),
          message='incompatible value_rowids'),
  ])
  def testMergePrecomputedEncodingStaticErrors(self, x, y, message):
    # [FUNCTION DOC]
    # Parameterized with two incompatible partitions and an expected message. Skipped in eager
    # mode (returns immediately). In graph mode merging in either order must raise a value error
    # matching the message when splits are requested, because static shapes already disagree.
    if context.executing_eagerly():
      return
    # Errors that are caught by static shape checks.
    x = x()
    y = y()
    with self.assertRaisesRegex(ValueError, message):
      x._merge_precomputed_encodings(y).row_splits()
    with self.assertRaisesRegex(ValueError, message):
      y._merge_precomputed_encodings(x).row_splits()

  @parameterized.named_parameters([
      dict(
          testcase_name='NRowsMismatchAlt',
          x=lambda: RowPartition.from_uniform_row_length(5, nrows=4, nvals=20),
          y=lambda: RowPartition.from_uniform_row_length(5, nrows=3, nvals=15),
          message='incompatible nrows'),
      dict(
          testcase_name='UniformRowLengthMismatch',
          x=lambda: RowPartition.from_uniform_row_length(5, nvals=20),
          y=lambda: RowPartition.from_uniform_row_length(2, nvals=8),
          message='incompatible (nvals|uniform_row_length)'),
      dict(
          testcase_name='RowSplitMismatch',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          y=lambda: RowPartition.from_row_splits([0, 5, 8]),
          message='incompatible row_splits'),
      dict(
          testcase_name='RowLengthMismatch',
          x=lambda: RowPartition.from_row_lengths([2, 0, 2]),
          y=lambda: RowPartition.from_row_lengths([0, 0, 2]),
          message='incompatible (row_splits|nvals)'),
      dict(
          testcase_name='ValueRowIdMismatch',
          x=lambda: RowPartition.from_value_rowids([0, 3, 3]),
          y=lambda: RowPartition.from_value_rowids([0, 0, 3]),
          message='incompatible row_splits'),  # row_splits is checked first
  ])
  def testMergePrecomputedEncodingRuntimeErrors(self, x, y, message):
    # [FUNCTION DOC]
    # Parameterized with two partitions whose values conflict. Merging in either order and
    # evaluating the splits must raise an invalid-argument error matching the message (detected
    # by runtime value checks, not static shapes).
    # Errors that are caught by runtime value checks.
    x = x()
    y = y()
    with self.assertRaisesRegex(errors.InvalidArgumentError, message):
      self.evaluate(x._merge_precomputed_encodings(y).row_splits())
    with self.assertRaisesRegex(errors.InvalidArgumentError, message):
      self.evaluate(y._merge_precomputed_encodings(x).row_splits())

  @parameterized.named_parameters([
      # It throws the right error, but it still complains.
      dict(
          testcase_name='NRowsMismatch',
          x=lambda: RowPartition.from_uniform_row_length(5, nvals=20),
          y=lambda: RowPartition.from_uniform_row_length(5, nvals=15),
          message='incompatible nvals',
          emessage='incompatible nrows'),
  ])
  def testMergePrecomputedEncodingStaticErrors2(self, x, y,
                                                message, emessage):
    # [FUNCTION DOC]
    # KNOWN ISSUE: named "static errors" but it expects a runtime invalid-argument error (in both
    # modes); only the message differs: the eager message is used when executing eagerly,
    # otherwise the graph message.
    # Merges two uniform-row-length partitions with the same row length but different value
    # counts, in both orders, evaluates the splits and asserts the error type and message.
    # Message error and type varies depending upon eager execution.
    x = x()
    y = y()

    error_type = errors_impl.InvalidArgumentError
    expected_message = emessage if context.executing_eagerly() else message
    with self.assertRaisesRegex(error_type, expected_message):
      self.evaluate(x._merge_precomputed_encodings(y).row_splits())
    with self.assertRaisesRegex(error_type, expected_message):
      self.evaluate(y._merge_precomputed_encodings(x).row_splits())

  @parameterized.named_parameters([
      dict(
          testcase_name='NoneSpecified',
          rp=(lambda: RowPartition.from_row_splits([0, 3, 8])),
          spec=RowPartitionSpec(nrows=None, nvals=None, dtype=dtypes.int64)),
      dict(
          testcase_name='NRowsSpecified',
          rp=(lambda: RowPartition.from_row_splits([0, 3, 8])),
          spec=RowPartitionSpec(nrows=2, nvals=None, dtype=dtypes.int64)),
      dict(
          testcase_name='NValsSpecified',
          rp=_get_specified_row_partition,
          spec=RowPartitionSpec(nrows=None, nvals=8, dtype=dtypes.int64))
  ])
  def testMergeWithSpecNoop(self, rp, spec):
    # [FUNCTION DOC]
    # Merging a partition with a spec that adds no new information (nothing specified, or only
    # values already known) must leave splits, static row count and static value count unchanged.
    rp = rp()
    actual = rp._merge_with_spec(spec)
    self.assertAllEqual(actual.row_splits(), rp.row_splits())
    self.assertAllEqual(actual.static_nrows, rp.static_nrows)
    self.assertAllEqual(actual.static_nvals, rp.static_nvals)

  @parameterized.named_parameters([
      dict(
          testcase_name='NRowsNValsUpdated',
          rp=(lambda: RowPartition.from_row_splits([0, 3, 8])),
          spec=RowPartitionSpec(nrows=2, nvals=8, dtype=dtypes.int64),
          expected=_get_specified_row_partition),
      dict(
          testcase_name='NValsUpdated',
          rp=(lambda: RowPartition.from_row_splits([0, 3, 8])),
          spec=RowPartitionSpec(nrows=None, nvals=8, dtype=dtypes.int64),
          expected=_get_specified_row_partition)])
  def testMergeWithSpecUpdate(self, rp, spec, expected):
    # [FUNCTION DOC]
    # Merging a splits-only partition with a spec giving row count and/or value count must yield
    # a partition whose splits and static row and value counts match the expected partition with
    # those counts set.
    rp = rp()
    expected = expected()
    actual = rp._merge_with_spec(spec)
    self.assertAllEqual(actual.row_splits(), expected.row_splits())
    self.assertAllEqual(actual.static_nrows, expected.static_nrows)
    self.assertAllEqual(actual.static_nvals, expected.static_nvals)

  @parameterized.named_parameters([
      dict(
          testcase_name='from_uniform_row_length',
          x=lambda: RowPartition.from_uniform_row_length(5, nvals=20),
          expected=True),
      dict(
          testcase_name='from_row_splits',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          expected=False),
      dict(
          testcase_name='from_row_lengths',
          x=lambda: RowPartition.from_row_lengths([2, 0, 2]),
          expected=False),
      dict(
          testcase_name='from_row_lengths_uniform',
          x=lambda: RowPartition.from_row_lengths([3, 3, 3]),
          expected=False),
  ])
  def testIsUniform(self, x, expected):
    # [FUNCTION DOC]
    # Asserts the uniformity predicate is true only for the uniform-row-length factory; partitions
    # from splits or lengths (even equal lengths) are not reported as uniform.
    x = x()
    self.assertEqual(expected, x.is_uniform())

  @parameterized.named_parameters([
      dict(
          testcase_name='doc_example',
          x=lambda: RowPartition.from_row_lengths([3, 2, 0, 2]),
          expected=[0, 1, 2, 0, 1, 0, 1]),
      dict(
          testcase_name='from_uniform_row_length',
          x=lambda: RowPartition.from_uniform_row_length(4, nvals=12),
          expected=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
      dict(
          testcase_name='from_row_splits',
          x=lambda: RowPartition.from_row_splits([0, 3, 8]),
          expected=[0, 1, 2, 0, 1, 2, 3, 4]),
  ])
  def testOffsetsInRows(self, x, expected):
    # [FUNCTION DOC]
    # Asserts the offsets-in-rows result (position of each value within its own row) equals the
    # expected list for partitions from lengths, uniform row length and splits.
    x = x()
    actual = x.offsets_in_rows()
    self.assertAllEqual(expected, actual)

  def testFromUniformRowLengthBugConvertToTensor(self):
    # [FUNCTION DOC]
    # Regression test: a 32-bit row count combined with 64-bit value count and row length and an
    # explicit 64-bit dtype used to fail. Asserts the resulting row count tensor is 64-bit.
    # This originally failed to run because nrows was dtypes.int32. I think
    # we may need to consider the semantics of the type of a RowPartition
    # if preferred_dtype is unspecified. Also, looking at convert_to_tensor:
    # dtype specifies the type of the output.
    # preferred_dtype/dtype_hint is a suggestion, and dtype_hint is the new
    # name.
    nrows = constant_op.constant(3, dtype=dtypes.int32)
    nvals = constant_op.constant(12, dtype=dtypes.int64)
    row_length = constant_op.constant(4, dtype=dtypes.int64)
    rp = RowPartition.from_uniform_row_length(row_length, nvals=nvals,
                                              nrows=nrows, dtype=dtypes.int64)
    self.assertEqual(rp.nrows().dtype, dtypes.int64)

  def testFromUniformRowLengthNvalDynamic(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead test; the body is entirely commented out and ends with a no-op, so it
    # asserts nothing (pending a fix for dynamic value counts).
    # Intended scenario: a traced function with a dynamic value count, row length twelve and
    # three rows should statically resolve the value count to thirty-six.
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically, should we determine nvals?
    # TODO(martinz): Uncomment after nvals is fixed.
    # @def_function.function(
    #     input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    # def foo(nvals):
    #   rp = RowPartition.from_uniform_row_length(12, nvals=nvals, nrows=3)
    #   nval_output = tensor_util.constant_value(rp.nvals())
    #   self.assertEqual(nval_output, 36)
    # foo(constant_op.constant(36, dtype=dtypes.int32))
    pass

  def testFromUniformRowLengthNvalDynamicNoValidate(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead test; the body is entirely commented out and ends with a no-op, so it
    # asserts nothing.
    # Intended scenario: same as the dynamic value count test but with validation disabled.
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically, should we determine nvals?
    # TODO(martinz): Uncomment after nvals is fixed.
    # @def_function.function(
    #     input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    # def foo(nvals):
    #   rp = RowPartition.from_uniform_row_length(12, nvals=nvals, nrows=3,
    #                                             validate=False)
    #   nval_output = tensor_util.constant_value(rp.nvals())
    #   self.assertEqual(nval_output, 36)
    # foo(constant_op.constant(36, dtype=dtypes.int32))
    pass

  def testFromUniformRowLengthNvalDynamicWrong(self):
    # [FUNCTION DOC]
    # Traces a function taking a dynamic value count (seven) with row length twelve and three
    # rows (inconsistent) and returning the value count. Asserts calling it and evaluating the
    # result raises an invalid-argument error.
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically and WRONG,
    # what should we do? We add a check, but checks are only checked for
    # row_splits.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(nvals):
      # [FUNCTION DOC]
      # Traced helper: builds a uniform-row-length partition with row length twelve, three rows
      # and the given dynamic value count, and returns its value count tensor.
      rp = RowPartition.from_uniform_row_length(12, nvals=nvals, nrows=3)
      return rp.nvals()

    with self.assertRaises(errors.InvalidArgumentError):
      nvals = foo(constant_op.constant(7, dtype=dtypes.int32))
      self.evaluate(nvals)

  def testFromUniformRowLengthNvalDynamicWrongRowSplits(self):
    # [FUNCTION DOC]
    # Same inconsistent setup as the previous test (value count seven against twelve by three),
    # but the traced function returns the row splits. Asserts evaluating them raises an
    # invalid-argument error, since the consistency check is attached to the splits.
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically and WRONG,
    # what should we do?
    # A key question is whether if nrows and uniform_row_length are known,
    # and nvals is given but not known statically and WRONG,
    # what should we do? We add a check, but checks are only checked for
    # row_splits.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(nvals):
      # [FUNCTION DOC]
      # Traced helper: builds a uniform-row-length partition with row length twelve, three rows
      # and the given dynamic value count, and returns its row splits tensor.
      rp = RowPartition.from_uniform_row_length(12, nvals=nvals, nrows=3)
      return rp.row_splits()

    with self.assertRaises(errors.InvalidArgumentError):
      rs = foo(constant_op.constant(7, dtype=dtypes.int32))
      self.evaluate(rs)

  def testFromUniformRowPartitionNrows(self):
    # [FUNCTION DOC]
    # Uniform row length three with four rows and no value count. Expects row count four, row
    # length three and a static value count of twelve.
    rp = RowPartition.from_uniform_row_length(3, nrows=4)
    self.assertAllEqual(4, rp.nrows())
    self.assertAllEqual(3, rp.uniform_row_length())
    self.assertAllEqual(12, rp.static_nvals)

  def testFromUniformRowPartitionNvalsStatic(self):
    # [FUNCTION DOC]
    # Uniform row length three with twelve values. Expects static row count four, static row
    # length three and static value count twelve.
    rp = RowPartition.from_uniform_row_length(3, nvals=12)
    self.assertAllEqual(4, rp.static_nrows)
    self.assertAllEqual(3, rp.static_uniform_row_length)
    self.assertAllEqual(12, rp.static_nvals)

  def testFromUniformRowPartitionNvalsStaticNoValidate(self):
    # [FUNCTION DOC]
    # Same static-value checks as the previous test, with all of row length, row count and value
    # count given and validation disabled. Expects the static values four, three and twelve.
    rp = RowPartition.from_uniform_row_length(3, nrows=4, nvals=12,
                                              validate=False)
    self.assertAllEqual(4, rp.static_nrows)
    self.assertAllEqual(3, rp.static_uniform_row_length)
    self.assertAllEqual(12, rp.static_nvals)

  def testFromUniformRowPartitionNvalsIs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead test; the body is commented out and does nothing.
    # Intended scenario: the value count tensor passed to the factory should be returned
    # unchanged (same object) by the value count accessor.
    # TODO(martinz): Uncomment after nvals is fixed.
    # nvals = constant_op.constant(12)
    # rp = RowPartition.from_uniform_row_length(3, nvals=nvals)
    # self.assertIs(rp.nvals(), nvals)
    pass

  def testFromUniformRowPartitionRowStartsStatic(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: name mentions uniform row partition but the test uses the row-starts factory.
    # Builds a partition from row starts and a value count of twelve and asserts the static value
    # count is twelve.
    rp = RowPartition.from_row_starts([0, 3, 6], nvals=12)
    self.assertAllEqual(12, rp.static_nvals)

  def testStaticNrows(self):
    # [FUNCTION DOC]
    # Partition from splits zero, three, four, five. Asserts the static row count is a plain
    # integer equal to three.
    rp = RowPartition.from_row_splits([0, 3, 4, 5])
    static_nrows = rp.static_nrows
    self.assertIsInstance(static_nrows, int)
    self.assertAllEqual(3, static_nrows)

  def testStaticNrowsUnknown(self):
    # [FUNCTION DOC]
    # Traces a function whose splits input has unknown shape and asserts inside it that the
    # static row count is nothing/unknown, then calls it with concrete 32-bit splits.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def foo(rs):
      # [FUNCTION DOC]
      # Traced helper: builds a partition from the unknown-shape splits argument and asserts its
      # static row count is nothing. Returns nothing.
      rp = RowPartition.from_row_splits(rs)
      static_nrows = rp.static_nrows
      self.assertIsNone(static_nrows)
    foo(array_ops.constant([0, 3, 4, 5], dtype=dtypes.int32))


@test_util.run_all_in_graph_and_eager_modes
class RowPartitionSpecTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the row-partition type specification class: construction and validation,
  # copying, serialization, component specs, compatibility, most-specific-compatible-type,
  # creation from a value, dtype change and merging. Runs in graph and eager modes; stateless.

  def testDefaultConstruction(self):
    # [FUNCTION DOC]
    # A default spec has no row count, value count or uniform row length and 64-bit dtype.
    spec = RowPartitionSpec()
    self.assertIsNone(spec.nrows)
    self.assertIsNone(spec.nvals)
    self.assertIsNone(spec.uniform_row_length)
    self.assertEqual(spec.dtype, dtypes.int64)

  @parameterized.parameters([
      (None, None, None, dtypes.int64, None, None, None, dtypes.int64),
      (5, None, None, dtypes.int32, 5, None, None, dtypes.int32),
      (None, 20, None, dtypes.int64, None, 20, None, dtypes.int64),
      (None, None, 8, dtypes.int64, None, None, 8, dtypes.int64),
      (5, None, 8, dtypes.int64, 5, 40, 8, dtypes.int64),  # nvals inferred
      (None, 20, 4, dtypes.int32, 5, 20, 4, dtypes.int32),  # nrows inferred
      (0, None, None, dtypes.int32, 0, 0, None, dtypes.int32),  # nvals inferred
      (None, None, 0, dtypes.int32, None, 0, 0, dtypes.int32),  # nvals inferred
  ])  # pyformat: disable
  def testConstruction(self, nrows, nvals, uniform_row_length, dtype,
                       expected_nrows, expected_nvals,
                       expected_uniform_row_length, expected_dtype):
    # [FUNCTION DOC]
    # Parameterized construction test. Builds a spec from row count, value count, uniform row
    # length and dtype, and asserts the stored attributes equal the expected ones, including
    # values inferred from the others (value count from rows times length, row count from value
    # count divided by length).
    spec = RowPartitionSpec(nrows, nvals, uniform_row_length, dtype)
    self.assertEqual(spec.nrows, expected_nrows)
    self.assertEqual(spec.nvals, expected_nvals)
    self.assertEqual(spec.uniform_row_length, expected_uniform_row_length)
    self.assertEqual(spec.dtype, expected_dtype)

  @parameterized.parameters([
      dict(dtype=dtypes.float32, error='dtype must be tf.int32 or tf.int64'),
      dict(nrows=0, nvals=5, error='.* not compatible .*'),
      dict(uniform_row_length=0, nvals=5, error='.* not compatible .*'),
      dict(nvals=11, uniform_row_length=5, error='.* not compatible .*'),
      dict(
          nrows=8, nvals=10, uniform_row_length=5,
          error='.* not compatible .*'),
  ])
  def testConstructionError(self,
                            nrows=None,
                            nvals=None,
                            uniform_row_length=None,
                            dtype=dtypes.int64,
                            error=None):
    # [FUNCTION DOC]
    # Parameterized over inconsistent spec arguments (non-integer dtype, or mutually incompatible
    # row count, value count and uniform row length). Constructing the spec must raise a value
    # error whose message matches the given pattern.
    with self.assertRaisesRegex(ValueError, error):
      RowPartitionSpec(nrows, nvals, uniform_row_length, dtype)

  @parameterized.parameters([
      RowPartitionSpec(),
      RowPartitionSpec(dtype=dtypes.int64),
      RowPartitionSpec(uniform_row_length=3)
  ])  # pyformat: disable

  def testDeepcopy(self, spec):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parameterized spec argument is immediately overwritten by a default spec,
    # so only the default spec is ever tested; a blank line also separates the decorator from
    # the definition.
    # Deep-copies a spec and asserts the debug representation of the copy equals the original.
    spec = RowPartitionSpec()
    spec_b = copy.deepcopy(spec)
    self.assertEqual(repr(spec), repr(spec_b))

  def testValueType(self):
    # [FUNCTION DOC]
    # Asserts a spec reports the row-partition class as the type of value it describes.
    spec = RowPartitionSpec()
    self.assertEqual(spec.value_type, RowPartition)

  @parameterized.parameters([
      dict(
          spec=RowPartitionSpec(),
          expected=(tensor_shape.TensorShape([None]),
                    tensor_shape.TensorShape([None]),
                    tensor_shape.TensorShape([None]), dtypes.int64)),
      dict(
          spec=RowPartitionSpec(dtype=dtypes.int32),
          expected=(tensor_shape.TensorShape([None]),
                    tensor_shape.TensorShape([None]),
                    tensor_shape.TensorShape([None]), dtypes.int32)),
      dict(
          spec=RowPartitionSpec(nrows=8, nvals=13),
          expected=(tensor_shape.TensorShape([8]),
                    tensor_shape.TensorShape([13]),
                    tensor_shape.TensorShape([None]), dtypes.int64)),
      dict(
          spec=RowPartitionSpec(nrows=8, uniform_row_length=2),
          expected=(
              tensor_shape.TensorShape([8]),
              tensor_shape.TensorShape([16]),  # inferred
              tensor_shape.TensorShape([2]),
              dtypes.int64)),
  ])
  def testSerialize(self, spec, expected):
    # [FUNCTION DOC]
    # Serializes a spec and compares with the expected tuple of row count shape, value count
    # shape, uniform row length shape and dtype (including inferred sizes). Shapes are compared
    # through their debug text because shape equality is unconventional.
    serialization = spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      dict(
          spec=RowPartitionSpec(),
          expected=tensor_spec.TensorSpec([None], dtypes.int64)),
      dict(
          spec=RowPartitionSpec(dtype=dtypes.int32),
          expected=tensor_spec.TensorSpec([None], dtypes.int32)),
      dict(
          spec=RowPartitionSpec(nrows=17, dtype=dtypes.int32),
          expected=tensor_spec.TensorSpec([18], dtypes.int32)),
      dict(
          spec=RowPartitionSpec(nvals=10, uniform_row_length=2),
          expected=tensor_spec.TensorSpec([6], dtypes.int64)),  # inferred nrow
  ])
  def testComponentSpecs(self, spec, expected):
    # [FUNCTION DOC]
    # Asserts the component tensor spec of a row-partition spec is a vector spec of the spec dtype
    # whose length is the row count plus one when known (inferred if needed), else unknown.
    self.assertEqual(spec._component_specs, expected)

  @parameterized.parameters([
      dict(
          rp_factory=lambda: RowPartition.from_row_splits([0, 3, 7]),
          components=[0, 3, 7]),
  ])
  def testToFromComponents(self, rp_factory, components):
    # [FUNCTION DOC]
    # Takes the type spec of a partition, converts the partition to components (the splits),
    # asserts they equal the expected list, rebuilds a partition from them and asserts it matches
    # the original via the module's equality helper.
    rp = rp_factory()
    spec = rp._type_spec
    actual_components = spec._to_components(rp)
    self.assertAllEqual(actual_components, components)
    rp_reconstructed = spec._from_components(actual_components)
    _assert_row_partition_equal(self, rp, rp_reconstructed)

  @parameterized.parameters([
      (RowPartitionSpec(), RowPartitionSpec()),
      (RowPartitionSpec(nrows=8), RowPartitionSpec(nrows=8)),
      (RowPartitionSpec(nrows=8), RowPartitionSpec(nrows=None)),
      (RowPartitionSpec(nvals=8), RowPartitionSpec(nvals=8)),
      (RowPartitionSpec(nvals=8), RowPartitionSpec(nvals=None)),
      (RowPartitionSpec(uniform_row_length=8),
       RowPartitionSpec(uniform_row_length=8)),
      (RowPartitionSpec(uniform_row_length=8),
       RowPartitionSpec(uniform_row_length=None)),
      (RowPartitionSpec(nvals=12), RowPartitionSpec(uniform_row_length=3)),
      (RowPartitionSpec(nrows=12), RowPartitionSpec(uniform_row_length=72)),
      (RowPartitionSpec(nrows=5), RowPartitionSpec(nvals=15)),
      (RowPartitionSpec(nvals=0), RowPartitionSpec(nrows=0)),
      (RowPartitionSpec(nvals=0), RowPartitionSpec(uniform_row_length=0)),
  ])
  def testIsCompatibleWith(self, spec1, spec2):
    # [FUNCTION DOC]
    # Asserts pairs of specs that can describe the same partition (equal, or one more specific,
    # or consistent through inference) are reported compatible.
    self.assertTrue(spec1.is_compatible_with(spec2))

  @parameterized.parameters([
      (RowPartitionSpec(), RowPartitionSpec(dtype=dtypes.int32)),
      (RowPartitionSpec(nvals=5), RowPartitionSpec(uniform_row_length=3)),
      (RowPartitionSpec(nrows=7,
                        nvals=12), RowPartitionSpec(uniform_row_length=3)),
      (RowPartitionSpec(nvals=5), RowPartitionSpec(nrows=0)),
      (RowPartitionSpec(nvals=5), RowPartitionSpec(uniform_row_length=0)),
  ])
  def testIsNotCompatibleWith(self, spec1, spec2):
    # [FUNCTION DOC]
    # Asserts pairs of specs with different dtypes or contradictory sizes are reported
    # incompatible.
    self.assertFalse(spec1.is_compatible_with(spec2))

  @parameterized.parameters([
      dict(
          spec1=RowPartitionSpec(nrows=8, nvals=3, dtype=dtypes.int32),
          spec2=RowPartitionSpec(nrows=8, nvals=3, dtype=dtypes.int32),
          expected=RowPartitionSpec(nrows=8, nvals=3, dtype=dtypes.int32)),
      dict(
          spec1=RowPartitionSpec(nrows=8, nvals=None),
          spec2=RowPartitionSpec(nrows=None, nvals=8),
          expected=RowPartitionSpec(nrows=None, nvals=None)),
      dict(
          spec1=RowPartitionSpec(nrows=8, nvals=33),
          spec2=RowPartitionSpec(nrows=3, nvals=13),
          expected=RowPartitionSpec(nrows=None, nvals=None)),
      dict(
          spec1=RowPartitionSpec(nrows=12, uniform_row_length=3),
          spec2=RowPartitionSpec(nrows=3, uniform_row_length=3),
          expected=RowPartitionSpec(nrows=None, uniform_row_length=3)),
      dict(
          spec1=RowPartitionSpec(5, 35, 7),
          spec2=RowPartitionSpec(8, 80, 10),
          expected=RowPartitionSpec(None, None, None)),
  ])
  def testMostSpecificCompatibleType(self, spec1, spec2, expected):
    # [FUNCTION DOC]
    # Asserts the most specific compatible type of two specs keeps only the attributes on which
    # they agree and relaxes the differing ones to unknown.
    actual = spec1.most_specific_compatible_type(spec2)
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      (RowPartitionSpec(), RowPartitionSpec(dtype=dtypes.int32)),
  ])
  def testMostSpecificCompatibleTypeError(self, spec1, spec2):
    # [FUNCTION DOC]
    # Specs with different dtypes have no common type: asking for the most specific compatible
    # type must raise a value error saying no type spec is compatible.
    with self.assertRaisesRegex(ValueError, 'No TypeSpec is compatible'):
      spec1.most_specific_compatible_type(spec2)

  def testNumRowsInt64(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: lives in the spec test class but tests a partition, not a spec.
    # Builds a partition from 64-bit splits and asserts its row count tensor is 64-bit.
    row_splits = array_ops.constant([0, 2, 3, 5], dtype=dtypes.int64)
    values = RowPartition.from_row_splits(row_splits)
    nrows = values.nrows()
    self.assertEqual(dtypes.int64, nrows.dtype)

  def testFromValue(self):
    # [FUNCTION DOC]
    # Asserts the spec derived from partitions built by each factory: splits and lengths give
    # only a row count; row ids give row count and value count; uniform row length gives value
    # count and row length.
    self.assertEqual(
        RowPartitionSpec.from_value(RowPartition.from_row_splits([0, 2, 8, 8])),
        RowPartitionSpec(nrows=3))
    self.assertEqual(
        RowPartitionSpec.from_value(
            RowPartition.from_row_lengths([5, 3, 0, 2])),
        RowPartitionSpec(nrows=4))
    self.assertEqual(
        RowPartitionSpec.from_value(
            RowPartition.from_value_rowids([0, 2, 2, 8])),
        RowPartitionSpec(nrows=9, nvals=4))
    self.assertEqual(
        RowPartitionSpec.from_value(
            RowPartition.from_uniform_row_length(
                nvals=12, uniform_row_length=3)),
        RowPartitionSpec(nvals=12, uniform_row_length=3))

  @parameterized.parameters([
      dict(original=RowPartitionSpec(),
           dtype=dtypes.int32,
           expected=RowPartitionSpec(dtype=dtypes.int32)),
      dict(original=RowPartitionSpec(dtype=dtypes.int32),
           dtype=dtypes.int64,
           expected=RowPartitionSpec()),
      dict(original=RowPartitionSpec(nvals=20, nrows=4, uniform_row_length=5),
           dtype=dtypes.int32,
           expected=RowPartitionSpec(nvals=20, nrows=4, uniform_row_length=5,
                                     dtype=dtypes.int32)),
  ])
  def testWithDType(self, original, dtype, expected):
    # [FUNCTION DOC]
    # Asserts changing the dtype of a spec returns a spec equal to the expected one, keeping all
    # size attributes.
    actual = original.with_dtype(dtype)
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      dict(a=RowPartitionSpec(),
           b=RowPartitionSpec(nrows=3, uniform_row_length=5),
           expected=RowPartitionSpec(nrows=3, uniform_row_length=5)),
      dict(a=RowPartitionSpec(nrows=3),
           b=RowPartitionSpec(uniform_row_length=5),
           expected=RowPartitionSpec(nrows=3, uniform_row_length=5)),
      dict(a=RowPartitionSpec(nvals=20),
           b=RowPartitionSpec(nrows=3),
           expected=RowPartitionSpec(nvals=20, nrows=3)),
      dict(a=RowPartitionSpec(nvals=20, dtype=dtypes.int32),
           b=RowPartitionSpec(nrows=3, dtype=dtypes.int32),
           expected=RowPartitionSpec(nvals=20, nrows=3, dtype=dtypes.int32)),
  ])
  def testMergeWith(self, a, b, expected):
    # [FUNCTION DOC]
    # Asserts merging two compatible specs (combining their known attributes) gives the expected
    # spec, and that the merge is symmetric (same result in both orders).
    actual = a._merge_with(b)
    actual_rev = b._merge_with(a)
    self.assertEqual(actual, expected)
    self.assertEqual(actual_rev, expected)

  @parameterized.parameters([
      dict(a=RowPartitionSpec(nrows=3, nvals=10),
           b=RowPartitionSpec(uniform_row_length=5),
           error_type=ValueError,
           error_regex='Merging incompatible RowPartitionSpecs'),
      dict(a=RowPartitionSpec(uniform_row_length=5, dtype=dtypes.int32),
           b=RowPartitionSpec(uniform_row_length=5, dtype=dtypes.int64),
           error_type=ValueError,
           error_regex='Merging RowPartitionSpecs with incompatible dtypes'),
  ])
  def testMergeWithRaises(self, a, b, error_type, error_regex):
    # [FUNCTION DOC]
    # Asserts merging specs with contradictory sizes or different dtypes raises the given error
    # type (value error) with a message matching the pattern.
    with self.assertRaisesRegex(error_type, error_regex):
      a._merge_with(b)


def _assert_row_partition_equal(test_class, actual, expected):
  # [FUNCTION DOC]
  # HELPER: Equality assertion used by the to/from components test.
  # Inputs: a test case instance (must be a TensorFlow test case), and two row partitions.
  # Asserts (type checks via plain assertions) that both have the same cached encodings and
  # uniform-row-length presence, then that every encoding cached on the expected partition has
  # equal values in the actual one. Returns nothing; raises assertion errors on mismatch.
  assert isinstance(test_class, test_util.TensorFlowTestCase)
  assert isinstance(actual, RowPartition)
  assert isinstance(expected, RowPartition)

  test_class.assertEqual(actual._has_precomputed_row_splits(),
                         expected._has_precomputed_row_splits())
  test_class.assertEqual(actual._has_precomputed_row_lengths(),
                         expected._has_precomputed_row_lengths())
  test_class.assertEqual(actual._has_precomputed_value_rowids(),
                         expected._has_precomputed_value_rowids())
  test_class.assertEqual(actual._has_precomputed_nrows(),
                         expected._has_precomputed_nrows())
  test_class.assertEqual(actual.uniform_row_length() is None,
                         expected.uniform_row_length() is None)

  if expected._has_precomputed_row_splits():
    test_class.assertAllEqual(actual.row_splits(), expected.row_splits())
  if expected._has_precomputed_row_lengths():
    test_class.assertAllEqual(actual.row_lengths(), expected.row_lengths())
  if expected._has_precomputed_value_rowids():
    test_class.assertAllEqual(actual.value_rowids(), expected.value_rowids())
  if expected._has_precomputed_nrows():
    test_class.assertAllEqual(actual.nrows(), expected.nrows())
  if expected.uniform_row_length() is not None:
    test_class.assertAllEqual(actual.uniform_row_length(),
                              expected.uniform_row_length())


if __name__ == '__main__':
  googletest.main()
