# [MODULE DOC]
# Unit tests for the DynamicRaggedShape class and its related functions in the ragged
# dynamic_ragged_shape module (construction from tensors, lengths and row partitions, slicing and
# indexing, num_row_partitions and inner-rank changes, broadcasting of shapes and tensors via
# internal broadcaster objects, elementwise ragged arithmetic with broadcasting, static shape
# inference, the Spec type, string representations and tf.data dataset batching/unbatching).
# The file first defines small module-level helper functions that build row partitions, ragged
# tensors and prime-number-filled tensors from nested "lengths" descriptions, then three test
# classes: the main shape tests, error-path tests, and tests for DynamicRaggedShape.Spec.
# Importing the module has no side effects other than imports; running it as a script starts the
# TensorFlow googletest main runner. Most tests run in both graph and eager mode.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.ragged.dynamic_ragged_shape."""

from typing import Sequence, Union

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.dynamic_ragged_shape import _LayerBroadcaster
from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.platform import googletest


def _to_row_partitions_from_lengths(
    lengths: Sequence[Union[int, Sequence[int]]]) -> Sequence[RowPartition]:
  # [FUNCTION DOC]
  # HELPER: test utility that builds row partitions for shapes used throughout this test file.
  # Converts a "lengths" description into a sequence of RowPartition objects. The input is a list
  # whose items are either an integer (uniform dimension size) or a list of integers (ragged row
  # lengths for that dimension). Delegates to the module's internal function that returns both
  # the partitions and the value count, and returns only the partitions. Raises whatever that
  # internal function raises for inconsistent lengths. No side effects.
  """Allow ragged and uniform shapes to be specified.

  For example, [2, [2,1], 2] represents a shape like:
  [[[0, 0], [0, 0]], [[0, 0]]]

  Args:
    lengths: a list of integers and lists of integers.

  Returns:
    a sequence of RowPartitions.
  """
  (result,
   _) = dynamic_ragged_shape._to_row_partitions_and_nvals_from_lengths(lengths)
  return result


def _to_ragged_tensor_from_lengths(
    values, lengths: Sequence[Union[int, Sequence[int]]]) -> RaggedTensor:
  # [FUNCTION DOC]
  # HELPER: test utility for building input data of a given ragged shape.
  # Takes flat values (anything convertible to a constant tensor) and a lengths description (list
  # of integers or lists of integers). Builds the row partitions from the lengths; if there are
  # none, returns just the dense constant tensor, otherwise returns a RaggedTensor built from the
  # constant values and the nested row partitions. No side effects.
  """Specify a ragged tensor (or tensor) from lengths and values."""
  row_partitions = _to_row_partitions_from_lengths(lengths)
  values = constant_op.constant(values)
  if not row_partitions:
    return values
  return RaggedTensor._from_nested_row_partitions(values, row_partitions)


def _divides(a, b):
  # [FUNCTION DOC]
  # HELPER: tiny arithmetic helper for the prime generator below.
  # Returns true when the first argument evenly divides the second (the remainder is zero).
  # Inputs are integers; pure function.
  return b % a == 0


def _next_prime(primes_so_far):
  # [FUNCTION DOC]
  # HELPER: prime-number generator used to make tensors whose elements are all distinct.
  # Takes the list of primes found so far (possibly empty) and returns the next prime, by
  # trying consecutive candidates starting at two (or one past the last known prime) until one is
  # not divisible by any known prime. Does not modify the input list. Pure function.
  first_candidate = 2
  if primes_so_far:
    first_candidate = primes_so_far[-1] + 1
  while True:
    if not any([_divides(x, first_candidate) for x in primes_so_far]):
      return first_candidate
    first_candidate = first_candidate + 1


def _lowest_primes(n):
  # [FUNCTION DOC]
  # HELPER: generator of distinct element values for test tensors.
  # Returns a list holding the lowest n prime numbers (n is a non-negative integer), built by
  # repeatedly asking for the next prime. Pure function.
  """Give the lowest n primes."""
  result = []
  for _ in range(n):
    result.append(_next_prime(result))
  return result


def _num_elements_of_lengths_with_rows(rows,
                                       lengths: Sequence[Union[int,
                                                               Sequence[int]]]):
  # [FUNCTION DOC]
  # HELPER: recursive worker for computing the number of elements in a shape.
  # Takes the number of rows seen so far and the remaining lengths list. If no lengths remain it
  # returns the row count. For an integer entry it multiplies the row count by it; for a list
  # entry (ragged lengths) it replaces the row count by the sum of the list. Recurses on the rest.
  # Returns an integer; pure function.
  """Helper function for _num_elements_of_lengths."""
  if not lengths:
    return rows
  next_length = lengths[0]
  if isinstance(next_length, int):
    return _num_elements_of_lengths_with_rows(next_length * rows, lengths[1:])
  else:
    return _num_elements_of_lengths_with_rows(sum(next_length), lengths[1:])


def _num_elements_of_lengths(lengths: Sequence[Union[int, Sequence[int]]]):
  # [FUNCTION DOC]
  # HELPER: static computation of how many scalar elements a shape described by lengths has.
  # Starts the recursive helper with a single row. Input is a list of integers and/or lists of
  # integers; returns an integer. The docstring promises equivalence with the number of elements
  # reported by the shape built from the same lengths, which the tests rely on.
  """Static version of DynamicRaggedShape.from_lengths(lengths)._num_elements()."""
  return _num_elements_of_lengths_with_rows(1, lengths)


def _to_prime_tensor_from_lengths(
    lengths: Sequence[Union[int, Sequence[int]]]) -> RaggedTensor:
  # [FUNCTION DOC]
  # HELPER: builds a tensor of distinct prime numbers with a requested shape.
  # Creates a DynamicRaggedShape from the lengths description, computes the number of elements
  # statically, and reshapes the first that-many primes into that shape with the ragged reshape
  # op. Returns a dense tensor or RaggedTensor. Used so broadcast results can be checked by
  # value, since every element is unique.
  """Create a tensor of primes with the shape specified."""
  shape = DynamicRaggedShape.from_lengths(lengths)
  num_elements = _num_elements_of_lengths(lengths)
  return ragged_array_ops.ragged_reshape(_lowest_primes(num_elements), shape)


@test_util.run_all_in_graph_and_eager_modes
class DynamicRaggedShapeTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for DynamicRaggedShape (and the broadcaster/dynamic-shape functions in the
  # dynamic_ragged_shape module). Decorated to run every test in both graph and eager modes, and
  # combines the TensorFlow test case with absl parameterized tests, so many methods are run once
  # per parameter dictionary. Defines assertion helpers (partition, shape, layer broadcaster and
  # broadcaster equality) and then tests for construction, indexing, broadcasting, arithmetic,
  # static shapes, merging, and datasets.

  def assertRowPartitionEq(self,
                           x: RowPartition,
                           y: RowPartition,
                           msg=None) -> None:
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the shape tests.
    # Asserts that two RowPartition objects have equal row splits. Inputs: x and y, the partitions
    # to compare, and an optional failure message. Fails the test if the splits differ; returns
    # nothing.
    self.assertAllEqual(x.row_splits(), y.row_splits(), msg=msg)

  def assertShapeEq(self,
                    x: DynamicRaggedShape,
                    y: DynamicRaggedShape,
                    msg=None) -> None:
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the shape tests.
    # Asserts that two DynamicRaggedShape objects are equal: both must be shape instances (plain
    # assertion failure otherwise), have the same number of row partitions, equal row splits in
    # each partition, and equal inner shapes. Optional message is prefixed to the failure text.
    # Returns nothing.
    assert isinstance(x, DynamicRaggedShape)
    assert isinstance(y, DynamicRaggedShape)
    if msg is None:
      msg = ''
    self.assertLen(
        x.row_partitions, len(y.row_partitions), msg=msg + ': length unequal')
    for i in range(len(x.row_partitions)):
      x_dims = x.row_partitions[i]
      y_dims = y.row_partitions[i]
      self.assertRowPartitionEq(
          x_dims, y_dims, msg=msg + ': row_partition ' + str(i))
    self.assertAllEqual(
        x.inner_shape, y.inner_shape, msg=msg + ': shapes unequal')

  def assertLayerBroadcasterEq(self, x: _LayerBroadcaster,
                               y: _LayerBroadcaster) -> None:
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the broadcaster tests.
    # Checks that both arguments are layer broadcaster objects and that their gather indices are
    # equal. Fails the test otherwise; returns nothing.
    assert isinstance(x, _LayerBroadcaster)
    assert isinstance(y, _LayerBroadcaster)
    self.assertAllEqual(x.gather_index, y.gather_index)

  def assertBroadcasterEq(self, x: dynamic_ragged_shape._Broadcaster,
                          y: dynamic_ragged_shape._Broadcaster) -> None:
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the broadcaster tests.
    # Checks that both arguments are internal broadcaster objects, that their source shapes and
    # target shapes are equal, that they have the same number of layer broadcasters, and that each
    # corresponding pair of layer broadcasters is equal. Returns nothing; fails the test on any
    # mismatch.
    assert isinstance(x, dynamic_ragged_shape._Broadcaster)
    assert isinstance(y, dynamic_ragged_shape._Broadcaster)
    self.assertShapeEq(x.source_shape, y.source_shape)
    self.assertShapeEq(x.target_shape, y.target_shape)
    self.assertLen(x._layer_broadcasters, len(y._layer_broadcasters))
    for x_layer, y_layer in zip(x._layer_broadcasters, y._layer_broadcasters):
      self.assertLayerBroadcasterEq(x_layer, y_layer)

  @parameterized.parameters([
      dict(value='x', row_partitions=[], inner_shape=()),
      dict(value=['a', 'b', 'c'], row_partitions=[], inner_shape=[3]),
      dict(
          value=[['a', 'b', 'c'], ['d', 'e', 'f']],
          row_partitions=(),
          inner_shape=[2, 3]),
      dict(
          value=[[['a', 'b', 'c'], ['d', 'e', 'f']]],
          row_partitions=(),
          inner_shape=[1, 2, 3]),
      dict(
          value=ragged_factory_ops.constant_value([['a', 'b', 'c'], ['d', 'e']],
                                                  ragged_rank=1),
          row_partitions=[[0, 3, 5]],
          inner_shape=[5]),
      dict(
          value=ragged_factory_ops.constant_value(
              [[['a', 'b', 'c'], ['d', 'e', 'f']]], ragged_rank=1),
          row_partitions=[[0, 2]],
          inner_shape=[2, 3]),
      dict(
          value=ragged_factory_ops.constant_value(
              [[[[1], [2]], [[3], [4]]], [[[5], [6]]]], ragged_rank=1),
          row_partitions=[[0, 2, 3]],
          inner_shape=[3, 2, 1]),
      dict(
          value=ragged_factory_ops.constant_value([[10, 20], [30]]),
          row_partitions=[[0, 2, 3]],
          inner_shape=[3]),
      # Docstring examples:
      dict(value=[[1, 2, 3], [4, 5, 6]], row_partitions=[], inner_shape=[2, 3]),
      dict(
          value=ragged_factory_ops.constant_value([[1, 2], [], [3, 4, 5]]),
          row_partitions=[[0, 2, 2, 5]],
          inner_shape=[5]),
      dict(
          value=ragged_factory_ops.constant_value([[[1, 2], [3, 4]], [[5, 6]]],
                                                  ragged_rank=1),
          row_partitions=[[0, 2, 3]],
          inner_shape=[3, 2]),
      dict(
          value=ragged_factory_ops.constant_value([[[1, 2], [3]], [[4, 5]]]),
          row_partitions=[[0, 2, 3], [0, 2, 3, 5]],
          inner_shape=[5]),
  ])
  def testFromTensor(self, value, row_partitions, inner_shape):
    # [FUNCTION DOC]
    # Scenario: building a shape from an existing tensor. Parameterized over dense and ragged
    # values (including documentation examples) with the expected row-split lists and inner shape.
    # Builds the shape with the from-tensor constructor, builds the expected shape from row
    # partitions made from row splits plus the inner shape, and asserts the two shapes are equal.
    shape = DynamicRaggedShape.from_tensor(value)
    row_partitions = [RowPartition.from_row_splits(x) for x in row_partitions]
    expected = DynamicRaggedShape(row_partitions, inner_shape)
    self.assertShapeEq(shape, expected)

  # pylint:disable=g-long-lambda
  @parameterized.parameters([
      # from_lengths           | row_partitions            | inner_shape
      # ---------------------- | --------------------------| -------------
      # []                     | []                        | []
      # [2, (3, 2)]            | [RP([3, 2])]              | [5]
      # [2, 2]                 | []                        | [2, 2]
      # [2, (3, 2), 7]         | [RP([3, 2])]              | [5, 7]
      # [2, (2, 2), 3]         | [RP([2, 2])]              | [4, 3]
      # [2, 2, 3]              | []                        | [2, 2, 3]
      # [2, (2, 1), (2, 0, 3)] | [RP(2, 1), RP([2, 0, 3])] | [5]

      dict(lengths=[], row_partitions=[], inner_shape=[]),
      dict(
          lengths=[2, (3, 2)],
          row_partitions=lambda: [RowPartition.from_row_lengths([3, 2])],
          inner_shape=[5]),
      dict(lengths=[2, 2], row_partitions=[], inner_shape=[2, 2]),
      dict(
          lengths=[2, (3, 2), 7],
          row_partitions=lambda: [RowPartition.from_row_lengths([3, 2])],
          inner_shape=[5, 7]),
      dict(
          lengths=[2, (2, 2), 3],
          row_partitions=lambda: [RowPartition.from_row_lengths([2, 2])],
          inner_shape=[4, 3]),
      dict(lengths=[2, 2, 3], row_partitions=[], inner_shape=[2, 2, 3]),
      dict(
          lengths=[2, (2, 1), (2, 0, 3)],
          row_partitions=lambda: [
              RowPartition.from_row_lengths([2, 1]),
              RowPartition.from_row_lengths([2, 0, 3])
          ],
          inner_shape=[5]),
      # from_lengths   | num_row    | row_partitions           | inner_shape
      #                : partitions :                          :
      # ---------------| -----------|--------------------------|------------
      # [2, (3, 2), 2] | 2          | [RP([3, 2]), URP(2, 10)] | [10]
      # [2, 2]         | 1          | [URP(2, 4)]              | [4]
      # [2, 2, 3]      | 0          | []                       | [2, 2, 3]
      # [2, 2, 3]      | 1          | [URP(2, 4)]              | [4, 3]
      # [2, 2, 3]      | 2          | [URP(2, 4), URP(3, 12)]  | [12]
      dict(lengths=[2, (3, 2), 2],
           num_row_partitions=2,
           row_partitions=lambda: [RowPartition.from_row_lengths([3, 2]),
                                   RowPartition.from_uniform_row_length(2, 10)],
           inner_shape=[10]),
      dict(lengths=[2, 2],
           num_row_partitions=1,
           row_partitions=lambda: [RowPartition.from_uniform_row_length(2, 4)],
           inner_shape=[4]),
      dict(lengths=[2, 2, 3],
           num_row_partitions=0,
           row_partitions=[],
           inner_shape=[2, 2, 3]),
      dict(lengths=[2, 2, 3],
           num_row_partitions=1,
           row_partitions=lambda: [RowPartition.from_uniform_row_length(2, 4)],
           inner_shape=[4, 3]),
      dict(lengths=[2, 2, 3],
           num_row_partitions=2,
           row_partitions=lambda: [RowPartition.from_uniform_row_length(2, 4),
                                   RowPartition.from_uniform_row_length(3, 12)],
           inner_shape=[12])
  ])
  def testFromLengths(self,
                      lengths,
                      row_partitions,
                      inner_shape,
                      num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: building a shape from a lengths description, with an optional number of row
    # partitions. If the expected row partitions are given as a callable, it is invoked first (so
    # partitions are created inside the test). Builds the shape from lengths and asserts it equals
    # a shape built directly from the expected row partitions and inner shape.
    if callable(row_partitions):
      row_partitions = row_partitions()
    shape = DynamicRaggedShape.from_lengths(
        lengths, num_row_partitions=num_row_partitions)
    expected = DynamicRaggedShape(row_partitions, inner_shape)
    self.assertShapeEq(shape, expected)

  @parameterized.parameters([
      dict(
          lengths=[2, (2, 1, 3)],
          num_row_partitions=1,
          msg='Shape not consistent'),
      dict(
          lengths=[2, 3],
          num_row_partitions=2,
          msg='num_row_partitions should be less than'),
      dict(
          lengths=[],
          num_row_partitions=3,
          msg='num_row_partitions==0 for a scalar shape'),
      dict(
          lengths=[(5, 3), 3],
          num_row_partitions='a',
          msg='num_row_partitions should be an int or None'),
      dict(
          lengths=[(5, 'a'), 3],
          num_row_partitions=0,
          msg='element of lengths should be int or tuple of ints'),
      dict(
          lengths=['a'],
          num_row_partitions=0,
          msg='element of lengths should be int or tuple of ints'),
      dict(lengths=7, num_row_partitions=0, msg='lengths should be a list')
  ])
  def testFromLengthsError(self, lengths, msg, num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: invalid arguments to the from-lengths constructor (inconsistent lengths, too many
    # row partitions, wrong types, scalar with partitions, non-list lengths). Asserts that a
    # value error whose message matches the expected text is raised in each case. The
    # num_row_partitions argument is optional and defaults to none.
    with self.assertRaisesRegex(ValueError, msg):
      DynamicRaggedShape.from_lengths(
          lengths, num_row_partitions=num_row_partitions)

  def testGetItemSliceRankUnknownA(self):
    # [FUNCTION DOC]
    # Scenario: slicing a shape of unknown rank. Only runs in graph mode (does nothing in eager
    # mode). Creates a shape from a placeholder with unknown shape, slices off the first element
    # and asserts the resulting rank is unknown (none).
    if not context.executing_eagerly():
      original_t = array_ops.placeholder_with_default(np.array([4, 5, 3]), None)
      sh = DynamicRaggedShape.from_tensor(original_t)
      known = sh[:1]
      self.assertIsNone(known.rank)

  def testGetItemSliceRankUnknownLong(self):
    # [FUNCTION DOC]
    # Scenario: slicing a shape of unknown rank past its end. Graph mode only (does nothing in
    # eager mode). Slices the first twenty entries of a shape built from a placeholder of unknown
    # shape and asserts that the resulting rank is unknown.
    if not context.executing_eagerly():
      original_t = array_ops.placeholder_with_default(np.array([4, 5, 3]), None)
      sh = DynamicRaggedShape.from_tensor(original_t)
      unknown = sh[:20]
      self.assertIsNone(unknown.rank)

  def testGetItemSliceRankKnownLong(self):
    # [FUNCTION DOC]
    # Scenario: slicing a known-rank shape past its end. Graph mode only. Builds a shape from a
    # constant vector of length three, slices the first twenty entries, and asserts the result
    # still has rank one.
    if not context.executing_eagerly():
      original_t = constant_op.constant([4, 5, 3], dtypes.float32)
      sh = DynamicRaggedShape.from_tensor(original_t)
      unknown = sh[:20]
      self.assertEqual(unknown.rank, 1)

  def testGetBroadcaster(self):
    # [FUNCTION DOC]
    # Scenario: the internal function that computes a broadcaster between two shapes, going from a
    # uniform shape to a larger uniform shape. Builds the origin and destination shapes, computes
    # the broadcaster, and asserts it equals an explicitly constructed broadcaster with the
    # expected gather indices for the two layers.
    origin_shape = DynamicRaggedShape(
        [RowPartition.from_uniform_row_length(1, 3)], inner_shape=[3])
    dest_shape = DynamicRaggedShape(
        [RowPartition.from_uniform_row_length(2, 6)], inner_shape=[6])
    actual = dynamic_ragged_shape._get_broadcaster(origin_shape, dest_shape)
    expected = dynamic_ragged_shape._Broadcaster(origin_shape, dest_shape, [
        _LayerBroadcaster.from_gather_index([0, 1, 2]),
        _LayerBroadcaster.from_gather_index([0, 0, 1, 1, 2, 2])
    ])
    self.assertBroadcasterEq(actual, expected)

  def testGetBroadcaster2(self):
    # [FUNCTION DOC]
    # Scenario: computing a broadcaster from a scalar (empty) shape to a ragged shape. Asserts the
    # result equals a broadcaster with the same source and target shapes and no layer
    # broadcasters.
    origin_shape = DynamicRaggedShape([], inner_shape=[])
    dest_shape = DynamicRaggedShape([RowPartition.from_row_splits([0, 2, 3])],
                                    inner_shape=[3])
    actual = dynamic_ragged_shape._get_broadcaster(origin_shape, dest_shape)
    expected = dynamic_ragged_shape._Broadcaster(origin_shape, dest_shape, [])
    self.assertBroadcasterEq(actual, expected)

  @parameterized.parameters([
      dict(lengths=[2, 3], axis=0, expected=2),
      dict(lengths=[2, 3], axis=1, expected=6),
      dict(lengths=[2, 3], axis=-1, expected=6),
      dict(lengths=[2, 3], axis=-2, expected=2),
      dict(lengths=[2, 3, 4], axis=0, expected=2),
      dict(lengths=[2, 3, 4], axis=1, expected=6),
      dict(lengths=[2, 3, 4], axis=2, expected=24),
      dict(lengths=[2, 3, 4], axis=-1, expected=24),
      dict(lengths=[2, 3, 4], axis=-2, expected=6),
      dict(lengths=[2, 3, 4], axis=-3, expected=2),
      dict(lengths=[2, (2, 3), 7], axis=0, expected=2),
      dict(lengths=[2, (2, 3), 7], axis=1, expected=5),
      dict(lengths=[2, (2, 3), 7], axis=2, expected=35),
      dict(lengths=[2, (2, 3), 7], axis=-1, expected=35),
      dict(lengths=[2, (2, 3), 7], axis=-2, expected=5),
      dict(lengths=[2, (2, 3), 7], axis=-3, expected=2),
  ])
  def testNumSlicesInDimension(self, lengths, axis, expected):
    # [FUNCTION DOC]
    # Scenario: the number of slices in a given dimension. Parameterized over uniform and ragged
    # lengths with positive and negative axis values. Builds the shape, calls the internal
    # number-of-slices method with the axis, and asserts it equals the expected count.
    original = DynamicRaggedShape.from_lengths(lengths)
    actual = original._num_slices_in_dimension(axis)
    self.assertAllEqual(expected, actual)

  @parameterized.parameters([
      dict(
          lengths=[2, 3],
          axis=0.5,
          error_type=TypeError,
          error_regex='axis must be an integer'),
  ])
  def testNumSlicesInDimensionRaises(self, lengths, axis, error_type,
                                     error_regex):
    # [FUNCTION DOC]
    # Scenario: invalid axis for the number-of-slices method. Parameters give lengths, a
    # non-integer axis, and the expected error type and message pattern. Asserts the method raises
    # that error.
    original = DynamicRaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._num_slices_in_dimension(axis)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2), 4],
          new_dense_rank=3,
          error_type=ValueError,
          error_regex='Cannot get an inner shape'),
      dict(
          lengths=[],
          new_dense_rank=3,
          error_type=ValueError,
          error_regex='old inner_rank cannot be zero'),
      dict(
          lengths=[2, 3],
          new_dense_rank=0,
          error_type=ValueError,
          error_regex='new_inner_rank cannot be zero'),
  ])
  def testAltInnerShapeRaises(self, lengths, new_dense_rank, error_type,
                              error_regex):
    # [FUNCTION DOC]
    # Scenario: error cases of the internal alternative-inner-shape method (inner shape that cannot
    # be computed for a ragged shape, zero old inner rank, zero new inner rank). Parameters give
    # lengths, new dense rank, and expected error type and message. Asserts that the error is
    # raised.
    original = DynamicRaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._alt_inner_shape(new_dense_rank)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2), 4], new_dense_rank=2, expected_inner_shape=[3,
                                                                          4]),
  ])
  def testAltInnerShape(self, lengths, new_dense_rank, expected_inner_shape):
    # [FUNCTION DOC]
    # Scenario: the alternative-inner-shape method with a valid new dense rank. Builds a shape from
    # lengths, requests the inner shape for the new rank and asserts it equals the expected inner
    # shape.
    original = DynamicRaggedShape.from_lengths(lengths)
    actual = original._alt_inner_shape(new_dense_rank)
    self.assertAllEqual(actual, expected_inner_shape)

  def testWithNumRowPartitionsDynamic(self):
    # [FUNCTION DOC]
    # Scenario: changing the number of row partitions of a shape whose inner shape is only known
    # at run time. Defines a compiled function with an int64 vector input signature and calls it
    # with a constant. Inside, the number of row partitions is raised to three and the result is
    # compared with an explicitly built shape that includes an extra uniform row partition.
    @def_function.function(
        input_signature=[tensor.TensorSpec([3], dtypes.int64)])
    def fun(x):
      # [FUNCTION DOC]
      # Nested function compiled with a fixed int64 vector signature. Takes the dynamic inner shape
      # tensor x, builds a shape from two row-length partitions and x, changes it to have three row
      # partitions, and asserts equality with the expected shape. Returns nothing.
      shape = DynamicRaggedShape([
          RowPartition.from_row_lengths([1, 3], dtype=dtypes.int64),
          RowPartition.from_row_lengths([2, 3, 4, 5], dtype=dtypes.int64)
      ], x)
      result = shape._with_num_row_partitions(3)
      expected = DynamicRaggedShape([
          RowPartition.from_row_lengths([1, 3], dtype=dtypes.int64),
          RowPartition.from_row_lengths([2, 3, 4, 5], dtype=dtypes.int64),
          RowPartition.from_uniform_row_length(
              2, nrows=14, nvals=28, dtype=dtypes.int64)
      ], [14 * 2, 3])
      self.assertShapeEq(expected, result)
    fun(constant_op.constant([14, 2, 3], dtype=dtypes.int64))

  @parameterized.parameters([
      dict(
          lengths=[2],
          new_dense_rank=2,
          error_type=ValueError,
          error_regex='Cannot change inner_rank if'),
  ])
  def testWithDenseRankRaises(self, lengths, new_dense_rank, error_type,
                              error_regex):
    # [FUNCTION DOC]
    # Scenario: changing the inner rank when it is not possible. Parameters give lengths, a new
    # dense rank, and the expected error. Builds the shape and asserts that the internal
    # with-inner-rank method raises the error. (The name says dense rank; the method called is the
    # inner-rank one.)
    original = DynamicRaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._with_inner_rank(new_dense_rank)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2)],
          num_row_partitions=2,
          error_type=ValueError,
          error_regex='num_row_partitions must be less than rank'),
      dict(
          lengths=[2],
          num_row_partitions=-1,
          error_type=ValueError,
          error_regex='num_row_partitions must be nonnegative'),
      dict(
          lengths=[2],
          num_row_partitions=0.5,
          error_type=ValueError,
          error_regex='num_row_partitions must be an int'),
  ])
  def testWithNumRowPartitionsRaises(self, lengths, num_row_partitions,
                                     error_type, error_regex):
    # [FUNCTION DOC]
    # Scenario: invalid arguments to the internal with-number-of-row-partitions method (too many,
    # negative, non-integer). Parameters give lengths, the bad count, and the expected error type
    # and message. Asserts the error is raised.
    original = DynamicRaggedShape.from_lengths(lengths)
    with self.assertRaisesRegex(error_type, error_regex):
      original._with_num_row_partitions(num_row_partitions)

  def testDimensionRaises(self):
    # [FUNCTION DOC]
    # Scenario: asking the internal dimension accessor for a non-integer index. Builds a ragged
    # shape and asserts a type error saying the index should be an integer is raised.
    original = DynamicRaggedShape.from_lengths([2, (1, 2)])
    with self.assertRaisesRegex(TypeError, 'index should be an int'):
      # This error is not exposed directly to the end user.
      original._dimension(0.5)

  @parameterized.parameters([
      # The whole shape (num_row_partitions=0, start=negative, stop=really big)
      dict(lengths=[2, 3], s=slice(-1000, 100), expected_lengths=[2, 3]),
      # The whole shape (num_row_partitions=0, stop=really big)
      dict(lengths=[2, 3], s=slice(0, 100), expected_lengths=[2, 3]),
      # The whole shape (num_row_partitions=0, stop=None)
      dict(lengths=[2, 3], s=slice(0, None), expected_lengths=[2, 3]),
      # start = None, num_row_partitions=1, stop = 3 < rank = 4
      dict(
          lengths=[2, (1, 2), 3, 4],
          s=slice(None, 3),
          expected_lengths=[2, (1, 2), 3]),
      # start = 1, num_row_partitions=1, stop = 4, rank = 4
      dict(
          lengths=[2, 3, 3, 4],
          num_row_partitions=1,
          s=slice(1, 4),
          expected_lengths=[3, 3, 4]),
      # start = 1, num_row_partitions=1, stop = 3 < rank = 4
      dict(
          lengths=[2, 3, 3, 4],
          num_row_partitions=1,
          s=slice(1, 3),
          expected_lengths=[3, 3]),
      # start = 1, num_row_partitions=2, stop = 3 < rank = 4
      dict(
          lengths=[2, 3, 4, 3, 4],
          num_row_partitions=2,
          s=slice(1, 3),
          expected_lengths=[3, 4]),
      # start = 0, num_row_partitions=1, stop = 3 < rank = 4
      dict(
          lengths=[2, (1, 2), 3, 4],
          s=slice(0, 3),
          expected_lengths=[2, (1, 2), 3]),
      # start = 0, num_row_partitions=0, stop < rank
      dict(lengths=[2, 3, 4], s=slice(0, 2), expected_lengths=[2, 3]),
      # start=0 < stop=2 <= num_row_partitions
      dict(
          lengths=[2, (1, 2), (3, 4, 5)],
          s=slice(0, 2),
          expected_lengths=[2, (1, 2)]),
      # start=0 < stop=1 <= num_row_partitions
      dict(lengths=[2, (1, 2), (3, 4, 5)], s=slice(0, 1), expected_lengths=[2]),
      # Reversed indices, gives scalar shape.
      dict(lengths=[2, 3], s=slice(2, 0), expected_lengths=[]),
      # The whole shape (num_row_partitions=0)
      dict(lengths=[2, 3], s=slice(0, 2), expected_lengths=[2, 3]),
  ])
  def testGetItemSlice(self,
                       lengths,
                       s,
                       expected_lengths,
                       num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: slicing a shape with slice objects, covering whole-shape slices, out-of-range
    # bounds, reversed bounds (gives scalar shape), and slices within and beyond the row
    # partitions. Builds the shape (optionally converting it to the given number of row
    # partitions), slices it, and asserts the result equals a shape built from the expected
    # lengths.
    original = DynamicRaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      original = original._with_num_row_partitions(num_row_partitions)
    expected = DynamicRaggedShape.from_lengths(expected_lengths)
    actual = original[s]
    self.assertShapeEq(expected, actual)

  @parameterized.parameters([
      dict(
          lengths=[2, (1, 2), 3, 4],
          index=0.5,
          error_type=TypeError,
          error_regex='Argument is not an int or a slice'),
      dict(
          lengths=[2, (1, 2), 3, 4],
          index=slice(0, 1, 2),
          error_type=IndexError,
          error_regex='Cannot stride through a shape'),
      dict(
          lengths=[2, (1, 2), 3, 4],
          index=1,
          error_type=ValueError,
          error_regex='Index 1 is not uniform'),
      dict(
          lengths=[2, 3, 3, 4],
          num_row_partitions=1,
          index=-20,
          error_type=IndexError,
          error_regex='Index must be non-negative'),
      dict(
          lengths=[2, 3, 3, 4],
          num_row_partitions=1,
          index=9,
          error_type=IndexError,
          error_regex='Index is too big'),
  ])
  def testGetItemRaisesStatic(self,
                              lengths,
                              index,
                              error_type,
                              error_regex,
                              num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: invalid indexing of a shape (non-integer index, strided slice, non-uniform index,
    # negative index too small, index too large). Optionally converts the shape to a given number
    # of row partitions, then asserts that indexing raises the expected error type with a message
    # matching the expected pattern.
    original = DynamicRaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      original = original._with_num_row_partitions(num_row_partitions)
    with self.assertRaisesRegex(error_type, error_regex):
      original[index]  # pylint: disable=pointless-statement

  def testBroadcastToAlt(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting a uniform-row-length ragged tensor to a larger uniform shape using
    # the broadcast-to function. Builds an origin with row length one and an expected tensor with
    # row length two, and asserts the broadcast result equals the expected tensor.
    origin = RaggedTensor.from_uniform_row_length([3, 4, 5],
                                                  uniform_row_length=1)
    expected = RaggedTensor.from_uniform_row_length([3, 3, 4, 4, 5, 5],
                                                    uniform_row_length=2)
    expected_shape = DynamicRaggedShape.from_tensor(expected)
    actual = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  @parameterized.parameters([
      dict(
          source_lengths=[3],
          target_lengths=[1, 3],
          target_num_row_partitions=1,
          expected_gather_indices=[[0, 1, 2]]),
      dict(  # BroadcastTensorTo4 broadcaster.
          source_lengths=[2, 3],
          target_lengths=[1, 2, 3],
          target_num_row_partitions=2,
          expected_gather_indices=[[0, 1], [0, 1, 2, 3, 4, 5]]),
      dict(  # raggedTensor1.
          source_lengths=[3, (1, 2, 1), 2, 2],
          source_num_row_partitions=3,
          target_lengths=[1, 1, 3, (1, 2, 1), 2, 2],
          target_num_row_partitions=5,
          expected_gather_indices=[[0, 1, 2], [0, 1, 2, 3],
                                   [0, 1, 2, 3, 4, 5, 6, 7],
                                   [
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15
                                   ]]),
  ])
  def testBroadcaster(self,
                      source_lengths,
                      target_lengths,
                      expected_gather_indices,
                      source_num_row_partitions=None,
                      target_num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: broadcaster construction between a source shape and a target shape given by
    # lengths, each optionally with an explicit number of row partitions. Computes the broadcaster
    # with the internal function and asserts it equals one constructed from the expected gather
    # indices for each layer.
    source = DynamicRaggedShape.from_lengths(source_lengths)
    if source_num_row_partitions is not None:
      source = source._with_num_row_partitions(source_num_row_partitions)
    target = DynamicRaggedShape.from_lengths(target_lengths)
    if target_num_row_partitions is not None:
      target = target._with_num_row_partitions(target_num_row_partitions)

    expected_gather_indices = [
        _LayerBroadcaster.from_gather_index(x) for x in expected_gather_indices
    ]
    actual = dynamic_ragged_shape._get_broadcaster(source, target)
    expected = dynamic_ragged_shape._Broadcaster(source, target,
                                                 expected_gather_indices)
    self.assertBroadcasterEq(actual, expected)

  def testRaggedGradientSimple1(self):
    # [FUNCTION DOC]
    # Scenario: gradient through ragged multiplication with broadcasting, using row splits
    # constructors. Graph mode only (returns immediately in eager mode). Asserts that the gradient
    # of the flat values of a ragged-times-column product with respect to the input values equals
    # the broadcast multipliers (ten, one hundred, one thousand repeated per row).
    if context.executing_eagerly():
      return
    def func(x):
      # [FUNCTION DOC]
      # Nested function: builds a ragged tensor from values x and fixed row splits, multiplies by a
      # column of scalars and returns the flat values of the product.
      rt1 = RaggedTensor.from_row_splits(
          values=x, row_splits=[0, 4, 7, 8], validate=False)
      rt2 = rt1 * [[10], [100], [1000]]
      return rt2.flat_values

    x = constant_op.constant([3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(ops.convert_to_tensor(g),
                        [10., 10., 10., 10., 100., 100., 100, 1000.])

  def testRaggedGradientSimple2(self):
    # [FUNCTION DOC]
    # Scenario: same gradient check as the first simple gradient test, but builds the ragged
    # tensor from a row partition. Graph mode only. Asserts the gradient with respect to the input
    # values equals the per-row multipliers.
    if context.executing_eagerly():
      return
    def func(x):
      # [FUNCTION DOC]
      # Nested function: builds a ragged tensor from values x and a row partition, multiplies by a
      # column of scalars and returns the flat values.
      rt1 = RaggedTensor._from_row_partition(
          x,
          RowPartition.from_row_splits(row_splits=[0, 4, 7, 8], validate=False))
      rt2 = rt1 * [[10], [100], [1000]]
      return rt2.flat_values

    x = constant_op.constant([3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(ops.convert_to_tensor(g),
                        [10., 10., 10., 10., 100., 100., 100, 1000.])

  def testRaggedGradientSimple3(self):
    # [FUNCTION DOC]
    # Scenario: same gradient check, with a row partition whose row splits use a 32-bit integer
    # type. Graph mode only. Asserts the gradient equals the per-row multipliers.
    if context.executing_eagerly():
      return
    def func(x):
      # [FUNCTION DOC]
      # Nested function: builds a ragged tensor from values x and a 32-bit-split row partition,
      # multiplies by a column of scalars and returns the flat values.
      rt1 = RaggedTensor._from_row_partition(
          x,
          RowPartition.from_row_splits(row_splits=[0, 4, 7, 8],
                                       dtype=dtypes.int32, validate=False))
      rt2 = rt1 * [[10], [100], [1000]]
      return rt2.flat_values

    x = constant_op.constant([3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(ops.convert_to_tensor(g),
                        [10., 10., 10., 10., 100., 100., 100, 1000.])

  def testRaggedMul(self):
    # [FUNCTION DOC]
    # Scenario: multiplication of a ragged tensor with 64-bit row splits by a column of scalars.
    # Graph mode only. Asserts the flat values of the product equal the elementwise products.
    if context.executing_eagerly():
      return
    x = constant_op.constant([3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0])
    rt1 = RaggedTensor._from_row_partition(
        x,
        RowPartition.from_row_splits(row_splits=[0, 4, 7, 8],
                                     dtype=dtypes.int64, validate=False))
    rt2 = rt1 * [[10], [100], [1000]]
    self.assertAllClose(rt2.flat_values,
                        [30.0, 10.0, 40.0, 10.0, 100.0, 0.0, 200.0, 1000.0])

  def testBroadcastToGradient(self):
    # [FUNCTION DOC]
    # Scenario: gradient of the broadcast-to function. Graph mode only. Broadcasts a dense column
    # of three values to a ragged shape with row lengths four, three, and one, and asserts the
    # gradient of the flat values with respect to the input equals the number of copies of each
    # row (four, three, one).
    if context.executing_eagerly():
      return
    def func(x):
      # [FUNCTION DOC]
      # Nested function: builds a target shape from one row partition, broadcasts input x to it
      # and returns the flat values of the result.
      target_shape = DynamicRaggedShape.from_row_partitions(
          [RowPartition.from_row_splits(row_splits=[0, 4, 7, 8])])

      rt = dynamic_ragged_shape.broadcast_to(x, target_shape)
      return rt.flat_values

    x = constant_op.constant([[3.0], [1.0], [4.0]])
    y = func(x)
    g = gradients_impl.gradients(ys=y, xs=x)[0]

    self.assertAllClose(g, [[4.], [3.], [1.]])

  def testBroadcastScalarToScalar(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting a scalar string to the shape of a scalar. Asserts the result equals
    # the original.
    origin = constant_op.constant(b'x')
    expected = origin
    expected_shape = DynamicRaggedShape.from_tensor(expected)
    actual = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  @parameterized.parameters([
      dict(lengths=[2, 3], axis=0),
      dict(lengths=[2, 3], axis=1),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=0),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=2),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=3),
  ])
  def testIsUniformTrue(self, lengths, axis, num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: the is-uniform query returning true. Parameterized by lengths, axis and optional
    # number of row partitions. Builds the shape, optionally changes its row partition count, and
    # asserts that the axis is reported as uniform.
    shape = DynamicRaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      shape = shape._with_num_row_partitions(num_row_partitions)
    actual = shape.is_uniform(axis)
    self.assertTrue(actual)

  @parameterized.parameters([
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=1),
      dict(
          lengths=[2, (2, 3), 2, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 4],
          num_row_partitions=3,
          axis=3),
  ])
  def testIsUniformFalse(self, lengths, num_row_partitions, axis):
    # [FUNCTION DOC]
    # Scenario: the is-uniform query returning false for ragged axes. Builds the shape with the
    # given number of row partitions and asserts the answer for the axis is false.
    shape = DynamicRaggedShape.from_lengths(lengths)._with_num_row_partitions(
        num_row_partitions)
    actual = shape.is_uniform(axis)
    self.assertFalse(actual)

  @parameterized.parameters([
      dict(
          lengths=[2, (2, 3), 7, 4],
          num_row_partitions=2,
          axis=10,
          error_type=IndexError,
          error_regex='Expected axis=10 < rank=4'),
      dict(
          lengths=[2, (2, 3), 7, 4],
          num_row_partitions=2,
          axis=-1,
          error_type=IndexError,
          error_regex='Negative axis values are not supported'),
      dict(
          lengths=[2, (2, 3), 7, 4],
          num_row_partitions=2,
          axis=0.5,
          error_type=TypeError,
          error_regex='axis must be an integer'),
  ])
  def testIsUniformRaises(self, lengths, num_row_partitions, axis, error_type,
                          error_regex):
    # [FUNCTION DOC]
    # Scenario: invalid axis for the is-uniform query (too large, negative, non-integer).
    # Asserts the expected index or type error with a matching message is raised.
    shape = DynamicRaggedShape.from_lengths(lengths)._with_num_row_partitions(
        num_row_partitions)
    with self.assertRaisesRegex(error_type, error_regex):
      shape.is_uniform(axis)

  @parameterized.parameters([
      dict(lengths=[2, 3], num_row_partitions_a=0, num_row_partitions_b=1),
      dict(
          lengths=[2, (2, 3), 7, 4],
          num_row_partitions_a=2,
          num_row_partitions_b=1),
      dict(
          lengths=[3, (2, 0, 1), 5],
          num_row_partitions_a=1,
          num_row_partitions_b=2)
  ])
  def testWithNumRowPartitions(self, lengths, num_row_partitions_a,
                               num_row_partitions_b):
    # [FUNCTION DOC]
    # Scenario: round trip of the row-partition count. Changes a shape to count a, then to count
    # b, asserting each time that the count was applied, then returns to the original count and
    # asserts equality with the original shape.
    shape = DynamicRaggedShape.from_lengths(lengths)
    original_row_partitions = shape.num_row_partitions
    shape_a = shape._with_num_row_partitions(num_row_partitions_a)
    self.assertEqual(shape_a.num_row_partitions, num_row_partitions_a)
    shape_b = shape_a._with_num_row_partitions(num_row_partitions_b)
    self.assertEqual(shape_b.num_row_partitions, num_row_partitions_b)
    actual = shape_b._with_num_row_partitions(original_row_partitions)
    self.assertShapeEq(actual, shape)

  @parameterized.parameters([
      dict(
          lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=-2, expected=7),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=0, expected=2),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=2, expected=7),
      dict(lengths=[2, (2, 3), 7, 4], num_row_partitions=2, axis=3, expected=4),
      dict(
          lengths=[2, (2, 3), 7, 4, 3],
          num_row_partitions=2,
          axis=4,
          expected=3),
      dict(lengths=[3], axis=0, expected=3),
      dict(lengths=[3, 4, 5], axis=0, expected=3),
      dict(lengths=[3, 4, 5], axis=1, expected=4),
      dict(lengths=[3, 4, 5], axis=2, expected=5),
  ])
  def testGetItem(self, lengths, axis, expected, num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: indexing a shape with an integer axis. Parameterized by lengths, axis, expected
    # size and optional row partition count. Asserts the indexed size equals the expected value.
    shape = DynamicRaggedShape.from_lengths(lengths)
    if num_row_partitions is not None:
      shape = shape._with_num_row_partitions(num_row_partitions)
    actual = shape[axis]
    self.assertAllEqual(actual, expected)

  def testNumElements(self):
    # [FUNCTION DOC]
    # Scenario: total number of elements of a shape with lengths two, three, four, five and two
    # row partitions. Asserts the internal element count equals one hundred twenty.
    shape = DynamicRaggedShape.from_lengths([2, 3, 4,
                                             5])._with_num_row_partitions(2)
    self.assertAllEqual(shape._num_elements(), 120)

  def test_to_row_partitions_from_lengths(self):
    # [FUNCTION DOC]
    # HELPER: this tests the module-level test utility rather than library code ("testing the
    # test"). Builds partitions from lengths one, two, three and asserts they equal partitions
    # made from the expected row splits.
    # Testing the test.
    actual = _to_row_partitions_from_lengths([1, 2, 3])
    expected = [
        RowPartition.from_row_splits([0, 2]),
        RowPartition.from_row_splits([0, 3, 6])
    ]
    self.assertRowPartitionEq(actual[0], expected[0])
    self.assertRowPartitionEq(actual[1], expected[1])

  @parameterized.parameters([
      dict(
          origin=b'x',
          expected_lengths=[2, (1, 2)],
          expected=[[b'x'], [b'x', b'x']]),
      dict(
          origin=b'x',
          expected_lengths=[1, 1, 1],
          expected_num_row_partitions=2,
          expected=[[[b'x']]]),
      dict(
          origin=[b'a', b'b', b'c'],
          expected_lengths=[3],
          expected=[b'a', b'b', b'c']),
      dict(
          origin=[b'a', b'b', b'c'],
          expected_lengths=[1, 1, 3],
          expected_num_row_partitions=2,
          expected=[[[b'a', b'b', b'c']]]),
      dict(
          origin=[[b'a', b'b', b'c'], [b'd', b'e', b'f']],
          expected_lengths=[1, 2, 3],
          expected_num_row_partitions=2,
          expected=[[[b'a', b'b', b'c'], [b'd', b'e', b'f']]]),
  ])
  def testBroadcastTensorTo(self,
                            origin,
                            expected_lengths,
                            expected,
                            expected_num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: broadcasting a dense tensor (scalar or small) to a shape created from lengths with
    # an optional number of row partitions. Asserts the broadcast result equals the expected
    # ragged value.
    origin = constant_op.constant(origin)
    expected_shape = DynamicRaggedShape.from_lengths(expected_lengths)
    if expected_num_row_partitions is not None:
      expected_shape = expected_shape._with_num_row_partitions(
          expected_num_row_partitions)
    expected = ragged_factory_ops.constant_value(expected)
    actual = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  def testBroadcastFlatValues(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting flat values between a source and a destination shape with extra
    # leading dimensions of size one. Gets the broadcaster and asserts that broadcasting the flat
    # values returns values equal to the original flat values.
    origin_lengths = [3, (1, 2, 1), 2, 2]
    dest_lengths = [1, 1, 3, (1, 2, 1), 2, 2]
    origin_values = constant_op.constant([
        b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l',
        b'm', b'n', b'o', b'p'
    ])
    origin_shape = DynamicRaggedShape.from_lengths(
        origin_lengths)._with_num_row_partitions(3)
    dest_shape = DynamicRaggedShape.from_lengths(
        dest_lengths)._with_num_row_partitions(5)

    broadcaster = dynamic_ragged_shape._get_broadcaster(origin_shape,
                                                        dest_shape)

    actual = broadcaster.broadcast_flat_values(origin_values)

    self.assertAllEqual(origin_values, actual)

  @parameterized.parameters([
      dict(
          origin_lengths=[3],
          origin_values=[b'a', b'b', b'c'],
          expected_lengths=[2],
          expected_values=[[b'a', b'b', b'c'], [b'a', b'b', b'c']]),
      dict(
          origin_lengths=[3, (3, 2, 4)],
          origin_values=[7, 4, 5, 6, 1, 2, 3, 7, 89],
          expected_lengths=[3, (3, 2, 4)],
          expected_values=[7, 4, 5, 6, 1, 2, 3, 7, 89]),
      dict(
          origin_lengths=[3, (3, 2, 4)],
          origin_values=[7, 4, 5, 6, 1, 2, 3, 7, 89],
          expected_lengths=[1, 3, (3, 2, 4)],
          expected_values=[7, 4, 5, 6, 1, 2, 3, 7, 89]),
      dict(
          origin_lengths=[3, (3, 2, 4)],
          origin_values=[7, 4, 5, 6, 1, 2, 3, 7, 89],
          expected_lengths=[1, 1, 3, (3, 2, 4)],
          expected_values=[7, 4, 5, 6, 1, 2, 3, 7, 89]),
      # Broadcast [1, 2, (1, 2)] to [2, 2, (1, 2, 1, 2)]
      dict(
          origin_lengths=[1, 2, (1, 2)],
          origin_values=[2, 3, 5],
          expected_lengths=[2, 2, (1, 2, 1, 2)],
          expected_values=[2, 3, 5, 2, 3, 5]),
      # Broadcast [2, 1, (1, 2)] to [2, 2, (1, 1, 2, 2)] (NEW)
      dict(
          origin_lengths=[2, 1, (1, 2)],
          origin_values=[2, 3, 5],
          expected_lengths=[2, 2, (1, 1, 2, 2)],
          expected_values=[2, 2, 3, 5, 3, 5]),
      dict(
          origin_lengths=[2, 1, 1],
          origin_values=[2, 3],  # [[[2]], [[3]]]
          expected_lengths=[2, 1, (3, 3)],
          expected_values=[2, 2, 2, 3, 3, 3]),
      dict(
          origin_lengths=[3],
          origin_values=[b'a', b'b', b'c'],
          expected_lengths=[4, 2, 3],
          expected_values=[
              b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b',
              b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a',
              b'b', b'c'
          ]),
      dict(
          origin_lengths=[2, 3],
          origin_values=[b'a', b'b', b'c', b'a', b'b', b'c'],
          expected_lengths=[4, 2, 3],
          expected_values=[
              b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b',
              b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c', b'a',
              b'b', b'c'
          ]),
      dict(
          origin_lengths=[3, (1, 2, 1), 2, 2],
          origin_values=[
              b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k',
              b'l', b'm', b'n', b'o', b'p'
          ],
          expected_lengths=[1, 1, 3, (1, 2, 1), 2, 2],
          expected_values=[
              b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k',
              b'l', b'm', b'n', b'o', b'p'
          ]),
      dict(
          origin_lengths=[3, (1, 2, 1), 2, 2],
          origin_values=[7, 4, 5, 6, 1, 2, 3, 7, 7, 4, 5, 6, 1, 2, 3, 7],
          expected_lengths=[1, 1, 3, (1, 2, 1), 2, 2],
          expected_values=[7, 4, 5, 6, 1, 2, 3, 7, 7, 4, 5, 6, 1, 2, 3, 7],
      ),
  ])
  def testBroadcastRaggedTo(self, origin_lengths, origin_values,
                            expected_lengths, expected_values):
    # [FUNCTION DOC]
    # Scenario: broadcasting ragged tensors to ragged target shapes. Builds an origin and an
    # expected tensor from lengths and values, takes the expected tensor's shape, broadcasts the
    # origin to it, and asserts equality with the expected.
    origin = _to_ragged_tensor_from_lengths(origin_values, origin_lengths)
    expected = _to_ragged_tensor_from_lengths(expected_values, expected_lengths)
    expected_shape = DynamicRaggedShape.from_tensor(expected)
    actual = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
    self.assertAllEqual(actual, expected)

  def testDynamicRaggedShapeFromTensor2(self):
    # [FUNCTION DOC]
    # Scenario: shape of a nested ragged tensor made from a constant value. Asserts the shape
    # computed from the tensor equals a shape from lengths with three row partitions.
    raw_rt = [[[[7, 4], [5, 6]], [[1, 2], [3, 7]]], [[[7, 4], [5, 6]]],
              [[[1, 2], [3, 7]]]]
    raw_rt = ragged_factory_ops.constant_value(raw_rt)
    actual_shape = DynamicRaggedShape.from_tensor(raw_rt)
    expected_shape = DynamicRaggedShape.from_lengths(
        [3, (2, 1, 1), 2, 2])._with_num_row_partitions(3)
    self.assertShapeEq(actual_shape, expected_shape)

  # pylint: disable=g-long-lambda
  @parameterized.parameters([
      # A row partition as opposed to a list of row partitions.
      dict(
          row_partitions=lambda: RowPartition.from_row_splits([0, 2, 3]),
          inner_shape=lambda: [4],
          error_type=TypeError,
          error_regex='row_partitions should be'),
      # A list of lists of integers for row_partitions.
      dict(
          row_partitions=lambda: [[0, 2, 3]],
          inner_shape=lambda: [4],
          error_type=TypeError,
          error_regex='row_partitions contains'),
      # nvals and nrows don't match (3 != 6) statically
      dict(
          row_partitions=lambda: [  # pylint: disable=g-long-lambda
              RowPartition.from_value_rowids([0, 2, 4], nrows=5),
              RowPartition.from_value_rowids([0, 2, 5], nrows=6)
          ],
          inner_shape=lambda: [3],
          validate=False,
          error_type=ValueError,
          error_regex='RowPartitions in DynamicRaggedShape do not'),
      # nvals and inner_shape[0] don't match (3 != 6) statically
      dict(
          row_partitions=lambda: [
              RowPartition.from_value_rowids([0, 2, 4], nrows=5),
          ],
          inner_shape=lambda: [6],
          validate=False,
          error_type=ValueError,
          error_regex='Last row partition does not match inner_shape.'),
  ])
  def testConstructorRaisesStatic(self,
                                  row_partitions,
                                  inner_shape,
                                  error_type,
                                  error_regex,
                                  validate=False,
                                  dtype=None):
    # [FUNCTION DOC]
    # Scenario: the constructor rejects invalid or inconsistent row partitions and inner shapes
    # statically (wrong container types, mismatched row counts and value counts). Parameters are
    # callables creating the inputs. Asserts the expected error type and message.
    row_partitions = row_partitions()
    inner_shape = inner_shape()
    with self.assertRaisesRegex(error_type, error_regex):
      DynamicRaggedShape(
          row_partitions, inner_shape, dtype=dtype, validate=validate)

  def testConstructorStaticOK(self):
    # [FUNCTION DOC]
    # Scenario: the constructor with valid statically known partitions and validation on.
    # Verifies the static values of the inner shape and the value row ids of both partitions can
    # still be recovered as constants.
    row_partitions = [
        RowPartition.from_value_rowids([0, 2, 4], nrows=5),
        RowPartition.from_value_rowids([0, 1, 2], nrows=3)
    ]
    inner_shape = [3]
    rts = DynamicRaggedShape(row_partitions, inner_shape, validate=True)
    static_inner_shape = tensor_util.constant_value(rts.inner_shape)
    static_valid_rowids0 = tensor_util.constant_value(
        rts.row_partitions[0].value_rowids())
    static_valid_rowids1 = tensor_util.constant_value(
        rts.row_partitions[1].value_rowids())
    self.assertAllEqual(static_inner_shape, [3])
    self.assertAllEqual(static_valid_rowids0, [0, 2, 4])
    self.assertAllEqual(static_valid_rowids1, [0, 1, 2])

  def testConstructorWithStaticInnerShape(self):
    # [FUNCTION DOC]
    # Scenario: the constructor given an explicit static inner shape with validation on. Verifies
    # constants for the inner shape and row ids can still be extracted.
    row_partitions = [
        RowPartition.from_value_rowids([0, 2, 4], nrows=5),
        RowPartition.from_value_rowids([0, 1, 2], nrows=3)
    ]
    inner_shape = [3]
    rts = DynamicRaggedShape(row_partitions, inner_shape, validate=True,
                             static_inner_shape=[3])
    static_inner_shape = tensor_util.constant_value(rts.inner_shape)
    static_valid_rowids0 = tensor_util.constant_value(
        rts.row_partitions[0].value_rowids())
    static_valid_rowids1 = tensor_util.constant_value(
        rts.row_partitions[1].value_rowids())
    self.assertAllEqual(static_inner_shape, [3])
    self.assertAllEqual(static_valid_rowids0, [0, 2, 4])
    self.assertAllEqual(static_valid_rowids1, [0, 1, 2])

  def testZeros(self):
    # [FUNCTION DOC]
    # Scenario: the zeros function given a ragged shape. Asserts the shape of the result matches
    # and the flat values are zeros of the expected size.
    shape_x = DynamicRaggedShape.from_lengths([3, (1, 3, 2), 4])
    foo = ragged_array_ops.zeros(shape_x)
    self.assertShapeEq(shape_x, DynamicRaggedShape.from_tensor(foo))
    self.assertAllEqual(array_ops.zeros([6, 4]), foo.flat_values)

  def testOnes(self):
    # [FUNCTION DOC]
    # Scenario: the ones function given a ragged shape. Asserts the result has that shape and flat
    # values are ones of the expected size.
    shape_x = DynamicRaggedShape.from_lengths([3, (1, 3, 2), 4])
    foo = ragged_array_ops.ones(shape_x)
    self.assertShapeEq(shape_x, DynamicRaggedShape.from_tensor(foo))
    self.assertAllEqual(array_ops.ones([6, 4]), foo.flat_values)

  def testReshapeTensor(self):
    # [FUNCTION DOC]
    # Scenario: ragged reshape of a dense tensor to a ragged shape. Asserts the result shape
    # matches and the flat values are zeros of the expected size.
    foo = array_ops.zeros([3, 2, 4])
    shape_b = DynamicRaggedShape.from_lengths([3, (3, 2, 1), 4])
    result = ragged_array_ops.ragged_reshape(foo, shape_b)
    self.assertShapeEq(shape_b, DynamicRaggedShape.from_tensor(result))
    self.assertAllEqual(array_ops.zeros([6, 4]), result.flat_values)

  def test_reshape_ragged_tensor(self):
    # [FUNCTION DOC]
    # Scenario: ragged reshape of a ragged tensor to a different ragged shape. Asserts the result
    # shape matches and flat values are the expected zeros.
    shape_x = DynamicRaggedShape.from_lengths([3, (1, 3, 2), 4])
    foo = ragged_array_ops.zeros(shape_x)
    shape_b = DynamicRaggedShape.from_lengths([3, (3, 2, 1), 4])
    result = ragged_array_ops.ragged_reshape(foo, shape_b)
    self.assertShapeEq(shape_b, DynamicRaggedShape.from_tensor(result))
    self.assertAllEqual(array_ops.zeros([6, 4]), result.flat_values)

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)],
          lengths_b=[3, (1, 4, 2)],
          lengths_e=[3, (1, 4, 2)]),
      dict(
          lengths_a=[1, 2, (1, 4)],
          lengths_b=[3, 2, (1, 4, 1, 4, 1, 4)],
          lengths_e=[3, 2, (1, 4, 1, 4, 1, 4)]),
      dict(
          lengths_a=[1, 1],
          num_row_partitions_a=1,
          lengths_b=[3, 5],
          num_row_partitions_b=1,
          lengths_e=[3, 5],
          num_row_partitions_e=1),
      dict(lengths_a=[1, 4, 5], lengths_b=[3, 1, 1], lengths_e=[3, 4, 5]),
      dict(lengths_a=[3], lengths_b=[4, 2, 1], lengths_e=[4, 2, 3]),
      dict(lengths_a=[2, 3], lengths_b=[4, 2, 1], lengths_e=[4, 2, 3]),
      # Outermost dimension-both partitioned
      # Also, neither has uniform_row_length
      dict(
          lengths_a=[2, (1, 3), 1],
          lengths_b=[2, (1, 3), (3, 4, 5, 6)],
          lengths_e=[2, (1, 3), (3, 4, 5, 6)]),
      # Outermost dimension-Only one is partitioned
      # Also, partitioned dimension doesn't have uniform_row_length
      dict(
          lengths_a=[2, 1, 5],
          lengths_b=[2, (1, 3), 5],
          num_row_partitions_b=2,
          lengths_e=[2, (1, 3), 5],
          num_row_partitions_e=2),

      # Cover [5, R], [1, 5, R]
      dict(
          lengths_a=[5, (1, 2, 0, 3, 1)],
          lengths_b=[1, 5, (1, 2, 0, 3, 1)],
          lengths_e=[1, 5, (1, 2, 0, 3, 1)]),
      # When two uniform row lengths are equal
      dict(
          lengths_a=[1, 5],
          num_row_partitions_a=1,
          lengths_b=[3, 5],
          num_row_partitions_b=1,
          lengths_e=[3, 5],
          num_row_partitions_e=1),
      # Dense + Partitioned dimension has uniform_row_length
      # [1, 3, [5, 1, 6]] and DENSE [2, 1, 1] -> [2, 3, [5, 1, 6, 5, 1, 6]]
      dict(
          lengths_a=[1, 3, (5, 1, 6)],
          lengths_b=[2, 1, 1],
          lengths_e=[2, 3, (5, 1, 6, 5, 1, 6)]),
      # Both partitioned; one has uniform_row_length
      # (uniform_row_length [2,1,1]) and [2,[1,3],[3,4,5,6]]
      dict(
          lengths_a=[2, 1, 1],
          num_row_partitions_a=2,
          lengths_b=[2, (1, 3), (3, 4, 5, 6)],
          lengths_e=[2, (1, 3), (3, 4, 5, 6)]),
      # When broadcasting uniform_row_length to uniform_row_length.
      # Also, both have uniform_row_length
      dict(
          lengths_a=[3, 1, 5],
          num_row_partitions_a=2,
          lengths_b=[3, 4, 5],
          num_row_partitions_b=2,
          lengths_e=[3, 4, 5],
          num_row_partitions_e=2),
      # When broadcasting above a U_R_L
      # [2,1, 5] and [2, [1,3], 5] -> [2, [1,3], 5]
      dict(
          lengths_a=[2, 1, 5],
          num_row_partitions_a=2,
          lengths_b=[2, (1, 3), 5],
          num_row_partitions_b=2,
          lengths_e=[2, (1, 3), 5],
          num_row_partitions_e=2),
      # What if the larger-dimensional shape has uniform_row_length on the
      # matching dim, but has larger dimensions above
      # ([3,1,5],[15]) vs ([2,1],[2]))
      dict(
          lengths_a=[3, 1, 5],
          num_row_partitions_a=2,
          lengths_b=[2, 1],
          num_row_partitions_b=1,
          lengths_e=[3, 2, 5],
          num_row_partitions_e=2),
      # Inner non-ragged dimensions
      # Can delegate to dense broadcast operations.
      # Implementation detail: not testable.
      # ([2, [1,2]],[3,2,1]) and ([2,1],[2,1,3])
      dict(
          lengths_a=[2, (1, 2), 2, 1],
          lengths_b=[2, 1, 1, 3],
          num_row_partitions_b=1,
          lengths_e=[2, (1, 2), 2, 3],
      ),
  ])
  def testBroadcastDynamicShapeExtended(self,
                                        lengths_a,
                                        lengths_b,
                                        lengths_e,
                                        num_row_partitions_a=None,
                                        num_row_partitions_b=None,
                                        num_row_partitions_e=None):
    # [FUNCTION DOC]
    # Scenario: extended broadcast of two shapes in both argument orders. Builds the shapes from
    # lengths (with optional row-partition counts), asserts the broadcast shape equals the
    # expected shape, and then checks each returned broadcaster by broadcasting prime-filled
    # tensors and comparing with the broadcast-to function result.
    # This test is predicated on the fact that broadcast_to is correct.
    # Thus, it tests:
    # Whether the shape generated is correct.
    # Whether broadcasting is the same as broadcast_to.
    # Instead of specifying values, it just uses primes.
    shape_a = DynamicRaggedShape.from_lengths(lengths_a)
    if num_row_partitions_a is not None:
      shape_a = shape_a._with_num_row_partitions(num_row_partitions_a)
    shape_b = DynamicRaggedShape.from_lengths(lengths_b)
    if num_row_partitions_b is not None:
      shape_b = shape_b._with_num_row_partitions(num_row_partitions_b)
    shape_e = DynamicRaggedShape.from_lengths(lengths_e)
    if num_row_partitions_e is not None:
      shape_e = shape_e._with_num_row_partitions(num_row_partitions_e)

    [actual, bc_a, bc_b
    ] = dynamic_ragged_shape.broadcast_dynamic_shape_extended(shape_a, shape_b)
    [actual_rev, bc_b_rev, bc_a_rev
    ] = dynamic_ragged_shape.broadcast_dynamic_shape_extended(shape_b, shape_a)
    self.assertShapeEq(actual, shape_e)
    self.assertShapeEq(actual_rev, shape_e)

    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a)
    bc_a_actual = bc_a.broadcast(rt_a)
    bc_a_actual_rev = bc_a_rev.broadcast(rt_a)
    bc_a_expected = dynamic_ragged_shape.broadcast_to(rt_a, shape_e)
    self.assertAllEqual(bc_a_expected, bc_a_actual)
    self.assertAllEqual(bc_a_expected, bc_a_actual_rev)

    rt_b = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_b)), shape_b)
    bc_b_expected = dynamic_ragged_shape.broadcast_to(rt_b, shape_e)
    bc_b_actual = bc_b.broadcast(rt_b)
    bc_b_actual_rev = bc_b_rev.broadcast(rt_b)
    self.assertAllEqual(bc_b_expected, bc_b_actual)
    self.assertAllEqual(bc_b_expected, bc_b_actual_rev)

  @parameterized.parameters([
      dict(
          lengths=[3, (1, 4, 2)],
          dense_rank=1,
          lengths_e=[3, (1, 4, 2)],
      ),
      dict(
          lengths=[3, (1, 4, 2), 5],
          dense_rank=2,
          lengths_e=[3, (1, 4, 2), 5],
      ),
      dict(
          lengths=[3],
          dense_rank=1,
          lengths_e=[3],
      ),
  ])
  def testWithDenseRank(self, lengths, dense_rank, lengths_e):
    # [FUNCTION DOC]
    # Scenario: setting the inner rank (called a dense rank in the name). Asserts the inner rank
    # equals the requested one and the static lengths equal the expected lengths.
    # Makes little sense with from_lengths/_with_num_row_partitions.
    original = DynamicRaggedShape.from_lengths(lengths)
    actual = original._with_inner_rank(dense_rank)
    self.assertAllEqual(actual.inner_rank, dense_rank)
    self.assertAllEqual(actual.static_lengths(), lengths_e)

  @parameterized.parameters([
      dict(
          rps=[3, [1, 4, 2]],
          lengths_e=[3, (1, 4, 2)],
          num_row_partitions_e=1,
      ),
      dict(
          rps=[3, [1, 4, 2], 2],
          lengths_e=[3, (1, 4, 2), 2],
          num_row_partitions_e=2,
      ),
  ])
  def testFromRowPartitions(self, rps, lengths_e, num_row_partitions_e):
    # [FUNCTION DOC]
    # Scenario: building a shape from row partitions made from lengths. Asserts it equals a shape
    # from expected lengths with the expected number of row partitions.
    rps = _to_row_partitions_from_lengths(rps)
    actual = DynamicRaggedShape.from_row_partitions(rps)
    expected = DynamicRaggedShape.from_lengths(
        lengths_e)._with_num_row_partitions(num_row_partitions_e)
    self.assertShapeEq(expected, actual)

  def testFromRowPartitionsError(self):
    # [FUNCTION DOC]
    # Scenario: building a shape from an empty list of row partitions. Asserts a value error
    # saying the partitions cannot be empty.
    with self.assertRaisesRegex(ValueError, 'row_partitions cannot be empty'):
      DynamicRaggedShape.from_row_partitions([])

  @parameterized.parameters([
      #=========================================================================
      # dimension[axis] is uniform inner; and row_lengths is a scalar
      #=========================================================================
      # shape: [BROADCAST(UNIFORM), UNIFORM, UNIFORM]
      dict(original_lengths=[1, 4, 5],
           broadcast_lengths=[3, 4, 5]),
      # shape: [UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(original_lengths=[3, 4, 1],
           broadcast_lengths=[3, 4, 5]),
      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM)]
      dict(original_lengths=[3, (3, 2, 8), 1],
           broadcast_lengths=[3, (3, 2, 8), 5]),
      # shape: [UNIFORM, RAGGED, RAGGED, UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(original_lengths=[2, (2, 1), (3, 2, 8), 3, 4, 1],
           broadcast_lengths=[2, (2, 1), (3, 2, 8), 3, 4, 5]),

      #=========================================================================
      # dimension[axis] is uniform inner; and row_lengths is a vector
      #=========================================================================
      # shape: [UNIFORM, BROADCAST(UNIFORM)]
      dict(original_lengths=[3, 1],
           broadcast_lengths=[3, (2, 0, 1)]),
      # shape: [UNIFORM, BROADCAST(UNIFORM), UNIFORM]
      dict(original_lengths=[3, 1, 5],
           broadcast_lengths=[3, (2, 0, 1), 5]),

      # shape: [UNIFORM, UNIFORM, BROADCAST(UNIFORM)]
      dict(original_lengths=[4, 3, 1],
           broadcast_lengths=[4, 3, (2, 0, 1, 3, 8, 2, 3, 4, 1, 8, 7, 0)]),

      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM)]
      dict(original_lengths=[2, (2, 1), 1],
           broadcast_lengths=[2, (2, 1), (2, 5, 3)]),

      # shape: [UNIFORM, RAGGED, UNIFORM, UNIFORM, BROADCAST(UNIFORM), UNIFORM]
      dict(original_lengths=[2, (2, 1), 3, 2, 1, 8],
           broadcast_lengths=[2, (2, 1), 3, 2, tuple(range(18)), 8]),

      #=========================================================================
      # dimension[axis] is uniform partitioned; and row_lengths is a scalar
      #=========================================================================
      # shape: [BROADCAST(UNIFORM), RAGGED]
      dict(original_lengths=[1, (5,)],
           broadcast_lengths=[3, (5, 5, 5)]),

      # shape: [BROADCAST(UNIFORM), UNIFORM, RAGGED]
      dict(original_lengths=[1, 3, (3, 0, 2)],
           broadcast_lengths=[2, 3, (3, 0, 2, 3, 0, 2)]),

      # shape: [BROADCAST(UNIFORM), RAGGED, RAGGED, UNIFORM, UNIFORM]
      dict(original_lengths=[1, (3,), (3, 5, 2), 9, 4, 5],
           broadcast_lengths=[3, (3, 3, 3), (3, 5, 2, 3, 5, 2, 3, 5, 2),
                              9, 4, 5]),

      # shape: [BROADCAST(UNIFORM), UNIFORM, RAGGED, UNIFORM]
      dict(original_lengths=[1, 2, (2, 1), (3, 5, 2), 2],
           broadcast_lengths=[2, 2, (2, 1, 2, 1), (3, 5, 2, 3, 5, 2), 2]),

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, UNIFORM]
      # This is wrong. should broadcast to [3, 2, (4, 4, 0, 0, 2, 2), 5]
      # dict(original_lengths=[3, 1, [4, 0, 2], 5],
      #      broadcast_lengths=[3, 2, [4, 0, 2, 4, 0, 2], 5]),
      dict(original_lengths=[3, 1, (4, 0, 2), 5],
           broadcast_lengths=[3, 2, (4, 4, 0, 0, 2, 2), 5]),

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED]
      dict(original_lengths=[2, 3, (1, 2, 3, 4, 5, 6)],
           broadcast_lengths=[2, 3, (1, 2, 3, 4, 5, 6)]),

      #=========================================================================
      # dimension[axis] is uniform partitioned; and row_lengths is a vector
      #=========================================================================
      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, UNIFORM]
      dict(original_lengths=[
          3,                          # axis=0
          1,                          # axis=1 (broadcast)
          (3, 1, 2),                  # axis=2
          5],                         # axis=3
           broadcast_lengths=[
               3,                          # axis=0
               (4, 1, 2),                  # axis=1 (broadcast)
               (3, 3, 3, 3, 1, 2, 2),      # axis=2
               5]),                        # axis=3

      # shape: [UNIFORM, BROADCAST(UNIFORM), RAGGED, RAGGED]
      dict(original_lengths=[
          3,                                         # axis=0
          1,                                         # axis=1 (broadcast)
          (3, 1, 2),                                 # axis=2
          (3, 1, 4, 1, 5, 9)],                       # axis=3
           broadcast_lengths=[
               3,                                         # axis=0
               (2, 0, 3),                                 # axis=1 (broadcast)
               (3, 3, 2, 2, 2),                           # axis=2
               (3, 1, 4, 3, 1, 4, 5, 9, 5, 9, 5, 9)]),    # axis=3

      # shape: [UNIFORM, RAGGED, BROADCAST(UNIFORM), RAGGED, RAGGED, UNIFORM]
      dict(original_lengths=[
          3,                                         # axis=0
          (2, 0, 1),                                 # axis=1
          1,                                         # axis=2 (broadcast)
          (3, 2, 1),                                 # axis=3
          (1, 0, 1, 0, 2, 3),                        # axis=4
          5],                                        # axis=5
           broadcast_lengths=[
               3,                                         # axis=0
               (2, 0, 1),                                 # axis=2
               (4, 1, 2),                                 # axis=2 (broadcast)
               (3, 3, 3, 3, 2, 1, 1),                     # axis=3
               (1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,    # axis=4
                2, 3, 3),
               5]),                                       # axis=5
      dict(original_lengths=[1, 1, 2, (2, 1)],
           broadcast_lengths=[2, 1, 2, (2, 1, 2, 1)]),
      dict(original_lengths=[2, 1, 2, (2, 1, 2, 1)],
           broadcast_lengths=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
      dict(original_lengths=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)],
           broadcast_lengths=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
      dict(original_lengths=[2, (2, 1), 2, 1],
           broadcast_lengths=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
  ])  # pyformat: disable
  def testBroadcastDimension(self, original_lengths, broadcast_lengths):
    # [FUNCTION DOC]
    # Scenario: broadcasting along a single dimension. Builds a prime-filled tensor of the
    # original lengths, broadcasts it to the shape of the broadcast lengths and asserts the
    # result's shape equals the intended shape.
    """Tests broadcast_to on a single dimension."""
    original_rt = _to_prime_tensor_from_lengths(original_lengths)
    bcast_shape = DynamicRaggedShape.from_lengths(broadcast_lengths)
    result_rt = dynamic_ragged_shape.broadcast_to(original_rt, bcast_shape)
    result_shape = DynamicRaggedShape.from_tensor(result_rt)

    self.assertShapeEq(bcast_shape, result_shape)

  def testAsRowPartitions(self):
    # [FUNCTION DOC]
    # Scenario: internal conversion of a shape to row partitions. For a shape with two ragged
    # levels asserts that two partitions are returned.
    my_shape = DynamicRaggedShape.from_lengths([3, (2, 0, 1), 5])
    rps = my_shape._as_row_partitions()
    self.assertLen(rps, 2)

  def testAsRowPartitionsRaises(self):
    # [FUNCTION DOC]
    # Scenario: converting a scalar shape to row partitions. Asserts a value error saying rank
    # must be at least one.
    my_shape = DynamicRaggedShape.from_lengths([])
    with self.assertRaisesRegex(ValueError,
                                'rank must be >= 1 for _as_row_partitions'):
      my_shape._as_row_partitions()

  def testToPrimeTensorFromDimSizes(self):
    # [FUNCTION DOC]
    # Scenario: the prime tensor test utility. Builds a prime tensor from lengths and asserts it
    # equals the expected ragged tensor of the first thirteen primes.
    """Tests the test utility."""
    original_lengths = [3, (3, 2, 8), 1]
    original_rt = _to_prime_tensor_from_lengths(original_lengths)
    expected_rt = _to_ragged_tensor_from_lengths(
        [[2], [3], [5], [7], [11], [13], [17], [19], [23], [29], [31], [37],
         [41]], [3, (3, 2, 8)])

    self.assertAllEqual(expected_rt, original_rt)

  @parameterized.parameters([
      # Broadcast scalar
      dict(x_dims=[], y_dims=[], expected_dims=[]),
      dict(x_dims=[], y_dims=[2], expected_dims=[2]),
      dict(x_dims=[], y_dims=[2, 3], expected_dims=[2, 3]),
      dict(
          x_dims=[],
          y_dims=[2, (2, 3), (5, 7, 2, 0, 9)],
          expected_dims=[2, (2, 3), (5, 7, 2, 0, 9)]),
      # Broadcast vector
      dict(x_dims=[3], y_dims=[4, 2, 3], expected_dims=[4, 2, 3]),
      dict(x_dims=[1], y_dims=[4, 2, 3], expected_dims=[4, 2, 3]),
      dict(x_dims=[3], y_dims=[4, 2, 1], expected_dims=[4, 2, 3]),
      dict(
          x_dims=[3], y_dims=[3, (2, 3, 1), 1], expected_dims=[3, (2, 3, 1),
                                                               3]),
      dict(x_dims=[1], y_dims=[3, (2, 1, 3)], expected_dims=[3, (2, 1, 3)]),
      dict(
          x_dims=[1], y_dims=[3, (2, 1, 3), 8], expected_dims=[3, (2, 1, 3),
                                                               8]),
      dict(
          x_dims=[1],
          y_dims=[2, (2, 3), (5, 7, 2, 0, 9)],
          expected_dims=[2, (2, 3), (5, 7, 2, 0, 9)]),
      # Mixed broadcasting
      dict(
          x_dims=[
              1,  # axis=0
              3,  # axis=1
              (3, 0, 2),  # axis=2
              1,  # axis=3
              2,  # axis=4
          ],
          y_dims=[
              2,  # axis=0
              1,  # axis=1
              1,  # axis=2
              (7, 2),  # axis=3
              1,  # axis=4
          ],
          expected_dims=[
              2,  # axis=0
              3,  # axis=1
              (3, 0, 2, 3, 0, 2),  # axis=2
              (7, 7, 7, 7, 7, 2, 2, 2, 2, 2),  # axis=3
              2,  # axis=4
          ]),
      dict(
          x_dims=[2, (2, 1), 2, 1],
          y_dims=[1, 1, 2, (2, 1)],
          expected_dims=[2, (2, 1), 2, (2, 1, 2, 1, 2, 1)]),
  ])
  def testBroadcastDynamicShape(self, x_dims, y_dims, expected_dims):
    # [FUNCTION DOC]
    # Scenario: broadcasting two shapes from lengths. For both argument orders asserts the shape
    # equals the expected one and that the two broadcasters, applied to prime tensors, equal the
    # broadcast-to result. Also asserts the two wrapper functions return the expected shape.
    shape_a = DynamicRaggedShape.from_lengths(x_dims)
    shape_b = DynamicRaggedShape.from_lengths(y_dims)
    shape_e = DynamicRaggedShape.from_lengths(expected_dims)
    [actual, bc_a, bc_b
    ] = dynamic_ragged_shape.broadcast_dynamic_shape_extended(shape_a, shape_b)
    [actual_rev, bc_b_rev, bc_a_rev
    ] = dynamic_ragged_shape.broadcast_dynamic_shape_extended(shape_b, shape_a)
    self.assertShapeEq(actual, shape_e)
    self.assertShapeEq(actual_rev, shape_e)

    rt_a = _to_prime_tensor_from_lengths(x_dims)
    bc_a_actual = bc_a.broadcast(rt_a)
    bc_a_actual_rev = bc_a_rev.broadcast(rt_a)
    bc_a_expected = dynamic_ragged_shape.broadcast_to(rt_a, shape_e)
    self.assertAllEqual(bc_a_expected, bc_a_actual)
    self.assertAllEqual(bc_a_expected, bc_a_actual_rev)

    rt_b = _to_prime_tensor_from_lengths(y_dims)
    bc_b_expected = dynamic_ragged_shape.broadcast_to(rt_b, shape_e)
    bc_b_actual = bc_b.broadcast(rt_b)
    bc_b_actual_rev = bc_b_rev.broadcast(rt_b)
    self.assertAllEqual(bc_b_expected, bc_b_actual)
    self.assertAllEqual(bc_b_expected, bc_b_actual_rev)

    # This just wraps broadcast_dynamic_shape_extended, so nothing
    # deeper is required.
    result1 = dynamic_ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
    self.assertShapeEq(shape_e, result1)

    # Again, just a wrapper.
    result2 = ragged_array_ops.broadcast_dynamic_shape(shape_a, shape_b)
    self.assertShapeEq(shape_e, result2)

  def testBroadcastDynamicShapeFirstLayer(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting the first layer of two shapes (sizes one and three). Asserts the
    # layer broadcasters have the expected gather indices.
    a_0 = constant_op.constant(1, dtypes.int64)
    b_0 = constant_op.constant(3, dtypes.int64)
    [a_layer, b_layer
    ] = dynamic_ragged_shape._broadcast_dynamic_shape_first_layer(a_0, b_0)
    expected_a_layer = _LayerBroadcaster.from_gather_index([0, 0, 0])
    expected_b_layer = _LayerBroadcaster.from_gather_index([0, 1, 2])
    self.assertLayerBroadcasterEq(expected_a_layer, a_layer)
    self.assertLayerBroadcasterEq(expected_b_layer, b_layer)

  def testBroadcastDynamicShapeNextLayer(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says next layer but the test calls the half-ragged variant and
    # makes no assertion; it only checks that the call does not raise.
    # Scenario: builds a uniform partition and a row-length partition with layer broadcasters and
    # calls the internal half-ragged next-layer function.
    a_1 = RowPartition.from_uniform_row_length(
        1, nvals=1, nrows=1, dtype_hint=dtypes.int64)
    b_1 = RowPartition.from_row_lengths([2, 1, 3], dtype_hint=dtypes.int64)
    ac_0 = _LayerBroadcaster.from_gather_index(
        constant_op.constant([0, 0, 0], dtype=dtypes.int64))
    bc_0 = _LayerBroadcaster.from_gather_index(
        constant_op.constant([0, 1, 2], dtype=dtypes.int64))
    dynamic_ragged_shape._broadcast_dynamic_shape_next_layer_half_ragged(
        ac_0, bc_0, a_1, b_1)

  def testBroadcastDynamicShapeRaisesLeft(self):
    # [FUNCTION DOC]
    # Scenario: the first argument to broadcast-dynamic-shape is not a shape. Asserts a type
    # error mentioning shape_x.
    shape = DynamicRaggedShape.from_tensor(constant_op.constant([1, 2, 3]))
    with self.assertRaisesRegex(TypeError, 'shape_x must be'):
      dynamic_ragged_shape.broadcast_dynamic_shape(1, shape)

  def testBroadcastDynamicShapeRaisesRight(self):
    # [FUNCTION DOC]
    # Scenario: the second argument to broadcast-dynamic-shape is not a shape. Asserts a type
    # error mentioning shape_y.
    shape = DynamicRaggedShape.from_tensor(constant_op.constant([1, 2, 3]))
    with self.assertRaisesRegex(TypeError, 'shape_y must be'):
      dynamic_ragged_shape.broadcast_dynamic_shape(shape, 1)

  def testBroadcastToRaises(self):
    # [FUNCTION DOC]
    # Scenario: broadcast-to with a target that is not a shape. Asserts a type error.
    rt = constant_op.constant([1, 2, 3])
    with self.assertRaisesRegex(TypeError, 'shape must be'):
      dynamic_ragged_shape.broadcast_to(rt, 1)

  @parameterized.parameters([
      dict(
          x=[[10], [20], [30]],  # shape=[3, 1]
          lengths=[3, 2],
          expected=[[10, 10], [20, 20], [30, 30]]),
      dict(
          x=[[10], [20], [30]],  # shape=[3, 1]
          lengths=[3, (3, 0, 2)],
          expected=ragged_factory_ops.constant_value(
              [[10, 10, 10], [], [30, 30]], dtype=np.int32)),
      dict(
          x=[[[1, 2, 3]], [[4, 5, 6]]],  # shape = [2, 1, 3]
          lengths=[2, (2, 3), 3],
          expected=ragged_factory_ops.constant_value(
              [[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]]],
              dtype=np.int32,
              ragged_rank=1)),
      dict(
          x=[[[1]], [[2]]],  # shape = [2, 1, 1]
          lengths=[2, (2, 3), (0, 2, 1, 2, 0)],
          expected=ragged_factory_ops.constant_value(
              [[[], [1, 1]], [[2], [2, 2], []]], dtype=np.int32,
              ragged_rank=2)),
      dict(
          x=10,
          lengths=[3, (3, 0, 2)],
          expected=ragged_factory_ops.constant_value([[10, 10, 10], [],
                                                      [10, 10]])),
      dict(
          x=ragged_factory_ops.constant_value([[[1], [2]], [[3]]],
                                              ragged_rank=1),
          lengths=[2, (2, 1), 2],
          expected=ragged_factory_ops.constant_value(
              [[[1, 1], [2, 2]], [[3, 3]]], ragged_rank=1)),
  ])
  def testRaggedBroadcastTo(self, x, lengths, expected):
    # [FUNCTION DOC]
    # Scenario: broadcasting dense or ragged values to a shape from lengths. Asserts the number
    # of row partitions and values equal expected, and the wrapper in the array ops module gives
    # the same result.
    shape = DynamicRaggedShape.from_lengths(lengths)
    result = dynamic_ragged_shape.broadcast_to(x, shape)
    self.assertEqual(
        getattr(result, 'num_row_partitions', 0),
        getattr(expected, 'num_row_partitions', 0))
    self.assertAllEqual(result, expected)

    # broadcast_to just calls dynamic_ragged_shape.broadcast_to, so
    # this should be sufficient.
    result2 = ragged_array_ops.broadcast_to(x, shape)
    self.assertAllEqual(result2, expected)

  @parameterized.parameters([
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]],
                                              dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged_factory_ops.constant_value([[11, 12, 13], [],
                                                      [34, 35]])),
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]],
                                              dtype=np.int32),
          y=10,
          expected=ragged_factory_ops.constant_value([[11, 12, 13], [],
                                                      [14, 15]])),
      dict(
          doc='x.shape=[1, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3]], dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged_factory_ops.constant_value(
              [[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.int32)),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, (D2)]; '
               'bcast.shape=[2, (D1), (D2)]'),
          x=ragged_factory_ops.constant_value([[[1], [2], [3]], [[4]]],
                                              ragged_rank=1),
          y=ragged_factory_ops.constant_value([[10, 20, 30]]),
          expected=ragged_factory_ops.constant_value([[[11, 21,
                                                        31], [12, 22, 32],
                                                       [13, 23, 33]],
                                                      [[14, 24, 34]]])),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, 1, 4]; '
               'bcast.shape=[2, (D1), 4]'),
          x=ragged_factory_ops.constant_value([[[10], [20]], [[30]]],
                                              ragged_rank=1),
          y=[[[1, 2, 3, 4]]],
          expected=ragged_factory_ops.constant_value(
              [[[11, 12, 13, 14], [21, 22, 23, 24]], [[31, 32, 33, 34]]],
              ragged_rank=1)),
      dict(
          doc=('x.shape=[2, (D1), 2, 1]; y.shape=[2, (D2)]; '
               'bcast.shape=[2, (D1), (2), (D2)'),
          x=ragged_factory_ops.constant_value(
              [[[[1], [2]], [[3], [4]]], [[[5], [6]]]], ragged_rank=1),
          y=ragged_factory_ops.constant_value([[10, 20], [30]]),
          expected=ragged_factory_ops.constant_value([[[[11, 21], [32]],
                                                       [[13, 23], [34]]],
                                                      [[[15, 25], [36]]]])),
  ])
  def testRaggedAddWithBroadcasting(self, x, y, expected, doc):
    # [FUNCTION DOC]
    # Scenario: adding ragged and dense tensors with broadcasting. The doc parameter is only a
    # description and is deleted. Converts inputs to 32-bit tensors, adds with the operator, and
    # asserts the same number of row partitions and the same values as expected.
    del doc
    expected_rrank = getattr(expected, 'num_row_partitions', 0)
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
    y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
    result = x + y
    result_rrank = getattr(result, 'num_row_partitions', 0)
    self.assertEqual(expected_rrank, result_rrank)
    if hasattr(expected, 'tolist'):
      expected = expected.tolist()
    self.assertAllEqual(result, expected)

  @parameterized.parameters([
      dict(lengths_a=[3, (1, 4, 2)], new_impl=True, op_max=10),  # Actual ops: 5
      dict(lengths_a=[3, (1, 4, 2)], new_impl=False, op_max=300),
  ])
  def testAddSelf(self, lengths_a, new_impl, op_max, num_row_partitions_a=None):
    # [FUNCTION DOC]
    # Scenario: graph-size check for adding a ragged tensor to itself. Graph mode only. Counts
    # graph nodes before and after using the new implementation or the old operator and asserts
    # the added nodes do not exceed op_max.
    if context.executing_eagerly():
      return
    shape_a0 = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a0)
    rt_b = rt_a
    g = rt_a.flat_values.graph if ragged_tensor.is_ragged(rt_a) else rt_a.graph
    nodes_at_a = len(g.as_graph_def().node)
    if new_impl:
      dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
          gen_math_ops.add_v2, rt_a, rt_b)
      nodes_at_b = len(g.as_graph_def().node)
      node_delta = nodes_at_b - nodes_at_a
      self.assertLessEqual(node_delta, op_max)
    else:
      if isinstance(rt_a, RaggedTensor):
        rt_a = rt_a.with_row_splits_dtype(dtypes.int32)
      rt_b = rt_a
      nodes_at_b = len(g.as_graph_def().node)
      rt_a + rt_b  # pylint: disable=pointless-statement
      nodes_at_d = len(g.as_graph_def().node)
      node_delta = nodes_at_d - nodes_at_b
      self.assertLessEqual(node_delta, op_max)

  def testAndSelfBool(self):
    # [FUNCTION DOC]
    # Scenario: logical-and of a boolean ragged tensor with itself via the elementwise op
    # implementation. Graph mode only. Asserts the result equals the input.
    if context.executing_eagerly():
      return
    values = constant_op.constant([True, False, True, True, True])
    rt_a = RaggedTensor.from_row_splits(values, [0, 3, 3, 5])
    result = dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
        gen_math_ops.logical_and, rt_a, rt_a)

    expected_values = values
    expected = RaggedTensor.from_row_splits(expected_values, [0, 3, 3, 5])

    self.assertAllEqual(result, expected)

  def testEquals(self):
    # [FUNCTION DOC]
    # Scenario: elementwise equality of a ragged tensor with a scalar. Graph mode only. Asserts
    # the boolean ragged result.
    if context.executing_eagerly():
      return

    rt_a = ragged_factory_ops.constant([[3, 1, 3], [3]])
    b = constant_op.constant(3)
    rt_expected = ragged_factory_ops.constant([[True, False, True], [True]])

    result = dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
        math_ops.equal, rt_a, b)
    self.assertAllEqual(result, rt_expected)

  def testEquals2(self):
    # [FUNCTION DOC]
    # Scenario: the equality operator on two ragged tensors with different values. Asserts the
    # comparison gives the boolean false.
    splits = constant_op.constant([0, 1])
    a = RaggedTensor.from_row_splits([[1, 2]], splits)
    b = RaggedTensor.from_row_splits([[3, 4, 5]], splits)
    self.assertIs(a == b, False)

  def testEquals3(self):
    # [FUNCTION DOC]
    # Scenario: same as the previous equality test, without a splits tensor. Asserts boolean
    # false.
    a = RaggedTensor.from_row_splits([[1, 2]], [0, 1])
    b = RaggedTensor.from_row_splits([[3, 4, 5]], [0, 1])
    self.assertIs(a == b, False)

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[], new_impl=True,
          max_num_ops=5),  # Actual ops: 1
      dict(
          lengths_a=[3, (1, 4, 2), 3, 2],
          lengths_b=[3, 2],
          new_impl=True,
          max_num_ops=5),  # Actual ops: 1
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[], new_impl=False,
          max_num_ops=5),  # Actual ops: 1
      dict(
          lengths_a=[3, (1, 4, 2), 3, 2],
          lengths_b=[3, 2],
          new_impl=False,
          max_num_ops=5),  # Actual ops: 1
  ])
  def testAdd(self,
              lengths_a,
              lengths_b,
              new_impl,
              max_num_ops,
              num_row_partitions_a=None,
              num_row_partitions_b=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the old implementation branch the number of added nodes is computed but
    # never asserted, so only the new implementation is checked.
    # Scenario: graph-size check when adding ragged tensors with broadcasting. Graph mode only.
    # Asserts the number of added nodes is within max_num_ops for the new implementation.
    if context.executing_eagerly():
      return

    shape_a0 = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    shape_b0 = DynamicRaggedShape.from_lengths(
        lengths_b, num_row_partitions=num_row_partitions_b)
    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a0)
    rt_b = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_b)), shape_b0)
    g = rt_a.flat_values.graph if ragged_tensor.is_ragged(rt_a) else rt_a.graph

    nodes_at_a = len(g.as_graph_def().node)
    if new_impl:
      dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
          gen_math_ops.add_v2,
          rt_a,
          rt_b)
      nodes_at_b = len(g.as_graph_def().node)
      num_nodes = nodes_at_b - nodes_at_a
      self.assertLessEqual(num_nodes, max_num_ops)
    else:
      if isinstance(rt_a, RaggedTensor):
        rt_a = rt_a.with_row_splits_dtype(dtypes.int32)
      if isinstance(rt_b, RaggedTensor):
        rt_b = rt_b.with_row_splits_dtype(dtypes.int32)
      nodes_at_b = len(g.as_graph_def().node)
      rt_a + rt_b  # pylint: disable=pointless-statement
      nodes_at_d = len(g.as_graph_def().node)
      num_nodes = nodes_at_d - nodes_at_b

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[],
          shape_e=[3, None], new_impl=False),
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[],
          shape_e=[3, None], new_impl=True),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          lengths_b=[5, 1, 3],
          shape_e=[5, None, 3], new_impl=False),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          lengths_b=[5, 1, 3],
          shape_e=[5, None, 3], new_impl=True),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          lengths_b=[3, 2, 1, 3],
          shape_e=[3, 2, None, 3], new_impl=False),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          lengths_b=[3, 2, 1, 3],
          shape_e=[3, 2, None, 3],
          new_impl=True),
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[3, 1],
          shape_e=[3, None], new_impl=False),
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[3, 1],
          shape_e=[3, None], new_impl=True),

  ])
  def testAddShape(self,
                   lengths_a,
                   lengths_b,
                   shape_e,
                   new_impl=False,
                   num_row_partitions_a=None,
                   num_row_partitions_b=None):
    # [FUNCTION DOC]
    # Scenario: static shape of the sum of two ragged tensors. Graph mode only. Adds with the new
    # implementation or the operator and asserts the static shape equals the expected one.
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    shape_b = DynamicRaggedShape.from_lengths(
        lengths_b, num_row_partitions=num_row_partitions_b)
    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_a)), shape_a)
    rt_b = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths_b)), shape_b)
    if new_impl:
      result = dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
          math_ops.add, rt_a, rt_b)
      shape_e = tensor_shape.TensorShape(shape_e)
      self.assertEqual(shape_e.as_list(), result.shape.as_list())
    else:
      if isinstance(rt_a, RaggedTensor):
        rt_a = rt_a.with_row_splits_dtype(dtypes.int32)
      if isinstance(rt_b, RaggedTensor):
        rt_b = rt_b.with_row_splits_dtype(dtypes.int32)
      result = rt_a + rt_b
      shape_e = tensor_shape.TensorShape(shape_e)
      self.assertEqual(shape_e.as_list(), result.shape.as_list())

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)], lengths_b=[],
          shape_e=[3, (1, 4, 2)]),
      dict(
          lengths_a=[5], lengths_b=[1],
          shape_e=[5]),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          lengths_b=[5, 1, 3],
          shape_e=[5, None, 3]),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          lengths_b=[3, 2, 1, 3],
          shape_e=[3, 2, None, 3]),
      dict(lengths_a=[3, (1, 4, 2)], lengths_b=[3, 1], shape_e=[3, None]),
      dict(lengths_a=[5, 1, 3], lengths_b=[2, 3], shape_e=[5, 2, 3]),
      dict(lengths_a=[5, 1, (3, 2, 4, 1, 3)], lengths_b=[2, 1],
           shape_e=[5, 2, None]),
      dict(lengths_a=[5, 4, 1, 3], lengths_b=[2, 1], shape_e=[5, 4, 2, 3]),
  ])
  def testBroadcastDynamicShapeStatic(self,
                                      lengths_a,
                                      lengths_b,
                                      shape_e,
                                      num_row_partitions_a=None,
                                      num_row_partitions_b=None):
    # [FUNCTION DOC]
    # Scenario: static lengths and tensor shape when broadcasting two shapes. Graph mode only.
    # Asserts static lengths and converted tensor shape equal the expected values.
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    shape_b = DynamicRaggedShape.from_lengths(
        lengths_b, num_row_partitions=num_row_partitions_b)

    result = dynamic_ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
    result_shape = result._to_tensor_shape()

    tensor_shape_e = [None if isinstance(x, tuple) else x for x in shape_e]
    self.assertEqual(shape_e, result.static_lengths())
    self.assertEqual(tensor_shape_e, result_shape.as_list())

  def testBroadcastDynamicShapePartiallyKnown(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting with a partly unknown inner shape. Graph mode only. Inside a compiled
    # function, a shape with a dynamic middle dimension is broadcast with a shape and the static
    # lengths are asserted as five, unknown, three.
    if context.executing_eagerly():
      return
    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.int64)])
    def fun(x):
      # [FUNCTION DOC]
      # Nested function with an int64 unknown-shape signature. Builds the shapes, broadcasts
      # them and asserts static lengths. Returns nothing.
      shape_a = DynamicRaggedShape([], array_ops_stack.stack([5, x, 3]))
      shape_b = DynamicRaggedShape.from_lengths([1, 3], dtype=dtypes.int64)
      result = dynamic_ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
      self.assertAllEqual([5, None, 3], result.static_lengths())
    fun(constant_op.constant(2, dtype=dtypes.int64))

  def testBroadcastDynamicShapePartiallyKnownNiceToHave(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: given the name, this documents desired behavior; it expects the dynamic
    # dimension to be resolved to two, which may not hold.
    # Scenario: like the previous test but expects a known size from broadcasting. Graph mode
    # only.
    if context.executing_eagerly():
      return
    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.int64)])
    def fun(x):
      # [FUNCTION DOC]
      # Nested function: broadcasts a shape with an unknown dimension against a known one and
      # asserts static lengths of five, two, three.
      shape_a = DynamicRaggedShape([], array_ops_stack.stack([5, x, 3]))
      shape_b = DynamicRaggedShape.from_lengths([2, 3], dtype=dtypes.int64)
      result = dynamic_ragged_shape.broadcast_dynamic_shape(shape_a, shape_b)
      self.assertAllEqual([5, 2, 3], result.static_lengths())
    fun(constant_op.constant(2, dtype=dtypes.int64))

  def testFromRowPartitionsStatic(self):
    # [FUNCTION DOC]
    # Scenario: static lengths of a shape from one row partition. Graph mode only. Asserts three
    # rows and lengths four, two, three.
    if context.executing_eagerly():
      return
    rp = RowPartition.from_row_lengths([4, 2, 3])
    result = DynamicRaggedShape.from_row_partitions([rp])
    self.assertEqual([3, (4, 2, 3)], result.static_lengths())

  @parameterized.parameters([
      dict(
          lengths_a=[3, (1, 4, 2)], dim=0,
          expected=3),
      dict(
          lengths_a=[5], dim=0,
          expected=5),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          dim=0,
          expected=5),
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          dim=2,
          expected=3),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          dim=1,
          expected=2),
      dict(lengths_a=[5, 1, 3], dim=0, expected=5),
  ])
  def testDimStatic(self, lengths_a, dim, expected):
    # [FUNCTION DOC]
    # Scenario: static value of a single dimension. Graph mode only. Asserts the constant value
    # of the indexed dimension equals the expected value.
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(lengths_a)
    result = tensor_util.constant_value(shape_a[dim])
    self.assertEqual(result, expected)

  @parameterized.parameters([
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3],
          shape_e=[5, (1, 4, 2, 1, 3), 3],
          new_num_row_partitions=2),  # Fails
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          shape_e=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          new_num_row_partitions=3),  # Fails
  ])
  def testNumRowPartitionShapeStatic(self,
                                     lengths_a,
                                     shape_e,
                                     new_num_row_partitions,
                                     num_row_partitions_a=None):
    # [FUNCTION DOC]
    # Scenario: static lengths after setting the number of row partitions. Graph mode only.
    # Asserts they equal the expected lengths.
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    result = shape_a._with_num_row_partitions(new_num_row_partitions)
    self.assertEqual(shape_e, result.static_lengths())

  @parameterized.parameters([
      dict(lengths_a=[5, (1, 4, 2, 1, 3), 3]),
      dict(lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3]),
  ])
  def testFromLengthsNRowsStatic(self, lengths_a):
    # [FUNCTION DOC]
    # Scenario: number of rows of each partition is statically known. Graph mode only.
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(lengths_a)
    for rp in shape_a.row_partitions:
      actual = tensor_util.constant_value(rp.nrows())
      self.assertIsNotNone(actual, 'Failed on ' + str(rp))

  @parameterized.parameters([
      dict(
          lengths_a=[5, (1, 4, 2, 1, 3), 3], inner_shape=[33],
          new_inner_rank=1),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          inner_shape=[36],
          new_inner_rank=1),
      dict(
          lengths_a=[3, 2, (1, 4, 2, 1, 3, 1), 3, 4],
          inner_shape=[36, 4],
          new_inner_rank=2),
  ])
  def testAltInnerShapeStatic(self,
                              lengths_a,
                              inner_shape,
                              new_inner_rank,
                              num_row_partitions_a=None):
    # [FUNCTION DOC]
    # Scenario: static value of the alternative inner shape. Graph mode only. Asserts the static
    # shape equals the expected inner shape.
    if context.executing_eagerly():
      return
    shape_a = DynamicRaggedShape.from_lengths(
        lengths_a, num_row_partitions=num_row_partitions_a)
    result = shape_a._alt_inner_shape(new_inner_rank)
    result_static = tensor_util.constant_value_as_shape(result)
    self.assertEqual(inner_shape, result_static.as_list())

  @parameterized.parameters([
      dict(
          lengths=[3, (1, 4, 2)],
          shape_e=[3, None]),
      dict(
          lengths=[3, (1, 4, 2)],
          shape_e=[3, None]),
      dict(
          lengths=[5, (1, 4, 2, 1, 3), 3],
          shape_e=[5, None, 3]),
      dict(
          lengths=[5, (1, 4, 2, 1, 3), 3],
          shape_e=[5, None, 3]),
      dict(
          lengths=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          shape_e=[3, 2, None, 3]),
      dict(
          lengths=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          shape_e=[3, 2, None, 3]),
  ])
  def testStaticShape(self,
                      lengths,
                      shape_e,
                      num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: static shape of a ragged reshape result. Graph mode only. Asserts the static shape
    # list equals the expected.
    # Testing the shape has enough information.
    # In particular, any uniform_row_length should be reproduced.
    if context.executing_eagerly():
      return
    shape = DynamicRaggedShape.from_lengths(
        lengths, num_row_partitions=num_row_partitions)
    rt_a = ragged_array_ops.ragged_reshape(
        _lowest_primes(_num_elements_of_lengths(lengths)), shape)
    shape_e = tensor_shape.TensorShape(shape_e)
    self.assertEqual(shape_e.as_list(), rt_a.shape.as_list())

  @parameterized.parameters([
      dict(
          lengths=[5, (1, 4, 2, 1, 3), 3],
          shape_e=[5, (1, 4, 2, 1, 3), 3]),
      dict(
          lengths=[3, 2, (1, 4, 2, 1, 3, 1), 3],
          shape_e=[3, 2, (1, 4, 2, 1, 3, 1), 3]),
  ])
  def testWithNumRowPartitionsStatic(self,
                                     lengths,
                                     shape_e,
                                     num_row_partitions=None):
    # [FUNCTION DOC]
    # Scenario: static lengths after changing to one fewer row partition than rank. Graph mode
    # only. Asserts they equal the expected.
    # Note that this test loses the later static values.
    if context.executing_eagerly():
      return
    shape = DynamicRaggedShape.from_lengths(
        lengths, num_row_partitions=num_row_partitions)
    shape_b = shape._with_num_row_partitions(shape.rank - 1)
    self.assertEqual(shape_e, shape_b.static_lengths())

  def testWithNumRowPartitionsStaticAlt(self):
    # [FUNCTION DOC]
    # Scenario: static lengths after changing to zero row partitions. Graph mode only. Asserts
    # the dense static lengths.
    # Note that this test loses the later static values.
    if context.executing_eagerly():
      return
    shape = DynamicRaggedShape.from_lengths(
        [5, 2, 3], num_row_partitions=2)
    shape_b = shape._with_num_row_partitions(0)
    self.assertEqual([5, 2, 3], shape_b.static_lengths())

  def testWithNumRowPartitionsDType(self):
    # [FUNCTION DOC]
    # Scenario: dtype is preserved. Builds a shape with 32-bit inner shape, changes the number of
    # row partitions and asserts the dtype stays 32-bit.
    # Note that this test loses the later static values.
    shape = DynamicRaggedShape([], constant_op.constant([5, 2, 3],
                                                        dtype=dtypes.int32))
    self.assertEqual(shape.dtype, dtypes.int32)

    result = shape._with_num_row_partitions(2)
    self.assertEqual(result.dtype, dtypes.int32)

  def test_merge_with(self):
    # [FUNCTION DOC]
    # Scenario: merging a shape with itself. Asserts the result equals the original.
    original = DynamicRaggedShape.from_lengths([2, (3, 5), 6])
    result = original._merge_with(original)
    self.assertShapeEq(result, original)

  def test_merge_with_spec(self):
    # [FUNCTION DOC]
    # Scenario: merging a shape with a compatible spec. Asserts the result equals the original.
    original = DynamicRaggedShape.from_lengths([2, (3, 5), 6],
                                               dtype=dtypes.int64)
    spec = DynamicRaggedShape.Spec(
        row_partitions=[
            RowPartitionSpec(nrows=2,
                             nvals=8,
                             dtype=dtypes.int64)
        ],
        static_inner_shape=tensor_shape.TensorShape([8, 6]),
        dtype=dtypes.int64)
    result = original._merge_with_spec(spec)
    self.assertShapeEq(result, original)

  def test_merge_with_spec_raises(self):
    # [FUNCTION DOC]
    # Scenario: merging a 64-bit shape with a 32-bit spec. Asserts a value error that the
    # partition and spec are not compatible.
    original = DynamicRaggedShape.from_lengths([2, (3, 5), 6],
                                               dtype=dtypes.int64)
    spec = DynamicRaggedShape.Spec(
        row_partitions=[
            RowPartitionSpec(nrows=2,
                             nvals=8,
                             dtype=dtypes.int32)
        ],
        static_inner_shape=tensor_shape.TensorShape([8, 6]),
        dtype=dtypes.int32)
    with self.assertRaisesRegex(
        ValueError,
        'RowPartition and RowPartitionSpec are not compatible'):
      original._merge_with_spec(spec)

  def test_merge_with_spec_uniform(self):
    # [FUNCTION DOC]
    # Scenario: merging a ragged shape that has uniform rows with a uniform spec. Asserts the
    # result equals a shape with one uniform row partition.
    original = DynamicRaggedShape.from_lengths(
        [2, (4, 4), 6], dtype=dtypes.int64)
    spec = DynamicRaggedShape.Spec._from_tensor_shape(
        tensor_shape.TensorShape([2, 4, 6]),
        num_row_partitions=0,
        dtype=dtypes.int64)
    result = original._merge_with_spec(spec)
    original = DynamicRaggedShape.from_lengths([2, 4, 6],
                                               num_row_partitions=1,
                                               dtype=dtypes.int64)
    self.assertShapeEq(result, original)

  @parameterized.parameters([
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]],
                                              dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged_factory_ops.constant_value([[11, 12, 13], [],
                                                      [34, 35]])),
      dict(
          doc='x.shape=[3, (D1)]; y.shape=[]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3], [], [4, 5]],
                                              dtype=np.int32),
          y=10,
          expected=ragged_factory_ops.constant_value([[11, 12, 13], [],
                                                      [14, 15]])),
      dict(
          doc='x.shape=[1, (D1)]; y.shape=[3, 1]; bcast.shape=[3, (D1)]',
          x=ragged_factory_ops.constant_value([[1, 2, 3]], dtype=np.int32),
          y=[[10], [20], [30]],
          expected=ragged_factory_ops.constant_value(
              [[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.int32)),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, (D2)]; '
               'bcast.shape=[2, (D1), (D2)]'),
          x=ragged_factory_ops.constant_value([[[1], [2], [3]], [[4]]],
                                              ragged_rank=1),
          y=ragged_factory_ops.constant_value([[10, 20, 30]]),
          expected=ragged_factory_ops.constant_value([[[11, 21,
                                                        31], [12, 22, 32],
                                                       [13, 23, 33]],
                                                      [[14, 24, 34]]])),
      dict(
          doc=('x.shape=[2, (D1), 1]; y.shape=[1, 1, 4]; '
               'bcast.shape=[2, (D1), 4]'),
          x=ragged_factory_ops.constant_value([[[10], [20]], [[30]]],
                                              ragged_rank=1),
          y=[[[1, 2, 3, 4]]],
          expected=ragged_factory_ops.constant_value(
              [[[11, 12, 13, 14], [21, 22, 23, 24]], [[31, 32, 33, 34]]],
              ragged_rank=1)),
      dict(
          doc=('x.shape=[2, (D1), 2, 1]; y.shape=[2, (D2)]; '
               'bcast.shape=[2, (D1), (2), (D2)'),
          x=ragged_factory_ops.constant_value(
              [[[[1], [2]], [[3], [4]]], [[[5], [6]]]], ragged_rank=1),
          y=ragged_factory_ops.constant_value([[10, 20], [30]]),
          expected=ragged_factory_ops.constant_value([[[[11, 21], [32]],
                                                       [[13, 23], [34]]],
                                                      [[[15, 25], [36]]]])),
  ])
  def testRaggedDispatchImplWithBroadcasting(self, x, y, expected, doc):
    # [FUNCTION DOC]
    # Scenario: same as the add-with-broadcasting test, but calls the elementwise op
    # implementation directly with the add operation. The doc parameter is deleted.
    del doc
    expected_rrank = getattr(expected, 'num_row_partitions', 0)
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
    y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
    result = dynamic_ragged_shape.ragged_binary_elementwise_op_impl(
        gen_math_ops.add_v2, x, y)
    result_rrank = getattr(result, 'num_row_partitions', 0)
    self.assertEqual(expected_rrank, result_rrank)
    if hasattr(expected, 'tolist'):
      expected = expected.tolist()
    self.assertAllEqual(result, expected)

  def testDimensions(self):
    # [FUNCTION DOC]
    # Scenario: dimension accessor on a shape from an inner shape. Asserts dimension zero is one.
    a = DynamicRaggedShape._from_inner_shape([1, 2, 3])
    self.assertAllEqual(1, a._dimension(0))

  def testGetItemIsInstanceTensor(self):
    # [FUNCTION DOC]
    # Scenario: indexing a dense shape returns a tensor instance.
    a = dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape([1, 2, 3])
    self.assertIsInstance(a[0], tensor.Tensor)

  @parameterized.parameters([
      dict(
          lengths=[2, 2],
          num_row_partitions=1,
          expected=[2, 2]),
      dict(lengths=[2, 2], num_row_partitions=0, expected=[2, 2]),
      dict(
          lengths=[2, (1, 2), 2], num_row_partitions=1, expected=[2, (1, 2), 2])
  ])
  def testStaticLengths(self,
                        lengths,
                        num_row_partitions,
                        expected,
                        expected_eager=None):
    # [FUNCTION DOC]
    # Scenario: static lengths for shapes with various row partition counts. Asserts they equal
    # expected (or the eager-mode expectation when given and in eager mode).
    a = DynamicRaggedShape.from_lengths(lengths)._with_num_row_partitions(
        num_row_partitions)
    actual = a.static_lengths()
    if context.executing_eagerly() and expected_eager is not None:
      self.assertAllEqual(expected_eager, actual)
    else:
      self.assertAllEqual(expected, actual)

  def testStaticLengthsUnknown(self):
    # [FUNCTION DOC]
    # Scenario: static lengths when the row lengths are unknown inside a compiled function.
    # Asserts the result is two unknowns.

    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.int32)])
    def foo(row_lengths):
      # [FUNCTION DOC]
      # Nested function: builds a shape from a row-length partition of unknown length and an inner
      # shape of six, asserts static lengths are unknown, unknown.
      a = DynamicRaggedShape([RowPartition.from_row_lengths(row_lengths)], [6])
      actual = a.static_lengths()
      self.assertAllEqual([None, None], actual)

    foo([3, 3])

  def testStaticLengthsRankUnknown(self):
    # [FUNCTION DOC]
    # Scenario: static lengths when the rank is unknown. Only a prefix is known.
    # Note that the rank of the shape is unknown, so we can only provide a
    # prefix of the lengths.
    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.int32)])
    def foo(inner_shape):
      # [FUNCTION DOC]
      # Nested function: builds a shape whose inner shape tensor has unknown rank and asserts the
      # static lengths are two, (three, three), ellipsis.
      a = DynamicRaggedShape([RowPartition.from_row_lengths([3, 3])],
                             inner_shape)
      actual = a.static_lengths()
      self.assertAllEqual([2, (3, 3), ...], actual)

    foo([6, 3])

  def testReprRankKnown(self):
    # [FUNCTION DOC]
    # Scenario: string form of a shape with known rank. Asserts the exact text.
    a = DynamicRaggedShape.from_lengths([2, (1, 2), 3])
    actual = str(a)
    self.assertEqual(
        '<DynamicRaggedShape lengths=[2, (1, 2), 3] num_row_partitions=1>',
        actual)

  def assertDimsEqual(self, x: tensor_shape.TensorShape,
                      y: tensor_shape.TensorShape):
    # [FUNCTION DOC]
    # HELPER: assertion for comparing tensor shapes.
    # If the first has unknown rank, asserts the second does too and returns. Otherwise asserts
    # second has known rank and equal dimension lists.
    if x.rank is None:
      self.assertIsNone(
          y.rank,
          'x has an unknown rank, but y does not: x={}, y={}'.format(x, y))
      return
    self.assertIsNotNone(
        y.rank,
        'y has an unknown rank, but x does not: x={}, y={}'.format(x, y))
    self.assertAllEqual(x.as_list(), y.as_list())

  def testToTensorShapeRankKnown(self):
    # [FUNCTION DOC]
    # Scenario: conversion to tensor shape with known rank. Asserts two, unknown, three.
    a = DynamicRaggedShape.from_lengths([2, (1, 2), 3])
    actual = a._to_tensor_shape()
    self.assertDimsEqual(tensor_shape.TensorShape([2, None, 3]), actual)

  def testReprRankUnknown(self):
    # [FUNCTION DOC]
    # Scenario: string form for unknown rank inside a compiled function. Asserts text with
    # ellipsis.

    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.int32)])
    def foo(inner_shape):
      # [FUNCTION DOC]
      # Nested function: builds a shape with unknown-rank inner shape, converts to a string and
      # asserts it.
      a = DynamicRaggedShape([RowPartition.from_row_lengths([3, 3])],
                             inner_shape)
      actual = str(a)
      self.assertEqual(
          '<DynamicRaggedShape lengths=[2, (3, 3), ...] num_row_partitions=1>',
          actual)

    foo([6, 3])

  def testToTensorShapeRankUnknown(self):
    # [FUNCTION DOC]
    # Scenario: conversion to tensor shape with unknown rank inside a compiled function. Asserts
    # unknown-rank shape.
    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.int32)])
    def foo(inner_shape):
      # [FUNCTION DOC]
      # Nested function: builds the shape, converts to a tensor shape and asserts unknown rank.
      a = DynamicRaggedShape([RowPartition.from_row_lengths([3, 3])],
                             inner_shape)
      actual = a._to_tensor_shape()
      self.assertDimsEqual(
          tensor_shape.TensorShape(None), actual)

    foo([6, 3])

  def testBroadcastDynamicShapeExtendedRankOne(self):
    # [FUNCTION DOC]
    # Scenario: extended broadcast of rank-one shapes, one and three. Asserts the shape is three
    # and the broadcasters yield expected values.
    a = DynamicRaggedShape._from_inner_shape([1])
    b = DynamicRaggedShape._from_inner_shape([3])
    (c, ac, bc) = dynamic_ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = DynamicRaggedShape._from_inner_shape([3])
    self.assertShapeEq(c, expected_c)
    ac_result = ac.broadcast(constant_op.constant([4]))
    self.assertAllEqual(ac_result, [4, 4, 4])
    bc_result = bc.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(bc_result, [4, 7, 1])

  def testBroadcastDynamicShapeExtendedRankOneRev(self):
    # [FUNCTION DOC]
    # Scenario: same as above with arguments reversed (three, one).
    a = DynamicRaggedShape._from_inner_shape([3])
    b = DynamicRaggedShape._from_inner_shape([1])
    (c, ac, bc) = dynamic_ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = DynamicRaggedShape._from_inner_shape([3])
    self.assertShapeEq(c, expected_c)
    bc_result = bc.broadcast(constant_op.constant([4]))
    self.assertAllEqual(bc_result, [4, 4, 4])
    ac_result = ac.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(ac_result, [4, 7, 1])

  def testBroadcastDynamicShapeExtendedRankOneIdentity(self):
    # [FUNCTION DOC]
    # Scenario: extended broadcast of identical shapes. Both broadcasters leave values unchanged.
    a = DynamicRaggedShape._from_inner_shape([3])
    b = DynamicRaggedShape._from_inner_shape([3])
    (c, ac, bc) = dynamic_ragged_shape.broadcast_dynamic_shape_extended(a, b)
    expected_c = DynamicRaggedShape._from_inner_shape([3])
    self.assertShapeEq(c, expected_c)
    bc_result = bc.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(bc_result, [4, 7, 1])
    ac_result = ac.broadcast(constant_op.constant([4, 7, 1]))
    self.assertAllEqual(ac_result, [4, 7, 1])

  def testFromGatherLayerIndexRaises(self):
    # [FUNCTION DOC]
    # Scenario: building a layer broadcaster from a float gather index. Asserts a value error.
    bad_gather_index = constant_op.constant([0.0, 0.5, 1.0])
    with self.assertRaisesRegex(ValueError, 'gather_index must be'):
      _LayerBroadcaster.from_gather_index(bad_gather_index)

  ### Tests mostly for code coverage ###########################################

  def testFindPreferredDtypeIntNone(self):
    # [FUNCTION DOC]
    # Scenario: coverage test for finding a dtype from an integer and none. Asserts none.
    actual = dynamic_ragged_shape._find_dtype(3, None)
    self.assertIsNone(actual)

  @parameterized.parameters([
      dict(
          source_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          target_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda: [int],
          dtype=None,
          error_type=TypeError,
          error_regex=r'Not a LayerBroadcaster'),
      dict(
          source_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          target_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda: _LayerBroadcaster.from_gather_index(
              [0, 1, 2]),
          dtype=None,
          error_type=TypeError,
          error_regex=r'layer'),
      dict(
          source_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          target_shape=lambda: None,
          layer_broadcasters=lambda:
          [_LayerBroadcaster.from_gather_index([0, 1, 2])],
          dtype=None,
          error_type=TypeError,
          error_regex='target_shape is not a DynamicRaggedShape'),
      dict(
          source_shape=lambda: None,
          target_shape=lambda: DynamicRaggedShape._from_inner_shape([3]),
          layer_broadcasters=lambda:
          [_LayerBroadcaster.from_gather_index([0, 1, 2])],
          dtype=None,
          error_type=TypeError,
          error_regex='source_shape is not a DynamicRaggedShape')
  ])
  def testBroadcasterInitRaises(self, source_shape, target_shape,
                                layer_broadcasters, dtype, error_type,
                                error_regex):
    # [FUNCTION DOC]
    # Scenario: invalid broadcaster constructor arguments. Parameters are callables producing
    # source shape, target shape and layer broadcasters. Asserts the expected type error.
    source_shape = source_shape()
    target_shape = target_shape()
    layer_broadcasters = layer_broadcasters()
    with self.assertRaisesRegex(error_type, error_regex):
      dynamic_ragged_shape._Broadcaster(
          source_shape, target_shape, layer_broadcasters, dtype=dtype)

  def testBroadcasterRepr(self):
    # [FUNCTION DOC]
    # Scenario: string form of a broadcaster. Asserts it contains the source shape text.
    source_shape = DynamicRaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    target_shape = DynamicRaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    layer_broadcasters = [
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2])),
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2]))
    ]
    bc = dynamic_ragged_shape._Broadcaster(source_shape, target_shape,
                                           layer_broadcasters)
    actual = str(bc)
    self.assertRegex(actual, '.src_shape..DynamicRaggedShape')

  def testBroadcasterWithDtype(self):
    # [FUNCTION DOC]
    # Scenario: changing the dtype of a broadcaster. Asserts the new broadcaster has the new dtype.
    source_shape = DynamicRaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    target_shape = DynamicRaggedShape(
        [RowPartition.from_row_splits(constant_op.constant([0, 1, 2]))],
        constant_op.constant([3]))
    layer_broadcasters = [
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2])),
        _LayerBroadcaster.from_gather_index(constant_op.constant([0, 1, 2]))
    ]
    bc = dynamic_ragged_shape._Broadcaster(
        source_shape, target_shape, layer_broadcasters, dtype=dtypes.int32)

    bc2 = bc.with_dtype(dtypes.int64)
    self.assertEqual(bc2.dtype, dtypes.int64)

  # TODO(martinz): This doesn't work for ragged_tensor_shape.
  # Uncomment when we switch over the implementation.
  #    dict(dtype=dtypes.int32)
  @parameterized.parameters([
      dict(dtype=dtypes.int64)
  ])
  def testBroadcasterWithDenseDType(self, dtype):
    # [FUNCTION DOC]
    # Scenario: broadcasting a dense tensor with a ragged tensor of given row-splits dtype, both
    # orders. Asserts the result row splits keep that dtype. Only the 64-bit case is run.
    a = constant_op.constant([[4]])
    b = RaggedTensor.from_row_splits([[2], [3], [4], [5]], [0, 3, 4])
    b = b.with_row_splits_dtype(dtype)
    c = a + b
    self.assertEqual(c.row_splits.dtype, dtype)
    d = b + a
    self.assertEqual(d.row_splits.dtype, dtype)

  @parameterized.parameters([
      dict(dtype_left=dtypes.int64,
           dtype_right=dtypes.int32),
      dict(dtype_left=dtypes.int32,
           dtype_right=dtypes.int64)])
  def testBroadcastWithDifferentDenseShapeDTypes(self, dtype_left,
                                                 dtype_right):
    # [FUNCTION DOC]
    # Scenario: broadcasting dense shapes with different integer types. Asserts the result dtype
    # is 64-bit.
    s_left = DynamicRaggedShape._from_inner_shape(
        constant_op.constant([4, 1], dtype_left))
    s_right = DynamicRaggedShape._from_inner_shape(
        constant_op.constant([1, 4], dtype_right))
    s_result = dynamic_ragged_shape.broadcast_dynamic_shape(s_left, s_right)
    self.assertEqual(s_result.dtype, dtypes.int64)

  def testBroadcastFlatValuesToDenseExpand(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting flat values of a uniform ragged tensor to a larger dense shape.
    # Asserts the expected nested values.
    source = RaggedTensor.from_uniform_row_length([0, 1, 2, 3], 2)
    target_shape = DynamicRaggedShape._from_inner_shape([1, 2, 2])
    broadcaster = dynamic_ragged_shape._get_broadcaster(
        DynamicRaggedShape.from_tensor(source), target_shape)
    flat_values = broadcaster.broadcast_flat_values(source)
    self.assertAllEqual(flat_values, [[[0, 1], [2, 3]]])

  # TODO(edloper): Confirm that this is the expected behavior.
  def testBroadcastFlatValuesToDenseExpandInnerDimensionsFalse(self):
    # [FUNCTION DOC]
    # Scenario: same as above with inner dimensions turned off. Asserts the flat result. A todo
    # in the code asks to confirm this behavior.
    source = RaggedTensor.from_uniform_row_length([0, 1, 2, 3], 2)
    target_shape = DynamicRaggedShape._from_inner_shape([1, 2, 2])
    broadcaster = dynamic_ragged_shape._get_broadcaster(
        DynamicRaggedShape.from_tensor(source), target_shape)
    flat_values = broadcaster.broadcast_flat_values(
        source, inner_dimensions=False)
    self.assertAllEqual(flat_values, [[0, 1], [2, 3]])

  def testGetLayerBroadcastersFromRPSRaisesTypeError(self):
    # [FUNCTION DOC]
    # Scenario: layer broadcasters from row partitions with a bad class argument. Asserts type
    # error.
    with self.assertRaisesRegex(TypeError, 'Not a _LayerBroadcaster'):
      dynamic_ragged_shape._get_layer_broadcasters_from_rps(int, [], [])

  def testGetBroadcasterRankDrop(self):
    # [FUNCTION DOC]
    # Scenario: broadcaster to a lower-rank target shape. Asserts a value error about being
    # unable to broadcast.
    with self.assertRaisesRegex(ValueError, 'Cannot broadcast'):
      a = DynamicRaggedShape._from_inner_shape([3, 4, 5])
      b = DynamicRaggedShape._from_inner_shape([4, 5])
      dynamic_ragged_shape._get_broadcaster(a, b)

  @parameterized.parameters([
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: None,
          error_type=TypeError,
          error_regex='b_1 should be a RowPartition'),
      dict(
          ac_0=lambda: None,
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='ac_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: None,
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='bc_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: None,
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='a_1 should be a RowPartition')
  ])
  def testBroadcastDynamicShapeNextLayerHalfRaggedRaises(
      self, ac_0, bc_0, a_1, b_1, error_type, error_regex):
    # [FUNCTION DOC]
    # Scenario: type validation of the half-ragged next-layer function. Each case passes none
    # for one argument, and the test asserts the matching type error.
    ac_0 = ac_0()
    bc_0 = bc_0()
    a_1 = a_1()
    b_1 = b_1()
    with self.assertRaisesRegex(error_type, error_regex):
      dynamic_ragged_shape._broadcast_dynamic_shape_next_layer_half_ragged(
          ac_0, bc_0, a_1, b_1)

  @parameterized.parameters([
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: None,
          error_type=TypeError,
          error_regex='b_1 should be a RowPartition'),
      dict(
          ac_0=lambda: None,
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='ac_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: None,
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='bc_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: None,
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='a_1 should be a RowPartition')
  ])
  def testBroadcastDynamicShapeNextLayerBothUniformRaises(
      self, ac_0, bc_0, a_1, b_1, error_type, error_regex):
    # [FUNCTION DOC]
    # Scenario: type validation of the both-uniform next-layer function. Each case passes none for
    # one argument. Asserts the matching type error.
    ac_0 = ac_0()
    bc_0 = bc_0()
    a_1 = a_1()
    b_1 = b_1()
    with self.assertRaisesRegex(error_type, error_regex):
      dynamic_ragged_shape._broadcast_dynamic_shape_next_layer_both_uniform(
          ac_0, bc_0, a_1, b_1)

  @parameterized.parameters([
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: None,
          error_type=TypeError,
          error_regex='b_1 should be a RowPartition'),
      dict(
          ac_0=lambda: None,
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='ac_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: None,
          a_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='bc_0 should be a _LayerBroadcaster'),
      dict(
          ac_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          bc_0=lambda: _LayerBroadcaster.from_gather_index([0, 1, 2]),
          a_1=lambda: None,
          b_1=lambda: RowPartition.from_row_splits([0, 1, 2]),
          error_type=TypeError,
          error_regex='a_1 should be a RowPartition')
  ])
  def testBroadcastDynamicShapeNextLayerRaises(self, ac_0, bc_0, a_1, b_1,
                                               error_type, error_regex):
    # [FUNCTION DOC]
    # Scenario: type validation of the general next-layer function. Each case passes none for one
    # argument. Asserts the matching type error.
    ac_0 = ac_0()
    bc_0 = bc_0()
    a_1 = a_1()
    b_1 = b_1()
    with self.assertRaisesRegex(error_type, error_regex):
      dynamic_ragged_shape._broadcast_dynamic_shape_next_layer(
          ac_0, bc_0, a_1, b_1)

  @parameterized.parameters([
      dict(
          left_dtype=dtypes.int64,
          right_dtype=dtypes.int64,
          expected_dtype=dtypes.int64),
      dict(
          left_dtype=dtypes.int32,
          right_dtype=dtypes.int32,
          expected_dtype=dtypes.int32)
  ])
  def testAddingRowSplits(self, left_dtype, right_dtype, expected_dtype):
    # [FUNCTION DOC]
    # Scenario: adding ragged tensors with the same row-splits dtype. Asserts the result keeps it.
    x = ragged_factory_ops.constant([[1, 2]]).with_row_splits_dtype(left_dtype)
    y = ragged_factory_ops.constant([[1, 2]]).with_row_splits_dtype(right_dtype)
    z = math_ops.add(x, y)
    self.assertEqual(z.row_splits.dtype, expected_dtype)

  @parameterized.parameters([
      dict(left_dtype=dtypes.int32, right_dtype=dtypes.int64),
      dict(left_dtype=dtypes.int64, right_dtype=dtypes.int32),
  ])
  def testAddingRowSplitsError(self, left_dtype, right_dtype):
    # [FUNCTION DOC]
    # Scenario: adding ragged tensors with different row-splits dtypes. Asserts a value error.
    x = ragged_factory_ops.constant([[1, 2]]).with_row_splits_dtype(left_dtype)
    y = ragged_factory_ops.constant([[1, 2]]).with_row_splits_dtype(right_dtype)
    with self.assertRaisesRegex(
        ValueError, 'Input RaggedTensors have mismatched row_splits dtypes'):
      math_ops.add(x, y)

  def testAddRowPartitionsInvalidV1(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says V1 but the test runs only in eager mode and returns otherwise.
    # Scenario: adding row partitions to a shape with flat values whose length does not match.
    # Asserts an invalid-argument or value error with message about the last partition.
    if not context.executing_eagerly():
      return

    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        'Last row partition does not match flat_values.'):
      rt = ragged_factory_ops.constant([[3], [4, 5], [6]])
      rt_shape = DynamicRaggedShape.from_tensor(rt)
      new_flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e'])
      rt_shape._add_row_partitions(new_flat_values, validate=True)

  # Example #1:
  # [2, (3, 1), 5], num_row_partitions = 1, outer_axis = 0, inner_axis = 1.
  # Result: [4, 5], num_row_partitions = 0.
  # Example #2:
  # [2, (2, 1), (7, 8, 9), 5], num_row_partitions = 2, outer_axis = 1,
  #     inner_axis = 2.
  # Result: [2, (15, 9), 5], num_row_partitions = 1.
  # Example #3:
  # [2, (2, 1), (7, 8, 9), 5], num_row_partitions = 2, outer_axis = 0,
  #     inner_axis = 1.
  # Result: [(7, 8, 9), 5], num_row_partitions = 1.
  # Here, we are merging the tail of the row_partitions,
  # but the inner_shape is unchanged.

  @parameterized.parameters([
      # NOOP
      dict(
          lengths=[2, (3, 1), 5],
          num_row_partitions=1,
          outer_axis=1,
          inner_axis=1,
          expected_lengths=[2, (3, 1), 5],
          expected_num_row_partitions=1),
      # Where num_row_partitions == 0
      dict(
          lengths=[2, 7, 5, 4],
          num_row_partitions=0,
          outer_axis=1,
          inner_axis=2,
          expected_lengths=[2, 35, 4],
          expected_num_row_partitions=0),
      # Where inner_axis <= self.num_row_partitions
      dict(
          lengths=[2, (3, 1), 5],
          num_row_partitions=1,
          outer_axis=0,
          inner_axis=1,
          expected_lengths=[4, 5],
          expected_num_row_partitions=0),
      dict(
          lengths=[2, (2, 1), (7, 8, 9), 5],
          num_row_partitions=2,
          outer_axis=1,
          inner_axis=2,
          expected_lengths=[2, (15, 9), 5],
          expected_num_row_partitions=1),
      # outer_axis > num_row_partitions (only inner_shape changed)
      dict(
          lengths=[2, (1, 2), 5, 3],
          num_row_partitions=1,
          outer_axis=2,
          inner_axis=3,
          expected_lengths=[2, (1, 2), 15],
          expected_num_row_partitions=1),
      # outer_axis <= num_row_partitions
      # inner_axis > num_row_partitions (everything changes)
      # (If outer_axis == 0, all row_partitions are truncated).
      dict(
          lengths=[2, (2, 1), (7, 8, 9), 2, 5],
          num_row_partitions=2,
          outer_axis=0,
          inner_axis=3,
          expected_lengths=[48, 5],
          expected_num_row_partitions=0),
      dict(
          lengths=[2, (2, 1), (7, 8, 9), 2, 5],
          num_row_partitions=2,
          outer_axis=1,
          inner_axis=3,
          expected_lengths=[2, (30, 18), 5],
          expected_num_row_partitions=1),
  ])
  def test_merge_dims(self, lengths, num_row_partitions, outer_axis, inner_axis,
                      expected_lengths, expected_num_row_partitions):
    # [FUNCTION DOC]
    # Scenario: merging dimensions of shapes. Builds a shape, merges the outer to inner axis
    # and asserts the result equals the expected shape from lengths and partition count.
    original = DynamicRaggedShape.from_lengths(
        lengths, num_row_partitions=num_row_partitions)
    actual = original._merge_dims(outer_axis, inner_axis)
    expected = DynamicRaggedShape.from_lengths(expected_lengths,
                                               expected_num_row_partitions)
    self.assertShapeEq(actual, expected)

  def test_merge_dims_special(self):
    # [FUNCTION DOC]
    # Scenario: merging the first two dimensions of a ragged shape. Asserts the new first size
    # is three.
    rt = ragged_factory_ops.constant([[[1, 2], [3]], [[4]]])
    original = DynamicRaggedShape.from_tensor(rt)
    actual = original._merge_dims(0, 1)
    self.assertAllEqual(actual[0], 3)

  def testGetItemRankNoneTruncate(self):
    # [FUNCTION DOC]
    # Scenario: truncating slice on a shape of unknown rank inside a compiled function. Asserts
    # the result equals the original.
    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.int32)])
    def foo(x):
      # [FUNCTION DOC]
      # Nested function: builds a shape from x, slices it and asserts equality with the original.
      rts = DynamicRaggedShape.from_tensor(x)
      actual = rts[:1]
      self.assertShapeEq(rts, actual)

    foo([1, 2, 3])

  def test_dataset_only_dense(self):
    # [FUNCTION DOC]
    # Scenario: creating a dataset from a dense shape. Only checks no error.
    ragged = DynamicRaggedShape.from_lengths([4, 5, 2, 3])
    dataset_ops.DatasetV2.from_tensors(ragged)

  def test_dataset_only_ragged(self):
    # [FUNCTION DOC]
    # Scenario: creating a dataset from a shape with a ragged part. Only checks no error.
    ragged = DynamicRaggedShape.from_lengths([4, (3, 0, 4, 5), 2, 3])
    dataset_ops.DatasetV2.from_tensors(ragged)

  def test_ragged_dataset(self):
    # [FUNCTION DOC]
    # Scenario: creating a dataset from a ragged tensor. Only checks no error.
    rt = RaggedTensor.from_row_splits(array_ops.zeros([5, 2, 3]), [0, 3, 5])
    dataset_ops.DatasetV2.from_tensors(rt)

  def test_ones_shape(self):
    # [FUNCTION DOC]
    # Scenario: the ones function with boolean dtype. Asserts static lengths of the result shape.
    ragged = DynamicRaggedShape.from_lengths([4, (3, 0, 4, 5)])
    ones = dynamic_ragged_shape.ones(ragged, dtype=bool)
    sh2 = DynamicRaggedShape.from_tensor(ones)
    self.assertAllEqual(sh2.static_lengths(), [4, (3, 0, 4, 5)])

  def test_dataset_only_simple_ragged(self):
    # [FUNCTION DOC]
    # Scenario: creating a dataset from a simple ragged shape. Only checks no error.
    ragged = DynamicRaggedShape.from_lengths([4, (3, 0, 4, 5)])
    dataset_ops.DatasetV2.from_tensors(ragged)

  # ValueError: _to_batched_tensor_list doesn't support ragged_rank=0 yet
  def test_unbatch_batch_dense(self):
    # [FUNCTION DOC]
    # Scenario: unbatching then batching a dense shape dataset. In eager mode asserts static
    # lengths of unbatched and rebatched elements.
    ragged = DynamicRaggedShape.from_lengths([4, 5, 2, 3])
    ds = dataset_ops.DatasetV2.from_tensors(ragged)
    dsu = ds.unbatch()
    if context.executing_eagerly():
      values = list(dsu)
      self.assertAllEqual(values[0].static_lengths(), [5, 2, 3])
      self.assertAllEqual(values[2].static_lengths(), [5, 2, 3])

    dsb = dsu.batch(2)
    if context.executing_eagerly():
      valuesb = list(dsb)
      self.assertAllEqual(valuesb[0].static_lengths(), [2, 5, 2, 3])
      self.assertAllEqual(valuesb[1].static_lengths(), [2, 5, 2, 3])

  def test_unbatch_batch_values_shape_0(self):
    # [FUNCTION DOC]
    # Scenario: unbatch then batch a one-dimensional shape. In eager mode asserts the batch
    # equals the original.
    batched = DynamicRaggedShape.from_lengths([2])
    batch_size = 2
    ds = dataset_ops.Dataset.from_tensors(batched)
    ds2 = ds.unbatch()
    if context.executing_eagerly():
      v = list(ds2.batch(batch_size))
      self.assertAllEqual(v[0], batched)

  def test_unbatch_batch_values_shape_1(self):
    # [FUNCTION DOC]
    # Scenario: unbatch then batch a two-dimensional shape. In eager mode asserts the batch equals
    # the shape with one row partition.
    batched = DynamicRaggedShape.from_lengths([2, 3])
    rebatched = DynamicRaggedShape.from_lengths([2, 3], num_row_partitions=1)

    batch_size = 2
    ds = dataset_ops.Dataset.from_tensors(batched)
    ds2 = ds.unbatch()
    if context.executing_eagerly():
      v = list(ds2.batch(batch_size))
      self.assertAllEqual(v[0], rebatched)

  def test_unbatch_dense_matrix(self):
    # [FUNCTION DOC]
    # Scenario: unbatching a matrix shape dataset. In eager mode asserts static lengths of rows.
    ragged = DynamicRaggedShape.from_lengths([2, 3])
    ds = dataset_ops.DatasetV2.from_tensors(ragged)
    dsu = ds.unbatch()
    if context.executing_eagerly():
      values = list(dsu)
      self.assertAllEqual(values[0].static_lengths(), [3])
      self.assertAllEqual(values[1].static_lengths(), [3])

  def test_unbatch_dense_vector(self):
    # [FUNCTION DOC]
    # Scenario: unbatching a vector shape dataset. In eager mode asserts empty static lengths.
    ragged = DynamicRaggedShape.from_lengths([3])
    ds = dataset_ops.DatasetV2.from_tensors(ragged)
    dsu = ds.unbatch()
    if context.executing_eagerly():
      values = list(dsu)
      self.assertAllEqual(values[0].static_lengths(), [])
      self.assertAllEqual(values[1].static_lengths(), [])

  def test_unbatch_ragged(self):
    # [FUNCTION DOC]
    # Scenario: unbatching a ragged shape dataset. In eager mode only creates an iterator.
    ragged = DynamicRaggedShape.from_lengths([4, (3, 0, 4, 5), 2, 3])
    ds = dataset_ops.DatasetV2.from_tensors(ragged)
    dsu = ds.unbatch()
    if context.executing_eagerly():
      dsu.__iter__()

  def test_unbatch_batch_ragged(self):
    # [FUNCTION DOC]
    # Scenario: unbatch then batch a ragged shape dataset. In eager mode asserts static lengths
    # of elements and batches.
    ragged = DynamicRaggedShape.from_lengths([4, (3, 0, 4, 5), 2, 3])
    ds = dataset_ops.DatasetV2.from_tensors(ragged)
    dsu = ds.unbatch()
    if context.executing_eagerly():
      values = list(dsu)
      self.assertAllEqual(values[0].static_lengths(), [3, 2, 3])
      self.assertAllEqual(values[2].static_lengths(), [4, 2, 3])

    dsb = dsu.batch(2)
    if context.executing_eagerly():
      valuesb = list(dsb)
      self.assertAllEqual(valuesb[0].static_lengths(), [2, (3, 0), 2, 3])
      self.assertAllEqual(valuesb[1].static_lengths(), [2, (4, 5), 2, 3])


class DynamicRaggedShapeErrorTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for error paths of DynamicRaggedShape, mostly with unknown ranks or invalid inputs.
  # A plain parameterized test case, run in whatever mode the default is.

  @parameterized.parameters([
      # Broadcast [1, 2, (1, 2)] to [1, 2, (2, 1)] (FAIL)
      dict(
          origin_lengths=[2, 1, (1, 2)],
          origin_values=[2, 3, 5],
          expected_lengths=[1, 2, (2, 1)]),
      # Broadcast [2, 1, (1, 1)] -> [2, 1, (5, 5)] (UNSUPPORTED)
      dict(
          origin_lengths=[2, 1, (1, 1)],
          origin_values=[2, 3],
          expected_lengths=[2, 1, (5, 5)]),
      # Broadcast [1, 2, (1, 2)] to [2, 2, (2, 1, 1, 2)] (FAIL)
      dict(
          origin_lengths=[1, 2, (1, 2)],
          origin_values=[2, 3, 5],
          expected_lengths=[2, 2, (2, 1, 1, 2)]),
      # Broadcast w.shape = [2,1,(1,3)] to w'.shape = [2,1,(3,3)] (UNSUPPORTED)
      dict(
          origin_lengths=[2, 1, (1, 3)],
          origin_values=[2, 3, 5, 7],  # [[[2]], [[3, 5, 7]]]
          expected_lengths=[2, 1, (3, 3)]),
  ])
  def testBroadcastRaggedError(self, origin_lengths, origin_values,
                               expected_lengths):
    # [FUNCTION DOC]
    # Scenario: unsupported or failing ragged broadcasts. Parameters give origin lengths, values and
    # target lengths. Opens a graph session, builds the origin tensor and target shape, broadcasts
    # and runs the result in the session; asserts that an invalid-argument error saying the
    # broadcast cannot be done is raised. Side effect: creates a session.
    # I pulled this out of the tensorflow test case, so that I could have
    # more control.
    # However this error is being generated, it confuses assertRaises,
    # but it exists.
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                r'Cannot broadcast'):
      sess = session.Session()
      with sess.as_default():
        origin = _to_ragged_tensor_from_lengths(origin_values, origin_lengths)
        expected_shape = DynamicRaggedShape.from_lengths(expected_lengths)

        rt = dynamic_ragged_shape.broadcast_to(origin, expected_shape)
        sess.run([rt])

  @parameterized.parameters([
      # nvals and nrows don't match (3 != 4) dynamically
      dict(
          row_partitions=lambda: [  # pylint: disable=g-long-lambda
              RowPartition.from_uniform_row_length(1, 3, nrows=3),
              RowPartition.from_uniform_row_length(1, 4, nrows=4)
          ],
          inner_shape=lambda: [4],
          validate=True,
          error_regex='RowPartitions in DynamicRaggedShape do not'),
      # nvals and inner_shape[0] don't match (3 != 4) dynamically
      dict(
          row_partitions=lambda: [  # pylint: disable=g-long-lambda
              RowPartition.from_uniform_row_length(1, 3, nrows=3),
          ],
          inner_shape=lambda: [4],
          validate=True,
          error_regex='Last row partition does not match inner_shape.'),
  ])
  def testConstructorRaisesDynamic(self,
                                   row_partitions,
                                   inner_shape,
                                   error_regex,
                                   validate=False,
                                   dtype=None):
    # [FUNCTION DOC]
    # Scenario: dynamic validation in the shape constructor. Parameters are callables creating
    # inconsistent row partitions and inner shapes, a validate flag and expected message. Inside a
    # session, constructs the shape with the validate flag and runs the inner shape tensor;
    # asserts that an invalid-argument or value error with the matching message is raised.
    with self.assertRaisesRegex((errors_impl.InvalidArgumentError, ValueError),
                                error_regex):
      sess = session.Session()
      with sess.as_default():
        row_partitions = row_partitions()
        inner_shape = inner_shape()
        rts = DynamicRaggedShape(
            row_partitions, inner_shape, dtype=dtype, validate=validate)
        sess.run([rts.inner_shape])

  def testRankNone(self):
    # [FUNCTION DOC]
    # Scenario: shape built from an inner shape of unknown rank. Calls a compiled function and
    # asserts the shape's rank is none.

    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.int32)])
    def foo(x):
      # [FUNCTION DOC]
      # Nested function with an int32 unknown-shape signature. Builds a shape from the inner
      # shape x and asserts its rank is unknown (none).
      rts = DynamicRaggedShape._from_inner_shape(x)
      self.assertIsNone(rts.rank)

    foo([3, 7, 5])

  def testNumSlicesInDimensionRankNone(self):
    # [FUNCTION DOC]
    # Scenario: number of slices in a dimension with unknown rank and negative axis. Asserts a
    # value error that the rank is undefined.
    with self.assertRaisesRegex(ValueError, 'rank is undefined'):

      @def_function.function(
          input_signature=[tensor.TensorSpec(None, dtypes.int32)])
      def foo(x):
        # [FUNCTION DOC]
        # Nested function: builds a unknown-rank shape from x and asks for slices in dimension
        # minus one, which raises when traced.
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts._num_slices_in_dimension(-1)

      foo([3, 7, 5])

  def testGetItemRankNone(self):
    # [FUNCTION DOC]
    # Scenario: indexing with negative index on an unknown-rank shape. Asserts a value error that
    # rank must be known.
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor.TensorSpec(None, dtypes.int32)])
      def foo(x):
        # [FUNCTION DOC]
        # Nested function: builds an unknown-rank shape from x and indexes it with minus one.
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts[-1]  # pylint: disable=pointless-statement

      foo([3, 7, 5])

  def testWithDenseRankRankNone(self):
    # [FUNCTION DOC]
    # Scenario: setting the inner rank of an unknown-rank shape. Asserts a value error that rank
    # must be known. (The name says dense rank.)
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor.TensorSpec(None, dtypes.int32)])
      def foo(x):
        # [FUNCTION DOC]
        # Nested function: builds an unknown-rank shape and sets the inner rank to one.
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts._with_inner_rank(1)

      foo([3, 7, 5])

  def testWithRaggedRankRankNone(self):
    # [FUNCTION DOC]
    # Scenario: setting the number of row partitions on an unknown-rank shape. Asserts a value
    # error that rank must be known. (The name says ragged rank.)
    with self.assertRaisesRegex(ValueError, 'Rank must be known to'):

      @def_function.function(
          input_signature=[tensor.TensorSpec(None, dtypes.int32)])
      def foo(x):
        # [FUNCTION DOC]
        # Nested function: builds an unknown-rank shape and sets the row partition count to one.
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts._with_num_row_partitions(1)

      foo([3, 7, 5])

  def testAsRowPartitionsRankNone(self):
    # [FUNCTION DOC]
    # Scenario: converting an unknown-rank shape to row partitions. Asserts a value error; the
    # message pattern is empty because the real text does not match.
    # Error is readable, but does not match strings correctly.
    with self.assertRaisesRegex(ValueError, ''):

      @def_function.function(
          input_signature=[tensor.TensorSpec(None, dtypes.int32)])
      def foo(x):
        # [FUNCTION DOC]
        # Nested function: builds an unknown-rank shape and converts it to row partitions.
        rts = DynamicRaggedShape._from_inner_shape(x)
        rts._as_row_partitions()

      foo([3, 7, 5])

  def testBroadcastDynamicShapeExtendedRankNone(self):
    # [FUNCTION DOC]
    # Scenario: extended broadcast with an unknown-rank shape. Asserts a value error about
    # unknown rank.
    with self.assertRaisesRegex(ValueError,
                                'Unable to broadcast: unknown rank'):

      @def_function.function(
          input_signature=[tensor.TensorSpec(None, dtypes.int32)])
      def foo(x):
        # [FUNCTION DOC]
        # Nested function: broadcasts an unknown-rank shape with a rank-three shape.
        a = DynamicRaggedShape._from_inner_shape(x)
        b = DynamicRaggedShape._from_inner_shape([1, 1, 1])
        dynamic_ragged_shape.broadcast_dynamic_shape_extended(a, b)

      foo([3, 7, 5])

  def testBroadcastDynamicShapeUnmatchedTypes6432(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting a 64-bit shape with a 32-bit shape. Asserts a value error that
    # dtypes do not match.
    shape_int64 = DynamicRaggedShape.from_lengths([3, (0, 2, 3)],
                                                  dtype=dtypes.int64)
    shape_int32 = DynamicRaggedShape.from_lengths([3, (0, 2, 3)],
                                                  dtype=dtypes.int32)
    with self.assertRaisesRegex(ValueError, "Dtypes don't match"):
      dynamic_ragged_shape.broadcast_dynamic_shape(shape_int64, shape_int32)

  def testBroadcastDynamicShapeUnmatchedTypes3264(self):
    # [FUNCTION DOC]
    # Scenario: same as above with the arguments reversed (32-bit first). Asserts the same error.
    shape_int64 = DynamicRaggedShape.from_lengths([3, (0, 2, 3)],
                                                  dtype=dtypes.int64)
    shape_int32 = DynamicRaggedShape.from_lengths([3, (0, 2, 3)],
                                                  dtype=dtypes.int32)
    with self.assertRaisesRegex(ValueError, "Dtypes don't match"):
      dynamic_ragged_shape.broadcast_dynamic_shape(shape_int32, shape_int64)

  def testGetIdentityBroadcasterRankNone(self):
    # [FUNCTION DOC]
    # Scenario: identity broadcaster of an unknown-rank shape. Asserts a value error that the
    # shape must have a rank.
    with self.assertRaisesRegex(ValueError, 'Shape must have a'):

      @def_function.function(
          input_signature=[tensor.TensorSpec(None, dtypes.int32)])
      def foo(x):
        # [FUNCTION DOC]
        # Nested function: builds an unknown-rank shape and asks for its identity broadcaster.
        rts = DynamicRaggedShape._from_inner_shape(x)
        dynamic_ragged_shape._get_identity_broadcaster(rts)

      foo([3, 7, 5])

  def testLayerBroadcasterRepr(self):
    # [FUNCTION DOC]
    # Scenario: string form of a layer broadcaster. Asserts text mentioning a tensor of shape
    # three with 32-bit integer type.
    index = constant_op.constant([0, 1, 2], name='testLayerBroadcasterRepr')
    lb = _LayerBroadcaster.from_gather_index(index)
    actual = str(lb)
    self.assertRegex(actual, '.*Tensor.*, shape=.3... dtype=int32.')

  def testGetBroadcasterRankNoneLeft(self):
    # [FUNCTION DOC]
    # Scenario: getting a broadcaster between unknown-rank shapes. Asserts a value error that
    # the rank of source and target must be known.
    with self.assertRaisesRegex(ValueError, 'Rank of source and target must'):

      @def_function.function(
          input_signature=[tensor.TensorSpec(None, dtypes.int32)])
      def foo(x):
        # [FUNCTION DOC]
        # Nested function: builds two unknown-rank shapes and requests a broadcaster.
        rts_a = DynamicRaggedShape._from_inner_shape(x)
        rts_b = DynamicRaggedShape._from_inner_shape(x)
        dynamic_ragged_shape._get_broadcaster(rts_a, rts_b)

      foo([3, 7, 5])

  def testFromTensorDType(self):
    # [FUNCTION DOC]
    # Scenario: dtype of a shape from a ragged tensor. Asserts row splits and shape dtypes are
    # 64-bit.
    x = ragged_factory_ops.constant([[1, 2]])
    self.assertEqual(x.row_splits.dtype, dtypes.int64)
    shape_x = DynamicRaggedShape.from_tensor(x)
    self.assertEqual(shape_x.dtype, dtypes.int64)

  def testAddingRowSplits(self):
    # [FUNCTION DOC]
    # Scenario: adding a ragged tensor to itself. Asserts the row splits remain 64-bit.
    x = ragged_factory_ops.constant([[1, 2]])
    self.assertEqual(x.row_splits.dtype, dtypes.int64)

    y = math_ops.add(x, x)
    self.assertEqual(y.row_splits.dtype, dtypes.int64)

  def testHashingWithMask(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test makes no assertion; it only runs the operations, so it only fails if
    # one raises. The computed mask is only used to select zeros.
    # Scenario: hashing string values of a ragged tensor with a mask. Builds the tensor, computes
    # a mask, hashes the strings, adds one and selects zeros where the mask is true.
    inp_data = ragged_factory_ops.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=dtypes.string)
    mask = math_ops.equal(inp_data, '')
    values = string_ops.string_to_hash_bucket_strong(
        inp_data, 3, name='hash', key=[0xDECAFCAFFE, 0xDECAFCAFFE])
    values = math_ops.add(values, array_ops.ones_like(values))
    local_zeros = array_ops.zeros_like(values)
    values = array_ops.where(mask, local_zeros, values)

  def testAddRowPartitionsInvalid(self):
    # [FUNCTION DOC]
    # Scenario: adding row partitions with flat values of the wrong size, with validation, in a
    # session. Asserts an invalid-argument or value error about the last partition.
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        'Last row partition does not match flat_values.'):
      sess = session.Session()
      with sess.as_default():
        rt = ragged_factory_ops.constant([[3], [4, 5], [6]])
        rt_shape = DynamicRaggedShape.from_tensor(rt)
        new_flat_values = constant_op.constant(['a', 'b', 'c'])
        rt2 = rt_shape._add_row_partitions(new_flat_values, validate=True)
        sess.run([rt2])


class DynamicRaggedShapeSpecTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for DynamicRaggedShape.Spec (the type specification of a shape): construction from
  # tensor shapes, dimension queries, dtype and row-partition changes, merging, batching, string
  # form and creation from a value. Defines assertion helpers for row partition specs, tensor
  # shapes, tensor specs and shape specs. A plain parameterized test case.

  def assertRowPartitionSpecEqual(self,
                                  a: RowPartitionSpec,
                                  b: RowPartitionSpec,
                                  msg='') -> None:
    # [FUNCTION DOC]
    # HELPER: custom assertion for row partition specs. Compares number of rows, number of
    # values, uniform row length and dtype of two specs, with an optional message.
    self.assertEqual(a.nrows, b.nrows, msg)
    self.assertEqual(a.nvals, b.nvals, msg)
    self.assertEqual(a.uniform_row_length, b.uniform_row_length, msg)
    self.assertEqual(a.dtype, b.dtype, msg)

  def assertTensorShapeEqual(self, a: tensor_shape.TensorShape,
                             b: tensor_shape.TensorShape) -> None:
    # [FUNCTION DOC]
    # HELPER: custom assertion that two tensor shapes are equal.
    self.assertEqual(a, b)

  def assertTensorSpecEqual(self,
                            a: tensor.TensorSpec,
                            b: tensor.TensorSpec) -> None:
    # [FUNCTION DOC]
    # HELPER: custom assertion that two tensor specs have equal shapes and dtypes.
    self.assertTensorShapeEqual(a.shape, b.shape)
    self.assertEqual(a.dtype, b.dtype)

  def assertDynamicRaggedShapeSpecEqual(self,
                                        a: DynamicRaggedShape.Spec,
                                        b: DynamicRaggedShape.Spec) -> None:
    # [FUNCTION DOC]
    # HELPER: custom assertion that two shape specs are equal. Compares static inner shapes,
    # inner shape tensor specs and each row partition spec in order.
    # KNOWN ISSUE: the loop variables shadow the arguments and the zip does not check that the
    # two have the same number of partitions, so a spec with extra partitions can go unnoticed.
    self.assertTensorShapeEqual(a._static_inner_shape, b._static_inner_shape)
    self.assertTensorSpecEqual(a._inner_shape, b._inner_shape)
    for i, (a, b) in enumerate(zip(a._row_partitions, b._row_partitions)):
      self.assertRowPartitionSpecEqual(a, b, 'Error in partition ' + str(i))

  @parameterized.parameters([
      # Unknown dimension
      dict(
          shape=tensor_shape.TensorShape(None),
          num_row_partitions=1,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int32),
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int32)
              ],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int32)),
      # Unknown dimension, dense
      dict(
          shape=tensor_shape.TensorShape(None),
          num_row_partitions=0,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int32)),
      # Scalar
      dict(
          shape=tensor_shape.TensorShape([]),
          num_row_partitions=0,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32)),
      # Vector
      dict(
          shape=tensor_shape.TensorShape([7]),
          num_row_partitions=0,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([7]),
              dtype=dtypes.int32)),
      # Generic
      dict(
          shape=tensor_shape.TensorShape([5, 3, None, 4, 2, 5]),
          num_row_partitions=3,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=5,
                      nvals=15,
                      uniform_row_length=3,
                      dtype=dtypes.int32),
                  RowPartitionSpec(
                      nrows=15,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int32),
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=4,
                      dtype=dtypes.int32)
              ],
              static_inner_shape=tensor_shape.TensorShape([None, 2, 5]),
              dtype=dtypes.int32)),
      # Generic, Dense
      dict(
          shape=tensor_shape.TensorShape([5, 3, None, 4, 2, 5]),
          num_row_partitions=0,
          dtype=dtypes.int32,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape(
                  [5, 3, None, 4, 2, 5]),
              dtype=dtypes.int32)),
  ])
  def test_from_tensor_shape(self, shape, num_row_partitions, dtype, expected):
    # [FUNCTION DOC]
    # Scenario: creating a spec from a tensor shape, row partition count and dtype. Asserts the
    # result equals the expected spec.
    spec = DynamicRaggedShape.Spec._from_tensor_shape(shape, num_row_partitions,
                                                      dtype)
    self.assertDynamicRaggedShapeSpecEqual(spec, expected)

  @parameterized.parameters([
      # Ridiculous DType.
      dict(
          shape=tensor_shape.TensorShape(None),
          num_row_partitions=1,
          dtype=dtypes.float32,
          error_type=ValueError,
          error_regex='dtype must be tf.int32 or tf.int64'),
      # num_row_partitions positive for scalar.
      dict(
          shape=tensor_shape.TensorShape([]),
          num_row_partitions=1,
          dtype=dtypes.int32,
          error_type=ValueError,
          error_regex='num_row_partitions should be zero ' +
          'if shape is a scalar or vector.'),
      dict(
          shape=tensor_shape.TensorShape([1, 2, 3]),
          num_row_partitions=3,
          dtype=dtypes.int32,
          error_type=ValueError,
          error_regex='num_row_partitions must be less than rank')
  ])
  def test_from_tensor_shape_raises(self, shape, num_row_partitions, dtype,
                                    error_type, error_regex):
    # [FUNCTION DOC]
    # Scenario: invalid arguments when creating a spec from a tensor shape (bad dtype, partitions
    # for a scalar, too many partitions). Asserts a value error with matching message.
    with self.assertRaisesRegex(error_type, error_regex):
      DynamicRaggedShape.Spec._from_tensor_shape(shape, num_row_partitions,
                                                 dtype)

  def test_from_tensor_shape_raises_dtype(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the arguments look misplaced: a tensor shape is passed where the number of row
    # partitions is expected, and the empty list as the shape; the expected dtype error only
    # happens to be raised.
    # Scenario: float dtype for a spec. Asserts a value error that dtype must be int32 or int64.
    with self.assertRaisesRegex(ValueError,
                                'dtype must be tf.int32 or tf.int64'):
      DynamicRaggedShape.Spec._from_tensor_shape(
          [], tensor_shape.TensorShape([1, 2, 3]), dtypes.float32)

  def test_from_row_partition_inner_shape_and_dtype_raises_dtype(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests an inner shape check, but the test only builds a spec with
    # partitions of mixed dtypes.
    # Scenario: spec whose row partitions have 32-bit and 64-bit dtypes. Asserts a value error
    # that the dtype does not match the expected one.
    with self.assertRaisesRegex(
        ValueError, r'dtype of .* is .*int64.*: expected .*int32.*'):
      DynamicRaggedShape.Spec(
          row_partitions=[
              RowPartitionSpec(
                  nrows=None,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int32),
              RowPartitionSpec(
                  nrows=None,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape(None),
          dtype=dtypes.int32)

  def test_ranks(self):
    # [FUNCTION DOC]
    # Scenario: rank properties of a spec. Asserts inner rank four, two row partitions, rank six.
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, None, 7, 4, 2, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)

    self.assertEqual(spec.inner_rank, 4)
    self.assertEqual(spec.num_row_partitions, 2)
    self.assertEqual(spec.rank, 6)

  def test_dimension_simple(self):
    # [FUNCTION DOC]
    # Scenario: dimension sizes of a spec. Asserts known sizes and unknown (none) for the
    # unknown one.
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, None, 7, 4, 2, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)

    self.assertEqual(spec._dimension(0), 5)
    self.assertIsNone(spec._dimension(1))
    self.assertEqual(spec._dimension(2), 7)
    self.assertEqual(spec._dimension(3), 4)
    self.assertEqual(spec._dimension(4), 2)
    self.assertEqual(spec._dimension(5), 5)

  @parameterized.parameters([
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          dimension=0),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          dimension=1),
  ])
  def test_dimension_none(self, spec, dimension):
    # [FUNCTION DOC]
    # Scenario: dimension of a spec of unknown rank. Asserts none.
    actual = spec._dimension(dimension)
    self.assertIsNone(actual)

  @parameterized.parameters([
      # Scalar.
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              [], 0, dtypes.int32),
          dimension=0,
          error_type=ValueError,
          error_regex='Index out of range: 0.'),
      # Scalar.
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              [], 0, dtypes.int32),
          dimension=1,
          error_type=ValueError,
          error_regex='Index out of range: 1.'),
  ])
  def test_dimension_raises(self, spec, dimension, error_type, error_regex):
    # [FUNCTION DOC]
    # Scenario: out-of-range dimension for a scalar spec. Asserts a value error.
    with self.assertRaisesRegex(error_type, error_regex):
      spec._dimension(dimension)

  def test_num_slices_in_dimension_ragged(self):
    # [FUNCTION DOC]
    # Scenario: number of slices per dimension for a ragged spec. Asserts products of known
    # sizes, none once an unknown size is hit, and none for negative axis -2.
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)

    self.assertEqual(spec._num_slices_in_dimension(0), 5)
    self.assertEqual(spec._num_slices_in_dimension(1), 5 * 3)
    self.assertEqual(spec._num_slices_in_dimension(2), 5 * 3 * 7)
    self.assertEqual(spec._num_slices_in_dimension(3), 5 * 3 * 7 * 4)
    self.assertIsNone(spec._num_slices_in_dimension(4))
    self.assertIsNone(spec._num_slices_in_dimension(5))
    self.assertIsNone(spec._num_slices_in_dimension(-2))

  def test_num_slices_in_dimension_ragged_alt(self):
    # [FUNCTION DOC]
    # Scenario: number of slices for an alternative ragged spec. Asserts known values and none.
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, 3, None, 2]),
        num_row_partitions=3,
        dtype=dtypes.int32)

    self.assertEqual(spec._num_slices_in_dimension(0), 5)
    self.assertEqual(spec._num_slices_in_dimension(1), 5 * 3)
    self.assertIsNone(spec._num_slices_in_dimension(2))
    self.assertIsNone(spec._num_slices_in_dimension(3))

  def test_num_slices_in_dimension_dense_known(self):
    # [FUNCTION DOC]
    # Scenario: number of slices for a fully known dense spec: five, fifteen, sixty.
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        [5, 3, 4], 0, dtypes.int32)

    self.assertEqual(spec._num_slices_in_dimension(0), 5)
    self.assertEqual(spec._num_slices_in_dimension(1), 15)
    self.assertEqual(spec._num_slices_in_dimension(2), 60)

  @parameterized.parameters([
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          dimension='CRAZY',
          error_type=TypeError,
          error_regex='axis must be an integer'),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          dimension=-1,
          error_type=ValueError,
          error_regex='axis=-1 may only be negative' +
          ' if rank is statically known.')
  ])
  def test_num_slices_in_dimension_raises(self, spec, dimension, error_type,
                                          error_regex):
    # [FUNCTION DOC]
    # Scenario: invalid axis for slice counts (non-integer, negative with unknown rank). Asserts
    # the matching error.
    with self.assertRaisesRegex(error_type, error_regex):
      spec._num_slices_in_dimension(dimension)

  def test_with_dtype(self):
    # [FUNCTION DOC]
    # Scenario: changing the spec dtype. Asserts the spec and its row partitions use 64-bit.
    spec = DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)
    actual = spec.with_dtype(dtypes.int64)
    self.assertEqual(actual.dtype, dtypes.int64)
    self.assertEqual(actual._row_partitions[0].dtype, dtypes.int64)
    self.assertEqual(actual._row_partitions[1].dtype, dtypes.int64)

  @parameterized.parameters([
      dict(
          original=DynamicRaggedShape.Spec._from_tensor_shape(
              shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
              num_row_partitions=2,
              dtype=dtypes.int32),
          num_row_partitions=3,
          expected=DynamicRaggedShape.Spec._from_tensor_shape(
              shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
              num_row_partitions=3,
              dtype=dtypes.int32)),
      dict(
          original=DynamicRaggedShape.Spec._from_tensor_shape(
              shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
              num_row_partitions=2,
              dtype=dtypes.int32),
          num_row_partitions=1,
          expected=DynamicRaggedShape.Spec._from_tensor_shape(
              shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
              num_row_partitions=1,
              dtype=dtypes.int32)),
  ])
  def test_with_num_row_partitions(self, original, num_row_partitions,
                                   expected):
    # [FUNCTION DOC]
    # Scenario: changing the row partition count of a spec. Asserts equality with the expected.
    actual = original._with_num_row_partitions(num_row_partitions)
    self.assertDynamicRaggedShapeSpecEqual(actual, expected)

  @parameterized.parameters([
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              None, 0, dtypes.int32),
          num_row_partitions=2,
          error_type=ValueError,
          error_regex='Changing num_row_partitions with unknown rank'),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              [1, 2, 3, 4], 0, dtypes.int32),
          num_row_partitions=4,
          error_type=ValueError,
          error_regex='Number of row partitions too large'),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
              [1, 2, 3, 4], 0, dtypes.int32),
          num_row_partitions=-3,
          error_type=ValueError,
          error_regex='Number of row partitions negative'),
  ])
  def test_with_num_row_partitions_raises(self, spec, num_row_partitions,
                                          error_type, error_regex):
    # [FUNCTION DOC]
    # Scenario: invalid row partition counts (unknown rank, too large, negative). Asserts value
    # errors with matching messages.
    with self.assertRaisesRegex(error_type, error_regex):
      spec._with_num_row_partitions(num_row_partitions)

  def test_truncate(self):
    # [FUNCTION DOC]
    # Scenario: truncating a spec to each rank from zero to six. Asserts the rank and the
    # dimensions match the original.
    spec = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
        shape=tensor_shape.TensorShape([5, 3, 7, 4, None, 5]),
        num_row_partitions=2,
        dtype=dtypes.int32)

    for new_rank in range(7):
      truncation = spec._truncate(new_rank)
      self.assertEqual(truncation.rank, new_rank)
      for i in range(new_rank):
        self.assertEqual(
            truncation._dimension(i), spec._dimension(i),
            'Mismatch on new_rank ' + str(new_rank) + ' on dimension ' + str(i))

  def test_truncate_unknown(self):
    # [FUNCTION DOC]
    # Scenario: truncating a spec whose inner shape has unknown rank to rank four. Asserts the
    # inner shape becomes two unknown dimensions.
    spec = DynamicRaggedShape.Spec(
        row_partitions=[
            RowPartitionSpec(
                nrows=3, nvals=7, uniform_row_length=None, dtype=dtypes.int32),
            RowPartitionSpec(
                nrows=7,
                nvals=None,
                uniform_row_length=None,
                dtype=dtypes.int32)
        ],
        static_inner_shape=tensor_shape.TensorShape(None),
        dtype=dtypes.int32)
    expected = DynamicRaggedShape.Spec(
        row_partitions=[
            RowPartitionSpec(
                nrows=3, nvals=7, uniform_row_length=None, dtype=dtypes.int32),
            RowPartitionSpec(
                nrows=7,
                nvals=None,
                uniform_row_length=None,
                dtype=dtypes.int32)
        ],
        static_inner_shape=tensor_shape.TensorShape([None, None]),
        dtype=dtypes.int32)
    actual = spec._truncate(4)
    self.assertDynamicRaggedShapeSpecEqual(actual, expected)

  @parameterized.parameters([
      # Standard scalar
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32),
          expected=0),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected=1),
      # Not knowing the shape of the inner shape is weird.
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int64),
          expected=None),
  ])
  def test_inner_rank(self, spec, expected):
    # [FUNCTION DOC]
    # Scenario: inner rank of specs (scalar gives zero, vector gives one, unknown gives none).
    actual = spec.inner_rank
    self.assertEqual(expected, actual)

  @parameterized.parameters([
      # Standard scalar
      dict(
          other_spec=tensor.TensorSpec([], dtypes.float32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int64)),
      dict(
          other_spec=ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=None,
                                   nvals=None,
                                   uniform_row_length=None,
                                   dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64)),
      dict(
          other_spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=None,
                                   nvals=None,
                                   uniform_row_length=None,
                                   dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=None,
                                   nvals=None,
                                   uniform_row_length=None,
                                   dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64)),
  ])
  def test_from_spec(self, other_spec, expected):
    # [FUNCTION DOC]
    # Scenario: creating a shape spec from another spec (tensor spec, ragged spec or shape spec).
    # Asserts the expected result.
    actual = DynamicRaggedShape.Spec._from_spec(other_spec)
    self.assertDynamicRaggedShapeSpecEqual(expected, actual)

  @parameterized.parameters([
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=None,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([None]),
          inner_shape=tensor.TensorSpec([1], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=None,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([None, 3]),
          inner_shape=tensor.TensorSpec([2], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=6,
                  nvals=None,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([None]),
          inner_shape=tensor.TensorSpec([1], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=6, nvals=60, uniform_row_length=10, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([60]),
          inner_shape=tensor.TensorSpec([1], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=6, nvals=60, uniform_row_length=10, dtype=dtypes.int64),
              RowPartitionSpec(
                  nrows=60,
                  nvals=120,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([120]),
          inner_shape=tensor.TensorSpec([1], dtypes.int64)),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=6, nvals=60, uniform_row_length=10, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape(None),
          inner_shape=tensor.TensorSpec([None], dtypes.int64))
  ])
  def test_constructor_idempotent(self, row_partitions, static_inner_shape,
                                  inner_shape):
    # [FUNCTION DOC]
    # Scenario: the constructor keeps already consistent information unchanged. Builds the spec
    # and asserts the inner shape and partitions equal the inputs.
    # The constructor detects if there is any additional information that
    # can be inferred from what is given.
    original = dynamic_ragged_shape.DynamicRaggedShape.Spec(
        row_partitions, static_inner_shape, inner_shape.dtype)
    self.assertTensorShapeEqual(original._static_inner_shape,
                                static_inner_shape)
    self.assertTensorSpecEqual(original._inner_shape, inner_shape)
    for i, (a, b) in enumerate(zip(original._row_partitions, row_partitions)):
      self.assertRowPartitionSpecEqual(a, b, 'Error in partition ' + str(i))

  @parameterized.parameters([
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=3,
                      nvals=None,
                      uniform_row_length=4,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected_row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          expected_static_inner_shape=tensor_shape.TensorShape([12]),
          expected_inner_shape=tensor.TensorSpec([1], dtypes.int64)),
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=3,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([30]),
              dtype=dtypes.int64),
          expected_row_partitions=[
              RowPartitionSpec(
                  nrows=10, nvals=30, uniform_row_length=3, dtype=dtypes.int64)
          ],
          expected_static_inner_shape=tensor_shape.TensorShape([30]),
          expected_inner_shape=tensor.TensorSpec([1], dtypes.int64)),
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=10,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected_row_partitions=[
              RowPartitionSpec(
                  nrows=6, nvals=60, uniform_row_length=10, dtype=dtypes.int64)
          ],
          expected_static_inner_shape=tensor_shape.TensorShape([60]),
          expected_inner_shape=tensor.TensorSpec([1], dtypes.int64)),
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64),
                  RowPartitionSpec(
                      nrows=60,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([120]),
              dtype=dtypes.int64),
          expected_row_partitions=[
              RowPartitionSpec(
                  nrows=6,
                  nvals=60,
                  uniform_row_length=None,
                  dtype=dtypes.int64),
              RowPartitionSpec(
                  nrows=60,
                  nvals=120,
                  uniform_row_length=None,
                  dtype=dtypes.int64)
          ],
          expected_static_inner_shape=tensor_shape.TensorShape([120]),
          expected_inner_shape=tensor.TensorSpec([1], dtypes.int64)),
  ])
  def test_constructor_improvements(self, original, expected_row_partitions,
                                    expected_static_inner_shape,
                                    expected_inner_shape):
    # [FUNCTION DOC]
    # Scenario: the constructor infers missing information (value counts, row counts, inner
    # shape). Asserts the inferred values on the given spec.
    # KNOWN ISSUE: the original argument is already built by the parameter list so the
    # constructor is not invoked inside the test.
    # Note that self_merge is only idempotent if no data is partially present.
    self.assertTensorShapeEqual(original._static_inner_shape,
                                expected_static_inner_shape)
    self.assertTensorSpecEqual(original._inner_shape, expected_inner_shape)
    for i, (a, b) in enumerate(
        zip(original._row_partitions, expected_row_partitions)):
      self.assertRowPartitionSpecEqual(a, b, 'Error in partition ' + str(i))

  @parameterized.parameters([
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([]),
          dtype=dtypes.int64,
          error_type=ValueError,
          msg='If row_partitions are provided, must have inner_rank > 0'),
      dict(
          row_partitions=RowPartitionSpec(
              nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64),
          static_inner_shape=tensor_shape.TensorShape([]),
          dtype=dtypes.int64,
          error_type=TypeError,
          msg='row_partitions should be an Iterable'),
      dict(
          row_partitions=[1, 2, 3],
          static_inner_shape=tensor_shape.TensorShape([12]),
          dtype=dtypes.int64,
          error_type=TypeError,
          msg='row_partitions should be an Iterable of RowPartitionSpecs'),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=3,
          dtype=dtypes.int64,
          error_type=ValueError,
          msg='Dimensions 12 and 3'),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([2]),
          dtype=456,
          error_type=TypeError,
          msg='Cannot convert'),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([12]),
          dtype=dtypes.int32,
          error_type=ValueError,
          msg='dtype of RowPartitionSpec'),
      dict(
          row_partitions=[
              RowPartitionSpec(
                  nrows=3, nvals=12, uniform_row_length=4, dtype=dtypes.int64)
          ],
          static_inner_shape=tensor_shape.TensorShape([11]),
          dtype=dtypes.int64,
          error_type=ValueError,
          msg='Dimensions 12 and 11 are not compatible'),
      dict(
          row_partitions=[
              RowPartitionSpec(nvals=3, dtype=dtypes.int64),
              RowPartitionSpec(uniform_row_length=4, dtype=dtypes.int64),
              RowPartitionSpec(nrows=17, dtype=dtypes.int64),
          ],
          static_inner_shape=tensor_shape.TensorShape([20]),
          dtype=dtypes.int64,
          error_type=ValueError,
          msg='Dimensions 17 and 12 are not compatible'),
  ])
  def test_constructor_raises(self, row_partitions, static_inner_shape,
                              dtype, error_type, msg):
    # [FUNCTION DOC]
    # Scenario: invalid spec constructor inputs. Asserts the matching error type and message.
    # Note that self_merge is only idempotent if no data is partially present.
    with self.assertRaisesRegex(error_type, msg):
      dynamic_ragged_shape.DynamicRaggedShape.Spec(
          row_partitions=row_partitions,
          static_inner_shape=static_inner_shape,
          dtype=dtype)

  @parameterized.parameters([
      # Unknown rank
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape(None)),
      # Scalar
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape([])),
      # Vector
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3]),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape([3])),
      # Dense
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3, 2, None]),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape([3, 2, None])),
      # Ragged
      dict(
          original=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=6,
                                   nvals=None,
                                   uniform_row_length=10,
                                   dtype=dtypes.int64),
                  RowPartitionSpec(nrows=60,
                                   nvals=None,
                                   uniform_row_length=None,
                                   dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([120]),
              dtype=dtypes.int64),
          expected=tensor_shape.TensorShape([6, 10, None])),

  ])
  def test_to_tensor_shape(self, original, expected):
    # [FUNCTION DOC]
    # Scenario: converting specs to tensor shapes. Asserts equality with expected.
    # Note that self_merge is only idempotent if no data is partially present.
    actual = original._to_tensor_shape()
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      dict(
          a=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32),
          b=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([]),
              dtype=dtypes.int32)),
      dict(
          a=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3, None]),
              dtype=dtypes.int32),
          b=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([None, 4]),
              dtype=dtypes.int32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3, 4]),
              dtype=dtypes.int32)),
      dict(
          a=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          b=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=10,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=60,
                      uniform_row_length=10,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([60]),
              dtype=dtypes.int64)),
      dict(
          a=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          b=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([None, 10]),
              dtype=dtypes.int64),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=60,
                      uniform_row_length=10,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([60]),
              dtype=dtypes.int64))
  ])
  def test_merge_with(self,
                      a: DynamicRaggedShape.Spec,
                      b: DynamicRaggedShape.Spec,
                      expected: DynamicRaggedShape.Spec):
    # [FUNCTION DOC]
    # Scenario: merging two specs in both orders. Asserts both equal the expected merged spec.
    actual = a._merge_with(b)
    actual_rev = b._merge_with(a)

    self.assertDynamicRaggedShapeSpecEqual(actual, expected)
    self.assertDynamicRaggedShapeSpecEqual(actual_rev, expected)

  @parameterized.parameters([
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=3,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([3]),
              dtype=dtypes.int64),
          batch_size=3,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=3,
                      nvals=18,
                      uniform_row_length=6,
                      dtype=dtypes.int64),
                  RowPartitionSpec(
                      nrows=18,
                      nvals=9,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([9]),
              dtype=dtypes.int64)),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=3,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([3]),
              dtype=dtypes.int64),
          batch_size=3,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=3,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64),
                  RowPartitionSpec(
                      nrows=None,
                      nvals=9,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([9]),
              dtype=dtypes.int64)),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64),
          batch_size=3,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=3,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64),
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int64)),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int64),
          batch_size=3,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=3,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64),
                  RowPartitionSpec(
                      nrows=None,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int64)),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=None,
                      nvals=6,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([6, 4]),
              dtype=dtypes.int64),
          batch_size=3,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=3,
                      nvals=None,
                      uniform_row_length=None,
                      dtype=dtypes.int64),
                  RowPartitionSpec(
                      nrows=None,
                      nvals=18,
                      uniform_row_length=None,
                      dtype=dtypes.int64)
              ],
              static_inner_shape=tensor_shape.TensorShape([18, 4]),
              dtype=dtypes.int64)),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int32),
          batch_size=3,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape(None),
              dtype=dtypes.int32)),
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([8, 9]),
              dtype=dtypes.int32),
          batch_size=7,
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([7, 8, 9]),
              dtype=dtypes.int32)),
  ])
  def test_batch(self,
                 spec: DynamicRaggedShape.Spec,
                 batch_size: int,
                 expected: DynamicRaggedShape.Spec):
    # [FUNCTION DOC]
    # Scenario: the batch encoder adds a leading batch dimension to a spec. Asserts equality
    # with the expected spec.
    encoder = dynamic_ragged_shape._DynamicRaggedShapeBatchEncoder()
    actual = encoder.batch(spec, batch_size)
    self.assertDynamicRaggedShapeSpecEqual(actual, expected)

  @parameterized.parameters([
      dict(
          spec=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(
                      nrows=6,
                      nvals=3,
                      uniform_row_length=None,
                      dtype=dtypes.int32)],
              static_inner_shape=tensor_shape.TensorShape([3]),
              dtype=dtypes.int32),
          expected=dynamic_ragged_shape.DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int32))
  ])
  def test_unbatch(self, spec: DynamicRaggedShape.Spec,
                   expected: DynamicRaggedShape.Spec):
    # [FUNCTION DOC]
    # Scenario: the batch encoder removes the leading dimension from a spec. Asserts equality.
    encoder = dynamic_ragged_shape._DynamicRaggedShapeBatchEncoder()
    actual = encoder.unbatch(spec)
    self.assertDynamicRaggedShapeSpecEqual(actual, expected)

  def test_repr(self):
    # [FUNCTION DOC]
    # Scenario: official string form of a spec. Asserts the exact text.
    original = dynamic_ragged_shape.DynamicRaggedShape.Spec(
        row_partitions=[
            RowPartitionSpec(
                nrows=6,
                nvals=None,
                uniform_row_length=None,
                dtype=dtypes.int64)
        ],
        static_inner_shape=tensor_shape.TensorShape([None]),
        dtype=dtypes.int64)
    representation = repr(original)
    static_inner_shape = tensor_shape.TensorShape([None])
    expected = ('DynamicRaggedShape.Spec(' +
                'row_partitions=(RowPartitionSpec(' +
                'nrows=6, nvals=None, uniform_row_length=None, ' +
                'dtype=tf.int64),), ' +
                f'static_inner_shape={static_inner_shape!r}, ' +
                'dtype=tf.int64)')
    self.assertEqual(representation, expected)

  @parameterized.parameters([
      dict(
          lengths=[3, 4, 5],
          expected=DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=tensor_shape.TensorShape([3, 4, 5]),
              dtype=dtypes.int64)),
      dict(
          lengths=[2, (4, 1), 5],
          expected=DynamicRaggedShape.Spec(
              row_partitions=[RowPartitionSpec(nrows=2, nvals=5)],
              static_inner_shape=tensor_shape.TensorShape([5, 5]),
              dtype=dtypes.int64)),
      dict(
          lengths=[2, (4, 1), 5],
          dtype=dtypes.int32,
          expected=DynamicRaggedShape.Spec(
              row_partitions=[
                  RowPartitionSpec(nrows=2, nvals=5, dtype=dtypes.int32)],
              static_inner_shape=tensor_shape.TensorShape([5, 5]),
              dtype=dtypes.int32)),
  ])
  def test_from_value(self, lengths, expected, dtype=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: asserts equality with a tensor-shape comparison helper even though both are
    # specs; works since it only tests equality.
    # Scenario: creating a spec from a value shape built from lengths and optional dtype.
    original = DynamicRaggedShape.from_lengths(lengths)
    if dtype is not None:
      original = original.with_dtype(dtype)
    actual = dynamic_ragged_shape.DynamicRaggedShape.Spec.from_value(original)
    self.assertTensorShapeEqual(actual, expected)

if __name__ == '__main__':
  googletest.main()
