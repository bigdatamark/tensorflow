# [MODULE DOC]
# KNOWN ISSUE: the original module docstring names an obsolete location (third_party tensorflow
# ragged_tensor); the code under test now lives in tensorflow/python/ops/ragged/ragged_tensor.py.
# Unit tests for the RaggedTensor class, the RaggedTensorValue container and the RaggedTensorSpec type
# specification (python/ops/ragged/ragged_tensor.py and ragged_tensor_value.py).
# Covered areas: the private constructor, every public factory (from value row ids, row splits, row
# starts, row limits, row lengths, uniform row length, and their nested variants), component
# accessors, static shape inference, string representations, with_values and with_flat_values,
# Session run / feed / partial-run support in graph mode, tensor consumers, validation errors raised by
# the factories, conversion to and from the variant encoding (plain and batched), tf.data
# unbatching, gradients through ragged ops, variant conversion and map_fn, conversion to numpy,
# setting shapes, and the RaggedTensorSpec serialization / component / batching behaviour.
# Import-time side effects: importing the gradient modules (array, math, tensor array) only to register
# gradient functions. Running the file as a script starts the googletest main runner.
# This is a test module, not a library; it exposes one tiny helper plus two test case classes.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for third_party.tensorflow.python.ops.ragged_tensor."""

import functools
from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.ragged.row_partition import RowPartition

from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


def int32array(values):
  # [FUNCTION DOC]
  # HELPER: small shared test utility that builds expected numpy arrays for the numpy conversion test.
  # Converts a list of numbers into a numpy array whose element type is 32-bit integer.
  # Input is any array-like sequence of integers; returns the new numpy array; no side effects.
  return np.array(values, dtype=np.int32)


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering RaggedTensor: construction, factories, accessors, shape, printing, variant
  # encoding, gradients, numpy conversion and set-shape. Every test runs both in graph mode and in
  # eager mode (decorator that runs all tests in both modes); many tests branch on the current mode
  # because static shapes and some features differ. Also derives from the parameterized test case to
  # allow data-driven tests. Sets long failure messages (class attribute) so assertion messages are
  # appended to the default message. Has no per-test fixtures; each test builds its own tensors.
  longMessage = True  # Property in unittest.Testcase. pylint: disable=invalid-name

  # =============================================================================
  # RaggedTensor class docstring examples
  # =============================================================================

  def testClassDocStringExamples(self):
    # [FUNCTION DOC]
    # Re-runs the examples from the RaggedTensor class documentation to make sure they stay correct.
    # Builds the same 5-row ragged tensor (rows of 4, 0, 3, 1 and 0 values) through row splits, row
    # lengths, value row ids (with 5 rows), row starts and row limits, and asserts all five equal the
    # expected nested list. Then builds a two-level ragged tensor by nesting (checking ragged rank 2),
    # the same via the nested row splits factory, and a tensor with a uniform inner dimension (checking
    # the static shape is 2 by unknown by 3). Local variables are deleted between sections.
    # From section: "Component Tensors"
    rt = RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(rt, [[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    del rt

    # From section: "Alternative Row-Partitioning Schemes"
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    rt1 = RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
    rt2 = RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
    rt3 = RaggedTensor.from_value_rowids(
        values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
    rt4 = RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
    rt5 = RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
    for rt in (rt1, rt2, rt3, rt4, rt5):
      self.assertAllEqual(rt, [[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    del rt1, rt2, rt3, rt4, rt5

    # From section: "Multiple Ragged Dimensions"
    inner_rt = RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    outer_rt = RaggedTensor.from_row_splits(
        values=inner_rt, row_splits=[0, 3, 3, 5])
    self.assertEqual(outer_rt.ragged_rank, 2)
    self.assertAllEqual(outer_rt,
                        [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del inner_rt, outer_rt

    # From section: "Multiple Ragged Dimensions"
    rt = RaggedTensor.from_nested_row_splits(
        flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
        nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8]))
    self.assertAllEqual(rt, [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del rt

    # From section: "Uniform Inner Dimensions"
    rt = RaggedTensor.from_row_splits(
        values=array_ops.ones([5, 3]), row_splits=[0, 2, 5])
    self.assertAllEqual(
        rt, [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    self.assertEqual(rt.shape.as_list(), [2, None, 3])
    del rt

  # =============================================================================
  # RaggedTensorValue Constructor
  # =============================================================================

  def testRaggedTensorValueConstruction(self):
    # [FUNCTION DOC]
    # Tests the numpy-backed RaggedTensorValue container. Builds one with ragged rank 1 from string values
    # and 64-bit row splits and checks row splits dtype, the shape (5 rows, unknown ragged dimension),
    # the number of nested row splits arrays, and that row splits, values, nested row splits and flat
    # values return the inputs. Repeats for ragged rank 2 (values being another RaggedTensorValue),
    # checking shape, both levels of splits and the flat values.
    values = np.array(b'a b c d e f g'.split())
    splits = np.array([0, 2, 5, 6, 6, 7], dtype=np.int64)
    splits2 = np.array([0, 3, 5], dtype=np.int64)

    # Test construction of a RaggedTensorValue with ragged_rank=1.
    rt_value = ragged_tensor_value.RaggedTensorValue(values, splits)
    self.assertEqual(rt_value.row_splits.dtype, np.int64)
    self.assertEqual(rt_value.shape, (5, None))
    self.assertLen(rt_value.nested_row_splits, 1)
    self.assertAllEqual(splits, rt_value.row_splits)
    self.assertAllEqual(values, rt_value.values)
    self.assertAllEqual(splits, rt_value.nested_row_splits[0])
    self.assertAllEqual(values, rt_value.flat_values)

    # Test construction of a RaggedTensorValue with ragged_rank=2.
    rt_value = ragged_tensor_value.RaggedTensorValue(
        values=ragged_tensor_value.RaggedTensorValue(values, splits),
        row_splits=splits2)
    self.assertEqual(rt_value.row_splits.dtype, np.int64)
    self.assertEqual(rt_value.shape, (2, None, None))
    self.assertLen(rt_value.nested_row_splits, 2)
    self.assertAllEqual(splits2, rt_value.row_splits)
    self.assertAllEqual(splits, rt_value.values.row_splits)
    self.assertAllEqual(splits2, rt_value.nested_row_splits[0])
    self.assertAllEqual(splits, rt_value.nested_row_splits[1])
    self.assertAllEqual(values, rt_value.values.values)
    self.assertAllEqual(values, rt_value.flat_values)

  # =============================================================================
  # RaggedTensor Constructor (private)
  # =============================================================================

  def testRaggedTensorConstruction(self):
    # [FUNCTION DOC]
    # Checks the private RaggedTensor constructor works when the internal flag is set: builds a row
    # partition from row splits and a string values tensor, constructs the ragged tensor directly, and
    # asserts its contents equal the expected nested list of byte strings.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rp = RowPartition.from_row_splits(row_splits)
    rt = RaggedTensor(values=values, row_partition=rp, internal=True)

    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testRaggedTensorConstructionErrors(self):
    # [FUNCTION DOC]
    # Checks argument validation of the private constructor. Expects a value error saying the constructor
    # is private when the internal flag is missing; a type error when values is neither a Tensor nor a
    # RaggedTensor (a plain range); and a type error when the row partition argument is a plain list
    # instead of a RowPartition.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rp = RowPartition.from_row_splits(row_splits)

    with self.assertRaisesRegex(ValueError,
                                'RaggedTensor constructor is private'):
      RaggedTensor(values=values, row_partition=rp)

    with self.assertRaisesRegex(
        TypeError, r'type\(values\) must be one of: Tensor, RaggedTensor'):
      RaggedTensor(values=range(7), row_partition=rp, internal=True)

    with self.assertRaisesRegex(
        TypeError, 'Argument `row_partition` must be a RowPartition'):
      RaggedTensor(
          values=values, row_partition=[0, 2, 2, 5, 6, 7], internal=True)

  # =============================================================================
  # RaggedTensor Factory Ops
  # =============================================================================

  def testFromValueRowIdsWithDerivedNRows(self):
    # [FUNCTION DOC]
    # Factory test: builds a ragged tensor from values and value row ids without giving the number of
    # rows (static, validate off). Asserts dtype is string, static shape is 5 by unknown, ragged rank is 1,
    # that values and value row ids are returned as the identical objects (cached), the number of rows is
    # 5, and the full contents match the expected nested byte strings.
    # nrows is known at graph creation time.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithDerivedNRowsDynamic(self):
    # [FUNCTION DOC]
    # Same as the derived-rows test but the value row ids are wrapped in a placeholder with unknown
    # shape. In eager mode the shape is still 5 by unknown, in graph mode it is fully unknown (unknown
    # by unknown). Asserts ragged rank 1, identity of the cached values and row ids, evaluated row count
    # 5 and the expected contents.
    # nrows is not known at graph creation time.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    value_rowids = array_ops.placeholder_with_default(value_rowids, shape=None)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    if context.executing_eagerly():
      self.assertEqual(rt.shape.as_list(), [5, None])
    else:
      self.assertEqual(rt.shape.as_list(), [None, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithExplicitNRows(self):
    # [FUNCTION DOC]
    # Factory test with an explicit row count of 7 greater than the derived 5. Asserts string dtype,
    # static shape 7 by unknown, ragged rank 1, that values, value row ids and the row count tensor are the
    # identical cached objects, and that the contents have two extra empty rows at the end.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(7, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(
        values, value_rowids, nrows, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [7, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertAllEqual(
        rt, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g'], [], []])

  def testFromValueRowIdsWithExplicitNRowsEqualToDefault(self):
    # [FUNCTION DOC]
    # Factory test where the explicitly given row count (5) equals the count that would have been
    # derived. Asserts string dtype, shape 5 by unknown, ragged rank 1, identity of cached values, row
    # ids and row count, equality of evaluated row ids and row count, and the expected contents.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(
        values, value_rowids, nrows, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, nrows)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithEmptyValues(self):
    # [FUNCTION DOC]
    # Factory test with empty values and empty value row ids. Asserts the result defaults to float32,
    # has static shape 0 by unknown, ragged rank 1, empty values and row ids (shape of size 0), zero rows
    # and an empty list as contents.
    rt = RaggedTensor.from_value_rowids([], [])
    rt_nrows = rt.nrows()
    self.assertEqual(rt.dtype, dtypes.float32)
    self.assertEqual(rt.shape.as_list(), [0, None])
    self.assertEqual(rt.ragged_rank, 1)
    self.assertEqual(rt.values.shape.as_list(), [0])
    self.assertEqual(rt.value_rowids().shape.as_list(), [0])
    self.assertAllEqual(rt_nrows, 0)
    self.assertAllEqual(rt, [])

  def testFromRowSplits(self):
    # [FUNCTION DOC]
    # Factory test for building from row splits (validate off). Asserts dtype string, shape 5 by unknown,
    # ragged rank 1, that values and row splits are returned as the identical objects, the row count 5 and
    # the expected contents.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    rt = RaggedTensor.from_row_splits(values, row_splits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_row_splits, row_splits)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowSplitsWithDifferentSplitTypes(self):
    # [FUNCTION DOC]
    # Checks the row splits dtype that results from different input types: a plain list and a 64-bit
    # numpy array and a 64-bit tensor give 64-bit splits; a 32-bit numpy array and a 32-bit tensor keep
    # 32-bit splits. Only the dtype of the row splits is asserted.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    splits1 = [0, 2, 2, 5, 6, 7]
    splits2 = np.array([0, 2, 2, 5, 6, 7], np.int64)
    splits3 = np.array([0, 2, 2, 5, 6, 7], np.int32)
    splits4 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    splits5 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int32)
    rt1 = RaggedTensor.from_row_splits(values, splits1)
    rt2 = RaggedTensor.from_row_splits(values, splits2)
    rt3 = RaggedTensor.from_row_splits(values, splits3)
    rt4 = RaggedTensor.from_row_splits(values, splits4)
    rt5 = RaggedTensor.from_row_splits(values, splits5)
    self.assertEqual(rt1.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt2.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt3.row_splits.dtype, dtypes.int32)
    self.assertEqual(rt4.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt5.row_splits.dtype, dtypes.int32)

  def testFromRowSplitsWithEmptySplits(self):
    # [FUNCTION DOC]
    # Checks that building from empty values and an empty row splits list raises a value error whose
    # message says the row splits tensor may not be empty.
    err_msg = 'row_splits tensor may not be empty'
    with self.assertRaisesRegex(ValueError, err_msg):
      RaggedTensor.from_row_splits([], [])

  def testFromRowStarts(self):
    # [FUNCTION DOC]
    # Factory test for building from row starts (validate off). Asserts string dtype, shape 5 by unknown,
    # ragged rank 1, the values identity, row count 5, that the row starts accessor returns the input
    # values, and the expected contents.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)

    rt = RaggedTensor.from_row_starts(values, row_starts, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_starts = rt.row_starts()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_starts, row_starts)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLimits(self):
    # [FUNCTION DOC]
    # Factory test for building from row limits (validate off). Asserts string dtype, shape 5 by unknown,
    # ragged rank 1, the values identity, row count 5, that the row limits accessor equals the input
    # and the expected contents.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_limits = constant_op.constant([2, 2, 5, 6, 7], dtypes.int64)

    rt = RaggedTensor.from_row_limits(values, row_limits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_limits = rt.row_limits()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_limits, row_limits)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLengths(self):
    # [FUNCTION DOC]
    # Factory test for building from row lengths (validate off). Asserts string dtype, shape 5 by unknown,
    # ragged rank 1, identity of the values and of the row lengths tensor (cached), row count 5, equal
    # row lengths and the expected contents.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_lengths = constant_op.constant([2, 0, 3, 1, 1], dtypes.int64)

    rt = RaggedTensor.from_row_lengths(values, row_lengths, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_lengths = rt.row_lengths()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_row_lengths, row_lengths)  # cached_nrows
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_lengths, row_lengths)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLengthsInt32(self):
    # [FUNCTION DOC]
    # Checks that row lengths given as a 32-bit tensor work, including when the values are themselves a
    # ragged tensor: builds a ragged tensor from four values and 32-bit lengths, nests it with another
    # row lengths factory call and asserts the outer row lengths evaluate to 2, 1, 0.
    rt = RaggedTensor.from_row_lengths([1, 2, 3, 4],
                                       constant_op.constant([1, 0, 3],
                                                            dtype=dtypes.int32))
    rt2 = RaggedTensor.from_row_lengths(rt, [2, 1, 0])
    self.assertAllEqual([2, 1, 0], rt2.row_lengths())

  def testFromUniformRowLength(self):
    # [FUNCTION DOC]
    # Tests building from a uniform row length. Starts with 16 values and row length 2 (with and without
    # an explicit row count of 8), asserting both give eight rows of two and static shape 8 by 2. Then
    # repeats the construction on top of the previous result (row length 2, row count 4) to get shape
    # 4 by 2 by 2, and once more to get shape 2 by 2 by 2 by 2, asserting the nested list contents and
    # equality of the implicit and explicit row count variants.
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    a1 = RaggedTensor.from_uniform_row_length(values, 2)
    a2 = RaggedTensor.from_uniform_row_length(values, 2, 8)
    self.assertAllEqual(
        a1,
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    self.assertAllEqual(a1, a2)
    self.assertEqual(a1.shape.as_list(), [8, 2])
    self.assertEqual(a2.shape.as_list(), [8, 2])

    b1 = RaggedTensor.from_uniform_row_length(a1, 2)
    b2 = RaggedTensor.from_uniform_row_length(a1, 2, 4)
    self.assertAllEqual(b1, [[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                             [[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    self.assertAllEqual(b1, b2)
    self.assertEqual(b1.shape.as_list(), [4, 2, 2])
    self.assertEqual(b2.shape.as_list(), [4, 2, 2])

    c1 = RaggedTensor.from_uniform_row_length(b1, 2)
    c2 = RaggedTensor.from_uniform_row_length(b1, 2, 2)
    self.assertAllEqual(c1, [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                             [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    self.assertAllEqual(c1, c2)
    self.assertEqual(c1.shape.as_list(), [2, 2, 2, 2])
    self.assertEqual(c2.shape.as_list(), [2, 2, 2, 2])

  def testFromUniformRowLengthWithEmptyValues(self):
    # [FUNCTION DOC]
    # Tests uniform row length with empty values. Row length 0 with 10 rows gives static shape 10 by 0;
    # nesting with row length 2 gives 5 by 2 by 0. Also checks the divide-by-zero case where no row count
    # is given and the row length is zero (shape 0 by 0), and with an explicit row count of zero.
    empty_values = []
    a = RaggedTensor.from_uniform_row_length(empty_values, 0, nrows=10)
    self.assertEqual(a.shape.as_list(), [10, 0])

    b = RaggedTensor.from_uniform_row_length(a, 2)
    self.assertEqual(b.shape.as_list(), [5, 2, 0])

    # Make sure we avoid divide-by-zero when finding nrows for nvals=rowlen=0.
    c = RaggedTensor.from_uniform_row_length(empty_values, 0)
    self.assertEqual(c.shape.as_list(), [0, 0])
    d = RaggedTensor.from_uniform_row_length(empty_values, 0, nrows=0)
    self.assertEqual(d.shape.as_list(), [0, 0])

  def testFromUniformRowLengthWithPlaceholders(self):
    # [FUNCTION DOC]
    # Tests uniform row length when inputs are placeholders with unknown shape. Builds three tensors
    # (placeholder values with constant length, placeholder values with placeholder length, constant
    # values with placeholder length), asserting all evaluate to two rows of three. Static shapes: in
    # eager mode all are 2 by 3; in graph mode the first is unknown by 3 and the others fully unknown.
    # Also nests one more level and asserts contents, and checks a placeholder zero-length case does not
    # divide by zero (shape 0 by 0 in eager mode, unknown by unknown in graph mode).
    ph_values = array_ops.placeholder_with_default([1, 2, 3, 4, 5, 6], [None])
    ph_rowlen = array_ops.placeholder_with_default(3, None)
    rt1 = RaggedTensor.from_uniform_row_length(ph_values, 3)
    rt2 = RaggedTensor.from_uniform_row_length(ph_values, ph_rowlen)
    rt3 = RaggedTensor.from_uniform_row_length([1, 2, 3, 4, 5, 6], ph_rowlen)
    self.assertAllEqual(rt1, [[1, 2, 3], [4, 5, 6]])
    self.assertAllEqual(rt2, [[1, 2, 3], [4, 5, 6]])
    self.assertAllEqual(rt3, [[1, 2, 3], [4, 5, 6]])
    if context.executing_eagerly():
      self.assertEqual(rt1.shape.as_list(), [2, 3])
      self.assertEqual(rt2.shape.as_list(), [2, 3])
      self.assertEqual(rt3.shape.as_list(), [2, 3])
    else:
      self.assertEqual(rt1.shape.as_list(), [None, 3])
      self.assertEqual(rt2.shape.as_list(), [None, None])
      self.assertEqual(rt3.shape.as_list(), [None, None])

    b = RaggedTensor.from_uniform_row_length(rt1, 2)
    self.assertAllEqual(b, [[[1, 2, 3], [4, 5, 6]]])

    # Make sure we avoid divide-by-zero when finding nrows for nvals=rowlen=0.
    ph_empty_values = array_ops.placeholder_with_default(
        array_ops.zeros([0], dtypes.int64), [None])
    ph_zero = array_ops.placeholder_with_default(0, [])
    c = RaggedTensor.from_uniform_row_length(ph_empty_values, ph_zero)
    if context.executing_eagerly():
      self.assertEqual(c.shape.as_list(), [0, 0])
    else:
      self.assertEqual(c.shape.as_list(), [None, None])

  def testFromNestedValueRowIdsWithDerivedNRows(self):
    # [FUNCTION DOC]
    # Tests building a ragged tensor of rank 2 from values and a list of two value row id tensors with
    # derived row counts. Asserts string dtype, shape 4 by unknown by unknown, ragged rank 2, that the
    # innermost values are the identical object, that each level returns the input row ids, and the
    # expected three-level nested contents.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_values_values = rt_values.values
    rt_values_value_rowids = rt_values.value_rowids()

    self.assertIs(rt_values_values, values)
    self.assertAllEqual(rt_value_rowids, nested_value_rowids[0])
    self.assertAllEqual(rt_values_value_rowids, nested_value_rowids[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedRowPartitions(self):
    # [FUNCTION DOC]
    # Tests the private factory that takes a flat values tensor and a list of two row partitions (made
    # from 64-bit row splits). Asserts string dtype, shape 4 by unknown by unknown, ragged rank 2 and the
    # expected nested contents.
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [[0, 2, 3, 3, 5], [0, 2, 2, 5, 6, 7]]
    nested_row_partition = [
        RowPartition.from_row_splits(constant_op.constant(x, dtypes.int64))
        for x in nested_row_splits
    ]

    rt = RaggedTensor._from_nested_row_partitions(
        flat_values, nested_row_partition, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedValueRowIdsWithExplicitNRows(self):
    # [FUNCTION DOC]
    # Tests the nested value row ids factory with explicit row counts for both levels (6 and 6).
    # Asserts dtype, shape 6 by unknown by unknown, ragged rank 2, identity of the innermost values,
    # equality of the row ids and row counts at each level, and nested contents including trailing empty
    # rows.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    nrows = [
        constant_op.constant(6, dtypes.int64),
        constant_op.constant(6, dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids,
                                               nrows)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [6, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()
    rt_values_values = rt_values.values
    rt_values_value_rowids = rt_values.value_rowids()
    rt_values_nrows = rt_values.nrows()

    self.assertIs(rt_values_values, values)
    self.assertAllEqual(rt_value_rowids, nested_value_rowids[0])
    self.assertAllEqual(rt_values_value_rowids, nested_value_rowids[1])
    self.assertAllEqual(rt_nrows, nrows[0])
    self.assertAllEqual(rt_values_nrows, nrows[1])
    self.assertAllEqual(rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [],
                             [[b'f'], [b'g'], []], [], []])

  def testFromNestedValueRowIdsWithExplicitNRowsMismatch(self):
    # [FUNCTION DOC]
    # Checks that passing a list of row counts whose length differs from the list of value row ids
    # raises a value error stating the two arguments must have the same length.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    nrows = [constant_op.constant(6, dtypes.int64)]
    with self.assertRaisesRegex(
        ValueError, 'Argument `nested_nrows` must have the same length as '
        'argument `nested_value_rowids`'):
      RaggedTensor.from_nested_value_rowids(values, nested_value_rowids, nrows)

  def testFromNestedValueRowIdsWithNonListInput(self):
    # [FUNCTION DOC]
    # Checks type validation of the nested value row ids factory: a single tensor instead of a list
    # of row ids raises a type error naming the nested value row ids argument; a single tensor instead of
    # a list for the row counts raises a type error naming the nested row counts argument.
    with self.assertRaisesRegex(
        TypeError, 'Argument `nested_value_rowids` must be a list of Tensors'):
      RaggedTensor.from_nested_value_rowids(
          [1, 2, 3], constant_op.constant([[0, 1, 2], [0, 1, 2]], dtypes.int64))
    with self.assertRaisesRegex(
        TypeError, 'Argument `nested_nrows` must be a list of Tensors'):
      RaggedTensor.from_nested_value_rowids([1, 2, 3], [[0, 1, 2], [0, 1, 2]],
                                            constant_op.constant([3, 3]))

  def testFromNestedRowSplits(self):
    # [FUNCTION DOC]
    # Tests the nested row splits factory with two 64-bit split tensors (validate off). Asserts string
    # dtype, shape 4 by unknown by unknown, ragged rank 2, the identity of the flat values and of each
    # row splits tensor at each level, and the nested contents.
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_row_splits(
        flat_values, nested_row_splits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_values_values = rt_values.values
    rt_values_row_splits = rt_values.row_splits

    self.assertIs(rt_values_values, flat_values)
    self.assertIs(rt_row_splits, nested_row_splits[0])
    self.assertIs(rt_values_row_splits, nested_row_splits[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testWithRowSplits(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertions compare values only, so a failure to actually change the splits dtype
    # would not be detected.
    # Tests changing the row splits dtype. Builds a rank 2 ragged tensor from 64-bit splits, converts it
    # to 32-bit row splits, then asserts dtype, shape, ragged rank, equal values of the splits at each
    # level and the unchanged contents.
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_row_splits(
        flat_values, nested_row_splits, validate=False)

    rt = rt.with_row_splits_dtype(dtypes.int32)

    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_values_values = rt_values.values
    rt_values_row_splits = rt_values.row_splits

    self.assertAllEqual(rt_values_values, flat_values)
    self.assertAllEqual(rt_row_splits, nested_row_splits[0])
    self.assertAllEqual(rt_values_row_splits, nested_row_splits[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedRowSplitsWithNonListInput(self):
    # [FUNCTION DOC]
    # Checks that passing a single tensor instead of a list of row splits tensors to the nested row
    # splits factory raises a type error mentioning the nested row splits argument.
    with self.assertRaisesRegex(
        TypeError, '`nested_row_splits` must be a list of Tensors'):
      RaggedTensor.from_nested_row_splits(
          [1, 2], constant_op.constant([[0, 1, 2], [0, 1, 2]], dtypes.int64))

  def testFromValueRowIdsWithBadNRows(self):
    # [FUNCTION DOC]
    # Checks error messages for bad row counts and ranks in the value row ids factory. Expects a value
    # error for a negative row count (-2), for a row count smaller than the largest row id plus one (given
    # as 2 and as 4 with the largest id being 4), for value row ids of rank 2 (must be rank 1) and for a
    # row count of rank 1 (must be rank 0).
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    with self.assertRaisesRegex(ValueError, r'Expected nrows >= 0; got -2'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=array_ops.placeholder_with_default(value_rowids, None),
          nrows=-2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=2, '
        r'value_rowids\[-1\]=4'):
      RaggedTensor.from_value_rowids(
          values=values, value_rowids=value_rowids, nrows=2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=4, '
        r'value_rowids\[-1\]=4'):
      RaggedTensor.from_value_rowids(
          values=values, value_rowids=value_rowids, nrows=4)

    with self.assertRaisesRegex(ValueError, r'Shape \(7, 1\) must have rank 1'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=array_ops.expand_dims(value_rowids, 1),
          nrows=nrows)

    with self.assertRaisesRegex(ValueError, r'Shape \(1,\) must have rank 0'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=value_rowids,
          nrows=array_ops.expand_dims(nrows, 0))

  def testCondWithTensorsFromValueIds(self):
    # [FUNCTION DOC]
    # Regression test (bug 141166460): a ragged tensor made from value row ids can be returned from both
    # branches of a conditional with a placeholder predicate, and the result equals the original.
    # b/141166460
    rt = RaggedTensor.from_value_rowids([1, 2, 3], [0, 0, 2])
    c = array_ops.placeholder_with_default(True, None)
    result = cond.cond(c, lambda: rt, lambda: rt)
    self.assertAllEqual(rt, result)

  def testGraphMismatch(self):
    # [FUNCTION DOC]
    # Only runs in graph mode (does nothing in eager mode). Creates values in one graph and row splits in
    # another graph, then asserts that building a ragged tensor from them raises a value error saying the
    # tensors must be from the same graph.
    if not context.executing_eagerly():
      with ops.Graph().as_default():
        values = constant_op.constant([1, 2, 3], dtypes.int64)
      with ops.Graph().as_default():
        splits = constant_op.constant([0, 2, 3], dtypes.int64)
      with self.assertRaisesRegex(ValueError,
                                  '.* must be from the same graph as .*'):
        RaggedTensor.from_row_splits(values, splits)

  @parameterized.named_parameters([
      dict(
          testcase_name='Rank0',
          tensor='a'),
      dict(
          testcase_name='Rank1',
          tensor=['a', 'b']),
  ])
  def testFromTensorRankError(self, tensor):
    # [FUNCTION DOC]
    # Parameterized over a scalar string and a rank 1 list of strings. Asserts that converting a tensor
    # of rank below 2 to a ragged tensor raises a value error saying the rank must be greater than 1.
    with self.assertRaisesRegex(ValueError, 'must be greater than 1'):
      RaggedTensor.from_tensor(tensor)

  # =============================================================================
  # Ragged Value & Row-Partitioning Tensor Accessors
  # =============================================================================

  def testRaggedTensorAccessors_2d(self):
    # [FUNCTION DOC]
    # Tests accessors on a 2-D string ragged tensor built from row splits and from value row ids. For
    # both, asserts contents, values (7 entries), value row ids, row count, row splits, row starts, row
    # limits, row lengths, flat values and that there is one nested row splits array.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    rt1 = RaggedTensor.from_row_splits(values, row_splits)
    rt2 = RaggedTensor.from_value_rowids(values, value_rowids)

    for rt in [rt1, rt2]:
      self.assertAllEqual(
          rt, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])
      self.assertAllEqual(rt.values, [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertEqual(rt.values.shape.dims[0].value, 7)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 2, 2, 2, 3, 4])
      self.assertAllEqual(rt.nrows(), 5)
      self.assertAllEqual(rt.row_splits, [0, 2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_starts(), [0, 2, 2, 5, 6])
      self.assertAllEqual(rt.row_limits(), [2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_lengths(), [2, 0, 3, 1, 1])
      self.assertAllEqual(rt.flat_values,
                          [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertLen(rt.nested_row_splits, 1)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 2, 5, 6, 7])

  def testRaggedTensorAccessors_3d_with_ragged_rank_1(self):
    # [FUNCTION DOC]
    # Tests accessors on a 3-D tensor with ragged rank 1 and dense inner dimension 2, built from row
    # splits, value row ids and row lengths. Asserts contents, values, row ids, row count, splits, starts,
    # limits, lengths, row lengths along axis 2 (ragged result), flat values, the single nested row
    # splits and the single nested value row ids array.
    values = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    row_lengths = constant_op.constant([2, 0, 3, 1, 1])
    rt1 = RaggedTensor.from_row_splits(values, row_splits)
    rt2 = RaggedTensor.from_value_rowids(values, value_rowids)
    rt3 = RaggedTensor.from_row_lengths(values, row_lengths)

    for rt in [rt1, rt2, rt3]:
      self.assertAllEqual(rt, [[[0, 1], [2, 3]], [], [[4, 5], [6, 7], [8, 9]],
                               [[10, 11]], [[12, 13]]])
      self.assertAllEqual(
          rt.values,
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertEqual(rt.values.shape.dims[0].value, 7)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 2, 2, 2, 3, 4])
      self.assertAllEqual(rt.nrows(), 5)
      self.assertAllEqual(rt.row_splits, [0, 2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_starts(), [0, 2, 2, 5, 6])
      self.assertAllEqual(rt.row_limits(), [2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_lengths(), [2, 0, 3, 1, 1])
      self.assertAllEqual(
          rt.row_lengths(axis=2), [[2, 2], [], [2, 2, 2], [2], [2]])
      self.assertAllEqual(
          rt.flat_values,
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertLen(rt.nested_row_splits, 1)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 2, 5, 6, 7])
      self.assertLen(rt.nested_value_rowids(), 1)

      self.assertAllEqual(rt.nested_value_rowids()[0], [0, 0, 2, 2, 2, 3, 4])

  def testRaggedTensorAccessors_3d_with_ragged_rank_2(self):
    # [FUNCTION DOC]
    # Tests accessors on a string tensor with ragged rank 2, built from nested row splits and from nested
    # value row ids. Asserts contents, values (5 rows), value row ids, row count 4, splits, starts,
    # limits, lengths, flat values, both nested row splits arrays and both nested value row ids arrays.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    rt1 = RaggedTensor.from_nested_row_splits(values, nested_row_splits)
    rt2 = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids)

    for rt in [rt1, rt2]:
      self.assertAllEqual(
          rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])
      self.assertAllEqual(
          rt.values, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])
      self.assertEqual(rt.values.shape.dims[0].value, 5)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 1, 3, 3])
      self.assertAllEqual(rt.nrows(), 4)
      self.assertAllEqual(rt.row_splits, [0, 2, 3, 3, 5])
      self.assertAllEqual(rt.row_starts(), [0, 2, 3, 3])
      self.assertAllEqual(rt.row_limits(), [2, 3, 3, 5])
      self.assertAllEqual(rt.row_lengths(), [2, 1, 0, 2])
      self.assertAllEqual(rt.flat_values,
                          [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertLen(rt.nested_row_splits, 2)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 3, 3, 5])
      self.assertAllEqual(rt.nested_row_splits[1], [0, 2, 2, 5, 6, 7])
      self.assertLen(rt.nested_value_rowids(), 2)
      self.assertAllEqual(rt.nested_value_rowids()[0], [0, 0, 1, 3, 3])
      self.assertAllEqual(rt.nested_value_rowids()[1], [0, 0, 2, 2, 2, 3, 4])

  # =============================================================================
  # RaggedTensor.shape
  # =============================================================================

  def testShape(self):
    # [FUNCTION DOC]
    # Tests static shape inference. Ragged tensors over flat values give shape 5 by unknown; with inner
    # dimension 2 gives 5 by unknown by 2; a 3-D dense inner part gives 3 by unknown by 2 by 2; nesting
    # adds another unknown dimension. In graph mode only: values with unknown rank give unknown rank,
    # and splits with unknown shape give unknown by unknown.
    """Tests for RaggedTensor.shape."""
    rt1 = RaggedTensor.from_row_splits(b'a b c d e f g'.split(),
                                       [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt1.shape.as_list(), [5, None])

    rt2 = RaggedTensor.from_row_splits(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]],
        [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt2.shape.as_list(), [5, None, 2])

    rt3 = RaggedTensor.from_row_splits(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], [0, 2, 2, 3])
    self.assertEqual(rt3.shape.as_list(), [3, None, 2, 2])

    rt4 = RaggedTensor.from_row_splits(rt3, [0, 1, 3, 3])
    self.assertEqual(rt4.shape.as_list(), [3, None, None, 2, 2])

    if not context.executing_eagerly():
      rt5 = RaggedTensor.from_row_splits(
          array_ops.placeholder(dtype=dtypes.string), [0, 2, 3, 5])
      self.assertIsNone(rt5.shape.ndims)

      rt6 = RaggedTensor.from_row_splits(
          [1, 2, 3], array_ops.placeholder(dtype=dtypes.int64))
      self.assertEqual(rt6.shape.as_list(), [None, None])

  def testGetShape(self):
    # [FUNCTION DOC]
    # Checks that the shape property and the get-shape method return the same dimension list for a
    # ragged tensor made from string values and row splits.
    rt = RaggedTensor.from_row_splits(b'a b c d e f g'.split(),
                                      [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt.shape.as_list(), rt.get_shape().as_list())

  # =============================================================================
  # RaggedTensor.__str__
  # =============================================================================
  def testRaggedTensorStr(self):
    # [FUNCTION DOC]
    # Tests the text representation (repr and str, which must be identical). Builds a ragged tensor of
    # bytes from row splits. In eager mode the expected text is the tensor tag followed by the nested
    # list; in graph mode it is the description naming the values and row splits tensors by their op
    # names with shape and dtype (64-bit splits).
    values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g']
    row_splits = [0, 2, 5, 6, 6, 7]
    rt = RaggedTensor.from_row_splits(values, row_splits, validate=False)
    splits_type = 'int64'
    if context.executing_eagerly():
      expected_repr = '<tf.RaggedTensor {}>'.format([[b'a', b'b'],
                                                     [b'c', b'd', b'e'], [b'f'],
                                                     [], [b'g']])
    else:
      expected_repr = (
          'tf.RaggedTensor(values=Tensor("RaggedFromRowSplits/values:0", '
          'shape=(7,), dtype=string), '
          'row_splits=Tensor('
          '"RaggedFromRowSplits/RowPartitionFromRowSplits/row_splits:0",'
          ' shape=(6,), dtype={}))').format(splits_type)
    self.assertEqual(repr(rt), expected_repr)
    self.assertEqual(str(rt), expected_repr)

  def testRaggedTensorValueStr(self):
    # [FUNCTION DOC]
    # Tests str and repr of the numpy-based RaggedTensorValue. Builds one from a byte string array and
    # 64-bit splits, then compares the whitespace-normalized text with the expected string form (nested
    # list) and the expected repr (values array with byte-string dtype and row splits array).
    values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g']
    row_splits = [0, 2, 5, 6, 6, 7]
    rt = ragged_tensor_value.RaggedTensorValue(
        np.array(values), np.array(row_splits, dtype=np.int64))
    expected_str = '<tf.RaggedTensorValue {}>'.format([[b'a', b'b'],
                                                       [b'c', b'd', b'e'],
                                                       [b'f'], [], [b'g']])
    expected_repr = ("tf.RaggedTensorValue(values=array({}, dtype='|S1'), "
                     'row_splits=array({}))'.format(values, row_splits))
    self.assertEqual(' '.join(str(rt).split()), expected_str)
    self.assertEqual(' '.join(repr(rt).split()), expected_repr)

  def testRaggedTensorStrWithZeroSizeInnerShape(self):
    # [FUNCTION DOC]
    # Regression test for bug 226112826. Only does anything in eager mode: builds a ragged tensor from a
    # dense values tensor of shape 9 by 0 and row lengths 4, 3, 2 and asserts the whitespace-normalized
    # repr is the nested empty-list text. Does nothing in graph mode.
    # Tests that b/226112826 is fixed.
    if context.executing_eagerly():
      rt = RaggedTensor.from_row_lengths(array_ops.zeros([9, 0]), [4, 3, 2])
      expected_repr = (
          '<tf.RaggedTensor [[[], [], [], []], [[], [], []], [[], []]]>')
      self.assertEqual(' '.join(repr(rt).split()), expected_repr)

  # =============================================================================
  # RaggedTensor.with_values() and RaggedTensor.with_flat_values().
  # =============================================================================

  def testWithValues(self):
    # [FUNCTION DOC]
    # Tests with_values and with_flat_values. Adds 10 to the values of a rank 1 ragged tensor, multiplies the
    # flat values of a rank 2 ragged tensor by 10, and expands the dimension of the values of the first
    # (adding an inner dimension). Asserts the row structure is kept and contents are as expected.
    rt1 = ragged_factory_ops.constant([[1, 2], [3, 4, 5], [6], [], [7]])
    rt2 = ragged_factory_ops.constant([[[1, 2], [3, 4, 5]], [[6]], [], [[],
                                                                        [7]]])

    rt1_plus_10 = rt1.with_values(rt1.values + 10)
    rt2_times_10 = rt2.with_flat_values(rt2.flat_values * 10)
    rt1_expanded = rt1.with_values(array_ops.expand_dims(rt1.values, axis=1))

    self.assertAllEqual(rt1_plus_10, [[11, 12], [13, 14, 15], [16], [], [17]])
    self.assertAllEqual(rt2_times_10,
                        [[[10, 20], [30, 40, 50]], [[60]], [], [[], [70]]])
    self.assertAllEqual(rt1_expanded,
                        [[[1], [2]], [[3], [4], [5]], [[6]], [], [[7]]])

  # =============================================================================
  # Session.run
  # =============================================================================
  def testSessionRun(self):
    # [FUNCTION DOC]
    # Graph mode only (returns immediately in eager mode). Runs a session on a dictionary holding two
    # ragged tensors and asserts the result has the same keys and each value converts to the expected
    # nested list.
    if context.executing_eagerly():
      return

    rt1 = ragged_factory_ops.constant([[1, 2, 3], [4]])
    rt2 = ragged_factory_ops.constant([[[], [1, 2]], [[3]]])
    with self.test_session() as session:
      result = session.run({'rt1': rt1, 'rt2': rt2})
      self.assertCountEqual(result.keys(), ['rt1', 'rt2'])
      self.assertEqual(result['rt1'].to_list(), [[1, 2, 3], [4]])
      self.assertEqual(result['rt2'].to_list(), [[[], [1, 2]], [[3]]])

  def testSessionRunFeed(self):
    # [FUNCTION DOC]
    # Graph mode only (returns immediately in eager mode). Builds ragged tensors out of placeholders (one
    # ragged rank 1, one nested ragged rank 2), feeds them with ragged tensor values created from
    # constants and fetches them back. Asserts the keys and the nested list contents of both results.
    if context.executing_eagerly():
      return

    rt1 = RaggedTensor.from_row_splits(
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.int64))
    rt2 = RaggedTensor.from_nested_row_splits(
        array_ops.placeholder(dtypes.int32), [
            array_ops.placeholder(dtypes.int64),
            array_ops.placeholder(dtypes.int64)
        ])

    rt1_feed_val = ragged_factory_ops.constant_value([[1, 2, 3], [4]])
    rt2_feed_val = ragged_factory_ops.constant_value([[[], [1, 2]], [[3]]])

    with self.test_session() as session:
      fetches = {'rt1': rt1, 'rt2': rt2}
      feeds = {rt1: rt1_feed_val, rt2: rt2_feed_val}
      result = session.run(fetches, feed_dict=feeds)
      self.assertCountEqual(result.keys(), ['rt1', 'rt2'])
      self.assertEqual(result['rt1'].to_list(), [[1, 2, 3], [4]])
      self.assertEqual(result['rt2'].to_list(), [[[], [1, 2]], [[3]]])

  def testSessionPartialRunFeed(self):
    # [FUNCTION DOC]
    # Graph mode only (returns immediately in eager mode). Uses partial session runs: two ragged
    # placeholders a, b and a scalar placeholder c; computes row sums of a times b and of a plus c. The
    # first partial run feeds a and b and expects sums 22 and 8; the second feeds c (value 3) and expects
    # 15 and 7.
    if context.executing_eagerly():
      return

    # Placeholder inputs.
    a = RaggedTensor.from_row_splits(
        array_ops.placeholder(dtypes.int32, shape=[None], name='a.values'),
        array_ops.placeholder(dtypes.int64, name='a.row_splits'))
    b = RaggedTensor.from_row_splits(
        array_ops.placeholder(dtypes.int32, shape=[None], name='b.values'),
        array_ops.placeholder(dtypes.int64, name='b.row_splits'))
    c = array_ops.placeholder(dtypes.int32, shape=[], name='c')

    # Feed values for placeholder inputs.
    a_val = ragged_factory_ops.constant_value([[1, 2, 3], [4]])
    b_val = ragged_factory_ops.constant_value([[5, 4, 3], [2]])
    c_val = 3

    # Compute some values.
    r1 = ragged_math_ops.reduce_sum(a * b, axis=1)
    r2 = ragged_math_ops.reduce_sum(a + c, axis=1)

    with self.test_session() as session:
      handle = session.partial_run_setup([r1, r2], [a, b, c])

      res1 = session.partial_run(handle, r1, feed_dict={a: a_val, b: b_val})
      self.assertAllEqual(res1, [22, 8])

      res2 = session.partial_run(handle, r2, feed_dict={c: c_val})
      self.assertAllEqual(res2, [15, 7])

  # Test case for GitHub issue 24679.
  def testEagerForLoop(self):
    # [FUNCTION DOC]
    # Regression test for GitHub issue 24679. Eager mode only (returns in graph mode). Iterates over a
    # ragged tensor of three rows with a Python-style loop and asserts each element equals the
    # corresponding input row.
    if not context.executing_eagerly():
      return

    values = [[1., 2.], [3., 4., 5.], [6.]]
    r = ragged_factory_ops.constant(values)
    i = 0
    for elem in r:
      self.assertAllEqual(elem, values[i])
      i += 1

  def testConsumers(self):
    # [FUNCTION DOC]
    # Graph mode only (returns in eager mode). Creates a ragged tensor from placeholders, applies a
    # reduction to it, and asserts the ragged tensor reports exactly one consumer operation.
    if context.executing_eagerly():
      return

    a = RaggedTensor.from_row_splits(
        array_ops.placeholder(dtypes.int32, shape=[None], name='a.values'),
        array_ops.placeholder(dtypes.int64, name='a.row_splits'),
        validate=False)
    ragged_math_ops.reduce_sum(a)
    self.assertLen(a.consumers(), 1)

  @parameterized.parameters([
      {
          'descr': 'from_value_rowids',
          'factory': RaggedTensor.from_value_rowids,
          'test': RaggedTensor.value_rowids,
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'value_rowids': [0, 0, 1, 1, 2, 2],
          },
          'tensor_field': 'value_rowids',
          'value_rowids': [0, 1, 2],
          'nrows': 10
      },
      {
          'descr': 'from_row_splits',
          'factory': RaggedTensor.from_row_splits,
          # row_splits is a property, not a function.
          'test': (lambda rt: rt.row_splits),
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'row_splits': [0, 2, 4, 6],
          },
          'tensor_field': 'row_splits',
          'row_splits': [0, 1, 2, 3]
      },
      {
          'descr': 'from_row_lengths',
          'factory': RaggedTensor.from_row_lengths,
          'test': RaggedTensor.row_lengths,
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'row_lengths': [2, 2, 2],
          },
          'tensor_field': 'row_lengths',
          'row_lengths': [1, 1, 1],
      },
      # from_row_starts
      {
          'descr': 'from_row_starts',
          'factory': RaggedTensor.from_row_starts,
          'test': RaggedTensor.row_starts,
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'row_starts': [0, 2, 4]
          },
          'tensor_field': 'row_starts',
          'row_starts': [0, 1, 2]
      },
      # from_row_limits
      {
          'descr': 'from_row_limits',
          'factory': RaggedTensor.from_row_limits,
          'test': RaggedTensor.row_limits,
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'row_limits': [2, 4, 6]
          },
          'tensor_field': 'row_limits',
          'row_limits': [3]
      },
      # from_uniform_row_length
      {
          'descr': 'from_uniform_row_length',
          'factory': RaggedTensor.from_uniform_row_length,
          # One cannot extract uniform_row_length or nvals, so we return
          # nvals//nrows = uniform_row_length, where nvals = 3
          'test': (lambda rt: 3 // (rt.shape[0])),
          'values': {
              'values': [1, 2, 3, 4, 5, 6],
              'uniform_row_length': 2
          },
          'tensor_field': 'uniform_row_length',
          'uniform_row_length': 3
      },
  ])
  def testFactoryTypePreference(self, descr, test, factory, values,
                                tensor_field, **kwargs):
    # [FUNCTION DOC]
    # Parameterized over the six single-level factories (value row ids, row splits, row lengths, row starts,
    # row limits, uniform row length). Each case gives the argument values, the name of the row-partitioning
    # tensor argument, an accessor to read it back, and a second argument set. Builds a ragged tensor
    # with that tensor argument as 32-bit tensor, then builds a second tensor using the first one as the
    # values with the second argument set (plain Python numbers), and asserts the accessor of the second
    # tensor returns the expected data. This checks that the dtype of nested row partitions follows the
    # dtype of the inner tensor. The descr parameter is only a label and is unused in the body.
    # When input tensors have shape information, some of these errors will be
    # detected statically.
    def op_cast(k, v):
      # [FUNCTION DOC]
      # HELPER: internal helper for the factory type preference test.
      # Returns the value converted to a 32-bit constant tensor when the key is the name of the
      # row-partitioning argument under test; otherwise returns the value unchanged. Pure function.
      if k == tensor_field:
        return constant_op.constant(v, dtype=dtypes.int32)
      else:
        return v

    value_copy = {k: op_cast(k, v) for k, v in values.items()}
    rt = factory(**value_copy)

    kw_copy = {k: v for k, v in kwargs.items()}
    kw_copy['values'] = rt
    rt2 = factory(**kw_copy)
    self.assertAllEqual(kwargs[tensor_field], test(rt2))

  @parameterized.parameters([
      # from_value_rowids
      {
          'descr': 'bad rank for value_rowids',
          'factory': RaggedTensor.from_value_rowids,
          'values': [[1, 2], [3, 4]],
          'value_rowids': [[1, 2], [3, 4]],
          'nrows': 10,
      },
      {
          'descr': 'bad rank for nrows',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [1, 2, 3, 4],
          'nrows': [10],
      },
      {
          'descr': 'len(values) != len(value_rowids)',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [1, 2, 3, 4, 5],
          'nrows': 10,
      },
      {
          'descr': 'negative value_rowid',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [-5, 2, 3, 4],
          'nrows': 10,
      },
      {
          'descr': 'non-monotonic-increasing value_rowid',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [4, 3, 2, 1],
          'nrows': 10,
      },
      {
          'descr': 'value_rowid > nrows',
          'factory': RaggedTensor.from_value_rowids,
          'values': [1, 2, 3, 4],
          'value_rowids': [1, 2, 3, 4],
          'nrows': 2,
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_value_rowids,
          'values': 10,
          'value_rowids': [1, 2, 3, 4],
          'nrows': 10,
      },
      # from_row_splits
      {
          'descr': 'bad rank for row_splits',
          'factory': RaggedTensor.from_row_splits,
          'values': [[1, 2], [3, 4]],
          'row_splits': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_splits[0] != 0',
          'factory': RaggedTensor.from_row_splits,
          'values': [1, 2, 3, 4],
          'row_splits': [2, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_splits',
          'factory': RaggedTensor.from_row_splits,
          'values': [1, 2, 3, 4],
          'row_splits': [0, 3, 2, 4],
      },
      {
          'descr': 'row_splits[0] != nvals',
          'factory': RaggedTensor.from_row_splits,
          'values': [1, 2, 3, 4],
          'row_splits': [0, 2, 3, 5],
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_row_splits,
          'values': 10,
          'row_splits': [0, 1],
      },
      # from_row_lengths
      {
          'descr': 'bad rank for row_lengths',
          'factory': RaggedTensor.from_row_lengths,
          'values': [1, 2, 3, 4],
          'row_lengths': [[1, 2], [1, 0]],
      },
      {
          'descr': 'negative row_lengths',
          'factory': RaggedTensor.from_row_lengths,
          'values': [1, 2, 3, 4],
          'row_lengths': [3, -1, 2],
      },
      {
          'descr': 'sum(row_lengths) != nvals',
          'factory': RaggedTensor.from_row_lengths,
          'values': [1, 2, 3, 4],
          'row_lengths': [2, 4, 2, 8],
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_row_lengths,
          'values': 10,
          'row_lengths': [0, 1],
      },
      # from_row_starts
      {
          'descr': 'bad rank for row_starts',
          'factory': RaggedTensor.from_row_starts,
          'values': [[1, 2], [3, 4]],
          'row_starts': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_starts[0] != 0',
          'factory': RaggedTensor.from_row_starts,
          'values': [1, 2, 3, 4],
          'row_starts': [2, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_starts',
          'factory': RaggedTensor.from_row_starts,
          'values': [1, 2, 3, 4],
          'row_starts': [0, 3, 2, 4],
      },
      {
          'descr': 'row_starts[0] > nvals',
          'factory': RaggedTensor.from_row_starts,
          'values': [1, 2, 3, 4],
          'row_starts': [0, 2, 3, 5],
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_row_starts,
          'values': 10,
          'row_starts': [0, 1],
      },
      # from_row_limits
      {
          'descr': 'bad rank for row_limits',
          'factory': RaggedTensor.from_row_limits,
          'values': [[1, 2], [3, 4]],
          'row_limits': [[1, 2], [3, 4]],
      },
      {
          'descr': 'row_limits[0] < 0',
          'factory': RaggedTensor.from_row_limits,
          'values': [1, 2, 3, 4],
          'row_limits': [-1, 3, 4],
      },
      {
          'descr': 'non-monotonic-increasing row_limits',
          'factory': RaggedTensor.from_row_limits,
          'values': [1, 2, 3, 4],
          'row_limits': [0, 3, 2, 4],
      },
      {
          'descr': 'row_limits[0] != nvals',
          'factory': RaggedTensor.from_row_limits,
          'values': [1, 2, 3, 4],
          'row_limits': [0, 2, 3, 5],
      },
      {
          'descr': 'bad rank for values',
          'factory': RaggedTensor.from_row_limits,
          'values': 10,
          'row_limits': [0, 1],
      },
      # from_uniform_row_length
      {
          'descr': 'rowlen * nrows != nvals (1)',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4, 5],
          'uniform_row_length': 3,
      },
      {
          'descr': 'rowlen * nrows != nvals (2)',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4, 5],
          'uniform_row_length': 6,
      },
      {
          'descr': 'rowlen * nrows != nvals (3)',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4, 5, 6],
          'uniform_row_length': 3,
          'nrows': 3,
      },
      {
          'descr': 'rowlen must be a scalar',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4],
          'uniform_row_length': [2],
      },
      {
          'descr': 'rowlen must be nonnegative',
          'factory': RaggedTensor.from_uniform_row_length,
          'values': [1, 2, 3, 4],
          'uniform_row_length': -1,
      },
  ])
  def testFactoryValidation(self, descr, factory, **kwargs):
    # [FUNCTION DOC]
    # Parameterized over many invalid inputs to each factory (bad rank, wrong length, negative or
    # non-monotonic ids, mismatched totals and others; each has a description label that is unused in the
    # body). Asserts the factory fails with an invalid argument error or a value error, possibly detected
    # statically. In graph mode only, repeats after wrapping every argument in a 64-bit placeholder with
    # unknown shape and asserts an invalid argument error at run time.
    # When input tensors have shape information, some of these errors will be
    # detected statically.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(factory(**kwargs))

    # Remove shape information (by wrapping tensors in placeholders), and check
    # that we detect the errors when the graph is run.
    if not context.executing_eagerly():

      def wrap_arg(v):
        # [FUNCTION DOC]
        # HELPER: internal helper for the factory validation test.
        # Converts a value to a 64-bit constant and wraps it in a placeholder-with-default of unknown shape so
        # static shape checks cannot catch the error; returns that tensor. No other side effects.
        return array_ops.placeholder_with_default(
            constant_op.constant(v, dtype=dtypes.int64),
            tensor_shape.TensorShape(None))

      kwargs = dict((k, wrap_arg(v)) for (k, v) in kwargs.items())

      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(factory(**kwargs))

  # =============================================================================
  # RaggedTensor Variant conversion
  # =============================================================================

  @parameterized.named_parameters(
      {
          'testcase_name': 'Shape_5_none',
          'ragged_constant': [[1, 2], [3, 4, 5], [6], [], [7]],
          'ragged_rank': 1
      }, {
          'testcase_name': 'Shape_4_none_2',
          'ragged_constant': [[[1, 2]], [], [[3, 4]], []],
          'ragged_rank': 1
      }, {
          'testcase_name': 'Shape_1_none_none',
          'ragged_constant': [[[1], [2, 3, 4, 5, 6, 7]], [[]]],
          'ragged_rank': 2
      })
  def testRaggedToVariant(self, ragged_constant, ragged_rank):
    # [FUNCTION DOC]
    # Parameterized over three ragged constants (ragged ranks 1 and 2). Encodes the ragged tensor as a
    # variant tensor (not batched) and asserts the result is a scalar (empty shape) of variant dtype.
    rt = ragged_factory_ops.constant(ragged_constant, ragged_rank=ragged_rank)
    et = rt._to_variant()
    self.assertEqual(et.shape.as_list(), [])
    self.assertEqual(et.dtype, dtypes.variant)

  @parameterized.parameters(
      {
          'ragged_constant': [[1, 2], [3, 4, 5], [6], [], [7]],
          'ragged_rank': 1,
          'num_batched_elems': 5
      }, {
          'ragged_constant': [[[1, 2]], [], [[3, 4]], []],
          'ragged_rank': 1,
          'num_batched_elems': 4
      }, {
          'ragged_constant': [[[1], [2, 3, 4, 5, 6, 7]], [[]]],
          'ragged_rank': 2,
          'num_batched_elems': 2
      })
  def testRaggedToBatchedVariant(self, ragged_constant, ragged_rank,
                                 num_batched_elems):
    # [FUNCTION DOC]
    # Parameterized over three ragged constants with the expected number of batched elements. Encodes
    # the ragged tensor with the batched flag and asserts the variant tensor has one element per outer
    # row (shape equal to that number) and variant dtype.
    rt = ragged_factory_ops.constant(ragged_constant, ragged_rank=ragged_rank)
    et = rt._to_variant(batched_input=True)
    self.assertEqual(et.shape.as_list(), [num_batched_elems])
    self.assertEqual(et.dtype, dtypes.variant)

  @parameterized.parameters(
      # 2D test cases.
      {
          'ragged_constant': [[]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1, 2]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1], [2], [3]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          'ragged_rank': 1,
      },
      {
          'ragged_constant': [[1, 2], [3, 4, 5], [6], [], [7]],
          'ragged_rank': 1,
      },
      # 3D test cases.
      {
          'ragged_constant': [[[]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1, 2]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1, 2], [3, 4]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]],
          'ragged_rank': 2,
      },
      {
          'ragged_constant': [[[1, 2]], [], [[3, 4]], []],
          'ragged_rank': 2,
      },
      # 4D test cases.
      {
          'ragged_constant': [[[[1, 2], [3, 4]]],
                              [[[0, 0], [0, 0]], [[5, 6], [7, 8]]], []],
          'ragged_rank': 3,
      },
      # dtype `string`.
      {
          'ragged_constant': [['a'], ['b'], ['c']],
          'ragged_rank': 1,
          'dtype': dtypes.string,
      },
      {
          'ragged_constant': [[['a', 'b'], ['c', 'd']]],
          'ragged_rank': 2,
          'dtype': dtypes.string,
      },
      {
          'ragged_constant': [[[['a', 'b'], ['c', 'd']]],
                              [[['e', 'f'], ['g', 'h']], [['i', 'j'],
                                                          ['k', 'l']]], []],
          'ragged_rank': 3,
          'dtype': dtypes.string,
      })
  def testVariantRoundTrip(self,
                           ragged_constant,
                           ragged_rank,
                           dtype=dtypes.int32):
    # [FUNCTION DOC]
    # Parameterized over 2-D to 4-D integer cases and string cases (default dtype is 32-bit integer).
    # Encodes the ragged tensor to a variant and decodes it with the same ragged rank and dtype, asserting
    # the result equals the original.
    rt = ragged_factory_ops.constant(
        ragged_constant, ragged_rank=ragged_rank, dtype=dtype)
    et = rt._to_variant()
    round_trip_rt = RaggedTensor._from_variant(
        et, dtype, output_ragged_rank=ragged_rank)
    self.assertAllEqual(rt, round_trip_rt)

  def testBatchedVariantRoundTripInputRaggedRankInferred(self):
    # [FUNCTION DOC]
    # Encodes a ragged tensor of ten single-element rows as batched variant, reshapes the variants to
    # 5 by 2, and decodes with output ragged rank 2 leaving the input ragged rank to be inferred. Asserts
    # the decoded tensor equals the expected five rows of two single-element rows.
    ragged_rank = 1
    rt = ragged_factory_ops.constant(
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
        ragged_rank=ragged_rank)
    batched_variant = rt._to_variant(batched_input=True)
    nested_batched_variant = array_ops.reshape(batched_variant, [5, 2])
    decoded_rt = RaggedTensor._from_variant(
        nested_batched_variant,
        dtype=dtypes.int32,
        output_ragged_rank=ragged_rank + 1)
    expected_rt = ragged_factory_ops.constant([[[0], [1]], [[2], [3]], [[4],
                                                                        [5]],
                                               [[6], [7]], [[8], [9]]])
    self.assertAllEqual(decoded_rt, expected_rt)

  def testBatchedVariantRoundTripWithInputRaggedRank(self):
    # [FUNCTION DOC]
    # Same scenario as the inferred-rank test, but passes the input ragged rank explicitly (one less than
    # the output). Asserts the decoded tensor equals the expected nested list.
    ragged_rank = 1
    rt = ragged_factory_ops.constant(
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
        ragged_rank=ragged_rank)
    batched_variant = rt._to_variant(batched_input=True)
    nested_batched_variant = array_ops.reshape(batched_variant, [5, 2])
    decoded_rt = RaggedTensor._from_variant(
        nested_batched_variant,
        dtype=dtypes.int32,
        output_ragged_rank=ragged_rank + 1,
        input_ragged_rank=ragged_rank - 1)
    expected_rt = ragged_factory_ops.constant([[[0], [1]], [[2], [3]], [[4],
                                                                        [5]],
                                               [[6], [7]], [[8], [9]]])
    self.assertAllEqual(decoded_rt, expected_rt)

  def testUnbatchVariant(self):  # b/141789000
    # [FUNCTION DOC]
    # Regression test for bug 141789000. Encodes a ragged tensor of four rows as batched variants, decodes
    # each batch element separately with output ragged rank 0, and asserts each equals the corresponding
    # row of the original.
    rt = ragged_factory_ops.constant([[1, 2, 3], [4, 5], [], [6, 7, 8, 9]])
    batched = rt._to_variant(batched_input=True)
    for i in range(4):
      row = RaggedTensor._from_variant(
          batched[i], dtype=dtypes.int32, output_ragged_rank=0)
      self.assertAllEqual(rt[i], row)

  def testUnbatchVariantInDataset(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the graph-mode branch loops only over the first three of the four rows and repeatedly
    # compares against the same get-next tensor, so it does not check all rows the way the eager branch does.
    # Creates a dataset from the slices of a ragged tensor. In eager mode asserts each element equals
    # the corresponding row. In graph mode uses a one-shot iterator and compares session results against
    # the iterator output.
    rt = ragged_factory_ops.constant([[1, 2, 3], [4, 5], [], [6, 7, 8, 9]])
    ds = dataset_ops.Dataset.from_tensor_slices(rt)
    if context.executing_eagerly():
      for i, value in enumerate(ds):
        self.assertAllEqual(rt[i], value)
    else:
      it = dataset_ops.make_one_shot_iterator(ds)
      out = it.get_next()
      with self.cached_session() as sess:
        for i in range(3):
          self.assertAllEqual(sess.run(rt[i]), out)

  def testToVariantInvalidParams(self):
    # [FUNCTION DOC]
    # Calls the raw ragged-to-variant op with invalid nested splits (a rank 0 element, then rank 3 data)
    # and batched input set. Asserts either a value error or an invalid argument error whose message says
    # the input should be rank 1 but is rank 0 and then rank 2 respectively.
    self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                           r'be rank 1 but is rank 0',
                           gen_ragged_conversion_ops.ragged_tensor_to_variant,
                           rt_nested_splits=[0, 1, 2],
                           rt_dense_values=[0, 1, 2],
                           batched_input=True)

    self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                           r'be rank 1 but is rank 2',
                           gen_ragged_conversion_ops.ragged_tensor_to_variant,
                           rt_nested_splits=[[[0]], [[1]], [[2]]],
                           rt_dense_values=[0, 1, 2],
                           batched_input=True)

  def testFromVariantInvalidParams(self):
    # [FUNCTION DOC]
    # Encodes a four row ragged tensor as batched variants, reshapes to 2 by 2 and decodes with output
    # ragged rank 1 and input ragged rank 1. Asserts a value error saying the output ragged rank must
    # be equal to a function of the input ragged rank.
    rt = ragged_factory_ops.constant([[0], [1], [2], [3]])
    batched_variant = rt._to_variant(batched_input=True)
    nested_batched_variant = array_ops.reshape(batched_variant, [2, 2])
    with self.assertRaisesRegex(ValueError,
                                r'`output_ragged_rank` \(1\) must be equal to'):
      RaggedTensor._from_variant(
          nested_batched_variant,
          dtype=dtypes.int32,
          output_ragged_rank=1,
          input_ragged_rank=1)

  def testUnbatchToTensor(self):
    # [FUNCTION DOC]
    # Uses the type specification of a ragged tensor of four single-element rows. Converts the tensor to
    # a batched tensor list, unstacks each component, rebuilds each element from the unbatched spec,
    # and asserts the number of elements, that each is compatible with the unbatched spec, and that
    # each equals the expected dense tensor. The unbatched spec can yield a dense tensor instead of a
    # ragged one.
    batched = ragged_factory_ops.constant([[0], [1], [2], [3]])
    unbatched = [constant_op.constant(x) for x in [[0], [1], [2], [3]]]
    batched_spec = type_spec.type_spec_from_value(batched)

    # Note that the unbatched_spec is derived from the batched spec, so it can
    # add back a ragged instead of a dense tensor.
    unbatched_spec = batched_spec._unbatch()
    batched_tensor_list = batched_spec._to_batched_tensor_list(batched)
    unbatched_tensor_lists = zip(
        *[array_ops_stack.unstack(tensor) for tensor in batched_tensor_list])
    actual_unbatched = [
        batched_spec._unbatch()._from_tensor_list(tensor_list)
        for tensor_list in unbatched_tensor_lists]
    self.assertLen(actual_unbatched, len(unbatched))
    for x in actual_unbatched:
      self.assertTrue(unbatched_spec.is_compatible_with(x))

    for (actual, expected) in zip(actual_unbatched, unbatched):
      self.assertAllEqual(actual, expected)

  def testDatasetUnbatchTwice(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in graph mode the test asserts nothing, it only builds the dataset.
    # Creates a dataset holding one nested ragged tensor and unbatches it twice. In eager mode asserts the
    # first element of the result equals a list with the single element zero.
    batched = ragged_factory_ops.constant([[[0], [1], [5]], [[2], [3]]])
    ds = dataset_ops.Dataset.from_tensors(batched)
    ds2 = ds.unbatch()
    ds3 = ds2.unbatch()
    if context.executing_eagerly():
      value = next(iter(ds3))
      self.assertAllEqual([0], value)

  def testDatasetUnbatchToScalar(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in graph mode the test asserts nothing, it only builds the dataset.
    # Creates a dataset from a ragged tensor of four single-element rows and unbatches it twice. In eager
    # mode asserts that the first element is the scalar 0.
    batched = ragged_factory_ops.constant([[0], [1], [2], [3]])
    ds = dataset_ops.Dataset.from_tensors(batched)
    ds2 = ds.unbatch()
    ds3 = ds2.unbatch()
    if context.executing_eagerly():
      value = next(iter(ds3))
      self.assertAllEqual(0, value)

  def testBatchToTensor(self):
    # [FUNCTION DOC]
    # Inverse of the unbatch test: converts each of four dense single-element tensors to tensor lists
    # through the unbatched spec, stacks them, rebuilds with the batched spec for batch size 4 and
    # asserts the result equals the original ragged tensor.
    batched = ragged_factory_ops.constant([[0], [1], [2], [3]])
    unbatched = [constant_op.constant(x) for x in [[0], [1], [2], [3]]]
    batched_spec = type_spec.type_spec_from_value(batched)

    # Note that the unbatched_spec is derived from the batched spec, so it can
    # add back a ragged instead of a dense tensor.
    unbatched_spec = batched_spec._unbatch()
    unbatched_tensor_lists = [unbatched_spec._to_tensor_list(x)
                              for x in unbatched]
    batched_tensor_list = [array_ops_stack.stack(tensors)
                           for tensors in zip(*unbatched_tensor_lists)]
    actual_batched = unbatched_spec._batch(4)._from_tensor_list(
        batched_tensor_list)
    self.assertAllEqual(actual_batched, batched)

  def _testGradient(self, func, x, expected_grad, grad_y=None):
    # [FUNCTION DOC]
    # HELPER: shared gradient checker for the gradient tests in this class.
    # Inputs: func (a callable mapping the ragged tensor to an output), x (nested list for the input,
    # converted to a ragged tensor), expected_grad (nested list, or nothing when no gradient is
    # expected), grad_y (optional upstream gradient, also converted to a ragged tensor, default none).
    # Eager mode: records func under a gradient tape that watches x and queries the gradient; graph
    # mode: uses the symbolic gradients function. If expected_grad is nothing, asserts the gradient is
    # nothing; otherwise converts it to a dense or ragged tensor and asserts closeness to the expectation.
    # Returns nothing.
    x = ragged_factory_ops.constant(x)
    if grad_y is not None:
      grad_y = ragged_factory_ops.constant(grad_y)
    if context.executing_eagerly():
      with backprop.GradientTape() as t:
        t.watch(x)
        y = func(x)
      g = t.gradient(y, x, grad_y)
    else:
      y = func(x)
      g = gradients_impl.gradients(ys=y, xs=x, grad_ys=grad_y)[0]
    if expected_grad is None:
      self.assertIsNone(g)
    else:
      g = ragged_tensor.convert_to_tensor_or_ragged_tensor(g)
      self.assertAllClose(g, expected_grad)

  @parameterized.named_parameters([
      dict(
          testcase_name='RaggedInput',
          func=lambda x: math_ops.reduce_prod(x, axis=1),
          x=[[1., 2.], [3.]],
          expected=[[2., 1.], [1.]]),
      dict(
          testcase_name='RaggedOutput',
          func=lambda x: ragged_concat_ops.stack([x, x[:1]]),
          x=[3., 2.],
          expected=[2., 1.]),
      dict(
          testcase_name='RaggedInputAndOutput',
          func=lambda x: array_ops_stack.stack([x, x * x]),
          x=[[1., 2.], [3.]],
          expected=[[3., 5.], [7.]]),
      dict(
          testcase_name='RaggedOutputWithGradYs',
          func=lambda x: ragged_concat_ops.stack([x, x[:1]]),
          x=[3., 2.],
          grad_ys=[[1., 1.], [1.]],
          expected=[2., 1.]),
      dict(
          testcase_name='RaggedInputAndOutputWithGradYs',
          func=lambda x: array_ops_stack.stack([x, x * x]),
          x=[[1., 2.], [3.]],
          grad_ys=[[[1., 1.], [1.]], [[1., 1.], [1.]]],
          expected=[[3., 5.], [7.]]),
      dict(
          testcase_name='RaggedRank3',
          func=lambda x: ragged_concat_ops.stack([x, (x * x)[:, 1:]]),
          x=[[[1., 2.], [3., 4., 5.]], [[6.]]],
          expected=[[[1.0, 1.0], [7.0, 9.0, 11.0]], [[1.0]]]),
      dict(
          testcase_name='RaggedIndexedSlices',
          func=lambda x: ragged_gather_ops.gather(x, [0, 2]),
          x=[[1., 2.], [3.], [4., 5., 6.]],
          expected=[[1., 1.], [0.], [1., 1., 1.]]),
  ])
  def testGradient(self, func, x, expected, grad_ys=None):
    # [FUNCTION DOC]
    # Parameterized (named cases): reduce product over a ragged axis, ragged stack output, stack of
    # ragged input, the same two with explicit upstream gradients, a ragged rank 3 stack, and a gather
    # (indexed slices) case. Delegates to the gradient checker with the function, input, expected gradient
    # and optional upstream gradient.
    self._testGradient(func, x, expected, grad_ys)

  def testHigherOrderGradient(self):
    # [FUNCTION DOC]
    # Computes second derivatives with nested gradient tapes on a ragged tensor: y is x cubed, so the
    # first derivative is 3 x squared and the second 6 x. Asserts both nested lists.
    x = ragged_factory_ops.constant([[1.0, 2.0], [3.0]])

    with backprop.GradientTape() as t2:
      t2.watch(x)
      with backprop.GradientTape() as t1:
        t1.watch(x)
        y = x * x * x
      dy_dx = t1.gradient(y, x)
    d2y_dx2 = t2.gradient(dy_dx, x)

    self.assertAllEqual(dy_dx, [[3.0, 12.0], [27.0]])
    self.assertAllEqual(d2y_dx2, [[6.0, 12.0], [18.0]])

  def testUnconnectedGradient(self):
    # [FUNCTION DOC]
    # Watches a ragged tensor but computes the output from unrelated constants; asserts the gradient is
    # nothing (no connection).
    x = ragged_factory_ops.constant([[1.0, 2.0], [3.0]])

    with backprop.GradientTape() as t:
      t.watch(x)
      y = ragged_factory_ops.constant([[2.0, 4.0], [6.0]])
    self.assertIsNone(t.gradient(y, x))

  def testStopGradient(self):
    # [FUNCTION DOC]
    # Gradient test: output is the sum of x times a ragged tensor derived from x with its values passed
    # through stop-gradient. Uses a local function and asserts via the checker that only the direct
    # path contributes gradients, giving the nested expected gradient.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: multiplies the ragged input by a column of constants, stops the gradient on
      # the values of the product, multiplies the input by it and returns the sum of all elements.
      y = x * constant_op.constant([[1.], [3.]])
      y = y.with_values(array_ops.stop_gradient(y.values))
      z = x * y
      return math_ops.reduce_sum(z)

    self._testGradient(func, [[1., 2.], [3., 4., 5.]],
                       [[1., 2.], [9., 12., 15.]])

  def testStopGradientNoneComponent(self):
    # [FUNCTION DOC]
    # Gradient test where all paths are stopped: the function returns a ragged tensor whose values have
    # passed through stop-gradient. Asserts the computed gradient is nothing.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: multiplies the ragged input by a column of constants, replaces the values with
      # their stop-gradient version and returns the resulting ragged tensor.
      y = x * constant_op.constant([[1.], [3.]])
      y = y.with_values(array_ops.stop_gradient(y.values))
      return y

    self._testGradient(func, [[1., 2], [3, 4, 5]], None)

  def testRaggedVariantGradients(self):
    # [FUNCTION DOC]
    # Gradient test through variant encoding. A local function builds a ragged tensor from a flat input
    # with row splits 0, 4, 7, 8, scales rows by 10, 100, 1000, encodes it to a variant (not batched),
    # decodes it and returns its flat values. Asserts the gradient is the row scale per element.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: scales the rows of a ragged tensor made from the flat input, round-trips it
      # through a non-batched variant and returns the flat values.
      rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
      rt2 = rt1 * [[10], [100], [1000]]
      v = rt2._to_variant(batched_input=False)
      rt3 = RaggedTensor._from_variant(v, dtype=rt2.dtype, output_ragged_rank=1)
      return rt3.flat_values

    self._testGradient(func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                       [10., 10., 10., 10., 100., 100., 100., 1000.])

  def testRaggedVariantGradientsEmptyRows(self):
    # [FUNCTION DOC]
    # Same as the variant gradient test, with row splits that include empty rows and scales of 10 to 60.
    # Asserts the gradient is the row scale for elements in non-empty rows.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: scales rows (some empty) of a ragged tensor made from the flat input,
      # round-trips it through a non-batched variant and returns the flat values.
      rt1 = RaggedTensor.from_row_splits(
          values=x, row_splits=[0, 2, 2, 4, 7, 7, 8])
      rt2 = rt1 * [[10], [20], [30], [40], [50], [60]]
      v = rt2._to_variant(batched_input=False)
      rt3 = RaggedTensor._from_variant(v, dtype=rt2.dtype, output_ragged_rank=1)
      return rt3.flat_values

    self._testGradient(func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                       [10., 10., 30., 30., 40., 40., 40., 60.])

  def testRaggedVariantSteps(self):
    # [FUNCTION DOC]
    # Forward-only version of the variant test: builds the tensor from eight values, multiplies rows
    # by 10, 100 and 1000, round-trips through a variant and asserts the flat values equal the products.
    x = [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0]
    rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
    rt2 = rt1 * [[10], [100], [1000]]
    v = rt2._to_variant(batched_input=False)
    rt3 = RaggedTensor._from_variant(v, dtype=rt2.dtype, output_ragged_rank=1)
    self.assertAllClose([30., 10., 40., 10., 100., 0., 200., 1000.],
                        rt3.flat_values)

  def testRaggedVariantGradientsBatched(self):
    # [FUNCTION DOC]
    # Like the variant gradient test but the variant is created in batched mode. Asserts the same
    # gradient (row scale per element).

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: scales rows, round-trips through a batched variant and returns flat values.
      rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
      rt2 = rt1 * [[10], [100], [1000]]
      v = rt2._to_variant(batched_input=True)
      rt3 = RaggedTensor._from_variant(v, dtype=rt2.dtype, output_ragged_rank=1)
      return rt3.flat_values

    self._testGradient(func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                       [10., 10., 10., 10., 100., 100., 100., 1000.])

  def testRaggedVariantGradientsEmptyRowsBatched(self):
    # [FUNCTION DOC]
    # Batched variant gradient test with empty rows in the splits. Asserts gradient equals the row scale
    # for elements in non-empty rows.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: scales rows (some empty), round-trips through a batched variant and
      # returns flat values.
      rt1 = RaggedTensor.from_row_splits(
          values=x, row_splits=[0, 2, 2, 4, 7, 7, 8])
      rt2 = rt1 * [[10], [20], [30], [40], [50], [60]]
      v = rt2._to_variant(batched_input=True)
      rt3 = RaggedTensor._from_variant(v, dtype=rt2.dtype, output_ragged_rank=1)
      return rt3.flat_values

    self._testGradient(func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                       [10., 10., 30., 30., 40., 40., 40., 60.])

  def testRaggedVariantGradientsEmptyOutputBatched(self):
    # [FUNCTION DOC]
    # Batched variant gradient test where all rows are empty and the input is empty. Asserts the
    # gradient is an empty list.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: builds a ragged tensor with all-empty rows from the flat input, scales it,
      # round-trips through a batched variant and returns the flat values.
      rt1 = RaggedTensor.from_row_splits(
          values=x, row_splits=[0, 0, 0, 0, 0, 0, 0])
      rt2 = rt1 * [[10], [20], [30], [40], [50], [60]]
      v = rt2._to_variant(batched_input=True)
      rt3 = RaggedTensor._from_variant(v, dtype=rt2.dtype, output_ragged_rank=1)
      return rt3.flat_values

    self._testGradient(func, [], [])

  def testRaggedVariantGradientsBatchedAndSliced(self):
    # [FUNCTION DOC]
    # Gradient test of one element of the batched variant. A local function takes input x and an index i
    # and decodes only the i-th variant as a plain tensor. For indices 0, 1 and 2 the expected gradient is
    # the row scale on that row's elements and zero elsewhere.

    def func(x, i):
      # [FUNCTION DOC]
      # Local function under test: scales rows of a ragged tensor from x, encodes batched, picks the variant at
      # position i and decodes it with output ragged rank 0 (a dense row).
      rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
      rt2 = rt1 * [[10], [100], [1000]]
      v_slice = rt2._to_variant(batched_input=True)[i]
      return RaggedTensor._from_variant(
          v_slice, dtype=rt2.dtype, output_ragged_rank=0)

    self._testGradient(
        functools.partial(func, i=0), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [10., 10., 10., 10., 0., 0., 0., 0.])
    self._testGradient(
        functools.partial(func, i=1), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 0., 0., 100., 100., 100., 0.])
    self._testGradient(
        functools.partial(func, i=2), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 0., 0., 0., 0., 0., 1000.])

  def testRaggedVariantGradientsEmptyRowsBatchedAndSliced(self):
    # [FUNCTION DOC]
    # Same as the sliced batched test with empty rows in the splits; checks indices 0 to 5, expecting
    # zeros for empty rows and the row scale elsewhere.

    def func(x, i):
      # [FUNCTION DOC]
      # Local function under test: scales rows (some empty), encodes batched, picks the variant at i and
      # decodes it as a dense row.
      rt1 = RaggedTensor.from_row_splits(
          values=x, row_splits=[0, 2, 2, 4, 7, 7, 8])
      rt2 = rt1 * [[10], [20], [30], [40], [50], [60]]
      v_slice = rt2._to_variant(batched_input=True)[i]
      return RaggedTensor._from_variant(
          v_slice, dtype=rt2.dtype, output_ragged_rank=0)

    self._testGradient(
        functools.partial(func, i=0), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [10., 10., 0., 0., 0., 0., 0., 0.])
    self._testGradient(
        functools.partial(func, i=1), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 0., 0., 0., 0., 0., 0.])
    self._testGradient(
        functools.partial(func, i=2), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 30., 30., 0., 0., 0., 0.])
    self._testGradient(
        functools.partial(func, i=3), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 0., 0., 40., 40., 40., 0.])
    self._testGradient(
        functools.partial(func, i=4), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 0., 0., 0., 0., 0., 0.])
    self._testGradient(
        functools.partial(func, i=5), [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
        [0., 0., 0., 0., 0., 0., 0., 60.])

  def testRaggedVariantGradientsRaggedRank0(self):
    # [FUNCTION DOC]
    # Gradient test of the variant path with no ragged dimension: the input doubled is encoded with no
    # nested splits and decoded with ragged rank 0. Asserts every gradient equals 2.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: doubles the input, encodes it into a non-batched variant with no splits
      # and decodes it with output ragged rank 0.
      x2 = x * 2
      v = gen_ragged_conversion_ops.ragged_tensor_to_variant(
          [], x2, batched_input=False)
      return RaggedTensor._from_variant(v, dtype=x2.dtype, output_ragged_rank=0)

    self._testGradient(func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                       [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

  def testRaggedVariantGradientsRaggedRank3(self):
    # [FUNCTION DOC]
    # Gradient test of the variant path for a ragged rank 3 tensor. Asserts every gradient equals 2.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: doubles the input, builds a ragged rank 3 tensor from three splits lists,
      # round-trips through a non-batched variant and returns the flat values.
      x2 = x * 2
      rt1 = RaggedTensor.from_nested_row_splits(
          x2, ([0, 0, 3], [0, 2, 2, 3], [0, 4, 7, 8]))
      v = rt1._to_variant(batched_input=False)
      rt3 = RaggedTensor._from_variant(v, dtype=x2.dtype, output_ragged_rank=3)
      return rt3.flat_values

    self._testGradient(func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                       [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

  def testRaggedVariantGradientsViaMapFn(self):
    # [FUNCTION DOC]
    # Gradient through mapping over rows. A ragged tensor of three rows is constant; the function maps over
    # its rows computing the root of the mean of squared (row times x), then sums. Asserts the gradient
    # with respect to scalar 3.0 is about 14.653377.
    rt = RaggedTensor.from_row_splits(
        values=[3, 1.0, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 7, 8])

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: maps the per-row root-mean-square function over the outer ragged tensor
      # using the scalar input and returns the sum.

      def transform_row(row):
        # [FUNCTION DOC]
        # Local function under test: scales one row by the outer scalar and returns the root of the mean of squares
        # as a one-element tensor.
        return math_ops.sqrt(
            math_ops.reduce_mean(math_ops.square(row * x), keepdims=True))

      return math_ops.reduce_sum(map_fn.map_fn(transform_row, rt))

    self._testGradient(func, 3.0, 14.653377)

  def testRaggedVariantGradientsEmptyRowsViaMapFn(self):
    # [FUNCTION DOC]
    # Same as the map function gradient test with rows including empty ones (splits 0,2,2,4,7,7,8). Asserts
    # the gradient at 3.0 is about 17.206844.
    rt = RaggedTensor.from_row_splits(
        values=[3, 1.0, 4, 1, 5, 9, 2, 6], row_splits=[0, 2, 2, 4, 7, 7, 8])

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: maps the per-row root-mean-square function over the ragged tensor and
      # returns the sum.

      def transform_row(row):
        # [FUNCTION DOC]
        # Local function under test: scales one row by the outer scalar and returns the root of the mean of squares
        # as a one-element tensor.
        return math_ops.sqrt(
            math_ops.reduce_mean(math_ops.square(row * x), keepdims=True))

      return math_ops.reduce_sum(map_fn.map_fn(transform_row, rt))

    self._testGradient(func, 3.0, 17.206844)

  def testRaggedVariantGradientsEmptyOutputViaMapFn(self):
    # [FUNCTION DOC]
    # Map function gradient test where the ragged tensor has no values and three empty rows. Asserts the
    # gradient at 3.0 is 0.0.
    rt = RaggedTensor.from_row_splits(
        values=[], row_splits=[0, 0, 0, 0])

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: maps the per-row root-mean-square function over the empty ragged tensor
      # and returns the sum.

      def transform_row(row):
        # [FUNCTION DOC]
        # Local function under test: scales one row by the outer scalar and returns the root of the mean of squares
        # as a one-element tensor.
        return math_ops.sqrt(
            math_ops.reduce_mean(math_ops.square(row * x), keepdims=True))

      return math_ops.reduce_sum(map_fn.map_fn(transform_row, rt))

    self._testGradient(func, 3.0, 0.0)

  def testRaggedVariantGradientsViaMapFnReduce(self):
    # [FUNCTION DOC]
    # Gradient through a mapped maximum reduction per row. Builds a ragged tensor from the input and maps
    # the max reduction over rows with a scalar output spec. Asserts the gradient is one at each row's
    # maximum position and zero elsewhere.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: builds a ragged tensor from the flat input and maps the maximum reduction
      # over its rows with a declared scalar output signature.
      rt1 = RaggedTensor.from_row_splits(values=x, row_splits=[0, 4, 7, 8])
      return map_fn.map_fn(
          math_ops.reduce_max,
          rt1,
          fn_output_signature=tensor_lib.TensorSpec((), x.dtype))

    self._testGradient(func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])

  def testRaggedVariantGradientsEmptyRowsViaMapFnReduce(self):
    # [FUNCTION DOC]
    # Same as the mapped maximum gradient test, with empty rows in the splits. Asserts expected gradient
    # positions.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: builds a ragged tensor (with empty rows) from the flat input and maps the
      # maximum reduction over its rows.
      rt1 = RaggedTensor.from_row_splits(
          values=x, row_splits=[0, 2, 2, 4, 7, 7, 8])
      return map_fn.map_fn(
          math_ops.reduce_max,
          rt1,
          fn_output_signature=tensor_lib.TensorSpec((), x.dtype))

    self._testGradient(func, [3.0, 1.0, 4.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                       [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])

  def testRaggedVariantGradientsEmptyOutputViaMapFnReduce(self):
    # [FUNCTION DOC]
    # Mapped maximum gradient test with an empty input and all rows empty. Asserts an empty gradient.

    def func(x):
      # [FUNCTION DOC]
      # Local function under test: builds a ragged tensor of empty rows from the flat input and maps the
      # maximum reduction over its rows.
      rt1 = RaggedTensor.from_row_splits(
          values=x, row_splits=[0, 0, 0, 0])
      return map_fn.map_fn(
          math_ops.reduce_max,
          rt1,
          fn_output_signature=tensor_lib.TensorSpec((), x.dtype))

    self._testGradient(func, [], [])

  def testRaggedVariantGradientsErrors(self):
    # [FUNCTION DOC]
    # Graph mode only (returns in eager mode). Stacks the variant of a ragged tensor into a rank 2
    # variant tensor, decodes it with ragged rank 3 and asks for gradients; asserts a value error saying
    # the variant conversion can only produce 0-D or 1-D output gradients.
    if context.executing_eagerly():
      return

    rt = RaggedTensor.from_row_splits([1.0, 2.0], row_splits=[0, 2, 2])
    v1 = rt._to_variant()
    v2 = array_ops_stack.stack([array_ops_stack.stack([v1])])
    y = RaggedTensor._from_variant(v2, rt.dtype, output_ragged_rank=3)

    with self.assertRaisesRegex(
        ValueError, 'Unable to compute gradient: RaggedTensorToVariant '
        'can currently only generate 0D or 1D output.'):
      gradients_impl.gradients(ys=y.flat_values, xs=rt.flat_values)

  def assertNumpyObjectTensorsRecursivelyEqual(self, a, b, msg):
    # [FUNCTION DOC]
    # HELPER: custom assertion for the numpy conversion test.
    # Compares two numpy arrays. Inputs: a (the expected array), b (the actual array) and msg (the
    # message shown on failure). If the first is an object-dtype array it asserts equal dtype, shape and
    # length and recurses element-wise; otherwise it asserts value equality. Returns nothing; raises
    # assertion failures. Needed because numpy array equality uses identity for object arrays.
    """Check that two numpy arrays are equal.

    For arrays with dtype=object, check values recursively to see if a and b
    are equal.  (c.f. `np.array_equal`, which checks dtype=object values using
    object identity.)

    Args:
      a: A numpy array.
      b: A numpy array.
      msg: Message to display if a != b.
    """
    if isinstance(a, np.ndarray) and a.dtype == object:
      self.assertEqual(a.dtype, b.dtype, msg)
      self.assertEqual(a.shape, b.shape, msg)
      self.assertLen(a, len(b), msg)
      for a_val, b_val in zip(a, b):
        self.assertNumpyObjectTensorsRecursivelyEqual(a_val, b_val, msg)
    else:
      self.assertAllEqual(a, b, msg)

  @parameterized.named_parameters([
      ('Shape_2_R',
       [[1, 2], [3, 4, 5]],
       np.array([int32array([1, 2]), int32array([3, 4, 5])], dtype=object)),
      ('Shape_2_2',
       [[1, 2], [3, 4]],
       np.array([[1, 2], [3, 4]])),
      ('Shape_2_R_2',
       [[[1, 2], [3, 4]], [[5, 6]]],
       np.array([int32array([[1, 2], [3, 4]]), int32array([[5, 6]])],
                dtype=object)),
      ('Shape_3_2_R',
       [[[1], []], [[2, 3], [4]], [[], [5, 6, 7]]],
       np.array([[int32array([1]), int32array([])],
                 [int32array([2, 3]), int32array([4])],
                 [int32array([]), int32array([5, 6, 7])]], dtype=object)),
      ('Shape_0_R',
       ragged_factory_ops.constant_value([], ragged_rank=1, dtype=np.int32),
       np.zeros([0, 0], dtype=np.int32)),
      ('Shape_0_R_2',
       ragged_factory_ops.constant_value([], ragged_rank=1,
                                         inner_shape=(2,), dtype=np.int32),
       np.zeros([0, 0, 2], dtype=np.int32)),
  ])  # pyformat: disable
  def testRaggedTensorNumpy(self, rt, expected):
    # [FUNCTION DOC]
    # Parameterized over ragged lists, dense lists and ragged tensor values with expected numpy arrays
    # (object arrays of row arrays when rows differ, dense arrays when uniform, empty arrays). Converts
    # input to a ragged tensor. In eager mode asserts the numpy conversion matches the expectation using
    # the recursive comparison helper; in graph mode asserts a value error saying numpy is only
    # supported in eager mode.
    if isinstance(rt, list):
      rt = ragged_factory_ops.constant(rt, dtype=dtypes.int32)
    else:
      rt = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt)
    if context.executing_eagerly():
      actual = rt.numpy()
      self.assertNumpyObjectTensorsRecursivelyEqual(
          expected, actual, 'Expected %r, got %r' % (expected, actual))
    else:
      with self.assertRaisesRegex(ValueError, 'only supported in eager mode'):
        rt.numpy()

  @parameterized.parameters([
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, None),
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, [None, None, None]),
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, [2, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, None),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [None, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [2, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [2, None, 3]),
      ([[[1, 2, 3]]], 1, [1, 1, None]),
      ([[[1, 2, 3]]], 1, [1, 1, 3]),
  ])
  def testRaggedTensorSetShape(self, rt, rt_ragged_rank, shape):
    # [FUNCTION DOC]
    # Parameterized over nested integer lists, a ragged rank and a shape (or nothing). Builds a ragged
    # tensor, sets its static shape, asserts the shape is compatible, and when a shape was given that the
    # rank is known and every non-unknown dimension equals the requested size.
    rt1 = ragged_factory_ops.constant(rt, ragged_rank=rt_ragged_rank)
    rt1._set_shape(shape)
    rt1.shape.assert_is_compatible_with(shape)
    if shape is not None:
      self.assertIsNot(rt1.shape.rank, None)
      for a, b in zip(rt1.shape, shape):
        if b is not None:
          self.assertEqual(a, b)

  @parameterized.parameters([
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, None),
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, [None, None, None]),
      ([[[1, 2], [3, 4, 5]], [[6]]], 2, [2, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, None),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [None, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [2, None, None]),
      ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]], 1, [2, None, 3]),
      ([[[1, 2, 3]]], 1, [1, 1, None]),
      ([[[1, 2, 3]]], 1, [1, 1, 3]),
  ])
  def testRaggedTensorSetShapeWithPlaceholders(self, rt, rt_ragged_rank, shape):
    # [FUNCTION DOC]
    # Same cases as the set shape test, but the components are replaced with unknown-shape placeholders
    # first. Applies the shape and asserts compatibility, known rank and matching known dimensions.
    rt2 = nest.map_structure(
        lambda x: array_ops.placeholder_with_default(x, None),
        ragged_factory_ops.constant(rt, ragged_rank=rt_ragged_rank),
        expand_composites=True)
    rt2._set_shape(shape)
    rt2.shape.assert_is_compatible_with(shape)
    if shape is not None:
      self.assertIsNot(rt2.shape.rank, None)
      for a, b in zip(rt2.shape, shape):
        if b is not None:
          self.assertEqual(a, b)

  def testRaggedTensorSetShapeUniformRowLength(self):
    # [FUNCTION DOC]
    # Builds a ragged tensor with ragged rank 1 from a dense 2 by 3 by 1 input and sets shape 2, 3, 1;
    # then converts its components to unknown-shape placeholders and sets the same shape again.
    # Only checks that no error is raised.
    rt = [[[1], [2], [3]], [[4], [5], [6]]]

    rt1 = RaggedTensor.from_tensor(rt, ragged_rank=1)
    rt1._set_shape([2, 3, 1])

    rt2 = nest.map_structure(
        lambda x: array_ops.placeholder_with_default(x, None),
        rt1,
        expand_composites=True)
    rt2._set_shape([2, 3, 1])

  def testRaggedTensorSetShapeInconsistentShapeError(self):
    # [FUNCTION DOC]
    # Builds a tensor from a dense 2 by 3 by 1 input with ragged rank 1, checks the shape is 2, 3, 1, and
    # asserts that setting an inconsistent shape raises a value error when the last dimension is 5, with
    # the message about an inconsistent size when the middle dimension is 5, and when the first is 5.
    rt = RaggedTensor.from_tensor([[[1], [2], [3]], [[4], [5], [6]]],
                                  ragged_rank=1)
    self.assertEqual(rt.shape.as_list(), [2, 3, 1])
    with self.assertRaises(ValueError):
      rt._set_shape([None, None, 5])
    with self.assertRaisesRegex(ValueError, 'Inconsistent size'):
      rt._set_shape([None, 5, None])
    with self.assertRaises(ValueError):
      rt._set_shape([5, None, None])


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorSpecTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Test case for RaggedTensorSpec, the type specification of ragged tensors: construction,
  # value type, serialization, component specs, conversion to and from components and tensor lists
  # (plain and batched), flat tensor specs and full types, batch, unbatch and compatibility. Runs in
  # both graph and eager modes; uses parameterized tests; provides one small assertion helper.

  def assertAllTensorsEqual(self, list1, list2):
    # [FUNCTION DOC]
    # HELPER: shared assertion for component list comparison in the spec tests.
    # Asserts two lists have the same length and that each pair of corresponding entries is equal.
    # Inputs are two lists of tensors or values; returns nothing; raises assertion failures.
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    # [FUNCTION DOC]
    # Tests the spec constructor. A spec with ragged rank 1 has unknown shape rank, default dtype
    # float32, 64-bit splits and ragged rank 1 (checked through private and public attributes). A spec with
    # shape of three unknown dimensions infers ragged rank 2. Errors: missing both shape and ragged rank
    # raises a value error; a tensor as the ragged rank raises a type error requiring an integer; ragged
    # rank 2 with a shape of rank 2 raises a value error saying it must be less than the rank.
    spec1 = RaggedTensorSpec(ragged_rank=1)
    self.assertIsNone(spec1._shape.rank)
    self.assertEqual(spec1._dtype, dtypes.float32)
    self.assertEqual(spec1._row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1._ragged_rank, 1)

    self.assertIsNone(spec1.shape.rank)
    self.assertEqual(spec1.dtype, dtypes.float32)
    self.assertEqual(spec1.row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1.ragged_rank, 1)

    spec2 = RaggedTensorSpec(shape=[None, None, None])
    self.assertEqual(spec2._shape.as_list(), [None, None, None])
    self.assertEqual(spec2._dtype, dtypes.float32)
    self.assertEqual(spec2._row_splits_dtype, dtypes.int64)
    self.assertEqual(spec2._ragged_rank, 2)

    with self.assertRaisesRegex(ValueError, 'Must specify ragged_rank'):
      RaggedTensorSpec()
    with self.assertRaisesRegex(TypeError, '`ragged_rank` must be an int'):
      RaggedTensorSpec(ragged_rank=constant_op.constant(1))
    with self.assertRaisesRegex(
        ValueError,
        r'Argument `ragged_rank` \(2\) must be less than rank \(2\).'):
      RaggedTensorSpec(ragged_rank=2, shape=[None, None])

  def testValueType(self):
    # [FUNCTION DOC]
    # Asserts that the value type is the RaggedTensor class when ragged rank is 1 and the ordinary
    # Tensor class when ragged rank is 0.
    spec1 = RaggedTensorSpec(ragged_rank=1)
    self.assertEqual(spec1.value_type, RaggedTensor)
    spec2 = RaggedTensorSpec(ragged_rank=0)
    self.assertEqual(spec2.value_type, tensor_lib.Tensor)

  @parameterized.parameters([
      (RaggedTensorSpec(ragged_rank=1),
       (tensor_shape.TensorShape(None), dtypes.float32, 1, dtypes.int64)),
      (RaggedTensorSpec(shape=[5, None, None]),
       (tensor_shape.TensorShape([5, None, None]), dtypes.float32,
        2, dtypes.int64)),
      (RaggedTensorSpec(shape=[5, None, None], dtype=dtypes.int32),
       (tensor_shape.TensorShape([5, None, None]), dtypes.int32, 2,
        dtypes.int64)),
      (RaggedTensorSpec(ragged_rank=1, row_splits_dtype=dtypes.int32),
       (tensor_shape.TensorShape(None), dtypes.float32, 1, dtypes.int32)),
  ])  # pyformat: disable
  def testSerialize(self, rt_spec, expected):
    # [FUNCTION DOC]
    # Parameterized over four specs (different shapes, dtypes, ragged ranks and splits dtypes).
    # Serializes each spec and compares text representations with the expected tuple of shape, dtype,
    # ragged rank and splits dtype (text comparison because shape equality is unconventional).
    serialization = rt_spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      (RaggedTensorSpec(ragged_rank=0, shape=[5, 3]), [
          tensor_lib.TensorSpec([5, 3], dtypes.float32),
      ]),
      (RaggedTensorSpec(ragged_rank=1), [
          tensor_lib.TensorSpec(None, dtypes.float32),
          tensor_lib.TensorSpec([None], dtypes.int64)
      ]),
      (RaggedTensorSpec(ragged_rank=1, row_splits_dtype=dtypes.int32), [
          tensor_lib.TensorSpec(None, dtypes.float32),
          tensor_lib.TensorSpec([None], dtypes.int32),
      ]),
      (RaggedTensorSpec(ragged_rank=2), [
          tensor_lib.TensorSpec(None, dtypes.float32),
          tensor_lib.TensorSpec([None], dtypes.int64),
          tensor_lib.TensorSpec([None], dtypes.int64),
      ]),
      (RaggedTensorSpec(shape=[5, None, None], dtype=dtypes.string), [
          tensor_lib.TensorSpec([None], dtypes.string),
          tensor_lib.TensorSpec([6], dtypes.int64),
          tensor_lib.TensorSpec([None], dtypes.int64),
      ]),
  ])
  def testComponentSpecs(self, rt_spec, expected):
    # [FUNCTION DOC]
    # Parameterized over five specs. Asserts the component specifications (a values tensor spec followed
    # by one splits tensor spec per ragged level, with shapes derived from static information such as a
    # row count of 5 giving 6 splits) equal the expected list.
    self.assertEqual(rt_spec._component_specs, expected)

  @parameterized.parameters([
      {
          'rt_spec': RaggedTensorSpec(ragged_rank=0),
          'rt': [1.0, 2.0, 3.0],
          'components': [[1.0, 2.0, 3.0]]
      },
      {
          'rt_spec': RaggedTensorSpec(ragged_rank=1),
          'rt': [[1.0, 2.0], [3.0]],
          'components': [[1.0, 2.0, 3.0], [0, 2, 3]]
      },
      {
          'rt_spec': RaggedTensorSpec(shape=[2, None, None]),
          'rt': [[[1.0, 2.0], [3.0]], [[], [4.0]]],
          'components': [[1.0, 2.0, 3.0, 4.0], [0, 2, 4], [0, 2, 3, 3, 4]]
      },
  ])
  def testToFromComponents(self, rt_spec, rt, components):
    # [FUNCTION DOC]
    # Parameterized over ragged ranks 0, 1 and 2. Converts a ragged constant into components with the
    # spec, asserts the component list matches the expected values and splits, rebuilds the tensor
    # from the components and asserts equality with the original.
    rt = ragged_factory_ops.constant(rt)
    actual_components = rt_spec._to_components(rt)
    self.assertAllTensorsEqual(actual_components, components)
    rt_reconstructed = rt_spec._from_components(actual_components)
    self.assertAllEqual(rt, rt_reconstructed)

  @parameterized.parameters([
      {
          'flat_value_spec': tensor_lib.TensorSpec(None, dtypes.float32),
          'row_splits_spec': tensor_lib.TensorSpec(None, dtypes.int64),
      },
      {
          'flat_value_spec': tensor_lib.TensorSpec([None,], dtypes.float32),
          'row_splits_spec': tensor_lib.TensorSpec(None, dtypes.int64),
      },
      {
          'flat_value_spec': tensor_lib.TensorSpec(None, dtypes.float32),
          'row_splits_spec': tensor_lib.TensorSpec([None,], dtypes.int64),
      },
      {
          'flat_value_spec': tensor_lib.TensorSpec([None,], dtypes.float32),
          'row_splits_spec': tensor_lib.TensorSpec([None,], dtypes.int64),
      },
      {
          'flat_value_spec': tensor_lib.TensorSpec([4,], dtypes.float32),
          'row_splits_spec': tensor_lib.TensorSpec(None, dtypes.int64),
      },
      {
          'flat_value_spec': tensor_lib.TensorSpec(None, dtypes.float32),
          'row_splits_spec': tensor_lib.TensorSpec([3,], dtypes.int64),
      },
  ])
  def testToFromComponentsStaticUnknownShape(self, flat_value_spec,
                                             row_splits_spec):
    # [FUNCTION DOC]
    # Parameterized over combinations of known and unknown shapes for the flat values and the row splits.
    # Defines a compiled function with that input signature that rebuilds a ragged tensor from the
    # components using a spec with shape 2 by unknown, asserts inside that the static shape is 2 by
    # unknown, and adds a constant ragged tensor. Calls it on concrete inputs and asserts the sum.
    rt_spec = RaggedTensorSpec(shape=[2, None], ragged_rank=1)
    tester = self

    @def_function.function(input_signature=[flat_value_spec, row_splits_spec])
    def test_fn(flat_value, row_splits):
      # [FUNCTION DOC]
      # Compiled function (with a fixed input signature) under test: rebuilds a ragged tensor from flat value and
      # row splits using the enclosing spec, asserts through the enclosing test case that the static shape
      # is 2 by unknown, and returns the sum of the tensor with a constant ragged tensor of ones.
      # Apply static shape information saved in rt_spec to rt.
      rt = rt_spec._from_components([flat_value, row_splits])
      tester.assertEqual(rt.shape.as_list(), [2, None])
      return rt + ragged_factory_ops.constant([[1.0, 1.0, 1.0], [1.0]])

    result = test_fn([1.0, 2.0, 3.0, 4.0], [0, 3, 4])
    expected_result = ragged_factory_ops.constant([[2.0, 3.0, 4.0], [5.0]])
    self.assertAllEqual(result, expected_result)

  @test_util.run_v1_only('RaggedTensorValue is deprecated in v2')
  def testFromNumpyComponents(self):
    # [FUNCTION DOC]
    # Only runs with the graph-mode (version 1) API because RaggedTensorValue is deprecated in version 2.
    # Rebuilds from numpy components: ragged rank 1 and 2 give ragged tensor values with the right
    # contents; ragged rank 0 gives a plain numpy array with the values.
    spec1 = RaggedTensorSpec(ragged_rank=1, dtype=dtypes.int32)
    rt1 = spec1._from_components([np.array([1, 2, 3]), np.array([0, 2, 3])])
    self.assertIsInstance(rt1, ragged_tensor_value.RaggedTensorValue)
    self.assertAllEqual(rt1, [[1, 2], [3]])

    spec2 = RaggedTensorSpec(ragged_rank=2, dtype=dtypes.int32)
    rt2 = spec2._from_components(
        [np.array([1, 2, 3]),
         np.array([0, 2, 3]),
         np.array([0, 0, 2, 3])])
    self.assertIsInstance(rt2, ragged_tensor_value.RaggedTensorValue)
    self.assertAllEqual(rt2, [[[], [1, 2]], [[3]]])

    spec3 = RaggedTensorSpec(ragged_rank=0, dtype=dtypes.int32)
    rt3 = spec3._from_components([np.array([1, 2, 3])])
    self.assertIsInstance(rt3, np.ndarray)
    self.assertAllEqual(rt3, [1, 2, 3])

  @parameterized.parameters([
      RaggedTensorSpec(ragged_rank=0, shape=[5, 3]),
      RaggedTensorSpec(ragged_rank=1),
      RaggedTensorSpec(ragged_rank=1, row_splits_dtype=dtypes.int32),
      RaggedTensorSpec(ragged_rank=2, dtype=dtypes.string),
      RaggedTensorSpec(shape=[5, None, None]),
  ])
  def testFlatTensorSpecs(self, rt_spec):
    # [FUNCTION DOC]
    # Parameterized over five specs. Asserts the flat tensor specs consist of exactly one variant tensor
    # spec of unknown shape.
    self.assertEqual(rt_spec._flat_tensor_specs,
                     [tensor_lib.TensorSpec(None, dtypes.variant)])

  @parameterized.parameters([
      (dtypes.float32, full_type_pb2.TFT_FLOAT),
      (dtypes.string, full_type_pb2.TFT_STRING),
  ])
  def testFullTypesForFlatTensors(self, dt, ft):
    # [FUNCTION DOC]
    # Parameterized over float32 and string dtypes with the matching full type id. Builds a ragged rank 2
    # spec, computes the full types for its flat tensors and asserts there is one per flat tensor spec and
    # that it is the ragged full type with a single argument having the element type id.
    rt_spec = RaggedTensorSpec(ragged_rank=2, dtype=dt)
    full_type_list = fulltypes_for_flat_tensors(rt_spec)
    expect = [
        full_type_pb2.FullTypeDef(
            type_id=full_type_pb2.TFT_RAGGED,
            args=[full_type_pb2.FullTypeDef(type_id=ft)])
    ]
    self.assertEqual(len(rt_spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)

  @parameterized.named_parameters([
      {
          'testcase_name': 'RaggedRank0',
          'rt_spec': RaggedTensorSpec(ragged_rank=0),
          'rt': [1.0, 2.0, 3.0],
      },
      {
          'testcase_name': 'RaggedRank1',
          'rt_spec': RaggedTensorSpec(ragged_rank=1),
          'rt': [[1.0, 2.0], [3.0]]
      },
      {
          'testcase_name': 'RaggedRank2',
          'rt_spec': RaggedTensorSpec(shape=[2, None, None]),
          'rt': [[[1.0, 2.0], [3.0]], [[], [4.0]]]
      },
  ])
  def testToFromTensorList(self, rt_spec, rt):
    # [FUNCTION DOC]
    # Parameterized over ragged ranks 0, 1 and 2. Converts a ragged constant to a tensor list and back
    # using the spec and asserts equality with the original.
    rt = ragged_factory_ops.constant(rt)
    tensor_list = rt_spec._to_tensor_list(rt)
    rt_reconstructed = rt_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(rt, rt_reconstructed)

  @parameterized.named_parameters([
      # TODO(b/141789000) Test ragged_rank=0 when support is added.
      {
          'testcase_name': 'RaggedRank1',
          'rt_spec': RaggedTensorSpec(ragged_rank=1),
          'rt': [[1.0, 2.0], [3.0]]
      },
      {
          'testcase_name': 'RaggedRank2',
          'rt_spec': RaggedTensorSpec(shape=[2, None, None]),
          'rt': [[[1.0, 2.0], [3.0]], [[], [4.0]]]
      },
  ])
  def testToFromBatchedTensorList(self, rt_spec, rt):
    # [FUNCTION DOC]
    # Parameterized over ragged ranks 1 and 2 (ragged rank 0 is a TODO tied to bug 141789000). Converts
    # to a batched tensor list and back, asserts equality, then takes the first element of every
    # component, rebuilds a tensor with the unbatched spec and asserts it equals the first row.
    rt = ragged_factory_ops.constant(rt)
    tensor_list = rt_spec._to_batched_tensor_list(rt)
    rt_reconstructed = rt_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(rt, rt_reconstructed)
    first_row = rt_spec._unbatch()._from_tensor_list(
        [t[0] for t in tensor_list])
    self.assertAllEqual(rt[0], first_row)

  def testToFromBatchedTensorListPreservesUniformRowLengths(self):
    # [FUNCTION DOC]
    # Builds a ragged tensor from a dense 3 by 4 by 5 tensor with ragged rank 2, round trips through a
    # batched tensor list and asserts equality, that both shapes are fully defined and that the shapes match.
    rt = RaggedTensor.from_tensor(array_ops.zeros([3, 4, 5]), ragged_rank=2)
    rt_spec = rt._type_spec
    tensor_list = rt_spec._to_batched_tensor_list(rt)
    rt_reconstructed = rt_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(rt, rt_reconstructed)
    self.assertTrue(rt.shape.is_fully_defined())
    self.assertTrue(rt_reconstructed.shape.is_fully_defined())
    self.assertEqual(rt.shape.as_list(), rt_reconstructed.shape.as_list())

  @parameterized.parameters([
      (RaggedTensorSpec([2, None], dtypes.float32, 1), 32,
       RaggedTensorSpec([32, 2, None], dtypes.float32, 2)),
      (RaggedTensorSpec([4, None], dtypes.float32, 1), None,
       RaggedTensorSpec([None, 4, None], dtypes.float32, 2)),
      (RaggedTensorSpec([2], dtypes.float32,
                        -1), 32, RaggedTensorSpec([32, 2], dtypes.float32, 0)),
  ])
  def testBatch(self, spec, batch_size, expected):
    # [FUNCTION DOC]
    # Parameterized over three specs with batch size 32 (twice) and an unknown batch size; the last spec
    # has ragged rank minus one. Asserts the batched spec equals the expected spec with the batch
    # dimension added in front and the ragged rank increased by one.
    self.assertEqual(spec._batch(batch_size), expected)

  @parameterized.parameters([
      (RaggedTensorSpec([32, None, None], dtypes.float32, 2),
       RaggedTensorSpec([None, None], dtypes.float32, 1)),
      (RaggedTensorSpec([None, None, None], dtypes.float32, 2),
       RaggedTensorSpec([None, None], dtypes.float32, 1)),
      (RaggedTensorSpec([32, 2], dtypes.float32, 0),
       RaggedTensorSpec([2], dtypes.float32, -1)),
      (RaggedTensorSpec([32, None, 4], dtypes.float32, 1, dtypes.int32),
       RaggedTensorSpec([None, 4], dtypes.float32, 0, dtypes.int32)),
  ])  # pyformat: disable
  def testUnbatch(self, spec, expected):
    # [FUNCTION DOC]
    # Parameterized over four specs. Asserts unbatching removes the leading dimension and reduces the ragged
    # rank by one (minus one for a ragged rank 0 spec), keeping the splits dtype.
    self.assertEqual(spec._unbatch(), expected)

  def testIsCompatibleWith(self):
    # [FUNCTION DOC]
    # Builds four specs differing in shape, dtype and ragged rank and asserts that only the first is
    # compatible with the second, that all other pairs are incompatible, and that the fourth (plain tensor
    # spec) is compatible with a concrete dense tensor.
    spec1 = RaggedTensorSpec([32, None, None], dtypes.float32, 2)
    spec2 = RaggedTensorSpec(None, dtypes.float32, 2)
    spec3 = RaggedTensorSpec(None, dtypes.int32, 1)
    spec4 = RaggedTensorSpec([None], dtypes.int32, 0)

    self.assertTrue(spec1.is_compatible_with(spec2))
    self.assertFalse(spec1.is_compatible_with(spec3))
    self.assertFalse(spec1.is_compatible_with(spec4))
    self.assertFalse(spec2.is_compatible_with(spec3))
    self.assertFalse(spec2.is_compatible_with(spec4))
    self.assertFalse(spec3.is_compatible_with(spec4))
    self.assertTrue(spec4.is_compatible_with(constant_op.constant([1, 2, 3])))


if __name__ == '__main__':
  googletest.main()
