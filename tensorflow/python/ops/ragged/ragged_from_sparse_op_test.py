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
# [MODULE DOC]
# Test module for RaggedTensor.from_sparse: conversion of rank-2 sparse tensors to ragged
# tensors, empty input, rank validation, partially unknown shapes, and rejection of sparse
# tensors that are not right-ragged. Runs in graph and eager modes. No import-time side effects.
"""Tests for RaggedTensor.from_sparse."""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToSparseOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the name suggests ragged-to-sparse conversion, but the tests cover the opposite
  # direction, RaggedTensor.from_sparse.
  # Test case for converting sparse tensors to ragged tensors, run in graph and eager modes.
  # No shared fixtures.

  def testDocStringExample(self):
    # [FUNCTION DOC]
    # Converts the documented 4 by 3 sparse example and asserts the result equals rows
    # [1,2,3], [4], [], [5]. Returns nothing.
    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [0, 2], [1, 0], [3, 0]],
        values=[1, 2, 3, 4, 5],
        dense_shape=[4, 3])
    rt = RaggedTensor.from_sparse(st)

    self.assertAllEqual(rt, [[1, 2, 3], [4], [], [5]])

  def testEmpty(self):
    # [FUNCTION DOC]
    # Converts a sparse tensor with no entries and dense shape 4 by 3 and asserts the result is
    # four empty rows. Returns nothing.
    st = sparse_tensor.SparseTensor(
        indices=array_ops.zeros([0, 2], dtype=dtypes.int64),
        values=[],
        dense_shape=[4, 3])
    rt = RaggedTensor.from_sparse(st)

    self.assertAllEqual(rt, [[], [], [], []])

  def testBadSparseTensorRank(self):
    # [FUNCTION DOC]
    # Asserts from_sparse raises a value error saying the input rank must be 2 for a rank-1
    # sparse tensor and a rank-3 one, and (graph mode only) for a sparse tensor whose indices and
    # dense shape are placeholders of unknown shape. Returns nothing.
    st1 = sparse_tensor.SparseTensor(indices=[[0]], values=[0], dense_shape=[3])
    self.assertRaisesRegex(ValueError, r'rank\(st_input\) must be 2',
                           RaggedTensor.from_sparse, st1)

    st2 = sparse_tensor.SparseTensor(
        indices=[[0, 0, 0]], values=[0], dense_shape=[3, 3, 3])
    self.assertRaisesRegex(ValueError, r'rank\(st_input\) must be 2',
                           RaggedTensor.from_sparse, st2)

    if not context.executing_eagerly():
      st3 = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=[0],
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertRaisesRegex(ValueError, r'rank\(st_input\) must be 2',
                             RaggedTensor.from_sparse, st3)

  def testGoodPartialSparseTensorRank(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in eager mode the body is skipped entirely, so nothing is tested there.
    # Graph-mode only. Builds sparse tensors where either the dense shape or the indices are
    # placeholders (rank only partly known) and checks from_sparse accepts them without raising.
    # Returns nothing.
    if not context.executing_eagerly():
      st1 = sparse_tensor.SparseTensor(
          indices=[[0, 0]],
          values=[0],
          dense_shape=array_ops.placeholder(dtypes.int64))
      st2 = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=[0],
          dense_shape=[4, 3])

      # Shouldn't throw ValueError
      RaggedTensor.from_sparse(st1)
      RaggedTensor.from_sparse(st2)

  def testNonRaggedSparseTensor(self):
    # [FUNCTION DOC]
    # Three sparse tensors violate the right-ragged requirement: first index does not start at
    # column zero, a new row does not start at column zero, and an index skips a cell within a
    # row. For each, evaluating the conversion must raise an invalid-argument error stating the
    # sparse tensor is not right-ragged. Returns nothing.
    # "index_suffix" means the value of the innermost dimension of the index
    # (i.e., indices[i][-1]).
    # See comments in _assert_sparse_indices_are_ragged_right() for more
    # details/background.

    # index_suffix of first index is not zero.
    st1 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [0, 2], [2, 0]], values=[1, 2, 3], dense_shape=[3, 3])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'.*SparseTensor is not right-ragged'):
      self.evaluate(RaggedTensor.from_sparse(st1))
    # index_suffix of an index that starts a new row is not zero.
    st2 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [2, 1]], values=[1, 2, 3], dense_shape=[3, 3])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'.*SparseTensor is not right-ragged'):
      self.evaluate(RaggedTensor.from_sparse(st2))
    # index_suffix of an index that continues a row skips a cell.
    st3 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [0, 1], [0, 3]], values=[1, 2, 3], dense_shape=[3, 3])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'.*SparseTensor is not right-ragged'):
      self.evaluate(RaggedTensor.from_sparse(st3))


if __name__ == '__main__':
  googletest.main()
