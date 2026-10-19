# [MODULE DOC]
# Tests for the diagonal-related array operations: matrix_diag, matrix_set_diag, matrix_diag_part
# (including the banded "V2" variants with a diagonal range, padding value and alignment option) and
# the general-rank diag / diag_part operations, plus their gradients.
# The module defines shared numeric fixtures (square, tall and fat matrices with expected packed and
# padded diagonals) and helpers that re-pack diagonals for the four alignment modes, followed by test
# cases for each operation. Nothing is registered at import time; the file runs the TensorFlow test
# runner when executed as a script.
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

import itertools

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


default_v2_alignment = "LEFT_LEFT"
alignment_list = ["RIGHT_LEFT", "LEFT_RIGHT", "LEFT_LEFT", "RIGHT_RIGHT"]


def zip_to_first_list_length(a, b):
  # [FUNCTION DOC]
  # HELPER: small utility for pairing padding values with alignment names in the tests below.
  # Pairs each element of the first list with the element at the same position of the second list.
  # If the second list is longer it is truncated to the first list's length; if shorter, it is padded
  # with nothing/null values. Takes two lists, returns a lazy sequence of pairs; no side effects.
  if len(b) > len(a):
    return zip(a, b[:len(a)])
  return zip(a, b + [None] * (len(a) - len(b)))


def repack_diagonals(packed_diagonals,
                     diag_index,
                     num_rows,
                     num_cols,
                     align=None):
  # [FUNCTION DOC]
  # HELPER: builds expected inputs/outputs for the alignment variants of the banded diagonal ops.
  # KNOWN ISSUE: the loop variable reuses the name of the diag_index parameter, shadowing it after the
  # lower and upper bounds were already extracted; harmless but misleading.
  # Converts diagonals that are stored in the default left-justified layout into another alignment.
  # Inputs: packed_diagonals (array of diagonals, with optional leading batch dimensions), diag_index
  # (pair of lower and upper diagonal offsets), num_rows and num_cols (size of the matrix the diagonals
  # belong to), align (optional string such as right-left naming the alignment of super and sub
  # diagonals; nothing or the default left-left alignment returns the input unchanged).
  # For every diagonal in the band it decides whether it is left-aligned and, if not, shifts it right by
  # the padding amount, writing into a zero-initialised copy. Returns the re-packed array; does not
  # modify the input.
  # The original test cases are LEFT_LEFT aligned.
  if align == default_v2_alignment or align is None:
    return packed_diagonals

  align = align.split("_")
  d_lower, d_upper = diag_index
  batch_dims = packed_diagonals.ndim - (2 if d_lower < d_upper else 1)
  max_diag_len = packed_diagonals.shape[-1]
  index = (slice(None),) * batch_dims
  repacked_diagonals = np.zeros_like(packed_diagonals)

  # Aligns each diagonal row-by-row.
  for diag_index in range(d_lower, d_upper + 1):
    diag_len = min(num_rows + min(0, diag_index), num_cols - max(0, diag_index))
    row_index = d_upper - diag_index
    padding_len = max_diag_len - diag_len
    left_align = (diag_index >= 0 and
                  align[0] == "LEFT") or (diag_index <= 0 and
                                          align[1] == "LEFT")
    # Prepares index tuples.
    extra_dim = tuple() if d_lower == d_upper else (row_index,)
    packed_last_dim = (slice(None),) if left_align else (slice(0, diag_len, 1),)
    repacked_last_dim = (slice(None),) if left_align else (slice(
        padding_len, max_diag_len, 1),)
    packed_index = index + extra_dim + packed_last_dim
    repacked_index = index + extra_dim + repacked_last_dim

    # Repacks the diagonal.
    repacked_diagonals[repacked_index] = packed_diagonals[packed_index]
  return repacked_diagonals


def repack_diagonals_in_tests(tests, align=None):
  # [FUNCTION DOC]
  # HELPER: applies the diagonal re-packing to a whole dictionary of test cases.
  # Takes tests (mapping from a pair of diagonal offsets to a pair of packed and padded diagonals) and
  # an optional alignment name. With nothing or the default alignment it returns the mapping as is;
  # otherwise it returns a new mapping where every packed-diagonals array is re-packed for the
  # requested alignment using the matrix size taken from the padded array. Padded arrays are unchanged.
  # The original test cases are LEFT_LEFT aligned.
  if align == default_v2_alignment or align is None:
    return tests

  new_tests = dict()
  # Loops through each case.
  for diag_index, (packed_diagonals, padded_diagonals) in tests.items():
    num_rows, num_cols = padded_diagonals.shape[-2:]
    repacked_diagonals = repack_diagonals(
        packed_diagonals, diag_index, num_rows, num_cols, align=align)
    new_tests[diag_index] = (repacked_diagonals, padded_diagonals)

  return new_tests


# Test cases shared by MatrixDiagV2, MatrixDiagPartV2, and MatrixSetDiagV2.
def square_cases(align=None):
  # [FUNCTION DOC]
  # HELPER: shared fixture generator for the MatrixDiag, MatrixDiagPart and MatrixSetDiag tests.
  # Builds a batch of two 5x5 integer matrices and a mapping from diagonal ranges (single sub-diagonal,
  # a lower band, a mixed band and an upper band) to the expected packed diagonals and the expected
  # banded matrices padded with zeros. Optional align argument re-packs the diagonals for that
  # alignment. Returns a pair of the matrix batch and the mapping.
  # pyformat: disable
  mat = np.array([[[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 1],
                   [3, 4, 5, 6, 7],
                   [8, 9, 1, 2, 3],
                   [4, 5, 6, 7, 8]],
                  [[9, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9],
                   [1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 1],
                   [2, 3, 4, 5, 6]]])
  tests = dict()
  # tests[d_lower, d_upper] = (packed_diagonals, padded_diagonals)
  tests[-1, -1] = (np.array([[6, 4, 1, 7],
                             [5, 2, 8, 5]]),
                   np.array([[[0, 0, 0, 0, 0],
                              [6, 0, 0, 0, 0],
                              [0, 4, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 7, 0]],
                             [[0, 0, 0, 0, 0],
                              [5, 0, 0, 0, 0],
                              [0, 2, 0, 0, 0],
                              [0, 0, 8, 0, 0],
                              [0, 0, 0, 5, 0]]]))
  tests[-4, -3] = (np.array([[[8, 5],
                              [4, 0]],
                             [[6, 3],
                              [2, 0]]]),
                   np.array([[[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [8, 0, 0, 0, 0],
                              [4, 5, 0, 0, 0]],
                             [[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [6, 0, 0, 0, 0],
                              [2, 3, 0, 0, 0]]]))
  tests[-2, 1] = (np.array([[[2, 8, 6, 3, 0],
                             [1, 7, 5, 2, 8],
                             [6, 4, 1, 7, 0],
                             [3, 9, 6, 0, 0]],
                            [[1, 7, 4, 1, 0],
                             [9, 6, 3, 9, 6],
                             [5, 2, 8, 5, 0],
                             [1, 7, 4, 0, 0]]]),
                  np.array([[[1, 2, 0, 0, 0],
                             [6, 7, 8, 0, 0],
                             [3, 4, 5, 6, 0],
                             [0, 9, 1, 2, 3],
                             [0, 0, 6, 7, 8]],
                            [[9, 1, 0, 0, 0],
                             [5, 6, 7, 0, 0],
                             [1, 2, 3, 4, 0],
                             [0, 7, 8, 9, 1],
                             [0, 0, 4, 5, 6]]]))
  tests[2, 4] = (np.array([[[5, 0, 0],
                            [4, 1, 0],
                            [3, 9, 7]],
                           [[4, 0, 0],
                            [3, 9, 0],
                            [2, 8, 5]]]),
                 np.array([[[0, 0, 3, 4, 5],
                            [0, 0, 0, 9, 1],
                            [0, 0, 0, 0, 7],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           [[0, 0, 2, 3, 4],
                            [0, 0, 0, 8, 9],
                            [0, 0, 0, 0, 5],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]]))
  # pyformat: enable
  return (mat, repack_diagonals_in_tests(tests, align))


def tall_cases(align=None):
  # [FUNCTION DOC]
  # HELPER: shared fixture generator for the diagonal tests, using a batch of two 5x3 (more rows than
  # columns) matrices.
  # Returns a pair of the matrix batch and a mapping from diagonal ranges (main diagonal, lower bands,
  # a band spanning both sides, and an upper band) to expected packed diagonals and zero-padded banded
  # matrices, optionally re-packed for the given alignment (optional argument).
  # pyformat: disable
  mat = np.array([[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [9, 8, 7],
                   [6, 5, 4]],
                  [[3, 2, 1],
                   [1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [9, 8, 7]]])
  tests = dict()
  tests[0, 0] = (np.array([[1, 5, 9],
                           [3, 2, 6]]),
                 np.array([[[1, 0, 0],
                            [0, 5, 0],
                            [0, 0, 9],
                            [0, 0, 0]],
                           [[3, 0, 0],
                            [0, 2, 0],
                            [0, 0, 6],
                            [0, 0, 0]]]))
  tests[-4, -3] = (np.array([[[9, 5],
                              [6, 0]],
                             [[7, 8],
                              [9, 0]]]),
                   np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [9, 0, 0],
                              [6, 5, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [7, 0, 0],
                              [9, 8, 0]]]))
  tests[-2, -1] = (np.array([[[4, 8, 7],
                              [7, 8, 4]],
                             [[1, 5, 9],
                              [4, 8, 7]]]),
                   np.array([[[0, 0, 0],
                              [4, 0, 0],
                              [7, 8, 0],
                              [0, 8, 7],
                              [0, 0, 4]],
                             [[0, 0, 0],
                              [1, 0, 0],
                              [4, 5, 0],
                              [0, 8, 9],
                              [0, 0, 7]]]))
  tests[-2, 1] = (np.array([[[2, 6, 0],
                             [1, 5, 9],
                             [4, 8, 7],
                             [7, 8, 4]],
                            [[2, 3, 0],
                             [3, 2, 6],
                             [1, 5, 9],
                             [4, 8, 7]]]),
                  np.array([[[1, 2, 0],
                             [4, 5, 6],
                             [7, 8, 9],
                             [0, 8, 7],
                             [0, 0, 4]],
                            [[3, 2, 0],
                             [1, 2, 3],
                             [4, 5, 6],
                             [0, 8, 9],
                             [0, 0, 7]]]))
  tests[1, 2] = (np.array([[[3, 0],
                            [2, 6]],
                           [[1, 0],
                            [2, 3]]]),
                 np.array([[[0, 2, 3],
                            [0, 0, 6],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           [[0, 2, 1],
                            [0, 0, 3],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]]))
  # pyformat: enable
  return (mat, repack_diagonals_in_tests(tests, align))


def fat_cases(align=None):
  # [FUNCTION DOC]
  # HELPER: shared fixture generator for the diagonal tests, using a batch of two 3x4 (more columns
  # than rows) matrices.
  # Returns a pair of the matrix batch and a mapping from diagonal ranges (super-diagonal, lower band,
  # mixed bands) to expected packed diagonals and zero-padded banded matrices, optionally re-packed for
  # the given alignment (optional argument).
  # pyformat: disable
  mat = np.array([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 1, 2, 3]],
                  [[4, 5, 6, 7],
                   [8, 9, 1, 2],
                   [3, 4, 5, 6]]])
  tests = dict()
  tests[2, 2] = (np.array([[3, 8],
                           [6, 2]]),
                 np.array([[[0, 0, 3, 0],
                            [0, 0, 0, 8],
                            [0, 0, 0, 0]],
                           [[0, 0, 6, 0],
                            [0, 0, 0, 2],
                            [0, 0, 0, 0]]]))
  tests[-2, 0] = (np.array([[[1, 6, 2],
                             [5, 1, 0],
                             [9, 0, 0]],
                            [[4, 9, 5],
                             [8, 4, 0],
                             [3, 0, 0]]]),
                  np.array([[[1, 0, 0, 0],
                             [5, 6, 0, 0],
                             [9, 1, 2, 0]],
                            [[4, 0, 0, 0],
                             [8, 9, 0, 0],
                             [3, 4, 5, 0]]]))
  tests[-1, 1] = (np.array([[[2, 7, 3],
                             [1, 6, 2],
                             [5, 1, 0]],
                            [[5, 1, 6],
                             [4, 9, 5],
                             [8, 4, 0]]]),
                  np.array([[[1, 2, 0, 0],
                             [5, 6, 7, 0],
                             [0, 1, 2, 3]],
                            [[4, 5, 0, 0],
                             [8, 9, 1, 0],
                             [0, 4, 5, 6]]]))
  tests[0, 3] = (np.array([[[4, 0, 0],
                            [3, 8, 0],
                            [2, 7, 3],
                            [1, 6, 2]],
                           [[7, 0, 0],
                            [6, 2, 0],
                            [5, 1, 6],
                            [4, 9, 5]]]),
                 np.array([[[1, 2, 3, 4],
                            [0, 6, 7, 8],
                            [0, 0, 2, 3]],
                           [[4, 5, 6, 7],
                            [0, 9, 1, 2],
                            [0, 0, 5, 6]]]))
  # pyformat: enable
  return (mat, repack_diagonals_in_tests(tests, align))


def all_tests(align=None):
  # [FUNCTION DOC]
  # HELPER: convenience aggregator of the square, tall and fat fixtures.
  # Returns a list with the result of the three case generators for the given optional alignment.
  # Not referenced by tests in this file.
  return [square_cases(align), tall_cases(align), fat_cases(align)]


class MatrixDiagTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the matrix_diag operation, which builds (batches of) matrices from diagonal vectors, with
  # optional diagonal offsets or bands, padding value, alignment and explicit row/column counts.
  # Uses the shared square/tall/fat fixtures and a class-specific extra set of banded cases.

  def _moreCases(self, align=None):
    # [FUNCTION DOC]
    # HELPER: extra fixture generator for the matrix_diag tests.
    # Takes an optional alignment. Builds a batch of two sets of three diagonals (length four) and the
    # expected zero-padded banded matrices for three diagonal ranges (lower band, centered band, upper
    # band). Returns a pair whose first element is nothing/null (there is no source matrix) and whose
    # second is the mapping of diagonal ranges to (diagonals, expected matrices), re-packed for alignment.
    # Diagonal bands.
    # pyformat: disable
    vecs = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
                      [5, 6, 7, 8],
                      [9, 8, 7, 6]],
                     [[5, 4, 3, 2],
                      [1, 2, 3, 4],
                      [5, 6, 7, 8]]])
    tests = dict()
    tests[-3, -1] = (vecs,
                     np.array([[[0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0],
                                [5, 2, 0, 0, 0],
                                [9, 6, 3, 0, 0],
                                [0, 8, 7, 4, 0]],
                               [[0, 0, 0, 0, 0],
                                [5, 0, 0, 0, 0],
                                [1, 4, 0, 0, 0],
                                [5, 2, 3, 0, 0],
                                [0, 6, 3, 2, 0]]]))
    tests[-1, 1] = (vecs,
                    np.array([[[5, 1, 0, 0],
                               [9, 6, 2, 0],
                               [0, 8, 7, 3],
                               [0, 0, 7, 8]],
                              [[1, 5, 0, 0],
                               [5, 2, 4, 0],
                               [0, 6, 3, 3],
                               [0, 0, 7, 4]]]))
    tests[2, 4] = (vecs,
                   np.array([[[0, 0, 9, 5, 1, 0],
                              [0, 0, 0, 8, 6, 2],
                              [0, 0, 0, 0, 7, 7],
                              [0, 0, 0, 0, 0, 6],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 5, 1, 5, 0],
                              [0, 0, 0, 6, 2, 4],
                              [0, 0, 0, 0, 7, 3],
                              [0, 0, 0, 0, 0, 8],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]]))
    # pyformat: enable
    return (None, repack_diagonals_in_tests(tests, align))

  @test_util.run_deprecated_v1
  def testVector(self):
    # [FUNCTION DOC]
    # Checks matrix_diag on an unbatched vector (graph mode, deprecated v1 style). Verifies the static
    # shape and value of the plain diagonal matrix against numpy, then for several single offsets
    # (super and sub diagonals), then for banded cases in every alignment mode using the first batch
    # entry of the square and extra fixtures. Expects shape and value equality.
    with self.session():
      v = np.array([1.0, 2.0, 3.0])
      mat = np.diag(v)
      v_diag = array_ops.matrix_diag(v)
      self.assertEqual((3, 3), v_diag.get_shape())
      self.assertAllEqual(v_diag, mat)

      # {Sub,Super}diagonals.
      for offset in [1, -2, 5]:
        mat = np.diag(v, offset)
        v_diag = array_ops.matrix_diag(v, k=offset)
        self.assertEqual(mat.shape, v_diag.get_shape())
        self.assertAllEqual(v_diag, mat)

      # Diagonal bands.
      for align in alignment_list:
        for _, tests in [self._moreCases(align), square_cases(align)]:
          for diags, (vecs, solution) in tests.items():
            v_diags = array_ops.matrix_diag(vecs[0], k=diags, align=align)
            self.assertEqual(v_diags.get_shape(), solution[0].shape)
            self.assertAllEqual(v_diags, solution[0])

  def _testVectorBatch(self, dtype):
    # [FUNCTION DOC]
    # HELPER: dtype-parametrised body of the batch-vector test.
    # Given a numpy dtype, builds a batch of two vectors, checks matrix_diag gives the expected batch of
    # diagonal matrices (shape and values), then checks several offsets against stacked numpy results,
    # then banded cases with different padding values (0, 555, -11) paired with the alignment modes,
    # comparing against the fixture solutions where zeros are replaced by the padding value.
    with self.cached_session():
      v_batch = np.array([[1.0, 0.0, 3.0], [4.0, 5.0, 6.0]]).astype(dtype)
      mat_batch = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
                            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0],
                             [0.0, 0.0, 6.0]]]).astype(dtype)
      v_batch_diag = array_ops.matrix_diag(v_batch)
      self.assertEqual((2, 3, 3), v_batch_diag.get_shape())
      self.assertAllEqual(v_batch_diag, mat_batch)

      # {Sub,Super}diagonals.
      for offset in [1, -2, 5]:
        v_batch_diag = array_ops.matrix_diag(v_batch, k=offset)
        mats = [
            np.diag(v_batch[i], offset) for i in range(0, v_batch.shape[0])
        ]
        mat_batch = np.stack(mats, axis=0)
        self.assertEqual(mat_batch.shape, v_batch_diag.get_shape())
        self.assertAllEqual(v_batch_diag, mat_batch)

      # Diagonal bands with padding_value.
      for padding_value, align in zip_to_first_list_length([0, 555, -11],
                                                           alignment_list):
        for _, tests in [self._moreCases(align), square_cases(align)]:
          for diags, (vecs, solution) in tests.items():
            v_diags = array_ops.matrix_diag(
                vecs.astype(dtype),
                k=diags,
                padding_value=padding_value,
                align=align)
            mask = solution == 0
            solution = (solution + padding_value * mask).astype(dtype)
            self.assertEqual(v_diags.get_shape(), solution.shape)
            self.assertAllEqual(v_diags, solution)

  @test_util.run_deprecated_v1
  def testVectorBatch(self):
    # [FUNCTION DOC]
    # Runs the batch-vector helper for bfloat16, float32, float64, int32, int64 and bool element types.
    self._testVectorBatch(dtypes_lib.bfloat16.as_numpy_dtype)
    self._testVectorBatch(np.float32)
    self._testVectorBatch(np.float64)
    self._testVectorBatch(np.int32)
    self._testVectorBatch(np.int64)
    self._testVectorBatch(np.bool_)

  @test_util.run_deprecated_v1
  def testRectangularBatch(self):
    # [FUNCTION DOC]
    # Tests matrix_diag with explicit num_rows and/or num_cols for non-square outputs. Builds a list of
    # expected output sizes per diagonal range from the square, extra, tall and fat fixtures. For each
    # padding value and alignment it checks (a) both sizes given on tall and fat cases, (b) only num_rows
    # given, truncating the expected solution columns and re-packing the input, (c) only num_cols given,
    # truncating rows. Asserts that shape and values match the solution with padding applied.
    with self.cached_session():
      # Stores expected num_rows and num_cols (when the other is given).
      # expected[d_lower, d_upper] = (expected_num_rows, expected_num_cols)
      test_list = list()

      # Square cases:
      expected = {
          (-1, -1): (5, 4),
          (-4, -3): (5, 2),
          (-2, 1): (5, 5),
          (2, 4): (3, 5),
      }
      # Do not change alignment yet. Re-alignment needs to happen after the
      # solution shape is updated.
      test_list.append((expected, square_cases()))

      # More cases:
      expected = {(-3, -1): (5, 4), (-1, 1): (4, 4), (2, 4): (4, 6)}
      test_list.append((expected, self._moreCases()))

      # Tall cases
      expected = {
          (0, 0): (3, 3),
          (-4, -3): (5, 2),
          (-2, -1): (4, 3),
          (-2, 1): (3, 3),
          (1, 2): (2, 3)
      }
      test_list.append((expected, tall_cases()))

      # Fat cases
      expected = {
          (2, 2): (2, 4),
          (-2, 0): (3, 3),
          (-1, 1): (3, 3),
          (0, 3): (3, 3)
      }
      test_list.append((expected, fat_cases()))

      for padding_value, align in zip_to_first_list_length([0, 555, -11],
                                                           alignment_list):
        # Giving both num_rows and num_cols
        for _, tests in [tall_cases(align), fat_cases(align)]:
          for diags, (vecs, solution) in tests.items():
            v_diags = array_ops.matrix_diag(
                vecs,
                k=diags,
                num_rows=solution.shape[-2],
                num_cols=solution.shape[-1],
                padding_value=padding_value,
                align=align)
            mask = solution == 0
            solution = solution + padding_value * mask
            self.assertEqual(v_diags.get_shape(), solution.shape)
            self.assertAllEqual(v_diags, solution)

        # Giving just num_rows.
        for expected, (_, tests) in test_list:
          for diags, (_, new_num_cols) in expected.items():
            vecs, solution = tests[diags]
            solution = solution.take(indices=range(new_num_cols), axis=-1)
            # Repacks the diagonal input according to the new solution shape.
            vecs = repack_diagonals(
                vecs, diags, solution.shape[-2], new_num_cols, align=align)
            v_diags = array_ops.matrix_diag(
                vecs,
                k=diags,
                num_rows=solution.shape[-2],
                padding_value=padding_value,
                align=align)
            mask = solution == 0
            solution = solution + padding_value * mask
            self.assertEqual(v_diags.get_shape(), solution.shape)
            self.assertAllEqual(v_diags, solution)

        # Giving just num_cols.
        for expected, (_, tests) in test_list:
          for diags, (new_num_rows, _) in expected.items():
            vecs, solution = tests[diags]
            solution = solution.take(indices=range(new_num_rows), axis=-2)
            # Repacks the diagonal input according to the new solution shape.
            vecs = repack_diagonals(
                vecs, diags, new_num_rows, solution.shape[-1], align=align)
            v_diags = array_ops.matrix_diag(
                vecs,
                k=diags,
                num_cols=solution.shape[-1],
                padding_value=padding_value,
                align=align)
            mask = solution == 0
            solution = solution + padding_value * mask
            self.assertEqual(v_diags.get_shape(), solution.shape)
            self.assertAllEqual(v_diags, solution)

  @test_util.run_deprecated_v1
  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Checks that matrix_diag on a scalar (rank 0) raises a value error mentioning that the input must
    # be at least rank 1, at graph construction time.
    with self.assertRaisesRegex(ValueError, "must be at least rank 1"):
      array_ops.matrix_diag(0)

  @test_util.run_deprecated_v1
  def testInvalidShapeAtEval(self):
    # [FUNCTION DOC]
    # Feeds a scalar into a placeholder of unknown shape passed to matrix_diag and expects an
    # operation error at run time saying the diagonal must be at least one-dimensional.
    with self.session():
      v = array_ops.placeholder(dtype=dtypes_lib.float32)
      with self.assertRaisesOpError("diagonal must be at least 1-dim"):
        array_ops.matrix_diag(v).eval(feed_dict={v: 0.0})

  @test_util.run_deprecated_v1
  def testGrad(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of matrix_diag using the gradient checker (error under 1e-4) for
    # vector and matrix inputs with the default main diagonal, then for a single sub-diagonal and a band
    # of three diagonals under each alignment mode. Uses random float32 inputs.
    shapes = ((3,), (7, 4))
    with self.session():
      for shape in shapes:
        x = constant_op.constant(np.random.rand(*shape), np.float32)
        y = array_ops.matrix_diag(x)
        error = gradient_checker.compute_gradient_error(x,
                                                        x.get_shape().as_list(),
                                                        y,
                                                        y.get_shape().as_list())
        self.assertLess(error, 1e-4)

    # {Sub,super}diagonals/band.
    tests = dict()  # tests[shape] = (d_lower, d_upper)
    tests[(3,)] = (-1, -1)
    tests[(7, 3, 4)] = (-1, 1)
    with self.session():
      for shape, diags in tests.items():
        x = constant_op.constant(np.random.rand(*shape), np.float32)
        for align in alignment_list:
          y = array_ops.matrix_diag(x, k=diags, align=align)
          error = gradient_checker.compute_gradient_error(
              x,
              x.get_shape().as_list(), y,
              y.get_shape().as_list())
          self.assertLess(error, 1e-4)


class MatrixSetDiagTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the matrix_set_diag operation, which replaces the diagonal or diagonal band of a matrix
  # (or batch) with given values, including alignment modes, shape validation and gradients.

  @test_util.run_deprecated_v1
  def testSquare(self):
    # [FUNCTION DOC]
    # Sets the main diagonal of a 3x3 matrix and checks shape and values; then, for banded cases from the
    # square fixture in all alignment modes, sets a band on random integer matrices and expects the result
    # to equal the old values outside the band mask plus the banded values.
    with self.session():
      v = np.array([1.0, 2.0, 3.0])
      mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
      mat_set_diag = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 1.0],
                               [1.0, 1.0, 3.0]])
      output = array_ops.matrix_set_diag(mat, v)
      self.assertEqual((3, 3), output.get_shape())
      self.assertAllEqual(mat_set_diag, self.evaluate(output))

      # Diagonal bands.
      for align in alignment_list:
        _, tests = square_cases(align)
        for diags, (vecs, banded_mat) in tests.items():
          mask = banded_mat[0] == 0
          input_mat = np.random.randint(10, size=mask.shape)
          solution = input_mat * mask + banded_mat[0]
          output = array_ops.matrix_set_diag(
              input_mat, vecs[0], k=diags, align=align)
          self.assertEqual(output.get_shape(), solution.shape)
          self.assertAllEqual(output, solution)

  @test_util.run_deprecated_v1
  def testRectangular(self):
    # [FUNCTION DOC]
    # Like the square test but for 2x3 and 3x2 matrices with the main diagonal, then for banded tall and
    # fat fixture cases in every alignment; asserts shape and values of the output.
    with self.session():
      v = np.array([3.0, 4.0])
      mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
      expected = np.array([[3.0, 1.0, 0.0], [1.0, 4.0, 1.0]])
      output = array_ops.matrix_set_diag(mat, v)
      self.assertEqual((2, 3), output.get_shape())
      self.assertAllEqual(expected, self.evaluate(output))

      v = np.array([3.0, 4.0])
      mat = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
      expected = np.array([[3.0, 1.0], [1.0, 4.0], [1.0, 1.0]])
      output = array_ops.matrix_set_diag(mat, v)
      self.assertEqual((3, 2), output.get_shape())
      self.assertAllEqual(expected, self.evaluate(output))

      # Diagonal bands.
      for align in alignment_list:
        for _, tests in [tall_cases(align), fat_cases(align)]:
          for diags, (vecs, banded_mat) in tests.items():
            mask = banded_mat[0] == 0
            input_mat = np.random.randint(10, size=mask.shape)
            solution = input_mat * mask + banded_mat[0]
            output = array_ops.matrix_set_diag(
                input_mat, vecs[0], k=diags, align=align)
            self.assertEqual(output.get_shape(), solution.shape)
            self.assertAllEqual(output, solution)

  def _testSquareBatch(self, dtype):
    # [FUNCTION DOC]
    # HELPER: dtype-parametrised body of the square batch test.
    # For a numpy dtype, sets the diagonal on a batch of two 3x3 matrices and checks shape and values,
    # then checks banded diagonals under each alignment on random matrices cast to the dtype.
    with self.cached_session():
      v_batch = np.array([[-1.0, 0.0, -3.0], [-4.0, -5.0, -6.0]]).astype(dtype)
      mat_batch = np.array([[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [1.0, 0.0, 3.0]],
                            [[4.0, 0.0, 4.0], [0.0, 5.0, 0.0],
                             [2.0, 0.0, 6.0]]]).astype(dtype)

      mat_set_diag_batch = np.array([[[-1.0, 0.0, 3.0], [0.0, 0.0, 0.0],
                                      [1.0, 0.0, -3.0]],
                                     [[-4.0, 0.0, 4.0], [0.0, -5.0, 0.0],
                                      [2.0, 0.0, -6.0]]]).astype(dtype)

      output = array_ops.matrix_set_diag(mat_batch, v_batch)
      self.assertEqual((2, 3, 3), output.get_shape())
      self.assertAllEqual(mat_set_diag_batch, self.evaluate(output))

      # Diagonal bands.
      for align in alignment_list:
        _, tests = square_cases(align)
        for diags, (vecs, banded_mat) in tests.items():
          mask = banded_mat == 0
          input_mat = np.random.randint(10, size=mask.shape).astype(dtype)
          solution = (input_mat * mask + banded_mat).astype(dtype)
          output = array_ops.matrix_set_diag(
              input_mat, vecs.astype(dtype), k=diags, align=align)
          self.assertEqual(output.get_shape(), solution.shape)
          self.assertAllEqual(output, solution)

  @test_util.run_deprecated_v1
  def testSquareBatch(self):
    # [FUNCTION DOC]
    # Runs the square batch helper for bfloat16, float32, float64, int32, int64 and bool.
    self._testSquareBatch(dtypes_lib.bfloat16.as_numpy_dtype)
    self._testSquareBatch(np.float32)
    self._testSquareBatch(np.float64)
    self._testSquareBatch(np.int32)
    self._testSquareBatch(np.int64)
    self._testSquareBatch(np.bool_)

  @test_util.run_deprecated_v1
  def testRectangularBatch(self):
    # [FUNCTION DOC]
    # Sets the main diagonal on a batch of 2x3 matrices and checks shape and values, then banded cases
    # from the tall and fat fixtures in every alignment on random integer matrices, comparing with the
    # old values outside the mask plus the band values.
    with self.session():
      v_batch = np.array([[-1.0, -2.0], [-4.0, -5.0]])
      mat_batch = np.array([[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]],
                            [[4.0, 0.0, 4.0], [0.0, 5.0, 0.0]]])

      mat_set_diag_batch = np.array([[[-1.0, 0.0, 3.0], [0.0, -2.0, 0.0]],
                                     [[-4.0, 0.0, 4.0], [0.0, -5.0, 0.0]]])
      output = array_ops.matrix_set_diag(mat_batch, v_batch)
      self.assertEqual((2, 2, 3), output.get_shape())
      self.assertAllEqual(mat_set_diag_batch, self.evaluate(output))

      # Diagonal bands.
      for align in alignment_list:
        for _, tests in [tall_cases(align), fat_cases(align)]:
          for diags, pair in tests.items():
            vecs, banded_mat = pair
            mask = banded_mat == 0
            input_mat = np.random.randint(10, size=mask.shape)
            solution = input_mat * mask + banded_mat
            output = array_ops.matrix_set_diag(
                input_mat, vecs, k=diags, align=align)
            self.assertEqual(output.get_shape(), solution.shape)
            self.assertAllEqual(output, solution)

  @test_util.run_deprecated_v1
  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Expects value errors when the input has rank below 2 (scalar matrix argument) and when the
    # diagonal argument is a scalar (needs rank at least 1), at graph construction time.
    with self.assertRaisesRegex(ValueError, "must be at least rank 2"):
      array_ops.matrix_set_diag(0, [0])
    with self.assertRaisesRegex(ValueError, "must be at least rank 1"):
      array_ops.matrix_set_diag([[0]], 0)

  @test_util.run_deprecated_v1
  def testInvalidShapeAtEval(self):
    # [FUNCTION DOC]
    # Uses placeholders with unknown shape. Expects run-time operation errors for a scalar input (needs at
    # least 2 dimensions), a scalar diagonal (at least 1 dimension) and mismatching leading batch
    # dimensions between input and diagonal.
    with self.session():
      v = array_ops.placeholder(dtype=dtypes_lib.float32)
      with self.assertRaisesOpError("input must be at least 2-dim"):
        array_ops.matrix_set_diag(v, [v]).eval(feed_dict={v: 0.0})
      with self.assertRaisesOpError("diagonal must be at least 1-dim"):
        array_ops.matrix_set_diag([[v]], v).eval(feed_dict={v: 0.0})

      d = array_ops.placeholder(dtype=dtypes_lib.float32)
      with self.assertRaisesOpError(
          "first dimensions of diagonal don't match"):
        array_ops.matrix_set_diag(v, d).eval(feed_dict={
            v: np.zeros((2, 3, 3)),
            d: np.ones((2, 4))
        })

  def _testGrad(self, input_shape, diag_shape, diags, align):
    # [FUNCTION DOC]
    # HELPER: gradient check body shared by the gradient test.
    # Takes input_shape, diag_shape, diags (offset or band) and align. Creates random float32 input and
    # diagonal constants, applies matrix_set_diag, and asserts the numerical gradient error with respect
    # to both the matrix and the diagonal is below 1e-4.
    with self.session():
      x = constant_op.constant(
          np.random.rand(*input_shape), dtype=dtypes_lib.float32)
      x_diag = constant_op.constant(
          np.random.rand(*diag_shape), dtype=dtypes_lib.float32)

      y = array_ops.matrix_set_diag(x, x_diag, k=diags, align=align)
      error_x = gradient_checker.compute_gradient_error(x,
                                                        x.get_shape().as_list(),
                                                        y,
                                                        y.get_shape().as_list())
      self.assertLess(error_x, 1e-4)
      error_x_diag = gradient_checker.compute_gradient_error(
          x_diag,
          x_diag.get_shape().as_list(), y,
          y.get_shape().as_list())
      self.assertLess(error_x_diag, 1e-4)

  @test_util.run_deprecated_v1
  def testGrad(self):
    # [FUNCTION DOC]
    # Iterates over several input shapes (square, tall, fat, batched), the main diagonal and a three-
    # diagonal band, and every alignment, deriving the diagonal shape from the band size, and calls the
    # gradient check helper for each combination.
    input_shapes = [(3, 4, 4), (3, 3, 4), (3, 4, 3), (7, 4, 8, 8)]
    diag_bands = [(0, 0)]

    diag_bands.append((-1, 1))
    for input_shape, diags, align in itertools.product(input_shapes, diag_bands,
                                                       alignment_list):
      lower_diag_index, upper_diag_index = diags
      num_diags = upper_diag_index - lower_diag_index + 1
      num_diags_dim = () if num_diags == 1 else (num_diags,)
      diag_shape = input_shape[:-2] + num_diags_dim + (min(input_shape[-2:]),)
      self._testGrad(input_shape, diag_shape, diags, align)

  @test_util.run_deprecated_v1
  def testGradWithNoShapeInformation(self):
    # [FUNCTION DOC]
    # Builds matrix_set_diag from placeholders without shape information, takes gradients with a given
    # upstream gradient, runs the graph feeding ones and twice-ones, and asserts the gradient w.r.t. the
    # diagonal equals the diagonal of the upstream gradient and that w.r.t. the matrix equals the upstream
    # gradient with its diagonal zeroed.
    with self.session() as sess:
      v = array_ops.placeholder(dtype=dtypes_lib.float32)
      mat = array_ops.placeholder(dtype=dtypes_lib.float32)
      grad_input = array_ops.placeholder(dtype=dtypes_lib.float32)
      output = array_ops.matrix_set_diag(mat, v)
      grads = gradients_impl.gradients(output, [mat, v], grad_ys=grad_input)
      grad_input_val = np.random.rand(3, 3).astype(np.float32)
      grad_vals = sess.run(
          grads,
          feed_dict={
              v: 2 * np.ones(3),
              mat: np.ones((3, 3)),
              grad_input: grad_input_val
          })
      self.assertAllEqual(np.diag(grad_input_val), grad_vals[1])
      self.assertAllEqual(grad_input_val - np.diag(np.diag(grad_input_val)),
                          grad_vals[0])

  @test_util.run_deprecated_v1
  def testInvalidDiagRankAtEval(self):
    # [FUNCTION DOC]
    # Runs on CPU with the XLA jit level turned off. Feeds a rank 3 input and a rank 1 diagonal together
    # with a three-wide band into matrix_set_diag and expects a run-time operation error saying the
    # diagonal tensor rank must be large enough to hold a diagonal-count dimension.
    config = config_pb2.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = (
        config_pb2.OptimizerOptions.OFF
    )
    with self.session(config=config, use_gpu=False):
      with ops.device("/cpu:0"):
        v = array_ops.placeholder(dtype=dtypes_lib.float32)
        d = array_ops.placeholder(dtype=dtypes_lib.float32)
        # Input rank 3
        input_shape = (2, 3, 4)
        # Diag rank 1. Triggers check because rank <= 3-2=1.
        diag_shape = (10,)
        k = (-1, 1)

        with self.assertRaisesOpError(
            "Diagonal tensor rank must be large enough to contain a "
            "diagonal-count dimension"
        ):
          array_ops.matrix_set_diag(v, d, k=k).eval(
              feed_dict={v: np.zeros(input_shape), d: np.zeros(diag_shape)}
          )


class MatrixDiagPartTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the matrix_diag_part operation, which extracts the diagonal or a band of diagonals from
  # matrices, including padding, alignment, unknown shapes, validation and gradients.

  @test_util.run_deprecated_v1
  def testSquare(self):
    # [FUNCTION DOC]
    # Extracts the main diagonal of a 3x3 matrix, then single super/sub diagonals with offsets, then bands
    # from the square fixture in every alignment (first batch entry); asserts shape and values.
    with self.session():
      v = np.array([1.0, 2.0, 3.0])
      mat = np.diag(v)
      mat_diag = array_ops.matrix_diag_part(mat)
      self.assertEqual((3,), mat_diag.get_shape())
      self.assertAllEqual(mat_diag, v)

      for offset in [-2, 3]:
        mat = np.diag(v, offset)
        mat_diag = array_ops.matrix_diag_part(mat, k=offset)
        self.assertEqual((3,), mat_diag.get_shape())
        self.assertAllEqual(mat_diag, v)

      # Diagonal bands.
      for align in alignment_list:
        mat, tests = square_cases(align)
        for diags, pair in tests.items():
          solution, _ = pair
          mat_diag = array_ops.matrix_diag_part(mat[0], k=diags, align=align)
          self.assertEqual(mat_diag.get_shape(), solution[0].shape)
          self.assertAllEqual(mat_diag, solution[0])

  @test_util.run_deprecated_v1
  def testRectangular(self):
    # [FUNCTION DOC]
    # Extracts the main diagonal of 2x3 and 3x2 matrices, then banded diagonals from the tall and fat
    # fixtures in every alignment; asserts shape and values.
    with self.session():
      mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      mat_diag = array_ops.matrix_diag_part(mat)
      self.assertAllEqual(mat_diag, np.array([1.0, 5.0]))
      mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      mat_diag = array_ops.matrix_diag_part(mat)
      self.assertAllEqual(mat_diag, np.array([1.0, 4.0]))

      # Diagonal bands.
      for align in alignment_list:
        for mat, tests in [tall_cases(align), fat_cases(align)]:
          for diags, pair in tests.items():
            solution, _ = pair
            mat_diag = array_ops.matrix_diag_part(
                mat[0], k=diags, align=align)
            self.assertEqual(mat_diag.get_shape(), solution[0].shape)
            self.assertAllEqual(mat_diag, solution[0])

  def _testSquareBatch(self, dtype):
    # [FUNCTION DOC]
    # HELPER: dtype-parametrised body of the square batch test.
    # For a numpy dtype, extracts the main diagonal from a batch of 3x3 matrices, then banded diagonals
    # with padding values (0, 555, -11) paired with alignments; expected values replace zeros by the
    # padding value, and shapes are compared.
    with self.cached_session():
      v_batch = np.array([[1.0, 0.0, 3.0], [4.0, 5.0, 6.0]]).astype(dtype)
      mat_batch = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
                            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0],
                             [0.0, 0.0, 6.0]]]).astype(dtype)
      self.assertEqual(mat_batch.shape, (2, 3, 3))
      mat_batch_diag = array_ops.matrix_diag_part(mat_batch)
      self.assertEqual((2, 3), mat_batch_diag.get_shape())
      self.assertAllEqual(mat_batch_diag, v_batch)

      # Diagonal bands with padding_value.
      for padding_value, align in zip_to_first_list_length([0, 555, -11],
                                                           alignment_list):
        mat, tests = square_cases(align)
        for diags, pair in tests.items():
          solution, _ = pair
          mat_batch_diag = array_ops.matrix_diag_part(
              mat.astype(dtype),
              k=diags,
              padding_value=padding_value,
              align=align)
          mask = solution == 0
          solution = (solution + padding_value * mask).astype(dtype)
          self.assertEqual(mat_batch_diag.get_shape(), solution.shape)
          self.assertAllEqual(mat_batch_diag, solution)

  @test_util.run_deprecated_v1
  def testSquareBatch(self):
    # [FUNCTION DOC]
    # Runs the square batch helper for bfloat16, float32, float64, int32, int64 and bool.
    self._testSquareBatch(dtypes_lib.bfloat16.as_numpy_dtype)
    self._testSquareBatch(np.float32)
    self._testSquareBatch(np.float64)
    self._testSquareBatch(np.int32)
    self._testSquareBatch(np.int64)
    self._testSquareBatch(np.bool_)

  @test_util.run_deprecated_v1
  def testRectangularBatch(self):
    # [FUNCTION DOC]
    # Extracts the main diagonal from a batch of 2x3 matrices, then banded diagonals from the tall and fat
    # fixtures with padding values and alignments, checking shape and values.
    with self.session():
      v_batch = np.array([[1.0, 2.0], [4.0, 5.0]])
      mat_batch = np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]]])
      self.assertEqual(mat_batch.shape, (2, 2, 3))
      mat_batch_diag = array_ops.matrix_diag_part(mat_batch)
      self.assertEqual((2, 2), mat_batch_diag.get_shape())
      self.assertAllEqual(mat_batch_diag, v_batch)

      # Diagonal bands with padding_value and align.
      for padding_value, align in zip_to_first_list_length([0, 555, -11],
                                                           alignment_list):
        for mat, tests in [tall_cases(align), fat_cases(align)]:
          for diags, pair in tests.items():
            solution, _ = pair
            mat_batch_diag = array_ops.matrix_diag_part(
                mat, k=diags, padding_value=padding_value, align=align)
            mask = solution == 0
            solution = solution + padding_value * mask
            self.assertEqual(mat_batch_diag.get_shape(), solution.shape)
            self.assertAllEqual(mat_batch_diag, solution)

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Uses a placeholder of unknown dimensions and extracts the first sub-diagonal (offset minus one);
    # feeds a 3x3 matrix and expects the values four and eight.
    matrix = array_ops.placeholder(dtypes_lib.int32, shape=[None, None])
    result = array_ops.matrix_diag_part(matrix, k=-1)
    input_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    with self.session():
      result_eval = result.eval(feed_dict={matrix: input_matrix})
    self.assertAllEqual([4, 8], result_eval)

  @test_util.run_deprecated_v1
  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Expects a value error that the input must be at least rank 2 when a scalar is passed.
    with self.assertRaisesRegex(ValueError, "must be at least rank 2"):
      array_ops.matrix_diag_part(0)

  @test_util.run_deprecated_v1
  def testInvalidShapeAtEval(self):
    # [FUNCTION DOC]
    # Feeds a scalar to a placeholder of unknown shape and expects a run-time operation error saying the
    # input must be at least 2-dimensional.
    with self.session():
      v = array_ops.placeholder(dtype=dtypes_lib.float32)
      with self.assertRaisesOpError("input must be at least 2-dim"):
        array_ops.matrix_diag_part(v).eval(feed_dict={v: 0.0})

  @test_util.run_deprecated_v1
  def testGrad(self):
    # [FUNCTION DOC]
    # Checks the numerical gradient (error under 1e-4) of matrix_diag_part on several shapes with the
    # default diagonal, then for a sub-diagonal and a three-diagonal band under each alignment, with
    # random float32 inputs.
    shapes = ((3, 3), (2, 3), (3, 2), (5, 3, 3))
    with self.session():
      for shape in shapes:
        x = constant_op.constant(np.random.rand(*shape), dtype=np.float32)
        y = array_ops.matrix_diag_part(x)
        error = gradient_checker.compute_gradient_error(x,
                                                        x.get_shape().as_list(),
                                                        y,
                                                        y.get_shape().as_list())
        self.assertLess(error, 1e-4)

    # {Sub,super}diagonals/band.
    tests = dict()  # tests[shape] = (d_lower, d_upper)
    tests[(3, 3)] = (-1, -1)
    tests[(7, 3, 4)] = (-1, 1)
    with self.session():
      for align in alignment_list:
        for shape, diags in tests.items():
          x = constant_op.constant(np.random.rand(*shape), np.float32)
          y = array_ops.matrix_diag_part(input=x, k=diags, align=align)
          error = gradient_checker.compute_gradient_error(
              x,
              x.get_shape().as_list(), y,
              y.get_shape().as_list())
          self.assertLess(error, 1e-4)


class DiagTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the general diag operation, which builds a tensor of twice the rank from an input, with
  # its inverse diag_part used for round-trip verification. Covers ranks 0 through 4 for integer, float
  # and complex types.

  def _diagOp(self, diag, dtype, expected_ans, use_gpu):
    # [FUNCTION DOC]
    # HELPER: runs diag on one device setting and checks the result.
    # KNOWN ISSUE: the inverse check applies diag_part to the expected answer rather than to the diag
    # output, so it does not verify the actual diag result is invertible.
    # Takes diag (numpy array), dtype, expected_ans (expected result array) and use_gpu flag. Casts the
    # input to the dtype, runs diag, and runs diag_part on the expected answer. Asserts the diag result
    # is close to the expected answer, the diag_part result is close to the input, and that static shapes
    # match.
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.diag(ops.convert_to_tensor(diag.astype(dtype)))
      out = self.evaluate(tf_ans)
      tf_ans_inv = array_ops.diag_part(expected_ans)
      inv_out = self.evaluate(tf_ans_inv)
    self.assertAllClose(out, expected_ans)
    self.assertAllClose(inv_out, diag)
    self.assertShapeEqual(expected_ans, tf_ans)
    self.assertShapeEqual(diag, tf_ans_inv)

  def diagOp(self, diag, dtype, expected_ans):
    # [FUNCTION DOC]
    # HELPER: public (non-test) wrapper that runs the check for both the CPU (use_gpu false) and the GPU
    # (use_gpu true) settings. Takes the input array, dtype and expected answer; returns nothing.
    self._diagOp(diag, dtype, expected_ans, False)
    self._diagOp(diag, dtype, expected_ans, True)

  def testEmptyTensor(self):
    # [FUNCTION DOC]
    # Checks diag of an empty vector yields a 0x0 result, using int32.
    x = np.array([])
    expected_ans = np.empty([0, 0])
    self.diagOp(x, np.int32, expected_ans)

  def testRankOneIntTensor(self):
    # [FUNCTION DOC]
    # Checks diag of a length-3 integer vector gives the expected 3x3 diagonal matrix for int32 and int64.
    x = np.array([1, 2, 3])
    expected_ans = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    self.diagOp(x, np.int32, expected_ans)
    self.diagOp(x, np.int64, expected_ans)

  def testRankOneFloatTensor(self):
    # [FUNCTION DOC]
    # Checks diag of a length-3 float vector gives the expected 3x3 diagonal matrix for float32 and
    # float64.
    x = np.array([1.1, 2.2, 3.3])
    expected_ans = np.array([[1.1, 0, 0], [0, 2.2, 0], [0, 0, 3.3]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)

  def testRankOneComplexTensor(self):
    # [FUNCTION DOC]
    # Checks diag of a length-3 complex vector gives the expected 3x3 complex diagonal matrix for
    # complex64 and complex128.
    for dtype in [np.complex64, np.complex128]:
      x = np.array([1.1 + 1.1j, 2.2 + 2.2j, 3.3 + 3.3j], dtype=dtype)
      expected_ans = np.array(
          [[1.1 + 1.1j, 0 + 0j, 0 + 0j], [0 + 0j, 2.2 + 2.2j, 0 + 0j],
           [0 + 0j, 0 + 0j, 3.3 + 3.3j]],
          dtype=dtype)
      self.diagOp(x, dtype, expected_ans)

  def testRankTwoIntTensor(self):
    # [FUNCTION DOC]
    # Checks diag of a 2x3 integer matrix gives the expected 2x3x2x3 tensor for int32 and int64.
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected_ans = np.array([[[[1, 0, 0], [0, 0, 0]], [[0, 2, 0], [0, 0, 0]],
                              [[0, 0, 3], [0, 0, 0]]],
                             [[[0, 0, 0], [4, 0, 0]], [[0, 0, 0], [0, 5, 0]],
                              [[0, 0, 0], [0, 0, 6]]]])
    self.diagOp(x, np.int32, expected_ans)
    self.diagOp(x, np.int64, expected_ans)

  def testRankTwoFloatTensor(self):
    # [FUNCTION DOC]
    # Checks diag of a 2x3 float matrix gives the expected 2x3x2x3 tensor for float32 and float64.
    x = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    expected_ans = np.array(
        [[[[1.1, 0, 0], [0, 0, 0]], [[0, 2.2, 0], [0, 0, 0]],
          [[0, 0, 3.3], [0, 0, 0]]], [[[0, 0, 0], [4.4, 0, 0]],
                                      [[0, 0, 0], [0, 5.5, 0]], [[0, 0, 0],
                                                                 [0, 0, 6.6]]]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)

  def testRankTwoComplexTensor(self):
    # [FUNCTION DOC]
    # Checks diag of a 2x3 complex matrix gives the expected rank 4 tensor for complex64 and complex128.
    for dtype in [np.complex64, np.complex128]:
      x = np.array(
          [[1.1 + 1.1j, 2.2 + 2.2j, 3.3 + 3.3j],
           [4.4 + 4.4j, 5.5 + 5.5j, 6.6 + 6.6j]],
          dtype=dtype)
      expected_ans = np.array(
          [[[[1.1 + 1.1j, 0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j, 0 + 0j]], [
              [0 + 0j, 2.2 + 2.2j, 0 + 0j], [0 + 0j, 0 + 0j, 0 + 0j]
          ], [[0 + 0j, 0 + 0j, 3.3 + 3.3j], [0 + 0j, 0 + 0j, 0 + 0j]]], [[
              [0 + 0j, 0 + 0j, 0 + 0j], [4.4 + 4.4j, 0 + 0j, 0 + 0j]
          ], [[0 + 0j, 0 + 0j, 0 + 0j], [0 + 0j, 5.5 + 5.5j, 0 + 0j]
             ], [[0 + 0j, 0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j, 6.6 + 6.6j]]]],
          dtype=dtype)
      self.diagOp(x, dtype, expected_ans)

  def testRankThreeFloatTensor(self):
    # [FUNCTION DOC]
    # Checks diag of a 2x2x2 float tensor gives the expected rank 6 tensor for float32 and float64.
    x = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]])
    expected_ans = np.array([[[[[[1.1, 0], [0, 0]], [[0, 0], [0, 0]]],
                               [[[0, 2.2], [0, 0]], [[0, 0], [0, 0]]]],
                              [[[[0, 0], [3.3, 0]], [[0, 0], [0, 0]]],
                               [[[0, 0], [0, 4.4]], [[0, 0], [0, 0]]]]],
                             [[[[[0, 0], [0, 0]], [[5.5, 0], [0, 0]]],
                               [[[0, 0], [0, 0]], [[0, 6.6], [0, 0]]]],
                              [[[[0, 0], [0, 0]], [[0, 0], [7.7, 0]]],
                               [[[0, 0], [0, 0]], [[0, 0], [0, 8.8]]]]]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)

  def testRankThreeComplexTensor(self):
    # [FUNCTION DOC]
    # Checks diag of a 2x2x2 complex tensor gives the expected rank 6 tensor for complex64 and complex128.
    for dtype in [np.complex64, np.complex128]:
      x = np.array(
          [[[1.1 + 1.1j, 2.2 + 2.2j], [3.3 + 3.3j, 4.4 + 4.4j]],
           [[5.5 + 5.5j, 6.6 + 6.6j], [7.7 + 7.7j, 8.8 + 8.8j]]],
          dtype=dtype)
      expected_ans = np.array(
          [[[[[[1.1 + 1.1j, 0 + 0j], [0 + 0j, 0 + 0j]], [[0 + 0j, 0 + 0j], [
              0 + 0j, 0 + 0j
          ]]], [[[0 + 0j, 2.2 + 2.2j], [0 + 0j, 0 + 0j]], [[0 + 0j, 0 + 0j], [
              0 + 0j, 0 + 0j
          ]]]], [[[[0 + 0j, 0 + 0j], [3.3 + 3.3j, 0 + 0j]], [[0 + 0j, 0 + 0j], [
              0 + 0j, 0 + 0j
          ]]], [[[0 + 0j, 0 + 0j], [0 + 0j, 4.4 + 4.4j]], [[0 + 0j, 0 + 0j], [
              0 + 0j, 0 + 0j
          ]]]]], [[[[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], [
              [5.5 + 5.5j, 0 + 0j], [0 + 0j, 0 + 0j]
          ]], [[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], [[0 + 0j, 6.6 + 6.6j], [
              0 + 0j, 0 + 0j
          ]]]], [[[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], [[0 + 0j, 0 + 0j], [
              7.7 + 7.7j, 0 + 0j
          ]]], [[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]],
                [[0 + 0j, 0 + 0j], [0 + 0j, 8.8 + 8.8j]]]]]],
          dtype=dtype)
      self.diagOp(x, dtype, expected_ans)

  def testRankFourNumberTensor(self):
    # [FUNCTION DOC]
    # For float32, float64, int64 and int32, checks diag of a rank 4 tensor of shape 2x1x2x3 gives the
    # expected rank 8 tensor.
    for dtype in [np.float32, np.float64, np.int64, np.int32]:
      # Input with shape [2, 1, 2, 3]
      x = np.array(
          [[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 9], [10, 11, 12]]]], dtype=dtype)
      # Output with shape [2, 1, 2, 3, 2, 1, 2, 3]
      expected_ans = np.array(
          [[[[[[[[1, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]], [
              [[[0, 2, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]
          ], [[[[0, 0, 3], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]]], [[
              [[[0, 0, 0], [4, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]
          ], [[[[0, 0, 0], [0, 5, 0]]], [[[0, 0, 0], [0, 0, 0]]]], [
              [[[0, 0, 0], [0, 0, 6]]], [[[0, 0, 0], [0, 0, 0]]]
          ]]]], [[[[[[[0, 0, 0], [0, 0, 0]]], [[[7, 0, 0], [0, 0, 0]]]], [
              [[[0, 0, 0], [0, 0, 0]]], [[[0, 8, 0], [0, 0, 0]]]
          ], [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 9], [0, 0, 0]]]]], [[
              [[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [10, 0, 0]]]
          ], [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 11, 0]]]
             ], [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 12]]]]]]]],
          dtype=dtype)
      self.diagOp(x, dtype, expected_ans)

  @test_util.run_deprecated_v1
  def testInvalidRank(self):
    # [FUNCTION DOC]
    # Expects a value error that the input must be at least rank 1 when diag gets a scalar.
    with self.assertRaisesRegex(ValueError, "must be at least rank 1"):
      array_ops.diag(0.0)

  def testInvalidInput(self):
    # [FUNCTION DOC]
    # Calls matrix_diag with an absurdly large diagonal offset and a small rank 4 input; expects either an
    # invalid-argument error or a value error, whether raised at construction or on evaluation.
    with self.session():
      with self.assertRaises((errors.InvalidArgumentError, ValueError)):
        op = array_ops.matrix_diag(
            k=1070828000000, diagonal=np.ones((2, 2, 2, 2)))
        self.evaluate(op)


class DiagPartOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the diag_part operation, which extracts the "diagonal" of an even-rank tensor whose first
  # half of dimensions equals its second half. Fixes the numpy random seed in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Seeds numpy's random generator with zero so random inputs are reproducible. Returns nothing.
    np.random.seed(0)

  def _diagPartOp(self, tensor, dtype, expected_ans, use_gpu):
    # [FUNCTION DOC]
    # HELPER: runs diag_part on one device setting and checks the result.
    # Takes tensor (numpy array), dtype, expected_ans and use_gpu flag. Casts the input, runs diag_part,
    # asserts the result is close to the expected answer and static shapes match.
    with self.cached_session(use_gpu=use_gpu):
      tensor = ops.convert_to_tensor(tensor.astype(dtype))
      tf_ans_inv = array_ops.diag_part(tensor)
      inv_out = self.evaluate(tf_ans_inv)
    self.assertAllClose(inv_out, expected_ans)
    self.assertShapeEqual(expected_ans, tf_ans_inv)

  def diagPartOp(self, tensor, dtype, expected_ans):
    # [FUNCTION DOC]
    # HELPER: public (non-test) wrapper that runs the check for the CPU and GPU settings.
    # Takes tensor, dtype and expected answer; returns nothing.
    self._diagPartOp(tensor, dtype, expected_ans, False)
    self._diagPartOp(tensor, dtype, expected_ans, True)

  def testRankTwoFloatTensor(self):
    # [FUNCTION DOC]
    # Checks diag_part of a random 3x3 matrix equals its main diagonal for float32 and float64.
    x = np.random.rand(3, 3)
    i = np.arange(3)
    expected_ans = x[i, i]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testRankFourFloatTensorUnknownShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says rank four but the test uses a rank 2 (3x3) matrix.
    # Sets the static shape of the input tensor to unknown, partially unknown (first or second
    # dimension) variants and checks diag_part still gives the diagonal with matching shape on CPU.
    x = np.random.rand(3, 3)
    i = np.arange(3)
    expected_ans = x[i, i]
    for shape in None, (None, 3), (3, None):
      with self.cached_session(use_gpu=False):
        t = ops.convert_to_tensor(x.astype(np.float32))
        t.set_shape(shape)
        tf_ans = array_ops.diag_part(t)
        out = self.evaluate(tf_ans)
      self.assertAllClose(out, expected_ans)
      self.assertShapeEqual(expected_ans, tf_ans)

  def testRankFourFloatTensor(self):
    # [FUNCTION DOC]
    # Checks diag_part of a random 2x3x2x3 tensor equals the elements whose first-half and second-half
    # indices coincide, for float32 and float64.
    x = np.random.rand(2, 3, 2, 3)
    i = np.arange(2)[:, None]
    j = np.arange(3)
    expected_ans = x[i, j, i, j]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testRankSixFloatTensor(self):
    # [FUNCTION DOC]
    # Checks diag_part of a random 2x2x2x2x2x2 tensor equals the matching-index elements, for float32 and
    # float64.
    x = np.random.rand(2, 2, 2, 2, 2, 2)
    i = np.arange(2)[:, None, None]
    j = np.arange(2)[:, None]
    k = np.arange(2)
    expected_ans = x[i, j, k, i, j, k]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testRankEightComplexTensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: name says complex, but the random data is real-valued and only cast to complex types.
    # Checks diag_part on a random rank 8 tensor (2,2,2,3,2,2,2,3) cast to complex64 and complex128.
    x = np.random.rand(2, 2, 2, 3, 2, 2, 2, 3)
    i = np.arange(2)[:, None, None, None]
    j = np.arange(2)[:, None, None]
    k = np.arange(2)[:, None]
    l = np.arange(3)
    expected_ans = x[i, j, k, l, i, j, k, l]
    self.diagPartOp(x, np.complex64, expected_ans)
    self.diagPartOp(x, np.complex128, expected_ans)

  @test_util.run_deprecated_v1
  def testOddRank(self):
    # [FUNCTION DOC]
    # Expects value errors from diag_part for tensors of odd rank (rank 1, rank 3) and for a scalar.
    w = np.random.rand(2)
    x = np.random.rand(2, 2, 2)
    self.assertRaises(ValueError, self.diagPartOp, w, np.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, x, np.float32, 0)
    with self.assertRaises(ValueError):
      array_ops.diag_part(0.0)

  @test_util.run_deprecated_v1
  def testUnevenDimensions(self):
    # [FUNCTION DOC]
    # Expects value errors when the two halves of the shape do not match (2x5, and 2x1x2x3).
    w = np.random.rand(2, 5)
    x = np.random.rand(2, 1, 2, 3)
    self.assertRaises(ValueError, self.diagPartOp, w, np.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, x, np.float32, 0)


class DiagGradOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests numerical gradients of the diag operation.

  @test_util.run_deprecated_v1
  def testDiagGrad(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: defines an unused local list named errors that shadows the imported errors module.
    # Seeds numpy with zero. For shapes of rank 1, 2 and 3 and float32 and float64 inputs, computes the
    # numerical gradient error of diag and asserts it is below 1e-4, logging the error.
    np.random.seed(0)
    shapes = ((3,), (3, 3), (3, 3, 3))
    dtypes = (dtypes_lib.float32, dtypes_lib.float64)
    with self.session(use_gpu=False):
      errors = []
      for shape in shapes:
        for dtype in dtypes:
          x1 = constant_op.constant(np.random.rand(*shape), dtype=dtype)
          y = array_ops.diag(x1)
          error = gradient_checker.compute_gradient_error(
              x1,
              x1.get_shape().as_list(), y,
              y.get_shape().as_list())
          tf_logging.info("error = %f", error)
          self.assertLess(error, 1e-4)


class DiagGradPartOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests numerical gradients of the diag_part operation.

  @test_util.run_deprecated_v1
  def testDiagPartGrad(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: defines an unused local list named errors that shadows the imported errors module.
    # Seeds numpy with zero. For shapes 3x3 and 3x3x3x3 and float32 and float64, computes the numerical
    # gradient error of diag_part and asserts it is below 1e-4, logging the error.
    np.random.seed(0)
    shapes = ((3, 3), (3, 3, 3, 3))
    dtypes = (dtypes_lib.float32, dtypes_lib.float64)
    with self.session(use_gpu=False):
      errors = []
      for shape in shapes:
        for dtype in dtypes:
          x1 = constant_op.constant(np.random.rand(*shape), dtype=dtype)
          y = array_ops.diag_part(x1)
          error = gradient_checker.compute_gradient_error(
              x1,
              x1.get_shape().as_list(), y,
              y.get_shape().as_list())
          tf_logging.info("error = %f", error)
          self.assertLess(error, 1e-4)


if __name__ == "__main__":
  test.main()
