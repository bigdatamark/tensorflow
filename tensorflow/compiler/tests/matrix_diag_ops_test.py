# [MODULE DOC]
# XLA compiler test module for the matrix diagonal ops (matrix diag, matrix set diag, matrix
# diag part), version one and two, including bands, alignments, shapes and padding values.
# Defines shared test-case builders, alignment helpers and three test classes. Runs the test
# runner when executed as a script. Defines module-level alignment constants on import.
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
"""Tests for XLA matrix diag ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import googletest

default_v2_alignment = "LEFT_LEFT"
alignment_list = ["RIGHT_LEFT", "LEFT_RIGHT"]


def zip_to_first_list_length(a, b):
  # [FUNCTION DOC]
  # HELPER: pairs each element of the first list with the corresponding element of the second,
  # truncating the second if longer or padding it with null values if shorter. Returns an
  # iterable of pairs with the length of the first list.
  if len(b) > len(a):
    return zip(a, b[:len(a)])
  return zip(a, b + [None] * (len(a) - len(b)))


# Routines to convert test cases to have diagonals in a specified alignment.
# Copied from //third_party/tensorflow/python/kernel_tests/array_ops/
# diag_op_test.py
def repack_diagonals(packed_diagonals,
                     diag_index,
                     num_rows,
                     num_cols,
                     align=None):
  # [FUNCTION DOC]
  # HELPER: converts compact diagonals from the default left-left alignment to another one.
  # Inputs: packed_diagonals (numpy array), diag_index (pair of lower and upper band indices),
  # num_rows and num_cols of the matrix, align (optional string like right-left naming the
  # alignment of super and sub diagonals). Returns the input unchanged for default or none,
  # otherwise a new array where each diagonal is shifted left or right within its row.
  # Note the loop reuses the name of the diag_index parameter as its counter after the band
  # was unpacked, which is harmless.
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
  # HELPER: re-aligns every test case of a mapping. Inputs: tests (mapping from band to a pair of
  # compact diagonals and padded diagonals) and align (optional string). If align is none or the
  # default left-left alignment, returns the same mapping; otherwise returns a new mapping with
  # the compact diagonals repacked for the alignment and the padded diagonals unchanged.
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
# Copied from //third_party/tensorflow/python/kernel_tests/array_ops/
# diag_op_test.py
def square_cases(align=None):
  # [FUNCTION DOC]
  # HELPER: test data for a square pair of matrices, shared by the diag, diag part and set diag
  # tests. Input: align (optional string, default none). Returns a pair: the batch of matrices,
  # and a mapping from band (lower, upper) to a pair of compact diagonals and padded diagonal
  # matrices, repacked to the requested alignment.
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
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagonals)
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
  # HELPER: test data for a tall (more rows than columns) pair of matrices. Input: align
  # (optional string, default none). Returns a pair: the batch of matrices, and a mapping from
  # band (lower, upper) to a pair of compact diagonals and padded diagonal matrices, repacked to
  # the requested alignment.
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
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagonals)
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
  # HELPER: test data for a fat (more columns than rows) pair of matrices. Input: align (optional
  # string, default none). Returns a pair: the batch of matrices, and a mapping from band
  # (lower, upper) to a pair of the compact diagonals and the padded diagonal matrices, repacked
  # to the requested alignment.
  # pyformat: disable
  mat = np.array([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 1, 2, 3]],
                  [[4, 5, 6, 7],
                   [8, 9, 1, 2],
                   [3, 4, 5, 6]]])
  tests = dict()
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagonals)
  tests[0, 0] = (np.array([[1, 6, 2],
                           [4, 9, 5]]),
                 np.array([[[1, 0, 0, 0],
                            [0, 6, 0, 0],
                            [0, 0, 2, 0]],
                           [[4, 0, 0, 0],
                            [0, 9, 0, 0],
                            [0, 0, 5, 0]]]))
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
  # HELPER: returns the list of the square, tall and fat test case sets, each built with the
  # given alignment (optional, default none meaning the default alignment).
  return [square_cases(align), tall_cases(align), fat_cases(align)]


class MatrixDiagTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the matrix diag op (building matrices from diagonals, version one and
  # two semantics including bands, alignments, sizes and padding value) on the XLA test device.

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     high_level=True,
                                     rtol=1e-3,
                                     atol=1e-5):
    # [FUNCTION DOC]
    # HELPER: verifies that the matrix diag op produces the expected solution. Inputs: params
    # (mapping with the diagonal and other op arguments; the diagonal entry is overwritten with a
    # placeholder), solution (expected numpy array), high_level (boolean, default true; picks the
    # high-level op or the low-level generated op), rtol (default 0.001), atol (default 0.00001).
    # For each numeric type except the 8-bit ones, runs the op in the test scope feeding the
    # diagonal cast to that type and asserts the output dtype and values.
    """Verifies that matrix_diag produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_diag.
      solution: numpy array representing the expected output of matrix_diag.
      high_level: call high_level matrix_diag
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    diagonal = params["diagonal"]
    with self.session() as session:
      for dtype in self.numeric_types - {np.int8, np.uint8}:
        expected = solution.astype(dtype)
        with self.test_scope():
          params["diagonal"] = array_ops.placeholder(
              dtype, diagonal.shape, name="diagonal")
          if high_level:
            # wraps gen_array_ops.matrix_diag_v3
            output = array_ops.matrix_diag(**params)
          else:
            # TODO(b/201086188): Remove this case once MatrixDiag V1 is removed.
            output = gen_array_ops.matrix_diag(**params)
        result = session.run(output,
                             {params["diagonal"]: diagonal.astype(dtype)})
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)

  # Generic tests applicable to both v1 and v2 ops.
  # Originally from unary_ops_tests.py.
  def _testV1Level(self, high_level):
    # [FUNCTION DOC]
    # HELPER: shared body for the version-one matrix diag tests. Input: high_level (boolean).
    # Checks vectors of shape two-by-two, four and a batch of two-by-two-by-three produce the
    # expected diagonal matrices.
    # pyformat: disable
    vecs1 = np.array([[1, 2],
                      [3, 4]])
    solution1 = np.array([[[1, 0], [0, 2]],
                          [[3, 0], [0, 4]]])
    vecs2 = np.array([1, 2, 3, 4])
    solution2 = np.array([[1, 0, 0, 0],
                          [0, 2, 0, 0],
                          [0, 0, 3, 0],
                          [0, 0, 0, 4]])
    vecs3 = np.array([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7,  8,  9],  # pylint: disable=bad-whitespace
                       [10, 11, 12]]])
    solution3 = np.array([[[[1, 0, 0],
                            [0, 2, 0],
                            [0, 0, 3]],
                           [[4, 0, 0],
                            [0, 5, 0],
                            [0, 0, 6]]],
                          [[[7, 0, 0],
                            [0, 8, 0],
                            [0, 0, 9]],
                           [[10, 0, 0],
                            [0, 11, 0],
                            [0, 0, 12]]]])
    # pyformat: enable
    self._assertOpOutputMatchesExpected({"diagonal": vecs1}, solution1,
                                        high_level)
    self._assertOpOutputMatchesExpected({"diagonal": vecs2}, solution2,
                                        high_level)
    self._assertOpOutputMatchesExpected({"diagonal": vecs3}, solution3,
                                        high_level)

  def testV1(self):
    # [FUNCTION DOC]
    # Runs the generic version-one scenario through the high-level matrix diag op.
    self._testV1Level(True)

  def testV1LowLevel(self):
    # [FUNCTION DOC]
    # Runs the generic version-one scenario through the low-level generated matrix diag op.
    self._testV1Level(False)

  # From here onwards are v2-only tests.
  def testSquare(self):
    # [FUNCTION DOC]
    # Scenario: building one square matrix (first of the batch) from a single diagonal set for
    # every alignment and band; expects the first padded matrix.
    for align in alignment_list:
      for _, tests in [square_cases(align)]:
        for diag_index, (vecs, solution) in tests.items():
          params = {"diagonal": vecs[0], "k": diag_index, "align": align}
          self._assertOpOutputMatchesExpected(params, solution[0])

  def testSquareBatch(self):
    # [FUNCTION DOC]
    # Scenario: building a batch of square matrices from a batch of diagonals for every alignment
    # and band; expects the padded diagonal matrices.
    for align in alignment_list:
      for _, tests in [square_cases(align)]:
        for diag_index, (vecs, solution) in tests.items():
          params = {"diagonal": vecs, "k": diag_index, "align": align}
          self._assertOpOutputMatchesExpected(params, solution)

  def testRectangularBatch(self):
    # [FUNCTION DOC]
    # Scenario: building rectangular matrices from diagonals. First gives both the row and column
    # counts matching the solution shape (tall and fat cases, first alignment). Then, cycling
    # through the alignments, supplies only the row count or only the column count with a reshaped
    # expected result (truncated to the new size) and diagonals repacked for the new shape, and
    # checks the op infers the other dimension correctly.
    # Stores expected num_rows and num_cols (when the other is given).
    # expected[(d_lower, d_upper)] = (expected_num_rows, expected_num_cols)
    test_list = list()

    # Do not align the test cases here. Re-alignment needs to happen after the
    # solution shape is updated.
    # Square cases:
    expected = {
        (-1, -1): (5, 4),
        (-4, -3): (5, 2),
        (-2, 1): (5, 5),
        (2, 4): (3, 5),
    }
    test_list.append((expected, square_cases()))

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

    # Giving both num_rows and num_cols
    align = alignment_list[0]
    for _, tests in [tall_cases(align), fat_cases(align)]:
      for diag_index, (vecs, solution) in tests.items():
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs,
                "k": diag_index,
                "num_rows": solution.shape[-2],
                "num_cols": solution.shape[-1],
                "align": align
            }, solution)

    # We go through each alignment in a round-robin manner.
    align_index = 0

    # Giving just num_rows or num_cols.
    for expected, (_, tests) in test_list:
      for diag_index, (new_num_rows, new_num_cols) in expected.items():
        align = alignment_list[align_index]
        align_index = (align_index + 1) % len(alignment_list)
        vecs, solution = tests[diag_index]
        solution_given_num_rows = solution.take(
            indices=range(new_num_cols), axis=-1)
        # Repacks the diagonal input according to the new solution shape.
        vecs_given_num_rows = repack_diagonals(
            vecs,
            diag_index,
            solution_given_num_rows.shape[-2],
            new_num_cols,
            align=align)
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs_given_num_rows,
                "k": diag_index,
                "num_rows": solution_given_num_rows.shape[-2],
                "align": align
            }, solution_given_num_rows)
        solution_given_num_cols = solution.take(
            indices=range(new_num_rows), axis=-2)
        # Repacks the diagonal input according to the new solution shape.
        vecs_given_num_cols = repack_diagonals(
            vecs,
            diag_index,
            new_num_rows,
            solution_given_num_cols.shape[-1],
            align=align)
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs_given_num_cols,
                "k": diag_index,
                "num_cols": solution_given_num_cols.shape[-1],
                "align": align
            }, solution_given_num_cols)

  def testPadding(self):
    # [FUNCTION DOC]
    # Scenario: building diagonal matrices with a custom padding value (555 for the first
    # alignment, -11 for the second). Gives both row and column counts, the padding value and the
    # alignment; expects the padded matrix where zeros of the plain solution are replaced by the
    # padding value.
    for padding_value, align in zip_to_first_list_length([555, -11],
                                                         alignment_list):
      for _, tests in all_tests(align):
        for diag_index, (vecs, solution) in tests.items():
          mask = (solution == 0)
          solution = solution + (mask * padding_value)
          self._assertOpOutputMatchesExpected(
              {
                  "diagonal": vecs,
                  "k": diag_index,
                  "num_rows": solution.shape[-2],
                  "num_cols": solution.shape[-1],
                  "padding_value": padding_value,
                  "align": align
              }, solution)


class MatrixSetDiagTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the matrix set diag op (replacing the diagonal band of matrices, version
  # one and two semantics) on the XLA test device.

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     high_level=True,
                                     rtol=1e-3,
                                     atol=1e-5):
    # [FUNCTION DOC]
    # HELPER: verifies that the matrix set diag op produces the expected solution. Inputs: params
    # (mapping with the input matrix and the diagonal; both entries are overwritten with
    # placeholders), solution (expected numpy array), high_level (boolean, default true; picks the
    # high-level op or the low-level generated op), rtol (default 0.001), atol (default 0.00001).
    # For each numeric type except the 8-bit ones, runs the op in the test scope and asserts the
    # output dtype and values, with looser bfloat16 tolerance.
    """Verifies that matrix_set_diag produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_set_diag.
      solution: numpy array representing the expected output of matrix_set_diag.
      high_level: call high_level matrix_set_diag
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    input = params["input"]  # pylint: disable=redefined-builtin
    diagonal = params["diagonal"]
    with self.session() as session:
      for dtype in self.numeric_types - {np.int8, np.uint8}:
        expected = solution.astype(dtype)
        with self.test_scope():
          params["input"] = array_ops.placeholder(
              dtype, input.shape, name="input")
          params["diagonal"] = array_ops.placeholder(
              dtype, diagonal.shape, name="diagonal")
          if high_level:
            # wraps gen_array_ops.matrix_set_diag_v3
            output = array_ops.matrix_set_diag(**params)
          else:
            # TODO(b/201086188): Remove this case once MatrixDiag V1 is removed.
            output = gen_array_ops.matrix_set_diag(**params)
        result = session.run(
            output, {
                params["input"]: input.astype(dtype),
                params["diagonal"]: diagonal.astype(dtype)
            })
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)

  # Generic tests applicable to both v1 and v2 ops.
  # Originally from binary_ops_tests.py.
  def _testV1Level(self, high_level):
    # [FUNCTION DOC]
    # HELPER: shared body for the version-one set diag tests. Input: high_level (boolean). Builds
    # square and rectangular cases, with and without batch dimension, where input matrices have
    # their main diagonal replaced by the given vector, and verifies each with the shared checker.
    test_cases = list()

    # pyformat: disable
    # pylint: disable=bad-whitespace
    # Square cases.
    input = np.array([[0, 1, 0],  # pylint: disable=redefined-builtin
                      [1, 0, 1],
                      [1, 1, 1]])
    diag = np.array([1, 2, 3])
    solution = np.array([[1, 1, 0],
                         [1, 2, 1],
                         [1, 1, 3]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))

    input = np.array([[[1, 0, 3],
                       [0, 2, 0],
                       [1, 0, 3]],
                      [[4, 0, 4],
                       [0, 5, 0],
                       [2, 0, 6]]])
    diag = np.array([[-1,  0, -3],
                     [-4, -5, -6]])
    solution = np.array([[[-1, 0,  3],
                          [ 0, 0,  0],
                          [ 1, 0, -3]],
                         [[-4,  0,  4],
                          [ 0, -5,  0],
                          [ 2,  0, -6]]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))

    # Rectangular cases.
    input = np.array([[0, 1, 0],
                      [1, 0, 1]])
    diag = np.array([3, 4])
    solution = np.array([[3, 1, 0],
                         [1, 4, 1]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))

    input = np.array([[0, 1],
                      [1, 0],
                      [1, 1]])
    diag = np.array([3, 4])
    solution = np.array([[3, 1],
                         [1, 4],
                         [1, 1]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))

    input = np.array([[[1, 0, 3],
                       [0, 2, 0]],
                      [[4, 0, 4],
                       [0, 5, 0]]])
    diag = np.array([[-1, -2], [-4, -5]])
    solution = np.array([[[-1,  0, 3],
                          [ 0, -2, 0]],
                         [[-4,  0, 4],
                          [ 0, -5, 0]]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))
    # pylint: enable=bad-whitespace
    # pyformat: enable

    for test in test_cases:
      self._assertOpOutputMatchesExpected(test[0], test[1], high_level)

  def testV1(self):
    # [FUNCTION DOC]
    # Runs the generic version-one scenario through the high-level matrix set diag op.
    self._testV1Level(True)

  def testV1LowLevel(self):
    # [FUNCTION DOC]
    # Runs the generic version-one scenario through the low-level generated matrix set diag op.
    self._testV1Level(False)

  # From here onwards are v2-only tests.
  def testSingleMatrix(self):
    # [FUNCTION DOC]
    # Scenario: set diagonals on a single matrix (first of the batch), same construction as the
    # batch test but using one matrix and its first compact diagonals entry.
    for align in alignment_list:
      for _, tests in all_tests(align):
        for diag_index, (vecs, banded_mat) in tests.items():
          mask = (banded_mat[0] == 0)
          input_mat = np.random.randint(10, size=mask.shape)
          solution = input_mat * mask + banded_mat[0]
          self._assertOpOutputMatchesExpected(
              {
                  "input": input_mat,
                  "diagonal": vecs[0],
                  "k": diag_index,
                  "align": align
              }, solution)

  def testBatch(self):
    # [FUNCTION DOC]
    # Scenario: set diagonals on a batch. Builds random integer input matrices, masks out the band
    # positions and adds the padded band values to form the expected result; checks the set diag
    # op for every alignment, case set and band.
    for align in alignment_list:
      for _, tests in all_tests(align):
        for diag_index, (vecs, banded_mat) in tests.items():
          mask = (banded_mat == 0)
          input_mat = np.random.randint(10, size=mask.shape)
          solution = input_mat * mask + banded_mat
          self._assertOpOutputMatchesExpected(
              {
                  "input": input_mat,
                  "diagonal": vecs,
                  "k": diag_index,
                  "align": align
              }, solution)


class MatrixDiagPartTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the matrix diag part op (extracting diagonals, version one and two
  # semantics with band, alignment and padding value) on the XLA test device.

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     high_level=True,
                                     rtol=1e-3,
                                     atol=1e-5):
    # [FUNCTION DOC]
    # HELPER: verifies that the matrix diag part op produces the expected solution.
    # KNOWN ISSUE: after the branch that picks the high-level or low-level op, the output is
    # recomputed unconditionally with the high-level op, so the low-level branch result is
    # discarded and the high_level flag has no effect.
    # Inputs: params (mapping of op arguments, must contain the input matrix; its input entry is
    # overwritten with a placeholder), solution (expected numpy array), high_level (boolean,
    # default true), rtol (default 0.001) and atol (default 0.00001). For each numeric type except
    # the 8-bit ones, builds the op in the test scope, runs it feeding the data cast to that type
    # and asserts the output dtype and values (with looser bfloat16 tolerance). The docstring
    # wording about matrix_set_diag for high_level is a copy-paste slip.
    """Verifies that matrix_diag_part produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_diag_part.
      solution: numpy array representing the expected output.
      high_level: call high_level matrix_set_diag
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    input = params["input"]  # pylint: disable=redefined-builtin
    with self.session() as session:
      for dtype in self.numeric_types - {np.int8, np.uint8}:
        expected = solution.astype(dtype)
        with self.test_scope():
          params["input"] = array_ops.placeholder(
              dtype, input.shape, name="input")
          if high_level:
            # wraps gen_array_ops.matrix_diag_part_v3
            output = array_ops.matrix_diag_part(**params)
          else:
            # TODO(b/201086188): Remove this case once MatrixDiag V1 is removed.
            output = gen_array_ops.matrix_diag_part(**params)
          output = array_ops.matrix_diag_part(**params)
        result = session.run(output, {
            params["input"]: input.astype(dtype),
        })
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)

  # Generic tests applicable to both v1 and v2 ops.
  # Originally from unary_ops_tests.py.
  def _testV1Level(self, high_level):
    # [FUNCTION DOC]
    # HELPER: shared body for the version-one diag part tests. Input: high_level (boolean choosing
    # the high-level or the low-level op). Uses a batch of three two-by-four matrices holding
    # consecutive numbers and expects the main diagonals [0,5], [8,13], [16,21].
    matrices = np.arange(3 * 2 * 4).reshape([3, 2, 4])
    solution = np.array([[0, 5], [8, 13], [16, 21]])
    self._assertOpOutputMatchesExpected({"input": matrices}, solution,
                                        high_level)

  def testV1(self):
    # [FUNCTION DOC]
    # Runs the generic version-one scenario through the high-level matrix diag part op.
    self._testV1Level(True)

  def testV1LowLevel(self):
    # [FUNCTION DOC]
    # Runs the generic version-one scenario through the low-level generated matrix diag part op.
    self._testV1Level(False)

  # From here onwards are v2-only tests.
  def testSingleMatrix(self):
    # [FUNCTION DOC]
    # Scenario: diagonal extraction from a single matrix (first of the batch). For every alignment
    # and each case set and band, expects the first compact diagonals entry as the result.
    for align in alignment_list:
      test_list = [square_cases(align), tall_cases(align), fat_cases(align)]
      for mat, tests in test_list:
        for diag_index, (solution, _) in tests.items():
          self._assertOpOutputMatchesExpected(
              {
                  "input": mat[0],
                  "k": diag_index,
                  "align": align
              }, solution[0])

  def testBatch(self):
    # [FUNCTION DOC]
    # Scenario: batched diagonal extraction. For every alignment and every case set (square, tall,
    # fat), feeds the full batch of matrices with each diagonal band and alignment, and expects the
    # compact diagonals as the result. Checks dtype and values across numeric types.
    for align in alignment_list:
      for mat, tests in all_tests(align):
        for diag_index, (solution, _) in tests.items():
          self._assertOpOutputMatchesExpected(
              {
                  "input": mat,
                  "k": diag_index,
                  "align": align
              }, solution)

  def testPadding(self):
    # [FUNCTION DOC]
    # Scenario: extract diagonals with a custom padding value (555 for the first alignment, -11 for
    # the second; the list is padded with null values if shorter). Expected result is the compact
    # diagonals with the padding value substituted wherever the compact entries are zero, for all
    # cases, passing the padding value and alignment. Checks dtype and values across numeric types.
    for padding_value, align in zip_to_first_list_length([555, -11],
                                                         alignment_list):
      for mat, tests in all_tests(align):
        for diag_index, (solution, _) in tests.items():
          mask = (solution == 0)
          solution = solution + (mask * padding_value)
          self._assertOpOutputMatchesExpected(
              {
                  "input": mat,
                  "k": diag_index,
                  "padding_value": padding_value,
                  "align": align
              }, solution)


if __name__ == "__main__":
  googletest.main()
