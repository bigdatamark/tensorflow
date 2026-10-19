# [MODULE DOC]
# Tests for the edit distance (Levenshtein distance between sparse sequences) operation exposed as
# the array-ops edit_distance function and its raw generated op.
# Defines a small constant-conversion helper and one test class that checks normalized and
# unnormalized distances, string inputs, rank 3 inputs, empty hypotheses or truths, invalid indices
# and invalid (rank zero) shapes. No import-time side effects; runs the TensorFlow test runner when
# executed as a script.
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
"""Tests for tensorflow.kernels.edit_distance_op."""

import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def ConstantOf(x):
  # [FUNCTION DOC]
  # HELPER: converts nested Python lists or arrays into constant tensors for the edit distance tests.
  # Takes any array-like value. Converts it to a numpy array and, unless it holds byte or unicode
  # strings, casts it to 64-bit integers. Returns the resulting constant tensor.
  x = np.asarray(x)
  # Convert to int64 if it's not a string or unicode
  if x.dtype.char not in "SU":
    x = np.asarray(x, dtype=np.int64)
  return constant_op.constant(x)


class EditDistanceTest(test.TestCase):
  # [CLASS DOC]
  # Tests for edit_distance on sparse hypothesis and truth inputs. Provides two helpers, one working on
  # sparse tensors and one building sparse inputs from (indices, values, shape) tuples.

  def _testEditDistanceST(self,
                          hypothesis_st,
                          truth_st,
                          normalize,
                          expected_output,
                          expected_shape,
                          expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: core check used by the edit distance tests.
    # Takes hypothesis_st and truth_st (sparse tensors), normalize flag, expected_output (expected
    # distances), expected_shape (expected static shape) and optional expected_err_re (regular expression
    # of an expected operation error). Computes edit_distance. If no error is expected, asserts the static
    # shape and that evaluated values are close to the expected output; otherwise asserts that evaluating
    # raises an operation error matching the expression.
    edit_distance = array_ops.edit_distance(
        hypothesis=hypothesis_st, truth=truth_st, normalize=normalize)

    if expected_err_re is None:
      self.assertEqual(edit_distance.get_shape(), expected_shape)
      output = self.evaluate(edit_distance)
      self.assertAllClose(output, expected_output)
    else:
      with self.assertRaisesOpError(expected_err_re):
        self.evaluate(edit_distance)

  def _testEditDistance(self,
                        hypothesis,
                        truth,
                        normalize,
                        expected_output,
                        expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: builds sparse inputs and runs the core check in graph mode twice.
    # Takes hypothesis and truth (tuples of indices, values and dense shape), normalize flag,
    # expected_output and optional expected_err_re. Derives the expected static shape as the element-wise
    # maximum of the hypothesis and truth dense shapes without the last dimension. Runs the check once with
    # sparse tensor value inputs and once with sparse tensor inputs, each in a fresh graph and session.
    # Shape inference figures out the shape from the shape variables
    # Explicit tuple() needed since zip returns an iterator in Python 3.
    expected_shape = [
        max(h, t) for h, t in tuple(zip(hypothesis[2], truth[2]))[:-1]
    ]

    # SparseTensorValue inputs.
    with ops.Graph().as_default() as g, self.session(g):
      # hypothesis and truth are (index, value, shape) tuples
      self._testEditDistanceST(
          hypothesis_st=sparse_tensor.SparseTensorValue(
              *[ConstantOf(x) for x in hypothesis]),
          truth_st=sparse_tensor.SparseTensorValue(
              *[ConstantOf(x) for x in truth]),
          normalize=normalize,
          expected_output=expected_output,
          expected_shape=expected_shape,
          expected_err_re=expected_err_re)

    # SparseTensor inputs.
    with ops.Graph().as_default() as g, self.session(g):
      # hypothesis and truth are (index, value, shape) tuples
      self._testEditDistanceST(
          hypothesis_st=sparse_tensor.SparseTensor(
              *[ConstantOf(x) for x in hypothesis]),
          truth_st=sparse_tensor.SparseTensor(*[ConstantOf(x) for x in truth]),
          normalize=normalize,
          expected_output=expected_output,
          expected_shape=expected_shape,
          expected_err_re=expected_err_re)

  def testEditDistanceNormalized(self):
    # [FUNCTION DOC]
    # Two-row example with normalization on; expects distances one and one half.
    hypothesis_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
    hypothesis_values = [0, 1, 1, -1]
    hypothesis_shape = [2, 2]
    truth_indices = [[0, 0], [1, 0], [1, 1]]
    truth_values = [0, 1, 1]
    truth_shape = [2, 2]
    expected_output = [1.0, 0.5]

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceUnnormalized(self):
    # [FUNCTION DOC]
    # Two-row example with normalization off and different lengths; expects distances two and two.
    hypothesis_indices = [[0, 0], [1, 0], [1, 1]]
    hypothesis_values = [10, 10, 11]
    hypothesis_shape = [2, 2]
    truth_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
    truth_values = [1, 2, 1, -1]
    truth_shape = [2, 3]
    expected_output = [2.0, 2.0]

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=False,
        expected_output=expected_output)

  def testEditDistanceProperDistance(self):
    # [FUNCTION DOC]
    # Uses character strings ("algorithm" and "altruistic") stored as string values in the sparse tensors.
    # Expects an unnormalized distance of six per row and normalized distances of six divided by the truth
    # length when normalization is on.
    # In this case, the values are individual characters stored in the
    # SparseTensor (type DT_STRING)
    hypothesis_indices = ([[0, i] for i, _ in enumerate("algorithm")] +
                          [[1, i] for i, _ in enumerate("altruistic")])
    hypothesis_values = [x for x in "algorithm"] + [x for x in "altruistic"]
    hypothesis_shape = [2, 11]
    truth_indices = ([[0, i] for i, _ in enumerate("altruistic")] +
                     [[1, i] for i, _ in enumerate("algorithm")])
    truth_values = [x for x in "altruistic"] + [x for x in "algorithm"]
    truth_shape = [2, 11]
    expected_unnormalized = [6.0, 6.0]
    expected_normalized = [6.0 / len("altruistic"), 6.0 / len("algorithm")]

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=False,
        expected_output=expected_unnormalized)

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_normalized)

  def testEditDistance3D(self):
    # [FUNCTION DOC]
    # Rank 3 inputs with normalization on. Expects infinity where the truth is empty, one where the
    # hypothesis is empty, zero for an exact match.
    hypothesis_indices = [[0, 0, 0], [1, 0, 0]]
    hypothesis_values = [0, 1]
    hypothesis_shape = [2, 1, 1]
    truth_indices = [[0, 1, 0], [1, 0, 0], [1, 1, 0]]
    truth_values = [0, 1, 1]
    truth_shape = [2, 2, 1]
    expected_output = [
        [np.inf, 1.0],  # (0,0): no truth, (0,1): no hypothesis
        [0.0, 1.0]
    ]  # (1,0): match,    (1,1): no hypothesis

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceZeroLengthHypothesis(self):
    # [FUNCTION DOC]
    # Empty hypothesis against a one-element truth, normalized; expects a distance of one.
    hypothesis_indices = np.empty((0, 2), dtype=np.int64)
    hypothesis_values = []
    hypothesis_shape = [1, 0]
    truth_indices = [[0, 0]]
    truth_values = [0]
    truth_shape = [1, 1]
    expected_output = [1.0]

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceZeroLengthTruth(self):
    # [FUNCTION DOC]
    # One-element hypothesis against an empty truth, normalized; expects infinity (division by zero).
    hypothesis_indices = [[0, 0]]
    hypothesis_values = [0]
    hypothesis_shape = [1, 1]
    truth_indices = np.empty((0, 2), dtype=np.int64)
    truth_values = []
    truth_shape = [1, 0]
    expected_output = [np.inf]  # Normalized, loss is 1/0 = inf

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceZeroLengthHypothesisAndTruth(self):
    # [FUNCTION DOC]
    # Both sequences empty, normalized; expects zero because the sequences match exactly.
    hypothesis_indices = np.empty((0, 2), dtype=np.int64)
    hypothesis_values = []
    hypothesis_shape = [1, 0]
    truth_indices = np.empty((0, 2), dtype=np.int64)
    truth_values = []
    truth_shape = [1, 0]
    expected_output = [0]  # Normalized is 0 because of exact match

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceBadIndices(self):
    # [FUNCTION DOC]
    # Supplies huge negative indices and mismatching shapes with normalization off. The expected output is
    # a dummy; the test asserts that evaluation raises an operation error about an inner product that would
    # write outside the output buffer, or about a negative dimension.
    hypothesis_indices = np.full((3, 3), -1250999896764, dtype=np.int64)
    hypothesis_values = np.zeros(3, dtype=np.int64)
    hypothesis_shape = np.zeros(3, dtype=np.int64)
    truth_indices = np.full((3, 3), -1250999896764, dtype=np.int64)
    truth_values = np.full([3], 2, dtype=np.int64)
    truth_shape = np.full([3], 2, dtype=np.int64)
    expected_output = []  # dummy; ignored

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=False,
        expected_output=expected_output,
        expected_err_re=(r"inner product -\d+ which would require writing "
                         "to outside of the buffer for the output tensor|"
                         r"Dimension -\d+ must be >= 0"))

  def testEmptyShapeWithEditDistanceRaisesError(self):
    # [FUNCTION DOC]
    # Feeds sparse inputs whose dense shape is empty (rank zero) in three ways and expects errors saying
    # the shapes must have rank at least 2: the raw op called eagerly (invalid-argument or value error),
    # the raw op inside a compiled function (value error at tracing), and the Python wrapper path through
    # the helper (value error).
    para = {
        "hypothesis_indices": [[]],
        "hypothesis_values": ["tmp/"],
        "hypothesis_shape": [],
        "truth_indices": [[]],
        "truth_values": [""],
        "truth_shape": [],
        "normalize": False,
    }

    # Check edit distance raw op with empty shape in eager mode.
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        (
            r"Input Hypothesis SparseTensors must have rank at least 2, but"
            " hypothesis_shape rank is: 0|Input SparseTensors must have rank "
            "at least 2, but truth_shape rank is: 0"
        ),
    ):
      array_ops.gen_array_ops.EditDistance(**para)

    # Check raw op with tf.function
    @def_function.function
    def TestFunction():
      # [FUNCTION DOC]
      # HELPER: nested function wrapped as a compiled function for the enclosing test.
      # Takes no arguments and returns nothing; calls the raw edit distance op with the
      # empty-shape parameters so the shape validation error is raised while it is traced.
      """Wrapper function for edit distance call."""
      array_ops.gen_array_ops.EditDistance(**para)

    with self.assertRaisesRegex(
        ValueError,
        (
            "Input Hypothesis SparseTensors must have rank at least 2, but"
            " hypothesis_shape rank is: 0"
        ),
    ):
      TestFunction()

    # Check with python wrapper API
    hypothesis_indices = [[]]
    hypothesis_values = [0]
    hypothesis_shape = []
    truth_indices = [[]]
    truth_values = [1]
    truth_shape = []
    expected_output = []  # dummy ignored

    with self.assertRaisesRegex(
        ValueError,
        (
            "Input Hypothesis SparseTensors must have rank at least 2, but"
            " hypothesis_shape rank is: 0"
        ),
    ):
      self._testEditDistance(
          hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
          truth=(truth_indices, truth_values, truth_shape),
          normalize=False,
          expected_output=expected_output,
      )


if __name__ == "__main__":
  test.main()
