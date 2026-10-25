# [MODULE DOC]
# Tests for the candidate sampling ops. Uses the all-candidates sampler (which returns every class
# with expected count one) to check sampled candidates, expected counts and accidental hits, and the
# log-uniform sampler to check seed determinism and out-of-range label errors. Running the file as a
# script starts the test runner.
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
"""Tests for CandidateSamplerOp."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RangeSamplerOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the range sampler (candidate sampling) ops. Class constants fix a batch of three
  # examples with two true labels each, a class range of five, as many sampled candidates as the
  # range, and the true label matrix used by all tests.

  BATCH_SIZE = 3
  NUM_TRUE = 2
  RANGE = 5
  NUM_SAMPLED = RANGE

  TRUE_LABELS = [[1, 2], [0, 4], [3, 3]]

  @test_util.run_deprecated_v1
  def testTrueCandidates(self):
    # [FUNCTION DOC]
    # Test (graph mode only): evaluates a constant index vector and a constant true-candidate vector
    # reshaped to batch by num-true, and asserts they equal the expected indices and the class true
    # label matrix. It does not call a sampler.
    with self.cached_session() as sess:
      indices = constant_op.constant([0, 0, 1, 1, 2, 2])
      true_candidates_vec = constant_op.constant([1, 2, 0, 4, 3, 3])
      true_candidates_matrix = array_ops.reshape(
          true_candidates_vec, [self.BATCH_SIZE, self.NUM_TRUE])
      indices_val, true_candidates_val = sess.run(
          [indices, true_candidates_matrix])

    self.assertAllEqual(indices_val, [0, 0, 1, 1, 2, 2])
    self.assertAllEqual(true_candidates_val, self.TRUE_LABELS)

  def testSampledCandidates(self):
    # [FUNCTION DOC]
    # Test: runs the all-candidate sampler on the true labels and asserts the sampled candidates are
    # every class id from zero to four in order, with a static shape of the number sampled.
    with self.cached_session():
      true_classes = constant_op.constant(
          [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
      sampled_candidates, _, _ = candidate_sampling_ops.all_candidate_sampler(
          true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
      result = self.evaluate(sampled_candidates)

    expected_ids = [0, 1, 2, 3, 4]
    self.assertAllEqual(result, expected_ids)
    self.assertEqual(sampled_candidates.get_shape(), [self.NUM_SAMPLED])

  def testTrueLogExpectedCount(self):
    # [FUNCTION DOC]
    # Test: runs the all-candidate sampler and takes the log of the true expected counts. Asserts the
    # logs are all zero (expected count one) and that both the counts and logs have static shape of
    # batch size by num true.
    with self.cached_session():
      true_classes = constant_op.constant(
          [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
      _, true_expected_count, _ = candidate_sampling_ops.all_candidate_sampler(
          true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
      true_log_expected_count = math_ops.log(true_expected_count)
      result = self.evaluate(true_log_expected_count)

    self.assertAllEqual(result, [[0.0] * self.NUM_TRUE] * self.BATCH_SIZE)
    self.assertEqual(true_expected_count.get_shape(),
                     [self.BATCH_SIZE, self.NUM_TRUE])
    self.assertEqual(true_log_expected_count.get_shape(),
                     [self.BATCH_SIZE, self.NUM_TRUE])

  def testSampledLogExpectedCount(self):
    # [FUNCTION DOC]
    # Test: runs the all-candidate sampler and takes the log of the sampled expected counts. Asserts
    # all logs are zero and that the counts and logs have static shape of the number sampled.
    with self.cached_session():
      true_classes = constant_op.constant(
          [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
      _, _, sampled_expected_count = candidate_sampling_ops.all_candidate_sampler(  # pylint: disable=line-too-long
          true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
      sampled_log_expected_count = math_ops.log(sampled_expected_count)
      result = self.evaluate(sampled_log_expected_count)

    self.assertAllEqual(result, [0.0] * self.NUM_SAMPLED)
    self.assertEqual(sampled_expected_count.get_shape(), [self.NUM_SAMPLED])
    self.assertEqual(sampled_log_expected_count.get_shape(), [self.NUM_SAMPLED])

  def testAccidentalHits(self):
    # [FUNCTION DOC]
    # Test: samples all candidates, then computes accidental hits against the true labels. Asserts the
    # three outputs (indices, ids, weights) are rank 1, that every hit id is one of the true labels of
    # its example, and that every weight is below minus 1e37 (a large negative mask value).
    with self.cached_session() as sess:
      true_classes = constant_op.constant(
          [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
      sampled_candidates, _, _ = candidate_sampling_ops.all_candidate_sampler(
          true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
      accidental_hits = candidate_sampling_ops.compute_accidental_hits(
          true_classes, sampled_candidates, self.NUM_TRUE)
      indices, ids, weights = self.evaluate(accidental_hits)

    self.assertEqual(1, accidental_hits[0].get_shape().ndims)
    self.assertEqual(1, accidental_hits[1].get_shape().ndims)
    self.assertEqual(1, accidental_hits[2].get_shape().ndims)
    for index, id_, weight in zip(indices, ids, weights):
      self.assertTrue(id_ in self.TRUE_LABELS[index])
      self.assertLess(weight, -1.0e37)

  @test_util.run_deprecated_v1
  def testSeed(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment says a seed of zero means random seeds, but the test actually passes no
    # seed (nothing) for the random case.
    # Test (graph mode only): with the log-uniform sampler, asserts that a fixed non-zero seed gives
    # identical draws on repeat, and that with no seed ten pairs of draws are the same at most twice
    # (allowing rare coincidences).

    def draw(seed):
      # [FUNCTION DOC]
      # HELPER: draws one sample set for testSeed.
      # Takes a seed (a number or nothing). Within a cached session, runs the log-uniform candidate
      # sampler on the fixed true labels (unique sampling, range max five) and returns the evaluated
      # sampled candidates as an array.
      with self.cached_session():
        true_classes = constant_op.constant(
            [[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
        sampled, _, _ = candidate_sampling_ops.log_uniform_candidate_sampler(
            true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True, 5, seed=seed)
        return self.evaluate(sampled)

    # Non-zero seed. Repeatable.
    for seed in [1, 12, 123, 1234]:
      self.assertAllEqual(draw(seed), draw(seed))
    # Seed=0 means random seeds.
    num_same = 0
    for _ in range(10):
      if np.allclose(draw(None), draw(None)):
        num_same += 1
    # Accounts for the fact that the same random seed may be picked
    # twice very rarely.
    self.assertLessEqual(num_same, 2)

  def testCandidateOutOfRange(self):
    # [FUNCTION DOC]
    # Test: runs the log-uniform sampler with a true label above the range maximum, then one below
    # zero. Each must raise a value error or invalid-argument error mentioning out of range.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "out of range"):
      self.evaluate(
          candidate_sampling_ops.log_uniform_candidate_sampler(
              true_classes=[[0, 10]],
              num_true=2,
              num_sampled=1000,
              unique=False,
              range_max=2))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "out of range"):
      self.evaluate(
          candidate_sampling_ops.log_uniform_candidate_sampler(
              true_classes=[[0, -10]],
              num_true=2,
              num_sampled=1000,
              unique=False,
              range_max=2))


if __name__ == "__main__":
  test.main()
