# [MODULE DOC]
# Test module for the approximate top-k operations (approximate max-k and min-k) compiled with
# XLA just-in-time compilation on a TPU device. Tests build random databases, run the approximate
# operation, and require the recall against an exact numpy sort to be at least 0.95; they also
# cover maximum inner product search, L2 nearest neighbor search, a multi-dimensional reduction,
# gradients and multiple ops in one function. Exposes no public API; running as a script starts
# the test runner. No import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for approx_max_k and approx_min_k."""

import itertools
from absl.testing import parameterized

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import test
from tensorflow.python.eager.def_function import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables


class ApproxTopkTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for approximate top-k. Holds a seeded random number generator
  # (fixed seed 42) created in setUp, and a recall helper. Tests run on the first TPU device and
  # therefore require TPU hardware.

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the base test setup, then creates the per-test random number generator with fixed
    # seed 42 and stores it for generating deterministic test data. Returns nothing.
    test.TestCase.setUp(self)
    self._rng = np.random.default_rng(42)

  def compute_recall(self, result_neighbors, ground_truth_neighbors):
    # [FUNCTION DOC]
    # HELPER: recall metric used by the approximate top-k tests.
    # Takes two two-dimensional integer arrays of dataset indices (approximate results and
    # exact ground truth) with the same number of rows; asserts both are two-dimensional and
    # have equal row counts. For each row counts how many result indices appear in that row's
    # ground-truth set, and returns the total hits divided by the number of ground-truth entries.
    """Computes the recall of an approximate nearest neighbor search.

    Args:
      result_neighbors: int32 numpy array of the shape [num_queries,
        neighbors_per_query] where the values are the indices of the dataset.
      ground_truth_neighbors: int32 numpy array of with shape [num_queries,
        ground_truth_neighbors_per_query] where the values are the indices of
        the dataset.

    Returns:
      The recall.
    """
    self.assertLen(result_neighbors.shape, 2)
    self.assertLen(ground_truth_neighbors.shape, 2)
    self.assertEqual(result_neighbors.shape[0], ground_truth_neighbors.shape[0])
    gt_sets = [set(np.asarray(x)) for x in ground_truth_neighbors]

    def hits_per_q(q, nn_per_q):
      # [FUNCTION DOC]
      # HELPER: nested counter for compute_recall. Given a query row number and that row's
      # result indices, returns how many of them are in the ground-truth set of that row.
      return len(list(x for x in nn_per_q if x.item() in gt_sets[q]))

    hits = sum(
        hits_per_q(q, nn_per_q) for q, nn_per_q in enumerate(result_neighbors))
    return hits / ground_truth_neighbors.size

  @parameterized.parameters(
      itertools.product(
          [1, 10],  # k
          [100, 500],  # row_size
          [1, 10, 128],  # num_rows
          [True, False],  # aggregate_to_topk
      ))
  def test_non_fused_max_k(self, k, row_size, num_rows, aggregate_to_topk):
    # [FUNCTION DOC]
    # Parameterized over k, row size, number of rows and whether to aggregate to top-k. Builds a
    # database where each row is a random permutation of distinct floats, runs approximate max-k
    # (jit compiled, on TPU) and compares returned indices with the exact top-k by sorting.
    # Asserts recall is at least 0.95.
    row = np.arange(row_size, dtype=np.float32)
    db = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))

    @function(jit_compile=True)
    def ann(db, k):
      # [FUNCTION DOC]
      # Jit-compiled wrapper returning the approximate max-k result (values and indices) of db
      # with the enclosing test's aggregate-to-top-k setting.
      return nn_ops.approx_max_k(db, k, aggregate_to_topk=aggregate_to_topk)

    with ops.device('/device:TPU:0'):
      db_op = variables.Variable(db)
      result = ann(db_op, k)[1]

    gt = np.argsort(-db)[:, :k]
    ann_recall = self.compute_recall(result.numpy(), gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [1, 10],  # k
          [100, 500],  # row_size
          [1, 10, 128],  # num_rows
          [True, False],  # aggregate_to_topk
      ))
  def test_non_fused_min_k(self, k, row_size, num_rows, aggregate_to_topk):
    # [FUNCTION DOC]
    # Same as the max-k test but for approximate min-k: rows are random permutations, the
    # smallest k entries per row are requested, and the indices are compared with an exact
    # ascending sort. Asserts recall is at least 0.95.
    # Use the new rng api
    row = np.arange(row_size, dtype=np.float32)
    db = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))

    @function(jit_compile=True)
    def ann(db, k=10):
      # [FUNCTION DOC]
      # Jit-compiled wrapper returning the approximate min-k result of db; k defaults to 10 but
      # the test always passes it. Uses the enclosing aggregate-to-top-k setting.
      return nn_ops.approx_min_k(db, k, aggregate_to_topk=aggregate_to_topk)

    with ops.device('/device:TPU:0'):
      db_op = variables.Variable(db)
      result = ann(db_op, k)[1]

    gt = np.argsort(db)[:, :k]
    ann_recall = self.compute_recall(result.numpy(), gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [1, 10],  # k
          [100, 500],  # db_size
          [1, 10, 128],  # qy_size
          [2, 32],  # feature dim
      ))
  # MIPS = Maximal Inner Product Search
  def test_mips(self, k, db_size, qy_size, feature_dim):
    # [FUNCTION DOC]
    # Maximum inner product search test. Parameterized over k, database size, query count and
    # feature dimension. Generates random queries and database, computes scores as query-database
    # inner products and finds approximate max-k per query on TPU. Ground truth is the exact
    # ranking computed from negated scores; asserts recall at least 0.95.
    qy = self._rng.random([qy_size, feature_dim], dtype=np.float32)
    db = self._rng.random([db_size, feature_dim], dtype=np.float32)

    @function(jit_compile=True)
    def ann(qy, db, k):
      # [FUNCTION DOC]
      # Jit-compiled search: multiplies queries by the transposed database to get scores and
      # returns the approximate max-k of the scores.
      scores = math_ops.matmul(qy, db, transpose_b=True)
      return nn_ops.approx_max_k(scores, k)

    with ops.device('/device:TPU:0'):
      qy_op = variables.Variable(qy)
      db_op = variables.Variable(db)
      result = ann(qy_op, db_op, k)[1]
      scores = -math_ops.matmul(qy_op, db_op, transpose_b=True)

    gt = np.argsort(scores.numpy())[:, :k]
    ann_recall = self.compute_recall(result.numpy(), gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [1, 10],  # k
          [100, 500],  # db_size
          [10, 128],  # qy_size
          [2, 8],  # feature dim
      ))
  # L2ANN = Approximate Nearest Neighbor search in the L2 metric space
  def test_l2ann(self, k, db_size, qy_size, feature_dim):
    # [FUNCTION DOC]
    # Approximate nearest neighbor test in L2 space. Parameterized over k, database size, query
    # count and feature dimension. Precomputes half the squared norm of each database row, scores
    # each pair as that value minus the inner product (equivalent to L2 distance ranking), and
    # runs approximate min-k on TPU. Ground truth is the exact ascending ranking of the same
    # scores; asserts recall at least 0.95.
    qy = self._rng.random([qy_size, feature_dim], dtype=np.float32)
    db = self._rng.random([db_size, feature_dim], dtype=np.float32)
    db_half_norm_sq = np.linalg.norm(db, axis=1)**2 / 2

    @function(jit_compile=True)
    def ann(qy, db, db_half_norm_sq, k):
      # [FUNCTION DOC]
      # Jit-compiled search: scores are half squared norms minus the query-database inner
      # products; returns the approximate min-k of those scores.
      scores = db_half_norm_sq - math_ops.matmul(qy, db, transpose_b=True)
      return nn_ops.approx_min_k(scores, k)

    with ops.device('/device:TPU:0'):
      qy_op = variables.Variable(qy)
      db_op = variables.Variable(db)
      db_half_norm_sq_op = variables.Variable(db_half_norm_sq)
      result = ann(qy_op, db_op, db_half_norm_sq_op, k)[1]
      scores = db_half_norm_sq_op - math_ops.matmul(
          qy_op, db_op, transpose_b=True)

    gt = np.argsort(scores.numpy())[:, :k]
    ann_recall = self.compute_recall(result.numpy(), gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  def test_highdim(self):
    # [FUNCTION DOC]
    # Tests approximate min-k on a four-dimensional random array reducing along dimension 2 with
    # k of 5. Exact ground truth is a sort along that axis. Both result and ground truth are
    # transposed and flattened into rows of k entries, then asserts recall at least 0.95.
    db = self._rng.random([2, 10, 200, 3], dtype=np.float32)
    k = 5

    @function(jit_compile=True)
    def ann(db, k):
      # [FUNCTION DOC]
      # Jit-compiled wrapper returning approximate min-k of db along reduction dimension 2.
      return nn_ops.approx_min_k(db, k=k, reduction_dimension=2)

    with ops.device('/device:TPU:0'):
      db_op = variables.Variable(db)
      result = ann(db_op, k)[1]

    gt = np.argsort(db, axis=2)[:, :, :k, :]
    flat_idx = np.reshape(
        np.transpose(result.numpy(), [0, 1, 3, 2]), [2 * 10 * 3, k])
    flat_gt = np.reshape(np.transpose(gt, [0, 1, 3, 2]), [2 * 10 * 3, k])
    ann_recall = self.compute_recall(flat_idx, flat_gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [dtypes.bfloat16, dtypes.float16, dtypes.float32],
          [1, 10],  # k
          [100, 500],  # row_size
          [1, 10, 128],  # num_rows
      ))
  def test_gradients(self, dtype, k, row_size, num_rows):
    # [FUNCTION DOC]
    # Parameterized over dtype (bfloat16, float16, float32), k, row size and row count. Checks
    # the gradient of approximate max-k values: the gradient with respect to the database must
    # equal the upstream gradients scattered to the positions of the selected indices (zero
    # elsewhere). Both are computed on TPU and compared with default closeness tolerances.
    row = np.arange(row_size, dtype=np.float32)
    db = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))
    out_grads = self._rng.random([num_rows, k])

    @function(jit_compile=True)
    def ann_with_grads(db, out_grads):
      # [FUNCTION DOC]
      # Jit-compiled: records approximate max-k on a gradient tape, takes the gradient of the
      # values with the given upstream gradients, and independently builds the expected gradient
      # by scattering upstream gradients to the selected row/index positions. Returns a list of
      # the expected gradient and the computed gradient.
      with backprop.GradientTape() as tape:
        val, idx = nn_ops.approx_max_k(db, k)
      result_in_grads = tape.gradient(val, db, out_grads)
      lifted_k_idx = array_ops.reshape(idx, [num_rows, k, 1])
      iota_idx = array_ops.broadcast_to(
          array_ops.reshape(math_ops.range(num_rows), [num_rows, 1, 1]),
          [num_rows, k, 1])
      lifted_idx = array_ops.concat([iota_idx, lifted_k_idx], axis=2)
      k_idx_s = array_ops.reshape(lifted_idx, [num_rows * k, 2])
      k_gra_s = array_ops.reshape(out_grads, [num_rows * k])
      expected_in_grads = array_ops.scatter_nd(k_idx_s, k_gra_s,
                                               [num_rows, row_size])
      return [expected_in_grads, result_in_grads]

    with ops.device('/device:TPU:0'):
      db_op = variables.Variable(db, dtype=dtype)
      out_grads_op = variables.Variable(out_grads, dtype=dtype)
      expected_in_grads, result_in_grads = ann_with_grads(db_op, out_grads_op)

    self.assertAllClose(expected_in_grads, result_in_grads)

  # Tests that multiple ops are supported and the comparison functions are
  # renamed properly to avoid conflict while using the MLIR bridge.
  def test_multiple_ops(self):
    # [FUNCTION DOC]
    # Compiles a function containing two approximate max-k operations on different databases
    # (k of 1, aggregated to top-k) to verify comparison functions are renamed to avoid conflicts
    # in the MLIR bridge. Asserts each result has recall at least 0.95 against exact sorting.
    k = 1

    row_size = 100
    num_rows = 10

    row = np.arange(row_size, dtype=np.float32)
    db1 = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))
    db2 = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))

    @function(jit_compile=True)
    def ann(db1, db2):
      # [FUNCTION DOC]
      # Jit-compiled: runs approximate max-k separately on each of two databases and returns both
      # results as a pair.
      result1 = nn_ops.approx_max_k(db1, k, aggregate_to_topk=True)
      result2 = nn_ops.approx_max_k(db2, k, aggregate_to_topk=True)
      return (result1, result2)

    with ops.device('/device:TPU:0'):
      db1_op = variables.Variable(db1)
      db2_op = variables.Variable(db2)
      result1, result2 = ann(db1_op, db2_op)

    gt = np.argsort(-db1)[:, :k]
    ann_recall = self.compute_recall(result1[1].numpy(), gt)
    self.assertGreaterEqual(ann_recall, 0.95)

    gt = np.argsort(-db2)[:, :k]
    ann_recall = self.compute_recall(result2[1].numpy(), gt)
    self.assertGreaterEqual(ann_recall, 0.95)

if __name__ == '__main__':
  test.main()
