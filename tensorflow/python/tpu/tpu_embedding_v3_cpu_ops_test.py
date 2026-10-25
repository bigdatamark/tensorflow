# [MODULE DOC]
# Tests for the SparseCore CPU-side XLA operations used by TPU embedding v3 (converting sparse
# features to per-core coordinate lists, sorting, statistics, CSR-wrapped conversion). Contains
# pure Python reference implementations (combiner scaling, deduplication, statistics, CSR layout
# with physical replicas) used as golden values, and one test class. Run as a script it enables
# v2 behavior and starts the test runner. No import time side effects.
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import functools
import itertools
import math

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu.ops import gen_xla_ops as xla_ops


def _get_combiner_scale_contribution(x, combiner):
  # [FUNCTION DOC]
  # HELPER: reference per-entry contribution to a row's combiner total. For the sum combiner it
  # is 1.0, for mean it is the weight x itself, and for any other combiner (square-root-n) it is
  # x squared. Pure function.
  if combiner == "sum":
    return 1.0
  elif combiner == "mean":
    return x
  else:
    return x * x


def _get_combiner_scale_transform(x, combiner):
  # [FUNCTION DOC]
  # HELPER: reference final scale factor for a row total x. Returns 1.0 for the sum combiner;
  # for mean, the reciprocal of x (0 when x is 0); otherwise the reciprocal of the square root
  # of x (0 when x is 0). Pure function.
  if combiner == "sum":
    return 1.0
  elif combiner == "mean":
    return 0 if x == 0 else (1 / x)
  else:
    return 0 if x == 0 else (1 / math_ops.sqrt(x))


def convert_input_to_coo_tensor(
    indices_or_row_splits, values, weight, sample_count, combiner
):
  # [FUNCTION DOC]
  # HELPER: reference (golden) conversion of an input feature to deduplicated coordinate lists.
  # Inputs: indices_or_row_splits, either a two-column index matrix (rank 2, first column is the
  # row), a row-splits vector (rank 1) or a scalar-like value (each value is its own row);
  # values (column ids); weight (per-value weights); sample_count; combiner name (sum, mean or
  # anything else meaning square-root-n).
  # Behavior: derives each value's row id, merges consecutive entries with the same row and
  # column by adding gains, accumulates the per-row scale contribution, converts it to a scale
  # factor and multiplies the gains by it.
  # Returns three tensors: row ids and column ids (32-bit integers) and gains (32-bit floats).
  # No side effects. The sample count must cover all row ids, otherwise an index error occurs.
  if indices_or_row_splits.shape.rank >= 2:
    row_ids_before_dedup = indices_or_row_splits[:, 0]
  elif indices_or_row_splits.shape.rank == 1:
    row_ids_before_dedup = []
    current_row_id = -1
    for i, _ in enumerate(values):
      while i == indices_or_row_splits[current_row_id + 1]:
        current_row_id += 1
      row_ids_before_dedup.append(current_row_id)
  else:
    row_ids_before_dedup = math_ops.range(0, values.shape[0])

  col_ids = []
  row_ids = []
  gains = []
  gains_rescale = [0] * sample_count
  for i in range(values.shape[0]):
    gains_rescale[row_ids_before_dedup[i]] += _get_combiner_scale_contribution(
        weight[i], combiner
    )
    if (
        row_ids
        and row_ids[-1] == row_ids_before_dedup[i]
        and col_ids[-1] == values[i]
    ):
      gains[-1] += weight[i]
    else:
      row_ids.append(row_ids_before_dedup[i])
      col_ids.append(values[i])
      gains.append(weight[i])

  for i in range(sample_count):
    gains_rescale[i] = _get_combiner_scale_transform(gains_rescale[i], combiner)
  for i in range(len(row_ids)):
    gains[i] *= gains_rescale[row_ids[i]]

  row_ids = ops.convert_to_tensor(row_ids, dtype=dtypes.int32)
  col_ids = ops.convert_to_tensor(col_ids, dtype=dtypes.int32)
  gains = ops.convert_to_tensor(gains, dtype=dtypes.float32)

  return row_ids, col_ids, gains


def _compute_sparse_core_stats(row_ids, col_ids, num_sc_shards):
  # [FUNCTION DOC]
  # HELPER: reference computation of SparseCore id statistics for the stats op test.
  # Inputs: parallel sequences of row ids and column ids, and the number of shards. Entries are
  # sorted by (column id, row id); per shard (column id modulo shard count) it counts every
  # distinct (column, row) pair as an id and every distinct column as a unique id.
  # Returns a pair: the maximum id count and the maximum unique id count over the shards.
  # No side effects.
  max_ids = np.zeros(num_sc_shards)
  max_unique_ids = np.zeros(num_sc_shards)
  previous_col_id = -1
  previous_row_id = -1
  for col_id, row_id in sorted(zip(col_ids, row_ids)):
    if col_id != previous_col_id:
      max_ids[col_id % num_sc_shards] += 1
      max_unique_ids[col_id % num_sc_shards] += 1
    else:
      if previous_row_id != row_id:
        max_ids[col_id % num_sc_shards] += 1
    previous_col_id = col_id
    previous_row_id = row_id
  return max(max_ids), max(max_unique_ids)


def _convert_coo_tensor_to_csr_with_physical_replica(
    row_ids,
    col_ids,
    gains,
    splits,
    sample_count,
    num_replica,
    max_minibatches_per_sc,
    max_ids_per_chip_per_sample,
    table_vocab_size,
):
  # [FUNCTION DOC]
  # HELPER: pure Python reference (golden) implementation of the CSR-wrapped coordinate
  # conversion, used to check the corresponding XLA op.
  # Inputs: row_ids, column ids and gains of the sorted coordinate entries; splits, a scalar (or
  # array of masks that is OR-ed together) tensor whose set bits mark where the vocabulary is
  # divided into minibatch buckets; the sample count (must be a multiple of 4, since 4 sparse
  # cores per replica are assumed); replica count; the maximum minibatches per core; the maximum
  # ids per chip per sample; and the table vocabulary size.
  # Behavior: decodes the split bits into bucket boundaries, groups entries by physical replica
  # (column id modulo replicas times 4), core (row divided by samples per core) and minibatch
  # bucket, then lays out row pointers and ids padded to multiples of 8 (padding value 8).
  # Returns a tuple of seven values: row pointers, sorted sample ids, sorted token ids, sorted
  # gains, the unpadded row pointer size, the unpadded id count and the minibatches per core,
  # the last three as 32-bit integer scalars.
  # Raises an assertion error if the sample count is not a multiple of 4 or if the number of
  # minibatches per core exceeds the allowed maximum.
  num_sc_per_replica = 4
  num_physical_replica = num_replica * num_sc_per_replica
  assert (
      sample_count % num_sc_per_replica == 0
  ), f"sample count should be multiply of 4 instead got {sample_count}"

  per_sc_sample_count = sample_count // num_sc_per_replica

  splits = splits.numpy()
  if splits.size > 1:
    splits = functools.reduce(lambda x, y: x | y, splits)

  max_division_level = 6
  max_divisions = 1 << max_division_level

  division_size = (table_vocab_size + max_divisions - 1) // max_divisions

  bucket_splits = []
  current_index = 0
  while splits > 0:
    if splits % 2 == 1:
      split_level = int(current_index + 1).bit_length() - 1
      split_offset = current_index + 1 - (1 << split_level)
      split_size = 1 << (max_division_level - 1 - split_level)
      bucket_splits.append(split_size + split_offset * split_size * 2)
    splits >>= 1
    current_index += 1

  bucket_splits.sort()

  num_minibatch_per_sc = len(bucket_splits) + 1

  embedding_lookup_inputs = []
  for row_id, col_id, gain in zip(row_ids, col_ids, gains):
    embedding_lookup_inputs.append(
        (col_id % num_physical_replica, col_id, row_id, gain)
    )
  # sort based on replica id first, then col_id.
  embedding_lookup_inputs.sort()

  total_minibatches = num_minibatch_per_sc * num_sc_per_replica

  assert num_minibatch_per_sc <= max_minibatches_per_sc, (
      f"Get {num_minibatch_per_sc} minibatches per sparse core, but the"
      " number of max minibatches per sparse core is"
      f" {max_minibatches_per_sc}"
  )

  minibatches = [[] for _ in range(total_minibatches)]

  def calculate_minibatch_id(col_id):
    # [FUNCTION DOC]
    # HELPER: nested helper of the reference CSR conversion. Given a column id, returns the index
    # of the first bucket split whose boundary (split value times division size) lies above the
    # column id, or the number of splits if none does. Uses the enclosing bucket splits and
    # division size.
    for i, bucket_split in enumerate(bucket_splits):
      if bucket_split * division_size > col_id:
        return i
    return len(bucket_splits)

  for embedding_lookup_input in embedding_lookup_inputs:
    sc_id = embedding_lookup_input[2] // per_sc_sample_count
    minibatch_id = calculate_minibatch_id(embedding_lookup_input[1])

    minibatches[sc_id * num_minibatch_per_sc + minibatch_id].append(
        embedding_lookup_input
    )

  def round_up_to(x, round_value):
    # [FUNCTION DOC]
    # HELPER: rounding utility for the reference CSR conversion. Returns x rounded up to the next
    # multiple of round_value (unchanged if already a multiple).
    return x + -x % round_value

  max_ids_per_chip = max_ids_per_chip_per_sample * sample_count

  padded_row_pointers_size = round_up_to(num_physical_replica, 8)

  total_row_pinters_size = padded_row_pointers_size * (
      max_minibatches_per_sc * num_sc_per_replica
  )

  row_pointers = np.full(total_row_pinters_size, 8, dtype=np.int32)
  sorted_sample_ids = np.full(max_ids_per_chip, 8, dtype=np.int32)
  sorted_token_ids = np.full(max_ids_per_chip, 8, dtype=np.int32)
  sorted_gains = np.full(max_ids_per_chip, 8, dtype=np.float32)

  id_index = 0
  row_pointers_index = 0
  for minibatch in minibatches:
    index = 0
    for replica_id in range(num_physical_replica):
      while index < len(minibatch) and replica_id == minibatch[index][0]:
        sorted_token_ids[id_index] = minibatch[index][1] // num_physical_replica
        sorted_sample_ids[id_index] = minibatch[index][2] % per_sc_sample_count
        sorted_gains[id_index] = minibatch[index][3]
        index += 1
        id_index += 1
      row_pointers[row_pointers_index] = id_index
      id_index = round_up_to(id_index, 8)
      row_pointers_index += 1

    for i in range(
        row_pointers_index,
        round_up_to(row_pointers_index, padded_row_pointers_size),
    ):
      row_pointers[i] = id_index
    row_pointers_index = round_up_to(
        row_pointers_index, padded_row_pointers_size
    )

  row_pointers_unpadded_size = total_minibatches * padded_row_pointers_size
  ids_unpadded_size = id_index

  return (
      row_pointers,
      sorted_sample_ids,
      sorted_token_ids,
      sorted_gains,
      np.array(row_pointers_unpadded_size, dtype=np.int32),
      np.array(ids_unpadded_size, dtype=np.int32),
      np.array(num_minibatch_per_sc, dtype=np.int32),
  )


class TpuEmbeddingV3CPUOpsTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Tests for the XLA SparseCore host (CPU) operations used by TPU embedding v3: converting a
  # sparse feature to per-core coordinate lists, sorting them, computing id statistics, and
  # building the CSR-wrapped coordinate representation (with and without id dropping). Each test
  # compares op outputs with the pure Python reference helpers defined in this module. No
  # shared fixtures.

  @parameterized.parameters(
      *list(
          itertools.product(
              [16, 32],
              [1024, 2048],
              ["sum", "mean", "sqrtn"],
              [0, 56, 1600000],
              [0, 12, 20],
          )
      )
  )
  def test_convert_to_list_of_sparse_core_coo_tensor(
      self, sample_count, token_count, combiner, col_offset, col_shift
  ):
    # [FUNCTION DOC]
    # Parameterized over the product of sample count (16, 32), token count (1024, 2048), combiner
    # (sum, mean, sqrtn), column offset and column shift. Builds a random sparse feature,
    # computes the expected row ids, column ids and gains in Python (row ids regrouped per
    # sparse core with the row offset and stacked-table sample count; column ids shifted within the
    # shard bits and offset; gains normalized by the combiner), then runs the conversion operation
    # with 512 shards and 4 cores per chip. Asserts the concatenated per-core row ids, column ids
    # and gains are close to the expected values. Uses random data.
    sparse_feature = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[
                [i % sample_count, i]
                for i in np.random.randint(low=0, high=1024, size=token_count)
            ],
            values=np.random.randint(low=0, high=1024, size=token_count),
            dense_shape=[sample_count, 1024],
        )
    )

    num_sc_per_chip = 4
    num_chip = 128

    row_offset = sample_count
    num_sc_shards = num_sc_per_chip * num_chip
    stacked_table_sample_count = sample_count * 4

    num_sc_shards_bit = int(math.log2(num_sc_shards))
    num_sc_shards_bit_mod = (1 << num_sc_shards_bit) - 1
    num_sc_shards_bit_mod_inv = bitwise_ops.invert(num_sc_shards_bit_mod)

    row_ids, col_ids, gains = convert_input_to_coo_tensor(
        indices_or_row_splits=sparse_feature.indices,
        values=sparse_feature.values,
        weight=np.ones(shape=token_count),
        sample_count=sample_count,
        combiner=combiner,
    )

    golden_row_ids = (
        row_ids % (sample_count // num_sc_per_chip)
        + int(row_offset // num_sc_per_chip)
        + int(stacked_table_sample_count // num_sc_per_chip)
        * (row_ids // (sample_count // num_sc_per_chip))
    )
    golden_col_ids = (
        bitwise_ops.bitwise_and(col_ids + col_shift, num_sc_shards_bit_mod)
        + bitwise_ops.bitwise_and(col_ids, num_sc_shards_bit_mod_inv)
        + col_offset
    )

    row_ids_list, col_ids_list, gains_list = (
        xla_ops.convert_to_list_of_sparse_core_coo_tensors(
            indices_or_row_splits=math_ops.cast(
                sparse_feature.indices, dtype=dtypes.int32
            ),
            values=math_ops.cast(sparse_feature.values, dtype=dtypes.int32),
            weights=1.0,
            sample_count=sample_count,
            combiner=combiner,
            num_sc_per_chip=4,
            row_offset=row_offset,
            col_offset=col_offset,
            col_shift=col_shift,
            num_sc_shards=num_sc_shards,
            stacked_table_sample_count=stacked_table_sample_count,
        )
    )

    self.assertAllClose(golden_row_ids, array_ops.concat(row_ids_list, axis=0))
    self.assertAllClose(golden_col_ids, array_ops.concat(col_ids_list, axis=0))
    self.assertAllClose(gains, array_ops.concat(gains_list, axis=0))

  def test_convert_to_list_of_sparse_core_coo_tensors(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name differs from the parameterized test above only by a trailing letter
    # (tensor versus tensors) and does not reflect what it covers (the whole chain through
    # sorting and CSR conversion, not only the first conversion).
    # Scenario: end-to-end check of the CPU ops chain: 16 samples, 1024 tokens, sum combiner.
    # Converts to per-core coordinate lists, sorts each of the 4 cores' lists, converts the sorted
    # lists to the CSR-wrapped coordinate form (no id dropping, no splits), and compares with a
    # pure Python reference that builds the same structure with physical replicas. Asserts the
    # row pointers and the sorted sample ids, token ids and gains agree (compared only up to the
    # unpadded sizes) and that the number of minibatches per core is equal.
    sample_count = 16
    token_count = 1024
    combiner = "sum"
    sparse_feature = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % sample_count, i] for i in np.arange(token_count)],
            values=np.arange(token_count),
            dense_shape=[sample_count, 1024],
        )
    )

    row_ids_list, col_ids_list, gains_list = (
        xla_ops.convert_to_list_of_sparse_core_coo_tensors(
            indices_or_row_splits=math_ops.cast(
                sparse_feature.indices, dtype=dtypes.int32
            ),
            values=math_ops.cast(sparse_feature.values, dtype=dtypes.int32),
            weights=1.0,
            sample_count=sample_count,
            combiner=combiner,
            num_sc_per_chip=4,
            row_offset=0,
            col_offset=0,
            col_shift=0,
            num_sc_shards=16,
            stacked_table_sample_count=sample_count,
        )
    )

    sorted_row_ids_list = []
    sorted_col_ids_list = []
    sorted_gains_list = []
    id_counts_list = []
    for i in range(4):
      (
          sorted_row_ids,
          sorted_col_ids,
          sorted_gains,
          id_counts,
      ) = xla_ops.sort_list_of_sparse_core_coo_tensors(
          row_ids_list=[row_ids_list[i]],
          col_ids_list=[col_ids_list[i]],
          gains_list=[gains_list[i]],
          sample_count_list=[sample_count // 4],
          col_offset_list=[0],
          num_replica=4,
          table_vocab_size=16384,
          feature_width=16,
          num_sc_per_chip=4,
          max_ids_per_sparse_core=256,
          max_unique_ids_per_sparse_core=256,
          table_name="table",
      )
      sorted_row_ids_list.append(sorted_row_ids)
      sorted_col_ids_list.append(sorted_col_ids)
      sorted_gains_list.append(sorted_gains)
      id_counts_list.append(id_counts)

    (
        row_pointers,
        sorted_sample_ids,
        sorted_token_ids,
        sorted_gains,
        row_pointers_unpadded_size,
        ids_unpadded_size,
        num_minibatches_per_sc,
    ) = xla_ops.convert_to_sparse_core_csr_wrapped_coo_tensor(
        sorted_row_ids_list=sorted_row_ids_list,
        sorted_col_ids_list=sorted_col_ids_list,
        sorted_gains_list=sorted_gains_list,
        id_counts_list=id_counts_list,
        splits=constant_op.constant(0, dtype=dtypes.int64),
        sample_count_per_sc=sample_count // 4,
        max_minibatches_per_sc=4,
        max_ids_per_chip_per_sample=64,
        table_vocab_size=16384,
        feature_width=16,
        num_replica=4,
        allow_id_dropping=False,
        table_name="table",
    )

    (
        golden_row_pointers,
        golden_sorted_sample_ids,
        golden_sorted_token_ids,
        golden_sorted_gains,
        golden_row_pointers_unpadded_size,
        golden_ids_unpadded_size,
        golden_num_minibatches_per_sc,
    ) = _convert_coo_tensor_to_csr_with_physical_replica(
        row_ids=array_ops.concat(row_ids_list, axis=0),
        col_ids=array_ops.concat(col_ids_list, axis=0),
        gains=array_ops.concat(gains_list, axis=0),
        splits=constant_op.constant(0, dtype=dtypes.int64),
        sample_count=sample_count,
        num_replica=4,
        max_minibatches_per_sc=4,
        max_ids_per_chip_per_sample=64,
        table_vocab_size=16384,
    )

    self.assertAllClose(
        golden_row_pointers[:golden_row_pointers_unpadded_size],
        row_pointers[:row_pointers_unpadded_size],
    )
    self.assertAllClose(
        golden_sorted_sample_ids[:golden_ids_unpadded_size],
        sorted_sample_ids[:ids_unpadded_size],
    )
    self.assertAllClose(
        golden_sorted_token_ids[:golden_ids_unpadded_size],
        sorted_token_ids[:ids_unpadded_size],
    )
    self.assertAllClose(
        golden_sorted_gains[:golden_ids_unpadded_size],
        sorted_gains[:ids_unpadded_size],
    )
    self.assertEqual(golden_num_minibatches_per_sc, num_minibatches_per_sc)

  def test_get_stats_from_list_of_sparse_core_coo_tensors(self):
    # [FUNCTION DOC]
    # Scenario: random sparse feature (16 samples, 1024 tokens, sum combiner, 16 shards). The
    # golden maximum ids and maximum unique ids per sparse core are computed in Python from each
    # of four sample slices using the stats helper. The feature is converted to per-core lists and
    # the statistics operation is run for each core; the maxima over the four cores are
    # compared. Asserts both the maximum ids and the maximum unique ids equal the golden values.
    # Uses random data, so the input differs between runs.
    sample_count = 16
    token_count = 1024
    combiner = "sum"
    sparse_feature = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[
                [i % sample_count, i]
                for i in np.random.randint(low=0, high=1024, size=token_count)
            ],
            values=np.random.randint(low=0, high=1024, size=token_count),
            dense_shape=[sample_count, 1024],
        )
    )
    max_ids_golden = 0
    max_unique_ids_golden = 0

    for i in range(4):
      sparse_feature_slice = sparse_ops.sparse_slice(
          sparse_feature,
          [i * sample_count // 4, 0],
          [sample_count // 4, 1024],
      )
      max_ids_per_sparse_core, max_uniques_per_sparse_core = (
          _compute_sparse_core_stats(
              sparse_feature_slice.indices[:, 0],
              sparse_feature_slice.values,
              num_sc_shards=16,
          )
      )
      max_ids_golden = max(max_ids_golden, max_ids_per_sparse_core)
      max_unique_ids_golden = max(
          max_unique_ids_golden, max_uniques_per_sparse_core
      )

    row_ids_list, col_ids_list, gains_list = (
        xla_ops.convert_to_list_of_sparse_core_coo_tensors(
            indices_or_row_splits=math_ops.cast(
                sparse_feature.indices, dtype=dtypes.int32
            ),
            values=math_ops.cast(sparse_feature.values, dtype=dtypes.int32),
            weights=1.0,
            sample_count=sample_count,
            combiner=combiner,
            num_sc_per_chip=4,
            row_offset=0,
            col_offset=0,
            col_shift=0,
            num_sc_shards=16,
            stacked_table_sample_count=sample_count,
        )
    )

    max_ids = 0
    max_uniques = 0
    for i in range(4):
      max_ids_per_sparse_core, max_unique_ids_per_sparse_core = (
          xla_ops.get_stats_from_list_of_sparse_core_coo_tensors(
              row_ids_list=[row_ids_list[i]],
              col_ids_list=[col_ids_list[i]],
              gains_list=[gains_list[i]],
              sample_count_list=[sample_count // 4],
              col_offset_list=[0],
              num_replica=4,
              table_vocab_size=16384,
              feature_width=16,
              num_sc_per_chip=4,
              table_name="table",
          )
      )
      max_ids = max(max_ids, max_ids_per_sparse_core)
      max_uniques = max(max_uniques, max_unique_ids_per_sparse_core)

    self.assertEqual(max_ids, max_ids_golden)
    self.assertEqual(max_uniques, max_unique_ids_golden)

  def test_sort_list_of_sparse_core_coo_tensors(self):
    # [FUNCTION DOC]
    # Scenario: 16 samples and 1024 tokens with identity-like ids, 4 chips, 4 cores per chip.
    # Converts the sparse feature to per-core coordinate lists, then for each of the 4 cores runs
    # the sort operation. The expected order is computed in Python by sorting tuples of
    # (column id modulo 16, column id, row id, gain). Asserts the sorted row ids equal the row
    # ids modulo the samples per core, the sorted column ids equal column id divided by the
    # number of physical replicas (chips times 4), and the sorted gains match.
    sample_count = 16
    token_count = 1024
    combiner = "sum"
    num_chips = 4
    sparse_feature = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % sample_count, i] for i in np.arange(token_count)],
            values=np.arange(token_count),
            dense_shape=[sample_count, 1024],
        )
    )

    row_ids_list, col_ids_list, gains_list = (
        xla_ops.convert_to_list_of_sparse_core_coo_tensors(
            indices_or_row_splits=math_ops.cast(
                sparse_feature.indices, dtype=dtypes.int32
            ),
            values=math_ops.cast(sparse_feature.values, dtype=dtypes.int32),
            weights=1.0,
            sample_count=sample_count,
            combiner=combiner,
            num_sc_per_chip=4,
            row_offset=0,
            col_offset=0,
            col_shift=0,
            num_sc_shards=num_chips * 4,
            stacked_table_sample_count=sample_count,
        )
    )

    for i in range(4):
      (
          sorted_row_ids,
          sorted_col_ids,
          sorted_gains,
          _,
      ) = xla_ops.sort_list_of_sparse_core_coo_tensors(
          row_ids_list=[row_ids_list[i]],
          col_ids_list=[col_ids_list[i]],
          gains_list=[gains_list[i]],
          sample_count_list=[sample_count // 4],
          col_offset_list=[0],
          num_replica=num_chips,
          table_vocab_size=16384,
          feature_width=16,
          num_sc_per_chip=4,
          max_ids_per_sparse_core=256,
          max_unique_ids_per_sparse_core=256,
          table_name="table",
      )

      embedding_lookup_inputs = []
      for row_id, col_id, gain in zip(
          row_ids_list[i], col_ids_list[i], gains_list[i]
      ):
        embedding_lookup_inputs.append((col_id % 16, col_id, row_id, gain))
      # sort based on replica id first, then col_id.
      embedding_lookup_inputs.sort()

      self.assertAllClose(
          sorted_row_ids,
          [inp[2] % (sample_count // 4) for inp in embedding_lookup_inputs],
      )
      self.assertAllClose(
          sorted_col_ids,
          [inp[1] // (num_chips * 4) for inp in embedding_lookup_inputs],
      )
      self.assertAllClose(
          sorted_gains, [inp[3] for inp in embedding_lookup_inputs]
      )

  def test_id_dropping_with_convert_to_list_of_sparse_core_coo_tensors(self):
    # [FUNCTION DOC]
    # Scenario: 16 samples, 1024 tokens, sum combiner, 4 sparse cores per chip. Converts the
    # sparse feature to per-core coordinate lists, sorts each core's list, then calls the
    # CSR-wrapped conversion operation with a very small maximum ids per chip per sample (8).
    # Asserts that with id dropping disallowed the operation raises an error (any exception),
    # and then calls it again with id dropping allowed, which is expected to succeed (no output
    # is checked, only that no error is raised).
    sample_count = 16
    token_count = 1024
    combiner = "sum"
    sparse_feature = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % sample_count, i] for i in np.arange(token_count)],
            values=np.arange(token_count),
            dense_shape=[sample_count, 1024],
        )
    )

    row_ids_list, col_ids_list, gains_list = (
        xla_ops.convert_to_list_of_sparse_core_coo_tensors(
            indices_or_row_splits=math_ops.cast(
                sparse_feature.indices, dtype=dtypes.int32
            ),
            values=math_ops.cast(sparse_feature.values, dtype=dtypes.int32),
            weights=1.0,
            sample_count=sample_count,
            combiner=combiner,
            num_sc_per_chip=4,
            row_offset=0,
            col_offset=0,
            col_shift=0,
            num_sc_shards=16,
            stacked_table_sample_count=sample_count,
        )
    )

    sorted_row_ids_list = []
    sorted_col_ids_list = []
    sorted_gains_list = []
    id_counts_list = []
    for i in range(4):
      (
          sorted_row_ids,
          sorted_col_ids,
          sorted_gains,
          id_counts,
      ) = xla_ops.sort_list_of_sparse_core_coo_tensors(
          row_ids_list=[row_ids_list[i]],
          col_ids_list=[col_ids_list[i]],
          gains_list=[gains_list[i]],
          sample_count_list=[sample_count // 4],
          col_offset_list=[0],
          num_replica=4,
          table_vocab_size=16384,
          feature_width=16,
          num_sc_per_chip=4,
          max_ids_per_sparse_core=256,
          max_unique_ids_per_sparse_core=256,
          table_name="table",
      )
      sorted_row_ids_list.append(sorted_row_ids)
      sorted_col_ids_list.append(sorted_col_ids)
      sorted_gains_list.append(sorted_gains)
      id_counts_list.append(id_counts)

    # If not allow id dropping, the op will fail with very small
    # max_ids_per_chip_per_sample.
    with self.assertRaises(Exception):
      xla_ops.convert_to_sparse_core_csr_wrapped_coo_tensor(
          sorted_row_ids_list=sorted_row_ids_list,
          sorted_col_ids_list=sorted_col_ids_list,
          sorted_gains_list=sorted_gains_list,
          id_counts_list=id_counts_list,
          splits=constant_op.constant(0, dtype=dtypes.int64),
          sample_count_per_sc=sample_count // 4,
          max_minibatches_per_sc=4,
          max_ids_per_chip_per_sample=8,
          table_vocab_size=16384,
          feature_width=16,
          num_replica=4,
          allow_id_dropping=False,
          table_name="table",
      )

    # Allow id dropping, the op will succeed,
    xla_ops.convert_to_sparse_core_csr_wrapped_coo_tensor(
        sorted_row_ids_list=sorted_row_ids_list,
        sorted_col_ids_list=sorted_col_ids_list,
        sorted_gains_list=sorted_gains_list,
        id_counts_list=id_counts_list,
        splits=constant_op.constant(0, dtype=dtypes.int64),
        sample_count_per_sc=sample_count // 4,
        max_minibatches_per_sc=4,
        max_ids_per_chip_per_sample=8,
        table_vocab_size=16384,
        feature_width=16,
        num_replica=4,
        allow_id_dropping=True,
        table_name="table",
    )


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
