# [MODULE DOC]
# Tests for the random index shuffle ops: the raw generated op and the public stateless index
# shuffle function. Verifies the shuffle is a permutation of the index range (no repeats, within
# bounds), dtype preservation across integer index and seed dtypes, batched inputs with broadcast
# seed and maximum, unknown shapes inside a compiled function, and the invalid-argument errors for
# negative values or an index above the maximum. Module constants list the seeds, maximum indices,
# round counts and dtypes used to parameterize tests. Running as a script starts the test runner.
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
"""Tests for random index shuffle ops."""

import itertools

from absl.testing import parameterized

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_index_shuffle_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.platform import test

_SEEDS = ((74, 117), (42, 5))
_MAX_INDEX = (129, 2_389)
_ROUNDS = (4, 8)
_DTYPES = (dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64)


class StatelessOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for the stateless random index shuffle (permutation of an index range
  # determined by a seed). Most tests are generated from products of seeds, dtypes, maximum
  # indices and round counts; no shared instance state.

  @parameterized.parameters(
      itertools.product(_SEEDS, _DTYPES, _MAX_INDEX, _DTYPES, _ROUNDS))
  def testRawOp(self, seed, seed_dtype, max_index, index_dtype, rounds):
    # [FUNCTION DOC]
    # Exercises the raw generated shuffle op one index at a time. Inputs (supplied by the
    # parameterization): a two-number seed, the seed dtype, the maximum index, the index dtype and
    # the number of rounds. Skips cases with maximum above 200 as too slow in graph mode. Builds
    # a three-element seed, shuffles every index from zero to the maximum and asserts the result
    # keeps the index dtype, lies within bounds and has not been produced before (a permutation).
    if max_index > 200:
      self.skipTest('Too slow in graph mode.')
    seen = (max_index + 1) * [False]
    seed = math_ops.cast([seed[0], seed[1], 42], seed_dtype)
    for index in range(max_index + 1):
      new_index = gen_random_index_shuffle_ops.random_index_shuffle(
          math_ops.cast(index, index_dtype),
          seed,
          max_index=math_ops.cast(max_index, index_dtype),
          rounds=rounds)
      self.assertEqual(new_index.dtype, index_dtype)
      new_index = self.evaluate(new_index)
      self.assertGreaterEqual(new_index, 0)
      self.assertLessEqual(new_index, max_index)
      self.assertFalse(seen[new_index])
      seen[new_index] = True

  @parameterized.parameters(
      itertools.product(_SEEDS, _DTYPES, _MAX_INDEX, _DTYPES))
  def testUnbatched(self, seed, seed_dtype, max_index, index_dtype):
    # [FUNCTION DOC]
    # Same permutation check as the raw-op test but through the public stateless index shuffle
    # function, one scalar index at a time with a two-element seed and default rounds. Inputs come
    # from the parameterization (seed, seed dtype, maximum index, index dtype). Skips maximum
    # above 200 as too slow in graph mode. Asserts dtype preservation, bounds and uniqueness.
    if max_index > 200:
      self.skipTest('Too slow in graph mode.')
    seen = (max_index + 1) * [False]
    seed = math_ops.cast(seed, seed_dtype)
    for index in range(max_index + 1):
      new_index = stateless.index_shuffle(
          math_ops.cast(index, index_dtype),
          seed,
          max_index=math_ops.cast(max_index, index_dtype))
      self.assertEqual(new_index.dtype, index_dtype)
      new_index = self.evaluate(new_index)
      self.assertGreaterEqual(new_index, 0)
      self.assertLessEqual(new_index, max_index)
      self.assertFalse(seen[new_index])
      seen[new_index] = True

  @parameterized.parameters(
      itertools.product(_SEEDS, _DTYPES, _MAX_INDEX, _DTYPES))
  def testBatchedBroadcastSeedAndMaxval(self, seed, seed_dtype, max_index,
                                        index_dtype):
    # [FUNCTION DOC]
    # Checks shuffling a whole vector of indices at once, with a single seed and a single
    # (Python integer) maximum broadcast over the batch. Inputs come from the parameterization.
    # Asserts the output keeps the index dtype, values lie between zero and the maximum, the
    # length is maximum plus one, and all values are distinct (a full permutation).
    seed = math_ops.cast(seed, seed_dtype)
    index = math_ops.cast(range(max_index + 1), index_dtype)
    new_index = stateless.index_shuffle(index, seed, max_index=max_index)
    self.assertEqual(new_index.dtype, index_dtype)
    new_index = self.evaluate(new_index)
    self.assertAllGreaterEqual(new_index, 0)
    self.assertAllLessEqual(new_index, max_index)
    self.assertLen(new_index, max_index + 1)
    self.assertLen(set(new_index), max_index + 1)

  def test_unknown_shape(self):
    # [FUNCTION DOC]
    # Checks the shuffle works inside a compiled function when the number of indices is only
    # known at run time. Calls the function with a repeat count of two and asserts the results
    # lie between zero and the maximum of ten.

    @def_function.function
    def shuffle(repeats):
      # [FUNCTION DOC]
      # HELPER: compiled function used by the unknown-shape test. Input repeats is a scalar
      # integer tensor giving how many copies of the index two to create; returns the shuffled
      # indices using a fixed seed and maximum index ten. Output length is dynamic.
      indices = array_ops.repeat(2, repeats)
      return stateless.index_shuffle(indices, seed=(1, 2), max_index=10)

    new_index = shuffle(constant_op.constant(2))
    self.assertAllGreaterEqual(new_index, 0)
    self.assertAllLessEqual(new_index, 10)

  def test_negative_index(self):
    # [FUNCTION DOC]
    # Error case: a negative index must raise an invalid-argument error whose message says the
    # index must be non-negative.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, 'index must be >= 0'
    ):
      self.evaluate(stateless.index_shuffle(-1, seed=(1, 2), max_index=10))

  def test_negative_max_index(self):
    # [FUNCTION DOC]
    # Error case: a negative maximum index must raise an invalid-argument error whose message
    # says the maximum index must be non-negative.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, 'max_index must be >= 0'
    ):
      self.evaluate(stateless.index_shuffle(0, seed=(1, 2), max_index=-1))

  def test_index_greater_than_max_index(self):
    # [FUNCTION DOC]
    # Error case: an index of five with a maximum of four must raise an invalid-argument error
    # whose message says the maximum index must be at least the index.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, 'max_index must be >= index'
    ):
      self.evaluate(stateless.index_shuffle(5, seed=(1, 2), max_index=4))


if __name__ == '__main__':
  test.main()
