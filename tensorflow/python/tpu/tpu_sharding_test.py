# [MODULE DOC]
# KNOWN ISSUE: the module docstring says it tests tpu_function helpers, but the tests cover the
# ShardingPolicy class of the tpu_sharding module.
# Unit tests for ShardingPolicy: freezing, frozen immutability, string form, merging, sharded,
# unpartitioned and unsharded shape computation, and scalar handling. Runs through the standard
# test main entry point when executed directly.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Tests for tpu_function helpers."""


from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_sharding


class ShardingTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the ShardingPolicy class from the tpu_sharding module. Each test builds
  # fresh policies; there is no shared fixture.

  def testFreeze(self):
    # [FUNCTION DOC]
    # Scenario: a fresh policy is frozen and gets default shard count and dimension; a policy
    # with explicit values (17 shards, dimension 23) keeps them after freezing. Asserts equality.
    """Tests that freezing a policy applies default values."""
    p1 = tpu_sharding.ShardingPolicy()
    p1.freeze()
    self.assertEqual(p1.number_of_shards,
                     tpu_sharding._DEFAULT_NUMBER_OF_SHARDS)
    self.assertEqual(p1.shard_dimension, tpu_sharding._DEFAULT_SHARD_DIMENSION)
    p2 = tpu_sharding.ShardingPolicy()
    p2.set_number_of_shards(17)
    p2.set_shard_dimension(23)
    p2.freeze()
    self.assertEqual(p2.number_of_shards, 17)
    self.assertEqual(p2.shard_dimension, 23)

  def testFrozen(self):
    # [FUNCTION DOC]
    # Scenario: a frozen default policy is given a different shard count and a different shard
    # dimension. Asserts that each attempt raises a value error.
    """Tests that frozen policies can't be changed."""
    p1 = tpu_sharding.ShardingPolicy()
    p1.freeze()
    with self.assertRaises(ValueError):
      p1.set_number_of_shards(17)
    with self.assertRaises(ValueError):
      p1.set_shard_dimension(22)

  def testStr(self):
    # [FUNCTION DOC]
    # Scenario: text form of a policy is checked while unset, with only the shard count set
    # (still the unset form), and with both set (shows 17 shards and dimension 8).
    """Tests the string representation."""
    p1 = tpu_sharding.ShardingPolicy()
    self.assertEqual(str(p1), "ShardingPolicy(unset)")
    p1.set_number_of_shards(17)
    self.assertEqual(str(p1), "ShardingPolicy(unset)")
    p1.set_shard_dimension(8)
    self.assertEqual(str(p1), "ShardingPolicy(17 shards dimension 8)")

  def testMerge(self):
    # [FUNCTION DOC]
    # Scenario: merges policies into an empty one, then merges a partial policy, then freezes
    # and merges compatible policies (no change) and conflicting ones. Asserts values are copied
    # when unfrozen, unchanged for compatible frozen merges, and that conflicting shard count or
    # dimension on a frozen policy raises a value error.
    """Tests that merging works."""
    p1 = tpu_sharding.ShardingPolicy()
    p1.set_number_of_shards(17)
    p1.set_shard_dimension(23)
    p2 = tpu_sharding.ShardingPolicy()
    p2.merge(p1)
    self.assertEqual(p2.number_of_shards, 17)
    self.assertEqual(p2.shard_dimension, 23)
    p1 = tpu_sharding.ShardingPolicy()
    p1.set_shard_dimension(12)
    p2.merge(p1)
    self.assertEqual(p2.number_of_shards, 17)
    self.assertEqual(p2.shard_dimension, 12)
    p2.freeze()
    p2.merge(p1)
    self.assertEqual(p2.number_of_shards, 17)
    self.assertEqual(p2.shard_dimension, 12)
    p1.set_number_of_shards(1)
    with self.assertRaises(ValueError):
      p2.merge(p1)
    p1 = tpu_sharding.ShardingPolicy()
    p1.set_number_of_shards(17)
    p2.merge(p1)
    p1.set_shard_dimension(2)
    with self.assertRaises(ValueError):
      p2.merge(p1)

  def testGetShardedShape(self):
    # [FUNCTION DOC]
    # Scenario: three shards along dimension one. Asserts a four by nine shape yields four by
    # three; after freezing, changing the dimension raises a value error; out-of-range shard
    # indices, an unknown-rank shape, and a size not divisible by three raise value errors; a
    # non-shape string raises a type error.
    """Tests getting a sharded shape."""
    p = tpu_sharding.ShardingPolicy()
    p.set_number_of_shards(3)
    p.set_shard_dimension(1)
    self.assertEqual(p.get_sharded_shape([4, 9]), [4, 3])
    p.freeze()
    with self.assertRaises(ValueError):
      p.set_shard_dimension(0)
    with self.assertRaises(ValueError):
      _ = p.get_sharded_shape([4, 9], shard_index=4)
    with self.assertRaises(ValueError):
      _ = p.get_sharded_shape([4, 9], shard_index=-1)
    with self.assertRaises(TypeError):
      _ = p.get_sharded_shape("not_a_shape")
    with self.assertRaises(ValueError):
      _ = p.get_sharded_shape(tensor_shape.TensorShape(None))
    with self.assertRaises(ValueError):
      _ = p.get_sharded_shape([4, 10], shard_index=-1)

  def testGetUnpartitionedShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says "getting a sharded shape" but the test exercises the
    # unpartitioned shape.
    # Scenario: three shards, dimension one, four partitions. Asserts a three by five shape
    # becomes three by twenty, and after freezing that an unknown size in the shard dimension
    # raises a value error.
    """Tests getting a sharded shape."""
    p = tpu_sharding.ShardingPolicy()
    p.set_number_of_shards(3)
    p.set_shard_dimension(1)
    p.set_number_of_partitions(4)
    self.assertEqual(p.get_unpartitioned_shape([3, 5]), [3, 20])
    p.freeze()
    with self.assertRaises(ValueError):
      _ = p.get_unpartitioned_shape([3, None])

  def testGetUnshardedShape(self):
    # [FUNCTION DOC]
    # Scenario: two shards along dimension one. Asserts two shards of four by three combine to
    # four by six. Asserts value errors for a wrong number of shards (one or three), mismatched
    # shard shapes, an unknown shape, and a shard of too small rank; a non-shape string
    # raises a type error.
    """Tests getting an unsharded shape."""
    p = tpu_sharding.ShardingPolicy()
    p.set_number_of_shards(2)
    p.set_shard_dimension(1)
    self.assertEqual(p.get_unsharded_shape([[4, 3], [4, 3]]), [4, 6])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([[4, 3]])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([[4, 3], [4, 3], [4, 3]])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([[4, 3], [4, 2]])
    with self.assertRaises(TypeError):
      _ = p.get_unsharded_shape([[4, 3], "not_a_shape"])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([None, [4, 3]])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([[2], [4, 3]])

  def testScalar(self):
    # [FUNCTION DOC]
    # Scenario: a frozen default policy (one shard) is applied to an empty (scalar) shape.
    # Asserts both sharded and unsharded shapes stay empty.
    """Tests sharding and unsharding scalars."""
    p = tpu_sharding.ShardingPolicy()
    p.freeze()
    self.assertEqual(p.get_sharded_shape([]), [])
    self.assertEqual(p.get_unsharded_shape([[]]), [])


if __name__ == "__main__":
  test.main()
