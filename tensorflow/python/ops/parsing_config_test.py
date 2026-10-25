# [MODULE DOC]
# Unit tests for the ragged feature spec in the parsing_config module: re-creation of partition
# objects by type name and the validating constructor with valid and invalid partitions.
# Run as a test binary through the platform test main entry point.
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_config
from tensorflow.python.platform import test


class RaggedFeatureTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the ragged feature spec: partition validation helper and constructor. No shared
  # fixtures.

  def test_validate_and_recreate_partition_row_splits(self):
    # [FUNCTION DOC]
    # Passes a row-splits partition with a key through the validation helper using the allowed
    # types. Expects a row-splits partition with the same key back.
    """Tests _validate_and_recreate_partition with RowSplits."""
    partition_types = parsing_config.RaggedFeature._PARTITION_TYPES
    partition = parsing_config.RaggedFeature.RowSplits(key="test_key")
    recreated_partition = (
        parsing_config.RaggedFeature._validate_and_recreate_partition(
            partition, partition_types
        )
    )
    self.assertIsInstance(
        recreated_partition, parsing_config.RaggedFeature.RowSplits
    )
    self.assertEqual(recreated_partition.key, "test_key")

  def test_validate_and_recreate_partition_row_lengths(self):
    # [FUNCTION DOC]
    # Same scenario as the row-splits test but with a row-lengths partition; expects a row-lengths
    # partition with the same key.
    """Tests _validate_and_recreate_partition with RowLengths."""
    partition_types = parsing_config.RaggedFeature._PARTITION_TYPES
    partition = parsing_config.RaggedFeature.RowLengths(key="test_key2")
    recreated_partition = (
        parsing_config.RaggedFeature._validate_and_recreate_partition(
            partition, partition_types
        )
    )
    self.assertIsInstance(
        recreated_partition, parsing_config.RaggedFeature.RowLengths
    )
    self.assertEqual(recreated_partition.key, "test_key2")

  def test_validate_and_recreate_partition_uniform_row_length(self):
    # [FUNCTION DOC]
    # Uses a uniform-row-length partition of length five; expects the same kind back with
    # length five.
    """Tests _validate_and_recreate_partition with UniformRowLength."""
    partition_types = parsing_config.RaggedFeature._PARTITION_TYPES
    partition = parsing_config.RaggedFeature.UniformRowLength(length=5)
    recreated_partition = (
        parsing_config.RaggedFeature._validate_and_recreate_partition(
            partition, partition_types
        )
    )
    self.assertIsInstance(
        recreated_partition, parsing_config.RaggedFeature.UniformRowLength
    )
    self.assertEqual(recreated_partition.length, 5)

  def test_validate_and_recreate_partition_invalid(self):
    # [FUNCTION DOC]
    # Defines a local class that is not a partition type and passes an instance to the helper.
    # Expects a type error whose message says the partition cannot be cast to the allowed types.
    """Tests _validate_and_recreate_partition with an invalid partition type."""
    partition_types = parsing_config.RaggedFeature._PARTITION_TYPES

    class InvalidPartition:
      # [CLASS DOC]
      # Local stand-in for an unsupported partition kind; holds only a key attribute.

      def __init__(self, key):
        # [FUNCTION DOC]
        # Stores the given key on the instance.
        self.key = key

    partition = InvalidPartition(key="invalid_key")
    with self.assertRaisesRegex(
        TypeError, "cannot be cast to any of the partition"
    ):
      parsing_config.RaggedFeature._validate_and_recreate_partition(
          partition, partition_types
      )

  def test_ragged_feature_new_with_valid_partitions(self):
    # [FUNCTION DOC]
    # Builds a ragged feature with a 64-bit integer dtype and a row-splits plus a uniform-row-length
    # partition. Expects the dtype kept, two partitions stored, and each of the right kind with
    # its key (first) and length three (second).
    """Tests RaggedFeature constructor with valid partitions."""
    partitions = [
        parsing_config.RaggedFeature.RowSplits(key="s1"),
        parsing_config.RaggedFeature.UniformRowLength(length=3),
    ]
    feature = parsing_config.RaggedFeature(
        dtype=dtypes.int64, partitions=partitions
    )
    self.assertEqual(feature.dtype, dtypes.int64)
    self.assertLen(feature.partitions, 2)
    self.assertIsInstance(
        feature.partitions[0], parsing_config.RaggedFeature.RowSplits
    )
    self.assertEqual(feature.partitions[0].key, "s1")
    self.assertIsInstance(
        feature.partitions[1], parsing_config.RaggedFeature.UniformRowLength
    )
    self.assertEqual(feature.partitions[1].length, 3)

  def test_ragged_feature_new_with_invalid_partitions(self):
    # [FUNCTION DOC]
    # Constructs a ragged feature whose partitions list holds an unsupported local object.
    # Expects a type error saying the partition cannot be cast to the allowed partition types.
    """Tests RaggedFeature constructor with invalid partitions."""

    class InvalidPartition:
      # [CLASS DOC]
      # Local stand-in for an unsupported partition kind; holds only a key attribute.

      def __init__(self, key):
        # [FUNCTION DOC]
        # Stores the given key on the instance.
        self.key = key

    partitions = [
        InvalidPartition(key="invalid_key"),
    ]
    with self.assertRaisesRegex(
        TypeError, "cannot be cast to any of the partition"
    ):
      parsing_config.RaggedFeature(dtype=dtypes.int64, partitions=partitions)


if __name__ == "__main__":
  test.main()
