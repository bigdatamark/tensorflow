# [MODULE DOC]
# Tests for the experimental index flat map dataset, which flattens per-element splits using a
# user-provided index map (global index to element index and offset). Covers string splitting,
# caching, global shuffle, identity and nested-list maps, error cases, and checkpointing.
# Also holds small module-level helpers (split function, metadata builder, index map builder).
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
"""Tests for the index flat map dataset."""

from typing import Any, Callable, Optional, Union

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import index_flat_map_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test

_IndexType = index_flat_map_op._IndexType


class IndexFlatMapTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Behavioral tests for the index flat map transformation.
  """Tests for global shuffling of index flat map datasets."""

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(use_tensors=[True, False])))
  def test_split_strings(self, use_tensors: bool):
    # [FUNCTION DOC]
    # Splits four space-separated strings into words with an index map derived from precomputed
    # metadata. The index map may take and return plain integers or 64-bit tensors depending on
    # use_tensors. Asserts the nine words are produced in order.
    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    # The metadata is [(0, 2, 0), (2, 6, 1), (6, 8, 2), (8, 9, 3)].
    metadata = _get_metadata(input_data)

    def _index_map_func(index: _IndexType) -> tuple[_IndexType, _IndexType]:
      # [FUNCTION DOC]
      # HELPER: wraps the metadata-based index map; converts the index, calls the map and
      # converts the resulting element index and offset to the configured representation.
      index = _maybe_convert_to_tensor(index)
      element_index, offset = _get_index_map_func(metadata)(index)
      return (_maybe_convert_to_tensor(element_index),
              _maybe_convert_to_tensor(offset))

    def _maybe_convert_to_tensor(value: Any) -> _IndexType:
      # [FUNCTION DOC]
      # HELPER: casts the value to a 64-bit integer tensor when use_tensors is set, otherwise
      # returns it unchanged.
      return math_ops.cast(value, dtypes.int64) if use_tensors else value

    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = index_flat_map_op.index_flat_map(dataset, _split, _index_map_func)
    output = self.getDatasetOutput(dataset)
    self.assertEqual(output,
                     [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8"])

  @combinations.generate(test_base.default_test_combinations())
  def test_cache(self):
    # [FUNCTION DOC]
    # Applies index flat map to a cached dataset of strings and asserts the nine split words
    # are produced in order.
    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    metadata = _get_metadata(input_data)

    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = dataset.cache()
    dataset = index_flat_map_op.index_flat_map(
        dataset, _split, _get_index_map_func(metadata))
    output = self.getDatasetOutput(dataset)
    self.assertEqual(output,
                     [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8"])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              repetitions=[1, 3],
              seed=[None, 42],
              reshuffle_each_iteration=[True, False])))
  def test_global_shuffle(
      self,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    # [FUNCTION DOC]
    # Applies index flat map with declared output cardinality nine (asserts cardinality), repeats
    # it, then globally shuffles with the given seed and reshuffle flag. Asserts the output has
    # the same words as expected but in a different order.
    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    metadata = _get_metadata(input_data)

    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = index_flat_map_op.index_flat_map(
        dataset, _split, _get_index_map_func(metadata), output_cardinality=9)
    self.assertEqual(self.evaluate(dataset.cardinality()), 9)

    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    expected = [
        b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8"] * repetitions
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(dataset_range=[0, 10])))
  def test_identity_map(self, dataset_range: int):
    # [FUNCTION DOC]
    # With an identity map and an index map sending index to (index, offset zero), a range
    # dataset (size zero or ten) is reproduced unchanged.

    def _map_func(element: Any) -> Any:
      # [FUNCTION DOC]
      # HELPER: identity map returning its input element.
      return element

    def _index_map_func(index: int) -> tuple[int, int]:
      # [FUNCTION DOC]
      # HELPER: maps an index to element index equal to the index with offset zero.
      return (index, 0)

    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = index_flat_map_op.index_flat_map(
        dataset, _map_func, _index_map_func)
    self.assertDatasetProduces(dataset, list(range(dataset_range)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_range=[0, 10],
              map_output=[
                  [[1, 2], [3, 4], [5, 6]],
                  [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
              ],
              use_tensors=[True, False])))
  def test_nested_list(
      self, input_range: int, map_output: list[Any], use_tensors: bool):
    # [FUNCTION DOC]
    # Each input element maps to a fixed nested list (optionally as a 64-bit tensor); the index
    # map splits global index by the list length. Asserts the output is the nested list's
    # sub-elements repeated for each input element.

    def _map_func(_) -> Union[tensor.Tensor, list[list[int]]]:
      # [FUNCTION DOC]
      # HELPER: ignores its input and returns the fixed nested list, as a tensor if use_tensors.
      return (constant_op.constant(map_output, dtype=dtypes.int64)
              if use_tensors else map_output)

    def _index_map_func(i: int) -> tuple[int, int]:
      # [FUNCTION DOC]
      # HELPER: maps index i to element index (integer division by the list length) and offset
      # (remainder).
      return (i // len(map_output), i % len(map_output))

    dataset = dataset_ops.Dataset.range(input_range)
    dataset = index_flat_map_op.index_flat_map(
        dataset, _map_func, _index_map_func)
    self.assertDatasetProduces(dataset, map_output * input_range)

  @combinations.generate(test_base.default_test_combinations())
  def test_offset_out_of_range(self):
    # [FUNCTION DOC]
    # Uses an index map that always returns offset 1000; asserts reading raises an
    # invalid-argument error about the invalid index map function offset.

    def _index_map_func(_) -> tuple[int, int]:
      # [FUNCTION DOC]
      # HELPER: ignores the index and returns element index zero with offset 1000.
      return (0, 1000)

    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = index_flat_map_op.index_flat_map(dataset, _split, _index_map_func)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "invalid `index_map_func` which returns offset 1000"):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def test_invalid_map_fn_type(self):
    # [FUNCTION DOC]
    # Uses an index map that returns a string rather than two integers; asserts reading raises an
    # invalid-argument error saying two int values were expected.

    def _index_map_func(_) -> str:
      # [FUNCTION DOC]
      # HELPER: deliberately invalid index map that returns a string.
      # Expected to return two integers.
      return "Hello"

    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = index_flat_map_op.index_flat_map(dataset, _split, _index_map_func)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "expected to return two int values"):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def test_unknown_cardinality(self):
    # [FUNCTION DOC]
    # Applies index flat map without declaring output cardinality, then global shuffle; asserts an
    # invalid-argument error saying global shuffle needs a non-empty finite cardinality.
    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "`global_shuffle` requires the input dataset to have a non-empty "
        "finite cardinality."):
      dataset = index_flat_map_op.index_flat_map(
          dataset, _split, _get_index_map_func(_get_metadata(input_data)))
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)


class IndexFlatMapCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for index flat map datasets, with and without global shuffle.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              repetitions=[1, 3],
              symbolic_checkpoint=[True, False])))
  def test_index_flat_map(
      self,
      verify_fn: Callable[..., None],
      repetitions: int,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Runs the checkpoint verification callback on an index flat map dataset (optionally
    # repeated, with symbolic checkpointing per parameter), expecting nine outputs per repetition.

    input_data = ["0 1", "2 3 4 5", "6 7", "8"]

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the index-flat-mapped, optionally repeated dataset with the symbolic
      # checkpoint option applied.
      dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
      dataset = index_flat_map_op.index_flat_map(
          dataset, _split, _get_index_map_func(_get_metadata(input_data)))
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(self, _build_dataset, num_outputs=9 * repetitions)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              repetitions=[1, 3],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def test_global_shuffle(
      self,
      verify_fn: Callable[..., None],
      repetitions: list[int],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Checkpoint verification on an index flat map dataset with declared cardinality nine, optional
    # repetition and a seeded global shuffle; compares outputs as unordered items when
    # reshuffling each iteration. The repetitions parameter is annotated as a list but is an int.

    input_data = ["0 1", "2 3 4 5", "6 7", "8"]

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the index-flat-mapped, repeated, globally shuffled dataset with the
      # symbolic checkpoint option applied.
      dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
      dataset = index_flat_map_op.index_flat_map(
          dataset,
          _split,
          _get_index_map_func(_get_metadata(input_data)),
          output_cardinality=9)
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)

      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=9 * repetitions,
        assert_items_equal=reshuffle_each_iteration)


def _split(element: str) -> tensor.Tensor:
  # [FUNCTION DOC]
  # HELPER: splits a string element on single spaces into a ragged tensor of words; used as the
  # map function of the index flat map tests.
  return ragged_string_ops.string_split_v2(element, " ")


def _get_metadata(input_data: list[str]) -> tensor.Tensor:
  # [FUNCTION DOC]
  # HELPER: builds the metadata matrix for a list of strings. Each row holds the start index,
  # end index (exclusive) of that string's words in the flattened output, and the string's
  # position. Returns a 64-bit integer tensor.
  """Given a list of strings, creates a metadata matrix."""

  metadata = []
  for i, data in enumerate(input_data):
    split_data = data.split()
    last_index = metadata[-1][1] if metadata else 0
    metadata.append((last_index, last_index + len(split_data), i))
  return constant_op.constant(metadata, dtype=dtypes.int64)


def _get_index_map_func(
    metadata: tensor.Tensor) -> Callable[[int], tuple[int, int]]:
  # [FUNCTION DOC]
  # HELPER: turns a metadata tensor (see the metadata builder) into an index map function that
  # takes a flattened index and returns the owning element index and the offset inside it.
  """Turns a `metadata` Tensor into an index map function."""

  def _index_map_func(index: Union[int, tensor.Tensor]) -> tuple[int, int]:
    # [FUNCTION DOC]
    # HELPER: scans the metadata rows until the index falls before a row's end, then returns that
    # row's position and the index minus the row's start. If the index is past all rows, returns
    # the row count with offset zero.
    element_index = 0
    while (element_index < metadata.shape[0] and
           index >= array_ops.gather_nd(metadata, [element_index, 1])):
      element_index += 1
    offset = (
        index - array_ops.gather_nd(metadata, [element_index, 0])
        if element_index < metadata.shape[0]
        else constant_op.constant(0, dtype=dtypes.int64))
    return (element_index, offset)

  return _index_map_func


if __name__ == "__main__":
  test.main()
