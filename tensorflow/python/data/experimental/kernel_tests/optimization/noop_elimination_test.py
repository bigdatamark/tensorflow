# [MODULE DOC]
# Tests for the NoopElimination tf.data optimization, which removes transformations that do not
# change the dataset (skip zero, repeat once, take of negative count, identity map, single shard,
# etc.) while keeping real ones. Contains a module-level generator of named transformation cases,
# one test class and a standard test main entry point.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `NoopElimination` optimization."""
import functools

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import test


def _test_combinations():
  # [FUNCTION DOC]
  # HELPER: builds the parameter combinations for the noop elimination test.
  # Takes no arguments. Returns a list of parameter dictionaries each with an initial-dataset
  # factory (range of ten), a named transformation and the expected name of the op that must
  # remain (nothing when the transformation is expected to be eliminated).

  def make_range():
    # [FUNCTION DOC]
    # HELPER: returns a new dataset of integers 0 through 9.
    return dataset_ops.Dataset.range(10)

  def fn_with_side_effect(arg):
    # [FUNCTION DOC]
    # Map body with a side effect: prints its argument and returns it unchanged.
    logging_ops.print_v2(arg)
    return arg

  # Test case for map function with capture args
  def apply_map_with_capture(ds):
    # [FUNCTION DOC]
    # Takes a dataset; returns it mapped to pairs of the element and a captured constant (-1).
    const = constant_op.constant(-1, dtype=dtypes.int64)
    return ds.map(lambda x: (x, const))

  # Test case for map functions with multiple components
  def apply_map_with_multiple_components(ds):
    # [FUNCTION DOC]
    # Takes a dataset; applies a parallel map duplicating the element into a pair (not
    # eliminated), then a pair-to-pair identity map (eliminated). Returns the resulting dataset.
    ds = ds.map(lambda x: (x, x), num_parallel_calls=2)  # Not eliminated
    return ds.map(lambda x, y: (x, y))  # Eliminated

  parallel_map_name = "ParallelMap"

  cases = [
      ("Skip0", lambda ds: ds.skip(0), None),
      ("SkipN", lambda ds: ds.skip(5), "FiniteSkip"),
      ("Repeat1", lambda ds: ds.repeat(1), None),
      ("RepeatN", lambda ds: ds.repeat(10), "FiniteRepeat[0]"),
      ("Prefetch0", lambda ds: ds.prefetch(0), "Prefetch"),
      ("PrefetchN", lambda ds: ds.prefetch(1), "Prefetch"),
      ("Take-1", lambda ds: ds.take(-1), None),
      ("TakeN", lambda ds: ds.take(2), "FiniteTake"),
      ("MapIdentity", lambda ds: ds.map(lambda x: x), None),
      ("MapNonIdentity", lambda ds: ds.map(lambda x: x * 2), "Map"),
      ("MapWithSideEffect", lambda ds: ds.map(fn_with_side_effect), "Map"),
      ("MapWithCapture", apply_map_with_capture, "Map"),
      (
          "MapWithMultipleComponents",
          apply_map_with_multiple_components,
          parallel_map_name,
      ),
      ("MapRestructure", lambda ds: ds.map(lambda x: {"value": x}), ""),
      (
          "PMapIdentity",
          lambda ds: ds.map(lambda x: x, num_parallel_calls=2),
          None,
      ),
      (
          "PMapNonIdentity",
          lambda ds: ds.map(lambda x: x * 2, num_parallel_calls=2),
          parallel_map_name,
      ),
      ("Shard1", lambda ds: ds.shard(1, 0), None),
      ("ShardN", lambda ds: ds.shard(2, 0), "Shard"),
  ]

  def reduce_fn(result, case):
    # [FUNCTION DOC]
    # HELPER: reducer folding cases into one combination list. Takes the accumulated list and a
    # case (name, transformation, expected op name); returns the list extended with a combination.
    name, transformation, expected = case
    return result + combinations.combine(
        init_dataset_fn=make_range,
        transformation=combinations.NamedObject(name, transformation),
        expected_name=expected)

  test_combinations = functools.reduce(reduce_fn, cases, [])

  return test_combinations


class NoopEliminationTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the noop elimination graph optimization of tf.data.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_combinations()))
  def testNoopElimination(self, init_dataset_fn, transformation, expected_name):
    # [FUNCTION DOC]
    # Scenario: builds the initial dataset, expects via assert_next either the named op followed
    # by a finite take (transformation kept) or just the finite take (transformation eliminated),
    # applies the transformation then take(1) with only noop_elimination enabled, and fetches one
    # element to trigger the assertion. Nothing else is asserted on the value.
    """Runs a noop elimination test case.

    Args:
      init_dataset_fn: Function to create the initial dataset
      transformation: Transformation to apply
      expected_name: Name of the transformation if it is not eliminated
    """
    dataset = init_dataset_fn()

    if expected_name:
      dataset = dataset.apply(
          testing.assert_next([expected_name, "FiniteTake"]))
    else:
      dataset = dataset.apply(testing.assert_next(["FiniteTake"]))

    dataset = dataset.apply(transformation)
    dataset = dataset.take(1)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)

    # Run the first iteration for the side effect of checking the assertion.
    get_next = self.getNext(dataset)
    self.evaluate(get_next())


if __name__ == "__main__":
  test.main()
