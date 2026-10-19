# [MODULE DOC]
# General tests for the static tf.data optimizations: behavior with stateful functions, very large
# inputs, nested datasets inside flat map, map parallelization and prefetch injection driven by
# options, datasets capturing reference variables in graph mode (optimizations must not change
# results), and warm-start of the iterator. Contains a module-level combination generator, one
# test class and a standard test main entry point.
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
"""Tests for the static tf.data optimizations."""
import functools
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _captured_refvar_test_combinations():
  # [FUNCTION DOC]
  # HELPER: builds dataset-factory parameters for the captured reference variable test.
  # Takes no arguments. Returns a list of parameter dictionaries, each holding a named dataset
  # factory (map, flat map, filter, map-and-batch, group-by-reducer, group-by-window, scan). Each
  # factory takes a variable and returns a small dataset whose functions capture that variable.

  def make_map_dataset(var):
    # [FUNCTION DOC]
    # Returns a one-element dataset mapped by adding the captured variable.
    return dataset_ops.Dataset.from_tensors(0).map(lambda x: x + var)

  def make_flat_map_dataset(var):
    # [FUNCTION DOC]
    # Returns a flat map over a one-element dataset yielding a dataset holding the variable.
    return dataset_ops.Dataset.from_tensors(
        0).flat_map(lambda _: dataset_ops.Dataset.from_tensors(var))

  def make_filter_dataset(var):
    # [FUNCTION DOC]
    # Returns a one-element dataset filtered by comparing the element to the captured variable.
    return dataset_ops.Dataset.from_tensors(0).filter(lambda x: x < var)

  def make_map_and_batch_dataset(var):
    # [FUNCTION DOC]
    # Returns a one-element dataset through the experimental map-and-batch (batch size 1) with a
    # function adding the captured variable.

    def map_fn(x):
      # [FUNCTION DOC]
      # Map body: adds the captured variable to the input.
      return x + var

    return dataset_ops.Dataset.from_tensors(0).apply(
        batching.map_and_batch(map_fn, 1))

  def make_group_by_reducer_dataset(var):
    # [FUNCTION DOC]
    # Returns range(5) grouped by parity with a reducer whose finalize function returns the
    # captured variable (init zero, reduce keeps the state).
    reducer = grouping.Reducer(
        init_func=lambda _: 0,
        reduce_func=lambda x, y: x,
        finalize_func=lambda _: var)
    return dataset_ops.Dataset.range(5).apply(
        grouping.group_by_reducer(lambda x: x % 2, reducer))

  def make_group_by_window_dataset(var):
    # [FUNCTION DOC]
    # Returns ten zeros grouped into one window of size ten (single key) with a window reduce
    # function that yields the captured variable.

    def reduce_fn(key, bucket):
      # [FUNCTION DOC]
      # Window reduce body: ignores key and bucket, returns a one-element dataset of the variable.
      del key, bucket
      return dataset_ops.Dataset.from_tensors(var)

    return dataset_ops.Dataset.from_tensors(0).repeat(10).apply(
        grouping.group_by_window(lambda _: 0, reduce_fn, 10))

  def make_scan_dataset(var):
    # [FUNCTION DOC]
    # Returns a one-element dataset through a scan whose step function adds the captured variable
    # to element plus state and increments the state.
    return dataset_ops.Dataset.from_tensors(0).apply(
        scan_ops.scan(
            0, lambda old_state, elem: (old_state + 1, elem + old_state + var)))

  cases = [
      # Core datasets
      ("Map", make_map_dataset),
      ("FlatMap", make_flat_map_dataset),
      ("Filter", make_filter_dataset),
      # Experimental datasets
      ("MapAndBatch", make_map_and_batch_dataset),
      ("GroupByReducer", make_group_by_reducer_dataset),
      ("GroupByWindow", make_group_by_window_dataset),
      ("Scan", make_scan_dataset)
  ]

  def reduce_fn(x, y):
    # [FUNCTION DOC]
    # HELPER: reducer folding cases into one combination list. Takes the accumulated list x and a
    # case y (name, dataset factory); returns x extended with the named combination.
    name, dataset_fn = y
    return x + combinations.combine(
        dataset_fn=combinations.NamedObject(name, dataset_fn))

  return functools.reduce(reduce_fn, cases, [])


class OptimizationTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for assorted behaviors of the static tf.data graph optimizations.

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationStatefulFunction(self):
    # [FUNCTION DOC]
    # Scenario: range(10) mapped by a random uniform scalar and batched, with default
    # optimizations off. Fetches one element to check a stateful function does not break.
    dataset = dataset_ops.Dataset.range(
        10).map(lambda _: random_ops.random_uniform([])).batch(10)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset)
    self.evaluate(get_next())

  # TODO(b/123354468)
  @combinations.generate(test_base.graph_only_combinations())
  def testOptimizationLargeInputFromTensor(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): a dataset from a placeholder fed with a very large integer array
    # (512 by 1024 by 1025) is initialized and one element fetched, to check large inputs work
    # when passed through the optimization pipeline. Nothing else asserted.
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None))
    dataset = dataset_ops.Dataset.from_tensors(input_t)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([512, 1024, 1025], np.int32)})
      self.evaluate(get_next)

  # TODO(b/123354468)
  @combinations.generate(test_base.graph_only_combinations())
  def testOptimizationLargeInputFromTensorSlices(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): like the large from-tensor test but slicing a placeholder fed
    # with a 1 by 512 by 1024 by 1025 integer array; initializes and fetches one element.
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None, None))
    dataset = dataset_ops.Dataset.from_tensor_slices(input_t)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([1, 512, 1024, 1025], np.int32)})
      self.evaluate(get_next)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationNestedDataset(self):
    # [FUNCTION DOC]
    # Scenario: a flat map whose inner dataset has a skip of zero before a cache; with only noop
    # elimination enabled the inner skip must be removed (inner assert_next expects the cache op
    # directly). Output is a single zero.

    def flat_map_fn(_):
      # [FUNCTION DOC]
      # Flat map body: returns a one-element dataset with an assert_next for the cache op, a
      # skip of zero (to be eliminated) and a cache.
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
      dataset = dataset.skip(0)  # Should be removed by noop elimination
      dataset = dataset.cache()
      return dataset

    dataset = dataset_ops.Dataset.range(1)
    dataset = dataset.flat_map(flat_map_fn)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationNestedDatasetWithModifiedRetval(self):
    # [FUNCTION DOC]
    # Scenario: a flat map whose inner dataset maps (identity) then batches by one; with only
    # map-and-batch fusion enabled the inner pipeline must fuse into MapAndBatch (inner
    # assert_next). Output is a single batch holding zero.

    def flat_map_fn(_):
      # [FUNCTION DOC]
      # Flat map body: returns a one-element dataset with an assert_next for MapAndBatch, an
      # identity map and a batch of size one.
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset = dataset.apply(testing.assert_next(["MapAndBatch"]))
      # Should be fused by map and batch fusion
      dataset = dataset.map(lambda x: x)
      dataset = dataset.batch(1)
      return dataset

    dataset = dataset_ops.Dataset.range(1)
    dataset = dataset.flat_map(flat_map_fn)

    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_and_batch_fusion = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[[0]])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(autotune=[True, False, None]),
          combinations.combine(map_parallelization=[True, False, None])))
  def testOptimizationMapParallelization(self, autotune, map_parallelization):
    # [FUNCTION DOC]
    # Scenario: map adding one over range(5) with the autotune-enabled option and the map
    # parallelization option each unset, true or false. assert_next expects ParallelMap unless
    # either is explicitly false (then Map). Output is 1..5.
    dataset = dataset_ops.Dataset.range(5)
    if autotune is not False and map_parallelization is not False:  # pylint: disable=g-bool-id-comparison
      dataset = dataset.apply(testing.assert_next(["ParallelMap"]))
    else:
      dataset = dataset.apply(testing.assert_next(["Map"]))
    dataset = dataset.map(lambda x: x + 1)

    options = options_lib.Options()
    if autotune is not None:
      options.autotune.enabled = autotune
    if map_parallelization is not None:
      options.experimental_optimization.map_parallelization = (
          map_parallelization)
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=list(range(1, 6)))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(existing_prefetch=[True, False]),
                         combinations.combine(autotune=[True, False]),
                         combinations.combine(inject_prefetch=[True, False])))
  def testOptimizationInjectPrefetch(self, existing_prefetch, autotune,
                                     inject_prefetch):
    # [FUNCTION DOC]
    # Scenario: range(5) mapped (autotuned parallelism) and batched by one, optionally followed by
    # an existing prefetch. Parameters: existing_prefetch, autotune enabled, inject_prefetch
    # option. A Prefetch op is expected before the root only when autotune and injection are on
    # and no prefetch already exists; otherwise only the root is expected. Output is arrays [1]..[5].
    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.map(
        lambda x: x + 1, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.batch(1)
    if existing_prefetch:
      dataset = dataset.prefetch(1)
    if autotune and inject_prefetch and not existing_prefetch:
      dataset = dataset.apply(testing.assert_next(["Prefetch", "Root"]))
    else:
      dataset = dataset.apply(testing.assert_next(["Root"]))

    options = options_lib.Options()
    options.autotune.enabled = autotune
    options.experimental_optimization.map_and_batch_fusion = False
    if not inject_prefetch:
      options.experimental_optimization.inject_prefetch = False
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=[np.array([x]) for x in
                                                         range(1, 6)])

  # Reference variables are not supported in eager mode.
  @combinations.generate(
      combinations.times(test_base.graph_only_combinations(),
                         _captured_refvar_test_combinations()))
  def testOptimizationWithCapturedRefVar(self, dataset_fn):
    # [FUNCTION DOC]
    # Scenario (graph mode only, reference variable): a dataset built by the given factory
    # captures a non-resource variable that is incremented before each fetch. An optimized copy
    # (noop elimination and map-and-batch fusion on, warm start off) and the unoptimized dataset
    # are iterated in lock step until end of data; outputs must be equal each step, showing
    # optimizations are disabled for datasets capturing reference variables.
    """Tests that default optimizations are disabled with ref variables."""
    variable = variable_scope.get_variable(
        "v", initializer=0, use_resource=False)
    assign_op = variable.assign_add(1)
    unoptimized_dataset = dataset_fn(variable)

    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_warm_start = False
    optimized_dataset = unoptimized_dataset.with_options(options)
    optimized_it = dataset_ops.make_initializable_iterator(optimized_dataset)

    # Check that outputs are the same in the optimized and unoptimized cases,
    # when the variable value is changing.
    unoptimized_it = dataset_ops.make_initializable_iterator(
        unoptimized_dataset)
    with ops.control_dependencies([assign_op]):
      unoptimized_output = unoptimized_it.get_next()
      optimized_output = optimized_it.get_next()

    self.evaluate(variable.initializer)
    self.evaluate((unoptimized_it.initializer, optimized_it.initializer))
    while True:
      try:
        unoptimized, optimized = self.evaluate((unoptimized_output,
                                                optimized_output))
        self.assertEqual(unoptimized, optimized)
      except errors.OutOfRangeError:
        break

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(warm_start=[True, False]),
      )
  )
  def testOptimizationWarmStart(self, warm_start):
    # [FUNCTION DOC]
    # Scenario (eager only): range(10) mapped by a function incrementing a counter variable and
    # prefetched, with the warm-start option on or off. An iterator is created but never read. If
    # warm start is on, the counter must eventually become positive (polled with growing sleeps up
    # to ten seconds); if off, the counter must still be zero.
    dataset = dataset_ops.Dataset.range(10)
    counter = variables.Variable(0)

    def update_counter(x):
      # [FUNCTION DOC]
      # Map body: increments the captured counter variable and returns the input unchanged.
      counter.assign_add(1)
      return x

    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    if warm_start:
      options.experimental_warm_start = True
    else:
      options.experimental_warm_start = False
    dataset = dataset.with_options(options)
    dataset = dataset.map(update_counter).prefetch(10)
    unused_iter = iter(dataset)

    if warm_start:
      for sleep_time_secs in [0.1, 0.2, 0.5, 2, 5, 10]:
        if counter.numpy() == 0:
          time.sleep(sleep_time_secs)
        else:
          break
      self.assertGreater(counter.numpy(), 0)
    else:
      self.assertEqual(counter.numpy(), 0)


if __name__ == "__main__":
  test.main()
