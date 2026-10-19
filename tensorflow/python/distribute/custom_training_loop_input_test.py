# [MODULE DOC]
# Tests for feeding distributed datasets into custom training loops under distribution
# strategies. Covers iteration inside and outside compiled functions, getting the next element as
# an optional, nested and dynamically shaped outputs, reductions over dynamic shapes, prefetch
# placement on devices, remainder handling when batches do not divide evenly, and out-of-range
# behavior. Defines a dataset helper, an assertion mixin and one large test class. No import-time
# side effects beyond imports; runs through the distribute test utility main entry point.

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for custom training loops."""

from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.tpu import tpu
from tensorflow.python.util import nest


def get_dataset_from_tensor_slices(inp_array):
  # [FUNCTION DOC]
  # HELPER: dataset construction helper shared by the tests in this module.
  # Takes an array-like input and returns a dataset with one element per slice along the first
  # dimension. Uses the V2 dataset class, but falls back to the V1 dataset class when TF2
  # behavior is not enabled (workaround for a tracked bug).
  dataset = dataset_ops.DatasetV2.from_tensor_slices(inp_array)
  # TODO(b/138326910): Remove Dataset V1 version once bug resolved.
  if not tf2.enabled():
    dataset = dataset_ops.Dataset.from_tensor_slices(inp_array)
  return dataset


class AssertFlattenedMixin(object):
  # [CLASS DOC]
  # HELPER: mixin for test cases that supplies a flattening comparison assertion for per-replica
  # results. Expects to be combined with a test case class providing the assertion methods.
  """Mixin for specialized asserts."""

  def assert_equal_flattened(self, expected_results, actual_results):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the input iteration tests.
    # Inputs: expected_results, a list of flat lists of numbers; actual_results, a list whose items
    # are sequences of per-replica tensors. Asserts equal lengths, then for each item concatenates
    # the numeric contents of the per-replica values in order and asserts equality with the
    # expected list. Returns nothing; fails the test on mismatch.
    """Asserts that flattened results are equal.

    Due to the number of replicas in the strategy, the output may have a
    different structure and needs to be flattened for comparison.

    Args:
      expected_results: The results expected as a result of a computation.
      actual_results: The actual results of a computation.
    """
    self.assertEqual(len(expected_results), len(actual_results))

    for i, expected_result in enumerate(expected_results):
      final_result = []
      actual_result = actual_results[i]
      for val in actual_result:
        final_result.extend(val.numpy())
      self.assertAllEqual(expected_result, final_result)


class InputIterationTest(test.TestCase, parameterized.TestCase,
                         AssertFlattenedMixin):
  # [CLASS DOC]
  # Tests for how distributed datasets and their iterators are consumed by custom training loops
  # under many distribution strategies (eager mode). Uses the flattening assertion mixin and the
  # module dataset helper; several tests are specific to TPU or multi-device strategies.

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testConstantNumpyInput(self, distribution):
    # [FUNCTION DOC]
    # Runs a compiled function that passes a plain constant (2.0) as the argument to a per-replica
    # computation squaring it. Asserts the local results are 4.0 on every replica.

    @def_function.function
    def run(x):
      # [FUNCTION DOC]
      # Runs the computation on the strategy with the given value and returns the local
      # per-replica results.

      def computation(x):
        # [FUNCTION DOC]
        # Returns the square of the input.
        return math_ops.square(x)

      outputs = distribution.experimental_local_results(
          distribution.run(computation, args=(x,)))
      return outputs

    self.assertAllEqual(
        constant_op.constant(4., shape=(distribution.num_replicas_in_sync)),
        run(2.))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testStatefulExperimentalRunAlwaysExecute(self, distribution):
    # [FUNCTION DOC]
    # Verifies that a stateful per-replica computation whose result is not used still executes.
    # A mean-aggregated variable is incremented inside a replica function run from a compiled
    # step; asserts the variable equals 1.0 afterwards.
    with distribution.scope():
      v = variables.Variable(
          0.0, aggregation=variables.VariableAggregation.MEAN)

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Runs the increment on the strategy, ignoring its result, and returns a scalar zero.

      def assign_add():
        # [FUNCTION DOC]
        # Adds 1.0 to the captured variable.
        v.assign_add(1.0)

      distribution.run(assign_add)
      return array_ops.zeros([])

    train_step()
    self.assertAllEqual(1.0, v.numpy())

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.strategies_minus_tpu,
          mode=["eager"]))
  def testFullEager(self, distribution):
    # [FUNCTION DOC]
    # For non-TPU strategies, iterates a distributed dataset in pure eager mode, running a
    # squaring step per batch on the strategy. Asserts flattened results are 25, 36, 49, 64.
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)

    def train_step(data):
      # [FUNCTION DOC]
      # Returns the square of the input batch.
      return math_ops.square(data)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = distribution.experimental_local_results(
          distribution.run(train_step, args=(x,)))
      results.append(output)
    self.assert_equal_flattened([[25., 36.], [49., 64.]], results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies, mode=["eager"]))
  def testGetNextAsOptional(self, distribution):
    # [FUNCTION DOC]
    # Takes the next element of a distributed iterator as an optional inside a compiled function,
    # extracts its value and runs a squaring step on the strategy. Asserts the first batch
    # squared (25, 36) is produced.
    data = [5., 6., 7., 8.]
    dataset = get_dataset_from_tensor_slices(data).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    iterator = iter(dist_dataset)

    def train_step(data):
      # [FUNCTION DOC]
      # Returns the square of the input batch.
      return math_ops.square(data)

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Fetches the next optional element from the iterator, takes its value, runs the step on
      # the strategy and returns the local per-replica results.
      return distribution.experimental_local_results(
          distribution.run(
              train_step, args=(iterator.get_next_as_optional().get_value(),)))

    self.assert_equal_flattened([[25., 36.]], [run(iterator)])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies, mode=["eager"]))
  def testGetNextAsOptionalExampleUsage(self, distribution):
    # [FUNCTION DOC]
    # Exercises the documented usage pattern of the optional-next API: a compiled loop of up to
    # six steps over a dataset of eight elements in batches of two, breaking out when the optional
    # has no value. Asserts only that execution completes without error.
    global_batch_size = 2
    steps_per_loop = 6
    dataset = dataset_ops.Dataset.range(
        8, output_type=dtypes.int32).batch(global_batch_size)
    distributed_iterator = iter(
        distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def train_fn(distributed_iterator):
      # [FUNCTION DOC]
      # Loop body: repeatedly gets the next optional element, stops when it is empty, and
      # otherwise runs the step on the strategy with the element's value. Returns nothing.

      def step_fn(x):
        # [FUNCTION DOC]
        # Identity step: returns its input unchanged.
        return x

      for _ in math_ops.range(steps_per_loop):
        optional_data = distributed_iterator.get_next_as_optional()
        if not optional_data.has_value():
          break
        distribution.run(step_fn, args=(optional_data.get_value(),))

    train_fn(distributed_iterator)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.tpu_strategies, mode=["eager"]))
  def testFullEagerTPU(self, distribution):
    # [FUNCTION DOC]
    # For TPU strategies, running a step outside a compiled function must fail. Asserts that a
    # not-implemented error mentioning that pure eager execution is unsupported is raised.
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)

    def train_step(data):
      # [FUNCTION DOC]
      # Returns the square of the input batch.
      return math_ops.square(data)

    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    with self.assertRaisesRegex(NotImplementedError,
                                "does not support pure eager execution"):
      distribution.run(train_step, args=(next(input_iterator),))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testStepInFunction(self, distribution):
    # [FUNCTION DOC]
    # Wraps the per-replica step in a compiled function and runs it on the strategy for each batch
    # of a distributed dataset. Asserts flattened results are 25, 36, 49, 64.
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)

    @def_function.function
    def train_step(data):
      # [FUNCTION DOC]
      # Returns the square of the input batch.
      return math_ops.square(data)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = distribution.experimental_local_results(
          distribution.run(train_step, args=(x,)))
      results.append(output)
    self.assert_equal_flattened([[25., 36.], [49., 64.]], results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testRunInFunction(self, distribution):
    # [FUNCTION DOC]
    # Puts the strategy run call itself inside a compiled function, invoked once per batch of a
    # distributed dataset. Asserts flattened results are 25, 36, 49, 64.
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)

    def train_step(data):
      # [FUNCTION DOC]
      # Returns the square of the input batch.
      return math_ops.square(data)

    @def_function.function
    def f_train_step(input_data):
      # [FUNCTION DOC]
      # Runs the step on the strategy with the given batch and returns the local results.
      return distribution.experimental_local_results(
          distribution.run(train_step, args=(input_data,)))

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = f_train_step(x)
      results.append(output)
    self.assert_equal_flattened([[25., 36.], [49., 64.]], results)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
          ],
          mode=["eager"]))
  def testNestedOutput(self, distribution):
    # [FUNCTION DOC]
    # For a multi-device CPU/GPU strategy and TPU strategies, runs a computation returning a list
    # holding a mapping of two values and maps the local-results function over that structure.
    # Asserts each replica's values equal its input minus one and plus one.
    dataset = get_dataset_from_tensor_slices([0, 1, 2, 3]).batch(2)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element, runs the computation on the strategy and returns the nested
      # structure of local per-replica results.

      def computation(x):
        # [FUNCTION DOC]
        # Returns a one-element list containing a mapping with "a" (input minus one) and "b"
        # (input plus one).
        return [{
            "a": x - 1,
            "b": x + 1
        }]

      inputs = next(iterator)
      outputs = distribution.run(computation, args=(inputs,))
      return nest.map_structure(distribution.experimental_local_results,
                                outputs)

    results = run(input_iterator)
    for replica in range(distribution.num_replicas_in_sync):
      # The input dataset is range(4), so the replica id is same as input.
      self.assertAllEqual(results[0]["a"][replica], [replica - 1])
      self.assertAllEqual(results[0]["b"][replica], [replica + 1])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testRunInFunctionAutoGraphApplication(self, distribution):
    # [FUNCTION DOC]
    # Checks that a plain (non-compiled) step passed to the strategy run call from inside a
    # compiled function is still converted correctly by graph conversion. Asserts flattened results
    # are 25, 36, 49, 64.
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)

    def train_step(data):
      # [FUNCTION DOC]
      # Returns the square of the input batch.
      return math_ops.square(data)

    @def_function.function
    def f_train_step(input_data):
      # [FUNCTION DOC]
      # Runs the step on the strategy with the given batch and returns the local results.
      return distribution.experimental_local_results(
          distribution.run(train_step, args=(input_data,)))

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = f_train_step(x)
      results.append(output)
    self.assert_equal_flattened([[25., 36.], [49., 64.]], results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetIterationInFunction(self, distribution):
    # [FUNCTION DOC]
    # Iterates a distributed dataset twice inside a compiled function: once accumulating a step
    # count and a product of batch means via reduce, once running a variable-incrementing step.
    # Asserts two steps, the expected product of means, and the variable going from 1 to 3.
    with distribution.scope():
      a = variables.Variable(
          1.0, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)

    def train_step(_):
      # [FUNCTION DOC]
      # Adds 1.0 to the captured variable, ignoring its input.
      a.assign_add(1.0)

    @def_function.function
    def f_train_step(dist_dataset):
      # [FUNCTION DOC]
      # Loops over the distributed dataset computing the step count and the running product of
      # mean-reduced batches, then loops again running the step. Returns both accumulated values.
      number_of_steps = constant_op.constant(0.0)
      product_of_means = constant_op.constant(2.0)
      for x in dist_dataset:  # loop with values modified each iteration
        number_of_steps += 1
        product_of_means *= math_ops.cast(
            distribution.reduce("MEAN", x, axis=0), product_of_means.dtype)

      for y in dist_dataset:  # loop with no intermediate state
        distribution.run(train_step, args=(y,))

      return number_of_steps, product_of_means

    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    number_of_steps, product_of_means = f_train_step(dist_dataset)
    self.assertEqual(2, number_of_steps.numpy())
    self.assertNear((2 * (5+6)/2 * (7+8)/2), product_of_means.numpy(), 1e-3)

    # We set the initial value of `a` to 1 and iterate through the dataset 2
    # times(4/2 where 4 is the number of dataset elements and 2 is the batch
    # size). Hence the final result is 3.
    self.assertEqual(3.0, (a.numpy()))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetAssertWithDynamicBatch(self, distribution):
    # [FUNCTION DOC]
    # Regression test for an issue with assertions on dynamically sized batches. A dataset of
    # three values in batches of two is iterated in a compiled function with a step that asserts
    # the maximum is at most 100 before squaring. Asserts flattened results 25, 36 then 49.
    # Regression test for github issue 33517.
    def step_fn(data):
      # [FUNCTION DOC]
      # Asserts the maximum of the batch is at most 100, then (under that control dependency)
      # returns the square of the batch.
      assert_op = control_flow_assert.Assert(
          math_ops.less_equal(math_ops.reduce_max(data), 100.), [data])
      with ops.control_dependencies([assert_op]):
        return math_ops.square(data)

    @def_function.function
    def train(dataset):
      # [FUNCTION DOC]
      # Iterates the distributed dataset for two steps, running the step on the strategy for each
      # element and collecting local results. Returns the list of per-step results.
      results = []
      iterator = iter(dataset)
      # we iterate through the loop 5 times since we have 3 elements and a
      # global batch of 2.
      for _ in range(2):
        elem = next(iterator)
        output = distribution.experimental_local_results(
            distribution.run(step_fn, args=(elem,)))
        results.append(output)
      return results

    dataset = dataset_ops.DatasetV2.from_tensor_slices([5., 6., 7.,]).batch(2)
    # TODO(b/138326910): Remove Dataset V1 version once bug resolved.
    if not tf2.enabled():
      dataset = dataset_ops.Dataset.from_tensor_slices([5., 6., 7.,]).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = train(dist_dataset)

    expected_results = [[25., 36.], [49.]]
    self.assertEqual(len(expected_results), len(results))

    # Need to expand results since output will be grouped differently depending
    # on the number of replicas.
    for i, expected_result in enumerate(expected_results):
      final_result = []
      actual_result = results[i]
      for val in actual_result:
        final_result.extend(val.numpy())
      self.assertAllEqual(expected_result, final_result)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDistributeDatasetIteratorWithoutFunction(self, distribution):
    # [FUNCTION DOC]
    # Builds a distributed dataset from a dataset-producing function and takes one element in pure
    # eager mode. Asserts the local results equal the first values of the data, one per replica.
    data = [5., 6., 7., 8.]
    input_iterator = iter(
        distribution.distribute_datasets_from_function(
            lambda _: get_dataset_from_tensor_slices(data)))

    self.assertAllEqual(
        distribution.experimental_local_results(input_iterator.get_next()),
        data[0:distribution.num_replicas_in_sync])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]
      ))
  def testDistributeDatasetIteratorWithFunction(self, distribution):
    # [FUNCTION DOC]
    # For multi-device strategies, gets the next element of a function-built distributed dataset
    # inside a compiled function. Asserts values equal the first data values and that each
    # result's backing device is the corresponding worker device.
    data = [5., 6., 7., 8.]
    input_iterator = iter(
        distribution.distribute_datasets_from_function(
            lambda _: get_dataset_from_tensor_slices(data)))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Returns the local per-replica results of the iterator's next element.
      return distribution.experimental_local_results(iterator.get_next())

    local_results = run(input_iterator)
    self.assertAllEqual(local_results,
                        data[0:distribution.num_replicas_in_sync])
    backing_devices = [result.backing_device for result in local_results]
    self.assertAllEqual(backing_devices, distribution.extended.worker_devices)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]
      ))
  def testDistributeDatasetPrefetch(self, distribution):
    # [FUNCTION DOC]
    # For multi-device strategies, takes the next element of a distributed dataset eagerly and
    # asserts each replica's value is already placed on its worker device (prefetch to device).
    data = [5., 6., 7., 8.]
    input_iterator = iter(
        distribution.experimental_distribute_dataset(
            get_dataset_from_tensor_slices(data).batch(2)))

    local_results = distribution.experimental_local_results(
        input_iterator.get_next())

    backing_devices = [result.backing_device for result in local_results]
    self.assertAllEqual(backing_devices, distribution.extended.worker_devices)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]
      ))
  def testDistributeDatasetFunctionPrefetch(self, distribution):
    # [FUNCTION DOC]
    # Same as the dataset prefetch test but the distributed dataset is built from a dataset
    # function. Asserts each replica's value is on its worker device.
    data = [5., 6., 7., 8.]
    input_iterator = iter(
        distribution.distribute_datasets_from_function(
            lambda _: get_dataset_from_tensor_slices(data)))

    local_results = distribution.experimental_local_results(
        input_iterator.get_next())

    backing_devices = [result.backing_device for result in local_results]
    self.assertAllEqual(backing_devices, distribution.extended.worker_devices)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.tpu_strategies,
          mode=["eager"]
      ))
  def testDistributeDatasetHostPrefetch(self, distribution):
    # [FUNCTION DOC]
    # For TPU strategies, distributes a dataset with the fetch-to-device option disabled and
    # asserts each replica's value stays on the host CPU device.
    data = [5., 6., 7., 8.]
    input_iterator = iter(
        distribution.experimental_distribute_dataset(
            get_dataset_from_tensor_slices(data).batch(2),
            distribute_lib.InputOptions(experimental_fetch_to_device=False)))

    local_results = distribution.experimental_local_results(
        input_iterator.get_next())

    for result in local_results:
      self.assertEqual(result.backing_device,
                       device_util.resolve("/device:CPU:0"))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.tpu_strategies,
          mode=["eager"]
      ))
  def testDistributeDatasetFunctionHostPrefetch(self, distribution):
    # [FUNCTION DOC]
    # For TPU strategies, builds a distributed dataset from a dataset function with the
    # fetch-to-device option disabled and asserts each replica's value stays on the host CPU.
    data = [5., 6., 7., 8.]
    input_iterator = iter(
        distribution.distribute_datasets_from_function(
            lambda _: get_dataset_from_tensor_slices(data),
            distribute_lib.InputOptions(experimental_fetch_to_device=False)))

    local_results = distribution.experimental_local_results(
        input_iterator.get_next())

    for result in local_results:
      self.assertEqual(result.backing_device,
                       device_util.resolve("/device:CPU:0"))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]
      ))
  def testDynamicShapes(self, distribution):
    # [FUNCTION DOC]
    # Distributes three values in one batch of four (so replicas get different sizes) and takes
    # the mean per replica in a compiled function. Assumes exactly two replicas; asserts means
    # 5.5 and 7.0.
    dataset = get_dataset_from_tensor_slices([5., 6., 7.]).batch(4)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element, runs the mean computation on the strategy and returns the local
      # results.
      def computation(x):
        # [FUNCTION DOC]
        # Returns the mean of the input.
        return math_ops.reduce_mean(x)
      inputs = next(iterator)
      outputs = distribution.experimental_local_results(
          distribution.run(computation, args=(inputs,)))
      return outputs

    # This assumes that there are exactly 2 replicas
    self.assertAllEqual([5.5, 7.], run(input_iterator))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.tpu_strategy, mode=["eager"]))
  def testDynamicShapesWithRunOptionsBucketizing(self, distribution):
    # [FUNCTION DOC]
    # For a TPU strategy, repeats the dynamic-shape mean test with run options enabling
    # bucketizing of dynamic shapes. Assumes two replicas; asserts means 5.5 and 7.0.
    dataset = get_dataset_from_tensor_slices([5., 6., 7.]).batch(4)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))
    options = distribute_lib.RunOptions(
        experimental_bucketizing_dynamic_shape=True)

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element, runs the mean computation with the run options and returns the
      # local results.

      def computation(x):
        # [FUNCTION DOC]
        # Returns the mean of the input.
        return math_ops.reduce_mean(x)

      inputs = next(iterator)
      outputs = distribution.experimental_local_results(
          distribution.run(
              computation, args=(inputs,), options=options))
      return outputs

    # This assumes that there are exactly 2 replicas
    self.assertAllEqual([5.5, 7.], run(input_iterator))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.tpu_strategy, mode=["eager"]))
  def testDynamicShapesWithRunOptionsDisableDynamicPadder(self, distribution):
    # [FUNCTION DOC]
    # For a TPU strategy, zips a data dataset with a mask dataset, disables the XLA dynamic padder
    # through run options and sums the masked values per replica. Assumes two replicas; asserts
    # sums 5 and 7.
    dataset = get_dataset_from_tensor_slices([5, 6, 7]).batch(4)
    mask_dataset = get_dataset_from_tensor_slices([1, 0, 1]).batch(4)
    dataset = dataset_ops.DatasetV2.zip((dataset, mask_dataset))

    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))
    options = distribute_lib.RunOptions(
        experimental_xla_options=tpu.XLAOptions(
            enable_xla_dynamic_padder=False))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element, runs the masked-sum computation with the run options and returns
      # the local results.

      def computation(inputs):
        # [FUNCTION DOC]
        # Takes a pair of data and mask values, multiplies them and returns the sum.
        x, mask = inputs
        y = x * mask
        return math_ops.reduce_sum(y)

      inputs = next(iterator)
      outputs = distribution.experimental_local_results(
          distribution.run(computation, args=(inputs,), options=options))
      return outputs

    # This assumes that there are exactly 2 replicas
    self.assertAllEqual([5, 7], run(input_iterator))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]))
  def testDynamicOutputsWithX64(self, distribution):
    # [FUNCTION DOC]
    # Distributes a single 64-bit integer value so one replica gets data and the other gets an
    # empty batch, and doubles it per replica. Assumes two replicas; asserts the first result is
    # 10 and the second is empty.
    dataset = get_dataset_from_tensor_slices(
        [5]).map(lambda x: math_ops.cast(x, dtypes.int64)).batch(2)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element, runs the doubling computation on the strategy and returns the
      # local results.

      def computation(x):
        # [FUNCTION DOC]
        # Returns the input added to itself.
        return math_ops.add(x, x)

      inputs = next(iterator)
      outputs = distribution.experimental_local_results(
          distribution.run(computation, args=(inputs,)))
      return outputs

    # This assumes that there are exactly 2 replicas
    result = run(input_iterator)
    self.assertAllEqual([10], result[0])
    self.assertAllEqual([], result[1])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]
      ))
  def testDynamicShapesWithGetNextOutsideFunction(self, distribution):
    # [FUNCTION DOC]
    # Like the dynamic-shape mean test, but the next element is fetched eagerly outside the
    # compiled function and passed in. Assumes two replicas; asserts means 5.5 and 7.0.
    dataset = get_dataset_from_tensor_slices([5., 6., 7.]).batch(4)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(inputs):
      # [FUNCTION DOC]
      # Runs the mean computation on the strategy with the given distributed input and returns the
      # local results.
      def computation(x):
        # [FUNCTION DOC]
        # Returns the mean of the input.
        return math_ops.reduce_mean(x)
      outputs = distribution.experimental_local_results(
          distribution.run(computation, args=(inputs,)))
      return outputs

    # This assumes that there are exactly 2 replicas
    self.assertAllEqual([5.5, 7.], run(next(input_iterator)))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]
      ))
  def testStrategyReduceWithDynamicShapes(self, distribution):
    # [FUNCTION DOC]
    # Reduces a dynamically sized distributed batch (three values, batch of four) across replicas
    # with a mean along the first axis, inside a compiled function. Asserts the result is 6.0.
    dataset = get_dataset_from_tensor_slices([5., 6., 7.]).batch(4)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element and returns its mean-reduction across replicas along axis zero.
      inputs = next(iterator)
      return distribution.reduce(reduce_util.ReduceOp.MEAN, inputs, axis=0)

    self.assertAllEqual(6., run(input_iterator))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]
      ))
  def testStrategyReduceWithDynamicShapesRank2(self, distribution):
    # [FUNCTION DOC]
    # Rank-two variant of the dynamic-shape reduce test: three rows of ones in a batch of four,
    # mean-reduced along axis zero in a compiled function. Asserts the result is a pair of ones.
    dataset = get_dataset_from_tensor_slices(
        [[1., 1.], [1., 1.], [1., 1.]]).batch(4)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element and returns its mean-reduction across replicas along axis zero.
      inputs = next(iterator)
      return distribution.reduce(reduce_util.ReduceOp.MEAN, inputs, axis=0)

    self.assertAllEqual([1., 1.], run(input_iterator))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]
      ))
  def testDynamicShapesWithSizeOp(self, distribution):
    # [FUNCTION DOC]
    # Computes the element count of each replica's dynamically sized slice of a batch of three
    # values. Assumes two replicas; asserts sizes 2 and 1.
    dataset = get_dataset_from_tensor_slices([5., 6., 7.]).batch(4)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(inputs):
      # [FUNCTION DOC]
      # Runs the size computation on the strategy with the given input and returns local results.
      def computation(x):
        # [FUNCTION DOC]
        # Returns the number of elements of the input.
        return array_ops.size_v2(x)
      outputs = distribution.experimental_local_results(
          distribution.run(computation, args=(inputs,)))
      return outputs

    # This assumes that there are exactly 2 replicas
    self.assertAllEqual([2, 1], run(next(input_iterator)))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]))
  def testSegmentSumWithDynamicNumberOfSegments(self, distribution):
    # [FUNCTION DOC]
    # Applies an unsorted segment sum whose segment count is the dynamic batch size, on a
    # dataset of five zeros in batches of three. Assumes two replicas; asserts output shapes
    # (3,) and (2,).

    def dataset_fn(_):
      # [FUNCTION DOC]
      # Dataset function (input context ignored): builds a dataset of five zero integers in
      # batches of three and returns it.
      data = array_ops.zeros(5, dtype=dtypes.int32)
      dataset = get_dataset_from_tensor_slices(data)
      dataset = dataset.batch(3)
      return dataset

    input_iterator = iter(
        distribution.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def step_fn(example):
      # [FUNCTION DOC]
      # Sums the example into a single segment, using the dynamic first dimension as the number
      # of segments, and returns the result.
      segment_ids = array_ops.zeros_like_v2(example)
      num_segment = array_ops.shape(example)[0]
      # If number of segments is dynamic, output should be a dynamic shape.
      return math_ops.unsorted_segment_sum(example, segment_ids, num_segment)

    # This assumes that there are exactly 2 replicas
    outputs = distribution.experimental_local_results(
        distribution.run(step_fn, args=(next(input_iterator),)))
    self.assertAllEqual((3,), outputs[0].shape)
    self.assertAllEqual((2,), outputs[1].shape)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]))
  def testReshapeWithDynamicInputs(self, distribution):
    # [FUNCTION DOC]
    # Tiles and reshapes dynamically sized inputs in several ways (including a duplicated
    # reshape output) to test dynamic dimension propagation. Assumes two replicas; asserts the
    # static shapes of all five outputs on each replica.

    def dataset_fn(_):
      # [FUNCTION DOC]
      # Dataset function (input context ignored): builds a dataset of five zero integer tensors
      # of shape one by two, batched by three, and returns it.
      data = array_ops.zeros((5, 1, 2), dtype=dtypes.int32)
      dataset = get_dataset_from_tensor_slices(data)
      dataset = dataset.batch(3)
      return dataset

    input_iterator = iter(
        distribution.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def step_fn(example):
      # [FUNCTION DOC]
      # Tiles the example along its second axis by its dynamic first dimension, then produces
      # four reshapes of the tiled value using the dynamic dimension in various positions.
      # Returns a list of five tensors, with the first reshape repeated at the end.
      # example: [<=3, 1, 2]
      # tile: [<=3, <=3, 2]
      tile = array_ops.tile(example, [1, array_ops.shape(example)[0], 1])
      # reshape1: [<=(3*3 = 9), 2]
      reshape1 = array_ops.reshape(tile, [-1, 2])

      # reshape2: [<=3, <=3, 2]
      reshape2 = array_ops.reshape(
          reshape1,
          [array_ops.shape(example)[0],
           array_ops.shape(example)[0], 2])

      # reshape3: [<=3, -1, 2]
      reshape3 = array_ops.reshape(reshape1,
                                   [array_ops.shape(example)[0], -1, 2])
      # reshape4: [-1, <=3, 2]
      reshape4 = array_ops.reshape(reshape1,
                                   [-1, array_ops.shape(example)[0], 2])
      # Reshape1 is duplicated in order to test dynamic dimension on copies.
      return [reshape1, reshape2, reshape3, reshape4, reshape1]

    # This assumes that there are exactly 2 replicas
    outputs = distribution.experimental_local_results(
        distribution.run(step_fn, args=(next(input_iterator),)))
    self.assertAllEqual((9, 2), outputs[0][0].shape)
    self.assertAllEqual((3, 3, 2), outputs[0][1].shape)
    self.assertAllEqual((3, 3, 2), outputs[0][2].shape)
    self.assertAllEqual((3, 3, 2), outputs[0][3].shape)
    self.assertAllEqual((9, 2), outputs[0][4].shape)

    self.assertAllEqual((4, 2), outputs[1][0].shape)
    self.assertAllEqual((2, 2, 2), outputs[1][1].shape)
    self.assertAllEqual((2, 2, 2), outputs[1][2].shape)
    self.assertAllEqual((2, 2, 2), outputs[1][3].shape)
    self.assertAllEqual((4, 2), outputs[1][4].shape)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]))
  def testDynamicShapesWithFirstReplicaNotMaximumShape(self, distribution):
    # [FUNCTION DOC]
    # Builds a dataset whose first batch has narrower rows than its second, so the first replica
    # does not hold the maximum shape, and takes the mean per replica. Assumes two replicas;
    # asserts means 1.5 and 2.0.
    def dataset_fn(_):
      # [FUNCTION DOC]
      # Dataset function (input context ignored): concatenates a dataset of two-wide rows with
      # one of three-wide rows and batches by two, dropping any remainder. Returns the dataset.
      dataset1 = get_dataset_from_tensor_slices([[1., 2.], [1., 2.]])
      dataset2 = get_dataset_from_tensor_slices([[1., 2., 3.],
                                                 [1., 2., 3.]])
      dataset = dataset1.concatenate(dataset2)
      dataset = dataset.batch(2, drop_remainder=True)
      return dataset

    input_iterator = iter(
        distribution.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(inputs):
      # [FUNCTION DOC]
      # Runs the mean computation on the strategy with the given distributed input and returns
      # the local results.
      def computation(x):
        # [FUNCTION DOC]
        # Returns the mean of the input.
        return math_ops.reduce_mean(x)
      outputs = distribution.experimental_local_results(
          distribution.run(computation, args=(inputs,)))
      return outputs

    # This assumes that there are exactly 2 replicas
    self.assertAllEqual([1.5, 2.], run(next(input_iterator)))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]))
  def testMapFnWithDynamicInputs(self, distribution):
    # [FUNCTION DOC]
    # Applies a mapped embedding lookup over dynamically sized batches of a 20 by 300 by 32 zero
    # dataset in batches of sixteen. Assumes two replicas; asserts the output shapes are 16 and 4
    # rows, each of shape 300 by 32 by 128.

    def dataset_fn(_):
      # [FUNCTION DOC]
      # Dataset function (input context ignored): builds a dataset of zero integer tensors of
      # shape 300 by 32, batched by sixteen, and returns it.
      data = array_ops.zeros((20, 300, 32), dtype=dtypes.int32)
      dataset = get_dataset_from_tensor_slices(data)
      dataset = dataset.batch(16)
      return dataset

    input_iterator = iter(
        distribution.distribute_datasets_from_function(dataset_fn))

    def embedding_lookup(inputs):
      # [FUNCTION DOC]
      # Gathers rows from a one-row zero embedding table of width 128 using the flattened inputs
      # and reshapes the result to the input shape with an extra trailing dimension of 128.
      embedding_weights = array_ops.zeros((1, 128))
      flat_inputs = array_ops.reshape(inputs, [-1])
      embeddings = array_ops.gather(embedding_weights, flat_inputs)
      embeddings = array_ops.reshape(embeddings, inputs.shape.as_list() + [128])
      return embeddings

    @def_function.function
    def step_fn(example):
      # [FUNCTION DOC]
      # Maps the embedding lookup over the example along its first axis with a float output
      # type and returns the stacked result.
      return map_fn.map_fn(
          embedding_lookup, example, fn_output_signature=dtypes.float32)

    # This assumes that there are exactly 2 replicas
    outputs = distribution.experimental_local_results(
        distribution.run(step_fn, args=(next(input_iterator),)))
    self.assertAllEqual((16, 300, 32, 128), outputs[0].shape)
    self.assertAllEqual((4, 300, 32, 128), outputs[1].shape)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetDistributeEvenlyDivisibleDrop(self, distribution):
    # [FUNCTION DOC]
    # Distributes two values in a batch of two with remainder dropping, a batch size evenly
    # divisible across workers. Takes one element eagerly and asserts the concatenated local
    # results are 5 and 6.
    # If the batch size is evenly divisible by the number of workers and we set
    # drop_remainder=True on the dataset, then DistributedIterator will use a
    # different (and more efficient) code path which avoids some control flow
    # ops.
    dataset = get_dataset_from_tensor_slices([5., 6.]).batch(
        2, drop_remainder=True)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    data = next(input_iterator)

    expected_result = [5., 6.]
    final_result = []
    actual_result = distribution.experimental_local_results(data)
    for val in actual_result:
      final_result.extend(val)
    self.assertAllEqual(expected_result, final_result)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetDistributeNotDivisibleDrop(self, distribution):
    # [FUNCTION DOC]
    # Distributes two values in batches of one with remainder dropping, a batch not evenly
    # divisible by the number of workers. Takes one element and asserts the concatenated local
    # results contain only the value 5.
    # If each batch is not evenly divisible by the number of workers,
    # the remainder will be dropped.
    dataset = get_dataset_from_tensor_slices([5., 6.]).batch(
        1, drop_remainder=True)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    data = next(input_iterator)

    expected_result = [5.]
    final_result = []
    actual_result = distribution.experimental_local_results(data)
    for val in actual_result:
      final_result.extend(val)
    self.assertAllEqual(expected_result, final_result)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetDistributeEvenlyDivisibleNoDrop(self, distribution):
    # [FUNCTION DOC]
    # Distributes two values in a batch of two without remainder dropping, forcing the optional
    # based iterator path. Takes one element and asserts the concatenated local results are 5 and 6.
    # Setting drop_remainder=False on the dataset causes DistributedIterator
    # to use get_next_as_optional(), even if the batched dataset is evenly
    # divisible by the number of workers.
    dataset = get_dataset_from_tensor_slices([5., 6.]).batch(
        2, drop_remainder=False)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    data = next(input_iterator)

    expected_result = [5., 6.]
    final_result = []
    actual_result = distribution.experimental_local_results(data)
    for val in actual_result:
      final_result.extend(val)
    self.assertAllEqual(expected_result, final_result)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetPartialBatchWithMixedOutputs(self, distribution):
    # [FUNCTION DOC]
    # Runs a computation returning both a fixed-size and a data-dependent output on a partial
    # batch of one value in a batch of two. Asserts the fixed output is zeros on every replica,
    # the first replica's dynamic output is 25, and the other replicas' dynamic outputs are empty.
    # Dynamic output size with a mix of static and dynamic outputs
    dataset = get_dataset_from_tensor_slices([5.]).batch(2)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element and runs the mixed-output computation on the strategy, returning
      # the per-replica result structure.

      def computation(x):
        # [FUNCTION DOC]
        # Returns a pair: a fixed vector of three zeros and the square of the input.
        # Fixed size output with a dynamic sized output.
        return array_ops.zeros([3]), math_ops.square(x)

      return distribution.run(
          computation, args=(next(iterator),))

    results = run(input_iterator)

    # First result is fixed for all replicas.
    for replica_id in range(distribution.num_replicas_in_sync):
      self.assertAllEqual([0., 0., 0.],
                          distribution.experimental_local_results(
                              results[0])[replica_id])
    # Only first replica has distributed dataset computation.
    self.assertAllEqual([25.],
                        distribution.experimental_local_results(results[1])[0])
    # Other replicas have no distributed dataset computation.
    for replica_id in range(1, distribution.num_replicas_in_sync):
      self.assertAllEqual([],
                          distribution.experimental_local_results(
                              results[1])[replica_id])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testIterationInsideFunction(self, distribution):
    # [FUNCTION DOC]
    # Iterates a distributed dataset of four values in batches of two inside a compiled function
    # for two steps, running a squaring step each time. Asserts flattened results 25, 36, 49, 64.

    def step_fn(data):
      # [FUNCTION DOC]
      # Returns the square of the input batch.
      return math_ops.square(data)

    @def_function.function
    def train(dataset):
      # [FUNCTION DOC]
      # Iterates the distributed dataset for two steps, running the step on the strategy for each
      # element. Returns the list of local per-replica results.
      results = []
      iterator = iter(dataset)
      # we iterate through the loop 2 times since we have 4 elements and a
      # global batch of 2.
      for _ in range(2):
        elem = next(iterator)
        output = distribution.experimental_local_results(
            distribution.run(step_fn, args=(elem,)))
        results.append(output)
      return results

    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = train(dist_dataset)
    self.assert_equal_flattened([[25., 36.], [49., 64.]], results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testIterationOutsideFunction(self, distribution):
    # [FUNCTION DOC]
    # Iterates the distributed dataset eagerly for two steps, calling a compiled function that
    # runs a squaring step on each fetched element. Asserts flattened results 25, 36, 49, 64.

    def train_step(data):
      # [FUNCTION DOC]
      # Returns the square of the input batch.
      return math_ops.square(data)

    @def_function.function
    def f_train_step(input_data):
      # [FUNCTION DOC]
      # Runs the step on the strategy with the given batch and returns the local results.
      return distribution.experimental_local_results(
          distribution.run(train_step, args=(input_data,)))

    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    iterator = iter(dist_dataset)
    results = []
    # we iterate through the loop 2 times since we have 4 elements and a
    # global batch of 2.
    for _ in range(2):
      output = f_train_step(next(iterator))
      results.append(output)
    self.assert_equal_flattened([[25., 36.], [49., 64.]], results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testMultiDeviceDataCapturedFunction(self, distribution):
    # [FUNCTION DOC]
    # Builds a distributed dataset from a function that captures a constant tensor, and uses a
    # variable inside a compiled training step that runs and reduces twice (once in a dataset
    # range loop). Asserts the mean result is 10.0 with one replica, otherwise 7.5.
    inputs = constant_op.constant([2., 3.])
    dataset = lambda _: dataset_ops.Dataset.from_tensor_slices(inputs).repeat(5)
    input_iterator = iter(
        distribution.distribute_datasets_from_function(dataset))
    with distribution.scope():
      var = variables.Variable(1.0)

    @def_function.function
    def train_step(input_iterator):
      # [FUNCTION DOC]
      # Runs the function on the strategy with the next element, mean-reduces the outputs,
      # repeats once inside a loop over a range dataset, and returns the last mean.

      def func(inputs):
        # [FUNCTION DOC]
        # Returns the square of the input plus the captured variable.
        return math_ops.square(inputs) + var

      per_replica_outputs = distribution.run(
          func, (next(input_iterator),))
      mean = distribution.reduce(
          reduce_util.ReduceOp.MEAN, per_replica_outputs, axis=None)
      for _ in dataset_ops.Dataset.range(1):
        per_replica_outputs = distribution.run(
            func, (next(input_iterator),))
        mean = distribution.reduce(
            reduce_util.ReduceOp.MEAN, per_replica_outputs, axis=None)
      return mean

    with distribution.scope():
      if distribution.num_replicas_in_sync == 1:
        self.assertAlmostEqual(10.0, self.evaluate(train_step(input_iterator)))
      else:
        self.assertAlmostEqual(7.5, self.evaluate(train_step(input_iterator)))

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testDatasetOutOfRange(self, distribution):
    # [FUNCTION DOC]
    # Runs a compiled step that sums each batch into a sum-aggregated variable, repeatedly, past
    # the end of a four-element dataset. Asserts an out-of-range error is raised and that the
    # variable ends at 26.0 (the sum of all elements).
    with distribution.scope():
      a = variables.Variable(
          0.0, aggregation=variables.VariableAggregation.SUM)

    def train_step(val):
      # [FUNCTION DOC]
      # Adds the sum of the input batch to the captured variable.
      a.assign_add(math_ops.reduce_sum(val))

    @def_function.function
    def f_train_step(iterator):
      # [FUNCTION DOC]
      # Runs the step on the strategy with the iterator's next element and returns the variable.
      distribution.run(train_step, args=(next(iterator),))
      return a

    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    iterator = iter(dist_dataset)
    with self.assertRaises(errors.OutOfRangeError):
      for _ in range(100):
        f_train_step(iterator)

    self.assertAlmostEqual(26.0, a.numpy())

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.multidevice_strategies,
          mode=["eager"]))
  def testComputeLossWithDynamicShapes(self, distribution):
    # [FUNCTION DOC]
    # Computes a weighted loss with unit weights on dynamically sized per-replica slices of a
    # batch of three values. Assumes two replicas; asserts the losses are 5.5 and 7.0.
    dataset = get_dataset_from_tensor_slices([5., 6., 7.]).batch(4)
    input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # Takes the next element, runs the loss computation on the strategy and returns the local
      # results.

      def computation(x):
        # [FUNCTION DOC]
        # Returns the weighted loss of the input using weights of one for every element.
        return losses.compute_weighted_loss(x, weights=array_ops.ones_like(x))

      inputs = next(iterator)
      outputs = distribution.experimental_local_results(
          distribution.run(computation, args=(inputs,)))
      return outputs

    # This assumes that there are exactly 2 replicas
    self.assertAllEqual([5.5, 7.], run(input_iterator))


if __name__ == "__main__":
  test_util.main()
