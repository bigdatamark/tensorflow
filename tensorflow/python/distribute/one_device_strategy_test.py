# [MODULE DOC]
# Tests for the one-device strategy, run across eager and graph modes with CPU and GPU one-device
# strategy combinations (and remote-worker variants). Most tests delegate to shared scenario
# helpers in the strategy test library; a few check input prefetching device placement directly.

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
"""Tests for class OneDeviceStrategy."""

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import device as tf_device


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.one_device_strategy,
            strategy_combinations.one_device_strategy_gpu
        ],
        mode=["eager", "graph"]))
class OneDeviceStrategyTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.OneDeviceDistributionTestBase):
  # [CLASS DOC]
  # Test case for the one-device strategy, parameterized over the one-device strategy on CPU and on
  # GPU, in eager and graph modes. Inherits scenario helpers from the shared distribution test
  # bases.

  def testMinimizeLoss(self, distribution):
    # [FUNCTION DOC]
    # Scenario: loss minimization with the strategy. Runs the eager or graph version of the shared
    # minimize-loss scenario depending on the current mode; assertions are in the helpers.
    if context.executing_eagerly():
      self._test_minimize_loss_eager(distribution)
    else:
      self._test_minimize_loss_graph(distribution)

  def testReplicaId(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared replica-id scenario (replica id in a replica context).
    self._test_replica_id(distribution)

  def testCallAndMergeExceptions(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared scenario checking error handling in call-for-each-replica
    # and merge-call.
    self._test_call_and_merge_exceptions(distribution)

  def testReplicateDataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: input function producing a dataset of range ten. Skipped in graph mode when TF2
    # behavior is enabled. Builds an input function that checks the input context (one replica, one
    # input pipeline, pipeline id zero) and asserts the iterable yields one-element lists of 0..9.
    if tf2.enabled() and not context.executing_eagerly():
      self.skipTest("Skipping test since we do not support graph mode in TF 2")
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i] for i in range(10)]
    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=1,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterable(distribution, input_fn, expected_values)

  def testMakeInputFnIteratorWithDataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: input-function iterator built from a function returning a range-ten dataset, with
    # the input context expected to show one replica, one pipeline and id zero. Asserts the iterator
    # yields the expected one-element lists on the worker devices.
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i] for i in range(10)]
    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=1,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    iterator = distribution.make_input_fn_iterator(input_fn)
    self._test_input_fn_iterator(
        iterator, distribution.extended.worker_devices, expected_values)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    # [FUNCTION DOC]
    # Scenario: like the dataset variant but the input function returns a callable (the next-element
    # method of a one-shot iterator). Asserts the iterator yields the expected values, ignoring
    # order and without testing reinitialization.
    def fn():
      # [FUNCTION DOC]
      # HELPER: input function body. Creates a range-ten dataset, makes a one-shot iterator and
      # returns its next-element callable.
      dataset = dataset_ops.Dataset.range(10)
      it = dataset_ops.make_one_shot_iterator(dataset)
      return it.get_next
    expected_values = [[i] for i in range(10)]
    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=1,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    iterator = distribution.make_input_fn_iterator(input_fn)
    self._test_input_fn_iterator(
        iterator, distribution.extended.worker_devices, expected_values,
        test_reinitialize=False, ignore_order=True)

  def testNumpyDataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared NumPy dataset scenario.
    self._test_numpy_dataset(distribution)

  def testRun(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared run scenario.
    self._test_run(distribution)

  def testAllReduceSum(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared all-reduce sum scenario.
    self._test_all_reduce_sum(distribution)

  def testAllReduceSumGradients(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared scenario for gradients of the all-reduce sum.
    self._test_all_reduce_sum_gradients(distribution)

  def testAllReduceSumGradientTape(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared scenario for all-reduce sum under a gradient tape.
    self._test_all_reduce_sum_gradient_tape(distribution)

  def testAllReduceMean(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared all-reduce mean scenario.
    self._test_all_reduce_mean(distribution)

  def testAllReduceMeanGradients(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared scenario for gradients of the all-reduce mean.
    self._test_all_reduce_mean_gradients(distribution)

  def testAllReduceMeanGradientTape(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared scenario for all-reduce mean under a gradient tape.
    self._test_all_reduce_mean_gradient_tape(distribution)

  def testTrainableVariables(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared trainable-variables scenario.
    self._test_trainable_variable(distribution)

  def test_prefetch_to_device_dataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: range-100 dataset batched by the replica count and distributed with fetching to the
    # device enabled. Takes the first element (eagerly, or through an initializable iterator for V1
    # distributed datasets; other graph combinations are skipped). Asserts the element's device type
    # equals the strategy's worker device type.
    input_options = distribute_lib.InputOptions(
        experimental_fetch_to_device=True)
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.batch(distribution.num_replicas_in_sync)
    dataset = distribution.experimental_distribute_dataset(
        dataset, options=input_options)
    if context.executing_eagerly():
      item = next(iter(dataset))
    else:
      if isinstance(dataset, input_lib_v1.DistributedDatasetV1):
        item = dataset.make_initializable_iterator().get_next()
      else:
        self.skipTest("unsupported test combination")
    device_types = (
        tf_device.DeviceSpec.from_string(item.device).device_type)
    expected_device_types = (
        tf_device.DeviceSpec.from_string(
            distribution.extended.worker_devices[0]).device_type)
    self.assertAllEqual(device_types, expected_device_types)

  def test_prefetch_to_host_dataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: same as the device prefetch test but with fetching to the device disabled. Asserts
    # the element's device type is CPU (host).
    input_options = distribute_lib.InputOptions(
        experimental_fetch_to_device=False)
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.batch(distribution.num_replicas_in_sync)
    dataset = distribution.experimental_distribute_dataset(
        dataset, options=input_options)
    if context.executing_eagerly():
      item = next(iter(dataset))
    else:
      if isinstance(dataset, input_lib_v1.DistributedDatasetV1):
        item = dataset.make_initializable_iterator().get_next()
      else:
        self.skipTest("unsupported test combination")
    self.assertAllEqual(
        tf_device.DeviceSpec.from_string(item.device).device_type, "CPU")


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.one_device_strategy_on_worker_1,
            strategy_combinations.one_device_strategy_gpu_on_worker_1
        ],
        mode=["eager", "graph"]))
class OneDeviceStrategyOnRemoteWorkerTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.OneDeviceDistributionTestBase):
  # [CLASS DOC]
  # Tests for the one-device strategy placed on a remote worker (CPU and GPU variants, eager and
  # graph modes). Checks colocation of the compute device and the input device.

  def testDeviceAndInputDeviceAreColocated(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared scenario asserting the strategy's device and its input
    # device are on the same worker.
    self._test_device_and_input_device_are_colocated(distribution)

  def testDeviceAndInputDeviceAreColocatedWithFunction(self, distribution):
    # [FUNCTION DOC]
    # Scenario: delegates to the shared scenario checking the same colocation when running inside a
    # compiled function.
    self._test_device_and_input_device_are_colocated_with_function(distribution)


if __name__ == "__main__":
  test.main()
