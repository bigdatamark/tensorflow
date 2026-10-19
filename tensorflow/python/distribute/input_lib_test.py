# [MODULE DOC]
# Unit tests for the distributed input pipeline library (input_lib and input_util) of
# tf.distribute. The tests build datasets or dataset-producing functions, wrap them for a
# distribution strategy (mirrored, central storage, TPU, one-device, multi-worker mirrored), and
# check the per-replica values produced by iterating, including batch splitting, uneven final
# batches, auto-sharding options, ragged/sparse tensors, caching/shuffling across epochs, static
# shape propagation of optional get-next, device placement controlled by input options, and
# reading through the tf.data service.
# Contents: a shared base test class with iteration helpers, and four parameterized test classes
# (general iterator tests, tensor-type tests, per-device input option tests, tf.data service
# tests). Test methods are generated per strategy via the combinations machinery, so many
# combinations are skipped when the needed hardware or cluster is absent.
# Import-time side effects: none beyond imports; running the file as a program starts the
# standard distributed test main entry point.
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
"""Tests for the input_lib library."""

import collections

from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util as framework_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor as ragged_tensor_lib
from tensorflow.python.util import nest


class DistributedIteratorTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the distributed input tests in this module.
  # Base test case providing helpers that wrap a dataset or input function into a distributed
  # dataset or iterator (V1 or V2 flavor depending on whether TF2 behavior is enabled), iterate
  # it with the requested iteration style, and compare per-replica values against expectations.
  # It defines no tests itself; the concrete test classes below derive from it.

  # The passed input_context is to create a sharded dataset in between-graph
  # case.
  # TODO(yuefengz): rewrite the following method to make it less DRY.
  def _wrap_iterator(self,
                     input_type,
                     dataset_or_input_fn,
                     input_workers,
                     devices,
                     num_replicas_in_sync,
                     strategy,
                     input_context=None):
    # [FUNCTION DOC]
    # HELPER: builds a V1 distributed iterator for the iterator-wrapping test combinations.
    # Creates a V1 distributed iterator from either an input function or a dataset.
    # Parameters: input_type (the string input_fn or anything else meaning dataset), the dataset or
    # input function, the input workers object, the list of devices, the number of replicas in sync,
    # the strategy, and an optional input context (only valid when input type is a dataset; it is used
    # for sharding in multi-worker mirrored strategy).
    # For the input function type it asserts the input context is absent, creates one input context
    # per worker (pipeline count equal to the worker count, replica count equal to the number of
    # devices) and returns an input-function iterator. Otherwise it returns a dataset iterator built
    # with the replica count and optional input context. Returns the iterator. There is a
    # to-do note in the code asking for a less repetitive rewrite.
    # The `input_context` passed in is to shard dataset for
    # MultiWorkerMirroredStrategy. It doesn't apply to in-graph case where
    # multiple InputContexts are needed.
    if input_type == "input_fn":
      self.assertIsNone(
          input_context,
          msg=("`The input_context` arg is only used to shard dataset in "
               "`MultiWorkerMirroredStrategy` when the input type is dataset."))

      input_contexts = []
      for i in range(input_workers.num_workers):
        input_contexts.append(
            distribute_lib.InputContext(
                # Note: `input_workers.num_workers` is always 1 in between-graph
                # case.
                num_input_pipelines=input_workers.num_workers,
                input_pipeline_id=i,
                num_replicas_in_sync=len(devices)))

      iterator = input_lib_v1.InputFunctionIterator(dataset_or_input_fn,
                                                    input_workers,
                                                    input_contexts, strategy)
    else:
      iterator = input_lib_v1.DatasetIterator(
          dataset_or_input_fn,
          input_workers,
          strategy,
          num_replicas_in_sync=num_replicas_in_sync,
          input_context=input_context)
    return iterator

  def _wrap_dataset(self,
                    input_type,
                    dataset,
                    input_workers,
                    num_replicas_in_sync,
                    strategy,
                    input_context=None):
    # [FUNCTION DOC]
    # HELPER: wraps a dataset into a distributed dataset for the dataset-wrapping test combinations.
    # Parameters: input_type, the dataset (or dataset function for the function type), the input
    # workers, the number of replicas in sync, the strategy and an optional input context.
    # When input type is dataset it returns a distributed dataset: the V2 class when TF2 behavior is
    # enabled, otherwise the V1 class. For any other input type it asks the strategy to distribute
    # datasets from the given function and returns the result.
    if input_type == "dataset":
      if tf2.enabled():
        return input_lib.DistributedDataset(
            input_workers,
            strategy,
            dataset,
            num_replicas_in_sync=num_replicas_in_sync,
            input_context=input_context)
      else:
        return input_lib_v1.DistributedDatasetV1(
            dataset,
            input_workers,
            strategy,
            num_replicas_in_sync=num_replicas_in_sync,
            input_context=input_context)
    else:
      return strategy.distribute_datasets_from_function(dataset)

  def _assert_iterator_values(self,
                              iterator,
                              expected_values,
                              evaluate_fn,
                              devices,
                              enable_get_next_as_optional=False):
    # [FUNCTION DOC]
    # HELPER: checks that an iterator yields the expected per-replica values in order.
    # Parameters: the iterator, the list of expected values (one entry per step, each a list with one
    # value per replica), an evaluation callback, the list of devices, and a flag (default false)
    # choosing whether to fetch elements through the optional get-next API and unwrap the value.
    # For each expected step it fetches the next element, selects each replica's component, evaluates
    # them together and records the result; afterwards it asserts every replica value equals the
    # expected one. Returns nothing. Errors from the iterator (such as running out of data) propagate.
    actual_values = []
    for _ in range(len(expected_values)):
      if enable_get_next_as_optional:
        next_element = iterator.get_next_as_optional().get_value()
      else:
        next_element = iterator.get_next()
      computed_value = evaluate_fn([
          distribute_utils.select_replica(r, next_element)
          for r in range(len(devices))
      ])
      actual_values.append(computed_value)
    for expected_value, actual_value in zip(expected_values, actual_values):
      for expected, actual in zip(expected_value, actual_value):
        self.assertAllEqual(expected, actual)

  def _assert_dataset_values_for_loop(self, dataset, expected_values,
                                      evaluate_fn, devices):
    # [FUNCTION DOC]
    # HELPER: checks the values produced by a plain loop over a distributed dataset.
    # Parameters: the distributed dataset, the expected values (a list of steps, each a list of
    # per-replica values), an evaluation callback (unused; the test case evaluate method is used
    # instead) and the list of devices.
    # Loops over the dataset, evaluates every replica's component of each element, and then asserts
    # each collected value equals the expected value. Returns nothing.
    actual_values = []
    for x in dataset:
      computed_value = self.evaluate(
          [distribute_utils.select_replica(r, x) for r in range(len(devices))])
      actual_values.append(computed_value)
    for expected_value, actual_value in zip(expected_values, actual_values):
      for expected, actual in zip(expected_value, actual_value):
        self.assertAllEqual(expected, actual)

  def _test_input_iteration(self,
                            input_type,
                            api_type,
                            iteration_type,
                            dataset_or_input_fn,
                            worker_device_pairs,
                            expected_values,
                            strategy,
                            sess=None,
                            num_replicas_in_sync=None,
                            input_context=None):
    # [FUNCTION DOC]
    # HELPER: shared driver that runs one input iteration scenario for most tests in this module.
    # Parameters: input_type (input function or dataset), api_type (wrap into iterator or wrap into
    # dataset), iteration_type (get next or for loop), the dataset or input function, the list of
    # worker and device pairs, the expected values, the strategy, an optional session, an optional
    # replica count, and an optional input context.
    # Skips the test for unsupported combinations (for loop outside eager mode, iterator wrapping
    # with for loop, or iterator wrapping with input functions). Otherwise it builds the input
    # workers, then creates an iterator (via the iterator helper) or a distributed dataset (via the
    # dataset helper, then an iterator in eager mode or an initializable iterator in graph mode when
    # available; otherwise skips). If the iterator is a composite tensor it asserts that its structure
    # matches its type spec. For get-next iteration it checks values, expects an out-of-range error
    # when exhausted, re-creates or re-initializes the iterator and checks values again; in TF2 mode
    # with dataset wrapping it also checks the optional get-next path, including that the final
    # optional has no value and that fetching further raises an invalid-argument error. For the for
    # loop type in eager mode it loops over the dataset and compares values. Returns nothing.
    if iteration_type == "for_loop" and not context.executing_eagerly():
      self.skipTest("unsupported test combination.")

    if api_type == "wrap_into_iterator" and iteration_type == "for_loop":
      self.skipTest("unsupported test combination.")

    if api_type == "wrap_into_iterator" and input_type == "input_fn":
      self.skipTest("unsupported test combination.")

    devices = nest.flatten([ds for _, ds in worker_device_pairs])
    input_workers = input_lib.InputWorkers(worker_device_pairs)

    if api_type == "wrap_into_iterator":
      iterator = self._wrap_iterator(
          input_type,
          dataset_or_input_fn,
          input_workers,
          devices,
          num_replicas_in_sync,
          strategy,
          input_context=input_context)
    else:
      # wrapping into a dataset:
      dataset = self._wrap_dataset(
          input_type,
          dataset_or_input_fn,
          input_workers,
          num_replicas_in_sync,
          strategy,
          input_context=input_context)

      if ops.executing_eagerly_outside_functions():
        iterator = iter(dataset)
      else:
        if isinstance(dataset, input_lib_v1.DistributedDatasetV1):
          iterator = dataset.make_initializable_iterator()
        else:
          self.skipTest("unsupported test combination")

    if isinstance(iterator, composite_tensor.CompositeTensor):
      nest.assert_same_structure(
          iterator, iterator._type_spec, expand_composites=True)

    if iteration_type == "get_next":
      evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
      if not ops.executing_eagerly_outside_functions():
        evaluate(control_flow_ops.group(iterator.initializer))

      def test_get_next(iterator):
        # [FUNCTION DOC]
        # HELPER: nested check used by the driver for plain get-next iteration.
        # Takes an iterator. Asserts the expected values come out, asserts a further pass raises an
        # out-of-range error, then re-initializes the iterator (graph mode) or creates a new one from the
        # dataset (eager; skips the test when wrapping into an iterator) and asserts the values again.
        # Returns nothing.
        self._assert_iterator_values(iterator, expected_values, evaluate,
                                     devices)

        with self.assertRaises(errors.OutOfRangeError):
          self._assert_iterator_values(iterator, expected_values, evaluate,
                                       devices)

        # After re-initializing the iterator, should be able to iterate again.
        if not ops.executing_eagerly_outside_functions():
          evaluate(control_flow_ops.group(iterator.initializer))
        else:
          if api_type == "wrap_into_iterator":
            self.skipTest("unsupported test combination")
          else:
            iterator = iter(dataset)

        self._assert_iterator_values(iterator, expected_values, evaluate,
                                     devices)

      def test_get_next_as_optional(iterator):
        # [FUNCTION DOC]
        # HELPER: nested check used by the driver for optional get-next iteration.
        # Takes an iterator. Asserts the expected values come out through the optional API, then that the
        # next optional holds no value, and that trying to read a value from further optionals raises an
        # invalid-argument error. Returns nothing.
        self._assert_iterator_values(
            iterator,
            expected_values,
            evaluate,
            devices,
            enable_get_next_as_optional=True)

        next_element = iterator.get_next_as_optional()
        self.assertFalse(self.evaluate(next_element.has_value()))
        with self.assertRaises(errors.InvalidArgumentError):
          self._assert_iterator_values(
              iterator, [0],
              evaluate,
              devices,
              enable_get_next_as_optional=True)

      test_get_next(iterator)

      # re-initializing the iterator
      if not tf2.enabled():
        # TODO(yuefengz): we should split this function.
        return
      else:
        if api_type == "wrap_into_iterator":
          return
        else:
          iterator = iter(dataset)

      test_get_next_as_optional(iterator)

    if iteration_type == "for_loop" and context.executing_eagerly():
      self._assert_dataset_values_for_loop(dataset, expected_values,
                                           self.evaluate, devices)

  def _create_dataset_or_input_fn(self, input_type, input_fn):
    # [FUNCTION DOC]
    # HELPER: picks what to hand to the wrapping helpers.
    # Parameters: input_type and an input function that takes an input context. Returns the function
    # itself when input type is input function; otherwise calls it with a default input context and
    # returns the resulting dataset.
    if input_type == "input_fn":
      return input_fn
    else:
      return input_fn(distribute_lib.InputContext())


class DistributedIteratorTest(DistributedIteratorTestBase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # General tests for distributed datasets and iterators across strategies, devices and
  # iteration styles. Covers basic iteration, tuple elements, auto-shard policy, uneven batches,
  # batch splitting, caching and shuffling across epochs, optional get-next static shapes, and
  # looping inside compiled functions. Uses the helpers of the base class; combinations of
  # strategies and options are generated by decorators.

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu
          ]))
  def testMultiDeviceIterInitialize(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a V1 one-shot iterator over a distributed dataset with a GPU and a CPU replica
    # device. Skipped when TF2 behavior is enabled. Builds a distributed dataset of ten numbers,
    # makes a one-shot iterator and runs the initializer evaluation inside a compiled function.
    # Expectation: the call completes without error (no explicit assertion).
    if tf2.enabled():
      self.skipTest("Only V1 is supported.")
    worker_device_pairs = [("/device:CPU:0", ["/device:GPU:0",
                                              "/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.DatasetV1.range(10)

    input_workers = input_lib.InputWorkers(worker_device_pairs)

    dist_dataset = input_util.get_distributed_dataset(
        dataset_fn(distribute_lib.InputContext()), input_workers, distribution)

    iterator = dataset_ops.make_one_shot_iterator(dist_dataset)

    @def_function.function
    def init_func_for_iter():
      # [FUNCTION DOC]
      # Nested compiled function that evaluates the iterator initializer through the test case
      # evaluate method. Takes no arguments, returns nothing.
      self.evaluate(iterator.initializer)

    init_func_for_iter()

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testOneDeviceCPU(self, input_type, api_type, iteration_type, distribution,
                       enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: one CPU device, ten-element range dataset, with all iteration and wrapping
    # combinations and both settings of the optional get-next flag on the strategy.
    # Sets the flag, then runs the shared iteration driver expecting one value per step, in order.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.Dataset.range(10)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [[i] for i in range(10)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(input_type, api_type, iteration_type,
                               dataset_or_input_fn, worker_device_pairs,
                               expected_values, distribution)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["input_fn", "dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_one_gpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
      )
  )
  def testAutoShardExplicit(self, input_type, distribution):
    # [FUNCTION DOC]
    # Scenario: explicit auto-sharding of a distributed dataset in two shards (eager only).
    # Builds a batched range dataset, wraps it, applies the auto shard function with two shards for
    # index zero and then index one, and iterates each result.
    # Expectation: shard zero yields the even numbers and shard one the odd numbers, grouped in pairs
    # per step when the strategy has two worker devices, or one per step otherwise.
    worker_device_pairs = [(
        "/device:CPU:0",
        distribution.extended.worker_devices,
    )]
    dataset_fn = lambda _: dataset_ops.Dataset.range(10).batch(1)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn
    )
    input_workers = input_lib.InputWorkers(worker_device_pairs)
    distribution.extended.experimental_enable_get_next_as_optional = True
    dataset = self._wrap_dataset(
        input_type,
        dataset_or_input_fn,
        input_workers,
        num_replicas_in_sync=None,
        strategy=distribution)

    dataset1 = input_ops.auto_shard_dataset(dataset, 2, 0)
    iterator = iter(dataset1)

    if len(distribution.extended.worker_devices) == 2:
      expected_values = [[0, 2], [4, 6], [8]]
    else:
      expected_values = [[0], [2], [4], [6], [8]]
    for element, expected in zip(iterator, expected_values):
      local = distribution.experimental_local_results(element)
      local_list = array_ops.concat(local, axis=0).numpy().tolist()
      self.assertAllEqual(local_list, expected)

    if len(distribution.extended.worker_devices) == 2:
      expected_values = [[1, 3], [5, 7], [9]]
    else:
      expected_values = [[1], [3], [5], [7], [9]]
    dataset2 = input_ops.auto_shard_dataset(dataset, 2, 1)
    iterator = iter(dataset2)
    for element, expected in zip(iterator, expected_values):
      local = distribution.experimental_local_results(element)
      local_list = array_ops.concat(local, axis=0).numpy().tolist()
      self.assertAllEqual(local_list, expected)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[strategy_combinations.multi_worker_mirrored_2x1_cpu],
          enable_get_next_as_optional=[True, False]))
  def testOneDeviceCPUMultiWorker(self, input_type, api_type, iteration_type,
                                  distribution, enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: multi-worker mirrored strategy with two workers and one CPU each, dataset wrapping
    # only. Sets the optional get-next flag, then runs the iteration driver with a range dataset.
    # Expectation: values arrive one per step in order (no sharding is passed in).
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.DatasetV1.range(10)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [[i] for i in range(10)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(input_type, api_type, iteration_type,
                               dataset_or_input_fn, worker_device_pairs,
                               expected_values, distribution)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu
          ],
          enable_get_next_as_optional=[True, False]))
  def testTwoDevicesOneGPUOneCPU(self, input_type, api_type, iteration_type,
                                 distribution, enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: one worker with a GPU and a CPU replica device for mirrored and central storage
    # strategies. Runs the iteration driver on a ten-element range dataset.
    # Expectation: each step yields two consecutive numbers, one per replica.
    worker_device_pairs = [("/device:CPU:0", ["/device:GPU:0",
                                              "/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.Dataset.range(10)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [[i, i + 1] for i in range(0, 10, 2)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(input_type, api_type, iteration_type,
                               dataset_or_input_fn, worker_device_pairs,
                               expected_values, distribution)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[strategy_combinations.tpu_strategy],
          enable_get_next_as_optional=[True, False]))
  def testTPU(self, input_type, api_type, iteration_type, distribution,
              enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: TPU strategy. Groups the strategy worker devices by their host device into an
    # ordered mapping and uses its items as worker and device pairs, then runs the iteration driver
    # on a range dataset. Expectation: each step yields two consecutive numbers.
    worker_device_pairs = collections.OrderedDict()
    for tpu_device in distribution.extended.worker_devices:
      host_device = device_util.get_host_for_device(tpu_device)
      worker_device_pairs.setdefault(host_device, [])
      worker_device_pairs[host_device].append(tpu_device)
    worker_device_pairs = worker_device_pairs.items()
    dataset_fn = lambda _: dataset_ops.Dataset.range(10)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [[i, i + 1] for i in range(0, 10, 2)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(input_type, api_type, iteration_type,
                               dataset_or_input_fn, worker_device_pairs,
                               expected_values, distribution)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTupleDataset(self, input_type, api_type, iteration_type, distribution,
                       enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: elements are pairs of a number and its square, with two replica devices.
    # Builds the dataset by zipping a range with its squared version, then runs the iteration driver.
    # Expectation: each step contains two pairs (consecutive numbers with their squares).
    worker_device_pairs = [("/device:CPU:0", ["/device:GPU:0",
                                              "/device:CPU:0"])]

    def dataset_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory. Takes an input context (ignored) and returns a dataset zipping
      # a ten-element range with the squares of the same range.
      del ctx
      dataset1 = dataset_ops.Dataset.range(10)
      dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
      return dataset_ops.Dataset.zip((dataset1, dataset2))

    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [
        [(i, i**2), (i + 1, (i + 1)**2)] for i in range(0, 10, 2)
    ]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(input_type, api_type, iteration_type,
                               dataset_or_input_fn, worker_device_pairs,
                               expected_values, distribution)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call
          ],
          enable_get_next_as_optional=[True, False]))
  def testTupleDatasetMultiworker(self, input_type, api_type, iteration_type,
                                  distribution, enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: same tuple dataset as the previous test but with multi-worker mirrored strategy and
    # two GPUs, dataset wrapping only. No input context is passed, so there is no sharding.
    # Expectation: each step contains two pairs of consecutive numbers and their squares.
    worker_device_pairs = [("/device:CPU:0", ["/device:GPU:0",
                                              "/device:GPU:1"])]

    def dataset_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory. Takes an input context (ignored) and returns a dataset zipping
      # a ten-element range with the squares of the same range.
      del ctx
      dataset1 = dataset_ops.Dataset.range(10)
      dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
      return dataset_ops.Dataset.zip((dataset1, dataset2))

    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [
        [(i, i**2), (i + 1, (i + 1)**2)] for i in range(0, 10, 2)
    ]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    # Input_context is not passed in and thus no sharding.
    self._test_input_iteration(input_type, api_type, iteration_type,
                               dataset_or_input_fn, worker_device_pairs,
                               expected_values, distribution)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ]))
  def testIterableIterator(self, distribution):
    # [FUNCTION DOC]
    # Scenario: iterating a distributed dataset iterator in a plain loop for strategies with a single
    # replica per worker. Builds a distributed dataset from a ten-element range and enumerates over
    # the iterator. Expectation: the local results of the i-th element equal a list with the number i.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]
    input_workers = input_lib.InputWorkers(worker_device_pairs)

    dataset = dataset_ops.Dataset.range(10)
    dist_dataset = input_util.get_distributed_dataset(dataset, input_workers,
                                                      distribution)

    iterator = iter(dist_dataset)
    for i, element in enumerate(iterator):
      self.assertAllEqual(distribution.experimental_local_results(element), [i])

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_two_cpus,
          ],
          use_iterator=[False, True]))
  def testIteratorAndDatasetEnumerateError(self, distribution, use_iterator):
    # [FUNCTION DOC]
    # Scenario: enumerating a distributed dataset, or an iterator of one (selected by the use iterator
    # parameter), inside a compiled function. Expectation: it raises a not-implemented error because
    # enumerate is not supported within compiled functions for these types.
    # enumerate is not supported within tf.function for these types.
    dataset = dataset_ops.Dataset.range(10).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    if use_iterator:
      iterable = iter(dist_dataset)
    else:
      iterable = dist_dataset

    @def_function.function
    def enumerate_fn(iterable):
      # [FUNCTION DOC]
      # Nested compiled function that enumerates the given iterable and gets the local results of each
      # batch. Used to provoke the not-implemented error. Returns nothing.
      for _, batch in enumerate(iterable):
        distribution.experimental_local_results(batch)

    with self.assertRaises(NotImplementedError):
      enumerate_fn(iterable)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_two_cpus,
          ]))
  def testIterableIteratorError(self, distribution):
    # [FUNCTION DOC]
    # Scenario: using the iterator of a distributed dataset outside and then inside the strategy
    # scope. First, advancing the iterator inside a replica function run without a scope must raise a
    # value error. Then inside the scope the first element is passed through a replica function and
    # its local results must equal the first expected batch (the whole batch for one replica, or one
    # element per replica for two). Finally a fresh iterator is looped without scope (the default
    # replica context) and every element's local results must equal the expected batches.
    dataset = dataset_ops.Dataset.range(10).batch(2)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    iterator = iter(dist_dataset)
    # Raises error when next(iterator) is called without strategy scope
    with self.assertRaises(ValueError):

      def replica_fn1(iterator):
        # [FUNCTION DOC]
        # Nested replica function that advances the given iterator and returns its next element. Expected
        # to fail with a value error when run outside the strategy scope.
        return next(iterator)

      distribution.run(replica_fn1, args=(iterator,))

    if distribution.num_replicas_in_sync == 1:
      expected_result = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[8, 9]]]
    elif distribution.num_replicas_in_sync == 2:
      expected_result = [[[0], [1]], [[2], [3]], [[4], [5]], [[6], [7]],
                         [[8], [9]]]

    with distribution.scope():

      def replica_fn2(iterator):
        # [FUNCTION DOC]
        # Nested replica function that returns its argument unchanged.
        return iterator

      result = distribution.run(replica_fn2, args=(next(iterator),))
      self.assertAllEqual(
          distribution.experimental_local_results(result), expected_result[0])

    # Confirm default ReplicaContext also works
    iterator = iter(dist_dataset)
    for i, element in enumerate(iterator):
      self.assertAllEqual(
          distribution.experimental_local_results(element), expected_result[i])

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          drop_remainder=[True, False],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu
          ]))
  def testUnevenDatasetBatches(self, input_type, api_type, iteration_type,
                               drop_remainder, distribution):
    # [FUNCTION DOC]
    # Scenario: nine elements batched by two, with and without dropping the remainder, with a GPU
    # and a CPU replica device. The optional get-next flag is turned on.
    # Expectation: with dropping, two full global batches of two replicas; without dropping, a third
    # global batch in which only the first replica has data (one element) and the second is empty.
    worker_device_pairs = [("/device:CPU:0", ["/device:GPU:0",
                                              "/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.Dataset.range(9).batch(  # pylint: disable=g-long-lambda
        2, drop_remainder=drop_remainder)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    # The last global batch only contains data for one replica.
    if drop_remainder:
      expected_values = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    else:
      expected_values = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8], []]]
    distribution.extended.experimental_enable_get_next_as_optional = True
    self._test_input_iteration(input_type, api_type, iteration_type,
                               dataset_or_input_fn, worker_device_pairs,
                               expected_values, distribution)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          drop_remainder=[True, False],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ]))
  def testUnevenDatasetBatchesMultiWorker(self, input_type, api_type,
                                          iteration_type, drop_remainder,
                                          distribution):
    # [FUNCTION DOC]
    # Scenario: multi-worker mirrored strategy, two workers with one replica each, nine elements.
    # Reads the worker count and this worker's index from the cluster resolver (asserted present).
    # For the input function type the dataset is manually sharded by worker and batched by one; for
    # the dataset type it is batched by two with the given remainder option.
    # Expectation: dataset type with dropping gives four steps of the worker's shard; otherwise five
    # steps where the last step has one element on worker zero and an empty batch on worker one.
    # Passes the replica count and an input context from the strategy to the driver.
    # Actual devices don't matter in this test as long as the number of global
    # repices is 2.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]
    cr = distribution.cluster_resolver
    self.assertIsNotNone(cr)
    worker_count = multi_worker_util.worker_count(cr.cluster_spec(),
                                                  cr.task_type)
    id_in_cluster = multi_worker_util.id_in_cluster(cr.cluster_spec(),
                                                    cr.task_type, cr.task_id)

    def dataset_fn(_):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory. Takes an ignored input context and builds a nine-element range;
      # for the input function type it shards by worker and batches by one, otherwise batches by two
      # with the remainder option.
      dataset = dataset_ops.Dataset.range(9)

      if input_type == "input_fn":
        # When input_fn is used, there is no automatic rebatching and sharding,
        # so we add them here.
        return dataset.shard(worker_count, id_in_cluster).batch(1)
      else:
        return dataset.batch(2, drop_remainder=drop_remainder)

    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    if drop_remainder and input_type == "dataset":
      if id_in_cluster == 0:
        expected_values = [[[0]], [[2]], [[4]], [[6]]]
      else:
        expected_values = [[[1]], [[3]], [[5]], [[7]]]
    else:
      # The last global batch only contains data for one replica.
      if id_in_cluster == 0:
        expected_values = [[[0]], [[2]], [[4]], [[6]], [[8]]]
      else:
        expected_values = [[[1]], [[3]], [[5]], [[7]], [[]]]
    distribution.extended.experimental_enable_get_next_as_optional = True
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution,
        num_replicas_in_sync=distribution.num_replicas_in_sync,
        input_context=distribution.extended._make_input_context())

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          drop_remainder=[True, False],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call
          ]))
  def testUnevenDatasetBatchesMultiWorkerFourReplicas(self, input_type,
                                                      api_type, iteration_type,
                                                      drop_remainder,
                                                      distribution):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment says the number of global replicas is two, but the strategy used here
    # has four global replicas (two workers with two GPUs each).
    # Scenario: like the two-replica multi-worker test but with two GPUs per worker and fifteen
    # elements, batched by four for the dataset type. Expectation: with dropping, three steps of two
    # per-replica batches of this worker's shard; otherwise four steps, where the last step has only
    # one element on worker one's first replica and nothing on its second replica.
    # Actual devices don't matter in this test as long as the number of global
    # repices is 2.
    worker_device_pairs = [("/device:CPU:0", ["/device:GPU:0",
                                              "/device:GPU:1"])]
    cr = distribution.cluster_resolver
    self.assertIsNotNone(cr)
    worker_count = multi_worker_util.worker_count(cr.cluster_spec(),
                                                  cr.task_type)
    id_in_cluster = multi_worker_util.id_in_cluster(cr.cluster_spec(),
                                                    cr.task_type, cr.task_id)

    def dataset_fn(_):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory. Takes an ignored input context and builds a fifteen-element
      # range; for the input function type it shards by worker and batches by one, otherwise batches
      # by four with the remainder option.
      dataset = dataset_ops.Dataset.range(15)

      if input_type == "input_fn":
        # When input_fn is used, there is no automatic rebatching and sharding,
        # so we add them here.
        return dataset.shard(worker_count, id_in_cluster).batch(1)
      else:
        return dataset.batch(4, drop_remainder=drop_remainder)

    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    # The last global batch only contains data for one replica.
    if drop_remainder and input_type == "dataset":
      if id_in_cluster == 0:
        expected_values = [[[0], [2]], [[4], [6]], [[8], [10]]]
      else:
        expected_values = [[[1], [3]], [[5], [7]], [[9], [11]]]
    else:
      if id_in_cluster == 0:
        expected_values = [[[0], [2]], [[4], [6]], [[8], [10]], [[12], [14]]]
      else:
        expected_values = [[[1], [3]], [[5], [7]], [[9], [11]], [[13], []]]
    distribution.extended.experimental_enable_get_next_as_optional = True
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution,
        num_replicas_in_sync=distribution.num_replicas_in_sync,
        input_context=distribution.extended._make_input_context())

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          num_replicas_in_sync=[None, 2],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu
          ],
          enable_get_next_as_optional=[True, False]))
  def testBatchSplitting(self, input_type, api_type, iteration_type,
                         num_replicas_in_sync, distribution,
                         enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: global batches of ten from a hundred-element range are split across two replicas;
    # the replica count in sync is either unset or two. Computes the per-replica batch size (the
    # batch size divided by the replica count when given, otherwise the batch size) and expects each
    # step to hold two consecutive ranges of that size. Runs the iteration driver.
    worker_device_pairs = [("/device:CPU:0", ["/device:GPU:0",
                                              "/device:CPU:0"])]
    batch_size = 10
    dataset_fn = lambda _: dataset_ops.Dataset.range(100).batch(batch_size)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    updated_batch_size = (
        batch_size //
        num_replicas_in_sync if num_replicas_in_sync else batch_size)
    expected_values = [[
        range(i, i + updated_batch_size),
        range(i + updated_batch_size, i + 2 * updated_batch_size)
    ] for i in range(0, 100, updated_batch_size * 2)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution,
        sess=None,
        num_replicas_in_sync=num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          api_type=["wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          num_replicas_in_sync=[None, 2],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call
          ],
          enable_get_next_as_optional=[True, False]))
  def testBatchSplittingMultiWorker(self, input_type, api_type, iteration_type,
                                    num_replicas_in_sync, distribution,
                                    enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: batch splitting as in the single-worker version but with multi-worker mirrored
    # strategy and two GPUs per worker (dataset wrapping only). Asserts a cluster resolver exists,
    # builds a batched hundred-element dataset, and expects each step to hold two consecutive ranges
    # of the per-replica batch size. Runs the iteration driver.
    worker_device_pairs = [("/device:CPU:0", ["/device:GPU:0",
                                              "/device:GPU:1"])]
    batch_size = 10
    cr = distribution.cluster_resolver
    self.assertIsNotNone(cr)

    def dataset_fn(_):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory. Takes an ignored input context and returns a hundred-element
      # range dataset batched by the batch size of the enclosing test.
      dataset = dataset_ops.Dataset.range(100).batch(batch_size)
      return dataset

    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    updated_batch_size = (
        batch_size //
        num_replicas_in_sync if num_replicas_in_sync else batch_size)
    expected_values = [
        [  # pylint: disable=g-complex-comprehension
            range(i, i + updated_batch_size),
            range(i + updated_batch_size, i + 2 * updated_batch_size)
        ] for i in range(0, 100, updated_batch_size * 2)
    ]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution,
        sess=None,
        num_replicas_in_sync=num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ],
      ))
  def testCacheAcrossIteration(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a shuffled, cached, batched dataset of sixteen elements distributed by the strategy
    # (skipped unless TF2 behavior is enabled). Collects the local results of one full pass twice.
    # Expectation: both epochs are identical, because the cache fixes the order after the first pass.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    dataset = dataset_ops.Dataset.range(16).shuffle(16).cache().batch(4)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    first_epoch = list(
        distribution.experimental_local_results(x) for x in dist_dataset)
    second_epoch = list(
        distribution.experimental_local_results(x) for x in dist_dataset)

    self.assertAllEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ],
          reshuffle=[True, False]))
  def testShuffleAcrossIterations(self, distribution, reshuffle):
    # [FUNCTION DOC]
    # Scenario: a shuffled twelve-element dataset batched by four with reshuffling each iteration
    # turned on or off per the parameter (skipped unless TF2 behavior is enabled). Collects two
    # epochs of local results. Expectation: epochs differ when reshuffling is on and are equal when
    # it is off.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    dataset = dataset_ops.Dataset.range(12).shuffle(
        12, reshuffle_each_iteration=reshuffle).batch(4)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    first_epoch = list(
        distribution.experimental_local_results(x) for x in dist_dataset)
    second_epoch = list(
        distribution.experimental_local_results(x) for x in dist_dataset)

    if reshuffle:
      self.assertNotAllEqual(first_epoch, second_epoch)
    else:
      self.assertAllEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ]))
  def testGetNextOptionalShapeFinite(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a finite dataset of feature and label tensors batched with remainder dropped. In a
    # compiled function, loops over the distributed dataset and takes local results.
    # Expectation: the leading dimension of each replica's feature and label is unknown (static
    # shape with an unknown batch size) while the feature width stays ten.
    batch_size = 8
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "feature": array_ops.ones([batch_size, 10]),
        "label": array_ops.ones([batch_size]),
    })
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    @def_function.function
    def train_fn():
      # [FUNCTION DOC]
      # Nested compiled function that loops over the distributed dataset and asserts static shapes of
      # each replica's feature (unknown batch size, width ten) and label (unknown batch size). Takes no
      # arguments, returns nothing.
      for data in dist_dataset:
        data = nest.map_structure(distribution.experimental_local_results, data)
        feature = data["feature"]
        label = data["label"]

        # Assert the shapes are still static from all replicas.
        for replica_id in range(len(distribution.extended.worker_devices)):
          self.assertEqual([None, 10],
                           feature[replica_id].shape.as_list())
          self.assertEqual([None], label[replica_id].shape.as_list())

    train_fn()

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ]))
  def testGetNextOptionalShapeInfinite(self, distribution):
    # [FUNCTION DOC]
    # Scenario: an infinite (repeated) batched dataset of features and labels. In a compiled function
    # takes one element through the optional get-next API and unwraps its value.
    # Expectation: each replica's feature has the per-replica batch size and width ten, and the label
    # has the per-replica batch size, as static shapes.
    batch_size = 8
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "feature": array_ops.ones([batch_size, 10]),
        "label": array_ops.ones([batch_size]),
    })
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    per_replica_batch_size = batch_size // distribution.num_replicas_in_sync

    @def_function.function
    def train_fn():
      # [FUNCTION DOC]
      # Nested compiled function that fetches one element through optional get-next and asserts the
      # static shapes of every replica's feature and label equal the per-replica batch size (and width
      # ten for features). Takes no arguments, returns nothing.
      data = iter(dist_dataset).get_next_as_optional().get_value()
      data = nest.map_structure(distribution.experimental_local_results, data)
      feature = data["feature"]
      label = data["label"]

      # Assert the shapes are still static from all replicas.
      for replica_id in range(len(distribution.extended.worker_devices)):
        self.assertEqual([per_replica_batch_size, 10],
                         feature[replica_id].shape.as_list())
        self.assertEqual([per_replica_batch_size],
                         label[replica_id].shape.as_list())

    train_fn()

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ]))
  def testGetNextOptionalShapeEmpty(self, distribution):
    # [FUNCTION DOC]
    # Scenario: same infinite dataset as the previous test, but inspects the element spec of the
    # optional rather than a value. Expectation: the component specs of the feature and label for
    # each replica have the per-replica batch size as the static leading dimension (and width ten for
    # features). Single (non-tuple) component specs are wrapped into one-element tuples first.
    batch_size = 8
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "feature": array_ops.ones([batch_size, 10]),
        "label": array_ops.ones([batch_size]),
    })
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    per_replica_batch_size = batch_size // distribution.num_replicas_in_sync

    @def_function.function
    def train_fn():
      # [FUNCTION DOC]
      # Nested compiled function that gets an optional from the iterator without reading its value and
      # asserts the static shapes in its element spec for each replica. Takes no arguments, returns
      # nothing.
      data = iter(dist_dataset).get_next_as_optional()
      feature_specs = data.element_spec["feature"]._component_specs
      value_specs = data.element_spec["label"]._component_specs
      if not isinstance(feature_specs, tuple):
        feature_specs = (feature_specs,)
        value_specs = (value_specs,)
      # Assert the shapes are still static from all replicas.
      for replica_id in range(len(distribution.extended.worker_devices)):
        self.assertEqual([per_replica_batch_size, 10],
                         feature_specs[replica_id].shape.as_list())
        self.assertEqual([per_replica_batch_size],
                         value_specs[replica_id].shape.as_list())

    train_fn()

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ],
          input_type=["dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          auto_shard_policy=[AutoShardPolicy.AUTO, AutoShardPolicy.OFF]))
  def testAutoshardingOption(self, distribution, input_type, api_type,
                             iteration_type, auto_shard_policy):
    # [FUNCTION DOC]
    # Scenario: multi-worker mirrored strategy (two workers) with a four-element range dataset whose
    # options set the auto shard policy to automatic or off. Computes this worker's index and runs
    # the iteration driver with an input context from the strategy.
    # Expectation: with automatic sharding worker zero sees the numbers zero and two and worker one
    # sees one and three; with sharding off every worker sees all four numbers.
    cr = distribution.cluster_resolver
    self.assertIsNotNone(cr)
    id_in_cluster = multi_worker_util.id_in_cluster(cr.cluster_spec(),
                                                    cr.task_type, cr.task_id)
    ds_option = options_lib.Options()
    ds_option.experimental_distribute.auto_shard_policy = auto_shard_policy
    dataset_fn = (
        lambda _: dataset_ops.Dataset.range(4).with_options(ds_option))
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]
    if auto_shard_policy == AutoShardPolicy.AUTO:
      if id_in_cluster == 0:
        expected_values = [[0], [2]]
      else:
        expected_values = [[1], [3]]
    else:
      expected_values = [[0], [1], [2], [3]]
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution,
        input_context=distribution.extended._make_input_context())

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ],
          input_type=["input_fn"],
          api_type=["wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"]))
  def testDifferentDatasetsMultiWorker(self, distribution, input_type, api_type,
                                       iteration_type):
    # [FUNCTION DOC]
    # Scenario: with an input function, pipeline zero produces four batches of two (eight elements)
    # and the other pipeline produces nine elements batched by two (five batches). The optional
    # get-next flag is on.
    # Expectation: the first four steps are the same on both workers; the fifth step is empty on
    # worker zero and holds the single leftover element on the other worker.
    cr = distribution.cluster_resolver
    self.assertIsNotNone(cr)
    id_in_cluster = multi_worker_util.id_in_cluster(cr.cluster_spec(),
                                                    cr.task_type, cr.task_id)

    def dataset_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an input context; returns eight elements batched by two
      # for input pipeline zero and nine elements batched by two for any other pipeline.
      if ctx.input_pipeline_id == 0:
        return dataset_ops.Dataset.range(8).batch(2)
      else:
        return dataset_ops.Dataset.range(9).batch(2)

    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]

    if id_in_cluster == 0:
      expected_values = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[]]]
    else:
      expected_values = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[8]]]
    distribution.extended.experimental_enable_get_next_as_optional = True
    self._test_input_iteration(input_type, api_type, iteration_type,
                               dataset_or_input_fn, worker_device_pairs,
                               expected_values, distribution)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ],
          mode=["eager"]))
  def testLoopOverDatasetInTFFunction(self, strategy):
    # [FUNCTION DOC]
    # Scenario: looping over a distributed dataset inside a compiled function on a multi-worker
    # mirrored strategy. A ten-element dataset of squared floats is batched by four and distributed.
    # A sum-aggregated variable is created in the strategy scope; the compiled function loops over
    # the dataset and runs a replica function adding the batch sum to the variable.
    # Expectation: the variable equals 285 (the sum of squares of zero through nine).
    dataset = dataset_ops.Dataset.range(10).map(lambda x: {  # pylint: disable=g-long-lambda
        "y": math_ops.cast(x, dtypes.float32) ** 2,
    }).batch(4)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
      v = variables.Variable(0.0, aggregation=variables.VariableAggregation.SUM)

    @def_function.function
    def iterator_fn(dist_dataset):
      # [FUNCTION DOC]
      # Nested compiled function that takes a distributed dataset and, for each element, runs the
      # add function on the strategy. Returns nothing.

      def assign_add_fn(data):
        # [FUNCTION DOC]
        # Nested replica function taking one batch (a mapping with a key for squared values) and adding
        # the sum of its values to the enclosing test's variable. Returns nothing.
        v.assign_add(math_ops.reduce_sum(data["y"]))

      for data in dist_dataset:
        strategy.run(assign_add_fn, args=(data,))

    iterator_fn(dist_dataset)
    self.assertEqual(v.numpy(), 285.0)


class DistributedIteratorTensorTypeTest(DistributedIteratorTestBase,
                                        parameterized.TestCase):
  # [CLASS DOC]
  # Tests for distributed datasets whose elements are not plain dense tensors: ragged and sparse
  # tensors, including rebatching across replicas, summation of all batches, and the
  # optional get-next behavior. Derives from the shared base class.
  """Tests for DistributedDataset with non-dense tensors."""

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
          ],
          input_type=["dataset", "input_fn"],
          drop_remainder=[False, True],
          defun_type=["lambda", "tf_function"],
      ))
  def testRaggedSparse(self, distribution, input_type, drop_remainder,
                       defun_type):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself unconditionally as its first statement (references to bug
    # reports), so everything after that is dead code and nothing is verified.
    # Intended scenario: a dataset of dense, ragged and sparse versions of the same data (twenty rows
    # with lengths cycling through zero to three), batched by a global size of eight, with or without
    # dropping the remainder, via dataset or input function, run directly or as a compiled function.
    # It would check the per-replica dense batches, check the ragged and sparse parts match the dense
    # ones, then sum all batches with a while loop and a for loop and compare the sums
    # (200 when the remainder is dropped, otherwise 310; for loops in compiled functions always use
    # optional get-next so can also give 310 with input functions).
    """Test with `RaggedTensor`s and `SparseTensor`s."""
    self.skipTest("b/213596871, b/214574707")

    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    defun = {
        "lambda": lambda f: f,
        "tf_function": def_function.function
    }[defun_type]
    distribution.extended.experimental_enable_get_next_as_optional = True
    global_batch_size = 8

    def dataset_fn(ctx=None):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an optional input context (default creates one). Builds
      # a dataset of dense, ragged and sparse versions of row data whose row lengths cycle from zero to
      # three, batches it by the per-replica share of the global batch size honoring the remainder
      # option, and shards it by the context's pipeline count and id. Returns the dataset.
      ctx = ctx or distribute_lib.InputContext()
      batch_size = ctx.get_per_replica_batch_size(global_batch_size)
      # Use 20 which isn't divisible by 8 to test partial batch behavior.
      row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
      ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
          np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
      dataset = dataset_ops.DatasetV2.from_tensor_slices({
          "dense": ragged_tensor.to_tensor(),
          "ragged": ragged_tensor,
          "sparse": ragged_tensor.to_sparse(),
      })
      dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
      return dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)

    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)
    dataset = self._wrap_dataset(input_type, dataset_or_input_fn,
                                 distribution.extended._input_workers,
                                 distribution.num_replicas_in_sync,
                                 distribution)
    # Assert that the tensors are rebatched and sparsity is preserved.
    per_replica_batch = defun(lambda x: next(iter(x)))(dataset)
    self.assertAllEqual(
        distribute_utils.select_replica(0, per_replica_batch["dense"]),
        [[0., 0., 0.], [1., 0., 0.], [2., 2., 0.], [3., 3., 3.]])
    self.assertAllEqual(
        distribute_utils.select_replica(1, per_replica_batch["dense"]),
        [[0., 0., 0.], [5., 0., 0.], [6., 6., 0.], [7., 7., 7.]])
    # Transitively check the ragged and sparse tensors by densification.
    for i in range(2):
      self.assertLen(
          distribute_utils.select_replica(i,
                                          per_replica_batch["ragged"]).values,
          6)
      self.assertAllEqual(
          distribute_utils.select_replica(
              i, per_replica_batch["ragged"]).to_tensor(),
          distribute_utils.select_replica(i, per_replica_batch["dense"]))
      self.assertLen(
          distribute_utils.select_replica(i,
                                          per_replica_batch["sparse"]).indices,
          6)
      self.assertAllEqual(
          sparse_ops.sparse_tensor_to_dense(
              distribute_utils.select_replica(i, per_replica_batch["sparse"])),
          distribute_utils.select_replica(i, per_replica_batch["dense"]))
    # Iterate through all the batches and sum them up.
    def sum_batch(per_replica_features):
      # [FUNCTION DOC]
      # HELPER: sums values of a per-replica feature mapping. Takes a mapping from feature names to
      # per-replica values. For each feature it runs a reduction on each replica (summing the values of
      # sparse tensors, otherwise all elements) and then reduces across replicas by addition. Returns a
      # mapping from names to totals.
      """Sums the `PerReplica` values in the `per_replica_features` map."""

      def map_fn(per_replica_values):
        # [FUNCTION DOC]
        # HELPER: sums one per-replica value on each replica (using the values part for sparse
        # tensors) and reduces the per-replica sums across replicas with addition. Returns the total.
        per_replica_sums = distribution.run(
            (lambda x: math_ops.reduce_sum(x.values)) if all(
                map(sparse_tensor.is_sparse, per_replica_values.values)) else
            math_ops.reduce_sum, (per_replica_values,))
        return distribution.reduce(
            reduce_util.ReduceOp.SUM, per_replica_sums, axis=None)

      return nest.map_structure(map_fn, per_replica_features)

    def _reduce(state, batch):
      # [FUNCTION DOC]
      # HELPER: accumulation step. Takes a state mapping from names to running totals and a batch,
      # computes the batch sums and returns a new mapping with the totals increased.
      sums = sum_batch(batch)
      return {name: value + sums[name] for name, value in state.items()}

    def sum_for_loop(dataset):
      # [FUNCTION DOC]
      # HELPER: sums all batches by looping over the dataset. Starts from zero totals for dense, ragged
      # and sparse features and accumulates each batch. Returns the mapping of totals.
      sums = {"dense": 0., "ragged": 0., "sparse": 0.}
      for batch in dataset:
        sums = _reduce(sums, batch)
      return sums

    def sum_while_loop(iterator, reduce_fn):
      # [FUNCTION DOC]
      # HELPER: sums batches by repeatedly calling a reduce function with the running totals and the
      # iterator until the iterator is exhausted (stop-iteration or out-of-range error). Takes the
      # iterator and the reduce function; returns the mapping of totals.
      sums = {"dense": 0., "ragged": 0., "sparse": 0.}
      while True:
        try:
          sums = reduce_fn(sums, iterator)
        except (StopIteration, errors.OutOfRangeError):
          return sums

    while_sums = sum_while_loop(
        iter(dataset),
        defun(lambda state, iterator: _reduce(state, next(iterator))))
    self.assertAllEqual(
        nest.flatten(while_sums),
        # When there's no partial batch, the sum is smaller.
        [200. if drop_remainder else 310.] * 3)
    for_sums = defun(sum_for_loop)(dataset)
    # For loops always call get next as optional inside tf functions, so we
    # expect 310 here when using an input function (as there are 5 batches of
    # size 4 round robined over 2 replicas.
    expected_for_sum = 200.
    if (not drop_remainder or
        (defun_type == "tf_function" and input_type == "input_fn")):
      expected_for_sum = 310.
    self.assertAllEqual(nest.flatten(for_sums), [expected_for_sum] * 3)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu
          ],
          input_type=["dataset", "input_fn"],
          drop_remainder=[False, True],
          tensor_type=["sparse", "ragged"],
          enable_get_next_as_optional=[True, False]))
  def testRaggedSparseGetNextAsOptional(self, distribution, input_type,
                                        drop_remainder, tensor_type,
                                        enable_get_next_as_optional):
    # [FUNCTION DOC]
    # Scenario: a dataset of either sparse or ragged tensors (chosen by the tensor type parameter),
    # sharded and batched by the per-replica share of a global batch of eight, built through the
    # strategy from a dataset or an input function. Skipped unless TF2 behavior is enabled.
    # Expectation: the iterator's internal get-next-as-optional flag is on only when the remainder
    # is not dropped and the strategy option is enabled.
    """Test with `RaggedTensor`s and `SparseTensor`s."""
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    global_batch_size = 8

    def dataset_fn(ctx=None):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an optional input context. Builds a dataset of the
      # chosen ragged or sparse form from twenty rows, shards it by the context, then batches it by the
      # per-replica share of the global batch size honoring the remainder option.
      ctx = ctx or distribute_lib.InputContext()
      batch_size = ctx.get_per_replica_batch_size(global_batch_size)
      # Use 20 which isn't divisible by 8 to test partial batch behavior.
      row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
      ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
          np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
      dataset = dataset_ops.DatasetV2.from_tensor_slices({
          tensor_type: (ragged_tensor if tensor_type == "ragged" else
                        ragged_tensor.to_sparse()),
      })
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
      return dataset.batch(batch_size, drop_remainder=drop_remainder)

    if input_type == "dataset":
      ds = distribution.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
    else:
      ds = distribution.distribute_datasets_from_function(dataset_fn)
    iterator = iter(ds)

    self.assertEqual(iterator._enable_get_next_as_optional,
                     (not drop_remainder) and enable_get_next_as_optional)

  @combinations.generate(
      combinations.combine(
          tf_api_version=2,
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
          ],
          input_type=["dataset", "input_fn"],
          drop_remainder=[False, True],
      ))
  def testRaggedSparseGetNextAsOptionalInLoop(self, distribution, input_type,
                                              drop_remainder):
    # [FUNCTION DOC]
    # Scenario: dense, ragged and sparse data distributed from a dataset or input function and
    # summed with a loop of optional get-next calls inside a compiled function (TF2 only).
    # Expectation: the total for each of the three features is 200 when the remainder is dropped with
    # a dataset, and 310 otherwise (an input function always takes the optional path).
    """Test with `RaggedTensor`s and `SparseTensor`s."""
    global_batch_size = 8

    def dataset_fn(ctx=None):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an optional input context. Builds the dense, ragged and
      # sparse dataset of twenty rows, batches it by the per-replica share of the global batch size
      # honoring the remainder option, and shards it by the context's pipeline count and id.
      ctx = ctx or distribute_lib.InputContext()
      batch_size = ctx.get_per_replica_batch_size(global_batch_size)
      # Use 20 which isn't divisible by 8 to test partial batch behavior.
      row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
      ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
          np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
      dataset = dataset_ops.DatasetV2.from_tensor_slices({
          "dense": ragged_tensor.to_tensor(),
          "ragged": ragged_tensor,
          "sparse": ragged_tensor.to_sparse(),
      })
      dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
      return dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)

    if input_type == "dataset":
      ds = distribution.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
    else:
      ds = distribution.distribute_datasets_from_function(dataset_fn)

    # Iterate through all the batches and sum them up.
    def sum_batch(per_replica_features):
      # [FUNCTION DOC]
      # HELPER: sums values of a per-replica feature mapping. Applies the nested map function to each
      # feature and returns a mapping from names to cross-replica totals.
      """Sums the `PerReplica` values in the `per_replica_features` map."""

      def map_fn(per_replica_values):
        # [FUNCTION DOC]
        # HELPER: runs a per-replica sum of one feature (through the nested sum helper) and reduces the
        # per-replica sums across replicas with addition. Returns the total.

        def _sum(value):
          # [FUNCTION DOC]
          # HELPER: replica function returning the sum of the values of a sparse tensor, or the sum of all
          # elements of a dense or ragged value.
          if sparse_tensor.is_sparse(value):
            return math_ops.reduce_sum(value.values)
          else:
            return math_ops.reduce_sum(value)

        per_replica_sums = distribution.run(_sum, args=(per_replica_values,))
        return distribution.reduce(
            reduce_util.ReduceOp.SUM, per_replica_sums, axis=None)

      return nest.map_structure(map_fn, per_replica_features)

    def _reduce(state, batch):
      # [FUNCTION DOC]
      # HELPER: accumulation step. Takes a state mapping from names to totals and a batch, and returns
      # a new mapping with each total increased by the batch's sum.
      sums = sum_batch(batch)
      return {name: value + sums[name] for name, value in state.items()}

    def sum_while_loop(ds):
      # [FUNCTION DOC]
      # HELPER: takes a distributed dataset, creates an iterator, and repeatedly fetches optionals
      # while one has a value, accumulating sums of dense, ragged and sparse features. Returns the
      # mapping of totals. Used inside a compiled function.
      iterator = iter(ds)
      sums = {"dense": 0., "ragged": 0., "sparse": 0.}
      try_next = constant_op.constant(True)

      while try_next:
        opt_iterate = iterator.get_next_as_optional()
        if opt_iterate.has_value():
          sums = _reduce(sums, opt_iterate.get_value())
        else:
          try_next = False
      return sums

    sums = def_function.function(sum_while_loop)(ds)
    # For loops always call get next as optional inside tf functions, so we
    # expect 310 here when using an input function (as there are 5 batches of
    # size 4 round robined over 2 replicas.
    expected_for_sum = 200.
    if not drop_remainder or input_type == "input_fn":
      expected_for_sum = 310.
    self.assertAllEqual(nest.flatten(sums), [expected_for_sum] * 3)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ]))
  def testMWMSPartialBatch(self, input_type, api_type, iteration_type,
                           distribution):
    # [FUNCTION DOC]
    # Scenario: two workers with one replica each, each seeing a twelve-element dataset batched by
    # eight (global batch of eight in total), with auto sharding turned off.
    # Expectation: the global batch is rebatched statically into per-worker batches of four, giving
    # three steps of four consecutive numbers on each worker.
    # Test case: 2 workers, 1 replica each.
    # This test simulates the sharded behavior when we have two files each with
    # 12 elements and a global batch size of 8. When we consider the dataset in
    # aggregate (non-distributed), there are 24 elements divided into 3 batches
    # of size 8. Hence, the correct distributed behavior is for each replica to
    # see sub-batches of size 4, over three steps.
    def dataset_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an ignored input context. Returns a twelve-element range
      # batched by eight with auto sharding disabled through dataset options.
      del ctx
      dataset = dataset_ops.Dataset.range(12).batch(8)

      # Set the sharding behavior to OFF for simplicity of test setup; namely,
      # `dataset` defines the per-worker dataset and will not be further
      # sharded. Each worker will see a dataset that is
      # tf.data.Dataset.range(12).batch(8).rebatch(...).
      options = options_lib.Options()
      options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
      dataset = dataset.with_options(options)
      return dataset

    dataset = self._create_dataset_or_input_fn(input_type, dataset_fn)

    # Actual devices don't matter in this test as long as there is 1 local
    # replica.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]

    # Each test runs individually on each worker, so we compare the
    # values on each worker. Each worker should rebatch its dataset into
    # smaller batches of size 4.
    expected_values = [[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9, 10, 11]]]
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset,
        worker_device_pairs,
        expected_values,
        distribution,
        num_replicas_in_sync=distribution.num_replicas_in_sync,
        input_context=distribution.extended._make_input_context())

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ]))
  def testMWMSPartialBatchWithLegacyRebatch(self, input_type, api_type,
                                            iteration_type, distribution):
    # [FUNCTION DOC]
    # Scenario: like the partial batch test, but the dataset is built from two batch sizes (eight and
    # four) by mapping ranges without a batching dataset, so the global batch size cannot be inferred
    # and the legacy dynamic rebatching is used.
    # Expectation: four steps per worker: two batches of four, then the last partial batch split into
    # two steps of two elements each.
    # Test case: 2 workers, 1 replica each.
    # This test simulates the sharded behavior when we have two files each with
    # 12 elements and a global batch size of 8. When we consider the dataset in
    # aggregate (non-distributed), there are 24 elements divided into 3 batches
    # of size 8. Hence, the correct distributed behavior is for each replica to
    # see sub-batches of size 4, over three steps. However, when we create a
    # DistributedDataset and cannot statically infer the intended global batch
    # size (e.g. if the user does not use a batching dataset), each worker will
    # rebatch based on the dynamic batch size of the data encountered, even when
    # it encounters partial batches. The last per-worker partial batch (size 4)
    # ends up being split into two replicas, resulting in 4 steps in total, of
    # (global) batch sizes 8, 8, 4, 4.
    def dataset_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an ignored input context. Zips offsets with batch sizes,
      # maps each pair to a range of numbers, and disables auto sharding via dataset options. Returns
      # the dataset.
      del ctx
      # The following dataset is equivalent to
      # tf.data.Dataset.range(12).batch(8), but does not use a batching dataset.
      # This causes DistributedDataset to use LegacyRebatch instead.
      batch_sizes = dataset_ops.Dataset.from_tensor_slices([8, 4])
      offsets = dataset_ops.Dataset.from_tensor_slices([0, 8])
      dataset = dataset_ops.Dataset.zip((offsets, batch_sizes))

      def map_fn(offset, batch_size):
        # [FUNCTION DOC]
        # HELPER: takes an offset and a batch size and returns the range of integers from the offset up to
        # offset plus batch size.
        return math_ops.range(offset, offset + batch_size)

      dataset = dataset.map(map_fn)

      # Set the sharding behavior to OFF for simplicity of test setup; namely,
      # `dataset` defines the per-worker dataset and will not be further
      # sharded. Each worker will see a dataset that is equivalent to
      # tf.data.Dataset.range(12).batch(8).rebatch(...).
      options = options_lib.Options()
      options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
      dataset = dataset.with_options(options)
      return dataset

    dataset = self._create_dataset_or_input_fn(input_type, dataset_fn)

    # Actual devices don't matter in this test as long as the number of global
    # replicas is 2.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]

    # Each test runs individually on each worker, so we compare the
    # values on each worker. Each worker should rebatch its dataset into
    # smaller batches of size 4.
    expected_values = [[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9]], [[10, 11]]]
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset,
        worker_device_pairs,
        expected_values,
        distribution,
        num_replicas_in_sync=distribution.num_replicas_in_sync,
        input_context=distribution.extended._make_input_context())

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ],
          auto_shard_policy=[AutoShardPolicy.AUTO, AutoShardPolicy.DATA]))
  def testMWMSWithDataSharding(self, input_type, api_type, iteration_type,
                               distribution, auto_shard_policy):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected values are only defined for worker index zero or one; any other index
    # leaves them undefined and the test would fail with an unbound-name error.
    # Scenario: two workers with one replica each; an eight-element range batched by three, with auto
    # shard policy automatic or by data. Verifies that sharding by data is paired with legacy
    # rebatching so batch sizes across workers add up to three.
    # Expectation: worker zero sees the batches of elements zero-one, three-four, and six; worker one
    # sees two, five, and seven.
    # Test case: 2 workers, 1 replica each.
    # This test simulates the sharded behavior the dataset is sharded by data
    # and the batch size is indivisible by the number of replicas. This checks
    # that the elements are as expected and the batch size across all workers
    # adds up to 3. This test will only pass if the autoshard rewrite rewrites
    # RebatchDatasetV2 to legacy RebatchDataset when sharding by data.
    def dataset_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an ignored input context. Returns an eight-element range
      # batched by three with the auto shard policy set to the enclosing test's parameter. (A comment
      # inside it still mentions turning sharding off, which is not what is done.)
      del ctx
      dataset = dataset_ops.Dataset.range(8).batch(3)

      # Set the sharding behavior to OFF for simplicity of test setup; namely,
      # `dataset` defines the per-worker dataset and will not be further
      # sharded. Each worker will see a dataset that is
      # tf.data.Dataset.range(12).batch(8).rebatch(...).
      options = options_lib.Options()
      options.experimental_distribute.auto_shard_policy = auto_shard_policy
      dataset = dataset.with_options(options)
      return dataset

    dataset = self._create_dataset_or_input_fn(input_type, dataset_fn)

    # Actual devices don't matter in this test as long as there is 1 local
    # replica.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]

    # Each test runs individually on each worker, so we compare the
    # values on each worker. We expect each worker to see different shards of
    # data.
    cr = distribution.cluster_resolver
    worker_id = multi_worker_util.id_in_cluster(cr.cluster_spec(), cr.task_type,
                                                cr.task_id)

    if worker_id == 0:
      expected_values = [[[0, 1]], [[3, 4]], [[6]]]
    elif worker_id == 1:
      expected_values = [[[2]], [[5]], [[7]]]

    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset,
        worker_device_pairs,
        expected_values,
        distribution,
        num_replicas_in_sync=distribution.num_replicas_in_sync,
        input_context=distribution.extended._make_input_context())


@framework_test_util.with_eager_op_as_function
class DistributedIteratorPerDeviceTest(DistributedIteratorTestBase,
                                       parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the input options of distributed datasets created from a function: placing the dataset
  # on device, fetching to device, the replication mode (per worker or per replica) and the per
  # replica prefetch buffer size. Decorated to run eager ops as functions. Uses two-device mirrored
  # strategies.
  """Tests for PER_WORKER and PER_REPLICA's InputOptions variants."""

  @combinations.generate(
      combinations.combine(
          input_options=[
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=True,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_WORKER),
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=True,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_REPLICA),
          ],
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.mirrored_strategy_with_two_cpus,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ]))
  def testDevicePlacementForPerWorkerValuesWithPrefetch(self, distribution,
                                                        input_options):
    # [FUNCTION DOC]
    # Scenario: distributing a four-element dataset from a function with input options that do not
    # place the dataset on device but fetch to device, in both per-worker and per-replica modes.
    # Expectation: for each element the two per-replica values live, and are backed, on the
    # strategy's first and second worker devices respectively (checked with plain assertions).

    def dataset_fn(input_context):  # pylint: disable=[unused-argument]
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an input context (unused) and returning a dataset of the
      # numbers one to four.
      return dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4])

    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, input_options)

    for x in ds:
      assert x.values[0].device == distribution.extended.worker_devices[0]
      assert x.values[0].backing_device == distribution.extended.worker_devices[
          0]
      assert x.values[1].device == distribution.extended.worker_devices[1]
      assert x.values[1].backing_device == distribution.extended.worker_devices[
          1]

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.mirrored_strategy_with_two_cpus,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          input_options=[
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=False,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_WORKER)
          ],
          mode=["eager"],
      ))
  def testDevicePlacementForPerWorkerValuesWithoutPrefetch(
      self, distribution, input_options):
    # [FUNCTION DOC]
    # Scenario: per-worker replication with no placement on device and no fetch to device.
    # Each element is passed through a replica function that returns it unchanged.
    # Expectation: both per-replica values, and their backing devices, are on the local host CPU
    # device (checked with plain assertions).

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an input context and returning a four-element dataset
      # filled with the input pipeline id.
      return dataset_ops.Dataset.from_tensor_slices(
          np.full(4, input_context.input_pipeline_id))

    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, input_options)

    for x in ds:
      x = distribution.run(lambda inputs: inputs, args=(x,))
      assert x.values[
          0].device == "/job:localhost/replica:0/task:0/device:CPU:0"
      assert x.values[
          0].backing_device == "/job:localhost/replica:0/task:0/device:CPU:0"
      assert x.values[
          1].device == "/job:localhost/replica:0/task:0/device:CPU:0"
      assert x.values[
          1].backing_device == "/job:localhost/replica:0/task:0/device:CPU:0"

  @combinations.generate(
      combinations.combine(
          input_options=[
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=True,
                  experimental_fetch_to_device=False,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_WORKER),
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=True,
                  experimental_fetch_to_device=True,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_REPLICA)
          ],
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.mirrored_strategy_with_two_cpus,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ]))
  def testDevicePlacementForInvalidCombinations(self, distribution,
                                                input_options):
    # [FUNCTION DOC]
    # Scenario: input options that are invalid: placing the dataset on device together with per-worker
    # replication and no fetch to device, or together with per-replica replication and fetching to
    # device. Expectation: distributing datasets from the function raises a value error.

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an input context and returning a four-element dataset
      # filled with the input pipeline id.
      return dataset_ops.Dataset.from_tensor_slices(
          np.full(4, input_context.input_pipeline_id))

    with self.assertRaises(ValueError):
      distribution.experimental_distribute_datasets_from_function(
          dataset_fn, input_options)

  @combinations.generate(
      combinations.combine(
          input_options=[
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=False,
                  experimental_per_replica_buffer_size=2),
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=True,
                  experimental_per_replica_buffer_size=2),
          ],
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.mirrored_strategy_with_two_cpus,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ]))
  def testPrefetchBufferSizeInputOptions(self, distribution, input_options):
    # [FUNCTION DOC]
    # Scenario: input options with a per-replica buffer size of two, with and without fetching to
    # device. Distributes a function producing two rows of five numbers scaled by one plus the
    # pipeline id, then reads the first element.
    # Expectation: replica zero gets the first row (one to five) and replica one the second row
    # (six to ten), checked with plain assertions.

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an input context; returns a dataset of two rows of five
      # numbers (one to ten) multiplied by one plus the input pipeline id.
      return dataset_ops.Dataset.from_tensor_slices(
          np.arange(1, 11).reshape(
              (2, 5)) * (input_context.input_pipeline_id + 1))

    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, input_options)

    # validating the values
    x = next(iter(ds))
    assert np.array_equal(x.values[0].numpy(), np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(x.values[1].numpy(), np.array([6, 7, 8, 9, 10]))

  @combinations.generate(
      combinations.combine(
          input_options=[
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=False,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_WORKER),
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=True,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_WORKER),
          ],
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.mirrored_strategy_with_two_cpus,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ]))
  def testOutputValuesForPerWorkerInputOptions(self, distribution,
                                               input_options):
    # [FUNCTION DOC]
    # Scenario: per-worker replication, no placement on device, with and without fetching to device.
    # Distributes a function producing two rows of five numbers and reads the first element.
    # Expectation: replica zero holds one to five and replica one holds six to ten.

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an input context; returns a dataset of two rows of five
      # numbers (one to ten) multiplied by one plus the input pipeline id.
      return dataset_ops.Dataset.from_tensor_slices(
          np.arange(1, 11).reshape(
              (2, 5)) * (input_context.input_pipeline_id + 1))

    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, input_options)

    # validating the values
    x = next(iter(ds))
    assert np.array_equal(x.values[0].numpy(), np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(x.values[1].numpy(), np.array([6, 7, 8, 9, 10]))

  @combinations.generate(
      combinations.combine(
          input_options=[
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=True,
                  experimental_fetch_to_device=False,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_REPLICA),
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=False,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_REPLICA),
              distribute_lib.InputOptions(
                  experimental_place_dataset_on_device=False,
                  experimental_fetch_to_device=True,
                  experimental_replication_mode=distribute_lib
                  .InputReplicationMode.PER_REPLICA),
          ],
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.mirrored_strategy_with_two_cpus,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ]))
  def testOutputValuesForPerReplicaInputOptions(self, distribution,
                                                input_options):
    # [FUNCTION DOC]
    # Scenario: per-replica replication with different placement and fetch settings. Distributes a
    # function producing the numbers one to nine scaled by one plus the pipeline id, and iterates.
    # Expectation: for each step replica zero holds the i-th number and replica one holds twice that;
    # the loop must run over all nine numbers (checked with plain assertions).

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an input context; returns a dataset of the numbers one to
      # nine multiplied by one plus the input pipeline id.
      return dataset_ops.Dataset.from_tensor_slices(
          np.arange(1, 10) * (input_context.input_pipeline_id + 1))

    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, input_options)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    for i, x in enumerate(ds):
      # validating the values
      assert x.values[0].numpy() == expected[i]
      assert x.values[1].numpy() == expected[i] * 2
      loop_num = i
    assert loop_num == len(expected) - 1


class DistributedIteratorTfDataServiceTest(DistributedIteratorTestBase,
                                           parameterized.TestCase):
  # [CLASS DOC]
  # Tests for distributed datasets and iterators that read from the tf.data service. The fixture
  # starts a dispatcher and three workers in the main process and records the dispatcher address in
  # the shared test environment, so the test body (which also runs in worker processes) can reach
  # the service.
  """Tests for distributed iterators which read from tf.data service."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture. Calls the base setup, records the worker count (three), and, only in the main
    # process of the test combination framework, starts a dispatcher server and three worker servers
    # (heartbeat interval of 100 milliseconds, dispatcher timeout of 1000 milliseconds) and stores the
    # dispatcher address in the shared environment. Side effects: starts servers kept on the instance.
    super(DistributedIteratorTfDataServiceTest, self).setUp()
    self.num_workers = 3
    if combinations.in_main_process():
      self.dispatcher = server_lib.DispatchServer()
      self.workers = []
      for _ in range(self.num_workers):
        self.workers.append(
            server_lib.WorkerServer(
                server_lib.WorkerConfig(
                    dispatcher_address=self.dispatcher.target.split("://")[1],
                    heartbeat_interval_ms=100,
                    dispatcher_timeout_ms=1000)))
      combinations.env().tf_data_service_dispatcher = self.dispatcher.target

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ]))
  def testTfDataService(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a range dataset from one to forty-nine is distributed through the tf.data service
    # with sharding off (every service worker produces all elements), then made a distributed dataset
    # and iterated.
    # Collects all nonzero local results (zero elements are padding), gathers them across replicas.
    # Expectation: results are non-empty; gathered values contain each of the numbers once per service
    # worker; and the initialization-time histogram for this strategy class has a positive count.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]
    input_workers = input_lib.InputWorkers(worker_device_pairs)

    dataset = dataset_ops.Dataset.range(1, 50)
    dataset = dataset.apply(
        data_service_ops._distribute(
            processing_mode=data_service_ops.ShardingPolicy.OFF,
            service=combinations.env().tf_data_service_dispatcher,
            job_name="foo"))

    dist_dataset = input_util.get_distributed_dataset(dataset, input_workers,
                                                      distribution)
    iterator = iter(dist_dataset)
    results = []
    for element in iterator:
      local_results = distribution.experimental_local_results(element)
      for result in local_results:
        # input_lib.distributed_dataset may add extra '0' elements to pad
        # per-replica results.
        if result.numpy() != 0:
          results.append(result.numpy())
    self.assertNotEmpty(results)
    gathered = distribution.gather(constant_op.constant(results), axis=0)
    self.assertCountEqual(self.num_workers * list(range(1, 50)), gathered)

    histogram_proto = (
        input_lib._distributed_dataset_initialization_time_milliseconds
        .get_cell(distribution.__class__.__name__, "1").value())
    self.assertGreater(histogram_proto.num, 0.0)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ]))
  def testDistributeDatasetFromFunction(self, distribution):
    # [FUNCTION DOC]
    # Scenario: registers a range dataset with the service under a fixed dataset id (so repeated
    # registration in chief and worker processes refers to the same dataset), and builds a
    # distributed dataset from a function that reads that registered dataset with sharding off.
    # Creates input contexts per worker, iterates, drops zero padding elements, gathers.
    # Expectation: non-empty results, each number appearing once per service worker, and the
    # from-function initialization histogram for this strategy class has a positive count.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]
    input_workers = input_lib.InputWorkers(worker_device_pairs)
    input_contexts = []
    num_workers = input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(distribute_lib.InputContext(
          num_input_pipelines=num_workers,
          input_pipeline_id=i,
          num_replicas_in_sync=num_workers))

    dataset = dataset_ops.Dataset.range(1, 50)
    dataset_id = "dataset_id"
    # The body of this test is run on both the chief and the workers, so
    # `register_dataset` will be called multiple times. We use a pre-defined
    # `dataset_id` so that the tf.data service will understand that the
    # registered datasets are the same.
    data_service_ops.register_dataset(
        service=combinations.env().tf_data_service_dispatcher,
        dataset=dataset, dataset_id=dataset_id)

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an ignored input context; returns a dataset read from the
      # tf.data service using the registered dataset id, the original element spec, sharding off and a
      # shared job name.
      del input_context
      return data_service_ops.from_dataset_id(
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=combinations.env().tf_data_service_dispatcher,
          dataset_id=dataset_id,
          element_spec=dataset.element_spec,
          job_name="shared_job")

    dist_dataset = input_util.get_distributed_datasets_from_function(
        dataset_fn, input_workers, input_contexts, distribution)

    iterator = iter(dist_dataset)
    results = []
    for element in iterator:
      local_results = distribution.experimental_local_results(element)
      for result in local_results:
        # input_lib.distributed_dataset may add extra '0' elements to pad
        # per-replica results.
        if result.numpy() != 0:
          results.append(result.numpy())
    self.assertNotEmpty(results)
    gathered = distribution.gather(constant_op.constant(results), axis=0)
    self.assertCountEqual(self.num_workers * list(range(1, 50)), gathered)
    histogram_proto = (
        input_lib
        ._distributed_dataset_from_function_initialization_time_milliseconds
        .get_cell(distribution.__class__.__name__, "1").value())
    self.assertGreater(histogram_proto.num, 0.0)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
          ]))
  def testDistributeDatasetFromFunctionNested(self, distribution):
    # [FUNCTION DOC]
    # Scenario: elements are nested extension types (an outer type containing an inner type holding a
    # tensor) produced by a function-based distributed dataset of nine elements. Iterates and
    # collects all local results.
    # Expectation: the results equal the expected nested value repeated once per worker per element.
    worker_device_pairs = [("/device:CPU:0", ["/device:CPU:0"])]
    input_workers = input_lib.InputWorkers(worker_device_pairs)
    input_contexts = []
    num_workers = input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(
          distribute_lib.InputContext(
              num_input_pipelines=num_workers,
              input_pipeline_id=i,
              num_replicas_in_sync=num_workers))

    class InnerType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Extension type with one field holding a tensor. Used as the inner component of the nested
      # element in the test.
      tensor: tensor.Tensor

    class OuterType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Extension type with one field holding an inner-type value. Used as the element type of the
      # dataset in the nested distribution test.
      inner: InnerType

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: nested dataset factory taking an ignored input context; returns a dataset of nine
      # elements mapped to the outer type by the data function.
      del input_context

      def data_fn(batch_id) -> OuterType:
        # [FUNCTION DOC]
        # HELPER: map function taking and ignoring a batch id; returns an outer-type value whose inner
        # value holds a fixed two by two float tensor.
        del batch_id

        return OuterType(
            inner=InnerType(tensor=constant_op.constant([[0., 1.], [2., 3.]])))

      return dataset_ops.Dataset.range(1, 10).map(data_fn)

    dist_dataset = input_util.get_distributed_datasets_from_function(
        dataset_fn, input_workers, input_contexts, distribution)

    iterator = iter(dist_dataset)
    results = []
    for element in iterator:
      local_results = distribution.experimental_local_results(element)
      for result in local_results:
        results.append(result)

    expect_component = OuterType(
        inner=InnerType(tensor=constant_op.constant([[0., 1.], [2., 3.]])))
    self.assertCountEqual(
        num_workers * [expect_component for _ in range(1, 10)], results)

if __name__ == "__main__":
  test_util.main()
