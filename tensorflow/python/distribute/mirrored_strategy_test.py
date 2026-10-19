# [MODULE DOC]
# Unit tests for MirroredStrategy (synchronous data-parallel training across the devices of one
# machine). Covers reductions, input pipelines and dataset replication, variable creation and
# placement, replica contexts, merge calls, cross-device ops selection (collective versus NCCL
# versus reduction to one device), gradients, multi-worker configuration parsing, tf.function and
# autograph interaction, and error handling. Many classes are parameterized over several strategy
# and mode combinations (graph and eager), some requiring GPUs. Also defines small module-level
# helpers (replica id functions, a custom variable creator, etc.) shared by the tests. Defines a
# module-level flag telling whether the test was launched as a GPU test. Running the file as a
# script starts the distribute test runner.
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
"""Tests for MirroredStrategy."""

import json
import sys

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util as util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.training import server_lib
from tensorflow.python.util import traceback_utils


GPU_TEST = "test_gpu" in sys.argv[0]


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.mirrored_strategy_with_two_gpus,
            strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
        ],
        mode=["graph", "eager"]))
class MirroredTwoDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Tests for MirroredStrategy on two devices, run for three strategy variants (GPU and CPU, two
  # GPUs, two GPUs without merge call) in graph and eager modes. Mixes shared base-class checks
  # (minimize loss, replica id, run, all-reduce, summaries) with specific tests for reductions,
  # input iteration and device prefetching.

  def testMinimizeLoss(self, distribution):
    # [FUNCTION DOC]
    # Runs the shared minimize-loss scenario: eager variant in eager mode, graph variant otherwise.
    # Delegates all assertions to the base-class helpers.
    if context.executing_eagerly():
      self._test_minimize_loss_eager(distribution)
    else:
      self._test_minimize_loss_graph(distribution)

  def testReplicaId(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check that each replica sees its own replica id.
    self._test_replica_id(distribution)

  def testNumReplicasInSync(self, distribution):
    # [FUNCTION DOC]
    # Asserts the strategy reports exactly two replicas in sync.
    self.assertEqual(2, distribution.num_replicas_in_sync)

  def testCallAndMergeExceptions(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of error handling in per-replica calls and merge
    # calls.
    self._test_call_and_merge_exceptions(distribution)

  def testRunRegroupError(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a per-replica function returns lists whose length depends on the replica id, so the
    # results cannot be regrouped across replicas. Asserts that calling it for each replica raises
    # an assertion error. Skipped when the strategy does not use merge calls (collective
    # all-reduce lacks int32 support on GPU).
    if not distribution.extended._use_merge_call():
      self.skipTest("Collective all-reduce does not support int32 on GPU.")
    def run_fn():
      # [FUNCTION DOC]
      # Per-replica function: returns a list of integers whose length equals the replica id, so
      # lengths differ across replicas and regrouping the results fails.
      replica_id = int(self.evaluate(_replica_id()))
      # Generates a list with different lengths on different devices.
      # Will fail in _regroup() (if more than one device).
      return list(range(replica_id))

    with distribution.scope(), self.assertRaises(AssertionError):
      distribution.extended.call_for_each_replica(run_fn)

  def testReduceToCpu(self, distribution):
    # [FUNCTION DOC]
    # Scenario: call the replica-id function on each replica and sum-reduce the per-replica results
    # with no axis. Asserts the result equals the sum of all replica ids (0 up to replicas minus
    # one). Skipped when the strategy does not use merge calls.
    if not distribution.extended._use_merge_call():
      self.skipTest("Collective all-reduce does not support int32 on GPU.")

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(_replica_id)
      reduced = distribution.reduce(reduce_util.ReduceOp.SUM, result, axis=None)
      expected = sum(range(distribution.num_replicas_in_sync))
      self.assertEqual(expected, self.evaluate(reduced))

  def testReduceToCpuNested(self, distribution):
    # [FUNCTION DOC]
    # Scenario: run a function on every replica that returns a pair (input plus one, input minus
    # one) for the input 3, then sum-reduce the nested result. Asserts the reduced pair equals
    # (4 and 2, each multiplied by the number of replicas). Skipped when merge calls are unused.
    if not distribution.extended._use_merge_call():
      self.skipTest("Collective all-reduce does not support int32 on GPU.")

    with distribution.scope():
      def replica_fn(input_tensor):
        # [FUNCTION DOC]
        # Per-replica function: returns a pair holding the input plus one and the input minus one.
        return input_tensor + constant_op.constant(
            1.0), input_tensor - constant_op.constant(1.0)

      input_tensor = constant_op.constant(3.0)
      run_result = distribution.run(replica_fn, args=(input_tensor,))
      reduced_result = distribution.reduce("SUM", run_result, axis=None)
      expected_result = (4 * distribution.num_replicas_in_sync,
                         2 * distribution.num_replicas_in_sync)

      self.assertEqual(expected_result, self.evaluate(reduced_result))

  def reduce_axis_helper(self, distribution, replica_squared_fn):
    # [FUNCTION DOC]
    # HELPER: shared checker for the axis-reduction tests.
    # Calls the given per-replica function on every replica (it yields a list whose length depends
    # on the replica id), then reduces along axis zero twice. With sum, asserts the result is near
    # the sum over replicas of x times x plus one. With mean, asserts the result is that sum
    # divided by the total element count. Returns nothing.
    with distribution.scope():
      num_replicas = distribution.num_replicas_in_sync
      result = distribution.extended.call_for_each_replica(replica_squared_fn)
      # sum
      reduced = distribution.reduce(reduce_util.ReduceOp.SUM, result, axis=0)
      expected = sum(x * (x + 1) for x in range(num_replicas))
      self.assertNear(expected, self.evaluate(reduced), 0.00001)

      # mean
      reduced = distribution.reduce(reduce_util.ReduceOp.MEAN, result, axis=0)
      expected /= sum(x + 1 for x in range(num_replicas))
      self.assertNear(expected, self.evaluate(reduced), 0.00001)

  def testReduceAxisToCpu(self, distribution):
    # [FUNCTION DOC]
    # Scenario: for float32 and int32, each replica produces a list of its replica id repeated
    # (id plus one) times; the axis-reduction helper checks sum and mean reductions. Skipped when
    # merge calls are unused.
    if not distribution.extended._use_merge_call():
      self.skipTest("Collective all-reduce does not support int32 on GPU.")
    for dtype in (dtypes.float32, dtypes.int32):
      def replica_squared_fn(dtype=dtype):
        # [FUNCTION DOC]
        # Per-replica function: returns a vector of the replica id repeated (id plus one) times,
        # cast to the given data type (default is the loop variable's type).
        # Lists with different lengths on different replicas.
        replica_id = _replica_id_as_int()
        return array_ops.identity(
            math_ops.cast([replica_id] * (replica_id + 1), dtype))

      self.reduce_axis_helper(distribution, replica_squared_fn)

  def set_v2_tensorshape(self, v2):
    # [FUNCTION DOC]
    # HELPER: toggles the global tensor-shape behavior. Takes a boolean; when true enables the
    # version 2 tensor shape semantics, otherwise disables them. Mutates global state; returns
    # nothing.
    if v2:
      tensor_shape.enable_v2_tensorshape()
    else:
      tensor_shape.disable_v2_tensorshape()

  def testReduceAxisToCpuUnknownShape(self, distribution):
    # [FUNCTION DOC]
    # Scenario: like the axis-reduction test, but the per-replica tensors have their shape erased
    # (unknown size, or unknown rank) via a placeholder with default, under both v1 and v2 tensor
    # shape semantics, for float32 and int32. The original global shape setting is saved and
    # restored in a finally step. Skipped when merge calls are unused.
    if not distribution.extended._use_merge_call():
      self.skipTest("Collective all-reduce does not support int32 on GPU.")
    original_v2 = tensor_shape._TENSORSHAPE_V2_OVERRIDE  # pylint: disable=protected-access
    try:
      for v2 in (False, True):
        self.set_v2_tensorshape(v2)
        for dtype in (dtypes.float32, dtypes.int32):
          for shape in ((None,), None):  # Test both unknown size and rank.
            def replica_squared_fn(dtype=dtype, shape=shape):
              # [FUNCTION DOC]
              # Per-replica function: builds a vector of the replica id repeated (id plus one)
              # times in the given data type, then hides its static shape using the given shape
              # (unknown size or unknown rank) through a placeholder with default.
              # Lists with different lengths on different replicas.
              replica_id = _replica_id_as_int()
              tensor = math_ops.cast([replica_id] * (replica_id + 1), dtype)
              # Erase shape information
              return array_ops.placeholder_with_default(tensor, shape=shape)

            self.reduce_axis_helper(distribution, replica_squared_fn)
    finally:
      self.set_v2_tensorshape(original_v2)

  def testReplicateDataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: input function over a range of ten numbers, expecting two replicas, one input
    # pipeline, pipeline id zero. Asserts iterating the distributed dataset yields pairs of
    # consecutive numbers. Skipped in graph mode when TF2 behavior is enabled.
    if tf2.enabled() and not context.executing_eagerly():
      self.skipTest("Skipping test since we do not support graph mode in TF 2")

    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i, i+1] for i in range(0, 10, 2)]
    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=2,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterable(distribution, input_fn, expected_values)

  def testMakeInputFnIteratorWithDataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: build an input-function iterator from a dataset of ten numbers (two replicas, one
    # input pipeline). Asserts, via the base-class helper, that the iterator yields consecutive
    # pairs across the worker devices.
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=2,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    iterator = distribution.make_input_fn_iterator(input_fn)
    self._test_input_fn_iterator(iterator, distribution.extended.worker_devices,
                                 expected_values)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the input function returns a get-next callable (from a one-shot iterator over an
    # interleaved dataset) rather than a dataset. Asserts the iterator yields each number twice,
    # once per replica, ignoring order and without testing reinitialization.
    def fn():
      # [FUNCTION DOC]
      # Input function: interleaves two ranges of ten numbers, creates a one-shot iterator and
      # returns its get-next callable.
      dataset = dataset_ops.Dataset.range(2).interleave(
          (lambda _: dataset_ops.Dataset.range(10)), cycle_length=2)
      it = dataset_ops.make_one_shot_iterator(dataset)
      return it.get_next
    expected_values = [[i, i] for i in range(0, 10)]

    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=2,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    iterator = distribution.make_input_fn_iterator(input_fn)
    self._test_input_fn_iterator(iterator, distribution.extended.worker_devices,
                                 expected_values, test_reinitialize=False,
                                 ignore_order=True)

  def testNumpyDataset(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of datasets built from in-memory arrays.
    self._test_numpy_dataset(distribution)

  def testGlobalStepUpdate(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of global step updates under the strategy.
    self._test_global_step_update(distribution)

  def testRun(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of running a function on all replicas.
    self._test_run(distribution)

  def testAllReduceSum(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of all-reduce with sum.
    self._test_all_reduce_sum(distribution)

  def testAllReduceSumGradients(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of gradients through a sum all-reduce.
    self._test_all_reduce_sum_gradients(distribution)

  def testAllReduceSumGradientTape(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of sum all-reduce gradients using a gradient tape.
    self._test_all_reduce_sum_gradient_tape(distribution)

  def testAllReduceMean(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of all-reduce with mean.
    self._test_all_reduce_mean(distribution)

  def testAllReduceMeanGradients(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of gradients through a mean all-reduce.
    self._test_all_reduce_mean_gradients(distribution)

  def testAllReduceMeanGradientTape(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of mean all-reduce gradients using a gradient tape.
    self._test_all_reduce_mean_gradient_tape(distribution)

  def testSummaryForReplicaZeroOnly(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check that summaries are written only by replica zero.
    self._test_summary_for_replica_zero_only(distribution)

  def testTrainableVariables(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of trainable variable tracking under the strategy.
    self._test_trainable_variable(distribution)

  def test_prefetch_to_device_dataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: distribute a batched range dataset with the fetch-to-device option on. Takes the
    # first element (graph mode: from an initializable iterator, skipped for unsupported dataset
    # types). Asserts the device type of each per-replica tensor equals the device type of the
    # matching worker device.
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
    device_types = [
        tf_device.DeviceSpec.from_string(tensor.device).device_type for
        tensor in item.values]
    expected_device_types = [
        tf_device.DeviceSpec.from_string(device).device_type for
        device in distribution.extended.worker_devices]
    self.assertAllEqual(device_types, expected_device_types)

  def test_prefetch_to_host_dataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: like the prefetch-to-device test but with fetch-to-device turned off. Asserts that
    # every per-replica tensor of the first element lives on a CPU device.
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
    device_types = {
        tf_device.DeviceSpec.from_string(tensor.device).device_type for
        tensor in item.values}
    self.assertAllEqual(list(device_types), ["CPU"])


@combinations.generate(
    combinations.combine(
        mode=["eager", "graph"], required_gpus=[2]))
class MirroredCollectiveOpTest(strategy_test_lib.DistributionTestBase,
                               strategy_test_lib.TwoDeviceDistributionTestBase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Tests which cross-device reduction implementation MirroredStrategy picks (collective all-reduce
  # with ring or NCCL, or reduction to one device) for different device lists. Requires two GPUs;
  # runs in graph and eager modes.

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs the parent teardown then resets the global eager context so changes to logical devices
    # do not leak between tests.
    super(MirroredCollectiveOpTest, self).tearDown()
    context._reset_context()

  def testAllCpu(self):
    # [FUNCTION DOC]
    # Scenario: inside a compiled function, create a strategy over two CPU devices. In eager
    # (outside-function) mode asserts the collective ops use collective all-reduce with the ring
    # implementation; otherwise asserts reduction to one device is used.
    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # Compiled body of the test: builds the strategy and performs the assertions described above.
      strategy = mirrored_strategy.MirroredStrategy(["CPU:0", "CPU:1"])
      if ops.executing_eagerly_outside_functions():
        self.assertIsInstance(
            strategy.extended._collective_ops,
            cross_device_ops_lib.CollectiveAllReduce)
        self.assertEqual(
            strategy.extended._collective_ops._options.implementation,
            collective_util.CommunicationImplementation.RING)
      else:
        self.assertIsInstance(strategy.extended._collective_ops,
                              cross_device_ops_lib.ReductionToOneDevice)
    fn()

  def testMixedDevices(self):
    # [FUNCTION DOC]
    # Scenario: inside a compiled function, create a strategy over one CPU and one GPU. Asserts the
    # collective ops fall back to reduction to one device.
    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # Compiled body of the test: builds the mixed-device strategy and asserts the choice.
      strategy = mirrored_strategy.MirroredStrategy(["CPU:0", "GPU:0"])
      self.assertIsInstance(
          strategy.extended._collective_ops,
          cross_device_ops_lib.ReductionToOneDevice)
    fn()

  def testAllPhysicalGpu(self):
    # [FUNCTION DOC]
    # Scenario: inside a compiled function, create a strategy over two physical GPUs. Asserts the
    # collective ops are collective all-reduce using the NCCL implementation.
    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # Compiled body of the test: builds the two-GPU strategy and asserts the choice.
      strategy = mirrored_strategy.MirroredStrategy(["GPU:0", "GPU:1"])
      self.assertIsInstance(
          strategy.extended._collective_ops,
          cross_device_ops_lib.CollectiveAllReduce)
      self.assertEqual(
          strategy.extended._collective_ops._options.implementation,
          collective_util.CommunicationImplementation.NCCL)
    fn()

  def testVirtualGpu(self):
    # [FUNCTION DOC]
    # Scenario: reset the context, split the second physical GPU into two logical GPUs with a 1024
    # memory limit each, then inside a compiled function create a strategy over three GPUs. In
    # eager (outside-function) mode asserts collective all-reduce with ring implementation;
    # otherwise asserts the collective ops equal the reduction-to-one-device class itself.
    # Logical devices cannot be changed after context initialization.
    context._reset_context()

    physical_gpus = context.context().list_physical_devices(device_type="GPU")
    context.context().set_logical_device_configuration(physical_gpus[1], [
        context.LogicalDeviceConfiguration(memory_limit=1024),
        context.LogicalDeviceConfiguration(memory_limit=1024)
    ])
    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # Compiled body of the test: builds the three-GPU strategy and asserts the choice.
      strategy = mirrored_strategy.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2"])
      if ops.executing_eagerly_outside_functions():
        self.assertIsInstance(
            strategy.extended._collective_ops,
            cross_device_ops_lib.CollectiveAllReduce)
        self.assertEqual(
            strategy.extended._collective_ops._options.implementation,
            collective_util.CommunicationImplementation.RING)
      else:
        self.assertEqual(strategy.extended._collective_ops,
                         cross_device_ops_lib.ReductionToOneDevice)
    fn()


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"], required_gpus=[2], use_default=[True, False]))
class MirroredGetCrossDeviceOpTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests which cross-device ops object MirroredStrategy returns for a variable on two GPUs, with
  # and without a user-supplied NCCL all-reduce. Parameterized by whether the default is used.

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs the parent teardown then resets the global eager context.
    super().tearDown()
    context._reset_context()

  def testGpusCollectiveOp(self, use_default):
    # [FUNCTION DOC]
    # Scenario: create a two-GPU strategy (default cross-device ops when use_default is true,
    # otherwise an explicit NCCL all-reduce) and a variable in its scope. Inside a compiled
    # function asserts the chosen cross-device ops is collective all-reduce when using the default
    # or when XLA is enabled, and NCCL all-reduce otherwise.

    @def_function.function(jit_compile=util.is_xla_enabled())
    def fn(var, use_default):
      # [FUNCTION DOC]
      # Compiled body: asserts the type of the cross-device ops chosen for the given variable.

      if use_default or util.is_xla_enabled():
        self.assertIsInstance(
            strategy.extended._get_cross_device_ops(var),
            cross_device_ops_lib.CollectiveAllReduce)
      else:
        self.assertIsInstance(
            strategy.extended._get_cross_device_ops(var),
            cross_device_ops_lib.NcclAllReduce)

    strategy = mirrored_strategy.MirroredStrategy(
        ["GPU:0", "GPU:1"],
        cross_device_ops=None
        if use_default else cross_device_ops_lib.NcclAllReduce())
    with strategy.scope():
      var = variables.Variable(1.)

    fn(var, use_default)

  def testVirtualGpusCollectiveOp(self, use_default):
    # [FUNCTION DOC]
    # Scenario: reset the context and split the second physical GPU into two logical GPUs, create
    # a three-GPU strategy (default or explicit NCCL ops) with a variable, and inside a compiled
    # function assert the chosen cross-device ops: with the default or XLA, collective all-reduce
    # with ring implementation; otherwise NCCL all-reduce.
    # Logical devices cannot be changed after context initialization.
    context._reset_context()

    physical_gpus = context.context().list_physical_devices(device_type="GPU")
    context.context().set_logical_device_configuration(physical_gpus[1], [
        context.LogicalDeviceConfiguration(memory_limit=1024),
        context.LogicalDeviceConfiguration(memory_limit=1024)
    ])

    @def_function.function(jit_compile=util.is_xla_enabled())
    def fn(var, use_default):
      # [FUNCTION DOC]
      # Compiled body: asserts the type (and ring implementation) of the cross-device ops chosen
      # for the given variable.

      if use_default or util.is_xla_enabled():
        self.assertIsInstance(
            strategy.extended._get_cross_device_ops(var),
            cross_device_ops_lib.CollectiveAllReduce)
        self.assertEqual(
            strategy.extended._get_cross_device_ops(
                var)._options.implementation,
            collective_util.CommunicationImplementation.RING)
      else:
        self.assertIsInstance(
            strategy.extended._get_cross_device_ops(var),
            cross_device_ops_lib.NcclAllReduce)

    strategy = mirrored_strategy.MirroredStrategy(
        ["GPU:0", "GPU:1", "GPU:2"],
        cross_device_ops=None
        if use_default else cross_device_ops_lib.NcclAllReduce())

    with strategy.scope():
      var = variables.Variable(1.)

    fn(var, use_default)

  def testCpusCollectiveOp(self, use_default):
    # [FUNCTION DOC]
    # Scenario: a two-CPU strategy with a variable; the use_default parameter is ignored. Skipped
    # under XLA. Inside a function compiled with XLA, asserts reduction to one device is used when
    # eager execution is not active outside functions, and collective all-reduce otherwise.
    del use_default
    if util.is_xla_enabled():
      self.skipTest("Only expected to run under non-XLA context.")

    @def_function.function(jit_compile=True)
    def fn(var):
      # [FUNCTION DOC]
      # Compiled body: asserts the type of cross-device ops chosen for the given variable.

      if not ops.executing_eagerly_outside_functions():
        self.assertIsInstance(
            strategy.extended._get_cross_device_ops(var),
            cross_device_ops_lib.ReductionToOneDevice)
      else:
        self.assertIsInstance(
            strategy.extended._get_cross_device_ops(var),
            cross_device_ops_lib.CollectiveAllReduce)

    strategy = mirrored_strategy.MirroredStrategy(["CPU:0", "CPU:1"])
    with strategy.scope():
      var = variables.Variable(1.)

    fn(var)

  def testMixedDevicesCollectiveOp(self, use_default):
    # [FUNCTION DOC]
    # Scenario: a strategy over one CPU and one GPU with a variable; use_default is ignored.
    # Skipped under XLA. Asserts the cross-device ops for the variable is reduction to one device.
    del use_default
    if util.is_xla_enabled():
      self.skipTest("All devices should be identical in XLA context.")

    # XLA is not supported if devices are not of the same type.
    strategy = mirrored_strategy.MirroredStrategy(["CPU:0", "GPU:0"])
    with strategy.scope():
      var = variables.Variable(1.)

    self.assertIsInstance(
        strategy.extended._get_cross_device_ops(var),
        cross_device_ops_lib.ReductionToOneDevice)

  def testMirroredStrategyInt32VariableCollectiveOp(self, use_default):
    # [FUNCTION DOC]
    # Scenario: a two-GPU strategy (default or explicit NCCL ops) with an integer variable. Skipped
    # under XLA. Asserts reduction to one device is chosen because collective ops do not support
    # 32-bit integers on GPU.
    if util.is_xla_enabled():
      self.skipTest("Only expected to run under non-XLA context.")

    strategy = mirrored_strategy.MirroredStrategy(
        ["GPU:0", "GPU:1"],
        cross_device_ops=None
        if use_default else cross_device_ops_lib.NcclAllReduce())
    with strategy.scope():
      # CollevtiveOp does not support int32 on GPU.
      var = variables.Variable(1)

    self.assertIsInstance(
        strategy.extended._get_cross_device_ops(var),
        cross_device_ops_lib.ReductionToOneDevice)


def one_device_combinations():
  # [FUNCTION DOC]
  # HELPER: supplies the parameter combinations for the one-device tests. Takes no arguments;
  # returns the cross product of the one-CPU and one-GPU mirrored strategies with graph and eager
  # modes.
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_one_cpu,
          strategy_combinations.mirrored_strategy_with_one_gpu,
      ],
      mode=["graph", "eager"])


@combinations.generate(one_device_combinations())
class MirroredOneDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.OneDeviceDistributionTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Tests for MirroredStrategy on a single device (one CPU or one GPU), in graph and eager modes.
  # Mostly delegates to shared base-class scenarios for minimize loss, replica id, run and
  # all-reduce with sum and mean (including gradients).

  def testMinimizeLoss(self, distribution):
    # [FUNCTION DOC]
    # Runs the shared minimize-loss scenario: eager variant in eager mode, graph variant otherwise.
    if context.executing_eagerly():
      self._test_minimize_loss_eager(distribution)
    else:
      self._test_minimize_loss_graph(distribution)

  def testReplicaId(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of replica ids.
    self._test_replica_id(distribution)

  def testCallAndMergeExceptions(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of errors in per-replica and merge calls.
    self._test_call_and_merge_exceptions(distribution)

  def testRun(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of running a function on the replicas.
    self._test_run(distribution)

  def testAllReduceSum(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of sum all-reduce.
    self._test_all_reduce_sum(distribution)

  def testAllReduceSumGradients(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of gradients through sum all-reduce.
    self._test_all_reduce_sum_gradients(distribution)

  def testAllReduceSumGradientTape(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of sum all-reduce gradients with a gradient tape.
    self._test_all_reduce_sum_gradient_tape(distribution)

  def testAllReduceMean(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of mean all-reduce.
    self._test_all_reduce_mean(distribution)

  def testAllReduceMeanGradients(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of gradients through mean all-reduce.
    self._test_all_reduce_mean_gradients(distribution)

  def testAllReduceMeanGradientTape(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the shared base-class check of mean all-reduce gradients with a gradient tape.
    self._test_all_reduce_mean_gradient_tape(distribution)


class MirroredStrategyVariableCreatorStackTest(
    test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that variable-creator scopes are tracked per replica thread rather than shared.

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph"]))
  def testCreatorStacksAreThreadLocal(self, distribution):
    # [FUNCTION DOC]
    # Scenario (graph mode): the main thread installs a creator returning the text main_thread; each
    # replica function installs its own creator appending a thread and replica-id suffix, creates a
    # variable, and pauses via a merge call so the other thread runs. Asserts the local results are
    # the main creator text with the thread_0 and thread_1 suffixes, showing creator stacks do not
    # leak between replica threads.
    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: installs a thread-specific creator, creates a variable, performs a
      # no-op merge call (which switches threads) and returns the variable.
      replica_id_str = str(self.evaluate(_replica_id()))

      def thread_creator_fn(next_creator, **kwargs):
        # [FUNCTION DOC]
        # Variable creator: calls the next creator and appends a suffix naming the replica thread.
        return next_creator(**kwargs) + ":thread_" + replica_id_str

      with variable_scope.variable_creator_scope(thread_creator_fn):
        # Create a variable in this scope.
        v = variable_v1.VariableV1(1.0)

        # This will pause the current thread, and execute the other thread.
        distribute_lib.get_replica_context().merge_call(lambda _: _)
      return v

    def main_thread_creator(next_creator, **kwargs):
      # [FUNCTION DOC]
      # Variable creator for the main thread: ignores the next creator and arguments and returns
      # the fixed text main_thread.
      # We are not using the underlying next_creator for test purposes.
      del next_creator, kwargs
      return "main_thread"

    with context.graph_mode(), \
        distribution.scope(), \
        variable_scope.variable_creator_scope(main_thread_creator):
      result = distribution.extended.call_for_each_replica(model_fn)
      result = distribution.experimental_local_results(result)
      expected = ("main_thread:thread_0", "main_thread:thread_1")
      self.assertEqual(expected, result)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class MirroredStrategyCallForEachReplicaTest(test.TestCase):
  # [CLASS DOC]
  # Tests calling functions on each replica of a GPU-and-CPU MirroredStrategy, including compiled
  # functions, trace caching, and merge-call restrictions inside compiled code.

  def testExecutingEagerlyOutsideFunction(self, distribution):
    # [FUNCTION DOC]
    # Scenario: query whether eager execution is active outside functions, both at top level and in
    # a function graph, inside and outside the strategy scope and inside a per-replica function.
    # Asserts the per-replica value and the scope value match, and the original value is preserved.
    """Verify we preserve the value of executing_eagerly_outside_functions()."""
    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: returns whether eager execution is active outside functions.
      return ops.executing_eagerly_outside_functions()

    originally = ops.executing_eagerly_outside_functions()
    with distribution.scope():
      in_scope = ops.executing_eagerly_outside_functions()
      in_model_fn = distribution.extended.call_for_each_replica(model_fn)
      unwrapped = distribution.experimental_local_results(in_model_fn)
      self.assertEqual(in_scope, unwrapped[0])
      self.assertEqual(in_scope, originally)

    # Verify this all again, but this time in a FuncGraph.
    with func_graph.FuncGraph("fg").as_default(), distribution.scope():
      in_scope = ops.executing_eagerly_outside_functions()
      in_model_fn = distribution.extended.call_for_each_replica(model_fn)
      unwrapped = distribution.experimental_local_results(in_model_fn)
      self.assertEqual(in_scope, unwrapped[0])
      self.assertEqual(in_scope, originally)

  def testFunctionInCallForEachReplica(self, distribution):
    # [FUNCTION DOC]
    # Scenario: call a compiled function on each replica; it records a trace entry and returns the
    # replica id. Asserts the local results are 0 and 1 and the function was traced once per
    # replica.
    traces = []
    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # Compiled per-replica function: records a trace marker and returns the replica id.
      traces.append(1)
      return distribute_lib.get_replica_context().replica_id_in_sync_group

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertEqual(
          (0, 1),
          self.evaluate(distribution.experimental_local_results(result)))
      self.assertLen(traces, distribution.num_replicas_in_sync)

  def testFunctionInCallForEachReplicaInsideAnotherFunction(self, distribution):
    # [FUNCTION DOC]
    # Scenario: like the previous test, but the per-replica call happens inside another compiled
    # step function. Asserts local results are 0 and 1 and one trace per replica.
    traces = []
    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # Compiled per-replica function: records a trace marker and returns the replica id.
      traces.append(1)
      return distribute_lib.get_replica_context().replica_id_in_sync_group

    @def_function.function
    def step():
      # [FUNCTION DOC]
      # Compiled outer function: calls the per-replica function on every replica and returns the
      # result.
      return distribution.extended.call_for_each_replica(model_fn)

    with distribution.scope():
      result = step()
      self.assertEqual(
          (0, 1),
          self.evaluate(distribution.experimental_local_results(result)))
      self.assertLen(traces, distribution.num_replicas_in_sync)

  def testControlFlowFunctionInCallForEachReplicaWithMergeCall(
      self, distribution):
    # [FUNCTION DOC]
    # Scenario: a compiled per-replica function runs a while loop whose body performs a merge call
    # that sum-reduces the loop counter. Asserts that calling it on each replica raises a runtime
    # error saying merge_call was called while defining a new graph.

    def merge_fn(strategy, value):
      # [FUNCTION DOC]
      # Merge function: sum-reduces the given value across replicas using the strategy.
      return strategy.reduce(reduce_util.ReduceOp.SUM, value, axis=None)

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # Compiled per-replica function: runs a loop (while counter is below two) whose body merges.

      def body_fn(i):
        # [FUNCTION DOC]
        # Loop body: performs a merge call with the merge function on the counter and returns it.
        return distribute_lib.get_replica_context().merge_call(
            merge_fn, args=(i,))

      return while_loop.while_loop_v2(lambda i: i < 2, body_fn, [0])

    with distribution.scope():
      with self.assertRaisesRegex(
          RuntimeError, "`merge_call` called while defining a new graph."):
        distribution.extended.call_for_each_replica(model_fn)

  def testNestedFunctionInCallForEachReplicaWithMergeCall(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the per-replica function calls a nested compiled function that performs a merge
    # call on a constant. Asserts a runtime error about merge_call while defining a new graph.

    def merge_fn(strategy, value):
      # [FUNCTION DOC]
      # Merge function: sum-reduces the given value across replicas using the strategy.
      return strategy.reduce(reduce_util.ReduceOp.SUM, value, axis=None)

    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: calls and returns the nested compiled function.

      @def_function.function
      def model_fn_nested():
        # [FUNCTION DOC]
        # Nested compiled function: merge-calls the merge function with the constant one.
        t = constant_op.constant(1)
        return distribute_lib.get_replica_context().merge_call(
            merge_fn, args=(t,))

      return model_fn_nested()

    with distribution.scope():
      with self.assertRaisesRegex(
          RuntimeError, "`merge_call` called while defining a new graph."):
        distribution.extended.call_for_each_replica(model_fn)

  def testFunctionInCallForEachReplicaWithMergeCall(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a compiled per-replica function performs a no-op merge call at its top level and
    # returns zero. Asserts the call succeeds and the result evaluates to zero.
    def merge_fn(_):
      # [FUNCTION DOC]
      # Merge function that does nothing.
      pass

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # Compiled per-replica function: performs the no-op merge call and returns 0.
      distribute_lib.get_replica_context().merge_call(merge_fn)
      return 0.

    with distribution.scope():
      self.assertEqual(
          self.evaluate(distribution.extended.call_for_each_replica(model_fn)),
          0.)

  def testFunctionInCallForEachReplicaCached(self, distribution):
    # [FUNCTION DOC]
    # Scenario: call the same compiled per-replica function ten times. After the first call asserts
    # the trace list is non-empty (records the device count); on later calls asserts no further
    # tracing occurred (the trace count stays the same).
    traces = []

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # Compiled per-replica function: records a trace marker (only runs while tracing).
      traces.append(None)

    self.assertEmpty(traces)

    for i in range(10):
      distribution.extended.call_for_each_replica(model_fn)

      if i == 0:
        num_devices = len(traces)
        self.assertGreater(num_devices, 0)
      else:
        # model_fn should not have been re-evaluated so the length should remain
        # the same.
        self.assertLen(traces, num_devices)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph"]))
class MirroredStrategyNameScopeTest(test.TestCase):
  # [CLASS DOC]
  # Tests (graph mode only) that name scopes and variable scopes are respected, and that tensor
  # and variable names on the second replica get a replica prefix or suffix.
  # NOTE(priyag): Names and name scopes are ignored in eager, hence we are not
  # testing this in eager mode.

  def testNameScope(self, distribution):
    # [FUNCTION DOC]
    # Scenario: inside a main name scope, a per-replica function creates constants a and b in a
    # nested scope with a merge call between. Asserts the first replica's names are prefixed with
    # the main and foo scopes and the second replica's names additionally carry a replica_1 segment.
    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: in a name scope foo, creates constant a, does a no-op merge call,
      # creates constant b, and returns both.
      with ops.name_scope("foo"):
        a = constant_op.constant(1.0, name="a")
        distribute_lib.get_replica_context().merge_call(lambda _: _)
        b = constant_op.constant(1.0, name="b")
      return a, b

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        result = distribution.extended.call_for_each_replica(model_fn)
        self.assertEqual(2, len(result))
        for v, name in zip(result, ["a", "b"]):
          self.assertIsInstance(v, values.DistributedValues)
          v0, v1 = distribution.experimental_local_results(v)
          self.assertEqual("main/foo/" + name + ":0", v0.name)
          self.assertEqual("main/replica_1/foo/" + name + ":0", v1.name)

  def testWithDefaultName(self, distribution):
    # [FUNCTION DOC]
    # Scenario: like the name scope test but using a name scope with only a default name. Asserts
    # the names are the default scope plus constant name for replica zero and carry a replica_1
    # prefix for the second replica.
    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: in a default-named scope foo, creates constants a and b with a no-op
      # merge call between and returns both.
      with ops.name_scope(None, "foo"):
        a = constant_op.constant(1.0, name="a")
        distribute_lib.get_replica_context().merge_call(lambda _: _)
        b = constant_op.constant(2.0, name="b")
      return a, b

    with context.graph_mode(), distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertEqual(2, len(result))
      for v, name in zip(result, ["a", "b"]):
        self.assertIsInstance(v, values.DistributedValues)
        v0, v1 = distribution.experimental_local_results(v)
        self.assertEqual("foo/" + name + ":0", v0.name)
        self.assertEqual("replica_1/foo/" + name + ":0", v1.name)

  # variable_v1.VariableV1() respects name scopes when creating
  # variables. On the other hand variable_scope.get_variable() ignores name
  # scopes but respects variable scope when creating variables. We test both
  # methods of creating variables to make sure that we have the same
  # variable names in both cases.
  def testNameScopeWithVariable(self, distribution):
    # [FUNCTION DOC]
    # Scenario: create variables with explicit names under name scopes: a in the main scope, b in
    # the per-replica function, c inside a merge call in a nested scope. Asserts replica zero names
    # follow the scopes and the second replica's variables get a replica_1 suffix.
    def in_cross_replica(_):
      # [FUNCTION DOC]
      # Merge function: creates and returns a variable named c.
      c = variable_v1.VariableV1(1.0, name="c")
      return c

    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates variable b, then in scope foo merge-calls to create c;
      # returns both.
      b = variable_v1.VariableV1(1.0, name="b")
      with ops.name_scope("foo"):
        c = distribute_lib.get_replica_context().merge_call(in_cross_replica)
      return b, c

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        a = variable_v1.VariableV1(1.0, name="a")
        result = distribution.extended.call_for_each_replica(model_fn)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = distribution.experimental_local_results(a)
      b0, b1 = distribution.experimental_local_results(result_b)
      c0, c1 = distribution.experimental_local_results(result_c)
      self.assertEqual("main/a:0", a0.name)
      self.assertEqual("main/a/replica_1:0", a1.name)
      self.assertEqual("main/b:0", b0.name)
      self.assertEqual("main/b/replica_1:0", b1.name)
      self.assertEqual("main/foo/c:0", c0.name)
      self.assertEqual("main/foo/c/replica_1:0", c1.name)

  def testNameScopeWithGetVariable(self, distribution):
    # [FUNCTION DOC]
    # Scenario: like the previous test but creating variables via get-variable, which ignores name
    # scopes. Asserts replica zero names are just a, b, c and the second replica's names have a
    # replica_1 suffix, with no scope prefixes.
    def in_cross_replica(_):
      # [FUNCTION DOC]
      # Merge function: gets or creates a variable named c of shape one and returns it.
      c = variable_scope.get_variable("c", [1])
      return c

    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: gets variable b, then in scope foo merge-calls to get c; returns both.
      b = variable_scope.get_variable("b", [1])
      with ops.name_scope("foo"):
        c = distribute_lib.get_replica_context().merge_call(in_cross_replica)
      return b, c

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        a = variable_scope.get_variable("a", [1])
        result = distribution.extended.call_for_each_replica(model_fn)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = distribution.experimental_local_results(a)
      b0, b1 = distribution.experimental_local_results(result_b)
      c0, c1 = distribution.experimental_local_results(result_c)
      self.assertEqual("a:0", a0.name)
      self.assertEqual("a/replica_1:0", a1.name)
      self.assertEqual("b:0", b0.name)
      self.assertEqual("b/replica_1:0", b1.name)
      self.assertEqual("c:0", c0.name)
      self.assertEqual("c/replica_1:0", c1.name)

  def testVariableScopeWithGetVariable(self, distribution):
    # [FUNCTION DOC]
    # Scenario: variables are fetched with get-variable under explicit variable scopes (main for a,
    # foo for c inside a merge call). Asserts replica zero names include the variable scopes and
    # the second replica's names carry a replica_1 suffix.

    def in_cross_replica(_):
      # [FUNCTION DOC]
      # Merge function: gets or creates a variable named c of shape one and returns it.
      c = variable_scope.get_variable("c", [1])
      return c

    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: gets variable b, then in variable scope foo merge-calls to get c;
      # returns both.
      b = variable_scope.get_variable("b", [1])
      with variable_scope.variable_scope("foo"):
        c = distribute_lib.get_replica_context().merge_call(in_cross_replica)
      return b, c

    with context.graph_mode(), distribution.scope():
      with variable_scope.variable_scope("main"):
        a = variable_scope.get_variable("a", [1])
        result = distribution.extended.call_for_each_replica(model_fn)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = distribution.experimental_local_results(a)
      b0, b1 = distribution.experimental_local_results(result_b)
      c0, c1 = distribution.experimental_local_results(result_c)
      self.assertEqual("main/a:0", a0.name)
      self.assertEqual("main/a/replica_1:0", a1.name)
      self.assertEqual("main/b:0", b0.name)
      self.assertEqual("main/b/replica_1:0", b1.name)
      self.assertEqual("main/foo/c:0", c0.name)
      self.assertEqual("main/foo/c/replica_1:0", c1.name)


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored3Devices",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.MirroredStrategy(
                    ["/device:GPU:0", "/device:GPU:1", "/device:CPU:0"]),
                required_gpus=2)
        ],
        mode=["graph", "eager"]))
class MirroredThreeDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Tests MirroredStrategy over three devices (two GPUs and a CPU); requires two GPUs.

  def testThreeDevices(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a per-replica function creates a variable named foo and does a no-op merge call.
    # Asserts the combined result is recognized as a mirrored variable named foo:0.
    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates variable foo, performs a no-op merge call, returns it.
      v = variable_v1.VariableV1(1.0, name="foo")
      distribute_lib.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertTrue(distribute_utils.is_mirrored(result))
      self.assertEqual("foo:0", result.name)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class MirroredVariableUpdateTest(test.TestCase):
  # [CLASS DOC]
  # Tests assign, add-assign and subtract-assign on mirrored variables in replica context and
  # cross-replica context, with different aggregation types.
  # The following tests check assign, assign_add and assign_sub on Mirrored
  # variables in replica and cross replica context.

  def testAssignMirroredVarReplicaContextWithoutAggregationType(self,
                                                                distribution):
    # [FUNCTION DOC]
    # Scenario: a mirrored variable (initial value 1, no aggregation set) is assigned 5 from each
    # replica. Asserts the variable then reads 5.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates and returns a variable named foo with initial value one.
      v = variable_v1.VariableV1(1.0, name="foo")
      return v

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: assigns 5 to the mirrored variable and returns the result.
        return mirrored_var.assign(5.0)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(5.0, self.evaluate(mirrored_var))

  def testAssignMirroredVarReplicaContextWithSum(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a mirrored variable with sum aggregation is assigned the same plain value 5 from
    # every replica. With merge calls, asserts a value error saying the non-per-replica value
    # cannot be reduced with the sum operation; without merge calls, asserts the first replica's
    # result is 5.
    # Test that we don't reduce a non-per-replica value with the "sum"
    # aggregation type.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates a variable foo (initial value one, sum aggregation).
      v = variable_v1.VariableV1(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.SUM)
      return v

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: assigns 5 to the mirrored variable and returns the result.
        return mirrored_var.assign(5.0)

      if distribution.extended._use_merge_call():
        with self.assertRaisesRegex(
            ValueError, "A non-DistributedValues value 5.0 cannot be reduced "
            "with the given reduce op ReduceOp.SUM."):
          self.evaluate(distribution.experimental_local_results(
              distribution.extended.call_for_each_replica(model_fn)))
      else:
        result = self.evaluate(
            distribution.experimental_local_results(
                distribution.extended.call_for_each_replica(model_fn)))
        self.assertAllEqual(result[0], 5.0)

  def testAssignMirroredVarCrossDeviceContext(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a mirrored variable (initial value 1) is assigned 6 directly from cross-replica
    # context. Asserts it reads 1 before and the assignment result is 6.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates and returns a variable foo with initial value one.
      return variable_v1.VariableV1(1.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign(6.0))
      self.assertEqual(6.0, mirrored_var_result)

  def testAssignMirroredVarReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a mirrored variable with mean aggregation (initial value 1) is assigned each
    # replica's id (0 and 1). Asserts the variable then reads 0.5, the mean.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates a variable foo (initial value one, mean aggregation).
      return variable_v1.VariableV1(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: assigns the replica id (cast to the variable type) to the variable.
        value = math_ops.cast(
            distribute_lib.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign(value)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(0.5, self.evaluate(mirrored_var))

  def testAssignMirroredVarReplicaContextWithSingleValue(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a mean-aggregated mirrored variable (initial value 1) is assigned the same plain
    # value 5 by every replica. Asserts it reads 5 afterwards.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates a variable foo (initial value one, mean aggregation).
      return variable_v1.VariableV1(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: assigns 5 to the mirrored variable and returns the result.
        return mirrored_var.assign(5.0)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(5.0, self.evaluate(mirrored_var))

  def testAssignAddMirroredVarCrossDeviceContext(self, distribution):
    # [FUNCTION DOC]
    # Scenario: from cross-replica context, add 6 to a mirrored variable (initial value 1) with
    # read-value true, then add 2 with read-value false. Asserts the returned value is 7, both
    # component values are 7 then 9, and the variable's device list still matches the strategy's
    # worker devices.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates and returns a variable foo with initial value one.
      return variable_v1.VariableV1(1.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      # read_value == True
      mirrored_var_result = self.evaluate(
          mirrored_var.assign_add(6.0, read_value=True))
      self.assertEqual(7.0, mirrored_var_result)
      self.assertEqual(
          7.0,
          self.evaluate(
              distribution.experimental_local_results(mirrored_var)[0]))
      self.assertEqual(
          7.0,
          self.evaluate(
              distribution.experimental_local_results(mirrored_var)[1]))
      self.assertEqual(
          distribution.extended.worker_devices[0], mirrored_var._devices[0])
      self.assertEqual(
          distribution.extended.worker_devices[1], mirrored_var._devices[1])

      # read_value == False
      self.evaluate(mirrored_var.assign_add(2.0, read_value=False))
      self.assertEqual(
          9.0,
          self.evaluate(
              distribution.experimental_local_results(mirrored_var)[0]))
      self.assertEqual(
          9.0,
          self.evaluate(
              distribution.experimental_local_results(mirrored_var)[1]))
      self.assertEqual(
          distribution.extended.worker_devices[0], mirrored_var._devices[0])
      self.assertEqual(
          distribution.extended.worker_devices[1], mirrored_var._devices[1])

  def testAssignAddMirroredVarReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a mean-aggregated mirrored variable (initial value 1) gets each replica's id added
    # (0 and 1). Asserts it reads 1.5 afterwards.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates a variable foo (initial value one, mean aggregation).
      return variable_v1.VariableV1(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: adds the replica id (cast to the variable type) to the variable.
        value = math_ops.cast(
            distribute_lib.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign_add(value)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(1.5, self.evaluate(mirrored_var))

  def testAssignAddMirroredVarReplicaContextWithSingleValue(self, distribution):
    # [FUNCTION DOC]
    # Scenario: every replica adds the same plain value 5 to a mean-aggregated mirrored variable
    # (initial value 1). Asserts it reads 6 afterwards.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates a variable foo (initial value one, mean aggregation).
      return variable_v1.VariableV1(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: adds 5 to the mirrored variable and returns the result.
        return mirrored_var.assign_add(5.0)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(6.0, self.evaluate(mirrored_var))

  def testAssignSubMirroredVarCrossDeviceContext(self, distribution):
    # [FUNCTION DOC]
    # Scenario: from cross-replica context, subtract 2 from a mirrored variable (initial value 5).
    # Asserts the result and both component values are 3 and the device list matches the worker
    # devices.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates and returns a variable foo with initial value five.
      return variable_v1.VariableV1(5.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign_sub(2.0))
      self.assertEqual(3.0, mirrored_var_result)
      self.assertEqual(
          3.0,
          self.evaluate(
              distribution.experimental_local_results(mirrored_var)[0]))
      self.assertEqual(
          3.0,
          self.evaluate(
              distribution.experimental_local_results(mirrored_var)[1]))
      self.assertEqual(
          distribution.extended.worker_devices[0], mirrored_var._devices[0])
      self.assertEqual(
          distribution.extended.worker_devices[1], mirrored_var._devices[1])

  def testAssignSubMirroredVarReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Scenario: each replica subtracts its replica id (0 and 1) from a mean-aggregated mirrored
    # variable (initial value 5). Asserts it reads 4.5 afterwards.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates a variable foo (initial value five, mean aggregation).
      return variable_v1.VariableV1(
          5.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: subtracts the replica id (cast to the variable type).
        value = math_ops.cast(
            distribute_lib.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign_sub(value)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(4.5, self.evaluate(mirrored_var))

  def testAssignSubMirroredVarReplicaContextWithSingleValue(self, distribution):
    # [FUNCTION DOC]
    # Scenario: every replica subtracts the same plain value 1 from a mean-aggregated mirrored
    # variable (initial value 5). Asserts it reads 4 afterwards.
    def var_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates a variable foo (initial value five, mean aggregation).
      return variable_v1.VariableV1(
          5.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: subtracts 1 from the mirrored variable and returns the result.
        return mirrored_var.assign_sub(1.0)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(4.0, self.evaluate(mirrored_var))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class MirroredAndSyncOnReadVariableInitializerTest(test.TestCase):
  # [CLASS DOC]
  # Tests that mirrored and sync-on-read variables report uninitialized until their initializer
  # runs (graph mode only).

  def testAssignMirroredVarInitializer(self, distribution):
    # [FUNCTION DOC]
    # Scenario (graph mode): create a mirrored variable. Asserts it is mirrored, reports not
    # initialized, and reports initialized after running its initializer.
    # This test is not eager compatible since in eager variables are initialized
    # upon construction instead of once the initialization op is run.
    with context.graph_mode():
      def var_fn():
        # [FUNCTION DOC]
        # Per-replica function: creates and returns a variable foo with initial value one.
        v = variable_v1.VariableV1(1.0, name="foo")
        return v

      with distribution.scope():
        mirrored_var = distribution.extended.call_for_each_replica(var_fn)
        self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
        self.assertFalse(self.evaluate(mirrored_var.is_initialized()))
        self.evaluate(mirrored_var.initializer)
        self.assertTrue(self.evaluate(mirrored_var.is_initialized()))

  def testAssignReplicaLocalVarInitializer(self, distribution):
    # [FUNCTION DOC]
    # Scenario (graph mode): create a sync-on-read variable with sum aggregation. Asserts it is
    # recognized as sync-on-read, reports not initialized, and reports initialized after running
    # its initializer.
    # This test is not eager compatible since in eager variables are initialized
    # upon construction instead of once the initialization op is run.
    with context.graph_mode():
      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: creates a sync-on-read variable (initial value one, sum
        # aggregation), asserts it is sync-on-read, and returns it.
        v_sum = variable_v1.VariableV1(
            1.0,
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        self.assertTrue(distribute_utils.is_sync_on_read(v_sum))
        return v_sum

      with distribution.scope():
        sync_on_read_var = distribution.extended.call_for_each_replica(
            model_fn)
        self.assertTrue(distribute_utils.is_sync_on_read(sync_on_read_var))
        self.assertFalse(self.evaluate(sync_on_read_var.is_initialized()))
        self.evaluate(sync_on_read_var.initializer)
        self.assertTrue(self.evaluate(sync_on_read_var.is_initialized()))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class SyncOnReadVariableAssignTest(test.TestCase):
  # [CLASS DOC]
  # Tests assigning to sync-on-read variables from cross-replica context and reading them back
  # under sum and mean aggregation.

  def testAssignReplicaLocalVarSumAggregation(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a sync-on-read variable with sum aggregation holds 1 on each of two replicas.
    # Asserts reading it gives 2 (the sum); after assigning 6 from cross-replica context, reading
    # gives 6 (the value is split across replicas and summed back).
    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates and returns a sync-on-read variable (initial value one, sum
      # aggregation).
      v_sum = variable_v1.VariableV1(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM)
      return v_sum

    with distribution.scope():
      sync_on_read_var = distribution.extended.call_for_each_replica(model_fn)
      self.assertTrue(distribute_utils.is_sync_on_read(sync_on_read_var))
      self.evaluate(variables.global_variables_initializer())
      # Each replica has a value of 1.0 assigned to it in replica context.
      # When we read the value using `read_var` we should see the SUM of each of
      # values on each of the replicas.
      self.assertEqual(2.0, self.evaluate(
          distribution.extended.read_var(sync_on_read_var)))
      # Assigning 6.0 in cross replica context will assign a value of
      # 6.0/num_replicas to each replica.
      tlv_ops = sync_on_read_var.assign(6.0)
      self.evaluate(tlv_ops)
      # On reading the sync on read var we should get the assigned value back.
      # The value on all the replicas are added before being returned by
      # `read_var`.
      self.assertEqual(6.0, self.evaluate(
          distribution.extended.read_var(sync_on_read_var)))

  def testAssignReplicaLocalVarMeanAggregation(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a sync-on-read variable with mean aggregation holds 1 on each replica. Asserts
    # reading it gives 1; after assigning 6 from cross-replica context, reading gives 6.
    def model_fn():
      # [FUNCTION DOC]
      # Per-replica function: creates and returns a sync-on-read variable (initial value one, mean
      # aggregation).
      v_sum = variable_v1.VariableV1(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.MEAN)
      return v_sum

    with distribution.scope():
      sync_on_read_var = distribution.extended.call_for_each_replica(model_fn)
      self.assertTrue(distribute_utils.is_sync_on_read(sync_on_read_var))
      self.evaluate(variables.global_variables_initializer())
      # Each replica has a value of 1.0 assigned to it in replica context.
      # When we read the value using `read_var` we should see the MEAN of values
      # on all replicas which is the value assigned in replica context.
      self.assertEqual(1.0, self.evaluate(
          distribution.extended.read_var(sync_on_read_var)))
      tlv_ops = sync_on_read_var.assign(6.0)
      self.evaluate(tlv_ops)
      # On reading the sync on read var we should get the MEAN of all values
      # which is equal to the value assigned.
      self.assertEqual(6.0, self.evaluate(
          distribution.extended.read_var(sync_on_read_var)))


class MockModel(object):
  # [CLASS DOC]
  # HELPER: tiny stand-in model used by the multi-worker and gradient tests. Holds a list of
  # variables (one with value 1.25, optionally a second with value 2.0) and is callable like a
  # layer to produce a scalar from them.

  def __init__(self, two_variables=False):
    # [FUNCTION DOC]
    # Constructor. Takes a boolean two_variables (default false). Creates a variable named
    # dummy_var1 (1.25) and, if requested, dummy_var2 (2.0), stored in the variables list.
    self.variables = []
    self.variables.append(variable_v1.VariableV1(1.25, name="dummy_var1"))
    if two_variables:
      self.variables.append(variable_v1.VariableV1(2.0, name="dummy_var2"))

  def __call__(self, factor=2):
    # [FUNCTION DOC]
    # Evaluates the model: returns the first variable multiplied by factor (default 2), plus the
    # second variable when the model has more than one variable.
    x = factor * self.variables[0]
    if len(self.variables) > 1:
      x += self.variables[1]
    return x


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.MirroredStrategy(
                    devices=mirrored_strategy.all_local_devices(),
                    cross_device_ops=cross_device_ops_lib.ReductionToOneDevice(
                    ),
                ),
                required_gpus=1)
        ],
        mode=["graph"]))
class MultiWorkerMirroredStrategyTest(
    multi_worker_test_base.MultiWorkerTestBase,
    strategy_test_lib.DistributionTestBase):
  # [CLASS DOC]
  # Tests MirroredStrategy configured with a two-worker cluster specification (graph mode only,
  # requires a GPU), using reduction to one device for cross-device ops.

  def _configure_distribution_strategy(self, distribution):
    # [FUNCTION DOC]
    # HELPER: configures the given strategy with a cluster specification of two workers (tasks zero
    # and one of the worker job). Returns nothing; mutates the strategy.
    cluster_spec = server_lib.ClusterSpec({
        "worker": ["/job:worker/task:0", "/job:worker/task:1"]
    })
    distribution.configure(cluster_spec=cluster_spec)

  def test_num_replicas_in_sync(self, distribution):
    # [FUNCTION DOC]
    # Scenario: configure two workers, then assert the number of replicas in sync equals the local
    # GPU count times two.
    self._configure_distribution_strategy(distribution)
    # We calculate the total number of gpus across the workers(2) specified in
    # the cluster spec.
    self.assertEqual(context.num_gpus() * 2, distribution.num_replicas_in_sync)

  def testMinimizeLossGraph(self, distribution):
    # [FUNCTION DOC]
    # Scenario: configure two workers, then run the shared graph-mode minimize-loss check with
    # learning rate 0.05.
    self._configure_distribution_strategy(distribution)
    self._test_minimize_loss_graph(distribution, learning_rate=0.05)

  def testDeviceScope(self, distribution):
    # [FUNCTION DOC]
    # Scenario: configure two workers; in the strategy scope create a constant, and one under an
    # explicit CPU device. Asserts the first is placed on worker task zero and the second on that
    # task's CPU device.
    """Test the device scope of multi-worker MirroredStrategy."""
    self._configure_distribution_strategy(distribution)
    with distribution.scope():
      a = constant_op.constant(1.)
      with ops.device("/cpu:0"):
        b = constant_op.constant(1.)
      self.assertEqual(a.device, "/job:worker/task:0")
      self.assertEqual(b.device, "/job:worker/task:0/device:CPU:0")

  def testMakeInputFnIteratorWithDataset(self, distribution):
    # [FUNCTION DOC]
    # Scenario: configure two workers; the input function builds a range-of-100 dataset. Asserts
    # (via the base-class helper) the iterator yields groups of consecutive numbers per GPU,
    # repeated per worker, with two input pipelines whose ids vary per call.
    self._configure_distribution_strategy(distribution)
    dataset_fn = lambda: dataset_ops.Dataset.range(100)
    num_gpus = context.num_gpus()
    num_workers = 2

    expected_values = [[i+j for j in range(num_gpus)] * num_workers
                       for i in range(0, 100, num_gpus)]

    with context.graph_mode(), self.cached_session() as sess:
      # `expected_input_pipeline_id` is None because the input_fn will be called
      # multiple times, each with a different input_pipeline_id.
      input_fn = self._input_fn_to_test_input_context(
          dataset_fn,
          expected_num_replicas_in_sync=num_workers*num_gpus,
          expected_num_input_pipelines=num_workers,
          expected_input_pipeline_id=None)
      iterator = distribution.make_input_fn_iterator(input_fn)
      self._test_input_fn_iterator(
          iterator, distribution.extended.worker_devices, expected_values, sess)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    # [FUNCTION DOC]
    # Scenario: configure two workers; the input function returns the get-next callable of a
    # one-shot iterator over a range of 100. Asserts (via the base-class helper, without
    # reinitialization and ignoring order) the iterator yields groups of consecutive numbers per
    # GPU repeated per worker.
    self._configure_distribution_strategy(distribution)
    def fn():
      # [FUNCTION DOC]
      # Input function: creates a one-shot iterator over a range of 100 and returns its get-next
      # callable.
      dataset = dataset_ops.Dataset.range(100)
      it = dataset_ops.make_one_shot_iterator(dataset)
      return it.get_next
    num_gpus = context.num_gpus()
    num_workers = 2

    expected_values = []
    for i in range(0, 100, num_gpus):
      expected_values.append([i+j for j in range(num_gpus)] * num_workers)

    with context.graph_mode(), self.cached_session() as sess:
      # `expected_input_pipeline_id` is None because the input_fn will be called
      # multiple times, each with a different input_pipeline_id.
      input_fn = self._input_fn_to_test_input_context(
          fn,
          expected_num_replicas_in_sync=num_workers*num_gpus,
          expected_num_input_pipelines=num_workers,
          expected_input_pipeline_id=None)
      iterator = distribution.make_input_fn_iterator(input_fn)
      self._test_input_fn_iterator(
          iterator, distribution.extended.worker_devices, expected_values, sess,
          test_reinitialize=False, ignore_order=True)

  def testUpdateConfigProto(self, distribution):
    # [FUNCTION DOC]
    # Scenario: configure the strategy with two fake workers and update an empty session
    # configuration through it. Asserts the returned configuration isolates session state.
    distribution.configure(cluster_spec={"worker": ["fake1", "fake2"]})

    config_proto = config_pb2.ConfigProto()
    new_config = distribution.update_config_proto(config_proto)

    # Verify isolate_session_state
    self.assertTrue(new_config.isolate_session_state)


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.MirroredStrategy(
                    devices=["/job:worker/task:0/gpu:{}".format(
                        i) for i in range(context.num_gpus())]),
                required_gpus=1)
        ],
        mode=["graph"]))
class RemoteSingleWorkerMirroredStrategyGraph(
    multi_worker_test_base.SingleWorkerTestBaseGraph,
    strategy_test_lib.RemoteSingleWorkerMirroredStrategyBase):
  # [CLASS DOC]
  # Tests MirroredStrategy over the GPUs of a remote single worker (graph mode, requires a GPU),
  # delegating each scenario to base-class helpers.

  def _get_num_gpus(self):
    # [FUNCTION DOC]
    # HELPER: tells the base class how many GPUs are available; returns the local GPU count.
    return context.num_gpus()

  def testNumReplicasInSync(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the base-class check of the replicas-in-sync count.
    self._testNumReplicasInSync(distribution)

  def testMinimizeLoss(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the base-class minimize-loss check.
    self._testMinimizeLoss(distribution)

  def testDeviceScope(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the base-class device scope check.
    self._testDeviceScope(distribution)

  def testMakeInputFnIteratorWithDataset(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the base-class check of input iterators built from datasets.
    self._testMakeInputFnIteratorWithDataset(distribution)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    # [FUNCTION DOC]
    # Delegates to the base-class check of input iterators built from callables.
    self._testMakeInputFnIteratorWithCallable(distribution)


class MultiWorkerMirroredStrategyTestWithChief(
    multi_worker_test_base.MultiWorkerTestBase,
    strategy_test_lib.DistributionTestBase):
  # [CLASS DOC]
  # Tests MirroredStrategy against an in-process cluster with two workers and a chief, using
  # reduction to one device for cross-device ops. The cluster is created once per class.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture: creates a local in-process cluster of two workers, no parameter servers
    # and one chief, storing the cluster specification and the chief's address as the default
    # target on the class. Returns nothing; starts servers.
    """Create a local cluster with 2 workers and 1 chief."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=2, num_ps=0, has_chief=True)
    cls._default_target = "grpc://" + cls._cluster_spec["chief"][0]

  def _make_cross_device_ops(self):
    # [FUNCTION DOC]
    # HELPER: returns a new reduction-to-one-device cross-device ops object (overridable by
    # subclasses to use other implementations).
    return cross_device_ops_lib.ReductionToOneDevice()

  def testMinimizeLossGraph(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): create a strategy over default devices with the helper's cross-device
    # ops, configure it with the cluster, and run the shared minimize-loss check with learning rate
    # 0.05.
    with context.graph_mode():
      strategy = mirrored_strategy.MirroredStrategy(
          cross_device_ops=self._make_cross_device_ops())
      strategy.configure(cluster_spec=self._cluster_spec)
      self._test_minimize_loss_graph(strategy, learning_rate=0.05)

  def testMinimizeLossGraphMirroredStrategy(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): like the previous test but with all local devices passed explicitly.
    with context.graph_mode():
      strategy = mirrored_strategy.MirroredStrategy(
          mirrored_strategy.all_local_devices(),
          cross_device_ops=self._make_cross_device_ops())
      strategy.configure(cluster_spec=self._cluster_spec)
      self._test_minimize_loss_graph(strategy, learning_rate=0.05)

  def testMinimizeLossGraphMirroredStrategyWithOneNode(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): set the cluster environment variable to a cluster containing only the
    # chief and create a strategy; with no GPUs asserts reduction to one device is used. Then the
    # test unconditionally skips, citing a known bug, so the minimize-loss check after the skip
    # never runs (dead code).
    # KNOWN ISSUE: the minimize-loss check at the end is unreachable because the test skips first.
    with context.graph_mode():
      cluster_spec = {}
      cluster_spec["chief"] = self._cluster_spec["chief"]
      tf_config = {"cluster": cluster_spec}
      with test.mock.patch.dict("os.environ",
                                {"TF_CONFIG": json.dumps(tf_config)}):
        strategy = mirrored_strategy.MirroredStrategy()
        if context.num_gpus() == 0:
          self.assertIsInstance(strategy.extended._cross_device_ops,
                                cross_device_ops_lib.ReductionToOneDevice)
      self.skipTest("b/130551176, run the following once fixed.")
      self._test_minimize_loss_graph(strategy, learning_rate=0.05)

  def testInitializeFromTFConfig(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): set the cluster environment variable to the full cluster (two workers
    # and a chief) and create a strategy. Asserts the replica count equals three times the larger
    # of the local GPU count and one.
    with context.graph_mode():
      tf_config = {"cluster": self._cluster_spec}
      with test.mock.patch.dict("os.environ",
                                {"TF_CONFIG": json.dumps(tf_config)}):
        strategy = mirrored_strategy.MirroredStrategy(
            cross_device_ops=self._make_cross_device_ops())
        self.assertEqual(
            max(context.num_gpus(), 1) * 3, strategy.num_replicas_in_sync)

  def testSummaryForReplicaZeroOnly(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): configure a strategy over all local devices with the cluster and run
    # the shared check that only replica zero writes summaries.
    with context.graph_mode():
      strategy = mirrored_strategy.MirroredStrategy(
          mirrored_strategy.all_local_devices(),
          cross_device_ops=self._make_cross_device_ops())
      strategy.configure(cluster_spec=self._cluster_spec)
      self._test_summary_for_replica_zero_only(strategy)


class MirroredVariableStopGradientTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests gradient computation involving variables created under a one-device mirrored strategy.

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_one_gpu,
          ],
          mode=["graph"]))
  def testMirroredVariableAsStopGradient(self, distribution):
    # [FUNCTION DOC]
    # Scenario (graph mode): in the strategy scope, compute a product of a constant and a
    # variable, then take gradients of the variable with respect to that product while marking the
    # variable as a stop-gradient. Asserts the gradient is null.
    with distribution.scope():
      inp = constant_op.constant(1.0)
      x = variables.Variable(1.0)
      y = inp*x
      grads = gradients.gradients(x, y, stop_gradients=x)
      self.assertIsNone(grads[0])


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["eager"]))
class FunctionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager-mode tests of compiled functions used with MirroredStrategy: device placement of
  # backward-pass nodes, autograph preservation, and traceback filtering preservation.

  def testBackwardFunctionDevicePlacement(self, distribution):
    # [FUNCTION DOC]
    # Scenario: create mirrored variables w and b; trace a compiled forward function on their
    # primary components; run, on every replica, a gradient tape computing gradients of the loss
    # with respect to w and b through the concrete function, with run metadata enabled. Asserts
    # both gradients equal 1 and that the gradient multiply node appears in the partitioned graphs
    # on both the GPU and CPU devices (i.e. the backward function runs on each replica's device).
    with distribution.scope():
      w = variable_v1.VariableV1([1.5], name="w")
      b = variable_v1.VariableV1([0.5], name="b")

    @def_function.function
    def forward(x, w, b):
      # [FUNCTION DOC]
      # Compiled forward computation: input times w plus b.
      return x * w + b

    x = array_ops.identity([1.0], name="x_useless")
    concrete_forward = forward.get_concrete_function(x, w._primary, b._primary)

    with distribution.scope():

      def replica_fn():
        # [FUNCTION DOC]
        # Per-replica function: under a gradient tape, runs the traced forward function on the
        # local components, subtracts one, and returns gradients with respect to w and b.
        with backprop.GradientTape() as t:
          x = array_ops.identity([1.0], name="x")
          loss = concrete_forward(x, w._get(), b._get()) - [1.0]
          return t.gradient(loss, [w, b])

      def step_fn():
        # [FUNCTION DOC]
        # Runs the per-replica function on all replicas and returns the per-replica results.
        return distribution.run(replica_fn)

      context.enable_run_metadata()
      g1, g2 = step_fn()
      run_metadata = context.export_run_metadata()
      context.disable_run_metadata()
      self.assertEqual(self.evaluate(g1._primary), 1.0)
      self.assertEqual(self.evaluate(g2._primary), 1.0)

      # Verify that this node runs on both devices.
      node_name = "gradients_mul_grad_mul_1_x"
      devices_for_this_node = set()
      for partition_graph in run_metadata.partition_graphs:
        for node in partition_graph.node:
          if node.name == node_name:
            devices_for_this_node.add(node.device)
      devices = [device_util.resolve("/device:GPU:0"),
                 device_util.resolve("/device:CPU:0")]
      self.assertSetEqual(devices_for_this_node, set(devices))

  def testFuctionPreservesAutoGraph(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a helper function, called from a compiled function that is run on the replicas,
    # asserts it executes inside autograph-generated code. Verifies autograph conversion is
    # preserved when running through the strategy. (The method name has a typo: "Fuction".)
    def f():
      # [FUNCTION DOC]
      # Helper called by the compiled function: asserts that it runs inside autograph-generated
      # code and returns one.
      self.assertTrue(converter_testing.is_inside_generated_code())
      return 1

    with distribution.scope():

      @def_function.function
      def replica_fn():
        # [FUNCTION DOC]
        # Compiled per-replica function: calls and returns the helper.
        return f()

      distribution.run(replica_fn)

  def testPreserveTracebackFiltering(self, distribution):
    # [FUNCTION DOC]
    # Scenario: disable traceback filtering globally (asserting it reads as disabled), then run a
    # function on the replicas that asserts filtering is still disabled. Verifies the setting is
    # preserved inside strategy run. Note it does not restore the global setting afterwards.
    traceback_utils.disable_traceback_filtering()
    self.assertFalse(traceback_utils.is_traceback_filtering_enabled())

    def f():
      # [FUNCTION DOC]
      # Per-replica function: asserts traceback filtering is disabled.
      self.assertFalse(traceback_utils.is_traceback_filtering_enabled())

    distribution.run(f)


def _replica_id():
  # [FUNCTION DOC]
  # HELPER: returns the current replica's id within the sync group as a tensor (wrapping a plain
  # number in a constant if needed, then passing through an identity operation). Must be called
  # inside a replica context.
  replica_id = distribute_lib.get_replica_context().replica_id_in_sync_group
  if not isinstance(replica_id, tensor_lib.Tensor):
    replica_id = constant_op.constant(replica_id)
  return array_ops.identity(replica_id)


def _replica_id_as_int():
  # [FUNCTION DOC]
  # HELPER: returns the current replica's id within the sync group as a plain number, extracting
  # the constant value when the id is a tensor. Must be called inside a replica context.
  replica_id = distribute_lib.get_replica_context().replica_id_in_sync_group
  if isinstance(replica_id, tensor_lib.Tensor):
    replica_id = tensor_util.constant_value(replica_id)
  return replica_id


if __name__ == "__main__":
  # TODO(b/172304955)
  test_util.main(config_logical_devices=False)
