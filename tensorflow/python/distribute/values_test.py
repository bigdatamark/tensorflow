# [MODULE DOC]
# Tests for the distributed values library (values.py and the TPU variable variants). Covers
# creation of distributed values from a function (including placement, nested structures, sparse
# tensors and use inside functions), per-replica value misuse, per-worker resource tracing in
# dataset map, the delegate wrapper (attribute forwarding, operator overloads, copying), and
# sync-on-read ("replica local") variables: properties, conversion, reads, assignments and
# save / restore round trips between mirrored and ordinary variables. Run as a program it starts
# the distribute test main.

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
"""Tests for the distributed values library."""

import copy
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util as ds_test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import saver as saver_lib


def _device_str(d):
  # [FUNCTION DOC]
  # HELPER: small test utility.
  # KNOWN ISSUE: appears unused in this file (dead helper).
  # Takes a device index and returns the full device name text of the GPU with that index.
  return "/device:GPU:" + str(d)


def _nested_value(d):
  # [FUNCTION DOC]
  # HELPER: small test utility.
  # KNOWN ISSUE: appears unused in this file (dead helper).
  # Takes a suffix string and returns a nested structure: a tuple of a string, a list (string,
  # mapping of two strings, string) and a string, each built by appending the suffix to a letter.
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


def mirrored_and_tpu_strategy_combinations():
  # [FUNCTION DOC]
  # HELPER: builds the shared test parameter combinations. Returns combinations over two mirrored
  # strategies, three TPU strategies (plain, packed variables, spmd), crossed with graph and eager
  # modes, for use with the combinations decorator.
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
          strategy_combinations.tpu_strategy_spmd,
      ],
      mode=["graph", "eager"])


class DistributedValuesTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the strategy method that creates distributed values from a per-replica value function:
  # tensors, arrays, tuples, nested per-replica structures, sparse tensors, use in a compiled
  # function, and device placement. All tests run eagerly and skip when version-2 behavior is off.

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueFromTensor(self, distribution):
    # [FUNCTION DOC]
    # Skips unless V2 behavior is enabled. Builds distributed values from a function returning the
    # same constant tensor on every replica, gathers them, and asserts the result is a vector of
    # ones with one entry per replica in sync.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    single_value = constant_op.constant(1)
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; ignores the replica context and returns the shared constant.
      del ctx
      return single_value

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    self.assertAllEqual(
        ds_test_util.gather(distribution, distributed_values),
        constant_op.constant(1., shape=(distribution.num_replicas_in_sync)))

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueSingleNumpyArrayConstant(self, distribution):
    # [FUNCTION DOC]
    # Skips unless V2 behavior is enabled. Builds distributed values from a function returning the
    # same three-element numpy array on every replica, gathers, and asserts the result holds that
    # array repeated once per replica.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    array_value = np.array([1., 2., 3.])
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; ignores the replica context and returns the shared array.
      del ctx
      return array_value

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    self.assertAllEqual(
        ds_test_util.gather(distribution, distributed_values).numpy(),
        [[1., 2., 3.]] * distribution.num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueTupleConstant(self, distribution):
    # [FUNCTION DOC]
    # Skips unless V2 behavior is enabled. Builds distributed values from a function returning the
    # same tuple of three numbers on every replica, gathers, and asserts the result is a tuple of
    # three lists, each repeating one tuple element once per replica.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    tuple_value = (1., 2., 3.)
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; ignores the replica context and returns the shared tuple.
      del ctx
      return tuple_value
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    distributed_values = ds_test_util.gather(distribution, distributed_values)

    # Expected output for 2 replicas:
    # ([1.0, 1.0], [2.0, 2.0], [3.0, 3.0])
    expected = tuple([v for i in range(distribution.num_replicas_in_sync)]
                     for v in tuple_value)
    self.assertAllEqual(distributed_values, expected)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueNestedStructurePerReplica(self, distribution):
    # [FUNCTION DOC]
    # Skips unless V2 behavior is enabled. Builds distributed values from a function returning a
    # tuple whose entries are scaled by the replica index, gathers, and asserts each entry holds
    # the base number multiplied by 0, 1, ... across replicas.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    tuple_value = (1., 2., 3.)
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; returns a tuple of the base numbers each multiplied by the replica
      # id in the sync group from the context.
      per_replica = []
      for val in tuple_value:
        per_replica.append(val * ctx.replica_id_in_sync_group)
      return tuple(per_replica)
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    distributed_values = ds_test_util.gather(distribution, distributed_values)

    # Expected output for 2 replicas:
    # ([0.0, 1.0], [0.0, 2.0], [0.0, 3.0])
    expected = tuple([v * i for i in range(distribution.num_replicas_in_sync)]
                     for v in tuple_value)
    self.assertAllEqual(distributed_values, expected)

  # NOTE(priyag): Cannot test this with MultiWorkerMirroredStrategy because
  # collective ops do not support SparseTensors.
  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueSpareTensor(self, distribution):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "Spare" but the test is about sparse tensors (typo).
    # Skips unless V2 behavior is enabled. Builds distributed values from a function returning a
    # sparse tensor (two non-zero entries in a three by four shape), gets the local results, and
    # asserts each replica's densified tensor has the expected two non-zero entries.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; ignores the context and returns the shared sparse tensor.
      del ctx
      return sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    local_results = distribution.experimental_local_results(distributed_values)
    for i in range(distribution.num_replicas_in_sync):
      self.assertAllEqual(
          sparse_ops.sparse_tensor_to_dense(local_results[i]),
          [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]])

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueExtractFromArray(self, distribution):
    # [FUNCTION DOC]
    # Skips unless V2 behavior is enabled. Builds distributed values from a function that picks
    # the entry of a range indexed by the replica id, gathers, and asserts the result equals the
    # range 0 .. number of replicas minus one.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    multiple_values = range(distribution.num_replicas_in_sync)
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; returns the range entry at the replica id in the sync group.
      return multiple_values[ctx.replica_id_in_sync_group]
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    distributed_values = ds_test_util.gather(distribution, distributed_values)
    expected = range(distribution.num_replicas_in_sync)
    self.assertAllEqual(distributed_values, expected)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueAndRun(self, distribution):
    # [FUNCTION DOC]
    # Skips unless V2 behavior is enabled. Inside a compiled function, creates distributed values
    # holding the replica index, runs a squaring computation on every replica, gathers the outputs,
    # and asserts they equal the squares of 0 .. number of replicas minus one.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    @def_function.function
    def run():
      # [FUNCTION DOC]
      # HELPER: compiled body. Creates distributed values from an index-picking function, runs the
      # squaring computation on all replicas, and returns the gathered outputs.
      multiple_values = range(distribution.num_replicas_in_sync)
      def value_fn(ctx):
        # [FUNCTION DOC]
        # HELPER: value function; returns the range entry at the replica id in the sync group.
        return multiple_values[ctx.replica_id_in_sync_group]
      distributed_values = (
          distribution.experimental_distribute_values_from_function(value_fn))

      def computation(x):
        # [FUNCTION DOC]
        # HELPER: per-replica computation; returns the square of its input.
        return math_ops.square(x)

      outputs = ds_test_util.gather(
          distribution,
          distribution.run(computation, args=(distributed_values,)))
      return outputs

    results = run()

    expected = [i**2 for i in range(distribution.num_replicas_in_sync)]
    self.assertAllEqual(results, expected)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ] + strategy_combinations.multiworker_strategies,
          mode=["eager"]))
  def testMakeDistributedValueDefaultDevicePlacement(self, distribution):
    # [FUNCTION DOC]
    # Skips unless V2 behavior is enabled. Builds distributed values from a function returning a
    # constant without explicit placement, and asserts each component lives on the default device
    # (the device an identity op of a constant gets).
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; ignores the context and returns a floating-point constant one.
      del ctx
      return constant_op.constant(1.0)
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    default_device = array_ops.identity(constant_op.constant(1.0)).device
    for i in range(len(distribution.extended.worker_devices)):
      self.assertAllEqual(distributed_values._values[i].device, default_device)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ] + strategy_combinations.multiworker_strategies,
          mode=["eager"],
          op_type=[constant_op.constant, array_ops.identity]))
  def testMakeDistributedValueExplicitDevicePlacement(self, distribution,
                                                      op_type):
    # [FUNCTION DOC]
    # Parameterized over constant and identity ops. Skips unless V2 behavior is enabled. Builds
    # distributed values from a function that creates the value under an explicit worker device
    # chosen by replica id, and asserts each component is placed on the matching worker device.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    worker_devices = distribution.extended.worker_devices
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; selects a worker device by replica id modulo device count and
      # creates the floating-point value one there using the parameterized op type.
      # In multi client setup, worker_devices is just the devices on that
      # worker.
      worker_device_id = ctx.replica_id_in_sync_group % len(worker_devices)
      with ops.device(worker_devices[worker_device_id]):
        return op_type(1.0)

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    for i in range(len(distribution.extended.worker_devices)):
      self.assertAllEqual(distributed_values._values[i].device,
                          worker_devices[i])


class PerReplicaTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests misuse of per-replica values (using them where a regular tensor is expected).

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ] + strategy_combinations.multiworker_strategies,
          mode=["eager"]))
  def testUsePerReplicaInvalidContextGivesError(self, distribution):
    # [FUNCTION DOC]
    # Skips unless V2 behavior is enabled. Creates per-replica distributed values, then tries to
    # cast them to a float type outside a replica context and asserts a value error whose message
    # says the use is not inside a replica context.
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    multiple_values = range(distribution.num_replicas_in_sync)
    def value_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: value function; returns the range entry at the replica id in the sync group.
      return multiple_values[ctx.replica_id_in_sync_group]
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    with self.assertRaisesRegex(ValueError, "not inside a replica context"):
      math_ops.cast(distributed_values, dtypes.float32)


class PerWorkerResourceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests how often a dataset map function is traced, which matters for per-worker resources.

  @combinations.generate(
      combinations.combine(dataset_fn_as_tf_function=[True, False]))
  def testMapFnTracing(self, dataset_fn_as_tf_function):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the counter attribute is named "traced once" but counts total traces.
    # Parameterized on whether the dataset-building function is wrapped as a compiled function.
    # Builds five datasets (repeat, batch, map with a counting map function) and asserts the map
    # function was traced once when the builder is a compiled function (traced once and reused),
    # or five times (once per pipeline) when it is a plain function.
    # For a PerWorkerResource to correctly behave when used in dataset.map,
    # it has to be that the map_fn is not traced only once such that
    # PerWorkerResource.local_table can return the correct resource. This test
    # can detect the potential breakage of this behavior on TAP.
    self._traced_once = 0

    def map_fn(x):
      # [FUNCTION DOC]
      # HELPER: identity map function that increments the trace counter on the test instance each
      # time it is traced; returns its input.
      self._traced_once += 1
      return x

    def dataset_fn():
      # [FUNCTION DOC]
      # HELPER: builds a dataset of a repeated three-element tensor, batched by two (dropping the
      # remainder), mapped with the counting function. Returns the dataset.
      dataset = dataset_ops.DatasetV2.from_tensors([0, 1, 2]).repeat().batch(
          2, drop_remainder=True)
      dataset = dataset.map(map_fn)
      return dataset

    datasets = []
    number_of_input_pipelines = 5

    if dataset_fn_as_tf_function:
      dataset_fn = def_function.function(dataset_fn)
      expected_tracing_times = 1
    else:
      expected_tracing_times = number_of_input_pipelines

    for _ in range(number_of_input_pipelines):
      datasets.append(dataset_fn())

    self.assertEqual(self._traced_once, expected_tracing_times)


class DistributedDelegateTest(test.TestCase):
  # [CLASS DOC]
  # Tests the delegate wrapper that forwards attribute access and operators to its first component
  # value, in graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def testGetAttr(self):
    # [FUNCTION DOC]
    # Wraps two simple objects with different attribute values in a delegate; asserts reading the
    # attribute returns the first object's value and reading a missing attribute raises an
    # attribute error.
    class Foo(object):
      # [CLASS DOC]
      # HELPER: tiny test object holding a single attribute x.

      def __init__(self, x):
        # [FUNCTION DOC]
        # HELPER: stores x on the instance.
        self.x = x

    v = values_lib.DistributedDelegate((Foo(7), Foo(8)))
    self.assertEqual(7, v.x)
    with self.assertRaises(AttributeError):
      _ = v.y

  @test_util.run_in_graph_and_eager_modes
  def testOperatorOverride(self):
    # [FUNCTION DOC]
    # Wraps the numbers seven and eight in a delegate and asserts it acts like the first value (seven)
    # for arithmetic (including power with a modulus), comparison and bitwise operators with plain
    # numbers on either side, negation, inversion and absolute value; indexing raises a type error.
    v = values_lib.DistributedDelegate((7, 8))
    # v should act like int(7).
    self.assertEqual(8, v + 1)
    self.assertEqual(10, 3 + v)
    self.assertEqual(14, v + v)
    self.assertEqual(5, v - 2)
    self.assertEqual(6, 13 - v)
    self.assertEqual(0, v - v)
    self.assertEqual(14, v * 2)
    self.assertEqual(21, 3 * v)
    self.assertEqual(49, v * v)
    self.assertEqual(3.5, v / 2)
    self.assertEqual(1.5, 10.5 / v)
    self.assertEqual(3, v // 2)
    self.assertEqual(2, 15 // v)
    self.assertEqual(1, v % 2)
    self.assertEqual(2, 16 % v)
    # pylint: disable=g-generic-assert
    self.assertTrue(v < 12)
    self.assertTrue(v <= 12)
    self.assertFalse(v > 12)
    self.assertFalse(v >= 12)
    self.assertFalse(12 < v)
    self.assertFalse(12 <= v)
    self.assertTrue(12 > v)
    self.assertTrue(12 >= v)
    # pylint: enable=g-generic-assert
    self.assertEqual(3, v & 3)
    self.assertEqual(3, 11 & v)
    self.assertEqual(15, v | 8)
    self.assertEqual(23, 16 | v)
    self.assertEqual(4, v ^ 3)
    self.assertEqual(12, 11 ^ v)
    self.assertEqual(343, pow(v, 3))
    self.assertEqual(3, pow(v, 3, 10))
    self.assertEqual(128, pow(2, v))
    self.assertEqual(-7, -v)
    self.assertEqual(~7, ~v)
    self.assertEqual(7, abs(v))
    with self.assertRaises(TypeError):
      _ = v[2]

  @test_util.run_in_graph_and_eager_modes
  def testCopy(self):
    # [FUNCTION DOC]
    # Wraps two simple objects in a delegate; asserts that both a shallow copy and a deep copy of
    # the delegate expose the same attribute value as the original.

    class Foo(object):
      # [CLASS DOC]
      # HELPER: tiny test object holding a single attribute x.

      def __init__(self, x):
        # [FUNCTION DOC]
        # HELPER: stores x on the instance.
        self.x = x

    v = values_lib.DistributedDelegate((Foo(7), Foo(8)))
    v_shallow_copy = copy.copy(v)
    self.assertEqual(v.x, v_shallow_copy.x)
    v_deep_copy = copy.deepcopy(v)
    self.assertEqual(v.x, v_deep_copy.x)


_TPU_STRATEGIES = (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)


def _make_replica_local(method, strategy=None):
  # [FUNCTION DOC]
  # HELPER: test fixture factory for sync-on-read variables. Parameters: method is the variable
  # aggregation mode; strategy is optional (default none). Uses a GPU and a CPU device when no
  # strategy is given, otherwise the strategy's worker devices. Creates one resource variable per
  # device (names v and v/replica, initial values one and two) and wraps them in the TPU
  # sync-on-read class for TPU strategies, otherwise the regular sync-on-read class. Returns a
  # pair: the list of component variables and the wrapper.
  if strategy is None:
    devices = ("/device:GPU:0", "/device:CPU:0")
  else:
    devices = strategy.extended.worker_devices

  v = []
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))

  if (strategy is not None) and isinstance(strategy, _TPU_STRATEGIES):
    var_cls = tpu_values.TPUSyncOnReadVariable
  else:
    var_cls = values_lib.SyncOnReadVariable
  replica_local = var_cls(strategy, v, method)
  return v, replica_local


class DistributedVariableTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests sync-on-read (replica local) distributed variables: properties, passing to compiled
  # functions, tensor conversion, reads in different contexts, deterministic input names, and
  # save / restore round trips for mean and sum aggregation. Holds a class-level soft-placement
  # session configuration and helper methods for assigning, saving and restoring.

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent teardown, then resets the global eager context so tests do not leak state.
    super().tearDown()
    context._reset_context()

  def _assign_replica_local(self, v, new):
    # [FUNCTION DOC]
    # HELPER: assigns each value in the list new to the corresponding component variable in v,
    # under that variable's device, and evaluates the assignments. Returns nothing.
    for var, n in zip(v, new):
      with ops.device(var.device):
        self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    # [FUNCTION DOC]
    # HELPER: creates a saver for the single variable var and saves it with session sess to a
    # checkpoint prefix inside the test temporary directory. Returns a pair: the save path and the
    # saver.
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    # [FUNCTION DOC]
    # HELPER: saves var with session sess (via the save-and-return-saver helper) and returns only
    # the save path.
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes with soft placement. Skips in eager mode without a GPU. Creates
    # a sum-aggregated sync-on-read variable and asserts its constraint, name, dtype and shape equal
    # the first component's, and its aggregation is sum.
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")
    v, replica_local = _make_replica_local(
        variable_scope.VariableAggregation.SUM)

    self.assertEqual(v[0].constraint, replica_local.constraint)
    self.assertEqual(v[0].name, replica_local.name)
    self.assertEqual(v[0].dtype, replica_local.dtype)
    self.assertEqual(v[0].shape, replica_local.shape)
    self.assertEqual(variable_scope.VariableAggregation.SUM,
                     replica_local.aggregation)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu
          ],
          mode=["eager"]))
  def testCanPassToDefFun(self, distribution):
    # [FUNCTION DOC]
    # Creates a mean-aggregated sync-on-read variable of value one under the strategy scope, passes
    # it to a compiled function that adds one, and asserts the evaluated result is two.

    @def_function.function
    def add1(x):
      # [FUNCTION DOC]
      # HELPER: returns its input plus the floating-point number one.
      return x + 1.

    with distribution.scope():
      v = variables_lib.Variable(
          1.,
          aggregation=variables_lib.VariableAggregation.MEAN,
          synchronization=variables_lib.VariableSynchronization.ON_READ)

    self.assertEqual(2., self.evaluate(add1(v)))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testTensorConversion(self, distribution):
    # [FUNCTION DOC]
    # In graph mode, creates a sum-aggregated sync-on-read variable for the strategy and converts
    # it to a tensor both without and with the reference flag; asserts each conversion is a tensor
    # with the variable's data type (resource variables convert even when a reference is requested).
    with context.graph_mode():
      _, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)
      converted = ops.convert_to_tensor(replica_local, as_ref=False)
      self.assertIsInstance(converted, tensor.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

      converted = ops.convert_to_tensor(replica_local, as_ref=True)
      # Resources variable are converted to tensors as well when as_ref is True.
      self.assertIsInstance(converted, tensor.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

  @combinations.generate(combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ], mode=["eager"]))
  def testValueInCrossReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Creates an only-first-replica sync-on-read variable and, in cross-replica context, asserts
    # its value is a tensor equal to the first component's value.
    value_list, replica_local = _make_replica_local(
        variable_scope.VariableAggregation.ONLY_FIRST_REPLICA, distribution)

    self.assertIsInstance(replica_local.value(), tensor.Tensor)
    self.assertEqual(self.evaluate(replica_local.value()),
                     self.evaluate(value_list[0].value()))

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy_packed_var,
          ],
          mode=["eager"]))
  def testValueInDefaultReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Creates two sum-aggregated sync-on-read variables at zero; runs a replica function that adds
    # one to the first and two to the second on every replica; then asserts the sum of the two
    # variables outside the replica context is six (aggregated across the two replicas).
    with distribution.scope():
      v1 = variables_lib.Variable(
          0.0,
          aggregation=variables_lib.VariableAggregation.SUM,
          synchronization=variables_lib.VariableSynchronization.ON_READ)
      v2 = variables_lib.Variable(
          0.0,
          aggregation=variables_lib.VariableAggregation.SUM,
          synchronization=variables_lib.VariableSynchronization.ON_READ)

    @def_function.function
    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: replica function adding one to the first variable and two to the second.
      v1.assign_add(1.0)
      v2.assign_add(2.0)

    distribution.run(replica_fn)
    sum_v = v1 + v2
    self.assertEqual(sum_v, 6.0)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.tpu_strategy_packed_var,
          ],
          mode=["eager"]))
  def testValueInFunctionCrossReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Creates an on-write variable at zero (packed TPU strategy), assigns one inside a compiled
    # function, asserts the variable equals one, and asserts the function's graph text mentions the
    # composite device, showing the variable is passed as a packed composite input.
    with distribution.scope():
      v1 = variables_lib.Variable(
          0.0,
          aggregation=variables_lib.VariableAggregation.NONE,
          synchronization=variables_lib.VariableSynchronization.ON_WRITE)

    @def_function.function
    def assign_fn():
      # [FUNCTION DOC]
      # HELPER: compiled function assigning one to the variable.
      v1.assign(1.0)

    assign_fn()
    self.assertEqual(v1, 1.0)

    # Make sure the function graph has composite variable as inputs.
    graph_def = assign_fn.get_concrete_function().graph.as_graph_def()
    self.assertRegex(str(graph_def), "device:COMPOSITE:0")

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.tpu_strategy_packed_var,
          ],
          mode=["eager"]))
  def testReplicatedValueNameDeterministic(self, distribution):
    # [FUNCTION DOC]
    # Packed TPU strategy. Creates two named variables, runs a function that updates both through
    # the strategy inside a compiled function, and asserts the concrete function has two graph
    # inputs named after the variables (without a non-deterministic unique id).
    with distribution.scope():
      v1 = variables_lib.Variable(0.0, name="test_var_1")
      v2 = variables_lib.Variable(0.0, name="test_var_2")

    def fn():
      # [FUNCTION DOC]
      # HELPER: replica function adding one to the first variable, two to the second, returning
      # their sum.
      v1.assign_add(1.0)
      v2.assign_add(2.0)
      return v1 + v2

    @def_function.function
    def dist_run_fn():
      # [FUNCTION DOC]
      # HELPER: compiled function running the replica function through the strategy and returning
      # the result.
      a = distribution.run(fn)
      return a

    concrete_fn = dist_run_fn.get_concrete_function()
    inputs = concrete_fn.graph.inputs
    self.assertLen(inputs, 2)
    # Before cl/433948982, input name will include a non-deterministic uid,
    # e.g. "test_var_1_139726389910864/handle/inputs_0:0"
    self.assertEqual(inputs[0].name, "test_var_1/handle/inputs_0:0")
    self.assertEqual(inputs[1].name, "test_var_2/handle/inputs_0:0")

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreReplicaLocalSumOneGraph(self, distribution):
    # [FUNCTION DOC]
    # In one graph session: creates a sum-aggregated sync-on-read variable, sets components to three
    # and four, saves (total seven), changes components to five and six, restores, and asserts
    # both components become 3.5 (the saved total is split evenly).
    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 7.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])

        # Restores the saved value of 7. which gets divided equally
        # between the variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreReplicaLocalMeanOneGraph(self, distribution):
    # [FUNCTION DOC]
    # Skips in eager mode without a GPU. In one graph session: creates a mean-aggregated variable,
    # sets components to three and four, saves (mean 3.5), changes them to five and six, restores,
    # and asserts both components become 3.5.
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])

        # Restores the saved value of 3.5 to both variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _save_replica_local_mean(self, distribution):
    # [FUNCTION DOC]
    # HELPER: in a fresh graph session, creates a mean-aggregated sync-on-read variable for the
    # strategy, sets components to three and four, saves it under the strategy scope (value 3.5),
    # then changes components to five and six. Returns the checkpoint save path.
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])
    return save_path

  def _save_replica_local_sum(self, distribution):
    # [FUNCTION DOC]
    # HELPER: in a fresh graph session, creates a sum-aggregated sync-on-read variable for the
    # strategy, sets components to 1.5 and two, saves it under the strategy scope (total 3.5), then
    # changes components to five and six. Returns the checkpoint save path.
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [1.5, 2.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])
    return save_path

  def _save_normal(self):
    # [FUNCTION DOC]
    # HELPER: in a fresh graph session, creates an ordinary resource variable, sets it to 3.5, saves
    # it, then changes it to five. Returns the checkpoint save path.
    """Save variables without mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(3.5))

      # Saves the current value of var, 3.5.
      save_path = self._save(sess, var)

      # Change the values between save and restore.
      self.evaluate(var.assign(5.))
    return save_path

  def _restore_normal(self, save_path):
    # [FUNCTION DOC]
    # HELPER: in a fresh graph session, creates an ordinary resource variable, sets it to eight,
    # restores it from the given checkpoint path, and asserts it equals 3.5.
    """Restore to variables without mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3.5 to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3.5, self.evaluate(var))

  def _restore_replica_local_mean(self, save_path, distribution):
    # [FUNCTION DOC]
    # HELPER: in a fresh graph session, creates a mean-aggregated sync-on-read variable for the
    # strategy, sets components to seven and eight, restores from the given checkpoint path under
    # the strategy scope, and asserts both components equal 3.5.
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _restore_replica_local_sum(self, save_path, distribution):
    # [FUNCTION DOC]
    # HELPER: restore half of the sum save/restore tests.
    # KNOWN ISSUE: the inline comment in the body says the restored value is 3.5 to both variables,
    # but the asserted value is 1.75 each.
    # In a fresh graph session, creates a sum-aggregated sync-on-read variable for the
    # strategy, sets components to seven and eight, restores from the given checkpoint path under
    # the strategy scope, and asserts both components equal 1.75 (the saved total split in two).
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([1.75, 1.75], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalRestoreReplicaLocalMean(self, distribution):
    # [FUNCTION DOC]
    # Saves a mean-aggregated sync-on-read variable, then restores it into a new mean-aggregated
    # one (the restore helper asserts the expected values).
    save_path = self._save_replica_local_mean(distribution)
    self._restore_replica_local_mean(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalRestoreReplicaLocalSum(self, distribution):
    # [FUNCTION DOC]
    # Saves a sum-aggregated sync-on-read variable, then restores it into a new sum-aggregated
    # one (the restore helper asserts the expected values).
    save_path = self._save_replica_local_sum(distribution)
    self._restore_replica_local_sum(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalMeanRestoreNormal(self, distribution):
    # [FUNCTION DOC]
    # Saves a mean-aggregated sync-on-read variable, then restores the checkpoint into an ordinary
    # variable and checks it holds the saved mean 3.5.
    save_path = self._save_replica_local_mean(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalSumRestoreNormal(self, distribution):
    # [FUNCTION DOC]
    # Saves a sum-aggregated sync-on-read variable (total 3.5), then restores the checkpoint into
    # an ordinary variable and checks it holds 3.5.
    save_path = self._save_replica_local_sum(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreReplicaLocalMean(self, distribution):
    # [FUNCTION DOC]
    # Saves an ordinary variable (3.5), then restores it into a mean-aggregated sync-on-read
    # variable and checks both components become 3.5.
    save_path = self._save_normal()
    self._restore_replica_local_mean(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreReplicaLocalSum(self, distribution):
    # [FUNCTION DOC]
    # Saves an ordinary variable (3.5), then restores it into a sum-aggregated sync-on-read
    # variable and checks both components become 1.75 (value split between components).
    save_path = self._save_normal()
    self._restore_replica_local_sum(save_path, distribution)


if __name__ == "__main__":
  ds_test_util.main()
