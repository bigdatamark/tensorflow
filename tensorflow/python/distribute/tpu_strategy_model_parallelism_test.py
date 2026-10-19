# [MODULE DOC]
# Tests for the TPU distribution strategy with model parallelism (SPMD partitioning across several
# logical cores per replica). Covers the structure of SPMD variables (replicated variables, packed
# or unpacked), reading and updating them in graph and eager modes, creation from callables,
# checkpointing, summaries, outside compilation, and manual sharding ops. Requires a real TPU and
# defines command-line flags for the TPU name, GCP project and zone at import time. Non-SPMD
# logical device tests are skipped as unsupported.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPUStrategy."""

import os
import unittest

from absl.testing import parameterized

from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import summary_test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu_replication

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  # [FUNCTION DOC]
  # HELPER: test utility building a cluster resolver for the TPU tests.
  # Creates and returns a TPU cluster resolver from the tpu, zone and project command-line flags.
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy(enable_spmd=False):
  # [FUNCTION DOC]
  # HELPER: test utility creating the TPU strategy used by every test in this file.
  # Input: enable_spmd (default false) turns on SPMD XLA partitioning. Connects to the remote
  # cluster, initializes the TPU system, uses half of the cores as the number of replicas and
  # builds a device assignment of two cores per replica. Returns a pair of the new TPU strategy
  # and the number of replicas. Side effects: connects to the cluster and initializes the TPU.
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
  num_replicas = resolver.get_tpu_system_metadata().num_cores // 2
  device_assignment = device_assignment_lib.DeviceAssignment.build(
      topology, num_replicas=num_replicas, computation_shape=[1, 1, 1, 2])
  strategy = tpu_lib.TPUStrategyV2(
      resolver,
      experimental_device_assignment=device_assignment,
      experimental_spmd_xla_partitioning=enable_spmd)
  return strategy, num_replicas


class TPUStrategyModelParallelismTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the TPU strategy with model parallelism, mostly SPMD partitioning with two
  # cores per replica. Each test builds its own strategy through the module helper; needs a TPU.

  @parameterized.named_parameters([("packed", True), ("unpacked", False)])
  def test_spmd_variable_structure(self, enable_packing):
    # [FUNCTION DOC]
    # Parameterized by whether packed variables are enabled. Creates an on-read and an on-write
    # variable in an SPMD strategy scope, then checks that reading each inside a strategy run gives
    # the original tensor on every replica, and that each component is a replicated TPU variable.
    # When packing: the first component has a packed distributed variable, all share the same
    # packed instance, and its distributed variables match the underlying resources one to one.
    # Without packing, components have no packed variable.
    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)

    # pylint: disable=protected-access
    if enable_packing:
      self.assertTrue(strategy._enable_packed_variable_in_eager_mode,
                      "packed variables should be enabled by default")
    else:
      strategy._enable_packed_variable_in_eager_mode = False
    # pylint: enable=protected-access

    tensor = constant_op.constant([[0., 1.], [2., 3.]])

    # Test TPUMirroredVariable and TPUSyncOnReadVariable
    with strategy.scope():
      v = variables.Variable(
          tensor, name="v", synchronization=vs.VariableSynchronization.ON_READ)
      w = variables.Variable(
          tensor, name="w", synchronization=vs.VariableSynchronization.ON_WRITE)

    def test_read(x):
      # [FUNCTION DOC]
      # HELPER: nested check. Runs a compiled function reading the given variable on every
      # replica and asserts each local result is close to the expected tensor.
      @def_function.function
      def fn():
        # [FUNCTION DOC]
        # Returns the current value of the captured variable.
        return x.read_value()

      results = strategy.run(fn)
      results = strategy.experimental_local_results(results)

      for i in range(num_replicas):
        self.assertAllClose(results[i], tensor)

    def test_structure(values):
      # [FUNCTION DOC]
      # HELPER: nested check of the component structure of a variable's values (see the enclosing
      # test for the exact expectations about packed and unpacked variables).
      for i, value in enumerate(values):
        self.assertIsInstance(
            value, tpu_replicated_variable.TPUReplicatedVariable)
        packed_var = getattr(value, "_packed_var", None)
        if enable_packing:
          if i == 0:
            self.assertIsInstance(packed_var, packed.PackedDistributedVariable)
          else:
            self.assertIs(packed_var, values[0]._packed_var,  # pylint: disable=protected-access
                          "all vals should share the same packed var instance")
        else:
          self.assertIsNone(packed_var)

      if enable_packing:
        # pylint: disable=protected-access
        resources = sum((value._vars for value in values), [])
        dist_vars = packed_var._distributed_variables
        # pylint: enable=protected-access
        self.assertLen(resources, len(dist_vars))
        for dist_var, resource in zip(dist_vars, resources):
          self.assertIs(dist_var, resource)

    test_read(v)
    test_structure(v.values)
    test_read(w)
    test_structure(w.values)

  @unittest.skip("Non-SPMD model parallelism is no longer supported")
  def test_logical_device_assignment(self):
    # [FUNCTION DOC]
    # Skipped (non-SPMD model parallelism unsupported). Would create variables on logical devices
    # 0 and 1, check device placement of their local results, run a function that multiplies on
    # each logical device and expect the recorded op devices to be replicated core 0 and 1, and
    # expect the summed result to equal 30 times the number of replicas.
    strategy, num_replicas = get_tpu_strategy()
    with strategy.scope():
      v = variables.Variable(2.)
      with strategy.extended.experimental_logical_device(1):
        w = variables.Variable(3.)

    self.assertLen(strategy.experimental_local_results(v), num_replicas)
    self.assertLen(strategy.experimental_local_results(w), num_replicas)
    self.assertEqual("/job:localhost/replica:0/task:0/device:TPU:0",
                     strategy.experimental_local_results(v)[0].device)
    self.assertEqual("/job:localhost/replica:0/task:0/device:TPU:1",
                     strategy.experimental_local_results(w)[0].device)

    logical_devices = []

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # Multiplies x by the first variable on logical device 0, then by the second variable on
      # logical device 1, records the devices of the two intermediate results in the enclosing
      # list and returns the final product.
      replica_ctx = distribute_lib.get_replica_context()
      with replica_ctx.experimental_logical_device(0):
        y = v * x
      with replica_ctx.experimental_logical_device(1):
        z = w * y
      logical_devices.append((y.device, z.device))
      return z

    result = strategy.run(f, args=(5.,))

    self.assertEqual(
        [("/device:TPU_REPLICATED_CORE:0", "/device:TPU_REPLICATED_CORE:1")],
        logical_devices)

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(30. * num_replicas,
                       self.evaluate(strategy.reduce("SUM", result, axis=None)))

  @unittest.skip("Non-SPMD model parallelism is no longer supported")
  def test_paritioned_model_checkpointing(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name misspells "partitioned"; also skipped as unsupported.
    # Would build a two-variable model with variables on different logical devices, save a
    # checkpoint, change the weights, check the output (expected 20 times replicas), restore the
    # checkpoint, assert everything was consumed and matched, and expect 30 times replicas.

    class PartitionedModel(module.Module):
      # [CLASS DOC]
      # Module holding two variables placed on logical devices 0 and 1, whose call multiplies an
      # input by both in sequence on those devices. Requires an active strategy.

      def __init__(self, v, w):
        # [FUNCTION DOC]
        # Takes initial values for the two variables. Asserts a strategy is active and creates
        # the first variable on logical device 0 and the second on logical device 1.
        super(PartitionedModel, self).__init__()

        assert distribute_lib.has_strategy()
        strategy = distribute_lib.get_strategy()

        with strategy.extended.experimental_logical_device(0):
          self.v = variables.Variable(v)
        with strategy.extended.experimental_logical_device(1):
          self.w = variables.Variable(w)

      def __call__(self, x):
        # [FUNCTION DOC]
        # Returns x times the first variable (on logical device 0) times the second variable
        # (on logical device 1), using the current replica context.
        replica_ctx = distribute_lib.get_replica_context()
        with replica_ctx.experimental_logical_device(0):
          y = self.v * x
        with replica_ctx.experimental_logical_device(1):
          z = self.w * y
        return z

      def change_weights_op(self, v_new, w_new):
        # [FUNCTION DOC]
        # Returns a grouped operation assigning new values to both variables.
        return control_flow_ops.group(
            [self.v.assign(v_new), self.w.assign(w_new)])

    strategy, num_replicas = get_tpu_strategy()
    with strategy.scope():
      model = PartitionedModel(2., 3.)

    checkpoint_dir = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = util.Checkpoint(model=model)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      checkpoint.save(file_prefix=checkpoint_prefix)

      self.evaluate(model.change_weights_op(1., 4.))
      result = strategy.run(def_function.function(model), args=(5.0,))
      self.assertEqual(20. * num_replicas,
                       self.evaluate(strategy.reduce("SUM", result, axis=None)))

      status = checkpoint.restore(
          checkpoint_management.latest_checkpoint(checkpoint_dir))
      status.run_restore_ops(sess)  # must run restore op in non-eager mode.
      status.assert_consumed()
      status.assert_existing_objects_matched()
      result = strategy.run(def_function.function(model), args=(5.0,))
      self.assertEqual(30. * num_replicas,
                       self.evaluate(strategy.reduce("SUM", result, axis=None)))

  def test_spmd_cannot_assign_tensor_to_logical_device(self):
    # [FUNCTION DOC]
    # With an SPMD strategy, assigning a tensor to a logical device is expected to raise a value
    # error.
    strategy, _ = get_tpu_strategy(enable_spmd=True)
    x = constant_op.constant([0, 1])
    with self.assertRaises(ValueError):
      strategy.experimental_assign_to_logical_device(x, 0)

  def test_spmd_variable_created_from_callable(self):
    # [FUNCTION DOC]
    # Creates a variable in an SPMD strategy scope from a callable random-normal initializer and
    # asserts that all component variables of the first value hold identical contents.
    initilizer = lambda: random_ops.random_normal(shape=(16, 16))
    strategy, _ = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      w = variables.Variable(initilizer)
    value0 = w.values[0]
    for v in value0.variables:
      self.assertAllEqual(v, value0.variables[0])

  def test_spmd_variable_read(self):
    # [FUNCTION DOC]
    # Creates a random matrix variable in an SPMD scope, checks component and overall shapes, then
    # runs a matrix multiply of a random batch with it on all replicas and asserts the summed
    # result equals the expected product times the replica count, within loose tolerances.
    batch_size = 32
    num_feature_in = 16
    num_feature_out = 8

    x = random_ops.random_uniform((batch_size, num_feature_in),
                                  dtype=dtypes.float32)
    w_init = random_ops.random_uniform((num_feature_in, num_feature_out),
                                       dtype=dtypes.float32)

    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      w = variables.Variable(w_init, dtype=dtypes.float32)

    self.assertEqual(w.values[0].variables[0].shape.as_list(),
                     [num_feature_in, num_feature_out])

    self.assertEqual(w.shape.as_list(), [num_feature_in, num_feature_out])

    def step_fn(batch_features):
      # [FUNCTION DOC]
      # Returns the matrix product of the batch features with the captured variable.
      predict = math_ops.matmul(batch_features, w)
      return predict

    @def_function.function
    def train_fn(batch_features):
      # [FUNCTION DOC]
      # Compiled function running the step function on all replicas and returning its result.
      return strategy.run(step_fn, args=(batch_features,))

    result = train_fn(x)
    self.assertAllClose(
        strategy.reduce("SUM", result, axis=None),
        math_ops.matmul(x, w_init) * num_replicas,
        rtol=5e-03,
        atol=5e-03)

  def test_spmd_variable_read_init_scope(self):
    # [FUNCTION DOC]
    # Creates a ones matrix variable in an SPMD scope, reads it inside the init scope of a
    # compiled function run on all replicas, and asserts the mean-reduced result equals the
    # variable's value read directly.
    strategy, _ = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      v = variables.Variable(array_ops.ones((4, 4), dtype=dtypes.float32))

    @def_function.function
    def read_v():
      # [FUNCTION DOC]
      # Inside the init scope, returns the captured variable's current value.
      with ops.init_scope():
        return v.read_value()

    result = strategy.reduce("MEAN", strategy.run(read_v), axis=None)
    self.assertAllClose(result, v.read_value())

  def test_spmd_variable_update(self):
    # [FUNCTION DOC]
    # Creates a random variable in an SPMD scope, asserts it is a TPU mirrored variable that is
    # replicated or sharded to logical cores, then inside compiled strategy runs applies assign,
    # assign-sub and assign-add with random values and asserts each summed result equals the
    # expected tracked value times the replica count.
    batch_size = 1024
    num_feature_in = 256

    x = random_ops.random_uniform((batch_size, num_feature_in),
                                  dtype=dtypes.float32)
    w_init = random_ops.random_uniform((batch_size, num_feature_in),
                                       dtype=dtypes.float32)

    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      w = variables.Variable(w_init, dtype=dtypes.float32)

    self.assertIsInstance(w, tpu_values.TPUMirroredVariable)
    self.assertTrue(w._is_replicated_or_sharded_to_logical_cores())

    def make_strategy_run(fn):
      # [FUNCTION DOC]
      # HELPER: given an update function, returns a compiled function that runs it on all
      # replicas with a single value argument.

      def run(value):
        # [FUNCTION DOC]
        # Runs the captured update function on all replicas with the given value and returns the
        # per-replica result.
        return strategy.run(fn, args=(value,))

      return def_function.function(run)

    result = make_strategy_run(w.assign)(x)
    self.assertAllClose(
        strategy.reduce("SUM", result, axis=None), x * num_replicas)

    delta = random_ops.random_uniform((batch_size, num_feature_in),
                                      dtype=dtypes.float32)
    result = make_strategy_run(w.assign_sub)(delta)
    x -= delta
    self.assertAllClose(
        strategy.reduce("SUM", result, axis=None), x * num_replicas)

    delta = random_ops.random_uniform((batch_size, num_feature_in),
                                      dtype=dtypes.float32)
    result = make_strategy_run(w.assign_add)(delta)
    x += delta
    self.assertAllClose(
        strategy.reduce("SUM", result, axis=None), x * num_replicas)

  def test_spmd_variable_eager_update(self):
    # [FUNCTION DOC]
    # Creates a random variable in an SPMD scope and, eagerly, applies assign, assign-sub and
    # assign-add, asserting after each that the numpy value equals the expected value (the
    # assigned value, then minus a delta, then the assigned value plus a delta).
    batch_size = 32
    num_feature_in = 16

    x = random_ops.random_uniform((batch_size, num_feature_in),
                                  dtype=dtypes.float32)
    w_init = random_ops.random_uniform((batch_size, num_feature_in),
                                       dtype=dtypes.float32)

    strategy, _ = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      w = variables.Variable(w_init, dtype=dtypes.float32)

    w.assign(x)
    result = w.numpy()
    self.assertAllClose(result, x)

    x1 = random_ops.random_uniform((batch_size, num_feature_in),
                                   dtype=dtypes.float32)
    w.assign_sub(x1)
    result = w.numpy()
    self.assertAllClose(result, x - x1)

    x2 = random_ops.random_uniform((batch_size, num_feature_in),
                                   dtype=dtypes.float32)
    w.assign(x)
    w.assign_add(x2)
    result = w.numpy()
    self.assertAllClose(result, x + x2)

  def test_spmd_model_checkpointing(self):
    # [FUNCTION DOC]
    # Builds a linear model in an SPMD scope, saves a checkpoint, changes the weights and checks a
    # run (input split across two logical devices) matches the new weights times replicas, then
    # restores the checkpoint, asserts all objects were consumed and matched, and checks the
    # result matches the original weights again. Uses loose tolerances.

    class LinearModel(module.Module):
      # [CLASS DOC]
      # Minimal module holding one weight variable; calling it multiplies the input by it.

      def __init__(self, w):
        # [FUNCTION DOC]
        # Creates the weight variable from the given initial value.
        super(LinearModel, self).__init__()
        self.w = variables.Variable(w)

      def __call__(self, x):
        # [FUNCTION DOC]
        # Returns the matrix product of the input with the weight variable.
        return math_ops.matmul(x, self.w)

      def change_weights_op(self, w_new):
        # [FUNCTION DOC]
        # Returns the operation assigning a new value to the weight variable.
        return self.w.assign(w_new)

    batch_size = 32
    num_feature_in = 16
    num_feature_out = 8
    w1 = random_ops.random_uniform((num_feature_in, num_feature_out),
                                   dtype=dtypes.float32)
    w2 = random_ops.random_uniform((num_feature_in, num_feature_out),
                                   dtype=dtypes.float32)
    x = random_ops.random_uniform((batch_size, num_feature_in),
                                  dtype=dtypes.float32)

    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)
    with strategy.scope():
      model = LinearModel(w1)

    checkpoint_dir = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = util.Checkpoint(model=model)

    @def_function.function
    def step_fn(x):
      # [FUNCTION DOC]
      # Splits the input across logical devices along its second dimension in two parts and
      # returns the model applied to it.
      x = strategy.experimental_split_to_logical_devices(x, [1, 2])
      return model(x)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      checkpoint.save(file_prefix=checkpoint_prefix)

      self.evaluate(model.change_weights_op(w2))
      result = strategy.run(step_fn, args=(x,))
      self.assertAllClose(
          math_ops.matmul(x, w2) * num_replicas,
          self.evaluate(strategy.reduce("SUM", result, axis=None)),
          rtol=5e-3,
          atol=5e-3)

      status = checkpoint.restore(
          checkpoint_management.latest_checkpoint(checkpoint_dir))
      status.run_restore_ops(sess)  # must run restore op in non-eager mode.
      status.assert_consumed()
      status.assert_existing_objects_matched()
      result = strategy.run(step_fn, args=(x,))
      self.assertAllClose(
          math_ops.matmul(x, w1) * num_replicas,
          self.evaluate(strategy.reduce("SUM", result, axis=None)),
          rtol=5e-3,
          atol=5e-3)

  def test_spmd_with_summary(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the soft device placement setting is only restored at the end, so a failing
    # assertion leaves it changed for later tests.
    # Enables soft device placement, runs a function ten times on an SPMD strategy that writes a
    # scalar summary and increments a step variable, then asserts every component of the step
    # equals the expected count, the log directory has the expected number of events, and each
    # logged scalar equals its step times the multiplier. Restores the placement setting.
    original_device_placement = config.get_soft_device_placement()
    config.set_soft_device_placement(True)

    strategy, _ = get_tpu_strategy(enable_spmd=True)
    summary_dir = self.get_temp_dir()
    writer = summary_ops.create_file_writer_v2(summary_dir)
    const_multiple = 2
    num_iters = 10
    expected_event_count = num_iters + 1

    with strategy.scope():
      step = variables.Variable(1, dtype=dtypes.int64)

    @def_function.function
    def run():
      # [FUNCTION DOC]
      # Writes a scalar summary of the step times a constant at the current step, then increments
      # the step variable. Returns nothing.
      with writer.as_default():
        with summary_ops.record_if(True):
          summary_ops.scalar("result", step * const_multiple, step=step)
          step.assign_add(1)

    for _ in range(num_iters):
      strategy.run(run, args=())

    for val in step.values:
      for var in val.variables:
        self.assertAllEqual(expected_event_count, var)

    events = summary_test_util.events_from_logdir(summary_dir)
    self.assertLen(events, expected_event_count)

    # Event[0] is generic metadata and summary_ops data starts at event[1].
    for logged_step in range(1, expected_event_count):
      self.assertEqual(events[logged_step].summary.value[0].simple_value,
                       logged_step * const_multiple)

    config.set_soft_device_placement(original_device_placement)

  # Tests SPMD with outside compilation. One test case is for replicated
  # sharding of the input tensor and one case is for split sharding of the input
  # tensor.
  @parameterized.parameters([False, True])
  def test_spmd_with_outside_comp(self, split):
    # [FUNCTION DOC]
    # Parameterized by whether the input is split across logical devices. Runs a function adding
    # one on device, one on the host via outside compilation, and one more on device, and asserts
    # the summed result equals the input plus three times the replica count.
    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)

    def host_inc(x):
      # [FUNCTION DOC]
      # Host-side function returning its input plus one.
      return x + 1

    @def_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Optionally splits x, adds one, applies the host function through outside compilation,
      # adds one more, and returns the result.
      if split:
        x = strategy.experimental_split_to_logical_devices(x, [1, 2])
      y = x + 1
      z = tpu_replication.outside_compilation(host_inc, y)
      a = z + 1
      return a

    arg = constant_op.constant(0, shape=(2, 2), dtype=dtypes.int64)
    result = strategy.run(fn, args=(arg,))
    self.assertAllEqual(
        (arg + 3) * num_replicas,
        self.evaluate(strategy.reduce("SUM", result, axis=None)))

  # Tests auto_to_manual_spmd_partition and manual_to_auto_spmd_partition.
  # The internal versions of these ops are XlaSpmdFullToShardShape and
  # XlaSpmdShardToFullShape.
  def test_manual_sharding_ops(self):
    # [FUNCTION DOC]
    # Runs a function that splits the input, converts it to manual partitioning, adds one, and
    # converts back to automatic partitioning, then asserts the summed result equals the input
    # plus one times the replica count.
    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)

    @def_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Splits x across two logical devices, switches to manual sharding, adds one, switches
      # back to automatic sharding with the full shape, and returns the result.
      x_split = strategy.experimental_split_to_logical_devices(x, [1, 2])
      split_sharding = xla_sharding.get_op_sharding(x_split.op)
      x_manual = xla_sharding.auto_to_manual_spmd_partition(
          x_split, split_sharding
      )
      y_manual = x_manual + 1
      y_split = xla_sharding.manual_to_auto_spmd_partition(
          y_manual, split_sharding, (2, 2)
      )
      return y_split

    arg = constant_op.constant(0, shape=(2, 2), dtype=dtypes.int64)
    result = strategy.run(fn, args=(arg,))
    self.assertAllEqual(
        (arg + 1) * num_replicas,
        self.evaluate(strategy.reduce("SUM", result, axis=None)),
    )

  # Test mapping of a host-side function onto each shard.
  def test_spmd_with_map_outside_comp_inc(self):
    # [FUNCTION DOC]
    # Maps a host-side increment over each shard of a split input and asserts the summed result
    # equals the input plus one times the replica count.
    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)

    def host_inc(x):
      # [FUNCTION DOC]
      # Host-side function returning its input plus one.
      return x + 1

    @def_function.function
    def fn(a):
      # [FUNCTION DOC]
      # Splits a, maps the host function over each shard with outside compilation, splits again
      # and returns the result.
      b = strategy.experimental_split_to_logical_devices(a, [2, 1])
      c = tpu_replication.experimental_map_outside_compilation(host_inc, b)
      d = strategy.experimental_split_to_logical_devices(c, [2, 1])
      return d

    arg = constant_op.constant(
        [[0, 1], [2, 3]], shape=(2, 2), dtype=dtypes.int64
    )
    result = strategy.run(fn, args=(arg,))
    expected = (arg + 1) * num_replicas
    self.assertAllEqual(
        expected, self.evaluate(strategy.reduce("SUM", result, axis=None))
    )

  # Test mapping of an l2_normalize host-side function onto each shard. This is
  # not a point-wise function so the result is different from ordinary outside
  # compilation.
  def test_spmd_with_map_outside_comp_l2norm(self):
    # [FUNCTION DOC]
    # Maps a host-side L2 normalization over each shard (rows) of a split float input and asserts
    # the summed result equals the row-wise normalized input times the replica count.
    strategy, num_replicas = get_tpu_strategy(enable_spmd=True)

    def host_norm(x):
      # [FUNCTION DOC]
      # Host-side function returning the L2-normalized input over all elements of the shard.
      return nn.l2_normalize(x)

    @def_function.function
    def fn(a):
      # [FUNCTION DOC]
      # Splits a, maps the host normalization over each shard with outside compilation, splits
      # again and returns the result.
      b = strategy.experimental_split_to_logical_devices(a, [2, 1])
      c = tpu_replication.experimental_map_outside_compilation(host_norm, b)
      d = strategy.experimental_split_to_logical_devices(c, [2, 1])
      return d

    arg = constant_op.constant([[0, 1], [2, 3]], dtype=dtypes.float32)
    result = strategy.run(fn, args=(arg,))
    expected = nn.l2_normalize(arg, axis=1) * num_replicas
    self.assertAllEqual(
        expected, self.evaluate(strategy.reduce("SUM", result, axis=None))
    )


if __name__ == "__main__":
  test.main()
