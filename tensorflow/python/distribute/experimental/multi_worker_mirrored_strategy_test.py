# [MODULE DOC]
# Multi-client tests for the experimental DTensor-backed multi-worker mirrored strategy. Each
# client process reads its cluster configuration from the TF_CONFIG environment variable and the
# number of clients and local devices from command-line flags. Two test classes: strategy
# behavior (creation, environment parsing, variables, devices, run, gather, reduce, local
# results) and dataset and value distribution. A module-level configuration function sets up
# TF_CONFIG and logical devices per client, and the main entry launches the multi-client
# harness. The flags are defined by the imported multi-client test utilities.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Test for MultiWorkerMirroredStrategy backed by DTensor API."""

import json
import os

from absl import flags
from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python.tests import multi_client_test_util
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.distribute.experimental import multi_worker_mirrored_strategy as mwms
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as tf_test


class MultiWorkerMirroredStrategyTest(tf_test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the core behavior of the DTensor multi-worker mirrored strategy in a multi-client
  # setting. Fixture attributes: number of clients, number of local devices per client (both
  # from flags) and this client's id (from the task index in the cluster configuration).

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: calls the parent setUp, reads the client count and local device count
    # from the command-line flags and parses the TF_CONFIG environment variable (which must be
    # set, else a missing-key error occurs) to store this client's id as an integer.
    super().setUp()
    self.num_client = flags.FLAGS.num_clients
    self.num_local_devices = flags.FLAGS.num_local_devices

    tf_config = json.loads(os.environ['TF_CONFIG'])
    self.client_id = int(tf_config['task']['index'])

  def test_strategy_creation_with_default_cluster_resolver(self):
    # [FUNCTION DOC]
    # Creates the strategy with no arguments. Asserts the mesh exists, its global device ids
    # number clients times local devices, its local device ids number local devices, and the
    # stored cluster resolver is the environment-based (TF_CONFIG) resolver.
    strategy = mwms.MultiWorkerMirroredStrategy()
    mesh = strategy.mesh
    self.assertIsNotNone(mesh)
    self.assertLen(mesh.global_device_ids(),
                   self.num_client * self.num_local_devices)
    self.assertLen(mesh.local_device_ids(), self.num_local_devices)
    self.assertIsInstance(strategy._cluster_resolver,
                          tfconfig_cluster_resolver.TFConfigClusterResolver)

  def test_invalid_init_arguments(self):
    # [FUNCTION DOC]
    # Passes both a placeholder mesh object and a cluster resolver to the strategy constructor
    # and asserts a value error saying they cannot be provided together.
    mesh = object()
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()

    with self.assertRaisesRegex(
        ValueError,
        'Mesh and cluster_resolver can not be provided at the same time'):
      mwms.MultiWorkerMirroredStrategy(
          mesh=mesh,
          cluster_resolver=cluster_resolver)

  def test_parse_dtensor_env_var_from_cluster_resolver(self):
    # [FUNCTION DOC]
    # Calls the private environment-variable parser with the environment-based resolver and
    # compares against TF_CONFIG. Asserts four entries: the jobs equal the comma-joined worker
    # addresses, number of clients equals the client flag, client id equals the task index (as
    # text), and job name is worker.
    cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()

    dtensor_env_vars = mwms._parse_dtensor_env_var_from_cluster_resolver(
        cluster_resolver)

    tf_config = json.loads(os.environ['TF_CONFIG'])
    worker_jobs = ','.join(tf_config['cluster']['worker'])
    client_id = tf_config['task']['index']

    self.assertLen(dtensor_env_vars, 4)
    self.assertEqual(dtensor_env_vars['DTENSOR_JOBS'], worker_jobs)
    self.assertEqual(dtensor_env_vars['DTENSOR_NUM_CLIENTS'],
                     str(self.num_client))
    self.assertEqual(dtensor_env_vars['DTENSOR_CLIENT_ID'], client_id)
    self.assertEqual(dtensor_env_vars['DTENSOR_JOB_NAME'], 'worker')

  @parameterized.named_parameters([
      ('py_floats', lambda: [1.0, 2.0], True),
      ('np_floats', lambda: np.array([1.0, 2.0]), True),
      ('tf_const', lambda: constant_op.constant([1.0, 2.0]), True),
      ('py_floats_callable', lambda: [1.0, 2.0], False),
      ('np_floats_callable', lambda: np.array([1.0, 2.0]), False),
      ('tf_const_callable', lambda: constant_op.constant([1.0, 2.0]), False),
  ])
  def test_variable_creation(self, init_value, convert_callable):
    # [FUNCTION DOC]
    # Parameterized over python floats, numpy arrays and tensors, given directly (factory called
    # first) or as initializer callables. Creates a variable in the strategy scope and asserts it
    # is a DTensor variable with the fully replicated rank-one layout on the strategy mesh.
    if convert_callable:
      init_value = init_value()
    strategy = mwms.MultiWorkerMirroredStrategy()
    with strategy.scope():
      v = variables.Variable(init_value)

    self.assertIsInstance(v, d_variable.DVariable)
    self.assertIsNotNone(v.layout)
    self.assertEqual(v.layout, layout.Layout.replicated(strategy.mesh, rank=1))

  def test_strategy_extension(self):
    # [FUNCTION DOC]
    # Asserts the strategy's extended object is an instance of the standard strategy extended
    # base type.
    strategy = mwms.MultiWorkerMirroredStrategy()
    self.assertIsInstance(strategy.extended, distribute_lib.StrategyExtendedV2)

  def test_num_replica_in_sync(self):
    # [FUNCTION DOC]
    # Asserts the replicas in sync equals number of clients times local devices per client.
    strategy = mwms.MultiWorkerMirroredStrategy()
    self.assertEqual(strategy.num_replicas_in_sync,
                     self.num_client * self.num_local_devices)

  def test_mesh(self):
    # [FUNCTION DOC]
    # Asserts the strategy's mesh property is not null.
    strategy = mwms.MultiWorkerMirroredStrategy()
    self.assertIsNotNone(strategy.mesh)

  def test_worker_devices(self):
    # [FUNCTION DOC]
    # Asserts the worker devices have one entry per local device and equal the tuple of the
    # mesh's local devices.
    strategy = mwms.MultiWorkerMirroredStrategy()
    worker_devices = strategy.extended.worker_devices
    self.assertLen(worker_devices, self.num_local_devices)
    self.assertEqual(worker_devices, tuple(strategy.mesh.local_devices()))

  def test_parameter_devices(self):
    # [FUNCTION DOC]
    # Asserts the parameter devices have one entry per local device and equal the tuple of the
    # mesh's local devices.
    strategy = mwms.MultiWorkerMirroredStrategy()
    parameter_devices = strategy.extended.parameter_devices
    self.assertLen(parameter_devices, self.num_local_devices)
    self.assertEqual(parameter_devices, tuple(strategy.mesh.local_devices()))

  def test_variable_created_in_scope(self):
    # [FUNCTION DOC]
    # Creates variables inside the first strategy's scope, outside any scope, and inside a second
    # strategy's scope. Asserts each strategy reports true only for the variable created in its
    # own scope.
    strategy1 = mwms.MultiWorkerMirroredStrategy()
    with strategy1.scope():
      v1 = variables.Variable(constant_op.constant([1.0, 2.0]))

    v2 = variables.Variable(constant_op.constant([1.0, 2.0]))

    strategy2 = mwms.MultiWorkerMirroredStrategy()
    with strategy2.scope():
      v3 = variables.Variable(constant_op.constant([1.0, 2.0]))

    self.assertTrue(strategy1.extended.variable_created_in_scope(v1))
    self.assertFalse(strategy1.extended.variable_created_in_scope(v2))
    self.assertFalse(strategy1.extended.variable_created_in_scope(v3))
    self.assertTrue(strategy2.extended.variable_created_in_scope(v3))

  def test_colocate_vars_with(self):
    # [FUNCTION DOC]
    # Creates a variable, then another inside a colocation context of the first, in the strategy
    # scope. Asserts both have the same layout.
    strategy = mwms.MultiWorkerMirroredStrategy()
    with strategy.scope():
      v1 = variables.Variable(constant_op.constant([1.0, 2.0]))
      with strategy.extended.colocate_vars_with(v1):
        v2 = variables.Variable(constant_op.constant([2.0, 3.0]))

    # We assert the layout for the variable, and make sure they are same.
    self.assertEqual(v1.layout, v2.layout)

  def test_in_multi_worker_mode(self):
    # [FUNCTION DOC]
    # Asserts the extended object reports it is in multi-worker mode.
    strategy = mwms.MultiWorkerMirroredStrategy()
    self.assertTrue(strategy.extended._in_multi_worker_mode())

  def test_run_with_distribute_value_input(self):
    # [FUNCTION DOC]
    # Distributes the replica id as a per-replica value, runs a doubling function, and asserts a
    # DTensor distributed value with one entry per local device, where each local entry equals
    # twice the global replica id (client id times local devices plus local index), as a
    # one-element tensor.
    strategy = mwms.MultiWorkerMirroredStrategy()

    def value_fn(value_context):
      # [FUNCTION DOC]
      # Value function: returns the replica id within the sync group from the value context.
      return value_context.replica_id_in_sync_group
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))

    @def_function.function
    def replica_fn(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: returns the input multiplied by two.
      return inputs * 2

    result = strategy.run(replica_fn, args=(distributed_values,))
    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertLen(result.values, self.num_local_devices)
    # Note that the scalar value from
    # experimental_distribute_values_from_function will be up rank to 1D since
    # batched shared dtensor need at least be 1D.

    for i in range(self.num_local_devices):
      self.assertAllClose(
          result.values[i],
          constant_op.constant(
              [(self.client_id * self.num_local_devices + i) * 2]))

  def test_nested_structure_output(self):
    # [FUNCTION DOC]
    # Distributes a per-replica mapping built from the total replica count (a: that count as a
    # number; b: two-element tensor of count plus one and plus two), runs a function doubling every
    # entry, and asserts the output mapping has two keys of DTensor distributed values, with one
    # local value per device equal to the doubled expected numbers.
    strategy = mwms.MultiWorkerMirroredStrategy()
    def value_fn(ctx):
      # [FUNCTION DOC]
      # Value function: returns a mapping with the replica count as a number and a two-element
      # tensor holding the count plus one and plus two.
      value = float(ctx.num_replicas_in_sync)
      return {'a': value,
              'b': constant_op.constant([value + 1.0, value + 2.0])}
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))

    @def_function.function
    def replica_fn(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: returns a new mapping with every entry of the input mapping
      # multiplied by two.
      result = {}
      for key in inputs:
        result[key] = inputs[key] * 2.0
      return result

    result = strategy.run(replica_fn, args=(distributed_values,))
    self.assertLen(result.keys(), 2)
    self.assertIsInstance(result['a'], dtensor_util.DTensorDistributedValue)
    self.assertLen(result['a'].values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(
          result['a'].values[i],
          constant_op.constant([strategy.num_replicas_in_sync * 2.0]))

    self.assertIsInstance(result['b'], dtensor_util.DTensorDistributedValue)
    self.assertLen(result['b'].values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(
          result['b'].values[i],
          constant_op.constant([(strategy.num_replicas_in_sync + 1.0) * 2.0,
                                (strategy.num_replicas_in_sync + 2.0) * 2.0]))

  def test_inputs_with_dtensor_distribute_values(self):
    # [FUNCTION DOC]
    # Distributes a scalar three to every replica, runs a doubling function, then runs an
    # add-one function on that result. Asserts DTensor distributed values with one local entry per
    # device equal to [6.0] after the first run and [7.0] after the second.

    @def_function.function
    def replica_fn_1(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: returns the input multiplied by two.
      return inputs * 2.0

    @def_function.function
    def replica_fn_2(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: returns the input plus one.
      return inputs + 1.0

    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant(3.0)
    d_tensor_input = strategy.experimental_distribute_values_from_function(
        lambda _: tensor_input)

    result_1 = strategy.run(replica_fn_1, args=(d_tensor_input,))
    self.assertIsInstance(result_1, dtensor_util.DTensorDistributedValue)
    self.assertLen(result_1.values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(result_1.values[i], constant_op.constant([6.0]))

    result_2 = strategy.run(replica_fn_2, args=(result_1,))
    self.assertIsInstance(result_2, dtensor_util.DTensorDistributedValue)
    self.assertLen(result_2.values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(result_2.values[i], constant_op.constant([7.0]))

  def test_get_replica_context(self):
    # [FUNCTION DOC]
    # Checks replica context behavior: a default context exists outside scopes, none inside the
    # strategy scope outside run, and inside the replica function it is the DTensor replica
    # context. The function multiplies its input (integer three) by the replica count, so each
    # local result equals three times the total replicas in sync.
    strategy = mwms.MultiWorkerMirroredStrategy()

    tensor_input = constant_op.constant(3)
    d_tensor_input = strategy.experimental_distribute_values_from_function(
        lambda _: tensor_input)

    @def_function.function
    def replica_fn(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: asserts the current replica context is a DTensor replica
      # context and returns the input times the number of replicas in sync.
      replica_context = distribute_lib.get_replica_context()
      self.assertIsInstance(replica_context, dtensor_util.DTensorReplicaContext)
      return inputs * replica_context.num_replicas_in_sync

    # Default replica context
    self.assertIsNotNone(distribute_lib.get_replica_context())
    with strategy.scope():
      self.assertIsNone(distribute_lib.get_replica_context())

      result = strategy.run(replica_fn, args=(d_tensor_input,))

    self.assertLen(result.values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertAllClose(
          result.values[i],
          constant_op.constant([3 * strategy.num_replicas_in_sync]))

  def test_gather_non_dtensor_value(self):
    # [FUNCTION DOC]
    # Gathers a plain scalar tensor along axis zero and asserts it is returned unchanged.
    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant(3.0)

    result = strategy.gather(tensor_input, axis=0)
    self.assertAllClose(result, tensor_input)

  def test_gather_dtensor_value(self):
    # [FUNCTION DOC]
    # Distributes per-replica row tensors of length stride (clients times local devices), each a
    # range starting at replica id times stride. Gathers along axis zero and asserts this
    # client's local rows (shape local devices by stride) hold the consecutive numbers for its
    # replicas; then gathers along axis one and asserts a single row of local devices times
    # stride consecutive numbers.
    strategy = mwms.MultiWorkerMirroredStrategy()
    stride = self.num_client * self.num_local_devices

    def value_fn(value_context):
      # [FUNCTION DOC]
      # Value function: returns a one-by-stride tensor holding stride consecutive integers
      # starting at the replica id times stride.
      start = value_context.replica_id_in_sync_group * stride
      return array_ops.reshape(
          math_ops.range(start=start, limit=start + stride), shape=(1, stride)
      )
    distribute_result = strategy.experimental_distribute_values_from_function(
        value_fn
    )
    # distribute_result is a DTensorDistributedValue.
    # The shape of the global tensor is [stride, stride],
    # and each worker gets [stride/2, stride].
    result = strategy.gather(distribute_result, axis=0)

    start = stride * self.num_local_devices * self.client_id
    end = start + stride * self.num_local_devices
    self.assertEqual(result.shape, [self.num_local_devices, stride])
    self.assertAllClose(
        result,
        array_ops.reshape(
            math_ops.range(start=start, limit=end),
            shape=(self.num_local_devices, -1),
        ),
    )

    result = strategy.gather(distribute_result, axis=1)
    self.assertEqual(result.shape, [1, self.num_local_devices * stride])
    self.assertAllClose(
        result,
        array_ops.reshape(
            math_ops.range(start=start, limit=end), shape=(1, -1)
        ),
    )

  def test_reduce_mean_non_dtensor_value(self):
    # [FUNCTION DOC]
    # Reduces (mean, axis zero) a plain tensor; asserts a value error saying the input types are
    # unsupported (the message names the single-worker mirrored strategy).
    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'
    ):
      strategy.reduce(reduce_util.ReduceOp.MEAN, tensor_input, axis=0)

  def test_reduce_sum_non_dtensor_value(self):
    # [FUNCTION DOC]
    # Reduces (sum, axis zero) a plain tensor; asserts a value error saying the input types are
    # unsupported (the message names the single-worker mirrored strategy).
    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'
    ):
      strategy.reduce(reduce_util.ReduceOp.SUM, tensor_input, axis=0)

  def test_reduce_mean_distribute_value(self):
    # [FUNCTION DOC]
    # Distributes a two by two matrix per replica (base matrix plus four times the replica id)
    # and reduces with mean over no axis, axis zero and axis one. The reduction is global across
    # all workers, so every worker sees the same values: the base matrix (or its row or column
    # means) plus an offset derived from the total replica count.
    strategy = mwms.MultiWorkerMirroredStrategy()

    @def_function.function
    def value_fn(value_context):
      # [FUNCTION DOC]
      # Compiled value function: returns the base two by two matrix plus four times the replica
      # id.
      i = value_context.replica_id_in_sync_group
      return constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + i * 4.0

    distribute_value = strategy.experimental_distribute_values_from_function(
        value_fn
    )
    # replica 0 has [[0.0, 1.0],[2.0, 3.0]] and
    # replica 1 has [[4.0, 5.0],[6.0, 7.0]]. Each worker has 4 replicas.
    # For worker 2, it has replica 4 ~ 7.

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=None
    )
    # This should be a global reduce and each worker should have same value.
    # [[14.0, 15.0],[16.0, 17.0]]
    final = (self.num_local_devices * self.num_client - 1) * 2.0
    self.assertAllClose(
        result, constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + final
    )

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=0
    )
    # [15.0, 16.0]
    self.assertAllClose(result, constant_op.constant([0.0, 1.0]) + final + 1)

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=1
    )

    self.assertAllClose(result, constant_op.constant([0.5, 2.5]) + final)

  def test_reduce_sum_distribute_value(self):
    # [FUNCTION DOC]
    # Same distributed matrices as the mean test, reduced with sum over no axis, axis zero and
    # axis one. The expected constants ([[112,120],[128,136]], [240,256], [232,264]) correspond to
    # a total of eight replicas in the global reduction, as the comments in the test describe.
    strategy = mwms.MultiWorkerMirroredStrategy()

    @def_function.function
    def value_fn(value_context):
      # [FUNCTION DOC]
      # Compiled value function: returns the base two by two matrix plus four times the replica
      # id.
      i = value_context.replica_id_in_sync_group
      return constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + i * 4.0

    distribute_value = strategy.experimental_distribute_values_from_function(
        value_fn
    )
    # replica 0 has [[0.0, 1.0],[2.0, 3.0]] and
    # replica 1 has [[4.0, 5.0],[6.0, 7.0]]. Each worker has 4 replicas.
    # For worker 2, it has replica 4 ~ 7.
    # The shape of the global tensor is [16, 2], and each worker gets [8, 2].

    result = strategy.reduce(
        reduce_util.ReduceOp.SUM, distribute_value, axis=None
    )
    self.assertAllClose(result, [[112.0, 120.0], [128.0, 136.0]])

    result = strategy.reduce(reduce_util.ReduceOp.SUM, distribute_value, axis=0)
    self.assertAllClose(result, constant_op.constant([240.0, 256.0]))

    result = strategy.reduce(reduce_util.ReduceOp.SUM, distribute_value, axis=1)
    self.assertAllClose(result, constant_op.constant([232.0, 264.0]))

  def test_reduce_mean_mirrored_value(self):
    # [FUNCTION DOC]
    # Creates a replicated two by two variable in scope and reduces it with mean over no axis,
    # axis zero and axis one; asserts [[1,2],[3,4]], [2,3] and [1.5,3.5].
    strategy = mwms.MultiWorkerMirroredStrategy()

    with strategy.scope():
      v = variables.Variable(constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    self.assertIsInstance(v, d_variable.DVariable)

    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=None)
    self.assertAllClose(result, constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=0)
    self.assertAllClose(result, constant_op.constant([2.0, 3.0]))
    result = strategy.reduce(reduce_util.ReduceOp.MEAN, v, axis=1)
    self.assertAllClose(result, constant_op.constant([1.5, 3.5]))

  def test_reduce_sum_mirrored_value(self):
    # [FUNCTION DOC]
    # Creates a replicated two by two variable in scope and reduces it with sum over no axis,
    # axis zero and axis one; asserts [[1,2],[3,4]], [4,6] and [3,7].
    strategy = mwms.MultiWorkerMirroredStrategy()

    with strategy.scope():
      v = variables.Variable(constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    self.assertIsInstance(v, d_variable.DVariable)

    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=None)
    self.assertAllClose(result, constant_op.constant([[1.0, 2.0], [3.0, 4.0]]))
    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=0)
    self.assertAllClose(result, constant_op.constant([4.0, 6.0]))
    result = strategy.reduce(reduce_util.ReduceOp.SUM, v, axis=1)
    self.assertAllClose(result, constant_op.constant([3.0, 7.0]))

  def test_reduce_value_device(self):
    # [FUNCTION DOC]
    # Reduces (mean, no axis) a plain tensor and asserts the result's device string mentions
    # the first CPU device.
    strategy = mwms.MultiWorkerMirroredStrategy()
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    result = strategy.reduce(reduce_util.ReduceOp.MEAN, tensor_input, axis=None)
    self.assertIn('CPU:0', result.device)

  def test_experimental_local_results(self):
    # [FUNCTION DOC]
    # Runs an argument-less function returning [3.0] and fetches local results; asserts a tuple
    # with one entry per local device, each equal to [3.0].
    @def_function.function
    def replica_fn():
      # [FUNCTION DOC]
      # Compiled replica function with no inputs: returns a one-element tensor holding three.
      return constant_op.constant([3.0])

    strategy = mwms.MultiWorkerMirroredStrategy()
    result = strategy.run(replica_fn)
    local_result = strategy.experimental_local_results(result)

    self.assertIsInstance(local_result, tuple)
    self.assertLen(local_result, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertEqual(local_result[i], constant_op.constant([3.0]))

  def test_experimental_local_results_with_inputs(self):
    # [FUNCTION DOC]
    # Distributes a per-replica mapping from the replica count, runs a function doubling every
    # entry, and fetches local results. Asserts a tuple with one mapping per local device, each
    # equal to a: [16.0], b: [18.0, 20.0] (which assumes a total of eight replicas).
    strategy = mwms.MultiWorkerMirroredStrategy()

    def value_fn(ctx):
      # [FUNCTION DOC]
      # Value function: returns a mapping with the replica count as a number and a two-element
      # tensor holding the count plus one and plus two.
      value = float(ctx.num_replicas_in_sync)
      return {'a': value, 'b': constant_op.constant([value + 1.0, value + 2.0])}

    distributed_values = strategy.experimental_distribute_values_from_function(
        value_fn
    )

    @def_function.function
    def replica_fn(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: returns a new mapping with every entry of the input mapping
      # multiplied by two.
      result = {}
      for key in inputs:
        result[key] = inputs[key] * 2.0
      return result

    result = strategy.run(replica_fn, args=(distributed_values,))
    local_result = strategy.experimental_local_results(result)
    self.assertIsInstance(local_result, tuple)
    self.assertLen(local_result, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertDictEqual(
          local_result[i],
          {
              'a': constant_op.constant([16.0]),
              'b': constant_op.constant([18.0, 20.0]),
          },
      )


class StrategyDatasetTest(tf_test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests dataset and value distribution through the DTensor multi-worker strategy. Fixture:
  # client count, local device count, client id, seeded random image and label tensors and a
  # repeating dataset of image-label pairs.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: calls the parent setUp, reads client and device counts from flags,
    # parses the TF_CONFIG environment variable for this client's id, creates seeded random
    # image (8x8x3) and label (1) tensors and a repeating dataset from them.
    super().setUp()
    self.num_client = flags.FLAGS.num_clients
    self.num_local_devices = flags.FLAGS.num_local_devices

    tf_config = json.loads(os.environ['TF_CONFIG'])
    self.client_id = int(tf_config['task']['index'])

    self.images = stateless_random_ops.stateless_random_uniform(
        [8, 8, 3], seed=(1, 2), minval=0, maxval=255)
    self.labels = stateless_random_ops.stateless_random_uniform(
        [1], seed=(1, 2), minval=0, maxval=10)

    self.dataset = dataset_ops.Dataset.from_tensors(
        (self.images, self.labels)).repeat()

  def test_create_batched_dataset(self):
    # [FUNCTION DOC]
    # Distributes the fixture dataset batched by two per device times clients. Asserts the first
    # element has global batch shapes for images and labels, and that unpacking yields only this
    # client's local shards (one per local device).
    strategy = mwms.MultiWorkerMirroredStrategy()
    global_batch_size = self.num_client * self.num_local_devices * 2
    dataset = self.dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    element = next(iter(distributed_dataset))
    batched_image, batched_label = element
    self.assertEqual(batched_image.shape, [global_batch_size, 8, 8, 3])
    self.assertEqual(batched_label.shape, [global_batch_size, 1])

    # After unpack, it should only get the local shards.
    self.assertLen(d_api.unpack(batched_image), self.num_local_devices)
    self.assertLen(d_api.unpack(batched_label), self.num_local_devices)

  def test_uneven_batched_dataset(self):
    # [FUNCTION DOC]
    # Tries to distribute a repeating generator dataset of variable-length lists; asserts a
    # value error saying a static batch size is required.
    elements = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
    dataset = dataset_ops.Dataset.from_generator(
        lambda: elements, dtypes.int64).repeat()
    strategy = mwms.MultiWorkerMirroredStrategy()
    with self.assertRaisesRegex(ValueError, 'requires a static batch size'):
      strategy.experimental_distribute_dataset(dataset)

  def test_deprecated_strategy_methods(self):
    # [FUNCTION DOC]
    # Asserts the legacy dataset-iterator and input-function-iterator methods each raise a
    # not-implemented error saying they are only available in the V1 API.
    strategy = mwms.MultiWorkerMirroredStrategy()
    with self.assertRaisesRegex(
        NotImplementedError, 'only available in the V1 API'):
      strategy.make_dataset_iterator(self.dataset)

    with self.assertRaisesRegex(
        NotImplementedError, 'only available in the V1 API'):
      strategy.make_input_fn_iterator(lambda _: self.dataset)

  def test_distribute_dataset_from_fn(self):
    # [FUNCTION DOC]
    # Distributes a dataset built by a function giving per-replica batches of four. The global
    # batch is four times the total replicas. Asserts the element spec has the global batch
    # shapes (float32) and equals the iterator's spec, the first element has global batch shapes,
    # and unpacking gives one local shard per device each with batch size four.
    strategy = mwms.MultiWorkerMirroredStrategy()
    local_batch_size = 4
    global_batch_size = local_batch_size * strategy.num_replicas_in_sync

    def dataset_fn(option):
      # [FUNCTION DOC]
      # Dataset function: ignores its input-context argument and returns the repeating fixture
      # dataset batched by the local batch size (dropping remainders) with prefetch two.
      del option
      return dataset_ops.Dataset.from_tensors(
          (self.images, self.labels)).repeat().batch(
              local_batch_size, drop_remainder=True).prefetch(2)

    distributed_dataset = strategy.distribute_datasets_from_function(
        dataset_fn, None)
    iterator = iter(distributed_dataset)

    self.assertEqual(distributed_dataset.element_spec,
                     (tensor_spec.TensorSpec(shape=(global_batch_size, 8, 8, 3),
                                             dtype=dtypes.float32, name=None),
                      tensor_spec.TensorSpec(shape=(global_batch_size, 1),
                                             dtype=dtypes.float32, name=None)))
    self.assertEqual(distributed_dataset.element_spec, iterator.element_spec)

    batched_image, batched_label = next(iterator)
    self.assertEqual(batched_image.shape, [global_batch_size, 8, 8, 3])
    self.assertEqual(batched_label.shape, [global_batch_size, 1])

    # After unpack, it should only get the local shards.
    unpacked_images = d_api.unpack(batched_image)
    self.assertLen(unpacked_images, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertEqual(unpacked_images[i].shape, [local_batch_size, 8, 8, 3])

  def test_distribute_values_from_function(self):
    # [FUNCTION DOC]
    # Distributes per-replica scalars taken from a nine-element numpy array by replica id.
    # Asserts the result's layout is batch-sharded rank one on the strategy mesh, that unpacking
    # gives one value per local device, and that each equals one plus the global replica index
    # (client id times local devices plus local index).
    array_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    def value_fn(ctx):
      # [FUNCTION DOC]
      # Value function: returns the numpy element selected by the replica id.
      return array_value[ctx.replica_id_in_sync_group]
    strategy = mwms.MultiWorkerMirroredStrategy()
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))

    self.assertEqual(d_api.fetch_layout(distributed_values),
                     layout.Layout.batch_sharded(
                         strategy.mesh, batch_dim='batch', rank=1))
    unpacked_value = d_api.unpack(distributed_values)
    self.assertLen(unpacked_value, self.num_local_devices)
    start = 1.0 + self.num_local_devices * self.client_id
    for i in range(self.num_local_devices):
      self.assertEqual(unpacked_value[i], start + i)

  def test_distribute_dataset_in_tf_function(self):
    # [FUNCTION DOC]
    # Distributes a dataset with global batch four times the replicas, then runs a compiled step
    # that takes the next element and returns the images. Asserts a DTensor distributed value with
    # one local entry per device, each of shape local batch four by 8x8x3.
    strategy = mwms.MultiWorkerMirroredStrategy()
    local_batch_size = 4
    global_batch_size = local_batch_size * strategy.num_replicas_in_sync
    dataset = self.dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)

    @def_function.function
    def step_fn(iterator):
      # [FUNCTION DOC]
      # Compiled step function: takes the next (image, label) pair from the iterator, discards the
      # labels and returns the images.
      images, labels = next(iterator)
      del labels
      return images

    result = strategy.run(step_fn, args=(iter(distributed_dataset),))
    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertLen(result.values, self.num_local_devices)
    for i in range(self.num_local_devices):
      self.assertEqual(result.values[i].shape, [local_batch_size, 8, 8, 3])


def client_config_function(config_params):
  # [FUNCTION DOC]
  # HELPER: per-client setup callback for the multi-client test harness (passed to the main entry).
  # Input: a mapping with this client's id, the list of worker job addresses and the number of
  # devices per client. Writes the TF_CONFIG environment variable (cluster of workers, task type
  # worker with this client's index), picks the device type (GPU if physical GPUs exist, else TPU
  # if present, else CPU), resets the test context, configures virtual logical devices for
  # non-TPU types, and asserts the expected number of local logical devices exists (assertion
  # error with a mis-configuration message otherwise). Returns nothing.
  client_id = config_params['client_id']
  worker_jobs = config_params['worker_jobs']
  num_devices = config_params['num_devices']

  os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': worker_jobs
      },
      'task': {'type': 'worker', 'index': f'{client_id}'}
  })

  if config.list_physical_devices('GPU'):
    device_type = 'GPU'
  elif test_util.is_tpu_present():
    device_type = 'TPU'
  else:
    device_type = 'CPU'

  # reset_logical_devices
  test_util.reset_context()
  if device_type != 'TPU':
    # Configure virtual devices. This does not initialize the TensorFlow
    # context.
    test_util.reset_logical_devices(device_type, num_devices)

  # Validates the correct number of devices are created.
  logical_devices = test_util.list_local_logical_devices(device_type)
  assert len(logical_devices) == num_devices, (
      logical_devices,
      f'Test is mis-configured: expecting {num_devices} logical_devices.')


if __name__ == '__main__':
  test_backend_util.handle_test_main(
      multi_client_test_util.multi_client_main, client_config_function)
