# [MODULE DOC]
# Tests for the experimental DTensor-backed MirroredStrategy. Four test classes: general
# strategy behavior on a two-device batch mesh (variables, devices, run, gather, reduce,
# local results, replica context), rejection of a two-dimensional mesh, strategy creation from
# explicit, implicit or conflicting device and mesh arguments, and dataset / value distribution.
# Uses the DTensor test base class to configure meshes on TPU, GPU or CPU. No import-time side
# effects besides the test main entry.
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
"""Test for MirroredStrategy backed by DTensor API."""

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.distribute.experimental import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables


class StrategyBaseTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests the core behavior of the DTensor mirrored strategy: variable creation and layout,
  # mesh and extended properties, devices, run with various input types, replica context,
  # gather, reduce and local results. Fixture: a one-dimensional two-device batch mesh stored
  # as the mesh attribute.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: calls the parent setUp and builds and registers a two-device mesh
    # with one batch dimension for TPU, GPU and CPU, stored in the mesh attribute.
    super().setUp()
    global_ids = test_util.create_device_ids_array((2,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh(['batch'], global_ids, local_ids,
                            test_util.create_device_list((2,), device))
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh = self.configTestMesh(mesh_dict)

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
    # Parameterized over python floats, numpy arrays and tensors, each given either directly
    # (the factory is called first) or as an initializer callable (convert_callable false).
    # Creates a variable inside the strategy scope and asserts it is a DTensor variable with a
    # layout equal to the fully replicated rank-one layout on the mesh.
    if convert_callable:
      init_value = init_value()
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    with strategy.scope():
      v = variables.Variable(init_value)

    self.assertIsInstance(v, d_variable.DVariable)
    self.assertIsNotNone(v.layout)
    self.assertEqual(v.layout, layout.Layout.replicated(self.mesh, rank=1))

  def test_variable_creation_with_dtype(self):
    # [FUNCTION DOC]
    # Creates a scalar integer variable with explicit 64-bit integer type and only-first-replica
    # aggregation inside the strategy scope; asserts it is a DTensor variable with that type.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    with strategy.scope():
      v = variables.Variable(
          0, dtype='int64',
          aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)
    self.assertIsInstance(v, d_variable.DVariable)
    self.assertEqual(v.dtype, dtypes.int64)

  def test_mesh(self):
    # [FUNCTION DOC]
    # Asserts the strategy's mesh property equals the mesh it was constructed with.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    self.assertEqual(strategy.mesh, self.mesh)

  def test_strategy_extension(self):
    # [FUNCTION DOC]
    # Asserts the strategy's extended object is an instance of the standard strategy extended
    # base type.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    self.assertIsInstance(strategy.extended, distribute_lib.StrategyExtendedV2)

  def test_num_replica_in_sync(self):
    # [FUNCTION DOC]
    # Asserts the strategy reports two replicas in sync for the two-device mesh.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    self.assertEqual(strategy.num_replicas_in_sync, 2)

  def test_worker_devices(self):
    # [FUNCTION DOC]
    # Asserts the extended worker devices has two entries and equals the tuple of the mesh's
    # local devices.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    worker_devices = strategy.extended.worker_devices
    self.assertLen(worker_devices, 2)
    self.assertEqual(worker_devices, tuple(self.mesh.local_devices()))

  def test_parameter_devices(self):
    # [FUNCTION DOC]
    # Asserts the extended parameter devices has two entries and equals the tuple of the mesh's
    # local devices.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    parameter_devices = strategy.extended.parameter_devices
    self.assertLen(parameter_devices, 2)
    self.assertEqual(parameter_devices, tuple(self.mesh.local_devices()))

  def test_variable_created_in_scope(self):
    # [FUNCTION DOC]
    # Creates one variable inside the first strategy's scope, one outside any strategy, and one
    # inside a second strategy's scope. Asserts each strategy's variable-created-in-scope check
    # is true only for the variable created under that strategy.
    strategy1 = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    with strategy1.scope():
      v1 = variables.Variable(constant_op.constant([1.0, 2.0]))

    v2 = variables.Variable(constant_op.constant([1.0, 2.0]))

    strategy2 = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    with strategy2.scope():
      v3 = variables.Variable(constant_op.constant([1.0, 2.0]))

    self.assertTrue(strategy1.extended.variable_created_in_scope(v1))
    self.assertFalse(strategy1.extended.variable_created_in_scope(v2))
    self.assertFalse(strategy1.extended.variable_created_in_scope(v3))
    self.assertTrue(strategy2.extended.variable_created_in_scope(v3))

  def test_colocate_vars_with(self):
    # [FUNCTION DOC]
    # Inside the strategy scope creates a variable, then a second variable inside a colocation
    # context of the first. Asserts both variables have the same layout.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    with strategy.scope():
      v1 = variables.Variable(constant_op.constant([1.0, 2.0]))
      with strategy.extended.colocate_vars_with(v1):
        v2 = variables.Variable(constant_op.constant([2.0, 3.0]))

    # We assert the layout for the variable, and make sure they are same.
    self.assertEqual(v1.layout, v2.layout)

  def test_in_multi_worker_mode(self):
    # [FUNCTION DOC]
    # Asserts the extended object reports it is not in multi-worker mode.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    self.assertFalse(strategy.extended._in_multi_worker_mode())

  def test_run_with_tensor_inputs(self):
    # [FUNCTION DOC]
    # Calls strategy run eagerly with a plain scalar tensor argument and a compiled doubling
    # function. Asserts a value error stating the input types are unsupported, since plain
    # tensors are not accepted outside a compiled context.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    tensor_input = constant_op.constant(3.0)

    @def_function.function
    def replica_fn(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: returns the input multiplied by two.
      return inputs * 2.0

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'):
      strategy.run(replica_fn, args=(tensor_input,))

  def test_run_with_graph_tensor_inputs(self):
    # [FUNCTION DOC]
    # Inside a compiled outer function, creates a scalar tensor and passes it to strategy run
    # with a doubling function, under the strategy scope. Unlike the eager case this does not
    # raise; the input is treated as a global input. Asserts the result equals six.
    # Note that this is potentially a sharp edge for the user, since the eager
    # test case was raising an error, but the graph context will run, by treat
    # the inputs as a global inputs.
    # TODO(scottzhu): Mitigate this eager/graph behavior difference in future.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

    @def_function.function
    def replica_fn(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: returns the input multiplied by two.
      return inputs * 2.0

    @def_function.function
    def run_fn():
      # [FUNCTION DOC]
      # Compiled outer function: creates a scalar tensor of three and runs the replica function
      # on it through the strategy, returning the result.
      tensor_input = constant_op.constant(3.0)
      return strategy.run(replica_fn, args=(tensor_input,))

    with strategy.scope():
      result = run_fn()
    self.assertEqual(result, constant_op.constant(6.0))

  def test_run_with_unsupported_input_types(self):
    # [FUNCTION DOC]
    # Calls strategy run with a list holding an integer and a string. Asserts a value error
    # stating the input types are unsupported.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    random_inputs = [123, '456']

    @def_function.function
    def replica_fn(inputs):
      # [FUNCTION DOC]
      # Compiled replica function: returns the input multiplied by two (never reached
      # successfully in this test).
      return inputs * 2.0

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'):
      strategy.run(replica_fn, args=(random_inputs,))

  def test_run_with_distribute_value_input(self):
    # [FUNCTION DOC]
    # Builds distributed values whose per-replica value is the replica id, runs a doubling
    # function on them, and asserts the result is a DTensor distributed value with two
    # per-replica values: [0] and [2] (scalars are promoted to rank one).
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

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
    self.assertLen(result.values, 2)
    # Note that the scalar value from
    # experimental_distribute_values_from_function will be up rank to 1D since
    # batched shared dtensor need at least be 1D. So the result from the
    # strategy.run is [0], instead of just 0.
    self.assertAllClose(result.values[0], constant_op.constant([0]))
    self.assertAllClose(result.values[1], constant_op.constant([2]))

  def test_run_without_input(self):
    # [FUNCTION DOC]
    # Runs a compiled argument-less function returning a one-element tensor of one through the
    # strategy. Asserts the result is a DTensor distributed value with two values, each [1.0].
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

    @def_function.function
    def replica_fn():
      # [FUNCTION DOC]
      # Compiled replica function with no inputs: returns a one-element tensor holding one.
      return constant_op.constant([1.0])

    result = strategy.run(replica_fn)
    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertLen(result.values, 2)

    self.assertAllClose(result.values[0], constant_op.constant([1.0]))
    self.assertAllClose(result.values[1], constant_op.constant([1.0]))

  def test_nested_structure_output(self):
    # [FUNCTION DOC]
    # Distributes a mapping per replica (key a: a scalar from a numpy array indexed by replica
    # id; key b: a two-element tensor derived from it), runs a function doubling every entry,
    # and asserts the output is a mapping of two keys whose entries are DTensor distributed
    # values with the expected doubled per-replica values (a: 6 and 4; b: [8,10] and [6,8]).
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    array_value = np.array([3., 2., 1.])
    def value_fn(ctx):
      # [FUNCTION DOC]
      # Value function: picks the numpy element for the replica id and returns a mapping with a
      # scalar entry and a two-element tensor entry (value plus one, value plus two).
      value = array_value[ctx.replica_id_in_sync_group]
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
    self.assertAllClose(result['a'].values[0], constant_op.constant([6.0]))
    self.assertAllClose(result['a'].values[1], constant_op.constant([4.0]))

    self.assertIsInstance(result['b'], dtensor_util.DTensorDistributedValue)
    self.assertAllClose(result['b'].values[0],
                        constant_op.constant([8.0, 10.0]))
    self.assertAllClose(result['b'].values[1], constant_op.constant([6.0, 8.0]))

  def test_inputs_with_dtensor_distribute_values(self):
    # [FUNCTION DOC]
    # Distributes a scalar three to all replicas, runs a doubling function, then feeds that
    # distributed result into a second function adding one. Asserts both results are DTensor
    # distributed values with two entries each: six and six, then seven and seven.

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

    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    tensor_input = constant_op.constant(3.0)
    d_tensor_input = strategy.experimental_distribute_values_from_function(
        lambda _: tensor_input)

    result_1 = strategy.run(replica_fn_1, args=(d_tensor_input,))
    self.assertIsInstance(result_1, dtensor_util.DTensorDistributedValue)
    self.assertLen(result_1.values, 2)
    self.assertAllClose(result_1.values[0], constant_op.constant([6.0]))
    self.assertAllClose(result_1.values[1], constant_op.constant([6.0]))

    result_2 = strategy.run(replica_fn_2, args=(result_1,))
    self.assertIsInstance(result_2, dtensor_util.DTensorDistributedValue)
    self.assertLen(result_2.values, 2)
    self.assertAllClose(result_2.values[0], constant_op.constant([7.0]))
    self.assertAllClose(result_2.values[1], constant_op.constant([7.0]))

  def test_run_with_nullary_ops(self):
    # [FUNCTION DOC]
    # Runs an argument-less compiled function returning a one-element tensor of three through
    # the strategy; asserts a DTensor distributed value whose two entries are both [3.0].

    @def_function.function
    def replica_fn():
      # [FUNCTION DOC]
      # Compiled replica function with no inputs: returns a one-element tensor holding three.
      return constant_op.constant([3.0])

    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    result = strategy.run(replica_fn)

    self.assertIsInstance(result, dtensor_util.DTensorDistributedValue)
    self.assertAllClose(result.values[0], constant_op.constant([3.0]))
    self.assertAllClose(result.values[1], constant_op.constant([3.0]))

  def test_get_replica_context(self):
    # [FUNCTION DOC]
    # Verifies replica context behavior: outside any scope a default context exists; inside the
    # strategy scope but outside run there is none; inside the replica function the context is
    # the DTensor replica context. The function multiplies its input (integer three, distributed
    # to both replicas) by the number of replicas, so both result values equal [6].
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

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

    self.assertLen(result.values, 2)
    self.assertAllClose(result.values[0], constant_op.constant([6]))
    self.assertAllClose(result.values[1], constant_op.constant([6]))

  def test_gather_non_dtensor_value(self):
    # [FUNCTION DOC]
    # Gathers a plain scalar tensor along axis zero and asserts the result is unchanged.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    tensor_input = constant_op.constant(3.0)

    result = strategy.gather(tensor_input, axis=0)
    self.assertAllClose(result, tensor_input)

  def test_gather_dtensor_value(self):
    # [FUNCTION DOC]
    # Distributes per-replica tensors of shape one by two by three (a range of six numbers
    # starting at the replica id) and gathers them along axes zero, one and two. Asserts the
    # resulting shapes (2,2,3), (1,4,3), (1,2,6) and the concatenated contents for each axis.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

    def value_fn(value_context):
      # [FUNCTION DOC]
      # Value function: returns a tensor of shape one by two by three holding six consecutive
      # integers starting at the replica id.
      start = value_context.replica_id_in_sync_group
      return array_ops.reshape(math_ops.range(start=start, limit=start + 6),
                               shape=(1, 2, 3))

    distribute_result = strategy.experimental_distribute_values_from_function(
        value_fn)
    result = strategy.gather(distribute_result, axis=0)
    self.assertEqual(result.shape, [2, 2, 3])
    self.assertAllClose(result, [[[0, 1, 2], [3, 4, 5]],
                                 [[1, 2, 3], [4, 5, 6]]])
    result = strategy.gather(distribute_result, axis=1)
    self.assertEqual(result.shape, [1, 4, 3])
    self.assertAllClose(result, [[[0, 1, 2], [3, 4, 5], [1, 2, 3], [4, 5, 6]]])
    result = strategy.gather(distribute_result, axis=2)
    self.assertEqual(result.shape, [1, 2, 6])
    self.assertAllClose(result, [[[0, 1, 2, 1, 2, 3], [3, 4, 5, 4, 5, 6]]])

  def test_reduce_mean_non_dtensor_value(self):
    # [FUNCTION DOC]
    # Reduces (mean, axis zero) a plain tensor; asserts a value error saying the input types are
    # unsupported.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'):
      strategy.reduce(reduce_util.ReduceOp.MEAN, tensor_input, axis=0)

  def test_reduce_sum_non_dtensor_value(self):
    # [FUNCTION DOC]
    # Reduces (sum, axis zero) a plain tensor; asserts a value error saying the input types are
    # unsupported.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    with self.assertRaisesRegex(
        ValueError, 'Unsupported input types for MirroredStrategy.'):
      strategy.reduce(reduce_util.ReduceOp.SUM, tensor_input, axis=0)

  def test_reduce_mean_distribute_value(self):
    # [FUNCTION DOC]
    # Distributes a two by two matrix per replica (replica one holds zero to three, replica two
    # holds four to seven) and reduces with mean over no axis, axis zero and axis one. Asserts
    # the results: [[2,3],[4,5]], [3,4] and [2.5,4.5].
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

    @def_function.function
    def value_fn(value_context):
      # [FUNCTION DOC]
      # Compiled value function: returns the base two by two matrix plus the replica id times
      # the replica count times two, so each replica gets a distinct matrix.
      i = value_context.replica_id_in_sync_group
      n = value_context.num_replicas_in_sync
      return constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + i * n * 2.0

    distribute_value = strategy.experimental_distribute_values_from_function(
        value_fn)
    # replica 1 has [[0.0, 1.0],[2.0, 3.0]] and replica 2 has
    # [[4.0, 5.0],[6.0, 7.0]]

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=None)
    self.assertAllClose(result, constant_op.constant([[2.0, 3.0], [4.0, 5.0]]))

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=0)
    self.assertAllClose(result, constant_op.constant([3.0, 4.0]))

    result = strategy.reduce(
        reduce_util.ReduceOp.MEAN, distribute_value, axis=1)
    self.assertAllClose(result, constant_op.constant([2.5, 4.5]))

  def test_reduce_sum_distribute_value(self):
    # [FUNCTION DOC]
    # Same distributed matrices as the mean test, reduced with sum over no axis, axis zero and
    # axis one. Asserts the results: [[4,6],[8,10]], [12,16] and [10,18].
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

    @def_function.function
    def value_fn(value_context):
      # [FUNCTION DOC]
      # Compiled value function: returns the base two by two matrix plus the replica id times
      # the replica count times two, so each replica gets a distinct matrix.
      i = value_context.replica_id_in_sync_group
      n = value_context.num_replicas_in_sync
      return constant_op.constant([[0.0, 1.0], [2.0, 3.0]]) + i * n * 2.0

    distribute_value = strategy.experimental_distribute_values_from_function(
        value_fn)
    # replica 1 has [[0.0, 1.0],[2.0, 3.0]] and replica 2 has
    # [[4.0, 5.0],[6.0, 7.0]]

    result = strategy.reduce(
        reduce_util.ReduceOp.SUM, distribute_value, axis=None)
    self.assertAllClose(result, constant_op.constant([[4.0, 6.0], [8.0, 10.0]]))

    result = strategy.reduce(
        reduce_util.ReduceOp.SUM, distribute_value, axis=0)
    self.assertAllClose(result, constant_op.constant([12.0, 16.0]))

    result = strategy.reduce(
        reduce_util.ReduceOp.SUM, distribute_value, axis=1)
    self.assertAllClose(result, constant_op.constant([10.0, 18.0]))

  def test_reduce_mean_mirrored_value(self):
    # [FUNCTION DOC]
    # Creates a mirrored (replicated) two by two variable in scope and reduces it with mean over
    # no axis, axis zero and axis one; asserts [[1,2],[3,4]], [2,3] and [1.5,3.5].
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

    with  strategy.scope():
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
    # Creates a mirrored (replicated) two by two variable in scope and reduces it with sum over
    # no axis, axis zero and axis one; asserts [[1,2],[3,4]], [4,6] and [3,7].
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)

    with  strategy.scope():
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
    # Reduces (mean, no axis) a plain tensor through the strategy and asserts the result's device
    # string mentions the first CPU device. (Other tests reduce plain tensors along axis zero
    # and expect an unsupported-input error; this one uses no axis and expects success.)
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    tensor_input = constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    result = strategy.reduce(reduce_util.ReduceOp.MEAN, tensor_input, axis=None)
    self.assertIn('CPU:0', result.device)

  def test_experimental_local_results(self):
    # [FUNCTION DOC]
    # Runs an argument-less function returning [3.0] and asks for the local results; asserts
    # a tuple of two entries each equal to [3.0].
    @def_function.function
    def replica_fn():
      # [FUNCTION DOC]
      # Compiled replica function with no inputs: returns a one-element tensor holding three.
      return constant_op.constant([3.0])

    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    result = strategy.run(replica_fn)
    local_result = strategy.experimental_local_results(result)

    self.assertIsInstance(local_result, tuple)
    self.assertLen(local_result, 2)
    self.assertEqual(local_result[0], constant_op.constant([3.0]))
    self.assertEqual(local_result[1], constant_op.constant([3.0]))

  def test_experimental_local_results_with_inputs(self):
    # [FUNCTION DOC]
    # Distributes a per-replica mapping, runs a function doubling every entry, and takes the
    # local results. Asserts a tuple of two mappings: replica one has a: [6], b: [8,10];
    # replica two has a: [4], b: [6,8].
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    array_value = np.array([3., 2.])
    def value_fn(ctx):
      # [FUNCTION DOC]
      # Value function: picks the numpy element for the replica id and returns a mapping with a
      # scalar entry and a two-element tensor entry (value plus one, value plus two).
      value = array_value[ctx.replica_id_in_sync_group]
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
    local_result = strategy.experimental_local_results(result)
    self.assertIsInstance(local_result, tuple)
    self.assertLen(local_result, 2)
    self.assertDictEqual(local_result[0],
                         {'a': constant_op.constant([6.0]),
                          'b': constant_op.constant([8.0, 10.0])})
    self.assertDictEqual(local_result[1],
                         {'a': constant_op.constant([4.0]),
                          'b': constant_op.constant([6.0, 8.0])})


class InvalidMeshTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests that the strategy rejects a mesh that is not one-dimensional. Fixture: a two-dimensional
  # mesh with batch and model dimensions, stored as mesh_2d.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: calls the parent setUp and builds and registers a two-dimensional mesh
    # (batch and model, shape two by one) for TPU, GPU and CPU, stored as mesh_2d.
    super().setUp()
    global_ids = test_util.create_device_ids_array((2, 1))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh(['batch', 'model'], global_ids, local_ids,
                            test_util.create_device_list((2,), device))
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh_2d = self.configTestMesh(mesh_dict)

  def test_invalid_mesh_shape(self):
    # [FUNCTION DOC]
    # Constructs the strategy with the two-dimensional mesh and asserts a value error stating the
    # mesh must be one-dimensional and received a two-dimensional one.
    with self.assertRaisesRegex(
        ValueError, 'The mesh for MirroredStrategy must be 1D, received: 2D'):
      mirrored_strategy.MirroredStrategy(mesh=self.mesh_2d)


class StrategyCreationTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests creating the strategy from an explicit device list, from auto-detected devices, and
  # with conflicting mesh and device arguments. Fixture: the preferred device type, stored as
  # device_type, with two logical devices configured for non-TPU types.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: calls the parent setUp, determines the preferred device type, resets
    # the logical devices to two for non-TPU types (side effect on global device configuration)
    # and stores the type as device_type.
    super().setUp()
    device_type = test_util.preferred_device_type()
    if device_type != 'TPU':
      test_util.reset_logical_devices(device_type, 2)
    self.device_type = device_type

  def test_explicit_device_list(self):
    # [FUNCTION DOC]
    # Creates the strategy from two explicit device names. Asserts the mesh has two local
    # devices, shape [2], a single dimension named batch, local device names for devices zero and
    # one of the device type, and that a host mesh is available.
    device_list = [f'/{self.device_type}:{i}' for i in range(2)]
    strategy = mirrored_strategy.MirroredStrategy(devices=device_list)
    mesh = strategy.mesh
    self.assertEqual(mesh.num_local_devices(), 2)
    self.assertEqual(mesh.shape(), [2,])
    self.assertEqual(mesh.dim_names, ['batch'])
    self.assertIn(
        f'/job:localhost/replica:0/task:0/device:{self.device_type}:0',
        mesh.local_devices()[0])
    self.assertIn(
        f'/job:localhost/replica:0/task:0/device:{self.device_type}:1',
        mesh.local_devices()[1])
    # Also make sure the host mesh works since it is required by dataset
    self.assertIsNotNone(mesh.host_mesh())

  def test_implicit_device_list(self):
    # [FUNCTION DOC]
    # Creates the strategy with no arguments so devices are auto-detected. Asserts the mesh has
    # two local devices, shape [2], the expected local device names, and an available host mesh.
    strategy = mirrored_strategy.MirroredStrategy()
    mesh = strategy.mesh
    self.assertEqual(mesh.num_local_devices(), 2)
    self.assertEqual(mesh.shape(), [2,])
    self.assertIn(
        f'/job:localhost/replica:0/task:0/device:{self.device_type}:0',
        mesh.local_devices()[0])
    self.assertIn(
        f'/job:localhost/replica:0/task:0/device:{self.device_type}:1',
        mesh.local_devices()[1])
    # Also make sure the host mesh works since it is required by dataset
    self.assertIsNotNone(mesh.host_mesh())

  def test_mesh_with_device_list(self):
    # [FUNCTION DOC]
    # Builds a mesh from two devices, then creates the strategy with both the mesh and the device
    # list. Asserts a value error saying mesh and devices cannot be given together.
    device_list = [f'/{self.device_type}:{i}' for i in range(2)]
    mesh = mesh_util.create_mesh([('batch', 2)], devices=device_list)
    with self.assertRaisesRegex(
        ValueError, 'Mesh and devices can not be provided at the same time'):
      _ = mirrored_strategy.MirroredStrategy(mesh=mesh, devices=device_list)


class StrategyDatasetTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests dataset and value distribution through the DTensor mirrored strategy. Fixture: a
  # two-device batch mesh, random image (8x8x3) and label (1) tensors from fixed seeds, and an
  # infinitely repeating dataset of (image, label) pairs.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: calls the parent setUp, builds and registers a two-device batch mesh,
    # creates seeded random image and label tensors and a repeating dataset from them. Stores
    # mesh, images, labels and dataset attributes.
    super().setUp()
    global_ids = test_util.create_device_ids_array((2,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh(['batch'], global_ids, local_ids,
                            test_util.create_device_list((2,), device))
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh = self.configTestMesh(mesh_dict)

    self.images = stateless_random_ops.stateless_random_uniform(
        [8, 8, 3], seed=(1, 2), minval=0, maxval=255)
    self.labels = stateless_random_ops.stateless_random_uniform(
        [1], seed=(1, 2), minval=0, maxval=10)

    self.dataset = dataset_ops.Dataset.from_tensors(
        (self.images, self.labels)).repeat()

  def test_create_batched_dataset(self):
    # [FUNCTION DOC]
    # Batches the fixture dataset by eight with prefetch, distributes it and takes one element.
    # Asserts the global image and label shapes are batch by 8x8x3 and batch by 1, and that
    # unpacking each yields one shard per local device.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    global_batch_size = 8
    dataset = self.dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    element = next(iter(distributed_dataset))
    batched_image, batched_label = element
    self.assertEqual(batched_image.shape, [global_batch_size, 8, 8, 3])
    self.assertEqual(batched_label.shape, [global_batch_size, 1])

    # Make sure when unpack the tensor, each of them has enough shards.
    self.assertLen(d_api.unpack(batched_image), self.mesh.num_local_devices())
    self.assertLen(d_api.unpack(batched_label), self.mesh.num_local_devices())

  def test_uneven_batched_dataset(self):
    # [FUNCTION DOC]
    # Creates a repeating dataset from a generator of variable-length lists and tries to
    # distribute it; asserts a value error saying a static batch size is required.
    elements = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
    dataset = dataset_ops.Dataset.from_generator(
        lambda: elements, dtypes.int64).repeat()
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    with self.assertRaisesRegex(ValueError, 'requires a static batch size'):
      strategy.experimental_distribute_dataset(dataset)

  def test_create_partial_batched_dataset(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is skipped immediately (last partial batch unsupported, a tracked
    # bug), so everything after the skip call is dead code that never runs.
    # Intended scenario: a thirty-element dataset batched by eight should produce batches of
    # 8, 8, 8 and 6, each with the expected shapes and one shard per local device.
    # TODO(b/210887657): Support last partial batch.
    self.skipTest('Test failed due to last partial batch')
    dataset = dataset_ops.Dataset.from_tensors(
        (self.images, self.labels)).repeat(30)  # There is a last partial batch

    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    global_batch_size = 8
    dataset = dataset.batch(global_batch_size).prefetch(2)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    expected_element_batch_size = [8, 8, 8, 6]
    # The last batch with 6 element will fail to produce with StopIteration.
    iterator = iter(distributed_dataset)
    for batch_size in expected_element_batch_size:
      element = next(iterator)
      batched_image, batched_label = element
      self.assertEqual(batched_image.shape, [batch_size, 8, 8, 3])
      self.assertEqual(batched_label.shape, [batch_size, 1])

      # Make sure when unpack the tensor, each of them has enough shards.
      self.assertLen(d_api.unpack(batched_image), self.mesh.num_local_devices())
      self.assertLen(d_api.unpack(batched_label), self.mesh.num_local_devices())

  def test_deprecated_strategy_methods(self):
    # [FUNCTION DOC]
    # Asserts that the legacy dataset-iterator and input-function-iterator creation methods each
    # raise a not-implemented error saying they are only available in the V1 API.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    with self.assertRaisesRegex(
        NotImplementedError, 'only available in the V1 API'):
      strategy.make_dataset_iterator(self.dataset)

    with self.assertRaisesRegex(
        NotImplementedError, 'only available in the V1 API'):
      strategy.make_input_fn_iterator(lambda _: self.dataset)

  def test_distribute_dataset_from_fn(self):
    # [FUNCTION DOC]
    # Distributes a dataset built by a function producing per-replica batches of four. Asserts
    # the element spec is the global batch (8 by 8x8x3 images, 8 by 1 labels, float32) and equals
    # the iterator's spec, the first element has global batch size eight, and unpacking gives one
    # shard per device each with batch size four.
    local_batch_size = 4
    global_batch_size = 8
    def dataset_fn(option):
      # [FUNCTION DOC]
      # Dataset function: ignores its input-context argument and returns the repeating fixture
      # dataset batched by the local batch size (dropping remainders) with prefetch two.
      del option
      return dataset_ops.Dataset.from_tensors(
          (self.images, self.labels)).repeat().batch(
              local_batch_size, drop_remainder=True).prefetch(2)
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    distributed_dataset = strategy.distribute_datasets_from_function(
        dataset_fn, None)
    iterator = iter(distributed_dataset)

    self.assertEqual(distributed_dataset.element_spec,
                     (tensor_spec.TensorSpec(shape=(8, 8, 8, 3),
                                             dtype=dtypes.float32, name=None),
                      tensor_spec.TensorSpec(shape=(8, 1),
                                             dtype=dtypes.float32, name=None)))
    self.assertEqual(distributed_dataset.element_spec, iterator.element_spec)

    batched_image, batched_label = next(iterator)
    self.assertEqual(batched_image.shape, [global_batch_size, 8, 8, 3])
    self.assertEqual(batched_label.shape, [global_batch_size, 1])

    # Make sure there are two shards when unpack, and each of them has 4 as
    # batch size
    unpacked_images = d_api.unpack(batched_image)
    self.assertLen(unpacked_images, self.mesh.num_local_devices())
    self.assertEqual(unpacked_images[0].shape, [local_batch_size, 8, 8, 3])
    self.assertEqual(unpacked_images[1].shape, [local_batch_size, 8, 8, 3])

  def test_distribute_values_from_function(self):
    # [FUNCTION DOC]
    # Distributes per-replica scalars taken from a numpy array by replica id and asserts the
    # result is a batch-sharded rank-one DTensor equal to [3, 2] in double precision.
    array_value = np.array([3., 2., 1.])
    def value_fn(ctx):
      # [FUNCTION DOC]
      # Value function: returns the numpy element selected by the replica id.
      return array_value[ctx.replica_id_in_sync_group]
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))
    self.assertDTensorEqual(
        constant_op.constant([3., 2.], dtype=dtypes.float64),
        layout.Layout.batch_sharded(self.mesh, batch_dim='batch', rank=1),
        distributed_values)

  def test_distribute_values_from_function_with_nested_structure(self):
    # [FUNCTION DOC]
    # Distributes a per-replica mapping (a: scalar, b: two-element tensor). Asserts the result is
    # a mapping; entry a is a batch-sharded DTensor [3,2] whose shards are promoted to rank one
    # ([3] and [2]); entry b is a batch-sharded DTensor [4,5,3,4].
    array_value = np.array([3., 2., 1.])
    def value_fn(ctx):
      # [FUNCTION DOC]
      # Value function: picks the numpy element for the replica id and returns a mapping with a
      # scalar entry and a two-element tensor entry (value plus one, value plus two).
      value = array_value[ctx.replica_id_in_sync_group]
      return {'a': value,
              'b': constant_op.constant([value + 1.0, value + 2.0])}
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    distributed_values = (
        strategy.experimental_distribute_values_from_function(
            value_fn))
    self.assertIsInstance(distributed_values, dict)
    self.assertDTensorEqual(
        constant_op.constant([3., 2.], dtype=dtypes.float64),
        layout.Layout.batch_sharded(self.mesh, batch_dim='batch', rank=1),
        distributed_values['a'])
    unpacked_a = d_api.unpack(distributed_values['a'])
    # Note that this might have a slight behavior difference, the original
    # mirrored strategy may return scalar for each PerReplica. The DTensor
    # implementation is more strict and always ensures the PerReplica
    # value has the same rank as the global-view Tensor.
    self.assertAllClose(unpacked_a[0], [3.])
    self.assertAllClose(unpacked_a[1], [2.])
    self.assertDTensorEqual(
        constant_op.constant([4., 5., 3., 4.], dtype=dtypes.float64),
        layout.Layout.batch_sharded(self.mesh, batch_dim='batch', rank=1),
        distributed_values['b'])

  def test_distribute_dataset_in_tf_function(self):
    # [FUNCTION DOC]
    # Distributes a global-batch-8 dataset, then runs a compiled step that pulls the next element
    # from the iterator and returns the images. Asserts the result is a DTensor distributed value
    # with one entry per device, each with local batch size four and shape 4x8x8x3.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    local_batch_size = 4
    global_batch_size = 8
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
    self.assertLen(result.values, self.mesh.num_local_devices())
    self.assertEqual(result.values[0].shape, [local_batch_size, 8, 8, 3])
    self.assertEqual(result.values[1].shape, [local_batch_size, 8, 8, 3])


if __name__ == '__main__':
  test.main()
