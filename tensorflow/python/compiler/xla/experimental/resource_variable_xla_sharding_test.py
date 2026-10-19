# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# TPU test checking that XLA sharding annotations on resource variables propagate to the reads of
# those variables, including optimizer slot variables, and that this behavior can be disabled
# temporarily or permanently. Contains small graph-inspection helper functions and one test class.
# Requires 8 TPU devices (tests skip otherwise). Run as a script it launches the test runner.

from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment
from tensorflow.python.tpu import tpu
from tensorflow.python.training import adagrad


# Gets all the nodes of `op` in graph that have `input_node_name` as one of the
# inputs
def _get_op_nodes_with_input(input_node_name, op, graph):
  # [FUNCTION DOC]
  # HELPER: graph inspection utility for the tests. Inputs: input_node_name (string name of an
  # input), op (string operation type) and graph (a graph definition). Returns a list of all nodes
  # of that operation type that list the given name among their inputs.
  nodes_with_input = []
  for node in graph.node:
    nodes_with_input += [
        node
        for input in node.input
        if input == input_node_name and node.op == op
    ]
  return nodes_with_input


# Gets XlaSharding ops connected to ReadVariableOp for the given variable_name
def _get_xla_sharding_nodes_for_variable(variable_name, graph):
  # [FUNCTION DOC]
  # HELPER: finds the sharding annotation nodes applied to a variable's value. Inputs: variable_name
  # (string) and graph (a graph definition). Finds the variable-read nodes consuming the variable,
  # then the sharding nodes consuming those reads. Returns the list of sharding nodes.
  read_variable_op_nodes = _get_op_nodes_with_input(
      variable_name, 'ReadVariableOp', graph
  )
  xla_sharding_op_nodes = []
  for read_variable_op_node in read_variable_op_nodes:
    xla_sharding_op_nodes += _get_op_nodes_with_input(
        read_variable_op_node.name, 'XlaSharding', graph
    )
  return xla_sharding_op_nodes


def _get_xla_sharding_proto_from_node(node):
  # [FUNCTION DOC]
  # HELPER: decodes the sharding message stored in the sharding attribute of a graph node. Input:
  # node, a sharding node. Returns the parsed sharding message.
  sharding_proto = xla_sharding.xla_data_pb2.OpSharding()
  sharding_proto.ParseFromString(node.attr['sharding'].s)
  return sharding_proto


class ResourceVariableXlaShardingTest(test.TestCase):
  # [CLASS DOC]
  # Tests for automatic XLA sharding of resource variables on an 8-TPU system. Shared fixture: the
  # sharding feature is enabled, the TPU system is initialized and a device assignment over a
  # 2x2x1x2 computation shape with one replica is built.

  def setUp(self) -> None:
    # [FUNCTION DOC]
    # Per-test setup: runs parent setup, enables XLA sharding for resource variables, initializes
    # the TPU system (keeping its topology), skips the test unless exactly 8 logical TPU devices
    # exist, and builds the device assignment used by tests. Changes global context state.
    super().setUp()

    context.enable_xla_sharding_for_resource_variables()
    self.topology = tpu_cluster_resolver.initialize_tpu_system()
    if len(config.list_logical_devices('TPU')) != 8:
      self.skipTest('All tests require 8 TPUs.')

    self.da = device_assignment.DeviceAssignment.build(
        self.topology, computation_shape=[2, 2, 1, 2], num_replicas=1
    )

  def test_xla_sharding_ops_created_for_optimizer_slot_variables(self):
    # [FUNCTION DOC]
    # Scenario: a variable of eight floats is split across 8 devices, then an Adagrad optimizer
    # minimizes the summed result inside a replicated TPU computation. Asserts the computation
    # result is 176, that exactly one Adagrad update node exists, and that both the variable input
    # and the slot variable input of that update each have exactly one sharding node whose sharding
    # message equals the one set on the original variable.
    w = variables.Variable(
        initial_value=math_ops.range(8, dtype=dtypes.float32),
        name='w',
    )
    self.assertIsInstance(w, resource_variable_ops.BaseResourceVariable)
    w = xla_sharding.split(
        w,
        split_dimension=0,
        num_devices=8,
    )
    sharding_proto = xla_sharding.xla_data_pb2.OpSharding()
    sharding_proto.ParseFromString(xla_sharding.get_tensor_sharding(w))
    opt = adagrad.AdagradOptimizer(1.0)

    @def_function.function
    def computation(x):
      # [FUNCTION DOC]
      # HELPER: traced function that replicates the TPU function over the device assignment with x
      # as the single input and returns the replicated outputs.
      def tpu_fn(x):
        # [FUNCTION DOC]
        # HELPER: per-replica body: adds the sharded variable to x, sums the result into a loss,
        # runs one optimizer minimization step on the variable, and returns the loss.
        y = math_ops.add(w, x)
        loss = math_ops.reduce_sum(y)
        opt.minimize(loss, None, [w])
        return loss

      output = tpu.replicate(tpu_fn, [[x]], device_assignment=self.da)
      return output

    inputs = array_ops.reshape(math_ops.range(16, dtype=dtypes.float32), (2, 8))
    result = computation(inputs)
    self.assertSequenceEqual([[176.0]], self.evaluate(result))
    graph = computation.get_concrete_function(inputs).graph.as_graph_def()

    update_op_nodes = [
        node for node in graph.node if node.op == 'ResourceApplyAdagrad'
    ]
    self.assertLen(update_op_nodes, 1)
    update_op_node = update_op_nodes[0]

    var_input_name = update_op_node.input[0]
    var_sharding_nodes = _get_xla_sharding_nodes_for_variable(
        var_input_name, graph
    )
    self.assertLen(var_sharding_nodes, 1)
    self.assertProtoEquals(
        _get_xla_sharding_proto_from_node(var_sharding_nodes[0]), sharding_proto
    )

    slot_var_input_name = update_op_node.input[1]
    slot_var_sharding_nodes = _get_xla_sharding_nodes_for_variable(
        slot_var_input_name, graph
    )
    self.assertLen(slot_var_sharding_nodes, 1)
    self.assertProtoEquals(
        _get_xla_sharding_proto_from_node(slot_var_sharding_nodes[0]),
        sharding_proto,
    )

  def test_disabling_xla_sharding_ops_temporarily(self):
    # [FUNCTION DOC]
    # Scenario: with sharding for resource variables enabled, splitting a variable inside a
    # temporary-disable scope must raise an attribute error about the tensor operation being
    # undefined in eager mode; after the scope, splitting works again. After permanently
    # disabling the feature, the same temporary-disable scope and a plain split outside it both
    # raise that error.
    w = variables.Variable(
        initial_value=math_ops.range(8, dtype=dtypes.float32),
        name='w',
    )
    self.assertIsInstance(w, resource_variable_ops.BaseResourceVariable)

    context.enable_xla_sharding_for_resource_variables()
    with context.temporarily_disable_xla_sharding_for_resource_variables():
      with self.assertRaisesRegex(
          AttributeError,
          '.*Tensor.op is undefined when eager execution is enabled.*',
      ):
        xla_sharding.split(
            w,
            split_dimension=0,
            num_devices=8,
        )

    # xla_sharding_for_resource_variables is enabled again. Following line
    # doesn't throw an error.
    xla_sharding.split(
        w,
        split_dimension=0,
        num_devices=8,
    )

    context.disable_xla_sharding_for_resource_variables()
    with context.temporarily_disable_xla_sharding_for_resource_variables():
      with self.assertRaisesRegex(
          AttributeError,
          '.*Tensor.op is undefined when eager execution is enabled.*',
      ):
        xla_sharding.split(
            w,
            split_dimension=0,
            num_devices=8,
        )

    # xla_sharding_for_resource_variables stays disabled.
    with self.assertRaisesRegex(
        AttributeError,
        '.*Tensor.op is undefined when eager execution is enabled.*',
    ):
      xla_sharding.split(
          w,
          split_dimension=0,
          num_devices=8,
      )


if __name__ == '__main__':
  test.main()
