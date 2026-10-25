# [MODULE DOC]
# Functional tests for the slot creation helpers in the training package. Defines a small
# helper function and one test class that builds slots (from variables, tensors, partitioned
# variables, scalars, dynamically shaped inputs and XLA sharded variables) inside fresh graphs
# and checks resulting names, shapes, types, values, save-slice info and sharding annotations.
# Runs under the standard test runner when executed as the main program. No import-time side
# effects beyond imports.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional test for slot_creator."""

import numpy as np

from xla import xla_data_pb2
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import slot_creator


def initialized_value(var):
  # [FUNCTION DOC]
  # HELPER: small shared test utility used to build slot initial values from variables.
  # Input: var, a variable. Returns a tensor that is the variable's current value if it is
  # already initialized, otherwise its initial value, chosen at run time by a conditional.
  # Adds ops to the current graph; no other side effects.
  return cond.cond(
      variable_v1.is_variable_initialized(var), var.read_value,
      lambda: var.initial_value)


class SlotCreatorTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the slot creation functions. There are no shared fixtures; every test
  # builds its own graph and session, creates a slot, and asserts on the slot's operation name,
  # shape, element type, value and, for some, partition or XLA sharding metadata.

  def testCreateSlotFromVariable(self):
    # [FUNCTION DOC]
    # Scenario: a two element float variable gets a slot made from its initialized value. After
    # running global initialization, asserts the slot's operation name is the variable name
    # plus the slot name, its shape is two, its type is 32-bit float and its values equal the
    # variable's initial values.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable([1.0, 2.5], name="var")
      slot = slot_creator.create_slot(v, initialized_value(v), name="slot")

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("var/slot", slot.op.name)
      self.assertEqual([2], slot.get_shape().as_list())
      self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
      self.assertAllEqual([1.0, 2.5], self.evaluate(slot))

  def testCreateSlotFromTensor(self):
    # [FUNCTION DOC]
    # Scenario: a constant tensor is the primary and the slot's initial value is the constant
    # doubled. Asserts the slot name is the constant's name plus the slot name, shape two,
    # 32-bit float type, and values equal to the doubled constant.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      v = constant_op.constant([1.0, 2.5], name="const")
      slot = slot_creator.create_slot(v, v * 2, name="slot")

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("const/slot", slot.op.name)
      self.assertEqual([2], slot.get_shape().as_list())
      self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
      self.assertAllEqual([2.0, 5.0], self.evaluate(slot))

  def testCreateZerosSlotFromVariable(self):
    # [FUNCTION DOC]
    # Scenario: a zeros slot with 64-bit float type is created for a two element float variable,
    # outside any control dependencies. Asserts name is variable name plus slot name, shape
    # two, type 64-bit float (overriding the primary's type), and all values zero.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable([1.0, 2.5], name="var")
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64)

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("var/slot", slot.op.name)
      self.assertEqual([2], slot.get_shape().as_list())
      self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
      self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

  def testCreateZerosSlotFromDynamicShapedVariable(self):
    # [FUNCTION DOC]
    # Scenario: a variable is created with an initializer whose shape is only known at run time
    # (a placeholder with default of unknown dimension) and shape validation off. A zeros slot
    # of 64-bit float type is created. Asserts the slot name, that the run-time shape is two,
    # the type, and that values are zeros.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      dyn_shape = constant_op.constant([2], dtype=dtypes.int32)
      dyn_shape = array_ops.placeholder_with_default(dyn_shape,
                                                     shape=[None])
      v = variable_scope.get_variable(
          "var",
          initializer=random_ops.random_uniform(dyn_shape,
                                                dtype=dtypes.float64),
          validate_shape=False)
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64)

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("var/slot", slot.op.name)
      self.assertEqual([2], array_ops.shape(slot).eval())
      self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
      self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

  def testCreateZerosSlotFromTensor(self):
    # [FUNCTION DOC]
    # Scenario: a zeros slot is created for a constant tensor with the default type. Asserts the
    # slot name uses the constant's name, shape two, 32-bit float type inherited from the
    # primary, and zero values.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      v = constant_op.constant([1.0, 2.5], name="const")
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(v, name="slot")

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("const/slot", slot.op.name)
      self.assertEqual([2], slot.get_shape().as_list())
      self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
      self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

  def testCreateZerosSlotFromDynamicShapedTensor(self):
    # [FUNCTION DOC]
    # Scenario: the primary is a tensor of unknown static shape (placeholder with default of
    # random 64-bit floats, named const). A 64-bit zeros slot is created. Asserts the slot name,
    # run-time shape two, type 64-bit float, and zero values.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      v = random_ops.random_uniform([2], dtype=dtypes.float64)
      v = array_ops.placeholder_with_default(v, shape=[None], name="const")
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64)

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("const/slot", slot.op.name)
      self.assertEqual([2], array_ops.shape(slot).eval())
      self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
      self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

  def testCreateSlotFromVariableRespectsScope(self):
    # [FUNCTION DOC]
    # Scenario: inside an open variable scope named scope, a variable is created and a slot is
    # made for it. Only asserts the slot's operation name is the scope name repeated twice, then
    # the variable name, then the slot name (the scope is applied both by the variable and the
    # slot). Related to a past issue discussion numbered 2740.
    # See discussion on #2740.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      with variable_scope.variable_scope("scope"):
        v = variables.Variable([1.0, 2.5], name="var")
        slot = slot_creator.create_slot(v, initialized_value(v), name="slot")
        self.assertEqual("scope/scope/var/slot", slot.op.name)

  def testCreateSlotFromFirstMDimensionVariable(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the legacy test_session helper rather than cached_session, unlike its
    # sibling tests; behavior is equivalent here but the helper is deprecated.
    # Scenario: a two by two variable is split into two partitions; for each partition a slot is
    # made whose initial value is a separate two element variable (the slot has fewer
    # dimensions than the primary). Asserts per partition: slot name contains the part index,
    # shape two, 32-bit float type, values one and two and a half, and the slot's save-slice
    # info has full shape two, offset equal to the part index and slice shape one.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.test_session():
      s = variables.Variable([1.0, 2.5], name="var")
      p_v = variable_scope.get_variable(
          "var",
          shape=[2, 2],
          partitioner=partitioned_variables.fixed_size_partitioner(2))
      for i, v in enumerate(p_v):
        slot = slot_creator.create_slot(v, initialized_value(s), name="slot")
        si = slot._save_slice_info

        self.evaluate(variables.global_variables_initializer())

        self.assertEqual("var/part_%d/slot" % i, slot.op.name)
        self.assertEqual([2], slot.get_shape().as_list())
        self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
        self.assertAllEqual([1.0, 2.5], slot)
        self.assertAllEqual([2], si.full_shape)
        self.assertAllEqual([i], si.var_offset)
        self.assertAllEqual([1], si.var_shape)

  def testCreateSlotFromScalarVariable(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the legacy test_session helper rather than cached_session.
    # Scenario: a two by two variable is partitioned in two and each partition gets a slot
    # initialized from a scalar variable. Asserts the slot name contains the part index, the
    # slot is a scalar (empty shape), 32-bit float, with value one. Scalar slots get no slice
    # info.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.test_session():
      s = variables.Variable(1.0, name="var")
      p_v = variable_scope.get_variable(
          "var",
          shape=[2, 2],
          partitioner=partitioned_variables.fixed_size_partitioner(2))
      for i, v in enumerate(p_v):
        slot = slot_creator.create_slot(v, initialized_value(s), name="slot")

        self.evaluate(variables.global_variables_initializer())

        self.assertEqual("var/part_%d/slot" % i, slot.op.name)
        self.assertEqual([], slot.get_shape().as_list())
        self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
        self.assertAllEqual(1.0, slot)

  def testCreateSlotFromVariableCopyXlaSharding(self):
    # [FUNCTION DOC]
    # Scenario: a variable is annotated with a mesh split sharding (without a sharding op) and
    # a slot is created from its initialized value with sharding copy enabled. Asserts the
    # slot's tensor sharding equals the variable's.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable([1.0, 2.5], name="var")
      v = xla_sharding.mesh_split(
          v, np.array([0, 1]), [0], use_sharding_op=False)
      slot = slot_creator.create_slot(
          v, initialized_value(v), name="slot", copy_xla_sharding=True)
      self.assertEqual(
          xla_sharding.get_tensor_sharding(v),
          xla_sharding.get_tensor_sharding(slot))

  def testCreateZerosSlotFromVariableCopyXlaSharding(self):
    # [FUNCTION DOC]
    # Scenario: same as the value-based sharding copy test but using the zeros slot creator with
    # 64-bit type and sharding copy enabled. Asserts the slot's tensor sharding equals the
    # variable's.
    # slot_creator is used only in optimizer V1.
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable([1.0, 2.5], name="var")
      v = xla_sharding.mesh_split(
          v, np.array([0, 1]), [0], use_sharding_op=False)
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64, copy_xla_sharding=True)
      self.assertEqual(
          xla_sharding.get_tensor_sharding(v),
          xla_sharding.get_tensor_sharding(slot))

  def testCreateSlotWithoutXlaSharding(self):
    # [FUNCTION DOC]
    # Scenario: a sharded two element variable gets a slot whose initial value is a scalar
    # constant, with sharding copy enabled. Because the ranks differ the sharding must not be
    # copied. Asserts the slot has no sharding annotation and that it differs from the
    # variable's.
    # slot_creator is used only in optimizer V1.
    # The SPMD sharding annotations should not be copied since the primary
    # variable and slot variable have different ranks.
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable([1.0, 2.5], name="var")
      v = xla_sharding.mesh_split(
          v, np.array([0, 1]), [0], use_sharding_op=False)
      with ops.control_dependencies(None):
        slot = slot_creator.create_slot(
            v,
            constant_op.constant(10, name="const"),
            name="slot",
            copy_xla_sharding=True)
      self.assertIsNone(xla_sharding.get_tensor_sharding(slot))
      self.assertNotEqual(
          xla_sharding.get_tensor_sharding(v),
          xla_sharding.get_tensor_sharding(slot))

  def testCreateSlotWithCustomReplicatedXlaSharding(self):
    # [FUNCTION DOC]
    # Scenario: a sharded variable gets a zeros slot with sharding copy enabled, then the slot
    # is explicitly marked replicated. Asserts the slot's sharding now differs from the
    # variable's and, after parsing the serialized sharding message, equals a replicated
    # sharding message.
    # slot_creator is used only in optimizer V1.
    # We insert our own custom replicated XLA sharding that overrides the SPMD
    # sharding copied over by the slot_creator.
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable([1.0, 2.5], name="var")
      v = xla_sharding.mesh_split(
          v, np.array([0, 1]), [0], use_sharding_op=False)
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64, copy_xla_sharding=True)
        slot = xla_sharding.replicate(slot, use_sharding_op=False)

      self.assertNotEqual(
          xla_sharding.get_tensor_sharding(v),
          xla_sharding.get_tensor_sharding(slot))

      slot_sharding = xla_sharding.get_tensor_sharding(slot)
      slot_proto = xla_data_pb2.OpSharding()
      slot_proto.ParseFromString(slot_sharding)
      self.assertEqual(
          slot_proto,
          xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED))

  def testCreateSlotWithCustomSplitXlaSharding(self):
    # [FUNCTION DOC]
    # Scenario: a sharded four element variable gets a zeros slot with sharding copy enabled,
    # then the slot is explicitly split along dimension zero across four devices. Asserts the
    # slot's sharding differs from the variable's and, once parsed, equals a tiled sharding
    # message with tile dimensions of four and devices zero through three.
    # slot_creator is used only in optimizer V1.
    # We insert our own custom split XLA sharding that overrides the SPMD
    # sharding copied over by the slot_creator.
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable([1.0, 2.5, 10.0, 15.1], name="var")
      v = xla_sharding.mesh_split(
          v, np.array([0, 1]), [0], use_sharding_op=False)
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64, copy_xla_sharding=True)
        slot = xla_sharding.split(
            slot, split_dimension=0, num_devices=4, use_sharding_op=False)

      self.assertNotEqual(
          xla_sharding.get_tensor_sharding(v),
          xla_sharding.get_tensor_sharding(slot))

      slot_sharding = xla_sharding.get_tensor_sharding(slot)
      slot_proto = xla_data_pb2.OpSharding()
      slot_proto.ParseFromString(slot_sharding)
      self.assertEqual(
          slot_proto,
          xla_data_pb2.OpSharding(
              type=xla_data_pb2.OpSharding.OTHER,
              tile_assignment_dimensions=[4],
              tile_assignment_devices=range(4)))


if __name__ == "__main__":
  test.main()
