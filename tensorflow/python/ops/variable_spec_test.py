# [MODULE DOC]
# Unit tests for VariableSpec, the type specification of resource variables (exposed through the
# resource variable ops module). The module defines one test class that runs every test in both
# graph and eager modes. The tests cover spec properties, compatibility checks against other specs,
# variables and tensors, creation of a spec from a variable, conversion to and from components
# (the resource handle), error handling for bad components, serialization, repr, hashing,
# equality, subtype checks and most-specific-common-supertype computation.
# Import-time side effects: only a module-level alias for the VariableSpec class. When run as a
# script it starts the TensorFlow test runner.
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
"""Tests for VariableSpec."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest

VariableSpec = resource_variable_ops.VariableSpec


@test_util.run_all_in_graph_and_eager_modes
class VariableSpecTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the VariableSpec type specification. The class decorator runs each test
  # in both graph mode and eager mode. It supports parameterized tests and holds no shared
  # fixtures; each test builds its own specs and variables.

  def test_properties(self):
    # [FUNCTION DOC]
    # Checks the public properties of a spec: name, shape, dtype, trainable flag, value type and
    # component specs. A spec built with only a shape must have no name, float32 dtype, be
    # trainable, have the resource variable class as value type and a single scalar resource
    # tensor component. A second spec with explicit shape, float64 dtype and trainable false
    # must report those values. Returns nothing; fails on assertion mismatch.
    spec = VariableSpec(shape=(None, None, None))
    self.assertIsNone(spec.name)
    self.assertAllEqual(spec.shape.as_list(), [None, None, None])
    self.assertEqual(spec.dtype, dtypes.float32)
    self.assertTrue(spec.trainable)
    self.assertIs(spec.value_type, resource_variable_ops.ResourceVariable)
    self.assertAllEqual(spec._component_specs,
                        [tensor_spec.TensorSpec([], dtypes.resource)])

    spec2 = VariableSpec(shape=(1, 2, 3), dtype=dtypes.float64,
                         trainable=False)
    self.assertEqual(spec2.shape.as_list(), [1, 2, 3])
    self.assertEqual(spec2.dtype, dtypes.float64)
    self.assertFalse(spec2.trainable)
    self.assertIs(spec2.value_type, resource_variable_ops.ResourceVariable)
    self.assertAllEqual(spec2._component_specs,
                        [tensor_spec.TensorSpec([], dtypes.resource)])

  def test_compatibility(self):
    # [FUNCTION DOC]
    # Checks is_compatible_with between specs, variables and tensors. A spec with unknown shape
    # is compatible with a more specific shape spec but a spec with shape [None, 15] is not
    # compatible with a rank-1 spec. Specs are compatible with variables of matching dtype and
    # shape, and a spec with a different dtype is incompatible. A plain tensor is never
    # compatible with a variable spec. Returns nothing.
    spec = VariableSpec(shape=None)
    spec2 = VariableSpec(shape=[None, 15])
    spec3 = VariableSpec(shape=[None])

    self.assertTrue(spec.is_compatible_with(spec2))
    self.assertFalse(spec2.is_compatible_with(spec3))

    var = resource_variable_ops.ResourceVariable(
        initial_value=np.ones((3, 15), dtype=np.float32))
    var2 = resource_variable_ops.ResourceVariable(
        initial_value=np.ones((3,), dtype=np.int32))

    self.assertTrue(spec.is_compatible_with(var))
    self.assertFalse(spec2.is_compatible_with(var2))

    spec4 = VariableSpec(shape=None, dtype=dtypes.int32)
    spec5 = VariableSpec(shape=[None], dtype=dtypes.int32)

    self.assertFalse(spec.is_compatible_with(spec4))
    self.assertTrue(spec4.is_compatible_with(spec5))
    self.assertTrue(spec4.is_compatible_with(var2))

    tensor = constant_op.constant([1, 2, 3])
    self.assertFalse(spec4.is_compatible_with(tensor))

  @parameterized.parameters([
      dict(
          initial_value=[1, 2, 3],
          shape=[3],
          dtype=dtypes.int32,
          trainable=False),
      dict(
          initial_value=[[1., 2., 3.]],
          shape=[1, None]),
  ])
  def testFromValue(self,
                    initial_value=None,
                    shape=None,
                    dtype=dtypes.float32,
                    trainable=True):
    # [FUNCTION DOC]
    # Parameterized test of building a spec from an existing variable. Inputs (supplied by the
    # parameter sets): initial_value, shape, dtype (default float32) and trainable (default
    # true). A resource variable is created with these, then a spec is made from it. Asserts the
    # spec shape, dtype and trainable flag equal the inputs and that the alias id is unset.
    # Returns nothing.
    var = resource_variable_ops.ResourceVariable(
        initial_value=initial_value,
        shape=shape,
        dtype=dtype,
        trainable=trainable)
    spec = resource_variable_ops.VariableSpec.from_value(var)
    self.assertEqual(spec.shape, shape)
    self.assertEqual(spec.dtype, dtype)
    self.assertEqual(spec.trainable, trainable)
    self.assertIsNone(spec.alias_id)

  @parameterized.parameters([
      dict(
          initial_value=[1., 2., 3.],
          shape=[3]),
      dict(
          initial_value=[1., 2., 3.],
          shape=None),
      dict(
          initial_value=[[1, 2, 3]],
          shape=[1, None],
          dtype=dtypes.int32),
      dict(
          initial_value=[[1, 2, 3]],
          shape=[None, None],
          dtype=dtypes.int32),
  ])
  def testToFromComponents(self,
                           initial_value=None,
                           shape=None,
                           dtype=dtypes.float32,
                           trainable=True):
    # [FUNCTION DOC]
    # Parameterized test of the round trip between a variable and its components. Inputs come
    # from the parameter sets: initial value, shape, dtype and trainable flag. Creates a
    # variable (initializing it in graph mode), makes a spec from it, converts the variable to
    # components and asserts the result is a list of length one holding the variable handle.
    # Rebuilds a variable from the components and asserts the same value and trainable flag.
    # Returns nothing.
    var = resource_variable_ops.ResourceVariable(
        initial_value=initial_value,
        shape=shape,
        dtype=dtype,
        trainable=trainable)
    if not context.executing_eagerly():
      self.evaluate(var.initializer)

    spec = resource_variable_ops.VariableSpec.from_value(var)
    components = spec._to_components(var)
    self.assertIsInstance(components, list)
    self.assertLen(components, 1)
    self.assertIs(components[0], var.handle)
    rebuilt_var = spec._from_components(components)
    self.assertAllEqual(rebuilt_var.read_value(), var.read_value())
    self.assertEqual(rebuilt_var.trainable, trainable)

  def testFromComponentsSetHandleData(self):
    # [FUNCTION DOC]
    # Checks that rebuilding a variable from components fills in handle data. Takes the handle
    # data of a real variable as the expectation, then creates a resource placeholder inside a
    # fresh graph that has no handle data (asserted unset). A spec with shape [1] and float32
    # converts that placeholder into a variable; the resulting handle must have handle data set
    # whose shape and dtype equal the expected ones. Returns nothing.
    v = resource_variable_ops.ResourceVariable([1.])
    if not context.executing_eagerly():
      self.evaluate(v.initializer)

    expected_handle_data = resource_variable_ops.get_eager_safe_handle_data(
        v.handle)

    with ops.Graph().as_default():
      # Create a resource tensor without handle data. tf.placeholder could only
      # be called in graph mode.
      handle1 = array_ops.placeholder(dtypes.resource, [])
    handle1_data = resource_variable_ops.get_eager_safe_handle_data(handle1)
    self.assertFalse(handle1_data.is_set)

    spec = resource_variable_ops.VariableSpec(shape=[1], dtype=dtypes.float32)
    # Spec should set the handle shape and dtype of handle1.
    handle2 = spec._from_components([handle1]).handle
    handle2_data = resource_variable_ops.get_eager_safe_handle_data(handle2)
    self.assertTrue(handle2_data.is_set)
    self.assertEqual(handle2_data.shape_and_type[0].shape,
                     expected_handle_data.shape_and_type[0].shape)
    self.assertEqual(handle2_data.shape_and_type[0].dtype,
                     expected_handle_data.shape_and_type[0].dtype)

  def testFromComponentsError(self):
    # [FUNCTION DOC]
    # Checks the validation in rebuilding a variable from components. Passing a bare tensor must
    # raise a type error saying a list or tuple is required; a list with two tensors must raise
    # a value error saying only the resource handle is allowed; a list with a non-resource
    # tensor must raise a value error saying a resource tensor is required. Returns nothing.
    spec = resource_variable_ops.VariableSpec(shape=[1], dtype=dtypes.float32)
    self.assertRaisesRegex(TypeError, "must be a list or tuple",
                           spec._from_components, constant_op.constant(1.))
    self.assertRaisesRegex(ValueError,
                           "must only contain its resource handle",
                           spec._from_components,
                           [constant_op.constant(1.), constant_op.constant(2.)])
    self.assertRaisesRegex(ValueError, "must be a resource tensor",
                           spec._from_components, [constant_op.constant(1.)])

  def testComponentSpecs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is disabled; it unconditionally skips itself (bug reference about the
    # resource variable not yet being a composite tensor), so the checks after the skip never run.
    # Intended behavior: flattening a spec with composites expanded yields exactly one scalar
    # resource-typed tensor spec. Returns nothing.
    self.skipTest("b/209081027: re-enable this test after ResourceVariable "
                  "becomes a subclass of CompositeTensor.")
    spec = resource_variable_ops.VariableSpec([1, 3], dtypes.float32)
    handle_specs = nest.flatten(spec, expand_composites=True)
    self.assertLen(handle_specs, 1)
    handle_spec = handle_specs[0]
    self.assertAllEqual(handle_spec.shape, [])
    self.assertEqual(handle_spec.dtype, dtypes.resource)

  def testValueType(self):
    # [FUNCTION DOC]
    # Asserts that a spec of shape [1, 3] and float32 reports the resource variable class as its
    # value type. Returns nothing.
    spec = resource_variable_ops.VariableSpec([1, 3], dtypes.float32)
    self.assertIs(spec.value_type, resource_variable_ops.ResourceVariable)

  def testSerialize(self):
    # [FUNCTION DOC]
    # Checks spec serialization. A spec with shape, int32 dtype, trainable false and alias id 1
    # must serialize to a tuple of those four values in that order, and deserializing that tuple
    # must give a spec equal to the original. Returns nothing.
    shape = [1, 3]
    dtype = dtypes.int32
    trainable = False
    alias_id = 1
    spec = resource_variable_ops.VariableSpec(shape, dtype, trainable, alias_id)
    serialization = spec._serialize()
    expected_serialization = (shape, dtype, trainable, alias_id)
    self.assertEqual(serialization, expected_serialization)
    rebuilt_spec = spec._deserialize(serialization)
    self.assertEqual(rebuilt_spec, spec)

  def testRepr(self):
    # [FUNCTION DOC]
    # Checks the string representation of a spec with shape (1, 3), int32 and trainable false:
    # it must name the class and list shape, dtype, trainable flag and an unset alias id in the
    # exact expected text. Returns nothing.
    shape = (1, 3)
    dtype = dtypes.int32
    trainable = False
    spec = resource_variable_ops.VariableSpec(shape, dtype, trainable)
    spec_repr = repr(spec)
    expected_repr = ("VariableSpec(shape=(1, 3), dtype=tf.int32, "
                     "trainable=False, alias_id=None)")
    self.assertEqual(spec_repr, expected_repr)

  def testHash(self):
    # [FUNCTION DOC]
    # Checks that a spec hashes the same as the tuple of its shape, dtype, trainable flag and
    # alias id (here unset). Returns nothing.
    shape = (1, 3)
    dtype = dtypes.int32
    trainable = False
    alias_id = None
    spec = resource_variable_ops.VariableSpec(shape, dtype, trainable)
    spec_hash = hash(spec)
    expected_hash = hash((shape, dtype, trainable, alias_id))
    self.assertEqual(spec_hash, expected_hash)

  def testEquality(self):
    # [FUNCTION DOC]
    # Checks spec equality. Identical specs are equal; a spec with an alias id differs from one
    # without; specs with equal alias ids are equal; and a difference in any one of shape,
    # dtype, trainable flag or alias id makes two specs unequal. Returns nothing.
    spec = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False)
    spec2 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False)
    self.assertEqual(spec, spec2)
    # Test alias_id=None
    spec3 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False, 1)
    self.assertNotEqual(spec, spec3)
    spec4 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False, 1)
    self.assertEqual(spec3, spec4)
    # Test shape
    spec5 = resource_variable_ops.VariableSpec([1, 5], dtypes.float32, False, 1)
    self.assertNotEqual(spec4, spec5)
    # Test dtype
    spec6 = resource_variable_ops.VariableSpec([1, 3], dtypes.int32, False, 1)
    self.assertNotEqual(spec4, spec6)
    # Test trainable
    spec7 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, True, 1)
    self.assertNotEqual(spec7, spec4)
    # Test alias_id
    spec8 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False, 2)
    self.assertNotEqual(spec8, spec4)

  def testisSubtypeOf(self):
    # [FUNCTION DOC]
    # Checks the subtype relation between specs that share dtype, trainable flag and alias id: a
    # spec with a concrete shape is a subtype of one with unknown shape but not the reverse.
    # Comparing with a spec whose alias id is unset (in either direction) must raise a
    # not-implemented error. Returns nothing.
    spec = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False, 1)
    spec2 = resource_variable_ops.VariableSpec(None, dtypes.float32, False, 1)
    self.assertTrue(spec.is_subtype_of(spec2))
    self.assertFalse(spec2.is_subtype_of(spec))
    spec3 = resource_variable_ops.VariableSpec(None, dtypes.float32, False)
    with self.assertRaises(NotImplementedError):
      spec.is_subtype_of(spec3)
    with self.assertRaises(NotImplementedError):
      spec3.is_subtype_of(spec)

  def testMostSpecificCommonSupertype(self):
    # [FUNCTION DOC]
    # Checks computing the most specific common supertype of specs. Two specs differing in one
    # dimension (3 versus 2) yield a spec with that dimension unknown, both when alias ids are 1
    # and when they are unset. Mixing a spec with an alias id and one without must raise a
    # not-implemented error in either direction. Returns nothing.
    spec = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False, 1)
    spec2 = resource_variable_ops.VariableSpec([1, 2], dtypes.float32, False, 1)
    spec3 = spec.most_specific_common_supertype([spec2])
    expected_spec = resource_variable_ops.VariableSpec(
        [1, None], dtypes.float32, False, 1)
    self.assertEqual(spec3, expected_spec)

    spec4 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False)
    spec5 = resource_variable_ops.VariableSpec([1, 2], dtypes.float32, False)
    spec6 = spec4.most_specific_common_supertype([spec5])
    expected_spec = resource_variable_ops.VariableSpec(
        [1, None], dtypes.float32, False)
    self.assertEqual(spec6, expected_spec)

    with self.assertRaises(NotImplementedError):
      spec.most_specific_common_supertype([spec4])
    with self.assertRaises(NotImplementedError):
      spec4.most_specific_common_supertype([spec])


if __name__ == "__main__":
  test.main()
