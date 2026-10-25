# [MODULE DOC]
# Unit tests for saveable_object_util: the compatibility converter that wraps legacy
# SaveableObject-based trackables so they can be saved and restored through the newer
# serialize-to-tensors / restore-from-tensors protocol, and the helper that detects whether a
# trackable uses that newer protocol. Defines small helper saveables and trackable resource
# classes used only by these tests, and runs the TensorFlow test main when executed directly.
# No import-time side effects beyond class definitions.
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
"""Tests for saveable_object_util."""

import os

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util


class _VarSaveable(saveable_object.SaveableObject):
  # [CLASS DOC]
  # HELPER: test fixture for the saveable compatibility converter tests.
  # A legacy SaveableObject that wraps a single variable. It exposes exactly one save spec
  # holding the variable's current value, and restores by assigning the single restored tensor
  # to the variable.

  def __init__(self, var, slice_spec, name):
    # [FUNCTION DOC]
    # HELPER: constructor for the test saveable.
    # Builds one save spec from the current value of the given variable, the slice specification
    # string and the checkpoint key name, then initializes the base SaveableObject with the
    # variable as the wrapped op, that spec list and the name. Returns nothing; reads the variable.
    specs = [saveable_object.SaveSpec(var.read_value(), slice_spec, name)]
    super().__init__(var, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # HELPER: restore hook of the test saveable.
    # Assigns the first restored tensor to the wrapped variable and returns the resulting
    # assignment operation. The restored shapes argument is ignored. Side effect: mutates the
    # variable.
    return self.op.assign(restored_tensors[0])


def _create_converted_trackable(obj):
  # [FUNCTION DOC]
  # HELPER: builds a compatibility converter around a trackable for the tests below.
  # Asks the saveable utilities for the saveable factories of the given trackable, instantiates
  # each factory with its attribute name, and wraps the resulting saveables together with the
  # trackable in a SaveableCompatibilityConverter, which is returned. No exceptions are handled
  # here; errors from the utilities propagate.
  saveable_factories = saveable_object_util.saveable_objects_from_trackable(obj)
  saveables = [factory(name) for name, factory in saveable_factories.items()]
  return saveable_object_util.SaveableCompatibilityConverter(obj, saveables)


class SaveableCompatibilityConverterTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering SaveableCompatibilityConverter: converting trackables with zero, one, and
  # one renamed legacy saveable into objects that serialize to and restore from tensor mappings.
  # Uses the module-level helpers for creating the converter and the test saveable.

  def test_convert_no_saveable(self):
    # [FUNCTION DOC]
    # Scenario: a plain trackable with no saveables is converted. Expects serialization to give an
    # empty mapping, restoring from an empty mapping to succeed silently, and restoring from a
    # mapping that has any key to raise a value error saying the object could not be restored.
    t = base.Trackable()
    converter = _create_converted_trackable(t)
    self.assertEmpty(converter._serialize_to_tensors())
    converter._restore_from_tensors({})

    with self.assertRaisesRegex(ValueError, "Could not restore object"):
      converter._restore_from_tensors({"": 0})

  def test_convert_single_saveable(self):
    # [FUNCTION DOC]
    # Scenario: a trackable with one variable exposed through a legacy saveable factory named
    # "a" is converted. Expects serialization to give exactly one entry "a" whose value is the
    # variable's value (5). Restoring from an empty mapping or from a mapping with an unknown
    # key must raise a value error ("Could not restore object"). Restoring with key "a" and a
    # new value must update the variable to that value (123).

    class MyTrackable(base.Trackable):
      # [CLASS DOC]
      # Test trackable holding one float variable and exposing it through a legacy saveable
      # factory under the attribute name "a".

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the variable attribute with initial value 5.0.
        self.a = variables.Variable(5.0)

      def _gather_saveables_for_checkpoint(self):
        # [FUNCTION DOC]
        # Returns a mapping from the name "a" to a factory that builds a test saveable for the
        # variable with the given checkpoint name unchanged.
        return {"a": lambda name: _VarSaveable(self.a, "", name)}

    t = MyTrackable()
    converter = _create_converted_trackable(t)

    serialized_tensors = converter._serialize_to_tensors()
    self.assertLen(serialized_tensors, 1)
    self.assertIn("a", serialized_tensors)
    self.assertEqual(5, self.evaluate(serialized_tensors["a"]))

    with self.assertRaisesRegex(ValueError, "Could not restore object"):
      converter._restore_from_tensors({})
    with self.assertRaisesRegex(ValueError, "Could not restore object"):
      converter._restore_from_tensors({"not_a": 1.})

    self.assertEqual(5, self.evaluate(t.a))
    converter._restore_from_tensors({"a": 123.})
    self.assertEqual(123, self.evaluate(t.a))

  def test_convert_single_saveable_renamed(self):
    # [FUNCTION DOC]
    # Scenario: the legacy saveable renames its spec by appending a suffix to the key. Expects
    # serialization to produce one entry under the renamed key (value 15), restoring with the
    # old key to raise a value error ("Could not restore object"), and restoring with the
    # renamed key to assign the new value (456) to the variable.

    class MyTrackable(base.Trackable):
      # [CLASS DOC]
      # Test trackable with one float variable whose saveable uses a renamed checkpoint key
      # (the attribute name plus a value suffix).

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the variable attribute with initial value 15.0.
        self.a = variables.Variable(15.0)

      def _gather_saveables_for_checkpoint(self):
        # [FUNCTION DOC]
        # Returns a mapping from "a" to a factory building a test saveable whose spec name is the
        # given name with a value suffix appended.
        return {"a": lambda name: _VarSaveable(self.a, "", name + "-value")}

    t = MyTrackable()
    converter = _create_converted_trackable(t)

    serialized_tensors = converter._serialize_to_tensors()

    self.assertLen(serialized_tensors, 1)
    self.assertEqual(15, self.evaluate(serialized_tensors["a-value"]))

    with self.assertRaisesRegex(ValueError, "Could not restore object"):
      converter._restore_from_tensors({"a": 1.})

    self.assertEqual(15, self.evaluate(t.a))
    converter._restore_from_tensors({"a-value": 456.})
    self.assertEqual(456, self.evaluate(t.a))


class _MultiSpecSaveable(saveable_object.SaveableObject):
  # [CLASS DOC]
  # HELPER: test fixture saveable producing two save specs (for attributes a and b of an object).
  # KNOWN ISSUE: this class is defined but never used by any test in this file (dead code).
  # Restoring assigns the first restored tensor to attribute a and the second to attribute b.

  def __init__(self, obj, name):
    # [FUNCTION DOC]
    # HELPER: constructor for the unused multi-spec test saveable.
    # Stores the object and creates two save specs from its attributes a and b, with names
    # derived from the given name by suffixes; initializes the base class with no op.
    self.obj = obj
    specs = [
        saveable_object.SaveSpec(obj.a, "", name + "-a"),
        saveable_object.SaveSpec(obj.b, "", name + "-b")]
    super(_MultiSpecSaveable, self).__init__(None, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # HELPER: restore hook of the unused multi-spec test saveable.
    # Assigns the first restored tensor to attribute a and the second to attribute b of the
    # wrapped object; the shapes argument is ignored. Returns nothing.
    del restored_shapes  # Unused.
    self.obj.a.assign(restored_tensors[0])
    self.obj.b.assign(restored_tensors[1])


class State(resource.TrackableResource):
  # [CLASS DOC]
  # HELPER: test fixture. A trackable resource wrapping a scalar float32 variable resource
  # handle, with read and assign operations. Subclasses add the checkpointing protocol under
  # test (legacy saveable or serialize-to-tensors).

  def __init__(self, initial_value):
    # [FUNCTION DOC]
    # Initializes the trackable resource, remembers the initial value, and immediately
    # initializes the underlying resource with it. Side effect: creates a resource handle and
    # assigns the value.
    super().__init__()
    self._initial_value = initial_value
    self._initialize()

  def _create_resource(self):
    # [FUNCTION DOC]
    # Creates and returns a new scalar float32 variable resource handle with an anonymous
    # shared name and a fixed op name, in the default container.
    return gen_resource_variable_ops.var_handle_op(
        shape=[],
        dtype=dtypes.float32,
        shared_name=context.anonymous_name(),
        name="StateVar",
        container="")

  def _initialize(self):
    # [FUNCTION DOC]
    # Assigns the stored initial value to the variable resource. Returns nothing.
    gen_resource_variable_ops.assign_variable_op(self.resource_handle,
                                                 self._initial_value)

  def _destroy_resource(self):
    # [FUNCTION DOC]
    # Destroys the variable resource, ignoring the error if it no longer exists. Returns nothing.
    gen_resource_variable_ops.destroy_resource_op(self.resource_handle,
                                                  ignore_lookup_error=True)

  def read(self):
    # [FUNCTION DOC]
    # Returns the current float32 value of the variable resource as a tensor.
    return gen_resource_variable_ops.read_variable_op(self.resource_handle,
                                                      dtypes.float32)

  def assign(self, value):
    # [FUNCTION DOC]
    # Assigns the given value to the variable resource. Returns nothing; mutates resource state.
    gen_resource_variable_ops.assign_variable_op(self.resource_handle, value)


class _StateSaveable(saveable_object.SaveableObject):
  # [CLASS DOC]
  # HELPER: test fixture legacy saveable for a State object. Exposes one save spec with the
  # state's current value and restores by assigning the restored tensor back to the state.

  def __init__(self, obj, name):
    # [FUNCTION DOC]
    # HELPER: constructor. Creates one save spec from the state's current value under the given
    # name, stores the state, and initializes the base class with the state and that spec.
    spec = saveable_object.SaveSpec(obj.read(), "", name)
    self.obj = obj
    super(_StateSaveable, self).__init__(obj, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # HELPER: restore hook. Assigns the first restored tensor to the wrapped state; the shapes
    # argument is ignored. Returns nothing; mutates the state.
    del restored_shapes  # Unused.
    self.obj.assign(restored_tensors[0])


class SaveableState(State):
  # [CLASS DOC]
  # HELPER: test fixture. A State that checkpoints through the legacy saveable mechanism, using
  # a saveable factory registered under the name "value".

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Returns a mapping from "value" to a factory that builds a state saveable for this object
    # with the given checkpoint name.
    return {
        "value": lambda name: _StateSaveable(self, name)
    }


class TrackableState(State):
  # [CLASS DOC]
  # HELPER: test fixture. A State that checkpoints through the newer serialize-to-tensors and
  # restore-from-tensors protocol, using the key "value".

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # Returns a mapping with the single key "value" holding the current state value tensor.
    return {
        "value": self.read()
    }

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # Assigns the tensor stored under "value" in the given mapping to the state and returns the
    # result of the assignment. Raises a key error if "value" is missing.
    return self.assign(restored_tensors["value"])


class SaveableCompatibilityEndToEndTest(test.TestCase):
  # [CLASS DOC]
  # Test case that checks end to end, through real checkpoint files, that the legacy saveable
  # path and the new tensor serialization path are interchangeable.

  def test_checkpoint_comparison(self):
    # [FUNCTION DOC]
    # Scenario: a legacy-saveable State (value 5) and a new-protocol State (value 10) are saved
    # in one checkpoint under keys a and b. A new checkpoint object with the roles swapped reads
    # it; asserts everything is consumed and that the values were swapped. Then a fresh legacy
    # State (value 0) is wrapped by the compatibility converter and restored from the same file
    # first via key a (expects 5) and then via key b (expects 10), allowing partial matches.
    # Side effect: writes a checkpoint file in the test temp directory.
    saveable_state = SaveableState(5.)
    trackable_state = TrackableState(10.)

    # First test that SaveableState and TrackableState are equivalent by
    # saving a checkpoint with both objects and swapping values.

    self.assertEqual(5, self.evaluate(saveable_state.read()))
    self.assertEqual(10, self.evaluate(trackable_state.read()))

    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    checkpoint.Checkpoint(a=saveable_state, b=trackable_state).write(ckpt_path)

    status = checkpoint.Checkpoint(b=saveable_state,
                                   a=trackable_state).read(ckpt_path)
    status.assert_consumed()

    self.assertEqual(10, self.evaluate(saveable_state.read()))
    self.assertEqual(5, self.evaluate(trackable_state.read()))

    # Test that the converted SaveableState is compatible with the checkpoint
    # saved above.
    to_convert = SaveableState(0.0)

    converted_saveable_state = _create_converted_trackable(to_convert)

    checkpoint.Checkpoint(a=converted_saveable_state).read(
        ckpt_path).assert_existing_objects_matched().expect_partial()
    self.assertEqual(5, self.evaluate(to_convert.read()))

    checkpoint.Checkpoint(b=converted_saveable_state).read(
        ckpt_path).assert_existing_objects_matched().expect_partial()
    self.assertEqual(10, self.evaluate(to_convert.read()))


class HasSerializeToTensorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the helper that reports whether a trackable uses the serialize-to-tensors
  # protocol rather than legacy saveables.
  def test_has_serialize_to_tensor(self):
    # [FUNCTION DOC]
    # Scenario: three trackable classes are defined: one overriding serialize-to-tensors, one
    # overriding nothing, and a subclass of the first that also defines the legacy saveable
    # gathering method. Expects the helper to return true for the first, false for the second,
    # and false for the subclass because it defines the legacy method itself.
    class ReturnsTrue(base.Trackable):
      # [CLASS DOC]
      # Trackable that overrides serialize-to-tensors, returning an empty mapping.
      def _serialize_to_tensors(self):
        # [FUNCTION DOC]
        # Returns an empty mapping of tensors.
        return {}

    class ReturnsFalse(base.Trackable):
      # [CLASS DOC]
      # Trackable that overrides neither checkpointing method.
      pass

    class SubclassReturnsFalse(ReturnsTrue):
      # [CLASS DOC]
      # Subclass that inherits serialize-to-tensors but also defines the legacy saveable method.
      def _gather_saveables_for_checkpoint(self):
        # [FUNCTION DOC]
        # Returns an empty mapping of saveable factories.
        return {}

    self.assertTrue(saveable_object_util.trackable_has_serialize_to_tensor(
        ReturnsTrue()))
    self.assertFalse(saveable_object_util.trackable_has_serialize_to_tensor(
        ReturnsFalse()))

    # This should return False, because even though its parent class has
    # `_serialize_to_tensors`, the class itself defines
    # `_gather_saveables_for_checkpoint`.
    self.assertFalse(saveable_object_util.trackable_has_serialize_to_tensor(
        SubclassReturnsFalse()))


if __name__ == "__main__":
  test.main()
