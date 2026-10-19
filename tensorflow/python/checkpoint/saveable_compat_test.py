# [MODULE DOC]
# Tests for the SaveableObject compatibility layer: checkpoints written by the new tensor
# serialization must match the metadata and keys of legacy saveable-object checkpoints, and the
# force-conversion flag must behave correctly. Uses a stored legacy lookup table checkpoint from
# the test data directory. Tests mutate the module-wide conversion flag.
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

"""Tests for SaveableObject compatibility."""

import os

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint.testdata import generate_checkpoint
from tensorflow.python.eager import test
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training.saving import saveable_object


_LEGACY_TABLE_CHECKPOINT_PATH = test.test_src_dir_path(
    "python/checkpoint/testdata/table_legacy_saveable_object")


class SaveableCompatTest(test.TestCase):
  # [CLASS DOC]
  # Tests that checkpoints of a lookup table module match the stored legacy checkpoint.

  def test_lookup_table_compatibility(self):
    # [FUNCTION DOC]
    # With forced conversion disabled, writes a checkpoint of a table module and compares it with
    # the stored legacy checkpoint: the table node's first attribute name and checkpoint key, and
    # the set of stored variable names, must be equal. Then expects the legacy checkpoint to be
    # fully consumed when read into the current module.
    saveable_compat.force_checkpoint_conversion(False)

    table_module = generate_checkpoint.TableModule()
    ckpt = checkpoint.Checkpoint(table_module)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_path = os.path.join(checkpoint_directory, "ckpt")
    ckpt.write(checkpoint_path)

    # Ensure that the checkpoint metadata and keys are the same.
    legacy_metadata = checkpoint.object_metadata(_LEGACY_TABLE_CHECKPOINT_PATH)
    metadata = checkpoint.object_metadata(checkpoint_path)

    def _get_table_node(object_metadata):
      # [FUNCTION DOC]
      # HELPER: finds the proto node of the child named lookup_table under the root node (node 0)
      # of the given object metadata and returns it; returns nothing if there is no such child.
      for child in object_metadata.nodes[0].children:
        if child.local_name == "lookup_table":
          return object_metadata.nodes[child.node_id]

    table_proto = _get_table_node(metadata)
    legacy_table_proto = _get_table_node(legacy_metadata)
    self.assertAllEqual(
        [table_proto.attributes[0].name,
         table_proto.attributes[0].checkpoint_key],
        [legacy_table_proto.attributes[0].name,
         legacy_table_proto.attributes[0].checkpoint_key])
    legacy_reader = checkpoint_utils.load_checkpoint(
        _LEGACY_TABLE_CHECKPOINT_PATH)
    reader = checkpoint_utils.load_checkpoint(checkpoint_path)
    self.assertEqual(
        legacy_reader.get_variable_to_shape_map().keys(),
        reader.get_variable_to_shape_map().keys())

    # Ensure that previous checkpoint can be loaded into current table.
    ckpt.read(_LEGACY_TABLE_CHECKPOINT_PATH).assert_consumed()


class TestForceCheckpointConversionFlag(test.TestCase):
  # [CLASS DOC]
  # Tests behavior when the force-checkpoint-conversion flag is enabled. Note the flag is global
  # and these tests do not reset it afterwards.

  def test_checkpoint(self):
    # [FUNCTION DOC]
    # Enables forced conversion, inserts a key into a table, writes a checkpoint, then restores
    # into a fresh table. Expects the fresh table to miss the key (default -1) before restore and
    # return the inserted value after restore with everything consumed.
    saveable_compat.force_checkpoint_conversion()

    table_module = generate_checkpoint.TableModule()
    table_module.lookup_table.insert(3, 9)
    ckpt = checkpoint.Checkpoint(table_module)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_path = os.path.join(checkpoint_directory, "ckpt")
    ckpt.write(checkpoint_path)

    new_table_module = generate_checkpoint.TableModule()
    self.assertEqual(-1, self.evaluate(new_table_module.lookup_table.lookup(3)))

    new_ckpt = checkpoint.Checkpoint(new_table_module)
    new_ckpt.read(checkpoint_path).assert_consumed()
    self.assertEqual(9, self.evaluate(new_table_module.lookup_table.lookup(3)))

  def test_backwards_compatibility(self):
    # [FUNCTION DOC]
    # With forced conversion enabled, reads the stored legacy table checkpoint into a table that
    # had a different entry inserted. Expects full consumption, the inserted key to be gone, and
    # the legacy contents (key 2 mapping to 4) to be present.
    saveable_compat.force_checkpoint_conversion()

    table_module = generate_checkpoint.TableModule()
    table_module.lookup_table.insert(3, 9)
    self.assertEqual(9, self.evaluate(table_module.lookup_table.lookup(3)))

    ckpt = checkpoint.Checkpoint(table_module)
    ckpt.read(_LEGACY_TABLE_CHECKPOINT_PATH).assert_consumed()
    self.assertEqual(-1, self.evaluate(table_module.lookup_table.lookup(3)))
    self.assertEqual(4, self.evaluate(table_module.lookup_table.lookup(2)))

  def test_forward_compatibility(self):
    # [FUNCTION DOC]
    # Defines a deprecated-style trackable (legacy saveable with two specs named with suffixes a
    # and b) and a new-style trackable using the legacy saveable name decorator and tensor
    # serialization. Saves the new one with the flag off and then on, and each time restores the
    # deprecated one from it, expecting it to read the new values (3 and 4).

    class _MultiSpecSaveable(saveable_object.SaveableObject):
      # [CLASS DOC]
      # HELPER: test legacy saveable exposing two tensors (variables a and b of the wrapped
      # object) as specs named with suffixes a and b; restore assigns them back.

      def __init__(self, obj, name):
        # [FUNCTION DOC]
        # Stores the wrapped object, builds two save specs for its variables a and b with names
        # derived from the given name, and initializes the base class with no op.
        self.obj = obj
        specs = [
            saveable_object.SaveSpec(obj.a, "", name + "-a"),
            saveable_object.SaveSpec(obj.b, "", name + "-b")]
        super(_MultiSpecSaveable, self).__init__(None, specs, name)

      def restore(self, restored_tensors, restored_shapes):
        # [FUNCTION DOC]
        # Assigns the first restored tensor to variable a and the second to variable b of the
        # wrapped object. Shapes are ignored.
        del restored_shapes  # Unused.
        self.obj.a.assign(restored_tensors[0])
        self.obj.b.assign(restored_tensors[1])

    class DeprecatedTrackable(base.Trackable):
      # [CLASS DOC]
      # HELPER: test trackable with two variables (initial values 1 and 2) saved through the
      # legacy saveable gathering mechanism.

      def __init__(self):
        # [FUNCTION DOC]
        # Creates variables a (1.0) and b (2.0).
        self.a = variables.Variable(1.0)
        self.b = variables.Variable(2.0)

      def _gather_saveables_for_checkpoint(self):
        # [FUNCTION DOC]
        # Returns a mapping from the key foo to a factory that builds the two-spec saveable for
        # this object given a name.
        return {"foo": lambda name: _MultiSpecSaveable(self, name)}

    @saveable_compat.legacy_saveable_name("foo")
    class NewTrackable(base.Trackable):
      # [CLASS DOC]
      # HELPER: test trackable decorated with the legacy saveable name foo; two variables (3 and
      # 4) saved through tensor serialization with suffix keys so the checkpoint keys match the
      # legacy saveable's.

      def __init__(self):
        # [FUNCTION DOC]
        # Creates variables a (3.0) and b (4.0).
        self.a = variables.Variable(3.0)
        self.b = variables.Variable(4.0)

      def _serialize_to_tensors(self):
        # [FUNCTION DOC]
        # Returns a mapping from the suffix keys for a and b to the two variables.
        return {"-a": self.a, "-b": self.b}

      def _restore_from_tensors(self, restored_tensors):
        # [FUNCTION DOC]
        # Assigns restored values for the two suffix keys to variables a and b and returns the
        # grouped assign operation.
        return control_flow_ops.group(
            self.a.assign(restored_tensors["-a"]),
            self.b.assign(restored_tensors["-b"]))

    new = NewTrackable()

    # Test with the checkpoint conversion flag disabled (normal compatibility).
    saveable_compat.force_checkpoint_conversion(False)
    checkpoint_path = os.path.join(self.get_temp_dir(), "ckpt")
    checkpoint.Checkpoint(new).write(checkpoint_path)

    dep = DeprecatedTrackable()
    checkpoint.Checkpoint(dep).read(checkpoint_path).assert_consumed()
    self.assertEqual(3, self.evaluate(dep.a))
    self.assertEqual(4, self.evaluate(dep.b))

    # Now test with the checkpoint conversion flag enabled (forward compat).
    # The deprecated object will try to load from the new checkpoint.
    saveable_compat.force_checkpoint_conversion()
    checkpoint_path = os.path.join(self.get_temp_dir(), "ckpt2")
    checkpoint.Checkpoint(new).write(checkpoint_path)

    dep = DeprecatedTrackable()
    checkpoint.Checkpoint(dep).read(checkpoint_path).assert_consumed()
    self.assertEqual(3, self.evaluate(dep.a))
    self.assertEqual(4, self.evaluate(dep.b))

if __name__ == "__main__":
  test.main()
