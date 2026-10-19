# [MODULE DOC]
# Tests for restore.py: the saveable-name extraction helper and restore_nodes, which restores
# selected checkpoint nodes into trackable objects. Covers error cases (bad argument types,
# missing nodes, null trackables, value mismatches, size mismatches) and successful partial
# restores for both serialize-to-tensors trackables and legacy saveable-object trackables.
# Saves checkpoints in the test temp directory. Runs with the TensorFlow test runner.
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
"""Tests for restore.py."""

import os

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import restore
from tensorflow.python.eager import test
from tensorflow.python.module import module
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.training.saving import saveable_object


class ExtractSaveablenameTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the saveable-name extraction helper and for restore_nodes in the restore
  # module. No shared fixtures; each test builds its own trackables and checkpoints.

  def test_standard_saveable_name(self):
    # [FUNCTION DOC]
    # Checks the extraction helper keeps the key up to and including the attributes segment:
    # a key with a trailing number and a key that already ends at the segment both yield the
    # prefix ending in the attributes marker with a slash.
    self.assertEqual(
        "object_path/.ATTRIBUTES/",
        restore._extract_saveable_name("object_path/.ATTRIBUTES/123"))
    self.assertEqual(
        "object/path/ATTRIBUTES/.ATTRIBUTES/",
        restore._extract_saveable_name("object/path/ATTRIBUTES/.ATTRIBUTES/"))

  def test_restore_nodes_error_cases_high_level(self):
    # [FUNCTION DOC]
    # Saves a root with one leaf, then expects restore_nodes to raise a value error when given
    # a list instead of a dictionary, when a node id (3) is not in the checkpoint, and when the
    # trackable for a node is null.
    root = autotrackable.AutoTrackable()
    root.leaf = autotrackable.AutoTrackable()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = autotrackable.AutoTrackable()
    root2.leaf = autotrackable.AutoTrackable()

    with self.assertRaisesRegex(
        ValueError,
        "Expecting a dictionary of node_id to Trackable for nodes_to_restore."):
      restore.restore_nodes(root_save_path, [0, 1])

    with self.assertRaisesRegex(
        ValueError,
        "The expected node_id: 3 to Trackable <.*?> to restore does not exist "
        "in the checkpoint."):
      restore.restore_nodes(root_save_path, {3: root2})

    with self.assertRaisesRegex(
        ValueError,
        "Expecting a valid Trackable to node_id: 0 but got trackable: None."):
      restore.restore_nodes(root_save_path, {0: None})

  def test_restore_nodes_error_cases_trackable_ckpt_view_mismatch(self):
    # [FUNCTION DOC]
    # Saves a trackable holding no tensors, then tries to restore it into a trackable that
    # serializes a tensor. Expects a value error saying the trackable expects checkpointed
    # values but the checkpoint has none for node 0.

    class MyTrackable(base.Trackable):
      # [CLASS DOC]
      # Test trackable with a child module and no tensors to checkpoint.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores an empty module as attribute a.
        self.a = module.Module()

    class MyTrackable2(base.Trackable):
      # [CLASS DOC]
      # Test trackable holding a variable and serializing one tensor named a.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores a variable with value 5 as attribute a.
        self.a = variables.Variable(5.0)

      def _serialize_to_tensors(self):
        # [FUNCTION DOC]
        # Returns a mapping with one entry, a, holding a new variable of value 5.
        return {"a": variables.Variable(5.0)}

    root = MyTrackable()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackable2()
    with self.assertRaisesRegex(
        ValueError,
        "Trackable <.*?> expects checkpointed values but checkpoint does not "
        "contain serialized tensors for node_id: 0."):
      restore.restore_nodes(root_save_path, {0: root2})

  def test_restore_nodes_has_serialize_to_tensor(self):
    # [FUNCTION DOC]
    # Saves a root trackable with a leaf, each holding a variable of 5. Restores only node 0 into
    # a fresh pair whose root variable was set to 3: root becomes 5, leaf is unchanged. Then
    # restores only node 1 into another fresh pair: leaf becomes 5, root is unchanged.

    class MyTrackable(base.Trackable):
      # [CLASS DOC]
      # Test trackable with one variable that serializes to and restores from tensors.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores a variable with value 5 as attribute a.
        self.a = variables.Variable(5.0)

      def _restore_from_tensors(self, restored_tensors):
        # [FUNCTION DOC]
        # Assigns the restored tensor named a to the variable and returns the assign result.
        return self.a.assign(restored_tensors["a"])

      def _serialize_to_tensors(self):
        # [FUNCTION DOC]
        # Returns a mapping from name a to the variable.
        return {"a": self.a}

    root = MyTrackable()
    leaf = MyTrackable()
    root._track_trackable(leaf, "leaf")
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackable()
    leaf2 = MyTrackable()
    root2._track_trackable(leaf2, "leaf")
    root2.a.assign(3.0)

    # Restore root
    restore.restore_nodes(root_save_path, {0: root2})
    self.assertEqual(root2.a.numpy(), 5.0)  # Restored from 3.0 to 5.0
    self.assertEqual(leaf2.a.numpy(), 5.0)  # Unchanged

    root3 = MyTrackable()
    leaf3 = MyTrackable()
    root3._track_trackable(leaf3, "leaf")
    leaf3.a.assign(3.0)

    # Restore leaf
    restore.restore_nodes(root_save_path, {1: leaf3})
    self.assertEqual(root3.a.numpy(), 5.0)  # Unchanged
    self.assertEqual(leaf3.a.numpy(), 5.0)  # Restored from 3.0 to 5.0.

  def test_restore_nodes_with_different_number_of_serialized_to_tensors(self):
    # [FUNCTION DOC]
    # Saves a trackable with one tensor and tries to restore into one with two, expecting a value
    # error about the sizes (2 versus 1). Then saves the two-tensor one and restores into the
    # one-tensor trackable, expecting the mirrored error (1 versus 2).

    class MyTrackableA(base.Trackable):
      # [CLASS DOC]
      # Test trackable with one serialized variable named a.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores a variable with value 5 as attribute a.
        self.a = variables.Variable(5.0)

      def _restore_from_tensors(self, restored_tensors):
        # [FUNCTION DOC]
        # Assigns the restored tensor named a to the variable and returns the assign result.
        return self.a.assign(restored_tensors["a"])

      def _serialize_to_tensors(self):
        # [FUNCTION DOC]
        # Returns a mapping from name a to the variable.
        return {"a": self.a}

    class MyTrackableAandB(base.Trackable):
      # [CLASS DOC]
      # Test trackable with two serialized variables named a and b.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores variables a (value 5) and b (value 6).
        self.a = variables.Variable(5.0)
        self.b = variables.Variable(6.0)

      def _restore_from_tensors(self, restored_tensors):
        # [FUNCTION DOC]
        # Assigns restored tensors a and b to the variables and returns the grouped operation.
        return control_flow_ops.group(
            self.a.assign(restored_tensors["a"]),
            self.b.assign(restored_tensors["b"])
        )

      def _serialize_to_tensors(self):
        # [FUNCTION DOC]
        # Returns a mapping from names a and b to the two variables.
        return {"a": self.a, "b": self.b}

    root = MyTrackableA()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableAandB()

    with self.assertRaisesRegex(
        ValueError,
        "Size for serialized_tensors for Trackable: 2 did not match size for "
        "serialized_tensors for checkpoint: 1."):
      restore.restore_nodes(root_save_path, {0: root2})

    root = MyTrackableAandB()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableA()

    with self.assertRaisesRegex(
        ValueError,
        "Size for serialized_tensors for Trackable: 1 did not match size for "
        "serialized_tensors for checkpoint: 2."):
      restore.restore_nodes(root_save_path, {0: root2})

  def test_restore_nodes_not_serialize_to_tensor(self):
    # [FUNCTION DOC]
    # Same partial-restore scenario as the serialize-to-tensors test, but the trackable uses the
    # legacy saveable-object mechanism. Restoring node 0 updates only the root variable (3 to 5),
    # restoring node 1 updates only the leaf variable.

    class _VarSaveable(saveable_object.SaveableObject):
      # [CLASS DOC]
      # Test saveable object wrapping the variable of a trackable under a single save spec.

      def __init__(self, obj, name):
        # [FUNCTION DOC]
        # Remembers the owning trackable and builds one save spec for its variable named with
        # the given name plus a suffix, then initializes the base class with no op.
        self.obj = obj
        specs = [saveable_object.SaveSpec(obj.a, "", name + "-a")]
        super(_VarSaveable, self).__init__(None, specs, name)

      def restore(self, restored_tensors, restored_shapes):
        # [FUNCTION DOC]
        # Assigns the first restored tensor to the owner's variable; shapes are ignored.
        del restored_shapes  # Unused.
        self.obj.a.assign(restored_tensors[0])

    class MyTrackable(base.Trackable):
      # [CLASS DOC]
      # Test trackable with one variable exposed via a saveable factory.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores a variable with value 5 as attribute a.
        self.a = variables.Variable(5.0)

      def _gather_saveables_for_checkpoint(self):
        # [FUNCTION DOC]
        # Returns a mapping from name a to a factory building the variable saveable for a name.
        return {"a": lambda name: _VarSaveable(self, name)}

    root = MyTrackable()
    leaf = MyTrackable()
    root._track_trackable(leaf, "leaf")
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackable()
    leaf2 = MyTrackable()
    root2._track_trackable(leaf2, "leaf")
    root2.a.assign(3.0)

    # Restore root
    restore.restore_nodes(root_save_path, {0: root2})
    self.assertEqual(root2.a.numpy(), 5.0)  # Restored from 3.0 to 5.0
    self.assertEqual(leaf2.a.numpy(), 5.0)  # Unchanged

    root3 = MyTrackable()
    leaf3 = MyTrackable()
    root3._track_trackable(leaf3, "leaf")
    leaf3.a.assign(3.0)

    # Restore leaf
    restore.restore_nodes(root_save_path, {1: leaf3})
    self.assertEqual(root3.a.numpy(), 5.0)  # Unchanged
    self.assertEqual(leaf3.a.numpy(), 5.0)  # Restored from 3.0 to 5.0.

  def test_restore_nodes_not_serialize_to_tensor_error_cases(self):
    # [FUNCTION DOC]
    # Error cases for legacy saveable trackables: restoring a checkpoint with no tensors into a
    # trackable with saveables raises "expects checkpointed values"; restoring a one-saveable
    # checkpoint into a two-saveable trackable, and the reverse, each raise a value error about
    # mismatched saveable counts (2 versus 1, and 1 versus 2).

    class _VarSaveable(saveable_object.SaveableObject):
      # [CLASS DOC]
      # Test saveable object wrapping the variable of a trackable under a single save spec.

      def __init__(self, obj, name):
        # [FUNCTION DOC]
        # Remembers the owning trackable and builds one save spec for its variable, then
        # initializes the base class with no op.
        self.obj = obj
        specs = [saveable_object.SaveSpec(obj.a, "", name + "-a")]
        super(_VarSaveable, self).__init__(None, specs, name)

      def restore(self, restored_tensors, restored_shapes):
        # [FUNCTION DOC]
        # Assigns the first restored tensor to the owner's variable; shapes are ignored.
        del restored_shapes  # Unused.
        self.obj.a.assign(restored_tensors[0])

    class MyTrackable(base.Trackable):
      # [CLASS DOC]
      # Test trackable with a child module and no tensors to checkpoint.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores an empty module as attribute a.
        self.a = module.Module()

    class MyTrackableWithSingleSaveable(base.Trackable):
      # [CLASS DOC]
      # Test trackable exposing one saveable factory named foo.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores a variable with value 1 as attribute a.
        self.a = variables.Variable(1.0)

      def _gather_saveables_for_checkpoint(self):
        # [FUNCTION DOC]
        # Returns a mapping from foo to a factory building the variable saveable.
        return {"foo": lambda name: _VarSaveable(self, name)}

    class MyTrackableWithMultiSaveables(base.Trackable):
      # [CLASS DOC]
      # Test trackable exposing two saveable factories named foo and bar.

      def __init__(self):
        # [FUNCTION DOC]
        # Stores a variable with value 1 as attribute a.
        self.a = variables.Variable(1.0)

      def _gather_saveables_for_checkpoint(self):
        # [FUNCTION DOC]
        # Returns a mapping from foo and bar to factories each building a variable saveable.
        return {
            "foo": lambda name: _VarSaveable(self, name),
            "bar": lambda name: _VarSaveable(self, name)
        }

    root = MyTrackable()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableWithMultiSaveables()
    with self.assertRaisesRegex(
        ValueError,
        "Trackable <.*?> expects checkpointed values but checkpoint does not "
        "contain serialized tensors for node_id: 0."):
      restore.restore_nodes(root_save_path, {0: root2})

    root = MyTrackableWithSingleSaveable()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableWithMultiSaveables()
    with self.assertRaisesRegex(
        ValueError,
        "Size for saveable_objects for Trackable: 2 did not match the size for "
        "serialized_tensors for checkpoint: 1."):
      restore.restore_nodes(root_save_path, {0: root2})

    root = MyTrackableWithMultiSaveables()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableWithSingleSaveable()
    with self.assertRaisesRegex(
        ValueError,
        "Size for saveable_objects for Trackable: 1 did not match the size for "
        "serialized_tensors for checkpoint: 2."):
      restore.restore_nodes(root_save_path, {0: root2})

if __name__ == "__main__":
  test.main()
