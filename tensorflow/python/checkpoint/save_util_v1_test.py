# [MODULE DOC]
# Unit tests for the v1 checkpoint serialization (save_util_v1). Registers a test checkpoint saver
# at import time for a fixture trackable class, then checks serialization with and without an
# object map.
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
# =============================================================================
"""Tests for the checkpoint/util.py."""

from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.eager import test
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import object_identity


class TrackableWithRegisteredSaver(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # HELPER: test fixture trackable.
  # Empty auto-trackable subclass whose instances are claimed by the checkpoint saver registered
  # below under the name RegisteredSaver.
  pass


registration.register_checkpoint_saver(
    name="RegisteredSaver",
    predicate=lambda x: isinstance(x, TrackableWithRegisteredSaver),
    save_fn=lambda trackables, file_prefix: [],
    restore_fn=lambda trackables, merged_prefix: None)


class SerializationTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the v1 serialize_gathered_objects function: a root with one variable and one
  # object using a registered saver.

  def test_serialize_gathered_objects(self):
    # [FUNCTION DOC]
    # Builds a root with a variable and a registered-saver child, serializes it, and expects
    # exactly one saveable object wrapping the variable, and registered savers mapping the custom
    # saver name to the child under its object name.
    root = autotrackable.AutoTrackable()
    root.v = variables.Variable(1.0)
    root.registered = TrackableWithRegisteredSaver()
    named_saveable_objects, _, _, registered_savers = (
        save_util_v1.serialize_gathered_objects(
            graph_view.ObjectGraphView(root)))

    self.assertLen(named_saveable_objects, 1)
    self.assertIs(named_saveable_objects[0].op, root.v)
    self.assertDictEqual(
        {"Custom.RegisteredSaver": {"registered": root.registered}},
        registered_savers)

  def test_serialize_gathered_objects_with_map(self):
    # [FUNCTION DOC]
    # Same setup as the basic test but passes an object map replacing the variable and the
    # registered child with copies. Expects the saveable to wrap the copied variable (not the
    # original) and the registered savers entry to hold the copied child.
    root = autotrackable.AutoTrackable()
    root.v = variables.Variable(1.0)
    root.registered = TrackableWithRegisteredSaver()

    copy_of_registered = TrackableWithRegisteredSaver()
    copy_of_v = variables.Variable(1.0)
    object_map = object_identity.ObjectIdentityDictionary()
    object_map[root.registered] = copy_of_registered
    object_map[root.v] = copy_of_v

    named_saveable_objects, _, _, registered_savers = (
        save_util_v1.serialize_gathered_objects(
            graph_view.ObjectGraphView(root), object_map))

    self.assertLen(named_saveable_objects, 1)
    self.assertIsNot(named_saveable_objects[0].op, root.v)
    self.assertIs(named_saveable_objects[0].op, copy_of_v)

    ret_value = registered_savers["Custom.RegisteredSaver"]["registered"]
    self.assertIsNot(root.registered, ret_value)
    self.assertIs(copy_of_registered, ret_value)


if __name__ == "__main__":
  test.main()
