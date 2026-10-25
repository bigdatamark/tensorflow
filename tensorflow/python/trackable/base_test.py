# [MODULE DOC]
# Unit tests for the Trackable base class interface (dependency tracking, variable creation with
# custom getters, and checkpoint save/restore behavior with get_config). Test-only module; run
# via the TensorFlow test runner with eager execution enabled when executed as a script.

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import os
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.trackable import base


class InterfaceTests(test.TestCase):
  # [CLASS DOC]
  # Tests for the basic Trackable interface in the base module: dependency tracking with and
  # without overwrite, custom-getter variable creation, and checkpoint behavior of objects that
  # expose a configuration method. Uses the standard TensorFlow test case with temporary dirs.

  def testOverwrite(self):
    # [FUNCTION DOC]
    # Scenario: a root Trackable tracks a leaf under the name "leaf"; the children mapping must
    # hold exactly that one entry. Tracking a different object under the same name must raise a
    # value error, while doing so with overwrite enabled must succeed and replace the child, so the
    # children mapping again has one entry named "leaf" pointing to the new object.
    root = base.Trackable()
    leaf = base.Trackable()
    root._track_trackable(leaf, name="leaf")
    (current_name, current_dependency), = root._trackable_children().items()
    self.assertIs(leaf, current_dependency)
    self.assertEqual("leaf", current_name)
    duplicate_name_dep = base.Trackable()
    with self.assertRaises(ValueError):
      root._track_trackable(duplicate_name_dep, name="leaf")
    root._track_trackable(duplicate_name_dep, name="leaf", overwrite=True)
    (current_name, current_dependency), = root._trackable_children().items()
    self.assertIs(duplicate_name_dep, current_dependency)
    self.assertEqual("leaf", current_name)

  def testAddVariableOverwrite(self):
    # [FUNCTION DOC]
    # Scenario: a variable "v" is added to a root via the custom-getter helper; the object list of
    # the root is then the root and that variable. In a fresh graph, re-adding "v" with overwrite
    # enabled replaces the variable (object list is root and the new variable). In another fresh
    # graph, re-adding with overwrite disabled must raise a value error mentioning that the name is
    # already declared as a dependency.
    root = base.Trackable()
    a = root._add_variable_with_custom_getter(
        name="v", shape=[], getter=variable_scope.get_variable)
    self.assertEqual([root, a], util.list_objects(root))
    with ops.Graph().as_default():
      b = root._add_variable_with_custom_getter(
          name="v", shape=[], overwrite=True,
          getter=variable_scope.get_variable)
      self.assertEqual([root, b], util.list_objects(root))
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  "already declared as a dependency"):
        root._add_variable_with_custom_getter(
            name="v", shape=[], overwrite=False,
            getter=variable_scope.get_variable)

  def testAssertConsumedWithUnusedPythonState(self):
    # [FUNCTION DOC]
    # Scenario: an object with a get_config callable returning an empty mapping is saved in a
    # checkpoint under a temporary directory; a checkpoint holding a plain Trackable is then
    # restored from it and must pass the assert-consumed check, showing unused Python state is not
    # reported as unconsumed. Side effect: writes a checkpoint file to the test temp directory.
    has_config = base.Trackable()
    has_config.get_config = lambda: {}
    saved = util.Checkpoint(obj=has_config)
    save_path = saved.save(os.path.join(self.get_temp_dir(), "ckpt"))
    restored = util.Checkpoint(obj=base.Trackable())
    restored.restore(save_path).assert_consumed()

  def testBuggyGetConfig(self):
    # [FUNCTION DOC]
    # Scenario: a Trackable subclass whose get_config returns a non-serializable object is saved in
    # a checkpoint. Saving is expected to complete without raising, i.e. a faulty configuration
    # method must not break checkpoint saving. Side effect: writes a checkpoint to the temp dir.

    class NotSerializable(object):
      # [CLASS DOC]
      # Local placeholder class with no members; its instances cannot be serialized to JSON and
      # are returned by the faulty get_config below.
      pass

    class GetConfigRaisesError(base.Trackable):
      # [CLASS DOC]
      # Local Trackable subclass whose configuration method returns a non-serializable object,
      # used to check that checkpoint saving tolerates a broken get_config.

      def get_config(self):
        # [FUNCTION DOC]
        # KNOWN ISSUE: despite the name of the enclosing class ("raises error"), this does not
        # raise; it returns an unserializable value. Takes no inputs; no side effects.
        # Returns an instance of the non-serializable placeholder class instead of a mapping.
        return NotSerializable()

    util.Checkpoint(obj=GetConfigRaisesError()).save(
        os.path.join(self.get_temp_dir(), "ckpt"))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
