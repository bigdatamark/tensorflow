# [MODULE DOC]
# Unit tests for AutoTrackable dependency tracking: attribute assignment and deletion, tracked
# lists and nested containers, detection of out-of-band edits that make an object unsaveable, and
# assertion helpers working on tracked containers. Defines one test class and runs with the TF
# test runner when executed as a script. No public API.
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

import numpy as np
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest


class InterfaceTests(test.TestCase):
  # [CLASS DOC]
  # Test case exercising the public behavior of AutoTrackable objects (dependency creation,
  # removal, list wrapping and checkpoint saving errors). No shared fixtures; each test builds
  # its own object graph.

  def testMultipleAssignment(self):
    # [FUNCTION DOC]
    # Test: assigns a trackable to an attribute, reassigns the same value (no effect), then
    # explicitly tracking a different object under the existing name raises a value error saying
    # it is already declared. Assigning another object through attribute syntax is allowed, and
    # tracking with overwrite succeeds. Asserts the dependency lookup and the children mapping
    # both return the newest object.
    root = autotrackable.AutoTrackable()
    root.leaf = autotrackable.AutoTrackable()
    root.leaf = root.leaf
    duplicate_name_dep = autotrackable.AutoTrackable()
    with self.assertRaisesRegex(ValueError, "already declared"):
      root._track_trackable(duplicate_name_dep, name="leaf")
    # No error; we're overriding __setattr__, so we can't really stop people
    # from doing this while maintaining backward compatibility.
    root.leaf = duplicate_name_dep
    root._track_trackable(duplicate_name_dep, name="leaf", overwrite=True)
    self.assertIs(duplicate_name_dep, root._lookup_dependency("leaf"))
    self.assertIs(duplicate_name_dep, root._trackable_children()["leaf"])

  def testRemoveDependency(self):
    # [FUNCTION DOC]
    # Test: assigns a child, checks one child and one unconditional dependency exist, deletes the
    # attribute and asserts the attribute is gone and both counts drop to zero; then reassigns a
    # child and asserts the dependency is tracked again.
    root = autotrackable.AutoTrackable()
    root.a = autotrackable.AutoTrackable()
    self.assertEqual(1, len(root._trackable_children()))
    self.assertEqual(1, len(root._unconditional_checkpoint_dependencies))
    self.assertIs(root.a, root._trackable_children()["a"])
    del root.a
    self.assertFalse(hasattr(root, "a"))
    self.assertEqual(0, len(root._trackable_children()))
    self.assertEqual(0, len(root._unconditional_checkpoint_dependencies))
    root.a = autotrackable.AutoTrackable()
    self.assertEqual(1, len(root._trackable_children()))
    self.assertEqual(1, len(root._unconditional_checkpoint_dependencies))
    self.assertIs(root.a, root._trackable_children()["a"])

  def testListBasic(self):
    # [FUNCTION DOC]
    # Test: assigns a list holding a trackable and appends another one after assignment. Asserts
    # both appear in the listed objects of the root and inside the tracked list child named l.
    a = autotrackable.AutoTrackable()
    b = autotrackable.AutoTrackable()
    a.l = [b]
    c = autotrackable.AutoTrackable()
    a.l.append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertIn("l", a._trackable_children())
    direct_a_dep = a._trackable_children()["l"]
    self.assertIn(b, direct_a_dep)
    self.assertIn(c, direct_a_dep)

  @test_util.run_in_graph_and_eager_modes
  def testMutationDirtiesList(self):
    # [FUNCTION DOC]
    # Test (runs in graph and eager modes): inserts a trackable at the front of a tracked list,
    # which replaces an existing element position, then tries to save a checkpoint and asserts a
    # value error saying a list element was replaced.
    a = autotrackable.AutoTrackable()
    b = autotrackable.AutoTrackable()
    a.l = [b]
    c = autotrackable.AutoTrackable()
    a.l.insert(0, c)
    checkpoint = util.Checkpoint(a=a)
    with self.assertRaisesRegex(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testOutOfBandEditDirtiesList(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): keeps an outside reference to a list, assigns it to a tracked
    # attribute, then appends to the original list through the held reference. Saving a
    # checkpoint must raise a value error saying the wrapped list was modified.
    a = autotrackable.AutoTrackable()
    b = autotrackable.AutoTrackable()
    held_reference = [b]
    a.l = held_reference
    c = autotrackable.AutoTrackable()
    held_reference.append(c)
    checkpoint = util.Checkpoint(a=a)
    with self.assertRaisesRegex(ValueError, "The wrapped list was modified"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testNestedLists(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): builds nested lists of trackables, appends to inner lists
    # (including a plain integer that must not be listed as an object) and asserts all trackables
    # are discovered. Saves checkpoints successfully after adding a no-dependency list and
    # mutating it, then replaces an inner element and asserts that saving raises a value error
    # about a replaced list element.
    a = autotrackable.AutoTrackable()
    a.l = []
    b = autotrackable.AutoTrackable()
    a.l.append([b])
    c = autotrackable.AutoTrackable()
    a.l[0].append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    a.l[0].append(1)
    d = autotrackable.AutoTrackable()
    a.l[0].append(d)
    a_deps = util.list_objects(a)
    self.assertIn(d, a_deps)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertNotIn(1, a_deps)
    e = autotrackable.AutoTrackable()
    f = autotrackable.AutoTrackable()
    a.l1 = [[], [e]]
    a.l1[0].append(f)
    a_deps = util.list_objects(a)
    self.assertIn(e, a_deps)
    self.assertIn(f, a_deps)
    checkpoint = util.Checkpoint(a=a)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    a.l[0].append(data_structures.NoDependency([]))
    a.l[0][-1].append(5)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    # Dirtying the inner list means the root object is unsaveable.
    a.l[0][1] = 2
    with self.assertRaisesRegex(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testAssertions(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): stores a mapping with a list of zero arrays and checks that
    # flattening, close-equality assertions and structure mapping work on the tracked container as
    # on plain data. Also stores a mapping of tensors and asserts that evaluating it matches the
    # expected ones and zeros arrays.
    a = autotrackable.AutoTrackable()
    a.l = {"k": [np.zeros([2, 2])]}
    self.assertAllEqual(nest.flatten({"k": [np.zeros([2, 2])]}),
                        nest.flatten(a.l))
    self.assertAllClose({"k": [np.zeros([2, 2])]}, a.l)
    nest.map_structure(self.assertAllClose, a.l, {"k": [np.zeros([2, 2])]})
    a.tensors = {"k": [array_ops.ones([2, 2]), array_ops.zeros([3, 3])]}
    self.assertAllClose({"k": [np.ones([2, 2]), np.zeros([3, 3])]},
                        self.evaluate(a.tensors))


if __name__ == "__main__":
  test.main()
