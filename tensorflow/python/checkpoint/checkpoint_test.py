# [MODULE DOC]
# Tests for object-based checkpointing (Checkpoint save/restore/write/read): add-variable helper,
# custom saveables, deferred restoration, dependency graphs, checkpoint state, options and write
# callbacks, async checkpointing, path-like arguments, tensor serialization and templates.
# Runs eager execution at startup when executed as a script.
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

import copy
import os
import pathlib
import sys
import weakref

from absl.testing import parameterized

from tensorflow.python.checkpoint import async_checkpoint_helper
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import stack
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib

try:
  import psutil  # pylint: disable=g-import-not-at-top
  psutil_import_succeeded = True
except ImportError:
  psutil_import_succeeded = False


class NonLayerTrackable(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # HELPER: small test fixture. An auto-trackable object (not a Keras layer) that owns one scalar
  # variable named a_variable created through the checkpoint add-variable utility.

  def __init__(self):
    # [FUNCTION DOC]
    # Calls the parent constructor and creates the scalar variable a_variable on the instance via
    # the add-variable utility.
    super().__init__()
    self.a_variable = trackable_utils.add_variable(
        self, name="a_variable", shape=[])


class InterfaceTests(test.TestCase):
  # [CLASS DOC]
  # Tests the add-variable helper used to attach variables to trackable objects (shape and dtype
  # handling, naming, duplicate detection, checkpoint key names).

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testAddVariable(self):
    # [FUNCTION DOC]
    # Adds variables with several initializer styles (constant, ones within a variable scope, bare
    # initializer class with dtype) to a trackable. Asserts: giving both shape and a constant
    # initializer raises a value error; a duplicate name on the same object raises "already
    # declared"; variable names and values are as expected (names are uniquified in graph mode
    # only); and the checkpoint keys are exactly the five expected attribute keys.
    obj = NonLayerTrackable()
    with self.assertRaisesRegex(ValueError, "do not specify shape"):
      trackable_utils.add_variable(
          obj, name="shape_specified_twice", shape=[], initializer=1)
    constant_initializer = trackable_utils.add_variable(
        obj, name="constant_initializer", initializer=1)
    with variable_scope.variable_scope("some_variable_scope"):
      ones_initializer = trackable_utils.add_variable(
          obj,
          name="ones_initializer",
          shape=[2],
          initializer=init_ops.ones_initializer(dtype=dtypes.float32))
    bare_initializer = trackable_utils.add_variable(
        obj,
        name="bare_initializer",
        shape=[2, 2],
        dtype=dtypes.float64,
        initializer=init_ops.zeros_initializer)

    # Even in graph mode, there are no naming conflicts between objects, only
    # naming conflicts within an object.
    other_duplicate = resource_variable_ops.ResourceVariable(
        name="duplicate", initial_value=1.)
    duplicate = trackable_utils.add_variable(
        obj, name="duplicate", shape=[])
    with self.assertRaisesRegex(ValueError, "'duplicate'.*already declared"):
      trackable_utils.add_variable(obj, name="duplicate", shape=[])

    self.evaluate(trackable_utils.gather_initializers(obj))
    self.assertEqual("constant_initializer:0", constant_initializer.name)
    self.assertEqual(1, self.evaluate(constant_initializer))
    self.assertEqual("some_variable_scope/ones_initializer:0",
                     ones_initializer.name)
    self.assertAllEqual([1, 1], self.evaluate(ones_initializer))
    self.assertAllEqual([[0., 0.],
                         [0., 0.]], self.evaluate(bare_initializer))
    self.assertEqual("a_variable:0", obj.a_variable.name)
    self.assertEqual("duplicate:0", other_duplicate.name)
    if context.executing_eagerly():
      # When executing eagerly, there's no uniquification of variable names. The
      # checkpoint name will be the same.
      self.assertEqual("duplicate:0", duplicate.name)
    else:
      # The .name attribute may be globally influenced, but the checkpoint name
      # won't be (tested below).
      self.assertEqual("duplicate_1:0", duplicate.name)

    expected_checkpoint_names = {
        "a_variable/.ATTRIBUTES/VARIABLE_VALUE",
        "bare_initializer/.ATTRIBUTES/VARIABLE_VALUE",
        "constant_initializer/.ATTRIBUTES/VARIABLE_VALUE",
        "duplicate/.ATTRIBUTES/VARIABLE_VALUE",
        "ones_initializer/.ATTRIBUTES/VARIABLE_VALUE",
    }
    actual_checkpoint_names = _get_all_checkpoint_names(obj)
    self.assertEqual(expected_checkpoint_names, set(actual_checkpoint_names))

  def testInitNotCalled(self):
    # [FUNCTION DOC]
    # Defines a trackable subclass whose constructor does not call the parent constructor and
    # checks that adding a variable to it still works (parent initialization happens implicitly).

    class NoInit(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: test fixture whose constructor deliberately skips the parent constructor.

      def __init__(self):
        # [FUNCTION DOC]
        # Does nothing, deliberately not calling the parent constructor.
        pass

    # __init__ for Trackable will be called implicitly.
    trackable_utils.add_variable(NoInit(), "var", shape=[])

  def testShapeDtype(self):
    # [FUNCTION DOC]
    # Adds variables with an explicit dtype (64-bit float) with and without a shape; asserts the
    # dtype is honored and a ones-initialized vector of length three evaluates to ones.
    root = autotrackable.AutoTrackable()
    v1 = trackable_utils.add_variable(
        root, name="v1", initializer=3., dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, v1.dtype)
    v2 = trackable_utils.add_variable(
        root,
        name="v2",
        shape=[3],
        initializer=init_ops.ones_initializer,
        dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, v2.dtype)
    self.assertAllEqual([1., 1., 1.], self.evaluate(v2))


class _MirroringSaveable(saver_lib.BaseSaverBuilder.SaveableObject):
  # [CLASS DOC]
  # HELPER: test fixture. A saveable object that saves only a primary variable's value but, on
  # restore, writes the restored value into both the primary and a mirrored variable.

  def __init__(self, primary_variable, mirrored_variable, name):
    # [FUNCTION DOC]
    # Inputs: primary_variable (variable whose value is saved), mirrored_variable (variable that
    # also receives the value on restore), name (string name of the saveable). Reads the primary
    # value, builds a single save spec with an empty slice spec and calls the parent constructor.
    self._primary_variable = primary_variable
    self._mirrored_variable = mirrored_variable
    tensor = self._primary_variable.read_value()
    spec = saver_lib.BaseSaverBuilder.SaveSpec(
        tensor=tensor,
        slice_spec="",
        name=name)
    super().__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # Inputs: restored_tensors (sequence with exactly one tensor), restored_shapes (unused).
    # Returns a grouped operation that assigns the restored tensor to both the primary and the
    # mirrored variable.
    """Restore the same value into both variables."""
    tensor, = restored_tensors
    return control_flow_ops.group(
        self._primary_variable.assign(tensor),
        self._mirrored_variable.assign(tensor))


class _OwnsMirroredVariables(base.Trackable):
  # [CLASS DOC]
  # HELPER: test fixture. A raw trackable owning two resource variables (non_dep_variable and
  # mirrored) that reports a custom mirroring saveable for checkpointing, so only the first
  # variable is stored and both are restored from it.
  """A Trackable object which returns a more complex SaveableObject."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates two resource variables through the variable-scope getter: non_dep_variable
    # (initial value 6) and mirrored (initial value 15). Takes no arguments.
    self.non_dep_variable = variable_scope.get_variable(
        name="non_dep_variable", initializer=6., use_resource=True)
    self.mirrored = variable_scope.get_variable(
        name="mirrored", initializer=15., use_resource=True)

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Returns a mapping from the variable-value key to a factory that builds the mirroring
    # saveable for this object (primary: non_dep_variable, mirrored: mirrored).
    def _saveable_factory(name=self.non_dep_variable.name):
      # [FUNCTION DOC]
      # HELPER: factory used by the saveable gathering.
      # Input: name (string, defaults to the primary variable's name). Returns a new mirroring
      # saveable over the two variables with that name.
      return _MirroringSaveable(
          primary_variable=self.non_dep_variable,
          mirrored_variable=self.mirrored,
          name=name)
    return {base.VARIABLE_VALUE_KEY: _saveable_factory}

  # The Saver sorts by name before parsing, so we need a name property.
  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the name of the primary variable; needed because the name-based saver
    # sorts its inputs by name.
    return self.non_dep_variable.name


def _get_all_checkpoint_names(root):
  # [FUNCTION DOC]
  # HELPER: shared by the naming tests.
  # Input: root, a trackable root object. Serializes its object graph view and returns a flat list
  # of every checkpoint key (string) across all serialized tensor mappings. No side effects.
  serialized_tensors, _, _, _ = save_util.serialize_graph_view(
      graph_view.ObjectGraphView(root))
  checkpoint_names = []
  for tensor_dict in serialized_tensors.values():
    checkpoint_names.extend(tensor_dict.keys())
  return checkpoint_names


class CheckpointingTests(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Main tests for object-based checkpoint save/restore: custom saveables, deferred and
  # overlapping restores, dependency tracking, checkpoint state files, options and callbacks,
  # path-like arguments, reference cycles and root-object arguments. Some tests are parameterized
  # over asynchronous checkpointing.

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False)
    )
  @test_util.run_in_graph_and_eager_modes
  def testMoreComplexSaveableReturned(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Input: enable_async_ckpt (boolean, parameterized). Saves a checkpoint of the mirroring
    # fixture (primary 42), changes both variables, restores and asserts both equal 42; repeats
    # with primary 44/45 to confirm both end at 44. Syncs the checkpoint when async is on.
    v = _OwnsMirroredVariables()
    checkpoint = trackable_utils.Checkpoint(v=v)
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    self.evaluate(v.non_dep_variable.assign(42.))
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    save_path = checkpoint.save(file_prefix=prefix, options=ckpt_options)
    # TODO(chienchunh): Identify why sync needs to be called here.
    if enable_async_ckpt:
      checkpoint.sync()
    self.evaluate(v.non_dep_variable.assign(43.))
    self.evaluate(v.mirrored.assign(44.))
    checkpoint.restore(save_path).assert_consumed().initialize_or_restore()
    self.assertEqual(42., self.evaluate(v.non_dep_variable))
    self.assertEqual(42., self.evaluate(v.mirrored))
    self.evaluate(v.non_dep_variable.assign(44.))
    save_path = checkpoint.save(file_prefix=prefix, options=ckpt_options)
    # TODO(chienchunh): Identify why sync needs to be called here.
    if enable_async_ckpt:
      checkpoint.sync()
    self.evaluate(v.non_dep_variable.assign(45.))
    checkpoint.restore(save_path).assert_consumed().initialize_or_restore()
    self.assertEqual(44., self.evaluate(v.non_dep_variable))
    self.assertEqual(44., self.evaluate(v.mirrored))

  @test_util.run_in_graph_and_eager_modes
  def testMoreComplexSaveableReturnedWithGlobalName(self):
    # [FUNCTION DOC]
    # Saves and restores the mirroring fixture with the name-based saver in a cached session;
    # asserts both variables end at the saved value 42 after being overwritten.
    # The same object can also be saved using the name-based saver.
    v = _OwnsMirroredVariables()
    saver = saver_lib.Saver(var_list=[v])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    with self.cached_session() as sess:
      self.evaluate(v.non_dep_variable.assign(42.))
      save_path = saver.save(sess, prefix)
      self.evaluate(v.non_dep_variable.assign(43.))
      self.evaluate(v.mirrored.assign(44.))
      saver.restore(sess, save_path)
      self.assertEqual(42., self.evaluate(v.non_dep_variable))
      self.assertEqual(42., self.evaluate(v.mirrored))

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False)
    )
  @test_util.run_in_graph_and_eager_modes
  def testAssertConsumedNoCheckpoint(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Input: enable_async_ckpt (boolean, parameterized). Saves a checkpoint of one variable,
    # restores it, deletes the checkpoint object, and asserts the restore status still reports
    # everything consumed without error.
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    v = variable_scope.get_variable(name="v", initializer=0.)
    self.evaluate(v.initializer)
    ckpt = trackable_utils.Checkpoint(v=v)
    self.evaluate(trackable_utils.gather_initializers(ckpt))
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    save_path = ckpt.save(file_prefix=prefix, options=ckpt_options)
    status = ckpt.restore(save_path=save_path)
    del ckpt
    status.assert_consumed()

  def testDeepCopyCheckpoint(self):
    # [FUNCTION DOC]
    # Deep-copies a checkpoint holding a variable, assigns 2 through the copy, asserts the original
    # variable is still 1, saves via the copy, restores into the original and asserts it becomes 2.
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    v = variables_lib.Variable(1.)
    original_ckpt = trackable_utils.Checkpoint(v=v)
    copied_ckpt = copy.deepcopy(original_ckpt)
    copied_ckpt.v.assign(2.)
    self.assertAllClose(1., v)
    save_path = copied_ckpt.save(file_prefix=prefix)
    original_ckpt.restore(save_path=save_path).assert_consumed()
    self.assertAllClose(2., v)

  @test_util.run_in_graph_and_eager_modes
  def testPassingCheckpointOptions(self):
    # [FUNCTION DOC]
    # Saves and restores with an options object selecting a localhost CPU io device; asserts
    # consumption, and in graph mode that every save and restore operation is placed on that
    # device.
    localhost = "/job:localhost/device:CPU:0"
    options = checkpoint_options.CheckpointOptions(
        experimental_io_device=localhost)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    v = variable_scope.get_variable(name="v", initializer=0.)
    self.evaluate(v.initializer)
    ckpt = trackable_utils.Checkpoint(v=v)
    self.evaluate(trackable_utils.gather_initializers(ckpt))
    save_path = ckpt.save(file_prefix=prefix, options=options)
    status = ckpt.restore(save_path=save_path, options=options)
    del ckpt
    status.assert_consumed()

    # In graph mode, verify that the save and restore ops were set to run on
    # localhost.
    if not context.executing_eagerly():
      for op in ops.get_default_graph().get_operations():
        if op.type in ("SaveV2", "RestoreV2"):
          self.assertEqual(localhost, op.device)

  @test_util.run_in_graph_and_eager_modes
  def testFreezing(self):
    # [FUNCTION DOC]
    # Uses a GPU if available. Saves with a frozen saver (variable 3, save counter 12), overwrites
    # and restores with it, then restores with a second frozen saver on an identical graph, and
    # finally restores as a normal object-based checkpoint with deferred variable creation.
    # Asserts the variable is 3 and the save counter 12 each time.
    with test_util.use_gpu():
      # Save an object-based checkpoint using a frozen saver
      directory = self.get_temp_dir()
      prefix = os.path.join(directory, "ckpt")
      v = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
      checkpoint = trackable_utils.Checkpoint(v=v)
      self.evaluate(v.assign(3))
      # Create the save counter so assert_consumed doesn't complain about it not
      # existing in the checkpoint on restore.
      self.evaluate(checkpoint.save_counter.assign(12))
      saver = trackable_utils.frozen_saver(checkpoint)
      with ops.device("cpu:0"):
        prefix_tensor = constant_op.constant(prefix)
      self.evaluate(saver.save(prefix_tensor))
      self.evaluate(v.assign(10))
      # Use the frozen saver to restore the same object graph
      self.evaluate(saver.restore(prefix_tensor))
      self.assertEqual(3, self.evaluate(v))

      # Restore using another frozen saver on an identical object graph
      del v, checkpoint, saver
      v = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
      checkpoint = trackable_utils.Checkpoint(v=v)
      saver = trackable_utils.frozen_saver(checkpoint)
      self.evaluate(saver.restore(prefix_tensor))
      self.assertEqual(3, self.evaluate(v))

      # Restore as an object-based checkpoint
      del v, checkpoint, saver
      checkpoint = trackable_utils.Checkpoint()
      status = checkpoint.restore(prefix)
      v = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
      if context.executing_eagerly():
        self.assertEqual(12, self.evaluate(checkpoint.save_counter))
        self.assertEqual(0, self.evaluate(v))
      checkpoint.v = v
      status.assert_consumed().run_restore_ops()
      self.assertEqual(3, self.evaluate(v))
      self.assertEqual(12, self.evaluate(checkpoint.save_counter))

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False)
    )
  @test_util.run_in_graph_and_eager_modes
  def testCustomNumbering(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Input: enable_async_ckpt (boolean, parameterized). Writes five checkpoints with prefixes
    # numbered by a step variable that grows by two; fails if any returned path does not end with
    # the expected step suffix (the write method does not append its own counter).
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, "ckpt")
    step = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
    checkpoint = trackable_utils.Checkpoint(step=step)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    self.evaluate(step.initializer)
    for i in range(5):
      path = checkpoint.write("%s-%d" % (prefix, self.evaluate(step)),
                              options=ckpt_options)
      expected_suffix = "-%d" % (2 * i,)
      if not path.endswith(expected_suffix):
        self.fail("%s should have suffix %s" % (path, expected_suffix))
      self.evaluate(step.assign_add(2))

  def testPartialRestoreWarningAttribute(self):
    # [FUNCTION DOC]
    # Eager mode. Saves two variables, restores into a root whose first attribute is a plain
    # trackable (no value slot). Deleting the root must free it (checked via a weak reference) and
    # log a warning naming the unrestored first attribute but not the second, with a hint to
    # call the expect-partial method. Also asserts the second variable was restored to 3.
    with context.eager_mode():
      original_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(2.),
                                                 v2=variables_lib.Variable(3.))
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = original_root.save(prefix)
      partial_root = trackable_utils.Checkpoint(v1=base.Trackable(),
                                                v2=variables_lib.Variable(0.))
      weak_partial_root = weakref.ref(partial_root)
      with test.mock.patch.object(logging, "warning") as mock_log:
        # Note: Unlike in testPartialRestoreWarningObject, the warning actually
        # prints immediately here, since all of the objects have been created
        # and there's no deferred restoration sitting around.
        partial_root.restore(save_path)
        self.assertEqual(3., partial_root.v2.numpy())
        del partial_root
        self.assertIsNone(weak_partial_root())
        messages = str(mock_log.call_args_list)
      self.assertIn("(root).v1", messages)
      self.assertNotIn("(root).v2", messages)
      self.assertIn("expect_partial()", messages)

  def testAttributeException(self):
    # [FUNCTION DOC]
    # Eager mode. Restores a two-variable checkpoint into a root whose first attribute cannot hold
    # a value; asserts the consumption check raises an assertion error mentioning unused
    # attributes for that first attribute.
    with context.eager_mode():
      original_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(2.),
                                                 v2=variables_lib.Variable(3.))
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = original_root.save(prefix)
      partial_root = trackable_utils.Checkpoint(v1=base.Trackable(),
                                                v2=variables_lib.Variable(0.))
      status = partial_root.restore(save_path)
      with self.assertRaisesRegex(AssertionError,
                                  r"Unused attributes(.|\n)*\(root\).v1"):
        status.assert_consumed()

  def testSilencePartialWarning(self):
    # [FUNCTION DOC]
    # Eager mode. Restores a two-variable checkpoint into a root with only one variable and marks
    # the restore as partial. Asserts the restored value is 2 and that deleting the root frees the
    # root and variable and logs no warning.
    with context.eager_mode():
      original_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(2.),
                                                 v2=variables_lib.Variable(3.))
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = original_root.save(prefix)
      partial_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(0.))
      weak_partial_root = weakref.ref(partial_root)
      weak_v1 = weakref.ref(partial_root.v1)
      partial_root.restore(save_path).expect_partial()
      self.assertEqual(2., partial_root.v1.numpy())
      with test.mock.patch.object(logging, "warning") as mock_log:
        del partial_root
        self.assertIsNone(weak_partial_root())
        self.assertIsNone(weak_v1())
        self.assertEmpty(mock_log.call_args_list)

  def _get_checkpoint_name(self, name):
    # [FUNCTION DOC]
    # HELPER: used by testVariableNameEscaping.
    # Input: name, string variable name. Adds a 1x2 float64 variable with that name to a fresh
    # trackable, returns its first checkpoint key, and also checks the key is usable as part of an
    # operation name scope.
    root = autotrackable.AutoTrackable()
    trackable_utils.add_variable(
        root, name=name, shape=[1, 2], dtype=dtypes.float64)
    checkpoint_key = _get_all_checkpoint_names(root)[0]
    with ops.name_scope("root/" + checkpoint_key):
      pass  # Make sure we can use this as an op name if we prefix it.
    return checkpoint_key

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testVariableNameEscaping(self):
    # [FUNCTION DOC]
    # Asserts the checkpoint key escaping rules: slashes become a dot-S sequence, and names
    # containing reserved fragments are escaped, always followed by the variable-value suffix.
    suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
    self.assertEqual(r"a.Sb.Sc" + suffix, self._get_checkpoint_name(r"a/b/c"))
    self.assertEqual(r"b" + suffix, self._get_checkpoint_name(r"b"))
    self.assertEqual(r"c.S" + suffix, self._get_checkpoint_name(r"c/"))
    self.assertEqual(r"d.S..S" + suffix, self._get_checkpoint_name(r"d/.S"))
    self.assertEqual(r"d.S..ATTRIBUTES.Sf" + suffix,
                     self._get_checkpoint_name(r"d/.ATTRIBUTES/f"))

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNumberedPath(self):
    # [FUNCTION DOC]
    # Nests a variable one level (root, leaf, variable v) and asserts the checkpoint key is the
    # slash-joined path ending with the variable-value suffix.
    root = autotrackable.AutoTrackable()
    leaf = autotrackable.AutoTrackable()
    root.leaf = leaf
    trackable_utils.add_variable(leaf, name="v", shape=[])
    checkpoint_key = _get_all_checkpoint_names(root)[0]
    self.assertEqual(r"leaf/v/.ATTRIBUTES/VARIABLE_VALUE", checkpoint_key)

  @test_util.run_in_graph_and_eager_modes
  def testLocalNameValidation(self):
    # [FUNCTION DOC]
    # Tracks a child under the reserved name for attributes and asserts that the leading dot is
    # escaped in the resulting checkpoint key, avoiding a clash with the reserved name.
    root = autotrackable.AutoTrackable()
    leaf = autotrackable.AutoTrackable()
    # Dots are escaped, which avoids conflicts with reserved names.
    root._track_trackable(leaf, name=".ATTRIBUTES")
    trackable_utils.add_variable(trackable=leaf, name="a", shape=[])
    checkpoint_key = _get_all_checkpoint_names(root)[0]
    self.assertEqual("..ATTRIBUTES/a/.ATTRIBUTES/VARIABLE_VALUE",
                     checkpoint_key)

  @test_util.run_in_graph_and_eager_modes
  def testLateDependencyTracking(self):
    # [FUNCTION DOC]
    # Saves a checkpoint of an object that adds a dependency (with a variable at 123) after
    # construction. Restores into a fresh object before the dependency exists: existing objects
    # match but consumption fails. After adding the dependency, consumption succeeds and the
    # variable restores to 123.

    class Dependency(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: test fixture whose build method creates a variable named var.

      def build(self):
        # [FUNCTION DOC]
        # Creates the variable var (initial value 0) on this object.
        self.var = trackable_utils.add_variable(
            self, "var", initializer=0.)

    class LateDependencies(trackable_utils.Checkpoint):
      # [CLASS DOC]
      # HELPER: test fixture. A checkpoint object that attaches its dependency late.

      def add_dep(self):
        # [FUNCTION DOC]
        # Creates a dependency, builds its variable after assignment to this object, and stores
        # it as attribute dep.
        self.dep = Dependency()
        self.dep.build()

    original = LateDependencies()
    original.add_dep()
    self.evaluate(state_ops.assign(original.dep.var, 123.))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = original.save(checkpoint_prefix)
    load_into = LateDependencies()
    status = load_into.restore(save_path)
    status.assert_existing_objects_matched()
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    load_into.add_dep()
    status.assert_consumed()
    status.assert_existing_objects_matched().run_restore_ops()
    self.assertEqual(123., self.evaluate(load_into.dep.var))

  @test_util.run_in_graph_and_eager_modes
  def testDepAfterVar(self):
    # [FUNCTION DOC]
    # Saves a checkpoint of an object whose dependency has its variable built before being
    # attached (value -14). Restores into a fresh object, then adds the dependency; asserts
    # consumption and that the variable becomes -14.

    class Dependency(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: test fixture whose build method creates a variable named var.

      def build(self):
        # [FUNCTION DOC]
        # Creates the variable var (initial value 0) on this object.
        self.var = trackable_utils.add_variable(
            self, "var", initializer=0.)

    class DepAfterVar(trackable_utils.Checkpoint):
      # [CLASS DOC]
      # HELPER: test fixture. A checkpoint object that attaches a dependency only after the
      # dependency's variable has been created.

      def add_dep(self):
        # [FUNCTION DOC]
        # Creates a dependency, builds its variable first, then attaches it as attribute dep.
        dep = Dependency()
        dep.build()
        self.dep = dep

    dep_after_var = DepAfterVar()
    dep_after_var.add_dep()
    self.evaluate(state_ops.assign(dep_after_var.dep.var, -14.))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = dep_after_var.save(checkpoint_prefix)

    loaded_dep_after_var = DepAfterVar()
    status = loaded_dep_after_var.restore(save_path)
    loaded_dep_after_var.add_dep()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(-14., self.evaluate(loaded_dep_after_var.dep.var))

  @test_util.run_in_graph_and_eager_modes
  def testOverlappingRestores(self):
    # [FUNCTION DOC]
    # Saves two checkpoints (values 12 and 13). Restores each into a separate empty root, then
    # attaches the same dependency to both. Asserts each restore is consumed and the variable
    # takes 12 then 13. Repeats with the restore order reversed and asserts the last restore
    # wins (12 both times).
    checkpoint_directory = self.get_temp_dir()
    save_root = trackable_utils.Checkpoint()
    save_root.dep = autotrackable.AutoTrackable()
    save_root.dep.var = trackable_utils.add_variable(
        save_root.dep, name="var", initializer=0.)
    self.evaluate(state_ops.assign(save_root.dep.var, 12.))
    first_path = save_root.save(os.path.join(checkpoint_directory, "first"))
    self.evaluate(state_ops.assign(save_root.dep.var, 13.))
    second_path = save_root.save(os.path.join(checkpoint_directory, "second"))

    first_root = trackable_utils.Checkpoint()
    second_root = trackable_utils.Checkpoint()
    first_status = first_root.restore(first_path)
    second_status = second_root.restore(second_path)
    load_dep = autotrackable.AutoTrackable()
    load_dep.var = trackable_utils.add_variable(
        load_dep, name="var", shape=[])
    first_root.dep = load_dep
    first_status.assert_consumed()
    first_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(load_dep.var))
    second_root.dep = load_dep
    second_status.assert_consumed()
    second_status.run_restore_ops()
    self.assertEqual(13., self.evaluate(load_dep.var))

    # Try again with the order of the restore() reversed. The last restore
    # determines the final value.
    first_root = trackable_utils.Checkpoint()
    second_root = trackable_utils.Checkpoint()
    second_status = second_root.restore(second_path)
    first_status = first_root.restore(first_path)
    load_dep = autotrackable.AutoTrackable()
    load_dep.var = trackable_utils.add_variable(
        load_dep, name="var", shape=[])
    first_root.dep = load_dep
    first_status.assert_consumed()
    first_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(load_dep.var))
    second_root.dep = load_dep
    second_status.assert_consumed()
    second_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(load_dep.var))

  @test_util.run_in_graph_and_eager_modes
  def testAmbiguousLoad(self):
    # [FUNCTION DOC]
    # Saves a graph where one shared object is reachable through two parents, then restores into
    # a graph with two separate objects. Asserts both the consumption check and the
    # existing-objects-matched check raise assertion errors (one checkpoint object cannot be split).
    # Not OK to split one checkpoint object into two
    checkpoint_directory = self.get_temp_dir()
    save_root = trackable_utils.Checkpoint()
    save_root.dep_one = autotrackable.AutoTrackable()
    save_root.dep_two = autotrackable.AutoTrackable()
    dep_three = autotrackable.AutoTrackable()
    save_root.dep_one.dep_three = dep_three
    save_root.dep_two.dep_three = dep_three
    trackable_utils.add_variable(dep_three, name="var", initializer=0.)
    self.evaluate(trackable_utils.gather_initializers(save_root))
    save_path = save_root.save(os.path.join(checkpoint_directory, "ckpt"))
    load_root = trackable_utils.Checkpoint()
    status = load_root.restore(save_path)
    load_root.dep_one = autotrackable.AutoTrackable()
    load_root.dep_two = autotrackable.AutoTrackable()
    load_root.dep_one.dep_three = autotrackable.AutoTrackable()
    load_root.dep_two.dep_three = autotrackable.AutoTrackable()
    trackable_utils.add_variable(
        load_root.dep_one.dep_three, name="var", initializer=0.)
    trackable_utils.add_variable(
        load_root.dep_two.dep_three, name="var", initializer=0.)
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    with self.assertRaises(AssertionError):
      status.assert_existing_objects_matched()

  @test_util.run_in_graph_and_eager_modes
  def testObjectsCombined(self):
    # [FUNCTION DOC]
    # Saves two separate objects holding different variables (32 and 64), then restores into one
    # Python object referenced under both names holding both variables. Asserts the restore is
    # consumed, matched and both values are restored.
    # Currently fine to load two checkpoint objects into one Python object
    checkpoint_directory = self.get_temp_dir()
    save_root = trackable_utils.Checkpoint()
    save_root.dep_one = autotrackable.AutoTrackable()
    save_root.dep_two = autotrackable.AutoTrackable()
    trackable_utils.add_variable(
        save_root.dep_one, name="var1", initializer=32., dtype=dtypes.float64)
    trackable_utils.add_variable(
        save_root.dep_two, name="var2", initializer=64., dtype=dtypes.float64)
    self.evaluate(trackable_utils.gather_initializers(save_root))
    save_path = save_root.save(os.path.join(checkpoint_directory, "ckpt"))
    load_root = trackable_utils.Checkpoint()
    load_root.dep_one = autotrackable.AutoTrackable()
    load_root.dep_two = load_root.dep_one
    v1 = trackable_utils.add_variable(
        load_root.dep_one, name="var1", shape=[], dtype=dtypes.float64)
    v2 = trackable_utils.add_variable(
        load_root.dep_one, name="var2", shape=[], dtype=dtypes.float64)
    status = load_root.restore(
        save_path).assert_consumed().assert_existing_objects_matched()
    status.run_restore_ops()
    self.assertEqual(32., self.evaluate(v1))
    self.assertEqual(64., self.evaluate(v2))

  @test_util.run_in_graph_and_eager_modes
  def testEmptyContainersIgnored(self):
    # [FUNCTION DOC]
    # Saves a checkpoint holding only an empty list and restores into a root with differently
    # named empty lists. Asserts consumption, matched objects and a nontrivial match all hold,
    # i.e. empty containers do not cause failures.
    checkpoint_directory = self.get_temp_dir()
    save_root = trackable_utils.Checkpoint(a=[])
    path = save_root.save(checkpoint_directory)
    load_root = trackable_utils.Checkpoint(b=[])
    load_root.dep = []
    load_root.dep.append([])
    status = load_root.restore(path)
    status.assert_consumed()
    status.assert_existing_objects_matched()
    status.assert_nontrivial_match()

  @test_util.run_in_graph_and_eager_modes
  def testDependencyLoop(self):
    # [FUNCTION DOC]
    # Builds two checkpoint objects that reference each other (a reference cycle, so eager garbage
    # is expected), each with a variable, and saves. Tests deferred loading into a new cyclic
    # graph (consumption fails until variables exist, then values restore), and re-restoring
    # after variables were changed.
    # Note: this test creates garbage during eager execution because it
    # purposefully creates a reference cycle.
    first = trackable_utils.Checkpoint()
    second = trackable_utils.Checkpoint()
    first.second = second
    second.first = first
    first.v = trackable_utils.add_variable(
        first, "v1", initializer=[3., 1., 4.])
    second.v = trackable_utils.add_variable(
        second, "v2", initializer=[1., 1., 2., 3.])
    self.evaluate(trackable_utils.gather_initializers(first))
    checkpoint_directory = self.get_temp_dir()
    save_path = first.save(os.path.join(checkpoint_directory, "ckpt"))

    # Test deferred loading
    first_load = trackable_utils.Checkpoint()
    status = first_load.restore(save_path)
    second_load = autotrackable.AutoTrackable()
    first_load.second = second_load
    second_load.first = first_load
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    first_load.v = trackable_utils.add_variable(
        first_load, "v1", shape=[3])
    second_load.v = trackable_utils.add_variable(
        second_load, "v2", shape=[4])
    status.assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([3., 1., 4.], self.evaluate(first_load.v))
    self.assertAllEqual([1., 1., 2., 3.], self.evaluate(second_load.v))

    # Test loading when variables have already been created
    self.evaluate(first_load.v.assign([2., 7., 1.]))
    self.assertAllEqual([2., 7., 1.], self.evaluate(first_load.v))
    self.evaluate(second_load.v.assign([2., 7., 1., 8.]))
    self.assertAllEqual([2., 7., 1., 8.], self.evaluate(second_load.v))
    status = first_load.restore(save_path).assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([3., 1., 4.], self.evaluate(first_load.v))
    self.assertAllEqual([1., 1., 2., 3.], self.evaluate(second_load.v))

  @test_util.run_in_graph_and_eager_modes
  def testRestoreOnAssign(self):
    # [FUNCTION DOC]
    # Saves two variables (4 and 8). Restores into an object that has only the second one, then
    # creates the first variable separately, changes it, attaches it, and runs the restore ops
    # again. Asserts the attached variable is restored to 4 once it is attached.
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    first = trackable_utils.Checkpoint()
    first.var1 = variables_lib.Variable(0., name="outside_var")
    first.var2 = variables_lib.Variable(0., name="blah")
    self.evaluate(first.var1.assign(4.))
    self.evaluate(first.var2.assign(8.))
    save_path = first.save(checkpoint_prefix)

    second = trackable_utils.Checkpoint()
    second.var2 = variables_lib.Variable(0., name="blah")
    status = second.restore(save_path)
    recreated_var1 = variables_lib.Variable(0., name="outside_var")
    status.run_restore_ops()
    self.assertEqual(8., self.evaluate(second.var2))
    self.evaluate(recreated_var1.assign(-2.))
    self.assertEqual(-2., self.evaluate(recreated_var1))
    second.var1 = recreated_var1
    status.run_restore_ops()
    self.assertEqual(4., self.evaluate(recreated_var1))

  @test_util.run_in_graph_and_eager_modes
  def testCheckpointState(self):
    # [FUNCTION DOC]
    # Saves ten times with the same prefix. Asserts the directory contains the state file and all
    # ten numbered index files (nothing is deleted by default).
    # No checkpoints are deleted by default
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    obj = autotrackable.AutoTrackable()
    obj.var = variable_scope.get_variable(name="v", initializer=0.)
    self.evaluate(trackable_utils.gather_initializers(obj))
    saver = trackable_utils.Checkpoint(obj=obj)
    for _ in range(10):
      saver.save(checkpoint_prefix)
    expected_filenames = ["checkpoint"]
    for checkpoint_number in range(1, 11):
      expected_filenames.append("ckpt-%d.index" % (checkpoint_number,))
    self.assertEmpty(
        set(expected_filenames)
        - set(os.listdir(checkpoint_directory)))

  @test_util.run_in_graph_and_eager_modes
  def testCheckpointStateChangingVarList(self):
    # [FUNCTION DOC]
    # Adds a new variable before each of ten saves. Asserts all index files exist, the latest
    # checkpoint is number ten and the state lists only that one path. After overwriting all
    # variables, restores number six and then ten and asserts which variables are restored (those
    # existing at save time) and which keep the overwritten value.
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    obj = autotrackable.AutoTrackable()
    obj.var = variable_scope.get_variable(name="v", initializer=0.)
    self.evaluate(trackable_utils.gather_initializers(obj))
    checkpoint = trackable_utils.Checkpoint(obj=obj)
    looped_variables = []
    for iteration in range(10):
      new_variable = resource_variable_ops.ResourceVariable(iteration)
      self.evaluate(new_variable.initializer)
      setattr(checkpoint, "var_%d" % iteration, new_variable)
      checkpoint.save(checkpoint_prefix)
      looped_variables.append(new_variable)
    expected_filenames = ["checkpoint"]
    # We've copied the saver each time, but checkpoint management should still
    # be consistent. Nothing gets deleted.
    for checkpoint_number in range(1, 11):
      expected_filenames.append("ckpt-%d.index" % (checkpoint_number,))
    self.assertEmpty(
        set(expected_filenames)
        - set(os.listdir(checkpoint_directory)))
    self.assertEqual(
        checkpoint_prefix + "-10",
        checkpoint_management.latest_checkpoint(checkpoint_directory))
    # The checkpoint list only contains the most recent checkpoint, but they're
    # all on disk. This means we won't eventually run into proto size limits.
    self.assertEqual(
        [checkpoint_prefix + "-10"],
        (checkpoint_management.get_checkpoint_state(checkpoint_directory)
         .all_model_checkpoint_paths))
    for v in looped_variables:
      self.evaluate(v.assign(314))
    checkpoint.restore(checkpoint_prefix + "-6").run_restore_ops()
    self.assertEqual(314, self.evaluate(checkpoint.var_9))
    self.assertEqual(314, self.evaluate(checkpoint.var_8))
    self.assertEqual(314, self.evaluate(checkpoint.var_6))
    self.assertEqual(5, self.evaluate(checkpoint.var_5))
    self.assertEqual(1, self.evaluate(checkpoint.var_1))
    self.assertEqual(0, self.evaluate(checkpoint.var_0))
    checkpoint.restore(checkpoint_prefix + "-10").run_restore_ops()
    self.assertEqual(9, self.evaluate(checkpoint.var_9))
    self.assertEqual(8, self.evaluate(checkpoint.var_8))
    self.assertEqual(1, self.evaluate(checkpoint.var_1))
    self.assertEqual(0, self.evaluate(checkpoint.var_0))

  @test_util.run_in_graph_and_eager_modes
  def test_restore_after_adding_empty_trackable_data_structure(self):
    # [FUNCTION DOC]
    # Saves a model checkpoint, then restores into a new model that additionally holds a dict and
    # a list-named mapping. Asserts existing objects are still matched and restore ops run.
    model = NonLayerTrackable()
    checkpoint = trackable_utils.Checkpoint(model=model)
    checkpoint.restore(None).initialize_or_restore()
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    del model, checkpoint

    model = NonLayerTrackable()
    model.dict = {"a": 1}
    model.list = {"b": 1}
    checkpoint = trackable_utils.Checkpoint(model=model)
    load_status = checkpoint.restore(save_path)
    load_status.assert_existing_objects_matched().run_restore_ops()

  @test_util.run_in_graph_and_eager_modes
  def test_write_checkpoint_path_str_from_function(self):
    # [FUNCTION DOC]
    # Writes a checkpoint from inside a compiled function using a Python string prefix, then reads
    # it into another checkpoint and asserts matched, consumed and value 1. Changes the variable
    # to 3, writes again and asserts the reader sees 3.

    checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(1.))

    @def_function.function
    def _write_checkpoint():
      # [FUNCTION DOC]
      # HELPER: compiled function that writes the checkpoint to the fixed prefix and returns the
      # resulting path.
      save_path = save_checkpoint.write(checkpoint_prefix)
      return save_path

    self.evaluate([save_checkpoint.v.initializer])
    self.evaluate(_write_checkpoint())
    load_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(0.))
    # Use read() instead of restore() which allows us to check that all
    # existing objects were loaded.
    status = load_checkpoint.read(checkpoint_prefix)
    status.assert_existing_objects_matched()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(1., self.evaluate(load_checkpoint.v))
    self.evaluate(save_checkpoint.v.assign(3.))
    self.evaluate(_write_checkpoint())
    self.evaluate(save_checkpoint.v.assign(0.))
    status = load_checkpoint.read(checkpoint_prefix)
    status.assert_existing_objects_matched()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(3., self.evaluate(load_checkpoint.v))

  @test_util.run_in_graph_and_eager_modes
  def test_write_checkpoint_path_tensor_from_function(self):
    # [FUNCTION DOC]
    # Same scenario as the string-prefix test, but the prefix is passed to the compiled function
    # as a string tensor. Asserts the reader sees 1 and then 3 after the second write.
    # Same as the previous test, but the path is a tensor not a python string.
    checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")

    checkpoint_prefix_tensor = constant_op.constant(checkpoint_prefix)

    save_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(1.))

    @def_function.function
    def _write_checkpoint(prefix):
      # [FUNCTION DOC]
      # HELPER: compiled function. Input: prefix, string tensor path prefix. Writes the checkpoint
      # and returns the resulting path.
      save_path = save_checkpoint.write(prefix)
      return save_path

    self.evaluate([save_checkpoint.v.initializer])
    self.evaluate(_write_checkpoint(checkpoint_prefix_tensor))
    load_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(0.))
    # Use read() instead of restore() which allows us to check that all
    # existing objects were loaded.
    status = load_checkpoint.read(checkpoint_prefix)
    status.assert_existing_objects_matched()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(1., self.evaluate(load_checkpoint.v))
    self.evaluate(save_checkpoint.v.assign(3.))
    self.evaluate(_write_checkpoint(checkpoint_prefix_tensor))
    self.evaluate(save_checkpoint.v.assign(0.))
    status = load_checkpoint.read(checkpoint_prefix)
    status.assert_existing_objects_matched()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(3., self.evaluate(load_checkpoint.v))

  @test_util.run_in_graph_and_eager_modes
  def test_write_checkpoint_path_tensor_does_not_exist_from_function(self):
    # [FUNCTION DOC]
    # Writes from a compiled function with a tensor prefix inside a directory that does not exist;
    # asserts a not-found error is raised on evaluation.
    # Same as the previous test, but the path is a tensor not a python string.
    checkpoint_prefix = os.path.join(
        self.get_temp_dir(), "DOES_NOT_EXIST", "ckpt")

    checkpoint_prefix_tensor = constant_op.constant(checkpoint_prefix)

    save_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(1.))

    @def_function.function
    def _write_checkpoint(prefix):
      # [FUNCTION DOC]
      # HELPER: compiled function. Input: prefix, string tensor path prefix. Writes the checkpoint
      # and returns the resulting path.
      save_path = save_checkpoint.write(prefix)
      return save_path

    self.evaluate([save_checkpoint.v.initializer])
    with self.assertRaises(errors_impl.NotFoundError):
      self.evaluate(_write_checkpoint(checkpoint_prefix_tensor))

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False))
  def test_inititialize_with_data_structures(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Input: enable_async_ckpt (boolean, parameterized). Creates a checkpoint from a list and a
    # dict of variables, saves, and restores into a checkpoint with different values (joining the
    # async thread first by restoring on the saver). Asserts the list becomes 0,1 and dict 2,3.
    checkpoint = trackable_utils.Checkpoint(
        a=[variables_lib.Variable(0.), variables_lib.Variable(1.)],
        b={"a": variables_lib.Variable(2.), "b": variables_lib.Variable(3.)})
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    save_path = checkpoint.save(file_prefix=checkpoint_prefix,
                                options=ckpt_options)
    load_checkpoint = trackable_utils.Checkpoint(
        a=[variables_lib.Variable(4.), variables_lib.Variable(5.)],
        b={"a": variables_lib.Variable(6.), "b": variables_lib.Variable(7.)})
    # When async checkpoint is enabled, we need to first make sure that the
    # checkpoint saving is fully complete before the checkpoint file can be
    # loaded by another checkpoint instance. Calling checkpoint.restore() is a
    # trick to make sure its async thread is joined.
    if enable_async_ckpt:
      checkpoint.restore(save_path)
    load_checkpoint.restore(save_path)
    self.assertAllClose(self.evaluate(load_checkpoint.a), [0, 1])
    self.assertAllClose(self.evaluate(load_checkpoint.b), {"a": 2, "b": 3})

  def _create_trackable(self):
    # [FUNCTION DOC]
    # HELPER: builds a fresh model for the root-object tests. Returns an instance of a small
    # trackable model holding one variable (initial 2) that multiplies its input by it.
    class Model(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: test fixture. Trackable model with variable v (initial 2), callable on a number
      # or tensor.

      def __init__(self):
        # [FUNCTION DOC]
        # Creates variable v with initial value 2.
        self.v = variables_lib.Variable(2.)

      def __call__(self, x):
        # [FUNCTION DOC]
        # Input: x, number or tensor. Returns the variable v multiplied by x.
        return self.v * x
    return Model()

  def test_initialize_with_root_object(self):
    # [FUNCTION DOC]
    # Saves a checkpoint built with a model as the root object (model has an extra variable set to
    # 5). Restores into a new model and asserts outputs match; then attaches a deferred variable
    # and asserts it picks up the saved value 5.
    model = self._create_trackable()
    input_value = constant_op.constant([[3.]])
    expected_output = self.evaluate(model(input_value))
    model.deferred_variable = variables_lib.Variable(5.)

    checkpoint = trackable_utils.Checkpoint(model)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    new_model = self._create_trackable()
    load_checkpoint = trackable_utils.Checkpoint(new_model)
    load_checkpoint.restore(save_path)
    self.assertAllClose(expected_output, new_model(input_value))

    new_model.deferred_variable = variables_lib.Variable(1.)
    self.assertEqual(self.evaluate(new_model.deferred_variable), 5)

  def test_initialize_with_root_object_and_kwargs(self):
    # [FUNCTION DOC]
    # Asserts constructing a checkpoint with a root model plus a keyword clashing with an existing
    # attribute raises a value error ("already exists"). Saves with a root model and a separate
    # variable, then restores in three configurations: the same layout, a swapped layout where one
    # variable is attached to the root, and no root object. Asserts values (3 and 5) and the save
    # counter (1) in each case.
    model = self._create_trackable()
    model.v.assign(3.)
    separate_variable = variables_lib.Variable(5.)

    with self.assertRaisesRegex(ValueError, "root.v already exists"):
      trackable_utils.Checkpoint(model, v=separate_variable)

    checkpoint = trackable_utils.Checkpoint(
        model, separate_variable=separate_variable)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    # Case 1: Loading checkpoint with same configuration.
    new_model = self._create_trackable()
    separate_variable = variables_lib.Variable(1.)
    load_checkpoint = trackable_utils.Checkpoint(
        new_model, separate_variable=separate_variable)
    load_checkpoint.restore(save_path).assert_consumed()
    self.assertEqual(self.evaluate(new_model.v), 3)
    self.assertEqual(self.evaluate(separate_variable), 5)
    self.assertEqual(self.evaluate(load_checkpoint.save_counter), 1)

    # Case 2: Loading checkpoint where v and separate_variable are swapped:
    # v is not attached to the root, while separate variable is attached to root
    new_model = autotrackable.AutoTrackable()
    new_model.separate_variable = variables_lib.Variable(200.)
    v = variables_lib.Variable(100.)
    load_checkpoint = trackable_utils.Checkpoint(new_model, v=v)
    load_checkpoint.restore(save_path).assert_consumed()
    self.assertEqual(self.evaluate(v), 3)
    self.assertEqual(self.evaluate(new_model.separate_variable), 5)
    self.assertEqual(self.evaluate(load_checkpoint.save_counter), 1)

    # Case 3: Loading checkpoint where no root object is specified
    separate_variable = variables_lib.Variable(200.)
    v = variables_lib.Variable(100.)
    load_checkpoint = trackable_utils.Checkpoint(
        v=v, separate_variable=separate_variable)
    load_checkpoint.restore(save_path).assert_consumed()
    self.assertEqual(self.evaluate(v), 3)
    self.assertEqual(self.evaluate(new_model.separate_variable), 5)
    self.assertEqual(self.evaluate(load_checkpoint.save_counter), 1)

  def test_checkpoint_saved_model_compatibility(self):
    # [FUNCTION DOC]
    # Exports a model as a saved model directory, then restores it as a checkpoint (partial
    # restore). Asserts a nonexistent path raises a not-found error with the combined
    # checkpoint-or-saved-model message, outputs match after restore, and a deferred variable
    # takes the saved value 5.
    model = self._create_trackable()
    input_value = constant_op.constant([[3.]])
    expected_output = self.evaluate(model(input_value))
    model.deferred_variable = variables_lib.Variable(5.)
    saved_model_dir = os.path.join(self.get_temp_dir(), "saved_model")
    saved_model_save.save(model, saved_model_dir)

    new_model = self._create_trackable()
    load_checkpoint = trackable_utils.Checkpoint(new_model)

    with self.assertRaisesRegex(
        errors_impl.NotFoundError,
        "Error when restoring from checkpoint or SavedModel"):
      load_checkpoint.restore(saved_model_dir + "no").expect_partial()

    load_checkpoint.restore(saved_model_dir).expect_partial()
    self.assertAllClose(expected_output, new_model(input_value))

    new_model.deferred_variable = variables_lib.Variable(1.)
    self.assertEqual(self.evaluate(new_model.deferred_variable), 5)

  def test_deferred_dependency_avoids_reference_cycles(self):
    # [FUNCTION DOC]
    # Counts finalizer calls of a trackable subclass. Asserts that deleting a saved checkpoint and
    # root finalizes the child immediately (count 1), and that the same holds after a partial
    # restore with deferred dependencies (count 2), proving no reference cycles keep them alive.
    # Tests that there are no reference cycles when running garbage collection.
    # Python uses reference counts as the primary garbage collector, which will
    # not delete and finalize (__del__) objects in a cycle. The deletion is
    # eventually triggered by gc, which only runs when the garbage has reached
    # a certain threshold.

    delete_counter = 0

    class TrackableWithDel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: test fixture. Trackable that increments an enclosing counter when finalized.

      def __del__(self):
        # [FUNCTION DOC]
        # Finalizer: increments the enclosing delete counter by one. Returns nothing.
        nonlocal delete_counter
        delete_counter += 1

    x = autotrackable.AutoTrackable()
    x.v = variables_lib.Variable(100.)
    x.has_del = TrackableWithDel()

    checkpoint = trackable_utils.Checkpoint(x)
    checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    self.assertEqual(delete_counter, 0)
    del checkpoint
    del x
    self.assertEqual(delete_counter, 1)

    no_v = autotrackable.AutoTrackable()
    no_v.has_del = TrackableWithDel()
    checkpoint = trackable_utils.Checkpoint(no_v)
    checkpoint.restore(save_path).expect_partial()
    del checkpoint
    del no_v
    self.assertEqual(delete_counter, 2)

  def test_defer_objects_with_values_only(self):
    # [FUNCTION DOC]
    # Saves a graph with a branch holding a value, a branch with only empty children, and a root
    # variable. After restoring into an empty root, asserts only two deferred dependencies exist
    # (the valued branch and the variable) and that the count drops as the valued branch and the
    # variable are attached, while attaching the empty branch changes nothing.
    # Tests that deferred dependencies are only added if the node in the
    # object graph has children or checkpointed values.
    root = autotrackable.AutoTrackable()
    root.branch_with_value = autotrackable.AutoTrackable()
    root.branch_with_value.v = variables_lib.Variable(5.0)
    root.branch_no_value = autotrackable.AutoTrackable()
    root.branch_no_value.child = autotrackable.AutoTrackable()
    root.v = variables_lib.Variable(1.0)

    checkpoint = trackable_utils.Checkpoint(model=root)
    checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    new_root = autotrackable.AutoTrackable()
    checkpoint = trackable_utils.Checkpoint(model=new_root)
    checkpoint.restore(save_path)

    # root should have two nodes with values/children (`branch-with_value`/`v`).
    self.assertLen(new_root._deferred_dependencies, 2)

    new_root.branch_no_value = autotrackable.AutoTrackable()
    self.assertLen(new_root._deferred_dependencies, 2)

    new_root.branch_with_value = autotrackable.AutoTrackable()
    self.assertLen(new_root._deferred_dependencies, 1)

    new_root.v = variables_lib.Variable(1.0)
    self.assertEmpty(new_root._deferred_dependencies, 1)

  def test_root_arg(self):
    # [FUNCTION DOC]
    # Saves two checkpoints, each with a root object plus extra variables with swapped layouts,
    # then restores each into the other. Asserts values follow the names in the object graph
    # (root attributes and top-level keyword variables are interchangeable), e.g. root.v becomes 5.
    root = autotrackable.AutoTrackable()
    root.v = variables_lib.Variable(1)
    w = variables_lib.Variable(2)
    y = variables_lib.Variable(3)
    root_ckpt = trackable_utils.Checkpoint(root=root, w=w, y=y)

    root2 = autotrackable.AutoTrackable()
    root2.w = variables_lib.Variable(4)
    v2 = variables_lib.Variable(5)
    z = variables_lib.Variable(6)
    root2_ckpt = trackable_utils.Checkpoint(root=root2,
                                            v=v2,
                                            z=z)

    root_save_path = root_ckpt.save(os.path.join(self.get_temp_dir(),
                                                 "root_ckpt"))
    root2_save_path = root2_ckpt.save(os.path.join(self.get_temp_dir(),
                                                   "root2_ckpt"))

    root_ckpt.restore(root2_save_path)
    root2_ckpt.restore(root_save_path)

    self.assertEqual(root.v.numpy(), 5)
    self.assertEqual(w.numpy(), 4)
    self.assertEqual(y.numpy(), 3)

    self.assertEqual(root2.w.numpy(), 2)
    self.assertEqual(v2.numpy(), 1)
    self.assertEqual(z.numpy(), 6)

  def test_weakref_root(self):
    # [FUNCTION DOC]
    # Builds a checkpoint whose root is a weak reference to a trackable. Saves, mutates, restores
    # and asserts the value reverts to 1. After deleting the root, asserts the variable is only
    # referenced by the retained ref (reference count of two, including the temporary).
    root = autotrackable.AutoTrackable()
    root.v = variables_lib.Variable(1)
    ref = root.v.ref()

    ckpt = trackable_utils.Checkpoint(root=weakref.ref(root))
    save_path = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))
    root.v.assign(2)
    ckpt.restore(save_path)
    self.assertEqual(root.v.numpy(), 1)

    del root

    # Verifying if the variable is only referenced from `ref`.
    # We expect the reference counter to be 1, but `sys.getrefcount` reports
    # one higher reference counter because a temporary is created when we call
    # sys.getrefcount().  Hence check if the number returned is 2.
    # https://docs.python.org/3/library/sys.html#sys.getrefcount
    self.assertEqual(sys.getrefcount(ref.deref()), 2)

  def test_restore_incompatible_shape(self):
    # [FUNCTION DOC]
    # Saves a vector of length two and restores it into a variable of length one; asserts a value
    # error mentioning an incompatible tensor with shape.
    v = variables_lib.Variable([1.0, 1.0])
    w = variables_lib.Variable([1.0])
    ckpt = trackable_utils.Checkpoint(v=v)
    save_path = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))

    with self.assertRaisesRegex(ValueError, "incompatible tensor with shape"):
      trackable_utils.Checkpoint(v=w).restore(save_path)

  def test_save_restore_fspath(self):
    # [FUNCTION DOC]
    # Saves using a path-like object as the prefix and restores using a path-like result; asserts
    # the original variable still reads 1.
    v = variables_lib.Variable(1.0)
    w = variables_lib.Variable(0.0)
    ckpt = trackable_utils.Checkpoint(v=v)
    prefix = pathlib.Path(self.get_temp_dir()) / "ckpt"
    save_path = ckpt.save(prefix)
    save_path = pathlib.Path(save_path)
    ckpt2 = trackable_utils.Checkpoint(v=w)
    ckpt2.restore(save_path)
    self.assertEqual(ckpt.v.numpy(), 1.0)

  def test_read_write_fspath(self):
    # [FUNCTION DOC]
    # Writes with a path-like prefix and reads back with a path-like result into another
    # checkpoint; asserts the original variable still reads 1.
    # KNOWN ISSUE: the assertion checks the original checkpoint's variable, not the one that was
    # read into, so it does not verify that the read actually restored the value.
    v = variables_lib.Variable(1.0)
    w = variables_lib.Variable(0.0)
    ckpt = trackable_utils.Checkpoint(v=v)
    prefix = pathlib.Path(self.get_temp_dir()) / "ckpt"
    save_path = ckpt.write(prefix)
    save_path = pathlib.Path(save_path)
    ckpt2 = trackable_utils.Checkpoint(v=w)
    ckpt2.read(save_path)
    self.assertEqual(ckpt.v.numpy(), 1.0)

  @test_util.run_deprecated_v1
  def test_save_in_graph_but_no_session(self):
    # [FUNCTION DOC]
    # Deprecated v1 graph mode. With no default session, writing a checkpoint is expected to raise
    # a runtime error telling the user to create a session.
    v = variables_lib.Variable(1.0)
    ckpt = trackable_utils.Checkpoint(v=v)
    self.evaluate(v.initializer)
    prefix = pathlib.Path(self.get_temp_dir()) / "ckpt"
    with stack.default_session(None):
      with self.assertRaisesRegex(RuntimeError, "create a session"):
        ckpt.write(prefix)

  def test_ckpt_files_closed_after_restoration(self):
    # [FUNCTION DOC]
    # Skipped if the process-inspection library is unavailable. Saves and restores a checkpoint,
    # then asserts none of the process's open files refers to the checkpoint path.
    if not psutil_import_succeeded:
      self.skipTest(
          "psutil is required to check that we've closed our files.")
    root = autotrackable.AutoTrackable()
    root.v = variables_lib.Variable(1)
    ckpt = trackable_utils.Checkpoint(root=root)
    save_path = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))

    root2 = autotrackable.AutoTrackable()
    ckpt2 = trackable_utils.Checkpoint(root=root2)
    ckpt2.restore(save_path)

    proc = psutil.Process()
    for file in proc.open_files():
      self.assertNotIn(save_path, file[0])

  def testLookupCache(self):
    # [FUNCTION DOC]
    # Saves a model with three variables and restores into a subclass that records which names are
    # looked up with a cached dependency mapping; asserts all three names were looked up that way.
    """Ensure that Checkpoint.restore passes cached dependencies to lookup."""
    root = autotrackable.AutoTrackable()
    root.v1 = variables_lib.Variable(1)
    root.v2 = variables_lib.Variable(2)
    root.v3 = variables_lib.Variable(3)

    ckpt = trackable_utils.Checkpoint(model=root)
    save_path = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))

    called_with_cache = []

    class LookupOverride(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: test fixture. Trackable that records every dependency lookup made with a cache.
      def _lookup_dependency(self, name, cached_dependencies=None):
        # [FUNCTION DOC]
        # Inputs: name (string), cached_dependencies (optional mapping). Appends name to the
        # enclosing recording list when a cache was supplied, then delegates to the parent lookup
        # and returns its result.
        if cached_dependencies is not None:
          called_with_cache.append(name)
        return super()._lookup_dependency(name, cached_dependencies)

    root2 = LookupOverride()
    ckpt2 = trackable_utils.Checkpoint(model=root2)
    ckpt2.restore(save_path)
    self.assertCountEqual(called_with_cache, ["v1", "v2", "v3"])

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False)
    )
  def testCallbackWithManager(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Input: enable_async_ckpt (boolean, parameterized; chooses an async helper or a plain
    # checkpoint). Saves through a checkpoint manager with options holding two write callbacks
    # (one taking the save path, one taking nothing). Asserts the user's options still list two
    # callbacks (internal ones are not leaked), the manager's latest checkpoint is the save path,
    # and the callbacks ran in order exactly once each.
    """Tests experimental_write_callback with a checkpoint manager."""
    # 1. Define checkpoint and manager accordingly
    v = variables_lib.Variable(1.)
    if enable_async_ckpt:
      ckpt = async_checkpoint_helper.AsyncCheckpointHelper(
          trackable_utils.Checkpoint,
          v=v
      )
    else:
      ckpt = trackable_utils.Checkpoint(v=v)

    checkpoint_manager = checkpoint_management.CheckpointManager(
        checkpoint=ckpt,
        directory=os.path.join(self.get_temp_dir(), "ckpt"),
        max_to_keep=None,
        checkpoint_name="test-callback",
    )

    # 2. Define 2 callbacks, which will be executed in order as stated in the
    # expected behavior of `CheckpointOptions.experimental_write_callbacks`
    testing_list = []
    test_str = "callback 2 was here"
    # Define callback 1 that takes in 1 argument
    def my_callback_1(save_path):
      # [FUNCTION DOC]
      # HELPER: test callback. Input: save_path, string. Appends it to the shared recording list.
      testing_list.append(save_path)

    # Define callback 2 that takes in 0 argument
    def my_callback_2():
      # [FUNCTION DOC]
      # HELPER: test callback taking no arguments. Appends a fixed marker string to the shared
      # recording list.
      testing_list.append(test_str)

    # 3. Save with `options`
    options = checkpoint_options.CheckpointOptions(
        experimental_write_callbacks=[my_callback_1, my_callback_2]
    )
    save_path = checkpoint_manager.save(options=options)

    # 4. Assert results
    if enable_async_ckpt:
      checkpoint_manager.sync()  # otherwise callbacks may not have finished
    # Ensure that user's options is not mutated by internal mechanisms. Here,
    # we would internally register a callback `_record_and_sweep_state()`.
    # Users should not have access to it, hence length still being 2.
    self.assertLen(options.experimental_write_callbacks, 2)
    # Ensure `_record_and_sweep_state()` executes and sets `_latest_checkpoint`
    self.assertEqual(save_path, checkpoint_manager._latest_checkpoint)
    # Ensure my_callback_1 is executed first
    self.assertEqual(save_path, testing_list[0])
    # Ensure my_callback_2 is executed second
    self.assertEqual(test_str, testing_list[1])
    # Ensure nothing else is written to `testing_list`
    self.assertLen(testing_list, 2)

  @parameterized.named_parameters(
      ("_async_ckpt_save", True, False, True),
      ("_async_ckpt_write", True, False, False),
      ("_regular_ckptV1_save", False, True, True),
      ("_regular_ckptV1_write", False, True, False),
      ("_regular_ckptV2_save", False, False, True),
      ("_regular_ckptV2_write", False, False, False)
  )
  def testCallbackWithoutManager(self, enable_async_ckpt, use_v1, use_save):
    # [FUNCTION DOC]
    # Inputs (parameterized booleans): enable_async_ckpt (use the async helper), use_v1 (use the
    # v1 checkpoint class when not async), use_save (call save, otherwise write). Saves or writes
    # with options holding two write callbacks and no manager. Asserts the options still hold two
    # callbacks, and the callbacks ran in order exactly once each (path first, marker second).
    """Tests experimental_write_callback without using a checkpoint manager."""
    # Note that the underlying checkpoint of `AsyncCheckpoint.save()` will call
    # `Checkpoint.save()`.
    # The underlying checkpoint of `AsyncCheckpoint.write()` will call
    # `Checkpoint.write()`.

    # 1. Define checkpoint instance accordingly
    v = variables_lib.Variable(1.)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")

    if enable_async_ckpt:
      ckpt = async_checkpoint_helper.AsyncCheckpointHelper(
          trackable_utils.Checkpoint,
          v=v
      )
    else:
      if use_v1:
        ckpt = trackable_utils.CheckpointV1(v=v)
      else:
        ckpt = trackable_utils.Checkpoint(v=v)

    # 2. Define 2 callbacks, which will be executed in order as stated in the
    # expected behavior of `CheckpointOptions.experimental_write_callbacks`
    testing_list = []
    test_str = "callback 2 was here"
    # Define callback 1 that takes in 1 argument
    def my_callback_1(save_path):
      # [FUNCTION DOC]
      # HELPER: test callback. Input: save_path, string. Appends it to the shared recording list.
      testing_list.append(save_path)

    # Define callback 2 that takes in 0 argument
    def my_callback_2():
      # [FUNCTION DOC]
      # HELPER: test callback taking no arguments. Appends a fixed marker string to the shared
      # recording list.
      testing_list.append(test_str)

    # 3. Save with `options`
    options = checkpoint_options.CheckpointOptions(
        experimental_write_callbacks=[my_callback_1, my_callback_2]
    )
    if use_save:
      save_path = ckpt.save(prefix, options=options)
    else:
      save_path = ckpt.write(prefix, options=options)

    # 4. Assert results
    if enable_async_ckpt:
      ckpt.sync()  # otherwise callbacks may not have finished
    # Ensure that user's options is not mutated by internal mechanisms. Here,
    # if we are using regular checkpoint's save(), we would internally register
    # a callback `_update_checkpoint_state_internal()`. Users should not have
    # access to it, hence length still being 2.
    self.assertLen(options.experimental_write_callbacks, 2)
    # Ensure my_callback_1 is executed first
    self.assertEqual(save_path, testing_list[0])
    # Ensure my_callback_2 is executed second
    self.assertEqual(test_str, testing_list[1])
    # Ensure nothing else is written to `testing_list`
    self.assertLen(testing_list, 2)

  def test_callback_argument_error(self):
    # [FUNCTION DOC]
    # Builds checkpoint options with one valid one-argument callback and one two-argument
    # callback; asserts an assertion error is raised at construction.
    """Ensure passing in a callback with more than 1 argument raises error."""
    # Define callback 1 that takes in 1 argument
    def my_callback_1(save_path):
      # [FUNCTION DOC]
      # HELPER: valid test callback. Input: save_path. Returns it unchanged.
      return save_path

    # Define callback 2 that takes in 2 arguments (would raise error)
    def my_callback_2(save_path, another_argument):
      # [FUNCTION DOC]
      # HELPER: invalid test callback (two parameters). Returns both inputs as a pair.
      return save_path, another_argument

    with self.assertRaises(AssertionError):
      _ = checkpoint_options.CheckpointOptions(
          experimental_write_callbacks=[my_callback_1, my_callback_2]
      )

  def test_checkpoint_options_copyable(self):
    # [FUNCTION DOC]
    # Makes a shallow copy of checkpoint options (io device, async flag, one callback), modifies
    # the copy's scalar fields and appends a callback to its list. Asserts the original's device
    # and flag are unchanged and its callback list still has length one.
    # KNOWN ISSUE: the docstring says the options are copied with deep copy, but the code uses a
    # shallow copy; the original's callback list staying at length one depends on the options
    # class copying that list itself.
    """Ensure that `CheckpointOptions` can be copied with `copy.deepcopy()`."""
    def my_callback(save_path):
      # [FUNCTION DOC]
      # HELPER: test callback. Input: save_path. Returns it unchanged.
      return save_path

    def my_callback_2(save_path):
      # [FUNCTION DOC]
      # HELPER: test callback. Input: save_path, string. Returns it with a fixed suffix added.
      return save_path + "some string"

    options_original = checkpoint_options.CheckpointOptions(
        experimental_io_device="CPU:0",
        enable_async=True,
        experimental_write_callbacks=[my_callback]
    )

    options_copy = copy.copy(options_original)

    options_copy.enable_async = False
    options_copy.experimental_io_device = "CPU:1"
    options_copy.experimental_write_callbacks.append(my_callback_2)

    # Check that the original options instance is not affected
    self.assertEqual(options_original.experimental_io_device, "CPU:0")
    self.assertEqual(options_original.enable_async, True)
    self.assertLen(options_original.experimental_write_callbacks, 1)


class SerializeToTensorTest(test.TestCase):
  # [CLASS DOC]
  # Tests the tensor-based serialization hooks of trackables (custom keys and metadata, and
  # compatibility of reference variables with the v1 saver).

  @test_util.run_in_graph_and_eager_modes
  def test_keys_and_metadata(self):
    # [FUNCTION DOC]
    # Saves two multi-tensor trackables (values 1,2 and 3,4) under a checkpoint root. Asserts the
    # object metadata lists two attributes per node with the expected checkpoint keys, the raw
    # checkpoint reader returns the right values, and that restoring after zeroing the variables
    # brings back 1, 2, 3, 4.

    class MultiTensor(base.Trackable):
      # [CLASS DOC]
      # HELPER: test fixture. Trackable holding two variables that serializes itself as a mapping
      # of two named tensors and restores both from such a mapping.

      def __init__(self, v1, v2):
        # [FUNCTION DOC]
        # Inputs: v1 and v2, the two variables to store as attributes.
        self.v1 = v1
        self.v2 = v2

      def _serialize_to_tensors(self):
        # [FUNCTION DOC]
        # Returns a mapping from the names v1 and v2 to the corresponding variables.
        return {"v1": self.v1, "v2": self.v2}

      def _restore_from_tensors(self, restored_tensors):
        # [FUNCTION DOC]
        # Input: restored_tensors, mapping with entries v1 and v2. Returns a grouped assign
        # operation setting both variables from it.
        return control_flow_ops.group(
            self.v1.assign(restored_tensors["v1"]),
            self.v2.assign(restored_tensors["v2"]))

    root = MultiTensor(variables_lib.Variable(1), variables_lib.Variable(2))
    child = MultiTensor(variables_lib.Variable(3), variables_lib.Variable(4))

    ckpt = trackable_utils.Checkpoint(root=root, child=child)
    self.evaluate([root.v1.initializer, root.v2.initializer,
                   child.v1.initializer, child.v2.initializer])
    save_path = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))

    # Check the checkpoint contents and metadata.
    reader = checkpoint_utils.load_checkpoint(save_path)
    object_proto = trackable_utils.object_metadata(save_path)
    root_attributes = object_proto.nodes[0].attributes
    self.assertLen(root_attributes, 2)
    self.assertDictEqual(
        {"v1": "/.ATTRIBUTES/v1", "v2": "/.ATTRIBUTES/v2"},
        {attr.name: attr.checkpoint_key for attr in root_attributes})
    self.assertEqual(1, reader.get_tensor("/.ATTRIBUTES/v1"))
    self.assertEqual(2, reader.get_tensor("/.ATTRIBUTES/v2"))

    child_attributes = object_proto.nodes[1].attributes
    self.assertLen(child_attributes, 2)
    self.assertDictEqual(
        {"v1": "child/.ATTRIBUTES/v1", "v2": "child/.ATTRIBUTES/v2"},
        {attr.name: attr.checkpoint_key for attr in child_attributes})
    self.assertEqual(3, reader.get_tensor("child/.ATTRIBUTES/v1"))
    self.assertEqual(4, reader.get_tensor("child/.ATTRIBUTES/v2"))

    # Try restoring the checkpoint.
    self.evaluate([root.v1.assign(0), root.v2.assign(0), child.v1.assign(0),
                   child.v2.assign(0)])
    ckpt.restore(save_path).assert_consumed().run_restore_ops()

    self.assertAllEqual([1, 2, 3, 4],
                        self.evaluate([root.v1, root.v2, child.v1, child.v2]))

  @test_util.run_in_graph_and_eager_modes
  def test_reference_variable(self):
    # [FUNCTION DOC]
    # Creates a non-resource (reference) variable with value 5, writes an object-based checkpoint,
    # sets 10 and saves with the v1 saver. Asserts the object-based checkpoint reads back 5 and
    # the v1 saver restores 10, i.e. both formats work with reference variables.
    # Test that refvariable is compatible with tf1 saver / tf2 checkpoint.

    with self.cached_session() as sess:
      root = autotrackable.AutoTrackable()
      root.v = variable_v1.VariableV1(5, use_resource=False)
      sess.run(root.v.initializer)
      ckpt = trackable_utils.Checkpoint(root)
      ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
      ckpt.write(ckpt_path)

      sess.run(root.v.assign(10))
      saver = saver_lib.Saver(var_list=[root.v])
      save_path = saver.save(sess, os.path.join(self.get_temp_dir(), "saver"))

      ckpt.read(ckpt_path).assert_consumed().run_restore_ops()
      self.assertEqual(5, sess.run(root.v))

      saver.restore(sess, save_path)
      self.assertEqual(10, sess.run(root.v))


class TemplateTests(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Tests that variable-scope templates, including nested templates, save and restore through
  # object-based checkpoints.

  @test_util.run_in_graph_and_eager_modes
  def test_trackable_save_restore_nested(self):
    # [FUNCTION DOC]
    # Builds an outer template containing two inner templates, saves with values 20 and 25, then
    # restores into a differently named template. Asserts the outer template tracks two children,
    # each inner template tracks one variable, the restore is consumed, and values are 20, 25, 25
    # (the second inner template is called twice and shares its variable).

    def _inner_template():
      # [FUNCTION DOC]
      # HELPER: inner template body. Gets or creates a one-element zero-initialized variable named
      # v and returns it.
      v = variable_scope.get_variable(
          "v", shape=[1], initializer=init_ops.zeros_initializer())
      return v

    def _outer_template():
      # [FUNCTION DOC]
      # HELPER: outer template body. Creates two inner templates, calls the first once and the
      # second twice. Returns a pair: the two templates, and the three variables obtained.
      first_inner = template.make_template("i1", _inner_template)
      second_inner = template.make_template("i2", _inner_template)
      v1 = first_inner()
      v2 = second_inner()
      v3 = second_inner()
      return (first_inner, second_inner), (v1, v2, v3)

    with variable_scope.variable_scope("ignored"):
      save_template = template.make_template("s1", _outer_template)
      save_root = trackable_utils.Checkpoint(my_template=save_template)
      (inner_template_one, inner_template_two), _ = save_template()
    self.evaluate(inner_template_one.variables[0].assign([20.]))
    self.evaluate(inner_template_two.variables[0].assign([25.]))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = save_root.save(checkpoint_prefix)

    load_template = template.make_template("s2", _outer_template)
    load_root = trackable_utils.Checkpoint(my_template=load_template)
    status = load_root.restore(save_path)
    (inner_template_one, inner_template_two), (v1, v2, v3) = load_template()
    outer_template_dependencies = load_root.my_template._trackable_children()
    self.assertLen(outer_template_dependencies, 2)
    self.assertDictEqual({"i1": inner_template_one, "i2": inner_template_two},
                         outer_template_dependencies)
    self.assertLen(inner_template_one._trackable_children(), 1)
    self.assertIn("v", inner_template_one._trackable_children())
    self.assertLen(inner_template_two._trackable_children(), 1)
    self.assertIn("v", inner_template_two._trackable_children())
    status.assert_consumed().run_restore_ops()
    self.assertAllEqual([20.], self.evaluate(v1))
    self.assertAllEqual([25.], self.evaluate(v2))
    self.assertAllEqual([25.], self.evaluate(v3))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
