# [MODULE DOC]
# End-to-end tests for saving and loading with registered serializable classes and registered
# checkpoint savers. Defines helper classes (a variable subclass and a stack of such parts), a
# custom saver that writes only stacks, and a save/load cycling helper. Tests cover custom proto
# serialization, deserialization dependencies, round trips through SavedModel, checkpoint
# round trips, strict and non-strict predicate restore, saver ordering and migration from
# pre-registration checkpoints. Import-time side effects: registers the part and stack classes and
# a checkpoint saver named stacks in the global registries.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests saving with registered Trackable classes and checkpoint functions."""

import os
import tempfile

from absl.testing import parameterized

from google.protobuf import wrappers_pb2
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable


@registration.register_serializable()
class Part(resource_variable_ops.ResourceVariable):
  # [CLASS DOC]
  # HELPER: test fixture. A resource variable subclass registered as a serializable class, used as
  # an element of a stack. Constructed from an initial value; on SavedModel load it is recreated
  # with a fixed two-element zero value, and its data are restored by the custom stack saver.

  def __init__(self, value):
    # [FUNCTION DOC]
    # Initializes the variable directly from the given initial value (list or tensor) by calling
    # the internal argument-based initializer, bypassing the normal variable constructor.
    self._init_from_args(value)

  @classmethod
  def _deserialize_from_proto(cls, **kwargs):
    # [FUNCTION DOC]
    # Hook called by the SavedModel loader. Ignores all keyword arguments and returns a new part
    # holding two zeros; real values are filled in later by the registered stack restore function.
    return cls([0, 0])

  def _export_to_saved_model_graph(self, object_map, tensor_map, **kwargs):
    # [FUNCTION DOC]
    # Hook called when exporting to a SavedModel graph. Creates a new part with zeros of the same
    # shape and dtype, records it in the object map under this object and records the new handle
    # in the tensor map under this handle (both mutated). Returns a list with this part's handle.
    p = Part(array_ops.zeros(self.shape, self.dtype))
    object_map[self] = p
    tensor_map[self.handle] = p.handle
    return [self.handle]


@registration.register_serializable()
class Stack(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # HELPER: test fixture. A trackable holding a list of parts and exposing a compiled function
  # that stacks their values into one tensor. Registered as a serializable class; saved and
  # restored by the custom stacks saver.

  def __init__(self, parts=None):
    # [FUNCTION DOC]
    # Stores the given list of parts (optional, default nothing) as the parts attribute.
    self.parts = parts

  @def_function.function(input_signature=[])
  def value(self):
    # [FUNCTION DOC]
    # Compiled function with an empty input signature. Returns a single tensor made by stacking
    # the values of all parts along a new first axis.
    return array_ops_stack.stack(self.parts)


def get_tensor_slices(trackables):
  # [FUNCTION DOC]
  # HELPER: shared by the stack save and restore functions.
  # Input: mapping of unique prefix strings to trackable objects. Skips parts and, for each
  # stack, records a tensor name (prefix plus slash value), an empty slice spec, an identity copy
  # of the stack's value placed on the CPU device, and the stack itself. Returns four parallel
  # lists: tensor names, slice specs, tensors, and the stacks. Evaluates each stack's value.
  tensor_names = []
  shapes_and_slices = []
  tensors = []
  restored_trackables = []
  for obj_prefix, obj in trackables.items():
    if isinstance(obj, Part):
      continue  # only save stacks
    tensor_names.append(obj_prefix + "/value")
    shapes_and_slices.append("")
    x = obj.value()
    with ops.device("/device:CPU:0"):
      tensors.append(array_ops.identity(x))
    restored_trackables.append(obj)

  return tensor_names, shapes_and_slices, tensors, restored_trackables


def save_stacks_and_parts(trackables, file_prefix):
  # [FUNCTION DOC]
  # HELPER: registered custom save function for the stacks saver.
  # KNOWN ISSUE: the docstring says it saves stacks and parts, but parts are skipped; only the
  # stacks' stacked values are written.
  # Inputs: mapping of prefixes to trackables and the file prefix string. Writes one checkpoint
  # shard containing the stack values using the save op. Returns the file prefix.
  """Save stack and part objects to a checkpoint shard."""
  tensor_names, shapes_and_slices, tensors, _ = get_tensor_slices(trackables)
  io_ops.save_v2(file_prefix, tensor_names, shapes_and_slices, tensors)
  return file_prefix


def restore_stacks_and_parts(trackables, merged_prefix):
  # [FUNCTION DOC]
  # HELPER: registered custom restore function for the stacks saver.
  # Inputs: mapping of prefixes to trackables and the merged checkpoint prefix. Reads the stacked
  # value of every stack from the checkpoint with the restore op, reshapes it to the stack's
  # current value shape, unstacks it and assigns each slice to the matching part variable.
  # Returns nothing; mutates the part variables.
  tensor_names, shapes_and_slices, tensors, restored_trackables = (
      get_tensor_slices(trackables))
  dtypes = [t.dtype for t in tensors]
  restored_tensors = io_ops.restore_v2(merged_prefix, tensor_names,
                                       shapes_and_slices, dtypes)
  for trackable, restored_tensor in zip(restored_trackables, restored_tensors):
    expected_shape = trackable.value().get_shape()
    restored_tensor = array_ops.reshape(restored_tensor, expected_shape)
    parts = array_ops_stack.unstack(restored_tensor)
    for part, restored_part in zip(trackable.parts, parts):
      part.assign(restored_part)


registration.register_checkpoint_saver(
    name="stacks",
    predicate=lambda x: isinstance(x, (Stack, Part)),
    save_fn=save_stacks_and_parts,
    restore_fn=restore_stacks_and_parts)


def cycle(obj, cycles, signatures=None, options=None):
  # [FUNCTION DOC]
  # HELPER: round-trip utility for the tests.
  # Repeats the given number of times: save the current object to a fresh temporary directory
  # (preferring a GPU device if present), load it back, and use the loaded object and its
  # signatures for the next round. Inputs: obj to save, cycles count, optional signatures and
  # save options. Returns the object loaded in the last cycle (unbound error if cycles is zero).
  # Side effects: creates temporary directories and writes SavedModels.
  to_save = obj
  for _ in range(cycles):
    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    # If available, we'll run the save and restore preferring the GPU. This
    # just makes sure we aren't throwing errors and have enough
    # device("CPU") blocks to satisfy the placer.
    with test_util.use_gpu():
      save.save(to_save, path, signatures, options=options)
      loaded = load.load(path)
      signatures = loaded.signatures
    to_save = loaded
  return loaded


@parameterized.named_parameters(
    dict(testcase_name="ReloadOnce", cycles=1),
    dict(testcase_name="ReloadTwice", cycles=2),
    dict(testcase_name="ReloadThrice", cycles=3))
class SavedModelTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case (one, two and three save/load cycles) for SavedModel round trips of
  # objects using registered serializable classes and registered checkpoint savers.

  def test_registered_serializable(self, cycles):
    # [FUNCTION DOC]
    # Scenario: registers a module class (name unique per cycle count) that serializes its name
    # into a string-value proto and rebuilds itself from it. A module named a with variable set to
    # five is cycled through save and load. Asserts the loaded object is of the registered class,
    # the variable is five and the name is a.

    @registration.register_serializable(name=f"SaveAndLoad{cycles}")
    class Module(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: locally registered module holding a variable and a name that round-trips through
      # a custom string proto.

      def __init__(self, name="module"):
        # [FUNCTION DOC]
        # Creates a variable of value one and stores the name (default module).
        self.v = variables.Variable(1.)
        self.name = name

      def _serialize_to_proto(self, **unused_kwargs):
        # [FUNCTION DOC]
        # Returns a string-value proto containing the module's name. Ignores keyword arguments.
        return wrappers_pb2.StringValue(value=self.name)

      @classmethod
      def _deserialize_from_proto(cls, proto, **unused_kwargs):
        # [FUNCTION DOC]
        # Rebuilds a module from the packed proto: if it is a string-value proto, unpacks it and
        # constructs the class with that name; otherwise raises an assertion error describing the
        # expected and received proto types. Ignores keyword arguments.
        if proto.Is(wrappers_pb2.StringValue.DESCRIPTOR):
          unpacked = wrappers_pb2.StringValue()
          proto.Unpack(unpacked)
          return cls(name=unpacked.value)
        raise AssertionError(
            "Did not receive proto of correct type during deserialization. "
            f"Expected type {wrappers_pb2.StringValue.DESCRIPTOR.full_name}, "
            f"got {proto.TypeName()}")

    m = Module("a")
    m.v.assign(5)
    loaded = cycle(m, cycles)
    self.assertIsInstance(loaded, Module)
    self.assertEqual(5, loaded.v.numpy())
    self.assertEqual("a", loaded.name)

  def test_none_proto(self, cycles):
    # [FUNCTION DOC]
    # Scenario: a registered module that does not override proto serialization (default gives no
    # proto) but whose deserialization hook returns a module named deserialized. After cycling a
    # module named a, asserts the deserialization hook received an empty proto, the loaded object
    # is of the registered class, the variable is five and the name is deserialized.

    @registration.register_serializable(name=f"NoneProto{cycles}")
    class Module(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: locally registered module relying on the default (empty) serialization proto.

      def __init__(self, name="module"):
        # [FUNCTION DOC]
        # Creates a variable of value one and stores the name (default module).
        self.v = variables.Variable(1.)
        self.name = name

      # Leave _serialize_to_proto as the default (returns `None`).

      @classmethod
      def _deserialize_from_proto(cls, proto, **unused_kwargs):
        # [FUNCTION DOC]
        # Asserts (through the enclosing test case) that the proto has zero size, then returns a
        # new module named deserialized. Ignores keyword arguments.
        self.assertEqual(proto.ByteSize(), 0)
        return cls("deserialized")

    m = Module("a")
    m.v.assign(5)
    loaded = cycle(m, cycles)
    self.assertIsInstance(loaded, Module)
    self.assertEqual(5, loaded.v.numpy())
    self.assertEqual("deserialized", loaded.name)

  def test_deserialization_dependencies(self, cycles):
    # [FUNCTION DOC]
    # Scenario: a registered module declares its variable as a deserialization dependency, and its
    # deserialization hook receives the loaded dependencies and builds the module from the
    # variable. After cycling a module with variable five, asserts the dependency was provided to
    # the hook, the loaded object is of the registered class, and the variable is five.
    @registration.register_serializable(name=f"Dependency{cycles}")
    class Module(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: locally registered module that wraps a variable (given or newly created) and
      # reports it as its deserialization dependency.

      def __init__(self, v=None):
        # [FUNCTION DOC]
        # Stores the given variable, or creates a new variable of value one if none is given.
        self.v = v if v is not None else variables.Variable(1.)

      def _deserialization_dependencies(self, children):
        # [FUNCTION DOC]
        # Ignores the children argument and returns a mapping with the single entry v for the
        # module's variable, declaring it must be loaded before this object is rebuilt.
        del children  # Unused.
        return {"v": self.v}

      @classmethod
      def _deserialize_from_proto(cls, dependencies, **unused_kwargs):
        # [FUNCTION DOC]
        # Asserts (through the enclosing test case) that the dependencies mapping contains v, then
        # returns a new module built around that variable. Ignores other keyword arguments.
        self.assertIn("v", dependencies)
        return cls(v=dependencies["v"])

    m = Module()
    m.v.assign(5)
    loaded = cycle(m, cycles)
    self.assertIsInstance(loaded, Module)
    self.assertEqual(5, loaded.v.numpy())

  def test_registered_saver(self, cycles):
    # [FUNCTION DOC]
    # Scenario: a stack of three parts is cycled through save and load, which uses the registered
    # stacks checkpoint saver. Asserts the loaded stack's stacked value equals the original's.
    p1 = Part([1, 4])
    p2 = Part([2, 5])
    p3 = Part([3, 6])
    s = Stack([p1, p2, p3])
    loaded = cycle(s, cycles)
    self.assertAllEqual(s.value(), loaded.value())


class SingleCycleTest(test.TestCase):
  # [CLASS DOC]
  # Test case for registered checkpoint savers outside the parameterized cycling: graph-mode
  # loading, plain checkpoints, v1 SavedModel compatibility, predicate strictness, saver call
  # ordering and backward compatibility of migrated objects.

  @test_util.deprecated_graph_mode_only
  def test_registered_saver_fails_in_saved_model_graph_mode(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode-only test): inside an eager context, save a stack of three parts to a
    # directory. Then load it in graph mode. Expects a not-implemented error saying a registered
    # checkpoint saver is not supported in graph mode.
    with context.eager_mode():
      p1 = Part([1, 4])
      p2 = Part([2, 5])
      p3 = Part([3, 6])
      s = Stack([p1, p2, p3])
      save_dir = os.path.join(self.get_temp_dir(), "save_dir")
      save.save(s, save_dir)

    with self.assertRaisesRegex(
        NotImplementedError,
        "registered checkpoint saver is not supported in graph mode"):
      load.load(save_dir)

  def test_registered_saver_checkpoint(self):
    # [FUNCTION DOC]
    # Scenario: two stacks over the same parts in different orders are written to a checkpoint
    # and then deleted. A fresh stack of zero parts is restored from the first stack's key and
    # asserted to equal the first saved value; restoring from the second stack's key is asserted
    # to equal the second saved value (partial restore expected).
    p1 = Part([1, 4])
    p2 = Part([2, 5])
    p3 = Part([3, 6])
    s = Stack([p1, p2, p3])
    s2 = Stack([p3, p1, p2])

    expected_value_s = s.value()
    expected_value_s2 = s2.value()

    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    util.Checkpoint(s=s, s2=s2).write(ckpt_path)

    del s, s2, p1, p2, p3

    restore_s = Stack([Part([0, 0]) for _ in range(3)])
    util.Checkpoint(s=restore_s).read(ckpt_path).expect_partial()
    self.assertAllEqual(expected_value_s, restore_s.value())
    util.Checkpoint(s2=restore_s).read(ckpt_path).expect_partial()
    self.assertAllEqual(expected_value_s2, restore_s.value())

  def test_compatible_with_v1_savedmodel(self):
    # [FUNCTION DOC]
    # Scenario: a stack of three parts is saved with a serving signature returning its value.
    # The SavedModel is then loaded with the v1 loader in a new graph and session. Asserts running
    # the signature's value output equals the value computed before saving.
    p1 = Part([1, 4])
    p2 = Part([2, 5])
    p3 = Part([3, 6])
    s = Stack([p1, p2, p3])
    save_path = os.path.join(self.get_temp_dir(), "savedmodel")

    @def_function.function(input_signature=[])
    def serve():
      # [FUNCTION DOC]
      # HELPER: serving signature for the compatibility test. Takes no inputs and returns a
      # mapping with one entry, value, holding the stack's stacked value.
      return {"value": s.value()}

    exported_value = serve()["value"]

    save.save(s, save_path, signatures=serve)
    with ops.Graph().as_default(), session.Session() as sess:
      metagraph = loader.load(sess, ["serve"], save_path)
      value_output = metagraph.signature_def["serving_default"].outputs["value"]
      self.assertAllEqual(exported_value, sess.run(value_output.name))

  def test_non_strict_predicate(self):
    # [FUNCTION DOC]
    # Scenario: registers a saver (no-op functions) for a local class with strict predicate
    # restore turned off. A checkpoint is written from an instance, then read into a plain
    # trackable that fails the predicate. Expects the read to finish without raising an error.
    class NonStrictPredicateClass(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: empty trackable class selected by the saver predicate in this test.
      pass
    registration.register_checkpoint_saver(
        name="NonStrictPredicate",
        predicate=lambda x: isinstance(x, NonStrictPredicateClass),
        save_fn=lambda **kwargs: [],
        restore_fn=lambda **kwargs: None,
        strict_predicate_restore=False)

    root = NonStrictPredicateClass()
    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    util.Checkpoint(root).write(ckpt_path)

    root2 = autotrackable.AutoTrackable()
    # This should run without throwing an error.
    util.Checkpoint(root2).read(ckpt_path)

  def test_strict_predicate(self):
    # [FUNCTION DOC]
    # Scenario: same setup as the non-strict test but with strict predicate restore turned on.
    # Reading the checkpoint into a plain trackable that fails the predicate is expected to raise
    # a value error saying the saver cannot be used.
    class StrictPredicateClass(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: empty trackable class selected by the saver predicate in this test.
      pass
    registration.register_checkpoint_saver(
        name="StrictPredicate",
        predicate=lambda x: isinstance(x, StrictPredicateClass),
        save_fn=lambda **kwargs: [],
        restore_fn=lambda **kwargs: None,
        strict_predicate_restore=True)

    root = StrictPredicateClass()
    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    util.Checkpoint(root).write(ckpt_path)

    root2 = autotrackable.AutoTrackable()
    with self.assertRaisesRegex(ValueError, "saver cannot be used"):
      util.Checkpoint(root2).read(ckpt_path)

  def test_registered_saver_is_called_before_save_after_load(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests save and restore ordering is checked, but the checkpoint is
    # only written; the restore function is never run, so its assertions never execute.
    # Scenario (skipped unless eager): registers a saver whose save function asserts the checkpoint
    # directory is still empty when it runs, for a local class holding a variable of value 123.
    # Writes a checkpoint from an instance; the assertions inside the save function are the check.
    if not context.executing_eagerly():
      self.skipTest("This test must run under eager mode.")

    class RestoreClass(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: empty trackable class selected by the saver predicate in this test.
      pass
    def save_fn(trackables, file_prefix):
      # [FUNCTION DOC]
      # HELPER: registered save function. Ignores the trackables, lists the directory containing
      # the file prefix (an eager string tensor) and asserts it is empty. Returns nothing.
      del trackables  # Unused.
      # Check that directory is empty
      files = gfile.ListDirectory(os.path.dirname(file_prefix.numpy()))
      self.assertEmpty(files)

    def restore_fn(trackables, merged_prefix):
      # [FUNCTION DOC]
      # KNOWN ISSUE: takes the next item of the dictionary values view, which is not an iterator,
      # so it would raise a type error if it were ever called (it is not called in this test).
      # HELPER: registered restore function. Ignores the prefix, takes the root trackable and
      # asserts its variable equals 123. Returns nothing.
      del merged_prefix  # Unused.
      root = next(trackables.values())
      self.assertEqual(root.v.numpy(), 123)

    registration.register_checkpoint_saver(
        name="OptionalRestore",
        predicate=lambda x: isinstance(x, RestoreClass),
        save_fn=save_fn,
        restore_fn=restore_fn)

    root = RestoreClass()
    root.v = variables.Variable(123.0)

    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    util.Checkpoint(root).write(ckpt_path)

  def test_migration_backwards_compatibility(self):
    # [FUNCTION DOC]
    # Scenario: a class without a registered saver serializes its name as a tensor; a second
    # class uses a registered saver that writes the name under a suffixed key and, on restore,
    # falls back to the unsuffixed key if the suffixed one is not found. A checkpoint is written
    # from an instance of each; then the old-style checkpoint is read into the registered-saver
    # object. Asserts the object's name equals the old checkpoint's saved name (bytes before).
    # Tests that objects migrated to using the advanced saver registration can
    # use pre-migration checkpoints.

    class NoRegisteredSaver(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: stands for a pre-migration object. Holds a name and serializes it as a string
      # tensor through the standard tensor-serialization hook.

      def __init__(self, name):
        # [FUNCTION DOC]
        # Stores the name string.
        self.name = name

      def _serialize_to_tensors(self):
        # [FUNCTION DOC]
        # Returns a mapping with one entry, name, holding the name as a string constant.
        return {"name": constant_op.constant(self.name)}

    class RegisteredSaver(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: stands for a migrated object saved through the registered saver in this test.
      # Holds only a name.

      def __init__(self, name):
        # [FUNCTION DOC]
        # Stores the name string.
        self.name = name

    def _get_tensors(trackables, append_name=True):
      # [FUNCTION DOC]
      # HELPER: shared by the migration saver's functions. Input: mapping of prefixes to
      # trackables and a flag (default true) choosing whether the tensor name is the prefix plus
      # the word name or just the prefix. Returns four parallel lists: tensor names, empty slice
      # specs, string constant tensors of each object's name, and the objects.
      tensor_names = []
      shapes_and_slices = []
      tensors = []
      restored_trackables = []
      for obj_prefix, obj in trackables.items():
        tensor_names.append(obj_prefix + "name" if append_name else obj_prefix)
        shapes_and_slices.append("")
        tensors.append(constant_op.constant(obj.name))
        restored_trackables.append(obj)
      return tensor_names, shapes_and_slices, tensors, restored_trackables

    def save_fn(trackables, file_prefix):
      # [FUNCTION DOC]
      # HELPER: registered save function of the migrated saver. Writes each object's name under
      # the suffixed key to one shard with the save op and returns the file prefix.
      tensor_names, shapes_and_slices, tensors, _ = _get_tensors(trackables)
      io_ops.save_v2(file_prefix, tensor_names, shapes_and_slices, tensors)
      return file_prefix

    def restore_fn(trackables, merged_prefix):
      # [FUNCTION DOC]
      # HELPER: registered restore function of the migrated saver. Tries to restore names using
      # the suffixed keys; if the checkpoint reports a not-found error (a pre-migration
      # checkpoint), retries with the unsuffixed keys. Assigns each restored tensor to the
      # object's name attribute (mutating it). Returns nothing.
      tensor_names, shapes_and_slices, tensors, restored_trackables = (
          _get_tensors(trackables))
      dtypes = [t.dtype for t in tensors]
      try:
        restored_tensors = io_ops.restore_v2(merged_prefix, tensor_names,
                                             shapes_and_slices, dtypes)
      except errors_impl.NotFoundError:
        # If a NotFoundError is caught, then it means that the checkpoint
        # was written prior to the saver registration migration.
        tensor_names, shapes_and_slices, tensors, restored_trackables = (
            _get_tensors(trackables, append_name=False))
        restored_tensors = io_ops.restore_v2(merged_prefix, tensor_names,
                                             shapes_and_slices, dtypes)
      for trackable, name_tensor in zip(restored_trackables, restored_tensors):
        trackable.name = name_tensor

    registration.register_checkpoint_saver(
        name="MigratedSaver",
        predicate=lambda x: isinstance(x, RegisteredSaver),
        save_fn=save_fn,
        restore_fn=restore_fn,
    )

    before = NoRegisteredSaver("before")
    after = RegisteredSaver("after")
    before_ckpt_path = os.path.join(self.get_temp_dir(), "before_ckpt")
    util.Checkpoint(before).write(before_ckpt_path)

    after_ckpt = util.Checkpoint(after)
    after_ckpt_path = os.path.join(self.get_temp_dir(), "after_ckpt")
    after_ckpt.write(after_ckpt_path)

    # Try loading the pre-migrated checkpoint to the migrated object.
    after_ckpt.read(before_ckpt_path)
    self.assertEqual(b"before", self.evaluate(after.name))


if __name__ == "__main__":
  test.main()
