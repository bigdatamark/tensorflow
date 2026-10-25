# [MODULE DOC]
# Tests for the session-based SavedModel loader class. Exports small graphs (two variables whose
# sum is exposed by two signatures, optionally with an init op assigning a new value) using both
# the public and the internal builder classes, then exercises load, graph import, import scopes,
# variable restore, init op execution, tag lookup and parse errors. Defines module-level path
# constants and helpers and a parameterized test class.
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
# ==============================================================================
"""Tests for SavedModelLoader class."""

import os
import shutil

from absl.testing import parameterized

from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.training import saver as tf_saver


def _get_export_dir(label):
  # [FUNCTION DOC]
  # HELPER: path builder for test export directories.
  # Returns the path of a subdirectory named by label (string) under the test temp directory.
  return os.path.join(test.get_temp_dir(), label)


def _tensor_name(name):
  # [FUNCTION DOC]
  # HELPER: maps a variable name (string) to the name of the tensor that reads its value. When
  # resource variables are enabled the read op output name is used, otherwise the variable's
  # first output tensor name. Returns the tensor name string.
  if resource_variables_toggle.resource_variables_enabled():
    return name + "/Read/ReadVariableOp:0"
  return name + ":0"


SIMPLE_ADD_SAVED_MODEL = _get_export_dir("simple_add_saved_model")
SAVED_MODEL_WITH_MAIN_OP = _get_export_dir("saved_model_with_main_op")


def build_graph_helper():
  # [FUNCTION DOC]
  # HELPER: fixture graph builder shared by the export methods.
  # Builds a new graph with variables x (5) and y (11), their sum z, and two signature
  # definitions: foo (input x, output z) and bar (inputs x and y, output z). Returns a triple: the
  # graph, a mapping from signature names to definitions, and the variable y.
  g = ops.Graph()
  with g.as_default():
    x = variable_v1.VariableV1(5, name="x")
    y = variable_v1.VariableV1(11, name="y")
    z = x + y

    foo_sig_def = signature_def_utils.build_signature_def({
        "foo_input": utils.build_tensor_info(x)
    }, {"foo_output": utils.build_tensor_info(z)})
    bar_sig_def = signature_def_utils.build_signature_def({
        "bar_x": utils.build_tensor_info(x),
        "bar_y": utils.build_tensor_info(y)
    }, {"bar_z": utils.build_tensor_info(z)})
  return g, {"foo": foo_sig_def, "bar": bar_sig_def}, y


@parameterized.parameters((saved_model_builder.SavedModelBuilder,),
                          (saved_model_builder._SavedModelBuilder,))
class SavedModelLoaderTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the session-based loader class. The whole class is parameterized over two builder
  # classes (public and internal), so each test method receives the builder class as an argument.
  # Shared fixtures: export helpers writing models to fixed temp directories, removed in tear down.

  def export_simple_graph(self, builder_cls):
    # [FUNCTION DOC]
    # HELPER: fixture exporter. Builds the standard graph, initializes variables in a session and
    # saves one MetaGraph tagged foo_graph with both signatures using the given builder class,
    # into the simple add model directory. Returns nothing; writes files.
    g, sig_def_map, _ = build_graph_helper()
    with session.Session(graph=g) as sess:
      self.evaluate(variables.global_variables_initializer())
      builder = builder_cls(SIMPLE_ADD_SAVED_MODEL)
      builder.add_meta_graph_and_variables(sess, ["foo_graph"], sig_def_map)
      builder.save()

  def export_graph_with_main_op(self, builder_cls):
    # [FUNCTION DOC]
    # HELPER: fixture exporter. Like the simple export, but also adds an init op that assigns 7 to
    # variable y. The init op is passed as the init op argument for the internal builder class and
    # as the main op argument for the public one. Saves into the main op model directory.
    g, sig_def_map, y = build_graph_helper()
    with session.Session(graph=g) as sess:
      self.evaluate(variables.global_variables_initializer())
      assign_op = control_flow_ops.group(state_ops.assign(y, 7))

      builder = builder_cls(SAVED_MODEL_WITH_MAIN_OP)

      if builder_cls == saved_model_builder._SavedModelBuilder:
        builder.add_meta_graph_and_variables(
            sess, ["foo_graph"], sig_def_map, init_op=assign_op)
      else:
        builder.add_meta_graph_and_variables(
            sess, ["foo_graph"], sig_def_map, main_op=assign_op)
      builder.save()

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs the parent tear down, then recursively deletes the test temp directory (ignoring
    # errors) so exported models do not leak between tests.
    super(SavedModelLoaderTest, self).tearDown()
    shutil.rmtree(test.get_temp_dir(), ignore_errors=True)

  def test_load_function(self, builder_cls):
    # [FUNCTION DOC]
    # In graph mode, exports the simple model, loads it with the combined load into a fresh
    # session and asserts x is 5 and y is 11. Then exports the model with the init op and asserts
    # that after loading x is 5 and y is 7 (the init op ran).
    # Force test to run in graph mode.
    # The SavedModelLoader.load method is a v1-only API that requires a session
    # to work.
    with ops.Graph().as_default():
      self.export_simple_graph(builder_cls)
      loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
      with self.session(graph=ops.Graph()) as sess:
        loader.load(sess, ["foo_graph"])
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(11, sess.run(_tensor_name("y")))

      self.export_graph_with_main_op(builder_cls)
      loader2 = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      with self.session(graph=ops.Graph()) as sess:
        loader2.load(sess, ["foo_graph"])
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(7, sess.run(_tensor_name("y")))

  def test_load_graph(self, builder_cls):
    # [FUNCTION DOC]
    # Exports the simple model and imports only its graph into a new graph. Asserts x and y exist,
    # the sum tensor does not (lookup by its name raises a key error), and that evaluating x and y
    # raises a failed-precondition error because variables were not initialized or restored.
    self.export_simple_graph(builder_cls)
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    graph = ops.Graph()
    loader.load_graph(graph, ["foo_graph"])

    x = graph.get_tensor_by_name(_tensor_name("x"))
    y = graph.get_tensor_by_name(_tensor_name("y"))

    with self.assertRaises(KeyError):
      graph.get_tensor_by_name(_tensor_name("z"))

    with graph.as_default(), self.session():
      # Check that x and y are not initialized
      with self.assertRaises(errors.FailedPreconditionError):
        self.evaluate(x)
      with self.assertRaises(errors.FailedPreconditionError):
        self.evaluate(y)

  def test_load_with_import_scope(self, builder_cls):
    # [FUNCTION DOC]
    # Imports the init-op model under the scope baz. Asserts a default saver fails to restore
    # (not-found error) while the returned saver works. For the internal builder, running init ops
    # without the scope raises a not-found error and with the scope succeeds; for the public
    # builder it succeeds directly. Asserts baz x is 5 and baz y is 7. Then checks the combined
    # load with scope baa gives the same values.
    # Force test to run in graph mode.
    # The SavedModelLoader.restore_variables and SavedModelLoader.run_init_ops
    # methods are v1-only APIs that require a session to work.
    with ops.Graph().as_default():
      self.export_graph_with_main_op(builder_cls)
      loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      with self.session(graph=ops.Graph()) as sess:
        saver, _ = loader.load_graph(
            sess.graph, ["foo_graph"], import_scope="baz")

        # The default saver should not work when the import scope is set.
        with self.assertRaises(errors.NotFoundError):
          loader.restore_variables(sess, tf_saver.Saver())

        loader.restore_variables(sess, saver)

        if builder_cls == saved_model_builder._SavedModelBuilder:
          with self.assertRaises(errors.NotFoundError):
            loader.run_init_ops(sess, ["foo_graph"])
          loader.run_init_ops(sess, ["foo_graph"], import_scope="baz")
        else:
          loader.run_init_ops(sess, ["foo_graph"])

        self.assertEqual(5, sess.run(_tensor_name("baz/x")))
        self.assertEqual(7, sess.run(_tensor_name("baz/y")))

      # Test combined load function.
      loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      with self.session(graph=ops.Graph()) as sess:
        loader.load(sess, ["foo_graph"], import_scope="baa")
        self.assertEqual(5, sess.run(_tensor_name("baa/x")))
        self.assertEqual(7, sess.run(_tensor_name("baa/y")))

  def test_restore_variables(self, builder_cls):
    # [FUNCTION DOC]
    # In a session with newly created variables x and y (zero) and their product, asserts that
    # restoring with no saver raises a value error, and that restoring with a default saver loads
    # the checkpoint values so the product is 55.
    # Force test to run in graph mode.
    # The SavedModelLoader.restore_variables method is a v1-only API requiring a
    # session to work.
    with ops.Graph().as_default():
      self.export_graph_with_main_op(builder_cls)
      loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      with self.session() as sess:
        x = variable_v1.VariableV1(0, name="x")
        y = variable_v1.VariableV1(0, name="y")
        z = x * y

        self.evaluate(variables.global_variables_initializer())

        # There are variables to restore, so a saver must be created.
        with self.assertRaises(ValueError):
          loader.restore_variables(sess, None)

        loader.restore_variables(sess, tf_saver.Saver())
        self.assertEqual(55, self.evaluate(z))

  def test_run_init_op(self, builder_cls):
    # [FUNCTION DOC]
    # Imports the init-op model graph, restores variables (x is 5, y is 11), then runs the init ops
    # and asserts y becomes 7 while x stays 5.
    # Force test to run in graph mode.
    # The SavedModelLoader.restore_variables and SavedModelLoader.run_init_ops
    # methods are v1-only APIs that require a session to work.
    with ops.Graph().as_default():
      self.export_graph_with_main_op(builder_cls)
      loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      graph = ops.Graph()
      saver, _ = loader.load_graph(graph, ["foo_graph"])
      with self.session(graph=graph) as sess:
        loader.restore_variables(sess, saver)
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(11, sess.run(_tensor_name("y")))

        loader.run_init_ops(sess, ["foo_graph"])
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(7, sess.run(_tensor_name("y")))

  def test_parse_saved_model(self, builder_cls):
    # [FUNCTION DOC]
    # Exports the simple model and asserts the MetaGraph selected by the foo_graph tag exists and
    # contains both the foo and bar signature definitions.
    self.export_simple_graph(builder_cls)
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    meta_graph = loader.get_meta_graph_def_from_tags(["foo_graph"])
    self.assertIsNotNone(meta_graph)
    self.assertIn("foo", meta_graph.signature_def)
    self.assertIn("bar", meta_graph.signature_def)

  def test_load_invalid_meta_graph(self, builder_cls):
    # [FUNCTION DOC]
    # Asserts that looking up a MetaGraph with an empty tag list, a list with an empty string, or
    # an unknown tag each raises a runtime error.
    self.export_simple_graph(builder_cls)
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags([])
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags([""])
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags(["not_a_graph"])

  def test_load_saved_model_with_no_variables(self, builder_cls):
    # [FUNCTION DOC]
    # Saves a model whose variables are in a non-global collection together with an explicit
    # saver. Asserts that importing the graph yields a saver even though no saveable objects are
    # registered, and that the combined load still restores x as 5 and y as 11.
    """Test that SavedModel runs saver when there appear to be no variables.

    When no variables are detected, this may mean that the variables were saved
    to different collections, or the collections weren't saved to the
    SavedModel. If the SavedModel MetaGraphDef contains a saver, it should still
    run in either of these cases.

    Args:
      builder_cls: SavedModelBuilder or _SavedModelBuilder class
    """
    # Force test to run in graph mode.
    # The SavedModelBuilder.add_meta_graph_and_variables and
    # SavedModelLoader.load methods are v1-only APIs that require a session to
    # work.
    with ops.Graph().as_default():
      path = _get_export_dir("no_variable_saved_model")
      with session.Session(graph=ops.Graph()) as sess:
        x = variable_v1.VariableV1(
            5, name="x", collections=["not_global_variable"])
        y = variable_v1.VariableV1(
            11, name="y", collections=["not_global_variable"])
        self.assertFalse(variables._all_saveable_objects())
        z = x + y
        self.evaluate(variables.variables_initializer([x, y]))

        foo_sig_def = signature_def_utils.build_signature_def(
            {"foo_input": utils.build_tensor_info(x)},
            {"foo_output": utils.build_tensor_info(z)})

        builder = saved_model_builder.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            sess, ["foo_graph"], {"foo": foo_sig_def},
            saver=tf_saver.Saver([x, y]))
        builder.save()

      loader = loader_impl.SavedModelLoader(path)
      with self.session(graph=ops.Graph()) as sess:
        saver, _ = loader.load_graph(sess.graph, ["foo_graph"])
        self.assertFalse(variables._all_saveable_objects())
        self.assertIsNotNone(saver)

      with self.session(graph=ops.Graph()) as sess:
        loader.load(sess, ["foo_graph"])
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(11, sess.run(_tensor_name("y")))

  def test_load_saved_model_graph_with_return_elements(self, builder_cls):
    # [FUNCTION DOC]
    # Imports the graph requesting return elements y and x, asserting the returned list holds those
    # tensors in the requested order; requesting a nonexistent tensor raises a value error saying
    # it was not found in the graph.
    """Ensure that the correct elements are returned."""
    self.export_simple_graph(builder_cls)
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    graph = ops.Graph()
    _, ret = loader.load_graph(graph, ["foo_graph"],
                               return_elements=["y:0", "x:0"])

    self.assertEqual(graph.get_tensor_by_name("y:0"), ret[0])
    self.assertEqual(graph.get_tensor_by_name("x:0"), ret[1])

    with self.assertRaisesRegex(ValueError, "not found in graph"):
      loader.load_graph(graph, ["foo_graph"], return_elements=["z:0"])

  def test_parse_saved_model_exception(self, builder_cls):
    # [FUNCTION DOC]
    # Parses a nonexistent directory and asserts an I/O error is raised whose message contains
    # the OS path separator followed by an opening brace (the file name alternatives).
    """Test that error message for not exist model have OS-depend delimiter in path"""
    path = _get_export_dir("not_existing_dir")
    pattern = os.path.sep + "{"
    with self.assertRaises(IOError) as err:
      loader_impl.parse_saved_model(path)
    self.assertTrue(pattern in str(err.exception))


if __name__ == "__main__":
  test.main()
