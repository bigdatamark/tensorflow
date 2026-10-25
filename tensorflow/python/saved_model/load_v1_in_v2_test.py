# [MODULE DOC]
# Tests for loading TF1-style SavedModels (built with the SavedModel builder or simple_save in
# graph mode) through the public load function in eager and graph modes. Covers variable import
# (resource and reference), signatures, assets and lookup tables, control flow, sparse and
# ragged inputs, multiple MetaGraphs, input ordering, trainable flags, partitioned variables
# and re-saving imported models in TF2 format. Single test class; no import-time side effects.
#
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
"""Tests for importing a TF v1-style SavedModel when executing eagerly."""

import os
import shutil

from absl.testing import parameterized

from tensorflow.core.framework import variable_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function as framework_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.training import saver


class LoadTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (with parameterized support) for loading v1-style SavedModels in the TF2 loader.
  # Contains helper methods that build and export small TF1 graphs into temporary directories
  # and test methods that load them and check signatures, variables, assets and re-saving.

  def _v1_single_metagraph_saved_model(self, use_resource):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the variable-import tests.
    # Builds a graph with a placeholder, a distractor reference variable, a variable v (resource
    # if use_resource is true, otherwise reference) and a local non-trainable resource variable;
    # the output is the product of the three. Initializes, then exports with simple_save, using the
    # local variable initializer as legacy init op. Returns the export directory path (under the
    # test temp directory). Side effect: writes files.
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=None, dtype=dtypes.float32, name="start"
      )
      if use_resource:
        distractor = ref_variable.RefVariable(-1.0, name="distractor")
        v = resource_variable_ops.ResourceVariable(3.0, name="v")
      else:
        # "distractor" gets saved in the checkpoint and so used in the restore
        # function, but not in the pruned function for the signature. This tests
        # node naming: it needs to be consistent (and ideally always the same as
        # the node in the original GraphDef) for the resource manager to find
        # the right variable.
        distractor = ref_variable.RefVariable(-1.0, name="distractor")
        v = ref_variable.RefVariable(3.0, name="v")
      local_variable = variable_v1.VariableV1(
          1.0,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          trainable=False,
          use_resource=True,
      )
      output = array_ops.identity(start * v * local_variable, name="output")
      with session_lib.Session() as session:
        session.run(
            [v.initializer, distractor.initializer, local_variable.initializer]
        )
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": start},
            outputs={"output": output},
            legacy_init_op=local_variable.initializer,
        )
    return path

  @test_util.run_in_graph_and_eager_modes
  def test_pretty_printed_signature(self):
    # [FUNCTION DOC]
    # Loads the single-MetaGraph model with resource variables, initializes variables, and takes
    # the serving default signature function. Asserts the short pretty-printed signature equals
    # the expected text (one unknown-shape float input named start, dictionary output), the
    # verbose form matches a pattern including two resource captures, and the repr and str of
    # the function match expected patterns.
    imported = load.load(
        self._v1_single_metagraph_saved_model(use_resource=True)
    )
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(variables.local_variables_initializer())
    concrete_fn = imported.signatures["serving_default"]

    summary = (
        "(start: TensorSpec(shape=<unknown>, dtype=tf.float32,"
        " name='start')) -> Dict[['output', TensorSpec(shape=<unknown>,"
        " dtype=tf.float32, name=None)]]"
    )
    details = (
        r"Input Parameters:\n"
        r"  start \(POSITIONAL_OR_KEYWORD\): TensorSpec\(shape=<unknown>,"
        r" dtype=tf\.float32, name='start'\)\n"
        r"Output Type:\n"
        r"  Dict\[\['output', TensorSpec\(shape=<unknown>,"
        r" dtype=tf\.float32, name=None\)\]\]\n"
        r"Captures:\n"
        r"  \d+: TensorSpec\(shape=\(\), dtype=tf\.resource, name=None\)\n"
        r"  \d+: TensorSpec\(shape=\(\), dtype=tf\.resource, name=None\)"
    )
    self.assertEqual(
        concrete_fn.pretty_printed_signature(verbose=False), summary
    )
    self.assertRegex(
        concrete_fn.pretty_printed_signature(verbose=True), details
    )
    self.assertRegex(repr(concrete_fn), r"<ConcreteFunction .* at .*")
    self.assertRegex(str(concrete_fn), r"ConcreteFunction " + details)

  @test_util.run_in_graph_and_eager_modes
  def test_resource_variable_import(self):
    # [FUNCTION DOC]
    # Loads the model with resource variables and checks the signature output (6 for input 2),
    # imported variable values (3 and 1), that assigning to the imported variables changes outputs
    # (8, then 24), that the first variable is trainable and the second is not, and that a
    # gradient tape watches only the trainable variable with gradient 8 for input 4.
    imported = load.load(
        self._v1_single_metagraph_saved_model(use_resource=True)
    )
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(variables.local_variables_initializer())
    fn = imported.signatures["serving_default"]
    self.assertEqual(
        {"output": 6.0}, self.evaluate(fn(constant_op.constant(2.0)))
    )
    self.assertAllEqual([3.0, 1.0], self.evaluate(imported.variables))
    self.evaluate(imported.variables[0].assign(4.0))
    self.assertEqual(
        {"output": 8.0}, self.evaluate(fn(start=constant_op.constant(2.0)))
    )
    self.evaluate(imported.variables[1].assign(2.0))
    self.assertEqual(
        {"output": 24.0}, self.evaluate(fn(start=constant_op.constant(3.0)))
    )
    self.assertTrue(imported.variables[0].trainable)
    self.assertFalse(imported.variables[1].trainable)
    with backprop.GradientTape() as tape:
      output = fn(start=constant_op.constant(4.0))
    self.assertEqual(imported.variables[:1], list(tape.watched_variables()))
    self.assertEqual(
        8.0, self.evaluate(tape.gradient(output, imported.variables[0]))
    )

  @test_util.run_in_graph_and_eager_modes
  def test_ref_variable_import(self):
    # [FUNCTION DOC]
    # Loads the model built with reference variables, runs the table initializers and the
    # saved-model-initializers collection, then asserts the signature output is 6 for input 2.
    saved = self._v1_single_metagraph_saved_model(use_resource=False)
    imported = load.load(saved)
    fn = imported.signatures["serving_default"]
    self.evaluate(lookup_ops.tables_initializer())
    self.evaluate(ops.get_collection("saved_model_initializers"))
    self.assertEqual(
        6.0, self.evaluate(fn(start=constant_op.constant(2.0))["output"])
    )

  def _v1_output_shape_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the output-shape test.
    # Builds a graph whose output has static shape of one element (set informationally) from a
    # placeholder of unknown length, and saves it with the SavedModel builder under the serving
    # tag with a serving default signature. Returns the export path. Side effect: writes files.
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=[None], dtype=dtypes.float32, name="start"
      )
      output = array_ops.identity(start, name="output")
      output.set_shape([1])  # Ok to use [1] because shape is only informational
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        builder = builder_impl.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                "serving_default": signature_def_utils.build_signature_def(
                    {"start": utils_impl.build_tensor_info(start)},
                    {"output": utils_impl.build_tensor_info(output)},
                )
            },
        )
        builder.save()
    return path

  def test_restore_output_shapes(self):
    # [FUNCTION DOC]
    # Loads the output-shape model and asserts the loaded signature function's first output has
    # the static shape of one element, i.e. output shapes from the signature are restored.
    saved = self._v1_output_shape_saved_model()
    imported = load.load(saved)
    fn = imported.signatures["serving_default"]
    self.assertEqual(tensor_shape.TensorShape([1]), fn.outputs[0].shape)

  def _v1_multi_metagraph_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the multiple-MetaGraph test.
    # Builds a graph with a placeholder and a variable of value 21 and exports two MetaGraphs:
    # one tagged first (signature first_key, output start times the variable) and one tagged
    # second (signature second_key, output the variable). Returns the export path.
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=[None], dtype=dtypes.float32, name="start"
      )
      v = resource_variable_ops.ResourceVariable(21.0)
      first_output = array_ops.identity(start * v, name="first_output")
      second_output = array_ops.identity(v, name="second_output")
      with session_lib.Session() as session:
        session.run(v.initializer)
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        builder = builder_impl.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            session,
            tags=["first"],
            signature_def_map={
                "first_key": signature_def_utils.build_signature_def(
                    {"first_start": utils_impl.build_tensor_info(start)},
                    {
                        "first_output": utils_impl.build_tensor_info(
                            first_output
                        )
                    },
                )
            },
        )
        builder.add_meta_graph(
            tags=["second"],
            signature_def_map={
                "second_key": signature_def_utils.build_signature_def(
                    {"second_start": utils_impl.build_tensor_info(start)},
                    {
                        "second_output": utils_impl.build_tensor_info(
                            second_output
                        )
                    },
                )
            },
        )
        builder.save()
    return path

  def test_multi_meta_graph_loading(self):
    # [FUNCTION DOC]
    # Asserts that loading without tags raises a value error mentioning two MetaGraphs; that
    # loading with the first tag works and its signature returns 42 for input 2; that loading
    # with the second tag (given as a set) works, calling its signature with a wrong keyword
    # argument or an extra argument raises a type error naming second_start, and calling it with
    # the right keyword returns 21.
    with self.assertRaisesRegex(ValueError, "2 MetaGraphs"):
      load.load(self._v1_multi_metagraph_saved_model())
    first_imported = load.load(
        self._v1_multi_metagraph_saved_model(), tags=["first"]
    )
    self.assertEqual(
        {"first_output": 42.0},
        self.evaluate(
            first_imported.signatures["first_key"](
                first_start=constant_op.constant(2.0)
            )
        ),
    )
    second_imported = load.load(
        self._v1_multi_metagraph_saved_model(), tags=set(["second"])
    )
    with self.assertRaisesRegex(TypeError, "second_start"):
      second_imported.signatures["second_key"](x=constant_op.constant(2.0))
    with self.assertRaisesRegex(TypeError, "second_start"):
      second_imported.signatures["second_key"](
          second_start=constant_op.constant(2.0), x=constant_op.constant(2.0)
      )
    self.assertEqual(
        {"second_output": 21.0},
        self.evaluate(
            second_imported.signatures["second_key"](
                second_start=constant_op.constant(2.0)
            )
        ),
    )

  def _v1_asset_saved_model(self, clear_shared_name):
    # [FUNCTION DOC]
    # HELPER: fixture builder for asset and lookup table tests.
    # Writes a vocabulary file with three words, builds a hash table initialized from it with a
    # string-input lookup, optionally clears the shared name attribute of the table op when
    # clear_shared_name (boolean) is true, exports with simple_save using the table initializer as
    # legacy init op, then deletes the original vocabulary file. Returns the export path.
    export_graph = ops.Graph()
    vocab_path = os.path.join(self.get_temp_dir(), "vocab.txt")
    with open(vocab_path, "w") as f:
      f.write("alpha\nbeta\ngamma\n")
    with export_graph.as_default():
      initializer = lookup_ops.TextFileInitializer(
          vocab_path,
          key_dtype=dtypes.string,
          key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
          value_dtype=dtypes.int64,
          value_index=lookup_ops.TextFileIndex.LINE_NUMBER,
      )
      table = lookup_ops.HashTable(initializer, default_value=-1)
      start = array_ops.placeholder(shape=None, dtype=dtypes.string, name="in")
      output = table.lookup(start, name="out")
      if clear_shared_name:
        export_graph.get_operation_by_name("hash_table")._clear_attr(
            "shared_name"
        )
      with session_lib.Session() as session:
        session.run([table.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": start},
            outputs={"output": output},
            legacy_init_op=table.initializer,
        )
    file_io.delete_file(vocab_path)
    return path

  @test_util.run_in_graph_and_eager_modes
  def test_asset_loading(self):
    # [FUNCTION DOC]
    # Loads the asset model and checks lookup of gamma and alpha gives indices 2 and 0. Then
    # re-saves the imported model with TF2 save, deletes the original directory, clears the table
    # initializer collection, reloads and checks the same lookup result, and repeats once more for
    # a third generation, verifying the assets are carried across repeated save and load.
    first_path = self._v1_asset_saved_model(clear_shared_name=False)
    imported = load.load(first_path)
    self.evaluate(lookup_ops.tables_initializer())
    fn = imported.signatures["serving_default"]
    self.assertAllClose(
        {"output": [2, 0]}, fn(start=constant_op.constant(["gamma", "alpha"]))
    )
    second_path = os.path.join(
        self.get_temp_dir(), "saved_model", str(ops.uid())
    )
    save.save(imported, second_path, signatures=imported.signatures)
    shutil.rmtree(first_path)
    del ops.get_collection_ref(ops.GraphKeys.TABLE_INITIALIZERS)[:]
    second_import = load.load(second_path)
    self.evaluate(lookup_ops.tables_initializer())
    fn = second_import.signatures["serving_default"]
    self.assertAllClose(
        {"output": [2, 0]}, fn(start=constant_op.constant(["gamma", "alpha"]))
    )

    third_path = os.path.join(
        self.get_temp_dir(), "saved_model", str(ops.uid())
    )
    save.save(second_import, third_path, signatures=second_import.signatures)
    shutil.rmtree(second_path)
    del ops.get_collection_ref(ops.GraphKeys.TABLE_INITIALIZERS)[:]
    third_import = load.load(third_path)
    self.evaluate(lookup_ops.tables_initializer())
    fn = third_import.signatures["serving_default"]
    self.assertAllClose(
        {"output": [2, 0]}, fn(start=constant_op.constant(["gamma", "alpha"]))
    )

  @test_util.run_in_graph_and_eager_modes
  def test_node_name_sharing(self):
    # [FUNCTION DOC]
    # Loads the asset model whose table op had its shared name attribute cleared and verifies the
    # lookup still works (gamma gives 2, alpha gives 0), i.e. node naming stays consistent.
    fourth_path = self._v1_asset_saved_model(clear_shared_name=True)
    fourth_import = load.load(fourth_path)
    self.evaluate(lookup_ops.tables_initializer())
    fn = fourth_import.signatures["serving_default"]
    self.assertAllClose(
        {"output": [2, 0]}, fn(start=constant_op.constant(["gamma", "alpha"]))
    )

  def _v1_cond_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the conditional test.
    # Builds a graph with a boolean placeholder selecting between constant one and zero via a
    # conditional, and exports it with simple_save. Returns the export path.
    export_graph = ops.Graph()
    with export_graph.as_default():
      branch_selector = array_ops.placeholder(
          name="branch_selector", shape=[], dtype=dtypes.bool
      )
      output = cond.cond(
          branch_selector,
          lambda: array_ops.ones([]),
          lambda: array_ops.zeros([]),
      )
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"branch_selector": branch_selector},
            outputs={"output": output},
        )
    return path

  def test_cond(self):
    # [FUNCTION DOC]
    # Loads the conditional model and asserts the signature returns 1.0 for true and 0.0 for false.
    first_path = self._v1_cond_saved_model()
    imported = load.load(first_path)
    function = imported.signatures["serving_default"]
    self.assertAllClose({"output": 1.0}, function(constant_op.constant(True)))
    self.assertAllClose({"output": 0.0}, function(constant_op.constant(False)))

  def _v1_while_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the while-loop test.
    # Builds a loop that sums indices from zero up to and including an integer placeholder
    # value, and exports it with simple_save. Returns the export path.
    export_graph = ops.Graph()
    with export_graph.as_default():
      loop_iterations = array_ops.placeholder(
          name="loop_iterations", shape=[], dtype=dtypes.int32
      )
      _, output = while_loop.while_loop(
          lambda index, accum: index <= loop_iterations,
          lambda index, accum: (index + 1, accum + index),
          [constant_op.constant(0),
           constant_op.constant(0)],
      )
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"loop_iterations": loop_iterations},
            outputs={"output": output},
        )
    return path

  def test_while(self):
    # [FUNCTION DOC]
    # Loads the while-loop model and asserts the signature returns 10 for input 4 and 15 for 5.
    first_path = self._v1_while_saved_model()
    imported = load.load(first_path)
    function = imported.signatures["serving_default"]
    self.assertAllClose({"output": 10}, function(constant_op.constant(4)))
    self.assertAllClose({"output": 15}, function(constant_op.constant(5)))

  def _v1_nested_while_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the nested while-loop test.
    # Builds an outer loop whose body adds the result of an inner loop (sum of indices up to the
    # current index) to an accumulator, with an integer placeholder as bound, and exports it with
    # simple_save. Returns the export path.
    export_graph = ops.Graph()
    with export_graph.as_default():

      def _inner_while(loop_iterations):
        # [FUNCTION DOC]
        # HELPER: builds the inner loop for the nested while fixture.
        # Takes an integer tensor bound and returns the sum of indices from zero up to and
        # including it, computed with a while loop in the current graph.
        _, output = while_loop.while_loop(
            lambda index, accum: index <= loop_iterations,
            lambda index, accum: (index + 1, accum + index),
            [constant_op.constant(0),
             constant_op.constant(0)],
        )
        return output

      loop_iterations = array_ops.placeholder(
          name="loop_iterations", shape=[], dtype=dtypes.int32
      )
      _, output = while_loop.while_loop(
          lambda index, accum: index <= loop_iterations,
          lambda index, accum: (index + 1, accum + _inner_while(index)),
          [constant_op.constant(0),
           constant_op.constant(0)],
      )
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"loop_iterations": loop_iterations},
            outputs={"output": output},
        )
    return path

  def test_nested_while(self):
    # [FUNCTION DOC]
    # Loads the nested while-loop model and asserts the signature returns 20 for input 4 and
    # 35 for input 5.
    first_path = self._v1_nested_while_saved_model()
    imported = load.load(first_path)
    function = imported.signatures["serving_default"]
    self.assertAllClose({"output": 20}, function(constant_op.constant(4)))
    self.assertAllClose({"output": 35}, function(constant_op.constant(5)))

  def _no_signatures_model(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for tests needing a model without signatures.
    # Builds a graph computing input plus one (placeholder x, identity op out) and saves it with
    # the SavedModel builder under the serving tag with an empty signature map. Returns the path.
    export_graph = ops.Graph()
    with export_graph.as_default():
      inp = array_ops.placeholder(name="x", shape=[], dtype=dtypes.float32)
      array_ops.identity(inp + 1.0, name="out")
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={},
            assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
        )
        b.save()
    return path

  def test_no_signature(self):
    # [FUNCTION DOC]
    # Loads the model with no signatures and asserts the imported signatures mapping is empty.
    path = self._no_signatures_model()
    imported = load.load(path)
    self.assertEqual([], list(imported.signatures.keys()))

  def _signature_with_no_inputs(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for tests with an input-less signature.
    # Builds a graph with an unused placeholder and a random normal output of two elements, and
    # saves it with a signature named key that has no inputs and one output named value.
    # Returns the export path.
    export_graph = ops.Graph()
    with export_graph.as_default():
      array_ops.placeholder(name="x", shape=[], dtype=dtypes.float32)
      output = random_ops.random_normal([2])
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                "key": signature_def_utils.build_signature_def(
                    {}, dict(value=utils_impl.build_tensor_info(output))
                )
            },
        )
        b.save()
    return path

  def test_signature_with_no_inputs(self):
    # [FUNCTION DOC]
    # Loads the input-less signature model, calls the key signature with no arguments, and
    # asserts the value output has shape of two elements.
    path = self._signature_with_no_inputs()
    imported = load.load(path)
    self.assertEqual([2], imported.signatures["key"]()["value"].shape)

  def test_version_info(self):
    # [FUNCTION DOC]
    # Loads a model and asserts the imported object's TensorFlow version and git version strings
    # equal the current library version and git version.
    path = self._signature_with_no_inputs()
    imported = load.load(path)
    self.assertEqual(versions.__version__, imported.tensorflow_version)
    self.assertEqual(versions.__git_version__, imported.tensorflow_git_version)

  def _unfed_placeholder_signature(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the unfed-placeholder test.
    # Builds a graph whose output (placeholder x times random values) depends on a placeholder,
    # but exports a signature with no inputs, so the placeholder is never fed. Returns the path.
    export_graph = ops.Graph()
    with export_graph.as_default():
      x = array_ops.placeholder(name="x", shape=[], dtype=dtypes.float32)
      output = x * random_ops.random_normal([2])
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                "key": signature_def_utils.build_signature_def(
                    {}, dict(value=utils_impl.build_tensor_info(output))
                )
            },
        )
        b.save()
    return path

  def test_unfed_placeholder_exception(self):
    # [FUNCTION DOC]
    # Asserts that loading the unfed-placeholder model raises the unliftable error whose message
    # says a signature needs an input for each placeholder, followed by the original lift message.
    path = self._unfed_placeholder_signature()
    with self.assertRaisesRegex(
        lift_to_graph.UnliftableError,
        "signature needs an input for each placeholder.*\n\nUnable to lift",
    ):
      load.load(path)

  def test_custom_pruning(self):
    # [FUNCTION DOC]
    # Loads the no-signature model, prunes the root's function with feed x and fetch out, and
    # asserts calling it with ones gives 2.0; also checks the graph exposes the x tensor.
    path = self._no_signatures_model()
    root = load.load(path)
    fn = root.prune("x:0", "out:0")
    self.assertEqual(2.0, self.evaluate(fn(x=array_ops.ones([]))))
    root.graph.as_graph_element("x:0")

  def _no_trainable_variable_attribute(self, trainable):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the trainable-fallback test.
    # Defines a resource variable subclass whose proto lacks the trainable field, creates one
    # with the given trainable flag (boolean), saves it with the SavedModel builder with no
    # signatures, and returns the export path.
    """A SavedModel where the VariableDef has no 'trainable' (it's false)."""

    class _MissingFieldsVariable(resource_variable_ops.ResourceVariable):
      # [CLASS DOC]
      # HELPER: test double producing a variable definition without a trainable field.
      # Resource variable subclass that overrides proto conversion to drop some fields.

      def to_proto(self, export_scope=None):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the initializer name of the reduced proto is filled from the full proto's
        # snapshot name, not its initializer name, which looks like a mix-up (it is a test
        # double, so it may be deliberate).
        # Converts to a variable definition proto, keeping only the variable name, initial value
        # name, initializer name, slice info and resource flag, so trainable is left unset.
        # Takes an optional export scope string and returns the reduced proto.
        full_proto = super(_MissingFieldsVariable, self).to_proto(export_scope)
        return variable_pb2.VariableDef(
            variable_name=full_proto.variable_name,
            initial_value_name=full_proto.initial_value_name,
            initializer_name=full_proto.snapshot_name,
            save_slice_info_def=full_proto.save_slice_info_def,
            is_resource=full_proto.is_resource,
        )

    export_graph = ops.Graph()
    with export_graph.as_default():
      v = _MissingFieldsVariable(3.0, trainable=trainable)
      with session_lib.Session() as session:
        session.run([v.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session, tags=[tag_constants.SERVING], signature_def_map={}
        )
        b.save()

    return path

  def test_trainable_not_set_in_proto(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the library version is overridden globally and only restored at the end of the
    # test, not in a cleanup or finally step, so a failing assertion leaves the fake version set.
    # Fakes an old library version (1.7.0) so the loader falls back to collections for the
    # trainable flag. Saves variables lacking the field with trainable true and false and asserts
    # the loaded variable follows the collections (true then false). Restores the real version.
    """If a VariableDef has no 'trainable', we fall back to collections."""
    real_tf_version = versions.__version__
    # Pretend to be exported from an older version of TensorFlow, so trainable
    # will follow collections instead of checking VariableDefs.
    versions.__version__ = "1.7.0"
    path = self._no_trainable_variable_attribute(trainable=True)
    root = load.load(path)
    self.assertTrue(root.variables[0].trainable)
    path = self._no_trainable_variable_attribute(trainable=False)
    root = load.load(path)
    self.assertFalse(root.variables[0].trainable)
    versions.__version__ = real_tf_version

  def _export_variable(self, **kwargs_for_variable):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the trainable-in-proto test.
    # Creates a resource variable of value 3 passing the given keyword arguments to its
    # constructor, saves a SavedModel with no signatures, and returns the export path.
    """A 1.x SavedModel with a single variable."""
    export_graph = ops.Graph()
    with export_graph.as_default():
      v = resource_variable_ops.ResourceVariable(3.0, **kwargs_for_variable)
      with session_lib.Session() as session:
        session.run([v.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session, tags=[tag_constants.SERVING], signature_def_map={}
        )
        b.save()

    return path

  def test_trainable_in_proto(self):
    # [FUNCTION DOC]
    # Exports variables with the trainable flag set in the proto, one true and one false (the
    # latter also put into the trainable collection), loads each and asserts the loaded
    # variable's trainable flag follows the proto rather than the collections.
    """If a VariableDef has a trainable property, we do not use collections."""
    path = self._export_variable(
        trainable=True, collections=[ops.GraphKeys.GLOBAL_VARIABLES]
    )
    root = load.load(path)
    self.assertTrue(root.variables[0].trainable)
    path = self._export_variable(
        trainable=False,
        collections=[
            ops.GraphKeys.GLOBAL_VARIABLES,
            ops.GraphKeys.TRAINABLE_VARIABLES,
        ],
    )
    root = load.load(path)
    self.assertFalse(root.variables[0].trainable)

  def _model_with_sparse_output(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the sparse output test.
    # Builds a graph whose output is a sparse tensor (one element, values from an integer
    # placeholder) multiplied by two and exports it in v1 format. Returns the export path.
    """Generate a graph with a SparseTensor output and serialize in V1 format"""
    export_graph = ops.Graph()
    with export_graph.as_default():
      in_placeholder = array_ops.placeholder(dtype=dtypes.int64, shape=[1])
      out_sparse_tensor = (
          sparse_tensor.SparseTensor(
              indices=[[0]], values=in_placeholder, dense_shape=[1]
          )
          * 2
      )
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": in_placeholder},
            outputs={"output": out_sparse_tensor},
        )
    return path

  def test_load_sparse_outputs(self):
    # [FUNCTION DOC]
    # Loads the sparse output model, calls the signature with 42 and asserts the sparse output's
    # values equal 84.
    path = self._model_with_sparse_output()
    imported = load.load(path)
    imported_fn = imported.signatures["serving_default"]
    forty_two = constant_op.constant([42], dtype=dtypes.int64)
    self.assertEqual([84], imported_fn(forty_two)["output"].values.numpy())

  def _model_with_sparse_input(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the sparse input test.
    # Builds a graph taking a two-by-two sparse placeholder and returning a sparse tensor
    # rebuilt from its parts and multiplied by two; exports in v1 format. Returns the path.
    """Generate a graph with a SparseTensor input and serialize in V1 format."""
    export_graph = ops.Graph()
    with export_graph.as_default():
      in_sparse_placeholder = array_ops.sparse_placeholder(
          dtype=dtypes.int64, shape=[2, 2]
      )
      out_sparse_tensor = (
          sparse_tensor.SparseTensor(
              indices=in_sparse_placeholder.indices,
              values=in_sparse_placeholder.values,
              dense_shape=in_sparse_placeholder.dense_shape,
          )
          * 2
      )
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": in_sparse_placeholder},
            outputs={"output": out_sparse_tensor},
        )
    return path

  def test_load_sparse_inputs(self):
    # [FUNCTION DOC]
    # Loads the sparse input model and calls the signature with the three expanded arguments
    # (start indices, values and dense shape); asserts the output values are 84, 86 and 88.
    path = self._model_with_sparse_input()
    imported = load.load(path)
    imported_fn = imported.signatures["serving_default"]
    indices = constant_op.constant([[0, 0], [0, 1], [1, 1]], dtype=dtypes.int64)
    values = constant_op.constant([42, 43, 44], dtype=dtypes.int64)
    dense_shape = constant_op.constant([2, 2], dtype=dtypes.int64)
    result = imported_fn(
        start_indices=indices,
        start_values=values,
        start_dense_shape=dense_shape,
    )
    self.assertAllEqual([84, 86, 88], result["output"].values.numpy())

  def _model_with_ragged_input(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the ragged input test.
    # Builds a graph with a ragged float placeholder multiplied by two, exports in v1 format
    # with inputs x and outputs y. Returns the export path.
    """Generate a graph with a RaggedTensor input and serialize in V1 format."""
    export_graph = ops.Graph()
    with export_graph.as_default():
      x = ragged_factory_ops.placeholder(dtypes.float32, 1, [])
      y = x * 2
      with session_lib.Session() as sess:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(sess, path, inputs={"x": x}, outputs={"y": y})
    return path

  def test_load_ragged_inputs(self):
    # [FUNCTION DOC]
    # Loads the ragged input model and calls the signature with the two component arguments
    # (values and row splits) of a ragged tensor; asserts the result is the doubled ragged values.
    path = self._model_with_ragged_input()
    imported = load.load(path)
    imported_fn = imported.signatures["serving_default"]
    x = ragged_factory_ops.constant([[10.0, 20.0], [30.0]])
    result = imported_fn(x_component_0=x.values, x_component_1=x.row_splits)
    self.assertAllEqual(result["y"], [[20.0, 40.0], [60.0]])

  def _model_with_defun(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the legacy defined-function test.
    # Builds nested legacy functions (f calls g calls z, each adding one) applied to an integer
    # placeholder and exports in v1 format. Returns the export path.
    """Generate a graph with a Defun and serialize in V1 format."""
    export_graph = ops.Graph()
    with export_graph.as_default():

      @framework_function.Defun(dtypes.int64)
      def z(x):
        # [FUNCTION DOC]
        # HELPER: innermost legacy function of the fixture; returns its input plus one.
        return x + 1

      @framework_function.Defun(dtypes.int64)
      def g(x):
        # [FUNCTION DOC]
        # HELPER: middle legacy function of the fixture; returns z of the input plus one.
        return z(x) + 1

      @framework_function.Defun(dtypes.int64)
      def f(x):
        # [FUNCTION DOC]
        # HELPER: outer legacy function of the fixture; returns g of the input plus one.
        return g(x) + 1

      in_placeholder = array_ops.placeholder(dtype=dtypes.int64, shape=[1])
      out = f(in_placeholder)
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": in_placeholder},
            outputs={"output": out},
        )
    return path

  def test_load_defun(self):
    # [FUNCTION DOC]
    # Loads the legacy defined-function model and asserts the signature maps 42 to 45.
    path = self._model_with_defun()
    imported = load.load(path)
    imported_fn = imported.signatures["serving_default"]
    forty_two = constant_op.constant([42], dtype=dtypes.int64)
    self.assertEqual([45], imported_fn(forty_two)["output"].numpy())

  def test_load_and_restore_partitioned_variables(self):
    # [FUNCTION DOC]
    # Exports a model with a two-way partitioned resource variable of value 13 (output is input
    # times the variable), and separately writes a name-based checkpoint with other values. Loads
    # the model and asserts the variables hold 13 in both partitions, then restores from the
    # checkpoint via the root's restore function and asserts the new values and that the
    # signature output for input 2 is the doubled new values.
    export_graph = ops.Graph()
    with export_graph.as_default():
      partitioned_var = variable_scope.get_variable(
          "a",
          shape=[6],
          initializer=init_ops.constant_initializer(13),
          partitioner=partitioned_variables.fixed_size_partitioner(2),
          use_resource=True,
      )
      x = array_ops.placeholder(shape=[], dtype=dtypes.float32)
      y = x * partitioned_var
      with session_lib.Session() as session:
        session.run(variables.global_variables_initializer())
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session, path, inputs={"x": x}, outputs={"y": y}
        )

        # Create a name-based checkpoint with different values.
        session.run(partitioned_var.assign([[5, 4, 3], [2, 1, 0]]))
        ckpt_path = os.path.join(self.get_temp_dir(), "restore_ckpt")
        saver.Saver().save(session, ckpt_path)

    imported = load.load(path)
    self.assertAllClose(
        self.evaluate(imported.variables), [[13, 13, 13], [13, 13, 13]]
    )

    self.evaluate(imported.restore(ckpt_path))
    self.assertAllClose(
        self.evaluate(imported.variables), [[5, 4, 3], [2, 1, 0]]
    )
    self.assertAllClose(
        self.evaluate(
            imported.signatures["serving_default"](constant_op.constant(2.0))
        ),
        {"y": [10, 8, 6, 4, 2, 0]},
    )

  def test_structured_input_signature(self):
    # [FUNCTION DOC]
    # Loads the single-MetaGraph model with reference variables and asserts the signature's
    # structured input signature has no positional arguments and one keyword argument named start
    # with an unknown-shape tensor spec.
    path = self._v1_single_metagraph_saved_model(False)
    imported = load.load(path)
    args, kwargs = imported.signatures[
        "serving_default"
    ].structured_input_signature
    self.assertEqual(args, ())
    self.assertAllEqual(
        kwargs, {"start": tensor_spec.TensorSpec(shape=None, name="start")}
    )

  def _model_with_multiple_inputs(self, input_names, compute_fn, var_value):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the multiple-inputs test.
    # Creates one scalar float placeholder per name in input_names (sequence of strings), a
    # variable initialized to var_value, and an output computed by compute_fn (callable taking the
    # tuple of placeholders and the variable). Saves a signature mapping each name to its
    # placeholder. Returns the export path.
    export_graph = ops.Graph()
    with export_graph.as_default():
      inputs = tuple(
          array_ops.placeholder(shape=(), dtype=dtypes.float32, name=name)
          for name in input_names
      )
      v = resource_variable_ops.ResourceVariable(var_value)
      output = array_ops.identity(compute_fn(inputs, v), name="output")
      with session_lib.Session() as session:
        session.run(v.initializer)
        path = os.path.join(
            self.get_temp_dir(), "tf1_saved_model", str(ops.uid())
        )
        builder = builder_impl.SavedModelBuilder(path)
        feeds = {
            name: utils_impl.build_tensor_info(input)
            for name, input in zip(input_names, inputs)
        }
        builder.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                "serving_default": signature_def_utils.build_signature_def(
                    feeds,
                    {"output": utils_impl.build_tensor_info(output)},
                )
            },
        )
        builder.save()
    return path

  @parameterized.named_parameters(
      (f"_{input_names_idx}_{reverse}", input_names, reverse)  # pylint: disable=g-complex-comprehension
      for reverse in (False, True)
      for input_names_idx, input_names in enumerate((
          ("input1", "input2"),
          ("input1", "input./-"),
      ))
  )
  def test_multiple_inputs(self, input_names, reverse):
    # [FUNCTION DOC]
    # Parameterized over two sets of input names (one containing unusual characters) and whether
    # their order is reversed. Exports a model with those inputs, loads it, and asserts the
    # signature function inputs are ordered by sorted name, calling it by keyword or by position
    # (sorted order) gives the expected output, then re-saves in TF2 format, reloads and checks
    # the keyword call again (positional calls are no longer supported after TF2 saving).
    if reverse:
      input_names = tuple(reversed(input_names))

    def compute_fn(ls, a):
      # [FUNCTION DOC]
      # HELPER: output formula shared by export and expectation. Starting from a, repeatedly adds
      # each element of the sequence ls and multiplies by a; returns the final value.
      result = a
      for x in ls:
        result = (result + x) * a
      return result

    var_value = 21.0
    path = self._model_with_multiple_inputs(
        input_names, compute_fn=compute_fn, var_value=21.0
    )
    imported = load.load(path)
    sorted_with_idx = sorted(
        zip(range(len(input_names)), input_names), key=lambda x: x[1]
    )
    fn = imported.signatures["serving_default"]
    for i, (_, input_name) in enumerate(sorted_with_idx):
      self.assertEqual(fn.inputs[i].name, f"{input_name}:0")
    inputs = tuple(i + 2.0 for i in range(len(input_names)))
    expected_output = compute_fn(inputs, var_value)
    # Call `fn`` with keyword arguments
    self.assertEqual(
        self.evaluate(
            fn(**{
                name: constant_op.constant(v)
                for name, v in zip(input_names, inputs)
            })["output"]
        ),
        expected_output,
    )
    # Call `fn`` with positional arguments
    self.assertEqual(
        self.evaluate(fn(*(inputs[i] for i, _ in sorted_with_idx))["output"]),
        expected_output,
    )

    # Test saving the model again in TF2
    path2 = os.path.join(self.get_temp_dir(), "tf2_saved_model", str(ops.uid()))
    save.save(imported, path2, imported.signatures)

    imported2 = load.load(path2)
    fn = imported2.signatures["serving_default"]
    # Call `fn`` with keyword arguments
    self.assertEqual(
        self.evaluate(
            fn(**{
                name: constant_op.constant(v)
                for name, v in zip(input_names, inputs)
            })["output"]
        ),
        expected_output,
    )
    # `fn` can no longer be called with positional arguments, because
    # during TF2 saving, those positional-keyword-hybrid arguments are
    # converted to keyword-only arguments.

  def _v1_multi_input_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the input-ordering test.
    # Builds a graph with two float placeholders input1 and input2 and a variable of 21; output
    # is input1 times the variable plus input2. Saves a serving default signature. Returns the path.
    export_graph = ops.Graph()
    with export_graph.as_default():
      input1 = array_ops.placeholder(
          shape=[None], dtype=dtypes.float32, name="input1"
      )
      input2 = array_ops.placeholder(
          shape=[None], dtype=dtypes.float32, name="input2"
      )
      v = resource_variable_ops.ResourceVariable(21.0)
      output = array_ops.identity(input1 * v + input2, name="output")
      with session_lib.Session() as session:
        session.run(v.initializer)
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        builder = builder_impl.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                "serving_default": signature_def_utils.build_signature_def(
                    {
                        "input1": utils_impl.build_tensor_info(input1),
                        "input2": utils_impl.build_tensor_info(input2),
                    },
                    {"output": utils_impl.build_tensor_info(output)},
                )
            },
        )
        builder.save()
    return path

  def test_v1_input_ordered(self):
    # [FUNCTION DOC]
    # Loads the two-input model and asserts the signature function's inputs are ordered input1
    # first, then input2.
    path = self._v1_multi_input_saved_model()
    imported = load.load(path)
    self.assertEqual(
        imported.signatures["serving_default"].inputs[0].name, "input1:0"
    )
    self.assertEqual(
        imported.signatures["serving_default"].inputs[1].name, "input2:0"
    )

  def test_resave_signature(self):
    # [FUNCTION DOC]
    # Regression test (referenced bug in the comment below): exports a v1 model whose inputs
    # a and b have names input_2 and input_1 in non-matching order, loads it, re-saves it with TF2
    # save using its signatures, reloads, and asserts calling the signature by keyword a and b
    # yields the expected output 5.0 (the value of a).
    # Tests that signatures saved using TF1 can be resaved with TF2.
    # See b/211666001 for context.
    export_graph = ops.Graph()
    with export_graph.as_default():
      a = array_ops.placeholder(
          shape=[None, 1], dtype=dtypes.float32, name="input_2"
      )
      b = array_ops.placeholder(
          shape=[None, 2], dtype=dtypes.float32, name="input_1"
      )
      c = array_ops.identity(a)
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session, path, inputs={"a": a, "b": b}, outputs={"c": c}
        )
    imported = load.load(path)
    path2 = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
    save.save(imported, path2, imported.signatures)

    imported2 = load.load(path2)
    self.assertEqual(
        imported2.signatures["serving_default"](
            a=constant_op.constant([5.0]), b=constant_op.constant([1.0, 3.0])
        )["c"].numpy(),
        5.0,
    )


if __name__ == "__main__":
  test.main()
