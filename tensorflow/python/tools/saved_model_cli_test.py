# [MODULE DOC]
# Unit tests for the saved_model_cli command line tool (show, run, scan, aot_compile_cpu and
# freeze_model commands and its argument preprocessing helpers). Tests set the parsed flags
# directly, create the argument parser, call the command function and capture standard output
# and error with a context manager to compare printed text. Models come from a checked-in
# half_plus_two test SavedModel or from small models saved into the test temporary directory.
# Expensive compile tests are skipped unless XLA is built in; the whole class is skipped on
# Windows. Importing defines a module constant with the test model path.
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
"""Tests for SavedModelCLI tool."""
import contextlib
import io
import os
import pickle
import platform
import shutil
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.tools import saved_model_cli
from tensorflow.python.trackable import autotrackable

SAVED_MODEL_PATH = ('cc/saved_model/testdata/half_plus_two/00000123')


@contextlib.contextmanager
def captured_output():
  # [FUNCTION DOC]
  # HELPER: Shared test utility (a context manager) for checking what the tool prints.
  # While active, replaces the process standard output and error streams with in-memory text
  # buffers and yields the pair (output buffer, error buffer). The original streams are always
  # restored on exit, even on exceptions. Takes no arguments.
  new_out, new_err = io.StringIO(), io.StringIO()
  old_out, old_err = sys.stdout, sys.stderr
  try:
    sys.stdout, sys.stderr = new_out, new_err
    yield sys.stdout, sys.stderr
  finally:
    sys.stdout, sys.stderr = old_out, old_err


class SavedModelCLITestCase(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the saved_model_cli tool. Combines the TensorFlow test base class with
  # parameterized testing. Skipped entirely on Windows. Holds a helper to save a small dummy
  # model and a nested model class used by the XLA ahead-of-time compilation tests.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test setup: calls the parent setup, then skips the test when running on Windows.
    super(SavedModelCLITestCase, self).setUp()
    if platform.system() == 'Windows':
      self.skipTest('Skipping failing tests on Windows.')

  def _save_dummy_model(self, get_ops_mock):
    # [FUNCTION DOC]
    # HELPER: Fixture builder shared by several show tests.
    # Takes get_ops_mock, the patched op listing function of the tool, and sets its return value
    # to a fixed set with a single op so output is deterministic. Builds a small trackable model
    # holding a concrete function that multiplies two scalar floats, saves it to a dummy_model
    # directory in the test temporary directory, and returns that directory path.
    class DummyModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: Throwaway model type local to the dummy model fixture.
      # Trackable model whose constructor stores a concrete function (two scalar float inputs)
      # wrapping the multiply method as an attribute so it can be saved.
      """Model with a callable concrete function."""

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor: wraps the multiply method as a function with a fixed signature of two
        # scalar floats and stores its concrete function in an attribute for saving.
        function = def_function.function(
            self.multiply,
            input_signature=[
                tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
                tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32)
            ])
        self.pure_concrete_function = function.get_concrete_function()

      def multiply(self, a, b):
        # [FUNCTION DOC]
        # Returns the product of the two inputs a and b. Wrapped by the constructor into the
        # saved concrete function.
        return a * b

    # Mocking _get_ops_in_metagraph because it returns a nondeterministically
    # ordered set of ops.
    get_ops_mock.return_value = {'Op1'}
    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = DummyModel()
    with self.cached_session():
      save.save(dummy_model, saved_model_dir)
    return saved_model_dir

  @test.mock.patch.object(saved_model_cli, '_get_ops_in_metagraph')
  def testShowCommandAll(self, get_ops_mock):
    # [FUNCTION DOC]
    # Runs the show command with the all option on the half_plus_two test model while the op
    # listing is patched to return three fixed ops. Asserts the printed text contains the full
    # expected description of every signature def plus the op header, that the op listing function
    # was called exactly once, that all three op names appear, and that nothing went to error output.
    # Mocking _get_ops_in_metagraph because it returns a nondeterministically
    # ordered set of ops.
    get_ops_mock.return_value = {'Op1', 'Op2', 'Op3'}
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    # pylint: disable=line-too-long
    exp_out = """MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classify_x2_to_y3']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y3:0
  Method name is: tensorflow/serving/classify

signature_def['classify_x_to_y']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: tf_example:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/classify

signature_def['regress_x2_to_y3']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y3:0
  Method name is: tensorflow/serving/regress

signature_def['regress_x_to_y']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: tf_example:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/regress

signature_def['regress_x_to_y2']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: tf_example:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y2:0
  Method name is: tensorflow/serving/regress

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/predict
The MetaGraph with tag set ['serve'] contains the following ops:"""
    # pylint: enable=line-too-long

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', base_path, '--all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    get_ops_mock.assert_called_once()
    output = out.getvalue().strip()
    self.maxDiff = None  # Produce useful error msg if the comparison fails
    self.assertIn(exp_out, output)
    self.assertIn('Op1', output)
    self.assertIn('Op2', output)
    self.assertIn('Op3', output)
    self.assertEqual(err.getvalue().strip(), '')

  @test.mock.patch.object(saved_model_cli, '_get_ops_in_metagraph')
  def testShowAllWithFunctions(self, get_ops_mock):
    # [FUNCTION DOC]
    # Defines a local trackable model with three traced functions (one with an unconstrained
    # signature called with concrete values, one with a fixed signature, and a call method with a
    # default argument), saves it, and runs show with the all option while ops are patched to one
    # fixed op. Asserts the printed output exactly equals the expected text listing the signature
    # defs and a Concrete Functions section with each function's arguments, and no error output.

    class DummyModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # Local trackable model with three traced functions: func1 (branches on a flag),
      # func2 (fixed two by two float signature, adds two) and a call method with a default
      # argument. Used to check how show lists concrete functions of a saved model.
      """Model with callable polymorphic functions specified."""

      @def_function.function
      def func1(self, a, b, c):
        # [FUNCTION DOC]
        # Traced function: returns the sum of a and b when the flag c is true, otherwise their
        # product.
        if c:
          return a + b
        else:
          return a * b

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=(2, 2), dtype=dtypes.float32)
      ])
      def func2(self, x):
        # [FUNCTION DOC]
        # Traced function with a fixed two by two float signature; returns x plus two.
        return x + 2

      @def_function.function
      def __call__(self, y, c=7):
        # [FUNCTION DOC]
        # Traced call method: returns y plus twice c, where c defaults to seven.
        return y + 2 * c

    # Mocking _get_ops_in_metagraph because it returns a nondeterministically
    # ordered set of ops.
    get_ops_mock.return_value = {'Op1'}
    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = DummyModel()
    # Call with specific values to create new polymorphic function traces.
    dummy_model.func1(constant_op.constant(5), constant_op.constant(9), True)
    dummy_model(constant_op.constant(5))
    with self.cached_session():
      save.save(dummy_model, saved_model_dir)

    exp_out = """MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (2, 2)
        name: serving_default_x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: (2, 2)
        name: PartitionedCall:0
  Method name is: tensorflow/serving/predict
The MetaGraph with tag set ['serve'] contains the following ops: {'Op1'}

Concrete Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          y: TensorSpec(shape=(), dtype=tf.int32, name='y')
        Argument #2
          DType: int
          Value: 7

  Function Name: 'func1'
    Option #1
      Callable with:
        Argument #1
          a: TensorSpec(shape=(), dtype=tf.int32, name='a')
        Argument #2
          b: TensorSpec(shape=(), dtype=tf.int32, name='b')
        Argument #3
          DType: bool
          Value: True

  Function Name: 'func2'
    Option #1
      Callable with:
        Argument #1
          x: TensorSpec(shape=(2, 2), dtype=tf.float32, name='x')
""".strip()  # pylint: enable=line-too-long

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', saved_model_dir, '--all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  @test.mock.patch.object(saved_model_cli, '_get_ops_in_metagraph')
  def testShowAllWithPureConcreteFunction(self, get_ops_mock):
    # [FUNCTION DOC]
    # Saves the dummy multiply model through the shared fixture and runs show with the all
    # option. Asserts the output exactly equals the expected text, including a Concrete
    # Functions entry for the stored pure concrete function with its two float arguments, and
    # that nothing is written to error output.
    saved_model_dir = self._save_dummy_model(get_ops_mock)

    exp_out = """MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['a'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_a:0
    inputs['b'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_b:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: PartitionedCall:0
  Method name is: tensorflow/serving/predict
The MetaGraph with tag set ['serve'] contains the following ops: {'Op1'}

Concrete Functions:
  Function Name: 'pure_concrete_function'
    Option #1
      Callable with:
        Argument #1
          a: TensorSpec(shape=(), dtype=tf.float32, name='a')
        Argument #2
          b: TensorSpec(shape=(), dtype=tf.float32, name='b')
""".strip()  # pylint: enable=line-too-long

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', saved_model_dir, '--all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandTags(self):
    # [FUNCTION DOC]
    # Runs show with only a directory on the test model. Asserts the output is exactly the
    # message listing the available tag-sets (just serve) and that error output is empty.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    exp_out = 'The given SavedModel contains the following tag-sets:\n\'serve\''

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(['saved_model_cli', 'show', '--dir', base_path])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  @test.mock.patch.object(saved_model_cli, '_get_ops_in_metagraph')
  def testShowAllWithCustomOp(self, get_ops_mock):
    # [FUNCTION DOC]
    # Saves the dummy model, then edits the saved proto on disk to add a node with an unknown
    # custom op to a library function and rewrites the file. Runs show with the all option and
    # asserts the output says the ops could not be listed due to custom ops, with no error output.
    saved_model_dir = self._save_dummy_model(get_ops_mock)
    # Manually edit in a custom op into one of the functions
    saved_model_proto = loader_impl.parse_saved_model(saved_model_dir)
    saved_model_proto.meta_graphs[0].graph_def.library.function[0].node_def.add(
        name='TestCustomOp', op='CustomOp')
    file_io.atomic_write_string_to_file(
        os.path.join(saved_model_dir, 'saved_model.pb'),
        saved_model_proto.SerializeToString(deterministic=True))
    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', saved_model_dir, '--all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.assertIn('could not be listed due to the existence of custom ops',
                  output)
    self.assertEqual(err.getvalue().strip(), '')

  @test.mock.patch.object(saved_model_cli, '_get_ops_in_metagraph')
  def testShowAllWithInvalidFuncGraph(self, get_ops_mock):
    # [FUNCTION DOC]
    # Saves the dummy model, then edits the saved proto to add a variable handle node to a
    # library function so that loading the model fails. Runs show with the all option and
    # asserts the output says the functions could not be listed due to unknown reasons, with no
    # error output.
    saved_model_dir = self._save_dummy_model(get_ops_mock)
    # Manually edit one of the functions to make `tf.saved_model.load` fail.
    saved_model_proto = loader_impl.parse_saved_model(saved_model_dir)
    saved_model_proto.meta_graphs[0].graph_def.library.function[0].node_def.add(
        name='TestInvalidOp', op='VarHandleOp')
    file_io.atomic_write_string_to_file(
        os.path.join(saved_model_dir, 'saved_model.pb'),
        saved_model_proto.SerializeToString(deterministic=True))
    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', saved_model_dir, '--all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.assertIn('could not be listed due to unknown reasons', output)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandSignature(self):
    # [FUNCTION DOC]
    # Runs show with a directory and the serve tag-set. Asserts the output is a header followed
    # by one line for each of the six signature def keys of the test model, in the given order,
    # and that error output is empty.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    exp_header = ('The given SavedModel MetaGraphDef contains SignatureDefs '
                  'with the following keys:')
    exp_start = 'SignatureDef key: '
    exp_keys = [
        '"classify_x2_to_y3"', '"classify_x_to_y"', '"regress_x2_to_y3"',
        '"regress_x_to_y"', '"regress_x_to_y2"', '"serving_default"'
    ]

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', base_path, '--tag_set', 'serve'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    # Order of signatures does not matter
    self.assertMultiLineEqual(
        output,
        '\n'.join([exp_header] +
                  [exp_start + exp_key for exp_key in exp_keys]))
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandErrorNoTagSet(self):
    # [FUNCTION DOC]
    # Runs show with a tag-set that does not exist in the model and asserts a runtime error is raised.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'show', '--dir', base_path,
        '--tag_set', 'badtagset'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaises(RuntimeError):
      saved_model_cli.show()

  def testShowCommandInputsOutputs(self):
    # [FUNCTION DOC]
    # Runs show with tag-set serve and the serving default signature. Asserts the output is exactly
    # the description of the one float input x, the one float output y (shape unknown by one)
    # and the predict method name, and that error output is empty.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    expected_output = (
        'The given SavedModel SignatureDef contains the following input(s):\n'
        '  inputs[\'x\'] tensor_info:\n'
        '      dtype: DT_FLOAT\n      shape: (-1, 1)\n      name: x:0\n'
        'The given SavedModel SignatureDef contains the following output(s):\n'
        '  outputs[\'y\'] tensor_info:\n'
        '      dtype: DT_FLOAT\n      shape: (-1, 1)\n      name: y:0\n'
        'Method name is: tensorflow/serving/predict')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'show', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default'
    ])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.assertEqual(output, expected_output)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandListOps(self):
    # [FUNCTION DOC]
    # Runs show with tag-set serve and the list ops option on the real test model (no patching).
    # Asserts the ops header is printed and that a fixed set of sixteen expected op names (for
    # example variable, add, restore, parse example, no-op) appear, with no error output.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'show', '--dir', base_path, '--tag_set', 'serve',
        '--list_ops'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.assertIn(
        'The MetaGraph with tag set [\'serve\'] contains the following ops:',
        output)
    self.assertIn('\'VariableV2\'', output)
    self.assertIn('\'Add\'', output)
    self.assertIn('\'RestoreV2\'', output)
    self.assertIn('\'ShardedFilename\'', output)
    self.assertIn('\'Placeholder\'', output)
    self.assertIn('\'Mul\'', output)
    self.assertIn('\'Pack\'', output)
    self.assertIn('\'Reshape\'', output)
    self.assertIn('\'SaveV2\'', output)
    self.assertIn('\'Const\'', output)
    self.assertIn('\'Identity\'', output)
    self.assertIn('\'Assign\'', output)
    self.assertIn('\'ParseExample\'', output)
    self.assertIn('\'StringJoin\'', output)
    self.assertIn('\'MergeV2Checkpoints\'', output)
    self.assertIn('\'NoOp\'', output)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandListOpsNoTags(self):
    # [FUNCTION DOC]
    # Runs show with the list ops option but no tag-set and no all option. Asserts the output is
    # exactly a message saying the option must be paired with a tag-set or the all option followed
    # by the available tag-sets, and that error output is empty.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    exp_out = ('--list_ops must be paired with a tag-set or with --all.\n'
               'The given SavedModel contains the following tag-sets:\n'
               '\'serve\'').strip()

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'show', '--dir', base_path, '--list_ops'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testPrintREFTypeTensor(self):
    # [FUNCTION DOC]
    # Calls the private tensor info printer with a tensor info whose dtype is the float reference
    # type. Asserts the printed text contains the reference type name and error output is empty.
    ref_tensor_info = meta_graph_pb2.TensorInfo(
        dtype=types_pb2.DT_FLOAT_REF)
    with captured_output() as (out, err):
      saved_model_cli._print_tensor_info(ref_tensor_info)
    self.assertIn('DT_FLOAT_REF', out.getvalue().strip())
    self.assertEqual(err.getvalue().strip(), '')

  def testInputPreProcessFormats(self):
    # [FUNCTION DOC]
    # Parses an inputs string with a file path plus variable key and a bare file name, and an
    # expressions string (zeros array and a list) in unsafe mode. Asserts the first parse maps
    # names to (path, key) pairs with a null key when none is given, the expression results equal
    # the expected arrays, and each result mapping has two entries. Prints one result.
    input_str = 'input1=/path/file.txt[ab3];input2=file2'
    input_expr_str = 'input3=np.zeros([2,2]);input4=[4,5]'
    input_dict = saved_model_cli.preprocess_inputs_arg_string(input_str)
    input_expr_dict = saved_model_cli.preprocess_input_exprs_arg_string(
        input_expr_str, safe=False)
    self.assertEqual(input_dict['input1'], ('/path/file.txt', 'ab3'))
    self.assertEqual(input_dict['input2'], ('file2', None))
    print(input_expr_dict['input3'])
    self.assertAllClose(input_expr_dict['input3'], np.zeros([2, 2]))
    self.assertAllClose(input_expr_dict['input4'], [4, 5])
    self.assertLen(input_dict, 2)
    self.assertLen(input_expr_dict, 2)

  def testInputPreProcessExamplesWithStrAndBytes(self):
    # [FUNCTION DOC]
    # Parses an input examples string whose single example has a text feature of a string and a
    # bytes feature. Deserializes the first resulting Example and asserts it equals the expected
    # proto with both features stored as byte lists.
    input_examples_str = 'inputs=[{"text":["foo"], "bytes":[b"bar"]}]'
    input_dict = saved_model_cli.preprocess_input_examples_arg_string(
        input_examples_str)
    feature = example_pb2.Example.FromString(input_dict['inputs'][0])
    self.assertProtoEquals(
        """
          features {
            feature {
              key: "bytes"
              value {
                bytes_list {
                  value: "bar"
                }
              }
            }
            feature {
              key: "text"
              value {
                bytes_list {
                  value: "foo"
                }
              }
            }
          }
    """, feature)

  def testInputPreprocessExampleWithCodeInjection(self):
    # [FUNCTION DOC]
    # Passes an input examples string containing a function call (a shell command) instead of a
    # literal. Asserts a runtime error saying it is not a valid literal is raised, showing the
    # parser does not evaluate arbitrary code.
    input_examples_str = 'inputs=os.system("echo hacked")'
    with self.assertRaisesRegex(RuntimeError, 'not a valid python literal.'):
      saved_model_cli.preprocess_input_examples_arg_string(input_examples_str)

  def testInputPreProcessFileNames(self):
    # [FUNCTION DOC]
    # Parses an inputs string with Windows style paths (drive letters, spaces, short names) and a
    # name containing a colon. Asserts the results keep the full paths, extract the bracketed key
    # when present, and use a null key otherwise.
    input_str = (r'inputx=C:\Program Files\data.npz[v:0];'
                 r'input:0=c:\PROGRA~1\data.npy')
    input_dict = saved_model_cli.preprocess_inputs_arg_string(input_str)
    self.assertEqual(input_dict['inputx'], (r'C:\Program Files\data.npz',
                                            'v:0'))
    self.assertEqual(input_dict['input:0'], (r'c:\PROGRA~1\data.npy', None))

  def testInputPreProcessErrorBadFormat(self):
    # [FUNCTION DOC]
    # Feeds malformed argument strings: unbalanced brackets and a missing equals sign to the inputs
    # parser (each must raise a runtime error), and a colon instead of an equals sign to the
    # expressions parser in unsafe mode (runtime error mentioning an incorrect format).
    input_str = 'inputx=file[[v1]v2'
    with self.assertRaises(RuntimeError):
      saved_model_cli.preprocess_inputs_arg_string(input_str)
    input_str = 'inputx:file'
    with self.assertRaises(RuntimeError):
      saved_model_cli.preprocess_inputs_arg_string(input_str)
    input_str = 'inputx:np.zeros((5))'
    with self.assertRaisesRegex(RuntimeError, 'format is incorrect'):
      saved_model_cli.preprocess_input_exprs_arg_string(input_str, safe=False)

  def testInputParserNPY(self):
    # [FUNCTION DOC]
    # Saves two arrays as separate numpy files in the temporary directory and loads them through
    # the inputs string loader, one with a bracketed key and one without. Asserts both loaded
    # values equal the saved arrays.
    x0 = np.array([[1], [2]])
    x1 = np.array(range(6)).reshape(2, 3)
    input0_path = os.path.join(test.get_temp_dir(), 'input0.npy')
    input1_path = os.path.join(test.get_temp_dir(), 'input1.npy')
    np.save(input0_path, x0)
    np.save(input1_path, x1)
    input_str = 'x0=' + input0_path + '[x0];x1=' + input1_path
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        input_str, '', '')
    self.assertTrue(np.all(feed_dict['x0'] == x0))
    self.assertTrue(np.all(feed_dict['x1'] == x1))

  def testInputParserNPZ(self):
    # [FUNCTION DOC]
    # Saves a numpy archive with one array and loads it twice via the inputs loader: once with the
    # array key in brackets and once without a key (allowed since the archive has one entry).
    # Asserts both loaded inputs equal the saved array.
    x0 = np.array([[1], [2]])
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0)
    input_str = 'x=' + input_path + '[a];y=' + input_path
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        input_str, '', '')
    self.assertTrue(np.all(feed_dict['x'] == x0))
    self.assertTrue(np.all(feed_dict['y'] == x0))

  def testInputParserPickle(self):
    # [FUNCTION DOC]
    # Writes three pickle files: a mapping with an array under a key, a bare array, and another
    # array. Loads them through the inputs loader (the first with a key, the second with a key that
    # is ignored for non-mappings, the third with none). Asserts each loaded value equals the expected array.
    pkl0 = {'a': 5, 'b': np.array(range(4))}
    pkl1 = np.array([1])
    pkl2 = np.array([[1], [3]])
    input_path0 = os.path.join(test.get_temp_dir(), 'pickle0.pkl')
    input_path1 = os.path.join(test.get_temp_dir(), 'pickle1.pkl')
    input_path2 = os.path.join(test.get_temp_dir(), 'pickle2.pkl')
    with open(input_path0, 'wb') as f:
      pickle.dump(pkl0, f)
    with open(input_path1, 'wb') as f:
      pickle.dump(pkl1, f)
    with open(input_path2, 'wb') as f:
      pickle.dump(pkl2, f)
    input_str = 'x=' + input_path0 + '[b];y=' + input_path1 + '[c];'
    input_str += 'z=' + input_path2
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        input_str, '', '')
    self.assertTrue(np.all(feed_dict['x'] == pkl0['b']))
    self.assertTrue(np.all(feed_dict['y'] == pkl1))
    self.assertTrue(np.all(feed_dict['z'] == pkl2))

  def testInputParserErrorNoName(self):
    # [FUNCTION DOC]
    # Saves a numpy archive with two arrays and loads it without naming a key. Asserts a runtime
    # error is raised because the choice is ambiguous.
    x0 = np.array([[1], [2]])
    x1 = np.array(range(5))
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0, b=x1)
    input_str = 'x=' + input_path
    with self.assertRaises(RuntimeError):
      saved_model_cli.load_inputs_from_input_arg_string(input_str, '', '')

  def testInputParserErrorWrongName(self):
    # [FUNCTION DOC]
    # Saves a numpy archive with two arrays and loads it naming a key that does not exist.
    # Asserts a runtime error is raised.
    x0 = np.array([[1], [2]])
    x1 = np.array(range(5))
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0, b=x1)
    input_str = 'x=' + input_path + '[c]'
    with self.assertRaises(RuntimeError):
      saved_model_cli.load_inputs_from_input_arg_string(input_str, '', '')

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputExamples(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (whether to add the TFRT runtime flag; only the false case is
    # registered). Runs the run command on the regress signature with two input examples and an
    # output directory, then loads the saved outputs file and asserts it equals the expected
    # column of values six and four.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(),
                              'input_examples' + ('tfrt' if use_tfrt else ''))

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x_to_y', '--input_examples',
        'inputs=[{"x":[8.0],"x2":[5.0]}, {"x":[4.0],"x2":[3.0]}]',
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(os.path.join(output_dir, 'outputs.npy'))
    y_expected = np.array([[6.0], [4.0]])
    self.assertAllEqual(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandLongInputExamples(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case is registered). Saves a local model whose
    # function parses serialized examples with ten scalar float features and sums them, exported
    # as a signature. Runs the run command with two long input examples and asserts the saved
    # outputs equal the sums forty five and fifty five.

    class DummyModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # Local trackable model with one traced function taking a batch of serialized examples,
      # used to test running the tool with long input example lists.
      """Model with callable polymorphic functions specified."""

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=None, dtype=dtypes.string),
      ])
      def func(self, inputs):
        # [FUNCTION DOC]
        # Traced function: parses the serialized examples in inputs into ten scalar float
        # features (variable0 to variable9) and returns a mapping with the key outputs holding
        # the sum of all features per example.
        ex = parsing_ops.parse_example(serialized=inputs, features={
            'variable0': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable1': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable2': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable3': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable4': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable5': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable6': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable7': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable8': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable9': parsing_config.FixedLenFeature(
                (), dtypes.float32),
        })
        return {'outputs': sum(ex.values())}

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = DummyModel()
    func = getattr(dummy_model, 'func')

    with self.cached_session():
      save.save(dummy_model, saved_model_dir, signatures={'func': func})

    output_dir = os.path.join(
        test.get_temp_dir(),
        'long_input_examples' + ('tfrt' if use_tfrt else ''))

    saved_model_cli.flags.FLAGS.unparse_flags()
    input_examples = (
        'inputs=[{"variable0":[0.0],"variable1":[1.0],"variable2":[2.0],'
        '"variable3":[3.0],"variable4":[4.0],"variable5":[5.0],'
        '"variable6":[6.0],"variable7":[7.0],"variable8":[8.0],'
        '"variable9":[9.0]}, {"variable0":[10.0],"variable1":[1.0],'
        '"variable2":[2.0],"variable3":[3.0],"variable4":[4.0],'
        '"variable5":[5.0],"variable6":[6.0],"variable7":[7.0],'
        '"variable8":[8.0],"variable9":[9.0]}]')
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', saved_model_dir, '--tag_set', 'serve',
        '--signature_def', 'func', '--input_examples', input_examples,
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(os.path.join(output_dir, 'outputs.npy'))
    y_expected = np.array([45.0, 55.0])
    self.assertAllEqual(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandExistingOutdir(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Saves an input archive, deletes any old
    # outputs file, and runs the regress signature with the temporary directory (which already
    # exists) as output directory. Asserts the outputs file holds the expected values three and
    # a half, and four.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(), 'testRunCommand_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'outputs.npy')
    if os.path.exists(output_file):
      os.remove(output_file)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x2_to_y3', '--inputs',
        'inputs=' + input_path + '[x0]', '--outdir', test.get_temp_dir()
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(output_file)
    y_expected = np.array([[3.5], [4.0]])
    self.assertAllClose(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandNewOutdir(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Removes any existing new_dir output
    # directory, runs the serving default signature with an input archive, and asserts the tool
    # created the directory and wrote output y with values two and a half, and three.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandNewOutdir_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default', '--inputs', 'x=' +
        input_path + '[x0]', '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(os.path.join(output_dir, 'y.npy'))
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandOutOverwrite(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Pre-creates an empty output file y in the
    # output directory, then runs with the overwrite option. Asserts the file now contains the
    # expected values two and a half, and three.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandOutOverwrite_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'y.npy')
    open(output_file, 'a').close()

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default', '--inputs', 'x=' +
        input_path + '[x0]', '--outdir', test.get_temp_dir(),
        '--overwrite'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(output_file)
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInvalidInputKeyError(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Runs the regress signature giving an
    # input expression for a name that is not one of the signature's input keys, and asserts a value error is raised.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x2_to_y3',
        '--input_exprs', 'x2=[1,2,3]'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaises(ValueError):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInvalidSignature(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Runs with a signature name that does not
    # exist and asserts a value error saying the signature could not be found.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'INVALID_SIGNATURE',
        '--input_exprs', 'x2=[1,2,3]'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError,
                                'Could not find signature '
                                '"INVALID_SIGNATURE"'):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputExamplesNotListError(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Gives input examples whose value is a single
    # mapping rather than a list and asserts a value error saying it must be a list.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x_to_y',
        '--input_examples', 'inputs={"x":8.0,"x2":5.0}',
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError, 'must be a list'):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputExamplesFeatureValueNotListError(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Gives input examples whose feature values
    # are bare numbers instead of lists and asserts a value error saying the feature value must be a list.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x_to_y',
        '--input_examples', 'inputs=[{"x":8.0,"x2":5.0}]',
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError, 'feature value must be a list'):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputExamplesFeatureBadType(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Gives input examples whose feature value is
    # a nested list and asserts a value error saying the type is not supported.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x_to_y',
        '--input_examples', 'inputs=[{"x":[[1],[2]]}]',
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError, 'is not supported'):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandOutputFileExistError(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Pre-creates an output file y in the output
    # directory and runs without the overwrite option; asserts a runtime error is raised.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandOutOverwrite_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'y.npy')
    open(output_file, 'a').close()

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default', '--inputs', 'x=' +
        input_path + '[x0]', '--outdir', test.get_temp_dir()
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaises(RuntimeError):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputNotGivenError(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Runs a signature without any input option
    # (no inputs, expressions or examples) and asserts an attribute error is raised.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaises(AttributeError):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandWithDebuggerEnabled(self, use_tfrt):
    # [FUNCTION DOC]
    # Parameterized by use_tfrt (only the false case). Runs with the debugger option while the
    # command line debug wrapper session class is patched with a fake that returns the session
    # unchanged. Asserts the wrapper was constructed with a session argument and that the output
    # file y holds the expected values two and a half, and three.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandNewOutdir_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default', '--inputs', 'x=' +
        input_path + '[x0]', '--outdir', output_dir, '--tf_debug'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()

    def fake_wrapper_session(sess):
      # [FUNCTION DOC]
      # HELPER: Fake used as the side effect of the patched debug wrapper class.
      # Takes a session and returns it unchanged so no interactive debugger starts.
      return sess

    with test.mock.patch.object(
        local_cli_wrapper,
        'LocalCLIDebugWrapperSession',
        side_effect=fake_wrapper_session,
        autospec=True) as fake:
      saved_model_cli.run()
      fake.assert_called_with(test.mock.ANY)

    y_actual = np.load(os.path.join(output_dir, 'y.npy'))
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  def testScanCommand(self):
    # [FUNCTION DOC]
    # Runs scan on the test model with default settings. Asserts the output says the serve
    # meta graph has none of the default denylisted ops (the set includes read file, write file
    # and print ops by name).
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'scan', '--dir', base_path])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, _):
      saved_model_cli.scan()
    output = out.getvalue().strip()
    self.assertIn(('MetaGraph with tag set [\'serve\'] does not contain the '
                   'default denylisted ops: {\''), output)
    self.assertIn('\'ReadFile\'', output)
    self.assertIn('\'WriteFile\'', output)
    self.assertIn('\'PrintV2\'', output)

  def testScanCommandFoundCustomDenylistedOp(self):
    # [FUNCTION DOC]
    # Runs scan with the serve tag-set and a custom denylist of three op names, two of which are
    # present in the model. Asserts the output reports denylisted ops found and lists exactly the
    # two present ones, in either order.
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'scan', '--dir', base_path, '--tag_set', 'serve', '--op_denylist',
        'VariableV2,Assign,Relu6'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, _):
      saved_model_cli.scan()
    output = out.getvalue().strip()
    self.assertIn(('MetaGraph with tag set [\'serve\'] contains the following'
                   ' denylisted ops:'), output)
    self.assertTrue(('{\'VariableV2\', \'Assign\'}' in output) or
                    ('{\'Assign\', \'VariableV2\'}' in output))

  def testAOTCompileCPUWrongSignatureDefKey(self):
    # [FUNCTION DOC]
    # Skipped unless XLA is built in. Runs the ahead-of-time compile command with a signature key
    # that does not exist in the model and asserts a value error saying the signature def could
    # not be found.
    if not test.is_built_with_xla():
      self.skipTest('Skipping test because XLA is not compiled in.')

    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'aot_compile_cpu_dir')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'aot_compile_cpu', '--dir', base_path, '--tag_set', 'serve',
        '--output_prefix', output_dir, '--cpp_class', 'Compiled',
        '--signature_def_key', 'MISSING'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError, 'Unable to find signature_def'):
      saved_model_cli.aot_compile_cpu()

  class AOTCompileDummyModel(autotrackable.AutoTrackable):
    # [CLASS DOC]
    # Nested trackable model used by the compile and freeze tests. Holds two float variables
    # (a read-only one and one that is written to) and three traced functions: func2 with an
    # unused second input, func3 with a large input, and func_write that assigns to a variable.
    """Model compatible with XLA compilation."""

    def __init__(self):
      # [FUNCTION DOC]
      # Constructor: creates the two scalar float variables, my_var and write_var, both
      # initialized to one.
      self.var = variables.Variable(1.0, name='my_var')
      self.write_var = variables.Variable(1.0, name='write_var')

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=(2, 2), dtype=dtypes.float32),
        # Test unused inputs.
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
    ])
    def func2(self, x, y):
      # [FUNCTION DOC]
      # Traced function with a two by two float input x and an unused scalar y. Returns a mapping
      # with key res holding x plus the variable my_var; y is deliberately ignored.
      del y
      return {'res': x + self.var}

    @def_function.function(input_signature=[
        # Test large inputs.
        tensor_spec.TensorSpec(shape=(2048, 16), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
    ])
    def func3(self, x, y):
      # [FUNCTION DOC]
      # Traced function with a large 2048 by 16 float input x and an unused scalar y. Returns a
      # mapping with key res holding x plus my_var; y is ignored. Exercises large constants.
      del y
      return {'res': x + self.var}

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
    ])
    def func_write(self, x, y):
      # [FUNCTION DOC]
      # Traced function with two scalar float inputs; y is ignored. Assigns x plus my_var to
      # write_var (a side effect on model state) and returns a mapping with key res holding that variable.
      del y
      self.write_var.assign(x + self.var)
      return {'res': self.write_var}

  @parameterized.named_parameters(
      ('VariablesToFeedNone', '', 'func2', 'x86_64-pc-linux'),
      (
          'VariablesToFeedNoneTargetAarch64Linux',
          '',
          'func2',
          'aarch64-none-linux-gnu',
      ),
      (
          'VariablesToFeedNoneTargetAarch64Android',
          '',
          'func2',
          'aarch64-none-android',
      ),
      ('VariablesToFeedAll', 'all', 'func2', 'x86_64-pc-linux'),
      ('VariablesToFeedMyVar', 'my_var', 'func2', 'x86_64-pc-linux'),
      ('VariablesToFeedNoneLargeConstant', '', 'func3', 'x86_64-pc-linux'),
      ('WriteToWriteVar', 'all', 'func_write', 'x86_64-pc-linux'),
  )
  def testAOTCompileCPUFreezesAndCompiles(self, variables_to_feed, func,
                                          target_triple):
    # [FUNCTION DOC]
    # Parameterized over variables_to_feed (none, all or a named variable), func (which model
    # function to export) and target_triple (compilation target). Skips if XLA or the target
    # is unavailable, or for ARM targets on s390x. Saves the model with the chosen function as
    # signature, runs the ahead-of-time compile command with a Generated class name and output
    # prefix, with the warning logger patched. Asserts a warning about the unused input y
    # being pruned, that object, header, metadata and makefile outputs exist, the header declares
    # the class and the x and res accessors but not y, setters for fed variables are const-qualified
    # except for the written variable, and the makefile defines the C++ ABI macro.
    if not test.is_built_with_xla():
      self.skipTest('Skipping test because XLA is not compiled in.')
    if not test.is_cpu_target_available(target_triple.partition('-')[0]):
      self.skipTest('Skipping test because target support is not compiled in.')

    if platform.machine() == 's390x' and 'aarch64' in str(target_triple):
      self.skipTest('Skipping arm tests on s390x.')

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = self.AOTCompileDummyModel()
    func = getattr(dummy_model, func)
    with self.cached_session():
      self.evaluate(dummy_model.var.initializer)
      self.evaluate(dummy_model.write_var.initializer)
      save.save(dummy_model, saved_model_dir, signatures={'func': func})

    output_prefix = os.path.join(test.get_temp_dir(), 'aot_compile_cpu_dir/out')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',  # Use the default serving signature_key.
        'aot_compile_cpu', '--dir', saved_model_dir, '--tag_set', 'serve',
        '--signature_def_key', 'func', '--output_prefix', output_prefix,
        '--variables_to_feed', variables_to_feed,
        '--cpp_class', 'Generated'
        ] + (['--target_triple', target_triple] if target_triple else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with test.mock.patch.object(logging, 'warn') as captured_warn:
      saved_model_cli.aot_compile_cpu()
    self.assertRegex(
        str(captured_warn.call_args),
        'Signature input key \'y\'.*has been pruned while freezing the '
        'graph.')
    self.assertTrue(file_io.file_exists('{}.o'.format(output_prefix)))
    self.assertTrue(file_io.file_exists('{}.h'.format(output_prefix)))
    self.assertTrue(file_io.file_exists(
        '{}_metadata.o'.format(output_prefix)))
    self.assertTrue(
        file_io.file_exists('{}_makefile.inc'.format(output_prefix)))
    header_contents = file_io.read_file_to_string(
        '{}.h'.format(output_prefix))
    self.assertIn('class Generated', header_contents)
    self.assertIn('arg_feed_x_data', header_contents)
    self.assertIn('result_fetch_res_data', header_contents)
    # arg_y got filtered out as it's not used by the output.
    self.assertNotIn('arg_feed_y_data', header_contents)
    if variables_to_feed:
      # Read-only-variables' setters preserve constness.
      self.assertIn('set_var_param_my_var_data(const float', header_contents)
      self.assertNotIn('set_var_param_my_var_data(float', header_contents)
    if func == dummy_model.func_write:  # pylint: disable=comparison-with-callable
      # Writeable variables setters do not preserve constness.
      self.assertIn('set_var_param_write_var_data(float', header_contents)
      self.assertNotIn('set_var_param_write_var_data(const float',
                       header_contents)

    makefile_contents = file_io.read_file_to_string(
        '{}_makefile.inc'.format(output_prefix))
    self.assertIn('-D_GLIBCXX_USE_CXX11_ABI=', makefile_contents)

  def testFreezeModel(self):
    # [FUNCTION DOC]
    # Skipped unless XLA is built in. Saves the compile-test model with func2 as signature and
    # runs the freeze model command feeding all variables, with the warning logger patched.
    # Asserts the output directory contains a frozen graph file and a config text proto file.
    if not test.is_built_with_xla():
      self.skipTest('Skipping test because XLA is not compiled in.')

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = self.AOTCompileDummyModel()
    func = getattr(dummy_model, 'func2')
    with self.cached_session():
      self.evaluate(dummy_model.var.initializer)
      self.evaluate(dummy_model.write_var.initializer)
      save.save(dummy_model, saved_model_dir, signatures={'func': func})

    output_prefix = os.path.join(test.get_temp_dir(), 'aot_compile_cpu_dir/out')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',  # Use the default seving signature_key.
        'freeze_model', '--dir', saved_model_dir, '--tag_set', 'serve',
        '--signature_def_key', 'func', '--output_prefix', output_prefix,
        '--variables_to_feed', 'all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with test.mock.patch.object(logging, 'warn'):
      saved_model_cli.freeze_model()
    self.assertTrue(
        file_io.file_exists(os.path.join(output_prefix, 'frozen_graph.pb')))
    self.assertTrue(
        file_io.file_exists(os.path.join(output_prefix, 'config.pbtxt')))


if __name__ == '__main__':
  test.main()
