# [MODULE DOC]
# Tests for the XLA compile module: the compile context (unsupported op detection, variable rules,
# nesting, tagging), the compile entry point in eager and traced code, and the argument-count check.
# Test module; no import-time side effects.
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
"""Tests for python.compiler.xla.xla."""

from absl.testing import parameterized

from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.tpu import tpu_feed


_EXPECTED_LOSS = 1
_EXPECTED_FEATURE = 2
_EXPECTED_LABEL = 3


class XLACompileContextTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the XLA compile context: detection of unsupported ops, variable handling, nesting
  # errors, compile-id tagging, pivot control dependencies, and feed/fetch prevention.
  # Shared fixture: a helper that builds a context with a pivot no-op in the default graph.

  def create_test_xla_compile_context(self):
    # [FUNCTION DOC]
    # HELPER: shared fixture builder for the tests in this class.
    # Creates a unique computation name in the default graph, adds a no-op pivot named after it,
    # and returns a new compile context for that name and pivot. The context is not entered.
    computation_name = ops.get_default_graph().unique_name('computation')
    pivot = control_flow_ops.no_op(name=computation_name + '/pivot')
    return xla.XLACompileContext(name=computation_name, pivot=pivot)

  @test_util.run_v1_only('Testing graph mode behavior only')
  def test_report_unsupported_operations_graph_mode(self):
    # [FUNCTION DOC]
    # Graph mode only (v1). Inside a compile context creates audio, histogram, image, scalar and
    # tensor summaries, a merged summary and a print op, then asserts the context recorded exactly
    # those operations (in creation order) as unsupported.
    """Tests that unsupported operations are detected."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    dummy_tensor = constant_op.constant(1.1)
    audio_summary = summary.audio('audio_summary', dummy_tensor, 0.5)
    histogram_summary = summary.histogram('histogram_summary', dummy_tensor)
    image_summary = summary.image('image_summary', dummy_tensor)
    scalar_summary = summary.scalar('scalar_summary', dummy_tensor)
    tensor_summary = summary.tensor_summary('tensor_summary', dummy_tensor)
    summary.merge(
        [
            audio_summary, histogram_summary, image_summary, scalar_summary,
            tensor_summary
        ],
        name='merge_summary')
    logging_ops.Print(dummy_tensor, [dummy_tensor], name='print_op')
    context.Exit()

    unsupported_ops_names = [op.name for op in context._unsupported_ops]
    self.assertEqual(unsupported_ops_names, [
        u'audio_summary', u'histogram_summary', u'image_summary',
        u'scalar_summary', u'tensor_summary', u'merge_summary/merge_summary',
        u'print_op'
    ])

  @test_util.run_v1_only('Testing graph mode behavior only')
  def test_resource_variable_graph_mode(self):
    # [FUNCTION DOC]
    # Graph mode only. Assigns to a resource variable inside an entered compile context and
    # asserts nothing raises (resource variables are allowed).
    """Tests that resource variable usage is allowed."""
    a = variable_scope.get_variable(
        name='variable_a', use_resource=True, initializer=1)

    context = self.create_test_xla_compile_context()
    context.Enter()
    a.assign(2)
    context.Exit()

  def test_resource_variable_in_function(self):
    # [FUNCTION DOC]
    # Within a traced function, enters a compile context, assigns two to a resource variable, exits
    # the context, and returns the assignment result. Asserts the evaluated result equals two.
    """Tests that resource variable usage is allowed."""
    a = variable_scope.get_variable(
        name='variable_a', use_resource=True, initializer=1)

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # HELPER: traced function; assigns two to the outer variable inside a compile context and
      # returns the assignment result.
      context = self.create_test_xla_compile_context()
      context.Enter()
      o = a.assign(2)
      context.Exit()
      return o

    self.assertEqual(self.evaluate(func()), 2)

  @test_util.run_v1_only('Testing v1-only ref variable handling.')
  def test_non_resource_variable_error(self):
    # [FUNCTION DOC]
    # Graph mode only. Assigns to a non-resource (reference) variable inside a compile context and
    # asserts a not-implemented error saying non-resource variables are unsupported in XLA.
    """Tests that non-resource variable usage is disallowed."""
    a = variable_scope.get_variable(
        name='variable_a', shape=(1), use_resource=False)

    context = self.create_test_xla_compile_context()
    context.Enter()
    with self.assertRaisesRegex(
        NotImplementedError, 'Non-resource Variables are not supported inside '
        r'XLA computations \(operator name: Assign\)'):
      state_ops.assign(a, a + 1)
    context.Exit()

  @test_util.build_as_function_and_v1_graph
  def test_nested_xla_compile_error(self):
    # [FUNCTION DOC]
    # Enters one compile context and then a second; creating an op in the inner one must raise a
    # value error saying compiled computations cannot be nested.
    """Tests that nested XLA computation leads to fatal error."""
    context1 = self.create_test_xla_compile_context()
    context1.Enter()

    context2 = self.create_test_xla_compile_context()
    context2.Enter()
    with self.assertRaisesRegex(ValueError,
                                'XLA compiled computations cannot be nested'):
      constant_op.constant(1)
    context2.Exit()
    context1.Exit()

  @test_util.build_as_function_and_v1_graph
  def test_xla_compile_attr(self):
    # [FUNCTION DOC]
    # Creates a constant inside a compile context and asserts its node definition carries the
    # compile-id attribute.
    """Tests that ops are tagged with XLA compile ID attribute."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertIn('_xla_compile_id', op.op.node_def.attr)

  @test_util.build_as_function_and_v1_graph
  def test_op_without_input(self):
    # [FUNCTION DOC]
    # Creates an input-less constant inside a compile context and asserts the context's pivot op
    # appears among the op's control inputs.
    """Tests that ops without inputs depend on pivot correctly."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()

    self.assertIn(context._pivot, op.op.control_inputs)

  @test_util.run_v1_only('Testing graph mode behavior only')
  def test_external_control_edges_graph_mode(self):
    # [FUNCTION DOC]
    # Graph mode only. Checks a control dependency on an outside op is normally recorded, then runs
    # a while loop whose body creates an op inside a compile context under that same dependency and
    # asserts the outside op is NOT among the new op's control inputs (external edges are dropped).
    """Tests that external control edges are handled correctly."""
    i = constant_op.constant(1)
    op1 = constant_op.constant(1)

    with ops.control_dependencies([op1]):
      op2 = constant_op.constant(1)
    self.assertIn(op1.op, op2.op.control_inputs)

    def while_body(i):
      # [FUNCTION DOC]
      # HELPER: loop body for the test; ignores its loop variable, creates a constant in a compile
      # context under an external control dependency, asserts the dependency is dropped, and
      # returns the new constant.
      del i  # unused
      context = self.create_test_xla_compile_context()
      context.Enter()
      with ops.control_dependencies([op1]):
        op3 = constant_op.constant(1)
      context.Exit()
      self.assertNotIn(op1.op, op3.op.control_inputs)
      return op3

    while_loop.while_loop(
        cond=lambda i: math_ops.less(i, 10), body=while_body, loop_vars=[i])

  @test_util.build_as_function_and_v1_graph
  def test_op_output_marked_as_seen(self):
    # [FUNCTION DOC]
    # Creates a constant in a compile context and asserts its name is in the context's set of
    # seen values.
    """Tests that any op output is marked as seen in context."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()

    self.assertIn(op.name, context._values)

  @test_util.build_as_function_and_v1_graph
  def test_op_is_in_context(self):
    # [FUNCTION DOC]
    # Creates one op outside and one inside a compile context and asserts the XLA-context check is
    # false for the first and true for the second.
    """Tests that XLACompileContext is recognized as an XLA context."""
    op1 = constant_op.constant(1)
    context = self.create_test_xla_compile_context()
    context.Enter()
    op2 = constant_op.constant(2)
    context.Exit()
    self.assertFalse(control_flow_util.IsInXLAContext(op1.op))
    self.assertTrue(control_flow_util.IsInXLAContext(op2.op))

  @test_util.build_as_function_and_v1_graph
  def test_op_prevent_feeding(self):
    # [FUNCTION DOC]
    # Asserts an op created in a compile context is reported as not feedable by its graph.
    """Tests that ops created inside XLACompileContext can not be fed."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertFalse(op.graph.is_feedable(op.op))

  @test_util.build_as_function_and_v1_graph
  def test_op_prevent_fetching(self):
    # [FUNCTION DOC]
    # Asserts an op created in a compile context is reported as not fetchable by its graph.
    """Tests that ops created inside XLACompileContext can not be fetched."""
    context = self.create_test_xla_compile_context()
    context.Enter()
    op = constant_op.constant(1)
    context.Exit()
    self.assertFalse(op.graph.is_fetchable(op.op))


class XlaCompileTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the compile entry point of the XLA module when used eagerly and inside traced
  # functions, including variable writes. No shared fixtures.

  @test_util.run_v2_only
  @test_util.disable_tfrt(
      'Legacy XLA test. It depends on EncapsulateXlaComputationsPass.')
  def test_xla_compile_eager(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says compile raises an exception in eager mode, but the test
    # asserts it succeeds and that adding one and two evaluates to three.
    # Only for v2. Compiles an add computation with inputs one and two and checks the result.
    """Tests that xla.compile raises proper exception when used eagerly."""

    def computation(a, b):
      # [FUNCTION DOC]
      # HELPER: computation passed to the compiler; returns the sum of its two arguments.
      return a + b

    self.assertEqual(self.evaluate(xla.compile(computation, [1, 2])[0]), 3)

  @test_util.disable_tfrt(
      'Legacy XLA test. It depends on EncapsulateXlaComputationsPass.')
  def test_xla_compile_in_function(self):
    # [FUNCTION DOC]
    # Inside a traced function, compiles a computation that adds one to its input, and asserts
    # that calling the function with one returns two as the first output.
    """Tests that xla.compile works in tf.function."""

    @def_function.function
    def func_wrapper(a):
      # [FUNCTION DOC]
      # HELPER: traced function; compiles an inner add-one computation on its argument and returns
      # the compiled outputs.

      def compute(a):
        # [FUNCTION DOC]
        # HELPER: computation to be compiled; returns its argument plus one.
        return a + 1

      return xla.compile(compute, [a])

    self.assertEqual(self.evaluate(func_wrapper(1))[0], 2)

  @test_util.disable_tfrt(
      'Legacy XLA test. It depends on EncapsulateXlaComputationsPass.')
  def test_xla_compile_write_variable_in_function(self):
    # [FUNCTION DOC]
    # Creates a resource variable initialized to one, and a traced function compiling a computation
    # that adds one, subtracts two and reads it. After initializing, asserts the first output is
    # zero.
    """Tests that xla.compile works with variable in tf.function."""
    a = variable_scope.get_variable(
        name='variable_a', use_resource=True, initializer=1)

    @def_function.function
    def func_wrapper():
      # [FUNCTION DOC]
      # HELPER: traced function; compiles the inner variable-updating computation and returns
      # its outputs.

      def compute():
        # [FUNCTION DOC]
        # HELPER: computation to be compiled; adds one to the outer variable, subtracts two, and
        # returns the variable's read value.
        a.assign_add(1)
        a.assign_sub(2)
        return a.read_value()

      return xla.compile(compute)

    self.evaluate(a.initializer)
    self.assertEqual(self.evaluate(func_wrapper())[0], 0)


class CheckFunctionArgumentCountTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the argument-count checking helper of the XLA module, using functions with plain
  # arguments, defaults, variadic arguments, and optionally an infeed queue that supplies some
  # arguments. The helper returns nothing when valid, else a description of the expected count.

  def test_simple(self):
    # [FUNCTION DOC]
    # For a function with three required arguments, asserts the checker accepts three and rejects
    # two with the message about exactly three arguments; with a two-slot infeed queue, accepts
    # one and rejects two.
    """Tests that arg checker works for functions with no varargs or defaults.
    """

    def func(x, y, z):
      # [FUNCTION DOC]
      # HELPER: function under check; takes three required arguments and returns their sum.
      return x + y + z

    self.assertEqual(None, xla.check_function_argument_count(func, 3, None))
    self.assertEqual('exactly 3 arguments',
                     xla.check_function_argument_count(func, 2, None))
    queue = tpu_feed.InfeedQueue(2)
    self.assertEqual(None, xla.check_function_argument_count(func, 1, queue))
    self.assertEqual('exactly 3 arguments',
                     xla.check_function_argument_count(func, 2, queue))

  def test_default_args(self):
    # [FUNCTION DOC]
    # For a function with two required and one defaulted argument, asserts the checker accepts two
    # or three arguments, rejects one ("at least 2") and four ("at most 3"), and the same limits
    # shifted by one when a one-slot infeed queue supplies an argument.
    """Tests that arg checker works for a function with no varargs."""

    def func(x, y, z=17):
      # [FUNCTION DOC]
      # HELPER: function under check; two required and one defaulted argument; returns the sum.
      return x + y + z

    self.assertEqual(None, xla.check_function_argument_count(func, 3, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 2, None))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 1, None))
    self.assertEqual('at most 3 arguments',
                     xla.check_function_argument_count(func, 4, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None, xla.check_function_argument_count(func, 2, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 1, queue))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 0, queue))
    self.assertEqual('at most 3 arguments',
                     xla.check_function_argument_count(func, 4, queue))

  def test_var_args(self):
    # [FUNCTION DOC]
    # For a function with two required arguments and variadic extras, asserts the checker accepts
    # two or more arguments and rejects fewer ("at least 2"), with and without a one-slot infeed
    # queue (which shifts the counts by one).
    """Tests that arg checker works for a function with varargs."""

    def func(x, y, *z):
      # [FUNCTION DOC]
      # HELPER: function under check; two required plus variadic arguments; returns the two
      # required values summed with the number of extras.
      return x + y + len(z)

    self.assertEqual(None, xla.check_function_argument_count(func, 2, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 3, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 4, None))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 1, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None, xla.check_function_argument_count(func, 1, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 2, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 3, queue))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 0, queue))

  def test_var_args_and_defaults(self):
    # [FUNCTION DOC]
    # For a function with two required, one defaulted and variadic arguments, asserts the checker
    # accepts two or more arguments and rejects fewer ("at least 2"), with and without a one-slot
    # infeed queue (which shifts the counts by one).
    """Tests that arg checker works for a function with varargs and defaults."""

    def func(x, y, z=17, *q):  # pylint: disable=keyword-arg-before-vararg
      # [FUNCTION DOC]
      # HELPER: function under check; two required, one defaulted and variadic arguments; returns
      # their sum plus the number of extras.
      return x + y + z + len(q)

    self.assertEqual(None, xla.check_function_argument_count(func, 2, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 3, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 4, None))
    self.assertEqual(None, xla.check_function_argument_count(func, 5, None))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 1, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None, xla.check_function_argument_count(func, 1, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 2, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 3, queue))
    self.assertEqual(None, xla.check_function_argument_count(func, 4, queue))
    self.assertEqual('at least 2 arguments',
                     xla.check_function_argument_count(func, 0, queue))


if __name__ == '__main__':
  test.main()
