# [MODULE DOC]
# Tests for the experimental XLA jit scope module: compile and scope attributes on ops created
# within scopes, gradient scope naming, seeded initializers, and legacy compiled functions.
# Test module; no import-time side effects.
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
"""Tests for python.compiler.xla.jit."""

from absl.testing import parameterized

from tensorflow.python.compiler.xla import jit
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def enable_jit_nonstateful(node_def):
  # [FUNCTION DOC]
  # HELPER: predicate passed as the jit-scope switch in the tests below.
  # Takes an operation definition node, looks up the registered op definition, and returns true when
  # the op is not stateful (so it may be compiled). Raises a value error if the op is not registered.
  op_def = op_def_registry.get(node_def.op)
  if op_def is None:
    raise ValueError("Unregistered op being created: %s" % node_def)

  return not op_def.is_stateful


class JITTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the experimental jit scope: attribute marking of ops, scope numbering and nesting,
  # seeded initializers and interaction with legacy graph-defined functions. Uses a shared
  # compute helper method.

  def compute(self, use_jit, compute_fn):
    # [FUNCTION DOC]
    # HELPER: shared runner for the tests in this class.
    # Seeds the random generator, builds a new graph and session, calls the supplied builder
    # callback inside a jit scope configured by use_jit (a boolean or a predicate on nodes),
    # initializes global variables, and returns a pair of the built tensor and its evaluated value.
    random_seed.set_random_seed(1234)
    with self.session(graph=ops.Graph()) as sess:
      with jit.experimental_jit_scope(use_jit):
        r = compute_fn()
      sess.run(variables.global_variables_initializer())
      return (r, sess.run(r))

  @test_util.run_v2_only
  def testJITInEager(self):
    # [FUNCTION DOC]
    # Only for the v2 behavior. Entering a jit scope while eager execution is on must raise a
    # runtime error saying the scope is unsupported in eager mode; the test asserts that message.

    with self.assertRaisesRegex(
        RuntimeError, "xla.experimental.jit_scope is not supported when eager "
        "execution is enabled. Try use it inside tf.function."):
      with jit.experimental_jit_scope(True):
        constant_op.constant(1)

  @test_util.build_as_function_and_v1_graph
  def testJITCreateOpsLambda(self):
    # [FUNCTION DOC]
    # Builds a random-uniform op under several jit settings (off, predicate for non-stateful ops,
    # always on) and asserts the compile attribute: false when off, false on the stateful sampler
    # under the predicate, true on the sampler when always on, and true on the output op when
    # enabled. Also asserts values match across runs and between off and predicate settings.
    """Test several ways of customizing the compilation attribute."""
    def create_ops():
      # [FUNCTION DOC]
      # HELPER: builder callback; creates a random uniform tensor of one element under a variable
      # scope with a seeded uniform initializer and returns it.
      with variable_scope.variable_scope(
          "root",
          initializer=init_ops.random_uniform_initializer(
              -0.1, 0.1, seed=2)):
        inputs = random_ops.random_uniform((1,), minval=-10, maxval=10, seed=1)
        return inputs
    v_false_1_t, v_false_1 = self.compute(False, create_ops)
    _, v_false_2 = self.compute(False, create_ops)
    v_true_1_t, v_true_1 = self.compute(enable_jit_nonstateful, create_ops)
    _, v_true_2 = self.compute(enable_jit_nonstateful, create_ops)
    v_all_true_t, _ = self.compute(True, create_ops)
    self.assertFalse(v_false_1_t.op.get_attr("_XlaCompile"))
    v_true_1_t_sampler_op = v_true_1_t.graph.get_operation_by_name(
        "root/random_uniform/RandomUniform")
    v_all_true_t_sampler_op = v_all_true_t.graph.get_operation_by_name(
        "root/random_uniform/RandomUniform")

    self.assertFalse(v_true_1_t_sampler_op.get_attr("_XlaCompile"))
    self.assertTrue(v_all_true_t_sampler_op.get_attr("_XlaCompile"))

    self.assertTrue(v_true_1_t.op.get_attr("_XlaCompile"))
    self.assertTrue(v_all_true_t.op.get_attr("_XlaCompile"))

    # Additionally ensure that where no JIT compilation happens on the
    # random_uniform op, the output values are identical to the case
    # where no JIT compilation happens anywhere.
    self.assertAllClose(v_false_1, v_false_2)
    self.assertAllClose(v_true_1, v_true_2)
    self.assertAllClose(v_false_1, v_true_1)

  @test_util.build_as_function_and_v1_graph
  def testJITXlaScope(self):
    # [FUNCTION DOC]
    # Creates constants inside sequential and nested jit scopes and asserts the scope attribute:
    # a new top-level scope increments the counter (0, 1, 2) while nested scopes inherit their
    # parent's scope name.
    with self.session(graph=ops.Graph()):
      with jit.experimental_jit_scope(True):
        # XlaScope 0
        a1 = constant_op.constant(1)
      with jit.experimental_jit_scope(True):
        # XlaScope 1
        a2 = constant_op.constant(1)
        with jit.experimental_jit_scope(True):
          # XlaScope still 1, depth 1
          a3 = constant_op.constant(1)
          with jit.experimental_jit_scope(True):
            # XlaScope still 1, depth 2
            a4 = constant_op.constant(1)
          # XlaScope still 1, depth 1
          a5 = constant_op.constant(1)
      with jit.experimental_jit_scope(True):
        # XlaScope now 2, depth 0
        a6 = constant_op.constant(1)

    self.assertEqual(b"jit_scope_0", a1.op.get_attr("_XlaScope"))
    self.assertEqual(b"jit_scope_1", a2.op.get_attr("_XlaScope"))
    self.assertEqual(b"jit_scope_1", a3.op.get_attr("_XlaScope"))
    self.assertEqual(b"jit_scope_1", a4.op.get_attr("_XlaScope"))
    self.assertEqual(b"jit_scope_1", a5.op.get_attr("_XlaScope"))
    self.assertEqual(b"jit_scope_2", a6.op.get_attr("_XlaScope"))

  @test_util.build_as_function_and_v1_graph
  def testJITVariableSeed(self):
    # [FUNCTION DOC]
    # Creates a seeded variable under jit off and under the non-stateful predicate and asserts all
    # runs give identical values, i.e. stateful initializers are not compiled by the predicate.
    """Test that the stateful initializer is not marked for compilation.

    XLA does not currently support seeded initialization and XLA initializers
    therefore return different values than non-XLA counterparts.  Here
    we ensure that if we can disable JIT compilation for the initializers and
    get the same variable values as if no JIT compilation happened.
    """
    def create_ops():
      # [FUNCTION DOC]
      # HELPER: builder callback; creates and returns a one-element variable named var under a
      # variable scope with a seeded uniform initializer.
      with variable_scope.variable_scope(
          "root",
          initializer=init_ops.random_uniform_initializer(
              -0.1, 0.1, seed=2)):
        inputs = variable_scope.get_variable("var", (1,))
        return inputs
    _, v_false_1 = self.compute(False, create_ops)
    _, v_false_2 = self.compute(False, create_ops)
    _, v_true_1 = self.compute(enable_jit_nonstateful, create_ops)
    _, v_true_2 = self.compute(enable_jit_nonstateful, create_ops)
    self.assertAllClose(v_false_1, v_false_2)
    self.assertAllClose(v_true_1, v_true_2)
    self.assertAllClose(v_false_1, v_true_1)

  @test_util.build_as_function_and_v1_graph
  def testDefunNoJitScope(self):
    # [FUNCTION DOC]
    # Defines a compiled legacy function multiplying two inputs outside any jit scope, calls it,
    # and asserts the function definition has the compile attribute set and its own scope name
    # derived from the function name.
    with self.session(graph=ops.Graph()):

      @function.Defun(compiled=True, noinline=True)
      def mulop(x1, x2):
        # [FUNCTION DOC]
        # HELPER: function under test; returns the product of its two arguments.
        return x1 * x2
      x = constant_op.constant(1.0)
      r = mulop(x, x)

      # Ensure the forward function is compiled.
      graph_def = r.graph.as_graph_def()
      func_attrs = graph_def.library.function[0].attr
      self.assertTrue(func_attrs["_XlaCompile"].b)
      # No enclosing jit scope so function sets its own value for _XlaScope.
      self.assertEqual(b"function_mulop", func_attrs["_XlaScope"].s)

  @test_util.build_as_function_and_v1_graph
  def testDefunInheritsJitScope(self):
    # [FUNCTION DOC]
    # Defines and calls a compiled legacy function inside a jit scope and asserts the compile
    # attribute is set and the scope attribute is inherited from the enclosing scope (scope zero).
    with self.session(graph=ops.Graph()):
      with jit.experimental_jit_scope(True):
        @function.Defun(compiled=True, noinline=True)
        def mulop(x1, x2):
          # [FUNCTION DOC]
          # HELPER: function under test; returns the product of its two arguments.
          return x1 * x2
        x = constant_op.constant(1.0)
        r = mulop(x, x)

      # Ensure the forward function is compiled.
      graph_def = r.graph.as_graph_def()
      func_attrs = graph_def.library.function[0].attr
      self.assertTrue(func_attrs["_XlaCompile"].b)
      # Ensure _XlaScope is inherited from enclosing context.
      self.assertEqual(b"jit_scope_0", func_attrs["_XlaScope"].s)


class CompilationEnabledInGradientTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that gradient ops of operations created inside jit scopes inherit compilation and scope
  # attributes, including with separate gradient scopes and legacy compiled functions.

  @test_util.build_as_function_and_v1_graph
  def testCompilationInGradient(self):
    # [FUNCTION DOC]
    # Multiplies a matrix by itself twice, only the second inside a jit scope, takes the gradient,
    # and asserts gradient ops of the compiled op carry a true compile attribute while those of
    # the uncompiled op have no such attribute (value error on lookup). Checks the gradient value.
    with self.cached_session():
      x = constant_op.constant([[3.]])
      y_nc = math_ops.matmul(x, x, name="not_compiled")
      with jit.experimental_jit_scope():
        y_c = math_ops.matmul(y_nc, y_nc, name="compiled")
      x_grads = gradients.gradients([y_c], [x])[0]
      operations = x.graph.get_operations()
      c_grad_ops = [
          op for op in operations if "gradients/compiled" in op.name]
      nc_grad_ops = [
          op for op in operations if "gradients/not_compiled" in op.name]
      self.assertGreater(len(c_grad_ops), 0)
      self.assertGreater(len(nc_grad_ops), 0)
      for cg in c_grad_ops:
        self.assertTrue(cg.get_attr("_XlaCompile"))
      for ncg in nc_grad_ops:
        with self.assertRaisesRegex(ValueError, "[Nn]o attr named"):
          ncg.get_attr("_XlaCompile")

      # d/dx (x ** 4) = 4 * (x ** 3)
      self.assertAllClose([[108]], x_grads)

  @test_util.build_as_function_and_v1_graph
  def testCompilationGradientScopeNames(self):
    # [FUNCTION DOC]
    # Builds two matmuls in two successive jit scopes, takes their gradients, and asserts the
    # gradient ops are compiled and reuse the same scope names (zero and one) as their forward ops.
    with self.session(graph=ops.Graph()):
      with jit.experimental_jit_scope():
        # XlaScope 0
        a1 = constant_op.constant([[1.]])
        a1t = math_ops.matmul(a1, a1)
      with jit.experimental_jit_scope():
        # XlaScope 1
        a2 = constant_op.constant([[1.]])
        a2t = math_ops.matmul(a2, a2)

      self.assertEqual(b"jit_scope_0", a1.op.get_attr("_XlaScope"))
      self.assertEqual(b"jit_scope_1", a2.op.get_attr("_XlaScope"))
      grad_a1 = gradients.gradients(a1t, a1, name="GA")[0]
      grad_a2 = gradients.gradients(a2t, a2, name="GB")[0]
      grad_a1 = grad_a1.op.inputs[0]
      grad_a2 = grad_a2.op.inputs[0]
      self.assertTrue(grad_a1.op.get_attr("_XlaCompile"))
      self.assertTrue(grad_a2.op.get_attr("_XlaCompile"))
      self.assertEqual(b"jit_scope_0", grad_a1.op.get_attr("_XlaScope"))
      self.assertEqual(b"jit_scope_1", grad_a2.op.get_attr("_XlaScope"))

  @test_util.build_as_function_and_v1_graph
  def testCompilationSeparateGradientScopeNames(self):
    # [FUNCTION DOC]
    # Same as the previous scope-name test but with separate compiled gradients enabled: asserts
    # gradient ops are compiled and their scope names get a gradient suffix with the gradient name.
    with self.session(graph=ops.Graph()):
      with jit.experimental_jit_scope(True, separate_compiled_gradients=True):
        # XlaScope 0
        a1 = constant_op.constant([[1.]])
        a1t = math_ops.matmul(a1, a1)
      with jit.experimental_jit_scope(True, separate_compiled_gradients=True):
        # XlaScope 1
        a2 = constant_op.constant([[1.]])
        a2t = math_ops.matmul(a2, a2)

      self.assertEqual(b"jit_scope_0", a1.op.get_attr("_XlaScope"))
      self.assertEqual(b"jit_scope_1", a2.op.get_attr("_XlaScope"))
      grad_a1 = gradients.gradients(a1t, a1, name="GA")[0]
      grad_a2 = gradients.gradients(a2t, a2, name="GB")[0]
      grad_a1 = grad_a1.op.inputs[0]
      grad_a2 = grad_a2.op.inputs[0]
      self.assertTrue(grad_a1.op.get_attr("_XlaCompile"))
      self.assertTrue(grad_a2.op.get_attr("_XlaCompile"))
      self.assertEqual(b"jit_scope_0_grad_GA",
                       grad_a1.op.get_attr("_XlaScope"))
      self.assertEqual(b"jit_scope_1_grad_GB",
                       grad_a2.op.get_attr("_XlaScope"))

  @test_util.build_as_function_and_v1_graph
  def testPlaysNicelyWithDefun(self):
    # [FUNCTION DOC]
    # Calls a compiled legacy function in a jit scope and differentiates it. Asserts the forward
    # function is compiled with scope zero, the gradient op is compiled with the same scope, and
    # running gives values one, one and two.
    with self.session(graph=ops.Graph()) as sess:
      with jit.experimental_jit_scope(True):
        @function.Defun(compiled=True, noinline=True)
        def mulop(x1, x2):
          # [FUNCTION DOC]
          # HELPER: function under test; returns the product of its two arguments.
          return x1 * x2
        x = constant_op.constant(1.0)
        r = mulop(x, x)
        g_r = gradients.gradients(r, x, name="GA")[0]

      # Ensure the forward function is compiled.
      graph_def = r.graph.as_graph_def()
      func_attrs = graph_def.library.function[0].attr
      self.assertTrue(func_attrs["_XlaCompile"].b)
      self.assertEqual(b"jit_scope_0", func_attrs["_XlaScope"].s)

      # Ensure the gradient (SymbolicGradient) is compiled, with the same
      # _XlaScope as the function itself.
      grad_op = g_r.op.inputs[0].op
      self.assertTrue(grad_op.get_attr("_XlaCompile"))
      self.assertEqual(b"jit_scope_0", grad_op.get_attr("_XlaScope"))

      # Ensure the ops run: grad(x1*x1) = 2*x1
      self.assertAllClose([1.0, 1.0, 2.0], sess.run([x, r, g_r]))

  @test_util.build_as_function_and_v1_graph
  def testPlaysNicelyWithDefunSeparateGradientScope(self):
    # [FUNCTION DOC]
    # Like the previous test but the function requests separate compiled gradients: the gradient op
    # must be compiled with a distinct scope name carrying the gradient suffix, and values match.
    with self.session(graph=ops.Graph()) as sess:
      with jit.experimental_jit_scope(True):

        @function.Defun(
            compiled=True, noinline=True, separate_compiled_gradients=True)
        def mulop(x1, x2):
          # [FUNCTION DOC]
          # HELPER: function under test; returns the product of its two arguments.
          return x1 * x2

        x = constant_op.constant(1.0)
        r = mulop(x, x)
        g_r = gradients.gradients(r, x, name="GA")[0]

      # Ensure the forward function is compiled.
      graph_def = r.graph.as_graph_def()
      func_attrs = graph_def.library.function[0].attr
      self.assertTrue(func_attrs["_XlaCompile"].b)
      self.assertEqual(b"jit_scope_0", func_attrs["_XlaScope"].s)

      # Ensure the gradient (SymbolicGradient) is compiled, with a different
      # _XlaScope from the function itself.
      grad_op = g_r.op.inputs[0].op
      self.assertTrue(grad_op.get_attr("_XlaCompile"))
      self.assertEqual(b"jit_scope_0_grad_GA",
                       grad_op.get_attr("_XlaScope"))

      # Ensure the ops run: grad(x1*x1) = 2*x1
      self.assertAllClose([1.0, 1.0, 2.0], sess.run([x, r, g_r]))


if __name__ == "__main__":
  test.main()
