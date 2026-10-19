# [MODULE DOC]
# Test module for the while_v2 implementation of while loops (control flow version two), run
# through its public loop-building entry point. It defines one large parameterized test case plus
# small module-level helpers. The tests cover forward results and gradients (single and multiple
# loop variables, nested loops, double derivatives, tensor lists and tensor arrays), pruning of
# unused accumulators in optimized graphs, which tensors get accumulated for gradients, naming of
# the loop and its condition and body functions, name scope and device inheritance, lowering and
# executor configuration, graph flags, importing a serialized graph, and ragged shape invariants.
# Module-level helpers wrap random and fill ops, build a scalar shape tensor and optimize a graph.
# There are no import-time side effects. Running as a script starts the TensorFlow test runner.
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
"""Tests for while_v2."""

from absl.testing import parameterized

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.while_v2 import while_loop as while_loop_v2
from tensorflow.python.platform import test


def random_gamma(shape):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: Random-op factory used by the parameterized random-ops shape test.
  # Takes a shape tensor and returns gamma-distributed random samples with shape parameter one, so
  # the result has the requested shape. Draws from the global random state.
  return random_ops.random_gamma(shape, 1.0)


def random_gamma_with_alpha_beta(shape):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: Random-op factory used by the parameterized random-ops shape test.
  # Takes a shape tensor and returns gamma samples using fixed per-row shape parameters (a column of
  # four values) and fixed rate parameters (a row of two values), so the result shape is the requested
  # shape followed by four and two. Draws from the global random state.
  return random_ops.random_gamma(
      shape, alpha=[[1.], [3.], [5.], [6.]], beta=[[3., 4.]])


def random_poisson_v2(shape):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: Random-op factory used by the parameterized random-ops shape test.
  # Takes a shape tensor and returns Poisson samples with rate one, shaped as requested.
  # Draws from the global random state.
  return random_ops.random_poisson_v2(shape, 1.0)


def random_poisson_v2_with_lam(shape):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: Random-op factory used by the parameterized random-ops shape test.
  # Takes a shape tensor and returns Poisson samples using two fixed rate values, so the result shape
  # is the requested shape followed by two. Draws from the global random state.
  return random_ops.random_poisson_v2(shape, [12.2, 3.3])


def fill(shape):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: Fill-op factory used by the parameterized fill-ops shape test.
  # Takes a shape tensor and returns a tensor of that shape filled with the float value one.
  return array_ops.fill(shape, 1.0)


class WhileV2Test(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the while_v2 loop implementation, also parameterized through absl. Covers results
  # and gradients of loops in graph and eager/function modes, nested and consecutive loops, tensor
  # list accumulators and pruning, naming, device and name-scope handling, session configuration and
  # serialized-graph import. Has no shared fixtures; each test builds its own tensors, and several
  # use the module helpers for random ops, scalar shapes and graph optimization. Several tests are
  # restricted to legacy graph mode, to the second TensorFlow version only, or enable control flow v2.

  @test_util.run_deprecated_v1
  def testSingleLoopVar(self):
    # [FUNCTION DOC]
    # Builds a loop that squares a scalar starting at two while it is below eight, returning a single
    # tensor instead of a structure, and takes its gradient. Asserts the result is sixteen and the
    # gradient with respect to the start value is thirty-two. Legacy graph mode only.
    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8., lambda v: v * v, [x], return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @test_util.run_deprecated_v1
  def testSingleLoopVarBackPropFalse(self):
    # [FUNCTION DOC]
    # Same squaring loop as the single-variable test but with backpropagation disabled. Asserts the
    # gradient computation yields a list containing only a null value (no gradient flows through the
    # loop) while the forward result is still sixteen. Legacy graph mode only.
    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8.,
        lambda v: v * v, [x],
        return_same_structure=False,
        back_prop=False)
    grad = gradients_impl.gradients(ret, [x])
    self.assertEqual(grad, [None])
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 16.)

  @test_util.run_deprecated_v1
  def testCustomGradient(self):
    # [FUNCTION DOC]
    # Runs a loop whose body calls a function with a user-defined gradient that also references an
    # outer constant and a variable. Initializes variables, then asserts the forward result is sixteen
    # and the gradient with respect to the input is thirty-two. Legacy graph mode only.
    x = constant_op.constant(2.)
    n = constant_op.constant(1., name="const-n")
    m = variables.Variable(1.0)
    self.evaluate(variables.global_variables_initializer())

    def body_fn(v):  # pylint: disable=invalid-name
      # [FUNCTION DOC]
      # HELPER: Loop body used by the custom-gradient test.
      # Takes the loop value and returns it passed through the inner custom-gradient function.

      @custom_gradient.custom_gradient
      def inner_fn(v):  # pylint: disable=invalid-name
        # [FUNCTION DOC]
        # HELPER: Custom-gradient function wrapped around the squaring step in the custom-gradient test.
        # Returns the value squared times the captured variable together with a gradient function.

        def grad_fn(dy, variables=None):  # pylint: disable=invalid-name, unused-argument, redefined-outer-name
          # [FUNCTION DOC]
          # HELPER: Gradient function for the inner custom-gradient function.
          # Takes the upstream gradient and an optional list of variables; returns the input gradient (upstream
          # times two times the value times the captured constant and variable) and a list of variable
          # gradients containing the squared value. Test-only math chosen to give thirty-two overall.
          return dy * 2 * v * n * m, [v * v]

        return v * v * m, grad_fn

      return inner_fn(v)

    ret = while_loop_v2(
        lambda v: v < 8., body_fn, [x], return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @test_util.run_v1_only("b/120545219")
  def testReturnSameStructureTrue(self):
    # [FUNCTION DOC]
    # Builds the squaring loop with the option that returns the same structure as the inputs, so the
    # result is a one-element list. Runs it in a session and asserts the result is a list of length one
    # holding sixteen and that the gradient is thirty-two. Graph mode only (skipped for the second
    # version, with a bug reference).
    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8., lambda v: v * v, [x], return_same_structure=True)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session() as sess:
      eval_result = sess.run(ret)
      self.assertIsInstance(eval_result, list)
      self.assertLen(eval_result, 1)
      self.assertEqual(16., eval_result[0])
      self.assertSequenceEqual(sess.run(grad), [32.])

  def testVerifyInputOutputTypesMatch(self):
    # [FUNCTION DOC]
    # Builds, inside a compiled function, a loop whose body casts a 32-bit float loop variable to 16-bit
    # float and adds one. Asserts that building raises a type error whose message says the loop variable
    # enters the loop with the 32-bit type but has the 16-bit type after one iteration.

    @def_function.function
    def BuildWhile():
      # [FUNCTION DOC]
      # HELPER: Compiled function used by the type-mismatch test.
      # Creates a 32-bit float constant and runs a loop that exits while the value is below ten, with a
      # body that changes the type. Takes no arguments and returns nothing; the error is raised while
      # tracing.
      x = constant_op.constant(1., dtypes.float32)

      def Body(x):
        # [FUNCTION DOC]
        # HELPER: Loop body for the type-mismatch test.
        # Takes the loop value, casts it to 16-bit float and adds one, deliberately changing the type.
        return math_ops.cast(x, dtypes.float16) + 1

      while_loop_v2(lambda x: x < 10, Body, [x])

    with self.assertRaisesRegex(
        TypeError,
        r"Loop var Const:0 enters the loop with type <dtype: 'float32'> "
        r"but has type <dtype: 'float16'> after 1 iteration."):
      BuildWhile()

  @parameterized.parameters(dtypes.float32, dtypes.float64)
  def testGradientTapeResourceVariable(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over 32-bit and 64-bit float types. In eager mode creates a variable of the given
    # type with value one and a compiled function that records a gradient tape around a two-iteration
    # loop multiplying a value of two by the variable each time. Asserts the gradient with respect to
    # the variable equals four.
    with context.eager_mode():
      v = variables.Variable(1., dtype=dtype)

      @def_function.function
      def fnWithLoop():  # pylint: disable=invalid-name
        # [FUNCTION DOC]
        # HELPER: Compiled function used by the resource-variable gradient test.
        # Runs the two-iteration loop under a gradient tape and returns the gradient of the final value with
        # respect to the captured variable. Takes no arguments.
        with backprop.GradientTape() as tape:
          _, x = while_loop_v2(
              lambda i, _: i < 2,
              lambda i, x: (i + 1, x * v),
              [0, constant_op.constant(2., dtype=dtype)])
        return tape.gradient(x, v)

      self.assertAllEqual(fnWithLoop(), 4.0)

  def testDeferredCaptures(self):
    # [FUNCTION DOC]
    # In eager mode, creates a constant ten and a compiled function whose loop body returns a value
    # captured at call time (through the graph's deferred capture mechanism) with an int32 scalar spec.
    # Asserts calling the function returns ten.
    with context.eager_mode():
      c = constant_op.constant(10)

      @def_function.function
      def F():
        # [FUNCTION DOC]
        # HELPER: Compiled function used by the deferred-capture test.
        # Runs a loop limited to one iteration whose body returns the call-time captured value, and returns
        # the single loop result. Takes no arguments.

        def Body(_):
          # [FUNCTION DOC]
          # HELPER: Loop body for the deferred-capture test.
          # Ignores the loop variable and returns a call-time captured value of the outer constant.
          return ops.get_default_graph().capture_call_time_value(
              lambda: c, tensor_spec.TensorSpec([], dtypes.int32))

        x, = while_loop_v2(lambda i: True, Body, [0], maximum_iterations=1)
        return x

      self.assertAllEqual(F(), 10)

  def checkIteratedGradients(self, func):
    # [FUNCTION DOC]
    # HELPER: Shared checker for the iterated-gradient tests.
    # Takes a function of one scalar tensor. In eager mode, three times in a row: compares the
    # theoretical and numerical gradients of the compiled function at input one (relative tolerance
    # one thousandth), then replaces the function by its gradient function and asserts the numerical
    # value matches the compiled new function's value. Raises assertion errors on mismatch.
    with context.eager_mode():

      def _Grad(f):
        # [FUNCTION DOC]
        # HELPER: Builds the gradient function for the iterated-gradient checker.
        # Takes a function and returns a new function computing its gradient.
        def _GradFunction(primal):
          # [FUNCTION DOC]
          # HELPER: Inner function returned by the gradient builder.
          # Takes a tensor, watches it on a gradient tape, applies the wrapped function and returns the
          # gradient of the output with respect to the input.
          with backprop.GradientTape() as tape:
            tape.watch(primal)
            primal_out = f(primal)
          return tape.gradient(primal_out, primal)
        return _GradFunction

      f = func
      one = constant_op.constant(1.)

      for _ in range(3):
        theoretical, numerical = gradient_checker_v2.compute_gradient(
            def_function.function(f), [one])
        self.assertAllClose(theoretical, numerical, rtol=1e-3)
        f = _Grad(f)
        self.assertAllClose(array_ops.reshape(numerical, []),
                            def_function.function(f)(one),
                            rtol=1e-3)

  def testIteratedGradients(self):
    # [FUNCTION DOC]
    # Defines a function applying cosine to a value over two loop iterations and passes it to the
    # iterated-gradient checker, which verifies first, second and third derivatives.

    def _Func(x):
      # [FUNCTION DOC]
      # HELPER: Function under test for the iterated-gradient check.
      # Takes a value, runs a two-iteration loop applying cosine each time and returns the final value.
      _, z = while_loop_v2(
          lambda i, _: i < 2,
          lambda i, y: (i + 1, math_ops.cos(y)),
          [0, x])
      return z

    self.checkIteratedGradients(_Func)

  def testIteratedGradientsWithList(self):
    # [FUNCTION DOC]
    # Like the iterated-gradient test, but the loop also pushes each value onto a tensor list and the
    # result adds the sum of the stacked list. Checks first, second and third derivatives through the
    # shared checker.

    def _Func(x):
      # [FUNCTION DOC]
      # HELPER: Function under test for the list variant of the iterated-gradient check.
      # Takes a value, runs a two-iteration loop applying cosine and pushing values onto a float tensor
      # list, and returns the final value plus the sum of the stacked list.
      results = list_ops.empty_tensor_list(
          element_shape=[], element_dtype=dtypes.float32)

      def _LoopBody(i, y, handle):
        # [FUNCTION DOC]
        # HELPER: Loop body for the list variant.
        # Takes counter, value and list handle; returns counter plus one, cosine of the value, and the
        # list with the value pushed on.
        return (i + 1, math_ops.cos(y),
                list_ops.tensor_list_push_back(handle, y))

      _, z, results = while_loop_v2(
          lambda i, _, h: i < 2, _LoopBody, [0, x, results])
      return z + math_ops.reduce_sum(list_ops.tensor_list_stack(
          results, dtypes.float32))

    self.checkIteratedGradients(_Func)

  def testGradWhileGradWhileWithVariable(self):
    # [FUNCTION DOC]
    # In eager mode, creates a variable and a compiled function whose outer loop calls an inner
    # function that itself differentiates an inner loop. Computes the gradient of the outer function
    # under a persistent tape and asserts the numerical and theoretical gradients agree and match the
    # tape value (tolerance one thousandth), covering gradient-of-loop-in-loop with a variable.
    with context.eager_mode():
      v = variables.Variable(1.)

      @def_function.function
      def _Func(x):
        # [FUNCTION DOC]
        # HELPER: Compiled function under test.
        # Takes a tensor and runs a two-iteration outer loop whose body returns the gradient computed by
        # the inner helper; returns the final value.

        def _Inner(a):
          # [FUNCTION DOC]
          # HELPER: Inner differentiating step.
          # Takes a tensor, watches it on a tape, runs a two-iteration loop applying cosine of the captured
          # variable plus the value, and returns the gradient of the result with respect to the input.
          with backprop.GradientTape() as tape:
            tape.watch(a)
            _, b = while_loop_v2(
                lambda i, _: i < 2,
                lambda i, y: (i + 1, math_ops.cos(v + y)),
                [0, a])
          return tape.gradient(b, a)

        _, z = while_loop_v2(
            lambda i, _: i < 2,
            lambda i, y: (i + 1, _Inner(y)),
            [0, x])
        return z

      with backprop.GradientTape(persistent=True) as tape:
        x = constant_op.constant(1.)
        tape.watch(x)
        y = _Func(x)
      dx, _ = tape.gradient(y, [x, v])
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          _Func, [x])
      self.assertAllClose(numerical, theoretical, rtol=1e-3)
      self.assertAllClose(array_ops.reshape(numerical, []),
                          dx, rtol=1e-3)

  def testThreeNestWithLists(self):
    # [FUNCTION DOC]
    # In eager mode, wraps the sine function in three nested loops that each push values on a tensor
    # list, differentiates it inside a compiled function with a persistent tape, and asserts the tape
    # gradient matches the numerical and theoretical gradients (tolerance three thousandths).
    with context.eager_mode():
      def _WrapInWhile(f):
        # [FUNCTION DOC]
        # HELPER: Wrapper builder for the nested-loop test.
        # Takes a function and returns a function that runs a two-iteration loop applying the function to the
        # cosine of the value while pushing values on a list, returning value plus sum of stacked list.
        def _Wrapped(x):
          # [FUNCTION DOC]
          # HELPER: Function returned by the wrapper builder.
          # Takes a value and runs the loop described by the wrapper, returning the final value plus the sum
          # of the stacked list of visited values.
          results = list_ops.empty_tensor_list(
              element_shape=[], element_dtype=dtypes.float32)

          def _LoopBody(i, y, handle):
            # [FUNCTION DOC]
            # HELPER: Loop body for the wrapper.
            # Takes counter, value and list handle; returns counter plus one, the wrapped function applied to
            # the cosine of the value, and the list with the value pushed on.
            return (i + 1, f(math_ops.cos(y)),
                    list_ops.tensor_list_push_back(handle, y))

          _, z, results = while_loop.while_loop(lambda i, _, h: i < 2,
                                                _LoopBody, [0, x, results])
          return z + math_ops.reduce_sum(list_ops.tensor_list_stack(
              results, dtypes.float32))
        return _Wrapped

      f = math_ops.sin

      target_function = _WrapInWhile(_WrapInWhile(_WrapInWhile(f)))

      @def_function.function
      def _TapeFromGraphMode(x):
        # [FUNCTION DOC]
        # HELPER: Compiled function used by the nested-loop test.
        # Takes a tensor, watches it on a persistent tape, applies the three-nest function and returns the
        # gradient with respect to the input.
        with backprop.GradientTape(persistent=True) as tape:
          tape.watch(x)
          y = target_function(x)
        return tape.gradient(y, x)

      x = constant_op.constant(1.)
      dx = _TapeFromGraphMode(x)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          target_function, [x])
      self.assertAllClose(numerical, theoretical, rtol=3e-3)
      self.assertAllClose(array_ops.reshape(numerical, []), dx, rtol=3e-3)

  def testDeviceLabelsInherited(self):
    # [FUNCTION DOC]
    # Builds a compiled function with a loop inside a CPU device scope (index ten) whose body checks
    # that the cosine result inherits that device, then creates an identity under another device scope
    # (index eleven) and checks that label. The assertions run at trace time, triggered by requesting
    # the concrete function; nothing is evaluated.
    def _LoopBody(i, y):
      # [FUNCTION DOC]
      # HELPER: Loop body for the device-label test.
      # Takes counter and value, asserts the cosine result has the first CPU device, creates an identity
      # under the second CPU device and asserts that, then returns counter plus one and the identity.
      result = math_ops.cos(y)
      self.assertIn("CPU:10", result.device)
      with ops.device("CPU:11"):
        result = array_ops.identity(result)
      self.assertIn("CPU:11", result.device)
      return i + 1, result

    @def_function.function
    def _FunctionWithWhileLoop():
      # [FUNCTION DOC]
      # HELPER: Compiled function for the device-label test.
      # Runs a two-iteration loop under the first CPU device scope and returns the final value.
      x = constant_op.constant(1.)
      with ops.device("CPU:10"):
        _, z = while_loop_v2(
            lambda i, _: i < 2,
            _LoopBody,
            [0, x])
      return z
    # The test assertion runs at trace time.
    _FunctionWithWhileLoop.get_concrete_function()

  def testExternalControlDependencies(self):
    # [FUNCTION DOC]
    # In a fresh graph and session, creates and initializes a variable, builds an increment operation
    # and a loop whose body depends on it through a control dependency. Runs the loop and asserts the
    # variable became two, showing external control dependencies are honored inside the loop.
    with ops.Graph().as_default(), self.test_session():
      v = variables.Variable(1.)
      self.evaluate(v.initializer)
      op = v.assign_add(1.)

      def body_fn(i):  # pylint: disable=invalid-name
        # [FUNCTION DOC]
        # HELPER: Loop body for the control-dependency test.
        # Takes the counter and returns it plus one under a control dependency on the increment operation.
        with ops.control_dependencies([op]):
          return i + 1

      loop = while_loop_v2(lambda i: i < 1, body_fn, [0])
      loop[0].op.run()
      self.assertAllEqual(self.evaluate(v), 2.0)

  @test_util.run_deprecated_v1
  def testMultipleLoopVarsBasic(self):
    # [FUNCTION DOC]
    # Runs a loop with two variables (five and three) multiplying the first by the second while below
    # forty-five, with the second unchanged. Asserts results forty-five and three and that the gradient
    # of the outputs with respect to the first input is nine. Legacy graph mode only.
    x = constant_op.constant(5.)
    y = constant_op.constant(3.)

    # x = 5.
    # y = 3.
    # while x < 45.:
    #   x = x * y
    ret = while_loop_v2(
        lambda v, _: v < 45.,
        lambda v, w: (v * w, w), [x, y],
        return_same_structure=False)
    # ret = [x*y^2, y]

    # Note: This is simply d_ret[0]/d_x since d_ret[1]/d_x is 0.
    grad = gradients_impl.gradients(ret, [x])  # [2*x*y]
    with self.cached_session():
      self.assertSequenceEqual(self.evaluate(ret), [45., 3.])
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  @test_util.run_deprecated_v1
  def testMultipleLoopNonscalarCond(self):
    # [FUNCTION DOC]
    # Same loop as the basic two-variable test but the first variable is a one-by-one matrix, so the
    # loop condition is not a scalar. Asserts results forty-five and three and gradient nine.
    # Legacy graph mode only.
    x = constant_op.constant([[5.]])
    y = constant_op.constant(3.)

    # x = 5.
    # y = 3.
    # while x < 45.:
    #   x = x * y
    ret = while_loop_v2(
        lambda v, _: v < 45.,
        lambda v, w: (v * w, w), [x, y],
        return_same_structure=False)
    # ret == [x*y^2, y]

    # Note: This is simply d_ret[0]/d_x since d_ret[1]/d_x is 0.
    grad = gradients_impl.gradients(ret, [x])  # [2*x*y]
    with self.cached_session():
      self.assertSequenceEqual(self.evaluate(ret), [45., 3.])
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  @test_util.run_deprecated_v1
  def testMultipleLoopVars(self):
    # [FUNCTION DOC]
    # Runs a two-variable loop whose body updates the first as product and the second as sum. Computes
    # six gradients (each output and both together with respect to each input) and asserts the results
    # one hundred twenty and twenty-three, and gradients thirty-nine, four, forty-three, fifty-five,
    # six and sixty-one. Legacy graph mode only.
    x = constant_op.constant(5.)
    y = constant_op.constant(3.)

    # x = 5.
    # y = 3.
    # while x < 45.:
    #   x = x * y
    #   y = x + y
    ret = while_loop_v2(
        lambda v, _: v < 45.,
        lambda v, w: (v * w, v + w), [x, y],
        return_same_structure=False)
    # ret = [y*x**2 + x*y**2, x*y + x + y]

    gradx_0 = gradients_impl.gradients(ret[0], [x])  # [2*x*y + y**2]
    gradx_1 = gradients_impl.gradients(ret[1], [x])  # [y + 1]
    gradx_2 = gradients_impl.gradients(ret, [x])  # [2*x*y + y**2 + 2*y + 1]
    grady_0 = gradients_impl.gradients(ret[0], [y])  # [2*x*y + x**2]
    grady_1 = gradients_impl.gradients(ret[1], [y])  # [x + 1]
    grady_2 = gradients_impl.gradients(ret, [y])  # [2*x*y + x**2 + x + 1]
    with self.cached_session():
      self.assertSequenceEqual(self.evaluate(ret), [120., 23.])
      self.assertSequenceEqual(self.evaluate(gradx_0), [39.])
      self.assertSequenceEqual(self.evaluate(gradx_1), [4.])
      self.assertSequenceEqual(self.evaluate(gradx_2), [43.])
      self.assertSequenceEqual(self.evaluate(grady_0), [55.])
      self.assertSequenceEqual(self.evaluate(grady_1), [6.])
      self.assertSequenceEqual(self.evaluate(grady_2), [61.])

  @test_util.run_deprecated_v1
  def testGradientTape(self):
    # [FUNCTION DOC]
    # Runs a squaring loop under a gradient tape watching the input of two with exit bound four, then
    # asserts the tape gradient evaluated in a session equals four. Legacy graph mode only.
    with backprop.GradientTape() as t:
      x = constant_op.constant(2.)
      t.watch(x)
      ret = while_loop_v2(
          lambda v: v < 4., lambda v: v * v, [x],
          return_same_structure=False)  # x**2
    grad = t.gradient(ret, x)
    with self.cached_session() as sess:
      self.assertAllEqual(sess.run(grad), 4.0)

  @test_util.run_deprecated_v1
  def testMultipleWhileLoops(self):
    # [FUNCTION DOC]
    # Chains two squaring loops (the second consumes the first) so the result is the input to the
    # fourth power, then takes the gradient and the gradient of that gradient. Asserts values
    # thirty-two and forty-eight for input two. Legacy graph mode only.
    x = constant_op.constant(2.)
    ret1 = while_loop_v2(
        lambda v: v < 4., lambda v: v * v, [x],
        return_same_structure=False)  # x**2
    ret2 = while_loop_v2(
        lambda v: v < 16., lambda v: v * v, [ret1],
        return_same_structure=False)  # x**4
    grad = gradients_impl.gradients(ret2, [x])  # 4x**3
    grad_grad = gradients_impl.gradients(grad, [x])  # 12x**2
    with self.cached_session():
      self.assertSequenceEqual(self.evaluate(grad), [32.])
      self.assertSequenceEqual(self.evaluate(grad_grad), [48.])

  def testMultipleWhileLoopsWithFunc(self):
    # [FUNCTION DOC]
    # Builds, in a compiled function, two independent named loops. Gets the concrete function and
    # asserts both loops are of the stateless kind and have no control inputs, i.e. no ordering is
    # imposed between independent stateless loops.
    x = constant_op.constant(2.)

    @def_function.function
    def Fn():
      # [FUNCTION DOC]
      # HELPER: Compiled function for the test.
      # Builds two independent squaring loops named first and second over a captured constant and returns
      # both results. Takes no arguments.
      ret1 = while_loop_v2(
          lambda v: v < 4.,
          lambda v: v * v, [x],
          return_same_structure=False,
          name="while_1")  # x**2
      ret2 = while_loop_v2(
          lambda v: v < 16.,
          lambda v: v * v, [x],
          return_same_structure=False,
          name="while_2")  # x**4
      return ret1, ret2

    concrete_fn = Fn.get_concrete_function()
    while_1 = concrete_fn.graph.get_operation_by_name("while_1")
    while_2 = concrete_fn.graph.get_operation_by_name("while_2")
    self.assertEqual(while_1.type, "StatelessWhile")
    self.assertEqual(while_2.type, "StatelessWhile")
    self.assertEmpty(while_1.control_inputs)
    self.assertEmpty(while_2.control_inputs)

  def testMultipleWhileLoopsGradStateless(self):
    # [FUNCTION DOC]
    # Builds a compiled function with two loops and a gradient tape over their sum, then inspects the
    # function graph. Asserts that there are exactly four loop operations (two forward, two gradient),
    # all stateless, and that none has control inputs.

    @def_function.function
    def Fn():
      # [FUNCTION DOC]
      # HELPER: Compiled function for the test.
      # Creates a constant, runs two named squaring loops on it under a gradient tape, and returns the
      # gradient of their sum with respect to the constant.
      x = constant_op.constant(2.)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        ret1 = while_loop_v2(
            lambda v: v < 4.,
            lambda v: v * v, [x],
            return_same_structure=False,
            name="while_1")  # x**2
        ret2 = while_loop_v2(
            lambda v: v < 16.,
            lambda v: v * v, [x],
            return_same_structure=False,
            name="while_2")  # x**4
        loss = ret1 + ret2
      return tape.gradient(loss, x)

    graph = Fn.get_concrete_function().graph
    while_ops = [op for op in graph.get_operations() if "While" in op.type]
    self.assertAllEqual([op.type for op in while_ops], ["StatelessWhile"] * 4,
                        "Must have exactly 4 StatelessWhile ops.")
    for op in while_ops:
      self.assertEmpty(op.control_inputs,
                       "{} should not have any control inputs".format(op.name))

  def testMultipleWhileLoopsWithDeps(self):
    # [FUNCTION DOC]
    # Builds a compiled function with two loops whose bodies both assign to the same variable (a
    # stateful effect). Asserts both loops are the stateful kind, the first has no control inputs and
    # the second has exactly one control input, namely the first loop, so stateful ordering is kept.
    x = variables.Variable(2.)
    c = constant_op.constant(2.)

    @def_function.function
    def Fn():
      # [FUNCTION DOC]
      # HELPER: Compiled function for the test.
      # Defines two stateful loop bodies and runs two named loops over a captured constant, returning both.

      def Body1(v):
        # [FUNCTION DOC]
        # HELPER: First loop body.
        # Assigns the variable to itself (a stateful no-op) and returns the loop value times the variable.
        x.assign(x)
        return v * x

      ret1 = while_loop_v2(
          lambda v: v < 4.,
          Body1, [c],
          return_same_structure=False,
          name="while_1")  # 2x

      def Body2(v):
        # [FUNCTION DOC]
        # HELPER: Second loop body.
        # Assigns the variable to itself and returns the loop value times the variable squared.
        x.assign(x)
        return v * x * x

      ret2 = while_loop_v2(
          lambda v: v < 16.,
          Body2, [c],
          return_same_structure=False,
          name="while_2")  # 4x
      return ret1, ret2

    concrete_fn = Fn.get_concrete_function()
    while_1 = concrete_fn.graph.get_operation_by_name("while_1")
    while_2 = concrete_fn.graph.get_operation_by_name("while_2")
    self.assertEqual(while_1.type, "While")
    self.assertEqual(while_2.type, "While")
    self.assertEmpty(while_1.control_inputs)
    self.assertLen(while_2.control_inputs, 1)
    self.assertIs(while_2.control_inputs[0], while_1)

  def testMultipleWhileLoopsWithVarsDeps(self):
    # [FUNCTION DOC]
    # Builds a compiled function with four loops touching two different variables (two each) plus one
    # stateless loop. Asserts the four are stateful and the stateless one is stateless, that loops on
    # the same variable are chained by one control input each (second on first, fourth on third),
    # and that the first, third and stateless loops have no control inputs.
    x1 = variables.Variable(2.)
    x2 = variables.Variable(3.)
    c = constant_op.constant(2.)

    @def_function.function
    def Fn():
      # [FUNCTION DOC]
      # HELPER: Compiled function for the test.
      # Defines four stateful bodies (two per variable), runs four stateful loops and one stateless loop
      # over a captured constant, and returns all five results.

      def Body1(v):
        # [FUNCTION DOC]
        # HELPER: Body for the first loop. Assigns the first variable to itself and returns value times it.
        x1.assign(x1)
        return v * x1

      ret1 = while_loop_v2(
          lambda v: v < 4.,
          Body1, [c],
          return_same_structure=False,
          name="while_1")  # 2x

      def Body2(v):
        # [FUNCTION DOC]
        # HELPER: Body for the second loop. Assigns the first variable to itself and returns value times
        # its square.
        x1.assign(x1)
        return v * x1 * x1

      ret2 = while_loop_v2(
          lambda v: v < 16.,
          Body2, [c],
          return_same_structure=False,
          name="while_2")  # 4x

      def Body3(v):
        # [FUNCTION DOC]
        # HELPER: Body for the third loop. Assigns the second variable to itself and returns value times it.
        x2.assign(x2)
        return v * x2

      ret3 = while_loop_v2(
          lambda v: v < 4.,
          Body3, [c],
          return_same_structure=False,
          name="while_3")  # 3x

      def Body4(v):
        # [FUNCTION DOC]
        # HELPER: Body for the fourth loop. Assigns the second variable to itself and returns value times
        # its square.
        x2.assign(x2)
        return v * x2 * x2

      ret4 = while_loop_v2(
          lambda v: v < 16.,
          Body4, [c],
          return_same_structure=False,
          name="while_4")  # 9x
      ret5 = while_loop_v2(
          lambda v: v < 16.,
          lambda v: v * v, [c],
          return_same_structure=False,
          name="while_stateless")  # x**2
      return ret1, ret2, ret3, ret4, ret5

    concrete_fn = Fn.get_concrete_function()
    while_1 = concrete_fn.graph.get_operation_by_name("while_1")
    while_2 = concrete_fn.graph.get_operation_by_name("while_2")
    while_3 = concrete_fn.graph.get_operation_by_name("while_3")
    while_4 = concrete_fn.graph.get_operation_by_name("while_4")
    while_stateless = concrete_fn.graph.get_operation_by_name(
        "while_stateless")
    self.assertEqual(while_1.type, "While")
    self.assertEqual(while_2.type, "While")
    self.assertEqual(while_3.type, "While")
    self.assertEqual(while_4.type, "While")
    self.assertEqual(while_stateless.type, "StatelessWhile")
    self.assertEmpty(while_1.control_inputs)
    self.assertLen(while_2.control_inputs, 1)
    self.assertIs(while_2.control_inputs[0], while_1)
    self.assertEmpty(while_3.control_inputs)
    self.assertLen(while_4.control_inputs, 1)
    self.assertIs(while_4.control_inputs[0], while_3)
    self.assertEmpty(while_stateless.control_inputs)

  @test_util.run_deprecated_v1
  def testDoubleDerivative(self):
    # [FUNCTION DOC]
    # Runs a loop that squares a value (input two, bound eight), takes the gradient and then the
    # gradient of the gradient. Asserts results sixteen, thirty-two and forty-eight. Legacy graph mode.
    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8., lambda v: v**2, [x],
        return_same_structure=False)  # x**4
    grad = gradients_impl.gradients(ret, [x])  # 4x**3
    grad_grad = gradients_impl.gradients(grad, [x])  # 12x**2
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])
      self.assertSequenceEqual(self.evaluate(grad_grad), [48.])

  @test_util.run_v2_only
  def testMultipleWhileLoopsEager(self):
    # [FUNCTION DOC]
    # Second-version-only test: a compiled function builds two chained squaring loops and returns the
    # gradient and second gradient; asserts they are thirty-two and forty-eight.

    @def_function.function
    def Func():
      # [FUNCTION DOC]
      # HELPER: Compiled function for the test.
      # Builds chained squaring loops on a constant two and returns the first and second gradients.
      x = constant_op.constant(2.)
      ret1 = while_loop_v2(
          lambda v: v < 4., lambda v: v * v, [x],
          return_same_structure=False)  # x**2
      ret2 = while_loop_v2(
          lambda v: v < 16.,
          lambda v: v * v, [ret1],
          return_same_structure=False)  # x**4
      grad = gradients_impl.gradients(ret2, [x])[0]  # 4x**3
      grad_grad = gradients_impl.gradients(grad, [x])[0]  # 12x**2
      return grad, grad_grad

    grad, grad_grad = Func()
    self.assertEqual(grad.numpy(), 32.)
    self.assertEqual(grad_grad.numpy(), 48.)

  @test_util.run_v2_only
  def testDoubleDerivativeEager(self):
    # [FUNCTION DOC]
    # Second-version-only test: a compiled function squares a value in a loop and returns the result,
    # gradient and second gradient; asserts sixteen, thirty-two and forty-eight.

    @def_function.function
    def Func():
      # [FUNCTION DOC]
      # HELPER: Compiled function for the test.
      # Runs one squaring loop on a constant two and returns result, gradient and second gradient.
      x = constant_op.constant(2.)
      ret = while_loop_v2(
          lambda v: v < 8., lambda v: v**2, [x],
          return_same_structure=False)  # x**4
      grad = gradients_impl.gradients(ret, [x])[0]  # 4x**3
      grad_grad = gradients_impl.gradients(grad, [x])[0]  # 12x**2
      return ret, grad, grad_grad

    ret, grad, grad_grad = Func()
    self.assertEqual(ret.numpy(), 16.)
    self.assertEqual(grad.numpy(), 32.)
    self.assertEqual(grad_grad.numpy(), 48.)

  def _testPruning(self):
    # [FUNCTION DOC]
    # HELPER: Shared body for the V1 and V2 pruning tests.
    # Builds a loop that pushes values on a tensor list, registers the first output as the training
    # operation and optimizes the graph. Asserts the list is pruned away when unused (a specific count
    # of loop-entry nodes, none of variant type, no push-back nodes), then registers the stacked list as
    # used and asserts the list entry and push-back are kept. Modifies the graph's train-op collection.
    # Expected counts depend on whether control flow v2 is enabled.
    x = constant_op.constant(1)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=x.dtype, element_shape=x.shape)

    def Cond(x, tl):
      # [FUNCTION DOC]
      # HELPER: Loop condition. Takes the counter and the list (ignored); true while the counter is below five.
      del tl  # Unused for Cond.
      return x < 5

    def Body(x, tl):
      # [FUNCTION DOC]
      # HELPER: Loop body. Takes counter and list; returns counter plus one and the list with the counter pushed on.
      return x + 1, list_ops.tensor_list_push_back(tl, x)

    outputs = while_loop.while_loop(Cond, Body, [x, tensor_list])

    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(outputs[0])

    g = GetOptimizedGraph()
    # TODO(b/136034023): while_v2 adds an extra loop_counter which is not pruned
    # away, causing an extra Enter node.
    enter_count = 2 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 1
    self.assertLen([n for n in g.node if n.op == "Enter"], enter_count)
    # Test that the TensorList is pruned out.
    self.assertEmpty([
        n for n in g.node if n.op == "Enter" and
        n.attr["T"].type == dtypes.variant.as_datatype_enum
    ])
    self.assertEmpty([n for n in g.node if n.op == "TensorListPushBack"])

    stack = list_ops.tensor_list_stack(outputs[1], element_dtype=x.dtype)
    train_op.append(stack)
    g = GetOptimizedGraph()
    # TODO(b/136034023): while_v2 adds an extra loop_counter which is not pruned
    # away, causing an extra Enter node.
    enter_count = 3 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 2
    self.assertLen([n for n in g.node if n.op == "Enter"], enter_count)
    # Test that the TensorList is not pruned out.
    self.assertNotEmpty([
        n for n in g.node if n.op == "Enter" and
        n.attr["T"].type == dtypes.variant.as_datatype_enum
    ])
    self.assertNotEmpty([n for n in g.node if n.op == "TensorListPushBack"])

  @test_util.run_deprecated_v1
  def testPruningV1(self):
    # [FUNCTION DOC]
    # Runs the shared pruning check with the default (legacy) control flow. Legacy graph mode only.
    self._testPruning()

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  def testPruningV2(self):
    # [FUNCTION DOC]
    # Runs the shared pruning check with control flow v2 enabled. Legacy graph mode only.
    self._testPruning()

  def _testDoNotAccumulateInvariants(self):
    # [FUNCTION DOC]
    # HELPER: Shared body for the invariant-accumulation tests.
    # Chooses the expected push operation name by the active control flow version. Builds a loop of five
    # iterations multiplying by a captured constant, takes the gradient, registers it as the training
    # operation and optimizes the graph. Asserts exactly one push node exists: only the loop variable is
    # accumulated, the loop invariant is not. Modifies the train-op collection.
    push_op = ("TensorListPushBack"
               if control_flow_v2_toggles.control_flow_v2_enabled() else
               "StackPushV2")

    # Tests that loop invariants, i.e., tensors that are "captured" by the
    # while loop and not passed as loop variables are not accumulated in
    # gradient computation.
    v = constant_op.constant(5.0, name="v")

    r = while_loop.while_loop(
        lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)

    output = gradients_impl.gradients(r, v)[0]
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(output)

    g = GetOptimizedGraph()
    # The gradient for v * x requires the value of both v and x. Since v is a
    # loop invariant it is not accumulated so we have just one accumulator for
    # x.
    self.assertLen([n for n in g.node if n.op == push_op], 1)

  @test_util.run_deprecated_v1
  def testDoNotAccumulateInvariantsV1(self):
    # [FUNCTION DOC]
    # Runs the shared invariant-accumulation check with default control flow. Legacy graph mode only.
    self._testDoNotAccumulateInvariants()

  @test_util.run_deprecated_v1
  @test_util.enable_control_flow_v2
  def testDoNotAccumulateInvariantsV2(self):
    # [FUNCTION DOC]
    # Runs the shared invariant-accumulation check with control flow v2 enabled. Legacy graph mode only.
    self._testDoNotAccumulateInvariants()

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testPruningNested(self):
    # [FUNCTION DOC]
    # Builds an outer loop (bound twenty-five) whose body runs an inner loop (bound five) pushing on a
    # tensor list, with output of all intermediates enabled. Asserts first that, with only the counter
    # used, no variant-typed loop entries, push-back nodes or old-style loop nodes remain, then that
    # after stacking the list output the variant entries and push-back are present. The enter-count
    # checks are commented out upstream pending a bug fix.
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    x = constant_op.constant(0)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=x.dtype, element_shape=x.shape)

    def Cond(x, tl):
      # [FUNCTION DOC]
      # HELPER: Outer loop condition. Takes counter and list (ignored); true while counter is below
      # twenty-five.
      del tl  # Unused for Cond.
      return x < 25

    def Body(x, tl):
      # [FUNCTION DOC]
      # HELPER: Outer loop body.
      # Takes counter and list, runs the inner loop and returns the last two inner outputs (outer
      # counter and list).

      def InnerCond(inner_x, unused_outer_x, unused_tl):
        # [FUNCTION DOC]
        # HELPER: Inner loop condition. True while the inner counter is below five; other inputs ignored.
        return inner_x < 5

      def InnerBody(inner_x, outer_x, tl):
        # [FUNCTION DOC]
        # HELPER: Inner loop body.
        # Returns the inner counter plus one, the outer counter plus one, and the list with the captured
        # outer counter pushed on.
        return inner_x + 1, outer_x + 1, list_ops.tensor_list_push_back(tl, x)

      inner_x = constant_op.constant(0)
      return while_loop.while_loop(InnerCond, InnerBody, [inner_x, x, tl])[1:]

    outputs = while_loop.while_loop(Cond, Body, [x, tensor_list])

    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(outputs[0])

    g = GetOptimizedGraph()
    # TODO(b/136034023): while_v2 adds an extra loop_counter which is not pruned
    # away, causing an extra Enter node.
    # enter_count = 4 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 2
    # self.assertLen([n for n in g.node if n.op == "Enter"], enter_count)
    # Test that the TensorList is pruned out.
    self.assertEmpty([
        n for n in g.node if n.op == "Enter" and
        n.attr["T"].type == dtypes.variant.as_datatype_enum
    ])
    self.assertEmpty([n for n in g.node if n.op == "TensorListPushBack"])
    self.assertEmpty([n for n in g.node if n.op == "_While"])

    stack = list_ops.tensor_list_stack(outputs[1], element_dtype=x.dtype)
    train_op.append(stack)
    g = GetOptimizedGraph()
    # TODO(b/136034023): while_v2 adds an extra loop_counter which is not pruned
    # away, causing an extra Enter node.
    # enter_count = 3 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 2
    # self.assertLen([n for n in g.node if n.op == "Enter"], enter_count)
    # Test that the TensorList is not pruned out.
    self.assertNotEmpty([
        n for n in g.node if n.op == "Enter" and
        n.attr["T"].type == dtypes.variant.as_datatype_enum
    ])
    self.assertNotEmpty([n for n in g.node if n.op == "TensorListPushBack"])

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testPruningNested2(self):
    # [FUNCTION DOC]
    # Builds three nested loops (outer, mid, inner) where the mid body takes a gradient of the inner
    # loop, bounded by the size of a placeholder, with output of all intermediates enabled. Optimizes
    # the graph with the final output registered as the training operation and asserts exactly one
    # push-back node, i.e. only one accumulator is created.
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    v = constant_op.constant(5.0, name="v")

    p = array_ops.placeholder(dtype=dtypes.int32)

    def MidBodyBuilder(iterations):
      # [FUNCTION DOC]
      # HELPER: Builds the mid loop body for a given iteration bound tensor; returns the body function.

      def MidBody(i, x):
        # [FUNCTION DOC]
        # HELPER: Mid loop body.
        # Takes counter and value, runs an inner loop multiplying the captured constant, and returns
        # counter plus one together with the gradient of value plus the inner result with respect to the
        # captured constant.
        r = while_loop.while_loop(
            lambda *_: True,
            lambda i, x: (i + 1, math_ops.multiply(v, x, name="my_mul")),
            (0, x),
            maximum_iterations=iterations,
            name="inner")
        return (i + 1, gradients_impl.gradients(x + r[1], v)[0])

      return MidBody

    def OuterBody(i, x):
      # [FUNCTION DOC]
      # HELPER: Outer loop body.
      # Takes counter and value, computes the iteration bound from the placeholder's size and returns
      # counter plus one and value plus the second output of the mid loop.
      iterations = array_ops.size(p, name="iterations")
      return (i + 1, x + while_loop.while_loop(
          lambda *_: True,
          MidBodyBuilder(iterations), (0, x),
          maximum_iterations=iterations,
          name="mid")[1])

    def CreateWhileLoop():
      # [FUNCTION DOC]
      # HELPER: Builds the outer loop (five iterations) under a CPU device scope and returns an identity
      # of its second output.
      with ops.device("/cpu:0"):
        r = while_loop.while_loop(
            lambda *_: True,
            OuterBody, (0, 1.0),
            maximum_iterations=5,
            name="outer")
        return array_ops.identity(r[1])

    output = CreateWhileLoop()
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(output)

    g = GetOptimizedGraph()
    self.assertLen([n for n in g.node if n.op == "TensorListPushBack"], 1)

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testPruningNested3(self):
    # [FUNCTION DOC]
    # Builds a loop multiplying by a captured constant, takes the gradient with respect to the constant
    # with output of all intermediates enabled, optimizes the graph and asserts exactly one push-back
    # node.
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    v = constant_op.constant(5.0, name="v")

    def CreateWhileLoop():
      # [FUNCTION DOC]
      # HELPER: Builds a five-iteration loop multiplying by a captured constant and returns an identity of it.
      r = while_loop.while_loop(
          lambda _: True,
          lambda x: math_ops.multiply(v, x, name="my_mul"), [1.0],
          maximum_iterations=5,
          name="outer")
      return array_ops.identity(r)

    r = CreateWhileLoop()
    output = gradients_impl.gradients(r, v)[0]
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(output)

    g = GetOptimizedGraph()
    self.assertLen([n for n in g.node if n.op == "TensorListPushBack"], 1)

  def _assertNotAccumulated(self, while_op, index):
    # [FUNCTION DOC]
    # HELPER: Assertion helper for the accumulation tests.
    # Takes a loop operation and an input index. Fetches the loop body graph and asserts that none of the
    # consumers of that body input is a push-back operation (the input is not accumulated).
    """Asserts that `while_op` input at `index` is not accumulated."""
    body_graph = while_v2._get_graph(while_op, "body", "_body_graph")
    placeholder = body_graph.inputs[index]
    self.assertNotIn("TensorListPushBack",
                     [op.type for op in placeholder.consumers()])

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testDoNotOutputLoopCounterAsIntermediate(self):
    # [FUNCTION DOC]
    # With all-intermediates output enabled, builds a bounded loop and asserts the loop counter input
    # (index zero) of the loop operation is not accumulated.
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    v = constant_op.constant(5.0, name="v")
    r = while_loop.while_loop(
        lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)
    # Skip over Identity.
    while_op = r.op.inputs[0].op
    self._assertNotAccumulated(while_op, 0)

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testDoNotOutputLoopInvariantAsIntermediate(self):
    # [FUNCTION DOC]
    # With all-intermediates output enabled, builds a bounded loop capturing a constant, finds the input
    # index of that captured constant on the loop operation and asserts it is not accumulated.
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE

    def GetInputIndex(op, tensor):
      # [FUNCTION DOC]
      # HELPER: Finds an input by identity.
      # Takes an operation and a tensor, returns the index of the first input that is the same object, or
      # nothing (null) if absent. Needed because tensors are not hashable.
      for index, inp in enumerate(op.inputs):
        if inp is tensor:
          return index

    v = constant_op.constant(5.0, name="v")
    r = while_loop.while_loop(
        lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)
    # Skip over Identity.
    while_op = r.op.inputs[0].op
    # We can't directly use while_op.inputs.index() because Tensors are not
    # hashable.
    index = GetInputIndex(while_op, v)
    self._assertNotAccumulated(while_op, index)

  @test_util.run_deprecated_v1
  def testCaptureExternalTensorInCond(self):
    # [FUNCTION DOC]
    # Runs a loop whose condition uses an outside constant (value plus one below nine) and whose body
    # triples the value, starting at two. Asserts result eighteen and gradient nine. Legacy graph mode.
    x = constant_op.constant(2.)
    y = constant_op.constant(1.)
    ret = while_loop_v2(
        lambda v: v + y < 9.,
        lambda v: v * 3., [x],
        return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 18.)
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  @test_util.run_deprecated_v1
  def testCaptureExternalTensorInBody(self):
    # [FUNCTION DOC]
    # Runs a loop whose body multiplies by an outside constant three, starting at two with bound eight.
    # Asserts result eighteen and gradient nine. Legacy graph mode only.
    x = constant_op.constant(2.)
    y = constant_op.constant(3.)
    ret = while_loop_v2(
        lambda v: v < 8., lambda v: v * y, [x], return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 18.)
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  @test_util.run_deprecated_v1
  def testLoopWithTensorListPushBack(self):
    # [FUNCTION DOC]
    # Runs a loop that carries a tensor list and pushes two elements per iteration while squaring the
    # value (from two, bound five). Asserts result sixteen and gradient thirty-two. Legacy graph mode.
    x = constant_op.constant(2.)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=ScalarShape())

    def Cond(x, tl):
      # [FUNCTION DOC]
      # HELPER: Loop condition. Takes value and list (ignored); true while value is below five.
      del tl  # Unused for Cond.
      return x < 5.

    def Body(x, tl):
      # [FUNCTION DOC]
      # HELPER: Loop body. Pushes the value and then a constant one hundred on the list; returns the value
      # squared and the list.
      tl = list_ops.tensor_list_push_back(tl, x)
      tl = list_ops.tensor_list_push_back(tl, constant_op.constant(100.))
      return x**2., tl

    ret = while_loop_v2(
        Cond, Body, [x, tensor_list], return_same_structure=False)
    grad = gradients_impl.gradients(ret[0], x)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(ret[0]), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @test_util.run_deprecated_v1
  def testDuplicateAccumulator(self):
    # [FUNCTION DOC]
    # Runs a loop that already pushes its input on a tensor list. Finds the loop operation, looks at the
    # consumers of the value input in its body graph and asserts there is exactly one push-back (no
    # duplicate accumulator added). Then asserts result sixteen and gradient thirty-two. Legacy graph mode.
    x = constant_op.constant(2.)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=ScalarShape())

    def Cond(x, tl):
      # [FUNCTION DOC]
      # HELPER: Loop condition. Takes value and list (ignored); true while value is below five.
      del tl  # Unused for Cond.
      return x < 5.

    def Body(x, tl):
      # [FUNCTION DOC]
      # HELPER: Loop body. Pushes the value on the list (so a gradient accumulator already exists) and
      # returns the value squared and the list.
      # There is an accumulator in the loop already so we should not add
      # another.
      tl = list_ops.tensor_list_push_back(tl, x)
      return x**2., tl

    ret = while_loop_v2(
        Cond, Body, [x, tensor_list], return_same_structure=False)

    for op in ops.get_default_graph().get_operations():
      if op.type == "While" or op.type == "StatelessWhile":
        while_op = op

    body_graph = while_v2._get_graph(while_op, "body", "_body_graph")
    x_input_index = [i for i, inp in enumerate(while_op.inputs) if inp == x][0]
    x_input_t = body_graph.inputs[x_input_index]
    accumulator_count = len(
        [c for c in x_input_t.consumers() if c.type == "TensorListPushBack"])
    self.assertEqual(accumulator_count, 1)

    grad = gradients_impl.gradients(ret[0], x)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(ret[0]), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @parameterized.named_parameters(
      ("UnknownShape", None),
      ("PartiallyDefinedShape", [None, 2]),
      ("FullyDefinedShape", [1, 2]),
  )
  @test_util.run_deprecated_v1
  def testAccumulatorElementShape(self, shape):
    # [FUNCTION DOC]
    # Parameterized over unknown, partially known and fully known shapes. Builds a loop with two
    # placeholders of that shape, takes first and second derivatives, then pops the accumulated lists of
    # the forward and gradient loops and asserts the popped element shape matches the parameter
    # (unknown has no dimensions; otherwise the lists are equal). Legacy graph mode only.

    def MatchShape(actual_tensor_shape):
      # [FUNCTION DOC]
      # HELPER: Shape assertion. Takes a tensor shape; if the parameter is null asserts the shape has no
      # dimension info, otherwise asserts its list equals the parameter. Compares as lists because
      # dimension equality is not reliable with unknown sizes.
      # Compare the shapes, treating None dimensions as equal. We do not
      # directly check actual_tensor_shape and tf.TensorShape(shape) for
      # equality because tf.Dimension.__eq__ returns None if either dimension is
      # None.
      if shape is None:
        self.assertIsNone(actual_tensor_shape.dims)
      else:
        self.assertListEqual(actual_tensor_shape.as_list(), shape)

    def GetAccumulatorForInputAtIndex(while_op, idx):
      # [FUNCTION DOC]
      # HELPER: Takes a loop operation and input index; in its body graph finds the first push-back
      # consuming that input and returns the loop output carrying the accumulated list.
      body_graph = while_v2._get_graph(while_op, "body", "_body_graph")
      y_input_t = body_graph.inputs[idx]
      push_back_node = [c for c in y_input_t.consumers()
                        if c.type == "TensorListPushBack"][0]
      output_idx = body_graph.outputs.index(push_back_node.outputs[0])
      return while_op.outputs[output_idx]

    x = array_ops.placeholder(dtype=dtypes.float32, shape=shape)
    y = array_ops.placeholder(dtype=dtypes.float32, shape=shape)

    # Forward pass.
    ret = while_loop_v2(lambda v, u: v < 8.,
                        lambda v, u: (math_ops.pow(v, u), u),
                        [x, y],
                        return_same_structure=True)
    while_op = ret[0].op.inputs[0].op
    # Gradient pass.
    grad = gradients_impl.gradients(ret[0], x)
    # Note: There is an Identity b/w grad[0] and the While op.
    grad_while_op = grad[0].op.inputs[0].op

    # Get the TensorList output of While op containing the accumulated values
    # of y.
    x_input_index = [i for i, inp in enumerate(while_op.inputs) if x == inp][0]
    output = GetAccumulatorForInputAtIndex(while_op, x_input_index)
    _, val = list_ops.tensor_list_pop_back(output,
                                           element_dtype=dtypes.float32)
    MatchShape(val.shape)

    # Take second derivative to generate intermediate grad_while_op outputs
    gradients_impl.gradients(grad, x)

    # Get the TensorList output of gradient While op containing the accumulated
    # values of grad_x (note that grad_x is needed by the second derivative).
    # grad_while_op.inputs:
    grad_output_index = grad_while_op.outputs.index(grad[0].op.inputs[0])
    grad_output = GetAccumulatorForInputAtIndex(grad_while_op,
                                                grad_output_index)
    _, val = list_ops.tensor_list_pop_back(grad_output,
                                           element_dtype=dtypes.float32)
    MatchShape(val.shape)

  def _createWhile(self, name):
    # [FUNCTION DOC]
    # HELPER: Builds the loop used by the default-name test.
    # KNOWN ISSUE: the name argument is accepted but never used; the loop is always created with the
    # default name, so callers passing empty or null are not testing different names.
    # Builds a three-iteration counter loop, asserts the underlying operation is of the stateless kind
    # and returns that operation.
    """Helper function testDefaultName."""
    output = while_v2.while_loop(
        lambda i: i < 3,
        lambda i: i + 1, [constant_op.constant(0)],
        return_same_structure=False)
    while_op = output.op.inputs[0].op
    self.assertEqual(while_op.type, "StatelessWhile")
    return while_op

  def testDefaultName(self):
    # [FUNCTION DOC]
    # In fresh graphs, builds loops and checks default naming: top-level name is the plain loop name,
    # the condition and body functions get names with the loop prefix and numbering; under a name scope
    # the operation name is prefixed by the scope and a second loop gets a numeric suffix with matching
    # function names.
    with ops.Graph().as_default():
      while_op = self._createWhile(None)
      self.assertEqual(while_op.name, "while")
      self.assertRegex(while_op.get_attr("cond").name, r"while_cond_\d*")
      self.assertRegex(while_op.get_attr("body").name, r"while_body_\d*")

    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        while1_op = self._createWhile("")
        self.assertEqual(while1_op.name, "foo/while")
        self.assertRegex(while1_op.get_attr("cond").name, r"foo_while_cond_\d*")
        self.assertRegex(while1_op.get_attr("body").name, r"foo_while_body_\d*")

        while2_op = self._createWhile(None)
        self.assertEqual(while2_op.name, "foo/while_1")
        self.assertRegex(
            while2_op.get_attr("cond").name, r"foo_while_1_cond_\d*")
        self.assertRegex(
            while2_op.get_attr("body").name, r"foo_while_1_body_\d*")

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  def testWhileAndTensorArray(self):
    # [FUNCTION DOC]
    # With control flow v2 enabled, applies a map function (which uses tensor arrays inside a loop)
    # multiplying elements by a parameter, and asserts the output is doubled and the gradient with
    # respect to the parameter is twenty-one. Legacy graph mode only.
    param = constant_op.constant(2.0)
    y0 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")
    # map_fn uses TensorArray internally.
    r = map_fn.map_fn(lambda x: math_ops.multiply(x, param), y0)
    grad = gradients_impl.gradients(r, param)[0]
    self.assertAllClose([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], self.evaluate(r))
    self.assertAllClose(21.0, self.evaluate(grad))

  @test_util.run_deprecated_v1
  def testNestedWhile(self):
    # [FUNCTION DOC]
    # Computes a geometric sum with an outer loop and an inner loop computing powers. Asserts the
    # result is three hundred sixty-four and the gradient with respect to the base is five hundred
    # forty-seven. Legacy graph mode only.
    # Compute sum of geometric progression: n^0 + n^1 + ... + n^m
    # We compute the pow using a while loop.
    n = constant_op.constant(3.)
    m = constant_op.constant(5.)
    sum_of_powers = constant_op.constant(0.)

    def Body(i, previous_sum):
      # [FUNCTION DOC]
      # HELPER: Outer loop body. Takes counter and running sum; computes a power with an inner loop and
      # returns counter minus one and the sum plus the power.
      prod = constant_op.constant(1.)
      return i - 1., previous_sum + while_loop_v2(
          lambda c, _: c > 0,
          lambda c, v: (c - 1., v * n), [i, prod],
          return_same_structure=False)[1]

    result = while_loop_v2(
        lambda i, _: i >= 0,
        Body, [m, sum_of_powers],
        return_same_structure=False)[1]
    grad = gradients_impl.gradients(result, [n])
    self.assertEqual(self.evaluate(result), 364.)
    self.assertSequenceEqual(self.evaluate(grad), [547.])

  @test_util.run_deprecated_v1
  def testNestedWhileWithLegacyDefun(self):
    # [FUNCTION DOC]
    # Same geometric-sum check as the nested test but the inner body is a legacy function-decorated
    # function whose output shapes are set to scalar. Asserts result three hundred sixty-four and
    # gradient five hundred forty-seven. Legacy graph mode only.
    n = constant_op.constant(3.)
    m = constant_op.constant(5.)
    sum_of_powers = constant_op.constant(0.)

    def Body(i, previous_sum):
      # [FUNCTION DOC]
      # HELPER: Outer loop body. Runs the inner loop using the wrapper and returns counter minus one and the
      # sum plus the inner result.
      prod = constant_op.constant(1.)

      def InnerBodyWrapper(c, v):
        # [FUNCTION DOC]
        # HELPER: Calls the legacy function on counter and value, sets both result shapes to scalar and
        # returns the results.

        @function.Defun(dtypes.float32, dtypes.float32)
        def InnerBody(c, v):
          # [FUNCTION DOC]
          # HELPER: Legacy-defined function of two 32-bit floats; returns counter minus one and value times the
          # captured base.
          return c - 1., v * n

        results = InnerBody(c, v)
        results[0].set_shape([])
        results[1].set_shape([])
        return results

      return i - 1., previous_sum + while_loop_v2(
          lambda c, _: c > 0,
          InnerBodyWrapper, [i, prod],
          return_same_structure=False)[1]

    result = while_loop_v2(
        lambda i, _: i >= 0,
        Body, [m, sum_of_powers],
        return_same_structure=False)[1]
    grad = gradients_impl.gradients(result, [n])
    self.assertEqual(self.evaluate(result), 364.)
    self.assertSequenceEqual(self.evaluate(grad), [547.])

  @test_util.run_deprecated_v1
  def testIdentityNodeInBody(self):
    # [FUNCTION DOC]
    # Runs a squaring loop whose body first passes the value through two identity operations. Asserts
    # result sixteen and gradient thirty-two. Legacy graph mode only.

    def Body(v):
      # [FUNCTION DOC]
      # HELPER: Loop body: applies identity twice and returns the value squared.
      v = array_ops.identity(v)
      v = array_ops.identity(v)
      return v * v

    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8., Body, [x], return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    self.assertEqual(self.evaluate(ret), 16.)
    self.assertSequenceEqual(self.evaluate(grad), [32.])

  @test_util.run_deprecated_v1
  def testForwardPassRewrite(self):
    # [FUNCTION DOC]
    # Builds a doubling loop and asserts it is stateless with three outputs. After the first gradient
    # computation the loop is rewritten to output the accumulated value, so four outputs are expected;
    # a second gradient computation must not rewrite it again (still four). Legacy graph mode only.
    x = constant_op.constant(1.0, name="x")
    output = while_v2.while_loop(lambda x: x < 10.0,
                                 lambda x: x * 2.0,
                                 [x])[0]
    while_op = output.op.inputs[0].op
    self.assertEqual(while_op.type, "StatelessWhile")
    # outputs = [loop_counter, max_iters, x]
    self.assertLen(while_op.outputs, 3)

    gradients_impl.gradients(output, x)
    # while_op should have been rewritten to output intermediates.
    # outputs = [loop_counter, max_iters, x, x_accumulator]
    self.assertLen(while_op.outputs, 4)

    gradients_impl.gradients(output, x)
    # Computing the gradient again shouldn't rewrite while_op again.
    self.assertLen(while_op.outputs, 4)

  @parameterized.named_parameters(
      ("RandomUniform", random_ops.random_uniform, [5, 3]),
      ("RandomNormal", random_ops.random_normal, [5, 3]),
      ("ParameterizedTruncatedNormal",
       random_ops.parameterized_truncated_normal, [5, 3]),
      ("TruncatedNormal", random_ops.truncated_normal, [5, 3]),
      ("RandomGamma", random_gamma, [5, 3]),
      ("RandomPoissonV2", random_poisson_v2, [5, 3]),
      ("RandomGammaWithAlphaBeta", random_gamma_with_alpha_beta, [5, 3, 4, 2]),
      ("RandomPoissonV2WithLam", random_poisson_v2_with_lam, [5, 3, 2]),
  )
  @test_util.run_deprecated_v1
  def testRandomOpsShape(self, random_fn, expected_shape):
    # [FUNCTION DOC]
    # Parameterized over random-op factories and the shape each should produce. Builds a three-iteration
    # loop whose body generates random values with the five-prefixed shape and asserts the static shape
    # equals the expected one. Legacy graph mode only.
    shape = constant_op.constant([3])

    def Body(i, u):
      # [FUNCTION DOC]
      # HELPER: Loop body. Takes counter and value; extends the shape with five, generates random values
      # with the parameterized function, asserts the static shape equals the expected shape and returns the
      # counter plus one and the new value.
      shape_extended = array_ops.concat([[5], shape], axis=0)
      u = random_fn(shape_extended)
      assert u.shape.as_list() == expected_shape, str(u.shape.as_list())
      return i + 1, u

    _, _ = while_loop_v2(
        cond=lambda i, _: i < 3,
        body=Body,
        loop_vars=[
            0,
            array_ops.zeros(expected_shape, dtype=dtypes.float32),
        ])

  @test_util.run_deprecated_v1
  def testReshapeShape(self):
    # [FUNCTION DOC]
    # Builds a three-iteration loop whose body flattens a value and reshapes it back to a dynamic
    # shape, asserting the static shapes are sixty and five-by-three-by-four. Legacy graph mode only.
    shape = constant_op.constant([3, 4])

    def Body(i, u):
      # [FUNCTION DOC]
      # HELPER: Loop body. Flattens the value, asserts static shape sixty, reshapes it with a shape
      # tensor, asserts static shape five by three by four and returns counter plus one and the value.
      shape_extended = array_ops.concat([[5], shape], axis=0)
      u = array_ops.reshape(u, [-1])
      assert u.shape.as_list() == [60], str(u.shape.as_list())
      u = array_ops.reshape(u, shape_extended)
      assert u.shape.as_list() == [5, 3, 4], str(u.shape.as_list())
      return i + 1, u

    _, _ = while_loop_v2(
        cond=lambda i, _: i < 3,
        body=Body,
        loop_vars=[
            0,
            array_ops.zeros([5, 3, 4], dtype=dtypes.float32),
        ])

  @parameterized.named_parameters(
      ("Zeros", array_ops.zeros),
      ("Ones", array_ops.ones),
      ("Fill", fill),
  )
  @test_util.run_deprecated_v1
  def testFillOpsShape(self, fill_fn):
    # [FUNCTION DOC]
    # Parameterized over zeros, ones and fill. Builds a loop whose body creates a value with a dynamic
    # shape and asserts the static shape is five by three by four. Legacy graph mode only.
    shape = constant_op.constant([3, 4])

    def Body(i, u):
      # [FUNCTION DOC]
      # HELPER: Loop body. Builds the shape from a constant, calls the parameterized fill function,
      # asserts static shape five by three by four, and returns counter plus one and the value.
      shape_extended = array_ops.concat([[5], shape], axis=0)
      u = fill_fn(shape_extended)
      assert u.shape.as_list() == [5, 3, 4], str(u.shape.as_list())
      return i + 1, u

    _, _ = while_loop_v2(
        cond=lambda i, _: i < 3,
        body=Body,
        loop_vars=[
            0,
            array_ops.zeros([5, 3, 4], dtype=dtypes.float32),
        ])

  @test_util.run_deprecated_v1
  def testExternalColocationGrad(self):
    # [FUNCTION DOC]
    # Builds a loop whose body colocates with an outside tensor and checks the result is sixteen and the
    # gradient is thirty-two (colocation with an external tensor must not break gradients). Legacy graph.
    external_t = constant_op.constant(2.)
    v0 = constant_op.constant(2.)

    def Body(v):
      # [FUNCTION DOC]
      # HELPER: Loop body: squares the value under colocation with the outside constant.
      with ops.colocate_with(external_t):
        return v * v

    ret = while_loop_v2(lambda v: v < 8., Body, [v0])[0]
    grad = gradients_impl.gradients(ret, [v0])[0]
    self.assertAllEqual(ret, 16.)
    self.assertAllEqual(grad, 32.)

  @test_util.run_deprecated_v1
  def testDoNotAccumulateConstNodes(self):
    # [FUNCTION DOC]
    # Builds a loop doubling the value, computes the gradient (which rewrites the forward loop as a side
    # effect) and counts push-back operations in the forward body graph. Asserts exactly one, since the
    # constant operand needs no accumulator. Legacy graph mode only.

    def Body(v):
      # [FUNCTION DOC]
      # HELPER: Loop body: multiplies the value by the constant two.
      return v * 2.0

    v0 = constant_op.constant(2.)
    ret = while_loop_v2(lambda v: v < 8., Body, [v0])[0]
    # Gradients computation has the side-effect of updating the forward op
    # which is what we want to test.
    unused_grad = gradients_impl.gradients(ret, [v0])[0]
    # ret is separated from the `While` op by an `Identity` so we skip over
    # that.
    forward_while_op = ret.op.inputs[0].op
    body_graph = while_v2._get_graph(forward_while_op, "body", "_body_graph")
    push_back_nodes = [
        o for o in body_graph.get_operations() if o.type == "TensorListPushBack"
    ]
    # Gradient of `Mul` requires accumulating both its inputs. But since one
    # of those is a Const (2.0), we should have just one accumulator.
    self.assertLen(push_back_nodes, 1)

  def testDoNotAccumulateForwardTensorsForReductionOps(self):
    # [FUNCTION DOC]
    # Runs, in a compiled function, a loop under a tape whose body uses a custom-gradient function. In
    # its gradient function, shape, rank and size of the forward value must belong to the forward graph
    # (not accumulated) while created zeros belong to the gradient graph. Assertions run during tracing;
    # the test only calls the function and asserts nothing else.

    @def_function.function
    def Fn():
      # [FUNCTION DOC]
      # HELPER: Compiled function. Watches a constant on a tape, runs a two-iteration loop and returns the
      # gradient of the result.
      with backprop.GradientTape() as tape:
        x = constant_op.constant(2.)
        tape.watch(x)

        def Body(i, x):
          # [FUNCTION DOC]
          # HELPER: Loop body. Records the forward graph and returns counter plus one and the value passed
          # through the custom-gradient function.
          forward_graph = ops.get_default_graph()

          @custom_gradient.custom_gradient
          def SquaredWithZeroGrad(x):
            # [FUNCTION DOC]
            # HELPER: Custom-gradient function. Returns twice the input with a gradient function returning zeros.

            def Grad(unused_g, variables=None):  # pylint: disable=redefined-outer-name
              # [FUNCTION DOC]
              # HELPER: Gradient function. Takes the upstream gradient (unused) and optional variables (deleted).
              # Asserts the shape, rank and size tensors were created in the forward graph and the zeros in the
              # current gradient graph; returns zeros of the input's shape.
              del variables
              gradient_graph = ops.get_default_graph()
              shape = gen_array_ops.shape(x)
              assert shape.graph is forward_graph
              rank = gen_array_ops.rank(x)
              assert rank.graph is forward_graph
              size = gen_array_ops.size(x)
              assert size.graph is forward_graph
              zeros = array_ops.zeros(shape)
              assert zeros.graph is gradient_graph
              return zeros

            return x * 2, Grad

          return i + 1, SquaredWithZeroGrad(x)

        _, result = while_loop_v2(lambda i, _: i < 2, Body, [0, x])
      grad = tape.gradient(result, x)
      return grad

    Fn()

  def testDoNotAccumulateForwardTensorsForTensorListReductionOps(self):
    # [FUNCTION DOC]
    # Like the previous test but the loop variable is a tensor list. In the gradient function the
    # element shape and length of the forward list must belong to the forward graph while the reserved
    # list belongs to the gradient graph. Differentiates with ones as output gradients; only calls the
    # compiled function.

    @def_function.function
    def Fn():
      # [FUNCTION DOC]
      # HELPER: Compiled function. Builds a one-element tensor list on a tape, runs a two-iteration loop and
      # returns the gradient with respect to the list using a list of ones as output gradient.
      with backprop.GradientTape() as tape:
        e = constant_op.constant(2.)
        x = list_ops.empty_tensor_list(
            element_dtype=dtypes.float32, element_shape=e.shape)
        x = list_ops.tensor_list_push_back(x, e)
        tape.watch(x)

        def Body(i, x):
          # [FUNCTION DOC]
          # HELPER: Loop body. Records the forward graph and returns counter plus one and the list passed
          # through the custom-gradient function.
          forward_graph = ops.get_default_graph()

          @custom_gradient.custom_gradient
          def IdentityWithZeroGrad(x):
            # [FUNCTION DOC]
            # HELPER: Custom-gradient function. Returns the list unchanged with a gradient function returning an
            # empty reserved list.

            def Grad(unused_g, variables=None):  # pylint: disable=redefined-outer-name
              # [FUNCTION DOC]
              # HELPER: Gradient function. Ignores upstream gradient and variables; asserts element shape and length
              # tensors come from the forward graph and the reserved list from the gradient graph; returns the list.
              del variables
              gradient_graph = ops.get_default_graph()
              shape = gen_list_ops.tensor_list_element_shape(
                  x, shape_type=dtypes.int32)
              assert shape.graph is forward_graph
              size = gen_list_ops.tensor_list_length(x)
              assert size.graph is forward_graph
              zeros = gen_list_ops.tensor_list_reserve(shape, size,
                                                       dtypes.float32)
              assert zeros.graph is gradient_graph
              return zeros

            return x, Grad

          return i + 1, IdentityWithZeroGrad(x)

        _, result = while_loop_v2(lambda i, _: i < 2, Body, [0, x])
      ones_like = list_ops.tensor_list_from_tensor(
          array_ops.ones_like(
              list_ops.tensor_list_stack(result, element_dtype=dtypes.float32)),
          element_shape=tensor_shape.TensorShape([]))
      grad = tape.gradient(result, x, output_gradients=[ones_like])
      return grad

    Fn()

  @test_util.run_v2_only
  def testInheritParentNameScope(self):
    # [FUNCTION DOC]
    # Second-version-only test: in a compiled function under a name scope, runs a loop whose condition
    # and body assert the active name scope is nested under the loop name (foo/while/cond and
    # foo/while/body). Assertions run at trace time.

    @def_function.function
    def F():
      # [FUNCTION DOC]
      # HELPER: Compiled function. Opens a name scope, defines condition and body and runs a loop.
      with ops.name_scope("foo"):

        def Cond(unused_i):
          # [FUNCTION DOC]
          # HELPER: Loop condition. Opens a nested scope, asserts the full scope is the expected one and
          # returns false.
          with ops.name_scope("cond"):
            actual_name_scope = ops.get_name_scope()
            expected_name_scope = "foo/while/cond"
            assert actual_name_scope == expected_name_scope, (
                "%s does not match %s" %
                (actual_name_scope, expected_name_scope))
          return False

        def Body(i):
          # [FUNCTION DOC]
          # HELPER: Loop body. Opens a nested scope, asserts the full scope is the expected one and returns the
          # counter unchanged.
          with ops.name_scope("body"):
            actual_name_scope = ops.get_name_scope()
            expected_name_scope = "foo/while/body"
            assert actual_name_scope == expected_name_scope, (
                "%s does not match %s" %
                (actual_name_scope, expected_name_scope))
          return i

        return while_v2.while_loop(Cond, Body, [0.])

    F()

  @test_util.run_deprecated_v1  # Need to pass RunMetadata.
  def testDisableLowering(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the module-level switch for disabling lowering is restored only after the
    # assertions; a failure before that leaves the flag set for later tests (no try/finally).
    # Sets the module-level switch disabling lowering with switch and merge, runs a squaring loop with
    # full tracing metadata, asserts result sixteen and that no executed node name contains switch, then
    # restores the switch. Legacy graph mode only.
    old = control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE
    control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE = True
    with self.session() as sess:
      x = constant_op.constant(2.)
      ret = while_loop_v2(
          lambda v: v < 8., lambda v: v * v, [x], return_same_structure=False)

      opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      self.assertEqual(sess.run(ret, options=opts, run_metadata=run_metadata),
                       16)
      for dev_stat in run_metadata.step_stats.dev_stats:
        for ns in dev_stat.node_stats:
          self.assertNotIn("switch", ns.node_name)
    control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE = old

  def _runBasicWithConfig(self, config):
    # [FUNCTION DOC]
    # HELPER: Shared body for configuration tests. Takes a session config; builds on CPU a loop counting a
    # value to one thousand and asserts a session using that config returns one thousand.
    with ops.device("/cpu:0"):
      x = constant_op.constant(0)
      ret, = while_loop_v2(lambda x: x < 1000, lambda x: x + 1, [x])
    with self.cached_session(config=config):
      self.assertEqual(1000, self.evaluate(ret))

  @test_util.run_deprecated_v1
  def testRunKernelsInline(self):
    # [FUNCTION DOC]
    # Builds a config with a negative inter-op thread setting (run kernels inline) and runs the shared
    # basic loop check. Legacy graph mode only.
    config = config_pb2.ConfigProto()
    config.inter_op_parallelism_threads = -1
    self._runBasicWithConfig(config)

  @test_util.run_deprecated_v1
  def testSingleThreadedExecution(self):
    # [FUNCTION DOC]
    # Builds a config selecting the single-threaded executor and runs the shared basic loop check.
    # Legacy graph mode only.
    config = config_pb2.ConfigProto()
    config.experimental.executor_type = "SINGLE_THREADED_EXECUTOR"
    self._runBasicWithConfig(config)

  def testIsControlFlowGraph(self):
    # [FUNCTION DOC]
    # In a compiled function, runs a two-iteration loop whose condition and body assert their graphs are
    # flagged as control-flow graphs. Calls it with zero and asserts the result is two.
    x = constant_op.constant(0)

    @def_function.function
    def F(c):
      # [FUNCTION DOC]
      # HELPER: Compiled function taking the initial counter; defines condition and body and returns the
      # loop result.

      def Cond(i):
        # [FUNCTION DOC]
        # HELPER: Loop condition. Asserts the graph is a control-flow graph and returns true while below two.
        self.assertTrue(i.graph.is_control_flow_graph)
        return i < 2

      def Body(i):
        # [FUNCTION DOC]
        # HELPER: Loop body. Adds one, asserts the graph is a control-flow graph and returns the value.
        i = i + 1
        self.assertTrue(i.graph.is_control_flow_graph)
        return i

      return while_loop_v2(Cond, Body, [c])

    ret, = F(x)
    self.assertEqual(2, self.evaluate(ret))

  def testImportFromSerializedWithFunctionInBody(self):
    # [FUNCTION DOC]
    # Parses an embedded text-format serialized graph (a loop whose body calls a compiled function
    # doubling its input), imports it in a compiled function and takes the gradient of the loop output
    # with respect to the imported constant. Asserts the call returns eight.
    serialized = """node {
      name: "Const"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_FLOAT
            tensor_shape {
            }
            float_val: 1.0
          }
        }
      }
    }
    node {
      name: "while/maximum_iterations"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: -1
          }
        }
      }
    }
    node {
      name: "while/loop_counter"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 0
          }
        }
      }
    }
    node {
      name: "while"
      op: "StatelessWhile"
      input: "while/loop_counter"
      input: "while/maximum_iterations"
      input: "Const"
      attr {
        key: "T"
        value {
          list {
            type: DT_INT32
            type: DT_INT32
            type: DT_FLOAT
          }
        }
      }
      attr {
        key: "_lower_using_switch_merge"
        value {
          b: true
        }
      }
      attr {
        key: "_num_original_outputs"
        value {
          i: 3
        }
      }
      attr {
        key: "_read_only_resource_inputs"
        value {
          list {
          }
        }
      }
      attr {
        key: "body"
        value {
          func {
            name: "while_body_822"
          }
        }
      }
      attr {
        key: "cond"
        value {
          func {
            name: "while_cond_821"
          }
        }
      }
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
            }
            shape {
            }
            shape {
            }
          }
        }
      }
      attr {
        key: "parallel_iterations"
        value {
          i: 10
        }
      }
    }
    node {
      name: "while/Identity"
      op: "Identity"
      input: "while"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node {
      name: "while/Identity_1"
      op: "Identity"
      input: "while:1"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node {
      name: "while/Identity_2"
      op: "Identity"
      input: "while:2"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    library {
      function {
        signature {
          name: "while_body_822"
          input_arg {
            name: "while_loop_counter"
            type: DT_INT32
          }
          input_arg {
            name: "while_maximum_iterations_0"
            type: DT_INT32
          }
          input_arg {
            name: "placeholder"
            type: DT_FLOAT
          }
          output_arg {
            name: "add"
            type: DT_INT32
          }
          output_arg {
            name: "while_maximum_iterations"
            type: DT_INT32
          }
          output_arg {
            name: "partitionedcall"
            type: DT_FLOAT
          }
        }
        node_def {
          name: "PartitionedCall"
          op: "PartitionedCall"
          input: "placeholder"
          attr {
            key: "Tin"
            value {
              list {
                type: DT_FLOAT
              }
            }
          }
          attr {
            key: "Tout"
            value {
              list {
                type: DT_FLOAT
              }
            }
          }
          attr {
            key: "_collective_manager_ids"
            value {
              list {
              }
            }
          }
          attr {
            key: "_read_only_resource_inputs"
            value {
              list {
              }
            }
          }
          attr {
            key: "config"
            value {
              s: ""
            }
          }
          attr {
            key: "config_proto"
            value {
              s: ""
            }
          }
          attr {
            key: "executor_type"
            value {
              s: ""
            }
          }
          attr {
            key: "f"
            value {
              func {
                name: "__inference_f_841"
              }
            }
          }
          experimental_debug_info {
            original_node_names: "PartitionedCall"
          }
        }
        node_def {
          name: "add/y"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_INT32
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape {
                }
                int_val: 1
              }
            }
          }
          experimental_debug_info {
            original_node_names: "add/y"
          }
        }
        node_def {
          name: "add_0"
          op: "AddV2"
          input: "while_loop_counter"
          input: "add/y:output:0"
          attr {
            key: "T"
            value {
              type: DT_INT32
            }
          }
          experimental_debug_info {
            original_node_names: "add"
          }
        }
        ret {
          key: "add"
          value: "add_0:z:0"
        }
        ret {
          key: "partitionedcall"
          value: "PartitionedCall:output:0"
        }
        ret {
          key: "while_maximum_iterations"
          value: "while_maximum_iterations_0"
        }
        arg_attr {
          key: 0
          value {
            attr {
              key: "_output_shapes"
              value {
                list {
                  shape {
                  }
                }
              }
            }
          }
        }
        arg_attr {
          key: 1
          value {
            attr {
              key: "_output_shapes"
              value {
                list {
                  shape {
                  }
                }
              }
            }
          }
        }
        arg_attr {
          key: 2
          value {
            attr {
              key: "_output_shapes"
              value {
                list {
                  shape {
                  }
                }
              }
            }
          }
        }
      }
      function {
        signature {
          name: "while_cond_821"
          input_arg {
            name: "while_loop_counter"
            type: DT_INT32
          }
          input_arg {
            name: "while_maximum_iterations"
            type: DT_INT32
          }
          input_arg {
            name: "placeholder"
            type: DT_FLOAT
          }
          output_arg {
            name: "less"
            type: DT_BOOL
          }
        }
        node_def {
          name: "Less/y"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_FLOAT
                tensor_shape {
                }
                float_val: 5.0
              }
            }
          }
          experimental_debug_info {
            original_node_names: "Less/y"
          }
        }
        node_def {
          name: "Less"
          op: "Less"
          input: "placeholder"
          input: "Less/y:output:0"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
          experimental_debug_info {
            original_node_names: "Less"
          }
        }
        ret {
          key: "less"
          value: "Less:z:0"
        }
        arg_attr {
          key: 0
          value {
            attr {
              key: "_output_shapes"
              value {
                list {
                  shape {
                  }
                }
              }
            }
          }
        }
        arg_attr {
          key: 1
          value {
            attr {
              key: "_output_shapes"
              value {
                list {
                  shape {
                  }
                }
              }
            }
          }
        }
        arg_attr {
          key: 2
          value {
            attr {
              key: "_output_shapes"
              value {
                list {
                  shape {
                  }
                }
              }
            }
          }
        }
      }
      function {
        signature {
          name: "__inference_f_841"
          input_arg {
            name: "mul_placeholder"
            type: DT_FLOAT
          }
          output_arg {
            name: "identity"
            type: DT_FLOAT
          }
        }
        node_def {
          name: "mul/y"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_FLOAT
                tensor_shape {
                }
                float_val: 2.0
              }
            }
          }
          experimental_debug_info {
            original_node_names: "mul/y"
          }
        }
        node_def {
          name: "mul"
          op: "Mul"
          input: "mul_placeholder"
          input: "mul/y:output:0"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
          experimental_debug_info {
            original_node_names: "mul"
          }
        }
        node_def {
          name: "Identity"
          op: "Identity"
          input: "mul:z:0"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
          experimental_debug_info {
            original_node_names: "Identity"
          }
        }
        ret {
          key: "identity"
          value: "Identity:output:0"
        }
        arg_attr {
          key: 0
          value {
            attr {
              key: "_output_shapes"
              value {
                list {
                  shape {
                  }
                }
              }
            }
          }
        }
      }
    }
    versions {
      producer: 399
      min_consumer: 12
    }
    """
    # Code for generating above graph:
    #
    # def Body(i):
    #   @tf.function
    #   def f():
    #     return i * 2
    #   return f()
    # tf.while_loop(lambda i: i < 5., Body, [tf.constant(1.)])
    graph_def = graph_pb2.GraphDef()
    text_format.Parse(serialized, graph_def)
    @def_function.function
    def F():
      # [FUNCTION DOC]
      # HELPER: Compiled function. Imports the parsed graph returning the constant and loop output, and
      # returns the gradient of the loop output with respect to the constant.
      x, y = importer.import_graph_def(
          graph_def, return_elements=["Const:0", "while:2"])
      grad_out, = gradients_impl.gradients(y, x)
      return grad_out
    self.assertAllEqual(F(), 8.0)

  def testIndexedSlicesInIncomingGrads(self):
    # [FUNCTION DOC]
    # Evaluates a compiled function computing the gradient through a gather of a squared-twice loop
    # output (so the incoming gradient is sparse indexed-slices). Asserts the result is thirty-two.
    @def_function.function
    def F():
      # [FUNCTION DOC]
      # HELPER: Compiled function. Runs a two-iteration squaring loop on a one-element constant, gathers
      # element zero and returns the gradient with respect to the input.
      x = constant_op.constant([2.])
      # Computes x^4
      ret = while_loop_v2(
          lambda _: True, lambda v: v * v, [x], return_same_structure=False,
          maximum_iterations=2)
      v = array_ops.gather(ret, [0])
      return gradients_impl.gradients(v, [x])[0]  # 4*x^3
    self.assertAllEqual(self.evaluate(F()), [32.])

  def testShapeInvariantsRaggedTensor(self):
    # [FUNCTION DOC]
    # Runs a compiled loop that concatenates a ragged tensor with itself once, with shape invariants of
    # an integer scalar and a ragged spec with unknown dimensions. Asserts the result is the ragged
    # tensor with its rows repeated.

    @def_function.function
    def TestFn(x):
      # [FUNCTION DOC]
      # HELPER: Compiled function. Takes a ragged tensor, runs a one-iteration loop doubling it along
      # axis zero with explicit shape invariants and returns the result.
      _, ret = while_loop_v2(
          lambda i, _: i < 1,
          lambda i, y: (i + 1, array_ops.concat([y, y], axis=0)),
          [0, x],
          shape_invariants=[
              tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
              ragged_tensor.RaggedTensorSpec(shape=[None, None])],
      )
      return ret

    x = ragged_factory_ops.constant([[1., 2.], [3.]])
    result = TestFn(x)
    expected_result = [[1., 2.], [3.], [1., 2.], [3.]]
    self.assertAllEqual(result, expected_result)


def ScalarShape():
  # [FUNCTION DOC]
  # HELPER: Returns an empty integer list as a tensor (the shape of a scalar). Used as element shape
  # for tensor lists in the tests.
  return ops.convert_to_tensor([], dtype=dtypes.int32)


def GetOptimizedGraph():
  # [FUNCTION DOC]
  # HELPER: Optimizer used by the pruning tests.
  # Snapshots the default graph as a meta-graph, creates a config with constant folding turned off and
  # manual memory optimization, and returns the optimized graph definition from the graph optimizer.
  # Takes no arguments; reads the default graph without modifying it.
  mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())
  config = config_pb2.ConfigProto()
  config.graph_options.rewrite_options.CopyFrom(
      rewriter_config_pb2.RewriterConfig(
          constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
          memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL))
  return tf_optimizer.OptimizeGraph(config, mg)


if __name__ == "__main__":
  test.main()
