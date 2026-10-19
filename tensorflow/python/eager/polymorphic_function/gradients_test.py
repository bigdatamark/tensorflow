# [MODULE DOC]
# Tests for taking gradients of and through polymorphic (tf.function) functions. Covers
# gradients computed inside and outside traced functions with gradient tapes, implicit gradient
# helpers and symbolic graph gradients, including higher-order and iterated gradients, nested
# and deeply nested functions, functions touching variables, functions with None outputs,
# keyword arguments, control flow (conditionals and loops), and a function lacking a second-order
# gradient. Defines one test class and a list of cosine derivative functions used as expected
# values. When run as a script it enables eager execution before starting the test runner.
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

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


_COS_DERIVATIVES = [math_ops.cos,
                    lambda x: -math_ops.sin(x),
                    lambda x: -math_ops.cos(x),
                    math_ops.sin,
                    math_ops.cos]


class FunctionGradientsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering gradient computation for functions decorated as polymorphic functions,
  # using tapes, implicit gradients and symbolic graph gradients. The shared fixture splits the
  # first physical CPU into four logical devices (see setUp). Some tests are parameterized or run
  # in both graph and eager modes.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture run before every test. Calls the parent setUp, then configures the first
    # physical CPU device to have four logical (virtual) CPU devices. Side effect: changes global
    # device configuration of the runtime.
    super(FunctionGradientsTest, self).setUp()
    cpus = config.list_physical_devices('CPU')
    # Set 4 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

  def testGraphModeWithGradients(self):
    # [FUNCTION DOC]
    # Scenario: a traced function computes the implicit gradient of a squared resource variable
    # (value 1.0). Asserts that calling it returns 2.0, the derivative of v squared.
    v = resource_variable_ops.ResourceVariable(1.0, name='v')

    @polymorphic_function.function
    def step():
      # [FUNCTION DOC]
      # Traced function under test. Defines an inner function squaring the captured variable and
      # returns the first gradient value from the implicit gradient helper applied to it.
      def inner():
        # [FUNCTION DOC]
        # HELPER: loss function returning the square of the captured variable.
        return v * v

      return backprop.implicit_grad(inner)()[0][0]

    self.assertAllEqual(step(), 2.0)

  def testGraphGradientVariable(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph with a cached session, a traced function returns twice a
    # variable. Symbolic gradients of its output with respect to the variable are taken, the
    # variable is initialized and asserted: gradient value equals 2.0 and its shape equals the
    # variable shape.
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable(1.0)

      @polymorphic_function.function
      def f():
        # [FUNCTION DOC]
        # Traced function under test; returns twice the captured variable.
        return 2.0 * v

      node = f()
      grads, = gradients_impl.gradients(node, v)
      v.initializer.run()
      self.assertAllEqual(grads, 2.0)
      self.assertEqual(grads.shape, v.shape)

  def testSymbolicHigherOrder(self):
    # [FUNCTION DOC]
    # Scenario: a traced function computes cosine inside a nested traced function and applies
    # symbolic gradients a given number of times. For each order from 0 to 4 asserts the result
    # at 1.0 is close to the matching derivative of cosine from the module's list.
    @polymorphic_function.function
    def f(x, order):
      # [FUNCTION DOC]
      # Traced function under test. Inputs: x, a scalar tensor; order, number of times to apply
      # symbolic gradients. Computes cosine of x via a nested traced function then differentiates
      # that many times with respect to x and returns the result.
      y = polymorphic_function.function(lambda: math_ops.cos(x))()
      for _ in range(order):
        y, = gradients_impl.gradients(y, [x])
      return y
    for order, expected in enumerate(_COS_DERIVATIVES):
      self.assertAllClose(
          expected(constant_op.constant(1.)),
          f(constant_op.constant(1.), order))

  @parameterized.parameters([dict(persistent=True),
                             dict(persistent=False)])
  def testSymbolicHigherOrderUnderTape(self, persistent):
    # [FUNCTION DOC]
    # Parameterized over a persistent flag for the tape. Same as the symbolic higher-order test
    # but the cosine is computed under an active gradient tape, which takes a different call
    # path. Asserts that both the symbolic derivative and a derivative derived from the tape
    # gradient match the expected cosine derivative for each order.
    @polymorphic_function.function
    def f(x, order):
      # [FUNCTION DOC]
      # Traced function under test. Inputs: x, scalar tensor; order, number of differentiations.
      # Watches x under a tape (persistent per the test parameter), computes cosine in a nested
      # traced function, takes the tape gradient, then differentiates the output symbolically
      # order times and, for positive orders, the tape gradient order minus one times. Returns
      # the pair of the symbolic result and the tape-based result (equal for order zero).
      with backprop.GradientTape(persistent=persistent) as tape:
        tape.watch(x)
        # Note that having a tape active, even if we don't use it, forces us
        # down a different function call path. Symbolic gradients should work
        # here too; correctness of tape gradients are tested elsewhere.
        y = polymorphic_function.function(lambda: math_ops.cos(x))()
      tape_dy = tape.gradient(y, x)
      for _ in range(order):
        y, = gradients_impl.gradients(y, [x])
      if order > 0:
        y1 = tape_dy
        for _ in range(order - 1):
          y1, = gradients_impl.gradients(y1, [x])
      else:
        y1 = y
      return y, y1
    for order, expected_f in enumerate(_COS_DERIVATIVES):
      expected = self.evaluate(expected_f(constant_op.constant(1.)))
      self.assertAllClose(
          (expected, expected),
          f(constant_op.constant(1.), order))

  def testIteratedGradientsNested(self):
    # [FUNCTION DOC]
    # Scenario: repeatedly wraps a traced cosine function with tape-based gradient transforms,
    # both on the original and on a re-traced version. At each level asserts that the plain,
    # traced and freshly traced calls at 1.0 match the corresponding cosine derivative.

    def _grad(f):
      # [FUNCTION DOC]
      # HELPER: returns a function computing the gradient of the given single-argument function
      # via a gradient tape.
      def _grad_function(primal):
        # [FUNCTION DOC]
        # HELPER: watches the primal tensor under a tape, applies the wrapped function and
        # returns the gradient of its output with respect to the primal.
        with backprop.GradientTape() as tape:
          tape.watch(primal)
          primal_out = f(primal)
        return tape.gradient(primal_out, primal)
      return _grad_function

    @polymorphic_function.function
    def _forward(x):
      # [FUNCTION DOC]
      # HELPER: traced base function returning the cosine of its input.
      return math_ops.cos(x)

    f = _forward
    traced_f = polymorphic_function.function(f)
    one = constant_op.constant(1.)
    for expected in _COS_DERIVATIVES:
      self.assertAllClose(expected(one), f(one))
      self.assertAllClose(expected(one), traced_f(one))
      self.assertAllClose(expected(one), polymorphic_function.function(f)(one))
      f = _grad(f)
      traced_f = polymorphic_function.function(_grad(traced_f))

  def testIteratedGradientsNestedWithVariable(self):
    # [FUNCTION DOC]
    # Scenario: like the nested iterated-gradient test but the cosine is of a captured variable
    # (value 2.0) and functions take no arguments. Each level wraps the previous one with a tape
    # gradient with respect to the watched variable. Asserts that the function and a freshly
    # traced copy match each cosine derivative evaluated at 2.0.

    def _grad(f):
      # [FUNCTION DOC]
      # HELPER: returns a no-argument function giving the gradient of the wrapped no-argument
      # function with respect to the single variable watched by a tape.
      def _grad_function():
        # [FUNCTION DOC]
        # HELPER: runs the wrapped function under a tape and returns the gradient of its output
        # with respect to the one watched variable.
        with backprop.GradientTape() as tape:
          primal_out = f()
        g, = tape.gradient(primal_out, tape.watched_variables())
        return g
      return _grad_function

    v = variables.Variable(2.)

    @polymorphic_function.function
    def _forward():
      # [FUNCTION DOC]
      # HELPER: traced base function returning the cosine of the captured variable.
      return math_ops.cos(v)

    f = _forward

    two = constant_op.constant(2.)

    for expected in _COS_DERIVATIVES:
      self.assertAllClose(expected(two), f())
      self.assertAllClose(expected(two), polymorphic_function.function(f)())
      f = _grad(f)

  def testIteratedGradientsPersistent(self):
    # [FUNCTION DOC]
    # Scenario: a persistent tape watches an input to a traced cosine function and its gradient
    # is taken repeatedly (the result of each gradient feeds the next). Asserts the value at each
    # step equals the corresponding cosine derivative at the start value 1.0.

    @polymorphic_function.function
    def _forward(z):
      # [FUNCTION DOC]
      # HELPER: traced function returning the cosine of its input.
      return math_ops.cos(z)

    f = _forward
    with backprop.GradientTape(persistent=True) as tape:
      start = constant_op.constant(1.)
      tape.watch(start)
      x = f(start)
      for expected in _COS_DERIVATIVES:
        self.assertAllClose(expected(start), x)
        x = tape.gradient(x, start)

  def testHigherOrderWithVariable(self):
    # [FUNCTION DOC]
    # Scenario: a traced function returns the cosine of a captured variable (1.0), called under
    # a persistent tape. Repeatedly takes the gradient with respect to the tape's watched
    # variables and asserts each value equals the matching cosine derivative at 1.0.

    v = variables.Variable(1.)

    @polymorphic_function.function
    def _forward():
      # [FUNCTION DOC]
      # HELPER: traced function returning the cosine of the captured variable.
      return math_ops.cos(v)

    f = _forward
    with backprop.GradientTape(persistent=True) as tape:
      x = f()
      for expected in _COS_DERIVATIVES:
        self.assertAllClose(expected(constant_op.constant(1.)), x)
        x, = tape.gradient(x, tape.watched_variables())

  def testGradientsChained(self):
    # [FUNCTION DOC]
    # Scenario: a first tape records a traced cosine call; a second tape watches the incoming
    # output gradient (2.0) while the first tape's gradient is computed with it. Asserts the
    # first gradient equals minus twice sine of the input, and the gradient of that with respect
    # to the output gradient equals minus sine of the input.

    @polymorphic_function.function
    def _forward(z):
      # [FUNCTION DOC]
      # HELPER: traced function returning the cosine of its input.
      return math_ops.cos(z)

    f = _forward
    x = constant_op.constant(1.)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = f(x)
    with backprop.GradientTape() as tt:
      doutputs = constant_op.constant(2.)
      tt.watch(doutputs)
      g = t.gradient(y, x, doutputs)
    self.assertAllClose(-2. * math_ops.sin(x), g)
    gg = tt.gradient(g, doutputs)
    # We're taking gradients with respect to doutputs, which is just a linear
    # function of the gradient.
    self.assertAllClose(-math_ops.sin(x), gg)

  def testSymGradGatherNd(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, a traced function gathers an element by index from a matrix and
    # the symbolic gradient of its output with respect to the input is evaluated. Asserts that the
    # gradient (an indexed-slices value) has values equal to a single 1.0.
    with ops.Graph().as_default(), self.cached_session():

      @polymorphic_function.function
      def f(x):
        # [FUNCTION DOC]
        # Traced function under test; gathers the element at index zero (n-dimensional gather).
        return array_ops.gather_nd(x, [[0]])

      c = constant_op.constant([[2.]])
      f_c = f(c)
      g, = gradients_impl.gradients(f_c, c)
      self.assertAllEqual(self.evaluate(g).values, [[1.0]])

  def testNoSymGradNestedDefun(self):
    # [FUNCTION DOC]
    # Scenario: a traced outer function contains a nested traced function doing an index gather,
    # and computes symbolic gradients inside the outer function. Asserts the gradient is an
    # indexed-slices object, then calls the outer function once.

    @polymorphic_function.function
    def outer():
      # [FUNCTION DOC]
      # Traced function under test. Calls a nested traced gather, takes the symbolic gradient of
      # the result with respect to its input, and asserts it is an indexed-slices instance.
      # Returns nothing.

      @polymorphic_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: nested traced function gathering the element at index zero.
        return array_ops.gather_nd(x, [[0]])

      c = constant_op.constant([[2.]])
      f_c = f(c)
      g, = gradients_impl.gradients(f_c, c)
      self.assertIsInstance(g, indexed_slices.IndexedSlices)

    outer()

  def testGraphFunctionWithGradients(self):
    # [FUNCTION DOC]
    # Scenario: like the graph-mode implicit gradient test, but a concrete function is obtained
    # first. Asserts the concrete function reports float32 output dtype, a scalar output shape,
    # and returns 2.0 when called.
    v = resource_variable_ops.ResourceVariable(1.0, name='v')

    @polymorphic_function.function
    def step():
      # [FUNCTION DOC]
      # Traced function under test; returns the implicit gradient of the squared variable.
      def inner():
        # [FUNCTION DOC]
        # HELPER: loss function returning the square of the captured variable.
        return v * v

      return backprop.implicit_grad(inner)()[0][0]

    step_op = step.get_concrete_function()
    self.assertEqual(step_op.output_dtypes, dtypes.float32)
    self.assertEqual(step_op.output_shapes, tensor_shape.TensorShape([]))
    self.assertAllEqual(step_op(), 2.0)

  @test_util.run_in_graph_and_eager_modes()
  def testDefunCondGradient(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. A traced function doubles its input if above 0.5, else
    # triples it. Asserts the tape gradient at 1.0 is 2.0 (the true branch).

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Traced function under test; a conditional returning two times x when x exceeds 0.5,
      # otherwise three times x.
      return cond.cond(x > 0.5, lambda: 2 * x, lambda: 3 * x)

    with backprop.GradientTape() as t:
      x = constant_op.constant(1.0)
      t.watch(x)
      y = f(x)
    self.assertAllEqual(self.evaluate(t.gradient(y, x)), 2.0)

  @test_util.run_in_graph_and_eager_modes()
  def testGraphLoopGradient(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. A traced function runs a while loop that doubles x twice.
    # Asserts the tape gradient at 1.0 is 4.0.

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Traced function under test; a while loop with a counter below two that doubles x each
      # iteration; returns the final x (the loop counter is discarded).
      return while_loop.while_loop(
          lambda _, i: i < 2, lambda x, i: (2 * x, i + 1), [x, 0]
      )[0]

    with backprop.GradientTape() as t:
      x = constant_op.constant(1.0)
      t.watch(x)
      y = f(x)
    self.assertAllEqual(self.evaluate(t.gradient(y, x)), 4.0)

  def testGraphLoopGradientInsideSession(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode with a placeholder x and a constant 2.0, a traced function loops
    # multiplying by x while the value is below 10. The symbolic gradient with respect to x is
    # evaluated in a session with x fed as 2.0. Asserts it equals 24.0 (three iterations).
    with ops.Graph().as_default():
      n = constant_op.constant(2.0)
      x = array_ops.placeholder(dtypes.float32, shape=None)

      @polymorphic_function.function
      def f():
        # [FUNCTION DOC]
        # Traced function under test; while loop starting at n, multiplying by the captured
        # placeholder until the value reaches 10, with unknown output shape. Returns the loop
        # result list.
        c = lambda n: n < 10
        b = lambda n: n * x
        return while_loop.while_loop(c, b, [n], [tensor_shape.unknown_shape()])

      l = f()
      dx = gradients_impl.gradients(l, [x])[0]

      with self.cached_session():
        self.assertEqual(dx.eval(feed_dict={x: 2.0}), 24.0)

  def testDefunDifferentiable(self):
    # [FUNCTION DOC]
    # Scenario: the implicit gradient of a traced function returning a squared variable (1.0) is
    # asserted equal to 2.0.
    v = resource_variable_ops.ResourceVariable(1.0)

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function under test; returns the square of the captured variable.
      return v * v

    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)

  def testDefunCanBeDifferentiatedTwice(self):
    # [FUNCTION DOC]
    # Scenario: the implicit gradient of the same traced function is computed twice in a row.
    # Asserts both results equal 2.0, ensuring the variable is watched again on the second call.
    v = resource_variable_ops.ResourceVariable(1.0)

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function under test; returns the square of the captured variable.
      return v * v

    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)
    # Ensure that v is watched again.
    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)

  def testSymbolicGradientVariableNoneNotZerosLike(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, a traced function takes a tensor and a variable, reads the
    # variable but does not use it in the result. The symbolic gradient with respect to both is
    # taken. Asserts the gradient for the variable is the null value (not a zeros tensor).
    with ops.Graph().as_default():
      v = variables.Variable(1.0)

      @polymorphic_function.function
      def f(x, v):
        # [FUNCTION DOC]
        # Traced function under test. Reads the variable value for effect only and returns x
        # squared, so the output does not depend on the variable.
        v.read_value()
        return x * x

      x = constant_op.constant(1.0)
      l = f(x, v)
      _, dv = gradients_impl.gradients(l, [x, v])
      with self.cached_session():
        v.initializer.run()
        self.assertEqual(dv, None)

  def testDefunCallBackprop(self):
    # [FUNCTION DOC]
    # Scenario: a traced function g computes the gradient function of another traced function f
    # (x plus x) inside itself. Asserts g at 2.0 returns 2.

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Traced function returning x added to itself; the function being differentiated.
      return math_ops.add(x, x)

    @polymorphic_function.function
    def g(x):
      # [FUNCTION DOC]
      # Traced function under test; returns the derivative of f with respect to its first
      # argument, computed via the gradient-function helper.
      return backprop.gradients_function(f, [0])(x)[0]

    self.assertAllEqual(2, g(constant_op.constant(2.)))

  @test_util.run_v1_only('b/120545219')
  def testGraphModeEagerGradError(self):
    # [FUNCTION DOC]
    # Runs only for TF1 (v1 only; linked to a bug). In graph mode, computing an implicit
    # value-and-gradient of a function that creates a variable but does not access trainable
    # variables through the tape is expected to raise a value error with a message saying no
    # trainable variables were accessed.
    # KNOWN ISSUE: despite the name mentioning eager, the test runs entirely in graph mode.
    with context.graph_mode():
      def f():
        # [FUNCTION DOC]
        # HELPER: function creating a variable via the variable scope (initial 1.0) and returning
        # it multiplied by 2.0; intentionally not traced.
        x = variable_scope.get_variable(
            'v', initializer=constant_op.constant(1.0))
        return x * constant_op.constant(2.0)

      with self.assertRaisesRegex(ValueError,
                                  'No trainable variables were accessed'):
        backprop.implicit_val_and_grad(f)()

  def testDefunCallBackpropUsingSameObjectForMultipleArguments(self):
    # [FUNCTION DOC]
    # Scenario: a traced function computes gradients of multiplication with respect to both
    # arguments while the same tensor object is passed for both. Asserts the results are
    # [1., 1.] for a tensor input and for a plain Python float input.

    @polymorphic_function.function
    def g(x):
      # [FUNCTION DOC]
      # Traced function under test; returns the gradients of multiply with respect to both
      # inputs evaluated at x and x.
      return backprop.gradients_function(math_ops.multiply, [0, 1])(x, x)

    def np_g(x):
      # [FUNCTION DOC]
      # HELPER: calls g and converts each returned gradient to a numpy value, returning a list.
      return [d.numpy() for d in g(x)]

    x = constant_op.constant(1.)
    self.assertAllEqual([1., 1.], np_g(x))
    self.assertAllEqual([1., 1.], np_g(1.))

  def testGradientTensorConversionWithDefun(self):
    # [FUNCTION DOC]
    # Scenario: a traced function adds a variable (3.0) to its input; the implicit gradient of
    # a wrapper at 1.0 is taken with respect to the variable. Asserts it equals 1.0.
    three = resource_variable_ops.ResourceVariable(3.0, name='v')

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Traced function under test; returns x plus the captured variable.
      return math_ops.add(x, three)

    def g(x):
      # [FUNCTION DOC]
      # HELPER: plain wrapper calling the traced function; used as the differentiated function.
      return f(x)

    g = backprop.implicit_grad(g)(constant_op.constant(1.0))[0][0]
    self.assertAllEqual(g, 1.0)

  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: matrix multiplication wrapped as a traced function is used to square a 2 by 2
    # matrix (first argument transposed); the gradient function is taken with respect to the
    # input. Asserts the gradient equals [[6, 6], [14, 14]].
    matmul = polymorphic_function.function(math_ops.matmul)

    def sq(x):
      # [FUNCTION DOC]
      # HELPER: returns the product of the transpose of x with x using the traced matmul.
      return matmul(x, x, transpose_a=True)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    grad_t, = backprop.gradients_function(sq, [0])(t)
    self.assertAllEqual(grad_t, [[6, 6], [14, 14]])

  def testGradientInFunction(self):
    # [FUNCTION DOC]
    # Scenario: a traced function computes the gradient function of a squaring lambda inside
    # itself. Asserts calling it at 1.0 returns 2.0.

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Traced function under test; returns the derivative of y squared evaluated at x.
      return backprop.gradients_function(lambda y: y * y, [0])(x)[0]

    self.assertAllEqual(f(constant_op.constant(1.0)), 2.0)

  def testGradientOfGatherWithDefun(self):
    # [FUNCTION DOC]
    # Scenario: the implicit gradient of a function summing two gathered elements of a variable
    # is computed both for the plain function and its traced version. Asserts the number of
    # gradients matches and the values, indices and dense shape of the indexed-slices gradients
    # are equal.
    v = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    def sum_gather():
      # [FUNCTION DOC]
      # HELPER: returns the sum of elements at indices one and two of the captured variable.
      return math_ops.reduce_sum(array_ops.gather(v, [1, 2]))

    grad_fn = backprop.implicit_grad(sum_gather)
    gradient = grad_fn()
    defun_grad_fn = backprop.implicit_grad(
        polymorphic_function.function(sum_gather))
    defun_gradient = defun_grad_fn()
    self.assertEqual(len(gradient), len(defun_gradient))

    gradient = gradient[0][0]
    defun_gradient = defun_gradient[0][0]
    self.assertAllEqual(gradient.values, defun_gradient.values)
    self.assertAllEqual(gradient.indices, defun_gradient.indices)
    self.assertAllEqual(gradient.dense_shape, defun_gradient.dense_shape)

  def testDifferentiableFunctionNoneOutputs(self):
    # [FUNCTION DOC]
    # Scenario: differentiability of traced functions returning a null value among their outputs.
    # First, a function returning its input and null is differentiated through a wrapper that
    # picks the first output; asserts gradient 1. Then a function returning null and a square is
    # called under a tape; asserts the null output is null, the square is 25.0 and the gradient
    # is 10.0.

    @polymorphic_function.function
    def my_function(x):
      # [FUNCTION DOC]
      # Traced function returning a pair: the input and a null value.
      return x, None

    def wrapper(x):
      # [FUNCTION DOC]
      # HELPER: returns only the first output of the traced function.
      return my_function(x)[0]

    g = backprop.gradients_function(wrapper, [0])(constant_op.constant(0.0))
    self.assertAllEqual(g[0], 1.)

    @polymorphic_function.function
    def foo(a):
      # [FUNCTION DOC]
      # Traced function returning a pair: a null value and the square of the input.
      return None, a * a

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      none, r = foo(x)
    g = tp.gradient(r, x)

    self.assertIs(none, None)
    self.assertAllEqual(r, 25.0)
    self.assertAllEqual(g, 2 * 5.0)

  @test_util.run_in_graph_and_eager_modes
  def testNestedDifferentiableFunction(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. An outer traced function calls an inner traced function
    # computing a times (a plus b) with b fixed at 1.0. Asserts the tape gradient of the outer
    # result at 5.0 is 11.0.
    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced inner function returning a multiplied by the sum of a and b.
      return a * math_ops.add(a, b)

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outer function calling the inner function with b equal to 1.0.
      return inner_fn(x, 1.0)

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunction(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Four nested traced functions (addition, wrapper, product,
    # outer with b fixed at 1.0) are differentiated with a tape outside. Asserts the gradient
    # at 5.0 is 11.0.
    @polymorphic_function.function
    def inner_inner_fn(a, b):
      # [FUNCTION DOC]
      # Innermost traced function returning the sum of its two inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function delegating to the innermost function.
      return inner_inner_fn(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a times the result of the inner function.
      return a * inner_fn(a, b)

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outermost function calling the middle function with b equal to 1.0.
      return middle_fn(x, 1.0)

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionWithMultipleGradCalls(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Three nested traced functions (sum, product, outer with b
    # fixed at 3.0). Asserts forward values (40.0), gradients computed several times at 5.0
    # (13.0) and at 4.0 (11.0), forward values of intermediate functions called directly, and
    # the gradient of the inner function (2.0), showing repeated gradient calls stay correct.
    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning the sum of its inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a multiplied by the inner function result.
      return math_ops.mul(a, inner_fn(a, b))

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outer function calling the middle function with b equal to 3.0.
      return middle_fn(x, 3.0)

    x = constant_op.constant(5.0)
    self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))

    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 3.0)
    self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
    self.assertAllEqual(middle_fn(3.0, x), 3.0 * (3.0 + 5.0))

    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 3.0)

    y = constant_op.constant(4.0)
    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = outer_fn(y)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 2 * 4.0 + 3.0)

    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = inner_fn(y, y)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionGradientTapeInDefun(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Like the deeply nested test, but the gradient tape lives
    # inside the outermost traced function. Asserts the returned gradient at 5.0 is 11.0.
    @polymorphic_function.function
    def inner_inner_fn(a, b):
      # [FUNCTION DOC]
      # Innermost traced function returning the sum of its two inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function delegating to the innermost function.
      return inner_inner_fn(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a times the result of the inner function.
      return a * inner_fn(a, b)

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outermost function; watches x under a tape, calls the middle function with b
      # equal to 1.0 and returns the tape gradient of the result with respect to x.
      with backprop.GradientTape() as tp:
        tp.watch(x)
        result = middle_fn(x, 1.0)
      grad = tp.gradient(result, x)
      return grad

    x = constant_op.constant(5.0)
    grad = outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionGradientTapeInNestedDefun(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. The tape and gradient are in an inner traced function
    # (almost_outer_fn) which an outer traced function calls. Asserts the gradient at 5.0 is 11.0.
    @polymorphic_function.function
    def inner_inner_fn(a, b):
      # [FUNCTION DOC]
      # Innermost traced function returning the sum of its two inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function delegating to the innermost function.
      return inner_inner_fn(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a times the result of the inner function.
      return a * inner_fn(a, b)

    @polymorphic_function.function
    def almost_outer_fn(x):
      # [FUNCTION DOC]
      # Traced function; watches x under a tape, calls the middle function with b equal to 1.0
      # and returns the tape gradient of the result with respect to x.
      with backprop.GradientTape() as tp:
        tp.watch(x)
        result = middle_fn(x, 1.0)
      grad = tp.gradient(result, x)
      return grad

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outermost function delegating to the function that computes the gradient.
      return almost_outer_fn(x)

    x = constant_op.constant(5.0)
    grad = outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionGradientTapeInMultNestedDefun(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Same as the nested-tape test with one more wrapping level
    # (outer_outer_fn). Asserts the gradient at 5.0 is 11.0.
    @polymorphic_function.function
    def inner_inner_fn(a, b):
      # [FUNCTION DOC]
      # Innermost traced function returning the sum of its two inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function delegating to the innermost function.
      return inner_inner_fn(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a times the result of the inner function.
      return a * inner_fn(a, b)

    @polymorphic_function.function
    def almost_outer_fn(x):
      # [FUNCTION DOC]
      # Traced function; watches x under a tape, calls the middle function with b equal to 1.0
      # and returns the tape gradient of the result with respect to x.
      with backprop.GradientTape() as tp:
        tp.watch(x)
        result = middle_fn(x, 1.0)
      grad = tp.gradient(result, x)
      return grad

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced function delegating to the function that computes the gradient.
      return almost_outer_fn(x)

    @polymorphic_function.function
    def outer_outer_fn(x):
      # [FUNCTION DOC]
      # Traced outermost function delegating to the outer function.
      return outer_fn(x)

    x = constant_op.constant(5.0)
    grad = outer_outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionTFGradientInDefun(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Like the tape-in-function test but the outermost traced
    # function uses symbolic graph gradients instead of a tape. Asserts the gradient at 5.0 is 11.0.
    @polymorphic_function.function
    def inner_inner_fn(a, b):
      # [FUNCTION DOC]
      # Innermost traced function returning the sum of its two inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function delegating to the innermost function.
      return inner_inner_fn(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a times the result of the inner function.
      return a * inner_fn(a, b)

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outermost function; calls the middle function with b equal to 1.0 and returns the
      # symbolic gradient of the result with respect to x.
      result = middle_fn(x, 1.0)
      return gradients_impl.gradients(result, [x])[0]

    x = constant_op.constant(5.0)
    grad = outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionTFGradientInNestedDefun(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. The symbolic gradient is computed in an inner traced
    # function that an outer traced function calls. Asserts the gradient at 5.0 is 11.0.
    @polymorphic_function.function
    def inner_inner_fn(a, b):
      # [FUNCTION DOC]
      # Innermost traced function returning the sum of its two inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function delegating to the innermost function.
      return inner_inner_fn(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a times the result of the inner function.
      return a * inner_fn(a, b)

    @polymorphic_function.function
    def almost_outer_fn(x):
      # [FUNCTION DOC]
      # Traced function calling the middle function with b equal to 1.0 and returning the
      # symbolic gradient of the result with respect to x.
      result = middle_fn(x, 1.0)
      return gradients_impl.gradients(result, [x])[0]

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outermost function delegating to the function that computes the gradient.
      return almost_outer_fn(x)

    x = constant_op.constant(5.0)
    grad = outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionTFGradientInMultNestedDefun(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Same as the nested symbolic-gradient test with one more
    # wrapping level (outer_outer_fn). Asserts the gradient at 5.0 is 11.0.
    @polymorphic_function.function
    def inner_inner_fn(a, b):
      # [FUNCTION DOC]
      # Innermost traced function returning the sum of its two inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function delegating to the innermost function.
      return inner_inner_fn(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a times the result of the inner function.
      return a * inner_fn(a, b)

    @polymorphic_function.function
    def almost_outer_fn(x):
      # [FUNCTION DOC]
      # Traced function calling the middle function with b equal to 1.0 and returning the
      # symbolic gradient of the result with respect to x.
      result = middle_fn(x, 1.0)
      return gradients_impl.gradients(result, [x])[0]

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced function delegating to the function that computes the gradient.
      return almost_outer_fn(x)

    @polymorphic_function.function
    def outer_outer_fn(x):
      # [FUNCTION DOC]
      # Traced outermost function delegating to the outer function.
      return outer_fn(x)

    x = constant_op.constant(5.0)
    grad = outer_outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  def testDeeplyNestedDifferentiableFunctionWithVariable(self):
    # [FUNCTION DOC]
    # Scenario: nested traced functions where the outer one passes a captured variable (1.0) as
    # the second argument. Asserts the tape gradient with respect to the input at 5.0 is 11.0.
    var = variables.Variable(constant_op.constant(1.0))

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning the sum of its inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a times the inner function result.
      return a * inner_fn(a, b)

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outer function calling the middle function with the captured variable.
      return middle_fn(x, var)

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  def testDeeplyNestedDifferentiableFunctionWithVariableMultipleGradCalls(self):
    # [FUNCTION DOC]
    # Scenario: like the multiple-gradient-call test but the second argument is a captured
    # variable (3.0). Asserts forward values and repeated tape gradients, then reassigns the
    # variable to 1.5 and asserts the gradient at 4.0 reflects the new value (9.5), and that the
    # inner function gradient with respect to its first input is 1.0.
    v = variables.Variable(constant_op.constant(3.0))

    @polymorphic_function.function
    def inner_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning the sum of its inputs.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def middle_fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a multiplied by the inner function result.
      return math_ops.mul(a, inner_fn(a, b))

    @polymorphic_function.function
    def outer_fn(x):
      # [FUNCTION DOC]
      # Traced outer function calling the middle function with the captured variable.
      return middle_fn(x, v)

    x = constant_op.constant(5.0)
    self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))

    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 3.0)
    self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
    self.assertAllEqual(middle_fn(v, x), 3.0 * (3.0 + 5.0))

    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 3.0)

    y = constant_op.constant(4.0)
    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = outer_fn(y)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 2 * 4.0 + 3.0)

    v.assign(constant_op.constant(1.5))
    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = outer_fn(y)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 2 * 4.0 + 1.5)

    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = inner_fn(y, v)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 1.0)

  def testDeeplyNestedDifferentiableFunctionWithVariableMultipleTFGrads(self):
    # [FUNCTION DOC]
    # Scenario: graph-mode version of the variable test using symbolic gradients (no tape) and a
    # resource variable (3.0). Asserts forward values and repeated symbolic gradients at 5.0
    # (13.0), at 4.0 (11.0), after assigning 1.5 to the variable (9.5), and the inner function
    # gradient (1.0).
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(3.0)
      v.initializer.run()

      @polymorphic_function.function
      def inner_fn(a, b):
        # [FUNCTION DOC]
        # Traced function returning the sum of its inputs.
        return math_ops.add(a, b)

      @polymorphic_function.function
      def middle_fn(a, b):
        # [FUNCTION DOC]
        # Traced function returning a multiplied by the inner function result.
        return math_ops.mul(a, inner_fn(a, b))

      @polymorphic_function.function
      def outer_fn(x):
        # [FUNCTION DOC]
        # Traced outer function calling the middle function with the variable.
        return middle_fn(x, v)

      x = constant_op.constant(5.0)
      self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))

      grad, = gradients_impl.gradients(outer_fn(x), x)

      self.assertAllEqual(grad, 2 * 5.0 + 3.0)
      self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
      self.assertAllEqual(middle_fn(v, x), 3.0 * (3.0 + 5.0))

      grad, = gradients_impl.gradients(outer_fn(x), x)

      self.assertAllEqual(grad, 2 * 5.0 + 3.0)

      y = constant_op.constant(4.0)
      grad, = gradients_impl.gradients(outer_fn(y), y)
      self.assertAllEqual(grad, 2 * 4.0 + 3.0)

      self.evaluate(v.assign(constant_op.constant(1.5)))
      grad, = gradients_impl.gradients(outer_fn(y), y)

      self.assertAllEqual(grad, 2 * 4.0 + 1.5)

      grad, = gradients_impl.gradients(inner_fn(y, v), y)
      self.assertAllEqual(grad, 1.0)

  def testNestedDifferentiableFunctionNoneOutputs(self):
    # [FUNCTION DOC]
    # Scenario: a traced function returns four outputs, two of them null, and is called from
    # another traced function. Under a persistent tape at 5.0, asserts the two non-null outputs
    # (30.0 and 10.0), that the null ones stay null, and gradients 11.0 and 2.0 respectively.
    @polymorphic_function.function
    def foo(a, b):
      # [FUNCTION DOC]
      # Traced function returning a tuple: null, a times (a plus b), null, and two times a.
      return None, a * math_ops.add(a, b), None, 2*a

    @polymorphic_function.function
    def bar(x):
      # [FUNCTION DOC]
      # Traced function calling foo with b equal to 1.0 and returning all its outputs.
      return foo(x, 1.0)

    x = constant_op.constant(5.0)
    with backprop.GradientTape(persistent=True) as tp:
      tp.watch(x)
      none1, r1, none2, r2 = bar(x)
    g1 = tp.gradient(r1, x)
    g2 = tp.gradient(r2, x)

    self.assertAllEqual(r1, 30.0)
    self.assertAllEqual(r2, 10.0)
    self.assertIs(none1, None)
    self.assertIs(none2, None)
    self.assertAllEqual(g1, 2 * 5.0 + 1.0)
    self.assertAllEqual(g2, 2.0)

  def testGradientWithKeywordArguments(self):
    # [FUNCTION DOC]
    # Scenario: a traced matmul is called with keyword arguments in various orders and mixes of
    # positional and keyword. Asserts the gradient function result and tape gradients for each
    # call form all equal [[6, 6], [14, 14]].
    matmul = polymorphic_function.function(math_ops.matmul)

    def sq(x):
      # [FUNCTION DOC]
      # HELPER: returns the product of the transpose of x with x, passing both operands to the
      # traced matmul as keyword arguments.
      return matmul(a=x, b=x, transpose_a=True)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    grad_t, = backprop.gradients_function(sq, [0])(t)
    self.assertAllEqual(grad_t, [[6, 6], [14, 14]])

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(t)
      one = matmul(t, b=t, transpose_a=True)
      two = matmul(b=t, a=t, transpose_a=True)
      three = matmul(a=t, b=t, transpose_a=True)

    for output in [one, two, three]:
      self.assertAllEqual(tape.gradient(output, t), [[6, 6], [14, 14]])

  def testGradientInFunctionWithKeywordArguments(self):
    # [FUNCTION DOC]
    # Scenario: a traced function computing a gradient inside is called with its argument given
    # by keyword. Asserts the result at 1.0 is 2.0.

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Traced function under test; returns the derivative of y squared evaluated at x.
      return backprop.gradients_function(lambda y: y * y, [0])(x)[0]

    self.assertAllEqual(f(x=constant_op.constant(1.0)), 2.0)

  def testFunctionHasNoSecondOrderGradient(self):
    # [FUNCTION DOC]
    # Scenario: a traced function wraps a traced sparse softmax cross-entropy op (which has no
    # second-order gradient) with logits shifted by a variable, and another traced function takes
    # a tape gradient of it. Asserts that building and running succeeds and the gradient has
    # shape of two elements. The nn gradient module is imported only for this test's sake.

    # This test needs nn_grad imported. We could just disable the lint error,
    # but this way if the test is deleted we'll know the import isn't needed.
    _ = nn_grad

    v = variables.Variable(1.)

    @polymorphic_function.function
    def f(labels, logits):
      # [FUNCTION DOC]
      # Traced function; applies a nested traced sparse softmax cross-entropy to the labels and
      # logits plus the captured variable, returning the per-example loss.
      return polymorphic_function.function(
          nn_ops.sparse_softmax_cross_entropy_with_logits)(
              labels=labels, logits=logits + v)

    @polymorphic_function.function
    def f_grad():
      # [FUNCTION DOC]
      # Traced function; watches a constant logits vector under a tape, calls f with label 1
      # and returns the tape gradient of the output with respect to the logits.
      with backprop.GradientTape() as tape:
        logits = constant_op.constant([1., 2.])
        tape.watch(logits)
        out = f(constant_op.constant(1), logits)
      return tape.gradient(out, logits)
    # Mainly we want to check that the function builds despite
    # sparse_softmax_cross_entropy_with_logits not having a second-order
    # gradient defined.
    self.assertAllEqual([2], f_grad().shape)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
