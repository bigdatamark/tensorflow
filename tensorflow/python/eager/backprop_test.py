# [MODULE DOC]
# Unit tests for the eager gradient machinery in the backprop module and its utilities.
# Contains BackpropTest (gradient tape, gradients_function, implicit_grad, make_vjp, custom
# gradients, watching rules, attribute helpers), JacobianTest and BatchJacobianTest (jacobian
# and batch jacobian via parallel-for or while loop), and AggregateIndexedSlicesGradientsTest
# (the helper from the backprop utilities module). Not a helper module; it only defines test
# classes and has no import-time side effects beyond imports. Running the file executes the
# TensorFlow test main entry point.
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
import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import record
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework.memory_checker import MemoryChecker
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.training import training


class BackpropTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the eager automatic-differentiation API (gradient tape, gradients_function,
  # implicit_grad, make_vjp, jacobians, custom gradients). Combines the TensorFlow test base class
  # with parameterized testing. Most tests build small scalar or matrix computations, ask the tape
  # or the gradient helpers for derivatives, and compare them with hand-computed values or with
  # graph-mode gradients. Several tests are decorated to also run in graph mode, to check for leaked
  # tensors or garbage, or to run only when a GPU is available.

  @test_util.run_in_graph_and_eager_modes
  def testAggregateGradients(self):
    # [FUNCTION DOC]
    # Scenario: the same variable is read three times through embedding lookups with different index
    # lists, so its gradient is a sum of three sparse (indexed-slices) gradients. The eager gradient
    # (converted to a dense tensor) is computed with the gradients function helper. In graph mode only,
    # the same computation is rebuilt with the graph gradients function, its sparse result is turned
    # into a dense one with an unsorted segment sum, and the two are asserted to be close.

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: the function being differentiated in the aggregation test.
      # Takes a parameter matrix, looks up three different row sets from it and returns the elementwise
      # product of the three results.
      ind1 = constant_op.constant(np.array([0, 1]))
      ind2 = constant_op.constant(np.array([2, 3]))
      ind3 = constant_op.constant(np.array([1, 3]))
      g1 = embedding_ops.embedding_lookup(x, ind1)
      g2 = embedding_ops.embedding_lookup(x, ind2)
      g3 = embedding_ops.embedding_lookup(x, ind3)
      return g1 * g2 * g3

    var_np = np.random.rand(4, 2).astype(np.float32)
    var = constant_op.constant(var_np)
    grad = backprop.gradients_function(fn, [0])(var)[0]
    grad = self.evaluate(ops.convert_to_tensor(grad))

    if not context.executing_eagerly():
      tf_var = array_ops.constant(var_np, dtypes.float32)
      tf_ind1 = array_ops.constant([0, 1])
      tf_ind2 = array_ops.constant([2, 3])
      tf_ind3 = array_ops.constant([1, 3])
      tf_g1 = embedding_ops.embedding_lookup(tf_var, tf_ind1)
      tf_g2 = embedding_ops.embedding_lookup(tf_var, tf_ind2)
      tf_g3 = embedding_ops.embedding_lookup(tf_var, tf_ind3)
      tf_y = tf_g1 * tf_g2 * tf_g3
      tf_grad = gradients.gradients(tf_y, [tf_var])[0]

      tf_dense_grad = math_ops.unsorted_segment_sum(tf_grad.values,
                                                    tf_grad.indices,
                                                    tf_grad.dense_shape[0])

      self.assertAllClose(grad, self.evaluate(tf_dense_grad))

  @test_util.run_in_graph_and_eager_modes
  def testAggregateGradientsWithTensor(self):
    # [FUNCTION DOC]
    # Scenario: gradient aggregation where one contribution is sparse (an embedding lookup) and the
    # other is dense (a scaled sum of the whole variable). The eager gradient is evaluated; in graph
    # mode only it is compared to the gradient computed by the graph gradients function.

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: function differentiated in the mixed sparse/dense aggregation test.
      # Multiplies an embedding lookup of the first two rows by twice the sum of all input entries.
      ind1 = constant_op.constant(np.array([0, 1]))
      # A mixture of IndexedSlices and dense tensor to aggregate.
      g1 = embedding_ops.embedding_lookup(x, ind1)
      g2 = math_ops.reduce_sum(x * constant_op.constant(2.0))
      return g1 * g2

    var_np = np.random.rand(4, 2).astype(np.float32)
    var = constant_op.constant(var_np)
    grad = backprop.gradients_function(fn, [0])(var)[0]
    grad = self.evaluate(ops.convert_to_tensor(grad))

    if not context.executing_eagerly():
      tf_var = array_ops.constant(var_np, dtypes.float32)
      tf_ind1 = array_ops.constant([0, 1])
      tf_g1 = embedding_ops.embedding_lookup(tf_var, tf_ind1)
      tf_g2 = math_ops.reduce_sum(tf_var * 2.0, axis=(0, 1))
      tf_y = tf_g1 * tf_g2
      tf_grad = gradients.gradients(tf_y, [tf_var])[0]

      self.assertAllClose(grad, tf_grad)

  def testImplicitGradWithResourceVariable(self):
    # [FUNCTION DOC]
    # Scenario: implicit gradient of a function that reads a captured resource variable and adds
    # constants. Asserts the single returned gradient equals 1.0 and that the paired variable is the
    # very same variable object (by identity).
    x = resource_variable_ops.ResourceVariable(
        initial_value=constant_op.constant(1.0), name='x')

    def fn():
      # [FUNCTION DOC]
      # HELPER: function passed to the implicit gradient call; adds the variable value and two constants.
      b = constant_op.constant(2.0)
      c = math_ops.add(x.value(), b)
      return math_ops.add(c, constant_op.constant(3.0))

    grads_and_vars = backprop.implicit_grad(fn)()
    self.assertAllEqual(grads_and_vars[0][0], 1.0)
    self.assertAllEqual(id(grads_and_vars[0][1]), id(x))

  @parameterized.named_parameters([('Function', def_function.function),
                                   ('NoFunction', lambda f: f)])
  def testNoOpBehaviorConsistent(self, decorator):
    # [FUNCTION DOC]
    # Parameterized test (decorator is either the graph-function wrapper or a pass-through) checking
    # that tapes treat "no-op" operations (identity, add zero, subtract zero, multiply by one)
    # consistently. Takes the decorator to apply to the inner function. Asserts the gradients with
    # respect to the input and the four no-op copies are 2, 3, 3, 3, 3 for input 10.

    @decorator
    def f(x):
      # [FUNCTION DOC]
      # HELPER: body under test. Creates four no-op copies of the input, watches all five tensors, builds
      # a loss with coefficient 2 on the input and 3 on each copy, and returns the gradients of the loss
      # with respect to all five tensors.
      # Test all different types of no-ops
      x1 = array_ops.identity(x)
      x2 = math_ops.add_v2(x, 0)
      x3 = math_ops.subtract(x, 0)
      x4 = math_ops.multiply(x, 1)
      with backprop.GradientTape() as t:
        t.watch(x)
        t.watch(x1)
        t.watch(x2)
        t.watch(x3)
        t.watch(x4)
        y1 = x * 2.
        y2 = x1 * 3.
        y3 = x2 * 3.
        y4 = x3 * 3.
        y5 = x4 * 3.
        loss = y1 + y2 + y3 + y4 + y5
      return t.gradient(loss, [x, x1, x2, x3, x4])

    self.assertAllClose([2., 3., 3., 3., 3.], f(constant_op.constant(10.)))

  def testResourceHandleOutputWithoutHandleData(self):
    # [FUNCTION DOC]
    # Scenario: a user-created resource handle (variable handle op with no handle data) is watched
    # along with a float tensor and both pass through an identity-n op under a tape. Asserts that the
    # gradient of the float output with respect to the float input is 1.0, i.e. the missing handle data
    # does not break gradient computation.
    # This is a bit of a weird thing to test since we try to maintain handle
    # data. But users do create their own resources, and those often do not have
    # any handle data.
    h = resource_variable_ops.var_handle_op(
        shape=[], dtype=dtypes.float32, shared_name='abc')

    with backprop.GradientTape() as tape:
      x = constant_op.constant(1.)
      tape.watch(x)
      tape.watch(h)
      y, h = array_ops.identity_n([x, h])

    self.assertAllClose(1., tape.gradient(y, x))

  def testGradientInsideLoop(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a tape is used inside a while loop body that also reads a resource
    # variable in the loop context. Asserts the gradient of a value computed from the variable is not
    # null.
    with ops.Graph().as_default():
      v = resource_variable_ops.ResourceVariable(1.0)

      def body(_):
        # [FUNCTION DOC]
        # HELPER: loop body for the while loop. Reads the variable (to force a read inside the loop),
        # records a tape around a multiplication, asserts the gradient is not null, and returns 1.0.
        # Ignores its loop-variable argument.
        _ = v + 1.0  # This reads the variable inside the loop context
        with backprop.GradientTape() as t:
          result = v * 2
        self.assertIsNotNone(t.gradient(result, v))
        return 1.0

      while_loop.while_loop(lambda i: False, body, [1.0])

  def testWhereGradient(self):
    # [FUNCTION DOC]
    # Scenario: where-select is special because only some of its arguments have differentiable types.
    # Asserts the gradient of a function choosing x below 10 and x squared otherwise is 1.0 at 5 and
    # 100.0 at 50.
    # Note: where is special because only some of its arguments are of
    # differentiable dtypes.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function being differentiated; selects x or x squared depending on whether x is below 10.
      return array_ops.where(x < 10, x, x * x)

    g = backprop.gradients_function(f)

    self.assertAllEqual(g(5.)[0], 1.0)
    self.assertAllEqual(g(50.)[0], 100.0)

  def testTwoTargets(self):
    # [FUNCTION DOC]
    # Scenario: a tape watches two tensors and the gradient of a list of two targets (2x and 3y)
    # is requested with respect to both. Asserts the gradients are 2.0 and 3.0.
    with backprop.GradientTape() as t:
      x = constant_op.constant(3.0)
      y = constant_op.constant(2.0)
      t.watch([x, y])
      xx = 2 * x
      yy = 3 * y
    dx, dy = t.gradient([xx, yy], [x, y])
    self.assertAllEqual(dx, 2.0)
    self.assertAllEqual(dy, 3.0)

  def testCustomGradientEmptyError(self):
    # [FUNCTION DOC]
    # Scenario: a custom-gradient function whose backward function returns an empty list although it
    # has one input. Asserts that asking the tape for the gradient raises a value error.

    @custom_gradient.custom_gradient
    def identity(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient identity function whose gradient function deliberately returns the wrong
      # number of gradients.

      def grad(_):
        # [FUNCTION DOC]
        # HELPER: intentionally wrong backward function; ignores its argument and returns an empty list.
        return []  # This return value is wrong!

      return x, grad

    x = variables.Variable(1.0)
    with backprop.GradientTape() as t:
      y = identity(x)
    with self.assertRaises(ValueError):
      t.gradient(y, [x])

  def test_stop_gradient_hides_downstream_ops(self):
    # [FUNCTION DOC]
    # Scenario: a graph function wraps a custom-gradient op that fails the test (raises an assertion
    # error) if its backward function runs; the input is passed through stop-gradient first. Asserts
    # the tape returns a null gradient, i.e. ops downstream of stop-gradient are ignored.

    @custom_gradient.custom_gradient
    def _backward_pass_error(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient identity whose backward function must never run.
      # Returns its input and a gradient function.

      def _grad(_):
        # [FUNCTION DOC]
        # HELPER: backward function that always raises an assertion error explaining that the tape failed
        # to ignore tensors downstream of stop-gradient. Ignores its argument.
        raise AssertionError(
            'Unexpectedly ran the backward function. This probably means that '
            'tf.GradientTape is not properly ignoring tensors downstream of '
            'tf.stop_gradient.')

      return x, _grad

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: graph function that simply applies the failing custom-gradient op to its input.
      return _backward_pass_error(x)

    x = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      y = f(array_ops.stop_gradient(x))

    self.assertIsNone(tape.gradient(y, x))

  def testOutputGradUsedInComputation(self):
    # [FUNCTION DOC]
    # Scenario: gradient of two targets (x*y and x) with respect to x using explicit output gradients
    # 1.0 and 2.0. Asserts the result is 4.0 (y times 1 plus 2 times 1, with y being 2).
    with backprop.GradientTape() as t:
      x = constant_op.constant(3.0)
      y = constant_op.constant(2.0)
      t.watch([x, y])
      loss = x * y
    dx, = t.gradient([loss, x], [x], output_gradients=[1.0, 2.0])
    self.assertAllEqual(dx, 4.0)

  def testDy(self):
    # [FUNCTION DOC]
    # Scenario: gradients function of the identity function called with an explicit upstream gradient
    # keyword argument of 2.0. Asserts the first returned gradient is 2.0.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: identity function used as the function under differentiation.
      return x

    grad_fn = backprop.gradients_function(f)
    self.assertAllEqual(2., grad_fn(1., dy=2.)[0])

  def testGradientInteger(self):
    # [FUNCTION DOC]
    # Scenario: differentiate x plus x for an integer tensor. Asserts the returned gradient is null
    # because integer dtypes are not differentiable.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function that returns x plus x.
      return x + x

    int_tensor = constant_op.constant(1)
    self.assertEqual(backprop.gradients_function(f)(int_tensor)[0], None)

  def testErrors(self):
    # [FUNCTION DOC]
    # Scenario: custom-gradient function whose backward function raises a runtime error. Asserts that
    # that error propagates out of the gradients function. A to-do notes the right error type should
    # eventually be raised.

    @custom_gradient.custom_gradient
    def f(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient identity function whose backward function raises a runtime error.

      def grad(_):
        # [FUNCTION DOC]
        # HELPER: backward function that always raises a runtime error. Ignores its argument.
        raise RuntimeError('x')

      return x, grad

    # TODO(apassos) raise the right error here
    with self.assertRaises(RuntimeError):
      backprop.gradients_function(f)(constant_op.constant(1.0))

  def testGradientsFunctionInCustomGradient(self):
    # [FUNCTION DOC]
    # Scenario: the forward part of a custom-gradient function itself calls the gradients function
    # (derivative of x squared). Asserts that calling the function with 1.0 returns 2.0.

    @custom_gradient.custom_gradient
    def f(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient function; forward value is the derivative of squaring at x, backward
      # function returns twice the incoming gradient.
      (y,) = backprop.gradients_function(lambda x: x * x)(x)

      def grad(dy):
        # [FUNCTION DOC]
        # HELPER: backward function returning a list with twice the incoming gradient.
        return [2 * dy]

      return y, grad

    self.assertAllEqual(f(1.0), 2.0)

  def testImplicitGradOverEmbeddingLookup(self):
    # [FUNCTION DOC]
    # Scenario: compare eager implicit gradients over an embedding lookup of a resource variable with
    # the graph-mode gradients of the equivalent computation (batch of 8 rows of size 512 from a
    # vocabulary of 1000). Asserts sparse indices and values agree, then applies both gradients with
    # gradient-descent optimizers (learning rate 0.1) and asserts the updated embeddings match.
    batch_size = 8
    embedding_size = 512
    vocab_size = 1000
    lrn_rate = 0.1
    random_init = random_ops.random_uniform([vocab_size, embedding_size])

    x = array_ops.ones((batch_size), dtypes.int64)
    embedding = resource_variable_ops.ResourceVariable(
        initial_value=random_init, dtype=dtypes.float32, name='embedding')

    def f():
      # [FUNCTION DOC]
      # HELPER: function whose implicit gradient is taken; returns one minus the embedding lookup.
      embedded_x = embedding_ops.embedding_lookup(embedding, x)
      return constant_op.constant(1.0, dtypes.float32) - embedded_x

    grad = backprop.implicit_grad(f)()[0][0]
    opt = training.GradientDescentOptimizer(lrn_rate)

    with ops.Graph().as_default(), self.cached_session():
      tf_x = array_ops.ones((batch_size), dtypes.int64)
      # TODO(ashankar,apassos): Change to ResourceVariable.
      tf_embedding = variables.Variable(
          random_init.numpy(), name='tf_embedding')
      tf_embedded_x = embedding_ops.embedding_lookup(tf_embedding, tf_x)
      tf_y = 1.0 - tf_embedded_x
      tf_grad = gradients.gradients(tf_y, [tf_embedding])[0]
      tf_opt = training.GradientDescentOptimizer(0.1)
      tf_embedding.initializer.run()

      self.assertAllClose(tf_grad.indices, grad.indices)
      self.assertAllClose(tf_grad.values, grad.values)

      tf_opt.apply_gradients([(tf_grad, tf_embedding)]).run()
      expected = self.evaluate(tf_embedding)
    opt.apply_gradients([(grad, embedding)])
    self.assertAllClose(expected, embedding.read_value())

  def testImplicitGradOrdering(self):
    # [FUNCTION DOC]
    # Scenario: implicit gradient of a function using two variables in the order v1 first, v0 second.
    # Asserts the returned (gradient, variable) pairs are ordered by variable creation, v0 then v1.
    v0 = resource_variable_ops.ResourceVariable(1.0)
    v1 = resource_variable_ops.ResourceVariable(2.0)

    def f():
      # [FUNCTION DOC]
      # HELPER: returns v1 squared plus v0 squared.
      x = v1 * v1
      y = v0 * v0
      return x + y

    grads = backprop.implicit_grad(f)()
    ordered_variables = [x[1] for x in grads]
    self.assertIs(ordered_variables[0], v0)
    self.assertIs(ordered_variables[1], v1)

  def testTapeNoOpGradient(self):
    # [FUNCTION DOC]
    # Scenario: target is the watched source itself (no operation). Asserts the gradient is 1.0.
    x = constant_op.constant(3.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = x
    self.assertEqual(t.gradient(y, x).numpy(), 1.0)

  def testTapeIdentityGradientIsIdentity(self):
    # [FUNCTION DOC]
    # Scenario: target is the identity of the watched source. Asserts the gradient is 1.0.
    x = constant_op.constant(3.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = array_ops.identity(x)
    self.assertEqual(t.gradient(y, x).numpy(), 1.0)

  def testFunctionIndexedSlicesGradient(self):
    # [FUNCTION DOC]
    # Scenario: a graph function adds one to its input, followed by a gather outside the function so
    # the incoming gradient is sparse. Asserts the gradient with respect to the input is [1.0].

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: graph function returning x plus 1.
      return x + 1

    with backprop.GradientTape() as t:
      x = constant_op.constant([1.0])
      t.watch(x)
      y = f(x)
      y = array_ops.gather(y, [0])
    self.assertAllEqual(t.gradient(y, x), [1.0])

  def testTapeGradientMultiTargetOneIsSource(self):
    # [FUNCTION DOC]
    # Scenario: targets are the source x itself and x squared (x is 2). Asserts the total gradient is
    # 5.0 (1 plus 4).
    x = constant_op.constant(2.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = x * x
    self.assertEqual(t.gradient([x, y], x).numpy(), 5.0)

  def testTapeNoOpGradientWithMultiTargetAllSource(self):
    # [FUNCTION DOC]
    # Scenario: the same source tensor is listed twice as target. Asserts the gradient is 2.0.
    x = constant_op.constant(3.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = x
    self.assertEqual(t.gradient([y, y], x).numpy(), 2.0)

  def testTapeNoOpGradientWithMultiTargetMultiSource(self):
    # [FUNCTION DOC]
    # Scenario: targets x, y and y squared with sources x and y. Asserts the gradients are 1.0 for x
    # and 11.0 for y (1 plus 10).
    x = constant_op.constant(3.0)
    y = constant_op.constant(5.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      t.watch(y)
      z = y * y
    self.assertAllEqual(t.gradient([x, y, z], [x, y]), [1.0, 11.0])

  def testTapeGradientStringTarget(self):
    # [FUNCTION DOC]
    # Scenario: target is a string tensor and the source a float. Asserts the gradient is null.
    s = constant_op.constant('unknown', dtype=dtypes.string)
    x = constant_op.constant(3.0)

    with backprop.GradientTape() as t:
      t.watch(x)
      t.watch(s)
    grads = t.gradient(s, x)
    self.assertEqual(grads, None)

  def testTapeNoOpGradientStringSourceAndTarget(self):
    # [FUNCTION DOC]
    # Scenario: both target and source are the same string tensor. Asserts the gradient is null.
    s = constant_op.constant('unknown', dtype=dtypes.string)

    with backprop.GradientTape() as t:
      t.watch(s)
    grads = t.gradient(s, s)
    self.assertEqual(grads, None)

  def testTapeNoOpGradientWithMultiTargetMultiSourceIncludeString(self):
    # [FUNCTION DOC]
    # Scenario: multiple targets and sources including a string tensor. Asserts the float sources give
    # 1.0 and 11.0 and the string source gives null.
    x = constant_op.constant(3.0)
    y = constant_op.constant(5.0)
    s = constant_op.constant('unknown', dtype=dtypes.string)

    with backprop.GradientTape() as t:
      t.watch(x)
      t.watch(y)
      t.watch(s)
      z = y * y
    grads = t.gradient([x, y, z, s], [x, y, s])
    self.assertAllEqual(grads[:2], [1.0, 11.0])
    self.assertEqual(grads[2], None)

  def testTapeNoOpOnVariableIsIdentity(self):
    # [FUNCTION DOC]
    # Scenario: target is the read value of a variable. Asserts the gradient with respect to the
    # variable is 1.0.
    v0 = resource_variable_ops.ResourceVariable(1.0)
    with backprop.GradientTape() as t:
      y = v0.read_value()
    self.assertEqual(t.gradient(y, v0).numpy(), 1.0)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testTapeNoOpGradient2By2(self):
    # [FUNCTION DOC]
    # Scenario: persistent tape watching a 2 by 2 tensor used as both target and source. Asserts the
    # gradient is a 2 by 2 tensor of ones. The decorators also check that no new tensors and no
    # garbage objects are leaked.
    a_2_by_2 = constant_op.constant(2.0, shape=[2, 2])
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
    dy_dy = tape.gradient(a_2_by_2, [a_2_by_2])[0]
    self.assertAllEqual(dy_dy.numpy(),
                        constant_op.constant(1.0, shape=[2, 2]).numpy())

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testTapeNoOpGradientMultiTarget2By2(self):
    # [FUNCTION DOC]
    # Scenario: same as the 2 by 2 case but the tensor is listed twice as target. Asserts the gradient
    # is a 2 by 2 tensor of twos and that no new Python objects leak when running eagerly.
    a_2_by_2 = constant_op.constant(2.0, shape=[2, 2])
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
    dy_dy = tape.gradient([a_2_by_2, a_2_by_2], [a_2_by_2])[0]
    self.assertAllEqual(dy_dy.numpy(),
                        constant_op.constant(2.0, shape=[2, 2]).numpy())

  def testTapeStopRecording(self):
    # [FUNCTION DOC]
    # Scenario: a product computed inside the tape's stop-recording block. Asserts the gradient is
    # null because the operation was not recorded.
    with backprop.GradientTape() as t:
      x = resource_variable_ops.ResourceVariable(1.0)
      with t.stop_recording():
        y = x * x
    self.assertEqual(t.gradient(y, x), None)

  def testTapeStopStartRecording(self):
    # [FUNCTION DOC]
    # Scenario: doubling is recorded, then a square is computed while recording is stopped and another
    # square after it resumes. Asserts the gradient of the unrecorded result is null and the gradient
    # of the recorded one with respect to the doubled value is 4.0.
    with backprop.GradientTape(persistent=True) as t:
      x = resource_variable_ops.ResourceVariable(1.0)
      x2 = x * 2  # This should be differentiated through.
      with t.stop_recording():
        y = x2 * x2
      z = x2 * x2
    self.assertEqual(t.gradient(y, x2), None)

    # If the x*2 was not differentiated through, this would be 2.0, not 4.0
    self.assertEqual(t.gradient(z, x2).numpy(), 4.0)

  def testTapeReset(self):
    # [FUNCTION DOC]
    # Scenario: a square is recorded, the tape is reset, and another square is added. Asserts the
    # gradient equals 2.0, i.e. only the post-reset operation counts.
    with backprop.GradientTape() as t:
      v = resource_variable_ops.ResourceVariable(1.0)
      loss = v * v
      t.reset()
      loss += v * v
    self.assertAllEqual(t.gradient(loss, v), 2.0)

  def testPythonMax(self):
    # [FUNCTION DOC]
    # Scenario: the built-in max applied to three variables (2, 3, 5) under a tape. Asserts the result
    # is 5 and the gradient list is null for the first two and 1.0 for the largest.
    x = [
        resource_variable_ops.ResourceVariable(2.),
        resource_variable_ops.ResourceVariable(3.),
        resource_variable_ops.ResourceVariable(5.)
    ]
    with backprop.GradientTape() as t:
      f = max(x)
    grad = t.gradient(f, x)
    self.assertAllEqual(self.evaluate(f), 5.)
    self.assertAllEqual(self.evaluate(grad), [None, None, 1.0])

  def testAutomaticWatchedVariables(self):
    # [FUNCTION DOC]
    # Scenario: variables used inside a tape are watched automatically. Asserts the watched list is
    # empty initially, contains the variable after use, is empty after reset, and again contains it
    # after use following the reset.
    with backprop.GradientTape() as t:
      self.assertEqual(0, len(t.watched_variables()))
      v = resource_variable_ops.ResourceVariable(1.0)
      loss = v * v
      self.assertAllEqual([v], t.watched_variables())

      t.reset()
      self.assertEqual(0, len(t.watched_variables()))
      loss += v * v
      self.assertAllEqual([v], t.watched_variables())

  def testExplicitWatchedVariables(self):
    # [FUNCTION DOC]
    # Scenario: same checks as the automatic case but the variable is explicitly watched, including
    # after a reset and re-watching. Asserts the watched-variable list contents at each step.
    with backprop.GradientTape() as t:
      self.assertEqual(0, len(t.watched_variables()))
      v = resource_variable_ops.ResourceVariable(1.0)
      t.watch(v)
      self.assertAllEqual([v], t.watched_variables())

      t.reset()
      self.assertEqual(0, len(t.watched_variables()))
      t.watch(v)
      self.assertAllEqual([v], t.watched_variables())

  @test_util.assert_no_new_tensors
  def testGradientNone(self):
    # [FUNCTION DOC]
    # Scenario: softmax cross-entropy returns two outputs and the gradient with respect to the second
    # is null. Asserts the gradient with respect to logits is [[-0.5, 0.5]]. Also checks no new tensors
    # are leaked.

    def loss(x, l):
      # [FUNCTION DOC]
      # HELPER: function of logits and labels returning the mean of the softmax cross-entropy loss.
      return math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits(logits=x, labels=l),
          constant_op.constant([0]))

    logits = constant_op.constant([[0.0, 0.0]])
    labels = constant_op.constant([[1.0, 0.0]])
    # softmax_cross_entropy_with_logits returns two outputs and in this case the
    # gradient wrt the second is None.
    g, = backprop.gradients_function(loss, [0])(logits, labels)
    self.assertAllEqual(g.numpy(), [[-0.5, 0.5]])

  @test_util.run_in_graph_and_eager_modes
  def testGradientWithinTapeBlock(self):
    # [FUNCTION DOC]
    # Scenario: computing the gradient inside the tape's with-block, for both non-persistent and
    # persistent tapes, runs in graph and eager modes. Asserts the gradient of twice a variable is 2.0.
    v1 = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(v1.initializer)
    with backprop.GradientTape() as t:
      loss = 2 * v1
      grad = t.gradient(loss, v1)
    self.assertAllEqual(self.evaluate(grad), 2.0)

    with backprop.GradientTape(persistent=True) as t:
      loss = 2 * v1
      grad = t.gradient(loss, v1)
    self.assertAllEqual(self.evaluate(grad), 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testNestedSelfContexts(self):
    # [FUNCTION DOC]
    # Scenario: entering a tape that is already active as its own context. Asserts a value error is
    # raised.
    v1 = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(v1.initializer)
    with backprop.GradientTape() as t:
      with self.assertRaises(ValueError):
        with t:
          pass

  @test_util.assert_no_new_tensors
  def testSecondGrad(self):
    # [FUNCTION DOC]
    # Scenario: second derivative of a softmax cross-entropy sum, computed by differentiating a
    # function that itself computes a gradient. Asserts the result is [[0.0]].

    def first(x):
      # [FUNCTION DOC]
      # HELPER: softmax cross-entropy of the input against a fixed label, summed over axis zero.
      l = constant_op.constant([[0.0]])
      x = nn_ops.softmax_cross_entropy_with_logits(labels=l, logits=x)
      x = math_ops.reduce_sum(x, constant_op.constant([0]))
      return x

    def second(x):
      # [FUNCTION DOC]
      # HELPER: takes the gradient of the first function and sums it over axis zero.
      grad = backprop.gradients_function(first, [0])(x)[0]
      return math_ops.reduce_sum(grad, constant_op.constant([0]))

    f = constant_op.constant([[0.1]])
    grad = backprop.gradients_function(second, [0])(f)[0]
    self.assertAllEqual([[0.0]], grad)

  @test_util.run_in_graph_and_eager_modes
  def testWatchingIsTapeLocal(self):
    # [FUNCTION DOC]
    # Scenario: two nested tapes, the outer watching only one non-trainable variable and the inner both.
    # Asserts the second derivative of x1 cubed is 12.0 and that the second derivative for x2 is null
    # because the outer tape never watched x2.
    x1 = resource_variable_ops.ResourceVariable(2.0, trainable=False)
    x2 = resource_variable_ops.ResourceVariable(2.0, trainable=False)

    with backprop.GradientTape() as tape1:
      with backprop.GradientTape() as tape2:
        tape1.watch(x1)
        tape2.watch([x1, x2])
        y = x1**3
        z = x2**2
        dy, dz = tape2.gradient([y, z], [x1, x2])
      d2y, d2z = tape1.gradient([dy, dz], [x1, x2])

    self.evaluate([x1.initializer, x2.initializer])
    self.assertEqual(self.evaluate(d2y), 12.0)
    self.assertIsNone(d2z)

  @test_util.assert_no_new_tensors
  def testMakeVJP(self):
    # [FUNCTION DOC]
    # Scenario: build a vector-Jacobian-product function for squaring with a non-persistent tape.
    # Asserts the result at 3.0 is 9.0 and applying the product with 2.0 gives 12.0.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: squares its input.
      return x * x

    wrapped_fn = backprop.make_vjp(f, persistent=False)
    result, vjp = wrapped_fn(constant_op.constant(3.0))
    self.assertAllEqual(result, 9.0)
    self.assertAllEqual(vjp(2.0)[0], 12.0)

  def testPersistentMakeVJP(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final assertion is an equality check with three arguments; the third (12.0)
    # is only the failure message, so the value 12.0 is never actually asserted.
    # Scenario: build a vector-Jacobian-product function with a persistent tape and call it twice.
    # Asserts the two results are equal.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: squares its input.
      return x * x

    wrapped_fn = backprop.make_vjp(f, persistent=True)
    _, vjp = wrapped_fn(constant_op.constant(3.0))
    vjp_result1 = vjp(2.0)[0]
    vjp_result2 = vjp(2.0)[0]
    self.assertAllEqual(vjp_result1, vjp_result2, 12.0)

  @test_util.assert_no_new_tensors
  def testGradGrad(self):
    # [FUNCTION DOC]
    # Scenario: second derivative of squaring using nested gradients functions at 3.0. Asserts the
    # result is 2.0 and that no tensors are leaked.

    def sq(x):
      # [FUNCTION DOC]
      # HELPER: squares its input.
      return x * x

    def grad(x):
      # [FUNCTION DOC]
      # HELPER: returns the first gradient of the squaring function.
      value = backprop.gradients_function(sq, [0])(x)[0]
      return value

    gradgrad = backprop.gradients_function(grad, [0])

    self.assertAllEqual(gradgrad(constant_op.constant(3.0))[0], 2.0)

  @test_util.assert_no_new_tensors
  def testGradGradExp(self):
    # [FUNCTION DOC]
    # Scenario: second derivative of exp at 0.0. Asserts it equals 1.0 and that no tensors leak.

    def grad(x):
      # [FUNCTION DOC]
      # HELPER: returns the first gradient of the exponential function.
      value = backprop.gradients_function(math_ops.exp, [0])(x)[0]
      return value

    gradgrad = backprop.gradients_function(grad, [0])

    self.assertAllEqual(gradgrad(constant_op.constant(0.0))[0], 1.0)

  @test_util.assert_no_new_tensors
  def testStopGradient(self):
    # [FUNCTION DOC]
    # Scenario: gradient of stop-gradient applied to an argmax. Asserts the gradient is null.
    grad = backprop.gradients_function(
        lambda x: array_ops.stop_gradient(math_ops.argmax(x)))
    self.assertAllEqual(grad([0.0])[0], None)

  @test_util.assert_no_new_tensors
  def testArgmax(self):
    # [FUNCTION DOC]
    # Scenario: gradient of argmax passed through stop-gradient. Asserts the gradient is null.

    def argmax(x):
      # [FUNCTION DOC]
      # HELPER: returns the stop-gradient of the argmax of the input.
      i = math_ops.argmax(x)
      return array_ops.stop_gradient(i)

    grad = backprop.gradients_function(argmax)
    self.assertAllEqual(grad([0.0])[0], None)

  @test_util.run_gpu_only
  @test_util.assert_no_new_tensors
  def testGPU(self):
    # [FUNCTION DOC]
    # Scenario (GPU-only): gradient of a function that moves the input to the GPU, adds constants and
    # moves the result back to the CPU. Asserts the gradient is 1.0. A to-do mentions making the
    # vector space aware of devices.

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: adds two constants to a GPU copy of the input on GPU zero and returns the CPU copy.
      with context.device('/gpu:0'):
        b = constant_op.constant(2.0)
        c = math_ops.add(x.gpu(), b)
        # TODO(apassos): remove cpu below by making TensorVSPace aware
        # of devices.
        return math_ops.add(c, constant_op.constant(3.0)).cpu()

    grad = backprop.gradients_function(fn, [0])(constant_op.constant(1.0))[0]
    self.assertAllEqual(grad, 1.0)

  @test_util.run_gpu_only
  @test_util.assert_no_new_tensors
  def testGPUImplicitGrad(self):
    # [FUNCTION DOC]
    # Scenario (GPU-only): implicit gradient of a function reading a variable created on GPU zero.
    # Asserts the gradient moved to the CPU equals 1.0.
    with context.device('gpu:0'):
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant(1.0), name='v')

    def f():
      # [FUNCTION DOC]
      # HELPER: reads the variable value on GPU zero.
      with context.device('gpu:0'):
        return v.read_value()

    self.assertEqual(backprop.implicit_grad(f)()[0][0].cpu().numpy(), 1.0)

  @test_util.assert_no_new_tensors
  def testCPU(self):
    # [FUNCTION DOC]
    # Scenario: gradient of adding two constants to the input on CPU. Asserts the gradient is 1.0 and
    # no tensors leak.

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: adds two constants to the input.
      b = constant_op.constant(2.0)
      c = math_ops.add(x, b)
      return math_ops.add(c, constant_op.constant(3.0))

    grad = backprop.gradients_function(fn, [0])(constant_op.constant(1.0))[0]
    self.assertAllEqual(grad, 1.0)

  @test_util.run_gpu_only
  @test_util.assert_no_new_tensors
  def testTensorCopyGPU2CPU2GPU(self):
    # [FUNCTION DOC]
    # Scenario (GPU-only): inputs created on GPU zero are copied to CPU inside the differentiated
    # function. Asserts the gradient with respect to the first input is 1.0.

    def f(a, b):
      # [FUNCTION DOC]
      # HELPER: sums the CPU copies of its two arguments.
      return a.cpu() + b.cpu()

    with context.device('/gpu:0'):
      a = constant_op.constant(1.0)
      b = constant_op.constant(2.0)

    grad = backprop.gradients_function(f, [0])(a, b)[0]
    self.assertAllEqual(grad, 1.0)

  @test_util.assert_no_new_tensors
  def testEmptyParams(self):
    # [FUNCTION DOC]
    # Scenario: gradients function called without specifying parameter indices, so it differentiates
    # with respect to all arguments of a product. Asserts each gradient equals the other argument.

    def fn(a, b):
      # [FUNCTION DOC]
      # HELPER: multiplies its two arguments.
      return a * b

    x = constant_op.constant(1.0)
    y = constant_op.constant(2.0)
    dx, dy = backprop.gradients_function(fn)(x, y)
    self.assertAllEqual(dx, y.numpy())
    self.assertAllEqual(dy, x.numpy())

  @test_util.assert_no_new_tensors
  def testUnconnectedNone(self):
    # [FUNCTION DOC]
    # Scenario: the function reads a variable but returns an unrelated constant. Asserts the implicit
    # gradient for the variable is null.
    v = resource_variable_ops.ResourceVariable(1.0, name='testUnconnectedNone')

    def f():
      # [FUNCTION DOC]
      # HELPER: reads the variable (discarding the result) and returns the constant 1.0.
      v.read_value()
      return constant_op.constant(1.0)

    self.assertEqual(backprop.implicit_grad(f)()[0][0], None)

  @test_util.assert_no_new_tensors
  def testGradientTapeReEnterContext(self):
    # [FUNCTION DOC]
    # Scenario: a tape context is exited and entered again, recording more operations. Asserts the
    # gradient of the later target with respect to the original source is [4.0].
    g = backprop.GradientTape()
    with g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = 2 * x
    with g:
      z = 2 * y
    grad = g.gradient(target=z, sources=[x])
    self.assertEqual(self.evaluate(grad), [4.0])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGradientTapeRepeatedSource(self):
    # [FUNCTION DOC]
    # Scenario: the same source is passed twice in the source list. Asserts both returned gradients
    # are 2.0. Runs in graph and eager modes and checks for tensor leaks.
    with backprop.GradientTape(persistent=False) as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = 2 * x
    grad = g.gradient(target=y, sources=[x, x])
    self.assertEqual(self.evaluate(grad), [2.0, 2.0])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testPersistentGradientTapeRepeatedSource(self):
    # [FUNCTION DOC]
    # Scenario: persistent tape with x squared plus x times y; gradient queried twice with repeated
    # and reordered sources. Asserts [11, 11] for sources x, x and [3, 11] for sources y, x.
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant(3.0)
      y = constant_op.constant(5.0)
      g.watch(x)
      g.watch(y)
      z = x * x + x * y
    grad = g.gradient(target=z, sources=[x, x])
    self.assertEqual(self.evaluate(grad), [11.0, 11.0])
    grad = g.gradient(target=z, sources=[y, x])
    self.assertEqual(self.evaluate(grad), [3.0, 11.0])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGradientTapeStructure(self):
    # [FUNCTION DOC]
    # Scenario: the structure of the sources (single tensor, tuple, nested list, mapping) is mirrored
    # by the returned gradients. Uses a persistent tape on a linear combination with different
    # constants (constants are cached). Asserts shape and values of each structure.
    with backprop.GradientTape(persistent=True) as g:
      # Using different constant values because constant tensors are
      # cached, leading to a different gradient then what one might expect.
      x1 = constant_op.constant(3.0)
      x2 = constant_op.constant(3.1)
      x3 = constant_op.constant(3.2)
      g.watch(x1)
      g.watch(x2)
      g.watch(x3)
      y = x1 + 2 * x2 + 3 * x3
    self.assertEqual(self.evaluate(g.gradient(y, x1)), [1.0])
    self.assertEqual(self.evaluate(g.gradient(y, (x1,))), (1.0,))
    self.assertEqual(self.evaluate(g.gradient(y, (x1, x2))), (1.0, 2.0))
    self.assertEqual(
        self.evaluate(g.gradient(y, [(x1, x2), (x2, x3)])), [(1.0, 2.0),
                                                             (2.0, 3.0)])
    self.assertEqual(
        self.evaluate(g.gradient(y, (x1, x2, [x1, x3]))),
        (1.0, 2.0, [1.0, 3.0]))
    self.assertEqual(
        self.evaluate(g.gradient(y, [x1, {
            'x2': x2,
            'x3': x3
        }])), [1.0, {
            'x2': 2.0,
            'x3': 3.0
        }])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGradientTape(self):
    # [FUNCTION DOC]
    # Scenario: nested tapes; the inner tape computes a gradient that is added to the outer value.
    # Asserts the inner gradient is 2.0 and the outer gradient is 6.0.
    with backprop.GradientTape() as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      with backprop.GradientTape() as gg:
        gg.watch(y)
        z = 2 * y
      inner_grad = gg.gradient(z, [y])[0]
      self.assertEqual(self.evaluate(inner_grad), 2.0)
      y += inner_grad
    grad = g.gradient(y, [x])[0]
    self.assertEqual(self.evaluate(grad), 6.0)

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGadientTapeCalledOnConstantTarget(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name contains a typo ("Gadient"); the scenario is really a gradient request
    # between two unrelated variables, not a constant target.
    # Scenario: gradient of one variable with respect to another unrelated variable. Asserts null.
    with backprop.GradientTape() as g:
      x = variables.Variable([3.0])
      y = variables.Variable([2.0])
    grad = g.gradient(x, y)
    self.assertAllEqual(grad, None)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only('b/120545219')
  def testGradientTapeWithCond(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the condition compares x with itself, so it is always false and the true branch is
    # never taken; the test only exercises the false branch (x squared).
    # Scenario: gradient through a conditional. In graph mode asserts a not-implemented error
    # mentioning the graph gradients function; in eager mode asserts the gradient is 6.0. Only runs
    # for the v1 behavior.
    x = constant_op.constant(3.0)

    def true_fn():
      # [FUNCTION DOC]
      # HELPER: true branch; returns x.
      return x

    def false_fn():
      # [FUNCTION DOC]
      # HELPER: false branch; returns x squared.
      return x * x

    with backprop.GradientTape() as g:
      g.watch(x)
      y = tf_cond.cond(x < x, true_fn, false_fn)

    if not context.executing_eagerly():
      with self.assertRaisesRegex(NotImplementedError, 'tf.gradients'):
        dy = g.gradient(y, [x])[0]
    else:
      dy = g.gradient(y, [x])[0]
      self.assertEqual(self.evaluate(dy), 6.0)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only('b/120545219')
  def testGradientTapeWithWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario: gradient through a while loop that doubles x twice (x is 2). In graph mode asserts a
    # not-implemented error mentioning the graph gradients function; in eager mode asserts 4.0.
    i = constant_op.constant(1)
    x = constant_op.constant(2.)

    def cond(i, _):
      # [FUNCTION DOC]
      # HELPER: loop condition; true while the counter is below 3.
      return i < 3

    def body(i, x):
      # [FUNCTION DOC]
      # HELPER: loop body; increments the counter and doubles x.
      return i + 1, x * 2

    with backprop.GradientTape() as g:
      g.watch([x])
      _, y = while_loop.while_loop(cond, body, [i, x])

    if not context.executing_eagerly():
      with self.assertRaisesRegex(NotImplementedError, 'tf.gradients'):
        dy = g.gradient(y, [x])[0]
    else:
      dy = g.gradient(y, [x])[0]
      self.assertEqual(self.evaluate(dy), 4.0)

  @test_util.assert_no_new_tensors
  def testGradientTapeGradientCalledMultipleTimes(self):
    # [FUNCTION DOC]
    # Scenario: a non-persistent tape is asked for a gradient twice. Asserts the second call raises a
    # runtime error saying a non-persistent tape can only be used once.
    with backprop.GradientTape() as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      z = y * y
    g.gradient(z, [x])
    with self.assertRaisesRegex(
        RuntimeError, 'A non-persistent GradientTape can only'):
      g.gradient(y, [x])

  @test_util.assert_no_new_tensors
  def testGradientTapeJacobianCalledMultipleTimes(self):
    # [FUNCTION DOC]
    # Scenario: same as above but with the jacobian method. Asserts the second call raises the
    # runtime error.
    with backprop.GradientTape() as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      z = y * y
    g.jacobian(z, [x])
    with self.assertRaisesRegex(
        RuntimeError, 'A non-persistent GradientTape can only'):
      g.jacobian(y, [x])

  @test_util.assert_no_new_tensors
  def testJacobianInsideGradientTapeScope(self):
    # [FUNCTION DOC]
    # Scenario: jacobian requested inside the tape's with-block. Asserts it is close to 108
    # (4 times 3 cubed).
    with backprop.GradientTape() as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      z = y * y
      self.assertAllClose(4. * 3. ** 3., g.jacobian(z, x))

  @test_util.assert_no_new_tensors
  def testBatchJacobianInsideGradientTapeScope(self):
    # [FUNCTION DOC]
    # Scenario: batch jacobian requested inside the with-block of a persistent tape. Asserts the
    # result is [[[108]]].
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant([[3.0]])
      g.watch(x)
      y = x * x
      z = y * y
      self.assertAllClose([[[4. * 3. ** 3.]]], g.batch_jacobian(z, x))

  def testBatchJacobianParallelIterations(self):
    # [FUNCTION DOC]
    # Scenario: batch jacobian with parallel iterations inside a graph function. Asserts a runtime error
    # is raised with a non-persistent tape (message mentions persistent and parallel iterations) and
    # that a persistent tape gives sixteen copies of 108.
    @def_function.function
    def f(persistent):
      # [FUNCTION DOC]
      # HELPER: graph function taking a flag for persistence; builds a tape, tiles a quartic of x 16
      # times and returns its batch jacobian with 8 parallel iterations.
      with backprop.GradientTape(persistent=persistent) as t:
        x = constant_op.constant([[3.0]])
        t.watch(x)
        y = x * x
        z = array_ops.tile(y * y, [1, 16])
      return t.batch_jacobian(z, x, parallel_iterations=8)
    with self.assertRaisesRegex(RuntimeError,
                                'persistent=True.*parallel_iterations'):
      f(persistent=False)
    self.assertAllClose([[[4. * 3. ** 3.]] * 16], f(persistent=True))

  @test_util.assert_no_new_tensors
  def testGradientTapeBatchJacobianCalledMultipleTimes(self):
    # [FUNCTION DOC]
    # Scenario: non-persistent tape asked for a batch jacobian twice. Asserts the second call raises
    # the runtime error.
    with backprop.GradientTape() as g:
      x = constant_op.constant([[3.0]])
      g.watch(x)
      y = x * x
      z = y * y
    g.batch_jacobian(z, x)
    with self.assertRaisesRegex(
        RuntimeError, 'A non-persistent GradientTape can only'):
      g.batch_jacobian(y, [x])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only('b/120545219')
  def testPersistentTape(self):
    # [FUNCTION DOC]
    # Scenario: persistent tape queried twice. Asserts the gradients of x to the fourth power and of
    # x squared at 3 are 108 and 6, then deletes the tape.
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      z = y * y
    dz_dx = g.gradient(z, [x])[0]
    self.assertEqual(self.evaluate(dz_dx), 4 * 3 * 3 * 3)
    dy_dx = g.gradient(y, [x])[0]
    self.assertEqual(self.evaluate(dy_dx), 2 * 3)
    del g

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testHigherOrderGradient(self):
    # [FUNCTION DOC]
    # Scenario: first, second and third derivatives of x cubed at 3 from a persistent tape. Asserts
    # the values 27, 27, 18 and 6.
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x**3  # y       := x^3
      dy_dx = g.gradient(y, x)  # dy/dx   := 3x^2
      d2y_dx2 = g.gradient(dy_dx, x)  # d2y/dx2 := 6x
    d3y_dx3 = g.gradient(d2y_dx2, x)  # d3y/dx3 := 6
    x = 3
    self.assertAllClose(self.evaluate(y), x**3)
    self.assertEqual(self.evaluate(dy_dx), 3 * x**2)
    self.assertEqual(self.evaluate(d2y_dx2), 6 * x)
    self.assertEqual(self.evaluate(d3y_dx3), 6)
    del g

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testPersistentNestedTape(self):
    # [FUNCTION DOC]
    # Scenario: persistent nested tapes; the inner gradient is queried twice. Asserts inner gradient
    # 2.0, outer gradient 6.0 and a gradient of the inner target of 12.0. Deletes both tapes.
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      with backprop.GradientTape(persistent=True) as gg:
        gg.watch(y)
        z = 2 * y
      for _ in range(2):
        inner_grad = gg.gradient(z, [y])[0]
        self.assertEqual(self.evaluate(inner_grad), 2.0)
      y += inner_grad
      del gg
    grad = g.gradient(y, [x])[0]
    self.assertEqual(self.evaluate(grad), 6.0)
    grad = g.gradient(z, [x])[0]
    self.assertEqual(self.evaluate(grad), 12.0)
    del g

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGradientTapeVariable(self):
    # [FUNCTION DOC]
    # Scenario: gradient of a variable's square with respect to the variable. Asserts 2.0.
    v = resource_variable_ops.ResourceVariable(1.0, name='v')
    self.evaluate(v.initializer)
    with backprop.GradientTape() as g:
      y = v * v
    grad = g.gradient(y, [v])[0]
    self.assertAllEqual(self.evaluate(grad), 2.0)

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testNestedGradients(self):
    # [FUNCTION DOC]
    # Scenario: gradient of x to the fourth power with respect to x and to the intermediate square.
    # Asserts 108 and 18.
    x = constant_op.constant(3.0)
    with backprop.GradientTape() as g:
      g.watch(x)
      y = x * x
      z = y * y
    dz_dx, dz_dy = g.gradient(z, [x, y])
    self.assertEqual(self.evaluate(dz_dx), 108.0)
    self.assertEqual(self.evaluate(dz_dy), 18.0)

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testUnconnectedGradientsDefault(self):
    # [FUNCTION DOC]
    # Scenario: source is watched but unrelated to the target. Asserts the default result is null.
    x = constant_op.constant(1.0)
    y = constant_op.constant(3.0)
    with backprop.GradientTape() as g:
      g.watch([x, y])
      z = y * 2
    dz_dx = g.gradient(z, x)
    self.assertEqual(dz_dx, None)

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testUnconnectedGradientsZeros(self):
    # [FUNCTION DOC]
    # Scenario: unconnected source with the unconnected-gradients option set to zero. Asserts a 2 by
    # 2 matrix of zeros.
    x = constant_op.constant(1.0, shape=[2, 2])
    y = constant_op.constant(3.0)
    with backprop.GradientTape() as g:
      g.watch([x, y])
      z = y * 2
    dz_dx = g.gradient(z, x, unconnected_gradients='zero')
    self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(dz_dx))

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testUnconnectedGradientsVariablesZeros(self):
    # [FUNCTION DOC]
    # Scenario: same as above using resource variables. Asserts a 2 by 2 matrix of zeros.
    x = resource_variable_ops.ResourceVariable(
        constant_op.constant(1., shape=[2, 2]))
    self.evaluate(x.initializer)
    y = resource_variable_ops.ResourceVariable(constant_op.constant(3.))
    self.evaluate(y.initializer)
    with backprop.GradientTape() as g:
      g.watch([x, y])
      z = y * 2
    dz_dx = g.gradient(z, x, unconnected_gradients='zero')
    self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(dz_dx))

  @test_util.run_in_graph_and_eager_modes
  def testUnknownUnconnectedGradientsValueGiven(self):
    # [FUNCTION DOC]
    # Scenario: gradient requested with an invalid unconnected-gradients option string. Asserts a value
    # error whose message says the value is unknown and names the offending string.
    x = constant_op.constant(1.0)
    y = constant_op.constant(1.0)
    with backprop.GradientTape() as g:
      g.watch([x, y])
      z = y * 2
    with self.assertRaisesRegex(
        ValueError, "Unknown value for unconnected_gradients: 'nonsense'"):
      g.gradient(z, x, unconnected_gradients='nonsense')

  @test_util.run_in_graph_and_eager_modes
  def testUnconnectedGradientsNestedDefunZeros(self):
    # [FUNCTION DOC]
    # Scenario: nested graph functions where the inner result passes through stop-gradient, so the
    # target is unconnected to the source. Asserts that with the zero option the gradient is 0.0.

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: graph function squaring its input; called by the outer graph function.
      return x * x

    @def_function.function
    def h(y):
      # [FUNCTION DOC]
      # HELPER: graph function that calls the squaring function and applies stop-gradient to the result.
      z = f(y)
      return array_ops.stop_gradient(z)

    x = constant_op.constant(1.0)
    with backprop.GradientTape() as g:
      g.watch(x)
      k = x + 2.
      y = h(k)

    dy_dx = g.gradient(y, x, unconnected_gradients='zero')
    self.assertEqual(0.0, self.evaluate(dy_dx))

  def testInvalidRecordOperationMessage(self):
    # [FUNCTION DOC]
    # Scenario: manually record an operation whose backward function returns no gradients although it
    # has one input. Asserts that asking for the gradient raises an internal error naming the
    # operation and saying too few gradients were returned.
    y = constant_op.constant(2.)
    x = constant_op.constant(1.)
    with backprop.GradientTape() as g:
      g.watch(x)
      record.record_operation('InvalidBackprop', [y], [x], lambda dy: [])
    with self.assertRaisesRegex(errors_impl.InternalError,
                                'InvalidBackprop.*too few gradients'):
      g.gradient(y, x)

  @test_util.assert_no_new_tensors
  def testEmptyParamsForValueAndGradFunction(self):
    # [FUNCTION DOC]
    # Scenario: value-and-gradient function without parameter indices on a product of two floats.
    # Asserts the value is 6 and the gradients are the other argument. Also checks for tensor leaks.

    def fn(a, b):
      # [FUNCTION DOC]
      # HELPER: multiplies its two arguments.
      return a * b

    val_and_grads_fn = backprop.val_and_grad_function(fn)

    x = 2.0
    y = 3.0
    val, (dx, dy) = val_and_grads_fn(x, y)
    self.assertAllClose(val, x * y)
    self.assertAllEqual(dx, y)
    self.assertAllEqual(dy, x)

  @test_util.assert_no_new_tensors
  def testNonEmptyParamsForValueAndGradFunction(self):
    # [FUNCTION DOC]
    # Scenario: value-and-gradient function restricted to parameter index 1. Asserts the value is 6,
    # exactly one gradient is returned, and it equals the first argument (2.0).

    def fn(a, b):
      # [FUNCTION DOC]
      # HELPER: multiplies its two arguments.
      return a * b

    val_and_grad_fn = backprop.val_and_grad_function(fn, params=[1])

    x = 2.0
    y = 3.0
    val, grads = val_and_grad_fn(x, y)
    self.assertAllClose(val, x * y)
    self.assertEqual(1, len(grads))
    self.assertAllEqual(grads[0], x)

  @test_util.run_gpu_only
  @test_util.assert_no_new_tensors
  def testTensorCopyCPU2GPU2CPU(self):
    # [FUNCTION DOC]
    # Scenario (GPU-only): forward pass copies CPU inputs to the GPU, adds them there, copies back to
    # CPU and adds a constant; the backward pass crosses devices in reverse. Asserts the gradient
    # with respect to the first input is 1.0.
    # forward: a (cpu->gpu) -> add (gpu) -> c (gpu->cpu) -> add (cpu) -> e (cpu)
    # back: e (cpu) -> add (cpu) -> c (cpu->gpu) -> add (gpu) -> grad (gpu->cpu)
    def f(a, b):
      # [FUNCTION DOC]
      # HELPER: adds the GPU copies of both arguments on GPU zero, copies the sum to CPU and adds 3.0.
      with context.device('/gpu:0'):
        c = math_ops.add(a.gpu(0), b.gpu(0))
      return math_ops.add(c.cpu(), constant_op.constant(3.0))

    with context.device('/cpu:0'):
      a = constant_op.constant(1.0)
      b = constant_op.constant(2.0)

    grad = backprop.gradients_function(f, [0])(a, b)[0]
    self.assertAllEqual(grad, 1.0)

  def testGetAttrType(self):
    # [FUNCTION DOC]
    # Scenario: look up the attribute type of the type attribute of the add op. Asserts it equals the
    # integer value of the type attribute kind.
    typ = backprop.op_attr_type('Add', 'T')
    self.assertEqual(typ, int(pywrap_tfe.TF_ATTR_TYPE))

  def testGetAttrList(self):
    # [FUNCTION DOC]
    # Scenario: look up the type of the kernel-size attribute of max-pool. Asserts it is a list holding
    # the integer attribute kind.
    typ = backprop.op_attr_type('MaxPool', 'ksize')
    self.assertEqual(typ, [int(pywrap_tfe.TF_ATTR_INT)])

  def testMakeAttrType(self):
    # [FUNCTION DOC]
    # Scenario: convert the integer 1 to a type attribute value. Asserts it equals the float32 dtype.
    self.assertEqual(dtypes.float32,
                     backprop.make_attr(int(pywrap_tfe.TF_ATTR_TYPE), 1))

  def testMakeAttrTypeList(self):
    # [FUNCTION DOC]
    # Scenario: convert a list with integer 1 using a list-of-types attribute kind. Asserts the result
    # is a list with float32.
    self.assertEqual([dtypes.float32],
                     backprop.make_attr([int(pywrap_tfe.TF_ATTR_TYPE)], [1]))

  def testMakeAttrString(self):
    # [FUNCTION DOC]
    # Scenario: convert the text "a" for a string attribute. Asserts the result is the bytes value a.
    self.assertEqual(b'a',
                     backprop.make_attr(int(pywrap_tfe.TF_ATTR_STRING), 'a'))

  def testMakeAttrStringList(self):
    # [FUNCTION DOC]
    # Scenario: convert a list with the text "a" for a string-list attribute. Asserts a list of bytes.
    self.assertEqual(
        [b'a'], backprop.make_attr([int(pywrap_tfe.TF_ATTR_STRING)], ['a']))

  def testMulType(self):
    # [FUNCTION DOC]
    # Scenario: gradient of x times x using the internal multiplication dispatcher at 3.0. Asserts
    # the gradient is 6.0.

    def mul(x):
      # [FUNCTION DOC]
      # HELPER: multiplies x by itself through the internal multiplication dispatcher.
      return math_ops._mul_dispatch(x, x)  # pylint: disable=protected-access

    self.assertAllEqual(backprop.gradients_function(mul)(3.0)[0].numpy(), 6.0)

  def testMakeAttrShape(self):
    # [FUNCTION DOC]
    # Scenario: for several shapes (scalar, unknown rank, fully known, partially unknown) convert the
    # shape with the shape-attribute kind. Asserts the result equals the shape's protocol-buffer form,
    # with an informative failure message naming the shape.
    for s in ([], None, [1, 2, 3], [None, None], [1, None, 3]):
      expected = tensor_shape.TensorShape(s).as_proto()
      actual = backprop.make_attr(int(pywrap_tfe.TF_ATTR_SHAPE), s)
      self.assertEqual(
          expected,
          actual,
          msg=('For shape %r, expected %r != %r actual' %
               (s, expected, actual)))

  def testMakeAttrShapeList(self):
    # [FUNCTION DOC]
    # Scenario: convert a list of shapes using the shape-list attribute kind. Asserts the result is the
    # list of the corresponding shape protocol-buffer messages.
    shape_list = [[], None, [1, 2, 3], [None, None], [1, None, 3]]
    self.assertEqual(
        [tensor_shape.TensorShape(s).as_proto() for s in shape_list],
        backprop.make_attr([int(pywrap_tfe.TF_ATTR_SHAPE)], shape_list))

  def testArgsGradientFunction(self):
    # [FUNCTION DOC]
    # Scenario: gradients function applied to a function taking variadic positional arguments.
    # Asserts the gradient of the first argument squared at 1.0 is 2.0.

    def f(*args):
      # [FUNCTION DOC]
      # HELPER: returns the square of its first positional argument.
      return args[0] * args[0]

    grad = backprop.gradients_function(f)
    self.assertAllEqual(grad(1.0)[0], 2.0)

  def testPartial(self):
    # [FUNCTION DOC]
    # Scenario: gradients function applied to a partially applied two-argument product with the first
    # argument fixed to 2.0. Asserts the gradient with respect to the remaining argument is 2.0.

    def f(x, y):
      # [FUNCTION DOC]
      # HELPER: multiplies its two arguments.
      return x * y

    part = functools.partial(f, constant_op.constant(2.0))
    self.assertAllEqual(
        backprop.gradients_function(part)(constant_op.constant(1.0))[0], 2.0)

  def testReturnSameThing(self):
    # [FUNCTION DOC]
    # Scenario: function returning its input and twice its input. Asserts the gradient at 1.0 is 3.0
    # (the outputs' gradients are summed).

    def f(x):
      # [FUNCTION DOC]
      # HELPER: returns a pair of the input and twice the input.
      return x, 2 * x

    self.assertAllEqual(backprop.gradients_function(f)(1.0)[0], 3.0)

  @test_util.assert_no_new_tensors
  def testExceptionSafety(self):
    # [FUNCTION DOC]
    # Scenario: a function that raises a value error is differentiated (error swallowed), then a
    # normal square function is differentiated afterwards. Asserts the later gradient is 2.0, i.e. the
    # failure did not leave the tape machinery in a bad state. Also checks for tensor leaks.

    def f(unused_x):
      # [FUNCTION DOC]
      # HELPER: always raises a value error; ignores its argument.
      raise ValueError()

    try:
      backprop.gradients_function(f)(1.0)
    except ValueError:
      pass

    def real_f(x):
      # [FUNCTION DOC]
      # HELPER: squares its input.
      return x * x

    self.assertAllEqual(backprop.gradients_function(real_f)(1.0)[0], 2.0)

  @test_util.assert_no_new_tensors
  def testMultiValueConvertToTensor(self):
    # [FUNCTION DOC]
    # Scenario: implicit gradient of the mean of a stack of two copies of a variable-derived value;
    # tests that tensor conversion works with lists of tape-tracked tensors. Asserts the gradient is
    # [1.0]. Also checks for leaks.
    x = resource_variable_ops.ResourceVariable(
        initial_value=array_ops.constant([1.0]), name='x')

    def fn():
      # [FUNCTION DOC]
      # HELPER: adds 1 to the variable value, stacks two copies and returns their mean.
      a = math_ops.add(x.value(), 1.0)
      # Make sure convert_to_tensor works correctly with list of TensorNodes.
      b = array_ops_stack.stack([a, a], axis=0)
      return math_ops.reduce_mean(b)

    grad = backprop.implicit_grad(fn)()[0][0]
    self.assertAllEqual([1.0], grad)

  def testOutput(self):
    # [FUNCTION DOC]
    # Scenario: function with two outputs (x plus 2, x squared) at [0, 1, 2]. Asserts the gradient is
    # [1, 3, 5], i.e. the sum of the gradients of both outputs.

    def multiout(x):
      # [FUNCTION DOC]
      # HELPER: returns the pair of x plus 2 and x squared.
      return x + 2, x * x

    x = constant_op.constant([0.0, 1.0, 2.0])

    grad = backprop.gradients_function(multiout)(x)[0]
    self.assertAllEqual([1.0, 3.0, 5.0], grad)

  def testMultiValuePreservesIfNotDiffedAgainst(self):
    # [FUNCTION DOC]
    # Scenario: convolution differentiated with respect to only the image parameter, with the kernel
    # and strides passed through. Asserts the gradient with a 1x1 kernel of 2.0 is [[[[2.0]]]].

    def tfe_conv2d(timage, tkernel, conv2dstrides):
      # [FUNCTION DOC]
      # HELPER: 2D convolution with same padding of image and kernel using the given strides.
      return nn_ops.conv2d(timage, tkernel, conv2dstrides, 'SAME')

    i = constant_op.constant([[[[1.0]]]])
    k = constant_op.constant([[[[2.0]]]])
    s = [1, 1, 1, 1]

    grad = backprop.gradients_function(tfe_conv2d, params=(0,))(i, k, s)[0]
    self.assertAllEqual([[[[2.0]]]], grad)

  def testSameObjectForMultipleArguments(self):
    # [FUNCTION DOC]
    # Scenario: the same object is passed for both arguments of a product, for tensors, Python
    # numbers, nested lists and a variable. Asserts the gradients for the two arguments are both
    # 1.0 (or the matching nested shape), i.e. the arguments are not conflated.

    def f(x, y):
      # [FUNCTION DOC]
      # HELPER: multiplies its two arguments.
      return math_ops.multiply(x, y)

    g = backprop.gradients_function(f)

    def np_g(x, y):
      # [FUNCTION DOC]
      # HELPER: calls the gradients function and returns the two gradients as plain numeric arrays.
      dx, dy = g(x, y)
      return [dx.numpy(), dy.numpy()]

    x = constant_op.constant(1.)
    self.assertAllEqual([1., 1.], np_g(x, x))
    x = 1.
    self.assertAllEqual([1., 1.], np_g(x, x))
    x = constant_op.constant([[1.]])
    self.assertAllEqual([[[1.]], [[1.]]], np_g(x, x))
    x = [[1.]]
    self.assertAllEqual([[[1.]], [[1.]]], np_g(x, x))

    v = resource_variable_ops.ResourceVariable(
        initial_value=1., name='testSameObjectForMultipleArguments.Variable')
    self.assertAllEqual([1., 1.], np_g(v, v))

  @test_util.assert_no_new_tensors
  def testImplicitGradientsCustomGradientAndCachedVariableValue(self):
    # [FUNCTION DOC]
    # Scenario: implicit gradient of a custom-gradient square whose backward function adds one to
    # the usual gradient. Asserts exactly one pair is returned with gradient 7 (at x of 3, 2*3+1) and
    # the same variable. Also checks for leaks.

    @custom_gradient.custom_gradient
    def my_square(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient function returning the square of its input with a backward function.
      result = math_ops.square(x)

      def grad(dr):
        # [FUNCTION DOC]
        # HELPER: backward function returning twice the incoming gradient times x, plus one.
        return 2 * dr * x + 1

      return result, grad

    x = resource_variable_ops.ResourceVariable(
        initial_value=3., name='X.' + self.id())

    def f():
      # [FUNCTION DOC]
      # HELPER: applies the custom square to the variable.
      return my_square(x)

    g = backprop.implicit_grad(f)

    grads_and_vars = g()
    self.assertEqual(1, len(grads_and_vars))
    grad, var = grads_and_vars[0]
    self.assertAllEqual(7, grad)
    self.assertAllEqual(x, var)

  def testJacobianCustomGradient(self):
    # [FUNCTION DOC]
    # Scenario: a custom gradient defined inside a graph function whose backward function computes
    # jacobians of a quadratic callable object with respect to its variables. The outer tape is asked
    # for the gradient with respect to the input x. Asserts it is null (the callable's output is not a
    # differentiable function of x through the custom gradient).

    class MyCallable(object):
      # [CLASS DOC]
      # HELPER: small callable object used by the test; holds three variables a, b, c (1, 2, 3) and
      # evaluates the quadratic a*x^2 + b*x + c when called.

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the three scalar variables a, b and c with initial values 1, 2 and 3.
        self.a = variables.Variable(1.)
        self.b = variables.Variable(2.)
        self.c = variables.Variable(3.)

      def __call__(self, x):
        # [FUNCTION DOC]
        # Takes a tensor x and returns a times x squared plus b times x plus c, elementwise.
        return self.a * x * x + self.b * x + self.c

    @def_function.function
    def call(c, x):
      # [FUNCTION DOC]
      # HELPER: graph function taking the callable and an input; returns the output of an inner
      # custom-gradient function.

      @custom_gradient.custom_gradient
      def _call():
        # [FUNCTION DOC]
        # HELPER: custom-gradient function evaluating the callable at x, returning the output and a
        # backward function.
        y = c(x)

        def grad(dy, variables=None):  # pylint: disable=redefined-outer-name
          # [FUNCTION DOC]
          # HELPER: backward function taking the upstream gradient and an optional list of variables. Inside
          # a persistent tape it re-evaluates the callable, computes twice the sum of the upstream gradient
          # times the jacobian for each variable, and returns an empty tuple for inputs plus that list.
          with backprop.GradientTape(persistent=True) as g:
            g.watch(variables)
            y = c(x)
          grad_vars = [
              2 * math_ops.reduce_sum(dy * g.jacobian(y, v)) for v in variables
          ]
          del g
          return (), grad_vars

        return y, grad

      return _call()

    c = MyCallable()
    x = constant_op.constant([1., 2., 3.])
    with backprop.GradientTape(persistent=True) as g:
      g.watch([c.a, c.b, c.c])
      y = call(c, x)
    self.assertAllEqual(g.gradient(y, x), None)

  @test_util.assert_no_new_tensors
  def testCustomGradient(self):
    # [FUNCTION DOC]
    # Scenario: gradient descent (rate 0.25, five steps) on a loss defined with a custom-gradient
    # multiplication by 2, using the implicit value-and-gradient function. Asserts the recorded losses
    # are 4, 3, 2, 1, 0. Also checks for leaks.

    @custom_gradient.custom_gradient
    def my_mul(x, y):
      # [FUNCTION DOC]
      # HELPER: custom-gradient multiplication of two values; backward returns the gradients scaled by
      # the other operand.
      result = x * y

      def grad(dr):
        # [FUNCTION DOC]
        # HELPER: backward function returning the incoming gradient times y and times x.
        return [dr * y, dr * x]

      return result, grad

    lr = 0.25
    x = resource_variable_ops.ResourceVariable(2., name='x')

    def loss(x):
      # [FUNCTION DOC]
      # HELPER: returns the custom product of 2.0 and the variable's value.
      return my_mul(2., x.read_value())

    loss_grads_fn = backprop.implicit_val_and_grad(loss)

    losses = []
    for _ in range(5):
      loss, grads_and_vars = loss_grads_fn(x)
      losses.append(loss.numpy())
      for (grad, var) in grads_and_vars:
        var.assign_sub(lr * grad)
    self.assertAllEqual(losses, [4.0, 3., 2., 1., 0.])

  @test_util.assert_no_new_tensors
  def testCustomGradientIdentity(self):
    # [FUNCTION DOC]
    # Scenario: custom-gradient identity whose backward function doubles the gradient. Asserts the
    # gradient at 1.0 is 2.0. Also checks for leaks.

    @custom_gradient.custom_gradient
    def my_identity(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient identity function.

      def grad(dresult):
        # [FUNCTION DOC]
        # HELPER: backward function returning a list with twice the incoming gradient.
        return [2 * dresult]

      return x, grad

    self.assertAllEqual(backprop.gradients_function(my_identity)(1.0)[0], 2.0)

  def testDifferentiatingFunctionThatReturnsNone(self):
    # [FUNCTION DOC]
    # Scenario: functions that return nothing are differentiated with the implicit value-and-gradient
    # helper and the value-and-gradient function. Asserts each raises a value error saying a function
    # returning none cannot be differentiated.

    def fn(x, y):
      # [FUNCTION DOC]
      # HELPER: computes a product but returns nothing.
      result = x * y  # pylint: disable=unused-variable

    x = constant_op.constant(1)
    y = constant_op.constant(2)

    loss_grads_fn = backprop.implicit_val_and_grad(fn)
    with self.assertRaisesRegex(
        ValueError, 'Cannot differentiate a function that returns None; '
        'did you forget to return a value from fn?'):
      loss_grads_fn(x, y)

    val_and_grads_fn = backprop.val_and_grad_function(fn)
    with self.assertRaisesRegex(
        ValueError, 'Cannot differentiate a function that returns None; '
        'did you forget to return a value from fn?'):
      val_and_grads_fn(x, y)

  def testZerosCacheDoesntLeakAcrossGraphs(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, a gradient of a concatenation including a constant part is computed
    # twice, each time in a fresh graph. Asserts both results are equal, i.e. cached zeros from one
    # graph are not reused in another.
    with ops.Graph().as_default():

      def get_grad():
        # [FUNCTION DOC]
        # HELPER: builds a fresh graph with a constant and a watched variable, splits and squares half of
        # it, concatenates with the constant, and returns the evaluated gradient with respect to the input.
        with ops.Graph().as_default(), self.cached_session():
          t = constant_op.constant(1, dtype=dtypes.float32, shape=(10, 4))
          x = constant_op.constant(2, dtype=dtypes.float32, shape=(10, 4))
          with backprop.GradientTape() as tape:
            tape.watch(x)
            x1, _ = array_ops.split(x, num_or_size_splits=2, axis=1)
            y1 = x1**2
            y = array_ops.concat([y1, t], axis=1)
          return self.evaluate(tape.gradient(y, x))

      grad1 = get_grad()
      grad2 = get_grad()

      self.assertAllEqual(grad1, grad2)

  @test_util.run_in_graph_and_eager_modes
  def testSelectivelyWatchVariables(self):
    # [FUNCTION DOC]
    # Scenario: tape with automatic variable watching disabled, explicitly watching one of two
    # variables. Asserts the watched list contains only that variable, the other gradient is null and
    # the watched variable gets 3.0.
    x1 = resource_variable_ops.ResourceVariable(1.0)
    x2 = resource_variable_ops.ResourceVariable(1.0)
    with backprop.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(x2)
      y = x1**2
      z = x2**3
    self.assertTupleEqual(tape.watched_variables(), (x2,))
    dy, dz = tape.gradient([y, z], [x1, x2])
    self.evaluate([x1.initializer, x2.initializer])
    self.assertIsNone(dy)
    self.assertEqual(self.evaluate(dz), 3.0)

  @test_util.run_in_graph_and_eager_modes
  def testDifferentiatingScalarCache(self):
    # [FUNCTION DOC]
    # Scenario: two separate constants with equal values are added; guards against an eager scalar
    # cache returning the same object. Asserts the gradient with respect to the first is [1.0] rather
    # than 2.
    # In the following test, if x2 = x1 (i.e the objects are the exact same),
    # then y is essentially, 2*x1, and dy/dx1 = 2.
    # When we had a pure scalar cache in eager, this would be the case. This
    # test prevents us from going back to that case.
    with backprop.GradientTape(persistent=False) as g:
      x1 = constant_op.constant(3.0)
      x2 = constant_op.constant(3.0)
      g.watch(x1)
      g.watch(x2)
      y = x1 + x2
    grad = g.gradient(target=y, sources=[x1])
    self.assertEqual(self.evaluate(grad), [1.0])

  def testVariablesAndConstantsProduceTheSameGradients(self):
    # [FUNCTION DOC]
    # Scenario: gradients of a cube and a square with respect to two equal values, computed once with
    # equal constants and once with variables. Asserts both give identical results (guards against
    # shared cached tensors mixing up the gradients).

    # In the following test, differentiating [y, z] against [a, b] gives:
    # (dy/da + dz/da, dy/db + dz/db).
    # If a and b are the same constant, dz/da will not be 0 (which it should
    # be).
    # This is solved by using variable since doing a read_value on a tensor will
    # produce a new tensor and corresponding TensorHandle, and not reuse the
    # same tensor (which would happen if we are using a cache and reusing
    # EagerTensor objects).
    def get_grads(a, b):
      # [FUNCTION DOC]
      # HELPER: records a cube of a and a square of b on a tape watching both and returns the gradients
      # of both results with respect to a and b.
      with backprop.GradientTape() as tape:
        tape.watch([a, b])
        y = a**3
        z = b**2
      return tape.gradient([y, z], [a, b])

    gradients_constants = get_grads(
        constant_op.constant(2.0), constant_op.constant(2.0))
    gradients_variables = get_grads(
        resource_variable_ops.ResourceVariable(2.0),
        resource_variable_ops.ResourceVariable(2.0))
    self.assertAllEqual(gradients_constants, gradients_variables)

  def testUnknownShapes(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): placeholder of unknown shape, cubed under a tape. Runs the graph with
    # value 2.0 and asserts the results are 8.0 and 12.0.
    with ops.Graph().as_default():
      with backprop.GradientTape() as tape:
        a = array_ops.placeholder(dtype=dtypes.float32, shape=None)
        tape.watch(a)
        b = a**3

      db_da = tape.gradient(b, a)

      with self.cached_session() as sess:
        self.assertEqual((8.0, 12.0), sess.run((b, db_da), feed_dict={a: 2.0}))

  @test_util.run_in_graph_and_eager_modes
  def testCustomGradientInEagerAndGraph(self):
    # [FUNCTION DOC]
    # Scenario: custom gradient with a backward function returning four times the incoming gradient,
    # run in both modes. Asserts the gradient is 4.0.

    @custom_gradient.custom_gradient
    def f(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient function returning x squared with a backward function that returns four
      # times the upstream gradient.
      y = x * x

      def grad(dy):
        # [FUNCTION DOC]
        # HELPER: backward function returning a list with four times the incoming gradient.
        return [4 * dy]

      return y, grad

    with backprop.GradientTape() as t:
      c = constant_op.constant(1.0)
      t.watch(c)
      g = f(c)
    self.assertAllEqual(self.evaluate(t.gradient(g, c)), 4.0)

  def testOverrideSecondOrderWithCustomGradient(self):
    # [FUNCTION DOC]
    # Scenario: a custom gradient whose first-order backward function itself uses a custom gradient to
    # override second-order behavior. Asserts the first-order gradient equals 4 times f cubed times
    # -1.1 and the second-order gradient equals the expected combination of terms using -1.1 and -2.1.

    @custom_gradient.custom_gradient
    def f(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient function returning x plus 10 with a first-order backward function.

      def first_order_grad(dz):
        # [FUNCTION DOC]
        # HELPER: backward function that multiplies the upstream gradient by an inner custom-gradient value.

        @custom_gradient.custom_gradient
        def first_order_custom(unused_x):
          # [FUNCTION DOC]
          # HELPER: custom-gradient function returning the constant -1.1 whose own backward function scales
          # by -2.1.

          def h(ddz):
            # [FUNCTION DOC]
            # HELPER: backward function returning -2.1 times the incoming gradient.
            return -2.1 * ddz

          return -1.1, h

        return dz * first_order_custom(x)

      return x + 10., first_order_grad

    c = constant_op.constant(1.)
    with backprop.GradientTape() as outer:
      outer.watch(c)
      with backprop.GradientTape() as inner:
        inner.watch(c)
        d = f(c)**4.
      dd = inner.gradient(d, c)
      self.assertAllClose(4. * f(c)**3. * -1.1, dd)
    self.assertAllClose(3. * 4. * f(c)**2. * -1.1 * -1.1 + 4. * f(c)**3. * -2.1,
                        outer.gradient(dd, c))

  @test_util.run_in_graph_and_eager_modes
  def testCustomGradientForwardprop(self):
    # [FUNCTION DOC]
    # Scenario: forward-mode style custom gradient. The forward function returns twice the constant
    # value of x; its backward function is itself a custom-gradient function returning 2.1 whose
    # backward returns the pair 2.2 and 3.1. Two nested tapes are used. Asserts the output is 2.0, the
    # first-order gradient with an explicit output gradient is 2.1, the second derivative with respect
    # to the input is 2.2, and the transpose gradient with respect to the output gradient is 3.1. Runs
    # in graph and eager modes.

    @custom_gradient.custom_gradient
    def f(x):
      # [FUNCTION DOC]
      # HELPER: custom-gradient function returning twice the constant value of x with backward function g.
      z = 2. * tensor_util.constant_value(x)

      def g(dz):
        # [FUNCTION DOC]
        # HELPER: backward function that delegates to a custom-gradient helper called with x and dz.

        @custom_gradient.custom_gradient
        def first_order(unused_x, unused_dz):
          # [FUNCTION DOC]
          # HELPER: custom-gradient function returning 2.1 whose backward returns 2.2 and 3.1.

          def second_order_and_transpose(unused_ddz):
            # [FUNCTION DOC]
            # HELPER: backward function ignoring its argument and returning the constants 2.2 and 3.1.
            return 2.2, 3.1

          return 2.1, second_order_and_transpose

        return first_order(x, dz)

      return z, g

    with backprop.GradientTape(persistent=True) as t:
      with backprop.GradientTape() as tt:
        c = constant_op.constant(1.)
        t.watch(c)
        tt.watch(c)
        output_grad = array_ops.ones([])
        t.watch(output_grad)
        output = f(c)
        self.assertAllClose(2., output)
      gc = tt.gradient(output, c, output_gradients=output_grad)
      self.assertAllClose(2.1, gc)
    ggc = t.gradient(gc, c)
    self.assertAllClose(2.2, ggc)
    # Note that executed eagerly this kind of transpose is not efficient. But
    # from a tf.function we could prune out the first-order gradient
    # computation.
    transpose = t.gradient(gc, output_grad)
    self.assertAllClose(3.1, transpose)

  @test_util.run_in_graph_and_eager_modes
  def testWatchBadThing(self):
    # [FUNCTION DOC]
    # Scenario: watching a NumPy array (not a tensor). Asserts a value error mentioning ndarray.
    g = backprop.GradientTape()
    with self.assertRaisesRegex(ValueError, 'ndarray'):
      g.watch(np.array(1.))

  def testWatchComposite(self):
    # [FUNCTION DOC]
    # Scenario: watching a sparse tensor expands it into its component tensors. A sparse sum is taken
    # and the gradient with respect to the sparse values is requested. Asserts it is [1.0, 1.0].
    """Test that tape.watch expands composites and watches component Tensors."""
    with backprop.GradientTape() as t:
      values = constant_op.constant([1.0, 2.0], dtypes.float32)
      s = sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]], values=values, dense_shape=[3, 4])
      t.watch(s)
      z = sparse_ops.sparse_reduce_sum_v2(s)
    result = t.gradient(z, values)
    self.assertAllEqual(result, [1.0, 1.0])

  def testWatchedVariablesAfterNonPersistentGradientCall(self):
    # [FUNCTION DOC]
    # Scenario: after the gradient of a non-persistent tape is computed, the watched-variables list
    # still holds the variable. Asserts it equals a tuple with the variable.
    with backprop.GradientTape(persistent=False) as tape:
      x = resource_variable_ops.ResourceVariable(1.0)
      tape.watch(x)
    tape.gradient(x, x)
    self.assertEqual((x,), tape.watched_variables())

  def testWatchedVariablesOnlyHasVariablesFromLastTape(self):
    # [FUNCTION DOC]
    # Scenario: the tape name is rebound to a second tape that watches a different variable.
    # Asserts the watched variables of that latest tape contain only its own variable.
    with backprop.GradientTape(persistent=False) as tape:
      x = resource_variable_ops.ResourceVariable(1.0)
      tape.watch(x)
    with backprop.GradientTape(persistent=False) as tape:
      z = resource_variable_ops.ResourceVariable(2.0)
      tape.watch(z)
    tape.gradient(z, z)
    self.assertEqual((z,), tape.watched_variables())

  def testWatchedVariablesRespectReset(self):
    # [FUNCTION DOC]
    # Scenario: variable watched, tape reset, another variable watched. Asserts the watched-variable
    # tuple holds the first variable, then after reset only the second one, and still after the
    # gradient call.
    with backprop.GradientTape(persistent=False) as tape:
      x = resource_variable_ops.ResourceVariable(1.0)
      tape.watch(x)
      self.assertEqual((x,), tape.watched_variables())
      tape.reset()
      z = resource_variable_ops.ResourceVariable(2.0)
      tape.watch(z)
      self.assertEqual((z,), tape.watched_variables())
    tape.gradient(z, z)
    self.assertEqual((z,), tape.watched_variables())

  def testNameScope(self):
    # [FUNCTION DOC]
    # Scenario: a function applying cosine twice inside a name scope is differentiated inside a graph
    # function. Asserts exactly two sine operations (the derivative of cosine) appear in the traced
    # graph and that each has the tape gradient prefix followed by the original name scope.

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: inside the name scope my_scope, adds two cosines of x and returns the sum.
      with ops.name_scope('my_scope'):
        a = math_ops.cos(x)
        b = math_ops.cos(x)
        return math_ops.add(a, b)

    @def_function.function
    def grad_fn(x):
      # [FUNCTION DOC]
      # HELPER: graph function returning the gradients of fn at x.
      return backprop.gradients_function(fn)(x)

    grad_ops = grad_fn.get_concrete_function(
        constant_op.constant(1.0)).graph.get_operations()
    num_sin_ops_found = 0
    for op in grad_ops:
      if op.type == 'Sin':
        num_sin_ops_found += 1
        self.assertIn('gradient_tape/my_scope/', op.name)
    self.assertEqual(num_sin_ops_found, 2)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testRecomputeGradWithDifferentShape(self):
    # [FUNCTION DOC]
    # Scenario: recompute-gradient decorated functions taking a list of variables of different shapes
    # (and, in a second case, a mapping keyed by variable references). Under a tape, asserts the
    # outputs add one to each input, i.e. [2, 3] and 2. The decorator checks no new Python objects
    # are created when running eagerly.
    @custom_gradient.recompute_grad
    def outer(x):
      # [FUNCTION DOC]
      # HELPER: recompute-gradient function returning a list with the first two entries each plus one.
      return [x[0] + 1, x[1] + 1]

    x = [
        variables.Variable([1.0, 2.0], name='a'),
        variables.Variable(1.0, name='b')
    ]
    with backprop.GradientTape():
      y = outer(x)
      self.assertAllEqual(y[0], [2.0, 3.0])
      self.assertAllEqual(y[1], 2.0)

    @custom_gradient.recompute_grad
    def outer_dict(x):
      # [FUNCTION DOC]
      # HELPER: recompute-gradient function that adds one to every value in the mapping and returns it.
      # Mutates the mapping it is given.
      for key in x.keys():
        x[key] = x[key] + 1
      return x

    x = {x[0].ref(): x[0], x[1].ref(): x[1]}
    with backprop.GradientTape():
      y = outer_dict(x)
      y = list(y.values())
      self.assertAllEqual(y[0], [2.0, 3.0])
      self.assertAllEqual(y[1], 2.0)

  @parameterized.parameters([(True), (False)])
  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testRecomputeGradWithNestedFunctionAndWhileLoop(self, reduce_retracing):
    # [FUNCTION DOC]
    # Scenario: parameterized by whether to reduce retracing. A recompute-gradient function nests three
    # levels of graph functions with a ten-iteration while loop calling the innermost one. Runs five
    # times in a tape with a memory checker. Asserts each result is 11.0 and that no memory is leaked
    # (except possibly one allocation).
    @custom_gradient.recompute_grad
    @def_function.function(reduce_retracing=reduce_retracing)
    def outer(x):
      # [FUNCTION DOC]
      # HELPER: outermost graph function (wrapped in recompute-gradient); calls the middle function.

      @def_function.function(reduce_retracing=reduce_retracing)
      def middle(y):
        # [FUNCTION DOC]
        # HELPER: graph function running a while loop for ten iterations that applies the inner function
        # each time and returns the final value.

        @def_function.function(reduce_retracing=reduce_retracing)
        def inner(z):
          # [FUNCTION DOC]
          # HELPER: innermost graph function adding one to its input.
          return z + 1

        i = constant_op.constant(0.0)
        c = lambda y, i: i < 10.
        b = lambda y, i: (inner(y), i + 1.0)
        y, i = while_loop.while_loop(c, b, [y, i])

        return y

      return middle(x)

    with MemoryChecker() as memory_checker:
      for _ in range(5):
        x = variables.Variable(1.0, name='x')
        with backprop.GradientTape():
          y = outer(x)
          self.assertAllEqual(y, 11.0)

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()


class JacobianTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the jacobian method of the gradient tape, covering the parallel-for and while-loop
  # implementations, graph-function use, persistent-tape requirements, parallel iterations,
  # nesting, convolution, sparse gradients, fold operations, conditionals and empty inputs.

  def _jacobian(self, experimental_use_pfor):
    # [FUNCTION DOC]
    # HELPER: builds and returns a jacobian and the expected answer for the tests in this class.
    # KNOWN ISSUE: the persistent flag uses the function object for checking eager mode rather than
    # calling it, so that part of the expression is always true and the tape is persistent whenever
    # the parallel-for option is off, regardless of the execution mode.
    # Takes a flag choosing the parallel-for implementation. Computes z as x squared times y for
    # two constant vectors, requests the jacobian with respect to x and y, and returns it together with
    # the expected diagonal matrices (2xy and x squared).
    persistent = context.executing_eagerly and not experimental_use_pfor
    with backprop.GradientTape(persistent=persistent) as g:
      x = constant_op.constant([1., 2.])
      y = constant_op.constant([3., 4.])
      g.watch(x)
      g.watch(y)
      z = x * x * y
    jacobian = g.jacobian(
        z, [x, y], experimental_use_pfor=experimental_use_pfor)
    answer = [array_ops.diag(2 * x * y), array_ops.diag(x * x)]
    return jacobian, answer

  @test_util.run_v1_only('b/120545219')
  def testPfor(self):
    # [FUNCTION DOC]
    # Scenario: jacobian via parallel-for. Asserts each jacobian equals the expected diagonal matrix.
    jacobian, answer = self._jacobian(experimental_use_pfor=True)
    for j, a in zip(jacobian, answer):
      self.assertAllEqual(a, j)

  @test_util.run_v1_only('b/120545219')
  def testWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario: jacobian via the while-loop implementation. Asserts equality with the expected answer.
    jacobian, answer = self._jacobian(experimental_use_pfor=False)
    for j, a in zip(jacobian, answer):
      self.assertAllEqual(a, j)

  @test_util.run_v1_only('b/120545219')
  def testPforDefun(self):
    # [FUNCTION DOC]
    # Scenario: parallel-for jacobian computed inside a graph function. Asserts equality.

    @def_function.function
    def _f():
      # [FUNCTION DOC]
      # HELPER: graph function returning the parallel-for jacobian and the expected answer.
      return self._jacobian(experimental_use_pfor=True)

    jacobian, answer = _f()
    for j, a in zip(jacobian, answer):
      self.assertAllEqual(a, j)

  @test_util.run_v1_only('b/120545219')
  def testWhileLoopDefun(self):
    # [FUNCTION DOC]
    # Scenario: while-loop jacobian computed inside a graph function. Asserts equality.

    @def_function.function
    def _f():
      # [FUNCTION DOC]
      # HELPER: graph function returning the while-loop jacobian and the expected answer.
      return self._jacobian(experimental_use_pfor=False)

    jacobian, answer = _f()
    for j, a in zip(jacobian, answer):
      self.assertAllEqual(a, j)

  @test_util.run_v1_only('b/120545219')
  def testPersistentTape(self):
    # [FUNCTION DOC]
    # Scenario (eager only; returns early otherwise): non-persistent tape with the while-loop
    # jacobian. Asserts a runtime error mentioning persistent is raised.
    if not context.executing_eagerly():
      return
    with backprop.GradientTape() as g:
      x = constant_op.constant([1.0, 2.0])
      g.watch(x)
      y = x * x
    with self.assertRaisesRegex(RuntimeError, 'persistent'):
      g.jacobian(y, x, experimental_use_pfor=False)

  @test_util.run_v1_only('b/120545219')
  def test_parallel_iterations(self):
    # [FUNCTION DOC]
    # Scenario: jacobian of a matrix product with 2 versus 3 parallel iterations on a persistent
    # tape. Asserts the two results are close.
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant([[1., 2], [3, 4]])
      g.watch(x)
      y = math_ops.matmul(x, x)
    self.assertAllClose(
        g.jacobian(y, x, parallel_iterations=2),
        g.jacobian(y, x, parallel_iterations=3))

  @test_util.run_in_graph_and_eager_modes
  def test_nested_jacobian(self):
    # [FUNCTION DOC]
    # Scenario (graph mode; skipped when eager because function-call conversion is not implemented,
    # tracked by a to-do): nested non-persistent tapes. The inner computes the jacobian of a sum of
    # squares and the outer the batch jacobian of that. Asserts the result is ten copies of a 2 by 2
    # diagonal of 2.
    if context.executing_eagerly():
      # TODO(agarwal): b/128842926
      self.skipTest('Conversion of function calls not implemented yet.')
    x = array_ops.ones((10, 2))
    with backprop.GradientTape(persistent=False) as g:
      g.watch(x)
      with backprop.GradientTape(persistent=False) as gg:
        gg.watch(x)
        y = math_ops.reduce_sum(math_ops.square(x))
      dy_x = gg.jacobian(y, x)
    dy_xx = g.batch_jacobian(dy_x, x)
    dy_xx_answer = [[[2., 0], [0, 2.]]] * 10
    self.assertAllClose(dy_xx_answer, self.evaluate(dy_xx))

  def test_nested_batch_jacobian_foldl(self):
    # [FUNCTION DOC]
    # Scenario: nested batch jacobians over a fold-left computation, applied twice. At each level
    # compares the theoretical and numerical gradients from the checker, then checks that the nested
    # jacobian function, both directly and inside a graph function, matches the numerical result
    # within tolerance.
    def _grad(f):
      # [FUNCTION DOC]
      # HELPER: higher-order function turning a function into one that returns its batch jacobian.
      def _grad_function(primal):
        # [FUNCTION DOC]
        # HELPER: records the wrapped function of the primal on a tape and returns the batch jacobian of
        # the output with respect to the primal.
        with backprop.GradientTape() as tape:
          tape.watch(primal)
          primal_out = f(primal)
        return tape.batch_jacobian(primal_out, primal)
      return _grad_function

    def _func(x):
      # [FUNCTION DOC]
      # HELPER: folds a cosine-of-sum accumulation over the transposed input and reshapes to 1 by 1.
      return array_ops.reshape(
          functional_ops.foldl_v2(lambda a, b: math_ops.cos(a + b),
                                  array_ops.transpose(x)),
          [1, 1])

    f = _func
    x = constant_op.constant([[1., 2.]])
    for _ in range(2):
      theoretical, numerical = gradient_checker_v2.compute_gradient(f, [x])
      self.assertAllClose(theoretical, numerical, rtol=1e-3)
      f = _grad(f)
      expected_flat = array_ops.reshape(numerical, [-1])
      self.assertAllClose(expected_flat,
                          array_ops.reshape(f(x), [-1]),
                          rtol=1e-3)
      self.assertAllClose(expected_flat,
                          array_ops.reshape(def_function.function(f)(x), [-1]),
                          rtol=1e-3)

  def test_grad_jacobian_conv(self):
    # [FUNCTION DOC]
    # Scenario: gradient of a batch jacobian through a convolution. Compares the numerical and
    # theoretical gradients of the inner computation (within a loose tolerance) and then asserts that
    # the gradient computed through an outer graph function matches the numerical one.
    def _inner(x):
      # [FUNCTION DOC]
      # HELPER: convolves the input with an all-ones kernel, sums squares over two axes, and returns the
      # sum of the batch jacobian with respect to the input.
      kernel = array_ops.ones([3, 3, 1, 9])
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = nn_ops.conv2d(x, kernel, strides=(1, 1), padding='SAME',
                          data_format='NHWC')
        reduced = math_ops.reduce_sum(y ** 2., axis=[2, 3])
      return math_ops.reduce_sum(tape.batch_jacobian(reduced, x))

    theoretical, numerical = gradient_checker_v2.compute_gradient(
        def_function.function(_inner), [array_ops.ones([10, 4, 4, 1])])
    self.assertAllClose(numerical, theoretical, rtol=1e-1)

    @def_function.function
    def _outer():
      # [FUNCTION DOC]
      # HELPER: graph function returning the gradient of the inner function at an all-ones input.
      with backprop.GradientTape() as tape:
        x = array_ops.ones([10, 4, 4, 1])
        tape.watch(x)
        y = _inner(x)
      return tape.gradient(y, x)

    self.assertAllClose(array_ops.reshape(numerical, [-1]),
                        array_ops.reshape(_outer(), [-1]), rtol=1e-1)

  @test_util.run_in_graph_and_eager_modes
  def test_indexed_slices(self):
    # [FUNCTION DOC]
    # Scenario: jacobian of an embedding lookup of random data (sparse gradient). Asserts the
    # parallel-for and loop versions are close. Runs in both modes.
    with backprop.GradientTape(persistent=True) as g:
      inp = random_ops.random_uniform([3, 2])
      g.watch(inp)
      output = nn.embedding_lookup(inp, [0, 2])
    self.assertAllClose(
        g.jacobian(output, inp, experimental_use_pfor=True),
        g.jacobian(output, inp, experimental_use_pfor=False))

  def test_foldl_partial_function(self):
    # [FUNCTION DOC]
    # Scenario: jacobian of a graph function wrapping a partially applied fold-left sum over zeros.
    # Asserts the jacobian is ones for both implementations on a persistent tape, and for the
    # parallel-for version on a non-persistent tape (which uses a different gradient path).
    x = array_ops.zeros([3])
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(x)
      result = def_function.function(
          functools.partial(functional_ops.foldl_v2, lambda a, b: a + b))(
              x)
    self.assertAllClose([1., 1., 1.],
                        tape.jacobian(result, x, experimental_use_pfor=True))
    self.assertAllClose([1., 1., 1.],
                        tape.jacobian(result, x, experimental_use_pfor=False))

    # Non-persistent tapes take a different function gradient path, but also
    # work with pfor=True.
    x = array_ops.zeros([3])
    with backprop.GradientTape() as tape:
      tape.watch(x)
      result = def_function.function(
          functools.partial(functional_ops.foldl_v2, lambda a, b: a + b))(
              x)
    self.assertAllClose([1., 1., 1.],
                        tape.jacobian(result, x, experimental_use_pfor=True))

  def test_foldl_pure_function(self):
    # [FUNCTION DOC]
    # Scenario: jacobian of a plain (not graph-wrapped) partially applied fold-left inside a graph
    # function. Asserts parallel-for and loop results are close.

    @def_function.function
    def compute_jacobian(use_pfor):
      # [FUNCTION DOC]
      # HELPER: graph function taking a flag; records a fold-left sum on a persistent tape and returns
      # the jacobian with the chosen implementation.
      x = array_ops.zeros([3])
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(x)
        result = functools.partial(functional_ops.foldl_v2, lambda a, b: a + b)(
            x)
      return tape.jacobian(result, x, experimental_use_pfor=use_pfor)

    self.assertAllClose(compute_jacobian(use_pfor=True),
                        compute_jacobian(use_pfor=False))

  def test_cond_func_grad_jacobian(self):
    # [FUNCTION DOC]
    # Scenario: graph function with a conditional (cube if positive, else square) at 1.0. Asserts the
    # gradient is 3.0 and the jacobian of that gradient is 6.0 for both implementations.

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: graph function returning x cubed when x is positive, otherwise x squared.
      y = tf_cond.cond(x > 0., lambda: x**3., lambda: x**2.)
      return y

    with backprop.GradientTape(persistent=True) as tape:
      x = constant_op.constant(1.)
      tape.watch(x)
      y = f(x)
      grad = tape.gradient(y, x)
    self.assertAllClose(3., grad)
    jacobian = tape.jacobian(grad, x, experimental_use_pfor=False)
    self.assertAllClose(6., jacobian)
    jacobian_pfor = tape.jacobian(grad, x, experimental_use_pfor=True)
    self.assertAllClose(6., jacobian_pfor)

  def test_empty_tensor_consistent_jacobian(self):
    # [FUNCTION DOC]
    # Scenario: jacobian of an output built from a variable and empty (zero-row) inputs. Asserts the
    # parallel-for and loop implementations give close, consistent results.
    variable = variables.Variable(1.0)
    inputs = (
        constant_op.constant(np.random.uniform(size=(0, 4))),
        constant_op.constant(np.random.uniform(size=(0, 3))),
    )
    with backprop.GradientTape(persistent=True) as tape:
      outputs = variable * math_ops.cast(
          array_ops.concat(inputs, axis=-1), dtypes.float32
      )

    jacobians_pfor = tape.jacobian(
        outputs,
        variable,
        experimental_use_pfor=True,
    )
    jacobians_loop = tape.jacobian(
        outputs,
        variable,
        experimental_use_pfor=False,
    )
    self.assertAllClose(jacobians_pfor, jacobians_loop)


@test_util.run_all_in_graph_and_eager_modes
class BatchJacobianTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the batch jacobian method of the gradient tape, run in graph and eager modes (class
  # decorator) and parameterized. Covers implementations, graph functions, persistent-tape
  # requirement, input validation, parallel iterations, degenerate shapes, zero dtype and slicing.

  def _batch_jacobian(self, experimental_use_pfor):
    # [FUNCTION DOC]
    # HELPER: builds and returns a batch jacobian and the expected answer for the tests in this class.
    # KNOWN ISSUE: the persistent flag uses the function object for checking eager mode rather than
    # calling it, so that part is always true and the tape is persistent whenever the parallel-for
    # option is off.
    # Takes a flag choosing the parallel-for implementation. Computes z as x squared times y for 2 by
    # 2 constants, requests the batch jacobian with respect to x, and returns it together with the
    # expected stack of diagonal matrices.
    persistent = context.executing_eagerly and not experimental_use_pfor
    with backprop.GradientTape(persistent=persistent) as g:
      x = constant_op.constant([[1., 2.], [3., 4.]])
      y = constant_op.constant([[3., 4.], [5., 6.]])
      g.watch(x)
      z = x * x * y
    batch_jacobian = g.batch_jacobian(
        z, x, experimental_use_pfor=experimental_use_pfor)
    answer = array_ops_stack.stack(
        [array_ops.diag(2 * x[0] * y[0]),
         array_ops.diag(2 * x[1] * y[1])])
    return batch_jacobian, answer

  def testPfor(self):
    # [FUNCTION DOC]
    # Scenario: batch jacobian via parallel-for equals the expected answer.
    batch_jacobian, answer = self._batch_jacobian(experimental_use_pfor=True)
    self.assertAllEqual(answer, batch_jacobian)

  def testWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario: batch jacobian via the while-loop implementation equals the expected answer.
    batch_jacobian, answer = self._batch_jacobian(experimental_use_pfor=False)
    self.assertAllEqual(answer, batch_jacobian)

  def testPforDefun(self):
    # [FUNCTION DOC]
    # Scenario: parallel-for batch jacobian inside a graph function equals the expected answer.

    @def_function.function
    def _f():
      # [FUNCTION DOC]
      # HELPER: graph function returning the parallel-for batch jacobian and expected answer.
      return self._batch_jacobian(experimental_use_pfor=True)

    batch_jacobian, answer = _f()
    self.assertAllEqual(answer, batch_jacobian)

  def testWhileLoopDefun(self):
    # [FUNCTION DOC]
    # Scenario: while-loop batch jacobian inside a graph function equals the expected answer.

    @def_function.function
    def _f():
      # [FUNCTION DOC]
      # HELPER: graph function returning the while-loop batch jacobian and expected answer.
      return self._batch_jacobian(experimental_use_pfor=False)

    batch_jacobian, answer = _f()
    self.assertAllEqual(answer, batch_jacobian)

  def testPersistentTape(self):
    # [FUNCTION DOC]
    # Scenario (eager only): non-persistent tape with the while-loop batch jacobian. Asserts a runtime
    # error mentioning persistent.
    if not context.executing_eagerly():
      return
    with backprop.GradientTape() as g:
      x = constant_op.constant([[1.0, 2.0]])
      g.watch(x)
      y = x * x
    with self.assertRaisesRegex(RuntimeError, 'persistent'):
      g.batch_jacobian(y, x, experimental_use_pfor=False)

  def testBadShape(self):
    # [FUNCTION DOC]
    # Scenario: target whose first dimension (4) differs from the source's (2). Asserts a value error
    # saying the first dimension is needed to match.
    x = random_ops.random_uniform([2, 3])
    with backprop.GradientTape() as g:
      y = array_ops.concat([x, x], axis=0)
    with self.assertRaisesRegex(ValueError, 'Need first dimension'):
      g.batch_jacobian(y, x)

  def testBadInputRank(self):
    # [FUNCTION DOC]
    # Scenario: source of rank one. Asserts a value error saying rank at least 2 is required.
    x = random_ops.random_uniform([2])
    with backprop.GradientTape() as g:
      y = random_ops.random_uniform([2, 2])
    with self.assertRaisesRegex(ValueError, 'must have rank at least 2'):
      g.batch_jacobian(y, x)

  def testBadOutputRank(self):
    # [FUNCTION DOC]
    # Scenario: target of rank one. Asserts a value error saying rank at least 2 is required.
    x = random_ops.random_uniform([2, 2])
    with backprop.GradientTape() as g:
      y = random_ops.random_uniform([2])
    with self.assertRaisesRegex(ValueError, 'must have rank at least 2'):
      g.batch_jacobian(y, x)

  def test_parallel_iterations(self):
    # [FUNCTION DOC]
    # Scenario: batch jacobian of a matrix product with 2 versus 3 parallel iterations. Asserts close.
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant([[1., 2], [3, 4]])
      g.watch(x)
      w = constant_op.constant([[1., 2, 3, 4], [5, 6, 7, 8]])
      y = math_ops.matmul(x, w)
    self.assertAllClose(
        g.batch_jacobian(y, x, parallel_iterations=2),
        g.batch_jacobian(y, x, parallel_iterations=3))

  @parameterized.parameters((True, True), (True, False), (False, True),
                            (False, False))
  def test_degenerate_shape(self, use_function, use_pfor):
    # [FUNCTION DOC]
    # Scenario: parameterized over whether to wrap in a graph function and whether to use parallel-for.
    # Batch jacobian of a squared tensor of shape 1 by 0. Asserts the resulting shape is 1, 0, 0.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: records the square of x on a persistent tape and returns the batch jacobian of the
      # result with respect to x using the chosen implementation.
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x**2
      return tape.batch_jacobian(y, x, experimental_use_pfor=use_pfor)

    if use_function:
      f = def_function.function(f)
    self.assertAllEqual([1, 0, 0], array_ops.shape(f(array_ops.zeros([1, 0]))))

  @parameterized.parameters((True,), (False))
  def test_zeros_type_correct(self, use_pfor):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parameter list is written as a tuple of one tuple and a bare boolean (not a
    # one-element tuple), so the second case passes the boolean in an unusual form.
    # Scenario: for float32 and float64, a graph function ignores its input and returns a constant, so
    # the jacobian is unconnected. With default and zero unconnected-gradient options asserts the
    # jacobian dtype equals the input dtype and the value is a single zero.
    for dtype in [dtypes.float32, dtypes.float64]:
      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: graph function that discards its input and returns a constant [[1]] of the loop's dtype.
        del x
        return constant_op.constant([[1.]], dtype=dtype)  # pylint: disable=cell-var-from-loop

      with backprop.GradientTape(persistent=True) as tape:
        x = constant_op.constant([[2.]], dtype=dtype)
        tape.watch(x)
        y = f(x)
      jac = tape.batch_jacobian(y, x, experimental_use_pfor=use_pfor)
      self.assertEqual(dtype, jac.dtype)
      self.assertAllClose([[[0.]]], jac)

      with backprop.GradientTape(persistent=True) as tape:
        x = constant_op.constant([[2.]], dtype=dtype)
        tape.watch(x)
        y = f(x)
      jac = tape.batch_jacobian(y, x, unconnected_gradients='zero',
                                experimental_use_pfor=use_pfor)
      self.assertEqual(dtype, jac.dtype)
      self.assertAllClose([[[0.]]], jac)

  def test_strided_slice(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no assertion; the test only checks that computing the batch jacobian through a
    # repeat and strided slice does not raise.
    # Scenario: repeat the input twice along an axis and slice with a dynamic bound, then compute the
    # batch jacobian.
    x = array_ops.ones([2, 4, 2])
    length = constant_op.constant([2, 3, 4, 4], dtype=dtypes.int64)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      y = array_ops.repeat(x, [2], axis=1)
      y = y[:, :math_ops.reduce_max(length), :]
    tape.batch_jacobian(y, x)


class AggregateIndexedSlicesGradientsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the aggregate-indexed-slices-gradients helper in the backprop utilities module:
  # empty, single, multiple, null-containing and mixed dense/sparse inputs.

  def _assert_indexed_slices_equal(self, left, right):
    # [FUNCTION DOC]
    # HELPER: converts two indexed-slices (or tensor) objects to dense tensors, evaluates them and
    # asserts they are equal.
    self.assertAllEqual(
        self.evaluate(ops.convert_to_tensor(left)),
        self.evaluate(ops.convert_to_tensor(right)))

  def testNoGradients(self):
    # [FUNCTION DOC]
    # Scenario: empty list. Asserts the result is null.
    self.assertIsNone(backprop_util.AggregateIndexedSlicesGradients([]))

  def testOneGradient(self):
    # [FUNCTION DOC]
    # Scenario: a single indexed-slices gradient. Asserts the result equals it.
    t = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    result = backprop_util.AggregateIndexedSlicesGradients([t])
    self._assert_indexed_slices_equal(t, result)

  def testMultipleGradients(self):
    # [FUNCTION DOC]
    # Scenario: two complementary indexed-slices gradients. Asserts the result equals their sum.
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = backprop_util.AggregateIndexedSlicesGradients([t0, t1])
    self._assert_indexed_slices_equal(total, result)

  def testMultipleGradientsWithNones(self):
    # [FUNCTION DOC]
    # Scenario: two indexed-slices plus a null entry. Asserts the result equals the sum of the two.
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    t3 = None
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = backprop_util.AggregateIndexedSlicesGradients([t0, t1, t3])
    self._assert_indexed_slices_equal(total, result)

  def testMixedTensorAndIndexedSlices(self):
    # [FUNCTION DOC]
    # Scenario: one indexed-slices and one dense tensor. Asserts the result equals the dense sum.
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = constant_op.constant([[0., 0.], [5, 6], [7., 8.]])
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = backprop_util.AggregateIndexedSlicesGradients([t0, t1])
    self._assert_indexed_slices_equal(total, result)


if __name__ == '__main__':
  test.main()
