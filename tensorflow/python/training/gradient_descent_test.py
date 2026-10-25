# [MODULE DOC]
# Functional tests for the v1 gradient descent optimizer of the training package. Covers
# ordinary and resource variables, callable and tensor learning rates, minimize on dense and
# sparse (embedding lookup) losses, gradients with respect to reference variables, global step
# increment and sparse indexed-slices gradients, for half, 32-bit and 64-bit floats. Each test
# runs in a fresh graph. No import-time registrations.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional test for GradientDescent."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class GradientDescentOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the gradient descent optimizer. Each test loops over three float types,
  # builds variables and gradients in a new graph with a cached session, applies one
  # optimizer step and checks the variables equal the initial values minus learning rate times
  # gradient. No shared fixtures.

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: two ordinary variables with constant gradients, learning rate 3.0, one apply
    # step. Asserts initial values, then values reduced by rate times gradient, and that the
    # optimizer reports no slot variables.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        optimizer = gradient_descent.GradientDescentOptimizer(3.0)
        sgd_op = optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))
        self.assertEqual(0, len(optimizer.variables()))

  def testBasicResourceVariable(self):
    # [FUNCTION DOC]
    # Scenario: same as the basic test but with resource variables, initialized explicitly via
    # resource initialization. Asserts initial values and values after one step with rate 3.0.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  def testBasicCallableParams(self):
    # [FUNCTION DOC]
    # Scenario: the learning rate is given as a no-argument callable returning 3.0, applied to
    # resource variables for one step. Asserts initial values and the values reduced by rate
    # times gradient.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lr = lambda: 3.0
        sgd_op = gradient_descent.GradientDescentOptimizer(lr).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  def testMinimizeResourceVariable(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test refers to the numpy module under its usual short name but this file
    # never imports it, so the expected-value computation would fail with a name error when
    # reached.
    # Scenario: a squared linear prediction loss over a resource matrix variable and a bias,
    # minimized with rate 1.0 for one step. Asserts initial values and that the updated values
    # equal the analytic gradient step (twice the prediction times the inputs).
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(var0, x) + var1
        loss = pred * pred
        sgd_op = gradient_descent.GradientDescentOptimizer(1.0).minimize(loss)
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

  def testMinimizeSparseResourceVariable(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: like the dense minimize test, it uses the numpy module under its short name
    # without importing it, which would raise a name error at the expected-value computation.
    # Scenario: same loss as the dense minimize test but the matrix variable is read through an
    # embedding lookup, producing a sparse gradient, and variables are initialized globally.
    # Asserts initial values and values after one step of rate 1.0 match the analytic step.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        pred += var1
        loss = pred * pred
        sgd_op = gradient_descent.GradientDescentOptimizer(1.0).minimize(loss)
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

  def testTensorLearningRate(self):
    # [FUNCTION DOC]
    # Scenario: the learning rate is a constant tensor (3.0) rather than a plain number, with
    # ordinary variables and one step. Asserts initial values and values reduced by rate times
    # gradient.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lrate = constant_op.constant(3.0)
        sgd_op = gradient_descent.GradientDescentOptimizer(
            lrate).apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))

  def testGradWrtRef(self):
    # [FUNCTION DOC]
    # Scenario: compute gradients of the sum of two single-element variables with respect to
    # both variables. Asserts each computed gradient equals one.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        opt = gradient_descent.GradientDescentOptimizer(3.0)
        values = [1.0, 3.0]
        vars_ = [variables.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
        self.evaluate(variables.global_variables_initializer())
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.0], self.evaluate(grad))

  def testWithGlobalStep(self):
    # [FUNCTION DOC]
    # Scenario: apply gradients with rate 3.0 while passing a non-trainable global step variable
    # starting at zero. Asserts the variables were updated by rate times gradient and the global
    # step became one.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        global_step = variables.Variable(0, trainable=False)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params and global_step
        self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01],
                                           self.evaluate(var1))
        self.assertAllCloseAccordingToType(1, self.evaluate(global_step))

  def testSparseBasic(self):
    # [FUNCTION DOC]
    # Scenario: sparse indexed-slices gradients each touching a different row of two-row
    # variables, rate 3.0, one step. Asserts that only the touched rows were reduced by rate
    # times gradient and untouched rows stayed unchanged.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant(
                [0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]),
            constant_op.constant([2, 1]))
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant(
                [0.01], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0], [2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0]], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([[1.0 - 3.0 * 0.1], [2.0]],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0 - 3.0 * 0.01]],
                                           self.evaluate(var1))


if __name__ == "__main__":
  test.main()
