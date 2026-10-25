# [MODULE DOC]
# Functional tests for the v1 proximal gradient descent optimizer. Covers dense updates (ordinary
# and resource variables) without regularization and with L1/L2 regularization against hard-coded
# expected numbers, a sparse resource-variable minimization, and equivalence with plain gradient
# descent (dense and sparse) when both regularization strengths are zero. Tests run in explicit
# graph mode sessions. Runs through the standard test main entry point.
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
"""Functional tests for Proximal Gradient Descent operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import proximal_gradient_descent


class ProximalGradientDescentOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the proximal gradient descent optimizer: numeric results of dense updates
  # under different regularization settings, sparse resource-variable training, and agreement with
  # plain gradient descent. Shared fixtures: a helper that applies an optimizer to two small
  # variables and returns their values, and the per-test default graph and cached session.

  def doTestProximalGradientDescentwithoutRegularization(
      self, use_resource=False):
    # [FUNCTION DOC]
    # HELPER: shared body for the plain and resource "without regularization" tests.
    # Input use_resource selects resource variables (true) or ordinary variables (false, default).
    # Scenario: two zero-initialized variables with fixed gradients, optimizer with learning rate
    # 3.0 and zero L1/L2. Checks the initial zeros, runs three update steps, and asserts the
    # variables equal minus three times learning rate times gradient (-0.9,-1.8 and -0.09,-0.18).
    with ops.Graph().as_default(), self.cached_session():
      if use_resource:
        var0 = resource_variable_ops.ResourceVariable([0.0, 0.0])
        var1 = resource_variable_ops.ResourceVariable([0.0, 0.0])
      else:
        var0 = variables.Variable([0.0, 0.0])
        var1 = variables.Variable([0.0, 0.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([0.0, 0.0], v0_val)
      self.assertAllClose([0.0, 0.0], v1_val)

      # Run 3 steps Proximal Gradient Descent.
      for _ in range(3):
        update.run()

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-0.9, -1.8]), v0_val)
      self.assertAllClose(np.array([-0.09, -0.18]), v1_val)

  def testProximalGradientDescentwithoutRegularization(self):
    # [FUNCTION DOC]
    # Test: runs the shared without-regularization scenario with ordinary variables and expects the
    # plain gradient descent result after three steps.
    self.doTestProximalGradientDescentwithoutRegularization(use_resource=False)

  def testResourceProximalGradientDescentwithoutRegularization(self):
    # [FUNCTION DOC]
    # Test: runs the shared without-regularization scenario with resource variables and expects the
    # same numeric result as the ordinary-variable test.
    self.doTestProximalGradientDescentwithoutRegularization(use_resource=True)

  def testProximalGradientDescentwithoutRegularization2(self):
    # [FUNCTION DOC]
    # Test: variables start at (1,2) and (4,3) with fixed small gradients; learning rate 3.0, no
    # regularization. Checks the initial values, runs three steps, and asserts the values
    # (0.1,0.2) and (3.91,2.82), i.e. plain gradient descent.
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 3 steps Proximal Gradient Descent
      for _ in range(3):
        update.run()

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([0.1, 0.2]), v0_val)
      self.assertAllClose(np.array([3.91, 2.82]), v1_val)

  def testMinimizeSparseResourceVariable(self):
    # [FUNCTION DOC]
    # Test: for single and double precision floats, builds a resource variable holding one row,
    # reads it through an embedding lookup, multiplies by a constant column, squares the result as
    # the loss and minimizes it with learning rate 1.0. Asserts the initial value, runs one step,
    # and asserts the variable becomes roughly (-111, -138) within a small absolute tolerance,
    # exercising the sparse resource update path.
    for dtype in [dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        loss = pred * pred
        sgd_op = proximal_gradient_descent.ProximalGradientDescentOptimizer(
            1.0).minimize(loss)
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([[-111, -138]],
                                           self.evaluate(var0),
                                           atol=0.01)

  def testProximalGradientDescentWithL1_L2(self):
    # [FUNCTION DOC]
    # Test: variables (1,2) and (4,3), fixed gradients, learning rate 3.0, L1 strength 0.001 and
    # L2 strength 2.0. Verifies initial values, runs ten steps, and asserts the variables shrink to
    # values very close to zero (-0.0495,-0.0995 and -0.0045,-0.0095).
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.001, l2_regularization_strength=2.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 10 steps Proximal Gradient Descent
      for _ in range(10):
        update.run()

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-0.0495, -0.0995]), v0_val)
      self.assertAllClose(np.array([-0.0045, -0.0095]), v1_val)

  def applyOptimizer(self, opt, steps=5, is_sparse=False):
    # [FUNCTION DOC]
    # HELPER: runs a given optimizer on two small variables for the equivalence tests.
    # Inputs: opt (optimizer instance), steps (number of update runs, default 5), is_sparse (when
    # true, variables are two-row columns and gradients are indexed slices touching one row each;
    # otherwise plain vectors with dense gradients). Applies gradients, initializes variables,
    # asserts the initial values, runs the update the given number of times and returns the pair
    # of final values of the two variables. Must be called inside a graph and session.
    if is_sparse:
      var0 = variables.Variable([[1.0], [2.0]])
      var1 = variables.Variable([[3.0], [4.0]])
      grads0 = indexed_slices.IndexedSlices(
          constant_op.constant(
              [0.1], shape=[1, 1]),
          constant_op.constant([0]),
          constant_op.constant([2, 1]))
      grads1 = indexed_slices.IndexedSlices(
          constant_op.constant(
              [0.02], shape=[1, 1]),
          constant_op.constant([1]),
          constant_op.constant([2, 1]))
    else:
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([3.0, 4.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.evaluate(variables.global_variables_initializer())

    v0_val, v1_val = self.evaluate([var0, var1])
    if is_sparse:
      self.assertAllClose([[1.0], [2.0]], v0_val)
      self.assertAllClose([[3.0], [4.0]], v1_val)
    else:
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

    # Run ProximalAdagrad for a few steps
    for _ in range(steps):
      update.run()

    v0_val, v1_val = self.evaluate([var0, var1])
    return v0_val, v1_val

  def testEquivSparseGradientDescentwithoutRegularization(self):
    # [FUNCTION DOC]
    # Test: applies the proximal optimizer (learning rate 3.0, zero L1/L2) and plain gradient
    # descent with sparse gradients, both in one graph and session. Asserts both variables end with
    # the same values in both runs.
    with ops.Graph().as_default(), self.cached_session():
      val0, val1 = self.applyOptimizer(
          proximal_gradient_descent.ProximalGradientDescentOptimizer(
              3.0,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0),
          is_sparse=True)

      val2, val3 = self.applyOptimizer(
          gradient_descent.GradientDescentOptimizer(3.0), is_sparse=True)

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)

  def testEquivGradientDescentwithoutRegularization(self):
    # [FUNCTION DOC]
    # Test: same as the sparse equivalence test but with dense gradients. Asserts the proximal
    # optimizer without regularization matches plain gradient descent on both variables.
    with ops.Graph().as_default(), self.cached_session():
      val0, val1 = self.applyOptimizer(
          proximal_gradient_descent.ProximalGradientDescentOptimizer(
              3.0,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0))

      val2, val3 = self.applyOptimizer(
          gradient_descent.GradientDescentOptimizer(3.0))

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)


if __name__ == "__main__":
  test.main()
